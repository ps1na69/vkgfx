// src/mesh_gltf.cpp
// Implements Mesh::loadGLTF() — loads .gltf / .glb into engine-ready Mesh objects.
//
// Design notes:
//  • Each glTF primitive → one Mesh + one PBRMaterial (exactly like existing OBJ loader).
//  • Node hierarchy world-transforms are accumulated and baked into each Mesh TRS.
//  • glTF metallicRoughness (G=rough, B=metal) is remapped → engine RMA (R=rough, G=metal, B=ao).
//  • STB_IMAGE_IMPLEMENTATION must not be defined here (it lives in texture.cpp).
//  • No skinning in this pass — Vertex struct has no joints/weights attributes.

#include <vkgfx/mesh.h>
#include <vkgfx/material.h>
#include <vkgfx/texture.h>
#include <vkgfx/context.h>

// fastgltf — only included in this TU
#include <fastgltf/core.hpp>
#include <fastgltf/types.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>

#include <stb_image.h>    // declaration only — STB_IMAGE_IMPLEMENTATION in texture.cpp

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>

#include <cassert>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace vkgfx {

    // ─────────────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────────────

    namespace {

        // Tangent generation — same algorithm as mesh.cpp (accumulated per-face tangents).
        // Runs when the glTF primitive has no TANGENT accessor.
        static void generateTangents(std::vector<Vertex>& verts,
            const std::vector<uint32_t>& idx) {
            // Accumulate tangents across shared vertices
            for (size_t i = 0; i + 2 < idx.size(); i += 3) {
                Vertex& v0 = verts[idx[i]];
                Vertex& v1 = verts[idx[i + 1]];
                Vertex& v2 = verts[idx[i + 2]];

                glm::vec3 edge1 = v1.position - v0.position;
                glm::vec3 edge2 = v2.position - v0.position;
                glm::vec2 dUV1 = v1.uv - v0.uv;
                glm::vec2 dUV2 = v2.uv - v0.uv;

                float det = dUV1.x * dUV2.y - dUV2.x * dUV1.y;
                if (std::abs(det) < 1e-6f) continue;
                float inv = 1.0f / det;

                glm::vec3 tangent{
                    inv * (dUV2.y * edge1.x - dUV1.y * edge2.x),
                    inv * (dUV2.y * edge1.y - dUV1.y * edge2.y),
                    inv * (dUV2.y * edge1.z - dUV1.y * edge2.z)
                };
                for (int j = 0; j < 3; ++j)
                    verts[idx[i + j]].tangent += glm::vec4(tangent, 0.f);
            }
            // Gram-Schmidt orthogonalise and normalise
            for (auto& v : verts) {
                glm::vec3 t = glm::vec3(v.tangent);
                if (glm::length(t) > 1e-6f) {
                    t = glm::normalize(t - v.normal * glm::dot(v.normal, t));
                    v.tangent = glm::vec4(t, 1.0f);
                }
                else {
                    // Fallback: pick an arbitrary perpendicular
                    glm::vec3 up = std::abs(v.normal.y) < 0.99f
                        ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
                    v.tangent = glm::vec4(glm::normalize(glm::cross(v.normal, up)), 1.0f);
                }
            }
        }

        // ── Accessor reader ───────────────────────────────────────────────────────────
        // Reads a typed accessor into an output vector using fastgltf::iterateAccessor.

        template<typename T>
        static std::vector<T> readAccessor(const fastgltf::Asset& asset, size_t accessorIdx) {
            const fastgltf::Accessor& acc = asset.accessors[accessorIdx];
            std::vector<T> out(acc.count);
            fastgltf::copyFromAccessor<T>(asset, acc, out.data());
            return out;
        }

        // Specialisation for uint16 → uint32 widening (index buffers)
        static std::vector<uint32_t> readIndexAccessor(const fastgltf::Asset& asset,
            size_t accessorIdx) {
            const fastgltf::Accessor& acc = asset.accessors[accessorIdx];
            std::vector<uint32_t> out(acc.count);
            if (acc.componentType == fastgltf::ComponentType::UnsignedShort) {
                std::vector<uint16_t> u16(acc.count);
                fastgltf::copyFromAccessor<uint16_t>(asset, acc, u16.data());
                for (size_t i = 0; i < u16.size(); ++i) out[i] = u16[i];
            }
            else if (acc.componentType == fastgltf::ComponentType::UnsignedByte) {
                std::vector<uint8_t> u8(acc.count);
                fastgltf::copyFromAccessor<uint8_t>(asset, acc, u8.data());
                for (size_t i = 0; i < u8.size(); ++i) out[i] = u8[i];
            }
            else {
                fastgltf::copyFromAccessor<uint32_t>(asset, acc, out.data());
            }
            return out;
        }

        // ── Image data extraction ─────────────────────────────────────────────────────
        // Returns decoded RGBA8 pixels + dimensions, or empty on failure.

        struct RawImage {
            std::vector<uint8_t> pixels; // RGBA8
            uint32_t             width = 0;
            uint32_t             height = 0;
            bool                 valid = false;
        };

        static RawImage extractImage(const fastgltf::Asset& asset,
            const fastgltf::Image& img,
            const std::filesystem::path& baseDir) {
            RawImage result;
            int w, h, ch;
            stbi_uc* px = nullptr;

            std::visit(fastgltf::visitor{
                // External file URI
                [&](const fastgltf::sources::URI& uri) {
                    auto fullPath = (baseDir / uri.uri.fspath()).string();
                    px = stbi_load(fullPath.c_str(), &w, &h, &ch, 4);
                },
                // Embedded byte array (e.g. base64 in .gltf or GLB binary chunk)
                [&](const fastgltf::sources::Array& arr) {
                    px = stbi_load_from_memory(
                        reinterpret_cast<const stbi_uc*>(arr.bytes.data()),
                        static_cast<int>(arr.bytes.size()), &w, &h, &ch, 4);
                },
                // Buffer view reference (most common for GLB)
                [&](const fastgltf::sources::BufferView& bvRef) {
                    const auto& bv = asset.bufferViews[bvRef.bufferViewIndex];
                    const auto& buf = asset.buffers[bv.bufferIndex];
                    std::visit(fastgltf::visitor{
                        [&](const fastgltf::sources::Array& ba) {
                            px = stbi_load_from_memory(
                                reinterpret_cast<const stbi_uc*>(ba.bytes.data() + bv.byteOffset),
                                static_cast<int>(bv.byteLength), &w, &h, &ch, 4);
                        },
                        [](auto&&) {}
                    }, buf.data);
                },
                [](auto&&) {}
                }, img.data);

            if (px) {
                result.width = static_cast<uint32_t>(w);
                result.height = static_cast<uint32_t>(h);
                result.pixels.assign(px, px + w * h * 4);
                result.valid = true;
                stbi_image_free(px);
            }
            return result;
        }

        // ── RMA channel remapping ─────────────────────────────────────────────────────
        // glTF metallicRoughness texture layout:  G=roughness, B=metallic, R=unused
        // Engine RMA texture layout:              R=roughness, G=metallic,  B=ao
        //
        // Combines the metallicRoughness image (mandatory) with an optional separate
        // occlusion image (if occlusion uses the same image as metallicRoughness,
        // the R channel of that image becomes AO).
        //
        // Returns a new RGBA8 image in engine RMA layout (A channel unused, set to 255).

        static std::vector<uint8_t> remapToRMA(
            const RawImage& mrImg,         // metallicRoughness texture (G=rough, B=metal)
            const RawImage& occImg,        // occlusion texture (R=ao), may be invalid
            bool            sameTexture)   // true if mr and occ are the same image
        {
            assert(mrImg.valid);
            const uint32_t pixelCount = mrImg.width * mrImg.height;
            std::vector<uint8_t> out(pixelCount * 4);

            for (uint32_t i = 0; i < pixelCount; ++i) {
                const uint8_t* src = mrImg.pixels.data() + i * 4;
                uint8_t* dst = out.data() + i * 4;

                dst[0] = src[1]; // R_out = G_in = roughness
                dst[1] = src[2]; // G_out = B_in = metallic
                // AO: prefer dedicated occlusion map R channel; fall back to mr.R or full white
                if (sameTexture) {
                    dst[2] = src[0]; // B_out = R_in = glTF occlusion (same texture)
                }
                else if (occImg.valid && occImg.width == mrImg.width && occImg.height == mrImg.height) {
                    dst[2] = occImg.pixels[i * 4 + 0]; // R channel of separate occlusion image
                }
                else {
                    dst[2] = 255; // no occlusion data → full AO
                }
                dst[3] = 255;
            }
            return out;
        }

        // ── Texture slot loading ──────────────────────────────────────────────────────
        // Returns nullptr if the material has no texture for that slot.

        static std::shared_ptr<Texture> loadTexture(
            const fastgltf::Asset& asset,
            const std::optional<fastgltf::TextureInfo>& texInfo,
            TextureCache& cache,
            const std::filesystem::path& baseDir,
            TextureDesc                               desc)
        {
            if (!texInfo.has_value()) return nullptr;

            size_t texIdx = texInfo->textureIndex;
            if (texIdx >= asset.textures.size()) return nullptr;

            const fastgltf::Texture& tex = asset.textures[texIdx];
            if (!tex.imageIndex.has_value()) return nullptr;

            const fastgltf::Image& img = asset.images[tex.imageIndex.value()];

            // External file? Use TextureCache::load() for deduplication.
            if (auto* uri = std::get_if<fastgltf::sources::URI>(&img.data)) {
                auto fullPath = (baseDir / uri->uri.fspath()).string();
                return cache.load(fullPath, desc);
            }

            // Embedded — decode and upload without path caching
            RawImage raw = extractImage(asset, img, baseDir);
            if (!raw.valid) {
                std::cerr << "[vkgfx] glTF: failed to decode embedded image '"
                    << img.name << "'\n";
                return nullptr;
            }
            return cache.loadFromMemory(raw.pixels.data(), raw.width, raw.height, desc);
        }

        // Loads + remaps the combined RMA texture.
        // Returns nullptr if neither metallicRoughness nor occlusion textures are present.
        static std::shared_ptr<Texture> loadRMATexture(
            const fastgltf::Asset& asset,
            const fastgltf::Material& mat,
            TextureCache& cache,
            const std::filesystem::path& baseDir)
        {
            const bool hasMR = mat.pbrData.metallicRoughnessTexture.has_value();
            const bool hasOcc = mat.occlusionTexture.has_value();
            if (!hasMR && !hasOcc) return nullptr;

            // Determine whether mr and occ share the same glTF image index
            size_t mrImgIdx = SIZE_MAX;
            size_t occImgIdx = SIZE_MAX;

            auto imgIdxOf = [&](size_t texIdx) -> size_t {
                if (texIdx >= asset.textures.size()) return SIZE_MAX;
                if (!asset.textures[texIdx].imageIndex.has_value()) return SIZE_MAX;
                return asset.textures[texIdx].imageIndex.value();
                };

            if (hasMR)  mrImgIdx = imgIdxOf(mat.pbrData.metallicRoughnessTexture->textureIndex);
            if (hasOcc) occImgIdx = imgIdxOf(mat.occlusionTexture->textureIndex);

            const bool sameImage = (hasMR && hasOcc && mrImgIdx == occImgIdx && mrImgIdx != SIZE_MAX);

            // Load the metallicRoughness image (always linear)
            RawImage mrRaw;
            if (hasMR && mrImgIdx != SIZE_MAX) {
                mrRaw = extractImage(asset, asset.images[mrImgIdx], baseDir);
            }
            if (!mrRaw.valid) {
                // Only occlusion present — create a white MR image so remapping still works
                // (roughness=1, metallic=0, AO from occlusion)
                if (hasOcc && occImgIdx != SIZE_MAX) {
                    RawImage occRaw = extractImage(asset, asset.images[occImgIdx], baseDir);
                    if (!occRaw.valid) return nullptr;
                    // Synthesise MR image: roughness=1 (G=255), metallic=0 (B=0)
                    mrRaw.width = occRaw.width;
                    mrRaw.height = occRaw.height;
                    mrRaw.pixels.resize(occRaw.width * occRaw.height * 4);
                    for (size_t i = 0; i < occRaw.width * occRaw.height; ++i) {
                        mrRaw.pixels[i * 4 + 0] = 255; // R (repurposed for AO in same-texture case)
                        mrRaw.pixels[i * 4 + 1] = 255; // G = roughness = 1.0
                        mrRaw.pixels[i * 4 + 2] = 0;   // B = metallic  = 0.0
                        mrRaw.pixels[i * 4 + 3] = 255;
                    }
                    mrRaw.valid = true;
                    // remap with occlusion = same texture (R channel = occ)
                    auto remapped = remapToRMA(mrRaw, occRaw, /*sameTexture=*/false);
                    TextureDesc desc{ VK_FORMAT_R8G8B8A8_UNORM, true, false, false };
                    return cache.loadFromMemory(remapped.data(),
                        occRaw.width, occRaw.height, desc);
                }
                return nullptr;
            }

            // Load separate occlusion image if needed
            RawImage occRaw;
            if (hasOcc && !sameImage && occImgIdx != SIZE_MAX) {
                occRaw = extractImage(asset, asset.images[occImgIdx], baseDir);
            }

            // Build remapped RMA image
            auto remapped = remapToRMA(mrRaw, occRaw, sameImage);
            TextureDesc desc{ VK_FORMAT_R8G8B8A8_UNORM, /*genMips=*/true, false, false };
            return cache.loadFromMemory(remapped.data(), mrRaw.width, mrRaw.height, desc);
        }

        // ── Node world-transform accumulation ─────────────────────────────────────────

        static glm::mat4 nodeLocalMatrix(const fastgltf::Node& node) {
            // fastgltf::Options::DecomposeNodeMatrices guarantees TRS variant
            const auto& trs = std::get<fastgltf::TRS>(node.transform);
            glm::vec3 t{ trs.translation[0], trs.translation[1], trs.translation[2] };
            // glTF quat: [x,y,z,w] — GLM quat ctor: (w,x,y,z)
            glm::quat r{ trs.rotation[3], trs.rotation[0], trs.rotation[1], trs.rotation[2] };
            glm::vec3 s{ trs.scale[0], trs.scale[1], trs.scale[2] };
            return glm::translate(glm::mat4(1.f), t)
                * glm::toMat4(r)
                * glm::scale(glm::mat4(1.f), s);
        }

        // Recursively collect (nodeIndex, worldTransform) pairs for nodes that carry a mesh.
        static void collectMeshNodes(
            const fastgltf::Asset& asset,
            size_t                 nodeIdx,
            const glm::mat4& parentWorld,
            std::vector<std::pair<size_t, glm::mat4>>& out)
        {
            const fastgltf::Node& node = asset.nodes[nodeIdx];
            glm::mat4 world = parentWorld * nodeLocalMatrix(node);

            if (node.meshIndex.has_value())
                out.emplace_back(nodeIdx, world);

            for (size_t child : node.children)
                collectMeshNodes(asset, child, world, out);
        }

        // Decompose world matrix into TRS components that Mesh::setPosition/Rotation/Scale accept.
        static void decomposeMatrix(const glm::mat4& m,
            glm::vec3& outPos,
            glm::quat& outRot,
            glm::vec3& outScale) {
            glm::vec3 skew; glm::vec4 persp;
            glm::decompose(m, outScale, outRot, outPos, skew, persp);
            outRot = glm::normalize(outRot);
        }

    } // anonymous namespace

    // ─────────────────────────────────────────────────────────────────────────────
    // Mesh::loadGLTF
    // ─────────────────────────────────────────────────────────────────────────────

    std::vector<std::shared_ptr<Mesh>> Mesh::loadGLTF(const std::string& path,
        Context& ctx,
        TextureCache& cache) {
        namespace fs = std::filesystem;

        if (!fs::exists(path)) {
            std::cerr << "[vkgfx] glTF file not found: " << path << "\n";
            return {};
        }

        const fs::path filePath(path);
        const fs::path baseDir = filePath.parent_path();

        // ── Parse with fastgltf ───────────────────────────────────────────────────
        fastgltf::Parser parser;

        constexpr auto options =
            fastgltf::Options::DecomposeNodeMatrices |
            fastgltf::Options::LoadExternalBuffers |
            fastgltf::Options::LoadExternalImages;

        auto gltfFile = fastgltf::MappedGltfFile::FromPath(filePath);
        if (!gltfFile) {
            std::cerr << "[vkgfx] Failed to open glTF: " << path << " — "
                << fastgltf::getErrorMessage(gltfFile.error()) << "\n";
            return {};
        }

        auto expected = parser.loadGltf(gltfFile.get(), baseDir, options);
        if (expected.error() != fastgltf::Error::None) {
            std::cerr << "[vkgfx] fastgltf parse error in '" << path << "': "
                << fastgltf::getErrorMessage(expected.error()) << "\n";
            return {};
        }

        const fastgltf::Asset& asset = expected.get();

        if (asset.scenes.empty()) {
            std::cerr << "[vkgfx] glTF has no scenes: " << path << "\n";
            return {};
        }

        // ── Pre-load all glTF materials → PBRMaterial ─────────────────────────────
        // Index-aligned with asset.materials so primitives can look them up by index.
        std::vector<std::shared_ptr<PBRMaterial>> materials;
        materials.reserve(asset.materials.size());

        for (const fastgltf::Material& gltfMat : asset.materials) {
            auto mat = std::make_shared<PBRMaterial>();

            // ── Base colour ───────────────────────────────────────────────────────
            const auto& bc = gltfMat.pbrData.baseColorFactor;
            mat->setAlbedo(bc[0], bc[1], bc[2], bc[3]);

            // ── PBR scalars ───────────────────────────────────────────────────────
            mat->setRoughness(gltfMat.pbrData.roughnessFactor);
            mat->setMetallic(gltfMat.pbrData.metallicFactor);

            // ── Emissive ──────────────────────────────────────────────────────────
            const auto& em = gltfMat.emissiveFactor;
            // Emissive strength is 1.0 when the extension isn't present
            float emStrength = gltfMat.emissiveStrength;
            mat->setEmissive(em[0], em[1], em[2], emStrength);

            // ── Albedo texture (sRGB) ─────────────────────────────────────────────
            TextureDesc srgbDesc{ VK_FORMAT_R8G8B8A8_SRGB, true, false, false };
            auto albedoTex = loadTexture(asset,
                gltfMat.pbrData.baseColorTexture, cache, baseDir, srgbDesc);
            if (albedoTex) mat->setAlbedoTexture(albedoTex);

            // ── Normal texture (linear) ───────────────────────────────────────────
            TextureDesc linDesc{ VK_FORMAT_R8G8B8A8_UNORM, true, false, false };

            // Convert optional<NormalTextureInfo> → optional<TextureInfo>
            std::optional<fastgltf::TextureInfo> normalTexInfo;
            if (gltfMat.normalTexture.has_value()) {
                fastgltf::TextureInfo ti;
                ti.textureIndex = gltfMat.normalTexture->textureIndex;
                ti.texCoordIndex = gltfMat.normalTexture->texCoordIndex;
                normalTexInfo.emplace(std::move(ti));
            }
            auto normalTex = loadTexture(asset, normalTexInfo, cache, baseDir, linDesc);
            if (normalTex) mat->setNormalTexture(normalTex);

            // ── RMA texture (linear, channel-remapped from glTF) ──────────────────
            auto rmaTex = loadRMATexture(asset, gltfMat, cache, baseDir);
            if (rmaTex) mat->setRMATexture(rmaTex);

            materials.push_back(std::move(mat));
        }

        // ── Collect mesh-carrying nodes from the default scene ────────────────────
        size_t sceneIdx = asset.defaultScene.value_or(0);
        const fastgltf::Scene& scene = asset.scenes[sceneIdx];

        std::vector<std::pair<size_t /*nodeIdx*/, glm::mat4 /*worldTransform*/>> meshNodes;
        for (size_t rootIdx : scene.nodeIndices)
            collectMeshNodes(asset, rootIdx, glm::mat4(1.f), meshNodes);

        // ── Build one engine Mesh per primitive ───────────────────────────────────
        std::vector<std::shared_ptr<Mesh>> result;

        for (auto& [nodeIdx, worldMat] : meshNodes) {
            const fastgltf::Node& node = asset.nodes[nodeIdx];
            const fastgltf::Mesh& gltfMesh = asset.meshes[node.meshIndex.value()];

            for (const fastgltf::Primitive& prim : gltfMesh.primitives) {
                if (prim.type != fastgltf::PrimitiveType::Triangles) continue;

                // ── Positions (required) ──────────────────────────────────────────
                auto posIt = prim.findAttribute("POSITION");
                if (posIt == prim.attributes.end()) {
                    std::cerr << "[vkgfx] glTF primitive missing POSITION, skipped\n";
                    continue;
                }

                auto positions = readAccessor<glm::vec3>(asset, posIt->accessorIndex);
                std::vector<Vertex> verts(positions.size());
                AABB aabb;
                for (size_t i = 0; i < positions.size(); ++i) {
                    verts[i].position = positions[i];
                    aabb.expand(positions[i]);
                }

                // ── Normals ───────────────────────────────────────────────────────
                if (auto it = prim.findAttribute("NORMAL"); it != prim.attributes.end()) {
                    auto normals = readAccessor<glm::vec3>(asset, it->accessorIndex);
                    for (size_t i = 0; i < normals.size(); ++i)
                        verts[i].normal = normals[i];
                }
                else {
                    // Flat normals: will be overwritten by generateTangents if UVs present
                    for (auto& v : verts) v.normal = { 0.f, 1.f, 0.f };
                }

                // ── UV set 0 — glTF uses top-left origin (no V-flip needed) ───────
                if (auto it = prim.findAttribute("TEXCOORD_0"); it != prim.attributes.end()) {
                    auto uvs = readAccessor<glm::vec2>(asset, it->accessorIndex);
                    for (size_t i = 0; i < uvs.size(); ++i)
                        verts[i].uv = uvs[i];
                }

                // ── Tangents (w = bitangent sign) ─────────────────────────────────
                bool hasTangents = false;
                if (auto it = prim.findAttribute("TANGENT"); it != prim.attributes.end()) {
                    auto tangents = readAccessor<glm::vec4>(asset, it->accessorIndex);
                    for (size_t i = 0; i < tangents.size(); ++i)
                        verts[i].tangent = tangents[i];
                    hasTangents = true;
                }

                // ── Indices ───────────────────────────────────────────────────────
                std::vector<uint32_t> indices;
                if (prim.indicesAccessor.has_value()) {
                    indices = readIndexAccessor(asset, prim.indicesAccessor.value());
                }
                else {
                    // Non-indexed: generate sequential indices
                    indices.resize(verts.size());
                    std::iota(indices.begin(), indices.end(), 0u);
                }

                // ── Generate tangents if absent ───────────────────────────────────
                if (!hasTangents)
                    generateTangents(verts, indices);

                // ── Create engine Mesh ────────────────────────────────────────────
                auto mesh = std::shared_ptr<Mesh>(new Mesh());
                mesh->m_aabb = aabb;
                mesh->upload(ctx, verts, indices);

                // Bake world transform into TRS
                glm::vec3 pos, scale; glm::quat rot;
                decomposeMatrix(worldMat, pos, rot, scale);
                mesh->setPosition(pos).setRotation(rot).setScale(scale);

                // ── Assign material ───────────────────────────────────────────────
                if (prim.materialIndex.has_value()) {
                    size_t mi = prim.materialIndex.value();
                    if (mi < materials.size())
                        mesh->setMaterial(materials[mi]);
                }

                result.push_back(std::move(mesh));
            }
        }

        std::cout << "[vkgfx] glTF loaded '" << filePath.filename().string()
            << "': " << result.size() << " primitive(s), "
            << materials.size() << " material(s)\n";
        return result;
    }

} // namespace vkgfx