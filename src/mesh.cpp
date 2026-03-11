// mesh.cpp — glTF 2.0 mesh loading via tinygltf.
//
// tinygltf handles JSON parsing, binary GLB reading, and image decoding
// (PNG/JPG via its bundled stb_image).  We extract vertex attributes,
// generate tangents if absent, upload to GPU, and build PBRMaterials.

#include "vkgfx/mesh.h"

// tinygltf — header-only glTF 2.0 loader.
// TINYGLTF_NO_STB_IMAGE_WRITE: we never write images.
// STB_IMAGE_IMPLEMENTATION is owned by stb_impl.cpp — must not be redefined here.
// TINYGLTF_NO_STB_IMAGE tells tinygltf not to define STB_IMAGE_IMPLEMENTATION itself.
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_STB_IMAGE
#define STBI_FAILURE_USERMSG
#include <tiny_gltf.h>

#include <numbers>
#include <unordered_map>
#include <cassert>

namespace vkgfx {

// ─── AABB ─────────────────────────────────────────────────────────────────────
AABB AABB::transform(const Mat4& m) const {
    AABB result;
    Vec3 corners[8] = {
        {min.x,min.y,min.z},{max.x,min.y,min.z},
        {min.x,max.y,min.z},{max.x,max.y,min.z},
        {min.x,min.y,max.z},{max.x,min.y,max.z},
        {min.x,max.y,max.z},{max.x,max.y,max.z},
    };
    for (auto& c : corners) result.expand(Vec3(m * Vec4(c, 1.f)));
    return result;
}

// ─── Mesh ─────────────────────────────────────────────────────────────────────
Mesh::Mesh(std::vector<Vertex> verts, std::vector<uint32_t> indices)
    : m_vertices(std::move(verts)), m_indices(std::move(indices))
{
    SubMesh sub;
    sub.indexOffset = 0;
    sub.indexCount  = static_cast<uint32_t>(m_indices.size());
    m_subMeshes.push_back(sub);
    computeBounds();
    computeTangents();
}

// ─── glTF helper: extract float attribute ────────────────────────────────────
static std::vector<float> gltf_get_floats(const tinygltf::Model& model,
                                            int accessorIdx, int componentsPerElem)
{
    const auto& acc  = model.accessors[accessorIdx];
    const auto& bv   = model.bufferViews[acc.bufferView];
    const auto& buf  = model.buffers[bv.buffer];
    size_t stride = bv.byteStride != 0 ? bv.byteStride
                                        : static_cast<size_t>(componentsPerElem) * sizeof(float);
    std::vector<float> out;
    out.reserve(acc.count * componentsPerElem);
    const uint8_t* src = buf.data.data() + bv.byteOffset + acc.byteOffset;
    for (size_t i = 0; i < acc.count; ++i) {
        const float* p = reinterpret_cast<const float*>(src + i * stride);
        for (int c = 0; c < componentsPerElem; ++c) out.push_back(p[c]);
    }
    return out;
}

// ─── glTF helper: extract uint32 indices ─────────────────────────────────────
static std::vector<uint32_t> gltf_get_indices(const tinygltf::Model& model, int accessorIdx)
{
    const auto& acc = model.accessors[accessorIdx];
    const auto& bv  = model.bufferViews[acc.bufferView];
    const auto& buf = model.buffers[bv.buffer];
    const uint8_t* src = buf.data.data() + bv.byteOffset + acc.byteOffset;
    std::vector<uint32_t> out;
    out.reserve(acc.count);
    for (size_t i = 0; i < acc.count; ++i) {
        if      (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
            out.push_back(reinterpret_cast<const uint32_t*>(src)[i]);
        else if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
            out.push_back(reinterpret_cast<const uint16_t*>(src)[i]);
        else if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
            out.push_back(src[i]);
    }
    return out;
}

// ─── glTF helper: load texture from model ────────────────────────────────────
static std::shared_ptr<Texture>
gltf_load_texture(std::shared_ptr<const Context> ctx,
                  const tinygltf::Model& model,
                  int texIdx,
                  bool srgb,
                  TextureCache* cache,
                  const std::filesystem::path& modelDir)
{
    if (texIdx < 0) return nullptr;
    const auto& tex   = model.textures[texIdx];
    const auto& image = model.images[tex.source];

    // External file reference — load via stb_image (PNG/JPG/TGA/BMP/HDR).
    if (!image.uri.empty()) {
        auto absPath = modelDir / image.uri;
        if (std::filesystem::exists(absPath)) {
            TextureSettings s;
            if (cache) return cache->get(absPath, srgb, s);
            return Texture::fromFile(ctx, absPath, srgb, s);
        }
    }

    // Embedded or decoded PNG/JPG — tinygltf already decoded to RGBA8.
    if (!image.image.empty()) {
        GltfImageData d;
        d.pixels = image.image.data();
        d.width  = static_cast<uint32_t>(image.width);
        d.height = static_cast<uint32_t>(image.height);
        d.srgb   = srgb;
        TextureSettings s;
        return Texture::fromGltfImage(ctx, d, s);
    }

    return nullptr;  // no data — caller will use white fallback
}

// ─── Mesh::fromGltf ───────────────────────────────────────────────────────────
std::vector<std::shared_ptr<Mesh>>
Mesh::fromGltf(std::shared_ptr<const Context> ctx,
               const std::filesystem::path& path,
               TextureCache* cache)
{
    tinygltf::Model    model;
    tinygltf::TinyGLTF loader;
    std::string        warn, err;

    bool ok = (path.extension() == ".glb")
            ? loader.LoadBinaryFromFile(&model, &err, &warn, path.string())
            : loader.LoadASCIIFromFile (&model, &err, &warn, path.string());

    if (!warn.empty()) std::cout << "[tinygltf] " << warn;
    if (!ok) throw std::runtime_error("[VKGFX] Failed to load glTF: " + err);

    const auto modelDir = path.parent_path();

    // Pre-load all materials so sub-meshes can share them.
    std::vector<std::shared_ptr<PBRMaterial>> mats;
    mats.reserve(model.materials.size());

    for (const auto& gm : model.materials) {
        auto m = std::make_shared<PBRMaterial>();
        const auto& pbr = gm.pbrMetallicRoughness;

        // Scalar factors
        m->setAlbedo({(float)pbr.baseColorFactor[0], (float)pbr.baseColorFactor[1],
                      (float)pbr.baseColorFactor[2], (float)pbr.baseColorFactor[3]});
        m->setMetallic((float)pbr.metallicFactor);
        m->setRoughness((float)pbr.roughnessFactor);

        if (!gm.emissiveFactor.empty())
            m->setEmissive(glm::length(Vec3((float)gm.emissiveFactor[0],
                                            (float)gm.emissiveFactor[1],
                                            (float)gm.emissiveFactor[2])));

        if (gm.alphaMode == "MASK")
            m->setAlphaCutoff((float)gm.alphaCutoff);

        // Textures
        auto loadTex = [&](int idx, bool srgb) {
            return gltf_load_texture(ctx, model, idx, srgb, cache, modelDir);
        };

        if (auto t = loadTex(pbr.baseColorTexture.index,    true))  m->setTexture(PBRMaterial::ALBEDO,     t);
        if (auto t = loadTex(gm.normalTexture.index,         false)) m->setTexture(PBRMaterial::NORMAL,     t);
        if (auto t = loadTex(pbr.metallicRoughnessTexture.index, false)) m->setTexture(PBRMaterial::METALROUGH, t);
        if (auto t = loadTex(gm.emissiveTexture.index,      true))  m->setTexture(PBRMaterial::EMISSIVE,   t);
        if (auto t = loadTex(gm.occlusionTexture.index,     false)) m->setTexture(PBRMaterial::AO,         t);

        mats.push_back(std::move(m));
    }

    // Build one Mesh per glTF mesh.
    std::vector<std::shared_ptr<Mesh>> out;
    out.reserve(model.meshes.size());

    for (const auto& gltfMesh : model.meshes) {
        auto mesh = std::make_shared<Mesh>();

        for (const auto& prim : gltfMesh.primitives) {
            if (prim.mode != TINYGLTF_MODE_TRIANGLES) continue;

            SubMesh sub;
            sub.indexOffset = static_cast<uint32_t>(mesh->m_indices.size());

            // ── Indices ───────────────────────────────────────────────────────
            std::vector<uint32_t> primIndices;
            if (prim.indices >= 0) {
                primIndices = gltf_get_indices(model, prim.indices);
            }

            // ── Vertex attributes ─────────────────────────────────────────────
            auto it_pos  = prim.attributes.find("POSITION");
            auto it_norm = prim.attributes.find("NORMAL");
            auto it_uv   = prim.attributes.find("TEXCOORD_0");
            auto it_tan  = prim.attributes.find("TANGENT");

            if (it_pos == prim.attributes.end()) continue;

            auto positions = gltf_get_floats(model, it_pos->second,  3);
            size_t vcount  = positions.size() / 3;

            std::vector<float> normals  (vcount * 3, 0.f);
            std::vector<float> uvs      (vcount * 2, 0.f);
            std::vector<float> tangents (vcount * 4, 0.f);  // vec4

            if (it_norm != prim.attributes.end())
                normals  = gltf_get_floats(model, it_norm->second, 3);
            if (it_uv   != prim.attributes.end())
                uvs      = gltf_get_floats(model, it_uv->second,   2);
            if (it_tan  != prim.attributes.end())
                tangents = gltf_get_floats(model, it_tan->second,  4);

            uint32_t baseVertex = static_cast<uint32_t>(mesh->m_vertices.size());

            for (size_t i = 0; i < vcount; ++i) {
                Vertex v{};
                v.position = { positions[i*3], positions[i*3+1], positions[i*3+2] };
                v.normal   = { normals[i*3],   normals[i*3+1],   normals[i*3+2]   };
                v.uv       = { uvs[i*2],        uvs[i*2+1]                          };
                v.tangent  = { tangents[i*4],   tangents[i*4+1],
                               tangents[i*4+2], tangents[i*4+3] };
                mesh->m_vertices.push_back(v);
            }

            // Offset indices by base vertex so all primitives share one VBO.
            for (uint32_t idx : primIndices)
                mesh->m_indices.push_back(baseVertex + idx);

            sub.indexCount = static_cast<uint32_t>(mesh->m_indices.size()) - sub.indexOffset;

            // Assign material (or default white PBR).
            if (prim.material >= 0 && prim.material < (int)mats.size())
                sub.material = mats[prim.material];
            else
                sub.material = std::make_shared<PBRMaterial>();

            mesh->m_subMeshes.push_back(std::move(sub));
        }

        // Generate tangents for any primitive that lacked TANGENT attributes.
        mesh->computeTangents();
        mesh->computeBounds();

        std::cout << "[VKGFX] glTF mesh: " << gltfMesh.name
                  << " (" << mesh->m_vertices.size() << " verts, "
                  << mesh->m_indices.size() / 3 << " tris, "
                  << mesh->m_subMeshes.size() << " sub-meshes)\n";

        out.push_back(std::move(mesh));
    }

    return out;
}

// ─── setMaterial / getMaterial ────────────────────────────────────────────────
void Mesh::setMaterial(std::shared_ptr<PBRMaterial> mat, uint32_t slot) {
    if (m_subMeshes.empty()) {
        SubMesh sub; sub.indexCount = static_cast<uint32_t>(m_indices.size());
        m_subMeshes.push_back(sub);
    }
    if (slot < m_subMeshes.size()) m_subMeshes[slot].material = std::move(mat);
}

std::shared_ptr<PBRMaterial> Mesh::getMaterial(uint32_t slot) const {
    return (slot < m_subMeshes.size()) ? m_subMeshes[slot].material : nullptr;
}

// ─── Procedural primitives ────────────────────────────────────────────────────
std::shared_ptr<Mesh> Mesh::createCube(float s) {
    float h = s * 0.5f;
    std::vector<Vertex> verts = {
        {{-h,-h, h},{0,0,1},{0,0}},{{ h,-h, h},{0,0,1},{1,0}},
        {{ h, h, h},{0,0,1},{1,1}},{{-h, h, h},{0,0,1},{0,1}},
        {{ h,-h,-h},{0,0,-1},{0,0}},{{-h,-h,-h},{0,0,-1},{1,0}},
        {{-h, h,-h},{0,0,-1},{1,1}},{{ h, h,-h},{0,0,-1},{0,1}},
        {{-h,-h,-h},{-1,0,0},{0,0}},{{-h,-h, h},{-1,0,0},{1,0}},
        {{-h, h, h},{-1,0,0},{1,1}},{{-h, h,-h},{-1,0,0},{0,1}},
        {{ h,-h, h},{1,0,0},{0,0}},{{ h,-h,-h},{1,0,0},{1,0}},
        {{ h, h,-h},{1,0,0},{1,1}},{{ h, h, h},{1,0,0},{0,1}},
        {{-h, h, h},{0,1,0},{0,0}},{{ h, h, h},{0,1,0},{1,0}},
        {{ h, h,-h},{0,1,0},{1,1}},{{-h, h,-h},{0,1,0},{0,1}},
        {{-h,-h,-h},{0,-1,0},{0,0}},{{ h,-h,-h},{0,-1,0},{1,0}},
        {{ h,-h, h},{0,-1,0},{1,1}},{{-h,-h, h},{0,-1,0},{0,1}},
    };
    std::vector<uint32_t> idx;
    for (uint32_t f = 0; f < 6; ++f) {
        uint32_t b = f*4;
        idx.insert(idx.end(),{b,b+1,b+2,b,b+2,b+3});
    }
    return std::make_shared<Mesh>(std::move(verts), std::move(idx));
}

std::shared_ptr<Mesh> Mesh::createSphere(float radius, uint32_t sectors, uint32_t stacks) {
    std::vector<Vertex> verts;
    std::vector<uint32_t> idx;
    const float pi = std::numbers::pi_v<float>;
    for (uint32_t i = 0; i <= stacks; ++i) {
        float phi = pi * (static_cast<float>(i) / stacks);
        for (uint32_t j = 0; j <= sectors; ++j) {
            float theta = 2.f * pi * (static_cast<float>(j) / sectors);
            Vec3 n{ std::sin(phi)*std::cos(theta), std::cos(phi), std::sin(phi)*std::sin(theta) };
            Vertex v;
            v.position = n * radius; v.normal = n;
            v.uv = { static_cast<float>(j)/sectors, static_cast<float>(i)/stacks };
            verts.push_back(v);
        }
    }
    for (uint32_t i = 0; i < stacks; ++i)
        for (uint32_t j = 0; j < sectors; ++j) {
            uint32_t a = i*(sectors+1)+j, b = a+sectors+1;
            if (i != 0)         idx.insert(idx.end(),{a,b,a+1});
            if (i != stacks-1)  idx.insert(idx.end(),{a+1,b,b+1});
        }
    return std::make_shared<Mesh>(std::move(verts), std::move(idx));
}

std::shared_ptr<Mesh> Mesh::createPlane(float size, uint32_t sub) {
    float h = size*0.5f;
    std::vector<Vertex> verts;
    std::vector<uint32_t> idx;
    for (uint32_t z = 0; z <= sub; ++z)
        for (uint32_t x = 0; x <= sub; ++x) {
            Vertex v;
            v.position = {-h + size*(float(x)/sub), 0.f, -h + size*(float(z)/sub)};
            v.normal   = {0.f,1.f,0.f};
            v.uv       = {float(x)/sub, float(z)/sub};
            verts.push_back(v);
        }
    for (uint32_t z = 0; z < sub; ++z)
        for (uint32_t x = 0; x < sub; ++x) {
            uint32_t tl=z*(sub+1)+x, tr=tl+1, bl=(z+1)*(sub+1)+x, br=bl+1;
            idx.insert(idx.end(),{tl,bl,tr, tr,bl,br});
        }
    return std::make_shared<Mesh>(std::move(verts), std::move(idx));
}

std::shared_ptr<Mesh> Mesh::createQuad() {
    std::vector<Vertex> v = {
        {{-1,-1,0},{0,0,1},{0,1}},{{1,-1,0},{0,0,1},{1,1}},
        {{1, 1,0},{0,0,1},{1,0}},{{-1,1,0},{0,0,1},{0,0}},
    };
    return std::make_shared<Mesh>(std::move(v), std::vector<uint32_t>{0,1,2,0,2,3});
}

// ─── Tangent generation ───────────────────────────────────────────────────────
void Mesh::computeTangents() {
    // Skip vertices that already have valid tangents (from glTF TANGENT accessor).
    bool hasTangents = false;
    for (auto& v : m_vertices)
        if (glm::length2(Vec3(v.tangent)) > 1e-4f) { hasTangents = true; break; }
    if (hasTangents) return;

    // MikkTSpace-style accumulation.
    for (size_t i = 0; i + 2 < m_indices.size(); i += 3) {
        auto& v0 = m_vertices[m_indices[i]];
        auto& v1 = m_vertices[m_indices[i+1]];
        auto& v2 = m_vertices[m_indices[i+2]];
        Vec3 e1 = v1.position - v0.position;
        Vec3 e2 = v2.position - v0.position;
        Vec2 d1 = v1.uv - v0.uv;
        Vec2 d2 = v2.uv - v0.uv;
        float f = 1.f / (d1.x*d2.y - d2.x*d1.y + 1e-6f);
        Vec3 t  = f * (d2.y * e1 - d1.y * e2);
        v0.tangent += Vec4(t, 0.f);
        v1.tangent += Vec4(t, 0.f);
        v2.tangent += Vec4(t, 0.f);
    }

    for (auto& v : m_vertices) {
        Vec3 t = Vec3(v.tangent);
        if (glm::length2(t) > 1e-8f) {
            t = glm::normalize(t);
        } else {
            Vec3 n   = glm::normalize(v.normal);
            Vec3 aux = (std::abs(n.x) <= std::abs(n.y) && std::abs(n.x) <= std::abs(n.z))
                       ? Vec3(1.f,0.f,0.f) : Vec3(0.f,1.f,0.f);
            t = glm::normalize(aux - n * glm::dot(aux, n));
        }
        v.tangent = Vec4(t, 1.f);  // w=+1 handedness default
    }
}

void Mesh::computeBounds() {
    m_bounds = {};
    for (auto& v : m_vertices) m_bounds.expand(v.position);
}

void Mesh::updateMatrix() const {
    if (!m_dirty) return;
    Mat4 T = glm::translate(Mat4(1.f), m_position);
    Mat4 R = glm::toMat4(glm::quat(glm::radians(m_rotation)));
    Mat4 S = glm::scale(Mat4(1.f), m_scale);
    m_model  = T * R * S;
    m_normal = glm::transpose(glm::inverse(m_model));
    m_dirty  = false;
}

} // namespace vkgfx
