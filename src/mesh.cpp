// src/mesh.cpp
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <vkgfx/mesh.h>
#include <vkgfx/context.h>
#include <vkgfx/material.h>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/quaternion.hpp>
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <cmath>

namespace vkgfx {

// ── Vertex hash for deduplication ─────────────────────────────────────────────

struct VertexHash {
    size_t operator()(const Vertex& v) const {
        size_t seed = 0;
        auto h = [&](float f) {
            seed ^= std::hash<float>{}(f) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        };
        h(v.position.x); h(v.position.y); h(v.position.z);
        h(v.normal.x);   h(v.normal.y);   h(v.normal.z);
        h(v.uv.x);       h(v.uv.y);
        return seed;
    }
};

struct VertexEq {
    bool operator()(const Vertex& a, const Vertex& b) const {
        return a.position == b.position
            && a.normal   == b.normal
            && a.uv       == b.uv;
    }
};

// ── Tangent generation (MikkTSpace-like simplified) ───────────────────────────

static void generateTangents(std::vector<Vertex>& verts, const std::vector<uint32_t>& idx) {
    for (size_t i = 0; i + 2 < idx.size(); i += 3) {
        Vertex& v0 = verts[idx[i]];
        Vertex& v1 = verts[idx[i + 1]];
        Vertex& v2 = verts[idx[i + 2]];

        glm::vec3 edge1 = v1.position - v0.position;
        glm::vec3 edge2 = v2.position - v0.position;
        glm::vec2 dUV1  = v1.uv - v0.uv;
        glm::vec2 dUV2  = v2.uv - v0.uv;

        float f = dUV1.x * dUV2.y - dUV2.x * dUV1.y;
        if (std::abs(f) < 1e-6f) continue;
        float inv = 1.0f / f;

        glm::vec3 tangent{
            inv * (dUV2.y * edge1.x - dUV1.y * edge2.x),
            inv * (dUV2.y * edge1.y - dUV1.y * edge2.y),
            inv * (dUV2.y * edge1.z - dUV1.y * edge2.z)
        };

        // Accumulate (normalise at end)
        for (int j = 0; j < 3; ++j)
            verts[idx[i + j]].tangent += glm::vec4(tangent, 0.f);
    }
    for (auto& v : verts) {
        glm::vec3 t = glm::vec3(v.tangent);
        if (glm::length(t) > 1e-6f) {
            t = glm::normalize(t - v.normal * glm::dot(v.normal, t));
            v.tangent = glm::vec4(t, 1.0f);
        } else {
            // Fallback
            glm::vec3 up = std::abs(v.normal.y) < 0.99f ? glm::vec3(0,1,0) : glm::vec3(1,0,0);
            v.tangent = glm::vec4(glm::normalize(glm::cross(v.normal, up)), 1.0f);
        }
    }
}

// ── upload ────────────────────────────────────────────────────────────────────

void Mesh::upload(Context& ctx, std::vector<Vertex>& verts, std::vector<uint32_t>& indices) {
    for (auto& v : verts) m_aabb.expand(v.position);

    m_vbo = ctx.uploadBuffer(verts.data(),
        sizeof(Vertex) * verts.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    m_ibo = ctx.uploadBuffer(indices.data(),
        sizeof(uint32_t) * indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    m_indexCount = static_cast<uint32_t>(indices.size());
    m_uploaded   = true;
}

void Mesh::destroy(Context& ctx) {
    ctx.destroyBuffer(m_vbo);
    ctx.destroyBuffer(m_ibo);
    m_uploaded = false;
}

Mesh::~Mesh() = default;

// ── loadOBJ ───────────────────────────────────────────────────────────────────

std::shared_ptr<Mesh> Mesh::loadOBJ(const std::string& path, Context& ctx) {
    if (!std::filesystem::exists(path))
        throw std::runtime_error("[vkgfx] OBJ not found: " + path);

    tinyobj::attrib_t                attrib;
    std::vector<tinyobj::shape_t>    shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    std::string dir = std::filesystem::path(path).parent_path().string();
    bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                path.c_str(), dir.c_str());
    if (!warn.empty()) std::cerr << "[tinyobj] " << warn << "\n";
    if (!ok)           throw std::runtime_error("[vkgfx] tinyobj error: " + err);

    std::unordered_map<Vertex, uint32_t, VertexHash, VertexEq> uniqueVerts;
    std::vector<Vertex>   verts;
    std::vector<uint32_t> indices;

    for (auto& shape : shapes) {
        for (auto& idx : shape.mesh.indices) {
            Vertex v{};
            int vi = idx.vertex_index * 3;
            v.position = {attrib.vertices[vi], attrib.vertices[vi+1], attrib.vertices[vi+2]};

            if (idx.normal_index >= 0) {
                int ni = idx.normal_index * 3;
                v.normal = {attrib.normals[ni], attrib.normals[ni+1], attrib.normals[ni+2]};
            }

            if (idx.texcoord_index >= 0) {
                int ti = idx.texcoord_index * 2;
                v.uv = {attrib.texcoords[ti], 1.0f - attrib.texcoords[ti+1]};
            }

            auto it = uniqueVerts.find(v);
            if (it == uniqueVerts.end()) {
                uint32_t newIdx = static_cast<uint32_t>(verts.size());
                uniqueVerts[v]  = newIdx;
                verts.push_back(v);
                indices.push_back(newIdx);
            } else {
                indices.push_back(it->second);
            }
        }
    }

    generateTangents(verts, indices);

    auto mesh = std::shared_ptr<Mesh>(new Mesh());
    mesh->upload(ctx, verts, indices);
    return mesh;
}

// ── createSphere ─────────────────────────────────────────────────────────────

std::shared_ptr<Mesh> Mesh::createSphere(float radius, uint32_t stacks,
                                          uint32_t slices, Context& ctx) {
    std::vector<Vertex>   verts;
    std::vector<uint32_t> indices;

    const float PI = 3.14159265f;
    for (uint32_t i = 0; i <= stacks; ++i) {
        float phi   = PI * float(i) / float(stacks);
        float cosPhi= std::cos(phi), sinPhi = std::sin(phi);

        for (uint32_t j = 0; j <= slices; ++j) {
            float theta    = 2.0f * PI * float(j) / float(slices);
            float cosTheta = std::cos(theta), sinTheta = std::sin(theta);

            Vertex v{};
            v.normal   = {sinPhi * cosTheta, cosPhi, sinPhi * sinTheta};
            v.position = v.normal * radius;
            v.uv       = {float(j) / float(slices), float(i) / float(stacks)};
            // Tangent along theta direction
            v.tangent  = glm::vec4(-sinTheta, 0, cosTheta, 1.0f);
            verts.push_back(v);
        }
    }

    for (uint32_t i = 0; i < stacks; ++i) {
        for (uint32_t j = 0; j < slices; ++j) {
            uint32_t a = i * (slices + 1) + j;
            uint32_t b = a + slices + 1;
            // CCW winding (verified by outward-normal dot test):
            indices.insert(indices.end(), {a, a + 1, b, a + 1, b + 1, b});
        }
    }

    auto mesh = std::shared_ptr<Mesh>(new Mesh());
    mesh->upload(ctx, verts, indices);
    return mesh;
}

// ── createBox ────────────────────────────────────────────────────────────────

std::shared_ptr<Mesh> Mesh::createBox(glm::vec3 h, Context& ctx) {
    struct FaceData { glm::vec3 n, t; glm::vec3 corners[4]; };
    // Corner order is CCW when viewed from outside each face (VK_FRONT_FACE_COUNTER_CLOCKWISE).
    // +X and -X required vertex reversal — verified by cross-product winding check.
    FaceData faces[6] = {
        {{ 1,0,0},{0,0,-1}, {{ h.x,-h.y,-h.z},{ h.x, h.y,-h.z},{ h.x, h.y, h.z},{ h.x,-h.y, h.z}}},
        {{-1,0,0},{0,0, 1}, {{-h.x,-h.y, h.z},{-h.x, h.y, h.z},{-h.x, h.y,-h.z},{-h.x,-h.y,-h.z}}},
        {{ 0,1,0},{1,0, 0}, {{-h.x, h.y, h.z},{ h.x, h.y, h.z},{ h.x, h.y,-h.z},{-h.x, h.y,-h.z}}},
        {{ 0,-1,0},{1,0,0}, {{-h.x,-h.y,-h.z},{ h.x,-h.y,-h.z},{ h.x,-h.y, h.z},{-h.x,-h.y, h.z}}},
        {{ 0,0,1},{1,0, 0}, {{-h.x,-h.y, h.z},{ h.x,-h.y, h.z},{ h.x, h.y, h.z},{-h.x, h.y, h.z}}},
        {{ 0,0,-1},{-1,0,0},{{ h.x,-h.y,-h.z},{-h.x,-h.y,-h.z},{-h.x, h.y,-h.z},{ h.x, h.y,-h.z}}}
    };

    std::vector<Vertex>   verts;
    std::vector<uint32_t> indices;
    glm::vec2 uvs[4] = {{0,1},{1,1},{1,0},{0,0}};

    for (auto& f : faces) {
        uint32_t base = static_cast<uint32_t>(verts.size());
        for (int k = 0; k < 4; ++k) {
            Vertex v{};
            v.position = f.corners[k];
            v.normal   = f.n;
            v.tangent  = glm::vec4(f.t, 1.0f);
            v.uv       = uvs[k];
            verts.push_back(v);
        }
        indices.insert(indices.end(), {base,base+1,base+2, base,base+2,base+3});
    }

    auto mesh = std::shared_ptr<Mesh>(new Mesh());
    mesh->upload(ctx, verts, indices);
    return mesh;
}

// ── createTriangle ───────────────────────────────────────────────────────────

std::shared_ptr<Mesh> Mesh::createTriangle(glm::vec3 a, glm::vec3 b,
                                            glm::vec3 c, Context& ctx) {
    glm::vec3 n = glm::normalize(glm::cross(b - a, c - a));
    glm::vec3 t = glm::normalize(b - a);

    std::vector<Vertex> verts = {
        {a, n, glm::vec4(t, 1.f), {0.f, 0.f}},
        {b, n, glm::vec4(t, 1.f), {1.f, 0.f}},
        {c, n, glm::vec4(t, 1.f), {0.5f, 1.f}},
    };
    std::vector<uint32_t> indices = {0, 1, 2};

    auto mesh = std::shared_ptr<Mesh>(new Mesh());
    mesh->upload(ctx, verts, indices);
    return mesh;
}

// ── Transform ────────────────────────────────────────────────────────────────

Mesh& Mesh::setPosition(glm::vec3 p) { m_position = p; return *this; }
Mesh& Mesh::setRotation(glm::quat q) { m_rotation = q; return *this; }
Mesh& Mesh::setScale(glm::vec3 s)    { m_scale    = s; return *this; }

glm::mat4 Mesh::modelMatrix() const {
    return glm::translate(glm::mat4(1), m_position)
         * glm::toMat4(m_rotation)
         * glm::scale(glm::mat4(1), m_scale);
}

glm::mat4 Mesh::normalMatrix() const {
    return glm::transpose(glm::inverse(modelMatrix()));
}


// ── Material ──────────────────────────────────────────────────────────────────

Mesh& Mesh::setMaterial(std::shared_ptr<PBRMaterial> m) {
    m_material = std::move(m);
    return *this;
}

} // namespace vkgfx
