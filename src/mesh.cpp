#include "vkgfx/mesh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <numbers>
#include <unordered_map>

namespace vkgfx {

// ─── AABB ─────────────────────────────────────────────────────────────────────
AABB AABB::transform(const Mat4& m) const {
    // Transform all 8 corners and re-expand
    AABB result;
    Vec3 corners[8] = {
        {min.x, min.y, min.z}, {max.x, min.y, min.z},
        {min.x, max.y, min.z}, {max.x, max.y, min.z},
        {min.x, min.y, max.z}, {max.x, min.y, max.z},
        {min.x, max.y, max.z}, {max.x, max.y, max.z},
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

std::shared_ptr<Mesh> Mesh::fromFile(const std::filesystem::path& path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t>    shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                path.string().c_str(),
                                path.parent_path().string().c_str());
    if (!warn.empty()) std::cout << "[TinyOBJ] " << warn;
    if (!ok) throw std::runtime_error("[VKGFX] Failed to load OBJ: " + err);

    auto mesh = std::make_shared<Mesh>();
    std::unordered_map<Vertex, uint32_t> uniqueVerts;

    for (auto& shape : shapes) {
        SubMesh sub;
        sub.indexOffset = static_cast<uint32_t>(mesh->m_indices.size());

        for (auto& idx : shape.mesh.indices) {
            Vertex v{};
            v.position = {
                attrib.vertices[3 * idx.vertex_index + 0],
                attrib.vertices[3 * idx.vertex_index + 1],
                attrib.vertices[3 * idx.vertex_index + 2],
            };
            if (idx.normal_index >= 0) {
                v.normal = {
                    attrib.normals[3 * idx.normal_index + 0],
                    attrib.normals[3 * idx.normal_index + 1],
                    attrib.normals[3 * idx.normal_index + 2],
                };
            }
            if (idx.texcoord_index >= 0) {
                v.uv = {
                    attrib.texcoords[2 * idx.texcoord_index + 0],
                    1.f - attrib.texcoords[2 * idx.texcoord_index + 1],
                };
            }
            // Deduplicate vertices
            if (auto it = uniqueVerts.find(v); it != uniqueVerts.end()) {
                mesh->m_indices.push_back(it->second);
            } else {
                auto i = static_cast<uint32_t>(mesh->m_vertices.size());
                uniqueVerts[v] = i;
                mesh->m_vertices.push_back(v);
                mesh->m_indices.push_back(i);
            }
        }
        sub.indexCount = static_cast<uint32_t>(mesh->m_indices.size()) - sub.indexOffset;
        mesh->m_subMeshes.push_back(sub);
    }

    mesh->computeTangents();
    mesh->computeBounds();

    std::cout << "[VKGFX] Loaded mesh: " << path.filename().string()
              << " (" << mesh->m_vertices.size() << " verts, "
              << mesh->m_indices.size() / 3 << " tris)\n";
    return mesh;
}

std::shared_ptr<Mesh> Mesh::createCube(float s) {
    float h = s * 0.5f;
    std::vector<Vertex> verts = {
        // Front
        {{-h,-h, h},{0,0,1},{0,0}}, {{ h,-h, h},{0,0,1},{1,0}},
        {{ h, h, h},{0,0,1},{1,1}}, {{-h, h, h},{0,0,1},{0,1}},
        // Back
        {{ h,-h,-h},{0,0,-1},{0,0}}, {{-h,-h,-h},{0,0,-1},{1,0}},
        {{-h, h,-h},{0,0,-1},{1,1}}, {{ h, h,-h},{0,0,-1},{0,1}},
        // Left
        {{-h,-h,-h},{-1,0,0},{0,0}}, {{-h,-h, h},{-1,0,0},{1,0}},
        {{-h, h, h},{-1,0,0},{1,1}}, {{-h, h,-h},{-1,0,0},{0,1}},
        // Right
        {{ h,-h, h},{1,0,0},{0,0}}, {{ h,-h,-h},{1,0,0},{1,0}},
        {{ h, h,-h},{1,0,0},{1,1}}, {{ h, h, h},{1,0,0},{0,1}},
        // Top
        {{-h, h, h},{0,1,0},{0,0}}, {{ h, h, h},{0,1,0},{1,0}},
        {{ h, h,-h},{0,1,0},{1,1}}, {{-h, h,-h},{0,1,0},{0,1}},
        // Bottom
        {{-h,-h,-h},{0,-1,0},{0,0}}, {{ h,-h,-h},{0,-1,0},{1,0}},
        {{ h,-h, h},{0,-1,0},{1,1}}, {{-h,-h, h},{0,-1,0},{0,1}},
    };
    std::vector<uint32_t> indices;
    for (uint32_t f = 0; f < 6; ++f) {
        uint32_t base = f * 4;
        indices.insert(indices.end(), {base,base+1,base+2, base,base+2,base+3});
    }
    return std::make_shared<Mesh>(std::move(verts), std::move(indices));
}

std::shared_ptr<Mesh> Mesh::createSphere(float radius, uint32_t sectors, uint32_t stacks) {
    std::vector<Vertex> verts;
    std::vector<uint32_t> indices;
    const float pi = std::numbers::pi_v<float>;

    for (uint32_t i = 0; i <= stacks; ++i) {
        float phi = pi * (static_cast<float>(i) / stacks);
        for (uint32_t j = 0; j <= sectors; ++j) {
            float theta = 2.f * pi * (static_cast<float>(j) / sectors);
            Vec3 n{
                std::sin(phi) * std::cos(theta),
                std::cos(phi),
                std::sin(phi) * std::sin(theta),
            };
            Vertex v;
            v.position = n * radius;
            v.normal   = n;
            v.uv       = { static_cast<float>(j) / sectors,
                           static_cast<float>(i) / stacks };
            verts.push_back(v);
        }
    }
    for (uint32_t i = 0; i < stacks; ++i) {
        for (uint32_t j = 0; j < sectors; ++j) {
            uint32_t a = i * (sectors + 1) + j;
            uint32_t b = a + sectors + 1;
            if (i != 0)           indices.insert(indices.end(), {a, b, a + 1});
            if (i != stacks - 1)  indices.insert(indices.end(), {a + 1, b, b + 1});
        }
    }
    return std::make_shared<Mesh>(std::move(verts), std::move(indices));
}

std::shared_ptr<Mesh> Mesh::createPlane(float size, uint32_t sub) {
    float h = size * 0.5f;
    std::vector<Vertex> verts;
    std::vector<uint32_t> indices;
    for (uint32_t z = 0; z <= sub; ++z) {
        for (uint32_t x = 0; x <= sub; ++x) {
            Vertex v;
            v.position = { -h + size * (static_cast<float>(x) / sub),
                            0.f,
                           -h + size * (static_cast<float>(z) / sub) };
            v.normal   = {0.f, 1.f, 0.f};
            v.uv       = { static_cast<float>(x) / sub, static_cast<float>(z) / sub };
            verts.push_back(v);
        }
    }
    for (uint32_t z = 0; z < sub; ++z) {
        for (uint32_t x = 0; x < sub; ++x) {
            uint32_t tl = z*(sub+1)+x, tr = tl+1, bl = (z+1)*(sub+1)+x, br = bl+1;
            indices.insert(indices.end(), {tl,bl,tr, tr,bl,br});
        }
    }
    return std::make_shared<Mesh>(std::move(verts), std::move(indices));
}

std::shared_ptr<Mesh> Mesh::createQuad() {
    std::vector<Vertex> verts = {
        {{-1,-1,0},{0,0,1},{0,1}},{{1,-1,0},{0,0,1},{1,1}},
        {{1, 1,0},{0,0,1},{1,0}},{{-1,1,0},{0,0,1},{0,0}},
    };
    return std::make_shared<Mesh>(std::move(verts), std::vector<uint32_t>{0,1,2,0,2,3});
}

void Mesh::setMaterial(std::shared_ptr<Material> mat, uint32_t slot) {
    if (m_subMeshes.empty()) {
        SubMesh sub;
        sub.indexCount = static_cast<uint32_t>(m_indices.size());
        m_subMeshes.push_back(sub);
    }
    if (slot < m_subMeshes.size())
        m_subMeshes[slot].material = std::move(mat);
}

std::shared_ptr<Material> Mesh::getMaterial(uint32_t slot) const {
    if (slot < m_subMeshes.size()) return m_subMeshes[slot].material;
    return nullptr;
}

void Mesh::computeTangents() {
    // Accumulate tangent contributions per triangle
    for (size_t i = 0; i + 2 < m_indices.size(); i += 3) {
        auto& v0 = m_vertices[m_indices[i]];
        auto& v1 = m_vertices[m_indices[i+1]];
        auto& v2 = m_vertices[m_indices[i+2]];
        Vec3 e1 = v1.position - v0.position;
        Vec3 e2 = v2.position - v0.position;
        Vec2 d1 = v1.uv - v0.uv;
        Vec2 d2 = v2.uv - v0.uv;
        float f = 1.f / (d1.x*d2.y - d2.x*d1.y + 1e-6f);
        Vec3 t = f * (d2.y * e1 - d1.y * e2);
        v0.tangent += t; v1.tangent += t; v2.tangent += t;
    }

    // Normalize, with a safe fallback for degenerate (zero-length) tangents.
    // FIX: previously left tangent as {0,0,0} when the triangle had no UV seam
    // (e.g. sphere poles, flat-UV geometry). A zero tangent causes the TBN
    // matrix to be singular, producing NaN/black pixels on normal-mapped surfaces.
    // We now fall back to an arbitrary vector perpendicular to the vertex normal.
    for (auto& v : m_vertices) {
        if (glm::length2(v.tangent) > 1e-8f) {
            v.tangent = glm::normalize(v.tangent);
        } else {
            // Build an orthogonal tangent from the normal using the
            // "frisvad" / "Hughes-Moeller" stable-normal-basis method.
            Vec3 n = glm::normalize(v.normal);
            // Pick the axis least parallel to n to avoid a near-zero cross product
            Vec3 aux = (std::abs(n.x) <= std::abs(n.y) && std::abs(n.x) <= std::abs(n.z))
                       ? Vec3(1.f, 0.f, 0.f)
                       : Vec3(0.f, 1.f, 0.f);
            v.tangent = glm::normalize(aux - n * glm::dot(aux, n));
        }
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
