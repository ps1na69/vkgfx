#pragma once
// mesh.h — glTF 2.0 mesh loading and GPU buffer management.
//
// Mesh::fromGltf() replaces the removed Mesh::fromFile() (OBJ/tinyobjloader).
// All mesh data comes from glTF 2.0; sub-meshes map 1-to-1 with glTF primitives.
// Materials are created as PBRMaterial from glTF PBR metallic-roughness data.

#include "material.h"
#include "texture.h"
#include <filesystem>
#include <functional>

namespace vkgfx {

struct AABB {
    Vec3 min{  std::numeric_limits<float>::max()};
    Vec3 max{ -std::numeric_limits<float>::max()};

    void expand(Vec3 p) { min = glm::min(min, p); max = glm::max(max, p); }
    [[nodiscard]] Vec3 center()  const { return (min + max) * 0.5f; }
    [[nodiscard]] Vec3 extents() const { return (max - min) * 0.5f; }
    [[nodiscard]] AABB transform(const Mat4& m) const;
};

struct SubMesh {
    uint32_t                      indexOffset = 0;
    uint32_t                      indexCount  = 0;
    std::shared_ptr<PBRMaterial>  material;
    AABB                          localBounds;
};

class Mesh {
public:
    Mesh() = default;
    Mesh(std::vector<Vertex> verts, std::vector<uint32_t> indices);
    ~Mesh() = default;

    // ── glTF loading ──────────────────────────────────────────────────────────
    // Load all meshes from a glTF 2.0 file (.gltf / .glb).
    // Returns one Mesh per glTF mesh node.  Textures are loaded from files
    // alongside the glTF; use KTX2 (.ktx2) for all texture assets.
    static std::vector<std::shared_ptr<Mesh>>
    fromGltf(std::shared_ptr<const Context> ctx,
             const std::filesystem::path& path,
             TextureCache* cache = nullptr);

    // ── Procedural primitives (unchanged) ────────────────────────────────────
    static std::shared_ptr<Mesh> createCube(float size = 1.f);
    static std::shared_ptr<Mesh> createSphere(float radius = 1.f,
                                               uint32_t sectors = 32,
                                               uint32_t stacks  = 16);
    static std::shared_ptr<Mesh> createPlane(float size = 1.f, uint32_t subdivisions = 1);
    static std::shared_ptr<Mesh> createQuad();

    [[nodiscard]] const std::vector<Vertex>&   vertices()  const { return m_vertices; }
    [[nodiscard]] const std::vector<uint32_t>& indices()   const { return m_indices; }
    [[nodiscard]] const std::vector<SubMesh>&  subMeshes() const { return m_subMeshes; }
    [[nodiscard]] std::vector<SubMesh>&        subMeshes()       { return m_subMeshes; }

    void setMaterial(std::shared_ptr<PBRMaterial> mat, uint32_t slot = 0);
    [[nodiscard]] std::shared_ptr<PBRMaterial> getMaterial(uint32_t slot = 0) const;

    void setPosition(Vec3 p)     { m_position = p; m_dirty = true; }
    void setRotation(Vec3 euler) { m_rotation = euler; m_dirty = true; }
    void setScale(Vec3 s)        { m_scale = s; m_dirty = true; }
    void setScale(float s)       { m_scale = Vec3(s); m_dirty = true; }

    [[nodiscard]] Vec3 position() const { return m_position; }
    [[nodiscard]] Vec3 rotation() const { return m_rotation; }
    [[nodiscard]] Vec3 scale()    const { return m_scale; }
    [[nodiscard]] const Mat4& modelMatrix()  const { updateMatrix(); return m_model; }
    [[nodiscard]] const Mat4& normalMatrix() const { updateMatrix(); return m_normal; }
    [[nodiscard]] const AABB& localBounds()  const { return m_bounds; }
    [[nodiscard]] AABB         worldBounds()  const { return m_bounds.transform(modelMatrix()); }

    [[nodiscard]] bool isVisible() const { return m_visible; }
    void setVisible(bool v) { m_visible = v; }

    AllocatedBuffer vertexBuffer;
    AllocatedBuffer indexBuffer;
    bool            gpuReady = false;

    void computeTangents();
    void computeBounds();

private:
    void updateMatrix() const;

    std::vector<Vertex>    m_vertices;
    std::vector<uint32_t>  m_indices;
    std::vector<SubMesh>   m_subMeshes;

    Vec3 m_position{0.f};
    Vec3 m_rotation{0.f};
    Vec3 m_scale{1.f};
    mutable Mat4 m_model{1.f};
    mutable Mat4 m_normal{1.f};
    mutable bool m_dirty = true;

    AABB m_bounds;
    bool m_visible = true;
};

} // namespace vkgfx
