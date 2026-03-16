#pragma once
// include/vkgfx/mesh.h

#include "types.h"
#include "vk_raii.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <memory>
#include <string>
#include <vector>

namespace vkgfx {

class Context;
class PBRMaterial;

struct AABB {
    glm::vec3 min{ 1e9f,  1e9f,  1e9f};
    glm::vec3 max{-1e9f, -1e9f, -1e9f};
    void expand(const glm::vec3& p) { min = glm::min(min, p); max = glm::max(max, p); }
    [[nodiscard]] glm::vec3 center()  const { return (min + max) * 0.5f; }
    [[nodiscard]] glm::vec3 extents() const { return (max - min) * 0.5f; }
};

class Mesh {
public:
    ~Mesh();
    Mesh(const Mesh&)            = delete;
    Mesh& operator=(const Mesh&) = delete;

    static std::shared_ptr<Mesh> loadOBJ    (const std::string& path, Context& ctx);
    static std::shared_ptr<Mesh> createSphere(float radius, uint32_t stacks,
                                              uint32_t slices, Context& ctx);
    static std::shared_ptr<Mesh> createBox  (glm::vec3 halfExtents, Context& ctx);

    [[nodiscard]] bool            uploaded()     const { return m_uploaded; }
    [[nodiscard]] uint32_t        indexCount()   const { return m_indexCount; }
    [[nodiscard]] const AABB&     aabb()         const { return m_aabb; }
    [[nodiscard]] const VkBuffer& vertexBuffer() const { return m_vbo.buffer; }
    [[nodiscard]] const VkBuffer& indexBuffer()  const { return m_ibo.buffer; }

    Mesh& setPosition(glm::vec3 p);
    Mesh& setRotation(glm::quat q);
    Mesh& setScale   (glm::vec3 s);
    [[nodiscard]] glm::mat4 modelMatrix()  const;
    [[nodiscard]] glm::mat4 normalMatrix() const;

    Mesh& setMaterial(std::shared_ptr<PBRMaterial> m);
    [[nodiscard]] PBRMaterial* material() const { return m_material.get(); }

    void destroy(Context& ctx);

private:
    Mesh() = default;
    void upload(Context& ctx, std::vector<Vertex>& verts, std::vector<uint32_t>& indices);

    AllocatedBuffer m_vbo{};
    AllocatedBuffer m_ibo{};
    uint32_t        m_indexCount = 0;
    AABB            m_aabb{};
    bool            m_uploaded   = false;

    glm::vec3 m_position{0.f, 0.f, 0.f};
    glm::quat m_rotation{1.f, 0.f, 0.f, 0.f};
    glm::vec3 m_scale   {1.f, 1.f, 1.f};

    std::shared_ptr<PBRMaterial> m_material;
};

} // namespace vkgfx
