#pragma once
// include/vkgfx/scene.h

#include "mesh.h"
#include "camera.h"
#include "types.h"

#include <memory>
#include <utility>
#include <vector>
#include <glm/glm.hpp>

namespace vkgfx {

// ── Light classes (scene-graph API, not GPU structs) ──────────────────────────

class DirectionalLight {
public:
    DirectionalLight& setDirection(float x, float y, float z);
    DirectionalLight& setDirection(glm::vec3 d);
    DirectionalLight& setColor(float r, float g, float b);
    DirectionalLight& setIntensity(float v);
    DirectionalLight& setEnabled(bool v);

    [[nodiscard]] bool      enabled()   const { return m_enabled; }
    [[nodiscard]] glm::vec3 direction() const { return m_direction; }
    [[nodiscard]] glm::vec3 color()     const { return m_color; }
    [[nodiscard]] float     intensity() const { return m_intensity; }

private:
    glm::vec3 m_direction = glm::normalize(glm::vec3(-0.4f, -1.0f, -0.3f));
    glm::vec3 m_color     = {1.0f, 0.98f, 0.95f};
    float     m_intensity = 5.0f;
    bool      m_enabled   = true;
};

class PointLight {
public:
    PointLight& setPosition(glm::vec3 p);
    PointLight& setColor(float r, float g, float b);
    PointLight& setIntensity(float v);
    PointLight& setRadius(float r);
    PointLight& setEnabled(bool v);
    /// Controls whether this light contributes to the point-shadow cubemap.
    /// Only the first light with castsShadow()==true is rendered into the cube map.
    PointLight& setCastsShadow(bool v) { m_castsShadow = v; return *this; }

    [[nodiscard]] glm::vec3 position()    const { return m_position; }
    [[nodiscard]] glm::vec3 color()       const { return m_color; }
    [[nodiscard]] float     intensity()   const { return m_intensity; }
    [[nodiscard]] float     radius()      const { return m_radius; }
    [[nodiscard]] bool      enabled()     const { return m_enabled; }
    [[nodiscard]] bool      castsShadow() const { return m_castsShadow; }

private:
    glm::vec3 m_position    = {0.f, 0.f, 0.f};
    glm::vec3 m_color       = {1.f, 1.f, 1.f};
    float     m_intensity   = 1.0f;
    float     m_radius      = 10.0f;
    bool      m_enabled     = true;
    bool      m_castsShadow = true;  // shadow-casting on by default
};

// ── Scene ─────────────────────────────────────────────────────────────────────

class Scene {
public:
    Scene() = default;

    void setCamera(Camera* cam) { m_camera = cam; }
    [[nodiscard]] Camera* camera() const { return m_camera; }

    void add(std::shared_ptr<Mesh> m)             { m_meshes.push_back(std::move(m)); }
    void add(std::shared_ptr<DirectionalLight> l) { m_dirLight = std::move(l); }
    void add(std::shared_ptr<PointLight> l) {
        if (m_pointLights.size() < MAX_POINT_LIGHTS)
            m_pointLights.push_back(std::move(l));
    }

    [[nodiscard]] std::vector<Mesh*> visibleMeshes() const;

    [[nodiscard]] const std::vector<std::shared_ptr<Mesh>>&       meshes()      const { return m_meshes; }
    [[nodiscard]] DirectionalLight*                                dirLight()    const { return m_dirLight.get(); }
    [[nodiscard]] const std::vector<std::shared_ptr<PointLight>>& pointLights() const { return m_pointLights; }

    void fillLightUBO(LightUBO& ubo, float iblIntensity, uint32_t gbufferDebug) const;

private:
    Camera*                                  m_camera = nullptr;
    std::vector<std::shared_ptr<Mesh>>       m_meshes;
    std::shared_ptr<DirectionalLight>        m_dirLight;
    std::vector<std::shared_ptr<PointLight>> m_pointLights;
};

} // namespace vkgfx
