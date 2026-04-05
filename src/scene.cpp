// src/scene.cpp
#include <vkgfx/scene.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace vkgfx {

// ── DirectionalLight ──────────────────────────────────────────────────────────

DirectionalLight& DirectionalLight::setDirection(float x, float y, float z) {
    m_direction = glm::normalize(glm::vec3(x, y, z));
    return *this;
}
DirectionalLight& DirectionalLight::setDirection(glm::vec3 d) {
    m_direction = glm::normalize(d);
    return *this;
}
DirectionalLight& DirectionalLight::setColor(float r, float g, float b) {
    m_color = {r, g, b};
    return *this;
}
DirectionalLight& DirectionalLight::setIntensity(float v) { m_intensity = v; return *this; }
DirectionalLight& DirectionalLight::setEnabled(bool v)    { m_enabled   = v; return *this; }

// ── PointLight ────────────────────────────────────────────────────────────────

PointLight& PointLight::setPosition(glm::vec3 p)           { m_position  = p; return *this; }
PointLight& PointLight::setColor(float r, float g, float b) { m_color = {r, g, b}; return *this; }
PointLight& PointLight::setIntensity(float v)              { m_intensity = v; return *this; }
PointLight& PointLight::setRadius(float r)                 { m_radius    = r; return *this; }
PointLight& PointLight::setEnabled(bool v)                 { m_enabled   = v; return *this; }

// ── Scene ─────────────────────────────────────────────────────────────────────

std::vector<Mesh*> Scene::visibleMeshes() const {
    if (!m_camera) {
        std::vector<Mesh*> all;
        all.reserve(m_meshes.size());
        for (auto& m : m_meshes) all.push_back(m.get());
        return all;
    }
    Frustum fr = m_camera->frustum();
    std::vector<Mesh*> visible;
    for (auto& mesh : m_meshes) {
        glm::mat4  model   = mesh->modelMatrix();
        glm::vec3  center  = glm::vec3(model * glm::vec4(mesh->aabb().center(), 1.f));
        glm::mat3  m3      = glm::mat3(model);
        glm::vec3  extents = glm::abs(m3[0]) * mesh->aabb().extents().x
                           + glm::abs(m3[1]) * mesh->aabb().extents().y
                           + glm::abs(m3[2]) * mesh->aabb().extents().z;
        if (fr.intersects(center, extents))
            visible.push_back(mesh.get());
    }
    return visible;
}

void Scene::fillLightUBO(LightUBO& ubo, float iblIntensity, uint32_t gbufferDebug) const {
    ubo = LightUBO{};

    // ── Directional sun ───────────────────────────────────────────────────────
    if (m_dirLight) {
        ubo.sunDirection    = glm::vec4(m_dirLight->direction(), 0.f);
        ubo.sunColor        = glm::vec4(m_dirLight->color(), m_dirLight->intensity());
        ubo.sunFlags.x      = m_dirLight->enabled() ? 1u : 0u;
        ubo.sunFlags.y      = m_dirLight->enabled() ? 1u : 0u; // shadow follows sun
    }

    // ── Point lights ──────────────────────────────────────────────────────────
    uint32_t count = 0;
    for (auto& light : m_pointLights) {
        if (!light || count >= MAX_POINT_LIGHTS) break;
        ubo.points[count].position = glm::vec4(light->position(), 1.f);
        ubo.points[count].color    = glm::vec4(light->color(), light->intensity());
        ubo.points[count].params   = glm::vec4(light->radius(), 0.f, 0.f, 0.f);
        ++count;
    }

    // ── Misc ──────────────────────────────────────────────────────────────────
    ubo.miscFlags.x = count;
    ubo.miscFlags.y = gbufferDebug;
    ubo.iblParams.x = iblIntensity;

    // ── Point shadow — light index 0 is the dedicated shadow caster ──────────
    // Enable when at least one point light exists, is enabled, and casts shadows.
    ubo.pointShadowFlags.x = 0u;
    for (auto& light : m_pointLights) {
        if (light && light->enabled() && light->castsShadow()) {
            ubo.pointShadowFlags.x = 1u;
            break;
        }
    }
}

} // namespace vkgfx
