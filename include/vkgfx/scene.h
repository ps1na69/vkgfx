#pragma once
// scene.h — scene graph for the deferred renderer.
//
// Scene owns meshes and lights.  The renderer pulls data from Scene each frame:
//   - meshes → geometry pass draw calls
//   - lights → light SSBO update (max MAX_LIGHTS dynamic lights)
//
// No shadow UBOs, no forward SceneUBO. Lights are submitted as GpuLight array.

#include "camera.h"
#include "mesh.h"
#include "thread_pool.h"

namespace vkgfx {

// ─── Light wrappers (CPU side) ────────────────────────────────────────────────
class Light {
public:
    virtual ~Light() = default;
    virtual LightType type()          const = 0;
    virtual GpuLight  toGpuLight()    const = 0;

    Light& setPosition (Vec3 p) { m_position  = p; return *this; }
    Light& setColor    (Vec3 c) { m_color     = c; return *this; }
    Light& setIntensity(float i){ m_intensity  = i; return *this; }

    [[nodiscard]] Vec3  position()  const { return m_position; }
    [[nodiscard]] Vec3  color()     const { return m_color; }
    [[nodiscard]] float intensity() const { return m_intensity; }

protected:
    Vec3  m_position  {0.f, 3.f, 0.f};
    Vec3  m_color     {1.f, 1.f, 1.f};
    float m_intensity = 1.f;
};

class PointLight : public Light {
public:
    PointLight& setRange(float r) { m_range = r; return *this; }
    LightType type() const override { return LightType::Point; }
    GpuLight toGpuLight() const override {
        GpuLight g{};
        g.position = {m_position, m_range};
        g.color    = {m_color, m_intensity};
        g.type     = 0;
        return g;
    }
private:
    float m_range = 30.f;
};

class DirectionalLight : public Light {
public:
    DirectionalLight& setDirection(Vec3 d) { m_direction = glm::normalize(d); return *this; }
    LightType type() const override { return LightType::Directional; }
    GpuLight toGpuLight() const override {
        GpuLight g{};
        g.position  = {m_position, 0.f};
        g.direction = {m_direction, 0.f};
        g.color     = {m_color, m_intensity};
        g.type      = 1;
        return g;
    }
private:
    Vec3 m_direction{0.f, -1.f, 0.f};
};

class SpotLight : public Light {
public:
    SpotLight& setDirection  (Vec3 d)   { m_direction = glm::normalize(d); return *this; }
    SpotLight& setInnerCone  (float d)  { m_innerDeg  = d; return *this; }
    SpotLight& setOuterCone  (float d)  { m_outerDeg  = d; return *this; }
    SpotLight& setRange      (float r)  { m_range     = r; return *this; }
    LightType type() const override { return LightType::Spot; }
    GpuLight toGpuLight() const override {
        GpuLight g{};
        g.position      = {m_position, m_range};
        g.direction     = {m_direction, std::cos(glm::radians(m_outerDeg))};
        g.color         = {m_color, m_intensity};
        g.type          = 2;
        g.innerAngleCos = std::cos(glm::radians(m_innerDeg));
        return g;
    }
private:
    Vec3  m_direction{0.f,-1.f,0.f};
    float m_innerDeg  = 20.f;
    float m_outerDeg  = 35.f;
    float m_range     = 20.f;
};

// ─── Scene ────────────────────────────────────────────────────────────────────
class Scene {
public:
    explicit Scene(Camera* camera = nullptr);

    Scene& add(std::shared_ptr<Mesh>  mesh);
    Scene& add(std::shared_ptr<Light> light);
    Scene& setCamera(Camera* camera) { m_camera = camera; return *this; }
    Scene& setAmbient(Vec3 color, float intensity = 0.05f) {
        m_ambientColor = color; m_ambientIntensity = intensity; return *this;
    }

    [[nodiscard]] Camera* camera()   const { return m_camera; }
    [[nodiscard]] const std::vector<std::shared_ptr<Mesh>>&  meshes() const { return m_meshes; }
    [[nodiscard]] const std::vector<std::shared_ptr<Light>>& lights() const { return m_lights; }
    [[nodiscard]] Vec3  ambientColor()     const { return m_ambientColor; }
    [[nodiscard]] float ambientIntensity() const { return m_ambientIntensity; }

    void remove(std::shared_ptr<Mesh>  mesh);
    void remove(std::shared_ptr<Light> light);
    void clear();

    // Fill a LightSSBO from the scene's light list. Returns light count.
    [[nodiscard]] uint32_t buildLightBuffer(LightSSBO& out) const;

    // CPU-side frustum culling; pass nullptr for single-threaded.
    [[nodiscard]] std::vector<Mesh*> visibleMeshes(ThreadPool* pool = nullptr) const;

private:
    Camera*                             m_camera          = nullptr;
    std::vector<std::shared_ptr<Mesh>>  m_meshes;
    std::vector<std::shared_ptr<Light>> m_lights;
    Vec3  m_ambientColor     {0.1f, 0.1f, 0.15f};
    float m_ambientIntensity = 0.05f;
};

} // namespace vkgfx
