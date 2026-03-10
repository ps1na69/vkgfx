#pragma once
#include "camera.h"
#include "mesh.h"
#include "thread_pool.h"  // for ThreadPool* parameter
#include <variant>

namespace vkgfx {

class Light {
public:
    virtual ~Light() = default;
    virtual LightType type() const = 0;
    virtual LightData toLightData() const = 0;

    Light& setPosition(Vec3 p)  { m_position = p;   return *this; }
    Light& setColor(Vec3 c)     { m_color = c;       return *this; }
    Light& setIntensity(float i){ m_intensity = i;   return *this; }
    Light& setCastShadows(bool b){ m_castShadows = b; return *this; }

    [[nodiscard]] Vec3  position()    const { return m_position; }
    [[nodiscard]] Vec3  color()       const { return m_color; }
    [[nodiscard]] float intensity()   const { return m_intensity; }
    [[nodiscard]] bool  castShadows() const { return m_castShadows; }

protected:
    Vec3  m_position   {0.f, 3.f, 0.f};
    Vec3  m_color      {1.f, 1.f, 1.f};
    float m_intensity  = 1.f;
    bool  m_castShadows = false;
};

class PointLight : public Light {
public:
    PointLight() = default;
    PointLight& setRange(float r)  { m_range = r; return *this; }
    [[nodiscard]] float range() const { return m_range; }

    LightType type() const override { return LightType::Point; }
    LightData toLightData() const override {
        LightData d;
        d.position  = {m_position, 0.f};
        d.color     = {m_color, m_intensity};
        d.direction = {0.f, 0.f, 0.f, 0.f};
        d.params    = {0.f, m_range, m_castShadows ? 1.f : 0.f, -1.0f};
        return d;
    }

private:
    float m_range = 30.f;
};

class DirectionalLight : public Light {
public:
    DirectionalLight() { m_position = {0.f, 1.f, 0.f}; }
    DirectionalLight& setDirection(Vec3 d) { m_direction = glm::normalize(d); return *this; }
    [[nodiscard]] Vec3 direction() const { return m_direction; }

    LightType type() const override { return LightType::Directional; }
    LightData toLightData() const override {
        LightData d;
        d.position  = {m_position, 1.f};
        d.color     = {m_color, m_intensity};
        d.direction = {m_direction, 0.f};
        d.params    = {0.f, 0.f, m_castShadows ? 1.f : 0.f, -1.0f};
        return d;
    }

private:
    Vec3 m_direction{0.f, -1.f, 0.f};
};

class SpotLight : public Light {
public:
    SpotLight() = default;
    SpotLight& setDirection(Vec3 d)     { m_direction = glm::normalize(d); return *this; }
    SpotLight& setInnerCone(float deg)  { m_innerCone = deg; return *this; }
    SpotLight& setOuterCone(float deg)  { m_outerCone = deg; return *this; }
    SpotLight& setRange(float r)        { m_range = r; return *this; }

    LightType type() const override { return LightType::Spot; }
    LightData toLightData() const override {
        LightData d;
        d.position  = {m_position, 2.f};
        d.color     = {m_color, m_intensity};
        d.direction = {m_direction, glm::radians(m_innerCone)};
        d.params    = {glm::radians(m_outerCone), m_range, m_castShadows ? 1.f : 0.f, -1.0f};
        return d;
    }

private:
    Vec3  m_direction{0.f, -1.f, 0.f};
    float m_innerCone = 20.f;
    float m_outerCone = 35.f;
    float m_range     = 20.f;
};

//Scene
class Scene {
public:
    explicit Scene(Camera* camera = nullptr);
    ~Scene() = default;

    Scene& add(std::shared_ptr<Mesh> mesh);
    Scene& add(std::shared_ptr<Light> light);
    Scene& setCamera(Camera* camera);
    Scene& setAmbient(Vec3 color, float intensity = 0.1f) {
        m_ambientColor = color; m_ambientIntensity = intensity; return *this;
    }

    [[nodiscard]] Camera*       camera()     const { return m_camera; }
    [[nodiscard]] const std::vector<std::shared_ptr<Mesh>>&  meshes()  const { return m_meshes; }
    [[nodiscard]] const std::vector<std::shared_ptr<Light>>& lights()  const { return m_lights; }
    [[nodiscard]] SceneUBO buildSceneUBO() const;

    void remove(std::shared_ptr<Mesh> mesh);
    void remove(std::shared_ptr<Light> light);
    void clear();

    // FIX: Accepts an optional ThreadPool* for parallel frustum culling.
    // Pass nullptr (default) for single-threaded culling.
    // The Renderer passes its own ThreadPool automatically.
    [[nodiscard]] std::vector<Mesh*> visibleMeshes(ThreadPool* pool = nullptr) const;

private:
    Camera*                              m_camera = nullptr;
    std::vector<std::shared_ptr<Mesh>>   m_meshes;
    std::vector<std::shared_ptr<Light>>  m_lights;
    Vec3                                 m_ambientColor{0.1f, 0.1f, 0.15f};
    float                                m_ambientIntensity = 0.05f;
};

} // namespace vkgfx
