#pragma once
// include/vkgfx/camera.h

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <array>

namespace vkgfx {

struct Frustum {
    // 6 planes in Ax+By+Cz+D form
    std::array<glm::vec4, 6> planes;
    /// Returns true if the AABB (center, extents) is at least partially inside.
    bool intersects(const glm::vec3& center, const glm::vec3& extents) const;
};

class Camera {
public:
    Camera() = default;

    Camera& setPosition(glm::vec3 p);
    Camera& setRotation(glm::quat q);
    Camera& lookAt(glm::vec3 target, glm::vec3 up = {0,1,0});
    Camera& setFov(float degrees);
    Camera& setNearFar(float near, float far);
    Camera& setAspect(float aspect);

    // First-person input helpers (call from example main loop)
    void moveForward(float amount);
    void moveRight(float amount);
    void moveUp(float amount);
    void rotateYaw(float degrees);
    void rotatePitch(float degrees);

    [[nodiscard]] glm::mat4   view()       const;
    [[nodiscard]] glm::mat4   projection() const;
    [[nodiscard]] glm::mat4   viewProj()   const;
    [[nodiscard]] glm::vec3   position()   const { return m_position; }
    [[nodiscard]] Frustum     frustum()    const;

private:
    glm::vec3 m_position  = {0, 0, -3};
    float     m_yaw       = 0.f;
    float     m_pitch     = 0.f;
    float     m_fovDeg    = 60.f;
    float     m_near      = 0.1f;
    float     m_far       = 1000.f;
    float     m_aspect    = 16.f / 9.f;
};

} // namespace vkgfx
