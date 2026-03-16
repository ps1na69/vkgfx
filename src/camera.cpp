// src/camera.cpp
#include <vkgfx/camera.h>
#include <glm/gtc/matrix_transform.hpp>
// GLM_ENABLE_EXPERIMENTAL is set via target_compile_definitions
#include <algorithm>
#include <glm/gtx/euler_angles.hpp>

namespace vkgfx {

// ── Setters ───────────────────────────────────────────────────────────────────

Camera& Camera::setPosition(glm::vec3 p) { m_position = p; return *this; }

Camera& Camera::setRotation(glm::quat q) {
    glm::vec3 euler = glm::eulerAngles(q);
    m_yaw   = glm::degrees(euler.y);
    m_pitch = glm::degrees(euler.x);
    return *this;
}

Camera& Camera::lookAt(glm::vec3 target, glm::vec3 up) {
    glm::vec3 dir = glm::normalize(target - m_position);
    m_pitch = glm::degrees(std::asin(dir.y));
    m_yaw   = glm::degrees(std::atan2(dir.x, -dir.z));
    return *this;
}

Camera& Camera::setFov(float degrees)   { m_fovDeg = degrees; return *this; }
Camera& Camera::setNearFar(float n, float f) { m_near = n; m_far = f; return *this; }
Camera& Camera::setAspect(float a)      { m_aspect = a; return *this; }

// ── FPS movement ──────────────────────────────────────────────────────────────

void Camera::moveForward(float amount) {
    float yr = glm::radians(m_yaw);
    float pr = glm::radians(m_pitch);
    glm::vec3 forward{
        std::sin(yr) * std::cos(pr),
        std::sin(pr),
       -std::cos(yr) * std::cos(pr)
    };
    m_position += glm::normalize(forward) * amount;
}

void Camera::moveRight(float amount) {
    float yr = glm::radians(m_yaw);
    glm::vec3 right{std::cos(yr), 0, std::sin(yr)};
    m_position += right * amount;
}

void Camera::moveUp(float amount) {
    m_position.y += amount;
}

void Camera::rotateYaw(float degrees)   { m_yaw   += degrees; }
void Camera::rotatePitch(float degrees) {
    m_pitch = std::clamp(m_pitch + degrees, -89.f, 89.f);
}

// ── Matrices ──────────────────────────────────────────────────────────────────

glm::mat4 Camera::view() const {
    float yr = glm::radians(m_yaw);
    float pr = glm::radians(m_pitch);
    glm::vec3 forward{
        std::sin(yr) * std::cos(pr),
        std::sin(pr),
       -std::cos(yr) * std::cos(pr)
    };
    return glm::lookAt(m_position, m_position + forward, {0,1,0});
}

glm::mat4 Camera::projection() const {
    glm::mat4 proj = glm::perspective(glm::radians(m_fovDeg), m_aspect, m_near, m_far);
    proj[1][1] *= -1.f;  // Vulkan Y-flip
    return proj;
}

glm::mat4 Camera::viewProj() const {
    return projection() * view();
}

// ── Frustum extraction (Gribb-Hartmann) ──────────────────────────────────────

Frustum Camera::frustum() const {
    glm::mat4 m = viewProj();
    Frustum f;
    // Left, Right, Bottom, Top, Near, Far
    f.planes[0] = m[3] + m[0];
    f.planes[1] = m[3] - m[0];
    f.planes[2] = m[3] + m[1];
    f.planes[3] = m[3] - m[1];
    f.planes[4] = m[3] + m[2];
    f.planes[5] = m[3] - m[2];
    for (auto& p : f.planes)
        p /= glm::length(glm::vec3(p));
    return f;
}

bool Frustum::intersects(const glm::vec3& center, const glm::vec3& extents) const {
    for (auto& p : planes) {
        float d = glm::dot(glm::vec3(p), center) + p.w;
        float r = glm::dot(extents, glm::abs(glm::vec3(p)));
        if (d + r < 0) return false;
    }
    return true;
}

} // namespace vkgfx
