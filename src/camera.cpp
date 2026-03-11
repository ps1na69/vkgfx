#include "vkgfx/camera.h"
#include "vkgfx/mesh.h"

namespace vkgfx {

// ─── Frustum ─────────────────────────────────────────────────────────────────
void Frustum::extract(const Mat4& vp) {
    const Mat4 m = glm::transpose(vp);
    planes[0] = m[3] + m[0];  // left
    planes[1] = m[3] - m[0];  // right
    planes[2] = m[3] + m[1];  // bottom
    planes[3] = m[3] - m[1];  // top
    planes[4] = m[2];           // near (Vulkan: z in [0,1])
    planes[5] = m[3] - m[2];  // far
    for (auto& p : planes) {
        float len = glm::length(Vec3(p));
        if (len > 0.f) p /= len;
    }
}

bool Frustum::containsAABB(const AABB& aabb) const {
    Vec3 c = aabb.center(), e = aabb.extents();
    for (const auto& p : planes) {
        float r = e.x*std::abs(p.x) + e.y*std::abs(p.y) + e.z*std::abs(p.z);
        if (glm::dot(Vec3(p), c) + p.w + r < 0.f) return false;
    }
    return true;
}

bool Frustum::containsSphere(Vec3 center, float radius) const {
    for (const auto& p : planes)
        if (glm::dot(Vec3(p), center) + p.w < -radius) return false;
    return true;
}

// ─── Camera ──────────────────────────────────────────────────────────────────
Vec3 Camera::forward() const {
    float y = glm::radians(m_yaw), p = glm::radians(m_pitch);
    return glm::normalize(Vec3{std::cos(p)*std::cos(y), std::sin(p), std::cos(p)*std::sin(y)});
}

Vec3 Camera::right() const { return glm::normalize(glm::cross(forward(), m_up)); }

void Camera::update() const {
    if (!m_dirty) return;
    m_target = m_position + forward();
    m_view   = glm::lookAt(m_position, m_target, m_up);
    if (m_projType == ProjectionType::Perspective)
        m_proj = glm::perspective(glm::radians(m_fov), m_aspect, m_near, m_far);
    else {
        float h = m_orthoSize * 0.5f;
        m_proj  = glm::ortho(-h*m_aspect, h*m_aspect, -h, h, m_near, m_far);
    }
    m_proj[1][1] *= -1.f;  // Vulkan Y-flip
    m_viewProj = m_proj * m_view;
    m_frustum.extract(m_viewProj);
    m_dirty = false;
}

FrameUBO Camera::toFrameUBO(float time) const {
    update();
    FrameUBO ubo{};
    ubo.view      = m_view;
    ubo.proj      = m_proj;
    ubo.viewProj  = m_viewProj;
    ubo.invView   = glm::inverse(m_view);
    ubo.invProj   = glm::inverse(m_proj);
    ubo.cameraPos = Vec4(m_position, 1.f);
    ubo.time      = time;
    return ubo;
}

} // namespace vkgfx
