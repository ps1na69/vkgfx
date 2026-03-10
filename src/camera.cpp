#include "vkgfx/camera.h"
#include "vkgfx/mesh.h"

namespace vkgfx {

// ─── Frustum ──────────────────────────────────────────────────────────────────
void Frustum::extract(const Mat4& vp) {
    // Gribb-Hartmann method: plane i is a linear combination of the ROWS of the
    // clip matrix.  GLM stores matrices column-major, so vp[i] is the i-th
    // *column*, not row.  Transposing first makes m[i] correspond to row i,
    // which is what the formulas below expect.
    const Mat4 m = glm::transpose(vp);
    planes[0] = m[3] + m[0]; // left
    planes[1] = m[3] - m[0]; // right
    planes[2] = m[3] + m[1]; // bottom
    planes[3] = m[3] - m[1]; // top
    planes[4] = m[2];          // near  (Vulkan: z in [0,1])
    planes[5] = m[3] - m[2]; // far
    for (auto& p : planes) {
        float len = glm::length(Vec3(p));
        if (len > 0.f) p /= len;
    }
}

bool Frustum::containsAABB(const AABB& aabb) const {
    Vec3 c = aabb.center();
    Vec3 e = aabb.extents();
    for (const auto& p : planes) {
        float r = e.x*std::abs(p.x) + e.y*std::abs(p.y) + e.z*std::abs(p.z);
        if (glm::dot(Vec3(p), c) + p.w + r < 0.f) return false; // outside
    }
    return true;
}

bool Frustum::containsSphere(Vec3 center, float radius) const {
    for (const auto& p : planes)
        if (glm::dot(Vec3(p), center) + p.w < -radius) return false;
    return true;
}

// ─── Camera ───────────────────────────────────────────────────────────────────
Vec3 Camera::forward() const {
    float y = glm::radians(m_yaw), p = glm::radians(m_pitch);
    return glm::normalize(Vec3{
        std::cos(p) * std::cos(y),
        std::sin(p),
        std::cos(p) * std::sin(y)
    });
}

Vec3 Camera::right() const {
    return glm::normalize(glm::cross(forward(), m_up));
}

void Camera::update() const {
    if (!m_dirty) return;
    // View from euler angles
    m_target = m_position + forward();
    m_view   = glm::lookAt(m_position, m_target, m_up);

    if (m_projType == ProjectionType::Perspective) {
        m_proj = glm::perspective(glm::radians(m_fov), m_aspect, m_near, m_far);
    } else {
        float h = m_orthoSize * 0.5f;
        m_proj = glm::ortho(-h * m_aspect, h * m_aspect, -h, h, m_near, m_far);
    }
    m_proj[1][1] *= -1.f; // Vulkan Y-flip

    m_viewProj = m_proj * m_view;
    m_frustum.extract(m_viewProj);
    m_dirty = false;
}

CameraUBO Camera::toUBO(float time) const {
    CameraUBO ubo;
    ubo.view     = viewMatrix();
    ubo.proj     = projMatrix();
    ubo.viewProj = viewProjMatrix();
    ubo.position = { m_position, m_near };
    ubo.params   = { m_far, m_fov, m_aspect, time };
    return ubo;
}

} // namespace vkgfx
