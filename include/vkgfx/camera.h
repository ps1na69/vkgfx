#pragma once
#include "types.h"

namespace vkgfx {

struct Frustum {
    Vec4 planes[6]; // normals + offset

    void extract(const Mat4& viewProj);
    [[nodiscard]] bool containsAABB(const struct AABB& aabb) const;
    [[nodiscard]] bool containsSphere(Vec3 center, float radius) const;
};

class Camera {
public:
    enum class ProjectionType { Perspective, Orthographic };

    Camera() = default;

    Camera& setPosition(Vec3 p)    { m_position = p; m_dirty = true; return *this; }
    Camera& setUp(Vec3 u)          { m_up = u;        m_dirty = true; return *this; }

    Camera& setYaw(float deg)      { m_yaw   = deg; m_dirty = true; return *this; }
    Camera& setPitch(float deg)    { m_pitch = deg; m_dirty = true; return *this; }

    void translate(Vec3 delta)     { m_position += delta; m_dirty = true; }
    void rotate(float dYaw, float dPitch) {
        m_yaw   += dYaw;
        m_pitch  = glm::clamp(m_pitch + dPitch, -89.f, 89.f);
        m_dirty  = true;
    }

    Camera& setFov(float degrees)     { m_fov   = degrees; m_dirty = true; return *this; }
    Camera& setNear(float n)           { m_near  = n;       m_dirty = true; return *this; }
    Camera& setFar(float f)            { m_far   = f;       m_dirty = true; return *this; }
    Camera& setAspect(float a)         { m_aspect = a;      m_dirty = true; return *this; }
    Camera& setOrthoSize(float s)      { m_orthoSize = s;   m_dirty = true; return *this; }
    Camera& setProjection(ProjectionType t) { m_projType = t; m_dirty = true; return *this; }

    [[nodiscard]] Vec3  position() const { return m_position; }
    [[nodiscard]] Vec3  forward()  const;
    [[nodiscard]] Vec3  right()    const;
    [[nodiscard]] Vec3  up()       const { return m_up; }
    [[nodiscard]] float fov()      const { return m_fov; }
    [[nodiscard]] float nearPlane() const { return m_near; }
    [[nodiscard]] float farPlane()  const { return m_far; }
    [[nodiscard]] float aspect()    const { return m_aspect; }

    [[nodiscard]] const Mat4& viewMatrix()     const { update(); return m_view; }
    [[nodiscard]] const Mat4& projMatrix()     const { update(); return m_proj; }
    [[nodiscard]] const Mat4& viewProjMatrix() const { update(); return m_viewProj; }
    [[nodiscard]] const Frustum& frustum()     const { update(); return m_frustum; }
    [[nodiscard]] CameraUBO toUBO(float time = 0.f) const;

private:
    void update() const;

    Vec3  m_position {0.f, 0.f,  -5.f};
    Vec3  m_up       {0.f, 1.f,   0.f};
    float m_yaw      = -120.f;
    float m_pitch    = 0.f;
    float m_fov      = 90.f;
    float m_near     = 0.1f;
    float m_far      = 1000.f;
    float m_aspect   = 16.f / 9.f;
    float m_orthoSize = 10.f;
    ProjectionType m_projType = ProjectionType::Perspective;

    mutable Mat4    m_view{1.f};
    mutable Mat4    m_proj{1.f};
    mutable Mat4    m_viewProj{1.f};
    mutable Vec3    m_target{0.f, 0.f, 0.f};
    mutable Frustum m_frustum;
    mutable bool    m_dirty = true;
};

} // namespace vkgfx
