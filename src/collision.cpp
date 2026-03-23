// src/collision.cpp
// Collision detection implementation for vkgfx.
// All narrowphase tests are closed-form analytical — no GJK.
// Reference: Christer Ericson "Real-Time Collision Detection" (2004).

#include <vkgfx/collision.h>
#include <vkgfx/mesh.h>

#include <glm/gtx/norm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <cassert>

namespace vkgfx {

// ─────────────────────────────────────────────────────────────────────────────
// Collider factory & world-space projection
// ─────────────────────────────────────────────────────────────────────────────

Collider Collider::makeSphere(float radius, glm::vec3 localCenter) {
    Collider c;
    c.m_type        = ShapeType::Sphere;
    c.m_localCenter = localCenter;
    c.m_halfSize    = {radius, radius, radius};
    return c;
}

Collider Collider::makeAABB(glm::vec3 halfExt, glm::vec3 localCenter) {
    Collider c;
    c.m_type        = ShapeType::AABB;
    c.m_localCenter = localCenter;
    c.m_halfSize    = halfExt;
    return c;
}

Collider Collider::makeOBB(glm::vec3 halfExt, glm::vec3 localCenter) {
    Collider c;
    c.m_type        = ShapeType::OBB;
    c.m_localCenter = localCenter;
    c.m_halfSize    = halfExt;
    return c;
}

Collider Collider::makeCapsule(float radius, float halfHeight, glm::vec3 localCenter) {
    Collider c;
    c.m_type        = ShapeType::Capsule;
    c.m_localCenter = localCenter;
    c.m_halfSize    = {radius, radius, radius};
    c.m_halfHeight  = halfHeight;
    return c;
}

Collider Collider::fitAABB(const Mesh& mesh) {
    const auto& ab = mesh.aabb();
    Collider c;
    c.m_type        = ShapeType::AABB;
    c.m_localCenter = ab.center();
    c.m_halfSize    = ab.extents();
    return c;
}

Collider Collider::fitSphere(const Mesh& mesh) {
    const auto& ab = mesh.aabb();
    Collider c;
    c.m_type        = ShapeType::Sphere;
    c.m_localCenter = ab.center();
    float r = glm::length(ab.extents());
    c.m_halfSize    = {r, r, r};
    return c;
}

// Apply model matrix to local center
static glm::vec3 applyModel(const glm::mat4& m, const glm::vec3& p) {
    return glm::vec3(m * glm::vec4(p, 1.f));
}

SphereShape Collider::worldSphere(const glm::mat4& model) const {
    SphereShape s;
    s.center = applyModel(model, m_localCenter);
    // Radius scales by the maximum scale component
    glm::vec3 sx(glm::length(glm::vec3(model[0])),
                 glm::length(glm::vec3(model[1])),
                 glm::length(glm::vec3(model[2])));
    s.radius = m_halfSize.x * glm::compMax(sx);
    return s;
}

AABBShape Collider::worldAABB(const glm::mat4& model) const {
    // Transform AABB into world space — produce world AABB
    // Uses the well-known "transform AABB by abs(matrix)" trick (Arvo 1990)
    glm::vec3 wCenter = applyModel(model, m_localCenter);
    glm::vec3 e = m_halfSize;
    glm::vec3 wx{}, wy{}, wz{};
    for (int i = 0; i < 3; ++i) {
        wx[i] = std::abs(model[0][i]) * e.x;
        wy[i] = std::abs(model[1][i]) * e.y;
        wz[i] = std::abs(model[2][i]) * e.z;
    }
    glm::vec3 wExt = wx + wy + wz;
    return {wCenter - wExt, wCenter + wExt};
}

OBBShape Collider::worldOBB(const glm::mat4& model) const {
    OBBShape o;
    o.center   = applyModel(model, m_localCenter);
    // Extract rotation and scale from model matrix
    glm::vec3 sx(glm::length(glm::vec3(model[0])),
                 glm::length(glm::vec3(model[1])),
                 glm::length(glm::vec3(model[2])));
    o.halfSize = m_halfSize * sx;
    // Orientation from the rotation part of the model matrix
    glm::mat3 rot(glm::vec3(model[0]) / sx[0],
                  glm::vec3(model[1]) / sx[1],
                  glm::vec3(model[2]) / sx[2]);
    o.orient = glm::quat_cast(rot);
    return o;
}

CapsuleShape Collider::worldCapsule(const glm::mat4& model) const {
    CapsuleShape c;
    glm::vec3 up    = glm::normalize(glm::vec3(model[1]));
    glm::vec3 wc    = applyModel(model, m_localCenter);
    float     scale = glm::length(glm::vec3(model[1]));
    c.a      = wc - up * (m_halfHeight * scale);
    c.b      = wc + up * (m_halfHeight * scale);
    c.radius = m_halfSize.x * glm::length(glm::vec3(model[0]));
    return c;
}

AABBShape Collider::broadphaseAABB(const glm::mat4& model) const {
    // Conservative world AABB over any shape type
    switch (m_type) {
        case ShapeType::Sphere: {
            auto s = worldSphere(model);
            return {s.center - s.radius, s.center + s.radius};
        }
        case ShapeType::AABB:
            return worldAABB(model);
        case ShapeType::OBB: {
            auto o = worldOBB(model);
            // Compute world AABB from OBB extents projected onto axes
            glm::vec3 wx = glm::abs(o.axis(0)) * o.halfSize.x;
            glm::vec3 wy = glm::abs(o.axis(1)) * o.halfSize.y;
            glm::vec3 wz = glm::abs(o.axis(2)) * o.halfSize.z;
            glm::vec3 e  = wx + wy + wz;
            return {o.center - e, o.center + e};
        }
        case ShapeType::Capsule: {
            auto c = worldCapsule(model);
            glm::vec3 mn = glm::min(c.a, c.b) - c.radius;
            glm::vec3 mx = glm::max(c.a, c.b) + c.radius;
            return {mn, mx};
        }
    }
    return {};
}

// ─────────────────────────────────────────────────────────────────────────────
// Narrowphase: helper math
// ─────────────────────────────────────────────────────────────────────────────

namespace Collisions {

// Closest point on line segment AB to point P
glm::vec3 closestPointOnSegment(const glm::vec3& p,
                                 const glm::vec3& a, const glm::vec3& b) {
    glm::vec3 ab = b - a;
    float     t  = glm::dot(p - a, ab);
    float     d  = glm::dot(ab, ab);
    if (d < 1e-10f) return a;
    t = glm::clamp(t / d, 0.f, 1.f);
    return a + ab * t;
}

// Closest point on AABB to point P
glm::vec3 closestPointOnAABB(const glm::vec3& p, const AABBShape& b) {
    return glm::clamp(p, b.min, b.max);
}

// Closest point on OBB to point P (Ericson p. 132)
glm::vec3 closestPointOnOBB(const glm::vec3& p, const OBBShape& b) {
    glm::vec3 d    = p - b.center;
    glm::vec3 q    = b.center;
    for (int i = 0; i < 3; ++i) {
        glm::vec3 ax  = b.axis(i);
        float     dist = glm::dot(d, ax);
        dist           = glm::clamp(dist, -b.halfSize[i], b.halfSize[i]);
        q             += dist * ax;
    }
    return q;
}

// Closest point on capsule's inner segment to point P
static float closestSegmentToSegment(
    const glm::vec3& p1, const glm::vec3& q1,  // segment 1
    const glm::vec3& p2, const glm::vec3& q2,  // segment 2
    float& s, float& t,
    glm::vec3& c1, glm::vec3& c2)
{
    // Ericson p. 149-150
    glm::vec3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    float a = glm::dot(d1, d1), e = glm::dot(d2, d2);
    float f = glm::dot(d2, r);
    if (a <= 1e-10f && e <= 1e-10f) { s=t=0; c1=p1; c2=p2; return glm::length2(c1-c2); }
    if (a <= 1e-10f) { s=0; t=f/e; t=glm::clamp(t,0.f,1.f); }
    else {
        float c = glm::dot(d1, r);
        if (e <= 1e-10f) { t=0; s=glm::clamp(-c/a,0.f,1.f); }
        else {
            float b = glm::dot(d1, d2), denom = a*e - b*b;
            if (denom != 0.f) s = glm::clamp((b*f - c*e) / denom, 0.f, 1.f);
            else              s = 0.f;
            t = (b*s + f) / e;
            if (t < 0.f)      { t=0; s=glm::clamp(-c/a, 0.f, 1.f); }
            else if (t > 1.f) { t=1; s=glm::clamp((b-c)/a, 0.f, 1.f); }
        }
    }
    c1 = p1 + d1 * s;
    c2 = p2 + d2 * t;
    return glm::length2(c1 - c2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sphere tests
// ─────────────────────────────────────────────────────────────────────────────

ContactInfo sphereVsSphere(const SphereShape& a, const SphereShape& b) {
    glm::vec3 d   = a.center - b.center;
    float     d2  = glm::length2(d);
    float     sum = a.radius + b.radius;
    if (d2 >= sum * sum) return {};
    float dist = std::sqrt(d2);
    ContactInfo c;
    c.hit    = true;
    c.normal = dist < 1e-6f ? glm::vec3(0,1,0) : d / dist;
    c.depth  = sum - dist;
    c.point  = b.center + c.normal * b.radius;
    return c;
}

ContactInfo sphereVsAABB(const SphereShape& s, const AABBShape& b) {
    glm::vec3 closest = closestPointOnAABB(s.center, b);
    glm::vec3 d       = s.center - closest;
    float     d2      = glm::length2(d);
    if (d2 >= s.radius * s.radius) return {};
    float dist = std::sqrt(d2);
    ContactInfo c;
    c.hit    = true;
    c.normal = dist < 1e-6f ? glm::vec3(0,1,0) : d / dist;
    c.depth  = s.radius - dist;
    c.point  = closest;
    return c;
}

ContactInfo sphereVsOBB(const SphereShape& s, const OBBShape& b) {
    glm::vec3 closest = closestPointOnOBB(s.center, b);
    glm::vec3 d       = s.center - closest;
    float     d2      = glm::length2(d);
    if (d2 >= s.radius * s.radius) return {};
    float dist = std::sqrt(d2);
    ContactInfo c;
    c.hit    = true;
    c.normal = dist < 1e-6f ? glm::vec3(0,1,0) : d / dist;
    c.depth  = s.radius - dist;
    c.point  = closest;
    return c;
}

ContactInfo sphereVsCapsule(const SphereShape& s, const CapsuleShape& cap) {
    glm::vec3 closest = closestPointOnSegment(s.center, cap.a, cap.b);
    return sphereVsSphere(s, {closest, cap.radius});
}

// ─────────────────────────────────────────────────────────────────────────────
// AABB tests
// ─────────────────────────────────────────────────────────────────────────────

ContactInfo aabbVsAABB(const AABBShape& a, const AABBShape& b) {
    // SAT: test 3 axes, find axis of minimum penetration
    float minDep = std::numeric_limits<float>::max();
    glm::vec3 minAxis{0,1,0};

    for (int i = 0; i < 3; ++i) {
        float aMin = a.min[i], aMax = a.max[i];
        float bMin = b.min[i], bMax = b.max[i];
        if (aMax < bMin || bMax < aMin) return {};  // gap on this axis
        float d1 = aMax - bMin;
        float d2 = bMax - aMin;
        float dep = std::min(d1, d2);
        if (dep < minDep) {
            minDep = dep;
            minAxis = glm::vec3(i==0, i==1, i==2);
            // Choose sign: push A away from B
            if (a.center()[i] < b.center()[i]) minAxis = -minAxis;
        }
    }
    ContactInfo c;
    c.hit    = true;
    c.normal = minAxis;
    c.depth  = minDep;
    c.point  = b.center() - minAxis * (minDep * 0.5f);
    return c;
}

ContactInfo aabbVsOBB(const AABBShape& a, const OBBShape& b) {
    // Treat AABB as identity OBB and delegate to OBB vs OBB
    OBBShape oa;
    oa.center   = a.center();
    oa.halfSize = a.extents();
    oa.orient   = glm::quat(1,0,0,0);
    return obbVsOBB(oa, b);
}

ContactInfo aabbVsCapsule(const AABBShape& a, const CapsuleShape& cap) {
    // Closest point on capsule segment to AABB center, then sphere-AABB
    glm::vec3 closest = closestPointOnSegment(a.center(), cap.a, cap.b);
    SphereShape s{closest, cap.radius};
    return sphereVsAABB(s, a);
}

// ─────────────────────────────────────────────────────────────────────────────
// OBB vs OBB (SAT with 15 separating axis candidates)
// Ericson p. 101-106
// ─────────────────────────────────────────────────────────────────────────────

ContactInfo obbVsOBB(const OBBShape& A, const OBBShape& B) {
    // Build rotation matrix expressing B in A's coordinate frame
    glm::vec3 aAxes[3] = { A.axis(0), A.axis(1), A.axis(2) };
    glm::vec3 bAxes[3] = { B.axis(0), B.axis(1), B.axis(2) };

    float R[3][3], AbsR[3][3];
    const float eps = 1e-6f;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            R[i][j]    = glm::dot(aAxes[i], bAxes[j]);
            AbsR[i][j] = std::abs(R[i][j]) + eps;
        }

    glm::vec3 t = B.center - A.center;
    // Project T onto A's axes
    glm::vec3 tA = { glm::dot(t, aAxes[0]),
                     glm::dot(t, aAxes[1]),
                     glm::dot(t, aAxes[2]) };

    float minPen = std::numeric_limits<float>::max();
    glm::vec3 minNorm{0,1,0};

    auto testAxis = [&](float ra, float rb, float tx, glm::vec3 ax) {
        float pen = ra + rb - std::abs(tx);
        if (pen < 0) return false;
        if (pen < minPen) { minPen = pen; minNorm = tx < 0 ? -ax : ax; }
        return true;
    };

    const float* ha = &A.halfSize[0];
    const float* hb = &B.halfSize[0];

    // A's 3 face axes
    for (int i = 0; i < 3; ++i)
        if (!testAxis(ha[i],
                      hb[0]*AbsR[i][0]+hb[1]*AbsR[i][1]+hb[2]*AbsR[i][2],
                      tA[i], aAxes[i])) return {};

    // B's 3 face axes
    for (int j = 0; j < 3; ++j) {
        float tb = glm::dot(t, bAxes[j]);
        if (!testAxis(ha[0]*AbsR[0][j]+ha[1]*AbsR[1][j]+ha[2]*AbsR[2][j],
                      hb[j], tb, bAxes[j])) return {};
    }

    // 9 cross-product axes A[i] × B[j]
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int i1 = (i+1)%3, i2 = (i+2)%3;
            int j1 = (j+1)%3, j2 = (j+2)%3;
            glm::vec3 ax = glm::cross(aAxes[i], bAxes[j]);
            float len = glm::length(ax);
            if (len < eps) continue;
            ax /= len;
            float tproj = tA[i1]*R[i2][j] - tA[i2]*R[i1][j];
            float ra = ha[i1]*AbsR[i2][j] + ha[i2]*AbsR[i1][j];
            float rb = hb[j1]*AbsR[i][j2] + hb[j2]*AbsR[i][j1];
            if (!testAxis(ra, rb, tproj, ax)) return {};
        }
    }

    ContactInfo c;
    c.hit    = true;
    c.normal = glm::normalize(minNorm);
    c.depth  = minPen;
    c.point  = B.center - c.normal * (minPen * 0.5f);
    return c;
}

ContactInfo capsuleVsCapsule(const CapsuleShape& a, const CapsuleShape& b) {
    float s, t;
    glm::vec3 c1, c2;
    float d2 = closestSegmentToSegment(a.a, a.b, b.a, b.b, s, t, c1, c2);
    float sum = a.radius + b.radius;
    if (d2 >= sum * sum) return {};
    float dist = std::sqrt(d2);
    ContactInfo c;
    c.hit    = true;
    c.normal = dist < 1e-6f ? glm::vec3(0,1,0) : (c1 - c2) / dist;
    c.depth  = sum - dist;
    c.point  = c2 + c.normal * b.radius;
    return c;
}

// ─────────────────────────────────────────────────────────────────────────────
// Ray casts
// ─────────────────────────────────────────────────────────────────────────────

std::optional<float> rayVsSphere(const Ray& r, const SphereShape& s) {
    glm::vec3 m  = r.origin - s.center;
    float     b  = glm::dot(m, r.direction);
    float     c  = glm::dot(m, m) - s.radius * s.radius;
    if (c > 0.f && b > 0.f) return std::nullopt;
    float disc = b*b - c;
    if (disc < 0.f) return std::nullopt;
    float t = -b - std::sqrt(disc);
    if (t < 0.f) t = 0.f;
    return t;
}

std::optional<float> rayVsAABB(const Ray& r, const AABBShape& b) {
    float tmin = 0.f, tmax = std::numeric_limits<float>::max();
    for (int i = 0; i < 3; ++i) {
        if (std::abs(r.direction[i]) < 1e-8f) {
            if (r.origin[i] < b.min[i] || r.origin[i] > b.max[i]) return std::nullopt;
        } else {
            float inv = 1.f / r.direction[i];
            float t1  = (b.min[i] - r.origin[i]) * inv;
            float t2  = (b.max[i] - r.origin[i]) * inv;
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax) return std::nullopt;
        }
    }
    return tmin;
}

std::optional<float> rayVsOBB(const Ray& r, const OBBShape& b) {
    // Transform ray into OBB local space, then test as AABB
    glm::vec3 d   = r.origin - b.center;
    glm::vec3 ld  = { glm::dot(d, b.axis(0)),
                      glm::dot(d, b.axis(1)),
                      glm::dot(d, b.axis(2)) };
    glm::vec3 ldir= { glm::dot(r.direction, b.axis(0)),
                      glm::dot(r.direction, b.axis(1)),
                      glm::dot(r.direction, b.axis(2)) };
    Ray lr{ ld, ldir };
    AABBShape localBox{ -b.halfSize, b.halfSize };
    return rayVsAABB(lr, localBox);
}

std::optional<float> rayVsCapsule(const Ray& r, const CapsuleShape& cap) {
    // Ray vs infinite cylinder, then clamp to caps (sphere tests)
    glm::vec3 ab  = cap.b - cap.a;
    glm::vec3 ao  = r.origin - cap.a;
    float     len = glm::length(ab);
    if (len < 1e-10f) return rayVsSphere(r, {cap.a, cap.radius});
    glm::vec3 axis = ab / len;

    glm::vec3 rp  = r.direction - axis * glm::dot(r.direction, axis);
    glm::vec3 ap  = ao - axis * glm::dot(ao, axis);
    float     A   = glm::dot(rp, rp);
    float     B   = 2.f * glm::dot(rp, ap);
    float     C   = glm::dot(ap, ap) - cap.radius * cap.radius;

    float tBest = std::numeric_limits<float>::max();
    bool  found = false;

    if (std::abs(A) > 1e-10f) {
        float disc = B*B - 4*A*C;
        if (disc >= 0.f) {
            float sq = std::sqrt(disc);
            for (float sign : {-1.f, 1.f}) {
                float t = (-B + sign * sq) / (2*A);
                if (t < 0) continue;
                // Check within cylinder length
                glm::vec3 p = r.origin + r.direction * t;
                float     h = glm::dot(p - cap.a, axis);
                if (h >= 0.f && h <= len) {
                    if (t < tBest) { tBest = t; found = true; }
                    break;
                }
            }
        }
    }

    // Sphere caps
    for (auto& endPt : {cap.a, cap.b}) {
        if (auto hit = rayVsSphere(r, {endPt, cap.radius}))
            if (*hit < tBest) { tBest = *hit; found = true; }
    }

    return found ? std::make_optional(tBest) : std::nullopt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Universal dispatch
// ─────────────────────────────────────────────────────────────────────────────

ContactInfo test(ShapeType typeA, const glm::mat4& modelA, const Collider& colA,
                 ShapeType typeB, const glm::mat4& modelB, const Collider& colB)
{
    using ST = ShapeType;

    // Sphere vs *
    if (typeA == ST::Sphere && typeB == ST::Sphere)
        return sphereVsSphere(colA.worldSphere(modelA), colB.worldSphere(modelB));
    if (typeA == ST::Sphere && typeB == ST::AABB)
        return sphereVsAABB(colA.worldSphere(modelA), colB.worldAABB(modelB));
    if (typeA == ST::AABB   && typeB == ST::Sphere) {
        auto c = sphereVsAABB(colB.worldSphere(modelB), colA.worldAABB(modelA));
        c.normal = -c.normal; return c;
    }
    if (typeA == ST::Sphere && typeB == ST::OBB)
        return sphereVsOBB(colA.worldSphere(modelA), colB.worldOBB(modelB));
    if (typeA == ST::OBB    && typeB == ST::Sphere) {
        auto c = sphereVsOBB(colB.worldSphere(modelB), colA.worldOBB(modelA));
        c.normal = -c.normal; return c;
    }
    if (typeA == ST::Sphere && typeB == ST::Capsule)
        return sphereVsCapsule(colA.worldSphere(modelA), colB.worldCapsule(modelB));
    if (typeA == ST::Capsule && typeB == ST::Sphere) {
        auto c = sphereVsCapsule(colB.worldSphere(modelB), colA.worldCapsule(modelA));
        c.normal = -c.normal; return c;
    }

    // AABB vs *
    if (typeA == ST::AABB && typeB == ST::AABB)
        return aabbVsAABB(colA.worldAABB(modelA), colB.worldAABB(modelB));
    if (typeA == ST::AABB && typeB == ST::OBB)
        return aabbVsOBB(colA.worldAABB(modelA), colB.worldOBB(modelB));
    if (typeA == ST::OBB  && typeB == ST::AABB) {
        auto c = aabbVsOBB(colB.worldAABB(modelB), colA.worldOBB(modelA));
        c.normal = -c.normal; return c;
    }
    if (typeA == ST::AABB && typeB == ST::Capsule)
        return aabbVsCapsule(colA.worldAABB(modelA), colB.worldCapsule(modelB));
    if (typeA == ST::Capsule && typeB == ST::AABB) {
        auto c = aabbVsCapsule(colB.worldAABB(modelB), colA.worldCapsule(modelA));
        c.normal = -c.normal; return c;
    }

    // OBB vs *
    if (typeA == ST::OBB && typeB == ST::OBB)
        return obbVsOBB(colA.worldOBB(modelA), colB.worldOBB(modelB));

    // Capsule vs capsule
    if (typeA == ST::Capsule && typeB == ST::Capsule)
        return capsuleVsCapsule(colA.worldCapsule(modelA), colB.worldCapsule(modelB));

    return {};
}

} // namespace Collisions

// ─────────────────────────────────────────────────────────────────────────────
// CollisionWorld
// ─────────────────────────────────────────────────────────────────────────────

CollisionWorld::CollisionWorld(float cellSize) : m_cellSize(cellSize) {}

CollisionObject& CollisionWorld::add(Mesh* mesh, Collider col, bool isStatic) {
    m_objects.push_back({mesh, std::move(col), {}, isStatic});
    return m_objects.back();
}

void CollisionWorld::remove(Mesh* mesh) {
    m_objects.erase(
        std::remove_if(m_objects.begin(), m_objects.end(),
                       [mesh](const CollisionObject& o){ return o.mesh == mesh; }),
        m_objects.end());
}

void CollisionWorld::clear() { m_objects.clear(); }

// ── Broadphase: uniform spatial grid ──────────────────────────────────────────

void CollisionWorld::rebuildGrid() {
    m_grid.clear();
    for (int i = 0; i < (int)m_objects.size(); ++i) {
        const auto& obj = m_objects[i];
        if (!obj.mesh) continue;
        AABBShape bp = obj.collider.broadphaseAABB(obj.mesh->modelMatrix());

        int x0 = (int)std::floor(bp.min.x / m_cellSize);
        int y0 = (int)std::floor(bp.min.y / m_cellSize);
        int z0 = (int)std::floor(bp.min.z / m_cellSize);
        int x1 = (int)std::floor(bp.max.x / m_cellSize);
        int y1 = (int)std::floor(bp.max.y / m_cellSize);
        int z1 = (int)std::floor(bp.max.z / m_cellSize);

        for (int x = x0; x <= x1; ++x)
            for (int y = y0; y <= y1; ++y)
                for (int z = z0; z <= z1; ++z)
                    m_grid[{x,y,z}].push_back(i);
    }
}

std::vector<std::pair<int,int>> CollisionWorld::broadphase() const {
    // Collect candidate pairs — deduplicate with a set of sorted index pairs
    struct PairHash {
        size_t operator()(std::pair<int,int> p) const {
            return std::hash<long long>{}((long long)p.first << 32 | p.second);
        }
    };
    std::unordered_set<std::pair<int,int>, PairHash> seen;
    std::vector<std::pair<int,int>> pairs;

    for (auto& [cell, indices] : m_grid) {
        for (size_t i = 0; i < indices.size(); ++i)
            for (size_t j = i + 1; j < indices.size(); ++j) {
                auto p = std::make_pair(
                    std::min(indices[i], indices[j]),
                    std::max(indices[i], indices[j]));
                if (seen.insert(p).second)
                    pairs.push_back(p);
            }
    }
    return pairs;
}

// ── MTV response ──────────────────────────────────────────────────────────────

void CollisionWorld::applyMTV(CollisionObject& a, CollisionObject& b,
                               const ContactInfo& c) {
    if (!a.mesh || !b.mesh) return;
    if (a.collider.isTrigger || b.collider.isTrigger) return;

    glm::vec3 push = c.normal * c.depth;
    // Extract world position from column 3 of the model matrix
    auto posOf = [](Mesh* m) { return glm::vec3(m->modelMatrix()[3]); };
    if (!a.isStatic && !b.isStatic) {
        a.mesh->setPosition(posOf(a.mesh) + push * 0.5f);
        b.mesh->setPosition(posOf(b.mesh) - push * 0.5f);
    } else if (!a.isStatic) {
        a.mesh->setPosition(posOf(a.mesh) + push);
    } else if (!b.isStatic) {
        b.mesh->setPosition(posOf(b.mesh) - push);
    }
}

// ── update ────────────────────────────────────────────────────────────────────

void CollisionWorld::update(bool applyResponse) {
    rebuildGrid();
    auto pairs = broadphase();

    for (auto [ai, bi] : pairs) {
        auto& A = m_objects[ai];
        auto& B = m_objects[bi];
        if (!A.mesh || !B.mesh) continue;

        // Layer filter
        if (!(A.collider.layer & B.collider.mask) &&
            !(B.collider.layer & A.collider.mask)) continue;

        // Skip two static objects — they never move
        if (A.isStatic && B.isStatic) continue;

        auto contact = Collisions::test(
            A.collider.type(), A.mesh->modelMatrix(), A.collider,
            B.collider.type(), B.mesh->modelMatrix(), B.collider);

        if (!contact.hit) continue;

        if (m_onContact) m_onContact({&A, &B, contact});
        if (applyResponse) applyMTV(A, B, contact);
    }
}

// ── Ray cast ──────────────────────────────────────────────────────────────────

RayHit CollisionWorld::castRay(const Ray& ray, float maxDist) const {
    RayHit best;
    best.t = maxDist;

    for (auto& obj : m_objects) {
        if (!obj.mesh) continue;
        glm::mat4 model = obj.mesh->modelMatrix();

        std::optional<float> t;
        switch (obj.collider.type()) {
            case ShapeType::Sphere:
                t = Collisions::rayVsSphere(ray, obj.collider.worldSphere(model)); break;
            case ShapeType::AABB:
                t = Collisions::rayVsAABB  (ray, obj.collider.worldAABB(model));   break;
            case ShapeType::OBB:
                t = Collisions::rayVsOBB   (ray, obj.collider.worldOBB(model));    break;
            case ShapeType::Capsule:
                t = Collisions::rayVsCapsule(ray, obj.collider.worldCapsule(model)); break;
        }

        if (t && *t < best.t) {
            best.t     = *t;
            best.point = ray.at(*t);
            best.mesh  = obj.mesh;
            best.hit   = true;
            // Approximate normal: sphere/AABB accurate, others approximate
            if (obj.collider.type() == ShapeType::Sphere) {
                auto s = obj.collider.worldSphere(model);
                best.normal = glm::normalize(best.point - s.center);
            }
        }
    }
    return best;
}

RayHit CollisionWorld::castRay(glm::vec3 origin, glm::vec3 dir, float maxDist) const {
    float len = glm::length(dir);
    if (len < 1e-10f) return {};
    return castRay(Ray{origin, dir / len}, maxDist);
}

// ── queryRadius ───────────────────────────────────────────────────────────────

std::vector<CollisionObject*> CollisionWorld::queryRadius(glm::vec3 center, float radius) const {
    std::vector<CollisionObject*> result;
    float r2 = radius * radius;
    for (auto& obj : m_objects) {
        if (!obj.mesh) continue;
        auto bp = obj.collider.broadphaseAABB(obj.mesh->modelMatrix());
        glm::vec3 closest = glm::clamp(center, bp.min, bp.max);
        if (glm::length2(center - closest) <= r2)
            result.push_back(const_cast<CollisionObject*>(&obj));
    }
    return result;
}

CollisionObject* CollisionWorld::findByTag(const std::string& tag) {
    for (auto& obj : m_objects)
        if (obj.tag == tag) return &obj;
    return nullptr;
}

} // namespace vkgfx
