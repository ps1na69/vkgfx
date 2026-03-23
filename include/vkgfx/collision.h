#pragma once
// include/vkgfx/collision.h
//
// Collision detection for vkgfx.
//
// Design:
//   - Shape types: Sphere, AABB, OBB, Capsule, Ray
//   - Broadphase : uniform spatial grid (fast for <1000 objects, zero alloc steady-state)
//   - Narrowphase: closed-form analytical tests (no GJK — unnecessary for a render engine)
//   - Response   : ContactInfo carries MTV (push A out of B along normal by depth)
//   - Integration: Mesh::setCollider(ColliderShape) / CollisionWorld alongside Scene
//
// All math follows Christer Ericson "Real-Time Collision Detection" (2004).

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/norm.hpp>

#include <vector>
#include <functional>
#include <optional>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>

namespace vkgfx {

class Mesh;

// ─────────────────────────────────────────────────────────────────────────────
// Shape descriptors
// ─────────────────────────────────────────────────────────────────────────────

enum class ShapeType { Sphere, AABB, OBB, Capsule };

// Sphere: centre + radius (world space)
struct SphereShape {
    glm::vec3 center{0.f};
    float     radius = 1.f;
};

// Axis-Aligned Bounding Box (world space)
struct AABBShape {
    glm::vec3 min{-0.5f};
    glm::vec3 max{ 0.5f};

    [[nodiscard]] glm::vec3 center()  const { return (min + max) * 0.5f; }
    [[nodiscard]] glm::vec3 extents() const { return (max - min) * 0.5f; }
};

// Oriented Bounding Box: centre, half-extents in local space, orientation
struct OBBShape {
    glm::vec3 center   {0.f};
    glm::vec3 halfSize {0.5f};      // half-extents in local space
    glm::quat orient   {1,0,0,0};   // world-space orientation

    // Returns the 3 normalised axis vectors (columns of rotation matrix)
    [[nodiscard]] glm::vec3 axis(int i) const {
        return glm::normalize(glm::rotate(orient, glm::vec3(i==0,i==1,i==2)));
    }
};

// Capsule: two endpoints + radius (world space)
struct CapsuleShape {
    glm::vec3 a{0.f, -0.5f, 0.f};
    glm::vec3 b{0.f,  0.5f, 0.f};
    float     radius = 0.5f;
};

// Ray: origin + normalised direction
struct Ray {
    glm::vec3 origin   {0.f};
    glm::vec3 direction{0.f, 0.f, 1.f}; // must be normalised

    [[nodiscard]] glm::vec3 at(float t) const { return origin + direction * t; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Collision result
// ─────────────────────────────────────────────────────────────────────────────

struct ContactInfo {
    bool      hit     = false;
    glm::vec3 normal  {0.f};   // points from B toward A (push A in this direction)
    float     depth   = 0.f;   // penetration depth (positive = overlapping)
    glm::vec3 point   {0.f};   // world-space contact point (on surface of B)

    explicit operator bool() const { return hit; }
};

struct RayHit {
    bool      hit     = false;
    float     t       = 0.f;   // distance along ray
    glm::vec3 point   {0.f};
    glm::vec3 normal  {0.f};
    Mesh*     mesh    = nullptr;

    explicit operator bool() const { return hit; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Collider component — attach to a Mesh
// ─────────────────────────────────────────────────────────────────────────────

class Collider {
public:
    // Collision layer / mask — only collide if (layerA & maskB) != 0
    uint32_t layer = 0xFFFFFFFF;
    uint32_t mask  = 0xFFFFFFFF;
    bool     isTrigger = false;  // trigger: fires callback but no MTV response

    // Build different shape types
    static Collider makeSphere (float radius,     glm::vec3 localCenter = {0,0,0});
    static Collider makeAABB   (glm::vec3 halfExt, glm::vec3 localCenter = {0,0,0});
    static Collider makeOBB    (glm::vec3 halfExt, glm::vec3 localCenter = {0,0,0});
    static Collider makeCapsule(float radius, float halfHeight, glm::vec3 localCenter = {0,0,0});
    // Automatically fit AABB to a Mesh's local AABB
    static Collider fitAABB    (const Mesh& mesh);
    static Collider fitSphere  (const Mesh& mesh);

    [[nodiscard]] ShapeType type() const { return m_type; }

    // Compute the world-space shape by applying the mesh's model matrix.
    // These are the functions called by CollisionWorld during narrowphase.
    [[nodiscard]] SphereShape  worldSphere (const glm::mat4& model) const;
    [[nodiscard]] AABBShape    worldAABB   (const glm::mat4& model) const;
    [[nodiscard]] OBBShape     worldOBB    (const glm::mat4& model) const;
    [[nodiscard]] CapsuleShape worldCapsule(const glm::mat4& model) const;

    // World-space AABB (conservative bound over any shape) — used for broadphase
    [[nodiscard]] AABBShape broadphaseAABB(const glm::mat4& model) const;

private:
    ShapeType m_type = ShapeType::Sphere;
    glm::vec3 m_localCenter{0.f};
    glm::vec3 m_halfSize   {0.5f};  // for AABB/OBB: half-extents; for sphere: x=radius
    float     m_halfHeight = 0.f;   // for capsule
};

// ─────────────────────────────────────────────────────────────────────────────
// Narrowphase — all collision test functions
// ─────────────────────────────────────────────────────────────────────────────

namespace Collisions {

// ── Sphere tests ──────────────────────────────────────────────────────────────
ContactInfo sphereVsSphere(const SphereShape& a, const SphereShape& b);
ContactInfo sphereVsAABB  (const SphereShape& s, const AABBShape&   b);
ContactInfo sphereVsOBB   (const SphereShape& s, const OBBShape&    b);
ContactInfo sphereVsCapsule(const SphereShape& s, const CapsuleShape& c);

// ── AABB tests ────────────────────────────────────────────────────────────────
ContactInfo aabbVsAABB  (const AABBShape& a, const AABBShape& b);
ContactInfo aabbVsOBB   (const AABBShape& a, const OBBShape&  b);
ContactInfo aabbVsCapsule(const AABBShape& a, const CapsuleShape& c);

// ── OBB tests ─────────────────────────────────────────────────────────────────
ContactInfo obbVsOBB    (const OBBShape& a,  const OBBShape& b);

// ── Capsule tests ─────────────────────────────────────────────────────────────
ContactInfo capsuleVsCapsule(const CapsuleShape& a, const CapsuleShape& b);

// ── Ray tests ─────────────────────────────────────────────────────────────────
std::optional<float> rayVsSphere (const Ray&, const SphereShape&);
std::optional<float> rayVsAABB   (const Ray&, const AABBShape&);
std::optional<float> rayVsOBB    (const Ray&, const OBBShape&);
std::optional<float> rayVsCapsule(const Ray&, const CapsuleShape&);

// ── Closest point helpers (public for gameplay use) ───────────────────────────
glm::vec3 closestPointOnSegment(const glm::vec3& p,
                                 const glm::vec3& a, const glm::vec3& b);
glm::vec3 closestPointOnAABB   (const glm::vec3& p, const AABBShape& b);
glm::vec3 closestPointOnOBB    (const glm::vec3& p, const OBBShape& b);

// Dispatch: test any two colliders in world space
ContactInfo test(ShapeType typeA, const glm::mat4& modelA, const Collider& colA,
                 ShapeType typeB, const glm::mat4& modelB, const Collider& colB);

} // namespace Collisions

// ─────────────────────────────────────────────────────────────────────────────
// CollisionObject — a Mesh with a Collider attached
// ─────────────────────────────────────────────────────────────────────────────

struct CollisionObject {
    Mesh*     mesh     = nullptr;   // non-owning (Scene owns meshes)
    Collider  collider;
    glm::vec3 velocity {0.f};       // optional — used for response integration
    bool      isStatic = false;     // static objects are never moved by response
    std::string tag;                // gameplay tag ("player", "wall", "pickup")
};

// ─────────────────────────────────────────────────────────────────────────────
// Collision event — passed to callbacks
// ─────────────────────────────────────────────────────────────────────────────

struct CollisionEvent {
    CollisionObject* objectA = nullptr;
    CollisionObject* objectB = nullptr;
    ContactInfo      contact;
};

using CollisionCallback = std::function<void(const CollisionEvent&)>;

// ─────────────────────────────────────────────────────────────────────────────
// CollisionWorld — broadphase + narrowphase over registered objects
// ─────────────────────────────────────────────────────────────────────────────

class CollisionWorld {
public:
    explicit CollisionWorld(float cellSize = 4.f);

    // ── Registration ──────────────────────────────────────────────────────────
    CollisionObject& add(Mesh* mesh, Collider collider, bool isStatic = false);
    void             remove(Mesh* mesh);
    void             clear();
    [[nodiscard]] std::vector<CollisionObject>& objects() { return m_objects; }

    // ── Per-frame update ──────────────────────────────────────────────────────
    // Runs broadphase + narrowphase. Fires callbacks and optionally applies MTV.
    // Call once per frame after updating mesh positions.
    void update(bool applyResponse = true);

    // ── Callbacks ─────────────────────────────────────────────────────────────
    // onContact fires for every overlapping pair (including triggers).
    void setOnContact(CollisionCallback cb) { m_onContact = std::move(cb); }

    // ── Ray casting ───────────────────────────────────────────────────────────
    // Returns the closest hit among all registered objects.
    [[nodiscard]] RayHit castRay(const Ray& ray, float maxDist = 1e9f) const;

    // Convenience: cast ray from world position along direction
    [[nodiscard]] RayHit castRay(glm::vec3 origin, glm::vec3 direction,
                                  float maxDist = 1e9f) const;

    // ── Queries ───────────────────────────────────────────────────────────────
    // Find all objects whose AABB overlaps the query sphere
    [[nodiscard]] std::vector<CollisionObject*>
        queryRadius(glm::vec3 center, float radius) const;

    // Find the first object with a given tag
    [[nodiscard]] CollisionObject*
        findByTag(const std::string& tag);

private:
    // ── Broadphase: uniform spatial grid ──────────────────────────────────────
    // Each object is hashed into grid cells by its broadphase AABB.
    // Only pairs sharing at least one cell are tested in narrowphase.
    // This avoids O(n²) narrowphase tests for sparse scenes.
    struct GridCell {
        int x, y, z;
        bool operator==(const GridCell& o) const {
            return x == o.x && y == o.y && z == o.z;
        }
    };
    struct GridCellHash {
        size_t operator()(const GridCell& c) const {
            // Large primes — avoids clustering
            return (size_t)(c.x * 73856093 ^ c.y * 19349663 ^ c.z * 83492791);
        }
    };

    void rebuildGrid();
    std::vector<std::pair<int,int>> broadphase() const;
    void applyMTV(CollisionObject& a, CollisionObject& b, const ContactInfo& c);

    float m_cellSize;
    std::vector<CollisionObject> m_objects;
    CollisionCallback            m_onContact;

    // grid: cell → list of object indices
    using Grid = std::unordered_map<GridCell, std::vector<int>, GridCellHash>;
    mutable Grid m_grid;
};

} // namespace vkgfx
