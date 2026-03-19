#pragma once
// include/vkgfx/types.h
// Vertex layout and GPU UBO structs.
//
// LAYOUT RULE: Every struct uses ONLY vec4/uvec4/mat4 members so the C++ and
// GLSL std140 layouts are trivially identical — no float[] padding arrays
// (GLSL std140 gives each array element stride=16, which breaks C++ packing).

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <array>
#include <cstdint>

namespace vkgfx {

// ── Vertex ────────────────────────────────────────────────────────────────────
struct Vertex {
    glm::vec3 position;  // location 0
    glm::vec3 normal;    // location 1
    glm::vec4 tangent;   // location 2  (w = bitangent sign)
    glm::vec2 uv;        // location 3

    static VkVertexInputBindingDescription bindingDescription() {
        VkVertexInputBindingDescription b{};
        b.binding   = 0;
        b.stride    = sizeof(Vertex);
        b.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return b;
    }

    static std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 4> a{};
        a[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,    static_cast<uint32_t>(offsetof(Vertex, position))};
        a[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT,    static_cast<uint32_t>(offsetof(Vertex, normal))};
        a[2] = {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, tangent))};
        a[3] = {3, 0, VK_FORMAT_R32G32_SFLOAT,       static_cast<uint32_t>(offsetof(Vertex, uv))};
        return a;
    }

    bool operator==(const Vertex& o) const {
        return position == o.position && normal == o.normal && uv == o.uv;
    }
};

// ── Push constants (128 bytes) ────────────────────────────────────────────────
struct MeshPush {
    glm::mat4 model;
    glm::mat4 normalMatrix;
};
static_assert(sizeof(MeshPush) == 128);

// ── Scene UBO ─────────────────────────────────────────────────────────────────
// set=0 binding=0 in G-buffer pass (vertex stage)
// set=1 binding=0 in lighting pass (fragment stage)
struct SceneUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewProj;
    glm::mat4 invViewProj;    // precomputed — avoids per-pixel inverse() in lighting
    glm::mat4 lightViewProj;  // ortho projection from directional light POV
    glm::vec4 cameraPos;      // xyz = world position, w unused
    glm::vec4 viewport;       // xy = (width, height), zw unused
};

// ── Light UBO ─────────────────────────────────────────────────────────────────
// set=1 binding=1 in lighting pass.
//
// LAYOUT — every member is a vec4/uvec4/mat4 so std140 stride is unambiguous.
// C++ sizeof and GLSL std140 offsets are identical by construction.
//
// offset   0: sunDirection   vec4    xyz=direction (from fragment toward light), w=unused
// offset  16: sunColor       vec4    xyz=color, w=intensity
// offset  32: sunFlags       uvec4   x=enabled, y=shadowEnabled, zw=0
// offset  48: points[0].pos  vec4    ...
// ...
// points[N].pos    offset 48 + N*48 + 0
// points[N].color  offset 48 + N*48 + 16
// points[N].params offset 48 + N*48 + 32  x=radius, y=unused, z=unused, w=unused
// offset 48 + 8*48 = 432: miscFlags  uvec4  x=pointCount, y=gbufferDebug, z=iblEnabled, w=0
// offset 448:              iblParams  vec4   x=iblIntensity, yzw=0

inline constexpr uint32_t MAX_POINT_LIGHTS = 8;

struct alignas(16) PointLightGPU {
    glm::vec4 position;   // xyz = world pos, w unused
    glm::vec4 color;      // xyz = color, w = intensity
    glm::vec4 params;     // x = radius, yzw unused
};
static_assert(sizeof(PointLightGPU) == 48);

struct LightUBO {
    // Directional sun — 3 vec4s = 48 bytes
    glm::vec4  sunDirection;        // offset   0
    glm::vec4  sunColor;            // offset  16  (w = intensity)
    glm::uvec4 sunFlags;            // offset  32  (x=enabled, y=shadowEnabled)

    // Point lights — 8 × 48 bytes = 384 bytes
    PointLightGPU points[MAX_POINT_LIGHTS]; // offset  48

    // Misc flags  — offset 432
    glm::uvec4 miscFlags;           // x=pointCount, y=gbufferDebug, zw=0

    // IBL — offset 448
    glm::vec4  iblParams;           // x=iblIntensity, yzw=0
};

// ── PBR material params ────────────────────────────────────────────────────────
// set=1 binding=4 in G-buffer pass.
// All vec4 to avoid any padding ambiguity.
struct PBRParams {
    glm::vec4  albedo             = {1.f, 1.f, 1.f, 1.f};
    glm::vec4  emissive           = {0.f, 0.f, 0.f, 0.f};  // rgb = color, w = strength
    glm::vec4  pbr                = {0.5f, 0.f, 1.f, 0.f}; // x=roughness, y=metallic, z=ao
    glm::uvec4 texFlags           = {0u, 0u, 0u, 0u};       // x=hasAlbedo, y=hasNormal, z=hasRMA, w=hasEmissive
};

// Convenience accessors (so existing code still compiles)
inline float& roughness(PBRParams& p) { return p.pbr.x; }
inline float& metallic (PBRParams& p) { return p.pbr.y; }
inline float& ao       (PBRParams& p) { return p.pbr.z; }

} // namespace vkgfx
