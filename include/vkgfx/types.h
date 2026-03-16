#pragma once
// include/vkgfx/types.h
// Vertex layout and GPU-side UBO structs.
// RULE: Vertex attribute locations MUST match shaders/gbuffer.vert exactly.
// RULE: Every struct used in a UBO must be alignas(16).

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <array>
#include <cstdint>

namespace vkgfx {

// ── Vertex ─────────────────────────────────────────────────────────────────────
// location 0 = position, 1 = normal, 2 = tangent (vec4, w=sign), 3 = uv
struct Vertex {
    glm::vec3 position;  // location 0
    glm::vec3 normal;    // location 1
    glm::vec4 tangent;   // location 2
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

// ── Push constants ────────────────────────────────────────────────────────────
// Must stay <= 128 bytes (minimum guaranteed by all Vulkan drivers)
struct MeshPush {
    glm::mat4 model;
    glm::mat4 normalMatrix;
};
static_assert(sizeof(MeshPush) == 128, "MeshPush must be exactly 128 bytes");

// ── Scene UBO (set=0, binding=0 in lighting pass) ─────────────────────────────
struct alignas(16) SceneUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewProj;
    glm::vec4 cameraPos;
    glm::vec2 viewport;
    float     time   = 0.f;
    float     _pad0  = 0.f;
};

// ── Light data structs used inside LightUBO ───────────────────────────────────
// NOTE: These are plain data structs for GPU upload — not the scene-graph classes.
// The scene-graph classes (DirectionalLight, PointLight) live in scene.h.

struct alignas(16) DirLightData {
    glm::vec4 direction;   // w unused
    glm::vec4 color;       // w = intensity
    uint32_t  enabled;
    float     _pad[3];
};

struct alignas(16) PointLightData {
    glm::vec4 position;    // w unused
    glm::vec4 color;       // w = intensity
    float     radius;
    float     _pad[3];
};

inline constexpr uint32_t MAX_POINT_LIGHTS = 8;

struct alignas(16) LightUBO {
    DirLightData  sun;
    PointLightData points[MAX_POINT_LIGHTS];
    uint32_t      pointCount   = 0;
    float         iblIntensity = 0.f;
    uint32_t      gbufferDebug = 0;
    float         _pad         = 0.f;
};

// ── PBR material params (UBO at set=1, binding=3) ─────────────────────────────
struct alignas(16) PBRParams {
    glm::vec4 albedo          = {1.f, 1.f, 1.f, 1.f};
    float     roughness        = 0.5f;
    float     metallic         = 0.0f;
    float     ao               = 1.0f;
    float     emissive         = 0.0f;
    uint32_t  hasAlbedoTex     = 0;
    uint32_t  hasNormalTex     = 0;
    uint32_t  hasRmaTex        = 0;
    float     _pad             = 0.f;
};

} // namespace vkgfx
