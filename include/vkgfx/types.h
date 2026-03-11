#pragma once

#include <array>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#define VULKAN_HPP_NO_EXCEPTIONS
#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#ifndef GLM_FORCE_RADIANS
#  define GLM_FORCE_RADIANS
#endif
#ifndef GLM_FORCE_DEPTH_ZERO_TO_ONE
#  define GLM_FORCE_DEPTH_ZERO_TO_ONE
#endif
#ifndef GLM_ENABLE_EXPERIMENTAL
#  define GLM_ENABLE_EXPERIMENTAL
#endif
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/hash.hpp>
#include <glm/gtx/quaternion.hpp>

namespace vkgfx {

using Vec2 = glm::vec2;
using Vec3 = glm::vec3;
using Vec4 = glm::vec4;
using Mat3 = glm::mat3;
using Mat4 = glm::mat4;
using Quat = glm::quat;

inline constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
inline constexpr uint32_t MAX_LIGHTS           = 256;
inline constexpr uint32_t MAX_DESCRIPTOR_SETS  = 1024;
inline constexpr uint32_t MAX_TEXTURES_PER_MAT = 5;

#ifndef NDEBUG
#  define VKGFX_ASSERT(cond, msg)                                        \
     do { if (!(cond)) {                                                  \
       std::cerr << "[VKGFX ASSERT] " << msg << "\n  at "                \
                 << __FILE__ << ":" << __LINE__ << std::endl;            \
       std::abort(); } } while(0)
#else
#  define VKGFX_ASSERT(cond, msg) ((void)0)
#endif

inline void VK_CHECK(VkResult result, std::string_view msg = "") {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string("[Vulkan Error] ") + std::string(msg) +
                                 " (code=" + std::to_string(static_cast<int>(result)) + ")");
}

[[nodiscard]] inline std::vector<char> readFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Failed to open file: " + path.string());
    size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> buf(size);
    file.seekg(0);
    file.read(buf.data(), static_cast<std::streamsize>(size));
    return buf;
}

enum class MSAASamples : uint32_t {
    x1 = VK_SAMPLE_COUNT_1_BIT,
    x2 = VK_SAMPLE_COUNT_2_BIT,
    x4 = VK_SAMPLE_COUNT_4_BIT,
    x8 = VK_SAMPLE_COUNT_8_BIT,
};

enum class LightType : uint32_t { Point = 0, Directional = 1, Spot = 2 };
enum class CullMode    { None, Front, Back, FrontAndBack };
enum class PolygonMode { Fill, Wireframe, Points };

// Vertex — tangent stored as Vec4 (xyz=tangent, w=handedness) per glTF 2.0.
struct Vertex {
    Vec3 position {0.f};
    Vec3 normal   {0.f, 1.f, 0.f};
    Vec2 uv       {0.f};
    Vec4 tangent  {1.f, 0.f, 0.f, 1.f};

    static VkVertexInputBindingDescription getBindingDescription() {
        return { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
    }
    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
        return {{
            { 0, 0, VK_FORMAT_R32G32B32_SFLOAT,   offsetof(Vertex, position) },
            { 1, 0, VK_FORMAT_R32G32B32_SFLOAT,   offsetof(Vertex, normal)   },
            { 2, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(Vertex, uv)       },
            { 3, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent)  },
        }};
    }
    bool operator==(const Vertex& o) const noexcept {
        return position == o.position && normal == o.normal && uv == o.uv;
    }
};

// ── GPU UBOs ──────────────────────────────────────────────────────────────────

// Frame-level UBO — set 0, binding 0 (geometry + lighting passes).
struct alignas(16) FrameUBO {
    Mat4  view;
    Mat4  proj;
    Mat4  viewProj;
    Mat4  invView;
    Mat4  invProj;
    Vec4  cameraPos;   // xyz = world pos
    float time;
    float pad[3];
};

// Per-material UBO — set 1, binding 5 (geometry pass).
struct alignas(16) MaterialUBO {
    Vec4  albedoFactor     = {1.f, 1.f, 1.f, 1.f};
    float metallicFactor   = 1.f;
    float roughnessFactor  = 1.f;
    float emissiveStrength = 1.f;
    float alphaCutoff      = 0.f;
};

// Per-object push constant (128 B exactly — fits Vulkan's minimum guarantee).
struct alignas(16) ModelPushConstant {
    Mat4 model;
    Mat4 normalMatrix;
};

// SSAO parameters — set 0, binding 3 (ssao pass).
// Must match ssao.frag layout exactly: proj, invProj, view, samples[32], noiseScale, radius, bias.
struct alignas(16) SSAOParams {
    Mat4  proj;
    Mat4  invProj;
    Mat4  view;          // world→view transform, used to convert G-buffer normals to view space
    Vec4  samples[32];   // hemisphere kernel in view space
    Vec2  noiseScale;    // screen_resolution / 4
    float radius;
    float bias;
};

// Tone-map push constant.
struct TonemapPC {
    float    exposure;
    uint32_t operator_;  // 0=ACES, 1=Reinhard, 2=Uncharted2
};

// SSAO blur push constant.
struct BlurPC { Vec2 texelSize; };

// GPU light matching lighting.frag GpuLight struct.
struct alignas(16) GpuLight {
    Vec4     position;       // xyz=world pos, w=range
    Vec4     direction;      // xyz=dir, w=spot outer cos
    Vec4     color;          // xyz=linear RGB, w=intensity
    uint32_t type;           // 0=point,1=directional,2=spot
    float    innerAngleCos;
    float    pad[2];
};

// Light SSBO layout (persistent-mapped).
struct LightSSBO {
    uint32_t count;
    uint32_t pad[3];
    GpuLight lights[MAX_LIGHTS];
};

// Frustum for CPU-side culling.
struct Frustum {
    Vec4 planes[6];
    void extract(const Mat4& viewProj);
    [[nodiscard]] bool containsAABB(const struct AABB& aabb) const;
    [[nodiscard]] bool containsSphere(Vec3 center, float radius) const;
};

} // namespace vkgfx

namespace std {
    template<> struct hash<vkgfx::Vertex> {
        size_t operator()(const vkgfx::Vertex& v) const {
            return ((hash<glm::vec3>()(v.position) ^
                    (hash<glm::vec3>()(v.normal) << 1)) >> 1) ^
                    (hash<glm::vec2>()(v.uv) << 1);
        }
    };
}
