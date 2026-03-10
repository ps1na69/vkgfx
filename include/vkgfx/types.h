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

//  Aliases
using Vec2 = glm::vec2;
using Vec3 = glm::vec3;
using Vec4 = glm::vec4;
using Mat3 = glm::mat3;
using Mat4 = glm::mat4;
using Quat = glm::quat;

// Constants
inline constexpr uint32_t MAX_FRAMES_IN_FLIGHT  = 2;
inline constexpr uint32_t MAX_SCENE_LIGHTS      = 8;
inline constexpr uint32_t MAX_DESCRIPTOR_SETS   = 256;
inline constexpr uint32_t MAX_TEXTURES_PER_MAT  = 5;
inline constexpr uint32_t SHADOW_MAP_RESOLUTION = 2048;
inline constexpr uint32_t MAX_SHADOW_MAPS       = 4;

// Error Handling
#ifndef NDEBUG
#  define VKGFX_ASSERT(cond, msg) \
     do { if (!(cond)) { std::cerr << "[VKGFX ASSERT] " << msg << "\n  at " << __FILE__ << ":" << __LINE__ << std::endl; std::abort(); } } while(0)
#else
#  define VKGFX_ASSERT(cond, msg) ((void)0)
#endif

inline void VK_CHECK(VkResult result, std::string_view msg = "") {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string("[Vulkan Error] ") + std::string(msg) +
                                 " (code=" + std::to_string(static_cast<int>(result)) + ")");
    }
}

// Utility
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
    x1  = VK_SAMPLE_COUNT_1_BIT,
    x2  = VK_SAMPLE_COUNT_2_BIT,
    x4  = VK_SAMPLE_COUNT_4_BIT,
    x8  = VK_SAMPLE_COUNT_8_BIT,
};

enum class LightType { Point, Directional, Spot };

enum class CullMode { None, Front, Back, FrontAndBack };

enum class PolygonMode { Fill, Wireframe, Points };

//Vertex
struct Vertex {
    Vec3 position{0.f};
    Vec3 normal{0.f, 1.f, 0.f};
    Vec2 uv{0.f};
    Vec3 tangent{1.f, 0.f, 0.f};

    static VkVertexInputBindingDescription getBindingDescription() {
        return { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
    }

    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
        return {{
            { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
            { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)   },
            { 2, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv)       },
            { 3, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, tangent)  },
        }};
    }

    bool operator==(const Vertex& o) const noexcept {
        return position == o.position && normal == o.normal && uv == o.uv;
    }
};

//uniforms
struct alignas(16) CameraUBO {
    Mat4 view;
    Mat4 proj;
    Mat4 viewProj;
    Vec4 position;   // w = near
    Vec4 params;     // x=far, y=fov, z=aspect, w=time
};

struct alignas(16) LightData {
    Vec4 position;   // w = type (0=point,1=dir,2=spot)
    Vec4 color;      // w = intensity
    Vec4 direction;  // w = innerCone
    Vec4 params;     // x=outerCone, y=range, z=castShadow, w=shadowBias
};

struct alignas(16) SceneUBO {
    LightData lights[MAX_SCENE_LIGHTS];
    Vec4      ambientColor;         // w = intensity
    int32_t   lightCount{0};
    int32_t   useLinearOutput{0};   // 1 = PP active, skip in-shader tonemapping
    int32_t   _pad[2];
};

struct alignas(16) ShadowUBO {
    Mat4    lightSpace[MAX_SHADOW_MAPS];  // VP per caster
    Vec4    params[MAX_SHADOW_MAPS];      // x=bias y=normalBias zw=unused
    int32_t count;
    int32_t _pad[3];
};

struct ShadowPushConstant {
    Mat4 lightSpace;  // 64 B
    Mat4 model;       // 64 B  (total = 128 B exactly)
};

struct alignas(16) ModelPushConstant {
    Mat4 model;
    Mat4 normalMatrix;  // transpose(inverse(model))
};

} // namespace vkgfx

// нада, (не трогати)
namespace std {
    template<> struct hash<vkgfx::Vertex> {
        size_t operator()(const vkgfx::Vertex& v) const {
            return ((hash<glm::vec3>()(v.position) ^
                    (hash<glm::vec3>()(v.normal) << 1)) >> 1) ^
                    (hash<glm::vec2>()(v.uv) << 1);
        }
    };
}
