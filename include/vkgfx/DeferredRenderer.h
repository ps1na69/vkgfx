#pragma once
// DeferredRenderer.h
// Top-level orchestrator for the deferred shading pipeline.
//
// Frame execution order
// ─────────────────────
//  1. Geometry Pass   – fills the G-buffer (position, normal, albedo, material, emissive)
//  2. SSAO Pass       – screen-space ambient occlusion, writes to ssaoImage
//  3. SSAO Blur       – 4x4 box blur to remove SSAO noise
//  4. Lighting Pass   – PBR Cook-Torrance, accumulates all lights into hdrImage
//  5. Tone-map Pass   – Filmic Aces / Reinhard, writes LDR result to swapchain image
//
// Replacing the existing forward renderer
// ────────────────────────────────────────
// Previously each mesh was drawn once per light (multi-pass forward) or with a
// large light array in a single fragment shader invocation (clustered forward).
// The geometry pass now separates shading from geometry: every object is drawn
// once regardless of light count, and lighting runs in screen space.  This
// gives O(screen_pixels × lights) complexity instead of
// O(geometry × lights), which is a large win when light count is high.
//
// Required per-frame UBO layout (set 0, binding 0):
//   struct FrameUniforms {
//       mat4  view;
//       mat4  proj;
//       mat4  viewProj;
//       mat4  invView;
//       mat4  invProj;
//       vec4  cameraPos;    // xyz = world pos
//       float time;
//       float pad[3];
//   };
//
// Required per-object push-constant (geometry pass only):
//   struct ObjectPC {
//       mat4 model;
//       mat4 normalMatrix;  // transpose(inverse(model))
//   };

#include "GBuffer.h"
#include "SSAOPass.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>
#include <glm/glm.hpp>

namespace vkgfx {

// ─── Light data (CPU side) ───────────────────────────────────────────────────

enum class LightType : uint32_t {
    Point       = 0,
    Directional = 1,
    Spot        = 2,
};

// Matches the std140 layout expected by the lighting pass shader.
struct alignas(16) GpuLight {
    glm::vec4  position;        // xyz = world pos, w = range (point/spot)
    glm::vec4  direction;       // xyz = normalised dir, w = spot outer angle (cos)
    glm::vec4  color;           // xyz = linear RGB, w = intensity (lux / candela)
    uint32_t   type;            // LightType enum value
    float      innerAngleCos;   // Spot inner cone (cos)
    float      padding[2];
};

// Maximum number of dynamic lights supported per frame.
// Increase if your scene needs more; the SSBO grows linearly so cost is low.
static constexpr uint32_t MAX_LIGHTS = 256;

// ─── Renderer config ─────────────────────────────────────────────────────────

struct DeferredRendererCreateInfo {
    VkDevice           device;
    VkPhysicalDevice   physicalDevice;
    VkRenderPass       swapchainRenderPass;  // Final blit / tone-map target
    VkFormat           swapchainFormat;
    uint32_t           width;
    uint32_t           height;
    uint32_t           framesInFlight;       // Usually 2 or 3
    VkDescriptorPool   descriptorPool;       // Pool large enough for all sets
    VkCommandPool      commandPool;
    VkQueue            graphicsQueue;
    VkPipelineCache    pipelineCache;        // Can be VK_NULL_HANDLE
};

// ─── Per-frame data ───────────────────────────────────────────────────────────

struct FrameUniforms {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewProj;
    glm::mat4 invView;
    glm::mat4 invProj;
    glm::vec4 cameraPos;
    float     time;
    float     pad[3];
};

// ─── Main renderer ────────────────────────────────────────────────────────────

class DeferredRenderer {
public:
    explicit DeferredRenderer(const DeferredRendererCreateInfo& ci);
    ~DeferredRenderer();

    // Call once when the swapchain is rebuilt (window resize etc.)
    void on_resize(uint32_t width, uint32_t height);

    // Begin the geometry pass for the given frame; returns the command buffer
    // to record mesh draw calls into.  Caller must call end_geometry_pass() when done.
    VkCommandBuffer begin_geometry_pass(uint32_t frameIndex);
    void            end_geometry_pass(uint32_t frameIndex);

    // Update the light list for the current frame (copied to GPU SSBO).
    void set_lights(uint32_t frameIndex,
                    const GpuLight* lights,
                    uint32_t count);

    // Record and submit SSAO + Lighting + Tone-map passes.
    // swapchainImageIndex is the index returned by vkAcquireNextImageKHR.
    void render_deferred_passes(uint32_t frameIndex,
                                uint32_t swapchainImageIndex,
                                VkFramebuffer swapchainFramebuffer,
                                const FrameUniforms& frame,
                                VkSemaphore waitSemaphore,
                                VkSemaphore signalSemaphore,
                                VkFence fence);

    // Expose G-buffer for debug visualization tools.
    const GBuffer& get_gbuffer() const { return m_gbuffer; }

private:
    // ── Initialization helpers ──
    void create_frame_uniforms_buffers();
    void create_light_ssbo();
    void create_geometry_renderpass();
    void create_geometry_framebuffer();
    void create_geometry_pipeline();
    void create_lighting_pipeline();
    void create_tonemap_pipeline();
    void create_descriptor_layouts();
    void create_descriptor_sets();
    void create_hdr_image();
    void destroy_swapchain_dependent();
    void rebuild_swapchain_dependent();

    // ── Recording helpers ──
    void record_ssao_pass(VkCommandBuffer cb, uint32_t frameIndex);
    void record_lighting_pass(VkCommandBuffer cb, uint32_t frameIndex);
    void record_tonemap_pass(VkCommandBuffer cb, uint32_t frameIndex,
                             VkFramebuffer swapchainFB);

    // ── Vulkan device handles (not owned) ──
    VkDevice           m_device           = VK_NULL_HANDLE;
    VkPhysicalDevice   m_physDevice       = VK_NULL_HANDLE;
    VkDescriptorPool   m_descriptorPool   = VK_NULL_HANDLE;
    VkCommandPool      m_commandPool      = VK_NULL_HANDLE;
    VkQueue            m_graphicsQueue    = VK_NULL_HANDLE;
    VkPipelineCache    m_pipelineCache    = VK_NULL_HANDLE;
    VkRenderPass       m_swapchainRP      = VK_NULL_HANDLE;
    VkFormat           m_swapchainFormat  = VK_FORMAT_UNDEFINED;

    // ── Dimensions ──
    uint32_t m_width  = 0;
    uint32_t m_height = 0;
    uint32_t m_framesInFlight = 2;

    // ── G-Buffer ──
    GBuffer  m_gbuffer{};

    // ── Geometry pass ──
    VkRenderPass              m_geomRenderPass  = VK_NULL_HANDLE;
    VkFramebuffer             m_geomFramebuffer = VK_NULL_HANDLE;
    VkPipelineLayout          m_geomPipeLayout  = VK_NULL_HANDLE;
    VkPipeline                m_geomPipeline    = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_geomCmdBuffers;

    // ── Lighting pass ──
    // Renders to an HDR intermediate image.
    VkImage               m_hdrImage      = VK_NULL_HANDLE;
    VkImageView           m_hdrImageView  = VK_NULL_HANDLE;
    VkDeviceMemory        m_hdrMemory     = VK_NULL_HANDLE;
    VkSampler             m_hdrSampler    = VK_NULL_HANDLE;
    VkRenderPass          m_lightingRP    = VK_NULL_HANDLE;
    VkFramebuffer         m_lightingFB    = VK_NULL_HANDLE;
    VkPipelineLayout      m_lightPipeLayout = VK_NULL_HANDLE;
    VkPipeline            m_lightPipeline  = VK_NULL_HANDLE;

    // ── SSAO pass ──
    SSAOPass m_ssao;

    // ── Tone-map pass ──
    VkPipelineLayout      m_tonemapPipeLayout = VK_NULL_HANDLE;
    VkPipeline            m_tonemapPipeline   = VK_NULL_HANDLE;

    // ── Descriptor layouts ──
    VkDescriptorSetLayout m_frameSetLayout   = VK_NULL_HANDLE;  // Set 0: FrameUniforms
    VkDescriptorSetLayout m_gbufferSetLayout = VK_NULL_HANDLE;  // Set 1: G-buffer textures
    VkDescriptorSetLayout m_lightSetLayout   = VK_NULL_HANDLE;  // Set 2: Light SSBO
    VkDescriptorSetLayout m_hdrSetLayout     = VK_NULL_HANDLE;  // Set 3: HDR image for tonemap

    // ── Per-frame descriptors and buffers ──
    struct PerFrame {
        // Frame UBO
        VkBuffer            frameUBO       = VK_NULL_HANDLE;
        VkDeviceMemory      frameUBOMem    = VK_NULL_HANDLE;
        void*               frameUBOPtr    = nullptr;  // persistently mapped

        // Light SSBO
        VkBuffer            lightSSBO      = VK_NULL_HANDLE;
        VkDeviceMemory      lightSSBOMem   = VK_NULL_HANDLE;
        void*               lightSSBOPtr   = nullptr;  // persistently mapped
        uint32_t            lightCount     = 0;

        // Descriptor sets
        VkDescriptorSet     frameSet       = VK_NULL_HANDLE;
        VkDescriptorSet     gbufferSet     = VK_NULL_HANDLE;
        VkDescriptorSet     lightSet       = VK_NULL_HANDLE;
        VkDescriptorSet     hdrSet         = VK_NULL_HANDLE;

        // Deferred command buffer (SSAO + lighting + tonemap in sequence)
        VkCommandBuffer     deferredCB     = VK_NULL_HANDLE;
    };
    std::vector<PerFrame> m_frames;
};

} // namespace vkgfx
