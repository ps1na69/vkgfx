#pragma once
// include/vkgfx/renderer.h
// Deferred renderer: G-buffer fill → lighting (IBL+sun) → tonemap → present.

#include "config.h"
#include "vk_raii.h"
#include "scene.h"
#include "ibl.h"

#include <vulkan/vulkan.h>
#include <memory>
#include <array>

namespace vkgfx {

class Window;
class Context;
class Swapchain;

inline constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

class Renderer {
public:
    explicit Renderer(Window& window, RendererConfig cfg = {});
    ~Renderer();

    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;

    void render  (Scene& scene);
    void shutdown();

    // Hot-reload: IBL toggle, sun toggle, SSAO, gbufferDebug.
    // Call rebuild() for HDR path / envMapSize changes.
    void applyConfig(const RendererConfig& cfg);
    void rebuild    (const RendererConfig& cfg);

    [[nodiscard]] const RendererConfig& config()  const { return m_cfg; }
    [[nodiscard]] Context&              context() const { return *m_ctx; }

private:
    // ── Init phases ───────────────────────────────────────────────────────────
    void initDescriptorPools();
    void initGBuffer();
    void initLightingPass();
    void initTonemapPass();
    void initPerFrameResources();
    void initIBL();
    void validateAssets();

    // ── Per-frame ─────────────────────────────────────────────────────────────
    void recordGBuffer (VkCommandBuffer cmd, Scene& scene, uint32_t frameIdx);
    void recordLighting(VkCommandBuffer cmd, uint32_t frameIdx);
    void recordTonemap (VkCommandBuffer cmd, uint32_t frameIdx);

    VkShaderModule loadShaderModule(const std::string& name) const;

    // ── Owned objects ─────────────────────────────────────────────────────────
    RendererConfig             m_cfg;
    std::unique_ptr<Context>   m_ctx;
    std::unique_ptr<Swapchain> m_swapchain;
    std::unique_ptr<IBLSystem> m_ibl;

    // G-buffer attachments: [0]=albedo [1]=normal [2]=RMA [3]=depth
    std::array<AllocatedImage, 4> m_gbuffer{};

    // Offscreen HDR target (lighting result, fed into tonemap)
    AllocatedImage m_hdrTarget{};

    // Render passes
    VkRenderPass  m_gbufferPass  = VK_NULL_HANDLE;
    VkRenderPass  m_lightingPass = VK_NULL_HANDLE;

    // Framebuffers
    VkFramebuffer m_gbufferFb    = VK_NULL_HANDLE;
    VkFramebuffer m_lightingFb   = VK_NULL_HANDLE;

    // Pipelines + layouts
    VkPipelineLayout m_gbufferLayout    = VK_NULL_HANDLE;
    VkPipeline       m_gbufferPipeline  = VK_NULL_HANDLE;
    VkPipelineLayout m_lightingLayout   = VK_NULL_HANDLE;
    VkPipeline       m_lightingPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_tonemapLayout    = VK_NULL_HANDLE;
    VkPipeline       m_tonemapPipeline  = VK_NULL_HANDLE;

    // Descriptor set layouts
    VkDescriptorSetLayout m_gbufferSetLayout   = VK_NULL_HANDLE; // set=0 in gbuffer pass (scene UBO)
    VkDescriptorSetLayout m_materialSetLayout  = VK_NULL_HANDLE; // set=1 in gbuffer pass (material)
    VkDescriptorSetLayout m_lightingSetLayout  = VK_NULL_HANDLE; // set=0 in lighting pass (gbuffer samplers)
    VkDescriptorSetLayout m_sceneSetLayout     = VK_NULL_HANDLE; // set=1 in lighting pass (scene+light UBO)
    VkDescriptorSetLayout m_iblSetLayout       = VK_NULL_HANDLE; // set=2 in lighting pass (IBL)
    VkDescriptorSetLayout m_tonemapSetLayout   = VK_NULL_HANDLE; // set=0 in tonemap pass

    // Shared samplers
    VkSampler m_gbufferSampler = VK_NULL_HANDLE;
    VkSampler m_hdrSampler     = VK_NULL_HANDLE;

    VkHandle<VkDescriptorPool> m_descriptorPool;

    struct PerFrame {
        VkHandle<VkSemaphore> imageAvailable;
        VkHandle<VkSemaphore> renderFinished;
        VkHandle<VkFence>     inFlight;
        VkCommandBuffer       cmd = VK_NULL_HANDLE;

        // Descriptor sets allocated once, updated as needed
        VkDescriptorSet gbufferSceneSet    = VK_NULL_HANDLE; // set=0 gbuffer pass
        VkDescriptorSet lightingGbufferSet = VK_NULL_HANDLE; // set=0 lighting pass
        VkDescriptorSet lightingSceneSet   = VK_NULL_HANDLE; // set=1 lighting pass
        VkDescriptorSet iblSet             = VK_NULL_HANDLE; // set=2 lighting pass
        VkDescriptorSet tonemapSet         = VK_NULL_HANDLE; // set=0 tonemap pass

        AllocatedBuffer sceneUbo{};
        AllocatedBuffer lightUbo{};
    };

    std::array<PerFrame, MAX_FRAMES_IN_FLIGHT> m_frames{};
    uint32_t m_frameIdx    = 0;
    uint32_t m_swapIdx     = 0;
    bool     m_initialized = false;
};

} // namespace vkgfx
