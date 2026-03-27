#pragma once
// include/vkgfx/renderer.h
// Deferred renderer: Shadow → G-buffer → Lighting (IBL+sun) → Tonemap → Present.
// Passes are orchestrated by a FrameGraph; render() is a thin frame loop.

#include "config.h"
#include "vk_raii.h"
#include "scene.h"
#include "ibl.h"
#include "frame_graph.h"   // ← added

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
    void initDefaultTextures();
    void initShadowPass();
    void validateAssets();
    void uploadMeshMaterials(Scene& scene);

    // ── Per-pass recording (called from FrameGraph execute callbacks) ─────────
    void recordShadowPass  (VkCommandBuffer cmd, Scene& scene);
    void recordGBuffer     (VkCommandBuffer cmd, Scene& scene, uint32_t frameIdx);
    void recordLighting    (VkCommandBuffer cmd, uint32_t frameIdx);
    void recordTonemap     (VkCommandBuffer cmd, uint32_t frameIdx);

    // ── Frame graph ───────────────────────────────────────────────────────────
    // Registers all passes and their resource I/O into m_frameGraph.
    // Called each frame from render() between reset() and compile().
    void buildFrameGraph(Scene& scene, uint32_t frameIdx);

    VkShaderModule loadShaderModule(const std::string& name) const;

    // ── Owned objects ─────────────────────────────────────────────────────────
    RendererConfig             m_cfg;
    std::unique_ptr<Context>   m_ctx;
    std::unique_ptr<Swapchain> m_swapchain;
    std::unique_ptr<IBLSystem> m_ibl;

    // Frame graph — orchestrates pass ordering and barrier placement
    FrameGraph m_frameGraph;   // ← added

    // G-buffer attachments:
    //   [0]=albedo  [1]=normal  [2]=RMA  [3]=emissive  [4]=shadowCoord  [5]=depth
    std::array<AllocatedImage, 6> m_gbuffer{};

    AllocatedImage m_hdrTarget{};

    VkRenderPass  m_gbufferPass  = VK_NULL_HANDLE;
    VkRenderPass  m_lightingPass = VK_NULL_HANDLE;

    VkFramebuffer m_gbufferFb    = VK_NULL_HANDLE;
    VkFramebuffer m_lightingFb   = VK_NULL_HANDLE;

    AllocatedImage      m_shadowMap{};
    VkRenderPass        m_shadowPass       = VK_NULL_HANDLE;
    VkFramebuffer       m_shadowFb         = VK_NULL_HANDLE;
    VkPipelineLayout    m_shadowLayout     = VK_NULL_HANDLE;
    VkPipeline          m_shadowPipeline   = VK_NULL_HANDLE;
    VkSampler           m_shadowSampler    = VK_NULL_HANDLE;
    static constexpr uint32_t SHADOW_MAP_SIZE = 2048;

    VkPipelineLayout m_gbufferLayout    = VK_NULL_HANDLE;
    VkPipeline       m_gbufferPipeline  = VK_NULL_HANDLE;
    VkPipelineLayout m_lightingLayout   = VK_NULL_HANDLE;
    VkPipeline       m_lightingPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_tonemapLayout    = VK_NULL_HANDLE;
    VkPipeline       m_tonemapPipeline  = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_gbufferSetLayout   = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_materialSetLayout  = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_lightingSetLayout  = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_sceneSetLayout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_iblSetLayout       = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_tonemapSetLayout   = VK_NULL_HANDLE;

    VkSampler m_gbufferSampler = VK_NULL_HANDLE;
    VkSampler m_hdrSampler     = VK_NULL_HANDLE;
    VkSampler m_fallbackSampler= VK_NULL_HANDLE;

    AllocatedImage m_fallbackWhite{};
    AllocatedImage m_fallbackNormal{};
    AllocatedImage m_fallbackRMA{};
    AllocatedImage m_fallbackCube{};

    VkHandle<VkDescriptorPool> m_descriptorPool;

    struct PerFrame {
        VkHandle<VkFence>     inFlight;
        VkCommandBuffer       cmd = VK_NULL_HANDLE;

        VkDescriptorSet gbufferSceneSet    = VK_NULL_HANDLE;
        VkDescriptorSet defaultMaterialSet = VK_NULL_HANDLE;
        VkDescriptorSet lightingGbufferSet = VK_NULL_HANDLE;
        VkDescriptorSet lightingSceneSet   = VK_NULL_HANDLE;
        VkDescriptorSet iblSet             = VK_NULL_HANDLE;
        VkDescriptorSet tonemapSet         = VK_NULL_HANDLE;

        AllocatedBuffer sceneUbo{};
        AllocatedBuffer lightUbo{};
        AllocatedBuffer defaultParamsUbo{};
    };

    std::array<PerFrame, MAX_FRAMES_IN_FLIGHT> m_frames{};

    std::vector<VkHandle<VkSemaphore>> m_acquireSemaphores;
    std::vector<VkHandle<VkSemaphore>> m_renderFinishedSems;

    std::vector<AllocatedBuffer> m_materialUbos;
    uint32_t m_frameIdx    = 0;
    uint32_t m_swapIdx     = 0;
    bool     m_initialized = false;
};

} // namespace vkgfx
