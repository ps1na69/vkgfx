#pragma once
// include/vkgfx/renderer.h

#include "config.h"
#include "vk_raii.h"
#include "scene.h"
#include "ibl.h"
#include "frame_graph.h"
#include "profiler.h"

#include <vulkan/vulkan.h>
#include <memory>
#include <array>
#include <chrono>
#include <functional>

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

    // ── GPU Profiler public API ───────────────────────────────────────────────

    /// Read the last completed frame's timing stats.
    /// Returns zeros when VKGFX_ENABLE_PROFILING is not defined.
    [[nodiscard]] const FrameStats& profilerStats() const { return m_profiler.stats(); }

    /// Register a callback invoked every frame inside the ImGui scope,
    /// after the built-in profiler overlay has been drawn.
    /// Use this to add your own ImGui windows without managing the frame lifecycle.
    /// No-op when VKGFX_ENABLE_PROFILING is not defined.
    void setImGuiCallback(std::function<void()> cb) { m_imguiCallback = std::move(cb); }

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
    void initPointShadowPass();
    void validateAssets();
    void uploadMeshMaterials(Scene& scene);
    void rebuildOffscreenResources();

    // ── ImGui lifecycle (compiled away when VKGFX_ENABLE_PROFILING is absent) ─
    void initImGui();
    void shutdownImGui();
    void beginImGuiFrame();
    void endImGuiFrame(VkCommandBuffer cmd);

    // ── Per-pass recording ────────────────────────────────────────────────────
    void recordShadowPass     (VkCommandBuffer cmd, Scene& scene);
    void recordPointShadowPass(VkCommandBuffer cmd, Scene& scene, uint32_t frameIdx);
    void recordGBuffer        (VkCommandBuffer cmd, Scene& scene, uint32_t frameIdx);
    void recordLighting       (VkCommandBuffer cmd, uint32_t frameIdx);
    void recordTonemap        (VkCommandBuffer cmd, uint32_t frameIdx);

    // ── Frame graph ───────────────────────────────────────────────────────────
    void buildFrameGraph(Scene& scene, uint32_t frameIdx);

    VkShaderModule loadShaderModule(const std::string& name) const;

    // ── Core objects ──────────────────────────────────────────────────────────
    RendererConfig              m_cfg;
    Window*                     m_window = nullptr;     // non-owning, for ImGui GLFW backend
    std::unique_ptr<Context>    m_ctx;
    std::unique_ptr<Swapchain>  m_swapchain;
    std::unique_ptr<IBLSystem>  m_ibl;
    std::unique_ptr<FrameGraph> m_frameGraph;

    // ── GPU profiler ──────────────────────────────────────────────────────────
    GpuProfiler           m_profiler;
    VkDescriptorPool      m_imguiPool         = VK_NULL_HANDLE;
    bool                  m_imguiInitialized  = false;
    std::function<void()> m_imguiCallback;

    // Per-frame draw statistics
    uint32_t m_drawCallCount = 0;
    uint64_t m_triangleCount = 0;

    // CPU frame timer
    using Clock     = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    TimePoint m_frameStart{};

    // ── G-buffer attachments ──────────────────────────────────────────────────
    std::array<AllocatedImage, 6> m_gbuffer{};
    AllocatedImage m_hdrTarget{};

    VkRenderPass  m_gbufferPass  = VK_NULL_HANDLE;
    VkRenderPass  m_lightingPass = VK_NULL_HANDLE;
    VkFramebuffer m_gbufferFb    = VK_NULL_HANDLE;
    VkFramebuffer m_lightingFb   = VK_NULL_HANDLE;

    // ── Directional shadow ────────────────────────────────────────────────────
    AllocatedImage      m_shadowMap{};
    VkRenderPass        m_shadowPass       = VK_NULL_HANDLE;
    VkFramebuffer       m_shadowFb         = VK_NULL_HANDLE;
    VkPipelineLayout    m_shadowLayout     = VK_NULL_HANDLE;
    VkPipeline          m_shadowPipeline   = VK_NULL_HANDLE;
    VkSampler           m_shadowSampler    = VK_NULL_HANDLE;
    static constexpr uint32_t SHADOW_MAP_SIZE = 2048;

    // ── Point-light shadow cubemap ────────────────────────────────────────────
    AllocatedImage    m_pointShadowCube{};
    VkImageView       m_pointCubeFaceViews[6]  = {};
    VkImageView       m_pointCubeSamplerView   = VK_NULL_HANDLE;
    VkRenderPass      m_pointShadowPass        = VK_NULL_HANDLE;
    VkFramebuffer     m_pointShadowFbs[6]      = {};
    VkSampler         m_pointShadowSampler     = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_pointShadowDsLayout = VK_NULL_HANDLE;
    VkPipelineLayout  m_pointShadowLayout      = VK_NULL_HANDLE;
    VkPipeline        m_pointShadowPipeline    = VK_NULL_HANDLE;
    static constexpr uint32_t POINT_SHADOW_SIZE = 512;

    // ── Pipelines ─────────────────────────────────────────────────────────────
    VkPipelineLayout m_gbufferLayout    = VK_NULL_HANDLE;
    VkPipeline       m_gbufferPipeline  = VK_NULL_HANDLE;
    VkPipelineLayout m_lightingLayout   = VK_NULL_HANDLE;
    VkPipeline       m_lightingPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_tonemapLayout    = VK_NULL_HANDLE;
    VkPipeline       m_tonemapPipeline  = VK_NULL_HANDLE;

    // ── Descriptor set layouts ────────────────────────────────────────────────
    VkDescriptorSetLayout m_gbufferSetLayout   = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_materialSetLayout  = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_lightingSetLayout  = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_sceneSetLayout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_iblSetLayout       = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_tonemapSetLayout   = VK_NULL_HANDLE;

    // ── Samplers + fallbacks ──────────────────────────────────────────────────
    VkSampler m_gbufferSampler  = VK_NULL_HANDLE;
    VkSampler m_hdrSampler      = VK_NULL_HANDLE;
    VkSampler m_fallbackSampler = VK_NULL_HANDLE;

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
        VkDescriptorSet pointShadowDs      = VK_NULL_HANDLE;

        AllocatedBuffer sceneUbo{};
        AllocatedBuffer lightUbo{};
        AllocatedBuffer defaultParamsUbo{};
        AllocatedBuffer pointShadowLightUbo{};
    };

    std::array<PerFrame, MAX_FRAMES_IN_FLIGHT> m_frames{};

    std::vector<VkHandle<VkSemaphore>> m_acquireSemaphores;
    std::vector<VkHandle<VkSemaphore>> m_renderFinishedSems;

    std::vector<AllocatedBuffer> m_materialUbos;

    VkExtent2D m_offscreenExtent{};

    uint32_t m_frameIdx     = 0;
    uint32_t m_swapIdx      = 0;
    bool     m_initialized  = false;
    bool     m_shuttingDown = false;
};

} // namespace vkgfx
