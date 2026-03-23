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
    void initDefaultTextures();
    void initShadowPass();
    void recordShadowPass(VkCommandBuffer cmd, Scene& scene); // 1x1 fallback textures for default material set
    void uploadMeshMaterials(Scene& scene); // allocate+write material descriptors
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

    // G-buffer attachments:
    //   [0]=albedo  [1]=normal  [2]=RMA  [3]=emissive  [4]=shadowCoord  [5]=depth
    std::array<AllocatedImage, 6> m_gbuffer{};

    // Offscreen HDR target (lighting result, fed into tonemap)
    AllocatedImage m_hdrTarget{};

    // Render passes
    VkRenderPass  m_gbufferPass  = VK_NULL_HANDLE;
    VkRenderPass  m_lightingPass = VK_NULL_HANDLE;

    // Framebuffers
    VkFramebuffer m_gbufferFb    = VK_NULL_HANDLE;
    VkFramebuffer m_lightingFb   = VK_NULL_HANDLE;

    // Shadow map pass
    AllocatedImage      m_shadowMap{};
    VkRenderPass        m_shadowPass       = VK_NULL_HANDLE;
    VkFramebuffer       m_shadowFb         = VK_NULL_HANDLE;
    VkPipelineLayout    m_shadowLayout     = VK_NULL_HANDLE;
    VkPipeline          m_shadowPipeline   = VK_NULL_HANDLE;
    VkSampler           m_shadowSampler    = VK_NULL_HANDLE;
    static constexpr uint32_t SHADOW_MAP_SIZE = 2048;

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
    VkSampler m_fallbackSampler= VK_NULL_HANDLE;

    // 1x1 fallback textures for defaultMaterialSet — valid descriptors that satisfy
    // the pipeline layout even when no real material textures are assigned.
    AllocatedImage m_fallbackWhite{};   // albedo fallback (1,1,1,1)
    AllocatedImage m_fallbackNormal{};  // normal fallback (0.5,0.5,1.0 = flat normal)
    AllocatedImage m_fallbackRMA{};     // RMA fallback (0.5 roughness, 0 metallic, 1 ao)
    AllocatedImage m_fallbackCube{};    // 1x1 black cube map — satisfies unwritten IBL set

    VkHandle<VkDescriptorPool> m_descriptorPool;

    struct PerFrame {
        VkHandle<VkFence>     inFlight;
        VkCommandBuffer       cmd = VK_NULL_HANDLE;

        // Descriptor sets allocated once, updated as needed
        VkDescriptorSet gbufferSceneSet    = VK_NULL_HANDLE; // set=0 gbuffer pass
        VkDescriptorSet defaultMaterialSet = VK_NULL_HANDLE; // set=1 gbuffer fallback
        VkDescriptorSet lightingGbufferSet = VK_NULL_HANDLE; // set=0 lighting pass
        VkDescriptorSet lightingSceneSet   = VK_NULL_HANDLE; // set=1 lighting pass
        VkDescriptorSet iblSet             = VK_NULL_HANDLE; // set=2 lighting pass
        VkDescriptorSet tonemapSet         = VK_NULL_HANDLE; // set=0 tonemap pass

        AllocatedBuffer sceneUbo{};
        AllocatedBuffer lightUbo{};
        AllocatedBuffer defaultParamsUbo{}; // PBRParams with all defaults for fallback set
    };

    std::array<PerFrame, MAX_FRAMES_IN_FLIGHT> m_frames{};

    // Per-frame-slot acquire semaphores (fence wait before acquire guarantees
    // the previous use of this slot's semaphore has been consumed).
    std::vector<VkHandle<VkSemaphore>> m_acquireSemaphores;

    // Per-swapchain-image renderFinished semaphores.
    // Signal: indexed by m_swapIdx (known after acquire).
    // Wait (present): same index — avoids reuse while the image is still queued.
    std::vector<VkHandle<VkSemaphore>> m_renderFinishedSems;

    // Per-material param UBOs — allocated in uploadMeshMaterials(), freed in shutdown()
    std::vector<AllocatedBuffer> m_materialUbos;
    uint32_t m_frameIdx    = 0;
    uint32_t m_swapIdx     = 0;
    bool     m_initialized = false;
};

} // namespace vkgfx
