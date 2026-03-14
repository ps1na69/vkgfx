#pragma once
// renderer.h — Deferred shading renderer.
//
// This is the only renderer in the engine. Forward rendering has been removed.
//
// Pipeline (via RenderGraph):
//   0. Shadow Pass    — CSM depth-only pass (4 cascades)
//   1. Geometry Pass  — fills G-buffer (position, normal, albedo, material, emissive)
//   2. SSAO Pass      — hemisphere ambient occlusion in view space
//   3. SSAO Blur      — depth-aware 4x4 box blur
//   4. Lighting Pass  — Cook-Torrance PBR + IBL + CSM shadows → HDR
//   5. Tonemap Pass   — ACES filmic, gamma correction, writes to swapchain
//
// GPU culling is dispatched before geometry: a compute shader writes
// VkDrawIndexedIndirectCommand for only the visible instances, and the
// geometry pass uses vkCmdDrawIndexedIndirectCount.

#include "window.h"
#include "swapchain.h"
#include "scene.h"
#include "GBuffer.h"
#include "thread_pool.h"
#include "render_graph.h"
#include "ibl.h"
#include "shadow.h"
#include "gpu_culling.h"
#include <filesystem>

namespace vkgfx {

struct RendererSettings {
    Vec4         clearColor     = {0.f, 0.f, 0.f, 1.f};
    bool         vsync          = true;
    bool         wireframe      = false;  // geometry pass polygon mode
    bool         frustumCulling = true;
    bool         validation     = true;
    float        ssaoRadius     = 0.5f;
    float        ssaoBias       = 0.025f;
    float        exposure       = 0.f;   // EV offset for tonemap
    uint32_t     tonemapOp      = 0;     // 0=ACES, 1=Reinhard, 2=Uncharted2
    uint32_t     workerThreads  = 0;     // 0 = auto-detect
    std::filesystem::path shaderDir = "shaders";
};

// GPU data tracked per Mesh.
struct MeshGPUData {
    // Material descriptor sets (one per frame-in-flight).
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> matDescSets{};
    // Per-frame MaterialUBO buffers (persistent-mapped HOST_COHERENT).
    std::array<AllocatedBuffer, MAX_FRAMES_IN_FLIGHT> matUBOs{};
    bool     initialized   = false;
    uint32_t writtenFrames = 0;  // bitmask — which frames have received a descriptor write
};

class Renderer {
public:
    explicit Renderer(Window& window, const RendererSettings& settings = {});
    ~Renderer();

    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Render one frame: geometry → SSAO → lighting → tonemap.
    void render(Scene& scene);
    void shutdown(Scene* scene = nullptr);

    void setWireframe(bool v) { m_settings.wireframe = v; }
    void setExposure(float ev){ m_settings.exposure  = ev; }

    struct Stats {
        uint32_t drawCalls     = 0;
        uint32_t culledObjects = 0;
        float    frameTimeMs   = 0.f;
        float    fps           = 0.f;
    };
    [[nodiscard]] const Stats& stats() const { return m_stats; }
    [[nodiscard]] std::shared_ptr<const Context> contextPtr() const { return m_ctx; }

private:
    // ── Initialisation ─────────────────────────────────────────────────────────
    void createDescriptorPool();
    void createDescriptorLayouts();

    void createGeometryRenderPass();
    void createSSAORenderPass();
    void createSSAOBlurRenderPass();
    void createLightingRenderPass();

    void createGeometryPipeline();
    void createSSAOPipeline();
    void createSSAOBlurPipeline();
    void createLightingPipeline();
    void createTonemapPipeline();

    void createSizeDependentResources();
    void destroySizeDependentResources();

    void createPerFrameBuffers();
    void createSSAOKernel();
    void allocateAndWriteDescriptorSets();

    VkShaderModule createShaderModule(const std::filesystem::path& path);

    // ── Per-frame mesh data ────────────────────────────────────────────────────
    MeshGPUData& getOrCreateMeshData(Mesh* mesh);
    void updateMaterialDescriptors(MeshGPUData& data, PBRMaterial* mat, uint32_t fi);
    void uploadMeshToGPU(Mesh& mesh);

    // ── Frame recording ────────────────────────────────────────────────────────
    void recordFrame(VkCommandBuffer cmd, uint32_t imageIdx,
                     Scene& scene, uint32_t fi);

    void recordGeometryPass(VkCommandBuffer cmd, const std::vector<Mesh*>& visible, uint32_t fi);
    void recordSSAOPass     (VkCommandBuffer cmd, uint32_t fi);
    void recordSSAOBlurPass (VkCommandBuffer cmd, uint32_t fi);
    void recordLightingPass (VkCommandBuffer cmd, uint32_t fi);
    void recordTonemapPass  (VkCommandBuffer cmd, uint32_t imageIdx, uint32_t fi);

    void handleResize();

    // ── Core engine objects ────────────────────────────────────────────────────
    Window&                  m_window;
    RendererSettings         m_settings;
    std::shared_ptr<Context> m_ctx;
    VkSurfaceKHR             m_surface = VK_NULL_HANDLE;
    std::unique_ptr<Swapchain> m_swapchain;

    VkPipelineCache m_pipelineCache = VK_NULL_HANDLE;

    // ── Descriptor pool + layouts ──────────────────────────────────────────────
    VkDescriptorPool      m_descPool        = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_frameLayout     = VK_NULL_HANDLE;  // set 0: FrameUBO
    VkDescriptorSetLayout m_matLayout       = VK_NULL_HANDLE;  // set 1: 5 textures + MaterialUBO
    VkDescriptorSetLayout m_gbufLayout      = VK_NULL_HANDLE;  // 7 samplers (G-buf + SSAO)
    VkDescriptorSetLayout m_ssaoLayout      = VK_NULL_HANDLE;  // 3 samplers + SSAOParams UBO
    VkDescriptorSetLayout m_ssaoBlurLayout  = VK_NULL_HANDLE;  // 2 samplers
    VkDescriptorSetLayout m_lightLayout     = VK_NULL_HANDLE;  // light SSBO
    VkDescriptorSetLayout m_hdrLayout       = VK_NULL_HANDLE;  // HDR sampler

    // ── G-buffer ───────────────────────────────────────────────────────────────
    GBuffer m_gbuffer;

    // ── Intermediate images ────────────────────────────────────────────────────
    AllocatedImage m_ssaoRaw;     // R8_UNORM — raw SSAO
    AllocatedImage m_ssaoBlur;    // R8_UNORM — blurred SSAO
    AllocatedImage m_hdrImage;    // RGBA16_SFLOAT — HDR lighting output
    VkSampler      m_screenSampler = VK_NULL_HANDLE;  // nearest clamp, shared by screen passes

    // ── Render passes ──────────────────────────────────────────────────────────
    VkRenderPass m_geomRP      = VK_NULL_HANDLE;
    VkRenderPass m_ssaoRP      = VK_NULL_HANDLE;
    VkRenderPass m_ssaoBlurRP  = VK_NULL_HANDLE;
    VkRenderPass m_lightingRP  = VK_NULL_HANDLE;

    // ── Framebuffers (size-dependent) ─────────────────────────────────────────
    VkFramebuffer m_geomFB      = VK_NULL_HANDLE;
    VkFramebuffer m_ssaoFB      = VK_NULL_HANDLE;
    VkFramebuffer m_ssaoBlurFB  = VK_NULL_HANDLE;
    VkFramebuffer m_lightingFB  = VK_NULL_HANDLE;

    // ── Pipelines ──────────────────────────────────────────────────────────────
    VkPipeline       m_geomPipeline      = VK_NULL_HANDLE;
    VkPipelineLayout m_geomPipeLayout    = VK_NULL_HANDLE;
    VkPipeline       m_ssaoPipeline      = VK_NULL_HANDLE;
    VkPipelineLayout m_ssaoPipeLayout    = VK_NULL_HANDLE;
    VkPipeline       m_ssaoBlurPipeline  = VK_NULL_HANDLE;
    VkPipelineLayout m_ssaoBlurPipeLayout= VK_NULL_HANDLE;
    VkPipeline       m_lightPipeline     = VK_NULL_HANDLE;
    VkPipelineLayout m_lightPipeLayout   = VK_NULL_HANDLE;
    VkPipeline       m_tonemapPipeline   = VK_NULL_HANDLE;
    VkPipelineLayout m_tonemapPipeLayout = VK_NULL_HANDLE;

    // ── Per-frame GPU resources ────────────────────────────────────────────────
    struct PerFrame {
        // Camera / frame data — always updated at start of frame.
        AllocatedBuffer frameUBO;   // FrameUBO (persistent-mapped)
        // Lights — filled from Scene::buildLightBuffer each frame.
        AllocatedBuffer lightSSBO;  // LightSSBO (persistent-mapped)
        // SSAO parameters — updated every frame with current proj matrices.
        AllocatedBuffer ssaoUBO;    // SSAOParams (persistent-mapped)

        // Descriptor sets — allocated once, images updated on resize.
        VkDescriptorSet frameSet    = VK_NULL_HANDLE;  // FrameUBO
        VkDescriptorSet gbufSet     = VK_NULL_HANDLE;  // G-buffer + SSAO
        VkDescriptorSet ssaoSet     = VK_NULL_HANDLE;  // SSAO input
        VkDescriptorSet ssaoBlurSet = VK_NULL_HANDLE;  // blur input
        VkDescriptorSet lightSet    = VK_NULL_HANDLE;  // Light SSBO
        VkDescriptorSet hdrSet      = VK_NULL_HANDLE;  // HDR → tonemap
    };
    std::array<PerFrame, MAX_FRAMES_IN_FLIGHT> m_frames;

    // SSAO kernel: 32 hemisphere samples in view space.
    std::array<Vec4, 32> m_ssaoKernel;
    // 4x4 tiled rotation noise texture.
    AllocatedImage m_ssaoNoise;
    VkSampler      m_ssaoNoiseSampler = VK_NULL_HANDLE;

    // Default 1x1 white texture (fallback for absent material slots).
    std::shared_ptr<Texture> m_whiteTexture;
    // Default flat-normal texture (0.5, 0.5, 1.0, 1.0 in UNORM space).
    std::shared_ptr<Texture> m_flatNormalTexture;

    // ── Per-mesh GPU data ──────────────────────────────────────────────────────
    std::unordered_map<Mesh*, MeshGPUData> m_meshData;
    std::mutex                             m_meshDataMutex; // guards getOrCreateMeshData

    // Deferred deletion queue — buffers freed after GPU is done with them.
    struct DeferredDelete {
        uint64_t        frameIndex;
        AllocatedBuffer vertexBuffer;
        AllocatedBuffer indexBuffer;
        std::array<AllocatedBuffer, MAX_FRAMES_IN_FLIGHT> matUBOs;
    };
    std::vector<DeferredDelete> m_deletionQueue;
    uint64_t m_frameCounter = 0;

    // Multithreaded draw recording.
    std::unique_ptr<ThreadPool> m_threadPool;
    std::vector<std::array<VkCommandPool,   MAX_FRAMES_IN_FLIGHT>> m_workerPools;
    std::vector<std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT>> m_workerCmds;
    void createWorkerPools(uint32_t count);
    void destroyWorkerPools();
    VkCommandBuffer recordSecondaryBatch(const std::vector<Mesh*>& batch,
                                          uint32_t fi, uint32_t workerIdx);
    void drawMesh(VkCommandBuffer cmd, Mesh& mesh, uint32_t fi, uint32_t& dc);

    uint32_t m_currentFrame = 0;
    Stats    m_stats;
    Scene*   m_currentScene = nullptr;  // set each frame in render(), used by recordLightingPass

    // Scratch buffers reused every frame to avoid heap allocations.
    std::vector<Mesh*> m_visibleScratch;

    // ── New systems ────────────────────────────────────────────────────────────
    std::unique_ptr<RenderGraph>  m_renderGraph;
    std::unique_ptr<IBLProbe>     m_iblProbe;
    std::unique_ptr<ShadowSystem> m_shadowSystem;
    std::unique_ptr<GPUCulling>   m_gpuCulling;

    // Descriptor set layouts for new lighting pass bindings (sets 3 & 4)
    VkDescriptorSetLayout m_iblLayout    = VK_NULL_HANDLE; // set 3: irr+pf+brdfLUT
    VkDescriptorSetLayout m_shadowLayout = VK_NULL_HANDLE; // set 4: shadowArray + shadowUBO

    // Fallback descriptor sets used when IBL probe has not been loaded.
    // Always valid; point to 1×1 black cubemaps and a white 2D BRDF LUT.
    AllocatedImage m_fallbackCubemap;                                          // 1×1 black 6-layer cube
    VkImageView    m_fallbackCubeView   = VK_NULL_HANDLE;
    VkSampler      m_fallbackCubeSampler = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_fallbackIBLSets{};

    // Per-frame IBL / shadow descriptor sets
    struct PerFrameExt {
        VkDescriptorSet iblSet    = VK_NULL_HANDLE;
        VkDescriptorSet shadowSet = VK_NULL_HANDLE;
        AllocatedBuffer shadowUBO;  // ShadowUBO (persistent-mapped)
    };
    std::array<PerFrameExt, MAX_FRAMES_IN_FLIGHT> m_framesExt;

    // Render-graph texture handles (populated in createSizeDependentResources)
    RGTextureHandle m_rgHDR      = RG_NULL_HANDLE;
    RGTextureHandle m_rgSSAO     = RG_NULL_HANDLE;
    RGTextureHandle m_rgSSAOBlur = RG_NULL_HANDLE;

    // Init helpers for new systems
    void initRenderGraph();
    void initIBL(const std::filesystem::path& hdrPath = "assets/sky.hdr");
    void initShadows();
    void initGPUCulling();
    void createIBLDescriptors();
    void createShadowDescriptors();
    void createFallbackIBLDescriptors(); // creates 1×1 dummy sets for when no HDR is loaded

    // Shadow geometry draw callback
    void recordShadowDraw(VkCommandBuffer cmd, uint32_t cascadeIdx,
                          const std::vector<Mesh*>& meshes, uint32_t fi);
};

} // namespace vkgfx
