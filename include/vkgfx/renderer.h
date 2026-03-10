#pragma once
#include "window.h"
#include "swapchain.h"
#include "scene.h"
#include "postprocess.h"
#include "thread_pool.h"
#include <filesystem>
#include <vector>

namespace vkgfx {

struct RendererSettings {
    MSAASamples  msaa           = MSAASamples::x4;
    Vec4         clearColor     = {0.05f, 0.05f, 0.07f, 1.f};
    bool         vsync          = true;
    bool         wireframe      = false;
    bool         frustumCulling = true;
    bool         validation     = true;
    std::filesystem::path shaderDir = "shaders";
    uint32_t     workerThreads  = 0;  // 0 = auto-detect
};

struct PipelineEntry {
    VkPipeline            pipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      layout         = VK_NULL_HANDLE;
    VkDescriptorSetLayout matSetLayout   = VK_NULL_HANDLE;
    VkDescriptorSetLayout globalSetLayout= VK_NULL_HANDLE;
};

struct MeshGPUData {
    AllocatedBuffer vertexBuffer;
    AllocatedBuffer indexBuffer;
    std::array<AllocatedBuffer, MAX_FRAMES_IN_FLIGHT> matBuffers;
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> matDescSets;
    bool     initialized   = false;
    uint32_t writtenFrames = 0;
};

// ── Draw key: used to sort draw calls by pipeline to minimise state changes ──
struct DrawKey {
    uint64_t pipelineHash = 0;
    uint32_t meshId       = 0;   // tie-break to group same-mesh submeshes
    bool operator<(const DrawKey& o) const {
        if (pipelineHash != o.pipelineHash) return pipelineHash < o.pipelineHash;
        return meshId < o.meshId;
    }
};

struct DrawCall {
    DrawKey  key;
    Mesh*    mesh     = nullptr;
    uint32_t subIndex = 0;  // index into mesh->subMeshes()
};

class Renderer {
public:
    explicit Renderer(Window& window, const RendererSettings& settings = {});
    ~Renderer();

    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;

    void render(Scene& scene);
    void shutdown(Scene* scene = nullptr);

    void setWireframe(bool v)  { m_settings.wireframe = v; }
    void setClearColor(Vec4 c) { m_settings.clearColor = c; }

    void setPostProcess(const PostProcessSettings& pp);
    [[nodiscard]] const PostProcessSettings& postProcess() const { return m_ppSettings; }

    struct Stats {
        uint32_t drawCalls      = 0;
        uint32_t culledObjects  = 0;
        uint32_t totalVertices  = 0;
        float    frameTimeMs    = 0.f;
        float    fps            = 0.f;
        uint32_t workerThreads  = 0;
    };
    [[nodiscard]] const Stats& stats() const { return m_stats; }
    [[nodiscard]] const Context& context() const { return *m_ctx; }
    [[nodiscard]] std::shared_ptr<const Context> contextPtr() const { return m_ctx; }

private:
    // ── Descriptor helpers ─────────────────────────────────────────────────────
    void createDescriptorPool();
    void createGlobalDescriptorSetLayout();
    void createGlobalDescriptorSets();
    void updateGlobalDescriptors(uint32_t frameIdx);

    // ── Pipeline management ────────────────────────────────────────────────────
    PipelineEntry& getOrCreatePipeline(std::string_view shaderName,
                                        const PipelineSettings& ps,
                                        VkDescriptorSetLayout matLayout);
    PipelineEntry  createPipeline(std::string_view shaderName,
                                   const PipelineSettings& ps,
                                   VkDescriptorSetLayout matLayout);
    VkShaderModule createShaderModule(const std::filesystem::path& path);
    void destroyPipelines();

    VkDescriptorSetLayout getOrCreateMatDescLayout(std::string_view shaderName,
                                                    uint32_t texCount);
    MeshGPUData& getOrCreateMeshData(Mesh* mesh, Material* mat);
    void updateMaterialDescriptors(MeshGPUData& data, Material* mat, uint32_t frameIdx);
    void uploadMeshToGPU(Mesh& mesh);
    // NOTE: freeMeshGPUData() removed — use the deferred deletion queue instead.
    //       Destroying buffers immediately (without waiting for in-flight frames)
    //       is a GPU use-after-free. The deletion queue is the correct mechanism.

    // ── Frame recording ────────────────────────────────────────────────────────
    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIdx,
                              Scene& scene, uint32_t frameIdx);

    // Fill a sorted draw-call list from visible meshes into an existing vector
    // (caller passes m_drawListScratch to avoid per-frame allocation).
    void buildDrawList(const std::vector<Mesh*>& meshes,
                       uint32_t frameIdx,
                       std::vector<DrawCall>& out);

    // Record a batch of draw calls using a secondary command buffer (worker-thread safe)
    VkCommandBuffer recordSecondaryBatch(const std::vector<DrawCall>& batch,
                                          uint32_t frameIdx, uint32_t workerIdx);

    void drawMeshSubMesh(VkCommandBuffer cmd, Mesh& mesh, uint32_t subIdx,
                         uint32_t frameIdx, uint32_t& drawCalls);

    // ── Per-worker secondary command pool ─────────────────────────────────────
    void createWorkerCommandPools(uint32_t workerCount);
    void destroyWorkerCommandPools();

    void handleResize();

    // ── Shadow ────────────────────────────────────────────────────────────────
    struct ShadowResources {
        AllocatedImage   depthArray;
        VkImageView      arrayView = VK_NULL_HANDLE;
        VkImageView      layerViews[MAX_SHADOW_MAPS] = {};
        VkFramebuffer    framebuffers[MAX_SHADOW_MAPS] = {};
        VkSampler        sampler          = VK_NULL_HANDLE;
        VkRenderPass     renderPass       = VK_NULL_HANDLE;
        VkPipeline       pipeline         = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout   = VK_NULL_HANDLE;
        std::array<AllocatedBuffer, MAX_FRAMES_IN_FLIGHT> ubos;
        const Light*     casters[MAX_SHADOW_MAPS] = {};
        Mat4             lightSpaces[MAX_SHADOW_MAPS];
        Frustum          frustums[MAX_SHADOW_MAPS];
        int              count = 0;
    };

    void createShadowResources();
    void destroyShadowResources();
    void updateShadowData(Scene& scene, uint32_t frameIdx);
    void recordShadowPass(VkCommandBuffer cmd, Scene& scene);
    SceneUBO buildSceneUBOWithShadows(Scene& scene) const;

    // ── Post-process ──────────────────────────────────────────────────────────
    void initPostProcess();
    void shutdownPostProcess();
    void createOffscreenResources();
    void destroyOffscreenResources();
    void createPPRenderPass();
    void destroyPPRenderPass();
    void createPPFramebuffers();
    void destroyPPFramebuffers();
    void createPPDescriptorLayoutAndSets();
    void destroyPPDescriptorResources();
    void createPPPipeline();
    void destroyPPPipeline();
    void updatePPDescriptors(uint32_t frameIdx);
    void recordPPPass(VkCommandBuffer cmd, uint32_t imageIdx, uint32_t frameIdx);

    // ── Members ───────────────────────────────────────────────────────────────
    Window&                  m_window;
    RendererSettings         m_settings;
    std::shared_ptr<Context> m_ctx;
    VkSurfaceKHR             m_surface = VK_NULL_HANDLE;
    std::unique_ptr<Swapchain> m_swapchain;

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_globalSetLayout = VK_NULL_HANDLE;
    std::array<VkDescriptorSet,    MAX_FRAMES_IN_FLIGHT> m_globalSets{};
    std::array<AllocatedBuffer,    MAX_FRAMES_IN_FLIGHT> m_cameraUBOs{};
    std::array<AllocatedBuffer,    MAX_FRAMES_IN_FLIGHT> m_sceneUBOs{};

    std::unordered_map<std::string, PipelineEntry>           m_pipelineCache;
    std::unordered_map<std::string, VkDescriptorSetLayout>   m_matDescLayouts;
    VkPipelineCache m_pipelineCache_vk = VK_NULL_HANDLE;

    std::unordered_map<Mesh*, MeshGPUData> m_meshData;
    std::shared_ptr<Texture> m_whiteTexture;

    // ── Deferred deletion queue ───────────────────────────────────────────────
    struct DeferredBuffers {
        uint64_t            frameIndex;
        AllocatedBuffer     vertexBuffer;
        AllocatedBuffer     indexBuffer;
        std::array<AllocatedBuffer, MAX_FRAMES_IN_FLIGHT> matBuffers;
    };
    std::vector<DeferredBuffers> m_deletionQueue;
    uint64_t m_frameCounter = 0;

    // ── Multithreaded recording ────────────────────────────────────────────────
    std::unique_ptr<ThreadPool>   m_threadPool;
    std::vector<std::array<VkCommandPool,   MAX_FRAMES_IN_FLIGHT>> m_workerCmdPools;
    std::vector<std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT>> m_workerCmdBuffers;
    // NOTE: m_pipelineMutex removed — all pipeline/layout creation is pre-warmed
    // on the main thread before workers start, so no lock is ever needed.

    // ── Render pass tracking ──────────────────────────────────────────────────
    VkRenderPass  m_sceneRenderPass  = VK_NULL_HANDLE;

    // ── Offscreen resources ───────────────────────────────────────────────────
    AllocatedImage m_offscreenColor;
    AllocatedImage m_offscreenDepth;
    VkRenderPass   m_offscreenRenderPass = VK_NULL_HANDLE;
    VkFramebuffer  m_offscreenFramebuffer = VK_NULL_HANDLE;
    VkSampler      m_offscreenSampler = VK_NULL_HANDLE;

    // ── PP pass ───────────────────────────────────────────────────────────────
    VkRenderPass                 m_ppRenderPass  = VK_NULL_HANDLE;
    std::vector<VkFramebuffer>   m_ppFramebuffers;
    VkDescriptorSetLayout        m_ppDescSetLayout = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_ppDescSets{};
    std::array<AllocatedBuffer,  MAX_FRAMES_IN_FLIGHT> m_ppUBOs{};
    VkPipeline       m_ppPipeline       = VK_NULL_HANDLE;
    VkPipelineLayout m_ppPipelineLayout = VK_NULL_HANDLE;
    // Tracks which frame slots have had their PP image descriptor written at
    // least once. The image view never changes, so we skip redundant writes.
    uint32_t m_ppDescWrittenFrames = 0;

    ShadowResources     m_shadow;
    PostProcessSettings m_ppSettings;
    bool                m_ppActive = false;

    uint32_t m_currentFrame = 0;
    Stats    m_stats;

    // Reusable scratch buffers — pre-allocated once and reused every frame to
    // avoid per-frame heap allocations from visibleMeshes() / buildDrawList().
    std::vector<Mesh*>     m_visibleScratch;
    std::vector<DrawCall>  m_drawListScratch;
};

} // namespace vkgfx
