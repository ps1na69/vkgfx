#pragma once
// render_graph.h — Lightweight frame render graph.
//
// Usage per frame:
//   graph.begin();
//   auto pass = graph.addPass("Geometry");
//   pass.reads(hDepth).writes(hPos, hNormal, hAlbedo);
//   pass.execute([](VkCommandBuffer cmd, RenderGraph& rg) { ... });
//   graph.compile(swapchainExtent);   // idempotent if nothing changed
//   graph.execute(cmd, imageIdx);     // inserts barriers, calls execute fns
//
// Resources declared with createTexture() are allocated lazily on first compile
// and reallocated on resize. The graph owns all transient images.

#include "context.h"
#include <functional>
#include <string>

namespace vkgfx {

// ── Handle types ─────────────────────────────────────────────────────────────
using RGTextureHandle = uint32_t;
inline constexpr RGTextureHandle RG_NULL_HANDLE   = UINT32_MAX;
inline constexpr RGTextureHandle RG_BACKBUFFER    = UINT32_MAX - 1; // swapchain image

// ── Texture descriptor ────────────────────────────────────────────────────────
struct RGTextureDesc {
    VkFormat               format     = VK_FORMAT_UNDEFINED;
    // {0,0} = match swapchain extent every compile
    float                  widthScale = 1.f;  // fraction of swapchain width
    float                  heightScale= 1.f;
    uint32_t               mipLevels  = 1;
    uint32_t               layers     = 1;
    VkSampleCountFlagBits  samples    = VK_SAMPLE_COUNT_1_BIT;
    VkImageUsageFlags      extraUsage = 0;
    const char*            name       = "";
};

// ── Pass record ───────────────────────────────────────────────────────────────
class RenderGraph;
class RGPassBuilder {
public:
    RGPassBuilder& reads (RGTextureHandle h);
    RGPassBuilder& writes(RGTextureHandle h);
    // For depth: the pass writes depth and reads nothing else
    RGPassBuilder& writesDepth(RGTextureHandle h);
    RGPassBuilder& execute(std::function<void(VkCommandBuffer, RenderGraph&)> fn);

    // Internal — used by RenderGraph
    struct PassData {
        std::string name;
        std::vector<RGTextureHandle> reads;
        std::vector<RGTextureHandle> writes;
        RGTextureHandle depthWrite = RG_NULL_HANDLE;
        std::function<void(VkCommandBuffer, RenderGraph&)> fn;
    };
    PassData& data() { return m_data; }

private:
    PassData m_data;
};

// ── Compiled resource ─────────────────────────────────────────────────────────
struct RGTexture {
    AllocatedImage  image;
    VkImageLayout   currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    RGTextureDesc   desc;
    bool            allocated     = false;
};

// ── Render graph ──────────────────────────────────────────────────────────────
class RenderGraph {
public:
    explicit RenderGraph(std::shared_ptr<Context> ctx);
    ~RenderGraph();

    // ── Resource declaration (call once after construction, or on config change)
    RGTextureHandle createTexture(const RGTextureDesc& desc);

    // ── Per-frame API ─────────────────────────────────────────────────────────
    void begin(); // clear pass list, keep resources
    RGPassBuilder& addPass(const char* name);

    // Must call after all addPass() calls when extent changes or on first frame.
    void compile(VkExtent2D swapchainExtent);

    // Execute all passes with automatic barriers.
    // swapchainImage / swapchainView: the current swapchain image (for RG_BACKBUFFER).
    void execute(VkCommandBuffer cmd,
                 VkImage swapchainImage, VkImageView swapchainView,
                 VkExtent2D swapchainExtent);

    // ── Resource accessors (valid after compile) ──────────────────────────────
    [[nodiscard]] VkImageView    getView  (RGTextureHandle h) const;
    [[nodiscard]] VkImage        getImage (RGTextureHandle h) const;
    [[nodiscard]] AllocatedImage& get     (RGTextureHandle h);
    [[nodiscard]] VkImageLayout  getLayout(RGTextureHandle h) const;

    void destroy(); // free all GPU resources

private:
    void reallocTextures(VkExtent2D ext);
    void insertBarrier(VkCommandBuffer cmd, RGTextureHandle h,
                       VkImageLayout newLayout,
                       VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                       VkPipelineStageFlags dstStage, VkAccessFlags dstAccess);

    std::shared_ptr<Context>              m_ctx;
    std::vector<RGTexture>                m_textures;
    std::vector<RGPassBuilder>            m_passes;
    VkExtent2D                            m_compiledExtent{0,0};
    bool                                  m_needsRealloc = true;

    // Transient backbuffer tracking
    VkImage     m_backbufferImage = VK_NULL_HANDLE;
    VkImageView m_backbufferView  = VK_NULL_HANDLE;
};

} // namespace vkgfx
