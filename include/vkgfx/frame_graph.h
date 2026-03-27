#pragma once
// include/vkgfx/frame_graph.h
//
// Lightweight DAG-based frame graph.
//
// Design contract:
//   • Each render pass declares which resources it reads/writes via PassBuilder.
//   • The frame graph topologically sorts passes, culls unused ones, and
//     pre-computes VkImageMemoryBarrier2 lists for each pass.
//   • Barriers are emitted with a single vkCmdPipelineBarrier2 call (sync2,
//     Vulkan 1.3 core) before each pass that needs them.
//   • Existing render passes that handle their own layout transitions via
//     initialLayout / finalLayout should declare their writes with
//     RGStates::rpManaged() as the "required before" state — the frame graph
//     then skips barrier insertion for those resource slots.
//
// Typical per-frame usage:
//   m_frameGraph.reset();
//   buildFrameGraph(scene, frameIdx);   // registers passes + resource I/O
//   m_frameGraph.compile();             // sort, cull, build barrier lists
//   m_frameGraph.execute(cmd);          // emit barriers + invoke callbacks

#include <vulkan/vulkan.h>
#include <functional>
#include <limits>
#include <string>
#include <vector>
#include <cstdint>

// Forward-declare VMA types so we don't pull in vk_mem_alloc.h into every TU
struct VmaAllocation_T;
using VmaAllocation = VmaAllocation_T*;

namespace vkgfx {

class Context;

// ─────────────────────────────────────────────────────────────────────────────
// Opaque resource handle
// ─────────────────────────────────────────────────────────────────────────────
using RGHandle = uint32_t;
inline constexpr RGHandle kRGNull = std::numeric_limits<uint32_t>::max();

// ─────────────────────────────────────────────────────────────────────────────
// Transient image description
// ─────────────────────────────────────────────────────────────────────────────
struct RGImageDesc {
    VkFormat            format      = VK_FORMAT_UNDEFINED;
    VkExtent2D          extent      = {0, 0};
    VkImageUsageFlags   usage       = 0;
    uint32_t            mipLevels   = 1;
    uint32_t            layers      = 1;
    VkImageCreateFlags  createFlags = 0;
    VkImageAspectFlags  aspect      = VK_IMAGE_ASPECT_COLOR_BIT;
};

// ─────────────────────────────────────────────────────────────────────────────
// Resource state — layout + sync info used for barrier generation
// ─────────────────────────────────────────────────────────────────────────────
struct RGState {
    VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2        access = VK_ACCESS_2_NONE;
    VkImageLayout         layout = VK_IMAGE_LAYOUT_UNDEFINED;

    // UNDEFINED layout means "render pass handles this transition; skip FG barrier"
    [[nodiscard]] bool isRpManaged() const { return layout == VK_IMAGE_LAYOUT_UNDEFINED; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Common state presets — use these in PassBuilder declarations
// ─────────────────────────────────────────────────────────────────────────────
namespace RGStates {

    // Render pass manages the transition — frame graph emits no barrier for this slot.
    // Use for writes where the render pass's own initialLayout/finalLayout handles it.
    inline constexpr RGState rpManaged() {
        return {}; // stages=NONE, access=NONE, layout=UNDEFINED
    }

    inline constexpr RGState colorWrite() {
        return {
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        };
    }

    inline constexpr RGState depthWrite() {
        return {
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        };
    }

    inline constexpr RGState depthReadOnly() {
        return {
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
        };
    }

    inline constexpr RGState fragmentSampled() {
        return {
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        };
    }

    inline constexpr RGState allGraphicsSampled() {
        return {
            VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        };
    }

    inline constexpr RGState transferSrc() {
        return {
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        };
    }

    inline constexpr RGState transferDst() {
        return {
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        };
    }

    inline constexpr RGState present() {
        return {
            VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
            VK_ACCESS_2_NONE,
            VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        };
    }

    inline constexpr RGState computeWrite() {
        return {
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_GENERAL
        };
    }

    inline constexpr RGState computeRead() {
        return {
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL
        };
    }

} // namespace RGStates

// ─────────────────────────────────────────────────────────────────────────────
// Forward declarations
// ─────────────────────────────────────────────────────────────────────────────
class FrameGraph;
class FrameGraphResources;

// ─────────────────────────────────────────────────────────────────────────────
// PassBuilder — given to the setup lambda; used to declare resource I/O
// ─────────────────────────────────────────────────────────────────────────────
class PassBuilder {
public:
    // ── Reads ──────────────────────────────────────────────────────────────────

    // Declare a read. The frame graph inserts a barrier before this pass to
    // transition the resource into `requiredState` if it is not already there.
    // Defaults to fragmentSampled() (the most common read case).
    void read(RGHandle h, RGState requiredState = RGStates::fragmentSampled());

    // ── Writes ─────────────────────────────────────────────────────────────────

    // Full write declaration.
    // requiredBefore:
    //   State the resource must be in BEFORE the pass.
    //   Frame graph inserts a barrier to reach this state if needed.
    //   Use RGStates::rpManaged() to skip — the render pass handles it.
    // resultAfter:
    //   State the resource will be in AFTER the pass completes.
    //   Frame graph uses this to determine what barriers subsequent passes need.
    void write(RGHandle h, RGState requiredBefore, RGState resultAfter);

    // Shorthand — color attachment in a render pass where:
    //   initialLayout = VK_IMAGE_LAYOUT_UNDEFINED  (render pass clears / don't-cares)
    //   finalLayout   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    // No FG barrier inserted before; result = fragmentSampled().
    void writeColorAttachment(RGHandle h);

    // Shorthand — depth attachment in a render pass where:
    //   initialLayout = VK_IMAGE_LAYOUT_UNDEFINED (cleared)
    //   finalLayout   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    void writeDepthAttachment(RGHandle h);

    // Like writeDepthAttachment but the render pass keeps the image in
    // DEPTH_STENCIL_ATTACHMENT_OPTIMAL after (e.g. shadow map that reuses
    // the initialLayout = SHADER_READ_ONLY transition inside its own render pass).
    // requiredBefore = fragmentSampled() (the render pass's initialLayout),
    // resultAfter    = fragmentSampled() (the render pass's finalLayout).
    void writeShadowMap(RGHandle h);

private:
    friend class FrameGraph;
    uint32_t    m_passIdx = 0;
    FrameGraph* m_fg      = nullptr;
};

// ─────────────────────────────────────────────────────────────────────────────
// FrameGraphResources — passed to each pass's execute callback
// Allows the callback to query the physical VkImage/VkImageView for any handle.
// ─────────────────────────────────────────────────────────────────────────────
class FrameGraphResources {
public:
    [[nodiscard]] VkImage       image (RGHandle h) const;
    [[nodiscard]] VkImageView   view  (RGHandle h) const;
    [[nodiscard]] VkImageLayout layout(RGHandle h) const;
    [[nodiscard]] VkFormat      format(RGHandle h) const;

private:
    friend class FrameGraph;
    const FrameGraph* m_fg = nullptr;
};

// ─────────────────────────────────────────────────────────────────────────────
// FrameGraph
// ─────────────────────────────────────────────────────────────────────────────
class FrameGraph {
public:
    explicit FrameGraph(Context& ctx);
    ~FrameGraph();

    FrameGraph(const FrameGraph&)            = delete;
    FrameGraph& operator=(const FrameGraph&) = delete;

    // ── Per-frame setup ────────────────────────────────────────────────────────

    // Clear passes from the previous frame. Frees any transient resources.
    // Imported resource registrations are also cleared — re-import each frame.
    void reset();

    // ── Resource registration ──────────────────────────────────────────────────

    // Import an externally managed VkImage (persistent G-buffer, shadow map, …).
    // initialLayout: the layout the resource is in at the start of this frame.
    //   Pass VK_IMAGE_LAYOUT_UNDEFINED if the first pass's render pass handles
    //   the initial transition itself (initialLayout=UNDEFINED in the render pass).
    //   Pass the actual current layout if a subsequent pass must sample it
    //   without a barrier (e.g. shadow map pre-transitioned to SHADER_READ_ONLY).
    RGHandle importImage(const std::string& name,
                          VkImage            image,
                          VkImageView        view,
                          VkFormat           format,
                          VkImageLayout      initialLayout,
                          VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

    // Create a transient image. Allocated by the frame graph after compile(),
    // freed on the next reset().
    RGHandle createImage(const std::string& name, const RGImageDesc& desc);

    // ── Pass registration ──────────────────────────────────────────────────────

    // Register a render pass.
    //   SetupFn   : void(PassBuilder&)          — declares resource I/O
    //   ExecuteFn : void(VkCommandBuffer,
    //                    const FrameGraphResources&) — records GPU commands
    //
    // Passes are registered in the order they should execute (the topological
    // sort will verify there are no cycles and may reorder for future optimisation).
    template<typename SetupFn, typename ExecuteFn>
    void addPass(const std::string& name, SetupFn&& setup, ExecuteFn&& exec) {
        const uint32_t passIdx = static_cast<uint32_t>(m_passes.size());
        m_passes.emplace_back();
        PassNode& node = m_passes.back();
        node.name      = name;
        node.execute   = std::forward<ExecuteFn>(exec);

        PassBuilder builder;
        builder.m_passIdx = passIdx;
        builder.m_fg      = this;
        std::forward<SetupFn>(setup)(builder);
    }

    // Mark a resource as a frame output.  The pass that writes it (and all
    // transitive producers) are protected from culling.  Call once per output
    // resource (typically the HDR target or swapchain image).
    void markOutput(RGHandle h);

    // ── Compile + execute ──────────────────────────────────────────────────────

    // Compile: cull unused passes, topological sort, pre-compute barrier lists
    // for transient resources.  Must be called after all passes are registered.
    void compile();

    // Execute: record all live passes in sorted order with their pre-computed
    // barriers.  `cmd` must be a recording command buffer.
    void execute(VkCommandBuffer cmd) const;

    // Dump the compiled pass graph to stdout (call after compile()).
    void dumpGraph() const;

private:
    friend class PassBuilder;
    friend class FrameGraphResources;

    // ── Internal resource node ────────────────────────────────────────────────
    struct ResourceNode {
        std::string        name;
        bool               imported  = false;

        VkImage            image     = VK_NULL_HANDLE;
        VkImageView        view      = VK_NULL_HANDLE;
        VkFormat           format    = VK_FORMAT_UNDEFINED;
        VkImageAspectFlags aspect    = VK_IMAGE_ASPECT_COLOR_BIT;

        // Transient-only
        RGImageDesc        desc      = {};
        VmaAllocation      vmaAlloc  = nullptr;

        // Evolves during compile() as each pass is processed in sorted order
        RGState            currentState = {};
    };

    // ── Internal pass structures ──────────────────────────────────────────────
    struct ResourceUse {
        RGHandle handle;
        RGState  requiredBefore;  // frame graph ensures this before the pass
        RGState  resultAfter;     // frame graph tracks this after the pass
        bool     isWrite;
    };

    struct Barrier {
        RGHandle handle;
        RGState  src;
        RGState  dst;
    };

    struct PassNode {
        std::string              name;
        std::vector<ResourceUse> uses;
        std::vector<Barrier>     barriers;  // pre-computed by buildBarriers()
        std::function<void(VkCommandBuffer, const FrameGraphResources&)> execute;
        bool     culled   = false;
        uint32_t refCount = 0;     // for culling pass
    };

    // ── Members ───────────────────────────────────────────────────────────────
    Context&                  m_ctx;
    std::vector<ResourceNode> m_resources;
    std::vector<PassNode>     m_passes;
    std::vector<uint32_t>     m_sortedOrder;  // live pass indices in execution order
    std::vector<RGHandle>     m_outputs;

    // ── Compilation steps ─────────────────────────────────────────────────────
    void cullPasses();
    void topologicalSort();
    void allocateTransients();
    void buildBarriers();
    void freeTransients();

    static void emitBarriers(VkCommandBuffer                  cmd,
                              const std::vector<Barrier>&       barriers,
                              const std::vector<ResourceNode>&  resources);
};

} // namespace vkgfx
