// src/frame_graph.cpp
// Implements FrameGraph: topological sort, pass culling, barrier pre-computation,
// transient resource allocation and execution.

#include <vkgfx/frame_graph.h>
#include <vkgfx/context.h>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace vkgfx {

// ─────────────────────────────────────────────────────────────────────────────
// PassBuilder
// ─────────────────────────────────────────────────────────────────────────────

void PassBuilder::read(RGHandle h, RGState requiredState) {
    m_fg->m_passes[m_passIdx].uses.push_back({
        h,
        requiredState,
        {},       // resultAfter unused for reads
        false
    });
}

void PassBuilder::write(RGHandle h, RGState requiredBefore, RGState resultAfter) {
    m_fg->m_passes[m_passIdx].uses.push_back({
        h,
        requiredBefore,
        resultAfter,
        true
    });
}

void PassBuilder::writeColorAttachment(RGHandle h) {
    // Render pass: initialLayout=UNDEFINED → attachment → finalLayout=SHADER_READ_ONLY_OPTIMAL.
    // Frame graph skips the barrier before (rpManaged) and tracks result as fragmentSampled.
    write(h, RGStates::rpManaged(), RGStates::fragmentSampled());
}

void PassBuilder::writeDepthAttachment(RGHandle h) {
    // Same pattern: UNDEFINED → depth → SHADER_READ_ONLY (finalLayout).
    write(h, RGStates::rpManaged(), RGStates::fragmentSampled());
}

void PassBuilder::writeShadowMap(RGHandle h) {
    // Shadow pass: initialLayout=SHADER_READ_ONLY_OPTIMAL (pre-transitioned at init),
    //              finalLayout=SHADER_READ_ONLY_OPTIMAL.
    // The resource is always in fragmentSampled state before and after.
    // The render pass itself handles SHADER_READ_ONLY → DEPTH_STENCIL → SHADER_READ_ONLY
    // internally, so the frame graph only needs to verify the resource is already in
    // SHADER_READ_ONLY before the pass.  Since it was imported in that state, no barrier.
    write(h, RGStates::fragmentSampled(), RGStates::fragmentSampled());
}

// ─────────────────────────────────────────────────────────────────────────────
// FrameGraphResources
// ─────────────────────────────────────────────────────────────────────────────

VkImage       FrameGraphResources::image (RGHandle h) const { return m_fg->m_resources[h].image; }
VkImageView   FrameGraphResources::view  (RGHandle h) const { return m_fg->m_resources[h].view; }
VkImageLayout FrameGraphResources::layout(RGHandle h) const { return m_fg->m_resources[h].currentState.layout; }
VkFormat      FrameGraphResources::format(RGHandle h) const { return m_fg->m_resources[h].format; }

// ─────────────────────────────────────────────────────────────────────────────
// FrameGraph — construction / destruction
// ─────────────────────────────────────────────────────────────────────────────

FrameGraph::FrameGraph(Context& ctx) : m_ctx(ctx) {}

FrameGraph::~FrameGraph() {
    freeTransients();
}

void FrameGraph::reset() {
    freeTransients();
    m_resources.clear();
    m_passes.clear();
    m_sortedOrder.clear();
    m_outputs.clear();
}

// ─────────────────────────────────────────────────────────────────────────────
// Resource registration
// ─────────────────────────────────────────────────────────────────────────────

RGHandle FrameGraph::importImage(const std::string& name,
                                   VkImage            image,
                                   VkImageView        view,
                                   VkFormat           format,
                                   VkImageLayout      initialLayout,
                                   VkImageAspectFlags aspect) {
    const RGHandle h = static_cast<RGHandle>(m_resources.size());
    ResourceNode node;
    node.name             = name;
    node.imported         = true;
    node.image            = image;
    node.view             = view;
    node.format           = format;
    node.aspect           = aspect;
    node.currentState.layout = initialLayout;
    m_resources.push_back(std::move(node));
    return h;
}

RGHandle FrameGraph::createImage(const std::string& name, const RGImageDesc& desc) {
    const RGHandle h = static_cast<RGHandle>(m_resources.size());
    ResourceNode node;
    node.name             = name;
    node.imported         = false;
    node.desc             = desc;
    node.format           = desc.format;
    node.aspect           = desc.aspect;
    node.currentState.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    m_resources.push_back(std::move(node));
    return h;
}

void FrameGraph::markOutput(RGHandle h) {
    m_outputs.push_back(h);
}

// ─────────────────────────────────────────────────────────────────────────────
// Transient resource management
// ─────────────────────────────────────────────────────────────────────────────

void FrameGraph::allocateTransients() {
    for (auto& res : m_resources) {
        if (res.imported || res.image != VK_NULL_HANDLE) continue;

        AllocatedImage img = m_ctx.allocateImage(
            res.desc.extent,
            res.desc.format,
            res.desc.usage,
            res.desc.mipLevels,
            res.desc.layers,
            res.desc.createFlags);

        img.view = m_ctx.createImageView(
            img.image,
            res.desc.format,
            res.desc.aspect,
            res.desc.mipLevels,
            res.desc.layers);

        res.image    = img.image;
        res.view     = img.view;
        res.vmaAlloc = static_cast<VmaAllocation>(img.allocation);
    }
}

void FrameGraph::freeTransients() {
    for (auto& res : m_resources) {
        if (res.imported || res.image == VK_NULL_HANDLE) continue;

        if (res.view != VK_NULL_HANDLE) {
            vkDestroyImageView(m_ctx.device(), res.view, nullptr);
            res.view = VK_NULL_HANDLE;
        }
        if (res.vmaAlloc) {
            vmaDestroyImage(m_ctx.vma(), res.image, res.vmaAlloc);
            res.image    = VK_NULL_HANDLE;
            res.vmaAlloc = nullptr;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass culling — backwards reachability from output resources
// ─────────────────────────────────────────────────────────────────────────────

void FrameGraph::cullPasses() {
    const uint32_t numPasses = static_cast<uint32_t>(m_passes.size());

    // Build resource → writer list
    std::vector<std::vector<uint32_t>> writers(m_resources.size());
    for (uint32_t pi = 0; pi < numPasses; ++pi)
        for (const auto& use : m_passes[pi].uses)
            if (use.isWrite)
                writers[use.handle].push_back(pi);

    // BFS backwards from output resources
    std::vector<bool> alive(numPasses, false);
    std::queue<uint32_t> workQ;

    auto markAlive = [&](uint32_t pi) {
        if (!alive[pi]) { alive[pi] = true; workQ.push(pi); }
    };

    for (RGHandle out : m_outputs)
        for (uint32_t pi : writers[out])
            markAlive(pi);

    while (!workQ.empty()) {
        const uint32_t pi = workQ.front(); workQ.pop();
        for (const auto& use : m_passes[pi].uses)
            if (!use.isWrite)
                for (uint32_t wp : writers[use.handle])
                    markAlive(wp);
    }

    for (uint32_t pi = 0; pi < numPasses; ++pi)
        m_passes[pi].culled = !alive[pi];
}

// ─────────────────────────────────────────────────────────────────────────────
// Topological sort — Kahn's algorithm
//
// Edges: A → B means "B depends on A" (A writes a resource that B reads).
// Passes are added in declaration order, which is typically already correct.
// The sort validates there are no cycles.
// ─────────────────────────────────────────────────────────────────────────────

void FrameGraph::topologicalSort() {
    const uint32_t n = static_cast<uint32_t>(m_passes.size());

    // Track last writer per resource as we scan passes in declaration order
    std::unordered_map<RGHandle, uint32_t> lastWriterOf;

    // Adjacency list and in-degree, deduplicating edges with a set
    std::vector<std::vector<uint32_t>> adj(n);
    std::vector<uint32_t>              inDegree(n, 0);
    // edge set to avoid counting duplicate dependencies
    std::unordered_set<uint64_t>       edgeSet;

    for (uint32_t pi = 0; pi < n; ++pi) {
        if (m_passes[pi].culled) continue;

        for (const auto& use : m_passes[pi].uses) {
            if (use.isWrite) {
                // Record this pass as the last writer of this resource
                lastWriterOf[use.handle] = pi;
            } else {
                // This pass reads use.handle — find its writer and add an edge
                auto it = lastWriterOf.find(use.handle);
                if (it != lastWriterOf.end()) {
                    const uint32_t writer = it->second;
                    if (writer == pi) continue; // self-loop, skip
                    // Deduplicate
                    const uint64_t key = (static_cast<uint64_t>(writer) << 32) | pi;
                    if (edgeSet.insert(key).second) {
                        adj[writer].push_back(pi);
                        ++inDegree[pi];
                    }
                }
            }
        }
    }

    // Kahn's BFS
    std::queue<uint32_t> ready;
    for (uint32_t i = 0; i < n; ++i)
        if (!m_passes[i].culled && inDegree[i] == 0)
            ready.push(i);

    m_sortedOrder.clear();
    m_sortedOrder.reserve(n);

    while (!ready.empty()) {
        const uint32_t curr = ready.front(); ready.pop();
        m_sortedOrder.push_back(curr);
        for (uint32_t next : adj[curr])
            if (--inDegree[next] == 0)
                ready.push(next);
    }

    // Cycle detection
    const uint32_t livePasses = static_cast<uint32_t>(
        std::count_if(m_passes.begin(), m_passes.end(),
                      [](const PassNode& p){ return !p.culled; }));

    if (static_cast<uint32_t>(m_sortedOrder.size()) != livePasses)
        throw std::runtime_error("[FrameGraph] Cycle detected in render pass dependency graph");
}

// ─────────────────────────────────────────────────────────────────────────────
// Barrier pre-computation
//
// Walk passes in sorted order, tracking each resource's "current state".
// For each pass use that has a non-rpManaged required state, emit a barrier
// if the current state differs from the required state.
// After the pass, update the resource's tracked state to resultAfter.
// ─────────────────────────────────────────────────────────────────────────────

void FrameGraph::buildBarriers() {
    // Resource states were set during registration (importImage / createImage).
    // We walk in sorted execution order.

    for (const uint32_t pi : m_sortedOrder) {
        PassNode& pass = m_passes[pi];
        pass.barriers.clear();

        for (const auto& use : pass.uses) {
            ResourceNode& res          = m_resources[use.handle];
            const RGState& requiredBefore = use.requiredBefore;

            // If the pass declares rpManaged, the render pass handles its own
            // transition — skip frame-graph barrier for this slot.
            if (!requiredBefore.isRpManaged()) {
                const bool layoutDiffers = (res.currentState.layout != requiredBefore.layout);
                // Also need a barrier if source access is non-zero (write-after-write,
                // write-after-read, read-after-write hazards).
                const bool needsSync =
                    layoutDiffers ||
                    (res.currentState.access != VK_ACCESS_2_NONE &&
                     res.currentState.access != requiredBefore.access);

                if (needsSync) {
                    Barrier b;
                    b.handle = use.handle;
                    b.src    = res.currentState;
                    b.dst    = requiredBefore;
                    pass.barriers.push_back(b);
                }
            }

            // Update tracked state after this pass:
            if (use.isWrite) {
                if (!use.resultAfter.isRpManaged())
                    res.currentState = use.resultAfter;
                // If resultAfter is rpManaged, we don't update — caller must ensure
                // the render pass really does leave the resource in the expected state.
            } else {
                // Read: layout stays the same; update stages/access for subsequent
                // WAR (write-after-read) hazard detection.
                if (!requiredBefore.isRpManaged())
                    res.currentState = requiredBefore;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// compile
// ─────────────────────────────────────────────────────────────────────────────

void FrameGraph::compile() {
    if (m_passes.empty()) return;

    // Cull passes not reachable from outputs. If no outputs are declared,
    // keep all passes (e.g. when the caller doesn't use culling).
    if (!m_outputs.empty())
        cullPasses();
    else
        for (auto& p : m_passes) p.culled = false;

    topologicalSort();
    allocateTransients();
    buildBarriers();
}

// ─────────────────────────────────────────────────────────────────────────────
// emitBarriers — batch all pre-computed barriers into one vkCmdPipelineBarrier2
// ─────────────────────────────────────────────────────────────────────────────

void FrameGraph::emitBarriers(VkCommandBuffer                  cmd,
                               const std::vector<Barrier>&       barriers,
                               const std::vector<ResourceNode>&  resources) {
    if (barriers.empty()) return;

    std::vector<VkImageMemoryBarrier2> imgBarriers;
    imgBarriers.reserve(barriers.size());

    for (const Barrier& b : barriers) {
        const ResourceNode& res = resources[b.handle];
        if (res.image == VK_NULL_HANDLE) continue;

        VkImageMemoryBarrier2 ib{};
        ib.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        // If src has no stage mask, use TOP_OF_PIPE (resource is in initial / unknown state)
        ib.srcStageMask        = b.src.stages ? b.src.stages
                                              : VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        ib.srcAccessMask       = b.src.access;
        ib.dstStageMask        = b.dst.stages;
        ib.dstAccessMask       = b.dst.access;
        ib.oldLayout           = b.src.layout;
        ib.newLayout           = b.dst.layout;
        ib.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.image               = res.image;
        ib.subresourceRange    = {
            res.aspect,
            0, VK_REMAINING_MIP_LEVELS,
            0, VK_REMAINING_ARRAY_LAYERS
        };
        imgBarriers.push_back(ib);
    }

    if (imgBarriers.empty()) return;

    VkDependencyInfo dep{};
    dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = static_cast<uint32_t>(imgBarriers.size());
    dep.pImageMemoryBarriers    = imgBarriers.data();
    vkCmdPipelineBarrier2(cmd, &dep);
}

// ─────────────────────────────────────────────────────────────────────────────
// execute
// ─────────────────────────────────────────────────────────────────────────────

void FrameGraph::execute(VkCommandBuffer cmd) const {
    FrameGraphResources res;
    res.m_fg = this;

    for (const uint32_t pi : m_sortedOrder) {
        const PassNode& pass = m_passes[pi];
        if (pass.culled) continue;

        // Emit pre-computed barriers for this pass
        emitBarriers(cmd, pass.barriers, m_resources);

        // Execute the pass callback
        if (pass.execute)
            pass.execute(cmd, res);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// dumpGraph
// ─────────────────────────────────────────────────────────────────────────────

void FrameGraph::dumpGraph() const {
    std::cout << "=== FrameGraph — " << m_sortedOrder.size() << " live pass(es) ===\n";
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_sortedOrder.size()); ++i) {
        const uint32_t  pi   = m_sortedOrder[i];
        const PassNode& pass = m_passes[pi];
        std::cout << "  [" << i << "] " << pass.name;
        if (pass.culled) { std::cout << " [CULLED]\n"; continue; }
        std::cout << " — " << pass.uses.size() << " resource use(s), "
                  << pass.barriers.size() << " FG barrier(s)\n";
        for (const auto& use : pass.uses) {
            const auto& res = m_resources[use.handle];
            std::cout << "    " << (use.isWrite ? "W" : "R") << " '" << res.name << "'";
            if (!use.requiredBefore.isRpManaged())
                std::cout << " layout=" << use.requiredBefore.layout;
            else
                std::cout << " [rp-managed]";
            if (use.isWrite && !use.resultAfter.isRpManaged())
                std::cout << " → " << use.resultAfter.layout;
            std::cout << "\n";
        }
        for (const auto& b : pass.barriers) {
            std::cout << "    BARRIER '" << m_resources[b.handle].name
                      << "' " << b.src.layout << " → " << b.dst.layout << "\n";
        }
    }
    std::cout << "=== end FrameGraph ===\n";
}

} // namespace vkgfx
