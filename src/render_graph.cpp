#include "vkgfx/render_graph.h"
#include <stdexcept>
#include <algorithm>
#include <cassert>

namespace vkgfx {

// ── RGPassBuilder ─────────────────────────────────────────────────────────────
RGPassBuilder& RGPassBuilder::reads(RGTextureHandle h) {
    m_data.reads.push_back(h); return *this;
}
RGPassBuilder& RGPassBuilder::writes(RGTextureHandle h) {
    m_data.writes.push_back(h); return *this;
}
RGPassBuilder& RGPassBuilder::writesDepth(RGTextureHandle h) {
    m_data.depthWrite = h; return *this;
}
RGPassBuilder& RGPassBuilder::execute(std::function<void(VkCommandBuffer, RenderGraph&)> fn) {
    m_data.fn = std::move(fn); return *this;
}

// ── RenderGraph ───────────────────────────────────────────────────────────────
RenderGraph::RenderGraph(std::shared_ptr<Context> ctx) : m_ctx(std::move(ctx)) {}

RenderGraph::~RenderGraph() { destroy(); }

void RenderGraph::destroy() {
    if (!m_ctx) return;
    for (auto& t : m_textures)
        if (t.allocated) m_ctx->destroyImage(t.image);
    m_textures.clear();
}

RGTextureHandle RenderGraph::createTexture(const RGTextureDesc& desc) {
    RGTexture t;
    t.desc        = desc;
    t.allocated   = false;
    m_textures.push_back(t);
    m_needsRealloc = true;
    return static_cast<RGTextureHandle>(m_textures.size() - 1);
}

void RenderGraph::begin() {
    m_passes.clear();
    // Reset layouts to UNDEFINED so barriers are inserted fresh each frame.
    // (In production you'd track persistent layouts, but for simplicity
    //  we let each pass's first barrier handle the transition.)
    for (auto& t : m_textures)
        t.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
}

RGPassBuilder& RenderGraph::addPass(const char* name) {
    m_passes.emplace_back();
    m_passes.back().data().name = name;
    return m_passes.back();
}

void RenderGraph::compile(VkExtent2D swapchainExtent) {
    bool extentChanged = (m_compiledExtent.width  != swapchainExtent.width ||
                          m_compiledExtent.height != swapchainExtent.height);
    if (extentChanged || m_needsRealloc) {
        reallocTextures(swapchainExtent);
        m_compiledExtent = swapchainExtent;
        m_needsRealloc = false;
    }
}

void RenderGraph::reallocTextures(VkExtent2D ext) {
    for (auto& t : m_textures) {
        if (t.allocated) {
            m_ctx->destroyImage(t.image);
            t.allocated = false;
        }
        uint32_t w = std::max(1u, static_cast<uint32_t>(ext.width  * t.desc.widthScale));
        uint32_t h = std::max(1u, static_cast<uint32_t>(ext.height * t.desc.heightScale));

        bool isDepth = (t.desc.format == VK_FORMAT_D32_SFLOAT ||
                        t.desc.format == VK_FORMAT_D16_UNORM ||
                        t.desc.format == VK_FORMAT_D24_UNORM_S8_UINT ||
                        t.desc.format == VK_FORMAT_D32_SFLOAT_S8_UINT);

        VkImageUsageFlags usage = t.desc.extraUsage | VK_IMAGE_USAGE_SAMPLED_BIT;
        if (isDepth) usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        else         usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        t.image = m_ctx->createImage(w, h, t.desc.mipLevels, t.desc.samples,
                                     t.desc.format, VK_IMAGE_TILING_OPTIMAL,
                                     usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                     t.desc.layers);
        t.image.mipLevels = t.desc.mipLevels;

        VkImageAspectFlags aspect = isDepth ? VK_IMAGE_ASPECT_DEPTH_BIT
                                            : VK_IMAGE_ASPECT_COLOR_BIT;
        m_ctx->createImageView(t.image, aspect);
        t.currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        t.allocated = true;
    }
}

void RenderGraph::insertBarrier(VkCommandBuffer cmd, RGTextureHandle h,
                                 VkImageLayout newLayout,
                                 VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                                 VkPipelineStageFlags dstStage, VkAccessFlags dstAccess)
{
    if (h == RG_BACKBUFFER) return; // handled externally

    auto& t = m_textures[h];
    if (t.currentLayout == newLayout) return;

    bool isDepth = (t.desc.format == VK_FORMAT_D32_SFLOAT ||
                    t.desc.format == VK_FORMAT_D16_UNORM ||
                    t.desc.format == VK_FORMAT_D24_UNORM_S8_UINT ||
                    t.desc.format == VK_FORMAT_D32_SFLOAT_S8_UINT);

    VkImageMemoryBarrier barrier{};
    barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout           = t.currentLayout;
    barrier.newLayout           = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image               = t.image.image;
    barrier.subresourceRange    = {
        static_cast<VkImageAspectFlags>(isDepth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT),
        0, t.desc.mipLevels, 0, t.desc.layers
    };
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    t.currentLayout = newLayout;
}

void RenderGraph::execute(VkCommandBuffer cmd,
                           VkImage swapchainImage, VkImageView swapchainView,
                           VkExtent2D swapchainExtent)
{
    m_backbufferImage = swapchainImage;
    m_backbufferView  = swapchainView;

    for (auto& pass : m_passes) {
        auto& data = pass.data();
        if (!data.fn) continue;

        // ── Transition READ resources to SHADER_READ_ONLY ────────────────────
        for (auto h : data.reads) {
            insertBarrier(cmd, h,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT);
        }

        // ── Transition WRITE resources to ATTACHMENT ─────────────────────────
        for (auto h : data.writes) {
            insertBarrier(cmd, h,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        }

        if (data.depthWrite != RG_NULL_HANDLE) {
            insertBarrier(cmd, data.depthWrite,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
        }

        // ── Execute pass ──────────────────────────────────────────────────────
        data.fn(cmd, *this);
    }
}

VkImageView RenderGraph::getView(RGTextureHandle h) const {
    if (h == RG_BACKBUFFER) return m_backbufferView;
    assert(h < m_textures.size() && m_textures[h].allocated);
    return m_textures[h].image.view;
}

VkImage RenderGraph::getImage(RGTextureHandle h) const {
    if (h == RG_BACKBUFFER) return m_backbufferImage;
    assert(h < m_textures.size() && m_textures[h].allocated);
    return m_textures[h].image.image;
}

AllocatedImage& RenderGraph::get(RGTextureHandle h) {
    assert(h < m_textures.size() && m_textures[h].allocated);
    return m_textures[h].image;
}

VkImageLayout RenderGraph::getLayout(RGTextureHandle h) const {
    if (h == RG_BACKBUFFER) return VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    return m_textures[h].currentLayout;
}

} // namespace vkgfx
