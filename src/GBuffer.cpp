#include "vkgfx/GBuffer.h"
#include <stdexcept>

namespace vkgfx {

// G-buffer attachment formats — chosen for bandwidth efficiency.
static constexpr VkFormat GBUF_FMT_WORLD_POS = VK_FORMAT_R16G16B16A16_SFLOAT;
static constexpr VkFormat GBUF_FMT_NORMAL    = VK_FORMAT_R16G16B16A16_SFLOAT;
static constexpr VkFormat GBUF_FMT_ALBEDO    = VK_FORMAT_R8G8B8A8_UNORM;
static constexpr VkFormat GBUF_FMT_MATERIAL  = VK_FORMAT_R8G8B8A8_UNORM;
static constexpr VkFormat GBUF_FMT_EMISSIVE  = VK_FORMAT_R16G16B16A16_SFLOAT;
static constexpr VkFormat GBUF_FMT_DEPTH     = VK_FORMAT_D32_SFLOAT;

static constexpr VkFormat GBUF_COLOR_FORMATS[GBUFFER_COLOR_COUNT] = {
    GBUF_FMT_WORLD_POS,
    GBUF_FMT_NORMAL,
    GBUF_FMT_ALBEDO,
    GBUF_FMT_MATERIAL,
    GBUF_FMT_EMISSIVE,
};

void gbuffer_create(GBuffer& gb, const Context& ctx, uint32_t width, uint32_t height) {
    gb.width  = width;
    gb.height = height;

    // Color attachments — written by geometry pass, sampled by lighting pass.
    const VkImageUsageFlags colorUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    for (uint32_t i = 0; i < GBUFFER_COLOR_COUNT; ++i) {
        gb.color[i] = ctx.createImage(width, height, 1, VK_SAMPLE_COUNT_1_BIT,
            GBUF_COLOR_FORMATS[i], VK_IMAGE_TILING_OPTIMAL, colorUsage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        gb.color[i].mipLevels = 1;
        ctx.createImageView(gb.color[i], VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // Depth — also read by SSAO pass via sampler.
    gb.depth = ctx.createImage(width, height, 1, VK_SAMPLE_COUNT_1_BIT,
        GBUF_FMT_DEPTH, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    gb.depth.mipLevels = 1;
    ctx.createImageView(gb.depth, VK_IMAGE_ASPECT_DEPTH_BIT);

    // Single nearest-clamp sampler — no filtering needed for deferred reads.
    VkSamplerCreateInfo si{};
    si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter    = VK_FILTER_NEAREST;
    si.minFilter    = VK_FILTER_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.maxLod       = 1.f;
    VK_CHECK(vkCreateSampler(ctx.device(), &si, nullptr, &gb.sampler), "GBuffer sampler");
}

void gbuffer_destroy(GBuffer& gb, const Context& ctx) {
    if (gb.sampler) { vkDestroySampler(ctx.device(), gb.sampler, nullptr); gb.sampler = VK_NULL_HANDLE; }
    for (auto& img : gb.color) ctx.destroyImage(img);
    ctx.destroyImage(gb.depth);
}

void gbuffer_attachment_descs(const GBuffer& gb,
                               VkAttachmentDescription* out,
                               uint32_t* count)
{
    // 5 color attachments.
    for (uint32_t i = 0; i < GBUFFER_COLOR_COUNT; ++i) {
        out[i] = {};
        out[i].format         = GBUF_COLOR_FORMATS[i];
        out[i].samples        = VK_SAMPLE_COUNT_1_BIT;
        out[i].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        out[i].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        out[i].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        out[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        out[i].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        out[i].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    // Depth attachment.
    out[GBUFFER_COLOR_COUNT] = {};
    out[GBUFFER_COLOR_COUNT].format         = GBUF_FMT_DEPTH;
    out[GBUFFER_COLOR_COUNT].samples        = VK_SAMPLE_COUNT_1_BIT;
    out[GBUFFER_COLOR_COUNT].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    out[GBUFFER_COLOR_COUNT].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    out[GBUFFER_COLOR_COUNT].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    out[GBUFFER_COLOR_COUNT].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    out[GBUFFER_COLOR_COUNT].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    out[GBUFFER_COLOR_COUNT].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    *count = GBUFFER_COLOR_COUNT + 1;
}

} // namespace vkgfx
