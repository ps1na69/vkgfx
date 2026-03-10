// GBuffer.cpp
// Creates and destroys all Vulkan objects that make up the G-buffer.
// Each attachment is an independent VkImage (not a swapchain image) so
// the geometry pass and the lighting pass can run in the same frame without
// aliasing issues.

#include "GBuffer.h"
#include <cassert>
#include <stdexcept>
#include <cstring>

namespace vkgfx {

// ─── Internal helpers ────────────────────────────────────────────────────────

// Find a memory type index that satisfies both the required type bits and the
// required property flags.  Throws if none is found.
static uint32_t find_memory_type(VkPhysicalDevice physDev,
                                 uint32_t typeBits,
                                 VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("GBuffer: no suitable memory type found");
}

// Allocate a single VkImage with dedicated device-local memory and create
// a matching VkImageView.
static GBufferImage create_attachment(VkDevice device,
                                      VkPhysicalDevice physDev,
                                      uint32_t width,
                                      uint32_t height,
                                      VkFormat format,
                                      VkImageUsageFlags usage,
                                      VkImageAspectFlags aspect)
{
    GBufferImage img{};
    img.format = format;

    // --- Image ---
    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = format;
    ici.extent        = {width, height, 1};
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    // OPTIMAL tiling is required for GPU-side render-targets.
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = usage;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;  // Cleared by the render pass

    if (vkCreateImage(device, &ici, nullptr, &img.image) != VK_SUCCESS)
        throw std::runtime_error("GBuffer: vkCreateImage failed");

    // --- Memory ---
    VkMemoryRequirements memReq{};
    vkGetImageMemoryRequirements(device, img.image, &memReq);

    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = memReq.size;
    mai.memoryTypeIndex = find_memory_type(physDev, memReq.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &mai, nullptr, &img.memory) != VK_SUCCESS)
        throw std::runtime_error("GBuffer: vkAllocateMemory failed");

    vkBindImageMemory(device, img.image, img.memory, 0);

    // --- View ---
    VkImageViewCreateInfo vci{};
    vci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image                           = img.image;
    vci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    vci.format                          = format;
    vci.subresourceRange.aspectMask     = aspect;
    vci.subresourceRange.baseMipLevel   = 0;
    vci.subresourceRange.levelCount     = 1;
    vci.subresourceRange.baseArrayLayer = 0;
    vci.subresourceRange.layerCount     = 1;

    if (vkCreateImageView(device, &vci, nullptr, &img.view) != VK_SUCCESS)
        throw std::runtime_error("GBuffer: vkCreateImageView failed");

    return img;
}

// ─── Public API ──────────────────────────────────────────────────────────────

void gbuffer_create(GBuffer& gb,
                    VkDevice device,
                    VkPhysicalDevice physicalDevice,
                    uint32_t width,
                    uint32_t height)
{
    gb.width  = width;
    gb.height = height;

    // All color attachments are both render targets (written by geometry pass)
    // and shader inputs (sampled by lighting pass).
    constexpr VkImageUsageFlags colorUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT;

    // World-space position – 16-bit float per channel gives enough precision
    // for typical scene scales without the memory cost of RGBA32.
    gb.attachments[GBUF_WORLD_POS] = create_attachment(
        device, physicalDevice, width, height,
        VK_FORMAT_R16G16B16A16_SFLOAT, colorUsage,
        VK_IMAGE_ASPECT_COLOR_BIT);

    // Normal – also 16-bit float.  Octahedron encoding could halve this to RG16
    // at the cost of a tiny bit of shader complexity – left as an exercise.
    gb.attachments[GBUF_NORMAL] = create_attachment(
        device, physicalDevice, width, height,
        VK_FORMAT_R16G16B16A16_SFLOAT, colorUsage,
        VK_IMAGE_ASPECT_COLOR_BIT);

    // Albedo – 8 bits per channel is fine; linear values stored, sRGB gamma
    // is applied only in the final tone-map pass.
    gb.attachments[GBUF_ALBEDO] = create_attachment(
        device, physicalDevice, width, height,
        VK_FORMAT_R8G8B8A8_UNORM, colorUsage,
        VK_IMAGE_ASPECT_COLOR_BIT);

    // Material (metallic, roughness) – 8 bits each is sufficient.
    gb.attachments[GBUF_MATERIAL] = create_attachment(
        device, physicalDevice, width, height,
        VK_FORMAT_R8G8B8A8_UNORM, colorUsage,
        VK_IMAGE_ASPECT_COLOR_BIT);

    // Emissive – HDR range required, 16-bit float per channel.
    gb.attachments[GBUF_EMISSIVE] = create_attachment(
        device, physicalDevice, width, height,
        VK_FORMAT_R16G16B16A16_SFLOAT, colorUsage,
        VK_IMAGE_ASPECT_COLOR_BIT);

    // Depth – used by both geometry and lighting passes.
    // We also sample it in the SSAO pass so add SAMPLED_BIT.
    gb.depth = create_attachment(
        device, physicalDevice, width, height,
        VK_FORMAT_D32_SFLOAT,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    // Single nearest-neighbour sampler shared across all G-buffer reads.
    // No interpolation: each pixel reads its own exact texel.
    VkSamplerCreateInfo sci{};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_NEAREST;
    sci.minFilter    = VK_FILTER_NEAREST;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod       = 0.0f;

    if (vkCreateSampler(device, &sci, nullptr, &gb.sampler) != VK_SUCCESS)
        throw std::runtime_error("GBuffer: vkCreateSampler failed");
}

void gbuffer_destroy(GBuffer& gb, VkDevice device)
{
    if (gb.sampler) {
        vkDestroySampler(device, gb.sampler, nullptr);
        gb.sampler = VK_NULL_HANDLE;
    }

    auto destroy_img = [&](GBufferImage& img) {
        if (img.view)   { vkDestroyImageView(device, img.view, nullptr);   img.view   = VK_NULL_HANDLE; }
        if (img.image)  { vkDestroyImage(device, img.image, nullptr);      img.image  = VK_NULL_HANDLE; }
        if (img.memory) { vkFreeMemory(device, img.memory, nullptr);       img.memory = VK_NULL_HANDLE; }
    };

    for (auto& att : gb.attachments)
        destroy_img(att);
    destroy_img(gb.depth);
}

void gbuffer_fill_attachment_descs(const GBuffer& gb,
                                   VkAttachmentDescription2* out,
                                   uint32_t* countOut)
{
    // Helper: build a color attachment desc that clears on load and stores on store.
    auto make_color = [](VkFormat fmt) -> VkAttachmentDescription2 {
        VkAttachmentDescription2 d{};
        d.sType          = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
        d.format         = fmt;
        d.samples        = VK_SAMPLE_COUNT_1_BIT;
        // Clear the G-buffer at the start of each frame to avoid stale data.
        d.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        d.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;  // Lighting pass reads these
        d.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        d.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        d.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        // Transition to SHADER_READ_ONLY after the geometry pass so the
        // lighting pass can sample without an explicit barrier.
        d.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        return d;
    };

    for (uint32_t i = 0; i < GBUFFER_ATTACHMENT_COUNT; ++i)
        out[i] = make_color(gb.attachments[i].format);

    // Depth attachment – cleared each frame, sampled in SSAO.
    out[GBUFFER_ATTACHMENT_COUNT].sType          = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
    out[GBUFFER_ATTACHMENT_COUNT].format         = gb.depth.format;
    out[GBUFFER_ATTACHMENT_COUNT].samples        = VK_SAMPLE_COUNT_1_BIT;
    out[GBUFFER_ATTACHMENT_COUNT].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    out[GBUFFER_ATTACHMENT_COUNT].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    out[GBUFFER_ATTACHMENT_COUNT].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    out[GBUFFER_ATTACHMENT_COUNT].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    out[GBUFFER_ATTACHMENT_COUNT].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    out[GBUFFER_ATTACHMENT_COUNT].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    *countOut = GBUFFER_ATTACHMENT_COUNT + 1;
}

void gbuffer_write_descriptor_set(const GBuffer& gb,
                                  VkDevice device,
                                  VkDescriptorSet dstSet)
{
    // Each G-buffer color attachment occupies one combined-image-sampler binding.
    // The depth texture gets the last binding (used by SSAO).
    std::array<VkDescriptorImageInfo, GBUFFER_ATTACHMENT_COUNT + 1> imageInfos{};

    for (uint32_t i = 0; i < GBUFFER_ATTACHMENT_COUNT; ++i) {
        imageInfos[i].sampler     = gb.sampler;
        imageInfos[i].imageView   = gb.attachments[i].view;
        imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    // Depth binding
    imageInfos[GBUFFER_ATTACHMENT_COUNT].sampler     = gb.sampler;
    imageInfos[GBUFFER_ATTACHMENT_COUNT].imageView   = gb.depth.view;
    imageInfos[GBUFFER_ATTACHMENT_COUNT].imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    std::array<VkWriteDescriptorSet, GBUFFER_ATTACHMENT_COUNT + 1> writes{};
    for (uint32_t i = 0; i <= GBUFFER_ATTACHMENT_COUNT; ++i) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = dstSet;
        writes[i].dstBinding      = i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo      = &imageInfos[i];
    }

    vkUpdateDescriptorSets(device,
                           static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
}

} // namespace vkgfx
