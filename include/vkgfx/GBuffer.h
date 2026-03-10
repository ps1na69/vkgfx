#pragma once
// GBuffer.h
// Manages all G-buffer attachments for the deferred shading geometry pass.
// Each attachment stores a different surface property written by the geometry
// pass and later sampled by the lighting pass.
//
// Layout:
//   Attachment 0 – WorldPos   : RGBA16_SFLOAT  (xyz = world-space pos, w unused)
//   Attachment 1 – Normal     : RGBA16_SFLOAT  (xyz = world-space normal after normal-map, w unused)
//   Attachment 2 – Albedo     : RGBA8_UNORM    (rgb = base color, a = occlusion baked)
//   Attachment 3 – Material   : RGBA8_UNORM    (r = metallic, g = roughness, ba unused)
//   Attachment 4 – Emissive   : RGBA16_SFLOAT  (rgb = emissive radiance – HDR range needed)
//   Depth        – Depth      : D32_SFLOAT     (standard reverse-Z or forward depth)
//
// BANDWIDTH NOTE: RGBA16 for position and normal costs ~8 bytes/pixel each.
// On mobile or bandwidth-constrained devices, replace WorldPos with a depth
// reconstruction approach (store only depth + reconstruct from inv-proj).

#include <vulkan/vulkan.h>
#include <array>
#include <cstdint>

namespace vkgfx {

// Number of color attachments in the G-buffer (not counting the depth attachment).
static constexpr uint32_t GBUFFER_ATTACHMENT_COUNT = 5;

// Indices into the GBuffer::attachments array – avoids magic numbers everywhere.
enum GBufferAttachment : uint32_t {
    GBUF_WORLD_POS  = 0,  // World-space position (or reconstruct from depth if optimising)
    GBUF_NORMAL     = 1,  // World-space normal (normal-mapped)
    GBUF_ALBEDO     = 2,  // Base color + baked AO in alpha
    GBUF_MATERIAL   = 3,  // Metallic (r), Roughness (g)
    GBUF_EMISSIVE   = 4,  // Emissive HDR radiance
};

// One Vulkan image + view + memory block.
struct GBufferImage {
    VkImage        image      = VK_NULL_HANDLE;
    VkImageView    view       = VK_NULL_HANDLE;
    VkDeviceMemory memory     = VK_NULL_HANDLE;
    VkFormat       format     = VK_FORMAT_UNDEFINED;
};

// Full G-buffer: all color images + depth + a single sampler for the lighting pass.
struct GBuffer {
    std::array<GBufferImage, GBUFFER_ATTACHMENT_COUNT> attachments{};
    GBufferImage   depth{};
    VkSampler      sampler    = VK_NULL_HANDLE;   // Nearest, clamp-to-edge – no filtering needed
    uint32_t       width      = 0;
    uint32_t       height     = 0;
};

// ─── Public interface ────────────────────────────────────────────────────────

// Allocate all G-buffer images at the given resolution.
// Call once at startup and again if the swapchain is resized.
void gbuffer_create(GBuffer& gb,
                    VkDevice device,
                    VkPhysicalDevice physicalDevice,
                    uint32_t width,
                    uint32_t height);

// Free all Vulkan objects owned by gb.
void gbuffer_destroy(GBuffer& gb, VkDevice device);

// Return a list of VkAttachmentDescription2 entries suitable for a render-pass
// creation that writes into the G-buffer (geometry pass).
// attachmentsOut must point to an array of at least GBUFFER_ATTACHMENT_COUNT+1
// (the +1 is the depth attachment).
void gbuffer_fill_attachment_descs(const GBuffer& gb,
                                   VkAttachmentDescription2* attachmentsOut,
                                   uint32_t* countOut);

// Fill a descriptor set with the G-buffer images so the lighting pass can read them.
// layout must have GBUFFER_ATTACHMENT_COUNT+1 combined-image-sampler bindings.
void gbuffer_write_descriptor_set(const GBuffer& gb,
                                  VkDevice device,
                                  VkDescriptorSet dstSet);

} // namespace vkgfx
