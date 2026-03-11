#pragma once
// GBuffer.h — G-buffer attachment management for the deferred geometry pass.
//
// Layout (all DEVICE_LOCAL):
//   [0] WorldPos   RGBA16_SFLOAT  — world-space position
//   [1] Normal     RGBA16_SFLOAT  — world-space normal (after normal mapping)
//   [2] Albedo     RGBA8_UNORM    — base color (rgb) + baked AO (a)
//   [3] Material   RGBA8_UNORM    — metallic (r), roughness (g)
//   [4] Emissive   RGBA16_SFLOAT  — HDR emissive radiance
//   [D] Depth      D32_SFLOAT     — depth, also read by SSAO pass
//
// Bandwidth note: WorldPos costs 8 B/px. On bandwidth-constrained hardware,
// replace with depth reconstruction via invProj (saves one attachment).

#include "context.h"
#include <array>
#include <cstdint>

namespace vkgfx {

inline constexpr uint32_t GBUFFER_COLOR_COUNT = 5;

enum GBufferSlot : uint32_t {
    GBUF_WORLD_POS = 0,
    GBUF_NORMAL    = 1,
    GBUF_ALBEDO    = 2,
    GBUF_MATERIAL  = 3,
    GBUF_EMISSIVE  = 4,
};

struct GBuffer {
    std::array<AllocatedImage, GBUFFER_COLOR_COUNT> color;
    AllocatedImage depth;
    VkSampler      sampler = VK_NULL_HANDLE;  // nearest clamp, shared by all attachments
    uint32_t       width   = 0;
    uint32_t       height  = 0;

    // Convenience accessors.
    [[nodiscard]] VkImageView colorView(uint32_t slot) const { return color[slot].view; }
    [[nodiscard]] VkImageView depthView()               const { return depth.view; }
};

// Allocate all G-buffer images at width×height.
// Call gbuffer_destroy() first if replacing an existing G-buffer (e.g. resize).
void gbuffer_create(GBuffer& gb, const Context& ctx, uint32_t width, uint32_t height);

// Free all Vulkan objects owned by gb.
void gbuffer_destroy(GBuffer& gb, const Context& ctx);

// Append attachment descriptions for a VkRenderPassCreateInfo.
// Outputs GBUFFER_COLOR_COUNT color descriptions + 1 depth description.
// finalLayout of color attachments: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
// finalLayout of depth attachment:  VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
void gbuffer_attachment_descs(const GBuffer& gb,
                               VkAttachmentDescription* out,
                               uint32_t* count);

} // namespace vkgfx
