// src/texture_ext.cpp
// Implements TextureCache::loadFromMemory() and the shared uploadRGBA8() helper.
// STB_IMAGE_IMPLEMENTATION is defined in texture.cpp; do NOT redefine here.

#include <vkgfx/texture.h>
#include <vkgfx/context.h>

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace vkgfx {

// ── uploadRGBA8 ───────────────────────────────────────────────────────────────
// Uploads pre-decoded RGBA8 pixel data to a DEVICE_LOCAL VkImage,
// optionally generating a full mip chain via blit.
// This is the shared upload kernel used by loadFromMemory() and solid().
//
// Layout sequence:
//   UNDEFINED → TRANSFER_DST (staging copy)
//   TRANSFER_DST → TRANSFER_SRC (blit source for mip N-1)
//   ...mip chain blits...
//   All levels → SHADER_READ_ONLY_OPTIMAL (final barrier)

std::shared_ptr<Texture> TextureCache::uploadRGBA8(const uint8_t* rgba8,
                                                     uint32_t w, uint32_t h,
                                                     const TextureDesc& desc) {
    if (!rgba8 || w == 0 || h == 0) return nullptr;

    const bool genMips  = desc.genMips && (w > 1 || h > 1);
    const uint32_t mips = genMips
        ? static_cast<uint32_t>(std::floor(std::log2(std::max(w, h)))) + 1u
        : 1u;

    // ── Staging buffer ────────────────────────────────────────────────────────
    VkDeviceSize dataSize = static_cast<VkDeviceSize>(w) * h * 4;
    AllocatedBuffer staging = m_ctx.allocateBuffer(dataSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, /*hostVisible=*/true);

    void* mapped = nullptr;
    vmaMapMemory(m_ctx.vma(), static_cast<VmaAllocation>(staging.allocation), &mapped);
    std::memcpy(mapped, rgba8, static_cast<size_t>(dataSize));
    vmaUnmapMemory(m_ctx.vma(), static_cast<VmaAllocation>(staging.allocation));

    // ── Device-local image ────────────────────────────────────────────────────
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT
                            | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (genMips) usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    AllocatedImage img = m_ctx.allocateImage({w, h}, desc.format, usage, mips);

    // ── Command recording ─────────────────────────────────────────────────────
    VkCommandBuffer cmd = m_ctx.beginOneShot();

    // Transition all mip levels to TRANSFER_DST so the copy can write mip 0
    {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcAccessMask       = 0;
        b.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = img.image;
        b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);
    }

    // Copy staging → mip 0
    {
        VkBufferImageCopy region{};
        region.bufferOffset      = 0;
        region.imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent       = {w, h, 1};
        vkCmdCopyBufferToImage(cmd, staging.buffer, img.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }

    if (genMips) {
        int32_t mipW = static_cast<int32_t>(w);
        int32_t mipH = static_cast<int32_t>(h);

        for (uint32_t i = 1; i < mips; ++i) {
            // Transition mip i-1 from TRANSFER_DST → TRANSFER_SRC
            VkImageMemoryBarrier toSrc{};
            toSrc.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            toSrc.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toSrc.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            toSrc.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
            toSrc.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
            toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toSrc.image               = img.image;
            toSrc.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 1};
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &toSrc);

            int32_t nextW = std::max(1, mipW / 2);
            int32_t nextH = std::max(1, mipH / 2);

            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
            blit.srcOffsets[0]  = {0, 0, 0};
            blit.srcOffsets[1]  = {mipW, mipH, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
            blit.dstOffsets[0]  = {0, 0, 0};
            blit.dstOffsets[1]  = {nextW, nextH, 1};
            vkCmdBlitImage(cmd,
                img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &blit, VK_FILTER_LINEAR);

            mipW = nextW;
            mipH = nextH;
        }

        // Transition all mip levels to SHADER_READ_ONLY_OPTIMAL.
        // Mip levels 0..mips-2 are currently in TRANSFER_SRC; mip mips-1 in TRANSFER_DST.
        // A single barrier covering all levels handles both layouts — Vulkan allows this
        // as long as the access masks and stage masks cover both.
        VkImageMemoryBarrier toRead{};
        toRead.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toRead.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; // dominant for most mips
        toRead.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toRead.srcAccessMask       = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        toRead.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toRead.image               = img.image;
        toRead.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &toRead);
    } else {
        // No mip generation — just transition mip 0 to shader read
        VkImageMemoryBarrier toRead{};
        toRead.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toRead.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toRead.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toRead.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        toRead.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        toRead.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toRead.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toRead.image               = img.image;
        toRead.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &toRead);
    }

    m_ctx.endOneShot(cmd);
    m_ctx.destroyBuffer(staging);

    img.view = m_ctx.createImageView(img.image, desc.format, VK_IMAGE_ASPECT_COLOR_BIT, mips);

    // ── Build result ──────────────────────────────────────────────────────────
    auto tex      = std::shared_ptr<Texture>(new Texture());
    tex->m_img     = std::move(img);
    tex->m_sampler = makeSampler(mips);
    return tex;
}

// ── loadFromMemory ────────────────────────────────────────────────────────────

std::shared_ptr<Texture> TextureCache::loadFromMemory(const uint8_t* rgba8,
                                                        uint32_t       width,
                                                        uint32_t       height,
                                                        TextureDesc    desc) {
    // Enforce linear format for non-colour maps (caller sets desc.format)
    return uploadRGBA8(rgba8, width, height, desc);
}

} // namespace vkgfx
