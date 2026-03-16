// src/texture.cpp
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// VMA — context.cpp owns VMA_IMPLEMENTATION; here we only need the types
#include <vk_mem_alloc.h>

#include <vkgfx/texture.h>
#include <vkgfx/context.h>

#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <cstring>

namespace vkgfx {

// ── Texture ───────────────────────────────────────────────────────────────────

void Texture::destroy(Context& ctx) {
    if (m_sampler != VK_NULL_HANDLE)
        vkDestroySampler(ctx.device(), m_sampler, nullptr);
    ctx.destroyImage(m_img);
    m_sampler = VK_NULL_HANDLE;
}

// ── TextureCache ──────────────────────────────────────────────────────────────

TextureCache::TextureCache(Context& ctx) : m_ctx(ctx) {}

TextureCache::~TextureCache() { clear(); }

void TextureCache::clear() {
    for (auto& [path, tex] : m_cache)
        tex->destroy(m_ctx);
    m_cache.clear();
}

std::shared_ptr<Texture> TextureCache::load(const std::string& path, TextureDesc desc) {
    auto it = m_cache.find(path);
    if (it != m_cache.end()) return it->second;

    if (!std::filesystem::exists(path)) {
        std::cerr << "[vkgfx] Texture not found: " << path
                  << " — using fallback 1×1 magenta\n";
        return solid(255, 0, 255);
    }

    std::shared_ptr<Texture> tex = desc.isHDR
        ? uploadHDR(path, desc)
        : uploadLDR(path, desc);

    m_cache[path] = tex;
    return tex;
}

std::shared_ptr<Texture> TextureCache::solid(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    std::string key = "__solid_" + std::to_string(r) + "_" + std::to_string(g) +
                      "_" + std::to_string(b) + "_" + std::to_string(a);
    auto it = m_cache.find(key);
    if (it != m_cache.end()) return it->second;

    uint8_t pixels[4] = {r, g, b, a};
    auto tex = std::make_shared<Texture>();

    tex->m_img = m_ctx.allocateImage({1, 1}, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    // Upload via staging
    AllocatedBuffer staging = m_ctx.allocateBuffer(4,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    void* mapped = nullptr;
    vmaMapMemory(m_ctx.vma(), (VmaAllocation)staging.allocation, &mapped);
    std::memcpy(mapped, pixels, 4);
    vmaUnmapMemory(m_ctx.vma(), (VmaAllocation)staging.allocation);

    VkCommandBuffer cmd = m_ctx.beginOneShot();

    // UNDEFINED → TRANSFER_DST
    VkImageMemoryBarrier b1{};
    b1.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b1.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    b1.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b1.image               = tex->m_img.image;
    b1.srcAccessMask       = 0;
    b1.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    b1.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b1);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {1, 1, 1};
    vkCmdCopyBufferToImage(cmd, staging.buffer, tex->m_img.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // TRANSFER_DST → SHADER_READ
    VkImageMemoryBarrier b2 = b1;
    b2.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b2.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b2);

    m_ctx.endOneShot(cmd);
    m_ctx.destroyBuffer(staging);

    tex->m_img.view  = m_ctx.createImageView(tex->m_img.image,
                           VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
    tex->m_sampler   = makeSampler(1);

    m_cache[key] = tex;
    return tex;
}

// ── uploadLDR ─────────────────────────────────────────────────────────────────

std::shared_ptr<Texture> TextureCache::uploadLDR(const std::string& path,
                                                   const TextureDesc& desc) {
    int w, h, ch;
    stbi_uc* pixels = stbi_load(path.c_str(), &w, &h, &ch, STBI_rgb_alpha);
    if (!pixels)
        throw std::runtime_error("[vkgfx] stbi_load failed: " + path);

    auto tex          = std::make_shared<Texture>();
    VkDeviceSize size = static_cast<VkDeviceSize>(w * h * 4);
    uint32_t mips     = desc.genMips
        ? static_cast<uint32_t>(std::floor(std::log2(std::max(w, h)))) + 1
        : 1;

    VkFormat fmt = desc.format;

    tex->m_img = m_ctx.allocateImage({(uint32_t)w, (uint32_t)h}, fmt,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, mips);

    AllocatedBuffer staging = m_ctx.allocateBuffer(size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    void* mapped = nullptr;
    vmaMapMemory(m_ctx.vma(), (VmaAllocation)staging.allocation, &mapped);
    std::memcpy(mapped, pixels, static_cast<size_t>(size));
    vmaUnmapMemory(m_ctx.vma(), (VmaAllocation)staging.allocation);
    stbi_image_free(pixels);

    VkCommandBuffer cmd = m_ctx.beginOneShot();

    // UNDEFINED → TRANSFER_DST (all mips)
    VkImageMemoryBarrier b{};
    b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.image               = tex->m_img.image;
    b.srcAccessMask       = 0;
    b.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {(uint32_t)w, (uint32_t)h, 1};
    vkCmdCopyBufferToImage(cmd, staging.buffer, tex->m_img.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Generate mipmaps via blit
    int32_t mipW = w, mipH = h;
    for (uint32_t i = 1; i < mips; ++i) {
        VkImageMemoryBarrier mb{};
        mb.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        mb.image               = tex->m_img.image;
        mb.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
        mb.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        mb.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        mb.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &mb);

        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
        blit.srcOffsets[1]  = {mipW, mipH, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
        blit.dstOffsets[1]  = {mipW > 1 ? mipW / 2 : 1, mipH > 1 ? mipH / 2 : 1, 1};
        vkCmdBlitImage(cmd,
            tex->m_img.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            tex->m_img.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_LINEAR);

        mb.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        mb.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &mb);

        if (mipW > 1) mipW /= 2;
        if (mipH > 1) mipH /= 2;
    }
    // Transition last mip
    VkImageMemoryBarrier last{};
    last.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    last.image               = tex->m_img.image;
    last.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    last.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
    last.oldLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    last.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    last.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, mips - 1, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &last);

    m_ctx.endOneShot(cmd);
    m_ctx.destroyBuffer(staging);

    tex->m_img.view = m_ctx.createImageView(tex->m_img.image, fmt,
                                              VK_IMAGE_ASPECT_COLOR_BIT, mips);
    tex->m_sampler  = makeSampler(mips);
    return tex;
}

// ── uploadHDR ─────────────────────────────────────────────────────────────────

std::shared_ptr<Texture> TextureCache::uploadHDR(const std::string& path,
                                                   const TextureDesc& /*desc*/) {
    int w, h, ch;
    float* pixels = stbi_loadf(path.c_str(), &w, &h, &ch, STBI_rgb_alpha);
    if (!pixels)
        throw std::runtime_error("[vkgfx] stbi_loadf failed: " + path);

    auto tex          = std::make_shared<Texture>();
    VkDeviceSize size = static_cast<VkDeviceSize>(w * h * 4 * sizeof(float));

    tex->m_img = m_ctx.allocateImage({(uint32_t)w, (uint32_t)h},
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    AllocatedBuffer staging = m_ctx.allocateBuffer(size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    void* mapped = nullptr;
    vmaMapMemory(m_ctx.vma(), (VmaAllocation)staging.allocation, &mapped);
    std::memcpy(mapped, pixels, static_cast<size_t>(size));
    vmaUnmapMemory(m_ctx.vma(), (VmaAllocation)staging.allocation);
    stbi_image_free(pixels);

    VkCommandBuffer cmd = m_ctx.beginOneShot();

    VkImageMemoryBarrier b{};
    b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.image               = tex->m_img.image;
    b.srcAccessMask       = 0;
    b.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {(uint32_t)w, (uint32_t)h, 1};
    vkCmdCopyBufferToImage(cmd, staging.buffer, tex->m_img.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);

    m_ctx.endOneShot(cmd);
    m_ctx.destroyBuffer(staging);

    tex->m_img.view = m_ctx.createImageView(tex->m_img.image,
        VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
    tex->m_sampler  = makeSampler(1);
    return tex;
}

// ── makeSampler ───────────────────────────────────────────────────────────────

VkSampler TextureCache::makeSampler(uint32_t mipLevels) {
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_ctx.gpu(), &props);

    VkSamplerCreateInfo si{};                                 // zero-init
    si.sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter               = VK_FILTER_LINEAR;
    si.minFilter               = VK_FILTER_LINEAR;
    si.addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.anisotropyEnable        = VK_TRUE;
    si.maxAnisotropy           = props.limits.maxSamplerAnisotropy;
    si.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    si.unnormalizedCoordinates = VK_FALSE;
    si.compareEnable           = VK_FALSE;
    si.compareOp               = VK_COMPARE_OP_ALWAYS;
    si.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.mipLodBias              = 0.f;
    si.minLod                  = 0.f;
    si.maxLod                  = static_cast<float>(mipLevels);

    VkSampler s = VK_NULL_HANDLE;
    vkCreateSampler(m_ctx.device(), &si, nullptr, &s);
    return s;
}

} // namespace vkgfx
