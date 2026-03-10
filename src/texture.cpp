#include "vkgfx/texture.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <cmath>

namespace vkgfx {

std::shared_ptr<Texture> Texture::fromFile(std::shared_ptr<const Context> ctx,
                                            const std::filesystem::path& path,
                                            const TextureSettings& settings)
{
    auto tex = std::shared_ptr<Texture>(new Texture(ctx));
    tex->m_path = path;

    int w, h, channels;
    stbi_uc* pixels = stbi_load(path.string().c_str(), &w, &h, &channels, STBI_rgb_alpha);
    if (!pixels) throw std::runtime_error("[VKGFX] Failed to load texture: " + path.string());

    tex->m_width  = static_cast<uint32_t>(w);
    tex->m_height = static_cast<uint32_t>(h);
    VkDeviceSize size = static_cast<VkDeviceSize>(w * h * 4);

    uint32_t mips = settings.generateMipmaps
        ? static_cast<uint32_t>(std::floor(std::log2(std::max(w, h)))) + 1
        : 1;

    // Staging buffer — created with VMA_ALLOCATION_CREATE_MAPPED_BIT so
    // staging.mapped is already valid; no vkMapMemory/vkUnmapMemory needed.
    auto staging = ctx->createBuffer(size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(staging.mapped, pixels, static_cast<size_t>(size));
    stbi_image_free(pixels);

    // Create device image
    tex->m_image = ctx->createImage(
        tex->m_width, tex->m_height, mips,
        VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_SRGB,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |  // needed for mipmap blitting
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    tex->m_image.mipLevels = mips;

    ctx->transitionImageLayout(tex->m_image.image, VK_FORMAT_R8G8B8A8_SRGB,
                               VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mips);
    ctx->copyBufferToImage(staging.buffer, tex->m_image.image, tex->m_width, tex->m_height);

    if (mips > 1)
        ctx->generateMipmaps(tex->m_image.image, VK_FORMAT_R8G8B8A8_SRGB,
                             tex->m_width, tex->m_height, mips);
    else
        ctx->transitionImageLayout(tex->m_image.image, VK_FORMAT_R8G8B8A8_SRGB,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);

    ctx->destroyBuffer(staging);
    ctx->createImageView(tex->m_image, VK_IMAGE_ASPECT_COLOR_BIT);
    tex->createSampler(settings);

    std::cout << "[VKGFX] Loaded texture: " << path.filename().string()
              << " (" << w << "x" << h << ", " << mips << " mips)\n";
    return tex;
}

std::shared_ptr<Texture> Texture::fromColor(std::shared_ptr<const Context> ctx,
                                             Vec4 color,
                                             const TextureSettings& settings)
{
    uint8_t rgba[4] = {
        static_cast<uint8_t>(color.r * 255.f),
        static_cast<uint8_t>(color.g * 255.f),
        static_cast<uint8_t>(color.b * 255.f),
        static_cast<uint8_t>(color.a * 255.f),
    };
    auto tex = std::shared_ptr<Texture>(new Texture(ctx));
    tex->m_width = tex->m_height = 1;

    auto staging = ctx->createBuffer(4,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(staging.mapped, rgba, 4);

    tex->m_image = ctx->createImage(1, 1, 1, VK_SAMPLE_COUNT_1_BIT,
        VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    tex->m_image.mipLevels = 1;

    ctx->transitionImageLayout(tex->m_image.image, VK_FORMAT_R8G8B8A8_SRGB,
                               VK_IMAGE_LAYOUT_UNDEFINED,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1);
    ctx->copyBufferToImage(staging.buffer, tex->m_image.image, 1, 1);
    ctx->transitionImageLayout(tex->m_image.image, VK_FORMAT_R8G8B8A8_SRGB,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    ctx->destroyBuffer(staging);
    ctx->createImageView(tex->m_image, VK_IMAGE_ASPECT_COLOR_BIT);

    TextureSettings s = settings;
    s.generateMipmaps = false;
    tex->createSampler(s);
    return tex;
}

std::shared_ptr<Texture> Texture::createRenderTarget(std::shared_ptr<const Context> ctx,
                                                       uint32_t w, uint32_t h,
                                                       VkFormat format,
                                                       VkImageUsageFlags usage)
{
    auto tex = std::shared_ptr<Texture>(new Texture(ctx));
    tex->m_width = w; tex->m_height = h;
    tex->m_image = ctx->createImage(w, h, 1, VK_SAMPLE_COUNT_1_BIT, format,
        VK_IMAGE_TILING_OPTIMAL, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    tex->m_image.mipLevels = 1;
    ctx->createImageView(tex->m_image, VK_IMAGE_ASPECT_COLOR_BIT);
    TextureSettings s;
    s.generateMipmaps = false;
    tex->createSampler(s);
    return tex;
}

Texture::~Texture() {
    if (m_sampler) vkDestroySampler(m_ctx->device(), m_sampler, nullptr);
    m_ctx->destroyImage(m_image);
}

void Texture::createSampler(const TextureSettings& settings) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(m_ctx->physicalDevice(), &props);

    VkSamplerCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    ci.magFilter        = settings.magFilter;
    ci.minFilter        = settings.minFilter;
    ci.addressModeU     = settings.addressMode;
    ci.addressModeV     = settings.addressMode;
    ci.addressModeW     = settings.addressMode;
    ci.anisotropyEnable = settings.anisotropy ? VK_TRUE : VK_FALSE;
    ci.maxAnisotropy    = settings.anisotropy ? props.limits.maxSamplerAnisotropy : 1.f;
    ci.borderColor      = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    ci.unnormalizedCoordinates = VK_FALSE;
    ci.compareEnable    = VK_FALSE;
    ci.compareOp        = VK_COMPARE_OP_ALWAYS;
    ci.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    ci.minLod           = 0.f;
    ci.maxLod           = static_cast<float>(m_image.mipLevels);
    ci.mipLodBias       = 0.f;
    VK_CHECK(vkCreateSampler(m_ctx->device(), &ci, nullptr, &m_sampler), "Create sampler");
}

// ─── TextureCache ─────────────────────────────────────────────────────────────
std::shared_ptr<Texture> TextureCache::get(const std::filesystem::path& path,
                                            const TextureSettings& settings)
{
    auto key = path.string();
    if (auto it = m_cache.find(key); it != m_cache.end()) return it->second;
    auto tex = Texture::fromFile(m_ctx, path, settings);
    m_cache[key] = tex;
    return tex;
}

} // namespace vkgfx
