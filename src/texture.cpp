// texture.cpp — Texture loading via stb_image and GPU upload.
//
// Supports: PNG, JPG, BMP, TGA, HDR, PIC (any format stb_image handles).
// Mipmaps are generated on the GPU via vkCmdBlitImage.
// Embedded glTF images (pre-decoded RGBA8 by tinygltf) skip the file load step.

#include "vkgfx/texture.h"

// STB_IMAGE_IMPLEMENTATION is defined in stb_impl.cpp — do not repeat here.
#include <stb_image.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <iostream>

namespace vkgfx {

static VkFormat pick_rgba8_format(bool srgb) {
    return srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
}

// ── Texture::fromFile ─────────────────────────────────────────────────────────
std::shared_ptr<Texture> Texture::fromFile(std::shared_ptr<const Context> ctx,
                                             const std::filesystem::path& path,
                                             bool srgb,
                                             const TextureSettings& settings)
{
    int w, h, channels;
    stbi_uc* pixels = stbi_load(path.string().c_str(), &w, &h, &channels, STBI_rgb_alpha);
    if (!pixels)
        throw std::runtime_error("[VKGFX] stb_image failed to load: " + path.string()
                                 + " — " + stbi_failure_reason());

    auto tex = std::shared_ptr<Texture>(new Texture(ctx));
    tex->m_width  = static_cast<uint32_t>(w);
    tex->m_height = static_cast<uint32_t>(h);
    tex->uploadRGBA8(pixels, tex->m_width, tex->m_height, srgb, settings);
    tex->createSampler(settings);

    stbi_image_free(pixels);

    std::cout << "[VKGFX] Loaded: " << path.filename().string()
              << " (" << w << "x" << h << ")\n";
    return tex;
}

// ── Texture::fromGltfImage ────────────────────────────────────────────────────
std::shared_ptr<Texture> Texture::fromGltfImage(std::shared_ptr<const Context> ctx,
                                                  const GltfImageData& data,
                                                  const TextureSettings& settings)
{
    if (!data.pixels || data.width == 0 || data.height == 0)
        return fromColor(ctx, {1.f, 1.f, 1.f, 1.f}, settings);

    auto tex = std::shared_ptr<Texture>(new Texture(ctx));
    tex->m_width  = data.width;
    tex->m_height = data.height;
    tex->uploadRGBA8(data.pixels, data.width, data.height, data.srgb, settings);
    tex->createSampler(settings);
    return tex;
}

// ── Texture::fromColor ────────────────────────────────────────────────────────
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
    TextureSettings s = settings;
    s.generateMipmaps = false;
    tex->uploadRGBA8(rgba, 1, 1, true, s);
    tex->createSampler(s);
    return tex;
}

// ── Texture::uploadRGBA8 ──────────────────────────────────────────────────────
void Texture::uploadRGBA8(const uint8_t* pixels, uint32_t w, uint32_t h,
                           bool srgb, const TextureSettings& settings)
{
    VkDeviceSize size = static_cast<VkDeviceSize>(w) * h * 4;
    VkFormat fmt = pick_rgba8_format(srgb);

    uint32_t mips = settings.generateMipmaps
        ? static_cast<uint32_t>(std::floor(std::log2(std::max(w, h)))) + 1u
        : 1u;

    auto staging = m_ctx->createBuffer(size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(staging.mapped, pixels, static_cast<size_t>(size));

    m_image = m_ctx->createImage(w, h, mips, VK_SAMPLE_COUNT_1_BIT, fmt,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_image.mipLevels = mips;

    m_ctx->transitionImageLayout(m_image.image, fmt,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mips);
    m_ctx->copyBufferToImage(staging.buffer, m_image.image, w, h);

    if (mips > 1)
        m_ctx->generateMipmaps(m_image.image, fmt, w, h, mips);
    else
        m_ctx->transitionImageLayout(m_image.image, fmt,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);

    m_ctx->destroyBuffer(staging);
    m_ctx->createImageView(m_image, VK_IMAGE_ASPECT_COLOR_BIT);
}

// ── Texture::createSampler ────────────────────────────────────────────────────
void Texture::createSampler(const TextureSettings& settings) {
    VkPhysicalDeviceProperties props{};
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
    ci.compareEnable    = VK_FALSE;
    ci.compareOp        = VK_COMPARE_OP_ALWAYS;
    ci.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    ci.minLod           = 0.f;
    ci.maxLod           = static_cast<float>(m_image.mipLevels);
    VK_CHECK(vkCreateSampler(m_ctx->device(), &ci, nullptr, &m_sampler));
}

// ── Texture destructor ────────────────────────────────────────────────────────
Texture::~Texture() {
    if (m_sampler) vkDestroySampler(m_ctx->device(), m_sampler, nullptr);
    if (m_image.allocation)
        m_ctx->destroyImage(m_image);
}

// ── TextureCache ──────────────────────────────────────────────────────────────
std::shared_ptr<Texture> TextureCache::get(const std::filesystem::path& path,
                                             bool srgb,
                                             const TextureSettings& s)
{
    auto key = path.string();
    if (auto it = m_cache.find(key); it != m_cache.end()) return it->second;
    auto tex = Texture::fromFile(m_ctx, path, srgb, s);
    m_cache[key] = tex;
    return tex;
}

} // namespace vkgfx
