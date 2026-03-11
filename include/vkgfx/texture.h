#pragma once
// texture.h — Image texture loading and GPU upload via stb_image.
//
// Public loading API:
//   Texture::fromFile()      — loads PNG / JPG / BMP / TGA / HDR from disk
//                              using stb_image; generates mipmaps on the GPU.
//   Texture::fromGltfImage() — uploads a decoded image block from tinygltf.
//   Texture::fromColor()     — 1x1 solid-colour fallback.

#include "context.h"
#include <filesystem>
#include <memory>
#include <cstdint>

namespace vkgfx {

struct TextureSettings {
    bool generateMipmaps = true;
    bool anisotropy      = true;
    VkFilter minFilter   = VK_FILTER_LINEAR;
    VkFilter magFilter   = VK_FILTER_LINEAR;
    VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT;
};

// Decoded image block supplied by the glTF loader (tinygltf).
struct GltfImageData {
    const uint8_t* pixels = nullptr;  // RGBA8 decoded pixels
    uint32_t       width  = 0;
    uint32_t       height = 0;
    bool           srgb   = true;     // albedo/emissive maps are sRGB
};

class Texture {
public:
    // Load any format stb_image supports: PNG, JPG, BMP, TGA, HDR, PIC.
    // srgb=true for albedo/emissive, false for normal/metalRough/AO/HDR maps.
    static std::shared_ptr<Texture> fromFile(std::shared_ptr<const Context> ctx,
                                              const std::filesystem::path& path,
                                              bool srgb = true,
                                              const TextureSettings& settings = {});

    // Upload a decoded image from the glTF loader (tinygltf).
    static std::shared_ptr<Texture> fromGltfImage(std::shared_ptr<const Context> ctx,
                                                   const GltfImageData& data,
                                                   const TextureSettings& settings = {});

    // 1x1 solid-colour texture (white, flat-normal, etc.).
    static std::shared_ptr<Texture> fromColor(std::shared_ptr<const Context> ctx,
                                               Vec4 color,
                                               const TextureSettings& settings = {});

    ~Texture();
    Texture(const Texture&)            = delete;
    Texture& operator=(const Texture&) = delete;

    [[nodiscard]] VkImageView view()    const { return m_image.view; }
    [[nodiscard]] VkSampler   sampler() const { return m_sampler; }
    [[nodiscard]] VkFormat    format()  const { return m_image.format; }
    [[nodiscard]] uint32_t    width()   const { return m_width; }
    [[nodiscard]] uint32_t    height()  const { return m_height; }
    [[nodiscard]] uint32_t    mips()    const { return m_image.mipLevels; }

    [[nodiscard]] VkDescriptorImageInfo descriptorInfo() const {
        return { m_sampler, m_image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    }

private:
    explicit Texture(std::shared_ptr<const Context> ctx) : m_ctx(std::move(ctx)) {}

    void uploadRGBA8(const uint8_t* pixels, uint32_t w, uint32_t h, bool srgb,
                     const TextureSettings& settings);
    void createSampler(const TextureSettings& settings);

    std::shared_ptr<const Context> m_ctx;
    AllocatedImage                 m_image;
    VkSampler                      m_sampler = VK_NULL_HANDLE;
    uint32_t                       m_width   = 0;
    uint32_t                       m_height  = 0;
};

// Simple path-keyed cache — avoids re-uploading the same file.
class TextureCache {
public:
    explicit TextureCache(std::shared_ptr<const Context> ctx) : m_ctx(std::move(ctx)) {}

    std::shared_ptr<Texture> get(const std::filesystem::path& path,
                                  bool srgb = true,
                                  const TextureSettings& s = {});
    void clear() { m_cache.clear(); }

private:
    std::shared_ptr<const Context>                            m_ctx;
    std::unordered_map<std::string, std::shared_ptr<Texture>> m_cache;
};

} // namespace vkgfx
