#pragma once
// include/vkgfx/texture.h

#include "vk_raii.h"
#include "context.h"
#include <string>
#include <memory>
#include <unordered_map>

namespace vkgfx {

struct TextureDesc {
    VkFormat  format    = VK_FORMAT_R8G8B8A8_SRGB;
    bool      genMips   = true;
    bool      isHDR     = false;   // load as R32G32B32A32_SFLOAT
    bool      isCube    = false;
};

class Texture {
public:
    Texture() = default;
    ~Texture() = default;

    // Non-copyable, movable
    Texture(const Texture&)            = delete;
    Texture& operator=(const Texture&) = delete;
    Texture(Texture&&)                 = default;
    Texture& operator=(Texture&&)      = default;

    [[nodiscard]] VkImageView   view()    const { return m_img.view; }
    [[nodiscard]] VkSampler     sampler() const { return m_sampler; }
    [[nodiscard]] bool          valid()   const { return m_img.image != VK_NULL_HANDLE; }

    // Use TextureCache to create textures
    void destroy(Context& ctx);

private:
    friend class TextureCache;
    AllocatedImage m_img;
    VkSampler      m_sampler = VK_NULL_HANDLE;
};

/// Loads textures from disk, deduplicates by path.
class TextureCache {
public:
    explicit TextureCache(Context& ctx);
    ~TextureCache();

    /// Load from file.  Returns cached entry on repeated calls with same path.
    /// Returns nullptr and logs error if path does not exist.
    std::shared_ptr<Texture> load(const std::string& path, TextureDesc desc = {});

    /// Create a 1×1 solid-colour fallback texture (for missing maps).
    std::shared_ptr<Texture> solid(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255);

    void clear();

private:
    std::shared_ptr<Texture> uploadLDR(const std::string& path, const TextureDesc& desc);
    std::shared_ptr<Texture> uploadHDR(const std::string& path, const TextureDesc& desc);
    VkSampler makeSampler(uint32_t mipLevels);

    Context&  m_ctx;
    std::unordered_map<std::string, std::shared_ptr<Texture>> m_cache;
};

} // namespace vkgfx
