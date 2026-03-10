#pragma once
#include "context.h"
#include <filesystem>
#include <memory>

namespace vkgfx {

struct TextureSettings {
    bool generateMipmaps = true;
    bool anisotropy      = true;
    VkFilter minFilter   = VK_FILTER_LINEAR;
    VkFilter magFilter   = VK_FILTER_LINEAR;
    VkSamplerAddressMode addressMode = VK_SAMPLER_ADDRESS_MODE_REPEAT;
};

class Texture {
public:
    // Factory functions accept a shared_ptr<Context> so the Texture can
    // co-own the Context and keep VMA alive beyond the Renderer's lifetime.
    // The const-ref overloads are removed — callers must pass the shared_ptr
    // obtained from Renderer::contextPtr().
    static std::shared_ptr<Texture> fromFile(std::shared_ptr<const Context> ctx,
                                              const std::filesystem::path& path,
                                              const TextureSettings& settings = {});

    static std::shared_ptr<Texture> fromColor(std::shared_ptr<const Context> ctx,
                                               Vec4 color,
                                               const TextureSettings& settings = {});

    static std::shared_ptr<Texture> createRenderTarget(std::shared_ptr<const Context> ctx,
                                                         uint32_t w, uint32_t h,
                                                         VkFormat format,
                                                         VkImageUsageFlags usage);

    ~Texture();

    Texture(const Texture&)            = delete;
    Texture& operator=(const Texture&) = delete;

    [[nodiscard]] VkImageView view()    const { return m_image.view; }
    [[nodiscard]] VkSampler   sampler() const { return m_sampler; }
    [[nodiscard]] VkFormat    format()  const { return m_image.format; }
    [[nodiscard]] uint32_t    width()   const { return m_width; }
    [[nodiscard]] uint32_t    height()  const { return m_height; }
    [[nodiscard]] uint32_t    mips()    const { return m_image.mipLevels; }
    [[nodiscard]] const std::filesystem::path& path() const { return m_path; }

    [[nodiscard]] VkDescriptorImageInfo descriptorInfo() const {
        return { m_sampler, m_image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    }

private:
    explicit Texture(std::shared_ptr<const Context> ctx) : m_ctx(std::move(ctx)) {}

    void createSampler(const TextureSettings& settings);

    std::shared_ptr<const Context> m_ctx;  // shared ownership keeps VMA alive
    AllocatedImage         m_image;
    VkSampler              m_sampler  = VK_NULL_HANDLE;
    uint32_t               m_width    = 0;
    uint32_t               m_height   = 0;
    std::filesystem::path  m_path;
};

class TextureCache {
public:
    explicit TextureCache(std::shared_ptr<const Context> ctx) : m_ctx(std::move(ctx)) {}

    std::shared_ptr<Texture> get(const std::filesystem::path& path,
                                  const TextureSettings& settings = {});
    void clear() { m_cache.clear(); }

private:
    std::shared_ptr<const Context> m_ctx;
    std::unordered_map<std::string, std::shared_ptr<Texture>> m_cache;
};

} // namespace vkgfx
