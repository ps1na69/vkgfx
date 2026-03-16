#pragma once
// include/vkgfx/swapchain.h

#include "vk_raii.h"
#include "context.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>

namespace vkgfx {

class Window;

struct SwapchainConfig {
    bool     vsync        = true;
    uint32_t preferredW   = 0;
    uint32_t preferredH   = 0;
};

class Swapchain {
public:
    Swapchain(Context& ctx, Window& window, const SwapchainConfig& cfg = {});
    ~Swapchain();

    Swapchain(const Swapchain&)            = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    /// Recreate after window resize. Blocks until device idle.
    void recreate();

    // ── Frame acquire / present ───────────────────────────────────────────────
    /// Returns VK_ERROR_OUT_OF_DATE_KHR or VK_SUBOPTIMAL_KHR if resize needed.
    VkResult acquireNext(VkSemaphore signal, uint32_t& outIndex);
    VkResult present(VkSemaphore wait, uint32_t imageIndex);

    // ── Accessors ─────────────────────────────────────────────────────────────
    [[nodiscard]] VkSwapchainKHR     handle()         const { return m_swapchain; }
    [[nodiscard]] VkFormat           format()         const { return m_format; }
    [[nodiscard]] VkExtent2D         extent()         const { return m_extent; }
    [[nodiscard]] uint32_t           imageCount()     const { return static_cast<uint32_t>(m_images.size()); }
    [[nodiscard]] VkImageView        imageView(uint32_t i) const { return m_imageViews[i]; }
    [[nodiscard]] VkImage            image(uint32_t i)     const { return m_images[i]; }

    // Tonemapped output render pass (final pass writes to swapchain image)
    [[nodiscard]] VkRenderPass       renderPass()     const { return m_renderPass; }
    [[nodiscard]] VkFramebuffer      framebuffer(uint32_t i) const { return m_framebuffers[i]; }

private:
    void create();
    void destroy();
    void createRenderPass();
    void createFramebuffers();

    Context&          m_ctx;
    Window&           m_window;
    SwapchainConfig   m_cfg;

    VkSwapchainKHR            m_swapchain  = VK_NULL_HANDLE;
    VkFormat                  m_format     = VK_FORMAT_UNDEFINED;
    VkExtent2D                m_extent     = {};
    std::vector<VkImage>      m_images;
    std::vector<VkImageView>  m_imageViews;
    VkRenderPass              m_renderPass = VK_NULL_HANDLE;
    std::vector<VkFramebuffer>m_framebuffers;
};

} // namespace vkgfx
