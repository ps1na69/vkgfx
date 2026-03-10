#pragma once
#include "context.h"

namespace vkgfx {

class Swapchain {
public:
    Swapchain(const Context& ctx, VkSurfaceKHR surface,
              uint32_t width, uint32_t height,
              bool vsync = true,
              VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT);
    ~Swapchain();

    Swapchain(const Swapchain&)            = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    // Per-frame resources indexed by frame-in-flight.
    struct FrameData {
        VkSemaphore   imageAvailableSemaphore = VK_NULL_HANDLE;
        VkFence       inFlightFence           = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer         = VK_NULL_HANDLE;
    };

    [[nodiscard]] VkResult acquireNextImage(uint32_t frameIdx, uint32_t& imageIdx);
    [[nodiscard]] VkResult present(uint32_t imageIdx);

    // Waits for the presentation fence of the given image index (maintenance1 path only).
    // Must be called before signalling renderFinishedSemaphore(imageIdx) in a new submit.
    // acquireNextImage() calls this automatically; exposed here for callers that need
    // fine-grained control.
    void waitPresentFence(uint32_t imageIdx);

    void recreate(uint32_t width, uint32_t height);

    [[nodiscard]] VkSwapchainKHR         handle()       const { return m_swapchain; }
    [[nodiscard]] VkRenderPass           renderPass()   const { return m_renderPass; }
    [[nodiscard]] VkExtent2D             extent()       const { return m_extent; }
    [[nodiscard]] VkFormat               format()       const { return m_format; }
    [[nodiscard]] uint32_t               imageCount()   const { return static_cast<uint32_t>(m_images.size()); }
    [[nodiscard]] VkFramebuffer          framebuffer(uint32_t idx) const { return m_framebuffers[idx]; }
    [[nodiscard]] const FrameData&       frame(uint32_t idx)       const { return m_frames[idx]; }
    [[nodiscard]] FrameData&             frame(uint32_t idx)             { return m_frames[idx]; }
    [[nodiscard]] VkSampleCountFlagBits  sampleCount()  const { return m_samples; }
    [[nodiscard]] VkImageView           imageView(uint32_t idx) const { return m_imageViews[idx]; }
    [[nodiscard]] VkSemaphore            imageAvailableSemaphore(uint32_t frameIdx) const;
    [[nodiscard]] VkSemaphore            renderFinishedSemaphore(uint32_t imageIdx) const;

private:
    void create(uint32_t w, uint32_t h);
    void destroy();
    void createRenderPass();
    void createFramebuffers();
    void createSyncObjects();
    void allocateCommandBuffers();

    VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>&) const;
    VkPresentModeKHR   choosePresentMode(const std::vector<VkPresentModeKHR>&) const;
    VkExtent2D         chooseExtent(const VkSurfaceCapabilitiesKHR&, uint32_t w, uint32_t h) const;

    const Context&       m_ctx;
    VkSurfaceKHR         m_surface    = VK_NULL_HANDLE;
    VkSwapchainKHR       m_swapchain  = VK_NULL_HANDLE;
    VkRenderPass         m_renderPass = VK_NULL_HANDLE;
    VkExtent2D           m_extent     = {};
    VkFormat             m_format     = VK_FORMAT_UNDEFINED;
    bool                 m_vsync      = true;
    VkSampleCountFlagBits m_samples   = VK_SAMPLE_COUNT_1_BIT;

    std::vector<VkImage>       m_images;
    std::vector<VkImageView>   m_imageViews;
    std::vector<VkFramebuffer> m_framebuffers;

    AllocatedImage m_colorTarget;
    AllocatedImage m_depthTarget;

    std::array<FrameData, MAX_FRAMES_IN_FLIGHT> m_frames;

    std::vector<VkSemaphore> m_renderFinishedSems;

    // Per-image present fences (VK_KHR_swapchain_maintenance1).
    // A fence in this array is passed to vkQueuePresentKHR and fires when the
    // presentation engine is done consuming the corresponding renderFinished
    // semaphore. Before reusing renderFinishedSems[imageIdx] we wait on
    // m_presentFences[imageIdx] so there is no aliased-semaphore race.
    std::vector<VkFence> m_presentFences;
    bool                 m_maintenance1 = false;

    VkCommandPool m_commandPool = VK_NULL_HANDLE;
};

} // namespace vkgfx
