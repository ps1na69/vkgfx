// src/swapchain.cpp
#include <vkgfx/swapchain.h>
#include <vkgfx/window.h>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace vkgfx {

Swapchain::Swapchain(Context& ctx, Window& window, const SwapchainConfig& cfg)
    : m_ctx(ctx), m_window(window), m_cfg(cfg)
{
    create();
    createRenderPass();
    createFramebuffers();
}

Swapchain::~Swapchain() {
    destroy();
    if (m_renderPass != VK_NULL_HANDLE)
        vkDestroyRenderPass(m_ctx.device(), m_renderPass, nullptr);
}

void Swapchain::recreate() {
    vkDeviceWaitIdle(m_ctx.device());
    destroy();
    create();
    createFramebuffers();
}

// ── create ────────────────────────────────────────────────────────────────────

void Swapchain::create() {
    VkSurfaceCapabilitiesKHR caps{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_ctx.gpu(), m_ctx.surface(), &caps);

    // Format selection: prefer SRGB
    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_ctx.gpu(), m_ctx.surface(), &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_ctx.gpu(), m_ctx.surface(), &fmtCount, formats.data());

    VkSurfaceFormatKHR chosen = formats[0];
    for (auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    }
    m_format = chosen.format;

    // Present mode
    uint32_t pmCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_ctx.gpu(), m_ctx.surface(), &pmCount, nullptr);
    std::vector<VkPresentModeKHR> modes(pmCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(m_ctx.gpu(), m_ctx.surface(), &pmCount, modes.data());

    VkPresentModeKHR pm = VK_PRESENT_MODE_FIFO_KHR;  // always available
    if (!m_cfg.vsync) {
        for (auto m : modes)
            if (m == VK_PRESENT_MODE_MAILBOX_KHR) { pm = m; break; }
    }

    // Extent
    if (caps.currentExtent.width != UINT32_MAX) {
        m_extent = caps.currentExtent;
    } else {
        m_extent.width  = std::clamp(m_window.width(),  caps.minImageExtent.width,  caps.maxImageExtent.width);
        m_extent.height = std::clamp(m_window.height(), caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    uint32_t imgCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) imgCount = std::min(imgCount, caps.maxImageCount);

    VkSwapchainCreateInfoKHR ci{};                            // zero-init
    ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface          = m_ctx.surface();
    ci.minImageCount    = imgCount;
    ci.imageFormat      = chosen.format;
    ci.imageColorSpace  = chosen.colorSpace;
    ci.imageExtent      = m_extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    uint32_t queueFamilies[] = {m_ctx.queues().graphics, m_ctx.queues().present};
    if (m_ctx.queues().graphics != m_ctx.queues().present) {
        ci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices   = queueFamilies;
    } else {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    ci.preTransform   = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode    = pm;
    ci.clipped        = VK_TRUE;

    VkResult res = vkCreateSwapchainKHR(m_ctx.device(), &ci, nullptr, &m_swapchain);
    if (res != VK_SUCCESS)
        throw std::runtime_error("[vkgfx] vkCreateSwapchainKHR failed");

    // Retrieve images
    uint32_t count = 0;
    vkGetSwapchainImagesKHR(m_ctx.device(), m_swapchain, &count, nullptr);
    m_images.resize(count);
    vkGetSwapchainImagesKHR(m_ctx.device(), m_swapchain, &count, m_images.data());

    // Create image views
    m_imageViews.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        m_imageViews[i] = m_ctx.createImageView(m_images[i], m_format,
                                                  VK_IMAGE_ASPECT_COLOR_BIT);
    }
}

void Swapchain::destroy() {
    for (auto fb : m_framebuffers)
        vkDestroyFramebuffer(m_ctx.device(), fb, nullptr);
    m_framebuffers.clear();

    for (auto v : m_imageViews)
        vkDestroyImageView(m_ctx.device(), v, nullptr);
    m_imageViews.clear();

    if (m_swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(m_ctx.device(), m_swapchain, nullptr);
        m_swapchain = VK_NULL_HANDLE;
    }
    m_images.clear();
}

// ── createRenderPass ──────────────────────────────────────────────────────────
// This is the final tonemapping → swapchain pass.

void Swapchain::createRenderPass() {
    VkAttachmentDescription colorAtt{};                       // zero-init
    colorAtt.format         = m_format;
    colorAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};                           // zero-init
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &colorRef;

    // Dependency: wait for lighting pass color write before reading
    VkSubpassDependency dep{};                                // zero-init
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpi{};                             // zero-init
    rpi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpi.attachmentCount = 1;
    rpi.pAttachments    = &colorAtt;
    rpi.subpassCount    = 1;
    rpi.pSubpasses      = &subpass;
    rpi.dependencyCount = 1;
    rpi.pDependencies   = &dep;

    VkResult res = vkCreateRenderPass(m_ctx.device(), &rpi, nullptr, &m_renderPass);
    if (res != VK_SUCCESS)
        throw std::runtime_error("[vkgfx] Swapchain render pass creation failed");
}

// ── createFramebuffers ────────────────────────────────────────────────────────

void Swapchain::createFramebuffers() {
    m_framebuffers.resize(m_imageViews.size());
    for (size_t i = 0; i < m_imageViews.size(); ++i) {
        VkFramebufferCreateInfo fi{};                         // zero-init
        fi.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fi.renderPass      = m_renderPass;
        fi.attachmentCount = 1;
        fi.pAttachments    = &m_imageViews[i];
        fi.width           = m_extent.width;
        fi.height          = m_extent.height;
        fi.layers          = 1;

        VkResult res = vkCreateFramebuffer(m_ctx.device(), &fi, nullptr, &m_framebuffers[i]);
        if (res != VK_SUCCESS)
            throw std::runtime_error("[vkgfx] vkCreateFramebuffer failed");
    }
}

// ── acquire / present ─────────────────────────────────────────────────────────

VkResult Swapchain::acquireNext(VkSemaphore signal, uint32_t& outIndex) {
    return vkAcquireNextImageKHR(m_ctx.device(), m_swapchain,
                                  UINT64_MAX, signal, VK_NULL_HANDLE, &outIndex);
}

VkResult Swapchain::present(VkSemaphore wait, uint32_t imageIndex) {
    VkPresentInfoKHR pi{};                                    // zero-init
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &wait;
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &m_swapchain;
    pi.pImageIndices      = &imageIndex;

    return vkQueuePresentKHR(m_ctx.presentQ(), &pi);
}

} // namespace vkgfx
