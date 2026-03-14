#include "vkgfx/swapchain.h"
#include <algorithm>

namespace vkgfx {

Swapchain::Swapchain(const Context& ctx, VkSurfaceKHR surface,
                     uint32_t width, uint32_t height,
                     bool vsync, VkSampleCountFlagBits samples)
    : m_ctx(ctx), m_surface(surface), m_vsync(vsync), m_samples(samples),
      m_maintenance1(ctx.hasMaintenance1())
{
    // Create a dedicated command pool for swapchain command buffers
    VkCommandPoolCreateInfo pi{};
    pi.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pi.queueFamilyIndex = *ctx.queueFamilies().graphics;
    pi.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(ctx.device(), &pi, nullptr, &m_commandPool), "Swapchain command pool");

    create(width, height);
    createSyncObjects();
    allocateCommandBuffers();
}

Swapchain::~Swapchain() {
    destroy();
    // Destroy per-image render-finished semaphores and present fences.
    for (auto sem : m_renderFinishedSems)
        if (sem) vkDestroySemaphore(m_ctx.device(), sem, nullptr);
    m_renderFinishedSems.clear();
    for (auto fence : m_presentFences)
        if (fence) vkDestroyFence(m_ctx.device(), fence, nullptr);
    m_presentFences.clear();
    // Destroy per-frame image-available semaphores and fences.
    for (auto& f : m_frames) {
        if (f.imageAvailableSemaphore) vkDestroySemaphore(m_ctx.device(), f.imageAvailableSemaphore, nullptr);
        if (f.inFlightFence)           vkDestroyFence(m_ctx.device(), f.inFlightFence, nullptr);
    }
    if (m_commandPool) vkDestroyCommandPool(m_ctx.device(), m_commandPool, nullptr);
}

void Swapchain::create(uint32_t w, uint32_t h) {
    auto support = m_ctx.querySwapchainSupport(m_surface);
    auto format  = chooseSurfaceFormat(support.formats);
    auto present = choosePresentMode(support.presentModes);
    auto extent  = chooseExtent(support.capabilities, w, h);

    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount)
        imageCount = support.capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR ci{};
    ci.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface          = m_surface;
    ci.minImageCount    = imageCount;
    ci.imageFormat      = format.format;
    ci.imageColorSpace  = format.colorSpace;
    ci.imageExtent      = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    const auto& qi = m_ctx.queueFamilies();
    uint32_t indices[] = { *qi.graphics, *qi.present };
    if (qi.graphics != qi.present) {
        ci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices   = indices;
    } else {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    ci.preTransform   = support.capabilities.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode    = present;
    ci.clipped        = VK_TRUE;

    VK_CHECK(vkCreateSwapchainKHR(m_ctx.device(), &ci, nullptr, &m_swapchain), "Create swapchain");
    m_format = format.format;
    m_extent = extent;

    // Get swapchain images
    uint32_t cnt;
    vkGetSwapchainImagesKHR(m_ctx.device(), m_swapchain, &cnt, nullptr);
    m_images.resize(cnt);
    vkGetSwapchainImagesKHR(m_ctx.device(), m_swapchain, &cnt, m_images.data());

    // Create image views
    m_imageViews.resize(cnt);
    for (uint32_t i = 0; i < cnt; ++i) {
        VkImageViewCreateInfo iv{};
        iv.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        iv.image                           = m_images[i];
        iv.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        iv.format                          = m_format;
        iv.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        iv.subresourceRange.baseMipLevel   = 0;
        iv.subresourceRange.levelCount     = 1;
        iv.subresourceRange.baseArrayLayer = 0;
        iv.subresourceRange.layerCount     = 1;
        VK_CHECK(vkCreateImageView(m_ctx.device(), &iv, nullptr, &m_imageViews[i]), "Swapchain image view");
    }

    // MSAA color + depth
    if (m_samples != VK_SAMPLE_COUNT_1_BIT) {
        m_colorTarget = m_ctx.createImage(extent.width, extent.height, 1, m_samples,
            m_format, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        m_ctx.createImageView(m_colorTarget, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    VkFormat depthFmt = m_ctx.findDepthFormat();
    m_depthTarget = m_ctx.createImage(extent.width, extent.height, 1, m_samples,
        depthFmt, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_ctx.createImageView(m_depthTarget, VK_IMAGE_ASPECT_DEPTH_BIT);

    createRenderPass();
    createFramebuffers();
}

void Swapchain::destroy() {
    m_ctx.destroyImage(m_colorTarget);
    m_ctx.destroyImage(m_depthTarget);
    for (auto fb : m_framebuffers) vkDestroyFramebuffer(m_ctx.device(), fb, nullptr);
    m_framebuffers.clear();
    if (m_renderPass) { vkDestroyRenderPass(m_ctx.device(), m_renderPass, nullptr); m_renderPass = VK_NULL_HANDLE; }
    for (auto iv : m_imageViews) vkDestroyImageView(m_ctx.device(), iv, nullptr);
    m_imageViews.clear();
    if (m_swapchain) { vkDestroySwapchainKHR(m_ctx.device(), m_swapchain, nullptr); m_swapchain = VK_NULL_HANDLE; }
}

void Swapchain::createRenderPass() {
    bool msaa = (m_samples != VK_SAMPLE_COUNT_1_BIT);
    VkFormat depthFmt = m_ctx.findDepthFormat();

    VkAttachmentDescription colorAttach{};
    colorAttach.format         = m_format;
    colorAttach.samples        = m_samples;
    colorAttach.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttach.storeOp        = msaa ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE;
    colorAttach.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttach.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttach.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttach.finalLayout    = msaa ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                      : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depthAttach{};
    depthAttach.format         = depthFmt;
    depthAttach.samples        = m_samples;
    depthAttach.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttach.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttach.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttach.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttach.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttach.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription resolveAttach{};
    if (msaa) {
        resolveAttach.format         = m_format;
        resolveAttach.samples        = VK_SAMPLE_COUNT_1_BIT;
        resolveAttach.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        resolveAttach.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        resolveAttach.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        resolveAttach.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        resolveAttach.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        resolveAttach.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    }

    VkAttachmentReference colorRef { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthRef { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
    VkAttachmentReference resolveRef { 2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;
    if (msaa) subpass.pResolveAttachments = &resolveRef;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::vector<VkAttachmentDescription> attachments = { colorAttach, depthAttach };
    if (msaa) attachments.push_back(resolveAttach);

    VkRenderPassCreateInfo rpci{};
    rpci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = static_cast<uint32_t>(attachments.size());
    rpci.pAttachments    = attachments.data();
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &subpass;
    rpci.dependencyCount = 1;
    rpci.pDependencies   = &dep;
    VK_CHECK(vkCreateRenderPass(m_ctx.device(), &rpci, nullptr, &m_renderPass), "Create render pass");
}

void Swapchain::createFramebuffers() {
    bool msaa = (m_samples != VK_SAMPLE_COUNT_1_BIT);
    m_framebuffers.resize(m_images.size());
    for (size_t i = 0; i < m_images.size(); ++i) {
        std::vector<VkImageView> attachments;
        if (msaa) {
            attachments = { m_colorTarget.view, m_depthTarget.view, m_imageViews[i] };
        } else {
            attachments = { m_imageViews[i], m_depthTarget.view };
        }
        VkFramebufferCreateInfo fci{};
        fci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fci.renderPass      = m_renderPass;
        fci.attachmentCount = static_cast<uint32_t>(attachments.size());
        fci.pAttachments    = attachments.data();
        fci.width           = m_extent.width;
        fci.height          = m_extent.height;
        fci.layers          = 1;
        VK_CHECK(vkCreateFramebuffer(m_ctx.device(), &fci, nullptr, &m_framebuffers[i]), "Create framebuffer");
    }
}

void Swapchain::createSyncObjects() {
    VkSemaphoreCreateInfo si{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo fi{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    // Per-frame resources: acquire semaphore + in-flight fence.
    for (auto& f : m_frames) {
        if (!f.imageAvailableSemaphore)
            VK_CHECK(vkCreateSemaphore(m_ctx.device(), &si, nullptr, &f.imageAvailableSemaphore),
                     "imageAvailable semaphore");
        if (!f.inFlightFence)
            VK_CHECK(vkCreateFence(m_ctx.device(), &fi, nullptr, &f.inFlightFence), "inFlight fence");
    }

    // Per-image resources: render-finished semaphore (submit signal/present wait).
    for (auto sem : m_renderFinishedSems)
        if (sem) vkDestroySemaphore(m_ctx.device(), sem, nullptr);

    m_renderFinishedSems.assign(m_images.size(), VK_NULL_HANDLE);
    for (auto& sem : m_renderFinishedSems)
        VK_CHECK(vkCreateSemaphore(m_ctx.device(), &si, nullptr, &sem), "renderFinished semaphore");

    // Per-image present fences (VK_KHR_swapchain_maintenance1 path only).
    // Created pre-signaled so the very first frame never blocks: on the first
    // use of imageIdx N there was no prior present to wait for.
    if (m_maintenance1) {
        for (auto fence : m_presentFences)
            if (fence) vkDestroyFence(m_ctx.device(), fence, nullptr);

        VkFenceCreateInfo pfi{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        pfi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        m_presentFences.assign(m_images.size(), VK_NULL_HANDLE);
        for (auto& fence : m_presentFences)
            VK_CHECK(vkCreateFence(m_ctx.device(), &pfi, nullptr, &fence), "presentFence");
    }
}

void Swapchain::allocateCommandBuffers() {
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = m_commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
    std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> cmds;
    VK_CHECK(vkAllocateCommandBuffers(m_ctx.device(), &ai, cmds.data()), "Allocate command buffers");
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) m_frames[i].commandBuffer = cmds[i];
}

VkSemaphore Swapchain::imageAvailableSemaphore(uint32_t frameIdx) const {
    return m_frames[frameIdx].imageAvailableSemaphore;
}

VkSemaphore Swapchain::renderFinishedSemaphore(uint32_t imageIdx) const {
    return m_renderFinishedSems[imageIdx];
}

void Swapchain::waitPresentFence(uint32_t imageIdx) {
    if (!m_maintenance1) return;
    // Block until the presentation engine has signalled this fence, meaning it is
    // fully done consuming renderFinishedSems[imageIdx] from the previous present
    // of this swapchain image.  Then reset it so it can be passed to the next present.
    vkWaitForFences(m_ctx.device(), 1, &m_presentFences[imageIdx], VK_TRUE, UINT64_MAX);
    vkResetFences(m_ctx.device(), 1, &m_presentFences[imageIdx]);
}



VkResult Swapchain::acquireNextImage(uint32_t frameIdx, uint32_t& imageIdx) {
    auto& f = m_frames[frameIdx];
    vkWaitForFences(m_ctx.device(), 1, &f.inFlightFence, VK_TRUE, UINT64_MAX);
    VkResult res = vkAcquireNextImageKHR(m_ctx.device(), m_swapchain, UINT64_MAX,
                                          f.imageAvailableSemaphore,
                                          VK_NULL_HANDLE, &imageIdx);
    // Wait for (and reset) the present fence of the just-acquired image so that
    // the caller can safely signal renderFinishedSems[imageIdx] in the upcoming
    // submit without racing the presentation engine's semaphore consumption.
    waitPresentFence(imageIdx);
    return res;
}

VkResult Swapchain::present(uint32_t imageIdx) {
    VkSemaphore renderFinished = m_renderFinishedSems[imageIdx];
    VkPresentInfoKHR pi{};
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &renderFinished;
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &m_swapchain;
    pi.pImageIndices      = &imageIdx;

    // Attach the per-image present fence so the driver can signal it when the
    // presentation engine has consumed renderFinished (VK_KHR_swapchain_maintenance1).
    VkSwapchainPresentFenceInfoEXT fenceInfo{};
    if (m_maintenance1) {
        fenceInfo.sType          = VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_FENCE_INFO_EXT;
        fenceInfo.swapchainCount = 1;
        fenceInfo.pFences        = &m_presentFences[imageIdx];
        pi.pNext = &fenceInfo;
    }

    return vkQueuePresentKHR(m_ctx.presentQueue(), &pi);
}

void Swapchain::recreate(uint32_t width, uint32_t height) {
    m_ctx.waitIdle();
    destroy();
    create(width, height);
    createSyncObjects();
}

VkSurfaceFormatKHR Swapchain::chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const {
    // Prefer RGBA over BGRA so shader output vec4(r,g,b,a) maps naturally to screen.
    // If RGBA_SRGB is unavailable (rare on Windows), fall back to BGRA_SRGB — the
    // tonemap shader will swap R/B channels in that case.
    for (const auto& f : formats)
        if (f.format == VK_FORMAT_R8G8B8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return f;
    for (const auto& f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return f;
    return formats[0];
}

VkPresentModeKHR Swapchain::choosePresentMode(const std::vector<VkPresentModeKHR>& modes) const {
    if (!m_vsync) {
        for (auto m : modes) if (m == VK_PRESENT_MODE_IMMEDIATE_KHR)  return m;
        for (auto m : modes) if (m == VK_PRESENT_MODE_MAILBOX_KHR)    return m; // Triple buffer
    }
    return VK_PRESENT_MODE_FIFO_KHR; // vsync
}

VkExtent2D Swapchain::chooseExtent(const VkSurfaceCapabilitiesKHR& caps, uint32_t w, uint32_t h) const {
    if (caps.currentExtent.width != UINT32_MAX) return caps.currentExtent;
    return {
        std::clamp(w, caps.minImageExtent.width,  caps.maxImageExtent.width),
        std::clamp(h, caps.minImageExtent.height, caps.maxImageExtent.height)
    };
}

} // namespace vkgfx
