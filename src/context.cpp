// src/context.cpp
// RULE: every VkStructure is zero-initialised via = {} before setting sType.
// RULE: vkDeviceWaitIdle is called before any resource is destroyed.

#include <vkgfx/context.h>
#include <vkgfx/window.h>

// VMA — define implementation in exactly one translation unit
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <stdexcept>
#include <iostream>
#include <vector>
#include <set>
#include <cstring>

namespace vkgfx {

// ── Validation callback ───────────────────────────────────────────────────────

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
    VkDebugUtilsMessageTypeFlagsEXT             /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* /*user*/)
{
    const char* prefix = (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
                         ? "[VK ERROR]" : "[VK WARN ]";
    std::cerr << prefix << " " << data->pMessage << "\n";
    return VK_FALSE;
}

static VkDebugUtilsMessengerCreateInfoEXT makeDebugCI() {
    VkDebugUtilsMessengerCreateInfoEXT ci{};      // zero-init
    ci.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debugCallback;
    return ci;
}

// ── Context ───────────────────────────────────────────────────────────────────

Context::Context(const ContextConfig& cfg) : m_cfg(cfg) {
    createInstance(cfg);
    if (cfg.validation) setupDebugMessenger();
    pickPhysicalDevice();
    createLogicalDevice(cfg);
    createVMA();
    createCommandPool();
}

Context::~Context() {
    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }

    // Destroy command pool BEFORE device — explicit reset so the VkHandle
    // destructor does not fire after vkDestroyDevice below.
    m_graphicsPool.reset();

    if (m_vma) {
        vmaDestroyAllocator(m_vma);
        m_vma = nullptr;
    }

    if (m_device != VK_NULL_HANDLE) {
        vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
    }

    if (m_debugMessenger != VK_NULL_HANDLE) {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) fn(m_instance, m_debugMessenger, nullptr);
    }

    if (m_surface != VK_NULL_HANDLE)
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);

    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }
}

// ── createInstance ────────────────────────────────────────────────────────────

void Context::createInstance(const ContextConfig& cfg) {
    VkApplicationInfo appInfo{};                              // zero-init
    appInfo.sType            = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = cfg.appName.c_str();
    appInfo.applicationVersion = VK_MAKE_VERSION(2, 0, 0);
    appInfo.pEngineName      = "vkgfx";
    appInfo.engineVersion    = VK_MAKE_VERSION(2, 0, 0);
    appInfo.apiVersion       = VK_API_VERSION_1_3;

    std::vector<const char*> extensions;
    if (!cfg.headless) {
        auto glfwExt = Window::requiredExtensions();
        extensions.insert(extensions.end(), glfwExt.begin(), glfwExt.end());
    }
    if (cfg.validation)
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    std::vector<const char*> layers;
    if (cfg.validation)
        layers.push_back("VK_LAYER_KHRONOS_validation");

    VkInstanceCreateInfo ci{};                                // zero-init
    ci.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo        = &appInfo;
    ci.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();
    ci.enabledLayerCount       = static_cast<uint32_t>(layers.size());
    ci.ppEnabledLayerNames     = layers.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCI{};
    if (cfg.validation) {
        debugCI = makeDebugCI();
        ci.pNext = &debugCI;
    }

    VkResult res = vkCreateInstance(&ci, nullptr, &m_instance);
    if (res != VK_SUCCESS)
        throw std::runtime_error("[vkgfx] vkCreateInstance failed: " + std::to_string(res));
}

// ── setupDebugMessenger ───────────────────────────────────────────────────────

void Context::setupDebugMessenger() {
    auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
    if (!fn) { std::cerr << "[vkgfx] vkCreateDebugUtilsMessengerEXT not found\n"; return; }

    auto ci = makeDebugCI();
    fn(m_instance, &ci, nullptr, &m_debugMessenger);
}

// ── pickPhysicalDevice ────────────────────────────────────────────────────────

static QueueFamilies findQueueFamilies(VkPhysicalDevice dev, VkSurfaceKHR surface) {
    QueueFamilies qf;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, props.data());

    for (uint32_t i = 0; i < count; ++i) {
        if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) qf.graphics = i;
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT)  qf.compute  = i;

        if (surface != VK_NULL_HANDLE) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &present);
            if (present) qf.present = i;
        } else {
            qf.present = qf.graphics;  // headless
        }
    }
    return qf;
}

void Context::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    if (count == 0)
        throw std::runtime_error("[vkgfx] No Vulkan physical device found");

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(m_instance, &count, devices.data());

    for (auto dev : devices) {
        auto qf = findQueueFamilies(dev, m_surface);
        if (qf.complete()) {
            m_gpu = dev;
            m_qf  = qf;
            break;
        }
    }

    if (m_gpu == VK_NULL_HANDLE)
        throw std::runtime_error("[vkgfx] No suitable GPU found");

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_gpu, &props);
    std::cout << "[vkgfx] GPU: " << props.deviceName << "\n";
}

// ── createLogicalDevice ───────────────────────────────────────────────────────

void Context::createLogicalDevice(const ContextConfig& cfg) {
    std::set<uint32_t> uniqueQueues = {m_qf.graphics, m_qf.present};
    if (m_qf.compute != UINT32_MAX) uniqueQueues.insert(m_qf.compute);

    float prio = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCIs;
    for (uint32_t family : uniqueQueues) {
        VkDeviceQueueCreateInfo qi{};                         // zero-init
        qi.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = family;
        qi.queueCount       = 1;
        qi.pQueuePriorities = &prio;
        queueCIs.push_back(qi);
    }

    VkPhysicalDeviceFeatures features{};                      // zero-init
    features.samplerAnisotropy = VK_TRUE;
    features.shaderStorageImageWriteWithoutFormat = VK_TRUE;  // for compute IBL

    std::vector<const char*> deviceExts = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    VkDeviceCreateInfo ci{};                                  // zero-init
    ci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.queueCreateInfoCount    = static_cast<uint32_t>(queueCIs.size());
    ci.pQueueCreateInfos       = queueCIs.data();
    ci.pEnabledFeatures        = &features;
    ci.enabledExtensionCount   = static_cast<uint32_t>(deviceExts.size());
    ci.ppEnabledExtensionNames = deviceExts.data();

    if (cfg.validation) {
        static const char* val = "VK_LAYER_KHRONOS_validation";
        ci.enabledLayerCount   = 1;
        ci.ppEnabledLayerNames = &val;
    }

    VkResult res = vkCreateDevice(m_gpu, &ci, nullptr, &m_device);
    if (res != VK_SUCCESS)
        throw std::runtime_error("[vkgfx] vkCreateDevice failed");

    vkGetDeviceQueue(m_device, m_qf.graphics, 0, &m_graphicsQ);
    vkGetDeviceQueue(m_device, m_qf.present,  0, &m_presentQ);
    if (m_qf.compute != UINT32_MAX)
        vkGetDeviceQueue(m_device, m_qf.compute, 0, &m_computeQ);
}

// ── createVMA ─────────────────────────────────────────────────────────────────

void Context::createVMA() {
    VmaAllocatorCreateInfo ai{};                              // zero-init
    ai.physicalDevice = m_gpu;
    ai.device         = m_device;
    ai.instance       = m_instance;
    ai.vulkanApiVersion = VK_API_VERSION_1_3;

    VkResult res = vmaCreateAllocator(&ai, &m_vma);
    if (res != VK_SUCCESS)
        throw std::runtime_error("[vkgfx] vmaCreateAllocator failed");
}

// ── createCommandPool ─────────────────────────────────────────────────────────

void Context::createCommandPool() {
    m_graphicsPool = makeCommandPool(m_device, m_qf.graphics);
}

// ── Buffer helpers ────────────────────────────────────────────────────────────

AllocatedBuffer Context::allocateBuffer(VkDeviceSize size,
                                         VkBufferUsageFlags usage,
                                         bool hostVisible) const {
    VkBufferCreateInfo bi{};                                  // zero-init
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size  = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo ai{};
    ai.usage = hostVisible
               ? VMA_MEMORY_USAGE_CPU_TO_GPU
               : VMA_MEMORY_USAGE_GPU_ONLY;
    if (hostVisible)
        ai.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;

    AllocatedBuffer buf{};
    vmaCreateBuffer(m_vma, &bi, &ai,
                    &buf.buffer, (VmaAllocation*)&buf.allocation, nullptr);
    return buf;
}

AllocatedBuffer Context::uploadBuffer(const void* data, VkDeviceSize size,
                                       VkBufferUsageFlags usage) const {
    // Staging buffer (host visible)
    AllocatedBuffer staging = allocateBuffer(size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);

    void* mapped = nullptr;
    vmaMapMemory(m_vma, (VmaAllocation)staging.allocation, &mapped);
    std::memcpy(mapped, data, static_cast<size_t>(size));
    vmaUnmapMemory(m_vma, (VmaAllocation)staging.allocation);

    // Device-local buffer
    AllocatedBuffer dst = allocateBuffer(size,
        usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);

    VkCommandBuffer cmd = beginOneShot();
    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &region);
    endOneShot(cmd);

    vmaDestroyBuffer(m_vma, staging.buffer, (VmaAllocation)staging.allocation);
    return dst;
}

void Context::destroyBuffer(AllocatedBuffer& buf) const {
    if (buf.buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(m_vma, buf.buffer, (VmaAllocation)buf.allocation);
    buf.buffer     = VK_NULL_HANDLE;
    buf.allocation = nullptr;
}

// ── Image helpers ─────────────────────────────────────────────────────────────

AllocatedImage Context::allocateImage(VkExtent2D extent, VkFormat format,
                                       VkImageUsageFlags usage,
                                       uint32_t mipLevels, uint32_t layers,
                                       VkImageCreateFlags flags) const {
    VkImageCreateInfo ii{};                                   // zero-init
    ii.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ii.imageType     = VK_IMAGE_TYPE_2D;
    ii.format        = format;
    ii.extent        = {extent.width, extent.height, 1};
    ii.mipLevels     = mipLevels;
    ii.arrayLayers   = layers;
    ii.samples       = VK_SAMPLE_COUNT_1_BIT;
    ii.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ii.usage         = usage;
    ii.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ii.flags         = flags;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    AllocatedImage img{};
    img.format    = format;
    img.extent    = extent;
    img.mipLevels = mipLevels;
    vmaCreateImage(m_vma, &ii, &ai,
                   &img.image, (VmaAllocation*)&img.allocation, nullptr);
    return img;
}

void Context::destroyImage(AllocatedImage& img) const {
    if (img.view  != VK_NULL_HANDLE) vkDestroyImageView(m_device, img.view,  nullptr);
    if (img.image != VK_NULL_HANDLE) vmaDestroyImage(m_vma, img.image, (VmaAllocation)img.allocation);
    img = AllocatedImage{};
}

VkImageView Context::createImageView(VkImage image, VkFormat format,
                                      VkImageAspectFlags aspect,
                                      uint32_t mipLevels, uint32_t layers,
                                      VkImageViewType viewType) const {
    VkImageViewCreateInfo vi{};                               // zero-init
    vi.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image    = image;
    vi.viewType = viewType;
    vi.format   = format;
    vi.subresourceRange.aspectMask     = aspect;
    vi.subresourceRange.baseMipLevel   = 0;
    vi.subresourceRange.levelCount     = mipLevels;
    vi.subresourceRange.baseArrayLayer = 0;
    vi.subresourceRange.layerCount     = layers;

    VkImageView v = VK_NULL_HANDLE;
    VkResult res  = vkCreateImageView(m_device, &vi, nullptr, &v);
    if (res != VK_SUCCESS)
        throw std::runtime_error("[vkgfx] vkCreateImageView failed");
    return v;
}

// ── One-shot commands ─────────────────────────────────────────────────────────

VkCommandBuffer Context::beginOneShot() const {
    VkCommandBufferAllocateInfo ai{};                         // zero-init
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = m_graphicsPool.get();
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(m_device, &ai, &cmd);

    VkCommandBufferBeginInfo bi{};                            // zero-init
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void Context::endOneShot(VkCommandBuffer cmd) const {
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};                                        // zero-init
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;

    vkQueueSubmit(m_graphicsQ, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_graphicsQ);
    vkFreeCommandBuffers(m_device, m_graphicsPool.get(), 1, &cmd);
}

// ── Format queries ────────────────────────────────────────────────────────────

VkFormat Context::findSupportedFormat(const std::vector<VkFormat>& candidates,
                                       VkImageTiling tiling,
                                       VkFormatFeatureFlags features) const {
    for (VkFormat f : candidates) {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(m_gpu, f, &props);
        bool ok = (tiling == VK_IMAGE_TILING_LINEAR  && (props.linearTilingFeatures  & features) == features)
               || (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features);
        if (ok) return f;
    }
    throw std::runtime_error("[vkgfx] No supported format found");
}

VkFormat Context::findDepthFormat() const {
    return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

} // namespace vkgfx
