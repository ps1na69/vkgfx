// ─────────────────────────────────────────────────────────────────────────────
// VMA implementation — must appear in exactly ONE translation unit.
// ─────────────────────────────────────────────────────────────────────────────
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include "vkgfx/context.h"
#include <cstring>

namespace vkgfx {

const std::vector<const char*> Context::s_deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};
const std::vector<const char*> Context::s_validationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void*)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        std::cerr << "[Vulkan] " << data->pMessage << std::endl;
    return VK_FALSE;
}

static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCI, const VkAllocationCallbacks* pA,
    VkDebugUtilsMessengerEXT* pM)
{
    auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance,"vkCreateDebugUtilsMessengerEXT");
    return fn ? fn(instance, pCI, pA, pM) : VK_ERROR_EXTENSION_NOT_PRESENT;
}
static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT m, const VkAllocationCallbacks* pA)
{
    auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance,"vkDestroyDebugUtilsMessengerEXT");
    if (fn) fn(instance, m, pA);
}

Context::Context(const CreateInfo& ci) {
    m_preferDedicated = ci.preferDedicated;
    createInstance(ci);
#ifndef NDEBUG
    if (ci.enableValidation) setupDebugMessenger();
#endif
}

void Context::initDevice(VkSurfaceKHR surface) {
    pickPhysicalDevice(surface, m_preferDedicated);
    createLogicalDevice();
    createCommandPool();
    createVmaAllocator();
}

Context::~Context() {
    if (m_allocator)      vmaDestroyAllocator(m_allocator);
    if (m_commandPool)    vkDestroyCommandPool(m_device, m_commandPool, nullptr);
    if (m_device)         vkDestroyDevice(m_device, nullptr);
    if (m_debugMessenger) DestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
    if (m_instance)       vkDestroyInstance(m_instance, nullptr);
}

void Context::createVmaAllocator() {
    VmaVulkanFunctions vkFuncs{};
    vkFuncs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vkFuncs.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo ci{};
    ci.physicalDevice   = m_physicalDevice;
    ci.device           = m_device;
    ci.instance         = m_instance;
    ci.vulkanApiVersion = VK_API_VERSION_1_3;
    ci.pVulkanFunctions = &vkFuncs;

    VK_CHECK(vmaCreateAllocator(&ci, &m_allocator), "VMA allocator");
    std::cout << "[VKGFX] VMA allocator initialised\n";
}

void Context::createInstance(const CreateInfo& ci) {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = ci.appName.c_str();
    appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
    appInfo.pEngineName = "VKGFX";
    appInfo.engineVersion = VK_MAKE_VERSION(1,0,0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    uint32_t glfwCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwCount);
    std::vector<const char*> extensions(glfwExts, glfwExts + glfwCount);

    bool useValidation = ci.enableValidation;
#ifdef NDEBUG
    useValidation = false;
#endif
    if (useValidation) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    {
        uint32_t cnt = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &cnt, nullptr);
        std::vector<VkExtensionProperties> avail(cnt);
        vkEnumerateInstanceExtensionProperties(nullptr, &cnt, avail.data());
        bool hasSurfMaint1 = false, hasGetSurfCaps2 = false;
        for (const auto& ext : avail) {
            if (!strcmp(ext.extensionName, VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME)) hasSurfMaint1 = true;
            if (!strcmp(ext.extensionName, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME)) hasGetSurfCaps2 = true;
        }
        if (hasSurfMaint1 && hasGetSurfCaps2) {
            extensions.push_back(VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME);
            extensions.push_back(VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME);
            m_instanceSupportsMaint1 = true;
        }
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCI{};
    if (useValidation && checkValidationLayerSupport()) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(s_validationLayers.size());
        createInfo.ppEnabledLayerNames = s_validationLayers.data();
        debugCI.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCI.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugCI.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        debugCI.pfnUserCallback = debugCallback;
        createInfo.pNext = &debugCI;
    }
    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &m_instance), "Failed to create Vulkan instance");
}

void Context::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debugCallback;
    CreateDebugUtilsMessengerEXT(m_instance, &ci, nullptr, &m_debugMessenger);
}

void Context::pickPhysicalDevice(VkSurfaceKHR surface, bool preferDedicated) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    if (count == 0) throw std::runtime_error("No Vulkan-capable GPUs found");
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(m_instance, &count, devices.data());
    int bestScore = -1;
    for (auto dev : devices) {
        if (!isDeviceSuitable(dev, surface)) continue;
        int score = rateDevice(dev);
        if (!preferDedicated) score = 1;
        if (score > bestScore) { bestScore = score; m_physicalDevice = dev; }
    }
    if (m_physicalDevice == VK_NULL_HANDLE) throw std::runtime_error("No suitable GPU found");
    m_queueFamilies = findQueueFamilies(m_physicalDevice, surface);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
    std::cout << "[VKGFX] Using GPU: " << props.deviceName << std::endl;
}

void Context::createLogicalDevice() {
    std::vector<VkDeviceQueueCreateInfo> queueCIs;
    float priority = 1.0f;
    for (uint32_t family : m_queueFamilies.uniqueFamilies()) {
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = family;
        qi.queueCount = 1;
        qi.pQueuePriorities = &priority;
        queueCIs.push_back(qi);
    }
    VkPhysicalDeviceFeatures features{};
    features.samplerAnisotropy = VK_TRUE;
    features.fillModeNonSolid  = VK_TRUE;
    features.sampleRateShading = VK_TRUE;

    std::vector<const char*> enabledExts(s_deviceExtensions.begin(), s_deviceExtensions.end());
    if (m_instanceSupportsMaint1) {
        uint32_t cnt = 0;
        vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &cnt, nullptr);
        std::vector<VkExtensionProperties> avail(cnt);
        vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &cnt, avail.data());
        std::set<std::string> availNames;
        for (const auto& ext : avail) availNames.insert(ext.extensionName);
        const char* surfMaint1DevExt = nullptr;
        if (availNames.count(VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME))
            surfMaint1DevExt = VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME;
        else if (availNames.count("VK_KHR_surface_maintenance1"))
            surfMaint1DevExt = "VK_KHR_surface_maintenance1";
        if (surfMaint1DevExt && availNames.count(VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME)) {
            enabledExts.push_back(surfMaint1DevExt);
            enabledExts.push_back(VK_KHR_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME);
            m_hasMaintenance1 = true;
            std::cout << "[VKGFX] VK_KHR_swapchain_maintenance1 enabled\n";
        }
    }

    VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT maintenance1Features{};
    maintenance1Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT;
    maintenance1Features.swapchainMaintenance1 = VK_TRUE;

    VkDeviceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.pQueueCreateInfos = queueCIs.data();
    ci.queueCreateInfoCount = static_cast<uint32_t>(queueCIs.size());
    ci.pEnabledFeatures = &features;
    ci.enabledExtensionCount = static_cast<uint32_t>(enabledExts.size());
    ci.ppEnabledExtensionNames = enabledExts.data();
    if (m_hasMaintenance1) ci.pNext = &maintenance1Features;
    VK_CHECK(vkCreateDevice(m_physicalDevice, &ci, nullptr, &m_device), "Failed to create logical device");

    vkGetDeviceQueue(m_device, *m_queueFamilies.graphics, 0, &m_graphicsQueue);
    vkGetDeviceQueue(m_device, *m_queueFamilies.present,  0, &m_presentQueue);
    uint32_t xferFamily = m_queueFamilies.transfer.value_or(*m_queueFamilies.graphics);
    vkGetDeviceQueue(m_device, xferFamily, 0, &m_transferQueue);
}

void Context::createCommandPool() {
    VkCommandPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.queueFamilyIndex = *m_queueFamilies.graphics;
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(m_device, &ci, nullptr, &m_commandPool), "Failed to create command pool");
}

bool Context::isDeviceSuitable(VkPhysicalDevice dev, VkSurfaceKHR surface) const {
    auto idx = findQueueFamilies(dev, surface);
    if (!idx.isComplete()) return false;
    if (!checkDeviceExtensionSupport(dev)) return false;
    uint32_t formatCount = 0, presentCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &formatCount, nullptr);
    vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &presentCount, nullptr);
    return formatCount > 0 && presentCount > 0;
}

int Context::rateDevice(VkPhysicalDevice dev) const {
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceFeatures feats;
    vkGetPhysicalDeviceProperties(dev, &props);
    vkGetPhysicalDeviceFeatures(dev, &feats);
    int score = 0;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000;
    score += static_cast<int>(props.limits.maxImageDimension2D / 1024);
    if (!feats.samplerAnisotropy) score -= 500;
    return score;
}

QueueFamilyIndices Context::findQueueFamilies(VkPhysicalDevice dev, VkSurfaceKHR surface) const {
    QueueFamilyIndices idx;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, families.data());
    for (uint32_t i = 0; i < count; ++i) {
        const auto& f = families[i];
        if (f.queueFlags & VK_QUEUE_GRAPHICS_BIT) idx.graphics = i;
        if (f.queueFlags & VK_QUEUE_COMPUTE_BIT)  idx.compute  = i;
        if ((f.queueFlags & VK_QUEUE_TRANSFER_BIT) && !(f.queueFlags & VK_QUEUE_GRAPHICS_BIT)) idx.transfer = i;
        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &presentSupport);
        if (presentSupport) idx.present = i;
        if (idx.isComplete()) break;
    }
    return idx;
}

bool Context::checkDeviceExtensionSupport(VkPhysicalDevice dev) const {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, available.data());
    std::set<std::string> required(s_deviceExtensions.begin(), s_deviceExtensions.end());
    for (const auto& ext : available) required.erase(ext.extensionName);
    return required.empty();
}

bool Context::checkValidationLayerSupport() const {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const char* name : s_validationLayers) {
        bool found = false;
        for (const auto& l : layers) if (!strcmp(l.layerName, name)) { found = true; break; }
        if (!found) return false;
    }
    return true;
}

SwapchainSupportDetails Context::querySwapchainSupport(VkSurfaceKHR surface) const {
    return querySwapchainSupport(m_physicalDevice, surface);
}
SwapchainSupportDetails Context::querySwapchainSupport(VkPhysicalDevice dev, VkSurfaceKHR surface) const {
    SwapchainSupportDetails d;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, surface, &d.capabilities);
    uint32_t cnt;
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &cnt, nullptr);
    if (cnt) { d.formats.resize(cnt); vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &cnt, d.formats.data()); }
    vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &cnt, nullptr);
    if (cnt) { d.presentModes.resize(cnt); vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &cnt, d.presentModes.data()); }
    return d;
}

uint32_t Context::findMemoryType(uint32_t typeMask, VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        if ((typeMask & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("Failed to find suitable memory type");
}

VkFormat Context::findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) const {
    for (VkFormat fmt : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(m_physicalDevice, fmt, &props);
        if (tiling == VK_IMAGE_TILING_LINEAR  && (props.linearTilingFeatures  & features) == features) return fmt;
        if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) return fmt;
    }
    throw std::runtime_error("Failed to find supported format");
}

VkFormat Context::findDepthFormat() const {
    return findSupportedFormat({ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
        VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

VkSampleCountFlagBits Context::maxSampleCount() const {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(m_physicalDevice, &props);
    auto counts = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts;
    for (auto c : { VK_SAMPLE_COUNT_64_BIT, VK_SAMPLE_COUNT_32_BIT, VK_SAMPLE_COUNT_16_BIT,
                    VK_SAMPLE_COUNT_8_BIT,  VK_SAMPLE_COUNT_4_BIT,  VK_SAMPLE_COUNT_2_BIT })
        if (counts & c) return c;
    return VK_SAMPLE_COUNT_1_BIT;
}

// ─── Buffer (VMA) ─────────────────────────────────────────────────────────────
AllocatedBuffer Context::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                       VmaMemoryUsage vmaUsage, bool persistentlyMapped) const
{
    AllocatedBuffer buf;
    buf.size = size;

    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = size;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = vmaUsage;
    if (persistentlyMapped)
        aci.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
    // For host-visible, prefer write-combined (faster CPU writes)
    if (vmaUsage == VMA_MEMORY_USAGE_AUTO_PREFER_HOST)
        aci.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VmaAllocationInfo allocInfo{};
    VK_CHECK(vmaCreateBuffer(m_allocator, &bci, &aci, &buf.buffer, &buf.allocation, &allocInfo), "VMA createBuffer");
    if (persistentlyMapped) buf.mapped = allocInfo.pMappedData;
    return buf;
}

AllocatedBuffer Context::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                       VkMemoryPropertyFlags props) const
{
    VmaMemoryUsage vmaUsage = VMA_MEMORY_USAGE_AUTO;
    bool persistentMap = false;
    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        vmaUsage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        persistentMap = true;
    } else if (props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        vmaUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    }
    return createBuffer(size, usage, vmaUsage, persistentMap);
}

void Context::destroyBuffer(AllocatedBuffer& buf) const {
    if (buf.buffer) {
        vmaDestroyBuffer(m_allocator, buf.buffer, buf.allocation);
        buf.buffer = VK_NULL_HANDLE; buf.allocation = VK_NULL_HANDLE; buf.mapped = nullptr;
    }
}

// ─── Image (VMA) ──────────────────────────────────────────────────────────────
AllocatedImage Context::createImage(uint32_t w, uint32_t h, uint32_t mipLevels,
                                     VkSampleCountFlagBits samples, VkFormat format,
                                     VkImageTiling tiling, VkImageUsageFlags usage,
                                     VkMemoryPropertyFlags props, uint32_t arrayLayers) const
{
    AllocatedImage img;
    img.format = format; img.mipLevels = mipLevels;

    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.extent        = { w, h, 1 };
    ici.mipLevels     = mipLevels;
    ici.arrayLayers   = arrayLayers;
    ici.format        = format;
    ici.tiling        = tiling;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.usage         = usage;
    ici.samples       = samples;
    ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = (props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                ? VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE : VMA_MEMORY_USAGE_AUTO;

    VK_CHECK(vmaCreateImage(m_allocator, &ici, &aci, &img.image, &img.allocation, nullptr), "VMA createImage");
    return img;
}

void Context::createImageView(AllocatedImage& img, VkImageAspectFlags aspectMask) const {
    VkImageViewCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ci.image = img.image; ci.viewType = VK_IMAGE_VIEW_TYPE_2D; ci.format = img.format;
    ci.subresourceRange = { aspectMask, 0, img.mipLevels, 0, 1 };
    VK_CHECK(vkCreateImageView(m_device, &ci, nullptr, &img.view), "createImageView");
}

void Context::destroyImage(AllocatedImage& img) const {
    if (img.view) { vkDestroyImageView(m_device, img.view, nullptr); img.view = VK_NULL_HANDLE; }
    if (img.image) {
        vmaDestroyImage(m_allocator, img.image, img.allocation);
        img.image = VK_NULL_HANDLE; img.allocation = VK_NULL_HANDLE;
    }
}

// ─── Command helpers ──────────────────────────────────────────────────────────
VkCommandBuffer Context::beginSingleTimeCommands() const {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandPool = m_commandPool; ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(m_device, &ai, &cmd);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}

void Context::endSingleTimeCommands(VkCommandBuffer cmd) const {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VkFenceCreateInfo fi{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence; vkCreateFence(m_device, &fi, nullptr, &fence);
    vkQueueSubmit(m_graphicsQueue, 1, &si, fence);
    vkWaitForFences(m_device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(m_device, fence, nullptr);
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmd);
}

void Context::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) const {
    auto cmd = beginSingleTimeCommands();
    VkBufferCopy region{ 0, 0, size };
    vkCmdCopyBuffer(cmd, src, dst, 1, &region);
    endSingleTimeCommands(cmd);
}

void Context::copyBufferToImage(VkBuffer src, VkImage dst, uint32_t w, uint32_t h) const {
    auto cmd = beginSingleTimeCommands();
    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { w, h, 1 };
    vkCmdCopyBufferToImage(cmd, src, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    endSingleTimeCommands(cmd);
}

void Context::transitionImageLayout(VkImage image, VkFormat, VkImageLayout oldLayout,
                                     VkImageLayout newLayout, uint32_t mipLevels) const
{
    auto cmd = beginSingleTimeCommands();
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout; barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 1 };
    VkPipelineStageFlags srcStage, dstStage;
    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0; barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT; dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT; dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else { srcStage = dstStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; }
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    endSingleTimeCommands(cmd);
}

void Context::generateMipmaps(VkImage image, VkFormat format, uint32_t w, uint32_t h, uint32_t mipLevels) const {
    VkFormatProperties props;
    vkGetPhysicalDeviceFormatProperties(m_physicalDevice, format, &props);
    if (!(props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
        throw std::runtime_error("Texture format does not support linear blitting");
    auto cmd = beginSingleTimeCommands();
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    int32_t mipW = w, mipH = h;
    for (uint32_t i = 1; i < mipLevels; ++i) {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        VkImageBlit blit{};
        blit.srcOffsets[1] = { mipW, mipH, 1 };
        blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i-1, 0, 1 };
        blit.dstOffsets[1] = { mipW > 1 ? mipW/2 : 1, mipH > 1 ? mipH/2 : 1, 1 };
        blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1 };
        vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        if (mipW > 1) mipW /= 2; if (mipH > 1) mipH /= 2;
    }
    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    endSingleTimeCommands(cmd);
}

AllocatedBuffer Context::uploadBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage) const {
    auto staging = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VMA_MEMORY_USAGE_AUTO_PREFER_HOST, /*persistentlyMapped=*/true);
    std::memcpy(staging.mapped, data, static_cast<size_t>(size));
    auto dest = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
                             VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, /*persistentlyMapped=*/false);
    copyBuffer(staging.buffer, dest.buffer, size);
    destroyBuffer(staging);
    return dest;
}

} // namespace vkgfx
