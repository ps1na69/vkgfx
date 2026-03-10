#pragma once
#include "types.h"

// ── Vulkan Memory Allocator ───────────────────────────────────────────────────
// VMA must be included before any Vulkan headers it wraps.
// The implementation (VMA_IMPLEMENTATION) lives in a single .cpp file (context.cpp).
#include <vk_mem_alloc.h>

namespace vkgfx {

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;
    std::optional<uint32_t> transfer;
    std::optional<uint32_t> compute;

    [[nodiscard]] bool isComplete() const { return graphics && present; }
    [[nodiscard]] std::set<uint32_t> uniqueFamilies() const {
        std::set<uint32_t> s;
        if (graphics) s.insert(*graphics);
        if (present)  s.insert(*present);
        if (transfer) s.insert(*transfer);
        if (compute)  s.insert(*compute);
        return s;
    }
};

struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR        capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
};

// ── AllocatedBuffer ───────────────────────────────────────────────────────────
// Extended with VMA allocation handle and persistent-mapping pointer.
// When mapped != nullptr the buffer is HOST_VISIBLE and permanently mapped —
// use memcpy(mapped, data, size) instead of vkMapMemory/vkUnmapMemory.
struct AllocatedBuffer {
    VkBuffer       buffer     = VK_NULL_HANDLE;
    VmaAllocation  allocation = VK_NULL_HANDLE;
    VkDeviceSize   size       = 0;
    void*          mapped     = nullptr; ///< Non-null when persistently mapped
};

// ── AllocatedImage ────────────────────────────────────────────────────────────
struct AllocatedImage {
    VkImage        image      = VK_NULL_HANDLE;
    VmaAllocation  allocation = VK_NULL_HANDLE;
    VkImageView    view       = VK_NULL_HANDLE;
    VkFormat       format     = VK_FORMAT_UNDEFINED;
    uint32_t       mipLevels  = 1;
};

class Context {
public:
    struct CreateInfo {
        std::string appName          = "VKGFX App";
        bool        enableValidation = true;
        bool        preferDedicated  = true;
    };

    explicit Context(const CreateInfo& ci = {});
    ~Context();

    void initDevice(VkSurfaceKHR surface);

    Context(const Context&)            = delete;
    Context& operator=(const Context&) = delete;

    [[nodiscard]] VkInstance       instance()       const { return m_instance; }
    [[nodiscard]] VkPhysicalDevice physicalDevice() const { return m_physicalDevice; }
    [[nodiscard]] VkDevice         device()         const { return m_device; }
    [[nodiscard]] VkQueue          graphicsQueue()  const { return m_graphicsQueue; }
    [[nodiscard]] VkQueue          presentQueue()   const { return m_presentQueue; }
    [[nodiscard]] VkQueue          transferQueue()  const { return m_transferQueue; }
    [[nodiscard]] VkCommandPool    commandPool()    const { return m_commandPool; }
    [[nodiscard]] VmaAllocator     allocator()      const { return m_allocator; }
    [[nodiscard]] const QueueFamilyIndices& queueFamilies() const { return m_queueFamilies; }

    [[nodiscard]] SwapchainSupportDetails querySwapchainSupport(VkSurfaceKHR surface) const;
    [[nodiscard]] SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice dev,
                                                                VkSurfaceKHR surface) const;

    [[nodiscard]] uint32_t findMemoryType(uint32_t typeMask, VkMemoryPropertyFlags props) const;
    [[nodiscard]] VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates,
                                               VkImageTiling tiling,
                                               VkFormatFeatureFlags features) const;
    [[nodiscard]] VkFormat findDepthFormat() const;
    [[nodiscard]] VkSampleCountFlagBits maxSampleCount() const;

    // ── Buffer allocation (VMA-backed) ────────────────────────────────────────
    // persistentlyMapped: if true and the usage flags include HOST_VISIBLE,
    // the returned buffer has its 'mapped' pointer set permanently.
    [[nodiscard]] AllocatedBuffer createBuffer(VkDeviceSize size,
                                               VkBufferUsageFlags usage,
                                               VmaMemoryUsage vmaUsage,
                                               bool persistentlyMapped = false) const;

    // Legacy overload: maps VkMemoryPropertyFlags to a VmaMemoryUsage heuristic
    [[nodiscard]] AllocatedBuffer createBuffer(VkDeviceSize size,
                                               VkBufferUsageFlags usage,
                                               VkMemoryPropertyFlags props) const;

    void destroyBuffer(AllocatedBuffer& buf) const;

    // ── Image allocation (VMA-backed) ─────────────────────────────────────────
    [[nodiscard]] AllocatedImage createImage(uint32_t w, uint32_t h,
                                             uint32_t mipLevels,
                                             VkSampleCountFlagBits samples,
                                             VkFormat format,
                                             VkImageTiling tiling,
                                             VkImageUsageFlags usage,
                                             VkMemoryPropertyFlags props,
                                             uint32_t arrayLayers = 1) const;
    void createImageView(AllocatedImage& img, VkImageAspectFlags aspectMask) const;
    void destroyImage(AllocatedImage& img) const;

    // ── Command helpers ───────────────────────────────────────────────────────
    [[nodiscard]] VkCommandBuffer beginSingleTimeCommands() const;
    void endSingleTimeCommands(VkCommandBuffer cmd) const;

    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) const;
    void copyBufferToImage(VkBuffer src, VkImage dst,
                           uint32_t w, uint32_t h) const;
    void transitionImageLayout(VkImage image, VkFormat format,
                               VkImageLayout oldLayout,
                               VkImageLayout newLayout,
                               uint32_t mipLevels = 1) const;
    void generateMipmaps(VkImage image, VkFormat format,
                         uint32_t w, uint32_t h,
                         uint32_t mipLevels) const;

    [[nodiscard]] AllocatedBuffer uploadBuffer(const void* data,
                                               VkDeviceSize size,
                                               VkBufferUsageFlags usage) const;

    void waitIdle() const { vkDeviceWaitIdle(m_device); }

    [[nodiscard]] bool hasMaintenance1() const { return m_hasMaintenance1; }

private:
    void createInstance(const CreateInfo& ci);
    void setupDebugMessenger();
    void pickPhysicalDevice(VkSurfaceKHR surface, bool preferDedicated);
    void createLogicalDevice();
    void createCommandPool();
    void createVmaAllocator();     ///< NEW: initialise VMA after device creation

    bool isDeviceSuitable(VkPhysicalDevice dev, VkSurfaceKHR surface) const;
    int  rateDevice(VkPhysicalDevice dev) const;
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev, VkSurfaceKHR surface) const;
    bool checkDeviceExtensionSupport(VkPhysicalDevice dev) const;
    bool checkValidationLayerSupport() const;

    VkInstance               m_instance       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice         m_physicalDevice = VK_NULL_HANDLE;
    VkDevice                 m_device         = VK_NULL_HANDLE;
    VkQueue                  m_graphicsQueue  = VK_NULL_HANDLE;
    VkQueue                  m_presentQueue   = VK_NULL_HANDLE;
    VkQueue                  m_transferQueue  = VK_NULL_HANDLE;
    VkCommandPool            m_commandPool    = VK_NULL_HANDLE;
    VmaAllocator             m_allocator      = VK_NULL_HANDLE; ///< NEW
    QueueFamilyIndices       m_queueFamilies;
    bool                     m_preferDedicated        = true;
    bool                     m_hasMaintenance1        = false;
    bool                     m_instanceSupportsMaint1 = false;

    static const std::vector<const char*> s_deviceExtensions;
    static const std::vector<const char*> s_validationLayers;
};

} // namespace vkgfx
