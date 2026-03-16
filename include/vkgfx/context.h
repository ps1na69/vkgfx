#pragma once
// include/vkgfx/context.h
// Owns: VkInstance, VkPhysicalDevice, VkDevice, queues, VMA allocator.
// Zero-inits every Vulkan struct.  Waits for device idle on destroy.

#include "vk_raii.h"
#include <vulkan/vulkan.h>
#include <string>
#include <vector>
#include <functional>
#include <cstdint>

// Forward-declare VmaAllocator to avoid pulling in vk_mem_alloc.h in headers
struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T*;

struct VmaAllocationCreateInfo;

namespace vkgfx {

struct ContextConfig {
    bool        validation = false;
    bool        headless   = false;   // no surface / no swapchain needed (tests)
    std::string appName    = "vkgfx";
};

struct QueueFamilies {
    uint32_t graphics = UINT32_MAX;
    uint32_t present  = UINT32_MAX;
    uint32_t compute  = UINT32_MAX;
    [[nodiscard]] bool complete() const {
        return graphics != UINT32_MAX && present != UINT32_MAX;
    }
};

class Context {
public:
    explicit Context(const ContextConfig& cfg = {});
    ~Context();

    Context(const Context&)            = delete;
    Context& operator=(const Context&) = delete;

    // ── Accessors ─────────────────────────────────────────────────────────────
    [[nodiscard]] bool              isValid()     const { return m_device != VK_NULL_HANDLE; }
    [[nodiscard]] VkInstance        instance()    const { return m_instance; }
    [[nodiscard]] VkPhysicalDevice  gpu()         const { return m_gpu; }
    [[nodiscard]] VkDevice          device()      const { return m_device; }
    [[nodiscard]] VkQueue           graphicsQ()   const { return m_graphicsQ; }
    [[nodiscard]] VkQueue           presentQ()    const { return m_presentQ; }
    [[nodiscard]] VkQueue           computeQ()    const { return m_computeQ; }
    [[nodiscard]] VmaAllocator      vma()         const { return m_vma; }
    [[nodiscard]] const QueueFamilies& queues()   const { return m_qf; }
    [[nodiscard]] VkCommandPool     graphicsPool()const { return m_graphicsPool.get(); }

    // ── Buffer helpers ────────────────────────────────────────────────────────
    /// Upload data to a DEVICE_LOCAL buffer via staging. Usage flags OR'd with TRANSFER_DST.
    AllocatedBuffer uploadBuffer(const void* data, VkDeviceSize size,
                                  VkBufferUsageFlags usage) const;

    /// Allocate a host-visible buffer mapped for UBO / staging use.
    AllocatedBuffer allocateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                    bool hostVisible = true) const;

    void destroyBuffer(AllocatedBuffer& buf) const;

    // ── Image helpers ─────────────────────────────────────────────────────────
    AllocatedImage allocateImage(VkExtent2D extent, VkFormat format,
                                  VkImageUsageFlags usage,
                                  uint32_t mipLevels  = 1,
                                  uint32_t layers     = 1,
                                  VkImageCreateFlags  flags = 0) const;

    void destroyImage(AllocatedImage& img) const;

    VkImageView createImageView(VkImage image, VkFormat format,
                                 VkImageAspectFlags aspect,
                                 uint32_t mipLevels = 1,
                                 uint32_t layers    = 1,
                                 VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D) const;

    // ── Command helpers ───────────────────────────────────────────────────────
    VkCommandBuffer beginOneShot() const;
    void            endOneShot(VkCommandBuffer cmd) const;

    // ── Surface (set before device creation when not headless) ───────────────
    void setSurface(VkSurfaceKHR surface) { m_surface = surface; }
    [[nodiscard]] VkSurfaceKHR surface() const { return m_surface; }

    // ── Format queries ────────────────────────────────────────────────────────
    [[nodiscard]] VkFormat findDepthFormat() const;
    [[nodiscard]] VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates,
                                                VkImageTiling tiling,
                                                VkFormatFeatureFlags features) const;

private:
    void createInstance(const ContextConfig& cfg);
    void setupDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDevice(const ContextConfig& cfg);
    void createVMA();
    void createCommandPool();

    ContextConfig     m_cfg;
    VkInstance        m_instance   = VK_NULL_HANDLE;
    VkSurfaceKHR      m_surface    = VK_NULL_HANDLE;
    VkPhysicalDevice  m_gpu        = VK_NULL_HANDLE;
    VkDevice          m_device     = VK_NULL_HANDLE;
    VkQueue           m_graphicsQ  = VK_NULL_HANDLE;
    VkQueue           m_presentQ   = VK_NULL_HANDLE;
    VkQueue           m_computeQ   = VK_NULL_HANDLE;
    QueueFamilies     m_qf;
    VmaAllocator      m_vma        = nullptr;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkHandle<VkCommandPool>  m_graphicsPool;
};

} // namespace vkgfx
