#pragma once
// include/vkgfx/vk_raii.h
// Lightweight RAII wrappers for Vulkan resources.
// Every wrapper:
//   - Zero-inits the underlying handle
//   - Destroys on scope exit via the provided deleter
//   - Is move-only (no copy)

#include <vulkan/vulkan.h>
#include <functional>
#include <utility>

namespace vkgfx {

/// Generic move-only RAII handle
template<typename T>
class VkHandle {
public:
    using Deleter = std::function<void(T)>;

    VkHandle() = default;
    explicit VkHandle(T handle, Deleter del)
        : m_handle(handle), m_del(std::move(del)) {}

    ~VkHandle() { reset(); }

    VkHandle(const VkHandle&) = delete;
    VkHandle& operator=(const VkHandle&) = delete;

    VkHandle(VkHandle&& o) noexcept : m_handle(o.m_handle), m_del(std::move(o.m_del)) {
        o.m_handle = VK_NULL_HANDLE;
    }
    VkHandle& operator=(VkHandle&& o) noexcept {
        if (this != &o) { reset(); m_handle = o.m_handle; m_del = std::move(o.m_del); o.m_handle = VK_NULL_HANDLE; }
        return *this;
    }

    void reset() {
        if (m_handle != VK_NULL_HANDLE && m_del) { m_del(m_handle); m_handle = VK_NULL_HANDLE; }
    }

    [[nodiscard]] T  get()       const { return m_handle; }
    [[nodiscard]] T* ptr()             { return &m_handle; }
    [[nodiscard]] explicit operator bool() const { return m_handle != VK_NULL_HANDLE; }

private:
    T       m_handle = VK_NULL_HANDLE;
    Deleter m_del;
};

// ── Convenience factory functions ─────────────────────────────────────────────

inline VkHandle<VkSemaphore> makeSemaphore(VkDevice dev) {
    VkSemaphoreCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkSemaphore s = VK_NULL_HANDLE;
    vkCreateSemaphore(dev, &ci, nullptr, &s);
    return VkHandle<VkSemaphore>(s, [dev](VkSemaphore x){ vkDestroySemaphore(dev, x, nullptr); });
}

inline VkHandle<VkFence> makeFence(VkDevice dev, bool signaled = true) {
    VkFenceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (signaled) ci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkFence f = VK_NULL_HANDLE;
    vkCreateFence(dev, &ci, nullptr, &f);
    return VkHandle<VkFence>(f, [dev](VkFence x){ vkDestroyFence(dev, x, nullptr); });
}

inline VkHandle<VkCommandPool> makeCommandPool(VkDevice dev, uint32_t queueFamily,
                                                VkCommandPoolCreateFlags flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT) {
    VkCommandPoolCreateInfo ci{};
    ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    ci.queueFamilyIndex = queueFamily;
    ci.flags            = flags;
    VkCommandPool p = VK_NULL_HANDLE;
    vkCreateCommandPool(dev, &ci, nullptr, &p);
    return VkHandle<VkCommandPool>(p, [dev](VkCommandPool x){ vkDestroyCommandPool(dev, x, nullptr); });
}

inline VkHandle<VkDescriptorPool> makeDescriptorPool(VkDevice dev,
    const VkDescriptorPoolSize* sizes, uint32_t sizeCount, uint32_t maxSets) {
    VkDescriptorPoolCreateInfo ci{};
    ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.poolSizeCount = sizeCount;
    ci.pPoolSizes    = sizes;
    ci.maxSets       = maxSets;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(dev, &ci, nullptr, &pool);
    return VkHandle<VkDescriptorPool>(pool, [dev](VkDescriptorPool x){ vkDestroyDescriptorPool(dev, x, nullptr); });
}

/// VMA-allocated buffer (holds handle + allocation together)
struct AllocatedBuffer {
    VkBuffer      buffer     = VK_NULL_HANDLE;
    void*         allocation = nullptr; ///< VmaAllocation opaque ptr
};

/// VMA-allocated image
struct AllocatedImage {
    VkImage       image      = VK_NULL_HANDLE;
    VkImageView   view       = VK_NULL_HANDLE;
    void*         allocation = nullptr; ///< VmaAllocation opaque ptr
    VkFormat      format     = VK_FORMAT_UNDEFINED;
    VkExtent2D    extent     = {};
    uint32_t      mipLevels  = 1;
};

} // namespace vkgfx
