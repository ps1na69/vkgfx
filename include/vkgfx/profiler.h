#pragma once
// include/vkgfx/profiler.h
//
// GpuProfiler — per-pass GPU timestamp queries, VRAM budget tracking,
// and an ImGui overlay window.
//
// All methods compile to no-ops when VKGFX_ENABLE_PROFILING is not defined,
// so user code never needs #ifdef guards.

#include <vulkan/vulkan.h>
#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <cstdint>

struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T*;

namespace vkgfx {

class Context;

// ── Public stats types ────────────────────────────────────────────────────────

struct PassTiming {
    std::string name;
    float       gpuMs = 0.f;
};

struct FrameStats {
    std::vector<PassTiming> passes;
    float    totalGpuMs   = 0.f;
    float    cpuFrameMs   = 0.f;
    uint32_t drawCalls    = 0;
    uint64_t triangles    = 0;
    float    vramUsedMB   = 0.f;
    float    vramBudgetMB = 0.f;
};

// ── GpuProfiler ──────────────────────────────────────────────────────────────

class GpuProfiler {
public:
    static constexpr uint32_t MAX_PASSES = 16;

    GpuProfiler()  = default;
    ~GpuProfiler() = default;
    GpuProfiler(const GpuProfiler&)            = delete;
    GpuProfiler& operator=(const GpuProfiler&) = delete;

    // ── Lifecycle ─────────────────────────────────────────────────────────────
    void init   (Context& ctx, uint32_t framesInFlight);
    void destroy(VkDevice device);

    [[nodiscard]] bool ready() const { return m_queryPool != VK_NULL_HANDLE; }

    // ── Per-frame GPU commands ────────────────────────────────────────────────
    void beginFrame(VkCommandBuffer cmd, uint32_t frameIdx);
    void beginPass (VkCommandBuffer cmd, uint32_t frameIdx, std::string_view name);
    void endPass   (VkCommandBuffer cmd, uint32_t frameIdx);

    // ── CPU-side readback (call after vkWaitForFences) ────────────────────────
    void readback(VkDevice device, uint32_t prevFrameIdx,
                  uint32_t drawCalls, uint64_t triangles,
                  VmaAllocator vma, float cpuFrameMs);

    // ── ImGui overlay (no-op when VKGFX_ENABLE_PROFILING not defined) ─────────
    void renderOverlay() const;

    [[nodiscard]] const FrameStats& stats() const { return m_stats; }

private:
    [[nodiscard]] uint32_t queryBegin(uint32_t fi, uint32_t passIdx) const {
        return (fi * MAX_PASSES + passIdx) * 2;
    }

    VkQueryPool m_queryPool      = VK_NULL_HANDLE;
    float       m_tsNs           = 1.f;
    uint32_t    m_framesInFlight = 2;

    struct FrameState {
        uint32_t passCount = 0;
        std::array<std::string, MAX_PASSES> names{};
    };
    std::vector<FrameState> m_frameState;

    FrameStats m_stats;

    PFN_vkCmdBeginDebugUtilsLabelEXT m_fnBeginLabel = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT   m_fnEndLabel   = nullptr;
};

} // namespace vkgfx
