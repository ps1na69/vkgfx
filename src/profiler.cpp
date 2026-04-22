// src/profiler.cpp
// GpuProfiler — timestamp queries, VRAM readback, ImGui overlay.
//
// The ImGui overlay section is only compiled when VKGFX_ENABLE_PROFILING is
// defined. All other methods (init, timestamps, readback) always compile and
// are silently inactive when the query pool could not be created.

#include <vkgfx/profiler.h>
#include <vkgfx/context.h>

#include <vk_mem_alloc.h>
#include <iostream>
#include <vector>
#include <cstring>

#ifdef VKGFX_ENABLE_PROFILING
#  include <imgui.h>
#endif

namespace vkgfx {

// ── init ─────────────────────────────────────────────────────────────────────

void GpuProfiler::init(Context& ctx, uint32_t framesInFlight) {
    m_framesInFlight = framesInFlight;
    m_frameState.resize(framesInFlight);

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(ctx.gpu(), &props);
    if (props.limits.timestampPeriod == 0.f) {
        std::cerr << "[vkgfx][WARN] Profiler: device does not support timestamps\n";
        return;
    }
    m_tsNs = props.limits.timestampPeriod;

    {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(ctx.gpu(), &count, nullptr);
        std::vector<VkQueueFamilyProperties> qfp(count);
        vkGetPhysicalDeviceQueueFamilyProperties(ctx.gpu(), &count, qfp.data());
        uint32_t gfx = ctx.queues().graphics;
        if (gfx < count && qfp[gfx].timestampValidBits == 0) {
            std::cerr << "[vkgfx][WARN] Profiler: graphics queue has no timestamp bits\n";
            return;
        }
    }

    uint32_t totalQueries = 2u * MAX_PASSES * framesInFlight;
    VkQueryPoolCreateInfo qci{};
    qci.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    qci.queryCount = totalQueries;
    if (vkCreateQueryPool(ctx.device(), &qci, nullptr, &m_queryPool) != VK_SUCCESS) {
        std::cerr << "[vkgfx][WARN] Profiler: vkCreateQueryPool failed\n";
        return;
    }
    vkResetQueryPool(ctx.device(), m_queryPool, 0, totalQueries);

    m_fnBeginLabel = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
        vkGetInstanceProcAddr(ctx.instance(), "vkCmdBeginDebugUtilsLabelEXT"));
    m_fnEndLabel = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
        vkGetInstanceProcAddr(ctx.instance(), "vkCmdEndDebugUtilsLabelEXT"));

    std::cout << "[vkgfx][INFO] Profiler: ready ("
              << totalQueries << " queries, " << m_tsNs << " ns/tick)\n";
}

// ── destroy ───────────────────────────────────────────────────────────────────

void GpuProfiler::destroy(VkDevice device) {
    if (m_queryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device, m_queryPool, nullptr);
        m_queryPool = VK_NULL_HANDLE;
    }
}

// ── beginFrame ────────────────────────────────────────────────────────────────

void GpuProfiler::beginFrame(VkCommandBuffer cmd, uint32_t frameIdx) {
    if (!ready()) return;
    uint32_t fi = frameIdx % m_framesInFlight;
    m_frameState[fi].passCount = 0;
    vkCmdResetQueryPool(cmd, m_queryPool, queryBegin(fi, 0), 2u * MAX_PASSES);
}

// ── beginPass / endPass ───────────────────────────────────────────────────────

void GpuProfiler::beginPass(VkCommandBuffer cmd, uint32_t frameIdx,
                             std::string_view name) {
    if (!ready()) return;
    uint32_t fi = frameIdx % m_framesInFlight;
    auto& fs = m_frameState[fi];
    if (fs.passCount >= MAX_PASSES) return;
    fs.names[fs.passCount] = std::string(name);

    if (m_fnBeginLabel) {
        static constexpr float kCol[4] = {0.2f, 0.6f, 1.0f, 1.0f};
        VkDebugUtilsLabelEXT lbl{};
        lbl.sType      = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
        lbl.pLabelName = fs.names[fs.passCount].c_str();
        std::copy(kCol, kCol + 4, lbl.color);
        m_fnBeginLabel(cmd, &lbl);
    }
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         m_queryPool, queryBegin(fi, fs.passCount));
}

void GpuProfiler::endPass(VkCommandBuffer cmd, uint32_t frameIdx) {
    if (!ready()) return;
    uint32_t fi = frameIdx % m_framesInFlight;
    auto& fs = m_frameState[fi];
    if (fs.passCount >= MAX_PASSES) return;
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         m_queryPool, queryBegin(fi, fs.passCount) + 1u);
    if (m_fnEndLabel) m_fnEndLabel(cmd);
    ++fs.passCount;
}

// ── readback ─────────────────────────────────────────────────────────────────

void GpuProfiler::readback(VkDevice device, uint32_t prevFrameIdx,
                             uint32_t drawCalls, uint64_t triangles,
                             VmaAllocator vma, float cpuFrameMs) {
    if (!ready()) return;
    uint32_t fi     = prevFrameIdx % m_framesInFlight;
    uint32_t nPasses = m_frameState[fi].passCount;

    m_stats.cpuFrameMs = cpuFrameMs;
    m_stats.drawCalls  = drawCalls;
    m_stats.triangles  = triangles;

    if (nPasses > 0) {
        std::vector<uint64_t> ticks(nPasses * 2u, 0ull);
        VkResult res = vkGetQueryPoolResults(
            device, m_queryPool, queryBegin(fi, 0), nPasses * 2u,
            nPasses * 2u * sizeof(uint64_t), ticks.data(),
            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

        if (res == VK_SUCCESS || res == VK_NOT_READY) {
            m_stats.passes.clear();
            m_stats.passes.reserve(nPasses);
            float total = 0.f;
            for (uint32_t p = 0; p < nPasses; ++p) {
                uint64_t t0 = ticks[p * 2u], t1 = ticks[p * 2u + 1u];
                float ms = (t1 >= t0)
                    ? static_cast<float>(static_cast<double>(t1 - t0) * m_tsNs / 1e6)
                    : 0.f;
                m_stats.passes.push_back({ m_frameState[fi].names[p], ms });
                total += ms;
            }
            m_stats.totalGpuMs = total;
        }
    }

    if (vma) {
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
        vmaGetHeapBudgets(vma, budgets);
        VkDeviceSize used = 0, budget = 0;
        for (uint32_t h = 0; h < VK_MAX_MEMORY_HEAPS; ++h) {
            if (!budgets[h].budget) continue;
            used   += budgets[h].usage;
            budget += budgets[h].budget;
        }
        constexpr float kMB = 1.f / (1024.f * 1024.f);
        m_stats.vramUsedMB   = static_cast<float>(used)   * kMB;
        m_stats.vramBudgetMB = static_cast<float>(budget) * kMB;
    }
}

// ── renderOverlay ─────────────────────────────────────────────────────────────

void GpuProfiler::renderOverlay() const {
#ifdef VKGFX_ENABLE_PROFILING
    if (!ImGui::GetCurrentContext()) return;

    const ImGuiWindowFlags kFlags =
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMouseInputs |
        ImGuiWindowFlags_AlwaysAutoResize;

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        ImVec2(vp->WorkPos.x + vp->WorkSize.x - 12.f,
               vp->WorkPos.y + 12.f),
        ImGuiCond_Always, ImVec2(1.f, 0.f));
    ImGui::SetNextWindowBgAlpha(0.75f);

    if (!ImGui::Begin("##vkgfx_profiler", nullptr, kFlags)) { ImGui::End(); return; }

    ImGui::TextColored(ImVec4(0.4f, 0.85f, 1.0f, 1.0f), "GPU Profiler");
    ImGui::Separator();

    constexpr ImGuiTableFlags kTbl =
        ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg |
        ImGuiTableFlags_SizingFixedFit;

    if (ImGui::BeginTable("##passes", 2, kTbl)) {
        ImGui::TableSetupColumn("Pass",     ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("GPU (ms)", ImGuiTableColumnFlags_WidthFixed, 72.f);
        ImGui::TableHeadersRow();

        for (const auto& pt : m_stats.passes) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(pt.name.c_str());
            ImGui::TableSetColumnIndex(1);
            ImVec4 col = (pt.gpuMs < 0.5f)
                ? ImVec4(0.45f, 0.90f, 0.45f, 1.0f)
                : (pt.gpuMs < 2.0f)
                ? ImVec4(1.0f, 0.85f, 0.2f, 1.0f)
                : ImVec4(1.0f, 0.35f, 0.35f, 1.0f);
            ImGui::TextColored(col, "%.3f", static_cast<double>(pt.gpuMs));
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextColored(ImVec4(1,1,1,0.7f), "Total GPU");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.3f", static_cast<double>(m_stats.totalGpuMs));
        ImGui::EndTable();
    }

    ImGui::Separator();
    ImGui::Text("CPU  : %.2f ms (%.0f FPS)",
        static_cast<double>(m_stats.cpuFrameMs),
        static_cast<double>(m_stats.cpuFrameMs > 0.f ? 1000.f / m_stats.cpuFrameMs : 0.f));
    ImGui::Text("Draws: %u   Tris: %llu",
        m_stats.drawCalls,
        static_cast<unsigned long long>(m_stats.triangles));

    if (m_stats.vramBudgetMB > 0.f) {
        float frac = m_stats.vramUsedMB / m_stats.vramBudgetMB;
        ImVec4 barCol = (frac < 0.70f)
            ? ImVec4(0.45f, 0.90f, 0.45f, 1.0f)
            : (frac < 0.90f)
            ? ImVec4(1.0f, 0.85f, 0.2f, 1.0f)
            : ImVec4(1.0f, 0.35f, 0.35f, 1.0f);
        ImGui::Text("VRAM ");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, barCol);
        ImGui::ProgressBar(frac, ImVec2(-1.f, 0.f));
        ImGui::PopStyleColor();
        ImGui::Text("  %.0f / %.0f MB",
            static_cast<double>(m_stats.vramUsedMB),
            static_cast<double>(m_stats.vramBudgetMB));
    }

    ImGui::End();
#endif
}

} // namespace vkgfx
