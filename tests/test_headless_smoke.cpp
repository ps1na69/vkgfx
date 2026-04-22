// tests/test_headless_smoke.cpp
// Headless smoke test: exercises Context creation and GpuProfiler init/destroy
// without a window or swapchain.  Runs in CI on Ubuntu and Windows.
//
// Registered with CTest as "headless_smoke".

#include <catch2/catch_test_macros.hpp>
#include <vkgfx/context.h>
#include <vkgfx/profiler.h>
#include <vulkan/vulkan.h>

using namespace vkgfx;

// ── Context creation ──────────────────────────────────────────────────────────

TEST_CASE("Headless smoke — Context creates and destroys cleanly") {
    ContextConfig cc;
    cc.headless   = true;
    cc.validation = false;  // validation layers may not be present in all CI envs

    REQUIRE_NOTHROW([&]{ Context ctx(cc); }());

    Context ctx(cc);
    REQUIRE(ctx.isValid());
    REQUIRE(ctx.device()           != VK_NULL_HANDLE);
    REQUIRE(ctx.instance()         != VK_NULL_HANDLE);
    REQUIRE(ctx.graphicsPool()     != VK_NULL_HANDLE);
    // No swapchain → presentQ can equal graphicsQ (headless path)
    REQUIRE(ctx.queues().graphics  != UINT32_MAX);
}

// ── GpuProfiler lifecycle ─────────────────────────────────────────────────────

TEST_CASE("GpuProfiler init and destroy on headless context") {
    ContextConfig cc;
    cc.headless   = true;
    cc.validation = false;

    Context ctx(cc);
    REQUIRE(ctx.isValid());

    GpuProfiler profiler;
    REQUIRE_FALSE(profiler.ready());

    // init() should either succeed (device supports timestamps) or gracefully
    // skip (timestampPeriod == 0) — must never throw or crash either way.
    REQUIRE_NOTHROW(profiler.init(ctx, 2));

    // Calling destroy() on a profiler that may not have initialised must be safe
    REQUIRE_NOTHROW(profiler.destroy(ctx.device()));
    REQUIRE_FALSE(profiler.ready());
}

TEST_CASE("GpuProfiler double-destroy is safe") {
    ContextConfig cc;
    cc.headless = true;
    Context ctx(cc);

    GpuProfiler profiler;
    profiler.init(ctx, 2);
    profiler.destroy(ctx.device());
    // Second destroy must not crash (VK_NULL_HANDLE guard)
    REQUIRE_NOTHROW(profiler.destroy(ctx.device()));
}

// ── Query pool coherence ──────────────────────────────────────────────────────

TEST_CASE("GpuProfiler readback on empty frame does not crash") {
    ContextConfig cc;
    cc.headless = true;
    Context ctx(cc);

    GpuProfiler profiler;
    profiler.init(ctx, 2);

    if (!profiler.ready()) {
        WARN("Skipping readback test — device has no timestamp support");
        profiler.destroy(ctx.device());
        return;
    }

    // readback with no passes recorded — should produce all-zero stats cleanly
    REQUIRE_NOTHROW(profiler.readback(ctx.device(),
        /*prevFrameIdx=*/0,
        /*drawCalls=*/0,
        /*triangles=*/0,
        /*vma=*/ctx.vma(),
        /*cpuFrameMs=*/1.0f));

    const FrameStats& s = profiler.stats();
    REQUIRE(s.passes.empty());
    REQUIRE(s.drawCalls == 0);
    REQUIRE(s.triangles == 0);
    // cpuFrameMs is set to the passed-in value; tolerate minor FP rounding
    REQUIRE(s.cpuFrameMs >= 0.9f);
    REQUIRE(s.cpuFrameMs <= 1.1f);

    profiler.destroy(ctx.device());
}

// ── AllocatedBuffer round-trip (existing but important baseline) ──────────────

TEST_CASE("AllocatedBuffer upload-destroy cycle is clean") {
    ContextConfig cc;
    cc.headless = true;
    Context ctx(cc);

    const uint32_t kData[] = {0xDEADBEEFu, 0xCAFEBABEu};
    AllocatedBuffer buf = ctx.uploadBuffer(kData, sizeof(kData),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    REQUIRE(buf.buffer     != VK_NULL_HANDLE);
    REQUIRE(buf.allocation != nullptr);

    ctx.destroyBuffer(buf);

    REQUIRE(buf.buffer     == VK_NULL_HANDLE);
    REQUIRE(buf.allocation == nullptr);
}
