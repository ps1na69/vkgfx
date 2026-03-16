// tests/test_ibl_load.cpp
// Validates that:
//   1. assets/sky.hdr exists (gives clear error if not)
//   2. IBLSystem::build() returns true
//   3. All output views are non-null after bake

#include <catch2/catch_test_macros.hpp>
#include <vkgfx/context.h>
#include <vkgfx/ibl.h>
#include <vkgfx/config.h>

#include <vulkan/vulkan.h>
#include <filesystem>
#include <iostream>

using namespace vkgfx;

// ── Fixture: headless context shared across IBL tests ─────────────────────────

static const std::string HDR_PATH  = "assets/sky.hdr";
static const std::string SHADER_DIR= "shaders";

TEST_CASE("sky.hdr asset exists in assets/ directory") {
    // This test deliberately fails with a clear message if the HDR is missing.
    bool exists = std::filesystem::exists(HDR_PATH);
    if (!exists) {
        std::cerr << "\n[test] MISSING: " << HDR_PATH << "\n"
                  << "       Download any equirectangular HDR (e.g. from polyhaven.com)\n"
                  << "       and save it as assets/sky.hdr to enable IBL tests.\n\n";
    }
    REQUIRE(exists);
}

TEST_CASE("IBLSystem builds all cube maps from sky.hdr") {
    if (!std::filesystem::exists(HDR_PATH)) {
        SKIP("assets/sky.hdr not present — skipping IBL build test");
    }

    // Use small sizes so the test is fast
    ContextConfig cc;
    cc.validation = true;
    cc.headless   = true;
    Context ctx(cc);
    REQUIRE(ctx.isValid());

    IBLSystem ibl(ctx);
    ibl.setShaderDir(SHADER_DIR);

    IBLConfig cfg;
    cfg.hdrPath        = HDR_PATH;
    cfg.envMapSize     = 64;   // small for speed
    cfg.irradianceSize = 8;
    cfg.intensity      = 1.0f;

    bool ok = ibl.build(cfg);
    REQUIRE(ok);
    REQUIRE(ibl.isReady());
    REQUIRE(ibl.irradianceView()  != VK_NULL_HANDLE);
    REQUIRE(ibl.prefilteredView() != VK_NULL_HANDLE);
    REQUIRE(ibl.brdfLutView()     != VK_NULL_HANDLE);
    REQUIRE(ibl.cubeSampler()     != VK_NULL_HANDLE);
    REQUIRE(ibl.brdfSampler()     != VK_NULL_HANDLE);
}

TEST_CASE("IBLSystem returns false for missing HDR path") {
    ContextConfig cc;
    cc.headless = true;
    Context ctx(cc);
    REQUIRE(ctx.isValid());

    IBLSystem ibl(ctx);
    ibl.setShaderDir(SHADER_DIR);

    IBLConfig cfg;
    cfg.hdrPath = "this/file/does/not/exist.hdr";
    cfg.envMapSize     = 64;
    cfg.irradianceSize = 8;

    bool ok = ibl.build(cfg);
    REQUIRE_FALSE(ok);
    REQUIRE_FALSE(ibl.isReady());
    REQUIRE(ibl.irradianceView()  == VK_NULL_HANDLE);
    REQUIRE(ibl.prefilteredView() == VK_NULL_HANDLE);
    REQUIRE(ibl.brdfLutView()     == VK_NULL_HANDLE);
}

TEST_CASE("IBLSystem can be rebuilt with different HDR config") {
    if (!std::filesystem::exists(HDR_PATH)) {
        SKIP("assets/sky.hdr not present — skipping IBL rebuild test");
    }

    ContextConfig cc;
    cc.headless = true;
    Context ctx(cc);

    IBLSystem ibl(ctx);
    ibl.setShaderDir(SHADER_DIR);

    IBLConfig cfg;
    cfg.hdrPath        = HDR_PATH;
    cfg.envMapSize     = 32;
    cfg.irradianceSize = 4;

    // First build
    REQUIRE(ibl.build(cfg));
    REQUIRE(ibl.isReady());

    // Rebuild with different size — should not crash or leak
    cfg.envMapSize = 64;
    REQUIRE(ibl.build(cfg));
    REQUIRE(ibl.isReady());
}
