#pragma once
// include/vkgfx/config.h
// Public runtime configuration for vkgfx.
// All toggles live here.  Pass a RendererConfig to Renderer at construction.
// Call Renderer::applyConfig() to update hot-reload fields at runtime.

#include <string>
#include <cstdint>

namespace vkgfx {

/// MSAA sample count
enum class MSAASamples : uint8_t { x1 = 1, x2 = 2, x4 = 4, x8 = 8 };

/// G-buffer channel to visualise (None = full lighting)
enum class GBufferDebugView : uint8_t {
    None      = 0,
    Albedo    = 1,
    Normal    = 2,
    Roughness = 3,
    Metallic  = 4,
    Depth     = 5,
    AO        = 6,
};

/// IBL settings
struct IBLConfig {
    bool        enabled     = true;          ///< Toggle IBL contribution
    std::string hdrPath     = "";            ///< Path to equirectangular HDR (e.g. assets/sky.hdr)
    float       intensity   = 1.0f;          ///< IBL intensity multiplier
    uint32_t    envMapSize  = 512;           ///< Prefiltered env cube face size
    uint32_t    irradianceSize = 32;         ///< Irradiance cube face size
};

/// Sun / directional light settings
struct SunConfig {
    bool    enabled     = true;
    float   direction[3]= {-0.4f, -1.0f, -0.3f};
    float   color[3]    = {1.0f, 0.98f, 0.95f};
    float   intensity   = 5.0f;
};

/// SSAO settings
struct SSAOConfig {
    bool     enabled    = false;
    uint32_t kernelSize = 32;
    float    radius     = 0.5f;
    float    bias       = 0.025f;
};

/// Top-level renderer configuration
struct RendererConfig {
    // Window / device
    MSAASamples msaa        = MSAASamples::x4;
    bool        vsync       = true;

    // Shader / asset directories (resolved relative to exe if relative)
    std::string shaderDir   = "shaders";
    std::string assetDir    = "assets";

    // Passes
    IBLConfig   ibl;
    SunConfig   sun;
    SSAOConfig  ssao;

    // Debug
    GBufferDebugView gbufferDebug = GBufferDebugView::None;
    bool             validationLayers = false;  // set via VKGFX_ENABLE_VALIDATION in Debug

    // Factory helpers
    /// Load from a JSON file. Returns default config + file overrides.
    static RendererConfig fromFile(const std::string& path);
    /// Save current config to JSON file.
    void save(const std::string& path) const;
};

} // namespace vkgfx
