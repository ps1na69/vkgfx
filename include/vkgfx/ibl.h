#pragma once
// ibl.h — Image-Based Lighting probe.
//
// An IBLProbe wraps:
//   - irradianceMap   : diffuse convolution cubemap (32x32 per face)
//   - prefilteredMap  : specular prefiltered cubemap (128x128, 5 mip levels)
//   - brdfLUT         : 2D BRDF integration LUT (512x512)
//
// Usage:
//   IBLProbe probe;
//   probe.loadFromEquirectangular(ctx, "sky.hdr");
//
//   // In lighting.frag: sample irradiance + prefiltered + brdfLUT
//   // IBLProbe exposes VkImageView / VkSampler for each map.

#include "context.h"
#include "texture.h"
#include <filesystem>

namespace vkgfx {

class IBLProbe {
public:
    IBLProbe() = default;
    ~IBLProbe() { destroy(); }

    IBLProbe(const IBLProbe&)            = delete;
    IBLProbe& operator=(const IBLProbe&) = delete;

    // Load from an equirectangular HDR image (.hdr via stb_image internally).
    void loadFromEquirectangular(std::shared_ptr<Context> ctx,
                                  const std::filesystem::path& path);

    // Load from a directory of pre-baked KTX2 cubemaps (fastest at runtime).
    void loadPrecomputed(std::shared_ptr<Context> ctx,
                          const std::filesystem::path& irradiancePath,
                          const std::filesystem::path& prefilteredPath,
                          const std::filesystem::path& brdfLUTPath);

    void destroy();

    // ── Accessors for descriptor writes ──────────────────────────────────────
    [[nodiscard]] VkImageView irradianceView()   const { return m_irradiance.view; }
    [[nodiscard]] VkImageView prefilteredView()  const { return m_prefiltered.view; }
    [[nodiscard]] VkImageView brdfLUTView()      const { return m_brdfLUT.view; }
    [[nodiscard]] VkSampler   cubeSampler()      const { return m_cubeSampler; }
    [[nodiscard]] VkSampler   brdfSampler()      const { return m_brdfSampler; }

    [[nodiscard]] uint32_t    prefilteredMips()  const { return PREFILTERED_MIPS; }
    [[nodiscard]] bool        isReady()          const { return m_ready; }

    static constexpr uint32_t IRRADIANCE_SIZE  = 32;
    static constexpr uint32_t PREFILTERED_SIZE = 128;
    static constexpr uint32_t PREFILTERED_MIPS = 5;
    static constexpr uint32_t BRDF_LUT_SIZE    = 512;

private:
    void generateFromCubemap(VkImage srcCubemap);
    void createSamplers();

    AllocatedImage m_equirect;    // intermediate HDR equirectangular
    AllocatedImage m_envCubemap;  // 512x512 environment cubemap
    AllocatedImage m_irradiance;  // 32x32 diffuse irradiance cubemap
    AllocatedImage m_prefiltered; // 128x128 specular prefiltered cubemap
    AllocatedImage m_brdfLUT;     // 512x512 BRDF integration LUT

    VkSampler m_cubeSampler = VK_NULL_HANDLE;
    VkSampler m_brdfSampler = VK_NULL_HANDLE;

    std::shared_ptr<Context> m_ctx;
    bool m_ready = false;
};

} // namespace vkgfx
