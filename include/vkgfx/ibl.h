#pragma once
// include/vkgfx/ibl.h
// Image-Based Lighting system.
// Loads an equirectangular HDR, converts to cubemap, generates:
//   - Environment cube (VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)
//   - Irradiance cube (diffuse IBL)
//   - Prefiltered env cube (specular IBL, 5 roughness mips)
//   - BRDF integration LUT (512x512 RG16F)
// All sizes come from IBLConfig. No hardcoded constants here.

#include "vk_raii.h"
#include "config.h"
#include <vulkan/vulkan.h>
#include <string>
#include <vector>

namespace vkgfx {

class Context;

class IBLSystem {
public:
    explicit IBLSystem(Context& ctx);
    ~IBLSystem();

    IBLSystem(const IBLSystem&) = delete;
    IBLSystem& operator=(const IBLSystem&) = delete;

    /// Set directory containing compiled shader .spv files. Call before build().
    void setShaderDir(const std::string& dir) { m_shaderDir = dir; }

    /// Load HDR and bake all cube maps.
    /// Returns false and logs error if hdrPath does not exist or bake fails.
    bool build(const IBLConfig& cfg);

    /// Release all GPU resources. Safe to call multiple times.
    void destroy();

    /// True after successful build()
    [[nodiscard]] bool isReady() const { return m_ready; }

    // ── Accessors for descriptor writes ──────────────────────────────────────
    [[nodiscard]] VkImageView irradianceView()  const { return m_irradiance.view; }
    [[nodiscard]] VkImageView prefilteredView() const { return m_prefiltered.view; }
    [[nodiscard]] VkImageView brdfLutView()     const { return m_brdfLut.view; }
    [[nodiscard]] VkSampler   cubeSampler()     const { return m_cubeSampler; }
    [[nodiscard]] VkSampler   brdfSampler()     const { return m_brdfSampler; }
    [[nodiscard]] float       intensity()       const { return m_intensity; }

private:
    bool loadEquirectangular(const std::string& path);
    void buildEnvCube(uint32_t size);
    void buildIrradiance(uint32_t size);
    void buildPrefiltered(uint32_t envSize);
    void buildBrdfLut();
    void createSamplers();

    void runCubeCompute(const std::string& shaderName,
                        const AllocatedImage& srcCube,
                        AllocatedImage& dstCube,
                        uint32_t dstSize, uint32_t dstMips);

    static void transitionImage(VkCommandBuffer cmd, VkImage image,
        VkImageLayout oldLayout, VkImageLayout newLayout,
        VkAccessFlags srcAccess, VkAccessFlags dstAccess,
        VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
        uint32_t mipLevels, uint32_t layers);

    Context&       m_ctx;
    std::string    m_shaderDir;

    AllocatedImage m_equirect{};
    AllocatedImage m_envCube{};
    AllocatedImage m_irradiance{};
    AllocatedImage m_prefiltered{};
    AllocatedImage m_brdfLut{};

    VkSampler      m_cubeSampler = VK_NULL_HANDLE;
    VkSampler      m_brdfSampler = VK_NULL_HANDLE;
    float          m_intensity   = 1.0f;
    bool           m_ready       = false;

    std::vector<VkImageView>      m_bakeCleanupViews;
    std::vector<VkDescriptorPool> m_bakeCleanupPools;
};

} // namespace vkgfx
