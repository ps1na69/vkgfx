#pragma once
// shadow.h — Cascaded Shadow Maps (CSM) for directional lights.
//
// Splits the camera frustum into NUM_CASCADES sub-frustums and renders
// a depth map for each. The lighting pass reads all cascades and selects
// the tightest-fitting one for each pixel using the view-space depth.
//
// Cascade selection uses a smooth blend zone to hide seams.

#include "context.h"
#include "camera.h"
#include <array>

namespace vkgfx {

inline constexpr uint32_t NUM_CASCADES   = 4;
inline constexpr uint32_t SHADOW_MAP_SIZE = 2048;

// Full shadow UBO — uploaded once per frame.
// Layout MUST match the GLSL ShadowUBO in both shadow_depth.vert and lighting.frag:
//   mat4  lightSpaceMatrix[4];   // all 4 cascade light-space matrices (256 bytes)
//   float splitDepths[4];        // view-space split depths for cascade selection (16 bytes)
struct alignas(16) ShadowUBO {
    glm::mat4 lightSpaceMatrix[NUM_CASCADES];  // 4 × 64 = 256 bytes
    float     splitDepths[NUM_CASCADES];        // 4 × 4  =  16 bytes
    // Total: 272 bytes — matches the GLSL std140 layout in lighting.frag exactly.
    // No padding needed: splitDepths[4] already ends on a 16-byte boundary.
};

// Per-cascade depth attachment.
struct ShadowCascade {
    AllocatedImage depthImage;    // SHADOW_MAP_SIZE × SHADOW_MAP_SIZE D32_SFLOAT
    VkFramebuffer  framebuffer = VK_NULL_HANDLE;
    VkRenderPass   renderPass  = VK_NULL_HANDLE;
};

class ShadowSystem {
public:
    ShadowSystem() = default;
    ~ShadowSystem() { destroy(); }

    ShadowSystem(const ShadowSystem&)            = delete;
    ShadowSystem& operator=(const ShadowSystem&) = delete;

    void init(std::shared_ptr<Context> ctx);
    void destroy();

    // Update cascade splits & light matrices each frame.
    // lightDir: normalized direction the light points (world space, towards scene).
    void update(const Camera& camera, glm::vec3 lightDir, float lambda = 0.75f);

    // Create depth-only pipeline layout + pipeline for geometry pass.
    void createPipeline(const std::filesystem::path& shaderDir,
                        VkDescriptorSetLayout sceneLayout);

    // Render all 4 cascades by calling drawFn for each (which should call
    // vkCmdDrawIndexed for all shadow-casting meshes).
    using DrawFn = std::function<void(VkCommandBuffer, uint32_t cascade)>;
    void renderCascades(VkCommandBuffer cmd, const DrawFn& drawFn);

    // Accessors for the lighting pass.
    [[nodiscard]] VkImageView  shadowArrayView() const { return m_shadowArrayView; }
    [[nodiscard]] VkSampler    shadowSampler()   const { return m_shadowSampler; }
    [[nodiscard]] VkImage      shadowArrayImage() const { return m_shadowArray.image; }
    [[nodiscard]] const ShadowUBO& shadowUBO()   const { return m_ubo; }
    [[nodiscard]] VkPipeline   pipeline()        const { return m_pipeline; }
    [[nodiscard]] VkPipelineLayout pipeLayout()  const { return m_pipeLayout; }

private:
    void createRenderPasses();
    void createFramebuffers();
    void createShadowArrayView();
    void createSampler();
    std::array<glm::vec3, 8> frustumCorners(const glm::mat4& proj,
                                              const glm::mat4& view) const;

    std::shared_ptr<Context>              m_ctx;
    std::array<ShadowCascade, NUM_CASCADES> m_cascades{};

    // Array image view over all cascade depth images for the lighting pass.
    AllocatedImage m_shadowArray;   // 2048×2048 array of 4 layers, D32_SFLOAT
    VkImageView    m_shadowArrayView = VK_NULL_HANDLE;
    VkSampler      m_shadowSampler   = VK_NULL_HANDLE;

    VkPipeline       m_pipeline   = VK_NULL_HANDLE;
    VkPipelineLayout m_pipeLayout = VK_NULL_HANDLE;

    ShadowUBO m_ubo{};
    float     m_nearClip = 0.1f;
    float     m_farClip  = 1000.f;
};

} // namespace vkgfx
