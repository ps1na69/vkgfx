#pragma once
// SSAOPass.h
// Screen-Space Ambient Occlusion (SSAO) pass.
//
// Algorithm overview:
//   For each pixel, sample SAMPLE_COUNT points in a hemisphere oriented along
//   the surface normal (read from the G-buffer normal attachment).  For each
//   sample, test whether it is occluded by comparing its depth against the
//   depth buffer.  Average the occlusion results and write a [0,1] occlusion
//   factor to a single-channel R8 image.  A separate blur pass smears the
//   result to remove temporal noise.
//
// The occlusion factor is multiplied into the ambient term inside the
// lighting pass.

#include <vulkan/vulkan.h>
#include <array>
#include <glm/glm.hpp>

namespace vkgfx {

struct GBuffer;  // forward-declared to avoid a circular header dependency

// Number of hemisphere samples per pixel.  More = higher quality, higher cost.
// 16–32 is a good range for real-time; blur compensates for noise at low counts.
static constexpr uint32_t SSAO_SAMPLE_COUNT = 32;
// Radius of the sampling hemisphere in world-space units.
static constexpr float    SSAO_RADIUS       = 0.5f;
// Bias prevents self-shadowing artefacts on flat surfaces.
static constexpr float    SSAO_BIAS         = 0.025f;

// Matches the std140 struct in ssao.frag.
struct alignas(16) SSAOUniforms {
    glm::mat4  proj;
    glm::mat4  invProj;
    glm::vec4  samples[SSAO_SAMPLE_COUNT];  // xyz = hemisphere offsets, w = unused
    glm::vec2  noiseScale;                  // screen_size / noise_tile_size
    float      radius;
    float      bias;
};

struct SSAOPassCreateInfo {
    VkDevice           device;
    VkPhysicalDevice   physicalDevice;
    uint32_t           width;
    uint32_t           height;
    uint32_t           framesInFlight;
    VkDescriptorPool   descriptorPool;
    VkPipelineCache    pipelineCache;
    const GBuffer*     gbuffer;             // Read normal + depth
};

class SSAOPass {
public:
    SSAOPass() = default;
    void create(const SSAOPassCreateInfo& ci);
    void destroy(VkDevice device);
    void on_resize(VkDevice device, VkPhysicalDevice physDev,
                   uint32_t w, uint32_t h, const GBuffer* gbuffer);

    // Record SSAO + blur commands into cb.
    // Updates the uniform buffer for frameIndex with fresh projection matrices.
    void record(VkCommandBuffer cb, uint32_t frameIndex,
                const glm::mat4& proj, const glm::mat4& invProj);

    // The blurred AO result; bound as a sampler in the lighting pass.
    VkImageView get_ao_view()  const { return m_blurView; }
    VkSampler   get_ao_sampler() const { return m_sampler; }

private:
    void generate_noise_texture(VkDevice device, VkPhysicalDevice physDev);
    void generate_kernel();
    void create_ao_images(VkDevice device, VkPhysicalDevice physDev,
                          uint32_t w, uint32_t h);
    void create_pipelines(VkDevice device, VkPipelineCache cache);
    void create_descriptor_sets(VkDevice device,
                                VkDescriptorPool pool,
                                const GBuffer* gbuffer);

    // Raw AO output (noisy)
    VkImage        m_rawImage   = VK_NULL_HANDLE;
    VkImageView    m_rawView    = VK_NULL_HANDLE;
    VkDeviceMemory m_rawMem     = VK_NULL_HANDLE;
    VkRenderPass   m_rawRP      = VK_NULL_HANDLE;
    VkFramebuffer  m_rawFB      = VK_NULL_HANDLE;

    // Blurred output
    VkImage        m_blurImage  = VK_NULL_HANDLE;
    VkImageView    m_blurView   = VK_NULL_HANDLE;
    VkDeviceMemory m_blurMem    = VK_NULL_HANDLE;
    VkRenderPass   m_blurRP     = VK_NULL_HANDLE;
    VkFramebuffer  m_blurFB     = VK_NULL_HANDLE;

    // 4x4 random rotation noise texture (tiled across screen)
    VkImage        m_noiseImage = VK_NULL_HANDLE;
    VkImageView    m_noiseView  = VK_NULL_HANDLE;
    VkDeviceMemory m_noiseMem   = VK_NULL_HANDLE;
    VkSampler      m_noiseSampler = VK_NULL_HANDLE;

    VkSampler      m_sampler    = VK_NULL_HANDLE;  // For reading AO in lighting pass

    // Pipelines
    VkPipelineLayout m_ssaoPipeLayout = VK_NULL_HANDLE;
    VkPipeline       m_ssaoPipeline   = VK_NULL_HANDLE;
    VkPipelineLayout m_blurPipeLayout = VK_NULL_HANDLE;
    VkPipeline       m_blurPipeline   = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_ssaoLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_blurLayout = VK_NULL_HANDLE;

    struct PerFrame {
        VkBuffer       ubo    = VK_NULL_HANDLE;
        VkDeviceMemory mem    = VK_NULL_HANDLE;
        void*          ptr    = nullptr;
        VkDescriptorSet ssaoSet = VK_NULL_HANDLE;
        VkDescriptorSet blurSet = VK_NULL_HANDLE;
    };
    std::vector<PerFrame> m_frames;

    // Pre-computed hemisphere kernel (sent to GPU once, not per-frame)
    std::array<glm::vec4, SSAO_SAMPLE_COUNT> m_kernel{};

    uint32_t m_width  = 0;
    uint32_t m_height = 0;
};

} // namespace vkgfx
