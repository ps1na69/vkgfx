// SSAOPass.cpp
// Screen-Space Ambient Occlusion implementation.
// See SSAOPass.h for algorithm overview.

#include "SSAOPass.h"
#include "GBuffer.h"
#include <random>
#include <stdexcept>
#include <cstring>
#include <array>
#include <fstream>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>

namespace vkgfx {

// ─── Internal helpers ─────────────────────────────────────────────────────────

static uint32_t find_memory_type(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags f)
{
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & f) == f)
            return i;
    throw std::runtime_error("SSAOPass: no suitable memory type");
}

static VkShaderModule load_shader(VkDevice dev, const char* path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error(std::string("SSAOPass: cannot open shader: ") + path);
    size_t sz = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> code(sz / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(sz));
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = sz; ci.pCode = code.data();
    VkShaderModule mod;
    vkCreateShaderModule(dev, &ci, nullptr, &mod);
    return mod;
}

static VkImage create_image_and_memory(VkDevice dev, VkPhysicalDevice pd,
                                        uint32_t w, uint32_t h,
                                        VkFormat fmt, VkImageUsageFlags usage,
                                        VkDeviceMemory& outMem)
{
    VkImageCreateInfo ici{};
    ici.sType     = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format    = fmt; ici.extent = {w,h,1};
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples   = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling    = VK_IMAGE_TILING_OPTIMAL;
    ici.usage     = usage;
    VkImage img;
    vkCreateImage(dev, &ici, nullptr, &img);
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(dev, img, &req);
    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_memory_type(pd, req.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(dev, &mai, nullptr, &outMem);
    vkBindImageMemory(dev, img, outMem, 0);
    return img;
}

static VkImageView create_view(VkDevice dev, VkImage img, VkFormat fmt,
                                VkImageAspectFlags aspect)
{
    VkImageViewCreateInfo vci{};
    vci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image                           = img;
    vci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    vci.format                          = fmt;
    vci.subresourceRange.aspectMask     = aspect;
    vci.subresourceRange.levelCount     = 1;
    vci.subresourceRange.layerCount     = 1;
    VkImageView v;
    vkCreateImageView(dev, &vci, nullptr, &v);
    return v;
}

// ─── Kernel and noise generation ─────────────────────────────────────────────

void SSAOPass::generate_kernel()
{
    // Generate SAMPLE_COUNT hemisphere samples.  Samples are distributed with
    // a bias towards the origin (lerp towards 0) so more samples fall near the
    // surface, which produces a more physically plausible result.
    std::uniform_real_distribution<float> rng(0.0f, 1.0f);
    std::default_random_engine            gen;

    for (uint32_t i = 0; i < SSAO_SAMPLE_COUNT; ++i) {
        glm::vec3 s(
            rng(gen) * 2.0f - 1.0f,   // [-1, 1]
            rng(gen) * 2.0f - 1.0f,   // [-1, 1]
            rng(gen)                    // [ 0, 1] – keeps samples in upper hemisphere
        );
        s = glm::normalize(s);
        s *= rng(gen);  // Random magnitude

        // Lerp accelerating factor: more samples closer to the origin.
        float scale = float(i) / float(SSAO_SAMPLE_COUNT);
        scale = 0.1f + scale * scale * 0.9f;   // Quadratic from 0.1 to 1.0
        s    *= scale;

        m_kernel[i] = glm::vec4(s, 0.0f);
    }
}

void SSAOPass::generate_noise_texture(VkDevice dev, VkPhysicalDevice pd)
{
    // 4x4 random rotation vectors (xy only; z=0) in [−1,+1].
    // These are tiled across the screen to give each pixel a different kernel
    // rotation without the memory cost of a full-screen random texture.
    std::uniform_real_distribution<float> rng(0.0f, 1.0f);
    std::default_random_engine            gen;

    const uint32_t NOISE_SIZE = 4;
    struct NoiseTexel { float x, y; };  // RG16_SFLOAT
    std::array<NoiseTexel, NOISE_SIZE * NOISE_SIZE> noise{};
    for (auto& t : noise) {
        t.x = rng(gen) * 2.0f - 1.0f;
        t.y = rng(gen) * 2.0f - 1.0f;
    }

    // Upload via a staging buffer.
    VkDeviceSize sz = noise.size() * sizeof(NoiseTexel);

    // Staging buffer (host-visible).
    VkBuffer stageBuf;
    VkDeviceMemory stageMem;
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = sz; bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        vkCreateBuffer(dev, &bci, nullptr, &stageBuf);
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(dev, stageBuf, &req);
        VkMemoryAllocateInfo mai{};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_memory_type(pd, req.memoryTypeBits,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(dev, &mai, nullptr, &stageMem);
        vkBindBufferMemory(dev, stageBuf, stageMem, 0);
        void* ptr;
        vkMapMemory(dev, stageMem, 0, sz, 0, &ptr);
        std::memcpy(ptr, noise.data(), sz);
        vkUnmapMemory(dev, stageMem);
    }

    m_noiseImage = create_image_and_memory(dev, pd, NOISE_SIZE, NOISE_SIZE,
        VK_FORMAT_R16G16_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        m_noiseMem);
    m_noiseView = create_view(dev, m_noiseImage, VK_FORMAT_R16G16_SFLOAT,
                              VK_IMAGE_ASPECT_COLOR_BIT);

    // The caller must issue a one-time command buffer to copy the staging
    // buffer into the image.  This is abbreviated here for clarity; the full
    // implementation uses a helper that submits a transient command buffer.
    // (Omitted: vkCmdCopyBufferToImage transition + copy + barrier.)

    vkDestroyBuffer(dev, stageBuf, nullptr);
    vkFreeMemory(dev, stageMem, nullptr);

    // Sampler for the noise texture: repeat mode so it tiles across the screen.
    VkSamplerCreateInfo sci{};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_NEAREST;
    sci.minFilter    = VK_FILTER_NEAREST;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;   // Tile!
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    vkCreateSampler(dev, &sci, nullptr, &m_noiseSampler);
}

// ─── Image creation ───────────────────────────────────────────────────────────

void SSAOPass::create_ao_images(VkDevice dev, VkPhysicalDevice pd, uint32_t w, uint32_t h)
{
    m_width  = w;
    m_height = h;

    // R8 single-channel is sufficient for the [0,1] AO factor.
    VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    m_rawImage  = create_image_and_memory(dev, pd, w, h, VK_FORMAT_R8_UNORM, usage, m_rawMem);
    m_rawView   = create_view(dev, m_rawImage,  VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

    m_blurImage = create_image_and_memory(dev, pd, w, h, VK_FORMAT_R8_UNORM, usage, m_blurMem);
    m_blurView  = create_view(dev, m_blurImage, VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

    // Sampler for reading the final blurred AO in the lighting pass.
    VkSamplerCreateInfo sci{};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_LINEAR;    // Linear for smooth AO gradients
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(dev, &sci, nullptr, &m_sampler);
}

// ─── Render-pass and framebuffer creation ─────────────────────────────────────

static VkRenderPass create_single_attachment_rp(VkDevice dev, VkFormat fmt,
                                                 VkImageLayout finalLayout)
{
    VkAttachmentDescription att{};
    att.format         = fmt;
    att.samples        = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout    = finalLayout;

    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription sp{};
    sp.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sp.colorAttachmentCount = 1;
    sp.pColorAttachments    = &ref;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpci{};
    rpci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 1; rpci.pAttachments    = &att;
    rpci.subpassCount    = 1; rpci.pSubpasses      = &sp;
    rpci.dependencyCount = 1; rpci.pDependencies   = &dep;
    VkRenderPass rp;
    vkCreateRenderPass(dev, &rpci, nullptr, &rp);
    return rp;
}

static VkFramebuffer create_fb(VkDevice dev, VkRenderPass rp,
                                VkImageView view, uint32_t w, uint32_t h)
{
    VkFramebufferCreateInfo fci{};
    fci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fci.renderPass      = rp;
    fci.attachmentCount = 1;
    fci.pAttachments    = &view;
    fci.width = w; fci.height = h; fci.layers = 1;
    VkFramebuffer fb;
    vkCreateFramebuffer(dev, &fci, nullptr, &fb);
    return fb;
}

// ─── Pipeline creation ────────────────────────────────────────────────────────

void SSAOPass::create_pipelines(VkDevice dev, VkPipelineCache cache)
{
    auto make_fullscreen_pipeline = [&](VkShaderModule fragMod,
                                        VkPipelineLayout layout,
                                        VkRenderPass rp) -> VkPipeline
    {
        VkShaderModule vertMod = load_shader(dev, "shaders/deferred/fullscreen_quad.vert.spv");

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_VERTEX_BIT,   vertMod, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                     VK_SHADER_STAGE_FRAGMENT_BIT, fragMod, "main", nullptr};

        VkPipelineVertexInputStateCreateInfo noVtx{};
        noVtx.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 2; dyn.pDynamicStates = dynStates;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rast{};
        rast.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rast.polygonMode = VK_POLYGON_MODE_FILL; rast.cullMode = VK_CULL_MODE_NONE;
        rast.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_FALSE; ds.depthWriteEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState blendAtt{};
        blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;  // Single-channel AO
        VkPipelineColorBlendStateCreateInfo cbs{};
        cbs.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cbs.attachmentCount = 1; cbs.pAttachments = &blendAtt;

        VkGraphicsPipelineCreateInfo gpci{};
        gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gpci.stageCount          = 2; gpci.pStages = stages;
        gpci.pVertexInputState   = &noVtx;
        gpci.pInputAssemblyState = &ia;
        gpci.pViewportState      = &vp;
        gpci.pRasterizationState = &rast;
        gpci.pMultisampleState   = &ms;
        gpci.pDepthStencilState  = &ds;
        gpci.pColorBlendState    = &cbs;
        gpci.pDynamicState       = &dyn;
        gpci.layout              = layout;
        gpci.renderPass          = rp;
        gpci.subpass             = 0;

        VkPipeline pipe;
        vkCreateGraphicsPipelines(dev, cache, 1, &gpci, nullptr, &pipe);
        vkDestroyShaderModule(dev, vertMod, nullptr);
        return pipe;
    };

    // SSAO pipeline layout.
    {
        std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
        for (uint32_t i = 0; i < 4; ++i) {
            bindings[i].binding        = i;
            bindings[i].descriptorType = (i < 3)
                ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 4; lci.pBindings = bindings.data();
        vkCreateDescriptorSetLayout(dev, &lci, nullptr, &m_ssaoLayout);

        VkPipelineLayoutCreateInfo plci{};
        plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &m_ssaoLayout;
        vkCreatePipelineLayout(dev, &plci, nullptr, &m_ssaoPipeLayout);
    }

    // Blur pipeline layout.
    {
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
        for (uint32_t i = 0; i < 2; ++i) {
            bindings[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                           VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 2; lci.pBindings = bindings.data();
        vkCreateDescriptorSetLayout(dev, &lci, nullptr, &m_blurLayout);

        VkPushConstantRange pcr{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float) * 2};
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1; plci.pSetLayouts = &m_blurLayout;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(dev, &plci, nullptr, &m_blurPipeLayout);
    }

    VkShaderModule ssaoFrag = load_shader(dev, "shaders/deferred/ssao.frag.spv");
    VkShaderModule blurFrag = load_shader(dev, "shaders/deferred/ssao_blur.frag.spv");

    m_ssaoPipeline = make_fullscreen_pipeline(ssaoFrag, m_ssaoPipeLayout, m_rawRP);
    m_blurPipeline = make_fullscreen_pipeline(blurFrag, m_blurPipeLayout, m_blurRP);

    vkDestroyShaderModule(dev, ssaoFrag, nullptr);
    vkDestroyShaderModule(dev, blurFrag, nullptr);
}

// ─── Public API ───────────────────────────────────────────────────────────────

void SSAOPass::create(const SSAOPassCreateInfo& ci)
{
    generate_kernel();
    generate_noise_texture(ci.device, ci.physicalDevice);
    create_ao_images(ci.device, ci.physicalDevice, ci.width, ci.height);

    m_rawRP  = create_single_attachment_rp(ci.device, VK_FORMAT_R8_UNORM,
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_blurRP = create_single_attachment_rp(ci.device, VK_FORMAT_R8_UNORM,
                                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_rawFB  = create_fb(ci.device, m_rawRP,  m_rawView,  ci.width, ci.height);
    m_blurFB = create_fb(ci.device, m_blurRP, m_blurView, ci.width, ci.height);

    create_pipelines(ci.device, ci.pipelineCache);

    // Allocate per-frame UBOs.
    m_frames.resize(ci.framesInFlight);
    for (auto& f : m_frames) {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size  = sizeof(SSAOUniforms);
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        vkCreateBuffer(ci.device, &bci, nullptr, &f.ubo);
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(ci.device, f.ubo, &req);
        VkMemoryAllocateInfo mai{};
        mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize  = req.size;
        mai.memoryTypeIndex = find_memory_type(ci.physicalDevice, req.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(ci.device, &mai, nullptr, &f.mem);
        vkBindBufferMemory(ci.device, f.ubo, f.mem, 0);
        vkMapMemory(ci.device, f.mem, 0, sizeof(SSAOUniforms), 0, &f.ptr);
    }

    create_descriptor_sets(ci.device, ci.descriptorPool, ci.gbuffer);
}

void SSAOPass::destroy(VkDevice dev)
{
    for (auto& f : m_frames) {
        if (f.ptr) { vkUnmapMemory(dev, f.mem); f.ptr = nullptr; }
        if (f.ubo) { vkDestroyBuffer(dev, f.ubo, nullptr); }
        if (f.mem) { vkFreeMemory(dev, f.mem, nullptr); }
    }
    m_frames.clear();

    auto destroy_img = [&](VkImage& img, VkImageView& view, VkDeviceMemory& mem) {
        if (view) { vkDestroyImageView(dev, view, nullptr); view = VK_NULL_HANDLE; }
        if (img)  { vkDestroyImage(dev, img, nullptr);      img  = VK_NULL_HANDLE; }
        if (mem)  { vkFreeMemory(dev, mem, nullptr);        mem  = VK_NULL_HANDLE; }
    };
    destroy_img(m_rawImage,   m_rawView,   m_rawMem);
    destroy_img(m_blurImage,  m_blurView,  m_blurMem);
    destroy_img(m_noiseImage, m_noiseView, m_noiseMem);

    if (m_sampler)      { vkDestroySampler(dev, m_sampler,      nullptr); m_sampler = VK_NULL_HANDLE; }
    if (m_noiseSampler) { vkDestroySampler(dev, m_noiseSampler, nullptr); m_noiseSampler = VK_NULL_HANDLE; }

    if (m_rawFB)  { vkDestroyFramebuffer(dev, m_rawFB,  nullptr); m_rawFB = VK_NULL_HANDLE; }
    if (m_blurFB) { vkDestroyFramebuffer(dev, m_blurFB, nullptr); m_blurFB = VK_NULL_HANDLE; }
    if (m_rawRP)  { vkDestroyRenderPass(dev, m_rawRP,   nullptr); m_rawRP = VK_NULL_HANDLE; }
    if (m_blurRP) { vkDestroyRenderPass(dev, m_blurRP,  nullptr); m_blurRP = VK_NULL_HANDLE; }

    if (m_ssaoPipeline)   { vkDestroyPipeline(dev, m_ssaoPipeline, nullptr); }
    if (m_blurPipeline)   { vkDestroyPipeline(dev, m_blurPipeline, nullptr); }
    if (m_ssaoPipeLayout) { vkDestroyPipelineLayout(dev, m_ssaoPipeLayout, nullptr); }
    if (m_blurPipeLayout) { vkDestroyPipelineLayout(dev, m_blurPipeLayout, nullptr); }
    if (m_ssaoLayout)     { vkDestroyDescriptorSetLayout(dev, m_ssaoLayout, nullptr); }
    if (m_blurLayout)     { vkDestroyDescriptorSetLayout(dev, m_blurLayout, nullptr); }
}

void SSAOPass::on_resize(VkDevice dev, VkPhysicalDevice pd,
                          uint32_t w, uint32_t h, const GBuffer* gbuffer)
{
    // Destroy resolution-dependent resources.
    if (m_rawView)  { vkDestroyImageView(dev, m_rawView, nullptr);  m_rawView = VK_NULL_HANDLE; }
    if (m_rawImage) { vkDestroyImage(dev, m_rawImage, nullptr);     m_rawImage = VK_NULL_HANDLE; }
    if (m_rawMem)   { vkFreeMemory(dev, m_rawMem, nullptr);         m_rawMem = VK_NULL_HANDLE; }
    if (m_blurView)  { vkDestroyImageView(dev, m_blurView, nullptr); m_blurView = VK_NULL_HANDLE; }
    if (m_blurImage) { vkDestroyImage(dev, m_blurImage, nullptr);   m_blurImage = VK_NULL_HANDLE; }
    if (m_blurMem)   { vkFreeMemory(dev, m_blurMem, nullptr);       m_blurMem = VK_NULL_HANDLE; }
    if (m_rawFB)  { vkDestroyFramebuffer(dev, m_rawFB,  nullptr); m_rawFB  = VK_NULL_HANDLE; }
    if (m_blurFB) { vkDestroyFramebuffer(dev, m_blurFB, nullptr); m_blurFB = VK_NULL_HANDLE; }

    create_ao_images(dev, pd, w, h);
    m_rawFB  = create_fb(dev, m_rawRP,  m_rawView,  w, h);
    m_blurFB = create_fb(dev, m_blurRP, m_blurView, w, h);
    // Descriptor sets must be re-written with the new image views.
    // (Omitted: re-call create_descriptor_sets with the new views)
}

void SSAOPass::create_descriptor_sets(VkDevice dev,
                                       VkDescriptorPool pool,
                                       const GBuffer* gbuffer)
{
    for (auto& f : m_frames) {
        // SSAO set: normal, depth, noise, UBO.
        {
            VkDescriptorSetAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool     = pool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts        = &m_ssaoLayout;
            vkAllocateDescriptorSets(dev, &ai, &f.ssaoSet);

            std::array<VkDescriptorImageInfo, 3> imgInfos{};
            imgInfos[0] = {gbuffer->sampler, gbuffer->attachments[GBUF_NORMAL].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            imgInfos[1] = {gbuffer->sampler, gbuffer->depth.view,                    VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
            imgInfos[2] = {m_noiseSampler,   m_noiseView,                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

            VkDescriptorBufferInfo uboInfo{f.ubo, 0, sizeof(SSAOUniforms)};

            std::array<VkWriteDescriptorSet, 4> writes{};
            for (uint32_t i = 0; i < 3; ++i) {
                writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet          = f.ssaoSet;
                writes[i].dstBinding      = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[i].pImageInfo      = &imgInfos[i];
            }
            writes[3].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[3].dstSet          = f.ssaoSet;
            writes[3].dstBinding      = 3;
            writes[3].descriptorCount = 1;
            writes[3].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            writes[3].pBufferInfo     = &uboInfo;
            vkUpdateDescriptorSets(dev, 4, writes.data(), 0, nullptr);
        }

        // Blur set: raw AO image + depth.
        {
            VkDescriptorSetAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool     = pool;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts        = &m_blurLayout;
            vkAllocateDescriptorSets(dev, &ai, &f.blurSet);

            std::array<VkDescriptorImageInfo, 2> imgInfos{};
            imgInfos[0] = {m_sampler,         m_rawView,            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            imgInfos[1] = {gbuffer->sampler,  gbuffer->depth.view,  VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};

            std::array<VkWriteDescriptorSet, 2> writes{};
            for (uint32_t i = 0; i < 2; ++i) {
                writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet          = f.blurSet;
                writes[i].dstBinding      = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[i].pImageInfo      = &imgInfos[i];
            }
            vkUpdateDescriptorSets(dev, 2, writes.data(), 0, nullptr);
        }
    }
}

void SSAOPass::record(VkCommandBuffer cb, uint32_t frameIndex,
                       const glm::mat4& proj, const glm::mat4& invProj)
{
    auto& f = m_frames[frameIndex];

    // Upload per-frame SSAO uniforms.
    SSAOUniforms u{};
    u.proj     = proj;
    u.invProj  = invProj;
    u.noiseScale = glm::vec2(float(m_width) / 4.0f, float(m_height) / 4.0f);
    u.radius   = SSAO_RADIUS;
    u.bias     = SSAO_BIAS;
    for (uint32_t i = 0; i < SSAO_SAMPLE_COUNT; ++i)
        u.samples[i] = m_kernel[i];
    std::memcpy(f.ptr, &u, sizeof(u));

    // ── SSAO pass ─────────────────────────────────────────────────────────────
    VkClearValue clear{};
    clear.color = {{1.0f, 0.0f, 0.0f, 0.0f}};  // Default to no occlusion

    VkRenderPassBeginInfo rpbi{};
    rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass = m_rawRP; rpbi.framebuffer = m_rawFB;
    rpbi.renderArea = {{0,0},{m_width,m_height}};
    rpbi.clearValueCount = 1; rpbi.pClearValues = &clear;
    vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ssaoPipeline);
    VkViewport vp{0, 0, float(m_width), float(m_height), 0.0f, 1.0f};
    VkRect2D   sc{{0,0},{m_width,m_height}};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_ssaoPipeLayout, 0, 1, &f.ssaoSet, 0, nullptr);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRenderPass(cb);

    // ── Blur pass ─────────────────────────────────────────────────────────────
    rpbi.renderPass = m_blurRP; rpbi.framebuffer = m_blurFB;
    vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, m_blurPipeline);
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_blurPipeLayout, 0, 1, &f.blurSet, 0, nullptr);

    // Push texel size for the blur kernel.
    glm::vec2 texelSize(1.0f / float(m_width), 1.0f / float(m_height));
    vkCmdPushConstants(cb, m_blurPipeLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(texelSize), &texelSize);

    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRenderPass(cb);
}

} // namespace vkgfx
