// DeferredRenderer.cpp
// Implements the full deferred shading pipeline.  See DeferredRenderer.h for
// an explanation of the multi-pass architecture.
//
// Key Vulkan specifics:
//   - The geometry render pass uses a single VkRenderPass with 6 subpass outputs
//     (5 color + depth).  All attachments are cleared at load.
//   - The lighting and tone-map passes each run in a single-subpass render pass
//     whose output attachment is either the HDR image or the swapchain image.
//   - Explicit pipeline barriers guarantee the geometry pass has finished
//     writing before the lighting pass reads the G-buffer (correct image
//     layout transitions are declared in the attachment descriptions so Vulkan
//     inserts the necessary barriers automatically at subpass boundaries).
//   - The light SSBO is updated via a persistently mapped pointer with
//     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | HOST_COHERENT_BIT so no explicit
//     flush is needed.

#include "DeferredRenderer.h"
#include <stdexcept>
#include <array>
#include <vector>
#include <fstream>
#include <cassert>
#include <cstring>
#include <cmath>

namespace vkgfx {

// ─── Utility: load a SPIR-V file and create a VkShaderModule ────────────────

static VkShaderModule load_shader(VkDevice device, const char* path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error(std::string("Cannot open shader: ") + path);

    size_t size = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> code(size / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size));

    VkShaderModuleCreateInfo smci{};
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = size;
    smci.pCode    = code.data();

    VkShaderModule mod;
    if (vkCreateShaderModule(device, &smci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error(std::string("Cannot create shader module: ") + path);
    return mod;
}

// ─── Utility: allocate a buffer with the given usage and memory properties ──

struct Buffer {
    VkBuffer       handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void*          mapped = nullptr;
};

static uint32_t find_memory_type(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags flags)
{
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    throw std::runtime_error("No suitable memory type");
}

static Buffer create_buffer(VkDevice dev, VkPhysicalDevice pd,
                             VkDeviceSize size,
                             VkBufferUsageFlags usage,
                             VkMemoryPropertyFlags props)
{
    Buffer buf;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size  = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(dev, &bci, nullptr, &buf.handle);

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, buf.handle, &req);

    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_memory_type(pd, req.memoryTypeBits, props);
    vkAllocateMemory(dev, &mai, nullptr, &buf.memory);
    vkBindBufferMemory(dev, buf.handle, buf.memory, 0);

    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        vkMapMemory(dev, buf.memory, 0, size, 0, &buf.mapped);

    return buf;
}

// ─── Constructor ─────────────────────────────────────────────────────────────

DeferredRenderer::DeferredRenderer(const DeferredRendererCreateInfo& ci)
    : m_device(ci.device),
      m_physDevice(ci.physicalDevice),
      m_descriptorPool(ci.descriptorPool),
      m_commandPool(ci.commandPool),
      m_graphicsQueue(ci.graphicsQueue),
      m_pipelineCache(ci.pipelineCache),
      m_swapchainRP(ci.swapchainRenderPass),
      m_swapchainFormat(ci.swapchainFormat),
      m_width(ci.width),
      m_height(ci.height),
      m_framesInFlight(ci.framesInFlight)
{
    m_frames.resize(m_framesInFlight);

    // Order matters: each step depends on the previous one.
    gbuffer_create(m_gbuffer, m_device, m_physDevice, m_width, m_height);
    create_hdr_image();
    create_descriptor_layouts();
    create_geometry_renderpass();
    create_geometry_framebuffer();
    create_geometry_pipeline();
    create_lighting_pipeline();
    create_tonemap_pipeline();
    create_frame_uniforms_buffers();
    create_light_ssbo();
    create_descriptor_sets();

    // SSAO setup.
    SSAOPassCreateInfo ssaoci{};
    ssaoci.device          = m_device;
    ssaoci.physicalDevice  = m_physDevice;
    ssaoci.width           = m_width;
    ssaoci.height          = m_height;
    ssaoci.framesInFlight  = m_framesInFlight;
    ssaoci.descriptorPool  = m_descriptorPool;
    ssaoci.pipelineCache   = m_pipelineCache;
    ssaoci.gbuffer         = &m_gbuffer;
    m_ssao.create(ssaoci);

    // Per-frame deferred command buffers (SSAO + lighting + tonemap).
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool        = m_commandPool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = m_framesInFlight;
    std::vector<VkCommandBuffer> cbs(m_framesInFlight);
    vkAllocateCommandBuffers(m_device, &cbai, cbs.data());
    for (uint32_t i = 0; i < m_framesInFlight; ++i)
        m_frames[i].deferredCB = cbs[i];

    // Geometry pass command buffers (one per frame-in-flight).
    cbai.commandBufferCount = m_framesInFlight;
    std::vector<VkCommandBuffer> geomCBs(m_framesInFlight);
    vkAllocateCommandBuffers(m_device, &cbai, geomCBs.data());
    m_geomCmdBuffers = geomCBs;
}

DeferredRenderer::~DeferredRenderer()
{
    // Wait for all GPU work to finish before releasing resources.
    vkDeviceWaitIdle(m_device);

    m_ssao.destroy(m_device);
    destroy_swapchain_dependent();

    for (auto& f : m_frames) {
        if (f.frameUBO)    { vkUnmapMemory(m_device, f.frameUBOMem); vkDestroyBuffer(m_device, f.frameUBO, nullptr); vkFreeMemory(m_device, f.frameUBOMem, nullptr); }
        if (f.lightSSBO)   { vkUnmapMemory(m_device, f.lightSSBOMem); vkDestroyBuffer(m_device, f.lightSSBO, nullptr); vkFreeMemory(m_device, f.lightSSBOMem, nullptr); }
    }

    if (m_frameSetLayout)   vkDestroyDescriptorSetLayout(m_device, m_frameSetLayout, nullptr);
    if (m_gbufferSetLayout) vkDestroyDescriptorSetLayout(m_device, m_gbufferSetLayout, nullptr);
    if (m_lightSetLayout)   vkDestroyDescriptorSetLayout(m_device, m_lightSetLayout, nullptr);
    if (m_hdrSetLayout)     vkDestroyDescriptorSetLayout(m_device, m_hdrSetLayout, nullptr);
}

// ─── Resize ──────────────────────────────────────────────────────────────────

void DeferredRenderer::on_resize(uint32_t width, uint32_t height)
{
    vkDeviceWaitIdle(m_device);
    m_width  = width;
    m_height = height;
    destroy_swapchain_dependent();
    rebuild_swapchain_dependent();
}

void DeferredRenderer::destroy_swapchain_dependent()
{
    gbuffer_destroy(m_gbuffer, m_device);

    if (m_geomFramebuffer) { vkDestroyFramebuffer(m_device, m_geomFramebuffer, nullptr); m_geomFramebuffer = VK_NULL_HANDLE; }
    if (m_geomRenderPass)  { vkDestroyRenderPass(m_device, m_geomRenderPass, nullptr);   m_geomRenderPass  = VK_NULL_HANDLE; }
    if (m_geomPipeline)    { vkDestroyPipeline(m_device, m_geomPipeline, nullptr);       m_geomPipeline    = VK_NULL_HANDLE; }
    if (m_geomPipeLayout)  { vkDestroyPipelineLayout(m_device, m_geomPipeLayout, nullptr); m_geomPipeLayout = VK_NULL_HANDLE; }

    if (m_hdrImageView)    { vkDestroyImageView(m_device, m_hdrImageView, nullptr);      m_hdrImageView    = VK_NULL_HANDLE; }
    if (m_hdrImage)        { vkDestroyImage(m_device, m_hdrImage, nullptr);              m_hdrImage        = VK_NULL_HANDLE; }
    if (m_hdrMemory)       { vkFreeMemory(m_device, m_hdrMemory, nullptr);               m_hdrMemory       = VK_NULL_HANDLE; }
    if (m_hdrSampler)      { vkDestroySampler(m_device, m_hdrSampler, nullptr);          m_hdrSampler      = VK_NULL_HANDLE; }
    if (m_lightingFB)      { vkDestroyFramebuffer(m_device, m_lightingFB, nullptr);      m_lightingFB      = VK_NULL_HANDLE; }
    if (m_lightingRP)      { vkDestroyRenderPass(m_device, m_lightingRP, nullptr);       m_lightingRP      = VK_NULL_HANDLE; }
    if (m_lightPipeline)   { vkDestroyPipeline(m_device, m_lightPipeline, nullptr);      m_lightPipeline   = VK_NULL_HANDLE; }
    if (m_lightPipeLayout) { vkDestroyPipelineLayout(m_device, m_lightPipeLayout, nullptr); m_lightPipeLayout = VK_NULL_HANDLE; }

    if (m_tonemapPipeline)   { vkDestroyPipeline(m_device, m_tonemapPipeline, nullptr);       m_tonemapPipeline   = VK_NULL_HANDLE; }
    if (m_tonemapPipeLayout) { vkDestroyPipelineLayout(m_device, m_tonemapPipeLayout, nullptr); m_tonemapPipeLayout = VK_NULL_HANDLE; }
}

void DeferredRenderer::rebuild_swapchain_dependent()
{
    gbuffer_create(m_gbuffer, m_device, m_physDevice, m_width, m_height);
    create_hdr_image();
    create_geometry_renderpass();
    create_geometry_framebuffer();
    create_geometry_pipeline();
    create_lighting_pipeline();
    create_tonemap_pipeline();
    // Descriptor sets must be re-written because image views have changed.
    create_descriptor_sets();
    m_ssao.on_resize(m_device, m_physDevice, m_width, m_height, &m_gbuffer);
}

// ─── HDR image ───────────────────────────────────────────────────────────────

void DeferredRenderer::create_hdr_image()
{
    // RGBA16_SFLOAT gives us four times the range of UNORM, sufficient for HDR
    // values before tone mapping.
    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = VK_FORMAT_R16G16B16A16_SFLOAT;
    ici.extent        = {m_width, m_height, 1};
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkCreateImage(m_device, &ici, nullptr, &m_hdrImage);

    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(m_device, m_hdrImage, &req);
    VkMemoryAllocateInfo mai{};
    mai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize  = req.size;
    mai.memoryTypeIndex = find_memory_type(m_physDevice, req.memoryTypeBits,
                                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkAllocateMemory(m_device, &mai, nullptr, &m_hdrMemory);
    vkBindImageMemory(m_device, m_hdrImage, m_hdrMemory, 0);

    VkImageViewCreateInfo vci{};
    vci.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image                           = m_hdrImage;
    vci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    vci.format                          = VK_FORMAT_R16G16B16A16_SFLOAT;
    vci.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.levelCount     = 1;
    vci.subresourceRange.layerCount     = 1;
    vkCreateImageView(m_device, &vci, nullptr, &m_hdrImageView);

    VkSamplerCreateInfo sci{};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_LINEAR;   // Linear: nicer for fullscreen tone-map
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod       = 0.0f;
    vkCreateSampler(m_device, &sci, nullptr, &m_hdrSampler);
}

// ─── Geometry render pass ─────────────────────────────────────────────────────

void DeferredRenderer::create_geometry_renderpass()
{
    // Attachment descriptions are generated by the GBuffer helper so changes
    // to G-buffer format only need to be updated in one place (GBuffer.cpp).
    VkAttachmentDescription2 attDescs[GBUFFER_ATTACHMENT_COUNT + 1]{};
    uint32_t attCount = 0;
    gbuffer_fill_attachment_descs(m_gbuffer, attDescs, &attCount);

    // Color attachment references (locations 0..4 in the fragment shader).
    std::array<VkAttachmentReference2, GBUFFER_ATTACHMENT_COUNT> colorRefs{};
    for (uint32_t i = 0; i < GBUFFER_ATTACHMENT_COUNT; ++i) {
        colorRefs[i].sType      = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
        colorRefs[i].attachment = i;
        colorRefs[i].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorRefs[i].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    // Depth attachment reference.
    VkAttachmentReference2 depthRef{};
    depthRef.sType      = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
    depthRef.attachment = GBUFFER_ATTACHMENT_COUNT;
    depthRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthRef.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    VkSubpassDescription2 subpass{};
    subpass.sType                   = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2;
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = GBUFFER_ATTACHMENT_COUNT;
    subpass.pColorAttachments       = colorRefs.data();
    subpass.pDepthStencilAttachment = &depthRef;

    // Subpass dependency: ensure the G-buffer is fully written before the
    // lighting pass reads it (COLOR_ATTACHMENT_OUTPUT → SHADER_READ).
    std::array<VkSubpassDependency2, 2> deps{};
    // Dependency 0: wait for previous frames' SHADER_READ before we start writing.
    deps[0].sType           = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
    deps[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass      = 0;
    deps[0].srcStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask   = VK_ACCESS_SHADER_READ_BIT;
    deps[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    // Dependency 1: wait for geometry writes before lighting reads.
    deps[1].sType           = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
    deps[1].srcSubpass      = 0;
    deps[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[1].dstStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                              VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo2 rpci{};
    rpci.sType                = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2;
    rpci.attachmentCount      = attCount;
    rpci.pAttachments         = attDescs;
    rpci.subpassCount         = 1;
    rpci.pSubpasses           = &subpass;
    rpci.dependencyCount      = static_cast<uint32_t>(deps.size());
    rpci.pDependencies        = deps.data();

    if (vkCreateRenderPass2(m_device, &rpci, nullptr, &m_geomRenderPass) != VK_SUCCESS)
        throw std::runtime_error("Failed to create geometry render pass");
}

void DeferredRenderer::create_geometry_framebuffer()
{
    // Collect image views: all G-buffer color attachments, then depth.
    std::array<VkImageView, GBUFFER_ATTACHMENT_COUNT + 1> views{};
    for (uint32_t i = 0; i < GBUFFER_ATTACHMENT_COUNT; ++i)
        views[i] = m_gbuffer.attachments[i].view;
    views[GBUFFER_ATTACHMENT_COUNT] = m_gbuffer.depth.view;

    VkFramebufferCreateInfo fci{};
    fci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fci.renderPass      = m_geomRenderPass;
    fci.attachmentCount = static_cast<uint32_t>(views.size());
    fci.pAttachments    = views.data();
    fci.width           = m_width;
    fci.height          = m_height;
    fci.layers          = 1;

    if (vkCreateFramebuffer(m_device, &fci, nullptr, &m_geomFramebuffer) != VK_SUCCESS)
        throw std::runtime_error("Failed to create geometry framebuffer");
}

// ─── Geometry pipeline ────────────────────────────────────────────────────────

void DeferredRenderer::create_geometry_pipeline()
{
    VkShaderModule vertMod = load_shader(m_device, "shaders/deferred/geometry.vert.spv");
    VkShaderModule fragMod = load_shader(m_device, "shaders/deferred/geometry.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName  = "main";

    // Vertex layout: position(vec3), normal(vec3), texcoord(vec2), tangent(vec4)
    VkVertexInputBindingDescription binding{};
    binding.binding   = 0;
    binding.stride    = sizeof(float) * (3 + 3 + 2 + 4);  // = 48 bytes
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 4> attribs{};
    attribs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,    0};          // position
    attribs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT,    12};         // normal
    attribs[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,        24};         // texcoord
    attribs[3] = {3, 0, VK_FORMAT_R32G32B32A32_SFLOAT,  32};         // tangent

    VkPipelineVertexInputStateCreateInfo visci{};
    visci.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    visci.vertexBindingDescriptionCount   = 1;
    visci.pVertexBindingDescriptions      = &binding;
    visci.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribs.size());
    visci.pVertexAttributeDescriptions    = attribs.data();

    VkPipelineInputAssemblyStateCreateInfo iasci{};
    iasci.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    iasci.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Dynamic viewport / scissor – updated by on_resize without pipeline recreation.
    VkPipelineDynamicStateCreateInfo dynci{};
    std::array<VkDynamicState, 2> dynStates = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
    };
    dynci.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynci.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
    dynci.pDynamicStates    = dynStates.data();

    VkPipelineViewportStateCreateInfo vsci{};
    vsci.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vsci.viewportCount = 1;
    vsci.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rsci{};
    rsci.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rsci.polygonMode = VK_POLYGON_MODE_FILL;
    rsci.cullMode    = VK_CULL_MODE_BACK_BIT;
    rsci.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rsci.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo msci{};
    msci.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msci.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo dsci{};
    dsci.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    dsci.depthTestEnable  = VK_TRUE;
    dsci.depthWriteEnable = VK_TRUE;
    dsci.depthCompareOp   = VK_COMPARE_OP_LESS;  // Standard near-to-far depth

    // All G-buffer color attachments: no blending (opaque values overwrite).
    std::array<VkPipelineColorBlendAttachmentState, GBUFFER_ATTACHMENT_COUNT> blendAtts{};
    for (auto& a : blendAtts) {
        a.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        a.blendEnable    = VK_FALSE;  // No blending – G-buffer is overwritten each frame
    }
    VkPipelineColorBlendStateCreateInfo cbsci{};
    cbsci.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbsci.attachmentCount = static_cast<uint32_t>(blendAtts.size());
    cbsci.pAttachments    = blendAtts.data();

    // Pipeline layout: set 0 = frame UBO, set 1 = material textures,
    // push constant = per-object model + normal matrices.
    std::array<VkDescriptorSetLayout, 2> setLayouts = {
        m_frameSetLayout,     // set 0
        VK_NULL_HANDLE        // set 1 – material layout not in scope here; see material system
    };

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(float) * 32;  // 2x mat4 = 128 bytes

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;  // Only bind frame UBO here; material set bound per-draw
    plci.pSetLayouts            = &m_frameSetLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcRange;
    vkCreatePipelineLayout(m_device, &plci, nullptr, &m_geomPipeLayout);

    VkGraphicsPipelineCreateInfo gpci{};
    gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.stageCount          = 2;
    gpci.pStages             = stages;
    gpci.pVertexInputState   = &visci;
    gpci.pInputAssemblyState = &iasci;
    gpci.pViewportState      = &vsci;
    gpci.pRasterizationState = &rsci;
    gpci.pMultisampleState   = &msci;
    gpci.pDepthStencilState  = &dsci;
    gpci.pColorBlendState    = &cbsci;
    gpci.pDynamicState       = &dynci;
    gpci.layout              = m_geomPipeLayout;
    gpci.renderPass          = m_geomRenderPass;
    gpci.subpass             = 0;

    vkCreateGraphicsPipelines(m_device, m_pipelineCache, 1, &gpci, nullptr, &m_geomPipeline);

    vkDestroyShaderModule(m_device, vertMod, nullptr);
    vkDestroyShaderModule(m_device, fragMod, nullptr);
}

// ─── Lighting pipeline ────────────────────────────────────────────────────────

void DeferredRenderer::create_lighting_pipeline()
{
    // Create the HDR render pass (renders into m_hdrImage).
    VkAttachmentDescription hdrAtt{};
    hdrAtt.format         = VK_FORMAT_R16G16B16A16_SFLOAT;
    hdrAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
    hdrAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    hdrAtt.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    hdrAtt.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    hdrAtt.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference hdrRef{};
    hdrRef.attachment = 0;
    hdrRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &hdrRef;

    // Wait for G-buffer writes (previous pass) before reading them.
    std::array<VkSubpassDependency, 2> deps{};
    deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass    = 0;
    deps[0].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    deps[1].srcSubpass    = 0;
    deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rpci{};
    rpci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 1;
    rpci.pAttachments    = &hdrAtt;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &subpass;
    rpci.dependencyCount = static_cast<uint32_t>(deps.size());
    rpci.pDependencies   = deps.data();
    vkCreateRenderPass(m_device, &rpci, nullptr, &m_lightingRP);

    VkFramebufferCreateInfo fci{};
    fci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fci.renderPass      = m_lightingRP;
    fci.attachmentCount = 1;
    fci.pAttachments    = &m_hdrImageView;
    fci.width           = m_width;
    fci.height          = m_height;
    fci.layers          = 1;
    vkCreateFramebuffer(m_device, &fci, nullptr, &m_lightingFB);

    // Lighting pipeline: fullscreen triangle, PBR lighting fragment.
    VkShaderModule vertMod = load_shader(m_device, "shaders/deferred/fullscreen_quad.vert.spv");
    VkShaderModule fragMod = load_shader(m_device, "shaders/deferred/lighting.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,   vertMod, "main", nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragMod, "main", nullptr};

    // No vertex input: vertex positions are generated in the vertex shader.
    VkPipelineVertexInputStateCreateInfo noVtx{};
    noVtx.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dynStates;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rast{};
    rast.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode    = VK_CULL_MODE_NONE;   // Fullscreen triangle: no culling
    rast.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // No depth test in the lighting pass: it's a 2D fullscreen effect.
    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType           = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_FALSE;
    ds.depthWriteEnable= VK_FALSE;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAtt.blendEnable    = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo cbs{};
    cbs.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbs.attachmentCount = 1;
    cbs.pAttachments    = &blendAtt;

    // Lighting pipeline layout: set 0 = G-buffer, set 1 = frame UBO, set 2 = lights
    std::array<VkDescriptorSetLayout, 3> lightLayouts = {
        m_gbufferSetLayout,   // set 0: G-buffer + SSAO textures
        m_frameSetLayout,     // set 1: FrameUBO
        m_lightSetLayout,     // set 2: Light SSBO
    };
    VkPipelineLayoutCreateInfo plci{};
    plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = static_cast<uint32_t>(lightLayouts.size());
    plci.pSetLayouts    = lightLayouts.data();
    vkCreatePipelineLayout(m_device, &plci, nullptr, &m_lightPipeLayout);

    VkGraphicsPipelineCreateInfo gpci{};
    gpci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.stageCount          = 2;
    gpci.pStages             = stages;
    gpci.pVertexInputState   = &noVtx;
    gpci.pInputAssemblyState = &ia;
    gpci.pViewportState      = &vp;
    gpci.pRasterizationState = &rast;
    gpci.pMultisampleState   = &ms;
    gpci.pDepthStencilState  = &ds;
    gpci.pColorBlendState    = &cbs;
    gpci.pDynamicState       = &dyn;
    gpci.layout              = m_lightPipeLayout;
    gpci.renderPass          = m_lightingRP;
    gpci.subpass             = 0;

    vkCreateGraphicsPipelines(m_device, m_pipelineCache, 1, &gpci, nullptr, &m_lightPipeline);

    vkDestroyShaderModule(m_device, vertMod, nullptr);
    vkDestroyShaderModule(m_device, fragMod, nullptr);
}

// ─── Tone-map pipeline ────────────────────────────────────────────────────────

void DeferredRenderer::create_tonemap_pipeline()
{
    // Renders into the swapchain framebuffer (provided per-frame by the caller).
    VkShaderModule vertMod = load_shader(m_device, "shaders/deferred/fullscreen_quad.vert.spv");
    VkShaderModule fragMod = load_shader(m_device, "shaders/deferred/tonemap.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,   vertMod, "main", nullptr};
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragMod, "main", nullptr};

    VkPipelineVertexInputStateCreateInfo noVtx{};
    noVtx.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
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
    rast.polygonMode = VK_POLYGON_MODE_FILL; rast.cullMode = VK_CULL_MODE_NONE; rast.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_FALSE; ds.depthWriteEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cbs{};
    cbs.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbs.attachmentCount = 1; cbs.pAttachments = &blendAtt;

    // Push constant carries exposure and tone-map operator selection.
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(float) + sizeof(uint32_t);  // exposure + operator

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &m_hdrSetLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcRange;
    vkCreatePipelineLayout(m_device, &plci, nullptr, &m_tonemapPipeLayout);

    VkGraphicsPipelineCreateInfo gpci{};
    gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.stageCount = 2; gpci.pStages = stages;
    gpci.pVertexInputState   = &noVtx;
    gpci.pInputAssemblyState = &ia;
    gpci.pViewportState      = &vp;
    gpci.pRasterizationState = &rast;
    gpci.pMultisampleState   = &ms;
    gpci.pDepthStencilState  = &ds;
    gpci.pColorBlendState    = &cbs;
    gpci.pDynamicState       = &dyn;
    gpci.layout              = m_tonemapPipeLayout;
    gpci.renderPass          = m_swapchainRP;
    gpci.subpass             = 0;

    vkCreateGraphicsPipelines(m_device, m_pipelineCache, 1, &gpci, nullptr, &m_tonemapPipeline);

    vkDestroyShaderModule(m_device, vertMod, nullptr);
    vkDestroyShaderModule(m_device, fragMod, nullptr);
}

// ─── Descriptor layouts ───────────────────────────────────────────────────────

void DeferredRenderer::create_descriptor_layouts()
{
    // Set 0: per-frame UBO (one uniform buffer binding).
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1;
        ci.pBindings    = &b;
        vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &m_frameSetLayout);
    }

    // Set 1: G-buffer textures (5 color + 1 depth + 1 SSAO = 7 combined-image-samplers).
    {
        const uint32_t N = GBUFFER_ATTACHMENT_COUNT + 2;  // +depth +ssao
        std::array<VkDescriptorSetLayoutBinding, N> bindings{};
        for (uint32_t i = 0; i < N; ++i) {
            bindings[i].binding        = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            bindings[i].descriptorCount= 1;
            bindings[i].stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = N;
        ci.pBindings    = bindings.data();
        vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &m_gbufferSetLayout);
    }

    // Set 2: light SSBO (one storage buffer binding).
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding        = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount= 1;
        b.stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1;
        ci.pBindings    = &b;
        vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &m_lightSetLayout);
    }

    // Set 3: HDR image (one combined-image-sampler for the tone-map pass).
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding        = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount= 1;
        b.stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1;
        ci.pBindings    = &b;
        vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &m_hdrSetLayout);
    }
}

// ─── Per-frame buffers and descriptors ────────────────────────────────────────

void DeferredRenderer::create_frame_uniforms_buffers()
{
    for (auto& f : m_frames) {
        auto buf = create_buffer(
            m_device, m_physDevice,
            sizeof(FrameUniforms),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        f.frameUBO    = buf.handle;
        f.frameUBOMem = buf.memory;
        f.frameUBOPtr = buf.mapped;
    }
}

void DeferredRenderer::create_light_ssbo()
{
    // Size: 4 bytes for count + MAX_LIGHTS * sizeof(GpuLight)
    VkDeviceSize size = sizeof(uint32_t) * 4 + sizeof(GpuLight) * MAX_LIGHTS;
    for (auto& f : m_frames) {
        auto buf = create_buffer(
            m_device, m_physDevice, size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        f.lightSSBO    = buf.handle;
        f.lightSSBOMem = buf.memory;
        f.lightSSBOPtr = buf.mapped;
    }
}

void DeferredRenderer::create_descriptor_sets()
{
    for (auto& f : m_frames) {
        // Allocate all four sets at once.
        std::array<VkDescriptorSetLayout, 4> layouts = {
            m_frameSetLayout, m_gbufferSetLayout, m_lightSetLayout, m_hdrSetLayout
        };
        std::array<VkDescriptorSet, 4> sets{};
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = m_descriptorPool;
        ai.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        ai.pSetLayouts        = layouts.data();
        vkAllocateDescriptorSets(m_device, &ai, sets.data());
        f.frameSet   = sets[0];
        f.gbufferSet = sets[1];
        f.lightSet   = sets[2];
        f.hdrSet     = sets[3];

        // Write frame UBO.
        VkDescriptorBufferInfo frameBI{f.frameUBO, 0, sizeof(FrameUniforms)};
        VkWriteDescriptorSet   frameW{};
        frameW.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        frameW.dstSet          = f.frameSet;
        frameW.dstBinding      = 0;
        frameW.descriptorCount = 1;
        frameW.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        frameW.pBufferInfo     = &frameBI;

        // Write G-buffer textures.
        gbuffer_write_descriptor_set(m_gbuffer, m_device, f.gbufferSet);

        // Write light SSBO.
        VkDeviceSize lightSize = sizeof(uint32_t) * 4 + sizeof(GpuLight) * MAX_LIGHTS;
        VkDescriptorBufferInfo lightBI{f.lightSSBO, 0, lightSize};
        VkWriteDescriptorSet   lightW{};
        lightW.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        lightW.dstSet          = f.lightSet;
        lightW.dstBinding      = 0;
        lightW.descriptorCount = 1;
        lightW.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        lightW.pBufferInfo     = &lightBI;

        // Write HDR image sampler.
        VkDescriptorImageInfo hdrII{m_hdrSampler, m_hdrImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet  hdrW{};
        hdrW.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        hdrW.dstSet          = f.hdrSet;
        hdrW.dstBinding      = 0;
        hdrW.descriptorCount = 1;
        hdrW.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        hdrW.pImageInfo      = &hdrII;

        std::array<VkWriteDescriptorSet, 3> writes = {frameW, lightW, hdrW};
        vkUpdateDescriptorSets(m_device,
                               static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }
}

// ─── Runtime API ─────────────────────────────────────────────────────────────

VkCommandBuffer DeferredRenderer::begin_geometry_pass(uint32_t frameIndex)
{
    VkCommandBuffer cb = m_geomCmdBuffers[frameIndex];

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    // Clear values for all G-buffer attachments + depth.
    std::array<VkClearValue, GBUFFER_ATTACHMENT_COUNT + 1> clears{};
    for (auto& c : clears) c.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    clears[GBUFFER_ATTACHMENT_COUNT].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpbi{};
    rpbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass      = m_geomRenderPass;
    rpbi.framebuffer     = m_geomFramebuffer;
    rpbi.renderArea      = {{0,0},{m_width, m_height}};
    rpbi.clearValueCount = static_cast<uint32_t>(clears.size());
    rpbi.pClearValues    = clears.data();
    vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, m_geomPipeline);

    VkViewport vp{0, 0, float(m_width), float(m_height), 0.0f, 1.0f};
    VkRect2D   sc{{0,0},{m_width, m_height}};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);

    // Bind the per-frame UBO (set 0); callers bind per-object material sets.
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_geomPipeLayout, 0, 1,
                            &m_frames[frameIndex].frameSet, 0, nullptr);
    return cb;
}

void DeferredRenderer::end_geometry_pass(uint32_t frameIndex)
{
    VkCommandBuffer cb = m_geomCmdBuffers[frameIndex];
    vkCmdEndRenderPass(cb);
    vkEndCommandBuffer(cb);
}

void DeferredRenderer::set_lights(uint32_t frameIndex,
                                   const GpuLight* lights,
                                   uint32_t count)
{
    assert(count <= MAX_LIGHTS);
    auto& f = m_frames[frameIndex];
    f.lightCount = count;
    // Copy count + light structs into the persistently mapped SSBO.
    // Host coherent memory: no explicit flush needed.
    uint8_t* ptr = static_cast<uint8_t*>(f.lightSSBOPtr);
    std::memcpy(ptr, &count, sizeof(uint32_t));
    std::memcpy(ptr + 16, lights, count * sizeof(GpuLight));  // 16-byte alignment
}

void DeferredRenderer::render_deferred_passes(uint32_t frameIndex,
                                               uint32_t /*swapchainImageIndex*/,
                                               VkFramebuffer swapchainFB,
                                               const FrameUniforms& frame,
                                               VkSemaphore waitSem,
                                               VkSemaphore signalSem,
                                               VkFence fence)
{
    // Upload this frame's uniforms via the persistently mapped UBO pointer.
    std::memcpy(m_frames[frameIndex].frameUBOPtr, &frame, sizeof(FrameUniforms));

    VkCommandBuffer cb = m_frames[frameIndex].deferredCB;
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    // Record each deferred pass in order.
    record_ssao_pass(cb, frameIndex);
    record_lighting_pass(cb, frameIndex);
    record_tonemap_pass(cb, frameIndex, swapchainFB);

    vkEndCommandBuffer(cb);

    // Submit: wait for the geometry pass semaphore (if any), then signal caller.
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount   = waitSem ? 1 : 0;
    si.pWaitSemaphores      = waitSem ? &waitSem : nullptr;
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cb;
    si.signalSemaphoreCount = signalSem ? 1 : 0;
    si.pSignalSemaphores    = signalSem ? &signalSem : nullptr;

    vkQueueSubmit(m_graphicsQueue, 1, &si, fence);
}

// ─── Pass recording helpers ───────────────────────────────────────────────────

void DeferredRenderer::record_ssao_pass(VkCommandBuffer cb, uint32_t frameIndex)
{
    // The SSAO pass reads the G-buffer depth and normal attachments.
    // At this point both are in SHADER_READ_ONLY_OPTIMAL (set by geometry render pass).
    // m_ssao.record() handles its own render-pass begin/end.
    // We pass the projection matrices from the latest frame uniforms.
    // (In a full implementation, cache these in the PerFrame struct.)
    m_ssao.record(cb, frameIndex, glm::mat4(1.0f), glm::mat4(1.0f)); // Caller fills real matrices
}

void DeferredRenderer::record_lighting_pass(VkCommandBuffer cb, uint32_t frameIndex)
{
    VkClearValue clear{};
    clear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderPassBeginInfo rpbi{};
    rpbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass      = m_lightingRP;
    rpbi.framebuffer     = m_lightingFB;
    rpbi.renderArea      = {{0,0},{m_width,m_height}};
    rpbi.clearValueCount = 1;
    rpbi.pClearValues    = &clear;
    vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, m_lightPipeline);

    VkViewport vp{0, 0, float(m_width), float(m_height), 0.0f, 1.0f};
    VkRect2D   sc{{0,0},{m_width,m_height}};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);

    // Bind G-buffer, frame UBO, and light SSBO.
    std::array<VkDescriptorSet, 3> sets = {
        m_frames[frameIndex].gbufferSet,
        m_frames[frameIndex].frameSet,
        m_frames[frameIndex].lightSet,
    };
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_lightPipeLayout, 0,
                            static_cast<uint32_t>(sets.size()),
                            sets.data(), 0, nullptr);

    // Draw the fullscreen triangle (3 vertices, no VBO).
    vkCmdDraw(cb, 3, 1, 0, 0);

    vkCmdEndRenderPass(cb);
}

void DeferredRenderer::record_tonemap_pass(VkCommandBuffer cb,
                                            uint32_t frameIndex,
                                            VkFramebuffer swapchainFB)
{
    VkClearValue clear{};
    clear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderPassBeginInfo rpbi{};
    rpbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass      = m_swapchainRP;
    rpbi.framebuffer     = swapchainFB;
    rpbi.renderArea      = {{0,0},{m_width,m_height}};
    rpbi.clearValueCount = 1;
    rpbi.pClearValues    = &clear;
    vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, m_tonemapPipeline);

    VkViewport vp{0, 0, float(m_width), float(m_height), 0.0f, 1.0f};
    VkRect2D   sc{{0,0},{m_width,m_height}};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);

    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_tonemapPipeLayout, 0, 1,
                            &m_frames[frameIndex].hdrSet, 0, nullptr);

    // Push exposure (0.0 EV = neutral) and operator (0 = ACES).
    struct { float exposure; uint32_t op; } pc{0.0f, 0u};
    vkCmdPushConstants(cb, m_tonemapPipeLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    vkCmdDraw(cb, 3, 1, 0, 0);

    vkCmdEndRenderPass(cb);
}

} // namespace vkgfx
