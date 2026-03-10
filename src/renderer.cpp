#include "vkgfx/renderer.h"
#include <chrono>
#include <algorithm>
#include <thread>
#include <future>
#include <unordered_set>

namespace vkgfx {

// ─── Constructor ──────────────────────────────────────────────────────────────
Renderer::Renderer(Window& window, const RendererSettings& settings)
    : m_window(window), m_settings(settings)
{
    Context::CreateInfo ci;
    ci.appName          = window.settings().title;
    ci.enableValidation = settings.validation;
    ci.preferDedicated  = true;

    m_ctx = std::make_shared<Context>(ci);
    m_surface = window.createSurface(m_ctx->instance());
    m_ctx->initDevice(m_surface);

    auto [w, h] = window.getFramebufferSize();
    VkSampleCountFlagBits samples    = static_cast<VkSampleCountFlagBits>(settings.msaa);
    VkSampleCountFlagBits maxSamples = m_ctx->maxSampleCount();
    while (samples > maxSamples) samples = static_cast<VkSampleCountFlagBits>(samples >> 1);

    m_swapchain = std::make_unique<Swapchain>(*m_ctx, m_surface, w, h,
                                               settings.vsync, samples);

    createDescriptorPool();
    createGlobalDescriptorSetLayout();
    createGlobalDescriptorSets();

    VkPipelineCacheCreateInfo pcCI{};
    pcCI.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    vkCreatePipelineCache(m_ctx->device(), &pcCI, nullptr, &m_pipelineCache_vk);

    m_whiteTexture = Texture::fromColor(m_ctx, {1,1,1,1});
    m_sceneRenderPass = m_swapchain->renderPass();

    createShadowResources();

    // ── Thread pool + per-worker secondary command pools ──────────────────────
    uint32_t workerCount = settings.workerThreads == 0
                           ? std::max(1u, std::thread::hardware_concurrency() - 1u)
                           : settings.workerThreads;
    m_threadPool = std::make_unique<ThreadPool>(workerCount);
    createWorkerCommandPools(workerCount);
    m_stats.workerThreads = workerCount;

    std::cout << "[VKGFX] Renderer initialised ("
              << w << "x" << h << ", MSAA x"
              << static_cast<uint32_t>(samples) << ", "
              << workerCount << " worker threads)\n";
}

Renderer::~Renderer() { shutdown(); }

// ─── shutdown ─────────────────────────────────────────────────────────────────
void Renderer::shutdown(Scene* scene) {
    if (!m_ctx) return;
    m_ctx->waitIdle();

    for (auto& entry : m_deletionQueue) {
        m_ctx->destroyBuffer(entry.vertexBuffer);
        m_ctx->destroyBuffer(entry.indexBuffer);
        for (auto& buf : entry.matBuffers) m_ctx->destroyBuffer(buf);
    }
    m_deletionQueue.clear();

    m_threadPool.reset();
    destroyWorkerCommandPools();

    destroyShadowResources();
    shutdownPostProcess();

    for (auto& [meshPtr, data] : m_meshData) {
        m_ctx->destroyBuffer(data.vertexBuffer);
        m_ctx->destroyBuffer(data.indexBuffer);
        for (auto& buf : data.matBuffers) m_ctx->destroyBuffer(buf);
        if (scene) {
            for (auto& m : scene->meshes()) {
                if (m.get() == meshPtr) {
                    m->vertexBuffer = {};
                    m->indexBuffer  = {};
                    m->gpuReady     = false;
                    break;
                }
            }
        }
    }
    m_meshData.clear();

    m_whiteTexture.reset();

    destroyPipelines();

    for (auto& ubo : m_cameraUBOs) m_ctx->destroyBuffer(ubo);
    for (auto& ubo : m_sceneUBOs)  m_ctx->destroyBuffer(ubo);

    if (m_globalSetLayout) vkDestroyDescriptorSetLayout(m_ctx->device(), m_globalSetLayout, nullptr);
    if (m_descriptorPool)  vkDestroyDescriptorPool(m_ctx->device(), m_descriptorPool, nullptr);
    if (m_pipelineCache_vk) vkDestroyPipelineCache(m_ctx->device(), m_pipelineCache_vk, nullptr);

    m_swapchain.reset();
    if (m_surface) { vkDestroySurfaceKHR(m_ctx->instance(), m_surface, nullptr); m_surface = VK_NULL_HANDLE; }
    m_ctx.reset();
}

// ─── setPostProcess ───────────────────────────────────────────────────────────
void Renderer::setPostProcess(const PostProcessSettings& pp) {
    bool wasActive = m_ppActive;
    m_ppSettings   = pp;

    if (pp.enabled && !wasActive) {
        m_ctx->waitIdle();
        destroyPipelines();
        m_ppActive = true;
        initPostProcess();
    } else if (!pp.enabled && wasActive) {
        m_ctx->waitIdle();
        shutdownPostProcess();
        m_sceneRenderPass = m_swapchain->renderPass();
        destroyPipelines();
    }
}

// ─── Post-process init / shutdown ─────────────────────────────────────────────
void Renderer::initPostProcess() {
    createOffscreenResources();
    createPPRenderPass();
    createPPFramebuffers();
    createPPDescriptorLayoutAndSets();
    createPPPipeline();
    m_ppDescWrittenFrames = 0;   // force image descriptor write on first frame
}

void Renderer::shutdownPostProcess() {
    destroyPPPipeline();
    destroyPPDescriptorResources();
    destroyPPFramebuffers();
    destroyPPRenderPass();
    destroyOffscreenResources();
    m_ppActive = false;
    m_ppDescWrittenFrames = 0;
}

// ─── Offscreen HDR resources ──────────────────────────────────────────────────
void Renderer::createOffscreenResources() {
    VkExtent2D ext = m_swapchain->extent();

    m_offscreenColor = m_ctx->createImage(
        ext.width, ext.height, 1, VK_SAMPLE_COUNT_1_BIT,
        VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_offscreenColor.mipLevels = 1;
    m_ctx->createImageView(m_offscreenColor, VK_IMAGE_ASPECT_COLOR_BIT);

    VkFormat depthFmt = m_ctx->findDepthFormat();
    m_offscreenDepth = m_ctx->createImage(
        ext.width, ext.height, 1, VK_SAMPLE_COUNT_1_BIT,
        depthFmt, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_offscreenDepth.mipLevels = 1;
    m_ctx->createImageView(m_offscreenDepth, VK_IMAGE_ASPECT_DEPTH_BIT);

    VkSamplerCreateInfo si{};
    si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter    = VK_FILTER_LINEAR;
    si.minFilter    = VK_FILTER_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.maxLod       = 0.f;
    VK_CHECK(vkCreateSampler(m_ctx->device(), &si, nullptr, &m_offscreenSampler), "Offscreen sampler");

    VkAttachmentDescription colorAtt{};
    colorAtt.format         = VK_FORMAT_R16G16B16A16_SFLOAT;
    colorAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentDescription depthAtt{};
    depthAtt.format         = depthFmt;
    depthAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
    depthAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAtt.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthRef{ 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription sub{};
    sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount    = 1;
    sub.pColorAttachments       = &colorRef;
    sub.pDepthStencilAttachment = &depthRef;

    std::array<VkSubpassDependency, 2> deps{};
    deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass    = 0;
    deps[0].srcStageMask  = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    deps[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    deps[1].srcSubpass    = 0;
    deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    std::array<VkAttachmentDescription, 2> attachments{ colorAtt, depthAtt };
    VkRenderPassCreateInfo rpCI{};
    rpCI.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpCI.attachmentCount = static_cast<uint32_t>(attachments.size());
    rpCI.pAttachments    = attachments.data();
    rpCI.subpassCount    = 1;
    rpCI.pSubpasses      = &sub;
    rpCI.dependencyCount = static_cast<uint32_t>(deps.size());
    rpCI.pDependencies   = deps.data();
    VK_CHECK(vkCreateRenderPass(m_ctx->device(), &rpCI, nullptr, &m_offscreenRenderPass),
             "Offscreen render pass");

    std::array<VkImageView, 2> fbViews{ m_offscreenColor.view, m_offscreenDepth.view };
    VkFramebufferCreateInfo fbCI{};
    fbCI.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbCI.renderPass      = m_offscreenRenderPass;
    fbCI.attachmentCount = static_cast<uint32_t>(fbViews.size());
    fbCI.pAttachments    = fbViews.data();
    fbCI.width           = ext.width;
    fbCI.height          = ext.height;
    fbCI.layers          = 1;
    VK_CHECK(vkCreateFramebuffer(m_ctx->device(), &fbCI, nullptr, &m_offscreenFramebuffer),
             "Offscreen framebuffer");

    m_sceneRenderPass = m_offscreenRenderPass;
}

void Renderer::destroyOffscreenResources() {
    if (m_offscreenFramebuffer) {
        vkDestroyFramebuffer(m_ctx->device(), m_offscreenFramebuffer, nullptr);
        m_offscreenFramebuffer = VK_NULL_HANDLE;
    }
    if (m_offscreenRenderPass) {
        vkDestroyRenderPass(m_ctx->device(), m_offscreenRenderPass, nullptr);
        m_offscreenRenderPass = VK_NULL_HANDLE;
    }
    if (m_offscreenSampler) {
        vkDestroySampler(m_ctx->device(), m_offscreenSampler, nullptr);
        m_offscreenSampler = VK_NULL_HANDLE;
    }
    m_ctx->destroyImage(m_offscreenColor);
    m_ctx->destroyImage(m_offscreenDepth);
}

// ─── PP render pass ───────────────────────────────────────────────────────────
void Renderer::createPPRenderPass() {
    VkAttachmentDescription colorAtt{};
    colorAtt.format         = m_swapchain->format();
    colorAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    VkSubpassDescription sub{};
    sub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments    = &colorRef;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpCI{};
    rpCI.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpCI.attachmentCount = 1;
    rpCI.pAttachments    = &colorAtt;
    rpCI.subpassCount    = 1;
    rpCI.pSubpasses      = &sub;
    rpCI.dependencyCount = 1;
    rpCI.pDependencies   = &dep;
    VK_CHECK(vkCreateRenderPass(m_ctx->device(), &rpCI, nullptr, &m_ppRenderPass), "PP render pass");
}

void Renderer::destroyPPRenderPass() {
    if (m_ppRenderPass) {
        vkDestroyRenderPass(m_ctx->device(), m_ppRenderPass, nullptr);
        m_ppRenderPass = VK_NULL_HANDLE;
    }
}

// ─── PP framebuffers ──────────────────────────────────────────────────────────
void Renderer::createPPFramebuffers() {
    uint32_t  n   = m_swapchain->imageCount();
    VkExtent2D ext = m_swapchain->extent();
    m_ppFramebuffers.resize(n, VK_NULL_HANDLE);

    for (uint32_t i = 0; i < n; ++i) {
        VkImageView view = m_swapchain->imageView(i);
        VkFramebufferCreateInfo fbCI{};
        fbCI.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbCI.renderPass      = m_ppRenderPass;
        fbCI.attachmentCount = 1;
        fbCI.pAttachments    = &view;
        fbCI.width           = ext.width;
        fbCI.height          = ext.height;
        fbCI.layers          = 1;
        VK_CHECK(vkCreateFramebuffer(m_ctx->device(), &fbCI, nullptr, &m_ppFramebuffers[i]),
                 "PP framebuffer");
    }
}

void Renderer::destroyPPFramebuffers() {
    for (auto& fb : m_ppFramebuffers)
        if (fb) { vkDestroyFramebuffer(m_ctx->device(), fb, nullptr); fb = VK_NULL_HANDLE; }
    m_ppFramebuffers.clear();
}

// ─── PP descriptor layout + sets ─────────────────────────────────────────────
void Renderer::createPPDescriptorLayoutAndSets() {
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0] = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
    bindings[1] = { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };

    VkDescriptorSetLayoutCreateInfo dslCI{};
    dslCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslCI.bindingCount = static_cast<uint32_t>(bindings.size());
    dslCI.pBindings    = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(m_ctx->device(), &dslCI, nullptr, &m_ppDescSetLayout),
             "PP descriptor set layout");

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_ppUBOs[i] = m_ctx->createBuffer(sizeof(PostProcessUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts;
    layouts.fill(m_ppDescSetLayout);
    VkDescriptorSetAllocateInfo dsAI{};
    dsAI.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAI.descriptorPool     = m_descriptorPool;
    dsAI.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    dsAI.pSetLayouts        = layouts.data();
    VK_CHECK(vkAllocateDescriptorSets(m_ctx->device(), &dsAI, m_ppDescSets.data()), "PP descriptor sets");
}

void Renderer::destroyPPDescriptorResources() {
    for (auto& ubo : m_ppUBOs) m_ctx->destroyBuffer(ubo);
    if (m_ppDescSetLayout) {
        vkDestroyDescriptorSetLayout(m_ctx->device(), m_ppDescSetLayout, nullptr);
        m_ppDescSetLayout = VK_NULL_HANDLE;
    }
}

void Renderer::updatePPDescriptors(uint32_t frameIdx) {
    // UBO is always updated (cheap persistent-mapped memcpy — ~64 bytes)
    auto ubo = m_ppSettings.toUBO();
    std::memcpy(m_ppUBOs[frameIdx].mapped, &ubo, sizeof(PostProcessUBO));

    // The offscreen image view never changes between frames. Only write the
    // combined-image-sampler descriptor once per frame slot; skip it on every
    // subsequent frame to avoid a redundant driver call.
    const uint32_t bit = 1u << frameIdx;
    if (m_ppDescWrittenFrames & bit) return;
    m_ppDescWrittenFrames |= bit;

    VkDescriptorImageInfo imgInfo{};
    imgInfo.sampler     = m_offscreenSampler;
    imgInfo.imageView   = m_offscreenColor.view;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo bufInfo{ m_ppUBOs[frameIdx].buffer, 0, sizeof(PostProcessUBO) };

    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = m_ppDescSets[frameIdx];
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo      = &imgInfo;

    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = m_ppDescSets[frameIdx];
    writes[1].dstBinding      = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].pBufferInfo     = &bufInfo;

    vkUpdateDescriptorSets(m_ctx->device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

// ─── PP pipeline ─────────────────────────────────────────────────────────────
void Renderer::createPPPipeline() {
    VkPipelineLayoutCreateInfo layoutCI{};
    layoutCI.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCI.setLayoutCount = 1;
    layoutCI.pSetLayouts    = &m_ppDescSetLayout;
    VK_CHECK(vkCreatePipelineLayout(m_ctx->device(), &layoutCI, nullptr, &m_ppPipelineLayout),
             "PP pipeline layout");

    auto vertPath = m_settings.shaderDir / "postprocess.vert.spv";
    auto fragPath = m_settings.shaderDir / "postprocess.frag.spv";
    VkShaderModule vertMod = createShaderModule(vertPath);
    VkShaderModule fragMod = createShaderModule(fragPath);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = vertMod; stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fragMod; stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{0,0,1,1,0,1}; VkRect2D sc{{0,0},{1,1}};
    VkPipelineViewportStateCreateInfo vpState{};
    vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpState.viewportCount = 1; vpState.pViewports = &vp;
    vpState.scissorCount  = 1; vpState.pScissors  = &sc;

    VkPipelineRasterizationStateCreateInfo rast{};
    rast.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode    = VK_CULL_MODE_NONE;
    rast.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth   = 1.f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    VkPipelineColorBlendAttachmentState blendAttach{};
    blendAttach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                 VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments    = &blendAttach;

    std::array<VkDynamicState, 2> dynStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
    dyn.pDynamicStates    = dynStates.data();

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount          = 2;
    pci.pStages             = stages;
    pci.pVertexInputState   = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState      = &vpState;
    pci.pRasterizationState = &rast;
    pci.pMultisampleState   = &ms;
    pci.pDepthStencilState  = &ds;
    pci.pColorBlendState    = &blend;
    pci.pDynamicState       = &dyn;
    pci.layout              = m_ppPipelineLayout;
    pci.renderPass          = m_ppRenderPass;
    pci.subpass             = 0;

    VK_CHECK(vkCreateGraphicsPipelines(m_ctx->device(), m_pipelineCache_vk, 1, &pci, nullptr, &m_ppPipeline),
             "PP graphics pipeline");

    vkDestroyShaderModule(m_ctx->device(), vertMod, nullptr);
    vkDestroyShaderModule(m_ctx->device(), fragMod, nullptr);
}

void Renderer::destroyPPPipeline() {
    if (m_ppPipeline)       { vkDestroyPipeline      (m_ctx->device(), m_ppPipeline,       nullptr); m_ppPipeline       = VK_NULL_HANDLE; }
    if (m_ppPipelineLayout) { vkDestroyPipelineLayout(m_ctx->device(), m_ppPipelineLayout, nullptr); m_ppPipelineLayout = VK_NULL_HANDLE; }
}

// ─── PP pass recording ────────────────────────────────────────────────────────
void Renderer::recordPPPass(VkCommandBuffer cmd, uint32_t imageIdx, uint32_t frameIdx) {
    updatePPDescriptors(frameIdx);

    VkExtent2D ext = m_swapchain->extent();

    VkRenderPassBeginInfo rpBI{};
    rpBI.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBI.renderPass        = m_ppRenderPass;
    rpBI.framebuffer       = m_ppFramebuffers[imageIdx];
    rpBI.renderArea.extent = ext;
    rpBI.clearValueCount   = 0;
    rpBI.pClearValues      = nullptr;
    vkCmdBeginRenderPass(cmd, &rpBI, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vp{};
    vp.width    = static_cast<float>(ext.width);
    vp.height   = static_cast<float>(ext.height);
    vp.minDepth = 0.f; vp.maxDepth = 1.f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {0,0}, ext };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ppPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                             m_ppPipelineLayout, 0, 1, &m_ppDescSets[frameIdx], 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmd);
}

// ─── render() ─────────────────────────────────────────────────────────────────
void Renderer::render(Scene& scene) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // Detect and enqueue stale mesh entries for deferred deletion
    {
        std::unordered_set<Mesh*> live;
        live.reserve(scene.meshes().size());
        for (auto& m : scene.meshes()) live.insert(m.get());

        std::vector<Mesh*> stale;
        for (auto& [ptr, data] : m_meshData)
            if (!live.count(ptr)) stale.push_back(ptr);

        for (auto* ptr : stale) {
            auto it = m_meshData.find(ptr);
            DeferredBuffers deferred;
            deferred.frameIndex   = m_frameCounter;
            deferred.vertexBuffer = it->second.vertexBuffer;
            deferred.indexBuffer  = it->second.indexBuffer;
            deferred.matBuffers   = it->second.matBuffers;
            it->second.vertexBuffer = {};
            it->second.indexBuffer  = {};
            for (auto& b : it->second.matBuffers) b = {};
            m_deletionQueue.push_back(std::move(deferred));
            m_meshData.erase(it);
        }
    }

    ++m_frameCounter;

    for (auto& mesh : scene.meshes())
        if (!mesh->gpuReady) uploadMeshToGPU(*mesh);

    uint32_t frameIdx = m_currentFrame;
    uint32_t imageIdx = 0;
    VkResult res = m_swapchain->acquireNextImage(frameIdx, imageIdx);
    if (res == VK_ERROR_OUT_OF_DATE_KHR) { handleResize(); return; }
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("Failed to acquire swapchain image");

    // Flush deferred deletions for frames that have fully retired
    {
        const uint64_t safeFrame = m_frameCounter > MAX_FRAMES_IN_FLIGHT
                                 ? m_frameCounter - MAX_FRAMES_IN_FLIGHT - 1
                                 : 0;
        auto it = m_deletionQueue.begin();
        while (it != m_deletionQueue.end()) {
            if (it->frameIndex <= safeFrame) {
                m_ctx->destroyBuffer(it->vertexBuffer);
                m_ctx->destroyBuffer(it->indexBuffer);
                for (auto& buf : it->matBuffers) m_ctx->destroyBuffer(buf);
                it = m_deletionQueue.erase(it);
            } else {
                ++it;
            }
        }
    }

    auto& frame = m_swapchain->frame(frameIdx);
    vkResetFences(m_ctx->device(), 1, &frame.inFlightFence);
    vkResetCommandBuffer(frame.commandBuffer, 0);

    updateGlobalDescriptors(frameIdx);

    m_stats.drawCalls = 0;
    recordCommandBuffer(frame.commandBuffer, imageIdx, scene, frameIdx);

    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    si.waitSemaphoreCount   = 1;
    VkSemaphore imgAvailSem = m_swapchain->imageAvailableSemaphore(frameIdx);
    si.pWaitSemaphores      = &imgAvailSem;
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &frame.commandBuffer;
    si.signalSemaphoreCount = 1;
    VkSemaphore renderFinishedSem = m_swapchain->renderFinishedSemaphore(imageIdx);
    si.pSignalSemaphores    = &renderFinishedSem;
    VK_CHECK(vkQueueSubmit(m_ctx->graphicsQueue(), 1, &si, frame.inFlightFence), "Queue submit");

    res = m_swapchain->present(imageIdx);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR || m_window.wasResized()) {
        m_window.resetResizeFlag();
        handleResize();
    }

    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    auto t1 = std::chrono::high_resolution_clock::now();
    m_stats.frameTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
    m_stats.fps = 1000.f / (m_stats.frameTimeMs + 1e-6f);
}

// ─── Descriptor pool ──────────────────────────────────────────────────────────
void Renderer::createDescriptorPool() {
    std::vector<VkDescriptorPoolSize> sizes = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         512 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 512 },
    };
    VkDescriptorPoolCreateInfo ci{};
    ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.maxSets       = MAX_DESCRIPTOR_SETS;
    ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
    ci.pPoolSizes    = sizes.data();
    VK_CHECK(vkCreateDescriptorPool(m_ctx->device(), &ci, nullptr, &m_descriptorPool), "Descriptor pool");
}

void Renderer::createGlobalDescriptorSetLayout() {
    std::array<VkDescriptorSetLayoutBinding, 4> bindings{{
        { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
    }};
    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ci.pBindings    = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(m_ctx->device(), &ci, nullptr, &m_globalSetLayout), "Global DSL");
}

void Renderer::createGlobalDescriptorSets() {
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_cameraUBOs[i] = m_ctx->createBuffer(sizeof(CameraUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        m_sceneUBOs[i]  = m_ctx->createBuffer(sizeof(SceneUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts;
    layouts.fill(m_globalSetLayout);
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = m_descriptorPool;
    ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    ai.pSetLayouts        = layouts.data();
    VK_CHECK(vkAllocateDescriptorSets(m_ctx->device(), &ai, m_globalSets.data()), "Global DS alloc");
}

void Renderer::updateGlobalDescriptors(uint32_t frameIdx) {
    if (!m_window.settings().width) return;

    VkDescriptorBufferInfo camInfo   { m_cameraUBOs[frameIdx].buffer, 0, sizeof(CameraUBO) };
    VkDescriptorBufferInfo sceneInfo { m_sceneUBOs[frameIdx].buffer,  0, sizeof(SceneUBO)  };
    VkDescriptorBufferInfo shadowInfo{ m_shadow.ubos[frameIdx].buffer, 0, sizeof(ShadowUBO) };
    VkDescriptorImageInfo  shadowImg {
        m_shadow.sampler, m_shadow.arrayView,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    };

    std::array<VkWriteDescriptorSet, 4> writes{};
    writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                  m_globalSets[frameIdx], 0, 0, 1,
                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &camInfo };
    writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                  m_globalSets[frameIdx], 1, 0, 1,
                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &sceneInfo };
    writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                  m_globalSets[frameIdx], 2, 0, 1,
                  VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &shadowInfo };
    writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
                  m_globalSets[frameIdx], 3, 0, 1,
                  VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &shadowImg, nullptr };
    vkUpdateDescriptorSets(m_ctx->device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

// ─── Pipeline management ──────────────────────────────────────────────────────
VkDescriptorSetLayout Renderer::getOrCreateMatDescLayout(std::string_view shaderName,
                                                          uint32_t texCount)
{
    std::string key = std::string(shaderName) + "::" + std::to_string(texCount);
    if (auto it = m_matDescLayouts.find(key); it != m_matDescLayouts.end()) return it->second;

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back({ 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
    for (uint32_t i = 0; i < texCount; ++i)
        bindings.push_back({ i+1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });

    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ci.pBindings    = bindings.data();
    VkDescriptorSetLayout layout;
    VK_CHECK(vkCreateDescriptorSetLayout(m_ctx->device(), &ci, nullptr, &layout), "Mat DSL");
    m_matDescLayouts[key] = layout;
    return layout;
}

PipelineEntry& Renderer::getOrCreatePipeline(std::string_view shaderName,
                                               const PipelineSettings& ps,
                                               VkDescriptorSetLayout matLayout)
{
    std::string key = std::string(shaderName) + "::"
                    + std::to_string(static_cast<int>(ps.cullMode)) + "::"
                    + std::to_string(static_cast<int>(ps.polygonMode)) + "::"
                    + std::to_string(ps.alphaBlend);
    if (auto it = m_pipelineCache.find(key); it != m_pipelineCache.end()) return it->second;

    auto entry = createPipeline(shaderName, ps, matLayout);
    m_pipelineCache[key] = entry;
    return m_pipelineCache[key];
}

PipelineEntry Renderer::createPipeline(std::string_view shaderName,
                                        const PipelineSettings& ps,
                                        VkDescriptorSetLayout matLayout)
{
    auto vertPath = m_settings.shaderDir / (std::string(shaderName) + ".vert.spv");
    auto fragPath = m_settings.shaderDir / (std::string(shaderName) + ".frag.spv");
    VkShaderModule vertModule = createShaderModule(vertPath);
    VkShaderModule fragModule = createShaderModule(fragPath);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = vertModule; stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fragModule; stages[1].pName = "main";

    auto bindDesc = Vertex::getBindingDescription();
    auto attrDesc = Vertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions    = &bindDesc;
    const uint32_t attrCount = (shaderName == "pbr") ? 4u : 3u;
    vi.vertexAttributeDescriptionCount = attrCount;
    vi.pVertexAttributeDescriptions    = attrDesc.data();

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{0,0,1,1,0,1}; VkRect2D sc{{0,0},{1,1}};
    VkPipelineViewportStateCreateInfo vpState{};
    vpState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vpState.viewportCount = 1; vpState.pViewports = &vp;
    vpState.scissorCount  = 1; vpState.pScissors  = &sc;

    VkPipelineRasterizationStateCreateInfo rast{};
    rast.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rast.polygonMode = (ps.polygonMode == PolygonMode::Wireframe) ? VK_POLYGON_MODE_LINE :
                       (ps.polygonMode == PolygonMode::Points)    ? VK_POLYGON_MODE_POINT :
                                                                     VK_POLYGON_MODE_FILL;
    rast.cullMode    = (ps.cullMode == CullMode::None)         ? VK_CULL_MODE_NONE :
                       (ps.cullMode == CullMode::Front)        ? VK_CULL_MODE_FRONT_BIT :
                       (ps.cullMode == CullMode::FrontAndBack) ? VK_CULL_MODE_FRONT_AND_BACK :
                                                                  VK_CULL_MODE_BACK_BIT;
    rast.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth   = 1.f;

    VkSampleCountFlagBits sampleCount = m_ppActive ? VK_SAMPLE_COUNT_1_BIT
                                                    : m_swapchain->sampleCount();
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = sampleCount;
    ms.sampleShadingEnable  = (sampleCount != VK_SAMPLE_COUNT_1_BIT) ? VK_TRUE : VK_FALSE;
    ms.minSampleShading     = 0.2f;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = ps.depthTest  ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = ps.depthWrite ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAttach{};
    blendAttach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                 VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    if (ps.alphaBlend) {
        blendAttach.blendEnable         = VK_TRUE;
        blendAttach.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blendAttach.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAttach.colorBlendOp        = VK_BLEND_OP_ADD;
        blendAttach.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttach.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blendAttach.alphaBlendOp        = VK_BLEND_OP_ADD;
    }
    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments    = &blendAttach;

    std::array<VkDynamicState, 2> dynStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
    dyn.pDynamicStates    = dynStates.data();

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.size       = sizeof(ModelPushConstant);

    std::array<VkDescriptorSetLayout, 2> setLayouts = { m_globalSetLayout, matLayout };
    VkPipelineLayoutCreateInfo layoutCI{};
    layoutCI.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCI.setLayoutCount         = static_cast<uint32_t>(setLayouts.size());
    layoutCI.pSetLayouts            = setLayouts.data();
    layoutCI.pushConstantRangeCount = 1;
    layoutCI.pPushConstantRanges    = &pushRange;

    PipelineEntry entry;
    VK_CHECK(vkCreatePipelineLayout(m_ctx->device(), &layoutCI, nullptr, &entry.layout), "Pipeline layout");

    VkGraphicsPipelineCreateInfo pci{};
    pci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pci.stageCount          = 2;
    pci.pStages             = stages;
    pci.pVertexInputState   = &vi;
    pci.pInputAssemblyState = &ia;
    pci.pViewportState      = &vpState;
    pci.pRasterizationState = &rast;
    pci.pMultisampleState   = &ms;
    pci.pDepthStencilState  = &ds;
    pci.pColorBlendState    = &blend;
    pci.pDynamicState       = &dyn;
    pci.layout              = entry.layout;
    pci.renderPass          = m_sceneRenderPass;
    pci.flags               = VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT;

    VK_CHECK(vkCreateGraphicsPipelines(m_ctx->device(), m_pipelineCache_vk, 1, &pci, nullptr, &entry.pipeline),
             "Create graphics pipeline: " + std::string(shaderName));

    vkDestroyShaderModule(m_ctx->device(), vertModule, nullptr);
    vkDestroyShaderModule(m_ctx->device(), fragModule, nullptr);
    return entry;
}

VkShaderModule Renderer::createShaderModule(const std::filesystem::path& path) {
    auto code = readFile(path);
    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode    = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(m_ctx->device(), &ci, nullptr, &mod), "Shader module: " + path.string());
    return mod;
}

void Renderer::destroyPipelines() {
    for (auto& [key, entry] : m_pipelineCache) {
        if (entry.pipeline) vkDestroyPipeline(m_ctx->device(), entry.pipeline, nullptr);
        if (entry.layout)   vkDestroyPipelineLayout(m_ctx->device(), entry.layout, nullptr);
    }
    m_pipelineCache.clear();
    for (auto& [key, layout] : m_matDescLayouts)
        vkDestroyDescriptorSetLayout(m_ctx->device(), layout, nullptr);
    m_matDescLayouts.clear();
}

// ─── Mesh GPU resources ───────────────────────────────────────────────────────
void Renderer::uploadMeshToGPU(Mesh& mesh) {
    if (mesh.vertices().empty()) { mesh.gpuReady = true; return; }
    VkDeviceSize vSize = sizeof(Vertex)   * mesh.vertices().size();
    VkDeviceSize iSize = sizeof(uint32_t) * mesh.indices().size();
    mesh.vertexBuffer = m_ctx->uploadBuffer(mesh.vertices().data(), vSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    if (!mesh.indices().empty())
        mesh.indexBuffer = m_ctx->uploadBuffer(mesh.indices().data(), iSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    mesh.gpuReady = true;

    auto& data = m_meshData[&mesh];
    data.vertexBuffer = mesh.vertexBuffer;
    data.indexBuffer  = mesh.indexBuffer;
}

MeshGPUData& Renderer::getOrCreateMeshData(Mesh* mesh, Material* mat) {
    auto& data = m_meshData[mesh];
    if (!data.initialized) {
        VkDescriptorSetLayout matLayout = getOrCreateMatDescLayout(mat->shaderName(), MAX_TEXTURES_PER_MAT);
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            VkDeviceSize propSize = std::max(mat->propertiesSize(), (size_t)16);
            data.matBuffers[i] = m_ctx->createBuffer(propSize,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts;
        layouts.fill(matLayout);
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = m_descriptorPool;
        ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
        ai.pSetLayouts        = layouts.data();
        VK_CHECK(vkAllocateDescriptorSets(m_ctx->device(), &ai, data.matDescSets.data()), "Mat DS alloc");
        data.initialized = true;
    }
    return data;
}

void Renderer::updateMaterialDescriptors(MeshGPUData& data, Material* mat, uint32_t frameIdx) {
    const uint32_t frameBit    = 1u << frameIdx;
    const bool     neverWritten = !(data.writtenFrames & frameBit);
    if (!mat->isFrameDirty(frameIdx) && !neverWritten) return;

    if (mat->propertiesData())
        std::memcpy(data.matBuffers[frameIdx].mapped,
                    mat->propertiesData(), mat->propertiesSize());

    std::vector<VkWriteDescriptorSet> writes;
    VkDescriptorBufferInfo bufInfo{ data.matBuffers[frameIdx].buffer, 0,
                                    std::max(mat->propertiesSize(), (size_t)16) };
    VkWriteDescriptorSet w0{};
    w0.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w0.dstSet = data.matDescSets[frameIdx];
    w0.dstBinding = 0; w0.descriptorCount = 1; w0.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w0.pBufferInfo = &bufInfo;
    writes.push_back(w0);

    std::vector<VkDescriptorImageInfo> imgInfos;
    for (uint32_t i = 0; i < MAX_TEXTURES_PER_MAT; ++i) {
        auto tex = mat->getTexture(i);
        if (!tex) tex = m_whiteTexture;
        imgInfos.push_back(tex->descriptorInfo());
    }
    for (uint32_t i = 0; i < MAX_TEXTURES_PER_MAT; ++i) {
        VkWriteDescriptorSet wi{};
        wi.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wi.dstSet = data.matDescSets[frameIdx];
        wi.dstBinding = i+1; wi.descriptorCount = 1;
        wi.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wi.pImageInfo = &imgInfos[i];
        writes.push_back(wi);
    }
    vkUpdateDescriptorSets(m_ctx->device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    data.writtenFrames |= frameBit;
    mat->markClean(frameIdx);
}

// ─── Frame recording ──────────────────────────────────────────────────────────
void Renderer::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIdx,
                                    Scene& scene, uint32_t frameIdx)
{
    if (auto* cam = scene.camera()) {
        cam->setAspect(m_window.getAspectRatio());
        auto camUBO   = cam->toUBO(m_window.getTime());
        auto sceneUBO = buildSceneUBOWithShadows(scene);
        std::memcpy(m_cameraUBOs[frameIdx].mapped, &camUBO,   sizeof(CameraUBO));
        std::memcpy(m_sceneUBOs[frameIdx].mapped,  &sceneUBO, sizeof(SceneUBO));
    }

    updateShadowData(scene, frameIdx);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi), "Begin command buffer");

    recordShadowPass(cmd, scene);

    VkRenderPassBeginInfo rpBI{};
    rpBI.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBI.renderArea.extent = m_swapchain->extent();

    std::array<VkClearValue, 2> clears{};
    clears[0].color        = {{ m_settings.clearColor.r, m_settings.clearColor.g,
                                 m_settings.clearColor.b, m_settings.clearColor.a }};
    clears[1].depthStencil = { 1.f, 0 };
    rpBI.clearValueCount   = static_cast<uint32_t>(clears.size());
    rpBI.pClearValues      = clears.data();

    if (m_ppActive) {
        rpBI.renderPass  = m_offscreenRenderPass;
        rpBI.framebuffer = m_offscreenFramebuffer;
    } else {
        rpBI.renderPass  = m_swapchain->renderPass();
        rpBI.framebuffer = m_swapchain->framebuffer(imageIdx);
    }

    // ── Build visible mesh list into scratch buffer (no heap alloc after warm-up)
    m_visibleScratch.clear();
    if (m_settings.frustumCulling) {
        m_visibleScratch = scene.visibleMeshes(m_threadPool.get());
    } else {
        m_visibleScratch.reserve(scene.meshes().size());
        for (auto& m : scene.meshes()) m_visibleScratch.push_back(m.get());
    }

    m_stats.culledObjects = static_cast<uint32_t>(scene.meshes().size()) -
                            static_cast<uint32_t>(m_visibleScratch.size());

    // Pre-warm all GPU data and pipelines on the main thread
    for (auto* mesh : m_visibleScratch) {
        for (auto& sub : mesh->subMeshes()) {
            auto* mat = sub.material.get();
            if (!mat) continue;
            if (auto* pbr = dynamic_cast<PBRMaterial*>(mat)) pbr->updateTextureFlags();
            auto& data = getOrCreateMeshData(mesh, mat);
            updateMaterialDescriptors(data, mat, frameIdx);
            auto matLayout = getOrCreateMatDescLayout(mat->shaderName(), MAX_TEXTURES_PER_MAT);
            PipelineSettings ps = mat->pipelineSettings;
            if (m_settings.wireframe) ps.polygonMode = PolygonMode::Wireframe;
            getOrCreatePipeline(mat->shaderName(), ps, matLayout);
        }
    }

    // ── Build sorted draw list into scratch buffer
    m_drawListScratch.clear();
    buildDrawList(m_visibleScratch, frameIdx, m_drawListScratch);

    // Populate totalVertices stat
    m_stats.totalVertices = 0;
    for (auto& dc : m_drawListScratch) {
        const auto& sub = dc.mesh->subMeshes()[dc.subIndex];
        m_stats.totalVertices += sub.indexCount;
    }

    uint32_t drawCalls  = 0;
    uint32_t workerCount = m_threadPool->threadCount();
    uint32_t drawCount   = static_cast<uint32_t>(m_drawListScratch.size());

    const bool useSecondary = (workerCount > 1 && drawCount > 0);
    vkCmdBeginRenderPass(cmd, &rpBI,
        useSecondary ? VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS
                     : VK_SUBPASS_CONTENTS_INLINE);

    if (useSecondary) {
        uint32_t batchSize = std::max(1u, (drawCount + workerCount - 1) / workerCount);
        std::vector<std::future<VkCommandBuffer>> futures;
        futures.reserve(workerCount);

        for (uint32_t wIdx = 0; wIdx < workerCount; ++wIdx) {
            uint32_t start = wIdx * batchSize;
            if (start >= drawCount) break;
            uint32_t end = std::min(start + batchSize, drawCount);
            std::vector<DrawCall> batch(m_drawListScratch.begin() + start,
                                        m_drawListScratch.begin() + end);

            futures.push_back(m_threadPool->submit([this, batch = std::move(batch),
                                                     frameIdx, wIdx]() mutable {
                return recordSecondaryBatch(batch, frameIdx, wIdx);
            }));
        }

        std::vector<VkCommandBuffer> secondaryCmds;
        secondaryCmds.reserve(futures.size());
        for (auto& f : futures) {
            auto secCmd = f.get();
            if (secCmd != VK_NULL_HANDLE) secondaryCmds.push_back(secCmd);
        }
        if (!secondaryCmds.empty())
            vkCmdExecuteCommands(cmd, static_cast<uint32_t>(secondaryCmds.size()),
                                 secondaryCmds.data());

        drawCalls = drawCount;
    } else {
        VkViewport vp{};
        vp.width    = static_cast<float>(m_swapchain->extent().width);
        vp.height   = static_cast<float>(m_swapchain->extent().height);
        vp.minDepth = 0.f; vp.maxDepth = 1.f;
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{ {0,0}, m_swapchain->extent() };
        vkCmdSetScissor(cmd, 0, 1, &sc);

        for (auto& dc : m_drawListScratch)
            drawMeshSubMesh(cmd, *dc.mesh, dc.subIndex, frameIdx, drawCalls);
    }

    vkCmdEndRenderPass(cmd);
    m_stats.drawCalls = drawCalls;

    if (m_ppActive) recordPPPass(cmd, imageIdx, frameIdx);

    VK_CHECK(vkEndCommandBuffer(cmd), "End command buffer");
}

// ─── Resize ───────────────────────────────────────────────────────────────────
void Renderer::handleResize() {
    auto [w, h] = m_window.getFramebufferSize();
    while (w == 0 || h == 0) {
        glfwWaitEvents();
        auto [nw, nh] = m_window.getFramebufferSize();
        w = nw; h = nh;
    }
    m_ctx->waitIdle();
    destroyPipelines();

    if (m_ppActive) {
        shutdownPostProcess();
        m_ppActive = true;
        m_swapchain->recreate(w, h);
        initPostProcess();
    } else {
        m_swapchain->recreate(w, h);
        m_sceneRenderPass = m_swapchain->renderPass();
    }
}

// ─── Shadow resources ─────────────────────────────────────────────────────────
void Renderer::createShadowResources() {
    VkFormat depthFmt = VK_FORMAT_D32_SFLOAT;
    {
        VkFormatProperties p;
        vkGetPhysicalDeviceFormatProperties(m_ctx->physicalDevice(), depthFmt, &p);
        if (!(p.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
            depthFmt = m_ctx->findDepthFormat();
    }

    m_shadow.depthArray = m_ctx->createImage(
        SHADOW_MAP_RESOLUTION, SHADOW_MAP_RESOLUTION,
        1, VK_SAMPLE_COUNT_1_BIT, depthFmt,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        MAX_SHADOW_MAPS);

    {
        VkImageViewCreateInfo ci{};
        ci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image    = m_shadow.depthArray.image;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        ci.format   = depthFmt;
        ci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, MAX_SHADOW_MAPS };
        VK_CHECK(vkCreateImageView(m_ctx->device(), &ci, nullptr, &m_shadow.arrayView), "Shadow array view");
    }

    for (uint32_t i = 0; i < MAX_SHADOW_MAPS; ++i) {
        VkImageViewCreateInfo ci{};
        ci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image    = m_shadow.depthArray.image;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format   = depthFmt;
        ci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, i, 1 };
        VK_CHECK(vkCreateImageView(m_ctx->device(), &ci, nullptr, &m_shadow.layerViews[i]), "Shadow layer view");
    }

    {
        VkSamplerCreateInfo ci{};
        ci.sType         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ci.magFilter     = VK_FILTER_LINEAR;
        ci.minFilter     = VK_FILTER_LINEAR;
        ci.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        ci.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        ci.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        ci.borderColor   = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        ci.compareEnable = VK_TRUE;
        ci.compareOp     = VK_COMPARE_OP_LESS;
        ci.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        VK_CHECK(vkCreateSampler(m_ctx->device(), &ci, nullptr, &m_shadow.sampler), "Shadow sampler");
    }

    {
        VkAttachmentDescription att{};
        att.format         = depthFmt;
        att.samples        = VK_SAMPLE_COUNT_1_BIT;
        att.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;   // LOAD_OP_CLEAR — prior content discarded
        att.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference ref{ 0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
        VkSubpassDescription sub{};
        sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.pDepthStencilAttachment = &ref;

        VkSubpassDependency dep{};
        dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass    = 0;
        dep.srcStageMask  = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dep.dstStageMask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dep.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

        VkRenderPassCreateInfo ci{};
        ci.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        ci.attachmentCount = 1;
        ci.pAttachments    = &att;
        ci.subpassCount    = 1;
        ci.pSubpasses      = &sub;
        ci.dependencyCount = 1;
        ci.pDependencies   = &dep;
        VK_CHECK(vkCreateRenderPass(m_ctx->device(), &ci, nullptr, &m_shadow.renderPass), "Shadow render pass");
    }

    for (uint32_t i = 0; i < MAX_SHADOW_MAPS; ++i) {
        VkFramebufferCreateInfo ci{};
        ci.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        ci.renderPass      = m_shadow.renderPass;
        ci.attachmentCount = 1;
        ci.pAttachments    = &m_shadow.layerViews[i];
        ci.width           = SHADOW_MAP_RESOLUTION;
        ci.height          = SHADOW_MAP_RESOLUTION;
        ci.layers          = 1;
        VK_CHECK(vkCreateFramebuffer(m_ctx->device(), &ci, nullptr, &m_shadow.framebuffers[i]), "Shadow FB");
    }

    {
        VkPushConstantRange pcRange{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPushConstant) };
        VkPipelineLayoutCreateInfo ci{};
        ci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges    = &pcRange;
        VK_CHECK(vkCreatePipelineLayout(m_ctx->device(), &ci, nullptr, &m_shadow.pipelineLayout), "Shadow PL");
    }

    {
        auto vertMod = createShaderModule(m_settings.shaderDir / "shadow.vert.spv");

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stage.module = vertMod;
        stage.pName  = "main";

        auto bindDesc = Vertex::getBindingDescription();
        auto attrDesc = Vertex::getAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount   = 1;
        vi.pVertexBindingDescriptions      = &bindDesc;
        vi.vertexAttributeDescriptionCount = 1;
        vi.pVertexAttributeDescriptions    = attrDesc.data();

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport vp{}; VkRect2D sc{};
        VkPipelineViewportStateCreateInfo vpState{};
        vpState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vpState.viewportCount = 1; vpState.pViewports = &vp;
        vpState.scissorCount  = 1; vpState.pScissors  = &sc;

        VkPipelineRasterizationStateCreateInfo rast{};
        rast.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rast.polygonMode             = VK_POLYGON_MODE_FILL;
        rast.cullMode                = VK_CULL_MODE_BACK_BIT;
        rast.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rast.depthBiasEnable         = VK_TRUE;
        rast.depthBiasConstantFactor = 2.0f;
        rast.depthBiasSlopeFactor    = 3.0f;
        rast.lineWidth               = 1.f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp   = VK_COMPARE_OP_LESS;

        std::array<VkDynamicState, 2> dynStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
        dyn.pDynamicStates    = dynStates.data();

        VkGraphicsPipelineCreateInfo ci{};
        ci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        ci.stageCount          = 1;
        ci.pStages             = &stage;
        ci.pVertexInputState   = &vi;
        ci.pInputAssemblyState = &ia;
        ci.pViewportState      = &vpState;
        ci.pRasterizationState = &rast;
        ci.pMultisampleState   = &ms;
        ci.pDepthStencilState  = &ds;
        ci.pDynamicState       = &dyn;
        ci.layout              = m_shadow.pipelineLayout;
        ci.renderPass          = m_shadow.renderPass;
        VK_CHECK(vkCreateGraphicsPipelines(m_ctx->device(), m_pipelineCache_vk, 1, &ci, nullptr, &m_shadow.pipeline),
                 "Shadow pipeline");
        vkDestroyShaderModule(m_ctx->device(), vertMod, nullptr);
    }

    for (auto& ubo : m_shadow.ubos)
        ubo = m_ctx->createBuffer(sizeof(ShadowUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    {
        auto cmd = m_ctx->beginSingleTimeCommands();
        VkImageMemoryBarrier bar{};
        bar.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        bar.srcAccessMask       = 0;
        bar.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        bar.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        bar.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.image               = m_shadow.depthArray.image;
        bar.subresourceRange    = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, MAX_SHADOW_MAPS };
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &bar);
        m_ctx->endSingleTimeCommands(cmd);
    }
}

void Renderer::destroyShadowResources() {
    auto dev = m_ctx->device();
    if (m_shadow.pipeline)       { vkDestroyPipeline      (dev, m_shadow.pipeline,       nullptr); m_shadow.pipeline       = VK_NULL_HANDLE; }
    if (m_shadow.pipelineLayout) { vkDestroyPipelineLayout(dev, m_shadow.pipelineLayout, nullptr); m_shadow.pipelineLayout = VK_NULL_HANDLE; }
    for (auto& fb : m_shadow.framebuffers) if (fb) { vkDestroyFramebuffer(dev, fb, nullptr); fb = VK_NULL_HANDLE; }
    if (m_shadow.renderPass)     { vkDestroyRenderPass    (dev, m_shadow.renderPass,     nullptr); m_shadow.renderPass     = VK_NULL_HANDLE; }
    if (m_shadow.sampler)        { vkDestroySampler       (dev, m_shadow.sampler,        nullptr); m_shadow.sampler        = VK_NULL_HANDLE; }
    for (auto& v : m_shadow.layerViews) if (v) { vkDestroyImageView(dev, v, nullptr); v = VK_NULL_HANDLE; }
    if (m_shadow.arrayView)      { vkDestroyImageView     (dev, m_shadow.arrayView,      nullptr); m_shadow.arrayView      = VK_NULL_HANDLE; }
    m_ctx->destroyImage(m_shadow.depthArray);
    for (auto& ubo : m_shadow.ubos) m_ctx->destroyBuffer(ubo);
    m_shadow.count = 0;
}

void Renderer::updateShadowData(Scene& scene, uint32_t frameIdx) {
    m_shadow.count = 0;
    for (auto& light : scene.lights()) {
        if (!light->castShadows() || m_shadow.count >= (int)MAX_SHADOW_MAPS) continue;
        int   idx = m_shadow.count++;
        LightData ld = light->toLightData();
        int   type   = (int)ld.position.w;
        Vec3  up;
        Mat4  view, proj;

        if (type == 1) {
            Vec3 dir = glm::normalize(Vec3(ld.direction));
            up = std::abs(dir.y) > 0.99f ? Vec3(1, 0, 0) : Vec3(0, 1, 0);

            // Shadow frustum constants — tune to your scene scale.
            // shadowRange = half-size of the ortho box in world units.
            // pullback    = how far behind the scene the shadow camera sits.
            // shadowFar   = depth range (must be > pullback + scene depth).
            const float shadowRange = 60.f;
            const float pullback    = 150.f;
            const float shadowNear  = 1.f;
            const float shadowFar   = 400.f;

            // --- Step 1: read current camera world position -------------------------
            // (filled into the CameraUBO earlier in recordCommandBuffer)
            Vec3 camPos = Vec3(0.f);
            if (m_cameraUBOs[frameIdx].mapped) {
                const auto* c = reinterpret_cast<const CameraUBO*>(m_cameraUBOs[frameIdx].mapped);
                camPos = Vec3(c->position);
            }

            // --- Step 2: texel-snap the shadow centre to prevent shadow swimming ---
            // Build a "reference" view matrix (centred at world origin) solely for
            // snapping. We never render with it — we only use it to measure light-
            // space X/Y so we can round to texel boundaries.
            Mat4 snapView = glm::lookAt(-dir * pullback, Vec3(0.f), up);

            // Project the camera's world position into that reference light-space.
            Vec3 camLS = Vec3(snapView * Vec4(camPos, 1.f));

            // The shadow ortho box covers 2*shadowRange world units mapped to
            // SHADOW_MAP_RESOLUTION texels, so one texel = 2*range / resolution.
            float texelSize = (2.f * shadowRange) / static_cast<float>(SHADOW_MAP_RESOLUTION);

            // Snap X and Y independently; leave Z free (depth is not snapped).
            camLS.x = std::floor(camLS.x / texelSize) * texelSize;
            camLS.y = std::floor(camLS.y / texelSize) * texelSize;

            // Unproject snapped position back to world space.
            Vec3 snappedCenter = Vec3(glm::inverse(snapView) * Vec4(camLS, 1.f));

            // --- Step 3: build final view + ortho from the snapped centre ----------
            Vec3 shadowEye = snappedCenter - dir * pullback;
            view = glm::lookAt(shadowEye, snappedCenter, up);
            proj = glm::ortho(-shadowRange, shadowRange,
                              -shadowRange, shadowRange,
                               shadowNear,  shadowFar);
        } else if (type == 2) {
            Vec3 pos = Vec3(ld.position);
            Vec3 dir = glm::normalize(Vec3(ld.direction));
            up   = std::abs(dir.y) > 0.99f ? Vec3(1,0,0) : Vec3(0,1,0);
            view = glm::lookAt(pos, pos + dir, up);
            float fov = ld.params.x * 2.f + glm::radians(5.f);
            proj = glm::perspective(fov, 1.f, 0.1f, ld.params.y);
        } else {
            m_shadow.count--;
            continue;
        }
        proj[1][1] *= -1.f;
        m_shadow.lightSpaces[idx] = proj * view;
        m_shadow.casters[idx]     = light.get();
        m_shadow.frustums[idx].extract(m_shadow.lightSpaces[idx]);
    }

    ShadowUBO ubo{};
    ubo.count = m_shadow.count;
    for (int i = 0; i < m_shadow.count; ++i) {
        ubo.lightSpace[i] = m_shadow.lightSpaces[i];
        ubo.params[i]     = Vec4(0.002f, 0.005f, 0.f, 0.f);
    }
    std::memcpy(m_shadow.ubos[frameIdx].mapped, &ubo, sizeof(ShadowUBO));
}

SceneUBO Renderer::buildSceneUBOWithShadows(Scene& scene) const {
    SceneUBO ubo = scene.buildSceneUBO();
    ubo.useLinearOutput = m_ppActive ? 1 : 0;

    for (int i = 0; i < m_shadow.count; ++i) {
        for (int j = 0; j < ubo.lightCount; ++j) {
            if (scene.lights()[j].get() == m_shadow.casters[i]) {
                ubo.lights[j].params.w = static_cast<float>(i);
                break;
            }
        }
    }
    return ubo;
}

void Renderer::recordShadowPass(VkCommandBuffer cmd, Scene& scene) {
    if (m_shadow.count == 0) return;

    {
        VkImageMemoryBarrier bar{};
        bar.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        bar.srcAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        bar.dstAccessMask       = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        bar.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        bar.newLayout           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.image               = m_shadow.depthArray.image;
        bar.subresourceRange    = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, MAX_SHADOW_MAPS };
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            0, 0, nullptr, 0, nullptr, 1, &bar);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadow.pipeline);

    VkViewport vp{ 0, 0, (float)SHADOW_MAP_RESOLUTION, (float)SHADOW_MAP_RESOLUTION, 0.f, 1.f };
    VkRect2D   sc{ {0, 0}, {SHADOW_MAP_RESOLUTION, SHADOW_MAP_RESOLUTION} };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor (cmd, 0, 1, &sc);

    for (int i = 0; i < m_shadow.count; ++i) {
        VkClearValue clearVal;
        clearVal.depthStencil = { 1.0f, 0 };
        VkRenderPassBeginInfo rpBI{};
        rpBI.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBI.renderPass        = m_shadow.renderPass;
        rpBI.framebuffer       = m_shadow.framebuffers[i];
        rpBI.renderArea.extent = { SHADOW_MAP_RESOLUTION, SHADOW_MAP_RESOLUTION };
        rpBI.clearValueCount   = 1;
        rpBI.pClearValues      = &clearVal;
        vkCmdBeginRenderPass(cmd, &rpBI, VK_SUBPASS_CONTENTS_INLINE);

        ShadowPushConstant pc;
        pc.lightSpace = m_shadow.lightSpaces[i];

        for (auto& mesh : scene.meshes()) {
            if (!mesh->gpuReady || mesh->vertices().empty()) continue;
            if (!m_shadow.frustums[i].containsAABB(mesh->worldBounds())) continue;

            pc.model = mesh->modelMatrix();
            vkCmdPushConstants(cmd, m_shadow.pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPushConstant), &pc);
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &mesh->vertexBuffer.buffer, &offset);
            if (mesh->indexBuffer.buffer != VK_NULL_HANDLE) {
                vkCmdBindIndexBuffer(cmd, mesh->indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
                for (auto& sub : mesh->subMeshes())
                    vkCmdDrawIndexed(cmd, sub.indexCount, 1, sub.indexOffset, 0, 0);
            } else {
                vkCmdDraw(cmd, static_cast<uint32_t>(mesh->vertices().size()), 1, 0, 0);
            }
        }

        vkCmdEndRenderPass(cmd);
    }

    {
        VkImageMemoryBarrier bar{};
        bar.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        bar.srcAccessMask       = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        bar.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        bar.oldLayout           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        bar.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.image               = m_shadow.depthArray.image;
        bar.subresourceRange    = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, MAX_SHADOW_MAPS };
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &bar);
    }
}

// ─── Worker command pools ─────────────────────────────────────────────────────
void Renderer::createWorkerCommandPools(uint32_t workerCount) {
    m_workerCmdPools.resize(workerCount);
    m_workerCmdBuffers.resize(workerCount);

    for (uint32_t wIdx = 0; wIdx < workerCount; ++wIdx) {
        for (uint32_t fIdx = 0; fIdx < MAX_FRAMES_IN_FLIGHT; ++fIdx) {
            VkCommandPoolCreateInfo ci{};
            ci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            ci.queueFamilyIndex = *m_ctx->queueFamilies().graphics;
            ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            VK_CHECK(vkCreateCommandPool(m_ctx->device(), &ci, nullptr,
                     &m_workerCmdPools[wIdx][fIdx]), "Worker command pool");

            VkCommandBufferAllocateInfo ai{};
            ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool        = m_workerCmdPools[wIdx][fIdx];
            ai.level              = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
            ai.commandBufferCount = 1;
            VK_CHECK(vkAllocateCommandBuffers(m_ctx->device(), &ai,
                     &m_workerCmdBuffers[wIdx][fIdx]), "Worker command buffer");
        }
    }
}

void Renderer::destroyWorkerCommandPools() {
    for (auto& pools : m_workerCmdPools)
        for (auto& pool : pools)
            if (pool) { vkDestroyCommandPool(m_ctx->device(), pool, nullptr); pool = VK_NULL_HANDLE; }
    m_workerCmdPools.clear();
    m_workerCmdBuffers.clear();
}

// ─── Build and sort draw list ─────────────────────────────────────────────────
void Renderer::buildDrawList(const std::vector<Mesh*>& meshes,
                              uint32_t /*frameIdx*/,
                              std::vector<DrawCall>& out)
{
    out.reserve(out.size() + meshes.size() * 2);

    for (uint32_t mId = 0; mId < static_cast<uint32_t>(meshes.size()); ++mId) {
        auto* mesh = meshes[mId];
        if (!mesh->gpuReady || mesh->vertices().empty()) continue;

        for (uint32_t sIdx = 0; sIdx < static_cast<uint32_t>(mesh->subMeshes().size()); ++sIdx) {
            const auto& sub = mesh->subMeshes()[sIdx];
            auto* mat = sub.material.get();
            if (!mat) continue;

            uint64_t h = std::hash<std::string_view>{}(mat->shaderName());
            h ^= static_cast<uint64_t>(mat->pipelineSettings.alphaBlend)  << 1;
            h ^= static_cast<uint64_t>(mat->pipelineSettings.cullMode)     << 2;
            h ^= static_cast<uint64_t>(mat->pipelineSettings.polygonMode)  << 4;
            h ^= static_cast<uint64_t>(m_settings.wireframe)               << 6;

            DrawCall dc;
            dc.key.pipelineHash = h;
            dc.key.meshId       = mId;
            dc.mesh             = mesh;
            dc.subIndex         = sIdx;
            out.push_back(dc);
        }
    }

    std::sort(out.begin(), out.end(), [](const DrawCall& a, const DrawCall& b) {
        return a.key < b.key;
    });
}

// ─── Record one submesh draw ──────────────────────────────────────────────────
void Renderer::drawMeshSubMesh(VkCommandBuffer cmd, Mesh& mesh, uint32_t subIdx,
                                uint32_t frameIdx, uint32_t& drawCalls)
{
    if (!mesh.gpuReady || mesh.vertices().empty()) return;
    const auto& sub = mesh.subMeshes()[subIdx];
    auto* mat = sub.material.get();
    if (!mat) return;

    auto matLayout = getOrCreateMatDescLayout(mat->shaderName(), MAX_TEXTURES_PER_MAT);
    PipelineSettings ps = mat->pipelineSettings;
    if (m_settings.wireframe) ps.polygonMode = PolygonMode::Wireframe;

    auto& pipeline = getOrCreatePipeline(mat->shaderName(), ps, matLayout);
    auto& data     = m_meshData.at(&mesh);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.pipeline);
    std::array<VkDescriptorSet, 2> sets = { m_globalSets[frameIdx], data.matDescSets[frameIdx] };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0,
                             static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);

    ModelPushConstant pc;
    pc.model        = mesh.modelMatrix();
    pc.normalMatrix = mesh.normalMatrix();
    vkCmdPushConstants(cmd, pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT,
                        0, sizeof(ModelPushConstant), &pc);

    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertexBuffer.buffer, &offset);
    vkCmdBindIndexBuffer(cmd, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, sub.indexCount, 1, sub.indexOffset, 0, 0);
    ++drawCalls;
}

// ─── Record secondary command buffer for a batch ─────────────────────────────
VkCommandBuffer Renderer::recordSecondaryBatch(const std::vector<DrawCall>& batch,
                                                uint32_t frameIdx, uint32_t workerIdx)
{
    if (batch.empty() || workerIdx >= m_workerCmdPools.size()) return VK_NULL_HANDLE;

    VkCommandBuffer cmd = m_workerCmdBuffers[workerIdx][frameIdx];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferInheritanceInfo inherit{};
    inherit.sType      = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
    inherit.renderPass = m_ppActive ? m_offscreenRenderPass : m_swapchain->renderPass();
    inherit.subpass    = 0;

    VkCommandBufferBeginInfo bi{};
    bi.sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags            = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT |
                          VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    bi.pInheritanceInfo = &inherit;
    vkBeginCommandBuffer(cmd, &bi);

    VkViewport vp{};
    vp.width    = static_cast<float>(m_swapchain->extent().width);
    vp.height   = static_cast<float>(m_swapchain->extent().height);
    vp.minDepth = 0.f; vp.maxDepth = 1.f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {0,0}, m_swapchain->extent() };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    uint32_t dummy = 0;
    for (const auto& dc : batch)
        drawMeshSubMesh(cmd, *dc.mesh, dc.subIndex, frameIdx, dummy);

    vkEndCommandBuffer(cmd);
    return cmd;
}

} // namespace vkgfx
