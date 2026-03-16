// src/renderer.cpp
// Deferred renderer: G-buffer → lighting (IBL+sun+points) → tonemap → present.
// Every VkStruct zero-initialised.  Shutdown calls vkDeviceWaitIdle first.

#include <vkgfx/renderer.h>
#include <vkgfx/window.h>
#include <vkgfx/context.h>
#include <vkgfx/swapchain.h>
#include <vkgfx/scene.h>
#include <vkgfx/material.h>

// VMA is included only in context.cpp where VMA_IMPLEMENTATION is defined.
// Here we only need VmaAllocation through the AllocatedBuffer/Image types.
#include <vk_mem_alloc.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <fstream>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <array>
#include <cstring>
#include <functional>

namespace vkgfx {

// ── Helpers ───────────────────────────────────────────────────────────────────

static VkShaderModule loadSPV(VkDevice device, const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("[vkgfx] Shader not found: " + path);
    size_t sz = static_cast<size_t>(f.tellg());
    std::vector<char> buf(sz);
    f.seekg(0); f.read(buf.data(), static_cast<std::streamsize>(sz));

    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = sz;
    ci.pCode    = reinterpret_cast<const uint32_t*>(buf.data());

    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("[vkgfx] vkCreateShaderModule failed: " + path);
    return mod;
}

// ── Construction ──────────────────────────────────────────────────────────────

Renderer::Renderer(Window& window, RendererConfig cfg)
    : m_cfg(std::move(cfg))
{
#ifdef VKGFX_ENABLE_VALIDATION
    m_cfg.validationLayers = true;
#endif

    ContextConfig cc;
    cc.validation = m_cfg.validationLayers;
    m_ctx = std::make_unique<Context>(cc);

    VkSurfaceKHR surface = window.createSurface(m_ctx->instance());
    m_ctx->setSurface(surface);

    SwapchainConfig sc;
    sc.vsync = m_cfg.vsync;
    m_swapchain = std::make_unique<Swapchain>(*m_ctx, window, sc);

    m_ibl = std::make_unique<IBLSystem>(*m_ctx);
    m_ibl->setShaderDir(m_cfg.shaderDir);

    validateAssets();
    initDescriptorPools();
    initGBuffer();
    initLightingPass();
    initTonemapPass();
    initPerFrameResources();

    if (m_cfg.ibl.enabled)
        initIBL();

    m_initialized = true;
}

Renderer::~Renderer() {
    if (m_initialized) shutdown();
}

// ── validateAssets ────────────────────────────────────────────────────────────

void Renderer::validateAssets() {
    namespace fs = std::filesystem;

    // Resolve shaderDir: try the configured value first, then common locations
    // relative to CWD and relative to typical VS output dirs.
    // This handles running from project root (VS default), build/Debug/, etc.
    const std::string probeShader = "gbuffer.vert.spv";
    std::vector<std::string> shaderCandidates = {
        m_cfg.shaderDir,           // as configured, e.g. "shaders"
        "shaders",
        "../shaders",              // one level up from CWD
        "../../shaders",
        "build/shaders",           // from project root pointing into build
        "build/Debug/shaders",
        "build/Release/shaders",
        "../build/shaders",
        "../build/Debug/shaders",
        "../build/Release/shaders",
    };

    bool shaderDirFound = false;
    for (auto& candidate : shaderCandidates) {
        if (fs::exists(candidate + "/" + probeShader)) {
            if (candidate != m_cfg.shaderDir) {
                std::cout << "[vkgfx] Resolved shaderDir: " << candidate << "\n";
                m_cfg.shaderDir = candidate;
                m_ibl->setShaderDir(candidate); // keep IBL in sync
            }
            shaderDirFound = true;
            break;
        }
    }

    if (!shaderDirFound) {
        std::string tried;
        for (auto& c : shaderCandidates) tried += "\n  " + c;
        throw std::runtime_error(
            "[vkgfx] Shader SPV files not found. Searched:" + tried +
            "\n  Run cmake --build to compile shaders first.");
    }

    // Verify all required SPVs are present in the resolved dir
    for (const char* name : {"gbuffer.vert.spv","gbuffer.frag.spv",
                              "lighting.vert.spv","lighting.frag.spv",
                              "tonemap.vert.spv","tonemap.frag.spv"}) {
        std::string p = m_cfg.shaderDir + "/" + name;
        if (!fs::exists(p))
            throw std::runtime_error("[vkgfx] Required shader missing: " + p);
    }

    // IBL HDR path — warn only, IBL is disabled gracefully if absent
    if (m_cfg.ibl.enabled && !m_cfg.ibl.hdrPath.empty()) {
        if (!fs::exists(m_cfg.ibl.hdrPath)) {
            std::cerr << "[vkgfx] sky.hdr not found: " << m_cfg.ibl.hdrPath
                      << " — IBL disabled\n";
            m_cfg.ibl.enabled = false;
        }
    }
}

// ── initDescriptorPools ───────────────────────────────────────────────────────

void Renderer::initDescriptorPools() {
    // Material set layout: set=1 in G-buffer pass
    // bindings: 0=albedoTex, 1=normalTex, 2=rmaTex, 3=PBRParams UBO
    {
        VkDescriptorSetLayoutBinding mb[4]{};
        for (uint32_t i = 0; i < 3; ++i) {
            mb[i].binding        = i;
            mb[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            mb[i].descriptorCount= 1;
            mb[i].stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        mb[3].binding        = 3;
        mb[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        mb[3].descriptorCount= 1;
        mb[3].stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 4;
        lci.pBindings    = mb;
        vkCreateDescriptorSetLayout(m_ctx->device(), &lci, nullptr, &m_materialSetLayout);
    }

    VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         128},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 128},
    };
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 512;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes    = sizes;

    m_descriptorPool = makeDescriptorPool(m_ctx->device(), sizes, 2, 512);
}

// ── initGBuffer ───────────────────────────────────────────────────────────────

void Renderer::initGBuffer() {
    VkExtent2D ext = m_swapchain->extent();

    const VkFormat formats[4] = {
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R8G8B8A8_UNORM,
        m_ctx->findDepthFormat()
    };

    const VkImageUsageFlags colorUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const VkImageUsageFlags depthUsage =
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    for (int i = 0; i < 3; ++i) {
        m_gbuffer[i] = m_ctx->allocateImage(ext, formats[i], colorUsage);
        m_gbuffer[i].view = m_ctx->createImageView(
            m_gbuffer[i].image, formats[i], VK_IMAGE_ASPECT_COLOR_BIT);
    }
    m_gbuffer[3] = m_ctx->allocateImage(ext, formats[3], depthUsage);
    m_gbuffer[3].view = m_ctx->createImageView(
        m_gbuffer[3].image, formats[3], VK_IMAGE_ASPECT_DEPTH_BIT);

    // ── Render pass ───────────────────────────────────────────────────────────
    VkAttachmentDescription atts[4]{};
    for (int i = 0; i < 3; ++i) {
        atts[i].format         = formats[i];
        atts[i].samples        = VK_SAMPLE_COUNT_1_BIT;
        atts[i].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        atts[i].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        atts[i].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        atts[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        atts[i].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        atts[i].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; // matches descriptor write
    }
    atts[3].format         = formats[3];
    atts[3].samples        = VK_SAMPLE_COUNT_1_BIT;
    atts[3].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[3].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    atts[3].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[3].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[3].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colRefs[3]{};
    for (int i = 0; i < 3; ++i) {
        colRefs[i].attachment = static_cast<uint32_t>(i);
        colRefs[i].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }
    VkAttachmentReference depRef{3, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 3;
    subpass.pColorAttachments       = colRefs;
    subpass.pDepthStencilAttachment = &depRef;

    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL; deps[0].dstSubpass = 0;
    deps[0].srcStageMask  = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                          | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                          | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].srcSubpass    = 0; deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                          | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                          | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rpi{};
    rpi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpi.attachmentCount = 4;
    rpi.pAttachments    = atts;
    rpi.subpassCount    = 1;
    rpi.pSubpasses      = &subpass;
    rpi.dependencyCount = 2;
    rpi.pDependencies   = deps;

    if (vkCreateRenderPass(m_ctx->device(), &rpi, nullptr, &m_gbufferPass) != VK_SUCCESS)
        throw std::runtime_error("[vkgfx] G-buffer render pass creation failed");

    // ── Framebuffer ───────────────────────────────────────────────────────────
    VkImageView fbViews[4] = {
        m_gbuffer[0].view, m_gbuffer[1].view,
        m_gbuffer[2].view, m_gbuffer[3].view
    };
    VkFramebufferCreateInfo fi{};
    fi.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fi.renderPass      = m_gbufferPass;
    fi.attachmentCount = 4;
    fi.pAttachments    = fbViews;
    fi.width           = ext.width;
    fi.height          = ext.height;
    fi.layers          = 1;
    vkCreateFramebuffer(m_ctx->device(), &fi, nullptr, &m_gbufferFb);

    // ── Descriptor set layout: set=0 (scene UBO) ─────────────────────────────
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding        = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        b.descriptorCount= 1;
        b.stageFlags     = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 1;
        lci.pBindings    = &b;
        vkCreateDescriptorSetLayout(m_ctx->device(), &lci, nullptr, &m_gbufferSetLayout);
    }

    // ── Pipeline ──────────────────────────────────────────────────────────────
    VkShaderModule vert = loadSPV(m_ctx->device(), m_cfg.shaderDir + "/gbuffer.vert.spv");
    VkShaderModule frag = loadSPV(m_ctx->device(), m_cfg.shaderDir + "/gbuffer.frag.spv");

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pc.offset     = 0;
    pc.size       = sizeof(MeshPush);

    VkDescriptorSetLayout dsls[] = {m_gbufferSetLayout, m_materialSetLayout};
    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 2;
    plci.pSetLayouts            = dsls;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pc;
    vkCreatePipelineLayout(m_ctx->device(), &plci, nullptr, &m_gbufferLayout);

    auto binding = Vertex::bindingDescription();
    auto attribs = Vertex::attributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &binding;
    vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribs.size());
    vi.pVertexAttributeDescriptions    = attribs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vps{};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1; vps.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = VK_CULL_MODE_BACK_BIT;
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo dss{};
    dss.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    dss.depthTestEnable  = VK_TRUE;
    dss.depthWriteEnable = VK_TRUE;
    dss.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.colorWriteMask = 0xF;
    std::array<VkPipelineColorBlendAttachmentState, 3> cbAtts = {cbAtt, cbAtt, cbAtt};

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 3;
    cb.pAttachments    = cbAtts.data();

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dynStates;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert; stages[0].pName = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag; stages[1].pName = "main";

    VkGraphicsPipelineCreateInfo gci{};
    gci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gci.stageCount          = 2; gci.pStages             = stages;
    gci.pVertexInputState   = &vi; gci.pInputAssemblyState = &ia;
    gci.pViewportState      = &vps; gci.pRasterizationState = &rs;
    gci.pMultisampleState   = &ms; gci.pDepthStencilState  = &dss;
    gci.pColorBlendState    = &cb; gci.pDynamicState       = &dyn;
    gci.layout              = m_gbufferLayout;
    gci.renderPass          = m_gbufferPass;

    if (vkCreateGraphicsPipelines(m_ctx->device(), VK_NULL_HANDLE,
                                   1, &gci, nullptr, &m_gbufferPipeline) != VK_SUCCESS)
        throw std::runtime_error("[vkgfx] G-buffer pipeline creation failed");

    vkDestroyShaderModule(m_ctx->device(), vert, nullptr);
    vkDestroyShaderModule(m_ctx->device(), frag, nullptr);
}

// ── initLightingPass ──────────────────────────────────────────────────────────

void Renderer::initLightingPass() {
    VkExtent2D ext = m_swapchain->extent();

    // Offscreen HDR colour target
    m_hdrTarget = m_ctx->allocateImage(ext, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    m_hdrTarget.view = m_ctx->createImageView(m_hdrTarget.image,
        VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

    VkAttachmentDescription hdrAtt{};
    hdrAtt.format         = VK_FORMAT_R16G16B16A16_SFLOAT;
    hdrAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
    hdrAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    hdrAtt.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    hdrAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    hdrAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    hdrAtt.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    hdrAtt.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference hdrRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription sub{};
    sub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments    = &hdrRef;

    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL; deps[0].dstSubpass = 0;
    deps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].srcSubpass    = 0; deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rpi{};
    rpi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpi.attachmentCount = 1; rpi.pAttachments = &hdrAtt;
    rpi.subpassCount    = 1; rpi.pSubpasses   = &sub;
    rpi.dependencyCount = 2; rpi.pDependencies= deps;
    vkCreateRenderPass(m_ctx->device(), &rpi, nullptr, &m_lightingPass);

    VkFramebufferCreateInfo fi{};
    fi.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fi.renderPass      = m_lightingPass;
    fi.attachmentCount = 1; fi.pAttachments = &m_hdrTarget.view;
    fi.width = ext.width; fi.height = ext.height; fi.layers = 1;
    vkCreateFramebuffer(m_ctx->device(), &fi, nullptr, &m_lightingFb);

    // Descriptor set layouts ──────────────────────────────────────────────────
    // set=0: 4 G-buffer samplers (albedo, normal, rma, depth)
    {
        VkDescriptorSetLayoutBinding gb[4]{};
        for (uint32_t i = 0; i < 4; ++i) {
            gb[i].binding        = i;
            gb[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            gb[i].descriptorCount= 1;
            gb[i].stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 4; lci.pBindings = gb;
        vkCreateDescriptorSetLayout(m_ctx->device(), &lci, nullptr, &m_lightingSetLayout);
    }
    // set=1: scene UBO + light UBO
    {
        VkDescriptorSetLayoutBinding sb[2]{};
        sb[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        sb[1] = {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 2; lci.pBindings = sb;
        vkCreateDescriptorSetLayout(m_ctx->device(), &lci, nullptr, &m_sceneSetLayout);
    }
    // set=2: IBL samplers
    {
        VkDescriptorSetLayoutBinding ib[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            ib[i].binding        = i;
            ib[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ib[i].descriptorCount= 1;
            ib[i].stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 3; lci.pBindings = ib;
        vkCreateDescriptorSetLayout(m_ctx->device(), &lci, nullptr, &m_iblSetLayout);
    }

    VkDescriptorSetLayout lDsls[] = {m_lightingSetLayout, m_sceneSetLayout, m_iblSetLayout};
    VkPipelineLayoutCreateInfo plci{};
    plci.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 3; plci.pSetLayouts = lDsls;
    vkCreatePipelineLayout(m_ctx->device(), &plci, nullptr, &m_lightingLayout);

    // Pipeline ────────────────────────────────────────────────────────────────
    VkShaderModule vert = loadSPV(m_ctx->device(), m_cfg.shaderDir + "/lighting.vert.spv");
    VkShaderModule frag = loadSPV(m_ctx->device(), m_cfg.shaderDir + "/lighting.frag.spv");

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vps{};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1; vps.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.f;
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo dss{};
    dss.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1; cb.pAttachments = &cbAtt;
    VkDynamicState dynS[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2; dyn.pDynamicStates = dynS;

    VkPipelineShaderStageCreateInfo stg[2]{};
    stg[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stg[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;   stg[0].module = vert; stg[0].pName = "main";
    stg[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stg[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT; stg[1].module = frag; stg[1].pName = "main";

    VkGraphicsPipelineCreateInfo gci{};
    gci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gci.stageCount          = 2; gci.pStages             = stg;
    gci.pVertexInputState   = &vi; gci.pInputAssemblyState = &ia;
    gci.pViewportState      = &vps; gci.pRasterizationState = &rs;
    gci.pMultisampleState   = &ms; gci.pDepthStencilState  = &dss;
    gci.pColorBlendState    = &cb; gci.pDynamicState       = &dyn;
    gci.layout              = m_lightingLayout;
    gci.renderPass          = m_lightingPass;
    vkCreateGraphicsPipelines(m_ctx->device(), VK_NULL_HANDLE, 1, &gci, nullptr, &m_lightingPipeline);

    vkDestroyShaderModule(m_ctx->device(), vert, nullptr);
    vkDestroyShaderModule(m_ctx->device(), frag, nullptr);
}

// ── initTonemapPass ───────────────────────────────────────────────────────────

void Renderer::initTonemapPass() {
    VkDescriptorSetLayoutBinding b{};
    b.binding        = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount= 1;
    b.stageFlags     = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 1; lci.pBindings = &b;
    vkCreateDescriptorSetLayout(m_ctx->device(), &lci, nullptr, &m_tonemapSetLayout);

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; pc.offset = 0; pc.size = 8;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1; plci.pSetLayouts            = &m_tonemapSetLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges    = &pc;
    vkCreatePipelineLayout(m_ctx->device(), &plci, nullptr, &m_tonemapLayout);

    VkShaderModule vert = loadSPV(m_ctx->device(), m_cfg.shaderDir + "/tonemap.vert.spv");
    VkShaderModule frag = loadSPV(m_ctx->device(), m_cfg.shaderDir + "/tonemap.frag.spv");

    VkPipelineVertexInputStateCreateInfo   vi{}; vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo ia{}; ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO; ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo     vps{}; vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO; vps.viewportCount = 1; vps.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{}; rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO; rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.lineWidth = 1.f;
    VkPipelineMultisampleStateCreateInfo   ms{}; ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO; ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo dss{}; dss.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    VkPipelineColorBlendAttachmentState cbAtt{}; cbAtt.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo    cb{}; cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO; cb.attachmentCount = 1; cb.pAttachments = &cbAtt;
    VkDynamicState dynS[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo      dyn{}; dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO; dyn.dynamicStateCount = 2; dyn.pDynamicStates = dynS;

    VkPipelineShaderStageCreateInfo stg[2]{};
    stg[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stg[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stg[0].module = vert; stg[0].pName = "main";
    stg[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stg[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stg[1].module = frag; stg[1].pName = "main";

    VkGraphicsPipelineCreateInfo gci{};
    gci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gci.stageCount=2; gci.pStages=stg;
    gci.pVertexInputState=&vi; gci.pInputAssemblyState=&ia; gci.pViewportState=&vps;
    gci.pRasterizationState=&rs; gci.pMultisampleState=&ms; gci.pDepthStencilState=&dss;
    gci.pColorBlendState=&cb; gci.pDynamicState=&dyn;
    gci.layout     = m_tonemapLayout;
    gci.renderPass = m_swapchain->renderPass();
    vkCreateGraphicsPipelines(m_ctx->device(), VK_NULL_HANDLE, 1, &gci, nullptr, &m_tonemapPipeline);

    vkDestroyShaderModule(m_ctx->device(), vert, nullptr);
    vkDestroyShaderModule(m_ctx->device(), frag, nullptr);
}

// ── initPerFrameResources ─────────────────────────────────────────────────────

void Renderer::initPerFrameResources() {
    // Samplers
    {
        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_LINEAR;
        sci.minFilter    = VK_FILTER_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxLod       = 1.f;
        vkCreateSampler(m_ctx->device(), &sci, nullptr, &m_gbufferSampler);
        vkCreateSampler(m_ctx->device(), &sci, nullptr, &m_hdrSampler);
    }

    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = m_ctx->graphicsPool();
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
    std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> cmds{};
    vkAllocateCommandBuffers(m_ctx->device(), &ai, cmds.data());

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        auto& f          = m_frames[i];
        f.cmd            = cmds[i];
        f.imageAvailable = makeSemaphore(m_ctx->device());
        f.renderFinished = makeSemaphore(m_ctx->device());
        f.inFlight       = makeFence(m_ctx->device(), true);

        f.sceneUbo = m_ctx->allocateBuffer(sizeof(SceneUBO),  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
        f.lightUbo = m_ctx->allocateBuffer(sizeof(LightUBO),  VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);

        // Allocate descriptor sets ─────────────────────────────────────────────
        // gbuffer scene set (set=0 in G-buffer pass)
        {
            VkDescriptorSetAllocateInfo dsai{};
            dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dsai.descriptorPool     = m_descriptorPool.get();
            dsai.descriptorSetCount = 1;
            dsai.pSetLayouts        = &m_gbufferSetLayout;
            vkAllocateDescriptorSets(m_ctx->device(), &dsai, &f.gbufferSceneSet);
        }
        // lighting sets (set=0 and set=1)
        {
            VkDescriptorSetLayout layouts[] = {m_lightingSetLayout, m_sceneSetLayout};
            VkDescriptorSet       sets[2]{};
            VkDescriptorSetAllocateInfo dsai{};
            dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dsai.descriptorPool     = m_descriptorPool.get();
            dsai.descriptorSetCount = 2;
            dsai.pSetLayouts        = layouts;
            vkAllocateDescriptorSets(m_ctx->device(), &dsai, sets);
            f.lightingGbufferSet = sets[0];
            f.lightingSceneSet   = sets[1];
        }
        // tonemap set (set=0)
        {
            VkDescriptorSetAllocateInfo dsai{};
            dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dsai.descriptorPool     = m_descriptorPool.get();
            dsai.descriptorSetCount = 1;
            dsai.pSetLayouts        = &m_tonemapSetLayout;
            vkAllocateDescriptorSets(m_ctx->device(), &dsai, &f.tonemapSet);
        }
        // IBL set (set=2 in lighting pass) — always allocate so the pipeline
        // layout is always fully satisfied even when IBL is disabled.
        // initIBL() will overwrite this with real IBL descriptors if IBL is on.
        // When IBL is off, iblIntensity=0 in the shader skips the IBL calculation,
        // but the set must still be bound to a valid (even unwritten) descriptor set.
        {
            VkDescriptorSetAllocateInfo dsai{};
            dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dsai.descriptorPool     = m_descriptorPool.get();
            dsai.descriptorSetCount = 1;
            dsai.pSetLayouts        = &m_iblSetLayout;
            vkAllocateDescriptorSets(m_ctx->device(), &dsai, &f.iblSet);
        }

        // Write gbufferSceneSet: scene UBO
        {
            VkDescriptorBufferInfo bi{f.sceneUbo.buffer, 0, sizeof(SceneUBO)};
            VkWriteDescriptorSet w{};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = f.gbufferSceneSet;
            w.dstBinding      = 0;
            w.descriptorCount = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w.pBufferInfo     = &bi;
            vkUpdateDescriptorSets(m_ctx->device(), 1, &w, 0, nullptr);
        }

        // Write lightingGbufferSet: 4 G-buffer samplers
        for (uint32_t g = 0; g < 4; ++g) {
            VkDescriptorImageInfo imgInfo{};
            imgInfo.sampler     = m_gbufferSampler;
            imgInfo.imageView   = m_gbuffer[g].view;
            imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkWriteDescriptorSet w{};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = f.lightingGbufferSet;
            w.dstBinding      = g;
            w.descriptorCount = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo      = &imgInfo;
            vkUpdateDescriptorSets(m_ctx->device(), 1, &w, 0, nullptr);
        }

        // Write lightingSceneSet: scene UBO + light UBO
        {
            VkDescriptorBufferInfo scBI{f.sceneUbo.buffer, 0, sizeof(SceneUBO)};
            VkDescriptorBufferInfo liBI{f.lightUbo.buffer, 0, sizeof(LightUBO)};
            VkWriteDescriptorSet ws[2]{};
            ws[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[0].dstSet          = f.lightingSceneSet; ws[0].dstBinding = 0;
            ws[0].descriptorCount = 1; ws[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            ws[0].pBufferInfo     = &scBI;
            ws[1]                 = ws[0];
            ws[1].dstBinding      = 1; ws[1].pBufferInfo = &liBI;
            vkUpdateDescriptorSets(m_ctx->device(), 2, ws, 0, nullptr);
        }

        // Write tonemapSet: HDR target sampler
        {
            VkDescriptorImageInfo hdrInfo{};
            hdrInfo.sampler     = m_hdrSampler;
            hdrInfo.imageView   = m_hdrTarget.view;
            hdrInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkWriteDescriptorSet w{};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = f.tonemapSet; w.dstBinding = 0;
            w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo      = &hdrInfo;
            vkUpdateDescriptorSets(m_ctx->device(), 1, &w, 0, nullptr);
        }
    }
}

// ── initIBL ───────────────────────────────────────────────────────────────────

void Renderer::initIBL() {
    bool ok = m_ibl->build(m_cfg.ibl);
    if (!ok) {
        std::cerr << "[vkgfx] IBL build failed — running without IBL\n";
        m_cfg.ibl.enabled = false;
        return;
    }

    for (auto& f : m_frames) {
        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = m_descriptorPool.get();
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &m_iblSetLayout;
        vkAllocateDescriptorSets(m_ctx->device(), &dsai, &f.iblSet);

        VkDescriptorImageInfo imgs[3]{};
        imgs[0] = {m_ibl->cubeSampler(), m_ibl->irradianceView(),  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        imgs[1] = {m_ibl->cubeSampler(), m_ibl->prefilteredView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        imgs[2] = {m_ibl->brdfSampler(), m_ibl->brdfLutView(),     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

        VkWriteDescriptorSet ws[3]{};
        for (int k = 0; k < 3; ++k) {
            ws[k].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[k].dstSet          = f.iblSet; ws[k].dstBinding = static_cast<uint32_t>(k);
            ws[k].descriptorCount = 1; ws[k].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            ws[k].pImageInfo      = &imgs[k];
        }
        vkUpdateDescriptorSets(m_ctx->device(), 3, ws, 0, nullptr);
    }
}

// ── applyConfig / rebuild ─────────────────────────────────────────────────────

void Renderer::applyConfig(const RendererConfig& cfg) {
    bool iblChanged = (cfg.ibl.enabled != m_cfg.ibl.enabled)
                   || (cfg.ibl.hdrPath  != m_cfg.ibl.hdrPath)
                   || (cfg.ibl.envMapSize != m_cfg.ibl.envMapSize);
    m_cfg = cfg;

    if (iblChanged) {
        vkDeviceWaitIdle(m_ctx->device());
        m_ibl->destroy();
        // Re-allocate dummy IBL sets so the pipeline layout's set=2 remains valid.
        // initIBL() below will overwrite them with real data if IBL is enabled.
        for (auto& f : m_frames) {
            if (f.iblSet == VK_NULL_HANDLE) {
                VkDescriptorSetAllocateInfo dsai{};
                dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                dsai.descriptorPool     = m_descriptorPool.get();
                dsai.descriptorSetCount = 1;
                dsai.pSetLayouts        = &m_iblSetLayout;
                vkAllocateDescriptorSets(m_ctx->device(), &dsai, &f.iblSet);
            }
        }
        if (m_cfg.ibl.enabled) initIBL();
    }
    // Sun/debug changes are applied each frame through the LightUBO upload
}

void Renderer::rebuild(const RendererConfig& cfg) {
    shutdown();
    m_cfg = cfg;
    initDescriptorPools();
    initGBuffer();
    initLightingPass();
    initTonemapPass();
    initPerFrameResources();
    if (m_cfg.ibl.enabled) initIBL();
    m_initialized = true;
}

// ── render ────────────────────────────────────────────────────────────────────

void Renderer::render(Scene& scene) {
    auto& f = m_frames[m_frameIdx];

    vkWaitForFences(m_ctx->device(), 1, f.inFlight.ptr(), VK_TRUE, UINT64_MAX);

    VkResult res = m_swapchain->acquireNext(f.imageAvailable.get(), m_swapIdx);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) {
        m_swapchain->recreate();
        return;
    }

    vkResetFences(m_ctx->device(), 1, f.inFlight.ptr());

    // ── Upload SceneUBO ───────────────────────────────────────────────────────
    if (Camera* cam = scene.camera()) {
        cam->setAspect(static_cast<float>(m_swapchain->extent().width) /
                       static_cast<float>(m_swapchain->extent().height));

        SceneUBO sceneData{};
        sceneData.view      = cam->view();
        sceneData.proj      = cam->projection();
        sceneData.viewProj  = cam->viewProj();
        sceneData.cameraPos = glm::vec4(cam->position(), 1.f);
        sceneData.viewport  = {static_cast<float>(m_swapchain->extent().width),
                               static_cast<float>(m_swapchain->extent().height)};

        void* mapped = nullptr;
        vmaMapMemory(m_ctx->vma(), static_cast<VmaAllocation>(f.sceneUbo.allocation), &mapped);
        std::memcpy(mapped, &sceneData, sizeof(SceneUBO));
        vmaUnmapMemory(m_ctx->vma(), static_cast<VmaAllocation>(f.sceneUbo.allocation));
    }

    // ── Upload LightUBO ───────────────────────────────────────────────────────
    LightUBO lightData{};
    float iblInt = (m_cfg.ibl.enabled && m_ibl->isReady()) ? m_ibl->intensity() : 0.f;
    scene.fillLightUBO(lightData, iblInt, static_cast<uint32_t>(m_cfg.gbufferDebug));

    // Config overrides when no scene directional light set
    if (!scene.dirLight()) {
        lightData.sun.enabled   = m_cfg.sun.enabled ? 1u : 0u;
        lightData.sun.direction = glm::vec4(m_cfg.sun.direction[0],
                                             m_cfg.sun.direction[1],
                                             m_cfg.sun.direction[2], 0.f);
        lightData.sun.color     = glm::vec4(m_cfg.sun.color[0],
                                             m_cfg.sun.color[1],
                                             m_cfg.sun.color[2],
                                             m_cfg.sun.intensity);
    }

    void* lmapped = nullptr;
    vmaMapMemory(m_ctx->vma(), static_cast<VmaAllocation>(f.lightUbo.allocation), &lmapped);
    std::memcpy(lmapped, &lightData, sizeof(LightUBO));
    vmaUnmapMemory(m_ctx->vma(), static_cast<VmaAllocation>(f.lightUbo.allocation));

    // ── Record ────────────────────────────────────────────────────────────────
    vkResetCommandBuffer(f.cmd, 0);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(f.cmd, &bi);

    VkExtent2D ext = m_swapchain->extent();

    // G-buffer pass ────────────────────────────────────────────────────────────
    {
        VkClearValue clears[4]{};
        clears[3].depthStencil = {1.0f, 0};
        VkRenderPassBeginInfo rbi{};
        rbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rbi.renderPass      = m_gbufferPass;
        rbi.framebuffer     = m_gbufferFb;
        rbi.renderArea      = {{0, 0}, ext};
        rbi.clearValueCount = 4; rbi.pClearValues = clears;
        vkCmdBeginRenderPass(f.cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gbufferPipeline);
        VkViewport vp{0.f, 0.f, (float)ext.width, (float)ext.height, 0.f, 1.f};
        VkRect2D   sc{{0, 0}, ext};
        vkCmdSetViewport(f.cmd, 0, 1, &vp);
        vkCmdSetScissor (f.cmd, 0, 1, &sc);

        // set=0: scene UBO
        vkCmdBindDescriptorSets(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_gbufferLayout, 0, 1, &f.gbufferSceneSet, 0, nullptr);

        for (Mesh* mesh : scene.visibleMeshes()) {
            MeshPush push{};
            push.model        = mesh->modelMatrix();
            push.normalMatrix = mesh->normalMatrix();
            vkCmdPushConstants(f.cmd, m_gbufferLayout,
                VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPush), &push);

            if (mesh->material() && mesh->material()->descriptorSet() != VK_NULL_HANDLE) {
                VkDescriptorSet matSet = mesh->material()->descriptorSet();
                // set=1: material
                vkCmdBindDescriptorSets(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_gbufferLayout, 1, 1, &matSet, 0, nullptr);
            }

            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(f.cmd, 0, 1, &mesh->vertexBuffer(), offsets);
            vkCmdBindIndexBuffer  (f.cmd, mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed      (f.cmd, mesh->indexCount(), 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(f.cmd);
        // G-buffer render pass transitions all attachments to SHADER_READ_ONLY_OPTIMAL
        // via finalLayout — subpass dependency handles the visibility guarantee.
    }

    // Lighting pass ────────────────────────────────────────────────────────────
    {
        VkClearValue clear{};
        VkRenderPassBeginInfo rbi{};
        rbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rbi.renderPass      = m_lightingPass;
        rbi.framebuffer     = m_lightingFb;
        rbi.renderArea      = {{0, 0}, ext};
        rbi.clearValueCount = 1; rbi.pClearValues = &clear;
        vkCmdBeginRenderPass(f.cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_lightingPipeline);
        VkViewport vp{0.f, 0.f, (float)ext.width, (float)ext.height, 0.f, 1.f};
        VkRect2D   sc{{0, 0}, ext};
        vkCmdSetViewport(f.cmd, 0, 1, &vp);
        vkCmdSetScissor (f.cmd, 0, 1, &sc);

        // Always bind all 3 descriptor sets.
        // set=2 (IBL) is always a valid allocated set — initPerFrameResources
        // allocates a dummy one; initIBL() fills it with real data when enabled.
        // The shader reads iblIntensity from the LightUBO: 0 = skip IBL sampling.
        VkDescriptorSet dsets[3] = {
            f.lightingGbufferSet,
            f.lightingSceneSet,
            f.iblSet
        };
        vkCmdBindDescriptorSets(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_lightingLayout, 0, 3, dsets, 0, nullptr);

        vkCmdDraw(f.cmd, 3, 1, 0, 0); // fullscreen triangle
        vkCmdEndRenderPass(f.cmd);
    }

    // Tonemap pass ─────────────────────────────────────────────────────────────
    {
        VkClearValue clear{};
        VkRenderPassBeginInfo rbi{};
        rbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rbi.renderPass      = m_swapchain->renderPass();
        rbi.framebuffer     = m_swapchain->framebuffer(m_swapIdx);
        rbi.renderArea      = {{0, 0}, ext};
        rbi.clearValueCount = 1; rbi.pClearValues = &clear;
        vkCmdBeginRenderPass(f.cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_tonemapPipeline);
        VkViewport vp{0.f, 0.f, (float)ext.width, (float)ext.height, 0.f, 1.f};
        VkRect2D   sc{{0, 0}, ext};
        vkCmdSetViewport(f.cmd, 0, 1, &vp);
        vkCmdSetScissor (f.cmd, 0, 1, &sc);

        vkCmdBindDescriptorSets(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_tonemapLayout, 0, 1, &f.tonemapSet, 0, nullptr);

        struct { float exposure; float gamma; } push{1.0f, 2.2f};
        vkCmdPushConstants(f.cmd, m_tonemapLayout,
            VK_SHADER_STAGE_FRAGMENT_BIT, 0, 8, &push);

        vkCmdDraw(f.cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(f.cmd);
    }

    vkEndCommandBuffer(f.cmd);

    // ── Submit ────────────────────────────────────────────────────────────────
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount   = 1; si.pWaitSemaphores   = f.imageAvailable.ptr();
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1; si.pCommandBuffers   = &f.cmd;
    si.signalSemaphoreCount = 1; si.pSignalSemaphores = f.renderFinished.ptr();
    vkQueueSubmit(m_ctx->graphicsQ(), 1, &si, f.inFlight.get());

    res = m_swapchain->present(f.renderFinished.get(), m_swapIdx);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR)
        m_swapchain->recreate();

    m_frameIdx = (m_frameIdx + 1) % MAX_FRAMES_IN_FLIGHT;
}

// ── shutdown ──────────────────────────────────────────────────────────────────

void Renderer::shutdown() {
    if (!m_initialized) return;
    vkDeviceWaitIdle(m_ctx->device());

    for (auto& f : m_frames) {
        m_ctx->destroyBuffer(f.sceneUbo);
        m_ctx->destroyBuffer(f.lightUbo);
        // Semaphores/fences are RAII VkHandles — destroyed automatically
    }

    auto dev = m_ctx->device();
    auto td  = [&](auto& h, auto fn) { if (h != VK_NULL_HANDLE) { fn(dev, h, nullptr); h = VK_NULL_HANDLE; } };

    td(m_gbufferPipeline,  vkDestroyPipeline);
    td(m_gbufferLayout,    vkDestroyPipelineLayout);
    td(m_lightingPipeline, vkDestroyPipeline);
    td(m_lightingLayout,   vkDestroyPipelineLayout);
    td(m_tonemapPipeline,  vkDestroyPipeline);
    td(m_tonemapLayout,    vkDestroyPipelineLayout);

    td(m_gbufferPass,   vkDestroyRenderPass);
    td(m_lightingPass,  vkDestroyRenderPass);
    td(m_gbufferFb,     vkDestroyFramebuffer);
    td(m_lightingFb,    vkDestroyFramebuffer);

    td(m_gbufferSetLayout,  vkDestroyDescriptorSetLayout);
    td(m_materialSetLayout, vkDestroyDescriptorSetLayout);
    td(m_lightingSetLayout, vkDestroyDescriptorSetLayout);
    td(m_sceneSetLayout,    vkDestroyDescriptorSetLayout);
    td(m_iblSetLayout,      vkDestroyDescriptorSetLayout);
    td(m_tonemapSetLayout,  vkDestroyDescriptorSetLayout);

    td(m_hdrSampler,     vkDestroySampler);
    td(m_gbufferSampler, vkDestroySampler);

    for (auto& img : m_gbuffer) m_ctx->destroyImage(img);
    m_ctx->destroyImage(m_hdrTarget);

    m_ibl->destroy();
    m_descriptorPool.reset();

    m_initialized = false;
}

} // namespace vkgfx
