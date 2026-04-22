// src/renderer.cpp
// Deferred renderer: G-buffer → lighting (IBL+sun+points) → tonemap → present.
// Every VkStruct zero-initialised.  Shutdown calls vkDeviceWaitIdle first.

#include <vkgfx/renderer.h>
#include <vkgfx/window.h>
#include <vkgfx/context.h>
#include <vkgfx/swapchain.h>
#include <vkgfx/scene.h>
#include <vkgfx/material.h>

#ifdef VKGFX_ENABLE_PROFILING
#  include <imgui.h>
#  include <imgui_impl_glfw.h>
#  include <imgui_impl_vulkan.h>
#endif

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
        ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        ci.codeSize = sz;
        ci.pCode = reinterpret_cast<const uint32_t*>(buf.data());

        VkShaderModule mod = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
            throw std::runtime_error("[vkgfx] vkCreateShaderModule failed: " + path);
        return mod;
    }

    // ── Construction ──────────────────────────────────────────────────────────────

    Renderer::Renderer(Window& window, RendererConfig cfg)
        : m_cfg(std::move(cfg)), m_window(&window)
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

        // --- Add this line to initialize the FrameGraph ---
        m_frameGraph = std::make_unique<FrameGraph>(*m_ctx);

        validateAssets();
        initDescriptorPools();
        initDefaultTextures();  // must be before initPerFrameResources (uses fallback textures)
        initShadowPass();
        initPointShadowPass();
        initGBuffer();
        initLightingPass();
        initTonemapPass();
        initPerFrameResources();

        if (m_cfg.ibl.enabled)
            initIBL();

        // Profiler and ImGui — initialised last so all Vulkan objects are ready
        if (m_cfg.profiling.enabled) {
            m_profiler.init(*m_ctx, MAX_FRAMES_IN_FLIGHT);
            if (m_cfg.profiling.showOverlay)
                initImGui();
        }

        m_initialized = true;
    }

    Renderer::~Renderer() {
        // Always release frame graph lambdas first, even if shutdown() was not
        // called, to avoid std::function holding dangling captured references
        // while member destructors are running in reverse declaration order.
        if (m_frameGraph) m_frameGraph->reset();
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
        for (const char* name : { "gbuffer.vert.spv","gbuffer.frag.spv",
                                  "lighting.vert.spv","lighting.frag.spv",
                                  "tonemap.vert.spv","tonemap.frag.spv",
                                  "shadow.vert.spv",
                                  "point_shadow.vert.spv","point_shadow.frag.spv" }) {
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
        // Material set layout (set=1 in G-buffer pass):
        //   binding 0 = albedoTex   (sampler2D)
        //   binding 1 = normalTex   (sampler2D)
        //   binding 2 = rmaTex      (sampler2D)
        //   binding 3 = emissiveTex (sampler2D)
        //   binding 4 = PBRParams   (UBO)
        {
            VkDescriptorSetLayoutBinding mb[5]{};
            for (uint32_t i = 0; i < 4; ++i) {
                mb[i].binding = i;
                mb[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                mb[i].descriptorCount = 1;
                mb[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            }
            mb[4].binding = 4;
            mb[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            mb[4].descriptorCount = 1;
            mb[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo lci{};
            lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            lci.bindingCount = 5;
            lci.pBindings = mb;
            vkCreateDescriptorSetLayout(m_ctx->device(), &lci, nullptr, &m_materialSetLayout);
        }

        VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         256},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256},
        };
        m_descriptorPool = makeDescriptorPool(m_ctx->device(), sizes, 2, 512);
    }

    // ── initGBuffer ───────────────────────────────────────────────────────────────

    void Renderer::initGBuffer() {
        VkExtent2D ext = m_swapchain->extent();
        m_offscreenExtent = ext; // tracked so render() can detect swapchain resize

        // G-buffer layout:
        //  [0] albedo       RGBA8_UNORM
        //  [1] normal       RGBA16_SFLOAT  (world-space encoded)
        //  [2] RMA          RGBA8_UNORM    (roughness/metallic/ao)
        //  [3] emissive     RGBA16_SFLOAT
        //  [4] shadowCoord  RGBA16_SFLOAT  (light-space xyz)
        //  [5] depth        D32_SFLOAT
        const VkFormat colorFmts[5] = {
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_FORMAT_R16G16B16A16_SFLOAT,
        };
        VkFormat depthFmt = m_ctx->findDepthFormat();

        const VkImageUsageFlags colorUsage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        const VkImageUsageFlags depthUsage =
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        for (int i = 0; i < 5; ++i) {
            m_gbuffer[i] = m_ctx->allocateImage(ext, colorFmts[i], colorUsage);
            m_gbuffer[i].view = m_ctx->createImageView(
                m_gbuffer[i].image, colorFmts[i], VK_IMAGE_ASPECT_COLOR_BIT);
        }
        m_gbuffer[5] = m_ctx->allocateImage(ext, depthFmt, depthUsage);
        m_gbuffer[5].view = m_ctx->createImageView(
            m_gbuffer[5].image, depthFmt, VK_IMAGE_ASPECT_DEPTH_BIT);

        // ── Render pass — 5 colour + 1 depth ─────────────────────────────────────
        VkAttachmentDescription atts[6]{};
        for (int i = 0; i < 5; ++i) {
            atts[i].format = colorFmts[i];
            atts[i].samples = VK_SAMPLE_COUNT_1_BIT;
            atts[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            atts[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            atts[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            atts[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            atts[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            atts[i].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        atts[5].format = depthFmt;
        atts[5].samples = VK_SAMPLE_COUNT_1_BIT;
        atts[5].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        atts[5].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        atts[5].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        atts[5].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        atts[5].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        atts[5].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference colRefs[5]{};
        for (int i = 0; i < 5; ++i) {
            colRefs[i].attachment = static_cast<uint32_t>(i);
            colRefs[i].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        VkAttachmentReference depRef{ 5, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 5;
        subpass.pColorAttachments = colRefs;
        subpass.pDepthStencilAttachment = &depRef;

        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL; deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
            | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
            | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].srcSubpass = 0; deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
            | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
            | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo rpi{};
        rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount = 6;
        rpi.pAttachments = atts;
        rpi.subpassCount = 1;
        rpi.pSubpasses = &subpass;
        rpi.dependencyCount = 2;
        rpi.pDependencies = deps;

        if (vkCreateRenderPass(m_ctx->device(), &rpi, nullptr, &m_gbufferPass) != VK_SUCCESS)
            throw std::runtime_error("[vkgfx] G-buffer render pass creation failed");

        // ── Framebuffer ───────────────────────────────────────────────────────────
        VkImageView fbViews[6] = {
            m_gbuffer[0].view, m_gbuffer[1].view, m_gbuffer[2].view,
            m_gbuffer[3].view, m_gbuffer[4].view, m_gbuffer[5].view
        };
        VkFramebufferCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fi.renderPass = m_gbufferPass;
        fi.attachmentCount = 6;
        fi.pAttachments = fbViews;
        fi.width = ext.width;
        fi.height = ext.height;
        fi.layers = 1;
        vkCreateFramebuffer(m_ctx->device(), &fi, nullptr, &m_gbufferFb);

        // ── Descriptor set layout: set=0 (scene UBO, vertex stage) ───────────────
        {
            VkDescriptorSetLayoutBinding b{};
            b.binding = 0;
            b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            b.descriptorCount = 1;
            b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

            VkDescriptorSetLayoutCreateInfo lci{};
            lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            lci.bindingCount = 1;
            lci.pBindings = &b;
            vkCreateDescriptorSetLayout(m_ctx->device(), &lci, nullptr, &m_gbufferSetLayout);
        }

        // ── Pipeline ──────────────────────────────────────────────────────────────
        VkShaderModule vert = loadSPV(m_ctx->device(), m_cfg.shaderDir + "/gbuffer.vert.spv");
        VkShaderModule frag = loadSPV(m_ctx->device(), m_cfg.shaderDir + "/gbuffer.frag.spv");

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pc.offset = 0;
        pc.size = sizeof(MeshPush);

        VkDescriptorSetLayout dsls[] = { m_gbufferSetLayout, m_materialSetLayout };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 2;
        plci.pSetLayouts = dsls;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pc;
        vkCreatePipelineLayout(m_ctx->device(), &plci, nullptr, &m_gbufferLayout);

        auto binding = Vertex::bindingDescription();
        auto attribs = Vertex::attributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribs.size());
        vi.pVertexAttributeDescriptions = attribs.data();

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vps{};
        vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vps.viewportCount = 1; vps.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_BACK_BIT;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo dss{};
        dss.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        dss.depthTestEnable = VK_TRUE;
        dss.depthWriteEnable = VK_TRUE;
        dss.depthCompareOp = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState cbAtt{};
        cbAtt.colorWriteMask = 0xF;
        std::array<VkPipelineColorBlendAttachmentState, 5> cbAtts = { cbAtt,cbAtt,cbAtt,cbAtt,cbAtt };

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 5;
        cb.pAttachments = cbAtts.data();

        VkDynamicState dynStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates = dynStates;

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag; stages[1].pName = "main";

        VkGraphicsPipelineCreateInfo gci{};
        gci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gci.stageCount = 2; gci.pStages = stages;
        gci.pVertexInputState = &vi; gci.pInputAssemblyState = &ia;
        gci.pViewportState = &vps; gci.pRasterizationState = &rs;
        gci.pMultisampleState = &ms; gci.pDepthStencilState = &dss;
        gci.pColorBlendState = &cb; gci.pDynamicState = &dyn;
        gci.layout = m_gbufferLayout;
        gci.renderPass = m_gbufferPass;

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
        hdrAtt.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        hdrAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        hdrAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        hdrAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        hdrAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        hdrAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        hdrAtt.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        hdrAtt.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference hdrRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 1;
        sub.pColorAttachments = &hdrRef;

        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL; deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[1].srcSubpass = 0; deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo rpi{};
        rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount = 1; rpi.pAttachments = &hdrAtt;
        rpi.subpassCount = 1; rpi.pSubpasses = &sub;
        rpi.dependencyCount = 2; rpi.pDependencies = deps;
        vkCreateRenderPass(m_ctx->device(), &rpi, nullptr, &m_lightingPass);

        VkFramebufferCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fi.renderPass = m_lightingPass;
        fi.attachmentCount = 1; fi.pAttachments = &m_hdrTarget.view;
        fi.width = ext.width; fi.height = ext.height; fi.layers = 1;
        vkCreateFramebuffer(m_ctx->device(), &fi, nullptr, &m_lightingFb);

        // Descriptor set layouts ──────────────────────────────────────────────────
        // set=0: 8 G-buffer bindings
        //   0=albedo  1=normal  2=rma  3=emissive  4=shadowCoord  5=depth
        //   6=shadowMap (sampler2DShadow, directional PCF)
        //   7=pointShadowMap (samplerCubeShadow, omnidirectional PCF)
        // All use COMBINED_IMAGE_SAMPLER in Vulkan; type is expressed in the shader.
        {
            VkDescriptorSetLayoutBinding gb[8]{};
            for (uint32_t i = 0; i < 8; ++i) {
                gb[i].binding = i;
                gb[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                gb[i].descriptorCount = 1;
                gb[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            }
            VkDescriptorSetLayoutCreateInfo lci{};
            lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            lci.bindingCount = 8; lci.pBindings = gb;
            vkCreateDescriptorSetLayout(m_ctx->device(), &lci, nullptr, &m_lightingSetLayout);
        }
        // set=1: scene UBO + light UBO
        {
            VkDescriptorSetLayoutBinding sb[2]{};
            sb[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            sb[1] = { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            VkDescriptorSetLayoutCreateInfo lci{};
            lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            lci.bindingCount = 2; lci.pBindings = sb;
            vkCreateDescriptorSetLayout(m_ctx->device(), &lci, nullptr, &m_sceneSetLayout);
        }
        // set=2: IBL samplers
        {
            VkDescriptorSetLayoutBinding ib[3]{};
            for (uint32_t i = 0; i < 3; ++i) {
                ib[i].binding = i;
                ib[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                ib[i].descriptorCount = 1;
                ib[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            }
            VkDescriptorSetLayoutCreateInfo lci{};
            lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            lci.bindingCount = 3; lci.pBindings = ib;
            vkCreateDescriptorSetLayout(m_ctx->device(), &lci, nullptr, &m_iblSetLayout);
        }

        VkDescriptorSetLayout lDsls[] = { m_lightingSetLayout, m_sceneSetLayout, m_iblSetLayout };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 3; plci.pSetLayouts = lDsls;
        vkCreatePipelineLayout(m_ctx->device(), &plci, nullptr, &m_lightingLayout);

        // Pipeline ────────────────────────────────────────────────────────────────
        VkShaderModule vert = loadSPV(m_ctx->device(), m_cfg.shaderDir + "/lighting.vert.spv");
        VkShaderModule frag = loadSPV(m_ctx->device(), m_cfg.shaderDir + "/lighting.frag.spv");

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
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
        VkDynamicState dynS[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 2; dyn.pDynamicStates = dynS;

        VkPipelineShaderStageCreateInfo stg[2]{};
        stg[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stg[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stg[0].module = vert; stg[0].pName = "main";
        stg[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stg[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stg[1].module = frag; stg[1].pName = "main";

        VkGraphicsPipelineCreateInfo gci{};
        gci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gci.stageCount = 2; gci.pStages = stg;
        gci.pVertexInputState = &vi; gci.pInputAssemblyState = &ia;
        gci.pViewportState = &vps; gci.pRasterizationState = &rs;
        gci.pMultisampleState = &ms; gci.pDepthStencilState = &dss;
        gci.pColorBlendState = &cb; gci.pDynamicState = &dyn;
        gci.layout = m_lightingLayout;
        gci.renderPass = m_lightingPass;
        vkCreateGraphicsPipelines(m_ctx->device(), VK_NULL_HANDLE, 1, &gci, nullptr, &m_lightingPipeline);

        vkDestroyShaderModule(m_ctx->device(), vert, nullptr);
        vkDestroyShaderModule(m_ctx->device(), frag, nullptr);
    }

    // ── initTonemapPass ───────────────────────────────────────────────────────────

    void Renderer::initTonemapPass() {
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0;
        b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 1; lci.pBindings = &b;
        vkCreateDescriptorSetLayout(m_ctx->device(), &lci, nullptr, &m_tonemapSetLayout);

        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; pc.offset = 0; pc.size = 8;

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &m_tonemapSetLayout;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pc;
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
        VkDynamicState dynS[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo      dyn{}; dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO; dyn.dynamicStateCount = 2; dyn.pDynamicStates = dynS;

        VkPipelineShaderStageCreateInfo stg[2]{};
        stg[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stg[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stg[0].module = vert; stg[0].pName = "main";
        stg[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stg[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stg[1].module = frag; stg[1].pName = "main";

        VkGraphicsPipelineCreateInfo gci{};
        gci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gci.stageCount = 2; gci.pStages = stg;
        gci.pVertexInputState = &vi; gci.pInputAssemblyState = &ia;
        gci.pViewportState = &vps; gci.pRasterizationState = &rs;
        gci.pMultisampleState = &ms; gci.pDepthStencilState = &dss;
        gci.pColorBlendState = &cb; gci.pDynamicState = &dyn;
        gci.layout = m_tonemapLayout;
        gci.renderPass = m_swapchain->renderPass();
        if (vkCreateGraphicsPipelines(m_ctx->device(), VK_NULL_HANDLE, 1, &gci, nullptr, &m_tonemapPipeline) != VK_SUCCESS)
            throw std::runtime_error("[vkgfx] Tonemap pipeline creation failed");

        vkDestroyShaderModule(m_ctx->device(), vert, nullptr);
        vkDestroyShaderModule(m_ctx->device(), frag, nullptr);
    }

    // ── initShadowPass ────────────────────────────────────────────────────────────
    // Creates a 2048x2048 depth-only render pass for directional shadow mapping.
    // The shadow map is sampled by the lighting pass with sampler2DShadow (PCF).

    void Renderer::initShadowPass() {
        VkFormat depthFmt = m_ctx->findDepthFormat();

        // Shadow map depth image
        m_shadowMap = m_ctx->allocateImage(
            { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE }, depthFmt,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        m_shadowMap.view = m_ctx->createImageView(
            m_shadowMap.image, depthFmt, VK_IMAGE_ASPECT_DEPTH_BIT);

        // Transition shadow map to SHADER_READ_ONLY_OPTIMAL immediately.
        // The render pass transitions it back to DEPTH_STENCIL on load (initialLayout=UNDEFINED),
        // then to SHADER_READ_ONLY on store (finalLayout=SHADER_READ_ONLY_OPTIMAL).
        // But when recordShadowPass() is skipped (no directional light), the image never
        // goes through the render pass — so it must start in SHADER_READ_ONLY_OPTIMAL
        // to match what the lighting pass descriptor expects.
        {
            VkCommandBuffer cmd = m_ctx->beginOneShot();
            VkImageMemoryBarrier b{};
            b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = m_shadowMap.image;
            b.srcAccessMask = 0;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            b.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &b);
            m_ctx->endOneShot(cmd);
        }

        // Comparison sampler for sampler2DShadow — hardware PCF
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE; // depth=1 outside shadow map = no shadow
        sci.compareEnable = VK_TRUE;
        sci.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;        // standard shadow compare
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.maxLod = 1.f;
        vkCreateSampler(m_ctx->device(), &sci, nullptr, &m_shadowSampler);

        // Depth-only render pass.
        // initialLayout = SHADER_READ_ONLY_OPTIMAL because we pre-transition the shadow map
        // at init time so it is always in a valid state even when this pass is skipped.
        VkAttachmentDescription depthAtt{};
        depthAtt.format = depthFmt;
        depthAtt.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAtt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        depthAtt.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference depRef{ 0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 0;
        sub.pDepthStencilAttachment = &depRef;

        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL; deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        deps[1].srcSubpass = 0; deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        VkRenderPassCreateInfo rpi{};
        rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount = 1; rpi.pAttachments = &depthAtt;
        rpi.subpassCount = 1; rpi.pSubpasses = &sub;
        rpi.dependencyCount = 2; rpi.pDependencies = deps;
        vkCreateRenderPass(m_ctx->device(), &rpi, nullptr, &m_shadowPass);

        VkFramebufferCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fi.renderPass = m_shadowPass;
        fi.attachmentCount = 1; fi.pAttachments = &m_shadowMap.view;
        fi.width = SHADOW_MAP_SIZE; fi.height = SHADOW_MAP_SIZE; fi.layers = 1;
        vkCreateFramebuffer(m_ctx->device(), &fi, nullptr, &m_shadowFb);

        // Push constant: model (64) + lightViewProj (64) = 128 bytes
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pc.offset = 0;
        pc.size = sizeof(MeshPush); // reuse same 128-byte struct

        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pc;
        vkCreatePipelineLayout(m_ctx->device(), &plci, nullptr, &m_shadowLayout);

        VkShaderModule vert = loadSPV(m_ctx->device(), m_cfg.shaderDir + "/shadow.vert.spv");

        auto binding = Vertex::bindingDescription();
        auto attribs = Vertex::attributeDescriptions();

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribs.size());
        vi.pVertexAttributeDescriptions = attribs.data();

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vps{};
        vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vps.viewportCount = 1; vps.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_FRONT_BIT;  // front-face cull reduces peter-panning
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth = 1.0f;
        rs.depthBiasEnable = VK_TRUE;
        rs.depthBiasConstantFactor = 1.25f;
        rs.depthBiasSlopeFactor = 1.75f;

        VkPipelineMultisampleStateCreateInfo msci{};
        msci.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        msci.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo dss{};
        dss.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        dss.depthTestEnable = VK_TRUE;
        dss.depthWriteEnable = VK_TRUE;
        dss.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

        VkDynamicState dynS[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 2; dyn.pDynamicStates = dynS;

        VkPipelineShaderStageCreateInfo stg{};
        stg.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stg.stage = VK_SHADER_STAGE_VERTEX_BIT;
        stg.module = vert; stg.pName = "main";

        VkGraphicsPipelineCreateInfo gci{};
        gci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gci.stageCount = 1; gci.pStages = &stg;
        gci.pVertexInputState = &vi; gci.pInputAssemblyState = &ia;
        gci.pViewportState = &vps; gci.pRasterizationState = &rs;
        gci.pMultisampleState = &msci; gci.pDepthStencilState = &dss;
        gci.pColorBlendState = &cb; gci.pDynamicState = &dyn;
        gci.layout = m_shadowLayout;
        gci.renderPass = m_shadowPass;
        vkCreateGraphicsPipelines(m_ctx->device(), VK_NULL_HANDLE, 1, &gci, nullptr, &m_shadowPipeline);

        vkDestroyShaderModule(m_ctx->device(), vert, nullptr);
    }

    // ── recordShadowPass ──────────────────────────────────────────────────────────

    void Renderer::recordShadowPass(VkCommandBuffer cmd, Scene& scene) {
        // Only render shadow pass when sun is enabled and shadow is on in config
        if (!m_cfg.sun.enabled) return;
        if (!scene.dirLight() || !scene.dirLight()->enabled()) return;

        // Compute lightViewProj (same calculation as SceneUBO upload)
        glm::vec3 lightDir = glm::normalize(scene.dirLight()->direction());
        glm::vec3 lightPos = -lightDir * 20.f;
        glm::mat4 lightView = glm::lookAt(lightPos, glm::vec3(0.f), glm::vec3(0.f, 1.f, 0.f));
        glm::mat4 lightProj = glm::ortho(-15.f, 15.f, -15.f, 15.f, 0.1f, 100.f);
        lightProj[1][1] *= -1.f;
        glm::mat4 lightVP = lightProj * lightView;

        VkClearValue depthClear{};
        depthClear.depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo rbi{};
        rbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rbi.renderPass = m_shadowPass;
        rbi.framebuffer = m_shadowFb;
        rbi.renderArea = { {0,0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE} };
        rbi.clearValueCount = 1; rbi.pClearValues = &depthClear;
        vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline);

        VkViewport vp{ 0.f, 0.f,
                      static_cast<float>(SHADOW_MAP_SIZE),
                      static_cast<float>(SHADOW_MAP_SIZE),
                      0.f, 1.f };
        VkRect2D sc{ {0,0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE} };
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);

        // Reuse MeshPush: model in .model, lightViewProj in .normalMatrix
        for (Mesh* mesh : scene.visibleMeshes()) {
            MeshPush push{};
            push.model = mesh->modelMatrix();
            push.normalMatrix = lightVP;  // shadow.vert reads lightViewProj from normalMatrix slot
            vkCmdPushConstants(cmd, m_shadowLayout,
                VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPush), &push);

            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(cmd, 0, 1, &mesh->vertexBuffer(), offsets);
            vkCmdBindIndexBuffer(cmd, mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, mesh->indexCount(), 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
    }

    // ── initPointShadowPass ───────────────────────────────────────────────────────
    // Creates a 512² depth-only cubemap render pass for omnidirectional point-light
    // shadow mapping.  The cube is rendered in 6 separate draw calls (one per face
    // framebuffer), which is more portable than the geometry-shader multi-layer path.
    // The result is sampled in the lighting pass as samplerCubeShadow + 8-tap PCF.

    void Renderer::initPointShadowPass() {
        VkFormat depthFmt = m_ctx->findDepthFormat();

        // ── Cube depth image (6 layers, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) ──────
        m_pointShadowCube = m_ctx->allocateImage(
            { POINT_SHADOW_SIZE, POINT_SHADOW_SIZE },
            depthFmt,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            /*mipLevels=*/1,
            /*layers=*/6,
            /*flags=*/VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);

        // Cube view — used by the lighting-pass samplerCubeShadow descriptor
        m_pointCubeSamplerView = m_ctx->createImageView(
            m_pointShadowCube.image, depthFmt,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            /*mips=*/1, /*layers=*/6,
            VK_IMAGE_VIEW_TYPE_CUBE);

        // Six individual 2D face views — one framebuffer attachment per face
        for (uint32_t face = 0; face < 6; ++face) {
            VkImageViewCreateInfo vci{};
            vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image    = m_pointShadowCube.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format   = depthFmt;
            vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, face, 1 };
            vkCreateImageView(m_ctx->device(), &vci, nullptr, &m_pointCubeFaceViews[face]);
        }

        // Pre-transition all 6 faces to SHADER_READ_ONLY_OPTIMAL.
        // Same reasoning as the directional shadow map: when there are no point
        // lights the pass is skipped entirely and the descriptor must still be valid.
        {
            VkCommandBuffer cmd = m_ctx->beginOneShot();
            VkImageMemoryBarrier b{};
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image               = m_pointShadowCube.image;
            b.srcAccessMask       = 0;
            b.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
            b.subresourceRange    = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 6 };
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &b);
            m_ctx->endOneShot(cmd);
        }

        // Comparison sampler for samplerCubeShadow — linear filtering, edge clamp.
        // CLAMP_TO_EDGE (not BORDER) avoids seam darkening at cube face boundaries.
        VkSamplerCreateInfo sci{};
        sci.sType         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter     = VK_FILTER_LINEAR;
        sci.minFilter     = VK_FILTER_LINEAR;
        sci.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.compareEnable = VK_TRUE;
        sci.compareOp     = VK_COMPARE_OP_LESS_OR_EQUAL;
        sci.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.maxLod        = 1.f;
        vkCreateSampler(m_ctx->device(), &sci, nullptr, &m_pointShadowSampler);

        // Depth-only render pass — identical structure to the directional shadow pass.
        // initialLayout = SHADER_READ_ONLY_OPTIMAL because we pre-transition above;
        // the render pass transitions back to DEPTH_STENCIL on load (clear) and then
        // to SHADER_READ_ONLY_OPTIMAL on store.
        VkAttachmentDescription depthAtt{};
        depthAtt.format         = depthFmt;
        depthAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
        depthAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        depthAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAtt.initialLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        depthAtt.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference depRef{ 0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
        VkSubpassDescription  sub{};
        sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount    = 0;
        sub.pDepthStencilAttachment = &depRef;

        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass      = VK_SUBPASS_EXTERNAL; deps[0].dstSubpass = 0;
        deps[0].srcStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[0].dstStageMask    = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].srcAccessMask   = VK_ACCESS_SHADER_READ_BIT;
        deps[0].dstAccessMask   = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        deps[1].srcSubpass      = 0; deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask    = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        deps[1].dstStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[1].srcAccessMask   = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;
        deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        VkRenderPassCreateInfo rpi{};
        rpi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpi.attachmentCount = 1; rpi.pAttachments  = &depthAtt;
        rpi.subpassCount    = 1; rpi.pSubpasses     = &sub;
        rpi.dependencyCount = 2; rpi.pDependencies  = deps;
        vkCreateRenderPass(m_ctx->device(), &rpi, nullptr, &m_pointShadowPass);

        // One framebuffer per cube face — each wraps the matching 2D face view
        for (uint32_t face = 0; face < 6; ++face) {
            VkFramebufferCreateInfo fi{};
            fi.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fi.renderPass      = m_pointShadowPass;
            fi.attachmentCount = 1;
            fi.pAttachments    = &m_pointCubeFaceViews[face];
            fi.width           = POINT_SHADOW_SIZE;
            fi.height          = POINT_SHADOW_SIZE;
            fi.layers          = 1;
            vkCreateFramebuffer(m_ctx->device(), &fi, nullptr, &m_pointShadowFbs[face]);
        }

        // Descriptor set layout: set=0, binding=0 — per-light UBO (vec4 lightPosAndFar)
        // Bound in both vertex and fragment stages so the fragment shader can compute
        // normalised linear depth without a second push constant.
        VkDescriptorSetLayoutBinding uboB{};
        uboB.binding         = 0;
        uboB.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboB.descriptorCount = 1;
        uboB.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 1;
        lci.pBindings    = &uboB;
        vkCreateDescriptorSetLayout(m_ctx->device(), &lci, nullptr, &m_pointShadowDsLayout);

        // Pipeline layout: set=0 (lightPosAndFar UBO) + 128-byte vertex push constant
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pc.offset     = 0;
        pc.size       = sizeof(PointShadowPush); // 128 bytes: model(64) + faceVP(64)

        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &m_pointShadowDsLayout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pc;
        vkCreatePipelineLayout(m_ctx->device(), &plci, nullptr, &m_pointShadowLayout);

        // Shaders
        VkShaderModule vert = loadSPV(m_ctx->device(),
                                       m_cfg.shaderDir + "/point_shadow.vert.spv");
        VkShaderModule frag = loadSPV(m_ctx->device(),
                                       m_cfg.shaderDir + "/point_shadow.frag.spv");

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
        vps.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vps.viewportCount = 1; vps.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode             = VK_POLYGON_MODE_FILL;
        rs.cullMode                = VK_CULL_MODE_FRONT_BIT; // front-face cull reduces peter-panning
        rs.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth               = 1.0f;
        rs.depthBiasEnable         = VK_TRUE;
        rs.depthBiasConstantFactor = 1.25f;
        rs.depthBiasSlopeFactor    = 1.75f;

        VkPipelineMultisampleStateCreateInfo msci{};
        msci.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        msci.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo dss{};
        dss.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        dss.depthTestEnable  = VK_TRUE;
        dss.depthWriteEnable = VK_TRUE;
        dss.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        // No colour attachments — depth only, attachmentCount stays 0

        VkDynamicState                   dynS[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 2; dyn.pDynamicStates = dynS;

        VkPipelineShaderStageCreateInfo stg[2]{};
        stg[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stg[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stg[0].module = vert; stg[0].pName = "main";
        stg[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stg[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stg[1].module = frag; stg[1].pName = "main";

        VkGraphicsPipelineCreateInfo gci{};
        gci.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gci.stageCount          = 2;  gci.pStages            = stg;
        gci.pVertexInputState   = &vi; gci.pInputAssemblyState = &ia;
        gci.pViewportState      = &vps; gci.pRasterizationState = &rs;
        gci.pMultisampleState   = &msci; gci.pDepthStencilState  = &dss;
        gci.pColorBlendState    = &cb; gci.pDynamicState        = &dyn;
        gci.layout              = m_pointShadowLayout;
        gci.renderPass          = m_pointShadowPass;
        vkCreateGraphicsPipelines(m_ctx->device(), VK_NULL_HANDLE,
                                   1, &gci, nullptr, &m_pointShadowPipeline);

        vkDestroyShaderModule(m_ctx->device(), vert, nullptr);
        vkDestroyShaderModule(m_ctx->device(), frag, nullptr);
    }

    // ── recordPointShadowPass ─────────────────────────────────────────────────────
    // Renders the scene's depth into all 6 faces of the point-shadow cubemap.
    // Only light index 0 (first enabled PointLight) acts as the shadow caster.
    // Each face is a separate render pass begin/end — no geometry shader required.

    void Renderer::recordPointShadowPass(VkCommandBuffer cmd, Scene& scene, uint32_t frameIdx) {
        // Find the first enabled PointLight that has shadow casting enabled.
        // This is the shadow caster for the single cube map slot.
        const PointLight* caster = nullptr;
        for (auto& pl : scene.pointLights()) {
            if (pl && pl->enabled() && pl->castsShadow()) { caster = pl.get(); break; }
        }
        if (!caster) return; // nothing to do

        auto& f = m_frames[frameIdx];

        // Upload lightPos + farPlane (= light radius) to the per-frame UBO.
        // The fragment shader uses this to compute normalised linear depth:
        //   gl_FragDepth = length(fragPos - lightPos) / farPlane
        struct { glm::vec4 lightPosAndFar; } ld;
        ld.lightPosAndFar = glm::vec4(caster->position(), caster->radius());
        {
            void* mapped = nullptr;
            vmaMapMemory(m_ctx->vma(),
                static_cast<VmaAllocation>(f.pointShadowLightUbo.allocation), &mapped);
            std::memcpy(mapped, &ld, sizeof(ld));
            vmaUnmapMemory(m_ctx->vma(),
                static_cast<VmaAllocation>(f.pointShadowLightUbo.allocation));
        }

        // Compute six face view-projection matrices.
        // 90° FOV + aspect=1.0 ensures each face covers exactly 1/6 of the sphere.
        // Vulkan NDC has Y pointing down, so proj[1][1] is negated (same as the
        // existing directional shadow pass convention in this engine).
        const glm::vec3 lpos     = caster->position();
        const float     farPlane = caster->radius();
        glm::mat4 proj = glm::perspective(glm::radians(90.0f), 1.0f, 0.01f, farPlane);
        proj[1][1] *= -1.f;

        // Face order matches VkCubeMapFace: +X -X +Y -Y +Z -Z
        struct FaceDir { glm::vec3 dir, up; };
        const FaceDir faces[6] = {
            {{ 1.f,  0.f,  0.f}, {0.f, -1.f,  0.f}}, // +X
            {{-1.f,  0.f,  0.f}, {0.f, -1.f,  0.f}}, // -X
            {{ 0.f,  1.f,  0.f}, {0.f,  0.f,  1.f}}, // +Y
            {{ 0.f, -1.f,  0.f}, {0.f,  0.f, -1.f}}, // -Y
            {{ 0.f,  0.f,  1.f}, {0.f, -1.f,  0.f}}, // +Z
            {{ 0.f,  0.f, -1.f}, {0.f, -1.f,  0.f}}, // -Z
        };
        glm::mat4 faceVP[6];
        for (int i = 0; i < 6; ++i)
            faceVP[i] = proj * glm::lookAt(lpos, lpos + faces[i].dir, faces[i].up);

        // Common viewport / scissor — fixed to the shadow map resolution
        const VkExtent2D ext{ POINT_SHADOW_SIZE, POINT_SHADOW_SIZE };
        VkViewport vp{ 0.f, 0.f,
                       static_cast<float>(POINT_SHADOW_SIZE),
                       static_cast<float>(POINT_SHADOW_SIZE),
                       0.f, 1.f };
        VkRect2D sc{ {0, 0}, ext };

        VkClearValue depthClear{};
        depthClear.depthStencil = { 1.0f, 0 };

        for (uint32_t face = 0; face < 6; ++face) {
            VkRenderPassBeginInfo rbi{};
            rbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rbi.renderPass      = m_pointShadowPass;
            rbi.framebuffer     = m_pointShadowFbs[face];
            rbi.renderArea      = { {0, 0}, ext };
            rbi.clearValueCount = 1;
            rbi.pClearValues    = &depthClear;
            vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              m_pointShadowPipeline);
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor (cmd, 0, 1, &sc);

            // Bind the light UBO (set=0) — same data for all 6 faces
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                m_pointShadowLayout, 0, 1, &f.pointShadowDs, 0, nullptr);

            // Draw each mesh with its face-specific view-projection
            for (Mesh* mesh : scene.visibleMeshes()) {
                PointShadowPush push{};
                push.model  = mesh->modelMatrix();
                push.faceVP = faceVP[face];
                vkCmdPushConstants(cmd, m_pointShadowLayout,
                    VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PointShadowPush), &push);

                VkDeviceSize offsets[] = { 0 };
                vkCmdBindVertexBuffers(cmd, 0, 1, &mesh->vertexBuffer(), offsets);
                vkCmdBindIndexBuffer  (cmd, mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed      (cmd, mesh->indexCount(), 1, 0, 0, 0);
            }

            vkCmdEndRenderPass(cmd);
        }
    }

    // ── initDefaultTextures ───────────────────────────────────────────────────────
    // Creates 1x1 solid-color images used as fallback descriptors.
    // Every binding in defaultMaterialSet must be written with valid image data.

    void Renderer::initDefaultTextures() {
        auto upload1x1 = [&](uint8_t r, uint8_t g, uint8_t b, uint8_t a) -> AllocatedImage {
            AllocatedImage img = m_ctx->allocateImage({ 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

            const uint8_t px[4] = { r, g, b, a };
            AllocatedBuffer staging = m_ctx->allocateBuffer(4,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
            void* mapped = nullptr;
            vmaMapMemory(m_ctx->vma(), static_cast<VmaAllocation>(staging.allocation), &mapped);
            std::memcpy(mapped, px, 4);
            vmaUnmapMemory(m_ctx->vma(), static_cast<VmaAllocation>(staging.allocation));

            VkCommandBuffer cmd = m_ctx->beginOneShot();

            VkImageMemoryBarrier b1{};
            b1.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b1.srcAccessMask = 0;
            b1.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b1.image = img.image;
            b1.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b1);

            VkBufferImageCopy region{};
            region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            region.imageExtent = { 1, 1, 1 };
            vkCmdCopyBufferToImage(cmd, staging.buffer, img.image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            VkImageMemoryBarrier b2 = b1;
            b2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b2);

            m_ctx->endOneShot(cmd);
            m_ctx->destroyBuffer(staging);

            img.view = m_ctx->createImageView(img.image, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_ASPECT_COLOR_BIT);
            return img;
            };

        // albedo fallback: white
        m_fallbackWhite = upload1x1(255, 255, 255, 255);
        // normal fallback: (0.5, 0.5, 1.0) = flat normal pointing +Z in tangent space
        m_fallbackNormal = upload1x1(128, 128, 255, 255);
        // RMA fallback: roughness=0.5 (128), metallic=0 (0), ao=1 (255)
        m_fallbackRMA = upload1x1(128, 0, 255, 255);

        // ── 1×1 black cube map for the dummy IBL descriptor set ─────────────────
        // All 6 faces = black (0,0,0,255). Used to write valid descriptors into
        // the IBL set (set=2) when IBL is disabled — Vulkan forbids unwritten descriptors.
        {
            m_fallbackCube = m_ctx->allocateImage(
                { 1, 1 }, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                1, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);

            // Upload 6 faces of black pixels
            const uint8_t black[4] = { 0, 0, 0, 255 };
            VkDeviceSize faceSize = 4;
            AllocatedBuffer staging = m_ctx->allocateBuffer(faceSize * 6,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
            void* mapped = nullptr;
            vmaMapMemory(m_ctx->vma(), static_cast<VmaAllocation>(staging.allocation), &mapped);
            for (int i = 0; i < 6; ++i)
                std::memcpy(static_cast<uint8_t*>(mapped) + i * faceSize, black, 4);
            vmaUnmapMemory(m_ctx->vma(), static_cast<VmaAllocation>(staging.allocation));

            VkCommandBuffer cmd = m_ctx->beginOneShot();

            VkImageMemoryBarrier b1{};
            b1.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b1.srcAccessMask = 0;
            b1.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b1.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b1.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b1.image = m_fallbackCube.image;
            b1.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6 };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b1);

            // One copy region per face
            VkBufferImageCopy regions[6]{};
            for (int i = 0; i < 6; ++i) {
                regions[i].bufferOffset = static_cast<VkDeviceSize>(i) * faceSize;
                regions[i].imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                                 static_cast<uint32_t>(i), 1 };
                regions[i].imageExtent = { 1, 1, 1 };
            }
            vkCmdCopyBufferToImage(cmd, staging.buffer, m_fallbackCube.image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6, regions);

            VkImageMemoryBarrier b2 = b1;
            b2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b2);

            m_ctx->endOneShot(cmd);
            m_ctx->destroyBuffer(staging);

            m_fallbackCube.view = m_ctx->createImageView(
                m_fallbackCube.image, VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_ASPECT_COLOR_BIT, 1, 6, VK_IMAGE_VIEW_TYPE_CUBE);
        }

        // Shared nearest-linear sampler for fallback textures
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_LINEAR;
        sci.minFilter = VK_FILTER_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.maxLod = 1.f;
        vkCreateSampler(m_ctx->device(), &sci, nullptr, &m_fallbackSampler);
    }

    // ── initPerFrameResources ─────────────────────────────────────────────────────

    void Renderer::initPerFrameResources() {
        // Samplers
        {
            VkSamplerCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter = VK_FILTER_LINEAR;
            sci.minFilter = VK_FILTER_LINEAR;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.maxLod = 1.f;
            vkCreateSampler(m_ctx->device(), &sci, nullptr, &m_gbufferSampler);
            vkCreateSampler(m_ctx->device(), &sci, nullptr, &m_hdrSampler);
        }

        VkCommandBufferAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = m_ctx->graphicsPool();
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
        std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> cmds{};
        vkAllocateCommandBuffers(m_ctx->device(), &ai, cmds.data());

        // Per-frame-slot acquire semaphores.
        m_acquireSemaphores.clear();
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
            m_acquireSemaphores.push_back(makeSemaphore(m_ctx->device()));

        // Per-swapchain-image renderFinished semaphores.
        // Each swapchain image has its own semaphore so present for image N never
        // conflicts with a pending present for a different image using the same slot.
        m_renderFinishedSems.clear();
        for (uint32_t i = 0; i < m_swapchain->imageCount(); ++i)
            m_renderFinishedSems.push_back(makeSemaphore(m_ctx->device()));

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            auto& f = m_frames[i];
            f.cmd = cmds[i];
            f.inFlight = makeFence(m_ctx->device(), true);

            f.sceneUbo = m_ctx->allocateBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
            f.lightUbo = m_ctx->allocateBuffer(sizeof(LightUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
            f.defaultParamsUbo = m_ctx->allocateBuffer(sizeof(PBRParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
            // Write default PBRParams (all zeros / defaults) — albedo=white, roughness=0.5
            {
                PBRParams defaults{};
                defaults.albedo = { 1.f, 1.f, 1.f, 1.f };
                defaults.emissive = { 0.f, 0.f, 0.f, 0.f };
                defaults.pbr = { 0.5f, 0.f, 1.f, 0.f }; // roughness, metallic, ao
                defaults.texFlags = { 0u, 0u, 0u, 0u };
                void* m = nullptr;
                vmaMapMemory(m_ctx->vma(), static_cast<VmaAllocation>(f.defaultParamsUbo.allocation), &m);
                std::memcpy(m, &defaults, sizeof(PBRParams));
                vmaUnmapMemory(m_ctx->vma(), static_cast<VmaAllocation>(f.defaultParamsUbo.allocation));
            }

            // Allocate descriptor sets ─────────────────────────────────────────────
            // gbuffer scene set (set=0 in G-buffer pass)
            {
                VkDescriptorSetAllocateInfo dsai{};
                dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                dsai.descriptorPool = m_descriptorPool.get();
                dsai.descriptorSetCount = 1;
                dsai.pSetLayouts = &m_gbufferSetLayout;
                vkAllocateDescriptorSets(m_ctx->device(), &dsai, &f.gbufferSceneSet);
            }
            // default material set (set=1 in G-buffer pass) — always bound as fallback.
            // Must be fully written: Vulkan validation errors if any binding is never updated.
            {
                VkDescriptorSetAllocateInfo dsai{};
                dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                dsai.descriptorPool = m_descriptorPool.get();
                dsai.descriptorSetCount = 1;
                dsai.pSetLayouts = &m_materialSetLayout;
                vkAllocateDescriptorSets(m_ctx->device(), &dsai, &f.defaultMaterialSet);

                // Write fallback descriptors using the shared 1×1 solid-color textures
                // created once in initDefaultTextures().
                // Bindings 0-3: texture samplers (albedo, normal, rma, emissive)
                VkDescriptorImageInfo texInfos[4]{};
                texInfos[0] = { m_fallbackSampler, m_fallbackWhite.view,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                texInfos[1] = { m_fallbackSampler, m_fallbackNormal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                texInfos[2] = { m_fallbackSampler, m_fallbackRMA.view,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                texInfos[3] = { m_fallbackSampler, m_fallbackWhite.view,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL }; // emissive black fallback

                VkWriteDescriptorSet tw[4]{};
                for (uint32_t b = 0; b < 4; ++b) {
                    tw[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    tw[b].dstSet = f.defaultMaterialSet;
                    tw[b].dstBinding = b;
                    tw[b].descriptorCount = 1;
                    tw[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    tw[b].pImageInfo = &texInfos[b];
                }
                vkUpdateDescriptorSets(m_ctx->device(), 4, tw, 0, nullptr);

                // Binding 4: PBRParams UBO
                VkDescriptorBufferInfo pbi{ f.defaultParamsUbo.buffer, 0, sizeof(PBRParams) };
                VkWriteDescriptorSet pw{};
                pw.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                pw.dstSet = f.defaultMaterialSet;
                pw.dstBinding = 4;
                pw.descriptorCount = 1;
                pw.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                pw.pBufferInfo = &pbi;
                vkUpdateDescriptorSets(m_ctx->device(), 1, &pw, 0, nullptr);
            }
            // lighting sets (set=0 and set=1)
            {
                VkDescriptorSetLayout layouts[] = { m_lightingSetLayout, m_sceneSetLayout };
                VkDescriptorSet       sets[2]{};
                VkDescriptorSetAllocateInfo dsai{};
                dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                dsai.descriptorPool = m_descriptorPool.get();
                dsai.descriptorSetCount = 2;
                dsai.pSetLayouts = layouts;
                vkAllocateDescriptorSets(m_ctx->device(), &dsai, sets);
                f.lightingGbufferSet = sets[0];
                f.lightingSceneSet = sets[1];
            }
            // tonemap set (set=0)
            {
                VkDescriptorSetAllocateInfo dsai{};
                dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                dsai.descriptorPool = m_descriptorPool.get();
                dsai.descriptorSetCount = 1;
                dsai.pSetLayouts = &m_tonemapSetLayout;
                vkAllocateDescriptorSets(m_ctx->device(), &dsai, &f.tonemapSet);
            }
            // IBL set (set=2 in lighting pass) — always allocate AND write valid fallback
            // descriptors. Vulkan validation error fires even in dead branches if a
            // bound descriptor was never written. Use the 1×1 black cube fallback.
            // initIBL() overwrites this with real cube maps when IBL is enabled.
            {
                VkDescriptorSetAllocateInfo dsai{};
                dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                dsai.descriptorPool = m_descriptorPool.get();
                dsai.descriptorSetCount = 1;
                dsai.pSetLayouts = &m_iblSetLayout;
                vkAllocateDescriptorSets(m_ctx->device(), &dsai, &f.iblSet);

                // Write fallback cube descriptors (black 1x1) for bindings 0,1,2
                // and a plain 2D fallback for the BRDF LUT (binding 2).
                VkDescriptorImageInfo cubeInfo{};
                cubeInfo.sampler = m_fallbackSampler;
                cubeInfo.imageView = m_fallbackCube.view;
                cubeInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkDescriptorImageInfo lutInfo{};
                lutInfo.sampler = m_fallbackSampler;
                lutInfo.imageView = m_fallbackWhite.view;
                lutInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                VkWriteDescriptorSet iblWrites[3]{};
                // binding 0: irradianceCube
                iblWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                iblWrites[0].dstSet = f.iblSet;
                iblWrites[0].dstBinding = 0;
                iblWrites[0].descriptorCount = 1;
                iblWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                iblWrites[0].pImageInfo = &cubeInfo;
                // binding 1: prefilteredCube
                iblWrites[1] = iblWrites[0];
                iblWrites[1].dstBinding = 1;
                // binding 2: brdfLut (2D)
                iblWrites[2] = iblWrites[0];
                iblWrites[2].dstBinding = 2;
                iblWrites[2].pImageInfo = &lutInfo;

                vkUpdateDescriptorSets(m_ctx->device(), 3, iblWrites, 0, nullptr);
            }

            // ── Point shadow per-frame resources ──────────────────────────────
            // Small UBO holds vec4(lightPos, farPlane) — written by recordPointShadowPass().
            f.pointShadowLightUbo = m_ctx->allocateBuffer(
                sizeof(glm::vec4), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
            {
                VkDescriptorSetAllocateInfo psai{};
                psai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                psai.descriptorPool     = m_descriptorPool.get();
                psai.descriptorSetCount = 1;
                psai.pSetLayouts        = &m_pointShadowDsLayout;
                vkAllocateDescriptorSets(m_ctx->device(), &psai, &f.pointShadowDs);

                VkDescriptorBufferInfo psBI{ f.pointShadowLightUbo.buffer, 0, sizeof(glm::vec4) };
                VkWriteDescriptorSet   psW{};
                psW.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                psW.dstSet          = f.pointShadowDs;
                psW.dstBinding      = 0;
                psW.descriptorCount = 1;
                psW.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                psW.pBufferInfo     = &psBI;
                vkUpdateDescriptorSets(m_ctx->device(), 1, &psW, 0, nullptr);
            }
            {
                VkDescriptorBufferInfo bi{ f.sceneUbo.buffer, 0, sizeof(SceneUBO) };
                VkWriteDescriptorSet w{};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = f.gbufferSceneSet;
                w.dstBinding = 0;
                w.descriptorCount = 1;
                w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                w.pBufferInfo = &bi;
                vkUpdateDescriptorSets(m_ctx->device(), 1, &w, 0, nullptr);
            }

            // Write lightingGbufferSet:
            //   bindings 0-5: G-buffer colour images + depth (regular sampler)
            //   binding  6  : shadow map      (comparison sampler for sampler2DShadow)
            for (uint32_t g = 0; g < 6; ++g) {
                VkDescriptorImageInfo imgInfo{};
                imgInfo.sampler = m_gbufferSampler;
                imgInfo.imageView = m_gbuffer[g].view;
                imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                VkWriteDescriptorSet w{};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = f.lightingGbufferSet;
                w.dstBinding = g;
                w.descriptorCount = 1;
                w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w.pImageInfo = &imgInfo;
                vkUpdateDescriptorSets(m_ctx->device(), 1, &w, 0, nullptr);
            }
            // Binding 6: directional shadow map (comparison sampler)
            {
                VkDescriptorImageInfo shadowInfo{};
                shadowInfo.sampler = m_shadowSampler;
                shadowInfo.imageView = m_shadowMap.view;
                shadowInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                VkWriteDescriptorSet w{};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = f.lightingGbufferSet;
                w.dstBinding = 6;
                w.descriptorCount = 1;
                w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w.pImageInfo = &shadowInfo;
                vkUpdateDescriptorSets(m_ctx->device(), 1, &w, 0, nullptr);
            }
            // Binding 7: point light shadow cubemap (comparison sampler)
            {
                VkDescriptorImageInfo ptInfo{};
                ptInfo.sampler     = m_pointShadowSampler;
                ptInfo.imageView   = m_pointCubeSamplerView;
                ptInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                VkWriteDescriptorSet w{};
                w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet          = f.lightingGbufferSet;
                w.dstBinding      = 7;
                w.descriptorCount = 1;
                w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w.pImageInfo      = &ptInfo;
                vkUpdateDescriptorSets(m_ctx->device(), 1, &w, 0, nullptr);
            }

            // Write lightingSceneSet: scene UBO + light UBO
            {
                VkDescriptorBufferInfo scBI{ f.sceneUbo.buffer, 0, sizeof(SceneUBO) };
                VkDescriptorBufferInfo liBI{ f.lightUbo.buffer, 0, sizeof(LightUBO) };
                VkWriteDescriptorSet ws[2]{};
                ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                ws[0].dstSet = f.lightingSceneSet; ws[0].dstBinding = 0;
                ws[0].descriptorCount = 1; ws[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                ws[0].pBufferInfo = &scBI;
                ws[1] = ws[0];
                ws[1].dstBinding = 1; ws[1].pBufferInfo = &liBI;
                vkUpdateDescriptorSets(m_ctx->device(), 2, ws, 0, nullptr);
            }

            // Write tonemapSet: HDR target sampler
            {
                VkDescriptorImageInfo hdrInfo{};
                hdrInfo.sampler = m_hdrSampler;
                hdrInfo.imageView = m_hdrTarget.view;
                hdrInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                VkWriteDescriptorSet w{};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = f.tonemapSet; w.dstBinding = 0;
                w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w.pImageInfo = &hdrInfo;
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
            dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dsai.descriptorPool = m_descriptorPool.get();
            dsai.descriptorSetCount = 1;
            dsai.pSetLayouts = &m_iblSetLayout;
            vkAllocateDescriptorSets(m_ctx->device(), &dsai, &f.iblSet);

            VkDescriptorImageInfo imgs[3]{};
            imgs[0] = { m_ibl->cubeSampler(), m_ibl->irradianceView(),  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            imgs[1] = { m_ibl->cubeSampler(), m_ibl->prefilteredView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            imgs[2] = { m_ibl->brdfSampler(), m_ibl->brdfLutView(),     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

            VkWriteDescriptorSet ws[3]{};
            for (int k = 0; k < 3; ++k) {
                ws[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                ws[k].dstSet = f.iblSet; ws[k].dstBinding = static_cast<uint32_t>(k);
                ws[k].descriptorCount = 1; ws[k].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                ws[k].pImageInfo = &imgs[k];
            }
            vkUpdateDescriptorSets(m_ctx->device(), 3, ws, 0, nullptr);
        }
    }

    // ── applyConfig / rebuild ─────────────────────────────────────────────────────

    void Renderer::applyConfig(const RendererConfig& cfg) {
        bool iblChanged = (cfg.ibl.enabled != m_cfg.ibl.enabled)
            || (cfg.ibl.hdrPath != m_cfg.ibl.hdrPath)
            || (cfg.ibl.envMapSize != m_cfg.ibl.envMapSize);
        m_cfg = cfg;

        if (iblChanged) {
            vkDeviceWaitIdle(m_ctx->device());
            m_ibl->destroy();
            // After destroying IBL the old cube map views are gone.
            // Re-write fallback descriptors into every iblSet so set=2 is always valid.
            // If IBL is re-enabled, initIBL() will overwrite with real views.
            for (auto& f : m_frames) {
                // Allocate if for some reason not yet allocated
                if (f.iblSet == VK_NULL_HANDLE) {
                    VkDescriptorSetAllocateInfo dsai{};
                    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                    dsai.descriptorPool = m_descriptorPool.get();
                    dsai.descriptorSetCount = 1;
                    dsai.pSetLayouts = &m_iblSetLayout;
                    vkAllocateDescriptorSets(m_ctx->device(), &dsai, &f.iblSet);
                }
                // Always re-write fallback — previous real IBL views are now dead
                VkDescriptorImageInfo ci{ m_fallbackSampler, m_fallbackCube.view,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                VkDescriptorImageInfo li{ m_fallbackSampler, m_fallbackWhite.view,
                                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                VkWriteDescriptorSet ws[3]{};
                ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                ws[0].dstSet = f.iblSet; ws[0].dstBinding = 0;
                ws[0].descriptorCount = 1; ws[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                ws[0].pImageInfo = &ci;
                ws[1] = ws[0]; ws[1].dstBinding = 1;
                ws[2] = ws[0]; ws[2].dstBinding = 2; ws[2].pImageInfo = &li;
                vkUpdateDescriptorSets(m_ctx->device(), 3, ws, 0, nullptr);
            }
            if (m_cfg.ibl.enabled) initIBL();
        }
        // Sun/debug changes are applied each frame through the LightUBO upload
    }

    void Renderer::rebuild(const RendererConfig& cfg) {
        shutdown();
        m_cfg = cfg;
        initDescriptorPools();
        initDefaultTextures();
        initShadowPass();
        initPointShadowPass();
        initGBuffer();
        initLightingPass();
        initTonemapPass();
        initPerFrameResources();
        if (m_cfg.ibl.enabled) initIBL();
        if (m_cfg.profiling.enabled) {
            m_profiler.init(*m_ctx, MAX_FRAMES_IN_FLIGHT);
            if (m_cfg.profiling.showOverlay)
                initImGui();
        }
        m_initialized = true;
    }

    // ── uploadMeshMaterials ───────────────────────────────────────────────────────
    // Called from render() before the first draw: ensures every mesh that has a
    // PBRMaterial gets a valid VkDescriptorSet.  Safe to call every frame — skips
    // meshes whose material already has a set.

    void Renderer::uploadMeshMaterials(Scene& scene) {
        for (auto& meshPtr : scene.meshes()) {
            PBRMaterial* mat = meshPtr->material();
            if (!mat || mat->descriptorSet() != VK_NULL_HANDLE) continue;

            // Allocate set=1 for this material
            VkDescriptorSetAllocateInfo dsai{};
            dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dsai.descriptorPool = m_descriptorPool.get();
            dsai.descriptorSetCount = 1;
            dsai.pSetLayouts = &m_materialSetLayout;
            VkDescriptorSet ds = VK_NULL_HANDLE;
            if (vkAllocateDescriptorSets(m_ctx->device(), &dsai, &ds) != VK_SUCCESS) {
                std::cerr << "[vkgfx] Failed to allocate material descriptor set\n";
                continue;
            }
            mat->setDescriptorSet(ds);

            // Fallbacks for missing texture slots
            VkDescriptorImageInfo fallbacks[4]{};
            fallbacks[0] = { m_fallbackSampler, m_fallbackWhite.view,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL }; // albedo
            fallbacks[1] = { m_fallbackSampler, m_fallbackNormal.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL }; // normal
            fallbacks[2] = { m_fallbackSampler, m_fallbackRMA.view,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL }; // rma
            fallbacks[3] = { m_fallbackSampler, m_fallbackWhite.view,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL }; // emissive (black — but emissive.w=0)

            // Texture samplers (bindings 0-3: albedo, normal, rma, emissive)
            std::shared_ptr<Texture> texSlots[4] = {
                mat->albedoTex(), mat->normalTex(), mat->rmaTex(), nullptr
            };
            VkDescriptorImageInfo imgInfos[4]{};
            VkWriteDescriptorSet  tw[4]{};
            for (uint32_t b = 0; b < 4; ++b) {
                imgInfos[b] = (texSlots[b] && texSlots[b]->valid())
                    ? VkDescriptorImageInfo{ texSlots[b]->sampler(),
                                            texSlots[b]->view(),
                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL }
                : fallbacks[b];
                tw[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                tw[b].dstSet = ds;
                tw[b].dstBinding = b;
                tw[b].descriptorCount = 1;
                tw[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                tw[b].pImageInfo = &imgInfos[b];
            }
            vkUpdateDescriptorSets(m_ctx->device(), 4, tw, 0, nullptr);

            // Binding 4: PBRParams UBO
            AllocatedBuffer paramsUbo = m_ctx->allocateBuffer(sizeof(PBRParams),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true);
            {
                void* m = nullptr;
                vmaMapMemory(m_ctx->vma(),
                    static_cast<VmaAllocation>(paramsUbo.allocation), &m);
                std::memcpy(m, &mat->params(), sizeof(PBRParams));
                vmaUnmapMemory(m_ctx->vma(),
                    static_cast<VmaAllocation>(paramsUbo.allocation));
            }
            m_materialUbos.push_back(paramsUbo);

            VkDescriptorBufferInfo pbi{ paramsUbo.buffer, 0, sizeof(PBRParams) };
            VkWriteDescriptorSet pw{};
            pw.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            pw.dstSet = ds;
            pw.dstBinding = 4;
            pw.descriptorCount = 1;
            pw.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            pw.pBufferInfo = &pbi;
            vkUpdateDescriptorSets(m_ctx->device(), 1, &pw, 0, nullptr);
        }
    }
    // ── shutdown ──────────────────────────────────────────────────────────────────

    void Renderer::shutdown() {
        if (!m_initialized) return;
        m_shuttingDown = true;

        vkDeviceWaitIdle(m_ctx->device());

        // Release frame graph lambdas first to avoid dangling captured references.
        if (m_frameGraph) m_frameGraph->reset();

        // ImGui must be shut down before Vulkan objects it references are freed.
        if (m_imguiInitialized) shutdownImGui();
        m_profiler.destroy(m_ctx->device());

        for (auto& f : m_frames) {
            m_ctx->destroyBuffer(f.sceneUbo);
            m_ctx->destroyBuffer(f.lightUbo);
            m_ctx->destroyBuffer(f.defaultParamsUbo);
            m_ctx->destroyBuffer(f.pointShadowLightUbo);
            // Semaphores/fences are RAII VkHandles — destroyed automatically
        }

        auto dev = m_ctx->device();
        auto td = [&](auto& h, auto fn) { if (h != VK_NULL_HANDLE) { fn(dev, h, nullptr); h = VK_NULL_HANDLE; } };

        td(m_gbufferPipeline, vkDestroyPipeline);
        td(m_gbufferLayout, vkDestroyPipelineLayout);
        td(m_lightingPipeline, vkDestroyPipeline);
        td(m_lightingLayout, vkDestroyPipelineLayout);
        td(m_tonemapPipeline, vkDestroyPipeline);
        td(m_tonemapLayout, vkDestroyPipelineLayout);

        td(m_gbufferPass, vkDestroyRenderPass);
        td(m_lightingPass, vkDestroyRenderPass);
        td(m_gbufferFb, vkDestroyFramebuffer);
        td(m_lightingFb, vkDestroyFramebuffer);

        td(m_gbufferSetLayout, vkDestroyDescriptorSetLayout);
        td(m_materialSetLayout, vkDestroyDescriptorSetLayout);
        td(m_lightingSetLayout, vkDestroyDescriptorSetLayout);
        td(m_sceneSetLayout, vkDestroyDescriptorSetLayout);
        td(m_iblSetLayout, vkDestroyDescriptorSetLayout);
        td(m_tonemapSetLayout, vkDestroyDescriptorSetLayout);

        td(m_hdrSampler, vkDestroySampler);
        td(m_gbufferSampler, vkDestroySampler);
        td(m_fallbackSampler, vkDestroySampler);
        td(m_shadowSampler, vkDestroySampler);

        td(m_shadowPipeline, vkDestroyPipeline);
        td(m_shadowLayout, vkDestroyPipelineLayout);
        td(m_shadowPass, vkDestroyRenderPass);
        td(m_shadowFb, vkDestroyFramebuffer);

        // ── Point shadow cubemap cleanup ──────────────────────────────────────
        td(m_pointShadowPipeline, vkDestroyPipeline);
        td(m_pointShadowLayout, vkDestroyPipelineLayout);
        td(m_pointShadowDsLayout, vkDestroyDescriptorSetLayout);
        td(m_pointShadowPass, vkDestroyRenderPass);
        td(m_pointShadowSampler, vkDestroySampler);
        for (uint32_t fi = 0; fi < 6; ++fi) {
            td(m_pointShadowFbs[fi], vkDestroyFramebuffer);
            if (m_pointCubeFaceViews[fi] != VK_NULL_HANDLE) {
                vkDestroyImageView(dev, m_pointCubeFaceViews[fi], nullptr);
                m_pointCubeFaceViews[fi] = VK_NULL_HANDLE;
            }
        }
        if (m_pointCubeSamplerView != VK_NULL_HANDLE) {
            vkDestroyImageView(dev, m_pointCubeSamplerView, nullptr);
            m_pointCubeSamplerView = VK_NULL_HANDLE;
        }

        for (auto& img : m_gbuffer) m_ctx->destroyImage(img);
        m_ctx->destroyImage(m_hdrTarget);
        m_ctx->destroyImage(m_shadowMap);
        m_ctx->destroyImage(m_pointShadowCube);
        m_ctx->destroyImage(m_fallbackWhite);
        m_ctx->destroyImage(m_fallbackNormal);
        m_ctx->destroyImage(m_fallbackRMA);
        m_ctx->destroyImage(m_fallbackCube);

        m_ibl->destroy();

        for (auto& buf : m_materialUbos)
            m_ctx->destroyBuffer(buf);
        m_materialUbos.clear();

        m_descriptorPool.reset();
        m_acquireSemaphores.clear();
        m_renderFinishedSems.clear();

        m_initialized  = false;
        m_shuttingDown = false;
    }

    // ── rebuildOffscreenResources ─────────────────────────────────────────────────
    // Called from render() when m_swapchain->extent() no longer matches m_offscreenExtent.
    // Destroys and re-creates G-buffer images, HDR target, and their framebuffers,
    // then re-writes the descriptor sets that reference those image views.
    // Does NOT touch render passes, pipelines, or UBO descriptor sets — those are
    // extent-independent and never need to be re-created on resize.

    void Renderer::rebuildOffscreenResources() {
        // Do not attempt to rebuild while shutting down — the swapchain and
        // G-buffer images are already being destroyed, any resize event that
        // arrives via vkDeviceWaitIdle's Win32 message pump must be ignored.
        if (m_shuttingDown) return;

        vkDeviceWaitIdle(m_ctx->device());

        VkExtent2D newExt = m_swapchain->extent();
        if (newExt.width == 0 || newExt.height == 0) return; // window minimised

        // ── Destroy stale G-buffer images + framebuffer ───────────────────────────
        for (auto& img : m_gbuffer) m_ctx->destroyImage(img);
        if (m_gbufferFb != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(m_ctx->device(), m_gbufferFb, nullptr);
            m_gbufferFb = VK_NULL_HANDLE;
        }

        // ── Destroy stale HDR target + lighting framebuffer ───────────────────────
        m_ctx->destroyImage(m_hdrTarget);
        if (m_lightingFb != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(m_ctx->device(), m_lightingFb, nullptr);
            m_lightingFb = VK_NULL_HANDLE;
        }

        m_offscreenExtent = newExt;

        // ── Re-create G-buffer images ─────────────────────────────────────────────
        const VkFormat colorFmts[5] = {
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_FORMAT_R16G16B16A16_SFLOAT,
        };
        VkFormat depthFmt = m_ctx->findDepthFormat();

        const VkImageUsageFlags colorUsage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        const VkImageUsageFlags depthUsage =
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        for (int i = 0; i < 5; ++i) {
            m_gbuffer[i] = m_ctx->allocateImage(newExt, colorFmts[i], colorUsage);
            m_gbuffer[i].view = m_ctx->createImageView(
                m_gbuffer[i].image, colorFmts[i], VK_IMAGE_ASPECT_COLOR_BIT);
        }
        m_gbuffer[5] = m_ctx->allocateImage(newExt, depthFmt, depthUsage);
        m_gbuffer[5].view = m_ctx->createImageView(
            m_gbuffer[5].image, depthFmt, VK_IMAGE_ASPECT_DEPTH_BIT);

        // ── Re-create G-buffer framebuffer ────────────────────────────────────────
        VkImageView fbViews[6] = {
            m_gbuffer[0].view, m_gbuffer[1].view, m_gbuffer[2].view,
            m_gbuffer[3].view, m_gbuffer[4].view, m_gbuffer[5].view
        };
        VkFramebufferCreateInfo fi{};
        fi.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fi.renderPass      = m_gbufferPass;
        fi.attachmentCount = 6;
        fi.pAttachments    = fbViews;
        fi.width           = newExt.width;
        fi.height          = newExt.height;
        fi.layers          = 1;
        vkCreateFramebuffer(m_ctx->device(), &fi, nullptr, &m_gbufferFb);

        // ── Re-create HDR target ──────────────────────────────────────────────────
        m_hdrTarget = m_ctx->allocateImage(newExt, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        m_hdrTarget.view = m_ctx->createImageView(
            m_hdrTarget.image, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

        // ── Re-create lighting framebuffer ────────────────────────────────────────
        fi = {};
        fi.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fi.renderPass      = m_lightingPass;
        fi.attachmentCount = 1;
        fi.pAttachments    = &m_hdrTarget.view;
        fi.width           = newExt.width;
        fi.height          = newExt.height;
        fi.layers          = 1;
        vkCreateFramebuffer(m_ctx->device(), &fi, nullptr, &m_lightingFb);

        // ── Re-write per-frame descriptor sets that reference the new views ───────
        // lightingGbufferSet bindings 0-5 reference G-buffer views.
        // tonemapSet binding 0 references the HDR target view.
        // Shadow map (binding 6), point shadow cubemap (binding 7), and UBO sets
        // are extent-independent — unchanged on resize.
        for (auto& f : m_frames) {
            for (uint32_t g = 0; g < 6; ++g) {
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

            VkDescriptorImageInfo hdrInfo{};
            hdrInfo.sampler     = m_hdrSampler;
            hdrInfo.imageView   = m_hdrTarget.view;
            hdrInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkWriteDescriptorSet w{};
            w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet          = f.tonemapSet;
            w.dstBinding      = 0;
            w.descriptorCount = 1;
            w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.pImageInfo      = &hdrInfo;
            vkUpdateDescriptorSets(m_ctx->device(), 1, &w, 0, nullptr);
        }

        std::cout << "[vkgfx] Offscreen resources rebuilt: "
                  << newExt.width << "×" << newExt.height << "\n";

        // ImGui fonts / descriptor sets are not extent-dependent — no rebuild needed.
    }

    // ── initImGui ─────────────────────────────────────────────────────────────────
    // Sets up Dear ImGui with the Vulkan + GLFW backends.  Creates a private
    // descriptor pool (ImGui manages its own descriptor sets internally).
    // Uploads the default font atlas via a one-shot command buffer.
    // Compiled to a no-op stub unless VKGFX_ENABLE_PROFILING is defined.

    void Renderer::initImGui() {
#ifdef VKGFX_ENABLE_PROFILING
        if (!m_window) return;

        // Dedicated pool for ImGui — one combined-image-sampler per font texture
        // plus a safety margin for any user-added textures.
        VkDescriptorPoolSize poolSizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16 },
        };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pci.maxSets       = 16;
        pci.poolSizeCount = 1;
        pci.pPoolSizes    = poolSizes;
        vkCreateDescriptorPool(m_ctx->device(), &pci, nullptr, &m_imguiPool);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.IniFilename  = nullptr;  // disable imgui.ini — keeps the output dir clean

        // Style: dark + rounded corners, semi-transparent windows
        ImGui::StyleColorsDark();
        ImGui::GetStyle().WindowRounding   = 6.f;
        ImGui::GetStyle().FrameRounding    = 4.f;
        ImGui::GetStyle().GrabRounding     = 4.f;
        ImGui::GetStyle().WindowBorderSize = 0.f;

        ImGui_ImplGlfw_InitForVulkan(m_window->handle(), /*install_callbacks=*/true);

        ImGui_ImplVulkan_InitInfo ii{};
        ii.Instance        = m_ctx->instance();
        ii.PhysicalDevice  = m_ctx->gpu();
        ii.Device          = m_ctx->device();
        ii.QueueFamily     = m_ctx->queues().graphics;
        ii.Queue           = m_ctx->graphicsQ();
        ii.DescriptorPool  = m_imguiPool;
        ii.RenderPass      = m_swapchain->renderPass(); // tonemap → swapchain pass
        ii.MinImageCount   = 2;
        ii.ImageCount      = m_swapchain->imageCount();
        ii.MSAASamples     = VK_SAMPLE_COUNT_1_BIT;    // swapchain image is 1x
        ImGui_ImplVulkan_Init(&ii);

        // Upload font atlas via a one-shot command buffer
        VkCommandBuffer cmd = m_ctx->beginOneShot();
        ImGui_ImplVulkan_CreateFontsTexture();
        m_ctx->endOneShot(cmd);
        ImGui_ImplVulkan_DestroyFontsTexture();  // free CPU-side staging data

        m_imguiInitialized = true;
        std::cout << "[vkgfx][INFO] ImGui overlay initialised\n";
#endif
    }

    // ── shutdownImGui ─────────────────────────────────────────────────────────────

    void Renderer::shutdownImGui() {
#ifdef VKGFX_ENABLE_PROFILING
        if (!m_imguiInitialized) return;
        vkDeviceWaitIdle(m_ctx->device());
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        if (m_imguiPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_ctx->device(), m_imguiPool, nullptr);
            m_imguiPool = VK_NULL_HANDLE;
        }
        m_imguiInitialized = false;
#endif
    }

    // ── beginImGuiFrame ───────────────────────────────────────────────────────────
    // Called at the very start of render() on the CPU, before GPU commands.
    // Starts a new ImGui frame, calls user ImGui code and the profiler overlay.
    // The resulting draw data is consumed by endImGuiFrame() at record time.

    void Renderer::beginImGuiFrame() {
#ifdef VKGFX_ENABLE_PROFILING
        if (!m_imguiInitialized) return;
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Built-in profiler overlay
        if (m_cfg.profiling.showOverlay)
            m_profiler.renderOverlay();

        // User-registered ImGui callback
        if (m_imguiCallback)
            m_imguiCallback();

        ImGui::Render();
#endif
    }

    // ── endImGuiFrame ─────────────────────────────────────────────────────────────
    // Submits ImGui draw data into the active command buffer.
    // Must be called while the swapchain render pass is active (inside recordTonemap).

    void Renderer::endImGuiFrame(VkCommandBuffer cmd) {
#ifdef VKGFX_ENABLE_PROFILING
        if (!m_imguiInitialized) return;
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
#else
        (void)cmd;
#endif
    }

} // namespace vkgfx