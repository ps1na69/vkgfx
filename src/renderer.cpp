// renderer.cpp — Deferred shading pipeline implementation.
//
// Pass order per frame:
//   1. Geometry   → fills G-buffer (position, normal, albedo, material, emissive + depth)
//   2. SSAO       → hemisphere AO in view-space   → R8 image
//   3. SSAO Blur  → depth-aware 4×4 box filter    → R8 image
//   4. Lighting   → Cook-Torrance PBR, all lights → RGBA16F HDR image
//   5. Tonemap    → ACES/Reinhard/Hable + gamma   → swapchain image

#include "vkgfx/renderer.h"
#include <chrono>
#include <algorithm>
#include <future>
#include <mutex>
#include <random>
#include <cstring>

namespace vkgfx {

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────
Renderer::Renderer(Window& window, const RendererSettings& settings)
    : m_window(window), m_settings(settings)
{
    Context::CreateInfo ci;
    ci.appName          = window.settings().title;
    ci.enableValidation = settings.validation;
    ci.preferDedicated  = true;
    m_ctx    = std::make_shared<Context>(ci);
    m_surface = window.createSurface(m_ctx->instance());
    m_ctx->initDevice(m_surface);

    auto [w, h] = window.getFramebufferSize();
    m_swapchain = std::make_unique<Swapchain>(*m_ctx, m_surface, w, h,
                                               settings.vsync, VK_SAMPLE_COUNT_1_BIT);

    VkPipelineCacheCreateInfo pcCI{};
    pcCI.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    vkCreatePipelineCache(m_ctx->device(), &pcCI, nullptr, &m_pipelineCache);

    m_whiteTexture      = Texture::fromColor(m_ctx, {1.f,1.f,1.f,1.f});
    m_flatNormalTexture = Texture::fromColor(m_ctx, {0.5f,0.5f,1.f,1.f});

    createDescriptorPool();
    createDescriptorLayouts();

    // Render passes must exist before we create pipelines and size-dep resources.
    createGeometryRenderPass();
    createSSAORenderPass();
    createSSAOBlurRenderPass();
    createLightingRenderPass();

    createPerFrameBuffers();
    createSSAOKernel();
    createSizeDependentResources();
    allocateAndWriteDescriptorSets();

    createGeometryPipeline();
    createSSAOPipeline();
    createSSAOBlurPipeline();
    createLightingPipeline();
    createTonemapPipeline();

    // ── New systems ─────────────────────────────────────────────────────────
    std::cout << "[VKGFX] init: render graph..." << std::flush;
    initRenderGraph();
    std::cout << " OK\n[VKGFX] init: shadows..." << std::flush;
    initShadows();
    std::cout << " OK\n[VKGFX] init: GPU culling..." << std::flush;
    initGPUCulling();
    std::cout << " OK\n[VKGFX] init: IBL..." << std::flush;
    initIBL();  // tries assets/sky.hdr; no-ops gracefully if missing
    std::cout << " OK" << std::endl;

    uint32_t wc = settings.workerThreads == 0
                ? std::max(1u, std::thread::hardware_concurrency() - 1u)
                : settings.workerThreads;
    m_threadPool = std::make_unique<ThreadPool>(wc);
    createWorkerPools(wc);

    std::cout << "[VKGFX] Deferred renderer ready ("
              << w << "x" << h << ", " << wc << " workers)\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Destructor / Shutdown
// ─────────────────────────────────────────────────────────────────────────────
Renderer::~Renderer() { shutdown(); }

void Renderer::shutdown(Scene* scene) {
    if (!m_ctx) return;

    m_ctx->waitIdle();

    for (auto& d : m_deletionQueue) {
        m_ctx->destroyBuffer(d.vertexBuffer);
        m_ctx->destroyBuffer(d.indexBuffer);
        for (auto& b : d.matUBOs) m_ctx->destroyBuffer(b);
    }
    m_deletionQueue.clear();

    m_threadPool.reset();
    destroyWorkerPools();

    for (auto& [ptr, data] : m_meshData) {
        // Destroy the GPU-side vertex/index buffers uploaded by uploadMeshToGPU().
        // These live on the Mesh object itself; the Mesh destructor cannot free them
        // because it has no Context reference, so the renderer must do it here.
        if (ptr) {
            m_ctx->destroyBuffer(ptr->vertexBuffer);
            m_ctx->destroyBuffer(ptr->indexBuffer);
        }
        for (auto& b : data.matUBOs) m_ctx->destroyBuffer(b);
    }
    m_meshData.clear();

    destroySizeDependentResources();

    auto dev = m_ctx->device();
    auto killPL = [&](VkPipeline& p, VkPipelineLayout& l) {
        if (p) { vkDestroyPipeline(dev, p, nullptr);       p = VK_NULL_HANDLE; }
        if (l) { vkDestroyPipelineLayout(dev, l, nullptr); l = VK_NULL_HANDLE; }
    };
    killPL(m_geomPipeline,     m_geomPipeLayout);
    killPL(m_ssaoPipeline,     m_ssaoPipeLayout);
    killPL(m_ssaoBlurPipeline, m_ssaoBlurPipeLayout);
    killPL(m_lightPipeline,    m_lightPipeLayout);
    killPL(m_tonemapPipeline,  m_tonemapPipeLayout);

    for (VkRenderPass* rp : {&m_geomRP,&m_ssaoRP,&m_ssaoBlurRP,&m_lightingRP})
        if (*rp) { vkDestroyRenderPass(dev, *rp, nullptr); *rp = VK_NULL_HANDLE; }

    for (VkDescriptorSetLayout* l : {&m_frameLayout,&m_matLayout,&m_gbufLayout,
                                      &m_ssaoLayout,&m_ssaoBlurLayout,&m_lightLayout,&m_hdrLayout})
        if (*l) { vkDestroyDescriptorSetLayout(dev, *l, nullptr); *l = VK_NULL_HANDLE; }

    for (auto& f : m_frames) {
        m_ctx->destroyBuffer(f.frameUBO);
        m_ctx->destroyBuffer(f.lightSSBO);
        m_ctx->destroyBuffer(f.ssaoUBO);
    }

    m_ctx->destroyImage(m_ssaoNoise);
    if (m_ssaoNoiseSampler) { vkDestroySampler(dev, m_ssaoNoiseSampler, nullptr); m_ssaoNoiseSampler=VK_NULL_HANDLE; }

    // ── New systems cleanup ──────────────────────────────────────────────────
    m_gpuCulling.reset();
    m_shadowSystem.reset();
    m_iblProbe.reset();
    m_renderGraph.reset();

    // Fallback IBL resources
    if (m_fallbackCubeView)    { vkDestroyImageView(dev, m_fallbackCubeView, nullptr);    m_fallbackCubeView = VK_NULL_HANDLE; }
    if (m_fallbackCubeSampler) { vkDestroySampler  (dev, m_fallbackCubeSampler, nullptr); m_fallbackCubeSampler = VK_NULL_HANDLE; }
    if (m_fallbackCubemap.image) m_ctx->destroyImage(m_fallbackCubemap);

    for (auto& fx : m_framesExt)
        m_ctx->destroyBuffer(fx.shadowUBO);

    for (VkDescriptorSetLayout* l : {&m_iblLayout, &m_shadowLayout})
        if (*l) { vkDestroyDescriptorSetLayout(dev, *l, nullptr); *l = VK_NULL_HANDLE; }

    m_whiteTexture.reset();
    m_flatNormalTexture.reset();

    if (m_descPool)      { vkDestroyDescriptorPool(dev, m_descPool,     nullptr); m_descPool=VK_NULL_HANDLE; }
    if (m_pipelineCache) { vkDestroyPipelineCache (dev, m_pipelineCache, nullptr); m_pipelineCache=VK_NULL_HANDLE; }

    m_swapchain.reset();
    if (m_surface) { vkDestroySurfaceKHR(m_ctx->instance(), m_surface, nullptr); m_surface=VK_NULL_HANDLE; }
    m_ctx.reset();
}

// ─────────────────────────────────────────────────────────────────────────────
// Descriptor pool
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::createDescriptorPool() {
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         512 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 512 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,          64 },
    };
    VkDescriptorPoolCreateInfo ci{};
    ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.maxSets       = MAX_DESCRIPTOR_SETS;
    ci.poolSizeCount = 3;
    ci.pPoolSizes    = sizes;
    VK_CHECK(vkCreateDescriptorPool(m_ctx->device(), &ci, nullptr, &m_descPool));
}

// ─────────────────────────────────────────────────────────────────────────────
// Descriptor set layouts
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::createDescriptorLayouts() {
    auto make = [this](std::initializer_list<VkDescriptorSetLayoutBinding> bindings) {
        std::vector<VkDescriptorSetLayoutBinding> b(bindings);
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = static_cast<uint32_t>(b.size());
        ci.pBindings    = b.data();
        VkDescriptorSetLayout l;
        VK_CHECK(vkCreateDescriptorSetLayout(m_ctx->device(), &ci, nullptr, &l));
        return l;
    };
    using B = VkDescriptorSetLayoutBinding;
    constexpr auto UBO  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    constexpr auto CIS  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    constexpr auto SSBO = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    constexpr auto VS   = VK_SHADER_STAGE_VERTEX_BIT;
    constexpr auto FS   = VK_SHADER_STAGE_FRAGMENT_BIT;

    m_frameLayout    = make({ B{0,UBO,1,VS|FS,nullptr} });
    m_matLayout      = make({ B{0,CIS,1,FS,nullptr}, B{1,CIS,1,FS,nullptr}, B{2,CIS,1,FS,nullptr},
                               B{3,CIS,1,FS,nullptr}, B{4,CIS,1,FS,nullptr}, B{5,UBO,1,FS,nullptr} });
    m_gbufLayout     = make({ B{0,CIS,1,FS,nullptr}, B{1,CIS,1,FS,nullptr}, B{2,CIS,1,FS,nullptr},
                               B{3,CIS,1,FS,nullptr}, B{4,CIS,1,FS,nullptr}, B{5,CIS,1,FS,nullptr},
                               B{6,CIS,1,FS,nullptr} });
    m_ssaoLayout     = make({ B{0,CIS,1,FS,nullptr}, B{1,CIS,1,FS,nullptr},
                               B{2,CIS,1,FS,nullptr}, B{3,UBO,1,FS,nullptr} });
    m_ssaoBlurLayout = make({ B{0,CIS,1,FS,nullptr}, B{1,CIS,1,FS,nullptr} });
    m_lightLayout    = make({ B{0,SSBO,1,FS,nullptr} });
    m_hdrLayout      = make({ B{0,CIS,1,FS,nullptr} });

    // IBL and shadow layouts created here (before pipeline creation) so
    // createLightingPipeline can include all 5 set layouts.
    m_iblLayout = make({
        B{0,CIS,1,FS,nullptr},   // irradianceMap
        B{1,CIS,1,FS,nullptr},   // prefilteredMap
        B{2,CIS,1,FS,nullptr},   // brdfLUT
    });
    m_shadowLayout = make({
        B{0,CIS,1,FS,nullptr},   // sampler2DArrayShadow
        B{1,UBO,1,FS,nullptr},   // ShadowUBO
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Render passes
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::createGeometryRenderPass() {
    // Descriptions are filled by gbuffer_attachment_descs() which returns
    // finalLayout=SHADER_READ_ONLY for color and DEPTH_STENCIL_READ_ONLY for depth.
    // We override the depth finalLayout to DEPTH_STENCIL_READ_ONLY_OPTIMAL so the
    // SSAO and lighting passes can sample from it directly.
    VkAttachmentDescription atts[GBUFFER_COLOR_COUNT + 1];
    uint32_t count;
    gbuffer_attachment_descs(m_gbuffer, atts, &count);

    VkAttachmentReference colorRefs[GBUFFER_COLOR_COUNT];
    for (uint32_t i = 0; i < GBUFFER_COLOR_COUNT; ++i)
        colorRefs[i] = {i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{GBUFFER_COLOR_COUNT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription sub{};
    sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount    = GBUFFER_COLOR_COUNT;
    sub.pColorAttachments       = colorRefs;
    sub.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL; deps[0].dstSubpass = 0;
    deps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].srcSubpass    = 0; deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rpCI{};
    rpCI.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpCI.attachmentCount=count; rpCI.pAttachments=atts;
    rpCI.subpassCount=1;        rpCI.pSubpasses=&sub;
    rpCI.dependencyCount=2;     rpCI.pDependencies=deps;
    VK_CHECK(vkCreateRenderPass(m_ctx->device(), &rpCI, nullptr, &m_geomRP), "Geometry RP");
}

static VkRenderPass makeSingleAttachRP(VkDevice dev, VkFormat fmt,
                                        VkImageLayout finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
{
    VkAttachmentDescription att{};
    att.format=fmt; att.samples=VK_SAMPLE_COUNT_1_BIT;
    att.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; att.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; att.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED; att.finalLayout=finalLayout;

    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount=1; sub.pColorAttachments=&ref;

    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass=VK_SUBPASS_EXTERNAL; deps[0].dstSubpass=0;
    deps[0].srcStageMask=VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].srcAccessMask=VK_ACCESS_SHADER_READ_BIT; deps[0].dstAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].srcSubpass=0; deps[1].dstSubpass=VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask=VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT; deps[1].dstAccessMask=VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo ci{};
    ci.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.attachmentCount=1; ci.pAttachments=&att;
    ci.subpassCount=1;    ci.pSubpasses=&sub;
    ci.dependencyCount=2; ci.pDependencies=deps;
    VkRenderPass rp;
    VK_CHECK(vkCreateRenderPass(dev, &ci, nullptr, &rp));
    return rp;
}

void Renderer::createSSAORenderPass()     { m_ssaoRP    = makeSingleAttachRP(m_ctx->device(), VK_FORMAT_R8_UNORM); }
void Renderer::createSSAOBlurRenderPass() { m_ssaoBlurRP = makeSingleAttachRP(m_ctx->device(), VK_FORMAT_R8_UNORM); }
void Renderer::createLightingRenderPass() { m_lightingRP = makeSingleAttachRP(m_ctx->device(), VK_FORMAT_R16G16B16A16_SFLOAT); }

// ─────────────────────────────────────────────────────────────────────────────
// Size-dependent resources (G-buffer, intermediate images, framebuffers)
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::createSizeDependentResources() {
    VkExtent2D ext = m_swapchain->extent();
    uint32_t w = ext.width, h = ext.height;

    gbuffer_create(m_gbuffer, *m_ctx, w, h);

    if (!m_screenSampler) {
        VkSamplerCreateInfo si{};
        si.sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter=VK_FILTER_NEAREST; si.minFilter=VK_FILTER_NEAREST;
        si.addressModeU=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST; si.maxLod=1.f;
        VK_CHECK(vkCreateSampler(m_ctx->device(), &si, nullptr, &m_screenSampler));
    }

    auto makeImg = [&](VkFormat fmt, VkImageUsageFlags usage, AllocatedImage& img) {
        img = m_ctx->createImage(w, h, 1, VK_SAMPLE_COUNT_1_BIT, fmt,
            VK_IMAGE_TILING_OPTIMAL, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        img.mipLevels = 1;
        m_ctx->createImageView(img, VK_IMAGE_ASPECT_COLOR_BIT);
    };

    constexpr VkImageUsageFlags sampledColor =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    makeImg(VK_FORMAT_R8_UNORM,               sampledColor, m_ssaoRaw);
    makeImg(VK_FORMAT_R8_UNORM,               sampledColor, m_ssaoBlur);
    makeImg(VK_FORMAT_R16G16B16A16_SFLOAT,    sampledColor, m_hdrImage);

    // ── Framebuffers ──────────────────────────────────────────────────────────
    {   // Geometry: 5 G-buffer color views + depth
        std::array<VkImageView, GBUFFER_COLOR_COUNT + 1> views;
        for (uint32_t i = 0; i < GBUFFER_COLOR_COUNT; ++i) views[i] = m_gbuffer.colorView(i);
        views[GBUFFER_COLOR_COUNT] = m_gbuffer.depthView();
        VkFramebufferCreateInfo fi{};
        fi.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fi.renderPass=m_geomRP; fi.attachmentCount=GBUFFER_COLOR_COUNT+1;
        fi.pAttachments=views.data(); fi.width=w; fi.height=h; fi.layers=1;
        VK_CHECK(vkCreateFramebuffer(m_ctx->device(), &fi, nullptr, &m_geomFB));
    }

    auto make1FB = [&](VkRenderPass rp, VkImageView v, VkFramebuffer& fb) {
        VkFramebufferCreateInfo fi{};
        fi.sType=VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fi.renderPass=rp; fi.attachmentCount=1; fi.pAttachments=&v;
        fi.width=w; fi.height=h; fi.layers=1;
        VK_CHECK(vkCreateFramebuffer(m_ctx->device(), &fi, nullptr, &fb));
    };
    make1FB(m_ssaoRP,     m_ssaoRaw.view,  m_ssaoFB);
    make1FB(m_ssaoBlurRP, m_ssaoBlur.view, m_ssaoBlurFB);
    make1FB(m_lightingRP, m_hdrImage.view, m_lightingFB);
}

void Renderer::destroySizeDependentResources() {
    auto dev = m_ctx->device();
    for (VkFramebuffer* fb : {&m_geomFB,&m_ssaoFB,&m_ssaoBlurFB,&m_lightingFB})
        if (*fb) { vkDestroyFramebuffer(dev, *fb, nullptr); *fb=VK_NULL_HANDLE; }
    gbuffer_destroy(m_gbuffer, *m_ctx);
    m_ctx->destroyImage(m_ssaoRaw);
    m_ctx->destroyImage(m_ssaoBlur);
    m_ctx->destroyImage(m_hdrImage);
    if (m_screenSampler) { vkDestroySampler(dev, m_screenSampler, nullptr); m_screenSampler=VK_NULL_HANDLE; }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-frame buffers
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::createPerFrameBuffers() {
    constexpr VkMemoryPropertyFlags hostFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (auto& f : m_frames) {
        f.frameUBO  = m_ctx->createBuffer(sizeof(FrameUBO),   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hostFlags);
        f.lightSSBO = m_ctx->createBuffer(sizeof(LightSSBO),  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostFlags);
        f.ssaoUBO   = m_ctx->createBuffer(sizeof(SSAOParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hostFlags);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SSAO hemisphere kernel + noise texture
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::createSSAOKernel() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (uint32_t i = 0; i < 32; ++i) {
        Vec3 s{ dist(rng)*2.f-1.f, dist(rng)*2.f-1.f, dist(rng) };
        s = glm::normalize(s) * dist(rng);
        float t = static_cast<float>(i) / 32.f;
        s *= glm::mix(0.1f, 1.f, t*t);
        m_ssaoKernel[i] = Vec4(s, 0.f);
    }

    std::array<glm::vec2, 16> noise;
    for (auto& n : noise) n = glm::normalize(glm::vec2(dist(rng)*2.f-1.f, dist(rng)*2.f-1.f));

    VkDeviceSize sz = sizeof(noise);
    auto staging = m_ctx->createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(staging.mapped, noise.data(), static_cast<size_t>(sz));

    m_ssaoNoise = m_ctx->createImage(4, 4, 1, VK_SAMPLE_COUNT_1_BIT,
        VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_ssaoNoise.mipLevels = 1;
    m_ctx->transitionImageLayout(m_ssaoNoise.image, VK_FORMAT_R16G16_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1);
    m_ctx->copyBufferToImage(staging.buffer, m_ssaoNoise.image, 4, 4);
    m_ctx->transitionImageLayout(m_ssaoNoise.image, VK_FORMAT_R16G16_SFLOAT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 1);
    m_ctx->destroyBuffer(staging);
    m_ctx->createImageView(m_ssaoNoise, VK_IMAGE_ASPECT_COLOR_BIT);

    VkSamplerCreateInfo si{};
    si.sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter=VK_FILTER_NEAREST; si.minFilter=VK_FILTER_NEAREST;
    si.addressModeU=VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeV=VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeW=VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST; si.maxLod=1.f;
    VK_CHECK(vkCreateSampler(m_ctx->device(), &si, nullptr, &m_ssaoNoiseSampler));
}

// ─────────────────────────────────────────────────────────────────────────────
// Descriptor sets — allocate and write all fixed sets
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::allocateAndWriteDescriptorSets() {
    auto alloc = [&](VkDescriptorSetLayout layout) {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool=m_descPool; ai.descriptorSetCount=1; ai.pSetLayouts=&layout;
        VkDescriptorSet ds;
        VK_CHECK(vkAllocateDescriptorSets(m_ctx->device(), &ai, &ds));
        return ds;
    };

    auto wUBO = [&](VkDescriptorSet ds, uint32_t bind, VkBuffer buf, VkDeviceSize sz) {
        VkDescriptorBufferInfo bi{buf,0,sz};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet=ds; w.dstBinding=bind; w.descriptorCount=1;
        w.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w.pBufferInfo=&bi;
        vkUpdateDescriptorSets(m_ctx->device(),1,&w,0,nullptr);
    };
    auto wSSBO = [&](VkDescriptorSet ds, uint32_t bind, VkBuffer buf, VkDeviceSize sz) {
        VkDescriptorBufferInfo bi{buf,0,sz};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet=ds; w.dstBinding=bind; w.descriptorCount=1;
        w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo=&bi;
        vkUpdateDescriptorSets(m_ctx->device(),1,&w,0,nullptr);
    };
    auto wImg = [&](VkDescriptorSet ds, uint32_t bind, VkImageView view, VkSampler samp,
                     VkImageLayout layout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        VkDescriptorImageInfo ii{samp,view,layout};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet=ds; w.dstBinding=bind; w.descriptorCount=1;
        w.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo=&ii;
        vkUpdateDescriptorSets(m_ctx->device(),1,&w,0,nullptr);
    };

    for (uint32_t fi=0; fi<MAX_FRAMES_IN_FLIGHT; ++fi) {
        auto& f = m_frames[fi];
        f.frameSet    = alloc(m_frameLayout);
        f.gbufSet     = alloc(m_gbufLayout);
        f.ssaoSet     = alloc(m_ssaoLayout);
        f.ssaoBlurSet = alloc(m_ssaoBlurLayout);
        f.lightSet    = alloc(m_lightLayout);
        f.hdrSet      = alloc(m_hdrLayout);

        wUBO(f.frameSet, 0, f.frameUBO.buffer, sizeof(FrameUBO));

        for (uint32_t s=0; s<GBUFFER_COLOR_COUNT; ++s)
            wImg(f.gbufSet, s, m_gbuffer.colorView(s), m_gbuffer.sampler);
        wImg(f.gbufSet, 5, m_gbuffer.depthView(), m_gbuffer.sampler,
             VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        wImg(f.gbufSet, 6, m_ssaoBlur.view, m_screenSampler);

        wImg(f.ssaoSet, 0, m_gbuffer.colorView(GBUF_NORMAL), m_gbuffer.sampler);
        wImg(f.ssaoSet, 1, m_gbuffer.depthView(), m_gbuffer.sampler,
             VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        wImg(f.ssaoSet, 2, m_ssaoNoise.view, m_ssaoNoiseSampler);
        wUBO(f.ssaoSet, 3, f.ssaoUBO.buffer, sizeof(SSAOParams));

        wImg(f.ssaoBlurSet, 0, m_ssaoRaw.view,          m_screenSampler);
        wImg(f.ssaoBlurSet, 1, m_gbuffer.depthView(), m_gbuffer.sampler,
             VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);

        wSSBO(f.lightSet, 0, f.lightSSBO.buffer, sizeof(LightSSBO));
        wImg(f.hdrSet,    0, m_hdrImage.view,    m_screenSampler);

    }

    // Fallback IBL sets (1×1 black cubemaps) used until a real HDR probe is loaded.
    createFallbackIBLDescriptors();
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline creation
// ─────────────────────────────────────────────────────────────────────────────
VkShaderModule Renderer::createShaderModule(const std::filesystem::path& path) {
    auto code = readFile(path);
    VkShaderModuleCreateInfo ci{};
    ci.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize=code.size(); ci.pCode=reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(m_ctx->device(), &ci, nullptr, &mod));
    return mod;
}

// Fullscreen-triangle pipeline — no vertex buffers, no depth test.
static VkPipeline buildScreenPipeline(VkDevice dev, VkPipelineCache cache,
                                       VkShaderModule vert, VkShaderModule frag,
                                       VkPipelineLayout layout, VkRenderPass rp,
                                       uint32_t colorCount = 1)
{
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module=vert; stages[0].pName="main";
    stages[1]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=frag; stages[1].pName="main";

    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vp{0,0,1,1,0,1}; VkRect2D sc{{0,0},{1,1}};
    VkPipelineViewportStateCreateInfo vpS{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vpS.viewportCount=1; vpS.pViewports=&vp; vpS.scissorCount=1; vpS.pScissors=&sc;
    VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rast.polygonMode=VK_POLYGON_MODE_FILL; rast.cullMode=VK_CULL_MODE_NONE;
    rast.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE; rast.lineWidth=1.f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    std::vector<VkPipelineColorBlendAttachmentState> blends(colorCount);
    for (auto& b : blends) b.colorWriteMask=0xF;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount=colorCount; blend.pAttachments=blends.data();
    VkDynamicState dynArr[2]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount=2; dyn.pDynamicStates=dynArr;

    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.stageCount=2; pci.pStages=stages;
    pci.pVertexInputState=&vi; pci.pInputAssemblyState=&ia; pci.pViewportState=&vpS;
    pci.pRasterizationState=&rast; pci.pMultisampleState=&ms;
    pci.pDepthStencilState=&ds;   pci.pColorBlendState=&blend; pci.pDynamicState=&dyn;
    pci.layout=layout; pci.renderPass=rp;

    VkPipeline pipe;
    VK_CHECK(vkCreateGraphicsPipelines(dev,cache,1,&pci,nullptr,&pipe));
    return pipe;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fallback IBL descriptors — 1×1 black cubemaps so set 3 is always bound.
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::createFallbackIBLDescriptors() {
    auto dev = m_ctx->device();

    // Destroy previous resources (called again on resize)
    if (m_fallbackCubeView)    { vkDestroyImageView(dev, m_fallbackCubeView, nullptr);   m_fallbackCubeView = VK_NULL_HANDLE; }
    if (m_fallbackCubeSampler) { vkDestroySampler  (dev, m_fallbackCubeSampler, nullptr); m_fallbackCubeSampler = VK_NULL_HANDLE; }
    if (m_fallbackCubemap.image) m_ctx->destroyImage(m_fallbackCubemap);

    // ── 1×1 black 6-layer cubemap ────────────────────────────────────────────
    constexpr VkFormat CUBE_FMT = VK_FORMAT_R8G8B8A8_UNORM;
    constexpr uint32_t CUBE_LAYERS = 6;

    // Create via raw VMA/Vulkan — Context::createImage doesn't set CUBE_COMPATIBLE_BIT.
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.flags       = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    ici.imageType   = VK_IMAGE_TYPE_2D;
    ici.format      = CUBE_FMT;
    ici.extent      = {1, 1, 1};
    ici.mipLevels   = 1;
    ici.arrayLayers = CUBE_LAYERS;
    ici.samples     = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ici.usage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo aci{}; aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    m_fallbackCubemap.format    = CUBE_FMT;
    m_fallbackCubemap.mipLevels = 1;
    VK_CHECK(vmaCreateImage(m_ctx->allocator(), &ici, &aci,
        &m_fallbackCubemap.image, &m_fallbackCubemap.allocation, nullptr), "fallback cubemap");

    // Upload 6 × 1×1 black pixels
    constexpr VkDeviceSize CUBE_SZ = 4 * CUBE_LAYERS; // RGBA8 × 6
    auto staging = m_ctx->createBuffer(CUBE_SZ, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memset(staging.mapped, 0, static_cast<size_t>(CUBE_SZ));

    VkCommandBuffer cmd = m_ctx->beginSingleTimeCommands();

    // Transition all 6 layers to TRANSFER_DST
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = m_fallbackCubemap.image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, CUBE_LAYERS};
    barrier.srcAccessMask = 0; barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    // Copy one pixel per face
    std::array<VkBufferImageCopy, CUBE_LAYERS> copies;
    for (uint32_t face = 0; face < CUBE_LAYERS; ++face) {
        copies[face] = {};
        copies[face].bufferOffset = face * 4;
        copies[face].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, face, 1};
        copies[face].imageExtent = {1, 1, 1};
    }
    vkCmdCopyBufferToImage(cmd, staging.buffer, m_fallbackCubemap.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, CUBE_LAYERS, copies.data());

    // Transition to SHADER_READ_ONLY
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);

    m_ctx->endSingleTimeCommands(cmd);
    m_ctx->destroyBuffer(staging);

    // Cube image view
    VkImageViewCreateInfo ivCI{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ivCI.image    = m_fallbackCubemap.image;
    ivCI.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    ivCI.format   = CUBE_FMT;
    ivCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, CUBE_LAYERS};
    VK_CHECK(vkCreateImageView(dev, &ivCI, nullptr, &m_fallbackCubeView), "fallback cube view");

    // Sampler
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_NEAREST; si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 1.f;
    VK_CHECK(vkCreateSampler(dev, &si, nullptr, &m_fallbackCubeSampler), "fallback cube sampler");

    // ── Allocate and write fallback IBL descriptor sets ──────────────────────
    // (re-use m_whiteTexture for the BRDF LUT slot)
    for (uint32_t fi = 0; fi < MAX_FRAMES_IN_FLIGHT; ++fi) {
        if (m_fallbackIBLSets[fi] == VK_NULL_HANDLE) {
            VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            ai.descriptorPool = m_descPool; ai.descriptorSetCount = 1; ai.pSetLayouts = &m_iblLayout;
            VK_CHECK(vkAllocateDescriptorSets(dev, &ai, &m_fallbackIBLSets[fi]));
        }
        auto writeCube = [&](uint32_t bind, VkImageView view) {
            VkDescriptorImageInfo ii{m_fallbackCubeSampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet=m_fallbackIBLSets[fi]; w.dstBinding=bind; w.descriptorCount=1;
            w.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo=&ii;
            vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
        };
        writeCube(0, m_fallbackCubeView);   // irradiance
        writeCube(1, m_fallbackCubeView);   // prefiltered
        // BRDF LUT — use white 2D texture as neutral fallback
        VkDescriptorImageInfo ii2 = m_whiteTexture->descriptorInfo();
        VkWriteDescriptorSet w2{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w2.dstSet=m_fallbackIBLSets[fi]; w2.dstBinding=2; w2.descriptorCount=1;
        w2.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w2.pImageInfo=&ii2;
        vkUpdateDescriptorSets(dev, 1, &w2, 0, nullptr);
    }
}

void Renderer::createGeometryPipeline() {
    VkPushConstantRange pc{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ModelPushConstant)};
    VkDescriptorSetLayout sets[2]={m_frameLayout,m_matLayout};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount=2; li.pSetLayouts=sets;
    li.pushConstantRangeCount=1; li.pPushConstantRanges=&pc;
    VK_CHECK(vkCreatePipelineLayout(m_ctx->device(),&li,nullptr,&m_geomPipeLayout));

    auto vert=createShaderModule(m_settings.shaderDir/"geometry.vert.spv");
    auto frag=createShaderModule(m_settings.shaderDir/"geometry.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage=VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module=vert; stages[0].pName="main";
    stages[1]={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage=VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module=frag; stages[1].pName="main";

    auto bind=Vertex::getBindingDescription();
    auto attr=Vertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount=1;   vi.pVertexBindingDescriptions=&bind;
    vi.vertexAttributeDescriptionCount=4; vi.pVertexAttributeDescriptions=attr.data();

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vp{0,0,1,1,0,1}; VkRect2D sc{{0,0},{1,1}};
    VkPipelineViewportStateCreateInfo vpS{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vpS.viewportCount=1; vpS.pViewports=&vp; vpS.scissorCount=1; vpS.pScissors=&sc;
    VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rast.polygonMode=m_settings.wireframe?VK_POLYGON_MODE_LINE:VK_POLYGON_MODE_FILL;
    rast.cullMode=VK_CULL_MODE_BACK_BIT; rast.frontFace=VK_FRONT_FACE_COUNTER_CLOCKWISE; rast.lineWidth=1.f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable=VK_TRUE; ds.depthWriteEnable=VK_TRUE; ds.depthCompareOp=VK_COMPARE_OP_LESS;
    std::array<VkPipelineColorBlendAttachmentState,GBUFFER_COLOR_COUNT> blends{};
    for (auto& b : blends) b.colorWriteMask=0xF;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount=GBUFFER_COLOR_COUNT; blend.pAttachments=blends.data();
    VkDynamicState dynArr[2]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount=2; dyn.pDynamicStates=dynArr;

    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.stageCount=2; pci.pStages=stages;
    pci.pVertexInputState=&vi; pci.pInputAssemblyState=&ia; pci.pViewportState=&vpS;
    pci.pRasterizationState=&rast; pci.pMultisampleState=&ms;
    pci.pDepthStencilState=&ds;   pci.pColorBlendState=&blend; pci.pDynamicState=&dyn;
    pci.layout=m_geomPipeLayout; pci.renderPass=m_geomRP;
    VK_CHECK(vkCreateGraphicsPipelines(m_ctx->device(),m_pipelineCache,1,&pci,nullptr,&m_geomPipeline));
    vkDestroyShaderModule(m_ctx->device(),vert,nullptr);
    vkDestroyShaderModule(m_ctx->device(),frag,nullptr);
}

void Renderer::createSSAOPipeline() {
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount=1; li.pSetLayouts=&m_ssaoLayout;
    VK_CHECK(vkCreatePipelineLayout(m_ctx->device(),&li,nullptr,&m_ssaoPipeLayout));
    auto vert=createShaderModule(m_settings.shaderDir/"fullscreen_quad.vert.spv");
    auto frag=createShaderModule(m_settings.shaderDir/"ssao.frag.spv");
    m_ssaoPipeline=buildScreenPipeline(m_ctx->device(),m_pipelineCache,vert,frag,m_ssaoPipeLayout,m_ssaoRP);
    vkDestroyShaderModule(m_ctx->device(),vert,nullptr);
    vkDestroyShaderModule(m_ctx->device(),frag,nullptr);
}

void Renderer::createSSAOBlurPipeline() {
    VkPushConstantRange pc{VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(BlurPC)};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount=1; li.pSetLayouts=&m_ssaoBlurLayout;
    li.pushConstantRangeCount=1; li.pPushConstantRanges=&pc;
    VK_CHECK(vkCreatePipelineLayout(m_ctx->device(),&li,nullptr,&m_ssaoBlurPipeLayout));
    auto vert=createShaderModule(m_settings.shaderDir/"fullscreen_quad.vert.spv");
    auto frag=createShaderModule(m_settings.shaderDir/"ssao_blur.frag.spv");
    m_ssaoBlurPipeline=buildScreenPipeline(m_ctx->device(),m_pipelineCache,vert,frag,m_ssaoBlurPipeLayout,m_ssaoBlurRP);
    vkDestroyShaderModule(m_ctx->device(),vert,nullptr);
    vkDestroyShaderModule(m_ctx->device(),frag,nullptr);
}

void Renderer::createLightingPipeline() {
    // All 5 sets the shader accesses must be declared in the pipeline layout,
    // even if some are conditionally populated at runtime.
    VkDescriptorSetLayout sets[5]={m_gbufLayout,m_frameLayout,m_lightLayout,m_iblLayout,m_shadowLayout};
    VkPushConstantRange pc{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(LightingPC)};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount=5; li.pSetLayouts=sets;
    li.pushConstantRangeCount=1; li.pPushConstantRanges=&pc;
    VK_CHECK(vkCreatePipelineLayout(m_ctx->device(),&li,nullptr,&m_lightPipeLayout));
    auto vert=createShaderModule(m_settings.shaderDir/"fullscreen_quad.vert.spv");
    auto frag=createShaderModule(m_settings.shaderDir/"lighting.frag.spv");
    m_lightPipeline=buildScreenPipeline(m_ctx->device(),m_pipelineCache,vert,frag,m_lightPipeLayout,m_lightingRP);
    vkDestroyShaderModule(m_ctx->device(),vert,nullptr);
    vkDestroyShaderModule(m_ctx->device(),frag,nullptr);
}

void Renderer::createTonemapPipeline() {
    VkPushConstantRange pc{VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(TonemapPC)};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount=1; li.pSetLayouts=&m_hdrLayout;
    li.pushConstantRangeCount=1; li.pPushConstantRanges=&pc;
    VK_CHECK(vkCreatePipelineLayout(m_ctx->device(),&li,nullptr,&m_tonemapPipeLayout));
    auto vert=createShaderModule(m_settings.shaderDir/"fullscreen_quad.vert.spv");
    auto frag=createShaderModule(m_settings.shaderDir/"tonemap.frag.spv");
    m_tonemapPipeline=buildScreenPipeline(m_ctx->device(),m_pipelineCache,vert,frag,
                                           m_tonemapPipeLayout,m_swapchain->renderPass());
    vkDestroyShaderModule(m_ctx->device(),vert,nullptr);
    vkDestroyShaderModule(m_ctx->device(),frag,nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-mesh GPU data management
// ─────────────────────────────────────────────────────────────────────────────
MeshGPUData& Renderer::getOrCreateMeshData(Mesh* mesh) {
    // Lock only for the initial allocation path. Once initialized, the entry
    // is stable and the lock is released immediately so workers don't stall.
    {
        std::lock_guard<std::mutex> lock(m_meshDataMutex);
        auto& d = m_meshData[mesh];
        if (d.initialized) return d;

        constexpr VkMemoryPropertyFlags hf =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t fi=0; fi<MAX_FRAMES_IN_FLIGHT; ++fi)
            d.matUBOs[fi] = m_ctx->createBuffer(sizeof(MaterialUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, hf);

        std::array<VkDescriptorSetLayout,MAX_FRAMES_IN_FLIGHT> layouts;
        layouts.fill(m_matLayout);
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool=m_descPool; ai.descriptorSetCount=MAX_FRAMES_IN_FLIGHT; ai.pSetLayouts=layouts.data();
        VK_CHECK(vkAllocateDescriptorSets(m_ctx->device(),&ai,d.matDescSets.data()));
        d.initialized=true;
        return d;
    }
}

void Renderer::updateMaterialDescriptors(MeshGPUData& data, PBRMaterial* mat, uint32_t fi) {
    if (!mat->isFrameDirty(fi) && (data.writtenFrames & (1u<<fi))) return;
    std::memcpy(data.matUBOs[fi].mapped, &mat->ubo(), sizeof(MaterialUBO));

    auto pick = [&](uint32_t slot, std::shared_ptr<Texture> fb) {
        auto t = mat->getTexture(slot);
        return t ? t->descriptorInfo() : fb->descriptorInfo();
    };
    VkDescriptorImageInfo imgs[MAX_TEXTURES_PER_MAT] = {
        pick(PBRMaterial::ALBEDO,    m_whiteTexture),
        pick(PBRMaterial::NORMAL,    m_flatNormalTexture),
        pick(PBRMaterial::METALROUGH,m_whiteTexture),
        pick(PBRMaterial::EMISSIVE,  m_whiteTexture),
        pick(PBRMaterial::AO,        m_whiteTexture),
    };
    VkDescriptorBufferInfo bufI{data.matUBOs[fi].buffer,0,sizeof(MaterialUBO)};

    std::array<VkWriteDescriptorSet,MAX_TEXTURES_PER_MAT+1> w{};
    for (uint32_t i=0;i<MAX_TEXTURES_PER_MAT;++i) {
        w[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet=data.matDescSets[fi]; w[i].dstBinding=i; w[i].descriptorCount=1;
        w[i].descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[i].pImageInfo=&imgs[i];
    }
    w[MAX_TEXTURES_PER_MAT]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[MAX_TEXTURES_PER_MAT].dstSet=data.matDescSets[fi]; w[MAX_TEXTURES_PER_MAT].dstBinding=MAX_TEXTURES_PER_MAT;
    w[MAX_TEXTURES_PER_MAT].descriptorCount=1;
    w[MAX_TEXTURES_PER_MAT].descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[MAX_TEXTURES_PER_MAT].pBufferInfo=&bufI;
    vkUpdateDescriptorSets(m_ctx->device(),static_cast<uint32_t>(w.size()),w.data(),0,nullptr);
    data.writtenFrames |= (1u<<fi);
    mat->markClean(fi);
}

void Renderer::uploadMeshToGPU(Mesh& mesh) {
    if (mesh.vertices().empty()) { mesh.gpuReady=true; return; }
    VkDeviceSize vs=sizeof(Vertex)*mesh.vertices().size();
    VkDeviceSize is=sizeof(uint32_t)*mesh.indices().size();
    mesh.vertexBuffer=m_ctx->uploadBuffer(mesh.vertices().data(),vs,VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    if (!mesh.indices().empty())
        mesh.indexBuffer=m_ctx->uploadBuffer(mesh.indices().data(),is,VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    mesh.gpuReady=true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Worker command pools
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::createWorkerPools(uint32_t count) {
    m_workerPools.resize(count);
    m_workerCmds.resize(count);
    for (uint32_t w=0; w<count; ++w)
        for (uint32_t f=0; f<MAX_FRAMES_IN_FLIGHT; ++f) {
            VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            ci.queueFamilyIndex=*m_ctx->queueFamilies().graphics;
            ci.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            VK_CHECK(vkCreateCommandPool(m_ctx->device(),&ci,nullptr,&m_workerPools[w][f]));
            VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ai.commandPool=m_workerPools[w][f]; ai.level=VK_COMMAND_BUFFER_LEVEL_SECONDARY; ai.commandBufferCount=1;
            VK_CHECK(vkAllocateCommandBuffers(m_ctx->device(),&ai,&m_workerCmds[w][f]));
        }
}

void Renderer::destroyWorkerPools() {
    for (auto& pools : m_workerPools)
        for (auto& p : pools)
            if (p) { vkDestroyCommandPool(m_ctx->device(),p,nullptr); p=VK_NULL_HANDLE; }
    m_workerPools.clear(); m_workerCmds.clear();
}

// ─────────────────────────────────────────────────────────────────────────────
// Draw a single mesh (all sub-meshes) into the geometry pass
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::drawMesh(VkCommandBuffer cmd, Mesh& mesh, uint32_t fi, uint32_t& dc) {
    if (!mesh.gpuReady || mesh.vertices().empty()) return;
    VkDeviceSize offset=0;
    vkCmdBindVertexBuffers(cmd,0,1,&mesh.vertexBuffer.buffer,&offset);
    if (mesh.indexBuffer.buffer!=VK_NULL_HANDLE)
        vkCmdBindIndexBuffer(cmd,mesh.indexBuffer.buffer,0,VK_INDEX_TYPE_UINT32);

    ModelPushConstant pc{mesh.modelMatrix(),mesh.normalMatrix()};
    vkCmdPushConstants(cmd,m_geomPipeLayout,VK_SHADER_STAGE_VERTEX_BIT,0,sizeof(pc),&pc);

    for (auto& sub : mesh.subMeshes()) {
        auto* mat=sub.material.get();
        if (!mat) continue;
        auto& data=getOrCreateMeshData(&mesh);
        updateMaterialDescriptors(data,mat,fi);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 m_geomPipeLayout,1,1,&data.matDescSets[fi],0,nullptr);
        if (mesh.indexBuffer.buffer!=VK_NULL_HANDLE)
            vkCmdDrawIndexed(cmd,sub.indexCount,1,sub.indexOffset,0,0);
        else
            vkCmdDraw(cmd,static_cast<uint32_t>(mesh.vertices().size()),1,0,0);
        ++dc;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-threaded secondary command buffer for geometry batch
// ─────────────────────────────────────────────────────────────────────────────
VkCommandBuffer Renderer::recordSecondaryBatch(const std::vector<Mesh*>& batch,
                                                uint32_t fi, uint32_t workerIdx)
{
    if (batch.empty()) return VK_NULL_HANDLE;
    VkCommandBuffer cmd=m_workerCmds[workerIdx][fi];
    vkResetCommandBuffer(cmd,0);

    VkCommandBufferInheritanceInfo inh{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
    inh.renderPass=m_geomRP; inh.subpass=0; inh.framebuffer=m_geomFB;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags=VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT|VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    bi.pInheritanceInfo=&inh;
    vkBeginCommandBuffer(cmd,&bi);

    VkExtent2D ext=m_swapchain->extent();
    VkViewport vp{0,0,(float)ext.width,(float)ext.height,0,1};
    vkCmdSetViewport(cmd,0,1,&vp);
    VkRect2D sc{{0,0},ext}; vkCmdSetScissor(cmd,0,1,&sc);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,m_geomPipeline);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,
                             m_geomPipeLayout,0,1,&m_frames[fi].frameSet,0,nullptr);
    uint32_t dummy=0;
    for (auto* m : batch) drawMesh(cmd,*m,fi,dummy);
    vkEndCommandBuffer(cmd);
    return cmd;
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-pass recording
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::recordGeometryPass(VkCommandBuffer cmd, const std::vector<Mesh*>& visible, uint32_t fi) {
    std::array<VkClearValue,GBUFFER_COLOR_COUNT+1> clears{};
    for (uint32_t i=0;i<GBUFFER_COLOR_COUNT;++i) clears[i].color={{0,0,0,1}};
    clears[GBUFFER_COLOR_COUNT].depthStencil={1.f,0};

    VkExtent2D ext=m_swapchain->extent();
    VkRenderPassBeginInfo rpBI{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpBI.renderPass=m_geomRP; rpBI.framebuffer=m_geomFB;
    rpBI.renderArea.extent=ext;
    rpBI.clearValueCount=static_cast<uint32_t>(clears.size());
    rpBI.pClearValues=clears.data();

    uint32_t wc=static_cast<uint32_t>(m_workerPools.size());
    uint32_t dc=static_cast<uint32_t>(visible.size());
    bool useSecondary = (wc>1 && dc>0);

    vkCmdBeginRenderPass(cmd,&rpBI,
        useSecondary?VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS:VK_SUBPASS_CONTENTS_INLINE);

    if (useSecondary) {
        uint32_t batchSz=(dc+wc-1)/wc;
        std::vector<std::future<VkCommandBuffer>> futures;
        for (uint32_t w=0; w<wc; ++w) {
            size_t s=w*batchSz, e=std::min<size_t>(s+batchSz,dc);
            if (s>=dc) break;
            std::vector<Mesh*> chunk(visible.begin()+s,visible.begin()+e);
            futures.push_back(m_threadPool->submit([this,chunk=std::move(chunk),fi,w]() mutable {
                return recordSecondaryBatch(chunk,fi,w);
            }));
        }
        std::vector<VkCommandBuffer> secondaries;
        uint32_t totalDC=0;
        for (auto& f : futures) { auto cb=f.get(); if(cb) secondaries.push_back(cb); ++totalDC; }
        if (!secondaries.empty())
            vkCmdExecuteCommands(cmd,static_cast<uint32_t>(secondaries.size()),secondaries.data());
        m_stats.drawCalls=totalDC;
    } else {
        VkViewport vp{0,0,(float)ext.width,(float)ext.height,0,1};
        vkCmdSetViewport(cmd,0,1,&vp);
        VkRect2D sc{{0,0},ext}; vkCmdSetScissor(cmd,0,1,&sc);
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,m_geomPipeline);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 m_geomPipeLayout,0,1,&m_frames[fi].frameSet,0,nullptr);
        uint32_t drawCount=0;
        for (auto* m : visible) drawMesh(cmd,*m,fi,drawCount);
        m_stats.drawCalls=drawCount;
    }
    vkCmdEndRenderPass(cmd);
}

void Renderer::recordSSAOPass(VkCommandBuffer cmd, uint32_t fi) {
    VkExtent2D ext=m_swapchain->extent();
    VkClearValue clr{}; clr.color={{1,1,1,1}};
    VkRenderPassBeginInfo rpBI{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpBI.renderPass=m_ssaoRP; rpBI.framebuffer=m_ssaoFB;
    rpBI.renderArea.extent=ext; rpBI.clearValueCount=1; rpBI.pClearValues=&clr;
    vkCmdBeginRenderPass(cmd,&rpBI,VK_SUBPASS_CONTENTS_INLINE);
    VkViewport vp{0,0,(float)ext.width,(float)ext.height,0,1};
    vkCmdSetViewport(cmd,0,1,&vp);
    VkRect2D sc{{0,0},ext}; vkCmdSetScissor(cmd,0,1,&sc);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,m_ssaoPipeline);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,
                             m_ssaoPipeLayout,0,1,&m_frames[fi].ssaoSet,0,nullptr);
    vkCmdDraw(cmd,3,1,0,0);
    vkCmdEndRenderPass(cmd);
}

void Renderer::recordSSAOBlurPass(VkCommandBuffer cmd, uint32_t fi) {
    VkExtent2D ext=m_swapchain->extent();
    VkClearValue clr{}; clr.color={{1,1,1,1}};
    VkRenderPassBeginInfo rpBI{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpBI.renderPass=m_ssaoBlurRP; rpBI.framebuffer=m_ssaoBlurFB;
    rpBI.renderArea.extent=ext; rpBI.clearValueCount=1; rpBI.pClearValues=&clr;
    vkCmdBeginRenderPass(cmd,&rpBI,VK_SUBPASS_CONTENTS_INLINE);
    VkViewport vp{0,0,(float)ext.width,(float)ext.height,0,1};
    vkCmdSetViewport(cmd,0,1,&vp);
    VkRect2D sc{{0,0},ext}; vkCmdSetScissor(cmd,0,1,&sc);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,m_ssaoBlurPipeline);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,
                             m_ssaoBlurPipeLayout,0,1,&m_frames[fi].ssaoBlurSet,0,nullptr);
    BlurPC blurPC;
    blurPC.texelSize={1.f/ext.width, 1.f/ext.height};
    vkCmdPushConstants(cmd,m_ssaoBlurPipeLayout,VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(BlurPC),&blurPC);
    vkCmdDraw(cmd,3,1,0,0);
    vkCmdEndRenderPass(cmd);
}

void Renderer::recordLightingPass(VkCommandBuffer cmd, uint32_t fi) {
    VkExtent2D ext=m_swapchain->extent();
    VkClearValue clr{}; clr.color={{0,0,0,1}};
    VkRenderPassBeginInfo rpBI{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpBI.renderPass=m_lightingRP; rpBI.framebuffer=m_lightingFB;
    rpBI.renderArea.extent=ext; rpBI.clearValueCount=1; rpBI.pClearValues=&clr;
    vkCmdBeginRenderPass(cmd,&rpBI,VK_SUBPASS_CONTENTS_INLINE);
    VkViewport vp{0,0,(float)ext.width,(float)ext.height,0,1};
    vkCmdSetViewport(cmd,0,1,&vp);
    VkRect2D sc{{0,0},ext}; vkCmdSetScissor(cmd,0,1,&sc);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,m_lightPipeline);

    // Always bind all 5 sets — use fallbacks when real resources aren't ready.
    bool hasIBL    = m_iblProbe && m_iblProbe->isReady() && m_framesExt[fi].iblSet != VK_NULL_HANDLE;
    bool hasShadow = m_shadowSystem != nullptr                && m_framesExt[fi].shadowSet != VK_NULL_HANDLE;

    VkDescriptorSet sets[5] = {
        m_frames[fi].gbufSet,
        m_frames[fi].frameSet,
        m_frames[fi].lightSet,
        hasIBL    ? m_framesExt[fi].iblSet    : m_fallbackIBLSets[fi],
        hasShadow ? m_framesExt[fi].shadowSet : m_framesExt[fi].shadowSet,  // shadow always inited
    };
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,m_lightPipeLayout,0,5,sets,0,nullptr);

    // Push lighting control flags + flat ambient (used when IBL is absent).
    LightingPC lpc{};
    lpc.flags            = (hasIBL ? 1u : 0u) | (hasShadow ? 2u : 0u)
                          | ((m_settings.debugGBuffer & 0xFu) << 4u);
    lpc.ambientR         = m_currentScene ? m_currentScene->ambientColor().r : 0.1f;
    lpc.ambientG         = m_currentScene ? m_currentScene->ambientColor().g : 0.1f;
    lpc.ambientB         = m_currentScene ? m_currentScene->ambientColor().b : 0.15f;
    lpc.ambientIntensity = m_currentScene ? m_currentScene->ambientIntensity() : 0.05f;
    vkCmdPushConstants(cmd,m_lightPipeLayout,VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(LightingPC),&lpc);

    vkCmdDraw(cmd,3,1,0,0);
    vkCmdEndRenderPass(cmd);
}

void Renderer::recordTonemapPass(VkCommandBuffer cmd, uint32_t imageIdx, uint32_t fi) {
    VkExtent2D ext=m_swapchain->extent();
    // Swapchain RP has loadOp=CLEAR on both color and depth attachments, so
    // we must provide 2 clear values even though our fullscreen triangle never
    // writes depth.
    VkClearValue clears[2]{};
    clears[0].color         = {{0,0,0,1}};
    clears[1].depthStencil  = {1.f, 0};
    VkRenderPassBeginInfo rpBI{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpBI.renderPass=m_swapchain->renderPass();
    rpBI.framebuffer=m_swapchain->framebuffer(imageIdx);
    rpBI.renderArea.extent=ext; rpBI.clearValueCount=2; rpBI.pClearValues=clears;
    vkCmdBeginRenderPass(cmd,&rpBI,VK_SUBPASS_CONTENTS_INLINE);
    VkViewport vp{0,0,(float)ext.width,(float)ext.height,0,1};
    vkCmdSetViewport(cmd,0,1,&vp);
    VkRect2D sc{{0,0},ext}; vkCmdSetScissor(cmd,0,1,&sc);
    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,m_tonemapPipeline);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,
                             m_tonemapPipeLayout,0,1,&m_frames[fi].hdrSet,0,nullptr);
    TonemapPC tmPC{m_settings.exposure, m_settings.tonemapOp};
    vkCmdPushConstants(cmd,m_tonemapPipeLayout,VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(TonemapPC),&tmPC);
    vkCmdDraw(cmd,3,1,0,0);
    vkCmdEndRenderPass(cmd);
}

// ─────────────────────────────────────────────────────────────────────────────
// Full frame recording
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::recordFrame(VkCommandBuffer cmd, uint32_t imageIdx,
                            Scene& scene, uint32_t fi)
{
    // 0. Shadow cascades — depth-only pass for directional light
    if (m_shadowSystem) {
        // Find first directional light from scene; fall back to sun direction
        glm::vec3 lightDir{0.f, -1.f, -0.5f};
        for (auto& l : scene.lights()) {
            if (l->type() == LightType::Directional) {
                lightDir = glm::vec3(l->toGpuLight().direction);
                break;
            }
        }
        m_shadowSystem->update(*scene.camera(), lightDir);

        // Upload ShadowUBO for the lighting pass
        const ShadowUBO& ubo = m_shadowSystem->shadowUBO();
        std::memcpy(m_framesExt[fi].shadowUBO.mapped, &ubo, sizeof(ShadowUBO));

        m_shadowSystem->renderCascades(cmd, [&](VkCommandBuffer c, uint32_t cascade) {
            recordShadowDraw(c, cascade, m_visibleScratch, fi);
        });
    }

    // 1. Geometry pass
    recordGeometryPass(cmd, m_visibleScratch, fi);

    // 2. SSAO pass
    recordSSAOPass(cmd, fi);

    // 3. SSAO blur pass
    recordSSAOBlurPass(cmd, fi);

    // 4. Explicit barrier: ensure shadow array is in DEPTH_STENCIL_READ_ONLY and
    //    all cascade writes are visible to the lighting fragment shader.
    //    The render pass finalLayout transitions cover the layout, but this barrier
    //    makes the memory explicitly available even across render pass boundaries.
    if (m_shadowSystem) {
        VkImageMemoryBarrier shadowBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        shadowBarrier.srcAccessMask       = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        shadowBarrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        shadowBarrier.oldLayout           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        shadowBarrier.newLayout           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        shadowBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        shadowBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        shadowBarrier.image               = m_shadowSystem->shadowArrayImage();
        shadowBarrier.subresourceRange    = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, NUM_CASCADES};
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &shadowBarrier);
    }

    // 5. Lighting pass — reads SSAO blurred (gbufSet binding 6).
    //    The SSAO render pass already transitions ssaoBlur to SHADER_READ_ONLY.
    recordLightingPass(cmd, fi);

    // 5. Tonemap to swapchain
    recordTonemapPass(cmd, imageIdx, fi);
}

// ─────────────────────────────────────────────────────────────────────────────
// Resize
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::handleResize() {
    auto [w,h] = m_window.getFramebufferSize();
    while (w==0||h==0) { glfwWaitEvents(); std::tie(w,h)=m_window.getFramebufferSize(); }
    m_ctx->waitIdle();
    destroySizeDependentResources();
    m_swapchain->recreate(w,h);
    createSizeDependentResources();
    allocateAndWriteDescriptorSets();  // re-bind recreated images
}

// ─────────────────────────────────────────────────────────────────────────────
// render() — called once per frame by the application
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::render(Scene& scene) {
    m_currentScene = &scene;
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float>(now - startTime).count();

    uint32_t fi = m_currentFrame;

    // ── Promote async IBL probe once the background thread finishes ──────────
    // We wait for the fence FIRST so we know the GPU is idle for this slot,
    // then swap in the real probe and update the descriptor sets.
    auto& frame = m_swapchain->frame(fi);

    // Wait for the previous use of this frame slot to finish.
    vkWaitForFences(m_ctx->device(),1,&frame.inFlightFence,VK_TRUE,UINT64_MAX);

    // Acquire swapchain image.
    uint32_t imageIdx;
    VkResult acquireRes = m_swapchain->acquireNextImage(fi, imageIdx);
    if (acquireRes == VK_ERROR_OUT_OF_DATE_KHR) { handleResize(); return; }
    if (acquireRes != VK_SUCCESS && acquireRes != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("[VKGFX] Failed to acquire swapchain image");

    vkResetFences(m_ctx->device(),1,&frame.inFlightFence);

    // ── Update per-frame GPU buffers ─────────────────────────────────────────

    // FrameUBO — camera matrices + time
    if (scene.camera()) {
        auto ubo = scene.camera()->toFrameUBO(time);
        std::memcpy(m_frames[fi].frameUBO.mapped, &ubo, sizeof(FrameUBO));

        // SSAOParams — write directly to the mapped GPU buffer (avoids 720B stack alloc).
        {
            VkExtent2D ext=m_swapchain->extent();
            auto* ssaoP = static_cast<SSAOParams*>(m_frames[fi].ssaoUBO.mapped);
            ssaoP->proj      = ubo.proj;
            ssaoP->invProj   = ubo.invProj;
            ssaoP->view      = ubo.view;
            for (uint32_t i=0; i<32; ++i) ssaoP->samples[i]=m_ssaoKernel[i];
            ssaoP->noiseScale = {static_cast<float>(ext.width)/4.f, static_cast<float>(ext.height)/4.f};
            ssaoP->radius    = m_settings.ssaoRadius;
            ssaoP->bias      = m_settings.ssaoBias;
        }
    }

    // LightSSBO — build directly into the persistently-mapped GPU buffer.
    // Previously allocated 16,400 bytes (LightSSBO = uint32 + GpuLight[256]) on
    // the stack every frame, which combined with other large locals (SSAOParams=720B,
    // FrameUBO=352B) created a stack frame large enough to corrupt the caller's stack
    // on Windows — manifesting as MSVC RTC #2 "stack around variable 'cx' corrupted".
    {
        auto* lightBuf = static_cast<LightSSBO*>(m_frames[fi].lightSSBO.mapped);
        lightBuf->count = 0;
        scene.buildLightBuffer(*lightBuf);
    }

    // ── CPU-side frustum culling (also feeds GPU culling instance list) ──────
    m_visibleScratch = scene.visibleMeshes(m_threadPool.get());
    m_stats.culledObjects = static_cast<uint32_t>(scene.meshes().size()) -
                             static_cast<uint32_t>(m_visibleScratch.size());

    // ── Upload any new meshes ────────────────────────────────────────────────
    for (auto* mesh : m_visibleScratch)
        if (!mesh->gpuReady) uploadMeshToGPU(*mesh);

    // ── GPU culling — build instance list and dispatch compute ───────────────
    if (m_gpuCulling) {
        std::vector<GpuInstance> instances;
        instances.reserve(m_visibleScratch.size());
        for (uint32_t idx = 0; auto* mesh : m_visibleScratch) {
            GpuInstance gi{};
            AABB wb       = mesh->worldBounds();
            gi.model       = mesh->modelMatrix();
            gi.aabbMin     = glm::vec4(wb.min, 0.f);
            gi.aabbMax     = glm::vec4(wb.max, 0.f);
            gi.meshIdx     = idx;
            gi.materialIdx = 0;
            instances.push_back(gi);
            ++idx;
        }
        m_gpuCulling->uploadInstances(instances);
        // Note: actual dispatch (cull()) is called in recordFrame before geometry,
        // after the command buffer has been opened.
    }

    // ── Flush deferred deletions (buffers retired > 1 frame ago) ────────────
    m_deletionQueue.erase(
        std::remove_if(m_deletionQueue.begin(), m_deletionQueue.end(),
            [&](DeferredDelete& d) {
                if (m_frameCounter - d.frameIndex >= MAX_FRAMES_IN_FLIGHT) {
                    m_ctx->destroyBuffer(d.vertexBuffer);
                    m_ctx->destroyBuffer(d.indexBuffer);
                    for (auto& b : d.matUBOs) m_ctx->destroyBuffer(b);
                    return true;
                }
                return false;
            }),
        m_deletionQueue.end());

    // ── Record command buffer ────────────────────────────────────────────────
    VkCommandBuffer cmd = frame.commandBuffer;
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd,&bi));
    recordFrame(cmd, imageIdx, scene, fi);
    VK_CHECK(vkEndCommandBuffer(cmd));

    // ── Submit ───────────────────────────────────────────────────────────────
    VkSemaphore waitSem   = m_swapchain->imageAvailableSemaphore(fi);
    VkSemaphore signalSem = m_swapchain->renderFinishedSemaphore(imageIdx);
    VkPipelineStageFlags waitMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount  =1; si.pWaitSemaphores  =&waitSem;   si.pWaitDstStageMask=&waitMask;
    si.commandBufferCount  =1; si.pCommandBuffers  =&cmd;
    si.signalSemaphoreCount=1; si.pSignalSemaphores=&signalSem;
    VK_CHECK(vkQueueSubmit(m_ctx->graphicsQueue(),1,&si,frame.inFlightFence));

    // ── Present ──────────────────────────────────────────────────────────────
    VkResult presentRes = m_swapchain->present(imageIdx);
    if (presentRes==VK_ERROR_OUT_OF_DATE_KHR || presentRes==VK_SUBOPTIMAL_KHR ||
        m_window.wasResized())
    {
        m_window.resetResizeFlag();
        handleResize();
    } else if (presentRes != VK_SUCCESS) {
        throw std::runtime_error("[VKGFX] Failed to present");
    }

    m_currentFrame = (fi + 1) % MAX_FRAMES_IN_FLIGHT;
    ++m_frameCounter;

    // ── Frame timing stats ───────────────────────────────────────────────────
    static auto lastFpsTime = std::chrono::high_resolution_clock::now();
    static uint32_t fpsFrames = 0;
    ++fpsFrames;
    auto elapsed = std::chrono::duration<float>(now - lastFpsTime).count();
    if (elapsed >= 1.f) {
        m_stats.fps = static_cast<float>(fpsFrames) / elapsed;
        m_stats.frameTimeMs = 1000.f / m_stats.fps;
        fpsFrames = 0;
        lastFpsTime = now;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Render Graph init
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::initRenderGraph() {
    m_renderGraph = std::make_unique<RenderGraph>(m_ctx);

    auto [w, h] = m_window.getFramebufferSize();

    RGTextureDesc hdrDesc{};
    hdrDesc.format      = VK_FORMAT_R16G16B16A16_SFLOAT;
    hdrDesc.widthScale  = 1.f; hdrDesc.heightScale = 1.f;
    hdrDesc.name        = "HDR";
    m_rgHDR = m_renderGraph->createTexture(hdrDesc);

    RGTextureDesc ssaoDesc{};
    ssaoDesc.format      = VK_FORMAT_R8_UNORM;
    ssaoDesc.widthScale  = 1.f; ssaoDesc.heightScale = 1.f;
    ssaoDesc.name        = "SSAO_Raw";
    m_rgSSAO = m_renderGraph->createTexture(ssaoDesc);

    RGTextureDesc ssaoBlurDesc = ssaoDesc;
    ssaoBlurDesc.name    = "SSAO_Blur";
    m_rgSSAOBlur = m_renderGraph->createTexture(ssaoBlurDesc);

    m_renderGraph->compile({static_cast<uint32_t>(w), static_cast<uint32_t>(h)});
}

// ─────────────────────────────────────────────────────────────────────────────
// IBL init
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::initIBL(const std::filesystem::path& hdrPath) {
    if (!std::filesystem::exists(hdrPath)) {
        std::cout << "[VKGFX] IBL: " << hdrPath << " not found — using flat ambient\n";
        return;
    }
    m_iblProbe = std::make_unique<IBLProbe>();
    m_iblProbe->loadFromEquirectangular(m_ctx, hdrPath);

    // Create descriptor set layouts for IBL (set 3) and shadow (set 4)
    createIBLDescriptors();
}

void Renderer::createIBLDescriptors() {
    if (!m_iblProbe || !m_iblProbe->isReady()) return;
    // m_iblLayout is already created in createDescriptorLayouts().
    auto dev = m_ctx->device();

    // Allocate and write one set per frame (same IBL images across all frames)
    for (auto& fx : m_framesExt) {
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = m_descPool; ai.descriptorSetCount = 1; ai.pSetLayouts = &m_iblLayout;
        VK_CHECK(vkAllocateDescriptorSets(dev, &ai, &fx.iblSet));

        auto writeImg = [&](uint32_t bind, VkImageView view, VkSampler samp) {
            VkDescriptorImageInfo ii{samp, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkWriteDescriptorSet   w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet=fx.iblSet; w.dstBinding=bind; w.descriptorCount=1;
            w.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo=&ii;
            vkUpdateDescriptorSets(dev,1,&w,0,nullptr);
        };
        writeImg(0, m_iblProbe->irradianceView(),  m_iblProbe->cubeSampler());
        writeImg(1, m_iblProbe->prefilteredView(), m_iblProbe->cubeSampler());
        writeImg(2, m_iblProbe->brdfLUTView(),     m_iblProbe->brdfSampler());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shadow system init
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::initShadows() {
    m_shadowSystem = std::make_unique<ShadowSystem>();
    std::cout << "  shadow::init..." << std::flush;
    m_shadowSystem->init(m_ctx);
    std::cout << " OK  pipeline..." << std::flush;
    m_shadowSystem->createPipeline(m_settings.shaderDir, m_frameLayout);
    std::cout << " OK  descriptors..." << std::flush;
    createShadowDescriptors();
    std::cout << " OK" << std::flush;
}

void Renderer::createShadowDescriptors() {
    // m_shadowLayout is already created in createDescriptorLayouts().
    auto dev = m_ctx->device();

    for (auto& fx : m_framesExt) {
        // Fix: use VkMemoryPropertyFlags, not VmaMemoryUsage
        fx.shadowUBO = m_ctx->createBuffer(sizeof(ShadowUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = m_descPool; ai.descriptorSetCount = 1; ai.pSetLayouts = &m_shadowLayout;
        VK_CHECK(vkAllocateDescriptorSets(dev, &ai, &fx.shadowSet));

        // Shadow map array sampler — must use DEPTH_STENCIL_READ_ONLY_OPTIMAL for sampler2DArrayShadow
        VkDescriptorImageInfo imgInfo{
            m_shadowSystem->shadowSampler(),
            m_shadowSystem->shadowArrayView(),
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
        };
        VkWriteDescriptorSet w0{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w0.dstSet=fx.shadowSet; w0.dstBinding=0; w0.descriptorCount=1;
        w0.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w0.pImageInfo=&imgInfo;
        vkUpdateDescriptorSets(dev,1,&w0,0,nullptr);

        // ShadowUBO
        VkDescriptorBufferInfo bufInfo{fx.shadowUBO.buffer, 0, sizeof(ShadowUBO)};
        VkWriteDescriptorSet w1{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w1.dstSet=fx.shadowSet; w1.dstBinding=1; w1.descriptorCount=1;
        w1.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w1.pBufferInfo=&bufInfo;
        vkUpdateDescriptorSets(dev,1,&w1,0,nullptr);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU Culling init
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::initGPUCulling() {
    m_gpuCulling = std::make_unique<GPUCulling>();
    m_gpuCulling->init(m_ctx, 65536);
    // Pipeline requires compiled SPIR-V — compile shaders first with glslc:
    //   glslc shaders/gpu_cull.comp -o shaders/gpu_cull.comp.spv
    std::filesystem::path spv = m_settings.shaderDir / "gpu_cull.comp.spv";
    if (std::filesystem::exists(spv))
        m_gpuCulling->createPipeline(m_settings.shaderDir);
    else
        std::cout << "[VKGFX] GPU culling shader not found at " << spv
                  << " — falling back to CPU culling\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// Shadow draw callback
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::recordShadowDraw(VkCommandBuffer cmd, uint32_t cascadeIdx,
                                  const std::vector<Mesh*>& meshes, uint32_t fi)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowSystem->pipeline());
    // No descriptor sets needed — the light-space matrix is in the push constant.

    // Grab the per-cascade light-space matrix from the shadow UBO we already filled.
    const ShadowUBO* ubo = reinterpret_cast<const ShadowUBO*>(m_framesExt[fi].shadowUBO.mapped);

    for (auto* mesh : meshes) {
        if (!mesh || !mesh->gpuReady) continue;

        struct PushData { glm::mat4 model; glm::mat4 lightSpace; } pc;
        pc.model      = mesh->modelMatrix();
        pc.lightSpace = ubo->lightSpaceMatrix[cascadeIdx];
        vkCmdPushConstants(cmd, m_shadowSystem->pipeLayout(),
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushData), &pc);

        VkBuffer vb      = mesh->vertexBuffer.buffer;
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off);
        vkCmdBindIndexBuffer(cmd, mesh->indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, static_cast<uint32_t>(mesh->indices().size()), 1, 0, 0, 0);
    }
}

} // namespace vkgfx

