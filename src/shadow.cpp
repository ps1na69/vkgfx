#include "vkgfx/shadow.h"
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <fstream>
#include <vector>

namespace vkgfx {

static std::vector<char> readSpv(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::ate|std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Shadow: cannot open " + p.string());
    size_t sz = f.tellg(); std::vector<char> buf(sz);
    f.seekg(0); f.read(buf.data(), sz); return buf;
}

// ── init ──────────────────────────────────────────────────────────────────────
void ShadowSystem::init(std::shared_ptr<Context> ctx) {
    m_ctx = ctx;

    // Allocate one 2D array texture (4 layers) — all cascades share format.
    m_shadowArray = ctx->createImage(
        SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1, VK_SAMPLE_COUNT_1_BIT,
        VK_FORMAT_D32_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, NUM_CASCADES);
    m_shadowArray.mipLevels = 1;

    // Per-cascade: create individual 2D views into the array layers
    for (uint32_t c = 0; c < NUM_CASCADES; ++c) {
        auto& cascade = m_cascades[c];
        cascade.depthImage = m_shadowArray; // share the VkImage
        // Per-layer view
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image    = m_shadowArray.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format   = VK_FORMAT_D32_SFLOAT;
        vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, c, 1};
        VK_CHECK(vkCreateImageView(ctx->device(), &vi, nullptr, &cascade.depthImage.view));
    }

    createRenderPasses();
    createFramebuffers();
    createShadowArrayView();
    createSampler();

    // Transition the array to depth attachment initial layout
    VkCommandBuffer cmd = ctx->beginSingleTimeCommands();
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = m_shadowArray.image;
    b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, NUM_CASCADES};
    b.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0,
        0,nullptr, 0,nullptr, 1,&b);
    ctx->endSingleTimeCommands(cmd);
}

// ── createRenderPasses ────────────────────────────────────────────────────────
void ShadowSystem::createRenderPasses() {
    VkAttachmentDescription depth{};
    depth.format         = VK_FORMAT_D32_SFLOAT;
    depth.samples        = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    depth.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkAttachmentReference depthRef{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL; deps[0].dstSubpass = 0;
    deps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    deps[1].srcSubpass    = 0; deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask  = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    deps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo rpCI{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpCI.attachmentCount = 1; rpCI.pAttachments  = &depth;
    rpCI.subpassCount    = 1; rpCI.pSubpasses     = &sub;
    rpCI.dependencyCount = 2; rpCI.pDependencies  = deps;

    // All cascades share the same render pass description
    for (auto& c : m_cascades)
        VK_CHECK(vkCreateRenderPass(m_ctx->device(), &rpCI, nullptr, &c.renderPass));
}

// ── createFramebuffers ────────────────────────────────────────────────────────
void ShadowSystem::createFramebuffers() {
    for (uint32_t c = 0; c < NUM_CASCADES; ++c) {
        auto& cascade = m_cascades[c];
        VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fi.renderPass      = cascade.renderPass;
        fi.attachmentCount = 1;
        fi.pAttachments    = &cascade.depthImage.view;
        fi.width  = SHADOW_MAP_SIZE;
        fi.height = SHADOW_MAP_SIZE;
        fi.layers = 1;
        VK_CHECK(vkCreateFramebuffer(m_ctx->device(), &fi, nullptr, &cascade.framebuffer));
    }
}

// ── createShadowArrayView ─────────────────────────────────────────────────────
void ShadowSystem::createShadowArrayView() {
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image    = m_shadowArray.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    vi.format   = VK_FORMAT_D32_SFLOAT;
    vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, NUM_CASCADES};
    VK_CHECK(vkCreateImageView(m_ctx->device(), &vi, nullptr, &m_shadowArrayView));
}

// ── createSampler ─────────────────────────────────────────────────────────────
void ShadowSystem::createSampler() {
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter    = VK_FILTER_LINEAR;
    si.minFilter    = VK_FILTER_LINEAR;
    si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW =
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    si.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE; // depth=1 outside
    si.maxLod       = 1.f;
    // Hardware PCF
    si.compareEnable = VK_TRUE;
    si.compareOp     = VK_COMPARE_OP_LESS_OR_EQUAL;
    VK_CHECK(vkCreateSampler(m_ctx->device(), &si, nullptr, &m_shadowSampler));
}

// ── createPipeline ────────────────────────────────────────────────────────────
void ShadowSystem::createPipeline(const std::filesystem::path& shaderDir,
                                   VkDescriptorSetLayout sceneLayout)
{
    auto dev = m_ctx->device();

    // Push constant: model matrix (64 bytes) + cascade index (4 bytes)
    VkPushConstantRange pc{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(glm::mat4) + 4};
    VkDescriptorSetLayout layouts[] = { sceneLayout };
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1; li.pSetLayouts = layouts;
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(dev, &li, nullptr, &m_pipeLayout));

    auto vertCode = readSpv(shaderDir / "shadow_depth.vert.spv");
    VkShaderModuleCreateInfo smCI{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smCI.codeSize = vertCode.size();
    smCI.pCode    = reinterpret_cast<const uint32_t*>(vertCode.data());
    VkShaderModule vert;
    VK_CHECK(vkCreateShaderModule(dev, &smCI, nullptr, &vert));

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stage.module = vert; stage.pName = "main";

    // Vertex layout: position only (vec3 at binding 0, location 0)
    VkVertexInputBindingDescription binding{0, sizeof(float)*12, VK_VERTEX_INPUT_RATE_VERTEX};
    // stride = sizeof(Vertex) = vec3+vec3+vec2+vec4 = 12+12+8+16 = 48 bytes = 12 floats
    VkVertexInputAttributeDescription attr{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount   = 1; vi.pVertexBindingDescriptions   = &binding;
    vi.vertexAttributeDescriptionCount = 1; vi.pVertexAttributeDescriptions = &attr;

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport vp{0,0,SHADOW_MAP_SIZE,SHADOW_MAP_SIZE,0,1};
    VkRect2D   sc{{0,0},{SHADOW_MAP_SIZE,SHADOW_MAP_SIZE}};
    VkPipelineViewportStateCreateInfo vpS{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vpS.viewportCount=1; vpS.pViewports=&vp; vpS.scissorCount=1; vpS.pScissors=&sc;

    VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode    = VK_CULL_MODE_FRONT_BIT;  // front-face culling for shadow maps (Peter-pan)
    rast.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth   = 1.f;
    rast.depthBiasEnable         = VK_TRUE;
    rast.depthBiasConstantFactor = 2.5f;
    rast.depthBiasSlopeFactor    = 2.5f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};

    VkGraphicsPipelineCreateInfo pCI{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pCI.stageCount          = 1; pCI.pStages = &stage;
    pCI.pVertexInputState   = &vi;
    pCI.pInputAssemblyState = &ia;
    pCI.pViewportState      = &vpS;
    pCI.pRasterizationState = &rast;
    pCI.pMultisampleState   = &ms;
    pCI.pDepthStencilState  = &ds;
    pCI.pColorBlendState    = &blend;
    pCI.layout              = m_pipeLayout;
    pCI.renderPass          = m_cascades[0].renderPass;

    VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pCI, nullptr, &m_pipeline));
    vkDestroyShaderModule(dev, vert, nullptr);
}

// ── frustumCorners ────────────────────────────────────────────────────────────
std::array<glm::vec3, 8> ShadowSystem::frustumCorners(const glm::mat4& proj,
                                                        const glm::mat4& view) const
{
    glm::mat4 inv = glm::inverse(proj * view);
    std::array<glm::vec3, 8> corners;
    int i = 0;
    for (int x : {-1,1}) for (int y : {-1,1}) for (int z : {0,1}) {
        glm::vec4 pt = inv * glm::vec4(x, y, z, 1.f);
        corners[i++] = glm::vec3(pt) / pt.w;
    }
    return corners;
}

// ── update ────────────────────────────────────────────────────────────────────
void ShadowSystem::update(const Camera& camera, glm::vec3 lightDir, float lambda)
{
    m_nearClip = camera.nearPlane();
    m_farClip  = camera.farPlane();

    // Practical split scheme (Nvidia CSM):
    //   split_i = λ * near*(far/near)^(i/n) + (1-λ) * (near + (far-near)*i/n)
    float range = m_farClip - m_nearClip;
    float ratio = m_farClip / m_nearClip;

    std::array<float,NUM_CASCADES+1> splits;
    splits[0] = m_nearClip;
    for (uint32_t i = 1; i <= NUM_CASCADES; ++i) {
        float p      = i / static_cast<float>(NUM_CASCADES);
        float logSpl = m_nearClip * std::pow(ratio, p);
        float uniSpl = m_nearClip + range * p;
        splits[i]    = lambda * logSpl + (1.f - lambda) * uniSpl;
    }

    const glm::mat4& camView = camera.viewMatrix();
    glm::vec3 ld = glm::normalize(lightDir);

    for (uint32_t c = 0; c < NUM_CASCADES; ++c) {
        float near = splits[c], far = splits[c+1];

        // Build sub-frustum projection
        glm::mat4 subProj = glm::perspective(
            glm::radians(camera.fov()), camera.aspect(), near, far);
        subProj[1][1] *= -1; // Vulkan Y-flip

        auto corners = frustumCorners(subProj, camView);

        // Frustum centre
        glm::vec3 centre{0};
        for (auto& v : corners) centre += v;
        centre /= 8.f;

        // Stable radius: half diagonal of the sub-frustum
        float radius = 0.f;
        for (auto& v : corners) radius = std::max(radius, glm::length(v - centre));
        radius = std::ceil(radius * 16.f) / 16.f; // snap to texel grid

        glm::vec3 lightPos = centre - ld * radius;
        glm::mat4 lightView = glm::lookAt(lightPos, centre, {0,1,0});

        // Snap to shadow map texels to prevent shimmering
        float texelSz = 2.f * radius / SHADOW_MAP_SIZE;
        glm::mat4 snap = glm::mat4(1.f);
        snap[3].x = -std::fmod(lightView[3].x, texelSz);
        snap[3].y = -std::fmod(lightView[3].y, texelSz);
        lightView  = snap * lightView;

        glm::mat4 lightProj = glm::ortho(-radius, radius, -radius, radius,
                                          0.f, 2.f * radius);
        lightProj[1][1] *= -1;

        m_ubo.cascades[c].lightSpaceMatrix = lightProj * lightView;
        m_ubo.cascades[c].splitDepth       = (subProj * camView * glm::vec4(0,0,-far,1)).z;
    }
}

// ── renderCascades ────────────────────────────────────────────────────────────
void ShadowSystem::renderCascades(VkCommandBuffer cmd, const DrawFn& drawFn) {
    VkClearValue clear{};
    clear.depthStencil = {1.f, 0};

    for (uint32_t c = 0; c < NUM_CASCADES; ++c) {
        auto& cascade = m_cascades[c];

        // Transition this layer back to attachment (lighting pass reads as SHADER_READ_ONLY,
        // so we need to transition before writing again)
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = m_shadowArray.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, c, 1};
        b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0,
            0,nullptr, 0,nullptr, 1,&b);

        VkRenderPassBeginInfo rpBI{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rpBI.renderPass  = cascade.renderPass;
        rpBI.framebuffer = cascade.framebuffer;
        rpBI.renderArea  = {{0,0},{SHADOW_MAP_SIZE,SHADOW_MAP_SIZE}};
        rpBI.clearValueCount = 1; rpBI.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &rpBI, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

        VkViewport vp{0,0,SHADOW_MAP_SIZE,SHADOW_MAP_SIZE,0,1};
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{{0,0},{SHADOW_MAP_SIZE,SHADOW_MAP_SIZE}};
        vkCmdSetScissor(cmd, 0, 1, &sc);

        drawFn(cmd, c);

        vkCmdEndRenderPass(cmd);
    }

    // Transition entire array to SHADER_READ_ONLY for lighting pass
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = m_shadowArray.image;
    b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, NUM_CASCADES};
    b.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
        0,nullptr, 0,nullptr, 1,&b);
}

// ── destroy ───────────────────────────────────────────────────────────────────
void ShadowSystem::destroy() {
    if (!m_ctx) return;
    auto dev = m_ctx->device();
    if (m_pipeline)   { vkDestroyPipeline(dev, m_pipeline, nullptr);       m_pipeline   = VK_NULL_HANDLE; }
    if (m_pipeLayout) { vkDestroyPipelineLayout(dev, m_pipeLayout, nullptr); m_pipeLayout = VK_NULL_HANDLE; }
    if (m_shadowArrayView) { vkDestroyImageView(dev, m_shadowArrayView, nullptr); m_shadowArrayView = VK_NULL_HANDLE; }
    if (m_shadowSampler)   { vkDestroySampler(dev, m_shadowSampler, nullptr);     m_shadowSampler   = VK_NULL_HANDLE; }
    for (auto& c : m_cascades) {
        if (c.framebuffer) { vkDestroyFramebuffer(dev, c.framebuffer, nullptr); c.framebuffer = VK_NULL_HANDLE; }
        if (c.renderPass)  { vkDestroyRenderPass(dev, c.renderPass, nullptr);   c.renderPass  = VK_NULL_HANDLE; }
        // Per-layer views (not the shared image — that's destroyed once below)
        if (c.depthImage.view) { vkDestroyImageView(dev, c.depthImage.view, nullptr); c.depthImage.view = VK_NULL_HANDLE; }
    }
    m_ctx->destroyImage(m_shadowArray);
    m_ctx.reset();
}

} // namespace vkgfx
