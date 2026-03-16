// src/ibl.cpp
// IBL bake pipeline.
// Steps: load equirect HDR → envCube (compute) → irradiance (compute)
//        → prefiltered specular (compute, 5 mips) → BRDF LUT (compute)
// Every VkStruct is zero-initialised.  vkDeviceWaitIdle called before destroy.

#include <vkgfx/ibl.h>
#include <vkgfx/context.h>

// VMA_IMPLEMENTATION is defined only in context.cpp.
// Here we just use the types.
#include <vk_mem_alloc.h>

// stb_image — only declare; context.cpp / texture.cpp already provide
// the implementation translation unit via STB_IMAGE_IMPLEMENTATION.
// We only need stbi_loadf and stbi_info here.
#include <stb_image.h>

#include <vulkan/vulkan.h>

#include <fstream>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace vkgfx {

// ── Internal helpers ──────────────────────────────────────────────────────────

static VkShaderModule loadSPV(VkDevice device, const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("[vkgfx/ibl] Shader not found: " + path);
    size_t sz = static_cast<size_t>(f.tellg());
    std::vector<char> buf(sz);
    f.seekg(0);
    f.read(buf.data(), static_cast<std::streamsize>(sz));

    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = sz;
    ci.pCode    = reinterpret_cast<const uint32_t*>(buf.data());

    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &ci, nullptr, &mod) != VK_SUCCESS)
        throw std::runtime_error("[vkgfx/ibl] vkCreateShaderModule failed: " + path);
    return mod;
}

static VkPipeline makeComputePipeline(VkDevice device,
                                       VkShaderModule shader,
                                       VkPipelineLayout layout) {
    VkPipelineShaderStageCreateInfo stage{};
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName  = "main";

    VkComputePipelineCreateInfo ci{};
    ci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage  = stage;
    ci.layout = layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline);
    return pipeline;
}

// ── IBLSystem ─────────────────────────────────────────────────────────────────

IBLSystem::IBLSystem(Context& ctx) : m_ctx(ctx) {}

IBLSystem::~IBLSystem() { destroy(); }

void IBLSystem::destroy() {
    VkDevice dev = m_ctx.device();
    if (dev == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(dev);

    if (m_cubeSampler != VK_NULL_HANDLE) vkDestroySampler(dev, m_cubeSampler, nullptr);
    if (m_brdfSampler != VK_NULL_HANDLE) vkDestroySampler(dev, m_brdfSampler, nullptr);
    m_cubeSampler = m_brdfSampler = VK_NULL_HANDLE;

    m_ctx.destroyImage(m_equirect);
    m_ctx.destroyImage(m_envCube);
    m_ctx.destroyImage(m_irradiance);
    m_ctx.destroyImage(m_prefiltered);
    m_ctx.destroyImage(m_brdfLut);
    m_ready = false;
}

// ── build ─────────────────────────────────────────────────────────────────────

bool IBLSystem::build(const IBLConfig& cfg) {
    if (!std::filesystem::exists(cfg.hdrPath)) {
        std::cerr << "[vkgfx/ibl] HDR not found: " << cfg.hdrPath << "\n";
        return false;
    }

    destroy();
    m_intensity = cfg.intensity;

    try {
        if (!loadEquirectangular(cfg.hdrPath)) return false;
        buildEnvCube(cfg.envMapSize);
        buildIrradiance(cfg.irradianceSize);
        buildPrefiltered(cfg.envMapSize);
        buildBrdfLut();
        createSamplers();
    } catch (const std::exception& e) {
        std::cerr << "[vkgfx/ibl] Build failed: " << e.what() << "\n";
        destroy();
        return false;
    }

    m_ready = true;
    std::cout << "[vkgfx/ibl] Ready. HDR=" << cfg.hdrPath
              << " env=" << cfg.envMapSize
              << " irr=" << cfg.irradianceSize << "\n";
    return true;
}

// ── loadEquirectangular ───────────────────────────────────────────────────────

bool IBLSystem::loadEquirectangular(const std::string& path) {
    int w = 0, h = 0, ch = 0;
    if (!stbi_info(path.c_str(), &w, &h, &ch)) {
        std::cerr << "[vkgfx/ibl] stbi_info failed: " << path << "\n";
        return false;
    }

    float* pixels = stbi_loadf(path.c_str(), &w, &h, &ch, STBI_rgb_alpha);
    if (!pixels) {
        std::cerr << "[vkgfx/ibl] stbi_loadf failed: " << path << "\n";
        return false;
    }

    VkDeviceSize size = static_cast<VkDeviceSize>(w * h * 4 * sizeof(float));

    m_equirect = m_ctx.allocateImage(
        {static_cast<uint32_t>(w), static_cast<uint32_t>(h)},
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    AllocatedBuffer staging = m_ctx.allocateBuffer(size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);

    void* mapped = nullptr;
    vmaMapMemory(m_ctx.vma(), static_cast<VmaAllocation>(staging.allocation), &mapped);
    std::memcpy(mapped, pixels, static_cast<size_t>(size));
    vmaUnmapMemory(m_ctx.vma(), static_cast<VmaAllocation>(staging.allocation));
    stbi_image_free(pixels);

    VkCommandBuffer cmd = m_ctx.beginOneShot();

    transitionImage(cmd, m_equirect.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 1, 1);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
    vkCmdCopyBufferToImage(cmd, staging.buffer, m_equirect.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    transitionImage(cmd, m_equirect.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 1, 1);

    m_ctx.endOneShot(cmd);
    m_ctx.destroyBuffer(staging);

    m_equirect.view = m_ctx.createImageView(m_equirect.image,
        VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
    return true;
}

// ── transitionImage ───────────────────────────────────────────────────────────

void IBLSystem::transitionImage(VkCommandBuffer cmd, VkImage image,
    VkImageLayout oldLayout, VkImageLayout newLayout,
    VkAccessFlags srcAccess, VkAccessFlags dstAccess,
    VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
    uint32_t mipLevels, uint32_t layers)
{
    VkImageMemoryBarrier b{};
    b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    b.oldLayout           = oldLayout;
    b.newLayout           = newLayout;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image               = image;
    b.srcAccessMask       = srcAccess;
    b.dstAccessMask       = dstAccess;
    b.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, layers};
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// ── runCubeCompute ────────────────────────────────────────────────────────────
// Generic helper: runs a compute shader that reads a source cube/2D image
// and writes to a destination cube image.
// bindings: 0 = combined image sampler (src), 1 = storage image (dst)

void IBLSystem::runCubeCompute(const std::string& shaderName,
                                const AllocatedImage& srcImg,
                                AllocatedImage& dstCube,
                                uint32_t dstSize, uint32_t dstMips)
{
    VkDevice dev = m_ctx.device();

    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 2; dslci.pBindings = bindings;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(dev, &plci, nullptr, &layout);

    VkShaderModule shader = loadSPV(dev, m_shaderDir + "/" + shaderName);
    VkPipeline pipeline   = makeComputePipeline(dev, shader, layout);

    VkSamplerCreateInfo sci{};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod       = 1.f;
    VkSampler sampler = VK_NULL_HANDLE;
    vkCreateSampler(dev, &sci, nullptr, &sampler);

    // Per-output-mip storage image view (2D_ARRAY over 6 faces)
    // For single-mip outputs (irradiance), dstMips == 1.
    VkImageView dstArrayView = m_ctx.createImageView(
        dstCube.image, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_ASPECT_COLOR_BIT, dstMips, 6, VK_IMAGE_VIEW_TYPE_2D_ARRAY);

    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    poolSizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1; dpci.poolSizeCount = 2; dpci.pPoolSizes = poolSizes;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(dev, &dpci, nullptr, &pool);

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &dsl;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    vkAllocateDescriptorSets(dev, &dsai, &ds);

    VkDescriptorImageInfo srcInfo{};
    srcInfo.sampler     = sampler;
    srcInfo.imageView   = srcImg.view;
    srcInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo dstInfo{};
    dstInfo.imageView   = dstArrayView;
    dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = ds; writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1; writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &srcInfo;
    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = ds; writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1; writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &dstInfo;
    vkUpdateDescriptorSets(dev, 2, writes, 0, nullptr);

    VkCommandBuffer cmd = m_ctx.beginOneShot();

    transitionImage(cmd, dstCube.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        0, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        dstMips, 6);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);

    uint32_t groups = (dstSize + 15) / 16;
    vkCmdDispatch(cmd, groups, groups, 6);

    transitionImage(cmd, dstCube.image,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        dstMips, 6);

    m_ctx.endOneShot(cmd);

    vkDestroyImageView(dev, dstArrayView, nullptr);
    vkDestroySampler(dev, sampler, nullptr);
    vkDestroyPipeline(dev, pipeline, nullptr);
    vkDestroyPipelineLayout(dev, layout, nullptr);
    vkDestroyDescriptorPool(dev, pool, nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    vkDestroyShaderModule(dev, shader, nullptr);
}

// ── buildEnvCube ──────────────────────────────────────────────────────────────

void IBLSystem::buildEnvCube(uint32_t size) {
    m_envCube = m_ctx.allocateImage(
        {size, size}, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        1, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);

    runCubeCompute("equirect_to_cube.comp.spv", m_equirect, m_envCube, size, 1);

    m_envCube.view = m_ctx.createImageView(m_envCube.image,
        VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
        1, 6, VK_IMAGE_VIEW_TYPE_CUBE);
}

// ── buildIrradiance ───────────────────────────────────────────────────────────

void IBLSystem::buildIrradiance(uint32_t size) {
    m_irradiance = m_ctx.allocateImage(
        {size, size}, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        1, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);

    runCubeCompute("irradiance.comp.spv", m_envCube, m_irradiance, size, 1);

    m_irradiance.view = m_ctx.createImageView(m_irradiance.image,
        VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
        1, 6, VK_IMAGE_VIEW_TYPE_CUBE);
}

// ── buildPrefiltered ──────────────────────────────────────────────────────────

void IBLSystem::buildPrefiltered(uint32_t size) {
    const uint32_t MIP_LEVELS = 5;

    m_prefiltered = m_ctx.allocateImage(
        {size, size}, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        MIP_LEVELS, 6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);

    VkDevice dev = m_ctx.device();

    struct PrefiltPush { float roughness; };

    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 2; dslci.pBindings = bindings;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset     = 0;
    pc.size       = sizeof(PrefiltPush);

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1; plci.pSetLayouts            = &dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges    = &pc;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(dev, &plci, nullptr, &layout);

    VkShaderModule shader  = loadSPV(dev, m_shaderDir + "/prefilter.comp.spv");
    VkPipeline     pipeline= makeComputePipeline(dev, shader, layout);

    VkSamplerCreateInfo sci{};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod       = 1.f;
    VkSampler cubeSampler = VK_NULL_HANDLE;
    vkCreateSampler(dev, &sci, nullptr, &cubeSampler);

    VkCommandBuffer cmd = m_ctx.beginOneShot();

    transitionImage(cmd, m_prefiltered.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        0, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        MIP_LEVELS, 6);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    for (uint32_t mip = 0; mip < MIP_LEVELS; ++mip) {
        uint32_t mipSize    = std::max(1u, size >> mip);
        float    roughness  = static_cast<float>(mip) / static_cast<float>(MIP_LEVELS - 1);

        // Per-mip storage view (2D_ARRAY, this mip level only)
        VkImageViewCreateInfo mvcI{};
        mvcI.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        mvcI.image    = m_prefiltered.image;
        mvcI.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        mvcI.format   = VK_FORMAT_R32G32B32A32_SFLOAT;
        mvcI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 6};
        VkImageView mipView = VK_NULL_HANDLE;
        vkCreateImageView(dev, &mvcI, nullptr, &mipView);

        VkDescriptorPoolSize poolSizes[2]{};
        poolSizes[0] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
        poolSizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.maxSets = 1; dpci.poolSizeCount = 2; dpci.pPoolSizes = poolSizes;
        VkDescriptorPool mipPool = VK_NULL_HANDLE;
        vkCreateDescriptorPool(dev, &dpci, nullptr, &mipPool);

        VkDescriptorSetAllocateInfo dsai{};
        dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dsai.descriptorPool     = mipPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts        = &dsl;
        VkDescriptorSet ds = VK_NULL_HANDLE;
        vkAllocateDescriptorSets(dev, &dsai, &ds);

        VkDescriptorImageInfo srcInfo{};
        srcInfo.sampler     = cubeSampler;
        srcInfo.imageView   = m_envCube.view;
        srcInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkDescriptorImageInfo dstInfo{};
        dstInfo.imageView   = mipView;
        dstInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet ws[2]{};
        ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[0].dstSet = ds; ws[0].dstBinding = 0;
        ws[0].descriptorCount = 1; ws[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ws[0].pImageInfo = &srcInfo;
        ws[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[1].dstSet = ds; ws[1].dstBinding = 1;
        ws[1].descriptorCount = 1; ws[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        ws[1].pImageInfo = &dstInfo;
        vkUpdateDescriptorSets(dev, 2, ws, 0, nullptr);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                layout, 0, 1, &ds, 0, nullptr);

        PrefiltPush push{roughness};
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), &push);

        uint32_t groups = (mipSize + 15) / 16;
        vkCmdDispatch(cmd, groups, groups, 6);

        m_bakeCleanupViews.push_back(mipView);
        m_bakeCleanupPools.push_back(mipPool);
    }

    transitionImage(cmd, m_prefiltered.image,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        MIP_LEVELS, 6);

    m_ctx.endOneShot(cmd);

    for (auto v : m_bakeCleanupViews)  vkDestroyImageView   (dev, v, nullptr);
    for (auto p : m_bakeCleanupPools)  vkDestroyDescriptorPool(dev, p, nullptr);
    m_bakeCleanupViews.clear();
    m_bakeCleanupPools.clear();

    vkDestroySampler          (dev, cubeSampler, nullptr);
    vkDestroyPipeline         (dev, pipeline,    nullptr);
    vkDestroyPipelineLayout   (dev, layout,      nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl,       nullptr);
    vkDestroyShaderModule     (dev, shader,      nullptr);

    m_prefiltered.view = m_ctx.createImageView(m_prefiltered.image,
        VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT,
        MIP_LEVELS, 6, VK_IMAGE_VIEW_TYPE_CUBE);
}

// ── buildBrdfLut ──────────────────────────────────────────────────────────────

void IBLSystem::buildBrdfLut() {
    const uint32_t LUT_SIZE = 512;
    VkDevice dev = m_ctx.device();

    m_brdfLut = m_ctx.allocateImage(
        {LUT_SIZE, LUT_SIZE}, VK_FORMAT_R16G16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    VkDescriptorSetLayoutBinding binding{};
    binding.binding        = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binding.descriptorCount= 1;
    binding.stageFlags     = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 1; dslci.pBindings = &binding;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl);

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(dev, &plci, nullptr, &layout);

    VkShaderModule shader   = loadSPV(dev, m_shaderDir + "/brdf_lut.comp.spv");
    VkPipeline     pipeline = makeComputePipeline(dev, shader, layout);

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(dev, &dpci, nullptr, &pool);

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &dsl;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    vkAllocateDescriptorSets(dev, &dsai, &ds);

    m_brdfLut.view = m_ctx.createImageView(m_brdfLut.image,
        VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

    VkDescriptorImageInfo imgInfo{};
    imgInfo.imageView   = m_brdfLut.view;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = ds; w.dstBinding = 0;
    w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w.pImageInfo = &imgInfo;
    vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);

    VkCommandBuffer cmd = m_ctx.beginOneShot();

    transitionImage(cmd, m_brdfLut.image,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        0, VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 1, 1);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, nullptr);
    uint32_t groups = (LUT_SIZE + 15) / 16;
    vkCmdDispatch(cmd, groups, groups, 1);

    transitionImage(cmd, m_brdfLut.image,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 1, 1);

    m_ctx.endOneShot(cmd);

    vkDestroyPipeline         (dev, pipeline, nullptr);
    vkDestroyPipelineLayout   (dev, layout,   nullptr);
    vkDestroyDescriptorPool   (dev, pool,     nullptr);
    vkDestroyDescriptorSetLayout(dev, dsl,    nullptr);
    vkDestroyShaderModule     (dev, shader,   nullptr);
}

// ── createSamplers ────────────────────────────────────────────────────────────

void IBLSystem::createSamplers() {
    VkDevice dev = m_ctx.device();

    VkSamplerCreateInfo sci{};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod       = 5.f; // prefiltered has 5 mips
    vkCreateSampler(dev, &sci, nullptr, &m_cubeSampler);

    sci.maxLod = 1.f;
    vkCreateSampler(dev, &sci, nullptr, &m_brdfSampler);
}

} // namespace vkgfx
