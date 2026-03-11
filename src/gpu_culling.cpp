#include "vkgfx/gpu_culling.h"
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <vector>

namespace vkgfx {

static std::vector<char> readSpv(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::ate|std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("[GPUCulling] Cannot open " + p.string());
    size_t sz = f.tellg(); std::vector<char> buf(sz);
    f.seekg(0); f.read(buf.data(), sz); return buf;
}

// ── init ──────────────────────────────────────────────────────────────────────
void GPUCulling::init(std::shared_ptr<Context> ctx, uint32_t maxInstances) {
    m_ctx = ctx;
    m_maxInstances = maxInstances;

    VkDeviceSize instanceSz  = maxInstances * sizeof(GpuInstance);
    VkDeviceSize indirectSz  = maxInstances * sizeof(VkDrawIndexedIndirectCommand);
    VkDeviceSize meshTableSz = maxInstances * sizeof(MeshDrawData); // generous upper bound

    // Instance buffer — device local (written via staging each frame)
    m_instanceBuf  = ctx->createBuffer(instanceSz,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // Mesh table — device local
    m_meshTableBuf = ctx->createBuffer(meshTableSz,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // Indirect draw command buffer — written by compute, read by vkCmdDrawIndexedIndirectCount
    m_indirectBuf  = ctx->createBuffer(indirectSz,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // Count buffer — 1 uint32 written by compute atomicAdd, read by vkCmdDrawIndexedIndirectCount
    m_countBuf = ctx->createBuffer(sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // Frustum UBO — host-coherent, updated each frame
    m_frustumUBO = ctx->createBuffer(sizeof(CullFrustum),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST, true);

    createDescriptorResources();
}

// ── createDescriptorResources ─────────────────────────────────────────────────
void GPUCulling::createDescriptorResources() {
    auto dev = m_ctx->device();

    // Layout: binding 0 = instances SSBO, 1 = mesh table SSBO,
    //         2 = indirect SSBO, 3 = count SSBO, 4 = frustum UBO
    using B = VkDescriptorSetLayoutBinding;
    B bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 5; lci.pBindings = bindings;
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &lci, nullptr, &m_dsLayout));

    VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
    };
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(dev, &pci, nullptr, &m_pool));

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = m_pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &m_dsLayout;
    VK_CHECK(vkAllocateDescriptorSets(dev, &ai, &m_ds));

    // Write descriptor set
    auto wSSBO = [&](uint32_t bind, VkBuffer buf, VkDeviceSize sz) {
        VkDescriptorBufferInfo bi{buf, 0, sz};
        VkWriteDescriptorSet   w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet=m_ds; w.dstBinding=bind; w.descriptorCount=1;
        w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo=&bi;
        vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
    };
    auto wUBO = [&](uint32_t bind, VkBuffer buf, VkDeviceSize sz) {
        VkDescriptorBufferInfo bi{buf, 0, sz};
        VkWriteDescriptorSet   w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet=m_ds; w.dstBinding=bind; w.descriptorCount=1;
        w.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w.pBufferInfo=&bi;
        vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
    };
    wSSBO(0, m_instanceBuf.buffer,  m_maxInstances * sizeof(GpuInstance));
    wSSBO(1, m_meshTableBuf.buffer, m_maxInstances * sizeof(MeshDrawData));
    wSSBO(2, m_indirectBuf.buffer,  m_maxInstances * sizeof(VkDrawIndexedIndirectCommand));
    wSSBO(3, m_countBuf.buffer,     sizeof(uint32_t));
    wUBO (4, m_frustumUBO.buffer,   sizeof(CullFrustum));
}

// ── createPipeline ────────────────────────────────────────────────────────────
void GPUCulling::createPipeline(const std::filesystem::path& shaderDir) {
    auto dev = m_ctx->device();

    // Push constant: instance count
    VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1; li.pSetLayouts = &m_dsLayout;
    li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pc;
    VK_CHECK(vkCreatePipelineLayout(dev, &li, nullptr, &m_pipeLayout));

    auto code = readSpv(shaderDir / "gpu_cull.comp.spv");
    VkShaderModuleCreateInfo smCI{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smCI.codeSize = code.size();
    smCI.pCode    = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule comp;
    VK_CHECK(vkCreateShaderModule(dev, &smCI, nullptr, &comp));

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = comp; stage.pName = "main";

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage  = stage;
    ci.layout = m_pipeLayout;
    VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &m_pipeline));
    vkDestroyShaderModule(dev, comp, nullptr);
}

// ── uploadInstances ───────────────────────────────────────────────────────────
void GPUCulling::uploadInstances(const std::vector<GpuInstance>& instances) {
    if (instances.empty()) return;
    VkDeviceSize sz = instances.size() * sizeof(GpuInstance);
    AllocatedBuffer staging = m_ctx->createBuffer(sz,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, true);
    std::memcpy(staging.mapped, instances.data(), sz);
    m_ctx->copyBuffer(staging.buffer, m_instanceBuf.buffer, sz);
    m_ctx->destroyBuffer(staging);
}

// ── uploadMeshTable ───────────────────────────────────────────────────────────
void GPUCulling::uploadMeshTable(const std::vector<MeshDrawData>& table) {
    if (table.empty()) return;
    VkDeviceSize sz = table.size() * sizeof(MeshDrawData);
    AllocatedBuffer staging = m_ctx->createBuffer(sz,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, true);
    std::memcpy(staging.mapped, table.data(), sz);
    m_ctx->copyBuffer(staging.buffer, m_meshTableBuf.buffer, sz);
    m_ctx->destroyBuffer(staging);
}

// ── cull ──────────────────────────────────────────────────────────────────────
void GPUCulling::cull(VkCommandBuffer cmd, const CullFrustum& frustum,
                       uint32_t instanceCount)
{
    if (instanceCount == 0) return;

    // Update frustum UBO
    std::memcpy(m_frustumUBO.mapped, &frustum, sizeof(CullFrustum));

    // Reset indirect count to 0
    vkCmdFillBuffer(cmd, m_countBuf.buffer, 0, sizeof(uint32_t), 0);

    // Barrier: ensure fill is done before compute reads it
    VkBufferMemoryBarrier b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = m_countBuf.buffer; b.offset = 0; b.size = sizeof(uint32_t);
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
        0,nullptr, 1,&b, 0,nullptr);

    // Dispatch culling shader — 64 threads per workgroup
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             m_pipeLayout, 0, 1, &m_ds, 0, nullptr);
    vkCmdPushConstants(cmd, m_pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(uint32_t), &instanceCount);
    uint32_t groups = (instanceCount + 63) / 64;
    vkCmdDispatch(cmd, groups, 1, 1);

    // Barrier: compute writes → indirect draw reads
    VkBufferMemoryBarrier barriers[2]{};
    barriers[0] = b;
    barriers[0].buffer = m_indirectBuf.buffer;
    barriers[0].size   = instanceCount * sizeof(VkDrawIndexedIndirectCommand);
    barriers[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    barriers[1] = b;
    barriers[1].buffer = m_countBuf.buffer;
    barriers[1].size   = sizeof(uint32_t);
    barriers[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barriers[1].dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0,
        0,nullptr, 2,barriers, 0,nullptr);
}

// ── drawIndirect ──────────────────────────────────────────────────────────────
void GPUCulling::drawIndirect(VkCommandBuffer cmd) {
    // vkCmdDrawIndexedIndirectCount reads the actual draw count from countBuf,
    // so only surviving instances are drawn.
    vkCmdDrawIndexedIndirectCount(cmd,
        m_indirectBuf.buffer, 0,
        m_countBuf.buffer,    0,
        m_maxInstances,
        sizeof(VkDrawIndexedIndirectCommand));
}

// ── destroy ───────────────────────────────────────────────────────────────────
void GPUCulling::destroy() {
    if (!m_ctx) return;
    auto dev = m_ctx->device();
    if (m_pipeline)  { vkDestroyPipeline(dev, m_pipeline, nullptr); m_pipeline = VK_NULL_HANDLE; }
    if (m_pipeLayout){ vkDestroyPipelineLayout(dev, m_pipeLayout, nullptr); m_pipeLayout = VK_NULL_HANDLE; }
    if (m_pool)      { vkDestroyDescriptorPool(dev, m_pool, nullptr); m_pool = VK_NULL_HANDLE; }
    if (m_dsLayout)  { vkDestroyDescriptorSetLayout(dev, m_dsLayout, nullptr); m_dsLayout = VK_NULL_HANDLE; }
    m_ctx->destroyBuffer(m_instanceBuf);
    m_ctx->destroyBuffer(m_meshTableBuf);
    m_ctx->destroyBuffer(m_indirectBuf);
    m_ctx->destroyBuffer(m_countBuf);
    m_ctx->destroyBuffer(m_frustumUBO);
    m_ctx.reset();
}

} // namespace vkgfx
