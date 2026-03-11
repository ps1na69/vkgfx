#pragma once
// gpu_culling.h — GPU-driven frustum culling via compute shader.
//
// Replaces the CPU scene.visibleMeshes() loop for large object counts.
// Workflow per frame:
//   1. UploadInstanceBuffer(instances)  — write per-object AABB + model matrix
//   2. dispatch compute shader         — reads instances, writes draw commands
//   3. vkCmdDrawIndexedIndirectCount   — GPU culls; only surviving draws execute
//
// The compute shader outputs VkDrawIndexedIndirectCommand into a device-local
// buffer.  The count of surviving draws is stored in a separate counter buffer.

#include "context.h"
#include <vector>

namespace vkgfx {

// Per-instance data uploaded once per frame to the instance buffer.
struct alignas(16) GpuInstance {
    glm::mat4 model;
    glm::vec4 aabbMin;   // .xyz = min, .w = unused
    glm::vec4 aabbMax;   // .xyz = max, .w = unused
    uint32_t  meshIdx;   // index into global mesh draw-data table
    uint32_t  materialIdx;
    uint32_t  pad[2];
};

// Per-mesh draw data stored in a read-only SSBO.
struct alignas(16) MeshDrawData {
    uint32_t firstIndex;
    uint32_t indexCount;
    int32_t  vertexOffset;
    uint32_t pad;
};

// Frustum planes uploaded in a UBO for the culling shader.
struct alignas(16) CullFrustum {
    glm::vec4 planes[6]; // Ax+By+Cz+D, normalised
};

class GPUCulling {
public:
    GPUCulling() = default;
    ~GPUCulling() { destroy(); }

    GPUCulling(const GPUCulling&)            = delete;
    GPUCulling& operator=(const GPUCulling&) = delete;

    void init(std::shared_ptr<Context> ctx, uint32_t maxInstances = 65536);
    void destroy();

    // Call once before rendering: populate instance data from scene objects.
    void uploadInstances(const std::vector<GpuInstance>& instances);

    // Upload the global mesh draw-data table (indexed by GpuInstance::meshIdx).
    void uploadMeshTable(const std::vector<MeshDrawData>& table);

    // Dispatch culling compute shader.
    // frustum: extracted planes from the camera's view-proj matrix.
    void cull(VkCommandBuffer cmd, const CullFrustum& frustum, uint32_t instanceCount);

    // Bind + draw everything that survived culling (one multi-draw call).
    void drawIndirect(VkCommandBuffer cmd);

    // ── Descriptor / pipeline accessors ──────────────────────────────────────
    [[nodiscard]] VkDescriptorSetLayout descLayout()      const { return m_dsLayout; }
    [[nodiscard]] VkDescriptorSet       descSet()         const { return m_ds; }
    [[nodiscard]] VkBuffer              indirectBuffer()  const { return m_indirectBuf.buffer; }
    [[nodiscard]] VkBuffer              countBuffer()     const { return m_countBuf.buffer; }

    void createPipeline(const std::filesystem::path& shaderDir);

private:
    void createDescriptorResources();

    std::shared_ptr<Context> m_ctx;
    uint32_t                 m_maxInstances = 0;

    // Device-local instance + mesh table buffers (updated via staging each frame)
    AllocatedBuffer m_instanceBuf;     // GpuInstance array
    AllocatedBuffer m_meshTableBuf;    // MeshDrawData array
    AllocatedBuffer m_indirectBuf;     // VkDrawIndexedIndirectCommand output
    AllocatedBuffer m_countBuf;        // uint32 surviving draw count
    AllocatedBuffer m_frustumUBO;      // CullFrustum (host-mapped)

    VkDescriptorPool       m_pool     = VK_NULL_HANDLE;
    VkDescriptorSetLayout  m_dsLayout = VK_NULL_HANDLE;
    VkDescriptorSet        m_ds       = VK_NULL_HANDLE;

    VkPipeline       m_pipeline   = VK_NULL_HANDLE;
    VkPipelineLayout m_pipeLayout = VK_NULL_HANDLE;
};

} // namespace vkgfx
