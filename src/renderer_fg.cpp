// src/renderer_fg.cpp
// Frame-graph integration for Renderer.
//
// This file provides:
//   1. recordGBuffer()   — extracted from the inline code in the old render()
//   2. recordLighting()  — extracted from the inline code in the old render()
//   3. recordTonemap()   — extracted from the inline code in the old render()
//   4. buildFrameGraph() — registers all passes into m_frameGraph each frame
//   5. render()          — updated to drive the frame graph instead of
//                          recording passes directly
//
// INTEGRATION INSTRUCTIONS (one-time):
//   a. In your existing renderer.cpp, DELETE the old render() implementation
//      (lines 1407–1637 in the uploaded file).
//   b. Also DELETE the three inline pass-recording blocks that were embedded
//      in that render() (G-buffer block lines 1503–1552, lighting 1554–1585,
//      tonemap 1587–1613).  recordShadowPass() is unchanged and stays in
//      renderer.cpp.
//   c. Add this file to the build — file(GLOB_RECURSE) picks it up automatically.
//   d. Add #include "frame_graph.h" in renderer.h (already done in the
//      renderer.h delivered alongside this file).
//   e. Update renderer.h constructor initialiser: m_frameGraph(*m_ctx) must be
//      added (see Renderer::Renderer() body note below).

#include <vkgfx/renderer.h>
#include <vkgfx/window.h>
#include <vkgfx/context.h>
#include <vkgfx/swapchain.h>
#include <vkgfx/scene.h>
#include <vkgfx/material.h>
#include <vkgfx/frame_graph.h>

#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <cstring>
#include <iostream>

namespace vkgfx {

// ─────────────────────────────────────────────────────────────────────────────
// recordGBuffer
// Records the G-buffer render pass onto `cmd`.
// Was previously inline inside render(); now a proper private method so it
// can be used as a FrameGraph execute callback.
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::recordGBuffer(VkCommandBuffer cmd, Scene& scene, uint32_t frameIdx) {
    auto& f   = m_frames[frameIdx];
    VkExtent2D ext = m_swapchain->extent();

    VkClearValue clears[6]{};
    clears[5].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rbi{};
    rbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rbi.renderPass      = m_gbufferPass;
    rbi.framebuffer     = m_gbufferFb;
    rbi.renderArea      = {{0, 0}, ext};
    rbi.clearValueCount = 6;
    rbi.pClearValues    = clears;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_gbufferPipeline);
    VkViewport vp{0.f, 0.f, static_cast<float>(ext.width),
                              static_cast<float>(ext.height), 0.f, 1.f};
    VkRect2D sc{{0, 0}, ext};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor (cmd, 0, 1, &sc);

    // set=0: scene UBO
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_gbufferLayout, 0, 1, &f.gbufferSceneSet, 0, nullptr);

    // set=1: default material fallback (meshes with real materials rebind below)
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_gbufferLayout, 1, 1, &f.defaultMaterialSet, 0, nullptr);

    for (Mesh* mesh : scene.visibleMeshes()) {
        MeshPush push{};
        push.model        = mesh->modelMatrix();
        push.normalMatrix = mesh->normalMatrix();
        vkCmdPushConstants(cmd, m_gbufferLayout,
            VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MeshPush), &push);

        if (mesh->material() && mesh->material()->descriptorSet() != VK_NULL_HANDLE) {
            VkDescriptorSet matSet = mesh->material()->descriptorSet();
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                m_gbufferLayout, 1, 1, &matSet, 0, nullptr);
        }

        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, &mesh->vertexBuffer(), offsets);
        vkCmdBindIndexBuffer  (cmd, mesh->indexBuffer(), 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed      (cmd, mesh->indexCount(), 1, 0, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
    // G-buffer render pass transitions all attachments to SHADER_READ_ONLY_OPTIMAL
    // via finalLayout — subpass dependencies ensure visibility to the lighting pass.
}

// ─────────────────────────────────────────────────────────────────────────────
// recordLighting
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::recordLighting(VkCommandBuffer cmd, uint32_t frameIdx) {
    auto& f   = m_frames[frameIdx];
    VkExtent2D ext = m_swapchain->extent();

    VkClearValue clear{};
    VkRenderPassBeginInfo rbi{};
    rbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rbi.renderPass      = m_lightingPass;
    rbi.framebuffer     = m_lightingFb;
    rbi.renderArea      = {{0, 0}, ext};
    rbi.clearValueCount = 1;
    rbi.pClearValues    = &clear;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_lightingPipeline);
    VkViewport vp{0.f, 0.f, static_cast<float>(ext.width),
                              static_cast<float>(ext.height), 0.f, 1.f};
    VkRect2D sc{{0, 0}, ext};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor (cmd, 0, 1, &sc);

    VkDescriptorSet dsets[3] = {
        f.lightingGbufferSet,
        f.lightingSceneSet,
        f.iblSet
    };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_lightingLayout, 0, 3, dsets, 0, nullptr);

    vkCmdDraw(cmd, 3, 1, 0, 0); // fullscreen triangle
    vkCmdEndRenderPass(cmd);
    // HDR target transitions to SHADER_READ_ONLY_OPTIMAL via finalLayout.
}

// ─────────────────────────────────────────────────────────────────────────────
// recordTonemap
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::recordTonemap(VkCommandBuffer cmd, uint32_t frameIdx) {
    auto& f   = m_frames[frameIdx];
    VkExtent2D ext = m_swapchain->extent();

    VkClearValue clear{};
    VkRenderPassBeginInfo rbi{};
    rbi.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rbi.renderPass      = m_swapchain->renderPass();
    rbi.framebuffer     = m_swapchain->framebuffer(m_swapIdx);
    rbi.renderArea      = {{0, 0}, ext};
    rbi.clearValueCount = 1;
    rbi.pClearValues    = &clear;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_tonemapPipeline);
    VkViewport vp{0.f, 0.f, static_cast<float>(ext.width),
                              static_cast<float>(ext.height), 0.f, 1.f};
    VkRect2D sc{{0, 0}, ext};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor (cmd, 0, 1, &sc);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_tonemapLayout, 0, 1, &f.tonemapSet, 0, nullptr);

    struct { float exposure; float gamma; } push{1.0f, 2.2f};
    vkCmdPushConstants(cmd, m_tonemapLayout,
        VK_SHADER_STAGE_FRAGMENT_BIT, 0, 8, &push);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
    // Swapchain render pass transitions image to PRESENT_SRC_KHR via finalLayout.
}

// ─────────────────────────────────────────────────────────────────────────────
// buildFrameGraph
//
// Registers all render passes and their resource dependencies into m_frameGraph
// for this frame.  Called after reset() and before compile().
//
// Resource import state rationale:
//   Shadow map:       pre-transitioned to SHADER_READ_ONLY_OPTIMAL at init.
//                     initLayout = SHADER_READ_ONLY — no FG barrier before shadow pass.
//   G-buffer [0..5]:  cleared each frame; render pass handles UNDEFINED→attachment.
//                     initLayout = UNDEFINED — FG skips barrier before G-buffer pass.
//   HDR target:       cleared each frame; render pass handles UNDEFINED→attachment.
//                     initLayout = UNDEFINED — FG skips barrier before lighting pass.
//
// Because every existing render pass uses appropriate finalLayout transitions
// (→SHADER_READ_ONLY_OPTIMAL), the frame graph emits zero barriers for the
// current pass set.  The graph still provides:
//   • Explicit dependency documentation
//   • Pass culling (shadow skipped when no directional light)
//   • Infrastructure for future transient resources / compute passes
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::buildFrameGraph(Scene& scene, uint32_t frameIdx) {
    // ── Import persistent resources ───────────────────────────────────────────

    // Shadow map — pre-transitioned to SHADER_READ_ONLY_OPTIMAL at initShadowPass().
    // The shadow render pass uses initialLayout=SHADER_READ_ONLY_OPTIMAL (stays there).
    const RGHandle hShadow = m_frameGraph.importImage(
        "shadow_map",
        m_shadowMap.image, m_shadowMap.view,
        m_ctx->findDepthFormat(),
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    // G-buffer attachments — cleared each frame (UNDEFINED initial layout).
    const RGHandle hAlbedo = m_frameGraph.importImage(
        "gbuf_albedo",
        m_gbuffer[0].image, m_gbuffer[0].view,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_UNDEFINED);

    const RGHandle hNormal = m_frameGraph.importImage(
        "gbuf_normal",
        m_gbuffer[1].image, m_gbuffer[1].view,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED);

    const RGHandle hRMA = m_frameGraph.importImage(
        "gbuf_rma",
        m_gbuffer[2].image, m_gbuffer[2].view,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_UNDEFINED);

    const RGHandle hEmissive = m_frameGraph.importImage(
        "gbuf_emissive",
        m_gbuffer[3].image, m_gbuffer[3].view,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED);

    const RGHandle hShadowCoord = m_frameGraph.importImage(
        "gbuf_shadow_coord",
        m_gbuffer[4].image, m_gbuffer[4].view,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED);

    const RGHandle hDepth = m_frameGraph.importImage(
        "gbuf_depth",
        m_gbuffer[5].image, m_gbuffer[5].view,
        m_ctx->findDepthFormat(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    // HDR target — cleared each frame by lighting render pass.
    const RGHandle hHDR = m_frameGraph.importImage(
        "hdr_target",
        m_hdrTarget.image, m_hdrTarget.view,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED);

    // Mark HDR as the frame output — ensures the full pipeline is kept alive.
    m_frameGraph.markOutput(hHDR);

    // ── Shadow pass ────────────────────────────────────────────────────────────
    // Conditionally registered: culled by the frame graph when there is no sun.
    // If the pass IS registered but scene has no directional light, recordShadowPass()
    // returns immediately after vkCmdBeginRenderPass (the clear still executes,
    // which is fine — depth clears to 1.0 = "no shadow" in the lighting pass).
    const bool hasSun = m_cfg.sun.enabled &&
                        scene.dirLight() && scene.dirLight()->enabled();
    if (hasSun) {
        m_frameGraph.addPass("shadow",
            [&](PassBuilder& b) {
                // Shadow map: initialLayout=SHADER_READ_ONLY, finalLayout=SHADER_READ_ONLY.
                // writeShadowMap() declares required=fragmentSampled, result=fragmentSampled.
                // Frame graph checks: current=SHADER_READ_ONLY == required=SHADER_READ_ONLY
                // → no barrier emitted.
                b.writeShadowMap(hShadow);
            },
            [this, &scene](VkCommandBuffer cmd, const FrameGraphResources&) {
                recordShadowPass(cmd, scene);
            });
    }

    // ── G-buffer pass ──────────────────────────────────────────────────────────
    m_frameGraph.addPass("gbuffer",
        [&](PassBuilder& b) {
            // Reads the shadow map (G-buffer writes light-space shadow coords into
            // gbuf_shadow_coord; the shadow map itself is not read here but we track
            // the read from shadow → lighting correctly via the shadow coord write).
            b.writeColorAttachment(hAlbedo);
            b.writeColorAttachment(hNormal);
            b.writeColorAttachment(hRMA);
            b.writeColorAttachment(hEmissive);
            b.writeColorAttachment(hShadowCoord);
            b.writeDepthAttachment(hDepth);
        },
        [this, &scene, frameIdx](VkCommandBuffer cmd, const FrameGraphResources&) {
            recordGBuffer(cmd, scene, frameIdx);
        });

    // ── Lighting pass ──────────────────────────────────────────────────────────
    m_frameGraph.addPass("lighting",
        [&](PassBuilder& b) {
            // Reads all G-buffer outputs.  All are already in SHADER_READ_ONLY_OPTIMAL
            // after the G-buffer render pass finalLayout transitions — no FG barriers.
            b.read(hAlbedo);
            b.read(hNormal);
            b.read(hRMA);
            b.read(hEmissive);
            b.read(hShadowCoord);
            b.read(hDepth);
            b.read(hShadow);
            // Writes HDR target.  Render pass: UNDEFINED → COLOR_ATTACHMENT → SHADER_READ_ONLY.
            b.writeColorAttachment(hHDR);
        },
        [this, frameIdx](VkCommandBuffer cmd, const FrameGraphResources&) {
            recordLighting(cmd, frameIdx);
        });

    // ── Tonemap pass ───────────────────────────────────────────────────────────
    m_frameGraph.addPass("tonemap",
        [&](PassBuilder& b) {
            // Reads HDR target.  After lighting pass it's in SHADER_READ_ONLY_OPTIMAL.
            b.read(hHDR);
            // Swapchain image is managed entirely by m_swapchain->renderPass():
            // we don't import it into the frame graph (no FG handle, no FG barrier).
            // The swapchain render pass handles UNDEFINED → PRESENT via finalLayout.
        },
        [this, frameIdx](VkCommandBuffer cmd, const FrameGraphResources&) {
            recordTonemap(cmd, frameIdx);
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// render — updated: replaces the old render() in renderer.cpp
//
// Structurally identical to the previous render() except lines 1491–1549
// (the inline shadow/gbuffer/lighting/tonemap recording blocks) are replaced
// by the three FrameGraph calls below.
//
// IMPORTANT: delete the old render() implementation from renderer.cpp and
// use only this one.
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::render(Scene& scene) {
    auto& f = m_frames[m_frameIdx];

    // Wait for this frame slot's previous submission to complete.
    // Safe to mutate CPU-side state (material descriptors, UBOs) afterwards.
    vkWaitForFences(m_ctx->device(), 1, f.inFlight.ptr(), VK_TRUE, UINT64_MAX);

    // Allocate and write material descriptor sets for any new meshes added since
    // last frame.  Fence wait above guarantees GPU isn't reading stale sets.
    uploadMeshMaterials(scene);

    // Acquire next swapchain image.
    VkSemaphore acquireSem = m_acquireSemaphores[m_frameIdx].get();
    VkResult res = m_swapchain->acquireNext(acquireSem, m_swapIdx);
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
        // Driver may have signaled acquireSem — recreate it so next acquire is clean.
        m_acquireSemaphores[m_frameIdx] = makeSemaphore(m_ctx->device());
        m_swapchain->recreate();
        return;
    }
    const bool needsRecreate = (res == VK_SUBOPTIMAL_KHR);

    vkResetFences(m_ctx->device(), 1, f.inFlight.ptr());

    // ── Upload SceneUBO ───────────────────────────────────────────────────────
    if (Camera* cam = scene.camera()) {
        cam->setAspect(static_cast<float>(m_swapchain->extent().width) /
                       static_cast<float>(m_swapchain->extent().height));

        SceneUBO sceneData{};
        sceneData.view        = cam->view();
        sceneData.proj        = cam->projection();
        sceneData.viewProj    = cam->viewProj();
        sceneData.invViewProj = glm::inverse(cam->viewProj());
        sceneData.cameraPos   = glm::vec4(cam->position(), 1.f);
        sceneData.viewport    = glm::vec4(
            static_cast<float>(m_swapchain->extent().width),
            static_cast<float>(m_swapchain->extent().height), 0.f, 0.f);

        if (scene.dirLight() && scene.dirLight()->enabled()) {
            glm::vec3 lightDir = glm::normalize(scene.dirLight()->direction());
            glm::vec3 lightPos = -lightDir * 20.f;
            glm::mat4 lightView = glm::lookAt(lightPos,
                                               glm::vec3(0.f),
                                               glm::vec3(0.f, 1.f, 0.f));
            glm::mat4 lightProj = glm::ortho(-15.f, 15.f, -15.f, 15.f, 0.1f, 100.f);
            lightProj[1][1] *= -1.f;
            sceneData.lightViewProj = lightProj * lightView;
        }

        void* mapped = nullptr;
        vmaMapMemory(m_ctx->vma(),
            static_cast<VmaAllocation>(f.sceneUbo.allocation), &mapped);
        std::memcpy(mapped, &sceneData, sizeof(SceneUBO));
        vmaUnmapMemory(m_ctx->vma(),
            static_cast<VmaAllocation>(f.sceneUbo.allocation));
    }

    // ── Upload LightUBO ───────────────────────────────────────────────────────
    LightUBO lightData{};
    const float iblInt = (m_cfg.ibl.enabled && m_ibl->isReady())
                       ? m_ibl->intensity() : 0.f;
    scene.fillLightUBO(lightData, iblInt,
                       static_cast<uint32_t>(m_cfg.gbufferDebug));

    if (!scene.dirLight()) {
        lightData.sunDirection = glm::vec4(m_cfg.sun.direction[0],
                                           m_cfg.sun.direction[1],
                                           m_cfg.sun.direction[2], 0.f);
        lightData.sunColor     = glm::vec4(m_cfg.sun.color[0],
                                           m_cfg.sun.color[1],
                                           m_cfg.sun.color[2],
                                           m_cfg.sun.intensity);
        lightData.sunFlags.x   = m_cfg.sun.enabled ? 1u : 0u;
        lightData.sunFlags.y   = m_cfg.sun.enabled ? 1u : 0u;
    }

    void* lmapped = nullptr;
    vmaMapMemory(m_ctx->vma(),
        static_cast<VmaAllocation>(f.lightUbo.allocation), &lmapped);
    std::memcpy(lmapped, &lightData, sizeof(LightUBO));
    vmaUnmapMemory(m_ctx->vma(),
        static_cast<VmaAllocation>(f.lightUbo.allocation));

    // ── Record ────────────────────────────────────────────────────────────────
    vkResetCommandBuffer(f.cmd, 0);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(f.cmd, &bi);

    // ── Frame graph ───────────────────────────────────────────────────────────
    // reset()         — clear pass/resource lists from last frame
    // buildFrameGraph — register passes and resource I/O for this frame
    // compile()       — topological sort, cull unused passes, build barrier lists
    // execute()       — emit barriers + invoke pass callbacks in sorted order
    m_frameGraph.reset();
    buildFrameGraph(scene, m_frameIdx);
    m_frameGraph.compile();
    m_frameGraph.execute(f.cmd);
    // ── end frame graph ───────────────────────────────────────────────────────

    vkEndCommandBuffer(f.cmd);

    // ── Submit ────────────────────────────────────────────────────────────────
    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount   = 1; si.pWaitSemaphores   = &acquireSem;
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1; si.pCommandBuffers   = &f.cmd;
    VkSemaphore renderDoneSem = m_renderFinishedSems[m_swapIdx].get();
    si.signalSemaphoreCount = 1; si.pSignalSemaphores = &renderDoneSem;
    vkQueueSubmit(m_ctx->graphicsQ(), 1, &si, f.inFlight.get());

    res = m_swapchain->present(renderDoneSem, m_swapIdx);
    if (needsRecreate || res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR)
        m_swapchain->recreate();

    m_frameIdx = (m_frameIdx + 1) % MAX_FRAMES_IN_FLIGHT;
}

} // namespace vkgfx
