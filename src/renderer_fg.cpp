// src/renderer_fg.cpp
// Frame-graph integration: recordGBuffer/Lighting/Tonemap, buildFrameGraph, render().

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
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::recordGBuffer(VkCommandBuffer cmd, Scene& scene, uint32_t frameIdx) {
    auto& f = m_frames[frameIdx];
    // Use m_offscreenExtent: G-buffer framebuffer was created at this size.
    const VkExtent2D ext = m_offscreenExtent;

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

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_gbufferLayout, 0, 1, &f.gbufferSceneSet, 0, nullptr);
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
}

// ─────────────────────────────────────────────────────────────────────────────
// recordLighting
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::recordLighting(VkCommandBuffer cmd, uint32_t frameIdx) {
    auto& f = m_frames[frameIdx];
    // Use m_offscreenExtent: HDR target framebuffer was created at this size.
    const VkExtent2D ext = m_offscreenExtent;

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
        f.lightingGbufferSet, f.lightingSceneSet, f.iblSet
    };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_lightingLayout, 0, 3, dsets, 0, nullptr);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);
}

// ─────────────────────────────────────────────────────────────────────────────
// recordTonemap
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::recordTonemap(VkCommandBuffer cmd, uint32_t frameIdx) {
    auto& f = m_frames[frameIdx];
    // Use m_swapchain->extent() here: the swapchain framebuffer is owned by
    // Swapchain and is always sized to match the current swapchain extent.
    // By the time this executes, rebuildOffscreenResources() has already been
    // called if sizes diverged, so m_swapchain->extent() == m_offscreenExtent.
    const VkExtent2D ext = m_swapchain->extent();

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
    // Swapchain render pass finalLayout = PRESENT_SRC_KHR — no FG barrier needed.
}

// ─────────────────────────────────────────────────────────────────────────────
// buildFrameGraph
//
// BUG FIX: markOutput(hHDR) was previously called here, which kept only the
// lighting pass alive (it writes hHDR) and culled tonemap (which only reads
// hHDR). The swapchain image was never transitioned → PRESENT_SRC_KHR error.
//
// Fix: don't call markOutput at all. FrameGraph::compile() skips cullPasses()
// when m_outputs is empty and keeps all registered passes alive.
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::buildFrameGraph(Scene& scene, uint32_t frameIdx) {
    const RGHandle hShadow = m_frameGraph->importImage(
        "shadow_map",
        m_shadowMap.image, m_shadowMap.view,
        m_ctx->findDepthFormat(),
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    const RGHandle hAlbedo = m_frameGraph->importImage(
        "gbuf_albedo",
        m_gbuffer[0].image, m_gbuffer[0].view,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_UNDEFINED);

    const RGHandle hNormal = m_frameGraph->importImage(
        "gbuf_normal",
        m_gbuffer[1].image, m_gbuffer[1].view,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED);

    const RGHandle hRMA = m_frameGraph->importImage(
        "gbuf_rma",
        m_gbuffer[2].image, m_gbuffer[2].view,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_UNDEFINED);

    const RGHandle hEmissive = m_frameGraph->importImage(
        "gbuf_emissive",
        m_gbuffer[3].image, m_gbuffer[3].view,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED);

    const RGHandle hShadowCoord = m_frameGraph->importImage(
        "gbuf_shadow_coord",
        m_gbuffer[4].image, m_gbuffer[4].view,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED);

    const RGHandle hDepth = m_frameGraph->importImage(
        "gbuf_depth",
        m_gbuffer[5].image, m_gbuffer[5].view,
        m_ctx->findDepthFormat(),
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    const RGHandle hHDR = m_frameGraph->importImage(
        "hdr_target",
        m_hdrTarget.image, m_hdrTarget.view,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_LAYOUT_UNDEFINED);

    // NOTE: markOutput intentionally NOT called.
    // The swapchain image is the real frame output but is not imported as an FG
    // resource (Swapchain doesn't expose per-image VkImage handles publicly).
    // Without markOutput, compile() skips culling and all registered passes run.

    const bool hasSun = m_cfg.sun.enabled &&
                        scene.dirLight() && scene.dirLight()->enabled();
    if (hasSun) {
        m_frameGraph->addPass("shadow",
            [&](PassBuilder& b) {
                b.writeShadowMap(hShadow);
            },
            [this, &scene](VkCommandBuffer cmd, const FrameGraphResources&) {
                recordShadowPass(cmd, scene);
            });
    }

    m_frameGraph->addPass("gbuffer",
        [&](PassBuilder& b) {
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

    m_frameGraph->addPass("lighting",
        [&](PassBuilder& b) {
            b.read(hAlbedo);
            b.read(hNormal);
            b.read(hRMA);
            b.read(hEmissive);
            b.read(hShadowCoord);
            b.read(hDepth);
            b.read(hShadow);
            b.writeColorAttachment(hHDR);
        },
        [this, frameIdx](VkCommandBuffer cmd, const FrameGraphResources&) {
            recordLighting(cmd, frameIdx);
        });

    m_frameGraph->addPass("tonemap",
        [&](PassBuilder& b) {
            b.read(hHDR);
            // Swapchain image not tracked: its render pass handles the transition
            // UNDEFINED → PRESENT_SRC_KHR via finalLayout.
        },
        [this, frameIdx](VkCommandBuffer cmd, const FrameGraphResources&) {
            recordTonemap(cmd, frameIdx);
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// render
// ─────────────────────────────────────────────────────────────────────────────
void Renderer::render(Scene& scene) {
    auto& f = m_frames[m_frameIdx];

    vkWaitForFences(m_ctx->device(), 1, f.inFlight.ptr(), VK_TRUE, UINT64_MAX);
    uploadMeshMaterials(scene);

    VkSemaphore acquireSem = m_acquireSemaphores[m_frameIdx].get();
    VkResult res = m_swapchain->acquireNext(acquireSem, m_swapIdx);
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
        m_acquireSemaphores[m_frameIdx] = makeSemaphore(m_ctx->device());
        m_swapchain->recreate();
        return;
    }
    const bool needsRecreate = (res == VK_SUBOPTIMAL_KHR);

    // ── Resize check (BEFORE resetting the fence) ─────────────────────────────
    // If the swapchain extent no longer matches our offscreen resources, rebuild
    // them synchronously and skip this frame.  The fence is NOT reset here, so
    // the next frame's WaitForFences call returns immediately (fence is still
    // signaled from the previous submission).
    {
        const VkExtent2D sw = m_swapchain->extent();
        if (sw.width  != m_offscreenExtent.width ||
            sw.height != m_offscreenExtent.height) {
            // rebuildOffscreenResources calls vkDeviceWaitIdle internally.
            rebuildOffscreenResources();
            // Also recreate the swapchain so its framebuffers match the new extent.
            m_swapchain->recreate();
            return;
        }
    }

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

    m_frameGraph->reset();
    buildFrameGraph(scene, m_frameIdx);
    m_frameGraph->compile();
    m_frameGraph->execute(f.cmd);

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
