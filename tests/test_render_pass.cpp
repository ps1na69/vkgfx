// tests/test_render_pass.cpp
// Creates a headless Vulkan context and verifies G-buffer render pass creation.
// Fails if any Vulkan call returns an error or if validation layers fire.

#include <catch2/catch_test_macros.hpp>
#include <vkgfx/context.h>
#include <vkgfx/types.h>
#include <vulkan/vulkan.h>
#include <array>
#include <stdexcept>

using namespace vkgfx;

TEST_CASE("Headless Context initialises without Vulkan errors") {
    ContextConfig cc;
    cc.validation = true;
    cc.headless   = true;

    Context ctx(cc);
    REQUIRE(ctx.isValid());
    REQUIRE(ctx.device()   != VK_NULL_HANDLE);
    REQUIRE(ctx.instance() != VK_NULL_HANDLE);
}

TEST_CASE("G-buffer render pass creates successfully with correct attachment layouts") {
    ContextConfig cc;
    cc.validation = true;
    cc.headless   = true;
    Context ctx(cc);
    REQUIRE(ctx.isValid());

    VkFormat depthFmt = ctx.findDepthFormat();

    // 4 attachments: albedo, normal, RMA, depth
    // finalLayout MUST be SHADER_READ_ONLY_OPTIMAL so lighting pass descriptor
    // writes (which use SHADER_READ_ONLY_OPTIMAL) are valid.
    VkAttachmentDescription atts[4]{};

    // albedo — RGBA8
    atts[0].format         = VK_FORMAT_R8G8B8A8_UNORM;
    atts[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    atts[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    atts[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[0].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; // FIX: must match descriptor write

    // normal — RGBA16F
    atts[1]        = atts[0];
    atts[1].format = VK_FORMAT_R16G16B16A16_SFLOAT;

    // RMA — RGBA8
    atts[2] = atts[0];

    // depth
    atts[3].format         = depthFmt;
    atts[3].samples        = VK_SAMPLE_COUNT_1_BIT;
    atts[3].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[3].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    atts[3].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[3].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[3].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[3].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colRefs[3]{};
    for (int i = 0; i < 3; ++i) {
        colRefs[i].attachment = i;
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

    VkRenderPass rp = VK_NULL_HANDLE;
    VkResult res = vkCreateRenderPass(ctx.device(), &rpi, nullptr, &rp);

    REQUIRE(res == VK_SUCCESS);
    REQUIRE(rp  != VK_NULL_HANDLE);

    vkDestroyRenderPass(ctx.device(), rp, nullptr);
}

TEST_CASE("Depth format is findable on this device") {
    ContextConfig cc;
    cc.headless = true;
    Context ctx(cc);
    REQUIRE(ctx.isValid());

    VkFormat fmt = VK_FORMAT_UNDEFINED;
    REQUIRE_NOTHROW(fmt = ctx.findDepthFormat());
    REQUIRE(fmt != VK_FORMAT_UNDEFINED);

    // Must be one of the accepted depth formats
    bool valid = (fmt == VK_FORMAT_D32_SFLOAT)
              || (fmt == VK_FORMAT_D32_SFLOAT_S8_UINT)
              || (fmt == VK_FORMAT_D24_UNORM_S8_UINT);
    REQUIRE(valid);
}

TEST_CASE("AllocatedBuffer upload and destroy cycle does not leak") {
    ContextConfig cc;
    cc.headless = true;
    Context ctx(cc);
    REQUIRE(ctx.isValid());

    uint32_t data[] = {1, 2, 3, 4};
    AllocatedBuffer buf = ctx.uploadBuffer(data, sizeof(data),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

    REQUIRE(buf.buffer     != VK_NULL_HANDLE);
    REQUIRE(buf.allocation != nullptr);

    ctx.destroyBuffer(buf);

    REQUIRE(buf.buffer     == VK_NULL_HANDLE);
    REQUIRE(buf.allocation == nullptr);
}
