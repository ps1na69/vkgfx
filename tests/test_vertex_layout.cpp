// tests/test_vertex_layout.cpp
#include <catch2/catch_test_macros.hpp>
#include <vkgfx/types.h>
#include <vkgfx/config.h>   // for GBufferDebugView
#include <vulkan/vulkan.h>

using namespace vkgfx;

static_assert(std::is_standard_layout_v<Vertex>,
    "Vertex must be standard-layout for offsetof to be well-defined");

static_assert(sizeof(MeshPush) == 128,
    "MeshPush must be 128 bytes — 2x mat4");

static_assert(alignof(SceneUBO) == 16, "SceneUBO must be alignas(16)");
static_assert(alignof(LightUBO) == 16, "LightUBO must be alignas(16)");

TEST_CASE("Vertex binding description is correct") {
    auto b = Vertex::bindingDescription();
    REQUIRE(b.binding   == 0);
    REQUIRE(b.stride    == sizeof(Vertex));
    REQUIRE(b.inputRate == VK_VERTEX_INPUT_RATE_VERTEX);
}

TEST_CASE("Vertex attribute locations match gbuffer.vert shader") {
    auto attribs = Vertex::attributeDescriptions();
    REQUIRE(attribs.size() == 4);

    // location 0 — position vec3
    REQUIRE(attribs[0].location == 0);
    REQUIRE(attribs[0].format   == VK_FORMAT_R32G32B32_SFLOAT);
    REQUIRE(attribs[0].offset   == offsetof(Vertex, position));

    // location 1 — normal vec3
    REQUIRE(attribs[1].location == 1);
    REQUIRE(attribs[1].format   == VK_FORMAT_R32G32B32_SFLOAT);
    REQUIRE(attribs[1].offset   == offsetof(Vertex, normal));

    // location 2 — tangent vec4
    REQUIRE(attribs[2].location == 2);
    REQUIRE(attribs[2].format   == VK_FORMAT_R32G32B32A32_SFLOAT);
    REQUIRE(attribs[2].offset   == offsetof(Vertex, tangent));

    // location 3 — uv vec2
    REQUIRE(attribs[3].location == 3);
    REQUIRE(attribs[3].format   == VK_FORMAT_R32G32_SFLOAT);
    REQUIRE(attribs[3].offset   == offsetof(Vertex, uv));
}

TEST_CASE("Vertex stride equals sum of component sizes") {
    constexpr size_t expected = 3*4 + 3*4 + 4*4 + 2*4; // 48 bytes
    REQUIRE(sizeof(Vertex) == expected);
}

TEST_CASE("MeshPush is exactly 128 bytes") {
    REQUIRE(sizeof(MeshPush) == 128);
}

TEST_CASE("LightUBO fits in a UBO (max 65536 bytes)") {
    REQUIRE(sizeof(LightUBO) <= 65536);
}

TEST_CASE("MAX_POINT_LIGHTS matches lighting.frag array size") {
    // lighting.frag declares: PointLight points[8]
    REQUIRE(MAX_POINT_LIGHTS == 8);
}

TEST_CASE("GBufferDebugView values match lighting.frag switch cases") {
    REQUIRE(static_cast<uint8_t>(GBufferDebugView::None)      == 0);
    REQUIRE(static_cast<uint8_t>(GBufferDebugView::Albedo)    == 1);
    REQUIRE(static_cast<uint8_t>(GBufferDebugView::Normal)    == 2);
    REQUIRE(static_cast<uint8_t>(GBufferDebugView::Roughness) == 3);
    REQUIRE(static_cast<uint8_t>(GBufferDebugView::Metallic)  == 4);
    REQUIRE(static_cast<uint8_t>(GBufferDebugView::Depth)     == 5);
    REQUIRE(static_cast<uint8_t>(GBufferDebugView::AO)        == 6);
}
