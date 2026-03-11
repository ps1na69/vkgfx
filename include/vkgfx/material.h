#pragma once
// material.h — PBR material system for the deferred shading pipeline.
//
// Only one material type: PBRMaterial.
// Phong / Unlit materials are removed — the deferred pipeline exclusively
// uses PBR metallic-roughness (matching glTF 2.0 and geometry.frag bindings).
//
// Descriptor set layout (set 1, geometry pass):
//   binding 0: sampler2D albedoMap
//   binding 1: sampler2D normalMap
//   binding 2: sampler2D metallicRoughnessMap  (B=metallic, G=roughness)
//   binding 3: sampler2D emissiveMap
//   binding 4: sampler2D aoMap
//   binding 5: uniform MaterialUBO

#include "texture.h"
#include "types.h"
#include <filesystem>

namespace vkgfx {

class PBRMaterial {
public:
    enum Slot : uint32_t {
        ALBEDO      = 0,
        NORMAL      = 1,
        METALROUGH  = 2,
        EMISSIVE    = 3,
        AO          = 4,
    };

    PBRMaterial() = default;

    // Fluent property setters — mark GPU buffer dirty so renderer re-uploads.
    PBRMaterial& setAlbedo(Vec4 v)       { m_ubo.albedoFactor     = v;   markDirty(); return *this; }
    PBRMaterial& setMetallic(float v)    { m_ubo.metallicFactor    = v;   markDirty(); return *this; }
    PBRMaterial& setRoughness(float v)   { m_ubo.roughnessFactor   = v;   markDirty(); return *this; }
    PBRMaterial& setEmissive(float s)    { m_ubo.emissiveStrength  = s;   markDirty(); return *this; }
    PBRMaterial& setAlphaCutoff(float v) { m_ubo.alphaCutoff       = v;   markDirty(); return *this; }

    void setTexture(uint32_t slot, std::shared_ptr<Texture> tex) {
        if (slot < MAX_TEXTURES_PER_MAT) { m_textures[slot] = std::move(tex); markDirty(); }
    }
    [[nodiscard]] std::shared_ptr<Texture> getTexture(uint32_t slot) const {
        return (slot < MAX_TEXTURES_PER_MAT) ? m_textures[slot] : nullptr;
    }

    [[nodiscard]] const MaterialUBO& ubo()  const { return m_ubo; }
    [[nodiscard]] MaterialUBO&       ubo()        { return m_ubo; }

    // Dirty tracking — one bit per frame-in-flight.
    [[nodiscard]] bool isFrameDirty(uint32_t fi) const { return (m_dirty & (1u << fi)) != 0; }
    void markClean(uint32_t fi) { m_dirty &= ~(1u << fi); }
    void markDirty()            { m_dirty  = (1u << MAX_FRAMES_IN_FLIGHT) - 1u; }

private:
    MaterialUBO m_ubo{};
    std::array<std::shared_ptr<Texture>, MAX_TEXTURES_PER_MAT> m_textures{};
    uint32_t    m_dirty = (1u << MAX_FRAMES_IN_FLIGHT) - 1u;  // initially dirty
};

} // namespace vkgfx
