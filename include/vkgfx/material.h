#pragma once
// include/vkgfx/material.h

#include "types.h"
#include "texture.h"
#include <memory>
#include <string>

namespace vkgfx {

class Context;
class TextureCache;

class PBRMaterial {
public:
    PBRMaterial() = default;

    // ── Scalar setters ────────────────────────────────────────────────────────
    PBRMaterial& setAlbedo   (float r, float g, float b, float a = 1.f);
    PBRMaterial& setRoughness(float v);
    PBRMaterial& setMetallic (float v);
    PBRMaterial& setAO       (float v);
    // Emissive: rgb = color, w = intensity multiplier (e.g. 3.0 for glowing)
    PBRMaterial& setEmissive (float r, float g, float b, float intensity = 1.f);
    PBRMaterial& setEmissive (float intensity); // shorthand: white emissive at given strength

    // ── Texture setters ───────────────────────────────────────────────────────
    PBRMaterial& setAlbedoTexture  (std::shared_ptr<Texture> t);
    PBRMaterial& setNormalTexture  (std::shared_ptr<Texture> t);
    PBRMaterial& setRMATexture     (std::shared_ptr<Texture> t);
    PBRMaterial& setEmissiveTexture(std::shared_ptr<Texture> t);

    // ── Accessors ─────────────────────────────────────────────────────────────
    [[nodiscard]] const PBRParams& params() const { return m_params; }
    [[nodiscard]] PBRParams&       params()       { return m_params; }

    [[nodiscard]] std::shared_ptr<Texture> albedoTex()   const { return m_albedoTex; }
    [[nodiscard]] std::shared_ptr<Texture> normalTex()   const { return m_normalTex; }
    [[nodiscard]] std::shared_ptr<Texture> rmaTex()      const { return m_rmaTex; }
    [[nodiscard]] std::shared_ptr<Texture> emissiveTex() const { return m_emissiveTex; }

    // Descriptor set (set=1 in G-buffer pass) — allocated by the renderer.
    void setDescriptorSet(VkDescriptorSet ds) { m_descriptorSet = ds; }
    [[nodiscard]] VkDescriptorSet descriptorSet() const { return m_descriptorSet; }

    void writeDescriptors(VkDevice device,
                          VkDescriptorSet set,
                          TextureCache& cache) const;

    static VkDescriptorSetLayout createLayout(VkDevice device);

private:
    PBRParams  m_params{};
    std::shared_ptr<Texture> m_albedoTex;
    std::shared_ptr<Texture> m_normalTex;
    std::shared_ptr<Texture> m_rmaTex;
    std::shared_ptr<Texture> m_emissiveTex;
    VkDescriptorSet          m_descriptorSet = VK_NULL_HANDLE;
};

} // namespace vkgfx
