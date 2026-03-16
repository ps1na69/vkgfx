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

    PBRMaterial& setAlbedo   (float r, float g, float b, float a = 1.f);
    PBRMaterial& setRoughness(float v);
    PBRMaterial& setMetallic (float v);
    PBRMaterial& setAO       (float v);
    PBRMaterial& setEmissive (float v);

    PBRMaterial& setAlbedoTexture(std::shared_ptr<Texture> t);
    PBRMaterial& setNormalTexture(std::shared_ptr<Texture> t);
    PBRMaterial& setRMATexture   (std::shared_ptr<Texture> t);

    [[nodiscard]] const PBRParams& params() const { return m_params; }
    [[nodiscard]] PBRParams&       params()       { return m_params; }

    [[nodiscard]] std::shared_ptr<Texture> albedoTex() const { return m_albedoTex; }
    [[nodiscard]] std::shared_ptr<Texture> normalTex() const { return m_normalTex; }
    [[nodiscard]] std::shared_ptr<Texture> rmaTex()    const { return m_rmaTex; }

    // Descriptor set for this material (set=1 in G-buffer pass).
    // Allocated by the renderer; stored here for per-draw binding.
    void setDescriptorSet(VkDescriptorSet ds) { m_descriptorSet = ds; }
    [[nodiscard]] VkDescriptorSet descriptorSet() const { return m_descriptorSet; }

    // Write sampler descriptors into an already-allocated set.
    void writeDescriptors(VkDevice device,
                          VkDescriptorSet set,
                          TextureCache& cache) const;

    static VkDescriptorSetLayout createLayout(VkDevice device);

private:
    PBRParams  m_params{};
    std::shared_ptr<Texture> m_albedoTex;
    std::shared_ptr<Texture> m_normalTex;
    std::shared_ptr<Texture> m_rmaTex;
    VkDescriptorSet          m_descriptorSet = VK_NULL_HANDLE;
};

} // namespace vkgfx
