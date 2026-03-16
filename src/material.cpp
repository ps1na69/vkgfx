// src/material.cpp
#include <vkgfx/material.h>
#include <vkgfx/texture.h>
#include <vkgfx/context.h>
#include <vulkan/vulkan.h>
#include <stdexcept>

namespace vkgfx {

// ── Fluent setters ────────────────────────────────────────────────────────────

PBRMaterial& PBRMaterial::setAlbedo(float r, float g, float b, float a) {
    m_params.albedo = {r, g, b, a};
    return *this;
}
PBRMaterial& PBRMaterial::setRoughness(float v)  { m_params.roughness  = v; return *this; }
PBRMaterial& PBRMaterial::setMetallic(float v)   { m_params.metallic   = v; return *this; }
PBRMaterial& PBRMaterial::setAO(float v)         { m_params.ao         = v; return *this; }
PBRMaterial& PBRMaterial::setEmissive(float v)   { m_params.emissive   = v; return *this; }

PBRMaterial& PBRMaterial::setAlbedoTexture(std::shared_ptr<Texture> t) {
    m_albedoTex = std::move(t);
    m_params.hasAlbedoTex = m_albedoTex ? 1u : 0u;
    return *this;
}
PBRMaterial& PBRMaterial::setNormalTexture(std::shared_ptr<Texture> t) {
    m_normalTex = std::move(t);
    m_params.hasNormalTex = m_normalTex ? 1u : 0u;
    return *this;
}
PBRMaterial& PBRMaterial::setRMATexture(std::shared_ptr<Texture> t) {
    m_rmaTex = std::move(t);
    m_params.hasRmaTex = m_rmaTex ? 1u : 0u;
    return *this;
}

// ── createLayout ─────────────────────────────────────────────────────────────

VkDescriptorSetLayout PBRMaterial::createLayout(VkDevice device) {
    // set=1: binding 0=albedoTex, 1=normalTex, 2=rmaTex, 3=PBRParams UBO
    VkDescriptorSetLayoutBinding bindings[4]{};
    for (int i = 0; i < 3; ++i) {
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    bindings[3].binding         = 3;
    bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = 4;
    ci.pBindings    = bindings;

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &layout) != VK_SUCCESS)
        throw std::runtime_error("[vkgfx] PBRMaterial descriptor set layout failed");
    return layout;
}

// ── writeDescriptors ─────────────────────────────────────────────────────────

void PBRMaterial::writeDescriptors(VkDevice device,
                                    VkDescriptorSet set,
                                    TextureCache& cache) const {
    // Always provide valid textures — use fallback solid colours for missing slots
    auto albedo = m_albedoTex ? m_albedoTex : cache.solid(255, 255, 255);
    auto normal = m_normalTex ? m_normalTex : cache.solid(128, 128, 255); // flat normal
    auto rma    = m_rmaTex    ? m_rmaTex    : cache.solid(
        static_cast<uint8_t>(m_params.roughness * 255),
        static_cast<uint8_t>(m_params.metallic  * 255),
        static_cast<uint8_t>(m_params.ao        * 255));

    VkDescriptorImageInfo imgs[3]{};
    imgs[0] = {albedo->sampler(), albedo->view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imgs[1] = {normal->sampler(), normal->view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imgs[2] = {rma->sampler(),    rma->view(),    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    VkWriteDescriptorSet writes[4]{};
    for (int i = 0; i < 3; ++i) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = set;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo      = &imgs[i];
    }

    // Upload PBRParams as a small UBO — caller must provide a VkBuffer for binding 3.
    // For simplicity, the renderer uploads params to a staging buffer and updates here.
    // The UBO binding (3) is handled by the renderer via a per-material UBO.
    // Here we only write the sampler descriptors.
    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
}

} // namespace vkgfx
