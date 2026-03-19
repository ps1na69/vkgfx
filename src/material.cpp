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
PBRMaterial& PBRMaterial::setRoughness(float v) {
    m_params.pbr.x = v; return *this;
}
PBRMaterial& PBRMaterial::setMetallic(float v) {
    m_params.pbr.y = v; return *this;
}
PBRMaterial& PBRMaterial::setAO(float v) {
    m_params.pbr.z = v; return *this;
}
PBRMaterial& PBRMaterial::setEmissive(float r, float g, float b, float intensity) {
    m_params.emissive = glm::vec4(r, g, b, intensity);
    return *this;
}
PBRMaterial& PBRMaterial::setEmissive(float intensity) {
    m_params.emissive = glm::vec4(1.f, 1.f, 1.f, intensity);
    return *this;
}

PBRMaterial& PBRMaterial::setAlbedoTexture(std::shared_ptr<Texture> t) {
    m_albedoTex = std::move(t);
    m_params.texFlags.x = m_albedoTex ? 1u : 0u;
    return *this;
}
PBRMaterial& PBRMaterial::setNormalTexture(std::shared_ptr<Texture> t) {
    m_normalTex = std::move(t);
    m_params.texFlags.y = m_normalTex ? 1u : 0u;
    return *this;
}
PBRMaterial& PBRMaterial::setRMATexture(std::shared_ptr<Texture> t) {
    m_rmaTex = std::move(t);
    m_params.texFlags.z = m_rmaTex ? 1u : 0u;
    return *this;
}
PBRMaterial& PBRMaterial::setEmissiveTexture(std::shared_ptr<Texture> t) {
    m_emissiveTex = std::move(t);
    m_params.texFlags.w = m_emissiveTex ? 1u : 0u;
    return *this;
}

// ── createLayout ─────────────────────────────────────────────────────────────

VkDescriptorSetLayout PBRMaterial::createLayout(VkDevice device) {
    // set=1: 0=albedoTex, 1=normalTex, 2=rmaTex, 3=emissiveTex, 4=PBRParams UBO
    VkDescriptorSetLayoutBinding b[5]{};
    for (int i = 0; i < 4; ++i) {
        b[i].binding         = i;
        b[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    b[4].binding         = 4;
    b[4].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[4].descriptorCount = 1;
    b[4].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = 5;
    ci.pBindings    = b;

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &layout) != VK_SUCCESS)
        throw std::runtime_error("[vkgfx] PBRMaterial descriptor set layout failed");
    return layout;
}

// ── writeDescriptors ─────────────────────────────────────────────────────────

void PBRMaterial::writeDescriptors(VkDevice device,
                                    VkDescriptorSet set,
                                    TextureCache& cache) const {
    auto albedo = m_albedoTex ? m_albedoTex : cache.solid(255, 255, 255);
    auto normal = m_normalTex ? m_normalTex : cache.solid(128, 128, 255);
    auto rma    = m_rmaTex    ? m_rmaTex    : cache.solid(
        static_cast<uint8_t>(m_params.pbr.x * 255),
        static_cast<uint8_t>(m_params.pbr.y * 255),
        static_cast<uint8_t>(m_params.pbr.z * 255));

    VkDescriptorImageInfo imgs[3]{};
    imgs[0] = {albedo->sampler(), albedo->view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imgs[1] = {normal->sampler(), normal->view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    imgs[2] = {rma->sampler(),    rma->view(),    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

    VkWriteDescriptorSet ws[3]{};
    for (int i = 0; i < 3; ++i) {
        ws[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws[i].dstSet          = set;
        ws[i].dstBinding      = i;
        ws[i].descriptorCount = 1;
        ws[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ws[i].pImageInfo      = &imgs[i];
    }
    vkUpdateDescriptorSets(device, 3, ws, 0, nullptr);
}

} // namespace vkgfx
