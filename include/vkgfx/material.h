#pragma once
#include "texture.h"
#include <filesystem>

namespace vkgfx {

// ─── Material properties (CPU side) ──────────────────────────────────────────
struct alignas(16) PBRProperties {
    Vec4  albedo        = {1.f, 1.f, 1.f, 1.f};
    float metallic      = 0.f;
    float roughness     = 0.5f;
    float ao            = 1.f;
    float emissiveScale = 0.f;
    Vec4  emissiveColor = {0.f, 0.f, 0.f, 1.f};
    int   useAlbedoMap  = 0;
    int   useNormalMap  = 0;
    int   useMetalRoughMap = 0;
    int   useAOMap      = 0;
    int   useEmissiveMap = 0;
    int   _pad[3];
};

struct alignas(16) PhongProperties {
    Vec4  ambient  = {0.1f, 0.1f, 0.1f, 1.f};
    Vec4  diffuse  = {0.8f, 0.8f, 0.8f, 1.f};
    Vec4  specular = {1.f, 1.f, 1.f, 1.f};
    float shininess = 32.f;
    int   useDiffuseMap  = 0;
    int   useNormalMap   = 0;
    int   _pad;
};

// ─── Pipeline settings (baked into pipeline at creation) ─────────────────────
struct PipelineSettings {
    CullMode       cullMode    = CullMode::Back;
    PolygonMode    polygonMode = PolygonMode::Fill;
    bool           depthTest   = true;
    bool           depthWrite  = true;
    bool           alphaBlend  = false;
    MSAASamples    msaa        = MSAASamples::x1;
    VkRenderPass   renderPass  = VK_NULL_HANDLE;  // set by renderer
};

// ─── Base Material ────────────────────────────────────────────────────────────
class Material {
public:
    virtual ~Material() = default;

    virtual void setTexture(uint32_t slot, std::shared_ptr<Texture> tex) = 0;
    [[nodiscard]] virtual std::shared_ptr<Texture> getTexture(uint32_t slot) const = 0;
    [[nodiscard]] virtual const void* propertiesData()  const = 0;
    [[nodiscard]] virtual size_t      propertiesSize()  const = 0;
    [[nodiscard]] virtual std::string_view shaderName() const = 0;
    [[nodiscard]] virtual bool isDirty() const { return m_dirty != 0; }
    /** Returns true if this specific frame's descriptor set needs re-writing. */
    [[nodiscard]] bool isFrameDirty(uint32_t frameIdx) const {
        return (m_dirty & (1u << frameIdx)) != 0;
    }
    void markClean(uint32_t frameIdx) { m_dirty &= ~(1u << frameIdx); }
    void markDirty() { m_dirty = (1u << MAX_FRAMES_IN_FLIGHT) - 1u; } // all frames

    PipelineSettings pipelineSettings;

protected:
    uint32_t m_dirty = (1u << MAX_FRAMES_IN_FLIGHT) - 1u; // all frames dirty initially
};

//PBR Material
class PBRMaterial : public Material {
public:
    enum TextureSlot : uint32_t {
        ALBEDO      = 0,
        NORMAL      = 1,
        METALROUGH  = 2,
        AO          = 3,
        EMISSIVE    = 4,
    };

    PBRMaterial() = default;

    // Fluent setters
    PBRMaterial& setAlbedo(Vec4 v)        { m_props.albedo = v;              markDirty(); return *this; }
    PBRMaterial& setMetallic(float v)     { m_props.metallic = v;            markDirty(); return *this; }
    PBRMaterial& setRoughness(float v)    { m_props.roughness = v;           markDirty(); return *this; }
    PBRMaterial& setEmissive(Vec4 c, float s = 1.f) {
        m_props.emissiveColor = c; m_props.emissiveScale = s; markDirty(); return *this;
    }

    void setTexture(uint32_t slot, std::shared_ptr<Texture> tex) override {
        if (slot < MAX_TEXTURES_PER_MAT) { m_textures[slot] = std::move(tex); markDirty(); }
    }
    [[nodiscard]] std::shared_ptr<Texture> getTexture(uint32_t slot) const override {
        return (slot < MAX_TEXTURES_PER_MAT) ? m_textures[slot] : nullptr;
    }
    [[nodiscard]] const void* propertiesData() const override { return &m_props; }
    [[nodiscard]] size_t      propertiesSize() const override { return sizeof(PBRProperties); }
    [[nodiscard]] std::string_view shaderName() const override { return "pbr"; }
    [[nodiscard]] const PBRProperties& properties() const { return m_props; }
    PBRProperties& properties() { return m_props; }

    void updateTextureFlags() {
        m_props.useAlbedoMap     = m_textures[ALBEDO]     ? 1 : 0;
        m_props.useNormalMap     = m_textures[NORMAL]     ? 1 : 0;
        m_props.useMetalRoughMap = m_textures[METALROUGH] ? 1 : 0;
        m_props.useAOMap         = m_textures[AO]         ? 1 : 0;
        m_props.useEmissiveMap   = m_textures[EMISSIVE]   ? 1 : 0;
    }

private:
    PBRProperties m_props;
    std::array<std::shared_ptr<Texture>, MAX_TEXTURES_PER_MAT> m_textures{};
};

//Phong Material
class PhongMaterial : public Material {
public:
    enum TextureSlot : uint32_t { DIFFUSE = 0, NORMAL = 1 };

    PhongMaterial() = default;

    PhongMaterial& setAmbient(Vec4 v)  { m_props.ambient  = v; markDirty(); return *this; }
    PhongMaterial& setDiffuse(Vec4 v)  { m_props.diffuse  = v; markDirty(); return *this; }
    PhongMaterial& setSpecular(Vec4 v) { m_props.specular = v; markDirty(); return *this; }
    PhongMaterial& setShininess(float v){ m_props.shininess = v; markDirty(); return *this; }

    void setTexture(uint32_t slot, std::shared_ptr<Texture> tex) override {
        if (slot < 2) { m_textures[slot] = std::move(tex); markDirty(); }
    }
    [[nodiscard]] std::shared_ptr<Texture> getTexture(uint32_t slot) const override {
        return (slot < 2) ? m_textures[slot] : nullptr;
    }
    [[nodiscard]] const void* propertiesData() const override { return &m_props; }
    [[nodiscard]] size_t      propertiesSize() const override { return sizeof(PhongProperties); }
    [[nodiscard]] std::string_view shaderName() const override { return "phong"; }

private:
    PhongProperties m_props;
    std::array<std::shared_ptr<Texture>, 2> m_textures{};
};

} // namespace vkgfx
