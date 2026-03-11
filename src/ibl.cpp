#include "vkgfx/ibl.h"
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <vector>
#include <array>

// stb_image for HDR loading (implementation defined in texture.cpp)
#include <stb_image.h>

namespace vkgfx {

// ── helpers ───────────────────────────────────────────────────────────────────
static VkShaderModule loadSpv(VkDevice dev, const std::vector<uint32_t>& code) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size() * 4;
    ci.pCode    = code.data();
    VkShaderModule m;
    VK_CHECK(vkCreateShaderModule(dev, &ci, nullptr, &m));
    return m;
}

// Minimal pipeline runner used to bake offline textures via compute shaders
// embedded as SPIR-V uint32 arrays.
struct OfflineCompute {
    VkDevice          dev;
    VkPipeline        pipeline     = VK_NULL_HANDLE;
    VkPipelineLayout  pipeLayout   = VK_NULL_HANDLE;
    VkDescriptorPool  pool         = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
    VkDescriptorSet   ds           = VK_NULL_HANDLE;
    VkShaderModule    shader       = VK_NULL_HANDLE;

    void destroy() {
        if (pipeline)  { vkDestroyPipeline(dev, pipeline, nullptr); pipeline = VK_NULL_HANDLE; }
        if (pipeLayout){ vkDestroyPipelineLayout(dev, pipeLayout, nullptr); pipeLayout = VK_NULL_HANDLE; }
        if (pool)      { vkDestroyDescriptorPool(dev, pool, nullptr); pool = VK_NULL_HANDLE; }
        if (dsLayout)  { vkDestroyDescriptorSetLayout(dev, dsLayout, nullptr); dsLayout = VK_NULL_HANDLE; }
        if (shader)    { vkDestroyShaderModule(dev, shader, nullptr); shader = VK_NULL_HANDLE; }
    }
};

// ── IBLProbe::createSamplers ──────────────────────────────────────────────────
void IBLProbe::createSamplers() {
    auto dev = m_ctx->device();

    // Cubemap sampler — trilinear, 8x aniso for prefiltered specular
    {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter    = VK_FILTER_LINEAR;
        si.minFilter    = VK_FILTER_LINEAR;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod       = static_cast<float>(PREFILTERED_MIPS);
        si.anisotropyEnable  = VK_TRUE;
        si.maxAnisotropy     = 8.f;
        VK_CHECK(vkCreateSampler(dev, &si, nullptr, &m_cubeSampler));
    }

    // BRDF LUT sampler — linear, clamp
    {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter    = VK_FILTER_LINEAR;
        si.minFilter    = VK_FILTER_LINEAR;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod       = 1.f;
        VK_CHECK(vkCreateSampler(dev, &si, nullptr, &m_brdfSampler));
    }
}

// ── IBLProbe::destroy ─────────────────────────────────────────────────────────
void IBLProbe::destroy() {
    if (!m_ctx) return;
    auto dev = m_ctx->device();
    m_ctx->destroyImage(m_equirect);
    m_ctx->destroyImage(m_envCubemap);
    m_ctx->destroyImage(m_irradiance);
    m_ctx->destroyImage(m_prefiltered);
    m_ctx->destroyImage(m_brdfLUT);
    if (m_cubeSampler) { vkDestroySampler(dev, m_cubeSampler, nullptr); m_cubeSampler = VK_NULL_HANDLE; }
    if (m_brdfSampler) { vkDestroySampler(dev, m_brdfSampler, nullptr); m_brdfSampler = VK_NULL_HANDLE; }
    m_ctx.reset();
    m_ready = false;
}

// ── CPU baking helpers ────────────────────────────────────────────────────────
// Rather than shipping embedded SPIR-V, we generate the probe maps on the CPU
// using the standard cubemap projection math. This is done once at startup and
// results are stored in device-local images.

static constexpr float PI = 3.14159265359f;

// Convert direction to equirectangular UV
static glm::vec2 dirToEquirect(glm::vec3 d) {
    float phi   = std::atan2(d.z, d.x);       // [-π, π]
    float theta = std::asin(glm::clamp(d.y, -1.f, 1.f)); // [-π/2, π/2]
    return { (phi + PI) / (2.f * PI), (theta + PI * 0.5f) / PI };
}

// Bilinear sample from float RGBA equirectangular buffer
static glm::vec3 sampleEquirect(const float* data, int W, int H, glm::vec3 dir) {
    auto uv   = dirToEquirect(glm::normalize(dir));
    float u   = uv.x * (W - 1);
    float v   = (1.f - uv.y) * (H - 1);
    int   x0  = static_cast<int>(u), y0 = static_cast<int>(v);
    int   x1  = std::min(x0 + 1, W - 1), y1 = std::min(y0 + 1, H - 1);
    float fx  = u - x0, fy = v - y0;
    auto fetch = [&](int x, int y) -> glm::vec3 {
        const float* p = data + (y * W + x) * 4;
        return {p[0], p[1], p[2]};
    };
    return glm::mix(glm::mix(fetch(x0,y0), fetch(x1,y0), fx),
                    glm::mix(fetch(x0,y1), fetch(x1,y1), fx), fy);
}

// Face direction for cubemap face f, texel (u,v) in [-1,1]
static glm::vec3 faceDir(int face, float u, float v) {
    switch (face) {
        case 0: return glm::normalize(glm::vec3( 1,  v, -u)); // +X
        case 1: return glm::normalize(glm::vec3(-1,  v,  u)); // -X
        case 2: return glm::normalize(glm::vec3( u,  1, -v)); // +Y
        case 3: return glm::normalize(glm::vec3( u, -1,  v)); // -Y
        case 4: return glm::normalize(glm::vec3( u,  v,  1)); // +Z
        default: return glm::normalize(glm::vec3(-u,  v, -1)); // -Z
    }
}

// ── Upload 6-face RGBA32F data into a cubemap AllocatedImage ─────────────────
static void uploadCubemap(const Context& ctx, AllocatedImage& img,
                           uint32_t size, uint32_t mips,
                           const std::vector<std::vector<glm::vec4>>& faceData)
{
    // Each face level
    VkDeviceSize totalBytes = 0;
    std::vector<VkDeviceSize> offsets;
    for (uint32_t m = 0; m < mips; ++m) {
        uint32_t s = std::max(1u, size >> m);
        for (int f = 0; f < 6; ++f) {
            offsets.push_back(totalBytes);
            totalBytes += s * s * sizeof(glm::vec4);
        }
    }

    AllocatedBuffer staging = ctx.createBuffer(totalBytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, true);

    auto* dst = static_cast<uint8_t*>(staging.mapped);
    for (size_t i = 0; i < faceData.size(); ++i)
        std::memcpy(dst + offsets[i], faceData[i].data(),
                    faceData[i].size() * sizeof(glm::vec4));

    // Transition to TRANSFER_DST
    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 6};
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0,nullptr, 0,nullptr, 1,&b);

        std::vector<VkBufferImageCopy> copies;
        for (uint32_t m = 0; m < mips; ++m) {
            uint32_t s = std::max(1u, size >> m);
            for (uint32_t f = 0; f < 6; ++f) {
                VkBufferImageCopy c{};
                c.bufferOffset = offsets[m * 6 + f];
                c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, f, 1};
                c.imageExtent      = {s, s, 1};
                copies.push_back(c);
            }
        }
        vkCmdCopyBufferToImage(cmd, staging.buffer, img.image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               static_cast<uint32_t>(copies.size()), copies.data());

        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
                             0,nullptr, 0,nullptr, 1,&b);
    }
    ctx.endSingleTimeCommands(cmd);
    ctx.destroyBuffer(staging);
}

// ── allocCubemap ──────────────────────────────────────────────────────────────
static AllocatedImage allocCubemap(const Context& ctx, uint32_t size, uint32_t mips,
                                    VkFormat fmt = VK_FORMAT_R16G16B16A16_SFLOAT) {
    auto img = ctx.createImage(size, size, mips, VK_SAMPLE_COUNT_1_BIT, fmt,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 6);
    img.mipLevels = mips;

    // Create cubemap view manually
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image    = img.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    vi.format   = fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 6};
    VK_CHECK(vkCreateImageView(ctx.device(), &vi, nullptr, &img.view));
    return img;
}

// ── loadFromEquirectangular ───────────────────────────────────────────────────
void IBLProbe::loadFromEquirectangular(std::shared_ptr<Context> ctx,
                                        const std::filesystem::path& path) {
    m_ctx = ctx;
    m_ready = false;
    createSamplers();

    // ── 1. Load HDR equirectangular ──────────────────────────────────────────
    int W, H, ch;
    float* hdrData = stbi_loadf(path.string().c_str(), &W, &H, &ch, 4);
    if (!hdrData)
        throw std::runtime_error("[IBL] Failed to load HDR: " + path.string());

    // ── 2. Build environment cubemap (512) from equirect ─────────────────────
    constexpr uint32_t ENV_SIZE = 512;
    m_envCubemap = allocCubemap(*ctx, ENV_SIZE, 1);
    {
        std::vector<std::vector<glm::vec4>> faces(6, std::vector<glm::vec4>(ENV_SIZE * ENV_SIZE));
        for (int f = 0; f < 6; ++f) {
            for (uint32_t y = 0; y < ENV_SIZE; ++y) {
                for (uint32_t x = 0; x < ENV_SIZE; ++x) {
                    float u = (x + 0.5f) / ENV_SIZE * 2.f - 1.f;
                    float v = (y + 0.5f) / ENV_SIZE * 2.f - 1.f;
                    glm::vec3 dir = faceDir(f, u, v);
                    glm::vec3 col = sampleEquirect(hdrData, W, H, dir);
                    faces[f][y * ENV_SIZE + x] = {col, 1.f};
                }
            }
        }
        uploadCubemap(*ctx, m_envCubemap, ENV_SIZE, 1, faces);
    }
    stbi_image_free(hdrData);

    // ── 3. Irradiance map (32) — diffuse convolution ─────────────────────────
    constexpr uint32_t IRR = IRRADIANCE_SIZE;
    m_irradiance = allocCubemap(*ctx, IRR, 1);
    {
        std::vector<std::vector<glm::vec4>> faces(6, std::vector<glm::vec4>(IRR * IRR));
        constexpr uint32_t SAMPLES_PHI   = 64;
        constexpr uint32_t SAMPLES_THETA = 32;
        for (int f = 0; f < 6; ++f) {
            for (uint32_t y = 0; y < IRR; ++y) {
                for (uint32_t x = 0; x < IRR; ++x) {
                    float u = (x + 0.5f) / IRR * 2.f - 1.f;
                    float v = (y + 0.5f) / IRR * 2.f - 1.f;
                    glm::vec3 N = faceDir(f, u, v);
                    // Build TBN around N
                    glm::vec3 up  = std::abs(N.y) < 0.999f ? glm::vec3(0,1,0) : glm::vec3(1,0,0);
                    glm::vec3 R   = glm::normalize(glm::cross(up, N));
                    glm::vec3 U   = glm::cross(N, R);

                    glm::vec3 irr{0};
                    uint32_t count = 0;
                    for (uint32_t pi = 0; pi < SAMPLES_PHI; ++pi) {
                        float phi = 2.f * PI * pi / SAMPLES_PHI;
                        for (uint32_t ti = 0; ti < SAMPLES_THETA; ++ti) {
                            float theta = PI * 0.5f * ti / SAMPLES_THETA;
                            float sinT  = std::sin(theta), cosT = std::cos(theta);
                            float sinP  = std::sin(phi),   cosP = std::cos(phi);
                            glm::vec3 sampleDir = sinT * cosP * R + sinT * sinP * U + cosT * N;
                            // sample env cubemap cpu-side by reverse-equirect (we still have hdrData)
                            // instead we'll resample from faces we already built:
                            // pick best face from envCubemap faces — for now sample equirect again
                            // (we freed hdrData, so we use a lower-quality approach:
                            //  actual irradiance from env faces requires a CPU cubemap sampler.
                            //  Simple fix: we pass equirect ptr earlier — but it's freed.
                            //  Instead: iterate env faces to find nearest sample.)
                            // Practical solution: keep a CPU copy of ENV faces
                            irr += glm::vec3(0); // placeholder — filled below
                            ++count;
                        }
                    }
                    faces[f][y * IRR + x] = {irr, 1.f};
                }
            }
        }
        // --- Better approach: keep CPU env faces and sample from those ---
        // Reload equirect in-place for irradiance pass
        hdrData = stbi_loadf(path.string().c_str(), &W, &H, &ch, 4);
        for (int f = 0; f < 6; ++f) {
            for (uint32_t y = 0; y < IRR; ++y) {
                for (uint32_t x = 0; x < IRR; ++x) {
                    float u = (x + 0.5f) / IRR * 2.f - 1.f;
                    float v = (y + 0.5f) / IRR * 2.f - 1.f;
                    glm::vec3 N = faceDir(f, u, v);
                    glm::vec3 up  = std::abs(N.y) < 0.999f ? glm::vec3(0,1,0) : glm::vec3(1,0,0);
                    glm::vec3 R   = glm::normalize(glm::cross(up, N));
                    glm::vec3 U   = glm::cross(N, R);
                    glm::vec3 irr{0}; uint32_t cnt = 0;
                    for (uint32_t pi = 0; pi < SAMPLES_PHI; ++pi) {
                        float phi  = 2.f * PI * pi / SAMPLES_PHI;
                        for (uint32_t ti = 0; ti < SAMPLES_THETA; ++ti) {
                            float theta = PI * 0.5f * ti / SAMPLES_THETA;
                            float sinT  = std::sin(theta), cosT = std::cos(theta);
                            glm::vec3 sd = sinT*std::cos(phi)*R + sinT*std::sin(phi)*U + cosT*N;
                            irr += sampleEquirect(hdrData, W, H, sd) * cosT * sinT;
                            ++cnt;
                        }
                    }
                    irr = (PI / cnt) * irr;
                    faces[f][y * IRR + x] = {irr, 1.f};
                }
            }
        }
        stbi_image_free(hdrData);
        uploadCubemap(*ctx, m_irradiance, IRR, 1, faces);
    }

    // ── 4. Prefiltered specular map (128, 5 mips) ────────────────────────────
    hdrData = stbi_loadf(path.string().c_str(), &W, &H, &ch, 4);
    constexpr uint32_t PF   = PREFILTERED_SIZE;
    constexpr uint32_t MIPS = PREFILTERED_MIPS;
    m_prefiltered = allocCubemap(*ctx, PF, MIPS);
    {
        std::vector<std::vector<glm::vec4>> allFaces; // 6 faces × MIPS
        for (uint32_t m = 0; m < MIPS; ++m) {
            float roughness = static_cast<float>(m) / (MIPS - 1);
            float a         = roughness * roughness;
            uint32_t sz     = std::max(1u, PF >> m);
            constexpr uint32_t SAMP = 256;
            for (int f = 0; f < 6; ++f) {
                std::vector<glm::vec4> face(sz * sz);
                for (uint32_t y = 0; y < sz; ++y) {
                    for (uint32_t x = 0; x < sz; ++x) {
                        float u = (x + 0.5f) / sz * 2.f - 1.f;
                        float v = (y + 0.5f) / sz * 2.f - 1.f;
                        glm::vec3 N = glm::normalize(faceDir(f, u, v));
                        glm::vec3 R = N, V = N;
                        glm::vec3 col{0}; float totalW = 0.f;
                        // GGX importance sampling (Van der Corput + Hammersley)
                        for (uint32_t i = 0; i < SAMP; ++i) {
                            // Hammersley sequence
                            float r1 = static_cast<float>(i) / SAMP;
                            uint32_t bits = i;
                            bits = (bits << 16u) | (bits >> 16u);
                            bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
                            bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
                            bits = ((bits & 0x0f0f0f0fu) << 4u) | ((bits & 0xf0f0f0f0u) >> 4u);
                            bits = ((bits & 0x00ff00ffu) << 8u) | ((bits & 0xff00ff00u) >> 8u);
                            float r2 = static_cast<float>(bits) * 2.3283064365386963e-10f;

                            float phi   = 2.f * PI * r1;
                            float a2    = a * a;
                            float cosT  = std::sqrt((1.f - r2) / std::max(1e-7f, 1.f + (a2 - 1.f) * r2));
                            float sinT  = std::sqrt(1.f - cosT * cosT);
                            // Tangent space half-vector
                            glm::vec3 up  = std::abs(N.y) < 0.999f ? glm::vec3(0,1,0) : glm::vec3(1,0,0);
                            glm::vec3 Tx   = glm::normalize(glm::cross(up, N));
                            glm::vec3 Ty   = glm::cross(N, Tx);
                            glm::vec3 Hvec = glm::normalize(sinT*std::cos(phi)*Tx + sinT*std::sin(phi)*Ty + cosT*N);
                            glm::vec3 L    = glm::normalize(2.f * glm::dot(V, Hvec) * Hvec - V);
                            float NdotL    = glm::max(glm::dot(N, L), 0.f);
                            if (NdotL > 0.f) {
                                col    += sampleEquirect(hdrData, W, H, L) * NdotL;
                                totalW += NdotL;
                            }
                        }
                        col = totalW > 0.f ? col / totalW : glm::vec3(0.f);
                        face[y * sz + x] = {col, 1.f};
                    }
                }
                allFaces.push_back(std::move(face));
            }
        }
        uploadCubemap(*ctx, m_prefiltered, PF, MIPS, allFaces);
    }
    stbi_image_free(hdrData);

    // ── 5. BRDF integration LUT (512x512 RG16F) ──────────────────────────────
    constexpr uint32_t LUT = BRDF_LUT_SIZE;
    m_brdfLUT = ctx->createImage(LUT, LUT, 1, VK_SAMPLE_COUNT_1_BIT,
        VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_brdfLUT.mipLevels = 1;
    ctx->createImageView(m_brdfLUT, VK_IMAGE_ASPECT_COLOR_BIT);
    {
        constexpr uint32_t SAMP = 1024;
        std::vector<glm::u16vec2> lutData(LUT * LUT);
        auto floatToHalf = [](float f) -> uint16_t {
            // Simple float→half (no denormals, no inf)
            uint32_t x; std::memcpy(&x, &f, 4);
            return static_cast<uint16_t>(((x>>16)&0x8000)|
                (((x&0x7f800000)-0x38000000)>>13)|(x>>13&0x3ff));
        };
        for (uint32_t j = 0; j < LUT; ++j) {
            float NdotV = (j + 0.5f) / LUT;
            for (uint32_t i = 0; i < LUT; ++i) {
                float roughness = (i + 0.5f) / LUT;
                float a2 = roughness * roughness * roughness * roughness;
                glm::vec3 V{std::sqrt(1.f - NdotV*NdotV), 0.f, NdotV};
                float A = 0.f, B = 0.f;
                for (uint32_t s = 0; s < SAMP; ++s) {
                    float r1 = static_cast<float>(s) / SAMP;
                    uint32_t bits = s;
                    bits = (bits<<16)|(bits>>16);
                    bits = ((bits&0x55555555)<<1)|((bits&0xAAAAAAAA)>>1);
                    bits = ((bits&0x33333333)<<2)|((bits&0xCCCCCCCC)>>2);
                    bits = ((bits&0x0f0f0f0f)<<4)|((bits&0xf0f0f0f0)>>4);
                    bits = ((bits&0x00ff00ff)<<8)|((bits&0xff00ff00)>>8);
                    float r2   = static_cast<float>(bits)*2.3283064365386963e-10f;
                    float phi  = 2.f*PI*r1;
                    float cosT = std::sqrt((1.f-r2)/std::max(1e-7f,1.f+(a2-1.f)*r2));
                    float sinT = std::sqrt(1.f-cosT*cosT);
                    glm::vec3 H{sinT*std::cos(phi), sinT*std::sin(phi), cosT};
                    glm::vec3 L = glm::normalize(2.f*glm::dot(V,H)*H-V);
                    float NdotL = glm::max(L.z, 0.f);
                    float NdotH = glm::max(H.z, 0.f);
                    float VdotH = glm::max(glm::dot(V,H), 0.f);
                    if (NdotL > 0.f) {
                        float rk = roughness+1.f; float k=(rk*rk)/8.f;
                        float G  = (NdotV/(NdotV*(1.f-k)+k)) * (NdotL/(NdotL*(1.f-k)+k));
                        float Gv = G*VdotH/(NdotH*NdotV+1e-7f);
                        float Fc = std::pow(1.f-VdotH, 5.f);
                        A += Gv*(1.f-Fc);
                        B += Gv*Fc;
                    }
                }
                lutData[j*LUT+i] = { floatToHalf(A/SAMP), floatToHalf(B/SAMP) };
            }
        }
        AllocatedBuffer staging = ctx->createBuffer(LUT*LUT*4,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, true);
        std::memcpy(staging.mapped, lutData.data(), LUT*LUT*4);

        VkCommandBuffer cmd = ctx->beginSingleTimeCommands();
        {
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = m_brdfLUT.image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&b);
            VkBufferImageCopy c{}; c.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
            c.imageExtent={LUT,LUT,1};
            vkCmdCopyBufferToImage(cmd,staging.buffer,m_brdfLUT.image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&c);
            b.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            b.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
            b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,0,0,nullptr,0,nullptr,1,&b);
        }
        ctx->endSingleTimeCommands(cmd);
        ctx->destroyBuffer(staging);
    }

    m_ready = true;
    std::cout << "[IBL] Probe loaded from " << path << "\n";
}

// ── loadPrecomputed ───────────────────────────────────────────────────────────
void IBLProbe::loadPrecomputed(std::shared_ptr<Context> ctx,
                                const std::filesystem::path& irradiancePath,
                                const std::filesystem::path& prefilteredPath,
                                const std::filesystem::path& brdfLUTPath)
{
    // For pre-baked KTX2 cubemaps — load via Texture::fromKtx2 and expose views.
    // This path is much faster at startup than CPU baking.
    // Full implementation would use libktx cubemap loading.
    // For now, fall back to equirectangular bake with a warning.
    std::cerr << "[IBL] loadPrecomputed: KTX2 cubemap path not yet implemented. "
              << "Use loadFromEquirectangular() instead.\n";
    (void)ctx; (void)irradiancePath; (void)prefilteredPath; (void)brdfLUTPath;
}

} // namespace vkgfx
