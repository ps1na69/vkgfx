// ibl.cpp — Image-Based Lighting probe baking
//
// Generates three GPU textures from a single equirectangular HDR image:
//
//   m_irradiance   — 32x32 per-face diffuse irradiance cubemap
//   m_prefiltered  — 128x128 specular radiance cubemap, PREFILTERED_MIPS mips
//   m_brdfLUT      — 512x512 RG16F BRDF integration LUT
//
// BRDF LUT axis convention (must match lighting.frag):
//   u = NdotV    (column / x-axis)
//   v = roughness (row    / y-axis)
//   Shader samples as: texture(brdfLUT, vec2(NdotV, roughness))
//
// The HDR source is loaded once and reused for all three baking passes.
// Face convolution is parallelised over std::thread workers.

#include "vkgfx/ibl.h"
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <thread>
#include <atomic>
#include <vector>
#include <stb_image.h>

namespace vkgfx {

static constexpr float PI = 3.14159265359f;

// ─── Low-discrepancy sampling ─────────────────────────────────────────────────

static float radicalInverse(uint32_t bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u)  | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u)  | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u)  | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u)  | ((bits & 0xFF00FF00u) >> 8u);
    return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

static glm::vec2 hammersley(uint32_t i, uint32_t N) {
    return { static_cast<float>(i) / static_cast<float>(N), radicalInverse(i) };
}

// ─── Geometry helpers ─────────────────────────────────────────────────────────

static glm::vec2 dirToEquirect(const glm::vec3& d) {
    return { (std::atan2(d.z, d.x) + PI) / (2.f * PI),
             (std::asin(glm::clamp(d.y, -1.f, 1.f)) + PI * 0.5f) / PI };
}

static glm::vec3 sampleEquirect(const float* data, int W, int H,
                                  const glm::vec3& dir)
{
    glm::vec2 uv = dirToEquirect(glm::normalize(dir));
    float uf = uv.x * (W - 1);
    float vf = (1.f - uv.y) * (H - 1);
    int x0 = std::max(0, std::min(static_cast<int>(uf), W-1));
    int y0 = std::max(0, std::min(static_cast<int>(vf), H-1));
    int x1 = std::min(x0+1, W-1), y1 = std::min(y0+1, H-1);
    float fx = uf - x0, fy = vf - y0;
    auto fetch = [&](int x, int y) {
        const float* p = data + (y * W + x) * 4;
        return glm::vec3(p[0], p[1], p[2]);
    };
    return glm::mix(glm::mix(fetch(x0,y0), fetch(x1,y0), fx),
                    glm::mix(fetch(x0,y1), fetch(x1,y1), fx), fy);
}

// Cubemap face direction: face 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z
static glm::vec3 faceDir(int face, float u, float v) {
    switch (face) {
        case 0: return glm::normalize(glm::vec3( 1,  v, -u));
        case 1: return glm::normalize(glm::vec3(-1,  v,  u));
        case 2: return glm::normalize(glm::vec3( u,  1, -v));
        case 3: return glm::normalize(glm::vec3( u, -1,  v));
        case 4: return glm::normalize(glm::vec3( u,  v,  1));
        default:return glm::normalize(glm::vec3(-u,  v, -1));
    }
}

static void buildTangentFrame(const glm::vec3& N, glm::vec3& T, glm::vec3& B) {
    glm::vec3 up = (std::abs(N.y) < 0.999f) ? glm::vec3(0,1,0) : glm::vec3(1,0,0);
    T = glm::normalize(glm::cross(up, N));
    B = glm::cross(N, T);
}

// GGX importance-sampled half-vector (tangent space).
static glm::vec3 ggxHalfVector(float r1, float r2, float roughness) {
    float a2   = roughness * roughness * roughness * roughness;
    float cosT = std::sqrt((1.f - r2) / std::max(1e-7f, 1.f + (a2-1.f)*r2));
    float sinT = std::sqrt(1.f - cosT*cosT);
    float phi  = 2.f * PI * r1;
    return { sinT * std::cos(phi), sinT * std::sin(phi), cosT };
}

// ─── GPU helpers ──────────────────────────────────────────────────────────────

// Allocate cubemap image + VK_IMAGE_VIEW_TYPE_CUBE view.
static AllocatedImage allocCubemap(const Context& ctx,
                                    uint32_t size, uint32_t mips,
                                    VkFormat fmt = VK_FORMAT_R16G16B16A16_SFLOAT)
{
    auto img = ctx.createImage(size, size, mips, VK_SAMPLE_COUNT_1_BIT, fmt,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 6);
    img.mipLevels = mips;
    VkImageViewCreateInfo vi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image            = img.image;
    vi.viewType         = VK_IMAGE_VIEW_TYPE_CUBE;
    vi.format           = fmt;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 6 };
    VK_CHECK(vkCreateImageView(ctx.device(), &vi, nullptr, &img.view),
             "allocCubemap view");
    return img;
}

// Upload facePixels[mip*6+face] into a pre-allocated cubemap image.
static void uploadCubemap(const Context& ctx, AllocatedImage& img,
                           uint32_t baseSize, uint32_t mips,
                           const std::vector<std::vector<glm::vec4>>& facePixels)
{
    VkDeviceSize total = 0;
    std::vector<VkDeviceSize> offsets;
    offsets.reserve(mips * 6);
    for (uint32_t m = 0; m < mips; ++m) {
        uint32_t s = std::max(1u, baseSize >> m);
        for (uint32_t f = 0; f < 6; ++f) {
            offsets.push_back(total);
            total += static_cast<VkDeviceSize>(s) * s * sizeof(glm::vec4);
        }
    }

    AllocatedBuffer staging = ctx.createBuffer(total,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, true);
    auto* dst = static_cast<uint8_t*>(staging.mapped);
    for (size_t i = 0; i < facePixels.size(); ++i)
        std::memcpy(dst + offsets[i], facePixels[i].data(),
                    facePixels[i].size() * sizeof(glm::vec4));

    VkCommandBuffer cmd = ctx.beginSingleTimeCommands();
    {
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        b.image=img.image; b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,mips,0,6};
        b.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0,0,nullptr,0,nullptr,1,&b);

        std::vector<VkBufferImageCopy> copies;
        copies.reserve(mips * 6);
        for (uint32_t m = 0; m < mips; ++m) {
            uint32_t s = std::max(1u, baseSize >> m);
            for (uint32_t f = 0; f < 6; ++f) {
                VkBufferImageCopy c{};
                c.bufferOffset     = offsets[m*6+f];
                c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, f, 1};
                c.imageExtent      = {s, s, 1};
                copies.push_back(c);
            }
        }
        vkCmdCopyBufferToImage(cmd, staging.buffer, img.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            static_cast<uint32_t>(copies.size()), copies.data());

        b.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,0,nullptr,0,nullptr,1,&b);
    }
    ctx.endSingleTimeCommands(cmd);
    ctx.destroyBuffer(staging);
}

// ─── IBLProbe::createSamplers ─────────────────────────────────────────────────

void IBLProbe::createSamplers() {
    auto dev = m_ctx->device();
    {
        VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        si.magFilter=VK_FILTER_LINEAR; si.minFilter=VK_FILTER_LINEAR;
        si.mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU=si.addressModeV=si.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.minLod=0.f; si.maxLod=static_cast<float>(PREFILTERED_MIPS-1);
        si.anisotropyEnable=VK_TRUE; si.maxAnisotropy=8.f;
        VK_CHECK(vkCreateSampler(dev, &si, nullptr, &m_cubeSampler), "IBL cube sampler");
    }
    {
        VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        si.magFilter=VK_FILTER_LINEAR; si.minFilter=VK_FILTER_LINEAR;
        si.mipmapMode=VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU=si.addressModeV=si.addressModeW=VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.minLod=0.f; si.maxLod=0.f;
        VK_CHECK(vkCreateSampler(dev, &si, nullptr, &m_brdfSampler), "IBL brdf sampler");
    }
}

// ─── IBLProbe::destroy ────────────────────────────────────────────────────────

void IBLProbe::destroy() {
    if (!m_ctx) return;
    auto dev = m_ctx->device();
    m_ctx->destroyImage(m_envCubemap);
    m_ctx->destroyImage(m_irradiance);
    m_ctx->destroyImage(m_prefiltered);
    m_ctx->destroyImage(m_brdfLUT);
    if (m_cubeSampler) { vkDestroySampler(dev, m_cubeSampler, nullptr); m_cubeSampler=VK_NULL_HANDLE; }
    if (m_brdfSampler) { vkDestroySampler(dev, m_brdfSampler, nullptr); m_brdfSampler=VK_NULL_HANDLE; }
    m_ctx.reset();
    m_ready = false;
}

// ─── IBLProbe::loadFromEquirectangular ───────────────────────────────────────

void IBLProbe::loadFromEquirectangular(std::shared_ptr<Context> ctx,
                                        const std::filesystem::path& path)
{
    m_ctx   = ctx;
    m_ready = false;
    createSamplers();

    // ── Load HDR once — reused for all three baking passes ──────────────────
    int imgW, imgH, ch;
    float* hdr = stbi_loadf(path.string().c_str(), &imgW, &imgH, &ch, 4);
    if (!hdr) throw std::runtime_error("[IBL] Failed to load HDR: " + path.string());

    // ── 1. Environment cubemap (512 × 512, 1 mip) ───────────────────────────
    constexpr uint32_t ENV = 512;
    m_envCubemap = allocCubemap(*ctx, ENV, 1);
    {
        std::vector<std::vector<glm::vec4>> faces(6, std::vector<glm::vec4>(ENV*ENV));
        std::vector<std::thread> workers;
        for (int f = 0; f < 6; ++f) {
            workers.emplace_back([&, f]() {
                for (uint32_t y = 0; y < ENV; ++y)
                    for (uint32_t x = 0; x < ENV; ++x) {
                        float u = (x+0.5f)/ENV*2.f-1.f, v = (y+0.5f)/ENV*2.f-1.f;
                        glm::vec3 c = sampleEquirect(hdr, imgW, imgH, faceDir(f, u, v));
                        faces[f][y*ENV+x] = {c, 1.f};
                    }
            });
        }
        for (auto& t : workers) t.join();
        uploadCubemap(*ctx, m_envCubemap, ENV, 1, faces);
    }

    // ── 2. Irradiance map (32 × 32, 1 mip) ─────────────────────────────────
    // Diffuse Lambertian convolution.
    constexpr uint32_t IRR  = IRRADIANCE_SIZE;
    constexpr uint32_t SPHI = 128, STHETA = 64;
    m_irradiance = allocCubemap(*ctx, IRR, 1);
    {
        std::vector<std::vector<glm::vec4>> faces(6, std::vector<glm::vec4>(IRR*IRR));
        std::vector<std::thread> workers;
        for (int f = 0; f < 6; ++f) {
            workers.emplace_back([&, f]() {
                for (uint32_t y = 0; y < IRR; ++y)
                    for (uint32_t x = 0; x < IRR; ++x) {
                        float u = (x+0.5f)/IRR*2.f-1.f, v = (y+0.5f)/IRR*2.f-1.f;
                        glm::vec3 N = faceDir(f, u, v);
                        glm::vec3 T, B; buildTangentFrame(N, T, B);
                        glm::vec3 irr{0.f}; float weight=0.f;
                        for (uint32_t pi = 0; pi < SPHI; ++pi) {
                            float phi = 2.f*PI*pi/SPHI;
                            for (uint32_t ti = 0; ti < STHETA; ++ti) {
                                float theta = PI*0.5f*(ti+0.5f)/STHETA;
                                float sinT=std::sin(theta), cosT=std::cos(theta);
                                glm::vec3 sd = sinT*std::cos(phi)*T + sinT*std::sin(phi)*B + cosT*N;
                                float w = cosT * sinT;
                                irr    += sampleEquirect(hdr, imgW, imgH, sd) * w;
                                weight += w;
                            }
                        }
                        irr = (weight>0.f) ? PI*irr/weight : glm::vec3(0.f);
                        faces[f][y*IRR+x] = {irr, 1.f};
                    }
            });
        }
        for (auto& t : workers) t.join();
        uploadCubemap(*ctx, m_irradiance, IRR, 1, faces);
    }

    // ── 3. Prefiltered specular map (128 × 128, 5 mips) ─────────────────────
    // GGX importance sampling, split-sum approximation (R=V=N).
    constexpr uint32_t PF   = PREFILTERED_SIZE;
    constexpr uint32_t PMIP = PREFILTERED_MIPS;
    constexpr uint32_t PSMP = 512;
    m_prefiltered = allocCubemap(*ctx, PF, PMIP);
    {
        std::vector<std::vector<glm::vec4>> fp(PMIP*6);
        for (uint32_t m = 0; m < PMIP; ++m) {
            uint32_t sz = std::max(1u, PF>>m);
            for (uint32_t f = 0; f < 6; ++f) fp[m*6+f].resize(sz*sz);
        }
        std::vector<std::thread> workers;
        workers.reserve(PMIP*6);
        for (uint32_t m = 0; m < PMIP; ++m) {
            float roughness = (PMIP>1) ? static_cast<float>(m)/(PMIP-1) : 0.f;
            uint32_t sz = std::max(1u, PF>>m);
            for (uint32_t f = 0; f < 6; ++f) {
                workers.emplace_back([&, m, f, roughness, sz]() {
                    auto& out = fp[m*6+f];
                    for (uint32_t y = 0; y < sz; ++y)
                        for (uint32_t x = 0; x < sz; ++x) {
                            float u = (x+0.5f)/sz*2.f-1.f, v = (y+0.5f)/sz*2.f-1.f;
                            glm::vec3 N = glm::normalize(faceDir(f, u, v));
                            glm::vec3 T, B; buildTangentFrame(N, T, B);
                            glm::vec3 col{0.f}; float totalW=0.f;
                            for (uint32_t s = 0; s < PSMP; ++s) {
                                glm::vec2 Xi = hammersley(s, PSMP);
                                glm::vec3 Ht = ggxHalfVector(Xi.x, Xi.y, roughness);
                                glm::vec3 H  = glm::normalize(Ht.x*T + Ht.y*B + Ht.z*N);
                                glm::vec3 L  = glm::normalize(2.f*glm::dot(N,H)*H - N);
                                float NdotL  = glm::max(glm::dot(N,L), 0.f);
                                if (NdotL > 0.f) {
                                    col    += sampleEquirect(hdr, imgW, imgH, L) * NdotL;
                                    totalW += NdotL;
                                }
                            }
                            out[y*sz+x] = {(totalW>0.f ? col/totalW : glm::vec3(0.f)), 1.f};
                        }
                });
            }
        }
        for (auto& t : workers) t.join();
        uploadCubemap(*ctx, m_prefiltered, PF, PMIP, fp);
    }

    stbi_image_free(hdr);

    // ── 4. BRDF integration LUT (512 × 512, RG16F) ──────────────────────────
    // u-axis (column) = NdotV,    matches shader vec2(NdotV, roughness).x
    // v-axis (row)    = roughness, matches shader vec2(NdotV, roughness).y
    // So outer loop = roughness (row), inner loop = NdotV (column).
    constexpr uint32_t LSIZ = BRDF_LUT_SIZE;
    constexpr uint32_t LSMP = 1024;

    m_brdfLUT = ctx->createImage(LSIZ, LSIZ, 1, VK_SAMPLE_COUNT_1_BIT,
        VK_FORMAT_R16G16_SFLOAT, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    m_brdfLUT.mipLevels = 1;
    ctx->createImageView(m_brdfLUT, VK_IMAGE_ASPECT_COLOR_BIT);
    {
        auto f32ToF16 = [](float f) -> uint16_t {
            f = std::max(0.f, std::min(f, 1.f));
            if (f == 0.f) return 0;
            uint32_t x; std::memcpy(&x, &f, 4);
            return static_cast<uint16_t>(
                ((x>>16)&0x8000u) |
                (((x&0x7F800000u)-0x38000000u)>>13) |
                ((x>>13)&0x03FFu));
        };

        std::vector<uint32_t> lutData(LSIZ * LSIZ);

        // row = roughness (v), col = NdotV (u)
        auto bakeRow = [&](uint32_t row) {
            float roughness = (row + 0.5f) / LSIZ;
            float a2 = roughness*roughness*roughness*roughness;
            for (uint32_t col = 0; col < LSIZ; ++col) {
                float NdotV = (col + 0.5f) / LSIZ;
                glm::vec3 V{std::sqrt(1.f-NdotV*NdotV), 0.f, NdotV};
                float A=0.f, Bv=0.f;
                for (uint32_t s = 0; s < LSMP; ++s) {
                    glm::vec2 Xi = hammersley(s, LSMP);
                    glm::vec3 Ht = ggxHalfVector(Xi.x, Xi.y, roughness);
                    glm::vec3 L  = glm::normalize(2.f*glm::dot(V,Ht)*Ht - V);
                    float NdotL  = glm::max(L.z, 0.f);
                    float NdotH  = glm::max(Ht.z, 0.f);
                    float VdotH  = glm::max(glm::dot(V,Ht), 0.f);
                    if (NdotL > 0.f) {
                        // Smith G with IBL k = a²/2
                        float k  = a2 * 0.5f;
                        float G  = (NdotV/(NdotV*(1.f-k)+k))
                                 * (NdotL/(NdotL*(1.f-k)+k));
                        float Gv = G*VdotH / std::max(NdotH*NdotV, 1e-7f);
                        float Fc = std::pow(1.f-VdotH, 5.f);
                        A  += Gv*(1.f-Fc);
                        Bv += Gv*Fc;
                    }
                }
                uint16_t r16 = f32ToF16(A  / LSMP);
                uint16_t g16 = f32ToF16(Bv / LSMP);
                lutData[row*LSIZ+col] = static_cast<uint32_t>(r16)
                                      | (static_cast<uint32_t>(g16) << 16);
            }
        };

        uint32_t hw = std::max(1u, std::thread::hardware_concurrency());
        std::atomic<uint32_t> nextRow{0};
        std::vector<std::thread> workers(hw);
        for (auto& t : workers) t = std::thread([&](){
            for (;;) {
                uint32_t row = nextRow.fetch_add(1, std::memory_order_relaxed);
                if (row >= LSIZ) break;
                bakeRow(row);
            }
        });
        for (auto& t : workers) t.join();

        VkDeviceSize bytes = LSIZ * LSIZ * sizeof(uint32_t);
        AllocatedBuffer staging = ctx->createBuffer(bytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, true);
        std::memcpy(staging.mapped, lutData.data(), bytes);

        VkCommandBuffer cmd = ctx->beginSingleTimeCommands();
        {
            VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            b.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.srcQueueFamilyIndex=b.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
            b.image=m_brdfLUT.image; b.subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
            b.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,nullptr,0,nullptr,1,&b);
            VkBufferImageCopy c{};
            c.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
            c.imageExtent={LSIZ,LSIZ,1};
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

void IBLProbe::loadPrecomputed(std::shared_ptr<Context>,
                                const std::filesystem::path&,
                                const std::filesystem::path&,
                                const std::filesystem::path&)
{
    std::cerr << "[IBL] loadPrecomputed: KTX2 path not yet implemented.\n";
}

} // namespace vkgfx
