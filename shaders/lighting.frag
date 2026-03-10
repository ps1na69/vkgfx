#version 450
// lighting.frag
// Deferred lighting pass – reads the G-buffer and computes physically based
// lighting for every on-screen pixel.
//
// PBR Model: Cook-Torrance specular BRDF + Lambertian diffuse.
//
//   Lo(p, ωo) = ∫ (kd * albedo/π + ks * DFG/(4·(n·ωo)·(n·ωi))) * Li(p,ωi) * (n·ωi) dωi
//
// where:
//   D = GGX Normal Distribution Function  (microfacet distribution)
//   F = Fresnel-Schlick approximation      (surface reflectance)
//   G = Smith-Schlick-GGX geometry term   (microfacet self-shadowing)
//
// IMPORTANT: All colour values are in linear space throughout this shader.
// The tone-map pass converts to display gamma.
//
// Light types supported: point, directional, spot.
// Attenuation: quadratic (physically correct inverse-square law) for point/spot.

// ─── Full-screen quad inputs ─────────────────────────────────────────────────
layout(location = 0) in  vec2 inUV;
layout(location = 0) out vec4 outHDR;     // Linear HDR colour written to hdrImage

// ─── G-buffer samplers (set 0) ───────────────────────────────────────────────
layout(set = 0, binding = 0) uniform sampler2D gWorldPos;
layout(set = 0, binding = 1) uniform sampler2D gNormal;
layout(set = 0, binding = 2) uniform sampler2D gAlbedo;
layout(set = 0, binding = 3) uniform sampler2D gMaterial;
layout(set = 0, binding = 4) uniform sampler2D gEmissive;
layout(set = 0, binding = 5) uniform sampler2D gDepth;
layout(set = 0, binding = 6) uniform sampler2D aoTexture;   // SSAO result

// ─── Frame UBO (set 1, binding 0) ────────────────────────────────────────────
layout(set = 1, binding = 0) uniform FrameUBO {
    mat4  view;
    mat4  proj;
    mat4  viewProj;
    mat4  invView;
    mat4  invProj;
    vec4  cameraPos;
    float time;
} frame;

// ─── Light SSBO (set 2, binding 0) ───────────────────────────────────────────
// Dynamic array of lights stored in a Shader Storage Buffer Object.
// SSBO avoids the 64KB UBO size limit and allows the CPU to update arbitrary
// numbers of lights per frame via vkCmdUpdateBuffer or a mapped pointer.
struct GpuLight {
    vec4  position;      // xyz = world pos, w = range
    vec4  direction;     // xyz = direction, w = spot outer angle (cos)
    vec4  color;         // xyz = linear RGB, w = intensity
    uint  type;          // 0=point, 1=directional, 2=spot
    float innerAngleCos;
    float pad[2];
};
layout(set = 2, binding = 0) readonly buffer LightBuffer {
    uint     count;
    GpuLight lights[];
} lightBuf;

// ─── Constants ───────────────────────────────────────────────────────────────
const float PI     = 3.14159265359;
const float INV_PI = 0.31830988618;
// Minimum roughness to avoid division-by-zero in the GGX denominator.
const float MIN_ROUGHNESS = 0.04;

// ─── Cook-Torrance BRDF functions ────────────────────────────────────────────

// GGX Normal Distribution Function (D).
// Concentration of microfacets aligned with the half-vector h.
float D_GGX(float NdotH, float roughness)
{
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Schlick-GGX geometry shadowing sub-term (one side).
float G1_SchlickGGX(float NdotV, float k)
{
    return NdotV / (NdotV * (1.0 - k) + k);
}

// Smith joint geometry term G = G1(n·v) * G1(n·l).
// k remapping for direct lighting: k = (roughness+1)^2 / 8
float G_Smith(float NdotV, float NdotL, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) * 0.125;   // (r^2 / 8)
    return G1_SchlickGGX(NdotV, k) * G1_SchlickGGX(NdotL, k);
}

// Fresnel-Schlick approximation.
// F0 is the surface reflectance at normal incidence.
// For non-metals F0 = 0.04; for metals it equals the albedo.
vec3 F_Schlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Full Cook-Torrance specular term (already divided by the denominator).
vec3 specularBRDF(float D, vec3 F, float G, float NdotV, float NdotL)
{
    // Denominator clamped to avoid NaN at grazing angles.
    float denom = 4.0 * max(NdotV, 0.001) * max(NdotL, 0.001);
    return (D * F * G) / denom;
}

// ─── Light attenuation ────────────────────────────────────────────────────────

// Physically correct inverse-square law attenuation with a smooth cutoff at
// the light's effective range (stored in light.position.w).
float point_attenuation(float dist, float range)
{
    float d2    = dist * dist;
    float r2    = range * range;
    // Numerator: standard 1/d^2.  The min(1, ...) caps the value near the
    // light source.  The squared falloff window smoothly drops to 0 at range.
    float num   = clamp(1.0 - (d2 * d2) / (r2 * r2), 0.0, 1.0);
    return (num * num) / max(d2, 0.0001);
}

float spot_attenuation(vec3 L, vec3 spotDir, float cosOuter, float cosInner)
{
    float cosAngle = dot(-L, spotDir);
    // Smooth transition between inner and outer cone using smoothstep.
    return smoothstep(cosOuter, cosInner, cosAngle);
}

// ─── Per-light radiance contribution ─────────────────────────────────────────

vec3 evaluate_light(uint idx,
                    vec3 worldPos, vec3 N, vec3 V,
                    vec3 albedo, float metallic, float roughness,
                    vec3 F0)
{
    GpuLight light = lightBuf.lights[idx];
    vec3  lightColor = light.color.rgb * light.color.w;  // rgb * intensity

    vec3  L;
    float attenuation = 1.0;

    if (light.type == 1u) {
        // Directional light – infinite distance, no attenuation.
        L = normalize(-light.direction.xyz);
    } else {
        // Point / Spot: compute L from world position.
        vec3 delta = light.position.xyz - worldPos;
        float dist = length(delta);
        L = delta / dist;

        attenuation = point_attenuation(dist, light.position.w);

        if (light.type == 2u) {
            // Additional spot cone attenuation.
            attenuation *= spot_attenuation(L, normalize(light.direction.xyz),
                                            light.direction.w,   // outer cos
                                            light.innerAngleCos);
        }
    }

    // ── Compute BRDF terms ────────────────────────────────────────────────────
    vec3  H      = normalize(V + L);
    float NdotL  = max(dot(N, L),  0.0);
    float NdotV  = max(dot(N, V),  0.0);
    float NdotH  = max(dot(N, H),  0.0);
    float HdotV  = max(dot(H, V),  0.0);

    // Early-out for back-facing lights (avoids wasted BRDF evaluation).
    if (NdotL <= 0.0) return vec3(0.0);

    float D = D_GGX(NdotH, roughness);
    vec3  F = F_Schlick(HdotV, F0);
    float G = G_Smith(NdotV, NdotL, roughness);

    vec3 spec = specularBRDF(D, F, G, NdotV, NdotL);

    // Energy conservation: the specular term F already represents the
    // reflected fraction.  The diffuse fraction is (1 - F) * (1 - metallic).
    // Metals have no diffuse term (they absorb and re-emit as coloured specular).
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    vec3 diff = kD * albedo * INV_PI;

    // Combine and scale by the incoming radiance and angle term.
    vec3 radiance = lightColor * attenuation;
    return (diff + spec) * radiance * NdotL;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

void main()
{
    // ── Read G-buffer ─────────────────────────────────────────────────────────
    vec3  worldPos  = texture(gWorldPos,  inUV).rgb;
    vec3  N         = normalize(texture(gNormal,   inUV).rgb);
    vec4  albedoAO  = texture(gAlbedo,   inUV);
    vec4  material  = texture(gMaterial, inUV);
    vec3  emissive  = texture(gEmissive, inUV).rgb;

    vec3  albedo    = albedoAO.rgb;
    float bakedAO   = albedoAO.a;
    float metallic  = material.r;
    float roughness = max(material.g, MIN_ROUGHNESS);

    // ── SSAO ──────────────────────────────────────────────────────────────────
    // Multiply baked AO and dynamic SSAO together for maximum darkening.
    float ssao      = texture(aoTexture, inUV).r;
    float finalAO   = bakedAO * ssao;

    // ── Camera direction ──────────────────────────────────────────────────────
    vec3  V = normalize(frame.cameraPos.xyz - worldPos);

    // ── Fresnel base reflectance (F0) ─────────────────────────────────────────
    // Dialectric surfaces have a constant F0 ≈ 0.04 (matches common materials).
    // Metals take their F0 from the albedo colour (complex refractive index).
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // ── Accumulate all lights ─────────────────────────────────────────────────
    vec3 Lo = vec3(0.0);
    for (uint i = 0; i < lightBuf.count; ++i) {
        Lo += evaluate_light(i, worldPos, N, V, albedo, metallic, roughness, F0);
    }

    // ── Ambient term (IBL approximation) ─────────────────────────────────────
    // A simple constant ambient serves as a fallback when no IBL probe is
    // available.  Replace with a proper prefiltered-environment + BRDF LUT
    // when IBL is added.
    const vec3 AMBIENT_RADIANCE = vec3(0.03);
    vec3 ambient = AMBIENT_RADIANCE * albedo * finalAO;

    // ── Final HDR colour ──────────────────────────────────────────────────────
    vec3 color = ambient + Lo + emissive;

    // No gamma correction here – the tone-map pass handles display transform.
    outHDR = vec4(color, 1.0);
}
