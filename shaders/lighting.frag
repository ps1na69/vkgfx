#version 450
// shaders/lighting.frag  —  Deferred PBR lighting pass.
//
// G-buffer bindings (set=0):
//   0=albedo  1=normal  2=RMA  3=emissive  4=shadowCoord  5=depth  6=shadowMap
//
// Scene UBO (set=1 binding=0) and Light UBO (set=1 binding=1) MUST match
// types.h LightUBO and SceneUBO exactly (std140).
//
// IBL (set=2):  0=irradiance  1=prefiltered  2=brdfLut

layout(location = 0) in  vec2 inUV;
layout(location = 0) out vec4 outColor;

// ── G-buffer ──────────────────────────────────────────────────────────────────
layout(set = 0, binding = 0) uniform sampler2D   gAlbedo;
layout(set = 0, binding = 1) uniform sampler2D   gNormal;
layout(set = 0, binding = 2) uniform sampler2D   gRMA;
layout(set = 0, binding = 3) uniform sampler2D   gEmissive;
layout(set = 0, binding = 4) uniform sampler2D   gShadowCoord;
layout(set = 0, binding = 5) uniform sampler2D   gDepth;
layout(set = 0, binding = 6) uniform sampler2DShadow shadowMap;

// ── Scene UBO (set=1 binding=0) ───────────────────────────────────────────────
layout(set = 1, binding = 0) uniform SceneUBO {
    mat4  view;
    mat4  proj;
    mat4  viewProj;
    mat4  invViewProj;
    mat4  lightViewProj;
    vec4  cameraPos;
    vec4  viewport;
} scene;

// ── Light UBO (set=1 binding=1) ───────────────────────────────────────────────
// Layout MUST match types.h LightUBO exactly.
// Using only vec4/uvec4 — no float[] arrays (those get stride=16 in std140).
//
// offset   0: sunDirection  vec4
// offset  16: sunColor      vec4   (w=intensity)
// offset  32: sunFlags      uvec4  (x=enabled, y=shadowEnabled)
// offset  48: points[8]     PointLightGPU  (each 48 bytes: pos vec4, color vec4, params vec4)
// offset 432: miscFlags     uvec4  (x=pointCount, y=gbufferDebug)
// offset 448: iblParams     vec4   (x=iblIntensity)

struct PointLightGPU {
    vec4 position;   // xyz=pos
    vec4 color;      // xyz=color, w=intensity
    vec4 params;     // x=radius
};

layout(set = 1, binding = 1) uniform LightUBO {
    vec4           sunDirection;
    vec4           sunColor;          // w = intensity
    uvec4          sunFlags;          // x=enabled, y=shadowEnabled
    PointLightGPU  points[8];
    uvec4          miscFlags;         // x=pointCount, y=gbufferDebug
    vec4           iblParams;         // x=iblIntensity
} lights;

// ── IBL (set=2) ───────────────────────────────────────────────────────────────
layout(set = 2, binding = 0) uniform samplerCube irradianceCube;
layout(set = 2, binding = 1) uniform samplerCube prefilteredCube;
layout(set = 2, binding = 2) uniform sampler2D   brdfLut;

// ── PBR helpers ───────────────────────────────────────────────────────────────
const float PI = 3.14159265359;

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

float GeometrySchlickGGX(float NdotX, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotX / max(NdotX * (1.0 - k) + k, 1e-7);
}

float GeometrySmith(float NdotV, float NdotL, float roughness) {
    return GeometrySchlickGGX(NdotV, roughness)
         * GeometrySchlickGGX(NdotL, roughness);
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0)
              * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 evalBRDF(vec3 albedo, float roughness, float metallic,
              vec3 N, vec3 V, vec3 L, vec3 F0, vec3 radiance) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);

    vec3  H    = normalize(V + L);
    float NDF  = DistributionGGX(N, H, roughness);
    float G    = GeometrySmith(NdotV, NdotL, roughness);
    vec3  F    = FresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3  num  = NDF * G * F;
    float den  = 4.0 * NdotV * NdotL + 1e-4;
    vec3  spec = num / den;
    vec3  kD   = (vec3(1.0) - F) * (1.0 - metallic);
    return (kD * albedo / PI + spec) * radiance * NdotL;
}

// Reconstruct world pos using precomputed invViewProj (no per-pixel inverse())
vec3 worldPosFromDepth(float depth) {
    vec4 clip  = vec4(inUV * 2.0 - 1.0, depth, 1.0);
    vec4 world = scene.invViewProj * clip;
    return world.xyz / world.w;
}

// PCF shadow — 3×3 kernel, sampler2DShadow does hardware compare per tap.
// bias is slope-scaled by the caller to avoid acne on curved surfaces.
float shadowPCF(vec3 coord, float bias) {
    if (coord.z >= 1.0) return 1.0;
    float shadow = 0.0;
    vec2  texel  = 1.0 / textureSize(shadowMap, 0);
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y)
            shadow += texture(shadowMap,
                vec3(coord.xy + vec2(x, y) * texel, coord.z - bias));
    return shadow / 9.0;
}

// Windowed inverse-square attenuation (UE4 / Epic approach)
float pointAtten(vec3 frag, vec3 lpos, float radius) {
    float d2 = dot(frag - lpos, frag - lpos);
    float r2 = radius * radius;
    float w  = clamp(1.0 - (d2 * d2) / (r2 * r2), 0.0, 1.0);
    return (w * w) / max(d2, 0.0001);
}

void main() {
    // ── Read G-buffer ─────────────────────────────────────────────────────────
    vec3  albedo    = texture(gAlbedo,      inUV).rgb;
    vec3  Nenc      = texture(gNormal,      inUV).rgb;
    vec3  rma       = texture(gRMA,         inUV).rgb;
    vec3  emissive  = texture(gEmissive,    inUV).rgb;
    vec3  shadowXYZ = texture(gShadowCoord, inUV).xyz;
    float depth     = texture(gDepth,       inUV).r;

    float roughness = max(rma.r, 0.05);
    float metallic  = rma.g;
    float ao        = rma.b;
    vec3  N         = normalize(Nenc * 2.0 - 1.0);

    // ── Unpack misc flags ─────────────────────────────────────────────────────
    uint  pointCount   = lights.miscFlags.x;
    uint  debugView    = lights.miscFlags.y;
    float iblIntensity = lights.iblParams.x;
    uint  sunEnabled   = lights.sunFlags.x;
    uint  shadowOn     = lights.sunFlags.y;

    // ── Debug views ───────────────────────────────────────────────────────────
    if (debugView == 1u) { outColor = vec4(albedo,          1.0); return; }
    if (debugView == 2u) { outColor = vec4(N * .5 + .5,     1.0); return; }
    if (debugView == 3u) { outColor = vec4(vec3(roughness), 1.0); return; }
    if (debugView == 4u) { outColor = vec4(vec3(metallic),  1.0); return; }
    if (debugView == 5u) { outColor = vec4(vec3(depth),     1.0); return; }
    if (debugView == 6u) { outColor = vec4(vec3(ao),        1.0); return; }
    if (debugView == 7u) { outColor = vec4(emissive,        1.0); return; }

    // ── Sky background ────────────────────────────────────────────────────────
    // Reconstruct view direction from UV, sample the env cube at mip 0.
    if (depth >= 1.0) {
        vec4 clipPos   = vec4(inUV * 2.0 - 1.0, 1.0, 1.0);
        vec4 viewDirH  = scene.invViewProj * clipPos;
        vec3 skyDir    = normalize(viewDirH.xyz / viewDirH.w - scene.cameraPos.xyz);
        vec3 skyColor  = textureLod(prefilteredCube, skyDir, 0.0).rgb;
        outColor = vec4(skyColor, 1.0);
        return;
    }

    // ── PBR setup ─────────────────────────────────────────────────────────────
    vec3  worldPos = worldPosFromDepth(depth);
    vec3  V        = normalize(scene.cameraPos.xyz - worldPos);
    float NdotV    = max(dot(N, V), 0.0);
    vec3  F0       = mix(vec3(0.04), albedo, metallic);
    vec3  Lo       = vec3(0.0);

    // ── Shadow ────────────────────────────────────────────────────────────────
    float shadow = 1.0;
    if (shadowOn != 0u && sunEnabled != 0u) {
        // Slope-scaled bias: increases at grazing angles to prevent acne on
        // curved surfaces (sphere self-shadowing). The formula:
        //   bias = mix(maxBias, minBias, NdotL)
        // gives a large bias when N and L are nearly perpendicular (where
        // shadow acne is worst) and a small bias when they are aligned.
        vec3  sunL   = normalize(-lights.sunDirection.xyz);
        float NdotL  = max(dot(N, sunL), 0.0);
        float bias   = mix(0.02, 0.002, NdotL);  // 0.02 at grazing, 0.002 at direct
        shadow = shadowPCF(shadowXYZ, bias);
    }

    // ── Directional sun ───────────────────────────────────────────────────────
    if (sunEnabled != 0u) {
        vec3 L        = normalize(-lights.sunDirection.xyz);
        vec3 radiance = lights.sunColor.rgb * lights.sunColor.w;
        Lo += evalBRDF(albedo, roughness, metallic, N, V, L, F0, radiance) * shadow;
    }

    // ── Point lights ──────────────────────────────────────────────────────────
    for (uint i = 0u; i < pointCount; ++i) {
        vec3  lpos    = lights.points[i].position.xyz;
        float radius  = lights.points[i].params.x;
        float atten   = pointAtten(worldPos, lpos, radius);
        if (atten <= 0.0) continue;
        vec3  L        = normalize(lpos - worldPos);
        vec3  radiance = lights.points[i].color.rgb
                       * lights.points[i].color.w * atten;
        Lo += evalBRDF(albedo, roughness, metallic, N, V, L, F0, radiance);
    }

    // ── IBL ambient ───────────────────────────────────────────────────────────
    vec3 ambient = vec3(0.03) * albedo * ao;

    if (iblIntensity > 0.0) {
        vec3 kS = FresnelSchlickRoughness(NdotV, F0, roughness);
        vec3 kD = (1.0 - kS) * (1.0 - metallic);

        vec3 irradiance  = texture(irradianceCube, N).rgb;
        vec3 diffuse     = kD * irradiance * albedo;

        vec3 R           = reflect(-V, N);
        float mipLevel   = roughness * 4.0;
        vec3 prefiltered = textureLod(prefilteredCube, R, mipLevel).rgb;
        vec2 envBRDF     = texture(brdfLut, vec2(NdotV, roughness)).rg;
        vec3 specular    = prefiltered * (kS * envBRDF.x + envBRDF.y);

        ambient = (diffuse + specular) * ao * iblIntensity;
    }

    // ── Compose ───────────────────────────────────────────────────────────────
    outColor = vec4(Lo + ambient + emissive, 1.0);
}
