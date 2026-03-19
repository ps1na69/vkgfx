#version 450
// shaders/gbuffer.frag  —  G-buffer fill pass.
// Output attachments (must match Renderer::initGBuffer order):
//   0  gAlbedo      RGBA8_UNORM
//   1  gNormal      RGBA16_SFLOAT   world-space encoded to [0,1]
//   2  gRMA         RGBA8_UNORM     r=roughness g=metallic b=ao
//   3  gEmissive    RGBA16_SFLOAT
//   4  gShadowCoord RGBA16_SFLOAT   xy=shadow UV, z=shadow depth
//   depth D32_SFLOAT (implicit)

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inUV;
layout(location = 4) in vec4 inLightSpacePos;

// Material textures (set=1 bindings 0-3)
layout(set = 1, binding = 0) uniform sampler2D albedoTex;
layout(set = 1, binding = 1) uniform sampler2D normalTex;
layout(set = 1, binding = 2) uniform sampler2D rmaTex;
layout(set = 1, binding = 3) uniform sampler2D emissiveTex;

// PBRParams (set=1 binding=4) — must match types.h PBRParams exactly
// albedo  vec4   offset  0
// emissive vec4  offset 16
// pbr     vec4   offset 32   x=roughness y=metallic z=ao
// texFlags uvec4 offset 48   x=hasAlbedo y=hasNormal z=hasRMA w=hasEmissive
layout(set = 1, binding = 4) uniform PBRParams {
    vec4  albedo;
    vec4  emissive;   // rgb=color, w=intensity
    vec4  pbr;        // x=roughness, y=metallic, z=ao
    uvec4 texFlags;   // x=hasAlbedo, y=hasNormal, z=hasRMA, w=hasEmissive
} mat;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outRMA;
layout(location = 3) out vec4 outEmissive;
layout(location = 4) out vec4 outShadowCoord;

void main() {
    // ── Albedo ────────────────────────────────────────────────────────────────
    vec4 albedo = mat.albedo;
    if (mat.texFlags.x != 0u)
        albedo *= texture(albedoTex, inUV);
    if (albedo.a < 0.1) discard;
    outAlbedo = vec4(albedo.rgb, 1.0);

    // ── Normal ────────────────────────────────────────────────────────────────
    vec3 N = normalize(inNormal);
    if (mat.texFlags.y != 0u) {
        vec3 T   = normalize(inTangent.xyz);
        vec3 B   = cross(N, T) * inTangent.w;
        mat3 TBN = mat3(T, B, N);
        vec3 tn  = texture(normalTex, inUV).rgb * 2.0 - 1.0;
        N = normalize(TBN * tn);
    }
    outNormal = vec4(N * 0.5 + 0.5, 1.0);

    // ── RMA ───────────────────────────────────────────────────────────────────
    float r = clamp(mat.pbr.x, 0.05, 1.0);
    float m = mat.pbr.y;
    float a = mat.pbr.z;
    if (mat.texFlags.z != 0u) {
        vec3 rma = texture(rmaTex, inUV).rgb;
        r = clamp(rma.r, 0.05, 1.0);
        m = rma.g;
        a = rma.b;
    }
    outRMA = vec4(r, m, a, 1.0);

    // ── Emissive ──────────────────────────────────────────────────────────────
    vec3 em = mat.emissive.rgb * mat.emissive.w;
    if (mat.texFlags.w != 0u)
        em *= texture(emissiveTex, inUV).rgb;
    outEmissive = vec4(em, 1.0);

    // ── Shadow coord ──────────────────────────────────────────────────────────
    vec3 proj  = inLightSpacePos.xyz / inLightSpacePos.w;
    proj.xy    = proj.xy * 0.5 + 0.5;
    outShadowCoord = vec4(proj, 1.0);
}
