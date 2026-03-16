#version 450
// shaders/gbuffer.frag
// G-buffer fill pass.
// Outputs (attachment layout must match Renderer::initGBuffer):
//   0 = albedo   (RGBA8_UNORM)
//   1 = normal   (RGBA16_SFLOAT, xyz in world space, w unused)
//   2 = RMA      (RGBA8_UNORM: r=roughness, g=metallic, b=ao, a=unused)
//   depth        (D32_SFLOAT handled implicitly)

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inUV;

// Material textures — set=1
layout(set = 1, binding = 0) uniform sampler2D albedoTex;
layout(set = 1, binding = 1) uniform sampler2D normalTex;
layout(set = 1, binding = 2) uniform sampler2D rmaTex;     // R=roughness, G=metallic, B=ao

// Material params — set=1, binding=3
layout(set = 1, binding = 3) uniform PBRParams {
    vec4     albedo;
    float    roughness;
    float    metallic;
    float    ao;
    float    emissive;
    uint     hasAlbedoTex;
    uint     hasNormalTex;
    uint     hasRmaTex;
    float    _pad;
} mat;

layout(location = 0) out vec4 outAlbedo;   // attachment 0
layout(location = 1) out vec4 outNormal;   // attachment 1
layout(location = 2) out vec4 outRMA;      // attachment 2

void main() {
    // Albedo
    vec4 albedo = mat.albedo;
    if (mat.hasAlbedoTex != 0)
        albedo *= texture(albedoTex, inUV);
    outAlbedo = albedo;

    // Normal (TBN perturbation if normal map present)
    vec3 N = normalize(inNormal);
    if (mat.hasNormalTex != 0) {
        vec3 T  = normalize(inTangent.xyz);
        vec3 B  = cross(N, T) * inTangent.w;
        mat3 TBN = mat3(T, B, N);
        vec3 tn = texture(normalTex, inUV).xyz * 2.0 - 1.0;
        N = normalize(TBN * tn);
    }
    outNormal = vec4(N * 0.5 + 0.5, 1.0);  // encode to [0,1]

    // Roughness / Metallic / AO
    float r = mat.roughness;
    float m = mat.metallic;
    float a = mat.ao;
    if (mat.hasRmaTex != 0) {
        vec3 rma = texture(rmaTex, inUV).rgb;
        r = rma.r;
        m = rma.g;
        a = rma.b;
    }
    outRMA = vec4(r, m, a, 1.0);
}
