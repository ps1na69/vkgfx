#version 450
// geometry.frag
// Geometry pass fragment shader.
// Samples the material textures and writes all surface properties into the
// G-buffer so the lighting pass can work entirely in screen space.
//
// Normal mapping: reads a tangent-space normal from normalMap, transforms it
// to world space via the TBN matrix from the vertex shader.  This is
// fundamental for high-frequency surface detail without extra geometry.

// ─── G-buffer output attachments (one per layout location) ──────────────────
layout(location = 0) out vec4 outWorldPos;    // RGBA16_SFLOAT
layout(location = 1) out vec4 outNormal;      // RGBA16_SFLOAT
layout(location = 2) out vec4 outAlbedo;      // RGBA8_UNORM  (a = baked AO)
layout(location = 3) out vec4 outMaterial;    // RGBA8_UNORM  (r=metallic, g=roughness)
layout(location = 4) out vec4 outEmissive;    // RGBA16_SFLOAT

// ─── Inputs from vertex shader ──────────────────────────────────────────────
layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inT;
layout(location = 3) in vec3 inB;
layout(location = 4) in vec3 inN;

// ─── Material textures (set 1, bindings 0..5) ────────────────────────────────
// All textures follow the glTF 2.0 PBR metallic-roughness convention.
layout(set = 1, binding = 0) uniform sampler2D albedoMap;
layout(set = 1, binding = 1) uniform sampler2D normalMap;
layout(set = 1, binding = 2) uniform sampler2D metallicRoughnessMap; // B=metallic, G=roughness
layout(set = 1, binding = 3) uniform sampler2D emissiveMap;
layout(set = 1, binding = 4) uniform sampler2D aoMap;                // baked ambient occlusion

// ─── Per-material constants (set 1, binding 5) ──────────────────────────────
layout(set = 1, binding = 5) uniform MaterialUBO {
    vec4  albedoFactor;         // Multiplied with albedoMap sample
    float metallicFactor;
    float roughnessFactor;
    float emissiveStrength;     // HDR emissive scale factor
    float alphaCutoff;          // For alpha-masked materials; 0 = disabled
} mat;

void main()
{
    // ── Albedo ────────────────────────────────────────────────────────────────
    vec4 albedoSample = texture(albedoMap, inTexCoord) * mat.albedoFactor;

    // Alpha masking – discard fragments below the cutoff (e.g. foliage).
    if (mat.alphaCutoff > 0.0 && albedoSample.a < mat.alphaCutoff)
        discard;

    // ── Normal mapping ────────────────────────────────────────────────────────
    // Sample tangent-space normal [0,1] and remap to [-1,+1].
    vec3 tsNormal = texture(normalMap, inTexCoord).xyz * 2.0 - 1.0;
    // The z component can be reconstructed for two-channel normal maps:
    //   tsNormal.z = sqrt(1.0 - dot(tsNormal.xy, tsNormal.xy));
    // Here we assume a full three-channel map.

    // Transform from tangent space to world space using the TBN matrix.
    // mat3(inT, inB, inN) is the TBN matrix; multiplying by tsNormal gives
    // the world-space normal.
    mat3 TBN       = mat3(normalize(inT), normalize(inB), normalize(inN));
    vec3 worldNorm = normalize(TBN * tsNormal);

    // ── Metallic / Roughness ──────────────────────────────────────────────────
    vec4 mrSample = texture(metallicRoughnessMap, inTexCoord);
    float metallic  = mrSample.b * mat.metallicFactor;
    float roughness = mrSample.g * mat.roughnessFactor;
    // Clamp roughness to [0.05, 1] to avoid specular aliasing from a
    // perfectly smooth surface when the BRDF denominator approaches zero.
    roughness = clamp(roughness, 0.05, 1.0);

    // ── Emissive ──────────────────────────────────────────────────────────────
    vec3 emissive = texture(emissiveMap, inTexCoord).rgb * mat.emissiveStrength;

    // ── Ambient occlusion (baked) ─────────────────────────────────────────────
    float ao = texture(aoMap, inTexCoord).r;

    // ── Write G-buffer ────────────────────────────────────────────────────────
    outWorldPos  = vec4(inWorldPos, 1.0);
    outNormal    = vec4(worldNorm,  0.0);
    outAlbedo    = vec4(albedoSample.rgb, ao);      // Pack baked AO into alpha
    outMaterial  = vec4(metallic, roughness, 0.0, 0.0);
    outEmissive  = vec4(emissive, 1.0);
}
