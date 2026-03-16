#version 450
// shaders/lighting.frag
// Deferred lighting pass.
// Reads G-buffer and applies:
//   - Cook-Torrance PBR (GGX NDF, Smith-G, Schlick-F)
//   - IBL (irradiance + prefiltered + BRDF LUT) — toggled via iblIntensity
//   - Directional sun light — toggled via sun.enabled
//   - Point lights
//   - G-buffer debug view (gbufferDebug != 0)

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

// G-buffer samplers — set=0
layout(set = 0, binding = 0) uniform sampler2D gAlbedo;
layout(set = 0, binding = 1) uniform sampler2D gNormal;
layout(set = 0, binding = 2) uniform sampler2D gRMA;
layout(set = 0, binding = 3) uniform sampler2D gDepth;

// Scene UBO — set=1, binding=0
layout(set = 1, binding = 0) uniform SceneUBO {
    mat4  view;
    mat4  proj;
    mat4  viewProj;
    vec4  cameraPos;
    vec2  viewport;
    float time;
    float _pad0;
} scene;

// Light UBO — set=1, binding=1
struct PointLight { vec4 pos; vec4 color; float radius; float pad[3]; };
layout(set = 1, binding = 1) uniform LightUBO {
    vec4      sunDir;
    vec4      sunColor;    // w = intensity
    uint      sunEnabled;
    float     pad0[3];
    PointLight points[8];
    uint       pointCount;
    float      iblIntensity;
    uint       gbufferDebug;
    float      _pad;
} lights;

// IBL samplers — set=2
layout(set = 2, binding = 0) uniform samplerCube irradianceCube;
layout(set = 2, binding = 1) uniform samplerCube prefilteredCube;
layout(set = 2, binding = 2) uniform sampler2D   brdfLut;

// ── BRDF helpers ──────────────────────────────────────────────────────────────
const float PI = 3.14159265359;

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float NH = max(dot(N, H), 0.0);
    float d  = NH * NH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return GeometrySchlickGGX(max(dot(N,V),0.0), roughness)
         * GeometrySchlickGGX(max(dot(N,L),0.0), roughness);
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Reconstruct world position from depth buffer
vec3 reconstructWorldPos(float depth) {
    vec2 ndc = inUV * 2.0 - 1.0;
    vec4 clip = vec4(ndc, depth, 1.0);
    vec4 view = inverse(scene.proj)  * clip;
    view /= view.w;
    vec4 world = inverse(scene.view) * view;
    return world.xyz;
}

vec3 evalLight(vec3 albedo, float roughness, float metallic,
               vec3 N, vec3 V, vec3 F0,
               vec3 L, vec3 radiance) {
    vec3  H   = normalize(V + L);
    float NDF = DistributionGGX(N, H, roughness);
    float G   = GeometrySmith(N, V, L, roughness);
    vec3  F   = FresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3  num = NDF * G * F;
    float den = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 1e-4;
    vec3  spec = num / den;
    vec3  kD   = (vec3(1.0) - F) * (1.0 - metallic);
    float NL   = max(dot(N, L), 0.0);
    return (kD * albedo / PI + spec) * radiance * NL;
}

void main() {
    // ── G-buffer read ─────────────────────────────────────────────────────────
    vec4  rawAlbedo = texture(gAlbedo,  inUV);
    vec4  rawNormal = texture(gNormal,  inUV);
    vec3  rma       = texture(gRMA,     inUV).rgb;
    float depth     = texture(gDepth,   inUV).r;

    vec3  albedo    = rawAlbedo.rgb;
    vec3  N         = normalize(rawNormal.rgb * 2.0 - 1.0);
    float roughness = rma.r;
    float metallic  = rma.g;
    float ao        = rma.b;

    // ── Debug views ───────────────────────────────────────────────────────────
    if (lights.gbufferDebug == 1) { outColor = vec4(albedo, 1.0);                    return; }
    if (lights.gbufferDebug == 2) { outColor = vec4(N * 0.5 + 0.5, 1.0);            return; }
    if (lights.gbufferDebug == 3) { outColor = vec4(vec3(roughness), 1.0);           return; }
    if (lights.gbufferDebug == 4) { outColor = vec4(vec3(metallic), 1.0);            return; }
    if (lights.gbufferDebug == 5) { outColor = vec4(vec3(depth), 1.0);               return; }
    if (lights.gbufferDebug == 6) { outColor = vec4(vec3(ao), 1.0);                  return; }

    // ── PBR setup ─────────────────────────────────────────────────────────────
    vec3 worldPos = reconstructWorldPos(depth);
    vec3 V   = normalize(scene.cameraPos.xyz - worldPos);
    vec3 F0  = mix(vec3(0.04), albedo, metallic);
    vec3 Lo  = vec3(0.0);

    // ── Sun light ─────────────────────────────────────────────────────────────
    if (lights.sunEnabled != 0) {
        vec3 L        = normalize(-lights.sunDir.xyz);
        vec3 radiance = lights.sunColor.rgb * lights.sunColor.w;
        Lo += evalLight(albedo, roughness, metallic, N, V, F0, L, radiance);
    }

    // ── Point lights ──────────────────────────────────────────────────────────
    for (uint i = 0; i < lights.pointCount; ++i) {
        vec3  lPos     = lights.points[i].pos.xyz;
        vec3  L        = normalize(lPos - worldPos);
        float dist     = length(lPos - worldPos);
        float atten    = clamp(1.0 - dist / lights.points[i].radius, 0.0, 1.0);
        atten          = atten * atten;
        vec3  radiance = lights.points[i].color.rgb * lights.points[i].color.w * atten;
        Lo += evalLight(albedo, roughness, metallic, N, V, F0, L, radiance);
    }

    // ── IBL ───────────────────────────────────────────────────────────────────
    vec3 ambient = vec3(0.03) * albedo * ao;
    if (lights.iblIntensity > 0.0) {
        vec3  kS         = FresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
        vec3  kD         = (1.0 - kS) * (1.0 - metallic);
        vec3  irradiance = texture(irradianceCube, N).rgb;
        vec3  diffuse    = kD * irradiance * albedo;

        const float MAX_REFLECT_LOD = 4.0;
        vec3  R          = reflect(-V, N);
        vec3  prefiltered= textureLod(prefilteredCube, R, roughness * MAX_REFLECT_LOD).rgb;
        vec2  envBRDF    = texture(brdfLut, vec2(max(dot(N, V), 0.0), roughness)).rg;
        vec3  specular   = prefiltered * (kS * envBRDF.x + envBRDF.y);

        ambient = (diffuse + specular) * ao * lights.iblIntensity;
    }

    // ── Final ─────────────────────────────────────────────────────────────────
    outColor = vec4(Lo + ambient, 1.0);
}
