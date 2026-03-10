#version 450
layout(early_fragment_tests) in;

const float PI     = 3.14159265359;
const float INV_PI = 0.31830988618;
const float EPS    = 1e-5;
const int   MAX_LIGHTS = 8;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;
layout(location = 3) in vec3 fragTangent;
layout(location = 4) in vec3 fragBitangent;
layout(location = 5) in vec3 fragCamPos;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view; mat4 proj; mat4 viewProj;
    vec4 position; vec4 params;
} camera;

struct LightData {
    vec4 position;   // xyz=world pos,  w=type (0=point 1=dir 2=spot)
    vec4 color;      // rgb=color,       w=intensity
    vec4 direction;  // xyz=direction,   w=innerCone (radians)
    vec4 params;     // x=outerCone,     y=range, z=castShadow, w=shadowMapIdx (-1=none)
};

layout(set = 0, binding = 1) uniform SceneUBO {
    LightData lights[MAX_LIGHTS];
    vec4      ambientColor;   // rgb=color, w=intensity
    int       lightCount;
    int       useLinearOutput;
    int       _pad[2];
} scene;

layout(set = 0, binding = 2) uniform ShadowUBO {
    mat4  lightSpace[4];
    vec4  params[4];   // x=constantBias, y=normalBias
    int   count;
    int   _pad[3];
} shadows;

layout(set = 0, binding = 3) uniform sampler2DArrayShadow shadowMaps;

layout(set = 1, binding = 0) uniform MaterialUBO {
    vec4  albedo;
    float metallic;
    float roughness;
    float ao;
    float emissiveScale;
    vec4  emissiveColor;
    int   useAlbedoMap;
    int   useNormalMap;
    int   useMetalRoughMap;
    int   useAOMap;
    int   useEmissiveMap;
    int   _pad[3];
} mat;

layout(set = 1, binding = 1) uniform sampler2D albedoMap;
layout(set = 1, binding = 2) uniform sampler2D normalMap;
layout(set = 1, binding = 3) uniform sampler2D metalRoughMap;
layout(set = 1, binding = 4) uniform sampler2D aoMap;
layout(set = 1, binding = 5) uniform sampler2D emissiveMap;

layout(location = 0) out vec4 outColor;

// ── GGX/Cook-Torrance BRDF ────────────────────────────────────────────────────

float D_GGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, EPS);
}

float G_Smith(float NdotV, float NdotL, float roughness) {
    float r  = roughness + 1.0;
    float k  = (r * r) * 0.125;
    float gv = NdotV / max(NdotV * (1.0 - k) + k, EPS);
    float gl = NdotL / max(NdotL * (1.0 - k) + k, EPS);
    return gv * gl;
}

vec3 F_Schlick(float cosTheta, vec3 F0) {
    float f = clamp(1.0 - cosTheta, 0.0, 1.0);
    float f2 = f * f;
    return F0 + (1.0 - F0) * f2 * f2 * f;
}

// ── Attenuation ───────────────────────────────────────────────────────────────

float pointAttenuation(float dist, float range) {
    // Physically-based inverse-square with smooth windowing at range
    float r  = dist / max(range, EPS);
    float r2 = r * r;
    float w  = clamp(1.0 - r2 * r2, 0.0, 1.0);
    return (w * w) / max(dist * dist + 1.0, EPS);
}

float spotAttenuation(vec3 L, vec3 spotDir, float innerCone, float outerCone) {
    // innerCone / outerCone are in radians; dot() returns a cosine, so convert.
    float cosAngle = dot(-L, normalize(spotDir));
    float cosInner = cos(innerCone);
    float cosOuter = cos(outerCone);
    return clamp((cosAngle - cosOuter) / max(cosInner - cosOuter, EPS), 0.0, 1.0);
}

// ── PCF shadow — 3×3 kernel with slope-scaled bias ────────────────────────────

float shadowFactor(int idx, vec3 worldPos, vec3 N, vec3 L) {
    vec4 sc = shadows.lightSpace[idx] * vec4(worldPos, 1.0);

    // Perspective divide
    vec3 proj;
    proj.xy = sc.xy / sc.w * 0.5 + 0.5;
    proj.z  = sc.z  / sc.w;   // [0,1] with GLM_FORCE_DEPTH_ZERO_TO_ONE

    // Fragments outside the shadow frustum (any side) are fully lit
    if (proj.x < 0.001 || proj.x > 0.999 ||
        proj.y < 0.001 || proj.y > 0.999 ||
        proj.z < 0.0   || proj.z > 0.999)
        return 1.0;

    // Slope-scaled bias: more bias when light hits at a shallow angle
    float NdotL = clamp(dot(N, L), 0.0, 1.0);
    float bias  = max(shadows.params[idx].y * (1.0 - NdotL), shadows.params[idx].x);
    float depth = proj.z - bias;

    vec2  ts     = 1.0 / vec2(textureSize(shadowMaps, 0).xy);
    float shadow = 0.0;
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
            shadow += texture(shadowMaps, vec4(proj.xy + vec2(x, y) * ts, float(idx), depth));
    return shadow / 9.0;
}

// ─────────────────────────────────────────────────────────────────────────────

void main() {

    // ── Albedo ────────────────────────────────────────────────────────────────
    // Textures are uploaded as VK_FORMAT_R8G8B8A8_SRGB so the Vulkan sampler
    // converts sRGB→linear automatically. Do NOT apply pow(x, 2.2) manually —
    // that would apply gamma correction twice, making colours far too dark.
    vec4 albedoSample = (mat.useAlbedoMap == 1)
                        ? texture(albedoMap, fragUV)
                        : mat.albedo;
    vec3  albedo = albedoSample.rgb;             // already linear
    float alpha  = albedoSample.a * mat.albedo.a;

    // ── Metal / Roughness / AO ────────────────────────────────────────────────
    float metallic  = mat.metallic;
    float roughness = clamp(mat.roughness, 0.04, 1.0);
    float ao        = mat.ao;

    if (mat.useMetalRoughMap == 1) {
        vec3 mr   = texture(metalRoughMap, fragUV).rgb;
        roughness = clamp(mr.g, 0.04, 1.0);
        metallic  = clamp(mr.b, 0.0,  1.0);
    }
    if (mat.useAOMap == 1)
        ao = texture(aoMap, fragUV).r;

    // ── Normal ────────────────────────────────────────────────────────────────
    vec3 N = normalize(fragNormal);
    if (mat.useNormalMap == 1) {
        vec3 tn = texture(normalMap, fragUV).rgb * 2.0 - 1.0;
        vec3 T  = normalize(fragTangent);
        vec3 B  = normalize(fragBitangent);
        N = normalize(mat3(T, B, N) * tn);
    }

    // V = unit vector from surface fragment toward the camera
    vec3 V = normalize(fragCamPos - fragWorldPos);

    // Specular colour at normal incidence (F0)
    // Dielectrics: 0.04 (glass/plastic baseline)
    // Metals:      use the albedo colour
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // ── Per-light radiance sum ────────────────────────────────────────────────
    vec3 Lo = vec3(0.0);

    for (int i = 0; i < scene.lightCount; i++) {
        LightData light = scene.lights[i];
        int  ltype  = int(light.position.w);
        vec3 lcolor = light.color.rgb * light.color.w;

        // L = unit vector from fragment surface toward the light source.
        // For point/spot lights this MUST be recomputed per-fragment from actual
        // world positions. Using a single shared vector would give the same
        // (wrong) direction to every fragment regardless of where it sits on the
        // surface — making every sphere look flat and incorrectly shaded.
        vec3  L;
        float atten = 1.0;

        if (ltype == 0) {
            // ── Point light ───────────────────────────────────────────────────
            vec3  delta = light.position.xyz - fragWorldPos;
            float dist  = length(delta);
            L     = delta / max(dist, EPS);
            atten = pointAttenuation(dist, light.params.y);

        } else if (ltype == 1) {
            // ── Directional light ─────────────────────────────────────────────
            // light.direction.xyz is the direction light travels (toward scene),
            // so the vector toward the light is its negation.
            L = normalize(-light.direction.xyz);

        } else {
            // ── Spot light ────────────────────────────────────────────────────
            vec3  delta = light.position.xyz - fragWorldPos;
            float dist  = length(delta);
            L     = delta / max(dist, EPS);
            atten = pointAttenuation(dist, light.params.y)
                  * spotAttenuation(L, light.direction.xyz,
                                    light.direction.w, light.params.x);
        }

        // cos(angle between N and L). Must be ≥ 0 — negative means back-facing,
        // which should contribute zero (not negative!) light.
        float NdotL = max(dot(N, L), 0.0);
        if (NdotL < EPS || atten < EPS) continue;

        // Shadow
        float shadow  = 1.0;
        int   sidx    = int(light.params.w);
        if (sidx >= 0 && sidx < shadows.count)
            shadow = shadowFactor(sidx, fragWorldPos, N, L);
        if (shadow < EPS) continue;

        // Half-vector for specular
        vec3  H     = normalize(V + L);
        float NdotV = max(dot(N, V), 1e-4);
        float NdotH = max(dot(N, H), 0.0);
        float VdotH = max(dot(V, H), 0.0);

        // Cook-Torrance microfacet BRDF
        float D  = D_GGX(NdotH, roughness);
        float G  = G_Smith(NdotV, NdotL, roughness);
        vec3  F  = F_Schlick(VdotH, F0);

        vec3 spec = (D * G * F) / max(4.0 * NdotV * NdotL, EPS);

        // Diffuse (Lambertian). kS = Fresnel (specular fraction).
        // kD = 1 - kS, zeroed for metals (they have no diffuse scattering).
        vec3 kD      = (vec3(1.0) - F) * (1.0 - metallic);
        vec3 diffuse = kD * albedo * INV_PI;

        Lo += (diffuse + spec) * lcolor * atten * NdotL * shadow;
    }

    // ── Ambient ───────────────────────────────────────────────────────────────
    // Clamp minimum to 0.05 so the dark side of objects never goes pitch-black.
    // Real environments always have some scattered/bounced light on every surface.
    float ambI = max(scene.ambientColor.w, 0.05);
    vec3  ambC = scene.ambientColor.rgb;
    vec3  ambient = ambC * ambI * albedo * ao;

    // ── Emission ──────────────────────────────────────────────────────────────
    // emissiveMap also uses sRGB format — hardware linearises it automatically.
    vec3 emissive = mat.emissiveColor.rgb * mat.emissiveScale;
    if (mat.useEmissiveMap == 1)
        emissive *= texture(emissiveMap, fragUV).rgb;

    // ── Composite ─────────────────────────────────────────────────────────────
    vec3 color = ambient + Lo + emissive;

    // Tone map + gamma encode only when no post-process pass handles it later
    if (scene.useLinearOutput == 0) {
        color = color / (color + vec3(1.0));                    // Reinhard
        color = pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));   // gamma encode
    }

    outColor = vec4(color, alpha);
}
