#version 450

const int   MAX_LIGHTS = 8;
const float EPS        = 1e-5;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;
layout(location = 3) in vec3 fragCamPos;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view; mat4 proj; mat4 viewProj;
    vec4 position; vec4 params;
} camera;

struct LightData {
    vec4 position;   // xyz=pos,  w=type (0=point 1=dir 2=spot)
    vec4 color;      // rgb=color, w=intensity
    vec4 direction;  // xyz=dir,   w=innerCone (radians)
    vec4 params;     // x=outerCone (radians), y=range, z=castShadow, w=shadowMapIdx (-1=none)
};

layout(set = 0, binding = 1) uniform SceneUBO {
    LightData lights[MAX_LIGHTS];
    vec4      ambientColor;  // rgb=color, w=intensity
    int       lightCount;
    int       _pad[3];
} scene;

// Shadow data — same bindings as PBR so the global descriptor set is shared.
layout(set = 0, binding = 2) uniform ShadowUBO {
    mat4  lightSpace[4];
    vec4  params[4];   // x=constantBias, y=normalBias
    int   count;
    int   _pad[3];
} shadows;

layout(set = 0, binding = 3) uniform sampler2DArrayShadow shadowMaps;

layout(set = 1, binding = 0) uniform MaterialUBO {
    vec4  ambient;
    vec4  diffuse;
    vec4  specular;
    float shininess;
    int   useDiffuseMap;
    int   useNormalMap;
    int   _pad;
} mat;

layout(set = 1, binding = 1) uniform sampler2D diffuseMap;
layout(set = 1, binding = 2) uniform sampler2D normalMap;

layout(location = 0) out vec4 outColor;

// ── Attenuation ───────────────────────────────────────────────────────────────

float pointAttenuation(float dist, float range) {
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

// ── PCF shadow — 3x3 kernel with slope-scaled bias ────────────────────────────

float shadowFactor(int idx, vec3 worldPos, vec3 N, vec3 L) {
    vec4 sc = shadows.lightSpace[idx] * vec4(worldPos, 1.0);

    vec3 proj;
    proj.xy = sc.xy / sc.w * 0.5 + 0.5;
    proj.z  = sc.z  / sc.w;   // [0,1] with GLM_FORCE_DEPTH_ZERO_TO_ONE

    // Fragments outside the shadow frustum (any side) are fully lit
    if (proj.x < 0.001 || proj.x > 0.999 ||
        proj.y < 0.001 || proj.y > 0.999 ||
        proj.z < 0.0   || proj.z > 0.999)
        return 1.0;

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
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(fragCamPos - fragWorldPos);

    vec4 diffuseSample = (mat.useDiffuseMap == 1)
                         ? texture(diffuseMap, fragUV)
                         : mat.diffuse;

    // Ambient base
    vec3 color = mat.ambient.rgb * scene.ambientColor.rgb * max(scene.ambientColor.w, 0.05);

    for (int i = 0; i < scene.lightCount; i++) {
        LightData light = scene.lights[i];
        int   ltype  = int(light.position.w);
        vec3  lcolor = light.color.rgb * light.color.w;

        vec3  L;
        float atten = 1.0;

        if (ltype == 0) {
            vec3  delta = light.position.xyz - fragWorldPos;
            float dist  = length(delta);
            L     = delta / max(dist, EPS);
            atten = pointAttenuation(dist, light.params.y);

        } else if (ltype == 1) {
            L = normalize(-light.direction.xyz);

        } else {
            vec3  delta = light.position.xyz - fragWorldPos;
            float dist  = length(delta);
            L     = delta / max(dist, EPS);
            atten = pointAttenuation(dist, light.params.y)
                  * spotAttenuation(L, light.direction.xyz,
                                    light.direction.w, light.params.x);
        }

        float NdotL = max(dot(N, L), 0.0);
        if (NdotL < EPS || atten < EPS) continue;

        // Shadow
        float shadow = 1.0;
        int   sidx   = int(light.params.w);
        if (sidx >= 0 && sidx < shadows.count)
            shadow = shadowFactor(sidx, fragWorldPos, N, L);
        if (shadow < EPS) continue;

        // Diffuse (Lambertian)
        vec3 diffuse = NdotL * diffuseSample.rgb;

        // Specular (Blinn-Phong)
        vec3  H    = normalize(V + L);
        float spec = pow(max(dot(N, H), 0.0), mat.shininess);

        color += (diffuse + spec * mat.specular.rgb) * lcolor * atten * shadow;
    }

    // Gamma encode (Phong always outputs to sRGB swapchain)
    color = pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));

    outColor = vec4(clamp(color, 0.0, 1.0), diffuseSample.a * mat.diffuse.a);
}
