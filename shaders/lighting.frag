#version 450
// lighting.frag — Deferred PBR lighting with optional IBL and cascaded shadow maps.
//
// pc.flags bit 0: hasIBL    — sample irradiance/prefiltered/brdfLUT from set 3
// pc.flags bit 1: hasShadow — apply shadow factor from set 4
// When IBL is absent, a flat ambient (pc.ambient*) is used instead.

layout(location = 0) in  vec2 inUV;
layout(location = 0) out vec4 outHDR;

// ── G-buffer (set 0) ─────────────────────────────────────────────────────────
layout(set = 0, binding = 0) uniform sampler2D gWorldPos;
layout(set = 0, binding = 1) uniform sampler2D gNormal;
layout(set = 0, binding = 2) uniform sampler2D gAlbedo;
layout(set = 0, binding = 3) uniform sampler2D gMaterial;
layout(set = 0, binding = 4) uniform sampler2D gEmissive;
layout(set = 0, binding = 5) uniform sampler2D gDepth;
layout(set = 0, binding = 6) uniform sampler2D aoTexture;

// ── Frame UBO (set 1) ────────────────────────────────────────────────────────
layout(set = 1, binding = 0) uniform FrameUBO {
    mat4  view; mat4 proj; mat4 viewProj; mat4 invView; mat4 invProj;
    vec4  cameraPos; float time;
} frame;

// ── Light SSBO (set 2) ───────────────────────────────────────────────────────
struct GpuLight {
    vec4 position; vec4 direction; vec4 color;
    uint type; float innerAngleCos; float pad[2];
};
layout(set = 2, binding = 0) readonly buffer LightBuffer {
    uint count; GpuLight lights[];
} lightBuf;

// ── IBL (set 3) — always bound; ignored when pc.flags bit 0 is clear ─────────
layout(set = 3, binding = 0) uniform samplerCube irradianceMap;
layout(set = 3, binding = 1) uniform samplerCube prefilteredMap;
layout(set = 3, binding = 2) uniform sampler2D   brdfLUT;

// ── Shadow maps (set 4) — always bound; ignored when pc.flags bit 1 is clear ─
layout(set = 4, binding = 0) uniform sampler2DArrayShadow shadowMap;
layout(set = 4, binding = 1) uniform ShadowUBO {
    mat4  lightSpaceMatrix[4];
    float splitDepths[4];
} shadow;

// ── Push constant — feature flags + flat ambient ─────────────────────────────
layout(push_constant) uniform LightingPC {
    uint  flags;             // bit 0=hasIBL, bit 1=hasShadow
                             // bits 4-7 = debug vis: 0=normal, 1=albedo, 2=normal, 3=metallic,
                             //                        4=roughness, 5=worldpos, 6=depth, 7=ao
    float ambientR;
    float ambientG;
    float ambientB;
    float ambientIntensity;
} pc;

const float PI     = 3.14159265359;
const float INV_PI = 0.31830988618;
const float MIN_ROUGH = 0.04;
const int   NUM_CASCADES = 4;

float D_GGX(float NdotH, float r) {
    float a2 = r*r*r*r;
    float d  = NdotH*NdotH*(a2-1.0)+1.0;
    return a2/(PI*d*d);
}
float G1(float n, float k) { return n/(n*(1.0-k)+k); }
float G_Smith(float NdotV, float NdotL, float r) {
    float rk = r+1.0; float k=(rk*rk)*0.125;
    return G1(NdotV,k)*G1(NdotL,k);
}
vec3 F_Schlick(float cos, vec3 F0)               { return F0+(1.0-F0)*pow(clamp(1.0-cos,0.0,1.0),5.0); }
vec3 F_SchlickR(float cos, vec3 F0, float r)     { return F0+(max(vec3(1.0-r),F0)-F0)*pow(clamp(1.0-cos,0.0,1.0),5.0); }

float point_atten(float d, float r) {
    float d2=d*d,r2=r*r;
    return (pow(clamp(1.0-(d2*d2)/(r2*r2),0.0,1.0),2.0))/max(d2,0.0001);
}

// Returns shadow multiplier [0,1] — 1.0 = fully lit, 0.0 = fully shadowed.
// Returns 1.0 (no shadow) when hasShadow flag is clear.
float shadowFactor(vec3 worldPos, float viewDepth) {
    if ((pc.flags & 2u) == 0u) return 1.0;
    int ci = NUM_CASCADES-1;
    for (int i = 0; i < NUM_CASCADES-1; ++i)
        if (viewDepth < shadow.splitDepths[i]) { ci = i; break; }
    vec4  ls   = shadow.lightSpaceMatrix[ci]*vec4(worldPos,1.0);
    vec3  proj = ls.xyz/ls.w;
    vec2  uv   = proj.xy*0.5+0.5;
    float dep  = proj.z;
    if (uv.x<0.0||uv.x>1.0||uv.y<0.0||uv.y>1.0||dep>1.0) return 1.0;
    float s=0.0;
    vec2 ts = vec2(1.0/2048.0);
    for (int dx=-1;dx<=1;++dx) for (int dy=-1;dy<=1;++dy)
        s += texture(shadowMap, vec4(uv+vec2(dx,dy)*ts, float(ci), dep-0.0005));
    return s/9.0;
}

vec3 evaluate_light(uint i, vec3 p, vec3 N, vec3 V, vec3 alb, float m, float r, vec3 F0) {
    GpuLight l  = lightBuf.lights[i];
    vec3 lc     = l.color.rgb*l.color.w;
    vec3 L; float a=1.0;
    if (l.type==1u) { L=normalize(-l.direction.xyz); }
    else {
        vec3 d=l.position.xyz-p; float dist=length(d); L=d/dist;
        a=point_atten(dist,l.position.w);
        if (l.type==2u) a*=smoothstep(l.direction.w,l.innerAngleCos,dot(-L,normalize(l.direction.xyz)));
    }
    vec3  H=normalize(V+L);
    float NdotL=max(dot(N,L),0.0),NdotV=max(dot(N,V),0.0),NdotH=max(dot(N,H),0.0),HdotV=max(dot(H,V),0.0);
    if (NdotL<=0.0) return vec3(0.0);
    float D=(D_GGX(NdotH,r));
    vec3  F=F_Schlick(HdotV,F0);
    float G=G_Smith(NdotV,NdotL,r);
    vec3 spec=(D*F*G)/(4.0*max(NdotV,0.001)*max(NdotL,0.001));
    spec = min(spec, vec3(10.0)); // clamp specular peaks before ACES to prevent blow-out
    vec3 kD=(vec3(1.0)-F)*(1.0-m);
    return (kD*alb*INV_PI+spec)*lc*a*NdotL;
}

// IBL ambient — only called when hasIBL flag is set.
vec3 iblAmbient(vec3 N, vec3 V, vec3 alb, float m, float r, float ao, vec3 F0) {
    float NdotV=max(dot(N,V),0.0);
    vec3 irr    = texture(irradianceMap,N).rgb;
    vec3 Fd     = F_SchlickR(NdotV,F0,r);
    vec3 kD     = (1.0-Fd)*(1.0-m);
    vec3 diff   = kD*irr*alb;
    vec3 R      = reflect(-V,N);
    float maxLOD = float(textureQueryLevels(prefilteredMap) - 1);
    vec3 pf     = textureLod(prefilteredMap, R, r * maxLOD).rgb;
    vec2 brdf   = texture(brdfLUT,vec2(NdotV,r)).rg;
    vec3 spec   = pf*(Fd*brdf.x+brdf.y);
    return (diff+spec)*ao;
}

// Flat ambient — Lambertian diffuse tinted by scene ambient color.
vec3 flatAmbient(vec3 alb, float ao, float m) {
    vec3 ambCol = vec3(pc.ambientR, pc.ambientG, pc.ambientB) * pc.ambientIntensity;
    return ambCol * alb * (1.0 - m) * ao;
}

void main() {
    // Early-out for background pixels — no geometry was drawn here.
    // The depth buffer is cleared to 1.0; geometry always writes a smaller value.
    // Without this, normalize(vec3(0)) = NaN propagates into IBL sampling and
    // produces undefined (driver-specific) colours — visible as cyan or white garbage.
    float depth = texture(gDepth, inUV).r;
    if (depth >= 1.0) {
        outHDR = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // ── Debug G-buffer visualization ─────────────────────────────────────────
    // Set RendererSettings::debugGBuffer = 1..7 to visualize individual channels.
    uint dbg = (pc.flags >> 4u) & 0xFu;
    if (dbg != 0u) {
        if (dbg == 1u) { outHDR = vec4(texture(gAlbedo,   inUV).rgb, 1.0); return; }
        if (dbg == 2u) { outHDR = vec4(texture(gNormal,   inUV).rgb * 0.5 + 0.5, 1.0); return; }
        if (dbg == 3u) { outHDR = vec4(vec3(texture(gMaterial, inUV).r), 1.0); return; } // metallic
        if (dbg == 4u) { outHDR = vec4(vec3(texture(gMaterial, inUV).g), 1.0); return; } // roughness
        if (dbg == 5u) { outHDR = vec4(fract(texture(gWorldPos, inUV).rgb), 1.0); return; }
        if (dbg == 6u) { outHDR = vec4(vec3(texture(gDepth, inUV).r), 1.0); return; }
        if (dbg == 7u) { outHDR = vec4(vec3(texture(aoTexture, inUV).r), 1.0); return; }
    }

    vec3  worldPos = texture(gWorldPos,inUV).rgb;
    vec3  rawN     = texture(gNormal,inUV).rgb;
    // Guard against zero-length or NaN normals (unwritten pixels, or degenerate TBN
    // from procedural meshes before the geometry pass was patched).
    if (any(isnan(rawN)) || any(isinf(rawN)) || dot(rawN, rawN) < 0.01) {
        outHDR = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec3  N        = normalize(rawN);
    vec4  albAO    = texture(gAlbedo,inUV);
    vec4  mat      = texture(gMaterial,inUV);
    vec3  emissive = texture(gEmissive,inUV).rgb;
    vec3  albedo   = albAO.rgb;
    float bakedAO  = albAO.a;
    float metallic = mat.r;
    float rough    = max(mat.g, MIN_ROUGH);
    float ssao     = texture(aoTexture,inUV).r;
    float finalAO  = bakedAO * ssao;
    vec3  V        = normalize(frame.cameraPos.xyz - worldPos);
    vec3  F0       = mix(vec3(0.04), albedo, metallic);
    float viewDep  = (frame.view * vec4(worldPos, 1.0)).z;

    // Direct lighting
    vec3 Lo = vec3(0.0);
    for (uint i = 0; i < lightBuf.count; ++i) {
        vec3 c = evaluate_light(i, worldPos, N, V, albedo, metallic, rough, F0);
        if (lightBuf.lights[i].type == 1u) c *= shadowFactor(worldPos, viewDep);
        Lo += c;
    }

    // Ambient — IBL when available, flat color fallback otherwise
    vec3 ambient;
    if ((pc.flags & 1u) != 0u)
        ambient = iblAmbient(N, V, albedo, metallic, rough, finalAO, F0);
    else
        ambient = flatAmbient(albedo, finalAO, metallic);

    // Guard individual terms before summing — if ambient itself is NaN (e.g. from
    // a corrupt irradiance sample), clamping outHDR to vec4(ambient) won't help.
    if (any(isnan(ambient))  || any(isinf(ambient)))  ambient  = vec3(0.0);
    if (any(isnan(Lo))       || any(isinf(Lo)))        Lo       = vec3(0.0);
    if (any(isnan(emissive)) || any(isinf(emissive)))  emissive = vec3(0.0);
    outHDR = vec4(ambient + Lo + emissive, 1.0);
}
