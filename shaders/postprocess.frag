#version 450

layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D uScene;

layout(set = 0, binding = 1) uniform PostProcessUBO {
    float exposure;
    float brightness;
    float contrast;
    float saturation;
    vec4  colorBalance;
    float bloomThreshold;
    float bloomStrength;
    float bloomRadius;
    int   toneMapMode;
    int   bloomEnabled;
    int   tonemapEnabled;
    float _pad;
} pp;

// ── Tone mapping ──────────────────────────────────────────────────────────────

vec3 tonemapReinhard(vec3 x) {
    return x / (x + vec3(1.0));
}

vec3 tonemapACES(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// ── Luminance (Rec. 709) ──────────────────────────────────────────────────────
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// ── Bloom ─────────────────────────────────────────────────────────────────────
const float EPS = 1e-5; // ← moved here, before bloomGather

vec3 bloomGather(vec2 uv) {
    vec2 texel = 1.0 / vec2(textureSize(uScene, 0));
    float r    = pp.bloomRadius;

    const vec2 inner[8] = vec2[8](
        vec2( 1.0000,  0.0000),
        vec2( 0.7071,  0.7071),
        vec2( 0.0000,  1.0000),
        vec2(-0.7071,  0.7071),
        vec2(-1.0000,  0.0000),
        vec2(-0.7071, -0.7071),
        vec2( 0.0000, -1.0000),
        vec2( 0.7071, -0.7071)
    );
    const vec2 outer[8] = vec2[8](
        vec2( 0.9239,  0.3827),
        vec2( 0.3827,  0.9239),
        vec2(-0.3827,  0.9239),
        vec2(-0.9239,  0.3827),
        vec2(-0.9239, -0.3827),
        vec2(-0.3827, -0.9239),
        vec2( 0.3827, -0.9239),
        vec2( 0.9239, -0.3827)
    );

    vec3  acc  = vec3(0.0);
    float wSum = 0.0;

    for (int i = 0; i < 8; i++) {
        vec3  s = texture(uScene, uv + inner[i] * r * texel).rgb * pp.exposure;
        float w = max(luma(s) - pp.bloomThreshold, 0.0);
        acc  += s * w;
        wSum += w;
    }
    for (int i = 0; i < 8; i++) {
        vec3  s = texture(uScene, uv + outer[i] * r * 2.5 * texel).rgb * pp.exposure;
        float w = max(luma(s) - pp.bloomThreshold, 0.0) * 0.45;
        acc  += s * w;
        wSum += w;
    }

    return (wSum > EPS) ? (acc / wSum) * pp.bloomStrength : vec3(0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
void main() {
    // 1. Sample + exposure
    vec3 color = texture(uScene, vUV).rgb * pp.exposure;

    // 2. Bloom
    if (pp.bloomEnabled != 0)
        color += bloomGather(vUV);

    // 3. Tone mapping (HDR → [0,1])
    if (pp.tonemapEnabled != 0) {
        if      (pp.toneMapMode == 2) color = tonemapACES(color);
        else if (pp.toneMapMode == 1) color = tonemapReinhard(color);
        else                          color = clamp(color, 0.0, 1.0);
    }

    // 4. Brightness
    color = color + vec3(pp.brightness);

    // 5. Contrast
    color = (color - 0.5) * pp.contrast + 0.5;

    // 6. Saturation
    color = mix(vec3(luma(color)), color, pp.saturation);

    // 7. Color balance
    color *= pp.colorBalance.rgb;

    outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}