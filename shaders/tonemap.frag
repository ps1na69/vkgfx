#version 450
// tonemap.frag
// HDR tone-mapping and gamma correction.
//
// Converts the linear HDR radiance from the lighting pass into display-referred
// LDR colour suitable for an 8-bit swapchain.
//
// Tone-map operators:
//   0 – ACES Filmic (default) : industry-standard, good contrast and colour
//   1 – Reinhard               : simple, perceptually uniform
//   2 – Uncharted 2 / Hable    : cinematic with shoulder and toe
//
// After tone mapping, sRGB gamma (γ ≈ 2.2) is applied.  If the swapchain
// image has the VK_FORMAT_*_SRGB suffix this is done in hardware automatically;
// otherwise we apply pow(color, 1/2.2) here.

layout(location = 0) in  vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D hdrImage;

layout(push_constant) uniform TonemapPC {
    float exposure;     // EV offset applied before tone mapping
    uint  operator_;    // 0=ACES, 1=Reinhard, 2=Uncharted2
} pc;

// ─── ACES Filmic ─────────────────────────────────────────────────────────────
// Narkowicz 2015: "ACES Filmic Tone Mapping Curve"
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 aces_filmic(vec3 x)
{
    const float A = 2.51;
    const float B = 0.03;
    const float C = 2.43;
    const float D = 0.59;
    const float E = 0.14;
    return clamp((x * (A * x + B)) / (x * (C * x + D) + E), 0.0, 1.0);
}

// ─── Reinhard ────────────────────────────────────────────────────────────────
vec3 reinhard(vec3 x)
{
    return x / (x + 1.0);
}

// ─── Uncharted 2 (Hable) ─────────────────────────────────────────────────────
vec3 hable_partial(vec3 x)
{
    const float A = 0.15, B = 0.50, C = 0.10;
    const float D = 0.20, E = 0.02, F = 0.30;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}
vec3 hable(vec3 x, float exposure)
{
    float W = 11.2;   // Linear white point
    vec3 num = hable_partial(x * exposure * 2.0);
    vec3 den = hable_partial(vec3(W));
    return num / den;
}

void main()
{
    vec3 hdr = texture(hdrImage, inUV).rgb;

    // Apply exposure: multiply by 2^EV.
    hdr *= pow(2.0, pc.exposure);

    vec3 ldr;
    if (pc.operator_ == 0u)
        ldr = aces_filmic(hdr);
    else if (pc.operator_ == 1u)
        ldr = reinhard(hdr);
    else
        ldr = hable(hdr, 1.0);

    // sRGB gamma correction (apply only when swapchain format is UNORM, not SRGB).
    // pow(x, 1/2.2) is a close approximation of the full sRGB piecewise function.
    ldr = pow(clamp(ldr, 0.0, 1.0), vec3(1.0 / 2.2));

    outColor = vec4(ldr, 1.0);
}
