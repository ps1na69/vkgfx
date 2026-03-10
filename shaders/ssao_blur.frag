#version 450
// ssao_blur.frag
// Simple 4x4 box blur for the raw SSAO output.
// Removes high-frequency noise introduced by the hemisphere sampling without
// blurring across depth discontinuities (objects edges) by using a simple
// depth-aware weight.
//
// A more sophisticated bilateral or cross-bilateral filter would give sharper
// results at object edges but at higher cost.  The box blur is a good default.

layout(location = 0) in  vec2 inUV;
layout(location = 0) out float outAO;

layout(set = 0, binding = 0) uniform sampler2D aoTex;
layout(set = 0, binding = 1) uniform sampler2D depthTex;

layout(push_constant) uniform BlurPC {
    vec2 texelSize;   // 1.0 / vec2(width, height)
} pc;

void main()
{
    float result = 0.0;
    float totalWeight = 0.0;

    // Reference depth for edge-aware weighting.
    float centerDepth = texture(depthTex, inUV).r;

    // 4x4 kernel, offset by -1.5 to -1.5 so it straddles the current pixel.
    for (int x = -1; x <= 2; ++x) {
        for (int y = -1; y <= 2; ++y) {
            vec2  offset    = vec2(float(x), float(y)) * pc.texelSize;
            vec2  sampleUV  = inUV + offset;
            float ao        = texture(aoTex,   sampleUV).r;
            float d         = texture(depthTex, sampleUV).r;

            // Depth similarity weight: reduce contribution across depth edges.
            float depthDiff = abs(d - centerDepth);
            float weight    = exp(-depthDiff * 500.0);   // Sharp depth falloff

            result      += ao * weight;
            totalWeight += weight;
        }
    }

    outAO = (totalWeight > 0.0) ? result / totalWeight : 1.0;
}
