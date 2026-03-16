#version 450
// shaders/tonemap.frag
// Reinhard tonemapping + gamma correction.

layout(location = 0) in  vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D hdrInput;

layout(push_constant) uniform TonemapPush {
    float exposure;
    float gamma;
} push;

void main() {
    vec3 hdr = texture(hdrInput, inUV).rgb;
    // Exposure
    hdr *= push.exposure;
    // Reinhard
    vec3 mapped = hdr / (hdr + vec3(1.0));
    // Gamma
    mapped = pow(mapped, vec3(1.0 / push.gamma));
    outColor = vec4(mapped, 1.0);
}
