#version 450
// shaders/point_shadow.frag
// Writes gl_FragDepth = distance(fragPos, lightPos) / farPlane.
//
// Storing normalised linear depth allows the lighting pass to reconstruct the
// compared depth trivially: currentDepth = length(fragToLight) / radius.
// Hardware comparison is done by the samplerCubeShadow sampler, so no special
// packing is required — the raw [0,1] depth value is compared directly.

layout(location = 0) in vec3 inFragPos; // world-space position from vertex shader

// set=0 binding=0: per-light data (updated by the CPU per point-light draw).
layout(set = 0, binding = 0) uniform PointShadowLightUBO {
    vec4 lightPosAndFar; // xyz = world-space light position, w = far plane (= light radius)
} light;

void main() {
    float dist     = length(inFragPos - light.lightPosAndFar.xyz);
    gl_FragDepth   = dist / light.lightPosAndFar.w;
}
