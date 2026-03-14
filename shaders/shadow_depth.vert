#version 450
// shadow_depth.vert — Depth-only pass for cascade shadow map rendering.
// Only position is needed; other vertex attributes are ignored.

layout(location = 0) in vec3 inPosition;

// Push constant: model matrix + precomputed lightSpace matrix for this cascade.
// Passing both matrices via push constant removes the need for any descriptor set,
// keeping the shadow pipeline layout trivially simple.
layout(push_constant) uniform PC {
    mat4 model;
    mat4 lightSpaceMatrix;  // pre-selected for this cascade by the CPU
};

void main() {
    gl_Position = lightSpaceMatrix * model * vec4(inPosition, 1.0);
}
