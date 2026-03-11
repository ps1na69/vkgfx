#version 450
// shadow_depth.vert — Depth-only pass for cascade shadow map rendering.
// Outputs only gl_Position; no fragment shader output needed.

layout(location = 0) in vec3 inPosition;
// Other vertex attributes present but not used (stride must match Vertex struct)
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec4 inTangent;

// Push constant: model matrix + cascade index
layout(push_constant) uniform PC {
    mat4 model;
    uint cascadeIdx;
};

// Shadow UBO provided via set 0
layout(set = 0, binding = 0) uniform ShadowUBO {
    mat4  lightSpaceMatrix[4];
    float splitDepths[4];
} shadow;

void main() {
    gl_Position = shadow.lightSpaceMatrix[cascadeIdx] * model * vec4(inPosition, 1.0);
}
