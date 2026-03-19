#version 450
// shaders/shadow.vert
// Depth-only shadow map pass.
// Reuses MeshPush (128 bytes): .model = model matrix, .normalMatrix = lightViewProj.

layout(location = 0) in vec3 inPosition;
// Other attributes declared but not used — same binding as G-buffer pass
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inUV;

layout(push_constant) uniform ShadowPush {
    mat4 model;
    mat4 lightViewProj;  // stored in the normalMatrix slot of MeshPush
} push;

void main() {
    gl_Position = push.lightViewProj * push.model * vec4(inPosition, 1.0);
}
