#version 450
// shaders/point_shadow.vert
// Depth-only pass for a single cube-map face.
//
// Push constants (128 bytes, vertex stage):
//   model    mat4  (bytes  0..63)  — object-to-world transform
//   faceVP   mat4  (bytes 64..127) — perspective view-proj for the current face
//
// Outputs world-space fragment position for the fragment shader which writes
// linear depth as  gl_FragDepth = length(fragPos - lightPos) / farPlane.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;   // declared but unused — same binding as G-buffer
layout(location = 2) in vec4 inTangent;  // idem
layout(location = 3) in vec2 inUV;       // idem

layout(push_constant) uniform PointShadowPush {
    mat4 model;
    mat4 faceVP;
} push;

layout(location = 0) out vec3 outFragPos; // world-space position

void main() {
    vec4 worldPos   = push.model * vec4(inPosition, 1.0);
    outFragPos      = worldPos.xyz;
    gl_Position     = push.faceVP * worldPos;
}
