#version 450

// Fullscreen triangle — no vertex buffer required.
// gl_VertexIndex:  0 → (-1,-1)   1 → (3,-1)   2 → (-1,3)
// The triangle covers the entire viewport in one draw call.

layout(location = 0) out vec2 vUV;

void main() {
    vUV         = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(vUV * 2.0 - 1.0, 0.0, 1.0);
}
