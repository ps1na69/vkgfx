#version 450
// shaders/lighting.vert
// Fullscreen triangle — no vertex buffer needed.
// gl_VertexIndex 0,1,2 → covers the whole screen.

layout(location = 0) out vec2 outUV;

void main() {
    // Generate a fullscreen triangle using vertex index
    outUV       = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
}
