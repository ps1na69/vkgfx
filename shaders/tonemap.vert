#version 450
// shaders/tonemap.vert
// Fullscreen triangle — shared pattern with lighting.vert.
// No vertex buffer needed.

layout(location = 0) out vec2 outUV;

void main() {
    outUV       = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
}
