#version 450
// fullscreen_quad.vert
// Generates a fullscreen triangle from vertex ID alone – no vertex buffer needed.
// Used by the lighting, SSAO, blur, and tone-map passes.
//
// Technique: a single oversized triangle (covering the NDC [-1,+1] square) is
// drawn by gl_VertexIndex 0..2.  The rasteriser clips it to the viewport,
// giving exactly one fragment per pixel.  This avoids two-triangle seam
// artefacts and requires zero CPU-side vertex data.

layout(location = 0) out vec2 outUV;

void main()
{
    // Map vertex index 0,1,2 to UV (0,0), (2,0), (0,2).
    outUV = vec2((gl_VertexIndex << 1) & 2,
                  gl_VertexIndex & 2);

    // NDC positions that cover the entire [-1,+1] square.
    gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
}
