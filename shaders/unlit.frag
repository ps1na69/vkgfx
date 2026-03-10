#version 450

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec3 fragNormal;

layout(set = 1, binding = 0) uniform MaterialUBO {
    vec4 color;
    int  useTexture;
    int  _pad[3];
} mat;

layout(set = 1, binding = 1) uniform sampler2D colorMap;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = (mat.useTexture == 1) ? texture(colorMap, fragUV) : mat.color;
}
