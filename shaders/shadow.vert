#version 450

layout(location = 0) in vec3 inPosition;

layout(push_constant) uniform PC {
    mat4 lightSpace;
    mat4 model;
} pc;

void main() {
    gl_Position = pc.lightSpace * pc.model * vec4(inPosition, 1.0);
}
