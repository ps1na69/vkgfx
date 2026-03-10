#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view; mat4 proj; mat4 viewProj;
    vec4 position; vec4 params;
} camera;

layout(push_constant) uniform Push {
    mat4 model;
    mat4 normalMatrix;
} push;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec3 fragNormal;

void main() {
    fragUV     = inUV;
    fragNormal = normalize(mat3(push.normalMatrix) * inNormal);
    gl_Position = camera.viewProj * push.model * vec4(inPosition, 1.0);
}
