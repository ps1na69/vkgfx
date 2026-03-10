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

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;
layout(location = 3) out vec3 fragCamPos;

void main() {
    vec4 worldPos = push.model * vec4(inPosition, 1.0);
    fragWorldPos  = worldPos.xyz;
    fragNormal    = normalize(mat3(push.normalMatrix) * inNormal);
    fragUV        = inUV;
    fragCamPos    = camera.position.xyz;
    gl_Position   = camera.viewProj * worldPos;
}
