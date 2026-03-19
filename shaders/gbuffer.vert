#version 450
// shaders/gbuffer.vert
// G-buffer fill — vertex shader.
// Locations must match Vertex struct in types.h exactly.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;   // w = bitangent sign
layout(location = 3) in vec2 inUV;

layout(push_constant) uniform Push {
    mat4 model;
    mat4 normalMatrix;
} push;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4  view;
    mat4  proj;
    mat4  viewProj;
    mat4  invViewProj;
    mat4  lightViewProj;
    vec4  cameraPos;
    vec2  viewport;
    float time;
    float _pad0;
} scene;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec4 outTangent;
layout(location = 3) out vec2 outUV;
layout(location = 4) out vec4 outLightSpacePos;  // for shadow mapping

void main() {
    vec4 worldPos     = push.model * vec4(inPosition, 1.0);
    outWorldPos       = worldPos.xyz;
    outNormal         = normalize(mat3(push.normalMatrix) * inNormal);
    outTangent        = vec4(normalize(mat3(push.model) * inTangent.xyz), inTangent.w);
    outUV             = inUV;
    outLightSpacePos  = scene.lightViewProj * worldPos;
    gl_Position       = scene.viewProj * worldPos;
}
