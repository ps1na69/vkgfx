#version 450
// shaders/gbuffer.vert
// G-buffer fill pass — vertex shader.
// Attribute locations MUST match Vertex struct in types.h:
//   0 = position, 1 = normal, 2 = tangent, 3 = uv

layout(location = 0) in vec3 inPosition;   // Vertex::position
layout(location = 1) in vec3 inNormal;     // Vertex::normal
layout(location = 2) in vec4 inTangent;    // Vertex::tangent (w = bitangent sign)
layout(location = 3) in vec2 inUV;         // Vertex::uv

// Push constants — MeshPush from types.h
layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 normalMatrix;
} push;

// Scene UBO — set=0, binding=0
layout(set = 0, binding = 0) uniform SceneUBO {
    mat4  view;
    mat4  proj;
    mat4  viewProj;
    vec4  cameraPos;
    vec2  viewport;
    float time;
    float _pad0;
} scene;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec4 outTangent;
layout(location = 3) out vec2 outUV;

void main() {
    vec4 worldPos   = push.model * vec4(inPosition, 1.0);
    outWorldPos     = worldPos.xyz;
    outNormal       = normalize(mat3(push.normalMatrix) * inNormal);
    outTangent      = vec4(normalize(mat3(push.model) * inTangent.xyz), inTangent.w);
    outUV           = inUV;
    gl_Position     = scene.viewProj * worldPos;
}
