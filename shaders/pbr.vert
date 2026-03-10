#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inTangent;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    vec4 position;  // xyz = world-space camera pos, w = near plane
    vec4 params;    // x=far, y=fov, z=aspect, w=time
} camera;

layout(push_constant) uniform Push {
    mat4 model;
    mat4 normalMatrix;  // transpose(inverse(model)) — correct for non-uniform scale
} push;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;
layout(location = 3) out vec3 fragTangent;
layout(location = 4) out vec3 fragBitangent;
layout(location = 5) out vec3 fragCamPos;

void main() {
    vec4 worldPos  = push.model * vec4(inPosition, 1.0);
    fragWorldPos   = worldPos.xyz;

    // Transform normal with the normal matrix (handles non-uniform scale correctly)
    vec3 N = normalize(mat3(push.normalMatrix) * inNormal);
    // Transform tangent with the model matrix (no scale correction needed for tangent)
    vec3 T = normalize(mat3(push.model) * inTangent);
    // Gram-Schmidt re-orthogonalise T against N (fixes precision drift)
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T);

    fragNormal    = N;
    fragTangent   = T;
    fragBitangent = B;
    fragUV        = inUV;
    fragCamPos    = camera.position.xyz;

    gl_Position = camera.viewProj * worldPos;
}
