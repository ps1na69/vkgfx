#version 450
// geometry.vert
// Geometry pass vertex shader.
// Transforms mesh vertices and forwards world-space position, normal, and
// tangent-space basis to the fragment shader for G-buffer fill.
//
// Modifications vs. the old forward shader:
//   - Removed per-vertex lighting calculations (moved entirely to lighting pass).
//   - Added TBN matrix outputs for normal mapping.
//   - Added world-space position output (needed by lighting pass for accurate
//     shadow and parallax calculations).

// ─── Vertex inputs ───────────────────────────────────────────────────────────
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec4 inTangent;   // xyz = tangent, w = bitangent sign

// ─── Per-object push constant ────────────────────────────────────────────────
// Using a push constant keeps per-object data small and avoids binding
// a new descriptor set for every draw call.
layout(push_constant) uniform ObjectPC {
    mat4 model;
    mat4 normalMatrix;  // = transpose(inverse(model)) – pre-computed on CPU
} pc;

// ─── Frame-level UBO (set 0, binding 0) ─────────────────────────────────────
layout(set = 0, binding = 0) uniform FrameUBO {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 invView;
    mat4 invProj;
    vec4 cameraPos;
    float time;
} frame;

// ─── Outputs to fragment shader ──────────────────────────────────────────────
layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec2 outTexCoord;
layout(location = 2) out vec3 outT;       // Tangent (world space)
layout(location = 3) out vec3 outB;       // Bitangent (world space)
layout(location = 4) out vec3 outN;       // Normal (world space, pre-normal-map)

void main()
{
    vec4 worldPos = pc.model * vec4(inPosition, 1.0);
    outWorldPos   = worldPos.xyz;
    outTexCoord   = inTexCoord;

    // Build TBN basis in world space using the normal matrix to handle
    // non-uniform scaling correctly.
    vec3 N = normalize(mat3(pc.normalMatrix) * inNormal);
    vec3 T = normalize(mat3(pc.normalMatrix) * inTangent.xyz);
    T = normalize(T - dot(T, N) * N);            // Re-orthogonalise (Gram-Schmidt)
    vec3 B = cross(N, T) * inTangent.w;          // inTangent.w flips bitangent if needed

    outN = N;
    outT = T;
    outB = B;

    gl_Position = frame.viewProj * worldPos;
}
