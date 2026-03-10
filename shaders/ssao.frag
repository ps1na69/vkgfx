#version 450
// ssao.frag
// Screen-Space Ambient Occlusion.
//
// For each pixel, SAMPLE_COUNT points are tested in a hemisphere above the
// surface.  A point is "occluded" if it lies inside geometry (its depth is
// greater than the value stored in the depth buffer at its screen position).
// The fraction of occluded samples is the raw AO factor written to the output.
//
// Noise: a 4x4 tiled random rotation texture randomises the sample kernel
// per pixel to break up banding.  A blur pass smooths the resulting noise.

layout(location = 0) in  vec2 inUV;
layout(location = 0) out float outAO;

// G-buffer inputs
layout(set = 0, binding = 0) uniform sampler2D gNormal;
layout(set = 0, binding = 1) uniform sampler2D gDepth;
layout(set = 0, binding = 2) uniform sampler2D noiseTex;   // 4x4 tiled rotation vectors

layout(set = 0, binding = 3) uniform SSAOParams {
    mat4  proj;
    mat4  invProj;
    vec4  samples[32];   // Hemisphere sample offsets in view space
    vec2  noiseScale;    // screen_res / 4  (tiles the noise texture)
    float radius;
    float bias;
} params;

const int   SAMPLE_COUNT = 32;
const float PI           = 3.14159265359;

// Reconstruct view-space position from depth and UV.
vec3 view_pos_from_depth(vec2 uv, float depth)
{
    // Convert UV and depth to NDC.
    vec4 ndc = vec4(uv * 2.0 - 1.0, depth, 1.0);
    // Unproject to view space using the inverse projection matrix.
    vec4 view = params.invProj * ndc;
    return view.xyz / view.w;   // Perspective divide
}

void main()
{
    // ── Read surface data ─────────────────────────────────────────────────────
    float depth = texture(gDepth,  inUV).r;

    // Skip skybox / background pixels (depth == 1.0 or 0.0 depending on convention).
    if (depth >= 0.9999) {
        outAO = 1.0;  // No occlusion on empty pixels
        return;
    }

    vec3 fragPosV  = view_pos_from_depth(inUV, depth);

    // Normal from G-buffer is world-space; transform to view space for the
    // hemisphere test (which is performed in view space to reuse the projection
    // matrix for sample depth comparisons).
    vec3 normalW   = normalize(texture(gNormal, inUV).rgb);
    // View-space normal: remove the translation component of the view matrix.
    // We only need mat3(view) but we don't have view in this pass; the calling
    // code pre-transforms the normal to view space and stores it in gNormal.
    // (Alternatively, pass an additional view-space normal attachment.)
    // For now we treat gNormal as already view-space (geometry.frag can be
    // switched to output view-space normals by multiplying by mat3(view)).
    vec3 normalV   = normalW;   // Assumed view-space (see geometry.frag)

    // ── TBN random rotation ───────────────────────────────────────────────────
    // The noise texture tiles across the screen with noiseScale, giving each
    // pixel a different random rotation axis.  This breaks up banding.
    vec2 noiseUV   = inUV * params.noiseScale;
    vec3 randomVec = normalize(vec3(texture(noiseTex, noiseUV).rg * 2.0 - 1.0, 0.0));

    // Build a local TBN basis aligned with the surface normal.
    vec3 tangent   = normalize(randomVec - normalV * dot(randomVec, normalV));
    vec3 bitangent = cross(normalV, tangent);
    mat3 TBN       = mat3(tangent, bitangent, normalV);

    // ── Sample hemisphere ─────────────────────────────────────────────────────
    float occlusion = 0.0;
    for (int i = 0; i < SAMPLE_COUNT; ++i)
    {
        // Rotate the pre-computed kernel sample by the TBN, then scale by radius.
        vec3 samplePosV = TBN * params.samples[i].xyz;
        samplePosV = fragPosV + samplePosV * params.radius;

        // Project the sample to screen space to look up its depth.
        vec4 sampleNDC  = params.proj * vec4(samplePosV, 1.0);
        sampleNDC.xyz  /= sampleNDC.w;            // Perspective divide
        vec2 sampleUV   = sampleNDC.xy * 0.5 + 0.5;

        // Depth of the geometry at the sample's screen position.
        float sampleDepth = texture(gDepth, sampleUV).r;
        vec3  sampleGeomV = view_pos_from_depth(sampleUV, sampleDepth);

        // A sample is occluded if the geometry at that screen position is
        // closer to the camera than the sample point.  A small bias prevents
        // self-shadowing on flat surfaces.
        float occluded = (sampleGeomV.z >= samplePosV.z + params.bias) ? 1.0 : 0.0;

        // Range check: samples far from the original fragment don't count.
        // smoothstep ramps off gently to avoid a hard occlusion edge at radius.
        float rangeCheck = smoothstep(0.0, 1.0,
                                      params.radius / abs(fragPosV.z - sampleGeomV.z));
        occlusion += occluded * rangeCheck;
    }

    // Normalise and invert so 1 = unoccluded, 0 = fully occluded.
    outAO = 1.0 - (occlusion / float(SAMPLE_COUNT));
}
