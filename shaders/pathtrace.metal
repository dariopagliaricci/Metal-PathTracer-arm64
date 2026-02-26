using namespace metal;

#if __METAL_VERSION__ >= 310
#include <metal_raytracing>
using namespace metal::raytracing;
#endif

#ifndef PT_DEBUG_TOOLS
#define PT_DEBUG_TOOLS 0
#endif

#ifndef PT_MNEE_SWRT_RAYS
#define PT_MNEE_SWRT_RAYS 0
#endif
#ifndef PT_MNEE_OCCLUSION_PARITY
#define PT_MNEE_OCCLUSION_PARITY 0
#endif

constant float kInfinity = 1e20f;
constant float kPi = 3.14159265358979323846f;
constexpr sampler environmentSampler(filter::linear,
                                     mip_filter::linear,
                                     address::repeat,
                                     coord::normalized);
constexpr sampler environmentSamplerNearest(filter::nearest,
                                            mip_filter::nearest,
                                            address::repeat,
                                            coord::normalized);
constant float kEpsilon = 1e-3f;
constant float kRayOriginEpsilon = 1e-4f;
constant float kSssThroughputCutoff = 1e-3f;
constant float3 kLuminanceWeights = float3(0.2126f, 0.7152f, 0.0722f);
constant uint kInvalidIndex = 0xffffffffu;
constant uint kWorkingColorSpaceLinearSRGB = 0u;
constant uint kWorkingColorSpaceACEScg = 1u;
constant uint kBvhTraversalStackSize = 128u;
constant float kHardwareOcclusionEpsilon = 5.0e-3f;
constant float kSpecularNeePdfFloor = 1.0e-4f;
constant float kSpecularNeeInvPdfClamp = 1.0e4f;
constant float kMisWeightClampMin = 1.0e-4f;
constant float kMisWeightClampMax = 0.9999f;
constant uint kParityReasonHitMiss = 1u;
constant uint kParityReasonT = 2u;
constant uint kParityReasonNormal = 4u;
constant uint kParityReasonId = 8u;
constant uint kParityReasonFrontFace = 16u;
constant uint kParityModeProbePixel = 1u;
constant uint kParityModeFirstInMedium = 2u;
constant uint kDebugViewNone = 0u;
constant uint kDebugViewBaseColor = 1u;
constant uint kDebugViewMetallic = 2u;
constant uint kDebugViewRoughness = 3u;
constant uint kDebugViewAO = 4u;

inline uint pcg_hash(uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline float rand_uniform(thread uint& state) {
    state = pcg_hash(state);
    return float(state) / 4294967296.0f;
}

inline float3 random_in_unit_sphere(thread uint& state) {
    while (true) {
        float3 p = float3(rand_uniform(state), rand_uniform(state), rand_uniform(state)) * 2.0f - 1.0f;
        if (dot(p, p) < 1.0f) {
            return p;
        }
    }
}

inline float3 random_unit_vector(thread uint& state) {
    return normalize(random_in_unit_sphere(state));
}

inline float2 random_in_unit_disk(thread uint& state) {
    while (true) {
        float2 p = float2(rand_uniform(state), rand_uniform(state)) * 2.0f - 1.0f;
        if (dot(p, p) < 1.0f) {
            return p;
        }
    }
}

inline bool near_zero(const float3 v) {
    const float s = 1e-6f;
    return (fabs(v.x) < s) && (fabs(v.y) < s) && (fabs(v.z) < s);
}

inline float3 linear_srgb_to_acescg(const float3 color) {
    const float3x3 m = float3x3(float3(0.613097f, 0.339523f, 0.047380f),
                                float3(0.070194f, 0.916354f, 0.013452f),
                                float3(0.020615f, 0.109569f, 0.869816f));
    return m * color;
}

inline float3 to_working_space(const float3 color,
                               constant PathtraceUniforms& uniforms) {
    if (uniforms.workingColorSpace == kWorkingColorSpaceACEScg) {
        return linear_srgb_to_acescg(color);
    }
    return color;
}

inline float3 decode_normal_map(float3 sample,
                                float normalScale,
                                bool flipGreen,
                                thread float& outLength) {
    float3 n = sample * 2.0f - 1.0f;
    if (flipGreen) {
        n.y = -n.y;
    }
    n.xy *= normalScale;
    outLength = length(n);
    float xyLen2 = dot(n.xy, n.xy);
    n.z = sqrt(max(1.0f - xyLen2, 0.0f));
    float len2 = dot(n, n);
    if (len2 > 1.0e-12f) {
        n *= rsqrt(len2);
    } else {
        n = float3(0.0f, 0.0f, 1.0f);
    }
    return n;
}

struct RayCone {
    float width;
    float spread;
};

struct PrimaryRayDiff {
    float3 dOdx;
    float3 dOdy;
    float3 dDdx;
    float3 dDdy;
};

inline RayCone make_primary_ray_cone(constant PathtraceUniforms& uniforms) {
    float pixelWorldX = length(uniforms.horizontal) / max(float(uniforms.width), 1.0f);
    float pixelWorldY = length(uniforms.vertical) / max(float(uniforms.height), 1.0f);
    float pixelFootprint = max(max(pixelWorldX, pixelWorldY), 1.0e-6f);
    float3 viewportCenter =
        uniforms.lowerLeftCorner + 0.5f * uniforms.horizontal + 0.5f * uniforms.vertical;
    float focusDistance = length(viewportCenter - uniforms.cameraOrigin);
    RayCone cone;
    cone.width = max(2.0f * uniforms.lensRadius, 0.0f);
    cone.spread = pixelFootprint / max(focusDistance, 1.0e-6f);
    return cone;
}

inline float ray_segment_world_length(const Ray ray, float t) {
    return max(t, 0.0f) * max(length(ray.direction), 1.0e-6f);
}

inline float ray_cone_width_at_distance(const RayCone cone, float distanceWorld) {
    return max(cone.width + cone.spread * max(distanceWorld, 0.0f), 1.0e-7f);
}

inline float ray_cone_lod_from_footprint(const MaterialTextureInfo textureInfo,
                                         float uvPerWorld,
                                         float footprintWorld) {
    if (textureInfo.width == 0u || textureInfo.height == 0u) {
        return 0.0f;
    }
    if (textureInfo.mipCount <= 1u || uvPerWorld <= 0.0f || footprintWorld <= 0.0f) {
        return 0.0f;
    }
    float maxRes = max(float(textureInfo.width), float(textureInfo.height));
    float texelFootprint = footprintWorld * uvPerWorld * maxRes;
    float lod = log2(max(texelFootprint, 1.0e-7f));
    float maxMip = float(textureInfo.mipCount - 1u);
    return clamp(lod, 0.0f, maxMip);
}

inline float surface_footprint_from_cone(float coneFootprintWorld,
                                         const float3 surfaceNormal,
                                         const float3 viewDir) {
    float3 n = normalize(surfaceNormal);
    float3 v = normalize(viewDir);
    float cosTheta = fabs(dot(n, v));
    return coneFootprintWorld / max(cosTheta, 1.0e-3f);
}

inline bool uv_world_gradients_from_partials(const float3 dPdu,
                                             const float3 dPdv,
                                             thread float3& dudP,
                                             thread float3& dvdP) {
    float a00 = dot(dPdu, dPdu);
    float a01 = dot(dPdu, dPdv);
    float a11 = dot(dPdv, dPdv);
    float det = a00 * a11 - a01 * a01;
    if (fabs(det) <= 1.0e-12f) {
        return false;
    }
    dudP = (a11 * dPdu - a01 * dPdv) / det;
    dvdP = (a00 * dPdv - a01 * dPdu) / det;
    return all(isfinite(dudP)) && all(isfinite(dvdP));
}

inline bool first_hit_uv_gradients_igehy(const Ray ray,
                                         const PrimaryRayDiff rayDiff,
                                         float tHit,
                                         const float3 geometricNormal,
                                         const float3 dudP,
                                         const float3 dvdP,
                                         thread float2& dUVdx,
                                         thread float2& dUVdy) {
    dUVdx = float2(0.0f);
    dUVdy = float2(0.0f);
    float3 N = normalize(geometricNormal);
    if (!all(isfinite(N)) || dot(N, N) <= 1.0e-12f) {
        return false;
    }
    float3 D = ray.direction;
    float denom = dot(N, D);
    if (!isfinite(denom) || fabs(denom) < 1.0e-6f) {
        return false;
    }
    float3 dtdxTerm = rayDiff.dOdx + tHit * rayDiff.dDdx;
    float3 dtdyTerm = rayDiff.dOdy + tHit * rayDiff.dDdy;
    if (!all(isfinite(dtdxTerm)) || !all(isfinite(dtdyTerm))) {
        return false;
    }
    float dtdx = -dot(N, dtdxTerm) / denom;
    float dtdy = -dot(N, dtdyTerm) / denom;
    if (!isfinite(dtdx) || !isfinite(dtdy)) {
        return false;
    }
    float3 dPdx = dtdxTerm + dtdx * D;
    float3 dPdy = dtdyTerm + dtdy * D;
    if (!all(isfinite(dPdx)) || !all(isfinite(dPdy))) {
        return false;
    }
    dUVdx = float2(dot(dudP, dPdx), dot(dvdP, dPdx));
    dUVdy = float2(dot(dudP, dPdy), dot(dvdP, dPdy));
    return all(isfinite(dUVdx)) && all(isfinite(dUVdy));
}

struct HitRecord {
    float3 point;
    float3 normal;
    float3 shadingNormal;
    float paddingNormal;
    float t;
    uint materialIndex;
    uint twoSided;
    uint frontFace;
    uint primitiveType;
    uint primitiveIndex;
    uint meshIndex;
    float2 barycentric;
    uint padding;
};

inline void compute_exclusion_indices(thread const HitRecord& rec,
                                      thread uint& meshIndexOut,
                                      thread uint& primitiveIndexOut) {
    if (rec.primitiveType == kPrimitiveTypeTriangle) {
        meshIndexOut = rec.meshIndex;
        primitiveIndexOut = rec.primitiveIndex;
    } else {
        meshIndexOut = kInvalidIndex;
        primitiveIndexOut = kInvalidIndex;
    }
}

struct TraversalCounters;

struct PathtraceDebugContext {
    device PathtraceDebugBuffer* buffer;
    uint integrator;
    uint pixelX;
    uint pixelY;
    uint sampleIndex;
    uint maxEntries;
    uint active;
};

#if PT_DEBUG_TOOLS
inline PathtraceDebugContext make_debug_context(constant PathtraceUniforms& uniforms,
                                                device PathtraceDebugBuffer* buffer,
                                                uint2 gid,
                                                uint sampleIndex,
                                                uint integrator) {
    PathtraceDebugContext ctx;
    ctx.buffer = buffer;
    ctx.integrator = integrator;
    ctx.pixelX = gid.x;
    ctx.pixelY = gid.y;
    ctx.sampleIndex = sampleIndex;
    ctx.maxEntries = (buffer != nullptr)
        ? min(uniforms.debugMaxEntries, kPathtraceDebugMaxEntries)
        : 0u;
    bool active = (buffer != nullptr) &&
                  (uniforms.debugPathActive != 0u) &&
                  (ctx.maxEntries > 0u) &&
                  (gid.x == uniforms.debugPixelX) &&
                  (gid.y == uniforms.debugPixelY);
    ctx.active = active ? 1u : 0u;
    return ctx;
}

inline void record_debug_event(thread const PathtraceDebugContext& ctx,
                               uint depth,
                               uint mediumDepthBefore,
                               uint mediumDepthAfter,
                               int mediumEvent,
                               uint frontFace,
                               uint materialIndex,
                               bool scatterIsDelta,
                               float3 throughput) {
    if (!ctx.active || ctx.buffer == nullptr) {
        return;
    }
    uint allowed = min(ctx.maxEntries, ctx.buffer->maxEntries);
    if (allowed == 0u) {
        return;
    }
    uint writeIndex = atomic_fetch_add_explicit(&ctx.buffer->writeIndex, 1u, memory_order_relaxed);
    if (writeIndex >= allowed) {
        return;
    }

    PathtraceDebugEntry entry;
    entry.integrator = ctx.integrator;
    entry.pixelX = ctx.pixelX;
    entry.pixelY = ctx.pixelY;
    entry.sampleIndex = ctx.sampleIndex;
    entry.depth = depth;
    entry.mediumDepthBefore = mediumDepthBefore;
    entry.mediumDepthAfter = mediumDepthAfter;
    entry.mediumEvent = mediumEvent;
    entry.frontFace = frontFace;
    entry.scatterIsDelta = scatterIsDelta ? 1u : 0u;
    entry.materialIndex = materialIndex;
    entry.reserved0 = 0u;
    entry.throughput = float4(throughput, 0.0f);
    ctx.buffer->entries[writeIndex] = entry;
}

inline HitRecord make_empty_hit_record() {
    HitRecord rec;
    rec.point = float3(0.0f);
    rec.normal = float3(0.0f);
    rec.shadingNormal = float3(0.0f);
    rec.paddingNormal = 0.0f;
    rec.t = 0.0f;
    rec.materialIndex = 0u;
    rec.twoSided = 0u;
    rec.frontFace = 0u;
    rec.primitiveType = kPrimitiveTypeNone;
    rec.primitiveIndex = kInvalidIndex;
    rec.meshIndex = kInvalidIndex;
    rec.barycentric = float2(0.0f);
    rec.padding = 0u;
    return rec;
}

inline void record_parity_entry(thread const PathtraceDebugContext& ctx,
                                constant PathtraceUniforms& uniforms,
                                uint depth,
                                thread const Ray& ray,
                                float tMin,
                                float tMax,
                                bool hwHit,
                                thread const HitRecord& hwRec,
                                bool swHit,
                                thread const HitRecord& swRec,
                                uint reasonMask) {
    if (reasonMask == 0u || ctx.buffer == nullptr) {
        return;
    }
    uint allowed = min(ctx.buffer->parityMaxEntries, kPathtraceParityMaxEntries);
    if (allowed == 0u) {
        return;
    }
    uint writeIndex =
        atomic_fetch_add_explicit(&ctx.buffer->parityWriteIndex, 1u, memory_order_relaxed);
    if (writeIndex >= allowed) {
        return;
    }

    PathtraceParityEntry entry;
    entry.frameIndex = uniforms.frameIndex;
    entry.pixelX = ctx.pixelX;
    entry.pixelY = ctx.pixelY;
    entry.depth = depth;
    entry.reasonMask = reasonMask;
    entry.hwHit = hwHit ? 1u : 0u;
    entry.swHit = swHit ? 1u : 0u;
    entry.hwFrontFace = hwRec.frontFace;
    entry.swFrontFace = swRec.frontFace;
    entry.hwMaterialIndex = hwRec.materialIndex;
    entry.swMaterialIndex = swRec.materialIndex;
    entry.hwMeshIndex = hwRec.meshIndex;
    entry.swMeshIndex = swRec.meshIndex;
    entry.hwPrimitiveIndex = hwRec.primitiveIndex;
    entry.swPrimitiveIndex = swRec.primitiveIndex;
    entry.hwT = hwRec.t;
    entry.swT = swRec.t;
    entry.tMin = tMin;
    entry.tMax = tMax;
    entry.rayOrigin = float4(ray.origin, 0.0f);
    entry.rayDirection = float4(ray.direction, 0.0f);
    entry.hwNormal = float4(hwRec.normal, 0.0f);
    entry.swNormal = float4(swRec.normal, 0.0f);
    ctx.buffer->parityEntries[writeIndex] = entry;
}
#else
inline PathtraceDebugContext make_debug_context(constant PathtraceUniforms& uniforms,
                                                device PathtraceDebugBuffer* buffer,
                                                uint2 gid,
                                                uint sampleIndex,
                                                uint integrator) {
    (void)uniforms;
    PathtraceDebugContext ctx;
    ctx.buffer = nullptr;
    ctx.integrator = integrator;
    ctx.pixelX = gid.x;
    ctx.pixelY = gid.y;
    ctx.sampleIndex = sampleIndex;
    ctx.maxEntries = 0u;
    ctx.active = 0u;
    return ctx;
}

inline void record_debug_event(thread const PathtraceDebugContext& ctx,
                               uint depth,
                               uint mediumDepthBefore,
                               uint mediumDepthAfter,
                               int mediumEvent,
                               uint frontFace,
                               uint materialIndex,
                               bool scatterIsDelta,
                               float3 throughput) {
    (void)ctx;
    (void)depth;
    (void)mediumDepthBefore;
    (void)mediumDepthAfter;
    (void)mediumEvent;
    (void)frontFace;
    (void)materialIndex;
    (void)scatterIsDelta;
    (void)throughput;
}

inline HitRecord make_empty_hit_record() {
    HitRecord rec;
    rec.point = float3(0.0f);
    rec.normal = float3(0.0f);
    rec.shadingNormal = float3(0.0f);
    rec.paddingNormal = 0.0f;
    rec.t = 0.0f;
    rec.materialIndex = 0u;
    rec.twoSided = 0u;
    rec.frontFace = 0u;
    rec.primitiveType = kPrimitiveTypeNone;
    rec.primitiveIndex = kInvalidIndex;
    rec.meshIndex = kInvalidIndex;
    rec.barycentric = float2(0.0f);
    rec.padding = 0u;
    return rec;
}

inline void record_parity_entry(thread const PathtraceDebugContext& ctx,
                                constant PathtraceUniforms& uniforms,
                                uint depth,
                                thread const Ray& ray,
                                float tMin,
                                float tMax,
                                bool hwHit,
                                thread const HitRecord& hwRec,
                                bool swHit,
                                thread const HitRecord& swRec,
                                uint reasonMask) {
    (void)ctx;
    (void)uniforms;
    (void)depth;
    (void)ray;
    (void)tMin;
    (void)tMax;
    (void)hwHit;
    (void)hwRec;
    (void)swHit;
    (void)swRec;
    (void)reasonMask;
}
#endif

inline float3 environment_color(texture2d<float, access::sample> environmentTexture,
                                const float3 direction,
                                float rotation,
                                float intensity,
                                constant PathtraceUniforms& uniforms);
inline float3 environment_color_lod(texture2d<float, access::sample> environmentTexture,
                                    const float3 direction,
                                    float rotation,
                                    float intensity,
                                    float lod,
                                    constant PathtraceUniforms& uniforms);

inline uint mesh_index_from_triangle(uint triIndex,
                                     device const MeshInfo* meshInfos,
                                     uint meshCount) {
    if (!meshInfos || meshCount == 0u) {
        return 0u;
    }
    for (uint i = 0; i < meshCount; ++i) {
        MeshInfo info = meshInfos[i];
        uint start = info.triangleOffset;
        uint end = start + info.triangleCount;
        if (triIndex >= start && triIndex < end) {
            return i;
        }
    }
    return min(triIndex, meshCount - 1u);
}

inline float2 barycentric_from_point(const float3 v0,
                                     const float3 v1,
                                     const float3 v2,
                                     const float3 p) {
    float3 v0v1 = v1 - v0;
    float3 v0v2 = v2 - v0;
    float3 v0p = p - v0;
    float d00 = dot(v0v1, v0v1);
    float d01 = dot(v0v1, v0v2);
    float d11 = dot(v0v2, v0v2);
    float d20 = dot(v0p, v0v1);
    float d21 = dot(v0p, v0v2);
    float denom = d00 * d11 - d01 * d01;
    if (fabs(denom) < 1.0e-8f) {
        return float2(0.0f, 0.0f);
    }
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    // Match Moller-Trumbore convention used by hit_triangle: x = weight for v1, y = weight for v2.
    return float2(v, w);
}

inline bool intersect_triangle_parametric(const float3 v0,
                                          const float3 v1,
                                          const float3 v2,
                                          thread const Ray& ray,
                                          float tMin,
                                          float tMax,
                                          thread float& tOut,
                                          thread float2& barycentricOut) {
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 pVec = cross(ray.direction, edge2);
    float det = dot(edge1, pVec);
    if (fabs(det) < 1.0e-8f) {
        return false;
    }

    float invDet = 1.0f / det;
    float3 tVec = ray.origin - v0;
    float u = dot(tVec, pVec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    float3 qVec = cross(tVec, edge1);
    float v = dot(ray.direction, qVec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) {
        return false;
    }

    float t = dot(edge2, qVec) * invDet;
    if (t < tMin || t > tMax) {
        return false;
    }

    tOut = t;
    barycentricOut = float2(u, v);
    return true;
}

inline float3 barycentric_weights_saturated(const float2 barycentric) {
    float3 weights = float3(1.0f - barycentric.x - barycentric.y, barycentric.x, barycentric.y);
    weights = max(weights, float3(0.0f));
    float sum = weights.x + weights.y + weights.z;
    if (!(sum > 1.0e-8f)) {
        return float3(1.0f, 0.0f, 0.0f);
    }
    return weights / sum;
}

inline float2 vertex_uv_set(const SceneVertex inVertex, const uint uvSet) {
    return (uvSet != 0u) ? inVertex.uv.zw : inVertex.uv.xy;
}

inline float3 interpolate_shading_normal(constant PathtraceUniforms& uniforms,
                                         uint meshIndex,
                                         uint primitiveIndex,
                                         float2 barycentric,
                                         device const MeshInfo* meshInfos,
                                         device const SceneVertex* vertices,
                                         device const uint3* meshIndices) {
    if (!meshInfos || !vertices || !meshIndices || uniforms.meshCount == 0u) {
        return float3(0.0f);
    }
    uint clampedMesh = min(meshIndex, uniforms.meshCount - 1u);
    MeshInfo info = meshInfos[clampedMesh];
    if (info.indexCount == 0u || info.triangleCount == 0u) {
        return float3(0.0f);
    }
    if (primitiveIndex < info.triangleOffset) {
        return float3(0.0f);
    }
    uint localIndex = primitiveIndex - info.triangleOffset;
    if (localIndex >= info.indexCount) {
        return float3(0.0f);
    }
    uint indexEntry = info.indexOffset + localIndex;
    uint3 triIndices = meshIndices[indexEntry];
    SceneVertex v0 = vertices[triIndices.x];
    SceneVertex v1 = vertices[triIndices.y];
    SceneVertex v2 = vertices[triIndices.z];
    float3 weights = barycentric_weights_saturated(barycentric);
    float3 n0 = v0.normal.xyz;
    float3 n1 = v1.normal.xyz;
    float3 n2 = v2.normal.xyz;
    float3 nLocal = n0 * weights.x + n1 * weights.y + n2 * weights.z;
    if (!all(isfinite(nLocal))) {
        return float3(0.0f);
    }
    float3x3 worldToLocal = float3x3(info.worldToLocal[0].xyz,
                                     info.worldToLocal[1].xyz,
                                     info.worldToLocal[2].xyz);
    float3x3 normalMatrix = transpose(worldToLocal);
    float3 shadingNormal = normalize(normalMatrix * nLocal);
    return shadingNormal;
}

inline float2 interpolate_uv(constant PathtraceUniforms& uniforms,
                             uint meshIndex,
                             uint primitiveIndex,
                             float2 barycentric,
                             const uint uvSet,
                             device const MeshInfo* meshInfos,
                             device const SceneVertex* vertices,
                             device const uint3* meshIndices) {
    if (!meshInfos || !vertices || !meshIndices || uniforms.meshCount == 0u) {
        return float2(0.0f);
    }
    uint clampedMesh = min(meshIndex, uniforms.meshCount - 1u);
    MeshInfo info = meshInfos[clampedMesh];
    if (info.indexCount == 0u || info.triangleCount == 0u) {
        return float2(0.0f);
    }
    if (primitiveIndex < info.triangleOffset) {
        return float2(0.0f);
    }
    uint localIndex = primitiveIndex - info.triangleOffset;
    if (localIndex >= info.indexCount) {
        return float2(0.0f);
    }
    uint indexEntry = info.indexOffset + localIndex;
    uint3 triIndices = meshIndices[indexEntry];
    SceneVertex v0 = vertices[triIndices.x];
    SceneVertex v1 = vertices[triIndices.y];
    SceneVertex v2 = vertices[triIndices.z];
    float3 weights = barycentric_weights_saturated(barycentric);
    float2 uv0 = vertex_uv_set(v0, uvSet);
    float2 uv1 = vertex_uv_set(v1, uvSet);
    float2 uv2 = vertex_uv_set(v2, uvSet);
    float2 uv = uv0 * weights.x + uv1 * weights.y + uv2 * weights.z;
    return uv;
}

inline float2 interpolate_uv(constant PathtraceUniforms& uniforms,
                             uint meshIndex,
                             uint primitiveIndex,
                             float2 barycentric,
                             device const MeshInfo* meshInfos,
                             device const SceneVertex* vertices,
                             device const uint3* meshIndices) {
    return interpolate_uv(uniforms,
                          meshIndex,
                          primitiveIndex,
                          barycentric,
                          0u,
                          meshInfos,
                          vertices,
                          meshIndices);
}

inline float4 interpolate_tangent(constant PathtraceUniforms& uniforms,
                                  uint meshIndex,
                                  uint primitiveIndex,
                                  float2 barycentric,
                                  device const MeshInfo* meshInfos,
                                  device const SceneVertex* vertices,
                                  device const uint3* meshIndices) {
    if (!meshInfos || !vertices || !meshIndices || uniforms.meshCount == 0u) {
        return float4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    uint clampedMesh = min(meshIndex, uniforms.meshCount - 1u);
    MeshInfo info = meshInfos[clampedMesh];
    if (info.indexCount == 0u || info.triangleCount == 0u) {
        return float4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    if (primitiveIndex < info.triangleOffset) {
        return float4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    uint localIndex = primitiveIndex - info.triangleOffset;
    if (localIndex >= info.indexCount) {
        return float4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    uint indexEntry = info.indexOffset + localIndex;
    uint3 triIndices = meshIndices[indexEntry];
    SceneVertex v0 = vertices[triIndices.x];
    SceneVertex v1 = vertices[triIndices.y];
    SceneVertex v2 = vertices[triIndices.z];
    float3 weights = barycentric_weights_saturated(barycentric);
    float4 t0 = v0.tangent;
    float4 t1 = v1.tangent;
    float4 t2 = v2.tangent;
    float4 tangentLocal = t0 * weights.x + t1 * weights.y + t2 * weights.z;
    float3x3 localToWorld = float3x3(info.localToWorld[0].xyz,
                                     info.localToWorld[1].xyz,
                                     info.localToWorld[2].xyz);
    float3 tangentWorld = localToWorld * tangentLocal.xyz;
    float tangentLenSq = dot(tangentWorld, tangentWorld);
    if (!all(isfinite(tangentWorld)) || tangentLenSq <= 1.0e-12f) {
        tangentWorld = float3(1.0f, 0.0f, 0.0f);
    } else {
        tangentWorld *= rsqrt(tangentLenSq);
    }

    float detSign = (determinant(localToWorld) < 0.0f) ? -1.0f : 1.0f;
    float handedness = (tangentLocal.w < 0.0f) ? -1.0f : 1.0f;
    return float4(tangentWorld, handedness * detSign);
}

inline bool triangle_surface_partials(constant PathtraceUniforms& uniforms,
                                      uint meshIndex,
                                      uint primitiveIndex,
                                      const uint uvSet,
                                      device const MeshInfo* meshInfos,
                                      device const SceneVertex* vertices,
                                      device const uint3* meshIndices,
                                      thread float3& dPduOut,
                                      thread float3& dPdvOut,
                                      thread float& uvPerWorldOut) {
    dPduOut = float3(0.0f);
    dPdvOut = float3(0.0f);
    uvPerWorldOut = 0.0f;
    if (!meshInfos || !vertices || !meshIndices || uniforms.meshCount == 0u) {
        return false;
    }
    uint clampedMesh = min(meshIndex, uniforms.meshCount - 1u);
    MeshInfo info = meshInfos[clampedMesh];
    if (info.indexCount == 0u || info.triangleCount == 0u) {
        return false;
    }
    if (primitiveIndex < info.triangleOffset) {
        return false;
    }
    uint localIndex = primitiveIndex - info.triangleOffset;
    if (localIndex >= info.indexCount) {
        return false;
    }

    uint indexEntry = info.indexOffset + localIndex;
    uint3 triIndices = meshIndices[indexEntry];
    SceneVertex v0 = vertices[triIndices.x];
    SceneVertex v1 = vertices[triIndices.y];
    SceneVertex v2 = vertices[triIndices.z];

    float4 wp0 = info.localToWorld * float4(v0.position.xyz, 1.0f);
    float4 wp1 = info.localToWorld * float4(v1.position.xyz, 1.0f);
    float4 wp2 = info.localToWorld * float4(v2.position.xyz, 1.0f);
    float3 p0 = wp0.xyz;
    float3 p1 = wp1.xyz;
    float3 p2 = wp2.xyz;
    float2 uv0 = vertex_uv_set(v0, uvSet);
    float2 uv1 = vertex_uv_set(v1, uvSet);
    float2 uv2 = vertex_uv_set(v2, uvSet);

    float3 edge1 = p1 - p0;
    float3 edge2 = p2 - p0;
    float2 dUV1 = uv1 - uv0;
    float2 dUV2 = uv2 - uv0;
    float det = dUV1.x * dUV2.y - dUV1.y * dUV2.x;
    if (fabs(det) > 1.0e-9f) {
        float invDet = 1.0f / det;
        float3 dPdu = (edge1 * dUV2.y - edge2 * dUV1.y) * invDet;
        float3 dPdv = (edge2 * dUV1.x - edge1 * dUV2.x) * invDet;
        float lenU = length(dPdu);
        float lenV = length(dPdv);
        if (lenU > 1.0e-8f && lenV > 1.0e-8f) {
            dPduOut = dPdu;
            dPdvOut = dPdv;
            uvPerWorldOut = max(1.0f / lenU, 1.0f / lenV);
            return isfinite(uvPerWorldOut) && uvPerWorldOut > 0.0f;
        }
    }

    float worldArea = length(cross(edge1, edge2));
    float uvArea = fabs(det);
    if (worldArea > 1.0e-12f && uvArea > 1.0e-12f) {
        float uvPerWorldFallback = sqrt(uvArea / worldArea);
        float3 tangentFallback = normalize(edge1);
        float3 bitangentFallback = normalize(cross(normalize(cross(edge1, edge2)), tangentFallback));
        if (!all(isfinite(tangentFallback)) || !all(isfinite(bitangentFallback))) {
            return false;
        }
        dPduOut = tangentFallback / max(uvPerWorldFallback, 1.0e-8f);
        dPdvOut = bitangentFallback / max(uvPerWorldFallback, 1.0e-8f);
        uvPerWorldOut = uvPerWorldFallback;
        return isfinite(uvPerWorldOut) && uvPerWorldOut > 0.0f;
    }
    return false;
}

inline bool triangle_surface_partials(constant PathtraceUniforms& uniforms,
                                      uint meshIndex,
                                      uint primitiveIndex,
                                      device const MeshInfo* meshInfos,
                                      device const SceneVertex* vertices,
                                      device const uint3* meshIndices,
                                      thread float3& dPduOut,
                                      thread float3& dPdvOut,
                                      thread float& uvPerWorldOut) {
    return triangle_surface_partials(uniforms,
                                     meshIndex,
                                     primitiveIndex,
                                     0u,
                                     meshInfos,
                                     vertices,
                                     meshIndices,
                                     dPduOut,
                                     dPdvOut,
                                     uvPerWorldOut);
}

inline bool compute_tangent_basis_from_uv(constant PathtraceUniforms& uniforms,
                                          uint meshIndex,
                                          uint primitiveIndex,
                                          const uint uvSet,
                                          device const MeshInfo* meshInfos,
                                          device const SceneVertex* vertices,
                                          device const uint3* meshIndices,
                                          const float3 shadingNormal,
                                          thread float3& tangent,
                                          thread float3& bitangent) {
    if (!meshInfos || !vertices || !meshIndices || uniforms.meshCount == 0u) {
        return false;
    }
    uint clampedMesh = min(meshIndex, uniforms.meshCount - 1u);
    MeshInfo info = meshInfos[clampedMesh];
    if (info.indexCount == 0u || info.triangleCount == 0u) {
        return false;
    }
    if (primitiveIndex < info.triangleOffset) {
        return false;
    }
    uint localIndex = primitiveIndex - info.triangleOffset;
    if (localIndex >= info.indexCount) {
        return false;
    }
    uint indexEntry = info.indexOffset + localIndex;
    uint3 triIndices = meshIndices[indexEntry];
    SceneVertex v0 = vertices[triIndices.x];
    SceneVertex v1 = vertices[triIndices.y];
    SceneVertex v2 = vertices[triIndices.z];
    float3 p0 = v0.position.xyz;
    float3 p1 = v1.position.xyz;
    float3 p2 = v2.position.xyz;
    float2 uv0 = vertex_uv_set(v0, uvSet);
    float2 uv1 = vertex_uv_set(v1, uvSet);
    float2 uv2 = vertex_uv_set(v2, uvSet);
    float3 edge1 = p1 - p0;
    float3 edge2 = p2 - p0;
    float2 dUV1 = uv1 - uv0;
    float2 dUV2 = uv2 - uv0;
    float denom = (dUV1.x * dUV2.y - dUV1.y * dUV2.x);
    if (fabs(denom) < 1.0e-8f) {
        return false;
    }
    float r = 1.0f / denom;
    float3 tangentLocal = (edge1 * dUV2.y - edge2 * dUV1.y) * r;
    float3 bitangentLocal = (edge2 * dUV1.x - edge1 * dUV2.x) * r;
    float3x3 localToWorld = float3x3(info.localToWorld[0].xyz,
                                     info.localToWorld[1].xyz,
                                     info.localToWorld[2].xyz);
    float3 tangentWorld = localToWorld * tangentLocal;
    float3 bitangentWorld = localToWorld * bitangentLocal;
    float tangentLenSq = dot(tangentWorld, tangentWorld);
    float bitangentLenSq = dot(bitangentWorld, bitangentWorld);
    if (!all(isfinite(tangentWorld)) || tangentLenSq <= 1.0e-12f ||
        !all(isfinite(bitangentWorld)) || bitangentLenSq <= 1.0e-12f) {
        return false;
    }
    tangentWorld *= rsqrt(tangentLenSq);
    bitangentWorld *= rsqrt(bitangentLenSq);
    tangent = normalize(tangentWorld - shadingNormal * dot(shadingNormal, tangentWorld));
    if (!all(isfinite(tangent)) || dot(tangent, tangent) <= 1.0e-6f) {
        return false;
    }
    float detSign = (determinant(localToWorld) < 0.0f) ? -1.0f : 1.0f;
    float handedness = (dot(cross(shadingNormal, tangent), bitangentWorld) < 0.0f) ? -1.0f : 1.0f;
    bitangent = normalize(cross(shadingNormal, tangent)) * (handedness * detSign);
    return true;
}

inline bool compute_tangent_basis_from_uv(constant PathtraceUniforms& uniforms,
                                          uint meshIndex,
                                          uint primitiveIndex,
                                          device const MeshInfo* meshInfos,
                                          device const SceneVertex* vertices,
                                          device const uint3* meshIndices,
                                          const float3 shadingNormal,
                                          thread float3& tangent,
                                          thread float3& bitangent) {
    return compute_tangent_basis_from_uv(uniforms,
                                         meshIndex,
                                         primitiveIndex,
                                         0u,
                                         meshInfos,
                                         vertices,
                                         meshIndices,
                                         shadingNormal,
                                         tangent,
                                         bitangent);
}

inline void build_onb(const float3 normal,
                      thread float3& tangent,
                      thread float3& bitangent) {
    float3 up = (fabs(normal.z) < 0.999f) ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
    tangent = normalize(cross(up, normal));
    bitangent = cross(normal, tangent);
}

inline float3 to_world(const float3 local, const float3 normal) {
    float3 tangent;
    float3 bitangent;
    build_onb(normal, tangent, bitangent);
    return local.x * tangent + local.y * bitangent + local.z * normal;
}

inline float3 to_local(const float3 vector, const float3 normal) {
    float3 tangent;
    float3 bitangent;
    build_onb(normal, tangent, bitangent);
    return float3(dot(vector, tangent), dot(vector, bitangent), dot(vector, normal));
}

inline float3 sample_cosine_hemisphere(thread uint& state) {
    float r1 = rand_uniform(state);
    float r2 = rand_uniform(state);
    float phi = 2.0f * kPi * r2;
    float r = sqrt(max(r1, 0.0f));
    float x = cos(phi) * r;
    float y = sin(phi) * r;
    float z = sqrt(max(1.0f - r1, 0.0f));
    return float3(x, y, z);
}

inline float lambert_pdf(const float3 normal, const float3 direction) {
    float3 dir = normalize(direction);
    float cosTheta = max(dot(normal, dir), 0.0f);
    return cosTheta > 0.0f ? (cosTheta / kPi) : 0.0f;
}

inline uint count_rect_lights(constant PathtraceUniforms& uniforms,
                              device const RectData* rectangles,
                              device const MaterialData* materials) {
    if (!rectangles || !materials || uniforms.rectangleCount == 0 || uniforms.materialCount == 0) {
        return 0u;
    }
    uint lightCount = 0u;
    for (uint i = 0; i < uniforms.rectangleCount; ++i) {
        uint matIndex = min(rectangles[i].materialTwoSided.x, uniforms.materialCount - 1);
        MaterialData material = materials[matIndex];
        if (static_cast<uint>(material.typeEta.x) == 3u &&
            any(material.emission.xyz != float3(0.0f))) {
            lightCount += 1u;
        }
    }
    return lightCount;
}

struct RectLightSample {
    float3 direction;
    float distance;
    float pdf;
    float3 emission;
    uint rectIndex;
};

struct MneeRectHit {
    float3 emission;
    float pdf;
};

inline bool mnee_rect_light_hit(constant PathtraceUniforms& uniforms,
                                device const RectData* rectangles,
                                device const MaterialData* materials,
                                texture2d<float, access::sample> environmentTexture,
                                uint lightCount,
                                thread const HitRecord& lightRec,
                                const float3 origin,
                                thread MneeRectHit& outHit);


inline bool sample_rect_light(constant PathtraceUniforms& uniforms,
                              device const RectData* rectangles,
                              device const MaterialData* materials,
                              texture2d<float, access::sample> environmentTexture,
                              thread const HitRecord& rec,
                              thread uint& state,
                              uint lightCount,
                              thread RectLightSample& outSample) {
    outSample.pdf = 0.0f;
    outSample.distance = 0.0f;
    outSample.direction = float3(0.0f);
    outSample.emission = float3(0.0f);
    outSample.rectIndex = kInvalidIndex;

    if (lightCount == 0u || !rectangles || !materials || uniforms.materialCount == 0) {
        return false;
    }

    uint selected = min(static_cast<uint>(rand_uniform(state) * float(lightCount)), lightCount - 1u);
    uint current = 0u;
    uint chosenRectIndex = kInvalidIndex;
    MaterialData chosenMaterial{};

    for (uint i = 0; i < uniforms.rectangleCount; ++i) {
        uint matIndex = min(rectangles[i].materialTwoSided.x, uniforms.materialCount - 1);
        MaterialData material = materials[matIndex];
        if (static_cast<uint>(material.typeEta.x) != 3u ||
            all(material.emission.xyz == float3(0.0f))) {
            continue;
        }
        if (current == selected) {
            chosenRectIndex = i;
            chosenMaterial = material;
            break;
        }
        current += 1u;
    }

    if (chosenRectIndex == kInvalidIndex) {
        return false;
    }

    const device RectData& rect = rectangles[chosenRectIndex];
    float u = rand_uniform(state);
    float v = rand_uniform(state);

    float3 edgeU = rect.edgeU.xyz;
    float3 edgeV = rect.edgeV.xyz;
    float3 samplePoint = rect.corner.xyz + u * edgeU + v * edgeV;
    float3 toLight = samplePoint - rec.point;
    float distSq = dot(toLight, toLight);
    if (distSq <= 0.0f) {
        return false;
    }

    float distance = sqrt(distSq);
    float3 direction = toLight / distance;

    float area = length(cross(edgeU, edgeV));
    if (area <= 0.0f) {
        return false;
    }

    float3 normal = rect.normalAndPlane.xyz;
    float cosLight = dot(-direction, normal);
    if (rect.materialTwoSided.y != 0u) {
        cosLight = fabs(cosLight);
    } else {
        if (cosLight <= 0.0f) {
            return false;
        }
    }
    if (cosLight <= 0.0f) {
        return false;
    }

    float pdfArea = 1.0f / area;
    float pdfDir = pdfArea * distSq / max(cosLight, 1e-6f);
    float selectionPdf = 1.0f / float(lightCount);
    float pdf = pdfDir * selectionPdf;
    if (pdf <= 0.0f || !isfinite(pdf)) {
        return false;
    }

    float3 emission = chosenMaterial.emission.xyz;
    if (chosenMaterial.emission.w > 0.0f &&
        environmentTexture.get_width() > 0 &&
        environmentTexture.get_height() > 0) {
        float3 sampleDir = -normal;
        float3 envColor = environment_color(environmentTexture,
                                            sampleDir,
                                            uniforms.environmentRotation,
                                            uniforms.environmentIntensity,
                                            uniforms);
        emission *= envColor;
    }

    if (all(emission == float3(0.0f))) {
        return false;
    }

    outSample.direction = direction;
    outSample.distance = distance;
    outSample.pdf = pdf;
    outSample.emission = emission;
    outSample.rectIndex = chosenRectIndex;
    return true;
}

inline float rect_light_pdf_for_hit(constant PathtraceUniforms& uniforms,
                                    device const RectData* rectangles,
                                    device const MaterialData* materials,
                                    uint lightCount,
                                    thread const HitRecord& lightRec,
                                    const float3 origin) {
    if (lightCount == 0u || !rectangles || !materials || uniforms.materialCount == 0) {
        return 0.0f;
    }
    if (lightRec.primitiveType != kPrimitiveTypeRectangle) {
        return 0.0f;
    }
    uint rectIndex = lightRec.primitiveIndex;
    if (rectIndex >= uniforms.rectangleCount) {
        return 0.0f;
    }

    const device RectData& rect = rectangles[rectIndex];
    uint matIndex = min(rect.materialTwoSided.x, uniforms.materialCount - 1);
    MaterialData material = materials[matIndex];
    if (static_cast<uint>(material.typeEta.x) != 3u ||
        all(material.emission.xyz == float3(0.0f))) {
        return 0.0f;
    }

    float3 edgeU = rect.edgeU.xyz;
    float3 edgeV = rect.edgeV.xyz;
    float area = length(cross(edgeU, edgeV));
    if (area <= 0.0f) {
        return 0.0f;
    }

    float3 toLight = lightRec.point - origin;
    float distSq = dot(toLight, toLight);
    if (distSq <= 0.0f) {
        return 0.0f;
    }

    float distance = sqrt(distSq);
    float3 direction = toLight / distance;
    float3 normal = rect.normalAndPlane.xyz;
    float cosLight = dot(-direction, normal);
    if (rect.materialTwoSided.y != 0u) {
        cosLight = fabs(cosLight);
    } else {
        if (cosLight <= 0.0f) {
            return 0.0f;
        }
    }
    if (cosLight <= 0.0f) {
        return 0.0f;
    }

    float pdfArea = 1.0f / area;
    float pdfDir = pdfArea * distSq / max(cosLight, 1e-6f);
    float selectionPdf = 1.0f / float(lightCount);
    return pdfDir * selectionPdf;
}


inline void set_face_normal(thread const Ray& ray,
                            const float3 outwardNormal,
                            thread HitRecord& rec) {
    if (dot(ray.direction, outwardNormal) < 0.0f) {
        rec.frontFace = 1u;
        rec.normal = outwardNormal;
    } else {
        rec.frontFace = 0u;
        rec.normal = -outwardNormal;
    }
    rec.shadingNormal = rec.normal;
}

inline float3 offset_ray_origin(thread const HitRecord& rec, const float3 direction) {
    float3 normal = rec.shadingNormal;
    if (!all(isfinite(normal)) || dot(normal, normal) <= 0.0f) {
        normal = rec.normal;
    }
    float sign = dot(direction, normal) >= 0.0f ? 1.0f : -1.0f;
    float distance = max(fabs(rec.t) * 1e-4f, kRayOriginEpsilon);
    float3 offset = normal * (sign * distance);
    float3 origin = rec.point + offset;
    // Small push along the outgoing direction helps avoid self-intersections at grazing angles.
    origin += direction * kRayOriginEpsilon * 0.5f;
    return origin;
}

inline float3 offset_surface_point(const float3 point,
                                   const float3 normal,
                                   const float3 direction) {
    float3 n = (all(isfinite(normal)) && dot(normal, normal) > 0.0f)
                   ? normalize(normal)
                   : float3(0.0f, 1.0f, 0.0f);
    float sign = dot(direction, n) >= 0.0f ? 1.0f : -1.0f;
    float3 origin = point + n * (sign * kRayOriginEpsilon * 4.0f);
    origin += direction * kRayOriginEpsilon * 0.5f;
    return origin;
}

inline bool intersect_aabb(const float3 boundsMin,
                           const float3 boundsMax,
                           const float3 rayOrigin,
                           const float3 invDir,
                           float tMin,
                           float tMax,
                           thread float& entryOut) {
    float3 t0 = (boundsMin - rayOrigin) * invDir;
    float3 t1 = (boundsMax - rayOrigin) * invDir;
    float3 tNear = min(t0, t1);
    float3 tFar = max(t0, t1);
    float entry = max(max(tNear.x, tNear.y), max(tNear.z, tMin));
    float exit = min(min(tFar.x, tFar.y), min(tFar.z, tMax));
    entryOut = entry;
    return exit >= entry;
}

inline bool hit_sphere(const SphereData sphere,
                       uint sphereIndex,
                       thread const Ray& ray,
                       float tMin,
                       float tMax,
                       thread HitRecord& rec) {
    float3 center = sphere.centerRadius.xyz;
    float radius = sphere.centerRadius.w;

    float3 oc = ray.origin - center;
    float a = dot(ray.direction, ray.direction);
    float half_b = dot(oc, ray.direction);
    float c = dot(oc, oc) - radius * radius;

    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f) {
        return false;
    }

    float sqrtD = sqrt(discriminant);
    float root = (-half_b - sqrtD) / a;
    if (root < tMin || root > tMax) {
        root = (-half_b + sqrtD) / a;
        if (root < tMin || root > tMax) {
            return false;
        }
    }

    rec.t = root;
    rec.point = ray_at(ray, rec.t);
    float3 outwardNormal = (rec.point - center) / radius;
    rec.twoSided = 1u;
    rec.meshIndex = 0u;
    rec.barycentric = float2(0.0f, 0.0f);
    set_face_normal(ray, outwardNormal, rec);
    rec.materialIndex = sphere.materialIndex.x;
    rec.primitiveType = kPrimitiveTypeSphere;
    rec.primitiveIndex = sphereIndex;
    return true;
}

inline bool hit_rectangle(const RectData rect,
                          uint rectIndex,
                          thread const Ray& ray,
                          float tMin,
                          float tMax,
                          thread HitRecord& rec) {
    float3 normal = rect.normalAndPlane.xyz;
    float denom = dot(normal, ray.direction);
    if (fabs(denom) < 1e-6f) {
        return false;
    }

    float planeConstant = rect.normalAndPlane.w;
    float t = (planeConstant - dot(normal, ray.origin)) / denom;
    if (t < tMin || t > tMax) {
        return false;
    }

    float3 point = ray_at(ray, t);
    float3 relative = point - rect.corner.xyz;

    float u = dot(relative, rect.edgeU.xyz) * rect.edgeU.w;
    float v = dot(relative, rect.edgeV.xyz) * rect.edgeV.w;
    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
        return false;
    }

    rec.t = t;
    rec.point = point;
    rec.twoSided = rect.materialTwoSided.y;
    rec.meshIndex = 0u;
    rec.barycentric = float2(0.0f, 0.0f);
    set_face_normal(ray, normal, rec);
    rec.materialIndex = rect.materialTwoSided.x;
    rec.primitiveType = kPrimitiveTypeRectangle;
    rec.primitiveIndex = rectIndex;
    return true;
}


inline float3 sky_color(const float3 direction) {
    float3 unit = normalize(direction);
    float t = 0.5f * (unit.y + 1.0f);
    return mix(float3(1.0f, 1.0f, 1.0f), float3(0.5f, 0.7f, 1.0f), t);
}

inline float environment_max_mip(texture2d<float, access::sample> environmentTexture) {
    uint mipCount = environmentTexture.get_num_mip_levels();
    if (mipCount == 0u) {
        return 0.0f;
    }
    return float(mipCount - 1u);
}

inline float environment_lod_from_roughness(float roughness,
                                            texture2d<float, access::sample> environmentTexture) {
    float maxMip = environment_max_mip(environmentTexture);
    if (maxMip <= 0.0f) {
        return 0.0f;
    }
    float alpha = clamp(roughness, 0.0f, 1.0f);
    alpha = alpha * alpha;
    float lod = alpha * maxMip;
    return clamp(lod, 0.0f, maxMip);
}

inline float visor_override_mask(const float3 baseColor,
                                 const float metallic,
                                 const float roughness) {
    float luminance = dot(baseColor, float3(0.2126f, 0.7152f, 0.0722f));
    float dark = 1.0f - smoothstep(0.12f, 0.30f, luminance);
    float nonMetal = 1.0f - smoothstep(0.15f, 0.40f, metallic);
    float smooth = 1.0f - smoothstep(0.12f, 0.30f, roughness);
    return clamp(dark * nonMetal * smooth, 0.0f, 1.0f);
}

inline float visor_override_blend(const float3 baseColor,
                                  const float metallic,
                                  const float roughness,
                                  const uint materialIndex,
                                  constant PathtraceUniforms& uniforms) {
    if (uniforms.debugEnableVisorOverride == 0u) {
        return 0.0f;
    }
    const int selectedMaterial = uniforms.debugVisorOverrideMaterialId;
    if (selectedMaterial >= 0) {
        return (materialIndex == static_cast<uint>(selectedMaterial)) ? 1.0f : 0.0f;
    }
    return visor_override_mask(baseColor, metallic, roughness);
}

inline float3 environment_color(texture2d<float, access::sample> environmentTexture,
                                const float3 direction,
                                float rotation,
                                float intensity,
                                constant PathtraceUniforms& uniforms) {
    float3 unit = normalize(direction);
    float cosTheta = cos(rotation);
    float sinTheta = sin(rotation);
    float3 rotated = float3(unit.x * cosTheta - unit.z * sinTheta,
                            unit.y,
                            unit.x * sinTheta + unit.z * cosTheta);
    float u = (atan2(rotated.z, rotated.x) + kPi) / (2.0f * kPi);
    float v = 0.5f - asin(clamp(rotated.y, -1.0f, 1.0f)) / kPi;
    sampler s = (uniforms.debugEnvNearest != 0u) ? environmentSamplerNearest : environmentSampler;
    float3 color = environmentTexture.sample(s, float2(u, v)).xyz * intensity;
    return to_working_space(color, uniforms);
}

inline float3 environment_color_lod(texture2d<float, access::sample> environmentTexture,
                                    const float3 direction,
                                    float rotation,
                                    float intensity,
                                    float lod,
                                    constant PathtraceUniforms& uniforms) {
    float3 unit = normalize(direction);
    float cosTheta = cos(rotation);
    float sinTheta = sin(rotation);
    float3 rotated = float3(unit.x * cosTheta - unit.z * sinTheta,
                            unit.y,
                            unit.x * sinTheta + unit.z * cosTheta);
    float u = (atan2(rotated.z, rotated.x) + kPi) / (2.0f * kPi);
    float v = 0.5f - asin(clamp(rotated.y, -1.0f, 1.0f)) / kPi;
    sampler s = (uniforms.debugEnvNearest != 0u) ? environmentSamplerNearest : environmentSampler;
    float3 color = environmentTexture.sample(s,
                                             float2(u, v),
                                             level(lod)).xyz * intensity;
    return to_working_space(color, uniforms);
}

inline bool environment_mip_override(constant PathtraceUniforms& uniforms,
                                     texture2d<float, access::sample> environmentTexture,
                                     thread float& outLod) {
    if (uniforms.debugEnvMipOverride < 0.0f) {
        return false;
    }
    uint mipCount = environmentTexture.get_num_mip_levels();
    if (mipCount <= 1u) {
        return false;
    }
    float maxMip = float(mipCount - 1u);
    outLod = clamp(uniforms.debugEnvMipOverride, 0.0f, maxMip);
    return true;
}

struct EnvironmentSample {
    float3 direction;
    float3 radiance;
    float pdf;
};

inline bool environment_sampling_available(constant PathtraceUniforms& uniforms,
                         device const EnvironmentAliasEntry* conditionalAlias,
                         device const EnvironmentAliasEntry* marginalAlias,
                         device const float* pdfTable) {
    return uniforms.environmentHasDistribution != 0u &&
           uniforms.environmentAliasCount > 0u &&
           uniforms.environmentMapWidth > 0u &&
           uniforms.environmentMapHeight > 0u &&
        conditionalAlias != nullptr &&
        marginalAlias != nullptr &&
           pdfTable != nullptr;
}

inline float environment_pdf(constant PathtraceUniforms& uniforms,
                             device const float* pdfTable,
                             const float3 direction) {
    if (uniforms.environmentHasDistribution == 0u ||
        uniforms.environmentAliasCount == 0u ||
        uniforms.environmentMapWidth == 0u ||
        uniforms.environmentMapHeight == 0u ||
        pdfTable == nullptr) {
        return 0.0f;
    }

    float3 unit = normalize(direction);
    float cosTheta = cos(uniforms.environmentRotation);
    float sinTheta = sin(uniforms.environmentRotation);
    float3 rotated = float3(unit.x * cosTheta - unit.z * sinTheta,
                            unit.y,
                            unit.x * sinTheta + unit.z * cosTheta);
    float u = (atan2(rotated.z, rotated.x) + kPi) / (2.0f * kPi);
    float v = 0.5f - asin(clamp(rotated.y, -1.0f, 1.0f)) / kPi;

    uint width = max(uniforms.environmentMapWidth, 1u);
    uint height = max(uniforms.environmentMapHeight, 1u);

    u = clamp(u, 0.0f, 0.99999994f);
    v = clamp(v, 0.0f, 0.99999994f);

    uint x = min(uint(u * float(width)), width - 1u);
    uint y = min(uint(v * float(height)), height - 1u);
    uint index = min(y * width + x, uniforms.environmentAliasCount - 1u);
    float value = pdfTable[index];
    if (!isfinite(value) || value <= 0.0f) {
        return 0.0f;
    }
    return value;
}

inline float power_heuristic(const float pdfA, const float pdfB) {
    float pdfA2 = pdfA * pdfA;
    float pdfB2 = pdfB * pdfB;
    float denom = pdfA2 + pdfB2;
    if (denom <= 0.0f) {
        return 0.0f;
    }
    float weight = pdfA2 / denom;
    if (!isfinite(weight)) {
        return 0.0f;
    }
    return clamp(weight, kMisWeightClampMin, kMisWeightClampMax);
}

inline bool sample_environment(constant PathtraceUniforms& uniforms,
                               texture2d<float, access::sample> environmentTexture,
                               device const EnvironmentAliasEntry* conditionalAlias,
                               device const EnvironmentAliasEntry* marginalAlias,
                               device const float* pdfTable,
                               thread uint& state,
                               thread EnvironmentSample& outSample) {
    outSample.direction = float3(0.0f);
    outSample.radiance = float3(0.0f);
    outSample.pdf = 0.0f;

    if (!environment_sampling_available(uniforms, conditionalAlias, marginalAlias, pdfTable) ||
        environmentTexture.get_width() == 0 ||
        environmentTexture.get_height() == 0) {
        return false;
    }

    uint width = max(uniforms.environmentMapWidth, 1u);
    uint height = max(uniforms.environmentMapHeight, 1u);

    float uMarginal = rand_uniform(state);
    float uConditional = rand_uniform(state);
    float uJitter = rand_uniform(state);

    float rowChoice = uMarginal * float(height);
    float rowFloor = floor(rowChoice);
    uint row = min(uint(rowFloor), height - 1u);
    float rowFraction = rowChoice - rowFloor;
    const device EnvironmentAliasEntry& rowEntry = marginalAlias[row];
    if (rowFraction >= rowEntry.threshold) {
        row = min(rowEntry.alias, height - 1u);
    }

    float colChoice = uConditional * float(width);
    float colFloor = floor(colChoice);
    uint col = min(uint(colFloor), width - 1u);
    uint aliasIndexBase = min(row * width + col, uniforms.environmentAliasCount - 1u);
    float colFraction = colChoice - colFloor;
    const device EnvironmentAliasEntry& conditionalEntry = conditionalAlias[aliasIndexBase];
    if (colFraction >= conditionalEntry.threshold) {
        col = min(conditionalEntry.alias, width - 1u);
        aliasIndexBase = min(row * width + col, uniforms.environmentAliasCount - 1u);
    }

    float fx = (float(col) + fract(uConditional)) / float(width);
    float fy = (float(row) + clamp(uJitter, 0.0f, 0.99999994f)) / float(height);

    float theta = fy * kPi;
    float phi = fx * (2.0f * kPi);
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);
    float3 mapDir = float3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));

    float cosRot = cos(uniforms.environmentRotation);
    float sinRot = sin(uniforms.environmentRotation);
    float3 worldDir = float3(mapDir.x * cosRot + mapDir.z * sinRot,
                             mapDir.y,
                             -mapDir.x * sinRot + mapDir.z * cosRot);

    float pdf = pdfTable[aliasIndexBase];
    if (!isfinite(pdf) || pdf <= 0.0f) {
        return false;
    }

    float3 radiance = environment_color(environmentTexture,
                                        worldDir,
                                        uniforms.environmentRotation,
                                        uniforms.environmentIntensity,
                                        uniforms);
    if (!all(isfinite(radiance))) {
        return false;
    }

    outSample.direction = worldDir;
    outSample.radiance = max(radiance, float3(0.0f));
    outSample.pdf = pdf;
    return true;
}

struct TraversalCounters {
    uint nodeVisits;
    uint leafPrimTests;
    uint internalVisits;
    uint internalBothVisited;
};

inline void reset_counters(thread TraversalCounters& counters) {
    counters.nodeVisits = 0u;
    counters.leafPrimTests = 0u;
    counters.internalVisits = 0u;
    counters.internalBothVisited = 0u;
}

inline bool hit_triangle(constant PathtraceUniforms& uniforms,
                         const TriangleData tri,
                         uint triangleIndex,
                         thread const Ray& ray,
                         float tMin,
                         float tMax,
                         thread HitRecord& rec) {
    float3 v0 = tri.v0.xyz;
    float3 v1 = tri.v1.xyz;
    float3 v2 = tri.v2.xyz;

    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 pVec = cross(ray.direction, edge2);
    float det = dot(edge1, pVec);
    if (fabs(det) < 1e-8f) {
        return false;
    }

    float invDet = 1.0f / det;
    float3 tVec = ray.origin - v0;
    float u = dot(tVec, pVec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    float3 qVec = cross(tVec, edge1);
    float v = dot(ray.direction, qVec) * invDet;
    if (v < 0.0f || (u + v) > 1.0f) {
        return false;
    }

    float t = dot(edge2, qVec) * invDet;
    if (t < tMin || t > tMax) {
        return false;
    }

    float3 outwardNormal = cross(edge1, edge2);
    if (!all(isfinite(outwardNormal))) {
        return false;
    }
    outwardNormal = normalize(outwardNormal);
    if (!all(isfinite(outwardNormal))) {
        return false;
    }

    rec.t = t;
    rec.point = ray_at(ray, t);
    rec.twoSided = 0u;
    rec.meshIndex = tri.metadata.y;
    rec.barycentric = float2(u, v);
    set_face_normal(ray, outwardNormal, rec);

    uint materialIndex = tri.metadata.x;
    if (uniforms.materialCount > 0u) {
        materialIndex = min(materialIndex, uniforms.materialCount - 1u);
    } else {
        materialIndex = 0u;
    }

    rec.materialIndex = materialIndex;
    rec.primitiveType = kPrimitiveTypeTriangle;
    rec.primitiveIndex = triangleIndex;
    return true;
}

inline bool brute_force_hit_triangles(constant PathtraceUniforms& uniforms,
                                      device const TriangleData* triangles,
                                      thread const Ray& ray,
                                      float tMin,
                                      thread float& closest,
                                      thread HitRecord& rec,
                                      thread uint& leafPrimTests,
                                      bool anyHitOnly) {
    if (!triangles || uniforms.triangleCount == 0u) {
        return false;
    }

    HitRecord tempRec;
    bool hitAnything = false;

    for (uint i = 0; i < uniforms.triangleCount; ++i) {
        leafPrimTests += 1u;
        if (hit_triangle(uniforms, triangles[i], i, ray, tMin, closest, tempRec)) {
            closest = tempRec.t;
            rec = tempRec;
            hitAnything = true;
            if (anyHitOnly) {
                return true;
            }
        }
    }

    return hitAnything;
}

inline bool brute_force_hit_spheres(constant PathtraceUniforms& uniforms,
                                    device const SphereData* spheres,
                                    thread const Ray& ray,
                                    float tMin,
                                    float tMax,
                                    thread HitRecord& rec,
                                    thread TraversalCounters& counters) {
    if (!spheres || uniforms.sphereCount == 0) {
        return false;
    }
    HitRecord tempRec;
    bool hitAnything = false;
    float closestSoFar = tMax;

    for (uint i = 0; i < uniforms.sphereCount; ++i) {
        counters.leafPrimTests += 1u;
        if (hit_sphere(spheres[i], i, ray, tMin, closestSoFar, tempRec)) {
            hitAnything = true;
            closestSoFar = tempRec.t;
            rec = tempRec;
        }
    }

    return hitAnything;
}

inline bool brute_force_hit_rectangles(constant PathtraceUniforms& uniforms,
                                       device const RectData* rectangles,
                                       thread const Ray& ray,
                                       float tMin,
                                       thread float& closest,
                                       thread HitRecord& rec) {
    if (!rectangles || uniforms.rectangleCount == 0) {
        return false;
    }

    HitRecord tempRec;
    bool hitAnything = false;

    for (uint i = 0; i < uniforms.rectangleCount; ++i) {
        if (hit_rectangle(rectangles[i], i, ray, tMin, closest, tempRec)) {
            closest = tempRec.t;
            rec = tempRec;
            hitAnything = true;
        }
    }

    return hitAnything;
}

inline bool traverse_bvh(device const BvhNode* nodes,
                         device const SphereData* spheres,
                         device const uint* primitiveIndices,
                         constant PathtraceUniforms& uniforms,
                         thread const Ray& ray,
                         float tMin,
                         bool anyHitOnly,
                         thread float& closest,
                         thread HitRecord& rec,
                         thread TraversalCounters& counters,
                         thread bool& earlyExit) {
    if (!nodes || !primitiveIndices || !spheres || uniforms.primitiveCount == 0) {
        return false;
    }

    float3 invDir = 1.0f / ray.direction;

    uint stack[kBvhTraversalStackSize];
    uint stackSize = 0;
    stack[stackSize++] = 0;

    bool hitAnything = false;

    while (stackSize > 0) {
        uint nodeIndex = stack[--stackSize];
        counters.nodeVisits += 1u;
        const device BvhNode& node = nodes[nodeIndex];
        float3 boundsMin = node.boundsMin.xyz;
        float3 boundsMax = node.boundsMax.xyz;
        float nodeEntry = 0.0f;

        if (!intersect_aabb(boundsMin, boundsMax, ray.origin, invDir, tMin, closest, nodeEntry)) {
            continue;
        }

        if (node.primitiveCount > 0) {
            for (uint i = 0; i < node.primitiveCount; ++i) {
                uint primIndex = primitiveIndices[node.primitiveOffset + i];
                if (primIndex >= uniforms.sphereCount) {
                    continue;
                }
                HitRecord tempRec;
                counters.leafPrimTests += 1u;
                if (hit_sphere(spheres[primIndex], primIndex, ray, tMin, closest, tempRec)) {
                    closest = tempRec.t;
                    rec = tempRec;
                    hitAnything = true;
                    if (anyHitOnly) {
                        earlyExit = true;
                        stackSize = 0;
                        break;
                    }
                }
            }
        } else {
            counters.internalVisits += 1u;

            float leftEntry = 0.0f;
            float rightEntry = 0.0f;
            bool leftHit = false;
            bool rightHit = false;

            if (node.leftChild != kInvalidIndex) {
                const device BvhNode& leftNode = nodes[node.leftChild];
                leftHit = intersect_aabb(leftNode.boundsMin.xyz,
                                          leftNode.boundsMax.xyz,
                                          ray.origin,
                                          invDir,
                                          tMin,
                                          closest,
                                          leftEntry);
            }
            if (node.rightChild != kInvalidIndex) {
                const device BvhNode& rightNode = nodes[node.rightChild];
                rightHit = intersect_aabb(rightNode.boundsMin.xyz,
                                           rightNode.boundsMax.xyz,
                                           ray.origin,
                                           invDir,
                                           tMin,
                                           closest,
                                           rightEntry);
            }

            if (leftHit && rightHit) {
                counters.internalBothVisited += 1u;
            }

            if (!leftHit && !rightHit) {
                continue;
            }

            if (leftHit && rightHit) {
                uint nearChild = node.leftChild;
                uint farChild = node.rightChild;
                if (rightEntry < leftEntry) {
                    nearChild = node.rightChild;
                    farChild = node.leftChild;
                }
                if (farChild != kInvalidIndex && stackSize < kBvhTraversalStackSize) {
                    stack[stackSize++] = farChild;
                }
                if (nearChild != kInvalidIndex && stackSize < kBvhTraversalStackSize) {
                    stack[stackSize++] = nearChild;
                }
            } else if (leftHit) {
                if (node.leftChild != kInvalidIndex && stackSize < kBvhTraversalStackSize) {
                    stack[stackSize++] = node.leftChild;
                }
            } else if (rightHit) {
                if (node.rightChild != kInvalidIndex && stackSize < kBvhTraversalStackSize) {
                    stack[stackSize++] = node.rightChild;
                }
            }
        }
    }

    return hitAnything;
}

inline bool traverse_bvh_triangles(device const BvhNode* nodes,
                                   device const TriangleData* triangles,
                                   device const uint* primitiveIndices,
                                   constant PathtraceUniforms& uniforms,
                                   thread const Ray& ray,
                                   float tMin,
                                   bool anyHitOnly,
                                   thread float& closest,
                                   thread HitRecord& rec,
                                   thread TraversalCounters& counters,
                                   thread bool& earlyExit) {
    if (!nodes || !primitiveIndices || !triangles || uniforms.primitiveCount == 0) {
        return false;
    }

    float3 invDir = 1.0f / ray.direction;

    uint stack[kBvhTraversalStackSize];
    uint stackSize = 0;
    stack[stackSize++] = 0;

    bool hitAnything = false;

    while (stackSize > 0) {
        uint nodeIndex = stack[--stackSize];
        counters.nodeVisits += 1u;
        const device BvhNode& node = nodes[nodeIndex];
        float3 boundsMin = node.boundsMin.xyz;
        float3 boundsMax = node.boundsMax.xyz;
        float nodeEntry = 0.0f;

        if (!intersect_aabb(boundsMin, boundsMax, ray.origin, invDir, tMin, closest, nodeEntry)) {
            continue;
        }

        if (node.primitiveCount > 0) {
            for (uint i = 0; i < node.primitiveCount; ++i) {
                uint primIndex = primitiveIndices[node.primitiveOffset + i];
                if (primIndex >= uniforms.triangleCount) {
                    continue;
                }
                HitRecord tempRec;
                counters.leafPrimTests += 1u;
                if (hit_triangle(uniforms, triangles[primIndex], primIndex, ray, tMin, closest, tempRec)) {
                    closest = tempRec.t;
                    rec = tempRec;
                    hitAnything = true;
                    if (anyHitOnly) {
                        earlyExit = true;
                        stackSize = 0;
                        break;
                    }
                }
            }
        } else {
            counters.internalVisits += 1u;

            float leftEntry = 0.0f;
            float rightEntry = 0.0f;
            bool leftHit = false;
            bool rightHit = false;

            if (node.leftChild != kInvalidIndex) {
                const device BvhNode& leftNode = nodes[node.leftChild];
                leftHit = intersect_aabb(leftNode.boundsMin.xyz,
                                          leftNode.boundsMax.xyz,
                                          ray.origin,
                                          invDir,
                                          tMin,
                                          closest,
                                          leftEntry);
            }
            if (node.rightChild != kInvalidIndex) {
                const device BvhNode& rightNode = nodes[node.rightChild];
                rightHit = intersect_aabb(rightNode.boundsMin.xyz,
                                           rightNode.boundsMax.xyz,
                                           ray.origin,
                                           invDir,
                                           tMin,
                                           closest,
                                           rightEntry);
            }

            if (leftHit && rightHit) {
                counters.internalBothVisited += 1u;
            }

            if (!leftHit && !rightHit) {
                continue;
            }

            if (leftHit && rightHit) {
                uint nearChild = node.leftChild;
                uint farChild = node.rightChild;
                if (rightEntry < leftEntry) {
                    nearChild = node.rightChild;
                    farChild = node.leftChild;
                }
                if (farChild != kInvalidIndex && stackSize < kBvhTraversalStackSize) {
                    stack[stackSize++] = farChild;
                }
                if (nearChild != kInvalidIndex && stackSize < kBvhTraversalStackSize) {
                    stack[stackSize++] = nearChild;
                }
            } else if (leftHit) {
                if (node.leftChild != kInvalidIndex && stackSize < kBvhTraversalStackSize) {
                    stack[stackSize++] = node.leftChild;
                }
            } else if (rightHit) {
                if (node.rightChild != kInvalidIndex && stackSize < kBvhTraversalStackSize) {
                    stack[stackSize++] = node.rightChild;
                }
            }
        }
    }

    return hitAnything;
}

inline bool traverse_bvh_triangles_segment(device const BvhNode* nodes,
                                           uint nodeRootOffset,
                                           device const TriangleData* triangles,
                                           device const uint* primitiveIndices,
                                           uint primIndexOffset,
                                           constant PathtraceUniforms& uniforms,
                                           const device SoftwareInstanceInfo& instance,
                                           uint triangleBaseOffset,
                                           thread const Ray& worldRay,
                                           float tMin,
                                           bool anyHitOnly,
                                           thread float& closestWorld,
                                           thread HitRecord& worldRec,
                                           thread TraversalCounters& counters,
                                           thread bool& earlyExit) {
    if (!nodes || !primitiveIndices || !triangles) {
        return false;
    }

    float4x4 worldToLocal4 = instance.worldToLocal;
    float4x4 localToWorld4 = instance.localToWorld;

    Ray localRay;
    localRay.origin = (worldToLocal4 * float4(worldRay.origin, 1.0f)).xyz;
    localRay.direction = (worldToLocal4 * float4(worldRay.direction, 0.0f)).xyz;

    float dirLenSqLocal = dot(localRay.direction, localRay.direction);
    if (!(dirLenSqLocal > 0.0f)) {
        return false;
    }

    float3 invDirLocal = 1.0f / localRay.direction;
    float denomWorld = dot(worldRay.direction, worldRay.direction);
    if (!(denomWorld > 0.0f)) {
        denomWorld = 1.0f;
    }

    float3x3 worldToLocal3 = float3x3(worldToLocal4[0].xyz,
                                      worldToLocal4[1].xyz,
                                      worldToLocal4[2].xyz);
    float3x3 normalMatrix = transpose(worldToLocal3);

    auto toLocalPoint = [&](float3 worldPoint) -> float3 {
        return (worldToLocal4 * float4(worldPoint, 1.0f)).xyz;
    };
    auto toWorldPoint = [&](float3 localPoint) -> float3 {
        return (localToWorld4 * float4(localPoint, 1.0f)).xyz;
    };
    auto worldParamToLocal = [&](float worldT) -> float {
        if (!isfinite(worldT)) {
            return INFINITY;
        }
        float3 worldPoint = worldRay.origin + worldRay.direction * worldT;
        float3 localPoint = toLocalPoint(worldPoint);
        return dot(localPoint - localRay.origin, localRay.direction) / dirLenSqLocal;
    };

    float localTMin = max(worldParamToLocal(tMin), 0.0f);
    float localClosest = worldParamToLocal(closestWorld);
    if (!isfinite(localClosest)) {
        localClosest = INFINITY;
    }

    uint stack[kBvhTraversalStackSize];
    uint stackSize = 0;
    stack[stackSize++] = nodeRootOffset;

    bool hitAnything = false;
    while (stackSize > 0) {
        uint nodeIndex = stack[--stackSize];
        counters.nodeVisits += 1u;
        const device BvhNode& node = nodes[nodeIndex];
        float nodeEntry = 0.0f;
        if (!intersect_aabb(node.boundsMin.xyz,
                            node.boundsMax.xyz,
                            localRay.origin,
                            invDirLocal,
                            localTMin,
                            localClosest,
                            nodeEntry)) {
            continue;
        }
        if (node.primitiveCount > 0) {
            for (uint i = 0; i < node.primitiveCount; ++i) {
                uint primLocal = primitiveIndices[primIndexOffset + node.primitiveOffset + i];
                uint triIndex = triangleBaseOffset + primLocal;
                if (triIndex >= uniforms.triangleCount) {
                    continue;
                }
                const TriangleData tri = triangles[triIndex];
                HitRecord localRec;
                counters.leafPrimTests += 1u;
                if (!hit_triangle(uniforms,
                                  tri,
                                  triIndex,
                                  localRay,
                                  localTMin,
                                  localClosest,
                                  localRec)) {
                    continue;
                }

                float3 worldPoint = toWorldPoint(localRec.point);
                float3 localGeom = cross(tri.v1.xyz - tri.v0.xyz, tri.v2.xyz - tri.v0.xyz);
                float3 worldNormal = (all(isfinite(localGeom)) && dot(localGeom, localGeom) > 0.0f)
                    ? (normalMatrix * localGeom)
                    : (normalMatrix * localRec.normal);
                if (!all(isfinite(worldPoint)) ||
                    !all(isfinite(worldNormal)) ||
                    dot(worldNormal, worldNormal) <= 0.0f) {
                    continue;
                }
                worldNormal = normalize(worldNormal);

                float worldT = dot(worldPoint - worldRay.origin, worldRay.direction) / denomWorld;
                if (!(worldT > tMin) || !(worldT < closestWorld)) {
                    continue;
                }

                HitRecord worldCandidate = localRec;
                worldCandidate.point = worldPoint;
                worldCandidate.t = worldT;
                set_face_normal(worldRay, worldNormal, worldCandidate);

                closestWorld = worldT;
                float localProjection = worldParamToLocal(worldT);
                if (isfinite(localProjection)) {
                    localClosest = max(localProjection, localTMin);
                }
                worldRec = worldCandidate;
                hitAnything = true;
                if (anyHitOnly) {
                    earlyExit = true;
                    stackSize = 0;
                    break;
                }
            }
        } else {
            counters.internalVisits += 1u;
            float leftEntry = 0.0f;
            float rightEntry = 0.0f;
            bool leftHit = false;
            bool rightHit = false;
            if (node.leftChild != kInvalidIndex) {
                const device BvhNode& leftNode = nodes[node.leftChild];
                leftHit = intersect_aabb(leftNode.boundsMin.xyz,
                                         leftNode.boundsMax.xyz,
                                         localRay.origin,
                                         invDirLocal,
                                         localTMin,
                                         localClosest,
                                         leftEntry);
            }
            if (node.rightChild != kInvalidIndex) {
                const device BvhNode& rightNode = nodes[node.rightChild];
                rightHit = intersect_aabb(rightNode.boundsMin.xyz,
                                          rightNode.boundsMax.xyz,
                                          localRay.origin,
                                          invDirLocal,
                                          localTMin,
                                          localClosest,
                                          rightEntry);
            }
            if (leftHit && rightHit) {
                counters.internalBothVisited += 1u;
            }
            if (!leftHit && !rightHit) {
                continue;
            }
            if (leftHit && rightHit) {
                uint nearChild = node.leftChild;
                uint farChild = node.rightChild;
                if (rightEntry < leftEntry) {
                    nearChild = node.rightChild;
                    farChild = node.leftChild;
                }
                if (farChild != kInvalidIndex && stackSize < kBvhTraversalStackSize) {
                    stack[stackSize++] = farChild;
                }
                if (nearChild != kInvalidIndex && stackSize < kBvhTraversalStackSize) {
                    stack[stackSize++] = nearChild;
                }
            } else if (leftHit) {
                if (node.leftChild != kInvalidIndex && stackSize < kBvhTraversalStackSize) {
                    stack[stackSize++] = node.leftChild;
                }
            } else if (rightHit) {
                if (node.rightChild != kInvalidIndex && stackSize < kBvhTraversalStackSize) {
                    stack[stackSize++] = node.rightChild;
                }
            }
        }
    }
    return hitAnything;
}

inline bool trace_scene_tlas_triangles(constant PathtraceUniforms& uniforms,
                                       device const BvhNode* tlasNodes,
                                       device const uint* tlasPrimIndices,
                                       device const SoftwareInstanceInfo* instanceInfos,
                                       device const BvhNode* blasNodes,
                                       device const uint* blasPrimIndices,
                                       device const TriangleData* triangles,
                                       device PathtraceStats* stats,
                                       thread const Ray& ray,
                                       float tMin,
                                       float tMax,
                                       bool anyHitOnly,
                                       thread HitRecord& rec) {
    if (!tlasNodes || !tlasPrimIndices || !instanceInfos || !blasNodes || !blasPrimIndices || !triangles) {
        return false;
    }
    float closest = tMax;
    bool hitAnything = false;
    TraversalCounters counters; reset_counters(counters);
    bool earlyExit = false;

    // Traverse TLAS similarly to BVH over instance AABBs
    float3 invDir = 1.0f / ray.direction;
    uint stack[kBvhTraversalStackSize]; uint stackSize = 0; stack[stackSize++] = 0;
    while (stackSize > 0) {
        uint nodeIndex = stack[--stackSize];
        counters.nodeVisits += 1u;
        const device BvhNode& node = tlasNodes[nodeIndex];
        float nodeEntry = 0.0f;
        if (!intersect_aabb(node.boundsMin.xyz, node.boundsMax.xyz, ray.origin, invDir, tMin, closest, nodeEntry)) {
            continue;
        }
        if (node.primitiveCount > 0) {
            for (uint i = 0; i < node.primitiveCount; ++i) {
                uint instanceId = tlasPrimIndices[node.primitiveOffset + i];
                const device SoftwareInstanceInfo& info = instanceInfos[instanceId];
                HitRecord tempRec = rec;
                float closestCopy = closest;
                if (traverse_bvh_triangles_segment(blasNodes,
                                                   info.blasRootNodeOffset,
                                                   triangles,
                                                   blasPrimIndices,
                                                   info.blasPrimIndexOffset,
                                                   uniforms,
                                                   info,
                                                   info.triangleBaseOffset,
                                                   ray,
                                                   tMin,
                                                   anyHitOnly,
                                                   closestCopy,
                                                   tempRec,
                                                   counters,
                                                   earlyExit)) {
                    closest = closestCopy;
                    rec = tempRec;
                    hitAnything = true;
                    if (anyHitOnly) { earlyExit = true; stackSize = 0; break; }
                }
            }
        } else {
            counters.internalVisits += 1u;
            float leftEntry=0.0f, rightEntry=0.0f; bool leftHit=false, rightHit=false;
            if (node.leftChild != kInvalidIndex) {
                const device BvhNode& leftNode = tlasNodes[node.leftChild];
                leftHit = intersect_aabb(leftNode.boundsMin.xyz, leftNode.boundsMax.xyz, ray.origin, invDir, tMin, closest, leftEntry);
            }
            if (node.rightChild != kInvalidIndex) {
                const device BvhNode& rightNode = tlasNodes[node.rightChild];
                rightHit = intersect_aabb(rightNode.boundsMin.xyz, rightNode.boundsMax.xyz, ray.origin, invDir, tMin, closest, rightEntry);
            }
            if (leftHit && rightHit) { counters.internalBothVisited += 1u; }
            if (!leftHit && !rightHit) { continue; }
            if (leftHit && rightHit) {
                uint nearChild = node.leftChild, farChild = node.rightChild;
                if (rightEntry < leftEntry) { nearChild = node.rightChild; farChild = node.leftChild; }
                if (farChild != kInvalidIndex && stackSize < kBvhTraversalStackSize) stack[stackSize++] = farChild;
                if (nearChild != kInvalidIndex && stackSize < kBvhTraversalStackSize) stack[stackSize++] = nearChild;
            } else if (leftHit) {
                if (node.leftChild != kInvalidIndex && stackSize < kBvhTraversalStackSize) stack[stackSize++] = node.leftChild;
            } else if (rightHit) {
                if (node.rightChild != kInvalidIndex && stackSize < kBvhTraversalStackSize) stack[stackSize++] = node.rightChild;
            }
        }
    }

    if (stats) {
        atomic_fetch_add_explicit(&stats->primaryRayCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->nodesVisited, counters.nodeVisits, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->leafPrimTests, counters.leafPrimTests, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->internalNodeVisits, counters.internalVisits, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->internalBothVisited, counters.internalBothVisited, memory_order_relaxed);
        if (anyHitOnly && earlyExit) {
            atomic_fetch_add_explicit(&stats->shadowRayCount, 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&stats->shadowRayEarlyExitCount, 1u, memory_order_relaxed);
        }
    }
    return hitAnything;
}

inline bool trace_scene_software(constant PathtraceUniforms& uniforms,
                        device const SphereData* spheres,
                        device const RectData* rectangles,
                        device const TriangleData* triangles,
                        // TLAS/BLAS resources (software)
                        device const BvhNode* tlasNodes,
                        device const uint* tlasPrimIndices,
                        device const SoftwareInstanceInfo* instanceInfos,
                        device const BvhNode* blasNodes,
                        device const uint* blasPrimIndices,
                        // Legacy BVH (spheres or triangles)
                        device const BvhNode* nodes,
                        device const uint* primitiveIndices,
                        device PathtraceStats* stats,
                        thread const Ray& ray,
                        float tMin,
                        float tMax,
                        bool anyHitOnly,
                        bool includeTriangles,
                        thread HitRecord& rec) {
    float closest = tMax;
    bool hitAnything = false;

    TraversalCounters counters;
    reset_counters(counters);
    bool earlyExit = false;

    rec.primitiveType = kPrimitiveTypeNone;
    rec.primitiveIndex = kInvalidIndex;

    // Try TLAS path for triangles first if available
    if (uniforms.softwareBvhType == kSoftwareBvhTriangles &&
        triangles && tlasNodes && tlasPrimIndices && instanceInfos && blasNodes && blasPrimIndices) {
        if (trace_scene_tlas_triangles(uniforms,
                                       tlasNodes,
                                       tlasPrimIndices,
                                       instanceInfos,
                                       blasNodes,
                                       blasPrimIndices,
                                       triangles,
                                       stats,
                                       ray,
                                       tMin,
                                       tMax,
                                       anyHitOnly,
                                       rec)) {
            return true;
        }
    }

    if (uniforms.intersectionMode == kIntersectionModeSoftwareBVH &&
        nodes && primitiveIndices && uniforms.primitiveCount > 0) {
        if (uniforms.softwareBvhType == kSoftwareBvhTriangles && triangles) {
            hitAnything = traverse_bvh_triangles(nodes,
                                                 triangles,
                                                 primitiveIndices,
                                                 uniforms,
                                                 ray,
                                                 tMin,
                                                 anyHitOnly,
                                                 closest,
                                                 rec,
                                                 counters,
                                                 earlyExit);
        } else if (spheres) {
            hitAnything = traverse_bvh(nodes,
                                       spheres,
                                       primitiveIndices,
                                       uniforms,
                                       ray,
                                       tMin,
                                       anyHitOnly,
                                       closest,
                                       rec,
                                       counters,
                                       earlyExit);
        }
    }

    if (!hitAnything) {
        hitAnything = brute_force_hit_spheres(uniforms, spheres, ray, tMin, closest, rec, counters);
    }

    if (brute_force_hit_rectangles(uniforms, rectangles, ray, tMin, closest, rec)) {
        hitAnything = true;
    }

    if (includeTriangles &&
        uniforms.softwareBvhType != kSoftwareBvhTriangles &&
        !(anyHitOnly && hitAnything) &&
        brute_force_hit_triangles(uniforms,
                                  triangles,
                                  ray,
                                  tMin,
                                  closest,
                                  rec,
                                  counters.leafPrimTests,
                                  anyHitOnly)) {
        hitAnything = true;
    }

    if (stats) {
        atomic_fetch_add_explicit(&stats->primaryRayCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->nodesVisited, counters.nodeVisits, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->leafPrimTests, counters.leafPrimTests, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->internalNodeVisits, counters.internalVisits, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->internalBothVisited, counters.internalBothVisited, memory_order_relaxed);
        if (anyHitOnly) {
            atomic_fetch_add_explicit(&stats->shadowRayCount, 1u, memory_order_relaxed);
            if (earlyExit) {
                atomic_fetch_add_explicit(&stats->shadowRayEarlyExitCount, 1u, memory_order_relaxed);
            }
        }
    }

    return hitAnything;
}

#if __METAL_VERSION__ >= 310
inline bool trace_scene_hardware(constant PathtraceUniforms& uniforms,
                                 acceleration_structure<instancing> accel,
                                 device const MeshInfo* meshInfos,
                                 device const TriangleData* triangleData,
                                 device const SceneVertex* sceneVertices,
                                 device const uint3* meshIndices,
                                 device const uint* instanceUserIds,
                                 device const SphereData* spheres,
                                 device const RectData* rectangles,
                                 device const BvhNode* nodes,
                                 device const uint* primitiveIndices,
                                 device PathtraceStats* stats,
                                 thread const Ray& ray,
                                 float tMin,
                                 float tMax,
                                 bool anyHitOnly,
                                 uint excludeMeshIndex,
                                 uint excludePrimitiveIndex,
                                 thread HitRecord& rec) {
    rec.primitiveType = kPrimitiveTypeNone;
    rec.primitiveIndex = kInvalidIndex;

    float closest = tMax;
    bool hitAnything = false;

    bool hardwareHit = false;
    HitRecord hardwareRec = rec;
    uint excludeMesh = excludeMeshIndex;
    uint excludePrim = excludePrimitiveIndex;

    if (stats) {
        atomic_fetch_add_explicit(&stats->hardwareRayCount, 1u, memory_order_relaxed);
    }

    bool hadCandidateDistance = false;
    float candidateDistance = closest;
    uint lastCandidateInstanceId = kInvalidIndex;
    uint lastCandidatePrimitiveId = kInvalidIndex;
    bool sawCandidate = false;
    uint retriesUsed = 0u;

    if (meshInfos != nullptr && triangleData != nullptr &&
        uniforms.meshCount > 0u && uniforms.triangleCount > 0u) {
        intersector<triangle_data, instancing> intersector;
        intersector.assume_geometry_type(geometry_type::triangle);
        intersector.set_triangle_cull_mode(triangle_cull_mode::none);
        const uint kHardwareExcludeMaxAttempts = 4u;
        uint maxAttempts = min(max(uniforms.hardwareExcludeMaxAttempts, 1u),
                               kHardwareExcludeMaxAttempts);
        Ray currentRay = ray;
        float currentTMin = tMin;
        for (uint attempt = 0u; attempt < maxAttempts; ++attempt) {
            raytracing::ray query(currentRay.origin, currentRay.direction, currentTMin, tMax);
            auto result = intersector.intersect(query, accel);
            if (result.type != intersection_type::none) {
                hadCandidateDistance = true;
                candidateDistance = result.distance;
                sawCandidate = true;
                lastCandidateInstanceId = result.instance_id;
                lastCandidatePrimitiveId = result.primitive_id;
            } else {
                candidateDistance = closest;
            }

            uint instanceId = result.instance_id;
            if (stats) {
                uint32_t resultType = static_cast<uint32_t>(result.type);
                atomic_store_explicit(&stats->hardwareLastResultType, resultType, memory_order_relaxed);
                atomic_store_explicit(&stats->hardwareLastPrimitiveId,
                                      result.primitive_id,
                                      memory_order_relaxed);
                uint32_t distanceBits = as_type<uint32_t>(result.distance);
                atomic_store_explicit(&stats->hardwareLastDistanceBits,
                                      distanceBits,
                                      memory_order_relaxed);
                if (result.type == intersection_type::none) {
                    atomic_fetch_add_explicit(&stats->hardwareResultNoneCount, 1u, memory_order_relaxed);
                }
            }

            if (result.type == intersection_type::none) {
                break;
            }

            uint meshIndex = instanceId;
            if (instanceUserIds != nullptr && instanceId < uniforms.meshCount) {
                meshIndex = instanceUserIds[instanceId];
            }
            if (meshIndex >= uniforms.meshCount) {
                meshIndex = mesh_index_from_triangle(result.primitive_id,
                                                     meshInfos,
                                                     uniforms.meshCount);
            }
            if (meshIndex >= uniforms.meshCount) {
                if (anyHitOnly) {
                    hardwareHit = true;
                }
                break;
            }

            MeshInfo info = meshInfos[meshIndex];
            if (info.triangleCount == 0u) {
                if (anyHitOnly) {
                    hardwareHit = true;
                }
                break;
            }

            uint primitiveId = result.primitive_id;
            if (primitiveId >= info.triangleCount) {
                // Some drivers may report primitive IDs in index-buffer space.
                uint primitiveIdFromIndex = primitiveId / 3u;
                if (primitiveIdFromIndex < info.triangleCount) {
                    primitiveId = primitiveIdFromIndex;
                } else {
                    if (anyHitOnly) {
                        hardwareHit = true;
                    }
                    break;
                }
            }
            uint triIndex = info.triangleOffset + primitiveId;
            if (triIndex >= uniforms.triangleCount) {
                if (anyHitOnly) {
                    hardwareHit = true;
                }
                break;
            }

            TriangleData tri = triangleData[triIndex];
            if (tri.metadata.y < uniforms.meshCount) {
                meshIndex = tri.metadata.y;
                info = meshInfos[meshIndex];
            }

            float3 localV0 = tri.v0.xyz;
            float3 localV1 = tri.v1.xyz;
            float3 localV2 = tri.v2.xyz;
            float4x4 localToWorld = info.localToWorld;
            float4x4 worldToLocal = info.worldToLocal;
            float3 worldV0 = (localToWorld * float4(localV0, 1.0f)).xyz;
            float3 worldV1 = (localToWorld * float4(localV1, 1.0f)).xyz;
            float3 worldV2 = (localToWorld * float4(localV2, 1.0f)).xyz;

            float resolvedT = 0.0f;
            float2 bary = float2(0.0f, 0.0f);
            bool resolvedHit =
                intersect_triangle_parametric(worldV0,
                                              worldV1,
                                              worldV2,
                                              currentRay,
                                              currentTMin,
                                              closest,
                                              resolvedT,
                                              bary);

            float3 worldPoint = ray_at(currentRay, result.distance);
            if (resolvedHit) {
                worldPoint = ray_at(currentRay, resolvedT);
            } else {
                float4 localPoint4 = worldToLocal * float4(worldPoint, 1.0f);
                float invW = (fabs(localPoint4.w) > 1.0e-8f) ? (1.0f / localPoint4.w) : 1.0f;
                float3 localPoint = localPoint4.xyz * invW;
                bary = barycentric_from_point(localV0, localV1, localV2, localPoint);
                resolvedT = result.distance;
            }

            float3 worldNormal = cross(worldV1 - worldV0, worldV2 - worldV0);
            if (!all(isfinite(worldNormal)) || length(worldNormal) <= 0.0f) {
                float3 localNormal = cross(localV1 - localV0, localV2 - localV0);
                float3x3 worldToLocal3 = float3x3(worldToLocal[0].xyz,
                                                  worldToLocal[1].xyz,
                                                  worldToLocal[2].xyz);
                worldNormal = transpose(worldToLocal3) * localNormal;
            }
            if (!all(isfinite(worldPoint)) || !all(isfinite(worldNormal))) {
                worldNormal = float3(0.0f);
            } else {
                float normalLen = length(worldNormal);
                if (normalLen > 0.0f) {
                    worldNormal /= normalLen;
                } else {
                    worldNormal = float3(0.0f);
                }
            }

            if (!(meshIndex < uniforms.meshCount &&
                  all(isfinite(worldNormal)) && length(worldNormal) > 0.0f)) {
                if (anyHitOnly) {
                    hardwareHit = true;
                }
                break;
            }

            float worldHitDistance = ray_segment_world_length(currentRay, resolvedT);
            if (!isfinite(worldHitDistance)) {
                worldHitDistance = kInfinity;
            }
            bool selfHit = (meshIndex == rec.meshIndex &&
                            triIndex == rec.primitiveIndex &&
                            fabs(worldHitDistance) <= kHardwareOcclusionEpsilon);
            bool excluded = (meshIndex == excludeMesh &&
                             (triIndex == excludePrim ||
                              (anyHitOnly && worldHitDistance <= kHardwareOcclusionEpsilon)));
            bool excludeMeshOnly = (excluded && excludePrim == kInvalidIndex);

            if (selfHit) {
                if (stats) {
                    atomic_fetch_add_explicit(&stats->hardwareSelfHitRejectedCount,
                                              1u,
                                              memory_order_relaxed);
                    uint32_t distBits = as_type<uint32_t>(worldHitDistance);
                    atomic_store_explicit(&stats->hardwareSelfHitLastDistanceBits,
                                          distBits,
                                          memory_order_relaxed);
                }
            }

            if (selfHit || excluded) {
                retriesUsed += 1u;
                float3 dir = currentRay.direction;
                float dirLenSq = dot(dir, dir);
                float3 dirStep = float3(0.0f, 0.0f, 1.0f);
                if (all(isfinite(dir)) && dirLenSq > 1.0e-12f) {
                    dirStep = dir * rsqrt(dirLenSq);
                }
                currentRay.origin = worldPoint + dirStep * kHardwareOcclusionEpsilon;
                currentTMin = 0.0f;
                if (excludeMeshOnly) {
                    excludeMesh = kInvalidIndex;
                }
                continue;
            }

            float3 interpolatedNormal = worldNormal;
            if (meshInfos && sceneVertices && meshIndices) {
                float3 candidate =
                    interpolate_shading_normal(uniforms,
                                               meshIndex,
                                               triIndex,
                                               bary,
                                               meshInfos,
                                               sceneVertices,
                                               meshIndices);
                if (all(isfinite(candidate)) && dot(candidate, candidate) > 0.0f) {
                    if (dot(candidate, worldNormal) < 0.0f) {
                        candidate = -candidate;
                    }
                    interpolatedNormal = normalize(candidate);
                }
            }

            bool triangleFrontFacing = (dot(currentRay.direction, worldNormal) < 0.0f);
#if __METAL_VERSION__ >= 310
            bool hardwareFrontFacing = result.triangle_front_facing;
            if (hardwareFrontFacing == triangleFrontFacing) {
                triangleFrontFacing = hardwareFrontFacing;
            }
#endif
            if (!triangleFrontFacing) {
                worldNormal = -worldNormal;
            }

            hardwareRec.t = resolvedT;
            hardwareRec.point = worldPoint;
            uint materialIndex = tri.metadata.x;
            if (uniforms.materialCount > 0u) {
                materialIndex = min(materialIndex, uniforms.materialCount - 1u);
            } else {
                materialIndex = 0u;
            }
            hardwareRec.materialIndex = materialIndex;
            hardwareRec.twoSided = 0u;
            hardwareRec.primitiveType = kPrimitiveTypeTriangle;
            hardwareRec.primitiveIndex = triIndex;
            hardwareRec.meshIndex = meshIndex;
            hardwareRec.barycentric = bary;
            hardwareRec.frontFace = triangleFrontFacing ? 1u : 0u;
            hardwareRec.normal = worldNormal;
            hardwareRec.shadingNormal = interpolatedNormal;
            closest = hardwareRec.t;
            hardwareHit = true;
            if (stats) {
                atomic_fetch_add_explicit(&stats->hardwareHitCount, 1u, memory_order_relaxed);
                atomic_store_explicit(&stats->hardwareLastInstanceId,
                                      meshIndex,
                                      memory_order_relaxed);
            }
            break;
        }
        if (stats) {
            uint retryBin = min(retriesUsed, 3u);
            atomic_fetch_add_explicit(&stats->hardwareExcludeRetryHistogram[retryBin],
                                      1u,
                                      memory_order_relaxed);
        }
    }

    if (!hardwareHit && stats) {
        atomic_fetch_add_explicit(&stats->hardwareMissCount, 1u, memory_order_relaxed);
        float missDistance = hadCandidateDistance ? candidateDistance : closest;
        if (!isfinite(missDistance) || missDistance <= 0.0f) {
            missDistance = 0.0f;
        }
        float logValue = log2(fmax(missDistance, 1.0e-6f));
        int binIndex = clamp(int(logValue) + 8, 0, 31);
        atomic_fetch_add_explicit(&stats->hardwareMissDistanceBins[binIndex],
                                  1u,
                                  memory_order_relaxed);
        uint32_t missBits = as_type<uint32_t>(missDistance);
        atomic_store_explicit(&stats->hardwareMissLastDistanceBits,
                              missBits,
                              memory_order_relaxed);
        uint missInstanceId = sawCandidate ? lastCandidateInstanceId : kInvalidIndex;
        uint missPrimitiveId = sawCandidate ? lastCandidatePrimitiveId : kInvalidIndex;
        atomic_store_explicit(&stats->hardwareMissLastInstanceId,
                              missInstanceId,
                              memory_order_relaxed);
        atomic_store_explicit(&stats->hardwareMissLastPrimitiveId,
                              missPrimitiveId,
                              memory_order_relaxed);
    }

    if (anyHitOnly) {
        if (hardwareHit) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->primaryRayCount, 1u, memory_order_relaxed);
                atomic_fetch_add_explicit(&stats->shadowRayCount, 1u, memory_order_relaxed);
                atomic_fetch_add_explicit(&stats->shadowRayEarlyExitCount, 1u, memory_order_relaxed);
            }
            rec = hardwareRec;
            return true;
        }
        return trace_scene_software(uniforms,
                                    spheres,
                                    rectangles,
                                    triangleData,
                                    /*tlas*/ nullptr,
                                    /*tlasPrim*/ nullptr,
                                    /*instances*/ nullptr,
                                    /*blas*/ nullptr,
                                    /*blasPrim*/ nullptr,
                                    nodes,
                                    primitiveIndices,
                                    stats,
                                    ray,
                                    tMin,
                                    tMax,
                                    /*anyHitOnly=*/true,
                                    /*includeTriangles=*/false,
                                    rec);
    }

    HitRecord bestRec = rec;
    if (hardwareHit) {
        bestRec = hardwareRec;
        hitAnything = true;
    }

    TraversalCounters counters;
    reset_counters(counters);
    bool earlyExit = false;

    if (nodes && primitiveIndices && uniforms.primitiveCount > 0 && spheres) {
        HitRecord tempRec = bestRec;
        float closestCopy = closest;
        if (traverse_bvh(nodes,
                         spheres,
                         primitiveIndices,
                         uniforms,
                         ray,
                         tMin,
                         /*anyHitOnly=*/false,
                         closestCopy,
                         tempRec,
                         counters,
                         earlyExit)) {
            closest = closestCopy;
            bestRec = tempRec;
            hitAnything = true;
        }
    }

    HitRecord sphereRec;
    if (brute_force_hit_spheres(uniforms, spheres, ray, tMin, closest, sphereRec, counters)) {
        closest = sphereRec.t;
        bestRec = sphereRec;
        hitAnything = true;
    }

    HitRecord rectRec = bestRec;
    if (brute_force_hit_rectangles(uniforms, rectangles, ray, tMin, closest, rectRec)) {
        closest = rectRec.t;
        bestRec = rectRec;
        hitAnything = true;
    }

    rec = bestRec;

    if (stats) {
        atomic_fetch_add_explicit(&stats->primaryRayCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->nodesVisited, counters.nodeVisits, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->leafPrimTests, counters.leafPrimTests, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->internalNodeVisits, counters.internalVisits, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->internalBothVisited, counters.internalBothVisited, memory_order_relaxed);
    }

    return hitAnything;
}
#endif

inline bool trace_scene_software_with_exclusion(constant PathtraceUniforms& uniforms,
                                                device const SphereData* spheres,
                                                device const RectData* rectangles,
                                                device const TriangleData* triangles,
                                                device const BvhNode* tlasNodes,
                                                device const uint* tlasPrimIndices,
                                                device const SoftwareInstanceInfo* instanceInfos,
                                                device const BvhNode* blasNodes,
                                                device const uint* blasPrimIndices,
                                                device const BvhNode* nodes,
                                                device const uint* primitiveIndices,
                                                device PathtraceStats* stats,
                                                thread const Ray& ray,
                                                float tMin,
                                                float tMax,
                                                uint excludeMeshIndex,
                                                uint excludePrimitiveIndex,
                                                thread HitRecord& rec) {
    if (excludeMeshIndex == kInvalidIndex || excludePrimitiveIndex == kInvalidIndex) {
        return trace_scene_software(uniforms,
                                    spheres,
                                    rectangles,
                                    triangles,
                                    tlasNodes,
                                    tlasPrimIndices,
                                    instanceInfos,
                                    blasNodes,
                                    blasPrimIndices,
                                    nodes,
                                    primitiveIndices,
                                    stats,
                                    ray,
                                    tMin,
                                    tMax,
                                    /*anyHitOnly=*/false,
                                    /*includeTriangles=*/true,
                                    rec);
    }

    const uint kMaxAttempts = 4u;
    uint maxAttempts = min(max(uniforms.hardwareExcludeMaxAttempts, 1u), kMaxAttempts);
    Ray currentRay = ray;
    float currentTMin = tMin;
    for (uint attempt = 0u; attempt < maxAttempts; ++attempt) {
        HitRecord tempRec = make_empty_hit_record();
        device PathtraceStats* attemptStats = (attempt == 0u) ? stats : nullptr;
        if (!trace_scene_software(uniforms,
                                  spheres,
                                  rectangles,
                                  triangles,
                                  tlasNodes,
                                  tlasPrimIndices,
                                  instanceInfos,
                                  blasNodes,
                                  blasPrimIndices,
                                  nodes,
                                  primitiveIndices,
                                  attemptStats,
                                  currentRay,
                                  currentTMin,
                                  tMax,
                                  /*anyHitOnly=*/false,
                                  /*includeTriangles=*/true,
                                  tempRec)) {
            return false;
        }

        bool excluded = (tempRec.primitiveType == kPrimitiveTypeTriangle &&
                         tempRec.meshIndex == excludeMeshIndex &&
                         tempRec.primitiveIndex == excludePrimitiveIndex);
        if (!excluded) {
            rec = tempRec;
            return true;
        }

        float advance = max(tempRec.t + kHardwareOcclusionEpsilon, kHardwareOcclusionEpsilon);
        currentRay.origin = ray_at(currentRay, advance);
        currentRay.origin += currentRay.direction * kHardwareOcclusionEpsilon;
        currentTMin = kHardwareOcclusionEpsilon;
    }

    return false;
}

inline float clamp01(const float value) {
    return clamp(value, 0.0f, 1.0f);
}

inline float3 clamp01(const float3 value) {
    return clamp(value, float3(0.0f), float3(1.0f));
}

inline bool material_texture_valid(constant PathtraceUniforms& uniforms, uint index) {
    return index != kInvalidIndex && index < uniforms.materialTextureCount;
}

inline MaterialTextureInfo material_texture_info(
    array<texture2d<float, access::sample>, kMaxMaterialTextures> textures,
    device const MaterialTextureInfo* textureInfos,
    constant PathtraceUniforms& uniforms,
    uint index) {
    MaterialTextureInfo info{};
    if (!material_texture_valid(uniforms, index)) {
        return info;
    }
    if (textureInfos) {
        info = textureInfos[index];
    }
    if (info.width == 0u) {
        info.width = textures[index].get_width();
    }
    if (info.height == 0u) {
        info.height = textures[index].get_height();
    }
    if (info.mipCount == 0u) {
        info.mipCount = textures[index].get_num_mip_levels();
    }
    return info;
}

constant uint kPbrTextureSlotBaseColor = 0u;
constant uint kPbrTextureSlotMetallicRoughness = 1u;
constant uint kPbrTextureSlotNormal = 2u;
constant uint kPbrTextureSlotOcclusion = 3u;
constant uint kPbrTextureSlotEmissive = 4u;
constant uint kPbrTextureSlotTransmission = 5u;

inline uint pbr_texture_uv_set(const MaterialData material, const uint slot) {
    switch (slot) {
        case kPbrTextureSlotBaseColor:
            return min(material.textureUvSet0.x, 1u);
        case kPbrTextureSlotMetallicRoughness:
            return min(material.textureUvSet0.y, 1u);
        case kPbrTextureSlotNormal:
            return min(material.textureUvSet0.z, 1u);
        case kPbrTextureSlotOcclusion:
            return min(material.textureUvSet0.w, 1u);
        case kPbrTextureSlotEmissive:
            return min(material.textureUvSet1.x, 1u);
        case kPbrTextureSlotTransmission:
            return min(material.textureUvSet1.y, 1u);
        default:
            return 0u;
    }
}

inline void pbr_texture_transform_rows(const MaterialData material,
                                       const uint slot,
                                       thread float3& row0,
                                       thread float3& row1) {
    switch (slot) {
        case kPbrTextureSlotBaseColor:
            row0 = material.textureTransform0.xyz;
            row1 = material.textureTransform1.xyz;
            break;
        case kPbrTextureSlotMetallicRoughness:
            row0 = material.textureTransform2.xyz;
            row1 = material.textureTransform3.xyz;
            break;
        case kPbrTextureSlotNormal:
            row0 = material.textureTransform4.xyz;
            row1 = material.textureTransform5.xyz;
            break;
        case kPbrTextureSlotOcclusion:
            row0 = material.textureTransform6.xyz;
            row1 = material.textureTransform7.xyz;
            break;
        case kPbrTextureSlotEmissive:
            row0 = material.textureTransform8.xyz;
            row1 = material.textureTransform9.xyz;
            break;
        case kPbrTextureSlotTransmission:
            row0 = material.textureTransform10.xyz;
            row1 = material.textureTransform11.xyz;
            break;
        default:
            row0 = float3(1.0f, 0.0f, 0.0f);
            row1 = float3(0.0f, 1.0f, 0.0f);
            break;
    }

    float linearSum = fabs(row0.x) + fabs(row0.y) + fabs(row1.x) + fabs(row1.y);
    if (!all(isfinite(row0)) || !all(isfinite(row1)) || !(linearSum > 1.0e-8f)) {
        row0 = float3(1.0f, 0.0f, 0.0f);
        row1 = float3(0.0f, 1.0f, 0.0f);
    }
}

inline float2 pbr_apply_uv_transform(const float2 uv,
                                     const float3 row0,
                                     const float3 row1) {
    return float2(dot(row0.xy, uv) + row0.z,
                  dot(row1.xy, uv) + row1.z);
}

inline float2 pbr_transform_uv_gradient(const float2 grad,
                                        const float3 row0,
                                        const float3 row1) {
    return float2(dot(row0.xy, grad),
                  dot(row1.xy, grad));
}

inline float pbr_transform_uv_per_world(const float uvPerWorld,
                                        const float3 row0,
                                        const float3 row1) {
    float sx = length(float2(row0.x, row1.x));
    float sy = length(float2(row0.y, row1.y));
    float scaleBound = max(max(sx, sy), 1.0e-6f);
    return uvPerWorld * scaleBound;
}

struct PbrTextureSamplingContext {
    float2 uv = float2(0.0f);
    bool hasIgehyGradients = false;
    float2 dUVdx = float2(0.0f);
    float2 dUVdy = float2(0.0f);
    float uvPerWorld = 0.0f;
};

inline PbrTextureSamplingContext make_pbr_texture_sampling_context(const MaterialData material,
                                                                   const uint slot,
                                                                   const float2 uv0,
                                                                   const float2 uv1,
                                                                   const bool hasGradients0,
                                                                   const float2 dUVdx0,
                                                                   const float2 dUVdy0,
                                                                   const float uvPerWorld0,
                                                                   const bool hasGradients1,
                                                                   const float2 dUVdx1,
                                                                   const float2 dUVdy1,
                                                                   const float uvPerWorld1) {
    PbrTextureSamplingContext ctx;
    uint uvSet = pbr_texture_uv_set(material, slot);
    float2 baseUv = (uvSet == 0u) ? uv0 : uv1;
    bool baseHasGradients = (uvSet == 0u) ? hasGradients0 : hasGradients1;
    float2 baseDx = (uvSet == 0u) ? dUVdx0 : dUVdx1;
    float2 baseDy = (uvSet == 0u) ? dUVdy0 : dUVdy1;
    float baseUvPerWorld = (uvSet == 0u) ? uvPerWorld0 : uvPerWorld1;

    float3 row0 = float3(1.0f, 0.0f, 0.0f);
    float3 row1 = float3(0.0f, 1.0f, 0.0f);
    pbr_texture_transform_rows(material, slot, row0, row1);
    ctx.uv = pbr_apply_uv_transform(baseUv, row0, row1);
    ctx.hasIgehyGradients = baseHasGradients;
    if (ctx.hasIgehyGradients) {
        ctx.dUVdx = pbr_transform_uv_gradient(baseDx, row0, row1);
        ctx.dUVdy = pbr_transform_uv_gradient(baseDy, row0, row1);
        if (!all(isfinite(ctx.dUVdx)) || !all(isfinite(ctx.dUVdy))) {
            ctx.hasIgehyGradients = false;
            ctx.dUVdx = float2(0.0f);
            ctx.dUVdy = float2(0.0f);
        }
    }
    ctx.uvPerWorld = pbr_transform_uv_per_world(baseUvPerWorld, row0, row1);
    return ctx;
}

inline float4 sample_material_texture(
    array<texture2d<float, access::sample>, kMaxMaterialTextures> textures,
    array<sampler, kMaxMaterialSamplers> materialSamplers,
    device const MaterialTextureInfo* textureInfos,
    constant PathtraceUniforms& uniforms,
    uint index,
    float2 uv,
    float4 fallback) {
    if (!material_texture_valid(uniforms, index)) {
        return fallback;
    }
    MaterialTextureInfo info = material_texture_info(textures, textureInfos, uniforms, index);
    uint samplerIndex = min(info.flags, kMaxMaterialSamplers - 1u);
    return textures[index].sample(materialSamplers[samplerIndex], uv);
}

inline float4 sample_material_texture_level(
    array<texture2d<float, access::sample>, kMaxMaterialTextures> textures,
    array<sampler, kMaxMaterialSamplers> materialSamplers,
    device const MaterialTextureInfo* textureInfos,
    constant PathtraceUniforms& uniforms,
    uint index,
    float2 uv,
    float4 fallback,
    float lod) {
    if (!material_texture_valid(uniforms, index)) {
        return fallback;
    }
    MaterialTextureInfo info = material_texture_info(textures, textureInfos, uniforms, index);
    uint samplerIndex = min(info.flags, kMaxMaterialSamplers - 1u);
    if (info.mipCount <= 1u) {
        return textures[index].sample(materialSamplers[samplerIndex], uv);
    }
    float maxMip = float(info.mipCount - 1u);
    float clampedLod = clamp(lod, 0.0f, maxMip);
    return textures[index].sample(materialSamplers[samplerIndex], uv, level(clampedLod));
}

inline float4 sample_material_texture_filtered(
    array<texture2d<float, access::sample>, kMaxMaterialTextures> textures,
    array<sampler, kMaxMaterialSamplers> materialSamplers,
    device const MaterialTextureInfo* textureInfos,
    constant PathtraceUniforms& uniforms,
    uint index,
    float2 uv,
    float4 fallback,
    float lod,
    bool hasIgehyGradients,
    float2 dUVdx,
    float2 dUVdy) {
    if (!material_texture_valid(uniforms, index)) {
        return fallback;
    }
    MaterialTextureInfo info = material_texture_info(textures, textureInfos, uniforms, index);
    uint samplerIndex = min(info.flags, kMaxMaterialSamplers - 1u);

    if (hasIgehyGradients &&
        all(isfinite(dUVdx)) &&
        all(isfinite(dUVdy))) {
        float gradMag = max(max(fabs(dUVdx.x), fabs(dUVdx.y)),
                            max(fabs(dUVdy.x), fabs(dUVdy.y)));
        if (gradMag > 0.0f && isfinite(gradMag)) {
            return textures[index].sample(materialSamplers[samplerIndex],
                                          uv,
                                          gradient2d(dUVdx, dUVdy));
        }
    }

    if (info.mipCount <= 1u) {
        return textures[index].sample(materialSamplers[samplerIndex], uv);
    }
    float maxMip = float(info.mipCount - 1u);
    float clampedLod = clamp(lod, 0.0f, maxMip);
    return textures[index].sample(materialSamplers[samplerIndex], uv, level(clampedLod));
}

inline float material_texture_lod_from_cone(
    array<texture2d<float, access::sample>, kMaxMaterialTextures> textures,
    device const MaterialTextureInfo* textureInfos,
    constant PathtraceUniforms& uniforms,
    uint index,
    float uvPerWorld,
    float footprintWorld) {
    if (!material_texture_valid(uniforms, index)) {
        return 0.0f;
    }
    MaterialTextureInfo info = material_texture_info(textures, textureInfos, uniforms, index);
    return ray_cone_lod_from_footprint(info, uvPerWorld, footprintWorld);
}

inline bool material_texture_lod_from_gradients(
    array<texture2d<float, access::sample>, kMaxMaterialTextures> textures,
    device const MaterialTextureInfo* textureInfos,
    constant PathtraceUniforms& uniforms,
    uint index,
    float2 dUVdx,
    float2 dUVdy,
    thread float& outLod) {
    outLod = 0.0f;
    if (!material_texture_valid(uniforms, index)) {
        return false;
    }
    if (!all(isfinite(dUVdx)) || !all(isfinite(dUVdy))) {
        return false;
    }
    MaterialTextureInfo info = material_texture_info(textures, textureInfos, uniforms, index);
    if (info.width == 0u || info.height == 0u || info.mipCount <= 1u) {
        return false;
    }
    float W = float(info.width);
    float H = float(info.height);
    float rho = max(max(fabs(dUVdx.x) * W, fabs(dUVdx.y) * H),
                    max(fabs(dUVdy.x) * W, fabs(dUVdy.y) * H));
    if (!isfinite(rho) || !(rho > 0.0f)) {
        return false;
    }
    float lod = log2(max(rho, 1.0e-8f));
    if (!isfinite(lod)) {
        return false;
    }
    float maxMip = float(info.mipCount - 1u);
    outLod = clamp(lod, 0.0f, maxMip);
    return isfinite(outLod);
}

inline float material_texture_lod_with_fallback(
    array<texture2d<float, access::sample>, kMaxMaterialTextures> textures,
    device const MaterialTextureInfo* textureInfos,
    constant PathtraceUniforms& uniforms,
    uint index,
    bool hasIgehyGradients,
    float2 dUVdx,
    float2 dUVdy,
    float uvPerWorld,
    float surfaceFootprintWorld) {
    float lod = 0.0f;
    bool gradientOk = false;
    if (hasIgehyGradients) {
        gradientOk = material_texture_lod_from_gradients(textures,
                                                         textureInfos,
                                                         uniforms,
                                                         index,
                                                         dUVdx,
                                                         dUVdy,
                                                         lod);
    }
    if (!gradientOk) {
        lod = material_texture_lod_from_cone(textures,
                                             textureInfos,
                                             uniforms,
                                             index,
                                             uvPerWorld,
                                             surfaceFootprintWorld);
    }
    if (!isfinite(lod)) {
        lod = material_texture_lod_from_cone(textures,
                                             textureInfos,
                                             uniforms,
                                             index,
                                             uvPerWorld,
                                             surfaceFootprintWorld);
    }
    return max(lod, 0.0f);
}


inline float safe_sqrt(const float value) {
    return sqrt(max(value, 0.0f));
}

inline float3 safe_normalize(const float3 v) {
    float len2 = dot(v, v);
    if (len2 <= 0.0f) {
        return float3(0.0f, 0.0f, 0.0f);
    }
    return v * rsqrt(len2);
}

inline float luminance_rgb(const float3 color) {
    return dot(color, kLuminanceWeights);
}

inline void stats_add_mnee_luma(device PathtraceStats* stats, const float3 contribution) {
    if (!stats) {
        return;
    }
    float luma = luminance_rgb(contribution);
    if (!isfinite(luma) || luma <= 0.0f) {
        return;
    }
    constexpr float kMneeLumaScale = 1024.0f;
    float scaled = luma * kMneeLumaScale;
    uint add = static_cast<uint>(min(scaled, 4294967295.0f));
    uint prev = atomic_fetch_add_explicit(&stats->mneeContributionLumaSumLo,
                                          add,
                                          memory_order_relaxed);
    if (prev + add < prev) {
        atomic_fetch_add_explicit(&stats->mneeContributionLumaSumHi, 1u, memory_order_relaxed);
    }
    atomic_fetch_add_explicit(&stats->mneeContributionCount, 1u, memory_order_relaxed);
}

struct FireflyClampParams {
    float clampFactor;
    float clampFloor;
    float throughputClamp;
    float specularTailClampBase;
    float specularTailClampRoughnessScale;
    float minSpecularPdf;
    float maxContribution;
    float enabled;
};

struct CarpaintLobeResult {
    float3 f;
    float pdf;
};

inline float plastic_coat_roughness(const MaterialData material);
inline float plastic_coat_f0(const MaterialData material);
inline float3 plastic_specular_tint(const MaterialData material);
inline float3 plastic_diffuse_transmission(const MaterialData material,
                                           const float cosThetaI,
                                           const float cosThetaO);
inline float clamp_specular_pdf(const float pdf, const FireflyClampParams params);
inline float3 clamp_specular_tail(const float3 value,
                                  const float roughness,
                                  const float3 f0,
                                  const FireflyClampParams params);
inline float ggx_D(const float alpha, const float cosThetaH);
inline float ggx_G1(const float alpha, const float cosTheta);
inline float ggx_pdf(const float alpha,
                     const float3 normal,
                     const float3 wo,
                     const float3 wi);
inline float3 schlick_fresnel(const float3 f0, const float cosTheta);
inline float3 material_base_color(const MaterialData material);
inline float lambert_pdf(const float3 normal, const float3 direction);
inline float3 dielectric_sigma_a(const MaterialData material) {
    return material.dielectricSigmaA.xyz;
}

inline float3 transmission_tint(const MaterialData material, const float cosTheta) {
    float thickness = max(material.typeEta.w, 0.0f);
    if (thickness <= 0.0f) {
        return float3(1.0f);
    }
    float3 sigmaA = max(dielectric_sigma_a(material), float3(0.0f));
    if (all(sigmaA <= float3(0.0f))) {
        return float3(1.0f);
    }
    float distance = thickness / max(fabs(cosTheta), 1.0e-3f);
    return clamp01(exp(-sigmaA * distance));
}

inline bool material_is_carpaint(const MaterialData material) {
    return static_cast<uint>(material.typeEta.x) == 6u;
}

inline float carpaint_base_metallic(const MaterialData material) {
    return clamp(material.carpaintBaseParams.x, 0.0f, 1.0f);
}

inline float carpaint_base_roughness(const MaterialData material) {
    return clamp(material.carpaintBaseParams.y, 0.0f, 1.0f);
}

inline float carpaint_flake_scale(const MaterialData material) {
    return max(material.carpaintBaseParams.z, 1.0e-4f);
}

inline float carpaint_flake_sample_weight(const MaterialData material) {
    return clamp(material.carpaintFlakeParams.x, 0.0f, 0.95f);
}

inline float carpaint_flake_roughness(const MaterialData material) {
    return clamp(material.carpaintFlakeParams.y, 0.0f, 1.0f);
}

inline float carpaint_flake_anisotropy(const MaterialData material) {
    return clamp(material.carpaintFlakeParams.z, -0.99f, 0.99f);
}

inline float carpaint_flake_normal_strength(const MaterialData material) {
    return clamp(material.carpaintFlakeParams.w, 0.0f, 1.0f);
}

inline float carpaint_coat_sample_weight(const MaterialData material) {
    return clamp(material.coatParams.z, 0.0f, 0.95f);
}

inline bool carpaint_has_base_conductor(const MaterialData material) {
    return (material.carpaintBaseEta.w > 0.0f || material.carpaintBaseK.w > 0.0f);
}

inline float3 carpaint_base_eta(const MaterialData material) {
    return max(material.carpaintBaseEta.xyz, float3(0.0f));
}

inline float3 carpaint_base_k(const MaterialData material) {
    return max(material.carpaintBaseK.xyz, float3(0.0f));
}

inline float3 fresnel_conductor(float cosThetaI, const float3 eta, const float3 k);

inline float3 carpaint_base_f0(const MaterialData material) {
    if (carpaint_has_base_conductor(material)) {
        return fresnel_conductor(1.0f, carpaint_base_eta(material), carpaint_base_k(material));
    }
    return clamp01(material.baseColorRoughness.xyz);
}

inline float3 carpaint_hash3(float3 p) {
    p = fract(p * 0.3183099f + float3(0.1f, 0.3f, 0.7f));
    p += dot(p, float3(p.y + 33.33f, p.z + 55.55f, p.x + 77.77f));
    return fract((p.xxy + p.yzz) * 13.5453123f);
}

inline float3 carpaint_flake_normal(const MaterialData material,
                                    const float3 position,
                                    const float3 normal) {
    float scale = carpaint_flake_scale(material);
    float3 samplePos = position * scale;
    float3 rand = carpaint_hash3(samplePos);
    float anis = carpaint_flake_anisotropy(material);
    float ax = max(1.0f - anis, 1.0e-3f);
    float ay = max(1.0f + anis, 1.0e-3f);
    float phi = 2.0f * kPi * rand.x;
    float r = sqrt(max(rand.y, 1.0e-4f));
    float x = r * cos(phi) * ax;
    float y = r * sin(phi) * ay;
    float m2 = clamp(x * x + y * y, 0.0f, 0.99f);
    float z = sqrt(max(1.0f - m2, 0.0f));
    float3 tangent;
    float3 bitangent;
    build_onb(normal, tangent, bitangent);
    float3 perturbed = normalize(x * tangent + y * bitangent + z * normal);
    float strength = carpaint_flake_normal_strength(material);
    return normalize(mix(normal, perturbed, strength));
}

inline CarpaintLobeResult carpaint_eval_coat(const MaterialData material,
                                             const float3 normal,
                                             const float3 wo,
                                             const float3 wi,
                                             const FireflyClampParams clampParams) {
    CarpaintLobeResult res;
    res.f = float3(0.0f);
    res.pdf = 0.0f;
    float cosThetaO = max(dot(normal, wo), 0.0f);
    float cosThetaI = max(dot(normal, wi), 0.0f);
    if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
        return res;
    }
    float roughness = plastic_coat_roughness(material);
    float alpha = max(roughness * roughness, 1.0e-4f);
    float3 wh = safe_normalize(wo + wi);
    if (dot(wh, normal) <= 0.0f || dot(wo, wh) <= 0.0f || dot(wi, wh) <= 0.0f) {
        return res;
    }
    float D = ggx_D(alpha, dot(normal, wh));
    float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
    float f0 = plastic_coat_f0(material);
    float3 F = schlick_fresnel(float3(f0), dot(wi, wh));
    float denom = 4.0f * cosThetaO * cosThetaI;
    float3 spec = F * (D * G / max(denom, 1.0e-6f));
    spec = clamp_specular_tail(spec * plastic_specular_tint(material), roughness, float3(f0), clampParams);
    float coatPdfRaw = ggx_pdf(alpha, normal, wo, wi);
    if (coatPdfRaw <= 0.0f) {
        return res;
    }
    res.pdf = clamp_specular_pdf(coatPdfRaw, clampParams);
    res.f = spec;
    return res;
}

inline CarpaintLobeResult carpaint_eval_flake(const MaterialData material,
                                              const float3 position,
                                              const float3 normal,
                                              const float3 wo,
                                              const float3 wi,
                                              const FireflyClampParams clampParams) {
    CarpaintLobeResult res;
    res.f = float3(0.0f);
    res.pdf = 0.0f;
    float3 flakeNormal = carpaint_flake_normal(material, position, normal);
    float cosThetaO = max(dot(flakeNormal, wo), 0.0f);
    float cosThetaI = max(dot(flakeNormal, wi), 0.0f);
    if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
        return res;
    }
    float flakeRoughness = max(carpaint_flake_roughness(material), 1.0e-3f);
    float alpha = flakeRoughness * flakeRoughness;
    float3 wh = safe_normalize(wo + wi);
    if (dot(wh, flakeNormal) <= 0.0f || dot(wo, wh) <= 0.0f || dot(wi, wh) <= 0.0f) {
        return res;
    }
    float D = ggx_D(alpha, dot(flakeNormal, wh));
    float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
    float3 F0 = carpaint_base_f0(material);
    float3 F = schlick_fresnel(F0, dot(wi, wh));
    float denom = 4.0f * cosThetaO * cosThetaI;
    float3 spec = F * (D * G / max(denom, 1.0e-6f));
    spec = clamp_specular_tail(spec * plastic_specular_tint(material), flakeRoughness, F0, clampParams);
    float coatAverage = clamp(material.coatParams.w, 0.0f, 1.0f);
    spec *= max(1.0f - coatAverage, 0.0f);
    float pdfRaw = ggx_pdf(alpha, flakeNormal, wo, wi);
    if (pdfRaw <= 0.0f) {
        return res;
    }
    res.pdf = clamp_specular_pdf(pdfRaw, clampParams);
    res.f = spec;
    return res;
}

inline CarpaintLobeResult carpaint_eval_base(const MaterialData material,
                                             const float3 normal,
                                             const float3 wo,
                                             const float3 wi,
                                             const FireflyClampParams clampParams) {
    CarpaintLobeResult res;
    res.f = float3(0.0f);
    res.pdf = 0.0f;
    float cosThetaO = max(dot(normal, wo), 0.0f);
    float cosThetaI = max(dot(normal, wi), 0.0f);
    if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
        return res;
    }
    float metallic = carpaint_base_metallic(material);
    float diffuseWeight = max(1.0f - metallic, 0.0f);
    float specWeight = max(metallic, 0.0f);
    if (diffuseWeight <= 1.0e-4f && specWeight <= 1.0e-4f) {
        return res;
    }

    float coatAverage = clamp(material.coatParams.w, 0.0f, 1.0f);
    float3 baseColor = material_base_color(material);
    float3 combined = float3(0.0f);
    float pdfDiffuse = 0.0f;
    float pdfSpec = 0.0f;

    if (diffuseWeight > 1.0e-4f) {
        float3 diffuse = baseColor / kPi;
        float3 coatTrans = plastic_diffuse_transmission(material, cosThetaI, cosThetaO);
        diffuse *= coatTrans * max(1.0f - coatAverage, 0.0f);
        diffuse = max(diffuse, float3(0.0f));
        combined += diffuseWeight * diffuse;
        pdfDiffuse = lambert_pdf(normal, wi);
    }

    if (specWeight > 1.0e-4f) {
        float roughness = max(carpaint_base_roughness(material), 1.0e-3f);
        float alpha = roughness * roughness;
        float3 wh = safe_normalize(wo + wi);
        if (dot(wh, normal) > 0.0f && dot(wo, wh) > 0.0f && dot(wi, wh) > 0.0f) {
            float D = ggx_D(alpha, dot(normal, wh));
            float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
            bool hasConductor = carpaint_has_base_conductor(material);
            float3 eta = carpaint_base_eta(material);
            float3 k = carpaint_base_k(material);
            float3 f0 = hasConductor ? fresnel_conductor(1.0f, eta, k)
                                     : clamp01(baseColor);
            float3 F = hasConductor ? fresnel_conductor(dot(wi, wh), eta, k)
                                    : schlick_fresnel(baseColor, dot(wi, wh));
            float denom = 4.0f * cosThetaO * cosThetaI;
            float3 spec = F * (D * G / max(denom, 1.0e-6f));
            spec = clamp_specular_tail(spec * plastic_specular_tint(material) * max(1.0f - coatAverage, 0.0f),
                                       roughness,
                                       f0,
                                       clampParams);
            spec = max(spec, float3(0.0f));
            combined += specWeight * spec;
            float pdfRaw = ggx_pdf(alpha, normal, wo, wi);
            if (pdfRaw > 0.0f) {
                pdfSpec = clamp_specular_pdf(pdfRaw, clampParams);
            }
        }
    }

    res.f = max(combined, float3(0.0f));
    res.pdf = diffuseWeight * pdfDiffuse + specWeight * pdfSpec;
    return res;
}

inline FireflyClampParams make_firefly_params(constant PathtraceUniforms& uniforms) {
    FireflyClampParams params;
    params.clampFactor = max(uniforms.fireflyClampFactor, 0.0f);
    params.clampFloor = max(uniforms.fireflyClampFloor, 0.0f);
    params.throughputClamp = max(uniforms.throughputClamp, 0.0f);
    params.specularTailClampBase = max(uniforms.specularTailClampBase, 0.0f);
    params.specularTailClampRoughnessScale = max(uniforms.specularTailClampRoughnessScale, 0.0f);
    params.minSpecularPdf = max(uniforms.minSpecularPdf, 0.0f);
    params.maxContribution = max(uniforms.fireflyClampMaxContribution, 0.0f);
    params.enabled = (uniforms.fireflyClampEnabled != 0u) ? 1.0f : 0.0f;
    return params;
}

inline float3 clamp_firefly_contribution(const float3 throughput,
                                         const float3 contribution,
                                         const FireflyClampParams params) {
    float3 combined = throughput * contribution;
    if (!all(isfinite(combined))) {
        return float3(0.0f);
    }

    float3 positive = max(combined, float3(0.0f));
    if (params.enabled < 0.5f) {
        return positive;
    }

    float lum = luminance_rgb(positive);
    float throughputLum = luminance_rgb(max(throughput, float3(0.0f)));
    float maxLum = max(throughputLum * params.clampFactor, params.clampFloor);
    if (params.maxContribution > 0.0f) {
        maxLum = max(maxLum, params.maxContribution);
    }

    if (lum > maxLum && lum > 0.0f) {
        float scale = maxLum / max(lum, 1e-6f);
        combined *= scale;
        positive = max(combined, float3(0.0f));
    }

    return positive;
}

inline float clamp_specular_pdf(const float pdf, const FireflyClampParams params) {
    if (!isfinite(pdf)) {
        return 0.0f;
    }
    if (pdf <= 0.0f) {
        return 0.0f;
    }
    if (params.minSpecularPdf <= 0.0f) {
        return pdf;
    }
    return max(pdf, params.minSpecularPdf);
}

inline float3 clamp_path_throughput(const float3 throughput, const FireflyClampParams params) {
    if (!all(isfinite(throughput))) {
        return float3(0.0f);
    }
    if (params.enabled < 0.5f || params.throughputClamp <= 0.0f) {
        return throughput;
    }
    float3 positive = max(throughput, float3(0.0f));
    float lum = luminance_rgb(positive);
    if (lum > params.throughputClamp && lum > 0.0f) {
        float scale = params.throughputClamp / max(lum, 1e-6f);
        return throughput * scale;
    }
    return throughput;
}

inline float3 clamp_specular_tail(const float3 value,
                                  const float roughness,
                                  const float3 f0,
                                  const FireflyClampParams params) {
    if (!all(isfinite(value))) {
        return float3(0.0f);
    }
    float3 positive = max(value, float3(0.0f));
    if (params.enabled < 0.5f) {
        return positive;
    }
    if (params.specularTailClampBase <= 0.0f && params.specularTailClampRoughnessScale <= 0.0f) {
        return positive;
    }
    float strength = max(max(f0.x, f0.y), f0.z);
    strength = max(strength, 1e-3f);
    float limit = (params.specularTailClampBase +
                   params.specularTailClampRoughnessScale * roughness) * strength;
    limit = max(limit, params.clampFloor);
    float lum = luminance_rgb(positive);
    if (lum > limit && lum > 0.0f) {
        float scale = limit / max(lum, 1e-6f);
        positive *= scale;
    }
    return positive;
}

inline float schlick_weight(const float cosTheta) {
    float m = clamp01(1.0f - cosTheta);
    float m2 = m * m;
    return m2 * m2 * m;
}

inline float3 schlick_fresnel(const float3 f0, const float cosTheta) {
    return f0 + (float3(1.0f) - f0) * schlick_weight(cosTheta);
}

inline float fresnel_dielectric_exact(float cosThetaI,
                                      float etaI,
                                      float etaT,
                                      thread float& outCosThetaT) {
    cosThetaI = clamp(cosThetaI, -1.0f, 1.0f);
    float absCosThetaI = fabs(cosThetaI);
    float sinThetaI2 = max(0.0f, 1.0f - absCosThetaI * absCosThetaI);
    float eta = etaI / etaT;
    float sinThetaT2 = eta * eta * sinThetaI2;

    if (sinThetaT2 >= 1.0f) {
        outCosThetaT = 0.0f;
        return 1.0f;  // Total internal reflection
    }

    float cosThetaT = safe_sqrt(1.0f - sinThetaT2);
    outCosThetaT = cosThetaT;

    float etaICosThetaI = etaI * absCosThetaI;
    float etaTCosThetaT = etaT * cosThetaT;

    float RsNum = etaICosThetaI - etaTCosThetaT;
    float RsDen = etaICosThetaI + etaTCosThetaT;
    float RpNum = etaT * absCosThetaI - etaI * cosThetaT;
    float RpDen = etaT * absCosThetaI + etaI * cosThetaT;

    float Rs = (RsNum / RsDen);
    float Rp = (RpNum / RpDen);
    return 0.5f * (Rs * Rs + Rp * Rp);
}

inline float3 fresnel_conductor(float cosThetaI, const float3 eta, const float3 k) {
    cosThetaI = clamp(cosThetaI, -1.0f, 1.0f);
    float cos2 = cosThetaI * cosThetaI;
    float sin2 = max(0.0f, 1.0f - cos2);

    float3 eta2 = eta * eta;
    float3 k2 = k * k;

    float3 t0 = eta2 - k2 - float3(sin2);
    float3 a2plusb2 = sqrt(max(t0 * t0 + 4.0f * eta2 * k2, float3(0.0f)));
    float3 a = sqrt(max(0.5f * (a2plusb2 + t0), float3(0.0f)));

    float3 term1 = a2plusb2 + float3(cos2);
    float3 term2 = 2.0f * float3(cosThetaI) * a;
    float3 Rs = (term1 - term2) / (term1 + term2);

    float3 term3 = float3(cos2) * a2plusb2 + float3(sin2 * sin2);
    float3 term4 = term2 * float3(sin2);
    float3 Rp = (term3 - term4) / (term3 + term4);

    return clamp01(0.5f * (Rs * Rs + Rp * Rp));
}

inline float ggx_lambda(const float alpha, const float cosTheta) {
    float absCosTheta = fabs(cosTheta);
    if (absCosTheta <= 0.0f) {
        return 0.0f;
    }
    float sinTheta = safe_sqrt(max(0.0f, 1.0f - absCosTheta * absCosTheta));
    if (sinTheta == 0.0f) {
        return 0.0f;
    }
    float tanTheta = sinTheta / absCosTheta;
    float a = alpha * tanTheta;
    return (-1.0f + sqrt(1.0f + a * a)) * 0.5f;
}

inline float ggx_G1(const float alpha, const float cosTheta) {
    return 1.0f / (1.0f + ggx_lambda(alpha, cosTheta));
}

inline float ggx_D(const float alpha, const float cosThetaH) {
    float absCosThetaH = fabs(cosThetaH);
    float a2 = alpha * alpha;
    float denom = absCosThetaH * absCosThetaH * (a2 - 1.0f) + 1.0f;
    return a2 / (kPi * denom * denom);
}

inline float ggx_pdf(const float alpha,
                     const float3 normal,
                     const float3 wo,
                     const float3 wi) {
    float3 wh = safe_normalize(wo + wi);
    float cosThetaH = dot(normal, wh);
    float dotWoWh = dot(wo, wh);
    float cosThetaO = dot(normal, wo);
    if (cosThetaO <= 0.0f || cosThetaH <= 0.0f || dotWoWh <= 0.0f) {
        return 0.0f;
    }
    float D = ggx_D(alpha, cosThetaH);
    float G1 = ggx_G1(alpha, cosThetaO);
    float denom = 4.0f * max(dotWoWh, 1e-6f);
    return D * G1 * cosThetaH / denom;
}

inline float ggx_vndf_pdf(const float alpha,
                          const float3 normal,
                          const float3 wo,
                          const float3 wh) {
    float cosThetaO = dot(normal, wo);
    float cosThetaH = dot(normal, wh);
    if (cosThetaO <= 0.0f || cosThetaH <= 0.0f) {
        return 0.0f;
    }
    float D = ggx_D(alpha, cosThetaH);
    float G1 = ggx_G1(alpha, cosThetaO);
    float denom = max(dot(wo, wh), 1.0e-6f);
    return D * G1 * cosThetaH / denom;
}

inline float3 sample_ggx_half_vector(const float3 normal,
                                     const float alpha,
                                     thread uint& state) {
    float u1 = rand_uniform(state);
    float u2 = rand_uniform(state);

    float phi = 2.0f * kPi * u1;
    float cosTheta = sqrt((1.0f - u2) / (1.0f + (alpha * alpha - 1.0f) * u2));
    float sinTheta = safe_sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));

    float3 hLocal = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    return safe_normalize(to_world(hLocal, normal));
}

inline float3 sample_ggx_vndf(const float3 normal,
                              const float3 wo,
                              const float roughness,
                              thread uint& state) {
    float3 woLocal = to_local(safe_normalize(wo), normal);
    woLocal.z = max(woLocal.z, 1.0e-6f);
    float alpha = max(roughness * roughness, 1.0e-4f);
    float3 Vh = safe_normalize(float3(alpha * woLocal.x, alpha * woLocal.y, woLocal.z));

    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = (lensq > 0.0f) ? float3(-Vh.y, Vh.x, 0.0f) * rsqrt(lensq)
                               : float3(1.0f, 0.0f, 0.0f);
    float3 T2 = cross(Vh, T1);

    float u1 = rand_uniform(state);
    float u2 = rand_uniform(state);
    float r = sqrt(u1);
    float phi = 2.0f * kPi * u2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.0f + Vh.z);
    float t2Adjusted = (1.0f - s) * safe_sqrt(max(0.0f, 1.0f - t1 * t1)) + s * t2;
    float t3 = safe_sqrt(max(0.0f, 1.0f - t1 * t1 - t2Adjusted * t2Adjusted));

    float3 Nh = t1 * T1 + t2Adjusted * T2 + t3 * Vh;
    float3 Ne = safe_normalize(float3(alpha * Nh.x, alpha * Nh.y, max(Nh.z, 0.0f)));
    return safe_normalize(to_world(Ne, normal));
}

inline float3 material_base_color(const MaterialData material) {
    return clamp01(material.baseColorRoughness.xyz);
}

inline float material_roughness(const MaterialData material) {
    float roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
    return max(roughness, 1e-3f);
}

inline bool material_has_conductor_ior(const MaterialData material) {
    return (material.conductorEta.w > 0.0f || material.conductorK.w > 0.0f ||
            any(material.conductorEta.xyz > float3(0.0f)) ||
            any(material.conductorK.xyz > float3(0.0f)));
}

inline float3 conductor_f0(const MaterialData material) {
    if (material_has_conductor_ior(material)) {
        return fresnel_conductor(1.0f, material.conductorEta.xyz, material.conductorK.xyz);
    }
    return clamp01(material_base_color(material));
}

inline bool material_is_plastic(const MaterialData material) {
    return static_cast<uint>(material.typeEta.x) == 4u;
}

inline float plastic_coat_ior(const MaterialData material) {
    return max(material.typeEta.y, 1.0f);
}

inline float plastic_coat_roughness(const MaterialData material) {
    float roughness = clamp(material.coatParams.x, 0.0f, 1.0f);
    return max(roughness, 1.0e-3f);
}

inline float environment_lighting_roughness(const MaterialData material) {
    uint type = static_cast<uint>(material.typeEta.x);
    switch (type) {
        case 1u: // Metal
        case 7u: // PBR Metallic-Roughness
            return clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        case 4u: // Plastic (use coat roughness for env lighting)
            return clamp(plastic_coat_roughness(material), 0.0f, 1.0f);
        case 6u: // CarPaint (use base roughness)
            return clamp(carpaint_base_roughness(material), 0.0f, 1.0f);
        default:
            return 1.0f;
    }
}

inline float plastic_coat_thickness(const MaterialData material) {
    return max(material.coatParams.y, 0.0f);
}

inline float plastic_coat_sample_weight(const MaterialData material) {
    return clamp(material.coatParams.z, 0.0f, 1.0f);
}

inline float plastic_coat_fresnel_average(const MaterialData material) {
    return clamp(material.coatParams.w, 0.0f, 1.0f);
}

inline float plastic_coat_f0(const MaterialData material) {
    float eta = plastic_coat_ior(material);
    float ratio = (eta - 1.0f) / max(eta + 1.0f, 1.0e-6f);
    float f0 = ratio * ratio;
    return clamp(f0, 0.0f, 0.999f);
}

inline float ior_from_f0(float f0) {
    float clamped = clamp(f0, 0.0f, 0.999f);
    float root = sqrt(clamped);
    float denom = max(1.0f - root, 1.0e-4f);
    return max((1.0f + root) / denom, 1.0f);
}

inline float3 plastic_coat_tint(const MaterialData material) {
    return clamp01(material.coatTint.xyz);
}

inline float3 plastic_coat_absorption(const MaterialData material) {
    return max(material.coatAbsorption.xyz, float3(0.0f));
}

inline float3 plastic_specular_tint(const MaterialData material) {
    float3 tint = plastic_coat_tint(material);
    float thickness = plastic_coat_thickness(material);
    if (thickness <= 0.0f) {
        return tint;
    }
    float3 absorption = plastic_coat_absorption(material);
    if (all(absorption <= float3(1.0e-6f))) {
        return tint;
    }
    return clamp01(tint * exp(-absorption * thickness));
}

inline float3 plastic_diffuse_transmission(const MaterialData material,
                                           const float cosThetaI,
                                           const float cosThetaO) {
    float thickness = plastic_coat_thickness(material);
    float3 tint = plastic_coat_tint(material);
    if (thickness <= 0.0f) {
        return tint;
    }
    float3 absorption = plastic_coat_absorption(material);
    float safeCosI = max(cosThetaI, 1.0e-3f);
    float safeCosO = max(cosThetaO, 1.0e-3f);
    float3 attenuationI = exp(-absorption * (thickness / safeCosI));
    float3 attenuationO = exp(-absorption * (thickness / safeCosO));
    return clamp01(tint * attenuationI * attenuationO);
}

inline bool material_is_subsurface(const MaterialData material) {
    return static_cast<uint>(material.typeEta.x) == 5u;
}

inline float3 sss_sigma_a(const MaterialData material,
                          const float3 baseColor,
                          float meanFreePath,
                          float anisotropy) {
    bool hasOverride = material.sssSigmaA.w > 0.5f;
    if (hasOverride) {
        float3 sigmaA = max(material.sssSigmaA.xyz, float3(1.0e-6f));
        return sigmaA;
    }
    float sigmaT = 1.0f / max(meanFreePath, 1.0e-4f);
    float3 sigmaS = clamp(baseColor, float3(0.0f), float3(0.999f)) * sigmaT;
    sigmaS = max(sigmaS, float3(0.0f));
    sigmaS *= max(1.0f - anisotropy, 0.01f);
    float3 sigmaA = max(float3(sigmaT) - sigmaS, float3(1.0e-6f));
    return sigmaA;
}

inline float3 sss_sigma_s_prime(const MaterialData material,
                                const float3 baseColor,
                                float meanFreePath,
                                float anisotropy) {
    bool hasOverride = material.sssSigmaA.w > 0.5f;
    if (hasOverride) {
        float3 sigmaS = max(material.sssSigmaS.xyz, float3(0.0f));
        sigmaS *= max(1.0f - anisotropy, 0.01f);
        return sigmaS;
    }
    float sigmaT = 1.0f / max(meanFreePath, 1.0e-4f);
    float3 sigmaS = clamp(baseColor, float3(0.0f), float3(0.999f)) * sigmaT;
    sigmaS = max(sigmaS, float3(0.0f));
    sigmaS *= max(1.0f - anisotropy, 0.01f);
    return sigmaS;
}

inline float3 normalized_diffusion_profile(const float radius,
                                           const float3 sigmaA,
                                           const float3 sigmaSPrime) {
    float3 sigmaTPrime = sigmaA + sigmaSPrime;
    float3 safeSigmaTPrime = max(sigmaTPrime, float3(1.0e-6f));
    float3 alphaPrime = clamp01(sigmaSPrime / safeSigmaTPrime);
    float3 D = 1.0f / max(3.0f * safeSigmaTPrime, float3(1.0e-6f));
    float3 sigmaTr = sqrt(max(sigmaA / D, float3(1.0e-6f)));
    float3 rVec = float3(max(radius, 1.0e-4f));
    float3 zr = 1.0f / safeSigmaTPrime;
    float3 dr = sqrt(rVec * rVec + zr * zr);
    float3 vr = zr + 4.0f * D;
    float3 dv = sqrt(rVec * rVec + vr * vr);
    float3 expDr = exp(-sigmaTr * dr);
    float3 expDv = exp(-sigmaTr * dv);
    float3 denomDr = max(dr * dr * dr, float3(1.0e-6f));
    float3 denomDv = max(dv * dv * dv, float3(1.0e-6f));
    float3 termDr = (zr * (float3(1.0f) + sigmaTr * dr)) / denomDr;
    float3 termDv = (vr * (float3(1.0f) + sigmaTr * dv)) / denomDv;
    float3 profile = (alphaPrime / (4.0f * kPi)) * (termDr * expDr + termDv * expDv);
    return max(profile, float3(0.0f));
}

inline float sss_sigma_tr_scalar(const float3 sigmaA,
                                 const float3 sigmaSPrime) {
    float3 sigmaTPrime = sigmaA + sigmaSPrime;
    float3 safeSigmaTPrime = max(sigmaTPrime, float3(1.0e-6f));
    float3 D = 1.0f / max(3.0f * safeSigmaTPrime, float3(1.0e-6f));
    float3 sigmaTr = sqrt(max(sigmaA / D, float3(1.0e-6f)));
    return max(luminance_rgb(sigmaTr), 1.0e-4f);
}

inline float sample_sss_radius(const float sigmaTrScalar, thread uint& state) {
    float u = rand_uniform(state);
    u = clamp(u, 1.0e-6f, 1.0f - 1.0e-6f);
    return -log(1.0f - u) / max(sigmaTrScalar, 1.0e-4f);
}

inline float pdf_sss_radius(const float radius, const float sigmaTrScalar) {
    if (radius <= 0.0f) {
        return 0.0f;
    }
    float sigma = max(sigmaTrScalar, 1.0e-4f);
    return sigma * exp(-sigma * radius);
}

inline float schlick_fresnel_scalar(const float f0, const float cosTheta) {
    float m = clamp01(1.0f - cosTheta);
    float m2 = m * m;
    float m5 = m2 * m2 * m;
    return f0 + (1.0f - f0) * m5;
}

inline float3 to_world_with_reference(const float3 local, const float3 reference) {
    float3 ref = safe_normalize(reference);
    float3 tangent;
    float3 bitangent;
    build_onb(ref, tangent, bitangent);
    return safe_normalize(local.x * tangent + local.y * bitangent + local.z * ref);
}

inline float3 sample_henyey_greenstein_local(const float g, thread uint& state) {
    float u1 = rand_uniform(state);
    float u2 = rand_uniform(state);
    float cosTheta = 0.0f;
    if (fabs(g) < 1.0e-3f) {
        cosTheta = 1.0f - 2.0f * u1;
    } else {
        float s = (1.0f - g * g) / (1.0f - g + 2.0f * g * u1);
        cosTheta = (1.0f + g * g - s * s) / (2.0f * g);
        cosTheta = clamp(cosTheta, -1.0f, 1.0f);
    }
    float sinTheta = safe_sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * kPi * u2;
    float3 local = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    return local;
}

inline float3 sample_henyey_greenstein_world(const float3 referenceDir,
                                             const float g,
                                             thread uint& state) {
    float3 local = sample_henyey_greenstein_local(g, state);
    return to_world_with_reference(local, referenceDir);
}

struct BsdfEvalResult {
    float3 value;
    float pdf;
    float directionalPdf;
    float areaPdf;
    bool isDelta;
    bool isBssrdf;
};

struct BsdfSampleResult {
    float3 direction;
    float3 weight;
    float pdf;
    float directionalPdf;
    float areaPdf;
    float3 exitPoint;
    float3 exitNormal;
    bool isDelta;
    bool isBssrdf;
    bool hasExitPoint;
    int mediumEvent;
    uint lobeType;
    float lobeRoughness;
};

inline BsdfSampleResult sample_sss_random_walk_software(constant PathtraceUniforms& uniforms,
                                                        const MaterialData material,
                                                        thread const HitRecord& rec,
                                                        const float3 wo,
                                                        const float3 incidentDir,
                                                        device const SphereData* spheres,
                                                        device const RectData* rectangles,
                                                        device const TriangleData* triangleData,
                                                        device const BvhNode* tlasNodes,
                                                        device const uint* tlasPrimIndices,
                                                        device const SoftwareInstanceInfo* instanceInfos,
                                                        device const BvhNode* blasNodes,
                                                        device const uint* blasPrimIndices,
                                                        device const BvhNode* nodes,
                                                        device const uint* primitiveIndices,
                                                        device PathtraceStats* stats,
                                                        thread uint& state,
                                                        const FireflyClampParams clampParams) {
    BsdfSampleResult result;
    result.direction = float3(0.0f);
    result.weight = float3(0.0f);
    result.pdf = 0.0f;
    result.directionalPdf = 0.0f;
    result.areaPdf = 0.0f;
    result.exitPoint = float3(0.0f);
    result.exitNormal = float3(0.0f);
    result.isDelta = false;
    result.isBssrdf = false;
    result.hasExitPoint = false;
    result.mediumEvent = 0;
    result.lobeType = 0u;
    result.lobeRoughness = 0.0f;

    if (rec.frontFace == 0u) {
        return result;
    }

    float pCoat = clamp(material.coatParams.z, 0.0f, 1.0f);
    float randLobe = rand_uniform(state);
    float coatRoughness = plastic_coat_roughness(material);
    float alpha = coatRoughness * coatRoughness;
    float f0 = plastic_coat_f0(material);
    float3 f0Color = float3(f0);
    float3 specTint = plastic_specular_tint(material);

    if (pCoat > 0.0f && randLobe < pCoat) {
        float3 wh = sample_ggx_vndf(rec.normal, wo, coatRoughness, state);
        if (dot(wh, rec.normal) <= 0.0f) {
            return result;
        }

        float3 wi = reflect(-wo, wh);
        wi = safe_normalize(wi);
        if (!all(isfinite(wi))) {
            return result;
        }

        float cosThetaI = dot(rec.normal, wi);
        float cosThetaO = dot(rec.normal, wo);
        if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
            return result;
        }

        float dotWiWh = dot(wi, wh);
        if (dotWiWh <= 0.0f) {
            return result;
        }

        float D = ggx_D(alpha, dot(rec.normal, wh));
        float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
        float3 F = schlick_fresnel(f0Color, dotWiWh);
        float denom = 4.0f * cosThetaO * cosThetaI;
        float3 spec = F * (D * G / max(denom, 1.0e-6f));
        spec = clamp_specular_tail(spec * specTint, coatRoughness, f0Color, clampParams);
        float specPdfRaw = ggx_pdf(alpha, rec.normal, wo, wi);
        if (specPdfRaw <= 0.0f) {
            return result;
        }
        float specPdf = clamp_specular_pdf(specPdfRaw, clampParams);
        float combinedPdf = max(pCoat * specPdf, 1.0e-6f);
        float3 weight = spec * cosThetaI / combinedPdf;
        weight = max(weight, float3(0.0f));
        if (!all(isfinite(weight))) {
            return result;
        }

        result.direction = wi;
        result.weight = weight;
        result.pdf = combinedPdf;
        result.directionalPdf = specPdf;
        result.areaPdf = 0.0f;
        result.isDelta = false;
        result.isBssrdf = false;
        result.hasExitPoint = false;
        return result;
    }

    float pDiffuse = max(1.0f - pCoat, 1.0e-3f);

    float anisotropy = clamp(material.sssSigmaS.w, -0.99f, 0.99f);
    float meanFreePath = max(material.sssParams.x, 1.0e-4f);
    float3 baseColor = material_base_color(material);
    float3 sigmaA = sss_sigma_a(material, baseColor, meanFreePath, anisotropy);
    float3 sigmaSPrime = sss_sigma_s_prime(material, baseColor, meanFreePath, anisotropy);
    float3 sigmaT = max(sigmaA + sigmaSPrime, float3(1.0e-6f));
    float sigmaTScalar = max(max(sigmaT.x, max(sigmaT.y, sigmaT.z)), 1.0e-4f);

    float3 throughput = float3(1.0f / pDiffuse);

    float etaOutside = 1.0f;
    float etaInside = max(material.typeEta.y, 1.0f);
    float3 entryNormal = rec.normal;
    float3 unitDir = incidentDir;
    float cosThetaI = dot(-unitDir, entryNormal);
    if (cosThetaI <= 0.0f) {
        return result;
    }
    float cosThetaT = 0.0f;
    float FrEntry = fresnel_dielectric_exact(cosThetaI, etaOutside, etaInside, cosThetaT);
    float3 enterDir = refract(unitDir, entryNormal, etaOutside / etaInside);
    if (!all(isfinite(enterDir)) || dot(enterDir, enterDir) <= 0.0f) {
        return result;
    }
    enterDir = safe_normalize(enterDir);

    float etaScaleEntry = (etaInside * etaInside) / (etaOutside * etaOutside);
    float directionScaleEntry = etaScaleEntry * (cosThetaT / max(cosThetaI, 1.0e-6f));
    throughput *= max(1.0f - FrEntry, 0.0f) * directionScaleEntry;
    if (material.sssParams.z > 0.5f) {
        throughput *= plastic_specular_tint(material);
    }

    float3 currentPos = offset_surface_point(rec.point, -entryNormal, enterDir);
    float3 currentDir = enterDir;

    uint maxSteps = max(uniforms.sssMaxSteps, 1u);
    for (uint step = 0u; step < maxSteps; ++step) {
        float xi = rand_uniform(state);
        xi = clamp(xi, 1.0e-6f, 1.0f - 1.0e-6f);
        float distance = -log(1.0f - xi) / sigmaTScalar;

        Ray boundaryRay;
        boundaryRay.origin = currentPos;
        boundaryRay.direction = currentDir;
        HitRecord boundaryRec;
        bool hitBoundary = trace_scene_software(uniforms,
                                                spheres,
                                                rectangles,
                                                triangleData,
                                                tlasNodes,
                                                tlasPrimIndices,
                                                instanceInfos,
                                                blasNodes,
                                                blasPrimIndices,
                                                nodes,
                                                primitiveIndices,
                                                stats,
                                                boundaryRay,
                                                kRayOriginEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/false,
                                                /*includeTriangles=*/true,
                                                boundaryRec);
        if (!hitBoundary) {
            break;
        }

        float boundaryDistance = max(boundaryRec.t, 1.0e-4f);
        if (distance < boundaryDistance) {
            float3 transmittance = exp(-sigmaT * distance);
            throughput *= transmittance;
            float3 scatterAlbedo = clamp01(sigmaSPrime / max(sigmaT, float3(1.0e-6f)));
            throughput *= scatterAlbedo;
            float throughputMax = max(throughput.x, max(throughput.y, throughput.z));
            if (throughputMax < kSssThroughputCutoff) {
                break;
            }
            currentPos += currentDir * distance;
            currentDir = sample_henyey_greenstein_world(-currentDir, anisotropy, state);
            if (!all(isfinite(currentDir)) || dot(currentDir, currentDir) <= 0.0f) {
                break;
            }
            currentDir = safe_normalize(currentDir);
            continue;
        }

        float travel = boundaryDistance;
        float3 transmittance = exp(-sigmaT * travel);
        throughput *= transmittance;
        float throughputMax = max(throughput.x, max(throughput.y, throughput.z));
        if (throughputMax < kSssThroughputCutoff) {
            break;
        }

        float3 exitPoint = boundaryRec.point;
        float3 outwardNormal = (boundaryRec.frontFace != 0u) ? boundaryRec.normal : -boundaryRec.normal;
        if (!all(isfinite(outwardNormal)) || dot(outwardNormal, outwardNormal) <= 0.0f) {
            break;
        }
        outwardNormal = safe_normalize(outwardNormal);

        float etaI = etaInside;
        float etaT = 1.0f;
        float cosExitI = dot(-currentDir, outwardNormal);
        if (cosExitI <= 0.0f) {
            currentPos = exitPoint;
            currentDir = reflect(currentDir, outwardNormal);
            currentDir = safe_normalize(currentDir);
            continue;
        }

        float cosExitT = 0.0f;
        float FrExit = fresnel_dielectric_exact(cosExitI, etaI, etaT, cosExitT);
        float3 refracted = refract(currentDir, outwardNormal, etaI / etaT);
        if (!all(isfinite(refracted)) || dot(refracted, refracted) <= 0.0f) {
            currentPos = exitPoint;
            currentDir = reflect(currentDir, outwardNormal);
            currentDir = safe_normalize(currentDir);
            continue;
        }
        refracted = safe_normalize(refracted);

        float etaScaleExit = (etaT * etaT) / (etaI * etaI);
        float directionScaleExit = etaScaleExit * (cosExitT / max(cosExitI, 1.0e-6f));
        throughput *= max(1.0f - FrExit, 0.0f) * directionScaleExit;
        if (material.sssParams.z > 0.5f) {
            throughput *= plastic_specular_tint(material);
        }
        throughput = max(throughput, float3(0.0f));
        if (!all(isfinite(throughput))) {
            break;
        }

        result.direction = refracted;
        result.weight = throughput;
        result.pdf = max(pDiffuse, 1.0e-4f);
        result.directionalPdf = 1.0f;
        result.areaPdf = 0.0f;
        result.exitPoint = exitPoint;
        result.exitNormal = outwardNormal;
        result.isDelta = false;
        result.isBssrdf = true;
        result.hasExitPoint = true;
        return result;
    }

    result.pdf = 0.0f;
    result.direction = float3(0.0f);
    result.weight = float3(0.0f);
    return result;
}

#if __METAL_VERSION__ >= 310
inline BsdfSampleResult sample_sss_random_walk_hardware(constant PathtraceUniforms& uniforms,
                                                        const MaterialData material,
                                                        thread const HitRecord& rec,
                                                        const float3 wo,
                                                        const float3 incidentDir,
                                                        acceleration_structure<instancing> accel,
                                                        device const MeshInfo* meshInfos,
                                                        device const TriangleData* triangleData,
                                                        device const SceneVertex* sceneVertices,
                                                        device const uint3* meshIndices,
                                                        device const uint* instanceUserIds,
                                                        device const SphereData* spheres,
                                                        device const RectData* rectangles,
                                                        device const BvhNode* nodes,
                                                        device const uint* primitiveIndices,
                                                        device PathtraceStats* stats,
                                                        thread uint& state,
                                                        const FireflyClampParams clampParams) {
    BsdfSampleResult result;
    result.direction = float3(0.0f);
    result.weight = float3(0.0f);
    result.pdf = 0.0f;
    result.directionalPdf = 0.0f;
    result.areaPdf = 0.0f;
    result.exitPoint = float3(0.0f);
    result.exitNormal = float3(0.0f);
    result.isDelta = false;
    result.isBssrdf = false;
    result.hasExitPoint = false;
    result.mediumEvent = 0;
    result.lobeType = 0u;
    result.lobeRoughness = 0.0f;

    if (rec.frontFace == 0u) {
        return result;
    }

    float pCoat = clamp(material.coatParams.z, 0.0f, 1.0f);
    float randLobe = rand_uniform(state);
    float coatRoughness = plastic_coat_roughness(material);
    float alpha = coatRoughness * coatRoughness;
    float f0 = plastic_coat_f0(material);
    float3 f0Color = float3(f0);
    float3 specTint = plastic_specular_tint(material);

    if (pCoat > 0.0f && randLobe < pCoat) {
        float3 wh = sample_ggx_vndf(rec.normal, wo, coatRoughness, state);
        if (dot(wh, rec.normal) <= 0.0f) {
            return result;
        }

        float3 wi = reflect(-wo, wh);
        wi = safe_normalize(wi);
        if (!all(isfinite(wi))) {
            return result;
        }

        float cosThetaI = dot(rec.normal, wi);
        float cosThetaO = dot(rec.normal, wo);
        if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
            return result;
        }

        float dotWiWh = dot(wi, wh);
        if (dotWiWh <= 0.0f) {
            return result;
        }

        float D = ggx_D(alpha, dot(rec.normal, wh));
        float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
        float3 F = schlick_fresnel(f0Color, dotWiWh);
        float denom = 4.0f * cosThetaO * cosThetaI;
        float3 spec = F * (D * G / max(denom, 1.0e-6f));
        spec = clamp_specular_tail(spec * specTint, coatRoughness, f0Color, clampParams);
        float specPdfRaw = ggx_pdf(alpha, rec.normal, wo, wi);
        if (specPdfRaw <= 0.0f) {
            return result;
        }
        float specPdf = clamp_specular_pdf(specPdfRaw, clampParams);
        float combinedPdf = max(pCoat * specPdf, 1.0e-6f);
        float3 weight = spec * cosThetaI / combinedPdf;
        weight = max(weight, float3(0.0f));
        if (!all(isfinite(weight))) {
            return result;
        }

        result.direction = wi;
        result.weight = weight;
        result.pdf = combinedPdf;
        result.directionalPdf = specPdf;
        result.areaPdf = 0.0f;
        result.isDelta = false;
        result.isBssrdf = false;
        result.hasExitPoint = false;
        return result;
    }

    float pDiffuse = max(1.0f - pCoat, 1.0e-3f);

    float anisotropy = clamp(material.sssSigmaS.w, -0.99f, 0.99f);
    float meanFreePath = max(material.sssParams.x, 1.0e-4f);
    float3 baseColor = material_base_color(material);
    float3 sigmaA = sss_sigma_a(material, baseColor, meanFreePath, anisotropy);
    float3 sigmaSPrime = sss_sigma_s_prime(material, baseColor, meanFreePath, anisotropy);
    float3 sigmaT = max(sigmaA + sigmaSPrime, float3(1.0e-6f));
    float sigmaTScalar = max(max(sigmaT.x, max(sigmaT.y, sigmaT.z)), 1.0e-4f);

    float3 throughput = float3(1.0f / pDiffuse);

    float etaOutside = 1.0f;
    float etaInside = max(material.typeEta.y, 1.0f);
    float3 entryNormal = rec.normal;
    float3 unitDir = incidentDir;
    float cosThetaI = dot(-unitDir, entryNormal);
    if (cosThetaI <= 0.0f) {
        return result;
    }
    float cosThetaT = 0.0f;
    float FrEntry = fresnel_dielectric_exact(cosThetaI, etaOutside, etaInside, cosThetaT);
    float3 enterDir = refract(unitDir, entryNormal, etaOutside / etaInside);
    if (!all(isfinite(enterDir)) || dot(enterDir, enterDir) <= 0.0f) {
        return result;
    }
    enterDir = safe_normalize(enterDir);

    float etaScaleEntry = (etaInside * etaInside) / (etaOutside * etaOutside);
    float directionScaleEntry = etaScaleEntry * (cosThetaT / max(cosThetaI, 1.0e-6f));
    throughput *= max(1.0f - FrEntry, 0.0f) * directionScaleEntry;
    if (material.sssParams.z > 0.5f) {
        throughput *= plastic_specular_tint(material);
    }

    float3 currentPos = offset_surface_point(rec.point, -entryNormal, enterDir);
    float3 currentDir = enterDir;

    uint maxSteps = max(uniforms.sssMaxSteps, 1u);
    for (uint step = 0u; step < maxSteps; ++step) {
        float xi = rand_uniform(state);
        xi = clamp(xi, 1.0e-6f, 1.0f - 1.0e-6f);
        float distance = -log(1.0f - xi) / sigmaTScalar;

        Ray boundaryRay;
        boundaryRay.origin = currentPos;
        boundaryRay.direction = currentDir;
        HitRecord boundaryRec;
        uint excludeMesh;
        uint excludePrim;
        compute_exclusion_indices(rec, excludeMesh, excludePrim);
        bool hitBoundary = trace_scene_hardware(uniforms,
                                                accel,
                                                meshInfos,
                                                triangleData,
                                                sceneVertices,
                                                meshIndices,
                                                instanceUserIds,
                                                spheres,
                                                rectangles,
                                                nodes,
                                                primitiveIndices,
                                                stats,
                                                boundaryRay,
                                                kRayOriginEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/false,
                                                excludeMesh,
                                                excludePrim,
                                                boundaryRec);
        if (!hitBoundary) {
            break;
        }

        float boundaryDistance = max(boundaryRec.t, 1.0e-4f);
        if (distance < boundaryDistance) {
            float3 transmittance = exp(-sigmaT * distance);
            throughput *= transmittance;
            float3 scatterAlbedo = clamp01(sigmaSPrime / max(sigmaT, float3(1.0e-6f)));
            throughput *= scatterAlbedo;
            float throughputMax = max(throughput.x, max(throughput.y, throughput.z));
            if (throughputMax < kSssThroughputCutoff) {
                break;
            }
            currentPos += currentDir * distance;
            currentDir = sample_henyey_greenstein_world(-currentDir, anisotropy, state);
            if (!all(isfinite(currentDir)) || dot(currentDir, currentDir) <= 0.0f) {
                break;
            }
            currentDir = safe_normalize(currentDir);
            continue;
        }

        float travel = boundaryDistance;
        float3 transmittance = exp(-sigmaT * travel);
        throughput *= transmittance;
        float throughputMax = max(throughput.x, max(throughput.y, throughput.z));
        if (throughputMax < kSssThroughputCutoff) {
            break;
        }

        float3 exitPoint = boundaryRec.point;
        float3 outwardNormal = (boundaryRec.frontFace != 0u) ? boundaryRec.normal : -boundaryRec.normal;
        if (!all(isfinite(outwardNormal)) || dot(outwardNormal, outwardNormal) <= 0.0f) {
            break;
        }
        outwardNormal = safe_normalize(outwardNormal);

        float etaI = etaInside;
        float etaT = 1.0f;
        float cosExitI = dot(-currentDir, outwardNormal);
        if (cosExitI <= 0.0f) {
            currentPos = exitPoint;
            currentDir = reflect(currentDir, outwardNormal);
            currentDir = safe_normalize(currentDir);
            continue;
        }

        float cosExitT = 0.0f;
        float FrExit = fresnel_dielectric_exact(cosExitI, etaI, etaT, cosExitT);
        float3 refracted = refract(currentDir, outwardNormal, etaI / etaT);
        if (!all(isfinite(refracted)) || dot(refracted, refracted) <= 0.0f) {
            currentPos = exitPoint;
            currentDir = reflect(currentDir, outwardNormal);
            currentDir = safe_normalize(currentDir);
            continue;
        }
        refracted = safe_normalize(refracted);

        float etaScaleExit = (etaT * etaT) / (etaI * etaI);
        float directionScaleExit = etaScaleExit * (cosExitT / max(cosExitI, 1.0e-6f));
        throughput *= max(1.0f - FrExit, 0.0f) * directionScaleExit;
        if (material.sssParams.z > 0.5f) {
            throughput *= plastic_specular_tint(material);
        }
        throughput = max(throughput, float3(0.0f));
        if (!all(isfinite(throughput))) {
            break;
        }

        result.direction = refracted;
        result.weight = throughput;
        result.pdf = max(pDiffuse, 1.0e-4f);
        result.directionalPdf = 1.0f;
        result.areaPdf = 0.0f;
        result.exitPoint = exitPoint;
        result.exitNormal = outwardNormal;
        result.isDelta = false;
        result.isBssrdf = true;
        result.hasExitPoint = true;
        return result;
    }

    result.pdf = 0.0f;
    result.direction = float3(0.0f);
    result.weight = float3(0.0f);
    return result;
}
#endif

inline bool material_is_delta(const MaterialData material) {
    uint type = static_cast<uint>(material.typeEta.x);
    if (type == 2u) {
        return true;
    }
    if (type == 1u) {
        float roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        return roughness <= 1e-3f;
    }
    if (type == 7u) {
        float roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        return roughness <= 1e-3f;
    }
    if (material_is_plastic(material)) {
        return false;
    }
    return false;
}

inline bool material_is_thin_dielectric(const MaterialData material) {
    uint type = static_cast<uint>(material.typeEta.x);
    return (type == 2u) && (material.typeEta.w > 0.5f);
}

inline float max_component(const float3 v) {
    return max(v.x, max(v.y, v.z));
}

inline float dielectric_f0_from_ior(const float ior) {
    float eta = max(ior, 1.0f);
    float num = eta - 1.0f;
    float den = max(eta + 1.0f, 1.0e-6f);
    float f0 = (num / den) * (num / den);
    return clamp(f0, 0.0f, 0.99f);
}

inline float pbr_specular_weight(const float3 f0) {
    return clamp(max_component(f0), 0.05f, 0.95f);
}

inline float2 dfg_approx(const float roughness, const float NoV) {
    const float4 c0 = float4(-1.0f, -0.0275f, -0.572f, 0.022f);
    const float4 c1 = float4(1.0f, 0.0425f, 1.04f, -0.04f);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28f * NoV)) * r.x + r.y;
    return float2(-1.04f, 1.04f) * a004 + r.zw;
}

inline float3 specular_energy_compensation(const float3 f0,
                                           const float roughness,
                                           const float NoV) {
    float NoVClamped = clamp(NoV, 0.0f, 1.0f);
    float2 dfg = dfg_approx(roughness, NoVClamped);
    float3 Fss = clamp(f0 * dfg.x + dfg.y, float3(0.0f), float3(0.99f));
    float3 Favg = f0 + (float3(1.0f) - f0) * (1.0f / 21.0f);
    float3 oneMinusFss = clamp(float3(1.0f) - Fss, float3(0.0f), float3(1.0f));
    float3 denom = max(float3(1.0f) - Favg * oneMinusFss, float3(1.0e-3f));
    float3 Fms = (Favg * oneMinusFss) / denom;
    float3 scale = (Fss + Fms) / max(Fss, float3(1.0e-4f));
    return clamp(scale, float3(1.0f), float3(2.0f));
}

inline BsdfEvalResult evaluate_pbr_metallic_roughness(const MaterialData material,
                                                      const float3 normal,
                                                      const float3 wo,
                                                      const float3 wi,
                                                      const FireflyClampParams clampParams,
                                                      const float diffuseOcclusion,
                                                      const bool specularOnly) {
    BsdfEvalResult result;
    result.value = float3(0.0f);
    result.pdf = 0.0f;
    result.isDelta = false;
    result.directionalPdf = 0.0f;
    result.areaPdf = 0.0f;
    result.isBssrdf = false;

    float cosThetaO = dot(normal, wo);
    float cosThetaI = dot(normal, wi);
    float absCosThetaO = fabs(cosThetaO);
    float absCosThetaI = fabs(cosThetaI);
    if (absCosThetaO <= 0.0f || absCosThetaI <= 0.0f) {
        return result;
    }

    float3 baseColor = clamp01(material.baseColorRoughness.xyz);
    float metallic = clamp(material.pbrParams.x, 0.0f, 1.0f);
    float roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
    float dielectricF0 = dielectric_f0_from_ior(material.typeEta.y);
    float3 f0 = mix(float3(dielectricF0), baseColor, metallic);
    float3 diffuseColor = baseColor * (1.0f - metallic);
    diffuseColor *= clamp(diffuseOcclusion, 0.0f, 1.0f);
    if (specularOnly) {
        diffuseColor = float3(0.0f);
    }

    float transmission = clamp(material.pbrExtras.z, 0.0f, 1.0f) * (1.0f - metallic);
    float reflectScale = 1.0f - transmission;
    float specWeightBase = specularOnly ? 1.0f : pbr_specular_weight(f0);
    float wSpec = specWeightBase * reflectScale;
    float wDiff = specularOnly ? 0.0f : (1.0f - specWeightBase) * reflectScale;
    float wTrans = transmission;
    float weightSum = wSpec + wDiff + wTrans;
    if (weightSum <= 0.0f) {
        return result;
    }
    float pSpec = wSpec / weightSum;
    float pDiff = wDiff / weightSum;
    float pTrans = wTrans / weightSum;

    if (cosThetaO * cosThetaI > 0.0f) {
        if (cosThetaO <= 0.0f || cosThetaI <= 0.0f) {
            return result;
        }
        float alpha = max(roughness * roughness, 1.0e-4f);
        float3 wh = safe_normalize(wo + wi);
        if (dot(wh, normal) <= 0.0f || dot(wo, wh) <= 0.0f || dot(wi, wh) <= 0.0f) {
            return result;
        }
        float D = ggx_D(alpha, dot(normal, wh));
        float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
        float3 F = schlick_fresnel(f0, dot(wi, wh));
        float denom = 4.0f * cosThetaO * cosThetaI;
        float3 spec = F * (D * G / max(denom, 1.0e-6f));
        spec *= specular_energy_compensation(f0, roughness, absCosThetaO);
        spec = clamp_specular_tail(spec, roughness, f0, clampParams);
        spec *= reflectScale;
        float pdfSpec = ggx_pdf(alpha, normal, wo, wi);

        float3 diffuse = (diffuseColor / kPi) * reflectScale;
        float pdfDiffuse = lambert_pdf(normal, wi);

        float pdf = pSpec * pdfSpec + pDiff * pdfDiffuse;
        if (pdf > 0.0f) {
            result.value = max(spec + diffuse, float3(0.0f));
            result.pdf = clamp_specular_pdf(pdf, clampParams);
            result.directionalPdf = result.pdf;
        }
        return result;
    }

    if (wTrans <= 0.0f) {
        return result;
    }

    float etaI = 1.0f;
    float etaT = max(material.typeEta.y, 1.0f);
    if (cosThetaO < 0.0f) {
        float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
    }
    float eta = etaI / etaT;
    float alpha = max(roughness * roughness, 1.0e-4f);
    float3 wh = safe_normalize(wo + wi * eta);
    if (!all(isfinite(wh)) || dot(wh, wh) <= 0.0f) {
        return result;
    }
    if (dot(wh, normal) <= 0.0f) {
        wh = -wh;
    }
    float cosThetaOWh = dot(wo, wh);
    float cosThetaIWh = dot(wi, wh);
    if (cosThetaOWh * cosThetaIWh > 0.0f) {
        return result;
    }

    float D = ggx_D(alpha, max(dot(normal, wh), 0.0f));
    float G = ggx_G1(alpha, absCosThetaO) * ggx_G1(alpha, absCosThetaI);
    float cosThetaT = 0.0f;
    float F = fresnel_dielectric_exact(cosThetaOWh, etaI, etaT, cosThetaT);
    float denom = cosThetaOWh + eta * cosThetaIWh;
    float denomSq = denom * denom;
    if (fabs(denomSq) <= 1.0e-8f) {
        return result;
    }
    float factor = (eta * eta) * fabs(cosThetaIWh) * fabs(cosThetaOWh);
    factor /= max(absCosThetaO * absCosThetaI * denomSq, 1.0e-6f);
    float3 ft = (1.0f - F) * D * G * factor;
    ft *= transmission_tint(material, absCosThetaI);
    ft *= transmission;

    float pdfWh = ggx_vndf_pdf(alpha, normal, wo, wh);
    float dwhDwi = fabs((eta * eta * cosThetaIWh) / max(denomSq, 1.0e-8f));
    float pdfTrans = pdfWh * dwhDwi;
    float pdf = pTrans * pdfTrans;
    if (pdf > 0.0f) {
        result.value = max(ft, float3(0.0f));
        result.pdf = clamp_specular_pdf(pdf, clampParams);
        result.directionalPdf = result.pdf;
    }
    return result;
}

inline BsdfSampleResult sample_pbr_metallic_roughness(const MaterialData material,
                                                      const float3 normal,
                                                      const float3 wo,
                                                      const float3 incidentDir,
                                                      thread uint& state,
                                                      const FireflyClampParams clampParams,
                                                      const float diffuseOcclusion,
                                                      const bool specularOnly) {
    BsdfSampleResult result;
    result.direction = float3(0.0f);
    result.weight = float3(0.0f);
    result.pdf = 0.0f;
    result.isDelta = false;
    result.directionalPdf = 0.0f;
    result.areaPdf = 0.0f;
    result.exitPoint = float3(0.0f);
    result.exitNormal = float3(0.0f);
    result.isBssrdf = false;
    result.hasExitPoint = false;
    result.mediumEvent = 0;
    result.lobeType = 0u;
    result.lobeRoughness = 0.0f;
    result.lobeType = 0u;
    result.lobeRoughness = 0.0f;

    float3 baseColor = clamp01(material.baseColorRoughness.xyz);
    float metallic = clamp(material.pbrParams.x, 0.0f, 1.0f);
    float roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
    float dielectricF0 = dielectric_f0_from_ior(material.typeEta.y);
    float3 f0 = mix(float3(dielectricF0), baseColor, metallic);
    float3 diffuseColor = baseColor * (1.0f - metallic);
    diffuseColor *= clamp(diffuseOcclusion, 0.0f, 1.0f);
    if (specularOnly) {
        diffuseColor = float3(0.0f);
    }

    float transmission = clamp(material.pbrExtras.z, 0.0f, 1.0f) * (1.0f - metallic);
    float reflectScale = 1.0f - transmission;
    float specWeightBase = specularOnly ? 1.0f : pbr_specular_weight(f0);
    float wSpec = specWeightBase * reflectScale;
    float wDiff = specularOnly ? 0.0f : (1.0f - specWeightBase) * reflectScale;
    float wTrans = transmission;
    float weightSum = wSpec + wDiff + wTrans;
    if (weightSum <= 0.0f) {
        return result;
    }
    float pSpec = wSpec / weightSum;
    float pDiff = wDiff / weightSum;
    float pTrans = wTrans / weightSum;
    float choose = rand_uniform(state);

    float3 wi = float3(0.0f);
    float pdfSpec = 0.0f;
    float pdfDiffuse = 0.0f;
    float pdfTrans = 0.0f;
    float3 f = float3(0.0f);

    if (choose < pSpec) {
        result.lobeType = 1u;
        result.lobeRoughness = roughness;
        if (roughness <= 1.0e-3f) {
            wi = reflect(incidentDir, normal);
            float cosThetaI = dot(normal, wi);
            if (cosThetaI <= 0.0f) {
                return result;
            }
            pdfSpec = 1.0f;
            float cosThetaO = max(dot(normal, wo), 0.0f);
            float3 F = schlick_fresnel(f0, cosThetaO);
            f = F * reflectScale;
            result.isDelta = true;
        } else {
            float3 wh = sample_ggx_vndf(normal, wo, roughness, state);
            wi = reflect(-wo, wh);
            float cosThetaI = dot(normal, wi);
            if (cosThetaI <= 0.0f) {
                return result;
            }
            float alpha = max(roughness * roughness, 1.0e-4f);
            float D = ggx_D(alpha, dot(normal, wh));
            float G = ggx_G1(alpha, max(dot(normal, wo), 0.0f)) * ggx_G1(alpha, cosThetaI);
            float3 F = schlick_fresnel(f0, dot(wi, wh));
            float denom = 4.0f * max(dot(normal, wo), 0.0f) * cosThetaI;
            f = F * (D * G / max(denom, 1.0e-6f));
            f *= specular_energy_compensation(f0, roughness, max(dot(normal, wo), 0.0f));
            f = clamp_specular_tail(f, roughness, f0, clampParams);
            f *= reflectScale;
            pdfSpec = ggx_pdf(alpha, normal, wo, wi);
        }
    } else if (choose < (pSpec + pDiff)) {
        result.lobeType = 0u;
        result.lobeRoughness = 1.0f;
        float3 local = sample_cosine_hemisphere(state);
        wi = safe_normalize(to_world(local, normal));
        float cosThetaI = dot(normal, wi);
        if (cosThetaI <= 0.0f) {
            return result;
        }
        f = (diffuseColor / kPi) * reflectScale;
        pdfDiffuse = lambert_pdf(normal, wi);
    } else {
        result.lobeType = 2u;
        result.lobeRoughness = roughness;
        float cosThetaO = dot(normal, wo);
        float absCosThetaO = fabs(cosThetaO);
        float etaI = 1.0f;
        float etaT = max(material.typeEta.y, 1.0f);
        if (cosThetaO < 0.0f) {
            float tmp = etaI;
            etaI = etaT;
            etaT = tmp;
        }
        float eta = etaI / etaT;
        if (roughness <= 1.0e-3f) {
            wi = refract(-wo, normal, eta);
            float dirLen2 = dot(wi, wi);
            if (dirLen2 <= 0.0f) {
                return result;
            }
            wi = wi * rsqrt(dirLen2);
            float cosThetaI = dot(normal, wi);
            float cosThetaT = 0.0f;
            float Fr = fresnel_dielectric_exact(cosThetaO, etaI, etaT, cosThetaT);
            float etaScale = (etaT * etaT) / (etaI * etaI);
            float directionScale = etaScale * (fabs(cosThetaT) / max(absCosThetaO, 1.0e-6f));
            float3 ft = float3(max(1.0f - Fr, 0.0f) * directionScale);
            ft *= transmission_tint(material, fabs(cosThetaI));
            f = transmission * ft;
            pdfTrans = 1.0f;
            result.isDelta = true;
        } else {
            float3 wh = sample_ggx_vndf(normal, wo, roughness, state);
            wi = refract(-wo, wh, eta);
            float dirLen2 = dot(wi, wi);
            if (dirLen2 <= 0.0f) {
                return result;
            }
            wi = wi * rsqrt(dirLen2);
            if (dot(wi, normal) * cosThetaO >= 0.0f) {
                return result;
            }
            float cosThetaI = dot(normal, wi);
            float absCosThetaI = fabs(cosThetaI);
            float cosThetaOWh = dot(wo, wh);
            float cosThetaIWh = dot(wi, wh);
            if (cosThetaOWh * cosThetaIWh > 0.0f) {
                return result;
            }
            float alpha = max(roughness * roughness, 1.0e-4f);
            float D = ggx_D(alpha, max(dot(normal, wh), 0.0f));
            float G = ggx_G1(alpha, absCosThetaO) * ggx_G1(alpha, absCosThetaI);
            float cosThetaT = 0.0f;
            float F = fresnel_dielectric_exact(cosThetaOWh, etaI, etaT, cosThetaT);
            float denom = cosThetaOWh + eta * cosThetaIWh;
            float denomSq = denom * denom;
            if (fabs(denomSq) <= 1.0e-8f) {
                return result;
            }
            float factor = (eta * eta) * fabs(cosThetaIWh) * fabs(cosThetaOWh);
            factor /= max(absCosThetaO * absCosThetaI * denomSq, 1.0e-6f);
            float3 ft = (1.0f - F) * D * G * factor;
            ft *= transmission_tint(material, absCosThetaI);
            f = transmission * ft;
            float pdfWh = ggx_vndf_pdf(alpha, normal, wo, wh);
            float dwhDwi = fabs((eta * eta * cosThetaIWh) / max(denomSq, 1.0e-8f));
            pdfTrans = pdfWh * dwhDwi;
        }
    }

    float cosThetaI = dot(normal, wi);
    float absCosThetaI = fabs(cosThetaI);
    if (absCosThetaI <= 0.0f) {
        return result;
    }

    float pdf = pSpec * pdfSpec + pDiff * pdfDiffuse + pTrans * pdfTrans;
    if (pdf <= 0.0f) {
        return result;
    }
    result.direction = wi;
    result.pdf = pdf;
    result.directionalPdf = pdf;
    result.weight = max(f * absCosThetaI / pdf, float3(0.0f));
    return result;
}

inline BsdfEvalResult evaluate_bsdf(const MaterialData material,
                                    const float3 position,
                                    const float3 normal,
                                    const float3 wo,
                                    const float3 wi,
                                    const FireflyClampParams clampParams,
                                    const uint sssMode,
                                    const float diffuseOcclusion,
                                    const bool specularOnly) {
    BsdfEvalResult result;
    result.value = float3(0.0f);
    result.pdf = 0.0f;
    result.isDelta = false;
    result.directionalPdf = 0.0f;
    result.areaPdf = 0.0f;
    result.isBssrdf = false;

    float cosThetaO = max(dot(normal, wo), 0.0f);
    float cosThetaI = max(dot(normal, wi), 0.0f);
    if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
        return result;
    }

    uint type = static_cast<uint>(material.typeEta.x);
    switch (type) {
        case 0u: { // Lambertian
            if (specularOnly) {
                return result;
            }
            float3 albedo = material_base_color(material);
            albedo *= clamp(diffuseOcclusion, 0.0f, 1.0f);
            result.value = albedo / kPi;
            result.pdf = lambert_pdf(normal, wi);
            result.directionalPdf = result.pdf;
            break;
        }
        case 1u: { // Metal (GGX)
            float roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
            if (roughness <= 1e-3f) {
                result.isDelta = true;
                return result;
            }
            float alpha = roughness * roughness;
            float3 wh = safe_normalize(wo + wi);
            if (dot(wh, normal) <= 0.0f || dot(wo, wh) <= 0.0f || dot(wi, wh) <= 0.0f) {
                return result;
            }

            float D = ggx_D(alpha, dot(normal, wh));
            float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
            float3 eta = material.conductorEta.xyz;
            float3 k = material.conductorK.xyz;
            float3 f0 = conductor_f0(material);
            float3 F;
            if (material_has_conductor_ior(material)) {
                F = fresnel_conductor(dot(wi, wh), eta, k);
            } else {
                F = schlick_fresnel(f0, dot(wi, wh));
            }
            float denom = 4.0f * cosThetaO * cosThetaI;
            float3 spec = F * (D * G / max(denom, 1e-6f));
            spec *= specular_energy_compensation(f0, roughness, cosThetaO);
            spec = clamp_specular_tail(spec, roughness, f0, clampParams);
            float pdf = ggx_pdf(alpha, normal, wo, wi);
            if (pdf <= 0.0f) {
                result.value = float3(0.0f);
                result.pdf = 0.0f;
                result.directionalPdf = 0.0f;
            } else {
                result.value = max(spec, float3(0.0f));
                result.pdf = clamp_specular_pdf(pdf, clampParams);
                result.directionalPdf = result.pdf;
            }
            break;
        }
        case 2u: { // Dielectric
            result.isDelta = true;
            break;
        }
        case 4u: { // Plastic (Diffuse base + Clearcoat)
            float coatRoughness = plastic_coat_roughness(material);
            float alpha = coatRoughness * coatRoughness;
            float f0 = plastic_coat_f0(material);
            float3 f0Color = float3(f0);

            float3 spec = float3(0.0f);
            float pdfSpec = 0.0f;
            float3 wh = safe_normalize(wo + wi);
            if (dot(wh, normal) > 0.0f && dot(wo, wh) > 0.0f && dot(wi, wh) > 0.0f) {
                float D = ggx_D(alpha, dot(normal, wh));
                float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
                float3 F = schlick_fresnel(f0Color, dot(wi, wh));
                float denom = 4.0f * cosThetaO * cosThetaI;
                spec = F * (D * G / max(denom, 1e-6f));
                spec = clamp_specular_tail(spec, coatRoughness, f0Color, clampParams);
                spec *= plastic_specular_tint(material);
                float rawPdf = ggx_pdf(alpha, normal, wo, wi);
                if (rawPdf > 0.0f) {
                    pdfSpec = clamp_specular_pdf(rawPdf, clampParams);
                }
                spec = max(spec, float3(0.0f));
            }

            float3 F_i = schlick_fresnel(f0Color, cosThetaI);
            float3 F_o = schlick_fresnel(f0Color, cosThetaO);
            float3 tint = plastic_diffuse_transmission(material, cosThetaI, cosThetaO);
            float3 diffuse = material_base_color(material) / kPi;
            diffuse *= clamp(diffuseOcclusion, 0.0f, 1.0f);
            diffuse *= tint;
            diffuse *= (float3(1.0f) - F_i) * (float3(1.0f) - F_o);
            diffuse *= max(1.0f - plastic_coat_fresnel_average(material), 0.0f);
            diffuse = max(diffuse, float3(0.0f));
            if (specularOnly) {
                diffuse = float3(0.0f);
            }

            float pdfDiffuse = lambert_pdf(normal, wi);
            float pCoat = clamp(plastic_coat_sample_weight(material), 0.0f, 1.0f);
            float pDiffuse = 1.0f - pCoat;
            if (specularOnly) {
                pCoat = 1.0f;
                pDiffuse = 0.0f;
            }
            result.value = spec + diffuse;
            result.pdf = pCoat * pdfSpec + pDiffuse * pdfDiffuse;
            result.directionalPdf = result.pdf;
            break;
        }
        case 5u: { // Subsurface scattering (handled via BSSRDF sampling)
            result.isBssrdf = true;
            result.value = float3(0.0f);
            result.pdf = 0.0f;
            result.directionalPdf = 0.0f;
            break;
        }
        case 6u: { // CarPaint (base + flakes + clearcoat)
            float cosThetaO = max(dot(normal, wo), 0.0f);
            float cosThetaI = max(dot(normal, wi), 0.0f);
            if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
                return result;
            }

            float pCoat = carpaint_coat_sample_weight(material);
            float pFlake = carpaint_flake_sample_weight(material);
            float pBase = max(1.0f - (pCoat + pFlake), 0.0f);
            float norm = pCoat + pFlake + pBase;
            if (norm <= 1.0e-6f) {
                pBase = 1.0f;
                pCoat = 0.0f;
                pFlake = 0.0f;
                norm = 1.0f;
            }
            pCoat /= norm;
            pFlake /= norm;
            pBase /= norm;

            CarpaintLobeResult coatRes = carpaint_eval_coat(material, normal, wo, wi, clampParams);
            CarpaintLobeResult flakeRes = carpaint_eval_flake(material, position, normal, wo, wi, clampParams);
            CarpaintLobeResult baseRes = carpaint_eval_base(material, normal, wo, wi, clampParams);

            result.value = pBase * baseRes.f + pFlake * flakeRes.f + pCoat * coatRes.f;
            result.pdf = pBase * baseRes.pdf + pFlake * flakeRes.pdf + pCoat * coatRes.pdf;
            result.directionalPdf = result.pdf;
            result.areaPdf = 0.0f;
            break;
        }
        case 7u: { // PBR Metallic-Roughness
            result = evaluate_pbr_metallic_roughness(material,
                                                     normal,
                                                     wo,
                                                     wi,
                                                     clampParams,
                                                     diffuseOcclusion,
                                                     specularOnly);
            break;
        }
        default:
            break;
    }

    if (result.pdf <= 0.0f || !all(isfinite(result.value))) {
        result.value = float3(0.0f);
    }
    return result;
}

inline BsdfSampleResult sample_bsdf(const MaterialData material,
                                    const float3 position,
                                    const float3 normal,
                                    const float3 wo,
                                    const float3 incidentDir,
                                    bool frontFace,
                                    thread uint& state,
                                    const FireflyClampParams clampParams,
                                    const uint sssMode,
                                    const float diffuseOcclusion,
                                    const bool specularOnly) {
    BsdfSampleResult result;
    result.direction = float3(0.0f);
    result.weight = float3(0.0f);
    result.pdf = 0.0f;
    result.isDelta = false;
    result.directionalPdf = 0.0f;
    result.areaPdf = 0.0f;
    result.exitPoint = float3(0.0f);
    result.exitNormal = float3(0.0f);
    result.isBssrdf = false;
    result.hasExitPoint = false;
    result.mediumEvent = 0;
    result.lobeType = 0u;
    result.lobeRoughness = 0.0f;

    uint type = static_cast<uint>(material.typeEta.x);
    switch (type) {
        case 0u: { // Lambertian
            if (specularOnly) {
                return result;
            }
            float3 local = sample_cosine_hemisphere(state);
            float3 wi = safe_normalize(to_world(local, normal));
            float cosThetaI = dot(normal, wi);
            if (cosThetaI <= 0.0f) {
                return result;
            }

            float pdf = lambert_pdf(normal, wi);
            if (pdf <= 0.0f) {
                return result;
            }

            float3 albedo = material_base_color(material);
            albedo *= clamp(diffuseOcclusion, 0.0f, 1.0f);
            float3 f = albedo / kPi;
            float3 weight = f * cosThetaI / pdf;
            weight = max(weight, float3(0.0f));
            if (!all(isfinite(weight))) {
                return result;
            }

            result.direction = wi;
            result.weight = weight;
            result.pdf = pdf;
            result.directionalPdf = pdf;
            result.isDelta = false;
            result.lobeType = 0u;
            result.lobeRoughness = 1.0f;
            break;
        }
        case 1u: { // Metal (GGX)
            float roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
            float3 eta = material.conductorEta.xyz;
            float3 k = material.conductorK.xyz;
            float3 f0 = conductor_f0(material);
            float3 wi;
            float3 F;
            if (roughness <= 1e-3f) {
                wi = reflect(incidentDir, normal);
                float cosThetaI = dot(normal, wi);
                if (cosThetaI <= 0.0f) {
                    return result;
                }
                float cosThetaO = dot(normal, wo);
                float cosTheta = max(cosThetaO, 0.0f);
                if (material_has_conductor_ior(material)) {
                    F = fresnel_conductor(cosTheta, eta, k);
                } else {
                    F = schlick_fresnel(f0, cosTheta);
                }
                result.direction = wi;
                result.weight = F;
                result.pdf = 1.0f;
                result.directionalPdf = 1.0f;
                result.isDelta = true;
                result.lobeType = 1u;
                result.lobeRoughness = roughness;
                break;
            }

            float alpha = roughness * roughness;
            float3 wh = sample_ggx_vndf(normal, wo, roughness, state);
            if (dot(wh, normal) <= 0.0f) {
                return result;
            }

            wi = reflect(-wo, wh);
            wi = safe_normalize(wi);

            if (!all(isfinite(wi))) {
                return result;
            }

            float cosThetaI = dot(normal, wi);
            float cosThetaO = dot(normal, wo);
            if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
                return result;
            }

            float dotWoWh = dot(wo, wh);
            if (dotWoWh <= 0.0f) {
                return result;
            }

            float D = ggx_D(alpha, dot(normal, wh));
            float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
            if (material_has_conductor_ior(material)) {
                F = fresnel_conductor(dot(wi, wh), eta, k);
            } else {
                F = schlick_fresnel(f0, dot(wi, wh));
            }

            float denom = 4.0f * cosThetaO * cosThetaI;
            float3 f = F * (D * G / max(denom, 1e-6f));
            f *= specular_energy_compensation(f0, roughness, cosThetaO);
            f = clamp_specular_tail(f, roughness, f0, clampParams);
            float pdf = ggx_pdf(alpha, normal, wo, wi);
            if (pdf <= 0.0f) {
                return result;
            }

            float clampedPdf = clamp_specular_pdf(pdf, clampParams);
            float3 weight = f * cosThetaI / clampedPdf;
            weight = max(weight, float3(0.0f));
            if (!all(isfinite(weight))) {
                return result;
            }
            result.direction = wi;
            result.weight = weight;
            result.pdf = clampedPdf;
            result.directionalPdf = clampedPdf;
            result.isDelta = false;
            result.lobeType = 1u;
            result.lobeRoughness = roughness;
            break;
        }
        case 4u: { // Plastic (Diffuse + Clearcoat)
            float cosThetaO = dot(normal, wo);
            if (cosThetaO <= 0.0f) {
                return result;
            }

            float coatRoughness = plastic_coat_roughness(material);
            float alpha = coatRoughness * coatRoughness;
            float f0 = plastic_coat_f0(material);
            float3 f0Color = float3(f0);
            float pCoat = clamp(plastic_coat_sample_weight(material), 0.0f, 1.0f);
            float pDiffuse = 1.0f - pCoat;
            float fresnelAvg = plastic_coat_fresnel_average(material);
            float3 specTint = plastic_specular_tint(material);
            if (specularOnly) {
                pCoat = 1.0f;
                pDiffuse = 0.0f;
            }

            float selector = rand_uniform(state);
            bool sampleCoat = (selector < pCoat) && (pCoat > 0.0f);

            if (sampleCoat) {
                float3 wh = sample_ggx_vndf(normal, wo, coatRoughness, state);
                if (dot(wh, normal) <= 0.0f) {
                    return result;
                }

                float3 wi = reflect(-wo, wh);
                wi = safe_normalize(wi);
                float cosThetaI = dot(normal, wi);
                if (cosThetaI <= 0.0f) {
                    return result;
                }

                float dotWiWh = dot(wi, wh);
                if (dotWiWh <= 0.0f) {
                    return result;
                }

                float D = ggx_D(alpha, dot(normal, wh));
                float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
                float3 F = schlick_fresnel(f0Color, dotWiWh);
                float denom = 4.0f * cosThetaO * cosThetaI;
                float3 spec = F * (D * G / max(denom, 1e-6f));
                spec = clamp_specular_tail(spec, coatRoughness, f0Color, clampParams);
                spec *= specTint;

                float specPdfRaw = ggx_pdf(alpha, normal, wo, wi);
                float specPdf = (specPdfRaw > 0.0f) ? clamp_specular_pdf(specPdfRaw, clampParams) : 0.0f;
                float diffusePdf = lambert_pdf(normal, wi);
                float combinedPdf = pCoat * specPdf + pDiffuse * diffusePdf;
                if (combinedPdf <= 0.0f) {
                    return result;
                }

                float3 weight = spec * cosThetaI / combinedPdf;
                if (!all(isfinite(weight))) {
                    return result;
                }

                result.direction = wi;
                result.weight = max(weight, float3(0.0f));
                result.pdf = combinedPdf;
                result.directionalPdf = combinedPdf;
                result.isDelta = false;
                result.lobeType = 1u;
                result.lobeRoughness = coatRoughness;
                break;
            }

            float3 local = sample_cosine_hemisphere(state);
            float3 wi = safe_normalize(to_world(local, normal));
            float cosThetaI = dot(normal, wi);
            if (cosThetaI <= 0.0f) {
                return result;
            }

            float3 base = material_base_color(material);
            float3 diffuse = base / kPi;
            diffuse *= clamp(diffuseOcclusion, 0.0f, 1.0f);
            float3 tintThrough = plastic_diffuse_transmission(material, cosThetaI, cosThetaO);
            float3 F_i = schlick_fresnel(f0Color, cosThetaI);
            float3 F_o = schlick_fresnel(f0Color, cosThetaO);
            diffuse *= tintThrough;
            diffuse *= (float3(1.0f) - F_i) * (float3(1.0f) - F_o);
            diffuse *= max(1.0f - fresnelAvg, 0.0f);
            diffuse = max(diffuse, float3(0.0f));
            if (specularOnly) {
                diffuse = float3(0.0f);
            }

            float diffusePdf = lambert_pdf(normal, wi);
            float specPdfRaw = ggx_pdf(alpha, normal, wo, wi);
            float specPdf = (specPdfRaw > 0.0f) ? clamp_specular_pdf(specPdfRaw, clampParams) : 0.0f;
            float combinedPdf = pCoat * specPdf + pDiffuse * diffusePdf;
            if (combinedPdf <= 0.0f) {
                return result;
            }

            float3 weight = diffuse * cosThetaI / combinedPdf;
            if (!all(isfinite(weight))) {
                return result;
            }

            result.direction = wi;
            result.weight = max(weight, float3(0.0f));
            result.pdf = combinedPdf;
            result.directionalPdf = combinedPdf;
            result.isDelta = false;
            result.lobeType = 0u;
            result.lobeRoughness = 1.0f;
            break;
        }
        case 5u: { // Subsurface scattering (separable diffusion)
            if (specularOnly) {
                return result;
            }
            bool useSeparable = (sssMode != 0u) &&
                                (material.sssParams.y < 0.5f) &&
                                (sssMode != 2u);
            float meanFreePath = max(material.sssParams.x, 1.0e-4f);
            useSeparable = useSeparable && (meanFreePath > 1.0e-4f);

            if (useSeparable) {
                float anisotropy = clamp(material.sssSigmaS.w, -0.99f, 0.99f);
                float3 baseColor = material_base_color(material);
                float3 sigmaA = sss_sigma_a(material, baseColor, meanFreePath, anisotropy);
                float3 sigmaSPrime = sss_sigma_s_prime(material, baseColor, meanFreePath, anisotropy);
                float sigmaTrScalar = sss_sigma_tr_scalar(sigmaA, sigmaSPrime);
                if (sigmaTrScalar <= 0.0f) {
                    useSeparable = false;
                } else {
                    float radius = sample_sss_radius(sigmaTrScalar, state);
                    radius = min(radius, meanFreePath * 10.0f);
                    float pdfRadius = pdf_sss_radius(radius, sigmaTrScalar);
                    if (pdfRadius <= 0.0f || !isfinite(pdfRadius)) {
                        useSeparable = false;
                    } else {
                        float phi = 2.0f * kPi * rand_uniform(state);
                        float sinPhi = sin(phi);
                        float cosPhi = cos(phi);
                        float3 tangent;
                        float3 bitangent;
                        build_onb(normal, tangent, bitangent);
                        float2 disp = radius * float2(cosPhi, sinPhi);
                        float3 exitPoint = position + tangent * disp.x + bitangent * disp.y;
                        float3 exitNormal = normal;

                        float3 localDir = sample_cosine_hemisphere(state);
                        float3 wi = safe_normalize(to_world(localDir, exitNormal));
                        float cosThetaExit = dot(exitNormal, wi);
                        float pdfDir = lambert_pdf(exitNormal, wi);
                        float pdfArea = pdfRadius / (2.0f * kPi * max(radius, 1.0e-4f));

                        if (cosThetaExit <= 0.0f || pdfDir <= 0.0f || pdfArea <= 0.0f) {
                            useSeparable = false;
                        } else {
                            float3 profile = normalized_diffusion_profile(radius, sigmaA, sigmaSPrime);
                            float3 coatTint = plastic_coat_tint(material);
                            float coatAverage = 1.0f - clamp(material.coatParams.w, 0.0f, 1.0f);
                            float coatTransmission = 1.0f;
                            if (material.sssParams.z > 0.5f) {
                                float coatIor = max(material.typeEta.z, 1.0f);
                                float f0 = ((coatIor - 1.0f) / (coatIor + 1.0f));
                                f0 *= f0;
                                float cosIn = max(dot(normal, wo), 0.0f);
                                float cosOut = cosThetaExit;
                                float transIn = 1.0f - schlick_fresnel_scalar(f0, cosIn);
                                float transOut = 1.0f - schlick_fresnel_scalar(f0, cosOut);
                                coatTransmission = clamp(transIn * transOut, 0.0f, 1.0f);
                                profile *= coatTint;
                            }

                            float3 weight = profile * cosThetaExit * coatAverage * coatTransmission;
                            float denom = max(pdfArea * pdfDir, 1.0e-6f);
                            weight = max(weight / denom, float3(0.0f));

                            if (!all(isfinite(weight))) {
                                useSeparable = false;
                            } else {
                                result.direction = wi;
                                result.weight = weight;
                                result.pdf = denom;
                                result.directionalPdf = pdfDir;
                                result.areaPdf = pdfArea;
                                result.exitPoint = exitPoint;
                                result.exitNormal = exitNormal;
                                result.isDelta = false;
                                result.isBssrdf = true;
                                result.hasExitPoint = true;
                                break;
                            }
                        }
                    }
                }
            }

            // Fallback to diffuse lambertian if separable SSS is disabled or failed
            float3 local = sample_cosine_hemisphere(state);
            float3 wi = safe_normalize(to_world(local, normal));
            float cosThetaI = dot(normal, wi);
            if (cosThetaI <= 0.0f) {
                return result;
            }
            float pdf = lambert_pdf(normal, wi);
            if (pdf <= 0.0f) {
                return result;
            }
            float3 albedo = material_base_color(material);
            float3 weight = (albedo / kPi) * cosThetaI / pdf;
            if (!all(isfinite(weight))) {
                return result;
            }
            result.direction = wi;
            result.weight = max(weight, float3(0.0f));
            result.pdf = pdf;
            result.directionalPdf = pdf;
            result.areaPdf = 0.0f;
            result.isDelta = false;
            result.isBssrdf = false;
            result.hasExitPoint = false;
            break;
        }
        case 6u: { // CarPaint (base + flakes + clearcoat)
            float pCoat = carpaint_coat_sample_weight(material);
            float pFlake = carpaint_flake_sample_weight(material);
            float pBase = max(1.0f - (pCoat + pFlake), 0.0f);
            float norm = pCoat + pFlake + pBase;
            if (norm <= 1.0e-6f) {
                pBase = 1.0f;
                pCoat = 0.0f;
                pFlake = 0.0f;
                norm = 1.0f;
            }
            pCoat /= norm;
            pFlake /= norm;
            pBase /= norm;

            float r = rand_uniform(state);
            uint lobe = 0u; // 0 = base, 1 = flake, 2 = coat
            uint selectedLobeType = 0u;
            float selectedLobeRoughness = 0.0f;
            float thresholdCoat = pCoat;
            float thresholdFlake = pCoat + pFlake;
            if (pCoat > 0.0f && r < thresholdCoat) {
                lobe = 2u;
            } else if (pFlake > 0.0f && r < thresholdFlake) {
                lobe = 1u;
            } else {
                lobe = 0u;
                if (pBase <= 1.0e-6f) {
                    if (pFlake > pCoat && pFlake > 0.0f) {
                        lobe = 1u;
                    } else if (pCoat > 0.0f) {
                        lobe = 2u;
                    }
                }
            }

            float3 wi;
            if (lobe == 2u) {
                float coatRoughness = plastic_coat_roughness(material);
                float3 wh = sample_ggx_vndf(normal, wo, coatRoughness, state);
                if (dot(wh, normal) <= 0.0f) {
                    return result;
                }
                wi = reflect(-wo, wh);
                wi = safe_normalize(wi);
                selectedLobeType = 1u;
                selectedLobeRoughness = coatRoughness;
            } else if (lobe == 1u) {
                float flakeRoughness = max(carpaint_flake_roughness(material), 1.0e-3f);
                float3 flakeNormal = carpaint_flake_normal(material, position, normal);
                float3 wh = sample_ggx_vndf(flakeNormal, wo, flakeRoughness, state);
                if (dot(wh, flakeNormal) <= 0.0f) {
                    return result;
                }
                wi = reflect(-wo, wh);
                wi = safe_normalize(wi);
                selectedLobeType = 1u;
                selectedLobeRoughness = flakeRoughness;
            } else {
                float metallic = carpaint_base_metallic(material);
                float diffuseWeight = max(1.0f - metallic, 0.0f);
                float specWeight = max(metallic, 0.0f);
                float weightSum = diffuseWeight + specWeight;
                float choose = rand_uniform(state);
                bool sampleSpec = (specWeight > 0.0f) && (weightSum > 0.0f) &&
                                  (choose < specWeight / max(weightSum, 1.0e-6f));
                if (sampleSpec) {
                    float baseRough = max(carpaint_base_roughness(material), 1.0e-3f);
                    float3 wh = sample_ggx_vndf(normal, wo, baseRough, state);
                    if (dot(wh, normal) <= 0.0f) {
                        return result;
                    }
                    wi = reflect(-wo, wh);
                    wi = safe_normalize(wi);
                    selectedLobeType = 1u;
                    selectedLobeRoughness = baseRough;
                } else {
                    float3 local = sample_cosine_hemisphere(state);
                    wi = safe_normalize(to_world(local, normal));
                    selectedLobeType = 0u;
                    selectedLobeRoughness = 1.0f;
                }
            }

            if (!all(isfinite(wi)) || dot(normal, wi) <= 0.0f) {
                return result;
            }

            CarpaintLobeResult coatRes = carpaint_eval_coat(material, normal, wo, wi, clampParams);
            CarpaintLobeResult flakeRes = carpaint_eval_flake(material, position, normal, wo, wi, clampParams);
            CarpaintLobeResult baseRes = carpaint_eval_base(material, normal, wo, wi, clampParams);

            float combinedPdf = pBase * baseRes.pdf + pFlake * flakeRes.pdf + pCoat * coatRes.pdf;
            if (combinedPdf <= 0.0f) {
                return result;
            }

            float3 selectedF = baseRes.f;
            float selectedPdf = baseRes.pdf;
            if (lobe == 1u) {
                selectedF = flakeRes.f;
                selectedPdf = flakeRes.pdf;
            } else if (lobe == 2u) {
                selectedF = coatRes.f;
                selectedPdf = coatRes.pdf;
            }
            if (selectedPdf <= 0.0f || !any(selectedF > float3(0.0f))) {
                return result;
            }
            float cosThetaI = max(dot(normal, wi), 0.0f);
            if (cosThetaI <= 0.0f) {
                return result;
            }
            float3 weight = selectedF * cosThetaI / combinedPdf;
            if (!all(isfinite(weight))) {
                return result;
            }
            result.direction = wi;
            result.weight = max(weight, float3(0.0f));
            result.pdf = combinedPdf;
            result.directionalPdf = max(selectedPdf, 0.0f);
            result.areaPdf = 0.0f;
            result.isDelta = false;
            result.isBssrdf = false;
            result.hasExitPoint = false;
            result.lobeType = selectedLobeType;
            result.lobeRoughness = selectedLobeRoughness;
            break;
        }
        case 7u: { // PBR Metallic-Roughness
            return sample_pbr_metallic_roughness(material,
                                                 normal,
                                                 wo,
                                                 incidentDir,
                                                 state,
                                                 clampParams,
                                                 diffuseOcclusion,
                                                 specularOnly);
        }
        case 2u: { // Dielectric
            result.isDelta = true;
            bool isThin = material_is_thin_dielectric(material);
            float refIdx = max(material.typeEta.y, 1.0f);
            float etaI = 1.0f;
            float etaT = refIdx;
            float3 unitDir = incidentDir;
            float cosThetaO = dot(-unitDir, normal);
            cosThetaO = clamp(cosThetaO, -1.0f, 1.0f);
            if (!isThin && !frontFace) {
                etaI = refIdx;
                etaT = 1.0f;
            }
            float relativeEta = etaI / etaT;
            float cosThetaT = 0.0f;
            float Fr = fresnel_dielectric_exact(cosThetaO, etaI, etaT, cosThetaT);

            float3 direction;
            float3 weight;
            if (rand_uniform(state) < Fr) {
                direction = reflect(unitDir, normal);
                weight = float3(Fr);
            } else {
                direction = refract(unitDir, normal, relativeEta);
                float dirLen2 = dot(direction, direction);
                if (dirLen2 <= 0.0f) {
                    direction = reflect(unitDir, normal);
                    weight = float3(Fr);
                } else {
                    direction = direction / sqrt(dirLen2);
                    float etaScale = (etaT * etaT) / (etaI * etaI);
                    float cosThetaI = fabs(cosThetaO);
                    float cosThetaTrans = fabs(cosThetaT);
                    float directionScale = etaScale * (cosThetaTrans / max(cosThetaI, 1e-6f));
                    weight = float3(max(1.0f - Fr, 0.0f) * directionScale);
                    if (!isThin) {
                        result.mediumEvent = frontFace ? 1 : -1;
                    }
                }
            }

            result.direction = safe_normalize(direction);
            result.weight = weight;
            result.pdf = 1.0f;
            result.directionalPdf = 1.0f;
            result.lobeType = 1u;
            result.lobeRoughness = 0.0f;
            break;
        }
        default:
            break;
    }

    return result;
}

inline float bsdf_cone_spread_increment(uint lobeType, float roughness, bool isDelta) {
    if (isDelta) {
        return 0.0f;
    }
    float clampedRoughness = clamp(roughness, 0.0f, 1.0f);
    if (lobeType == 0u) {  // diffuse
        return 0.55f;
    }
    if (lobeType == 1u) {  // glossy/specular microfacet
        return mix(0.03f, 0.45f, clampedRoughness);
    }
    return mix(0.10f, 0.60f, clampedRoughness);
}

inline float3 trace_path_software(constant PathtraceUniforms& uniforms,
                                  device const SphereData* spheres,
                                  device const RectData* rectangles,
                                  device const TriangleData* triangleData,
                                  device const MaterialData* materials,
                                  device const MeshInfo* meshInfos,
                                  device const SceneVertex* sceneVertices,
                                  device const uint3* meshIndices,
                                  Ray ray,
                                  const PrimaryRayDiff primaryRayDiff,
                                  thread uint& state,
                                  // TLAS/BLAS resources (software)
                                  device const BvhNode* tlasNodes,
                                  device const uint* tlasPrimIndices,
                                  device const BvhNode* blasNodes,
                                  device const uint* blasPrimIndices,
                                  device const SoftwareInstanceInfo* instanceInfos,
                                  device const BvhNode* nodes,
                                  device const uint* primitiveIndices,
                                  device PathtraceStats* stats,
                                  texture2d<float, access::sample> environmentTexture,
                                  array<texture2d<float, access::sample>, kMaxMaterialTextures> materialTextures,
                                  array<sampler, kMaxMaterialSamplers> materialSamplers,
                                  device const MaterialTextureInfo* materialTextureInfos,
                                  device const EnvironmentAliasEntry* environmentConditionalAlias,
                                  device const EnvironmentAliasEntry* environmentMarginalAlias,
                                  device const float* environmentPdf,
                                  // Optional AOV outputs
                                  thread float3* outFirstHitAlbedo = nullptr,
                                  thread float3* outFirstHitNormal = nullptr,
                                  thread PathtraceDebugContext* debugContext = nullptr) {
    float3 throughput = float3(1.0f, 1.0f, 1.0f);
    float3 radiance = float3(0.0f, 0.0f, 0.0f);
    float lastBsdfPdf = 1.0f;
    bool lastScatterWasDelta = true;
    bool isFirstHit = true;
    uint specularDepth = 0u;
    bool hadTransmission = false;
    float envLod = 0.0f;
    bool envLodActive = false;
    RayCone rayCone = make_primary_ray_cone(uniforms);
    HitRecord prevRec;
    bool prevValid = false;
    uint rectLightCount = (rectangles && uniforms.rectangleCount > 0 && materials)
                              ? count_rect_lights(uniforms, rectangles, materials)
                              : 0u;
    const bool envSampling = environment_sampling_available(uniforms,
                                                            environmentConditionalAlias,
                                                            environmentMarginalAlias,
                                                            environmentPdf);
    FireflyClampParams clampParams = make_firefly_params(uniforms);
    constexpr uint kMaxMediumStack = 8u;
    float3 mediumSigmaStack[kMaxMediumStack];
    for (uint i = 0; i < kMaxMediumStack; ++i) {
        mediumSigmaStack[i] = float3(0.0f);
    }
    uint mediumDepth = 0u;

    for (uint depth = 0; depth < uniforms.maxDepth; ++depth) {
        HitRecord rec;
        uint excludeMeshIndex = kInvalidIndex;
        uint excludePrimitiveIndex = kInvalidIndex;
        if (prevValid) {
            compute_exclusion_indices(prevRec, excludeMeshIndex, excludePrimitiveIndex);
        }
        if (!trace_scene_software_with_exclusion(uniforms,
                          spheres,
                          rectangles,
                          triangleData,
                          tlasNodes,
                          tlasPrimIndices,
                          instanceInfos,
                          blasNodes,
                          blasPrimIndices,
                          nodes,
                          primitiveIndices,
                          stats,
                          ray,
                          kEpsilon,
                          kInfinity,
                          excludeMeshIndex,
                          excludePrimitiveIndex,
                          rec)) {
            if (uniforms.debugViewMode != kDebugViewNone) {
                return float3(0.0f);
            }
            float3 background = sky_color(ray.direction);
            if (uniforms.backgroundMode == 1u) {
                background = uniforms.backgroundColor;
            } else if (uniforms.backgroundMode == 2u && environmentTexture.get_width() > 0 && environmentTexture.get_height() > 0) {
                float overrideLod = 0.0f;
                bool useOverride = environment_mip_override(uniforms, environmentTexture, overrideLod);
                if (useOverride) {
                    background = environment_color_lod(environmentTexture,
                                                       ray.direction,
                                                       uniforms.environmentRotation,
                                                       uniforms.environmentIntensity,
                                                       overrideLod,
                                                       uniforms);
                } else if (envLodActive) {
                    background = environment_color_lod(environmentTexture,
                                                       ray.direction,
                                                       uniforms.environmentRotation,
                                                       uniforms.environmentIntensity,
                                                       envLod,
                                                       uniforms);
                } else {
                    background = environment_color(environmentTexture,
                                                   ray.direction,
                                                   uniforms.environmentRotation,
                                                   uniforms.environmentIntensity,
                                                   uniforms);
                }
            }
            if (uniforms.backgroundMode != 2u) {
                background = to_working_space(background, uniforms);
            }
            if (debugContext) {
                record_debug_event(*debugContext,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   /*frontFace=*/0u,
                                   kInvalidIndex,
                                   /*scatterIsDelta=*/false,
                                   throughput);
            }
            float misWeight = 1.0f;
            bool useSpecularMis = (!lastScatterWasDelta) ||
                                  (uniforms.enableSpecularNee != 0u) ||
                                  ((ENABLE_MNEE_CAUSTICS != 0) && (uniforms.enableMnee != 0u));
            if (useSpecularMis && envSampling) {
                float lightPdf = environment_pdf(uniforms, environmentPdf, ray.direction);
                float denom = lastBsdfPdf + lightPdf;
                if (denom > 0.0f) {
                    misWeight = clamp(lastBsdfPdf / denom,
                                      kMisWeightClampMin,
                                      kMisWeightClampMax);
                }
            }
            float3 contribution = background * misWeight;
            radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
            break;
        }

        prevRec = rec;
        prevValid = true;

        if (!materials || uniforms.materialCount == 0) {
            break;
        }
        if (mediumDepth > 0u) {
            float3 sigma = mediumSigmaStack[mediumDepth - 1u];
            if (any(sigma > float3(0.0f))) {
                float segment = max(rec.t, 0.0f);
                float3 attenuation = exp(-sigma * segment);
                throughput *= attenuation;
            }
        }
        uint matIndex = min(rec.materialIndex, uniforms.materialCount - 1);
        MaterialData material = materials[matIndex];
        uint type = static_cast<uint>(material.typeEta.x);
        float3 incidentDir = normalize(ray.direction);
        float3 wo = -incidentDir;
        float hitDistanceWorld = ray_segment_world_length(ray, rec.t);
        bool surfaceIsDelta = material_is_delta(material);
        bool specularOnly = (uniforms.debugSpecularOnly != 0u);
        float diffuseOcclusion = 1.0f;
        float3 debugBaseColor = material_base_color(material);
        float debugMetallic = 0.0f;
        float debugRoughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        float debugAO = 1.0f;
        float3 shadingNormal = rec.shadingNormal;
        if (!all(isfinite(shadingNormal)) || dot(shadingNormal, shadingNormal) <= 0.0f) {
            shadingNormal = rec.normal;
        }
        if (rec.primitiveType == kPrimitiveTypeTriangle &&
            meshInfos && sceneVertices && meshIndices && uniforms.meshCount > 0u) {
            float3 candidate = interpolate_shading_normal(uniforms,
                                                          rec.meshIndex,
                                                          rec.primitiveIndex,
                                                          rec.barycentric,
                                                          meshInfos,
                                                          sceneVertices,
                                                          meshIndices);
            if (all(isfinite(candidate)) && dot(candidate, candidate) > 0.0f) {
                if (dot(candidate, rec.normal) < 0.0f) {
                    candidate = -candidate;
                }
                shadingNormal = normalize(candidate);
            }
        }
        if (type == 2u) { // Dielectric: force geometric normal for shading.
            float3 geomNormal = rec.normal;
            if (all(isfinite(geomNormal)) && dot(geomNormal, geomNormal) > 0.0f) {
                shadingNormal = geomNormal;
            }
            // Keep ray offsets consistent between SWRT/HWRT for glass.
            rec.shadingNormal = shadingNormal;
        }

        if (type == 7u && rec.primitiveType == kPrimitiveTypeTriangle &&
            meshInfos && sceneVertices && meshIndices && uniforms.meshCount > 0u) {
            float2 uv0 = interpolate_uv(uniforms,
                                        rec.meshIndex,
                                        rec.primitiveIndex,
                                        rec.barycentric,
                                        0u,
                                        meshInfos,
                                        sceneVertices,
                                        meshIndices);
            float2 uv1 = interpolate_uv(uniforms,
                                        rec.meshIndex,
                                        rec.primitiveIndex,
                                        rec.barycentric,
                                        1u,
                                        meshInfos,
                                        sceneVertices,
                                        meshIndices);
            float4 tangent = interpolate_tangent(uniforms,
                                                 rec.meshIndex,
                                                 rec.primitiveIndex,
                                                 rec.barycentric,
                                                 meshInfos,
                                                 sceneVertices,
                                                 meshIndices);
            if (material.typeEta.z > 0.5f) {
                rec.twoSided = 1u;
            }
            float coneFootprintWorld = ray_cone_width_at_distance(rayCone, hitDistanceWorld);
            float surfaceFootprintWorld =
                surface_footprint_from_cone(coneFootprintWorld, rec.normal, wo);
            float3 dPdu0 = float3(0.0f);
            float3 dPdv0 = float3(0.0f);
            float uvPerWorld0 = 0.0f;
            bool hasSurfacePartials0 = triangle_surface_partials(uniforms,
                                                                 rec.meshIndex,
                                                                 rec.primitiveIndex,
                                                                 0u,
                                                                 meshInfos,
                                                                 sceneVertices,
                                                                 meshIndices,
                                                                 dPdu0,
                                                                 dPdv0,
                                                                 uvPerWorld0);
            float3 dPdu1 = float3(0.0f);
            float3 dPdv1 = float3(0.0f);
            float uvPerWorld1 = 0.0f;
            bool hasSurfacePartials1 = triangle_surface_partials(uniforms,
                                                                 rec.meshIndex,
                                                                 rec.primitiveIndex,
                                                                 1u,
                                                                 meshInfos,
                                                                 sceneVertices,
                                                                 meshIndices,
                                                                 dPdu1,
                                                                 dPdv1,
                                                                 uvPerWorld1);
            float2 dUVdx0 = float2(0.0f);
            float2 dUVdy0 = float2(0.0f);
            bool hasIgehyGradients0 = false;
            if (depth == 0u && hasSurfacePartials0) {
                float3 dudP = float3(0.0f);
                float3 dvdP = float3(0.0f);
                if (uv_world_gradients_from_partials(dPdu0, dPdv0, dudP, dvdP)) {
                    hasIgehyGradients0 = first_hit_uv_gradients_igehy(ray,
                                                                      primaryRayDiff,
                                                                      rec.t,
                                                                      rec.normal,
                                                                      dudP,
                                                                      dvdP,
                                                                      dUVdx0,
                                                                      dUVdy0);
                }
            }
            float2 dUVdx1 = float2(0.0f);
            float2 dUVdy1 = float2(0.0f);
            bool hasIgehyGradients1 = false;
            if (depth == 0u && hasSurfacePartials1) {
                float3 dudP = float3(0.0f);
                float3 dvdP = float3(0.0f);
                if (uv_world_gradients_from_partials(dPdu1, dPdv1, dudP, dvdP)) {
                    hasIgehyGradients1 = first_hit_uv_gradients_igehy(ray,
                                                                      primaryRayDiff,
                                                                      rec.t,
                                                                      rec.normal,
                                                                      dudP,
                                                                      dvdP,
                                                                      dUVdx1,
                                                                      dUVdy1);
                }
            }

            PbrTextureSamplingContext baseColorCtx = make_pbr_texture_sampling_context(material,
                                                                                        kPbrTextureSlotBaseColor,
                                                                                        uv0,
                                                                                        uv1,
                                                                                        hasIgehyGradients0,
                                                                                        dUVdx0,
                                                                                        dUVdy0,
                                                                                        uvPerWorld0,
                                                                                        hasIgehyGradients1,
                                                                                        dUVdx1,
                                                                                        dUVdy1,
                                                                                        uvPerWorld1);
            PbrTextureSamplingContext ormCtx = make_pbr_texture_sampling_context(material,
                                                                                  kPbrTextureSlotMetallicRoughness,
                                                                                  uv0,
                                                                                  uv1,
                                                                                  hasIgehyGradients0,
                                                                                  dUVdx0,
                                                                                  dUVdy0,
                                                                                  uvPerWorld0,
                                                                                  hasIgehyGradients1,
                                                                                  dUVdx1,
                                                                                  dUVdy1,
                                                                                  uvPerWorld1);
            PbrTextureSamplingContext normalCtx = make_pbr_texture_sampling_context(material,
                                                                                     kPbrTextureSlotNormal,
                                                                                     uv0,
                                                                                     uv1,
                                                                                     hasIgehyGradients0,
                                                                                     dUVdx0,
                                                                                     dUVdy0,
                                                                                     uvPerWorld0,
                                                                                     hasIgehyGradients1,
                                                                                     dUVdx1,
                                                                                     dUVdy1,
                                                                                     uvPerWorld1);
            PbrTextureSamplingContext occlusionCtx = make_pbr_texture_sampling_context(material,
                                                                                        kPbrTextureSlotOcclusion,
                                                                                        uv0,
                                                                                        uv1,
                                                                                        hasIgehyGradients0,
                                                                                        dUVdx0,
                                                                                        dUVdy0,
                                                                                        uvPerWorld0,
                                                                                        hasIgehyGradients1,
                                                                                        dUVdx1,
                                                                                        dUVdy1,
                                                                                        uvPerWorld1);
            PbrTextureSamplingContext emissiveCtx = make_pbr_texture_sampling_context(material,
                                                                                       kPbrTextureSlotEmissive,
                                                                                       uv0,
                                                                                       uv1,
                                                                                       hasIgehyGradients0,
                                                                                       dUVdx0,
                                                                                       dUVdy0,
                                                                                       uvPerWorld0,
                                                                                       hasIgehyGradients1,
                                                                                       dUVdx1,
                                                                                       dUVdy1,
                                                                                       uvPerWorld1);
            PbrTextureSamplingContext transmissionCtx = make_pbr_texture_sampling_context(material,
                                                                                           kPbrTextureSlotTransmission,
                                                                                           uv0,
                                                                                           uv1,
                                                                                           hasIgehyGradients0,
                                                                                           dUVdx0,
                                                                                           dUVdy0,
                                                                                           uvPerWorld0,
                                                                                           hasIgehyGradients1,
                                                                                           dUVdx1,
                                                                                           dUVdy1,
                                                                                           uvPerWorld1);
            float3 baseFactor = to_working_space(material.baseColorRoughness.xyz, uniforms);
            float baseColorLod =
                material_texture_lod_with_fallback(materialTextures,
                                                   materialTextureInfos,
                                                   uniforms,
                                                   material.textureIndices0.x,
                                                   baseColorCtx.hasIgehyGradients,
                                                   baseColorCtx.dUVdx,
                                                   baseColorCtx.dUVdy,
                                                   baseColorCtx.uvPerWorld,
                                                   surfaceFootprintWorld);
            float4 baseColorSample =
                sample_material_texture_filtered(materialTextures,
                                                materialSamplers,
                                                materialTextureInfos,
                                                uniforms,
                                                material.textureIndices0.x,
                                                baseColorCtx.uv,
                                                float4(1.0f),
                                                baseColorLod,
                                                baseColorCtx.hasIgehyGradients,
                                                baseColorCtx.dUVdx,
                                                baseColorCtx.dUVdy);
            float3 baseColorSampleRgb = to_working_space(baseColorSample.xyz, uniforms);
            float3 baseColor = baseFactor * baseColorSampleRgb;

            float metallic = clamp(material.pbrParams.x, 0.0f, 1.0f);
            float roughness = clamp(material.pbrParams.y, 0.0f, 1.0f);
            float normalStrengthScale = 1.0f;
#if PT_DEBUG_TOOLS
            normalStrengthScale = max(uniforms.debugNormalStrengthScale, 0.0f);
#endif
            float normalScale = material.pbrParams.w * normalStrengthScale;
            bool disableOrmByMaterial = (material.materialFlags & kMaterialFlagDisableOrm) != 0u;
            bool useOrmTexture = !disableOrmByMaterial &&
                                 material_texture_valid(uniforms, material.textureIndices0.y);
#if PT_DEBUG_TOOLS
            useOrmTexture = useOrmTexture && (uniforms.debugDisableOrmTexture == 0u);
#endif
            if (useOrmTexture) {
                float ormLod =
                    material_texture_lod_with_fallback(materialTextures,
                                                       materialTextureInfos,
                                                       uniforms,
                                                       material.textureIndices0.y,
                                                       ormCtx.hasIgehyGradients,
                                                       ormCtx.dUVdx,
                                                       ormCtx.dUVdy,
                                                       ormCtx.uvPerWorld,
                                                       surfaceFootprintWorld);
#if PT_DEBUG_TOOLS
                ormLod = max(ormLod + uniforms.debugOrmLodBias, 0.0f);
#endif
                float3 mrSample =
                    sample_material_texture_level(materialTextures,
                                                  materialSamplers,
                                                  materialTextureInfos,
                                                  uniforms,
                                                  material.textureIndices0.y,
                                                  ormCtx.uv,
                                                  float4(1.0f),
                                                  ormLod).xyz;
                metallic = clamp(mrSample.z * metallic, 0.0f, 1.0f);
                roughness = clamp(mrSample.y * roughness, 0.0f, 1.0f);
            }
            float visorMask = visor_override_blend(baseColor, metallic, roughness, matIndex, uniforms);
            if (visorMask > 0.0f) {
                    float overrideRoughness =
                        clamp(uniforms.debugVisorOverrideRoughness, 0.0f, 1.0f);
                    float overrideF0 = clamp(uniforms.debugVisorOverrideF0, 0.0f, 0.12f);
                    metallic = mix(metallic, 0.0f, visorMask);
                    roughness = mix(roughness, overrideRoughness, visorMask);
                    material.typeEta.y = mix(material.typeEta.y,
                                             ior_from_f0(overrideF0),
                                             visorMask);
            }
            float normalLod =
                material_texture_lod_with_fallback(materialTextures,
                                                   materialTextureInfos,
                                                   uniforms,
                                                   material.textureIndices0.z,
                                                   normalCtx.hasIgehyGradients,
                                                   normalCtx.dUVdx,
                                                   normalCtx.dUVdy,
                                                   normalCtx.uvPerWorld,
                                                   surfaceFootprintWorld);
#if PT_DEBUG_TOOLS
            normalLod = max(normalLod + uniforms.debugNormalLodBias, 0.0f);
#endif

            float transmission = clamp(material.pbrExtras.z, 0.0f, 1.0f);
            if (material_texture_valid(uniforms, material.textureIndices1.y)) {
                float transmissionLod =
                    material_texture_lod_with_fallback(materialTextures,
                                                       materialTextureInfos,
                                                       uniforms,
                                                       material.textureIndices1.y,
                                                       transmissionCtx.hasIgehyGradients,
                                                       transmissionCtx.dUVdx,
                                                       transmissionCtx.dUVdy,
                                                       transmissionCtx.uvPerWorld,
                                                       surfaceFootprintWorld);
                float transmissionSample =
                    sample_material_texture_filtered(materialTextures,
                                                    materialSamplers,
                                                    materialTextureInfos,
                                                    uniforms,
                                                    material.textureIndices1.y,
                                                    transmissionCtx.uv,
                                                    float4(1.0f),
                                                    transmissionLod,
                                                    transmissionCtx.hasIgehyGradients,
                                                    transmissionCtx.dUVdx,
                                                    transmissionCtx.dUVdy).x;
                transmission = clamp(transmission * transmissionSample, 0.0f, 1.0f);
            }
            transmission *= (1.0f - metallic);

            float alpha = clamp(material.pbrExtras.x, 0.0f, 1.0f);
            alpha = clamp(alpha * baseColorSample.w, 0.0f, 1.0f);
            float alphaCutoff = clamp(material.pbrExtras.y, 0.0f, 1.0f);
            float alphaMode = material.pbrExtras.w;
            if (alphaMode > 0.5f) {
                bool discard = false;
                if (alphaMode < 1.5f) {
                    discard = alpha < alphaCutoff;
                } else {
                    discard = rand_uniform(state) > alpha;
                }
                if (discard) {
                    ray.origin = offset_ray_origin(rec, ray.direction);
                    prevRec = rec;
                    prevValid = true;
                    lastBsdfPdf = 1.0f;
                    lastScatterWasDelta = true;
                    specularDepth += 1u;
                    continue;
                }
            }

            material.pbrExtras.z = transmission;

            float occlusion = 1.0f;
            if (!disableOrmByMaterial && material_texture_valid(uniforms, material.textureIndices0.w)) {
                float occlusionLod =
                    material_texture_lod_with_fallback(materialTextures,
                                                       materialTextureInfos,
                                                       uniforms,
                                                       material.textureIndices0.w,
                                                       occlusionCtx.hasIgehyGradients,
                                                       occlusionCtx.dUVdx,
                                                       occlusionCtx.dUVdy,
                                                       occlusionCtx.uvPerWorld,
                                                       surfaceFootprintWorld);
                float occSample =
                    sample_material_texture_filtered(materialTextures,
                                                    materialSamplers,
                                                    materialTextureInfos,
                                                    uniforms,
                                                    material.textureIndices0.w,
                                                    occlusionCtx.uv,
                                                    float4(1.0f),
                                                    occlusionLod,
                                                    occlusionCtx.hasIgehyGradients,
                                                    occlusionCtx.dUVdx,
                                                    occlusionCtx.dUVdy).x;
                occlusion = mix(1.0f, occSample, clamp(material.pbrParams.z, 0.0f, 1.0f));
            }
            debugAO = occlusion;
            diffuseOcclusion = (uniforms.debugDisableAO != 0u) ? 1.0f : occlusion;
            if (uniforms.debugAoIndirectOnly != 0u && depth == 0u) {
                diffuseOcclusion = 1.0f;
            }
            debugBaseColor = baseColor;
            debugMetallic = metallic;
            debugRoughness = roughness;

            float3 emissive = to_working_space(material.emission.xyz, uniforms);
            if (material_texture_valid(uniforms, material.textureIndices1.x)) {
                float emissiveLod =
                    material_texture_lod_with_fallback(materialTextures,
                                                       materialTextureInfos,
                                                       uniforms,
                                                       material.textureIndices1.x,
                                                       emissiveCtx.hasIgehyGradients,
                                                       emissiveCtx.dUVdx,
                                                       emissiveCtx.dUVdy,
                                                       emissiveCtx.uvPerWorld,
                                                       surfaceFootprintWorld);
                float3 emissiveSample =
                    sample_material_texture_filtered(materialTextures,
                                                    materialSamplers,
                                                    materialTextureInfos,
                                                    uniforms,
                                                    material.textureIndices1.x,
                                                    emissiveCtx.uv,
                                                    float4(1.0f),
                                                    emissiveLod,
                                                    emissiveCtx.hasIgehyGradients,
                                                    emissiveCtx.dUVdx,
                                                    emissiveCtx.dUVdy).xyz;
                emissiveSample = to_working_space(emissiveSample, uniforms);
                emissive *= emissiveSample;
            }

            bool useNormalMap = material_texture_valid(uniforms, material.textureIndices0.z);
#if PT_DEBUG_TOOLS
            if (uniforms.debugDisableNormalMap != 0u) {
                useNormalMap = false;
            }
#endif
            if (normalScale <= 1.0e-4f) {
                useNormalMap = false;
            }
            float normalLength = 1.0f;
            float3 normalSampleTs = float3(0.0f, 0.0f, 1.0f);
            bool flipNormalGreen = false;
#if PT_DEBUG_TOOLS
            flipNormalGreen = uniforms.debugFlipNormalGreen != 0u;
#endif
            if (useNormalMap) {
                normalSampleTs =
                    sample_material_texture_filtered(materialTextures,
                                                    materialSamplers,
                                                    materialTextureInfos,
                                                    uniforms,
                                                    material.textureIndices0.z,
                                                    normalCtx.uv,
                                                    float4(0.5f, 0.5f, 1.0f, 1.0f),
                                                    normalLod,
                                                    normalCtx.hasIgehyGradients,
                                                    normalCtx.dUVdx,
                                                    normalCtx.dUVdy).xyz;
                normalSampleTs = decode_normal_map(normalSampleTs,
                                                   normalScale,
                                                   flipNormalGreen,
                                                   normalLength);
                float3 t = tangent.xyz;
                float3 b = float3(0.0f);
                bool hasBasis = false;
                bool trustVertexTangent = fabs(tangent.w) > 0.5f;
                if (trustVertexTangent && all(isfinite(t)) && dot(t, t) > 1.0e-6f) {
                    t = normalize(t - shadingNormal * dot(shadingNormal, t));
                    if (all(isfinite(t)) && dot(t, t) > 1.0e-6f) {
                        float tangentSign = (tangent.w < 0.0f) ? -1.0f : 1.0f;
                        b = normalize(cross(shadingNormal, t)) * tangentSign;
                        if (all(isfinite(b)) && dot(b, b) > 1.0e-6f) {
                            hasBasis = true;
                        }
                    }
                }
                if (!hasBasis) {
                    uint normalUvSet = pbr_texture_uv_set(material, kPbrTextureSlotNormal);
                    hasBasis = compute_tangent_basis_from_uv(uniforms,
                                                             rec.meshIndex,
                                                             rec.primitiveIndex,
                                                             normalUvSet,
                                                             meshInfos,
                                                             sceneVertices,
                                                             meshIndices,
                                                             shadingNormal,
                                                             t,
                                                             b);
                }
                if (!hasBasis) {
                    build_onb(shadingNormal, t, b);
                }
                float3 mapped = normalize(t * normalSampleTs.x +
                                          b * normalSampleTs.y +
                                          shadingNormal * normalSampleTs.z);
                if (dot(mapped, rec.normal) < 0.0f) {
                    mapped = -mapped;
                }
                shadingNormal = mapped;
            }

            if (useNormalMap) {
                float tok = max((1.0f - normalLength) / max(normalLength, 1.0e-6f), 0.0f);
                if (normalCtx.hasIgehyGradients &&
                    all(isfinite(normalCtx.dUVdx)) &&
                    all(isfinite(normalCtx.dUVdy))) {
                    float gradMag = max(max(fabs(normalCtx.dUVdx.x), fabs(normalCtx.dUVdx.y)),
                                        max(fabs(normalCtx.dUVdy.x), fabs(normalCtx.dUVdy.y)));
                    if (gradMag > 1.0e-6f && gradMag < 4.0f) {
                        float3 nDx = sample_material_texture_level(materialTextures,
                                                                   materialSamplers,
                                                                   materialTextureInfos,
                                                                   uniforms,
                                                                   material.textureIndices0.z,
                                                                   normalCtx.uv + normalCtx.dUVdx,
                                                                   float4(0.5f, 0.5f, 1.0f, 1.0f),
                                                                   normalLod).xyz;
                        float3 nDy = sample_material_texture_level(materialTextures,
                                                                   materialSamplers,
                                                                   materialTextureInfos,
                                                                   uniforms,
                                                                   material.textureIndices0.z,
                                                                   normalCtx.uv + normalCtx.dUVdy,
                                                                   float4(0.5f, 0.5f, 1.0f, 1.0f),
                                                                   normalLod).xyz;
                        float tmpLenDx = 1.0f;
                        float tmpLenDy = 1.0f;
                        nDx = decode_normal_map(nDx, normalScale, flipNormalGreen, tmpLenDx);
                        nDy = decode_normal_map(nDy, normalScale, flipNormalGreen, tmpLenDy);
                        float varianceX = max(1.0f - dot(normalSampleTs, nDx), 0.0f);
                        float varianceY = max(1.0f - dot(normalSampleTs, nDy), 0.0f);
                        float normalVariance = max(varianceX, varianceY);
                        tok += 0.35f * normalVariance;
                    }
                }
                roughness = clamp(sqrt(roughness * roughness + tok), 0.0f, 1.0f);
            }

            material.baseColorRoughness = float4(baseColor, roughness);
            material.pbrParams.x = metallic;
            material.emission = float4(emissive, 0.0f);
            rec.shadingNormal = shadingNormal;
        }

        if (uniforms.debugViewMode != kDebugViewNone) {
            float3 debugColor = float3(0.0f);
            switch (uniforms.debugViewMode) {
                case kDebugViewBaseColor:
                    debugColor = debugBaseColor;
                    break;
                case kDebugViewMetallic:
                    debugColor = float3(debugMetallic);
                    break;
                case kDebugViewRoughness:
                    debugColor = float3(debugRoughness);
                    break;
                case kDebugViewAO:
                    debugColor = float3(debugAO);
                    break;
                default:
                    break;
            }
            radiance = debugColor;
            break;
        }

        // Capture first hit AOVs (albedo and normal) for denoising
        if (isFirstHit) {
            isFirstHit = false;
            if (outFirstHitAlbedo != nullptr) {
                // Albedo is the diffuse color from the first hit
                *outFirstHitAlbedo = material_base_color(material);
            }
            if (outFirstHitNormal != nullptr) {
                // Store the world-space normal of first hit
                *outFirstHitNormal = shadingNormal;
            }
        }

        if (!specularOnly &&
            type == 7u &&
            any(material.emission.xyz != float3(0.0f)) &&
            (rec.frontFace != 0u || rec.twoSided != 0u)) {
            radiance += clamp_firefly_contribution(throughput, material.emission.xyz, clampParams);
        }

        if (type == 3u) {  // DiffuseLight
            if (specularOnly) {
                break;
            }
            float3 emission = material.emission.xyz;
            if (material.emission.w > 0.0f &&
                environmentTexture.get_width() > 0 &&
                environmentTexture.get_height() > 0 &&
                rec.frontFace != 0u) {
                float3 sampleDir = -shadingNormal;
                float3 envColor = environment_color(environmentTexture,
                                                    sampleDir,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity,
                                                    uniforms);
                emission *= envColor;
            }
            if (any(emission != float3(0.0f)) &&
                (rec.frontFace != 0u || rec.twoSided != 0u)) {
                float misWeight = 1.0f;
                bool useSpecularMis = (!lastScatterWasDelta) ||
                                      (uniforms.enableSpecularNee != 0u) ||
                                      ((ENABLE_MNEE_CAUSTICS != 0) && (uniforms.enableMnee != 0u));
                if (useSpecularMis && rectLightCount > 0u) {
                    float lightPdf = rect_light_pdf_for_hit(uniforms,
                                                            rectangles,
                                                            materials,
                                                            rectLightCount,
                                                            rec,
                                                            ray.origin);
                    float denom = lastBsdfPdf + lightPdf;
                    if (denom > 0.0f) {
                        misWeight = clamp(lastBsdfPdf / denom,
                                          kMisWeightClampMin,
                                          kMisWeightClampMax);
                    }
                }
                float3 contribution = emission * misWeight;
                radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
            }
            break;
        }

        if (!surfaceIsDelta && rectLightCount > 0u) {
            RectLightSample lightSample;
            if (sample_rect_light(uniforms,
                                  rectangles,
                                  materials,
                                  environmentTexture,
                                  rec,
                                  state,
                                  rectLightCount,
                                  lightSample)) {
                float nDotL = max(dot(shadingNormal, lightSample.direction), 0.0f);
                if (lightSample.pdf > 0.0f && nDotL > 0.0f) {
                    Ray shadowRay;
                    shadowRay.origin = offset_ray_origin(rec, lightSample.direction);
                    shadowRay.direction = lightSample.direction;
                    HitRecord shadowRec;
                    float shadowMax = max(lightSample.distance - kEpsilon, kEpsilon);
                    bool occluded = trace_scene_software(uniforms,
                                                         spheres,
                                                         rectangles,
                                                         triangleData,
                                                         tlasNodes,
                                                         tlasPrimIndices,
                                                         instanceInfos,
                                                         blasNodes,
                                                         blasPrimIndices,
                                                         nodes,
                                                         primitiveIndices,
                                                         stats,
                                                         shadowRay,
                                                         kEpsilon,
                                                         shadowMax,
                                                         /*anyHitOnly=*/true,
                                                         /*includeTriangles=*/true,
                                                         shadowRec);
                    if (!occluded) {
                        BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                                rec.point,
                                                                shadingNormal,
                                                                wo,
                                                                lightSample.direction,
                                                                clampParams,
                                                                uniforms.sssMode,
                                                                diffuseOcclusion,
                                                                specularOnly);
                        if (!bsdfEval.isDelta && !bsdfEval.isBssrdf) {
                            float3 bsdfValue = bsdfEval.value;
                            float maxComponent = max(max(bsdfValue.x, bsdfValue.y), bsdfValue.z);
                            if (maxComponent > 0.0f && lightSample.pdf > 0.0f) {
                                float bsdfPdf = bsdfEval.pdf;
                                float weight = 1.0f;
                                if (bsdfPdf > 0.0f) {
                                    float denom = lightSample.pdf + bsdfPdf;
                                    if (denom > 0.0f) {
                                        weight = clamp(lightSample.pdf / denom,
                                                       kMisWeightClampMin,
                                                       kMisWeightClampMax);
                                    }
                                }
                                float3 contribution = lightSample.emission * bsdfValue * nDotL;
                                contribution *= weight / lightSample.pdf;
                                if (all(isfinite(contribution))) {
                                    radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (!surfaceIsDelta && envSampling) {
            EnvironmentSample envSample;
            if (sample_environment(uniforms,
                                   environmentTexture,
                                   environmentConditionalAlias,
                                   environmentMarginalAlias,
                                   environmentPdf,
                                   state,
                                   envSample)) {
                float overrideLod = 0.0f;
                bool useOverride = environment_mip_override(uniforms, environmentTexture, overrideLod);
                if (environmentTexture.get_num_mip_levels() > 1u) {
                    float envRoughness = environment_lighting_roughness(material);
                    if (envRoughness < 0.95f) {
                        float envLod = environment_lod_from_roughness(envRoughness,
                                                                      environmentTexture);
                        envSample.radiance = environment_color_lod(environmentTexture,
                                                                   envSample.direction,
                                                                   uniforms.environmentRotation,
                                                                   uniforms.environmentIntensity,
                                                                   envLod,
                                                                   uniforms);
                    }
                }
                if (useOverride) {
                    envSample.radiance = environment_color_lod(environmentTexture,
                                                               envSample.direction,
                                                               uniforms.environmentRotation,
                                                               uniforms.environmentIntensity,
                                                               overrideLod,
                                                               uniforms);
                }
                float nDotL = max(dot(shadingNormal, envSample.direction), 0.0f);
                if (envSample.pdf > 0.0f && nDotL > 0.0f) {
                    Ray shadowRay;
                    shadowRay.origin = offset_ray_origin(rec, envSample.direction);
                    shadowRay.direction = envSample.direction;
                    HitRecord shadowRec;
                    bool occluded = trace_scene_software(uniforms,
                                                         spheres,
                                                         rectangles,
                                                         triangleData,
                                                         tlasNodes,
                                                         tlasPrimIndices,
                                                         instanceInfos,
                                                         blasNodes,
                                                         blasPrimIndices,
                                                         nodes,
                                                         primitiveIndices,
                                                         stats,
                                                         shadowRay,
                                                         kEpsilon,
                                                         kInfinity,
                                                         /*anyHitOnly=*/true,
                                                         /*includeTriangles=*/true,
                                                         shadowRec);
                    if (!occluded) {
                        BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                                rec.point,
                                                                shadingNormal,
                                                                wo,
                                                                envSample.direction,
                                                                clampParams,
                                                                uniforms.sssMode,
                                                                diffuseOcclusion,
                                                                specularOnly);
                        if (!bsdfEval.isDelta && !bsdfEval.isBssrdf) {
                            float3 bsdfValue = bsdfEval.value;
                            float maxComponent = max(max(bsdfValue.x, bsdfValue.y), bsdfValue.z);
                            if (maxComponent > 0.0f) {
                                float bsdfPdf = bsdfEval.pdf;
                                float weight = 1.0f;
                                if (bsdfPdf > 0.0f) {
                                    float denom = envSample.pdf + bsdfPdf;
                                    if (denom > 0.0f) {
                                        weight = clamp(envSample.pdf / denom,
                                                       kMisWeightClampMin,
                                                       kMisWeightClampMax);
                                    }
                                }
                                float3 contribution = envSample.radiance * bsdfValue * nDotL;
                                contribution *= weight / envSample.pdf;
                                if (all(isfinite(contribution))) {
                                    radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                                }
                            }
                        }
                    }
                }
            }
        }

        BsdfSampleResult bsdfSample;
        bool usedRandomWalk = false;
        bool enableRandomWalk = material_is_subsurface(material) &&
                                uniforms.sssMode == 2u &&
                                material.sssParams.y >= 0.5f &&
                                rec.frontFace != 0u;
        if (enableRandomWalk) {
            bsdfSample = sample_sss_random_walk_software(uniforms,
                                                         material,
                                                         rec,
                                                         wo,
                                                         incidentDir,
                                                         spheres,
                                                         rectangles,
                                                         triangleData,
                                                         tlasNodes,
                                                         tlasPrimIndices,
                                                         instanceInfos,
                                                         blasNodes,
                                                         blasPrimIndices,
                                                         nodes,
                                                         primitiveIndices,
                                                         stats,
                                                         state,
                                                         clampParams);
            usedRandomWalk = (bsdfSample.pdf > 0.0f);
        }
        if (!usedRandomWalk) {
            bsdfSample = sample_bsdf(material,
                                     rec.point,
                                     shadingNormal,
                                     wo,
                                     incidentDir,
                                     rec.frontFace != 0u,
                                     state,
                                     clampParams,
                                     uniforms.sssMode,
                                     diffuseOcclusion,
                                     specularOnly);
        }
        if (bsdfSample.pdf <= 0.0f) {
            break;
        }

        uint mediumDepthBefore = mediumDepth;
        if (bsdfSample.mediumEvent == 1) {
            float3 sigma = dielectric_sigma_a(material);
            sigma = max(sigma, float3(0.0f));
            if (mediumDepth < kMaxMediumStack) {
                mediumSigmaStack[mediumDepth] = sigma;
                mediumDepth += 1u;
            } else {
                mediumSigmaStack[kMaxMediumStack - 1u] = sigma;
            }
        } else if (bsdfSample.mediumEvent == -1) {
            if (mediumDepth > 0u) {
                mediumDepth -= 1u;
            }
        }
        uint mediumDepthAfter = mediumDepth;

        if (debugContext) {
            record_debug_event(*debugContext,
                               depth,
                               mediumDepthBefore,
                               mediumDepthAfter,
                               bsdfSample.mediumEvent,
                               rec.frontFace,
                               rec.materialIndex,
                               bsdfSample.isDelta,
                               throughput);
        }

        bool causticCandidate = (!surfaceIsDelta) && (specularDepth > 0u);
        uint nextSpecularDepth = bsdfSample.isDelta ? (specularDepth + 1u) : 0u;
        bool didTransmission = false;
        if (bsdfSample.isDelta && type == 2u) {
            float3 dir = bsdfSample.direction;
            if (all(isfinite(dir)) && dot(dir, dir) > 0.0f) {
                float side = (rec.frontFace != 0u) ? 1.0f : -1.0f;
                didTransmission = (dot(shadingNormal, dir) * side) < 0.0f;
            }
        }
        if (didTransmission) {
            hadTransmission = true;
        }
        specularDepth = nextSpecularDepth;
        (void)causticCandidate;
        (void)hadTransmission;

        float3 nextOrigin;
        if (bsdfSample.hasExitPoint) {
            float3 exitNormal = bsdfSample.exitNormal;
            bool normalValid = all(isfinite(exitNormal)) && dot(exitNormal, exitNormal) > 0.0f;
            if (!normalValid) {
                exitNormal = rec.normal;
                normalValid = all(isfinite(exitNormal)) && dot(exitNormal, exitNormal) > 0.0f;
            }
            if (!normalValid) {
                exitNormal = float3(0.0f, 1.0f, 0.0f);
            }
            exitNormal = normalize(exitNormal);
            nextOrigin = offset_surface_point(bsdfSample.exitPoint, exitNormal, bsdfSample.direction);
            // HWRT still reports misses when a refracted ray's origin sits inside the mesh.
            // Push further along the exit normal plus a bit down the outgoing direction
            // so the TLAS starts well outside the surface.
            float normalBias = max(kHardwareOcclusionEpsilon * 4.0f, kRayOriginEpsilon * 32.0f);
            nextOrigin += exitNormal * normalBias;
            float3 dir = bsdfSample.direction;
            if (!all(isfinite(dir)) || dot(dir, dir) <= 0.0f) {
                dir = exitNormal;
            } else {
                dir = normalize(dir);
            }
            float directionalBias = max(kHardwareOcclusionEpsilon * 8.0f, kRayOriginEpsilon * 32.0f);
            nextOrigin += dir * directionalBias;
        } else {
            nextOrigin = offset_ray_origin(rec, bsdfSample.direction);
        }

        bool useMnee = (ENABLE_MNEE_CAUSTICS != 0) && (uniforms.enableMnee != 0u);
        bool specNeeEnabled = (uniforms.enableSpecularNee != 0u);
        float dirLenSq = dot(bsdfSample.direction, bsdfSample.direction);
        bool specDirectionValid = (dirLenSq > 0.0f) && all(isfinite(bsdfSample.direction));
        bool mneeEligible = false;
#if ENABLE_MNEE_CAUSTICS
        mneeEligible = useMnee &&
                       bsdfSample.isDelta &&
                       ((bsdfSample.mediumEvent <= 0) || didTransmission) &&
                       (type == 2u) &&
                       (nextSpecularDepth == 1u) &&
                       specDirectionValid;
#endif
        if (mneeEligible) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEligibleCount, 1u, memory_order_relaxed);
            }
#if PT_MNEE_OCCLUSION_PARITY
            HitRecord mneeSwRec;
            bool swHit = trace_scene_software_with_exclusion(uniforms,
                                                             spheres,
                                                             rectangles,
                                                             triangleData,
                                                             tlasNodes,
                                                             tlasPrimIndices,
                                                             instanceInfos,
                                                             blasNodes,
                                                             blasPrimIndices,
                                                             nodes,
                                                             primitiveIndices,
                                                             stats,
                                                             ray,
                                                             kEpsilon,
                                                             kInfinity,
                                                             excludeMeshIndex,
                                                             excludePrimitiveIndex,
                                                             mneeSwRec);
            if (stats) {
                if (!swHit) {
                    atomic_fetch_add_explicit(&stats->mneeHitHwSwHitMissCount,
                                              1u,
                                              memory_order_relaxed);
                } else {
                    float epsT = max(1.0e-3f, 1.0e-4f * fabs(rec.t));
                    float tDiff = fabs(rec.t - mneeSwRec.t);
                    if (tDiff > epsT) {
                        atomic_fetch_add_explicit(&stats->mneeHitHwSwTDiffCount,
                                                  1u,
                                                  memory_order_relaxed);
                    }
                    if (rec.frontFace != mneeSwRec.frontFace ||
                        rec.materialIndex != mneeSwRec.materialIndex ||
                        rec.meshIndex != mneeSwRec.meshIndex ||
                        rec.primitiveIndex != mneeSwRec.primitiveIndex) {
                        atomic_fetch_add_explicit(&stats->mneeHitHwSwIdMismatchCount,
                                                  1u,
                                                  memory_order_relaxed);
                    }
                    float3 hwN = rec.normal;
                    float3 swN = mneeSwRec.normal;
                    if (all(isfinite(hwN)) && all(isfinite(swN)) &&
                        dot(hwN, hwN) > 0.0f && dot(swN, swN) > 0.0f) {
                        float nDot = dot(normalize(hwN), normalize(swN));
                        if (nDot < 0.99f) {
                            atomic_fetch_add_explicit(&stats->mneeHitHwSwNormalMismatchCount,
                                                      1u,
                                                      memory_order_relaxed);
                        }
                    }
                }
            }
#endif
        }
        bool specNeeEligible = specNeeEnabled &&
                               bsdfSample.isDelta &&
                               (bsdfSample.mediumEvent <= 0) &&
                               specDirectionValid &&
                               !mneeEligible;

        if (specNeeEligible && envSampling &&
            environmentTexture.get_width() > 0 &&
            environmentTexture.get_height() > 0) {
            Ray neeRay;
            neeRay.origin = nextOrigin;
            neeRay.direction = normalize(bsdfSample.direction);
            HitRecord shadowRec;
            bool occluded = trace_scene_software(uniforms,
                                                 spheres,
                                                 rectangles,
                                                 triangleData,
                                                 tlasNodes,
                                                 tlasPrimIndices,
                                                 instanceInfos,
                                                 blasNodes,
                                                 blasPrimIndices,
                                                 nodes,
                                                 primitiveIndices,
                                                 stats,
                                                 neeRay,
                                                 kEpsilon,
                                                 kInfinity,
                                                 /*anyHitOnly=*/true,
                                                 /*includeTriangles=*/true,
                                                 shadowRec);
            if (!occluded) {
                float envPdf = environment_pdf(uniforms, environmentPdf, neeRay.direction);
                envPdf = max(envPdf, kSpecularNeePdfFloor);
                float invEnvPdf = min(1.0f / envPdf, kSpecularNeeInvPdfClamp);
                float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                float denom = envPdf + bsdfPdf;
                float misWeight = (denom > 0.0f) ? (envPdf / denom) : 0.0f;
                misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                float3 envColor = environment_color(environmentTexture,
                                                    neeRay.direction,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity,
                                                    uniforms);
                float3 neeContribution = bsdfSample.weight * envColor * (misWeight * invEnvPdf);
                if (all(isfinite(neeContribution))) {
                    radiance += clamp_firefly_contribution(throughput, neeContribution, clampParams);
                    if (stats) {
                        atomic_fetch_add_explicit(&stats->specNeeEnvAddedCount, 1u, memory_order_relaxed);
                    }
                }
            } else if (stats) {
                atomic_fetch_add_explicit(&stats->specularNeeOcclusionHitCount,
                                          1u,
                                          memory_order_relaxed);
            }
        }

        if (specNeeEligible && rectLightCount > 0u) {
            Ray neeRay;
            neeRay.origin = nextOrigin;
            neeRay.direction = normalize(bsdfSample.direction);
            HitRecord lightRec;
            bool hitLight = trace_scene_software(uniforms,
                                                 spheres,
                                                 rectangles,
                                                 triangleData,
                                                 tlasNodes,
                                                 tlasPrimIndices,
                                                 instanceInfos,
                                                 blasNodes,
                                                 blasPrimIndices,
                                                 nodes,
                                                 primitiveIndices,
                                                 stats,
                                                 neeRay,
                                                 kEpsilon,
                                                 kInfinity,
                                                 /*anyHitOnly=*/false,
                                                 /*includeTriangles=*/true,
                                                 lightRec);
            if (hitLight) {
                MneeRectHit mneeHit;
                if (mnee_rect_light_hit(uniforms,
                                        rectangles,
                                        materials,
                                        environmentTexture,
                                        rectLightCount,
                                        lightRec,
                                        nextOrigin,
                                        mneeHit)) {
                    float lightPdf = max(mneeHit.pdf, kSpecularNeePdfFloor);
                    float invLightPdf = min(1.0f / lightPdf, kSpecularNeeInvPdfClamp);
                    float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                    float denom = lightPdf + bsdfPdf;
                    float misWeight = (denom > 0.0f) ? (lightPdf / denom) : 0.0f;
                    misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                    float3 contribution = bsdfSample.weight * mneeHit.emission *
                                          (misWeight * invLightPdf);
                    if (all(isfinite(contribution))) {
                        radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                        if (stats) {
                            atomic_fetch_add_explicit(&stats->specNeeRectAddedCount, 1u, memory_order_relaxed);
                        }
                    }
                }
            }
        }

#if ENABLE_MNEE_CAUSTICS
        if (mneeEligible && envSampling &&
            environmentTexture.get_width() > 0 &&
            environmentTexture.get_height() > 0) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEnvAttemptCount, 1u, memory_order_relaxed);
            }
            Ray mneeRay;
            mneeRay.origin = nextOrigin;
            mneeRay.direction = normalize(bsdfSample.direction);
            HitRecord shadowRec;
            bool occluded = trace_scene_software(uniforms,
                                                 spheres,
                                                 rectangles,
                                                 triangleData,
                                                 tlasNodes,
                                                 tlasPrimIndices,
                                                 instanceInfos,
                                                 blasNodes,
                                                 blasPrimIndices,
                                                 nodes,
                                                 primitiveIndices,
                                                 stats,
                                                 mneeRay,
                                                 kEpsilon,
                                                 kInfinity,
                                                 /*anyHitOnly=*/true,
                                                 /*includeTriangles=*/true,
                                                 shadowRec);
            if (!occluded) {
                float envPdf = environment_pdf(uniforms, environmentPdf, mneeRay.direction);
                envPdf = max(envPdf, kSpecularNeePdfFloor);
                float invEnvPdf = min(1.0f / envPdf, kSpecularNeeInvPdfClamp);
                float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                float denom = envPdf + bsdfPdf;
                float misWeight = (denom > 0.0f) ? (envPdf / denom) : 0.0f;
                misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                float3 envColor = environment_color(environmentTexture,
                                                    mneeRay.direction,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity,
                                                    uniforms);
                float3 neeContribution = bsdfSample.weight * envColor * (misWeight * invEnvPdf);
                if (all(isfinite(neeContribution))) {
                    radiance += clamp_firefly_contribution(throughput, neeContribution, clampParams);
                    if (stats) {
                        atomic_fetch_add_explicit(&stats->mneeEnvAddedCount, 1u, memory_order_relaxed);
                        stats_add_mnee_luma(stats, neeContribution);
                    }
                }
            }
        }

        if (mneeEligible && rectLightCount > 0u) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeRectAttemptCount, 1u, memory_order_relaxed);
            }
            Ray mneeRay;
            mneeRay.origin = nextOrigin;
            mneeRay.direction = normalize(bsdfSample.direction);
            HitRecord lightRec;
            bool hitLight = trace_scene_software(uniforms,
                                                 spheres,
                                                 rectangles,
                                                 triangleData,
                                                 tlasNodes,
                                                 tlasPrimIndices,
                                                 instanceInfos,
                                                 blasNodes,
                                                 blasPrimIndices,
                                                 nodes,
                                                 primitiveIndices,
                                                 stats,
                                                 mneeRay,
                                                 kEpsilon,
                                                 kInfinity,
                                                 /*anyHitOnly=*/false,
                                                 /*includeTriangles=*/true,
                                                 lightRec);
            if (hitLight) {
                MneeRectHit mneeHit;
                if (mnee_rect_light_hit(uniforms,
                                        rectangles,
                                        materials,
                                        environmentTexture,
                                        rectLightCount,
                                        lightRec,
                                        nextOrigin,
                                        mneeHit)) {
                    float lightPdf = max(mneeHit.pdf, kSpecularNeePdfFloor);
                    float invLightPdf = min(1.0f / lightPdf, kSpecularNeeInvPdfClamp);
                    float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                    float denom = lightPdf + bsdfPdf;
                    float misWeight = (denom > 0.0f) ? (lightPdf / denom) : 0.0f;
                    misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                    float3 contribution = bsdfSample.weight * mneeHit.emission *
                                          (misWeight * invLightPdf);
                    if (all(isfinite(contribution))) {
                        radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                        if (stats) {
                            atomic_fetch_add_explicit(&stats->mneeRectAddedCount, 1u, memory_order_relaxed);
                            stats_add_mnee_luma(stats, contribution);
                        }
                    }
                }
            }
        }

        if (mneeEligible && uniforms.enableMneeSecondary != 0u) {
            Ray chainRay;
            chainRay.origin = nextOrigin;
            chainRay.direction = normalize(bsdfSample.direction);
            HitRecord chainRec;
            bool chainHit = trace_scene_software(uniforms,
                                                 spheres,
                                                 rectangles,
                                                 triangleData,
                                                 tlasNodes,
                                                 tlasPrimIndices,
                                                 instanceInfos,
                                                 blasNodes,
                                                 blasPrimIndices,
                                                 nodes,
                                                 primitiveIndices,
                                                 stats,
                                                 chainRay,
                                                 kEpsilon,
                                                 kInfinity,
                                                 /*anyHitOnly=*/false,
                                                 /*includeTriangles=*/true,
                                                 chainRec);
            if (chainHit && materials && uniforms.materialCount > 0u) {
                bool chainHitIsLight = false;
                if (rectLightCount > 0u) {
                    MneeRectHit chainLightHit;
                    if (mnee_rect_light_hit(uniforms,
                                            rectangles,
                                            materials,
                                            environmentTexture,
                                            rectLightCount,
                                            chainRec,
                                            chainRay.origin,
                                            chainLightHit)) {
                        chainHitIsLight = true;
                    }
                }
                if (!chainHitIsLight) {
                    uint chainMatIndex = min(chainRec.materialIndex, uniforms.materialCount - 1u);
                    MaterialData chainMaterial = materials[chainMatIndex];
                    if (material_is_delta(chainMaterial)) {
                        float3 chainNormal = chainRec.normal;
                        if (!all(isfinite(chainNormal)) || dot(chainNormal, chainNormal) <= 0.0f) {
                            chainNormal = float3(0.0f, 1.0f, 0.0f);
                        }
                        chainNormal = normalize(chainNormal);
                        float3 chainIncident = normalize(chainRay.direction);
                        float3 chainWo = -chainIncident;
                        uint chainState = state;
                        BsdfSampleResult chainSample = sample_bsdf(chainMaterial,
                                                                   chainRec.point,
                                                                   chainNormal,
                                                                   chainWo,
                                                                   chainIncident,
                                                                   chainRec.frontFace != 0u,
                                                                   chainState,
                                                                   clampParams,
                                                                   uniforms.sssMode,
                                                                   1.0f,
                                                                   specularOnly);
                        if (chainSample.pdf > 0.0f &&
                            chainSample.isDelta &&
                            (chainSample.mediumEvent <= 0)) {
                            float3 chainDir = safe_normalize(chainSample.direction);
                            if (all(isfinite(chainDir)) && dot(chainDir, chainDir) > 0.0f) {
                                float3 chainOrigin = offset_ray_origin(chainRec, chainDir);
                                float3 combinedWeight = bsdfSample.weight * chainSample.weight;
                                float bsdfPdf = max(bsdfSample.directionalPdf * chainSample.directionalPdf,
                                                    kSpecularNeePdfFloor);
                                if (envSampling &&
                                    environmentTexture.get_width() > 0 &&
                                    environmentTexture.get_height() > 0) {
                                    Ray envRay;
                                    envRay.origin = chainOrigin;
                                    envRay.direction = normalize(chainDir);
                                    HitRecord envRec;
                                    bool occluded = trace_scene_software(uniforms,
                                                                         spheres,
                                                                         rectangles,
                                                                         triangleData,
                                                                         tlasNodes,
                                                                         tlasPrimIndices,
                                                                         instanceInfos,
                                                                         blasNodes,
                                                                         blasPrimIndices,
                                                                         nodes,
                                                                         primitiveIndices,
                                                                         stats,
                                                                         envRay,
                                                                         kEpsilon,
                                                                         kInfinity,
                                                                         /*anyHitOnly=*/true,
                                                                         /*includeTriangles=*/true,
                                                                         envRec);
                                    if (!occluded) {
                                        float envPdf = environment_pdf(uniforms, environmentPdf, envRay.direction);
                                        envPdf = max(envPdf, kSpecularNeePdfFloor);
                                        float invEnvPdf = min(1.0f / envPdf, kSpecularNeeInvPdfClamp);
                                        float denom = envPdf + bsdfPdf;
                                        float misWeight = (denom > 0.0f) ? (envPdf / denom) : 0.0f;
                                        misWeight = clamp(misWeight,
                                                          kMisWeightClampMin,
                                                          kMisWeightClampMax);
                                        float3 envColor = environment_color(environmentTexture,
                                                                            envRay.direction,
                                                                            uniforms.environmentRotation,
                                                                            uniforms.environmentIntensity,
                                                                            uniforms);
                                        float3 contribution = combinedWeight * envColor *
                                                              (misWeight * invEnvPdf);
                                        if (all(isfinite(contribution))) {
                                            radiance += clamp_firefly_contribution(throughput,
                                                                                   contribution,
                                                                                   clampParams);
                                        }
                                    }
                                }
                                if (rectLightCount > 0u) {
                                    Ray lightRay;
                                    lightRay.origin = chainOrigin;
                                    lightRay.direction = normalize(chainDir);
                                    HitRecord lightRec;
                                    bool hitLight = trace_scene_software(uniforms,
                                                                         spheres,
                                                                         rectangles,
                                                                         triangleData,
                                                                         tlasNodes,
                                                                         tlasPrimIndices,
                                                                         instanceInfos,
                                                                         blasNodes,
                                                                         blasPrimIndices,
                                                                         nodes,
                                                                         primitiveIndices,
                                                                         stats,
                                                                         lightRay,
                                                                         kEpsilon,
                                                                         kInfinity,
                                                                         /*anyHitOnly=*/false,
                                                                         /*includeTriangles=*/true,
                                                                         lightRec);
                                    if (hitLight) {
                                        MneeRectHit mneeHit;
                                        if (mnee_rect_light_hit(uniforms,
                                                                rectangles,
                                                                materials,
                                                                environmentTexture,
                                                                rectLightCount,
                                                                lightRec,
                                                                chainOrigin,
                                                                mneeHit)) {
                                            float lightPdf = max(mneeHit.pdf, kSpecularNeePdfFloor);
                                            float invLightPdf = min(1.0f / lightPdf, kSpecularNeeInvPdfClamp);
                                            float denom = lightPdf + bsdfPdf;
                                            float misWeight = (denom > 0.0f) ? (lightPdf / denom) : 0.0f;
                                            misWeight = clamp(misWeight,
                                                              kMisWeightClampMin,
                                                              kMisWeightClampMax);
                                            float3 contribution = combinedWeight * mneeHit.emission *
                                                                  (misWeight * invLightPdf);
                                            if (all(isfinite(contribution))) {
                                                radiance += clamp_firefly_contribution(throughput,
                                                                                       contribution,
                                                                                       clampParams);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

#endif

        throughput *= bsdfSample.weight;
        throughput = clamp_path_throughput(throughput, clampParams);

        if (!all(isfinite(throughput))) {
            break;
        }

        float maxThroughput = max(max(throughput.x, throughput.y), throughput.z);
        if (maxThroughput <= 0.0f) {
            break;
        }

        bool nextEnvLodActive = false;
        float nextEnvLod = 0.0f;
        if (bsdfSample.lobeType == 1u && !bsdfSample.isDelta) {
            float maxMip = environment_max_mip(environmentTexture);
            if (maxMip > 0.0f) {
                nextEnvLod = environment_lod_from_roughness(bsdfSample.lobeRoughness,
                                                            environmentTexture);
                nextEnvLodActive = true;
            }
        }
        envLodActive = nextEnvLodActive;
        envLod = nextEnvLod;

        rayCone.width = ray_cone_width_at_distance(rayCone, hitDistanceWorld);
        rayCone.spread = min(rayCone.spread +
                             bsdf_cone_spread_increment(bsdfSample.lobeType,
                                                        bsdfSample.lobeRoughness,
                                                        bsdfSample.isDelta),
                             1.5f);

        lastBsdfPdf = (bsdfSample.directionalPdf > 0.0f) ? bsdfSample.directionalPdf : bsdfSample.pdf;
        lastScatterWasDelta = bsdfSample.isDelta;
        ray.origin = nextOrigin;
        ray.direction = bsdfSample.direction;

        if (uniforms.useRussianRoulette != 0 && depth >= 5) {
            float continueProbability = clamp(maxThroughput, 0.05f, 0.95f);
            if (rand_uniform(state) > continueProbability) {
                break;
            }
            throughput /= continueProbability;
        }
    }

    return radiance;
}

#if __METAL_VERSION__ >= 310
inline float3 trace_path_hardware(constant PathtraceUniforms& uniforms,
                                  acceleration_structure<instancing> accel,
                                  device const MeshInfo* meshInfos,
                                  device const TriangleData* triangleData,
                                  device const uint* instanceUserIds,
                                  device const SphereData* spheres,
                                  device const RectData* rectangles,
                                  device const MaterialData* materials,
                                  device const SceneVertex* sceneVertices,
                                  device const uint3* meshIndices,
                                  device const BvhNode* tlasNodes,
                                  device const uint* tlasPrimIndices,
                                  device const BvhNode* blasNodes,
                                  device const uint* blasPrimIndices,
                                  device const SoftwareInstanceInfo* instanceInfos,
                                  Ray ray,
                                  const PrimaryRayDiff primaryRayDiff,
                                  thread uint& state,
                                  device const BvhNode* nodes,
                                  device const uint* primitiveIndices,
                                  device PathtraceStats* stats,
                                  texture2d<float, access::sample> environmentTexture,
                                  array<texture2d<float, access::sample>, kMaxMaterialTextures> materialTextures,
                                  array<sampler, kMaxMaterialSamplers> materialSamplers,
                                  device const MaterialTextureInfo* materialTextureInfos,
                                  device const EnvironmentAliasEntry* environmentConditionalAlias,
                                  device const EnvironmentAliasEntry* environmentMarginalAlias,
                                  device const float* environmentPdf,
                                  // Optional AOV outputs
                                  thread float3* outFirstHitAlbedo = nullptr,
                                  thread float3* outFirstHitNormal = nullptr,
                                  thread PathtraceDebugContext* debugContext = nullptr) {
    float3 throughput = float3(1.0f, 1.0f, 1.0f);
    float3 radiance = float3(0.0f, 0.0f, 0.0f);
    float lastBsdfPdf = 1.0f;
    bool lastScatterWasDelta = true;
    bool isFirstHit = true;
    float envLod = 0.0f;
    bool envLodActive = false;
    RayCone rayCone = make_primary_ray_cone(uniforms);
    uint rectLightCount = (rectangles && uniforms.rectangleCount > 0 && materials)
                              ? count_rect_lights(uniforms, rectangles, materials)
                              : 0u;
    const bool envSampling = environment_sampling_available(uniforms,
                                                            environmentConditionalAlias,
                                                            environmentMarginalAlias,
                                                            environmentPdf);
    FireflyClampParams clampParams = make_firefly_params(uniforms);
    constexpr uint kMaxMediumStack = 8u;
    float3 mediumSigmaStack[kMaxMediumStack];
    for (uint i = 0; i < kMaxMediumStack; ++i) {
        mediumSigmaStack[i] = float3(0.0f);
    }
    uint mediumDepth = 0u;
    HitRecord prevRec;
    bool prevValid = false;
    uint specularDepth = 0u;
    bool hadTransmission = false;
    bool parityInMediumDone = false;
    const bool softwareTrianglesAvailable =
        (tlasNodes && tlasPrimIndices && blasNodes && blasPrimIndices && instanceInfos && triangleData);
    const bool forcePureHWRTForGlass = (uniforms.forcePureHWRTForGlass != 0u);
    const bool enableMissFallback =
        !forcePureHWRTForGlass && (uniforms.enableHardwareMissFallback != 0u);
    const bool enableFirstHitFallback =
        !forcePureHWRTForGlass && (uniforms.enableHardwareFirstHitFromSoftware != 0u);
    const bool forceSoftware =
        !forcePureHWRTForGlass &&
        (uniforms.enableHardwareForceSoftware != 0u) &&
        softwareTrianglesAvailable;

    for (uint depth = 0; depth < uniforms.maxDepth; ++depth) {
        HitRecord rec;
        uint excludeMeshIndex = kInvalidIndex;
        uint excludePrimitiveIndex = kInvalidIndex;
        if (prevValid) {
            compute_exclusion_indices(prevRec, excludeMeshIndex, excludePrimitiveIndex);
        }
        const bool preferSoftwareForMedium =
            !forcePureHWRTForGlass && (mediumDepth > 0u) && softwareTrianglesAvailable;

        bool doParity = false;
#if PT_DEBUG_TOOLS
        if (debugContext &&
            uniforms.parityAssertEnabled != 0u &&
            uniforms.parityAssertMode != 0u &&
            debugContext->buffer != nullptr &&
            debugContext->pixelX == uniforms.parityPixelX &&
            debugContext->pixelY == uniforms.parityPixelY) {
            if (uniforms.parityAssertMode == kParityModeProbePixel) {
                doParity = (depth == 0u);
            } else if (uniforms.parityAssertMode == kParityModeFirstInMedium) {
                if (!parityInMediumDone && mediumDepth > 0u) {
                    doParity = true;
                    parityInMediumDone = true;
                }
            }
        }
#endif

#if PT_DEBUG_TOOLS
        if (doParity) {
            uint parityAllowed = min(debugContext->buffer->parityMaxEntries,
                                     kPathtraceParityMaxEntries);
            if (parityAllowed > 0u) {
                uint parityRecorded =
                    atomic_load_explicit(&debugContext->buffer->parityWriteIndex,
                                         memory_order_relaxed);
                if (parityRecorded < parityAllowed) {
                    atomic_fetch_add_explicit(&debugContext->buffer->parityChecksPerformed,
                                              1u,
                                              memory_order_relaxed);
                    if (mediumDepth > 0u) {
                        atomic_fetch_add_explicit(&debugContext->buffer->parityChecksInMedium,
                                                  1u,
                                                  memory_order_relaxed);
                    }
                    HitRecord hwRec = make_empty_hit_record();
                    HitRecord swRec = make_empty_hit_record();
                    bool hwHit = trace_scene_hardware(uniforms,
                                                      accel,
                                                      meshInfos,
                                                      triangleData,
                                                      sceneVertices,
                                                      meshIndices,
                                                      instanceUserIds,
                                                      spheres,
                                                      rectangles,
                                                      nodes,
                                                      primitiveIndices,
                                                      /*stats=*/nullptr,
                                                      ray,
                                                      kEpsilon,
                                                      kInfinity,
                                                      /*anyHitOnly=*/false,
                                                      excludeMeshIndex,
                                                      excludePrimitiveIndex,
                                                      hwRec);
                    bool swHit = trace_scene_software_with_exclusion(uniforms,
                                                                     spheres,
                                                                     rectangles,
                                                                     triangleData,
                                                                     tlasNodes,
                                                                     tlasPrimIndices,
                                                                     instanceInfos,
                                                                     blasNodes,
                                                                     blasPrimIndices,
                                                                     nodes,
                                                                     primitiveIndices,
                                                                     /*stats=*/nullptr,
                                                                     ray,
                                                                     kEpsilon,
                                                                     kInfinity,
                                                                     excludeMeshIndex,
                                                                     excludePrimitiveIndex,
                                                                     swRec);
                    uint reasonMask = 0u;
                    if (hwHit != swHit) {
                        reasonMask |= kParityReasonHitMiss;
                    }
                    if (hwHit && swHit) {
                        float epsT = max(1.0e-3f, 1.0e-4f * fabs(hwRec.t));
                        float tDiff = fabs(hwRec.t - swRec.t);
                        if (tDiff > epsT) {
                            reasonMask |= kParityReasonT;
                        }
                        float3 hwN = hwRec.normal;
                        float3 swN = swRec.normal;
                        if (all(isfinite(hwN)) && all(isfinite(swN)) &&
                            dot(hwN, hwN) > 0.0f && dot(swN, swN) > 0.0f) {
                            float nDot = dot(normalize(hwN), normalize(swN));
                            if (nDot < 0.99f) {
                                reasonMask |= kParityReasonNormal;
                            }
                        }
                        if (hwRec.frontFace != swRec.frontFace) {
                            reasonMask |= kParityReasonFrontFace;
                        }
                        if (hwRec.materialIndex != swRec.materialIndex ||
                            hwRec.meshIndex != swRec.meshIndex ||
                            hwRec.primitiveIndex != swRec.primitiveIndex) {
                            reasonMask |= kParityReasonId;
                        }
                    }
                    record_parity_entry(*debugContext,
                                        uniforms,
                                        depth,
                                        ray,
                                        kEpsilon,
                                        kInfinity,
                                        hwHit,
                                        hwRec,
                                        swHit,
                                        swRec,
                                        reasonMask);
                }
            }
        }
#endif

        bool hit = false;
#if PT_DEBUG_TOOLS
        if (forceSoftware) {
            hit = trace_scene_software(uniforms,
                                       spheres,
                                       rectangles,
                                       triangleData,
                                       tlasNodes,
                                       tlasPrimIndices,
                                       instanceInfos,
                                       blasNodes,
                                       blasPrimIndices,
                                       nodes,
                                       primitiveIndices,
                                       stats,
                                       ray,
                                       kEpsilon,
                                       kInfinity,
                                       /*anyHitOnly=*/false,
                                       /*includeTriangles=*/true,
                                       rec);
            if (hit && stats) {
                atomic_fetch_add_explicit(&stats->hardwareFallbackHitCount,
                                          1u,
                                          memory_order_relaxed);
            }
        } else if (depth == 0u && enableFirstHitFallback && softwareTrianglesAvailable) {
            hit = trace_scene_software(uniforms,
                                       spheres,
                                       rectangles,
                                       triangleData,
                                       tlasNodes,
                                       tlasPrimIndices,
                                       instanceInfos,
                                       blasNodes,
                                       blasPrimIndices,
                                       nodes,
                                       primitiveIndices,
                                       stats,
                                       ray,
                                       kEpsilon,
                                       kInfinity,
                                       /*anyHitOnly=*/false,
                                       /*includeTriangles=*/true,
                                       rec);
            if (hit && stats) {
                atomic_fetch_add_explicit(&stats->hardwareFirstHitFallbackCount,
                                          1u,
                                          memory_order_relaxed);
            }
        }

        if (!hit && preferSoftwareForMedium && softwareTrianglesAvailable) {
            hit = trace_scene_software(uniforms,
                                       spheres,
                                       rectangles,
                                       triangleData,
                                       tlasNodes,
                                       tlasPrimIndices,
                                       instanceInfos,
                                       blasNodes,
                                       blasPrimIndices,
                                       nodes,
                                       primitiveIndices,
                                       stats,
                                       ray,
                                       kEpsilon,
                                       kInfinity,
                                       /*anyHitOnly=*/false,
                                       /*includeTriangles=*/true,
                                       rec);
            if (hit && stats) {
                if (depth == 0u) {
                    atomic_fetch_add_explicit(&stats->hardwareFirstHitFallbackCount,
                                              1u,
                                              memory_order_relaxed);
                } else {
                    atomic_fetch_add_explicit(&stats->hardwareFallbackHitCount,
                                              1u,
                                              memory_order_relaxed);
                }
            }
        }
#endif

        if (!hit && !forceSoftware) {
            hit = trace_scene_hardware(uniforms,
                                       accel,
                                       meshInfos,
                                       triangleData,
                                       sceneVertices,
                                       meshIndices,
                                       instanceUserIds,
                                       spheres,
                                       rectangles,
                                       nodes,
                                       primitiveIndices,
                                       stats,
                                       ray,
                                       kEpsilon,
                                       kInfinity,
                                       /*anyHitOnly=*/false,
                                       excludeMeshIndex,
                                       excludePrimitiveIndex,
                                       rec);
        }

        if (!hit && !forceSoftware && enableMissFallback && softwareTrianglesAvailable) {
#if PT_DEBUG_TOOLS
            hit = trace_scene_software(uniforms,
                                       spheres,
                                       rectangles,
                                       triangleData,
                                       tlasNodes,
                                       tlasPrimIndices,
                                       instanceInfos,
                                       blasNodes,
                                       blasPrimIndices,
                                       nodes,
                                       primitiveIndices,
                                       stats,
                                       ray,
                                       kEpsilon,
                                       kInfinity,
                                       /*anyHitOnly=*/false,
                                       /*includeTriangles=*/true,
                                       rec);
            if (hit && stats) {
                atomic_fetch_add_explicit(&stats->hardwareFallbackHitCount,
                                          1u,
                                          memory_order_relaxed);
            }
#endif
        }

        if (!hit) {
            if (uniforms.debugViewMode != kDebugViewNone) {
                return float3(0.0f);
            }
            float3 background = sky_color(ray.direction);
            if (uniforms.backgroundMode == 1u) {
                background = uniforms.backgroundColor;
            } else if (uniforms.backgroundMode == 2u && environmentTexture.get_width() > 0 && environmentTexture.get_height() > 0) {
                float overrideLod = 0.0f;
                bool useOverride = environment_mip_override(uniforms, environmentTexture, overrideLod);
                if (useOverride) {
                    background = environment_color_lod(environmentTexture,
                                                       ray.direction,
                                                       uniforms.environmentRotation,
                                                       uniforms.environmentIntensity,
                                                       overrideLod,
                                                       uniforms);
                } else if (envLodActive) {
                    background = environment_color_lod(environmentTexture,
                                                       ray.direction,
                                                       uniforms.environmentRotation,
                                                       uniforms.environmentIntensity,
                                                       envLod,
                                                       uniforms);
                } else {
                    background = environment_color(environmentTexture,
                                                   ray.direction,
                                                   uniforms.environmentRotation,
                                                   uniforms.environmentIntensity,
                                                   uniforms);
                }
            }
            if (uniforms.backgroundMode != 2u) {
                background = to_working_space(background, uniforms);
            }
            if (debugContext) {
                record_debug_event(*debugContext,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   /*frontFace=*/0u,
                                   kInvalidIndex,
                                   /*scatterIsDelta=*/false,
                                   throughput);
            }
            float misWeight = 1.0f;
            bool useSpecularMis = (!lastScatterWasDelta) ||
                                  (uniforms.enableSpecularNee != 0u) ||
                                  ((ENABLE_MNEE_CAUSTICS != 0) && (uniforms.enableMnee != 0u));
            if (useSpecularMis && envSampling) {
                float lightPdf = environment_pdf(uniforms, environmentPdf, ray.direction);
                float denom = lastBsdfPdf + lightPdf;
                if (denom > 0.0f) {
                    misWeight = clamp(lastBsdfPdf / denom,
                                      kMisWeightClampMin,
                                      kMisWeightClampMax);
                }
            }
            float3 contribution = background * misWeight;
            radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
            break;
        }
        prevRec = rec;
        prevValid = true;

        if (!materials || uniforms.materialCount == 0) {
            break;
        }
        if (mediumDepth > 0u) {
            float3 sigma = mediumSigmaStack[mediumDepth - 1u];
            if (any(sigma > float3(0.0f))) {
                float segment = max(rec.t, 0.0f);
                float3 attenuation = exp(-sigma * segment);
                throughput *= attenuation;
            }
        }
        uint matIndex = min(rec.materialIndex, uniforms.materialCount - 1);
        MaterialData material = materials[matIndex];
        uint type = static_cast<uint>(material.typeEta.x);
        float3 incidentDir = normalize(ray.direction);
        float3 wo = -incidentDir;
        float hitDistanceWorld = ray_segment_world_length(ray, rec.t);
        bool surfaceIsDelta = material_is_delta(material);
        bool specularOnly = (uniforms.debugSpecularOnly != 0u);
        float diffuseOcclusion = 1.0f;
        float3 debugBaseColor = material_base_color(material);
        float debugMetallic = 0.0f;
        float debugRoughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        float debugAO = 1.0f;
        float3 shadingNormal = rec.shadingNormal;
        if (!all(isfinite(shadingNormal)) || dot(shadingNormal, shadingNormal) <= 0.0f) {
            shadingNormal = rec.normal;
        }
        if (rec.primitiveType == kPrimitiveTypeTriangle &&
            meshInfos && sceneVertices && meshIndices && uniforms.meshCount > 0u) {
            float3 candidate = interpolate_shading_normal(uniforms,
                                                          rec.meshIndex,
                                                          rec.primitiveIndex,
                                                          rec.barycentric,
                                                          meshInfos,
                                                          sceneVertices,
                                                          meshIndices);
            if (all(isfinite(candidate)) && dot(candidate, candidate) > 0.0f) {
                if (dot(candidate, rec.normal) < 0.0f) {
                    candidate = -candidate;
                }
                shadingNormal = normalize(candidate);
            }
        }
        if (type == 2u) { // Dielectric: force geometric normal for shading.
            float3 geomNormal = rec.normal;
            if (all(isfinite(geomNormal)) && dot(geomNormal, geomNormal) > 0.0f) {
                shadingNormal = geomNormal;
            }
            // Keep ray offsets consistent between SWRT/HWRT for glass.
            rec.shadingNormal = shadingNormal;
        }

        if (type == 7u && rec.primitiveType == kPrimitiveTypeTriangle &&
            meshInfos && sceneVertices && meshIndices && uniforms.meshCount > 0u) {
            float2 uv0 = interpolate_uv(uniforms,
                                        rec.meshIndex,
                                        rec.primitiveIndex,
                                        rec.barycentric,
                                        0u,
                                        meshInfos,
                                        sceneVertices,
                                        meshIndices);
            float2 uv1 = interpolate_uv(uniforms,
                                        rec.meshIndex,
                                        rec.primitiveIndex,
                                        rec.barycentric,
                                        1u,
                                        meshInfos,
                                        sceneVertices,
                                        meshIndices);
            float4 tangent = interpolate_tangent(uniforms,
                                                 rec.meshIndex,
                                                 rec.primitiveIndex,
                                                 rec.barycentric,
                                                 meshInfos,
                                                 sceneVertices,
                                                 meshIndices);
            if (material.typeEta.z > 0.5f) {
                rec.twoSided = 1u;
            }
            float coneFootprintWorld = ray_cone_width_at_distance(rayCone, hitDistanceWorld);
            float surfaceFootprintWorld =
                surface_footprint_from_cone(coneFootprintWorld, rec.normal, wo);
            float3 dPdu0 = float3(0.0f);
            float3 dPdv0 = float3(0.0f);
            float uvPerWorld0 = 0.0f;
            bool hasSurfacePartials0 = triangle_surface_partials(uniforms,
                                                                 rec.meshIndex,
                                                                 rec.primitiveIndex,
                                                                 0u,
                                                                 meshInfos,
                                                                 sceneVertices,
                                                                 meshIndices,
                                                                 dPdu0,
                                                                 dPdv0,
                                                                 uvPerWorld0);
            float3 dPdu1 = float3(0.0f);
            float3 dPdv1 = float3(0.0f);
            float uvPerWorld1 = 0.0f;
            bool hasSurfacePartials1 = triangle_surface_partials(uniforms,
                                                                 rec.meshIndex,
                                                                 rec.primitiveIndex,
                                                                 1u,
                                                                 meshInfos,
                                                                 sceneVertices,
                                                                 meshIndices,
                                                                 dPdu1,
                                                                 dPdv1,
                                                                 uvPerWorld1);
            float2 dUVdx0 = float2(0.0f);
            float2 dUVdy0 = float2(0.0f);
            bool hasIgehyGradients0 = false;
            if (depth == 0u && hasSurfacePartials0) {
                float3 dudP = float3(0.0f);
                float3 dvdP = float3(0.0f);
                if (uv_world_gradients_from_partials(dPdu0, dPdv0, dudP, dvdP)) {
                    hasIgehyGradients0 = first_hit_uv_gradients_igehy(ray,
                                                                      primaryRayDiff,
                                                                      rec.t,
                                                                      rec.normal,
                                                                      dudP,
                                                                      dvdP,
                                                                      dUVdx0,
                                                                      dUVdy0);
                }
            }
            float2 dUVdx1 = float2(0.0f);
            float2 dUVdy1 = float2(0.0f);
            bool hasIgehyGradients1 = false;
            if (depth == 0u && hasSurfacePartials1) {
                float3 dudP = float3(0.0f);
                float3 dvdP = float3(0.0f);
                if (uv_world_gradients_from_partials(dPdu1, dPdv1, dudP, dvdP)) {
                    hasIgehyGradients1 = first_hit_uv_gradients_igehy(ray,
                                                                      primaryRayDiff,
                                                                      rec.t,
                                                                      rec.normal,
                                                                      dudP,
                                                                      dvdP,
                                                                      dUVdx1,
                                                                      dUVdy1);
                }
            }

            PbrTextureSamplingContext baseColorCtx = make_pbr_texture_sampling_context(material,
                                                                                        kPbrTextureSlotBaseColor,
                                                                                        uv0,
                                                                                        uv1,
                                                                                        hasIgehyGradients0,
                                                                                        dUVdx0,
                                                                                        dUVdy0,
                                                                                        uvPerWorld0,
                                                                                        hasIgehyGradients1,
                                                                                        dUVdx1,
                                                                                        dUVdy1,
                                                                                        uvPerWorld1);
            PbrTextureSamplingContext ormCtx = make_pbr_texture_sampling_context(material,
                                                                                  kPbrTextureSlotMetallicRoughness,
                                                                                  uv0,
                                                                                  uv1,
                                                                                  hasIgehyGradients0,
                                                                                  dUVdx0,
                                                                                  dUVdy0,
                                                                                  uvPerWorld0,
                                                                                  hasIgehyGradients1,
                                                                                  dUVdx1,
                                                                                  dUVdy1,
                                                                                  uvPerWorld1);
            PbrTextureSamplingContext normalCtx = make_pbr_texture_sampling_context(material,
                                                                                     kPbrTextureSlotNormal,
                                                                                     uv0,
                                                                                     uv1,
                                                                                     hasIgehyGradients0,
                                                                                     dUVdx0,
                                                                                     dUVdy0,
                                                                                     uvPerWorld0,
                                                                                     hasIgehyGradients1,
                                                                                     dUVdx1,
                                                                                     dUVdy1,
                                                                                     uvPerWorld1);
            PbrTextureSamplingContext occlusionCtx = make_pbr_texture_sampling_context(material,
                                                                                        kPbrTextureSlotOcclusion,
                                                                                        uv0,
                                                                                        uv1,
                                                                                        hasIgehyGradients0,
                                                                                        dUVdx0,
                                                                                        dUVdy0,
                                                                                        uvPerWorld0,
                                                                                        hasIgehyGradients1,
                                                                                        dUVdx1,
                                                                                        dUVdy1,
                                                                                        uvPerWorld1);
            PbrTextureSamplingContext emissiveCtx = make_pbr_texture_sampling_context(material,
                                                                                       kPbrTextureSlotEmissive,
                                                                                       uv0,
                                                                                       uv1,
                                                                                       hasIgehyGradients0,
                                                                                       dUVdx0,
                                                                                       dUVdy0,
                                                                                       uvPerWorld0,
                                                                                       hasIgehyGradients1,
                                                                                       dUVdx1,
                                                                                       dUVdy1,
                                                                                       uvPerWorld1);
            PbrTextureSamplingContext transmissionCtx = make_pbr_texture_sampling_context(material,
                                                                                           kPbrTextureSlotTransmission,
                                                                                           uv0,
                                                                                           uv1,
                                                                                           hasIgehyGradients0,
                                                                                           dUVdx0,
                                                                                           dUVdy0,
                                                                                           uvPerWorld0,
                                                                                           hasIgehyGradients1,
                                                                                           dUVdx1,
                                                                                           dUVdy1,
                                                                                           uvPerWorld1);
            float3 baseFactor = to_working_space(material.baseColorRoughness.xyz, uniforms);
            float baseColorLod =
                material_texture_lod_with_fallback(materialTextures,
                                                   materialTextureInfos,
                                                   uniforms,
                                                   material.textureIndices0.x,
                                                   baseColorCtx.hasIgehyGradients,
                                                   baseColorCtx.dUVdx,
                                                   baseColorCtx.dUVdy,
                                                   baseColorCtx.uvPerWorld,
                                                   surfaceFootprintWorld);
            float4 baseColorSample =
                sample_material_texture_filtered(materialTextures,
                                                materialSamplers,
                                                materialTextureInfos,
                                                uniforms,
                                                material.textureIndices0.x,
                                                baseColorCtx.uv,
                                                float4(1.0f),
                                                baseColorLod,
                                                baseColorCtx.hasIgehyGradients,
                                                baseColorCtx.dUVdx,
                                                baseColorCtx.dUVdy);
            float3 baseColorSampleRgb = to_working_space(baseColorSample.xyz, uniforms);
            float3 baseColor = baseFactor * baseColorSampleRgb;

            float metallic = clamp(material.pbrParams.x, 0.0f, 1.0f);
            float roughness = clamp(material.pbrParams.y, 0.0f, 1.0f);
            float normalStrengthScale = 1.0f;
#if PT_DEBUG_TOOLS
            normalStrengthScale = max(uniforms.debugNormalStrengthScale, 0.0f);
#endif
            float normalScale = material.pbrParams.w * normalStrengthScale;
            bool disableOrmByMaterial = (material.materialFlags & kMaterialFlagDisableOrm) != 0u;
            bool useOrmTexture = !disableOrmByMaterial &&
                                 material_texture_valid(uniforms, material.textureIndices0.y);
#if PT_DEBUG_TOOLS
            useOrmTexture = useOrmTexture && (uniforms.debugDisableOrmTexture == 0u);
#endif
            if (useOrmTexture) {
                float ormLod =
                    material_texture_lod_with_fallback(materialTextures,
                                                       materialTextureInfos,
                                                       uniforms,
                                                       material.textureIndices0.y,
                                                       ormCtx.hasIgehyGradients,
                                                       ormCtx.dUVdx,
                                                       ormCtx.dUVdy,
                                                       ormCtx.uvPerWorld,
                                                       surfaceFootprintWorld);
#if PT_DEBUG_TOOLS
                ormLod = max(ormLod + uniforms.debugOrmLodBias, 0.0f);
#endif
                float3 mrSample =
                    sample_material_texture_level(materialTextures,
                                                  materialSamplers,
                                                  materialTextureInfos,
                                                  uniforms,
                                                  material.textureIndices0.y,
                                                  ormCtx.uv,
                                                  float4(1.0f),
                                                  ormLod).xyz;
                metallic = clamp(mrSample.z * metallic, 0.0f, 1.0f);
                roughness = clamp(mrSample.y * roughness, 0.0f, 1.0f);
            }
            float visorMask = visor_override_blend(baseColor, metallic, roughness, matIndex, uniforms);
            if (visorMask > 0.0f) {
                    float overrideRoughness =
                        clamp(uniforms.debugVisorOverrideRoughness, 0.0f, 1.0f);
                    float overrideF0 = clamp(uniforms.debugVisorOverrideF0, 0.0f, 0.12f);
                    metallic = mix(metallic, 0.0f, visorMask);
                    roughness = mix(roughness, overrideRoughness, visorMask);
                    material.typeEta.y = mix(material.typeEta.y,
                                             ior_from_f0(overrideF0),
                                             visorMask);
            }
            float normalLod =
                material_texture_lod_with_fallback(materialTextures,
                                                   materialTextureInfos,
                                                   uniforms,
                                                   material.textureIndices0.z,
                                                   normalCtx.hasIgehyGradients,
                                                   normalCtx.dUVdx,
                                                   normalCtx.dUVdy,
                                                   normalCtx.uvPerWorld,
                                                   surfaceFootprintWorld);
#if PT_DEBUG_TOOLS
            normalLod = max(normalLod + uniforms.debugNormalLodBias, 0.0f);
#endif

            float transmission = clamp(material.pbrExtras.z, 0.0f, 1.0f);
            if (material_texture_valid(uniforms, material.textureIndices1.y)) {
                float transmissionLod =
                    material_texture_lod_with_fallback(materialTextures,
                                                       materialTextureInfos,
                                                       uniforms,
                                                       material.textureIndices1.y,
                                                       transmissionCtx.hasIgehyGradients,
                                                       transmissionCtx.dUVdx,
                                                       transmissionCtx.dUVdy,
                                                       transmissionCtx.uvPerWorld,
                                                       surfaceFootprintWorld);
                float transmissionSample =
                    sample_material_texture_filtered(materialTextures,
                                                    materialSamplers,
                                                    materialTextureInfos,
                                                    uniforms,
                                                    material.textureIndices1.y,
                                                    transmissionCtx.uv,
                                                    float4(1.0f),
                                                    transmissionLod,
                                                    transmissionCtx.hasIgehyGradients,
                                                    transmissionCtx.dUVdx,
                                                    transmissionCtx.dUVdy).x;
                transmission = clamp(transmission * transmissionSample, 0.0f, 1.0f);
            }
            transmission *= (1.0f - metallic);

            float alpha = clamp(material.pbrExtras.x, 0.0f, 1.0f);
            alpha = clamp(alpha * baseColorSample.w, 0.0f, 1.0f);
            float alphaCutoff = clamp(material.pbrExtras.y, 0.0f, 1.0f);
            float alphaMode = material.pbrExtras.w;
            if (alphaMode > 0.5f) {
                bool discard = false;
                if (alphaMode < 1.5f) {
                    discard = alpha < alphaCutoff;
                } else {
                    discard = rand_uniform(state) > alpha;
                }
                if (discard) {
                    ray.origin = offset_ray_origin(rec, ray.direction);
                    prevRec = rec;
                    prevValid = true;
                    lastBsdfPdf = 1.0f;
                    lastScatterWasDelta = true;
                    specularDepth += 1u;
                    continue;
                }
            }

            material.pbrExtras.z = transmission;

            float occlusion = 1.0f;
            if (!disableOrmByMaterial && material_texture_valid(uniforms, material.textureIndices0.w)) {
                float occlusionLod =
                    material_texture_lod_with_fallback(materialTextures,
                                                       materialTextureInfos,
                                                       uniforms,
                                                       material.textureIndices0.w,
                                                       occlusionCtx.hasIgehyGradients,
                                                       occlusionCtx.dUVdx,
                                                       occlusionCtx.dUVdy,
                                                       occlusionCtx.uvPerWorld,
                                                       surfaceFootprintWorld);
                float occSample =
                    sample_material_texture_filtered(materialTextures,
                                                    materialSamplers,
                                                    materialTextureInfos,
                                                    uniforms,
                                                    material.textureIndices0.w,
                                                    occlusionCtx.uv,
                                                    float4(1.0f),
                                                    occlusionLod,
                                                    occlusionCtx.hasIgehyGradients,
                                                    occlusionCtx.dUVdx,
                                                    occlusionCtx.dUVdy).x;
                occlusion = mix(1.0f, occSample, clamp(material.pbrParams.z, 0.0f, 1.0f));
            }
            debugAO = occlusion;
            diffuseOcclusion = (uniforms.debugDisableAO != 0u) ? 1.0f : occlusion;
            if (uniforms.debugAoIndirectOnly != 0u && depth == 0u) {
                diffuseOcclusion = 1.0f;
            }
            debugBaseColor = baseColor;
            debugMetallic = metallic;
            debugRoughness = roughness;

            float3 emissive = to_working_space(material.emission.xyz, uniforms);
            if (material_texture_valid(uniforms, material.textureIndices1.x)) {
                float emissiveLod =
                    material_texture_lod_with_fallback(materialTextures,
                                                       materialTextureInfos,
                                                       uniforms,
                                                       material.textureIndices1.x,
                                                       emissiveCtx.hasIgehyGradients,
                                                       emissiveCtx.dUVdx,
                                                       emissiveCtx.dUVdy,
                                                       emissiveCtx.uvPerWorld,
                                                       surfaceFootprintWorld);
                float3 emissiveSample =
                    sample_material_texture_filtered(materialTextures,
                                                    materialSamplers,
                                                    materialTextureInfos,
                                                    uniforms,
                                                    material.textureIndices1.x,
                                                    emissiveCtx.uv,
                                                    float4(1.0f),
                                                    emissiveLod,
                                                    emissiveCtx.hasIgehyGradients,
                                                    emissiveCtx.dUVdx,
                                                    emissiveCtx.dUVdy).xyz;
                emissiveSample = to_working_space(emissiveSample, uniforms);
                emissive *= emissiveSample;
            }

            bool useNormalMap = material_texture_valid(uniforms, material.textureIndices0.z);
#if PT_DEBUG_TOOLS
            if (uniforms.debugDisableNormalMap != 0u) {
                useNormalMap = false;
            }
#endif
            if (normalScale <= 1.0e-4f) {
                useNormalMap = false;
            }
            float normalLength = 1.0f;
            if (useNormalMap) {
                float3 normalSample =
                    sample_material_texture_filtered(materialTextures,
                                                    materialSamplers,
                                                    materialTextureInfos,
                                                    uniforms,
                                                    material.textureIndices0.z,
                                                    normalCtx.uv,
                                                    float4(0.5f, 0.5f, 1.0f, 1.0f),
                                                    normalLod,
                                                    normalCtx.hasIgehyGradients,
                                                    normalCtx.dUVdx,
                                                    normalCtx.dUVdy).xyz;
                bool flipNormalGreen = false;
#if PT_DEBUG_TOOLS
                flipNormalGreen = uniforms.debugFlipNormalGreen != 0u;
#endif
                normalSample = decode_normal_map(normalSample,
                                                 normalScale,
                                                 flipNormalGreen,
                                                 normalLength);
                float3 t = tangent.xyz;
                float3 b = float3(0.0f);
                bool hasBasis = false;
                bool trustVertexTangent = fabs(tangent.w) > 0.5f;
                if (trustVertexTangent && all(isfinite(t)) && dot(t, t) > 1.0e-6f) {
                    t = normalize(t - shadingNormal * dot(shadingNormal, t));
                    if (all(isfinite(t)) && dot(t, t) > 1.0e-6f) {
                        float tangentSign = (tangent.w < 0.0f) ? -1.0f : 1.0f;
                        b = normalize(cross(shadingNormal, t)) * tangentSign;
                        if (all(isfinite(b)) && dot(b, b) > 1.0e-6f) {
                            hasBasis = true;
                        }
                    }
                }
                if (!hasBasis) {
                    uint normalUvSet = pbr_texture_uv_set(material, kPbrTextureSlotNormal);
                    hasBasis = compute_tangent_basis_from_uv(uniforms,
                                                             rec.meshIndex,
                                                             rec.primitiveIndex,
                                                             normalUvSet,
                                                             meshInfos,
                                                             sceneVertices,
                                                             meshIndices,
                                                             shadingNormal,
                                                             t,
                                                             b);
                }
                if (!hasBasis) {
                    build_onb(shadingNormal, t, b);
                }
                float3 mapped = normalize(t * normalSample.x +
                                          b * normalSample.y +
                                          shadingNormal * normalSample.z);
                if (dot(mapped, rec.normal) < 0.0f) {
                    mapped = -mapped;
                }
                shadingNormal = mapped;
            }

            if (useNormalMap) {
                float tok = max((1.0f - normalLength) / max(normalLength, 1.0e-6f), 0.0f);
                roughness = clamp(sqrt(roughness * roughness + tok), 0.0f, 1.0f);
            }

            material.baseColorRoughness = float4(baseColor, roughness);
            material.pbrParams.x = metallic;
            material.emission = float4(emissive, 0.0f);
            rec.shadingNormal = shadingNormal;
        }

        if (uniforms.debugViewMode != kDebugViewNone) {
            float3 debugColor = float3(0.0f);
            switch (uniforms.debugViewMode) {
                case kDebugViewBaseColor:
                    debugColor = debugBaseColor;
                    break;
                case kDebugViewMetallic:
                    debugColor = float3(debugMetallic);
                    break;
                case kDebugViewRoughness:
                    debugColor = float3(debugRoughness);
                    break;
                case kDebugViewAO:
                    debugColor = float3(debugAO);
                    break;
                default:
                    break;
            }
            radiance = debugColor;
            break;
        }

        // Capture first hit AOVs (albedo and normal) for denoising
        if (isFirstHit) {
            isFirstHit = false;
            if (outFirstHitAlbedo != nullptr) {
                // Albedo is the diffuse color from the first hit
                *outFirstHitAlbedo = material_base_color(material);
            }
            if (outFirstHitNormal != nullptr) {
                // Store the world-space normal of first hit
                *outFirstHitNormal = shadingNormal;
            }
        }

        if (!specularOnly &&
            type == 7u &&
            any(material.emission.xyz != float3(0.0f)) &&
            (rec.frontFace != 0u || rec.twoSided != 0u)) {
            radiance += clamp_firefly_contribution(throughput, material.emission.xyz, clampParams);
        }

        if (type == 3u) {  // DiffuseLight
            if (specularOnly) {
                break;
            }
            float3 emission = material.emission.xyz;
            if (material.emission.w > 0.0f &&
                environmentTexture.get_width() > 0 &&
                environmentTexture.get_height() > 0 &&
                rec.frontFace != 0u) {
                float3 sampleDir = -shadingNormal;
                float3 envColor = environment_color(environmentTexture,
                                                    sampleDir,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity,
                                                    uniforms);
                emission *= envColor;
            }
            if (any(emission != float3(0.0f)) &&
                (rec.frontFace != 0u || rec.twoSided != 0u)) {
                float misWeight = 1.0f;
                bool useSpecularMis = (!lastScatterWasDelta) ||
                                      (uniforms.enableSpecularNee != 0u) ||
                                      ((ENABLE_MNEE_CAUSTICS != 0) && (uniforms.enableMnee != 0u));
                if (useSpecularMis && rectLightCount > 0u) {
                    float lightPdf = rect_light_pdf_for_hit(uniforms,
                                                            rectangles,
                                                            materials,
                                                            rectLightCount,
                                                            rec,
                                                            ray.origin);
                    float denom = lastBsdfPdf + lightPdf;
                    if (denom > 0.0f) {
                        misWeight = clamp(lastBsdfPdf / denom,
                                          kMisWeightClampMin,
                                          kMisWeightClampMax);
                    }
                }
                float3 contribution = emission * misWeight;
                radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
            }
            break;
        }

        if (!surfaceIsDelta && rectLightCount > 0u) {
            RectLightSample lightSample;
            if (sample_rect_light(uniforms,
                                  rectangles,
                                  materials,
                                  environmentTexture,
                                  rec,
                                  state,
                                  rectLightCount,
                                  lightSample)) {
                float nDotL = max(dot(shadingNormal, lightSample.direction), 0.0f);
                if (lightSample.pdf > 0.0f && nDotL > 0.0f) {
                    Ray shadowRay;
                    shadowRay.origin = offset_ray_origin(rec, lightSample.direction);
                    shadowRay.direction = lightSample.direction;
                    HitRecord shadowRec;
                    float shadowMax = max(lightSample.distance - kEpsilon, kEpsilon);
                    uint shadowExcludeMesh;
                    uint shadowExcludePrim;
                    compute_exclusion_indices(rec, shadowExcludeMesh, shadowExcludePrim);
                    bool occluded = false;
#if PT_DEBUG_TOOLS
                    if (forceSoftware) {
                        occluded = trace_scene_software(uniforms,
                                                        spheres,
                                                        rectangles,
                                                        triangleData,
                                                        tlasNodes,
                                                        tlasPrimIndices,
                                                        instanceInfos,
                                                        blasNodes,
                                                        blasPrimIndices,
                                                        nodes,
                                                        primitiveIndices,
                                                        stats,
                                                        shadowRay,
                                                        kEpsilon,
                                                        shadowMax,
                                                        /*anyHitOnly=*/true,
                                                        /*includeTriangles=*/true,
                                                        shadowRec);
                    } else {
#endif
                        occluded = trace_scene_hardware(uniforms,
                                                        accel,
                                                        meshInfos,
                                                        triangleData,
                                                        sceneVertices,
                                                        meshIndices,
                                                        instanceUserIds,
                                                        spheres,
                                                        rectangles,
                                                        nodes,
                                                        primitiveIndices,
                                                        stats,
                                                        shadowRay,
                                                        kHardwareOcclusionEpsilon,
                                                        shadowMax,
                                                        /*anyHitOnly=*/true,
                                                        shadowExcludeMesh,
                                                        shadowExcludePrim,
                                                        shadowRec);
#if PT_DEBUG_TOOLS
                    }
#endif
                    if (!occluded) {
                        BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                                rec.point,
                                                                shadingNormal,
                                                                wo,
                                                                lightSample.direction,
                                                                clampParams,
                                                                uniforms.sssMode,
                                                                diffuseOcclusion,
                                                                specularOnly);
                        if (!bsdfEval.isDelta && !bsdfEval.isBssrdf) {
                            float3 bsdfValue = bsdfEval.value;
                            float maxComponent = max(max(bsdfValue.x, bsdfValue.y), bsdfValue.z);
                            if (maxComponent > 0.0f && lightSample.pdf > 0.0f) {
                                float bsdfPdf = bsdfEval.pdf;
                                float weight = 1.0f;
                                if (bsdfPdf > 0.0f) {
                                    float denom = lightSample.pdf + bsdfPdf;
                                    if (denom > 0.0f) {
                                        weight = clamp(lightSample.pdf / denom,
                                                       kMisWeightClampMin,
                                                       kMisWeightClampMax);
                                    }
                                }
                                float3 contribution = lightSample.emission * bsdfValue * nDotL;
                                contribution *= weight / lightSample.pdf;
                                if (all(isfinite(contribution))) {
                                    radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                                }
                            }
                        }
                    }
                }
            }
        }
        if (!surfaceIsDelta && envSampling) {
            EnvironmentSample envSample;
            if (sample_environment(uniforms,
                                   environmentTexture,
                                   environmentConditionalAlias,
                                   environmentMarginalAlias,
                                   environmentPdf,
                                   state,
                                   envSample)) {
                float overrideLod = 0.0f;
                bool useOverride = environment_mip_override(uniforms, environmentTexture, overrideLod);
                if (environmentTexture.get_num_mip_levels() > 1u) {
                    float envRoughness = environment_lighting_roughness(material);
                    if (envRoughness < 0.95f) {
                        float envLod = environment_lod_from_roughness(envRoughness,
                                                                      environmentTexture);
                        envSample.radiance = environment_color_lod(environmentTexture,
                                                                   envSample.direction,
                                                                   uniforms.environmentRotation,
                                                                   uniforms.environmentIntensity,
                                                                   envLod,
                                                                   uniforms);
                    }
                }
                if (useOverride) {
                    envSample.radiance = environment_color_lod(environmentTexture,
                                                               envSample.direction,
                                                               uniforms.environmentRotation,
                                                               uniforms.environmentIntensity,
                                                               overrideLod,
                                                               uniforms);
                }
                float nDotL = max(dot(shadingNormal, envSample.direction), 0.0f);
                if (envSample.pdf > 0.0f && nDotL > 0.0f) {
                    Ray shadowRay;
                    shadowRay.origin = offset_ray_origin(rec, envSample.direction);
                    shadowRay.direction = envSample.direction;
                    HitRecord shadowRec;
                    uint shadowExcludeMesh;
                    uint shadowExcludePrim;
                    compute_exclusion_indices(rec, shadowExcludeMesh, shadowExcludePrim);
                    bool occluded = false;
#if PT_DEBUG_TOOLS
                    if (forceSoftware) {
                        occluded = trace_scene_software(uniforms,
                                                        spheres,
                                                        rectangles,
                                                        triangleData,
                                                        tlasNodes,
                                                        tlasPrimIndices,
                                                        instanceInfos,
                                                        blasNodes,
                                                        blasPrimIndices,
                                                        nodes,
                                                        primitiveIndices,
                                                        stats,
                                                        shadowRay,
                                                        kEpsilon,
                                                        kInfinity,
                                                        /*anyHitOnly=*/true,
                                                        /*includeTriangles=*/true,
                                                        shadowRec);
                    } else {
#endif
                        occluded = trace_scene_hardware(uniforms,
                                                        accel,
                                                        meshInfos,
                                                        triangleData,
                                                        sceneVertices,
                                                        meshIndices,
                                                        instanceUserIds,
                                                        spheres,
                                                        rectangles,
                                                        nodes,
                                                        primitiveIndices,
                                                        stats,
                                                        shadowRay,
                                                        kHardwareOcclusionEpsilon,
                                                        kInfinity,
                                                        /*anyHitOnly=*/true,
                                                        shadowExcludeMesh,
                                                        shadowExcludePrim,
                                                        shadowRec);
#if PT_DEBUG_TOOLS
                    }
#endif
                    if (!occluded) {
                        BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                                rec.point,
                                                                shadingNormal,
                                                                wo,
                                                                envSample.direction,
                                                                clampParams,
                                                                uniforms.sssMode,
                                                                diffuseOcclusion,
                                                                specularOnly);
                        if (!bsdfEval.isDelta && !bsdfEval.isBssrdf) {
                            float3 bsdfValue = bsdfEval.value;
                            float maxComponent = max(max(bsdfValue.x, bsdfValue.y), bsdfValue.z);
                            if (maxComponent > 0.0f) {
                                float bsdfPdf = bsdfEval.pdf;
                                float weight = 1.0f;
                                if (bsdfPdf > 0.0f) {
                                    float denom = envSample.pdf + bsdfPdf;
                                    if (denom > 0.0f) {
                                        weight = clamp(envSample.pdf / denom,
                                                       kMisWeightClampMin,
                                                       kMisWeightClampMax);
                                    }
                                }
                                float3 contribution = envSample.radiance * bsdfValue * nDotL;
                                contribution *= weight / envSample.pdf;
                                if (all(isfinite(contribution))) {
                                    radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                                }
                            }
                        }
                    }
                }
            }
        }

        BsdfSampleResult bsdfSample;
        bool usedRandomWalk = false;
        bool enableRandomWalk = material_is_subsurface(material) &&
                                uniforms.sssMode == 2u &&
                                material.sssParams.y >= 0.5f &&
                                rec.frontFace != 0u;
        if (enableRandomWalk) {
#if PT_DEBUG_TOOLS
            if (forceSoftware) {
                bsdfSample = sample_sss_random_walk_software(uniforms,
                                                             material,
                                                             rec,
                                                             wo,
                                                             incidentDir,
                                                             spheres,
                                                             rectangles,
                                                             triangleData,
                                                             tlasNodes,
                                                             tlasPrimIndices,
                                                             instanceInfos,
                                                             blasNodes,
                                                             blasPrimIndices,
                                                             nodes,
                                                             primitiveIndices,
                                                             stats,
                                                             state,
                                                             clampParams);
            } else {
#endif
                bsdfSample = sample_sss_random_walk_hardware(uniforms,
                                                             material,
                                                             rec,
                                                             wo,
                                                             incidentDir,
                                                             accel,
                                                             meshInfos,
                                                             triangleData,
                                                             sceneVertices,
                                                             meshIndices,
                                                             instanceUserIds,
                                                             spheres,
                                                             rectangles,
                                                             nodes,
                                                             primitiveIndices,
                                                             stats,
                                                             state,
                                                             clampParams);
#if PT_DEBUG_TOOLS
            }
#endif
            usedRandomWalk = (bsdfSample.pdf > 0.0f);
        }
        if (!usedRandomWalk) {
            bsdfSample = sample_bsdf(material,
                                     rec.point,
                                     shadingNormal,
                                     wo,
                                     incidentDir,
                                     rec.frontFace != 0u,
                                     state,
                                     clampParams,
                                     uniforms.sssMode,
                                     diffuseOcclusion,
                                     specularOnly);
        }
        if (bsdfSample.pdf <= 0.0f) {
            break;
        }

        if (bsdfSample.mediumEvent == 1) {
            float3 sigma = dielectric_sigma_a(material);
            sigma = max(sigma, float3(0.0f));
            if (mediumDepth < kMaxMediumStack) {
                mediumSigmaStack[mediumDepth] = sigma;
                mediumDepth += 1u;
            } else {
                mediumSigmaStack[kMaxMediumStack - 1u] = sigma;
            }
        } else if (bsdfSample.mediumEvent == -1) {
            if (mediumDepth > 0u) {
                mediumDepth -= 1u;
            }
        }

        bool causticCandidate = (!surfaceIsDelta) && (specularDepth > 0u);
        uint nextSpecularDepth = bsdfSample.isDelta ? (specularDepth + 1u) : 0u;
        bool didTransmission = false;
        if (bsdfSample.isDelta && type == 2u) {
            float3 dir = bsdfSample.direction;
            if (all(isfinite(dir)) && dot(dir, dir) > 0.0f) {
                float side = (rec.frontFace != 0u) ? 1.0f : -1.0f;
                didTransmission = (dot(shadingNormal, dir) * side) < 0.0f;
            }
        }
        if (didTransmission) {
            hadTransmission = true;
        }
        specularDepth = nextSpecularDepth;
        (void)causticCandidate;

        float3 nextOrigin;
        if (bsdfSample.hasExitPoint) {
            float3 exitNormal = bsdfSample.exitNormal;
            bool normalValid = all(isfinite(exitNormal)) && dot(exitNormal, exitNormal) > 0.0f;
            if (!normalValid) {
                exitNormal = rec.normal;
                normalValid = all(isfinite(exitNormal)) && dot(exitNormal, exitNormal) > 0.0f;
            }
            if (!normalValid) {
                exitNormal = float3(0.0f, 1.0f, 0.0f);
            }
            exitNormal = normalize(exitNormal);
            nextOrigin = offset_surface_point(bsdfSample.exitPoint, exitNormal, bsdfSample.direction);
            // Match software/HWRT parity: bias exit points to avoid self-occlusion in HWRT.
            float normalBias = max(kHardwareOcclusionEpsilon * 4.0f, kRayOriginEpsilon * 32.0f);
            float directionalBias = max(kHardwareOcclusionEpsilon * 8.0f, kRayOriginEpsilon * 32.0f);
            if (uniforms.hardwareExitNormalBias > 0.0f) {
                normalBias = max(normalBias, uniforms.hardwareExitNormalBias);
            }
            if (uniforms.hardwareExitDirectionalBias > 0.0f) {
                directionalBias = max(directionalBias, uniforms.hardwareExitDirectionalBias);
            }
            nextOrigin += exitNormal * normalBias;
            float3 dir = bsdfSample.direction;
            if (!all(isfinite(dir)) || dot(dir, dir) <= 0.0f) {
                dir = exitNormal;
            } else {
                dir = normalize(dir);
            }
            nextOrigin += dir * directionalBias;
        } else {
            nextOrigin = offset_ray_origin(rec, bsdfSample.direction);
        }

        bool useMnee = (ENABLE_MNEE_CAUSTICS != 0) && (uniforms.enableMnee != 0u);
        bool specNeeEnabled = (uniforms.enableSpecularNee != 0u);
        float dirLenSq = dot(bsdfSample.direction, bsdfSample.direction);
        bool specDirectionValid = (dirLenSq > 0.0f) && all(isfinite(bsdfSample.direction));
        bool mneeEligible = false;
#if ENABLE_MNEE_CAUSTICS
        mneeEligible = useMnee &&
                       bsdfSample.isDelta &&
                       ((bsdfSample.mediumEvent <= 0) || didTransmission) &&
                       (type == 2u) &&
                       (nextSpecularDepth == 1u) &&
                       specDirectionValid;
#endif
        if (mneeEligible) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEligibleCount, 1u, memory_order_relaxed);
            }
        }
        bool specNeeEligible = specNeeEnabled &&
                               bsdfSample.isDelta &&
                               (bsdfSample.mediumEvent <= 0) &&
                               specDirectionValid &&
                               !mneeEligible;

        if (specNeeEligible && envSampling &&
            environmentTexture.get_width() > 0 &&
            environmentTexture.get_height() > 0) {
            Ray neeRay;
            neeRay.origin = nextOrigin;
            neeRay.direction = normalize(bsdfSample.direction);
            HitRecord shadowRec;
            uint neeExcludeMesh = kInvalidIndex;
            uint neeExcludePrim = kInvalidIndex;
#if !PT_MNEE_OCCLUSION_PARITY
            if (didTransmission) {
                compute_exclusion_indices(rec, neeExcludeMesh, neeExcludePrim);
                neeExcludePrim = kInvalidIndex;
            }
#endif
            bool occluded = false;
#if PT_DEBUG_TOOLS
            if (forceSoftware) {
                occluded = trace_scene_software(uniforms,
                                                spheres,
                                                rectangles,
                                                triangleData,
                                                tlasNodes,
                                                tlasPrimIndices,
                                                instanceInfos,
                                                blasNodes,
                                                blasPrimIndices,
                                                nodes,
                                                primitiveIndices,
                                                stats,
                                                neeRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/true,
                                                /*includeTriangles=*/true,
                                                shadowRec);
            } else {
#endif
                occluded = trace_scene_hardware(uniforms,
                                                accel,
                                                meshInfos,
                                                triangleData,
                                                sceneVertices,
                                                meshIndices,
                                                instanceUserIds,
                                                spheres,
                                                rectangles,
                                                nodes,
                                                primitiveIndices,
                                                stats,
                                                neeRay,
                                                kHardwareOcclusionEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/true,
                                                neeExcludeMesh,
                                                neeExcludePrim,
                                                shadowRec);
#if PT_DEBUG_TOOLS
            }
#endif
            if (!occluded) {
                float envPdf = environment_pdf(uniforms, environmentPdf, neeRay.direction);
                envPdf = max(envPdf, kSpecularNeePdfFloor);
                float invEnvPdf = min(1.0f / envPdf, kSpecularNeeInvPdfClamp);
                float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                float denom = envPdf + bsdfPdf;
                float misWeight = (denom > 0.0f) ? (envPdf / denom) : 0.0f;
                misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                float3 envColor = environment_color(environmentTexture,
                                                    neeRay.direction,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity,
                                                    uniforms);
                float3 neeContribution = bsdfSample.weight * envColor * (misWeight * invEnvPdf);
                if (all(isfinite(neeContribution))) {
                    radiance += clamp_firefly_contribution(throughput, neeContribution, clampParams);
                    if (stats) {
                        atomic_fetch_add_explicit(&stats->specNeeEnvAddedCount, 1u, memory_order_relaxed);
                    }
                }
            }
        }

        if (specNeeEligible && rectLightCount > 0u) {
            Ray neeRay;
            neeRay.origin = nextOrigin;
            neeRay.direction = normalize(bsdfSample.direction);
            HitRecord lightRec;
            uint neeExcludeMesh = kInvalidIndex;
            uint neeExcludePrim = kInvalidIndex;
            if (didTransmission) {
                compute_exclusion_indices(rec, neeExcludeMesh, neeExcludePrim);
                neeExcludePrim = kInvalidIndex;
            }
            bool hitLight = false;
#if PT_DEBUG_TOOLS
            if (forceSoftware) {
                hitLight = trace_scene_software(uniforms,
                                                spheres,
                                                rectangles,
                                                triangleData,
                                                tlasNodes,
                                                tlasPrimIndices,
                                                instanceInfos,
                                                blasNodes,
                                                blasPrimIndices,
                                                nodes,
                                                primitiveIndices,
                                                stats,
                                                neeRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/false,
                                                /*includeTriangles=*/true,
                                                lightRec);
            } else {
#endif
                hitLight = trace_scene_hardware(uniforms,
                                                accel,
                                                meshInfos,
                                                triangleData,
                                                sceneVertices,
                                                meshIndices,
                                                instanceUserIds,
                                                spheres,
                                                rectangles,
                                                nodes,
                                                primitiveIndices,
                                                stats,
                                                neeRay,
                                                kHardwareOcclusionEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/false,
                                                neeExcludeMesh,
                                                neeExcludePrim,
                                                lightRec);
#if PT_DEBUG_TOOLS
            }
#endif
            if (hitLight) {
                MneeRectHit mneeHit;
                if (mnee_rect_light_hit(uniforms,
                                        rectangles,
                                        materials,
                                        environmentTexture,
                                        rectLightCount,
                                        lightRec,
                                        nextOrigin,
                                        mneeHit)) {
                    float lightPdf = max(mneeHit.pdf, kSpecularNeePdfFloor);
                    float invLightPdf = min(1.0f / lightPdf, kSpecularNeeInvPdfClamp);
                    float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                    float denom = lightPdf + bsdfPdf;
                    float misWeight = (denom > 0.0f) ? (lightPdf / denom) : 0.0f;
                    misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                    float3 contribution = bsdfSample.weight * mneeHit.emission *
                                          (misWeight * invLightPdf);
                    if (all(isfinite(contribution))) {
                        radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                        if (stats) {
                            atomic_fetch_add_explicit(&stats->specNeeRectAddedCount, 1u, memory_order_relaxed);
                        }
                    }
                }
            }
        }

#if ENABLE_MNEE_CAUSTICS
        if (mneeEligible && envSampling &&
            environmentTexture.get_width() > 0 &&
            environmentTexture.get_height() > 0) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEnvAttemptCount, 1u, memory_order_relaxed);
            }
            Ray mneeRay;
            mneeRay.origin = nextOrigin;
            mneeRay.direction = normalize(bsdfSample.direction);
            HitRecord shadowRec;
            uint neeExcludeMesh = kInvalidIndex;
            uint neeExcludePrim = kInvalidIndex;
#if !PT_MNEE_OCCLUSION_PARITY
            if (didTransmission) {
                compute_exclusion_indices(rec, neeExcludeMesh, neeExcludePrim);
                neeExcludePrim = kInvalidIndex;
            }
#endif
            bool occluded = false;
#if PT_MNEE_SWRT_RAYS
            occluded = trace_scene_software(uniforms,
                                            spheres,
                                            rectangles,
                                            triangleData,
                                            tlasNodes,
                                            tlasPrimIndices,
                                            instanceInfos,
                                            blasNodes,
                                            blasPrimIndices,
                                            nodes,
                                            primitiveIndices,
                                            stats,
                                            mneeRay,
                                            kEpsilon,
                                            kInfinity,
                                            /*anyHitOnly=*/true,
                                            /*includeTriangles=*/true,
                                            shadowRec);
#else
#if PT_DEBUG_TOOLS
            if (forceSoftware) {
                occluded = trace_scene_software(uniforms,
                                                spheres,
                                                rectangles,
                                                triangleData,
                                                tlasNodes,
                                                tlasPrimIndices,
                                                instanceInfos,
                                                blasNodes,
                                                blasPrimIndices,
                                                nodes,
                                                primitiveIndices,
                                                stats,
                                                mneeRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/true,
                                                /*includeTriangles=*/true,
                                                shadowRec);
            } else {
#endif
#if PT_MNEE_OCCLUSION_PARITY
                bool occludedHw = trace_scene_hardware(uniforms,
                                                       accel,
                                                       meshInfos,
                                                       triangleData,
                                                       sceneVertices,
                                                       meshIndices,
                                                       instanceUserIds,
                                                       spheres,
                                                       rectangles,
                                                       nodes,
                                                       primitiveIndices,
                                                       stats,
                                                       mneeRay,
                                                       kEpsilon,
                                                       kInfinity,
                                                       /*anyHitOnly=*/true,
                                                       neeExcludeMesh,
                                                       neeExcludePrim,
                                                       shadowRec);
                HitRecord shadowRecSw;
                bool occludedSw = trace_scene_software(uniforms,
                                                       spheres,
                                                       rectangles,
                                                       triangleData,
                                                       tlasNodes,
                                                       tlasPrimIndices,
                                                       instanceInfos,
                                                       blasNodes,
                                                       blasPrimIndices,
                                                       nodes,
                                                       primitiveIndices,
                                                       stats,
                                                       mneeRay,
                                                       kEpsilon,
                                                       kInfinity,
                                                       /*anyHitOnly=*/true,
                                                       /*includeTriangles=*/true,
                                                       shadowRecSw);
                if (stats) {
                    if (occludedHw) {
                        atomic_fetch_add_explicit(&stats->mneeEnvHwOccludedCount, 1u, memory_order_relaxed);
                    }
                    if (occludedSw) {
                        atomic_fetch_add_explicit(&stats->mneeEnvSwOccludedCount, 1u, memory_order_relaxed);
                    }
                    if (occludedHw != occludedSw) {
                        atomic_fetch_add_explicit(&stats->mneeEnvHwSwMismatchCount, 1u, memory_order_relaxed);
                    }
                }
                occluded = occludedHw;
#else
                occluded = trace_scene_hardware(uniforms,
                                                accel,
                                                meshInfos,
                                                triangleData,
                                                sceneVertices,
                                                meshIndices,
                                                instanceUserIds,
                                                spheres,
                                                rectangles,
                                                nodes,
                                                primitiveIndices,
                                                stats,
                                                mneeRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/true,
                                                neeExcludeMesh,
                                                neeExcludePrim,
                                                shadowRec);
#endif
#if PT_DEBUG_TOOLS
            }
#endif
#endif
            if (!occluded) {
                float envPdf = environment_pdf(uniforms, environmentPdf, mneeRay.direction);
                envPdf = max(envPdf, kSpecularNeePdfFloor);
                float invEnvPdf = min(1.0f / envPdf, kSpecularNeeInvPdfClamp);
                float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                float denom = envPdf + bsdfPdf;
                float misWeight = (denom > 0.0f) ? (envPdf / denom) : 0.0f;
                misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                float3 envColor = environment_color(environmentTexture,
                                                    mneeRay.direction,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity,
                                                    uniforms);
                float3 neeContribution = bsdfSample.weight * envColor * (misWeight * invEnvPdf);
                if (all(isfinite(neeContribution))) {
                    radiance += clamp_firefly_contribution(throughput, neeContribution, clampParams);
                    if (stats) {
                        atomic_fetch_add_explicit(&stats->mneeEnvAddedCount, 1u, memory_order_relaxed);
                        stats_add_mnee_luma(stats, neeContribution);
                    }
                }
            }
        }

        if (mneeEligible && rectLightCount > 0u) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeRectAttemptCount, 1u, memory_order_relaxed);
            }
            Ray mneeRay;
            mneeRay.origin = nextOrigin;
            mneeRay.direction = normalize(bsdfSample.direction);
            HitRecord lightRec;
            bool hitLight = false;
            MneeRectHit mneeHit;
            bool mneeLight = false;
#if PT_MNEE_OCCLUSION_PARITY
            HitRecord lightRecSw;
            MneeRectHit mneeHitSw;
#endif
            uint neeExcludeMesh = kInvalidIndex;
            uint neeExcludePrim = kInvalidIndex;
            if (didTransmission) {
                compute_exclusion_indices(rec, neeExcludeMesh, neeExcludePrim);
                neeExcludePrim = kInvalidIndex;
            }
#if PT_MNEE_SWRT_RAYS
            hitLight = trace_scene_software(uniforms,
                                            spheres,
                                            rectangles,
                                            triangleData,
                                            tlasNodes,
                                            tlasPrimIndices,
                                            instanceInfos,
                                            blasNodes,
                                            blasPrimIndices,
                                            nodes,
                                            primitiveIndices,
                                            stats,
                                            mneeRay,
                                            kEpsilon,
                                            kInfinity,
                                            /*anyHitOnly=*/false,
                                            /*includeTriangles=*/true,
                                            lightRec);
#else
#if PT_DEBUG_TOOLS
            if (forceSoftware) {
                hitLight = trace_scene_software(uniforms,
                                                spheres,
                                                rectangles,
                                                triangleData,
                                                tlasNodes,
                                                tlasPrimIndices,
                                                instanceInfos,
                                                blasNodes,
                                                blasPrimIndices,
                                                nodes,
                                                primitiveIndices,
                                                stats,
                                                mneeRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/false,
                                                /*includeTriangles=*/true,
                                                lightRec);
            } else {
#endif
#if PT_MNEE_OCCLUSION_PARITY
                bool hitLightHw = trace_scene_hardware(uniforms,
                                                       accel,
                                                       meshInfos,
                                                       triangleData,
                                                       sceneVertices,
                                                       meshIndices,
                                                       instanceUserIds,
                                                       spheres,
                                                       rectangles,
                                                       nodes,
                                                       primitiveIndices,
                                                       stats,
                                                       mneeRay,
                                                       kEpsilon,
                                                       kInfinity,
                                                       /*anyHitOnly=*/false,
                                                       neeExcludeMesh,
                                                       neeExcludePrim,
                                                       lightRec);
                bool hitLightSw = trace_scene_software(uniforms,
                                                       spheres,
                                                       rectangles,
                                                       triangleData,
                                                       tlasNodes,
                                                       tlasPrimIndices,
                                                       instanceInfos,
                                                       blasNodes,
                                                       blasPrimIndices,
                                                       nodes,
                                                       primitiveIndices,
                                                       stats,
                                                       mneeRay,
                                                       kEpsilon,
                                                       kInfinity,
                                                       /*anyHitOnly=*/false,
                                                       /*includeTriangles=*/true,
                                                       lightRecSw);
                bool hwMneeLight = false;
                bool swMneeLight = false;
                if (hitLightHw) {
                    hwMneeLight = mnee_rect_light_hit(uniforms,
                                                      rectangles,
                                                      materials,
                                                      environmentTexture,
                                                      rectLightCount,
                                                      lightRec,
                                                      nextOrigin,
                                                      mneeHit);
                }
                if (hitLightSw) {
                    swMneeLight = mnee_rect_light_hit(uniforms,
                                                      rectangles,
                                                      materials,
                                                      environmentTexture,
                                                      rectLightCount,
                                                      lightRecSw,
                                                      nextOrigin,
                                                      mneeHitSw);
                }
                if (stats) {
                    if (hwMneeLight) {
                        atomic_fetch_add_explicit(&stats->mneeRectHwOccludedCount, 1u, memory_order_relaxed);
                    }
                    if (swMneeLight) {
                        atomic_fetch_add_explicit(&stats->mneeRectSwOccludedCount, 1u, memory_order_relaxed);
                    }
                    if (hwMneeLight != swMneeLight) {
                        atomic_fetch_add_explicit(&stats->mneeRectHwSwMismatchCount, 1u, memory_order_relaxed);
                    }
                }
                hitLight = hitLightHw;
                mneeLight = hwMneeLight;
#else
                hitLight = trace_scene_hardware(uniforms,
                                                accel,
                                                meshInfos,
                                                triangleData,
                                                sceneVertices,
                                                meshIndices,
                                                instanceUserIds,
                                                spheres,
                                                rectangles,
                                                nodes,
                                                primitiveIndices,
                                                stats,
                                                mneeRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/false,
                                                neeExcludeMesh,
                                                neeExcludePrim,
                                                lightRec);
#endif
#if PT_DEBUG_TOOLS
            }
#endif
#endif
            if (!mneeLight && hitLight) {
                mneeLight = mnee_rect_light_hit(uniforms,
                                                rectangles,
                                                materials,
                                                environmentTexture,
                                                rectLightCount,
                                                lightRec,
                                                nextOrigin,
                                                mneeHit);
            }
            if (mneeLight) {
                float lightPdf = max(mneeHit.pdf, kSpecularNeePdfFloor);
                float invLightPdf = min(1.0f / lightPdf, kSpecularNeeInvPdfClamp);
                float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                float denom = lightPdf + bsdfPdf;
                float misWeight = (denom > 0.0f) ? (lightPdf / denom) : 0.0f;
                misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                float3 contribution = bsdfSample.weight * mneeHit.emission *
                                      (misWeight * invLightPdf);
                if (all(isfinite(contribution))) {
                    radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                    if (stats) {
                        atomic_fetch_add_explicit(&stats->mneeRectAddedCount, 1u, memory_order_relaxed);
                        stats_add_mnee_luma(stats, contribution);
                    }
                }
            }
        }

        if (mneeEligible && uniforms.enableMneeSecondary != 0u) {
            Ray chainRay;
            chainRay.origin = nextOrigin;
            chainRay.direction = normalize(bsdfSample.direction);
            HitRecord chainRec;
            uint chainExcludeMesh = kInvalidIndex;
            uint chainExcludePrim = kInvalidIndex;
            bool chainHit = false;
#if PT_MNEE_SWRT_RAYS
            chainHit = trace_scene_software(uniforms,
                                            spheres,
                                            rectangles,
                                            triangleData,
                                            tlasNodes,
                                            tlasPrimIndices,
                                            instanceInfos,
                                            blasNodes,
                                            blasPrimIndices,
                                            nodes,
                                            primitiveIndices,
                                            stats,
                                            chainRay,
                                            kEpsilon,
                                            kInfinity,
                                            /*anyHitOnly=*/false,
                                            /*includeTriangles=*/true,
                                            chainRec);
#else
#if PT_DEBUG_TOOLS
            if (forceSoftware) {
                chainHit = trace_scene_software(uniforms,
                                                spheres,
                                                rectangles,
                                                triangleData,
                                                tlasNodes,
                                                tlasPrimIndices,
                                                instanceInfos,
                                                blasNodes,
                                                blasPrimIndices,
                                                nodes,
                                                primitiveIndices,
                                                stats,
                                                chainRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/false,
                                                /*includeTriangles=*/true,
                                                chainRec);
            } else {
#endif
                chainHit = trace_scene_hardware(uniforms,
                                                accel,
                                                meshInfos,
                                                triangleData,
                                                sceneVertices,
                                                meshIndices,
                                                instanceUserIds,
                                                spheres,
                                                rectangles,
                                                nodes,
                                                primitiveIndices,
                                                stats,
                                                chainRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/false,
                                                chainExcludeMesh,
                                                chainExcludePrim,
                                                chainRec);
#if PT_DEBUG_TOOLS
            }
#endif
#endif
            if (chainHit && materials && uniforms.materialCount > 0u) {
                bool chainHitIsLight = false;
                if (rectLightCount > 0u) {
                    MneeRectHit chainLightHit;
                    if (mnee_rect_light_hit(uniforms,
                                            rectangles,
                                            materials,
                                            environmentTexture,
                                            rectLightCount,
                                            chainRec,
                                            chainRay.origin,
                                            chainLightHit)) {
                        chainHitIsLight = true;
                    }
                }
                if (!chainHitIsLight) {
                    uint chainMatIndex = min(chainRec.materialIndex, uniforms.materialCount - 1u);
                    MaterialData chainMaterial = materials[chainMatIndex];
                    if (material_is_delta(chainMaterial)) {
                        float3 chainNormal = chainRec.normal;
                        if (!all(isfinite(chainNormal)) || dot(chainNormal, chainNormal) <= 0.0f) {
                            chainNormal = float3(0.0f, 1.0f, 0.0f);
                        }
                        chainNormal = normalize(chainNormal);
                        float3 chainIncident = normalize(chainRay.direction);
                        float3 chainWo = -chainIncident;
                        uint chainState = state;
                        BsdfSampleResult chainSample = sample_bsdf(chainMaterial,
                                                                   chainRec.point,
                                                                   chainNormal,
                                                                   chainWo,
                                                                   chainIncident,
                                                                   chainRec.frontFace != 0u,
                                                                   chainState,
                                                                   clampParams,
                                                                   uniforms.sssMode,
                                                                   1.0f,
                                                                   specularOnly);
                        if (chainSample.pdf > 0.0f &&
                            chainSample.isDelta &&
                            (chainSample.mediumEvent <= 0)) {
                            float3 chainDir = safe_normalize(chainSample.direction);
                            if (all(isfinite(chainDir)) && dot(chainDir, chainDir) > 0.0f) {
                                float3 chainOrigin = offset_ray_origin(chainRec, chainDir);
                                float3 combinedWeight = bsdfSample.weight * chainSample.weight;
                                float bsdfPdf = max(bsdfSample.directionalPdf * chainSample.directionalPdf,
                                                    kSpecularNeePdfFloor);
                                if (envSampling &&
                                    environmentTexture.get_width() > 0 &&
                                    environmentTexture.get_height() > 0) {
                                    if (stats) {
                                        atomic_fetch_add_explicit(&stats->mneeEnvAttemptCount, 1u, memory_order_relaxed);
                                    }
                                    Ray envRay;
                                    envRay.origin = chainOrigin;
                                    envRay.direction = normalize(chainDir);
                                    HitRecord envRec;
                                    uint chainOccMesh = kInvalidIndex;
                                    uint chainOccPrim = kInvalidIndex;
                                    bool occluded = false;
#if PT_MNEE_SWRT_RAYS
                                    occluded = trace_scene_software(uniforms,
                                                                    spheres,
                                                                    rectangles,
                                                                    triangleData,
                                                                    tlasNodes,
                                                                    tlasPrimIndices,
                                                                    instanceInfos,
                                                                    blasNodes,
                                                                    blasPrimIndices,
                                                                    nodes,
                                                                    primitiveIndices,
                                                                    stats,
                                                                    envRay,
                                                                    kEpsilon,
                                                                    kInfinity,
                                                                    /*anyHitOnly=*/true,
                                                                    /*includeTriangles=*/true,
                                                                    envRec);
#else
#if PT_DEBUG_TOOLS
                                    if (forceSoftware) {
                                        occluded = trace_scene_software(uniforms,
                                                                        spheres,
                                                                        rectangles,
                                                                        triangleData,
                                                                        tlasNodes,
                                                                        tlasPrimIndices,
                                                                        instanceInfos,
                                                                        blasNodes,
                                                                        blasPrimIndices,
                                                                        nodes,
                                                                        primitiveIndices,
                                                                        stats,
                                                                        envRay,
                                                                        kEpsilon,
                                                                        kInfinity,
                                                                        /*anyHitOnly=*/true,
                                                                        /*includeTriangles=*/true,
                                                                        envRec);
                                    } else {
#endif
#if PT_MNEE_OCCLUSION_PARITY
                                        bool occludedHw = trace_scene_hardware(uniforms,
                                                                               accel,
                                                                               meshInfos,
                                                                               triangleData,
                                                                               sceneVertices,
                                                                               meshIndices,
                                                                               instanceUserIds,
                                                                               spheres,
                                                                               rectangles,
                                                                               nodes,
                                                                               primitiveIndices,
                                                                               stats,
                                                                               envRay,
                                                                               kEpsilon,
                                                                               kInfinity,
                                                                               /*anyHitOnly=*/true,
                                                                               chainOccMesh,
                                                                               chainOccPrim,
                                                                               envRec);
                                        HitRecord envRecSw;
                                        bool occludedSw = trace_scene_software(uniforms,
                                                                               spheres,
                                                                               rectangles,
                                                                               triangleData,
                                                                               tlasNodes,
                                                                               tlasPrimIndices,
                                                                               instanceInfos,
                                                                               blasNodes,
                                                                               blasPrimIndices,
                                                                               nodes,
                                                                               primitiveIndices,
                                                                               stats,
                                                                               envRay,
                                                                               kEpsilon,
                                                                               kInfinity,
                                                                               /*anyHitOnly=*/true,
                                                                               /*includeTriangles=*/true,
                                                                               envRecSw);
                                        if (stats) {
                                            if (occludedHw) {
                                                atomic_fetch_add_explicit(&stats->mneeEnvHwOccludedCount, 1u, memory_order_relaxed);
                                            }
                                            if (occludedSw) {
                                                atomic_fetch_add_explicit(&stats->mneeEnvSwOccludedCount, 1u, memory_order_relaxed);
                                            }
                                            if (occludedHw != occludedSw) {
                                                atomic_fetch_add_explicit(&stats->mneeEnvHwSwMismatchCount, 1u, memory_order_relaxed);
                                            }
                                        }
                                        occluded = occludedHw;
#else
                                        occluded = trace_scene_hardware(uniforms,
                                                                        accel,
                                                                        meshInfos,
                                                                        triangleData,
                                                                        sceneVertices,
                                                                        meshIndices,
                                                                        instanceUserIds,
                                                                        spheres,
                                                                        rectangles,
                                                                        nodes,
                                                                        primitiveIndices,
                                                                        stats,
                                                                        envRay,
                                                                        kEpsilon,
                                                                        kInfinity,
                                                                        /*anyHitOnly=*/true,
                                                                        chainOccMesh,
                                                                        chainOccPrim,
                                                                        envRec);
#endif
#if PT_DEBUG_TOOLS
                                    }
#endif
#endif
                                    if (!occluded) {
                                        float envPdf = environment_pdf(uniforms, environmentPdf, envRay.direction);
                                        envPdf = max(envPdf, kSpecularNeePdfFloor);
                                        float invEnvPdf = min(1.0f / envPdf, kSpecularNeeInvPdfClamp);
                                        float denom = envPdf + bsdfPdf;
                                        float misWeight = (denom > 0.0f) ? (envPdf / denom) : 0.0f;
                                        misWeight = clamp(misWeight,
                                                          kMisWeightClampMin,
                                                          kMisWeightClampMax);
                                        float3 envColor = environment_color(environmentTexture,
                                                                            envRay.direction,
                                                                            uniforms.environmentRotation,
                                                                            uniforms.environmentIntensity,
                                                                            uniforms);
                                        float3 contribution = combinedWeight * envColor *
                                                              (misWeight * invEnvPdf);
                                        if (all(isfinite(contribution))) {
                                            radiance += clamp_firefly_contribution(throughput,
                                                                                   contribution,
                                                                                   clampParams);
                                            if (stats) {
                                                atomic_fetch_add_explicit(&stats->mneeEnvAddedCount, 1u, memory_order_relaxed);
                                                stats_add_mnee_luma(stats, contribution);
                                            }
                                        }
                                    }
                                }
                                if (rectLightCount > 0u) {
                                    if (stats) {
                                        atomic_fetch_add_explicit(&stats->mneeRectAttemptCount, 1u, memory_order_relaxed);
                                    }
                                    Ray lightRay;
                                    lightRay.origin = chainOrigin;
                                    lightRay.direction = normalize(chainDir);
                                    HitRecord lightRec;
                                    bool hitLight = false;
                                    MneeRectHit mneeHit;
                                    bool mneeLight = false;
#if PT_MNEE_OCCLUSION_PARITY
                                    HitRecord lightRecSw;
                                    MneeRectHit mneeHitSw;
#endif
                                    uint chainLightMesh = kInvalidIndex;
                                    uint chainLightPrim = kInvalidIndex;
#if PT_MNEE_SWRT_RAYS
                                    hitLight = trace_scene_software(uniforms,
                                                                    spheres,
                                                                    rectangles,
                                                                    triangleData,
                                                                    tlasNodes,
                                                                    tlasPrimIndices,
                                                                    instanceInfos,
                                                                    blasNodes,
                                                                    blasPrimIndices,
                                                                    nodes,
                                                                    primitiveIndices,
                                                                    stats,
                                                                    lightRay,
                                                                    kEpsilon,
                                                                    kInfinity,
                                                                    /*anyHitOnly=*/false,
                                                                    /*includeTriangles=*/true,
                                                                    lightRec);
#else
#if PT_DEBUG_TOOLS
                                    if (forceSoftware) {
                                        hitLight = trace_scene_software(uniforms,
                                                                        spheres,
                                                                        rectangles,
                                                                        triangleData,
                                                                        tlasNodes,
                                                                        tlasPrimIndices,
                                                                        instanceInfos,
                                                                        blasNodes,
                                                                        blasPrimIndices,
                                                                        nodes,
                                                                        primitiveIndices,
                                                                        stats,
                                                                        lightRay,
                                                                        kEpsilon,
                                                                        kInfinity,
                                                                        /*anyHitOnly=*/false,
                                                                        /*includeTriangles=*/true,
                                                                        lightRec);
                                    } else {
#endif
#if PT_MNEE_OCCLUSION_PARITY
                                        bool hitLightHw = trace_scene_hardware(uniforms,
                                                                               accel,
                                                                               meshInfos,
                                                                               triangleData,
                                                                               sceneVertices,
                                                                               meshIndices,
                                                                               instanceUserIds,
                                                                               spheres,
                                                                               rectangles,
                                                                               nodes,
                                                                               primitiveIndices,
                                                                               stats,
                                                                               lightRay,
                                                                               kEpsilon,
                                                                               kInfinity,
                                                                               /*anyHitOnly=*/false,
                                                                               chainLightMesh,
                                                                               chainLightPrim,
                                                                               lightRec);
                                        bool hitLightSw = trace_scene_software(uniforms,
                                                                               spheres,
                                                                               rectangles,
                                                                               triangleData,
                                                                               tlasNodes,
                                                                               tlasPrimIndices,
                                                                               instanceInfos,
                                                                               blasNodes,
                                                                               blasPrimIndices,
                                                                               nodes,
                                                                               primitiveIndices,
                                                                               stats,
                                                                               lightRay,
                                                                               kEpsilon,
                                                                               kInfinity,
                                                                               /*anyHitOnly=*/false,
                                                                               /*includeTriangles=*/true,
                                                                               lightRecSw);
                                        bool hwMneeLight = false;
                                        bool swMneeLight = false;
                                        if (hitLightHw) {
                                            hwMneeLight = mnee_rect_light_hit(uniforms,
                                                                              rectangles,
                                                                              materials,
                                                                              environmentTexture,
                                                                              rectLightCount,
                                                                              lightRec,
                                                                              chainOrigin,
                                                                              mneeHit);
                                        }
                                        if (hitLightSw) {
                                            swMneeLight = mnee_rect_light_hit(uniforms,
                                                                              rectangles,
                                                                              materials,
                                                                              environmentTexture,
                                                                              rectLightCount,
                                                                              lightRecSw,
                                                                              chainOrigin,
                                                                              mneeHitSw);
                                        }
                                        if (stats) {
                                            if (hwMneeLight) {
                                                atomic_fetch_add_explicit(&stats->mneeRectHwOccludedCount, 1u, memory_order_relaxed);
                                            }
                                            if (swMneeLight) {
                                                atomic_fetch_add_explicit(&stats->mneeRectSwOccludedCount, 1u, memory_order_relaxed);
                                            }
                                            if (hwMneeLight != swMneeLight) {
                                                atomic_fetch_add_explicit(&stats->mneeRectHwSwMismatchCount, 1u, memory_order_relaxed);
                                            }
                                        }
                                        hitLight = hitLightHw;
                                        mneeLight = hwMneeLight;
#else
                                        hitLight = trace_scene_hardware(uniforms,
                                                                        accel,
                                                                        meshInfos,
                                                                        triangleData,
                                                                        sceneVertices,
                                                                        meshIndices,
                                                                        instanceUserIds,
                                                                        spheres,
                                                                        rectangles,
                                                                        nodes,
                                                                        primitiveIndices,
                                                                        stats,
                                                                        lightRay,
                                                                        kEpsilon,
                                                                        kInfinity,
                                                                        /*anyHitOnly=*/false,
                                                                        chainLightMesh,
                                                                        chainLightPrim,
                                                                        lightRec);
#endif
#if PT_DEBUG_TOOLS
                                    }
#endif
#endif
                                    if (!mneeLight && hitLight) {
                                        mneeLight = mnee_rect_light_hit(uniforms,
                                                                        rectangles,
                                                                        materials,
                                                                        environmentTexture,
                                                                        rectLightCount,
                                                                        lightRec,
                                                                        chainOrigin,
                                                                        mneeHit);
                                    }
                                    if (mneeLight) {
                                        float lightPdf = max(mneeHit.pdf, kSpecularNeePdfFloor);
                                        float invLightPdf = min(1.0f / lightPdf, kSpecularNeeInvPdfClamp);
                                        float denom = lightPdf + bsdfPdf;
                                        float misWeight = (denom > 0.0f) ? (lightPdf / denom) : 0.0f;
                                        misWeight = clamp(misWeight,
                                                          kMisWeightClampMin,
                                                          kMisWeightClampMax);
                                        float3 contribution = combinedWeight * mneeHit.emission *
                                                              (misWeight * invLightPdf);
                                        if (all(isfinite(contribution))) {
                                            radiance += clamp_firefly_contribution(throughput,
                                                                                   contribution,
                                                                                   clampParams);
                                            if (stats) {
                                                atomic_fetch_add_explicit(&stats->mneeRectAddedCount, 1u, memory_order_relaxed);
                                                stats_add_mnee_luma(stats, contribution);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
#endif

        throughput *= bsdfSample.weight;
        throughput = clamp_path_throughput(throughput, clampParams);

        if (!all(isfinite(throughput))) {
            break;
        }

        float maxThroughput = max(max(throughput.x, throughput.y), throughput.z);
        if (maxThroughput <= 0.0f) {
            break;
        }

        bool nextEnvLodActive = false;
        float nextEnvLod = 0.0f;
        if (bsdfSample.lobeType == 1u && !bsdfSample.isDelta) {
            float maxMip = environment_max_mip(environmentTexture);
            if (maxMip > 0.0f) {
                nextEnvLod = environment_lod_from_roughness(bsdfSample.lobeRoughness,
                                                            environmentTexture);
                nextEnvLodActive = true;
            }
        }
        envLodActive = nextEnvLodActive;
        envLod = nextEnvLod;

        rayCone.width = ray_cone_width_at_distance(rayCone, hitDistanceWorld);
        rayCone.spread = min(rayCone.spread +
                             bsdf_cone_spread_increment(bsdfSample.lobeType,
                                                        bsdfSample.lobeRoughness,
                                                        bsdfSample.isDelta),
                             1.5f);

        lastBsdfPdf = (bsdfSample.directionalPdf > 0.0f) ? bsdfSample.directionalPdf : bsdfSample.pdf;
        lastScatterWasDelta = bsdfSample.isDelta;
        ray.origin = nextOrigin;
        ray.direction = bsdfSample.direction;

        if (uniforms.useRussianRoulette != 0 && depth >= 5) {
            float continueProbability = clamp(maxThroughput, 0.05f, 0.95f);
            if (rand_uniform(state) > continueProbability) {
                break;
            }
            throughput /= continueProbability;
        }
    }

    return radiance;
}
#endif
kernel void pathtraceIntegrateKernel(texture2d<float, access::read_write> radianceTexture [[texture(0)]],
                                     texture2d<uint, access::read_write> sampleCountTexture [[texture(1)]],
                                     texture2d<float, access::sample> environmentTexture [[texture(2)]],
                                     texture2d<float, access::read_write> albedoTexture [[texture(3)]],
                                     texture2d<float, access::read_write> normalTexture [[texture(4)]],
                                     array<texture2d<float, access::sample>, kMaxMaterialTextures> materialTextures [[texture(5)]],
                                     array<sampler, kMaxMaterialSamplers> materialSamplers [[sampler(0)]],
                                     constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                     device const BvhNode* nodes [[buffer(1)]],
                                     device const uint* primitiveIndices [[buffer(2)]],
                                     device const SphereData* spheres [[buffer(3)]],
                                     device const MaterialData* materials [[buffer(4)]],
                                     device PathtraceStats* stats [[buffer(5)]],
                                     device const RectData* rectangles [[buffer(6)]],
                                     device const EnvironmentAliasEntry* environmentConditionalAlias [[buffer(7)]],
                                     device const EnvironmentAliasEntry* environmentMarginalAlias [[buffer(8)]],
                                     device const float* environmentPdf [[buffer(9)]],
                                     device const TriangleData* triangleData [[buffer(10)]],
                                     device const BvhNode* tlasNodes [[buffer(11)]],
                                     device const uint* tlasPrimIndices [[buffer(12)]],
                                     device const BvhNode* blasNodes [[buffer(13)]],
                                     device const uint* blasPrimIndices [[buffer(14)]],
                                     device const SoftwareInstanceInfo* instanceInfos [[buffer(15)]],
                                     device const MeshInfo* meshInfos [[buffer(16)]],
                                     device const SceneVertex* sceneVertices [[buffer(17)]],
                                     device const uint3* meshIndices [[buffer(18)]],
                                     device PathtraceDebugBuffer* debugBuffer [[buffer(19)]],
                                     device const MaterialTextureInfo* materialTextureInfos [[buffer(20)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }

    float3 accumulated = radianceTexture.read(gid).xyz;
    uint previousCount = sampleCountTexture.read(gid).x;

    uint seed = uniforms.fixedRngSeed +                   // Deterministic base (0 if not set)
                uniforms.frameIndex * 9781u +
                gid.x * 6271u +
                gid.y * 13007u +
                (uniforms.sampleCount + previousCount) * 211u;
    thread uint rngState = seed;

    float u = (float(gid.x) + rand_uniform(rngState)) / float(uniforms.width);
    float v = (float(gid.y) + rand_uniform(rngState)) / float(uniforms.height);
    v = 1.0f - v;

    Ray ray;
    float3 pixelPosition =
        uniforms.lowerLeftCorner + u * uniforms.horizontal + v * uniforms.vertical;
    float2 diskSample = uniforms.lensRadius * random_in_unit_disk(rngState);
    float3 offset = uniforms.cameraU * diskSample.x + uniforms.cameraV * diskSample.y;
    ray.origin = uniforms.cameraOrigin + offset;
    ray.direction = pixelPosition - ray.origin;
    PrimaryRayDiff primaryRayDiff;
    primaryRayDiff.dOdx = float3(0.0f);
    primaryRayDiff.dOdy = float3(0.0f);
    primaryRayDiff.dDdx = uniforms.horizontal / max(float(uniforms.width), 1.0f);
    primaryRayDiff.dDdy = -uniforms.vertical / max(float(uniforms.height), 1.0f);

    float3 hitAlbedo = float3(0.0f);
    float3 hitNormal = float3(0.0f);

    PathtraceDebugContext debugCtx = make_debug_context(uniforms,
                                                        debugBuffer,
                                                        gid,
                                                        previousCount,
                                                        0u);
    thread PathtraceDebugContext* debugCtxPtr = nullptr;
#if PT_DEBUG_TOOLS
    debugCtxPtr = (debugBuffer && uniforms.debugPathActive != 0u) ? &debugCtx : nullptr;
#endif

    float3 sample = trace_path_software(uniforms,
                                        spheres,
                                        rectangles,
                                        triangleData,
                                        materials,
                                        meshInfos,
                                        sceneVertices,
                                        meshIndices,
                                        ray,
                                        primaryRayDiff,
                                        rngState,
                                        tlasNodes,
                                        tlasPrimIndices,
                                        blasNodes,
                                        blasPrimIndices,
                                        instanceInfos,
                                        nodes,
                                        primitiveIndices,
                                        stats,
                                        environmentTexture,
                                        materialTextures,
                                        materialSamplers,
                                        materialTextureInfos,
                                        environmentConditionalAlias,
                                        environmentMarginalAlias,
                                        environmentPdf,
                                        &hitAlbedo,
                                        &hitNormal,
                                        debugCtxPtr);
    if (!all(isfinite(sample))) {
        sample = float3(0.0f);
    } else {
        sample = max(sample, float3(0.0f));
    }

    uint newCount = previousCount + 1u;
    float3 newSum = accumulated + sample;

    radianceTexture.write(float4(newSum, 0.0f), gid);
    sampleCountTexture.write(newCount, gid);

    // Write AOV outputs (first hit albedo and normal)
    albedoTexture.write(float4(hitAlbedo, 1.0f), gid);
    normalTexture.write(float4(hitNormal * 0.5f + 0.5f, 1.0f), gid);  // Encode normal from [-1,1] to [0,1]
}

#if __METAL_VERSION__ >= 310
kernel void pathtraceIntegrateHardwareKernel(texture2d<float, access::read_write> radianceTexture [[texture(0)]],
                                             texture2d<uint, access::read_write> sampleCountTexture [[texture(1)]],
                                             texture2d<float, access::sample> environmentTexture [[texture(2)]],
                                             texture2d<float, access::read_write> albedoTexture [[texture(3)]],
                                             texture2d<float, access::read_write> normalTexture [[texture(4)]],
                                             array<texture2d<float, access::sample>, kMaxMaterialTextures> materialTextures [[texture(5)]],
                                             array<sampler, kMaxMaterialSamplers> materialSamplers [[sampler(0)]],
                                             constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                             acceleration_structure<instancing> accel [[buffer(1)]],
                                             device const MeshInfo* meshInfos [[buffer(2)]],
                                             device const TriangleData* triangleData [[buffer(3)]],
                                             device const uint* instanceUserIds [[buffer(13)]],
                                             device const BvhNode* nodes [[buffer(4)]],
                                             device const uint* primitiveIndices [[buffer(5)]],
                                             device const SphereData* spheres [[buffer(6)]],
                                             device const MaterialData* materials [[buffer(7)]],
                                             device PathtraceStats* stats [[buffer(8)]],
                                             device const RectData* rectangles [[buffer(9)]],
                                             device const EnvironmentAliasEntry* environmentConditionalAlias [[buffer(10)]],
                                             device const EnvironmentAliasEntry* environmentMarginalAlias [[buffer(11)]],
                                             device const float* environmentPdf [[buffer(12)]],
                                             device const SceneVertex* sceneVertices [[buffer(14)]],
                                             device const uint3* meshIndices [[buffer(15)]],
                                             device PathtraceDebugBuffer* debugBuffer [[buffer(16)]],
                                             device const BvhNode* tlasNodes [[buffer(17)]],
                                             device const uint* tlasPrimIndices [[buffer(18)]],
                                             device const BvhNode* blasNodes [[buffer(19)]],
                                             device const uint* blasPrimIndices [[buffer(20)]],
                                             device const SoftwareInstanceInfo* instanceInfos [[buffer(21)]],
                                             device const MaterialTextureInfo* materialTextureInfos [[buffer(22)]],
                                             uint2 gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }

    float3 accumulated = radianceTexture.read(gid).xyz;
    uint previousCount = sampleCountTexture.read(gid).x;

    uint seed = uniforms.fixedRngSeed +
                uniforms.frameIndex * 9781u +
                gid.x * 6271u +
                gid.y * 13007u +
                (uniforms.sampleCount + previousCount) * 211u;
    thread uint rngState = seed;

    float u = (float(gid.x) + rand_uniform(rngState)) / float(uniforms.width);
    float v = (float(gid.y) + rand_uniform(rngState)) / float(uniforms.height);
    v = 1.0f - v;

    Ray ray;
    float3 pixelPosition =
        uniforms.lowerLeftCorner + u * uniforms.horizontal + v * uniforms.vertical;
    float2 diskSample = uniforms.lensRadius * random_in_unit_disk(rngState);
    float3 offset = uniforms.cameraU * diskSample.x + uniforms.cameraV * diskSample.y;
    ray.origin = uniforms.cameraOrigin + offset;
    ray.direction = pixelPosition - ray.origin;
    PrimaryRayDiff primaryRayDiff;
    primaryRayDiff.dOdx = float3(0.0f);
    primaryRayDiff.dOdy = float3(0.0f);
    primaryRayDiff.dDdx = uniforms.horizontal / max(float(uniforms.width), 1.0f);
    primaryRayDiff.dDdy = -uniforms.vertical / max(float(uniforms.height), 1.0f);

    float3 hitAlbedo = float3(0.0f);
    float3 hitNormal = float3(0.0f);

    PathtraceDebugContext hwDebugCtx = make_debug_context(uniforms,
                                                          debugBuffer,
                                                          gid,
                                                          previousCount,
                                                          1u);
    thread PathtraceDebugContext* hwDebugPtr = nullptr;
#if PT_DEBUG_TOOLS
    hwDebugPtr =
        (debugBuffer && (uniforms.debugPathActive != 0u || uniforms.parityAssertEnabled != 0u))
            ? &hwDebugCtx
            : nullptr;
#endif

    float3 sample = trace_path_hardware(uniforms,
                                        accel,
                                        meshInfos,
                                        triangleData,
                                        instanceUserIds,
                                        spheres,
                                        rectangles,
                                        materials,
                                        sceneVertices,
                                        meshIndices,
                                        tlasNodes,
                                        tlasPrimIndices,
                                        blasNodes,
                                        blasPrimIndices,
                                        instanceInfos,
                                        ray,
                                        primaryRayDiff,
                                        rngState,
                                        nodes,
                                        primitiveIndices,
                                        stats,
                                        environmentTexture,
                                        materialTextures,
                                        materialSamplers,
                                        materialTextureInfos,
                                        environmentConditionalAlias,
                                        environmentMarginalAlias,
                                        environmentPdf,
                                        &hitAlbedo,
                                        &hitNormal,
                                        hwDebugPtr);
    if (!all(isfinite(sample))) {
        sample = float3(0.0f);
    } else {
        sample = max(sample, float3(0.0f));
    }

    uint newCount = previousCount + 1u;
    float3 newSum = accumulated + sample;

    radianceTexture.write(float4(newSum, 0.0f), gid);
    sampleCountTexture.write(newCount, gid);

    // Write AOV outputs (first hit albedo and normal)
    albedoTexture.write(float4(hitAlbedo, 1.0f), gid);
    normalTexture.write(float4(hitNormal * 0.5f + 0.5f, 1.0f), gid);  // Encode normal from [-1,1] to [0,1]
}
#endif

kernel void pathtracePresentKernel(texture2d<float, access::read> radianceTexture [[texture(0)]],
                                   texture2d<uint, access::read> sampleCountTexture [[texture(1)]],
                                   texture2d<float, access::write> outputTexture [[texture(2)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }

    float3 sum = radianceTexture.read(gid).xyz;
    uint count = sampleCountTexture.read(gid).x;
    float sampleCount = static_cast<float>(count);
    float3 average = (count > 0u) ? (sum / sampleCount) : float3(0.0f, 0.0f, 0.0f);

    outputTexture.write(float4(average, sampleCount), gid);
}

kernel void pathtraceClearKernel(texture2d<float, access::write> radianceTexture [[texture(0)]],
                                 texture2d<uint, access::write> sampleCountTexture [[texture(1)]],
                                 texture2d<float, access::write> outputTexture [[texture(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }

    radianceTexture.write(float4(0.0f), gid);
    sampleCountTexture.write(0u, gid);
    outputTexture.write(float4(0.0f), gid);
}
