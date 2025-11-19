using namespace metal;

#if __METAL_VERSION__ >= 310
#include <metal_raytracing>
using namespace metal::raytracing;
#endif

constant float kInfinity = 1e20f;
constant float kPi = 3.14159265358979323846f;
constexpr sampler environmentSampler(filter::linear,
                                     address::repeat,
                                     coord::normalized);
constant float kEpsilon = 1e-3f;
constant float kRayOriginEpsilon = 1e-4f;
constant float kSssThroughputCutoff = 1e-3f;
constant float3 kLuminanceWeights = float3(0.2126f, 0.7152f, 0.0722f);
constant uint kInvalidIndex = 0xffffffffu;
constant uint kBvhTraversalStackSize = 128u;
constant float kHardwareOcclusionEpsilon = 5.0e-3f;
constant float kSpecularNeePdfFloor = 1.0e-6f;
constant float kSpecularNeeInvPdfClamp = 1.0e6f;

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

inline float3 environment_color(texture2d<float, access::sample> environmentTexture,
                                const float3 direction,
                                float rotation,
                                float intensity);

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
    float u = 1.0f - v - w;
    return float2(u, v);
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
    localIndex = min(localIndex, info.indexCount - 1u);
    uint indexEntry = info.indexOffset + localIndex;
    uint3 triIndices = meshIndices[indexEntry];
    SceneVertex v0 = vertices[triIndices.x];
    SceneVertex v1 = vertices[triIndices.y];
    SceneVertex v2 = vertices[triIndices.z];
    float u = clamp(barycentric.x, 0.0f, 1.0f);
    float v = clamp(barycentric.y, 0.0f, 1.0f);
    float w = clamp(1.0f - u - v, 0.0f, 1.0f);
    float3 n0 = v0.normal.xyz;
    float3 n1 = v1.normal.xyz;
    float3 n2 = v2.normal.xyz;
    float3 nLocal = n0 * w + n1 * u + n2 * v;
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
                                            uniforms.environmentIntensity);
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

inline float3 environment_color(texture2d<float, access::sample> environmentTexture,
                                const float3 direction,
                                float rotation,
                                float intensity) {
    float3 unit = normalize(direction);
    float cosTheta = cos(rotation);
    float sinTheta = sin(rotation);
    float3 rotated = float3(unit.x * cosTheta - unit.z * sinTheta,
                            unit.y,
                            unit.x * sinTheta + unit.z * cosTheta);
    float u = (atan2(rotated.z, rotated.x) + kPi) / (2.0f * kPi);
    float v = 0.5f - asin(clamp(rotated.y, -1.0f, 1.0f)) / kPi;
    return environmentTexture.sample(environmentSampler, float2(u, v)).xyz * intensity;
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
    return pdfA2 / denom;
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
                                        uniforms.environmentIntensity);
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
                HitRecord localRec;
                counters.leafPrimTests += 1u;
                if (!hit_triangle(uniforms,
                                  triangles[triIndex],
                                  triIndex,
                                  localRay,
                                  localTMin,
                                  localClosest,
                                  localRec)) {
                    continue;
                }

                float3 worldPoint = toWorldPoint(localRec.point);
                float3 worldNormal = normalize(normalMatrix * localRec.normal);
                if (!all(isfinite(worldPoint)) || !all(isfinite(worldNormal))) {
                    continue;
                }

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

    if (stats) {
        atomic_fetch_add_explicit(&stats->hardwareRayCount, 1u, memory_order_relaxed);
    }

    bool hadCandidateDistance = false;
    float candidateDistance = closest;

    if (meshInfos != nullptr && triangleData != nullptr &&
        uniforms.meshCount > 0u && uniforms.triangleCount > 0u) {
        intersector<triangle_data, instancing> intersector;
        intersector.assume_geometry_type(geometry_type::triangle);
        intersector.set_triangle_cull_mode(triangle_cull_mode::none);
        const uint kHardwareExcludeMaxAttempts = 4u;
        Ray currentRay = ray;
        float currentTMin = tMin;
        for (uint attempt = 0u; attempt < kHardwareExcludeMaxAttempts; ++attempt) {
            raytracing::ray query(currentRay.origin, currentRay.direction, currentTMin, tMax);
            auto result = intersector.intersect(query, accel);
            if (result.type != intersection_type::none) {
                hadCandidateDistance = true;
                candidateDistance = result.distance;
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

            if (result.type == intersection_type::none || result.distance >= closest) {
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
                break;
            }

            MeshInfo info = meshInfos[meshIndex];
            if (info.triangleCount == 0u) {
                break;
            }

            uint primitiveId = min(result.primitive_id, info.triangleCount - 1u);
            uint triIndex = info.triangleOffset + primitiveId;
            if (triIndex >= uniforms.triangleCount) {
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
            float3 localNormal = cross(localV1 - localV0, localV2 - localV0);
            float3 worldPoint = ray_at(currentRay, result.distance);

            float4x4 localToWorld = info.localToWorld;
            float4x4 worldToLocal = info.worldToLocal;
            float3x3 worldToLocal3 = float3x3(worldToLocal[0].xyz,
                                              worldToLocal[1].xyz,
                                              worldToLocal[2].xyz);
            float3 worldNormal = transpose(worldToLocal3) * localNormal;

            if (!all(isfinite(worldPoint)) || !all(isfinite(worldNormal))) {
                worldNormal = float3(0.0f);
            }

            float normalLen = length(worldNormal);
            if (normalLen > 0.0f) {
                worldNormal /= normalLen;
            } else {
                float3 worldV0 = (localToWorld * float4(localV0, 1.0f)).xyz;
                float3 worldV1 = (localToWorld * float4(localV1, 1.0f)).xyz;
                float3 worldV2 = (localToWorld * float4(localV2, 1.0f)).xyz;
                float3 fallbackNormal = cross(worldV1 - worldV0, worldV2 - worldV0);
                if (all(isfinite(fallbackNormal)) && length(fallbackNormal) > 0.0f) {
                    worldNormal = normalize(fallbackNormal);
                } else {
                    worldNormal = float3(0.0f);
                }
            }

            if (!(meshIndex < uniforms.meshCount &&
                  all(isfinite(worldNormal)) && length(worldNormal) > 0.0f)) {
                break;
            }

            bool selfHit = (meshIndex == rec.meshIndex &&
                            triIndex == rec.primitiveIndex &&
                            fabs(result.distance) <= kHardwareOcclusionEpsilon);
            bool excluded = (meshIndex == excludeMeshIndex &&
                             triIndex == excludePrimitiveIndex);

            if (selfHit) {
                if (stats) {
                    atomic_fetch_add_explicit(&stats->hardwareSelfHitRejectedCount,
                                              1u,
                                              memory_order_relaxed);
                    uint32_t distBits = as_type<uint32_t>(result.distance);
                    atomic_store_explicit(&stats->hardwareSelfHitLastDistanceBits,
                                          distBits,
                                          memory_order_relaxed);
                }
            }

            if (selfHit || excluded) {
                float advance = max(result.distance + kHardwareOcclusionEpsilon,
                                    kHardwareOcclusionEpsilon);
                currentRay.origin = ray_at(currentRay, advance);
                currentRay.origin += currentRay.direction * kHardwareOcclusionEpsilon;
                currentTMin = kHardwareOcclusionEpsilon;
                continue;
            }

            float4 localPoint4 = worldToLocal * float4(worldPoint, 1.0f);
            float invW = (fabs(localPoint4.w) > 1.0e-8f) ? (1.0f / localPoint4.w) : 1.0f;
            float3 localPoint = localPoint4.xyz * invW;
            float2 bary = barycentric_from_point(localV0, localV1, localV2, localPoint);

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

            hardwareRec.t = result.distance;
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

inline float clamp01(const float value) {
    return clamp(value, 0.0f, 1.0f);
}

inline float3 clamp01(const float3 value) {
    return clamp(value, float3(0.0f), float3(1.0f));
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

struct FireflyClampParams {
    float clampFactor;
    float clampFloor;
    float throughputClamp;
    float specularTailClampBase;
    float specularTailClampRoughnessScale;
    float minSpecularPdf;
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
    params.minSpecularPdf = max(uniforms.minSpecularPdf, 1.0e-8f);
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

    if (lum > maxLum && lum > 0.0f) {
        float scale = maxLum / max(lum, 1e-6f);
        combined *= scale;
        positive = max(combined, float3(0.0f));
    }

    return positive;
}

inline float clamp_specular_pdf(const float pdf, const FireflyClampParams params) {
    float minPdf = max(params.minSpecularPdf, 1.0e-8f);
    if (!isfinite(pdf)) {
        return minPdf;
    }
    return max(pdf, minPdf);
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
    float denom = 4.0f * max(dotWoWh, 1e-6f);
    float D = ggx_D(alpha, max(cosThetaH, 0.0f));
    return D * max(cosThetaH, 0.0f) / denom;
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
                              const float alpha,
                              thread uint& state) {
    float3 woLocal = to_local(safe_normalize(wo), normal);
    woLocal.z = max(woLocal.z, 1.0e-6f);
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
    if (material_is_plastic(material)) {
        return false;
    }
    return false;
}

inline BsdfEvalResult evaluate_bsdf(const MaterialData material,
                                    const float3 position,
                                    const float3 normal,
                                    const float3 wo,
                                    const float3 wi,
                                    const FireflyClampParams clampParams,
                                    const uint sssMode) {
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
            float3 albedo = material_base_color(material);
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
            diffuse *= tint;
            diffuse *= (float3(1.0f) - F_i) * (float3(1.0f) - F_o);
            diffuse *= max(1.0f - plastic_coat_fresnel_average(material), 0.0f);
            diffuse = max(diffuse, float3(0.0f));

            float pdfDiffuse = lambert_pdf(normal, wi);
            float pCoat = clamp(plastic_coat_sample_weight(material), 0.0f, 1.0f);
            float pDiffuse = 1.0f - pCoat;
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
                                    const uint sssMode) {
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

    uint type = static_cast<uint>(material.typeEta.x);
    switch (type) {
        case 0u: { // Lambertian
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
                break;
            }

            float alpha = roughness * roughness;
            float3 wh = sample_ggx_half_vector(normal, alpha, state);
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
                f = clamp_specular_tail(f, roughness, f0, clampParams);
                float pdf = D * max(dot(normal, wh), 0.0f) / (4.0f * dotWoWh);
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
            float3 tintThrough = plastic_diffuse_transmission(material, cosThetaI, cosThetaO);
            float3 F_i = schlick_fresnel(f0Color, cosThetaI);
            float3 F_o = schlick_fresnel(f0Color, cosThetaO);
            diffuse *= tintThrough;
            diffuse *= (float3(1.0f) - F_i) * (float3(1.0f) - F_o);
            diffuse *= max(1.0f - fresnelAvg, 0.0f);
            diffuse = max(diffuse, float3(0.0f));

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
            break;
        }
        case 5u: { // Subsurface scattering (separable diffusion)
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
            } else if (lobe == 1u) {
                float flakeRoughness = max(carpaint_flake_roughness(material), 1.0e-3f);
                float alphaFlake = flakeRoughness * flakeRoughness;
                float3 flakeNormal = carpaint_flake_normal(material, position, normal);
                float3 wh = sample_ggx_half_vector(flakeNormal, alphaFlake, state);
                if (dot(wh, flakeNormal) <= 0.0f) {
                    return result;
                }
                wi = reflect(-wo, wh);
                wi = safe_normalize(wi);
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
                    float alphaBase = baseRough * baseRough;
                    float3 wh = sample_ggx_half_vector(normal, alphaBase, state);
                    if (dot(wh, normal) <= 0.0f) {
                        return result;
                    }
                    wi = reflect(-wo, wh);
                    wi = safe_normalize(wi);
                } else {
                    float3 local = sample_cosine_hemisphere(state);
                    wi = safe_normalize(to_world(local, normal));
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
            break;
        }
        case 2u: { // Dielectric
            result.isDelta = true;
            float refIdx = max(material.typeEta.y, 1.0f);
            float etaI = 1.0f;
            float etaT = refIdx;
            float3 unitDir = incidentDir;
            float cosThetaO = dot(-unitDir, normal);
            cosThetaO = clamp(cosThetaO, -1.0f, 1.0f);
            if (!frontFace) {
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
                    result.mediumEvent = frontFace ? 1 : -1;
                }
            }

            result.direction = safe_normalize(direction);
            result.weight = weight;
            result.pdf = 1.0f;
            result.directionalPdf = 1.0f;
            break;
        }
        default:
            break;
    }

    return result;
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
        if (!trace_scene_software(uniforms,
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
                          rec)) {
            float3 background = sky_color(ray.direction);
            if (uniforms.backgroundMode == 1u) {
                background = uniforms.backgroundColor;
            } else if (uniforms.backgroundMode == 2u && environmentTexture.get_width() > 0 && environmentTexture.get_height() > 0) {
                background = environment_color(environmentTexture,
                                               ray.direction,
                                               uniforms.environmentRotation,
                                               uniforms.environmentIntensity);
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
            if (!lastScatterWasDelta && envSampling) {
                float lightPdf = environment_pdf(uniforms, environmentPdf, ray.direction);
                float denom = lastBsdfPdf + lightPdf;
                if (denom > 0.0f) {
                    misWeight = lastBsdfPdf / denom;
                }
            }
            float3 contribution = background * misWeight;
            radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
            break;
        }

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
        bool surfaceIsDelta = material_is_delta(material);
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

        if (type == 3u) {  // DiffuseLight
            float3 emission = material.emission.xyz;
            if (material.emission.w > 0.0f &&
                environmentTexture.get_width() > 0 &&
                environmentTexture.get_height() > 0 &&
                rec.frontFace != 0u) {
                float3 sampleDir = -shadingNormal;
                float3 envColor = environment_color(environmentTexture,
                                                    sampleDir,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity);
                emission *= envColor;
            }
            if (any(emission != float3(0.0f)) &&
                (rec.frontFace != 0u || rec.twoSided != 0u)) {
                float misWeight = 1.0f;
                if (!lastScatterWasDelta && rectLightCount > 0u) {
                    float lightPdf = rect_light_pdf_for_hit(uniforms,
                                                            rectangles,
                                                            materials,
                                                            rectLightCount,
                                                            rec,
                                                            ray.origin);
                    float denom = lastBsdfPdf + lightPdf;
                    if (denom > 0.0f) {
                        misWeight = lastBsdfPdf / denom;
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
                                                                uniforms.sssMode);
                        if (!bsdfEval.isDelta && !bsdfEval.isBssrdf) {
                            float3 bsdfValue = bsdfEval.value;
                            float maxComponent = max(max(bsdfValue.x, bsdfValue.y), bsdfValue.z);
                            if (maxComponent > 0.0f && lightSample.pdf > 0.0f) {
                                float bsdfPdf = bsdfEval.pdf;
                                float weight = 1.0f;
                                if (bsdfPdf > 0.0f) {
                                    weight = lightSample.pdf / (lightSample.pdf + bsdfPdf);
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
                                                                uniforms.sssMode);
                        if (!bsdfEval.isDelta && !bsdfEval.isBssrdf) {
                            float3 bsdfValue = bsdfEval.value;
                            float maxComponent = max(max(bsdfValue.x, bsdfValue.y), bsdfValue.z);
                            if (maxComponent > 0.0f) {
                                float bsdfPdf = bsdfEval.pdf;
                                float weight = 1.0f;
                                if (bsdfPdf > 0.0f) {
                                    weight = envSample.pdf / (envSample.pdf + bsdfPdf);
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
                                     uniforms.sssMode);
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

        bool specNeeEligible = (uniforms.enableSpecularNee != 0u) &&
                               envSampling &&
                               bsdfSample.isDelta &&
                               (bsdfSample.mediumEvent <= 0) &&
                               environmentTexture.get_width() > 0 &&
                               environmentTexture.get_height() > 0;
        if (specNeeEligible) {
            float dirLenSq = dot(bsdfSample.direction, bsdfSample.direction);
            if (dirLenSq <= 0.0f || !all(isfinite(bsdfSample.direction))) {
                specNeeEligible = false;
            }
        }
        if (specNeeEligible) {
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
                float misWeight = power_heuristic(envPdf, bsdfPdf);
                float3 envColor = environment_color(environmentTexture,
                                                    neeRay.direction,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity);
                float3 neeContribution = bsdfSample.weight * envColor * (misWeight * invEnvPdf);
                if (all(isfinite(neeContribution))) {
                    radiance += clamp_firefly_contribution(throughput, neeContribution, clampParams);
                }
            } else if (stats) {
                atomic_fetch_add_explicit(&stats->specularNeeOcclusionHitCount,
                                          1u,
                                          memory_order_relaxed);
            }
        }

        throughput *= bsdfSample.weight;
        throughput = clamp_path_throughput(throughput, clampParams);

        if (!all(isfinite(throughput))) {
            break;
        }

        float maxThroughput = max(max(throughput.x, throughput.y), throughput.z);
        if (maxThroughput <= 0.0f) {
            break;
        }

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
                                  Ray ray,
                                  thread uint& state,
                                  device const BvhNode* nodes,
                                  device const uint* primitiveIndices,
                                  device PathtraceStats* stats,
                                  texture2d<float, access::sample> environmentTexture,
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

    for (uint depth = 0; depth < uniforms.maxDepth; ++depth) {
        HitRecord rec;
        uint excludeMeshIndex = kInvalidIndex;
        uint excludePrimitiveIndex = kInvalidIndex;
        if (prevValid) {
            compute_exclusion_indices(prevRec, excludeMeshIndex, excludePrimitiveIndex);
        }
        if (!trace_scene_hardware(uniforms,
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
                                  rec)) {
            float3 background = sky_color(ray.direction);
            if (uniforms.backgroundMode == 1u) {
                background = uniforms.backgroundColor;
            } else if (uniforms.backgroundMode == 2u && environmentTexture.get_width() > 0 && environmentTexture.get_height() > 0) {
                background = environment_color(environmentTexture,
                                               ray.direction,
                                               uniforms.environmentRotation,
                                               uniforms.environmentIntensity);
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
            if (!lastScatterWasDelta && envSampling) {
                float lightPdf = environment_pdf(uniforms, environmentPdf, ray.direction);
                float denom = lastBsdfPdf + lightPdf;
                if (denom > 0.0f) {
                    misWeight = lastBsdfPdf / denom;
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
        bool surfaceIsDelta = material_is_delta(material);
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

        if (type == 3u) {  // DiffuseLight
            float3 emission = material.emission.xyz;
            if (material.emission.w > 0.0f &&
                environmentTexture.get_width() > 0 &&
                environmentTexture.get_height() > 0 &&
                rec.frontFace != 0u) {
                float3 sampleDir = -shadingNormal;
                float3 envColor = environment_color(environmentTexture,
                                                    sampleDir,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity);
                emission *= envColor;
            }
            if (any(emission != float3(0.0f)) &&
                (rec.frontFace != 0u || rec.twoSided != 0u)) {
                float misWeight = 1.0f;
                if (!lastScatterWasDelta && rectLightCount > 0u) {
                    float lightPdf = rect_light_pdf_for_hit(uniforms,
                                                            rectangles,
                                                            materials,
                                                            rectLightCount,
                                                            rec,
                                                            ray.origin);
                    float denom = lastBsdfPdf + lightPdf;
                    if (denom > 0.0f) {
                        misWeight = lastBsdfPdf / denom;
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
                    bool occluded = trace_scene_hardware(uniforms,
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
                    if (!occluded) {
                        BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                                rec.point,
                                                                shadingNormal,
                                                                wo,
                                                                lightSample.direction,
                                                                clampParams,
                                                                uniforms.sssMode);
                        if (!bsdfEval.isDelta && !bsdfEval.isBssrdf) {
                            float3 bsdfValue = bsdfEval.value;
                            float maxComponent = max(max(bsdfValue.x, bsdfValue.y), bsdfValue.z);
                            if (maxComponent > 0.0f && lightSample.pdf > 0.0f) {
                                float bsdfPdf = bsdfEval.pdf;
                                float weight = 1.0f;
                                if (bsdfPdf > 0.0f) {
                                    weight = lightSample.pdf / (lightSample.pdf + bsdfPdf);
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
                float nDotL = max(dot(shadingNormal, envSample.direction), 0.0f);
                if (envSample.pdf > 0.0f && nDotL > 0.0f) {
                    Ray shadowRay;
                    shadowRay.origin = offset_ray_origin(rec, envSample.direction);
                    shadowRay.direction = envSample.direction;
                    HitRecord shadowRec;
                    uint shadowExcludeMesh;
                    uint shadowExcludePrim;
                    compute_exclusion_indices(rec, shadowExcludeMesh, shadowExcludePrim);
                    bool occluded = trace_scene_hardware(uniforms,
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
                    if (!occluded) {
                        BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                                rec.point,
                                                                shadingNormal,
                                                                wo,
                                                                envSample.direction,
                                                                clampParams,
                                                                uniforms.sssMode);
                        if (!bsdfEval.isDelta && !bsdfEval.isBssrdf) {
                            float3 bsdfValue = bsdfEval.value;
                            float maxComponent = max(max(bsdfValue.x, bsdfValue.y), bsdfValue.z);
                            if (maxComponent > 0.0f) {
                                float bsdfPdf = bsdfEval.pdf;
                                float weight = 1.0f;
                                if (bsdfPdf > 0.0f) {
                                    weight = envSample.pdf / (envSample.pdf + bsdfPdf);
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
                                     uniforms.sssMode);
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
        } else {
            nextOrigin = offset_ray_origin(rec, bsdfSample.direction);
        }

        bool specNeeEligible = (uniforms.enableSpecularNee != 0u) &&
                               envSampling &&
                               bsdfSample.isDelta &&
                               (bsdfSample.mediumEvent <= 0) &&
                               environmentTexture.get_width() > 0 &&
                               environmentTexture.get_height() > 0;
        if (specNeeEligible) {
            float dirLenSq = dot(bsdfSample.direction, bsdfSample.direction);
            if (dirLenSq <= 0.0f || !all(isfinite(bsdfSample.direction))) {
                specNeeEligible = false;
            }
        }
        if (specNeeEligible) {
            Ray neeRay;
            neeRay.origin = nextOrigin;
            neeRay.direction = normalize(bsdfSample.direction);
            HitRecord shadowRec;
            uint neeExcludeMesh;
            uint neeExcludePrim;
            compute_exclusion_indices(rec, neeExcludeMesh, neeExcludePrim);
            bool occluded = trace_scene_hardware(uniforms,
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
            if (!occluded) {
                float envPdf = environment_pdf(uniforms, environmentPdf, neeRay.direction);
                envPdf = max(envPdf, kSpecularNeePdfFloor);
                float invEnvPdf = min(1.0f / envPdf, kSpecularNeeInvPdfClamp);
                float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                float misWeight = power_heuristic(envPdf, bsdfPdf);
                float3 envColor = environment_color(environmentTexture,
                                                    neeRay.direction,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity);
                float3 neeContribution = bsdfSample.weight * envColor * (misWeight * invEnvPdf);
                if (all(isfinite(neeContribution))) {
                    radiance += clamp_firefly_contribution(throughput, neeContribution, clampParams);
                }
            }
        }

        throughput *= bsdfSample.weight;
        throughput = clamp_path_throughput(throughput, clampParams);

        if (!all(isfinite(throughput))) {
            break;
        }

        float maxThroughput = max(max(throughput.x, throughput.y), throughput.z);
        if (maxThroughput <= 0.0f) {
            break;
        }

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

    float3 hitAlbedo = float3(0.0f);
    float3 hitNormal = float3(0.0f);

    PathtraceDebugContext debugCtx = make_debug_context(uniforms,
                                                        debugBuffer,
                                                        gid,
                                                        previousCount,
                                                        0u);
    thread PathtraceDebugContext* debugCtxPtr =
        (debugBuffer && uniforms.debugPathActive != 0u) ? &debugCtx : nullptr;

    float3 sample = trace_path_software(uniforms,
                                        spheres,
                                        rectangles,
                                        triangleData,
                                        materials,
                                        meshInfos,
                                        sceneVertices,
                                        meshIndices,
                                        ray,
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

    float3 hitAlbedo = float3(0.0f);
    float3 hitNormal = float3(0.0f);

    PathtraceDebugContext hwDebugCtx = make_debug_context(uniforms,
                                                          debugBuffer,
                                                          gid,
                                                          previousCount,
                                                          1u);
    thread PathtraceDebugContext* hwDebugPtr =
        (debugBuffer && uniforms.debugPathActive != 0u) ? &hwDebugCtx : nullptr;

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
                                        ray,
                                        rngState,
                                        nodes,
                                        primitiveIndices,
                                        stats,
                                        environmentTexture,
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
