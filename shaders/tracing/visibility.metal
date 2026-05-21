inline bool use_visible_emitter_mis(uint depth,
                                    bool lastScatterWasDelta,
                                    constant PathtraceUniforms& uniforms) {
    if (depth == 0u) {
        return false;
    }
    return (!lastScatterWasDelta) ||
           (uniforms.enableSpecularNee != 0u) ||
           ((ENABLE_MNEE_CAUSTICS != 0) && (uniforms.enableMnee != 0u));
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
    if (area_primitive_is_disk(rect)) {
        float u = dot(relative, rect.edgeU.xyz) * rect.edgeU.w;
        float v = dot(relative, rect.edgeV.xyz) * rect.edgeV.w;
        if ((u * u + v * v) > 1.0f) {
            return false;
        }
    } else {
        float u = dot(relative, rect.edgeU.xyz) * rect.edgeU.w;
        float v = dot(relative, rect.edgeV.xyz) * rect.edgeV.w;
        if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
            return false;
        }
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
    if (environmentTexture.get_width() == 0 || environmentTexture.get_height() == 0 ||
        !isfinite(intensity) || intensity <= 0.0f) {
        return float3(0.0f);
    }
    float3 unit = safe_normalize(direction);
    if (!all(isfinite(unit)) || dot(unit, unit) <= 0.0f) {
        return float3(0.0f);
    }
    float cosTheta = cos(rotation);
    float sinTheta = sin(rotation);
    float3 rotated = float3(unit.x * cosTheta - unit.z * sinTheta,
                            unit.y,
                            unit.x * sinTheta + unit.z * cosTheta);
    float u = (atan2(rotated.z, rotated.x) + kPi) / (2.0f * kPi);
    float v = 0.5f - asin(clamp(rotated.y, -1.0f, 1.0f)) / kPi;
    if (!isfinite(u) || !isfinite(v)) {
        return float3(0.0f);
    }
    u = clamp(u, 0.0f, 0.99999994f);
    v = clamp(v, 0.0f, 0.99999994f);
    sampler s = (uniforms.debugEnvNearest != 0u) ? environmentSamplerNearest : environmentSampler;
    float3 color = environmentTexture.sample(s, float2(u, v)).xyz * intensity;
    if (!all(isfinite(color))) {
        return float3(0.0f);
    }
    color = to_working_space(color, uniforms);
    return all(isfinite(color)) ? color : float3(0.0f);
}

inline float3 environment_color_lod(texture2d<float, access::sample> environmentTexture,
                                    const float3 direction,
                                    float rotation,
                                    float intensity,
                                    float lod,
                                    constant PathtraceUniforms& uniforms) {
    if (environmentTexture.get_width() == 0 || environmentTexture.get_height() == 0 ||
        !isfinite(intensity) || intensity <= 0.0f || !isfinite(lod)) {
        return float3(0.0f);
    }
    float3 unit = safe_normalize(direction);
    if (!all(isfinite(unit)) || dot(unit, unit) <= 0.0f) {
        return float3(0.0f);
    }
    float cosTheta = cos(rotation);
    float sinTheta = sin(rotation);
    float3 rotated = float3(unit.x * cosTheta - unit.z * sinTheta,
                            unit.y,
                            unit.x * sinTheta + unit.z * cosTheta);
    float u = (atan2(rotated.z, rotated.x) + kPi) / (2.0f * kPi);
    float v = 0.5f - asin(clamp(rotated.y, -1.0f, 1.0f)) / kPi;
    if (!isfinite(u) || !isfinite(v)) {
        return float3(0.0f);
    }
    u = clamp(u, 0.0f, 0.99999994f);
    v = clamp(v, 0.0f, 0.99999994f);
    sampler s = (uniforms.debugEnvNearest != 0u) ? environmentSamplerNearest : environmentSampler;
    float3 color = environmentTexture.sample(s,
                                             float2(u, v),
                                             level(lod)).xyz * intensity;
    if (!all(isfinite(color))) {
        return float3(0.0f);
    }
    color = to_working_space(color, uniforms);
    return all(isfinite(color)) ? color : float3(0.0f);
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
                                       thread SoftwareShadowTraceAudit* shadowAudit,
                                       thread HitRecord& rec) {
    if (shadowAudit != nullptr) {
        shadowAudit->consulted = true;
        shadowAudit->softwareBvhType = uniforms.softwareBvhType;
        shadowAudit->hasTlas = (tlasNodes != nullptr) && (tlasPrimIndices != nullptr);
        shadowAudit->hasBlas = (blasNodes != nullptr) && (blasPrimIndices != nullptr);
        shadowAudit->hasInstanceInfo = (instanceInfos != nullptr);
    }
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
                    if (shadowAudit != nullptr) {
                        shadowAudit->hit = true;
                        shadowAudit->instanceIndex = instanceId;
                        shadowAudit->meshIndex = tempRec.meshIndex;
                        shadowAudit->materialIndex = tempRec.materialIndex;
                        shadowAudit->primitiveIndex = tempRec.primitiveIndex;
                        shadowAudit->primitiveType = tempRec.primitiveType;
                        shadowAudit->frontFace = (tempRec.frontFace != 0u);
                        shadowAudit->distance = tempRec.t;
                        shadowAudit->hitPosition = tempRec.point;
                        shadowAudit->blasRootNodeOffset = info.blasRootNodeOffset;
                        shadowAudit->blasPrimIndexOffset = info.blasPrimIndexOffset;
                        shadowAudit->triangleBaseOffset = info.triangleBaseOffset;
                    }
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
    return trace_scene_tlas_triangles(uniforms,
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
                                      nullptr,
                                      rec);
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
                        thread SoftwareShadowTraceAudit* shadowAudit,
                        thread HitRecord& rec) {
    float closest = tMax;
    bool hitAnything = false;
    bool usedTlasTriangles = false;

    TraversalCounters counters;
    reset_counters(counters);
    bool earlyExit = false;

    rec.primitiveType = kPrimitiveTypeNone;
    rec.primitiveIndex = kInvalidIndex;
    if (shadowAudit != nullptr) {
        shadowAudit->consulted = true;
        shadowAudit->softwareBvhType = uniforms.softwareBvhType;
        shadowAudit->hasTlas = (tlasNodes != nullptr) && (tlasPrimIndices != nullptr);
        shadowAudit->hasBlas = (blasNodes != nullptr) && (blasPrimIndices != nullptr);
        shadowAudit->hasInstanceInfo = (instanceInfos != nullptr);
    }

    // Try TLAS path for triangles first if available
    if (uniforms.softwareBvhType == kSoftwareBvhTriangles &&
        triangles && tlasNodes && tlasPrimIndices && instanceInfos && blasNodes && blasPrimIndices) {
        usedTlasTriangles = true;
        HitRecord triangleRec = rec;
        device PathtraceStats* tlasStats = anyHitOnly ? stats : nullptr;
        if (trace_scene_tlas_triangles(uniforms,
                                       tlasNodes,
                                       tlasPrimIndices,
                                       instanceInfos,
                                       blasNodes,
                                       blasPrimIndices,
                                       triangles,
                                       tlasStats,
                                       ray,
                                       tMin,
                                       tMax,
                                       anyHitOnly,
                                       shadowAudit,
                                       triangleRec)) {
            hitAnything = true;
            closest = triangleRec.t;
            rec = triangleRec;
            if (anyHitOnly) {
                return true;
            }
        }
    }

    if (!usedTlasTriangles &&
        uniforms.intersectionMode == kIntersectionModeSoftwareBVH &&
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

    float rectangleClosest = hitAnything ? (closest + kAreaPrimitiveHitSlop) : closest;
    if (brute_force_hit_rectangles(uniforms, rectangles, ray, tMin, rectangleClosest, rec)) {
        hitAnything = true;
        closest = rec.t;
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

inline bool trace_scene_software(constant PathtraceUniforms& uniforms,
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
                        bool anyHitOnly,
                        bool includeTriangles,
                        thread HitRecord& rec) {
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
                                anyHitOnly,
                                includeTriangles,
                                nullptr,
                                rec);
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
                                 thread HardwareShadowTraceAudit* shadowAudit,
                                 thread HitRecord& rec) {
    rec.primitiveType = kPrimitiveTypeNone;
    rec.primitiveIndex = kInvalidIndex;

    if (shadowAudit != nullptr) {
        *shadowAudit = HardwareShadowTraceAudit{};
        shadowAudit->hwTraceUsed = true;
    }

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

    if (meshInfos != nullptr && sceneVertices != nullptr && meshIndices != nullptr &&
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
            if (shadowAudit != nullptr && attempt == 0u) {
                shadowAudit->rawResultAvailable = true;
                shadowAudit->rawResultType = static_cast<uint>(result.type);
                shadowAudit->rawInstanceId = result.instance_id;
                shadowAudit->rawPrimitiveId = result.primitive_id;
                shadowAudit->rawDistance = result.distance;
            }
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

            if (primitiveId >= info.indexCount) {
                if (anyHitOnly) {
                    hardwareHit = true;
                }
                break;
            }
            uint indexEntry = info.indexOffset + primitiveId;
            if (primitiveId >= info.indexCount) {
                if (anyHitOnly) {
                    hardwareHit = true;
                }
                break;
            }

            uint3 triVertexIndices = meshIndices[indexEntry];
            const uint vertexRangeBegin = info.vertexOffset;
            const uint vertexRangeEnd = info.vertexOffset + info.vertexCount;
            if (triVertexIndices.x < vertexRangeBegin || triVertexIndices.x >= vertexRangeEnd ||
                triVertexIndices.y < vertexRangeBegin || triVertexIndices.y >= vertexRangeEnd ||
                triVertexIndices.z < vertexRangeBegin || triVertexIndices.z >= vertexRangeEnd) {
                if (anyHitOnly) {
                    hardwareHit = true;
                }
                break;
            }

            SceneVertex triV0 = sceneVertices[triVertexIndices.x];
            SceneVertex triV1 = sceneVertices[triVertexIndices.y];
            SceneVertex triV2 = sceneVertices[triVertexIndices.z];

            float3 localV0 = triV0.position.xyz;
            float3 localV1 = triV1.position.xyz;
            float3 localV2 = triV2.position.xyz;
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

            // Hardware-only near-mesh hit (H5): the hardware found a same-mesh
            // triangle at short distance that the software Möller–Trumbore test
            // cannot reproduce (!resolvedHit).  SWRT runs the identical test and
            // would also miss this triangle, so HWRT must reject it too to maintain
            // parity on primary bounce rays.
            bool hwOnlyNearMeshHit = (!resolvedHit &&
                                      !anyHitOnly &&
                                      meshIndex == excludeMesh &&
                                      worldHitDistance <= kHardwareOcclusionEpsilon);

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

            if (selfHit || excluded || hwOnlyNearMeshHit) {
                if (shadowAudit != nullptr) {
                    if (selfHit) {
                        shadowAudit->rejectionMask |= kShadowAuditRejectHwSelfHit;
                    }
                    if (excluded) {
                        shadowAudit->rejectionMask |= kShadowAuditRejectHwExcluded;
                    }
                    if (hwOnlyNearMeshHit) {
                        shadowAudit->rejectionMask |= kShadowAuditRejectHwOnlyNearMesh;
                    }
                }
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
                interpolatedNormal = -interpolatedNormal;
            }

            if (all(isfinite(interpolatedNormal)) && dot(interpolatedNormal, interpolatedNormal) > 0.0f) {
                if (dot(interpolatedNormal, worldNormal) < 0.0f) {
                    interpolatedNormal = -interpolatedNormal;
                }
                interpolatedNormal = normalize(interpolatedNormal);
            } else {
                interpolatedNormal = worldNormal;
            }

            hardwareRec.t = resolvedT;
            hardwareRec.point = worldPoint;
            uint materialIndex = info.materialIndex;
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
            if (shadowAudit != nullptr) {
                shadowAudit->resolvedHit = true;
                shadowAudit->resolvedMeshIndex = meshIndex;
                shadowAudit->resolvedMaterialIndex = materialIndex;
                shadowAudit->resolvedPrimitiveIndex = triIndex;
                shadowAudit->resolvedPrimitiveType = kPrimitiveTypeTriangle;
                shadowAudit->resolvedFrontFace = triangleFrontFacing;
                shadowAudit->resolvedDistance = resolvedT;
            }
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
    float rectangleClosest = hitAnything ? (closest + kAreaPrimitiveHitSlop) : closest;
    if (brute_force_hit_rectangles(uniforms, rectangles, ray, tMin, rectangleClosest, rectRec)) {
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
    return trace_scene_hardware(uniforms,
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
                                tMin,
                                tMax,
                                anyHitOnly,
                                excludeMeshIndex,
                                excludePrimitiveIndex,
                                nullptr,
                                rec);
}
#endif

inline bool trace_scene_software_with_exclusion(constant PathtraceUniforms& uniforms,
