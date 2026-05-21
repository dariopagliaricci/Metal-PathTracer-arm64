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
inline float3 safe_normalize(const float3 v);

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

inline float3 interpolate_shading_normal_raw(constant PathtraceUniforms& uniforms,
                                             uint meshIndex,
                                             uint primitiveIndex,
                                             float2 barycentric,
                                             device const MeshInfo* meshInfos,
                                             device const SceneVertex* vertices,
                                             device const uint3* meshIndices);

inline float4 interpolate_tangent_raw(constant PathtraceUniforms& uniforms,
                                      uint meshIndex,
                                      uint primitiveIndex,
                                      float2 barycentric,
                                      device const MeshInfo* meshInfos,
                                      device const SceneVertex* vertices,
                                      device const uint3* meshIndices);

inline float3 interpolate_shading_normal(constant PathtraceUniforms& uniforms,
                                         uint meshIndex,
                                         uint primitiveIndex,
                                         float2 barycentric,
                                         device const MeshInfo* meshInfos,
                                         device const SceneVertex* vertices,
                                         device const uint3* meshIndices) {
    float3 shadingNormalRaw = interpolate_shading_normal_raw(uniforms,
                                                             meshIndex,
                                                             primitiveIndex,
                                                             barycentric,
                                                             meshInfos,
                                                             vertices,
                                                             meshIndices);
    float lenSq = dot(shadingNormalRaw, shadingNormalRaw);
    if (!all(isfinite(shadingNormalRaw)) || lenSq <= 1.0e-12f) {
        return float3(0.0f);
    }
    return shadingNormalRaw * rsqrt(lenSq);
}

inline float3 interpolate_shading_normal_raw(constant PathtraceUniforms& uniforms,
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
    return normalMatrix * nLocal;
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

inline float4 interpolate_tangent_raw(constant PathtraceUniforms& uniforms,
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
    float detSign = (determinant(localToWorld) < 0.0f) ? -1.0f : 1.0f;
    float handedness = (tangentLocal.w < 0.0f) ? -1.0f : 1.0f;
    return float4(tangentWorld, handedness * detSign);
}

inline float4 interpolate_tangent(constant PathtraceUniforms& uniforms,
                                  uint meshIndex,
                                  uint primitiveIndex,
                                  float2 barycentric,
                                  device const MeshInfo* meshInfos,
                                  device const SceneVertex* vertices,
                                  device const uint3* meshIndices) {
    float4 tangentRaw = interpolate_tangent_raw(uniforms,
                                                meshIndex,
                                                primitiveIndex,
                                                barycentric,
                                                meshInfos,
                                                vertices,
                                                meshIndices);
    float3 tangentWorld = tangentRaw.xyz;
    float tangentLenSq = dot(tangentWorld, tangentWorld);
    if (!all(isfinite(tangentWorld)) || tangentLenSq <= 1.0e-12f) {
        tangentWorld = float3(1.0f, 0.0f, 0.0f);
    } else {
        tangentWorld *= rsqrt(tangentLenSq);
    }
    return float4(tangentWorld, tangentRaw.w);
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

inline bool area_primitive_is_disk(const RectData rect) {
    return rect.materialTwoSided.z == 1u;
}

inline float area_primitive_area(const RectData rect) {
    float baseArea = length(cross(rect.edgeU.xyz, rect.edgeV.xyz));
    return area_primitive_is_disk(rect) ? (kPi * baseArea) : baseArea;
}
