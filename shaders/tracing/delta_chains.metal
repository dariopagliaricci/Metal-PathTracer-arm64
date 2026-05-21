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

inline bool sss_use_separable(const uint sssMode,
                              const MaterialData material) {
    if (sssMode == 0u) {
        return false;
    }
    return (sssMode != 2u) && (material.sssParams.y < 0.5f);
}

inline bool sss_use_random_walk(const uint sssMode,
                                const MaterialData material) {
    return (sssMode == 2u) && (material.sssParams.y >= 0.5f);
}

inline RayCone wavefront_ray_cone(thread const WavefrontRay& ray) {
    RayCone cone;
    cone.width = max(ray.cone.x, 0.0f);
    cone.spread = max(ray.cone.y, 0.0f);
    return cone;
}

inline void wavefront_store_ray_cone(thread WavefrontRay& ray,
                                     const RayCone cone) {
    ray.cone = float2(max(cone.width, 0.0f), max(cone.spread, 0.0f));
}

inline PrimaryRayDiff wavefront_primary_ray_diff(thread const WavefrontRay& ray) {
    PrimaryRayDiff diff;
    diff.dOdx = ray.rayDiffDOdx.xyz;
    diff.dOdy = ray.rayDiffDOdy.xyz;
    diff.dDdx = ray.rayDiffDDdx.xyz;
    diff.dDdy = ray.rayDiffDDdy.xyz;
    return diff;
}

inline void wavefront_store_primary_ray_diff(thread WavefrontRay& ray,
                                             const PrimaryRayDiff diff) {
    ray.rayDiffDOdx = float4(diff.dOdx, 0.0f);
    ray.rayDiffDOdy = float4(diff.dOdy, 0.0f);
    ray.rayDiffDDdx = float4(diff.dDdx, 0.0f);
    ray.rayDiffDDdy = float4(diff.dDdy, 0.0f);
}

inline bool trace_environment_delta_chain_software(constant PathtraceUniforms& uniforms,
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
                                                   device const MaterialData* materials,
                                                   device PathtraceStats* stats,
                                                   thread uint& state,
                                                   const FireflyClampParams clampParams,
                                                   const bool specularOnly,
                                                   thread Ray currentRay,
                                                   thread float3& outEnvDirection,
                                                   thread float3& outChainWeight,
                                                   thread bool& outUsedDeltaChain) {
    outEnvDirection = safe_normalize(currentRay.direction);
    outChainWeight = float3(1.0f);
    outUsedDeltaChain = false;
    if (!all(isfinite(outEnvDirection)) || dot(outEnvDirection, outEnvDirection) <= 0.0f) {
        if (stats) {
            atomic_fetch_add_explicit(&stats->mneeEnvChainSampleRejectCount, 1u, memory_order_relaxed);
        }
        return false;
    }
    currentRay.direction = outEnvDirection;

    if (!materials || uniforms.materialCount == 0u ||
        !triangleData || !tlasNodes || !tlasPrimIndices ||
        !blasNodes || !blasPrimIndices || !instanceInfos ||
        !nodes || !primitiveIndices) {
        if (stats) {
            atomic_fetch_add_explicit(&stats->mneeEnvChainUnsupportedCount, 1u, memory_order_relaxed);
        }
        return false;
    }

    constexpr uint kMaxDeltaChainBounces = 6u;
    uint maxBounces = min(max(uniforms.maxDepth, 1u), kMaxDeltaChainBounces);

    HitRecord previousHit = make_empty_hit_record();
    bool previousHitValid = false;

    for (uint bounce = 0u; bounce < maxBounces; ++bounce) {
        uint excludeMesh = kInvalidIndex;
        uint excludePrim = kInvalidIndex;
        if (previousHitValid) {
            compute_exclusion_indices(previousHit, excludeMesh, excludePrim);
            excludePrim = kInvalidIndex;
        }

        HitRecord chainHit;
        bool blocked = trace_scene_software_with_exclusion(uniforms,
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
                                                           currentRay,
                                                           kEpsilon,
                                                           kInfinity,
                                                           excludeMesh,
                                                           excludePrim,
                                                           chainHit);
        if (!blocked) {
            outEnvDirection = currentRay.direction;
            return true;
        }

        uint matIndex = min(chainHit.materialIndex, uniforms.materialCount - 1u);
        MaterialData chainMaterial = materials[matIndex];
        if (!material_is_delta(chainMaterial)) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEnvChainNonDeltaBlockCount, 1u, memory_order_relaxed);
            }
            return false;
        }

        float3 chainNormal = chainHit.normal;
        if (!all(isfinite(chainNormal)) || dot(chainNormal, chainNormal) <= 0.0f) {
            chainNormal = float3(0.0f, 1.0f, 0.0f);
        } else {
            chainNormal = normalize(chainNormal);
        }

        float3 chainIncident = safe_normalize(currentRay.direction);
        if (!all(isfinite(chainIncident)) || dot(chainIncident, chainIncident) <= 0.0f) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEnvChainSampleRejectCount, 1u, memory_order_relaxed);
            }
            return false;
        }
        float3 chainWo = -chainIncident;

        BsdfSampleResult chainSample = material_closure_sample(chainMaterial,
                                                               chainHit.point,
                                                               chainNormal,
                                                               chainWo,
                                                               chainIncident,
                                                               chainHit.frontFace != 0u,
                                                               state,
                                                               clampParams,
                                                               uniforms.sssMode,
                                                               1.0f,
                                                               specularOnly);
        if (chainSample.pdf <= 0.0f || !chainSample.isDelta) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEnvChainSampleRejectCount, 1u, memory_order_relaxed);
            }
            return false;
        }
        if (!all(isfinite(chainSample.weight))) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEnvChainWeightRejectCount, 1u, memory_order_relaxed);
            }
            return false;
        }

        float3 nextDir = safe_normalize(chainSample.direction);
        if (!all(isfinite(nextDir)) || dot(nextDir, nextDir) <= 0.0f) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEnvChainSampleRejectCount, 1u, memory_order_relaxed);
            }
            return false;
        }

        outChainWeight *= chainSample.weight;
        if (!all(isfinite(outChainWeight)) || max_component(outChainWeight) <= 0.0f) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEnvChainWeightRejectCount, 1u, memory_order_relaxed);
            }
            return false;
        }
        outUsedDeltaChain = true;

        float3 nextOrigin;
        if (chainSample.hasExitPoint) {
            float3 exitNormal = chainSample.exitNormal;
            bool normalValid = all(isfinite(exitNormal)) && dot(exitNormal, exitNormal) > 0.0f;
            if (!normalValid) {
                exitNormal = chainHit.normal;
                normalValid = all(isfinite(exitNormal)) && dot(exitNormal, exitNormal) > 0.0f;
            }
            if (!normalValid) {
                exitNormal = float3(0.0f, 1.0f, 0.0f);
            }
            exitNormal = normalize(exitNormal);
            nextOrigin = offset_surface_point(chainSample.exitPoint, exitNormal, nextDir);
            float normalBias = max(kHardwareOcclusionEpsilon * 4.0f, kRayOriginEpsilon * 32.0f);
            float directionalBias = max(kHardwareOcclusionEpsilon * 8.0f, kRayOriginEpsilon * 32.0f);
            nextOrigin += exitNormal * normalBias;
            nextOrigin += nextDir * directionalBias;
        } else {
            nextOrigin = offset_ray_origin(chainHit, nextDir);
        }

        previousHit = chainHit;
        previousHitValid = true;
        currentRay.origin = nextOrigin;
        currentRay.direction = nextDir;
    }

    if (stats) {
        atomic_fetch_add_explicit(&stats->mneeEnvChainMaxBounceCount, 1u, memory_order_relaxed);
    }
    return false;
}

#if __METAL_VERSION__ >= 310
inline bool trace_environment_delta_chain_hardware(constant PathtraceUniforms& uniforms,
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
                                                   device const MaterialData* materials,
                                                   device PathtraceStats* stats,
                                                   thread uint& state,
                                                   const FireflyClampParams clampParams,
                                                   const bool specularOnly,
                                                   thread Ray currentRay,
                                                   thread float3& outEnvDirection,
                                                   thread float3& outChainWeight,
                                                   thread bool& outUsedDeltaChain) {
    outEnvDirection = safe_normalize(currentRay.direction);
    outChainWeight = float3(1.0f);
    outUsedDeltaChain = false;
    if (!all(isfinite(outEnvDirection)) || dot(outEnvDirection, outEnvDirection) <= 0.0f) {
        if (stats) {
            atomic_fetch_add_explicit(&stats->mneeEnvChainSampleRejectCount, 1u, memory_order_relaxed);
        }
        return false;
    }
    currentRay.direction = outEnvDirection;

    if (!materials || uniforms.materialCount == 0u ||
        !meshInfos || !triangleData || !sceneVertices || !meshIndices ||
        !instanceUserIds || !nodes || !primitiveIndices) {
        if (stats) {
            atomic_fetch_add_explicit(&stats->mneeEnvChainUnsupportedCount, 1u, memory_order_relaxed);
        }
        return false;
    }

    constexpr uint kMaxDeltaChainBounces = 6u;
    uint maxBounces = min(max(uniforms.maxDepth, 1u), kMaxDeltaChainBounces);

    HitRecord previousHit = make_empty_hit_record();
    bool previousHitValid = false;

    for (uint bounce = 0u; bounce < maxBounces; ++bounce) {
        uint excludeMesh = kInvalidIndex;
        uint excludePrim = kInvalidIndex;
        if (previousHitValid) {
            compute_exclusion_indices(previousHit, excludeMesh, excludePrim);
            excludePrim = kInvalidIndex;
        }

        HitRecord chainHit;
        bool blocked = trace_scene_hardware(uniforms,
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
                                            currentRay,
                                            kEpsilon,
                                            kInfinity,
                                            /*anyHitOnly=*/false,
                                            excludeMesh,
                                            excludePrim,
                                            chainHit);
        if (!blocked) {
            outEnvDirection = currentRay.direction;
            return true;
        }

        uint matIndex = min(chainHit.materialIndex, uniforms.materialCount - 1u);
        MaterialData chainMaterial = materials[matIndex];
        if (!material_is_delta(chainMaterial)) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEnvChainNonDeltaBlockCount, 1u, memory_order_relaxed);
            }
            return false;
        }

        float3 chainNormal = chainHit.normal;
        if (!all(isfinite(chainNormal)) || dot(chainNormal, chainNormal) <= 0.0f) {
            chainNormal = float3(0.0f, 1.0f, 0.0f);
        } else {
            chainNormal = normalize(chainNormal);
        }

        float3 chainIncident = safe_normalize(currentRay.direction);
        if (!all(isfinite(chainIncident)) || dot(chainIncident, chainIncident) <= 0.0f) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEnvChainSampleRejectCount, 1u, memory_order_relaxed);
            }
            return false;
        }
        float3 chainWo = -chainIncident;

        BsdfSampleResult chainSample = material_closure_sample(chainMaterial,
                                                               chainHit.point,
                                                               chainNormal,
                                                               chainWo,
                                                               chainIncident,
                                                               chainHit.frontFace != 0u,
                                                               state,
                                                               clampParams,
                                                               uniforms.sssMode,
                                                               1.0f,
                                                               specularOnly);
        if (chainSample.pdf <= 0.0f || !chainSample.isDelta) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEnvChainSampleRejectCount, 1u, memory_order_relaxed);
            }
            return false;
        }
        if (!all(isfinite(chainSample.weight))) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEnvChainWeightRejectCount, 1u, memory_order_relaxed);
            }
            return false;
        }

        float3 nextDir = safe_normalize(chainSample.direction);
        if (!all(isfinite(nextDir)) || dot(nextDir, nextDir) <= 0.0f) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEnvChainSampleRejectCount, 1u, memory_order_relaxed);
            }
            return false;
        }

        outChainWeight *= chainSample.weight;
        if (!all(isfinite(outChainWeight)) || max_component(outChainWeight) <= 0.0f) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEnvChainWeightRejectCount, 1u, memory_order_relaxed);
            }
            return false;
        }
        outUsedDeltaChain = true;

        float3 nextOrigin;
        if (chainSample.hasExitPoint) {
            float3 exitNormal = chainSample.exitNormal;
            bool normalValid = all(isfinite(exitNormal)) && dot(exitNormal, exitNormal) > 0.0f;
            if (!normalValid) {
                exitNormal = chainHit.normal;
                normalValid = all(isfinite(exitNormal)) && dot(exitNormal, exitNormal) > 0.0f;
            }
            if (!normalValid) {
                exitNormal = float3(0.0f, 1.0f, 0.0f);
            }
            exitNormal = normalize(exitNormal);
            nextOrigin = offset_surface_point(chainSample.exitPoint, exitNormal, nextDir);
            float normalBias = max(kHardwareOcclusionEpsilon * 4.0f, kRayOriginEpsilon * 32.0f);
            float directionalBias = max(kHardwareOcclusionEpsilon * 8.0f, kRayOriginEpsilon * 32.0f);
            nextOrigin += exitNormal * normalBias;
            nextOrigin += nextDir * directionalBias;
        } else {
            nextOrigin = offset_ray_origin(chainHit, nextDir);
        }

        previousHit = chainHit;
        previousHitValid = true;
        currentRay.origin = nextOrigin;
        currentRay.direction = nextDir;
    }

    if (stats) {
        atomic_fetch_add_explicit(&stats->mneeEnvChainMaxBounceCount, 1u, memory_order_relaxed);
    }
    return false;
}
#endif
