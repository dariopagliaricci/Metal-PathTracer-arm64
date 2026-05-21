constant uint kWavefrontExplicitRestirDiSurfaceAccepted = 0x52444941u;

kernel void wavefrontResetKernel(constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                 device WavefrontRay* activeRaysA [[buffer(1)]],
                                 device WavefrontRay* activeRaysB [[buffer(2)]],
                                 device WavefrontHitRecord* hitRecords [[buffer(3)]],
                                 device ShadowRay* shadowRays [[buffer(4)]],
                                 device PathState* pathStates [[buffer(5)]],
                                 device QueueCounters* queueCounters [[buffer(6)]],
                                 device PathtraceStats* stats [[buffer(7)]],
                                 device PathtraceDebugBuffer* debugBuffer [[buffer(8)]],
                                 uint gid [[thread_position_in_grid]]) {
    (void)uniformsBuffer;
    (void)activeRaysA;
    (void)activeRaysB;
    (void)hitRecords;
    (void)shadowRays;
    (void)pathStates;
    (void)debugBuffer;
    if (gid != 0u || !queueCounters) {
        return;
    }

    atomic_store_explicit(&queueCounters->activeRayCount, 0u, memory_order_relaxed);
    atomic_store_explicit(&queueCounters->nextRayCount, 0u, memory_order_relaxed);
    atomic_store_explicit(&queueCounters->shadowRayCount, 0u, memory_order_relaxed);
    atomic_store_explicit(&queueCounters->terminatedCount, 0u, memory_order_relaxed);
    atomic_store_explicit(&queueCounters->overflowFlag, 0u, memory_order_relaxed);
    atomic_store_explicit(&queueCounters->iterationIndex, 0u, memory_order_relaxed);
    const uint pixelCount = max(uniformsBuffer[0].width * uniformsBuffer[0].height, 1u);
    wavefront_init_queue_header(queueCounters->primaryRayQueue,
                                kWavefrontQueuePrimaryRay,
                                pixelCount,
                                uniformsBuffer[0].frameIndex,
                                1u,
                                1u,
                                2u);
    wavefront_init_queue_header(queueCounters->diffuseRayQueue,
                                kWavefrontQueueDiffuseRay,
                                pixelCount,
                                uniformsBuffer[0].frameIndex,
                                2u,
                                3u,
                                2u);
    wavefront_init_queue_header(queueCounters->glossyRayQueue,
                                kWavefrontQueueGlossyRay,
                                pixelCount,
                                uniformsBuffer[0].frameIndex,
                                3u,
                                3u,
                                2u);
    wavefront_init_queue_header(queueCounters->specularRayQueue,
                                kWavefrontQueueSpecularRay,
                                pixelCount,
                                uniformsBuffer[0].frameIndex,
                                4u,
                                3u,
                                2u);
    wavefront_init_queue_header(queueCounters->transmissionRayQueue,
                                kWavefrontQueueTransmissionRay,
                                pixelCount,
                                uniformsBuffer[0].frameIndex,
                                5u,
                                3u,
                                2u);
    wavefront_init_queue_header(queueCounters->shadowRayQueue,
                                kWavefrontQueueShadowRay,
                                wavefront_shadow_capacity(uniformsBuffer[0]),
                                uniformsBuffer[0].frameIndex,
                                6u,
                                3u,
                                4u);
    wavefront_init_queue_header(queueCounters->materialEvalQueue,
                                kWavefrontQueueMaterialEval,
                                pixelCount,
                                uniformsBuffer[0].frameIndex,
                                7u,
                                2u,
                                3u);
    wavefront_init_queue_header(queueCounters->reservoirUpdateQueue,
                                kWavefrontQueueReservoirUpdate,
                                pixelCount,
                                uniformsBuffer[0].frameIndex,
                                8u,
                                3u,
                                3u);
    wavefront_init_queue_header(queueCounters->mediumQueue,
                                kWavefrontQueueMedium,
                                0u,
                                uniformsBuffer[0].frameIndex,
                                9u,
                                3u,
                                3u);
    wavefront_init_queue_header(queueCounters->cacheQueryQueue,
                                kWavefrontQueueCacheQuery,
                                0u,
                                uniformsBuffer[0].frameIndex,
                                10u,
                                3u,
                                3u);
    wavefront_init_queue_header(queueCounters->cacheTrainingQueue,
                                kWavefrontQueueCacheTraining,
                                0u,
                                uniformsBuffer[0].frameIndex,
                                11u,
                                3u,
                                3u);
    wavefront_init_queue_header(queueCounters->guidingTrainingQueue,
                                kWavefrontQueueGuidingTraining,
                                0u,
                                uniformsBuffer[0].frameIndex,
                                12u,
                                3u,
                                3u);
    if (stats) {
        atomic_store_explicit(&stats->wavefrontActiveRayCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&stats->wavefrontNextRayCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&stats->wavefrontShadowRayCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&stats->wavefrontTerminatedCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&stats->wavefrontOverflowFlag, 0u, memory_order_relaxed);
        atomic_store_explicit(&stats->wavefrontTerminateMissCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&stats->wavefrontTerminateEmissiveCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&stats->wavefrontTerminateMaxDepthCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&stats->wavefrontTerminateOverflowCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&stats->wavefrontEnvShadowRayCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&stats->wavefrontAovAlbedoValidCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&stats->wavefrontAovNormalValidCount, 0u, memory_order_relaxed);
    }
}

kernel void wavefrontGeneratePrimaryKernel(constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                           device WavefrontRay* activeRays [[buffer(1)]],
                                           device PathState* pathStates [[buffer(2)]],
                                           device QueueCounters* queueCounters [[buffer(3)]],
                                           device PathtraceStats* stats [[buffer(4)]],
                                           texture2d<uint, access::read> sampleCountTexture [[texture(0)]],
                                           uint gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
    if (gid >= pixelCount) {
        return;
    }

    uint2 pixel = wavefront_pixel_coord(gid, uniforms);
    uint previousCount = sampleCountTexture.read(pixel).x;
    uint seed = uniforms.fixedRngSeed +
                uniforms.frameIndex * 9781u +
                pixel.x * 6271u +
                pixel.y * 13007u +
                (uniforms.sampleCount + previousCount) * 211u;
    thread uint rngState = seed;

    float u = (float(pixel.x) + rand_uniform(rngState)) / float(max(uniforms.width, 1u));
    float v = (float(pixel.y) + rand_uniform(rngState)) / float(max(uniforms.height, 1u));
    v = 1.0f - v;

    float3 pixelPosition =
        uniforms.lowerLeftCorner + u * uniforms.horizontal + v * uniforms.vertical;
    float2 diskSample = uniforms.lensRadius * random_in_unit_disk(rngState);
    float3 offset = uniforms.cameraU * diskSample.x + uniforms.cameraV * diskSample.y;
    float3 origin = uniforms.cameraOrigin + offset;
    float3 direction = pixelPosition - origin;

    WavefrontRay ray{};
    ray.originMinT = float4(origin, kEpsilon);
    ray.directionMaxT = float4(direction, kInfinity);
    PrimaryRayDiff primaryRayDiff;
    primaryRayDiff.dOdx = float3(0.0f);
    primaryRayDiff.dOdy = float3(0.0f);
    primaryRayDiff.dDdx = uniforms.horizontal / max(float(uniforms.width), 1.0f);
    primaryRayDiff.dDdy = -uniforms.vertical / max(float(uniforms.height), 1.0f);
    wavefront_store_primary_ray_diff(ray, primaryRayDiff);
    RayCone primaryCone = make_primary_ray_cone(uniforms);
    wavefront_store_ray_cone(ray, primaryCone);
    ray.pixelIndex = gid;
    ray.pathIndex = gid;
    ray.depth = 0u;
    ray.flags = 0u;
    ray.excludeMeshIndex = kInvalidIndex;
    ray.excludePrimitiveIndex = kInvalidIndex;
    activeRays[gid] = ray;

    PathState state{};
    state.throughput = float4(1.0f, 1.0f, 1.0f, 0.0f);
    state.radiance = float4(0.0f);
    state.firstHitAlbedo = float4(0.0f);
    state.firstHitNormal = float4(0.0f);
    state.firstHitPosition = float4(0.0f);
    state.firstHitMaterial = float4(0.0f);
    state.envLod = 0.0f;
    state.lastBsdfPdf = 1.0f;
    state.rngState = rngState;
    state.mediumDepth = 0u;
    state.currentMediumId = 0u;
    state.volumeScatteringDepth = 0u;
    SpectralPathState spectralState = spectral_make_path_state(uniforms, rngState, stats);
    state.spectralWavelengthNm = spectralState.wavelengthNm;
    state.spectralSampleIndex = spectralState.sampleIndex;
    state.alive = 1u;
    state.flags = kWavefrontFlagLastScatterDelta;
    pathStates[gid] = state;

    if (queueCounters) {
        atomic_fetch_add_explicit(&queueCounters->activeRayCount, 1u, memory_order_relaxed);
        wavefront_note_queue_enqueue(queueCounters->primaryRayQueue);
    }
    if (stats) {
        atomic_fetch_add_explicit(&stats->wavefrontPrimaryGeneratedCount, 1u, memory_order_relaxed);
    }
    if (stats && gid == pixelCount - 1u) {
        wavefront_store_queue_stats(queueCounters, stats);
    }
}

kernel void wavefrontPrepareIterationKernel(device QueueCounters* queueCounters [[buffer(0)]],
                                            device PathtraceStats* stats [[buffer(1)]],
                                            device PathState* pathStates [[buffer(2)]],
                                            constant PathtraceUniforms* uniformsBuffer [[buffer(3)]],
                                            uint gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
    if (pathStates && gid < pixelCount) {
        pathStates[gid].reserved0 = 0u;
        pathStates[gid].reserved1 = 0u;
    }
    if (gid == 0u && queueCounters) {
        atomic_store_explicit(&queueCounters->nextRayCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&queueCounters->shadowRayCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&queueCounters->diffuseRayQueue.activeCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&queueCounters->glossyRayQueue.activeCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&queueCounters->specularRayQueue.activeCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&queueCounters->transmissionRayQueue.activeCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&queueCounters->shadowRayQueue.activeCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&queueCounters->materialEvalQueue.activeCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&queueCounters->reservoirUpdateQueue.activeCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&queueCounters->mediumQueue.activeCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&queueCounters->cacheQueryQueue.activeCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&queueCounters->cacheTrainingQueue.activeCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&queueCounters->guidingTrainingQueue.activeCount, 0u, memory_order_relaxed);
        atomic_fetch_add_explicit(&queueCounters->iterationIndex, 1u, memory_order_relaxed);
        wavefront_store_queue_stats(queueCounters, stats);
    }
}

kernel void wavefrontBuildDispatchArgsKernel(device QueueCounters* queueCounters [[buffer(0)]],
                                             device uint* indirectArgs [[buffer(1)]],
                                             constant uint2& args [[buffer(2)]],
                                             uint gid [[thread_position_in_grid]]) {
    if (gid != 0u || !queueCounters || !indirectArgs) {
        return;
    }
    const uint queueType = args.x;
    const uint threadsPerThreadgroup = max(args.y, 1u);
    uint activeCount = 0u;
    switch (queueType) {
    case kWavefrontQueueShadowRay:
        activeCount = atomic_load_explicit(&queueCounters->shadowRayQueue.activeCount,
                                           memory_order_relaxed);
        break;
    case kWavefrontQueueDiffuseRay:
        activeCount = atomic_load_explicit(&queueCounters->diffuseRayQueue.activeCount,
                                           memory_order_relaxed);
        break;
    case kWavefrontQueueGlossyRay:
        activeCount = atomic_load_explicit(&queueCounters->glossyRayQueue.activeCount,
                                           memory_order_relaxed);
        break;
    case kWavefrontQueueSpecularRay:
        activeCount = atomic_load_explicit(&queueCounters->specularRayQueue.activeCount,
                                           memory_order_relaxed);
        break;
    case kWavefrontQueueTransmissionRay:
        activeCount = atomic_load_explicit(&queueCounters->transmissionRayQueue.activeCount,
                                           memory_order_relaxed);
        break;
    case kWavefrontQueueMaterialEval:
        activeCount = atomic_load_explicit(&queueCounters->materialEvalQueue.activeCount,
                                           memory_order_relaxed);
        break;
    case kWavefrontQueueReservoirUpdate:
        activeCount = atomic_load_explicit(&queueCounters->reservoirUpdateQueue.activeCount,
                                           memory_order_relaxed);
        break;
    case kWavefrontQueueMedium:
        activeCount = atomic_load_explicit(&queueCounters->mediumQueue.activeCount,
                                           memory_order_relaxed);
        break;
    case kWavefrontQueueCacheQuery:
        activeCount = atomic_load_explicit(&queueCounters->cacheQueryQueue.activeCount,
                                           memory_order_relaxed);
        break;
    case kWavefrontQueueCacheTraining:
        activeCount = atomic_load_explicit(&queueCounters->cacheTrainingQueue.activeCount,
                                           memory_order_relaxed);
        break;
    case kWavefrontQueueGuidingTraining:
        activeCount = atomic_load_explicit(&queueCounters->guidingTrainingQueue.activeCount,
                                           memory_order_relaxed);
        break;
    case kWavefrontQueuePrimaryRay:
    default:
        activeCount = atomic_load_explicit(&queueCounters->primaryRayQueue.activeCount,
                                           memory_order_relaxed);
        break;
    }
    indirectArgs[0] = max((activeCount + threadsPerThreadgroup - 1u) / threadsPerThreadgroup, 1u);
    indirectArgs[1] = 1u;
    indirectArgs[2] = 1u;
}

kernel void wavefrontCompactQueueKernel(device QueueCounters* queueCounters [[buffer(0)]],
                                        device PathtraceStats* stats [[buffer(1)]],
                                        constant PathtraceUniforms* uniformsBuffer [[buffer(2)]],
                                        uint gid [[thread_position_in_grid]]) {
    if (gid != 0u || !queueCounters) {
        return;
    }
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    if (uniforms.wavefrontCompactionEnabled == 0u) {
        wavefront_store_queue_stats(queueCounters, stats);
        return;
    }

    // The current wavefront queue storage is already compact-by-construction:
    // producers reserve contiguous slots with atomic counters. This pass is the
    // explicit compaction barrier/checkpoint used by Final/Research scheduling.
    const uint activeCount =
        atomic_load_explicit(&queueCounters->activeRayCount, memory_order_relaxed);
    const uint capacity = queueCounters->primaryRayQueue.capacity;
    const uint boundedActive = min(activeCount, capacity);
    atomic_store_explicit(&queueCounters->primaryRayQueue.activeCount,
                          boundedActive,
                          memory_order_relaxed);
    atomic_fetch_max_explicit(&queueCounters->primaryRayQueue.peakActiveCount,
                              boundedActive,
                              memory_order_relaxed);
    if (activeCount > capacity) {
        atomic_store_explicit(&queueCounters->overflowFlag, 1u, memory_order_relaxed);
        wavefront_note_queue_overflow(queueCounters->primaryRayQueue);
    }
    if (uniforms.wavefrontSchedulingPolicy == 2u) {
        wavefront_store_queue_stats(queueCounters, stats);
    }
}

kernel void wavefrontIntersectKernel(constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                     device const WavefrontRay* activeRays [[buffer(1)]],
                                     device WavefrontHitRecord* hitRecords [[buffer(2)]],
                                     device PathtraceStats* stats [[buffer(3)]],
                                     device const SphereData* spheres [[buffer(4)]],
                                     device const RectData* rectangles [[buffer(5)]],
                                     device const TriangleData* triangleData [[buffer(6)]],
                                     device const BvhNode* tlasNodes [[buffer(7)]],
                                     device const uint* tlasPrimIndices [[buffer(8)]],
                                     device const SoftwareInstanceInfo* instanceInfos [[buffer(9)]],
                                     device const BvhNode* blasNodes [[buffer(10)]],
                                     device const uint* blasPrimIndices [[buffer(11)]],
                                     device const BvhNode* nodes [[buffer(12)]],
                                     device const uint* primitiveIndices [[buffer(13)]],
                                     device const MeshInfo* meshInfos [[buffer(14)]],
                                     device const SceneVertex* sceneVertices [[buffer(15)]],
                                     device const uint3* meshIndices [[buffer(16)]],
                                     device QueueCounters* queueCounters [[buffer(17)]],
                                     uint gid [[thread_position_in_grid]]) {
    (void)meshInfos;
    (void)sceneVertices;
    (void)meshIndices;
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    const uint activeCount =
        queueCounters ? atomic_load_explicit(&queueCounters->activeRayCount, memory_order_relaxed) : 0u;
    if (gid >= activeCount) {
        return;
    }
    if (stats) {
        atomic_fetch_add_explicit(&stats->wavefrontIntersectProcessedCount, 1u, memory_order_relaxed);
    }

    WavefrontRay queuedRay = activeRays[gid];
    Ray ray;
    ray.origin = queuedRay.originMinT.xyz;
    ray.direction = queuedRay.directionMaxT.xyz;

    HitRecord rec = make_empty_hit_record();
    const bool hit = trace_scene_software_with_exclusion(uniforms,
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
                                                         queuedRay.originMinT.w,
                                                         queuedRay.directionMaxT.w,
                                                         queuedRay.excludeMeshIndex,
                                                         queuedRay.excludePrimitiveIndex,
                                                         rec);
    hitRecords[gid] = pack_wavefront_hit(rec, queuedRay.pathIndex, hit);
}

#if __METAL_VERSION__ >= 310
kernel void wavefrontIntersectHardwareKernel(constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                             acceleration_structure<instancing> accel [[buffer(1)]],
                                             device const WavefrontRay* activeRays [[buffer(2)]],
                                             device WavefrontHitRecord* hitRecords [[buffer(3)]],
                                             device const MeshInfo* meshInfos [[buffer(4)]],
                                             device const TriangleData* triangleData [[buffer(5)]],
                                             device const SceneVertex* sceneVertices [[buffer(6)]],
                                             device const uint3* meshIndices [[buffer(7)]],
                                             device const uint* instanceUserIds [[buffer(8)]],
                                             device const SphereData* spheres [[buffer(9)]],
                                             device const RectData* rectangles [[buffer(10)]],
                                             device const BvhNode* nodes [[buffer(11)]],
                                             device const uint* primitiveIndices [[buffer(12)]],
                                             device PathtraceStats* stats [[buffer(13)]],
                                             device QueueCounters* queueCounters [[buffer(14)]],
                                             uint gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    const uint activeCount =
        queueCounters ? atomic_load_explicit(&queueCounters->activeRayCount, memory_order_relaxed) : 0u;
    if (gid >= activeCount) {
        return;
    }
    if (stats) {
        atomic_fetch_add_explicit(&stats->wavefrontIntersectProcessedCount, 1u, memory_order_relaxed);
    }

    WavefrontRay queuedRay = activeRays[gid];
    Ray ray;
    ray.origin = queuedRay.originMinT.xyz;
    ray.direction = queuedRay.directionMaxT.xyz;

    HitRecord rec = make_empty_hit_record();
    const bool hit = trace_scene_hardware(uniforms,
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
                                          queuedRay.originMinT.w,
                                          queuedRay.directionMaxT.w,
                                          /*anyHitOnly=*/false,
                                          queuedRay.excludeMeshIndex,
                                          queuedRay.excludePrimitiveIndex,
                                          rec);
    hitRecords[gid] = pack_wavefront_hit(rec, queuedRay.pathIndex, hit);
}
#endif

kernel void wavefrontRestirDiDirectLightingKernel(
                                 constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                 device const WavefrontRay* activeRays [[buffer(1)]],
                                 device const WavefrontHitRecord* hitRecords [[buffer(2)]],
                                 device WavefrontRay* nextRays [[buffer(3)]],
                                 device ShadowRay* shadowRays [[buffer(4)]],
                                 device PathState* pathStates [[buffer(5)]],
                                 device QueueCounters* queueCounters [[buffer(6)]],
                                 device PathtraceStats* stats [[buffer(7)]],
                                 texture2d<float, access::sample> environmentTexture [[texture(0)]],
                                 device const MaterialData* materials [[buffer(8)]],
                                 device const RectData* rectangles [[buffer(9)]],
                                 device PathtraceDebugBuffer* debugBuffer [[buffer(10)]],
                                 device const EnvironmentAliasEntry* environmentConditionalAlias [[buffer(11)]],
                                 device const EnvironmentAliasEntry* environmentMarginalAlias [[buffer(12)]],
                                 device const float* environmentPdf [[buffer(13)]],
                                 device const MeshInfo* meshInfos [[buffer(14)]],
                                 device const SceneVertex* sceneVertices [[buffer(15)]],
                                 device const uint3* meshIndices [[buffer(16)]],
                                 device const MaterialTextureInfo* materialTextureInfos [[buffer(17)]],
                                 acceleration_structure<instancing> accel [[buffer(18)]],
                                 device const SphereData* spheres [[buffer(19)]],
                                 device const TriangleData* triangleData [[buffer(20)]],
                                 device const BvhNode* tlasNodes [[buffer(21)]],
                                 device const uint* tlasPrimIndices [[buffer(22)]],
                                 constant MaterialTextureArgumentBuffer& materialResources [[buffer(23)]],
                                 device const BvhNode* blasNodes [[buffer(24)]],
                                 device const uint* blasPrimIndices [[buffer(25)]],
                                 device const SoftwareInstanceInfo* instanceInfos [[buffer(26)]],
                                 device const BvhNode* nodes [[buffer(27)]],
                                 device const uint* primitiveIndices [[buffer(28)]],
                                 device const uint* instanceUserIds [[buffer(29)]],
                                 device const LightPrimitive* emissivePrimitives [[buffer(30)]],
                                 uint gid [[thread_position_in_grid]]) {
    (void)nextRays;
    (void)debugBuffer;
    (void)environmentConditionalAlias;
    (void)environmentMarginalAlias;
    (void)environmentPdf;
    (void)accel;
    (void)instanceUserIds;

    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    if (uniforms.explicitRestirDiLightingStage == 0u ||
        !(uniforms.directLightMode == kDirectLightModeRestirDi ||
          uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) ||
        uniforms.emissivePrimitiveCount == 0u ||
        !emissivePrimitives) {
        return;
    }

    const uint activeCount =
        queueCounters ? atomic_load_explicit(&queueCounters->activeRayCount, memory_order_relaxed) : 0u;
    if (gid >= activeCount) {
        return;
    }

    WavefrontRay queuedRay = activeRays[gid];
    const uint pathIndex = queuedRay.pathIndex;
    PathState state = pathStates[pathIndex];
    if (state.alive == 0u) {
        return;
    }

    const WavefrontHitRecord packedHit = hitRecords[gid];
    if (packedHit.hit == 0u) {
        return;
    }

    Ray incomingRay;
    incomingRay.origin = queuedRay.originMinT.xyz;
    incomingRay.direction = queuedRay.directionMaxT.xyz;
    HitRecord rec = unpack_wavefront_hit(packedHit);
    if (!all(isfinite(rec.shadingNormal)) || dot(rec.shadingNormal, rec.shadingNormal) <= 0.0f) {
        rec.shadingNormal = rec.normal;
    }
    rec.shadingNormal = safe_normalize(rec.shadingNormal);

    float3 lightingThroughput = state.throughput.xyz;
    const uint mediumDepth = min(state.mediumDepth, kWavefrontMediumStackSize);
    if (mediumDepth > 0u) {
        float3 sigma = state.mediumStack[mediumDepth - 1u].xyz;
        if (any(sigma > float3(0.0f))) {
            lightingThroughput *= exp(-sigma * max(rec.t, 0.0f));
        }
    }

    uint rngState = state.rngState;
    const float3 incidentDir = safe_normalize(incomingRay.direction);
    const float3 wo = safe_normalize(-incomingRay.direction);
    uint materialIndex = uniforms.materialCount > 0u
        ? min(rec.materialIndex, uniforms.materialCount - 1u)
        : 0u;
    MaterialData baseMaterial = materials ? materials[materialIndex] : MaterialData{};
    RayCone rayCone = wavefront_ray_cone(queuedRay);
    if (!(rayCone.width > 0.0f) && !(rayCone.spread > 0.0f)) {
        rayCone = make_primary_ray_cone(uniforms);
    }
    WavefrontSurfaceState surfaceState = resolve_wavefront_surface_state(
        uniforms,
        rec,
        baseMaterial,
        materialIndex,
        queuedRay.depth,
        incomingRay,
        wavefront_primary_ray_diff(queuedRay),
        incomingRay.direction,
        ray_segment_world_length(incomingRay, rec.t),
        rayCone,
        rngState,
        false,
        materialResources.textures,
        materialResources.samplers,
        materialTextureInfos,
        meshInfos,
        sceneVertices,
        meshIndices);
    if (surfaceState.alphaDiscarded || uniforms.debugViewMode != kDebugViewNone) {
        return;
    }

    rec = surfaceState.rec;
    MaterialData material = surfaceState.material;
    state.rngState = rngState;
    state.reserved1 = kWavefrontExplicitRestirDiSurfaceAccepted;
    pathStates[pathIndex] = state;
    if (static_cast<uint>(material.typeEta.x) == 3u || material_is_delta(material)) {
        return;
    }

    const float3 shadingNormal = rec.shadingNormal;
    const float diffuseOcclusion = surfaceState.diffuseOcclusion;
    const bool specularOnly = (uniforms.debugSpecularOnly != 0u);
    FireflyClampParams clampParams = make_firefly_params(uniforms);

    Reservoir reservoir = make_empty_reservoir();
    uint risTargetM = max(uniforms.risCandidateCount, 1u);
    if (uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid &&
        uniforms.frameIndex > 0u) {
        risTargetM *= 2u;
    }
    build_ris_reservoir_for_hit(uniforms,
                                emissivePrimitives,
                                rec,
                                material,
                                shadingNormal,
                                wo,
                                clampParams,
                                diffuseOcclusion,
                                specularOnly,
                                risTargetM,
                                rngState,
                                reservoir);

    if ((uniforms.directLightMode == kDirectLightModeRestirDi ||
         uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
        reservoir.valid &&
        queuedRay.depth == 0u &&
        rec.t > 0.0f &&
        uniforms.frameIndex > 0u) {
        uint temporalState = pcg_hash(uniforms.fixedRngSeed ^
                                      (uint(queuedRay.pixelIndex % max(uniforms.width, 1u)) * 1664525u) ^
                                      (uint(queuedRay.pixelIndex / max(uniforms.width, 1u)) * 1013904223u) ^
                                      ((uniforms.frameIndex - 1u) * 9781u) ^
                                      0x9E3779B9u);
        Reservoir previousReservoir = make_empty_reservoir();
        if (build_ris_reservoir_for_hit(uniforms,
                                        emissivePrimitives,
                                        rec,
                                        material,
                                        shadingNormal,
                                        wo,
                                        clampParams,
                                        diffuseOcclusion,
                                        specularOnly,
                                        risTargetM,
                                        temporalState,
                                        previousReservoir) &&
            previousReservoir.valid) {
            float phatCurrent = reservoir_winner_phat_for_hit(uniforms,
                                                              rec,
                                                              material,
                                                              shadingNormal,
                                                              wo,
                                                              clampParams,
                                                              diffuseOcclusion,
                                                              specularOnly,
                                                              previousReservoir.winner);
            if (phatCurrent > 0.0f &&
                isfinite(phatCurrent) &&
                previousReservoir.wSum > 0.0f &&
                isfinite(previousReservoir.wSum)) {
                merge_reservoir_winner(reservoir,
                                       previousReservoir.winner,
                                       previousReservoir.wSum,
                                       previousReservoir.M,
                                       rand_uniform(rngState));
            }
        }
    }

    if ((uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
        reservoir.valid &&
        queuedRay.depth == 0u &&
        rec.t > 0.0f &&
        uniforms.frameIndex > 0u) {
        int3 cacheCell = world_reuse_cell(rec.point, uniforms.worldReuseCellSize);
        uint cacheSourceFrameIndex =
            uniforms.frameIndex > 1u ? uniforms.frameIndex - 2u : uniforms.frameIndex - 1u;
        for (uint cacheProbe = 0u; cacheProbe < 2u; ++cacheProbe) {
            if (cacheProbe > cacheSourceFrameIndex) {
                continue;
            }
            uint candidateSourceFrameIndex = cacheSourceFrameIndex - cacheProbe;
            uint cacheState =
                pcg_hash(uniforms.fixedRngSeed ^
                         (uint(cacheCell.x) * 1664525u) ^
                         (uint(cacheCell.y) * 1013904223u) ^
                         (uint(cacheCell.z) * 747796405u) ^
                         (candidateSourceFrameIndex * 9781u) ^
                         (cacheProbe * 0x9E3779B9u) ^
                         0xD1B54A35u);
            Reservoir cachedReservoir = make_empty_reservoir();
            if (!build_ris_reservoir_for_hit(uniforms,
                                             emissivePrimitives,
                                             rec,
                                             material,
                                             shadingNormal,
                                             wo,
                                             clampParams,
                                             diffuseOcclusion,
                                             specularOnly,
                                             risTargetM,
                                             cacheState,
                                             cachedReservoir) ||
                !cachedReservoir.valid) {
                continue;
            }
            float phatCurrent = reservoir_winner_phat_for_hit(uniforms,
                                                              rec,
                                                              material,
                                                              shadingNormal,
                                                              wo,
                                                              clampParams,
                                                              diffuseOcclusion,
                                                              specularOnly,
                                                              cachedReservoir.winner);
            float phatCached = reservoir_winner_phat_for_hit(uniforms,
                                                             rec,
                                                             material,
                                                             shadingNormal,
                                                             wo,
                                                             clampParams,
                                                             diffuseOcclusion,
                                                             specularOnly,
                                                             cachedReservoir.winner);
            float denom = phatCurrent * float(max(reservoir.M, 1u)) +
                          phatCached * float(max(cachedReservoir.M, 1u));
            if (denom > 0.0f && isfinite(denom) && phatCurrent > 0.0f) {
                float mis = phatCurrent / denom;
                float mergeWeight = mis * cachedReservoir.wSum;
                if (mergeWeight > 0.0f && isfinite(mergeWeight)) {
                    update_reservoir(reservoir,
                                     cachedReservoir.winner,
                                     mergeWeight,
                                     rand_uniform(rngState));
                }
            }
        }
    }

    if (reservoir.valid && reservoir.M > 0u && reservoir.wSum > 0.0f) {
        RISSamplePayload winner = reservoir.winner;
        float distSq = ris_payload_distance_sq(rec, winner);
        if (distSq > 0.0f && isfinite(distSq)) {
            const bool winnerDirectional = ris_payload_is_directional(winner);
            float distance = winnerDirectional ? kInfinity : length(winner.position - rec.point);
            float3 omegaL = ris_payload_omega_l(rec, winner);
            float cosThetaSurface = saturate(dot(shadingNormal, omegaL));
            float cosThetaLight = winnerDirectional ? 1.0f : saturate(dot(winner.normal, -omegaL));
            BsdfEvalResult eval = evaluate_bsdf(material,
                                                rec.point,
                                                shadingNormal,
                                                wo,
                                                omegaL,
                                                clampParams,
                                                uniforms.sssMode,
                                                diffuseOcclusion,
                                                specularOnly);
            float phatW = p_hat(winner.emission,
                                eval.value,
                                cosThetaSurface,
                                cosThetaLight,
                                distSq);
            reservoir.W = (phatW > 0.0f)
                ? ((reservoir.wSum / float(reservoir.M)) / phatW)
                : 0.0f;
            if (!eval.isDelta &&
                !eval.isBssrdf &&
                reservoir.W > 0.0f &&
                cosThetaSurface > 0.0f &&
                all(isfinite(eval.value))) {
                float3 contribution = eval.value * winner.emission;
                contribution *= cosThetaSurface * reservoir.W;
                if (all(isfinite(contribution))) {
                    contribution = clamp_firefly_contribution(lightingThroughput,
                                                              contribution,
                                                              clampParams,
                                                              stats);
                    uint shadowIndex = wavefront_reserve_shadow_slot(uniforms,
                                                                     pathIndex,
                                                                     state,
                                                                     queueCounters,
                                                                     stats);
                    if (shadowIndex != kInvalidIndex) {
                        uint shadowExcludeMesh;
                        uint shadowExcludePrim;
                        compute_exclusion_indices(rec, shadowExcludeMesh, shadowExcludePrim);
                        ShadowRay shadow{};
                        shadow.originTMax = float4(offset_ray_origin(rec, omegaL),
                                                   winnerDirectional
                                                       ? kInfinity
                                                       : max(distance - kRayOriginEpsilon * 4.0f,
                                                             kRayOriginEpsilon));
                        shadow.direction = float4(omegaL, 0.0f);
                        shadow.contribution = float4(contribution, 0.0f);
                        shadow.throughput = float4(lightingThroughput, 0.0f);
                        shadow.pathIndex = pathIndex;
                        shadow.lightId = winner.primitiveIndex;
                        shadow.flags = winnerDirectional ? 0u : kWavefrontShadowFlagEmissivePrimitive;
                        shadow.excludeMeshIndex = shadowExcludeMesh;
                        shadow.excludePrimitiveIndex = shadowExcludePrim;
                        shadowRays[shadowIndex] = shadow;
                    }
                }
            }
        }
    }

    state.rngState = rngState;
    pathStates[pathIndex] = state;
}

kernel void wavefrontShadeKernel(constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                 device const WavefrontRay* activeRays [[buffer(1)]],
                                 device const WavefrontHitRecord* hitRecords [[buffer(2)]],
                                 device WavefrontRay* nextRays [[buffer(3)]],
                                 device ShadowRay* shadowRays [[buffer(4)]],
                                 device PathState* pathStates [[buffer(5)]],
                                 device QueueCounters* queueCounters [[buffer(6)]],
                                 device PathtraceStats* stats [[buffer(7)]],
                                 texture2d<float, access::sample> environmentTexture [[texture(0)]],
                                 device const MaterialData* materials [[buffer(8)]],
                                 device const RectData* rectangles [[buffer(9)]],
                                 device PathtraceDebugBuffer* debugBuffer [[buffer(10)]],
                                 device const EnvironmentAliasEntry* environmentConditionalAlias [[buffer(11)]],
                                 device const EnvironmentAliasEntry* environmentMarginalAlias [[buffer(12)]],
                                 device const float* environmentPdf [[buffer(13)]],
                                 device const MeshInfo* meshInfos [[buffer(14)]],
                                 device const SceneVertex* sceneVertices [[buffer(15)]],
                                 device const uint3* meshIndices [[buffer(16)]],
                                 device const MaterialTextureInfo* materialTextureInfos [[buffer(17)]],
                                 acceleration_structure<instancing> accel [[buffer(18)]],
                                 device const SphereData* spheres [[buffer(19)]],
                                 device const TriangleData* triangleData [[buffer(20)]],
                                 device const BvhNode* tlasNodes [[buffer(21)]],
                                 device const uint* tlasPrimIndices [[buffer(22)]],
                                 constant MaterialTextureArgumentBuffer& materialResources [[buffer(23)]],
                                 device const BvhNode* blasNodes [[buffer(24)]],
                                 device const uint* blasPrimIndices [[buffer(25)]],
                                 device const SoftwareInstanceInfo* instanceInfos [[buffer(26)]],
                                 device const BvhNode* nodes [[buffer(27)]],
                                 device const uint* primitiveIndices [[buffer(28)]],
                                 device const uint* instanceUserIds [[buffer(29)]],
                                 device const LightPrimitive* emissivePrimitives [[buffer(30)]],
                                 uint gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    const uint activeCount =
        queueCounters ? atomic_load_explicit(&queueCounters->activeRayCount, memory_order_relaxed) : 0u;
    if (gid >= activeCount) {
        return;
    }
    if (stats) {
        atomic_fetch_add_explicit(&stats->wavefrontShadeProcessedCount, 1u, memory_order_relaxed);
    }

    WavefrontRay queuedRay = activeRays[gid];
    const uint pathIndex = queuedRay.pathIndex;
    PathState state = pathStates[pathIndex];
    if (state.alive == 0u) {
        return;
    }
    uint mediumDepth = min(state.mediumDepth, kWavefrontMediumStackSize);
    float3 mediumSigmaStack[kWavefrontMediumStackSize];
    for (uint i = 0u; i < kWavefrontMediumStackSize; ++i) {
        mediumSigmaStack[i] = state.mediumStack[i].xyz;
    }

    uint2 pixel = wavefront_pixel_coord(queuedRay.pixelIndex, uniforms);
    PathtraceDebugContext debugCtx = make_debug_context(uniforms,
                                                        debugBuffer,
                                                        pixel,
                                                        uniforms.sampleCount,
                                                        2u);
    thread PathtraceDebugContext* debugCtxPtr = nullptr;
#if PT_DEBUG_TOOLS
    debugCtxPtr = (debugBuffer && uniforms.debugPathActive != 0u) ? &debugCtx : nullptr;
#endif

    thread uint rngState = state.rngState;
    FireflyClampParams clampParams = make_firefly_params(uniforms);

    const WavefrontHitRecord packedHit = hitRecords[gid];
    Ray incomingRay;
    incomingRay.origin = queuedRay.originMinT.xyz;
    incomingRay.direction = queuedRay.directionMaxT.xyz;

    if (packedHit.hit == 0u) {
        if (uniforms.debugViewMode != kDebugViewNone) {
            state.radiance.xyz = float3(0.0f);
            state.alive = 0u;
            wavefront_note_termination(queueCounters, stats, 3u);
            state.rngState = rngState;
            pathStates[pathIndex] = state;
            return;
        }
        float3 background = sky_color(incomingRay.direction);
        if (uniforms.backgroundMode == 1u) {
            background = uniforms.backgroundColor;
        } else if (uniforms.backgroundMode == 2u &&
                   environmentTexture.get_width() > 0 &&
                   environmentTexture.get_height() > 0) {
            float overrideLod = 0.0f;
            bool useOverride = environment_mip_override(uniforms, environmentTexture, overrideLod);
            bool envLodActive = (state.flags & kWavefrontFlagEnvLodActive) != 0u;
            if (useOverride) {
                background = environment_color_lod(environmentTexture,
                                                   incomingRay.direction,
                                                   uniforms.environmentRotation,
                                                   uniforms.environmentIntensity,
                                                   overrideLod,
                                                   uniforms);
            } else if (envLodActive) {
                background = environment_color_lod(environmentTexture,
                                                   incomingRay.direction,
                                                   uniforms.environmentRotation,
                                                   uniforms.environmentIntensity,
                                                   state.envLod,
                                                   uniforms);
            } else {
                background = environment_color(environmentTexture,
                                               incomingRay.direction,
                                               uniforms.environmentRotation,
                                               uniforms.environmentIntensity,
                                               uniforms);
            }
        }
        if (uniforms.backgroundMode != 2u) {
            background = to_working_space(background, uniforms);
        }
        float misWeight = 1.0f;
        bool useSpecularMis =
            use_visible_emitter_mis(queuedRay.depth,
                                    (state.flags & kWavefrontFlagLastScatterDelta) != 0u,
                                    uniforms);
        if (useSpecularMis && environment_sampling_available(uniforms,
                                                             environmentConditionalAlias,
                                                             environmentMarginalAlias,
                                                             environmentPdf)) {
            float lightPdf = environment_pdf(uniforms, environmentPdf, incomingRay.direction);
            float denom = state.lastBsdfPdf + lightPdf;
            if (denom > 0.0f) {
                misWeight = clamp(state.lastBsdfPdf / denom,
                                  kMisWeightClampMin,
                                  kMisWeightClampMax);
            }
        }
        state.radiance.xyz +=
            clamp_firefly_contribution(state.throughput.xyz,
                                       background * misWeight,
                                       clampParams,
                                       stats);
        state.alive = 0u;
        wavefront_note_termination(queueCounters, stats, 1u);
        if (debugCtxPtr) {
            record_debug_event(*debugCtxPtr,
                               kDebugEventBackground,
                               queuedRay.depth,
                               0u,
                               0u,
                               0,
                               0u,
                               0u,
                               false,
                               state.throughput.xyz,
                               float4(0.0f),
                               background);
        }
        state.rngState = rngState;
        pathStates[pathIndex] = state;
        return;
    }

    HitRecord rec = unpack_wavefront_hit(packedHit);
    if (!all(isfinite(rec.shadingNormal)) || dot(rec.shadingNormal, rec.shadingNormal) <= 0.0f) {
        rec.shadingNormal = rec.normal;
    }
    rec.shadingNormal = safe_normalize(rec.shadingNormal);
    if (volume_transport_enabled(uniforms)) {
        MediumDescriptor medium = volume_make_global_medium(uniforms);
        const float segment = max(rec.t, 0.0f);
        const float3 segmentEmission = volume_emission_integral(medium, segment);
        if (any(segmentEmission > float3(0.0f))) {
            state.radiance.xyz +=
                clamp_firefly_contribution(state.throughput.xyz, segmentEmission, clampParams, stats);
        }
        const uint maxVolumeEvents = max(medium.metadata.y, 1u);
        if (state.volumeScatteringDepth < maxVolumeEvents &&
            any(medium.sigmaSAnisotropy.xyz > float3(0.0f))) {
            float distancePdf = 1.0f;
            const float sampledDistance = volume_sample_distance(medium, rngState, distancePdf);
            if (sampledDistance > 0.0f && sampledDistance < segment) {
                const float3 scatterPoint = incomingRay.origin + incomingRay.direction * sampledDistance;
                const float3 transmittance = volume_transmittance(medium, sampledDistance, stats);
                const float majorant = max(volume_majorant(medium), 1.0e-6f);
                state.throughput.xyz *= transmittance * (medium.sigmaSAnisotropy.xyz / majorant);
                if (rectangles && materials && medium.metadata.z != 0u) {
                    const uint rectLightCount = count_rect_lights(uniforms, rectangles, materials);
                    if (rectLightCount > 0u) {
                        HitRecord volumeRec = rec;
                        volumeRec.point = scatterPoint;
                        RectLightSample lightSample;
                        if (sample_rect_light(uniforms,
                                              rectangles,
                                              materials,
                                              environmentTexture,
                                              volumeRec,
                                              rngState,
                                              rectLightCount,
                                              lightSample) &&
                            lightSample.pdf > 0.0f) {
                            const float phase = volume_phase_eval(medium,
                                                                  -incomingRay.direction,
                                                                  lightSample.direction);
                            const float shadowMax = max(lightSample.distance - kEpsilon, kEpsilon);
                            const float3 shadowTr = volume_transmittance(medium, shadowMax, stats);
                            const float3 contribution =
                                lightSample.emission * phase * shadowTr / max(lightSample.pdf, 1.0e-6f);
                            uint shadowIndex = wavefront_reserve_shadow_slot(uniforms,
                                                                             pathIndex,
                                                                             state,
                                                                             queueCounters,
                                                                             stats);
                            if (shadowIndex != kInvalidIndex) {
                                ShadowRay shadow{};
                                shadow.originTMax = float4(scatterPoint + lightSample.direction * kRayOriginEpsilon,
                                                           shadowMax);
                                shadow.direction = float4(lightSample.direction, 0.0f);
                                shadow.contribution = float4(contribution, 0.0f);
                                shadow.throughput = float4(state.throughput.xyz, 0.0f);
                                shadow.pathIndex = pathIndex;
                                shadow.lightId = lightSample.rectIndex;
                                shadow.flags = 0u;
                                shadow.excludeMeshIndex = kInvalidIndex;
                                shadow.excludePrimitiveIndex = kInvalidIndex;
                                shadowRays[shadowIndex] = shadow;
                                if (stats) {
                                    atomic_fetch_add_explicit(&stats->volumeNeeRayCount, 1u, memory_order_relaxed);
                                }
                            }
                        }
                    }
                }
                float phasePdf = 1.0f;
                const float3 phaseDir = volume_phase_sample(medium,
                                                            -incomingRay.direction,
                                                            rngState,
                                                            phasePdf);
                if (stats) {
                    atomic_fetch_add_explicit(&stats->volumeScatterEventCount, 1u, memory_order_relaxed);
                    atomic_fetch_add_explicit(&stats->volumePhaseSampleCount, 1u, memory_order_relaxed);
                }
                uint nextIndex =
                    atomic_fetch_add_explicit(&queueCounters->nextRayCount, 1u, memory_order_relaxed);
                if (stats) {
                    atomic_fetch_add_explicit(&stats->wavefrontNextRayEnqueueCount, 1u, memory_order_relaxed);
                }
                if (nextIndex < max(uniforms.width * uniforms.height, 1u)) {
                    wavefront_note_queue_enqueue(queueCounters->diffuseRayQueue);
                    WavefrontRay nextRay{};
                    nextRay.originMinT = float4(scatterPoint + phaseDir * kRayOriginEpsilon, kEpsilon);
                    nextRay.directionMaxT = float4(phaseDir, kInfinity);
                    wavefront_store_primary_ray_diff(nextRay, wavefront_primary_ray_diff(queuedRay));
                    wavefront_store_ray_cone(nextRay, wavefront_ray_cone(queuedRay));
                    nextRay.pixelIndex = queuedRay.pixelIndex;
                    nextRay.pathIndex = pathIndex;
                    nextRay.depth = queuedRay.depth + 1u;
                    nextRay.flags = 0u;
                    nextRay.excludeMeshIndex = kInvalidIndex;
                    nextRay.excludePrimitiveIndex = kInvalidIndex;
                    nextRays[nextIndex] = nextRay;
                    state.lastBsdfPdf = max(phasePdf, 1.0e-6f);
                    state.flags = 0u;
                    state.currentMediumId = medium.metadata.x;
                    state.volumeScatteringDepth += 1u;
                    state.alive = 1u;
                } else {
                    atomic_store_explicit(&queueCounters->overflowFlag, 1u, memory_order_relaxed);
                    wavefront_note_queue_overflow(queueCounters->diffuseRayQueue);
                    state.alive = 0u;
                    wavefront_note_termination(queueCounters, stats, 4u);
                }
                state.rngState = rngState;
                pathStates[pathIndex] = state;
                return;
            }
        } else if (any(medium.sigmaSAnisotropy.xyz > float3(0.0f)) && stats) {
            atomic_fetch_add_explicit(&stats->volumeFallbackCount, 1u, memory_order_relaxed);
        }
        state.throughput.xyz *= volume_transmittance(medium, segment, stats);
        state.currentMediumId = medium.metadata.x;
    }
    if (mediumDepth > 0u) {
        float3 sigma = mediumSigmaStack[mediumDepth - 1u];
        if (any(sigma > float3(0.0f))) {
            float segment = max(rec.t, 0.0f);
            state.throughput.xyz *= exp(-sigma * segment);
        }
    }
    float3 incidentDir = safe_normalize(incomingRay.direction);
    float3 wo = safe_normalize(-incomingRay.direction);
    uint materialIndex = rec.materialIndex;
    if (uniforms.materialCount > 0u) {
        materialIndex = min(materialIndex, uniforms.materialCount - 1u);
    } else {
        materialIndex = 0u;
    }
    MaterialData baseMaterial = materials ? materials[materialIndex] : MaterialData{};
    const bool explicitRestirDiLightingStage =
        uniforms.explicitRestirDiLightingStage != 0u &&
        (uniforms.directLightMode == kDirectLightModeRestirDi ||
         uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid);
    const bool explicitRestirDiSurfaceAccepted =
        explicitRestirDiLightingStage &&
        state.reserved1 == kWavefrontExplicitRestirDiSurfaceAccepted;
    RayCone rayCone = wavefront_ray_cone(queuedRay);
    if (!(rayCone.width > 0.0f) && !(rayCone.spread > 0.0f)) {
        rayCone = make_primary_ray_cone(uniforms);
    }
    PrimaryRayDiff primaryRayDiff = wavefront_primary_ray_diff(queuedRay);
    WavefrontSurfaceState surfaceState = resolve_wavefront_surface_state(
        uniforms,
        rec,
        baseMaterial,
        materialIndex,
        queuedRay.depth,
        incomingRay,
        primaryRayDiff,
        incomingRay.direction,
        ray_segment_world_length(incomingRay, rec.t),
        rayCone,
        rngState,
        explicitRestirDiSurfaceAccepted,
        materialResources.textures,
        materialResources.samplers,
        materialTextureInfos,
        meshInfos,
        sceneVertices,
        meshIndices);
    if (surfaceState.alphaDiscarded) {
        uint nextIndex =
            atomic_fetch_add_explicit(&queueCounters->nextRayCount, 1u, memory_order_relaxed);
        if (stats) {
            atomic_fetch_add_explicit(&stats->wavefrontNextRayEnqueueCount, 1u, memory_order_relaxed);
        }
        if (nextIndex < max(uniforms.width * uniforms.height, 1u)) {
            wavefront_note_queue_enqueue(queueCounters->diffuseRayQueue);
            WavefrontRay nextRay{};
            nextRay.originMinT = float4(offset_ray_origin(rec, incomingRay.direction), kEpsilon);
            nextRay.directionMaxT = float4(incomingRay.direction, kInfinity);
            wavefront_store_primary_ray_diff(nextRay, wavefront_primary_ray_diff(queuedRay));
            wavefront_store_ray_cone(nextRay, wavefront_ray_cone(queuedRay));
            nextRay.pixelIndex = queuedRay.pixelIndex;
            nextRay.pathIndex = pathIndex;
            nextRay.depth = queuedRay.depth;
            nextRay.flags = queuedRay.flags;
            compute_exclusion_indices(rec, nextRay.excludeMeshIndex, nextRay.excludePrimitiveIndex);
            nextRays[nextIndex] = nextRay;
            state.alive = 1u;
            state.rngState = rngState;
            pathStates[pathIndex] = state;
        } else {
            atomic_store_explicit(&queueCounters->overflowFlag, 1u, memory_order_relaxed);
            wavefront_note_queue_overflow(queueCounters->diffuseRayQueue);
            state.alive = 0u;
            wavefront_note_termination(queueCounters, stats, 4u);
            state.rngState = rngState;
            pathStates[pathIndex] = state;
        }
        return;
    }
    rec = surfaceState.rec;
    MaterialData material = surfaceState.material;
    SpectralPathState spectralState;
    spectralState.enabled = spectral_rendering_enabled(uniforms) && state.spectralWavelengthNm > 0.0f;
    spectralState.wavelengthNm = state.spectralWavelengthNm;
    spectralState.sampleIndex = state.spectralSampleIndex;
    spectralState.sensorRgb = spectral_sensor_rgb(state.spectralWavelengthNm);
    spectral_apply_material(material, uniforms, spectralState, stats);
    const uint type = static_cast<uint>(material.typeEta.x);
    const float diffuseOcclusion = surfaceState.diffuseOcclusion;
    const float3 shadingNormal = rec.shadingNormal;
    const float3 debugBaseColor = surfaceState.debugBaseColor;
    const float debugMetallic = surfaceState.debugMetallic;
    const float debugRoughness = surfaceState.debugRoughness;
    const float debugAO = surfaceState.debugAO;
    const float transmissionValue = material.pbrExtras.z;
    const uint rectLightCount = (rectangles && materials && uniforms.rectangleCount > 0u)
        ? count_rect_lights(uniforms, rectangles, materials)
        : 0u;
    const bool specularOnly = (uniforms.debugSpecularOnly != 0u);

    if (state.firstHitAlbedo.w <= 0.0f) {
        state.firstHitAlbedo = float4(material_closure_aov_albedo(material), 1.0f);
        state.firstHitNormal = float4(surfaceState.firstHitNormal, 1.0f);
        state.firstHitPosition = float4(surfaceState.rec.point, 1.0f);
        state.firstHitMaterial = material_closure_aov_features(material);
    }

    if (debugCtxPtr && surfaceState.debugHasNormalMap) {
        record_debug_event(*debugCtxPtr,
                           kDebugEventTbnBasis0,
                           queuedRay.depth,
                           0u,
                           0u,
                           0,
                           rec.frontFace,
                           materialIndex,
                           false,
                           surfaceState.debugVtxNormalRaw,
                           float4(surfaceState.debugNormalUv.x,
                                  surfaceState.debugNormalUv.y,
                                  surfaceState.debugTangentW,
                                  rec.t),
                           surfaceState.debugVtxNormal);
        record_debug_event(*debugCtxPtr,
                           kDebugEventTbnBasis1,
                           queuedRay.depth,
                           0u,
                           0u,
                           0,
                           rec.frontFace,
                           materialIndex,
                           false,
                           surfaceState.debugTangentRaw,
                           float4(surfaceState.debugTangent, 0.0f),
                           surfaceState.debugBitangent);
        record_debug_event(*debugCtxPtr,
                           kDebugEventTbnBasis2,
                           queuedRay.depth,
                           0u,
                           0u,
                           0,
                           rec.frontFace,
                           materialIndex,
                           false,
                           surfaceState.debugTexelRaw,
                           float4(surfaceState.debugTexelDecoded, 0.0f),
                           surfaceState.debugTexelDecoded);
        record_debug_event(*debugCtxPtr,
                           kDebugEventTbnBasis3,
                           queuedRay.depth,
                           0u,
                           0u,
                           0,
                           rec.frontFace,
                           materialIndex,
                           false,
                           float3(0.0f),
                           float4(0.0f),
                           surfaceState.debugSnWorld);
    }

    if (debugCtxPtr) {
        record_debug_event(*debugCtxPtr,
                           kDebugEventBsdfMaterial0,
                           queuedRay.depth,
                           0u,
                           0u,
                           0,
                           rec.frontFace,
                           materialIndex,
                           false,
                           debugBaseColor,
                           float4(surfaceState.debugNormalUv.x,
                                  surfaceState.debugNormalUv.y,
                                  debugMetallic,
                                  debugRoughness),
                           float3(material.pbrParams.w, transmissionValue, 0.0f));
    }

    if (uniforms.debugViewMode != kDebugViewNone) {
        const float3 debugColor =
            restir_debug_view_color(uniforms,
                                    rec,
                                    material,
                                    pixel,
                                    queuedRay.depth,
                                    materialResources.pathGuidingStates,
                                    materialResources.restirPtReservoirs,
                                    materialResources.radianceCacheStates,
                                    debugBaseColor,
                                    debugMetallic,
                                    debugRoughness,
                                    debugAO);
        state.radiance.xyz = debugColor;
        state.alive = 0u;
        wavefront_note_termination(queueCounters, stats, 3u);
        state.rngState = rngState;
        pathStates[pathIndex] = state;
        return;
    }

    if (!specularOnly &&
        type == 7u &&
        any(material.emission.xyz != float3(0.0f)) &&
        (rec.frontFace != 0u || rec.twoSided != 0u)) {
        float3 visibleEmission = camera_visible_emission(material, queuedRay.depth);
        state.radiance.xyz += clamp_firefly_contribution(state.throughput.xyz,
                                                         visibleEmission,
                                                         clampParams,
                                                         stats);
        if (debugCtxPtr) {
            record_debug_event(*debugCtxPtr,
                               kDebugEventVisibleEmitter,
                               queuedRay.depth,
                               0u,
                               0u,
                               0,
                               rec.frontFace,
                               materialIndex,
                               false,
                               state.throughput.xyz,
                               float4(rec.t, 0.0f, 0.0f, 0.0f),
                               visibleEmission);
        }
    }

    if (type == 3u) {
        float3 emission = material.emission.xyz;
        if (material.emission.w > 0.0f &&
            environmentTexture.get_width() > 0 &&
            environmentTexture.get_height() > 0 &&
            rec.frontFace != 0u) {
            float3 sampleDir = -rec.shadingNormal;
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
            bool useSpecularMis =
                use_visible_emitter_mis(queuedRay.depth,
                                        (state.flags & kWavefrontFlagLastScatterDelta) != 0u,
                                        uniforms);
            if (useSpecularMis && rectLightCount > 0u) {
                float lightPdf = rect_light_pdf_for_hit(uniforms,
                                                        rectangles,
                                                        materials,
                                                        rectLightCount,
                                                        rec,
                                                        incomingRay.origin);
                float denom = state.lastBsdfPdf + lightPdf;
                if (denom > 0.0f) {
                    misWeight = clamp(state.lastBsdfPdf / denom,
                                      kMisWeightClampMin,
                                      kMisWeightClampMax);
                }
            }
            state.radiance.xyz += clamp_firefly_contribution(state.throughput.xyz,
                                                             emission * misWeight,
                                                             clampParams,
                                                             stats);
            state.alive = 0u;
            wavefront_note_termination(queueCounters, stats, 2u);
            state.rngState = rngState;
            pathStates[pathIndex] = state;
            return;
        }
    }

    const bool envSampling = environment_sampling_available(uniforms,
                                                            environmentConditionalAlias,
                                                            environmentMarginalAlias,
                                                            environmentPdf);
    const bool useEmissiveDirect =
        uniforms.directLightMode != kDirectLightModeLegacyRect &&
        uniforms.emissivePrimitiveCount > 0u &&
        emissivePrimitives;
    const bool useFusedEmissiveDirect =
        useEmissiveDirect && !explicitRestirDiLightingStage;
    bool didSampleRect = false;
    bool didSampleEnv = false;

    if (!material_is_delta(material) &&
        useFusedEmissiveDirect &&
        uniforms.directLightMode == kDirectLightModeBaselineEmissive) {
        RISSamplePayload lightSample;
        if (sample_direct_light_baseline(uniforms,
                                         emissivePrimitives,
                                         rec,
                                         rngState,
                                         lightSample)) {
            const bool lightSampleDirectional = ris_payload_is_directional(lightSample);
            float distSq = ris_payload_distance_sq(rec, lightSample);
            float distance = lightSampleDirectional ? kInfinity : length(lightSample.position - rec.point);
            float3 omegaL = ris_payload_omega_l(rec, lightSample);
            float cosThetaSurface = saturate(dot(shadingNormal, omegaL));
            float cosThetaLight = lightSampleDirectional ? 1.0f : saturate(dot(lightSample.normal, -omegaL));
            if (lightSample.pdf > 0.0f &&
                distSq > 0.0f &&
                cosThetaSurface > 0.0f &&
                cosThetaLight > 0.0f) {
                BsdfEvalResult eval = evaluate_bsdf(material,
                                                    rec.point,
                                                    shadingNormal,
                                                    wo,
                                                    omegaL,
                                                    clampParams,
                                                    uniforms.sssMode,
                                                    diffuseOcclusion,
                                                    specularOnly);
                if (!eval.isDelta && !eval.isBssrdf) {
                    float3 contribution = eval.value * lightSample.emission;
                    contribution *= cosThetaSurface / max(lightSample.pdf, 1.0e-6f);
                    if (all(isfinite(contribution))) {
                        contribution = clamp_firefly_contribution(state.throughput.xyz,
                                                                  contribution,
                                                                  clampParams,
                                                                  stats);
                        uint shadowIndex = wavefront_reserve_shadow_slot(uniforms,
                                                                         pathIndex,
                                                                         state,
                                                                         queueCounters,
                                                                         stats);
                        if (shadowIndex != kInvalidIndex) {
                            uint shadowExcludeMesh;
                            uint shadowExcludePrim;
                            compute_exclusion_indices(rec, shadowExcludeMesh, shadowExcludePrim);
                            ShadowRay shadow{};
                            shadow.originTMax = float4(offset_ray_origin(rec, omegaL),
                                                       lightSampleDirectional
                                                           ? kInfinity
                                                           : max(distance - kEpsilon, kEpsilon));
                            shadow.direction = float4(omegaL, 0.0f);
                            shadow.contribution = float4(contribution, 0.0f);
                            shadow.throughput = float4(state.throughput.xyz, 0.0f);
                            shadow.pathIndex = pathIndex;
                            shadow.lightId = lightSample.primitiveIndex;
                            shadow.flags = lightSampleDirectional ? 0u : kWavefrontShadowFlagEmissivePrimitive;
                            shadow.excludeMeshIndex = shadowExcludeMesh;
                            shadow.excludePrimitiveIndex = shadowExcludePrim;
                            shadowRays[shadowIndex] = shadow;
                        }
                    }
                }
            }
        }
    } else if (!material_is_delta(material) && useFusedEmissiveDirect) {
        Reservoir reservoir = make_empty_reservoir();
        if (uniforms.directLightMode == kDirectLightModeRis ||
            uniforms.directLightMode == kDirectLightModeRisSpatialReuse ||
            uniforms.directLightMode == kDirectLightModeRisTemporalReuse ||
            uniforms.directLightMode == kDirectLightModeRisWorldReuse ||
            uniforms.directLightMode == kDirectLightModeRisRegirCache ||
            uniforms.directLightMode == kDirectLightModeRestirDi ||
            uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) {
            uint risTargetM = max(uniforms.risCandidateCount, 1u);
            if ((uniforms.directLightMode == kDirectLightModeRisRegirCache ||
                 uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
                uniforms.frameIndex > 0u) {
                risTargetM *= 2u;
            }
            build_ris_reservoir_for_hit(uniforms,
                                        emissivePrimitives,
                                        rec,
                                        material,
                                        shadingNormal,
                                        wo,
                                        clampParams,
                                        diffuseOcclusion,
                                        specularOnly,
                                        risTargetM,
                                        rngState,
                                        reservoir);

            if ((uniforms.directLightMode == kDirectLightModeRisTemporalReuse ||
                 uniforms.directLightMode == kDirectLightModeRestirDi ||
                 uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
                reservoir.valid &&
                queuedRay.depth == 0u &&
                rec.t > 0.0f &&
                uniforms.frameIndex > 0u) {
                uint temporalState = pcg_hash(uniforms.fixedRngSeed ^
                                              (uint(pixel.x) * 1664525u) ^
                                              (uint(pixel.y) * 1013904223u) ^
                                              ((uniforms.frameIndex - 1u) * 9781u) ^
                                              0x9E3779B9u);
                Reservoir previousReservoir = make_empty_reservoir();
                if (build_ris_reservoir_for_hit(uniforms,
                                                emissivePrimitives,
                                                rec,
                                                material,
                                                shadingNormal,
                                                wo,
                                                clampParams,
                                                diffuseOcclusion,
                                                specularOnly,
                                                risTargetM,
                                                temporalState,
                                                previousReservoir) &&
                    previousReservoir.valid) {
                    float phatCurrent = reservoir_winner_phat_for_hit(uniforms,
                                                                      rec,
                                                                      material,
                                                                      shadingNormal,
                                                                      wo,
                                                                      clampParams,
                                                                      diffuseOcclusion,
                                                                      specularOnly,
                                                                      previousReservoir.winner);
                    if (phatCurrent > 0.0f && isfinite(phatCurrent) &&
                        previousReservoir.wSum > 0.0f &&
                        isfinite(previousReservoir.wSum)) {
                        merge_reservoir_winner(reservoir,
                                               previousReservoir.winner,
                                               previousReservoir.wSum,
                                               previousReservoir.M,
                                               rand_uniform(rngState));
                    }
                }
            }

            if (uniforms.directLightMode == kDirectLightModeRisWorldReuse &&
                reservoir.valid &&
                queuedRay.depth == 0u &&
                rec.t > 0.0f &&
                uniforms.width > 0u &&
                uniforms.height > 0u) {
                int3 worldCell = world_reuse_cell(rec.point, uniforms.worldReuseCellSize);
                constexpr int2 offsets[4] = {
                    int2(-2, 0),
                    int2(2, 0),
                    int2(0, -2),
                    int2(0, 2)
                };
                for (uint n = 0u; n < 4u; ++n) {
                    int2 candidateCoordI = int2(pixel) + offsets[n];
                    if (candidateCoordI.x < 0 ||
                        candidateCoordI.y < 0 ||
                        candidateCoordI.x >= int(uniforms.width) ||
                        candidateCoordI.y >= int(uniforms.height)) {
                        continue;
                    }

                    uint2 candidateCoord = uint2(candidateCoordI);
                    Ray candidateRay = make_center_primary_ray_for_pixel(uniforms, candidateCoord);
                    HitRecord candidateRec = make_empty_hit_record();
                    bool candidateHit = trace_scene_software(uniforms,
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
                                                              candidateRay,
                                                              kEpsilon,
                                                              kInfinity,
                                                              /*anyHitOnly=*/false,
                                                              /*includeTriangles=*/true,
                                                              candidateRec);
                    if (!candidateHit || candidateRec.materialIndex >= uniforms.materialCount) {
                        continue;
                    }

                    int3 candidateCell =
                        world_reuse_cell(candidateRec.point, uniforms.worldReuseCellSize);
                    if (!world_reuse_cell_compatible(worldCell, candidateCell)) {
                        continue;
                    }

                    float depthRel = fabs(rec.t - candidateRec.t) / max(rec.t, kEpsilon);
                    if (!(depthRel <= 0.2f) || !isfinite(depthRel)) {
                        continue;
                    }
                    float normalDot = dot(safe_normalize(shadingNormal),
                                          safe_normalize(candidateRec.shadingNormal));
                    if (!(normalDot >= 0.85f) || !isfinite(normalDot)) {
                        continue;
                    }

                    MaterialData candidateMaterial = materials[candidateRec.materialIndex];
                    float3 candidateWo = safe_normalize(-candidateRay.direction);
                    uint candidateState =
                        pcg_hash(rngState ^
                                 (uint(candidateCell.x) * 1664525u) ^
                                 (uint(candidateCell.y) * 1013904223u) ^
                                 (uint(candidateCell.z) * 747796405u) ^
                                 ((n + 1u) * 2891336453u));
                    Reservoir candidateReservoir = make_empty_reservoir();
                    if (!build_ris_reservoir_for_hit(uniforms,
                                                     emissivePrimitives,
                                                     candidateRec,
                                                     candidateMaterial,
                                                     candidateRec.shadingNormal,
                                                     candidateWo,
                                                     clampParams,
                                                     diffuseOcclusion,
                                                     specularOnly,
                                                     risTargetM,
                                                     candidateState,
                                                     candidateReservoir) ||
                        !candidateReservoir.valid) {
                        continue;
                    }

                    float phatCurrent = reservoir_winner_phat_for_hit(uniforms,
                                                                      rec,
                                                                      material,
                                                                      shadingNormal,
                                                                      wo,
                                                                      clampParams,
                                                                      diffuseOcclusion,
                                                                      specularOnly,
                                                                      candidateReservoir.winner);
                    float phatCandidate = reservoir_winner_phat_for_hit(uniforms,
                                                                        candidateRec,
                                                                        candidateMaterial,
                                                                        candidateRec.shadingNormal,
                                                                        candidateWo,
                                                                        clampParams,
                                                                        diffuseOcclusion,
                                                                        specularOnly,
                                                                        candidateReservoir.winner);
                    float denom = phatCurrent * float(max(reservoir.M, 1u)) +
                                  phatCandidate * float(max(candidateReservoir.M, 1u));
                    if (!(denom > 0.0f) || !isfinite(denom) || !(phatCurrent > 0.0f)) {
                        continue;
                    }

                    float mis = phatCurrent / denom;
                    float mergeWeight = mis * candidateReservoir.wSum;
                    if (!(mergeWeight > 0.0f) || !isfinite(mergeWeight)) {
                        continue;
                    }

                    update_reservoir(reservoir,
                                     candidateReservoir.winner,
                                     mergeWeight,
                                     rand_uniform(rngState));
                }
            }

            if ((uniforms.directLightMode == kDirectLightModeRisRegirCache ||
                 uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
                reservoir.valid &&
                queuedRay.depth == 0u &&
                rec.t > 0.0f &&
                uniforms.frameIndex > 0u) {
                int3 cacheCell = world_reuse_cell(rec.point, uniforms.worldReuseCellSize);
                uint cacheSourceFrameIndex =
                    uniforms.frameIndex > 1u ? uniforms.frameIndex - 2u : uniforms.frameIndex - 1u;
                for (uint cacheProbe = 0u; cacheProbe < 2u; ++cacheProbe) {
                    if (cacheProbe > cacheSourceFrameIndex) {
                        continue;
                    }
                    uint candidateSourceFrameIndex = cacheSourceFrameIndex - cacheProbe;
                    uint cacheState =
                        pcg_hash(uniforms.fixedRngSeed ^
                                 (uint(cacheCell.x) * 1664525u) ^
                                 (uint(cacheCell.y) * 1013904223u) ^
                                 (uint(cacheCell.z) * 747796405u) ^
                                 (candidateSourceFrameIndex * 9781u) ^
                                 (cacheProbe * 0x9E3779B9u) ^
                                 0xD1B54A35u);
                    Reservoir cachedReservoir = make_empty_reservoir();
                    if (!build_ris_reservoir_for_hit(uniforms,
                                                     emissivePrimitives,
                                                     rec,
                                                     material,
                                                     shadingNormal,
                                                     wo,
                                                     clampParams,
                                                     diffuseOcclusion,
                                                     specularOnly,
                                                     risTargetM,
                                                     cacheState,
                                                     cachedReservoir) ||
                        !cachedReservoir.valid) {
                        continue;
                    }
                    float phatCurrent = reservoir_winner_phat_for_hit(uniforms,
                                                                      rec,
                                                                      material,
                                                                      shadingNormal,
                                                                      wo,
                                                                      clampParams,
                                                                      diffuseOcclusion,
                                                                      specularOnly,
                                                                      cachedReservoir.winner);
                    float phatCached = reservoir_winner_phat_for_hit(uniforms,
                                                                     rec,
                                                                     material,
                                                                     shadingNormal,
                                                                     wo,
                                                                     clampParams,
                                                                     diffuseOcclusion,
                                                                     specularOnly,
                                                                     cachedReservoir.winner);
                    float denom = phatCurrent * float(max(reservoir.M, 1u)) +
                                  phatCached * float(max(cachedReservoir.M, 1u));
                    if (denom > 0.0f && isfinite(denom) && phatCurrent > 0.0f) {
                        float mis = phatCurrent / denom;
                        float mergeWeight = mis * cachedReservoir.wSum;
                        if (mergeWeight > 0.0f && isfinite(mergeWeight)) {
                            update_reservoir(reservoir,
                                             cachedReservoir.winner,
                                             mergeWeight,
                                             rand_uniform(rngState));
                        }
                    }
                }
            }
        }

        if (reservoir.valid && reservoir.M > 0u && reservoir.wSum > 0.0f) {
            RISSamplePayload winner = reservoir.winner;
            float distSq = ris_payload_distance_sq(rec, winner);
            if (distSq > 0.0f && isfinite(distSq)) {
                const bool winnerDirectional = ris_payload_is_directional(winner);
                float distance = winnerDirectional ? kInfinity : length(winner.position - rec.point);
                float3 omegaL = ris_payload_omega_l(rec, winner);
                float cosThetaSurface = saturate(dot(shadingNormal, omegaL));
                float cosThetaLight = winnerDirectional ? 1.0f : saturate(dot(winner.normal, -omegaL));
                BsdfEvalResult eval = evaluate_bsdf(material,
                                                    rec.point,
                                                    shadingNormal,
                                                    wo,
                                                    omegaL,
                                                    clampParams,
                                                    uniforms.sssMode,
                                                    diffuseOcclusion,
                                                    specularOnly);
                float phatW = p_hat(winner.emission,
                                    eval.value,
                                    cosThetaSurface,
                                    cosThetaLight,
                                    distSq);
                reservoir.W = (phatW > 0.0f)
                            ? ((reservoir.wSum / float(reservoir.M)) / phatW)
                            : 0.0f;
                if (!eval.isDelta &&
                    !eval.isBssrdf &&
                    reservoir.W > 0.0f &&
                    cosThetaSurface > 0.0f &&
                    all(isfinite(eval.value))) {
                    float3 contribution = eval.value * winner.emission;
                    contribution *= cosThetaSurface * reservoir.W;
                    if (all(isfinite(contribution))) {
                        contribution = clamp_firefly_contribution(state.throughput.xyz,
                                                                  contribution,
                                                                  clampParams,
                                                                  stats);
                        uint shadowIndex = wavefront_reserve_shadow_slot(uniforms,
                                                                         pathIndex,
                                                                         state,
                                                                         queueCounters,
                                                                         stats);
                        if (shadowIndex != kInvalidIndex) {
                            uint shadowExcludeMesh;
                            uint shadowExcludePrim;
                            compute_exclusion_indices(rec, shadowExcludeMesh, shadowExcludePrim);
                            ShadowRay shadow{};
                            shadow.originTMax = float4(offset_ray_origin(rec, omegaL),
                                                       winnerDirectional
                                                           ? kInfinity
                                                           : max(distance - kRayOriginEpsilon * 4.0f,
                                                                 kRayOriginEpsilon));
                            shadow.direction = float4(omegaL, 0.0f);
                            shadow.contribution = float4(contribution, 0.0f);
                            shadow.throughput = float4(state.throughput.xyz, 0.0f);
                            shadow.pathIndex = pathIndex;
                            shadow.lightId = winner.primitiveIndex;
                            shadow.flags = winnerDirectional ? 0u : kWavefrontShadowFlagEmissivePrimitive;
                            shadow.excludeMeshIndex = shadowExcludeMesh;
                            shadow.excludePrimitiveIndex = shadowExcludePrim;
                            shadowRays[shadowIndex] = shadow;
                        }
                    }
                }
            }
        }
    } else if (!material_is_delta(material) && !useEmissiveDirect) {
        RectLightSample lightSample;
        if (sample_rect_light(uniforms,
                              rectangles,
                              materials,
                              environmentTexture,
                              rec,
                              rngState,
                              rectLightCount,
                              lightSample)) {
            didSampleRect = true;
            BsdfEvalResult eval = evaluate_bsdf(material,
                                                rec.point,
                                                rec.shadingNormal,
                                                wo,
                                                lightSample.direction,
                                                clampParams,
                                                uniforms.sssMode,
                                                diffuseOcclusion,
                                                specularOnly);
            if (!eval.isDelta &&
                !eval.isBssrdf &&
                eval.pdf > 0.0f &&
                all(isfinite(eval.value)) &&
                any(eval.value > float3(0.0f))) {
                const float3 contribution = clamp_firefly_contribution(
                    state.throughput.xyz,
                    eval.value * lightSample.emission / max(lightSample.pdf, 1.0e-6f),
                    clampParams,
                    stats);
                uint shadowIndex = wavefront_reserve_shadow_slot(uniforms,
                                                                 pathIndex,
                                                                 state,
                                                                 queueCounters,
                                                                 stats);
                if (shadowIndex != kInvalidIndex) {
                    uint shadowExcludeMesh;
                    uint shadowExcludePrim;
                    compute_exclusion_indices(rec, shadowExcludeMesh, shadowExcludePrim);
                    ShadowRay shadow{};
                    shadow.originTMax = float4(offset_ray_origin(rec, lightSample.direction),
                                               max(lightSample.distance - kRayOriginEpsilon * 4.0f,
                                                   kRayOriginEpsilon));
                    shadow.direction = float4(lightSample.direction, 0.0f);
                    shadow.contribution = float4(contribution, 0.0f);
                    shadow.throughput = float4(state.throughput.xyz, 0.0f);
                    shadow.pathIndex = pathIndex;
                    shadow.lightId = lightSample.rectIndex;
                    shadow.flags = 0u;
                    shadow.excludeMeshIndex = shadowExcludeMesh;
                    shadow.excludePrimitiveIndex = shadowExcludePrim;
                    shadowRays[shadowIndex] = shadow;
                }
            }
        }
    }

    if (!material_is_delta(material) && envSampling) {
        EnvironmentSample envSample;
        if (sample_environment(uniforms,
                               environmentTexture,
                               environmentConditionalAlias,
                               environmentMarginalAlias,
                               environmentPdf,
                               rngState,
                               envSample)) {
            didSampleEnv = true;
            float overrideLod = 0.0f;
            bool useOverride = environment_mip_override(uniforms, environmentTexture, overrideLod);
            if (environmentTexture.get_num_mip_levels() > 1u) {
                float envRoughness = environment_lighting_roughness(material);
                if (envRoughness < 0.95f) {
                    float envLod = environment_lod_from_roughness(envRoughness, environmentTexture);
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
                BsdfEvalResult eval = evaluate_bsdf(material,
                                                    rec.point,
                                                    shadingNormal,
                                                    wo,
                                                    envSample.direction,
                                                    clampParams,
                                                    uniforms.sssMode,
                                                    diffuseOcclusion,
                                                    specularOnly);
                if (!eval.isDelta &&
                    !eval.isBssrdf &&
                    eval.pdf > 0.0f &&
                    all(isfinite(eval.value)) &&
                    any(eval.value > float3(0.0f))) {
                    float weight = 1.0f;
                    float denom = envSample.pdf + eval.pdf;
                    if (denom > 0.0f) {
                        weight = clamp(envSample.pdf / denom,
                                       kMisWeightClampMin,
                                       kMisWeightClampMax);
                    }
                    float3 contributionFactor = eval.value * nDotL;
                    contributionFactor *= weight / envSample.pdf;
                    uint shadowIndex = wavefront_reserve_shadow_slot(uniforms,
                                                                     pathIndex,
                                                                     state,
                                                                     queueCounters,
                                                                     stats);
                    if (shadowIndex != kInvalidIndex) {
                        uint shadowExcludeMesh;
                        uint shadowExcludePrim;
                        compute_exclusion_indices(rec, shadowExcludeMesh, shadowExcludePrim);
                        ShadowRay shadow{};
                        shadow.originTMax = float4(offset_ray_origin(rec, envSample.direction), kInfinity);
                        shadow.direction = float4(envSample.direction, 0.0f);
                        shadow.contribution = float4(contributionFactor, 0.0f);
                        shadow.throughput = float4(state.throughput.xyz, 0.0f);
                        shadow.pathIndex = pathIndex;
                        shadow.lightId = kInvalidIndex;
                        shadow.flags = kWavefrontShadowFlagEnvironment;
                        shadow.excludeMeshIndex = shadowExcludeMesh;
                        shadow.excludePrimitiveIndex = shadowExcludePrim;
                        shadowRays[shadowIndex] = shadow;
                        if (stats) {
                            atomic_fetch_add_explicit(&stats->wavefrontEnvShadowRayCount,
                                                      1u,
                                                      memory_order_relaxed);
                        }
                    }
                }
            }
        }
    }
    if (queuedRay.depth + 1u >= max(uniforms.maxDepth, 1u)) {
        state.alive = 0u;
        wavefront_note_termination(queueCounters, stats, 3u);
        state.rngState = rngState;
        pathStates[pathIndex] = state;
        return;
    }

    BsdfSampleResult bsdfSample;
    bool usedRandomWalk = false;
    bool enableRandomWalk = material_is_subsurface(material) &&
                            sss_use_random_walk(uniforms.sssMode, material) &&
                            rec.frontFace != 0u;
    if (enableRandomWalk) {
#if __METAL_VERSION__ >= 310
        bool useHardwareRandomWalk =
            (uniforms.intersectionMode == kIntersectionModeHardwareRT) &&
            meshInfos && triangleData && sceneVertices && meshIndices && instanceUserIds;
        if (useHardwareRandomWalk) {
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
                                                         rngState,
                                                         clampParams);
            usedRandomWalk = (bsdfSample.pdf > 0.0f);
        } else
#endif
        {
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
                                                         rngState,
                                                         clampParams);
            usedRandomWalk = (bsdfSample.pdf > 0.0f);
        }
    }
    if (!usedRandomWalk) {
        bsdfSample = material_closure_sample(material,
                                             rec.point,
                                             rec.shadingNormal,
                                             wo,
                                             incidentDir,
                                             rec.frontFace != 0u,
                                             rngState,
                                             clampParams,
                                             uniforms.sssMode,
                                             diffuseOcclusion,
                                             specularOnly);
    }
    apply_restir_gi_diffuse_prototype(uniforms,
                                      material,
                                      rec.point,
                                      shadingNormal,
                                      wo,
                                      incidentDir,
                                      rec.frontFace != 0u,
                                      clampParams,
                                      diffuseOcclusion,
                                      specularOnly,
                                      queuedRay.depth,
                                      pixel,
                                      materialResources.restirPtReservoirs,
                                      stats,
                                      rngState,
                                      bsdfSample);
    apply_path_guiding_prototype(uniforms,
                                 material,
                                 rec.point,
                                 shadingNormal,
                                 wo,
                                 clampParams,
                                 diffuseOcclusion,
                                 specularOnly,
                                 queuedRay.depth,
                                 pixel,
                                 materialResources.pathGuidingStates,
                                 stats,
                                 rngState,
                                 bsdfSample);
    apply_restir_pt_experimental_path_reuse(uniforms,
                                            material,
                                            rec.point,
                                            shadingNormal,
                                            wo,
                                            state.throughput.xyz,
                                            clampParams,
                                            diffuseOcclusion,
                                            specularOnly,
                                            queuedRay.depth,
                                            pixel,
                                            materialResources.restirPtReservoirs,
                                            stats,
                                            bsdfSample);
    capture_restir_pt_research_scaffold(uniforms,
                                        material,
                                        rec.point,
                                        shadingNormal,
                                        state.throughput.xyz,
                                        specularOnly,
                                        queuedRay.depth,
                                        pixel,
                                        materialResources.restirPtReservoirs,
                                        stats,
                                        bsdfSample);
    if (debugCtxPtr) {
        record_debug_event(*debugCtxPtr,
                           kDebugEventBsdfRng,
                           queuedRay.depth,
                           0u,
                           0u,
                           0,
                           rec.frontFace,
                           materialIndex,
                           bsdfSample.isDelta,
                           rec.shadingNormal,
                           float4(as_type<float>(state.rngState),
                                  as_type<float>(rngState),
                                  float(type),
                                  usedRandomWalk ? 1.0f : 0.0f),
                           wo);
        record_debug_event(*debugCtxPtr,
                           kDebugEventBsdfState,
                           queuedRay.depth,
                           0u,
                           0u,
                           0,
                           rec.frontFace,
                           materialIndex,
                           bsdfSample.isDelta,
                           incidentDir,
                           float4(bsdfSample.pdf,
                                  bsdfSample.directionalPdf,
                                  float(bsdfSample.lobeType),
                                  bsdfSample.isDelta ? 1.0f : 0.0f),
                           bsdfSample.direction);
    }
    if (bsdfSample.pdf <= 0.0f ||
        !all(isfinite(bsdfSample.direction)) ||
        !all(isfinite(bsdfSample.weight)) ||
        dot(bsdfSample.direction, bsdfSample.direction) <= 0.0f) {
        state.alive = 0u;
        wavefront_note_termination(queueCounters, stats, 3u);
        state.rngState = rngState;
        pathStates[pathIndex] = state;
        return;
    }
    caustic_transport_note_path_vertex(uniforms,
                                       material,
                                       rec.point,
                                       rec.shadingNormal,
                                       state.throughput.xyz,
                                       queuedRay.depth,
                                       bsdfSample,
                                       stats);
    float3 radianceAfterCache = state.radiance.xyz;
    RadianceQuery cacheQuery = radiance_cache_make_query(uniforms,
                                                         material,
                                                         rec.point,
                                                         rec.shadingNormal,
                                                         queuedRay.depth,
                                                         bsdfSample);
    uint cacheFallbackReason = 0u;
    if (radiance_cache_can_query(cacheQuery,
                                 material,
                                 specularOnly,
                                 bsdfSample,
                                 queuedRay.depth,
                                 uniforms,
                                 cacheFallbackReason)) {
        wavefront_note_queue_enqueue(queueCounters->cacheQueryQueue);
    }
    if (radiance_cache_query_and_maybe_terminate(uniforms,
                                                 material,
                                                 rec.point,
                                                 rec.shadingNormal,
                                                 specularOnly,
                                                 queuedRay.depth,
                                                 bsdfSample,
                                                 materialResources.radianceCacheStates,
                                                 stats,
                                                 radianceAfterCache,
                                                 state.throughput.xyz)) {
        state.radiance.xyz = radianceAfterCache;
        state.alive = 0u;
        wavefront_note_termination(queueCounters, stats, 3u);
        state.rngState = rngState;
        pathStates[pathIndex] = state;
        return;
    }
    if (radiance_cache_train(materialResources.radianceCacheStates,
                             uniforms,
                             material,
                             rec.point,
                             rec.shadingNormal,
                             specularOnly,
                             queuedRay.depth,
                             bsdfSample,
                             stats)) {
        wavefront_note_queue_enqueue(queueCounters->cacheTrainingQueue);
    }

    uint mediumDepthBefore = mediumDepth;
    if (bsdfSample.mediumEvent == 1) {
        float3 sigma = dielectric_sigma_a(material);
        sigma = max(sigma, float3(0.0f));
        if (mediumDepth < kWavefrontMediumStackSize) {
            mediumSigmaStack[mediumDepth] = sigma;
            mediumDepth += 1u;
        } else {
            mediumSigmaStack[kWavefrontMediumStackSize - 1u] = sigma;
        }
    } else if (bsdfSample.mediumEvent == -1) {
        if (mediumDepth > 0u) {
            mediumDepth -= 1u;
        }
    }
    volume_note_boundary_event(bsdfSample.mediumEvent, stats);
    uint mediumDepthAfter = mediumDepth;

    const float3 nextDirection = safe_normalize(bsdfSample.direction);
    bool didTransmission = false;
    if (bsdfSample.isDelta && type == 2u) {
        float3 dir = bsdfSample.direction;
        if (all(isfinite(dir)) && dot(dir, dir) > 0.0f) {
            float side = (rec.frontFace != 0u) ? 1.0f : -1.0f;
            didTransmission = (dot(shadingNormal, dir) * side) < 0.0f;
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

    bool specNeeEnabled = (uniforms.enableSpecularNee != 0u);
    float dirLenSq = dot(bsdfSample.direction, bsdfSample.direction);
    bool specDirectionValid = (dirLenSq > 0.0f) && all(isfinite(bsdfSample.direction));
    bool specNeeEligible = specNeeEnabled &&
                           bsdfSample.isDelta &&
                           (bsdfSample.mediumEvent <= 0) &&
                           specDirectionValid;

    if (specNeeEligible && envSampling &&
        environmentTexture.get_width() > 0 &&
        environmentTexture.get_height() > 0) {
        Ray neeRay;
        neeRay.origin = nextOrigin;
        neeRay.direction = normalize(bsdfSample.direction);
        HitRecord shadowRec = make_empty_hit_record();
        uint neeExcludeMesh = kInvalidIndex;
        uint neeExcludePrim = kInvalidIndex;
        if (didTransmission) {
            compute_exclusion_indices(rec, neeExcludeMesh, neeExcludePrim);
            neeExcludePrim = kInvalidIndex;
        }
        bool occluded = false;
#if __METAL_VERSION__ >= 310
        if (uniforms.intersectionMode == kIntersectionModeHardwareRT &&
            meshInfos && triangleData && sceneVertices && meshIndices && instanceUserIds) {
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
                                            kEpsilon,
                                            kInfinity,
                                            /*anyHitOnly=*/true,
                                                neeExcludeMesh,
                                                neeExcludePrim,
                                            shadowRec);
        } else
#endif
        {
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
        }

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
                state.radiance.xyz +=
                    clamp_firefly_contribution(state.throughput.xyz, neeContribution, clampParams, stats);
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
        HitRecord lightRec = make_empty_hit_record();
        uint neeExcludeMesh = kInvalidIndex;
        uint neeExcludePrim = kInvalidIndex;
        if (didTransmission) {
            compute_exclusion_indices(rec, neeExcludeMesh, neeExcludePrim);
            neeExcludePrim = kInvalidIndex;
        }
        bool hitLight = false;
#if __METAL_VERSION__ >= 310
        if (uniforms.intersectionMode == kIntersectionModeHardwareRT &&
            meshInfos && triangleData && sceneVertices && meshIndices && instanceUserIds) {
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
                                            kEpsilon,
                                            kInfinity,
                                            /*anyHitOnly=*/false,
                                                neeExcludeMesh,
                                                neeExcludePrim,
                                            lightRec);
        } else
#endif
        {
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
        }
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
                    state.radiance.xyz +=
                        clamp_firefly_contribution(state.throughput.xyz, contribution, clampParams, stats);
                    if (stats) {
                        atomic_fetch_add_explicit(&stats->specNeeRectAddedCount, 1u, memory_order_relaxed);
                    }
                }
            }
        }
    }

    if (debugCtxPtr) {
        record_debug_event(*debugCtxPtr,
                           kDebugEventScatter,
                           queuedRay.depth,
                           mediumDepthBefore,
                           mediumDepthAfter,
                           bsdfSample.mediumEvent,
                           rec.frontFace,
                           materialIndex,
                           bsdfSample.isDelta,
                           state.throughput.xyz,
                           float4(bsdfSample.pdf,
                                  float(bsdfSample.lobeType),
                                  bsdfSample.lobeRoughness,
                                  didTransmission ? 1.0f : 0.0f),
                           nextDirection);
    }

    state.throughput.xyz *= bsdfSample.weight;
    state.throughput.xyz = clamp_path_throughput(state.throughput.xyz, clampParams, stats);
    if (!all(isfinite(state.throughput.xyz)) ||
        max(max(state.throughput.x, state.throughput.y), state.throughput.z) <= 0.0f) {
        state.alive = 0u;
        wavefront_note_termination(queueCounters, stats, 3u);
        state.rngState = rngState;
        pathStates[pathIndex] = state;
        return;
    }

    if (uniforms.useRussianRoulette != 0u && queuedRay.depth >= 5u) {
        float continueProbability =
            clamp(max(max(state.throughput.x, state.throughput.y), state.throughput.z), 0.05f, 0.95f);
        if (rand_uniform(rngState) > continueProbability) {
            state.alive = 0u;
            wavefront_note_termination(queueCounters, stats, 3u);
            state.rngState = rngState;
            pathStates[pathIndex] = state;
            return;
        }
        state.throughput.xyz /= continueProbability;
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
    rayCone.width = ray_cone_width_at_distance(rayCone,
                                               ray_segment_world_length(incomingRay, rec.t));
    rayCone.spread = min(rayCone.spread +
                         bsdf_cone_spread_increment(bsdfSample.lobeType,
                                                    bsdfSample.lobeRoughness,
                                                    bsdfSample.isDelta),
                         1.5f);
    uint nextIndex =
        atomic_fetch_add_explicit(&queueCounters->nextRayCount, 1u, memory_order_relaxed);
    if (stats) {
        atomic_fetch_add_explicit(&stats->wavefrontNextRayEnqueueCount, 1u, memory_order_relaxed);
    }
    if (nextIndex < max(uniforms.width * uniforms.height, 1u)) {
        wavefront_note_queue_enqueue(queueCounters->diffuseRayQueue);
        WavefrontRay nextRay{};
        nextRay.originMinT = float4(nextOrigin, kEpsilon);
        nextRay.directionMaxT = float4(nextDirection, kInfinity);
        PrimaryRayDiff clearedDiff{};
        wavefront_store_primary_ray_diff(nextRay, clearedDiff);
        wavefront_store_ray_cone(nextRay, rayCone);
        nextRay.pixelIndex = queuedRay.pixelIndex;
        nextRay.pathIndex = pathIndex;
        nextRay.depth = queuedRay.depth + 1u;
        nextRay.flags = bsdfSample.isDelta ? 1u : 0u;
        compute_exclusion_indices(rec, nextRay.excludeMeshIndex, nextRay.excludePrimitiveIndex);
        nextRays[nextIndex] = nextRay;
        state.envLod = nextEnvLod;
        state.lastBsdfPdf =
            (bsdfSample.directionalPdf > 0.0f) ? bsdfSample.directionalPdf : bsdfSample.pdf;
        state.mediumDepth = mediumDepth;
        state.alive = 1u;
        state.flags = nextEnvLodActive ? kWavefrontFlagEnvLodActive : 0u;
        if (bsdfSample.isDelta) {
            state.flags |= kWavefrontFlagLastScatterDelta;
        }
        for (uint i = 0u; i < kWavefrontMediumStackSize; ++i) {
            state.mediumStack[i] = float4(mediumSigmaStack[i], 0.0f);
        }
    } else {
        atomic_store_explicit(&queueCounters->overflowFlag, 1u, memory_order_relaxed);
        wavefront_note_queue_overflow(queueCounters->diffuseRayQueue);
        state.alive = 0u;
        wavefront_note_termination(queueCounters, stats, 4u);
    }

    state.rngState = rngState;
    pathStates[pathIndex] = state;
}

kernel void wavefrontShadowKernel(constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                  texture2d<float, access::sample> environmentTexture [[texture(0)]],
                                  device const ShadowRay* shadowRays [[buffer(1)]],
                                  device PathState* pathStates [[buffer(2)]],
                                  device QueueCounters* queueCounters [[buffer(3)]],
                                  device PathtraceStats* stats [[buffer(4)]],
                                  device const SphereData* spheres [[buffer(5)]],
                                  device const RectData* rectangles [[buffer(6)]],
                                  device const TriangleData* triangleData [[buffer(7)]],
                                  device const BvhNode* tlasNodes [[buffer(8)]],
                                  device const uint* tlasPrimIndices [[buffer(9)]],
                                  device const SoftwareInstanceInfo* instanceInfos [[buffer(10)]],
                                  device const BvhNode* blasNodes [[buffer(11)]],
                                  device const uint* blasPrimIndices [[buffer(12)]],
                                  device const BvhNode* nodes [[buffer(13)]],
                                  device const uint* primitiveIndices [[buffer(14)]],
                                  device const MaterialData* materials [[buffer(15)]],
                                  uint gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
    if (gid == 0u) {
        wavefront_store_queue_stats(queueCounters, stats);
    }
    if (gid >= pixelCount) {
        return;
    }

    PathState state = pathStates[gid];
    const uint slotCount =
        min(state.reserved0, kWavefrontShadowCapacityMultiplier);
    if (slotCount == 0u) {
        return;
    }

    for (uint slot = 0u; slot < slotCount; ++slot) {
        ShadowRay queued = shadowRays[gid * kWavefrontShadowCapacityMultiplier + slot];
        if (stats) {
            atomic_fetch_add_explicit(&stats->wavefrontShadowProcessedCount, 1u, memory_order_relaxed);
        }

        Ray shadowRay;
        shadowRay.origin = queued.originTMax.xyz;
        shadowRay.direction = queued.direction.xyz;
        HitRecord shadowRec = make_empty_hit_record();
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
                                             queued.originTMax.w,
                                             /*anyHitOnly=*/true,
                                             /*includeTriangles=*/true,
                                             shadowRec);
        if ((queued.flags & kWavefrontShadowFlagEmissivePrimitive) != 0u &&
            occluded &&
            shadowRec.primitiveType == kPrimitiveTypeTriangle &&
            shadowRec.primitiveIndex == queued.lightId) {
            occluded = false;
        }
        if ((queued.flags & kWavefrontShadowFlagEnvironment) != 0u) {
            bool connected = !occluded;
            float3 envDirection = shadowRay.direction;
            float3 chainWeight = float3(1.0f);
            bool usedDeltaChain = false;
            FireflyClampParams clampParams = make_firefly_params(uniforms);
            if (!connected &&
                (uniforms.enableMnee != 0u) &&
                (uniforms.enableMneeSecondary != 0u)) {
                connected = trace_environment_delta_chain_software(uniforms,
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
                                                                   materials,
                                                                   stats,
                                                                   state.rngState,
                                                                   clampParams,
                                                                   (uniforms.debugSpecularOnly != 0u),
                                                                   shadowRay,
                                                                   envDirection,
                                                                   chainWeight,
                                                                   usedDeltaChain);
            }
            if (connected) {
                float3 envRadiance = environment_color(environmentTexture,
                                                       envDirection,
                                                       uniforms.environmentRotation,
                                                       uniforms.environmentIntensity,
                                                       uniforms);
                float3 contribution = queued.contribution.xyz * envRadiance * chainWeight;
                state.radiance.xyz +=
                    clamp_firefly_contribution(queued.throughput.xyz, contribution, clampParams, stats);
            }
        } else if (!occluded) {
            state.radiance.xyz += queued.contribution.xyz;
        }
    }
    pathStates[gid] = state;
}

#if __METAL_VERSION__ >= 310
kernel void wavefrontShadowHardwareKernel(constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                          acceleration_structure<instancing> accel [[buffer(1)]],
                                          texture2d<float, access::sample> environmentTexture [[texture(0)]],
                                          device const ShadowRay* shadowRays [[buffer(2)]],
                                          device PathState* pathStates [[buffer(3)]],
                                          device QueueCounters* queueCounters [[buffer(4)]],
                                          device PathtraceStats* stats [[buffer(5)]],
                                          device const MeshInfo* meshInfos [[buffer(6)]],
                                          device const TriangleData* triangleData [[buffer(7)]],
                                          device const SceneVertex* sceneVertices [[buffer(8)]],
                                          device const uint3* meshIndices [[buffer(9)]],
                                          device const uint* instanceUserIds [[buffer(10)]],
                                          device const SphereData* spheres [[buffer(11)]],
                                          device const RectData* rectangles [[buffer(12)]],
                                          device const BvhNode* nodes [[buffer(13)]],
                                          device const uint* primitiveIndices [[buffer(14)]],
                                          device const MaterialData* materials [[buffer(15)]],
                                          device const BvhNode* tlasNodes [[buffer(16)]],
                                          device const uint* tlasPrimIndices [[buffer(17)]],
                                          device const BvhNode* blasNodes [[buffer(18)]],
                                          device const uint* blasPrimIndices [[buffer(19)]],
                                          device const SoftwareInstanceInfo* instanceInfos [[buffer(20)]],
                                          uint gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
    if (gid == 0u) {
        wavefront_store_queue_stats(queueCounters, stats);
    }
    if (gid >= pixelCount) {
        return;
    }

    PathState state = pathStates[gid];
    const uint slotCount =
        min(state.reserved0, kWavefrontShadowCapacityMultiplier);
    if (slotCount == 0u) {
        return;
    }

    for (uint slot = 0u; slot < slotCount; ++slot) {
        ShadowRay queued = shadowRays[gid * kWavefrontShadowCapacityMultiplier + slot];
        if (stats) {
            atomic_fetch_add_explicit(&stats->wavefrontShadowProcessedCount, 1u, memory_order_relaxed);
        }

        Ray shadowRay;
        shadowRay.origin = queued.originTMax.xyz;
        shadowRay.direction = queued.direction.xyz;
        HitRecord shadowRec = make_empty_hit_record();
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
                                             kEpsilon,
                                             queued.originTMax.w,
                                             /*anyHitOnly=*/true,
                                             queued.excludeMeshIndex,
                                             queued.excludePrimitiveIndex,
                                             shadowRec);
        if ((queued.flags & kWavefrontShadowFlagEmissivePrimitive) != 0u &&
            occluded &&
            shadowRec.primitiveType == kPrimitiveTypeTriangle &&
            shadowRec.primitiveIndex == queued.lightId) {
            occluded = false;
        }
        if ((queued.flags & kWavefrontShadowFlagEnvironment) != 0u) {
            bool connected = !occluded;
            float3 envDirection = shadowRay.direction;
            float3 chainWeight = float3(1.0f);
            bool usedDeltaChain = false;
            FireflyClampParams clampParams = make_firefly_params(uniforms);
            if (!connected &&
                (uniforms.enableMnee != 0u) &&
                (uniforms.enableMneeSecondary != 0u)) {
                connected = trace_environment_delta_chain_hardware(uniforms,
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
                                                                   materials,
                                                                   stats,
                                                                   state.rngState,
                                                                   clampParams,
                                                                   (uniforms.debugSpecularOnly != 0u),
                                                                   shadowRay,
                                                                   envDirection,
                                                                   chainWeight,
                                                                   usedDeltaChain);
            }
            if (connected) {
                float3 envRadiance = environment_color(environmentTexture,
                                                       envDirection,
                                                       uniforms.environmentRotation,
                                                       uniforms.environmentIntensity,
                                                       uniforms);
                float3 contribution = queued.contribution.xyz * envRadiance * chainWeight;
                state.radiance.xyz +=
                    clamp_firefly_contribution(queued.throughput.xyz, contribution, clampParams, stats);
            }
        } else if (!occluded) {
            state.radiance.xyz += queued.contribution.xyz;
        }
    }
    pathStates[gid] = state;
}
#endif

kernel void wavefrontSwapQueuesKernel(device QueueCounters* queueCounters [[buffer(0)]],
                                      device PathtraceStats* stats [[buffer(1)]],
                                      uint gid [[thread_position_in_grid]]) {
    if (gid != 0u || !queueCounters) {
        return;
    }
    const uint nextCount = atomic_load_explicit(&queueCounters->nextRayCount, memory_order_relaxed);
    atomic_store_explicit(&queueCounters->activeRayCount, nextCount, memory_order_relaxed);
    const uint boundedNextCount = min(nextCount, queueCounters->primaryRayQueue.capacity);
    atomic_store_explicit(&queueCounters->primaryRayQueue.activeCount, boundedNextCount, memory_order_relaxed);
    atomic_fetch_max_explicit(&queueCounters->primaryRayQueue.peakActiveCount,
                              boundedNextCount,
                              memory_order_relaxed);
    wavefront_store_queue_stats(queueCounters, stats);
}

kernel void wavefrontAccumulateKernel(texture2d<float, access::read_write> radianceTexture [[texture(0)]],
                                      texture2d<uint, access::read_write> sampleCountTexture [[texture(1)]],
                                      texture2d<float, access::write> albedoTexture [[texture(2)]],
                                      texture2d<float, access::write> normalTexture [[texture(3)]],
                                      texture2d<float, access::write> positionTexture [[texture(4)]],
                                      texture2d<float, access::write> materialFeatureTexture [[texture(5)]],
                                      texture2d<float, access::write> motionVectorTexture [[texture(6)]],
                                      constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                      device const PathState* pathStates [[buffer(1)]],
                                      device QueueCounters* queueCounters [[buffer(2)]],
                                      device PathtraceStats* stats [[buffer(3)]],
                                      uint gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
    if (gid >= pixelCount) {
        return;
    }

    uint2 pixel = wavefront_pixel_coord(gid, uniforms);
    float3 sample = pathStates[gid].radiance.xyz;
    if (!all(isfinite(sample))) {
        sample = float3(0.0f);
    } else {
        sample = max(sample, float3(0.0f));
    }

    float3 accumulated = radianceTexture.read(pixel).xyz;
    uint previousCount = sampleCountTexture.read(pixel).x;
    radianceTexture.write(float4(accumulated + sample, 0.0f), pixel);
    sampleCountTexture.write(previousCount + 1u, pixel);
    float3 firstHitAlbedo = pathStates[gid].firstHitAlbedo.xyz;
    float3 firstHitNormal = pathStates[gid].firstHitNormal.xyz;
    float4 firstHitPosition = pathStates[gid].firstHitPosition;
    float4 firstHitMaterial = pathStates[gid].firstHitMaterial;
    float albedoValid = pathStates[gid].firstHitAlbedo.w;
    float normalValid = pathStates[gid].firstHitNormal.w;
    if (albedoValid <= 0.0f) {
        firstHitAlbedo = float3(0.0f);
    } else if (stats) {
        atomic_fetch_add_explicit(&stats->wavefrontAovAlbedoValidCount, 1u, memory_order_relaxed);
    }
    if (normalValid <= 0.0f) {
        firstHitNormal = float3(0.0f);
    } else if (stats) {
        atomic_fetch_add_explicit(&stats->wavefrontAovNormalValidCount, 1u, memory_order_relaxed);
    }
    if (uniforms.debugViewMode == 17u && queueCounters) {
        const uint overflow =
            atomic_load_explicit(&queueCounters->overflowFlag, memory_order_relaxed);
        const uint primaryPeak =
            atomic_load_explicit(&queueCounters->primaryRayQueue.peakActiveCount,
                                 memory_order_relaxed);
        const uint shadowPeak =
            atomic_load_explicit(&queueCounters->shadowRayQueue.peakActiveCount,
                                 memory_order_relaxed);
        const float primaryOccupancy =
            (queueCounters->primaryRayQueue.capacity > 0u)
                ? static_cast<float>(primaryPeak) /
                      static_cast<float>(queueCounters->primaryRayQueue.capacity)
                : 0.0f;
        const float shadowOccupancy =
            (queueCounters->shadowRayQueue.capacity > 0u)
                ? static_cast<float>(shadowPeak) /
                      static_cast<float>(queueCounters->shadowRayQueue.capacity)
                : 0.0f;
        firstHitAlbedo = (overflow != 0u)
            ? float3(1.0f, 0.0f, 0.0f)
            : float3(primaryOccupancy, shadowOccupancy, 0.0f);
        firstHitNormal = float3(primaryOccupancy, shadowOccupancy, overflow != 0u ? 1.0f : 0.0f);
    }
    albedoTexture.write(float4(firstHitAlbedo, 1.0f), pixel);
    normalTexture.write(float4(firstHitNormal * 0.5f + 0.5f, 1.0f), pixel);
    positionTexture.write(firstHitPosition, pixel);
    materialFeatureTexture.write(firstHitMaterial, pixel);
    motionVectorTexture.write(float4(0.0f), pixel);
}
