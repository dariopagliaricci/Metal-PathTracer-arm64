inline uint wavefront_shadow_receiver_flags(const MaterialData receiver,
                                              const uint flags,
                                              const bool directional = false) {
    return flags | (interface_precision(receiver) ? kWavefrontShadowFlagInterfacePrecision : 0u) |
        (directional ? kWavefrontShadowFlagDirectional : 0u) |
        ((receiver.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u
            ? kWavefrontShadowFlagPavilionReceiver : 0u);
}

inline bool wavefront_reference_punctual_shadow(constant PathtraceUniforms& uniforms,
                                                const ShadowRay shadow,
                                                device const RectData* rectangles) {
    if ((shadow.flags & kWavefrontShadowFlagPavilionReceiver) == 0u) {
        return false;
    }
    return (shadow.flags & kWavefrontShadowFlagDirectional) != 0u ||
        ((shadow.flags & kWavefrontShadowFlagRectangleLight) != 0u && rectangles &&
         shadow.lightId < uniforms.rectangleCount &&
         area_primitive_is_reference_point(rectangles[shadow.lightId]));
}

constant uint kWavefrontExplicitRestirDiSurfaceAccepted = 0x52444941u;
constant bool kWavefrontShadeUseResolvedSurface [[function_constant(1)]];

inline void apply_resolved_surface_material(thread MaterialData& material,
                                            thread const WavefrontSurfaceState& surfaceState) {
    material.baseColorRoughness = surfaceState.resolvedBaseColorRoughness;
    material.typeEta = surfaceState.resolvedTypeEta;
    material.emission = surfaceState.resolvedEmission;
    material.pbrParams = surfaceState.resolvedPbrParams;
    material.pbrExtras = surfaceState.resolvedPbrExtras;
    if (static_cast<uint>(material.typeEta.x) == 7u &&
        (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u) {
        const float ior = max(material.typeEta.y, 1.0f);
        material.typeEta.y = surfaceState.rec.frontFace != 0u ? ior : 1.0f / ior;
    }
}

inline WavefrontResolvedSurface pack_resolved_surface(thread const WavefrontSurfaceState& surfaceState,
                                                      const uint pathIndex,
                                                      const uint materialIndex,
                                                      const uint rngState) {
    WavefrontResolvedSurface packed{};
    packed.hit = pack_wavefront_hit(surfaceState.rec, pathIndex, true);
    packed.materialBaseColorRoughness = surfaceState.resolvedBaseColorRoughness;
    packed.materialTypeEta = surfaceState.resolvedTypeEta;
    packed.materialEmission = surfaceState.resolvedEmission;
    packed.materialPbrParams = surfaceState.resolvedPbrParams;
    packed.materialPbrExtras = surfaceState.resolvedPbrExtras;
    packed.firstHitNormalDiffuse = float4(surfaceState.firstHitNormal,
                                          surfaceState.diffuseOcclusion);
    packed.meta = uint4(1u,
                        surfaceState.alphaDiscarded ? 1u : 0u,
                        materialIndex,
                        rngState);
    packed.recExtra = uint4(surfaceState.rec.twoSided, 0u, 0u, 0u);
    packed.sourceForwardWeight = float4(surfaceState.sourceForwardWeight,0.0f);
    return packed;
}

inline WavefrontSurfaceState unpack_resolved_surface(thread const WavefrontResolvedSurface& packed) {
    WavefrontSurfaceState surfaceState{};
    surfaceState.rec = unpack_wavefront_hit(packed.hit);
    surfaceState.rec.twoSided = packed.recExtra.x;
    surfaceState.resolvedBaseColorRoughness = packed.materialBaseColorRoughness;
    surfaceState.resolvedTypeEta = packed.materialTypeEta;
    surfaceState.resolvedEmission = packed.materialEmission;
    surfaceState.resolvedPbrParams = packed.materialPbrParams;
    surfaceState.resolvedPbrExtras = packed.materialPbrExtras;
    surfaceState.firstHitNormal = packed.firstHitNormalDiffuse.xyz;
    surfaceState.diffuseOcclusion = packed.firstHitNormalDiffuse.w;
    surfaceState.alphaDiscarded = packed.meta.y != 0u;
    surfaceState.sourceForwardWeight = packed.sourceForwardWeight.xyz;
    surfaceState.debugBaseColor = packed.materialBaseColorRoughness.xyz;
    surfaceState.debugMetallic = packed.materialPbrParams.x;
    surfaceState.debugRoughness = packed.materialBaseColorRoughness.w;
    surfaceState.debugAO = packed.firstHitNormalDiffuse.w;
    return surfaceState;
}

inline void wavefront_add_reconstruction_radiance(thread PathState& state,
                                                  const uint depth,
                                                  const float3 contribution) {
    state.radiance.xyz += contribution;
    if (depth == 0u) {
        state.directRadiance.xyz += contribution;
    } else {
        state.indirectRadiance.xyz += contribution;
    }
}

// Radiance cache training on the wavefront (see the deferred-training notes in radiance_cache.metal). A training
// path keeps its records in materialResources.radianceCacheTrainRecords, kRadianceCacheTrainRecordCount per
// training pixel: the training pixels of one frame are kRadianceCacheTrainingPathStride apart, so pixelIndex / stride
// is unique among them. The shade kernel writes a record at each cacheable vertex x. x's NEE is resolved later by
// the shadow kernels, which add it to the record's baseline, so the baseline holds what the path had accumulated
// through x as on the megakernel. wavefrontRadianceCacheTrainKernel trains from the records after the last bounce
// and invalidates them.
inline uint wavefront_radiance_cache_record_base(const uint pixelIndex) {
    return (pixelIndex / kRadianceCacheTrainingPathStride) * kRadianceCacheTrainRecordCount;
}

inline void wavefront_radiance_cache_note_vertex(device RadianceCacheTrainRecord* records,
                                                 device WavefrontQueueHeader& trainingQueue,
                                                 constant PathtraceUniforms& uniforms,
                                                 const MaterialData material,
                                                 const float3 position,
                                                 const float3 normal,
                                                 const bool specularOnly,
                                                 const uint depth,
                                                 const BsdfSampleResult sample,
                                                 const uint pixelIndex,
                                                 const float3 radianceBaseline,
                                                 const float3 throughputAtQuery) {
    const uint slot = radiance_cache_train_slot(uniforms, depth);
    if (!records || slot >= kRadianceCacheTrainRecordCount) {
        return;
    }
    RadianceCacheTrainRecord record;
    if (!radiance_cache_make_train_record(uniforms, material, position, normal, specularOnly, depth, sample,
                                          radianceBaseline, throughputAtQuery, record)) {
        return;
    }
    records[wavefront_radiance_cache_record_base(pixelIndex) + slot] = record;
    // The cache-training queue now counts records written (wavefront queue metrics); it holds no items.
    if (wavefront_note_queue_enqueue(trainingQueue) >= trainingQueue.capacity) {
        wavefront_note_queue_overflow(trainingQueue);
    }
}

// Adds a resolved shadow-ray contribution to the path and, on a training path, to the baseline of the record of the
// vertex that cast it. A wavefront path's index is its pixel index (wavefrontGeneratePrimaryKernel).
inline void wavefront_add_shadow_radiance(thread PathState& state,
                                          device RadianceCacheTrainRecord* records,
                                          constant PathtraceUniforms& uniforms,
                                          const uint pathIndex,
                                          const uint depth,
                                          const float3 contribution) {
    wavefront_add_reconstruction_radiance(state, depth, contribution);
    if (!records || !radiance_cache_is_training_path(uniforms, pathIndex)) {
        return;
    }
    const uint slot = radiance_cache_train_slot(uniforms, depth);
    if (slot >= kRadianceCacheTrainRecordCount) {
        return;
    }
    device RadianceCacheTrainRecord& record = records[wavefront_radiance_cache_record_base(pathIndex) + slot];
    if (radiance_cache_train_record_valid(record)) {
        record.baselineMeta.xyz += contribution;
    }
}

inline void wavefront_store_radiance_cache_work_item(
    device RadianceCacheWorkItem* items,
    device WavefrontQueueHeader& header,
    const RadianceQuery query,
    const float4 radianceConfidence,
    const float4 varianceBiasRisk,
    const uint pathIndex,
    const uint pixelIndex,
    const uint depth,
    const uint flags) {
    const uint slot = wavefront_note_queue_enqueue(header);
    if (slot >= header.capacity) {
        wavefront_note_queue_overflow(header);
        return;
    }
    if (!items) {
        return;
    }
    RadianceCacheWorkItem item;
    item.query = query;
    item.radianceConfidence = radianceConfidence;
    item.varianceBiasRisk = varianceBiasRisk;
    item.pathPixelDepthFlags = uint4(pathIndex, pixelIndex, depth, flags);
    items[slot] = item;
}

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
    const uint radianceCacheQueueCapacity =
        uniformsBuffer[0].radianceCacheMode == kRadianceCacheModePrototype ? pixelCount : 0u;
    wavefront_init_queue_header(queueCounters->cacheQueryQueue,
                                kWavefrontQueueCacheQuery,
                                radianceCacheQueueCapacity,
                                uniformsBuffer[0].frameIndex,
                                10u,
                                3u,
                                3u);
    wavefront_init_queue_header(queueCounters->cacheTrainingQueue,
                                kWavefrontQueueCacheTraining,
                                radianceCacheQueueCapacity,
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

    const bool deterministicDebugRay =
        (PT_DEFAULT_OFF(uniforms.debugPathActive) != 0u) &&
        (pixel.x == uniforms.debugPixelX) &&
        (pixel.y == uniforms.debugPixelY);
    float uJitter = deterministicDebugRay ? 0.5f : rand_uniform(rngState);
    float vJitter = deterministicDebugRay ? 0.5f : rand_uniform(rngState);
    float u = (float(pixel.x) + uJitter) / float(max(uniforms.width, 1u));
    float v = (float(pixel.y) + vJitter) / float(max(uniforms.height, 1u));
    v = 1.0f - v;

    float3 pixelPosition =
        uniforms.lowerLeftCorner + u * uniforms.horizontal + v * uniforms.vertical;
    float2 diskSample = deterministicDebugRay ? float2(0.0f)
                                              : (uniforms.lensRadius * random_in_unit_disk(rngState));
    float3 offset = uniforms.cameraU * diskSample.x + uniforms.cameraV * diskSample.y;
    float3 origin = uniforms.cameraOrigin + offset;
    PrimaryRayDiff primaryRayDiff;
    primaryRayDiff.dOdx = float3(0.0f);
    primaryRayDiff.dOdy = float3(0.0f);
    primaryRayDiff.dDdx = uniforms.horizontal / max(float(uniforms.width), 1.0f);
    primaryRayDiff.dDdy = -uniforms.vertical / max(float(uniforms.height), 1.0f);
    float3 direction = normalize_camera_ray_direction(pixelPosition - origin, primaryRayDiff);

    WavefrontRay ray{};
    ray.originMinT = float4(origin, kEpsilon);
    ray.directionMaxT = float4(direction, kInfinity);
    ray.mediumState = volume_make_global_medium_state(uniforms, 0u, 0u);
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
    state.directRadiance = float4(0.0f);
    state.indirectRadiance = float4(0.0f);
    state.firstHitAlbedo = float4(0.0f);
    state.firstHitNormal = float4(0.0f);
    state.firstHitPosition = float4(0.0f);
    state.firstHitMaterial = float4(0.0f);
    state.lastBsdfPdf = 1.0f;
    state.rngState = rngState;
    state.mediumDepth = 0u;
    state.currentMediumId = 0u;
    state.volumeScatteringDepth = 0u;
    SpectralPathState spectralState = spectral_make_path_state(uniforms, rngState, stats);
    state.spectralWavelengthNm = spectralState.wavelengthNm;
    state.spectralSampleIndex = spectralState.sampleIndex;
    if (spectralState.enabled) {
        // Channel j of the throughput carries wavelength j; inactive channels start at zero.
        state.throughput = float4(spectral_channel_mask(spectralState), 0.0f);
    }
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
        atomic_store_explicit(&queueCounters->reserved0, 0u, memory_order_relaxed);
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

// Two-sided cutout foliage takes direct light from either side, as in the
// megakernel (two_sided_alpha_cutout_direct_cos and its diffuse fallback).
inline bool wavefront_two_sided_cutout_foliage(constant PathtraceUniforms& uniforms,
                                               const MaterialData material,
                                               thread const HitRecord& rec) {
    return material_is_two_sided_alpha_cutout(uniforms, material, rec) &&
           (material.materialFlags & kMaterialFlagAlphaCutoutContinuation) == 0u;
}

// Freeze visibility state before the continuation updates the medium stack.
// Use geometric Ng for spawning and aim finite shadows at the sampled point.
inline void wavefront_prepare_source_shadow(const MaterialData material,
                                            thread const HitRecord& rec,
                                            thread const PathState& state,
                                            const float3 target,
                                            const bool finiteTarget,
                                            thread ShadowRay& shadow) {
    float3 direction = shadow.direction.xyz;
    float3 origin = offset_interface_origin(rec,direction);
    if (finiteTarget) {
        const float3 delta = target-origin;
        const float distance = length(delta);
        direction = delta/max(distance,1.0e-20f);
        shadow.originTMax.w = max(distance-1.0e-6f,1.0e-6f);
    }
    shadow.originTMax.xyz = origin;
    shadow.direction.xyz = direction;
    float3 sigma = state.mediumDepth > 0u ? state.mediumStack[min(state.mediumDepth-1u,kWavefrontMediumStackSize-1u)].xyz : float3(0.0f);
    uint mediumDepth = state.mediumDepth;
    interface_select_medium(material,rec,direction,sigma,mediumDepth);
    shadow.sourceSigma = float4(sigma,0.0f);
    shadow.sourceOrigin = float4(rec.point,1.0f);
}

// Surface cosine of the baseline, rectangle/punctual and environment light
// samples: the megakernel's two_sided_alpha_cutout_direct_cos, plus the Falcor
// face test on one-sided surfaces.
inline float wavefront_direct_light_cos(constant PathtraceUniforms& uniforms,
                                        const MaterialData material,
                                        thread const HitRecord& rec,
                                        const float3 normal,
                                        const float3 wi) {
    // EP8-f1: one rough-source-transmission rule for every executor and the RIS target (it was model 3 with alpha > 0
    // here, which differed only where the closure is delta or returns nothing below the surface).
    const float nDotL = dot(source_light_cos_normal(material, normal), wi);
    if (wavefront_two_sided_cutout_foliage(uniforms, material, rec) ||
        (material_is_regularized_dielectric(material) || source_rough_transmission(material)) || pavilion_transmission_nee_supported(material) ||
        pbr_thin_walled_transmission_nee_supported(material)) {
        return saturate(fabs(nDotL));
    }
    return saturate(nDotL) * falcor_face_visibility(material, rec, wi);
}

// Whether the shade kernel's fused mesh-emitter NEE can produce wi, so a BSDF
// scatter along wi reaches emission that NEE also estimates: the cosine of the
// baseline light-sample block, or the reservoir shading cosine.
inline bool wavefront_mesh_emitter_nee_covers_direction(constant PathtraceUniforms& uniforms,
                                                        const MaterialData material,
                                                        thread const HitRecord& rec,
                                                        const float3 normal,
                                                        const float3 wi) {
    if (uniforms.directLightMode == kDirectLightModeBaselineEmissive) {
        return wavefront_direct_light_cos(uniforms, material, rec, normal, wi) > 0.0f;
    }
    return direct_light_target_cos(uniforms, material, rec, normal, wi) > 0.0f;
}

// The megakernel alpha-tests every MASK/BLEND shadow hit, nearest first
// (trace_scene_software/hardware_alpha_aware_shadow). An any-hit on such a material
// may be a discarded texel, so all four shadow kernels, software and hardware,
// retrace those rays the same way.
inline bool wavefront_shadow_hit_alpha_tested(constant PathtraceUniforms& uniforms,
                                              thread const HitRecord& rec,
                                              device const MaterialData* materials) {
    return materials && rec.primitiveType == kPrimitiveTypeTriangle &&
           rec.materialIndex < uniforms.materialCount &&
           (materials[rec.materialIndex].pbrExtras.w > 0.5f || source_has_opacity(materials[rec.materialIndex]));
}

// The megakernel's shadow-ray tMin, the same as its bounce rays'.
inline float wavefront_shadow_t_min(const ShadowRay queued) {
    if (kSourceFeatures && (queued.flags & kWavefrontShadowFlagInterfacePrecision) != 0u) return kInterfaceMinT;
    return kEpsilon;
}

// Optical-origin distance correction of a ray spawned from a tagged source interface (rayDiffDOdx.w).
inline float wavefront_source_distance_offset(const WavefrontRay ray) {
    return kSourceFeatures ? ray.rayDiffDOdx.w : 0.0f;
}

inline float wavefront_shadow_software_t_min(const ShadowRay queued) {
    return (kSourceFeatures && (queued.flags & kWavefrontShadowFlagInterfacePrecision) != 0u) ? kInterfaceMinT : kEpsilon;
}

inline bool wavefront_alpha_cutout_continuation(
    constant PathtraceUniforms& uniforms,
    thread const HitRecord& rec,
    device const MaterialData* materials,
    constant MaterialTextureArgumentBuffer& materialResources,
    device const MaterialTextureInfo* materialTextureInfos,
    device const MeshInfo* meshInfos,
    device const SceneVertex* sceneVertices,
    device const uint3* meshIndices) {
    if (!materials || rec.primitiveType != kPrimitiveTypeTriangle ||
        rec.materialIndex >= uniforms.materialCount) {
        return false;
    }
    const MaterialData material = materials[rec.materialIndex];
    if ((material.materialFlags & (kMaterialFlagAlphaCutoutContinuation | kMaterialFlagSingleScatterSpecular)) == 0u ||
        material.pbrExtras.w <= 0.5f || material.pbrExtras.w >= 1.5f) {
        return false;
    }
    float alpha = clamp(material.pbrExtras.x, 0.0f, 1.0f);
    const uint textureIndex = material.textureIndices0.x;
    if (!material_texture_valid(uniforms, textureIndex)) return alpha < material.pbrExtras.y;
    // Resolve only opacity in the intersection loop. Full surface resolution
    // belongs in shade and needlessly loads normals/ORM/emission for rejected
    // cards, with excessive scratch storage on hardware traversal kernels.
    const uint uvSet = material.textureUvSet0.x == 1u ? 1u : 0u;
    const float2 uv = interpolate_uv(uniforms, rec.meshIndex, rec.primitiveIndex,
                                     rec.barycentric, uvSet, meshInfos, sceneVertices, meshIndices);
    // MASK tests the hit point's own alpha at mip 0, as shade (path_cutout_alpha) and shadow rays do.
    const PbrTextureSamplingContext ctx = make_pbr_texture_sampling_context(material,
        kPbrTextureSlotBaseColor, uv, uv, false, float2(0.0f), float2(0.0f), 0.0f,
        false, float2(0.0f), float2(0.0f), 0.0f);
    const uint samplerIndex = min(materialTextureInfos[textureIndex].flags, kMaxMaterialSamplers - 1u);
    float4 texel = materialResources.textures[textureIndex].sample(materialResources.samplers[samplerIndex],
        ctx.uv, level(0.0f));
    if ((material.materialFlags & kMaterialFlagBlackKeyAlphaFromRgb) != 0u) {
        texel.w = smoothstep(0.72f, 0.90f, dot(texel.xyz, kLuminanceWeights));
    }
    return clamp(alpha * texel.w, 0.0f, 1.0f) < material.pbrExtras.y;

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
                                     device const MaterialData* materials [[buffer(18)]],
                                     device const MaterialTextureInfo* materialTextureInfos [[buffer(19)]],
                                     constant MaterialTextureArgumentBuffer& materialResources [[buffer(23)]],
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
    ray.mediumState = queuedRay.mediumState;

    HitRecord rec = make_empty_hit_record();
    bool hit = false;
    float intersectionMinT = queuedRay.originMinT.w;
    for (uint alphaSkip = 0u; alphaSkip <= 64u; ++alphaSkip) {
        rec = make_empty_hit_record();
        hit = trace_scene_software_with_exclusion(uniforms,
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
                                                         intersectionMinT,
                                                         queuedRay.directionMaxT.w,
                                                         queuedRay.excludeMeshIndex,
                                                         queuedRay.excludePrimitiveIndex,
                                                         rec);
        if (!hit || alphaSkip == 64u ||
            !wavefront_alpha_cutout_continuation(uniforms, rec,
                                                  materials, materialResources,
                                                  materialTextureInfos, meshInfos,
                                                  sceneVertices, meshIndices)) {
            break;
        }
        // Retain the original ray origin: the accepted hit's t remains the
        // full segment distance used by attenuation and texture footprints.
        const float nextMinT = rec.t + kEpsilon;
        if (!isfinite(nextMinT) || nextMinT <= intersectionMinT ||
            nextMinT >= queuedRay.directionMaxT.w) {
            hit = false;
            break;
        }
        intersectionMinT = nextMinT;
    }
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
                                             device const MaterialData* materials [[buffer(18)]],
                                             device const MaterialTextureInfo* materialTextureInfos [[buffer(19)]],
                                             constant MaterialTextureArgumentBuffer& materialResources [[buffer(23)]],
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
    ray.mediumState = queuedRay.mediumState;

    HitRecord rec = make_empty_hit_record();
    bool hit = false;
    float intersectionMinT = queuedRay.originMinT.w;
    for (uint alphaSkip = 0u; alphaSkip <= 64u; ++alphaSkip) {
        rec = make_empty_hit_record();
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
                                          intersectionMinT,
                                          queuedRay.directionMaxT.w,
                                          /*anyHitOnly=*/false,
                                          queuedRay.excludeMeshIndex,
                                          queuedRay.excludePrimitiveIndex,
                                          rec);
        if (!hit || alphaSkip == 64u ||
            !wavefront_alpha_cutout_continuation(uniforms, rec,
                                                  materials, materialResources,
                                                  materialTextureInfos, meshInfos,
                                                  sceneVertices, meshIndices)) {
            break;
        }
        const float nextMinT = rec.t + kEpsilon;
        if (!isfinite(nextMinT) || nextMinT <= intersectionMinT ||
            nextMinT >= queuedRay.directionMaxT.w) {
            hit = false;
            break;
        }
        intersectionMinT = nextMinT;
    }
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
    (void)accel;
    (void)instanceUserIds;

    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    if (uniforms.explicitRestirDiLightingStage == 0u ||
        !(uniforms.directLightMode == kDirectLightModeRestirDi ||
          uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid)) {
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
    incomingRay.mediumState = queuedRay.mediumState;
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
            lightingThroughput *= exp(-sigma * max(rec.t + wavefront_source_distance_offset(queuedRay), 0.0f));
        }
    }

    uint rngState = state.rngState;
    const float3 incidentDir = safe_normalize(incomingRay.direction);
    const float3 wo = safe_normalize(-incomingRay.direction);
    uint materialIndex = uniforms.materialCount > 0u
        ? min(rec.materialIndex, uniforms.materialCount - 1u)
        : 0u;
    const MaterialData resolverMaterial = materials ? materials[materialIndex] : MaterialData{};
    RayCone rayCone = wavefront_ray_cone(queuedRay);
    if (!(rayCone.width > 0.0f) && !(rayCone.spread > 0.0f)) {
        rayCone = make_primary_ray_cone(uniforms);
    }
    WavefrontSurfaceState surfaceState = resolve_wavefront_surface_state(
        uniforms,
        rec,
        resolverMaterial,
        materialIndex,
        queuedRay.depth,
        (queuedRay.flags & kWavefrontRayFlagPavilionSmoothDifferentials) != 0u,
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
    if (surfaceState.alphaDiscarded || PT_DEFAULT_OFF(uniforms.debugViewMode) != kDebugViewNone) {
        return;
    }

    rec = surfaceState.rec;
    MaterialData material = materials ? materials[materialIndex] : MaterialData{};
    apply_resolved_surface_material(material, surfaceState);
    state.rngState = rngState;
    state.reserved1 = kWavefrontExplicitRestirDiSurfaceAccepted;
    pathStates[pathIndex] = state;
    if (!material_closure_direct_lighting_eligible(material)) {
        return;
    }

    const float3 shadingNormal = rec.shadingNormal;
    // The shade kernel's restirDiStageCovers reads the marked side; this stage's targets must use the same (EP8-f2).
    source_mark_closure_side(material, shadingNormal, wo);
    const float diffuseOcclusion = surfaceState.diffuseOcclusion;
    const bool specularOnly = (uniforms.debugSpecularOnly != 0u);
    FireflyClampParams clampParams = make_firefly_params(uniforms);

    Reservoir reservoir = make_empty_reservoir();
    uint risTargetM = max(uniforms.risCandidateCount, 1u);
    if (uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid &&
        uniforms.frameIndex > 0u) {
        risTargetM *= 2u;
    }
    build_restir_di_provider_reservoir_for_hit(uniforms,
                                               emissivePrimitives,
                                               rectangles,
                                               materials,
                                               environmentTexture,
                                               environmentConditionalAlias,
                                               environmentMarginalAlias,
                                               environmentPdf,
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
    if (stats && uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) {
        atomic_fetch_add_explicit(&stats->restirDiRegirCandidateCount,
                                  1u,
                                  memory_order_relaxed);
        if (reservoir.valid) {
            atomic_fetch_add_explicit(&stats->restirDiRegirAcceptedCount,
                                      1u,
                                      memory_order_relaxed);
        } else {
            atomic_fetch_add_explicit(&stats->restirDiRegirFallbackCount,
                                      1u,
                                      memory_order_relaxed);
        }
    }

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
        if (build_restir_di_provider_reservoir_for_hit(uniforms,
                                                       emissivePrimitives,
                                                       rectangles,
                                                       materials,
                                                       environmentTexture,
                                                       environmentConditionalAlias,
                                                       environmentMarginalAlias,
                                                       environmentPdf,
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
        queuedRay.depth == 0u &&
        rec.t > 0.0f &&
        uniforms.frameIndex > 0u) {
        int3 cacheCell = world_reuse_cell(rec.point, uniforms.worldReuseCellSize);
        uint cacheSourceFrameIndex =
            uniforms.frameIndex > 1u ? uniforms.frameIndex - 2u : uniforms.frameIndex - 1u;
        // The same merge as the fused shade kernel: the cached reservoirs share the pixel's builder and target at its
        // own hit point, so they stream in with their wSum and M grows by each probe's candidate count, valid or not.
        // This kernel builds every hybrid reservoir with risTargetM candidates (doubled above), and its temporal step
        // merges the previous reservoir only when both are valid. The pixel's reservoir enters with its own
        // contribution weight wSum / M and a fixed share of nominalM candidates: risTargetM, or twice that for the
        // hybrid, whose temporal step merges a second reservoir of risTargetM candidates when both are valid. Its
        // actual M depends on the samples there, so it cannot be its share: MIS weights that depend on the samples
        // would bias the merge.
        const uint nominalM = 2u * risTargetM;
        if (reservoir.valid && reservoir.M != nominalM) {
            reservoir.wSum *= float(nominalM) / float(reservoir.M);
        }
        uint mergedM = nominalM;
        uint candidateAttempts = 0u;
        uint accepted = 0u;
        for (uint cacheProbe = 0u; cacheProbe < 4u; ++cacheProbe) {
            if (cacheProbe > cacheSourceFrameIndex) {
                continue;
            }
            candidateAttempts += 1u;
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
            const bool cachedReservoirValid =
                build_restir_di_provider_reservoir_for_hit(uniforms,
                                                           emissivePrimitives,
                                                           rectangles,
                                                           materials,
                                                           environmentTexture,
                                                           environmentConditionalAlias,
                                                           environmentMarginalAlias,
                                                           environmentPdf,
                                                           rec,
                                                           material,
                                                           shadingNormal,
                                                           wo,
                                                           clampParams,
                                                           diffuseOcclusion,
                                                           specularOnly,
                                                           risTargetM,
                                                           cacheState,
                                                           cachedReservoir);
            mergedM += risTargetM;
            if (!cachedReservoirValid || !cachedReservoir.valid) {
                continue;
            }
            update_reservoir(reservoir, cachedReservoir.winner, cachedReservoir.wSum, rand_uniform(rngState));
            accepted += 1u;
        }
        if (reservoir.valid) {
            reservoir.M = mergedM;
        }
        if (stats && candidateAttempts > 0u) {
            atomic_fetch_add_explicit(&stats->restirDiRegirCandidateCount,
                                      candidateAttempts,
                                      memory_order_relaxed);
        }
        if (stats) {
            if (accepted > 0u) {
                atomic_fetch_add_explicit(&stats->restirDiRegirAcceptedCount,
                                          accepted,
                                          memory_order_relaxed);
            } else {
                atomic_fetch_add_explicit(&stats->restirDiRegirFallbackCount,
                                          1u,
                                          memory_order_relaxed);
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
            float cosThetaSurface = direct_light_target_cos(uniforms, material, rec, shadingNormal, omegaL);
            float cosThetaLight = winnerDirectional ? 1.0f : saturate(emissive_light_cosine(winner.flags, winner.normal, -omegaL));
            BsdfEvalResult eval = direct_light_target_bsdf(uniforms,
                                                           material,
                                                           rec,
                                                           shadingNormal,
                                                           wo,
                                                           omegaL,
                                                           clampParams,
                                                           diffuseOcclusion,
                                                           specularOnly,
                                                           winnerDirectional);
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
                        uint shadowFlags = winnerDirectional ? 0u : kWavefrontShadowFlagEmissivePrimitive;
                        if ((winner.flags & kLightPrimitiveFlagRectangle) != 0u) {
                            shadowFlags = kWavefrontShadowFlagRectangleLight;
                        }
                        ShadowRay shadow{};
                        shadow.originTMax = float4(offset_spawn_origin(rec, omegaL, material),
                                                   winnerDirectional
                                                       ? kInfinity
                                                       : max(distance - kEpsilon, kEpsilon));
                        shadow.direction = float4(omegaL, 0.0f);
                        shadow.contribution = float4(contribution, 0.0f);
                        shadow.throughput = float4(lightingThroughput, 0.0f);
                        shadow.mediumState =
                            volume_make_global_medium_state(uniforms,
                                                            state.mediumDepth,
                                                            state.volumeScatteringDepth);
                        shadow.pathIndex = pathIndex;
                        shadow.depth = queuedRay.depth;
                        shadow.lightId = winner.primitiveIndex;
                        shadow.flags = wavefront_shadow_receiver_flags(material, shadowFlags, winnerDirectional);
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

kernel void wavefrontShadeIndirectProbeKernel(
                                 constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                 device const WavefrontRay* activeRays [[buffer(1)]],
                                 device const WavefrontHitRecord* hitRecords [[buffer(2)]],
                                 device PathState* pathStates [[buffer(5)]],
                                 device QueueCounters* queueCounters [[buffer(6)]],
                                 uint gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    const uint activeCount =
        queueCounters ? atomic_load_explicit(&queueCounters->activeRayCount, memory_order_relaxed) : 0u;
    if (gid >= activeCount) {
        return;
    }

    if (gid == 0u && queueCounters) {
        uint observed = activeCount;
        if (activeRays && hitRecords && pathStates) {
            const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
            const WavefrontRay firstRay = activeRays[0];
            const uint pathIndex = min(firstRay.pathIndex, pixelCount - 1u);
            observed ^= (hitRecords[0].hit << 16u);
            observed ^= (pathStates[pathIndex].alive << 24u);
        }
        queueCounters->reserved1 = observed;
    }
}

kernel void wavefrontShadeSurfaceProbeKernel(
                                 constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                 device const WavefrontRay* activeRays [[buffer(1)]],
                                 device const WavefrontHitRecord* hitRecords [[buffer(2)]],
                                 device PathState* pathStates [[buffer(5)]],
                                 device QueueCounters* queueCounters [[buffer(6)]],
                                 device const MaterialData* materials [[buffer(8)]],
                                 device const MeshInfo* meshInfos [[buffer(14)]],
                                 device const SceneVertex* sceneVertices [[buffer(15)]],
                                 device const uint3* meshIndices [[buffer(16)]],
                                 device const MaterialTextureInfo* materialTextureInfos [[buffer(17)]],
                                 constant MaterialTextureArgumentBuffer& materialResources [[buffer(23)]],
                                 uint gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    const uint activeCount =
        queueCounters ? atomic_load_explicit(&queueCounters->activeRayCount, memory_order_relaxed) : 0u;
    if (gid >= activeCount || !activeRays || !hitRecords || !pathStates) {
        return;
    }

    const WavefrontRay queuedRay = activeRays[gid];
    const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
    if (queuedRay.pathIndex >= pixelCount) {
        return;
    }
    PathState state = pathStates[queuedRay.pathIndex];
    if (state.alive == 0u) {
        return;
    }

    const WavefrontHitRecord packedHit = hitRecords[gid];
    uint observed = activeCount ^ (packedHit.hit << 16u);
    if (packedHit.hit != 0u) {
        HitRecord rec = unpack_wavefront_hit(packedHit);
        if (!all(isfinite(rec.shadingNormal)) || dot(rec.shadingNormal, rec.shadingNormal) <= 0.0f) {
            rec.shadingNormal = rec.normal;
        }
        rec.shadingNormal = safe_normalize(rec.shadingNormal);
        Ray incomingRay;
        incomingRay.origin = queuedRay.originMinT.xyz;
        incomingRay.direction = queuedRay.directionMaxT.xyz;
        incomingRay.mediumState = queuedRay.mediumState;
        uint materialIndex = rec.materialIndex;
        if (uniforms.materialCount > 0u) {
            materialIndex = min(materialIndex, uniforms.materialCount - 1u);
        } else {
            materialIndex = 0u;
        }
        const MaterialData resolverMaterial = materials ? materials[materialIndex] : MaterialData{};
        RayCone rayCone = wavefront_ray_cone(queuedRay);
        if (!(rayCone.width > 0.0f) && !(rayCone.spread > 0.0f)) {
            rayCone = make_primary_ray_cone(uniforms);
        }
        const PrimaryRayDiff primaryRayDiff = wavefront_primary_ray_diff(queuedRay);
        WavefrontSurfaceState surfaceState = resolve_wavefront_surface_state(
            uniforms,
            rec,
            resolverMaterial,
            materialIndex,
            queuedRay.depth,
            (queuedRay.flags & kWavefrontRayFlagPavilionSmoothDifferentials) != 0u,
            incomingRay,
            primaryRayDiff,
            incomingRay.direction,
            ray_segment_world_length(incomingRay, rec.t),
            rayCone,
            state.rngState,
            false,
            materialResources.textures,
            materialResources.samplers,
            materialTextureInfos,
            meshInfos,
            sceneVertices,
            meshIndices);
        observed ^= (surfaceState.alphaDiscarded ? 0x80000000u : 0u);
        observed ^= ((uint(surfaceState.debugMetallic * 255.0f) & 0xffu) << 8u);
        observed ^= (materialIndex & 0xffu);
    }

    if (gid == 0u && queueCounters) {
        queueCounters->reserved1 = observed;
    }
}

kernel void wavefrontShadeResolveSurfaceKernel(
                                 constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                 device const WavefrontRay* activeRays [[buffer(1)]],
                                 device const WavefrontHitRecord* hitRecords [[buffer(2)]],
                                 device PathState* pathStates [[buffer(5)]],
                                 device QueueCounters* queueCounters [[buffer(6)]],
                                 device const MaterialData* materials [[buffer(8)]],
                                 device const MeshInfo* meshInfos [[buffer(14)]],
                                 device const SceneVertex* sceneVertices [[buffer(15)]],
                                 device const uint3* meshIndices [[buffer(16)]],
                                 device const MaterialTextureInfo* materialTextureInfos [[buffer(17)]],
                                 constant MaterialTextureArgumentBuffer& materialResources [[buffer(23)]],
                                 device WavefrontResolvedSurface* resolvedSurfaces [[buffer(30)]],
                                 uint gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    const uint activeCount =
        queueCounters ? atomic_load_explicit(&queueCounters->activeRayCount, memory_order_relaxed) : 0u;
    if (gid >= activeCount || !activeRays || !hitRecords || !pathStates || !resolvedSurfaces) {
        return;
    }

    WavefrontResolvedSurface invalid{};
    resolvedSurfaces[gid] = invalid;

    const WavefrontRay queuedRay = activeRays[gid];
    const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
    if (queuedRay.pathIndex >= pixelCount) {
        return;
    }
    PathState state = pathStates[queuedRay.pathIndex];
    if (state.alive == 0u) {
        return;
    }

    const WavefrontHitRecord packedHit = hitRecords[gid];
    if (packedHit.hit == 0u) {
        return;
    }

    HitRecord rec = unpack_wavefront_hit(packedHit);
    if (!all(isfinite(rec.shadingNormal)) || dot(rec.shadingNormal, rec.shadingNormal) <= 0.0f) {
        rec.shadingNormal = rec.normal;
    }
    rec.shadingNormal = safe_normalize(rec.shadingNormal);
    Ray incomingRay;
    incomingRay.origin = queuedRay.originMinT.xyz;
    incomingRay.direction = queuedRay.directionMaxT.xyz;
    incomingRay.mediumState = queuedRay.mediumState;

    uint materialIndex = rec.materialIndex;
    if (uniforms.materialCount > 0u) {
        materialIndex = min(materialIndex, uniforms.materialCount - 1u);
    } else {
        materialIndex = 0u;
    }
    const MaterialData resolverMaterial = materials ? materials[materialIndex] : MaterialData{};
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
    thread uint rngState = state.rngState;
    WavefrontSurfaceState surfaceState = resolve_wavefront_surface_state(
        uniforms,
        rec,
        resolverMaterial,
        materialIndex,
        queuedRay.depth,
        (queuedRay.flags & kWavefrontRayFlagPavilionSmoothDifferentials) != 0u,
        incomingRay,
        wavefront_primary_ray_diff(queuedRay),
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

    resolvedSurfaces[gid] = pack_resolved_surface(surfaceState,
                                                  queuedRay.pathIndex,
                                                  materialIndex,
                                                  rngState);
}

kernel void wavefrontShadeMetadataProbeKernel(
                                 constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                 device const WavefrontRay* activeRays [[buffer(1)]],
                                 device const WavefrontHitRecord* hitRecords [[buffer(2)]],
                                 device PathState* pathStates [[buffer(5)]],
                                 device QueueCounters* queueCounters [[buffer(6)]],
                                 device const MaterialData* materials [[buffer(8)]],
                                 device const MeshInfo* meshInfos [[buffer(14)]],
                                 device const SceneVertex* sceneVertices [[buffer(15)]],
                                 device const uint3* meshIndices [[buffer(16)]],
                                 device const MaterialTextureInfo* materialTextureInfos [[buffer(17)]],
                                 constant MaterialTextureArgumentBuffer& materialResources [[buffer(23)]],
                                 uint gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    const uint activeCount =
        queueCounters ? atomic_load_explicit(&queueCounters->activeRayCount, memory_order_relaxed) : 0u;
    if (gid >= activeCount || !activeRays || !hitRecords || !pathStates) {
        return;
    }

    const WavefrontRay queuedRay = activeRays[gid];
    const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
    if (queuedRay.pathIndex >= pixelCount) {
        return;
    }
    PathState state = pathStates[queuedRay.pathIndex];
    if (state.alive == 0u) {
        return;
    }

    const WavefrontHitRecord packedHit = hitRecords[gid];
    uint observed = activeCount ^ (packedHit.hit << 16u);
    if (packedHit.hit != 0u) {
        HitRecord rec = unpack_wavefront_hit(packedHit);
        if (!all(isfinite(rec.shadingNormal)) || dot(rec.shadingNormal, rec.shadingNormal) <= 0.0f) {
            rec.shadingNormal = rec.normal;
        }
        rec.shadingNormal = safe_normalize(rec.shadingNormal);
        Ray incomingRay;
        incomingRay.origin = queuedRay.originMinT.xyz;
        incomingRay.direction = queuedRay.directionMaxT.xyz;
        incomingRay.mediumState = queuedRay.mediumState;
        uint materialIndex = rec.materialIndex;
        if (uniforms.materialCount > 0u) {
            materialIndex = min(materialIndex, uniforms.materialCount - 1u);
        } else {
            materialIndex = 0u;
        }
        const MaterialData material = materials ? materials[materialIndex] : MaterialData{};
        const MaterialClosureMetadata primaryClosure =
            material_primary_closure_metadata(material);
        observed ^= ((primaryClosure.type & 0xffu) << 8u);
        observed ^= ((primaryClosure.flags & 0xffu) << 16u);
        observed ^= ((uint(primaryClosure.roughness * 255.0f) & 0xffu) << 24u);
    }

    if (gid == 0u && queueCounters) {
        queueCounters->reserved1 = observed;
    }
}

kernel void wavefrontShadeScalarProbeKernel(
                                 constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                 device const WavefrontRay* activeRays [[buffer(1)]],
                                 device const WavefrontHitRecord* hitRecords [[buffer(2)]],
                                 device PathState* pathStates [[buffer(5)]],
                                 device QueueCounters* queueCounters [[buffer(6)]],
                                 device const MaterialData* materials [[buffer(8)]],
                                 device const MeshInfo* meshInfos [[buffer(14)]],
                                 device const SceneVertex* sceneVertices [[buffer(15)]],
                                 device const uint3* meshIndices [[buffer(16)]],
                                 device const MaterialTextureInfo* materialTextureInfos [[buffer(17)]],
                                 constant MaterialTextureArgumentBuffer& materialResources [[buffer(23)]],
                                 uint gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    const uint activeCount =
        queueCounters ? atomic_load_explicit(&queueCounters->activeRayCount, memory_order_relaxed) : 0u;
    if (gid >= activeCount || !activeRays || !hitRecords || !pathStates) {
        return;
    }

    const WavefrontRay queuedRay = activeRays[gid];
    const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
    if (queuedRay.pathIndex >= pixelCount) {
        return;
    }
    PathState state = pathStates[queuedRay.pathIndex];
    if (state.alive == 0u) {
        return;
    }

    const WavefrontHitRecord packedHit = hitRecords[gid];
    uint observed = activeCount ^ (packedHit.hit << 16u);
    if (packedHit.hit != 0u) {
        HitRecord rec = unpack_wavefront_hit(packedHit);
        if (!all(isfinite(rec.shadingNormal)) || dot(rec.shadingNormal, rec.shadingNormal) <= 0.0f) {
            rec.shadingNormal = rec.normal;
        }
        rec.shadingNormal = safe_normalize(rec.shadingNormal);
        Ray incomingRay;
        incomingRay.origin = queuedRay.originMinT.xyz;
        incomingRay.direction = queuedRay.directionMaxT.xyz;
        incomingRay.mediumState = queuedRay.mediumState;
        uint materialIndex = rec.materialIndex;
        if (uniforms.materialCount > 0u) {
            materialIndex = min(materialIndex, uniforms.materialCount - 1u);
        } else {
            materialIndex = 0u;
        }
        const MaterialData material = materials ? materials[materialIndex] : MaterialData{};
        const uint materialType = static_cast<uint>(material.typeEta.x);
        const float roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        const float emissionWeight = luminance_rgb(max(material.emission.xyz, float3(0.0f)));
        observed ^= ((materialType & 0xffu) << 8u);
        observed ^= ((uint(roughness * 255.0f) & 0xffu) << 16u);
        observed ^= (emissionWeight > 0.0f ? 0x40000000u : 0u);
    }

    if (gid == 0u && queueCounters) {
        queueCounters->reserved1 = observed;
    }
}

kernel void wavefrontShadeBsdfProbeKernel(
                                 constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                 device const WavefrontRay* activeRays [[buffer(1)]],
                                 device const WavefrontHitRecord* hitRecords [[buffer(2)]],
                                 device PathState* pathStates [[buffer(5)]],
                                 device QueueCounters* queueCounters [[buffer(6)]],
                                 device const MaterialData* materials [[buffer(8)]],
                                 device const MeshInfo* meshInfos [[buffer(14)]],
                                 device const SceneVertex* sceneVertices [[buffer(15)]],
                                 device const uint3* meshIndices [[buffer(16)]],
                                 device const MaterialTextureInfo* materialTextureInfos [[buffer(17)]],
                                 constant MaterialTextureArgumentBuffer& materialResources [[buffer(23)]],
                                 uint gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    const uint activeCount =
        queueCounters ? atomic_load_explicit(&queueCounters->activeRayCount, memory_order_relaxed) : 0u;
    if (gid >= activeCount || !activeRays || !hitRecords || !pathStates) {
        return;
    }

    const WavefrontRay queuedRay = activeRays[gid];
    const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
    if (queuedRay.pathIndex >= pixelCount) {
        return;
    }
    PathState state = pathStates[queuedRay.pathIndex];
    if (state.alive == 0u) {
        return;
    }

    const WavefrontHitRecord packedHit = hitRecords[gid];
    uint observed = activeCount ^ (packedHit.hit << 16u);
    if (packedHit.hit != 0u) {
        HitRecord rec = unpack_wavefront_hit(packedHit);
        if (!all(isfinite(rec.shadingNormal)) || dot(rec.shadingNormal, rec.shadingNormal) <= 0.0f) {
            rec.shadingNormal = rec.normal;
        }
        rec.shadingNormal = safe_normalize(rec.shadingNormal);
        Ray incomingRay;
        incomingRay.origin = queuedRay.originMinT.xyz;
        incomingRay.direction = queuedRay.directionMaxT.xyz;
        incomingRay.mediumState = queuedRay.mediumState;
        uint materialIndex = rec.materialIndex;
        if (uniforms.materialCount > 0u) {
            materialIndex = min(materialIndex, uniforms.materialCount - 1u);
        } else {
            materialIndex = 0u;
        }
        thread uint rngState = state.rngState;
        {
            MaterialData material = materials ? materials[materialIndex] : MaterialData{};
            const float3 incidentDir = safe_normalize(incomingRay.direction);
            const float3 wo = safe_normalize(-incomingRay.direction);
            FireflyClampParams clampParams = make_firefly_params(uniforms);
            const SpectralPathState spectralState =
                spectral_path_state_from_hero(uniforms, state.spectralWavelengthNm, state.spectralSampleIndex);
            spectral_apply_material(material, uniforms, spectralState, nullptr);
            const MaterialClosureMetadata primaryClosure =
                material_primary_closure_metadata(material);
            const bool specularOnly = (uniforms.debugSpecularOnly != 0u);
            BsdfSampleResult bsdfSample = material_closure_sample(material,
                                                                  rec.point,
                                                                  rec.shadingNormal,
                                                                  wo,
                                                                  incidentDir,
                                                                  rec.frontFace != 0u,
                                                                  rngState,
                                                                  clampParams,
                                                                  PT_DEFAULT_OFF(uniforms.sssMode),
                                                                  1.0f,
                                                                  specularOnly);
            observed ^= ((primaryClosure.type & 0xffu) << 8u);
            observed ^= ((bsdfSample.lobeType & 0xffu) << 16u);
            observed ^= (bsdfSample.pdf > 0.0f ? 0x40000000u : 0u);
            observed ^= (all(isfinite(bsdfSample.weight)) ? 0x20000000u : 0u);
        }
    }

    if (gid == 0u && queueCounters) {
        queueCounters->reserved1 = observed;
    }
}

kernel void wavefrontShadeClosureProbeKernel(
                                 constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                 device const WavefrontRay* activeRays [[buffer(1)]],
                                 device const WavefrontHitRecord* hitRecords [[buffer(2)]],
                                 device PathState* pathStates [[buffer(5)]],
                                 device QueueCounters* queueCounters [[buffer(6)]],
                                 device const MaterialData* materials [[buffer(8)]],
                                 device const MeshInfo* meshInfos [[buffer(14)]],
                                 device const SceneVertex* sceneVertices [[buffer(15)]],
                                 device const uint3* meshIndices [[buffer(16)]],
                                 device const MaterialTextureInfo* materialTextureInfos [[buffer(17)]],
                                 constant MaterialTextureArgumentBuffer& materialResources [[buffer(23)]],
                                 uint gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    const uint activeCount =
        queueCounters ? atomic_load_explicit(&queueCounters->activeRayCount, memory_order_relaxed) : 0u;
    if (gid >= activeCount || !activeRays || !hitRecords || !pathStates) {
        return;
    }

    const WavefrontRay queuedRay = activeRays[gid];
    const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
    if (queuedRay.pathIndex >= pixelCount) {
        return;
    }
    PathState state = pathStates[queuedRay.pathIndex];
    if (state.alive == 0u) {
        return;
    }

    const WavefrontHitRecord packedHit = hitRecords[gid];
    uint observed = activeCount ^ (packedHit.hit << 16u);
    if (packedHit.hit != 0u) {
        HitRecord rec = unpack_wavefront_hit(packedHit);
        if (!all(isfinite(rec.shadingNormal)) || dot(rec.shadingNormal, rec.shadingNormal) <= 0.0f) {
            rec.shadingNormal = rec.normal;
        }
        rec.shadingNormal = safe_normalize(rec.shadingNormal);
        Ray incomingRay;
        incomingRay.origin = queuedRay.originMinT.xyz;
        incomingRay.direction = queuedRay.directionMaxT.xyz;
        incomingRay.mediumState = queuedRay.mediumState;
        uint materialIndex = rec.materialIndex;
        if (uniforms.materialCount > 0u) {
            materialIndex = min(materialIndex, uniforms.materialCount - 1u);
        } else {
            materialIndex = 0u;
        }
        const MaterialData resolverMaterial = materials ? materials[materialIndex] : MaterialData{};
        RayCone rayCone = wavefront_ray_cone(queuedRay);
        if (!(rayCone.width > 0.0f) && !(rayCone.spread > 0.0f)) {
            rayCone = make_primary_ray_cone(uniforms);
        }
        const PrimaryRayDiff primaryRayDiff = wavefront_primary_ray_diff(queuedRay);
        thread uint rngState = state.rngState;
        WavefrontSurfaceState surfaceState = resolve_wavefront_surface_state(
            uniforms,
            rec,
            resolverMaterial,
            materialIndex,
            queuedRay.depth,
            (queuedRay.flags & kWavefrontRayFlagPavilionSmoothDifferentials) != 0u,
            incomingRay,
            primaryRayDiff,
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
        if (!surfaceState.alphaDiscarded) {
            MaterialData material = materials ? materials[materialIndex] : MaterialData{};
            apply_resolved_surface_material(material, surfaceState);
            const SpectralPathState spectralState =
                spectral_path_state_from_hero(uniforms, state.spectralWavelengthNm, state.spectralSampleIndex);
            spectral_apply_material(material, uniforms, spectralState, nullptr);
            const MaterialClosureMetadata primaryClosure =
                material_primary_closure_metadata(material);
            observed ^= ((primaryClosure.type & 0xffu) << 8u);
            observed ^= (material_closure_is_delta(material) ? 0x40000000u : 0u);
            observed ^= (any(material_closure_aov_albedo(material) > float3(0.0f))
                             ? 0x20000000u
                             : 0u);
        } else {
            observed ^= 0x80000000u;
        }
    }

    if (gid == 0u && queueCounters) {
        queueCounters->reserved1 = observed;
    }
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

    uint2 pixel = wavefront_pixel_coord(queuedRay.pixelIndex, uniforms);
    PathtraceDebugContext debugCtx = make_debug_context(uniforms,
                                                        debugBuffer,
                                                        pixel,
                                                        uniforms.sampleCount,
                                                        2u);
    thread PathtraceDebugContext* debugCtxPtr = nullptr;
#if PT_DEBUG_TOOLS
    debugCtxPtr = (debugBuffer &&
                   (PT_DEFAULT_OFF(uniforms.debugPathActive) != 0u ||
                    PT_DEFAULT_OFF(uniforms.debugDirectLightAudit) != 0u))
        ? &debugCtx
        : nullptr;
#endif

    thread uint rngState = state.rngState;
    FireflyClampParams clampParams = make_firefly_params(uniforms);

    const WavefrontHitRecord packedHit = hitRecords[gid];
    Ray incomingRay;
    incomingRay.origin = queuedRay.originMinT.xyz;
    incomingRay.direction = queuedRay.directionMaxT.xyz;
    incomingRay.mediumState = queuedRay.mediumState;

    if (packedHit.hit == 0u) {
        if (PT_DEFAULT_OFF(uniforms.debugViewMode) != kDebugViewNone) {
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
            if (useOverride) {
                background = environment_color_lod(environmentTexture,
                                                   incomingRay.direction,
                                                   uniforms.environmentRotation,
                                                   uniforms.environmentIntensity,
                                                   overrideLod,
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
        {
            const SpectralPathState missSpectral = spectral_path_state_from_hero(
                uniforms, state.spectralWavelengthNm, state.spectralSampleIndex);
            background = spectral_radiance_to_channels(background, missSpectral);
        }
        float misWeight = 1.0f;
        bool useSpecularMis =
            use_visible_emitter_mis(queuedRay.depth,
                                    (state.flags & kWavefrontFlagLastScatterDelta) != 0u,
                                    (state.flags & kWavefrontFlagLastDeltaUnpaired) != 0u,
                                    uniforms);
        // Scatter points sample no environment light, so an escape after one keeps weight 1.
        if (useSpecularMis && (state.flags & kWavefrontFlagLastScatterVolume) == 0u &&
            environment_sampling_available(uniforms,
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
        float3 contribution = background * misWeight;
        if (queuedRay.depth == 0u && uniforms.backgroundMode == 2u &&
            environmentTexture.get_width() > 0u && environmentTexture.get_height() > 0u &&
            uniforms.environmentBackgroundScaleOffset != 0.0f) {
            contribution *= max(1.0f + uniforms.environmentBackgroundScaleOffset, 0.0f);
        }
        wavefront_add_reconstruction_radiance(
            state,
            queuedRay.depth,
            clamp_firefly_contribution(state.throughput.xyz,
                                       contribution,
                                       clampParams,
                                       stats));
        // Keep the physical HDR radiance for light endpoints and PDFs.
        caustic_transport_note_light_endpoint(
            uniforms,
            incomingRay.origin + safe_normalize(incomingRay.direction),
            -incomingRay.direction,
            incomingRay.direction,
            background,
            environment_sampling_available(uniforms,
                                           environmentConditionalAlias,
                                           environmentMarginalAlias,
                                           environmentPdf)
                ? environment_pdf(uniforms, environmentPdf, incomingRay.direction)
                : 0.0f,
            state.lastBsdfPdf,
            queuedRay.depth,
            state.currentMediumId,
            materialResources.causticLightVertices,
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
        const float3 segmentTransmittance = volume_transmittance(medium, segment, stats);
        if (PT_DEFAULT_OFF(uniforms.volumeDebug) != 0u) {
            state.firstHitMaterial =
                volume_debug_aov_features(segmentTransmittance,
                                          state.volumeScatteringDepth,
                                          volume_phase_id(uniforms),
                                          segmentEmission);
        }
        if (any(segmentEmission > float3(0.0f))) {
            wavefront_add_reconstruction_radiance(
                state,
                queuedRay.depth,
                clamp_firefly_contribution(state.throughput.xyz,
                                           segmentEmission,
                                           clampParams,
                                           stats));
        }
        const uint maxVolumeEvents = max(medium.metadata.y, 1u);
        bool sampledFreeFlight = false;
        float3 freeFlightWeight = float3(1.0f);
        if (state.volumeScatteringDepth < maxVolumeEvents &&
            any(medium.sigmaSAnisotropy.xyz > float3(0.0f))) {
            const VolumeFreeFlight flight = volume_sample_free_flight(medium, segment, rngState);
            const float sampledDistance = flight.distance;
            sampledFreeFlight = true;
            freeFlightWeight = flight.weight;
            if (flight.scattered) {
                const float3 scatterPoint = incomingRay.origin + incomingRay.direction * sampledDistance;
                const float3 transmittance = volume_transmittance(medium, sampledDistance, stats);
                if (PT_DEFAULT_OFF(uniforms.volumeDebug) != 0u) {
                    state.firstHitMaterial =
                        volume_debug_aov_features(transmittance,
                                                  state.volumeScatteringDepth + 1u,
                                                  volume_phase_id(uniforms),
                                                  segmentEmission);
                }
                // Free-flight weight sigma_s T(t) / mean(sigma_t T(t)) (volume_sample_free_flight).
                state.throughput.xyz *= freeFlightWeight;
                bool volumeRectNee = false;
                if (rectangles && materials && medium.metadata.z != 0u) {
                    const uint rectLightCount = count_rect_lights(uniforms, rectangles, materials);
                    volumeRectNee = rectLightCount > 0u;
                    if (rectLightCount > 0u) {
                        HitRecord volumeRec = rec;
                        volumeRec.point = scatterPoint;
                        for (uint neeDraw = 0u; neeDraw < rect_light_nee_draw_count(uniforms); ++neeDraw) {
                            RectLightSample lightSample;
                            if (sample_rect_light_nee_draw(uniforms,
                                                  rectangles,
                                                  materials,
                                                  environmentTexture,
                                                  volumeRec,
                                                  rngState,
                                                  rectLightCount,
                                                  lightSample, kAnalyticLightFilterAll, neeDraw) &&
                                lightSample.pdf > 0.0f) {
                                const float phase = volume_phase_eval(medium,
                                                                      -incomingRay.direction,
                                                                      lightSample.direction);
                                const float shadowMax = max(lightSample.distance - kEpsilon, kEpsilon);
                                const float3 shadowTr = volume_transmittance(medium, shadowMax, stats);
                                // The shadow kernel does not multiply by the path throughput, so the queued contribution
                                // carries it, as surface NEE's does and as the megakernels' volume NEE applies it.
                                // MIS against the phase-sampled hit of the same light (volume_nee_mis_weight).
                                const float misWeight = lightSample.isReferencePoint
                                    ? 1.0f : volume_nee_mis_weight(lightSample.pdf, phase);
                                const float3 contribution = clamp_firefly_contribution(
                                    state.throughput.xyz,
                                    lightSample.emission * phase * shadowTr * misWeight / max(lightSample.pdf, 1.0e-6f),
                                    clampParams);
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
                                    shadow.mediumState =
                                        volume_make_global_medium_state(uniforms,
                                                                        state.mediumDepth,
                                                                        state.volumeScatteringDepth + 1u);
                                    shadow.pathIndex = pathIndex;
                                    shadow.depth = queuedRay.depth;
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
                    nextRay.mediumState =
                        volume_make_global_medium_state(uniforms,
                                                        state.mediumDepth,
                                                        state.volumeScatteringDepth + 1u);
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
                    state.flags = kWavefrontFlagLastScatterVolume | (volumeRectNee ? kWavefrontFlagVolumeRectNee : 0u);
                    // A scatter is a real vertex: a later hole keeps this point, not an earlier one, as its MIS origin.
                    state.sourceNullOrigin.w = 0.0f;
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
        // Passed the segment: the free-flight weight T(s) / mean(T(s)) when a distance was sampled, else T(s).
        state.throughput.xyz *= sampledFreeFlight ? freeFlightWeight : segmentTransmittance;
        state.currentMediumId = medium.metadata.x;
    }
    if (mediumDepth > 0u) {
        float3 sigma = state.mediumStack[mediumDepth - 1u].xyz;
        if (any(sigma > float3(0.0f))) {
            float segment = max(rec.t + wavefront_source_distance_offset(queuedRay), 0.0f);
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
    bool explicitRestirDiLightingStage =
        uniforms.explicitRestirDiLightingStage != 0u &&
        (uniforms.directLightMode == kDirectLightModeRestirDi ||
         uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid);
    bool explicitRestirDiSurfaceAccepted =
        explicitRestirDiLightingStage &&
        state.reserved1 == kWavefrontExplicitRestirDiSurfaceAccepted;
    RayCone rayCone = wavefront_ray_cone(queuedRay);
    if (!(rayCone.width > 0.0f) && !(rayCone.spread > 0.0f)) {
        rayCone = make_primary_ray_cone(uniforms);
    }
    WavefrontSurfaceState surfaceState{};
    if (kWavefrontShadeUseResolvedSurface) {
        device const WavefrontResolvedSurface* resolvedSurfaces =
            reinterpret_cast<device const WavefrontResolvedSurface*>(materialTextureInfos);
        if (!resolvedSurfaces) {
            return;
        }
        const WavefrontResolvedSurface packedSurface = resolvedSurfaces[gid];
        if (packedSurface.meta.x == 0u) {
            return;
        }
        surfaceState = unpack_resolved_surface(packedSurface);
        rec = surfaceState.rec;
        materialIndex = packedSurface.meta.z;
        rngState = packedSurface.meta.w;
    } else {
        const MaterialData resolverMaterial = materials ? materials[materialIndex] : MaterialData{};
        PrimaryRayDiff primaryRayDiff = wavefront_primary_ray_diff(queuedRay);
        surfaceState = resolve_wavefront_surface_state(
            uniforms,
            rec,
            resolverMaterial,
            materialIndex,
            queuedRay.depth,
            (queuedRay.flags & kWavefrontRayFlagPavilionSmoothDifferentials) != 0u,
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
    }
    if (surfaceState.alphaDiscarded) {
        uint nextIndex =
            atomic_fetch_add_explicit(&queueCounters->nextRayCount, 1u, memory_order_relaxed);
        if (stats) {
            atomic_fetch_add_explicit(&stats->wavefrontNextRayEnqueueCount, 1u, memory_order_relaxed);
        }
        if (nextIndex < max(uniforms.width * uniforms.height, 1u)) {
            wavefront_note_queue_enqueue(queueCounters->diffuseRayQueue);
            WavefrontRay nextRay{};
            const bool sourceNull = source_has_opacity(materials[materialIndex]) ||
                (source_transport_enabled(uniforms) && source_is_thin(materials[materialIndex]));
            // Every alpha discard, a PBR MASK hole as well as a source null, is a null event for MIS: the last real
            // vertex's BSDF pdf, delta flags and origin stay, so the next emitter hit or miss takes the BSDF side of
            // the balance heuristic whose light side that vertex's NEE took (its alpha-aware shadow rays pass MASK
            // holes, and BLEND texels with the path's probability 1 - alpha; the specular-NEE ones included). A BLEND
            // pass is such a discard too.
            if (state.sourceNullOrigin.w == 0.0f)
                state.sourceNullOrigin = float4(incomingRay.origin,1.0f);
            nextRay.originMinT = float4(offset_ray_origin(rec, incomingRay.direction), kEpsilon);
            nextRay.directionMaxT = float4(incomingRay.direction, kInfinity);
            nextRay.mediumState =
                volume_make_global_medium_state(uniforms,
                                                state.mediumDepth,
                                                state.volumeScatteringDepth);
            wavefront_store_primary_ray_diff(nextRay, wavefront_primary_ray_diff(queuedRay));
            wavefront_store_ray_cone(nextRay, wavefront_ray_cone(queuedRay));
            nextRay.pixelIndex = queuedRay.pixelIndex;
            nextRay.pathIndex = pathIndex;
            nextRay.depth = queuedRay.depth + 1u;
            if (sourceNull) {
                state.throughput.xyz *= surfaceState.sourceForwardWeight;
                nextRay.mediumState = queuedRay.mediumState;
                // Preserve the original ray line and exact segment endpoints.
                nextRay.originMinT = float4(incomingRay.origin,source_forward_min_t(incomingRay,rec.t,source_ray_epsilon(uniforms)));
                nextRay.rayDiffDOdx.w = -rec.t;
            }
            nextRay.flags = sourceNull ? queuedRay.flags :
                (queuedRay.flags & ~kWavefrontRayFlagPavilionSmoothDifferentials);
            // A camera null has no BSDF/NEE competitor, even after depth advances.
            if (queuedRay.depth == 0u) {
                const uint cameraFlags = kWavefrontFlagLastScatterDelta | kWavefrontFlagLastDeltaUnpaired;
                nextRay.flags |= cameraFlags;
                state.flags |= cameraFlags;
            }
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
    float3 incomingMisOrigin = incomingRay.origin;
    if (state.sourceNullOrigin.w != 0.0f) incomingMisOrigin = state.sourceNullOrigin.xyz;
    state.sourceNullOrigin.w = 0.0f;
    MaterialData material = materials ? materials[materialIndex] : MaterialData{};
    apply_resolved_surface_material(material, surfaceState);
    regularize_dielectric_material(material, rec.frontFace != 0u, state.pathMaxRoughness);
    if (static_cast<uint>(material.typeEta.x) == 7u &&
        (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u) {
        // Surface resolution may already have converted a back-face IOR to
        // its reciprocal; the shared helper expects the authored IOR.
        material.typeEta.y = max(material.typeEta.y,
                                  1.0f / max(material.typeEta.y, 1.0e-6f));
        regularize_pavilion_pbr_material(material, state.pathMaxRoughness,
                                          rec.frontFace != 0u);
    }
    SpectralPathState spectralState =
        spectral_path_state_from_hero(uniforms, state.spectralWavelengthNm, state.spectralSampleIndex);
    spectralState.collapsed = (state.spectralFlags & kSpectralFlagCollapsed) != 0u;
    // Denoiser guides stay RGB: capture them before the material becomes per-wavelength.
    const float3 spectralGuideAlbedo =
        spectralState.enabled ? material_closure_aov_albedo(material) : float3(0.0f);
    const float4 spectralGuideFeatures =
        spectralState.enabled ? material_closure_aov_features(material) : float4(0.0f);
    spectral_apply_material(material, uniforms, spectralState, stats);
    if (spectralState.enabled && !spectralState.collapsed) {
        float3 spectralThroughput = state.throughput.xyz;
        spectral_collapse_at_dispersive_vertex(spectralState, material, uniforms, spectralThroughput, stats);
        if (spectralState.collapsed) {
            state.throughput.xyz = spectralThroughput;
            state.spectralFlags |= kSpectralFlagCollapsed;
        }
    }
    if (source_transport_enabled(uniforms) && source_is_thin(material)) material.sourceFresnel.z = 2.0f;
    const uint type = static_cast<uint>(material.typeEta.x);
    MaterialClosureMetadata primaryClosure = material_primary_closure_metadata(material);
    uint closureType = primaryClosure.type;
    float diffuseOcclusion = surfaceState.diffuseOcclusion;
    float3 shadingNormal = rec.shadingNormal;
    source_mark_closure_side(material, shadingNormal, -incomingRay.direction);
    const float3 debugBaseColor = material_base_color(material);
    const float debugMetallic = surfaceState.debugMetallic;
    const float debugRoughness = surfaceState.debugRoughness;
    const float debugAO = surfaceState.debugAO;
    const float transmissionValue = material.pbrExtras.z;
    const uint rectLightCount = (rectangles && materials && uniforms.rectangleCount > 0u)
        ? count_rect_lights(uniforms, rectangles, materials)
        : 0u;
    const bool specularOnly = (uniforms.debugSpecularOnly != 0u);

    const bool blackBackFace =
        (material.materialFlags & kMaterialFlagBlackBackFace) != 0u && rec.frontFace == 0u;
    // OIDN recommends guides at the first non-delta hit behind a mirror.
    // A source smooth conductor has deterministic reflection: defer only its
    // albedo/normal guides, with no radiance, sampling or primary-AOV changes.
    const bool followMirrorGuide = (uniforms.oidnDeltaGuidesAllowed & 2u) != 0u &&
        !blackBackFace && source_model(material) == 1.0f && material.sourceParams.z == 0.0f;
    // Same for the deterministic-interface source dielectrics (smooth glass and thin
    // sheets, the delta lobe of source_sample): the guides come from the first
    // non-delta vertex of the sampled path, whether it took the reflection or the
    // refraction branch, so they describe what the color shows through the glass
    // (a constant glass albedo and the glass normal would otherwise hide it).
    const bool followTransmissionGuide = (uniforms.oidnDeltaGuidesAllowed & 4u) != 0u &&
        !blackBackFace && source_model(material) == 3.0f &&
        (material.sourceParams.z == 0.0f || material.typeEta.w > 0.5f || material.sourceFresnel.w == 1.0f);
    // As in the megakernel, a Pavilion single-scatter specular primary hit (water,
    // glass) moves the guides to the first non-delta vertex behind it; a negative
    // albedo tag marks a guide still pending behind such delta surfaces.
    if (state.firstHitPosition.w <= 0.0f && uniforms.oidnDeltaGuidesAllowed != 0u &&
        (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u) {
        state.firstHitAlbedo.w = -1.0f;
    }
    const bool followSpecularGuide = state.firstHitAlbedo.w < 0.0f && material_closure_is_delta(material);
    // Falcor's denoiser graph uses primary normal guides. Reflected foliage
    // guides preserve its high-frequency structure on otherwise smooth glass.
    const bool setFirstHitGuides = state.firstHitAlbedo.w <= 0.0f && !followMirrorGuide && !followTransmissionGuide &&
        !followSpecularGuide;
    if (setFirstHitGuides) {
        state.firstHitAlbedo = float4(blackBackFace ? float3(0.0f)
                                                    : (spectralState.enabled ? spectralGuideAlbedo
                                                                             : material_closure_aov_albedo(material)),
                                      1.0f);
        state.firstHitNormal = float4(surfaceState.firstHitNormal, 1.0f);
    }
    // Position and material features are taken at the first hit even when a guide defers the albedo/normal.
    if (state.firstHitPosition.w <= 0.0f) {
        state.firstHitPosition = float4(surfaceState.rec.point, 1.0f);
        state.firstHitMaterial = spectralState.enabled ? spectralGuideFeatures
                                                       : material_closure_aov_features(material);
    }
    if (blackBackFace && PT_DEFAULT_OFF(uniforms.debugViewMode) == kDebugViewNone) {
        state.alive = 0u;
        wavefront_note_termination(queueCounters, stats, 2u);
        state.rngState = rngState;
        pathStates[pathIndex] = state;
        return;
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

    if (debugCtxPtr &&
        PT_DEFAULT_OFF(uniforms.debugDirectLightAudit) != 0u &&
        (rectLightCount > 0u || uniforms.emissivePrimitiveCount > 0u)) {
        record_debug_event(*debugCtxPtr,
                           kDebugEventDirectLightAuditMeta,
                           queuedRay.depth,
                           state.mediumDepth,
                           state.mediumDepth,
                           /*mediumEvent=*/0,
                           rec.frontFace,
                           materialIndex,
                           false,
                           rec.point,
                           float4(float(closureType),
                                  float(rec.primitiveIndex),
                                  material.typeEta.x,
                                  0.0f),
                           shadingNormal);
        record_debug_event(*debugCtxPtr,
                           kDebugEventDirectLightAuditMaterial,
                           queuedRay.depth,
                           state.mediumDepth,
                           state.mediumDepth,
                           /*mediumEvent=*/0,
                           rec.frontFace,
                           materialIndex,
                           false,
                           debugBaseColor,
                           float4(material_closure_aov_roughness(material),
                                  surfaceState.debugNormalUv.x,
                                  surfaceState.debugNormalUv.y,
                                  0.0f),
                           float3(float(material.textureIndices0.x), 0.0f, 0.0f));
        record_debug_event(*debugCtxPtr,
                           kDebugEventDirectLightAuditMaterialExt,
                           queuedRay.depth,
                           state.mediumDepth,
                           state.mediumDepth,
                           /*mediumEvent=*/0,
                           rec.frontFace,
                           materialIndex,
                           false,
                           material.pbrParams.xyz,
                           float4(material.pbrParams.w,
                                  material.typeEta.y,
                                  material.pbrExtras.z,
                                  diffuseOcclusion),
                           float3(material.pbrExtras.x,
                                  material.pbrExtras.y,
                                  material.pbrExtras.w));
    }

    if (PT_DEFAULT_OFF(uniforms.debugViewMode) != kDebugViewNone &&
        PT_DEFAULT_OFF(uniforms.debugViewMode) != kDebugViewCausticTransport) {
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
        material_closure_is_emissive(material) &&
        !material_closure_primary_is_emissive(material) &&
        (rec.frontFace != 0u || rec.twoSided != 0u)) {
        float3 visibleEmission = camera_visible_emission(material, queuedRay.depth);
        if (queuedRay.depth > 0u && (state.flags & kWavefrontFlagLastScatterDelta) == 0u &&
            (state.flags & kWavefrontFlagPreviousMeshNee) != 0u) {
            visibleEmission *= mesh_emitter_bsdf_hit_weight(uniforms, emissivePrimitives, rec,
                                                            incomingMisOrigin, state.lastBsdfPdf);
        }
        wavefront_add_reconstruction_radiance(
            state,
            queuedRay.depth,
            clamp_firefly_contribution(state.throughput.xyz,
                                       visibleEmission,
                                       clampParams,
                                       stats));
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

    if (material_closure_primary_is_emissive(material)) {
        if (specularOnly) {
            state.alive = 0u;
            wavefront_note_termination(queueCounters, stats, 2u);
            state.rngState = rngState;
            pathStates[pathIndex] = state;
            return;
        }
        float3 emission = material_closure_aov_emission(material);
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
            emission *= spectral_radiance_to_channels(envColor, spectralState);
        }
        if (any(emission != float3(0.0f)) &&
            (rec.frontFace != 0u || rec.twoSided != 0u)) {
            float misWeight = 1.0f;
            bool useSpecularMis =
                use_visible_emitter_mis(queuedRay.depth,
                                        (state.flags & kWavefrontFlagLastScatterDelta) != 0u,
                                        (state.flags & kWavefrontFlagLastDeltaUnpaired) != 0u,
                                        uniforms);
            if (useSpecularMis && rectLightCount > 0u &&
                ((state.flags & kWavefrontFlagLastScatterVolume) == 0u ||
                 (state.flags & kWavefrontFlagVolumeRectNee) != 0u)) {
                float lightPdf = rect_light_pdf_for_hit(uniforms,
                                                        rectangles,
                                                        materials,
                                                        rectLightCount,
                                                        rec,
                                                        incomingMisOrigin);
                float denom = state.lastBsdfPdf + lightPdf;
                if (denom > 0.0f) {
                    misWeight = clamp(state.lastBsdfPdf / denom,
                                      kMisWeightClampMin,
                                      kMisWeightClampMax);
                }
            }
            // Mesh emitters pair with the mesh-emitter NEE, not the rect lights.
            if (rec.primitiveType == kPrimitiveTypeTriangle &&
                queuedRay.depth > 0u && (state.flags & kWavefrontFlagLastScatterDelta) == 0u &&
                (state.flags & kWavefrontFlagPreviousMeshNee) != 0u) {
                misWeight = mesh_emitter_bsdf_hit_weight(uniforms, emissivePrimitives, rec,
                                                         incomingMisOrigin, state.lastBsdfPdf);
            }
            wavefront_add_reconstruction_radiance(
                state,
                queuedRay.depth,
                clamp_firefly_contribution(state.throughput.xyz,
                                           emission * misWeight,
                                           clampParams,
                                           stats));
            caustic_transport_note_light_endpoint(
                uniforms,
                rec.point,
                rec.shadingNormal,
                incomingRay.direction,
                emission,
                rectLightCount > 0u
                    ? rect_light_pdf_for_hit(uniforms,
                                             rectangles,
                                             materials,
                                             rectLightCount,
                                             rec,
                                             incomingRay.origin)
                    : 0.0f,
                state.lastBsdfPdf,
                queuedRay.depth,
                state.currentMediumId,
                materialResources.causticLightVertices,
                stats);
            state.alive = 0u;
            wavefront_note_termination(queueCounters, stats, 2u);
            state.rngState = rngState;
            pathStates[pathIndex] = state;
            return;
        }
    }

    // The ray cone grows over the traced ray. A subsurface walk replaces rec with its exit point below, and rec.t with
    // the walk's last segment, so the length is taken here, as the megakernel takes hitDistanceWorld.
    const float hitDistanceWorld = ray_segment_world_length(incomingRay, rec.t);

    // Subsurface (see subsurface_vertex_begin): a random-walk hit moves the shading vertex to the point where the
    // walk leaves the mesh; any other subsurface hit is shaded as its dense limit.
    bool subsurfaceExitVertex = false;
    if (material_is_subsurface(material)) {
        float3 pathThroughput = state.throughput.xyz;
        if (subsurface_vertex_begin(uniforms, material, rec, shadingNormal, wo, specularOnly, rngState, pathThroughput) ==
            kSubsurfaceVertexWalk) {
            HitRecord exitRec;
            float3 walkWeight = float3(0.0f);
            uint walkEvent;
#if __METAL_VERSION__ >= 310
            if (uniforms.intersectionMode == kIntersectionModeHardwareRT &&
                meshInfos && triangleData && sceneVertices && meshIndices && instanceUserIds) {
                walkEvent = sss_random_walk_hardware(uniforms, material, rec, accel, meshInfos, triangleData,
                                                     sceneVertices, meshIndices, instanceUserIds, spheres,
                                                     rectangles, nodes, primitiveIndices, stats, rngState,
                                                     exitRec, walkWeight);
            } else
#endif
            {
                walkEvent = sss_random_walk_software(uniforms, material, rec, spheres, rectangles, triangleData,
                                                     tlasNodes, tlasPrimIndices, instanceInfos, blasNodes,
                                                     blasPrimIndices, nodes, primitiveIndices,
                                                     meshInfos, sceneVertices, meshIndices, stats, rngState,
                                                     exitRec, walkWeight);
            }
            if (walkEvent != kSssWalkExit) {
                state.alive = 0u;
                wavefront_note_termination(queueCounters, stats, 3u);
                state.rngState = rngState;
                pathStates[pathIndex] = state;
                return;
            }
            pathThroughput *= walkWeight;
            subsurface_vertex_exit(material, rec, exitRec);
            shadingNormal = rec.shadingNormal;
            wo = shadingNormal;
            incidentDir = -wo;
            diffuseOcclusion = 1.0f;
            subsurfaceExitVertex = true;
        }
        if (!material_is_subsurface(material)) {
            // The explicit ReSTIR DI passes skip every subsurface surface (its evaluation is marked BSSRDF), so a
            // vertex shaded as a walk's exit point or as the dense limit takes its direct light here instead.
            explicitRestirDiLightingStage = false;
            explicitRestirDiSurfaceAccepted = false;
        }
        state.throughput.xyz = pathThroughput;
        primaryClosure = material_primary_closure_metadata(material);
        closureType = primaryClosure.type;
    }

    const bool sourceDirectAllowed = !source_transport_enabled(uniforms) || queuedRay.depth + 1u < uniforms.maxDepth;
    const bool envSampling = sourceDirectAllowed && environment_sampling_available(uniforms,
                                                            environmentConditionalAlias,
                                                            environmentMarginalAlias,
                                                            environmentPdf);
    const bool restirDiProviderDirectLighting =
        (uniforms.directLightMode == kDirectLightModeRestirDi ||
         uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
        restir_di_make_provider_set(uniforms, rectangles, materials).providerCount > 0u;
    const bool useEmissiveDirect = sourceDirectAllowed &&
        uniforms.directLightMode != kDirectLightModeLegacyRect &&
        ((uniforms.emissivePrimitiveCount > 0u && emissivePrimitives) ||
         restirDiProviderDirectLighting);
    const bool useFusedEmissiveDirect =
        useEmissiveDirect && !explicitRestirDiLightingStage;
    bool didSampleRect = false;
    bool didSampleEnv = false;
    uint risAuditRngBefore = 0u;
    uint risAuditRngAfter = 0u;
    uint risAuditTemporalRngBefore = 0u;
    uint risAuditTemporalRngAfter = 0u;

    if (!material_closure_is_delta(material) &&
        useFusedEmissiveDirect &&
        uniforms.directLightMode == kDirectLightModeBaselineEmissive) {
        RISSamplePayload lightSample;
        if (sample_direct_light_baseline(uniforms,
                                         emissivePrimitives,
                                         rec,
                                         rngState,
                                         lightSample) &&
            resolve_textured_direct_light_emission(uniforms, lightSample, materials,
                materialResources.textures, materialResources.samplers, materialTextureInfos,
                meshInfos, sceneVertices, meshIndices)) {
            lightSample.emission = spectral_radiance_to_channels(lightSample.emission, spectralState);
            const bool lightSampleDirectional = ris_payload_is_directional(lightSample);
            float distSq = ris_payload_distance_sq(rec, lightSample);
            float distance = lightSampleDirectional ? kInfinity : length(lightSample.position - rec.point);
            float3 omegaL = ris_payload_omega_l(rec, lightSample);
            float cosThetaSurface = wavefront_direct_light_cos(uniforms, material, rec, shadingNormal, omegaL);
            float cosThetaLight = lightSampleDirectional ? 1.0f : saturate(emissive_light_cosine(lightSample.flags, lightSample.normal, -omegaL));
            if (lightSample.pdf > 0.0f &&
                distSq > 0.0f &&
                cosThetaSurface > 0.0f &&
                cosThetaLight > 0.0f) {
                BsdfEvalResult eval = material_closure_eval(material,
                                                    rec.point,
                                                    shadingNormal,
                                                    wo,
                                                    omegaL,
                                                    clampParams,
                                                    PT_DEFAULT_OFF(uniforms.sssMode),
                                                    diffuseOcclusion,
                                                    specularOnly,
                                                    lightSampleDirectional);
                const bool cutoutBackLightUnsampled = apply_two_sided_alpha_cutout_diffuse_fallback(uniforms, material, rec, shadingNormal, omegaL,
                                                              diffuseOcclusion, eval);
                if (!eval.isDelta && !eval.isBssrdf) {
                    const float3 value = lightSampleDirectional ? eval.value
                        : reference_continuation_nee_value(material, shadingNormal, wo, omegaL, eval.value);
                    // Every mesh emitter is also reachable by the BSDF scatter.
                    const float weight = lightSampleDirectional ? 1.0f
                        : mesh_emitter_nee_mis_weight(uniforms, queuedRay.depth, lightSample.pdf, nee_mis_bsdf_pdf(eval, cutoutBackLightUnsampled));
                    float3 contribution = value * lightSample.emission;
                    contribution *= weight * cosThetaSurface / max(lightSample.pdf, 1.0e-6f);
                    contribution *= volume_surface_nee_transmittance(uniforms, sqrt(distSq), lightSampleDirectional, stats);
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
                            shadow.originTMax = float4(offset_spawn_origin(rec, omegaL, material),
                                                       lightSampleDirectional
                                                           ? kInfinity
                                                           : max(distance - kEpsilon, kEpsilon));
                            shadow.direction = float4(omegaL, 0.0f);
                            if (!lightSampleDirectional && (lightSample.flags & (kLightPrimitiveFlagTwoSided | kLightPrimitiveFlagTexturedEmission)) != 0u) {
                                const float3 visibilityDelta = lightSample.position - shadow.originTMax.xyz;
                                const float visibilityDistance = length(visibilityDelta);
                                shadow.direction.xyz = visibilityDelta / visibilityDistance;
                                shadow.originTMax.w = max(visibilityDistance - kEpsilon, kEpsilon);
                            }
                            shadow.contribution = float4(contribution, 0.0f);
                            shadow.throughput = float4(state.throughput.xyz, 0.0f);
                            shadow.mediumState =
                                volume_make_global_medium_state(uniforms,
                                                                state.mediumDepth,
                                                                state.volumeScatteringDepth);
                            shadow.pathIndex = pathIndex;
                            shadow.depth = queuedRay.depth;
                            shadow.lightId = lightSample.primitiveIndex;
                            shadow.flags = wavefront_shadow_receiver_flags(material,
                                lightSampleDirectional ? 0u : kWavefrontShadowFlagEmissivePrimitive,
                                lightSampleDirectional);
                            shadow.excludeMeshIndex = shadowExcludeMesh;
                            shadow.excludePrimitiveIndex = shadowExcludePrim;
                            if (source_transport_enabled(uniforms)) {
                                wavefront_prepare_source_shadow(material,rec,state,lightSample.position,!lightSampleDirectional,shadow);
                            }
                            shadowRays[shadowIndex] = shadow;
                        }
                    }
                }
            }
        }
    } else if (!material_closure_is_delta(material) && useFusedEmissiveDirect) {
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
            risAuditRngBefore = rngState;
            risAuditRngAfter = rngState;
            risAuditTemporalRngBefore = 0u;
            risAuditTemporalRngAfter = 0u;
#if PT_DEBUG_TOOLS
            record_ris_reservoir_candidate_audit(debugCtxPtr,
                                                 uniforms,
                                                 emissivePrimitives,
                                                 rec,
                                                 material,
                                                 shadingNormal,
                                                 wo,
                                                 clampParams,
                                                 diffuseOcclusion,
                                                 specularOnly,
                                                 risTargetM,
                                                 risAuditRngBefore,
                                                 queuedRay.depth,
                                                 state.mediumDepth,
                                                 materialIndex);
#endif
            if (uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) {
                build_restir_di_provider_reservoir_for_hit(uniforms,
                                                           emissivePrimitives,
                                                           rectangles,
                                                           materials,
                                                           environmentTexture,
                                                           environmentConditionalAlias,
                                                           environmentMarginalAlias,
                                                           environmentPdf,
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
            } else {
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
            }
            risAuditRngAfter = rngState;

            if ((uniforms.directLightMode == kDirectLightModeRisTemporalReuse ||
                 uniforms.directLightMode == kDirectLightModeRestirDi ||
                 uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
                reservoir.valid &&
                (queuedRay.depth == 0u && !subsurfaceExitVertex) &&
                rec.t > 0.0f &&
                uniforms.frameIndex > 0u) {
                uint temporalState = pcg_hash(uniforms.fixedRngSeed ^
                                              (uint(pixel.x) * 1664525u) ^
                                              (uint(pixel.y) * 1013904223u) ^
                                              ((uniforms.frameIndex - 1u) * 9781u) ^
                                              0x9E3779B9u);
                risAuditTemporalRngBefore = temporalState;
                Reservoir previousReservoir = make_empty_reservoir();
                const bool previousReservoirValid =
                    uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid
                        ? build_restir_di_provider_reservoir_for_hit(uniforms,
                                                                     emissivePrimitives,
                                                                     rectangles,
                                                                     materials,
                                                                     environmentTexture,
                                                                     environmentConditionalAlias,
                                                                     environmentMarginalAlias,
                                                                     environmentPdf,
                                                                     rec,
                                                                     material,
                                                                     shadingNormal,
                                                                     wo,
                                                                     clampParams,
                                                                     diffuseOcclusion,
                                                                     specularOnly,
                                                                     risTargetM,
                                                                     temporalState,
                                                                     previousReservoir)
                        : build_ris_reservoir_for_hit(uniforms,
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
                                                      previousReservoir);
                if (previousReservoirValid && previousReservoir.valid) {
                    risAuditTemporalRngAfter = temporalState;
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
                } else {
                    risAuditTemporalRngAfter = temporalState;
                }
            }

            if (uniforms.directLightMode == kDirectLightModeRisWorldReuse &&
                (queuedRay.depth == 0u && !subsurfaceExitVertex) &&
                rec.t > 0.0f &&
                uniforms.width > 0u &&
                uniforms.height > 0u) {
                int3 worldCell = world_reuse_cell(rec.point, uniforms.worldReuseCellSize);
                // Generalized RIS with pairwise MIS, as ris_spatial (reservoir.metal): each candidate pixel's fresh
                // reservoir streams into `merged` with its pairwise weight, then the canonical winner joins with its
                // share of every pair. The old merge weighted each candidate by p_c / (p_c M + p_n M), kept the
                // canonical unweighted and added 1 to M per candidate: 0.824 of the exact light on the EP12 sun plane.
                // Targets are compared per unit light area, which carries the solid-angle Jacobian of moving a
                // triangle-light sample between pixels.
                constexpr uint kWorldReusePairs = 4u;
                const float pairCount = float(kWorldReusePairs);
                const bool canonicalValid = reservoir.valid;
                const RISSamplePayload canonicalWinner = reservoir.winner;
                const float canonicalAreaTarget = canonicalValid
                    ? reservoir_winner_phat_for_hit(uniforms,
                                                    rec,
                                                    material,
                                                    shadingNormal,
                                                    wo,
                                                    clampParams,
                                                    diffuseOcclusion,
                                                    specularOnly,
                                                    canonicalWinner) *
                          ris_payload_solid_angle_per_area(rec, canonicalWinner)
                    : 0.0f;
                // A pair whose candidate is out of frame, misses, or fails the cell, depth or normal test belongs to
                // the canonical.
                float canonicalMis = 0.0f;
                Reservoir merged = make_empty_reservoir();
                constexpr int2 offsets[4] = {
                    int2(-2, 0),
                    int2(2, 0),
                    int2(0, -2),
                    int2(0, 2)
                };
                for (uint n = 0u; n < kWorldReusePairs; ++n) {
                    int2 candidateCoordI = int2(pixel) + offsets[n];
                    if (candidateCoordI.x < 0 ||
                        candidateCoordI.y < 0 ||
                        candidateCoordI.x >= int(uniforms.width) ||
                        candidateCoordI.y >= int(uniforms.height)) {
                        canonicalMis += 1.0f / pairCount;
                        continue;
                    }

                    uint2 candidateCoord = uint2(candidateCoordI);
                    Ray candidateRay = make_center_primary_ray_for_pixel(uniforms, candidateCoord);
                    HitRecord candidateRec = make_empty_hit_record();
                    bool candidateHit = false;
                    // Candidates are traced like the pixel's own rays: in hardware mode the software BVH has no
                    // triangles (the triangle buffer is not uploaded), so a software trace missed every candidate and
                    // ris_world ran as plain RIS.
#if __METAL_VERSION__ >= 310
                    if (uniforms.intersectionMode == kIntersectionModeHardwareRT &&
                        meshInfos && triangleData && sceneVertices && meshIndices && instanceUserIds) {
                        candidateHit = trace_scene_hardware(uniforms,
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
                                                                 candidateRay,
                                                                 kEpsilon,
                                                                 kInfinity,
                                                                 /*anyHitOnly=*/false,
                                                                 /*excludeMeshIndex=*/kInvalidIndex,
                                                                 /*excludePrimitiveIndex=*/kInvalidIndex,
                                                                 candidateRec);
                    } else
#endif
                    {
                        candidateHit = trace_scene_software(uniforms,
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
                    }
                    if (!candidateHit || candidateRec.materialIndex >= uniforms.materialCount) {
                        canonicalMis += 1.0f / pairCount;
                        continue;
                    }

                    int3 candidateCell =
                        world_reuse_cell(candidateRec.point, uniforms.worldReuseCellSize);
                    if (!world_reuse_cell_compatible(worldCell, candidateCell)) {
                        canonicalMis += 1.0f / pairCount;
                        continue;
                    }

                    float depthRel = fabs(rec.t - candidateRec.t) / max(rec.t, kEpsilon);
                    if (!(depthRel <= 0.2f) || !isfinite(depthRel)) {
                        canonicalMis += 1.0f / pairCount;
                        continue;
                    }
                    float normalDot = dot(safe_normalize(shadingNormal),
                                          safe_normalize(candidateRec.shadingNormal));
                    if (!(normalDot >= 0.85f) || !isfinite(normalDot)) {
                        canonicalMis += 1.0f / pairCount;
                        continue;
                    }

                    MaterialData candidateMaterial = materials[candidateRec.materialIndex];
                    float3 candidateWo = safe_normalize(-candidateRay.direction);
                    if (canonicalValid) {
                        canonicalMis += pairwise_mis_canonical_share(
                            canonicalAreaTarget,
                            reservoir_winner_phat_for_hit(uniforms,
                                                          candidateRec,
                                                          candidateMaterial,
                                                          candidateRec.shadingNormal,
                                                          candidateWo,
                                                          clampParams,
                                                          diffuseOcclusion,
                                                          specularOnly,
                                                          canonicalWinner) *
                                ris_payload_solid_angle_per_area(candidateRec, canonicalWinner),
                            pairCount);
                    }
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
                    float mergeWeight = pairwise_mis_neighbor_weight(
                        phatCurrent * ris_payload_solid_angle_per_area(rec, candidateReservoir.winner),
                        phatCandidate * ris_payload_solid_angle_per_area(candidateRec, candidateReservoir.winner),
                        candidateReservoir.wSum,
                        pairCount);
                    if (!(mergeWeight > 0.0f)) {
                        continue;
                    }
                    update_reservoir(merged,
                                     candidateReservoir.winner,
                                     mergeWeight,
                                     rand_uniform(rngState));
                }
                if (canonicalValid) {
                    update_reservoir(merged, canonicalWinner, canonicalMis * reservoir.wSum, rand_uniform(rngState));
                }
                if (merged.valid) {
                    merged.M = risTargetM;  // every reservoir merged here holds risTargetM candidates
                }
                reservoir = merged;
            }

            if ((uniforms.directLightMode == kDirectLightModeRisRegirCache ||
                 uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
                (queuedRay.depth == 0u && !subsurfaceExitVertex) &&
                rec.t > 0.0f &&
                uniforms.frameIndex > 0u) {
                int3 cacheCell = world_reuse_cell(rec.point, uniforms.worldReuseCellSize);
                uint cacheSourceFrameIndex =
                    uniforms.frameIndex > 1u ? uniforms.frameIndex - 2u : uniforms.frameIndex - 1u;
                // The cached reservoirs are fresh RIS reservoirs built at this pixel's own hit point, with the same
                // builder and target function as the pixel's reservoir and with random numbers seeded per cell and
                // frame. Reservoirs with one target at one point merge by plain streaming: each adds its wSum and M
                // grows by its candidate count (generalized RIS with MIS weights in proportion to M). Every probe is
                // merged, valid or not, since each drew risTargetM candidates. The old merge weighted each cached
                // reservoir by p(y_c) / (p(y) M + p(y_c) M_c), kept the two with the largest weights (a choice that
                // depends on the samples) and added 1 to M per merge: 0.946 of the exact light on the EP12 sun plane
                // under ris_regir and 0.962 under the hybrid.
                // The pixel's reservoir enters with its own contribution weight wSum / M and a fixed share of nominalM
                // candidates: risTargetM, or twice that for the hybrid, whose temporal step merges a second reservoir
                // of risTargetM candidates when both are valid. Its actual M depends on the samples there, so it cannot
                // be its share: MIS weights that depend on the samples would bias the merge.
                const uint nominalM =
                    (uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid ? 2u : 1u) * risTargetM;
                if (reservoir.valid && reservoir.M != nominalM) {
                    reservoir.wSum *= float(nominalM) / float(reservoir.M);
                }
                uint mergedM = nominalM;
                uint candidateAttempts = 0u;
                uint accepted = 0u;
                for (uint cacheProbe = 0u; cacheProbe < 4u; ++cacheProbe) {
                    if (cacheProbe > cacheSourceFrameIndex) {
                        continue;
                    }
                    candidateAttempts += 1u;
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
                    const bool cachedReservoirValid =
                        uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid
                        ? build_restir_di_provider_reservoir_for_hit(uniforms,
                                                                     emissivePrimitives,
                                                                     rectangles,
                                                                     materials,
                                                                     environmentTexture,
                                                                     environmentConditionalAlias,
                                                                     environmentMarginalAlias,
                                                                     environmentPdf,
                                                                     rec,
                                                                     material,
                                                                     shadingNormal,
                                                                     wo,
                                                                     clampParams,
                                                                     diffuseOcclusion,
                                                                     specularOnly,
                                                                     risTargetM,
                                                                     cacheState,
                                                                     cachedReservoir)
                        : build_ris_reservoir_for_hit(uniforms,
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
                                                      cachedReservoir);
                    mergedM += risTargetM;
                    if (!cachedReservoirValid || !cachedReservoir.valid) {
                        continue;
                    }
                    update_reservoir(reservoir, cachedReservoir.winner, cachedReservoir.wSum, rand_uniform(rngState));
                    accepted += 1u;
                }
                if (reservoir.valid) {
                    reservoir.M = mergedM;
                }
                if (stats && candidateAttempts > 0u) {
                    atomic_fetch_add_explicit(&stats->restirDiRegirCandidateCount,
                                              candidateAttempts,
                                              memory_order_relaxed);
                }
                if (stats) {
                    if (accepted > 0u) {
                        atomic_fetch_add_explicit(&stats->restirDiRegirAcceptedCount,
                                                  accepted,
                                                  memory_order_relaxed);
                    } else if (candidateAttempts > 0u) {
                        atomic_fetch_add_explicit(&stats->restirDiRegirFallbackCount,
                                                  1u,
                                                  memory_order_relaxed);
                    }
                }
            }
        }

        float3 risAuditBsdfValue = float3(0.0f);
        float3 risAuditContribution = float3(0.0f);
        float risAuditDistance = 0.0f;
        float risAuditNDotL = 0.0f;
        float risAuditBsdfPdf = 0.0f;
        bool risAuditQueued = false;
        if (reservoir.valid && reservoir.M > 0u && reservoir.wSum > 0.0f) {
            RISSamplePayload winner = reservoir.winner;
            float distSq = ris_payload_distance_sq(rec, winner);
            if (distSq > 0.0f && isfinite(distSq)) {
                const bool winnerDirectional = ris_payload_is_directional(winner);
                float distance = winnerDirectional ? kInfinity : length(winner.position - rec.point);
                float3 omegaL = ris_payload_omega_l(rec, winner);
                float cosThetaSurface = direct_light_target_cos(uniforms, material, rec, shadingNormal, omegaL);
                risAuditDistance = winnerDirectional ? distance : length(winner.position - rec.point);
                risAuditNDotL = cosThetaSurface;
                float cosThetaLight = winnerDirectional ? 1.0f : saturate(emissive_light_cosine(winner.flags, winner.normal, -omegaL));
                bool cutoutBackLightUnsampled = false;
                BsdfEvalResult eval = direct_light_target_bsdf(uniforms,
                                                               material,
                                                               rec,
                                                               shadingNormal,
                                                               wo,
                                                               omegaL,
                                                               clampParams,
                                                               diffuseOcclusion,
                                                               specularOnly,
                                                               winnerDirectional,
                                                               cutoutBackLightUnsampled);
                risAuditBsdfValue = eval.value;
                risAuditBsdfPdf = eval.pdf;
                float phatW = p_hat(winner.emission,
                                    eval.value,
                                    cosThetaSurface,
                                    cosThetaLight,
                                    distSq);
                // The reservoir builders count every drawn candidate in M.
                reservoir.W = (phatW > 0.0f)
                            ? ((reservoir.wSum / float(reservoir.M)) / phatW)
                            : 0.0f;
                // Candidates are weighted by the light's emission factor; the
                // shaded estimate needs the textured emission at the chosen point.
                const bool texturedWinner = !winnerDirectional &&
                    (winner.flags & kLightPrimitiveFlagTexturedEmission) != 0u;
                RISSamplePayload shadedWinner = winner;
                const bool winnerEmissionResolved = !texturedWinner ||
                    resolve_textured_direct_light_emission(uniforms, shadedWinner, materials,
                        materialResources.textures, materialResources.samplers, materialTextureInfos,
                        meshInfos, sceneVertices, meshIndices);
                // Plain RIS shares mesh emitters with the BSDF hit
                // (mesh_emitter_bsdf_hit_weight); reuse modes keep weight 1.
                const float risMisWeight = winnerDirectional ? 1.0f
                    : mesh_emitter_nee_mis_weight(uniforms, queuedRay.depth,
                          mesh_emitter_nee_candidate_count(uniforms) * winner.pdf, nee_mis_bsdf_pdf(eval, cutoutBackLightUnsampled));
                if (!eval.isDelta &&
                    !eval.isBssrdf &&
                    winnerEmissionResolved &&
                    reservoir.W > 0.0f &&
                    cosThetaSurface > 0.0f &&
                    all(isfinite(eval.value))) {
                    float3 contribution = eval.value * shadedWinner.emission;
                    contribution *= cosThetaSurface * reservoir.W * risMisWeight;
                    contribution *= volume_surface_nee_transmittance(uniforms, sqrt(distSq), winnerDirectional, stats);
                    if (all(isfinite(contribution))) {
                        risAuditContribution = contribution;
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
                            uint shadowFlags = winnerDirectional ? 0u : kWavefrontShadowFlagEmissivePrimitive;
                            if ((winner.flags & kLightPrimitiveFlagRectangle) != 0u) {
                                shadowFlags = kWavefrontShadowFlagRectangleLight;
                            }
                            ShadowRay shadow{};
                            shadow.originTMax = float4(offset_spawn_origin(rec, omegaL, material),
                                                       winnerDirectional
                                                           ? kInfinity
                                                           : max(distance - kEpsilon, kEpsilon));
                            shadow.direction = float4(omegaL, 0.0f);
                            if (texturedWinner) {
                                const float3 visibilityDelta = winner.position - shadow.originTMax.xyz;
                                const float visibilityDistance = length(visibilityDelta);
                                shadow.direction.xyz = visibilityDelta / visibilityDistance;
                                shadow.originTMax.w = max(visibilityDistance - kEpsilon, kEpsilon);
                            }
                            shadow.contribution = float4(contribution, 0.0f);
                            shadow.throughput = float4(state.throughput.xyz, 0.0f);
                            shadow.mediumState =
                                volume_make_global_medium_state(uniforms,
                                                                state.mediumDepth,
                                                                state.volumeScatteringDepth);
                            shadow.pathIndex = pathIndex;
                            shadow.depth = queuedRay.depth;
                            shadow.lightId = winner.primitiveIndex;
                            shadow.flags = wavefront_shadow_receiver_flags(material, shadowFlags, winnerDirectional);
                            shadow.excludeMeshIndex = shadowExcludeMesh;
                            shadow.excludePrimitiveIndex = shadowExcludePrim;
                            if (source_transport_enabled(uniforms)) {
                                wavefront_prepare_source_shadow(material,rec,state,winner.position,!winnerDirectional,shadow);
                            }
                            shadowRays[shadowIndex] = shadow;
                            risAuditQueued = true;
                        }
                    }
                }
            }
        }
        if (debugCtxPtr &&
            PT_DEFAULT_OFF(uniforms.debugDirectLightAudit) != 0u &&
            (uniforms.directLightMode == kDirectLightModeRis ||
             uniforms.directLightMode == kDirectLightModeRisSpatialReuse ||
             uniforms.directLightMode == kDirectLightModeRisTemporalReuse ||
             uniforms.directLightMode == kDirectLightModeRisWorldReuse ||
             uniforms.directLightMode == kDirectLightModeRisRegirCache ||
             uniforms.directLightMode == kDirectLightModeRestirDi ||
             uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid)) {
            record_debug_event(*debugCtxPtr,
                               kDebugEventRisAuditState,
                               queuedRay.depth,
                               state.mediumDepth,
                               state.mediumDepth,
                               /*mediumEvent=*/0,
                               rec.frontFace,
                               materialIndex,
                               false,
                               float3(reservoir.wSum,
                                      reservoir.W,
                                      reservoir.winner.pdf),
                               float4(float(reservoir.winner.primitiveIndex),
                                      float(reservoir.M),
                                      reservoir.valid ? 1.0f : 0.0f,
                                      risAuditQueued ? 1.0f : 0.0f),
                               float3(risAuditDistance,
                                      risAuditNDotL,
                                      risAuditBsdfPdf));
            record_debug_event(*debugCtxPtr,
                               kDebugEventRisAuditRng,
                               queuedRay.depth,
                               state.mediumDepth,
                               state.mediumDepth,
                               /*mediumEvent=*/0,
                               rec.frontFace,
                               materialIndex,
                               false,
                               float3(as_type<float>(risAuditRngBefore),
                                      as_type<float>(risAuditRngAfter),
                                      0.0f),
                               float4(as_type<float>(risAuditTemporalRngBefore),
                                      as_type<float>(risAuditTemporalRngAfter),
                                      0.0f,
                                      0.0f),
                               float3(0.0f));
            if (reservoir.valid) {
                record_debug_event(*debugCtxPtr,
                                   kDebugEventRisAuditWinnerA,
                                   queuedRay.depth,
                                   state.mediumDepth,
                                   state.mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   materialIndex,
                                   false,
                                   reservoir.winner.position,
                                   float4(reservoir.winner.bary.x,
                                          reservoir.winner.bary.y,
                                          0.0f,
                                          0.0f),
                                   reservoir.winner.normal);
                record_debug_event(*debugCtxPtr,
                                   kDebugEventRisAuditWinnerB,
                                   queuedRay.depth,
                                   state.mediumDepth,
                                   state.mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   materialIndex,
                                   false,
                                   reservoir.winner.emission,
                                   float4(0.0f),
                                   risAuditBsdfValue);
                record_debug_event(*debugCtxPtr,
                                   kDebugEventRisAuditWinnerC,
                                   queuedRay.depth,
                                   state.mediumDepth,
                                   state.mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   materialIndex,
                                   false,
                                   risAuditContribution,
                                   float4(0.0f),
                                   float3(0.0f));
            }
        }
    }
    // The fused emissive estimator and analytic rectangle estimator cover
    // disjoint lights.
    const uint independentRectFilter = kAnalyticLightFilterAll;
    const uint independentRectCount = rectLightCount;
    if (sourceDirectAllowed && !material_closure_is_delta(material) && independentRectCount > 0u) {
        for (uint neeDraw = 0u; neeDraw < rect_light_nee_draw_count(uniforms); ++neeDraw) {
            RectLightSample lightSample;
            if (sample_rect_light_nee_draw(uniforms,
                                  rectangles,
                                  materials,
                                  environmentTexture,
                                  rec,
                                  rngState,
                                  independentRectCount,
                                  lightSample,
                                  independentRectFilter, neeDraw)) {
                lightSample.emission = spectral_radiance_to_channels(lightSample.emission, spectralState);
                didSampleRect = true;
                BsdfEvalResult eval = material_closure_eval(material,
                                                    rec.point,
                                                    rec.shadingNormal,
                                                    wo,
                                                    lightSample.direction,
                                                    clampParams,
                                                    PT_DEFAULT_OFF(uniforms.sssMode),
                                                    diffuseOcclusion,
                                                    specularOnly,
                                                    lightSample.isReferencePoint);
                const bool cutoutBackLightUnsampled = apply_two_sided_alpha_cutout_diffuse_fallback(uniforms, material, rec, shadingNormal,
                                                              lightSample.direction, diffuseOcclusion, eval);
                if (!lightSample.isReferencePoint) {
                    eval.value = reference_continuation_nee_value(material, shadingNormal, wo,
                                                                   lightSample.direction, eval.value);
                }
                const float nDotL = wavefront_direct_light_cos(uniforms, material, rec, shadingNormal,
                                                               lightSample.direction);
                if (!eval.isDelta &&
                    !eval.isBssrdf &&
                    eval.pdf > 0.0f &&
                    all(isfinite(eval.value)) &&
                    any(eval.value > float3(0.0f)) &&
                    lightSample.pdf > 0.0f &&
                    nDotL > 0.0f) {
                    float bsdfPdf = nee_mis_bsdf_pdf(eval, cutoutBackLightUnsampled);
                    float weight = 1.0f;
                    const bool referenceSurface = (material.materialFlags &
                        (kMaterialFlagSingleScatterSpecular | kMaterialFlagReferenceDielectricContinuation)) != 0u;
                    if (bsdfPdf > 0.0f && !lightSample.isReferencePoint &&
                        (!referenceSurface || queuedRay.depth + 1u < uniforms.maxDepth)) {
                        float denom = lightSample.pdf + bsdfPdf;
                        if (denom > 0.0f) {
                            weight = clamp(lightSample.pdf / denom,
                                           kMisWeightClampMin,
                                           kMisWeightClampMax);
                        }
                    }
                    float3 rectContribution = float3(0.0f);
                    float causticMisWeight = weight;
                    const bool causticConnected = !lightSample.isReferencePoint &&
                        caustic_transport_connect_light_sample(uniforms,
                                                               material,
                                                               rec.point,
                                                               shadingNormal,
                                                               rec.normal,
                                                               lightSample.direction,
                                                               lightSample.position,
                                                               lightSample.normal,
                                                               false,
                                                               state.throughput.xyz,
                                                               lightSample.emission,
                                                               eval.value,
                                                               bsdfPdf,
                                                               lightSample.pdf,
                                                               nDotL,
                                                               queuedRay.depth,
                                                               state.currentMediumId,
                                                               materialResources.causticEyeVertices,
                                                               materialResources.causticLightVertices,
                                                               materialResources.causticConnections,
                                                               materialResources.causticConnectionResults,
                                                               stats,
                                                               rectContribution,
                                                               causticMisWeight);
                    if (causticConnected) {
                        weight = causticMisWeight;
                    } else {
                        rectContribution = lightSample.emission * eval.value * nDotL;
                        rectContribution *= weight / lightSample.pdf;
                    }
                    rectContribution *= volume_surface_nee_transmittance(uniforms, lightSample.distance, false, stats);
                    const float3 contribution = clamp_firefly_contribution(
                        state.throughput.xyz,
                        rectContribution,
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
                        shadow.originTMax = float4(offset_spawn_origin(rec, lightSample.direction, material),
                                                   max(lightSample.distance - kEpsilon, kEpsilon));
                        shadow.direction = float4(lightSample.direction, 0.0f);
                        shadow.contribution = float4(contribution, 0.0f);
                        shadow.throughput = float4(state.throughput.xyz, 0.0f);
                        shadow.mediumState =
                            volume_make_global_medium_state(uniforms,
                                                            state.mediumDepth,
                                                            state.volumeScatteringDepth);
                        shadow.pathIndex = pathIndex;
                        shadow.depth = queuedRay.depth;
                        shadow.lightId = lightSample.rectIndex;
                        shadow.flags = wavefront_shadow_receiver_flags(material, kWavefrontShadowFlagRectangleLight);
                        shadow.excludeMeshIndex = shadowExcludeMesh;
                        shadow.excludePrimitiveIndex = shadowExcludePrim;
                        if (source_transport_enabled(uniforms)) {
                            wavefront_prepare_source_shadow(material,rec,state,lightSample.position,true,shadow);
                        }
                        shadowRays[shadowIndex] = shadow;
                    }
                }
            }
        }
}

    if (!material_closure_is_delta(material) && envSampling) {
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
            if (useOverride) {
                envSample.radiance = environment_color_lod(environmentTexture,
                                                           envSample.direction,
                                                           uniforms.environmentRotation,
                                                           uniforms.environmentIntensity,
                                                           overrideLod,
                                                           uniforms);
            }
            envSample.radiance = spectral_radiance_to_channels(envSample.radiance, spectralState);
            float nDotL = wavefront_direct_light_cos(uniforms, material, rec, shadingNormal, envSample.direction);
            if (envSample.pdf > 0.0f && nDotL > 0.0f) {
                BsdfEvalResult eval = material_closure_eval(material,
                                                    rec.point,
                                                    shadingNormal,
                                                    wo,
                                                    envSample.direction,
                                                    clampParams,
                                                    PT_DEFAULT_OFF(uniforms.sssMode),
                                                    diffuseOcclusion,
                                                    specularOnly);
                const bool cutoutBackLightUnsampled = apply_two_sided_alpha_cutout_diffuse_fallback(uniforms, material, rec, shadingNormal,
                                                              envSample.direction, diffuseOcclusion, eval);
                eval.value = reference_continuation_nee_value(material, shadingNormal, wo,
                                                               envSample.direction, eval.value);
                if (!eval.isDelta &&
                    !eval.isBssrdf &&
                    eval.pdf > 0.0f &&
                    all(isfinite(eval.value)) &&
                    any(eval.value > float3(0.0f))) {
                    float weight = 1.0f;
                    float denom = envSample.pdf + nee_mis_bsdf_pdf(eval, cutoutBackLightUnsampled);
                    const bool referenceSurface = (material.materialFlags &
                        (kMaterialFlagSingleScatterSpecular | kMaterialFlagReferenceDielectricContinuation)) != 0u;
                    if (denom > 0.0f && (!referenceSurface || queuedRay.depth + 1u < uniforms.maxDepth)) {
                        weight = clamp(envSample.pdf / denom,
                                       kMisWeightClampMin,
                                       kMisWeightClampMax);
                    }
                    float3 causticConnectionContribution = float3(0.0f);
                    float causticMisWeight = weight;
                    const bool causticConnected =
                        caustic_transport_connect_light_sample(uniforms,
                                                               material,
                                                               rec.point,
                                                               shadingNormal,
                                                                       rec.normal,
                                                                       envSample.direction,
                                                                       rec.point + safe_normalize(envSample.direction),
                                                                       -safe_normalize(envSample.direction),
                                                                       true,
                                                                       state.throughput.xyz,
                                                                       envSample.radiance,
                                                               eval.value,
                                                               nee_mis_bsdf_pdf(eval, cutoutBackLightUnsampled),
                                                               envSample.pdf,
                                                               nDotL,
                                                               queuedRay.depth,
                                                               state.currentMediumId,
                                                               materialResources.causticEyeVertices,
                                                               materialResources.causticLightVertices,
                                                               materialResources.causticConnections,
                                                               materialResources.causticConnectionResults,
                                                               stats,
                                                               causticConnectionContribution,
                                                               causticMisWeight);
                    if (causticConnected) {
                        weight = causticMisWeight;
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
                        shadow.originTMax = float4(offset_spawn_origin(rec, envSample.direction, material), kInfinity);
                        shadow.direction = float4(envSample.direction, 0.0f);
                        shadow.contribution = float4(contributionFactor, 0.0f);
                        shadow.throughput = float4(state.throughput.xyz, 0.0f);
                        shadow.environmentRadiance = float4(envSample.radiance, 0.0f);
                        shadow.mediumState =
                            volume_make_global_medium_state(uniforms,
                                                            state.mediumDepth,
                                                            state.volumeScatteringDepth);
                        shadow.pathIndex = pathIndex;
                        shadow.depth = queuedRay.depth;
                        shadow.lightId = kInvalidIndex;
                        shadow.flags = kWavefrontShadowFlagEnvironment |
                            (interface_precision(material) ? kWavefrontShadowFlagInterfacePrecision : 0u);
                        shadow.excludeMeshIndex = shadowExcludeMesh;
                        shadow.excludePrimitiveIndex = shadowExcludePrim;
                        if (source_transport_enabled(uniforms)) {
                            wavefront_prepare_source_shadow(material,rec,state,float3(0.0f),false,shadow);
                        }
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

    BsdfSampleResult bsdfSample;
    uint rngStateBeforeBsdf = rngState;
    bsdfSample = material_closure_sample(material,
                                         rec.point,
                                         rec.shadingNormal,
                                         wo,
                                         incidentDir,
                                         rec.frontFace != 0u,
                                         rngState,
                                         clampParams,
                                         PT_DEFAULT_OFF(uniforms.sssMode),
                                         diffuseOcclusion,
                                         specularOnly);
    if (subsurfaceExitVertex) {
        // Keeps this vertex out of ReSTIR GI, ReSTIR PT, path guiding and the radiance cache, whose
        // per-pixel state belongs to the primary hit and not to the walk's exit point.
        bsdfSample.isBssrdf = true;
    }
    apply_restir_gi_production_diffuse(uniforms,
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
    apply_restir_pt_production_path_reuse(uniforms,
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
    capture_restir_pt_production_reservoir(uniforms,
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
                           float4(as_type<float>(rngStateBeforeBsdf),
                                  as_type<float>(rngState),
                                  float(closureType),
                                  subsurfaceExitVertex ? 1.0f : 0.0f),
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
        dot(bsdfSample.direction, bsdfSample.direction) <= 0.0f ||
        falcor_face_visibility(material, rec, bsdfSample.direction) == 0.0f) {
        state.alive = 0u;
        wavefront_note_termination(queueCounters, stats, 3u);
        state.rngState = rngState;
        pathStates[pathIndex] = state;
        return;
    }
    if (!bsdfSample.isDelta) {
        state.pathMaxRoughness = max(state.pathMaxRoughness,
                                      path_regularization_roughness(material));
    }
    caustic_transport_note_path_vertex(uniforms,
                                       material,
                                       rec.point,
                                       rec.shadingNormal,
                                       rec.normal,
                                       bsdfSample.direction,
                                       state.throughput.xyz,
                                       queuedRay.depth,
                                       state.currentMediumId,
                                       bsdfSample,
                                       materialResources.causticEyeVertices,
                                       materialResources.causticLightVertices,
                                       materialResources.causticConnections,
                                       materialResources.causticConnectionResults,
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
        wavefront_store_radiance_cache_work_item(materialResources.radianceCacheQueryItems,
                                                 queueCounters->cacheQueryQueue,
                                                 cacheQuery,
                                                 float4(0.0f),
                                                 float4(0.0f),
                                                 pathIndex,
                                                 queuedRay.pixelIndex,
                                                 queuedRay.depth,
                                                 0u);
    }
    float cacheConfidence = 0.0f;
    float cacheVariance = 0.0f;
    float cacheBiasRisk = 0.0f;
    // A training path never loads from the cache: its suffix is the unbiased sample the cache learns from.
    const bool cacheTrainingPath = radiance_cache_is_training_path(uniforms, queuedRay.pixelIndex);
    if (!cacheTrainingPath &&
        radiance_cache_query_and_maybe_terminate(uniforms,
                                                 material,
                                                 rec.point,
                                                 rec.shadingNormal,
                                                 specularOnly,
                                                 queuedRay.depth,
                                                 bsdfSample,
                                                 materialResources.radianceCacheStates,
                                                 stats,
                                                 radianceAfterCache,
                                                 state.throughput.xyz,
                                                 cacheConfidence,
                                                 cacheVariance,
                                                 cacheBiasRisk)) {
        const float3 cacheContribution = radianceAfterCache - state.radiance.xyz;
        state.radiance.xyz = radianceAfterCache;
        if (queuedRay.depth == 0u) {
            state.directRadiance.xyz += cacheContribution;
        } else {
            state.indirectRadiance.xyz += cacheContribution;
        }
        state.reconstructionConfidence = max(state.reconstructionConfidence, cacheConfidence);
        state.reconstructionVariance = max(state.reconstructionVariance, cacheVariance);
        state.reconstructionBiasRisk = max(state.reconstructionBiasRisk, cacheBiasRisk);
        state.alive = 0u;
        wavefront_note_termination(queueCounters, stats, 3u);
        state.rngState = rngState;
        pathStates[pathIndex] = state;
        return;
    }
    if (cacheTrainingPath) {
        wavefront_radiance_cache_note_vertex(materialResources.radianceCacheTrainRecords,
                                             queueCounters->cacheTrainingQueue,
                                             uniforms,
                                             material,
                                             rec.point,
                                             rec.shadingNormal,
                                             specularOnly,
                                             queuedRay.depth,
                                             bsdfSample,
                                             queuedRay.pixelIndex,
                                             state.radiance.xyz,
                                             state.throughput.xyz);
    }

    uint mediumDepthBefore = mediumDepth;
    float3 selectedSigma = float3(0.0f);
    if (interface_select_medium(material, rec, bsdfSample.direction, selectedSigma, mediumDepth)) {
        if (source_material_flag(material, kMaterialFlagExplicitInterior)) state.mediumStack[0] = float4(selectedSigma, 0.0f);
    } else if (bsdfSample.mediumEvent == 1) {
        float3 sigma = dielectric_sigma_a(material);
        sigma = max(sigma, float3(0.0f));
        if (mediumDepth < kWavefrontMediumStackSize) {
            state.mediumStack[mediumDepth] = float4(sigma, 0.0f);
            mediumDepth += 1u;
        } else {
            state.mediumStack[kWavefrontMediumStackSize - 1u] = float4(sigma, 0.0f);
        }
    } else if (bsdfSample.mediumEvent == -1) {
        if (mediumDepth > 0u) {
            mediumDepth -= 1u;
        }
    }
    volume_note_boundary_event(bsdfSample.mediumEvent, stats);
    uint mediumDepthAfter = mediumDepth;

    const float3 nextDirection = bsdfSample.direction;
    bool didTransmission = false;
    if (bsdfSample.isDelta && material_closure_is_transmissive(material)) {
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
        nextOrigin = (interface_precision(material) || source_transport_enabled(uniforms))
            ? offset_interface_origin(rec, bsdfSample.direction)
            : offset_ray_origin(rec, bsdfSample.direction);
    }

    bool specNeeEnabled = (uniforms.enableSpecularNee != 0u) && !source_transport_enabled(uniforms);
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
            occluded = trace_scene_hardware_alpha_aware_shadow<false>(uniforms,
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
                                                                      spawn_ray_min(material, kEpsilon),
                                                                      kInfinity,
                                                                      neeExcludeMesh,
                                                                      neeExcludePrim,
                                                                      materials,
                                                                      materialResources.textures,
                                                                      materialResources.samplers,
                                                                      materialTextureInfos,
                                                                      shadowRec);
        } else
#endif
        {
            occluded = trace_scene_software_alpha_aware_shadow<false>(uniforms,
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
                                                                      spawn_ray_min(material, kEpsilon),
                                                                      kInfinity,
                                                                      materials,
                                                                      materialResources.textures,
                                                                      materialResources.samplers,
                                                                      materialTextureInfos,
                                                                      meshInfos,
                                                                      sceneVertices,
                                                                      meshIndices,
                                                                      shadowRec);
        }

        if (!occluded) {
            float envPdf = environment_pdf(uniforms, environmentPdf, neeRay.direction);
            envPdf = max(envPdf, kSpecularNeePdfFloor);
            float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
            float denom = envPdf + bsdfPdf;
            float misWeight = (denom > 0.0f) ? (envPdf / denom) : 0.0f;
            misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
            float3 envColor = environment_color(environmentTexture,
                                                neeRay.direction,
                                                uniforms.environmentRotation,
                                                uniforms.environmentIntensity,
                                                uniforms);
            // The delta sample fixed this direction; it was not drawn from the
            // environment, so there is no 1/envPdf. This connection takes the
            // light strategy's MIS share and the continuation's environment miss
            // takes bsdfPdf / (bsdfPdf + envPdf), so the two sum to one.
            float3 neeContribution = bsdfSample.weight * envColor * misWeight;
            if (all(isfinite(neeContribution))) {
                wavefront_add_reconstruction_radiance(
                    state,
                    queuedRay.depth,
                    clamp_firefly_contribution(state.throughput.xyz,
                                               neeContribution,
                                               clampParams,
                                               stats));
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
            hitLight = trace_scene_hardware_alpha_aware_shadow<false>(uniforms,
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
                                                                      spawn_ray_min(material, kEpsilon),
                                                                      kInfinity,
                                                                      neeExcludeMesh,
                                                                      neeExcludePrim,
                                                                      materials,
                                                                      materialResources.textures,
                                                                      materialResources.samplers,
                                                                      materialTextureInfos,
                                                                      lightRec);
        } else
#endif
        {
            hitLight = trace_scene_software_alpha_aware_shadow<false>(uniforms,
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
                                                                      spawn_ray_min(material, kEpsilon),
                                                                      kInfinity,
                                                                      materials,
                                                                      materialResources.textures,
                                                                      materialResources.samplers,
                                                                      materialTextureInfos,
                                                                      meshInfos,
                                                                      sceneVertices,
                                                                      meshIndices,
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
                float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                float denom = lightPdf + bsdfPdf;
                float misWeight = (denom > 0.0f) ? (lightPdf / denom) : 0.0f;
                misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                // Delta direction, not a light sample: MIS share only, no 1/lightPdf
                // (the continuation's emitter hit takes the complement).
                float3 contribution = bsdfSample.weight * mneeHit.emission * misWeight;
                // Medium transmittance over the connection segment (see volume_surface_nee_transmittance).
                contribution *= volume_surface_nee_transmittance(uniforms, distance(lightRec.point, nextOrigin), false, stats);
                if (all(isfinite(contribution))) {
                    wavefront_add_reconstruction_radiance(
                        state,
                        queuedRay.depth,
                        clamp_firefly_contribution(state.throughput.xyz,
                                                   contribution,
                                                   clampParams,
                                                   stats));
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

    if (queuedRay.depth + 1u >= max(uniforms.maxDepth, 1u)) {
        state.alive = 0u;
        wavefront_note_termination(queueCounters, stats, 3u);
        state.rngState = rngState;
        pathStates[pathIndex] = state;
        return;
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

    rayCone.width = ray_cone_width_at_distance(rayCone, hitDistanceWorld);
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
        nextRay.originMinT = float4(nextOrigin, (interface_precision(material) || source_transport_enabled(uniforms)) ? source_ray_epsilon(uniforms) : kEpsilon);
        nextRay.directionMaxT = float4(nextDirection, kInfinity);
        nextRay.mediumState =
            volume_make_global_medium_state(uniforms, mediumDepth, state.volumeScatteringDepth);
        PrimaryRayDiff nextDiff = wavefront_primary_ray_diff(queuedRay);
        const bool smoothPrefix = queuedRay.depth == 0u ||
            (queuedRay.flags & kWavefrontRayFlagPavilionSmoothDifferentials) != 0u;
        const bool nextSmoothPrefix = smoothPrefix &&
            propagate_pavilion_planar_delta_differentials(nextDiff, incomingRay,
                                                          rec, material, bsdfSample);
        wavefront_store_primary_ray_diff(nextRay, nextDiff);
        // This once-consumed optical-origin correction belongs to this ray,
        // independently of changes in the medium stack at its next boundary.
        nextRay.rayDiffDOdx.w = (interface_precision(material) || source_transport_enabled(uniforms))
            ? dot(nextOrigin - rec.point, nextDirection) : 0.0f;
        wavefront_store_ray_cone(nextRay, rayCone);
        nextRay.pixelIndex = queuedRay.pixelIndex;
        nextRay.pathIndex = pathIndex;
        nextRay.depth = queuedRay.depth + 1u;
        nextRay.flags = (bsdfSample.isDelta ? 1u : 0u) |
            (nextSmoothPrefix ? kWavefrontRayFlagPavilionSmoothDifferentials : 0u);
        compute_exclusion_indices(rec, nextRay.excludeMeshIndex, nextRay.excludePrimitiveIndex);
        nextRays[nextIndex] = nextRay;
        state.lastBsdfPdf =
            (bsdfSample.directionalPdf > 0.0f) ? bsdfSample.directionalPdf : bsdfSample.pdf;
        state.mediumDepth = mediumDepth;
        state.alive = 1u;
        state.flags = 0u;
        // This vertex's NEE sampled the mesh emitters (fused baseline draw or
        // reservoir, or the explicit ReSTIR DI stage, whose target and shading take
        // direct_light_target_cos) and could have produced the scattered direction.
        const bool fusedMeshNeeCovers = useFusedEmissiveDirect && mesh_emitter_nee_mode(uniforms) &&
            !material_closure_is_delta(material) &&
            wavefront_mesh_emitter_nee_covers_direction(uniforms, material, rec, shadingNormal,
                                                        bsdfSample.direction);
        const bool restirDiStageCovers = explicitRestirDiSurfaceAccepted &&
            uniforms.emissivePrimitiveCount > 0u && material_closure_direct_lighting_eligible(material) &&
            direct_light_target_cos(uniforms, material, rec, shadingNormal, bsdfSample.direction) > 0.0f;
        if ((fusedMeshNeeCovers || restirDiStageCovers) && (!bsdfSample.isBssrdf || subsurfaceExitVertex)) {
            state.flags |= kWavefrontFlagPreviousMeshNee;
        }
        if (bsdfSample.isDelta) {
            state.flags |= kWavefrontFlagLastScatterDelta;
            // Without a specular connection along this delta direction, the next
            // emitter hit or environment miss is the only estimate for the path.
            if (!specNeeEligible) {
                state.flags |= kWavefrontFlagLastDeltaUnpaired;
            }
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

kernel void wavefrontBuildShadowIndicesKernel(constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                              device const PathState* pathStates [[buffer(1)]],
                                              device uint* shadowIndices [[buffer(2)]],
                                              device QueueCounters* queueCounters [[buffer(3)]],
                                              uint gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
    if (gid >= pixelCount || !pathStates || !shadowIndices || !queueCounters) {
        return;
    }

    const uint slotCount = min(pathStates[gid].reserved0, wavefront_shadow_stride(uniforms));
    const uint shadowCapacity = wavefront_shadow_capacity(uniforms);
    for (uint slot = 0u; slot < slotCount; ++slot) {
        const uint outIndex =
            atomic_fetch_add_explicit(&queueCounters->reserved0, 1u, memory_order_relaxed);
        if (outIndex < shadowCapacity) {
            shadowIndices[outIndex] = gid * wavefront_shadow_stride(uniforms) + slot;
        } else {
            atomic_store_explicit(&queueCounters->overflowFlag, 1u, memory_order_relaxed);
            wavefront_note_queue_overflow(queueCounters->shadowRayQueue);
        }
    }
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
                                  device const MeshInfo* meshInfos [[buffer(19)]],
                                  device const SceneVertex* sceneVertices [[buffer(20)]],
                                  device const uint3* meshIndices [[buffer(21)]],
                                  constant MaterialTextureArgumentBuffer& materialResources [[buffer(23)]],
                                  device const MaterialTextureInfo* materialTextureInfos [[buffer(24)]],
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
        min(state.reserved0, wavefront_shadow_stride(uniforms));
    if (slotCount == 0u) {
        return;
    }

    for (uint slot = 0u; slot < slotCount; ++slot) {
        ShadowRay queued = shadowRays[gid * wavefront_shadow_stride(uniforms) + slot];
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
                                             wavefront_shadow_software_t_min(queued),
                                             queued.originTMax.w,
                                             /*anyHitOnly=*/true,
                                             /*includeTriangles=*/true,
                                             shadowRec);
        const bool referenceMaskShadow = wavefront_reference_punctual_shadow(uniforms, queued, rectangles) &&
            pavilion_mask_shadow_hit(uniforms, shadowRec, materials);
        if (source_transport_enabled(uniforms) || scene_has_source_opacity(uniforms, materials) || (occluded && (referenceMaskShadow ||
            wavefront_shadow_hit_alpha_tested(uniforms, shadowRec, materials)))) {
            occluded = trace_scene_software_alpha_aware_shadow<true>(
                uniforms, spheres, rectangles, triangleData,
                tlasNodes, tlasPrimIndices, instanceInfos, blasNodes, blasPrimIndices,
                nodes, primitiveIndices, stats, shadowRay, wavefront_shadow_software_t_min(queued), queued.originTMax.w,
                materials, materialResources.textures, materialResources.samplers,
                materialTextureInfos, meshInfos, sceneVertices, meshIndices, shadowRec, referenceMaskShadow, queued.depth, queued.sourceSigma.xyz,
                dot(shadowRay.origin-queued.sourceOrigin.xyz,shadowRay.direction));
        }
        if ((queued.flags & kWavefrontShadowFlagEmissivePrimitive) != 0u &&
            occluded &&
            shadowRec.primitiveType == kPrimitiveTypeTriangle &&
            shadowRec.primitiveIndex == queued.lightId) {
            occluded = false;
        }
        if ((queued.flags & kWavefrontShadowFlagRectangleLight) != 0u &&
            occluded &&
            shadowRec.primitiveType == kPrimitiveTypeRectangle &&
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
                (PT_DEFAULT_OFF(uniforms.enableMnee) != 0u) &&
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
                float3 envRadiance = usedDeltaChain
                    ? environment_color(environmentTexture,
                                        envDirection,
                                        uniforms.environmentRotation,
                                        uniforms.environmentIntensity,
                                        uniforms)
                    : queued.environmentRadiance.xyz;
                float3 contribution = source_shadow_weight(queued.contribution.xyz, shadowRec) * envRadiance * chainWeight;
                wavefront_add_shadow_radiance(
                    state,
                    materialResources.radianceCacheTrainRecords,
                    uniforms,
                    gid,
                    queued.depth,
                    clamp_firefly_contribution(queued.throughput.xyz,
                                               contribution,
                                               clampParams,
                                               stats));
            }
        } else if (!occluded) {
            wavefront_add_shadow_radiance(state, materialResources.radianceCacheTrainRecords, uniforms, gid, queued.depth, source_shadow_weight(queued.contribution.xyz, shadowRec));
        }
    }
    pathStates[gid] = state;
}

kernel void wavefrontShadowIndexedKernel(constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
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
                                         device const uint* shadowIndices [[buffer(16)]],
                                  device const MeshInfo* meshInfos [[buffer(19)]],
                                  device const SceneVertex* sceneVertices [[buffer(20)]],
                                  device const uint3* meshIndices [[buffer(21)]],
                                  constant MaterialTextureArgumentBuffer& materialResources [[buffer(23)]],
                                  device const MaterialTextureInfo* materialTextureInfos [[buffer(24)]],
                                  uint gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    if (gid == 0u) {
        wavefront_store_queue_stats(queueCounters, stats);
    }
    const uint activeCount =
        queueCounters ? atomic_load_explicit(&queueCounters->shadowRayQueue.activeCount, memory_order_relaxed) : 0u;
    if (gid >= activeCount || !shadowIndices) {
        return;
    }

    const uint shadowSlot = shadowIndices[gid];
    if (shadowSlot >= wavefront_shadow_capacity(uniforms)) {
        return;
    }
    ShadowRay queued = shadowRays[shadowSlot];
    const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
    if (queued.pathIndex >= pixelCount) {
        return;
    }
    PathState state = pathStates[queued.pathIndex];
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
                                         wavefront_shadow_software_t_min(queued),
                                         queued.originTMax.w,
                                         /*anyHitOnly=*/true,
                                         /*includeTriangles=*/true,
                                         shadowRec);
    const bool referenceMaskShadow = wavefront_reference_punctual_shadow(uniforms, queued, rectangles) &&
        pavilion_mask_shadow_hit(uniforms, shadowRec, materials);
    if (source_transport_enabled(uniforms) || scene_has_source_opacity(uniforms, materials) || (occluded && (referenceMaskShadow ||
        wavefront_shadow_hit_alpha_tested(uniforms, shadowRec, materials)))) {
        occluded = trace_scene_software_alpha_aware_shadow<true>(
            uniforms, spheres, rectangles, triangleData,
            tlasNodes, tlasPrimIndices, instanceInfos, blasNodes, blasPrimIndices,
            nodes, primitiveIndices, stats, shadowRay, wavefront_shadow_software_t_min(queued), queued.originTMax.w,
            materials, materialResources.textures, materialResources.samplers,
            materialTextureInfos, meshInfos, sceneVertices, meshIndices, shadowRec, referenceMaskShadow, queued.depth, queued.sourceSigma.xyz,
            dot(shadowRay.origin-queued.sourceOrigin.xyz,shadowRay.direction));
    }
    if ((queued.flags & kWavefrontShadowFlagEmissivePrimitive) != 0u &&
        occluded &&
        shadowRec.primitiveType == kPrimitiveTypeTriangle &&
        shadowRec.primitiveIndex == queued.lightId) {
        occluded = false;
    }
    if ((queued.flags & kWavefrontShadowFlagRectangleLight) != 0u &&
        occluded &&
        shadowRec.primitiveType == kPrimitiveTypeRectangle &&
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
            (PT_DEFAULT_OFF(uniforms.enableMnee) != 0u) &&
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
            float3 envRadiance = usedDeltaChain
                ? environment_color(environmentTexture,
                                    envDirection,
                                    uniforms.environmentRotation,
                                    uniforms.environmentIntensity,
                                    uniforms)
                : queued.environmentRadiance.xyz;
            float3 contribution = source_shadow_weight(queued.contribution.xyz, shadowRec) * envRadiance * chainWeight;
            wavefront_add_shadow_radiance(
                state,
                materialResources.radianceCacheTrainRecords,
                uniforms,
                queued.pathIndex,
                queued.depth,
                clamp_firefly_contribution(queued.throughput.xyz,
                                           contribution,
                                           clampParams,
                                           stats));
        }
    } else if (!occluded) {
        wavefront_add_shadow_radiance(state, materialResources.radianceCacheTrainRecords, uniforms, queued.pathIndex, queued.depth, source_shadow_weight(queued.contribution.xyz, shadowRec));
    }
    pathStates[queued.pathIndex] = state;
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
                                          constant MaterialTextureArgumentBuffer& materialResources [[buffer(23)]],
                                          device const MaterialTextureInfo* materialTextureInfos [[buffer(24)]],
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
        min(state.reserved0, wavefront_shadow_stride(uniforms));
    if (slotCount == 0u) {
        return;
    }

    for (uint slot = 0u; slot < slotCount; ++slot) {
        ShadowRay queued = shadowRays[gid * wavefront_shadow_stride(uniforms) + slot];
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
                                             wavefront_shadow_t_min(queued),
                                             queued.originTMax.w,
                                             /*anyHitOnly=*/true,
                                             queued.excludeMeshIndex,
                                             queued.excludePrimitiveIndex,
                                             shadowRec);
        const bool referenceMaskShadow = wavefront_reference_punctual_shadow(uniforms, queued, rectangles) &&
            pavilion_mask_shadow_hit(uniforms, shadowRec, materials);
        if (source_transport_enabled(uniforms) || scene_has_source_opacity(uniforms, materials) || (occluded && (referenceMaskShadow ||
            wavefront_shadow_hit_alpha_tested(uniforms, shadowRec, materials)))) {
            occluded = trace_scene_hardware_alpha_aware_shadow<true>(
                uniforms, accel, meshInfos, triangleData, sceneVertices, meshIndices,
                instanceUserIds, spheres, rectangles, nodes, primitiveIndices, stats,
                shadowRay, wavefront_shadow_t_min(queued), queued.originTMax.w,
                queued.excludeMeshIndex, queued.excludePrimitiveIndex,
                materials, materialResources.textures, materialResources.samplers,
                materialTextureInfos, shadowRec, referenceMaskShadow, queued.depth, queued.sourceSigma.xyz,
                dot(shadowRay.origin-queued.sourceOrigin.xyz,shadowRay.direction));
        }
        if ((queued.flags & kWavefrontShadowFlagEmissivePrimitive) != 0u &&
            occluded &&
            shadowRec.primitiveType == kPrimitiveTypeTriangle &&
            shadowRec.primitiveIndex == queued.lightId) {
            occluded = false;
        }
        if ((queued.flags & kWavefrontShadowFlagRectangleLight) != 0u &&
            occluded &&
            shadowRec.primitiveType == kPrimitiveTypeRectangle &&
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
                (PT_DEFAULT_OFF(uniforms.enableMnee) != 0u) &&
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
                float3 envRadiance = usedDeltaChain
                    ? environment_color(environmentTexture,
                                        envDirection,
                                        uniforms.environmentRotation,
                                        uniforms.environmentIntensity,
                                        uniforms)
                    : queued.environmentRadiance.xyz;
                float3 contribution = source_shadow_weight(queued.contribution.xyz, shadowRec) * envRadiance * chainWeight;
                wavefront_add_shadow_radiance(
                    state,
                    materialResources.radianceCacheTrainRecords,
                    uniforms,
                    gid,
                    queued.depth,
                    clamp_firefly_contribution(queued.throughput.xyz,
                                               contribution,
                                               clampParams,
                                               stats));
            }
        } else if (!occluded) {
            wavefront_add_shadow_radiance(state, materialResources.radianceCacheTrainRecords, uniforms, gid, queued.depth, source_shadow_weight(queued.contribution.xyz, shadowRec));
        }
    }
    pathStates[gid] = state;
}

kernel void wavefrontShadowIndexedHardwareKernel(constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
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
                                                 device const uint* shadowIndices [[buffer(21)]],
                                          constant MaterialTextureArgumentBuffer& materialResources [[buffer(23)]],
                                          device const MaterialTextureInfo* materialTextureInfos [[buffer(24)]],
                                          uint gid [[thread_position_in_grid]]) {
    (void)tlasNodes;
    (void)tlasPrimIndices;
    (void)blasNodes;
    (void)blasPrimIndices;
    (void)instanceInfos;

    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    if (gid == 0u) {
        wavefront_store_queue_stats(queueCounters, stats);
    }
    const uint activeCount =
        queueCounters ? atomic_load_explicit(&queueCounters->shadowRayQueue.activeCount, memory_order_relaxed) : 0u;
    if (gid >= activeCount || !shadowIndices) {
        return;
    }

    const uint shadowSlot = shadowIndices[gid];
    if (shadowSlot >= wavefront_shadow_capacity(uniforms)) {
        return;
    }
    ShadowRay queued = shadowRays[shadowSlot];
    const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
    if (queued.pathIndex >= pixelCount) {
        return;
    }
    PathState state = pathStates[queued.pathIndex];
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
                                         wavefront_shadow_t_min(queued),
                                         queued.originTMax.w,
                                         /*anyHitOnly=*/true,
                                         queued.excludeMeshIndex,
                                         queued.excludePrimitiveIndex,
                                         shadowRec);
    const bool referenceMaskShadow = wavefront_reference_punctual_shadow(uniforms, queued, rectangles) &&
        pavilion_mask_shadow_hit(uniforms, shadowRec, materials);
    if (source_transport_enabled(uniforms) || scene_has_source_opacity(uniforms, materials) || (occluded && (referenceMaskShadow ||
        wavefront_shadow_hit_alpha_tested(uniforms, shadowRec, materials)))) {
        occluded = trace_scene_hardware_alpha_aware_shadow<true>(
            uniforms, accel, meshInfos, triangleData, sceneVertices, meshIndices,
            instanceUserIds, spheres, rectangles, nodes, primitiveIndices, stats,
            shadowRay, wavefront_shadow_t_min(queued), queued.originTMax.w,
            queued.excludeMeshIndex, queued.excludePrimitiveIndex,
            materials, materialResources.textures, materialResources.samplers,
            materialTextureInfos, shadowRec, referenceMaskShadow, queued.depth, queued.sourceSigma.xyz,
            dot(shadowRay.origin-queued.sourceOrigin.xyz,shadowRay.direction));
    }
    if ((queued.flags & kWavefrontShadowFlagEmissivePrimitive) != 0u &&
        occluded &&
        shadowRec.primitiveType == kPrimitiveTypeTriangle &&
        shadowRec.primitiveIndex == queued.lightId) {
        occluded = false;
    }
    if ((queued.flags & kWavefrontShadowFlagRectangleLight) != 0u &&
        occluded &&
        shadowRec.primitiveType == kPrimitiveTypeRectangle &&
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
            (PT_DEFAULT_OFF(uniforms.enableMnee) != 0u) &&
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
            float3 envRadiance = usedDeltaChain
                ? environment_color(environmentTexture,
                                    envDirection,
                                    uniforms.environmentRotation,
                                    uniforms.environmentIntensity,
                                    uniforms)
                : queued.environmentRadiance.xyz;
            float3 contribution = source_shadow_weight(queued.contribution.xyz, shadowRec) * envRadiance * chainWeight;
            wavefront_add_shadow_radiance(
                state,
                materialResources.radianceCacheTrainRecords,
                uniforms,
                queued.pathIndex,
                queued.depth,
                clamp_firefly_contribution(queued.throughput.xyz,
                                           contribution,
                                           clampParams,
                                           stats));
        }
    } else if (!occluded) {
        wavefront_add_shadow_radiance(state, materialResources.radianceCacheTrainRecords, uniforms, queued.pathIndex, queued.depth, source_shadow_weight(queued.contribution.xyz, shadowRec));
    }
    pathStates[queued.pathIndex] = state;
}
#endif

kernel void wavefrontSwapQueuesKernel(device QueueCounters* queueCounters [[buffer(0)]],
                                      device PathtraceStats* stats [[buffer(1)]],
                                      constant PathtraceUniforms* uniformsBuffer [[buffer(2)]],
                                      device PathState* pathStates [[buffer(3)]],
                                      uint gid [[thread_position_in_grid]]) {
    if (!queueCounters) {
        return;
    }
    if (uniformsBuffer && pathStates) {
        constant PathtraceUniforms& uniforms = uniformsBuffer[0];
        const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
        if (gid < pixelCount) {
            pathStates[gid].reserved0 = 0u;
            pathStates[gid].reserved1 = 0u;
        }
    }
    if (gid == 0u) {
        const uint nextCount = atomic_load_explicit(&queueCounters->nextRayCount, memory_order_relaxed);
        atomic_store_explicit(&queueCounters->activeRayCount, nextCount, memory_order_relaxed);
        const uint boundedNextCount = min(nextCount, queueCounters->primaryRayQueue.capacity);
        atomic_store_explicit(&queueCounters->primaryRayQueue.activeCount, boundedNextCount, memory_order_relaxed);
        atomic_fetch_max_explicit(&queueCounters->primaryRayQueue.peakActiveCount,
                                  boundedNextCount,
                                  memory_order_relaxed);
        atomic_store_explicit(&queueCounters->nextRayCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&queueCounters->shadowRayCount, 0u, memory_order_relaxed);
        atomic_store_explicit(&queueCounters->reserved0, 0u, memory_order_relaxed);
        atomic_store_explicit(&queueCounters->shadowRayQueue.activeCount, 0u, memory_order_relaxed);
        wavefront_store_queue_stats(queueCounters, stats);
    }
}

// Trains the radiance cache from the records of this sample's training paths, once every bounce has been shaded and
// every shadow ray resolved, then invalidates the records (the buffer starts zeroed, so records are valid only
// between a training path's shade and this kernel). One thread per record group: group g belongs to the training
// pixel of this frame with pixelIndex / kRadianceCacheTrainingPathStride == g.
kernel void wavefrontRadianceCacheTrainKernel(constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                              device const PathState* pathStates [[buffer(1)]],
                                              device RadianceCacheState* radianceCacheStates [[buffer(2)]],
                                              device RadianceCacheTrainRecord* records [[buffer(3)]],
                                              device PathtraceStats* stats [[buffer(4)]],
                                              uint gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
    const uint stride = kRadianceCacheTrainingPathStride;
    const uint pixelIndex = gid * stride + (stride - uniforms.frameIndex % stride) % stride;
    if (pixelIndex >= pixelCount || !radiance_cache_is_training_path(uniforms, pixelIndex)) {
        return;
    }
    const float3 radianceEnd = pathStates[pixelIndex].radiance.xyz;
    const uint base = wavefront_radiance_cache_record_base(pixelIndex);
    for (uint slot = 0u; slot < kRadianceCacheTrainRecordCount; ++slot) {
        const RadianceCacheTrainRecord record = records[base + slot];
        if (!radiance_cache_train_record_valid(record)) {
            continue;
        }
        radiance_cache_train_from_record(radianceCacheStates,
                                         uniforms,
                                         record,
                                         uniforms.radianceCacheMinDepth + slot,
                                         radianceEnd,
                                         stats);
        records[base + slot].baselineMeta.w = 0.0f;
    }
}

kernel void wavefrontAccumulateKernel(texture2d<float, access::read_write> radianceTexture [[texture(0)]],
                                      texture2d<uint, access::read_write> sampleCountTexture [[texture(1)]],
                                      texture2d<float, access::read_write> albedoTexture [[texture(2)]],
                                      texture2d<float, access::read_write> normalTexture [[texture(3)]],
                                      texture2d<float, access::write> positionTexture [[texture(4)]],
                                      texture2d<float, access::write> materialFeatureTexture [[texture(5)]],
                                      texture2d<float, access::write> motionVectorTexture [[texture(6)]],
                                      texture2d<float, access::read_write> directRadianceTexture [[texture(7)]],
                                      texture2d<float, access::read_write> indirectRadianceTexture [[texture(8)]],
                                      texture2d<float, access::read_write> radianceMomentTexture [[texture(9)]],
                                      texture2d<float, access::write> reconstructionMetricTexture [[texture(10)]],
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
    float3 directSample = pathStates[gid].directRadiance.xyz;
    float3 indirectSample = pathStates[gid].indirectRadiance.xyz;
    const SpectralPathState spectral = spectral_path_state_from_hero(uniforms,
                                                                     pathStates[gid].spectralWavelengthNm,
                                                                     pathStates[gid].spectralSampleIndex);
    if (spectral.enabled) {
        // The path carried per-wavelength radiance; apply the sensor and white balance once.
        sample = spectral_channels_to_rgb(sample, spectral, uniforms);
        directSample = spectral_channels_to_rgb(directSample, spectral, uniforms);
        indirectSample = spectral_channels_to_rgb(indirectSample, spectral, uniforms);
    }
    if (!all(isfinite(sample))) {
        sample = float3(0.0f);
    } else {
        sample = max(sample, float3(0.0f));
    }

    float3 accumulated = radianceTexture.read(pixel).xyz;
    uint previousCount = sampleCountTexture.read(pixel).x;
    radianceTexture.write(float4(accumulated + sample, 0.0f), pixel);
    const uint newCount = previousCount + 1u;
    sampleCountTexture.write(newCount, pixel);
    directRadianceTexture.write(float4(directRadianceTexture.read(pixel).xyz +
                                           max(directSample, float3(0.0f)),
                                       0.0f),
                                pixel);
    indirectRadianceTexture.write(float4(indirectRadianceTexture.read(pixel).xyz +
                                             max(indirectSample, float3(0.0f)),
                                         0.0f),
                                  pixel);
    const float sampleLuma = luminance_rgb(sample);
    radianceMomentTexture.write(float4(radianceMomentTexture.read(pixel).x +
                                           sampleLuma * sampleLuma,
                                       0.0f,
                                       0.0f,
                                       0.0f),
                                pixel);
    reconstructionMetricTexture.write(float4(pathStates[gid].reconstructionVariance,
                                             max(clamp(log2(float(newCount) + 1.0f) / 8.0f, 0.0f, 1.0f),
                                                 clamp(pathStates[gid].reconstructionConfidence, 0.0f, 1.0f)),
                                             clamp(pathStates[gid].reconstructionBiasRisk, 0.0f, 1.0f),
                                             float(newCount)),
                                      pixel);
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
    if (PT_DEFAULT_OFF(uniforms.debugViewMode) == 17u && queueCounters) {
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
    // Accumulate primary guides along with color, as Falcor does. The shared
    // helper's alpha history tag carries the average through jittered sky
    // misses and ignores stale guide textures after a radiance reset.
    write_pathtrace_denoise_guides(albedoTexture, normalTexture, pixel, previousCount,
                                  PT_DEFAULT_OFF(uniforms.debugViewMode) == kDebugViewNone &&
                                      uniforms.oidnDeltaGuidesAllowed != 0u,
                                  firstHitAlbedo, firstHitNormal);
    positionTexture.write(firstHitPosition, pixel);
    materialFeatureTexture.write(firstHitMaterial, pixel);
    motionVectorTexture.write(float4(0.0f), pixel);
}
