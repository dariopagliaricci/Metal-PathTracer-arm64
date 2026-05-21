inline uint2 wavefront_pixel_coord(const uint pixelIndex,
                                   constant PathtraceUniforms& uniforms) {
    uint width = max(uniforms.width, 1u);
    return uint2(pixelIndex % width, pixelIndex / width);
}

inline uint wavefront_shadow_capacity(constant PathtraceUniforms& uniforms) {
    return max(uniforms.width * uniforms.height * kWavefrontShadowCapacityMultiplier, 1u);
}

inline void wavefront_init_queue_header(device WavefrontQueueHeader& header,
                                        const uint queueType,
                                        const uint capacity,
                                        const uint frameIndex,
                                        const uint debugLabel,
                                        const uint producerPass,
                                        const uint consumerPass) {
    atomic_store_explicit(&header.activeCount, 0u, memory_order_relaxed);
    header.capacity = capacity;
    atomic_store_explicit(&header.overflowCount, 0u, memory_order_relaxed);
    header.queueType = queueType;
    header.frameIndex = frameIndex;
    header.debugLabel = debugLabel;
    header.producerPass = producerPass;
    header.consumerPass = consumerPass;
    header.lastResetFrame = frameIndex;
    atomic_store_explicit(&header.peakActiveCount, 0u, memory_order_relaxed);
    header.storagePolicy = capacity > 0u ? kWavefrontQueueStoragePhysical
                                         : kWavefrontQueueStorageMetadataOnly;
    header.reserved2 = 0u;
}

inline uint wavefront_note_queue_enqueue(device WavefrontQueueHeader& header) {
    const uint previous = atomic_fetch_add_explicit(&header.activeCount, 1u, memory_order_relaxed);
    atomic_fetch_max_explicit(&header.peakActiveCount, previous + 1u, memory_order_relaxed);
    return previous;
}

inline void wavefront_note_queue_overflow(device WavefrontQueueHeader& header) {
    atomic_fetch_add_explicit(&header.overflowCount, 1u, memory_order_relaxed);
}

inline uint wavefront_reserve_shadow_slot(constant PathtraceUniforms& uniforms,
                                          const uint pathIndex,
                                          thread PathState& state,
                                          device QueueCounters* queueCounters,
                                          device PathtraceStats* stats) {
    if (queueCounters) {
        atomic_fetch_add_explicit(&queueCounters->shadowRayCount, 1u, memory_order_relaxed);
    }
    if (stats) {
        atomic_fetch_add_explicit(&stats->wavefrontShadowRayEnqueueCount,
                                  1u,
                                  memory_order_relaxed);
    }

    const uint pixelCount = max(uniforms.width * uniforms.height, 1u);
    if (pathIndex >= pixelCount) {
        if (queueCounters) {
            atomic_store_explicit(&queueCounters->overflowFlag, 1u, memory_order_relaxed);
            wavefront_note_queue_overflow(queueCounters->shadowRayQueue);
        }
        return kInvalidIndex;
    }

    const uint localSlot = state.reserved0;
    state.reserved0 += 1u;
    if (localSlot >= kWavefrontShadowCapacityMultiplier) {
        if (queueCounters) {
            atomic_store_explicit(&queueCounters->overflowFlag, 1u, memory_order_relaxed);
            wavefront_note_queue_overflow(queueCounters->shadowRayQueue);
        }
        return kInvalidIndex;
    }
    if (queueCounters) {
        wavefront_note_queue_enqueue(queueCounters->shadowRayQueue);
    }
    return pathIndex * kWavefrontShadowCapacityMultiplier + localSlot;
}

inline void wavefront_store_queue_stats(device QueueCounters* queueCounters,
                                        device PathtraceStats* stats) {
    if (!queueCounters || !stats) {
        return;
    }
    atomic_store_explicit(&stats->wavefrontActiveRayCount,
                          atomic_load_explicit(&queueCounters->activeRayCount, memory_order_relaxed),
                          memory_order_relaxed);
    atomic_store_explicit(&stats->wavefrontNextRayCount,
                          atomic_load_explicit(&queueCounters->nextRayCount, memory_order_relaxed),
                          memory_order_relaxed);
    atomic_store_explicit(&stats->wavefrontShadowRayCount,
                          atomic_load_explicit(&queueCounters->shadowRayCount, memory_order_relaxed),
                          memory_order_relaxed);
    atomic_store_explicit(&stats->wavefrontTerminatedCount,
                          atomic_load_explicit(&queueCounters->terminatedCount, memory_order_relaxed),
                          memory_order_relaxed);
    atomic_store_explicit(&stats->wavefrontOverflowFlag,
                          atomic_load_explicit(&queueCounters->overflowFlag, memory_order_relaxed),
                          memory_order_relaxed);
}

inline void wavefront_note_termination(device QueueCounters* queueCounters,
                                       device PathtraceStats* stats,
                                       const uint reason) {
    if (queueCounters) {
        atomic_fetch_add_explicit(&queueCounters->terminatedCount, 1u, memory_order_relaxed);
    }
    if (!stats) {
        return;
    }
    switch (reason) {
        case 1u:
            atomic_fetch_add_explicit(&stats->wavefrontTerminateMissCount, 1u, memory_order_relaxed);
            break;
        case 2u:
            atomic_fetch_add_explicit(&stats->wavefrontTerminateEmissiveCount, 1u, memory_order_relaxed);
            break;
        case 3u:
            atomic_fetch_add_explicit(&stats->wavefrontTerminateMaxDepthCount, 1u, memory_order_relaxed);
            break;
        case 4u:
            atomic_fetch_add_explicit(&stats->wavefrontTerminateOverflowCount, 1u, memory_order_relaxed);
            break;
        default:
            break;
    }
}

inline WavefrontHitRecord pack_wavefront_hit(thread const HitRecord& rec,
                                             const uint pathIndex,
                                             const bool hit) {
    WavefrontHitRecord packed{};
    packed.pointT = float4(rec.point, rec.t);
    packed.normalFrontFace = float4(rec.normal, float(rec.frontFace));
    packed.shadingNormalPrimitiveType = float4(rec.shadingNormal, float(rec.primitiveType));
    packed.ids = uint4(rec.materialIndex, rec.primitiveIndex, rec.meshIndex, pathIndex);
    packed.barycentric = rec.barycentric;
    packed.hit = hit ? 1u : 0u;
    return packed;
}

inline HitRecord unpack_wavefront_hit(thread const WavefrontHitRecord& packed) {
    HitRecord rec = make_empty_hit_record();
    rec.point = packed.pointT.xyz;
    rec.t = packed.pointT.w;
    rec.normal = packed.normalFrontFace.xyz;
    rec.frontFace = packed.normalFrontFace.w > 0.5f ? 1u : 0u;
    rec.shadingNormal = packed.shadingNormalPrimitiveType.xyz;
    rec.primitiveType = static_cast<uint>(packed.shadingNormalPrimitiveType.w);
    rec.materialIndex = packed.ids.x;
    rec.primitiveIndex = packed.ids.y;
    rec.meshIndex = packed.ids.z;
    rec.barycentric = packed.barycentric;
    return rec;
}

inline float3 wavefront_background_radiance(constant PathtraceUniforms& uniforms,
                                            texture2d<float, access::sample> environmentTexture,
                                            const float3 direction) {
    float3 background = sky_color(direction);
    if (uniforms.backgroundMode == 1u) {
        background = uniforms.backgroundColor;
    } else if (uniforms.backgroundMode == 2u &&
               environmentTexture.get_width() > 0 &&
               environmentTexture.get_height() > 0) {
        background = environment_color(environmentTexture,
                                       direction,
                                       uniforms.environmentRotation,
                                       uniforms.environmentIntensity,
                                       uniforms);
    }
    if (uniforms.backgroundMode != 2u) {
        background = to_working_space(background, uniforms);
    }
    return background;
}
