constant float kRadianceCacheQuantizationScale = 4096.0f;
constant uint kRadianceCacheFlagValid = 1u;
constant uint kRadianceCacheFlagLowConfidence = 2u;
constant uint kRadianceCacheFlagTrained = 4u;
constant uint kRadianceCacheFlagInvalidated = 8u;
constant uint kRadianceCacheFallbackModeOff = 1u;
constant uint kRadianceCacheFallbackMaterial = 2u;
constant uint kRadianceCacheFallbackSample = 3u;
constant uint kRadianceCacheFallbackEmpty = 4u;
constant uint kRadianceCacheFallbackLowConfidence = 5u;
constant uint kRadianceCacheFallbackInvalid = 6u;
constant uint kRadianceCacheFallbackMetadataMismatch = 7u;
constant uint kRadianceCacheFallbackBiasRisk = 8u;
constant uint kRadianceCacheFallbackFireflyPolicy = 9u;
constant uint kRadianceCacheMaxStoredSamples = 1024u;

inline uint radiance_cache_capacity(constant PathtraceUniforms& uniforms) {
    return max(uniforms.width * uniforms.height * kRestirPathReservoirDepthCount, 1u);
}

inline uint radiance_cache_domain_key(constant PathtraceUniforms& uniforms,
                                      const float3 position,
                                      const uint depth,
                                      const uint materialType,
                                      const uint lobeType) {
    const float invCell = 1.0f / max(uniforms.radianceCacheCellSize, 1.0e-4f);
    const int3 cell = int3(floor(position * invCell));
    const uint hx = as_type<uint>(cell.x) * 73856093u;
    const uint hy = as_type<uint>(cell.y) * 19349663u;
    const uint hz = as_type<uint>(cell.z) * 83492791u;
    const uint hd = min(depth, kRestirPathReservoirDepthCount - 1u) * 2654435761u;
    const uint hm = materialType * 2246822519u;
    const uint hl = lobeType * 3266489917u;
    return pcg_hash(hx ^ hy ^ hz ^ hd ^ hm ^ hl ^ 0x9E3779B9u);
}

inline uint radiance_cache_index_for_query(constant PathtraceUniforms& uniforms,
                                           const RadianceQuery query) {
    return query.materialLobeMeta.w % radiance_cache_capacity(uniforms);
}

inline uint radiance_cache_quantize(const float value, const float maxRadiance) {
    const float clamped = clamp(value, 0.0f, max(maxRadiance, 1.0e-4f));
    return static_cast<uint>(round(clamped * kRadianceCacheQuantizationScale));
}

inline uint radiance_cache_quantize_unit(const float value) {
    return static_cast<uint>(round(clamp(value, 0.0f, 1.0f) * 65535.0f));
}

inline float radiance_cache_dequantize(const uint value) {
    return float(value) / kRadianceCacheQuantizationScale;
}

inline float radiance_cache_dequantize_unit(const uint value) {
    return clamp(float(value) / 65535.0f, 0.0f, 1.0f);
}

inline bool radiance_cache_state_matches_query(const RadianceCacheState state,
                                               const RadianceQuery query) {
    return state.domainKey == query.materialLobeMeta.w &&
           state.materialType == query.materialLobeMeta.x &&
           state.lobeType == query.materialLobeMeta.y;
}

inline bool radiance_cache_state_is_live_for_query(const RadianceCacheState state,
                                                   const RadianceQuery query) {
    return (state.flags & kRadianceCacheFlagValid) != 0u &&
           (state.flags & kRadianceCacheFlagInvalidated) == 0u &&
           state.invalidationState == 0u &&
           state.sampleCount > 0u &&
           radiance_cache_state_matches_query(state, query);
}

inline void radiance_cache_reset_cell_for_query(device RadianceCacheState& state,
                                                const RadianceQuery query,
                                                const uint frameIndex) {
    atomic_store_explicit(reinterpret_cast<device atomic_uint*>(&state.radianceR),
                          0u,
                          memory_order_relaxed);
    atomic_store_explicit(reinterpret_cast<device atomic_uint*>(&state.radianceG),
                          0u,
                          memory_order_relaxed);
    atomic_store_explicit(reinterpret_cast<device atomic_uint*>(&state.radianceB),
                          0u,
                          memory_order_relaxed);
    atomic_store_explicit(reinterpret_cast<device atomic_uint*>(&state.lumaMoment),
                          0u,
                          memory_order_relaxed);
    atomic_store_explicit(reinterpret_cast<device atomic_uint*>(&state.sampleCount),
                          0u,
                          memory_order_relaxed);
    state.positionRoughness = query.positionRoughness;
    state.normalVariance = float4(safe_normalize(query.normalDepth.xyz), 1.0f);
    state.frameTag = frameIndex;
    state.flags = kRadianceCacheFlagInvalidated;
    state.domainKey = query.materialLobeMeta.w;
    state.materialType = query.materialLobeMeta.x;
    state.lobeType = query.materialLobeMeta.y;
    state.confidenceQ = 0u;
    state.invalidationState = 0u;
}

inline RadianceQuery radiance_cache_make_query(constant PathtraceUniforms& uniforms,
                                               const MaterialData material,
                                               const float3 position,
                                               const float3 normal,
                                               const uint depth,
                                               const BsdfSampleResult sample) {
    RadianceQuery query;
    const uint materialType = static_cast<uint>(material.typeEta.x);
    const uint lobeType = sample.lobeType;
    const float roughness = clamp(max(material_roughness(material), sample.lobeRoughness), 0.0f, 1.0f);
    const float3 n = safe_normalize(normal);
    const float biasRisk = sample.isDelta ? 1.0f : (lobeType == 0u ? 0.0f : 0.75f);
    const uint domainKey = radiance_cache_domain_key(uniforms, position, depth, materialType, lobeType);
    query.positionRoughness = float4(position, roughness);
    query.normalDepth = float4(n, float(depth));
    query.outgoingBiasRisk = float4(safe_normalize(sample.direction), biasRisk);
    query.materialLobeMeta = uint4(materialType, lobeType, 0u, domainKey);
    return query;
}

inline bool radiance_cache_can_query(const RadianceQuery query,
                                     const MaterialData material,
                                     const bool specularOnly,
                                     const BsdfSampleResult sample,
                                     const uint depth,
                                     constant PathtraceUniforms& uniforms,
                                     thread uint& fallbackReason) {
    fallbackReason = kRadianceCacheFallbackModeOff;
    if (uniforms.radianceCacheMode != kRadianceCacheModePrototype) {
        return false;
    }
    fallbackReason = kRadianceCacheFallbackMaterial;
    if (depth < uniforms.radianceCacheMinDepth ||
        specularOnly ||
        !material_closure_is_cacheable(material) ||
        !material_closure_restir_diffuse_eligible(material)) {
        return false;
    }
    fallbackReason = kRadianceCacheFallbackSample;
    if (sample.isDelta ||
        sample.isBssrdf ||
        sample.hasExitPoint ||
        sample.mediumEvent != 0 ||
        sample.lobeType != 0u ||
        sample.pdf <= 0.0f ||
        sample.directionalPdf <= 0.0f ||
        !all(isfinite(query.positionRoughness.xyz)) ||
        !all(isfinite(query.normalDepth.xyz)) ||
        !all(isfinite(sample.weight))) {
        return false;
    }
    fallbackReason = 0u;
    return true;
}

inline bool radiance_cache_load(device const RadianceCacheState* radianceCacheStates,
                                constant PathtraceUniforms& uniforms,
                                const RadianceQuery query,
                                thread RadianceEstimate& estimate) {
    estimate.radianceConfidence = float4(0.0f);
    estimate.varianceBiasRisk = float4(1.0f, 1.0f, 0.0f, 0.0f);
    estimate.sampleAgeFallback = uint4(0u, 0u, 0u, kRadianceCacheFallbackModeOff);

    if (uniforms.radianceCacheMode != kRadianceCacheModePrototype || !radianceCacheStates) {
        return false;
    }

    const uint index = radiance_cache_index_for_query(uniforms, query);
    const RadianceCacheState state = radianceCacheStates[index];
    if ((state.flags & kRadianceCacheFlagValid) == 0u || state.sampleCount == 0u) {
        estimate.sampleAgeFallback.w = kRadianceCacheFallbackEmpty;
        return false;
    }
    if ((state.flags & kRadianceCacheFlagInvalidated) != 0u ||
        state.invalidationState != 0u ||
        state.frameTag == uniforms.frameIndex) {
        estimate.sampleAgeFallback.w = kRadianceCacheFallbackInvalid;
        return false;
    }
    if (!radiance_cache_state_matches_query(state, query)) {
        estimate.sampleAgeFallback.w = kRadianceCacheFallbackMetadataMismatch;
        return false;
    }

    const float normalDot = dot(safe_normalize(state.normalVariance.xyz),
                                safe_normalize(query.normalDepth.xyz));
    const float roughnessDelta = fabs(state.positionRoughness.w - query.positionRoughness.w);
    if (normalDot < 0.65f || roughnessDelta > 0.35f) {
        estimate.sampleAgeFallback.w = kRadianceCacheFallbackMetadataMismatch;
        return false;
    }

    const float samples = float(max(state.sampleCount, 1u));
    const float3 radiance = float3(radiance_cache_dequantize(state.radianceR),
                                   radiance_cache_dequantize(state.radianceG),
                                   radiance_cache_dequantize(state.radianceB)) / samples;
    const float luma = luminance_rgb(max(radiance, float3(0.0f)));
    const float meanLuma2 = radiance_cache_dequantize(state.lumaMoment) / samples;
    const float storedVariance = max(state.normalVariance.w, 0.0f);
    const float variance = max(max(meanLuma2 - luma * luma, 0.0f), storedVariance);
    const float variancePenalty = 1.0f / (1.0f + variance);
    const float sampleConfidence = clamp(log2(samples + 1.0f) / 6.0f, 0.0f, 1.0f);
    const float age = float(uniforms.frameIndex - min(state.frameTag, uniforms.frameIndex));
    const float ageConfidence = 1.0f / (1.0f + 0.04f * age);
    const float storedConfidence = radiance_cache_dequantize_unit(state.confidenceQ);
    const float confidence =
        clamp(max(storedConfidence, sampleConfidence * variancePenalty * ageConfidence), 0.0f, 1.0f);
    const float biasRisk = clamp(max(query.outgoingBiasRisk.w,
                                     variance / max(luma * luma + 1.0e-4f, 1.0e-4f)),
                                 0.0f,
                                 1.0f);

    estimate.radianceConfidence = float4(radiance, confidence);
    estimate.varianceBiasRisk = float4(variance, biasRisk, normalDot, roughnessDelta);
    estimate.sampleAgeFallback = uint4(state.sampleCount,
                                       static_cast<uint>(max(age, 0.0f)),
                                       state.flags,
                                       0u);
    if (!all(isfinite(radiance))) {
        estimate.sampleAgeFallback.w = kRadianceCacheFallbackInvalid;
        return false;
    }
    if (biasRisk > 0.85f) {
        estimate.sampleAgeFallback.w = kRadianceCacheFallbackBiasRisk;
        return false;
    }
    if (confidence < uniforms.radianceCacheMinConfidence) {
        estimate.sampleAgeFallback.w = kRadianceCacheFallbackLowConfidence;
        return false;
    }
    return true;
}

inline bool radiance_cache_train(device RadianceCacheState* radianceCacheStates,
                                 constant PathtraceUniforms& uniforms,
                                 const MaterialData material,
                                 const float3 position,
                                 const float3 normal,
                                 const bool specularOnly,
                                 const uint depth,
                                 const BsdfSampleResult sample,
                                 device PathtraceStats* stats) {
    if (!radianceCacheStates) {
        return false;
    }
    RadianceQuery query = radiance_cache_make_query(uniforms, material, position, normal, depth, sample);
    uint fallbackReason = 0u;
    if (!radiance_cache_can_query(query, material, specularOnly, sample, depth, uniforms, fallbackReason)) {
        return false;
    }
    if ((uniforms.frameIndex % max(uniforms.radianceCacheTrainingInterval, 1u)) != 0u) {
        return false;
    }

    const float3 trainingRadiance = max(sample.weight, float3(0.0f));
    const float luma = luminance_rgb(trainingRadiance);
    const float maxRadiance = max(uniforms.radianceCacheMaxRadiance, 1.0e-4f);
    if (!all(isfinite(trainingRadiance)) ||
        !isfinite(luma) ||
        any(trainingRadiance > float3(maxRadiance)) ||
        luma > maxRadiance) {
        if (stats) {
            atomic_fetch_add_explicit(&stats->radianceCacheInvalidCount, 1u, memory_order_relaxed);
        }
        return false;
    }
    if (!(luma > 0.0f)) {
        return false;
    }

    const uint index = radiance_cache_index_for_query(uniforms, query);
    device RadianceCacheState& state = radianceCacheStates[index];
    const RadianceCacheState previousState = state;
    if (!radiance_cache_state_is_live_for_query(previousState, query)) {
        if (previousState.sampleCount > 0u &&
            !radiance_cache_state_matches_query(previousState, query) &&
            stats) {
            atomic_fetch_add_explicit(&stats->radianceCacheInvalidCount, 1u, memory_order_relaxed);
        }
        radiance_cache_reset_cell_for_query(state, query, uniforms.frameIndex);
    }

    const uint previousSamples =
        atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&state.sampleCount),
                                  1u,
                                  memory_order_relaxed);
    if (previousSamples >= kRadianceCacheMaxStoredSamples) {
        atomic_store_explicit(reinterpret_cast<device atomic_uint*>(&state.sampleCount),
                              kRadianceCacheMaxStoredSamples,
                              memory_order_relaxed);
        if (stats) {
            atomic_fetch_add_explicit(&stats->radianceCacheFallbackCount, 1u, memory_order_relaxed);
        }
        return false;
    }
    atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&state.radianceR),
                              radiance_cache_quantize(trainingRadiance.x, maxRadiance),
                              memory_order_relaxed);
    atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&state.radianceG),
                              radiance_cache_quantize(trainingRadiance.y, maxRadiance),
                              memory_order_relaxed);
    atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&state.radianceB),
                              radiance_cache_quantize(trainingRadiance.z, maxRadiance),
                              memory_order_relaxed);
    atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&state.lumaMoment),
                              radiance_cache_quantize(luma * luma, maxRadiance),
                              memory_order_relaxed);

    const float samples = float(max(previousSamples + 1u, 1u));
    const float confidence = clamp(log2(samples + 1.0f) / 6.0f, 0.0f, 1.0f);
    const float varianceProxy = samples > 1.0f ? (luma * luma) / samples : 1.0f;
    state.positionRoughness = query.positionRoughness;
    state.normalVariance = float4(safe_normalize(normal), varianceProxy);
    state.frameTag = uniforms.frameIndex;
    state.flags = kRadianceCacheFlagValid | kRadianceCacheFlagTrained;
    state.domainKey = query.materialLobeMeta.w;
    state.materialType = query.materialLobeMeta.x;
    state.lobeType = query.materialLobeMeta.y;
    state.confidenceQ = radiance_cache_quantize_unit(confidence);
    state.invalidationState = 0u;
    if (stats) {
        atomic_fetch_add_explicit(&stats->radianceCacheTrainCount, 1u, memory_order_relaxed);
    }
    return true;
}

inline bool radiance_cache_query_and_maybe_terminate(
    constant PathtraceUniforms& uniforms,
    const MaterialData material,
    const float3 position,
    const float3 normal,
    const bool specularOnly,
    const uint depth,
    const BsdfSampleResult sample,
    device RadianceCacheState* radianceCacheStates,
    device PathtraceStats* stats,
    thread float3& radiance,
    const float3 throughput) {
    RadianceQuery query = radiance_cache_make_query(uniforms, material, position, normal, depth, sample);
    uint fallbackReason = 0u;
    if (!radiance_cache_can_query(query, material, specularOnly, sample, depth, uniforms, fallbackReason)) {
        if (stats && uniforms.radianceCacheMode == kRadianceCacheModePrototype) {
            atomic_fetch_add_explicit(&stats->radianceCacheFallbackCount, 1u, memory_order_relaxed);
        }
        return false;
    }
    if (stats) {
        atomic_fetch_add_explicit(&stats->radianceCacheQueryCount, 1u, memory_order_relaxed);
    }

    RadianceEstimate estimate;
    const bool hit = radiance_cache_load(radianceCacheStates, uniforms, query, estimate);
    if (!hit) {
        if (stats) {
            atomic_fetch_add_explicit(&stats->radianceCacheMissCount, 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&stats->radianceCacheFallbackCount, 1u, memory_order_relaxed);
        }
        return false;
    }

    const float3 contribution = throughput * estimate.radianceConfidence.xyz;
    if (!all(isfinite(contribution))) {
        if (stats) {
            atomic_fetch_add_explicit(&stats->radianceCacheInvalidCount, 1u, memory_order_relaxed);
        }
        return false;
    }
    radiance += contribution;
    if (stats) {
        atomic_fetch_add_explicit(&stats->radianceCacheHitCount, 1u, memory_order_relaxed);
    }
    return true;
}
