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

// Adds to a 64-bit diagnostics counter (PathtraceStats::radianceCacheStats): the low word of the value to the low
// word, its high word and the carry of that add (seen by the thread whose add wraps) to the high word.
inline void radiance_cache_stat_add(device PathtraceStats* stats, const uint index, const ulong value) {
    if (!stats || value == 0ul) {
        return;
    }
    device RadianceCacheStatCounter& counter = stats->radianceCacheStats[index];
    const uint low = static_cast<uint>(value & 0xFFFFFFFFul);
    uint high = static_cast<uint>(value >> 32);
    if (low != 0u) {
        const uint previous = atomic_fetch_add_explicit(&counter.lo, low, memory_order_relaxed);
        if (previous > 0xFFFFFFFFu - low) {
            high += 1u;
        }
    }
    if (high != 0u) {
        atomic_fetch_add_explicit(&counter.hi, high, memory_order_relaxed);
    }
}

// Records one atomic add of `added` to a cell's fixed-point sum whose previous value was `previous`: the largest sum
// reached (kRadianceCacheStatMaxStoredSumBase + channel, a running maximum in the low word) or, if the add wrapped
// past UINT32_MAX, a kRadianceCacheStatWrappedSums count.
inline void radiance_cache_stat_note_sum(device PathtraceStats* stats,
                                         const uint channel,
                                         const uint previous,
                                         const uint added) {
    if (!stats) {
        return;
    }
    if (previous > 0xFFFFFFFFu - added) {
        radiance_cache_stat_add(stats, kRadianceCacheStatWrappedSums, 1ul);
        return;
    }
    atomic_fetch_max_explicit(&stats->radianceCacheStats[kRadianceCacheStatMaxStoredSumBase + channel].lo,
                              previous + added,
                              memory_order_relaxed);
}

// Mirror of RadianceCacheTraining::StatLumaToFixed.
inline ulong radiance_cache_stat_luma_fixed(const float luma) {
    if (!isfinite(luma)) {
        return 0ul;
    }
    return static_cast<ulong>(round(clamp(luma, 0.0f, kRadianceCacheStatLumaAddMax) * kRadianceCacheStatLumaScale));
}

inline uint radiance_cache_stat_depth_bin(const uint depth) {
    return min(depth, kRestirPathReservoirDepthCount - 1u);
}

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

inline float radiance_cache_max_radiance(constant PathtraceUniforms& uniforms) {
    return max(uniforms.radianceCacheMaxRadiance, 1.0e-4f);
}

// Fixed-point scale for a cell's sums (mirror of RadianceCacheTraining::QuantizationScale): the largest power of
// two s with kRadianceCacheMaxStoredSamples * maxValue * s <= 4.0e9, so a full cell never wraps a uint. Pass
// maxRadiance for the RGB sums and maxRadiance^2 for the luminance moment. frexp gives the exponent exactly.
inline float radiance_cache_quantization_scale(const float maxValue) {
    const float bound = 4.0e9f / (float(kRadianceCacheMaxStoredSamples) * max(maxValue, 1.0e-8f));
    int exponent = 0;
    (void)frexp(bound, exponent);
    return ldexp(1.0f, exponent - 1);
}

inline uint radiance_cache_quantize(const float value, const float maxValue, const float scale) {
    return static_cast<uint>(round(clamp(value, 0.0f, maxValue) * scale));
}

inline uint radiance_cache_quantize_unit(const float value) {
    return static_cast<uint>(round(clamp(value, 0.0f, 1.0f) * 65535.0f));
}

inline float radiance_cache_dequantize(const uint value, const float scale) {
    return float(value) / scale;
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
    const MaterialClosureMetadata closure = material_closure_metadata_for_sample(material, sample);
    const uint materialType = closure.type;
    const uint lobeType = closure.lobeIndex;
    const float roughness = clamp(closure.roughness, 0.0f, 1.0f);
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
    if (PT_DEFAULT_OFF(uniforms.radianceCacheMode) != kRadianceCacheModePrototype) {
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

    if (PT_DEFAULT_OFF(uniforms.radianceCacheMode) != kRadianceCacheModePrototype || !radianceCacheStates) {
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
    const float maxRadiance = radiance_cache_max_radiance(uniforms);
    const float sumScale = radiance_cache_quantization_scale(maxRadiance);
    const float momentScale = radiance_cache_quantization_scale(maxRadiance * maxRadiance);
    const float3 radiance = float3(radiance_cache_dequantize(state.radianceR, sumScale),
                                   radiance_cache_dequantize(state.radianceG, sumScale),
                                   radiance_cache_dequantize(state.radianceB, sumScale)) / samples;
    const float luma = luminance_rgb(max(radiance, float3(0.0f)));
    const float meanLuma2 = radiance_cache_dequantize(state.lumaMoment, momentScale) / samples;
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

// Clamp-and-count (mirror of RadianceCacheTraining::ClampTarget): a target whose largest channel exceeds
// maxRadiance is scaled down to it, keeping the hue; clampedLuma is the luminance removed.
inline float3 radiance_cache_clamp_target(const float3 target,
                                          const float maxRadiance,
                                          thread bool& clamped,
                                          thread float& clampedLuma) {
    const float peak = max(target.x, max(target.y, target.z));
    clamped = peak > maxRadiance;
    clampedLuma = 0.0f;
    if (!clamped) {
        return target;
    }
    const float3 scaled = target * (maxRadiance / peak);
    clampedLuma = max(luminance_rgb(target) - luminance_rgb(scaled), 0.0f);
    return scaled;
}

// Adds one training sample to the cell of `query`. `target` must be finite, non-negative and at most
// maxRadiance per channel (radiance_cache_clamp_target).
inline bool radiance_cache_store_sample(device RadianceCacheState* radianceCacheStates,
                                        constant PathtraceUniforms& uniforms,
                                        const RadianceQuery query,
                                        const float3 target,
                                        device PathtraceStats* stats) {
    const uint index = radiance_cache_index_for_query(uniforms, query);
    device RadianceCacheState& state = radianceCacheStates[index];
    const RadianceCacheState previousState = state;
    if (!radiance_cache_state_is_live_for_query(previousState, query)) {
        if (previousState.sampleCount > 0u &&
            !radiance_cache_state_matches_query(previousState, query) &&
            stats) {
            atomic_fetch_add_explicit(&stats->radianceCacheInvalidCount, 1u, memory_order_relaxed);
            radiance_cache_stat_add(stats, kRadianceCacheStatCellEvictions, 1u);
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
            radiance_cache_stat_add(stats, kRadianceCacheStatTrainSaturated, 1u);
        }
        return false;
    }
    const float maxRadiance = radiance_cache_max_radiance(uniforms);
    const float sumScale = radiance_cache_quantization_scale(maxRadiance);
    const float momentScale = radiance_cache_quantization_scale(maxRadiance * maxRadiance);
    const float luma = luminance_rgb(target);
    const uint addR = radiance_cache_quantize(target.x, maxRadiance, sumScale);
    const uint addG = radiance_cache_quantize(target.y, maxRadiance, sumScale);
    const uint addB = radiance_cache_quantize(target.z, maxRadiance, sumScale);
    const uint addMoment = radiance_cache_quantize(luma * luma, maxRadiance * maxRadiance, momentScale);
    radiance_cache_stat_note_sum(stats, 0u,
                                 atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&state.radianceR),
                                                           addR, memory_order_relaxed),
                                 addR);
    radiance_cache_stat_note_sum(stats, 1u,
                                 atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&state.radianceG),
                                                           addG, memory_order_relaxed),
                                 addG);
    radiance_cache_stat_note_sum(stats, 2u,
                                 atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&state.radianceB),
                                                           addB, memory_order_relaxed),
                                 addB);
    radiance_cache_stat_note_sum(stats, 3u,
                                 atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&state.lumaMoment),
                                                           addMoment, memory_order_relaxed),
                                 addMoment);

    const float samples = float(max(previousSamples + 1u, 1u));
    const float confidence = clamp(log2(samples + 1.0f) / 6.0f, 0.0f, 1.0f);
    const float varianceProxy = samples > 1.0f ? (luma * luma) / samples : 1.0f;
    state.positionRoughness = query.positionRoughness;
    state.normalVariance = float4(safe_normalize(query.normalDepth.xyz), varianceProxy);
    state.frameTag = uniforms.frameIndex;
    state.flags = kRadianceCacheFlagValid | kRadianceCacheFlagTrained;
    state.domainKey = query.materialLobeMeta.w;
    state.materialType = query.materialLobeMeta.x;
    state.lobeType = query.materialLobeMeta.y;
    state.confidenceQ = radiance_cache_quantize_unit(confidence);
    state.invalidationState = 0u;
    if (stats) {
        atomic_fetch_add_explicit(&stats->radianceCacheTrainCount, 1u, memory_order_relaxed);
        radiance_cache_stat_add(stats, kRadianceCacheStatTrainAccepted, 1u);
    }
    return true;
}

// ---- Deferred training ------------------------------------------------------------------------------------
// A training path (one pixel in kRadianceCacheTrainingPathStride per frame, chosen without the RNG) never loads
// from the cache. At each cacheable vertex x it records the throughput at the query point T_q and the radiance
// accumulated so far R_b (x's own emission and NEE included). When the path ends, each record trains its cell with
// (R_end - R_b) / T_q: what a cache hit at x would have replaced, per unit of throughput. Mirrors
// include/renderer/RadianceCacheTraining.h; tested there (CTest radiance_cache_training_target).

constant uint kRadianceCacheTargetAccepted = 0u;
constant uint kRadianceCacheTargetClamped = 1u;
constant uint kRadianceCacheTargetRejectedNonFinite = 2u;
constant uint kRadianceCacheTargetRejectedZeroThroughput = 3u;

struct RadianceCacheTarget {
    float3 target;
    uint status;
    float targetLuma;
    float clampedLuma;
};

inline RadianceCacheTarget radiance_cache_training_target(const float3 throughputAtQuery,
                                                          const float3 radianceBaseline,
                                                          const float3 radianceEnd,
                                                          const float maxRadiance) {
    RadianceCacheTarget result;
    result.target = float3(0.0f);
    result.status = kRadianceCacheTargetRejectedNonFinite;
    result.targetLuma = 0.0f;
    result.clampedLuma = 0.0f;
    if (!all(isfinite(throughputAtQuery)) || !all(isfinite(radianceBaseline)) || !all(isfinite(radianceEnd))) {
        return result;
    }
    if (!all(throughputAtQuery > float3(0.0f))) {
        result.status = kRadianceCacheTargetRejectedZeroThroughput;
        return result;
    }
    const float3 target = max(radianceEnd - radianceBaseline, float3(0.0f)) / throughputAtQuery;
    if (!all(isfinite(target))) {
        return result;
    }
    bool clamped = false;
    result.targetLuma = luminance_rgb(target);
    result.target = radiance_cache_clamp_target(target, maxRadiance, clamped, result.clampedLuma);
    result.status = clamped ? kRadianceCacheTargetClamped : kRadianceCacheTargetAccepted;
    return result;
}

inline bool radiance_cache_is_training_path(constant PathtraceUniforms& uniforms, const uint pixelIndex) {
    return PT_DEFAULT_OFF(uniforms.radianceCacheMode) == kRadianceCacheModePrototype &&
           PT_DEFAULT_OFF(uniforms.debugViewMode) == kDebugViewNone &&
           (uniforms.frameIndex % max(uniforms.radianceCacheTrainingInterval, 1u)) == 0u &&
           ((pixelIndex + uniforms.frameIndex) % kRadianceCacheTrainingPathStride) == 0u;
}

// Record slot for a vertex at `depth`, or kRadianceCacheTrainRecordCount if the depth is not trained.
inline uint radiance_cache_train_slot(constant PathtraceUniforms& uniforms, const uint depth) {
    const uint minDepth = uniforms.radianceCacheMinDepth;
    if (depth < minDepth || depth - minDepth >= kRadianceCacheTrainRecordCount) {
        return kRadianceCacheTrainRecordCount;
    }
    return depth - minDepth;
}

inline uint radiance_cache_pack_normal(const float3 normal) {
    float3 n = normal / max(fabs(normal.x) + fabs(normal.y) + fabs(normal.z), 1.0e-20f);
    float2 e = n.xy;
    if (n.z < 0.0f) {
        e = (1.0f - fabs(n.yx)) * select(float2(-1.0f), float2(1.0f), n.xy >= float2(0.0f));
    }
    return pack_float_to_unorm2x16(e * 0.5f + 0.5f);
}

inline float3 radiance_cache_unpack_normal(const uint packed) {
    const float2 e = unpack_unorm2x16_to_float(packed) * 2.0f - 1.0f;
    float3 n = float3(e.x, e.y, 1.0f - fabs(e.x) - fabs(e.y));
    const float t = saturate(-n.z);
    n.xy += select(float2(t), float2(-t), n.xy >= float2(0.0f));
    return safe_normalize(n);
}

inline RadianceCacheTrainRecord radiance_cache_pack_train_record(const RadianceQuery query,
                                                                 const float3 throughputAtQuery,
                                                                 const float3 radianceBaseline) {
    const uint roughness8 = static_cast<uint>(round(clamp(query.positionRoughness.w, 0.0f, 1.0f) * 255.0f));
    const uint meta = min(query.materialLobeMeta.x, 0xFFFu) |
                      (min(query.materialLobeMeta.y, 0xFu) << 12u) |
                      (roughness8 << 16u) |
                      (1u << 31u);
    RadianceCacheTrainRecord record;
    record.positionKey = float4(query.positionRoughness.xyz, as_type<float>(query.materialLobeMeta.w));
    record.throughputNormal = float4(throughputAtQuery, as_type<float>(radiance_cache_pack_normal(query.normalDepth.xyz)));
    record.baselineMeta = float4(radianceBaseline, as_type<float>(meta));
    return record;
}

inline bool radiance_cache_train_record_valid(const RadianceCacheTrainRecord record) {
    return (as_type<uint>(record.baselineMeta.w) & (1u << 31u)) != 0u;
}

inline RadianceQuery radiance_cache_unpack_train_query(const RadianceCacheTrainRecord record, const uint depth) {
    const uint meta = as_type<uint>(record.baselineMeta.w);
    RadianceQuery query;
    query.positionRoughness = float4(record.positionKey.xyz, float((meta >> 16u) & 0xFFu) / 255.0f);
    query.normalDepth = float4(radiance_cache_unpack_normal(as_type<uint>(record.throughputNormal.w)), float(depth));
    query.outgoingBiasRisk = float4(0.0f);
    query.materialLobeMeta = uint4(meta & 0xFFFu, (meta >> 12u) & 0xFu, 0u, as_type<uint>(record.positionKey.w));
    return query;
}

// Builds the record for a vertex, or returns false if the vertex is not cacheable.
inline bool radiance_cache_make_train_record(constant PathtraceUniforms& uniforms,
                                             const MaterialData material,
                                             const float3 position,
                                             const float3 normal,
                                             const bool specularOnly,
                                             const uint depth,
                                             const BsdfSampleResult sample,
                                             const float3 radianceBaseline,
                                             const float3 throughputAtQuery,
                                             thread RadianceCacheTrainRecord& record) {
    const RadianceQuery query = radiance_cache_make_query(uniforms, material, position, normal, depth, sample);
    uint fallbackReason = 0u;
    if (!radiance_cache_can_query(query, material, specularOnly, sample, depth, uniforms, fallbackReason)) {
        return false;
    }
    record = radiance_cache_pack_train_record(query, throughputAtQuery, radianceBaseline);
    return true;
}

// Trains the cell of one record with the path's final radiance.
inline void radiance_cache_train_from_record(device RadianceCacheState* radianceCacheStates,
                                             constant PathtraceUniforms& uniforms,
                                             const RadianceCacheTrainRecord record,
                                             const uint depth,
                                             const float3 radianceEnd,
                                             device PathtraceStats* stats) {
    radiance_cache_stat_add(stats, kRadianceCacheStatTrainRecords, 1u);
    const RadianceCacheTarget result = radiance_cache_training_target(record.throughputNormal.xyz,
                                                                      record.baselineMeta.xyz,
                                                                      radianceEnd,
                                                                      radiance_cache_max_radiance(uniforms));
    if (result.status == kRadianceCacheTargetRejectedNonFinite) {
        radiance_cache_stat_add(stats, kRadianceCacheStatTrainRejectedNonFinite, 1u);
        return;
    }
    if (result.status == kRadianceCacheTargetRejectedZeroThroughput) {
        radiance_cache_stat_add(stats, kRadianceCacheStatTrainRejectedZeroThroughput, 1u);
        return;
    }
    radiance_cache_stat_add(stats, kRadianceCacheStatTargetLumaFixed, radiance_cache_stat_luma_fixed(result.targetLuma));
    if (result.status == kRadianceCacheTargetClamped) {
        radiance_cache_stat_add(stats, kRadianceCacheStatTrainClamped, 1u);
        radiance_cache_stat_add(stats,
                                kRadianceCacheStatClampedLumaFixed,
                                radiance_cache_stat_luma_fixed(result.clampedLuma));
    }
    radiance_cache_store_sample(radianceCacheStates,
                                uniforms,
                                radiance_cache_unpack_train_query(record, depth),
                                result.target,
                                stats);
}

// Per-path state for the megakernel and Embree-style integrators (records in thread memory).
struct RadianceCachePathTraining {
    RadianceCacheTrainRecord records[kRadianceCacheTrainRecordCount];
    uint validMask;
    bool active;
};

inline void radiance_cache_training_begin(thread RadianceCachePathTraining& training,
                                          constant PathtraceUniforms& uniforms,
                                          device RadianceCacheState* radianceCacheStates,
                                          const uint pixelIndex) {
    training.validMask = 0u;
    training.active = radianceCacheStates != nullptr && radiance_cache_is_training_path(uniforms, pixelIndex);
}

inline void radiance_cache_training_note_vertex(thread RadianceCachePathTraining& training,
                                                constant PathtraceUniforms& uniforms,
                                                const MaterialData material,
                                                const float3 position,
                                                const float3 normal,
                                                const bool specularOnly,
                                                const uint depth,
                                                const BsdfSampleResult sample,
                                                const float3 radianceBaseline,
                                                const float3 throughputAtQuery) {
    if (!training.active) {
        return;
    }
    const uint slot = radiance_cache_train_slot(uniforms, depth);
    if (slot >= kRadianceCacheTrainRecordCount) {
        return;
    }
    RadianceCacheTrainRecord record;
    if (radiance_cache_make_train_record(uniforms, material, position, normal, specularOnly, depth, sample,
                                         radianceBaseline, throughputAtQuery, record)) {
        training.records[slot] = record;
        training.validMask |= 1u << slot;
    }
}

inline void radiance_cache_training_finish(thread RadianceCachePathTraining& training,
                                           device RadianceCacheState* radianceCacheStates,
                                           constant PathtraceUniforms& uniforms,
                                           const float3 radianceEnd,
                                           device PathtraceStats* stats) {
    if (!training.active || training.validMask == 0u) {
        return;
    }
    for (uint slot = 0u; slot < kRadianceCacheTrainRecordCount; ++slot) {
        if ((training.validMask & (1u << slot)) != 0u) {
            radiance_cache_train_from_record(radianceCacheStates,
                                             uniforms,
                                             training.records[slot],
                                             uniforms.radianceCacheMinDepth + slot,
                                             radianceEnd,
                                             stats);
        }
    }
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
    const float3 throughput,
    thread float& outConfidence,
    thread float& outVariance,
    thread float& outBiasRisk) {
    outConfidence = 0.0f;
    outVariance = 0.0f;
    outBiasRisk = 0.0f;
    RadianceQuery query = radiance_cache_make_query(uniforms, material, position, normal, depth, sample);
    uint fallbackReason = 0u;
    if (!radiance_cache_can_query(query, material, specularOnly, sample, depth, uniforms, fallbackReason)) {
        if (stats && PT_DEFAULT_OFF(uniforms.radianceCacheMode) == kRadianceCacheModePrototype) {
            atomic_fetch_add_explicit(&stats->radianceCacheFallbackCount, 1u, memory_order_relaxed);
            radiance_cache_stat_add(stats, kRadianceCacheStatFallbackReasonBase + fallbackReason, 1u);
        }
        return false;
    }
    if (stats) {
        atomic_fetch_add_explicit(&stats->radianceCacheQueryCount, 1u, memory_order_relaxed);
        radiance_cache_stat_add(stats, kRadianceCacheStatDepthQueryBase + radiance_cache_stat_depth_bin(depth), 1u);
    }

    RadianceEstimate estimate;
    const bool hit = radiance_cache_load(radianceCacheStates, uniforms, query, estimate);
    if (!hit) {
        if (stats) {
            atomic_fetch_add_explicit(&stats->radianceCacheMissCount, 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&stats->radianceCacheFallbackCount, 1u, memory_order_relaxed);
            radiance_cache_stat_add(stats,
                                    kRadianceCacheStatFallbackReasonBase + min(estimate.sampleAgeFallback.w, 9u),
                                    1u);
        }
        return false;
    }

    const float3 contribution = throughput * estimate.radianceConfidence.xyz;
    if (!all(isfinite(contribution))) {
        if (stats) {
            atomic_fetch_add_explicit(&stats->radianceCacheInvalidCount, 1u, memory_order_relaxed);
            radiance_cache_stat_add(stats, kRadianceCacheStatFallbackReasonBase + kRadianceCacheFallbackInvalid, 1u);
        }
        return false;
    }
    radiance += contribution;
    outConfidence = estimate.radianceConfidence.w;
    outVariance = estimate.varianceBiasRisk.x;
    outBiasRisk = estimate.varianceBiasRisk.y;
    if (stats) {
        atomic_fetch_add_explicit(&stats->radianceCacheHitCount, 1u, memory_order_relaxed);
        const uint depthBin = radiance_cache_stat_depth_bin(depth);
        radiance_cache_stat_add(stats, kRadianceCacheStatDepthHitBase + depthBin, 1u);
        radiance_cache_stat_add(stats,
                                kRadianceCacheStatDepthHitLumaFixedBase + depthBin,
                                radiance_cache_stat_luma_fixed(luminance_rgb(estimate.radianceConfidence.xyz)));
    }
    return true;
}
