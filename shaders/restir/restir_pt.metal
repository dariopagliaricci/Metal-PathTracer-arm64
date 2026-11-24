struct RestirPtPathReservoir {
    float3 position;
    float3 normal;
    float3 direction;
    float3 throughput;
    float pdf;
    uint depth;
    uint materialType;
    uint lobeType;
    uint flags;
};

inline bool restir_pt_enhanced_mode(const uint mode) {
    return mode == kRestirPtModeEnhancedUnbiased || mode == kRestirPtModeEnhancedBiased;
}

inline bool restir_pt_any_capture_mode(const uint mode) {
    return mode == kRestirPtModeProductionReservoirCapture ||
           mode == kRestirPtModeProductionPathReuse ||
           restir_pt_enhanced_mode(mode);
}

inline bool restir_pt_any_reuse_mode(const uint mode) {
    return mode == kRestirPtModeProductionPathReuse || restir_pt_enhanced_mode(mode);
}

inline uint restir_pt_enhanced_feature_flags(const uint mode) {
    if (!restir_pt_enhanced_mode(mode)) {
        return 0u;
    }
    uint flags = kRestirPtEnhancedFeaturePairedSpatialReuse |
                 kRestirPtEnhancedFeatureFootprintReconnection |
                 kRestirPtEnhancedFeatureUnifiedDiGi |
                 kRestirPtEnhancedFeatureVectorShadingWeights |
                 kRestirPtEnhancedFeatureDualMotionVectors |
                 kRestirPtEnhancedFeatureReplayCompaction;
    if (mode == kRestirPtModeEnhancedBiased) {
        flags |= kRestirPtEnhancedFeatureDuplicationDecorrelation;
    }
    return flags;
}

inline uint restir_pt_replay_seed(constant PathtraceUniforms& uniforms,
                                  const uint2 pixelCoord,
                                  const uint depth) {
    uint h = uniforms.fixedRngSeed != 0u ? uniforms.fixedRngSeed : 2166136261u;
    h ^= pixelCoord.x + 0x9e3779b9u + (h << 6u) + (h >> 2u);
    h ^= pixelCoord.y + 0x85ebca6bu + (h << 6u) + (h >> 2u);
    h ^= depth + 0xc2b2ae35u + (h << 6u) + (h >> 2u);
    h ^= uniforms.frameIndex + 0x27d4eb2du + (h << 6u) + (h >> 2u);
    return h;
}

inline void record_restir_pt_enhanced_histograms(device PathtraceStats* stats,
                                                 const uint pathLength,
                                                 const uint reconnectionDepth) {
    const uint pathBin = min(pathLength, kRestirPathReservoirDepthCount - 1u);
    const uint reconnectBin = min(reconnectionDepth, kRestirPathReservoirDepthCount - 1u);
    atomic_fetch_add_explicit(&stats->restirPtEnhancedPathLengthHistogram[pathBin],
                              1u,
                              memory_order_relaxed);
    atomic_fetch_add_explicit(&stats->restirPtEnhancedReconnectionDepthHistogram[reconnectBin],
                              1u,
                              memory_order_relaxed);
}

inline bool restir_pt_production_capture_valid(const BsdfSampleResult sample,
                                             const MaterialData material,
                                             const float3 position,
                                             const float3 normal,
                                             const float3 throughput,
                                             const uint depth,
                                             const bool specularOnly,
                                             const bool allowPrimaryDepth = false) {
    return (depth >= 1u || allowPrimaryDepth) &&
           !specularOnly &&
           restir_diffuse_material_eligible(material) &&
           sample.pdf > 0.0f &&
           sample.directionalPdf > 0.0f &&
           !sample.isDelta &&
           !sample.isBssrdf &&
           !sample.hasExitPoint &&
           sample.mediumEvent == 0 &&
           sample.lobeType == 0u &&
           all(isfinite(position)) &&
           all(isfinite(normal)) &&
           dot(normal, normal) > 0.0f &&
           all(isfinite(sample.direction)) &&
           dot(sample.direction, sample.direction) > 0.0f &&
           all(isfinite(sample.weight)) &&
           all(sample.weight >= float3(0.0f)) &&
           all(isfinite(throughput)) &&
           all(throughput >= float3(0.0f));
}

inline bool load_restir_pt_reservoir_direction(device const RestirPtReservoirState* restirPtReservoirs,
                                               constant PathtraceUniforms& uniforms,
                                               const uint2 pixelCoord,
                                               const MaterialData material,
                                               const float3 position,
                                               const float3 normal,
                                               const uint depth,
                                               thread float3& direction,
                                               thread float& confidence) {
    if (!restirPtReservoirs) {
        return false;
    }
    const RestirPtReservoirState reservoir =
        restirPtReservoirs[restir_path_state_index(uniforms, pixelCoord, depth)];
    if (!restir_state_valid(reservoir.flags)) {
        return false;
    }

    if (reservoir.lobeType != 0u) {
        return false;
    }

    const float3 reservoirNormal = safe_normalize(reservoir.normalDepth.xyz);
    const float normalAgreement = dot(safe_normalize(normal), reservoirNormal);
    if (!(normalAgreement > 0.0f)) {
        return false;
    }

    const float3 candidate = safe_normalize(reservoir.directionWeight.xyz);
    if (!all(isfinite(candidate)) || dot(candidate, candidate) <= 0.0f) {
        return false;
    }
    const float cosTheta = dot(safe_normalize(normal), candidate);
    if (!(cosTheta > 0.0f)) {
        return false;
    }

    const float sampleLuma = max(reservoir.directionWeight.w, 0.0f);
    if (!(sampleLuma > 0.0f) || !isfinite(sampleLuma)) {
        return false;
    }

    direction = candidate;
    const float depthDelta = fabs(reservoir.normalDepth.w - float(depth));
    const float compatibility = clamp(normalAgreement, 0.0f, 1.0f) *
                                (1.0f / (1.0f + 0.25f * depthDelta));
    confidence = clamp(compatibility *
                           (sampleLuma / max(sampleLuma + reservoir.throughputLuma.w, 1.0e-4f)),
                       0.0625f,
                       1.0f);
    return true;
}

inline bool load_restir_pt_enhanced_candidate(device const RestirPtReservoirState* restirPtReservoirs,
                                              constant PathtraceUniforms& uniforms,
                                              const uint2 pixelCoord,
                                              const MaterialData material,
                                              const float3 position,
                                              const float3 normal,
                                              const uint depth,
                                              thread RestirPtReservoirState& reservoir,
                                              thread float& confidence) {
    if (!restirPtReservoirs || pixelCoord.x >= uniforms.width || pixelCoord.y >= uniforms.height) {
        return false;
    }
    reservoir = restirPtReservoirs[restir_path_state_index(uniforms, pixelCoord, depth)];
    if (!restir_state_valid(reservoir.flags) || reservoir.lobeType != 0u) {
        return false;
    }

    const float3 reservoirNormal = safe_normalize(reservoir.normalDepth.xyz);
    const float3 currentNormal = safe_normalize(normal);
    const float normalAgreement = dot(currentNormal, reservoirNormal);
    if (!(normalAgreement > 0.15f) || !isfinite(normalAgreement)) {
        return false;
    }

    const float sourcePdf = max(reservoir.reconnectionVertexSourcePdf.w, reservoir.positionPdf.w);
    const float target = max(reservoir.targetJacobianMis.x, reservoir.directionWeight.w);
    if (!(sourcePdf > 0.0f) || !(target > 0.0f) || !isfinite(sourcePdf) || !isfinite(target)) {
        return false;
    }

    const float footprint =
        max(length(position - reservoir.positionPdf.xyz), 0.0f) *
        max(sourcePdf, 1.0e-6f);
    const float roughnessGuard = clamp(material.pbrParams.y + 0.08f, 0.08f, 1.0f);
    if (!(footprint <= 0.02f + roughnessGuard * 4.0f) || !isfinite(footprint)) {
        return false;
    }

    const float age = float((uniforms.frameIndex >= reservoir.frameTag)
                                ? (uniforms.frameIndex - reservoir.frameTag)
                                : 0u);
    confidence = clamp(normalAgreement *
                           (target / max(target + reservoir.throughputLuma.w, 1.0e-4f)) *
                           (1.0f / (1.0f + age * 0.10f)),
                       0.0f,
                       1.0f);
    return confidence > 0.0f;
}

inline int2 restir_pt_reciprocal_pair_offset(const uint2 pixelCoord,
                                             const uint depth,
                                             const uint frameIndex,
                                             const uint pairIndex) {
    const uint selector = (depth * 3u + frameIndex + pairIndex) % 3u;
    if (selector == 0u) {
        return ((pixelCoord.x & 1u) == 0u) ? int2(1, 0) : int2(-1, 0);
    }
    if (selector == 1u) {
        return ((pixelCoord.y & 1u) == 0u) ? int2(0, 1) : int2(0, -1);
    }
    const bool even = ((pixelCoord.x ^ pixelCoord.y) & 1u) == 0u;
    return even ? int2(1, 1) : int2(-1, -1);
}

inline bool restir_pt_try_enhanced_candidate_direction(device const RestirPtReservoirState* restirPtReservoirs,
                                                       constant PathtraceUniforms& uniforms,
                                                       const uint2 candidatePixel,
                                                       const MaterialData material,
                                                       const float3 position,
                                                       const float3 normal,
                                                       const uint depth,
                                                       thread float3& direction,
                                                       thread float& confidence) {
    RestirPtReservoirState reservoir{};
    if (!load_restir_pt_enhanced_candidate(restirPtReservoirs,
                                           uniforms,
                                           candidatePixel,
                                           material,
                                           position,
                                           normal,
                                           depth,
                                           reservoir,
                                           confidence)) {
        return false;
    }
    const float3 candidate = safe_normalize(reservoir.directionWeight.xyz);
    if (!all(isfinite(candidate)) ||
        dot(candidate, candidate) <= 0.0f ||
        dot(safe_normalize(normal), candidate) <= 0.0f) {
        return false;
    }
    direction = candidate;
    return true;
}

inline void update_restir_pt_reservoir(device RestirPtReservoirState* restirPtReservoirs,
                                       constant PathtraceUniforms& uniforms,
                                       const uint2 pixelCoord,
                                       const MaterialData material,
                                       const float3 position,
                                       const float3 normal,
                                       const float3 throughput,
                                       const uint depth,
                                       const BsdfSampleResult sample) {
    if (!restirPtReservoirs || sample.pdf <= 0.0f || !all(isfinite(sample.direction))) {
        return;
    }
    const float sampleLuma = max(luminance_rgb(max(sample.weight, float3(0.0f))), 0.0f);
    const float throughputLuma = max(luminance_rgb(max(throughput, float3(0.0f))), 0.0f);
    if (!(sampleLuma > 0.0f) ||
        !isfinite(sampleLuma) ||
        !all(isfinite(position)) ||
        !all(isfinite(normal)) ||
        !all(isfinite(throughput))) {
        return;
    }

    RestirPtReservoirState reservoir{};
    reservoir.positionPdf = float4(position, sample.pdf);
    reservoir.normalDepth = float4(safe_normalize(normal), float(depth));
    reservoir.directionWeight = float4(safe_normalize(sample.direction), sampleLuma);
    reservoir.throughputLuma = float4(max(throughput, float3(0.0f)), throughputLuma);
    reservoir.reconnectionVertexSourcePdf = float4(position, max(sample.directionalPdf, sample.pdf));
    reservoir.targetJacobianMis = float4(sampleLuma, 1.0f, 1.0f, 1.0f);
    reservoir.vectorContributionScalar = float4(max(throughput * sample.weight, float3(0.0f)),
                                                sampleLuma);
    const MaterialClosureMetadata closure = material_closure_metadata_for_sample(material, sample);
    reservoir.materialType = closure.type;
    reservoir.lobeType = closure.lobeIndex;
    reservoir.frameTag = uniforms.frameIndex;
    reservoir.flags = 1u;
    reservoir.replaySeed = restir_pt_replay_seed(uniforms, pixelCoord, depth);
    reservoir.pathLength = min(depth + 1u, kRestirPathReservoirDepthCount - 1u);
    reservoir.techniqueIndex = 1u;
    reservoir.validityFlags = 1u;
    reservoir.materialLobeMetadata = (closure.type & 0xffffu) | ((closure.lobeIndex & 0xffffu) << 16u);
    reservoir.duplicateScoreQ = 0u;
    reservoir.cCapQ = 20u * 1024u;
    reservoir.enhancedModeFlags = restir_pt_enhanced_feature_flags(PT_DEFAULT_OFF(uniforms.restirPtMode));
    restirPtReservoirs[restir_path_state_index(uniforms, pixelCoord, depth)] = reservoir;
}

inline bool capture_restir_pt_production_reservoir(constant PathtraceUniforms& uniforms,
                                                const MaterialData material,
                                                const float3 position,
                                                const float3 normal,
                                                const float3 throughput,
                                                const bool specularOnly,
                                                const uint depth,
                                                const uint2 pixelCoord,
                                                device RestirPtReservoirState* restirPtReservoirs,
                                                device PathtraceStats* stats,
                                                const BsdfSampleResult sample) {
    if (!restir_pt_any_capture_mode(PT_DEFAULT_OFF(uniforms.restirPtMode)) || !stats) {
        return false;
    }

    const bool diffuseEligible =
        (depth >= 1u || restir_pt_enhanced_mode(PT_DEFAULT_OFF(uniforms.restirPtMode))) &&
        !specularOnly &&
        restir_diffuse_material_eligible(material) &&
        !sample.isDelta &&
        !sample.isBssrdf &&
        !sample.hasExitPoint &&
        sample.mediumEvent == 0 &&
        sample.lobeType == 0u;
    if (!diffuseEligible) {
        atomic_fetch_add_explicit(&stats->restirPtRejectedUnsupportedCount, 1u, memory_order_relaxed);
        return false;
    }

    if (!restir_pt_production_capture_valid(sample,
                                            material,
                                            position,
                                            normal,
                                            throughput,
                                            depth,
                                            specularOnly,
                                            restir_pt_enhanced_mode(PT_DEFAULT_OFF(uniforms.restirPtMode)))) {
        atomic_fetch_add_explicit(&stats->restirPtRejectedInvalidCount, 1u, memory_order_relaxed);
        return false;
    }

    RestirPtPathReservoir reservoir{};
    reservoir.position = position;
    reservoir.normal = safe_normalize(normal);
    reservoir.direction = safe_normalize(sample.direction);
    reservoir.throughput = throughput;
    reservoir.pdf = sample.directionalPdf;
    reservoir.depth = depth;
    const MaterialClosureMetadata closure = material_closure_metadata_for_sample(material, sample);
    reservoir.materialType = closure.type;
    reservoir.lobeType = closure.lobeIndex;
    reservoir.flags = 0u;
    (void)reservoir;
    update_restir_pt_reservoir(restirPtReservoirs,
                               uniforms,
                               pixelCoord,
                               material,
                               position,
                               normal,
                               throughput,
                               depth,
                               sample);

    const uint previousCandidate =
        atomic_fetch_add_explicit(&stats->restirPtCandidateCount, 1u, memory_order_relaxed);
    if (previousCandidate >= max(uniforms.restirPtMaxReservoirs, 1u)) {
        atomic_fetch_add_explicit(&stats->restirPtRejectedUnsupportedCount, 1u, memory_order_relaxed);
        return false;
    }

    atomic_fetch_add_explicit(&stats->restirPtReservoirUpdateCount, 1u, memory_order_relaxed);
    if (restir_pt_enhanced_mode(PT_DEFAULT_OFF(uniforms.restirPtMode))) {
        atomic_fetch_add_explicit(&stats->restirPtEnhancedCandidateCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->restirPtEnhancedInitialSelectedCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->restirPtEnhancedCanonicalSampleCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->restirPtEnhancedVectorWeightUseCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->restirPtEnhancedReplayCompactionInputCount,
                                  1u,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->restirPtEnhancedReplayCompactionOutputCount,
                                  1u,
                                  memory_order_relaxed);
        if (PT_DEFAULT_OFF(uniforms.restirPtMode) == kRestirPtModeEnhancedBiased) {
            atomic_fetch_add_explicit(&stats->restirPtEnhancedBiasedCCapApplicationCount,
                                      1u,
                                      memory_order_relaxed);
        }
        record_restir_pt_enhanced_histograms(stats, min(depth + 1u, kRestirPathReservoirDepthCount - 1u), depth);
    }
    if (PT_DEFAULT_OFF(uniforms.restirPtDebug) != 0u) {
        atomic_fetch_add_explicit(&stats->restirPtDebugRecordCount, 1u, memory_order_relaxed);
    }
    return true;
}

inline bool apply_restir_pt_production_path_reuse(constant PathtraceUniforms& uniforms,
                                                    const MaterialData material,
                                                    const float3 position,
                                                    const float3 normal,
                                                    const float3 wo,
                                                    const float3 throughput,
                                                    const FireflyClampParams clampParams,
                                                    const float diffuseOcclusion,
                                                    const bool specularOnly,
                                                    const uint depth,
                                                    const uint2 pixelCoord,
                                                    device RestirPtReservoirState* restirPtReservoirs,
                                                    device PathtraceStats* stats,
                                                    thread BsdfSampleResult& sample) {
    if (!restir_pt_any_reuse_mode(PT_DEFAULT_OFF(uniforms.restirPtMode)) || !stats) {
        return false;
    }
    const bool diffuseEligible =
        (depth >= 1u || restir_pt_enhanced_mode(PT_DEFAULT_OFF(uniforms.restirPtMode))) &&
        !specularOnly &&
        restir_diffuse_material_eligible(material) &&
        !sample.isDelta &&
        !sample.isBssrdf &&
        !sample.hasExitPoint &&
        sample.mediumEvent == 0 &&
        sample.lobeType == 0u;
    if (!diffuseEligible) {
        atomic_fetch_add_explicit(&stats->restirPtReuseFallbackCount, 1u, memory_order_relaxed);
        if (restir_pt_enhanced_mode(PT_DEFAULT_OFF(uniforms.restirPtMode))) {
            atomic_fetch_add_explicit(&stats->restirPtEnhancedFallbackCount, 1u, memory_order_relaxed);
            atomic_fetch_add_explicit(&stats->restirPtEnhancedInvalidShiftUnsupportedCount,
                                      1u,
                                      memory_order_relaxed);
        }
        return false;
    }
    if (!restir_pt_production_capture_valid(sample,
                                            material,
                                            position,
                                            normal,
                                            throughput,
                                            depth,
                                            specularOnly,
                                            restir_pt_enhanced_mode(PT_DEFAULT_OFF(uniforms.restirPtMode)))) {
        atomic_fetch_add_explicit(&stats->restirPtReuseRejectedInvalidCount, 1u, memory_order_relaxed);
        if (restir_pt_enhanced_mode(PT_DEFAULT_OFF(uniforms.restirPtMode))) {
            atomic_fetch_add_explicit(&stats->restirPtEnhancedTemporalRejectedCount,
                                      1u,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&stats->restirPtEnhancedInvalidShiftPdfCount,
                                      1u,
                                      memory_order_relaxed);
        }
        return false;
    }
    atomic_fetch_add_explicit(&stats->restirPtReuseCandidateCount, 1u, memory_order_relaxed);

    const float strength = clamp(uniforms.restirPtReuseStrength, 0.0f, 1.0f);
    if (strength <= 0.0f) {
        atomic_fetch_add_explicit(&stats->restirPtReuseFallbackCount, 1u, memory_order_relaxed);
        if (restir_pt_enhanced_mode(PT_DEFAULT_OFF(uniforms.restirPtMode))) {
            atomic_fetch_add_explicit(&stats->restirPtEnhancedFallbackCount, 1u, memory_order_relaxed);
        }
        return false;
    }

    const float3 axis = safe_normalize(normal);
    const float3 dir = safe_normalize(sample.direction);
    float3 reservoirDirection = float3(0.0f);
    float reservoirConfidence = 0.0f;
    bool temporalAccepted = false;
    bool spatialAccepted = false;
    bool dualTemporalAccepted = false;

    if (restir_pt_enhanced_mode(PT_DEFAULT_OFF(uniforms.restirPtMode))) {
        atomic_fetch_add_explicit(&stats->restirPtEnhancedReplayCompactionInputCount,
                                  1u,
                                  memory_order_relaxed);
        temporalAccepted = restir_pt_try_enhanced_candidate_direction(restirPtReservoirs,
                                                                      uniforms,
                                                                      pixelCoord,
                                                                      material,
                                                                      position,
                                                                      axis,
                                                                      depth,
                                                                      reservoirDirection,
                                                                      reservoirConfidence);

        if (!temporalAccepted && (restir_pt_enhanced_feature_flags(PT_DEFAULT_OFF(uniforms.restirPtMode)) &
                                  kRestirPtEnhancedFeatureDualMotionVectors) != 0u) {
            const int2 pixel = int2(int(pixelCoord.x), int(pixelCoord.y));
            const int2 center = int2(int(uniforms.width >> 1u), int(uniforms.height >> 1u));
            const int2 offset = int2((center.x > pixel.x) ? 1 : ((center.x < pixel.x) ? -1 : 0),
                                     (center.y > pixel.y) ? 1 : ((center.y < pixel.y) ? -1 : 0));
            const int2 dualPixel = pixel + offset;
            if (dualPixel.x >= 0 && dualPixel.y >= 0 &&
                dualPixel.x < int(uniforms.width) && dualPixel.y < int(uniforms.height)) {
                dualTemporalAccepted = restir_pt_try_enhanced_candidate_direction(
                    restirPtReservoirs,
                    uniforms,
                    uint2(uint(dualPixel.x), uint(dualPixel.y)),
                    material,
                    position,
                    axis,
                    depth,
                    reservoirDirection,
                    reservoirConfidence);
            }
        }

        if (temporalAccepted || dualTemporalAccepted) {
            atomic_fetch_add_explicit(&stats->restirPtEnhancedTemporalAcceptedCount,
                                      1u,
                                      memory_order_relaxed);
        } else {
            atomic_fetch_add_explicit(&stats->restirPtEnhancedTemporalRejectedCount,
                                      1u,
                                      memory_order_relaxed);
        }

        if ((restir_pt_enhanced_feature_flags(PT_DEFAULT_OFF(uniforms.restirPtMode)) &
             kRestirPtEnhancedFeaturePairedSpatialReuse) != 0u) {
            float3 spatialDirection = float3(0.0f);
            float spatialConfidence = 0.0f;
            const uint pairCount = max(uniforms.spatialReuseNeighborCount, 1u);
            for (uint pairIndex = 0u; pairIndex < pairCount; ++pairIndex) {
                const int2 pairOffset =
                    restir_pt_reciprocal_pair_offset(pixelCoord, depth, uniforms.frameIndex, pairIndex);
                const int2 neighborPixel = int2(int(pixelCoord.x), int(pixelCoord.y)) + pairOffset;
                if (neighborPixel.x < 0 || neighborPixel.y < 0 ||
                    neighborPixel.x >= int(uniforms.width) || neighborPixel.y >= int(uniforms.height)) {
                    continue;
                }
                if (restir_pt_try_enhanced_candidate_direction(restirPtReservoirs,
                                                               uniforms,
                                                               uint2(uint(neighborPixel.x),
                                                                     uint(neighborPixel.y)),
                                                               material,
                                                               position,
                                                               axis,
                                                               depth,
                                                               spatialDirection,
                                                               spatialConfidence)) {
                    spatialAccepted = true;
                    break;
                }
            }
            if (spatialAccepted) {
                if (temporalAccepted || dualTemporalAccepted) {
                    const float spatialWeight = clamp(spatialConfidence * 0.5f, 0.0f, 0.5f);
                    reservoirDirection =
                        safe_normalize(mix(reservoirDirection, spatialDirection, spatialWeight));
                    reservoirConfidence = max(reservoirConfidence, spatialConfidence);
                } else {
                    reservoirDirection = spatialDirection;
                    reservoirConfidence = spatialConfidence;
                }
                atomic_fetch_add_explicit(&stats->restirPtEnhancedSpatialAcceptedCount,
                                          1u,
                                          memory_order_relaxed);
            } else {
                atomic_fetch_add_explicit(&stats->restirPtEnhancedSpatialRejectedCount,
                                          1u,
                                          memory_order_relaxed);
            }
        }
    } else {
        temporalAccepted = load_restir_pt_reservoir_direction(restirPtReservoirs,
                                                              uniforms,
                                                              pixelCoord,
                                                              material,
                                                              position,
                                                              axis,
                                                              depth,
                                                              reservoirDirection,
                                                              reservoirConfidence);
    }

    if (!temporalAccepted && !dualTemporalAccepted && !spatialAccepted) {
        atomic_fetch_add_explicit(&stats->restirPtReuseFallbackCount, 1u, memory_order_relaxed);
        if (restir_pt_enhanced_mode(PT_DEFAULT_OFF(uniforms.restirPtMode))) {
            atomic_fetch_add_explicit(&stats->restirPtEnhancedFallbackCount, 1u, memory_order_relaxed);
        }
        return false;
    }
    if (restir_pt_enhanced_mode(PT_DEFAULT_OFF(uniforms.restirPtMode))) {
        atomic_fetch_add_explicit(&stats->restirPtEnhancedReplayCompactionOutputCount,
                                  1u,
                                  memory_order_relaxed);
    }
    float3 reused = safe_normalize(mix(dir, reservoirDirection, min(strength, 1.0f) * reservoirConfidence));

    const float cosTheta = dot(axis, reused);
    if (!(cosTheta > 0.0f) || !all(isfinite(reused))) {
        atomic_fetch_add_explicit(&stats->restirPtReuseRejectedInvalidCount, 1u, memory_order_relaxed);
        if (restir_pt_enhanced_mode(PT_DEFAULT_OFF(uniforms.restirPtMode))) {
            atomic_fetch_add_explicit(&stats->restirPtEnhancedTemporalRejectedCount,
                                      1u,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&stats->restirPtEnhancedInvalidShiftGeometryCount,
                                      1u,
                                      memory_order_relaxed);
        }
        return false;
    }
    if (!restir_reweight_bsdf_sample_for_direction(material,
                                                   position,
                                                   axis,
                                                   wo,
                                                   reused,
                                                   clampParams,
                                                   PT_DEFAULT_OFF(uniforms.sssMode),
                                                   diffuseOcclusion,
                                                   specularOnly,
                                                   sample)) {
        atomic_fetch_add_explicit(&stats->restirPtReuseRejectedInvalidCount, 1u, memory_order_relaxed);
        if (restir_pt_enhanced_mode(PT_DEFAULT_OFF(uniforms.restirPtMode))) {
            atomic_fetch_add_explicit(&stats->restirPtEnhancedTemporalRejectedCount,
                                      1u,
                                      memory_order_relaxed);
            atomic_fetch_add_explicit(&stats->restirPtEnhancedInvalidShiftMaterialCount,
                                      1u,
                                      memory_order_relaxed);
        }
        return false;
    }

    atomic_fetch_add_explicit(&stats->restirPtReuseAppliedCount, 1u, memory_order_relaxed);
    if (restir_pt_enhanced_mode(PT_DEFAULT_OFF(uniforms.restirPtMode))) {
        atomic_fetch_add_explicit(&stats->restirPtEnhancedCandidateCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->restirPtEnhancedCanonicalSampleCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->restirPtEnhancedVectorWeightUseCount, 1u, memory_order_relaxed);
        if (PT_DEFAULT_OFF(uniforms.restirPtMode) == kRestirPtModeEnhancedBiased) {
            atomic_fetch_add_explicit(&stats->restirPtEnhancedBiasedCCapApplicationCount,
                                      1u,
                                      memory_order_relaxed);
        }
        record_restir_pt_enhanced_histograms(stats,
                                             min(depth + 1u, kRestirPathReservoirDepthCount - 1u),
                                             depth);
    }
    return true;
}
