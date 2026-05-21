inline bool restir_prototype_diffuse_material_eligible(const MaterialData material) {
    return material_closure_restir_diffuse_eligible(material);
}

inline uint restir_pixel_state_index(constant PathtraceUniforms& uniforms,
                                     const uint2 pixelCoord) {
    const uint width = max(uniforms.width, 1u);
    return min(pixelCoord.y, max(uniforms.height, 1u) - 1u) * width +
           min(pixelCoord.x, width - 1u);
}

inline uint restir_path_state_index(constant PathtraceUniforms& uniforms,
                                    const uint2 pixelCoord,
                                    const uint depth) {
    return restir_pixel_state_index(uniforms, pixelCoord) * kRestirPathReservoirDepthCount +
           min(depth, kRestirPathReservoirDepthCount - 1u);
}

inline bool restir_state_valid(const uint flags) {
    return (flags & 1u) != 0u;
}

inline uint restir_debug_path_guiding_cell_hash(const float3 position, const float cellSize) {
    float invCell = 1.0f / max(cellSize, 1.0e-4f);
    int3 cell = int3(floor(position * invCell));
    uint hx = as_type<uint>(cell.x) * 73856093u;
    uint hy = as_type<uint>(cell.y) * 19349663u;
    uint hz = as_type<uint>(cell.z) * 83492791u;
    return pcg_hash(hx ^ hy ^ hz ^ 0xC2B2AE35u);
}

inline uint restir_debug_path_guiding_cell_base(constant PathtraceUniforms& uniforms,
                                                const float3 position) {
    const uint binCount = 64u;
    const uint capacity = max(uniforms.width * uniforms.height * kRestirPathReservoirDepthCount, binCount);
    const uint cellCount = max(capacity / binCount, 1u);
    const uint cellID =
        restir_debug_path_guiding_cell_hash(position, max(uniforms.pathGuidingCellSize, 1.0e-4f)) %
        cellCount;
    return cellID * binCount;
}

inline PathGuidingReservoirState restir_debug_path_guiding_best_bin(
    constant PathtraceUniforms& uniforms,
    const HitRecord rec,
    device const PathGuidingReservoirState* pathGuidingStates,
    thread bool& valid) {
    PathGuidingReservoirState best{};
    valid = false;
    if (!pathGuidingStates) {
        return best;
    }
    const uint base = restir_debug_path_guiding_cell_base(uniforms, rec.point);
    float bestWeight = 0.0f;
    for (uint bin = 0u; bin < 64u; ++bin) {
        const PathGuidingReservoirState state = pathGuidingStates[base + bin];
        if (!restir_state_valid(state.flags) || state.sampleCount == 0u) {
            continue;
        }
        const float meanLuma = float(state.reserved0) / (256.0f * float(max(state.sampleCount, 1u)));
        const float weight = max(meanLuma, 0.0f) * sqrt(float(max(state.sampleCount, 1u)));
        if (weight > bestWeight && isfinite(weight)) {
            bestWeight = weight;
            best = state;
            best.reserved0 = bin;
            valid = true;
        }
    }
    return best;
}

inline bool restir_live_debug_view_mode(const uint mode) {
    return mode >= kDebugViewRestirCandidateSourceId &&
           mode <= kDebugViewRadianceCache;
}

inline float3 restir_debug_hash_color(uint value) {
    value = pcg_hash(value);
    return float3(float((value >> 0u) & 0xffu) / 255.0f,
                  float((value >> 8u) & 0xffu) / 255.0f,
                  float((value >> 16u) & 0xffu) / 255.0f);
}

inline bool restir_world_debug_mode_active(const uint mode) {
    return mode == kDirectLightModeRisWorldReuse ||
           mode == kDirectLightModeRisRegirCache ||
           mode == kDirectLightModeRestirDiRegirHybrid;
}

inline float3 restir_debug_view_color(constant PathtraceUniforms& uniforms,
                                      const HitRecord rec,
                                      const MaterialData material,
                                      const uint2 pixelCoord,
                                      const uint depth,
                                      device const PathGuidingReservoirState* pathGuidingStates,
                                      device const RestirPtReservoirState* restirPtReservoirs,
                                      device const RadianceCacheState* radianceCacheStates,
                                      const float3 debugBaseColor,
                                      const float debugMetallic,
                                      const float debugRoughness,
                                      const float debugAO) {
    switch (uniforms.debugViewMode) {
        case kDebugViewBaseColor:
            return debugBaseColor;
        case kDebugViewMetallic:
            return float3(debugMetallic);
        case kDebugViewRoughness:
            return float3(debugRoughness);
        case kDebugViewAO:
            return float3(debugAO);
        default:
            break;
    }

    if (!restir_live_debug_view_mode(uniforms.debugViewMode)) {
        return float3(0.0f);
    }

    const bool eligible = restir_prototype_diffuse_material_eligible(material);
    const uint stateIndex = restir_path_state_index(uniforms, pixelCoord, depth);
    bool pathGuidingValid = false;
    const PathGuidingReservoirState pathGuidingBest =
        restir_debug_path_guiding_best_bin(uniforms, rec, pathGuidingStates, pathGuidingValid);
    const bool pathReservoirValid =
        restirPtReservoirs &&
        restir_state_valid(restirPtReservoirs[stateIndex].flags);
    const float pathGuidingConfidence =
        pathGuidingValid ? clamp(pathGuidingBest.directionConfidence.w, 0.0f, 1.0f) : 0.0f;
    const float ptReservoirConfidence =
        pathReservoirValid ? clamp(restirPtReservoirs[stateIndex].directionWeight.w /
                                       max(restirPtReservoirs[stateIndex].directionWeight.w + 1.0f, 1.0f),
                                   0.0f,
                                   1.0f)
                           : 0.0f;

    switch (uniforms.debugViewMode) {
        case kDebugViewRestirCandidateSourceId: {
            if (uniforms.directLightMode == kDirectLightModeLegacyRect ||
                uniforms.directLightMode == kDirectLightModeBaselineEmissive) {
                return float3(0.04f);
            }
            const uint cell =
                (pixelCoord.x / 8u) + (pixelCoord.y / 8u) * 131u +
                uniforms.directLightMode * 8191u + rec.materialIndex * 17u;
            return restir_debug_hash_color(cell);
        }
        case kDebugViewRestirReservoirConfidence: {
            const float confidence = max(pathGuidingConfidence, ptReservoirConfidence);
            return float3(confidence, confidence * 0.85f, confidence * 0.2f);
        }
        case kDebugViewRestirRegirCellId: {
            if (!restir_world_debug_mode_active(uniforms.directLightMode)) {
                return float3(0.0f);
            }
            const float cellSize = max(uniforms.worldReuseCellSize, 1.0e-4f);
            const int3 cell = int3(floor(rec.point / cellSize));
            const uint hash = uint(cell.x) * 73856093u ^
                              uint(cell.y) * 19349663u ^
                              uint(cell.z) * 83492791u ^
                              uniforms.directLightMode * 2654435761u;
            return restir_debug_hash_color(hash);
        }
        case kDebugViewRestirPathGuidingMask: {
            if (uniforms.pathGuidingMode != kPathGuidingModePrototype || !eligible) {
                return float3(0.0f);
            }
            if (!pathGuidingValid) {
                return float3(0.03f, 0.10f, 0.03f);
            }
            const float confidence = max(pathGuidingConfidence, 0.20f);
            const float bin = float(pathGuidingBest.reserved0 & 63u) / 63.0f;
            const float guidedChoice = (pathGuidingBest.flags & 2u) != 0u ? 1.0f : 0.25f;
            const float pdf = clamp(pathGuidingBest.weightMoment.z, 0.0f, 1.0f);
            return float3(confidence, bin, max(pdf, guidedChoice * 0.35f));
        }
        case kDebugViewRestirPtReuseMask: {
            if (uniforms.restirPtMode != kRestirPtModeExperimentalPathReuse || !eligible) {
                return float3(0.0f);
            }
            const float v = pathReservoirValid ? max(ptReservoirConfidence, 0.45f) : 0.16f;
            return float3(v, 0.45f * v, 0.05f * v);
        }
        case kDebugViewRestirSvgfVariance: {
            const float guide = clamp((1.0f - debugRoughness) * 0.5f +
                                          abs(debugAO - 0.5f),
                                      0.0f,
                                      1.0f);
            return float3(guide);
        }
        case kDebugViewRestirNanInfMask: {
            const bool invalid =
                !all(isfinite(debugBaseColor)) ||
                !isfinite(debugMetallic) ||
                !isfinite(debugRoughness) ||
                !isfinite(debugAO);
            return invalid ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f);
        }
        case kDebugViewRadianceCache: {
            if (uniforms.radianceCacheMode != kRadianceCacheModePrototype || !radianceCacheStates) {
                return float3(0.0f);
            }
            const float invCell = 1.0f / max(uniforms.radianceCacheCellSize, 1.0e-4f);
            const int3 cell = int3(floor(rec.point * invCell));
            const uint hx = as_type<uint>(cell.x) * 73856093u;
            const uint hy = as_type<uint>(cell.y) * 19349663u;
            const uint hz = as_type<uint>(cell.z) * 83492791u;
            const uint hd = min(depth, kRestirPathReservoirDepthCount - 1u) * 2654435761u;
            const uint hm = static_cast<uint>(material.typeEta.x) * 2246822519u;
            const uint capacity = max(uniforms.width * uniforms.height * kRestirPathReservoirDepthCount, 1u);
            const uint domainKey = pcg_hash(hx ^ hy ^ hz ^ hd ^ hm ^ 0x9E3779B9u);
            const uint index = domainKey % capacity;
            const RadianceCacheState cacheState = radianceCacheStates[index];
            if ((cacheState.flags & 1u) == 0u ||
                cacheState.sampleCount == 0u ||
                cacheState.domainKey != domainKey ||
                cacheState.invalidationState != 0u) {
                return float3(0.28f, 0.03f, 0.02f);
            }
            const float samples = float(max(cacheState.sampleCount, 1u));
            const float3 cachedRadiance =
                float3(float(cacheState.radianceR),
                       float(cacheState.radianceG),
                       float(cacheState.radianceB)) / (4096.0f * samples);
            const float luma = luminance_rgb(max(cachedRadiance, float3(0.0f)));
            const float meanLuma2 = float(cacheState.lumaMoment) / (4096.0f * samples);
            const float variance = max(meanLuma2 - luma * luma, 0.0f);
            const float confidence =
                clamp(max(float(cacheState.confidenceQ) / 65535.0f,
                          (log2(samples + 1.0f) / 6.0f) * (1.0f / (1.0f + variance))),
                      0.0f,
                      1.0f);
            const bool hit = confidence >= uniforms.radianceCacheMinConfidence;
            const float train = clamp(float(cacheState.sampleCount) / 64.0f, 0.0f, 1.0f);
            if (!hit) {
                return float3(0.55f, 0.22f, train);
            }
            const float debugLuma = clamp(luma / max(uniforms.radianceCacheMaxRadiance, 1.0f), 0.0f, 1.0f);
            return float3(confidence, train, debugLuma);
        }
        default:
            break;
    }
    return float3(0.0f);
}
