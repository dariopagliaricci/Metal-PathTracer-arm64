constant uint kPathGuidingBinDim = 8u;
constant uint kPathGuidingBinCount = kPathGuidingBinDim * kPathGuidingBinDim;
constant float kPathGuidingInvHemisphereSolidAngle = 0.15915494309189535f;

struct GuidingQuery {
    float3 position;
    float3 normal;
    float3 outgoingDirection;
    float roughness;
    uint materialID;
    uint depth;
};

struct GuidingSample {
    float3 direction;
    float pdf;
    float confidence;
    uint cellID;
    uint binID;
};

inline uint path_guiding_cell_hash(const float3 position, const float cellSize) {
    float invCell = 1.0f / max(cellSize, 1.0e-4f);
    int3 cell = int3(floor(position * invCell));
    uint hx = as_type<uint>(cell.x) * 73856093u;
    uint hy = as_type<uint>(cell.y) * 19349663u;
    uint hz = as_type<uint>(cell.z) * 83492791u;
    return pcg_hash(hx ^ hy ^ hz ^ 0xC2B2AE35u);
}

inline uint path_guiding_total_bin_capacity(constant PathtraceUniforms& uniforms) {
    return max(uniforms.width * uniforms.height * kRestirPathReservoirDepthCount, kPathGuidingBinCount);
}

inline uint path_guiding_cell_count(constant PathtraceUniforms& uniforms) {
    return max(path_guiding_total_bin_capacity(uniforms) / kPathGuidingBinCount, 1u);
}

inline uint path_guiding_cell_id(constant PathtraceUniforms& uniforms,
                                 const float3 position,
                                 const float cellSize) {
    return path_guiding_cell_hash(position, cellSize) % path_guiding_cell_count(uniforms);
}

inline uint path_guiding_cell_base_index(constant PathtraceUniforms& uniforms,
                                         const uint cellID) {
    return min(cellID, path_guiding_cell_count(uniforms) - 1u) * kPathGuidingBinCount;
}

inline float3 path_guiding_to_local(const float3 direction, const float3 normal) {
    float3 tangent;
    float3 bitangent;
    build_onb(safe_normalize(normal), tangent, bitangent);
    return float3(dot(direction, tangent), dot(direction, bitangent), dot(direction, safe_normalize(normal)));
}

inline float3 path_guiding_to_world(const float3 local, const float3 normal) {
    float3 tangent;
    float3 bitangent;
    const float3 axis = safe_normalize(normal);
    build_onb(axis, tangent, bitangent);
    return safe_normalize(local.x * tangent + local.y * bitangent + local.z * axis);
}

inline float2 path_guiding_octa_encode_hemisphere(const float3 localDirection) {
    float3 d = safe_normalize(float3(localDirection.x, localDirection.y, max(localDirection.z, 0.0f)));
    const float invL1 = 1.0f / max(abs(d.x) + abs(d.y) + abs(d.z), 1.0e-6f);
    float2 p = d.xy * invL1;
    return clamp(p * 0.5f + 0.5f, float2(0.0f), float2(0.999999f));
}

inline float3 path_guiding_octa_decode_hemisphere(const float2 uv) {
    float2 f = uv * 2.0f - 1.0f;
    float3 n = float3(f.x, f.y, 1.0f - abs(f.x) - abs(f.y));
    if (n.z < 0.0f) {
        float2 folded = (1.0f - abs(float2(n.y, n.x))) * sign(f);
        n = float3(folded.x, folded.y, 0.0f);
    }
    n.z = max(n.z, 1.0e-4f);
    return safe_normalize(n);
}

inline uint path_guiding_bin_id_for_direction(const float3 direction, const float3 normal) {
    const float3 local = path_guiding_to_local(safe_normalize(direction), normal);
    const float2 uv = path_guiding_octa_encode_hemisphere(local);
    const uint bx = min(static_cast<uint>(floor(uv.x * float(kPathGuidingBinDim))), kPathGuidingBinDim - 1u);
    const uint by = min(static_cast<uint>(floor(uv.y * float(kPathGuidingBinDim))), kPathGuidingBinDim - 1u);
    return by * kPathGuidingBinDim + bx;
}

inline float3 path_guiding_bin_center_direction(const uint binID, const float3 normal) {
    const uint bx = binID % kPathGuidingBinDim;
    const uint by = binID / kPathGuidingBinDim;
    const float2 uv =
        (float2(float(bx), float(by)) + float2(0.5f)) / float(kPathGuidingBinDim);
    return path_guiding_to_world(path_guiding_octa_decode_hemisphere(uv), normal);
}

inline bool path_guiding_material_allowed(const MaterialData material,
                                          const uint depth,
                                          const bool specularOnly) {
    return depth >= 1u &&
           !specularOnly &&
           material_closure_is_guideable(material);
}

inline bool path_guiding_candidate_valid(const BsdfSampleResult candidate,
                                         const MaterialData material,
                                         const uint depth,
                                         const bool specularOnly) {
    return path_guiding_material_allowed(material, depth, specularOnly) &&
           candidate.pdf > 0.0f &&
           candidate.directionalPdf > 0.0f &&
           !candidate.isDelta &&
           !candidate.isBssrdf &&
           !candidate.hasExitPoint &&
           candidate.mediumEvent == 0 &&
           candidate.lobeType == 0u &&
           all(isfinite(candidate.direction)) &&
           dot(candidate.direction, candidate.direction) > 0.0f &&
           all(isfinite(candidate.weight)) &&
           all(candidate.weight >= float3(0.0f));
}

inline float path_guiding_bin_weight(const PathGuidingReservoirState state,
                                     const uint frameIndex) {
    if (!restir_state_valid(state.flags) ||
        state.sampleCount == 0u ||
        state.frameTag >= frameIndex ||
        state.reserved0 == 0u) {
        return 0.0f;
    }
    const float meanLuma = float(state.reserved0) / (256.0f * float(max(state.sampleCount, 1u)));
    const float countWeight = sqrt(float(max(state.sampleCount, 1u)));
    const float weight = max(meanLuma, 1.0e-4f) * countWeight;
    return isfinite(weight) ? weight : 0.0f;
}

inline float path_guiding_cell_total_weight(device const PathGuidingReservoirState* pathGuidingStates,
                                            constant PathtraceUniforms& uniforms,
                                            const uint cellID,
                                            thread uint& sampleCountSum) {
    if (!pathGuidingStates) {
        sampleCountSum = 0u;
        return 0.0f;
    }
    const uint base = path_guiding_cell_base_index(uniforms, cellID);
    float total = 0.0f;
    sampleCountSum = 0u;
    for (uint bin = 0u; bin < kPathGuidingBinCount; ++bin) {
        const PathGuidingReservoirState state = pathGuidingStates[base + bin];
        total += path_guiding_bin_weight(state, uniforms.frameIndex);
        if (restir_state_valid(state.flags)) {
            sampleCountSum += min(state.sampleCount, 65535u);
        }
    }
    return isfinite(total) ? total : 0.0f;
}

inline float path_guiding_confidence_from_samples(const uint sampleCountSum) {
    return clamp(float(sampleCountSum) / 64.0f, 0.0f, 1.0f);
}

inline bool path_guiding_can_guide(const GuidingQuery query,
                                   const MaterialData material,
                                   const bool specularOnly) {
    return path_guiding_material_allowed(material, query.depth, specularOnly) &&
           all(isfinite(query.position)) &&
           all(isfinite(query.normal)) &&
           dot(query.normal, query.normal) > 0.0f;
}

inline float path_guiding_pdf_direction(device const PathGuidingReservoirState* pathGuidingStates,
                                        constant PathtraceUniforms& uniforms,
                                        const GuidingQuery query,
                                        const float3 direction) {
    if (!pathGuidingStates || !all(isfinite(direction))) {
        return 0.0f;
    }
    const float3 axis = safe_normalize(query.normal);
    const float cosTheta = dot(axis, safe_normalize(direction));
    if (!(cosTheta > 0.0f)) {
        return 0.0f;
    }
    const float cellSize = max(uniforms.pathGuidingCellSize, 1.0e-4f);
    const uint cellID = path_guiding_cell_id(uniforms, query.position, cellSize);
    uint sampleCountSum = 0u;
    const float total = path_guiding_cell_total_weight(pathGuidingStates, uniforms, cellID, sampleCountSum);
    if (!(total > 0.0f) || sampleCountSum == 0u) {
        return 0.0f;
    }
    const uint binID = path_guiding_bin_id_for_direction(direction, axis);
    const uint index = path_guiding_cell_base_index(uniforms, cellID) + binID;
    const float binWeight = path_guiding_bin_weight(pathGuidingStates[index], uniforms.frameIndex);
    if (!(binWeight > 0.0f)) {
        return 0.0f;
    }
    const float pdf = (binWeight / total) * float(kPathGuidingBinCount) * kPathGuidingInvHemisphereSolidAngle;
    return (pdf > 0.0f && isfinite(pdf)) ? pdf : 0.0f;
}

inline bool path_guiding_sample_direction(device const PathGuidingReservoirState* pathGuidingStates,
                                          constant PathtraceUniforms& uniforms,
                                          const GuidingQuery query,
                                          thread uint& rngState,
                                          thread GuidingSample& sample) {
    if (!pathGuidingStates) {
        return false;
    }
    const float cellSize = max(uniforms.pathGuidingCellSize, 1.0e-4f);
    const uint cellID = path_guiding_cell_id(uniforms, query.position, cellSize);
    uint sampleCountSum = 0u;
    const float total = path_guiding_cell_total_weight(pathGuidingStates, uniforms, cellID, sampleCountSum);
    if (!(total > 0.0f) || sampleCountSum == 0u) {
        return false;
    }

    float target = rand_uniform(rngState) * total;
    uint selectedBin = 0u;
    float selectedWeight = 0.0f;
    const uint base = path_guiding_cell_base_index(uniforms, cellID);
    for (uint bin = 0u; bin < kPathGuidingBinCount; ++bin) {
        const float weight = path_guiding_bin_weight(pathGuidingStates[base + bin], uniforms.frameIndex);
        if (weight <= 0.0f) {
            continue;
        }
        selectedBin = bin;
        selectedWeight = weight;
        if (target <= weight) {
            break;
        }
        target -= weight;
    }

    const float3 direction = path_guiding_bin_center_direction(selectedBin, query.normal);
    const float pdf =
        (selectedWeight / total) * float(kPathGuidingBinCount) * kPathGuidingInvHemisphereSolidAngle;
    if (!(pdf > 0.0f) || !isfinite(pdf)) {
        return false;
    }

    sample.direction = safe_normalize(direction);
    sample.pdf = pdf;
    sample.confidence = path_guiding_confidence_from_samples(sampleCountSum);
    sample.cellID = cellID;
    sample.binID = selectedBin;
    return sample.confidence > 0.0f;
}

inline bool path_guiding_reweight_sample_for_direction(const MaterialData material,
                                                       const float3 position,
                                                       const float3 normal,
                                                       const float3 wo,
                                                       const float3 wi,
                                                       const float proposalPdf,
                                                       const FireflyClampParams clampParams,
                                                       const uint sssMode,
                                                       const float diffuseOcclusion,
                                                       const bool specularOnly,
                                                       thread BsdfSampleResult& sample) {
    const float3 axis = safe_normalize(normal);
    const float3 direction = safe_normalize(wi);
    const float cosTheta = dot(axis, direction);
    if (!(cosTheta > 0.0f) || !(proposalPdf > 0.0f) || !isfinite(proposalPdf)) {
        return false;
    }

    BsdfEvalResult eval = evaluate_bsdf(material,
                                        position,
                                        axis,
                                        wo,
                                        direction,
                                        clampParams,
                                        sssMode,
                                        diffuseOcclusion,
                                        specularOnly);
    const float bsdfPdf = (eval.directionalPdf > 0.0f) ? eval.directionalPdf : eval.pdf;
    if (!(bsdfPdf > 0.0f) ||
        !isfinite(bsdfPdf) ||
        eval.isDelta ||
        eval.isBssrdf ||
        !all(isfinite(eval.value))) {
        return false;
    }

    const float3 weight = max(eval.value * (cosTheta / max(proposalPdf, 1.0e-6f)), float3(0.0f));
    if (!all(isfinite(weight)) || any(weight < float3(0.0f))) {
        return false;
    }

    sample.direction = direction;
    sample.weight = weight;
    sample.pdf = proposalPdf;
    sample.directionalPdf = proposalPdf;
    sample.areaPdf = 0.0f;
    sample.isDelta = false;
    sample.isBssrdf = false;
    sample.hasExitPoint = false;
    sample.mediumEvent = 0;
    sample.lobeType = 0u;
    sample.lobeRoughness = max(sample.lobeRoughness, 1.0f);
    return true;
}

inline bool restir_reweight_bsdf_sample_for_direction(const MaterialData material,
                                                      const float3 position,
                                                      const float3 normal,
                                                      const float3 wo,
                                                      const float3 wi,
                                                      const FireflyClampParams clampParams,
                                                      const uint sssMode,
                                                      const float diffuseOcclusion,
                                                      const bool specularOnly,
                                                      thread BsdfSampleResult& sample) {
    const float3 axis = safe_normalize(normal);
    const float3 direction = safe_normalize(wi);
    const float cosTheta = dot(axis, direction);
    if (!(cosTheta > 0.0f) || !all(isfinite(direction))) {
        return false;
    }

    BsdfEvalResult eval = evaluate_bsdf(material,
                                        position,
                                        axis,
                                        wo,
                                        direction,
                                        clampParams,
                                        sssMode,
                                        diffuseOcclusion,
                                        specularOnly);
    const float pdf = eval.pdf;
    const float directionalPdf = (eval.directionalPdf > 0.0f) ? eval.directionalPdf : pdf;
    if (!(pdf > 0.0f) ||
        !(directionalPdf > 0.0f) ||
        !isfinite(pdf) ||
        !isfinite(directionalPdf) ||
        eval.isDelta ||
        eval.isBssrdf ||
        !all(isfinite(eval.value))) {
        return false;
    }

    const float3 weight = max(eval.value * (cosTheta / max(pdf, 1.0e-6f)), float3(0.0f));
    if (!all(isfinite(weight)) || any(weight < float3(0.0f))) {
        return false;
    }

    sample.direction = direction;
    sample.weight = weight;
    sample.pdf = pdf;
    sample.directionalPdf = directionalPdf;
    sample.areaPdf = eval.areaPdf;
    sample.isDelta = false;
    sample.isBssrdf = false;
    sample.hasExitPoint = false;
    sample.mediumEvent = 0;
    sample.lobeType = 0u;
    sample.lobeRoughness = max(sample.lobeRoughness, 1.0f);
    return true;
}

inline void path_guiding_update(device PathGuidingReservoirState* pathGuidingStates,
                                constant PathtraceUniforms& uniforms,
                                const GuidingQuery query,
                                const float3 direction,
                                const float3 incidentRadiance,
                                const float weight,
                                const float guidingPdf,
                                const bool guidedChoice) {
    if (!pathGuidingStates ||
        !all(isfinite(direction)) ||
        !all(isfinite(incidentRadiance)) ||
        !(weight > 0.0f) ||
        !isfinite(weight)) {
        return;
    }

    const uint interval = max(uniforms.pathGuidingTrainingInterval, 1u);
    if ((uniforms.frameIndex % interval) != 0u) {
        return;
    }

    const float3 axis = safe_normalize(query.normal);
    if (dot(axis, safe_normalize(direction)) <= 0.0f) {
        return;
    }

    const float cellSize = max(uniforms.pathGuidingCellSize, 1.0e-4f);
    const uint cellID = path_guiding_cell_id(uniforms, query.position, cellSize);
    const uint binID = path_guiding_bin_id_for_direction(direction, axis);
    const uint index = path_guiding_cell_base_index(uniforms, cellID) + binID;

    const float sampleLuma = max(luminance_rgb(max(incidentRadiance, float3(0.0f))) * weight, 0.0f);
    if (!(sampleLuma > 0.0f) || !isfinite(sampleLuma)) {
        return;
    }

    const uint quantizedLuma =
        clamp(static_cast<uint>(sampleLuma * 256.0f), 1u, 1048576u);
    device atomic_uint* sampleCount =
        reinterpret_cast<device atomic_uint*>(&pathGuidingStates[index].sampleCount);
    device atomic_uint* lumaAccumulator =
        reinterpret_cast<device atomic_uint*>(&pathGuidingStates[index].reserved0);
    const uint previousCount =
        atomic_fetch_add_explicit(sampleCount, 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(lumaAccumulator, quantizedLuma, memory_order_relaxed);
    const float confidence = clamp(float(previousCount + 1u) / 16.0f, 0.0625f, 1.0f);

    pathGuidingStates[index].directionConfidence =
        float4(path_guiding_bin_center_direction(binID, axis), confidence);
    pathGuidingStates[index].weightMoment = float4(sampleLuma,
                                                   sampleLuma,
                                                   max(guidingPdf, 0.0f),
                                                   guidedChoice ? 1.0f : 0.0f);
    pathGuidingStates[index].frameTag = uniforms.frameIndex;
    pathGuidingStates[index].flags = 1u | (guidedChoice ? 2u : 0u);
}

inline bool apply_path_guiding_prototype(constant PathtraceUniforms& uniforms,
                                         const MaterialData material,
                                         const float3 position,
                                         const float3 normal,
                                         const float3 wo,
                                         const FireflyClampParams clampParams,
                                         const float diffuseOcclusion,
                                         const bool specularOnly,
                                         const uint depth,
                                         const uint2 pixelCoord,
                                         device PathGuidingReservoirState* pathGuidingStates,
                                         device PathtraceStats* stats,
                                         thread uint& rngState,
                                         thread BsdfSampleResult& sample) {
    (void)pixelCoord;
    if (uniforms.pathGuidingMode != kPathGuidingModePrototype) {
        return false;
    }
    if (!path_guiding_candidate_valid(sample, material, depth, specularOnly)) {
        if (stats) {
            atomic_fetch_add_explicit(&stats->pathGuidingFallbackCount, 1u, memory_order_relaxed);
        }
        return false;
    }
    if (stats) {
        atomic_fetch_add_explicit(&stats->pathGuidingCandidateCount, 1u, memory_order_relaxed);
    }

    const float strength = clamp(uniforms.pathGuidingStrength, 0.0f, 1.0f);
    if (strength <= 0.0f || !pathGuidingStates) {
        if (stats) {
            atomic_fetch_add_explicit(&stats->pathGuidingFallbackCount, 1u, memory_order_relaxed);
        }
        return false;
    }

    GuidingQuery query{};
    query.position = position;
    query.normal = safe_normalize(normal);
    query.outgoingDirection = safe_normalize(wo);
    query.roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
    query.materialID = 0u;
    query.depth = depth;
    if (!path_guiding_can_guide(query, material, specularOnly)) {
        if (stats) {
            atomic_fetch_add_explicit(&stats->pathGuidingFallbackCount, 1u, memory_order_relaxed);
        }
        return false;
    }

    GuidingSample guidedSample{};
    const bool hasGuidedDistribution =
        path_guiding_sample_direction(pathGuidingStates, uniforms, query, rngState, guidedSample);

    bool guidedChoice = false;
    float guidingPdf = 0.0f;
    float mixturePdf = sample.directionalPdf;
    BsdfSampleResult selectedSample = sample;
    if (hasGuidedDistribution) {
        const float guideChoiceProbability =
            clamp(strength * guidedSample.confidence, 0.0f, 0.95f);
        const float bsdfGuidingPdf =
            path_guiding_pdf_direction(pathGuidingStates, uniforms, query, sample.direction);
        const float bsdfPdf = max(sample.directionalPdf, sample.pdf);
        const float bsdfMixturePdf =
            (1.0f - guideChoiceProbability) * bsdfPdf + guideChoiceProbability * bsdfGuidingPdf;
        if (bsdfMixturePdf > 0.0f && isfinite(bsdfMixturePdf)) {
            BsdfSampleResult bsdfMixtureSample = sample;
            if (path_guiding_reweight_sample_for_direction(material,
                                                           position,
                                                           query.normal,
                                                           wo,
                                                           sample.direction,
                                                           bsdfMixturePdf,
                                                           clampParams,
                                                           uniforms.sssMode,
                                                           diffuseOcclusion,
                                                           specularOnly,
                                                           bsdfMixtureSample)) {
                selectedSample = bsdfMixtureSample;
                mixturePdf = bsdfMixturePdf;
            }
        }

        if (rand_uniform(rngState) < guideChoiceProbability) {
            BsdfEvalResult eval = evaluate_bsdf(material,
                                                position,
                                                query.normal,
                                                wo,
                                                guidedSample.direction,
                                                clampParams,
                                                uniforms.sssMode,
                                                diffuseOcclusion,
                                                specularOnly);
            const float guidedBsdfPdf = (eval.directionalPdf > 0.0f) ? eval.directionalPdf : eval.pdf;
            const float guidedMixturePdf =
                (1.0f - guideChoiceProbability) * guidedBsdfPdf +
                guideChoiceProbability * guidedSample.pdf;
            BsdfSampleResult guidedBsdfSample = sample;
            if (path_guiding_reweight_sample_for_direction(material,
                                                           position,
                                                           query.normal,
                                                           wo,
                                                           guidedSample.direction,
                                                           guidedMixturePdf,
                                                           clampParams,
                                                           uniforms.sssMode,
                                                           diffuseOcclusion,
                                                           specularOnly,
                                                           guidedBsdfSample)) {
                selectedSample = guidedBsdfSample;
                mixturePdf = guidedMixturePdf;
                guidingPdf = guidedSample.pdf;
                guidedChoice = true;
            }
        }
    }

    if (!(mixturePdf > 0.0f) || !isfinite(mixturePdf)) {
        if (stats) {
            atomic_fetch_add_explicit(&stats->pathGuidingInvalidCount, 1u, memory_order_relaxed);
        }
        return false;
    }

    sample = selectedSample;
    const float updatePdf = guidedChoice
        ? guidingPdf
        : path_guiding_pdf_direction(pathGuidingStates, uniforms, query, sample.direction);
    path_guiding_update(pathGuidingStates,
                        uniforms,
                        query,
                        sample.direction,
                        sample.weight,
                        1.0f,
                        updatePdf,
                        guidedChoice);

    if (guidedChoice) {
        if (stats) {
            atomic_fetch_add_explicit(&stats->pathGuidingUsedCount, 1u, memory_order_relaxed);
        }
        return true;
    }
    if (hasGuidedDistribution && stats) {
        atomic_fetch_add_explicit(&stats->pathGuidingFallbackCount, 1u, memory_order_relaxed);
    }
    return false;
}
