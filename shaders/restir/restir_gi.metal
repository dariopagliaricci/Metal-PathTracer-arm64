inline bool restir_gi_diffuse_candidate_valid(const BsdfSampleResult candidate,
                                              const MaterialData material,
                                              const uint depth,
                                              const bool specularOnly) {
    return depth == 0u &&
           !specularOnly &&
           restir_prototype_diffuse_material_eligible(material) &&
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

inline bool load_restir_gi_reservoir_direction(device const RestirPtReservoirState* restirGiReservoirs,
                                               constant PathtraceUniforms& uniforms,
                                               const uint2 pixelCoord,
                                               const float3 normal,
                                               thread float3& direction,
                                               thread float& confidence) {
    if (!restirGiReservoirs) {
        return false;
    }
    const RestirPtReservoirState reservoir =
        restirGiReservoirs[restir_path_state_index(uniforms, pixelCoord, 0u)];
    if (!restir_state_valid(reservoir.flags) || reservoir.lobeType != 0u) {
        return false;
    }

    const float3 axis = safe_normalize(normal);
    const float3 reservoirNormal = safe_normalize(reservoir.normalDepth.xyz);
    const float normalAgreement = dot(axis, reservoirNormal);
    if (!(normalAgreement > 0.0f)) {
        return false;
    }

    const float3 candidate = safe_normalize(reservoir.directionWeight.xyz);
    if (!all(isfinite(candidate)) || dot(candidate, candidate) <= 0.0f || dot(axis, candidate) <= 0.0f) {
        return false;
    }

    const float sampleLuma = max(reservoir.directionWeight.w, 0.0f);
    if (!(sampleLuma > 0.0f) || !isfinite(sampleLuma)) {
        return false;
    }

    direction = candidate;
    confidence = clamp(normalAgreement * sampleLuma /
                           max(sampleLuma + reservoir.throughputLuma.w, 1.0e-4f),
                       0.0625f,
                       1.0f);
    return true;
}

inline void update_restir_gi_reservoir(device RestirPtReservoirState* restirGiReservoirs,
                                       constant PathtraceUniforms& uniforms,
                                       const uint2 pixelCoord,
                                       const MaterialData material,
                                       const float3 position,
                                       const float3 normal,
                                       const BsdfSampleResult sample) {
    if (!restirGiReservoirs || sample.pdf <= 0.0f || !all(isfinite(sample.direction))) {
        return;
    }
    const float sampleLuma = max(luminance_rgb(max(sample.weight, float3(0.0f))), 0.0f);
    if (!(sampleLuma > 0.0f) ||
        !isfinite(sampleLuma) ||
        !all(isfinite(position)) ||
        !all(isfinite(normal))) {
        return;
    }

    RestirPtReservoirState reservoir{};
    reservoir.positionPdf = float4(position, sample.pdf);
    reservoir.normalDepth = float4(safe_normalize(normal), 0.0f);
    reservoir.directionWeight = float4(safe_normalize(sample.direction), sampleLuma);
    reservoir.throughputLuma = float4(1.0f, 1.0f, 1.0f, sampleLuma);
    reservoir.materialType = static_cast<uint>(material.typeEta.x);
    reservoir.lobeType = sample.lobeType;
    reservoir.frameTag = uniforms.frameIndex;
    reservoir.flags = 1u;
    restirGiReservoirs[restir_path_state_index(uniforms, pixelCoord, 0u)] = reservoir;
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
                                                      thread BsdfSampleResult& sample);

inline bool apply_restir_gi_diffuse_prototype(constant PathtraceUniforms& uniforms,
                                              const MaterialData material,
                                              const float3 position,
                                              const float3 normal,
                                              const float3 wo,
                                              const float3 incidentDir,
                                              const bool frontFace,
                                              const FireflyClampParams clampParams,
                                              const float diffuseOcclusion,
                                              const bool specularOnly,
                                              const uint depth,
                                              const uint2 pixelCoord,
                                              device RestirPtReservoirState* restirGiReservoirs,
                                              device PathtraceStats* stats,
                                              thread uint& state,
                                              thread BsdfSampleResult& sample) {
    if (uniforms.restirGiMode != kRestirGiModeDiffusePrototype) {
        return false;
    }
    if (!restir_gi_diffuse_candidate_valid(sample, material, depth, specularOnly)) {
        if (stats) {
            atomic_fetch_add_explicit(&stats->restirGiFallbackCount, 1u, memory_order_relaxed);
        }
        return false;
    }
    if (stats) {
        atomic_fetch_add_explicit(&stats->restirGiCandidateCount, 1u, memory_order_relaxed);
    }

    bool reused = false;
    float3 reservoirDirection = float3(0.0f);
    float reservoirConfidence = 0.0f;
    if (load_restir_gi_reservoir_direction(restirGiReservoirs,
                                           uniforms,
                                           pixelCoord,
                                           normal,
                                           reservoirDirection,
                                           reservoirConfidence)) {
        BsdfSampleResult reuseSample = sample;
        if (restir_reweight_bsdf_sample_for_direction(material,
                                                      position,
                                                      normal,
                                                      wo,
                                                      reservoirDirection,
                                                      clampParams,
                                                      uniforms.sssMode,
                                                      diffuseOcclusion,
                                                      specularOnly,
                                                      reuseSample)) {
            const float currentScore = luminance_rgb(max(sample.weight, float3(0.0f)));
            const float reuseScore = luminance_rgb(max(reuseSample.weight, float3(0.0f))) *
                                     clamp(reservoirConfidence, 0.0f, 1.0f);
            if (currentScore > 0.0f && reuseScore > 0.0f &&
                isfinite(currentScore) && isfinite(reuseScore)) {
                const float chooseReuse = reuseScore / max(currentScore + reuseScore, 1.0e-6f);
                if (rand_uniform(state) < chooseReuse) {
                    sample = reuseSample;
                    sample.direction = safe_normalize(sample.direction);
                    reused = true;
                }
            }
        }
    }

    update_restir_gi_reservoir(restirGiReservoirs,
                               uniforms,
                               pixelCoord,
                               material,
                               position,
                               normal,
                               sample);
    if (stats) {
        atomic_fetch_add_explicit(&stats->restirGiReservoirUpdateCount, 1u, memory_order_relaxed);
        if (reused) {
            atomic_fetch_add_explicit(&stats->restirGiAcceptCount, 1u, memory_order_relaxed);
        } else {
            atomic_fetch_add_explicit(&stats->restirGiRejectCount, 1u, memory_order_relaxed);
        }
    }
    return reused;
}
