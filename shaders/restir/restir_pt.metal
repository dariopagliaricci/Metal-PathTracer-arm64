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

inline bool restir_pt_research_capture_valid(const BsdfSampleResult sample,
                                             const MaterialData material,
                                             const float3 position,
                                             const float3 normal,
                                             const float3 throughput,
                                             const uint depth,
                                             const bool specularOnly) {
    return depth >= 1u &&
           !specularOnly &&
           restir_prototype_diffuse_material_eligible(material) &&
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
    reservoir.materialType = static_cast<uint>(material.typeEta.x);
    reservoir.lobeType = sample.lobeType;
    reservoir.frameTag = uniforms.frameIndex;
    reservoir.flags = 1u;
    restirPtReservoirs[restir_path_state_index(uniforms, pixelCoord, depth)] = reservoir;
}

inline bool capture_restir_pt_research_scaffold(constant PathtraceUniforms& uniforms,
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
    if ((uniforms.restirPtMode != kRestirPtModeResearch &&
         uniforms.restirPtMode != kRestirPtModeExperimentalPathReuse) || !stats) {
        return false;
    }

    const bool diffuseEligible =
        depth >= 1u &&
        !specularOnly &&
        restir_prototype_diffuse_material_eligible(material) &&
        !sample.isDelta &&
        !sample.isBssrdf &&
        !sample.hasExitPoint &&
        sample.mediumEvent == 0 &&
        sample.lobeType == 0u;
    if (!diffuseEligible) {
        atomic_fetch_add_explicit(&stats->restirPtRejectedUnsupportedCount, 1u, memory_order_relaxed);
        return false;
    }

    if (!restir_pt_research_capture_valid(sample, material, position, normal, throughput, depth, specularOnly)) {
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
    reservoir.materialType = static_cast<uint>(material.typeEta.x);
    reservoir.lobeType = sample.lobeType;
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
    if (uniforms.restirPtDebug != 0u) {
        atomic_fetch_add_explicit(&stats->restirPtDebugRecordCount, 1u, memory_order_relaxed);
    }
    return true;
}

inline bool apply_restir_pt_experimental_path_reuse(constant PathtraceUniforms& uniforms,
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
    if (uniforms.restirPtMode != kRestirPtModeExperimentalPathReuse || !stats) {
        return false;
    }
    const bool diffuseEligible =
        depth >= 1u &&
        !specularOnly &&
        restir_prototype_diffuse_material_eligible(material) &&
        !sample.isDelta &&
        !sample.isBssrdf &&
        !sample.hasExitPoint &&
        sample.mediumEvent == 0 &&
        sample.lobeType == 0u;
    if (!diffuseEligible) {
        atomic_fetch_add_explicit(&stats->restirPtReuseFallbackCount, 1u, memory_order_relaxed);
        return false;
    }
    if (!restir_pt_research_capture_valid(sample, material, position, normal, throughput, depth, specularOnly)) {
        atomic_fetch_add_explicit(&stats->restirPtReuseRejectedInvalidCount, 1u, memory_order_relaxed);
        return false;
    }
    atomic_fetch_add_explicit(&stats->restirPtReuseCandidateCount, 1u, memory_order_relaxed);

    const float strength = clamp(uniforms.restirPtReuseStrength, 0.0f, 1.0f);
    if (strength <= 0.0f) {
        atomic_fetch_add_explicit(&stats->restirPtReuseFallbackCount, 1u, memory_order_relaxed);
        return false;
    }

    const float3 axis = safe_normalize(normal);
    const float3 dir = safe_normalize(sample.direction);
    float3 reservoirDirection = float3(0.0f);
    float reservoirConfidence = 0.0f;
    if (!load_restir_pt_reservoir_direction(restirPtReservoirs,
                                            uniforms,
                                            pixelCoord,
                                            material,
                                            position,
                                            axis,
                                            depth,
                                            reservoirDirection,
                                            reservoirConfidence)) {
        atomic_fetch_add_explicit(&stats->restirPtReuseFallbackCount, 1u, memory_order_relaxed);
        return false;
    }
    float3 reused = safe_normalize(mix(dir, reservoirDirection, min(strength, 1.0f) * reservoirConfidence));

    const float cosTheta = dot(axis, reused);
    if (!(cosTheta > 0.0f) || !all(isfinite(reused))) {
        atomic_fetch_add_explicit(&stats->restirPtReuseRejectedInvalidCount, 1u, memory_order_relaxed);
        return false;
    }
    if (!restir_reweight_bsdf_sample_for_direction(material,
                                                   position,
                                                   axis,
                                                   wo,
                                                   reused,
                                                   clampParams,
                                                   uniforms.sssMode,
                                                   diffuseOcclusion,
                                                   specularOnly,
                                                   sample)) {
        atomic_fetch_add_explicit(&stats->restirPtReuseRejectedInvalidCount, 1u, memory_order_relaxed);
        return false;
    }

    atomic_fetch_add_explicit(&stats->restirPtReuseAppliedCount, 1u, memory_order_relaxed);
    return true;
}
