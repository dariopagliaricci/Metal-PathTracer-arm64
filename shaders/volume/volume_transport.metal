constant uint kVolumeTransportModeHomogeneous = 1u;
constant uint kVolumePhaseIsotropic = 0u;
constant uint kVolumePhaseHenyeyGreenstein = 1u;

inline bool volume_transport_enabled(constant PathtraceUniforms& uniforms) {
    const float3 sigmaA = max(uniforms.volumeSigmaA.xyz, float3(0.0f));
    const float3 sigmaS = max(uniforms.volumeSigmaSAnisotropy.xyz, float3(0.0f));
    const float3 sigmaT = sigmaA + sigmaS;
    const float derivedMajorant = max(max(sigmaT.x, sigmaT.y), sigmaT.z);
    return PT_DEFAULT_OFF(uniforms.volumeTransportMode) == kVolumeTransportModeHomogeneous &&
           isfinite(uniforms.volumeSigmaA.w) &&
           all(isfinite(sigmaT)) &&
           max(uniforms.volumeSigmaA.w, derivedMajorant) > 0.0f;
}

inline uint4 volume_make_medium_state(const uint mediumId,
                                      const uint mediumDepth,
                                      const uint scatteringDepth,
                                      const uint phaseFunction) {
    return uint4(mediumId, mediumDepth, scatteringDepth, phaseFunction);
}

inline uint volume_active_medium_id(constant PathtraceUniforms& uniforms) {
    return volume_transport_enabled(uniforms) ? 1u : 0u;
}

inline uint volume_phase_id(constant PathtraceUniforms& uniforms) {
    return uniforms.volumePhaseFunction == kVolumePhaseHenyeyGreenstein
               ? kVolumePhaseHenyeyGreenstein
               : kVolumePhaseIsotropic;
}

inline uint4 volume_make_global_medium_state(constant PathtraceUniforms& uniforms,
                                             const uint mediumDepth,
                                             const uint scatteringDepth) {
    return volume_make_medium_state(volume_active_medium_id(uniforms),
                                    mediumDepth,
                                    scatteringDepth,
                                    volume_phase_id(uniforms));
}

inline void volume_apply_medium_state(thread Ray& ray,
                                      constant PathtraceUniforms& uniforms,
                                      const uint mediumDepth,
                                      const uint scatteringDepth) {
    ray.mediumState = volume_make_global_medium_state(uniforms, mediumDepth, scatteringDepth);
}

inline float4 volume_debug_aov_features(const float3 transmittance,
                                        const uint scatteringDepth,
                                        const uint phaseFunction,
                                        const float3 volumeContribution) {
    const float tr = clamp((transmittance.x + transmittance.y + transmittance.z) * (1.0f / 3.0f),
                           0.0f,
                           1.0f);
    return float4(tr,
                  float(scatteringDepth),
                  float(phaseFunction),
                  max(luminance_rgb(max(volumeContribution, float3(0.0f))), 0.0f));
}

inline MediumDescriptor volume_make_global_medium(constant PathtraceUniforms& uniforms) {
    MediumDescriptor medium;
    const float3 sigmaA = max(uniforms.volumeSigmaA.xyz, float3(0.0f));
    const float3 sigmaS = max(uniforms.volumeSigmaSAnisotropy.xyz, float3(0.0f));
    const float3 sigmaT = sigmaA + sigmaS;
    const float derivedMajorant = max(max(sigmaT.x, sigmaT.y), sigmaT.z);
    const float majorant = max(max(uniforms.volumeSigmaA.w, derivedMajorant), 0.0f);
    medium.sigmaA = float4(sigmaA, majorant);
    medium.sigmaSAnisotropy = float4(sigmaS,
                                     clamp(uniforms.volumeSigmaSAnisotropy.w, -0.98f, 0.98f));
    medium.emissionPhase = float4(max(uniforms.volumeEmissionMaxEvents.xyz, float3(0.0f)),
                                  float(uniforms.volumePhaseFunction));
    medium.metadata = uint4(1u,
                            max(uint(uniforms.volumeEmissionMaxEvents.w), 1u),
                            uniforms.volumeNeeEnabled,
                            0u);
    return medium;
}

inline float3 volume_sigma_t(thread const MediumDescriptor& medium) {
    return max(medium.sigmaA.xyz + medium.sigmaSAnisotropy.xyz, float3(0.0f));
}

inline float volume_majorant(thread const MediumDescriptor& medium) {
    return max(max(medium.sigmaA.w, medium.sigmaSAnisotropy.x),
               max(medium.sigmaSAnisotropy.y, medium.sigmaSAnisotropy.z));
}

inline float3 volume_transmittance(thread const MediumDescriptor& medium,
                                   const float distance,
                                   device PathtraceStats* stats) {
    if (stats) {
        atomic_fetch_add_explicit(&stats->volumeTransmittanceQueryCount, 1u, memory_order_relaxed);
    }
    return exp(-volume_sigma_t(medium) * max(distance, 0.0f));
}

// Surface NEE through the global homogeneous medium: a light sample with a position is attenuated by the medium's
// transmittance exp(-sigma_t d) over the distance d from the shading point to the sampled point. In an absorbing medium
// (sigma_s 0) this is what the path segment that reaches the same light by BSDF sampling is attenuated by. Directional
// lights (suns) and the environment are not attenuated, because escaped paths are not: the medium applies only to
// segments that end at a hit. Reservoir targets leave the transmittance out, as they leave out visibility; the shaded
// contribution carries it. ReSTIR DI does not run with volumes (RestirDiDemotedByVolume), so its passes have no call.
inline float3 volume_surface_nee_transmittance(constant PathtraceUniforms& uniforms,
                                               const float distance,
                                               const bool directional,
                                               device PathtraceStats* stats) {
    if (directional || !volume_transport_enabled(uniforms)) {
        return float3(1.0f);
    }
    MediumDescriptor medium = volume_make_global_medium(uniforms);
    return volume_transmittance(medium, distance, stats);
}

inline float volume_phase_isotropic() {
    return 1.0f / (4.0f * kPi);
}

// Henyey-Greenstein in pbrt's convention: cosTheta = dot(wo, wi) with wo pointing back along the incoming ray, so g > 0
// scatters forward (wi near -wo), as in Arnold. volume_phase_sample draws from the same density.
inline float volume_phase_hg(const float cosTheta, const float g) {
    const float gg = clamp(g, -0.98f, 0.98f);
    const float denom = max(1.0f + gg * gg + 2.0f * gg * cosTheta, 1.0e-4f);
    return (1.0f - gg * gg) / (4.0f * kPi * denom * sqrt(denom));
}

inline float volume_phase_eval(thread const MediumDescriptor& medium,
                               const float3 wo,
                               const float3 wi) {
    const uint phase = uint(medium.emissionPhase.w + 0.5f);
    const float cosTheta = clamp(dot(safe_normalize(wo), safe_normalize(wi)), -1.0f, 1.0f);
    if (phase == kVolumePhaseHenyeyGreenstein) {
        return volume_phase_hg(cosTheta, medium.sigmaSAnisotropy.w);
    }
    return volume_phase_isotropic();
}

inline void volume_orthonormal_basis(const float3 n,
                                     thread float3& tangent,
                                     thread float3& bitangent) {
    const float3 up = fabs(n.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(0.0f, 1.0f, 0.0f);
    tangent = safe_normalize(cross(up, n));
    bitangent = cross(n, tangent);
}

inline float3 volume_phase_sample(thread const MediumDescriptor& medium,
                                  const float3 wo,
                                  thread uint& rngState,
                                  thread float& outPdf) {
    const uint phase = uint(medium.emissionPhase.w + 0.5f);
    float cosTheta = 1.0f - 2.0f * rand_uniform(rngState);
    if (phase == kVolumePhaseHenyeyGreenstein) {
        const float g = clamp(medium.sigmaSAnisotropy.w, -0.98f, 0.98f);
        if (fabs(g) > 1.0e-3f) {
            const float u = rand_uniform(rngState);
            const float sqrTerm = (1.0f - g * g) / max(1.0f - g + 2.0f * g * u, 1.0e-4f);
            // Negative: the frame is built on wo (backward), and g > 0 must scatter forward (see volume_phase_hg).
            cosTheta = clamp(-(1.0f + g * g - sqrTerm * sqrTerm) / (2.0f * g), -1.0f, 1.0f);
        }
    }
    const float phi = 2.0f * kPi * rand_uniform(rngState);
    const float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    const float3 local = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    float3 tangent;
    float3 bitangent;
    const float3 basisW = safe_normalize(wo);
    volume_orthonormal_basis(basisW, tangent, bitangent);
    const float3 wi = safe_normalize(local.x * tangent + local.y * bitangent + local.z * basisW);
    outPdf = volume_phase_eval(medium, wo, wi);
    return wi;
}

inline float volume_sample_distance(thread const MediumDescriptor& medium,
                                    thread uint& rngState,
                                    thread float& outPdf) {
    const float majorant = max(volume_majorant(medium), 1.0e-6f);
    const float u = max(rand_uniform(rngState), 1.0e-6f);
    const float distance = -log(u) / majorant;
    outPdf = majorant * exp(-majorant * distance);
    return distance;
}

inline void volume_note_boundary_event(const int mediumEvent,
                                       device PathtraceStats* stats) {
    if (stats && mediumEvent != 0) {
        atomic_fetch_add_explicit(&stats->volumeMediumBoundaryEventCount, 1u, memory_order_relaxed);
    }
}

// Free flight along a segment of length s that ends at a hit (pbrt-v3's HomogeneousMedium::Sample): one colour
// channel c is chosen uniformly and the distance t is drawn with density sigma_t[c] exp(-sigma_t[c] t); a channel with
// sigma_t 0 never stops the flight. The weight divides by the mixture of the three channels' densities, so the
// transmittance is accounted once and the weight stays at most 3 in any medium:
//   scatter at t < s:  sigma_s T(t) / mean(sigma_t T(t)),    pass (t >= s):  T(s) / mean(T(s)),    T(x) = exp(-sigma_t x).
// In a grey medium both are 1 times sigma_s / sigma_t and 1.
struct VolumeFreeFlight {
    bool scattered;
    float distance;
    float3 weight;
};

inline VolumeFreeFlight volume_sample_free_flight(thread const MediumDescriptor& medium,
                                                  const float segment,
                                                  thread uint& rngState) {
    const float3 sigmaT = volume_sigma_t(medium);
    const float3 sigmaS = max(medium.sigmaSAnisotropy.xyz, float3(0.0f));
    const uint channel = min(uint(rand_uniform(rngState) * 3.0f), 2u);
    const float sigmaC = sigmaT[channel];
    const float u = rand_uniform(rngState);
    // 1 - u >= 1e-30 keeps the log finite under fast math: a flight stops within 69 mean free paths of its channel.
    const float t = sigmaC > 0.0f ? -log(max(1.0f - u, 1.0e-30f)) / sigmaC : kInfinity;
    VolumeFreeFlight flight;
    flight.scattered = t < segment;
    flight.distance = flight.scattered ? t : segment;
    const float3 tr = exp(-sigmaT * max(flight.distance, 0.0f));
    const float pdf = flight.scattered ? dot(sigmaT * tr, float3(1.0f / 3.0f)) : dot(tr, float3(1.0f / 3.0f));
    flight.weight = pdf > 0.0f ? (flight.scattered ? sigmaS * tr : tr) / pdf : float3(0.0f);
    return flight;
}

// Balance-heuristic weight of volume NEE to a rect light against the phase-sampled direction to the same light, which
// takes phasePdf / (phasePdf + lightPdf) at its hit; clamped like the surface rect-light weights.
inline float volume_nee_mis_weight(const float lightPdf, const float phasePdf) {
    const float denom = lightPdf + phasePdf;
    return denom > 0.0f ? clamp(lightPdf / denom, kMisWeightClampMin, kMisWeightClampMax) : 1.0f;
}

inline float3 volume_emission_integral(thread const MediumDescriptor& medium,
                                       const float distance) {
    const float3 sigmaT = volume_sigma_t(medium);
    const float3 tr = exp(-sigmaT * max(distance, 0.0f));
    const float3 denom = max(sigmaT, float3(1.0e-6f));
    return medium.emissionPhase.xyz * ((float3(1.0f) - tr) / denom);
}
