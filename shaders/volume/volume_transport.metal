constant uint kVolumeTransportModeHomogeneous = 1u;
constant uint kVolumePhaseIsotropic = 0u;
constant uint kVolumePhaseHenyeyGreenstein = 1u;

inline bool volume_transport_enabled(constant PathtraceUniforms& uniforms) {
    const float3 sigmaA = max(uniforms.volumeSigmaA.xyz, float3(0.0f));
    const float3 sigmaS = max(uniforms.volumeSigmaSAnisotropy.xyz, float3(0.0f));
    const float3 sigmaT = sigmaA + sigmaS;
    const float derivedMajorant = max(max(sigmaT.x, sigmaT.y), sigmaT.z);
    return uniforms.volumeTransportMode == kVolumeTransportModeHomogeneous &&
           isfinite(uniforms.volumeSigmaA.w) &&
           all(isfinite(sigmaT)) &&
           max(uniforms.volumeSigmaA.w, derivedMajorant) > 0.0f;
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

inline float volume_phase_isotropic() {
    return 1.0f / (4.0f * kPi);
}

inline float volume_phase_hg(const float cosTheta, const float g) {
    const float gg = clamp(g, -0.98f, 0.98f);
    const float denom = max(1.0f + gg * gg - 2.0f * gg * cosTheta, 1.0e-4f);
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
            cosTheta = clamp((1.0f + g * g - sqrTerm * sqrTerm) / (2.0f * g), -1.0f, 1.0f);
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

inline float3 volume_emission_integral(thread const MediumDescriptor& medium,
                                       const float distance) {
    const float3 sigmaT = volume_sigma_t(medium);
    const float3 tr = exp(-sigmaT * max(distance, 0.0f));
    const float3 denom = max(sigmaT, float3(1.0e-6f));
    return medium.emissionPhase.xyz * ((float3(1.0f) - tr) / denom);
}
