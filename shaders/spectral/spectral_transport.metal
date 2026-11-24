constant uint kSpectralModeHeroWavelength = 1u;        // three wavelengths per path (C = 3)
constant uint kSpectralModeHeroWavelengthSingle = 2u;  // one wavelength per path (C = 1), the A/B baseline
constant uint kSpectralTexturePolicyRGBReflectanceD65 = 0u;
constant uint kSpectralMaterialPolicyRGBApproxD65 = 0u;
constant uint kSpectralMaterialPolicySampledTriplet = 1u;

// Channel-mapped spectral transport. With spectral mode on, channel j of every float3 color quantity inside the
// path (throughput, BSDF values, emission, light radiance) is the value at wavelength lambda_j:
//
//   lambda_j = min + (max - min) * frac(u + j / C),  j < C;  lambda_0 is the hero wavelength.
//
// Directions, pdfs and MIS weights are shared by all channels. Every RGB quantity entering the path goes through
// spectral_rgb_to_channels exactly once. The sensor response is applied once, when the path's radiance leaves
// the integrator (spectral_channels_to_rgb), with a white balance that maps a flat spectrum to RGB (1, 1, 1).
// Inactive channels (j >= C) replicate channel 0's wavelength so every per-channel value stays finite; their
// throughput is zero and spectral_channels_to_rgb masks them out.
struct SpectralPathState {
    float wavelengthNm;     // hero wavelength, channel 0
    uint sampleIndex;
    float3 wavelengthsNm;   // per channel; inactive channels hold the hero wavelength
    float3 sensor0;         // s(lambda_0)
    float3 sensor1;         // s(lambda_1)
    float3 sensor2;         // s(lambda_2)
    uint channelCount;      // C
    bool enabled;
    bool collapsed;         // throughput reduced to the hero channel at a dispersive vertex
};

// Wavefront PathState.spectralFlags bits.
constant uint kSpectralFlagCollapsed = 1u;

inline bool spectral_rendering_enabled(constant PathtraceUniforms& uniforms) {
    return (PT_DEFAULT_OFF(uniforms.spectralRenderingMode) == kSpectralModeHeroWavelength ||
            PT_DEFAULT_OFF(uniforms.spectralRenderingMode) == kSpectralModeHeroWavelengthSingle) &&
           uniforms.spectralWavelengthRange.y > uniforms.spectralWavelengthRange.x;
}

// Wavelengths carried per path (C).
inline uint spectral_channel_count(constant PathtraceUniforms& uniforms) {
    return PT_DEFAULT_OFF(uniforms.spectralRenderingMode) == kSpectralModeHeroWavelengthSingle ? 1u : 3u;
}

// Sampled range in nm: (min, max), clamped as the host does.
inline float2 spectral_wavelength_range_nm(constant PathtraceUniforms& uniforms) {
    const float minNm = max(min(uniforms.spectralWavelengthRange.x,
                                uniforms.spectralWavelengthRange.y - 1.0f),
                            300.0f);
    const float maxNm = min(max(uniforms.spectralWavelengthRange.y, minNm + 1.0f), 830.0f);
    return float2(minNm, maxNm);
}

// W_c = 1 / E[s_c(lambda)] over the range, computed on the host (SpectralSensor::WhiteBalance).
inline float3 spectral_white_balance(constant PathtraceUniforms& uniforms) {
    return float3(uniforms.spectralWavelengthRange.w,
                  uniforms.spectralWhiteBalanceG,
                  uniforms.spectralWhiteBalanceB);
}

inline float spectral_gaussian(const float wavelengthNm, const float centerNm, const float widthNm) {
    const float x = (wavelengthNm - centerNm) / max(widthNm, 1.0f);
    return exp(-0.5f * x * x);
}

// Kept identical to SpectralSensor::Response (include/renderer/SpectralSensor.h).
inline float3 spectral_sensor_rgb(const float wavelengthNm) {
    float3 response = float3(spectral_gaussian(wavelengthNm, 610.0f, 48.0f),
                             spectral_gaussian(wavelengthNm, 545.0f, 42.0f),
                             spectral_gaussian(wavelengthNm, 455.0f, 36.0f));
    const float normalizer = max(max(response.x, response.y), response.z);
    if (normalizer <= 1.0e-5f) {
        return float3(1.0f / 3.0f);
    }
    return response / normalizer;
}

inline float spectral_rgb_to_sample(const float3 rgb, const float3 sensorRgb) {
    const float denom = max(dot(sensorRgb, float3(1.0f)), 1.0e-5f);
    return max(dot(max(rgb, float3(0.0f)), sensorRgb) / denom, 0.0f);
}

inline float spectral_triplet_to_sample(const float3 triplet, const float wavelengthNm) {
    const float3 clamped = max(triplet, float3(0.0f));
    if (wavelengthNm <= 550.0f) {
        const float t = clamp((wavelengthNm - 450.0f) * (1.0f / 100.0f), 0.0f, 1.0f);
        return mix(clamped.x, clamped.y, t);
    }
    const float t = clamp((wavelengthNm - 550.0f) * (1.0f / 100.0f), 0.0f, 1.0f);
    return mix(clamped.y, clamped.z, t);
}

inline bool spectral_material_uses_sampled_triplet(thread const MaterialData& material) {
    return uint(max(material.spectralParams.z, 0.0f) + 0.5f) ==
           kSpectralMaterialPolicySampledTriplet;
}

inline float spectral_value_to_sample_at(const float3 values,
                                         const float wavelengthNm,
                                         const float3 sensorRgb,
                                         const bool explicitTriplet) {
    if (explicitTriplet) {
        return spectral_triplet_to_sample(values, wavelengthNm);
    }
    return spectral_rgb_to_sample(values, sensorRgb);
}

// RGB (or an explicit 450/550/650 nm triplet) -> per-channel spectral samples. Linear in `values` for
// non-negative input, so it commutes with scalar factors (attenuation, pdfs, MIS weights).
inline float3 spectral_rgb_to_channels(const float3 values,
                                       thread const SpectralPathState& spectral,
                                       const bool explicitTriplet) {
    return float3(spectral_value_to_sample_at(values, spectral.wavelengthsNm.x, spectral.sensor0, explicitTriplet),
                  spectral_value_to_sample_at(values, spectral.wavelengthsNm.y, spectral.sensor1, explicitTriplet),
                  spectral_value_to_sample_at(values, spectral.wavelengthsNm.z, spectral.sensor2, explicitTriplet));
}

// Light radiance entering the path (environment, background, light samples): RGB policy.
inline float3 spectral_radiance_to_channels(const float3 rgb, thread const SpectralPathState& spectral) {
    return spectral.enabled ? spectral_rgb_to_channels(rgb, spectral, false) : rgb;
}

// 1 for channels that carry a wavelength, 0 for inactive ones.
inline float3 spectral_channel_mask(thread const SpectralPathState& spectral) {
    return float3(1.0f,
                  spectral.channelCount > 1u ? 1.0f : 0.0f,
                  spectral.channelCount > 2u ? 1.0f : 0.0f);
}

// Per-channel spectral radiance -> RGB: W * sum_j L_j s(lambda_j) / C over active channels.
inline float3 spectral_channels_to_rgb(const float3 channels,
                                       thread const SpectralPathState& spectral,
                                       constant PathtraceUniforms& uniforms) {
    const float3 mask = spectral_channel_mask(spectral);
    const float3 sensorSum = channels.x * spectral.sensor0 +
                             (channels.y * mask.y) * spectral.sensor1 +
                             (channels.z * mask.z) * spectral.sensor2;
    return spectral_white_balance(uniforms) * sensorSum / float(max(spectral.channelCount, 1u));
}

// Fills the per-channel wavelengths and sensor responses from the hero wavelength and C.
inline void spectral_fill_channels(thread SpectralPathState& state, constant PathtraceUniforms& uniforms) {
    const float2 range = spectral_wavelength_range_nm(uniforms);
    const float span = range.y - range.x;
    const float u = clamp((state.wavelengthNm - range.x) / span, 0.0f, 1.0f);
    const float invChannels = 1.0f / float(max(state.channelCount, 1u));
    state.wavelengthsNm = float3(state.wavelengthNm);
    if (state.channelCount > 1u) {
        state.wavelengthsNm.y = range.x + span * fract(u + invChannels);
    }
    if (state.channelCount > 2u) {
        state.wavelengthsNm.z = range.x + span * fract(u + 2.0f * invChannels);
    }
    state.sensor0 = spectral_sensor_rgb(state.wavelengthsNm.x);
    state.sensor1 = spectral_sensor_rgb(state.wavelengthsNm.y);
    state.sensor2 = spectral_sensor_rgb(state.wavelengthsNm.z);
}

inline SpectralPathState spectral_disabled_path_state() {
    SpectralPathState state;
    state.wavelengthNm = 0.0f;
    state.sampleIndex = 0u;
    state.wavelengthsNm = float3(0.0f);
    state.sensor0 = float3(1.0f);
    state.sensor1 = float3(1.0f);
    state.sensor2 = float3(1.0f);
    state.channelCount = 0u;
    state.enabled = false;
    state.collapsed = false;
    return state;
}

inline float spectral_hash_to_unit(const uint hashValue) {
    return (float(hashValue & 0x00ffffffu) + 0.5f) * (1.0f / 16777216.0f);
}

inline SpectralPathState spectral_make_path_state(constant PathtraceUniforms& uniforms,
                                                  thread uint& rngState,
                                                  device PathtraceStats* stats) {
    SpectralPathState state = spectral_disabled_path_state();
    state.enabled = spectral_rendering_enabled(uniforms);
    if (!state.enabled) {
        return state;
    }

    const float2 range = spectral_wavelength_range_nm(uniforms);
    const uint spectralKey = pcg_hash(rngState ^
                                      (uniforms.frameIndex * 747796405u) ^
                                      (uniforms.fixedRngSeed * 2891336453u) ^
                                      0x9E3779B9u);
    const uint stratum = spectralKey & 1023u;
    const float jitter = spectral_hash_to_unit(pcg_hash(spectralKey ^ 0x85EBCA6Bu));
    const float u = (float(stratum) + jitter) * (1.0f / 1024.0f);
    state.wavelengthNm = mix(range.x, range.y, u);
    state.sampleIndex = stratum;
    state.channelCount = spectral_channel_count(uniforms);
    spectral_fill_channels(state, uniforms);
    if (stats) {
        atomic_fetch_add_explicit(&stats->spectralWavelengthSampleCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->spectralWavelengthChannelSampleCount, state.channelCount,
                                  memory_order_relaxed);
    }
    return state;
}

// Rebuilds the per-channel state from the hero wavelength stored in the wavefront PathState.
inline SpectralPathState spectral_path_state_from_hero(constant PathtraceUniforms& uniforms,
                                                       const float heroWavelengthNm,
                                                       const uint sampleIndex) {
    SpectralPathState state = spectral_disabled_path_state();
    state.enabled = spectral_rendering_enabled(uniforms) && heroWavelengthNm > 0.0f;
    if (!state.enabled) {
        return state;
    }
    state.wavelengthNm = heroWavelengthNm;
    state.sampleIndex = sampleIndex;
    state.channelCount = spectral_channel_count(uniforms);
    spectral_fill_channels(state, uniforms);
    return state;
}

inline float spectral_cauchy_ior_offset(const float wavelengthNm, const float strength) {
    const float lambdaUm = clamp(wavelengthNm * 0.001f, 0.3f, 0.83f);
    const float greenUm = 0.55f;
    return strength * ((1.0f / (lambdaUm * lambdaUm)) - (1.0f / (greenUm * greenUm)));
}

// Converts a material's RGB quantities to per-channel spectral samples (no sensor multiply: the sensor is applied
// once at accumulation). The IOR is dispersed at the hero wavelength only.
inline void spectral_apply_material(thread MaterialData& material,
                                    constant PathtraceUniforms& uniforms,
                                    thread const SpectralPathState& spectral,
                                    device PathtraceStats* stats) {
    if (!spectral.enabled) {
        return;
    }
    if (stats) {
        atomic_fetch_add_explicit(&stats->spectralMaterialLookupCount, 1u, memory_order_relaxed);
    }

    const bool explicitTriplet = spectral_material_uses_sampled_triplet(material);
    if (!explicitTriplet && stats) {
        atomic_fetch_add_explicit(&stats->spectralFallbackCount, 1u, memory_order_relaxed);
    }

    material.baseColorRoughness.xyz =
        clamp(spectral_rgb_to_channels(material.baseColorRoughness.xyz, spectral, explicitTriplet),
              float3(0.0f),
              float3(1.0f));

    if (any(material.emission.xyz > float3(0.0f))) {
        material.emission.xyz = spectral_rgb_to_channels(material.emission.xyz, spectral, explicitTriplet);
    }

    if (any(material.dielectricSigmaA.xyz > float3(0.0f))) {
        const float absorptionScale = max(material.spectralParams.y, 0.0f);
        material.dielectricSigmaA.xyz =
            spectral_rgb_to_channels(material.dielectricSigmaA.xyz * absorptionScale, spectral, explicitTriplet);
    }

    if (material.conductorEta.w > 0.0f) {
        material.conductorEta.xyz =
            max(spectral_rgb_to_channels(material.conductorEta.xyz, spectral, explicitTriplet), float3(0.0f));
        material.conductorK.xyz =
            max(spectral_rgb_to_channels(material.conductorK.xyz, spectral, explicitTriplet), float3(0.0f));
    }

    const MaterialClosureMetadata closure = material_primary_closure_metadata(material);
    if (material_closure_has_flag(closure, kMaterialClosureFlagTransmissive) ||
        material.pbrExtras.z > 0.0f) {
        const float dispersionScale =
            max(material.spectralParams.x, uniforms.spectralWavelengthRange.z);
        const float etaOffset = spectral_cauchy_ior_offset(spectral.wavelengthNm, dispersionScale);
        material.typeEta.y = clamp(material.typeEta.y + etaOffset, 1.0f, 3.0f);
        if (stats) {
            atomic_fetch_add_explicit(&stats->spectralDispersionEventCount, 1u, memory_order_relaxed);
        }
    }
}

// True when spectral_apply_material makes this vertex's IOR wavelength dependent (same predicate and strength).
inline bool spectral_vertex_is_dispersive(thread const MaterialData& material,
                                          constant PathtraceUniforms& uniforms) {
    const MaterialClosureMetadata closure = material_primary_closure_metadata(material);
    const bool transmissive = material_closure_has_flag(closure, kMaterialClosureFlagTransmissive) ||
                              material.pbrExtras.z > 0.0f;
    return transmissive && max(material.spectralParams.x, uniforms.spectralWavelengthRange.z) > 0.0f;
}

// At a dispersive vertex the sampled direction is valid for the hero wavelength only. Keep the hero channel,
// scaled by C to undo the 1/C averaging at accumulation (the hero alone is an unbiased uniform-wavelength
// estimator), and drop the secondary channels. Contributions added before the collapse keep all channels.
inline void spectral_collapse_at_dispersive_vertex(thread SpectralPathState& spectral,
                                                   thread const MaterialData& material,
                                                   constant PathtraceUniforms& uniforms,
                                                   thread float3& throughput,
                                                   device PathtraceStats* stats) {
    if (!spectral.enabled || spectral.collapsed || !spectral_vertex_is_dispersive(material, uniforms)) {
        return;
    }
    throughput = float3(throughput.x * float(spectral.channelCount), 0.0f, 0.0f);
    spectral.collapsed = true;
    if (stats) {
        atomic_fetch_add_explicit(&stats->spectralCollapseCount, 1u, memory_order_relaxed);
    }
}
