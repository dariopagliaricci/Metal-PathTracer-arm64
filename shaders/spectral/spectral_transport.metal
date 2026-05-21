constant uint kSpectralModeHeroWavelength = 1u;
constant uint kSpectralTexturePolicyRGBReflectanceD65 = 0u;

struct SpectralPathState {
    float wavelengthNm;
    uint sampleIndex;
    float3 sensorRgb;
    bool enabled;
};

inline bool spectral_rendering_enabled(constant PathtraceUniforms& uniforms) {
    return uniforms.spectralRenderingMode == kSpectralModeHeroWavelength &&
           uniforms.spectralWavelengthRange.y > uniforms.spectralWavelengthRange.x;
}

inline float spectral_gaussian(const float wavelengthNm, const float centerNm, const float widthNm) {
    const float x = (wavelengthNm - centerNm) / max(widthNm, 1.0f);
    return exp(-0.5f * x * x);
}

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

inline float spectral_hash_to_unit(const uint hashValue) {
    return (float(hashValue & 0x00ffffffu) + 0.5f) * (1.0f / 16777216.0f);
}

inline SpectralPathState spectral_make_path_state(constant PathtraceUniforms& uniforms,
                                                  thread uint& rngState,
                                                  device PathtraceStats* stats) {
    SpectralPathState state;
    state.enabled = spectral_rendering_enabled(uniforms);
    state.sampleIndex = 0u;
    state.wavelengthNm = 0.0f;
    state.sensorRgb = float3(1.0f);
    if (!state.enabled) {
        return state;
    }

    const float minNm = max(min(uniforms.spectralWavelengthRange.x,
                                uniforms.spectralWavelengthRange.y - 1.0f),
                            300.0f);
    const float maxNm = min(max(uniforms.spectralWavelengthRange.y, minNm + 1.0f), 830.0f);
    const uint spectralKey = pcg_hash(rngState ^
                                      (uniforms.frameIndex * 747796405u) ^
                                      (uniforms.fixedRngSeed * 2891336453u) ^
                                      0x9E3779B9u);
    const uint stratum = spectralKey & 1023u;
    const float jitter = spectral_hash_to_unit(pcg_hash(spectralKey ^ 0x85EBCA6Bu));
    const float u = (float(stratum) + jitter) * (1.0f / 1024.0f);
    state.wavelengthNm = mix(minNm, maxNm, u);
    state.sampleIndex = stratum;
    state.sensorRgb = spectral_sensor_rgb(state.wavelengthNm);
    if (stats) {
        atomic_fetch_add_explicit(&stats->spectralWavelengthSampleCount, 1u, memory_order_relaxed);
    }
    return state;
}

inline float spectral_cauchy_ior_offset(const float wavelengthNm, const float strength) {
    const float lambdaUm = clamp(wavelengthNm * 0.001f, 0.3f, 0.83f);
    const float greenUm = 0.55f;
    return strength * ((1.0f / (lambdaUm * lambdaUm)) - (1.0f / (greenUm * greenUm)));
}

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

    const float3 sensor = spectral.sensorRgb;
    const float3 baseColor = material.baseColorRoughness.xyz;
    const float baseSample = spectral_rgb_to_sample(baseColor, sensor);
    material.baseColorRoughness.xyz = clamp(baseSample * sensor, float3(0.0f), float3(1.0f));

    if (any(material.emission.xyz > float3(0.0f))) {
        const float emissionSample = spectral_rgb_to_sample(material.emission.xyz, sensor);
        material.emission.xyz = emissionSample * sensor;
    }

    if (any(material.dielectricSigmaA.xyz > float3(0.0f))) {
        const float absorptionScale = max(material.spectralParams.y, 0.0f);
        const float absorptionSample =
            spectral_rgb_to_sample(material.dielectricSigmaA.xyz * absorptionScale, sensor);
        material.dielectricSigmaA.xyz = absorptionSample * sensor;
    }

    if (material.conductorEta.w > 0.0f) {
        const float etaSample = spectral_rgb_to_sample(material.conductorEta.xyz, sensor);
        const float kSample = spectral_rgb_to_sample(material.conductorK.xyz, sensor);
        material.conductorEta.xyz = max(etaSample, 0.0f) * sensor;
        material.conductorK.xyz = max(kSample, 0.0f) * sensor;
    }

    const uint materialType = uint(material.typeEta.x + 0.5f);
    if (materialType == 2u || material.pbrExtras.z > 0.0f) {
        const float dispersionScale =
            max(material.spectralParams.x, uniforms.spectralWavelengthRange.z);
        const float etaOffset = spectral_cauchy_ior_offset(spectral.wavelengthNm, dispersionScale);
        material.typeEta.y = clamp(material.typeEta.y + etaOffset, 1.0f, 3.0f);
        if (stats) {
            atomic_fetch_add_explicit(&stats->spectralDispersionEventCount, 1u, memory_order_relaxed);
        }
    }
}
