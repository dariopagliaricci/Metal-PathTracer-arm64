inline float luminance_rgb(const float3 color) {
    return dot(color, kLuminanceWeights);
}

inline void stats_add_mnee_luma(device PathtraceStats* stats, const float3 contribution) {
    if (!stats) {
        return;
    }
    float luma = luminance_rgb(contribution);
    if (!isfinite(luma) || luma <= 0.0f) {
        return;
    }
    constexpr float kMneeLumaScale = 1024.0f;
    float scaled = luma * kMneeLumaScale;
    uint add = static_cast<uint>(min(scaled, 4294967295.0f));
    uint prev = atomic_fetch_add_explicit(&stats->mneeContributionLumaSumLo,
                                          add,
                                          memory_order_relaxed);
    if (prev + add < prev) {
        atomic_fetch_add_explicit(&stats->mneeContributionLumaSumHi, 1u, memory_order_relaxed);
    }
    atomic_fetch_add_explicit(&stats->mneeContributionCount, 1u, memory_order_relaxed);
}

struct FireflyClampParams {
    uint mode;
    float clampFactor;
    float clampFloor;
    float throughputClamp;
    float minSpecularPdf;
    float maxContribution;
    float enabled;
    uint spectralChannels; // 0 = RGB; otherwise wavelengths per path in spectral transport (1 or 3)
};

inline float plastic_coat_roughness(const MaterialData material);
inline float plastic_coat_f0(const MaterialData material);
inline float3 plastic_specular_tint(const MaterialData material);
inline float3 plastic_diffuse_transmission(const MaterialData material,
                                           const float cosThetaI,
                                           const float cosThetaO);
inline bool sss_use_separable(const uint sssMode, const MaterialData material);
inline bool sss_use_random_walk(const uint sssMode, const MaterialData material);
inline float clamp_specular_pdf(const float pdf, const FireflyClampParams params);
inline float3 sanitize_specular_value(const float3 value);
inline float ggx_D(const float alpha, const float cosThetaH);
inline float ggx_D(const float alpha, const float3 normal, const float3 wh);
inline float schlick_weight(const float cosTheta);
inline float ggx_G1(const float alpha, const float cosTheta);
inline float ggx_pdf(const float alpha,
                     const float3 normal,
                     const float3 wo,
                     const float3 wi,
                     const bool halfVectorD = false);
inline float3 schlick_fresnel(const float3 f0, const float cosTheta);
inline float3 material_base_color(const MaterialData material);
inline float lambert_pdf(const float3 normal, const float3 direction);
inline float3 dielectric_sigma_a(const MaterialData material) {
    return material.dielectricSigmaA.xyz;
}

// Type-7 BTDF tint for one interface crossing. KHR_materials_transmission tints
// transmitted light by the resolved base color (specular_btdf * baseColor);
// KHR_materials_volume adds absorption over its thickness. Reflection is untinted.
inline float3 transmission_tint(const MaterialData material, const float cosTheta) {
    const float3 surfaceTint = clamp01(material.baseColorRoughness.xyz);
    float thickness = max(material.typeEta.w, 0.0f);
    if (thickness <= 0.0f) {
        return surfaceTint;
    }
    float3 sigmaA = max(dielectric_sigma_a(material), float3(0.0f));
    if (all(sigmaA <= float3(0.0f))) {
        return surfaceTint;
    }
    float distance = thickness / max(fabs(cosTheta), 1.0e-3f);
    return surfaceTint * clamp01(exp(-sigmaA * distance));
}

inline float carpaint_base_metallic(const MaterialData material) {
    return clamp(material.carpaintBaseParams.x, 0.0f, 1.0f);
}

inline float carpaint_base_roughness(const MaterialData material) {
    return clamp(material.carpaintBaseParams.y, 0.0f, 1.0f);
}

inline float carpaint_flake_scale(const MaterialData material) {
    return max(material.carpaintBaseParams.z, 1.0e-4f);
}

// Effective flake coverage of the under-layer (see the CarPaint closure below).
inline float carpaint_flake_coverage(const MaterialData material) {
    return clamp(material.carpaintFlakeParams.x, 0.0f, 1.0f);
}

inline float carpaint_flake_roughness(const MaterialData material) {
    return clamp(material.carpaintFlakeParams.y, 0.0f, 1.0f);
}

inline float carpaint_flake_anisotropy(const MaterialData material) {
    return clamp(material.carpaintFlakeParams.z, -0.99f, 0.99f);
}

inline float carpaint_flake_normal_strength(const MaterialData material) {
    return clamp(material.carpaintFlakeParams.w, 0.0f, 1.0f);
}

inline float carpaint_coat_sample_weight(const MaterialData material) {
    return clamp(material.coatParams.z, 0.0f, 0.95f);
}

inline bool carpaint_has_base_conductor(const MaterialData material) {
    return (material.carpaintBaseEta.w > 0.0f || material.carpaintBaseK.w > 0.0f);
}

inline float3 carpaint_base_eta(const MaterialData material) {
    return max(material.carpaintBaseEta.xyz, float3(0.0f));
}

inline float3 carpaint_base_k(const MaterialData material) {
    return max(material.carpaintBaseK.xyz, float3(0.0f));
}

inline float3 fresnel_conductor(float cosThetaI, const float3 eta, const float3 k);

inline float3 carpaint_base_f0(const MaterialData material) {
    if (carpaint_has_base_conductor(material)) {
        return fresnel_conductor(1.0f, carpaint_base_eta(material), carpaint_base_k(material));
    }
    return clamp01(material.baseColorRoughness.xyz);
}

inline float3 carpaint_hash3(float3 p) {
    p = fract(p * 0.3183099f + float3(0.1f, 0.3f, 0.7f));
    p += dot(p, float3(p.y + 33.33f, p.z + 55.55f, p.x + 77.77f));
    return fract((p.xxy + p.yzz) * 13.5453123f);
}

inline float3 carpaint_flake_normal(const MaterialData material,
                                    const float3 position,
                                    const float3 normal) {
    float scale = carpaint_flake_scale(material);
    float3 samplePos = position * scale;
    float3 rand = carpaint_hash3(samplePos);
    float anis = carpaint_flake_anisotropy(material);
    float ax = max(1.0f - anis, 1.0e-3f);
    float ay = max(1.0f + anis, 1.0e-3f);
    float phi = 2.0f * kPi * rand.x;
    float r = sqrt(max(rand.y, 1.0e-4f));
    float x = r * cos(phi) * ax;
    float y = r * sin(phi) * ay;
    float m2 = clamp(x * x + y * y, 0.0f, 0.99f);
    float z = sqrt(max(1.0f - m2, 0.0f));
    float3 tangent;
    float3 bitangent;
    build_onb(normal, tangent, bitangent);
    float3 perturbed = normalize(x * tangent + y * bitangent + z * normal);
    float strength = carpaint_flake_normal_strength(material);
    return normalize(mix(normal, perturbed, strength));
}

// A closure (or one of its lobes) evaluated for a direction pair: its value f and the pdf its sampler draws wi with.
struct ClosureValuePdf {
    float3 value;
    float pdf;
};

inline float ggx_vndf_alpha(const float roughness) {
    return max(roughness * roughness, 1.0e-4f);  // sample_ggx_vndf's alpha
}

// GGX reflection lobe around lobeNormal. value: F D G2 / (4 cos_o cos_m(wi)), cos_m the cosine the integrator
// multiplies by (the macro normal; the lobe normal itself except for flakes). pdf: the VNDF reflection density for
// any reflected direction, also one below the lobe plane (sample_ggx_vndf can produce it; its value is zero).
inline ClosureValuePdf ggx_reflection_lobe(const float alpha,
                                           const float3 lobeNormal,
                                           const float cosThetaMacroI,
                                           const float3 wo,
                                           const float3 wi,
                                           const float3 F,
                                           const FireflyClampParams clampParams) {
    ClosureValuePdf res;
    res.value = float3(0.0f);
    res.pdf = 0.0f;
    float cosThetaO = dot(lobeNormal, wo);
    float3 wh = safe_normalize(wo + wi);
    float cosThetaH = dot(lobeNormal, wh);
    if (cosThetaO <= 0.0f || cosThetaH <= 0.0f || dot(wo, wh) <= 0.0f) {
        return res;
    }
    float D = ggx_D(alpha, lobeNormal, wh);
    float G1o = ggx_G1(alpha, cosThetaO);
    res.pdf = clamp_specular_pdf(D * G1o / (4.0f * cosThetaO), clampParams);
    float cosThetaI = dot(lobeNormal, wi);
    if (cosThetaI > 0.0f && cosThetaMacroI > 0.0f) {
        float G = G1o * ggx_G1(alpha, cosThetaI);
        res.value = sanitize_specular_value(F * (D * G / (4.0f * cosThetaO * cosThetaMacroI)));
    }
    return res;
}

// CarPaint (type 6): a clear coat over a flake/base layer, evaluated as one layered BSDF:
//   f = f_coat + T(wo, wi) * [c * f_flake + (1 - c) * ((1 - m) * baseColor / pi + m * f_metal)]
//   T = (1 - Fc(cos_o)) * (1 - Fc(cos_i)) * coat tint/absorption (plastic_diffuse_transmission)
// f_coat is the coat's GGX lobe with its own Schlick Fresnel Fc (coat IOR, typeEta.z), untinted: it reflects off
// the top surface. Light reaching the under-layer crosses the coat interface twice, (1 - Fc) each way at the
// external angles, so every view direction keeps albedo <= 1 (the GGX coat albedo stays below Fc(cos_o)).
// c (carpaintFlakeParams.x) is the effective flake coverage: the fraction of the under-layer covered by flakes
// (density-derived coverage x flakeReflectanceScale); the base fills the rest. m is the base metallic.
// A flake is a tilted micro-mirror patch: its GGX lobe lives in the flake frame and its value is defined against
// the macro cosine, f_flake * cos_i = F D G2 / (4 cos_o_flake), so a mirror flake reflects F * L at any tilt.
// Sampling is one-sample MIS over {coat, flake, diffuse, metal}: the probabilities (albedo estimates at wo) only
// choose the lobe; the sample is weighted by the full f * cos / pdf, pdf = sum_i p_i pdf_i, and evaluation
// returns the same f and pdf. GGX lobe pdfs are the VNDF reflection density D G1(wo) / (4 cos_o) that
// sample_ggx_vndf draws from, with the same alpha floor.
struct CarpaintLobeProbabilities {
    float coat;
    float flake;
    float diffuse;
    float metal;
};

inline float carpaint_coat_roughness(const MaterialData material) {
    return plastic_coat_roughness(material);
}

inline float carpaint_flake_lobe_roughness(const MaterialData material) {
    return max(carpaint_flake_roughness(material), 1.0e-3f);
}

inline float carpaint_metal_roughness(const MaterialData material) {
    return max(carpaint_base_roughness(material), 1.0e-3f);
}

inline float carpaint_coat_f0(const MaterialData material) {
    float eta = max(material.typeEta.z, 1.0f);
    float ratio = (eta - 1.0f) / (eta + 1.0f);
    return ratio * ratio;
}

inline float carpaint_coat_fresnel(const MaterialData material, const float cosTheta) {
    float f0 = carpaint_coat_f0(material);
    return f0 + (1.0f - f0) * schlick_weight(cosTheta);
}

// Hemispherical average of a Schlick Fresnel with normal reflectance f0.
inline float3 carpaint_average_fresnel(const float3 f0) {
    return f0 + (float3(1.0f) - f0) * (1.0f / 21.0f);
}

inline CarpaintLobeProbabilities carpaint_lobe_probabilities(const MaterialData material,
                                                             const float3 normal,
                                                             const float3 flakeNormal,
                                                             const float3 wo) {
    CarpaintLobeProbabilities probs;
    probs.coat = 0.0f;
    probs.flake = 0.0f;
    probs.diffuse = 0.0f;
    probs.metal = 0.0f;
    float cosThetaO = dot(normal, wo);
    if (cosThetaO <= 0.0f) {
        return probs;
    }
    float coverage = carpaint_flake_coverage(material);
    float metallic = carpaint_base_metallic(material);
    float coatReflect = carpaint_coat_fresnel(material, cosThetaO);
    float transmitted = 1.0f - coatReflect;
    float underF = luminance_rgb(carpaint_average_fresnel(carpaint_base_f0(material)));
    probs.coat = coatReflect;
    // The flake lobe can only be drawn when wo sees the flake's front side.
    probs.flake = (dot(flakeNormal, wo) > 0.0f) ? transmitted * coverage * underF : 0.0f;
    probs.diffuse = transmitted * (1.0f - coverage) * (1.0f - metallic) *
                    luminance_rgb(material_base_color(material));
    probs.metal = transmitted * (1.0f - coverage) * metallic * underF;
    float sum = probs.coat + probs.flake + probs.diffuse + probs.metal;
    if (!(sum > 0.0f)) {
        probs.coat = 0.0f;
        probs.flake = 0.0f;
        probs.diffuse = 0.0f;
        probs.metal = 0.0f;
        return probs;
    }
    float inv = 1.0f / sum;
    probs.coat *= inv;
    probs.flake *= inv;
    probs.diffuse *= inv;
    probs.metal *= inv;
    return probs;
}

inline ClosureValuePdf carpaint_closure_eval(const MaterialData material,
                                             const float3 normal,
                                             const float3 flakeNormal,
                                             const float3 wo,
                                             const float3 wi,
                                             const CarpaintLobeProbabilities probs,
                                             const FireflyClampParams clampParams) {
    ClosureValuePdf res;
    res.value = float3(0.0f);
    res.pdf = 0.0f;
    float cosThetaO = dot(normal, wo);
    float cosThetaI = dot(normal, wi);
    if (cosThetaO <= 0.0f || cosThetaI <= 0.0f) {
        return res;
    }
    float3 wh = safe_normalize(wo + wi);
    float coverage = carpaint_flake_coverage(material);
    float metallic = carpaint_base_metallic(material);
    float3 baseColor = material_base_color(material);

    ClosureValuePdf coat = ggx_reflection_lobe(ggx_vndf_alpha(carpaint_coat_roughness(material)),
                                               normal, cosThetaI, wo, wi,
                                               float3(carpaint_coat_fresnel(material, dot(wi, wh))),
                                               clampParams);
    ClosureValuePdf flake = ggx_reflection_lobe(ggx_vndf_alpha(carpaint_flake_lobe_roughness(material)),
                                                flakeNormal, cosThetaI, wo, wi,
                                                schlick_fresnel(carpaint_base_f0(material), dot(wi, wh)),
                                                clampParams);
    float3 metalF = carpaint_has_base_conductor(material)
        ? fresnel_conductor(dot(wi, wh), carpaint_base_eta(material), carpaint_base_k(material))
        : schlick_fresnel(baseColor, dot(wi, wh));
    ClosureValuePdf metal = ggx_reflection_lobe(ggx_vndf_alpha(carpaint_metal_roughness(material)),
                                                normal, cosThetaI, wo, wi, metalF, clampParams);
    float3 diffuse = baseColor * (1.0f / kPi);

    float3 coatTransmission = (1.0f - carpaint_coat_fresnel(material, cosThetaO)) *
                              (1.0f - carpaint_coat_fresnel(material, cosThetaI)) *
                              plastic_diffuse_transmission(material, cosThetaI, cosThetaO);
    float3 under = coverage * flake.value +
                   (1.0f - coverage) * ((1.0f - metallic) * diffuse + metallic * metal.value);
    res.value = max(coat.value + coatTransmission * under, float3(0.0f));
    res.pdf = probs.coat * coat.pdf + probs.flake * flake.pdf +
              probs.diffuse * lambert_pdf(normal, wi) + probs.metal * metal.pdf;
    return res;
}

inline FireflyClampParams make_firefly_params(constant PathtraceUniforms& uniforms) {
    FireflyClampParams params;
    params.mode = uniforms.fireflyClampMode;
    params.clampFactor = max(uniforms.fireflyClampFactor, 0.0f);
    params.clampFloor = max(uniforms.fireflyClampFloor, 0.0f);
    params.throughputClamp = max(uniforms.throughputClamp, 0.0f);
    params.minSpecularPdf = max(uniforms.minSpecularPdf, 0.0f);
    params.maxContribution = max(uniforms.fireflyClampMaxContribution, 0.0f);
    params.enabled = (uniforms.fireflyClampEnabled != 0u) ? 1.0f : 0.0f;
    // spectralRenderingMode 1 = three wavelengths per path, 2 = one (spectral_transport.metal, included later).
    params.spectralChannels = uniforms.spectralWavelengthRange.y > uniforms.spectralWavelengthRange.x
        ? (PT_DEFAULT_OFF(uniforms.spectralRenderingMode) == 1u ? 3u : (PT_DEFAULT_OFF(uniforms.spectralRenderingMode) == 2u ? 1u : 0u))
        : 0u;
    return params;
}

// Brightness used by the luminance-mode clamps. RGB: Rec.709 luminance. Spectral transport: the mean over the
// wavelength channels, so the clamp does not depend on which channel carries which wavelength.
inline float clamp_brightness(const float3 value, const FireflyClampParams params) {
    if (params.spectralChannels == 0u) {
        return luminance_rgb(value);
    }
    return params.spectralChannels == 1u ? value.x : (value.x + value.y + value.z) * (1.0f / 3.0f);
}

inline float3 clamp_firefly_contribution(const float3 throughput,
                                         const float3 contribution,
                                         const FireflyClampParams params,
                                         device PathtraceStats* stats) {
    float3 combined = throughput * contribution;
    if (!all(isfinite(combined))) {
        if (stats) {
            atomic_fetch_add_explicit(&stats->radianceNanInfCount, 1u, memory_order_relaxed);
        }
        return float3(0.0f);
    }

    float3 positive = max(combined, float3(0.0f));
    if (params.enabled < 0.5f) {
        return positive;
    }

    if (params.mode == 1u) {
        float maxComponent = max(positive.x, max(positive.y, positive.z));
        float maxAllowed = params.maxContribution;
        if (maxAllowed <= 0.0f) {
            float throughputMax = max(max(throughput.x, throughput.y), throughput.z);
            maxAllowed = max(throughputMax * params.clampFactor, params.clampFloor);
        }
        if (maxComponent > maxAllowed && maxComponent > 0.0f) {
            float scale = maxAllowed / max(maxComponent, 1e-6f);
            combined *= scale;
            positive = max(combined, float3(0.0f));
            if (stats) {
                atomic_fetch_add_explicit(&stats->clampEventCount, 1u, memory_order_relaxed);
            }
        }
    } else {
        float lum = clamp_brightness(positive, params);
        float throughputLum = clamp_brightness(max(throughput, float3(0.0f)), params);
        float maxLum = max(throughputLum * params.clampFactor, params.clampFloor);
        if (params.maxContribution > 0.0f) {
            maxLum = max(maxLum, params.maxContribution);
        }

        if (lum > maxLum && lum > 0.0f) {
            float scale = maxLum / max(lum, 1e-6f);
            combined *= scale;
            positive = max(combined, float3(0.0f));
            if (stats) {
                atomic_fetch_add_explicit(&stats->clampEventCount, 1u, memory_order_relaxed);
            }
        }
    }

    return positive;
}

inline float3 clamp_firefly_contribution(const float3 throughput,
                                         const float3 contribution,
                                         const FireflyClampParams params) {
    return clamp_firefly_contribution(throughput, contribution, params, nullptr);
}

inline float clamp_specular_pdf(const float pdf, const FireflyClampParams params) {
    if (!isfinite(pdf)) {
        return 0.0f;
    }
    if (pdf <= 0.0f) {
        return 0.0f;
    }
    if (params.minSpecularPdf <= 0.0f) {
        return pdf;
    }
    return max(pdf, params.minSpecularPdf);
}

inline float3 clamp_path_throughput(const float3 throughput,
                                    const FireflyClampParams params,
                                    device PathtraceStats* stats) {
    if (!all(isfinite(throughput))) {
        if (stats) {
            atomic_fetch_add_explicit(&stats->throughputNanInfCount, 1u, memory_order_relaxed);
        }
        return float3(0.0f);
    }
    if (params.enabled < 0.5f || params.throughputClamp <= 0.0f) {
        return throughput;
    }
    float3 positive = max(throughput, float3(0.0f));
    float lum = clamp_brightness(positive, params);
    if (lum > params.throughputClamp && lum > 0.0f) {
        float scale = params.throughputClamp / max(lum, 1e-6f);
        if (stats) {
            atomic_fetch_add_explicit(&stats->clampEventCount, 1u, memory_order_relaxed);
        }
        return throughput * scale;
    }
    return throughput;
}

inline float3 clamp_path_throughput(const float3 throughput, const FireflyClampParams params) {
    return clamp_path_throughput(throughput, params, nullptr);
}

// Sanitizes a GGX lobe value (non-finite or negative -> 0). It never caps the
// value: scaling f while the lobe pdf stays unchanged removed most of a sharp
// lobe's energy (f ~1e4 at r=0.02 against a cap of ~0.55) without bounding any
// estimator weight; VNDF sampling and MIS already keep f*cos/pdf bounded.
// Outliers are handled by clamp_path_throughput and clamp_firefly_contribution.
inline float3 sanitize_specular_value(const float3 value) {
    if (!all(isfinite(value))) {
        return float3(0.0f);
    }
    return max(value, float3(0.0f));
}

inline float schlick_weight(const float cosTheta) {
    float m = clamp01(1.0f - cosTheta);
    float m2 = m * m;
    return m2 * m2 * m;
}

inline float3 schlick_fresnel(const float3 f0, const float cosTheta) {
    return f0 + (float3(1.0f) - f0) * schlick_weight(cosTheta);
}

#include "../shared/closure_common.h"

inline float ggx_lambda(const float alpha, const float cosTheta) {
    float absCosTheta = fabs(cosTheta);
    if (absCosTheta <= 0.0f) {
        return 0.0f;
    }
    float sinTheta = safe_sqrt(max(0.0f, 1.0f - absCosTheta * absCosTheta));
    if (sinTheta == 0.0f) {
        return 0.0f;
    }
    float tanTheta = sinTheta / absCosTheta;
    float a = alpha * tanTheta;
    return (-1.0f + sqrt(1.0f + a * a)) * 0.5f;
}

inline float ggx_G1(const float alpha, const float cosTheta) {
    return 1.0f / (1.0f + ggx_lambda(alpha, cosTheta));
}

inline float ggx_D(const float alpha, const float cosThetaH) {
    float absCosThetaH = fabs(cosThetaH);
    float a2 = alpha * alpha;
    float denom = absCosThetaH * absCosThetaH * (a2 - 1.0f) + 1.0f;
    return a2 / (kPi * denom * denom);
}

// The same D, taking sin^2(theta_h) from the half vector. In float32 the
// cosine form cannot resolve lobes near the PBR alpha floor (1e-4):
// cos(theta_h) rounds to 1 below theta_h = 2.4e-4 rad, and a2 - 1 rounds to
// -1 below alpha = 1.7e-4, so D steps between a few values or is inf there.
inline float ggx_D(const float alpha, const float3 normal, const float3 wh) {
    float cosThetaH = dot(normal, wh);
    float sin2ThetaH = length_squared(cross(normal, wh));
    float a2 = alpha * alpha;
    float denom = sin2ThetaH + a2 * cosThetaH * cosThetaH;
    return (denom > 0.0f) ? a2 / (kPi * denom * denom) : 0.0f;
}

inline float ggx_pdf(const float alpha,
                     const float3 normal,
                     const float3 wo,
                     const float3 wi,
                     const bool halfVectorD) {
    float3 wh = safe_normalize(wo + wi);
    float cosThetaH = dot(normal, wh);
    float dotWoWh = dot(wo, wh);
    float cosThetaO = dot(normal, wo);
    if (cosThetaO <= 0.0f || cosThetaH <= 0.0f || dotWoWh <= 0.0f) {
        return 0.0f;
    }
    float D = halfVectorD ? ggx_D(alpha, normal, wh) : ggx_D(alpha, cosThetaH);
    float G1 = ggx_G1(alpha, cosThetaO);
    float denom = 4.0f * max(dotWoWh, 1e-6f);
    return D * G1 * cosThetaH / denom;
}

inline float ggx_vndf_pdf(const float alpha,
                          const float3 normal,
                          const float3 wo,
                          const float3 wh,
                          const bool halfVectorD = false) {
    float cosThetaO = dot(normal, wo);
    float cosThetaH = dot(normal, wh);
    float woDotWh = dot(wo, wh);
    if (cosThetaO <= 0.0f || cosThetaH <= 0.0f || woDotWh <= 0.0f) {
        return 0.0f;
    }
    // Density of the visible-normal half vector: D * G1(wo) * (wo.wh) / cos(theta_o).
    float D = halfVectorD ? ggx_D(alpha, normal, wh) : ggx_D(alpha, cosThetaH);
    float G1 = ggx_G1(alpha, cosThetaO);
    return D * G1 * woDotWh / cosThetaO;
}

inline float3 sample_ggx_vndf(const float3 normal,
                              const float3 wo,
                              const float roughness,
                              thread uint& state) {
    float3 woLocal = to_local(safe_normalize(wo), normal);
    woLocal.z = max(woLocal.z, 1.0e-6f);
    float alpha = max(roughness * roughness, 1.0e-4f);
    float3 Vh = safe_normalize(float3(alpha * woLocal.x, alpha * woLocal.y, woLocal.z));

    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = (lensq > 0.0f) ? float3(-Vh.y, Vh.x, 0.0f) * rsqrt(lensq)
                               : float3(1.0f, 0.0f, 0.0f);
    float3 T2 = cross(Vh, T1);

    float u1 = rand_uniform(state);
    float u2 = rand_uniform(state);
    float r = sqrt(u1);
    float phi = 2.0f * kPi * u2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5f * (1.0f + Vh.z);
    float t2Adjusted = (1.0f - s) * safe_sqrt(max(0.0f, 1.0f - t1 * t1)) + s * t2;
    float t3 = safe_sqrt(max(0.0f, 1.0f - t1 * t1 - t2Adjusted * t2Adjusted));

    float3 Nh = t1 * T1 + t2Adjusted * T2 + t3 * Vh;
    float3 Ne = safe_normalize(float3(alpha * Nh.x, alpha * Nh.y, max(Nh.z, 0.0f)));
    return safe_normalize(to_world(Ne, normal));
}

inline bool material_is_plastic(const MaterialData material) {
    return static_cast<uint>(material.typeEta.x) == 4u;
}

inline float plastic_coat_ior(const MaterialData material) {
    return max(material.typeEta.y, 1.0f);
}

inline float plastic_coat_roughness(const MaterialData material) {
    float roughness = clamp(material.coatParams.x, 0.0f, 1.0f);
    return max(roughness, 1.0e-3f);
}

inline float plastic_coat_thickness(const MaterialData material) {
    return max(material.coatParams.y, 0.0f);
}

inline float plastic_coat_fresnel_average(const MaterialData material) {
    return clamp(material.coatParams.w, 0.0f, 1.0f);
}

inline float plastic_coat_f0(const MaterialData material) {
    float eta = plastic_coat_ior(material);
    float ratio = (eta - 1.0f) / max(eta + 1.0f, 1.0e-6f);
    float f0 = ratio * ratio;
    return clamp(f0, 0.0f, 0.999f);
}

inline float ior_from_f0(float f0) {
    float clamped = clamp(f0, 0.0f, 0.999f);
    float root = sqrt(clamped);
    float denom = max(1.0f - root, 1.0e-4f);
    return max((1.0f + root) / denom, 1.0f);
}

inline float3 plastic_coat_tint(const MaterialData material) {
    return clamp01(material.coatTint.xyz);
}

inline float3 plastic_coat_absorption(const MaterialData material) {
    return max(material.coatAbsorption.xyz, float3(0.0f));
}

inline float3 plastic_specular_tint(const MaterialData material) {
    float3 tint = plastic_coat_tint(material);
    float thickness = plastic_coat_thickness(material);
    if (thickness <= 0.0f) {
        return tint;
    }
    float3 absorption = plastic_coat_absorption(material);
    if (all(absorption <= float3(1.0e-6f))) {
        return tint;
    }
    return clamp01(tint * exp(-absorption * thickness));
}

inline float3 plastic_diffuse_transmission(const MaterialData material,
                                           const float cosThetaI,
                                           const float cosThetaO) {
    float thickness = plastic_coat_thickness(material);
    float3 tint = plastic_coat_tint(material);
    if (thickness <= 0.0f) {
        return tint;
    }
    float3 absorption = plastic_coat_absorption(material);
    float safeCosI = max(cosThetaI, 1.0e-3f);
    float safeCosO = max(cosThetaO, 1.0e-3f);
    float3 attenuationI = exp(-absorption * (thickness / safeCosI));
    float3 attenuationO = exp(-absorption * (thickness / safeCosO));
    return clamp01(tint * attenuationI * attenuationO);
}

// Plastic (type 4): a diffuse base under a dielectric clear coat, evaluated as one BSDF:
//   f = f_coat + (1 - F(cos_i)) * (1 - F(cos_o)) * (1 - Favg) * T(wi, wo) * occlusion * baseColor / pi
// f_coat is the coat's GGX lobe with Schlick F (plastic_coat_f0) times the coat tint/absorption
// (plastic_specular_tint); T is the tint/absorption along both coat crossings (plastic_diffuse_transmission) and
// Favg the coat's hemispherical Fresnel average (coatParams.w).
// Sampling is one-sample MIS over {coat, diffuse}: the probabilities (albedo estimates at wo) only choose the lobe;
// the sample is weighted by the full f * cos / pdf, pdf = p_coat * pdf_coat + p_diffuse * cos_i / pi, and
// evaluation returns the same f and pdf. The coat pdf is the VNDF reflection density sample_ggx_vndf draws from.
struct PlasticLobeProbabilities {
    float coat;
    float diffuse;
};

// Absorption along one crossing of the coat at cosine `cosTheta` (the factor plastic_diffuse_transmission applies per
// crossing).
inline float3 plastic_coat_crossing(const MaterialData material, const float cosTheta) {
    float thickness = plastic_coat_thickness(material);
    if (thickness <= 0.0f) {
        return float3(1.0f);
    }
    return exp(-plastic_coat_absorption(material) * (thickness / max(cosTheta, 1.0e-3f)));
}

// The coat's exact dielectric Fresnel (unpolarized, from air into coatIOR). A coated random-walk subsurface material uses
// it for its coat lobe and for the walk's crossing of the coat, as Arnold's standard_surface does; other plastics keep
// Schlick's approximation (plastic_coat_f0).
inline float plastic_coat_fresnel_exact(const MaterialData material, const float cosTheta) {
    float cosThetaT;
    return fresnel_dielectric_exact(clamp(cosTheta, 0.0f, 1.0f), 1.0f, plastic_coat_ior(material), cosThetaT);
}

inline PlasticLobeProbabilities plastic_lobe_probabilities(const MaterialData material,
                                                           const float3 normal,
                                                           const float3 wo,
                                                           const bool specularOnly) {
    PlasticLobeProbabilities probs;
    probs.coat = 0.0f;
    probs.diffuse = 0.0f;
    float cosThetaO = dot(normal, wo);
    if (cosThetaO <= 0.0f) {
        return probs;
    }
    if ((material.materialFlags & kMaterialFlagSubsurfaceExit) != 0u) {
        probs.diffuse = 1.0f;
        return probs;
    }
    float f0 = plastic_coat_f0(material);
    float coatReflect = f0 + (1.0f - f0) * schlick_weight(cosThetaO);
    probs.coat = coatReflect * luminance_rgb(plastic_specular_tint(material));
    if (!specularOnly) {
        // (1 - F(cos_i)) averages to 1 - (f0 + (1 - f0) / 21) over the cosine-weighted hemisphere; T at cos_i = 1.
        float entering = (1.0f - f0) * (20.0f / 21.0f);
        probs.diffuse = (1.0f - coatReflect) * entering * max(1.0f - plastic_coat_fresnel_average(material), 0.0f) *
                        luminance_rgb(material_base_color(material) *
                                      plastic_diffuse_transmission(material, 1.0f, cosThetaO));
    }
    float sum = probs.coat + probs.diffuse;
    if (!(sum > 0.0f)) {
        probs.coat = 0.0f;
        probs.diffuse = 0.0f;
        return probs;
    }
    probs.coat /= sum;
    probs.diffuse /= sum;
    return probs;
}

inline ClosureValuePdf plastic_closure_eval(const MaterialData material,
                                            const float3 normal,
                                            const float3 wo,
                                            const float3 wi,
                                            const PlasticLobeProbabilities probs,
                                            const float diffuseOcclusion,
                                            const bool specularOnly,
                                            const FireflyClampParams clampParams) {
    ClosureValuePdf res;
    res.value = float3(0.0f);
    res.pdf = 0.0f;
    float cosThetaO = dot(normal, wo);
    float cosThetaI = dot(normal, wi);
    if (cosThetaO <= 0.0f || cosThetaI <= 0.0f) {
        return res;
    }
    if ((material.materialFlags & kMaterialFlagSubsurfaceExit) != 0u) {
        // Exit point of a coated subsurface walk: a white diffuse surface under the coat's tint and its absorption along
        // the light-side crossing. The coat's Fresnel weights the walk once, at its entry, for the view direction
        // (subsurface_vertex_begin), as in Arnold's layering; the exit takes no Fresnel and no (1 - Favg).
        res.value = material_base_color(material) * (clamp(diffuseOcclusion, 0.0f, 1.0f) / kPi) *
                    plastic_coat_tint(material) * plastic_coat_crossing(material, cosThetaI);
        res.pdf = lambert_pdf(normal, wi);
        return res;
    }
    float3 f0 = float3(plastic_coat_f0(material));
    float3 wh = safe_normalize(wo + wi);
    const float3 coatFresnel = (material.materialFlags & kMaterialFlagSubsurfaceCoatLobe) != 0u
                                   ? float3(plastic_coat_fresnel_exact(material, dot(wi, wh)))
                                   : schlick_fresnel(f0, dot(wi, wh));
    ClosureValuePdf coat = ggx_reflection_lobe(ggx_vndf_alpha(plastic_coat_roughness(material)),
                                               normal, cosThetaI, wo, wi,
                                               coatFresnel * plastic_specular_tint(material),
                                               clampParams);
    float3 diffuse = float3(0.0f);
    if (!specularOnly) {
        diffuse = material_base_color(material) * (clamp(diffuseOcclusion, 0.0f, 1.0f) / kPi) *
                  plastic_diffuse_transmission(material, cosThetaI, cosThetaO) *
                  (float3(1.0f) - schlick_fresnel(f0, cosThetaI)) * (float3(1.0f) - schlick_fresnel(f0, cosThetaO)) *
                  max(1.0f - plastic_coat_fresnel_average(material), 0.0f);
    }
    res.value = max(coat.value + diffuse, float3(0.0f));
    res.pdf = probs.coat * coat.pdf + probs.diffuse * lambert_pdf(normal, wi);
    return res;
}

inline bool material_is_subsurface(const MaterialData material) {
    return static_cast<uint>(material.typeEta.x) == 5u;
}

inline float3 sss_sigma_a(const MaterialData material,
                          const float3 baseColor,
                          float meanFreePath,
                          float anisotropy) {
    bool hasOverride = material.sssSigmaA.w > 0.5f;
    if (hasOverride) {
        float3 sigmaA = max(material.sssSigmaA.xyz, float3(1.0e-6f));
        return sigmaA;
    }
    float sigmaT = 1.0f / max(meanFreePath, 1.0e-4f);
    float3 sigmaS = clamp(baseColor, float3(0.0f), float3(0.999f)) * sigmaT;
    sigmaS = max(sigmaS, float3(0.0f));
    sigmaS *= max(1.0f - anisotropy, 0.01f);
    float3 sigmaA = max(float3(sigmaT) - sigmaS, float3(1.0e-6f));
    return sigmaA;
}

inline float3 sss_sigma_s_prime(const MaterialData material,
                                const float3 baseColor,
                                float meanFreePath,
                                float anisotropy) {
    bool hasOverride = material.sssSigmaA.w > 0.5f;
    if (hasOverride) {
        float3 sigmaS = max(material.sssSigmaS.xyz, float3(0.0f));
        sigmaS *= max(1.0f - anisotropy, 0.01f);
        return sigmaS;
    }
    float sigmaT = 1.0f / max(meanFreePath, 1.0e-4f);
    float3 sigmaS = clamp(baseColor, float3(0.0f), float3(0.999f)) * sigmaT;
    sigmaS = max(sigmaS, float3(0.0f));
    sigmaS *= max(1.0f - anisotropy, 0.01f);
    return sigmaS;
}

inline float3 normalized_diffusion_profile(const float radius,
                                           const float3 sigmaA,
                                           const float3 sigmaSPrime) {
    float3 sigmaTPrime = sigmaA + sigmaSPrime;
    float3 safeSigmaTPrime = max(sigmaTPrime, float3(1.0e-6f));
    float3 alphaPrime = clamp01(sigmaSPrime / safeSigmaTPrime);
    float3 D = 1.0f / max(3.0f * safeSigmaTPrime, float3(1.0e-6f));
    float3 sigmaTr = sqrt(max(sigmaA / D, float3(1.0e-6f)));
    float3 rVec = float3(max(radius, 1.0e-4f));
    float3 zr = 1.0f / safeSigmaTPrime;
    float3 dr = sqrt(rVec * rVec + zr * zr);
    float3 vr = zr + 4.0f * D;
    float3 dv = sqrt(rVec * rVec + vr * vr);
    float3 expDr = exp(-sigmaTr * dr);
    float3 expDv = exp(-sigmaTr * dv);
    float3 denomDr = max(dr * dr * dr, float3(1.0e-6f));
    float3 denomDv = max(dv * dv * dv, float3(1.0e-6f));
    float3 termDr = (zr * (float3(1.0f) + sigmaTr * dr)) / denomDr;
    float3 termDv = (vr * (float3(1.0f) + sigmaTr * dv)) / denomDv;
    float3 profile = (alphaPrime / (4.0f * kPi)) * (termDr * expDr + termDv * expDv);
    return max(profile, float3(0.0f));
}

inline float sss_sigma_tr_scalar(const float3 sigmaA,
                                 const float3 sigmaSPrime) {
    float3 sigmaTPrime = sigmaA + sigmaSPrime;
    float3 safeSigmaTPrime = max(sigmaTPrime, float3(1.0e-6f));
    float3 D = 1.0f / max(3.0f * safeSigmaTPrime, float3(1.0e-6f));
    float3 sigmaTr = sqrt(max(sigmaA / D, float3(1.0e-6f)));
    return max(luminance_rgb(sigmaTr), 1.0e-4f);
}

inline float sample_sss_radius(const float sigmaTrScalar, thread uint& state) {
    float u = rand_uniform(state);
    u = clamp(u, 1.0e-6f, 1.0f - 1.0e-6f);
    return -log(1.0f - u) / max(sigmaTrScalar, 1.0e-4f);
}

inline float pdf_sss_radius(const float radius, const float sigmaTrScalar) {
    if (radius <= 0.0f) {
        return 0.0f;
    }
    float sigma = max(sigmaTrScalar, 1.0e-4f);
    return sigma * exp(-sigma * radius);
}

inline float schlick_fresnel_scalar(const float f0, const float cosTheta) {
    float m = clamp01(1.0f - cosTheta);
    float m2 = m * m;
    float m5 = m2 * m2 * m;
    return f0 + (1.0f - f0) * m5;
}

inline float3 to_world_with_reference(const float3 local, const float3 reference) {
    float3 ref = safe_normalize(reference);
    float3 tangent;
    float3 bitangent;
    build_onb(ref, tangent, bitangent);
    return safe_normalize(local.x * tangent + local.y * bitangent + local.z * ref);
}

inline float3 sample_henyey_greenstein_local(const float g, thread uint& state) {
    float u1 = rand_uniform(state);
    float u2 = rand_uniform(state);
    float cosTheta = 0.0f;
    if (fabs(g) < 1.0e-3f) {
        cosTheta = 1.0f - 2.0f * u1;
    } else {
        float s = (1.0f - g * g) / (1.0f - g + 2.0f * g * u1);
        cosTheta = (1.0f + g * g - s * s) / (2.0f * g);
        cosTheta = clamp(cosTheta, -1.0f, 1.0f);
    }
    float sinTheta = safe_sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
    float phi = 2.0f * kPi * u2;
    float3 local = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    return local;
}

inline float3 sample_henyey_greenstein_world(const float3 referenceDir,
                                             const float g,
                                             thread uint& state) {
    float3 local = sample_henyey_greenstein_local(g, state);
    return to_world_with_reference(local, referenceDir);
}

// Random-walk subsurface (material method=randomwalk under renderer sss=randomwalk).
// The model is the one Arnold's randomwalk uses: light enters the mesh as a diffuse transmission, walks through a
// homogeneous medium behind an index-matched boundary, and where it leaves the mesh the path continues from an
// ordinary diffuse surface vertex. `base` is the surface colour a dense medium converges to, `mfp` is 1 / sigma_t and
// `g` is the Henyey-Greenstein anisotropy. The material `ior` plays no part in the walk.

// Single-scattering albedo whose semi-infinite medium reflects the fraction `colour` of diffuse light: the closed-form
// inverse of van de Hulst's fit (Chiang et al. 2016, "Practical and Controllable Subsurface Scattering for Production
// Path Tracing").
inline float3 sss_walk_single_scatter_albedo(const float3 colour) {
    float3 c = clamp(colour, float3(0.0f), float3(0.999f));
    float3 s = float3(4.09712f) + 4.20863f * c - sqrt(float3(9.59217f) + 41.6808f * c + 17.7126f * c * c);
    return clamp(float3(1.0f) - s * s, float3(0.0f), float3(1.0f));
}

struct SssWalkMedium {
    float3 sigmaT;
    float3 sigmaS;
    float g;
};

inline SssWalkMedium sss_walk_medium(const MaterialData material) {
    SssWalkMedium medium;
    medium.g = clamp(material.sssSigmaS.w, -0.99f, 0.99f);
    if (material.sssSigmaA.w > 0.5f) {
        // Explicit sigma_a and sigma_s are used as given.
        medium.sigmaS = max(material.sssSigmaS.xyz, float3(0.0f));
        medium.sigmaT = max(max(material.sssSigmaA.xyz, float3(0.0f)) + medium.sigmaS, float3(1.0e-6f));
        return medium;
    }
    // The inversion holds for isotropic scattering; the similarity relation carries it to g != 0.
    float3 reduced = sss_walk_single_scatter_albedo(material_base_color(material));
    float3 albedo = reduced / max(float3(1.0f - medium.g) + medium.g * reduced, float3(1.0e-6f));
    float sigmaT = 1.0f / max(material.sssParams.x, 1.0e-4f);
    medium.sigmaT = float3(sigmaT);
    medium.sigmaS = clamp01(albedo) * sigmaT;
    return medium;
}

struct SssWalkState {
    float3 position;
    float3 direction;
    float3 throughput;
    float3 channelProbability;  // probability of drawing this segment's length from each colour channel
    float distance;             // free-flight length drawn for the current segment
    float travelled;            // part of it already behind the walk (other objects' surfaces skipped on the way)
    float traceMin;             // tMin of the next trace: kRayOriginEpsilon from a surface, 0 from a scatter point
    uint skips;
};

constant uint kSssWalkContinue = 0u;   // the walk has another segment to trace
constant uint kSssWalkExit = 1u;       // left the mesh through its own boundary
constant uint kSssWalkAbsorbed = 2u;   // throughput fell below kSssThroughputCutoff
constant uint kSssWalkOpenMesh = 3u;   // found itself outside: the mesh is not closed
constant uint kSssWalkStepLimit = 4u;  // still inside after sssMaxSteps segments
constant uint kSssWalkMaxSkips = 8u;

// Draws the next free-flight length. The channel is chosen in proportion to the throughput and the weights divide by
// the density averaged over the channels, so media whose sigma_t differs per channel keep bounded weights.
inline void sss_walk_sample_segment(thread SssWalkState& walk,
                                    const SssWalkMedium medium,
                                    thread uint& state) {
    float sum = walk.throughput.x + walk.throughput.y + walk.throughput.z;
    walk.channelProbability = walk.throughput / max(sum, 1.0e-20f);
    float pick = rand_uniform(state);
    float sigma = medium.sigmaT.z;
    if (pick < walk.channelProbability.x) {
        sigma = medium.sigmaT.x;
    } else if (pick < walk.channelProbability.x + walk.channelProbability.y) {
        sigma = medium.sigmaT.y;
    }
    float xi = clamp(rand_uniform(state), 1.0e-6f, 1.0f - 1.0e-6f);
    walk.distance = -log(1.0f - xi) / sigma;
    walk.travelled = 0.0f;
    walk.skips = 0u;
}

// `entry` is a front-face hit, so its normal faces the incoming ray. The entry is a diffuse transmission: a
// cosine-weighted direction about the inward normal, with weight 1.
inline void sss_walk_begin(thread const HitRecord& entry,
                           const SssWalkMedium medium,
                           thread uint& state,
                           thread SssWalkState& walk) {
    float3 inward = -entry.normal;
    float3 direction = safe_normalize(to_world(sample_cosine_hemisphere(state), inward));
    walk.position = offset_surface_point(entry.point, inward, direction);
    walk.direction = direction;
    walk.throughput = float3(1.0f);
    walk.traceMin = kRayOriginEpsilon;
    sss_walk_sample_segment(walk, medium, state);
}

inline bool sss_walk_same_object(thread const HitRecord& entry, thread const HitRecord& hit) {
    if (hit.primitiveType != entry.primitiveType) {
        return false;
    }
    return hit.primitiveType == kPrimitiveTypeTriangle ? hit.meshIndex == entry.meshIndex
                                                       : hit.primitiveIndex == entry.primitiveIndex;
}

// Consumes the closest-hit trace of the current segment (`hit`, `boundary`) and returns one of the kSssWalk codes.
inline uint sss_walk_advance(thread SssWalkState& walk,
                             const SssWalkMedium medium,
                             thread const HitRecord& entry,
                             const bool hit,
                             thread const HitRecord& boundary,
                             thread uint& state) {
    if (!hit) {
        return kSssWalkOpenMesh;  // nothing ahead in any direction means the walk is outside its mesh
    }
    float remaining = walk.distance - walk.travelled;
    if (boundary.t >= remaining) {
        // Scatter inside the medium before reaching any surface.
        float3 transmittance = exp(-medium.sigmaT * walk.distance);
        float density = dot(walk.channelProbability, medium.sigmaT * transmittance);
        walk.throughput *= medium.sigmaS * transmittance / max(density, 1.0e-30f);
        if (!(max(walk.throughput.x, max(walk.throughput.y, walk.throughput.z)) >= kSssThroughputCutoff)) {
            return kSssWalkAbsorbed;
        }
        walk.position += walk.direction * remaining;
        walk.direction = sample_henyey_greenstein_world(walk.direction, medium.g, state);
        if (!all(isfinite(walk.direction)) || dot(walk.direction, walk.direction) <= 0.0f) {
            return kSssWalkAbsorbed;
        }
        // A scatter point lies inside the medium, not on a surface, so the next trace needs no offset. With one, a
        // boundary closer than kRayOriginEpsilon would be passed through and the exit lost as an open-mesh ending.
        walk.traceMin = 0.0f;
        sss_walk_sample_segment(walk, medium, state);
        return kSssWalkContinue;
    }
    if (!sss_walk_same_object(entry, boundary)) {
        // Another object's surface inside this one: the medium continues behind it.
        if (walk.skips >= kSssWalkMaxSkips) {
            return kSssWalkOpenMesh;
        }
        walk.skips += 1u;
        walk.travelled += boundary.t + kRayOriginEpsilon;
        walk.position = boundary.point + walk.direction * kRayOriginEpsilon;
        walk.traceMin = kRayOriginEpsilon;
        return kSssWalkContinue;
    }
    if (boundary.frontFace != 0u) {
        return kSssWalkOpenMesh;  // met its own surface from outside
    }
    // Reached the boundary from inside before the drawn length: the walk leaves the mesh here.
    float3 reach = exp(-medium.sigmaT * (walk.travelled + boundary.t));
    walk.throughput *= reach / max(dot(walk.channelProbability, reach), 1.0e-30f);
    return kSssWalkExit;
}

// Turns the boundary hit the walk left through (a back-face hit seen from inside) into the front-face hit record a
// ray arriving from outside would have produced.
inline HitRecord sss_walk_exit_record(thread const HitRecord& boundary) {
    HitRecord exitRec = boundary;
    exitRec.normal = -boundary.normal;
    exitRec.shadingNormal = -boundary.shadingNormal;
    exitRec.frontFace = 1u;
    return exitRec;
}

// A software trace records the flat triangle normal as the shading normal; the integrators interpolate the vertex
// normals after the trace, a step the walk's exit skips. This gives a software exit the shading normal a hardware
// trace records: the interpolated vertex normal, turned to the side of the geometric normal.
inline void sss_walk_exit_interpolate_normal(constant PathtraceUniforms& uniforms,
                                             thread HitRecord& exitRec,
                                             device const MeshInfo* meshInfos,
                                             device const SceneVertex* sceneVertices,
                                             device const uint3* meshIndices) {
    if (exitRec.primitiveType != kPrimitiveTypeTriangle || !meshInfos || !sceneVertices || !meshIndices ||
        uniforms.meshCount == 0u) {
        return;
    }
    float3 candidate = interpolate_shading_normal(uniforms, exitRec.meshIndex, exitRec.primitiveIndex,
                                                  exitRec.barycentric, meshInfos, sceneVertices, meshIndices);
    if (all(isfinite(candidate)) && dot(candidate, candidate) > 0.0f) {
        exitRec.shadingNormal = normalize(dot(candidate, exitRec.normal) < 0.0f ? -candidate : candidate);
    }
}

inline uint sss_random_walk_software(constant PathtraceUniforms& uniforms,
                                     const MaterialData material,
                                     thread const HitRecord& entry,
                                     device const SphereData* spheres,
                                     device const RectData* rectangles,
                                     device const TriangleData* triangleData,
                                     device const BvhNode* tlasNodes,
                                     device const uint* tlasPrimIndices,
                                     device const SoftwareInstanceInfo* instanceInfos,
                                     device const BvhNode* blasNodes,
                                     device const uint* blasPrimIndices,
                                     device const BvhNode* nodes,
                                     device const uint* primitiveIndices,
                                     device const MeshInfo* meshInfos,
                                     device const SceneVertex* sceneVertices,
                                     device const uint3* meshIndices,
                                     device PathtraceStats* stats,
                                     thread uint& state,
                                     thread HitRecord& exitRec,
                                     thread float3& weight) {
    SssWalkMedium medium = sss_walk_medium(material);
    SssWalkState walk;
    sss_walk_begin(entry, medium, state, walk);
    uint maxSteps = max(uniforms.sssMaxSteps, 1u);
    uint outcome = kSssWalkStepLimit;
    for (uint step = 0u; step < maxSteps; ++step) {
        Ray segment;
        segment.origin = walk.position;
        segment.direction = walk.direction;
        HitRecord boundary;
        bool hit = trace_scene_software(uniforms, spheres, rectangles, triangleData, tlasNodes, tlasPrimIndices,
                                        instanceInfos, blasNodes, blasPrimIndices, nodes, primitiveIndices, stats,
                                        segment, walk.traceMin, kInfinity, /*anyHitOnly=*/false,
                                        /*includeTriangles=*/true, boundary);
        uint event = sss_walk_advance(walk, medium, entry, hit, boundary, state);
        if (event == kSssWalkContinue) {
            continue;
        }
        if (event == kSssWalkExit) {
            exitRec = sss_walk_exit_record(boundary);
            sss_walk_exit_interpolate_normal(uniforms, exitRec, meshInfos, sceneVertices, meshIndices);
            weight = walk.throughput;
        }
        outcome = event;
        break;
    }
    sss_walk_note_outcome(stats, outcome, entry.materialIndex);
    return outcome;
}

#if __METAL_VERSION__ >= 310
inline uint sss_random_walk_hardware(constant PathtraceUniforms& uniforms,
                                     const MaterialData material,
                                     thread const HitRecord& entry,
                                     acceleration_structure<instancing> accel,
                                     device const MeshInfo* meshInfos,
                                     device const TriangleData* triangleData,
                                     device const SceneVertex* sceneVertices,
                                     device const uint3* meshIndices,
                                     device const uint* instanceUserIds,
                                     device const SphereData* spheres,
                                     device const RectData* rectangles,
                                     device const BvhNode* nodes,
                                     device const uint* primitiveIndices,
                                     device PathtraceStats* stats,
                                     thread uint& state,
                                     thread HitRecord& exitRec,
                                     thread float3& weight) {
    SssWalkMedium medium = sss_walk_medium(material);
    SssWalkState walk;
    sss_walk_begin(entry, medium, state, walk);
    uint maxSteps = max(uniforms.sssMaxSteps, 1u);
    uint outcome = kSssWalkStepLimit;
    for (uint step = 0u; step < maxSteps; ++step) {
        Ray segment;
        segment.origin = walk.position;
        segment.direction = walk.direction;
        HitRecord boundary;
        bool hit = trace_scene_hardware(uniforms, accel, meshInfos, triangleData, sceneVertices, meshIndices,
                                        instanceUserIds, spheres, rectangles, nodes, primitiveIndices, stats,
                                        segment, walk.traceMin, kInfinity, /*anyHitOnly=*/false,
                                        kInvalidIndex, kInvalidIndex, boundary);
        uint event = sss_walk_advance(walk, medium, entry, hit, boundary, state);
        if (event == kSssWalkContinue) {
            continue;
        }
        if (event == kSssWalkExit) {
            exitRec = sss_walk_exit_record(boundary);
            weight = walk.throughput;
        }
        outcome = event;
        break;
    }
    sss_walk_note_outcome(stats, outcome, entry.materialIndex);
    return outcome;
}
#endif

// What an integrator does at a hit on a subsurface material, before it shades the vertex.
constant uint kSubsurfaceVertexShade = 0u;  // shade `material` as it now is
constant uint kSubsurfaceVertexWalk = 1u;   // run the walk from the hit, then call subsurface_vertex_exit
constant uint kSubsurfaceVertexCoat = 2u;   // shade `material` (the coat lobe) as it now is, at the coat normal

inline bool subsurface_has_coat(const MaterialData material) {
    return material.sssParams.z > 0.5f;
}

// What a subsurface surface is in the limit of a vanishing mean free path: a diffuse surface of its base colour,
// under its coat when it has one (the plastic closure, with the scene's coatIOR). A hit where no walk runs is shaded
// as this, so it receives direct light like any other surface.
inline MaterialData subsurface_dense_limit_material(const MaterialData material) {
    MaterialData dense = material;
    if (subsurface_has_coat(material)) {
        dense.typeEta.x = 4.0f;
        dense.typeEta.y = max(material.typeEta.z, 1.0f);
    } else {
        dense.typeEta.x = 0.0f;
    }
    return dense;
}

// Called at every surface hit before direct lighting, on the integrator's local material copy.
// - Not subsurface, or the separable method under sss=separable: nothing changes.
// - No walk at this hit (sss off, a method the renderer mode does not run, a back-face hit, specular-only debug):
//   the material becomes its dense limit.
// - Random walk without a coat: the walk runs.
// - Random walk with a coat: one-sample choice between the coat lobe and the walk, with the plastic closure's lobe
//   probabilities. The coat lobe is shaded here as plastic with a black base and the exact dielectric Fresnel
//   (kMaterialFlagSubsurfaceCoatLobe). The walk takes the coat's (1 - F) and crossing for the view direction here, and
//   only the coat's tint and light-side absorption at its exit point (kMaterialFlagSubsurfaceExit). This is Arnold's
//   layering (backlog #49): the walk under the coat is weighted once, by what the coat does not reflect toward the view.
// `coatNormal` is the coat's normal: the shading normal, or the material's mapped normal (backlog #47,
// subsurface_coat_mapped_normal). It sets the lobe probabilities; on kSubsurfaceVertexCoat the caller shades the coat lobe
// at it. The walk's view-side crossing and Fresnel weight stay on the shading normal, and the walk starts from `rec`. The
// map changes the walk's value through the lobe choice and through the coat light it sends below the shading normal.
inline uint subsurface_vertex_begin(constant PathtraceUniforms& uniforms,
                                    thread MaterialData& material,
                                    thread const HitRecord& rec,
                                    const float3 shadingNormal,
                                    const float3 coatNormal,
                                    const float3 wo,
                                    const bool specularOnly,
                                    thread uint& state,
                                    thread float3& throughput) {
    if (!material_is_subsurface(material)) {
        return kSubsurfaceVertexShade;
    }
    const uint sssMode = PT_DEFAULT_OFF(uniforms.sssMode);
    if (sss_use_separable(sssMode, material)) {
        return kSubsurfaceVertexShade;
    }
    if (!sss_use_random_walk(sssMode, material) || rec.frontFace == 0u || specularOnly) {
        material = subsurface_dense_limit_material(material);
        return kSubsurfaceVertexShade;
    }
    if (!subsurface_has_coat(material)) {
        return kSubsurfaceVertexWalk;
    }
    MaterialData plastic = subsurface_dense_limit_material(material);
    PlasticLobeProbabilities probs = plastic_lobe_probabilities(plastic, coatNormal, wo, false);
    if (!(probs.coat + probs.diffuse > 0.0f)) {
        material = plastic;  // the view direction is below the shading normal: the plastic closure returns nothing
        return kSubsurfaceVertexShade;
    }
    const float cosThetaO = dot(shadingNormal, wo);
    if (rand_uniform(state) < probs.coat) {
        // A mapped coat normal reflects part of the coat's light below the shading normal, into the surface. As
        // in Arnold, whose subsurface gains what a normal map takes from its specular, that energy enters the walk:
        // half the coat picks estimate it with one coat sample and continue as a walk, the other half shade the coat.
        // Without a map the two normals are equal, nothing is lost, and the coat pick stays as it was.
        float coatShare = 1.0f;
        if (any(coatNormal != shadingNormal)) {
            coatShare = 0.5f;
            if (rand_uniform(state) < 0.5f) {
                const float coatRoughness = plastic_coat_roughness(plastic);
                const float3 wh = sample_ggx_vndf(coatNormal, wo, coatRoughness, state);
                const float3 wi = safe_normalize(reflect(-wo, wh));
                if (!(dot(wh, coatNormal) > 0.0f) || !(dot(wi, coatNormal) > 0.0f) ||
                    !(dot(wi, shadingNormal) <= 0.0f) || !(cosThetaO > 0.0f)) {
                    // The coat sample stays above the surface: nothing is lost on this pick. It ends at the coat
                    // vertex, with no walk to run or count.
                    plastic.baseColorRoughness.xyz = float3(0.0f);
                    plastic.materialFlags |= kMaterialFlagSubsurfaceCoatLobe;
                    material = plastic;
                    throughput = float3(0.0f);
                    return kSubsurfaceVertexCoat;
                }
                const float3 lost = plastic_coat_fresnel_exact(plastic, dot(wo, wh)) * plastic_specular_tint(plastic) *
                                    ggx_G1(ggx_vndf_alpha(coatRoughness), dot(coatNormal, wi));
                throughput *= lost * plastic_coat_crossing(plastic, cosThetaO) / (probs.coat * 0.5f);
                return kSubsurfaceVertexWalk;
            }
        }
        plastic.baseColorRoughness.xyz = float3(0.0f);
        plastic.materialFlags |= kMaterialFlagSubsurfaceCoatLobe;
        material = plastic;
        throughput /= probs.coat * coatShare;
        return kSubsurfaceVertexCoat;
    }
    throughput *= (1.0f - plastic_coat_fresnel_exact(plastic, cosThetaO)) * plastic_coat_crossing(plastic, cosThetaO) /
                  probs.diffuse;
    return kSubsurfaceVertexWalk;
}

// Moves the shading vertex to the point where the walk left the mesh. The hit record becomes the exit point's and the
// material a white diffuse surface (under the coat's tint and light-side absorption when coated), so direct light, MIS and the
// next ray origin are the ordinary ones. The caller then uses the exit normal as the view direction.
inline void subsurface_vertex_exit(thread MaterialData& material,
                                   thread HitRecord& rec,
                                   thread const HitRecord& exitRec) {
    rec = exitRec;
    if (subsurface_has_coat(material)) {
        material = subsurface_dense_limit_material(material);
        material.materialFlags |= kMaterialFlagSubsurfaceExit;
    } else {
        material.typeEta.x = 0.0f;
    }
    material.baseColorRoughness.xyz = float3(1.0f);
}

#include "../shared/source_closures.h"

// Path regularization for dielectrics flagged kMaterialFlagRegularizedDielectric:
// once a path has scattered off a rough surface, the smooth interface is treated
// as rough GGX with the path's maximum roughness, so the sun and small lights
// become reachable through it (e.g. a pool floor lit through the water surface).
// The integrator applies this to its local material copy: baseColorRoughness.w
// holds the regularized roughness and typeEta.y the relative IOR across the
// interface (far side over incident side).
inline bool material_is_regularized_dielectric(const MaterialData material) {
    return static_cast<uint>(material.typeEta.x) == 2u &&
           (material.materialFlags & kMaterialFlagRegularizedDielectric) != 0u &&
           material.baseColorRoughness.w > 0.0f;
}

inline void regularize_dielectric_material(thread MaterialData& material,
                                           const bool frontFace,
                                           const float pathRoughness) {
    if (source_model(material) == 3.0f) {
        source_regularize_dielectric(material, frontFace);
        return;
    }
    if (static_cast<uint>(material.typeEta.x) != 2u ||
        (material.materialFlags & kMaterialFlagRegularizedDielectric) == 0u ||
        material.typeEta.w > 0.5f ||
        !(pathRoughness > 0.0f)) {
        return;
    }
    const float ior = max(material.typeEta.y, 1.0f);
    material.typeEta.y = frontFace ? ior : 1.0f / ior;
    material.baseColorRoughness.w = min(pathRoughness, 1.0f);
}

// Roughness a non-delta scattering event contributes to path regularization.
inline float path_regularization_roughness(const MaterialData material) {
    return (static_cast<uint>(material.typeEta.x) == 0u)
        ? 1.0f
        : clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
}

// The material samples at least one delta lobe. Delta chains can continue
// through it by sampling the closure and keeping only delta samples.
inline bool material_has_delta_lobe(const MaterialData material) {
    if (source_model(material) > 0.0f) return source_has_delta_lobe(material);
    uint type = static_cast<uint>(material.typeEta.x);
    if (type == 2u) {
        return !material_is_regularized_dielectric(material);
    }
    if (type == 1u) {
        float roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        return roughness <= 1e-3f;
    }
    if (type == 7u) {
        float roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        return roughness <= 1e-3f;
    }
    if (material_is_plastic(material)) {
        return false;
    }
    return false;
}

// Every lobe that carries energy is delta, so light sampling has nothing to
// evaluate and integrators skip NEE. A roughness-0 PBR surface keeps its
// diffuse lobe unless metallic or transmission removes it; it must stay
// connectable, or that lobe loses delta lights and the MIS share of
// environment and emitter hits that only NEE would have supplied.
inline bool material_is_delta(const MaterialData material) {
    if (source_never_delta(material)) return false;
    if (!material_has_delta_lobe(material)) {
        return false;
    }
    if (static_cast<uint>(material.typeEta.x) == 7u) {
        // The diffuse lobe must be zero for every texel. Integrators classify
        // before textures are applied, and textures only scale metallic and
        // transmission down from their factors, which can add diffuse.
        const float3 baseColor = material_base_color(material);
        if (max(max(baseColor.x, baseColor.y), baseColor.z) <= 0.0f) {
            return true;
        }
        const float metallic = clamp(material.pbrParams.x, 0.0f, 1.0f);
        if (metallic > 0.0f && material.textureIndices0.y != kInvalidIndex) {
            return false;
        }
        const float transmission = (material.textureIndices1.y != kInvalidIndex)
            ? 0.0f : clamp(material.pbrExtras.z, 0.0f, 1.0f);
        return (1.0f - metallic) * (1.0f - transmission * (1.0f - metallic)) <= 0.0f;
    }
    return true;
}

inline bool material_is_thin_dielectric(const MaterialData material) {
    uint type = static_cast<uint>(material.typeEta.x);
    return (type == 2u) && (material.typeEta.w > 0.5f);
}

// glTF transmission without a volume is thin-walled: an infinitely thin sheet
// with the outside medium on both faces. Transmission does not refract (the
// smooth lobe continues along -wo; the rough lobe is the GGX reflection lobe
// mirrored through the tangent plane) and the Fresnel term always uses the
// authored IOR from outside, on either face. Tagged reference (Pavilion)
// materials keep their own refracting convention.
inline bool pbr_thin_walled_transmission(const MaterialData material) {
    return static_cast<uint>(material.typeEta.x) == 7u &&
        (material.materialFlags & kMaterialFlagThinWalledTransmission) != 0u &&
        (material.materialFlags & kMaterialFlagSingleScatterSpecular) == 0u;
}

// Light sampling may connect through the rough thin-walled BTDF: it is a
// non-delta lobe on the opposite hemisphere whose value and PDF are evaluated
// by evaluate_pbr_metallic_roughness. Without this, BSDF-sampled emitter and
// environment hits through rough thin glass were MIS-weighted against a light
// sample that always evaluated to zero.
inline bool pbr_thin_walled_transmission_nee_supported(const MaterialData material) {
    return pbr_thin_walled_transmission(material) &&
        material.pbrExtras.z > 0.0f && material.pbrParams.x < 1.0f &&
        material.baseColorRoughness.w > 1.0e-3f;
}

inline float3 dielectric_transmission_tint(const MaterialData material) {
    return ((material.materialFlags & kMaterialFlagDielectricTransmissionTint) != 0u)
        ? clamp(material_base_color(material), float3(0.0f), float3(1.0f))
        : float3(1.0f);
}

// Pavilion reference compatibility only. Path-Tracing/shading.glsl
// EvaluateRefraction puts incident/far eta^2 in both its BSDF and PDF, so
// sampleBSDF cancels it from continuation while direct-light evaluation keeps
// it. Preserve our actual sampling PDF; this is a legacy estimator convention,
// not the default physical radiance-transport model or exact area-light MIS.
inline bool dielectric_uses_reference_continuation(const MaterialData material) {
    return (material.materialFlags & kMaterialFlagReferenceDielectricContinuation) != 0u;
}

inline float dielectric_continuation_eta_scale(const MaterialData material,
                                                const float incidentOverFarEta) {
    return dielectric_uses_reference_continuation(material)
        ? 1.0f : incidentOverFarEta * incidentOverFarEta;
}

inline bool pavilion_transmission_nee_supported(const MaterialData material) {
    return static_cast<uint>(material.typeEta.x) == 7u &&
        (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u &&
        material.pbrExtras.z > 0.0f && material.pbrParams.x < 1.0f;
}

// Geometry/sky NEE competes with BSDF-hit paths, so it must use the same
// eta-free continuation convention as tagged reference sampling. Its actual
// direction PDF remains unchanged. Discrete reference points and suns retain
// the original reference's direct-light value and do not call this helper.
inline float3 reference_continuation_nee_value(const MaterialData material,
                                              const float3 normal,
                                              const float3 wo,
                                              const float3 wi,
                                              const float3 value) {
    const bool reference = dielectric_uses_reference_continuation(material) ||
        (static_cast<uint>(material.typeEta.x) == 7u &&
         (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u);
    if (reference && dot(normal, wo) * dot(normal, wi) < 0.0f) {
        const float farOverIncident = max(material.typeEta.y, 1.0e-3f);
        return value * farOverIncident * farOverIncident;
    }
    return value;
}

// Rough GGX dielectric (PBRT-v4 DielectricBxDF, radiance transport) for a
// regularized interface. `normal` faces wo; see material_is_regularized_dielectric.
inline BsdfEvalResult evaluate_regularized_dielectric(const MaterialData material,
                                                      const float3 normal,
                                                      const float3 wo,
                                                      const float3 wi,
                                                      const bool referencePunctual = false) {
    BsdfEvalResult result;
    result.value = float3(0.0f);
    result.pdf = 0.0f;
    result.directionalPdf = 0.0f;
    result.areaPdf = 0.0f;
    result.isDelta = false;
    result.isBssrdf = false;

    const float cosThetaO = dot(normal, wo);
    const float cosThetaI = dot(normal, wi);
    if (!(cosThetaO > 0.0f) || cosThetaI == 0.0f) {
        return result;
    }
    const float eta = max(material.typeEta.y, 1.0e-3f);
    const float roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
    const float alpha = max(roughness * roughness, 1.0e-4f);
    const bool reflection = cosThetaI > 0.0f;
    float3 wh = reflection ? (wo + wi) : (wo + wi * eta);
    if (!(dot(wh, wh) > 0.0f)) {
        return result;
    }
    wh = normalize(wh);
    if (dot(wh, normal) < 0.0f) {
        wh = -wh;
    }
    const float woDotWh = dot(wo, wh);
    const float wiDotWh = dot(wi, wh);
    if (woDotWh <= 0.0f || wiDotWh * cosThetaI <= 0.0f) {
        return result;
    }

    float cosThetaT = 0.0f;
    const float F = fresnel_dielectric_exact(woDotWh, 1.0f, eta, cosThetaT);
    // Path roughness can reach alpha=1e-4; keep both lobes and their VNDF
    // density finite when the half-vector cosine rounds to one.
    const float D = ggx_D(alpha, normal, wh);
    // The Pavilion source limits discrete-light values on water/glass too.
    // The actual VNDF density and ordinary continuation remain uncapped.
    const float valueD = referencePunctual && dielectric_uses_reference_continuation(material)
        ? min(D, 1.0f) : D;
    const float G1o = ggx_G1(alpha, cosThetaO);
    const float G = G1o * ggx_G1(alpha, cosThetaI);
    const float pdfWh = ggx_vndf_pdf(alpha, normal, wo, wh, /*halfVectorD=*/true);
    if (reflection) {
        result.value = float3(F * valueD * G / (4.0f * cosThetaO * cosThetaI));
        result.pdf = F * pdfWh / (4.0f * woDotWh);
    } else {
        const float denom = wiDotWh + woDotWh / eta;
        const float denomSq = denom * denom;
        if (!(denomSq > 1.0e-12f)) {
            return result;
        }
        // Radiance crossing into the far medium scales by eta^-2.
        const float ft = (1.0f - F) * valueD * G *
                         fabs(wiDotWh * woDotWh / (cosThetaI * cosThetaO * denomSq)) /
                         (eta * eta);
        result.value = ft * dielectric_transmission_tint(material);
        result.pdf = (1.0f - F) * pdfWh * fabs(wiDotWh) / denomSq;
    }
    result.directionalPdf = result.pdf;
    return result;
}

inline BsdfSampleResult sample_regularized_dielectric(const MaterialData material,
                                                      const float3 normal,
                                                      const float3 wo,
                                                      const bool frontFace,
                                                      thread uint& state) {
    BsdfSampleResult result;
    result.direction = float3(0.0f);
    result.weight = float3(0.0f);
    result.pdf = 0.0f;
    result.directionalPdf = 0.0f;
    result.areaPdf = 0.0f;
    result.exitPoint = float3(0.0f);
    result.exitNormal = float3(0.0f);
    result.isDelta = false;
    result.isBssrdf = false;
    result.hasExitPoint = false;
    result.mediumEvent = 0;
    result.lobeType = 1u;
    result.lobeRoughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);

    if (!(dot(normal, wo) > 0.0f)) {
        return result;
    }
    const float eta = max(material.typeEta.y, 1.0e-3f);
    const float3 wh = sample_ggx_vndf(normal, wo, result.lobeRoughness, state);
    const float woDotWh = dot(wo, wh);
    if (!(woDotWh > 0.0f)) {
        return result;
    }
    float cosThetaT = 0.0f;
    const float F = fresnel_dielectric_exact(woDotWh, 1.0f, eta, cosThetaT);
    const bool reflection = rand_uniform(state) < F;
    const float3 wi = reflection
        ? reflect(-wo, wh)
        : safe_normalize(-wo / eta + (woDotWh / eta - cosThetaT) * wh);
    const float cosThetaI = dot(normal, wi);
    if (reflection ? !(cosThetaI > 0.0f) : !(cosThetaI < 0.0f)) {
        return result;
    }
    const BsdfEvalResult eval = evaluate_regularized_dielectric(material, normal, wo, wi);
    if (!(eval.pdf > 0.0f)) {
        return result;
    }
    result.direction = wi;
    result.weight = eval.value * fabs(cosThetaI) / eval.pdf;
    if (!reflection && dielectric_uses_reference_continuation(material)) {
        result.weight *= eta * eta;
    }
    result.pdf = eval.pdf;
    result.directionalPdf = eval.pdf;
    result.mediumEvent = reflection ? 0 : (frontFace ? 1 : -1);
    return result;
}

inline float max_component(const float3 v) {
    return max(v.x, max(v.y, v.z));
}

inline float dielectric_f0_from_ior(const float ior) {
    float eta = max(ior, 1.0f);
    float num = eta - 1.0f;
    float den = max(eta + 1.0f, 1.0e-6f);
    float f0 = (num / den) * (num / den);
    return clamp(f0, 0.0f, 0.99f);
}

// Frostbite (normalized Disney) diffuse relative to Lambert, as in the
// reference renderer's StandardBSDF; linearRoughness is the authored value.
inline float frostbite_diffuse_scale(const float3 normal,
                                     const float3 wo,
                                     const float3 wi,
                                     const float linearRoughness) {
    float wiDotH = saturate(dot(wi, safe_normalize(wo + wi)));
    float fd90 = 0.5f * linearRoughness + 2.0f * wiDotH * wiDotH * linearRoughness;
    float wiScatter = 1.0f + (fd90 - 1.0f) * pow(1.0f - saturate(dot(normal, wi)), 5.0f);
    float woScatter = 1.0f + (fd90 - 1.0f) * pow(1.0f - saturate(dot(normal, wo)), 5.0f);
    return wiScatter * woScatter * mix(1.0f, 1.0f / 1.51f, linearRoughness);
}

inline float pbr_specular_weight(const float3 f0) {
    return clamp(max_component(f0), 0.05f, 0.95f);
}

inline BsdfEvalResult evaluate_pbr_metallic_roughness(const MaterialData material,
                                                      const float3 normal,
                                                      const float3 wo,
                                                      const float3 wi,
                                                      const FireflyClampParams clampParams,
                                                      const float diffuseOcclusion,
                                                      const bool specularOnly,
                                                      const bool referencePunctual = false,
                                                      const bool pbrFrontFace = true) {
    BsdfEvalResult result;
    result.value = float3(0.0f);
    result.pdf = 0.0f;
    result.isDelta = false;
    result.directionalPdf = 0.0f;
    result.areaPdf = 0.0f;
    result.isBssrdf = false;

    float cosThetaO = dot(normal, wo);
    float cosThetaI = dot(normal, wi);
    float absCosThetaO = fabs(cosThetaO);
    float absCosThetaI = fabs(cosThetaI);
    if (absCosThetaO <= 0.0f || absCosThetaI <= 0.0f) {
        return result;
    }

    float3 baseColor = clamp01(material.baseColorRoughness.xyz);
    float metallic = clamp(material.pbrParams.x, 0.0f, 1.0f);
    float roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
    // Delta lobes carry discrete mass, not a continuous value or PDF.
    const bool hasDeltaLobes = roughness <= 1.0e-3f;
    const bool singleScatterSpecular =
        (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u;
    const float relativeIor = max(material.typeEta.y, 1.0e-3f);
    float dielectricF0 = dielectric_f0_from_ior(singleScatterSpecular
        ? max(relativeIor, 1.0f / relativeIor) : material.typeEta.y);
    float3 f0 = mix(float3(dielectricF0), baseColor, metallic);
    float3 diffuseColor = baseColor * (1.0f - metallic);
    diffuseColor *= clamp(diffuseOcclusion, 0.0f, 1.0f);
    if (specularOnly) {
        diffuseColor = float3(0.0f);
    }

    float transmission = clamp(material.pbrExtras.z, 0.0f, 1.0f) * (1.0f - metallic);
    const bool glassInterface = !singleScatterSpecular && transmission > 0.0f;
    const bool thinWalled = glassInterface && pbr_thin_walled_transmission(material);
    float diffuseReflectScale = 1.0f - transmission;
    float specularReflectScale = singleScatterSpecular ? 1.0f : diffuseReflectScale;
    float specWeightBase = specularOnly ? 1.0f : pbr_specular_weight(f0);
    if (!specularOnly && (material.materialFlags & kMaterialFlagFalcorStandardBsdf) != 0u) {
        // Falcor picks lobes by albedo: Fresnel-weighted specular against diffuse,
        // so grazing reflections on dark glass are sampled often enough to converge.
        const float specAlbedo = luminance_rgb(schlick_fresnel(f0, saturate(dot(normal, wo))));
        specWeightBase = clamp(specAlbedo / max(specAlbedo + luminance_rgb(diffuseColor), 1.0e-6f),
                               0.05f, 0.95f);
    }
    float wSpec = specWeightBase * specularReflectScale;
    float wDiff = specularOnly ? 0.0f : (1.0f - specWeightBase) * diffuseReflectScale;
    // Ordinary PBR's rough reflection exports the mixture weight. A pure metal has
    // no diffuse energy; keep reference and Falcor sampling conventions intact.
    if (!singleScatterSpecular && (material.materialFlags & kMaterialFlagFalcorStandardBsdf) == 0u &&
        metallic == 1.0f && all(diffuseColor == float3(0.0f))) {
        wDiff = 0.0f;
    }
    float wTrans = transmission;
    float weightSum = wSpec + wDiff + wTrans;
    if (weightSum <= 0.0f) {
        return result;
    }

    if ((material.materialFlags & kMaterialFlagThinFoliageShadow) != 0u &&
        material.typeEta.z > 0.5f &&
        material.pbrExtras.w > 0.5f) {
        if (specularOnly) {
            return result;
        }
        float foliageCosI = absCosThetaI;
        if (foliageCosI <= 0.0f) {
            return result;
        }
        float foliageOcclusion = max(clamp(diffuseOcclusion, 0.0f, 1.0f), 0.65f);
        float3 foliageDiffuse = baseColor * (1.0f - metallic) * foliageOcclusion / kPi;
        float foliagePdf = max(foliageCosI / kPi, 1.0e-6f);
        result.value = max(foliageDiffuse, float3(0.0f));
        result.pdf = foliagePdf;
        result.directionalPdf = foliagePdf;
        return result;
    }

    float pSpec = wSpec / weightSum;
    float pDiff = wDiff / weightSum;
    float pTrans = wTrans / weightSum;

    if (cosThetaO * cosThetaI > 0.0f) {
        if (cosThetaO <= 0.0f || cosThetaI <= 0.0f) {
            return result;
        }
        float alpha = max(roughness * roughness, 1.0e-4f);
        float3 wh = safe_normalize(wo + wi);
        if (dot(wh, normal) <= 0.0f || dot(wo, wh) <= 0.0f || dot(wi, wh) <= 0.0f) {
            return result;
        }
        float D = ggx_D(alpha, normal, wh);
        float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
        float cosThetaT = 0.0f;
        const float referenceF = singleScatterSpecular
            ? fresnel_dielectric_exact(dot(wo, wh), 1.0f, relativeIor, cosThetaT)
            : 0.0f;
        float3 F = singleScatterSpecular
            ? float3(referenceF) * (1.0f - metallic) + schlick_fresnel(baseColor, dot(wi, wh)) * metallic
            : schlick_fresnel(f0, dot(wi, wh));
        float denom = 4.0f * cosThetaO * cosThetaI;
        // Reference-inspired leaf calibration: limit bright white/blue glints
        // that overwhelm the green texture. This lowers glossy energy while
        // preserving the actual VNDF sampling density for MIS.
        const bool referenceLeafCard = singleScatterSpecular &&
            material.pbrExtras.w > 0.5f && material.pbrExtras.w < 1.5f;
        // The source caps GGX D for discrete punctual shading. Keep this value
        // convention separate from geometry/env continuation and never cap PDF.
        const bool capPunctualD = singleScatterSpecular && referencePunctual;
        const float shadingD = (referenceLeafCard || capPunctualD) ? min(D, 1.0f) : D;
        float3 spec = hasDeltaLobes ? float3(0.0f) : F * (shadingD * G / max(denom, 1.0e-6f));
        if (!singleScatterSpecular && (material.materialFlags & kMaterialFlagFalcorStandardBsdf) == 0u) {
            spec *= specular_energy_compensation(f0, roughness, absCosThetaO);
        }
        spec = sanitize_specular_value(spec);
        spec *= specularReflectScale;
        // VNDF density divided by reflection Jacobian 4*(wo.wh); reuse the BSDF D.
        // Preserve the separate Falcor convention.
        float pdfSpec = hasDeltaLobes ? 0.0f : ((singleScatterSpecular ||
                         (material.materialFlags & kMaterialFlagFalcorStandardBsdf) == 0u)
            ? D * ggx_G1(alpha, cosThetaO) / (4.0f * cosThetaO)
            : ggx_vndf_pdf(alpha, normal, wo, wh, /*halfVectorD=*/true) / (4.0f * dot(wo, wh)));

        float3 diffuse = (diffuseColor / kPi) * diffuseReflectScale;
        if ((material.materialFlags & kMaterialFlagFalcorStandardBsdf) != 0u) {
            diffuse *= frostbite_diffuse_scale(normal, wo, wi, roughness);
        }
        if (singleScatterSpecular) {
            diffuse *= 1.0f - referenceF;
        }
        float pdfDiffuse = lambert_pdf(normal, wi);

        float pdf = pSpec * pdfSpec + pDiff * pdfDiffuse;
        if (glassInterface && !hasDeltaLobes) {
            // A thin-walled sheet has the outside medium on both faces.
            const bool entering = pbrFrontFace || thinWalled;
            const float etaI = entering ? 1.0f : max(material.typeEta.y, 1.0f);
            const float etaT = entering ? max(material.typeEta.y, 1.0f) : 1.0f;
            float glassCosThetaT = 0.0f;
            const float glassF = fresnel_dielectric_exact(dot(wo, wh), etaI, etaT, glassCosThetaT);
            // One exact-Fresnel interface; the glass reflection is single-scattering.
            spec += float3(transmission * glassF * D * G / max(denom, 1.0e-6f));
            pdf += pTrans * glassF * D * ggx_G1(alpha, cosThetaO) / (4.0f * cosThetaO);
        }
        if (pdf > 0.0f) {
            result.value = max(spec + diffuse, float3(0.0f));
            result.pdf = singleScatterSpecular ? pdf : clamp_specular_pdf(pdf, clampParams);
            result.directionalPdf = result.pdf;
        }
        return result;
    }

    if (hasDeltaLobes || wTrans <= 0.0f) {
        return result;
    }

    if (thinWalled) {
        // Thin-walled BTDF: the GGX reflection lobe of wi mirrored through the
        // tangent plane (no refraction), weighted by 1 - F at the outside IOR.
        // sample_pbr_metallic_roughness draws exactly this lobe.
        const float3 n = cosThetaO > 0.0f ? normal : -normal;
        const float3 wiMirror = wi - 2.0f * dot(wi, n) * n;
        const float alpha = max(roughness * roughness, 1.0e-4f);
        const float3 wh = safe_normalize(wo + wiMirror);
        const float woDotWh = dot(wo, wh);
        if (dot(wh, n) <= 0.0f || woDotWh <= 0.0f || dot(wiMirror, wh) <= 0.0f) {
            return result;
        }
        const float D = ggx_D(alpha, n, wh);
        const float G = ggx_G1(alpha, absCosThetaO) * ggx_G1(alpha, absCosThetaI);
        float cosThetaT = 0.0f;
        const float F = fresnel_dielectric_exact(woDotWh, 1.0f, max(material.typeEta.y, 1.0f), cosThetaT);
        float3 ft = float3(transmission * (1.0f - F) * D * G / max(4.0f * absCosThetaO * absCosThetaI, 1.0e-6f));
        ft *= transmission_tint(material, absCosThetaI);
        const float pdf = pTrans * (1.0f - F) * D * ggx_G1(alpha, absCosThetaO) / (4.0f * absCosThetaO);
        if (pdf > 0.0f) {
            result.value = max(ft, float3(0.0f));
            result.pdf = clamp_specular_pdf(pdf, clampParams);
            result.directionalPdf = result.pdf;
        }
        return result;
    }

    float etaI = 1.0f;
    float etaT = singleScatterSpecular ? relativeIor : max(material.typeEta.y, 1.0f);
    if (!singleScatterSpecular && (glassInterface ? !pbrFrontFace : cosThetaO < 0.0f)) {
        float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
    }
    float eta = etaI / etaT;
    float alpha = max(roughness * roughness, 1.0e-4f);
    // The VNDF sampler refracts by incident/far eta; the transmission half
    // vector is eta*wo+wi, with dwh/dwi = |wi.h|/denom^2.
    float3 wh = safe_normalize(wo * eta + wi);
    if (!all(isfinite(wh)) || dot(wh, wh) <= 0.0f) {
        return result;
    }
    if (dot(wh, normal) <= 0.0f) {
        wh = -wh;
    }
    float cosThetaOWh = dot(wo, wh);
    float cosThetaIWh = dot(wi, wh);
    if (cosThetaOWh * cosThetaIWh > 0.0f ||
        (singleScatterSpecular && (cosThetaOWh <= 0.0f || cosThetaIWh >= 0.0f))) {
        return result;
    }

    float D = ggx_D(alpha, normal, wh);
    float G = ggx_G1(alpha, absCosThetaO) * ggx_G1(alpha, absCosThetaI);
    float cosThetaT = 0.0f;
    float F = fresnel_dielectric_exact(cosThetaOWh, etaI, etaT, cosThetaT);
    float denom = eta * cosThetaOWh + cosThetaIWh;
    float denomSq = denom * denom;
    if (fabs(denomSq) <= 1.0e-8f) {
        return result;
    }
    float factor = (eta * eta) * fabs(cosThetaIWh) * fabs(cosThetaOWh);
    factor /= max(absCosThetaO * absCosThetaI * denomSq, 1.0e-6f);
    const float valueD = singleScatterSpecular && referencePunctual ? min(D, 1.0f) : D;
    float3 ft = (1.0f - F) * valueD * G * factor;
    ft *= transmission_tint(material, absCosThetaI);
    ft *= transmission;

    float pdfWh = singleScatterSpecular
        ? D * ggx_G1(alpha, absCosThetaO) * fabs(cosThetaOWh) / absCosThetaO
        : ggx_vndf_pdf(alpha, normal, wo, wh, /*halfVectorD=*/true);
    float dwhDwi = fabs(cosThetaIWh) / max(denomSq, 1.0e-8f);
    float pdfTrans = pdfWh * dwhDwi;
    if (glassInterface) {
        pdfTrans *= 1.0f - F;
    }
    float pdf = pTrans * pdfTrans;
    if (pdf > 0.0f) {
        result.value = max(ft, float3(0.0f));
        result.pdf = singleScatterSpecular ? pdf : clamp_specular_pdf(pdf, clampParams);
        result.directionalPdf = result.pdf;
    }
    return result;
}

inline BsdfSampleResult sample_pbr_metallic_roughness(const MaterialData material,
                                                      const float3 normal,
                                                      const float3 wo,
                                                      const float3 incidentDir,
                                                      const bool frontFace,
                                                      thread uint& state,
                                                      const FireflyClampParams clampParams,
                                                      const float diffuseOcclusion,
                                                      const bool specularOnly) {
    BsdfSampleResult result;
    result.direction = float3(0.0f);
    result.weight = float3(0.0f);
    result.pdf = 0.0f;
    result.isDelta = false;
    result.directionalPdf = 0.0f;
    result.areaPdf = 0.0f;
    result.exitPoint = float3(0.0f);
    result.exitNormal = float3(0.0f);
    result.isBssrdf = false;
    result.hasExitPoint = false;
    result.mediumEvent = 0;
    result.lobeType = 0u;
    result.lobeRoughness = 0.0f;
    result.lobeType = 0u;
    result.lobeRoughness = 0.0f;

    float3 baseColor = clamp01(material.baseColorRoughness.xyz);
    float metallic = clamp(material.pbrParams.x, 0.0f, 1.0f);
    float roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
    const bool singleScatterSpecular =
        (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u;
    const float relativeIor = max(material.typeEta.y, 1.0e-3f);
    float dielectricF0 = dielectric_f0_from_ior(singleScatterSpecular
        ? max(relativeIor, 1.0f / relativeIor) : material.typeEta.y);
    float3 f0 = mix(float3(dielectricF0), baseColor, metallic);
    float3 diffuseColor = baseColor * (1.0f - metallic);
    diffuseColor *= clamp(diffuseOcclusion, 0.0f, 1.0f);
    if (specularOnly) {
        diffuseColor = float3(0.0f);
    }

    float transmission = clamp(material.pbrExtras.z, 0.0f, 1.0f) * (1.0f - metallic);
    const bool glassInterface = !singleScatterSpecular && transmission > 0.0f;
    const bool thinWalled = glassInterface && pbr_thin_walled_transmission(material);
    float diffuseReflectScale = 1.0f - transmission;
    float specularReflectScale = singleScatterSpecular ? 1.0f : diffuseReflectScale;
    float specWeightBase = specularOnly ? 1.0f : pbr_specular_weight(f0);
    if (!specularOnly && (material.materialFlags & kMaterialFlagFalcorStandardBsdf) != 0u) {
        // Falcor picks lobes by albedo: Fresnel-weighted specular against diffuse,
        // so grazing reflections on dark glass are sampled often enough to converge.
        const float specAlbedo = luminance_rgb(schlick_fresnel(f0, saturate(dot(normal, wo))));
        specWeightBase = clamp(specAlbedo / max(specAlbedo + luminance_rgb(diffuseColor), 1.0e-6f),
                               0.05f, 0.95f);
    }
    float wSpec = specWeightBase * specularReflectScale;
    float wDiff = specularOnly ? 0.0f : (1.0f - specWeightBase) * diffuseReflectScale;
    // Ordinary PBR's rough reflection exports the mixture weight. A pure metal has
    // no diffuse energy; keep reference and Falcor sampling conventions intact.
    if (!singleScatterSpecular && (material.materialFlags & kMaterialFlagFalcorStandardBsdf) == 0u &&
        metallic == 1.0f && all(diffuseColor == float3(0.0f))) {
        wDiff = 0.0f;
    }
    float wTrans = transmission;
    float weightSum = wSpec + wDiff + wTrans;
    if (weightSum <= 0.0f) {
        return result;
    }
    float pSpec = wSpec / weightSum;
    float pDiff = wDiff / weightSum;
    float pTrans = wTrans / weightSum;
    float choose = rand_uniform(state);

    float3 wi = float3(0.0f);
    float pdfSpec = 0.0f;
    float pdfDiffuse = 0.0f;
    float pdfTrans = 0.0f;
    float3 f = float3(0.0f);

    if (choose < pSpec) {
        result.lobeType = 1u;
        result.lobeRoughness = roughness;
        if (roughness <= 1.0e-3f) {
            wi = reflect(incidentDir, normal);
            float cosThetaI = dot(normal, wi);
            if (cosThetaI <= 0.0f) {
                return result;
            }
            pdfSpec = 1.0f;
            float cosThetaO = max(dot(normal, wo), 0.0f);
            float3 F = schlick_fresnel(f0, cosThetaO);
            // A delta BRDF carries 1/cos(theta_i); the shared tail multiplies
            // by cos(theta_i), so the weight reduces to F * scale / pSpec.
            f = F * specularReflectScale / cosThetaI;
            result.isDelta = true;
        } else {
            float3 wh = sample_ggx_vndf(normal, wo, roughness, state);
            wi = reflect(-wo, wh);
            float cosThetaI = dot(normal, wi);
            if (cosThetaI <= 0.0f) {
                return result;
            }
            float alpha = max(roughness * roughness, 1.0e-4f);
            float D = ggx_D(alpha, normal, wh);
            float G = ggx_G1(alpha, max(dot(normal, wo), 0.0f)) * ggx_G1(alpha, cosThetaI);
            float3 F = schlick_fresnel(f0, dot(wi, wh));
            float denom = 4.0f * max(dot(normal, wo), 0.0f) * cosThetaI;
            f = F * (D * G / max(denom, 1.0e-6f));
            if (!singleScatterSpecular && (material.materialFlags & kMaterialFlagFalcorStandardBsdf) == 0u) {
                f *= specular_energy_compensation(f0, roughness, max(dot(normal, wo), 0.0f));
            }
            f = sanitize_specular_value(f);
            f *= specularReflectScale;
            pdfSpec = (singleScatterSpecular ||
                       (material.materialFlags & kMaterialFlagFalcorStandardBsdf) == 0u)
                ? D * ggx_G1(alpha, dot(normal, wo)) / (4.0f * dot(normal, wo))
                : ggx_vndf_pdf(alpha, normal, wo, wh, /*halfVectorD=*/true) / (4.0f * dot(wo, wh));
        }
    } else if (choose < (pSpec + pDiff)) {
        result.lobeType = 0u;
        result.lobeRoughness = 1.0f;
        float3 local = sample_cosine_hemisphere(state);
        wi = safe_normalize(to_world(local, normal));
        float cosThetaI = dot(normal, wi);
        if (cosThetaI <= 0.0f) {
            return result;
        }
        f = (diffuseColor / kPi) * diffuseReflectScale;
        if ((material.materialFlags & kMaterialFlagFalcorStandardBsdf) != 0u) {
            f *= frostbite_diffuse_scale(normal, wo, wi, roughness);
        }
        pdfDiffuse = lambert_pdf(normal, wi);
    } else {
        result.lobeType = 2u;
        result.lobeRoughness = roughness;
        float cosThetaO = dot(normal, wo);
        float absCosThetaO = fabs(cosThetaO);
        float etaI = 1.0f;
        float etaT = singleScatterSpecular ? relativeIor : max(material.typeEta.y, 1.0f);
        // `normal` always faces wo, so only the hit's face tells an exit.
        // A thin-walled sheet is entered from outside on either face.
        if (!frontFace && !singleScatterSpecular && !thinWalled) {
            float tmp = etaI;
            etaI = etaT;
            etaT = tmp;
        }
        float eta = etaI / etaT;
        if (glassInterface) {
            const bool deltaGlass = roughness <= 1.0e-3f;
            const float3 wh = deltaGlass ? normal : sample_ggx_vndf(normal, wo, roughness, state);
            const float woDotWh = dot(wo, wh);
            float cosThetaT = 0.0f;
            const float F = fresnel_dielectric_exact(woDotWh, etaI, etaT, cosThetaT);
            const bool reflectGlass = F >= 1.0f || rand_uniform(state) < F;
            if (thinWalled && !reflectGlass) {
                // Straight through (eta = 1); a rough sheet mirrors the
                // reflected microfacet direction through the tangent plane.
                const float3 reflected = reflect(-wo, wh);
                wi = deltaGlass ? -wo : reflected - 2.0f * dot(reflected, normal) * normal;
            } else {
                wi = reflectGlass ? reflect(-wo, wh) : refract(-wo, wh, eta);
            }
            const float dirLen2 = dot(wi, wi);
            if (!(dirLen2 > 0.0f)) {
                return result;
            }
            wi *= rsqrt(dirLen2);
            const float cosThetaI = dot(normal, wi);
            const float absCosThetaI = fabs(cosThetaI);
            if (absCosThetaI <= 0.0f || (reflectGlass ? cosThetaI <= 0.0f : cosThetaI >= 0.0f)) {
                return result;
            }
            result.lobeType = reflectGlass ? 1u : 2u;
            result.isDelta = deltaGlass;
            if (deltaGlass) {
                const float etaScale = (reflectGlass || thinWalled)
                    ? 1.0f : dielectric_continuation_eta_scale(material, eta);
                f = float3(transmission * (reflectGlass ? F : 1.0f - F) * etaScale / absCosThetaI);
                if (!reflectGlass) {
                    f *= transmission_tint(material, absCosThetaI);
                }
                pdfTrans = reflectGlass ? F : 1.0f - F;
            } else {
                const float alpha = max(roughness * roughness, 1.0e-4f);
                const float D = ggx_D(alpha, normal, wh);
                const float G = ggx_G1(alpha, absCosThetaO) * ggx_G1(alpha, absCosThetaI);
                const float pdfWh = ggx_vndf_pdf(alpha, normal, wo, wh, /*halfVectorD=*/true);
                if (reflectGlass) {
                    f = float3(transmission * F * D * G / max(4.0f * absCosThetaO * absCosThetaI, 1.0e-6f));
                    pdfTrans = F * pdfWh / (4.0f * max(woDotWh, 1.0e-6f));
                } else if (thinWalled) {
                    // The mirrored reflection keeps the reflection Jacobian 1/(4 wo.wh).
                    f = float3(transmission * (1.0f - F) * D * G / max(4.0f * absCosThetaO * absCosThetaI, 1.0e-6f)) *
                        transmission_tint(material, absCosThetaI);
                    pdfTrans = (1.0f - F) * pdfWh / (4.0f * max(woDotWh, 1.0e-6f));
                } else {
                    const float wiDotWh = dot(wi, wh);
                    const float denom = eta * woDotWh + wiDotWh;
                    const float denomSq = denom * denom;
                    if (denomSq <= 1.0e-8f) {
                        return result;
                    }
                    const float factor = eta * eta * fabs(woDotWh * wiDotWh) /
                        max(absCosThetaO * absCosThetaI * denomSq, 1.0e-6f);
                    f = float3(transmission * (1.0f - F) * D * G * factor) * transmission_tint(material, absCosThetaI);
                    pdfTrans = (1.0f - F) * pdfWh * fabs(wiDotWh) / denomSq;
                }
            }
        } else if (roughness <= 1.0e-3f) {
            wi = refract(-wo, normal, eta);
            float dirLen2 = dot(wi, wi);
            if (dirLen2 <= 0.0f) {
                return result;
            }
            wi = wi * rsqrt(dirLen2);
            float cosThetaI = dot(normal, wi);
            float cosThetaT = 0.0f;
            float Fr = fresnel_dielectric_exact(cosThetaO, etaI, etaT, cosThetaT);
            // Untagged transmission carries physical radiance, (etaI/etaT)^2, like
            // the type 2 dielectric and the rough lobe below. Tagged reference
            // materials keep their eta-free continuation.
            float etaScale = singleScatterSpecular
                ? 1.0f : dielectric_continuation_eta_scale(material, eta);
            // A delta BTDF carries 1/|cos(theta_i)|, cancelling the tail's cosine.
            float3 ft = float3(max(1.0f - Fr, 0.0f) * etaScale / max(fabs(cosThetaI), 1.0e-6f));
            ft *= transmission_tint(material, fabs(cosThetaI));
            f = transmission * ft;
            pdfTrans = 1.0f;
            result.isDelta = true;
        } else {
            float3 wh = sample_ggx_vndf(normal, wo, roughness, state);
            wi = refract(-wo, wh, eta);
            float dirLen2 = dot(wi, wi);
            if (dirLen2 <= 0.0f) {
                return result;
            }
            wi = wi * rsqrt(dirLen2);
            if (dot(wi, normal) * cosThetaO >= 0.0f) {
                return result;
            }
            float cosThetaI = dot(normal, wi);
            float absCosThetaI = fabs(cosThetaI);
            float cosThetaOWh = dot(wo, wh);
            float cosThetaIWh = dot(wi, wh);
            if (cosThetaOWh * cosThetaIWh > 0.0f ||
                (singleScatterSpecular && (cosThetaOWh <= 0.0f || cosThetaIWh >= 0.0f))) {
                return result;
            }
            float alpha = max(roughness * roughness, 1.0e-4f);
            float D = ggx_D(alpha, normal, wh);
            float G = ggx_G1(alpha, absCosThetaO) * ggx_G1(alpha, absCosThetaI);
            float cosThetaT = 0.0f;
            float F = fresnel_dielectric_exact(cosThetaOWh, etaI, etaT, cosThetaT);
            float denom = eta * cosThetaOWh + cosThetaIWh;
            float denomSq = denom * denom;
            if (fabs(denomSq) <= 1.0e-8f) {
                return result;
            }
            float factor = (eta * eta) * fabs(cosThetaIWh) * fabs(cosThetaOWh);
            factor /= max(absCosThetaO * absCosThetaI * denomSq, 1.0e-6f);
            float3 ft = (1.0f - F) * D * G * factor;
            ft *= transmission_tint(material, absCosThetaI);
            f = transmission * ft;
            float pdfWh = singleScatterSpecular
                ? D * ggx_G1(alpha, absCosThetaO) * fabs(cosThetaOWh) / absCosThetaO
                : ggx_vndf_pdf(alpha, normal, wo, wh, /*halfVectorD=*/true);
            float dwhDwi = fabs(cosThetaIWh) / max(denomSq, 1.0e-8f);
            pdfTrans = pdfWh * dwhDwi;
        }
    }

    float cosThetaI = dot(normal, wi);
    float absCosThetaI = fabs(cosThetaI);
    if (absCosThetaI <= 0.0f) {
        return result;
    }

    if (glassInterface && result.isDelta && cosThetaI > 0.0f) {
        const bool entering = frontFace || thinWalled;
        const float etaI = entering ? 1.0f : max(material.typeEta.y, 1.0f);
        const float etaT = entering ? max(material.typeEta.y, 1.0f) : 1.0f;
        float cosThetaT = 0.0f;
        const float F = fresnel_dielectric_exact(dot(normal, wo), etaI, etaT, cosThetaT);
        // Opaque and glass mirrors share the same discrete direction.
        f = ((1.0f - transmission) * schlick_fresnel(f0, dot(normal, wo)) + float3(transmission * F)) / absCosThetaI;
        pdfSpec = 1.0f;
        pdfTrans = F;
    }

    float pdf = pSpec * pdfSpec + pDiff * pdfDiffuse + pTrans * pdfTrans;
    if (pdf <= 0.0f) {
        return result;
    }
    result.direction = wi;
    result.pdf = pdf;
    result.directionalPdf = pdf;
    result.weight = max(f * absCosThetaI / pdf, float3(0.0f));
    if ((material.materialFlags & kMaterialFlagFalcorStandardBsdf) != 0u && !result.isDelta) {
        // Falcor keeps the selected lobe's throughput, but reports the whole
        // material's marginal direction density for light/BSDF MIS.
        const BsdfEvalResult eval = evaluate_pbr_metallic_roughness(material, normal, wo, wi,
                                                                   clampParams, diffuseOcclusion, specularOnly);
        result.pdf = eval.pdf;
        result.directionalPdf = eval.pdf;
    }
    if (singleScatterSpecular && !result.isDelta) {
        // The reference evaluates all lobes after choosing a direction. Return
        // the actual marginal density of our unchanged fixed-probability
        // sampler, so emitter-hit MIS agrees with direct-light evaluation.
        const BsdfEvalResult eval = evaluate_pbr_metallic_roughness(material,
                                                                   normal,
                                                                   wo,
                                                                   wi,
                                                                   clampParams,
                                                                   diffuseOcclusion,
                                                                   specularOnly);
        if (!(eval.pdf > 0.0f)) {
            result.pdf = 0.0f;
            result.weight = float3(0.0f);
            return result;
        }
        result.pdf = eval.pdf;
        result.directionalPdf = eval.pdf;
        result.weight = max(eval.value * absCosThetaI / eval.pdf, float3(0.0f));
        if (cosThetaI * dot(normal, wo) < 0.0f) {
            const float eta = 1.0f / relativeIor;
            // As with tagged water/glass, preserve the reference's eta-free
            // continuation but keep the actual normalized direction PDF.
            result.weight /= eta * eta;
        }
    }
    if (!result.isDelta && roughness > 1.0e-3f && result.lobeType < 2u &&
        cosThetaI * dot(normal, wo) > 0.0f &&
        !singleScatterSpecular && (material.materialFlags & kMaterialFlagFalcorStandardBsdf) == 0u) {
        // Export the mixture value and PDF, as Embree: the selected lobe's weight
        // gives the rare specular rays on dark walls about 100x the rest.
        const BsdfEvalResult eval = evaluate_pbr_metallic_roughness(material, normal, wo, wi,
            clampParams, diffuseOcclusion, specularOnly);
        result.pdf = eval.pdf;
        result.directionalPdf = eval.pdf;
        result.weight = eval.pdf > 0.0f ? max(eval.value * absCosThetaI / eval.pdf, float3(0.0f)) : float3(0.0f);
    }
    if (glassInterface && !result.isDelta && cosThetaI > 0.0f) {
        // Evaluate the overlapping reflected proposals with this hit's actual eta.
        // Lighting callers retain their existing entry-side assumption.
        const BsdfEvalResult eval = evaluate_pbr_metallic_roughness(material, normal, wo, wi,
            clampParams, diffuseOcclusion, specularOnly, false, frontFace);
        result.pdf = eval.pdf;
        result.directionalPdf = eval.pdf;
        result.weight = eval.pdf > 0.0f ? max(eval.value * absCosThetaI / eval.pdf, float3(0.0f)) : float3(0.0f);
    }
    return result;
}

inline BsdfEvalResult evaluate_bsdf(const MaterialData material,
                                    const float3 position,
                                    const float3 normal,
                                    const float3 wo,
                                    const float3 wi,
                                    const FireflyClampParams clampParams,
                                    const uint sssMode,
                                    const float diffuseOcclusion,
                                    const bool specularOnly,
                                    const bool referencePunctual = false) {
    BsdfEvalResult result;
    result.value = float3(0.0f);
    result.pdf = 0.0f;
    result.isDelta = false;
    result.directionalPdf = 0.0f;
    result.areaPdf = 0.0f;
    result.isBssrdf = false;

    if (source_model(material) > 0.0f) {
        return source_eval(material, normal, wo, wi, specularOnly);
    }

    if (material_is_regularized_dielectric(material)) {
        return evaluate_regularized_dielectric(material, normal, wo, wi, referencePunctual);
    }

    if (static_cast<uint>(material.typeEta.x) == 7u && (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u) {
        // Tagged PBR includes a BTDF on the opposite hemisphere. Dispatch it
        // before the generic reflection-only cosine guard below.
        return evaluate_pbr_metallic_roughness(material, normal, wo, wi, clampParams,
                                              diffuseOcclusion, specularOnly,
                                              referencePunctual);
    }

    float cosThetaO = max(dot(normal, wo), 0.0f);
    float cosThetaI = max(dot(normal, wi), 0.0f);
    // The rough thin-walled BTDF is evaluated on the opposite hemisphere.
    const bool thinWalledTransmission = cosThetaO > 0.0f && dot(normal, wi) < 0.0f &&
        pbr_thin_walled_transmission(material);
    if ((cosThetaI <= 0.0f || cosThetaO <= 0.0f) && !thinWalledTransmission) {
        return result;
    }

    uint type = static_cast<uint>(material.typeEta.x);
    switch (type) {
        case 0u: { // Lambertian
            if (specularOnly) {
                return result;
            }
            float3 albedo = material_base_color(material);
            albedo *= clamp(diffuseOcclusion, 0.0f, 1.0f);
            result.value = albedo / kPi;
            result.pdf = lambert_pdf(normal, wi);
            result.directionalPdf = result.pdf;
            break;
        }
        case 1u: { // Metal (GGX)
            float roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
            if (roughness <= 1e-3f) {
                result.isDelta = true;
                return result;
            }
            float alpha = max(roughness * roughness, 1.0e-4f);
            float3 wh = safe_normalize(wo + wi);
            if (dot(wh, normal) <= 0.0f || dot(wo, wh) <= 0.0f || dot(wi, wh) <= 0.0f) {
                return result;
            }

            float D = ggx_D(alpha, normal, wh);
            float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
            float3 eta = material.conductorEta.xyz;
            float3 k = material.conductorK.xyz;
            float3 f0 = conductor_f0(material);
            float3 F;
            if (material_has_conductor_ior(material)) {
                F = fresnel_conductor(dot(wi, wh), eta, k);
            } else {
                F = schlick_fresnel(f0, dot(wi, wh));
            }
            float denom = 4.0f * cosThetaO * cosThetaI;
            float3 spec = F * (D * G / max(denom, 1e-6f));
            spec *= specular_energy_compensation(f0, roughness, cosThetaO);
            spec = sanitize_specular_value(spec);
            float pdf = D * ggx_G1(alpha, cosThetaO) / (4.0f * cosThetaO);
            if (pdf <= 0.0f) {
                result.value = float3(0.0f);
                result.pdf = 0.0f;
                result.directionalPdf = 0.0f;
            } else {
                result.value = max(spec, float3(0.0f));
                result.pdf = clamp_specular_pdf(pdf, clampParams);
                result.directionalPdf = result.pdf;
            }
            break;
        }
        case 2u: { // Dielectric
            result.isDelta = true;
            break;
        }
        case 4u: { // Plastic (clear coat over a diffuse base), see plastic_closure_eval
            ClosureValuePdf closure =
                plastic_closure_eval(material, normal, wo, wi,
                                     plastic_lobe_probabilities(material, normal, wo, specularOnly),
                                     diffuseOcclusion, specularOnly, clampParams);
            result.value = closure.value;
            result.pdf = closure.pdf;
            result.directionalPdf = closure.pdf;
            break;
        }
        case 5u: { // Subsurface scattering (handled via BSSRDF sampling)
            result.isBssrdf = true;
            result.value = float3(0.0f);
            result.pdf = 0.0f;
            result.directionalPdf = 0.0f;
            break;
        }
        case 6u: { // CarPaint (clear coat over flakes + base), see carpaint_closure_eval
            float3 flakeNormal = carpaint_flake_normal(material, position, normal);
            ClosureValuePdf closure =
                carpaint_closure_eval(material, normal, flakeNormal, wo, wi,
                                      carpaint_lobe_probabilities(material, normal, flakeNormal, wo),
                                      clampParams);
            result.value = closure.value;
            result.pdf = closure.pdf;
            result.directionalPdf = closure.pdf;
            result.areaPdf = 0.0f;
            break;
        }
        case 7u: { // PBR Metallic-Roughness
            result = evaluate_pbr_metallic_roughness(material,
                                                     normal,
                                                     wo,
                                                     wi,
                                                     clampParams,
                                                     diffuseOcclusion,
                                                     specularOnly,
                                                     referencePunctual);
            break;
        }
        default:
            break;
    }

    if (result.pdf <= 0.0f || !all(isfinite(result.value))) {
        result.value = float3(0.0f);
    }
    return result;
}
