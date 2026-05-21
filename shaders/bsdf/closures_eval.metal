inline float safe_sqrt(const float value) {
    return sqrt(max(value, 0.0f));
}

inline float3 safe_normalize(const float3 v) {
    float len2 = dot(v, v);
    if (len2 <= 0.0f) {
        return float3(0.0f, 0.0f, 0.0f);
    }
    return v * rsqrt(len2);
}

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
    float specularTailClampBase;
    float specularTailClampRoughnessScale;
    float minSpecularPdf;
    float maxContribution;
    float enabled;
};

struct CarpaintLobeResult {
    float3 f;
    float pdf;
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
inline float3 clamp_specular_tail(const float3 value,
                                  const float roughness,
                                  const float3 f0,
                                  const FireflyClampParams params);
inline float ggx_D(const float alpha, const float cosThetaH);
inline float ggx_G1(const float alpha, const float cosTheta);
inline float ggx_pdf(const float alpha,
                     const float3 normal,
                     const float3 wo,
                     const float3 wi);
inline float3 schlick_fresnel(const float3 f0, const float cosTheta);
inline float3 material_base_color(const MaterialData material);
inline float lambert_pdf(const float3 normal, const float3 direction);
inline float3 dielectric_sigma_a(const MaterialData material) {
    return material.dielectricSigmaA.xyz;
}

inline float3 transmission_tint(const MaterialData material, const float cosTheta) {
    float thickness = max(material.typeEta.w, 0.0f);
    if (thickness <= 0.0f) {
        return float3(1.0f);
    }
    float3 sigmaA = max(dielectric_sigma_a(material), float3(0.0f));
    if (all(sigmaA <= float3(0.0f))) {
        return float3(1.0f);
    }
    float distance = thickness / max(fabs(cosTheta), 1.0e-3f);
    return clamp01(exp(-sigmaA * distance));
}

inline bool material_is_carpaint(const MaterialData material) {
    return static_cast<uint>(material.typeEta.x) == 6u;
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

inline float carpaint_flake_sample_weight(const MaterialData material) {
    return clamp(material.carpaintFlakeParams.x, 0.0f, 0.95f);
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

inline CarpaintLobeResult carpaint_eval_coat(const MaterialData material,
                                             const float3 normal,
                                             const float3 wo,
                                             const float3 wi,
                                             const FireflyClampParams clampParams) {
    CarpaintLobeResult res;
    res.f = float3(0.0f);
    res.pdf = 0.0f;
    float cosThetaO = max(dot(normal, wo), 0.0f);
    float cosThetaI = max(dot(normal, wi), 0.0f);
    if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
        return res;
    }
    float roughness = plastic_coat_roughness(material);
    float alpha = max(roughness * roughness, 1.0e-4f);
    float3 wh = safe_normalize(wo + wi);
    if (dot(wh, normal) <= 0.0f || dot(wo, wh) <= 0.0f || dot(wi, wh) <= 0.0f) {
        return res;
    }
    float D = ggx_D(alpha, dot(normal, wh));
    float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
    float f0 = plastic_coat_f0(material);
    float3 F = schlick_fresnel(float3(f0), dot(wi, wh));
    float denom = 4.0f * cosThetaO * cosThetaI;
    float3 spec = F * (D * G / max(denom, 1.0e-6f));
    spec = clamp_specular_tail(spec * plastic_specular_tint(material), roughness, float3(f0), clampParams);
    float coatPdfRaw = ggx_pdf(alpha, normal, wo, wi);
    if (coatPdfRaw <= 0.0f) {
        return res;
    }
    res.pdf = clamp_specular_pdf(coatPdfRaw, clampParams);
    res.f = spec;
    return res;
}

inline CarpaintLobeResult carpaint_eval_flake(const MaterialData material,
                                              const float3 position,
                                              const float3 normal,
                                              const float3 wo,
                                              const float3 wi,
                                              const FireflyClampParams clampParams) {
    CarpaintLobeResult res;
    res.f = float3(0.0f);
    res.pdf = 0.0f;
    float3 flakeNormal = carpaint_flake_normal(material, position, normal);
    float cosThetaO = max(dot(flakeNormal, wo), 0.0f);
    float cosThetaI = max(dot(flakeNormal, wi), 0.0f);
    if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
        return res;
    }
    float flakeRoughness = max(carpaint_flake_roughness(material), 1.0e-3f);
    float alpha = flakeRoughness * flakeRoughness;
    float3 wh = safe_normalize(wo + wi);
    if (dot(wh, flakeNormal) <= 0.0f || dot(wo, wh) <= 0.0f || dot(wi, wh) <= 0.0f) {
        return res;
    }
    float D = ggx_D(alpha, dot(flakeNormal, wh));
    float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
    float3 F0 = carpaint_base_f0(material);
    float3 F = schlick_fresnel(F0, dot(wi, wh));
    float denom = 4.0f * cosThetaO * cosThetaI;
    float3 spec = F * (D * G / max(denom, 1.0e-6f));
    spec = clamp_specular_tail(spec * plastic_specular_tint(material), flakeRoughness, F0, clampParams);
    float coatAverage = clamp(material.coatParams.w, 0.0f, 1.0f);
    spec *= max(1.0f - coatAverage, 0.0f);
    float pdfRaw = ggx_pdf(alpha, flakeNormal, wo, wi);
    if (pdfRaw <= 0.0f) {
        return res;
    }
    res.pdf = clamp_specular_pdf(pdfRaw, clampParams);
    res.f = spec;
    return res;
}

inline CarpaintLobeResult carpaint_eval_base(const MaterialData material,
                                             const float3 normal,
                                             const float3 wo,
                                             const float3 wi,
                                             const FireflyClampParams clampParams) {
    CarpaintLobeResult res;
    res.f = float3(0.0f);
    res.pdf = 0.0f;
    float cosThetaO = max(dot(normal, wo), 0.0f);
    float cosThetaI = max(dot(normal, wi), 0.0f);
    if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
        return res;
    }
    float metallic = carpaint_base_metallic(material);
    float diffuseWeight = max(1.0f - metallic, 0.0f);
    float specWeight = max(metallic, 0.0f);
    if (diffuseWeight <= 1.0e-4f && specWeight <= 1.0e-4f) {
        return res;
    }

    float coatAverage = clamp(material.coatParams.w, 0.0f, 1.0f);
    float3 baseColor = material_base_color(material);
    float3 combined = float3(0.0f);
    float pdfDiffuse = 0.0f;
    float pdfSpec = 0.0f;

    if (diffuseWeight > 1.0e-4f) {
        float3 diffuse = baseColor / kPi;
        float3 coatTrans = plastic_diffuse_transmission(material, cosThetaI, cosThetaO);
        diffuse *= coatTrans * max(1.0f - coatAverage, 0.0f);
        diffuse = max(diffuse, float3(0.0f));
        combined += diffuseWeight * diffuse;
        pdfDiffuse = lambert_pdf(normal, wi);
    }

    if (specWeight > 1.0e-4f) {
        float roughness = max(carpaint_base_roughness(material), 1.0e-3f);
        float alpha = roughness * roughness;
        float3 wh = safe_normalize(wo + wi);
        if (dot(wh, normal) > 0.0f && dot(wo, wh) > 0.0f && dot(wi, wh) > 0.0f) {
            float D = ggx_D(alpha, dot(normal, wh));
            float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
            bool hasConductor = carpaint_has_base_conductor(material);
            float3 eta = carpaint_base_eta(material);
            float3 k = carpaint_base_k(material);
            float3 f0 = hasConductor ? fresnel_conductor(1.0f, eta, k)
                                     : clamp01(baseColor);
            float3 F = hasConductor ? fresnel_conductor(dot(wi, wh), eta, k)
                                    : schlick_fresnel(baseColor, dot(wi, wh));
            float denom = 4.0f * cosThetaO * cosThetaI;
            float3 spec = F * (D * G / max(denom, 1.0e-6f));
            spec = clamp_specular_tail(spec * plastic_specular_tint(material) * max(1.0f - coatAverage, 0.0f),
                                       roughness,
                                       f0,
                                       clampParams);
            spec = max(spec, float3(0.0f));
            combined += specWeight * spec;
            float pdfRaw = ggx_pdf(alpha, normal, wo, wi);
            if (pdfRaw > 0.0f) {
                pdfSpec = clamp_specular_pdf(pdfRaw, clampParams);
            }
        }
    }

    res.f = max(combined, float3(0.0f));
    res.pdf = diffuseWeight * pdfDiffuse + specWeight * pdfSpec;
    return res;
}

inline FireflyClampParams make_firefly_params(constant PathtraceUniforms& uniforms) {
    FireflyClampParams params;
    params.mode = uniforms.fireflyClampMode;
    params.clampFactor = max(uniforms.fireflyClampFactor, 0.0f);
    params.clampFloor = max(uniforms.fireflyClampFloor, 0.0f);
    params.throughputClamp = max(uniforms.throughputClamp, 0.0f);
    params.specularTailClampBase = max(uniforms.specularTailClampBase, 0.0f);
    params.specularTailClampRoughnessScale = max(uniforms.specularTailClampRoughnessScale, 0.0f);
    params.minSpecularPdf = max(uniforms.minSpecularPdf, 0.0f);
    params.maxContribution = max(uniforms.fireflyClampMaxContribution, 0.0f);
    params.enabled = (uniforms.fireflyClampEnabled != 0u) ? 1.0f : 0.0f;
    return params;
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
        float lum = luminance_rgb(positive);
        float throughputLum = luminance_rgb(max(throughput, float3(0.0f)));
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
    float lum = luminance_rgb(positive);
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

inline float3 clamp_specular_tail(const float3 value,
                                  const float roughness,
                                  const float3 f0,
                                  const FireflyClampParams params) {
    if (!all(isfinite(value))) {
        return float3(0.0f);
    }
    float3 positive = max(value, float3(0.0f));
    if (params.enabled < 0.5f) {
        return positive;
    }
    if (params.specularTailClampBase <= 0.0f && params.specularTailClampRoughnessScale <= 0.0f) {
        return positive;
    }
    float strength = max(max(f0.x, f0.y), f0.z);
    strength = max(strength, 1e-3f);
    float limit = (params.specularTailClampBase +
                   params.specularTailClampRoughnessScale * roughness) * strength;
    limit = max(limit, params.clampFloor);
    float lum = luminance_rgb(positive);
    if (lum > limit && lum > 0.0f) {
        float scale = limit / max(lum, 1e-6f);
        positive *= scale;
    }
    return positive;
}

inline float schlick_weight(const float cosTheta) {
    float m = clamp01(1.0f - cosTheta);
    float m2 = m * m;
    return m2 * m2 * m;
}

inline float3 schlick_fresnel(const float3 f0, const float cosTheta) {
    return f0 + (float3(1.0f) - f0) * schlick_weight(cosTheta);
}

inline float fresnel_dielectric_exact(float cosThetaI,
                                      float etaI,
                                      float etaT,
                                      thread float& outCosThetaT) {
    cosThetaI = clamp(cosThetaI, -1.0f, 1.0f);
    float absCosThetaI = fabs(cosThetaI);
    float sinThetaI2 = max(0.0f, 1.0f - absCosThetaI * absCosThetaI);
    float eta = etaI / etaT;
    float sinThetaT2 = eta * eta * sinThetaI2;

    if (sinThetaT2 >= 1.0f) {
        outCosThetaT = 0.0f;
        return 1.0f;  // Total internal reflection
    }

    float cosThetaT = safe_sqrt(1.0f - sinThetaT2);
    outCosThetaT = cosThetaT;

    float etaICosThetaI = etaI * absCosThetaI;
    float etaTCosThetaT = etaT * cosThetaT;

    float RsNum = etaICosThetaI - etaTCosThetaT;
    float RsDen = etaICosThetaI + etaTCosThetaT;
    float RpNum = etaT * absCosThetaI - etaI * cosThetaT;
    float RpDen = etaT * absCosThetaI + etaI * cosThetaT;

    float Rs = (RsNum / RsDen);
    float Rp = (RpNum / RpDen);
    return 0.5f * (Rs * Rs + Rp * Rp);
}

inline float3 fresnel_conductor(float cosThetaI, const float3 eta, const float3 k) {
    cosThetaI = clamp(cosThetaI, -1.0f, 1.0f);
    float cos2 = cosThetaI * cosThetaI;
    float sin2 = max(0.0f, 1.0f - cos2);

    float3 eta2 = eta * eta;
    float3 k2 = k * k;

    float3 t0 = eta2 - k2 - float3(sin2);
    float3 a2plusb2 = sqrt(max(t0 * t0 + 4.0f * eta2 * k2, float3(0.0f)));
    float3 a = sqrt(max(0.5f * (a2plusb2 + t0), float3(0.0f)));

    float3 term1 = a2plusb2 + float3(cos2);
    float3 term2 = 2.0f * float3(cosThetaI) * a;
    float3 Rs = (term1 - term2) / (term1 + term2);

    float3 term3 = float3(cos2) * a2plusb2 + float3(sin2 * sin2);
    float3 term4 = term2 * float3(sin2);
    float3 Rp = (term3 - term4) / (term3 + term4);

    return clamp01(0.5f * (Rs * Rs + Rp * Rp));
}

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

inline float ggx_pdf(const float alpha,
                     const float3 normal,
                     const float3 wo,
                     const float3 wi) {
    float3 wh = safe_normalize(wo + wi);
    float cosThetaH = dot(normal, wh);
    float dotWoWh = dot(wo, wh);
    float cosThetaO = dot(normal, wo);
    if (cosThetaO <= 0.0f || cosThetaH <= 0.0f || dotWoWh <= 0.0f) {
        return 0.0f;
    }
    float D = ggx_D(alpha, cosThetaH);
    float G1 = ggx_G1(alpha, cosThetaO);
    float denom = 4.0f * max(dotWoWh, 1e-6f);
    return D * G1 * cosThetaH / denom;
}

inline float ggx_vndf_pdf(const float alpha,
                          const float3 normal,
                          const float3 wo,
                          const float3 wh) {
    float cosThetaO = dot(normal, wo);
    float cosThetaH = dot(normal, wh);
    if (cosThetaO <= 0.0f || cosThetaH <= 0.0f) {
        return 0.0f;
    }
    float D = ggx_D(alpha, cosThetaH);
    float G1 = ggx_G1(alpha, cosThetaO);
    float denom = max(dot(wo, wh), 1.0e-6f);
    return D * G1 * cosThetaH / denom;
}

inline float3 sample_ggx_half_vector(const float3 normal,
                                     const float alpha,
                                     thread uint& state) {
    float u1 = rand_uniform(state);
    float u2 = rand_uniform(state);

    float phi = 2.0f * kPi * u1;
    float cosTheta = sqrt((1.0f - u2) / (1.0f + (alpha * alpha - 1.0f) * u2));
    float sinTheta = safe_sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));

    float3 hLocal = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    return safe_normalize(to_world(hLocal, normal));
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

inline float3 material_base_color(const MaterialData material) {
    return clamp01(material.baseColorRoughness.xyz);
}

inline float material_roughness(const MaterialData material) {
    float roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
    return max(roughness, 1e-3f);
}

inline bool material_has_conductor_ior(const MaterialData material) {
    return (material.conductorEta.w > 0.0f || material.conductorK.w > 0.0f ||
            any(material.conductorEta.xyz > float3(0.0f)) ||
            any(material.conductorK.xyz > float3(0.0f)));
}

inline float3 conductor_f0(const MaterialData material) {
    if (material_has_conductor_ior(material)) {
        return fresnel_conductor(1.0f, material.conductorEta.xyz, material.conductorK.xyz);
    }
    return clamp01(material_base_color(material));
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

inline float environment_lighting_roughness(const MaterialData material) {
    uint type = static_cast<uint>(material.typeEta.x);
    switch (type) {
        case 1u: // Metal
        case 7u: // PBR Metallic-Roughness
            return clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        case 4u: // Plastic (use coat roughness for env lighting)
            return clamp(plastic_coat_roughness(material), 0.0f, 1.0f);
        case 6u: // CarPaint (use base roughness)
            return clamp(carpaint_base_roughness(material), 0.0f, 1.0f);
        default:
            return 1.0f;
    }
}

inline float plastic_coat_thickness(const MaterialData material) {
    return max(material.coatParams.y, 0.0f);
}

inline float plastic_coat_sample_weight(const MaterialData material) {
    return clamp(material.coatParams.z, 0.0f, 1.0f);
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

struct BsdfEvalResult {
    float3 value;
    float pdf;
    float directionalPdf;
    float areaPdf;
    bool isDelta;
    bool isBssrdf;
};

struct BsdfSampleResult {
    float3 direction;
    float3 weight;
    float pdf;
    float directionalPdf;
    float areaPdf;
    float3 exitPoint;
    float3 exitNormal;
    bool isDelta;
    bool isBssrdf;
    bool hasExitPoint;
    int mediumEvent;
    uint lobeType;
    float lobeRoughness;
};

inline BsdfSampleResult sample_sss_random_walk_software(constant PathtraceUniforms& uniforms,
                                                        const MaterialData material,
                                                        thread const HitRecord& rec,
                                                        const float3 wo,
                                                        const float3 incidentDir,
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
                                                        device PathtraceStats* stats,
                                                        thread uint& state,
                                                        const FireflyClampParams clampParams) {
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
    result.lobeType = 0u;
    result.lobeRoughness = 0.0f;

    if (rec.frontFace == 0u) {
        return result;
    }

    float pCoat = clamp(material.coatParams.z, 0.0f, 1.0f);
    float randLobe = rand_uniform(state);
    float coatRoughness = plastic_coat_roughness(material);
    float alpha = coatRoughness * coatRoughness;
    float f0 = plastic_coat_f0(material);
    float3 f0Color = float3(f0);
    float3 specTint = plastic_specular_tint(material);

    if (pCoat > 0.0f && randLobe < pCoat) {
        float3 wh = sample_ggx_vndf(rec.normal, wo, coatRoughness, state);
        if (dot(wh, rec.normal) <= 0.0f) {
            return result;
        }

        float3 wi = reflect(-wo, wh);
        wi = safe_normalize(wi);
        if (!all(isfinite(wi))) {
            return result;
        }

        float cosThetaI = dot(rec.normal, wi);
        float cosThetaO = dot(rec.normal, wo);
        if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
            return result;
        }

        float dotWiWh = dot(wi, wh);
        if (dotWiWh <= 0.0f) {
            return result;
        }

        float D = ggx_D(alpha, dot(rec.normal, wh));
        float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
        float3 F = schlick_fresnel(f0Color, dotWiWh);
        float denom = 4.0f * cosThetaO * cosThetaI;
        float3 spec = F * (D * G / max(denom, 1.0e-6f));
        spec = clamp_specular_tail(spec * specTint, coatRoughness, f0Color, clampParams);
        float specPdfRaw = ggx_pdf(alpha, rec.normal, wo, wi);
        if (specPdfRaw <= 0.0f) {
            return result;
        }
        float specPdf = clamp_specular_pdf(specPdfRaw, clampParams);
        float combinedPdf = max(pCoat * specPdf, 1.0e-6f);
        float3 weight = spec * cosThetaI / combinedPdf;
        weight = max(weight, float3(0.0f));
        if (!all(isfinite(weight))) {
            return result;
        }

        result.direction = wi;
        result.weight = weight;
        result.pdf = combinedPdf;
        result.directionalPdf = specPdf;
        result.areaPdf = 0.0f;
        result.isDelta = false;
        result.isBssrdf = false;
        result.hasExitPoint = false;
        return result;
    }

    float pDiffuse = max(1.0f - pCoat, 1.0e-3f);

    float anisotropy = clamp(material.sssSigmaS.w, -0.99f, 0.99f);
    float meanFreePath = max(material.sssParams.x, 1.0e-4f);
    float3 baseColor = material_base_color(material);
    float3 sigmaA = sss_sigma_a(material, baseColor, meanFreePath, anisotropy);
    float3 sigmaSPrime = sss_sigma_s_prime(material, baseColor, meanFreePath, anisotropy);
    float3 sigmaT = max(sigmaA + sigmaSPrime, float3(1.0e-6f));
    float sigmaTScalar = max(max(sigmaT.x, max(sigmaT.y, sigmaT.z)), 1.0e-4f);

    float3 throughput = float3(1.0f / pDiffuse);

    float etaOutside = 1.0f;
    float etaInside = max(material.typeEta.y, 1.0f);
    float3 entryNormal = rec.normal;
    float3 unitDir = incidentDir;
    float cosThetaI = dot(-unitDir, entryNormal);
    if (cosThetaI <= 0.0f) {
        return result;
    }
    float cosThetaT = 0.0f;
    float FrEntry = fresnel_dielectric_exact(cosThetaI, etaOutside, etaInside, cosThetaT);
    float3 enterDir = refract(unitDir, entryNormal, etaOutside / etaInside);
    if (!all(isfinite(enterDir)) || dot(enterDir, enterDir) <= 0.0f) {
        return result;
    }
    enterDir = safe_normalize(enterDir);

    float etaScaleEntry = (etaInside * etaInside) / (etaOutside * etaOutside);
    float directionScaleEntry = etaScaleEntry * (cosThetaT / max(cosThetaI, 1.0e-6f));
    throughput *= max(1.0f - FrEntry, 0.0f) * directionScaleEntry;
    if (material.sssParams.z > 0.5f) {
        throughput *= plastic_specular_tint(material);
    }

    float3 currentPos = offset_surface_point(rec.point, -entryNormal, enterDir);
    float3 currentDir = enterDir;

    uint maxSteps = max(uniforms.sssMaxSteps, 1u);
    for (uint step = 0u; step < maxSteps; ++step) {
        float xi = rand_uniform(state);
        xi = clamp(xi, 1.0e-6f, 1.0f - 1.0e-6f);
        float distance = -log(1.0f - xi) / sigmaTScalar;

        Ray boundaryRay;
        boundaryRay.origin = currentPos;
        boundaryRay.direction = currentDir;
        HitRecord boundaryRec;
        bool hitBoundary = trace_scene_software(uniforms,
                                                spheres,
                                                rectangles,
                                                triangleData,
                                                tlasNodes,
                                                tlasPrimIndices,
                                                instanceInfos,
                                                blasNodes,
                                                blasPrimIndices,
                                                nodes,
                                                primitiveIndices,
                                                stats,
                                                boundaryRay,
                                                kRayOriginEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/false,
                                                /*includeTriangles=*/true,
                                                boundaryRec);
        if (!hitBoundary) {
            break;
        }

        float boundaryDistance = max(boundaryRec.t, 1.0e-4f);
        if (distance < boundaryDistance) {
            float3 transmittance = exp(-sigmaT * distance);
            throughput *= transmittance;
            float3 scatterAlbedo = clamp01(sigmaSPrime / max(sigmaT, float3(1.0e-6f)));
            throughput *= scatterAlbedo;
            float throughputMax = max(throughput.x, max(throughput.y, throughput.z));
            if (throughputMax < kSssThroughputCutoff) {
                break;
            }
            currentPos += currentDir * distance;
            currentDir = sample_henyey_greenstein_world(-currentDir, anisotropy, state);
            if (!all(isfinite(currentDir)) || dot(currentDir, currentDir) <= 0.0f) {
                break;
            }
            currentDir = safe_normalize(currentDir);
            continue;
        }

        float travel = boundaryDistance;
        float3 transmittance = exp(-sigmaT * travel);
        throughput *= transmittance;
        float throughputMax = max(throughput.x, max(throughput.y, throughput.z));
        if (throughputMax < kSssThroughputCutoff) {
            break;
        }

        float3 exitPoint = boundaryRec.point;
        float3 outwardNormal = (boundaryRec.frontFace != 0u) ? boundaryRec.normal : -boundaryRec.normal;
        if (!all(isfinite(outwardNormal)) || dot(outwardNormal, outwardNormal) <= 0.0f) {
            break;
        }
        outwardNormal = safe_normalize(outwardNormal);

        float etaI = etaInside;
        float etaT = 1.0f;
        float cosExitI = dot(-currentDir, outwardNormal);
        if (cosExitI <= 0.0f) {
            currentPos = exitPoint;
            currentDir = reflect(currentDir, outwardNormal);
            currentDir = safe_normalize(currentDir);
            continue;
        }

        float cosExitT = 0.0f;
        float FrExit = fresnel_dielectric_exact(cosExitI, etaI, etaT, cosExitT);
        float3 refracted = refract(currentDir, outwardNormal, etaI / etaT);
        if (!all(isfinite(refracted)) || dot(refracted, refracted) <= 0.0f) {
            currentPos = exitPoint;
            currentDir = reflect(currentDir, outwardNormal);
            currentDir = safe_normalize(currentDir);
            continue;
        }
        refracted = safe_normalize(refracted);

        float etaScaleExit = (etaT * etaT) / (etaI * etaI);
        float directionScaleExit = etaScaleExit * (cosExitT / max(cosExitI, 1.0e-6f));
        throughput *= max(1.0f - FrExit, 0.0f) * directionScaleExit;
        if (material.sssParams.z > 0.5f) {
            throughput *= plastic_specular_tint(material);
        }
        throughput = max(throughput, float3(0.0f));
        if (!all(isfinite(throughput))) {
            break;
        }

        result.direction = refracted;
        result.weight = throughput;
        result.pdf = max(pDiffuse, 1.0e-4f);
        result.directionalPdf = 1.0f;
        result.areaPdf = 0.0f;
        result.exitPoint = exitPoint;
        result.exitNormal = outwardNormal;
        result.isDelta = false;
        result.isBssrdf = true;
        result.hasExitPoint = true;
        return result;
    }

    result.pdf = 0.0f;
    result.direction = float3(0.0f);
    result.weight = float3(0.0f);
    return result;
}

#if __METAL_VERSION__ >= 310
inline BsdfSampleResult sample_sss_random_walk_hardware(constant PathtraceUniforms& uniforms,
                                                        const MaterialData material,
                                                        thread const HitRecord& rec,
                                                        const float3 wo,
                                                        const float3 incidentDir,
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
                                                        const FireflyClampParams clampParams) {
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
    result.lobeType = 0u;
    result.lobeRoughness = 0.0f;

    if (rec.frontFace == 0u) {
        return result;
    }

    float pCoat = clamp(material.coatParams.z, 0.0f, 1.0f);
    float randLobe = rand_uniform(state);
    float coatRoughness = plastic_coat_roughness(material);
    float alpha = coatRoughness * coatRoughness;
    float f0 = plastic_coat_f0(material);
    float3 f0Color = float3(f0);
    float3 specTint = plastic_specular_tint(material);

    if (pCoat > 0.0f && randLobe < pCoat) {
        float3 wh = sample_ggx_vndf(rec.normal, wo, coatRoughness, state);
        if (dot(wh, rec.normal) <= 0.0f) {
            return result;
        }

        float3 wi = reflect(-wo, wh);
        wi = safe_normalize(wi);
        if (!all(isfinite(wi))) {
            return result;
        }

        float cosThetaI = dot(rec.normal, wi);
        float cosThetaO = dot(rec.normal, wo);
        if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
            return result;
        }

        float dotWiWh = dot(wi, wh);
        if (dotWiWh <= 0.0f) {
            return result;
        }

        float D = ggx_D(alpha, dot(rec.normal, wh));
        float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
        float3 F = schlick_fresnel(f0Color, dotWiWh);
        float denom = 4.0f * cosThetaO * cosThetaI;
        float3 spec = F * (D * G / max(denom, 1.0e-6f));
        spec = clamp_specular_tail(spec * specTint, coatRoughness, f0Color, clampParams);
        float specPdfRaw = ggx_pdf(alpha, rec.normal, wo, wi);
        if (specPdfRaw <= 0.0f) {
            return result;
        }
        float specPdf = clamp_specular_pdf(specPdfRaw, clampParams);
        float combinedPdf = max(pCoat * specPdf, 1.0e-6f);
        float3 weight = spec * cosThetaI / combinedPdf;
        weight = max(weight, float3(0.0f));
        if (!all(isfinite(weight))) {
            return result;
        }

        result.direction = wi;
        result.weight = weight;
        result.pdf = combinedPdf;
        result.directionalPdf = specPdf;
        result.areaPdf = 0.0f;
        result.isDelta = false;
        result.isBssrdf = false;
        result.hasExitPoint = false;
        return result;
    }

    float pDiffuse = max(1.0f - pCoat, 1.0e-3f);

    float anisotropy = clamp(material.sssSigmaS.w, -0.99f, 0.99f);
    float meanFreePath = max(material.sssParams.x, 1.0e-4f);
    float3 baseColor = material_base_color(material);
    float3 sigmaA = sss_sigma_a(material, baseColor, meanFreePath, anisotropy);
    float3 sigmaSPrime = sss_sigma_s_prime(material, baseColor, meanFreePath, anisotropy);
    float3 sigmaT = max(sigmaA + sigmaSPrime, float3(1.0e-6f));
    float sigmaTScalar = max(max(sigmaT.x, max(sigmaT.y, sigmaT.z)), 1.0e-4f);

    float3 throughput = float3(1.0f / pDiffuse);

    float etaOutside = 1.0f;
    float etaInside = max(material.typeEta.y, 1.0f);
    float3 entryNormal = rec.normal;
    float3 unitDir = incidentDir;
    float cosThetaI = dot(-unitDir, entryNormal);
    if (cosThetaI <= 0.0f) {
        return result;
    }
    float cosThetaT = 0.0f;
    float FrEntry = fresnel_dielectric_exact(cosThetaI, etaOutside, etaInside, cosThetaT);
    float3 enterDir = refract(unitDir, entryNormal, etaOutside / etaInside);
    if (!all(isfinite(enterDir)) || dot(enterDir, enterDir) <= 0.0f) {
        return result;
    }
    enterDir = safe_normalize(enterDir);

    float etaScaleEntry = (etaInside * etaInside) / (etaOutside * etaOutside);
    float directionScaleEntry = etaScaleEntry * (cosThetaT / max(cosThetaI, 1.0e-6f));
    throughput *= max(1.0f - FrEntry, 0.0f) * directionScaleEntry;
    if (material.sssParams.z > 0.5f) {
        throughput *= plastic_specular_tint(material);
    }

    float3 currentPos = offset_surface_point(rec.point, -entryNormal, enterDir);
    float3 currentDir = enterDir;

    uint maxSteps = max(uniforms.sssMaxSteps, 1u);
    for (uint step = 0u; step < maxSteps; ++step) {
        float xi = rand_uniform(state);
        xi = clamp(xi, 1.0e-6f, 1.0f - 1.0e-6f);
        float distance = -log(1.0f - xi) / sigmaTScalar;

        Ray boundaryRay;
        boundaryRay.origin = currentPos;
        boundaryRay.direction = currentDir;
        HitRecord boundaryRec;
        uint excludeMesh;
        uint excludePrim;
        compute_exclusion_indices(rec, excludeMesh, excludePrim);
        bool hitBoundary = trace_scene_hardware(uniforms,
                                                accel,
                                                meshInfos,
                                                triangleData,
                                                sceneVertices,
                                                meshIndices,
                                                instanceUserIds,
                                                spheres,
                                                rectangles,
                                                nodes,
                                                primitiveIndices,
                                                stats,
                                                boundaryRay,
                                                kRayOriginEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/false,
                                                excludeMesh,
                                                excludePrim,
                                                boundaryRec);
        if (!hitBoundary) {
            break;
        }

        float boundaryDistance = max(boundaryRec.t, 1.0e-4f);
        if (distance < boundaryDistance) {
            float3 transmittance = exp(-sigmaT * distance);
            throughput *= transmittance;
            float3 scatterAlbedo = clamp01(sigmaSPrime / max(sigmaT, float3(1.0e-6f)));
            throughput *= scatterAlbedo;
            float throughputMax = max(throughput.x, max(throughput.y, throughput.z));
            if (throughputMax < kSssThroughputCutoff) {
                break;
            }
            currentPos += currentDir * distance;
            currentDir = sample_henyey_greenstein_world(-currentDir, anisotropy, state);
            if (!all(isfinite(currentDir)) || dot(currentDir, currentDir) <= 0.0f) {
                break;
            }
            currentDir = safe_normalize(currentDir);
            continue;
        }

        float travel = boundaryDistance;
        float3 transmittance = exp(-sigmaT * travel);
        throughput *= transmittance;
        float throughputMax = max(throughput.x, max(throughput.y, throughput.z));
        if (throughputMax < kSssThroughputCutoff) {
            break;
        }

        float3 exitPoint = boundaryRec.point;
        float3 outwardNormal = (boundaryRec.frontFace != 0u) ? boundaryRec.normal : -boundaryRec.normal;
        if (!all(isfinite(outwardNormal)) || dot(outwardNormal, outwardNormal) <= 0.0f) {
            break;
        }
        outwardNormal = safe_normalize(outwardNormal);

        float etaI = etaInside;
        float etaT = 1.0f;
        float cosExitI = dot(-currentDir, outwardNormal);
        if (cosExitI <= 0.0f) {
            currentPos = exitPoint;
            currentDir = reflect(currentDir, outwardNormal);
            currentDir = safe_normalize(currentDir);
            continue;
        }

        float cosExitT = 0.0f;
        float FrExit = fresnel_dielectric_exact(cosExitI, etaI, etaT, cosExitT);
        float3 refracted = refract(currentDir, outwardNormal, etaI / etaT);
        if (!all(isfinite(refracted)) || dot(refracted, refracted) <= 0.0f) {
            currentPos = exitPoint;
            currentDir = reflect(currentDir, outwardNormal);
            currentDir = safe_normalize(currentDir);
            continue;
        }
        refracted = safe_normalize(refracted);

        float etaScaleExit = (etaT * etaT) / (etaI * etaI);
        float directionScaleExit = etaScaleExit * (cosExitT / max(cosExitI, 1.0e-6f));
        throughput *= max(1.0f - FrExit, 0.0f) * directionScaleExit;
        if (material.sssParams.z > 0.5f) {
            throughput *= plastic_specular_tint(material);
        }
        throughput = max(throughput, float3(0.0f));
        if (!all(isfinite(throughput))) {
            break;
        }

        result.direction = refracted;
        result.weight = throughput;
        result.pdf = max(pDiffuse, 1.0e-4f);
        result.directionalPdf = 1.0f;
        result.areaPdf = 0.0f;
        result.exitPoint = exitPoint;
        result.exitNormal = outwardNormal;
        result.isDelta = false;
        result.isBssrdf = true;
        result.hasExitPoint = true;
        return result;
    }

    result.pdf = 0.0f;
    result.direction = float3(0.0f);
    result.weight = float3(0.0f);
    return result;
}
#endif

inline bool material_is_delta(const MaterialData material) {
    uint type = static_cast<uint>(material.typeEta.x);
    if (type == 2u) {
        return true;
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

inline bool material_is_thin_dielectric(const MaterialData material) {
    uint type = static_cast<uint>(material.typeEta.x);
    return (type == 2u) && (material.typeEta.w > 0.5f);
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

inline float pbr_specular_weight(const float3 f0) {
    return clamp(max_component(f0), 0.05f, 0.95f);
}

inline float2 dfg_approx(const float roughness, const float NoV) {
    const float4 c0 = float4(-1.0f, -0.0275f, -0.572f, 0.022f);
    const float4 c1 = float4(1.0f, 0.0425f, 1.04f, -0.04f);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28f * NoV)) * r.x + r.y;
    return float2(-1.04f, 1.04f) * a004 + r.zw;
}

inline float3 specular_energy_compensation(const float3 f0,
                                           const float roughness,
                                           const float NoV) {
    float NoVClamped = clamp(NoV, 0.0f, 1.0f);
    float2 dfg = dfg_approx(roughness, NoVClamped);
    float3 Fss = clamp(f0 * dfg.x + dfg.y, float3(0.0f), float3(0.99f));
    float3 Favg = f0 + (float3(1.0f) - f0) * (1.0f / 21.0f);
    float3 oneMinusFss = clamp(float3(1.0f) - Fss, float3(0.0f), float3(1.0f));
    float3 denom = max(float3(1.0f) - Favg * oneMinusFss, float3(1.0e-3f));
    float3 Fms = (Favg * oneMinusFss) / denom;
    float3 scale = (Fss + Fms) / max(Fss, float3(1.0e-4f));
    return clamp(scale, float3(1.0f), float3(2.0f));
}

inline BsdfEvalResult evaluate_pbr_metallic_roughness(const MaterialData material,
                                                      const float3 normal,
                                                      const float3 wo,
                                                      const float3 wi,
                                                      const FireflyClampParams clampParams,
                                                      const float diffuseOcclusion,
                                                      const bool specularOnly) {
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
    float dielectricF0 = dielectric_f0_from_ior(material.typeEta.y);
    float3 f0 = mix(float3(dielectricF0), baseColor, metallic);
    float3 diffuseColor = baseColor * (1.0f - metallic);
    diffuseColor *= clamp(diffuseOcclusion, 0.0f, 1.0f);
    if (specularOnly) {
        diffuseColor = float3(0.0f);
    }

    float transmission = clamp(material.pbrExtras.z, 0.0f, 1.0f) * (1.0f - metallic);
    float reflectScale = 1.0f - transmission;
    float specWeightBase = specularOnly ? 1.0f : pbr_specular_weight(f0);
    float wSpec = specWeightBase * reflectScale;
    float wDiff = specularOnly ? 0.0f : (1.0f - specWeightBase) * reflectScale;
    float wTrans = transmission;
    float weightSum = wSpec + wDiff + wTrans;
    if (weightSum <= 0.0f) {
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
        float D = ggx_D(alpha, dot(normal, wh));
        float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
        float3 F = schlick_fresnel(f0, dot(wi, wh));
        float denom = 4.0f * cosThetaO * cosThetaI;
        float3 spec = F * (D * G / max(denom, 1.0e-6f));
        spec *= specular_energy_compensation(f0, roughness, absCosThetaO);
        spec = clamp_specular_tail(spec, roughness, f0, clampParams);
        spec *= reflectScale;
        float pdfSpec = ggx_pdf(alpha, normal, wo, wi);

        float3 diffuse = (diffuseColor / kPi) * reflectScale;
        float pdfDiffuse = lambert_pdf(normal, wi);

        float pdf = pSpec * pdfSpec + pDiff * pdfDiffuse;
        if (pdf > 0.0f) {
            result.value = max(spec + diffuse, float3(0.0f));
            result.pdf = clamp_specular_pdf(pdf, clampParams);
            result.directionalPdf = result.pdf;
        }
        return result;
    }

    if (wTrans <= 0.0f) {
        return result;
    }

    float etaI = 1.0f;
    float etaT = max(material.typeEta.y, 1.0f);
    if (cosThetaO < 0.0f) {
        float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
    }
    float eta = etaI / etaT;
    float alpha = max(roughness * roughness, 1.0e-4f);
    float3 wh = safe_normalize(wo + wi * eta);
    if (!all(isfinite(wh)) || dot(wh, wh) <= 0.0f) {
        return result;
    }
    if (dot(wh, normal) <= 0.0f) {
        wh = -wh;
    }
    float cosThetaOWh = dot(wo, wh);
    float cosThetaIWh = dot(wi, wh);
    if (cosThetaOWh * cosThetaIWh > 0.0f) {
        return result;
    }

    float D = ggx_D(alpha, max(dot(normal, wh), 0.0f));
    float G = ggx_G1(alpha, absCosThetaO) * ggx_G1(alpha, absCosThetaI);
    float cosThetaT = 0.0f;
    float F = fresnel_dielectric_exact(cosThetaOWh, etaI, etaT, cosThetaT);
    float denom = cosThetaOWh + eta * cosThetaIWh;
    float denomSq = denom * denom;
    if (fabs(denomSq) <= 1.0e-8f) {
        return result;
    }
    float factor = (eta * eta) * fabs(cosThetaIWh) * fabs(cosThetaOWh);
    factor /= max(absCosThetaO * absCosThetaI * denomSq, 1.0e-6f);
    float3 ft = (1.0f - F) * D * G * factor;
    ft *= transmission_tint(material, absCosThetaI);
    ft *= transmission;

    float pdfWh = ggx_vndf_pdf(alpha, normal, wo, wh);
    float dwhDwi = fabs((eta * eta * cosThetaIWh) / max(denomSq, 1.0e-8f));
    float pdfTrans = pdfWh * dwhDwi;
    float pdf = pTrans * pdfTrans;
    if (pdf > 0.0f) {
        result.value = max(ft, float3(0.0f));
        result.pdf = clamp_specular_pdf(pdf, clampParams);
        result.directionalPdf = result.pdf;
    }
    return result;
}

inline BsdfSampleResult sample_pbr_metallic_roughness(const MaterialData material,
                                                      const float3 normal,
                                                      const float3 wo,
                                                      const float3 incidentDir,
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
    float dielectricF0 = dielectric_f0_from_ior(material.typeEta.y);
    float3 f0 = mix(float3(dielectricF0), baseColor, metallic);
    float3 diffuseColor = baseColor * (1.0f - metallic);
    diffuseColor *= clamp(diffuseOcclusion, 0.0f, 1.0f);
    if (specularOnly) {
        diffuseColor = float3(0.0f);
    }

    float transmission = clamp(material.pbrExtras.z, 0.0f, 1.0f) * (1.0f - metallic);
    float reflectScale = 1.0f - transmission;
    float specWeightBase = specularOnly ? 1.0f : pbr_specular_weight(f0);
    float wSpec = specWeightBase * reflectScale;
    float wDiff = specularOnly ? 0.0f : (1.0f - specWeightBase) * reflectScale;
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
            f = F * reflectScale;
            result.isDelta = true;
        } else {
            float3 wh = sample_ggx_vndf(normal, wo, roughness, state);
            wi = reflect(-wo, wh);
            float cosThetaI = dot(normal, wi);
            if (cosThetaI <= 0.0f) {
                return result;
            }
            float alpha = max(roughness * roughness, 1.0e-4f);
            float D = ggx_D(alpha, dot(normal, wh));
            float G = ggx_G1(alpha, max(dot(normal, wo), 0.0f)) * ggx_G1(alpha, cosThetaI);
            float3 F = schlick_fresnel(f0, dot(wi, wh));
            float denom = 4.0f * max(dot(normal, wo), 0.0f) * cosThetaI;
            f = F * (D * G / max(denom, 1.0e-6f));
            f *= specular_energy_compensation(f0, roughness, max(dot(normal, wo), 0.0f));
            f = clamp_specular_tail(f, roughness, f0, clampParams);
            f *= reflectScale;
            pdfSpec = ggx_pdf(alpha, normal, wo, wi);
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
        f = (diffuseColor / kPi) * reflectScale;
        pdfDiffuse = lambert_pdf(normal, wi);
    } else {
        result.lobeType = 2u;
        result.lobeRoughness = roughness;
        float cosThetaO = dot(normal, wo);
        float absCosThetaO = fabs(cosThetaO);
        float etaI = 1.0f;
        float etaT = max(material.typeEta.y, 1.0f);
        if (cosThetaO < 0.0f) {
            float tmp = etaI;
            etaI = etaT;
            etaT = tmp;
        }
        float eta = etaI / etaT;
        if (roughness <= 1.0e-3f) {
            wi = refract(-wo, normal, eta);
            float dirLen2 = dot(wi, wi);
            if (dirLen2 <= 0.0f) {
                return result;
            }
            wi = wi * rsqrt(dirLen2);
            float cosThetaI = dot(normal, wi);
            float cosThetaT = 0.0f;
            float Fr = fresnel_dielectric_exact(cosThetaO, etaI, etaT, cosThetaT);
            float etaScale = (etaT * etaT) / (etaI * etaI);
            float directionScale = etaScale * (fabs(cosThetaT) / max(absCosThetaO, 1.0e-6f));
            float3 ft = float3(max(1.0f - Fr, 0.0f) * directionScale);
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
            if (cosThetaOWh * cosThetaIWh > 0.0f) {
                return result;
            }
            float alpha = max(roughness * roughness, 1.0e-4f);
            float D = ggx_D(alpha, max(dot(normal, wh), 0.0f));
            float G = ggx_G1(alpha, absCosThetaO) * ggx_G1(alpha, absCosThetaI);
            float cosThetaT = 0.0f;
            float F = fresnel_dielectric_exact(cosThetaOWh, etaI, etaT, cosThetaT);
            float denom = cosThetaOWh + eta * cosThetaIWh;
            float denomSq = denom * denom;
            if (fabs(denomSq) <= 1.0e-8f) {
                return result;
            }
            float factor = (eta * eta) * fabs(cosThetaIWh) * fabs(cosThetaOWh);
            factor /= max(absCosThetaO * absCosThetaI * denomSq, 1.0e-6f);
            float3 ft = (1.0f - F) * D * G * factor;
            ft *= transmission_tint(material, absCosThetaI);
            f = transmission * ft;
            float pdfWh = ggx_vndf_pdf(alpha, normal, wo, wh);
            float dwhDwi = fabs((eta * eta * cosThetaIWh) / max(denomSq, 1.0e-8f));
            pdfTrans = pdfWh * dwhDwi;
        }
    }

    float cosThetaI = dot(normal, wi);
    float absCosThetaI = fabs(cosThetaI);
    if (absCosThetaI <= 0.0f) {
        return result;
    }

    float pdf = pSpec * pdfSpec + pDiff * pdfDiffuse + pTrans * pdfTrans;
    if (pdf <= 0.0f) {
        return result;
    }
    result.direction = wi;
    result.pdf = pdf;
    result.directionalPdf = pdf;
    result.weight = max(f * absCosThetaI / pdf, float3(0.0f));
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
                                    const bool specularOnly) {
    BsdfEvalResult result;
    result.value = float3(0.0f);
    result.pdf = 0.0f;
    result.isDelta = false;
    result.directionalPdf = 0.0f;
    result.areaPdf = 0.0f;
    result.isBssrdf = false;

    float cosThetaO = max(dot(normal, wo), 0.0f);
    float cosThetaI = max(dot(normal, wi), 0.0f);
    if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
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
            float alpha = roughness * roughness;
            float3 wh = safe_normalize(wo + wi);
            if (dot(wh, normal) <= 0.0f || dot(wo, wh) <= 0.0f || dot(wi, wh) <= 0.0f) {
                return result;
            }

            float D = ggx_D(alpha, dot(normal, wh));
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
            spec = clamp_specular_tail(spec, roughness, f0, clampParams);
            float pdf = ggx_pdf(alpha, normal, wo, wi);
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
        case 4u: { // Plastic (Diffuse base + Clearcoat)
            float coatRoughness = plastic_coat_roughness(material);
            float alpha = coatRoughness * coatRoughness;
            float f0 = plastic_coat_f0(material);
            float3 f0Color = float3(f0);

            float3 spec = float3(0.0f);
            float pdfSpec = 0.0f;
            float3 wh = safe_normalize(wo + wi);
            if (dot(wh, normal) > 0.0f && dot(wo, wh) > 0.0f && dot(wi, wh) > 0.0f) {
                float D = ggx_D(alpha, dot(normal, wh));
                float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
                float3 F = schlick_fresnel(f0Color, dot(wi, wh));
                float denom = 4.0f * cosThetaO * cosThetaI;
                spec = F * (D * G / max(denom, 1e-6f));
                spec = clamp_specular_tail(spec, coatRoughness, f0Color, clampParams);
                spec *= plastic_specular_tint(material);
                float rawPdf = ggx_pdf(alpha, normal, wo, wi);
                if (rawPdf > 0.0f) {
                    pdfSpec = clamp_specular_pdf(rawPdf, clampParams);
                }
                spec = max(spec, float3(0.0f));
            }

            float3 F_i = schlick_fresnel(f0Color, cosThetaI);
            float3 F_o = schlick_fresnel(f0Color, cosThetaO);
            float3 tint = plastic_diffuse_transmission(material, cosThetaI, cosThetaO);
            float3 diffuse = material_base_color(material) / kPi;
            diffuse *= clamp(diffuseOcclusion, 0.0f, 1.0f);
            diffuse *= tint;
            diffuse *= (float3(1.0f) - F_i) * (float3(1.0f) - F_o);
            diffuse *= max(1.0f - plastic_coat_fresnel_average(material), 0.0f);
            diffuse = max(diffuse, float3(0.0f));
            if (specularOnly) {
                diffuse = float3(0.0f);
            }

            float pdfDiffuse = lambert_pdf(normal, wi);
            float pCoat = clamp(plastic_coat_sample_weight(material), 0.0f, 1.0f);
            float pDiffuse = 1.0f - pCoat;
            if (specularOnly) {
                pCoat = 1.0f;
                pDiffuse = 0.0f;
            }
            result.value = spec + diffuse;
            result.pdf = pCoat * pdfSpec + pDiffuse * pdfDiffuse;
            result.directionalPdf = result.pdf;
            break;
        }
        case 5u: { // Subsurface scattering (handled via BSSRDF sampling)
            result.isBssrdf = true;
            result.value = float3(0.0f);
            result.pdf = 0.0f;
            result.directionalPdf = 0.0f;
            break;
        }
        case 6u: { // CarPaint (base + flakes + clearcoat)
            float cosThetaO = max(dot(normal, wo), 0.0f);
            float cosThetaI = max(dot(normal, wi), 0.0f);
            if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
                return result;
            }

            float pCoat = carpaint_coat_sample_weight(material);
            float pFlake = carpaint_flake_sample_weight(material);
            float pBase = max(1.0f - (pCoat + pFlake), 0.0f);
            float norm = pCoat + pFlake + pBase;
            if (norm <= 1.0e-6f) {
                pBase = 1.0f;
                pCoat = 0.0f;
                pFlake = 0.0f;
                norm = 1.0f;
            }
            pCoat /= norm;
            pFlake /= norm;
            pBase /= norm;

            CarpaintLobeResult coatRes = carpaint_eval_coat(material, normal, wo, wi, clampParams);
            CarpaintLobeResult flakeRes = carpaint_eval_flake(material, position, normal, wo, wi, clampParams);
            CarpaintLobeResult baseRes = carpaint_eval_base(material, normal, wo, wi, clampParams);

            result.value = pBase * baseRes.f + pFlake * flakeRes.f + pCoat * coatRes.f;
            result.pdf = pBase * baseRes.pdf + pFlake * flakeRes.pdf + pCoat * coatRes.pdf;
            result.directionalPdf = result.pdf;
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
                                                     specularOnly);
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
