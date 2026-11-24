inline BsdfSampleResult sample_bsdf(const MaterialData material,
                                    const float3 position,
                                    const float3 normal,
                                    const float3 wo,
                                    const float3 incidentDir,
                                    bool frontFace,
                                    thread uint& state,
                                    const FireflyClampParams clampParams,
                                    const uint sssMode,
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

    if (source_model(material) > 0.0f) {
        return source_sample(material, normal, wo, frontFace, state, specularOnly);
    }
    uint type = static_cast<uint>(material.typeEta.x);
    switch (type) {
        case 0u: { // Lambertian
            if (specularOnly) {
                return result;
            }
            float3 local = sample_cosine_hemisphere(state);
            float3 wi = safe_normalize(to_world(local, normal));
            float cosThetaI = dot(normal, wi);
            if (cosThetaI <= 0.0f) {
                return result;
            }

            float pdf = lambert_pdf(normal, wi);
            if (pdf <= 0.0f) {
                return result;
            }

            float3 albedo = material_base_color(material);
            albedo *= clamp(diffuseOcclusion, 0.0f, 1.0f);
            float3 f = albedo / kPi;
            float3 weight = f * cosThetaI / pdf;
            weight = max(weight, float3(0.0f));
            if (!all(isfinite(weight))) {
                return result;
            }

            result.direction = wi;
            result.weight = weight;
            result.pdf = pdf;
            result.directionalPdf = pdf;
            result.isDelta = false;
            result.lobeType = 0u;
            result.lobeRoughness = 1.0f;
            break;
        }
        case 1u: { // Metal (GGX)
            float roughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
            float3 eta = material.conductorEta.xyz;
            float3 k = material.conductorK.xyz;
            float3 f0 = conductor_f0(material);
            float3 wi;
            float3 F;
            if (roughness <= 1e-3f) {
                wi = reflect(incidentDir, normal);
                float cosThetaI = dot(normal, wi);
                if (cosThetaI <= 0.0f) {
                    return result;
                }
                float cosThetaO = dot(normal, wo);
                float cosTheta = max(cosThetaO, 0.0f);
                if (material_has_conductor_ior(material)) {
                    F = fresnel_conductor(cosTheta, eta, k);
                } else {
                    F = schlick_fresnel(f0, cosTheta);
                }
                result.direction = wi;
                result.weight = F;
                result.pdf = 1.0f;
                result.directionalPdf = 1.0f;
                result.isDelta = true;
                result.lobeType = 1u;
                result.lobeRoughness = roughness;
                break;
            }

            float alpha = max(roughness * roughness, 1.0e-4f);
            float3 wh = sample_ggx_vndf(normal, wo, roughness, state);
            if (dot(wh, normal) <= 0.0f) {
                return result;
            }

            wi = reflect(-wo, wh);
            wi = safe_normalize(wi);

            if (!all(isfinite(wi))) {
                return result;
            }

            float cosThetaI = dot(normal, wi);
            float cosThetaO = dot(normal, wo);
            if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
                return result;
            }

            float dotWoWh = dot(wo, wh);
            if (dotWoWh <= 0.0f) {
                return result;
            }

            float D = ggx_D(alpha, normal, wh);
            float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
            if (material_has_conductor_ior(material)) {
                F = fresnel_conductor(dot(wi, wh), eta, k);
            } else {
                F = schlick_fresnel(f0, dot(wi, wh));
            }

            float denom = 4.0f * cosThetaO * cosThetaI;
            float3 f = F * (D * G / max(denom, 1e-6f));
            f *= specular_energy_compensation(f0, roughness, cosThetaO);
            f = sanitize_specular_value(f);
            float pdf = D * ggx_G1(alpha, cosThetaO) / (4.0f * cosThetaO);
            if (pdf <= 0.0f) {
                return result;
            }

            float clampedPdf = clamp_specular_pdf(pdf, clampParams);
            float3 weight = f * cosThetaI / clampedPdf;
            weight = max(weight, float3(0.0f));
            if (!all(isfinite(weight))) {
                return result;
            }
            result.direction = wi;
            result.weight = weight;
            result.pdf = clampedPdf;
            result.directionalPdf = clampedPdf;
            result.isDelta = false;
            result.lobeType = 1u;
            result.lobeRoughness = roughness;
            break;
        }
        case 4u: { // Plastic: one-sample MIS over the coat and the diffuse base (see plastic_closure_eval)
            PlasticLobeProbabilities probs = plastic_lobe_probabilities(material, normal, wo, specularOnly);
            if (probs.coat + probs.diffuse <= 0.0f) {
                return result;
            }
            float coatRoughness = plastic_coat_roughness(material);
            bool sampleCoat = rand_uniform(state) < probs.coat;
            float3 wi;
            if (sampleCoat) {
                float3 wh = sample_ggx_vndf(normal, wo, coatRoughness, state);
                if (dot(wh, normal) <= 0.0f) {
                    return result;
                }
                wi = safe_normalize(reflect(-wo, wh));
            } else {
                wi = safe_normalize(to_world(sample_cosine_hemisphere(state), normal));
            }
            float cosThetaI = dot(normal, wi);
            if (!all(isfinite(wi)) || cosThetaI <= 0.0f) {
                return result;
            }

            ClosureValuePdf closure =
                plastic_closure_eval(material, normal, wo, wi, probs, diffuseOcclusion, specularOnly, clampParams);
            if (closure.pdf <= 0.0f || !any(closure.value > float3(0.0f))) {
                return result;
            }
            float3 weight = closure.value * cosThetaI / closure.pdf;
            if (!all(isfinite(weight))) {
                return result;
            }
            result.direction = wi;
            result.weight = max(weight, float3(0.0f));
            result.pdf = closure.pdf;
            result.directionalPdf = closure.pdf;
            result.isDelta = false;
            result.lobeType = sampleCoat ? 1u : 0u;
            result.lobeRoughness = sampleCoat ? coatRoughness : 1.0f;
            break;
        }
        case 5u: { // Subsurface scattering (separable diffusion)
            if (specularOnly) {
                return result;
            }
            bool useSeparable = sss_use_separable(sssMode, material);
            float meanFreePath = max(material.sssParams.x, 1.0e-4f);
            useSeparable = useSeparable && (meanFreePath > 1.0e-4f);

            if (useSeparable) {
                float anisotropy = clamp(material.sssSigmaS.w, -0.99f, 0.99f);
                float3 baseColor = material_base_color(material);
                float3 sigmaA = sss_sigma_a(material, baseColor, meanFreePath, anisotropy);
                float3 sigmaSPrime = sss_sigma_s_prime(material, baseColor, meanFreePath, anisotropy);
                float sigmaTrScalar = sss_sigma_tr_scalar(sigmaA, sigmaSPrime);
                if (sigmaTrScalar <= 0.0f) {
                    useSeparable = false;
                } else {
                    float radius = sample_sss_radius(sigmaTrScalar, state);
                    radius = min(radius, meanFreePath * 10.0f);
                    float pdfRadius = pdf_sss_radius(radius, sigmaTrScalar);
                    if (pdfRadius <= 0.0f || !isfinite(pdfRadius)) {
                        useSeparable = false;
                    } else {
                        float phi = 2.0f * kPi * rand_uniform(state);
                        float sinPhi = sin(phi);
                        float cosPhi = cos(phi);
                        float3 tangent;
                        float3 bitangent;
                        build_onb(normal, tangent, bitangent);
                        float2 disp = radius * float2(cosPhi, sinPhi);
                        float3 exitPoint = position + tangent * disp.x + bitangent * disp.y;
                        float3 exitNormal = normal;

                        float3 localDir = sample_cosine_hemisphere(state);
                        float3 wi = safe_normalize(to_world(localDir, exitNormal));
                        float cosThetaExit = dot(exitNormal, wi);
                        float pdfDir = lambert_pdf(exitNormal, wi);
                        float pdfArea = pdfRadius / (2.0f * kPi * max(radius, 1.0e-4f));

                        if (cosThetaExit <= 0.0f || pdfDir <= 0.0f || pdfArea <= 0.0f) {
                            useSeparable = false;
                        } else {
                            float3 profile = normalized_diffusion_profile(radius, sigmaA, sigmaSPrime);
                            float3 coatTint = plastic_coat_tint(material);
                            float coatAverage = 1.0f - clamp(material.coatParams.w, 0.0f, 1.0f);
                            float coatTransmission = 1.0f;
                            if (material.sssParams.z > 0.5f) {
                                float coatIor = max(material.typeEta.z, 1.0f);
                                float f0 = ((coatIor - 1.0f) / (coatIor + 1.0f));
                                f0 *= f0;
                                float cosIn = max(dot(normal, wo), 0.0f);
                                float cosOut = cosThetaExit;
                                float transIn = 1.0f - schlick_fresnel_scalar(f0, cosIn);
                                float transOut = 1.0f - schlick_fresnel_scalar(f0, cosOut);
                                coatTransmission = clamp(transIn * transOut, 0.0f, 1.0f);
                                profile *= coatTint;
                            }

                            float3 weight = profile * cosThetaExit * coatAverage * coatTransmission;
                            float denom = max(pdfArea * pdfDir, 1.0e-6f);
                            weight = max(weight / denom, float3(0.0f));

                            if (!all(isfinite(weight))) {
                                useSeparable = false;
                            } else {
                                result.direction = wi;
                                result.weight = weight;
                                result.pdf = denom;
                                result.directionalPdf = pdfDir;
                                result.areaPdf = pdfArea;
                                result.exitPoint = exitPoint;
                                result.exitNormal = exitNormal;
                                result.isDelta = false;
                                result.isBssrdf = true;
                                result.hasExitPoint = true;
                                break;
                            }
                        }
                    }
                }
            }

            // Fallback to diffuse lambertian if separable SSS is disabled or failed
            float3 local = sample_cosine_hemisphere(state);
            float3 wi = safe_normalize(to_world(local, normal));
            float cosThetaI = dot(normal, wi);
            if (cosThetaI <= 0.0f) {
                return result;
            }
            float pdf = lambert_pdf(normal, wi);
            if (pdf <= 0.0f) {
                return result;
            }
            float3 albedo = material_base_color(material);
            float3 weight = (albedo / kPi) * cosThetaI / pdf;
            if (!all(isfinite(weight))) {
                return result;
            }
            result.direction = wi;
            result.weight = max(weight, float3(0.0f));
            result.pdf = pdf;
            result.directionalPdf = pdf;
            result.areaPdf = 0.0f;
            result.isDelta = false;
            result.isBssrdf = false;
            result.hasExitPoint = false;
            break;
        }
        case 6u: { // CarPaint: one-sample MIS over the closure's lobes (see carpaint_closure_eval)
            if (dot(normal, wo) <= 0.0f) {
                return result;
            }
            float3 flakeNormal = carpaint_flake_normal(material, position, normal);
            CarpaintLobeProbabilities probs = carpaint_lobe_probabilities(material, normal, flakeNormal, wo);
            float u = rand_uniform(state);
            float3 lobeNormal = normal;
            float lobeRoughness = 1.0f;
            bool diffuseLobe = false;
            if (u < probs.coat) {
                lobeRoughness = carpaint_coat_roughness(material);
            } else if (u < probs.coat + probs.flake) {
                lobeNormal = flakeNormal;
                lobeRoughness = carpaint_flake_lobe_roughness(material);
            } else if (u < probs.coat + probs.flake + probs.diffuse || probs.metal <= 0.0f) {
                diffuseLobe = true;
            } else {
                lobeRoughness = carpaint_metal_roughness(material);
            }
            if (probs.coat + probs.flake + probs.diffuse + probs.metal <= 0.0f) {
                return result;
            }

            float3 wi;
            if (diffuseLobe) {
                wi = safe_normalize(to_world(sample_cosine_hemisphere(state), normal));
            } else {
                float3 wh = sample_ggx_vndf(lobeNormal, wo, lobeRoughness, state);
                if (dot(wh, lobeNormal) <= 0.0f) {
                    return result;
                }
                wi = safe_normalize(reflect(-wo, wh));
            }
            float cosThetaI = dot(normal, wi);
            if (!all(isfinite(wi)) || cosThetaI <= 0.0f) {
                return result;
            }

            ClosureValuePdf closure =
                carpaint_closure_eval(material, normal, flakeNormal, wo, wi, probs, clampParams);
            if (closure.pdf <= 0.0f || !any(closure.value > float3(0.0f))) {
                return result;
            }
            float3 weight = closure.value * cosThetaI / closure.pdf;
            if (!all(isfinite(weight))) {
                return result;
            }
            result.direction = wi;
            result.weight = max(weight, float3(0.0f));
            result.pdf = closure.pdf;
            result.directionalPdf = closure.pdf;
            result.areaPdf = 0.0f;
            result.isDelta = false;
            result.isBssrdf = false;
            result.hasExitPoint = false;
            result.lobeType = diffuseLobe ? 0u : 1u;
            result.lobeRoughness = lobeRoughness;
            break;
        }
        case 7u: { // PBR Metallic-Roughness
            return sample_pbr_metallic_roughness(material,
                                                 normal,
                                                 wo,
                                                 incidentDir,
                                                 frontFace,
                                                 state,
                                                 clampParams,
                                                 diffuseOcclusion,
                                                 specularOnly);
        }
        case 2u: { // Dielectric
            if (material_is_regularized_dielectric(material)) {
                return sample_regularized_dielectric(material, normal, wo, frontFace, state);
            }
            result.isDelta = true;
            bool isThin = material_is_thin_dielectric(material);
            float refIdx = max(material.typeEta.y, 1.0f);
            float etaI = 1.0f;
            float etaT = refIdx;
            float3 unitDir = incidentDir;
            float cosThetaO = dot(-unitDir, normal);
            cosThetaO = clamp(cosThetaO, -1.0f, 1.0f);
            if (!isThin && !frontFace) {
                etaI = refIdx;
                etaT = 1.0f;
            }
            float relativeEta = etaI / etaT;
            float cosThetaT = 0.0f;
            float Fr = fresnel_dielectric_exact(cosThetaO, etaI, etaT, cosThetaT);
            if (isThin && Fr < 1.0f) {
                float Ft = 1.0f - Fr;
                Fr += (Ft * Ft * Fr) / max(1.0f - Fr * Fr, 1.0e-6f);
            }
            float3 transmissionTint = dielectric_transmission_tint(material);

            float3 direction;
            float3 weight;
            if (rand_uniform(state) < Fr) {
                direction = reflect(unitDir, normal);
                // The Fresnel branch probability is already accounted for by lobe selection.
                weight = float3(1.0f);
            } else if (isThin) {
                // A thin dielectric represents two coincident, parallel interfaces.
                direction = unitDir;
                weight = transmissionTint;
            } else {
                direction = refract(unitDir, normal, relativeEta);
                float dirLen2 = dot(direction, direction);
                if (dirLen2 <= 0.0f) {
                    direction = reflect(unitDir, normal);
                    weight = float3(1.0f);
                } else {
                    direction = direction / sqrt(dirLen2);
                    // Fresnel transmission cancels against its branch probability.
                    // Physical radiance transport scales by eta^-2; the tagged
                    // Pavilion reference uses its legacy continuation convention.
                    float etaScale = dielectric_continuation_eta_scale(material, relativeEta);
                    weight = transmissionTint * etaScale;
                    result.mediumEvent = frontFace ? 1 : -1;
                }
            }

            result.direction = safe_normalize(direction);
            result.weight = weight;
            result.pdf = 1.0f;
            result.directionalPdf = 1.0f;
            result.lobeType = 1u;
            result.lobeRoughness = 0.0f;
            break;
        }
        default:
            break;
    }

    return result;
}
