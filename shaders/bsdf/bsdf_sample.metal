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

            float alpha = roughness * roughness;
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

            float D = ggx_D(alpha, dot(normal, wh));
            float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
            if (material_has_conductor_ior(material)) {
                F = fresnel_conductor(dot(wi, wh), eta, k);
            } else {
                F = schlick_fresnel(f0, dot(wi, wh));
            }

            float denom = 4.0f * cosThetaO * cosThetaI;
            float3 f = F * (D * G / max(denom, 1e-6f));
            f *= specular_energy_compensation(f0, roughness, cosThetaO);
            f = clamp_specular_tail(f, roughness, f0, clampParams);
            float pdf = ggx_pdf(alpha, normal, wo, wi);
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
        case 4u: { // Plastic (Diffuse + Clearcoat)
            float cosThetaO = dot(normal, wo);
            if (cosThetaO <= 0.0f) {
                return result;
            }

            float coatRoughness = plastic_coat_roughness(material);
            float alpha = coatRoughness * coatRoughness;
            float f0 = plastic_coat_f0(material);
            float3 f0Color = float3(f0);
            float pCoat = clamp(plastic_coat_sample_weight(material), 0.0f, 1.0f);
            float pDiffuse = 1.0f - pCoat;
            float fresnelAvg = plastic_coat_fresnel_average(material);
            float3 specTint = plastic_specular_tint(material);
            if (specularOnly) {
                pCoat = 1.0f;
                pDiffuse = 0.0f;
            }

            float selector = rand_uniform(state);
            bool sampleCoat = (selector < pCoat) && (pCoat > 0.0f);

            if (sampleCoat) {
                float3 wh = sample_ggx_vndf(normal, wo, coatRoughness, state);
                if (dot(wh, normal) <= 0.0f) {
                    return result;
                }

                float3 wi = reflect(-wo, wh);
                wi = safe_normalize(wi);
                float cosThetaI = dot(normal, wi);
                if (cosThetaI <= 0.0f) {
                    return result;
                }

                float dotWiWh = dot(wi, wh);
                if (dotWiWh <= 0.0f) {
                    return result;
                }

                float D = ggx_D(alpha, dot(normal, wh));
                float G = ggx_G1(alpha, cosThetaO) * ggx_G1(alpha, cosThetaI);
                float3 F = schlick_fresnel(f0Color, dotWiWh);
                float denom = 4.0f * cosThetaO * cosThetaI;
                float3 spec = F * (D * G / max(denom, 1e-6f));
                spec = clamp_specular_tail(spec, coatRoughness, f0Color, clampParams);
                spec *= specTint;

                float specPdfRaw = ggx_pdf(alpha, normal, wo, wi);
                float specPdf = (specPdfRaw > 0.0f) ? clamp_specular_pdf(specPdfRaw, clampParams) : 0.0f;
                float diffusePdf = lambert_pdf(normal, wi);
                float combinedPdf = pCoat * specPdf + pDiffuse * diffusePdf;
                if (combinedPdf <= 0.0f) {
                    return result;
                }

                float3 weight = spec * cosThetaI / combinedPdf;
                if (!all(isfinite(weight))) {
                    return result;
                }

                result.direction = wi;
                result.weight = max(weight, float3(0.0f));
                result.pdf = combinedPdf;
                result.directionalPdf = combinedPdf;
                result.isDelta = false;
                result.lobeType = 1u;
                result.lobeRoughness = coatRoughness;
                break;
            }

            float3 local = sample_cosine_hemisphere(state);
            float3 wi = safe_normalize(to_world(local, normal));
            float cosThetaI = dot(normal, wi);
            if (cosThetaI <= 0.0f) {
                return result;
            }

            float3 base = material_base_color(material);
            float3 diffuse = base / kPi;
            diffuse *= clamp(diffuseOcclusion, 0.0f, 1.0f);
            float3 tintThrough = plastic_diffuse_transmission(material, cosThetaI, cosThetaO);
            float3 F_i = schlick_fresnel(f0Color, cosThetaI);
            float3 F_o = schlick_fresnel(f0Color, cosThetaO);
            diffuse *= tintThrough;
            diffuse *= (float3(1.0f) - F_i) * (float3(1.0f) - F_o);
            diffuse *= max(1.0f - fresnelAvg, 0.0f);
            diffuse = max(diffuse, float3(0.0f));
            if (specularOnly) {
                diffuse = float3(0.0f);
            }

            float diffusePdf = lambert_pdf(normal, wi);
            float specPdfRaw = ggx_pdf(alpha, normal, wo, wi);
            float specPdf = (specPdfRaw > 0.0f) ? clamp_specular_pdf(specPdfRaw, clampParams) : 0.0f;
            float combinedPdf = pCoat * specPdf + pDiffuse * diffusePdf;
            if (combinedPdf <= 0.0f) {
                return result;
            }

            float3 weight = diffuse * cosThetaI / combinedPdf;
            if (!all(isfinite(weight))) {
                return result;
            }

            result.direction = wi;
            result.weight = max(weight, float3(0.0f));
            result.pdf = combinedPdf;
            result.directionalPdf = combinedPdf;
            result.isDelta = false;
            result.lobeType = 0u;
            result.lobeRoughness = 1.0f;
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
        case 6u: { // CarPaint (base + flakes + clearcoat)
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

            float r = rand_uniform(state);
            uint lobe = 0u; // 0 = base, 1 = flake, 2 = coat
            uint selectedLobeType = 0u;
            float selectedLobeRoughness = 0.0f;
            float thresholdCoat = pCoat;
            float thresholdFlake = pCoat + pFlake;
            if (pCoat > 0.0f && r < thresholdCoat) {
                lobe = 2u;
            } else if (pFlake > 0.0f && r < thresholdFlake) {
                lobe = 1u;
            } else {
                lobe = 0u;
                if (pBase <= 1.0e-6f) {
                    if (pFlake > pCoat && pFlake > 0.0f) {
                        lobe = 1u;
                    } else if (pCoat > 0.0f) {
                        lobe = 2u;
                    }
                }
            }

            float3 wi;
            if (lobe == 2u) {
                float coatRoughness = plastic_coat_roughness(material);
                float3 wh = sample_ggx_vndf(normal, wo, coatRoughness, state);
                if (dot(wh, normal) <= 0.0f) {
                    return result;
                }
                wi = reflect(-wo, wh);
                wi = safe_normalize(wi);
                selectedLobeType = 1u;
                selectedLobeRoughness = coatRoughness;
            } else if (lobe == 1u) {
                float flakeRoughness = max(carpaint_flake_roughness(material), 1.0e-3f);
                float3 flakeNormal = carpaint_flake_normal(material, position, normal);
                float3 wh = sample_ggx_vndf(flakeNormal, wo, flakeRoughness, state);
                if (dot(wh, flakeNormal) <= 0.0f) {
                    return result;
                }
                wi = reflect(-wo, wh);
                wi = safe_normalize(wi);
                selectedLobeType = 1u;
                selectedLobeRoughness = flakeRoughness;
            } else {
                float metallic = carpaint_base_metallic(material);
                float diffuseWeight = max(1.0f - metallic, 0.0f);
                float specWeight = max(metallic, 0.0f);
                float weightSum = diffuseWeight + specWeight;
                float choose = rand_uniform(state);
                bool sampleSpec = (specWeight > 0.0f) && (weightSum > 0.0f) &&
                                  (choose < specWeight / max(weightSum, 1.0e-6f));
                if (sampleSpec) {
                    float baseRough = max(carpaint_base_roughness(material), 1.0e-3f);
                    float3 wh = sample_ggx_vndf(normal, wo, baseRough, state);
                    if (dot(wh, normal) <= 0.0f) {
                        return result;
                    }
                    wi = reflect(-wo, wh);
                    wi = safe_normalize(wi);
                    selectedLobeType = 1u;
                    selectedLobeRoughness = baseRough;
                } else {
                    float3 local = sample_cosine_hemisphere(state);
                    wi = safe_normalize(to_world(local, normal));
                    selectedLobeType = 0u;
                    selectedLobeRoughness = 1.0f;
                }
            }

            if (!all(isfinite(wi)) || dot(normal, wi) <= 0.0f) {
                return result;
            }

            CarpaintLobeResult coatRes = carpaint_eval_coat(material, normal, wo, wi, clampParams);
            CarpaintLobeResult flakeRes = carpaint_eval_flake(material, position, normal, wo, wi, clampParams);
            CarpaintLobeResult baseRes = carpaint_eval_base(material, normal, wo, wi, clampParams);

            float combinedPdf = pBase * baseRes.pdf + pFlake * flakeRes.pdf + pCoat * coatRes.pdf;
            if (combinedPdf <= 0.0f) {
                return result;
            }

            float3 selectedF = baseRes.f;
            float selectedPdf = baseRes.pdf;
            if (lobe == 1u) {
                selectedF = flakeRes.f;
                selectedPdf = flakeRes.pdf;
            } else if (lobe == 2u) {
                selectedF = coatRes.f;
                selectedPdf = coatRes.pdf;
            }
            if (selectedPdf <= 0.0f || !any(selectedF > float3(0.0f))) {
                return result;
            }
            float cosThetaI = max(dot(normal, wi), 0.0f);
            if (cosThetaI <= 0.0f) {
                return result;
            }
            float3 weight = selectedF * cosThetaI / combinedPdf;
            if (!all(isfinite(weight))) {
                return result;
            }
            result.direction = wi;
            result.weight = max(weight, float3(0.0f));
            result.pdf = combinedPdf;
            result.directionalPdf = max(selectedPdf, 0.0f);
            result.areaPdf = 0.0f;
            result.isDelta = false;
            result.isBssrdf = false;
            result.hasExitPoint = false;
            result.lobeType = selectedLobeType;
            result.lobeRoughness = selectedLobeRoughness;
            break;
        }
        case 7u: { // PBR Metallic-Roughness
            return sample_pbr_metallic_roughness(material,
                                                 normal,
                                                 wo,
                                                 incidentDir,
                                                 state,
                                                 clampParams,
                                                 diffuseOcclusion,
                                                 specularOnly);
        }
        case 2u: { // Dielectric
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

            float3 direction;
            float3 weight;
            if (rand_uniform(state) < Fr) {
                direction = reflect(unitDir, normal);
                // The Fresnel branch probability is already accounted for by lobe selection.
                weight = float3(1.0f);
            } else {
                direction = refract(unitDir, normal, relativeEta);
                float dirLen2 = dot(direction, direction);
                if (dirLen2 <= 0.0f) {
                    direction = reflect(unitDir, normal);
                    weight = float3(1.0f);
                } else {
                    direction = direction / sqrt(dirLen2);
                    float etaScale = (etaT * etaT) / (etaI * etaI);
                    float cosThetaI = fabs(cosThetaO);
                    float cosThetaTrans = fabs(cosThetaT);
                    float directionScale = etaScale * (cosThetaTrans / max(cosThetaI, 1e-6f));
                    weight = float3(max(directionScale, 0.0f));
                    if (!isThin) {
                        result.mediumEvent = frontFace ? 1 : -1;
                    }
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
