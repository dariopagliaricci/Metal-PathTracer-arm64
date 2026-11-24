// Direct-light surface terms shared by the integrators and the RIS/ReSTIR DI code (moved from
// tracing/path_integrators.metal, which restir/restir_di.metal precedes).

// Thin double-sided cutout foliage (glTF MASK), lit from both sides by direct
// light. BLEND is not a cutout: lighting it from behind let the environment
// through opaque double-sided BLEND exports (every Bistro material).
inline bool material_is_two_sided_alpha_cutout(constant PathtraceUniforms& uniforms,
                                               const MaterialData material,
                                               thread const HitRecord& rec) {
    return rec.twoSided != 0u &&
           material.typeEta.z > 0.5f &&
           material.pbrExtras.w > 0.5f && material.pbrExtras.w < 1.5f &&
           material_texture_valid(uniforms, material.textureIndices0.x);
}

// Falcor neither excludes the shaded triangle from the next ray nor starts it
// below the face, so light arriving from behind the geometric face (reachable
// through a tilted shading normal) is blocked by the surface itself.
inline float falcor_face_visibility(const MaterialData material,
                                    thread const HitRecord& rec,
                                    const float3 wi) {
    return ((material.materialFlags & kMaterialFlagFalcorStandardBsdf) != 0u &&
            dot(rec.normal, wi) <= 0.0f) ? 0.0f : 1.0f;
}

inline float two_sided_alpha_cutout_direct_cos(constant PathtraceUniforms& uniforms,
                                               const MaterialData material,
                                               thread const HitRecord& rec,
                                               const float3 normal,
                                               const float3 wi) {
    float nDotL = dot(source_light_cos_normal(material, normal), wi);
    if ((material_is_two_sided_alpha_cutout(uniforms, material, rec) &&
         (material.materialFlags & kMaterialFlagAlphaCutoutContinuation) == 0u) ||
        (material_is_regularized_dielectric(material) || source_rough_transmission(material)) ||
        pavilion_transmission_nee_supported(material) ||
        pbr_thin_walled_transmission_nee_supported(material)) {
        return saturate(fabs(nDotL));
    }
    return saturate(nDotL) * falcor_face_visibility(material, rec, wi);
}

// A two-sided MASK cutout lit from behind: its closure gives that back-side direction no value, so direct light uses a
// Lambertian term with |n.l|. No BSDF strategy produces that term: the fallback fires only where the closure gives the
// direction 0, and a BSDF-sampled direction carries the closure's value. A transmission lobe does not change this: a
// smooth one samples delta directions only, and a rough one that reaches the direction gives it a value, so the
// fallback does not fire. The pdf set here only keeps the callers' pdf > 0 tests and must not enter NEE's MIS weight
// (EP19: with it, a leaf lit from behind by the environment kept about half of its light). Returns true whenever it
// supplies the value; callers then pass a BSDF pdf of 0 to the MIS weight.
inline bool apply_two_sided_alpha_cutout_diffuse_fallback(constant PathtraceUniforms& uniforms,
                                                          const MaterialData material,
                                                          thread const HitRecord& rec,
                                                          const float3 normal,
                                                          const float3 wi,
                                                          const float diffuseOcclusion,
                                                          thread BsdfEvalResult& bsdfEval) {
    if ((material.materialFlags & (kMaterialFlagSingleScatterSpecular | kMaterialFlagAlphaCutoutContinuation)) != 0u ||
        !material_is_two_sided_alpha_cutout(uniforms, material, rec) ||
        dot(normal, wi) >= 0.0f ||
        any(bsdfEval.value > float3(0.0f))) {
        return false;
    }

    // The back-side copy of the PBR closure's diffuse lobe, so it carries that lobe's weight:
    // (1 - metallic) and (1 - transmission) (pbr eval in closures_eval.metal). A fully transmissive cutout gets none.
    const float metallic = clamp(material.pbrParams.x, 0.0f, 1.0f);
    float3 diffuseColor = clamp01(material.baseColorRoughness.xyz) * (1.0f - metallic);
    diffuseColor *= 1.0f - clamp(material.pbrExtras.z, 0.0f, 1.0f) * (1.0f - metallic);
    diffuseColor *= clamp(diffuseOcclusion, 0.0f, 1.0f);
    bsdfEval.value = diffuseColor / kPi;
    bsdfEval.pdf = max(fabs(dot(normal, wi)) / kPi, 1.0e-6f);
    bsdfEval.directionalPdf = bsdfEval.pdf;
    bsdfEval.areaPdf = 0.0f;
    bsdfEval.isDelta = false;
    bsdfEval.isBssrdf = false;
    return true;
}

// The BSDF pdf NEE's MIS weight should use: 0 where the two-sided cutout fallback supplied the value for a back-side
// direction no BSDF strategy samples.
inline float nee_mis_bsdf_pdf(const BsdfEvalResult bsdfEval, const bool cutoutBackLightUnsampled) {
    return cutoutBackLightUnsampled ? 0.0f : bsdfEval.pdf;
}

// The direct-light target shared by the RIS and ReSTIR DI builders, their reuse reweighting and the reservoir shading.
// A builder whose target is zero where the shading is not never draws that light; a reweighting or a finalization
// with a different target scales each estimate by the ratio of the two.
//
// Surface cosine: plain NEE's (two_sided_alpha_cutout_direct_cos), |n.l| on transmissive materials and on
// two-sided MASK cutouts, the Falcor face test otherwise.
inline float direct_light_target_cos(constant PathtraceUniforms& uniforms,
                                     const MaterialData material,
                                     thread const HitRecord& rec,
                                     const float3 normal,
                                     const float3 wi) {
    return two_sided_alpha_cutout_direct_cos(uniforms, material, rec, normal, wi);
}

// Material term of the target: plain NEE's value, with its Lambertian fallback for a two-sided MASK cutout lit from
// behind and its eta-free continuation convention on area and environment samples (reference dielectrics and Pavilion
// transmission, reference_continuation_nee_value).
// cutoutBackLightUnsampled reports whether the two-sided cutout fallback supplied the value for a direction no BSDF
// strategy samples (see nee_mis_bsdf_pdf).
inline BsdfEvalResult direct_light_target_bsdf(constant PathtraceUniforms& uniforms,
                                               const MaterialData material,
                                               thread const HitRecord& rec,
                                               const float3 normal,
                                               const float3 wo,
                                               const float3 wi,
                                               const FireflyClampParams clampParams,
                                               const float diffuseOcclusion,
                                               const bool specularOnly,
                                               const bool directional,
                                               thread bool& cutoutBackLightUnsampled) {
    BsdfEvalResult bsdfEval = material_closure_eval(material,
                                                    rec.point,
                                                    normal,
                                                    wo,
                                                    wi,
                                                    clampParams,
                                                    PT_DEFAULT_OFF(uniforms.sssMode),
                                                    diffuseOcclusion,
                                                    specularOnly,
                                                    directional);
    cutoutBackLightUnsampled =
        apply_two_sided_alpha_cutout_diffuse_fallback(uniforms, material, rec, normal, wi, diffuseOcclusion, bsdfEval);
    if (!directional) {
        bsdfEval.value = reference_continuation_nee_value(material, normal, wo, wi, bsdfEval.value);
    }
    return bsdfEval;
}

inline BsdfEvalResult direct_light_target_bsdf(constant PathtraceUniforms& uniforms,
                                               const MaterialData material,
                                               thread const HitRecord& rec,
                                               const float3 normal,
                                               const float3 wo,
                                               const float3 wi,
                                               const FireflyClampParams clampParams,
                                               const float diffuseOcclusion,
                                               const bool specularOnly,
                                               const bool directional) {
    bool cutoutBackLightUnsampled = false;
    return direct_light_target_bsdf(uniforms, material, rec, normal, wo, wi, clampParams, diffuseOcclusion,
                                    specularOnly, directional, cutoutBackLightUnsampled);
}
