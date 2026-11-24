// Tungsten source closures (EP8: shared by the Metal shaders and the Embree backend; formerly
// shaders/bsdf/source_closures.metal). Metal includes this from bsdf/closures_eval.metal.
#ifndef PT_SHARED_SOURCE_CLOSURES_H
#define PT_SHARED_SOURCE_CLOSURES_H

#include "portability.h"
#include "math.h"
#include "source_features.h"
#include "closure_common.h"

PT_SHARED_BEGIN

// Tungsten 88ea02044dbaf20472a8173b6752460b50c096d8 BSDFs.
// Public closure values exclude the outgoing cosine, unlike Tungsten's eval.
// Models: 1 conductor, 2 plastic, 3 dielectric/thin sheet, 4 smooth coated conductor, 5 Oren-Nayar diffuse.
inline float source_D(float alpha, float c, bool beckmann) {
    if (c <= 0.0f) return 0.0f;
    alpha = max(alpha, 1.0e-3f);
    float c2 = c*c, a2 = alpha*alpha;
    float t2 = max(1.0f-c2, 0.0f)/c2;
    if (beckmann) return exp(-t2/a2)/(kPi*a2*c2*c2);
    // Algebraically equivalent form avoids cancellation at small alpha.
    float d = max(1.0f-c2, 0.0f) + a2*c2;
    return a2/(kPi*d*d);
}

inline float source_G1(float alpha, float c, bool beckmann) {
    c = fabs(c);
    if (c <= 0.0f) return 0.0f;
    alpha = max(alpha, 1.0e-3f);
    float t2 = max(1.0f-c*c, 0.0f)/(c*c);
    if (!beckmann) return 2.0f/(1.0f+sqrt(1.0f+alpha*alpha*t2));
    if (t2 == 0.0f) return 1.0f;
    float a = 1.0f/(alpha*sqrt(t2));
    return a < 1.6f ? (3.535f*a+2.181f*a*a)/(1.0f+2.276f*a+2.577f*a*a) : 1.0f;
}

inline float3 source_sample_m(float alpha, bool beckmann, float3 n, PT_THREAD uint& state) {
    alpha = max(alpha, 1.0e-3f);
    float u = min(rand_uniform(state), 0.99999994f);
    float phi = 2.0f*kPi*rand_uniform(state);
    float t2 = beckmann ? -alpha*alpha*log(1.0f-u) : alpha*alpha*u/(1.0f-u);
    float c = rsqrt(1.0f+t2), s = sqrt(max(0.0f, 1.0f-c*c));
    return to_world(PT_F3(cos(phi)*s, sin(phi)*s, c), n);
}

inline float source_F(float c, float farOverNear, PT_THREAD float& ct) {
    if (farOverNear == 1.0f) { ct = fabs(c); return 0.0f; }
    return fresnel_dielectric_exact(fabs(c), 1.0f, farOverNear, ct);
}

inline float source_thin_R(float c, float ior) {
    float ct;
    if (source_F(c, ior, ct) >= 1.0f) return 1.0f;
    float rs = (c-ior*ct)/(c+ior*ct);
    float rp = (ior*c-ct)/(ior*c+ct);
    rs *= rs; rp *= rp;
    return rs/(1.0f+rs) + rp/(1.0f+rp);
}

inline bool source_is_thin(PT_IN(MaterialData) m) {
    return source_model(m) == 3.0f && m.typeEta.w > 0.5f;
}
inline float3 source_thin_forward(PT_IN(MaterialData) m, float3 n, float3 wo) {
    float c = clamp(fabs(dot(safe_normalize(n),safe_normalize(wo))),0.0f,1.0f), ct;
    float R = source_thin_R(c,m.sourceFresnel.w);
    source_F(c,m.sourceFresnel.w,ct);
    float3 T = float3(1.0f-R);
    if (ct > 0.0f) T *= exp(-m.coatAbsorption.xyz*(2.0f*m.coatParams.y/ct));
    return T*material_base_color(m)*m.sourceReflectance.xyz;
}

inline float source_alpha(PT_IN(MaterialData) m) { return max(m.sourceParams.z, 1.0e-3f); }
inline bool source_beckmann(PT_IN(MaterialData) m) { return m.sourceParams.y == 1.0f; }
inline bool source_rough_transmission(PT_IN(MaterialData) m) {
    return source_model(m) == 3.0f && m.sourceParams.z > 0.0f &&
        m.sourceParams.w > 0.5f && m.typeEta.w < 0.5f && m.sourceFresnel.w != 1.0f;
}
inline float source_sample_alpha(PT_IN(MaterialData) m, float co) {
    return max((1.2f-0.2f*sqrt(fabs(co)))*m.sourceParams.z, 1.0e-3f);
}
inline float3 source_conductor_tint(PT_IN(MaterialData) m) {
    return m.sourceReflectance.xyz * (m.sourceFresnel.z > 0.5f ? material_base_color(m) : float3(1.0f));
}
inline float source_coat_probability(PT_IN(MaterialData) m, float co, bool specularOnly) {
    float ct, F = source_F(co, m.sourceFresnel.w, ct);
    float substrate = m.sourceFresnel.y*(1.0f-F);
    if (m.sourceParams.x == 2.0f && m.sourceParams.z > 0.0f)
        substrate *= dot(material_base_color(m), float3(1.0f/3.0f));
    if (specularOnly && m.sourceParams.x == 2.0f) return 1.0f;
    return F/max(F+substrate, 1.0e-30f);
}

inline BsdfEvalResult source_conductor_eval(PT_IN(MaterialData) m, float3 n, float3 wo, float3 wi) {
    BsdfEvalResult r{};
    float co = dot(n,wo), ci = dot(n,wi);
    if (co <= 0.0f || ci <= 0.0f || m.sourceParams.z == 0.0f) return r;
    float3 h = safe_normalize(wo+wi);
    float ch = dot(n,h), oh = dot(wo,h);
    if (ch <= 0.0f || oh <= 0.0f) return r;
    float a = source_alpha(m);
    float D = source_D(a,ch,source_beckmann(m));
    float G = source_G1(a,co,source_beckmann(m))*source_G1(a,ci,source_beckmann(m));
    float3 F = fresnel_conductor(oh,m.conductorEta.xyz,m.conductorK.xyz);
    r.value = source_conductor_tint(m)*F*D*G/(4.0f*co*ci);
    if (m.sourceReflectance.w > 0.5f)
        r.value *= specular_energy_compensation(conductor_f0(m),sqrt(a),co);
    r.pdf = D*ch/(4.0f*oh);
    r.directionalPdf = r.pdf;
    return r;
}

inline BsdfEvalResult source_dielectric_eval(PT_IN(MaterialData) m, float3 n, float3 wo, float3 wi,
                                            bool reflectionOnly = false) {
    BsdfEvalResult r{};
    float co = dot(n,wo), ci = dot(n,wi);
    if (co <= 0.0f || ci == 0.0f || m.sourceParams.z == 0.0f || m.typeEta.w > 0.5f) return r;
    float ratio = reflectionOnly ? m.sourceFresnel.w : m.typeEta.y;
    bool onlyR = reflectionOnly || m.sourceParams.w < 0.5f;
    float eta = 1.0f/ratio;
    bool reflection = ci > 0.0f;
    if (onlyR && !reflection) return r;
    float3 h = safe_normalize(reflection ? wo+wi : eta*wo+wi);
    if (dot(n,h) < 0.0f) h = -h;
    float ch = dot(n,h), oh = dot(wo,h), ih = dot(wi,h);
    if (ch <= 0.0f || oh <= 0.0f || ih*ci <= 0.0f) return r;
    float ct, F = source_F(oh,ratio,ct);
    float a = source_alpha(m);
    float D = source_D(a,ch,source_beckmann(m));
    float G = source_G1(a,co,source_beckmann(m))*source_G1(a,ci,source_beckmann(m));
    float pm = source_D(source_sample_alpha(m,co),ch,source_beckmann(m))*ch;
    float3 tint = reflectionOnly ? float3(1.0f) : material_base_color(m)*m.sourceReflectance.xyz;
    if (reflection) {
        r.value = tint*(F*D*G/(4.0f*co*ci));
        r.pdf = pm/(4.0f*oh)*(onlyR ? 1.0f : F);
    } else {
        float den = eta*oh+ih;
        if (den == 0.0f) return r;
        r.value = tint*(fabs(oh*ih)*(1.0f-F)*D*G*eta*eta/(den*den*co*fabs(ci)));
        r.pdf = (1.0f-F)*pm*fabs(ih)/(den*den);
    }
    r.directionalPdf = r.pdf;
    return r;
}

// Raw authored Ns chooses relative eta (Tungsten signed local incidence).
// Only the dielectric closure faces its sampling frame; Ng remains untouched
// for boundary selection and precise continuation offsets.
inline void source_dielectric_frame(PT_THREAD MaterialData& m, PT_THREAD float3& n, float3 wo) {
    if (m.sourceParams.x == 3.0f &&
        (m.materialFlags & kMaterialFlagAuthoredDielectricNormal) != 0u) {
        const bool entering = dot(n, wo) >= 0.0f;
        const float ior = m.sourceFresnel.w;
        m.typeEta.y = entering || m.typeEta.w > 0.5f ? ior : 1.0f/ior;
        if (!entering) n = -n;
        // The sample's eval uses this already prepared frame/eta exactly once.
        m.materialFlags &= ~kMaterialFlagAuthoredDielectricNormal;
    }
}

// Tungsten OrenNayarBsdf. sourceParams.z holds the authored roughness (not an alpha);
// albedo comes from material_base_color. Its sampler mixes a uniform and a cosine hemisphere.
inline float source_oren_nayar_ratio(PT_IN(MaterialData) m) { return clamp(m.sourceParams.z, 0.01f, 1.0f); }

inline BsdfEvalResult source_oren_nayar_eval(PT_IN(MaterialData) m, float3 n, float3 wo, float3 wi) {
    BsdfEvalResult r{};
    float co = dot(n,wo), ci = dot(n,wi);
    if (co <= 0.0f || ci <= 0.0f) return r;
    co = min(co,1.0f); ci = min(ci,1.0f);
    float thetaO = acos(co), thetaI = acos(ci);
    float alpha = max(thetaO,thetaI), beta = min(thetaO,thetaI);
    float sinAlpha = sin(alpha);
    float denom = max(1.0f-co*co,0.0f)*max(1.0f-ci*ci,0.0f);
    float cosDeltaPhi = denom == 0.0f ? 1.0f : (dot(wo,wi)-co*ci)*rsqrt(denom);
    cosDeltaPhi = clamp(cosDeltaPhi,-1.0f,1.0f);
    float sigmaSq = 0.5f*m.sourceParams.z*m.sourceParams.z;
    float C1 = 1.0f - 0.5f*sigmaSq/(sigmaSq+0.33f);
    float C2 = 0.45f*sigmaSq/(sigmaSq+0.09f);
    float b = (2.0f/kPi)*beta;
    C2 *= cosDeltaPhi >= 0.0f ? sinAlpha : sinAlpha-b*b*b;
    float ab = (4.0f/(kPi*kPi))*alpha*beta;
    float C3 = 0.125f*(sigmaSq/(sigmaSq+0.09f))*ab*ab;
    float fr1 = C1 + cosDeltaPhi*C2*tan(beta) + (1.0f-fabs(cosDeltaPhi))*C3*tan(0.5f*(alpha+beta));
    float fr2 = 0.17f*sigmaSq/(sigmaSq+0.13f)*(1.0f-cosDeltaPhi*b*b);
    float3 albedo = material_base_color(m);
    r.value = (albedo*fr1 + albedo*albedo*fr2)*(1.0f/kPi);
    float ratio = source_oren_nayar_ratio(m);
    r.pdf = ratio/(2.0f*kPi) + (1.0f-ratio)*ci/kPi;
    r.directionalPdf = r.pdf;
    return r;
}

inline float3 source_sample_uniform_hemisphere(PT_THREAD uint& state) {
    float z = rand_uniform(state);
    float s = sqrt(max(1.0f-z*z,0.0f));
    float phi = 2.0f*kPi*rand_uniform(state);
    return PT_F3(s*cos(phi), s*sin(phi), z);
}

inline BsdfEvalResult source_eval(PT_IN(MaterialData) m, float3 n, float3 wo, float3 wi, bool specularOnly) {
    if (m.sourceParams.x == 5.0f) return specularOnly ? BsdfEvalResult{} : source_oren_nayar_eval(m,n,wo,wi);
    if (m.sourceParams.x == 1.0f) return source_conductor_eval(m,n,wo,wi);
    if (m.sourceParams.x == 3.0f) {
        MaterialData dielectric = m;
        source_dielectric_frame(dielectric, n, wo);
        return source_dielectric_eval(dielectric,n,wo,wi);
    }
    BsdfEvalResult r{};
    float co = dot(n,wo), ci = dot(n,wi);
    if (co <= 0.0f || ci <= 0.0f) return r;
    float ior = m.sourceFresnel.w, cto, cti;
    float Fo = source_F(co,ior,cto), Fi = source_F(ci,ior,cti);
    float p = source_coat_probability(m,co,specularOnly);
    if (m.sourceParams.x == 2.0f) {
        if (m.sourceParams.z > 0.0f) {
            r = source_dielectric_eval(m,n,wo,wi,true);
            r.pdf *= p;
        }
        if (!specularOnly) {
            float3 albedo = material_base_color(m);
            float3 den = 1.0f - (m.sourceParams.w > 0.5f ? albedo : float3(1.0f))*m.sourceFresnel.x;
            float3 diffuse = albedo/max(den,float3(1.0e-7f));
            diffuse *= exp(-m.coatAbsorption.xyz*m.coatParams.y*(1.0f/co+1.0f/ci));
            r.value += diffuse*((1.0f-Fi)*(1.0f-Fo)/(ior*ior*kPi));
            r.pdf += (1.0f-p)*ci/kPi;
        }
    } else if (m.sourceParams.x == 4.0f) {
        float3 subO = (wo-n*co)/ior + n*cto;
        float3 subI = (wi-n*ci)/ior + n*cti;
        r = source_conductor_eval(m,n,subO,subI);
        // Substrate eval excludes cosine: the cosine in Tungsten cancels
        // the coat's solid-angle Jacobian when converting to this interface.
        r.value *= (1.0f-Fi)*(1.0f-Fo)/(ior*ior)*
            exp(-m.coatAbsorption.xyz*m.coatParams.y*(1.0f/cto+1.0f/cti));
        r.pdf *= (1.0f-p)*ci/(ior*ior*cti);
    }
    r.directionalPdf = r.pdf;
    return r;
}

inline BsdfSampleResult source_sample(MaterialData m, float3 n, float3 wo, bool frontFace,
                                      PT_THREAD uint& state, bool specularOnly) {
    BsdfSampleResult r{};
    source_dielectric_frame(m, n, wo);
    float co = dot(n,wo);
    if (co <= 0.0f) return r;
    uint model = uint(m.sourceParams.x);
    r.lobeType = 1u;
    r.lobeRoughness = sqrt(max(m.sourceParams.z,0.0f));
    if (model == 3u && (m.sourceParams.z == 0.0f || m.typeEta.w > 0.5f || m.sourceFresnel.w == 1.0f)) {
        bool thin = m.typeEta.w > 0.5f;
        float ratio = thin ? m.sourceFresnel.w : m.typeEta.y;
        float ct, F = source_F(co,ratio,ct);
        if (thin) F = source_thin_R(co,ratio);
        float3 T = float3(1.0f-F);
        if (thin && ct > 0.0f)
            T *= exp(-m.coatAbsorption.xyz*(2.0f*m.coatParams.y/ct));
        float probability = thin ? 1.0f-dot(T,float3(1.0f/3.0f)) : F;
        if (!thin && m.sourceParams.w < 0.5f) probability = 1.0f;
        if (thin && m.sourceFresnel.z == 2.0f) {
            // Forward-event roulette was performed before conditional surface shading.
            float pForward = dot(source_thin_forward(m,n,wo),float3(1.0f/3.0f));
            r.direction = reflect(-wo,n);
            r.weight = material_base_color(m)*m.sourceReflectance.xyz * F/max(1.0f-pForward,1.0e-30f);
            r.isDelta = true; r.pdf = r.directionalPdf = 1.0f;
            return r;
        }
        bool reflection = rand_uniform(state) < probability;
        r.direction = reflection ? reflect(-wo,n) : thin ? -wo :
            safe_normalize(-wo/ratio + n*(co/ratio-ct));
        r.weight = material_base_color(m)*m.sourceReflectance.xyz;
        if (thin) r.weight *= reflection ? float3(F/max(probability,1.0e-30f)) : T/max(1.0f-probability,1.0e-30f);
        else if (m.sourceParams.w < 0.5f) r.weight *= F;
        if (!reflection && !thin) r.weight /= ratio*ratio;
        r.isDelta = true;
        r.pdf = reflection ? probability : 1.0f-probability;
        r.directionalPdf = r.pdf;
        r.mediumEvent = !reflection && !thin ? (frontFace ? 1 : -1) : 0;
        return r;
    }
    if (model == 1u && m.sourceParams.z == 0.0f) {
        r.direction = reflect(-wo,n);
        r.weight = source_conductor_tint(m)*fresnel_conductor(co,m.conductorEta.xyz,m.conductorK.xyz);
        r.isDelta = true; r.pdf = r.directionalPdf = 1.0f;
        return r;
    }
    float3 wi;
    if (model == 5u) {
        if (specularOnly) return r;
        const bool uniformHemisphere = rand_uniform(state) < source_oren_nayar_ratio(m);
        wi = to_world(uniformHemisphere ? source_sample_uniform_hemisphere(state) : sample_cosine_hemisphere(state), n);
        r.lobeType = 0u; r.lobeRoughness = 1.0f;
    } else if (model == 1u || model == 3u) {
        float a = model == 3u ? source_sample_alpha(m,co) : source_alpha(m);
        float3 h = source_sample_m(a,source_beckmann(m),n,state);
        float oh = dot(wo,h);
        if (oh <= 0.0f) return r;
        bool reflection = true;
        float ct = 0.0f;
        if (model == 3u) {
            float F = source_F(oh,m.typeEta.y,ct);
            reflection = m.sourceParams.w < 0.5f || rand_uniform(state) < F;
        }
        wi = reflection ? reflect(-wo,h) : safe_normalize(-wo/m.typeEta.y+h*(oh/m.typeEta.y-ct));
        if (reflection ? dot(n,wi) <= 0.0f : dot(n,wi) >= 0.0f) return r;
        if (!reflection) r.mediumEvent = frontFace ? 1 : -1;
    } else {
        float p = source_coat_probability(m,co,specularOnly);
        bool coat = rand_uniform(state) < p;
        if (coat && (model == 4u || m.sourceParams.z == 0.0f)) {
            float ct, F = source_F(co,m.sourceFresnel.w,ct);
            r.direction = reflect(-wo,n);
            r.weight = float3(F/max(p,1.0e-30f));
            r.pdf = r.directionalPdf = p; r.isDelta = true;
            return r;
        }
        if (model == 2u) {
            if (coat) {
                float3 h = source_sample_m(source_sample_alpha(m,co),source_beckmann(m),n,state);
                if (dot(wo,h) <= 0.0f) return r;
                wi = reflect(-wo,h);
            } else {
                wi = to_world(sample_cosine_hemisphere(state),n);
                r.lobeType = 0u; r.lobeRoughness = 1.0f;
            }
        } else {
            float ior = m.sourceFresnel.w, cto;
            source_F(co,ior,cto);
            float3 subO = (wo-n*co)/ior+n*cto;
            float3 h = source_sample_m(source_alpha(m),source_beckmann(m),n,state);
            if (dot(subO,h) <= 0.0f) return r;
            float3 subI = reflect(-subO,h);
            float cti = dot(n,subI), ce;
            if (cti <= 0.0f || source_F(cti,1.0f/ior,ce) >= 1.0f) return r;
            wi = (subI-n*cti)*ior+n*ce;
        }
    }
    float ci = dot(n,wi);
    if (model != 3u && ci <= 0.0f) return r;
    BsdfEvalResult e = source_eval(m,n,wo,wi,specularOnly);
    if (e.pdf <= 0.0f || !PT_ALL_FINITE3(e.value)) return r;
    r.direction = wi;
    r.pdf = r.directionalPdf = e.pdf;
    r.weight = e.value*fabs(ci)/e.pdf;
    return r;
}

// Source branches of general closure predicates, named so both backends call them (EP8).
// regularize_dielectric_material: a source dielectric's relative IOR across the interface (far over incident side).
inline void source_regularize_dielectric(PT_THREAD MaterialData& material, const bool frontFace) {
    const float ior = material.sourceFresnel.w;
    material.typeEta.y = frontFace || material.typeEta.w > 0.5f ? ior : 1.0f/ior;
}

// material_has_delta_lobe for source_model(material) > 0.
inline bool source_has_delta_lobe(PT_IN(MaterialData) material) {
    if (source_model(material) == 5.0f) return false;
    if (source_model(material) == 4.0f) return true;
    return material.sourceParams.z == 0.0f ||
        (source_model(material) == 3.0f && material.sourceFresnel.w == 1.0f) ||
        (source_model(material) == 3.0f && material.typeEta.w > 0.5f);
}

// An authored dielectric normal (kMaterialFlagAuthoredDielectricNormal) is the raw interpolated normal, not faced:
// source_dielectric_frame shades the side of it that wo is on. Integrators call this once the shading normal is final,
// on their local material copy, so the direct-light cosine (source_light_cos_normal) uses that same side; with the raw
// normal it cut every light on the closure's reflection side when the surface was seen from behind (EP8-f2).
inline void source_mark_closure_side(PT_THREAD MaterialData& material, float3 n, float3 wo) {
    if (source_model(material) == 3.0f &&
        (material.materialFlags & kMaterialFlagAuthoredDielectricNormal) != 0u && dot(n, wo) < 0.0f) {
        material.materialFlags |= kMaterialFlagSourceClosureBackSide;
    }
}

// The normal a direct-light cosine uses at this material: the side the closure shades.
inline float3 source_light_cos_normal(PT_IN(MaterialData) material, float3 n) {
    return (kSourceFeatures && (material.materialFlags & kMaterialFlagSourceClosureBackSide) != 0u) ? -n : n;
}

// material_is_delta: source plastic and smooth-coated conductor always keep a non-delta lobe.
inline bool source_never_delta(PT_IN(MaterialData) material) {
    return source_model(material) == 2.0f || source_model(material) == 4.0f;
}

PT_SHARED_END

#endif  // PT_SHARED_SOURCE_CLOSURES_H
