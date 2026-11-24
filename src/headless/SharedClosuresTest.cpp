// EP8: the shared Metal/C++ closure headers (shaders/shared) compiled as C++.
//
// For each Tungsten source model, two Monte Carlo estimates of the directional albedo must agree:
//   sampled: the mean of source_sample weights (non-delta samples only);
//   integrated: the integral of source_eval value * |cos| over the sphere, with uniform directions.
// They agree only when eval, pdf and the sampler describe the same lobe, so this exercises the C++ build of the
// same text the Metal library compiles (the dialect macros and the cpp_prelude.h overloads). It checks the C++ build
// against itself: a formula that changed eval and sample alike would pass. Metal-vs-C++ agreement is EP8b's test.
#include "shared/cpp_prelude.h"
#include "shared/source_closures.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using pt_shared::float3;
using pt_shared::MaterialData;

MaterialData BaseMaterial() {
    MaterialData m;
    std::memset(&m, 0, sizeof(m));
    m.baseColorRoughness = simd_make_float4(0.8f, 0.6f, 0.4f, 0.0f);
    m.typeEta = simd_make_float4(0.0f, 1.5f, 1.5f, 0.0f);
    m.sourceReflectance = simd_make_float4(1.0f, 1.0f, 1.0f, 0.0f);
    m.sourceFresnel = simd_make_float4(0.0f, 0.0f, 0.0f, 1.5f);
    return m;
}

struct Case {
    const char* name;
    MaterialData material;
    bool backSide = false;      // wo below the authored normal (frontFace false)
    bool specularOnly = false;
};

struct Estimate {
    float3 sampled{0.0f, 0.0f, 0.0f};
    float3 integrated{0.0f, 0.0f, 0.0f};
    bool finite = true;
};

Estimate Measure(const MaterialData& m, const float3& n, const float3& wo, bool frontFace, bool specularOnly,
                 uint32_t count) {
    Estimate e;
    uint32_t state = 12345u;
    double s[3] = {0, 0, 0}, g[3] = {0, 0, 0};
    for (uint32_t i = 0; i < count; ++i) {
        pt_shared::BsdfSampleResult r = pt_shared::source_sample(m, n, wo, frontFace, state, specularOnly);
        if (!pt_shared::all_finite(r.weight) || !std::isfinite(r.pdf)) e.finite = false;
        if (!r.isDelta && r.pdf > 0.0f) {
            s[0] += r.weight.x; s[1] += r.weight.y; s[2] += r.weight.z;
        }
        // Uniform direction on the sphere.
        float z = 1.0f - 2.0f * pt_shared::rand_uniform(state);
        float phi = 2.0f * pt_shared::kPi * pt_shared::rand_uniform(state);
        float rr = std::sqrt(std::fmax(0.0f, 1.0f - z * z));
        float3 wi = simd_make_float3(rr * std::cos(phi), rr * std::sin(phi), z);
        pt_shared::BsdfEvalResult v = pt_shared::source_eval(m, n, wo, wi, specularOnly);
        if (!pt_shared::all_finite(v.value)) e.finite = false;
        float w = std::fabs(simd::dot(n, wi)) * 4.0f * pt_shared::kPi;
        g[0] += v.value.x * w; g[1] += v.value.y * w; g[2] += v.value.z * w;
    }
    e.sampled = simd_make_float3(float(s[0] / count), float(s[1] / count), float(s[2] / count));
    e.integrated = simd_make_float3(float(g[0] / count), float(g[1] / count), float(g[2] / count));
    return e;
}

}  // namespace

int main() {
    std::vector<Case> cases;
    {
        MaterialData m = BaseMaterial();  // 1: rough GGX conductor with complex IOR
        m.sourceParams = simd_make_float4(1.0f, 2.0f, 0.35f, 0.0f);
        m.conductorEta = simd_make_float4(0.2f, 0.9f, 1.1f, 1.0f);
        m.conductorK = simd_make_float4(3.9f, 2.4f, 2.2f, 1.0f);
        m.sourceReflectance.w = 1.0f;  // energy compensation on
        cases.push_back({"conductor ggx 0.35", m});
        m.sourceParams.y = 1.0f;  // Beckmann
        cases.push_back({"conductor beckmann 0.35", m});
    }
    {
        MaterialData m = BaseMaterial();  // 2: rough plastic, nonlinear
        m.sourceParams = simd_make_float4(2.0f, 2.0f, 0.3f, 1.0f);
        m.sourceFresnel = simd_make_float4(0.6f, 0.9f, 0.0f, 1.5f);
        cases.push_back({"plastic ggx 0.3", m});
        m.sourceParams.y = 1.0f;
        cases.push_back({"plastic beckmann 0.3", m});
        cases.push_back({"plastic beckmann specularOnly", m, false, true});
    }
    {
        MaterialData m = BaseMaterial();  // 3: rough dielectric with refraction
        m.sourceParams = simd_make_float4(3.0f, 2.0f, 0.3f, 1.0f);
        m.typeEta.y = 1.5f;
        m.baseColorRoughness = simd_make_float4(1.0f, 1.0f, 1.0f, 0.0f);
        cases.push_back({"dielectric ggx 0.3 refract", m});
        m.sourceParams.w = 0.0f;  // reflection only
        cases.push_back({"dielectric ggx 0.3 reflect", m});
        m.sourceParams = simd_make_float4(3.0f, 1.0f, 0.3f, 1.0f);
        cases.push_back({"dielectric beckmann 0.3", m});
        // Authored normal seen from inside: source_dielectric_frame flips the frame and inverts the IOR.
        m.sourceParams.y = 2.0f;
        m.materialFlags |= PathTracerShaderTypes::kMaterialFlagAuthoredDielectricNormal;
        cases.push_back({"dielectric authored inside", m, true});
    }
    {
        MaterialData m = BaseMaterial();  // 4: smooth coat over a rough conductor
        m.sourceParams = simd_make_float4(4.0f, 2.0f, 0.3f, 0.0f);
        m.conductorEta = simd_make_float4(0.2f, 0.9f, 1.1f, 1.0f);
        m.conductorK = simd_make_float4(3.9f, 2.4f, 2.2f, 1.0f);
        m.sourceFresnel = simd_make_float4(0.6f, 0.9f, 0.0f, 1.5f);
        cases.push_back({"smooth coat conductor", m});
    }
    {
        MaterialData m = BaseMaterial();  // 5: Oren-Nayar
        m.sourceParams = simd_make_float4(5.0f, 0.0f, 0.6f, 0.0f);
        cases.push_back({"oren-nayar 0.6", m});
    }

    const float3 n = simd_make_float3(0.0f, 0.0f, 1.0f);
    const float3 wo = simd::normalize(simd_make_float3(0.4f, 0.1f, 0.8f));
    const uint32_t count = 1u << 21;
    int failures = 0;
    for (const Case& c : cases) {
        const float3 w = c.backSide ? simd_make_float3(wo.x, wo.y, -wo.z) : wo;
        Estimate e = Measure(c.material, n, w, !c.backSide, c.specularOnly, count);
        float worst = 0.0f;
        for (int k = 0; k < 3; ++k) {
            float a = e.sampled[k], b = e.integrated[k];
            float rel = std::fabs(a - b) / std::fmax(std::fmax(a, b), 1.0e-3f);
            worst = std::fmax(worst, rel);
        }
        bool ok = e.finite && worst < 0.03f && e.sampled.x > 0.0f;
        failures += ok ? 0 : 1;
        std::printf("%-28s sampled %.4f %.4f %.4f  integrated %.4f %.4f %.4f  worst rel %.4f %s\n", c.name,
                    e.sampled.x, e.sampled.y, e.sampled.z, e.integrated.x, e.integrated.y, e.integrated.z, worst,
                    ok ? "ok" : "FAIL");
    }
    // A smooth dielectric must sample a delta event.
    {
        MaterialData m = BaseMaterial();
        m.sourceParams = simd_make_float4(3.0f, 2.0f, 0.0f, 1.0f);
        uint32_t state = 7u;
        pt_shared::BsdfSampleResult r = pt_shared::source_sample(m, n, wo, true, state, false);
        bool ok = r.isDelta && pt_shared::source_has_delta_lobe(m) && !pt_shared::source_never_delta(m);
        failures += ok ? 0 : 1;
        std::printf("%-28s delta %d %s\n", "dielectric smooth", int(r.isDelta), ok ? "ok" : "FAIL");
    }
    // A smooth thin sheet samples delta events only; regularization sets the relative IOR by side.
    {
        MaterialData m = BaseMaterial();
        m.sourceParams = simd_make_float4(3.0f, 2.0f, 0.0f, 1.0f);
        m.typeEta.w = 1.0f;
        uint32_t state = 11u;
        bool ok = true;
        for (int i = 0; i < 64; ++i) {
            pt_shared::BsdfSampleResult r = pt_shared::source_sample(m, n, wo, true, state, false);
            ok = ok && r.isDelta && r.mediumEvent == 0 && pt_shared::all_finite(r.weight);
        }
        MaterialData front = BaseMaterial(), back = BaseMaterial();
        pt_shared::source_regularize_dielectric(front, true);
        pt_shared::source_regularize_dielectric(back, false);
        ok = ok && front.typeEta.y == 1.5f && std::fabs(back.typeEta.y - 1.0f / 1.5f) < 1.0e-7f;
        failures += ok ? 0 : 1;
        std::printf("%-28s %s\n", "thin sheet, regularize", ok ? "ok" : "FAIL");
    }
    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
