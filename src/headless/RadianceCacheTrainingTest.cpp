// CTest radiance_cache_training_target: the radiance cache training target on hand-built path suffixes.
// All values are dyadic so the expected results are exact in float arithmetic.

#include "renderer/RadianceCacheTraining.h"

#include <cstdio>
#include <limits>

namespace {

using namespace PathTracer::RadianceCacheTraining;

int g_failures = 0;

void Check(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAIL: %s\n", what);
        ++g_failures;
    } else {
        std::printf("ok:   %s\n", what);
    }
}

bool Equal(const simd::float3& a, const simd::float3& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

simd::float3 F3(float x, float y, float z) {
    return simd_make_float3(x, y, z);
}

}  // namespace

int main() {
    const float maxRadiance = 16.0f;

    // 1. A path suffix. At the query the path carries T_q = (0.5, 0.25, 0.125) and has accumulated
    //    R_b = 0.75 (emission and NEE at x and before). The next vertex adds T1 * L1 with T1 = T_q * w0,
    //    w0 = (0.5, 0.5, 1.0), L1 = (2, 4, 8); the vertex after adds T2 * L2 with T2 = T1 * (1, 0.5, 0.5),
    //    L2 = (1, 1, 2). Expected target: w0 * L1 + w0 * (1, 0.5, 0.5) * L2 = (1.5, 2.25, 9).
    {
        const simd::float3 tq = F3(0.5f, 0.25f, 0.125f);
        const simd::float3 rb = F3(0.75f, 0.75f, 0.75f);
        const simd::float3 t1 = tq * F3(0.5f, 0.5f, 1.0f);
        const simd::float3 t2 = t1 * F3(1.0f, 0.5f, 0.5f);
        const simd::float3 rend = rb + t1 * F3(2.0f, 4.0f, 8.0f) + t2 * F3(1.0f, 1.0f, 2.0f);
        const TargetResult r = ComputeTarget(tq, rb, rend, maxRadiance);
        Check(r.status == TargetStatus::Accepted, "suffix: accepted");
        Check(Equal(r.target, F3(1.5f, 2.25f, 9.0f)), "suffix: target = (1.5, 2.25, 9)");
        Check(r.targetLuma == Luminance(F3(1.5f, 2.25f, 9.0f)) && r.clampedLuma == 0.0f, "suffix: luma, no clamp");
    }

    // 2. Nothing added after x: a zero target is a valid sample (dropping it would bias the cache high).
    {
        const TargetResult r = ComputeTarget(F3(0.5f, 0.5f, 0.5f), F3(3.0f, 2.0f, 1.0f), F3(3.0f, 2.0f, 1.0f), maxRadiance);
        Check(r.status == TargetStatus::Accepted && Equal(r.target, F3(0.0f, 0.0f, 0.0f)), "empty suffix: zero target accepted");
    }

    // 3. Zero-throughput channel: the record is skipped (DESIGN Q4), whatever the other channels hold.
    {
        const TargetResult r = ComputeTarget(F3(0.5f, 0.0f, 0.25f), F3(0.0f, 0.0f, 0.0f), F3(1.0f, 0.0f, 1.0f), maxRadiance);
        Check(r.status == TargetStatus::RejectedZeroThroughput, "zero-throughput channel: rejected");
        const TargetResult n = ComputeTarget(F3(0.5f, -0.25f, 0.25f), F3(0.0f, 0.0f, 0.0f), F3(1.0f, 0.0f, 1.0f), maxRadiance);
        Check(n.status == TargetStatus::RejectedZeroThroughput, "negative-throughput channel: rejected");
    }

    // 4. Firefly-clamped contribution: the renderer added clamp(T1 * L1) = (0.5, 0.5, 0.5) instead of
    //    T1 * L1 = (64, 32, 16). The target uses what was accumulated: (0.5, 0.5, 0.5) / T_q.
    {
        const simd::float3 tq = F3(0.5f, 0.25f, 0.125f);
        const simd::float3 rb = F3(0.25f, 0.25f, 0.25f);
        const simd::float3 clampedContribution = F3(0.5f, 0.5f, 0.5f);
        const TargetResult r = ComputeTarget(tq, rb, rb + clampedContribution, maxRadiance);
        Check(r.status == TargetStatus::Accepted && Equal(r.target, F3(1.0f, 2.0f, 4.0f)), "firefly-clamped suffix trains as clamped");
    }

    // 5. Target above maxRadiance: scaled so the largest channel equals maxRadiance, hue kept,
    //    removed luminance reported.
    {
        const TargetResult r = ComputeTarget(F3(0.25f, 0.25f, 0.25f), F3(0.0f, 0.0f, 0.0f), F3(2.0f, 8.0f, 4.0f), maxRadiance);
        // target before clamp = (8, 32, 16); peak 32 > 16, scale 0.5 -> (4, 16, 8)
        Check(r.status == TargetStatus::Clamped, "over max: clamped");
        Check(Equal(r.target, F3(4.0f, 16.0f, 8.0f)), "over max: target = (4, 16, 8)");
        Check(r.targetLuma == Luminance(F3(8.0f, 32.0f, 16.0f)), "over max: luma before clamp");
        Check(r.clampedLuma == Luminance(F3(8.0f, 32.0f, 16.0f)) - Luminance(F3(4.0f, 16.0f, 8.0f)), "over max: clamped luma");
    }

    // 6. Large baseline, tiny throughput: R_b = 1024, T_q = 2^-20, suffix 2^-30, true target 2^-10.
    //    In float, 1024 + 2^-30 == 1024, so the suffix is lost and the target comes out 0. This is the
    //    precision limit of the difference form (DESIGN 2.2): it errs low, never negative or non-finite.
    {
        const float tiny = std::ldexp(1.0f, -20);
        const TargetResult r = ComputeTarget(F3(tiny, tiny, tiny), F3(1024.0f, 1024.0f, 1024.0f),
                                             F3(1024.0f, 1024.0f, 1024.0f) + F3(std::ldexp(1.0f, -30), 0.0f, 0.0f),
                                             maxRadiance);
        Check(r.status == TargetStatus::Accepted && r.target.x >= 0.0f && r.target.x <= std::ldexp(1.0f, -10),
              "large baseline, tiny throughput: finite, non-negative, not above the true value");
        // Representable case: R_b = 1, suffix 2^-10, T_q = 2^-4 -> target 2^-6 exactly.
        const TargetResult e = ComputeTarget(F3(0.0625f, 0.0625f, 0.0625f), F3(1.0f, 1.0f, 1.0f),
                                             F3(1.0f, 1.0f, 1.0f) + F3(std::ldexp(1.0f, -10), 0.0f, 0.0f), maxRadiance);
        Check(Equal(e.target, F3(std::ldexp(1.0f, -6), 0.0f, 0.0f)), "small suffix over a baseline: exact");
    }

    // 7. Non-finite inputs or outputs are rejected.
    {
        const float inf = std::numeric_limits<float>::infinity();
        const float nan = std::numeric_limits<float>::quiet_NaN();
        Check(ComputeTarget(F3(0.5f, 0.5f, 0.5f), F3(0.0f, 0.0f, 0.0f), F3(inf, 0.0f, 0.0f), maxRadiance).status ==
                  TargetStatus::RejectedNonFinite, "infinite radiance: rejected");
        Check(ComputeTarget(F3(nan, 0.5f, 0.5f), F3(0.0f, 0.0f, 0.0f), F3(1.0f, 1.0f, 1.0f), maxRadiance).status ==
                  TargetStatus::RejectedNonFinite, "NaN throughput: rejected");
        Check(ComputeTarget(F3(1.0e-38f, 0.5f, 0.5f), F3(0.0f, 0.0f, 0.0f), F3(1.0e30f, 0.0f, 0.0f), maxRadiance).status ==
                  TargetStatus::RejectedNonFinite, "overflowing division: rejected");
    }

    // 8. Training paths: exactly one pixel in each aligned group of kTrainingPathStride per frame,
    //    and every pixel is a training path once every kTrainingPathStride frames.
    {
        bool groupsOk = true;
        for (uint32_t frame = 0u; frame < 3u * kTrainingPathStride; ++frame) {
            for (uint32_t group = 0u; group < 64u; ++group) {
                uint32_t count = 0u;
                for (uint32_t i = 0u; i < kTrainingPathStride; ++i) {
                    count += IsTrainingPath(group * kTrainingPathStride + i, frame) ? 1u : 0u;
                }
                groupsOk = groupsOk && count == 1u;
            }
        }
        Check(groupsOk, "training paths: one per group of 8 per frame");
        uint32_t perPixel = 0u;
        for (uint32_t frame = 0u; frame < kTrainingPathStride; ++frame) {
            perPixel += IsTrainingPath(12345u, frame) ? 1u : 0u;
        }
        Check(perPixel == 1u, "training paths: each pixel once per 8 frames");
    }

    // 9. Quantization scales keep a full cell below UINT32_MAX for every accepted maxRadiance.
    {
        bool boundOk = true;
        for (float m : {1.0e-3f, 0.5f, 1.0f, 15.9f, 16.0f, 16.1f, 100.0f, 512.0f, kMaxRadianceLimit}) {
            const double sumMax = double(kMaxStoredSamples) * double(m) * double(QuantizationScale(m));
            const double momentMax = double(kMaxStoredSamples) * double(m) * double(m) * double(QuantizationScale(m * m));
            boundOk = boundOk && sumMax <= 4294967295.0 && momentMax <= 4294967295.0 &&
                      sumMax > 4.0e9 / 2.0 && momentMax > 4.0e9 / 2.0;
        }
        Check(boundOk, "quantization: full cell fits uint32 and uses more than half the range");
        Check(QuantizationScale(16.0f) == 131072.0f, "quantization: sum scale 2^17 at maxRadiance 16");
        Check(QuantizationScale(256.0f) == 8192.0f, "quantization: moment scale 2^13 at maxRadiance 16");
    }

    {
        // Luminance counters: 2^-16 steps (a dim cell's value of 0.001 is 66 steps), capped additions, no sign.
        Check(StatLumaToFixed(1.0f) == 65536u && StatLumaToFixed(0.001f) == 66u,
              "luma counter: 1.0 and 0.001 in 2^-16 steps");
        Check(StatLumaToFixed(-1.0f) == 0u && StatLumaToFixed(1.0e12f) == 65536000000000ull,
              "luma counter: negative is 0, additions capped at 1e9");
    }

    std::printf("%s (%d failure%s)\n", g_failures == 0 ? "PASS" : "FAIL", g_failures, g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
