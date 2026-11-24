#pragma once

// Training target for the radiance cache.
//
// A cacheable vertex x records the path throughput at its cache query point (T_q) and the radiance the path
// had accumulated by then, including x's own emission and NEE (R_b). When the path ends, the radiance it
// added after x is (R_end - R_b); divided by T_q per channel, it is one sample of what a cache hit at x
// would have added per unit of throughput. Contributions that the renderer clamped train as clamped,
// because R_end holds them as accumulated.
//
// shaders/cache/radiance_cache.metal (radiance_cache_training_target) mirrors this arithmetic for Metal;
// keep the two in step. src/headless/RadianceCacheTrainingTest.cpp tests this version.

#include <simd/simd.h>

#include <cmath>
#include <cstdint>

namespace PathTracer::RadianceCacheTraining {

// Records per path: one slot per key depth from radianceCacheMinDepth (the key caps depth at 7, so with the
// default minimum depth of 2 the slots cover depths 2..7). Matches kRadianceCacheTrainRecordCount in
// shaders/common.metal and include/MetalShaderTypes.h.
constexpr uint32_t kTrainRecordCount = 6u;
// One pixel in kTrainingPathStride is a training path in each frame: (pixelIndex + frame) % stride == 0.
constexpr uint32_t kTrainingPathStride = 8u;
// Largest accepted radianceCacheMaxRadiance (see QuantizationScale).
constexpr float kMaxRadianceLimit = 1023.0f;
// Cells stop accepting samples at this count (kRadianceCacheMaxStoredSamples in radiance_cache.metal).
constexpr uint32_t kMaxStoredSamples = 1024u;

// Diagnostics: 64-bit counters with one layout for Metal (PathtraceStats::radianceCacheStats, mirrored in
// shaders/common.metal, accumulated on the GPU from the last history reset) and Embree. Luminance sums are fixed
// point (value * kStatLumaScale, rounded per addition, each addition capped at kStatLumaAddMax): one step is
// 1.5e-5, small against the cached values of dim scenes, and a capped addition stays below 2^46.
enum RadianceCacheStat : uint32_t {
    kStatTrainRecords = 0u,             // records whose target was computed
    kStatTrainAccepted,                 // samples stored in a cell (clamped ones included)
    kStatTrainClamped,                  // targets scaled down to maxRadiance
    kStatTrainRejectedNonFinite,
    kStatTrainRejectedZeroThroughput,
    kStatTrainSaturated,                // cell already held kMaxStoredSamples
    kStatCellEvictions,                 // a cell holding another key was reset
    kStatTargetLumaFixed,               // sum of target luminance before the clamp
    kStatClampedLumaFixed,              // sum of luminance removed by the clamp
    kStatFallbackReasonBase,            // + fallback reason code (radiance_cache.metal kRadianceCacheFallback*)
    kStatDepthQueryBase = kStatFallbackReasonBase + 10u,  // + min(depth, 7): cache loads attempted
    kStatDepthHitBase = kStatDepthQueryBase + 8u,         // + min(depth, 7): cache hits
    kStatDepthHitLumaFixedBase = kStatDepthHitBase + 8u,  // + min(depth, 7): luminance of returned values
    // Metal only (Embree sums are float and leave these 0): the largest value any cell's fixed-point sum reached
    // (R, G, B, luminance second moment; a maximum, not a sum), and adds that would have wrapped a sum past UINT32_MAX.
    kStatMaxStoredSumBase = kStatDepthHitLumaFixedBase + 8u,
    kStatWrappedSums = kStatMaxStoredSumBase + 4u,
    kStatCount,
};
constexpr uint32_t kStatFallbackReasonCount = 10u;
constexpr uint32_t kStatDepthBins = 8u;
constexpr float kStatLumaScale = 65536.0f;
constexpr float kStatLumaAddMax = 1.0e9f;

inline const char* StatName(uint32_t index) {
    static const char* const kNames[kStatFallbackReasonBase] = {
        "train_records", "train_accepted", "train_clamped", "train_rejected_nonfinite",
        "train_rejected_zero_throughput", "train_saturated", "cell_evictions", "target_luma_sum",
        "clamped_luma_sum"};
    return index < kStatFallbackReasonBase ? kNames[index] : "";
}

inline const char* FallbackReasonName(uint32_t reason) {
    static const char* const kNames[kStatFallbackReasonCount] = {
        "none", "mode_off", "not_cacheable_material_or_depth", "not_cacheable_sample", "empty_cell",
        "low_confidence", "invalidated_or_same_frame", "metadata_mismatch", "bias_risk", "firefly_policy"};
    return reason < kStatFallbackReasonCount ? kNames[reason] : "unknown";
}

constexpr uint32_t kStatMaxStoredSumCount = 4u;

inline const char* MaxStoredSumName(uint32_t channel) {
    static const char* const kNames[kStatMaxStoredSumCount] = {"r", "g", "b", "luma_moment"};
    return channel < kStatMaxStoredSumCount ? kNames[channel] : "";
}

inline bool StatIsLumaFixed(uint32_t index) {
    return index == kStatTargetLumaFixed || index == kStatClampedLumaFixed ||
           (index >= kStatDepthHitLumaFixedBase && index < kStatCount);
}

inline uint64_t StatLumaToFixed(float luma) {
    const float clamped = std::fmin(std::fmax(luma, 0.0f), kStatLumaAddMax);
    return static_cast<uint64_t>(std::llround(static_cast<double>(clamped) * static_cast<double>(kStatLumaScale)));
}

enum class TargetStatus : uint32_t {
    Accepted = 0u,
    Clamped = 1u,                // largest channel scaled down to maxRadiance, hue kept
    RejectedNonFinite = 2u,
    RejectedZeroThroughput = 3u, // some channel of T_q is <= 0: that channel cannot be divided
};

struct TargetResult {
    simd::float3 target{0.0f, 0.0f, 0.0f};
    TargetStatus status = TargetStatus::RejectedNonFinite;
    float targetLuma = 0.0f;   // luminance of the target before any clamp
    float clampedLuma = 0.0f;  // luminance removed by the clamp
};

inline float Luminance(const simd::float3& v) {
    return 0.2126f * v.x + 0.7152f * v.y + 0.0722f * v.z;
}

inline bool AllFinite(const simd::float3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

inline bool IsTrainingPath(uint32_t pixelIndex, uint32_t frame) {
    return ((pixelIndex + frame) % kTrainingPathStride) == 0u;
}

// Record slot of a vertex at `depth` (radiance_cache_train_slot on Metal), or kTrainRecordCount if not trained.
inline uint32_t TrainSlot(uint32_t depth, uint32_t minDepth) {
    return (depth < minDepth || depth - minDepth >= kTrainRecordCount) ? kTrainRecordCount : depth - minDepth;
}

// Clamp-and-count: a target whose largest channel exceeds maxRadiance is scaled down to it, keeping the hue.
// Returns the clamped target; clampedLuma receives the luminance removed (0 when not clamped).
inline simd::float3 ClampTarget(const simd::float3& target, float maxRadiance, bool& clamped, float& clampedLuma) {
    const float limit = std::fmax(maxRadiance, 1.0e-4f);
    const float peak = std::fmax(target.x, std::fmax(target.y, target.z));
    clamped = peak > limit;
    clampedLuma = 0.0f;
    if (!clamped) {
        return target;
    }
    const simd::float3 scaled = target * (limit / peak);
    clampedLuma = std::fmax(Luminance(target) - Luminance(scaled), 0.0f);
    return scaled;
}

inline TargetResult ComputeTarget(const simd::float3& throughputAtQuery,
                                  const simd::float3& radianceBaseline,
                                  const simd::float3& radianceEnd,
                                  float maxRadiance) {
    TargetResult result;
    if (!AllFinite(throughputAtQuery) || !AllFinite(radianceBaseline) || !AllFinite(radianceEnd)) {
        result.status = TargetStatus::RejectedNonFinite;
        return result;
    }
    if (!(throughputAtQuery.x > 0.0f) || !(throughputAtQuery.y > 0.0f) || !(throughputAtQuery.z > 0.0f)) {
        result.status = TargetStatus::RejectedZeroThroughput;
        return result;
    }
    const simd::float3 suffix = simd_max(radianceEnd - radianceBaseline, simd_make_float3(0.0f, 0.0f, 0.0f));
    const simd::float3 target = suffix / throughputAtQuery;
    if (!AllFinite(target)) {
        result.status = TargetStatus::RejectedNonFinite;
        return result;
    }
    bool clamped = false;
    result.targetLuma = Luminance(target);
    result.target = ClampTarget(target, maxRadiance, clamped, result.clampedLuma);
    result.status = clamped ? TargetStatus::Clamped : TargetStatus::Accepted;
    return result;
}

// Fixed-point scale for a cell's summed radiance on Metal: the largest power of two s with
// kMaxStoredSamples * maxValue * s <= 4.0e9 (< UINT32_MAX = 4294967295, a 7% margin against float rounding).
// Use maxValue = maxRadiance for the RGB sums and maxRadiance^2 for the luminance second moment.
// frexp gives the exponent exactly (bound = m * 2^e, m in [0.5, 1)), so floor(log2(bound)) = e - 1 on CPU and GPU.
inline float QuantizationScale(float maxValue) {
    const float bound = 4.0e9f / (static_cast<float>(kMaxStoredSamples) * std::fmax(maxValue, 1.0e-8f));
    int exponent = 0;
    (void)std::frexp(bound, &exponent);
    return std::ldexp(1.0f, exponent - 1);
}

}  // namespace PathTracer::RadianceCacheTraining
