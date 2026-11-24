#pragma once

#include <algorithm>
#include <cmath>
#include <simd/simd.h>

// Host mirror of the spectral sensor in shaders/spectral/spectral_transport.metal. The white balance is only
// correct for the curves the shader evaluates, so spectral_sensor_rgb and Response() must stay identical.
namespace SpectralSensor {

struct WavelengthRange {
    float minNm = 380.0f;
    float maxNm = 720.0f;
};

// The range clamp UniformBuilder applies before upload (min in [300, 829], max in [min + 1, 830]).
inline WavelengthRange ClampRange(const float requestedMinNm, const float requestedMaxNm) {
    WavelengthRange range;
    range.minNm = std::clamp(std::min(requestedMinNm, requestedMaxNm - 1.0f), 300.0f, 829.0f);
    range.maxNm = std::clamp(std::max(requestedMaxNm, range.minNm + 1.0f), range.minNm + 1.0f, 830.0f);
    return range;
}

inline double Gaussian(const double wavelengthNm, const double centerNm, const double widthNm) {
    const double x = (wavelengthNm - centerNm) / std::max(widthNm, 1.0);
    return std::exp(-0.5 * x * x);
}

// s(lambda): three Gaussians (610/545/455 nm, widths 48/42/36) divided by their largest component.
inline simd::double3 Response(const double wavelengthNm) {
    const simd::double3 response = simd_make_double3(Gaussian(wavelengthNm, 610.0, 48.0),
                                                     Gaussian(wavelengthNm, 545.0, 42.0),
                                                     Gaussian(wavelengthNm, 455.0, 36.0));
    const double normalizer = std::max(std::max(response.x, response.y), response.z);
    if (normalizer <= 1.0e-5) {
        return simd_make_double3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
    }
    return response / normalizer;
}

constexpr int kWhiteBalanceSamples = 4096;

// W_c = 1 / E[s_c(lambda)] for lambda uniform over the range (midpoint rule), so that a flat spectrum
// accumulates to RGB (1, 1, 1): E[W_c * s_c(lambda)] = 1.
inline simd::float3 WhiteBalance(const WavelengthRange range) {
    simd::double3 sum = simd_make_double3(0.0, 0.0, 0.0);
    const double span = double(range.maxNm) - double(range.minNm);
    for (int i = 0; i < kWhiteBalanceSamples; ++i) {
        const double wavelengthNm = double(range.minNm) + span * (double(i) + 0.5) / double(kWhiteBalanceSamples);
        sum += Response(wavelengthNm);
    }
    const simd::double3 mean = sum / double(kWhiteBalanceSamples);
    return simd_make_float3(float(1.0 / mean.x), float(1.0 / mean.y), float(1.0 / mean.z));
}

// Uniforms are rebuilt every frame; the range rarely changes, so reuse the last result per thread.
inline simd::float3 WhiteBalanceCached(const WavelengthRange range) {
    thread_local WavelengthRange cachedRange{-1.0f, -1.0f};
    thread_local simd::float3 cachedWhiteBalance = simd_make_float3(0.0f, 0.0f, 0.0f);
    if (range.minNm != cachedRange.minNm || range.maxNm != cachedRange.maxNm) {
        cachedWhiteBalance = WhiteBalance(range);
        cachedRange = range;
    }
    return cachedWhiteBalance;
}

}  // namespace SpectralSensor
