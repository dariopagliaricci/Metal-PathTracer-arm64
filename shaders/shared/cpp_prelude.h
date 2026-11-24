// C++ side of the headers in shaders/shared (EP8). A C++ translation unit includes this first, then the shared
// headers it needs; their functions land in namespace pt_shared. The Metal shaders never include this file.
//
// It gives the shared text Metal's vocabulary: the vector types (Apple simd extended vectors, which accept
// float3(x) splats, .xyz swizzles and scalar-vector arithmetic as Metal does), MaterialData and the material flags
// from MetalShaderTypes.h, kPi, and Metal's math functions as overloads. Unqualified calls inside namespace
// pt_shared find these overloads first, so the shared code is written once with Metal's names.
#ifndef PT_SHARED_CPP_PRELUDE
#define PT_SHARED_CPP_PRELUDE

#ifdef __METAL_VERSION__
#error "shared/cpp_prelude.h is for C++ only"
#endif

#include <simd/simd.h>

#include <cmath>
#include <cstdint>

#include "MetalShaderTypes.h"

namespace pt_shared {

using float2 = simd::float2;
using float3 = simd::float3;
using float4 = simd::float4;
using int2 = simd::int2;
using int3 = simd::int3;
using uint2 = simd::uint2;
using uint = unsigned int;

using MaterialData = PathTracerShaderTypes::MaterialData;
using PathTracerShaderTypes::kMaterialFlagAuthoredDielectricNormal;
using PathTracerShaderTypes::kMaterialFlagSourceClosureBackSide;

constexpr float kPi = 3.14159265358979323846f;
// Embree has one build; the Metal source-feature switch is always on for it (shared/source_features.h).
constexpr bool kSourceFeatures = true;

// Scalars. min/max/clamp follow Metal's fmin/fmax semantics (a NaN operand yields the other operand).
inline float fabs(float x) { return std::fabs(x); }
inline float sqrt(float x) { return std::sqrt(x); }
inline float rsqrt(float x) { return 1.0f / std::sqrt(x); }
inline float exp(float x) { return std::exp(x); }
inline float log(float x) { return std::log(x); }
inline float sin(float x) { return std::sin(x); }
inline float cos(float x) { return std::cos(x); }
inline float acos(float x) { return std::acos(x); }
inline float tan(float x) { return std::tan(x); }
inline float min(float a, float b) { return std::fmin(a, b); }
inline float max(float a, float b) { return std::fmax(a, b); }
inline uint min(uint a, uint b) { return a < b ? a : b; }
inline uint max(uint a, uint b) { return a > b ? a : b; }
inline float clamp(float x, float lo, float hi) { return std::fmin(std::fmax(x, lo), hi); }
inline float saturate(float x) { return clamp(x, 0.0f, 1.0f); }
inline float mix(float a, float b, float t) { return a + (b - a) * t; }

// Vectors.
inline float dot(float2 a, float2 b) { return simd::dot(a, b); }
inline float dot(float3 a, float3 b) { return simd::dot(a, b); }
inline float3 cross(float3 a, float3 b) { return simd::cross(a, b); }
inline float length(float3 v) { return std::sqrt(simd::dot(v, v)); }
inline float3 normalize(float3 v) { return v * (1.0f / std::sqrt(simd::dot(v, v))); }
inline float3 reflect(float3 i, float3 n) { return i - 2.0f * simd::dot(n, i) * n; }
inline float3 fabs(float3 v) { return simd_make_float3(std::fabs(v.x), std::fabs(v.y), std::fabs(v.z)); }
inline float3 sqrt(float3 v) { return simd_make_float3(std::sqrt(v.x), std::sqrt(v.y), std::sqrt(v.z)); }
inline float3 exp(float3 v) { return simd_make_float3(std::exp(v.x), std::exp(v.y), std::exp(v.z)); }
inline float3 min(float3 a, float3 b) {
    return simd_make_float3(std::fmin(a.x, b.x), std::fmin(a.y, b.y), std::fmin(a.z, b.z));
}
inline float3 max(float3 a, float3 b) {
    return simd_make_float3(std::fmax(a.x, b.x), std::fmax(a.y, b.y), std::fmax(a.z, b.z));
}
inline float3 clamp(float3 x, float3 lo, float3 hi) { return min(max(x, lo), hi); }
inline float3 saturate(float3 x) { return clamp(x, float3(0.0f), float3(1.0f)); }
inline float3 mix(float3 a, float3 b, float t) { return a + (b - a) * t; }
inline float2 mix(float2 a, float2 b, float t) { return a + (b - a) * t; }
// Comparisons of extended vectors yield integer masks (-1 true, 0 false), as Metal's bool vectors.
inline bool all(int3 m) { return simd_all(m); }
inline bool any(int3 m) { return simd_any(m); }
inline bool all(int2 m) { return simd_all(m); }
inline bool any(int2 m) { return simd_any(m); }
inline bool all_finite(float3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }

// shaders/core/random_math.metal: the PCG hash stream (Embree's Rng::nextFloat is the same function of its state).
inline uint pcg_hash(uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
inline float rand_uniform(uint& state) {
    state = pcg_hash(state);
    return float(state) / 4294967296.0f;
}

}  // namespace pt_shared

#endif  // PT_SHARED_CPP_PRELUDE
