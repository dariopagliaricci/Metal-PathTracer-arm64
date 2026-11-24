// Shared scalar/vector helpers and the orthonormal-basis and cosine-hemisphere sampling the closures use (EP8).
// Metal includes this from scene/surface_access.metal; safe_normalize is forward-declared there for earlier code.
#ifndef PT_SHARED_MATH_H
#define PT_SHARED_MATH_H

#include "portability.h"

PT_SHARED_BEGIN

inline float safe_sqrt(const float value) {
    return sqrt(max(value, 0.0f));
}

inline float3 safe_normalize(const float3 v) {
    float len2 = dot(v, v);
    if (len2 <= 0.0f) {
        return PT_F3(0.0f, 0.0f, 0.0f);
    }
    return v * rsqrt(len2);
}

inline float clamp01(const float value) {
    return clamp(value, 0.0f, 1.0f);
}

inline float3 clamp01(const float3 value) {
    return clamp(value, float3(0.0f), float3(1.0f));
}

inline void build_onb(const float3 normal,
                      PT_THREAD float3& tangent,
                      PT_THREAD float3& bitangent) {
    float3 up = (fabs(normal.z) < 0.999f) ? PT_F3(0.0f, 0.0f, 1.0f) : PT_F3(1.0f, 0.0f, 0.0f);
    tangent = normalize(cross(up, normal));
    bitangent = cross(normal, tangent);
}

inline float3 to_world(const float3 local, const float3 normal) {
    float3 tangent;
    float3 bitangent;
    build_onb(normal, tangent, bitangent);
    return local.x * tangent + local.y * bitangent + local.z * normal;
}

inline float3 to_local(const float3 vector, const float3 normal) {
    float3 tangent;
    float3 bitangent;
    build_onb(normal, tangent, bitangent);
    return PT_F3(dot(vector, tangent), dot(vector, bitangent), dot(vector, normal));
}

inline float3 sample_cosine_hemisphere(PT_THREAD uint& state) {
    float r1 = rand_uniform(state);
    float r2 = rand_uniform(state);
    float phi = 2.0f * kPi * r2;
    float r = sqrt(max(r1, 0.0f));
    float x = cos(phi) * r;
    float y = sin(phi) * r;
    float z = sqrt(max(1.0f - r1, 0.0f));
    return PT_F3(x, y, z);
}

PT_SHARED_END

#endif  // PT_SHARED_MATH_H
