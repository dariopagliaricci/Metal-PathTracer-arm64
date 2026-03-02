#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "headless/EmbreeHeadlessRenderer.h"
#include "renderer/EnvImportanceSampler.h"
#include "MetalShaderTypes.h"
#include <simd/simd.h>

#if defined(PATH_TRACER_ENABLE_EMBREE)
#include <embree4/rtcore.h>
#endif

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kEpsilon = 1.0e-4f;
constexpr float kSpecularNeePdfFloor = 1.0e-4f;
constexpr float kSpecularNeeInvPdfClamp = 1.0e4f;
constexpr float kMisWeightClampMin = 1.0e-4f;
constexpr float kMisWeightClampMax = 0.9999f;

struct CameraBasis {
    simd::float3 origin{0.0f, 0.0f, 0.0f};
    simd::float3 lowerLeft{0.0f, 0.0f, 0.0f};
    simd::float3 horizontal{0.0f, 0.0f, 0.0f};
    simd::float3 vertical{0.0f, 0.0f, 0.0f};
    simd::float3 u{1.0f, 0.0f, 0.0f};
    simd::float3 v{0.0f, 1.0f, 0.0f};
    float lensRadius = 0.0f;
};

struct Ray {
    simd::float3 origin{0.0f, 0.0f, 0.0f};
    simd::float3 direction{0.0f, 0.0f, -1.0f};
};

struct Rng {
    uint32_t state = 1u;

    static uint32_t Hash(uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352d;
        x ^= x >> 15;
        x *= 0x846ca68b;
        x ^= x >> 16;
        return x;
    }

    float nextFloat() {
        state = Hash(state);
        return static_cast<float>(state & 0x00FFFFFFu) / 16777216.0f;
    }
};

struct EnvironmentMap {
    std::vector<float> rgba;
    uint32_t width = 0;
    uint32_t height = 0;
    PathTracer::EnvImportanceDistribution distribution;
    bool hasDistribution = false;
};

enum class GeometryType {
    Mesh = 0,
    Spheres = 1,
    Rectangles = 2,
};

struct GeometryData {
    GeometryType type = GeometryType::Mesh;
    const simd::float3* positions = nullptr;
    const simd::float3* normals = nullptr;
    const simd::float2* uvs = nullptr;
    const uint32_t* indices = nullptr;
    const simd::float4* spheres = nullptr;
    const uint32_t* materialIndices = nullptr;
    uint32_t indexCount = 0;
    uint32_t count = 0;
    uint32_t materialIndex = 0;
};

struct HitInfo {
    simd::float3 position{0.0f, 0.0f, 0.0f};
    simd::float3 normal{0.0f, 1.0f, 0.0f};
    simd::float3 shadingNormal{0.0f, 1.0f, 0.0f};
    float t = 0.0f;
    uint32_t materialIndex = 0;
    bool frontFace = true;
    bool twoSided = false;
    GeometryType primitiveType = GeometryType::Mesh;
    uint32_t primitiveIndex = 0;
};

struct RectLightInfo {
    uint32_t rectIndex = 0;
    simd::float3 corner{0.0f, 0.0f, 0.0f};
    simd::float3 edgeU{0.0f, 0.0f, 0.0f};
    simd::float3 edgeV{0.0f, 0.0f, 0.0f};
    simd::float3 normal{0.0f, 1.0f, 0.0f};
    simd::float3 baseEmission{0.0f, 0.0f, 0.0f};
    bool twoSided = false;
    bool emissionUsesEnv = false;
    float area = 0.0f;
};

struct BsdfEval {
    simd::float3 value{0.0f, 0.0f, 0.0f};
    float pdf = 0.0f;
    bool isDelta = false;
};

struct BsdfSample {
    simd::float3 direction{0.0f, 0.0f, 0.0f};
    simd::float3 weight{0.0f, 0.0f, 0.0f};
    float pdf = 0.0f;
    bool isDelta = false;
};

bool IsFinite(const simd::float3& value);

struct FireflyClampParams {
    float clampFactor = 0.0f;
    float clampFloor = 0.0f;
    float throughputClamp = 0.0f;
    float specularTailClampBase = 0.0f;
    float specularTailClampRoughnessScale = 0.0f;
    float minSpecularPdf = 1.0e-8f;
    float enabled = 0.0f;
};

float DegreesToRadians(float degrees) {
    return degrees * (kPi / 180.0f);
}

CameraBasis BuildCamera(const PathTracer::RenderSettings& settings, uint32_t width, uint32_t height) {
    CameraBasis basis;

    float aspect = width > 0 ? static_cast<float>(width) / static_cast<float>(height) : 1.0f;
    float vfov = std::clamp(settings.cameraVerticalFov, 1.0f, 179.0f);
    float defocusAngle = std::max(settings.cameraDefocusAngle, 0.0f);

    const float theta = DegreesToRadians(vfov);
    const float h = std::tan(theta * 0.5f);
    const float viewportHeight = 2.0f * h;
    const float viewportWidth = aspect * viewportHeight;

    const float distance = std::max(settings.cameraDistance, 0.1f);
    const float yaw = settings.cameraYaw;
    const float pitch = settings.cameraPitch;
    const float cosPitch = std::cos(pitch);
    const float sinPitch = std::sin(pitch);
    const float cosYaw = std::cos(yaw);
    const float sinYaw = std::sin(yaw);

    const simd::float3 offset = {
        distance * cosPitch * cosYaw,
        distance * sinPitch,
        distance * cosPitch * sinYaw
    };

    const simd::float3 lookAt = settings.cameraTarget;
    const simd::float3 lookFrom = lookAt + offset;
    const simd::float3 vup = {0.0f, 1.0f, 0.0f};

    const simd::float3 w = simd::normalize(lookFrom - lookAt);
    const simd::float3 u = simd::normalize(simd::cross(vup, w));
    const simd::float3 v = simd::cross(w, u);

    float focusDist = settings.cameraFocusDistance;
    if (focusDist <= 0.0f) {
        focusDist = distance;
    }

    basis.horizontal = focusDist * viewportWidth * u;
    basis.vertical = focusDist * viewportHeight * v;
    basis.lowerLeft = lookFrom - 0.5f * basis.horizontal - 0.5f * basis.vertical - focusDist * w;
    basis.origin = lookFrom;
    basis.u = u;
    basis.v = v;
    basis.lensRadius = focusDist * std::tan(DegreesToRadians(defocusAngle * 0.5f));

    return basis;
}

simd::float3 SampleInUnitDisk(Rng& rng) {
    for (int i = 0; i < 8; ++i) {
        float x = rng.nextFloat() * 2.0f - 1.0f;
        float y = rng.nextFloat() * 2.0f - 1.0f;
        float r2 = x * x + y * y;
        if (r2 <= 1.0f) {
            return {x, y, 0.0f};
        }
    }
    return {0.0f, 0.0f, 0.0f};
}

Ray GenerateCameraRay(const CameraBasis& camera,
                      uint32_t width,
                      uint32_t height,
                      uint32_t x,
                      uint32_t y,
                      Rng& rng) {
    float u = (static_cast<float>(x) + rng.nextFloat()) / static_cast<float>(width);
    float v = (static_cast<float>(y) + rng.nextFloat()) / static_cast<float>(height);
    v = 1.0f - v;

    simd::float3 direction = camera.lowerLeft + u * camera.horizontal + v * camera.vertical - camera.origin;
    simd::float3 origin = camera.origin;
    if (camera.lensRadius > 0.0f) {
        simd::float3 disk = SampleInUnitDisk(rng) * camera.lensRadius;
        simd::float3 offset = camera.u * disk.x + camera.v * disk.y;
        origin += offset;
        direction -= offset;
    }

    return Ray{origin, simd::normalize(direction)};
}

simd::float3 SkyColor(const simd::float3& direction) {
    simd::float3 unit = simd::normalize(direction);
    float t = 0.5f * (unit.y + 1.0f);
    return simd_make_float3(1.0f, 1.0f, 1.0f) * (1.0f - t) +
           simd_make_float3(0.5f, 0.7f, 1.0f) * t;
}

simd::float3 SampleEnvironment(const EnvironmentMap& env,
                               const simd::float3& direction,
                               float rotation,
                               float intensity) {
    if (env.width == 0 || env.height == 0 || env.rgba.empty()) {
        return {0.0f, 0.0f, 0.0f};
    }

    simd::float3 unit = simd::normalize(direction);
    float cosTheta = std::cos(rotation);
    float sinTheta = std::sin(rotation);
    simd::float3 rotated = {unit.x * cosTheta - unit.z * sinTheta,
                            unit.y,
                            unit.x * sinTheta + unit.z * cosTheta};
    float u = (std::atan2(rotated.z, rotated.x) + kPi) / (2.0f * kPi);
    float v = 0.5f - std::asin(std::clamp(rotated.y, -1.0f, 1.0f)) / kPi;

    float fx = u * static_cast<float>(env.width) - 0.5f;
    float fy = v * static_cast<float>(env.height) - 0.5f;
    int x0 = static_cast<int>(std::floor(fx));
    int y0 = static_cast<int>(std::floor(fy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float tx = fx - static_cast<float>(x0);
    float ty = fy - static_cast<float>(y0);

    auto wrap = [](int value, int max) {
        int m = value % max;
        return m < 0 ? m + max : m;
    };

    x0 = wrap(x0, static_cast<int>(env.width));
    x1 = wrap(x1, static_cast<int>(env.width));
    y0 = std::clamp(y0, 0, static_cast<int>(env.height) - 1);
    y1 = std::clamp(y1, 0, static_cast<int>(env.height) - 1);

    auto fetch = [&](int px, int py) {
        size_t index = (static_cast<size_t>(py) * env.width + static_cast<size_t>(px)) * 4ull;
        return simd_make_float3(env.rgba[index + 0], env.rgba[index + 1], env.rgba[index + 2]);
    };

    simd::float3 c00 = fetch(x0, y0);
    simd::float3 c10 = fetch(x1, y0);
    simd::float3 c01 = fetch(x0, y1);
    simd::float3 c11 = fetch(x1, y1);

    simd::float3 c0 = c00 * (1.0f - tx) + c10 * tx;
    simd::float3 c1 = c01 * (1.0f - tx) + c11 * tx;
    simd::float3 color = c0 * (1.0f - ty) + c1 * ty;
    return color * std::max(intensity, 0.0f);
}

simd::float3 EvaluateBackground(const PathTracer::RenderSettings& settings,
                                const EnvironmentMap* env,
                                const simd::float3& direction) {
    switch (settings.backgroundMode) {
        case PathTracer::RenderSettings::BackgroundMode::Solid:
            return settings.backgroundColor;
        case PathTracer::RenderSettings::BackgroundMode::Environment:
            if (env) {
                return SampleEnvironment(*env, direction,
                                         settings.environmentRotation,
                                         settings.environmentIntensity);
            }
            return SkyColor(direction);
        case PathTracer::RenderSettings::BackgroundMode::Gradient:
        default:
            return SkyColor(direction);
    }
}

simd::float3 Reflect(const simd::float3& v, const simd::float3& n) {
    return v - 2.0f * simd::dot(v, n) * n;
}

bool Refract(const simd::float3& v, const simd::float3& n, float eta, simd::float3& out) {
    float cosTheta = std::min(-simd::dot(v, n), 1.0f);
    simd::float3 rOutPerp = eta * (v + cosTheta * n);
    float k = 1.0f - simd::dot(rOutPerp, rOutPerp);
    if (k < 0.0f) {
        return false;
    }
    simd::float3 rOutParallel = -std::sqrt(k) * n;
    out = rOutPerp + rOutParallel;
    return true;
}

float Schlick(float cosine, float refIdx) {
    float r0 = (1.0f - refIdx) / (1.0f + refIdx);
    r0 = r0 * r0;
    float t = 1.0f - cosine;
    return r0 + (1.0f - r0) * t * t * t * t * t;
}

struct Onb {
    simd::float3 tangent{1.0f, 0.0f, 0.0f};
    simd::float3 bitangent{0.0f, 1.0f, 0.0f};
    simd::float3 normal{0.0f, 0.0f, 1.0f};
};

Onb BuildOnb(const simd::float3& n) {
    Onb onb;
    onb.normal = simd::normalize(n);
    simd::float3 up = (std::fabs(onb.normal.z) < 0.999f)
                          ? simd_make_float3(0.0f, 0.0f, 1.0f)
                          : simd_make_float3(1.0f, 0.0f, 0.0f);
    onb.tangent = simd::normalize(simd::cross(up, onb.normal));
    onb.bitangent = simd::cross(onb.normal, onb.tangent);
    return onb;
}

simd::float3 SampleCosineHemisphere(Rng& rng, const simd::float3& n, float& outPdf) {
    float r1 = rng.nextFloat();
    float r2 = rng.nextFloat();
    float r = std::sqrt(std::max(r1, 0.0f));
    float phi = 2.0f * kPi * r2;
    float x = std::cos(phi) * r;
    float y = std::sin(phi) * r;
    float z = std::sqrt(std::max(1.0f - r1, 0.0f));
    simd::float3 local = {x, y, z};
    Onb onb = BuildOnb(n);
    simd::float3 dir = local.x * onb.tangent + local.y * onb.bitangent + local.z * onb.normal;
    outPdf = z / kPi;
    return simd::normalize(dir);
}

float SchlickWeight(float cosTheta) {
    float m = std::clamp(1.0f - cosTheta, 0.0f, 1.0f);
    float m2 = m * m;
    return m2 * m2 * m;
}

simd::float3 SchlickFresnel(const simd::float3& f0, float cosTheta) {
    return f0 + (simd_make_float3(1.0f, 1.0f, 1.0f) - f0) * SchlickWeight(cosTheta);
}

float Luminance(const simd::float3& value) {
    return 0.2126f * value.x + 0.7152f * value.y + 0.0722f * value.z;
}

FireflyClampParams MakeFireflyParams(const PathTracer::RenderSettings& settings) {
    FireflyClampParams params;
    params.clampFactor = std::max(settings.fireflyClampFactor, 0.0f);
    params.clampFloor = std::max(settings.fireflyClampFloor, 0.0f);
    params.throughputClamp = std::max(settings.throughputClamp, 0.0f);
    params.specularTailClampBase = std::max(settings.specularTailClampBase, 0.0f);
    params.specularTailClampRoughnessScale = std::max(settings.specularTailClampRoughnessScale, 0.0f);
    params.minSpecularPdf = std::max(settings.minSpecularPdf, 1.0e-8f);
    params.enabled = settings.fireflyClampEnabled ? 1.0f : 0.0f;
    return params;
}

float LoadEmbreeEmissionScale() {
    const char* value = std::getenv("PATH_TRACER_EMBREE_EMISSION_SCALE");
    if (!value || *value == '\0') {
        return 1.0f;
    }
    char* end = nullptr;
    float parsed = std::strtof(value, &end);
    if (end == value || !std::isfinite(parsed) || parsed <= 0.0f) {
        return 1.0f;
    }
    return parsed;
}

simd::float3 ClampFireflyContribution(const simd::float3& throughput,
                                      const simd::float3& contribution,
                                      const FireflyClampParams& params) {
    simd::float3 combined = throughput * contribution;
    if (!IsFinite(combined)) {
        return simd_make_float3(0.0f, 0.0f, 0.0f);
    }

    simd::float3 positive = simd::max(combined, simd_make_float3(0.0f, 0.0f, 0.0f));
    if (params.enabled < 0.5f) {
        return positive;
    }

    float lum = Luminance(positive);
    simd::float3 throughputPositive = simd::max(throughput, simd_make_float3(0.0f, 0.0f, 0.0f));
    float throughputLum = Luminance(throughputPositive);
    float maxLum = std::max(throughputLum * params.clampFactor, params.clampFloor);

    if (lum > maxLum && lum > 0.0f) {
        float scale = maxLum / std::max(lum, 1.0e-6f);
        combined *= scale;
        positive = simd::max(combined, simd_make_float3(0.0f, 0.0f, 0.0f));
    }

    return positive;
}

float ClampSpecularPdf(float pdf, const FireflyClampParams& params) {
    float minPdf = std::max(params.minSpecularPdf, 1.0e-8f);
    if (!std::isfinite(pdf)) {
        return minPdf;
    }
    return std::max(pdf, minPdf);
}

simd::float3 ClampPathThroughput(const simd::float3& throughput,
                                 const FireflyClampParams& params) {
    if (!IsFinite(throughput)) {
        return simd_make_float3(0.0f, 0.0f, 0.0f);
    }
    if (params.enabled < 0.5f || params.throughputClamp <= 0.0f) {
        return throughput;
    }
    simd::float3 positive = simd::max(throughput, simd_make_float3(0.0f, 0.0f, 0.0f));
    float lum = Luminance(positive);
    if (lum > params.throughputClamp && lum > 0.0f) {
        float scale = params.throughputClamp / std::max(lum, 1.0e-6f);
        return throughput * scale;
    }
    return throughput;
}

simd::float3 ClampSpecularTail(const simd::float3& value,
                               float roughness,
                               const simd::float3& f0,
                               const FireflyClampParams& params) {
    if (!IsFinite(value)) {
        return simd_make_float3(0.0f, 0.0f, 0.0f);
    }
    simd::float3 positive = simd::max(value, simd_make_float3(0.0f, 0.0f, 0.0f));
    if (params.enabled < 0.5f) {
        return positive;
    }
    float strength = std::max({f0.x, f0.y, f0.z, 1.0e-3f});
    float limit = (params.specularTailClampBase +
                   params.specularTailClampRoughnessScale * roughness) * strength;
    limit = std::max(limit, params.clampFloor);
    float lum = Luminance(positive);
    if (lum > limit && lum > 0.0f) {
        float scale = limit / std::max(lum, 1.0e-6f);
        positive *= scale;
    }
    return positive;
}

float FresnelDielectricExact(float cosThetaI, float etaI, float etaT, float& outCosThetaT) {
    cosThetaI = std::clamp(cosThetaI, -1.0f, 1.0f);
    float absCosThetaI = std::fabs(cosThetaI);
    float sinThetaI2 = std::max(0.0f, 1.0f - absCosThetaI * absCosThetaI);
    float eta = etaI / etaT;
    float sinThetaT2 = eta * eta * sinThetaI2;

    if (sinThetaT2 >= 1.0f) {
        outCosThetaT = 0.0f;
        return 1.0f;
    }

    float cosThetaT = std::sqrt(std::max(0.0f, 1.0f - sinThetaT2));
    outCosThetaT = cosThetaT;

    float etaICosThetaI = etaI * absCosThetaI;
    float etaTCosThetaT = etaT * cosThetaT;

    float rsNum = etaICosThetaI - etaTCosThetaT;
    float rsDen = etaICosThetaI + etaTCosThetaT;
    float rpNum = etaT * absCosThetaI - etaI * cosThetaT;
    float rpDen = etaT * absCosThetaI + etaI * cosThetaT;

    float rs = rsNum / rsDen;
    float rp = rpNum / rpDen;
    return 0.5f * (rs * rs + rp * rp);
}

float DielectricF0FromIor(float ior) {
    float eta = std::max(ior, 1.0f);
    float num = eta - 1.0f;
    float den = std::max(eta + 1.0f, 1.0e-6f);
    float f0 = (num / den) * (num / den);
    return std::clamp(f0, 0.0f, 0.99f);
}

simd::float3 FresnelConductor(float cosThetaI, const simd::float3& eta, const simd::float3& k) {
    cosThetaI = std::clamp(cosThetaI, -1.0f, 1.0f);
    float cos2 = cosThetaI * cosThetaI;
    float sin2 = std::max(0.0f, 1.0f - cos2);

    simd::float3 eta2 = eta * eta;
    simd::float3 k2 = k * k;

    simd::float3 t0 = eta2 - k2 - simd_make_float3(sin2, sin2, sin2);
    simd::float3 a2plusb2 = simd::sqrt(simd::max(t0 * t0 + 4.0f * eta2 * k2,
                                                 simd_make_float3(0.0f, 0.0f, 0.0f)));
    simd::float3 a = simd::sqrt(simd::max(0.5f * (a2plusb2 + t0),
                                          simd_make_float3(0.0f, 0.0f, 0.0f)));

    simd::float3 term1 = a2plusb2 + simd_make_float3(cos2, cos2, cos2);
    simd::float3 term2 = 2.0f * simd_make_float3(cosThetaI, cosThetaI, cosThetaI) * a;
    simd::float3 rs = (term1 - term2) / (term1 + term2);

    simd::float3 term3 = simd_make_float3(cos2, cos2, cos2) * a2plusb2 +
                         simd_make_float3(sin2 * sin2, sin2 * sin2, sin2 * sin2);
    simd::float3 term4 = term2 * simd_make_float3(sin2, sin2, sin2);
    simd::float3 rp = (term3 - term4) / (term3 + term4);

    return simd::clamp(0.5f * (rs * rs + rp * rp),
                       simd_make_float3(0.0f, 0.0f, 0.0f),
                       simd_make_float3(1.0f, 1.0f, 1.0f));
}

float GgxLambda(float alpha, float cosTheta) {
    float absCosTheta = std::fabs(cosTheta);
    if (absCosTheta <= 0.0f) {
        return 0.0f;
    }
    float sinTheta = std::sqrt(std::max(0.0f, 1.0f - absCosTheta * absCosTheta));
    if (sinTheta == 0.0f) {
        return 0.0f;
    }
    float tanTheta = sinTheta / absCosTheta;
    float a = alpha * tanTheta;
    return (-1.0f + std::sqrt(1.0f + a * a)) * 0.5f;
}

float GgxG1(float alpha, float cosTheta) {
    return 1.0f / (1.0f + GgxLambda(alpha, cosTheta));
}

float GgxDistribution(float alpha, float cosThetaH) {
    float absCosThetaH = std::fabs(cosThetaH);
    float a2 = alpha * alpha;
    float denom = absCosThetaH * absCosThetaH * (a2 - 1.0f) + 1.0f;
    return a2 / (kPi * denom * denom);
}

float GgxPdf(float alpha, const simd::float3& normal, const simd::float3& wo, const simd::float3& wi) {
    simd::float3 wh = simd::normalize(wo + wi);
    float cosThetaH = simd::dot(normal, wh);
    float dotWoWh = simd::dot(wo, wh);
    float denom = 4.0f * std::max(dotWoWh, 1.0e-6f);
    float d = GgxDistribution(alpha, std::max(cosThetaH, 0.0f));
    return d * std::max(cosThetaH, 0.0f) / denom;
}

simd::float3 SampleGgxHalfVector(Rng& rng, float alpha, const simd::float3& n) {
    float u1 = rng.nextFloat();
    float u2 = rng.nextFloat();
    float phi = 2.0f * kPi * u1;
    float denom = 1.0f + (alpha * alpha - 1.0f) * u2;
    float cosTheta = std::sqrt(std::max((1.0f - u2) / std::max(denom, 1.0e-6f), 0.0f));
    float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
    simd::float3 local = {std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta};
    Onb onb = BuildOnb(n);
    simd::float3 h = local.x * onb.tangent + local.y * onb.bitangent + local.z * onb.normal;
    return simd::normalize(h);
}

simd::float3 ToLocal(const simd::float3& v, const Onb& onb) {
    return simd_make_float3(simd::dot(v, onb.tangent),
                            simd::dot(v, onb.bitangent),
                            simd::dot(v, onb.normal));
}

simd::float3 ToWorld(const simd::float3& v, const Onb& onb) {
    return v.x * onb.tangent + v.y * onb.bitangent + v.z * onb.normal;
}

simd::float3 SampleGgxVndf(Rng& rng, float roughness, const simd::float3& normal, const simd::float3& wo) {
    float alpha = std::max(roughness * roughness, 1.0e-4f);
    Onb onb = BuildOnb(normal);
    simd::float3 woLocal = ToLocal(simd::normalize(wo), onb);
    woLocal.z = std::max(woLocal.z, 1.0e-6f);

    simd::float3 vh = simd::normalize(simd_make_float3(alpha * woLocal.x,
                                                       alpha * woLocal.y,
                                                       woLocal.z));
    float lensq = vh.x * vh.x + vh.y * vh.y;
    simd::float3 t1 = lensq > 0.0f
                          ? simd_make_float3(-vh.y, vh.x, 0.0f) / std::sqrt(lensq)
                          : simd_make_float3(1.0f, 0.0f, 0.0f);
    simd::float3 t2 = simd::cross(vh, t1);

    float u1 = rng.nextFloat();
    float u2 = rng.nextFloat();
    float r = std::sqrt(u1);
    float phi = 2.0f * kPi * u2;
    float t1r = r * std::cos(phi);
    float t2r = r * std::sin(phi);
    float s = 0.5f * (1.0f + vh.z);
    float t2Adjusted = (1.0f - s) * std::sqrt(std::max(0.0f, 1.0f - t1r * t1r)) + s * t2r;
    float t3 = std::sqrt(std::max(0.0f, 1.0f - t1r * t1r - t2Adjusted * t2Adjusted));

    simd::float3 nh = t1r * t1 + t2Adjusted * t2 + t3 * vh;
    simd::float3 ne = simd::normalize(simd_make_float3(alpha * nh.x,
                                                       alpha * nh.y,
                                                       std::max(nh.z, 0.0f)));
    return simd::normalize(ToWorld(ne, onb));
}

bool IsFinite(const simd::float3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

simd::float3 Clamp01(const simd::float3& value) {
    return simd_make_float3(std::clamp(value.x, 0.0f, 1.0f),
                            std::clamp(value.y, 0.0f, 1.0f),
                            std::clamp(value.z, 0.0f, 1.0f));
}

simd::float3 Exp3(const simd::float3& value) {
    return simd_make_float3(std::exp(value.x),
                            std::exp(value.y),
                            std::exp(value.z));
}

simd::float3 Fract3(const simd::float3& value) {
    return simd_make_float3(value.x - std::floor(value.x),
                            value.y - std::floor(value.y),
                            value.z - std::floor(value.z));
}

simd::float3 MaterialBaseColor(const PathTracerShaderTypes::MaterialData& material) {
    return Clamp01(simd_make_float3(material.baseColorRoughness.x,
                                    material.baseColorRoughness.y,
                                    material.baseColorRoughness.z));
}

float MaterialRoughness(const PathTracerShaderTypes::MaterialData& material) {
    float roughness = std::clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
    return std::max(roughness, 1.0e-3f);
}

float DielectricF0(float ior) {
    float eta = std::max(ior, 1.0f);
    float r = (eta - 1.0f) / (eta + 1.0f);
    return r * r;
}

float PlasticCoatIor(const PathTracerShaderTypes::MaterialData& material) {
    return std::max(material.typeEta.y, 1.0f);
}

float PlasticCoatRoughness(const PathTracerShaderTypes::MaterialData& material) {
    float roughness = std::clamp(material.coatParams.x, 0.0f, 1.0f);
    return std::max(roughness, 1.0e-3f);
}

float PlasticCoatThickness(const PathTracerShaderTypes::MaterialData& material) {
    return std::max(material.coatParams.y, 0.0f);
}

float PlasticCoatSampleWeight(const PathTracerShaderTypes::MaterialData& material) {
    return std::clamp(material.coatParams.z, 0.0f, 1.0f);
}

float PlasticCoatFresnelAverage(const PathTracerShaderTypes::MaterialData& material) {
    return std::clamp(material.coatParams.w, 0.0f, 1.0f);
}

simd::float3 PlasticCoatTint(const PathTracerShaderTypes::MaterialData& material) {
    return Clamp01(simd_make_float3(material.coatTint.x,
                                    material.coatTint.y,
                                    material.coatTint.z));
}

simd::float3 PlasticCoatAbsorption(const PathTracerShaderTypes::MaterialData& material) {
    return simd::max(simd_make_float3(material.coatAbsorption.x,
                                      material.coatAbsorption.y,
                                      material.coatAbsorption.z),
                     simd_make_float3(0.0f, 0.0f, 0.0f));
}

simd::float3 PlasticSpecularTint(const PathTracerShaderTypes::MaterialData& material) {
    simd::float3 tint = PlasticCoatTint(material);
    float thickness = PlasticCoatThickness(material);
    if (thickness <= 0.0f) {
        return tint;
    }
    simd::float3 absorption = PlasticCoatAbsorption(material);
    if (absorption.x <= 1.0e-6f && absorption.y <= 1.0e-6f && absorption.z <= 1.0e-6f) {
        return tint;
    }
    simd::float3 atten = Exp3(-absorption * thickness);
    return Clamp01(tint * atten);
}

simd::float3 PlasticDiffuseTransmission(const PathTracerShaderTypes::MaterialData& material,
                                        float cosThetaI,
                                        float cosThetaO) {
    simd::float3 tint = PlasticCoatTint(material);
    float thickness = PlasticCoatThickness(material);
    if (thickness <= 0.0f) {
        return tint;
    }
    simd::float3 absorption = PlasticCoatAbsorption(material);
    float safeCosI = std::max(cosThetaI, 1.0e-3f);
    float safeCosO = std::max(cosThetaO, 1.0e-3f);
    simd::float3 attenI = Exp3(-absorption * (thickness / safeCosI));
    simd::float3 attenO = Exp3(-absorption * (thickness / safeCosO));
    return Clamp01(tint * attenI * attenO);
}

float CarpaintBaseMetallic(const PathTracerShaderTypes::MaterialData& material) {
    return std::clamp(material.carpaintBaseParams.x, 0.0f, 1.0f);
}

float CarpaintBaseRoughness(const PathTracerShaderTypes::MaterialData& material) {
    return std::clamp(material.carpaintBaseParams.y, 0.0f, 1.0f);
}

float CarpaintFlakeScale(const PathTracerShaderTypes::MaterialData& material) {
    return std::max(material.carpaintBaseParams.z, 1.0e-4f);
}

float CarpaintFlakeSampleWeight(const PathTracerShaderTypes::MaterialData& material) {
    return std::clamp(material.carpaintFlakeParams.x, 0.0f, 0.95f);
}

float CarpaintFlakeRoughness(const PathTracerShaderTypes::MaterialData& material) {
    return std::clamp(material.carpaintFlakeParams.y, 0.0f, 1.0f);
}

float CarpaintFlakeAnisotropy(const PathTracerShaderTypes::MaterialData& material) {
    return std::clamp(material.carpaintFlakeParams.z, -0.99f, 0.99f);
}

float CarpaintFlakeNormalStrength(const PathTracerShaderTypes::MaterialData& material) {
    return std::clamp(material.carpaintFlakeParams.w, 0.0f, 1.0f);
}

float CarpaintCoatSampleWeight(const PathTracerShaderTypes::MaterialData& material) {
    return std::clamp(material.coatParams.z, 0.0f, 0.95f);
}

bool CarpaintHasBaseConductor(const PathTracerShaderTypes::MaterialData& material) {
    return material.carpaintBaseEta.w > 0.0f || material.carpaintBaseK.w > 0.0f;
}

simd::float3 CarpaintBaseEta(const PathTracerShaderTypes::MaterialData& material) {
    return simd::max(simd_make_float3(material.carpaintBaseEta.x,
                                      material.carpaintBaseEta.y,
                                      material.carpaintBaseEta.z),
                     simd_make_float3(0.0f, 0.0f, 0.0f));
}

simd::float3 CarpaintBaseK(const PathTracerShaderTypes::MaterialData& material) {
    return simd::max(simd_make_float3(material.carpaintBaseK.x,
                                      material.carpaintBaseK.y,
                                      material.carpaintBaseK.z),
                     simd_make_float3(0.0f, 0.0f, 0.0f));
}

simd::float3 CarpaintBaseF0(const PathTracerShaderTypes::MaterialData& material) {
    if (CarpaintHasBaseConductor(material)) {
        return FresnelConductor(1.0f, CarpaintBaseEta(material), CarpaintBaseK(material));
    }
    return MaterialBaseColor(material);
}

simd::float3 CarpaintHash3(const simd::float3& p) {
    simd::float3 value = Fract3(p * 0.3183099f + simd_make_float3(0.1f, 0.3f, 0.7f));
    float dotValue = simd::dot(value,
                               simd_make_float3(value.y + 33.33f,
                                                value.z + 55.55f,
                                                value.x + 77.77f));
    value += simd_make_float3(dotValue, dotValue, dotValue);
    simd::float3 mixed = simd_make_float3(value.x + value.y,
                                          value.x + value.z,
                                          value.y + value.z);
    return Fract3(mixed * 13.5453123f);
}

simd::float3 CarpaintFlakeNormal(const PathTracerShaderTypes::MaterialData& material,
                                 const simd::float3& position,
                                 const simd::float3& normal) {
    float scale = CarpaintFlakeScale(material);
    simd::float3 samplePos = position * scale;
    simd::float3 rand = CarpaintHash3(samplePos);
    float anis = CarpaintFlakeAnisotropy(material);
    float ax = std::max(1.0f - anis, 1.0e-3f);
    float ay = std::max(1.0f + anis, 1.0e-3f);
    float phi = 2.0f * kPi * rand.x;
    float r = std::sqrt(std::max(rand.y, 1.0e-4f));
    float x = r * std::cos(phi) * ax;
    float y = r * std::sin(phi) * ay;
    float m2 = std::clamp(x * x + y * y, 0.0f, 0.99f);
    float z = std::sqrt(std::max(1.0f - m2, 0.0f));
    Onb onb = BuildOnb(normal);
    simd::float3 perturbed = simd::normalize(x * onb.tangent + y * onb.bitangent + z * onb.normal);
    float strength = CarpaintFlakeNormalStrength(material);
    simd::float3 mixed = normal * (1.0f - strength) + perturbed * strength;
    return simd::normalize(mixed);
}

bool MaterialHasConductorIor(const PathTracerShaderTypes::MaterialData& material) {
    return material.conductorEta.w > 0.0f || material.conductorK.w > 0.0f ||
           material.conductorEta.x > 0.0f || material.conductorEta.y > 0.0f ||
           material.conductorEta.z > 0.0f || material.conductorK.x > 0.0f ||
           material.conductorK.y > 0.0f || material.conductorK.z > 0.0f;
}

simd::float3 ConductorF0(const PathTracerShaderTypes::MaterialData& material) {
    if (MaterialHasConductorIor(material)) {
        return FresnelConductor(1.0f,
                                simd_make_float3(material.conductorEta.x,
                                                 material.conductorEta.y,
                                                 material.conductorEta.z),
                                simd_make_float3(material.conductorK.x,
                                                 material.conductorK.y,
                                                 material.conductorK.z));
    }
    return MaterialBaseColor(material);
}

bool MaterialIsDelta(const PathTracerShaderTypes::MaterialData& material) {
    const uint32_t type = static_cast<uint32_t>(material.typeEta.x);
    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Dielectric)) {
        return true;
    }
    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Metal)) {
        return MaterialRoughness(material) <= 1.0e-3f;
    }
    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
        return false;
    }
    return false;
}

bool MaterialUsesGeometricNormal(const PathTracerShaderTypes::MaterialData& material) {
    const uint32_t type = static_cast<uint32_t>(material.typeEta.x);
    // Match Metal: only dielectrics force geometric normals; other materials use shading normals.
    return type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Dielectric);
}

float PbrSpecularWeight(const simd::float3& f0) {
    float maxComp = std::max(f0.x, std::max(f0.y, f0.z));
    return std::clamp(maxComp, 0.05f, 0.95f);
}

float LambertPdf(const simd::float3& normal, const simd::float3& direction) {
    simd::float3 unit = simd::normalize(direction);
    float cosTheta = std::max(simd::dot(normal, unit), 0.0f);
    return cosTheta > 0.0f ? (cosTheta / kPi) : 0.0f;
}

float PowerHeuristic(float pdfA, float pdfB) {
    float pdfA2 = pdfA * pdfA;
    float pdfB2 = pdfB * pdfB;
    float denom = pdfA2 + pdfB2;
    return denom > 0.0f ? (pdfA2 / denom) : 0.0f;
}

float EnvironmentPdf(const EnvironmentMap& env, float rotation, const simd::float3& direction) {
    if (!env.hasDistribution || env.width == 0 || env.height == 0 ||
        env.distribution.texelPdf.empty()) {
        return 0.0f;
    }

    simd::float3 unit = simd::normalize(direction);
    float cosTheta = std::cos(rotation);
    float sinTheta = std::sin(rotation);
    simd::float3 rotated = {unit.x * cosTheta - unit.z * sinTheta,
                            unit.y,
                            unit.x * sinTheta + unit.z * cosTheta};
    float u = (std::atan2(rotated.z, rotated.x) + kPi) / (2.0f * kPi);
    float v = 0.5f - std::asin(std::clamp(rotated.y, -1.0f, 1.0f)) / kPi;

    u = std::clamp(u, 0.0f, 0.99999994f);
    v = std::clamp(v, 0.0f, 0.99999994f);

    uint32_t width = std::max(env.distribution.width, 1u);
    uint32_t height = std::max(env.distribution.height, 1u);
    uint32_t x = std::min(static_cast<uint32_t>(u * static_cast<float>(width)), width - 1u);
    uint32_t y = std::min(static_cast<uint32_t>(v * static_cast<float>(height)), height - 1u);
    size_t index = static_cast<size_t>(y) * width + static_cast<size_t>(x);
    if (index >= env.distribution.texelPdf.size()) {
        return 0.0f;
    }
    float value = env.distribution.texelPdf[index];
    return (std::isfinite(value) && value > 0.0f) ? value : 0.0f;
}

simd::float3 OffsetRayOrigin(const HitInfo& hit, const simd::float3& direction) {
    simd::float3 normal = hit.shadingNormal;
    if (simd::dot(normal, normal) <= 0.0f) {
        normal = hit.normal;
    }
    if (simd::dot(normal, normal) <= 0.0f) {
        normal = simd_make_float3(0.0f, 1.0f, 0.0f);
    }
    normal = simd::normalize(normal);
    float sign = simd::dot(direction, normal) >= 0.0f ? 1.0f : -1.0f;
    float distance = std::max(std::fabs(hit.t) * 1.0e-4f, kEpsilon);
    simd::float3 origin = hit.position + normal * (sign * distance);
    origin += direction * kEpsilon * 0.5f;
    return origin;
}

struct RectLightSample {
    simd::float3 direction{0.0f, 0.0f, 0.0f};
    float distance = 0.0f;
    float pdf = 0.0f;
    simd::float3 emission{0.0f, 0.0f, 0.0f};
    uint32_t rectIndex = 0;
    uint32_t lightIndex = 0;
    float area = 0.0f;
    float cosLight = 0.0f;
};

struct RectLightHit {
    simd::float3 emission{0.0f, 0.0f, 0.0f};
    float pdf = 0.0f;
};

struct CarpaintLobeResult {
    simd::float3 value{0.0f, 0.0f, 0.0f};
    float pdf = 0.0f;
};

struct DirectDebugConfig {
    bool enabled = false;
    uint32_t pixelX = 0;
    uint32_t pixelY = 0;
};

DirectDebugConfig LoadDirectDebugConfig(uint32_t width, uint32_t height) {
    DirectDebugConfig cfg;
    const char* enabled = std::getenv("PATH_TRACER_EMBREE_DEBUG_DIRECT");
    if (!enabled || std::atoi(enabled) == 0) {
        return cfg;
    }

    cfg.enabled = true;
    cfg.pixelX = width / 2u;
    cfg.pixelY = height / 2u;

    if (const char* px = std::getenv("PATH_TRACER_EMBREE_DEBUG_X")) {
        int value = std::atoi(px);
        if (value >= 0) {
            cfg.pixelX = std::min(static_cast<uint32_t>(value), width > 0 ? width - 1u : 0u);
        }
    }
    if (const char* py = std::getenv("PATH_TRACER_EMBREE_DEBUG_Y")) {
        int value = std::atoi(py);
        if (value >= 0) {
            cfg.pixelY = std::min(static_cast<uint32_t>(value), height > 0 ? height - 1u : 0u);
        }
    }

    return cfg;
}

bool SampleRectLight(const std::vector<RectLightInfo>& rectLights,
                     const EnvironmentMap* env,
                     const PathTracer::RenderSettings& settings,
                     const HitInfo& hit,
                     Rng& rng,
                     RectLightSample& outSample) {
    outSample = RectLightSample{};
    if (rectLights.empty()) {
        return false;
    }

    uint32_t selected = std::min(static_cast<uint32_t>(rng.nextFloat() * rectLights.size()),
                                 static_cast<uint32_t>(rectLights.size() - 1u));
    const RectLightInfo& light = rectLights[selected];
    outSample.lightIndex = selected;

    float u = rng.nextFloat();
    float v = rng.nextFloat();
    simd::float3 samplePoint = light.corner + u * light.edgeU + v * light.edgeV;
    simd::float3 toLight = samplePoint - hit.position;
    float distSq = simd::dot(toLight, toLight);
    if (distSq <= 0.0f) {
        return false;
    }

    float distance = std::sqrt(distSq);
    simd::float3 direction = toLight / distance;
    if (light.area <= 0.0f) {
        return false;
    }

    float cosLight = simd::dot(-direction, light.normal);
    if (light.twoSided) {
        cosLight = std::fabs(cosLight);
    } else if (cosLight <= 0.0f) {
        return false;
    }
    if (cosLight <= 0.0f) {
        return false;
    }

    float pdfArea = 1.0f / light.area;
    float pdfDir = pdfArea * distSq / std::max(cosLight, 1.0e-6f);
    float selectionPdf = 1.0f / static_cast<float>(rectLights.size());
    float pdf = pdfDir * selectionPdf;
    if (!(pdf > 0.0f) || !std::isfinite(pdf)) {
        return false;
    }

    simd::float3 emission = light.baseEmission;
    if (light.emissionUsesEnv && env && env->width > 0 && env->height > 0 && !env->rgba.empty()) {
        simd::float3 sampleDir = -light.normal;
        emission *= SampleEnvironment(*env, sampleDir,
                                      settings.environmentRotation,
                                      settings.environmentIntensity);
    }
    if (!(simd::dot(emission, emission) > 0.0f)) {
        return false;
    }

    outSample.direction = direction;
    outSample.distance = distance;
    outSample.pdf = pdf;
    outSample.emission = emission;
    outSample.rectIndex = light.rectIndex;
    outSample.area = light.area;
    outSample.cosLight = cosLight;
    return true;
}

float RectLightPdfForHit(const std::vector<int32_t>& rectLightIndexByRect,
                         const PathTracerShaderTypes::RectData* rectangles,
                         uint32_t rectangleCount,
                         uint32_t lightCount,
                         const HitInfo& lightHit,
                         const simd::float3& origin) {
    if (lightCount == 0 || !rectangles || rectangleCount == 0) {
        return 0.0f;
    }
    if (lightHit.primitiveType != GeometryType::Rectangles) {
        return 0.0f;
    }
    uint32_t rectIndex = lightHit.primitiveIndex;
    if (rectIndex >= rectangleCount) {
        return 0.0f;
    }
    if (rectLightIndexByRect.empty() ||
        rectIndex >= rectLightIndexByRect.size() ||
        rectLightIndexByRect[rectIndex] < 0) {
        return 0.0f;
    }

    const auto& rect = rectangles[rectIndex];
    simd::float3 edgeU = simd_make_float3(rect.edgeU.x, rect.edgeU.y, rect.edgeU.z);
    simd::float3 edgeV = simd_make_float3(rect.edgeV.x, rect.edgeV.y, rect.edgeV.z);
    float area = simd::length(simd::cross(edgeU, edgeV));
    if (area <= 0.0f) {
        return 0.0f;
    }

    simd::float3 toLight = lightHit.position - origin;
    float distSq = simd::dot(toLight, toLight);
    if (distSq <= 0.0f) {
        return 0.0f;
    }
    float distance = std::sqrt(distSq);
    simd::float3 direction = toLight / distance;
    simd::float3 normal = simd_make_float3(rect.normalAndPlane.x,
                                           rect.normalAndPlane.y,
                                           rect.normalAndPlane.z);
    float cosLight = simd::dot(-direction, normal);
    if (rect.materialTwoSided.y != 0u) {
        cosLight = std::fabs(cosLight);
    } else if (cosLight <= 0.0f) {
        return 0.0f;
    }
    if (cosLight <= 0.0f) {
        return 0.0f;
    }

    float pdfArea = 1.0f / area;
    float pdfDir = pdfArea * distSq / std::max(cosLight, 1.0e-6f);
    float selectionPdf = 1.0f / static_cast<float>(lightCount);
    return pdfDir * selectionPdf;
}

bool RectLightHitInfo(const std::vector<int32_t>& rectLightIndexByRect,
                      const std::vector<RectLightInfo>& rectLights,
                      const PathTracerShaderTypes::RectData* rectangles,
                      uint32_t rectangleCount,
                      const EnvironmentMap* env,
                      const PathTracer::RenderSettings& settings,
                      const HitInfo& hit,
                      const simd::float3& origin,
                      RectLightHit& outHit) {
    outHit = RectLightHit{};
    if (rectLights.empty() || !rectangles || rectangleCount == 0) {
        return false;
    }
    if (hit.primitiveType != GeometryType::Rectangles) {
        return false;
    }
    uint32_t rectIndex = hit.primitiveIndex;
    if (rectIndex >= rectangleCount ||
        rectIndex >= rectLightIndexByRect.size()) {
        return false;
    }
    int32_t lightIndex = rectLightIndexByRect[rectIndex];
    if (lightIndex < 0) {
        return false;
    }
    const RectLightInfo& light = rectLights[static_cast<size_t>(lightIndex)];
    if (!hit.frontFace && !light.twoSided) {
        return false;
    }

    simd::float3 emission = light.baseEmission;
    if (light.emissionUsesEnv && env && env->width > 0 && env->height > 0 && !env->rgba.empty()) {
        simd::float3 sampleDir = -hit.shadingNormal;
        emission *= SampleEnvironment(*env,
                                      sampleDir,
                                      settings.environmentRotation,
                                      settings.environmentIntensity);
    }
    if (!(simd::dot(emission, emission) > 0.0f)) {
        return false;
    }

    float pdf = RectLightPdfForHit(rectLightIndexByRect,
                                   rectangles,
                                   rectangleCount,
                                   static_cast<uint32_t>(rectLights.size()),
                                   hit,
                                   origin);
    if (!(pdf > 0.0f) || !std::isfinite(pdf)) {
        return false;
    }

    outHit.emission = emission;
    outHit.pdf = pdf;
    return true;
}

CarpaintLobeResult CarpaintEvalCoat(const PathTracerShaderTypes::MaterialData& material,
                                    const simd::float3& normal,
                                    const simd::float3& wo,
                                    const simd::float3& wi,
                                    const FireflyClampParams& clampParams) {
    CarpaintLobeResult result;
    float cosThetaO = std::max(simd::dot(normal, wo), 0.0f);
    float cosThetaI = std::max(simd::dot(normal, wi), 0.0f);
    if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
        return result;
    }
    float roughness = PlasticCoatRoughness(material);
    float alpha = roughness * roughness;
    simd::float3 wh = simd::normalize(wo + wi);
    if (simd::dot(wh, normal) <= 0.0f || simd::dot(wo, wh) <= 0.0f || simd::dot(wi, wh) <= 0.0f) {
        return result;
    }

    float D = GgxDistribution(alpha, simd::dot(normal, wh));
    float G = GgxG1(alpha, cosThetaO) * GgxG1(alpha, cosThetaI);
    float f0 = DielectricF0(PlasticCoatIor(material));
    simd::float3 f0Color = simd_make_float3(f0, f0, f0);
    simd::float3 F = SchlickFresnel(f0Color, simd::dot(wi, wh));
    float denom = 4.0f * cosThetaO * cosThetaI;
    simd::float3 spec = F * (D * G / std::max(denom, 1.0e-6f));
    spec = ClampSpecularTail(spec, roughness, f0Color, clampParams);
    spec *= PlasticSpecularTint(material);
    spec = simd::max(spec, simd_make_float3(0.0f, 0.0f, 0.0f));

    float pdf = GgxPdf(alpha, normal, wo, wi);
    if (pdf > 0.0f) {
        result.pdf = ClampSpecularPdf(pdf, clampParams);
        result.value = spec;
    }
    return result;
}

CarpaintLobeResult CarpaintEvalFlake(const PathTracerShaderTypes::MaterialData& material,
                                     const simd::float3& position,
                                     const simd::float3& normal,
                                     const simd::float3& wo,
                                     const simd::float3& wi,
                                     const FireflyClampParams& clampParams) {
    CarpaintLobeResult result;
    simd::float3 flakeNormal = CarpaintFlakeNormal(material, position, normal);
    float cosThetaO = std::max(simd::dot(flakeNormal, wo), 0.0f);
    float cosThetaI = std::max(simd::dot(flakeNormal, wi), 0.0f);
    if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
        return result;
    }
    float roughness = std::max(CarpaintFlakeRoughness(material), 1.0e-3f);
    float alpha = roughness * roughness;
    simd::float3 wh = simd::normalize(wo + wi);
    if (simd::dot(wh, flakeNormal) <= 0.0f ||
        simd::dot(wo, wh) <= 0.0f ||
        simd::dot(wi, wh) <= 0.0f) {
        return result;
    }

    float D = GgxDistribution(alpha, simd::dot(flakeNormal, wh));
    float G = GgxG1(alpha, cosThetaO) * GgxG1(alpha, cosThetaI);
    simd::float3 f0 = CarpaintBaseF0(material);
    simd::float3 F = SchlickFresnel(f0, simd::dot(wi, wh));
    float denom = 4.0f * cosThetaO * cosThetaI;
    simd::float3 spec = F * (D * G / std::max(denom, 1.0e-6f));
    spec = ClampSpecularTail(spec * PlasticSpecularTint(material), roughness, f0, clampParams);
    spec *= std::max(1.0f - PlasticCoatFresnelAverage(material), 0.0f);
    spec = simd::max(spec, simd_make_float3(0.0f, 0.0f, 0.0f));

    float pdf = GgxPdf(alpha, flakeNormal, wo, wi);
    if (pdf > 0.0f) {
        result.pdf = ClampSpecularPdf(pdf, clampParams);
        result.value = spec;
    }
    return result;
}

CarpaintLobeResult CarpaintEvalBase(const PathTracerShaderTypes::MaterialData& material,
                                    const simd::float3& normal,
                                    const simd::float3& wo,
                                    const simd::float3& wi,
                                    const FireflyClampParams& clampParams) {
    CarpaintLobeResult result;
    float cosThetaO = std::max(simd::dot(normal, wo), 0.0f);
    float cosThetaI = std::max(simd::dot(normal, wi), 0.0f);
    if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
        return result;
    }
    float metallic = CarpaintBaseMetallic(material);
    float diffuseWeight = std::max(1.0f - metallic, 0.0f);
    float specWeight = std::max(metallic, 0.0f);
    if (diffuseWeight <= 1.0e-4f && specWeight <= 1.0e-4f) {
        return result;
    }

    float coatAverage = PlasticCoatFresnelAverage(material);
    simd::float3 baseColor = MaterialBaseColor(material);
    simd::float3 combined = simd_make_float3(0.0f, 0.0f, 0.0f);
    float pdfDiffuse = 0.0f;
    float pdfSpec = 0.0f;

    if (diffuseWeight > 1.0e-4f) {
        simd::float3 diffuse = baseColor / kPi;
        simd::float3 coatTrans = PlasticDiffuseTransmission(material, cosThetaI, cosThetaO);
        diffuse *= coatTrans * std::max(1.0f - coatAverage, 0.0f);
        diffuse = simd::max(diffuse, simd_make_float3(0.0f, 0.0f, 0.0f));
        combined += diffuseWeight * diffuse;
        pdfDiffuse = LambertPdf(normal, wi);
    }

    if (specWeight > 1.0e-4f) {
        float roughness = std::max(CarpaintBaseRoughness(material), 1.0e-3f);
        float alpha = roughness * roughness;
        simd::float3 wh = simd::normalize(wo + wi);
        if (simd::dot(wh, normal) > 0.0f &&
            simd::dot(wo, wh) > 0.0f &&
            simd::dot(wi, wh) > 0.0f) {
            float D = GgxDistribution(alpha, simd::dot(normal, wh));
            float G = GgxG1(alpha, cosThetaO) * GgxG1(alpha, cosThetaI);
            bool hasConductor = CarpaintHasBaseConductor(material);
            simd::float3 eta = CarpaintBaseEta(material);
            simd::float3 k = CarpaintBaseK(material);
            simd::float3 f0 = hasConductor ? FresnelConductor(1.0f, eta, k) : baseColor;
            simd::float3 F = hasConductor ? FresnelConductor(simd::dot(wi, wh), eta, k)
                                          : SchlickFresnel(baseColor, simd::dot(wi, wh));
            float denom = 4.0f * cosThetaO * cosThetaI;
            simd::float3 spec = F * (D * G / std::max(denom, 1.0e-6f));
            spec = ClampSpecularTail(spec * PlasticSpecularTint(material) * std::max(1.0f - coatAverage, 0.0f),
                                     roughness,
                                     f0,
                                     clampParams);
            spec = simd::max(spec, simd_make_float3(0.0f, 0.0f, 0.0f));
            combined += specWeight * spec;
            float pdf = GgxPdf(alpha, normal, wo, wi);
            if (pdf > 0.0f) {
                pdfSpec = ClampSpecularPdf(pdf, clampParams);
            }
        }
    }

    result.value = simd::max(combined, simd_make_float3(0.0f, 0.0f, 0.0f));
    result.pdf = diffuseWeight * pdfDiffuse + specWeight * pdfSpec;
    return result;
}

BsdfEval EvaluateBsdf(const PathTracerShaderTypes::MaterialData& material,
                      const simd::float3& position,
                      const simd::float3& normal,
                      const simd::float3& wo,
                      const simd::float3& wi,
                      const FireflyClampParams& clampParams) {
    BsdfEval result;
    float cosThetaO = std::max(simd::dot(normal, wo), 0.0f);
    float cosThetaI = std::max(simd::dot(normal, wi), 0.0f);
    if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
        return result;
    }

    const uint32_t type = static_cast<uint32_t>(material.typeEta.x);
    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Lambertian) ||
        type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Subsurface)) {
        simd::float3 albedo = MaterialBaseColor(material);
        result.value = albedo / kPi;
        result.pdf = LambertPdf(normal, wi);
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
        simd::float3 baseColor = MaterialBaseColor(material);
        float metallic = std::clamp(material.pbrParams.x, 0.0f, 1.0f);
        float roughness = std::clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        float dielectricF0 = DielectricF0FromIor(material.typeEta.y);
        simd::float3 f0 = baseColor * metallic +
                          simd_make_float3(dielectricF0, dielectricF0, dielectricF0) * (1.0f - metallic);
        simd::float3 diffuseColor = baseColor * (1.0f - metallic);

        float alpha = std::max(roughness * roughness, 1.0e-4f);
        simd::float3 wh = simd::normalize(wo + wi);
        if (simd::dot(wh, normal) <= 0.0f ||
            simd::dot(wo, wh) <= 0.0f ||
            simd::dot(wi, wh) <= 0.0f) {
            return result;
        }
        float D = GgxDistribution(alpha, simd::dot(normal, wh));
        float G = GgxG1(alpha, cosThetaO) * GgxG1(alpha, cosThetaI);
        simd::float3 F = SchlickFresnel(f0, simd::dot(wi, wh));
        float denom = 4.0f * cosThetaO * cosThetaI;
        simd::float3 spec = F * (D * G / std::max(denom, 1.0e-6f));
        spec = ClampSpecularTail(spec, roughness, f0, clampParams);
        spec = simd::max(spec, simd_make_float3(0.0f, 0.0f, 0.0f));

        simd::float3 diffuse = diffuseColor / kPi;
        float pdfDiffuse = LambertPdf(normal, wi);
        float pdfSpec = GgxPdf(alpha, normal, wo, wi);
        float specPdfClamped = (pdfSpec > 0.0f) ? ClampSpecularPdf(pdfSpec, clampParams) : 0.0f;
        float specWeight = PbrSpecularWeight(f0);
        float pdf = specWeight * specPdfClamped + (1.0f - specWeight) * pdfDiffuse;
        if (pdf > 0.0f) {
            result.value = simd::max(spec + diffuse, simd_make_float3(0.0f, 0.0f, 0.0f));
            result.pdf = pdf;
        }
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Plastic)) {
        simd::float3 albedo = MaterialBaseColor(material);
        float coatRoughness = PlasticCoatRoughness(material);
        float alpha = coatRoughness * coatRoughness;
        float ior = PlasticCoatIor(material);
        float f0 = DielectricF0(ior);
        simd::float3 f0Color = simd_make_float3(f0, f0, f0);
        simd::float3 specularTint = PlasticSpecularTint(material);

        simd::float3 specular{0.0f, 0.0f, 0.0f};
        float specularPdf = 0.0f;
        simd::float3 wh = simd::normalize(wo + wi);
        if (simd::dot(wh, normal) > 0.0f &&
            simd::dot(wo, wh) > 0.0f &&
            simd::dot(wi, wh) > 0.0f) {
            float D = GgxDistribution(alpha, simd::dot(normal, wh));
            float G = GgxG1(alpha, cosThetaO) * GgxG1(alpha, cosThetaI);
            simd::float3 F = SchlickFresnel(f0Color, simd::dot(wi, wh));
            float denom = 4.0f * cosThetaO * cosThetaI;
            specular = F * (D * G / std::max(denom, 1.0e-6f));
            specular = ClampSpecularTail(specular, coatRoughness, f0Color, clampParams);
            specular *= specularTint;
            float pdf = GgxPdf(alpha, normal, wo, wi);
            if (pdf > 0.0f) {
                specularPdf = ClampSpecularPdf(pdf, clampParams);
            }
            specular = simd::max(specular, simd_make_float3(0.0f, 0.0f, 0.0f));
        }

        simd::float3 diffuse = albedo / kPi;
        simd::float3 tint = PlasticDiffuseTransmission(material, cosThetaI, cosThetaO);
        simd::float3 F_i = SchlickFresnel(f0Color, cosThetaI);
        simd::float3 F_o = SchlickFresnel(f0Color, cosThetaO);
        diffuse *= tint;
        diffuse *= (simd_make_float3(1.0f, 1.0f, 1.0f) - F_i) *
                   (simd_make_float3(1.0f, 1.0f, 1.0f) - F_o);
        diffuse *= std::max(1.0f - PlasticCoatFresnelAverage(material), 0.0f);
        diffuse = simd::max(diffuse, simd_make_float3(0.0f, 0.0f, 0.0f));

        float pdfDiffuse = LambertPdf(normal, wi);
        float pCoat = PlasticCoatSampleWeight(material);
        float pdf = pCoat * specularPdf + (1.0f - pCoat) * pdfDiffuse;
        if (pdf > 0.0f) {
            result.value = specular + diffuse;
            result.pdf = pdf;
        }
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Metal)) {
        float roughness = std::clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        if (roughness <= 1.0e-3f) {
            result.isDelta = true;
            return result;
        }
        float alpha = roughness * roughness;
        simd::float3 wh = simd::normalize(wo + wi);
        if (simd::dot(wh, normal) <= 0.0f || simd::dot(wo, wh) <= 0.0f || simd::dot(wi, wh) <= 0.0f) {
            return result;
        }

        float D = GgxDistribution(alpha, simd::dot(normal, wh));
        float G = GgxG1(alpha, cosThetaO) * GgxG1(alpha, cosThetaI);
        simd::float3 f0 = ConductorF0(material);
        simd::float3 F = MaterialHasConductorIor(material)
                             ? FresnelConductor(simd::dot(wi, wh),
                                                simd_make_float3(material.conductorEta.x,
                                                                 material.conductorEta.y,
                                                                 material.conductorEta.z),
                                                simd_make_float3(material.conductorK.x,
                                                                 material.conductorK.y,
                                                                 material.conductorK.z))
                             : SchlickFresnel(f0, simd::dot(wi, wh));
        float denom = 4.0f * cosThetaO * cosThetaI;
        simd::float3 spec = F * (D * G / std::max(denom, 1.0e-6f));
        spec = ClampSpecularTail(spec, roughness, f0, clampParams);
        float pdf = GgxPdf(alpha, normal, wo, wi);
        if (pdf > 0.0f) {
            result.value = simd::max(spec, simd_make_float3(0.0f, 0.0f, 0.0f));
            result.pdf = ClampSpecularPdf(pdf, clampParams);
        }
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::CarPaint)) {
        float pCoat = CarpaintCoatSampleWeight(material);
        float pFlake = CarpaintFlakeSampleWeight(material);
        float pBase = std::max(1.0f - (pCoat + pFlake), 0.0f);
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

        CarpaintLobeResult coatRes = CarpaintEvalCoat(material, normal, wo, wi, clampParams);
        CarpaintLobeResult flakeRes = CarpaintEvalFlake(material, position, normal, wo, wi, clampParams);
        CarpaintLobeResult baseRes = CarpaintEvalBase(material, normal, wo, wi, clampParams);

        result.value = pBase * baseRes.value + pFlake * flakeRes.value + pCoat * coatRes.value;
        result.pdf = pBase * baseRes.pdf + pFlake * flakeRes.pdf + pCoat * coatRes.pdf;
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Dielectric)) {
        result.isDelta = true;
        return result;
    }

    simd::float3 albedo = MaterialBaseColor(material);
    result.value = albedo / kPi;
    result.pdf = LambertPdf(normal, wi);
    return result;
}

BsdfSample SampleBsdf(const PathTracerShaderTypes::MaterialData& material,
                      const simd::float3& position,
                      const simd::float3& normal,
                      const simd::float3& wo,
                      const simd::float3& incidentDir,
                      bool frontFace,
                      Rng& rng,
                      const FireflyClampParams& clampParams) {
    BsdfSample result;
    const uint32_t type = static_cast<uint32_t>(material.typeEta.x);

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Lambertian) ||
        type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Subsurface)) {
        float pdf = 0.0f;
        simd::float3 wi = SampleCosineHemisphere(rng, normal, pdf);
        float cosThetaI = simd::dot(normal, wi);
        if (pdf <= 0.0f || cosThetaI <= 0.0f) {
            return result;
        }
        simd::float3 albedo = MaterialBaseColor(material);
        simd::float3 f = albedo / kPi;
        simd::float3 weight = f * cosThetaI / pdf;
        if (!IsFinite(weight)) {
            return result;
        }
        result.direction = wi;
        result.weight = simd::max(weight, simd_make_float3(0.0f, 0.0f, 0.0f));
        result.pdf = pdf;
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
        simd::float3 baseColor = MaterialBaseColor(material);
        float metallic = std::clamp(material.pbrParams.x, 0.0f, 1.0f);
        float roughness = std::clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        float dielectricF0 = DielectricF0FromIor(material.typeEta.y);
        simd::float3 f0 = baseColor * metallic +
                          simd_make_float3(dielectricF0, dielectricF0, dielectricF0) * (1.0f - metallic);
        simd::float3 diffuseColor = baseColor * (1.0f - metallic);
        float specWeight = PbrSpecularWeight(f0);
        float diffuseWeight = 1.0f - specWeight;

        simd::float3 wi;
        float pdfSpec = 0.0f;
        float pdfDiffuse = 0.0f;
        simd::float3 f = simd_make_float3(0.0f, 0.0f, 0.0f);

        if (rng.nextFloat() < specWeight) {
            if (roughness <= 1.0e-3f) {
                wi = simd::normalize(Reflect(incidentDir, normal));
                float cosThetaI = simd::dot(normal, wi);
                if (cosThetaI <= 0.0f) {
                    return result;
                }
                float cosThetaO = std::max(simd::dot(normal, wo), 0.0f);
                f = SchlickFresnel(f0, cosThetaO);
                pdfSpec = 1.0f;
                result.isDelta = true;
            } else {
                float alpha = roughness * roughness;
                simd::float3 wh = SampleGgxHalfVector(rng, alpha, normal);
                if (simd::dot(wh, normal) <= 0.0f) {
                    return result;
                }
                wi = simd::normalize(Reflect(-wo, wh));
                float cosThetaI = std::max(simd::dot(normal, wi), 0.0f);
                float cosThetaO = std::max(simd::dot(normal, wo), 0.0f);
                if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
                    return result;
                }
                float D = GgxDistribution(alpha, simd::dot(normal, wh));
                float G = GgxG1(alpha, cosThetaO) * GgxG1(alpha, cosThetaI);
                simd::float3 F = SchlickFresnel(f0, simd::dot(wi, wh));
                float denom = 4.0f * cosThetaO * cosThetaI;
                f = F * (D * G / std::max(denom, 1.0e-6f));
                f = ClampSpecularTail(f, roughness, f0, clampParams);
                pdfSpec = GgxPdf(alpha, normal, wo, wi);
            }
        } else {
            float pdf = 0.0f;
            wi = SampleCosineHemisphere(rng, normal, pdf);
            pdfDiffuse = pdf;
            float cosThetaI = std::max(simd::dot(normal, wi), 0.0f);
            if (pdfDiffuse <= 0.0f || cosThetaI <= 0.0f) {
                return result;
            }
            f = diffuseColor / kPi;
        }

        float cosThetaI = std::max(simd::dot(normal, wi), 0.0f);
        float specPdfClamped = (pdfSpec > 0.0f) ? ClampSpecularPdf(pdfSpec, clampParams) : 0.0f;
        float pdf = specWeight * specPdfClamped + diffuseWeight * pdfDiffuse;
        if (pdf <= 0.0f || cosThetaI <= 0.0f) {
            return result;
        }
        simd::float3 weight = f * cosThetaI / pdf;
        if (!IsFinite(weight)) {
            return result;
        }
        result.direction = wi;
        result.weight = simd::max(weight, simd_make_float3(0.0f, 0.0f, 0.0f));
        result.pdf = pdf;
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Plastic)) {
        simd::float3 albedo = MaterialBaseColor(material);
        float coatRoughness = PlasticCoatRoughness(material);
        float alpha = coatRoughness * coatRoughness;
        float ior = PlasticCoatIor(material);
        float f0 = DielectricF0(ior);
        simd::float3 f0Color = simd_make_float3(f0, f0, f0);
        simd::float3 specularTint = PlasticSpecularTint(material);
        float pCoat = PlasticCoatSampleWeight(material);

        simd::float3 wi;
        if (rng.nextFloat() < pCoat) {
            simd::float3 wh = SampleGgxHalfVector(rng, alpha, normal);
            if (simd::dot(wh, normal) <= 0.0f) {
                return result;
            }
            wi = simd::normalize(Reflect(-wo, wh));
        } else {
            float pdf = 0.0f;
            wi = SampleCosineHemisphere(rng, normal, pdf);
        }

        float cosThetaI = std::max(simd::dot(normal, wi), 0.0f);
        float cosThetaO = std::max(simd::dot(normal, wo), 0.0f);
        if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
            return result;
        }

        simd::float3 specular{0.0f, 0.0f, 0.0f};
        float specularPdf = 0.0f;
        simd::float3 wh = simd::normalize(wo + wi);
        if (simd::dot(wh, normal) > 0.0f &&
            simd::dot(wo, wh) > 0.0f &&
            simd::dot(wi, wh) > 0.0f) {
            float D = GgxDistribution(alpha, simd::dot(normal, wh));
            float G = GgxG1(alpha, cosThetaO) * GgxG1(alpha, cosThetaI);
            simd::float3 F = SchlickFresnel(f0Color, simd::dot(wi, wh));
            float denom = 4.0f * cosThetaO * cosThetaI;
            specular = F * (D * G / std::max(denom, 1.0e-6f));
            specular = ClampSpecularTail(specular, coatRoughness, f0Color, clampParams);
            specular *= specularTint;
            float pdf = GgxPdf(alpha, normal, wo, wi);
            if (pdf > 0.0f) {
                specularPdf = ClampSpecularPdf(pdf, clampParams);
            }
            specular = simd::max(specular, simd_make_float3(0.0f, 0.0f, 0.0f));
        }

        simd::float3 diffuse = albedo / kPi;
        simd::float3 tint = PlasticDiffuseTransmission(material, cosThetaI, cosThetaO);
        simd::float3 F_i = SchlickFresnel(f0Color, cosThetaI);
        simd::float3 F_o = SchlickFresnel(f0Color, cosThetaO);
        diffuse *= tint;
        diffuse *= (simd_make_float3(1.0f, 1.0f, 1.0f) - F_i) *
                   (simd_make_float3(1.0f, 1.0f, 1.0f) - F_o);
        diffuse *= std::max(1.0f - PlasticCoatFresnelAverage(material), 0.0f);
        diffuse = simd::max(diffuse, simd_make_float3(0.0f, 0.0f, 0.0f));

        float pdfDiffuse = LambertPdf(normal, wi);
        float pdf = pCoat * specularPdf + (1.0f - pCoat) * pdfDiffuse;
        if (pdf <= 0.0f) {
            return result;
        }

        simd::float3 f = specular + diffuse;
        simd::float3 weight = f * cosThetaI / pdf;
        if (!IsFinite(weight)) {
            return result;
        }

        result.direction = wi;
        result.weight = simd::max(weight, simd_make_float3(0.0f, 0.0f, 0.0f));
        result.pdf = pdf;
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Metal)) {
        float roughness = std::clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        simd::float3 f0 = ConductorF0(material);
        if (roughness <= 1.0e-3f) {
            simd::float3 wi = simd::normalize(Reflect(incidentDir, normal));
            float cosThetaI = simd::dot(normal, wi);
            if (cosThetaI <= 0.0f) {
                return result;
            }
            float cosTheta = std::max(simd::dot(normal, wo), 0.0f);
            simd::float3 F = MaterialHasConductorIor(material)
                                 ? FresnelConductor(cosTheta,
                                                    simd_make_float3(material.conductorEta.x,
                                                                     material.conductorEta.y,
                                                                     material.conductorEta.z),
                                                    simd_make_float3(material.conductorK.x,
                                                                     material.conductorK.y,
                                                                     material.conductorK.z))
                                 : SchlickFresnel(f0, cosTheta);
            result.direction = wi;
            result.weight = F;
            result.pdf = 1.0f;
            result.isDelta = true;
            return result;
        }

        float alpha = roughness * roughness;
        simd::float3 wh = SampleGgxHalfVector(rng, alpha, normal);
        if (simd::dot(wh, normal) <= 0.0f) {
            return result;
        }
        simd::float3 wi = simd::normalize(Reflect(-wo, wh));
        float cosThetaI = simd::dot(normal, wi);
        float cosThetaO = simd::dot(normal, wo);
        if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
            return result;
        }
        float dotWoWh = simd::dot(wo, wh);
        if (dotWoWh <= 0.0f) {
            return result;
        }

        simd::float3 F = MaterialHasConductorIor(material)
                             ? FresnelConductor(simd::dot(wi, wh),
                                                simd_make_float3(material.conductorEta.x,
                                                                 material.conductorEta.y,
                                                                 material.conductorEta.z),
                                                simd_make_float3(material.conductorK.x,
                                                                 material.conductorK.y,
                                                                 material.conductorK.z))
                             : SchlickFresnel(f0, simd::dot(wi, wh));
        float D = GgxDistribution(alpha, simd::dot(normal, wh));
        float G = GgxG1(alpha, cosThetaO) * GgxG1(alpha, cosThetaI);
        float denom = 4.0f * cosThetaO * cosThetaI;
        simd::float3 f = F * (D * G / std::max(denom, 1.0e-6f));
        f = ClampSpecularTail(f, roughness, f0, clampParams);
        float pdf = D * std::max(simd::dot(normal, wh), 0.0f) /
                    std::max(4.0f * dotWoWh, 1.0e-6f);
        if (pdf <= 0.0f) {
            return result;
        }
        float clampedPdf = ClampSpecularPdf(pdf, clampParams);
        simd::float3 weight = f * cosThetaI / clampedPdf;
        if (!IsFinite(weight)) {
            return result;
        }
        result.direction = wi;
        result.weight = simd::max(weight, simd_make_float3(0.0f, 0.0f, 0.0f));
        result.pdf = clampedPdf;
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::CarPaint)) {
        float pCoat = CarpaintCoatSampleWeight(material);
        float pFlake = CarpaintFlakeSampleWeight(material);
        float pBase = std::max(1.0f - (pCoat + pFlake), 0.0f);
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

        float r = rng.nextFloat();
        uint32_t lobe = 0u; // 0 = base, 1 = flake, 2 = coat
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

        simd::float3 wi;
        if (lobe == 2u) {
            float coatRoughness = PlasticCoatRoughness(material);
            simd::float3 wh = SampleGgxVndf(rng, coatRoughness, normal, wo);
            if (simd::dot(wh, normal) <= 0.0f) {
                return result;
            }
            wi = simd::normalize(Reflect(-wo, wh));
        } else if (lobe == 1u) {
            float flakeRoughness = std::max(CarpaintFlakeRoughness(material), 1.0e-3f);
            float alphaFlake = flakeRoughness * flakeRoughness;
            simd::float3 flakeNormal = CarpaintFlakeNormal(material, position, normal);
            simd::float3 wh = SampleGgxHalfVector(rng, alphaFlake, flakeNormal);
            if (simd::dot(wh, flakeNormal) <= 0.0f) {
                return result;
            }
            wi = simd::normalize(Reflect(-wo, wh));
        } else {
            float metallic = CarpaintBaseMetallic(material);
            float diffuseWeight = std::max(1.0f - metallic, 0.0f);
            float specWeight = std::max(metallic, 0.0f);
            float weightSum = diffuseWeight + specWeight;
            float choose = rng.nextFloat();
            bool sampleSpec = (specWeight > 0.0f) && (weightSum > 0.0f) &&
                              (choose < specWeight / std::max(weightSum, 1.0e-6f));
            if (sampleSpec) {
                float baseRough = std::max(CarpaintBaseRoughness(material), 1.0e-3f);
                float alphaBase = baseRough * baseRough;
                simd::float3 wh = SampleGgxHalfVector(rng, alphaBase, normal);
                if (simd::dot(wh, normal) <= 0.0f) {
                    return result;
                }
                wi = simd::normalize(Reflect(-wo, wh));
            } else {
                float pdf = 0.0f;
                wi = SampleCosineHemisphere(rng, normal, pdf);
            }
        }

        if (!IsFinite(wi) || simd::dot(normal, wi) <= 0.0f) {
            return result;
        }

        CarpaintLobeResult coatRes = CarpaintEvalCoat(material, normal, wo, wi, clampParams);
        CarpaintLobeResult flakeRes = CarpaintEvalFlake(material, position, normal, wo, wi, clampParams);
        CarpaintLobeResult baseRes = CarpaintEvalBase(material, normal, wo, wi, clampParams);

        float combinedPdf = pBase * baseRes.pdf + pFlake * flakeRes.pdf + pCoat * coatRes.pdf;
        if (combinedPdf <= 0.0f) {
            return result;
        }

        simd::float3 selectedF = baseRes.value;
        float selectedPdf = baseRes.pdf;
        if (lobe == 1u) {
            selectedF = flakeRes.value;
            selectedPdf = flakeRes.pdf;
        } else if (lobe == 2u) {
            selectedF = coatRes.value;
            selectedPdf = coatRes.pdf;
        }
        if (selectedPdf <= 0.0f ||
            !(selectedF.x > 0.0f || selectedF.y > 0.0f || selectedF.z > 0.0f)) {
            return result;
        }

        float cosThetaI = std::max(simd::dot(normal, wi), 0.0f);
        if (cosThetaI <= 0.0f) {
            return result;
        }
        simd::float3 weight = selectedF * cosThetaI / combinedPdf;
        if (!IsFinite(weight)) {
            return result;
        }

        result.direction = wi;
        result.weight = simd::max(weight, simd_make_float3(0.0f, 0.0f, 0.0f));
        result.pdf = combinedPdf;
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Dielectric)) {
        result.isDelta = true;
        float refIdx = std::max(material.typeEta.y, 1.0f);
        float etaI = 1.0f;
        float etaT = refIdx;
        float cosThetaO = std::clamp(simd::dot(-incidentDir, normal), -1.0f, 1.0f);
        if (!frontFace) {
            etaI = refIdx;
            etaT = 1.0f;
        }
        float relativeEta = etaI / etaT;
        float cosThetaT = 0.0f;
        float Fr = FresnelDielectricExact(cosThetaO, etaI, etaT, cosThetaT);

        simd::float3 direction;
        simd::float3 weight;
        if (rng.nextFloat() < Fr) {
            direction = Reflect(incidentDir, normal);
            weight = simd_make_float3(Fr, Fr, Fr);
        } else {
            simd::float3 refracted;
            if (!Refract(incidentDir, normal, relativeEta, refracted) ||
                simd::dot(refracted, refracted) <= 0.0f) {
                direction = Reflect(incidentDir, normal);
                weight = simd_make_float3(Fr, Fr, Fr);
            } else {
                direction = simd::normalize(refracted);
                float etaScale = (etaT * etaT) / (etaI * etaI);
                float cosThetaI = std::fabs(cosThetaO);
                float cosThetaTrans = std::fabs(cosThetaT);
                float directionScale = etaScale * (cosThetaTrans / std::max(cosThetaI, 1.0e-6f));
                float scale = std::max(1.0f - Fr, 0.0f) * directionScale;
                weight = simd_make_float3(scale, scale, scale);
            }
        }

        result.direction = simd::normalize(direction);
        result.weight = weight;
        result.pdf = 1.0f;
        result.isDelta = true;
        return result;
    }

    float pdf = 0.0f;
    simd::float3 wi = SampleCosineHemisphere(rng, normal, pdf);
    if (pdf <= 0.0f) {
        return result;
    }
    simd::float3 albedo = MaterialBaseColor(material);
    simd::float3 weight = albedo;
    if (!IsFinite(weight)) {
        return result;
    }
    result.direction = wi;
    result.weight = simd::max(weight, simd_make_float3(0.0f, 0.0f, 0.0f));
    result.pdf = pdf;
    return result;
}

bool LoadEnvironmentMap(const std::string& path, EnvironmentMap& outEnv) {
    outEnv = EnvironmentMap{};
    if (path.empty()) {
        return false;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        return false;
    }

    NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
    NSURL* url = [NSURL fileURLWithPath:nsPath];
    MTKTextureLoader* loader = [[MTKTextureLoader alloc] initWithDevice:device];
    NSDictionary* options = @{MTKTextureLoaderOptionSRGB : @NO,
                              MTKTextureLoaderOptionTextureUsage : @(MTLTextureUsageShaderRead),
                              MTKTextureLoaderOptionTextureStorageMode : @(MTLStorageModeShared),
                              MTKTextureLoaderOptionAllocateMipmaps : @NO,
                              MTKTextureLoaderOptionGenerateMipmaps : @NO};
    NSError* error = nil;
    id<MTLTexture> texture = [loader newTextureWithContentsOfURL:url options:options error:&error];
    if (!texture) {
        return false;
    }

    const NSUInteger width = texture.width;
    const NSUInteger height = texture.height;
    if (width == 0 || height == 0) {
        return false;
    }

    const MTLPixelFormat format = texture.pixelFormat;
    NSUInteger bytesPerPixel = 0;
    switch (format) {
        case MTLPixelFormatRGBA32Float:
            bytesPerPixel = sizeof(float) * 4u;
            break;
        case MTLPixelFormatRGBA16Float:
            bytesPerPixel = sizeof(uint16_t) * 4u;
            break;
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGRA8Unorm_sRGB:
            bytesPerPixel = sizeof(uint8_t) * 4u;
            break;
        default:
            return false;
    }

    const NSUInteger pixelCount = width * height;
    const NSUInteger bytesPerRow = width * bytesPerPixel;
    std::vector<uint8_t> rawData(static_cast<size_t>(bytesPerRow) * height);

    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    [texture getBytes:rawData.data()
          bytesPerRow:bytesPerRow
           fromRegion:region
          mipmapLevel:0];

    outEnv.rgba.assign(pixelCount * 4u, 0.0f);
    if (format == MTLPixelFormatRGBA32Float) {
        const float* src = reinterpret_cast<const float*>(rawData.data());
        std::copy(src, src + outEnv.rgba.size(), outEnv.rgba.begin());
    } else if (format == MTLPixelFormatRGBA16Float) {
        const __fp16* src = reinterpret_cast<const __fp16*>(rawData.data());
        for (size_t i = 0; i < outEnv.rgba.size(); ++i) {
            outEnv.rgba[i] = static_cast<float>(src[i]);
        }
    } else {
        const uint8_t* src = rawData.data();
        for (NSUInteger i = 0; i < pixelCount; ++i) {
            uint8_t r = src[i * 4u + 0u];
            uint8_t g = src[i * 4u + 1u];
            uint8_t b = src[i * 4u + 2u];
            uint8_t a = src[i * 4u + 3u];
            if (format == MTLPixelFormatBGRA8Unorm || format == MTLPixelFormatBGRA8Unorm_sRGB) {
                std::swap(r, b);
            }
            outEnv.rgba[i * 4u + 0u] = static_cast<float>(r) / 255.0f;
            outEnv.rgba[i * 4u + 1u] = static_cast<float>(g) / 255.0f;
            outEnv.rgba[i * 4u + 2u] = static_cast<float>(b) / 255.0f;
            outEnv.rgba[i * 4u + 3u] = static_cast<float>(a) / 255.0f;
        }
    }

    outEnv.width = static_cast<uint32_t>(width);
    outEnv.height = static_cast<uint32_t>(height);

    PathTracer::EnvImportanceDistribution dist;
    std::string errorMsg;
    if (PathTracer::BuildEnvImportanceDistribution(outEnv.rgba.data(),
                                                   outEnv.width,
                                                   outEnv.height,
                                                   &dist,
                                                   &errorMsg)) {
        outEnv.distribution = std::move(dist);
        outEnv.hasDistribution = true;
    }

    return true;
}

#if defined(PATH_TRACER_ENABLE_EMBREE)

void EmbreeErrorHandler(void*, RTCError code, const char* message) {
    if (code == RTC_ERROR_NONE || !message) {
        return;
    }
    std::cerr << "Embree error " << static_cast<int>(code) << ": " << message << std::endl;
}

struct EmbreeSceneData {
    RTCDevice device = nullptr;
    RTCScene scene = nullptr;
    std::vector<std::unique_ptr<GeometryData>> geometryData;
    std::vector<std::vector<simd::float3>> meshPositions;
    std::vector<std::vector<simd::float3>> meshNormals;
    std::vector<std::vector<simd::float2>> meshUvs;
    std::vector<std::vector<uint32_t>> meshIndices;
    std::vector<simd::float4> sphereData;
    std::vector<uint32_t> sphereMaterials;
    std::vector<simd::float3> rectPositions;
    std::vector<simd::float3> rectNormals;
    std::vector<uint32_t> rectIndices;
    std::vector<uint32_t> rectMaterials;
    std::vector<uint32_t> rectTriangleToRect;
    const PathTracerShaderTypes::RectData* rectangles = nullptr;
    uint32_t rectangleCount = 0;
};

simd::float3 TransformPoint(const simd::float4x4& transform, const simd::float3& point) {
    simd::float4 result = simd_mul(transform, simd_make_float4(point, 1.0f));
    return {result.x, result.y, result.z};
}

simd::float3x3 NormalMatrix(const simd::float4x4& transform) {
    simd::float4x4 worldToLocal = simd_inverse(transform);
    simd::float3x3 normalMatrix;
    normalMatrix.columns[0] = simd_make_float3(worldToLocal.columns[0].x,
                                               worldToLocal.columns[1].x,
                                               worldToLocal.columns[2].x);
    normalMatrix.columns[1] = simd_make_float3(worldToLocal.columns[0].y,
                                               worldToLocal.columns[1].y,
                                               worldToLocal.columns[2].y);
    normalMatrix.columns[2] = simd_make_float3(worldToLocal.columns[0].z,
                                               worldToLocal.columns[1].z,
                                               worldToLocal.columns[2].z);
    return normalMatrix;
}

simd::float3 TransformNormal(const simd::float3x3& normalMatrix, const simd::float3& normal) {
    return normalMatrix.columns[0] * normal.x +
           normalMatrix.columns[1] * normal.y +
           normalMatrix.columns[2] * normal.z;
}

bool BuildEmbreeScene(const PathTracer::SceneResources& resources, EmbreeSceneData& outData, std::string& error) {
    outData = EmbreeSceneData{};

    RTCDevice device = rtcNewDevice(nullptr);
    if (!device) {
        error = "Failed to create Embree device";
        return false;
    }
    rtcSetDeviceErrorFunction(device, EmbreeErrorHandler, nullptr);

    RTCScene scene = rtcNewScene(device);
    rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_HIGH);
    rtcSetSceneFlags(scene, RTC_SCENE_FLAG_NONE);

    outData.device = device;
    outData.scene = scene;

    const auto& meshes = resources.meshes();
    outData.meshPositions.reserve(meshes.size());
    outData.meshNormals.reserve(meshes.size());
    outData.meshUvs.reserve(meshes.size());
    outData.meshIndices.reserve(meshes.size());

    for (const auto& mesh : meshes) {
        if (mesh.vertices.empty() || mesh.indices.empty()) {
            continue;
        }

        const simd::float3x3 normalMatrix = NormalMatrix(mesh.localToWorld);

        std::vector<simd::float3> positions;
        std::vector<simd::float3> normals;
        std::vector<simd::float2> uvs;
        positions.reserve(mesh.vertices.size());
        normals.reserve(mesh.vertices.size());
        uvs.reserve(mesh.vertices.size());

        for (const auto& vertex : mesh.vertices) {
            positions.push_back(TransformPoint(mesh.localToWorld, vertex.position));
            simd::float3 n = TransformNormal(normalMatrix, vertex.normal);
            normals.push_back(simd::length(n) > 0.0f ? simd::normalize(n) : n);
            uvs.push_back(vertex.uv);
        }

        outData.meshPositions.emplace_back(std::move(positions));
        outData.meshNormals.emplace_back(std::move(normals));
        outData.meshUvs.emplace_back(std::move(uvs));
        outData.meshIndices.emplace_back(mesh.indices);

        auto geomData = std::make_unique<GeometryData>();
        geomData->type = GeometryType::Mesh;
        geomData->positions = outData.meshPositions.back().data();
        geomData->normals = outData.meshNormals.back().data();
        geomData->uvs = outData.meshUvs.back().data();
        geomData->indices = outData.meshIndices.back().data();
        geomData->indexCount = static_cast<uint32_t>(outData.meshIndices.back().size());
        geomData->materialIndex = mesh.materialIndex;

        RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
        float* vertexBuffer = reinterpret_cast<float*>(
            rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0,
                                    RTC_FORMAT_FLOAT3, sizeof(float) * 3,
                                    outData.meshPositions.back().size()));
        if (vertexBuffer) {
            const auto& positions = outData.meshPositions.back();
            for (size_t i = 0; i < positions.size(); ++i) {
                const simd::float3 p = positions[i];
                vertexBuffer[i * 3u + 0u] = p.x;
                vertexBuffer[i * 3u + 1u] = p.y;
                vertexBuffer[i * 3u + 2u] = p.z;
            }
        }

        unsigned int triCount = static_cast<unsigned int>(geomData->indexCount / 3);
        uint32_t* indexBuffer = reinterpret_cast<uint32_t*>(
            rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0,
                                    RTC_FORMAT_UINT3, sizeof(uint32_t) * 3,
                                    triCount));
        if (indexBuffer) {
            memcpy(indexBuffer, outData.meshIndices.back().data(),
                   sizeof(uint32_t) * geomData->indexCount);
        }

        rtcSetGeometryUserData(geometry, geomData.get());
        rtcCommitGeometry(geometry);
        rtcAttachGeometry(scene, geometry);
        rtcReleaseGeometry(geometry);

        outData.geometryData.emplace_back(std::move(geomData));
    }

    const auto* spheres = resources.spheresData();
    if (resources.sphereCount() > 0 && spheres) {
        outData.sphereData.reserve(resources.sphereCount());
        outData.sphereMaterials.reserve(resources.sphereCount());
        for (uint32_t i = 0; i < resources.sphereCount(); ++i) {
            simd::float4 centerRadius = spheres[i].centerRadius;
            outData.sphereData.push_back(centerRadius);
            outData.sphereMaterials.push_back(spheres[i].materialIndex.x);
        }

        auto geomData = std::make_unique<GeometryData>();
        geomData->type = GeometryType::Spheres;
        geomData->spheres = outData.sphereData.data();
        geomData->materialIndices = outData.sphereMaterials.data();
        geomData->count = static_cast<uint32_t>(outData.sphereData.size());

        RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_SPHERE_POINT);
        float* vertexBuffer = reinterpret_cast<float*>(
            rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0,
                                    RTC_FORMAT_FLOAT4, sizeof(float) * 4,
                                    outData.sphereData.size()));
        if (vertexBuffer) {
            memcpy(vertexBuffer, outData.sphereData.data(),
                   sizeof(simd::float4) * outData.sphereData.size());
        }
        rtcSetGeometryUserData(geometry, geomData.get());
        rtcCommitGeometry(geometry);
        rtcAttachGeometry(scene, geometry);
        rtcReleaseGeometry(geometry);

        outData.geometryData.emplace_back(std::move(geomData));
    }

    const auto* rectangles = resources.rectanglesData();
    if (resources.rectangleCount() > 0 && rectangles) {
        outData.rectangles = rectangles;
        outData.rectangleCount = resources.rectangleCount();
        outData.rectPositions.reserve(resources.rectangleCount() * 4u);
        outData.rectNormals.reserve(resources.rectangleCount() * 4u);
        outData.rectIndices.reserve(resources.rectangleCount() * 6u);
        outData.rectMaterials.reserve(resources.rectangleCount() * 2u);
        outData.rectTriangleToRect.reserve(resources.rectangleCount() * 2u);

        for (uint32_t i = 0; i < resources.rectangleCount(); ++i) {
            const PathTracerShaderTypes::RectData& rect = rectangles[i];
            simd::float3 corner = {rect.corner.x, rect.corner.y, rect.corner.z};
            simd::float3 edgeU = {rect.edgeU.x, rect.edgeU.y, rect.edgeU.z};
            simd::float3 edgeV = {rect.edgeV.x, rect.edgeV.y, rect.edgeV.z};
            simd::float3 normal = {rect.normalAndPlane.x, rect.normalAndPlane.y, rect.normalAndPlane.z};
            normal = simd::normalize(normal);

            uint32_t baseIndex = static_cast<uint32_t>(outData.rectPositions.size());
            outData.rectPositions.push_back(corner);
            outData.rectPositions.push_back(corner + edgeU);
            outData.rectPositions.push_back(corner + edgeV);
            outData.rectPositions.push_back(corner + edgeU + edgeV);

            for (int j = 0; j < 4; ++j) {
                outData.rectNormals.push_back(normal);
            }

            simd::float3 geomNormal = simd::normalize(simd::cross(edgeU, edgeV));
            bool flipWinding = simd::dot(geomNormal, normal) < 0.0f;

            if (!flipWinding) {
                outData.rectIndices.push_back(baseIndex + 0u);
                outData.rectIndices.push_back(baseIndex + 1u);
                outData.rectIndices.push_back(baseIndex + 2u);
                outData.rectMaterials.push_back(rect.materialTwoSided.x);
                outData.rectTriangleToRect.push_back(i);

                outData.rectIndices.push_back(baseIndex + 2u);
                outData.rectIndices.push_back(baseIndex + 1u);
                outData.rectIndices.push_back(baseIndex + 3u);
                outData.rectMaterials.push_back(rect.materialTwoSided.x);
                outData.rectTriangleToRect.push_back(i);
            } else {
                outData.rectIndices.push_back(baseIndex + 0u);
                outData.rectIndices.push_back(baseIndex + 2u);
                outData.rectIndices.push_back(baseIndex + 1u);
                outData.rectMaterials.push_back(rect.materialTwoSided.x);
                outData.rectTriangleToRect.push_back(i);

                outData.rectIndices.push_back(baseIndex + 1u);
                outData.rectIndices.push_back(baseIndex + 2u);
                outData.rectIndices.push_back(baseIndex + 3u);
                outData.rectMaterials.push_back(rect.materialTwoSided.x);
                outData.rectTriangleToRect.push_back(i);
            }
        }

        auto geomData = std::make_unique<GeometryData>();
        geomData->type = GeometryType::Rectangles;
        geomData->positions = outData.rectPositions.data();
        geomData->normals = outData.rectNormals.data();
        geomData->indices = outData.rectIndices.data();
        geomData->materialIndices = outData.rectMaterials.data();
        geomData->indexCount = static_cast<uint32_t>(outData.rectIndices.size());

        RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
        float* vertexBuffer = reinterpret_cast<float*>(
            rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0,
                                    RTC_FORMAT_FLOAT3, sizeof(float) * 3,
                                    outData.rectPositions.size()));
        if (vertexBuffer) {
            for (size_t i = 0; i < outData.rectPositions.size(); ++i) {
                const simd::float3 p = outData.rectPositions[i];
                vertexBuffer[i * 3u + 0u] = p.x;
                vertexBuffer[i * 3u + 1u] = p.y;
                vertexBuffer[i * 3u + 2u] = p.z;
            }
        }
        unsigned int triCount = static_cast<unsigned int>(outData.rectIndices.size() / 3);
        uint32_t* indexBuffer = reinterpret_cast<uint32_t*>(
            rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0,
                                    RTC_FORMAT_UINT3, sizeof(uint32_t) * 3,
                                    triCount));
        if (indexBuffer) {
            memcpy(indexBuffer, outData.rectIndices.data(),
                   sizeof(uint32_t) * outData.rectIndices.size());
        }

        rtcSetGeometryUserData(geometry, geomData.get());
        rtcCommitGeometry(geometry);
        rtcAttachGeometry(scene, geometry);
        rtcReleaseGeometry(geometry);

        outData.geometryData.emplace_back(std::move(geomData));
    }

    rtcCommitScene(scene);
    return true;
}

bool IntersectScene(const EmbreeSceneData& sceneData, const Ray& ray, HitInfo& outHit) {
    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);

    RTCRayHit rayHit{};
    rayHit.ray.org_x = ray.origin.x;
    rayHit.ray.org_y = ray.origin.y;
    rayHit.ray.org_z = ray.origin.z;
    rayHit.ray.dir_x = ray.direction.x;
    rayHit.ray.dir_y = ray.direction.y;
    rayHit.ray.dir_z = ray.direction.z;
    rayHit.ray.tnear = kEpsilon;
    rayHit.ray.tfar = std::numeric_limits<float>::infinity();
    rayHit.ray.mask = 0xFFFFFFFFu;
    rayHit.ray.flags = 0;
    rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(sceneData.scene, &rayHit, &args);
    if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        return false;
    }

    const GeometryData* geomData = reinterpret_cast<const GeometryData*>(
        rtcGetGeometryUserData(rtcGetGeometry(sceneData.scene, rayHit.hit.geomID)));
    if (!geomData) {
        return false;
    }

    outHit.t = rayHit.ray.tfar;
    outHit.position = ray.origin + outHit.t * ray.direction;

    simd::float3 Ng = {rayHit.hit.Ng_x, rayHit.hit.Ng_y, rayHit.hit.Ng_z};
    if (simd::dot(Ng, Ng) > 0.0f) {
        outHit.normal = simd::normalize(Ng);
    }

    outHit.frontFace = simd::dot(ray.direction, outHit.normal) < 0.0f;
    simd::float3 adjustedNormal = outHit.frontFace ? outHit.normal : -outHit.normal;

    uint32_t primID = rayHit.hit.primID;
    simd::float3 shadingNormal = adjustedNormal;
    uint32_t materialIndex = geomData->materialIndex;
    outHit.primitiveType = geomData->type;
    outHit.primitiveIndex = primID;
    outHit.twoSided = false;

    if (geomData->type == GeometryType::Mesh && geomData->indices && geomData->normals) {
        uint32_t base = primID * 3u;
        if (base + 2u < geomData->indexCount) {
            uint32_t i0 = geomData->indices[base + 0u];
            uint32_t i1 = geomData->indices[base + 1u];
            uint32_t i2 = geomData->indices[base + 2u];
            float u = rayHit.hit.u;
            float v = rayHit.hit.v;
            float w = 1.0f - u - v;
            simd::float3 n0 = geomData->normals[i0];
            simd::float3 n1 = geomData->normals[i1];
            simd::float3 n2 = geomData->normals[i2];
            simd::float3 interp = w * n0 + u * n1 + v * n2;
            if (simd::dot(interp, interp) > 0.0f) {
                shadingNormal = simd::normalize(interp);
                if (simd::dot(shadingNormal, adjustedNormal) < 0.0f) {
                    shadingNormal = -shadingNormal;
                }
            }
        }
    } else if (geomData->type == GeometryType::Rectangles && geomData->normals) {
        uint32_t index = primID * 3u;
        if (index < geomData->indexCount) {
            uint32_t vertIndex = geomData->indices[index];
            shadingNormal = geomData->normals[vertIndex];
            if (simd::dot(shadingNormal, adjustedNormal) < 0.0f) {
                shadingNormal = -shadingNormal;
            }
        }
        uint32_t rectIndex = primID;
        if (primID < sceneData.rectTriangleToRect.size()) {
            rectIndex = sceneData.rectTriangleToRect[primID];
        }
        outHit.primitiveIndex = rectIndex;
        outHit.primitiveType = GeometryType::Rectangles;
        if (sceneData.rectangles && rectIndex < sceneData.rectangleCount) {
            outHit.twoSided = sceneData.rectangles[rectIndex].materialTwoSided.y != 0u;
        }
        if (geomData->materialIndices) {
            materialIndex = geomData->materialIndices[primID];
        }
    } else if (geomData->type == GeometryType::Spheres && geomData->spheres) {
        if (primID < geomData->count) {
            simd::float4 sphere = geomData->spheres[primID];
            simd::float3 center = {sphere.x, sphere.y, sphere.z};
            simd::float3 normal = simd::normalize(outHit.position - center);
            shadingNormal = normal;
            adjustedNormal = simd::dot(ray.direction, normal) < 0.0f ? normal : -normal;
            outHit.normal = normal;
            outHit.frontFace = simd::dot(ray.direction, normal) < 0.0f;
            outHit.primitiveType = GeometryType::Spheres;
            outHit.primitiveIndex = primID;
            outHit.twoSided = true;
            if (geomData->materialIndices) {
                materialIndex = geomData->materialIndices[primID];
            }
        }
    }

    outHit.shadingNormal = shadingNormal;
    outHit.materialIndex = materialIndex;
    return true;
}

bool IsOccluded(const EmbreeSceneData& sceneData,
                const simd::float3& origin,
                const simd::float3& direction,
                float tMax) {
    RTCOccludedArguments args;
    rtcInitOccludedArguments(&args);

    RTCRay ray{};
    ray.org_x = origin.x;
    ray.org_y = origin.y;
    ray.org_z = origin.z;
    ray.dir_x = direction.x;
    ray.dir_y = direction.y;
    ray.dir_z = direction.z;
    ray.tnear = kEpsilon;
    ray.tfar = tMax;
    ray.mask = 0xFFFFFFFFu;
    ray.flags = 0;

    rtcOccluded1(sceneData.scene, &ray, &args);
    return ray.tfar < 0.0f;
}

#endif

}  // namespace

void EmbreeHeadlessRenderer::setMaxThreads(uint32_t maxThreads) {
    maxThreads_ = maxThreads;
}

bool EmbreeHeadlessRenderer::render(const HeadlessScene& scene,
                                    const HeadlessCamera&,
                                    const PathTracer::RenderSettings& settings,
                                    uint32_t sppTotal,
                                    bool verbose,
                                    HeadlessRenderOutput& out,
                                    std::string& error) {
#if !defined(PATH_TRACER_ENABLE_EMBREE)
    (void)scene;
    (void)settings;
    (void)sppTotal;
    error = "Embree backend not enabled (build with PATH_TRACER_ENABLE_EMBREE=ON)";
    return false;
#else
    if (!scene.resources) {
        error = "Embree backend requires scene resources";
        return false;
    }

    const uint32_t width = settings.renderWidth > 0 ? settings.renderWidth : 1280u;
    const uint32_t height = settings.renderHeight > 0 ? settings.renderHeight : 720u;

    EnvironmentMap envMap;
    EnvironmentMap* envPtr = nullptr;
    if (!settings.environmentMapPath.empty()) {
        if (LoadEnvironmentMap(settings.environmentMapPath, envMap)) {
            envPtr = &envMap;
        }
    }

    EmbreeSceneData sceneData;
    if (!BuildEmbreeScene(*scene.resources, sceneData, error)) {
        return false;
    }

    const auto* materials = scene.resources->materialsData();
    const uint32_t materialCount = scene.resources->materialCount();
    const auto* rectangles = scene.resources->rectanglesData();
    const uint32_t rectangleCount = scene.resources->rectangleCount();
    const float emissionScale = LoadEmbreeEmissionScale();

    std::vector<RectLightInfo> rectLights;
    std::vector<int32_t> rectLightIndexByRect;
    if (rectangles && rectangleCount > 0 && materials && materialCount > 0) {
        rectLights.reserve(rectangleCount);
        rectLightIndexByRect.assign(rectangleCount, -1);
        for (uint32_t i = 0; i < rectangleCount; ++i) {
            const auto& rect = rectangles[i];
            uint32_t matIndex = std::min(rect.materialTwoSided.x, materialCount - 1u);
            const auto& material = materials[matIndex];
            if (static_cast<uint32_t>(material.typeEta.x) !=
                static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::DiffuseLight)) {
                continue;
            }
            simd::float3 baseEmission = simd_make_float3(material.emission.x,
                                                         material.emission.y,
                                                         material.emission.z);
            baseEmission *= emissionScale;
            if (!(simd::dot(baseEmission, baseEmission) > 0.0f)) {
                continue;
            }

            RectLightInfo info;
            info.rectIndex = i;
            info.corner = simd_make_float3(rect.corner.x, rect.corner.y, rect.corner.z);
            info.edgeU = simd_make_float3(rect.edgeU.x, rect.edgeU.y, rect.edgeU.z);
            info.edgeV = simd_make_float3(rect.edgeV.x, rect.edgeV.y, rect.edgeV.z);
            info.normal = simd::normalize(simd_make_float3(rect.normalAndPlane.x,
                                                           rect.normalAndPlane.y,
                                                           rect.normalAndPlane.z));
            info.twoSided = rect.materialTwoSided.y != 0u;
            info.baseEmission = baseEmission;
            // Disable emitEnv for Embree rect lights to match SWRT energy during parity checks.
            info.emissionUsesEnv = false;
            info.area = simd::length(simd::cross(info.edgeU, info.edgeV));

            rectLightIndexByRect[i] = static_cast<int32_t>(rectLights.size());
            rectLights.push_back(info);
        }
    }
    const uint32_t rectLightCount = static_cast<uint32_t>(rectLights.size());
    const bool envMapAvailable = envPtr && envPtr->width > 0 && envPtr->height > 0 && !envPtr->rgba.empty();
    const bool envSampling = envPtr && envPtr->hasDistribution && envMapAvailable;
    const FireflyClampParams clampParams = MakeFireflyParams(settings);

    CameraBasis camera = BuildCamera(settings, width, height);
    DirectDebugConfig debugCfg = LoadDirectDebugConfig(width, height);

    std::vector<float> linearRGB(width * height * 3u, 0.0f);
    const uint32_t targetSamples = std::max<uint32_t>(1u, sppTotal);

    uint32_t seedBase = settings.fixedRngSeed != 0 ? settings.fixedRngSeed : 0x9e3779b9u;

    CFAbsoluteTime renderStart = CFAbsoluteTimeGetCurrent();

    constexpr uint32_t kTileSize = 16u;
    constexpr size_t kTileArea = static_cast<size_t>(kTileSize) * kTileSize;
    const uint32_t tilesX = (width + kTileSize - 1u) / kTileSize;
    const uint32_t tilesY = (height + kTileSize - 1u) / kTileSize;
    const uint32_t totalTiles = tilesX * tilesY;
    const uint32_t reportStride = std::max(1u, totalTiles / 100u);
    std::atomic<uint32_t> tilesDone{0};
    std::atomic<uint32_t> nextReport{reportStride};
    std::atomic<uint32_t> lastReportedPercent{0};
    std::mutex progressMutex;

    auto reportProgress = [&](uint32_t doneTiles) {
        uint32_t percent = static_cast<uint32_t>(
            (static_cast<uint64_t>(doneTiles) * 100u) / totalTiles);
        {
            std::lock_guard<std::mutex> lock(progressMutex);
            std::cout << "\rEmbree progress: " << percent << "% (" << doneTiles
                      << "/" << totalTiles << " tiles)" << std::flush;
        }
        lastReportedPercent.store(percent, std::memory_order_relaxed);
    };

    uint32_t threadLimit = maxThreads_;
    if (threadLimit == 0) {
        uint32_t hwThreads = static_cast<uint32_t>(std::thread::hardware_concurrency());
        if (hwThreads > 0) {
            threadLimit = hwThreads;
        }
    }

    uint32_t workerCount = threadLimit;
    if (workerCount == 0) {
        workerCount = 1;
    }

    auto renderTile = [&](uint32_t tileX, uint32_t tileY) {
        const uint32_t x0 = tileX * kTileSize;
        const uint32_t y0 = tileY * kTileSize;
        const uint32_t x1 = std::min(x0 + kTileSize, width);
        const uint32_t y1 = std::min(y0 + kTileSize, height);
        const uint32_t tileWidth = x1 - x0;
        const uint32_t tileHeight = y1 - y0;

        std::array<simd::float3, kTileArea> tilePixels;

        for (uint32_t localY = 0; localY < tileHeight; ++localY) {
            uint32_t y = y0 + localY;
            for (uint32_t localX = 0; localX < tileWidth; ++localX) {
                uint32_t x = x0 + localX;
                simd::float3 pixelRadiance = {0.0f, 0.0f, 0.0f};
                uint32_t pixelIndex = y * width + x;
                const bool debugPixel = debugCfg.enabled && x == debugCfg.pixelX && y == debugCfg.pixelY;

                for (uint32_t s = 0; s < targetSamples; ++s) {
                    const bool debugSample = debugPixel && s == 0;
                    Rng rng;
                    rng.state = Rng::Hash(seedBase ^ pixelIndex ^ (s * 0x9e3779b9u));

                    Ray ray = GenerateCameraRay(camera, width, height, x, y, rng);
                    simd::float3 throughput = {1.0f, 1.0f, 1.0f};
                    simd::float3 radiance = {0.0f, 0.0f, 0.0f};
                    float lastBsdfPdf = 1.0f;
                    bool lastScatterWasDelta = true;
                    uint32_t specularDepth = 0;
                    bool hadTransmission = false;

                    for (uint32_t depth = 0; depth < settings.maxDepth; ++depth) {
                        HitInfo hit;
                        if (!IntersectScene(sceneData, ray, hit)) {
                            simd::float3 background = EvaluateBackground(settings, envPtr, ray.direction);
                            float misWeight = 1.0f;
                            bool useSpecularMis = (!lastScatterWasDelta) ||
                                                  settings.enableSpecularNee ||
                                                  settings.enableMnee;
                            if (useSpecularMis && envSampling) {
                                float lightPdf = EnvironmentPdf(*envPtr, settings.environmentRotation, ray.direction);
                                float denom = lastBsdfPdf + lightPdf;
                                if (denom > 0.0f) {
                                    misWeight = lastBsdfPdf / denom;
                                }
                                misWeight = std::clamp(misWeight,
                                                       kMisWeightClampMin,
                                                       kMisWeightClampMax);
                            }
                            radiance += ClampFireflyContribution(throughput,
                                                                 background * misWeight,
                                                                 clampParams);
                            break;
                        }

                        if (!materials || materialCount == 0) {
                            break;
                        }

                        uint32_t matIndex = std::min(hit.materialIndex, materialCount - 1);
                        const auto& material = materials[matIndex];
                        uint32_t type = static_cast<uint32_t>(material.typeEta.x);
                        simd::float3 incidentDir = simd::normalize(ray.direction);
                        simd::float3 wo = -incidentDir;
                        simd::float3 shadingNormal = hit.shadingNormal;
                        if (simd::dot(shadingNormal, shadingNormal) <= 0.0f) {
                            shadingNormal = hit.normal;
                        }
                        if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Dielectric) ||
                            MaterialUsesGeometricNormal(material)) {
                            shadingNormal = hit.normal;
                        }
                        shadingNormal = simd::normalize(shadingNormal);
                        if (debugSample && depth == 0) {
                            simd::float3 baseColor = MaterialBaseColor(material);
                            std::cerr << "[EmbreeDirectDebug] hit"
                                      << " pixel=(" << x << "," << y << ")"
                                      << " materialType=" << type
                                      << " frontFace=" << (hit.frontFace ? 1 : 0)
                                      << " twoSided=" << (hit.twoSided ? 1 : 0)
                                      << " nDotV=" << std::max(simd::dot(shadingNormal, wo), 0.0f)
                                      << " baseLum=" << Luminance(baseColor)
                                      << " roughness=" << MaterialRoughness(material)
                                      << " rectLights=" << rectLightCount
                                      << std::endl;
                        }

                        if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::DiffuseLight)) {
                            simd::float3 emission = simd_make_float3(material.emission.x,
                                                                     material.emission.y,
                                                                     material.emission.z);
                            emission *= emissionScale;
                            if (material.emission.w > 0.0f && envMapAvailable && hit.frontFace) {
                                simd::float3 sampleDir = -shadingNormal;
                                emission *= SampleEnvironment(*envPtr,
                                                              sampleDir,
                                                              settings.environmentRotation,
                                                              settings.environmentIntensity);
                            }
                            if ((simd::dot(emission, emission) > 0.0f) &&
                                (hit.frontFace || hit.twoSided)) {
                                float misWeight = 1.0f;
                                bool useSpecularMis = (!lastScatterWasDelta) ||
                                                      settings.enableSpecularNee ||
                                                      settings.enableMnee;
                                if (useSpecularMis && rectLightCount > 0) {
                                    float lightPdf = RectLightPdfForHit(rectLightIndexByRect,
                                                                        rectangles,
                                                                        rectangleCount,
                                                                        rectLightCount,
                                                                        hit,
                                                                        ray.origin);
                                    float denom = lastBsdfPdf + lightPdf;
                                    if (denom > 0.0f) {
                                        misWeight = lastBsdfPdf / denom;
                                    }
                                    misWeight = std::clamp(misWeight,
                                                           kMisWeightClampMin,
                                                           kMisWeightClampMax);
                                }
                                if (debugSample && depth == 0) {
                                    std::cerr << "[EmbreeDirectDebug] emissive_hit"
                                              << " pixel=(" << x << "," << y << ")"
                                              << " depth=" << depth
                                              << " misWeight=" << misWeight
                                              << " emissionLum=" << Luminance(emission)
                                              << std::endl;
                                }
                                radiance += ClampFireflyContribution(throughput,
                                                                     emission * misWeight,
                                                                     clampParams);
                            }
                            break;
                        }

                        bool surfaceIsDelta = MaterialIsDelta(material);

                        if (!surfaceIsDelta && rectLightCount > 0) {
                            RectLightSample lightSample;
                            bool sampled = SampleRectLight(rectLights, envPtr, settings, hit, rng, lightSample);
                            if (debugSample && depth == 0 && !sampled) {
                                std::cerr << "[EmbreeDirectDebug] rect_sample_failed"
                                          << " pixel=(" << x << "," << y << ")"
                                          << std::endl;
                            }
                            if (sampled) {
                                float nDotL = std::max(simd::dot(shadingNormal, lightSample.direction), 0.0f);
                                if (lightSample.pdf > 0.0f && nDotL > 0.0f) {
                                    float shadowMax = std::max(lightSample.distance - kEpsilon, kEpsilon);
                                    bool occluded = IsOccluded(sceneData,
                                                               OffsetRayOrigin(hit, lightSample.direction),
                                                               lightSample.direction,
                                                               shadowMax);
                                    if (debugSample && depth == 0) {
                                        std::cerr << "[EmbreeDirectDebug] rect_sample"
                                                  << " pixel=(" << x << "," << y << ")"
                                                  << " nDotL=" << nDotL
                                                  << " cosLight=" << lightSample.cosLight
                                                  << " area=" << lightSample.area
                                                  << " dist=" << lightSample.distance
                                                  << " lightPdf=" << lightSample.pdf
                                                  << " occluded=" << (occluded ? 1 : 0)
                                                  << " emissionLum=" << Luminance(lightSample.emission)
                                                  << std::endl;
                                    }
                                    if (!occluded) {
                                    BsdfEval bsdfEval = EvaluateBsdf(material,
                                                                     hit.position,
                                                                     shadingNormal,
                                                                     wo,
                                                                     lightSample.direction,
                                                                     clampParams);
                                        if (debugSample && depth == 0) {
                                            std::cerr << "[EmbreeDirectDebug] rect_bsdf"
                                                      << " bsdfPdf=" << bsdfEval.pdf
                                                      << " bsdfLum=" << Luminance(bsdfEval.value)
                                                      << " isDelta=" << (bsdfEval.isDelta ? 1 : 0)
                                                      << std::endl;
                                        }
                                        if (!bsdfEval.isDelta && bsdfEval.pdf > 0.0f) {
                                            float weight = lightSample.pdf /
                                                           (lightSample.pdf + bsdfEval.pdf);
                                            simd::float3 contribution = lightSample.emission * bsdfEval.value * nDotL;
                                            contribution *= weight / lightSample.pdf;
                                            if (debugSample && depth == 0) {
                                                std::cerr << "[EmbreeDirectDebug] rect_contrib"
                                                          << " weight=" << weight
                                                          << " contribLum=" << Luminance(contribution)
                                                          << std::endl;
                                            }
                                            if (IsFinite(contribution)) {
                                                radiance += ClampFireflyContribution(throughput,
                                                                                     contribution,
                                                                                     clampParams);
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        if (!surfaceIsDelta && envSampling) {
                            PathTracer::EnvImportanceSample envSample =
                                PathTracer::SampleEnvironmentCpu(envPtr->distribution,
                                                                 rng.nextFloat(),
                                                                 rng.nextFloat(),
                                                                 rng.nextFloat(),
                                                                 settings.environmentRotation,
                                                                 settings.environmentIntensity,
                                                                 envPtr->rgba.data());
                            float nDotL = std::max(simd::dot(shadingNormal, envSample.direction), 0.0f);
                            if (envSample.pdf > 0.0f && nDotL > 0.0f) {
                                if (!IsOccluded(sceneData,
                                                OffsetRayOrigin(hit, envSample.direction),
                                                envSample.direction,
                                                std::numeric_limits<float>::infinity())) {
                                    simd::float3 envRadiance = SampleEnvironment(*envPtr,
                                                                                envSample.direction,
                                                                                settings.environmentRotation,
                                                                                settings.environmentIntensity);
                                BsdfEval bsdfEval = EvaluateBsdf(material,
                                                                 hit.position,
                                                                 shadingNormal,
                                                                 wo,
                                                                 envSample.direction,
                                                                 clampParams);
                                    if (!bsdfEval.isDelta && bsdfEval.pdf > 0.0f) {
                                        float weight = envSample.pdf / (envSample.pdf + bsdfEval.pdf);
                                        simd::float3 contribution = envRadiance * bsdfEval.value * nDotL;
                                        contribution *= weight / envSample.pdf;
                                        if (IsFinite(contribution)) {
                                            radiance += ClampFireflyContribution(throughput,
                                                                                 contribution,
                                                                                 clampParams);
                                        }
                                    }
                                }
                            }
                        }

                    BsdfSample bsdfSample = SampleBsdf(material,
                                                       hit.position,
                                                       shadingNormal,
                                                       wo,
                                                       incidentDir,
                                                       hit.frontFace,
                                                       rng,
                                                       clampParams);
                    if (bsdfSample.pdf <= 0.0f ||
                        simd::dot(bsdfSample.direction, bsdfSample.direction) <= 0.0f ||
                        !IsFinite(bsdfSample.weight)) {
                        break;
                    }

                    bool causticCandidate = (!surfaceIsDelta) && (specularDepth > 0);
                    uint32_t nextSpecularDepth = bsdfSample.isDelta ? (specularDepth + 1u) : 0u;
                    bool didTransmission = false;
                    if (bsdfSample.isDelta &&
                        type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Dielectric)) {
                        float side = hit.frontFace ? 1.0f : -1.0f;
                        didTransmission = (simd::dot(shadingNormal, bsdfSample.direction) * side) < 0.0f;
                    }
                    if (didTransmission) {
                        hadTransmission = true;
                    }
                    specularDepth = nextSpecularDepth;
                    (void)causticCandidate;
                    (void)hadTransmission;

                    bool useMnee = settings.enableMnee;
                    bool specNeeEnabled = settings.enableSpecularNee;
                    float dirLenSq = simd::dot(bsdfSample.direction, bsdfSample.direction);
                    bool specDirectionValid = (dirLenSq > 0.0f) && IsFinite(bsdfSample.direction);
                    bool mneeEligible = useMnee &&
                                        bsdfSample.isDelta &&
                                        specDirectionValid &&
                                        type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Dielectric) &&
                                        nextSpecularDepth == 1u;
                    bool specNeeEligible = specNeeEnabled &&
                                           bsdfSample.isDelta &&
                                           specDirectionValid &&
                                           !mneeEligible;

                    if (specNeeEligible && envSampling) {
                        simd::float3 neeDir = simd::normalize(bsdfSample.direction);
                        if (!IsOccluded(sceneData,
                                        OffsetRayOrigin(hit, neeDir),
                                        neeDir,
                                        std::numeric_limits<float>::infinity())) {
                            float envPdf = std::max(EnvironmentPdf(*envPtr,
                                                                   settings.environmentRotation,
                                                                   neeDir),
                                                    kSpecularNeePdfFloor);
                            float invEnvPdf = std::min(1.0f / envPdf, kSpecularNeeInvPdfClamp);
                            float bsdfPdf = std::max(bsdfSample.pdf, kSpecularNeePdfFloor);
                            float denom = envPdf + bsdfPdf;
                            float misWeight = denom > 0.0f ? (envPdf / denom) : 0.0f;
                            misWeight = std::clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                            simd::float3 envColor = SampleEnvironment(*envPtr,
                                                                      neeDir,
                                                                      settings.environmentRotation,
                                                                      settings.environmentIntensity);
                            simd::float3 neeContribution = bsdfSample.weight * envColor *
                                                           (misWeight * invEnvPdf);
                            if (IsFinite(neeContribution)) {
                                radiance += ClampFireflyContribution(throughput,
                                                                     neeContribution,
                                                                     clampParams);
                            }
                        }
                    }

                    if (specNeeEligible && rectLightCount > 0) {
                        simd::float3 neeDir = simd::normalize(bsdfSample.direction);
                        Ray neeRay;
                        neeRay.origin = OffsetRayOrigin(hit, neeDir);
                        neeRay.direction = neeDir;
                        HitInfo lightHit;
                        if (IntersectScene(sceneData, neeRay, lightHit)) {
                            RectLightHit rectHit;
                            if (RectLightHitInfo(rectLightIndexByRect,
                                                 rectLights,
                                                 rectangles,
                                                 rectangleCount,
                                                 envPtr,
                                                 settings,
                                                 lightHit,
                                                 neeRay.origin,
                                                 rectHit)) {
                                float lightPdf = std::max(rectHit.pdf, kSpecularNeePdfFloor);
                                float invLightPdf = std::min(1.0f / lightPdf, kSpecularNeeInvPdfClamp);
                                float bsdfPdf = std::max(bsdfSample.pdf, kSpecularNeePdfFloor);
                                float denom = lightPdf + bsdfPdf;
                                float misWeight = denom > 0.0f ? (lightPdf / denom) : 0.0f;
                                misWeight = std::clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                                simd::float3 neeContribution = bsdfSample.weight * rectHit.emission *
                                                               (misWeight * invLightPdf);
                                if (IsFinite(neeContribution)) {
                                    radiance += ClampFireflyContribution(throughput,
                                                                         neeContribution,
                                                                         clampParams);
                                }
                            }
                        }
                    }

                    if (mneeEligible && envSampling) {
                        simd::float3 mneeDir = simd::normalize(bsdfSample.direction);
                        if (!IsOccluded(sceneData,
                                        OffsetRayOrigin(hit, mneeDir),
                                        mneeDir,
                                        std::numeric_limits<float>::infinity())) {
                            float envPdf = std::max(EnvironmentPdf(*envPtr,
                                                                   settings.environmentRotation,
                                                                   mneeDir),
                                                    kSpecularNeePdfFloor);
                            float invEnvPdf = std::min(1.0f / envPdf, kSpecularNeeInvPdfClamp);
                            float bsdfPdf = std::max(bsdfSample.pdf, kSpecularNeePdfFloor);
                            float denom = envPdf + bsdfPdf;
                            float misWeight = denom > 0.0f ? (envPdf / denom) : 0.0f;
                            misWeight = std::clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                            simd::float3 envColor = SampleEnvironment(*envPtr,
                                                                      mneeDir,
                                                                      settings.environmentRotation,
                                                                      settings.environmentIntensity);
                            simd::float3 mneeContribution = bsdfSample.weight * envColor *
                                                            (misWeight * invEnvPdf);
                            if (IsFinite(mneeContribution)) {
                                radiance += ClampFireflyContribution(throughput,
                                                                     mneeContribution,
                                                                     clampParams);
                            }
                        }
                    }

                    if (mneeEligible && rectLightCount > 0) {
                        simd::float3 mneeDir = simd::normalize(bsdfSample.direction);
                        Ray mneeRay;
                        mneeRay.origin = OffsetRayOrigin(hit, mneeDir);
                        mneeRay.direction = mneeDir;
                        HitInfo lightHit;
                        if (IntersectScene(sceneData, mneeRay, lightHit)) {
                            RectLightHit rectHit;
                            if (RectLightHitInfo(rectLightIndexByRect,
                                                 rectLights,
                                                 rectangles,
                                                 rectangleCount,
                                                 envPtr,
                                                 settings,
                                                 lightHit,
                                                 mneeRay.origin,
                                                 rectHit)) {
                                float lightPdf = std::max(rectHit.pdf, kSpecularNeePdfFloor);
                                float invLightPdf = std::min(1.0f / lightPdf, kSpecularNeeInvPdfClamp);
                                float bsdfPdf = std::max(bsdfSample.pdf, kSpecularNeePdfFloor);
                                float denom = lightPdf + bsdfPdf;
                                float misWeight = denom > 0.0f ? (lightPdf / denom) : 0.0f;
                                misWeight = std::clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                                simd::float3 mneeContribution = bsdfSample.weight * rectHit.emission *
                                                                (misWeight * invLightPdf);
                                if (IsFinite(mneeContribution)) {
                                    radiance += ClampFireflyContribution(throughput,
                                                                         mneeContribution,
                                                                         clampParams);
                                }
                            }
                        }
                    }

                    if (mneeEligible && settings.enableMneeSecondary) {
                        simd::float3 chainDir = simd::normalize(bsdfSample.direction);
                        Ray chainRay;
                        chainRay.origin = OffsetRayOrigin(hit, chainDir);
                        chainRay.direction = chainDir;
                        HitInfo chainHit;
                        if (IntersectScene(sceneData, chainRay, chainHit)) {
                            bool chainHitIsLight = false;
                            if (rectLightCount > 0) {
                                RectLightHit chainLightHit;
                                if (RectLightHitInfo(rectLightIndexByRect,
                                                     rectLights,
                                                     rectangles,
                                                     rectangleCount,
                                                     envPtr,
                                                     settings,
                                                     chainHit,
                                                     chainRay.origin,
                                                     chainLightHit)) {
                                    chainHitIsLight = true;
                                }
                            }
                            if (!chainHitIsLight) {
                                uint32_t chainMatIndex = std::min(chainHit.materialIndex, materialCount - 1);
                                const auto& chainMaterial = materials[chainMatIndex];
                                if (MaterialIsDelta(chainMaterial)) {
                                    simd::float3 chainNormal = chainHit.normal;
                                    if (simd::dot(chainNormal, chainNormal) <= 0.0f) {
                                        chainNormal = {0.0f, 1.0f, 0.0f};
                                    }
                                    chainNormal = simd::normalize(chainNormal);
                                    simd::float3 chainIncident = simd::normalize(chainRay.direction);
                                    simd::float3 chainWo = -chainIncident;
                                    Rng chainRng = rng;
                                    BsdfSample chainSample = SampleBsdf(chainMaterial,
                                                                        chainHit.position,
                                                                        chainNormal,
                                                                        chainWo,
                                                                        chainIncident,
                                                                        chainHit.frontFace,
                                                                        chainRng,
                                                                        clampParams);
                                    if (chainSample.pdf > 0.0f &&
                                        chainSample.isDelta &&
                                        simd::dot(chainSample.direction, chainSample.direction) > 0.0f &&
                                        IsFinite(chainSample.weight)) {
                                        simd::float3 secondDir = simd::normalize(chainSample.direction);
                                        Ray secondRay;
                                        secondRay.origin = OffsetRayOrigin(chainHit, secondDir);
                                        secondRay.direction = secondDir;
                                        simd::float3 combinedWeight = bsdfSample.weight * chainSample.weight;
                                        float bsdfPdf = std::max(bsdfSample.pdf * chainSample.pdf, kSpecularNeePdfFloor);
                                        if (envSampling) {
                                            if (!IsOccluded(sceneData,
                                                            secondRay.origin,
                                                            secondRay.direction,
                                                            std::numeric_limits<float>::infinity())) {
                                                float envPdf = std::max(EnvironmentPdf(*envPtr,
                                                                                       settings.environmentRotation,
                                                                                       secondDir),
                                                                        kSpecularNeePdfFloor);
                                                float invEnvPdf = std::min(1.0f / envPdf, kSpecularNeeInvPdfClamp);
                                                float denom = envPdf + bsdfPdf;
                                                float misWeight = denom > 0.0f ? (envPdf / denom) : 0.0f;
                                                misWeight = std::clamp(misWeight,
                                                                       kMisWeightClampMin,
                                                                       kMisWeightClampMax);
                                                simd::float3 envColor = SampleEnvironment(*envPtr,
                                                                                          secondDir,
                                                                                          settings.environmentRotation,
                                                                                          settings.environmentIntensity);
                                                simd::float3 contribution = combinedWeight * envColor *
                                                                            (misWeight * invEnvPdf);
                                                if (IsFinite(contribution)) {
                                                    radiance += ClampFireflyContribution(throughput,
                                                                                         contribution,
                                                                                         clampParams);
                                                }
                                            }
                                        }
                                        if (rectLightCount > 0) {
                                            HitInfo lightHit;
                                            if (IntersectScene(sceneData, secondRay, lightHit)) {
                                                RectLightHit rectHit;
                                                if (RectLightHitInfo(rectLightIndexByRect,
                                                                     rectLights,
                                                                     rectangles,
                                                                     rectangleCount,
                                                                     envPtr,
                                                                     settings,
                                                                     lightHit,
                                                                     secondRay.origin,
                                                                     rectHit)) {
                                                    float lightPdf = std::max(rectHit.pdf, kSpecularNeePdfFloor);
                                                    float invLightPdf = std::min(1.0f / lightPdf, kSpecularNeeInvPdfClamp);
                                                    float denom = lightPdf + bsdfPdf;
                                                    float misWeight = denom > 0.0f ? (lightPdf / denom) : 0.0f;
                                                    misWeight = std::clamp(misWeight,
                                                                           kMisWeightClampMin,
                                                                           kMisWeightClampMax);
                                                    simd::float3 contribution = combinedWeight * rectHit.emission *
                                                                                (misWeight * invLightPdf);
                                                    if (IsFinite(contribution)) {
                                                        radiance += ClampFireflyContribution(throughput,
                                                                                             contribution,
                                                                                             clampParams);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                        throughput *= bsdfSample.weight;
                        throughput = ClampPathThroughput(throughput, clampParams);
                        if (!IsFinite(throughput)) {
                            break;
                        }
                        float maxComp = std::max({throughput.x, throughput.y, throughput.z});
                        if (maxComp <= 0.0f) {
                            break;
                        }

                        lastBsdfPdf = bsdfSample.pdf > 0.0f ? bsdfSample.pdf : lastBsdfPdf;
                        lastScatterWasDelta = bsdfSample.isDelta;

                        ray.origin = OffsetRayOrigin(hit, bsdfSample.direction);
                        ray.direction = bsdfSample.direction;

                        if (settings.enableRussianRoulette && depth >= 5) {
                            float rrProb = std::clamp(maxComp, 0.05f, 0.95f);
                            if (rng.nextFloat() > rrProb) {
                                break;
                            }
                            throughput /= rrProb;
                        }
                    }

                    pixelRadiance += radiance;
                }

                simd::float3 avg = pixelRadiance / static_cast<float>(targetSamples);
                size_t tileIndex = static_cast<size_t>(localY) * kTileSize + localX;
                tilePixels[tileIndex] = avg;
            }
        }

        for (uint32_t localY = 0; localY < tileHeight; ++localY) {
            uint32_t y = y0 + localY;
            for (uint32_t localX = 0; localX < tileWidth; ++localX) {
                size_t tileIndex = static_cast<size_t>(localY) * kTileSize + localX;
                const simd::float3 avg = tilePixels[tileIndex];
                size_t pixelIndex = static_cast<size_t>(y) * width + (x0 + localX);
                size_t base = pixelIndex * 3ull;
                linearRGB[base + 0] = avg.x;
                linearRGB[base + 1] = avg.y;
                linearRGB[base + 2] = avg.z;
            }
        }

        if (verbose) {
            uint32_t done = tilesDone.fetch_add(1, std::memory_order_relaxed) + 1u;
            uint32_t expected = nextReport.load(std::memory_order_relaxed);
            while (done >= expected) {
                if (nextReport.compare_exchange_weak(expected,
                                                     expected + reportStride,
                                                     std::memory_order_relaxed)) {
                    reportProgress(done);
                    break;
                }
            }
        }
    };

    if (workerCount == 1) {
        for (uint32_t tileY = 0; tileY < tilesY; ++tileY) {
            for (uint32_t tileX = 0; tileX < tilesX; ++tileX) {
                renderTile(tileX, tileY);
            }
        }
    } else {
        std::atomic<uint32_t> nextTile{0};
        std::vector<std::thread> workers;
        workers.reserve(workerCount);
        auto runWorker = [&]() {
            while (true) {
                uint32_t tile = nextTile.fetch_add(1, std::memory_order_relaxed);
                if (tile >= totalTiles) {
                    break;
                }
                uint32_t tileY = tile / tilesX;
                uint32_t tileX = tile % tilesX;
                renderTile(tileX, tileY);
            }
        };

        for (uint32_t i = 0; i < workerCount; ++i) {
            workers.emplace_back(runWorker);
        }
        for (auto& worker : workers) {
            worker.join();
        }
    }

    if (verbose && lastReportedPercent.load(std::memory_order_relaxed) < 100u) {
        reportProgress(totalTiles);
        std::cout << std::endl;
    }

    CFAbsoluteTime renderEnd = CFAbsoluteTimeGetCurrent();

    out.linearRGB = std::move(linearRGB);
    out.width = width;
    out.height = height;
    out.samples = targetSamples;
    out.totalSeconds = renderEnd - renderStart;
    out.avgMsPerSample = (targetSamples > 0)
                             ? ((out.totalSeconds * 1000.0) / targetSamples)
                             : 0.0;

    if (sceneData.scene) {
        rtcReleaseScene(sceneData.scene);
    }
    if (sceneData.device) {
        rtcReleaseDevice(sceneData.device);
    }

    return true;
#endif
}
