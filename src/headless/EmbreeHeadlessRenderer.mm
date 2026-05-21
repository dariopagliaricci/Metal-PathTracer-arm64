#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "headless/EmbreeHeadlessRenderer.h"
#include "renderer/EnvImportanceSampler.h"
#include "MetalShaderTypes.h"
#include <simd/simd.h>

#if defined(PATH_TRACER_ENABLE_EMBREE)
#include <embree4/rtcore.h>
#include <tbb/blocked_range2d.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#endif

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kEpsilon = 1.0e-4f;
constexpr float kSpecularNeePdfFloor = 1.0e-4f;
constexpr float kSpecularNeeInvPdfClamp = 1.0e4f;
constexpr float kMisWeightClampMin = 1.0e-4f;
constexpr float kMisWeightClampMax = 0.9999f;
constexpr uint32_t kInvalidTextureIndex = 0xFFFFFFFFu;
constexpr uint32_t kCpuRisPayloadAnalyticRectFlag = 1u << 31u;

std::string FormatEta(double secondsRemaining) {
    if (!std::isfinite(secondsRemaining) || secondsRemaining < 0.0) {
        return "--:--";
    }

    uint64_t totalSeconds = static_cast<uint64_t>(secondsRemaining + 0.5);
    uint64_t hours = totalSeconds / 3600ull;
    uint64_t minutes = (totalSeconds % 3600ull) / 60ull;
    uint64_t seconds = totalSeconds % 60ull;

    char buffer[32];
    if (hours > 0ull) {
        std::snprintf(buffer, sizeof(buffer), "%lluh %02llum %02llus",
                      static_cast<unsigned long long>(hours),
                      static_cast<unsigned long long>(minutes),
                      static_cast<unsigned long long>(seconds));
    } else {
        std::snprintf(buffer, sizeof(buffer), "%02llum %02llus",
                      static_cast<unsigned long long>(minutes),
                      static_cast<unsigned long long>(seconds));
    }
    return std::string(buffer);
}

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
    simd::float3 unnormalizedDirection{0.0f, 0.0f, -1.0f};
};

struct Rng {
    uint32_t state = 1u;

    static uint32_t Hash(uint32_t x) {
        x = x * 747796405u + 2891336453u;
        uint32_t word = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
        return (word >> 22u) ^ word;
    }

    float nextFloat() {
        state = Hash(state);
        return static_cast<float>(state) / 4294967296.0f;
    }
};

uint32_t CpuPcgHash(uint32_t state) {
    return Rng::Hash(state);
}

struct EnvironmentMap {
    struct MipLevel {
        std::vector<float> rgba;
        uint32_t width = 0;
        uint32_t height = 0;
    };
    std::vector<float> rgba;
    std::vector<MipLevel> mipLevels;
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
    const simd::float2* uvs1 = nullptr;
    const simd::float4* tangents = nullptr;
    const uint32_t* indices = nullptr;
    const simd::float4* spheres = nullptr;
    const uint32_t* materialIndices = nullptr;
    uint32_t indexCount = 0;
    uint32_t count = 0;
    uint32_t materialIndex = 0;
    uint32_t triangleOffset = 0;
};

struct HitInfo {
    simd::float3 position{0.0f, 0.0f, 0.0f};
    simd::float3 normal{0.0f, 1.0f, 0.0f};
    simd::float3 shadingNormal{0.0f, 1.0f, 0.0f};
    simd::float2 uv{0.0f, 0.0f};
    simd::float2 uv1{0.0f, 0.0f};
    simd::float4 tangent{1.0f, 0.0f, 0.0f, 1.0f};
    float uvPerWorld0 = 0.0f;
    float uvPerWorld1 = 0.0f;
    simd::float3 dPdu0{0.0f, 0.0f, 0.0f};
    simd::float3 dPdv0{0.0f, 0.0f, 0.0f};
    simd::float3 dPdu1{0.0f, 0.0f, 0.0f};
    simd::float3 dPdv1{0.0f, 0.0f, 0.0f};
    simd::float2 dUVdx0{0.0f, 0.0f};
    simd::float2 dUVdy0{0.0f, 0.0f};
    simd::float2 dUVdx1{0.0f, 0.0f};
    simd::float2 dUVdy1{0.0f, 0.0f};
    float t = 0.0f;
    uint32_t materialIndex = 0;
    bool frontFace = true;
    bool twoSided = false;
    bool hasUv = false;
    bool hasUv1 = false;
    bool hasSurfacePartials0 = false;
    bool hasSurfacePartials1 = false;
    bool hasIgehyGradients0 = false;
    bool hasIgehyGradients1 = false;
    bool hasTangent = false;
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
    bool isDisk = false;
    bool emissionUsesEnv = false;
    float area = 0.0f;
};

struct MeshLightInfo {
    uint32_t primitiveIndex = std::numeric_limits<uint32_t>::max();
    simd::float3 centroid{0.0f, 0.0f, 0.0f};
    simd::float3 edge1{0.0f, 0.0f, 0.0f};
    simd::float3 edge2{0.0f, 0.0f, 0.0f};
    simd::float3 normal{0.0f, 1.0f, 0.0f};
    simd::float3 emission{0.0f, 0.0f, 0.0f};
    float area = 0.0f;
    float power = 0.0f;
    float selectionPdf = 0.0f;
    float cumulativePower = 0.0f;
};

struct DirectionalLightInfo {
    simd::float3 direction{0.0f, 1.0f, 0.0f};
    simd::float3 emission{0.0f, 0.0f, 0.0f};
};

bool AreaPrimitiveIsDisk(const PathTracerShaderTypes::RectData& rect) {
    return rect.materialTwoSided.z ==
           static_cast<uint32_t>(PathTracerShaderTypes::AreaPrimitiveShape::Disk);
}

float AreaPrimitiveArea(const PathTracerShaderTypes::RectData& rect) {
    simd::float3 edgeU = simd_make_float3(rect.edgeU.x, rect.edgeU.y, rect.edgeU.z);
    simd::float3 edgeV = simd_make_float3(rect.edgeV.x, rect.edgeV.y, rect.edgeV.z);
    float baseArea = simd::length(simd::cross(edgeU, edgeV));
    return AreaPrimitiveIsDisk(rect) ? (kPi * baseArea) : baseArea;
}

struct BsdfEval {
    simd::float3 value{0.0f, 0.0f, 0.0f};
    float pdf = 0.0f;
    bool isDelta = false;
};

struct BsdfSample {
    simd::float3 direction{0.0f, 0.0f, 0.0f};
    simd::float3 weight{0.0f, 0.0f, 0.0f};
    float pdf = 0.0f;
    uint32_t lobeType = 0u;
    float lobeRoughness = 0.0f;
    bool isDelta = false;
};

struct CpuRisSamplePayload {
    uint32_t primitiveIndex = std::numeric_limits<uint32_t>::max();
    uint32_t flags = 0u;
    simd::float3 position{0.0f, 0.0f, 0.0f};
    simd::float3 normal{0.0f, 1.0f, 0.0f};
    simd::float3 emission{0.0f, 0.0f, 0.0f};
    float pdf = 0.0f;
};

struct CpuReservoir {
    CpuRisSamplePayload winner;
    float wSum = 0.0f;
    uint32_t M = 0u;
    float W = 0.0f;
    bool valid = false;
};

struct CpuRestirDiReservoirState {
    simd::float3 position{0.0f, 0.0f, 0.0f};
    float pdf = 0.0f;
    simd::float3 normal{0.0f, 1.0f, 0.0f};
    float targetPdf = 0.0f;
    simd::float3 emission{0.0f, 0.0f, 0.0f};
    float weight = 0.0f;
    uint32_t primitiveIndex = std::numeric_limits<uint32_t>::max();
    uint32_t flags = 0u;
    uint32_t stateFlags = 0u;
    uint32_t frameTag = 0u;
    uint32_t candidateCount = 0u;
    uint32_t temporalAccepted = 0u;
    uint32_t spatialAccepted = 0u;
    uint32_t surfacePrimitiveIndex = std::numeric_limits<uint32_t>::max();
};

struct CpuRestirDiVisibilityState {
    simd::float3 contribution{0.0f, 0.0f, 0.0f};
    float visible = 0.0f;
    uint32_t primitiveIndex = std::numeric_limits<uint32_t>::max();
    uint32_t flags = 0u;
    uint32_t stateFlags = 0u;
    uint32_t frameTag = 0u;
};

struct CpuPathReservoirState {
    bool valid = false;
    simd::float3 position{0.0f, 0.0f, 0.0f};
    simd::float3 normal{0.0f, 1.0f, 0.0f};
    simd::float3 direction{0.0f, 1.0f, 0.0f};
    simd::float3 throughput{1.0f, 1.0f, 1.0f};
    float pdf = 0.0f;
    float sampleLuma = 0.0f;
    float throughputLuma = 0.0f;
    uint32_t materialType = 0u;
    uint32_t lobeType = 0u;
    uint32_t depth = 0u;
    uint32_t frameTag = 0u;
};

struct CpuPathGuidingReservoirState {
    bool valid = false;
    simd::float3 direction{0.0f, 1.0f, 0.0f};
    float confidence = 0.0f;
    float sampleLuma = 0.0f;
    float lastLuma = 0.0f;
    float guidingPdf = 0.0f;
    bool guidedChoice = false;
    uint32_t sampleCount = 0u;
    uint32_t frameTag = 0u;
    uint32_t flags = 0u;
    uint32_t lumaAccumulator = 0u;
};

struct CpuRadianceCacheCell {
    bool valid = false;
    bool invalidated = false;
    uint32_t domainKey = 0u;
    uint32_t materialType = 0u;
    uint32_t lobeType = 0u;
    uint32_t frameTag = 0u;
    uint32_t sampleCount = 0u;
    simd::float3 position{0.0f, 0.0f, 0.0f};
    simd::float3 normal{0.0f, 1.0f, 0.0f};
    float roughness = 0.0f;
    simd::float3 radianceSum{0.0f, 0.0f, 0.0f};
    float lumaMomentSum = 0.0f;
    float storedVariance = 1.0f;
    float confidence = 0.0f;
};

struct CpuTexture {
    struct MipLevel {
        uint32_t width = 0;
        uint32_t height = 0;
        std::vector<float> rgba;
    };

    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<float> rgba;
    std::vector<MipLevel> mipLevels;
    PathTracer::MaterialTextureSamplerDesc samplerDesc{};
};

struct RayCone {
    float width = 0.0f;
    float spread = 0.0f;
};

struct PrimaryRayDiff {
    simd::float3 dOdx{0.0f, 0.0f, 0.0f};
    simd::float3 dOdy{0.0f, 0.0f, 0.0f};
    simd::float3 dDdx{0.0f, 0.0f, 0.0f};
    simd::float3 dDdy{0.0f, 0.0f, 0.0f};
};

struct EmbreeSceneData;

bool IsFinite(const simd::float3& value);

struct FireflyClampParams {
    uint32_t mode = 0u;
    float clampFactor = 0.0f;
    float clampFloor = 0.0f;
    float throughputClamp = 0.0f;
    float specularTailClampBase = 0.0f;
    float specularTailClampRoughnessScale = 0.0f;
    float minSpecularPdf = 1.0e-8f;
    float maxContribution = 0.0f;
    float enabled = 0.0f;
};

float DegreesToRadians(float degrees) {
    return degrees * (kPi / 180.0f);
}

simd::float3 SafeNormalize(simd::float3 v, simd::float3 fallback) {
    const float lenSq = simd::dot(v, v);
    if (!(lenSq > 1.0e-12f) || !std::isfinite(lenSq)) {
        return fallback;
    }
    return v / std::sqrt(lenSq);
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

    simd::float3 lookFrom{};
    simd::float3 w{};
    simd::float3 u{};
    simd::float3 v{};
    float focusDist = settings.cameraFocusDistance;

    if (settings.explicitCameraEnabled) {
        lookFrom = settings.explicitCameraPosition;
        const simd::float3 forward =
            SafeNormalize(settings.explicitCameraForward, simd_make_float3(0.0f, 0.0f, -1.0f));
        const simd::float3 upHint =
            SafeNormalize(settings.explicitCameraUp, simd_make_float3(0.0f, 1.0f, 0.0f));
        w = -forward;
        u = SafeNormalize(simd::cross(upHint, w), simd_make_float3(1.0f, 0.0f, 0.0f));
        v = SafeNormalize(simd::cross(w, u), simd_make_float3(0.0f, 1.0f, 0.0f));
        if (focusDist <= 0.0f) {
            focusDist = 1.0f;
        }
    } else {
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
        lookFrom = lookAt + offset;
        const simd::float3 vup = {0.0f, 1.0f, 0.0f};

        w = simd::normalize(lookFrom - lookAt);
        u = simd::normalize(simd::cross(vup, w));
        v = simd::cross(w, u);

        if (focusDist <= 0.0f) {
            focusDist = distance;
        }
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
                      Rng& rng,
                      bool deterministic = false) {
    float u = (static_cast<float>(x) + (deterministic ? 0.5f : rng.nextFloat())) / static_cast<float>(width);
    float v = (static_cast<float>(y) + (deterministic ? 0.5f : rng.nextFloat())) / static_cast<float>(height);
    v = 1.0f - v;

    simd::float3 direction = camera.lowerLeft + u * camera.horizontal + v * camera.vertical - camera.origin;
    simd::float3 origin = camera.origin;
    if (!deterministic && camera.lensRadius > 0.0f) {
        simd::float3 disk = SampleInUnitDisk(rng) * camera.lensRadius;
        simd::float3 offset = camera.u * disk.x + camera.v * disk.y;
        origin += offset;
        direction -= offset;
    }

    return Ray{origin, simd::normalize(direction), direction};
}

RayCone MakePrimaryRayCone(const CameraBasis& camera, uint32_t width, uint32_t height) {
    float pixelWorldX = simd::length(camera.horizontal) / std::max(static_cast<float>(width), 1.0f);
    float pixelWorldY = simd::length(camera.vertical) / std::max(static_cast<float>(height), 1.0f);
    float pixelFootprint = std::max(std::max(pixelWorldX, pixelWorldY), 1.0e-6f);
    simd::float3 viewportCenter = camera.lowerLeft + 0.5f * camera.horizontal + 0.5f * camera.vertical;
    float focusDistance = simd::length(viewportCenter - camera.origin);
    RayCone cone;
    cone.width = std::max(2.0f * camera.lensRadius, 0.0f);
    cone.spread = pixelFootprint / std::max(focusDistance, 1.0e-6f);
    return cone;
}

PrimaryRayDiff MakePrimaryRayDiff(const CameraBasis& camera, uint32_t width, uint32_t height) {
    PrimaryRayDiff diff;
    diff.dDdx = camera.horizontal / std::max(static_cast<float>(width), 1.0f);
    diff.dDdy = -camera.vertical / std::max(static_cast<float>(height), 1.0f);
    return diff;
}

float RayConeWidthAtDistance(const RayCone& cone, float distanceWorld) {
    return std::max(cone.width + cone.spread * std::max(distanceWorld, 0.0f), 1.0e-7f);
}

float BsdfConeSpreadIncrement(uint32_t lobeType, float roughness, bool isDelta) {
    if (isDelta) {
        return 0.0f;
    }
    float clampedRoughness = std::clamp(roughness, 0.0f, 1.0f);
    if (lobeType == 0u) {
        return 0.55f;
    }
    if (lobeType == 1u) {
        return 0.03f * (1.0f - clampedRoughness) + 0.45f * clampedRoughness;
    }
    return 0.10f * (1.0f - clampedRoughness) + 0.60f * clampedRoughness;
}

float SurfaceFootprintFromCone(float coneFootprintWorld,
                               const simd::float3& surfaceNormal,
                               const simd::float3& viewDir) {
    simd::float3 n = SafeNormalize(surfaceNormal, simd_make_float3(0.0f, 1.0f, 0.0f));
    simd::float3 v = SafeNormalize(viewDir, simd_make_float3(0.0f, 0.0f, 1.0f));
    float cosTheta = std::fabs(simd::dot(n, v));
    return coneFootprintWorld / std::max(cosTheta, 1.0e-3f);
}

float PrimarySurfaceFootprint(const RayCone& cone, const HitInfo& hit, const Ray& ray) {
    float coneFootprintWorld = RayConeWidthAtDistance(cone, hit.t);
    return SurfaceFootprintFromCone(coneFootprintWorld, hit.normal, -ray.direction);
}

bool ComputeTriangleSurfacePartials(const simd::float3& p0,
                                    const simd::float3& p1,
                                    const simd::float3& p2,
                                    const simd::float2& uv0,
                                    const simd::float2& uv1,
                                    const simd::float2& uv2,
                                    simd::float3& dPduOut,
                                    simd::float3& dPdvOut,
                                    float& uvPerWorldOut) {
    dPduOut = simd_make_float3(0.0f, 0.0f, 0.0f);
    dPdvOut = simd_make_float3(0.0f, 0.0f, 0.0f);
    uvPerWorldOut = 0.0f;
    simd::float3 edge1 = p1 - p0;
    simd::float3 edge2 = p2 - p0;
    simd::float2 dUV1 = uv1 - uv0;
    simd::float2 dUV2 = uv2 - uv0;
    float det = dUV1.x * dUV2.y - dUV1.y * dUV2.x;
    if (std::fabs(det) > 1.0e-9f) {
        float invDet = 1.0f / det;
        simd::float3 dPdu = (edge1 * dUV2.y - edge2 * dUV1.y) * invDet;
        simd::float3 dPdv = (edge2 * dUV1.x - edge1 * dUV2.x) * invDet;
        float lenU = simd::length(dPdu);
        float lenV = simd::length(dPdv);
        if (lenU > 1.0e-8f && lenV > 1.0e-8f && IsFinite(dPdu) && IsFinite(dPdv)) {
            dPduOut = dPdu;
            dPdvOut = dPdv;
            uvPerWorldOut = std::max(1.0f / lenU, 1.0f / lenV);
            return std::isfinite(uvPerWorldOut) && uvPerWorldOut > 0.0f;
        }
    }

    float worldArea = simd::length(simd::cross(edge1, edge2));
    float uvArea = std::fabs(det);
    if (worldArea > 1.0e-12f && uvArea > 1.0e-12f) {
        float uvPerWorldFallback = std::sqrt(uvArea / worldArea);
        simd::float3 surfaceNormal = SafeNormalize(simd::cross(edge1, edge2),
                                                   simd_make_float3(0.0f, 1.0f, 0.0f));
        simd::float3 tangentFallback = SafeNormalize(edge1, simd_make_float3(1.0f, 0.0f, 0.0f));
        simd::float3 bitangentFallback = SafeNormalize(simd::cross(surfaceNormal, tangentFallback),
                                                       simd_make_float3(0.0f, 1.0f, 0.0f));
        if (!IsFinite(tangentFallback) || !IsFinite(bitangentFallback)) {
            return false;
        }
        dPduOut = tangentFallback / std::max(uvPerWorldFallback, 1.0e-8f);
        dPdvOut = bitangentFallback / std::max(uvPerWorldFallback, 1.0e-8f);
        uvPerWorldOut = uvPerWorldFallback;
        return std::isfinite(uvPerWorldOut) && uvPerWorldOut > 0.0f;
    }
    return false;
}

float ComputeTriangleUvPerWorld(const simd::float3& p0,
                                const simd::float3& p1,
                                const simd::float3& p2,
                                const simd::float2& uv0,
                                const simd::float2& uv1,
                                const simd::float2& uv2) {
    simd::float3 dPdu;
    simd::float3 dPdv;
    float uvPerWorld = 0.0f;
    if (ComputeTriangleSurfacePartials(p0, p1, p2, uv0, uv1, uv2, dPdu, dPdv, uvPerWorld)) {
        return uvPerWorld;
    }

    simd::float3 edge1 = p1 - p0;
    simd::float3 edge2 = p2 - p0;
    simd::float2 dUV1 = uv1 - uv0;
    simd::float2 dUV2 = uv2 - uv0;
    float det = dUV1.x * dUV2.y - dUV1.y * dUV2.x;
    if (std::fabs(det) > 1.0e-9f) {
        float invDet = 1.0f / det;
        simd::float3 dPdu = (edge1 * dUV2.y - edge2 * dUV1.y) * invDet;
        simd::float3 dPdv = (edge2 * dUV1.x - edge1 * dUV2.x) * invDet;
        float lenU = simd::length(dPdu);
        float lenV = simd::length(dPdv);
        if (lenU > 1.0e-8f && lenV > 1.0e-8f) {
            return std::max(1.0f / lenU, 1.0f / lenV);
        }
    }

    float worldArea = simd::length(simd::cross(edge1, edge2));
    float uvArea = std::fabs(det);
    if (worldArea > 1.0e-12f && uvArea > 1.0e-12f) {
        return std::sqrt(uvArea / worldArea);
    }
    return 0.0f;
}

bool UvWorldGradientsFromPartials(const simd::float3& dPdu,
                                  const simd::float3& dPdv,
                                  simd::float3& dudP,
                                  simd::float3& dvdP) {
    float a00 = simd::dot(dPdu, dPdu);
    float a01 = simd::dot(dPdu, dPdv);
    float a11 = simd::dot(dPdv, dPdv);
    float det = a00 * a11 - a01 * a01;
    if (std::fabs(det) <= 1.0e-12f) {
        return false;
    }
    dudP = (a11 * dPdu - a01 * dPdv) / det;
    dvdP = (a00 * dPdv - a01 * dPdu) / det;
    return IsFinite(dudP) && IsFinite(dvdP);
}

bool FirstHitUvGradientsIgehy(const Ray& ray,
                              const PrimaryRayDiff& rayDiff,
                              float tHit,
                              const simd::float3& geometricNormal,
                              const simd::float3& dudP,
                              const simd::float3& dvdP,
                              simd::float2& dUVdx,
                              simd::float2& dUVdy) {
    dUVdx = simd_make_float2(0.0f, 0.0f);
    dUVdy = simd_make_float2(0.0f, 0.0f);
    simd::float3 n = SafeNormalize(geometricNormal, simd_make_float3(0.0f, 1.0f, 0.0f));
    if (!IsFinite(n) || simd::dot(n, n) <= 1.0e-12f) {
        return false;
    }
    simd::float3 direction = ray.unnormalizedDirection;
    float dirLen = simd::length(direction);
    if (!std::isfinite(dirLen) || dirLen <= 1.0e-8f) {
        direction = ray.direction;
        dirLen = 1.0f;
    }
    float tParam = tHit / std::max(dirLen, 1.0e-8f);
    float denom = simd::dot(n, direction);
    if (!std::isfinite(denom) || std::fabs(denom) < 1.0e-6f) {
        return false;
    }
    simd::float3 dtdxTerm = rayDiff.dOdx + tParam * rayDiff.dDdx;
    simd::float3 dtdyTerm = rayDiff.dOdy + tParam * rayDiff.dDdy;
    if (!IsFinite(dtdxTerm) || !IsFinite(dtdyTerm)) {
        return false;
    }
    float dtdx = -simd::dot(n, dtdxTerm) / denom;
    float dtdy = -simd::dot(n, dtdyTerm) / denom;
    if (!std::isfinite(dtdx) || !std::isfinite(dtdy)) {
        return false;
    }
    simd::float3 dPdx = dtdxTerm + dtdx * direction;
    simd::float3 dPdy = dtdyTerm + dtdy * direction;
    if (!IsFinite(dPdx) || !IsFinite(dPdy)) {
        return false;
    }
    dUVdx = simd_make_float2(simd::dot(dudP, dPdx), simd::dot(dvdP, dPdx));
    dUVdy = simd_make_float2(simd::dot(dudP, dPdy), simd::dot(dvdP, dPdy));
    return std::isfinite(dUVdx.x) && std::isfinite(dUVdx.y) &&
           std::isfinite(dUVdy.x) && std::isfinite(dUVdy.y);
}

void PopulateFirstHitUvGradients(HitInfo& hit, const Ray& ray, const PrimaryRayDiff& rayDiff) {
    hit.hasIgehyGradients0 = false;
    hit.hasIgehyGradients1 = false;
    hit.dUVdx0 = simd_make_float2(0.0f, 0.0f);
    hit.dUVdy0 = simd_make_float2(0.0f, 0.0f);
    hit.dUVdx1 = simd_make_float2(0.0f, 0.0f);
    hit.dUVdy1 = simd_make_float2(0.0f, 0.0f);
    if (hit.hasSurfacePartials0) {
        simd::float3 dudP;
        simd::float3 dvdP;
        if (UvWorldGradientsFromPartials(hit.dPdu0, hit.dPdv0, dudP, dvdP)) {
            hit.hasIgehyGradients0 = FirstHitUvGradientsIgehy(ray,
                                                              rayDiff,
                                                              hit.t,
                                                              hit.normal,
                                                              dudP,
                                                              dvdP,
                                                              hit.dUVdx0,
                                                              hit.dUVdy0);
        }
    }
    if (hit.hasSurfacePartials1) {
        simd::float3 dudP;
        simd::float3 dvdP;
        if (UvWorldGradientsFromPartials(hit.dPdu1, hit.dPdv1, dudP, dvdP)) {
            hit.hasIgehyGradients1 = FirstHitUvGradientsIgehy(ray,
                                                              rayDiff,
                                                              hit.t,
                                                              hit.normal,
                                                              dudP,
                                                              dvdP,
                                                              hit.dUVdx1,
                                                              hit.dUVdy1);
        }
    }
}

simd::float3 SkyColor(const simd::float3& direction) {
    simd::float3 unit = simd::normalize(direction);
    float t = 0.5f * (unit.y + 1.0f);
    return simd_make_float3(1.0f, 1.0f, 1.0f) * (1.0f - t) +
           simd_make_float3(0.5f, 0.7f, 1.0f) * t;
}

simd::float3 SampleEnvironmentData(const std::vector<float>& rgba,
                                   uint32_t width,
                                   uint32_t height,
                                   const simd::float3& direction,
                                   float rotation,
                                   float intensity) {
    if (width == 0 || height == 0 || rgba.empty()) {
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

    u = std::clamp(u, 0.0f, 0.99999994f);
    v = std::clamp(v, 0.0f, 0.99999994f);
    float fx = u * static_cast<float>(width) - 0.5f;
    float fy = v * static_cast<float>(height) - 0.5f;
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

    x0 = wrap(x0, static_cast<int>(width));
    x1 = wrap(x1, static_cast<int>(width));
    y0 = std::clamp(y0, 0, static_cast<int>(height) - 1);
    y1 = std::clamp(y1, 0, static_cast<int>(height) - 1);

    auto fetch = [&](int px, int py) {
        size_t index = (static_cast<size_t>(py) * width + static_cast<size_t>(px)) * 4ull;
        return simd_make_float3(rgba[index + 0], rgba[index + 1], rgba[index + 2]);
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

simd::float3 SampleEnvironmentLod(const EnvironmentMap& env,
                                  const simd::float3& direction,
                                  float rotation,
                                  float intensity,
                                  float lod) {
    if (env.mipLevels.empty()) {
        return SampleEnvironmentData(env.rgba, env.width, env.height, direction, rotation, intensity);
    }
    float maxMip = static_cast<float>(env.mipLevels.size() - 1u);
    float clampedLod = std::clamp(std::isfinite(lod) ? lod : 0.0f, 0.0f, maxMip);
    uint32_t level0 = static_cast<uint32_t>(std::floor(clampedLod));
    uint32_t level1 = std::min<uint32_t>(level0 + 1u, static_cast<uint32_t>(env.mipLevels.size() - 1u));
    float t = clampedLod - static_cast<float>(level0);
    const auto& mip0 = env.mipLevels[level0];
    const auto& mip1 = env.mipLevels[level1];
    simd::float3 c0 = SampleEnvironmentData(mip0.rgba, mip0.width, mip0.height, direction, rotation, intensity);
    simd::float3 c1 = SampleEnvironmentData(mip1.rgba, mip1.width, mip1.height, direction, rotation, intensity);
    return c0 * (1.0f - t) + c1 * t;
}

simd::float3 SampleEnvironment(const EnvironmentMap& env,
                               const simd::float3& direction,
                               float rotation,
                               float intensity) {
    return SampleEnvironmentLod(env, direction, rotation, intensity, 0.0f);
}

float EnvironmentMaxMip(const EnvironmentMap& env) {
    return env.mipLevels.empty() ? 0.0f : static_cast<float>(env.mipLevels.size() - 1u);
}

float EnvironmentLodFromRoughness(float roughness, const EnvironmentMap& env) {
    const float maxMip = EnvironmentMaxMip(env);
    if (maxMip <= 0.0f) {
        return 0.0f;
    }
    float alpha = std::clamp(roughness, 0.0f, 1.0f);
    alpha *= alpha;
    return std::clamp(alpha * maxMip, 0.0f, maxMip);
}

simd::float3 EvaluateBackground(const PathTracer::RenderSettings& settings,
                                const EnvironmentMap* env,
                                const simd::float3& direction,
                                bool lodActive = false,
                                float lod = 0.0f) {
    switch (settings.backgroundMode) {
        case PathTracer::RenderSettings::BackgroundMode::Solid:
            return settings.backgroundColor;
        case PathTracer::RenderSettings::BackgroundMode::Environment:
            if (env) {
                if (lodActive && EnvironmentMaxMip(*env) > 0.0f) {
                    return SampleEnvironmentLod(*env,
                                                direction,
                                                settings.environmentRotation,
                                                settings.environmentIntensity,
                                                lod);
                }
                return SampleEnvironment(*env,
                                         direction,
                                         settings.environmentRotation,
                                         settings.environmentIntensity);
            }
            return SkyColor(direction);
        case PathTracer::RenderSettings::BackgroundMode::Gradient:
        default:
            return SkyColor(direction);
    }
}

float SrgbToLinear(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    if (value <= 0.04045f) {
        return value / 12.92f;
    }
    return std::pow((value + 0.055f) / 1.055f, 2.4f);
}

bool ReadMaterialTexture(id<MTLTexture> texture, CpuTexture& outTexture) {
    outTexture = CpuTexture{};
    if (!texture) {
        return false;
    }

    const NSUInteger width = texture.width;
    const NSUInteger height = texture.height;
    if (width == 0 || height == 0) {
        return false;
    }

    outTexture.width = static_cast<uint32_t>(width);
    outTexture.height = static_cast<uint32_t>(height);

    const MTLPixelFormat format = texture.pixelFormat;
    const NSUInteger mipCount = std::max<NSUInteger>(texture.mipmapLevelCount, 1u);
    outTexture.mipLevels.resize(mipCount);

    auto readLevel = [&](NSUInteger level, CpuTexture::MipLevel& outLevel) -> bool {
        const NSUInteger levelWidth = std::max<NSUInteger>(texture.width >> level, 1u);
        const NSUInteger levelHeight = std::max<NSUInteger>(texture.height >> level, 1u);
        outLevel.width = static_cast<uint32_t>(levelWidth);
        outLevel.height = static_cast<uint32_t>(levelHeight);
        outLevel.rgba.resize(static_cast<size_t>(levelWidth) * static_cast<size_t>(levelHeight) * 4ull, 0.0f);
        const MTLRegion region = MTLRegionMake2D(0, 0, levelWidth, levelHeight);

        if (format == MTLPixelFormatRGBA8Unorm ||
            format == MTLPixelFormatRGBA8Unorm_sRGB ||
            format == MTLPixelFormatBGRA8Unorm ||
            format == MTLPixelFormatBGRA8Unorm_sRGB) {
            std::vector<uint8_t> bytes(static_cast<size_t>(levelWidth) * static_cast<size_t>(levelHeight) * 4ull, 0u);
            [texture getBytes:bytes.data()
                  bytesPerRow:levelWidth * 4u
                   fromRegion:region
                  mipmapLevel:level];

            const bool isBgra = (format == MTLPixelFormatBGRA8Unorm ||
                                 format == MTLPixelFormatBGRA8Unorm_sRGB);
            const bool isSrgb = (format == MTLPixelFormatRGBA8Unorm_sRGB ||
                                 format == MTLPixelFormatBGRA8Unorm_sRGB);
            for (size_t i = 0; i < static_cast<size_t>(levelWidth) * static_cast<size_t>(levelHeight); ++i) {
                size_t src = i * 4ull;
                float r = bytes[src + (isBgra ? 2u : 0u)] / 255.0f;
                float g = bytes[src + 1u] / 255.0f;
                float b = bytes[src + (isBgra ? 0u : 2u)] / 255.0f;
                float a = bytes[src + 3u] / 255.0f;
                if (isSrgb) {
                    r = SrgbToLinear(r);
                    g = SrgbToLinear(g);
                    b = SrgbToLinear(b);
                }
                size_t dst = i * 4ull;
                outLevel.rgba[dst + 0u] = r;
                outLevel.rgba[dst + 1u] = g;
                outLevel.rgba[dst + 2u] = b;
                outLevel.rgba[dst + 3u] = a;
            }
            return true;
        }

        if (format == MTLPixelFormatRGBA32Float) {
            [texture getBytes:outLevel.rgba.data()
                  bytesPerRow:levelWidth * sizeof(float) * 4u
                   fromRegion:region
                  mipmapLevel:level];
            return true;
        }

        return false;
    };

    for (NSUInteger level = 0; level < mipCount; ++level) {
        if (!readLevel(level, outTexture.mipLevels[level])) {
            outTexture = CpuTexture{};
            return false;
        }
    }
    outTexture.rgba = outTexture.mipLevels.front().rgba;
    return true;
}

simd::float2 ApplyTextureTransform(const simd::float2& uv,
                                   const simd::float4& row0,
                                   const simd::float4& row1) {
    return simd_make_float2(row0.x * uv.x + row0.y * uv.y + row0.z,
                            row1.x * uv.x + row1.y * uv.y + row1.z);
}

simd::float2 ApplyTextureGradientTransform(const simd::float2& gradient,
                                           const simd::float4& row0,
                                           const simd::float4& row1) {
    return simd_make_float2(row0.x * gradient.x + row0.y * gradient.y,
                            row1.x * gradient.x + row1.y * gradient.y);
}

float ApplyTextureUvPerWorldTransform(float uvPerWorld,
                                      const simd::float4& row0,
                                      const simd::float4& row1) {
    float sx = simd::length(simd_make_float2(row0.x, row1.x));
    float sy = simd::length(simd_make_float2(row0.y, row1.y));
    float scaleBound = std::max(std::max(sx, sy), 1.0e-6f);
    return uvPerWorld * scaleBound;
}

float Smoothstep(float edge0, float edge1, float x) {
    float t = std::clamp((x - edge0) / std::max(edge1 - edge0, 1.0e-8f), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

float ApplyWrapMode(float value, int32_t wrapMode) {
    constexpr int32_t kGltfWrapClampToEdge = 33071;
    constexpr int32_t kGltfWrapMirroredRepeat = 33648;
    constexpr int32_t kGltfWrapRepeat = 10497;
    switch (wrapMode) {
        case kGltfWrapClampToEdge:
            return std::clamp(value, 0.0f, 1.0f);
        case kGltfWrapMirroredRepeat: {
            float period = std::fmod(value, 2.0f);
            if (period < 0.0f) {
                period += 2.0f;
            }
            return period <= 1.0f ? period : (2.0f - period);
        }
        case kGltfWrapRepeat:
        default: {
            float f = value - std::floor(value);
            return f < 0.0f ? (f + 1.0f) : f;
        }
    }
}

simd::float4 SampleCpuTextureData(const std::vector<float>& rgba,
                                  uint32_t width,
                                  uint32_t height,
                                  const PathTracer::MaterialTextureSamplerDesc& samplerDesc,
                                  const simd::float2& uv) {
    if (width == 0 || height == 0 || rgba.empty()) {
        return simd_make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    }

    float u = ApplyWrapMode(uv.x, samplerDesc.wrapS);
    float v = ApplyWrapMode(uv.y, samplerDesc.wrapT);
    float fx = u * static_cast<float>(width) - 0.5f;
    float fy = v * static_cast<float>(height) - 0.5f;
    int x0 = static_cast<int>(std::floor(fx));
    int y0 = static_cast<int>(std::floor(fy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float tx = fx - static_cast<float>(x0);
    float ty = fy - static_cast<float>(y0);

    auto wrapIndex = [](int value, int max) {
        int m = value % max;
        return m < 0 ? m + max : m;
    };

    x0 = wrapIndex(x0, static_cast<int>(width));
    x1 = wrapIndex(x1, static_cast<int>(width));
    y0 = wrapIndex(y0, static_cast<int>(height));
    y1 = wrapIndex(y1, static_cast<int>(height));

    auto fetch = [&](int px, int py) {
        size_t idx = (static_cast<size_t>(py) * width + static_cast<size_t>(px)) * 4ull;
        return simd_make_float4(rgba[idx + 0u],
                                rgba[idx + 1u],
                                rgba[idx + 2u],
                                rgba[idx + 3u]);
    };

    simd::float4 c00 = fetch(x0, y0);
    simd::float4 c10 = fetch(x1, y0);
    simd::float4 c01 = fetch(x0, y1);
    simd::float4 c11 = fetch(x1, y1);
    simd::float4 c0 = c00 * (1.0f - tx) + c10 * tx;
    simd::float4 c1 = c01 * (1.0f - tx) + c11 * tx;
    return c0 * (1.0f - ty) + c1 * ty;
}

simd::float4 SampleCpuTexture(const CpuTexture& texture, const simd::float2& uv) {
    return SampleCpuTextureData(texture.rgba,
                                texture.width,
                                texture.height,
                                texture.samplerDesc,
                                uv);
}

simd::float4 SampleCpuTextureLevel(const CpuTexture& texture, const simd::float2& uv, uint32_t levelIndex) {
    if (texture.mipLevels.empty()) {
        return SampleCpuTexture(texture, uv);
    }
    levelIndex = std::min<uint32_t>(levelIndex, static_cast<uint32_t>(texture.mipLevels.size() - 1u));
    const auto& level = texture.mipLevels[levelIndex];
    return SampleCpuTextureData(level.rgba,
                                level.width,
                                level.height,
                                texture.samplerDesc,
                                uv);
}

simd::float4 SampleCpuTexture(const CpuTexture& texture, const simd::float2& uv, float lod) {
    if (texture.mipLevels.empty()) {
        return SampleCpuTexture(texture, uv);
    }
    if (texture.mipLevels.size() == 1u) {
        return SampleCpuTextureLevel(texture, uv, 0u);
    }
    float clampedLod = std::clamp(lod, 0.0f, static_cast<float>(texture.mipLevels.size() - 1u));
    uint32_t level0 = static_cast<uint32_t>(std::floor(clampedLod));
    uint32_t level1 = std::min<uint32_t>(level0 + 1u, static_cast<uint32_t>(texture.mipLevels.size() - 1u));
    float t = clampedLod - static_cast<float>(level0);
    simd::float4 c0 = SampleCpuTextureLevel(texture, uv, level0);
    simd::float4 c1 = SampleCpuTextureLevel(texture, uv, level1);
    return c0 * (1.0f - t) + c1 * t;
}

simd::float4 SampleCpuTextureAnisotropic(const CpuTexture& texture,
                                         const simd::float2& uv,
                                         const simd::float2& dUVdx,
                                         const simd::float2& dUVdy,
                                         float fallbackLod) {
    if (texture.mipLevels.size() <= 1u || texture.width == 0u || texture.height == 0u) {
        return SampleCpuTexture(texture, uv, fallbackLod);
    }
    if (!std::isfinite(dUVdx.x) || !std::isfinite(dUVdx.y) ||
        !std::isfinite(dUVdy.x) || !std::isfinite(dUVdy.y)) {
        return SampleCpuTexture(texture, uv, fallbackLod);
    }
    simd::float2 texelDx = simd_make_float2(dUVdx.x * static_cast<float>(texture.width),
                                            dUVdx.y * static_cast<float>(texture.height));
    simd::float2 texelDy = simd_make_float2(dUVdy.x * static_cast<float>(texture.width),
                                            dUVdy.y * static_cast<float>(texture.height));
    const float lenX = simd::length(texelDx);
    const float lenY = simd::length(texelDy);
    if (!std::isfinite(lenX) || !std::isfinite(lenY) || std::max(lenX, lenY) <= 1.0f) {
        return SampleCpuTexture(texture, uv, fallbackLod);
    }
    const bool xMajor = lenX >= lenY;
    const float majorLen = std::max(xMajor ? lenX : lenY, 1.0f);
    const float minorLen = std::max(xMajor ? lenY : lenX, 1.0f);
    const simd::float2 majorUv = xMajor ? dUVdx : dUVdy;
    const float minorLod =
        std::clamp(std::log2(std::max(minorLen, 1.0e-8f)),
                   0.0f,
                   static_cast<float>(texture.mipLevels.size() - 1u));
    const uint32_t taps =
        std::clamp(static_cast<uint32_t>(std::ceil(majorLen / minorLen)), 1u, 8u);
    if (taps <= 1u) {
        return SampleCpuTexture(texture, uv, minorLod);
    }
    simd::float4 accum = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint32_t i = 0; i < taps; ++i) {
        float offset = (static_cast<float>(i) + 0.5f) / static_cast<float>(taps) - 0.5f;
        accum += SampleCpuTexture(texture, uv + majorUv * offset, minorLod);
    }
    return accum / static_cast<float>(taps);
}

simd::float3 DecodeNormalMap(const simd::float3& encoded, float normalScale) {
    simd::float3 n = encoded * 2.0f - simd_make_float3(1.0f, 1.0f, 1.0f);
    n.x *= normalScale;
    n.y *= normalScale;
    float xy2 = n.x * n.x + n.y * n.y;
    n.z = std::sqrt(std::max(1.0f - std::min(xy2, 1.0f), 0.0f));
    return simd::normalize(n);
}

float TextureLodFromFootprint(const CpuTexture& texture, float uvPerWorld, float footprintWorld) {
    if (texture.mipLevels.size() <= 1u || uvPerWorld <= 0.0f || footprintWorld <= 0.0f) {
        return 0.0f;
    }
    float maxRes = std::max(static_cast<float>(texture.width), static_cast<float>(texture.height));
    float texelFootprint = footprintWorld * uvPerWorld * maxRes;
    float lod = std::log2(std::max(texelFootprint, 1.0e-7f));
    return std::clamp(lod, 0.0f, static_cast<float>(texture.mipLevels.size() - 1u));
}

bool TextureLodFromGradients(const CpuTexture& texture,
                             const simd::float2& dUVdx,
                             const simd::float2& dUVdy,
                             float& outLod) {
    outLod = 0.0f;
    if (texture.width == 0u || texture.height == 0u || texture.mipLevels.size() <= 1u) {
        return false;
    }
    if (!std::isfinite(dUVdx.x) || !std::isfinite(dUVdx.y) ||
        !std::isfinite(dUVdy.x) || !std::isfinite(dUVdy.y)) {
        return false;
    }
    float width = static_cast<float>(texture.width);
    float height = static_cast<float>(texture.height);
    float rho = std::max(std::max(std::fabs(dUVdx.x) * width,
                                  std::fabs(dUVdx.y) * height),
                         std::max(std::fabs(dUVdy.x) * width,
                                  std::fabs(dUVdy.y) * height));
    if (!std::isfinite(rho) || rho <= 0.0f) {
        return false;
    }
    float lod = std::log2(std::max(rho, 1.0e-8f));
    if (!std::isfinite(lod)) {
        return false;
    }
    outLod = std::clamp(lod, 0.0f, static_cast<float>(texture.mipLevels.size() - 1u));
    return std::isfinite(outLod);
}

float TextureLodWithFallback(const CpuTexture& texture,
                             bool hasIgehyGradients,
                             const simd::float2& dUVdx,
                             const simd::float2& dUVdy,
                             float uvPerWorld,
                             float surfaceFootprintWorld) {
    float lod = 0.0f;
    bool gradientOk = false;
    if (hasIgehyGradients) {
        gradientOk = TextureLodFromGradients(texture, dUVdx, dUVdy, lod);
    }
    if (!gradientOk) {
        lod = TextureLodFromFootprint(texture, uvPerWorld, surfaceFootprintWorld);
    }
    if (!std::isfinite(lod)) {
        lod = TextureLodFromFootprint(texture, uvPerWorld, surfaceFootprintWorld);
    }
    return std::max(lod, 0.0f);
}

struct CpuTextureSamplingContext {
    simd::float2 uv{0.0f, 0.0f};
    simd::float2 dUVdx{0.0f, 0.0f};
    simd::float2 dUVdy{0.0f, 0.0f};
    float uvPerWorld = 0.0f;
    bool hasIgehyGradients = false;
};

CpuTextureSamplingContext MakeCpuTextureSamplingContext(const HitInfo& hit,
                                                        uint32_t uvSet,
                                                        const simd::float4& row0,
                                                        const simd::float4& row1) {
    const bool useUv1 = uvSet == 1u && hit.hasUv1;
    simd::float2 uv = useUv1 ? hit.uv1 : hit.uv;
    CpuTextureSamplingContext ctx;
    ctx.uv = ApplyTextureTransform(uv, row0, row1);
    ctx.uvPerWorld = ApplyTextureUvPerWorldTransform(useUv1 ? hit.uvPerWorld1 : hit.uvPerWorld0,
                                                     row0,
                                                     row1);
    ctx.hasIgehyGradients = useUv1 ? hit.hasIgehyGradients1 : hit.hasIgehyGradients0;
    if (ctx.hasIgehyGradients) {
        ctx.dUVdx = ApplyTextureGradientTransform(useUv1 ? hit.dUVdx1 : hit.dUVdx0, row0, row1);
        ctx.dUVdy = ApplyTextureGradientTransform(useUv1 ? hit.dUVdy1 : hit.dUVdy0, row0, row1);
        if (!std::isfinite(ctx.dUVdx.x) || !std::isfinite(ctx.dUVdx.y) ||
            !std::isfinite(ctx.dUVdy.x) || !std::isfinite(ctx.dUVdy.y)) {
            ctx.hasIgehyGradients = false;
            ctx.dUVdx = simd_make_float2(0.0f, 0.0f);
            ctx.dUVdy = simd_make_float2(0.0f, 0.0f);
        }
    }
    return ctx;
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
    params.mode = static_cast<uint32_t>(settings.fireflyClampMode);
    params.clampFactor = std::max(settings.fireflyClampFactor, 0.0f);
    params.clampFloor = std::max(settings.fireflyClampFloor, 0.0f);
    params.throughputClamp = std::max(settings.throughputClamp, 0.0f);
    params.specularTailClampBase = std::max(settings.specularTailClampBase, 0.0f);
    params.specularTailClampRoughnessScale = std::max(settings.specularTailClampRoughnessScale, 0.0f);
    params.minSpecularPdf = std::max(settings.minSpecularPdf, 1.0e-8f);
    params.maxContribution = std::max(settings.fireflyClampMaxContribution, 0.0f);
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

float LoadEmbreeSssTravelScale() {
    const char* value = std::getenv("PATH_TRACER_EMBREE_SSS_TRAVEL_SCALE");
    if (!value || *value == '\0') {
        return 0.55f;
    }
    char* end = nullptr;
    float parsed = std::strtof(value, &end);
    if (end == value || !std::isfinite(parsed) || parsed < 0.0f) {
        return 0.55f;
    }
    return parsed;
}

simd::float3 ClampFireflyContribution(const simd::float3& throughput,
                                      const simd::float3& contribution,
                                      const FireflyClampParams& params,
                                      bool* outClamped = nullptr,
                                      bool* outInvalid = nullptr) {
    if (outClamped) {
        *outClamped = false;
    }
    if (outInvalid) {
        *outInvalid = false;
    }
    simd::float3 combined = throughput * contribution;
    if (!IsFinite(combined)) {
        if (outInvalid) {
            *outInvalid = true;
        }
        return simd_make_float3(0.0f, 0.0f, 0.0f);
    }

    simd::float3 positive = simd::max(combined, simd_make_float3(0.0f, 0.0f, 0.0f));
    if (params.enabled < 0.5f) {
        return positive;
    }

    if (params.mode == 1u) {
        float maxComponent = std::max({positive.x, positive.y, positive.z});
        float maxAllowed = params.maxContribution;
        if (maxAllowed <= 0.0f) {
            float throughputMax = std::max({throughput.x, throughput.y, throughput.z});
            maxAllowed = std::max(throughputMax * params.clampFactor, params.clampFloor);
        }
        if (maxComponent > maxAllowed && maxComponent > 0.0f) {
            float scale = maxAllowed / std::max(maxComponent, 1.0e-6f);
            combined *= scale;
            positive = simd::max(combined, simd_make_float3(0.0f, 0.0f, 0.0f));
            if (outClamped) {
                *outClamped = true;
            }
        }
    } else {
        float lum = Luminance(positive);
        simd::float3 throughputPositive = simd::max(throughput, simd_make_float3(0.0f, 0.0f, 0.0f));
        float throughputLum = Luminance(throughputPositive);
        float maxLum = std::max(throughputLum * params.clampFactor, params.clampFloor);
        if (params.maxContribution > 0.0f) {
            maxLum = std::max(maxLum, params.maxContribution);
        }

        if (lum > maxLum && lum > 0.0f) {
            float scale = maxLum / std::max(lum, 1.0e-6f);
            combined *= scale;
            positive = simd::max(combined, simd_make_float3(0.0f, 0.0f, 0.0f));
            if (outClamped) {
                *outClamped = true;
            }
        }
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
                                 const FireflyClampParams& params,
                                 bool* outClamped = nullptr,
                                 bool* outInvalid = nullptr) {
    if (outClamped) {
        *outClamped = false;
    }
    if (outInvalid) {
        *outInvalid = false;
    }
    if (!IsFinite(throughput)) {
        if (outInvalid) {
            *outInvalid = true;
        }
        return simd_make_float3(0.0f, 0.0f, 0.0f);
    }
    if (params.enabled < 0.5f || params.throughputClamp <= 0.0f) {
        return throughput;
    }
    simd::float3 positive = simd::max(throughput, simd_make_float3(0.0f, 0.0f, 0.0f));
    float lum = Luminance(positive);
    if (lum > params.throughputClamp && lum > 0.0f) {
        float scale = params.throughputClamp / std::max(lum, 1.0e-6f);
        if (outClamped) {
            *outClamped = true;
        }
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

simd::float2 DfgApprox(float roughness, float NoV) {
    const simd::float4 c0 = {-1.0f, -0.0275f, -0.572f, 0.022f};
    const simd::float4 c1 = {1.0f, 0.0425f, 1.04f, -0.04f};
    simd::float4 r = roughness * c0 + c1;
    float a004 = std::min(r.x * r.x, std::exp2(-9.28f * NoV)) * r.x + r.y;
    return simd_make_float2(-1.04f * a004 + r.z,
                            1.04f * a004 + r.w);
}

simd::float3 SpecularEnergyCompensation(const simd::float3& f0,
                                        float roughness,
                                        float NoV) {
    float NoVClamped = std::clamp(NoV, 0.0f, 1.0f);
    simd::float2 dfg = DfgApprox(roughness, NoVClamped);
    simd::float3 Fss = simd::clamp(f0 * dfg.x + dfg.y,
                                   simd_make_float3(0.0f, 0.0f, 0.0f),
                                   simd_make_float3(0.99f, 0.99f, 0.99f));
    simd::float3 Favg = f0 + (simd_make_float3(1.0f, 1.0f, 1.0f) - f0) * (1.0f / 21.0f);
    simd::float3 oneMinusFss = simd::clamp(simd_make_float3(1.0f, 1.0f, 1.0f) - Fss,
                                           simd_make_float3(0.0f, 0.0f, 0.0f),
                                           simd_make_float3(1.0f, 1.0f, 1.0f));
    simd::float3 denom = simd::max(simd_make_float3(1.0f, 1.0f, 1.0f) - Favg * oneMinusFss,
                                   simd_make_float3(1.0e-3f, 1.0e-3f, 1.0e-3f));
    simd::float3 Fms = (Favg * oneMinusFss) / denom;
    simd::float3 scale = (Fss + Fms) / simd::max(Fss, simd_make_float3(1.0e-4f, 1.0e-4f, 1.0e-4f));
    return simd::clamp(scale,
                       simd_make_float3(1.0f, 1.0f, 1.0f),
                       simd_make_float3(2.0f, 2.0f, 2.0f));
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
    float cosThetaO = simd::dot(normal, wo);
    float dotWoWh = simd::dot(wo, wh);
    if (cosThetaO <= 0.0f || cosThetaH <= 0.0f || dotWoWh <= 0.0f) {
        return 0.0f;
    }
    float denom = 4.0f * std::max(dotWoWh, 1.0e-6f);
    float d = GgxDistribution(alpha, std::max(cosThetaH, 0.0f));
    float g1 = GgxG1(alpha, cosThetaO);
    return d * g1 * std::max(cosThetaH, 0.0f) / denom;
}

float GgxVndfPdf(float alpha, const simd::float3& normal, const simd::float3& wo, const simd::float3& wh) {
    float cosThetaO = simd::dot(normal, wo);
    float cosThetaH = simd::dot(normal, wh);
    if (cosThetaO <= 0.0f || cosThetaH <= 0.0f) {
        return 0.0f;
    }
    float denom = std::max(simd::dot(wo, wh), 1.0e-6f);
    float d = GgxDistribution(alpha, cosThetaH);
    float g1 = GgxG1(alpha, cosThetaO);
    return d * g1 * cosThetaH / denom;
}

float GgxNdfPdf(float alpha, const simd::float3& normal, const simd::float3& wh) {
    float cosThetaH = std::max(simd::dot(normal, wh), 0.0f);
    return GgxDistribution(alpha, cosThetaH) * cosThetaH;
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

bool IsFinite2(const simd::float2& value) {
    return std::isfinite(value.x) && std::isfinite(value.y);
}

bool IsFinite4(const simd::float4& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
           std::isfinite(value.z) && std::isfinite(value.w);
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

PathTracerShaderTypes::MaterialData ShadeMaterialForHit(const EmbreeSceneData& sceneData,
                                                        const PathTracerShaderTypes::MaterialData& material,
                                                        const HitInfo& hit,
                                                        float debugNormalStrengthScale,
                                                        float surfaceFootprintWorld);
simd::float3 ApplyNormalMapForHit(const EmbreeSceneData& sceneData,
                                  const PathTracerShaderTypes::MaterialData& material,
                                  const HitInfo& hit,
                                  const simd::float3& shadingNormal,
                                  float debugNormalStrengthScale,
                                  float surfaceFootprintWorld);
float ApplyNormalVarianceToRoughness(const EmbreeSceneData& sceneData,
                                     const PathTracerShaderTypes::MaterialData& material,
                                     const HitInfo& hit,
                                     float debugNormalStrengthScale,
                                     float surfaceFootprintWorld);
simd::float3 PlasticDiffuseTransmission(const PathTracerShaderTypes::MaterialData& material,
                                        float cosThetaI,
                                        float cosThetaO);

float MaterialRoughness(const PathTracerShaderTypes::MaterialData& material) {
    float roughness = std::clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
    return std::max(roughness, 1.0e-3f);
}

float EnvironmentLightingRoughness(const PathTracerShaderTypes::MaterialData& material) {
    const uint32_t type = static_cast<uint32_t>(material.typeEta.x);
    switch (static_cast<PathTracerShaderTypes::MaterialType>(type)) {
        case PathTracerShaderTypes::MaterialType::Metal:
        case PathTracerShaderTypes::MaterialType::PbrMetallicRoughness:
            return std::clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        case PathTracerShaderTypes::MaterialType::Plastic:
            return std::clamp(std::max(material.coatParams.x, 1.0e-3f), 0.0f, 1.0f);
        case PathTracerShaderTypes::MaterialType::CarPaint:
            return std::clamp(material.pbrParams.y, 0.0f, 1.0f);
        default:
            return 1.0f;
    }
}

float DielectricF0(float ior) {
    float eta = std::max(ior, 1.0f);
    float r = (eta - 1.0f) / (eta + 1.0f);
    return r * r;
}

bool IsThinDielectric(const PathTracerShaderTypes::MaterialData& material) {
    return material.typeEta.w > 0.5f;
}

simd::float3 DielectricSigmaA(const PathTracerShaderTypes::MaterialData& material) {
    return simd::max(simd_make_float3(material.dielectricSigmaA.x,
                                      material.dielectricSigmaA.y,
                                      material.dielectricSigmaA.z),
                     simd_make_float3(0.0f, 0.0f, 0.0f));
}

simd::float3 DielectricTransmissionTint(const PathTracerShaderTypes::MaterialData& material,
                                        float cosTheta) {
    if (!IsThinDielectric(material)) {
        return simd_make_float3(1.0f, 1.0f, 1.0f);
    }
    simd::float3 sigmaA = DielectricSigmaA(material);
    if (sigmaA.x <= 1.0e-6f && sigmaA.y <= 1.0e-6f && sigmaA.z <= 1.0e-6f) {
        return simd_make_float3(1.0f, 1.0f, 1.0f);
    }
    float distance = 1.0f / std::max(std::fabs(cosTheta), 1.0e-3f);
    return Clamp01(Exp3(-sigmaA * distance));
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

float SssAnisotropy(const PathTracerShaderTypes::MaterialData& material) {
    return std::clamp(material.sssSigmaS.w, -0.99f, 0.99f);
}

float SssMeanFreePath(const PathTracerShaderTypes::MaterialData& material) {
    return std::max(material.sssParams.x, 1.0e-4f);
}

bool SssHasSigmaOverride(const PathTracerShaderTypes::MaterialData& material) {
    return material.sssSigmaA.w > 0.5f;
}

simd::float3 SssSigmaA(const PathTracerShaderTypes::MaterialData& material) {
    simd::float3 baseColor = MaterialBaseColor(material);
    float meanFreePath = SssMeanFreePath(material);
    float anisotropy = SssAnisotropy(material);
    if (SssHasSigmaOverride(material)) {
        return simd::max(simd_make_float3(material.sssSigmaA.x,
                                          material.sssSigmaA.y,
                                          material.sssSigmaA.z),
                         simd_make_float3(1.0e-6f, 1.0e-6f, 1.0e-6f));
    }
    float sigmaT = 1.0f / meanFreePath;
    simd::float3 sigmaS = Clamp01(baseColor) * sigmaT;
    sigmaS *= std::max(1.0f - anisotropy, 0.01f);
    return simd::max(simd_make_float3(sigmaT, sigmaT, sigmaT) - sigmaS,
                     simd_make_float3(1.0e-6f, 1.0e-6f, 1.0e-6f));
}

simd::float3 SssDiffuseColor(const PathTracerShaderTypes::MaterialData& material,
                             float cosThetaI,
                             float cosThetaO) {
    simd::float3 baseColor = MaterialBaseColor(material);
    float meanFreePath = SssMeanFreePath(material);
    simd::float3 sigmaA = SssSigmaA(material);
    const float travelScale = LoadEmbreeSssTravelScale();

    // Embree remains an oracle approximation here: use absorption over a
    // representative travel distance to keep SSS materials darker and more
    // saturated than a pure Lambert fallback.
    const float travel = std::max(meanFreePath * travelScale, 1.0e-4f);
    simd::float3 transmission = Exp3(-sigmaA * travel);
    simd::float3 diffuse = baseColor * transmission;

    if (material.sssParams.z > 0.5f) {
        diffuse *= PlasticDiffuseTransmission(material, cosThetaI, cosThetaO);
        diffuse *= std::max(1.0f - PlasticCoatFresnelAverage(material), 0.0f);
    }

    return Clamp01(diffuse);
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

bool MaterialAlphaMaskDiscards(const PathTracerShaderTypes::MaterialData& material) {
    const float alphaMode = material.pbrExtras.w;
    if (!(alphaMode > 0.5f)) {
        return false;
    }
    const float alpha = std::clamp(material.pbrExtras.x, 0.0f, 1.0f);
    const float cutoff = (alphaMode < 1.5f)
        ? std::clamp(material.pbrExtras.y, 0.0f, 1.0f)
        : 0.5f;
    return alpha < cutoff;
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

DirectDebugConfig LoadDirectDebugConfig(const PathTracer::RenderSettings& settings,
                                        uint32_t width,
                                        uint32_t height) {
    DirectDebugConfig cfg;
#if PT_DEBUG_TOOLS
    if (settings.enablePathDebug && settings.debugMaxEntries > 0) {
        cfg.enabled = true;
        cfg.pixelX = std::min(settings.debugPixelX, width > 0 ? width - 1u : 0u);
        cfg.pixelY = std::min(settings.debugPixelY, height > 0 ? height - 1u : 0u);
        return cfg;
    }
#endif
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

bool DeterministicRectLightAuditSample(const RectLightInfo& light,
                                       uint32_t lightCount,
                                       const EnvironmentMap* env,
                                       const PathTracer::RenderSettings& settings,
                                       const HitInfo& hit,
                                       RectLightSample& outSample) {
    outSample = RectLightSample{};
    simd::float3 samplePoint = light.corner;
    if (!light.isDisk) {
        samplePoint += 0.5f * light.edgeU + 0.5f * light.edgeV;
    }
    simd::float3 toLight = samplePoint - hit.position;
    float distSq = simd::dot(toLight, toLight);
    if (distSq <= 0.0f || light.area <= 0.0f) {
        return false;
    }
    float distance = std::sqrt(distSq);
    simd::float3 direction = toLight / distance;
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
    simd::float3 emission = light.baseEmission;
    if (light.emissionUsesEnv && env && env->width > 0 && env->height > 0 && !env->rgba.empty()) {
        simd::float3 sampleDir = -light.normal;
        emission *= SampleEnvironment(*env,
                                      sampleDir,
                                      settings.environmentRotation,
                                      settings.environmentIntensity);
    }
    if (!(simd::dot(emission, emission) > 0.0f)) {
        return false;
    }

    outSample.direction = direction;
    outSample.distance = distance;
    float selectionPdf = 1.0f / std::max(static_cast<float>(lightCount), 1.0f);
    outSample.pdf = pdfDir * selectionPdf;
    outSample.emission = emission;
    outSample.rectIndex = light.rectIndex;
    outSample.area = light.area;
    outSample.cosLight = cosLight;
    return true;
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

    simd::float3 samplePoint = light.corner;
    if (light.isDisk) {
        simd::float3 disk = SampleInUnitDisk(rng);
        samplePoint += disk.x * light.edgeU + disk.y * light.edgeV;
    } else {
        float u = rng.nextFloat();
        float v = rng.nextFloat();
        samplePoint += u * light.edgeU + v * light.edgeV;
    }
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
    float area = AreaPrimitiveArea(rect);
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
    const uint32_t type = static_cast<uint32_t>(material.typeEta.x);
    float signedCosThetaO = simd::dot(normal, wo);
    float signedCosThetaI = simd::dot(normal, wi);

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
        float absCosThetaO = std::fabs(signedCosThetaO);
        float absCosThetaI = std::fabs(signedCosThetaI);
        if (absCosThetaO <= 0.0f || absCosThetaI <= 0.0f) {
            return result;
        }

        simd::float3 baseColor = MaterialBaseColor(material);
        float metallic = std::clamp(material.pbrParams.x, 0.0f, 1.0f);
        float roughness = std::clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        float dielectricF0 = DielectricF0FromIor(material.typeEta.y);
        simd::float3 f0 = baseColor * metallic +
                          simd_make_float3(dielectricF0, dielectricF0, dielectricF0) * (1.0f - metallic);
        simd::float3 diffuseColor = baseColor * (1.0f - metallic) *
                                    std::clamp(material.pbrParams.z, 0.0f, 1.0f);

        float transmission = std::clamp(material.pbrExtras.z, 0.0f, 1.0f) * (1.0f - metallic);
        float reflectScale = 1.0f - transmission;
        float specWeightBase = PbrSpecularWeight(f0);
        float wSpec = specWeightBase * reflectScale;
        float wDiff = (1.0f - specWeightBase) * reflectScale;
        float wTrans = transmission;
        float weightSum = wSpec + wDiff + wTrans;
        if (weightSum <= 0.0f) {
            return result;
        }
        float pSpec = wSpec / weightSum;
        float pDiff = wDiff / weightSum;
        float pTrans = wTrans / weightSum;

        float alpha = std::max(roughness * roughness, 1.0e-4f);
        if (signedCosThetaO * signedCosThetaI > 0.0f) {
            if (signedCosThetaO <= 0.0f || signedCosThetaI <= 0.0f) {
                return result;
            }
            simd::float3 wh = simd::normalize(wo + wi);
            if (simd::dot(wh, normal) <= 0.0f ||
                simd::dot(wo, wh) <= 0.0f ||
                simd::dot(wi, wh) <= 0.0f) {
                return result;
            }
            float D = GgxDistribution(alpha, simd::dot(normal, wh));
            float G = GgxG1(alpha, signedCosThetaO) * GgxG1(alpha, signedCosThetaI);
            simd::float3 F = SchlickFresnel(f0, simd::dot(wi, wh));
            float denom = 4.0f * signedCosThetaO * signedCosThetaI;
            simd::float3 spec = F * (D * G / std::max(denom, 1.0e-6f));
            spec *= SpecularEnergyCompensation(f0, roughness, absCosThetaO);
            spec = ClampSpecularTail(spec, roughness, f0, clampParams);
            spec *= reflectScale;
            simd::float3 diffuse = (diffuseColor / kPi) * reflectScale;
            float pdfSpec = GgxPdf(alpha, normal, wo, wi);
            float pdfDiffuse = LambertPdf(normal, wi);
            float pdf = pSpec * pdfSpec + pDiff * pdfDiffuse;
            if (pdf > 0.0f) {
                result.value = simd::max(spec + diffuse, simd_make_float3(0.0f, 0.0f, 0.0f));
                result.pdf = ClampSpecularPdf(pdf, clampParams);
            }
            return result;
        }

        if (wTrans <= 0.0f) {
            return result;
        }
        float etaI = 1.0f;
        float etaT = std::max(material.typeEta.y, 1.0f);
        if (signedCosThetaO < 0.0f) {
            std::swap(etaI, etaT);
        }
        float eta = etaI / etaT;
        simd::float3 wh = simd::normalize(wo + wi * eta);
        if (!IsFinite(wh) || simd::dot(wh, wh) <= 0.0f) {
            return result;
        }
        if (simd::dot(wh, normal) <= 0.0f) {
            wh = -wh;
        }
        float cosThetaOWh = simd::dot(wo, wh);
        float cosThetaIWh = simd::dot(wi, wh);
        if (cosThetaOWh * cosThetaIWh > 0.0f) {
            return result;
        }
        float D = GgxDistribution(alpha, std::max(simd::dot(normal, wh), 0.0f));
        float G = GgxG1(alpha, absCosThetaO) * GgxG1(alpha, absCosThetaI);
        float cosThetaT = 0.0f;
        float F = FresnelDielectricExact(cosThetaOWh, etaI, etaT, cosThetaT);
        float denom = cosThetaOWh + eta * cosThetaIWh;
        float denomSq = denom * denom;
        if (std::fabs(denomSq) <= 1.0e-8f) {
            return result;
        }
        float factor = (eta * eta) * std::fabs(cosThetaIWh) * std::fabs(cosThetaOWh);
        factor /= std::max(absCosThetaO * absCosThetaI * denomSq, 1.0e-6f);
        simd::float3 ft = simd_make_float3(1.0f - F, 1.0f - F, 1.0f - F) * D * G * factor;
        ft *= DielectricTransmissionTint(material, absCosThetaI);
        ft *= transmission;
        float pdfWh = GgxVndfPdf(alpha, normal, wo, wh);
        float dwhDwi = std::fabs((eta * eta * cosThetaIWh) / std::max(denomSq, 1.0e-8f));
        float pdf = pTrans * pdfWh * dwhDwi;
        if (pdf > 0.0f) {
            result.value = simd::max(ft, simd_make_float3(0.0f, 0.0f, 0.0f));
            result.pdf = ClampSpecularPdf(pdf, clampParams);
        }
        return result;
    }

    float cosThetaO = std::max(signedCosThetaO, 0.0f);
    float cosThetaI = std::max(signedCosThetaI, 0.0f);
    if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Lambertian) ||
        type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Subsurface)) {
        simd::float3 albedo = (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Subsurface))
            ? SssDiffuseColor(material, cosThetaI, cosThetaO)
            : MaterialBaseColor(material);
        result.value = albedo / kPi;
        result.pdf = LambertPdf(normal, wi);
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

CpuReservoir MakeEmptyCpuReservoir() {
    return CpuReservoir{};
}

CpuRestirDiReservoirState MakeEmptyCpuRestirDiReservoir(uint32_t frameTag) {
    CpuRestirDiReservoirState reservoir;
    reservoir.frameTag = frameTag;
    return reservoir;
}

CpuRestirDiVisibilityState MakeEmptyCpuRestirDiVisibility(uint32_t frameTag) {
    CpuRestirDiVisibilityState visibility;
    visibility.frameTag = frameTag;
    return visibility;
}

bool CpuRestirDiReservoirValid(const CpuRestirDiReservoirState& reservoir) {
    return (reservoir.stateFlags & 1u) != 0u &&
           reservoir.candidateCount > 0u &&
           reservoir.weight > 0.0f &&
           reservoir.pdf > 0.0f &&
           IsFinite(reservoir.position) &&
           IsFinite(reservoir.normal) &&
           IsFinite(reservoir.emission);
}

bool CpuUpdateReservoir(CpuReservoir& reservoir,
                        const CpuRisSamplePayload& candidate,
                        float weight,
                        float u) {
    if (!(weight > 0.0f) || !std::isfinite(weight)) {
        return false;
    }
    float nextWeight = reservoir.wSum + weight;
    if (!(nextWeight > 0.0f) || !std::isfinite(nextWeight)) {
        return false;
    }
    reservoir.wSum = nextWeight;
    if (reservoir.M == 0u) {
        reservoir.M = 1u;
        reservoir.winner = candidate;
        reservoir.valid = true;
        return true;
    }
    reservoir.M += 1u;
    float acceptProbability = weight / reservoir.wSum;
    if (u < acceptProbability) {
        reservoir.winner = candidate;
        reservoir.valid = true;
        return true;
    }
    return false;
}

bool CpuRisPayloadIsDirectional(const CpuRisSamplePayload& payload) {
    return (payload.flags & PathTracerShaderTypes::kLightPrimitiveFlagDirectional) != 0u;
}

bool CpuRisPayloadIsAnalyticRect(const CpuRisSamplePayload& payload) {
    return (payload.flags & kCpuRisPayloadAnalyticRectFlag) != 0u;
}

simd::float3 CpuRisPayloadOmegaL(const HitInfo& hit, const CpuRisSamplePayload& payload) {
    return CpuRisPayloadIsDirectional(payload)
        ? simd::normalize(payload.position)
        : simd::normalize(payload.position - hit.position);
}

float CpuRisPayloadDistanceSq(const HitInfo& hit, const CpuRisSamplePayload& payload) {
    if (CpuRisPayloadIsDirectional(payload)) {
        return 1.0f;
    }
    simd::float3 delta = payload.position - hit.position;
    return simd::dot(delta, delta);
}

bool CpuShadowHitIsSelectedLight(const HitInfo& shadowHit, const CpuRisSamplePayload& payload) {
    if (CpuRisPayloadIsDirectional(payload)) {
        return false;
    }
    if (CpuRisPayloadIsAnalyticRect(payload)) {
        return shadowHit.primitiveType == GeometryType::Rectangles &&
               shadowHit.primitiveIndex == payload.primitiveIndex;
    }
    return shadowHit.primitiveType == GeometryType::Mesh &&
           shadowHit.primitiveIndex == payload.primitiveIndex;
}

float CpuPHat(const simd::float3& emission,
              const simd::float3& bsdf,
              float cosThetaSurface,
              float cosThetaLight,
              float distSq) {
    if (!(distSq > 0.0f) || !std::isfinite(distSq)) {
        return 0.0f;
    }
    float value = Luminance(emission) * Luminance(bsdf) *
                  cosThetaSurface * cosThetaLight / distSq;
    return (std::isfinite(value) && value > 0.0f) ? value : 0.0f;
}

bool CpuSampleDirectionalLight(const std::vector<DirectionalLightInfo>& directionalLights,
                               float providerPdf,
                               Rng& rng,
                               CpuRisSamplePayload& outSample) {
    outSample = CpuRisSamplePayload{};
    if (directionalLights.empty()) {
        return false;
    }
    uint32_t selected = std::min(static_cast<uint32_t>(rng.nextFloat() * directionalLights.size()),
                                 static_cast<uint32_t>(directionalLights.size() - 1u));
    const DirectionalLightInfo& light = directionalLights[selected];
    outSample.primitiveIndex = selected;
    outSample.flags = PathTracerShaderTypes::kLightPrimitiveFlagDirectional;
    outSample.position = light.direction;
    outSample.normal = -light.direction;
    outSample.emission = light.emission;
    outSample.pdf = providerPdf / static_cast<float>(directionalLights.size());
    return outSample.pdf > 0.0f && std::isfinite(outSample.pdf) && IsFinite(outSample.emission);
}

bool CpuSampleMeshLight(const std::vector<MeshLightInfo>& meshLights,
                        const HitInfo& hit,
                        float providerPdf,
                        Rng& rng,
                        CpuRisSamplePayload& outSample) {
    outSample = CpuRisSamplePayload{};
    if (meshLights.empty()) {
        return false;
    }
    const float uSelect = rng.nextFloat();
    auto selectedIt = std::lower_bound(meshLights.begin(),
                                       meshLights.end(),
                                       uSelect,
                                       [](const MeshLightInfo& light, float u) {
                                           return light.cumulativePower < u;
                                       });
    if (selectedIt == meshLights.end()) {
        selectedIt = meshLights.end() - 1;
    }
    const MeshLightInfo& light = *selectedIt;
    if (!(light.area > 0.0f) ||
        !(light.selectionPdf > 0.0f) ||
        !IsFinite(light.emission) ||
        !IsFinite(light.normal) ||
        !IsFinite(light.edge1) ||
        !IsFinite(light.edge2)) {
        return false;
    }

    simd::float3 normal = light.normal;
    const float normalLenSq = simd::dot(normal, normal);
    if (!(normalLenSq > 1.0e-12f) || !std::isfinite(normalLenSq)) {
        return false;
    }
    normal /= std::sqrt(normalLenSq);

    const simd::float3 v0 = light.centroid - (light.edge1 + light.edge2) * (1.0f / 3.0f);
    const float sqrtU = std::sqrt(std::max(rng.nextFloat(), 0.0f));
    const float b1 = sqrtU * (1.0f - rng.nextFloat());
    const float b2 = sqrtU - b1;
    const simd::float3 samplePoint = v0 + b1 * light.edge1 + b2 * light.edge2;
    const simd::float3 toLight = samplePoint - hit.position;
    const float distSq = simd::dot(toLight, toLight);
    if (!(distSq > 0.0f) || !std::isfinite(distSq)) {
        return false;
    }
    const float distance = std::sqrt(distSq);
    if (!(distance > 0.0f) || !std::isfinite(distance)) {
        return false;
    }
    const simd::float3 direction = toLight / distance;
    const float cosLight = simd::dot(-direction, normal);
    if (!(cosLight > 0.0f) || !std::isfinite(cosLight)) {
        return false;
    }

    const float pdfArea = 1.0f / light.area;
    const float pdfDir = pdfArea * distSq / std::max(cosLight, 1.0e-6f);
    outSample.primitiveIndex = light.primitiveIndex;
    outSample.flags = 0u;
    outSample.position = samplePoint;
    outSample.normal = normal;
    outSample.emission = light.emission;
    outSample.pdf = pdfDir * light.selectionPdf * providerPdf;
    return outSample.pdf > 0.0f && std::isfinite(outSample.pdf);
}

bool CpuSampleAnalyticRectLight(const std::vector<RectLightInfo>& rectLights,
                                const EnvironmentMap* env,
                                const PathTracer::RenderSettings& settings,
                                const HitInfo& hit,
                                float providerPdf,
                                Rng& rng,
                                CpuRisSamplePayload& outSample) {
    outSample = CpuRisSamplePayload{};
    RectLightSample rectSample;
    if (!SampleRectLight(rectLights, env, settings, hit, rng, rectSample)) {
        return false;
    }
    outSample.primitiveIndex = rectSample.rectIndex;
    outSample.flags = kCpuRisPayloadAnalyticRectFlag;
    outSample.position = hit.position + rectSample.direction * rectSample.distance;
    if (rectSample.lightIndex < rectLights.size()) {
        outSample.normal = rectLights[rectSample.lightIndex].normal;
    } else {
        outSample.normal = -rectSample.direction;
    }
    outSample.emission = rectSample.emission;
    outSample.pdf = rectSample.pdf * providerPdf;
    return outSample.pdf > 0.0f && std::isfinite(outSample.pdf);
}

float CpuReservoirWinnerPHatForHit(const HitInfo& hit,
                                   const PathTracerShaderTypes::MaterialData& material,
                                   const simd::float3& shadingNormal,
                                   const simd::float3& wo,
                                   const FireflyClampParams& clampParams,
                                   const CpuRisSamplePayload& winner) {
    float distSq = CpuRisPayloadDistanceSq(hit, winner);
    if (!(distSq > 0.0f) || !std::isfinite(distSq)) {
        return 0.0f;
    }
    simd::float3 omegaL = CpuRisPayloadOmegaL(hit, winner);
    float cosThetaSurface = std::clamp(simd::dot(shadingNormal, omegaL), 0.0f, 1.0f);
    float cosThetaLight = CpuRisPayloadIsDirectional(winner)
        ? 1.0f
        : std::clamp(simd::dot(winner.normal, -omegaL), 0.0f, 1.0f);
    BsdfEval bsdfEval = EvaluateBsdf(material, hit.position, shadingNormal, wo, omegaL, clampParams);
    if (bsdfEval.isDelta) {
        return 0.0f;
    }
    return CpuPHat(winner.emission, bsdfEval.value, cosThetaSurface, cosThetaLight, distSq);
}

bool CpuBuildRestirDiProviderReservoirForHit(
    const std::vector<DirectionalLightInfo>& directionalLights,
    const std::vector<MeshLightInfo>& meshLights,
    const std::vector<RectLightInfo>& rectLights,
    const EnvironmentMap* env,
    const PathTracer::RenderSettings& settings,
    const HitInfo& hit,
    const PathTracerShaderTypes::MaterialData& material,
    const simd::float3& shadingNormal,
    const simd::float3& wo,
    const FireflyClampParams& clampParams,
    uint32_t candidateCount,
    Rng& rng,
    CpuReservoir& reservoir) {
    reservoir = MakeEmptyCpuReservoir();
    const bool hasDirectionalProvider = !directionalLights.empty();
    const bool hasMeshProvider = !meshLights.empty();
    const bool hasAnalyticProvider = !rectLights.empty();
    const uint32_t providerCount = (hasDirectionalProvider ? 1u : 0u) +
                                   (hasMeshProvider ? 1u : 0u) +
                                   (hasAnalyticProvider ? 1u : 0u);
    if (providerCount == 0u) {
        return false;
    }
    const float providerPdf = 1.0f / static_cast<float>(providerCount);
    const uint32_t risTargetM = std::max(candidateCount, 1u);
    for (uint32_t i = 0; i < risTargetM; ++i) {
        uint32_t selected = std::min(static_cast<uint32_t>(rng.nextFloat() * providerCount),
                                     providerCount - 1u);
        CpuRisSamplePayload candidate;
        bool sampled = false;
        uint32_t ordinal = 0u;
        if (hasDirectionalProvider) {
            if (selected == ordinal) {
                sampled = CpuSampleDirectionalLight(directionalLights, providerPdf, rng, candidate);
            }
            ordinal += 1u;
        }
        if (!sampled && hasMeshProvider && selected == ordinal) {
            sampled = CpuSampleMeshLight(meshLights, hit, providerPdf, rng, candidate);
        }
        if (hasMeshProvider) {
            ordinal += 1u;
        }
        if (!sampled && hasAnalyticProvider && selected == ordinal) {
            sampled = CpuSampleAnalyticRectLight(rectLights, env, settings, hit, providerPdf, rng, candidate);
        }
        if (!sampled) {
            continue;
        }
        float distSq = CpuRisPayloadDistanceSq(hit, candidate);
        if (!(distSq > 0.0f) || !std::isfinite(distSq)) {
            continue;
        }
        simd::float3 omegaL = CpuRisPayloadOmegaL(hit, candidate);
        float cosThetaSurface = std::clamp(simd::dot(shadingNormal, omegaL), 0.0f, 1.0f);
        float cosThetaLight = CpuRisPayloadIsDirectional(candidate)
            ? 1.0f
            : std::clamp(simd::dot(candidate.normal, -omegaL), 0.0f, 1.0f);
        BsdfEval bsdfEval = EvaluateBsdf(material, hit.position, shadingNormal, wo, omegaL, clampParams);
        if (bsdfEval.isDelta) {
            continue;
        }
        float phat = CpuPHat(candidate.emission, bsdfEval.value, cosThetaSurface, cosThetaLight, distSq);
        if (!(phat > 0.0f) || !(candidate.pdf > 0.0f)) {
            continue;
        }
        float weight = phat / candidate.pdf;
        float u = (reservoir.M == 0u) ? 0.0f : rng.nextFloat();
        CpuUpdateReservoir(reservoir, candidate, weight, u);
    }
    return reservoir.valid && reservoir.M > 0u && reservoir.wSum > 0.0f;
}

CpuRestirDiReservoirState CpuRestirDiStateFromReservoir(
    const HitInfo& hit,
    const PathTracerShaderTypes::MaterialData& material,
    const simd::float3& shadingNormal,
    const simd::float3& wo,
    const FireflyClampParams& clampParams,
    uint32_t frameTag,
    const CpuReservoir& reservoir) {
    CpuRestirDiReservoirState state = MakeEmptyCpuRestirDiReservoir(frameTag);
    if (!reservoir.valid || reservoir.M == 0u || !(reservoir.wSum > 0.0f)) {
        return state;
    }
    float targetPdf = CpuReservoirWinnerPHatForHit(hit, material, shadingNormal, wo, clampParams, reservoir.winner);
    state.position = reservoir.winner.position;
    state.pdf = reservoir.winner.pdf;
    state.normal = reservoir.winner.normal;
    state.targetPdf = targetPdf;
    state.emission = reservoir.winner.emission;
    state.weight = reservoir.wSum;
    state.primitiveIndex = reservoir.winner.primitiveIndex;
    state.flags = reservoir.winner.flags;
    state.stateFlags = 1u;
    state.frameTag = frameTag;
    state.candidateCount = reservoir.M;
    state.surfacePrimitiveIndex = hit.primitiveIndex;
    return state;
}

bool CpuDiffuseRestirEligibleMaterial(const PathTracerShaderTypes::MaterialData& material) {
    const uint32_t type = static_cast<uint32_t>(material.typeEta.x);
    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
        const float metallic = std::clamp(material.pbrParams.x, 0.0f, 1.0f);
        const float transmission = std::clamp(material.pbrExtras.z, 0.0f, 1.0f) * (1.0f - metallic);
        return metallic <= 0.25f && transmission <= 0.05f;
    }
    return type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Lambertian);
}

bool CpuPathGuidingMaterialAllowed(const PathTracerShaderTypes::MaterialData& material,
                                   uint32_t depth) {
    if (depth < 1u || MaterialIsDelta(material)) {
        return false;
    }
    const uint32_t type = static_cast<uint32_t>(material.typeEta.x);
    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Lambertian)) {
        return true;
    }
    if (type != static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
        return false;
    }
    const float roughness = std::clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
    const float metallic = std::clamp(material.pbrParams.x, 0.0f, 1.0f);
    const float transmission = std::clamp(material.pbrExtras.z, 0.0f, 1.0f) * (1.0f - metallic);
    return roughness >= 0.35f && metallic <= 0.25f && transmission <= 0.05f;
}

bool CpuReweightBsdfSampleForDirection(const PathTracerShaderTypes::MaterialData& material,
                                       const simd::float3& position,
                                       const simd::float3& normal,
                                       const simd::float3& wo,
                                       const simd::float3& wi,
                                       const FireflyClampParams& clampParams,
                                       BsdfSample& sample) {
    if (!IsFinite(wi) || simd::dot(wi, wi) <= 0.0f) {
        return false;
    }
    simd::float3 dir = simd::normalize(wi);
    const float cosTheta = std::max(simd::dot(normal, dir), 0.0f);
    if (!(cosTheta > 0.0f)) {
        return false;
    }
    BsdfEval eval = EvaluateBsdf(material, position, normal, wo, dir, clampParams);
    if (eval.isDelta || !(eval.pdf > 0.0f) || !IsFinite(eval.value)) {
        return false;
    }
    simd::float3 weight = eval.value * (cosTheta / eval.pdf);
    if (!IsFinite(weight)) {
        return false;
    }
    sample.direction = dir;
    sample.weight = simd::max(weight, simd_make_float3(0.0f, 0.0f, 0.0f));
    sample.pdf = eval.pdf;
    sample.lobeType = 0u;
    sample.lobeRoughness = 1.0f;
    sample.isDelta = false;
    return Luminance(sample.weight) > 0.0f;
}

bool CpuBsdfDiffuseCandidateValid(const BsdfSample& sample,
                                  const PathTracerShaderTypes::MaterialData& material,
                                  uint32_t depth,
                                  bool requirePrimaryDepth) {
    if (requirePrimaryDepth && depth != 0u) {
        return false;
    }
    return CpuDiffuseRestirEligibleMaterial(material) &&
           sample.pdf > 0.0f &&
           !sample.isDelta &&
           sample.lobeType == 0u &&
           IsFinite(sample.direction) &&
           simd::dot(sample.direction, sample.direction) > 0.0f &&
           IsFinite(sample.weight) &&
           sample.weight.x >= 0.0f &&
           sample.weight.y >= 0.0f &&
           sample.weight.z >= 0.0f;
}

bool CpuLoadPathReservoirDirection(const CpuPathReservoirState& reservoir,
                                   const simd::float3& position,
                                   const simd::float3& normal,
                                   uint32_t depth,
                                   bool depthAware,
                                   float maxPositionDistance,
                                   simd::float3& direction,
                                   float& confidence) {
    (void)position;
    (void)maxPositionDistance;
    if (!reservoir.valid || reservoir.lobeType != 0u) {
        return false;
    }
    const simd::float3 axis = simd::normalize(normal);
    const simd::float3 reservoirNormal = simd::normalize(reservoir.normal);
    const float normalAgreement = simd::dot(axis, reservoirNormal);
    if (!(normalAgreement > 0.0f)) {
        return false;
    }
    const simd::float3 candidate = simd::normalize(reservoir.direction);
    if (!IsFinite(candidate) ||
        simd::dot(candidate, candidate) <= 0.0f ||
        simd::dot(axis, candidate) <= 0.0f) {
        return false;
    }
    if (!(reservoir.sampleLuma > 0.0f) || !std::isfinite(reservoir.sampleLuma)) {
        return false;
    }
    float compatibility = std::clamp(normalAgreement, 0.0f, 1.0f);
    if (depthAware) {
        const float depthDelta =
            std::fabs(static_cast<float>(reservoir.depth) - static_cast<float>(depth));
        compatibility *= 1.0f / (1.0f + 0.25f * depthDelta);
    }
    direction = candidate;
    confidence = std::clamp(compatibility *
                                (reservoir.sampleLuma /
                                 std::max(reservoir.sampleLuma + reservoir.throughputLuma, 1.0e-4f)),
                            0.0625f,
                            1.0f);
    return true;
}

void CpuUpdatePathReservoir(CpuPathReservoirState& reservoir,
                            const PathTracerShaderTypes::MaterialData& material,
                            const simd::float3& position,
                            const simd::float3& normal,
                            const simd::float3& throughput,
                            uint32_t depth,
                            uint32_t frameTag,
                            const BsdfSample& sample) {
    const float sampleLuma = std::max(Luminance(simd::max(sample.weight,
                                                          simd_make_float3(0.0f, 0.0f, 0.0f))),
                                     0.0f);
    const float throughputLuma = std::max(Luminance(simd::max(throughput,
                                                              simd_make_float3(0.0f, 0.0f, 0.0f))),
                                         0.0f);
    if (!(sample.pdf > 0.0f) ||
        !(sampleLuma > 0.0f) ||
        !std::isfinite(sampleLuma) ||
        !IsFinite(sample.direction) ||
        !IsFinite(position) ||
        !IsFinite(normal) ||
        !IsFinite(throughput)) {
        return;
    }
    reservoir.valid = true;
    reservoir.position = position;
    reservoir.normal = simd::normalize(normal);
    reservoir.direction = simd::normalize(sample.direction);
    reservoir.throughput = simd::max(throughput, simd_make_float3(0.0f, 0.0f, 0.0f));
    reservoir.pdf = sample.pdf;
    reservoir.sampleLuma = sampleLuma;
    reservoir.throughputLuma = throughputLuma;
    reservoir.materialType = static_cast<uint32_t>(material.typeEta.x);
    reservoir.lobeType = 0u;
    reservoir.depth = depth;
    reservoir.frameTag = frameTag;
}

uint32_t CpuRadianceCacheDomainKey(const PathTracer::RenderSettings& settings,
                                   const simd::float3& position,
                                   uint32_t depth,
                                   uint32_t materialType,
                                   uint32_t lobeType) {
    const float invCell = 1.0f / std::max(settings.radianceCacheCellSize, 1.0e-4f);
    const int32_t cx = static_cast<int32_t>(std::floor(position.x * invCell));
    const int32_t cy = static_cast<int32_t>(std::floor(position.y * invCell));
    const int32_t cz = static_cast<int32_t>(std::floor(position.z * invCell));
    uint32_t hx = static_cast<uint32_t>(cx) * 73856093u;
    uint32_t hy = static_cast<uint32_t>(cy) * 19349663u;
    uint32_t hz = static_cast<uint32_t>(cz) * 83492791u;
    uint32_t hd = std::min(depth, 7u) * 2654435761u;
    uint32_t hm = materialType * 2246822519u;
    uint32_t hl = lobeType * 3266489917u;
    return CpuPcgHash(hx ^ hy ^ hz ^ hd ^ hm ^ hl ^ 0x9E3779B9u);
}

bool CpuRadianceCacheCanQuery(const PathTracer::RenderSettings& settings,
                              const PathTracerShaderTypes::MaterialData& material,
                              const BsdfSample& sample,
                              uint32_t depth) {
    return settings.radianceCacheMode == PathTracer::RenderSettings::RadianceCacheMode::Prototype &&
           depth >= std::max(settings.radianceCacheMinDepth, 1u) &&
           CpuDiffuseRestirEligibleMaterial(material) &&
           sample.pdf > 0.0f &&
           !sample.isDelta &&
           sample.lobeType == 0u &&
           IsFinite(sample.weight);
}

constexpr uint32_t kCpuPathGuidingBinDim = 8u;
constexpr uint32_t kCpuPathGuidingBinCount = kCpuPathGuidingBinDim * kCpuPathGuidingBinDim;
constexpr uint32_t kCpuPathGuidingDepthCount = 8u;
constexpr float kCpuPathGuidingInvHemisphereSolidAngle = 0.15915494309189535f;

size_t CpuPathGuidingCellCount(uint32_t width, uint32_t height) {
    const size_t totalBinCapacity =
        std::max(static_cast<size_t>(width) * static_cast<size_t>(height) *
                     static_cast<size_t>(kCpuPathGuidingDepthCount),
                 static_cast<size_t>(kCpuPathGuidingBinCount));
    return std::max(totalBinCapacity / kCpuPathGuidingBinCount, static_cast<size_t>(1));
}

uint32_t CpuPathGuidingCellHash(const simd::float3& position, float cellSize) {
    const float invCell = 1.0f / std::max(cellSize, 1.0e-4f);
    const int32_t cx = static_cast<int32_t>(std::floor(position.x * invCell));
    const int32_t cy = static_cast<int32_t>(std::floor(position.y * invCell));
    const int32_t cz = static_cast<int32_t>(std::floor(position.z * invCell));
    const uint32_t hx = static_cast<uint32_t>(cx) * 73856093u;
    const uint32_t hy = static_cast<uint32_t>(cy) * 19349663u;
    const uint32_t hz = static_cast<uint32_t>(cz) * 83492791u;
    return CpuPcgHash(hx ^ hy ^ hz ^ 0xC2B2AE35u);
}

size_t CpuPathGuidingCellId(const PathTracer::RenderSettings& settings,
                            const simd::float3& position,
                            size_t cellCount) {
    return cellCount > 0u
        ? static_cast<size_t>(CpuPathGuidingCellHash(position,
                                                    std::max(settings.pathGuidingCellSize, 1.0e-4f))) %
              cellCount
        : 0u;
}

simd::float3 CpuPathGuidingToLocal(const simd::float3& direction,
                                   const simd::float3& normal) {
    const Onb onb = BuildOnb(normal);
    const simd::float3 dir = simd::normalize(direction);
    return simd_make_float3(simd::dot(dir, onb.tangent),
                            simd::dot(dir, onb.bitangent),
                            simd::dot(dir, onb.normal));
}

simd::float3 CpuPathGuidingToWorld(const simd::float3& local,
                                   const simd::float3& normal) {
    const Onb onb = BuildOnb(normal);
    return simd::normalize(local.x * onb.tangent +
                           local.y * onb.bitangent +
                           local.z * onb.normal);
}

simd::float2 CpuPathGuidingOctaEncodeHemisphere(const simd::float3& localDirection) {
    simd::float3 d = simd::normalize(simd_make_float3(localDirection.x,
                                                      localDirection.y,
                                                      std::max(localDirection.z, 0.0f)));
    const float invL1 =
        1.0f / std::max(std::fabs(d.x) + std::fabs(d.y) + std::fabs(d.z), 1.0e-6f);
    simd::float2 p = simd_make_float2(d.x * invL1, d.y * invL1);
    return simd_make_float2(std::clamp(p.x * 0.5f + 0.5f, 0.0f, 0.999999f),
                            std::clamp(p.y * 0.5f + 0.5f, 0.0f, 0.999999f));
}

simd::float3 CpuPathGuidingOctaDecodeHemisphere(const simd::float2& uv) {
    simd::float2 f = uv * 2.0f - simd_make_float2(1.0f, 1.0f);
    simd::float3 n = simd_make_float3(f.x, f.y, 1.0f - std::fabs(f.x) - std::fabs(f.y));
    if (n.z < 0.0f) {
        simd::float2 folded =
            simd_make_float2((1.0f - std::fabs(n.y)) * (f.x < 0.0f ? -1.0f : 1.0f),
                             (1.0f - std::fabs(n.x)) * (f.y < 0.0f ? -1.0f : 1.0f));
        n = simd_make_float3(folded.x, folded.y, 0.0f);
    }
    n.z = std::max(n.z, 1.0e-4f);
    return simd::normalize(n);
}

uint32_t CpuPathGuidingBinIdForDirection(const simd::float3& direction,
                                         const simd::float3& normal) {
    const simd::float3 local = CpuPathGuidingToLocal(simd::normalize(direction), normal);
    const simd::float2 uv = CpuPathGuidingOctaEncodeHemisphere(local);
    const uint32_t bx = std::min(static_cast<uint32_t>(
                                     std::floor(uv.x * static_cast<float>(kCpuPathGuidingBinDim))),
                                 kCpuPathGuidingBinDim - 1u);
    const uint32_t by = std::min(static_cast<uint32_t>(
                                     std::floor(uv.y * static_cast<float>(kCpuPathGuidingBinDim))),
                                 kCpuPathGuidingBinDim - 1u);
    return by * kCpuPathGuidingBinDim + bx;
}

simd::float3 CpuPathGuidingBinCenterDirection(uint32_t binId,
                                              const simd::float3& normal) {
    const uint32_t bx = binId % kCpuPathGuidingBinDim;
    const uint32_t by = binId / kCpuPathGuidingBinDim;
    const simd::float2 uv =
        (simd_make_float2(static_cast<float>(bx), static_cast<float>(by)) +
         simd_make_float2(0.5f, 0.5f)) /
        static_cast<float>(kCpuPathGuidingBinDim);
    return CpuPathGuidingToWorld(CpuPathGuidingOctaDecodeHemisphere(uv), normal);
}

bool CpuPathGuidingCandidateValid(const BsdfSample& sample,
                                  const PathTracerShaderTypes::MaterialData& material,
                                  uint32_t depth) {
    return CpuPathGuidingMaterialAllowed(material, depth) &&
           sample.pdf > 0.0f &&
           !sample.isDelta &&
           sample.lobeType == 0u &&
           IsFinite(sample.direction) &&
           simd::dot(sample.direction, sample.direction) > 0.0f &&
           IsFinite(sample.weight) &&
           sample.weight.x >= 0.0f &&
           sample.weight.y >= 0.0f &&
           sample.weight.z >= 0.0f;
}

float CpuPathGuidingBinWeight(const CpuPathGuidingReservoirState& state,
                              uint32_t frameIndex) {
    if (!state.valid ||
        state.sampleCount == 0u ||
        state.frameTag >= frameIndex ||
        state.lumaAccumulator == 0u) {
        return 0.0f;
    }
    const float meanLuma =
        static_cast<float>(state.lumaAccumulator) /
        (256.0f * static_cast<float>(std::max(state.sampleCount, 1u)));
    const float countWeight = std::sqrt(static_cast<float>(std::max(state.sampleCount, 1u)));
    const float weight = std::max(meanLuma, 1.0e-4f) * countWeight;
    return std::isfinite(weight) ? weight : 0.0f;
}

float CpuPathGuidingCellTotalWeight(const std::vector<CpuPathGuidingReservoirState>& states,
                                    size_t cellId,
                                    uint32_t frameIndex,
                                    uint32_t& sampleCountSum) {
    const size_t base = cellId * kCpuPathGuidingBinCount;
    float total = 0.0f;
    sampleCountSum = 0u;
    for (uint32_t bin = 0u; bin < kCpuPathGuidingBinCount && base + bin < states.size(); ++bin) {
        const CpuPathGuidingReservoirState& state = states[base + bin];
        total += CpuPathGuidingBinWeight(state, frameIndex);
        if (state.valid && state.frameTag < frameIndex && state.lumaAccumulator != 0u) {
            sampleCountSum += std::min(state.sampleCount, 65535u);
        }
    }
    return std::isfinite(total) ? total : 0.0f;
}

float CpuPathGuidingConfidenceFromSamples(uint32_t sampleCountSum) {
    return std::clamp(static_cast<float>(sampleCountSum) / 64.0f, 0.0f, 1.0f);
}

float CpuPathGuidingPdfDirection(const std::vector<CpuPathGuidingReservoirState>& states,
                                 const PathTracer::RenderSettings& settings,
                                 uint32_t frameIndex,
                                 const simd::float3& position,
                                 const simd::float3& normal,
                                 size_t cellCount,
                                 const simd::float3& direction) {
    if (states.empty() || !IsFinite(direction)) {
        return 0.0f;
    }
    const simd::float3 axis = simd::normalize(normal);
    const float cosTheta = simd::dot(axis, simd::normalize(direction));
    if (!(cosTheta > 0.0f)) {
        return 0.0f;
    }
    const size_t cellId = CpuPathGuidingCellId(settings, position, cellCount);
    uint32_t sampleCountSum = 0u;
    const float total = CpuPathGuidingCellTotalWeight(states, cellId, frameIndex, sampleCountSum);
    if (!(total > 0.0f) || sampleCountSum == 0u) {
        return 0.0f;
    }
    const uint32_t binId = CpuPathGuidingBinIdForDirection(direction, axis);
    const size_t index = cellId * kCpuPathGuidingBinCount + binId;
    if (index >= states.size()) {
        return 0.0f;
    }
    const float binWeight = CpuPathGuidingBinWeight(states[index], frameIndex);
    if (!(binWeight > 0.0f)) {
        return 0.0f;
    }
    const float pdf = (binWeight / total) * static_cast<float>(kCpuPathGuidingBinCount) *
                      kCpuPathGuidingInvHemisphereSolidAngle;
    return (pdf > 0.0f && std::isfinite(pdf)) ? pdf : 0.0f;
}

bool CpuPathGuidingSampleDirection(const std::vector<CpuPathGuidingReservoirState>& states,
                                   const PathTracer::RenderSettings& settings,
                                   uint32_t frameIndex,
                                   const simd::float3& position,
                                   const simd::float3& normal,
                                   size_t cellCount,
                                   Rng& rng,
                                   simd::float3& direction,
                                   float& pdf,
                                   float& confidence,
                                   size_t& cellId,
                                   uint32_t& selectedBin) {
    if (states.empty() || cellCount == 0u) {
        return false;
    }
    cellId = CpuPathGuidingCellId(settings, position, cellCount);
    uint32_t sampleCountSum = 0u;
    const float total = CpuPathGuidingCellTotalWeight(states, cellId, frameIndex, sampleCountSum);
    if (!(total > 0.0f) || sampleCountSum == 0u) {
        return false;
    }
    float target = rng.nextFloat() * total;
    float selectedWeight = 0.0f;
    selectedBin = 0u;
    const size_t base = cellId * kCpuPathGuidingBinCount;
    for (uint32_t bin = 0u; bin < kCpuPathGuidingBinCount && base + bin < states.size(); ++bin) {
        const float weight = CpuPathGuidingBinWeight(states[base + bin], frameIndex);
        if (weight <= 0.0f) {
            continue;
        }
        selectedBin = bin;
        selectedWeight = weight;
        if (target <= weight) {
            break;
        }
        target -= weight;
    }
    direction = CpuPathGuidingBinCenterDirection(selectedBin, normal);
    pdf = (selectedWeight / total) * static_cast<float>(kCpuPathGuidingBinCount) *
          kCpuPathGuidingInvHemisphereSolidAngle;
    confidence = CpuPathGuidingConfidenceFromSamples(sampleCountSum);
    return pdf > 0.0f && std::isfinite(pdf) && confidence > 0.0f && IsFinite(direction);
}

bool CpuPathGuidingReweightSampleForDirection(const PathTracerShaderTypes::MaterialData& material,
                                              const simd::float3& position,
                                              const simd::float3& normal,
                                              const simd::float3& wo,
                                              const simd::float3& wi,
                                              float proposalPdf,
                                              const FireflyClampParams& clampParams,
                                              BsdfSample& sample) {
    if (!(proposalPdf > 0.0f) || !std::isfinite(proposalPdf) ||
        !IsFinite(wi) || simd::dot(wi, wi) <= 0.0f) {
        return false;
    }
    const simd::float3 axis = simd::normalize(normal);
    const simd::float3 dir = simd::normalize(wi);
    const float cosTheta = simd::dot(axis, dir);
    if (!(cosTheta > 0.0f)) {
        return false;
    }
    const BsdfEval eval = EvaluateBsdf(material, position, axis, wo, dir, clampParams);
    if (eval.isDelta || !(eval.pdf > 0.0f) || !IsFinite(eval.value)) {
        return false;
    }
    const simd::float3 weight =
        simd::max(eval.value * (cosTheta / std::max(proposalPdf, 1.0e-6f)),
                  simd_make_float3(0.0f, 0.0f, 0.0f));
    if (!IsFinite(weight)) {
        return false;
    }
    sample.direction = dir;
    sample.weight = weight;
    sample.pdf = proposalPdf;
    sample.lobeType = 0u;
    sample.lobeRoughness = 1.0f;
    sample.isDelta = false;
    return true;
}

void CpuPathGuidingUpdate(std::vector<CpuPathGuidingReservoirState>& states,
                          const PathTracer::RenderSettings& settings,
                          uint32_t frameIndex,
                          const simd::float3& position,
                          const simd::float3& normal,
                          size_t cellCount,
                          const simd::float3& direction,
                          const simd::float3& incidentRadiance,
                          float weight,
                          float guidingPdf,
                          bool guidedChoice) {
    if (states.empty() ||
        !IsFinite(direction) ||
        !IsFinite(incidentRadiance) ||
        !(weight > 0.0f) ||
        !std::isfinite(weight)) {
        return;
    }
    const uint32_t interval = std::max(settings.pathGuidingTrainingInterval, 1u);
    if ((frameIndex % interval) != 0u) {
        return;
    }
    const simd::float3 axis = simd::normalize(normal);
    if (simd::dot(axis, simd::normalize(direction)) <= 0.0f) {
        return;
    }
    const size_t cellId = CpuPathGuidingCellId(settings, position, cellCount);
    const uint32_t binId = CpuPathGuidingBinIdForDirection(direction, axis);
    const size_t index = cellId * kCpuPathGuidingBinCount + binId;
    if (index >= states.size()) {
        return;
    }
    const float sampleLuma =
        std::max(Luminance(simd::max(incidentRadiance, simd_make_float3(0.0f, 0.0f, 0.0f))) *
                     weight,
                 0.0f);
    if (!(sampleLuma > 0.0f) || !std::isfinite(sampleLuma)) {
        return;
    }
    CpuPathGuidingReservoirState& state = states[index];
    const uint32_t quantizedLuma =
        std::clamp(static_cast<uint32_t>(sampleLuma * 256.0f), 1u, 1048576u);
    const uint32_t previousCount = state.sampleCount;
    state.valid = true;
    state.sampleCount += 1u;
    const uint64_t nextLumaAccumulator =
        static_cast<uint64_t>(state.lumaAccumulator) + static_cast<uint64_t>(quantizedLuma);
    state.lumaAccumulator =
        static_cast<uint32_t>(std::min<uint64_t>(nextLumaAccumulator,
                                                 std::numeric_limits<uint32_t>::max()));
    state.confidence =
        std::clamp(static_cast<float>(previousCount + 1u) / 16.0f, 0.0625f, 1.0f);
    state.direction = CpuPathGuidingBinCenterDirection(binId, axis);
    state.sampleLuma = sampleLuma;
    state.lastLuma = sampleLuma;
    state.guidingPdf = std::max(guidingPdf, 0.0f);
    state.guidedChoice = guidedChoice;
    state.frameTag = frameIndex;
    state.flags = 1u | (guidedChoice ? 2u : 0u);
}

bool CpuApplyPathGuidingPrototype(const PathTracer::RenderSettings& settings,
                                  const PathTracerShaderTypes::MaterialData& material,
                                  const simd::float3& position,
                                  const simd::float3& normal,
                                  const simd::float3& wo,
                                  const FireflyClampParams& clampParams,
                                  uint32_t depth,
                                  uint32_t frameIndex,
                                  std::vector<CpuPathGuidingReservoirState>& states,
                                  std::vector<std::mutex>& cellMutexes,
                                  size_t cellCount,
                                  std::atomic<uint64_t>& candidateCount,
                                  std::atomic<uint64_t>& usedCount,
                                  std::atomic<uint64_t>& fallbackCount,
                                  std::atomic<uint64_t>& invalidCount,
                                  Rng& rng,
                                  BsdfSample& sample) {
    if (settings.pathGuidingMode != PathTracer::RenderSettings::PathGuidingMode::Prototype) {
        return false;
    }
    if (!CpuPathGuidingCandidateValid(sample, material, depth)) {
        fallbackCount.fetch_add(1u, std::memory_order_relaxed);
        return false;
    }
    candidateCount.fetch_add(1u, std::memory_order_relaxed);

    const float strength = std::clamp(settings.pathGuidingStrength, 0.0f, 1.0f);
    if (strength <= 0.0f || states.empty() || cellMutexes.empty() || cellCount == 0u ||
        !IsFinite(position) || !IsFinite(normal) || simd::dot(normal, normal) <= 0.0f) {
        fallbackCount.fetch_add(1u, std::memory_order_relaxed);
        return false;
    }

    const simd::float3 axis = simd::normalize(normal);
    const size_t cellId = CpuPathGuidingCellId(settings, position, cellCount);
    std::lock_guard<std::mutex> lock(cellMutexes[cellId]);

    simd::float3 guidedDirection{0.0f, 1.0f, 0.0f};
    float guidedPdf = 0.0f;
    float guidedConfidence = 0.0f;
    size_t guidedCellId = 0u;
    uint32_t guidedBinId = 0u;
    const bool hasGuidedDistribution =
        CpuPathGuidingSampleDirection(states,
                                      settings,
                                      frameIndex,
                                      position,
                                      axis,
                                      cellCount,
                                      rng,
                                      guidedDirection,
                                      guidedPdf,
                                      guidedConfidence,
                                      guidedCellId,
                                      guidedBinId);
    (void)guidedCellId;
    (void)guidedBinId;

    bool guidedChoice = false;
    float guidingPdf = 0.0f;
    float mixturePdf = sample.pdf;
    BsdfSample selectedSample = sample;
    if (hasGuidedDistribution) {
        const float guideChoiceProbability =
            std::clamp(strength * guidedConfidence, 0.0f, 0.95f);
        const float bsdfGuidingPdf =
            CpuPathGuidingPdfDirection(states,
                                       settings,
                                       frameIndex,
                                       position,
                                       axis,
                                       cellCount,
                                       sample.direction);
        const float bsdfPdf = sample.pdf;
        const float bsdfMixturePdf =
            (1.0f - guideChoiceProbability) * bsdfPdf +
            guideChoiceProbability * bsdfGuidingPdf;
        if (bsdfMixturePdf > 0.0f && std::isfinite(bsdfMixturePdf)) {
            BsdfSample bsdfMixtureSample = sample;
            if (CpuPathGuidingReweightSampleForDirection(material,
                                                         position,
                                                         axis,
                                                         wo,
                                                         sample.direction,
                                                         bsdfMixturePdf,
                                                         clampParams,
                                                         bsdfMixtureSample)) {
                selectedSample = bsdfMixtureSample;
                mixturePdf = bsdfMixturePdf;
            }
        }

        if (rng.nextFloat() < guideChoiceProbability) {
            const BsdfEval eval =
                EvaluateBsdf(material, position, axis, wo, guidedDirection, clampParams);
            const float guidedBsdfPdf = eval.pdf;
            const float guidedMixturePdf =
                (1.0f - guideChoiceProbability) * guidedBsdfPdf +
                guideChoiceProbability * guidedPdf;
            BsdfSample guidedBsdfSample = sample;
            if (CpuPathGuidingReweightSampleForDirection(material,
                                                         position,
                                                         axis,
                                                         wo,
                                                         guidedDirection,
                                                         guidedMixturePdf,
                                                         clampParams,
                                                         guidedBsdfSample)) {
                selectedSample = guidedBsdfSample;
                mixturePdf = guidedMixturePdf;
                guidingPdf = guidedPdf;
                guidedChoice = true;
            }
        }
    }

    if (!(mixturePdf > 0.0f) || !std::isfinite(mixturePdf)) {
        invalidCount.fetch_add(1u, std::memory_order_relaxed);
        return false;
    }

    sample = selectedSample;
    const float updatePdf = guidedChoice
        ? guidingPdf
        : CpuPathGuidingPdfDirection(states,
                                     settings,
                                     frameIndex,
                                     position,
                                     axis,
                                     cellCount,
                                     sample.direction);
    CpuPathGuidingUpdate(states,
                         settings,
                         frameIndex,
                         position,
                         axis,
                         cellCount,
                         sample.direction,
                         sample.weight,
                         1.0f,
                         updatePdf,
                         guidedChoice);
    if (guidedChoice) {
        usedCount.fetch_add(1u, std::memory_order_relaxed);
        return true;
    }
    if (hasGuidedDistribution) {
        fallbackCount.fetch_add(1u, std::memory_order_relaxed);
    }
    return false;
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
        float cosThetaO = std::max(simd::dot(normal, wo), 0.0f);
        simd::float3 albedo = (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Subsurface))
            ? SssDiffuseColor(material, cosThetaI, cosThetaO)
            : MaterialBaseColor(material);
        simd::float3 f = albedo / kPi;
        simd::float3 weight = f * cosThetaI / pdf;
        if (!IsFinite(weight)) {
            return result;
        }
        result.direction = wi;
        result.weight = simd::max(weight, simd_make_float3(0.0f, 0.0f, 0.0f));
        result.pdf = pdf;
        result.lobeRoughness = 1.0f;
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
        simd::float3 baseColor = MaterialBaseColor(material);
        float metallic = std::clamp(material.pbrParams.x, 0.0f, 1.0f);
        float roughness = std::clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        float dielectricF0 = DielectricF0FromIor(material.typeEta.y);
        simd::float3 f0 = baseColor * metallic +
                          simd_make_float3(dielectricF0, dielectricF0, dielectricF0) * (1.0f - metallic);
        simd::float3 diffuseColor = baseColor * (1.0f - metallic) *
                                    std::clamp(material.pbrParams.z, 0.0f, 1.0f);
        float transmission = std::clamp(material.pbrExtras.z, 0.0f, 1.0f) * (1.0f - metallic);
        float reflectScale = 1.0f - transmission;
        float specWeightBase = PbrSpecularWeight(f0);
        float wSpec = specWeightBase * reflectScale;
        float wDiff = (1.0f - specWeightBase) * reflectScale;
        float wTrans = transmission;
        float weightSum = wSpec + wDiff + wTrans;
        if (weightSum <= 0.0f) {
            return result;
        }
        float pSpec = wSpec / weightSum;
        float pDiff = wDiff / weightSum;
        float pTrans = wTrans / weightSum;

        simd::float3 wi;
        float pdfSpec = 0.0f;
        float pdfDiffuse = 0.0f;
        float pdfTrans = 0.0f;
        uint32_t sampledLobeType = 0u;
        float sampledLobeRoughness = 1.0f;
        simd::float3 f = simd_make_float3(0.0f, 0.0f, 0.0f);

        float choose = rng.nextFloat();
        if (choose < pSpec) {
            sampledLobeType = 1u;
            sampledLobeRoughness = roughness;
            if (roughness <= 1.0e-3f) {
                wi = simd::normalize(Reflect(incidentDir, normal));
                float cosThetaI = simd::dot(normal, wi);
                if (cosThetaI <= 0.0f) {
                    return result;
                }
                float cosThetaO = std::max(simd::dot(normal, wo), 0.0f);
                f = SchlickFresnel(f0, cosThetaO) * reflectScale;
                pdfSpec = 1.0f;
                result.isDelta = true;
            } else {
                float alpha = roughness * roughness;
                simd::float3 wh = SampleGgxVndf(rng, roughness, normal, wo);
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
                f *= SpecularEnergyCompensation(f0, roughness, cosThetaO);
                f = ClampSpecularTail(f, roughness, f0, clampParams);
                f *= reflectScale;
                pdfSpec = GgxPdf(alpha, normal, wo, wi);
            }
        } else if (choose < (pSpec + pDiff)) {
            float pdf = 0.0f;
            wi = SampleCosineHemisphere(rng, normal, pdf);
            pdfDiffuse = pdf;
            float cosThetaI = std::max(simd::dot(normal, wi), 0.0f);
            if (pdfDiffuse <= 0.0f || cosThetaI <= 0.0f) {
                return result;
            }
            f = (diffuseColor / kPi) * reflectScale;
        } else {
            sampledLobeType = 2u;
            sampledLobeRoughness = roughness;
            float cosThetaO = simd::dot(normal, wo);
            float absCosThetaO = std::fabs(cosThetaO);
            float etaI = 1.0f;
            float etaT = std::max(material.typeEta.y, 1.0f);
            if (cosThetaO < 0.0f) {
                std::swap(etaI, etaT);
            }
            float eta = etaI / etaT;
            if (roughness <= 1.0e-3f) {
                if (!Refract(-wo, normal, eta, wi)) {
                    return result;
                }
                wi = simd::normalize(wi);
                float cosThetaI = simd::dot(normal, wi);
                float cosThetaT = 0.0f;
                float Fr = FresnelDielectricExact(cosThetaO, etaI, etaT, cosThetaT);
                float etaScale = (etaT * etaT) / std::max(etaI * etaI, 1.0e-6f);
                float directionScale = etaScale * (std::fabs(cosThetaT) / std::max(absCosThetaO, 1.0e-6f));
                simd::float3 ft = simd_make_float3(std::max(1.0f - Fr, 0.0f) * directionScale,
                                                   std::max(1.0f - Fr, 0.0f) * directionScale,
                                                   std::max(1.0f - Fr, 0.0f) * directionScale);
                ft *= DielectricTransmissionTint(material, std::fabs(cosThetaI));
                f = transmission * ft;
                pdfTrans = 1.0f;
                result.isDelta = true;
            } else {
                simd::float3 wh = SampleGgxVndf(rng, roughness, normal, wo);
                if (!Refract(-wo, wh, eta, wi)) {
                    return result;
                }
                wi = simd::normalize(wi);
                if (simd::dot(wi, normal) * cosThetaO >= 0.0f) {
                    return result;
                }
                float cosThetaI = simd::dot(normal, wi);
                float absCosThetaI = std::fabs(cosThetaI);
                float cosThetaOWh = simd::dot(wo, wh);
                float cosThetaIWh = simd::dot(wi, wh);
                if (cosThetaOWh * cosThetaIWh > 0.0f) {
                    return result;
                }
                float alpha = std::max(roughness * roughness, 1.0e-4f);
                float D = GgxDistribution(alpha, std::max(simd::dot(normal, wh), 0.0f));
                float G = GgxG1(alpha, absCosThetaO) * GgxG1(alpha, absCosThetaI);
                float cosThetaT = 0.0f;
                float F = FresnelDielectricExact(cosThetaOWh, etaI, etaT, cosThetaT);
                float denom = cosThetaOWh + eta * cosThetaIWh;
                float denomSq = denom * denom;
                if (std::fabs(denomSq) <= 1.0e-8f) {
                    return result;
                }
                float factor = (eta * eta) * std::fabs(cosThetaIWh) * std::fabs(cosThetaOWh);
                factor /= std::max(absCosThetaO * absCosThetaI * denomSq, 1.0e-6f);
                simd::float3 ft = simd_make_float3(1.0f - F, 1.0f - F, 1.0f - F) * D * G * factor;
                ft *= DielectricTransmissionTint(material, absCosThetaI);
                f = transmission * ft;
                float pdfWh = GgxVndfPdf(alpha, normal, wo, wh);
                float dwhDwi = std::fabs((eta * eta * cosThetaIWh) / std::max(denomSq, 1.0e-8f));
                pdfTrans = pdfWh * dwhDwi;
            }
        }

        float cosThetaI = std::fabs(simd::dot(normal, wi));
        float pdf = pSpec * pdfSpec + pDiff * pdfDiffuse + pTrans * pdfTrans;
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
        result.lobeType = sampledLobeType;
        result.lobeRoughness = sampledLobeRoughness;
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
        bool sampledPlasticCoat = rng.nextFloat() < pCoat;
        if (sampledPlasticCoat) {
            simd::float3 wh = SampleGgxVndf(rng, coatRoughness, normal, wo);
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
        result.lobeType = sampledPlasticCoat ? 1u : 0u;
        result.lobeRoughness = sampledPlasticCoat ? coatRoughness : 1.0f;
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
            result.lobeType = 1u;
            result.lobeRoughness = roughness;
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
        result.lobeType = 1u;
        result.lobeRoughness = roughness;
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
        result.lobeType = (lobe == 0u) ? 0u : 1u;
        result.lobeRoughness = (lobe == 2u) ? PlasticCoatRoughness(material)
                             : (lobe == 1u) ? CarpaintFlakeRoughness(material)
                                            : CarpaintBaseRoughness(material);
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Dielectric)) {
        result.isDelta = true;
        float refIdx = std::max(material.typeEta.y, 1.0f);
        bool isThin = material.typeEta.w > 0.5f;
        float etaI = 1.0f;
        float etaT = refIdx;
        float cosThetaO = std::clamp(simd::dot(-incidentDir, normal), -1.0f, 1.0f);
        if (!isThin && !frontFace) {
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
            weight = simd_make_float3(1.0f, 1.0f, 1.0f);
        } else {
            simd::float3 refracted;
            if (!Refract(incidentDir, normal, relativeEta, refracted) ||
                simd::dot(refracted, refracted) <= 0.0f) {
                direction = Reflect(incidentDir, normal);
                weight = simd_make_float3(1.0f, 1.0f, 1.0f);
            } else {
                direction = simd::normalize(refracted);
                float etaScale = (etaT * etaT) / (etaI * etaI);
                float cosThetaI = std::fabs(cosThetaO);
                float cosThetaTrans = std::fabs(cosThetaT);
                float directionScale = etaScale * (cosThetaTrans / std::max(cosThetaI, 1.0e-6f));
                float scale = std::max(directionScale, 0.0f);
                weight = simd_make_float3(scale, scale, scale);
            }
        }

        result.direction = simd::normalize(direction);
        result.weight = weight;
        result.pdf = 1.0f;
        result.lobeType = 2u;
        result.lobeRoughness = 0.0f;
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
    result.lobeRoughness = 1.0f;
    return result;
}

bool BuildEnvironmentDistribution(EnvironmentMap& outEnv) {
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
    return outEnv.hasDistribution;
}

void BuildEnvironmentMipChain(EnvironmentMap& env) {
    env.mipLevels.clear();
    if (env.width == 0u || env.height == 0u || env.rgba.empty()) {
        return;
    }
    env.mipLevels.push_back(EnvironmentMap::MipLevel{env.rgba, env.width, env.height});
    while (env.mipLevels.back().width > 1u || env.mipLevels.back().height > 1u) {
        const EnvironmentMap::MipLevel& prev = env.mipLevels.back();
        EnvironmentMap::MipLevel next;
        next.width = std::max(prev.width / 2u, 1u);
        next.height = std::max(prev.height / 2u, 1u);
        next.rgba.assign(static_cast<size_t>(next.width) * next.height * 4u, 0.0f);
        auto fetch = [&](uint32_t x, uint32_t y, uint32_t c) {
            x %= prev.width;
            y = std::min(y, prev.height - 1u);
            return prev.rgba[(static_cast<size_t>(y) * prev.width + x) * 4ull + c];
        };
        for (uint32_t y = 0; y < next.height; ++y) {
            for (uint32_t x = 0; x < next.width; ++x) {
                const uint32_t sx = x * 2u;
                const uint32_t sy = y * 2u;
                for (uint32_t c = 0; c < 4u; ++c) {
                    const float sum =
                        fetch(sx, sy, c) +
                        fetch(sx + 1u, sy, c) +
                        fetch(sx, sy + 1u, c) +
                        fetch(sx + 1u, sy + 1u, c);
                    next.rgba[(static_cast<size_t>(y) * next.width + x) * 4ull + c] = sum * 0.25f;
                }
            }
        }
        env.mipLevels.push_back(std::move(next));
    }
}

float RgbeToFloat(uint8_t channel, uint8_t exponent) {
    if (exponent == 0u) {
        return 0.0f;
    }
    return std::ldexp(static_cast<float>(channel), static_cast<int>(exponent) - (128 + 8));
}

bool LoadRadianceHdrCpu(const std::string& path, EnvironmentMap& outEnv) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    bool validFormat = false;
    uint32_t width = 0;
    uint32_t height = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line == "FORMAT=32-bit_rle_rgbe") {
            validFormat = true;
            continue;
        }
        if (line.rfind("-Y ", 0) == 0) {
            std::istringstream stream(line);
            std::string yToken;
            std::string xToken;
            stream >> yToken >> height >> xToken >> width;
            if (yToken != "-Y" || xToken != "+X") {
                return false;
            }
            break;
        }
    }

    if (!validFormat || width == 0u || height == 0u || width > 32767u) {
        return false;
    }

    std::vector<uint8_t> rgbe(static_cast<size_t>(width) * height * 4u, 0u);
    std::vector<uint8_t> scanline(static_cast<size_t>(width) * 4u, 0u);
    for (uint32_t y = 0; y < height; ++y) {
        uint8_t header[4] = {};
        if (!file.read(reinterpret_cast<char*>(header), sizeof(header))) {
            return false;
        }
        if (header[0] != 2u || header[1] != 2u ||
            ((static_cast<uint32_t>(header[2]) << 8u) | header[3]) != width) {
            return false;
        }

        for (uint32_t channel = 0; channel < 4u; ++channel) {
            uint32_t x = 0;
            while (x < width) {
                uint8_t count = 0;
                if (!file.read(reinterpret_cast<char*>(&count), sizeof(count))) {
                    return false;
                }
                if (count > 128u) {
                    uint32_t runLength = static_cast<uint32_t>(count - 128u);
                    uint8_t value = 0;
                    if (runLength == 0u || x + runLength > width ||
                        !file.read(reinterpret_cast<char*>(&value), sizeof(value))) {
                        return false;
                    }
                    for (uint32_t i = 0; i < runLength; ++i) {
                        scanline[(x++ * 4u) + channel] = value;
                    }
                } else {
                    uint32_t literalLength = static_cast<uint32_t>(count);
                    if (literalLength == 0u || x + literalLength > width) {
                        return false;
                    }
                    for (uint32_t i = 0; i < literalLength; ++i) {
                        uint8_t value = 0;
                        if (!file.read(reinterpret_cast<char*>(&value), sizeof(value))) {
                            return false;
                        }
                        scanline[(x++ * 4u) + channel] = value;
                    }
                }
            }
        }

        std::copy(scanline.begin(),
                  scanline.end(),
                  rgbe.begin() + static_cast<size_t>(y) * width * 4u);
    }

    outEnv = EnvironmentMap{};
    outEnv.width = width;
    outEnv.height = height;
    outEnv.rgba.assign(static_cast<size_t>(width) * height * 4u, 1.0f);
    for (size_t i = 0, pixelCount = static_cast<size_t>(width) * height; i < pixelCount; ++i) {
        const uint8_t exponent = rgbe[i * 4u + 3u];
        outEnv.rgba[i * 4u + 0u] = RgbeToFloat(rgbe[i * 4u + 0u], exponent);
        outEnv.rgba[i * 4u + 1u] = RgbeToFloat(rgbe[i * 4u + 1u], exponent);
        outEnv.rgba[i * 4u + 2u] = RgbeToFloat(rgbe[i * 4u + 2u], exponent);
        outEnv.rgba[i * 4u + 3u] = 1.0f;
    }
    BuildEnvironmentMipChain(outEnv);
    BuildEnvironmentDistribution(outEnv);
    return true;
}

bool LoadEnvironmentMap(const std::string& path, EnvironmentMap& outEnv) {
    outEnv = EnvironmentMap{};
    if (path.empty()) {
        return false;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        return LoadRadianceHdrCpu(path, outEnv);
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
        return LoadRadianceHdrCpu(path, outEnv);
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

    BuildEnvironmentMipChain(outEnv);
    BuildEnvironmentDistribution(outEnv);

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
    std::vector<std::vector<simd::float2>> meshUvs1;
    std::vector<std::vector<simd::float4>> meshTangents;
    std::vector<std::vector<uint32_t>> meshIndices;
    std::vector<simd::float4> sphereData;
    std::vector<uint32_t> sphereMaterials;
    std::vector<simd::float3> rectPositions;
    std::vector<simd::float3> rectNormals;
    std::vector<uint32_t> rectIndices;
    std::vector<uint32_t> rectMaterials;
    std::vector<uint32_t> rectTriangleToRect;
    std::unordered_map<uint32_t, CpuTexture> materialTextures;
    const PathTracerShaderTypes::RectData* rectangles = nullptr;
    uint32_t rectangleCount = 0;
};

PathTracerShaderTypes::MaterialData ShadeMaterialForHit(const EmbreeSceneData& sceneData,
                                                        const PathTracerShaderTypes::MaterialData& material,
                                                        const HitInfo& hit,
                                                        float debugNormalStrengthScale,
                                                        float surfaceFootprintWorld) {
    PathTracerShaderTypes::MaterialData shaded = material;
    if (!hit.hasUv || sceneData.materialTextures.empty()) {
        return shaded;
    }
    const uint32_t type = static_cast<uint32_t>(shaded.typeEta.x);

    auto sampleTexture = [&](uint32_t index,
                             uint32_t uvSet,
                             const simd::float4& row0,
                             const simd::float4& row1,
                             bool forceMip0 = false,
                             bool blackKeyAlphaFromRgb = false,
                             bool gradientFiltered = true) -> simd::float4 {
        auto found = sceneData.materialTextures.find(index);
        if (index == kInvalidTextureIndex || found == sceneData.materialTextures.end()) {
            return simd_make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        }
        CpuTextureSamplingContext ctx = MakeCpuTextureSamplingContext(hit, uvSet, row0, row1);
        float lod = 0.0f;
        if (!forceMip0 &&
            type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
            lod = TextureLodWithFallback(found->second,
                                         ctx.hasIgehyGradients,
                                         ctx.dUVdx,
                                         ctx.dUVdy,
                                         ctx.uvPerWorld,
                                         surfaceFootprintWorld);
        }
        simd::float4 sample = (gradientFiltered && ctx.hasIgehyGradients)
            ? SampleCpuTextureAnisotropic(found->second, ctx.uv, ctx.dUVdx, ctx.dUVdy, lod)
            : SampleCpuTexture(found->second, ctx.uv, lod);
        if (blackKeyAlphaFromRgb) {
            float luma = Luminance(simd_make_float3(sample.x, sample.y, sample.z));
            sample.w = Smoothstep(0.72f, 0.90f, luma);
        }
        return sample;
    };

    simd::float4 baseColorSample = sampleTexture(shaded.textureIndices0.x,
                                                 shaded.textureUvSet0.x,
                                                 shaded.textureTransform0,
                                                 shaded.textureTransform1,
                                                 (shaded.materialFlags & PathTracerShaderTypes::kMaterialFlagForceBaseColorMip0) != 0u,
                                                 (shaded.materialFlags & PathTracerShaderTypes::kMaterialFlagBlackKeyAlphaFromRgb) != 0u);
    shaded.baseColorRoughness.x *= baseColorSample.x;
    shaded.baseColorRoughness.y *= baseColorSample.y;
    shaded.baseColorRoughness.z *= baseColorSample.z;
    shaded.pbrExtras.x = std::clamp(shaded.pbrExtras.x * baseColorSample.w, 0.0f, 1.0f);

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
        bool ormDisabled = (shaded.materialFlags & PathTracerShaderTypes::kMaterialFlagDisableOrm) != 0u;
        if (!ormDisabled) {
            simd::float4 ormSample = sampleTexture(shaded.textureIndices0.y,
                                                   shaded.textureUvSet0.y,
                                                   shaded.textureTransform2,
                                                   shaded.textureTransform3,
                                                   false,
                                                   false,
                                                   false);
            shaded.pbrParams.x = std::clamp(shaded.pbrParams.x * ormSample.z, 0.0f, 1.0f);
            shaded.pbrParams.y = std::clamp(shaded.pbrParams.y * ormSample.y, 0.0f, 1.0f);
        }
        shaded.pbrParams.y = ApplyNormalVarianceToRoughness(sceneData,
                                                            shaded,
                                                            hit,
                                                            debugNormalStrengthScale,
                                                            surfaceFootprintWorld);
        shaded.baseColorRoughness.w = shaded.pbrParams.y;
        simd::float4 occlusionSample = sampleTexture(shaded.textureIndices0.w,
                                                     shaded.textureUvSet0.w,
                                                     shaded.textureTransform6,
                                                     shaded.textureTransform7);
        float occlusionStrength = std::clamp(shaded.pbrParams.z, 0.0f, 1.0f);
        float occlusion = std::clamp(occlusionSample.x, 0.0f, 1.0f);
        shaded.pbrParams.z = 1.0f + (occlusion - 1.0f) * occlusionStrength;
        simd::float4 transmissionSample = sampleTexture(shaded.textureIndices1.y,
                                                        shaded.textureUvSet1.y,
                                                        shaded.textureTransform10,
                                                        shaded.textureTransform11);
        shaded.pbrExtras.z = std::clamp(shaded.pbrExtras.z * transmissionSample.x, 0.0f, 1.0f) *
                             (1.0f - std::clamp(shaded.pbrParams.x, 0.0f, 1.0f));
        simd::float4 emissiveSample = sampleTexture(shaded.textureIndices1.x,
                                                    shaded.textureUvSet1.x,
                                                    shaded.textureTransform8,
                                                    shaded.textureTransform9);
        shaded.emission.x *= emissiveSample.x;
        shaded.emission.y *= emissiveSample.y;
        shaded.emission.z *= emissiveSample.z;
    }

    return shaded;
}

simd::float3 ApplyNormalMapForHit(const EmbreeSceneData& sceneData,
                                  const PathTracerShaderTypes::MaterialData& material,
                                  const HitInfo& hit,
                                  const simd::float3& shadingNormal,
                                  float debugNormalStrengthScale,
                                  float surfaceFootprintWorld) {
    const uint32_t index = material.textureIndices0.z;
    auto found = sceneData.materialTextures.find(index);
    if (index == kInvalidTextureIndex || found == sceneData.materialTextures.end()) {
        return shadingNormal;
    }
    if (!hit.hasTangent || !hit.hasUv) {
        return shadingNormal;
    }
    float normalScale = std::max(material.pbrParams.w * std::max(debugNormalStrengthScale, 0.0f), 0.0f);
    if (normalScale <= 1.0e-4f) {
        return shadingNormal;
    }

    CpuTextureSamplingContext ctx = MakeCpuTextureSamplingContext(hit,
                                                                  material.textureUvSet0.z,
                                                                  material.textureTransform4,
                                                                  material.textureTransform5);
    float lod = TextureLodWithFallback(found->second,
                                       ctx.hasIgehyGradients,
                                       ctx.dUVdx,
                                       ctx.dUVdy,
                                       ctx.uvPerWorld,
                                       surfaceFootprintWorld);
    simd::float4 texel = ctx.hasIgehyGradients
        ? SampleCpuTextureAnisotropic(found->second, ctx.uv, ctx.dUVdx, ctx.dUVdy, lod)
        : SampleCpuTexture(found->second, ctx.uv, lod);
    simd::float3 normalTs = DecodeNormalMap(simd_make_float3(texel.x, texel.y, texel.z), normalScale);

    simd::float3 t = simd_make_float3(hit.tangent.x, hit.tangent.y, hit.tangent.z);
    if (!IsFinite(t) || simd::dot(t, t) <= 1.0e-6f) {
        return shadingNormal;
    }
    t = simd::normalize(t - shadingNormal * simd::dot(shadingNormal, t));
    if (!IsFinite(t) || simd::dot(t, t) <= 1.0e-6f) {
        return shadingNormal;
    }
    float tangentSign = hit.tangent.w < 0.0f ? -1.0f : 1.0f;
    simd::float3 b = simd::normalize(simd::cross(shadingNormal, t)) * tangentSign;
    if (!IsFinite(b) || simd::dot(b, b) <= 1.0e-6f) {
        return shadingNormal;
    }

    simd::float3 mapped = simd::normalize(t * normalTs.x +
                                          b * normalTs.y +
                                          shadingNormal * normalTs.z);
    if (simd::dot(mapped, hit.normal) < 0.0f) {
        mapped = -mapped;
    }
    return IsFinite(mapped) ? mapped : shadingNormal;
}

float ApplyNormalVarianceToRoughness(const EmbreeSceneData& sceneData,
                                     const PathTracerShaderTypes::MaterialData& material,
                                     const HitInfo& hit,
                                     float debugNormalStrengthScale,
                                     float surfaceFootprintWorld) {
    const uint32_t index = material.textureIndices0.z;
    auto found = sceneData.materialTextures.find(index);
    if (index == kInvalidTextureIndex || found == sceneData.materialTextures.end()) {
        return std::clamp(material.pbrParams.y, 0.0f, 1.0f);
    }
    if (!hit.hasUv) {
        return std::clamp(material.pbrParams.y, 0.0f, 1.0f);
    }

    float normalScale = std::max(material.pbrParams.w * std::max(debugNormalStrengthScale, 0.0f), 0.0f);
    if (normalScale <= 1.0e-4f) {
        return std::clamp(material.pbrParams.y, 0.0f, 1.0f);
    }

    CpuTextureSamplingContext ctx = MakeCpuTextureSamplingContext(hit,
                                                                  material.textureUvSet0.z,
                                                                  material.textureTransform4,
                                                                  material.textureTransform5);
    const CpuTexture& texture = found->second;
    if (texture.width == 0 || texture.height == 0) {
        return std::clamp(material.pbrParams.y, 0.0f, 1.0f);
    }

    float lod = TextureLodWithFallback(texture,
                                       ctx.hasIgehyGradients,
                                       ctx.dUVdx,
                                       ctx.dUVdy,
                                       ctx.uvPerWorld,
                                       surfaceFootprintWorld);
    simd::float4 raw = ctx.hasIgehyGradients
        ? SampleCpuTextureAnisotropic(texture, ctx.uv, ctx.dUVdx, ctx.dUVdy, lod)
        : SampleCpuTexture(texture, ctx.uv, lod);
    simd::float3 n = simd_make_float3(raw.x, raw.y, raw.z) * 2.0f - simd_make_float3(1.0f, 1.0f, 1.0f);
    n.x *= normalScale;
    n.y *= normalScale;
    float normalLength = simd::length(n);
    float tok = std::max((1.0f - normalLength) / std::max(normalLength, 1.0e-6f), 0.0f);
    if (ctx.hasIgehyGradients) {
        float gradMag = std::max(std::max(std::fabs(ctx.dUVdx.x), std::fabs(ctx.dUVdx.y)),
                                 std::max(std::fabs(ctx.dUVdy.x), std::fabs(ctx.dUVdy.y)));
        if (gradMag > 1.0e-6f && gradMag < 4.0f) {
            simd::float4 rawDx = SampleCpuTexture(texture, ctx.uv + ctx.dUVdx, lod);
            simd::float4 rawDy = SampleCpuTexture(texture, ctx.uv + ctx.dUVdy, lod);
            simd::float3 nDx =
                DecodeNormalMap(simd_make_float3(rawDx.x, rawDx.y, rawDx.z), normalScale);
            simd::float3 nDy =
                DecodeNormalMap(simd_make_float3(rawDy.x, rawDy.y, rawDy.z), normalScale);
            simd::float3 nTs = DecodeNormalMap(simd_make_float3(raw.x, raw.y, raw.z), normalScale);
            float varianceX = std::max(1.0f - simd::dot(nTs, nDx), 0.0f);
            float varianceY = std::max(1.0f - simd::dot(nTs, nDy), 0.0f);
            tok += 0.35f * std::max(varianceX, varianceY);
        }
    }
    float roughness = std::clamp(material.pbrParams.y, 0.0f, 1.0f);
    return std::clamp(std::sqrt(roughness * roughness + tok), 0.0f, 1.0f);
}

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

float TransformDeterminantSign(const simd::float4x4& transform) {
    simd::float3x3 linear = simd_matrix(transform.columns[0].xyz,
                                        transform.columns[1].xyz,
                                        transform.columns[2].xyz);
    return simd_determinant(linear) < 0.0f ? -1.0f : 1.0f;
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

    const auto* materials = resources.materialsData();
    const uint32_t materialCount = resources.materialCount();
    if (materials && materialCount > 0) {
        for (uint32_t i = 0; i < materialCount; ++i) {
            const auto& material = materials[i];
            const uint32_t indices[] = {
                material.textureIndices0.x,
                material.textureIndices0.y,
                material.textureIndices0.z,
                material.textureIndices0.w,
                material.textureIndices1.x,
                material.textureIndices1.y,
            };
            for (uint32_t index : indices) {
                if (index == kInvalidTextureIndex || outData.materialTextures.count(index) != 0u) {
                    continue;
                }
                PathTracer::MaterialTextureCpuData cpuData;
                if (!resources.materialTextureCpuData(index, cpuData)) {
                    continue;
                }
                CpuTexture cpuTexture;
                cpuTexture.width = cpuData.width;
                cpuTexture.height = cpuData.height;
                cpuTexture.rgba = std::move(cpuData.rgba);
                cpuTexture.mipLevels.reserve(cpuData.mipLevels.size());
                for (auto& level : cpuData.mipLevels) {
                    CpuTexture::MipLevel mip;
                    mip.width = level.width;
                    mip.height = level.height;
                    mip.rgba = std::move(level.rgba);
                    cpuTexture.mipLevels.push_back(std::move(mip));
                }
                if (cpuTexture.mipLevels.empty() && !cpuTexture.rgba.empty()) {
                    CpuTexture::MipLevel mip;
                    mip.width = cpuTexture.width;
                    mip.height = cpuTexture.height;
                    mip.rgba = cpuTexture.rgba;
                    cpuTexture.mipLevels.push_back(std::move(mip));
                }
                cpuTexture.samplerDesc = cpuData.samplerDesc;
                outData.materialTextures.emplace(index, std::move(cpuTexture));
            }
        }
    }

    const auto& meshes = resources.meshes();
    outData.meshPositions.reserve(meshes.size());
    outData.meshNormals.reserve(meshes.size());
    outData.meshUvs.reserve(meshes.size());
    outData.meshIndices.reserve(meshes.size());

    uint32_t meshTriangleOffset = 0u;
    for (const auto& mesh : meshes) {
        if (mesh.vertices.empty() || mesh.indices.empty()) {
            continue;
        }

        const simd::float3x3 normalMatrix = NormalMatrix(mesh.localToWorld);
        const float transformDetSign = TransformDeterminantSign(mesh.localToWorld);

        std::vector<simd::float3> positions;
        std::vector<simd::float3> normals;
        std::vector<simd::float2> uvs;
        std::vector<simd::float2> uvs1;
        std::vector<simd::float4> tangents;
        positions.reserve(mesh.vertices.size());
        normals.reserve(mesh.vertices.size());
        uvs.reserve(mesh.vertices.size());
        uvs1.reserve(mesh.vertices.size());
        tangents.reserve(mesh.vertices.size());

        for (const auto& vertex : mesh.vertices) {
            positions.push_back(TransformPoint(mesh.localToWorld, vertex.position));
            simd::float3 n = TransformNormal(normalMatrix, vertex.normal);
            normals.push_back(simd::length(n) > 0.0f ? simd::normalize(n) : n);
            uvs.push_back(vertex.uv);
            uvs1.push_back(vertex.uv1);
            simd::float3 t = TransformNormal(normalMatrix,
                                             simd_make_float3(vertex.tangent.x,
                                                              vertex.tangent.y,
                                                              vertex.tangent.z));
            if (simd::length(t) > 0.0f) {
                t = simd::normalize(t);
            }
            tangents.push_back(simd_make_float4(t.x, t.y, t.z, vertex.tangent.w * transformDetSign));
        }

        outData.meshPositions.emplace_back(std::move(positions));
        outData.meshNormals.emplace_back(std::move(normals));
        outData.meshUvs.emplace_back(std::move(uvs));
        outData.meshUvs1.emplace_back(std::move(uvs1));
        outData.meshTangents.emplace_back(std::move(tangents));
        outData.meshIndices.emplace_back(mesh.indices);

        auto geomData = std::make_unique<GeometryData>();
        geomData->type = GeometryType::Mesh;
        geomData->positions = outData.meshPositions.back().data();
        geomData->normals = outData.meshNormals.back().data();
        geomData->uvs = outData.meshUvs.back().data();
        geomData->uvs1 = outData.meshUvs1.back().data();
        geomData->tangents = outData.meshTangents.back().data();
        geomData->indices = outData.meshIndices.back().data();
        geomData->indexCount = static_cast<uint32_t>(outData.meshIndices.back().size());
        geomData->materialIndex = mesh.materialIndex;
        geomData->triangleOffset = meshTriangleOffset;

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
        meshTriangleOffset += triCount;
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
        outData.rectPositions.reserve(resources.rectangleCount() * 16u);
        outData.rectNormals.reserve(resources.rectangleCount() * 16u);
        outData.rectIndices.reserve(resources.rectangleCount() * 48u);
        outData.rectMaterials.reserve(resources.rectangleCount() * 16u);
        outData.rectTriangleToRect.reserve(resources.rectangleCount() * 16u);

        for (uint32_t i = 0; i < resources.rectangleCount(); ++i) {
            const PathTracerShaderTypes::RectData& rect = rectangles[i];
            simd::float3 corner = {rect.corner.x, rect.corner.y, rect.corner.z};
            simd::float3 edgeU = {rect.edgeU.x, rect.edgeU.y, rect.edgeU.z};
            simd::float3 edgeV = {rect.edgeV.x, rect.edgeV.y, rect.edgeV.z};
            simd::float3 normal = {rect.normalAndPlane.x, rect.normalAndPlane.y, rect.normalAndPlane.z};
            normal = simd::normalize(normal);

            if (AreaPrimitiveIsDisk(rect)) {
                constexpr uint32_t kDiskSegments = 24u;
                uint32_t centerIndex = static_cast<uint32_t>(outData.rectPositions.size());
                outData.rectPositions.push_back(corner);
                outData.rectNormals.push_back(normal);

                for (uint32_t seg = 0; seg < kDiskSegments; ++seg) {
                    float angle = (2.0f * static_cast<float>(kPi) * static_cast<float>(seg)) /
                                  static_cast<float>(kDiskSegments);
                    simd::float3 ringPoint = corner + std::cos(angle) * edgeU + std::sin(angle) * edgeV;
                    outData.rectPositions.push_back(ringPoint);
                    outData.rectNormals.push_back(normal);
                }

                simd::float3 geomNormal = simd::normalize(simd::cross(edgeU, edgeV));
                bool flipWinding = simd::dot(geomNormal, normal) < 0.0f;
                for (uint32_t seg = 0; seg < kDiskSegments; ++seg) {
                    uint32_t ring0 = centerIndex + 1u + seg;
                    uint32_t ring1 = centerIndex + 1u + ((seg + 1u) % kDiskSegments);
                    if (!flipWinding) {
                        outData.rectIndices.push_back(centerIndex);
                        outData.rectIndices.push_back(ring0);
                        outData.rectIndices.push_back(ring1);
                    } else {
                        outData.rectIndices.push_back(centerIndex);
                        outData.rectIndices.push_back(ring1);
                        outData.rectIndices.push_back(ring0);
                    }
                    outData.rectMaterials.push_back(rect.materialTwoSided.x);
                    outData.rectTriangleToRect.push_back(i);
                }
            } else {
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
        outHit.primitiveIndex = geomData->triangleOffset + primID;
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
            if (geomData->uvs) {
                simd::float2 uv0 = geomData->uvs[i0];
                simd::float2 uv1 = geomData->uvs[i1];
                simd::float2 uv2 = geomData->uvs[i2];
                outHit.uv = w * uv0 + u * uv1 + v * uv2;
                outHit.hasUv = true;
                if (geomData->positions) {
                    simd::float3 p0 = geomData->positions[i0];
                    simd::float3 p1 = geomData->positions[i1];
                    simd::float3 p2 = geomData->positions[i2];
                    outHit.hasSurfacePartials0 =
                        ComputeTriangleSurfacePartials(p0,
                                                       p1,
                                                       p2,
                                                       uv0,
                                                       uv1,
                                                       uv2,
                                                       outHit.dPdu0,
                                                       outHit.dPdv0,
                                                       outHit.uvPerWorld0);
                }
            }
            if (geomData->uvs1) {
                simd::float2 uv10 = geomData->uvs1[i0];
                simd::float2 uv11 = geomData->uvs1[i1];
                simd::float2 uv12 = geomData->uvs1[i2];
                outHit.uv1 = w * uv10 + u * uv11 + v * uv12;
                outHit.hasUv1 = true;
                if (geomData->positions) {
                    simd::float3 p0 = geomData->positions[i0];
                    simd::float3 p1 = geomData->positions[i1];
                    simd::float3 p2 = geomData->positions[i2];
                    outHit.hasSurfacePartials1 =
                        ComputeTriangleSurfacePartials(p0,
                                                       p1,
                                                       p2,
                                                       uv10,
                                                       uv11,
                                                       uv12,
                                                       outHit.dPdu1,
                                                       outHit.dPdv1,
                                                       outHit.uvPerWorld1);
                }
            }
            if (geomData->tangents) {
                simd::float4 t0 = geomData->tangents[i0];
                simd::float4 t1 = geomData->tangents[i1];
                simd::float4 t2 = geomData->tangents[i2];
                outHit.tangent = w * t0 + u * t1 + v * t2;
                outHit.hasTangent = IsFinite4(outHit.tangent);
            } else if (geomData->positions && geomData->uvs) {
                simd::float3 p0 = geomData->positions[i0];
                simd::float3 p1 = geomData->positions[i1];
                simd::float3 p2 = geomData->positions[i2];
                simd::float2 uv0 = geomData->uvs[i0];
                simd::float2 uv1 = geomData->uvs[i1];
                simd::float2 uv2 = geomData->uvs[i2];
                simd::float3 edge1 = p1 - p0;
                simd::float3 edge2 = p2 - p0;
                simd::float2 dUv1 = uv1 - uv0;
                simd::float2 dUv2 = uv2 - uv0;
                float det = dUv1.x * dUv2.y - dUv1.y * dUv2.x;
                if (std::fabs(det) > 1.0e-8f) {
                    float invDet = 1.0f / det;
                    simd::float3 tangent = (edge1 * dUv2.y - edge2 * dUv1.y) * invDet;
                    simd::float3 bitangent = (edge2 * dUv1.x - edge1 * dUv2.x) * invDet;
                    if (IsFinite(tangent) && IsFinite(bitangent) &&
                        simd::dot(tangent, tangent) > 1.0e-6f &&
                        simd::dot(bitangent, bitangent) > 1.0e-6f) {
                        tangent = simd::normalize(tangent - shadingNormal * simd::dot(shadingNormal, tangent));
                        if (simd::dot(tangent, tangent) > 1.0e-6f) {
                            float sign = (simd::dot(simd::cross(shadingNormal, tangent), bitangent) < 0.0f) ? -1.0f : 1.0f;
                            outHit.tangent = simd_make_float4(tangent.x, tangent.y, tangent.z, sign);
                            outHit.hasTangent = true;
                        }
                    }
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

bool IntersectSceneSkippingAlpha(const EmbreeSceneData& sceneData,
                                 const Ray& ray,
                                 float maxT,
                                 const PathTracerShaderTypes::MaterialData* materials,
                                 uint32_t materialCount,
                                 float debugNormalStrengthScale,
                                 HitInfo& outHit) {
    Ray currentRay = ray;
    float traveled = 0.0f;
    constexpr uint32_t kMaxAlphaSkips = 128u;
    for (uint32_t i = 0; i < kMaxAlphaSkips; ++i) {
        HitInfo hit;
        if (!IntersectScene(sceneData, currentRay, hit)) {
            return false;
        }
        traveled += hit.t;
        if (!(traveled < maxT)) {
            return false;
        }
        if (!materials || hit.materialIndex >= materialCount) {
            hit.t = traveled;
            outHit = hit;
            return true;
        }
        PathTracerShaderTypes::MaterialData material =
            ShadeMaterialForHit(sceneData,
                                materials[hit.materialIndex],
                                hit,
                                debugNormalStrengthScale,
                                0.0f);
        if (!MaterialAlphaMaskDiscards(material)) {
            hit.t = traveled;
            outHit = hit;
            return true;
        }
        currentRay.origin = OffsetRayOrigin(hit, currentRay.direction);
    }
    return false;
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
                                    bool statsMode,
                                    bool captureAuxBuffers,
                                    HeadlessRenderOutput& out,
                                    std::string& error) {
    (void)captureAuxBuffers;
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
        if (!LoadEnvironmentMap(settings.environmentMapPath, envMap)) {
            error = "Failed to load environment map for Embree backend: " + settings.environmentMapPath;
            return false;
        }
        envPtr = &envMap;
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
    const auto& emissivePrimitives = scene.resources->emissivePrimitives();

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
            info.isDisk = AreaPrimitiveIsDisk(rect);
            info.baseEmission = baseEmission;
            // Disable emitEnv for Embree rect lights to match SWRT energy during parity checks.
            info.emissionUsesEnv = false;
            info.area = AreaPrimitiveArea(rect);

            rectLightIndexByRect[i] = static_cast<int32_t>(rectLights.size());
            rectLights.push_back(info);
        }
    }
    const uint32_t rectLightCount = static_cast<uint32_t>(rectLights.size());

    std::vector<DirectionalLightInfo> directionalLights;
    {
        directionalLights.reserve(emissivePrimitives.size());
        for (const auto& primitive : emissivePrimitives) {
            if ((primitive.flags & PathTracerShaderTypes::kLightPrimitiveFlagDirectional) == 0u) {
                continue;
            }
            const float dirLen = simd::length(primitive.centroid);
            if (!(dirLen > 1.0e-6f) || !std::isfinite(dirLen) || !IsFinite(primitive.emission)) {
                continue;
            }
            if (!(simd::dot(primitive.emission, primitive.emission) > 0.0f)) {
                continue;
            }

            DirectionalLightInfo info;
            info.direction = primitive.centroid / dirLen;
            info.emission = primitive.emission;
            directionalLights.push_back(info);
        }
    }

    std::vector<MeshLightInfo> meshLights;
    {
        float totalMeshLightPower = 0.0f;
        meshLights.reserve(emissivePrimitives.size());
        for (const auto& primitive : emissivePrimitives) {
            if ((primitive.flags & PathTracerShaderTypes::kLightPrimitiveFlagDirectional) != 0u) {
                continue;
            }
            simd::float3 emission = primitive.emission * emissionScale;
            const float emissionLuma = Luminance(emission);
            const float power = primitive.area * emissionLuma;
            if (primitive.triangleIndex == std::numeric_limits<uint32_t>::max() ||
                !(primitive.area > 0.0f) ||
                !(power > 0.0f) ||
                !std::isfinite(power) ||
                !IsFinite(emission) ||
                !IsFinite(primitive.normal) ||
                !IsFinite(primitive.edge1) ||
                !IsFinite(primitive.edge2)) {
                continue;
            }

            MeshLightInfo info;
            info.primitiveIndex = primitive.triangleIndex;
            info.centroid = primitive.centroid;
            info.edge1 = primitive.edge1;
            info.edge2 = primitive.edge2;
            info.normal = primitive.normal;
            info.emission = emission;
            info.area = primitive.area;
            info.power = power;
            totalMeshLightPower += power;
            meshLights.push_back(info);
        }

        if (totalMeshLightPower > 0.0f && std::isfinite(totalMeshLightPower)) {
            float cumulativePower = 0.0f;
            for (auto& light : meshLights) {
                light.selectionPdf = light.power / totalMeshLightPower;
                cumulativePower += light.selectionPdf;
                light.cumulativePower = std::min(cumulativePower, 1.0f);
            }
            if (!meshLights.empty()) {
                meshLights.back().cumulativePower = 1.0f;
            }
        } else {
            meshLights.clear();
        }
    }

    const bool envMapAvailable = envPtr && envPtr->width > 0 && envPtr->height > 0 && !envPtr->rgba.empty();
    const bool envSampling = envPtr && envPtr->hasDistribution && envMapAvailable;
    const bool restirDirectMode =
        settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RestirDi ||
        settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RestirDiRegirHybrid;
    const FireflyClampParams clampParams = MakeFireflyParams(settings);

    CameraBasis camera = BuildCamera(settings, width, height);
    const RayCone primaryCone = MakePrimaryRayCone(camera, width, height);
    const PrimaryRayDiff primaryRayDiff = MakePrimaryRayDiff(camera, width, height);
    DirectDebugConfig debugCfg = LoadDirectDebugConfig(settings, width, height);

    std::vector<float> linearRGB(width * height * 3u, 0.0f);
    const bool collectAlbedoGuide = settings.denoiseEnabled && settings.denoiseUseAlbedo;
    const bool collectNormalGuide = settings.denoiseEnabled && settings.denoiseUseNormal;
    std::vector<float> albedoLinearRGB;
    std::vector<float> normalLinearRGB;
    if (collectAlbedoGuide) {
        albedoLinearRGB.assign(width * height * 3u, 0.0f);
    }
    if (collectNormalGuide) {
        normalLinearRGB.assign(width * height * 3u, 0.0f);
    }
    const uint32_t targetSamples = std::max<uint32_t>(1u, sppTotal);

    uint32_t seedBase = settings.fixedRngSeed;

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
    std::mutex metricsMutex;
    std::atomic<uint64_t> approxPathSegmentCount{0};
    std::atomic<uint64_t> approxShadowRayCount{0};
    CFAbsoluteTime lastStatsPrint = renderStart;
    uint64_t lastStatsRayCount = 0;

    uint64_t throughputNanInfCount = 0;
    uint64_t radianceNanInfCount = 0;
    uint64_t clampEventCount = 0;
    HeadlessRenderOutput::MaterialMrDebugDump materialMrDebug{};
    bool materialMrDebugCaptured = false;
    HeadlessRenderOutput::DirectLightAuditDump directLightAudit{};
    bool directLightAuditCaptured = false;
    std::vector<float> maxThroughputByBounce;
    if (settings.maxDepth > 0) {
        maxThroughputByBounce.assign(settings.maxDepth, 0.0f);
    }

    std::vector<simd::float3> restirDiDirectSum;
    uint64_t restirDiInitialCandidateDispatchCount = 0;
    uint64_t restirDiTemporalReuseDispatchCount = 0;
    uint64_t restirDiSpatialReuseDispatchCount = 0;
    uint64_t restirDiVisibilityDispatchCount = 0;
    uint64_t restirDiApplyLightingDispatchCount = 0;
    if (restirDirectMode) {
        const size_t pixelCount = static_cast<size_t>(width) * height;
        restirDiDirectSum.assign(pixelCount, simd_make_float3(0.0f, 0.0f, 0.0f));
        std::vector<CpuRestirDiReservoirState> currentReservoirs(pixelCount);
        std::vector<CpuRestirDiReservoirState> previousReservoirs(pixelCount);
        std::vector<CpuRestirDiReservoirState> temporalReservoirs(pixelCount);
        std::vector<CpuRestirDiReservoirState> spatialReservoirs(pixelCount);
        std::vector<CpuRestirDiVisibilityState> visibilityStates(pixelCount);

        auto makeRestirSurface = [&](uint32_t x,
                                     uint32_t y,
                                     HitInfo& hit,
                                     PathTracerShaderTypes::MaterialData& material,
                                     simd::float3& shadingNormal,
                                     simd::float3& wo) -> bool {
            Rng centerRng;
            centerRng.state = Rng::Hash(seedBase ^
                                        (y * width + x) ^
                                        0xD1B54A35u);
            Ray centerRay = GenerateCameraRay(camera, width, height, x, y, centerRng, true);
            if (!IntersectSceneSkippingAlpha(sceneData,
                                             centerRay,
                                             std::numeric_limits<float>::infinity(),
                                             materials,
                                             materialCount,
                                             settings.debugNormalStrengthScale,
                                             hit) ||
                hit.materialIndex >= materialCount) {
                return false;
            }
            PopulateFirstHitUvGradients(hit, centerRay, primaryRayDiff);
            float surfaceFootprintWorld = PrimarySurfaceFootprint(primaryCone, hit, centerRay);
            material = ShadeMaterialForHit(sceneData,
                                           materials[hit.materialIndex],
                                           hit,
                                           settings.debugNormalStrengthScale,
                                           surfaceFootprintWorld);
            if (MaterialAlphaMaskDiscards(material)) {
                return false;
            }
            const uint32_t type = static_cast<uint32_t>(material.typeEta.x);
            if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::DiffuseLight)) {
                return false;
            }
            shadingNormal = hit.shadingNormal;
            if (simd::dot(shadingNormal, shadingNormal) <= 0.0f) {
                shadingNormal = hit.normal;
            }
            if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Dielectric) ||
                MaterialUsesGeometricNormal(material)) {
                shadingNormal = hit.frontFace ? hit.normal : -hit.normal;
            }
            if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
                shadingNormal = ApplyNormalMapForHit(sceneData,
                                                     material,
                                                     hit,
                                                     shadingNormal,
                                                     settings.debugNormalStrengthScale,
                                                     surfaceFootprintWorld);
            }
            if (!IsFinite(shadingNormal) || simd::dot(shadingNormal, shadingNormal) <= 0.0f) {
                return false;
            }
            shadingNormal = simd::normalize(shadingNormal);
            wo = -centerRay.direction;
            if (!IsFinite(wo) || simd::dot(wo, wo) <= 0.0f) {
                return false;
            }
            wo = simd::normalize(wo);
            return true;
        };

        auto previousHistoryValid = [](const CpuRestirDiReservoirState& current,
                                       const CpuRestirDiReservoirState& previous,
                                       uint32_t frameIndex) {
            if (!CpuRestirDiReservoirValid(current) ||
                !CpuRestirDiReservoirValid(previous)) {
                return false;
            }
            if (frameIndex == 0u || previous.frameTag + 1u != frameIndex) {
                return false;
            }
            if (current.primitiveIndex != previous.primitiveIndex ||
                current.flags != previous.flags) {
                return false;
            }
            if (!(current.targetPdf > 0.0f) || !(previous.targetPdf > 0.0f)) {
                return false;
            }
            float targetRatio = std::min(current.targetPdf, previous.targetPdf) /
                                std::max(std::max(current.targetPdf, previous.targetPdf), 1.0e-6f);
            return targetRatio >= 0.25f;
        };

        auto mergeTemporal = [&](const CpuRestirDiReservoirState& current,
                                 const CpuRestirDiReservoirState& previous,
                                 uint32_t frameIndex) {
            CpuRestirDiReservoirState out = current;
            if (!previousHistoryValid(current, previous, frameIndex)) {
                out.temporalAccepted = 0u;
                out.stateFlags &= ~2u;
                return out;
            }
            float mergedWeight = current.weight + previous.weight;
            if (!(mergedWeight > 0.0f) || !std::isfinite(mergedWeight)) {
                out.temporalAccepted = 0u;
                out.stateFlags &= ~2u;
                return out;
            }
            out.weight = mergedWeight;
            out.candidateCount = std::max(1u, current.candidateCount + previous.candidateCount);
            out.temporalAccepted = 1u;
            out.stateFlags |= 2u;
            out.frameTag = frameIndex;
            return out;
        };

        auto payloadFromRestirState = [](const CpuRestirDiReservoirState& state) {
            CpuRisSamplePayload payload;
            payload.primitiveIndex = state.primitiveIndex;
            payload.flags = state.flags;
            payload.position = CpuRisPayloadIsDirectional(payload)
                ? simd::normalize(state.position)
                : state.position;
            payload.normal = state.normal;
            payload.emission = state.emission;
            payload.pdf = state.pdf;
            return payload;
        };

        auto reservoirFromRestirState = [&](const CpuRestirDiReservoirState& state) {
            CpuReservoir reservoir = MakeEmptyCpuReservoir();
            if (!CpuRestirDiReservoirValid(state)) {
                return reservoir;
            }
            reservoir.winner = payloadFromRestirState(state);
            reservoir.wSum = state.weight;
            reservoir.M = std::max(state.candidateCount, 1u);
            reservoir.valid = true;
            return reservoir;
        };

        for (uint32_t s = 0; s < targetSamples; ++s) {
            std::fill(currentReservoirs.begin(),
                      currentReservoirs.end(),
                      MakeEmptyCpuRestirDiReservoir(s));
            std::fill(temporalReservoirs.begin(),
                      temporalReservoirs.end(),
                      MakeEmptyCpuRestirDiReservoir(s));
            std::fill(spatialReservoirs.begin(),
                      spatialReservoirs.end(),
                      MakeEmptyCpuRestirDiReservoir(s));
            std::fill(visibilityStates.begin(),
                      visibilityStates.end(),
                      MakeEmptyCpuRestirDiVisibility(s));

            ++restirDiInitialCandidateDispatchCount;
            tbb::parallel_for(uint32_t(0), height, [&](uint32_t y) {
                for (uint32_t x = 0; x < width; ++x) {
                    const size_t pixelIndex = static_cast<size_t>(y) * width + x;
                    HitInfo hit;
                    PathTracerShaderTypes::MaterialData material;
                    simd::float3 shadingNormal;
                    simd::float3 wo;
                    if (!makeRestirSurface(x, y, hit, material, shadingNormal, wo)) {
                        continue;
                    }
                    Rng rng;
                    rng.state = seedBase ^
                                (s * 9781u) ^
                                (x * 6271u) ^
                                (y * 13007u) ^
                                0xD1B54A35u;
                    CpuReservoir reservoir;
                    CpuBuildRestirDiProviderReservoirForHit(directionalLights,
                                                            meshLights,
                                                            rectLights,
                                                            envPtr,
                                                            settings,
                                                            hit,
                                                            material,
                                                            shadingNormal,
                                                            wo,
                                                            clampParams,
                                                            std::max(settings.risCandidateCount, 1u),
                                                            rng,
                                                            reservoir);
                    currentReservoirs[pixelIndex] =
                        CpuRestirDiStateFromReservoir(hit,
                                                      material,
                                                      shadingNormal,
                                                      wo,
                                                      clampParams,
                                                      s,
                                                      reservoir);
                }
            });

            ++restirDiTemporalReuseDispatchCount;
            tbb::parallel_for(size_t(0), pixelCount, [&](size_t pixelIndex) {
                temporalReservoirs[pixelIndex] =
                    mergeTemporal(currentReservoirs[pixelIndex], previousReservoirs[pixelIndex], s);
            });

            ++restirDiSpatialReuseDispatchCount;
            const uint32_t spatialNeighborCount = std::min(std::max(settings.spatialReuseNeighborCount, 1u), 4u);
            tbb::parallel_for(uint32_t(0), height, [&](uint32_t y) {
                static constexpr int kOffsets[4][2] = {
                    {-1, 0},
                    {1, 0},
                    {0, -1},
                    {0, 1},
                };
                for (uint32_t x = 0; x < width; ++x) {
                    const size_t pixelIndex = static_cast<size_t>(y) * width + x;
                    CpuRestirDiReservoirState stateReservoir = temporalReservoirs[pixelIndex];
                    CpuReservoir reservoir = reservoirFromRestirState(stateReservoir);
                    HitInfo hit;
                    PathTracerShaderTypes::MaterialData material;
                    simd::float3 shadingNormal;
                    simd::float3 wo;
                    const bool currentSurfaceValid =
                        makeRestirSurface(x, y, hit, material, shadingNormal, wo);
                    for (uint32_t n = 0; n < spatialNeighborCount; ++n) {
                        int nx = static_cast<int>(x) + kOffsets[n][0];
                        int ny = static_cast<int>(y) + kOffsets[n][1];
                        if (nx < 0 || ny < 0 ||
                            nx >= static_cast<int>(width) ||
                            ny >= static_cast<int>(height)) {
                            continue;
                        }
                        const size_t neighborIndex = static_cast<size_t>(ny) * width +
                                                     static_cast<uint32_t>(nx);
                        const CpuRestirDiReservoirState& neighborState =
                            temporalReservoirs[neighborIndex];
                        if (!currentSurfaceValid ||
                            !CpuRestirDiReservoirValid(neighborState) ||
                            !reservoir.valid) {
                            continue;
                        }
                        HitInfo neighborHit;
                        PathTracerShaderTypes::MaterialData neighborMaterial;
                        simd::float3 neighborNormal;
                        simd::float3 neighborWo;
                        if (!makeRestirSurface(static_cast<uint32_t>(nx),
                                               static_cast<uint32_t>(ny),
                                               neighborHit,
                                               neighborMaterial,
                                               neighborNormal,
                                               neighborWo)) {
                            continue;
                        }
                        float depthRel = std::fabs(hit.t - neighborHit.t) / std::max(hit.t, kEpsilon);
                        if (!(depthRel <= 0.1f) || !std::isfinite(depthRel)) {
                            continue;
                        }
                        float normalDot = simd::dot(simd::normalize(shadingNormal),
                                                    simd::normalize(neighborNormal));
                        if (!(normalDot >= 0.9f) || !std::isfinite(normalDot)) {
                            continue;
                        }
                        CpuReservoir neighborReservoir = reservoirFromRestirState(neighborState);
                        if (!neighborReservoir.valid) {
                            continue;
                        }
                        float phatCurrent =
                            CpuReservoirWinnerPHatForHit(hit,
                                                         material,
                                                         shadingNormal,
                                                         wo,
                                                         clampParams,
                                                         neighborReservoir.winner);
                        float phatNeighbor =
                            CpuReservoirWinnerPHatForHit(neighborHit,
                                                         neighborMaterial,
                                                         neighborNormal,
                                                         neighborWo,
                                                         clampParams,
                                                         neighborReservoir.winner);
                        float denom = phatCurrent * static_cast<float>(std::max(reservoir.M, 1u)) +
                                      phatNeighbor * static_cast<float>(std::max(neighborReservoir.M, 1u));
                        if (!(denom > 0.0f) || !std::isfinite(denom) || !(phatCurrent > 0.0f)) {
                            continue;
                        }
                        float mis = phatCurrent / denom;
                        float mergeWeight = mis * neighborReservoir.wSum;
                        if (!(mergeWeight > 0.0f) || !std::isfinite(mergeWeight)) {
                            continue;
                        }
                        Rng spatialRng;
                        spatialRng.state = Rng::Hash(seedBase ^
                                                      (s * 9781u) ^
                                                      (x * 1664525u) ^
                                                      (y * 1013904223u) ^
                                                      ((n + 1u) * 747796405u));
                        CpuUpdateReservoir(reservoir,
                                           neighborReservoir.winner,
                                           mergeWeight,
                                           spatialRng.nextFloat());
                    }
                    if (currentSurfaceValid && reservoir.valid) {
                        spatialReservoirs[pixelIndex] =
                            CpuRestirDiStateFromReservoir(hit,
                                                          material,
                                                          shadingNormal,
                                                          wo,
                                                          clampParams,
                                                          s,
                                                          reservoir);
                        spatialReservoirs[pixelIndex].spatialAccepted =
                            std::max(spatialReservoirs[pixelIndex].spatialAccepted, 1u);
                        spatialReservoirs[pixelIndex].stateFlags |= 4u;
                    } else {
                        spatialReservoirs[pixelIndex] = stateReservoir;
                    }
                }
            });

            ++restirDiVisibilityDispatchCount;
            tbb::parallel_for(uint32_t(0), height, [&](uint32_t y) {
                for (uint32_t x = 0; x < width; ++x) {
                    const size_t pixelIndex = static_cast<size_t>(y) * width + x;
                    CpuRestirDiReservoirState reservoir = spatialReservoirs[pixelIndex];
                    previousReservoirs[pixelIndex] = reservoir;
                    visibilityStates[pixelIndex] = MakeEmptyCpuRestirDiVisibility(s);
                    if (!CpuRestirDiReservoirValid(reservoir)) {
                        continue;
                    }
                    HitInfo hit;
                    PathTracerShaderTypes::MaterialData material;
                    simd::float3 shadingNormal;
                    simd::float3 wo;
                    if (!makeRestirSurface(x, y, hit, material, shadingNormal, wo)) {
                        continue;
                    }
                    CpuRisSamplePayload winner;
                    winner.primitiveIndex = reservoir.primitiveIndex;
                    winner.flags = reservoir.flags;
                    winner.position = CpuRisPayloadIsDirectional(winner)
                        ? simd::normalize(reservoir.position)
                        : reservoir.position;
                    winner.normal = reservoir.normal;
                    winner.emission = reservoir.emission;
                    winner.pdf = reservoir.pdf;

                    simd::float3 omegaL = CpuRisPayloadOmegaL(hit, winner);
                    float cosThetaSurface = std::clamp(simd::dot(shadingNormal, omegaL), 0.0f, 1.0f);
                    float distSq = CpuRisPayloadDistanceSq(hit, winner);
                    if (!(cosThetaSurface > 0.0f) || !(distSq > 0.0f)) {
                        continue;
                    }
                    BsdfEval bsdfEval = EvaluateBsdf(material,
                                                     hit.position,
                                                     shadingNormal,
                                                     wo,
                                                     omegaL,
                                                     clampParams);
                    if (bsdfEval.isDelta) {
                        continue;
                    }
                    float shadowMax = CpuRisPayloadIsDirectional(winner)
                        ? std::numeric_limits<float>::infinity()
                        : std::max(simd::length(winner.position - hit.position) - kEpsilon, kEpsilon);
                    HitInfo shadowHit;
                    Ray shadowRay;
                    shadowRay.origin = OffsetRayOrigin(hit, omegaL);
                    shadowRay.direction = omegaL;
                    bool occluded =
                        IntersectSceneSkippingAlpha(sceneData,
                                                    shadowRay,
                                                    shadowMax,
                                                    materials,
                                                    materialCount,
                                                    settings.debugNormalStrengthScale,
                                                    shadowHit);
                    if (occluded && CpuShadowHitIsSelectedLight(shadowHit, winner)) {
                        occluded = false;
                    }
                    float targetPdf = std::max(reservoir.targetPdf, 1.0e-6f);
                    float reservoirW =
                        (reservoir.weight / static_cast<float>(std::max(reservoir.candidateCount, 1u))) /
                        targetPdf;
                    CpuRestirDiVisibilityState visibility = MakeEmptyCpuRestirDiVisibility(s);
                    visibility.primitiveIndex = reservoir.primitiveIndex;
                    visibility.flags = reservoir.flags;
                    visibility.stateFlags = reservoir.stateFlags;
                    if (!occluded && reservoirW > 0.0f) {
                        visibility.contribution =
                            bsdfEval.value * winner.emission * (cosThetaSurface * reservoirW);
                        if (!IsFinite(visibility.contribution)) {
                            visibility.contribution = simd_make_float3(0.0f, 0.0f, 0.0f);
                        }
                        visibility.visible = 1.0f;
                    }
                    visibilityStates[pixelIndex] = visibility;
                    restirDiDirectSum[pixelIndex] +=
                        simd::max(visibility.contribution,
                                  simd_make_float3(0.0f, 0.0f, 0.0f));
                }
            });
            ++restirDiApplyLightingDispatchCount;
        }
    }

    constexpr uint32_t kCpuPathReservoirDepthCount = 8u;
    const size_t pixelCount = static_cast<size_t>(width) * height;
    const bool restirGiActive =
        settings.restirGiMode == PathTracer::RenderSettings::RestirGiMode::DiffusePrototype;
    const bool restirPtActive =
        settings.restirPtMode == PathTracer::RenderSettings::RestirPtMode::Research ||
        settings.restirPtMode == PathTracer::RenderSettings::RestirPtMode::ExperimentalPathReuse;
    const bool restirPtReuseActive =
        settings.restirPtMode == PathTracer::RenderSettings::RestirPtMode::ExperimentalPathReuse;
    const bool pathGuidingActive =
        settings.pathGuidingMode == PathTracer::RenderSettings::PathGuidingMode::Prototype;
    const bool radianceCacheActive =
        settings.radianceCacheMode == PathTracer::RenderSettings::RadianceCacheMode::Prototype;
    std::vector<CpuPathReservoirState> restirGiReservoirs(
        restirGiActive ? pixelCount : 0u);
    std::vector<CpuPathReservoirState> restirPtReservoirs(
        restirPtActive ? pixelCount * kCpuPathReservoirDepthCount : 0u);
    const size_t pathGuidingCellCount =
        pathGuidingActive ? CpuPathGuidingCellCount(width, height) : 0u;
    std::vector<CpuPathGuidingReservoirState> pathGuidingStates(
        pathGuidingActive ? pathGuidingCellCount * kCpuPathGuidingBinCount : 0u);
    std::vector<std::mutex> pathGuidingMutexes(pathGuidingCellCount);
    const size_t radianceCacheCapacity =
        radianceCacheActive ? std::max(pixelCount * kCpuPathReservoirDepthCount, size_t(1)) : 0u;
    std::vector<CpuRadianceCacheCell> radianceCacheCells(radianceCacheCapacity);
    std::vector<std::mutex> radianceCacheMutexes(radianceCacheCapacity);

    std::atomic<uint64_t> restirGiCandidateCount{0};
    std::atomic<uint64_t> restirGiReservoirUpdateCount{0};
    std::atomic<uint64_t> restirGiAcceptCount{0};
    std::atomic<uint64_t> restirGiRejectCount{0};
    std::atomic<uint64_t> restirGiFallbackCount{0};
    std::atomic<uint64_t> pathGuidingCandidateCount{0};
    std::atomic<uint64_t> pathGuidingUsedCount{0};
    std::atomic<uint64_t> pathGuidingFallbackCount{0};
    std::atomic<uint64_t> pathGuidingInvalidCount{0};
    std::atomic<uint64_t> restirPtCandidateCount{0};
    std::atomic<uint64_t> restirPtReservoirUpdateCount{0};
    std::atomic<uint64_t> restirPtRejectedInvalidCount{0};
    std::atomic<uint64_t> restirPtRejectedUnsupportedCount{0};
    std::atomic<uint64_t> restirPtDebugRecordCount{0};
    std::atomic<uint64_t> restirPtReuseCandidateCount{0};
    std::atomic<uint64_t> restirPtReuseAppliedCount{0};
    std::atomic<uint64_t> restirPtReuseRejectedInvalidCount{0};
    std::atomic<uint64_t> restirPtReuseFallbackCount{0};
    std::atomic<uint64_t> radianceCacheQueryCount{0};
    std::atomic<uint64_t> radianceCacheHitCount{0};
    std::atomic<uint64_t> radianceCacheMissCount{0};
    std::atomic<uint64_t> radianceCacheTrainCount{0};
    std::atomic<uint64_t> radianceCacheFallbackCount{0};
    std::atomic<uint64_t> radianceCacheInvalidCount{0};

    auto reportProgress = [&](uint32_t doneTiles) {
        uint32_t percent = static_cast<uint32_t>(
            (static_cast<uint64_t>(doneTiles) * 100u) / totalTiles);
        CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
        double elapsed = now - renderStart;
        double etaSeconds = 0.0;
        if (doneTiles > 0u && elapsed > 0.0) {
            etaSeconds = (elapsed / static_cast<double>(doneTiles)) *
                         static_cast<double>(totalTiles - doneTiles);
        }
        {
            std::lock_guard<std::mutex> lock(progressMutex);
            std::cout << "\rEmbree progress: " << percent << "% (" << doneTiles
                      << "/" << totalTiles << " tiles)"
                      << " ETA " << FormatEta(etaSeconds) << std::flush;
            if (statsMode && ((now - lastStatsPrint) >= 1.0 || doneTiles >= totalTiles)) {
                const uint64_t pathSegments = approxPathSegmentCount.load(std::memory_order_relaxed);
                const uint64_t shadowRays = approxShadowRayCount.load(std::memory_order_relaxed);
                const uint64_t tracedRays = pathSegments + shadowRays;
                const double deltaSeconds = std::max(now - lastStatsPrint, 1.0e-6);
                const uint64_t deltaRays =
                    (tracedRays >= lastStatsRayCount) ? (tracedRays - lastStatsRayCount) : tracedRays;
                const double raysPerSecond = static_cast<double>(deltaRays) / deltaSeconds;
                const uint64_t totalPaths =
                    static_cast<uint64_t>(width) * static_cast<uint64_t>(height) *
                    static_cast<uint64_t>(targetSamples);
                const double averageBounces =
                    totalPaths > 0u
                        ? (static_cast<double>(pathSegments) / static_cast<double>(totalPaths))
                        : 0.0;
                std::cout << std::endl
                          << "[Stats] spp=" << targetSamples << "/" << targetSamples
                          << " frameMs=" << std::fixed << std::setprecision(2)
                          << (deltaSeconds * 1000.0)
                          << " rays=" << tracedRays
                          << " raysPerSec=" << static_cast<uint64_t>(raysPerSecond + 0.5)
                          << " avgBounces=" << std::setprecision(3)
                          << averageBounces
                          << std::endl;
                lastStatsPrint = now;
                lastStatsRayCount = tracedRays;
            }
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

    std::unique_ptr<tbb::global_control> tbbControl;
    if (threadLimit > 0) {
        tbbControl = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism,
            threadLimit);
    }

    auto renderTile = [&](uint32_t tileX, uint32_t tileY) {
        const uint32_t x0 = tileX * kTileSize;
        const uint32_t y0 = tileY * kTileSize;
        const uint32_t x1 = std::min(x0 + kTileSize, width);
        const uint32_t y1 = std::min(y0 + kTileSize, height);
        const uint32_t tileWidth = x1 - x0;
        const uint32_t tileHeight = y1 - y0;

        std::array<simd::float3, kTileArea> tilePixels{};
        std::array<simd::float3, kTileArea> tileAlbedoPixels{};
        std::array<simd::float3, kTileArea> tileNormalPixels{};
        uint64_t tileThroughputNanInfCount = 0;
        uint64_t tileRadianceNanInfCount = 0;
        uint64_t tileClampEventCount = 0;
        uint64_t tilePathSegmentCount = 0;
        uint64_t tileShadowRayCount = 0;
        std::vector<float> tileMaxThroughputByBounce(maxThroughputByBounce.size(), 0.0f);

        for (uint32_t localY = 0; localY < tileHeight; ++localY) {
            uint32_t y = y0 + localY;
            for (uint32_t localX = 0; localX < tileWidth; ++localX) {
                uint32_t x = x0 + localX;
                simd::float3 pixelRadiance = {0.0f, 0.0f, 0.0f};
                simd::float3 pixelAlbedo = {0.0f, 0.0f, 0.0f};
                simd::float3 pixelNormal = {0.0f, 0.0f, 0.0f};
                uint32_t pixelIndex = y * width + x;
                if (restirDirectMode && pixelIndex < restirDiDirectSum.size()) {
                    pixelRadiance += restirDiDirectSum[pixelIndex];
                }
                const bool debugPixel = debugCfg.enabled && x == debugCfg.pixelX && y == debugCfg.pixelY;

                for (uint32_t s = 0; s < targetSamples; ++s) {
                    const bool debugSample = debugPixel && s == 0;
                    Rng rng;
                    rng.state = seedBase +
                                s * 9781u +
                                x * 6271u +
                                y * 13007u +
                                (s + s) * 211u;

                    Ray ray = GenerateCameraRay(camera,
                                                width,
                                                height,
                                                x,
                                                y,
                                                rng,
                                                debugPixel && s == 0);
                    RayCone rayCone = primaryCone;
                    simd::float3 throughput = {1.0f, 1.0f, 1.0f};
                    simd::float3 radiance = {0.0f, 0.0f, 0.0f};
                    float lastBsdfPdf = 1.0f;
                    bool lastScatterWasDelta = true;
                    bool envLodActive = false;
                    float envLod = 0.0f;
                    uint32_t specularDepth = 0;
                    bool hadTransmission = false;
                    auto addContribution = [&](const simd::float3& contribution) {
                        bool clamped = false;
                        bool invalid = false;
                        simd::float3 safeContribution =
                            ClampFireflyContribution(throughput,
                                                     contribution,
                                                     clampParams,
                                                     &clamped,
                                                     &invalid);
                        if (invalid) {
                            ++tileRadianceNanInfCount;
                            return;
                        }
                        if (clamped) {
                            ++tileClampEventCount;
                        }
                        if (!IsFinite(safeContribution)) {
                            ++tileRadianceNanInfCount;
                            return;
                        }
                        radiance += safeContribution;
                    };

                    for (uint32_t depth = 0; depth < settings.maxDepth; ++depth) {
                        ++tilePathSegmentCount;
                        HitInfo hit;
                        if (!IntersectScene(sceneData, ray, hit)) {
                            simd::float3 background =
                                EvaluateBackground(settings, envPtr, ray.direction, envLodActive, envLod);
                            float misWeight = 1.0f;
                            bool useSpecularMis = depth > 0u &&
                                                  ((!lastScatterWasDelta) ||
                                                   settings.enableSpecularNee ||
                                                   settings.enableMnee);
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
                            addContribution(background * misWeight);
                            break;
                        }

                        if (!materials || materialCount == 0) {
                            break;
                        }

                        uint32_t matIndex = std::min(hit.materialIndex, materialCount - 1);
                        if (depth == 0u) {
                            PopulateFirstHitUvGradients(hit, ray, primaryRayDiff);
                        }
                        float surfaceFootprintWorld =
                            SurfaceFootprintFromCone(RayConeWidthAtDistance(rayCone, hit.t),
                                                     hit.normal,
                                                     -ray.direction);
                        PathTracerShaderTypes::MaterialData material =
                            ShadeMaterialForHit(sceneData,
                                                materials[matIndex],
                                                hit,
                                                settings.debugNormalStrengthScale,
                                                surfaceFootprintWorld);
                        if (MaterialAlphaMaskDiscards(material)) {
                            ray.origin = OffsetRayOrigin(hit, ray.direction);
                            continue;
                        }
                        uint32_t type = static_cast<uint32_t>(material.typeEta.x);
                        simd::float3 incidentDir = simd::normalize(ray.direction);
                        simd::float3 wo = -incidentDir;
                        simd::float3 shadingNormal = hit.shadingNormal;
                        if (simd::dot(shadingNormal, shadingNormal) <= 0.0f) {
                            shadingNormal = hit.normal;
                        }
                        if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Dielectric) ||
                            MaterialUsesGeometricNormal(material)) {
                            shadingNormal = hit.frontFace ? hit.normal : -hit.normal;
                        }
                        if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
                            shadingNormal = ApplyNormalMapForHit(sceneData,
                                                                 material,
                                                                 hit,
                                                                 shadingNormal,
                                                                 settings.debugNormalStrengthScale,
                                                                 surfaceFootprintWorld);
                        }
                        shadingNormal = simd::normalize(shadingNormal);
                        if (depth == 0u) {
                            if (collectAlbedoGuide) {
                                pixelAlbedo += MaterialBaseColor(material);
                            }
                            if (collectNormalGuide) {
                                pixelNormal += shadingNormal;
                            }
                        }
#if PT_DEBUG_TOOLS
                        if (settings.debugDirectLightAudit &&
                            debugSample &&
                            depth == 0 &&
                            !directLightAuditCaptured) {
                            directLightAudit.available = true;
                            directLightAudit.pixelX = x;
                            directLightAudit.pixelY = y;
                            directLightAudit.sampleIndex = s;
                            directLightAudit.depth = depth;
                            directLightAudit.materialIndex = matIndex;
                            directLightAudit.materialType = type;
                            directLightAudit.primitiveIndex = hit.primitiveIndex;
                            directLightAudit.hitPosition = hit.position;
                            directLightAudit.shadingNormal = shadingNormal;
                            directLightAudit.wo = wo;
                            directLightAudit.uv0 = hit.uv;
                            directLightAudit.baseColor = MaterialBaseColor(material);
                            directLightAudit.roughness = MaterialRoughness(material);
                            directLightAudit.baseColorTextureIndex = material.textureIndices0.x;
                            directLightAudit.baseColorLod = 0.0f;
                            auto baseTexIt = sceneData.materialTextures.find(material.textureIndices0.x);
                            if (baseTexIt != sceneData.materialTextures.end() && hit.hasUv) {
                                CpuTextureSamplingContext baseCtx =
                                    MakeCpuTextureSamplingContext(hit,
                                                                  material.textureUvSet0.x,
                                                                  material.textureTransform0,
                                                                  material.textureTransform1);
                                directLightAudit.baseColorLod =
                                    ((material.materialFlags & PathTracerShaderTypes::kMaterialFlagForceBaseColorMip0) != 0u)
                                        ? 0.0f
                                        : TextureLodWithFallback(baseTexIt->second,
                                                                 baseCtx.hasIgehyGradients,
                                                                 baseCtx.dUVdx,
                                                                 baseCtx.dUVdy,
                                                                 baseCtx.uvPerWorld,
                                                                 surfaceFootprintWorld);
                            }
                            directLightAudit.entries.clear();
                            directLightAudit.entries.reserve(rectLightCount);
                            for (uint32_t lightIndex = 0; lightIndex < rectLightCount; ++lightIndex) {
                                RectLightSample auditLight;
                                if (!DeterministicRectLightAuditSample(rectLights[lightIndex],
                                                                       rectLightCount,
                                                                       envPtr,
                                                                       settings,
                                                                       hit,
                                                                       auditLight)) {
                                    continue;
                                }
                                HeadlessRenderOutput::DirectLightAuditEntry entry{};
                                entry.lightIndex = lightIndex;
                                entry.rectIndex = auditLight.rectIndex;
                                entry.lightDirection = auditLight.direction;
                                entry.distance = auditLight.distance;
                                entry.nDotL = std::max(simd::dot(shadingNormal, auditLight.direction), 0.0f);
                                entry.lightPdf = auditLight.pdf;
                                bool visible = false;
                                HitInfo shadowHit;
                                if (entry.nDotL > 0.0f && auditLight.pdf > 0.0f) {
                                    Ray shadowRay;
                                    shadowRay.origin = OffsetRayOrigin(hit, auditLight.direction);
                                    shadowRay.direction = auditLight.direction;
                                    float shadowMax = std::max(auditLight.distance - kEpsilon, kEpsilon);
                                    bool blocked =
                                        IntersectSceneSkippingAlpha(sceneData,
                                                                    shadowRay,
                                                                    shadowMax,
                                                                    materials,
                                                                    materialCount,
                                                                    settings.debugNormalStrengthScale,
                                                                    shadowHit);
                                    if (blocked &&
                                        shadowHit.primitiveType == GeometryType::Rectangles &&
                                        shadowHit.primitiveIndex == auditLight.rectIndex) {
                                        blocked = false;
                                    }
                                    visible = !blocked;
                                    if (!visible) {
                                        entry.occluderMaterialIndex = shadowHit.materialIndex;
                                        entry.occluderPrimitiveIndex = shadowHit.primitiveIndex;
                                        entry.occluderPrimitiveType = static_cast<uint32_t>(shadowHit.primitiveType);
                                        entry.occluderFrontFace = shadowHit.frontFace;
                                        entry.occluderDistance = shadowHit.t;
                                    }
                                }
                                entry.visible = visible;
                                BsdfEval bsdfEval = EvaluateBsdf(material,
                                                                 hit.position,
                                                                 shadingNormal,
                                                                 wo,
                                                                 auditLight.direction,
                                                                 clampParams);
                                entry.bsdfValue = bsdfEval.value;
                                entry.bsdfPdf = bsdfEval.pdf;
                                float misWeight = 1.0f;
                                if (bsdfEval.pdf > 0.0f) {
                                    float denom = auditLight.pdf + bsdfEval.pdf;
                                    if (denom > 0.0f) {
                                        misWeight = std::clamp(auditLight.pdf / denom,
                                                               kMisWeightClampMin,
                                                               kMisWeightClampMax);
                                    }
                                }
                                entry.misWeight = misWeight;
                                if (visible && entry.nDotL > 0.0f && auditLight.pdf > 0.0f && !bsdfEval.isDelta) {
                                    simd::float3 contribution = auditLight.emission * bsdfEval.value * entry.nDotL;
                                    contribution *= misWeight / auditLight.pdf;
                                    if (IsFinite(contribution)) {
                                        entry.contribution = contribution;
                                    }
                                }
                                directLightAudit.entries.push_back(entry);
                            }
                            directLightAuditCaptured = true;
                        }
                        if (settings.enablePathDebug &&
                            debugSample &&
                            depth == 0 &&
                            type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness) &&
                            !materialMrDebugCaptured) {
                            materialMrDebug.available = true;
                            materialMrDebug.pixelX = x;
                            materialMrDebug.pixelY = y;
                            materialMrDebug.sampleIndex = s;
                            materialMrDebug.depth = depth;
                            materialMrDebug.materialIndex = matIndex;
                            materialMrDebug.primitiveIndex = hit.primitiveIndex;
                            materialMrDebug.uv0 = hit.uv;
                            materialMrDebug.uv1 = hit.uv1;
                            materialMrDebug.baseColor = MaterialBaseColor(material);
                            materialMrDebug.metallic = material.pbrParams.x;
                            materialMrDebug.roughness = material.pbrParams.y;
                            materialMrDebug.roughnessInput = material.baseColorRoughness.w;
                            materialMrDebug.roughnessBeforeNormalVariance = material.pbrParams.y;
                            materialMrDebug.ao = material.pbrParams.z;
                            materialMrDebug.alpha = material.pbrExtras.x;
                            materialMrDebug.alphaCutoff = material.pbrExtras.y;
                            materialMrDebug.alphaMode = material.pbrExtras.w;
                            materialMrDebug.transmission = material.pbrExtras.z;
                            materialMrDebug.emissive = simd_make_float3(material.emission.x,
                                                                        material.emission.y,
                                                                        material.emission.z);
                            materialMrDebug.geometricNormal = hit.normal;
                            materialMrDebug.shadingNormal = hit.shadingNormal;
                            materialMrDebug.perturbedNormal = shadingNormal;
                            materialMrDebug.tangent = simd_make_float3(hit.tangent.x, hit.tangent.y, hit.tangent.z);
                            if (hit.hasTangent) {
                                simd::float3 t = simd::normalize(materialMrDebug.tangent -
                                                                 shadingNormal * simd::dot(shadingNormal, materialMrDebug.tangent));
                                float tangentSign = hit.tangent.w < 0.0f ? -1.0f : 1.0f;
                                materialMrDebug.tangent = t;
                                materialMrDebug.bitangent =
                                    simd::normalize(simd::cross(shadingNormal, t)) * tangentSign;
                            }
                            auto normalTexIt = sceneData.materialTextures.find(material.textureIndices0.z);
                            if (normalTexIt != sceneData.materialTextures.end() && hit.hasUv) {
                                CpuTextureSamplingContext normalCtx =
                                    MakeCpuTextureSamplingContext(hit,
                                                                  material.textureUvSet0.z,
                                                                  material.textureTransform4,
                                                                  material.textureTransform5);
                                float normalLod =
                                    TextureLodWithFallback(normalTexIt->second,
                                                           normalCtx.hasIgehyGradients,
                                                           normalCtx.dUVdx,
                                                           normalCtx.dUVdy,
                                                           normalCtx.uvPerWorld,
                                                           surfaceFootprintWorld);
                                simd::float4 raw = SampleCpuTexture(normalTexIt->second, normalCtx.uv, normalLod);
                                materialMrDebug.normalSampleRaw = simd_make_float3(raw.x, raw.y, raw.z);
                                materialMrDebug.normalScale = material.pbrParams.w;
                                materialMrDebug.normalLod = normalLod;
                                materialMrDebug.normalSampleTs =
                                    DecodeNormalMap(materialMrDebug.normalSampleRaw, materialMrDebug.normalScale);
                            }
                            float dielectricF0 = DielectricF0FromIor(material.typeEta.y);
                            materialMrDebug.f0 =
                                MaterialBaseColor(material) * material.pbrParams.x +
                                simd_make_float3(dielectricF0, dielectricF0, dielectricF0) * (1.0f - material.pbrParams.x);
                            materialMrDebug.specularWeight = PbrSpecularWeight(materialMrDebug.f0);
                            materialMrDebug.diffuseWeight = 1.0f - materialMrDebug.specularWeight;
                            materialMrDebug.ndotv = std::max(simd::dot(shadingNormal, wo), 0.0f);
                            materialMrDebugCaptured = true;
                        }
#endif
                        if (debugSample && depth == 0) {
                            simd::float3 baseColor = MaterialBaseColor(material);
                            float debugUvPerWorld = hit.uvPerWorld0;
                            if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness) &&
                                material.textureUvSet0.z == 1u) {
                                debugUvPerWorld = hit.uvPerWorld1;
                            }
                            std::cerr << "[EmbreeDirectDebug] hit"
                                      << " pixel=(" << x << "," << y << ")"
                                      << " materialType=" << type
                                      << " frontFace=" << (hit.frontFace ? 1 : 0)
                                      << " twoSided=" << (hit.twoSided ? 1 : 0)
                                      << " nDotV=" << std::max(simd::dot(shadingNormal, wo), 0.0f)
                                      << " baseLum=" << Luminance(baseColor)
                                      << " roughness=" << MaterialRoughness(material)
                                      << " uvPerWorld=" << debugUvPerWorld
                                      << " surfaceFootprint=" << surfaceFootprintWorld
                                      << " normalLod=" << materialMrDebug.normalLod
                                      << " rectLights=" << rectLightCount
                                      << std::endl;
                        }

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
                            if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::DiffuseLight)) {
                                float misWeight = 1.0f;
                                bool useSpecularMis = (depth > 0) &&
                                                      ((!lastScatterWasDelta) ||
                                                      settings.enableSpecularNee ||
                                                      settings.enableMnee);
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
                                addContribution(emission * misWeight);
                                break;
                            }
                            addContribution(emission);
                        }

                        bool surfaceIsDelta = MaterialIsDelta(material);
                        const bool restirDirectEligibleMaterial =
                            !restirDirectMode ||
                            type == static_cast<uint32_t>(
                                        PathTracerShaderTypes::MaterialType::PbrMetallicRoughness);

                        if ((!restirDirectMode || depth == 1u) &&
                            !surfaceIsDelta &&
                            restirDirectEligibleMaterial &&
                            !directionalLights.empty()) {
                            for (const DirectionalLightInfo& light : directionalLights) {
                                float nDotL = std::max(simd::dot(shadingNormal, light.direction), 0.0f);
                                if (!(nDotL > 0.0f)) {
                                    continue;
                                }

                                ++tileShadowRayCount;
                                Ray shadowRay;
                                shadowRay.origin = OffsetRayOrigin(hit, light.direction);
                                shadowRay.direction = light.direction;
                                HitInfo shadowHit;
                                const bool occluded =
                                    IntersectSceneSkippingAlpha(sceneData,
                                                                shadowRay,
                                                                1.0e20f,
                                                                materials,
                                                                materialCount,
                                                                settings.debugNormalStrengthScale,
                                                                shadowHit);
                                if (debugSample && depth == 0) {
                                    std::cerr << "[EmbreeDirectDebug] directional_sample"
                                              << " pixel=(" << x << "," << y << ")"
                                              << " nDotL=" << nDotL
                                              << " occluded=" << (occluded ? 1 : 0)
                                              << " emissionLum=" << Luminance(light.emission)
                                              << std::endl;
                                }
                                if (occluded) {
                                    continue;
                                }

                                BsdfEval bsdfEval = EvaluateBsdf(material,
                                                                 hit.position,
                                                                 shadingNormal,
                                                                 wo,
                                                                 light.direction,
                                                                 clampParams);
                                if (debugSample && depth == 0) {
                                    std::cerr << "[EmbreeDirectDebug] directional_bsdf"
                                              << " bsdfPdf=" << bsdfEval.pdf
                                              << " bsdfLum=" << Luminance(bsdfEval.value)
                                              << " isDelta=" << (bsdfEval.isDelta ? 1 : 0)
                                              << std::endl;
                                }
                                if (!bsdfEval.isDelta && bsdfEval.pdf > 0.0f) {
                                    simd::float3 contribution = light.emission * bsdfEval.value * nDotL;
                                    if (debugSample && depth == 0) {
                                        std::cerr << "[EmbreeDirectDebug] directional_contrib"
                                                  << " contribLum=" << Luminance(contribution)
                                                  << std::endl;
                                    }
                                    addContribution(contribution);
                                }
                            }
                        }

                        if (!restirDirectMode &&
                            !surfaceIsDelta &&
                            restirDirectEligibleMaterial &&
                            rectLightCount > 0) {
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
                                    ++tileShadowRayCount;
                                    HitInfo shadowHit;
                                    Ray shadowRay;
                                    shadowRay.origin = OffsetRayOrigin(hit, lightSample.direction);
                                    shadowRay.direction = lightSample.direction;
                                    bool occluded = false;
                                    if (IntersectSceneSkippingAlpha(sceneData,
                                                                    shadowRay,
                                                                    shadowMax,
                                                                    materials,
                                                                    materialCount,
                                                                    settings.debugNormalStrengthScale,
                                                                    shadowHit)) {
                                        occluded = !(shadowHit.primitiveType == GeometryType::Rectangles &&
                                                     shadowHit.primitiveIndex == lightSample.rectIndex);
                                    }
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
                                            if (materialMrDebugCaptured &&
                                                debugSample &&
                                                depth == 0 &&
                                                type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
                                                materialMrDebug.directContribution = contribution;
                                                materialMrDebug.directContributionKind = 2.0f;
                                            }
                                            if (debugSample && depth == 0) {
                                                std::cerr << "[EmbreeDirectDebug] rect_contrib"
                                                          << " weight=" << weight
                                                          << " contribLum=" << Luminance(contribution)
                                                          << std::endl;
                                            }
                                            addContribution(contribution);
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
                                Ray envShadowRay;
                                envShadowRay.origin = OffsetRayOrigin(hit, envSample.direction);
                                envShadowRay.direction = envSample.direction;
                                HitInfo envShadowHit;
                                if (!IntersectSceneSkippingAlpha(sceneData,
                                                                 envShadowRay,
                                                                 std::numeric_limits<float>::infinity(),
                                                                 materials,
                                                                 materialCount,
                                                                 settings.debugNormalStrengthScale,
                                                                 envShadowHit)) {
                                    float envNeeLod = 0.0f;
                                    bool envNeeLodActive = false;
                                    if (EnvironmentMaxMip(*envPtr) > 0.0f) {
                                        const float envRoughness = EnvironmentLightingRoughness(material);
                                        if (envRoughness < 0.95f) {
                                            envNeeLod = EnvironmentLodFromRoughness(envRoughness, *envPtr);
                                            envNeeLodActive = true;
                                        }
                                    }
                                    simd::float3 envRadiance =
                                        envNeeLodActive
                                            ? SampleEnvironmentLod(*envPtr,
                                                                   envSample.direction,
                                                                   settings.environmentRotation,
                                                                   settings.environmentIntensity,
                                                                   envNeeLod)
                                            : SampleEnvironment(*envPtr,
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
                                        if (materialMrDebugCaptured &&
                                            debugSample &&
                                            depth == 0 &&
                                            type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
                                            materialMrDebug.directContribution = contribution;
                                            materialMrDebug.directContributionKind = 1.0f;
                                            materialMrDebug.envLod = envNeeLodActive ? envNeeLod : 0.0f;
                                        }
                                        addContribution(contribution);
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
                    if (materialMrDebugCaptured &&
                        debugSample &&
                        depth == 0 &&
                        type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
                        materialMrDebug.sampledOutgoingDirection = bsdfSample.direction;
                        materialMrDebug.specularPdf = bsdfSample.pdf;
                        float NoV = std::max(simd::dot(shadingNormal, wo), 0.0f);
                        float NoL = std::max(simd::dot(shadingNormal, bsdfSample.direction), 0.0f);
                        simd::float3 wh = simd::normalize(wo + bsdfSample.direction);
                        if (simd::dot(wh, wh) > 0.0f && NoV > 0.0f && NoL > 0.0f) {
                            float alpha = std::max(material.pbrParams.y * material.pbrParams.y, 1.0e-4f);
                            materialMrDebug.dTerm =
                                GgxDistribution(alpha, std::max(simd::dot(shadingNormal, wh), 0.0f));
                            materialMrDebug.gTerm = GgxG1(alpha, NoV) * GgxG1(alpha, NoL);
                            materialMrDebug.fTerm =
                                SchlickFresnel(materialMrDebug.f0,
                                               std::max(simd::dot(bsdfSample.direction, wh), 0.0f));
                            float denom = std::max(4.0f * NoV * NoL, 1.0e-6f);
                            materialMrDebug.brdfSpecularValue =
                                materialMrDebug.fTerm * (materialMrDebug.dTerm * materialMrDebug.gTerm / denom);
                            if (bsdfSample.pdf > 0.0f) {
                                materialMrDebug.finalSpecularContribution =
                                    materialMrDebug.brdfSpecularValue * (NoL / bsdfSample.pdf);
                            }
                        }
                    }
                    if (bsdfSample.pdf <= 0.0f ||
                        simd::dot(bsdfSample.direction, bsdfSample.direction) <= 0.0f ||
                        !IsFinite(bsdfSample.weight)) {
                        break;
                    }

                    if (restirGiActive) {
                        if (CpuBsdfDiffuseCandidateValid(bsdfSample,
                                                         material,
                                                         depth,
                                                         /*requirePrimaryDepth=*/true)) {
                            restirGiCandidateCount.fetch_add(1u, std::memory_order_relaxed);
                            bool reused = false;
                            CpuPathReservoirState& reservoir = restirGiReservoirs[pixelIndex];
                            simd::float3 reservoirDirection;
                            float reservoirConfidence = 0.0f;
                            if (CpuLoadPathReservoirDirection(reservoir,
                                                              hit.position,
                                                              shadingNormal,
                                                              depth,
                                                              /*depthAware=*/false,
                                                              /*maxPositionDistance=*/0.75f,
                                                              reservoirDirection,
                                                              reservoirConfidence)) {
                                BsdfSample reuseSample = bsdfSample;
                                if (CpuReweightBsdfSampleForDirection(material,
                                                                      hit.position,
                                                                      shadingNormal,
                                                                      wo,
                                                                      reservoirDirection,
                                                                      clampParams,
                                                                      reuseSample)) {
                                    const float currentScore =
                                        Luminance(simd::max(bsdfSample.weight,
                                                            simd_make_float3(0.0f, 0.0f, 0.0f)));
                                    const float reuseScore =
                                        Luminance(simd::max(reuseSample.weight,
                                                            simd_make_float3(0.0f, 0.0f, 0.0f))) *
                                        std::clamp(reservoirConfidence, 0.0f, 1.0f);
                                    if (currentScore > 0.0f &&
                                        reuseScore > 0.0f &&
                                        std::isfinite(currentScore) &&
                                        std::isfinite(reuseScore)) {
                                        const float chooseReuse =
                                            reuseScore / std::max(currentScore + reuseScore, 1.0e-6f);
                                        if (rng.nextFloat() < chooseReuse) {
                                            bsdfSample = reuseSample;
                                            bsdfSample.direction = simd::normalize(bsdfSample.direction);
                                            reused = true;
                                        }
                                    }
                                }
                            }
                            CpuUpdatePathReservoir(reservoir,
                                                   material,
                                                   hit.position,
                                                   shadingNormal,
                                                   simd_make_float3(1.0f, 1.0f, 1.0f),
                                                   depth,
                                                   s,
                                                   bsdfSample);
                            reservoir.throughputLuma = reservoir.sampleLuma;
                            restirGiReservoirUpdateCount.fetch_add(1u, std::memory_order_relaxed);
                            if (reused) {
                                restirGiAcceptCount.fetch_add(1u, std::memory_order_relaxed);
                            } else {
                                restirGiRejectCount.fetch_add(1u, std::memory_order_relaxed);
                            }
                        } else {
                            restirGiFallbackCount.fetch_add(1u, std::memory_order_relaxed);
                        }
                    }

                    if (pathGuidingActive) {
                        CpuApplyPathGuidingPrototype(settings,
                                                     material,
                                                     hit.position,
                                                     shadingNormal,
                                                     wo,
                                                     clampParams,
                                                     depth,
                                                     s,
                                                     pathGuidingStates,
                                                     pathGuidingMutexes,
                                                     pathGuidingCellCount,
                                                     pathGuidingCandidateCount,
                                                     pathGuidingUsedCount,
                                                     pathGuidingFallbackCount,
                                                     pathGuidingInvalidCount,
                                                     rng,
                                                     bsdfSample);
                    }

                    if (restirPtReuseActive) {
                        if (depth >= 1u &&
                            CpuBsdfDiffuseCandidateValid(bsdfSample,
                                                         material,
                                                         depth,
                                                         /*requirePrimaryDepth=*/false)) {
                            restirPtReuseCandidateCount.fetch_add(1u, std::memory_order_relaxed);
                            const float strength =
                                std::clamp(settings.restirPtReuseStrength, 0.0f, 1.0f);
                            if (strength <= 0.0f) {
                                restirPtReuseFallbackCount.fetch_add(1u, std::memory_order_relaxed);
                            } else {
                                const size_t reservoirIndex =
                                    static_cast<size_t>(pixelIndex) * kCpuPathReservoirDepthCount +
                                    std::min(depth, kCpuPathReservoirDepthCount - 1u);
                                simd::float3 reservoirDirection;
                                float reservoirConfidence = 0.0f;
                                if (CpuLoadPathReservoirDirection(restirPtReservoirs[reservoirIndex],
                                                                  hit.position,
                                                                  shadingNormal,
                                                                  depth,
                                                                  /*depthAware=*/true,
                                                                  /*maxPositionDistance=*/1.0f,
                                                                  reservoirDirection,
                                                                  reservoirConfidence)) {
                                    const simd::float3 currentDir = simd::normalize(bsdfSample.direction);
                                    simd::float3 reusedDir =
                                        simd::normalize(currentDir * (1.0f - strength * reservoirConfidence) +
                                                        reservoirDirection * (strength * reservoirConfidence));
                                    BsdfSample reuseSample = bsdfSample;
                                    if (simd::dot(shadingNormal, reusedDir) > 0.0f &&
                                        IsFinite(reusedDir) &&
                                        CpuReweightBsdfSampleForDirection(material,
                                                                          hit.position,
                                                                          shadingNormal,
                                                                          wo,
                                                                          reusedDir,
                                                                          clampParams,
                                                                          reuseSample)) {
                                        bsdfSample = reuseSample;
                                        restirPtReuseAppliedCount.fetch_add(1u, std::memory_order_relaxed);
                                    } else {
                                        restirPtReuseRejectedInvalidCount.fetch_add(1u, std::memory_order_relaxed);
                                    }
                                } else {
                                    restirPtReuseFallbackCount.fetch_add(1u, std::memory_order_relaxed);
                                }
                            }
                        } else {
                            restirPtReuseFallbackCount.fetch_add(1u, std::memory_order_relaxed);
                        }
                    }

                    if (restirPtActive) {
                        if (depth >= 1u &&
                            CpuBsdfDiffuseCandidateValid(bsdfSample,
                                                         material,
                                                         depth,
                                                         /*requirePrimaryDepth=*/false) &&
                            IsFinite(throughput) &&
                            throughput.x >= 0.0f &&
                            throughput.y >= 0.0f &&
                            throughput.z >= 0.0f) {
                            const uint64_t previousCandidate =
                                restirPtCandidateCount.fetch_add(1u, std::memory_order_relaxed);
                            const size_t reservoirIndex =
                                static_cast<size_t>(pixelIndex) * kCpuPathReservoirDepthCount +
                                std::min(depth, kCpuPathReservoirDepthCount - 1u);
                            CpuUpdatePathReservoir(restirPtReservoirs[reservoirIndex],
                                                   material,
                                                   hit.position,
                                                   shadingNormal,
                                                   throughput,
                                                   depth,
                                                   s,
                                                   bsdfSample);
                            if (previousCandidate >= std::max(settings.restirPtMaxReservoirs, 1u)) {
                                restirPtRejectedUnsupportedCount.fetch_add(1u, std::memory_order_relaxed);
                            } else {
                                restirPtReservoirUpdateCount.fetch_add(1u, std::memory_order_relaxed);
                                if (settings.restirPtDebug) {
                                    restirPtDebugRecordCount.fetch_add(1u, std::memory_order_relaxed);
                                }
                            }
                        } else if (depth >= 1u && CpuDiffuseRestirEligibleMaterial(material)) {
                            restirPtRejectedInvalidCount.fetch_add(1u, std::memory_order_relaxed);
                        } else {
                            restirPtRejectedUnsupportedCount.fetch_add(1u, std::memory_order_relaxed);
                        }
                    }

                    bool radianceCacheTerminated = false;
                    if (radianceCacheActive) {
                        if (!CpuRadianceCacheCanQuery(settings, material, bsdfSample, depth)) {
                            radianceCacheFallbackCount.fetch_add(1u, std::memory_order_relaxed);
                        } else {
                            radianceCacheQueryCount.fetch_add(1u, std::memory_order_relaxed);
                            const uint32_t materialType = static_cast<uint32_t>(material.typeEta.x);
                            const uint32_t domainKey =
                                CpuRadianceCacheDomainKey(settings,
                                                          hit.position,
                                                          depth,
                                                          materialType,
                                                          0u);
                            const size_t cacheIndex =
                                radianceCacheCapacity > 0u ? (domainKey % radianceCacheCapacity) : 0u;
                            std::lock_guard<std::mutex> cacheLock(radianceCacheMutexes[cacheIndex]);
                            CpuRadianceCacheCell& cell = radianceCacheCells[cacheIndex];
                            bool cacheHit = false;
                            if (cell.valid &&
                                !cell.invalidated &&
                                cell.sampleCount > 0u &&
                                cell.domainKey == domainKey &&
                                cell.materialType == materialType &&
                                cell.lobeType == 0u &&
                                cell.frameTag < s) {
                                const float normalDot =
                                    simd::dot(simd::normalize(cell.normal),
                                              simd::normalize(shadingNormal));
                                const float roughnessDelta =
                                    std::fabs(cell.roughness - MaterialRoughness(material));
                                const simd::float3 estimate =
                                    cell.radianceSum /
                                    static_cast<float>(std::max(cell.sampleCount, 1u));
                                const float luma =
                                    Luminance(simd::max(estimate,
                                                        simd_make_float3(0.0f, 0.0f, 0.0f)));
                                const float meanLuma2 =
                                    cell.lumaMomentSum /
                                    static_cast<float>(std::max(cell.sampleCount, 1u));
                                const float variance =
                                    std::max(std::max(meanLuma2 - luma * luma, 0.0f),
                                             cell.storedVariance);
                                const float biasRisk =
                                    std::clamp(variance /
                                                   std::max(luma * luma + 1.0e-4f, 1.0e-4f),
                                               0.0f,
                                               1.0f);
                                if (normalDot >= 0.65f &&
                                    roughnessDelta <= 0.35f &&
                                    biasRisk <= 0.85f &&
                                    cell.confidence >= settings.radianceCacheMinConfidence &&
                                    IsFinite(estimate)) {
                                    addContribution(estimate);
                                    radianceCacheHitCount.fetch_add(1u, std::memory_order_relaxed);
                                    cacheHit = true;
                                    radianceCacheTerminated = true;
                                }
                            }
                            if (!cacheHit) {
                                radianceCacheMissCount.fetch_add(1u, std::memory_order_relaxed);
                                radianceCacheFallbackCount.fetch_add(1u, std::memory_order_relaxed);
                            }

                            if (!cacheHit &&
                                (s % std::max(settings.radianceCacheTrainingInterval, 1u)) == 0u) {
                                const simd::float3 trainingRadiance =
                                    simd::max(bsdfSample.weight,
                                              simd_make_float3(0.0f, 0.0f, 0.0f));
                                const float luma = Luminance(trainingRadiance);
                                const float maxRadiance =
                                    std::max(settings.radianceCacheMaxRadiance, 1.0e-4f);
                                if (!IsFinite(trainingRadiance) ||
                                    !std::isfinite(luma) ||
                                    trainingRadiance.x > maxRadiance ||
                                    trainingRadiance.y > maxRadiance ||
                                    trainingRadiance.z > maxRadiance ||
                                    luma > maxRadiance) {
                                    radianceCacheInvalidCount.fetch_add(1u, std::memory_order_relaxed);
                                } else if (luma > 0.0f) {
                                    if (!cell.valid ||
                                        cell.invalidated ||
                                        cell.domainKey != domainKey ||
                                        cell.materialType != materialType ||
                                        cell.lobeType != 0u) {
                                        if (cell.sampleCount > 0u && cell.domainKey != domainKey) {
                                            radianceCacheInvalidCount.fetch_add(1u, std::memory_order_relaxed);
                                        }
                                        cell = CpuRadianceCacheCell{};
                                        cell.valid = true;
                                        cell.domainKey = domainKey;
                                        cell.materialType = materialType;
                                        cell.lobeType = 0u;
                                    }
                                    if (cell.sampleCount >= 1024u) {
                                        radianceCacheFallbackCount.fetch_add(1u, std::memory_order_relaxed);
                                    } else {
                                        cell.sampleCount += 1u;
                                        cell.radianceSum += trainingRadiance;
                                        cell.lumaMomentSum += luma * luma;
                                        const float samples =
                                            static_cast<float>(std::max(cell.sampleCount, 1u));
                                        cell.position = hit.position;
                                        cell.normal = simd::normalize(shadingNormal);
                                        cell.roughness = MaterialRoughness(material);
                                        cell.storedVariance =
                                            samples > 1.0f ? (luma * luma) / samples : 1.0f;
                                        cell.confidence =
                                            std::clamp(std::log2(samples + 1.0f) / 6.0f,
                                                       0.0f,
                                                       1.0f);
                                        cell.frameTag = s;
                                        radianceCacheTrainCount.fetch_add(1u, std::memory_order_relaxed);
                                    }
                                }
                            }
                        }
                    }
                    if (radianceCacheTerminated) {
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
                        Ray neeVisibilityRay;
                        neeVisibilityRay.origin = OffsetRayOrigin(hit, neeDir);
                        neeVisibilityRay.direction = neeDir;
                        HitInfo neeVisibilityHit;
                        if (!IntersectSceneSkippingAlpha(sceneData,
                                                         neeVisibilityRay,
                                                         std::numeric_limits<float>::infinity(),
                                                         materials,
                                                         materialCount,
                                                         settings.debugNormalStrengthScale,
                                                         neeVisibilityHit)) {
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
                            addContribution(neeContribution);
                        }
                    }

                    if (specNeeEligible && rectLightCount > 0) {
                        simd::float3 neeDir = simd::normalize(bsdfSample.direction);
                        Ray neeRay;
                        neeRay.origin = OffsetRayOrigin(hit, neeDir);
                        neeRay.direction = neeDir;
                        HitInfo lightHit;
                        if (IntersectSceneSkippingAlpha(sceneData,
                                                        neeRay,
                                                        std::numeric_limits<float>::infinity(),
                                                        materials,
                                                        materialCount,
                                                        settings.debugNormalStrengthScale,
                                                        lightHit)) {
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
                                addContribution(neeContribution);
                            }
                        }
                    }

                    if (mneeEligible && envSampling) {
                        simd::float3 mneeDir = simd::normalize(bsdfSample.direction);
                        ++tileShadowRayCount;
	                        Ray mneeVisibilityRay;
	                        mneeVisibilityRay.origin = OffsetRayOrigin(hit, mneeDir);
	                        mneeVisibilityRay.direction = mneeDir;
	                        HitInfo mneeVisibilityHit;
	                        if (!IntersectSceneSkippingAlpha(sceneData,
	                                                         mneeVisibilityRay,
	                                                         std::numeric_limits<float>::infinity(),
	                                                         materials,
	                                                         materialCount,
	                                                         settings.debugNormalStrengthScale,
	                                                         mneeVisibilityHit)) {
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
                            addContribution(mneeContribution);
                        }
                    }

                    if (mneeEligible && rectLightCount > 0) {
                        simd::float3 mneeDir = simd::normalize(bsdfSample.direction);
                        Ray mneeRay;
                        mneeRay.origin = OffsetRayOrigin(hit, mneeDir);
                        mneeRay.direction = mneeDir;
                        HitInfo lightHit;
                        ++tileShadowRayCount;
                        if (IntersectSceneSkippingAlpha(sceneData,
                                                        mneeRay,
                                                        std::numeric_limits<float>::infinity(),
                                                        materials,
                                                        materialCount,
                                                        settings.debugNormalStrengthScale,
                                                        lightHit)) {
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
                                addContribution(mneeContribution);
                            }
                        }
                    }

                    if (mneeEligible && settings.enableMneeSecondary) {
                        simd::float3 chainDir = simd::normalize(bsdfSample.direction);
                        Ray chainRay;
                        chainRay.origin = OffsetRayOrigin(hit, chainDir);
                        chainRay.direction = chainDir;
                        HitInfo chainHit;
                        if (IntersectSceneSkippingAlpha(sceneData,
                                                        chainRay,
                                                        std::numeric_limits<float>::infinity(),
                                                        materials,
                                                        materialCount,
                                                        settings.debugNormalStrengthScale,
                                                        chainHit)) {
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
                                            ++tileShadowRayCount;
	                                            HitInfo secondVisibilityHit;
	                                            if (!IntersectSceneSkippingAlpha(sceneData,
	                                                                             secondRay,
	                                                                             std::numeric_limits<float>::infinity(),
	                                                                             materials,
	                                                                             materialCount,
	                                                                             settings.debugNormalStrengthScale,
	                                                                             secondVisibilityHit)) {
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
                                                addContribution(contribution);
                                            }
                                        }
                                        if (rectLightCount > 0) {
                                            HitInfo lightHit;
                                            ++tileShadowRayCount;
                                            if (IntersectSceneSkippingAlpha(sceneData,
                                                                            secondRay,
                                                                            std::numeric_limits<float>::infinity(),
                                                                            materials,
                                                                            materialCount,
                                                                            settings.debugNormalStrengthScale,
                                                                            lightHit)) {
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
                                                    addContribution(contribution);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                        throughput *= bsdfSample.weight;
                        bool throughputClamped = false;
                        bool throughputInvalid = false;
                        throughput = ClampPathThroughput(throughput,
                                                         clampParams,
                                                         &throughputClamped,
                                                         &throughputInvalid);
                        if (throughputClamped) {
                            ++tileClampEventCount;
                        }
                        if (throughputInvalid || !IsFinite(throughput)) {
                            ++tileThroughputNanInfCount;
                            break;
                        }
                        if (depth < tileMaxThroughputByBounce.size()) {
                            simd::float3 throughputPositive =
                                simd::max(throughput, simd_make_float3(0.0f, 0.0f, 0.0f));
                            float throughputLum = Luminance(throughputPositive);
                            tileMaxThroughputByBounce[depth] =
                                std::max(tileMaxThroughputByBounce[depth], throughputLum);
                        }
                        float maxComp = std::max({throughput.x, throughput.y, throughput.z});
                        if (maxComp <= 0.0f) {
                            break;
                        }

                        bool nextEnvLodActive = false;
                        float nextEnvLod = 0.0f;
                        if (bsdfSample.lobeType == 1u && !bsdfSample.isDelta && envPtr &&
                            EnvironmentMaxMip(*envPtr) > 0.0f) {
                            nextEnvLod = EnvironmentLodFromRoughness(bsdfSample.lobeRoughness, *envPtr);
                            nextEnvLodActive = true;
                        }
                        envLodActive = nextEnvLodActive;
                        envLod = nextEnvLod;

                        rayCone.width = RayConeWidthAtDistance(rayCone, hit.t);
                        rayCone.spread = std::min(rayCone.spread +
                                                      BsdfConeSpreadIncrement(bsdfSample.lobeType,
                                                                              bsdfSample.lobeRoughness,
                                                                              bsdfSample.isDelta),
                                                  1.5f);

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
                if (collectAlbedoGuide) {
                    tileAlbedoPixels[tileIndex] = pixelAlbedo / static_cast<float>(targetSamples);
                }
                if (collectNormalGuide) {
                    simd::float3 avgNormal = pixelNormal / static_cast<float>(targetSamples);
                    if (simd::dot(avgNormal, avgNormal) > 1.0e-12f) {
                        avgNormal = simd::normalize(avgNormal);
                    }
                    tileNormalPixels[tileIndex] = avgNormal;
                }
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
                if (collectAlbedoGuide) {
                    const simd::float3 albedo = tileAlbedoPixels[tileIndex];
                    albedoLinearRGB[base + 0] = albedo.x;
                    albedoLinearRGB[base + 1] = albedo.y;
                    albedoLinearRGB[base + 2] = albedo.z;
                }
                if (collectNormalGuide) {
                    const simd::float3 normal = tileNormalPixels[tileIndex];
                    normalLinearRGB[base + 0] = normal.x;
                    normalLinearRGB[base + 1] = normal.y;
                    normalLinearRGB[base + 2] = normal.z;
                }
            }
        }

        {
            std::lock_guard<std::mutex> metricsLock(metricsMutex);
            throughputNanInfCount += tileThroughputNanInfCount;
            radianceNanInfCount += tileRadianceNanInfCount;
            clampEventCount += tileClampEventCount;
            approxPathSegmentCount.fetch_add(tilePathSegmentCount, std::memory_order_relaxed);
            approxShadowRayCount.fetch_add(tileShadowRayCount, std::memory_order_relaxed);
            for (size_t i = 0; i < tileMaxThroughputByBounce.size(); ++i) {
                maxThroughputByBounce[i] =
                    std::max(maxThroughputByBounce[i], tileMaxThroughputByBounce[i]);
            }
        }

        if (verbose || statsMode) {
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

    tbb::parallel_for(
        tbb::blocked_range2d<uint32_t>(0, tilesY, 0, tilesX),
        [&](const tbb::blocked_range2d<uint32_t>& range) {
            for (uint32_t tileY = range.rows().begin(); tileY < range.rows().end(); ++tileY) {
                for (uint32_t tileX = range.cols().begin(); tileX < range.cols().end(); ++tileX) {
                    renderTile(tileX, tileY);
                }
            }
        });

    if ((verbose || statsMode) && lastReportedPercent.load(std::memory_order_relaxed) < 100u) {
        reportProgress(totalTiles);
        std::cout << std::endl;
    }

    CFAbsoluteTime renderEnd = CFAbsoluteTimeGetCurrent();

    out.linearRGB = std::move(linearRGB);
    out.albedoLinearRGB = std::move(albedoLinearRGB);
    out.normalLinearRGB = std::move(normalLinearRGB);
    out.width = width;
    out.height = height;
    out.samples = targetSamples;
    out.totalSeconds = renderEnd - renderStart;
    out.avgMsPerSample = (targetSamples > 0)
                             ? ((out.totalSeconds * 1000.0) / targetSamples)
                             : 0.0;
    out.performance.restirDiInitializeDispatchCount = restirDirectMode ? 1u : 0u;
    out.performance.restirDiInitialCandidateDispatchCount = restirDiInitialCandidateDispatchCount;
    out.performance.restirDiTemporalReuseDispatchCount = restirDiTemporalReuseDispatchCount;
    out.performance.restirDiSpatialReuseDispatchCount = restirDiSpatialReuseDispatchCount;
    out.performance.restirDiVisibilityDispatchCount = restirDiVisibilityDispatchCount;
    out.performance.restirDiApplyLightingDispatchCount = restirDiApplyLightingDispatchCount;
    out.performance.integrationDispatchCount = targetSamples;
    out.performance.totalComputeDispatchCount =
        out.performance.restirDiInitializeDispatchCount +
        out.performance.restirDiInitialCandidateDispatchCount +
        out.performance.restirDiTemporalReuseDispatchCount +
        out.performance.restirDiSpatialReuseDispatchCount +
        out.performance.restirDiVisibilityDispatchCount +
        out.performance.restirDiApplyLightingDispatchCount +
        out.performance.integrationDispatchCount;
    out.performance.restirDiReservoirBytesEach =
        restirDirectMode
            ? static_cast<uint64_t>(width) * static_cast<uint64_t>(height) *
                  sizeof(CpuRestirDiReservoirState)
            : 0u;
    out.performance.restirDiVisibilityBytes =
        restirDirectMode
            ? static_cast<uint64_t>(width) * static_cast<uint64_t>(height) *
                  sizeof(CpuRestirDiVisibilityState)
            : 0u;
    out.performance.restirDiResourceBytesTotal =
        restirDirectMode
            ? out.performance.restirDiReservoirBytesEach * 4u +
                  out.performance.restirDiVisibilityBytes
            : 0u;
    out.pbrMetrics.available = true;
    out.pbrMetrics.throughputNanInfCount = throughputNanInfCount;
    out.pbrMetrics.radianceNanInfCount = radianceNanInfCount;
    out.pbrMetrics.clampEventCount = clampEventCount;
    out.pbrMetrics.restirGiCandidateCount = restirGiCandidateCount.load(std::memory_order_relaxed);
    out.pbrMetrics.restirGiReservoirUpdateCount =
        restirGiReservoirUpdateCount.load(std::memory_order_relaxed);
    out.pbrMetrics.restirGiAcceptCount = restirGiAcceptCount.load(std::memory_order_relaxed);
    out.pbrMetrics.restirGiRejectCount = restirGiRejectCount.load(std::memory_order_relaxed);
    out.pbrMetrics.restirGiFallbackCount = restirGiFallbackCount.load(std::memory_order_relaxed);
    out.pbrMetrics.pathGuidingCandidateCount =
        pathGuidingCandidateCount.load(std::memory_order_relaxed);
    out.pbrMetrics.pathGuidingUsedCount =
        pathGuidingUsedCount.load(std::memory_order_relaxed);
    out.pbrMetrics.pathGuidingFallbackCount =
        pathGuidingFallbackCount.load(std::memory_order_relaxed);
    out.pbrMetrics.pathGuidingInvalidCount =
        pathGuidingInvalidCount.load(std::memory_order_relaxed);
    out.pbrMetrics.restirPtCandidateCount = restirPtCandidateCount.load(std::memory_order_relaxed);
    out.pbrMetrics.restirPtReservoirUpdateCount =
        restirPtReservoirUpdateCount.load(std::memory_order_relaxed);
    out.pbrMetrics.restirPtRejectedInvalidCount =
        restirPtRejectedInvalidCount.load(std::memory_order_relaxed);
    out.pbrMetrics.restirPtRejectedUnsupportedCount =
        restirPtRejectedUnsupportedCount.load(std::memory_order_relaxed);
    out.pbrMetrics.restirPtDebugRecordCount =
        restirPtDebugRecordCount.load(std::memory_order_relaxed);
    out.pbrMetrics.restirPtReuseCandidateCount =
        restirPtReuseCandidateCount.load(std::memory_order_relaxed);
    out.pbrMetrics.restirPtReuseAppliedCount =
        restirPtReuseAppliedCount.load(std::memory_order_relaxed);
    out.pbrMetrics.restirPtReuseRejectedInvalidCount =
        restirPtReuseRejectedInvalidCount.load(std::memory_order_relaxed);
    out.pbrMetrics.restirPtReuseFallbackCount =
        restirPtReuseFallbackCount.load(std::memory_order_relaxed);
    out.pbrMetrics.radianceCacheQueryCount =
        radianceCacheQueryCount.load(std::memory_order_relaxed);
    out.pbrMetrics.radianceCacheHitCount =
        radianceCacheHitCount.load(std::memory_order_relaxed);
    out.pbrMetrics.radianceCacheMissCount =
        radianceCacheMissCount.load(std::memory_order_relaxed);
    out.pbrMetrics.radianceCacheTrainCount =
        radianceCacheTrainCount.load(std::memory_order_relaxed);
    out.pbrMetrics.radianceCacheFallbackCount =
        radianceCacheFallbackCount.load(std::memory_order_relaxed);
    out.pbrMetrics.radianceCacheInvalidCount =
        radianceCacheInvalidCount.load(std::memory_order_relaxed);
    out.pbrMetrics.maxThroughputByBounce = std::move(maxThroughputByBounce);
    out.materialMrDebug = materialMrDebug;
    out.directLightAudit = directLightAudit;

    if (sceneData.scene) {
        rtcReleaseScene(sceneData.scene);
    }
    if (sceneData.device) {
        rtcReleaseDevice(sceneData.device);
    }

    return true;
#endif
}
