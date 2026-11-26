#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <bit>
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
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include <mach/mach.h>

#include "headless/EmbreeHeadlessRenderer.h"
#include "renderer/EnvImportanceSampler.h"
#include "MetalShaderTypes.h"
#include "shared/MipChain.h"
// EP8: the Tungsten source closures, compiled from the same headers as the Metal shaders (shaders/shared); they
// live in namespace pt_shared.
#include "shared/cpp_prelude.h"
#include "shared/source_closures.h"
#include "renderer/RadianceCacheTraining.h"
#include "renderer/SpectralSensor.h"
#include <simd/simd.h>
#include <sys/resource.h>
#include <chrono>

#if defined(PATH_TRACER_ENABLE_EMBREE)
#include <embree4/rtcore.h>
#include <tbb/blocked_range2d.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#endif

namespace {

// Everything in this namespace serves the Embree render path only; without Embree, render() is a stub.
#if defined(PATH_TRACER_ENABLE_EMBREE)

constexpr float kPi = 3.14159265358979323846f;
constexpr float kEpsilon = 1.0e-4f;
constexpr float kSpecularNeePdfFloor = 1.0e-4f;
constexpr float kMisWeightClampMin = 1.0e-4f;
constexpr float kMisWeightClampMax = 0.9999f;
constexpr uint32_t kInvalidTextureIndex = 0xFFFFFFFFu;

// Process start, as near as a static initializer gets: the load stages' wall times count from here.
const std::chrono::steady_clock::time_point kProcessStart = std::chrono::steady_clock::now();

// EP22: the process's memory at each load stage (phys_footprint counts compressed pages, as the OS's memory limits do),
// with the wall time and the process's CPU time (user + system, every thread) at that stage: a stage's CPU time over
// its wall time is how many threads it kept busy.
void LogLoadMemory(const char* stage) {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    const double cpuSeconds = static_cast<double>(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) +
                              static_cast<double>(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) * 1.0e-6;
    const double wallSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - kProcessStart).count();
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
        return;
    }
    std::ostringstream line;
    line << std::fixed << std::setprecision(2) << "[LoadMemory] " << stage << ": footprint "
         << static_cast<double>(info.phys_footprint) / 1073741824.0 << " GB (peak so far "
         << static_cast<double>(info.ledger_phys_footprint_peak) / 1073741824.0 << " GB), resident "
         << static_cast<double>(info.resident_size) / 1073741824.0 << " GB; at " << wallSeconds << " s wall, "
         << cpuSeconds << " s CPU";
    std::cout << line.str() << std::endl;
}

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
    // A mesh's own vertices in SceneResources, read for uv and uv1 (they are not transformed). Null for the
    // sphere and rectangle geometries, disks included.
    const PathTracer::SceneResources::MeshVertex* vertices = nullptr;
    const uint32_t* indices = nullptr;
    const simd::float4* spheres = nullptr;
    const uint32_t* materialIndices = nullptr;
    const PathTracerShaderTypes::RectData* disk = nullptr;
    uint32_t indexCount = 0;
    uint32_t count = 0;
    uint32_t materialIndex = 0;
    uint32_t triangleOffset = 0;
    // Index in SceneResources::meshes(), as Metal's MeshInfo index.
    uint32_t meshIndex = std::numeric_limits<uint32_t>::max();
    // The mesh's material has kMaterialFlagSingleScatterSpecular: a triangle without UV variation takes its unit
    // tangent frame for uv set 0, as Metal's triangle_surface_partials (unitFrameForConstantUv).
    bool unitFrameForConstantUv = false;
    // A mesh's world normals and tangents are computed at a hit from `vertices` (MeshWorldNormal, MeshWorldTangent)
    // rather than stored per instance: Pavilion's 21,089 instances kept 3.7 GB of them.
    bool worldFrameAtHit = false;
    simd::float3x3 normalMatrix = matrix_identity_float3x3;
    simd::float4x4 localToWorld = matrix_identity_float4x4;
    float transformDetSign = 1.0f;
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
    bool twoSidedAlphaCutout = false;  // Metal's material_is_two_sided_alpha_cutout; set when the hit is shaded.
    bool hasUv = false;
    bool hasUv1 = false;
    bool hasSurfacePartials0 = false;
    bool hasSurfacePartials1 = false;
    bool hasIgehyGradients0 = false;
    bool hasIgehyGradients1 = false;
    bool hasTangent = false;
    bool hasVertexTangent = false;
    bool authoredNormalWasFaced = false;
    // A mesh triangle's world edges p1 - p0, p2 - p0 and uv0 edges, for the source bump frame (SourceBumpNormal).
    bool hasSourceBumpFrame = false;
    simd::float3 sourceBumpQ1{0.0f, 0.0f, 0.0f};
    simd::float3 sourceBumpQ2{0.0f, 0.0f, 0.0f};
    simd::float2 sourceBumpE1{0.0f, 0.0f};
    simd::float2 sourceBumpE2{0.0f, 0.0f};
    GeometryType primitiveType = GeometryType::Mesh;
    uint32_t primitiveIndex = 0;
    uint32_t meshIndex = std::numeric_limits<uint32_t>::max();  // Set for mesh triangles only.
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
    bool isReferencePoint = false;
    bool inverseSquareFalloff = false;
    float jitterRadius = 0.0f;
    float legacyMinimumDistanceSq = 0.0f;
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
    bool twoSided = false;
    bool texturedEmission = false;
    const PathTracer::SceneResources::Mesh* texturedMesh = nullptr;
    const PathTracerShaderTypes::MaterialData* texturedMaterial = nullptr;
    uint32_t localTriangle = 0u;
};

struct DirectionalLightInfo {
    simd::float3 direction{0.0f, 1.0f, 0.0f};
    simd::float3 emission{0.0f, 0.0f, 0.0f};
    float cosDelta = 0.0f;  // 1 - cos(angular radius); zero is a delta sun.
};

bool AreaPrimitiveIsDisk(const PathTracerShaderTypes::RectData& rect) {
    return rect.materialTwoSided.z ==
           static_cast<uint32_t>(PathTracerShaderTypes::AreaPrimitiveShape::Disk);
}

bool AreaPrimitiveIsReferencePoint(const PathTracerShaderTypes::RectData& rect) {
    return rect.materialTwoSided.z ==
           static_cast<uint32_t>(PathTracerShaderTypes::AreaPrimitiveShape::ReferencePoint);
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
    int mediumEvent = 0;        // a solid dielectric refraction: +1 at a front face, -1 at a back face (Metal's)
    // Set at a subsurface walk's exit vertex, as Metal sets it there: ReSTIR GI, ReSTIR PT, path guiding and the
    // radiance cache refuse the sample, because their per-pixel state belongs to the primary hit, not the exit point.
    bool isBssrdf = false;
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

// A cacheable vertex of a radiance cache training path (RadianceCacheTraining.h): its cell and metadata, the
// throughput at the cache query and the radiance accumulated by then.
struct CpuRadianceCacheTrainRecord {
    size_t cellIndex = 0u;
    uint32_t domainKey = 0u;
    uint32_t materialType = 0u;
    simd::float3 position{0.0f, 0.0f, 0.0f};
    simd::float3 normal{0.0f, 1.0f, 0.0f};
    float roughness = 0.0f;
    simd::float3 throughputAtQuery{0.0f, 0.0f, 0.0f};
    simd::float3 radianceBaseline{0.0f, 0.0f, 0.0f};
};

struct CpuTexture {
    bool normalMap = false;
    struct MipLevel {
        uint32_t width = 0;
        uint32_t height = 0;
        std::vector<float> rgba;     // float texels; empty when rgba8 holds the level
        std::vector<uint8_t> rgba8;  // a level decoded from 8-bit texels, read through the tables below (EP22)
        // A level computed in float whose values are 8-bit steps (KeepFloatLevelAsByteSteps) is kept as bytes too, but
        // read through its own tables: colour at [0, 256), alpha at [256, 512). Null for every other level.
        std::unique_ptr<const std::array<float, 512>> byteTables;
    };

    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<MipLevel> mipLevels;  // level 0 is the full-size texture
    // EP22: the float each byte decodes to, for colour and for alpha, taken from SceneResources' own decode of this
    // texture and checked on every texel of every level kept as bytes, so those levels read exactly the floats they
    // replace at a quarter of the memory.
    std::array<float, 256> rgbFromByte{};
    std::array<float, 256> alphaFromByte{};
    PathTracer::MaterialTextureSamplerDesc samplerDesc{};
};

simd::float4 CpuTextureTexel(const CpuTexture& texture, const CpuTexture::MipLevel& level, size_t index) {
    if (!level.rgba8.empty()) {
        const uint8_t* texel = level.rgba8.data() + index;
        const float* rgb = level.byteTables ? level.byteTables->data() : texture.rgbFromByte.data();
        const float* alpha = level.byteTables ? level.byteTables->data() + 256 : texture.alphaFromByte.data();
        return simd_make_float4(rgb[texel[0]], rgb[texel[1]], rgb[texel[2]], alpha[texel[3]]);
    }
    return simd_make_float4(level.rgba[index], level.rgba[index + 1u], level.rgba[index + 2u], level.rgba[index + 3u]);
}

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
    float minSpecularPdf = 1.0e-8f;
    float maxContribution = 0.0f;
    float enabled = 0.0f;
    // 0 = RGB; otherwise the wavelengths per path in spectral transport (1 or 3), as Metal's FireflyClampParams.
    uint32_t spectralChannels = 0u;
};

// Metal's clamp_brightness (closures_eval.metal): Rec.709 luminance in RGB; in spectral transport the mean over the
// wavelength channels, so the luminance-mode clamps do not depend on which channel carries which wavelength.
float Luminance(const simd::float3& value);
float ClampBrightness(const simd::float3& value, const FireflyClampParams& params) {
    if (params.spectralChannels == 0u) {
        return Luminance(value);
    }
    return params.spectralChannels == 1u ? value.x : (value.x + value.y + value.z) * (1.0f / 3.0f);
}

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

bool ComputeTriangleSurfacePartials(const simd::float3& p0,
                                    const simd::float3& p1,
                                    const simd::float3& p2,
                                    const simd::float2& uv0,
                                    const simd::float2& uv1,
                                    const simd::float2& uv2,
                                    simd::float3& dPduOut,
                                    simd::float3& dPdvOut,
                                    float& uvPerWorldOut,
                                    bool unitFrameForConstantUv = false) {
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
    if (unitFrameForConstantUv && worldArea > 1.0e-12f) {
        // As Metal: a triangle without UV variation falls back to its unit tangent frame (one UV unit per world
        // unit), so the ray-cone LOD comes from the footprint instead of mip 0.
        const simd::float3 tangentFallback = simd::normalize(edge1);
        const simd::float3 bitangentFallback = simd::normalize(simd::cross(simd::normalize(simd::cross(edge1, edge2)), tangentFallback));
        if (IsFinite(tangentFallback) && IsFinite(bitangentFallback)) {
            dPduOut = tangentFallback;
            dPdvOut = bitangentFallback;
            uvPerWorldOut = 1.0f;
            return true;
        }
    }
    return false;
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

void PopulateFirstHitUvGradients(HitInfo& hit, const Ray& ray, const PrimaryRayDiff& rayDiff,
                                bool referenceGradientPlane = false) {
    const simd::float3 gradientNormal = referenceGradientPlane ? hit.shadingNormal : hit.normal;
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
                                                              gradientNormal,
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
                                                              gradientNormal,
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

simd::float3 SampleEnvironment(const EnvironmentMap& env,
                               const simd::float3& direction,
                               float rotation,
                               float intensity) {
    return SampleEnvironmentData(env.rgba, env.width, env.height, direction, rotation, intensity);
}

simd::float3 EvaluateBackground(const PathTracer::RenderSettings& settings,
                                const EnvironmentMap* env,
                                const simd::float3& direction) {
    switch (settings.backgroundMode) {
        case PathTracer::RenderSettings::BackgroundMode::Solid:
            return settings.backgroundColor;
        case PathTracer::RenderSettings::BackgroundMode::Environment:
            if (env) {
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

// True when the minification filter picks a texel within a level (NEAREST, NEAREST_MIPMAP_NEAREST,
// NEAREST_MIPMAP_LINEAR); Metal maps it to MTLSamplerMinMagFilterNearest (SceneResources::materialSamplerIndexForDesc).
bool CpuTextureMinNearest(const PathTracer::MaterialTextureSamplerDesc& samplerDesc) {
    const int32_t minFilter = samplerDesc.minFilter;
    return minFilter == 9728 || minFilter == 9984 || minFilter == 9986;
}

// True when Metal samples this texture at its nearest texel at every LOD: both filters pick the nearest texel.
bool CpuTextureNearestTexel(const PathTracer::MaterialTextureSamplerDesc& samplerDesc) {
    return samplerDesc.magFilter == 9728 && CpuTextureMinNearest(samplerDesc);
}

// True when one of the two filters picks the nearest texel and the other filters linearly.
bool CpuTextureMixesNearestAndLinear(const PathTracer::MaterialTextureSamplerDesc& samplerDesc) {
    return (samplerDesc.magFilter == 9728) != CpuTextureMinNearest(samplerDesc);
}

// Metal's texel filter for one sample: the magnification filter when the sample is magnified (LOD at most 0, as the
// GPU decides it, which a level-0 read is), else the minification filter. The two differ only for a texture that mixes a
// nearest and a linear filter.
bool CpuTextureSampleNearest(const PathTracer::MaterialTextureSamplerDesc& samplerDesc, bool magnified) {
    return magnified ? samplerDesc.magFilter == 9728 : CpuTextureMinNearest(samplerDesc);
}

simd::float4 SampleCpuTextureData(const CpuTexture& texture, const CpuTexture::MipLevel& level,
                                  const simd::float2& uv, bool magnified = true) {
    const uint32_t width = level.width;
    const uint32_t height = level.height;
    const PathTracer::MaterialTextureSamplerDesc& samplerDesc = texture.samplerDesc;
    const bool normalMap = texture.normalMap;
    if (width == 0 || height == 0 || (level.rgba.empty() && level.rgba8.empty())) {
        return simd_make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    }

    float u = ApplyWrapMode(uv.x, samplerDesc.wrapS);
    float v = ApplyWrapMode(uv.y, samplerDesc.wrapT);
    if (CpuTextureSampleNearest(samplerDesc, magnified)) {
        // The texel that contains the wrapped coordinate, as Metal's nearest filter picks it.
        const int x = std::clamp(static_cast<int>(std::floor(u * static_cast<float>(width))), 0,
                                 static_cast<int>(width) - 1);
        const int y = std::clamp(static_cast<int>(std::floor(v * static_cast<float>(height))), 0,
                                 static_cast<int>(height) - 1);
        return CpuTextureTexel(texture, level, (static_cast<size_t>(y) * width + static_cast<size_t>(x)) * 4ull);
    }
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
        return CpuTextureTexel(texture, level, (static_cast<size_t>(py) * width + static_cast<size_t>(px)) * 4ull);
    };

    simd::float4 c00 = fetch(x0, y0);
    simd::float4 c10 = fetch(x1, y0);
    simd::float4 c01 = fetch(x0, y1);
    simd::float4 c11 = fetch(x1, y1);
    if (normalMap) {
        simd::float4 n0 = c00 + (c10 - c00) * tx;
        simd::float4 n1 = c01 + (c11 - c01) * tx;
        return n0 + (n1 - n0) * ty;
    }
    simd::float4 c0 = c00 * (1.0f - tx) + c10 * tx;
    simd::float4 c1 = c01 * (1.0f - tx) + c11 * tx;
    return c0 * (1.0f - ty) + c1 * ty;
}

simd::float4 SampleCpuTexture(const CpuTexture& texture, const simd::float2& uv) {
    if (texture.mipLevels.empty()) {
        return simd_make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
    return SampleCpuTextureData(texture, texture.mipLevels.front(), uv);
}

// `magnified` is whether the sample's LOD is at most 0; an explicit level read has the level as its LOD.
simd::float4 SampleCpuTextureLevel(const CpuTexture& texture, const simd::float2& uv, uint32_t levelIndex,
                                   bool magnified) {
    if (texture.mipLevels.empty()) {
        return SampleCpuTexture(texture, uv);
    }
    levelIndex = std::min<uint32_t>(levelIndex, static_cast<uint32_t>(texture.mipLevels.size() - 1u));
    return SampleCpuTextureData(texture, texture.mipLevels[levelIndex], uv, magnified);
}

simd::float4 SampleCpuTextureLevel(const CpuTexture& texture, const simd::float2& uv, uint32_t levelIndex) {
    return SampleCpuTextureLevel(texture, uv, levelIndex, levelIndex == 0u);
}

// Mip mode of the glTF minification filter, as the Metal sampler maps it
// (SceneResources::materialSamplerIndexForDesc): NEAREST and LINEAR are not
// mipmapped and read level 0 only; *_MIPMAP_NEAREST reads the nearest level.
enum class CpuTextureMipMode { None, Nearest, Linear };

CpuTextureMipMode CpuTextureMipModeFor(const CpuTexture& texture) {
    switch (texture.samplerDesc.minFilter) {
        case 9728:  // NEAREST
        case 9729:  // LINEAR
            return CpuTextureMipMode::None;
        case 9984:  // NEAREST_MIPMAP_NEAREST
        case 9985:  // LINEAR_MIPMAP_NEAREST
            return CpuTextureMipMode::Nearest;
        default:    // *_MIPMAP_LINEAR, or unset (LINEAR_MIPMAP_LINEAR)
            return CpuTextureMipMode::Linear;
    }
}

// Metal enables anisotropy only for linear minification and magnification filters with mips
// (SceneResources::materialSamplerIndexForDesc), so never on a texture that mixes a nearest and a linear filter.
bool CpuTextureAnisotropyEnabled(const CpuTexture& texture) {
    const int32_t minFilter = texture.samplerDesc.minFilter;
    return texture.samplerDesc.magFilter != 9728 &&
           minFilter != 9728 && minFilter != 9729 && minFilter != 9984 && minFilter != 9986;
}

// `magnified`: whether the GPU magnifies this sample, which selects the magnification filter (CpuTextureSampleNearest).
simd::float4 SampleCpuTexture(const CpuTexture& texture, const simd::float2& uv, float lod, bool magnified) {
    if (texture.mipLevels.empty()) {
        return SampleCpuTexture(texture, uv);
    }
    const CpuTextureMipMode mipMode = CpuTextureMipModeFor(texture);
    if (texture.mipLevels.size() == 1u || mipMode == CpuTextureMipMode::None) {
        return SampleCpuTextureLevel(texture, uv, 0u, magnified);
    }
    float clampedLod = std::clamp(lod, 0.0f, static_cast<float>(texture.mipLevels.size() - 1u));
    if (mipMode == CpuTextureMipMode::Nearest) {
        return SampleCpuTextureLevel(texture, uv, static_cast<uint32_t>(std::floor(clampedLod + 0.5f)), magnified);
    }
    uint32_t level0 = static_cast<uint32_t>(std::floor(clampedLod));
    uint32_t level1 = std::min<uint32_t>(level0 + 1u, static_cast<uint32_t>(texture.mipLevels.size() - 1u));
    float t = clampedLod - static_cast<float>(level0);
    simd::float4 c0 = SampleCpuTextureLevel(texture, uv, level0, magnified);
    simd::float4 c1 = SampleCpuTextureLevel(texture, uv, level1, magnified);
    if (texture.normalMap) return c0 + (c1 - c0) * t;
    return c0 * (1.0f - t) + c1 * t;
}

// An isotropic sample is magnified when its LOD is at most 0.
simd::float4 SampleCpuTexture(const CpuTexture& texture, const simd::float2& uv, float lod) {
    return SampleCpuTexture(texture, uv, lod, !(lod > 0.0f));
}

simd::float4 SampleCpuTextureAnisotropic(const CpuTexture& texture,
                                         const simd::float2& uv,
                                         const simd::float2& dUVdx,
                                         const simd::float2& dUVdy,
                                         float fallbackLod,
                                         float textureLodBias = 0.0f) {
    if (texture.mipLevels.size() <= 1u || texture.width == 0u || texture.height == 0u ||
        !CpuTextureAnisotropyEnabled(texture) ||
        CpuTextureMipModeFor(texture) == CpuTextureMipMode::None) {
        return SampleCpuTexture(texture, uv, fallbackLod);
    }
    if (!std::isfinite(dUVdx.x) || !std::isfinite(dUVdx.y) ||
        !std::isfinite(dUVdy.x) || !std::isfinite(dUVdy.y)) {
        return SampleCpuTexture(texture, uv, fallbackLod);
    }
    const float gradientScale = std::exp2(textureLodBias);
    const simd::float2 scaledDx = dUVdx * gradientScale;
    const simd::float2 scaledDy = dUVdy * gradientScale;
    simd::float2 texelDx = simd_make_float2(scaledDx.x * static_cast<float>(texture.width),
                                            scaledDx.y * static_cast<float>(texture.height));
    simd::float2 texelDy = simd_make_float2(scaledDy.x * static_cast<float>(texture.width),
                                            scaledDy.y * static_cast<float>(texture.height));
    const float lenX = simd::length(texelDx);
    const float lenY = simd::length(texelDy);
    if (!std::isfinite(lenX) || !std::isfinite(lenY) || std::max(lenX, lenY) <= 1.0f) {
        return SampleCpuTexture(texture, uv, fallbackLod);
    }
    const bool xMajor = lenX >= lenY;
    const float majorLen = std::max(xMajor ? lenX : lenY, 1.0f);
    // The footprint's minor axis is its area over the major axis, as Metal's sampler measures it (a level-index probe
    // with skewed gradient pairs); the shorter gradient overstates it when the two gradients are nearly parallel.
    const float footprintArea = std::fabs(texelDx.x * texelDy.y - texelDx.y * texelDy.x);
    const float minorLen = std::max(footprintArea / majorLen, 1.0f);
    const simd::float2 majorUv = xMajor ? scaledDx : scaledDy;
    // Metal's sampler filters a texture with a side that is not a power of two with at most 4x anisotropy, whatever
    // the sampler's maxAnisotropy (measured on this hardware: RGBA8 and R32Float, 1974x1984, 1023x1024, 1000x1000).
    const bool powerOfTwo = PathTracer::MipChain::IsPowerOfTwo(texture.width) &&
                            PathTracer::MipChain::IsPowerOfTwo(texture.height);
    const uint32_t maxAnisotropy =
        std::min(std::clamp(texture.samplerDesc.maxAnisotropy, 1u, 16u), powerOfTwo ? 16u : 4u);
    // When the footprint is more anisotropic than the sampler allows, the taps cannot cover the major axis at the
    // minor axis's level. Anisotropic filtering then reads the level of majorLen / maxAnisotropy, so the taps cover it.
    const float minorLod =
        std::clamp(std::log2(std::max(std::max(minorLen, majorLen / static_cast<float>(maxAnisotropy)), 1.0e-8f)),
                   0.0f,
                   static_cast<float>(texture.mipLevels.size() - 1u));
    const uint32_t taps =
        std::clamp(static_cast<uint32_t>(std::ceil(majorLen / minorLen)), 1u, maxAnisotropy);
    if (taps <= 1u) {
        return SampleCpuTexture(texture, uv, minorLod);
    }
    simd::float4 reference = texture.normalMap ? SampleCpuTexture(texture, uv, minorLod) : simd_make_float4(0.0f);
    simd::float4 accum = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint32_t i = 0; i < taps; ++i) {
        float offset = (static_cast<float>(i) + 0.5f) / static_cast<float>(taps) - 0.5f;
        accum += SampleCpuTexture(texture, uv + majorUv * offset, minorLod) - reference;
    }
    return reference + accum / static_cast<float>(taps);
}

simd::float3 DecodeNormalMap(const simd::float3& encoded, float normalScale) {
    simd::float3 n = encoded * 2.0f - simd_make_float3(1.0f, 1.0f, 1.0f);
    n.x *= normalScale;
    n.y *= normalScale;
    return simd::dot(n, n) > 1.0e-12f ? simd::normalize(n) : simd_make_float3(0.0f, 0.0f, 1.0f);
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
                             float surfaceFootprintWorld,
                             float textureLodBias = 0.0f) {
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
    return std::max(lod + textureLodBias, 0.0f);
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
    params.minSpecularPdf = std::max(settings.minSpecularPdf, 1.0e-8f);
    params.maxContribution = std::max(settings.fireflyClampMaxContribution, 0.0f);
    params.enabled = settings.fireflyClampEnabled ? 1.0f : 0.0f;
    // The clamped range always spans at least 1 nm (SpectralSensor::ClampRange), so the mode alone decides.
    params.spectralChannels =
        settings.spectralRenderingMode == PathTracer::RenderSettings::SpectralRenderingMode::HeroWavelength
            ? (settings.spectralWavelengthsPerPath == 1u ? 1u : 3u)
            : 0u;
    return params;
}

// The rule RenderLoop.mm's EnvFlagEnabled applies to Metal's environment switches: set, non-empty, and not
// 0, false or no.
bool MetalEnvFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return false;
    }
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "no") != 0 && std::strcmp(value, "NO") != 0;
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
        float lum = ClampBrightness(positive, params);
        simd::float3 throughputPositive = simd::max(throughput, simd_make_float3(0.0f, 0.0f, 0.0f));
        float throughputLum = ClampBrightness(throughputPositive, params);
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
    float lum = ClampBrightness(positive, params);
    if (lum > params.throughputClamp && lum > 0.0f) {
        float scale = params.throughputClamp / std::max(lum, 1.0e-6f);
        if (outClamped) {
            *outClamped = true;
        }
        return throughput * scale;
    }
    return throughput;
}

simd::float3 SanitizeSpecularValue(const simd::float3& value) {
    if (!IsFinite(value)) {
        return simd_make_float3(0.0f, 0.0f, 0.0f);
    }
    // Mirrors sanitize_specular_value (closures_eval.metal): sanitize, never cap f.
    return simd::max(value, simd_make_float3(0.0f, 0.0f, 0.0f));
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

// Generated offline by generate_ess.py, a local tool script that is not part of this repository.
// Ess: separable Smith GGX, alpha=max(r*r,1e-4), 49 roughness x 49 sqrt(NoV).
static constexpr float kGgxSingleScatterAlbedo[2401] = {
    0.9360666f, 0.9690419f, 0.9982691f, 0.9998319f, 0.9999479f, 0.9999787f, 0.9999898f, 0.9999945f, 0.9999968f, 0.9999980f, 0.9999987f, 0.9999991f, 0.9999994f, 0.9999995f, 0.9999997f, 0.9999997f, 0.9999998f, 0.9999998f, 0.9999999f, 0.9999999f, 0.9999999f, 0.9999999f, 0.9999999f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f,
    0.9361361f, 0.8816224f, 0.9642667f, 0.9927504f, 0.9978852f, 0.9991537f, 0.9998001f, 0.9998946f, 0.9999386f, 0.9999618f, 0.9999750f, 0.9999829f, 0.9999880f, 0.9999913f, 0.9999935f, 0.9999951f, 0.9999962f, 0.9999971f, 0.9999977f, 0.9999981f, 0.9999985f, 0.9999988f, 0.9999990f, 0.9999992f, 0.9999993f, 0.9999994f, 0.9999995f, 0.9999996f, 0.9999996f, 0.9999997f, 0.9999997f, 0.9999998f, 0.9999998f, 0.9999998f, 0.9999999f, 0.9999999f, 0.9999999f, 0.9999999f, 0.9999999f, 0.9999999f, 0.9999999f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f, 1.0000000f,
    0.9361514f, 0.9130131f, 0.8816218f, 0.9203838f, 0.9642662f, 0.9846032f, 0.9927503f, 0.9962251f, 0.9978853f, 0.9986857f, 0.9991540f, 0.9996917f, 0.9998008f, 0.9998580f, 0.9998953f, 0.9999210f, 0.9999393f, 0.9999526f, 0.9999625f, 0.9999700f, 0.9999757f, 0.9999801f, 0.9999836f, 0.9999864f, 0.9999887f, 0.9999905f, 0.9999920f, 0.9999932f, 0.9999942f, 0.9999951f, 0.9999958f, 0.9999964f, 0.9999969f, 0.9999974f, 0.9999978f, 0.9999981f, 0.9999984f, 0.9999986f, 0.9999988f, 0.9999990f, 0.9999992f, 0.9999993f, 0.9999995f, 0.9999996f, 0.9999997f, 0.9999998f, 0.9999999f, 0.9999999f, 1.0000000f,
    0.9361532f, 0.9258218f, 0.8977168f, 0.8816192f, 0.9026362f, 0.9376927f, 0.9642641f, 0.9797618f, 0.9881630f, 0.9927498f, 0.9953644f, 0.9969226f, 0.9978859f, 0.9985034f, 0.9988483f, 0.9991554f, 0.9995859f, 0.9997437f, 0.9998039f, 0.9998448f, 0.9998751f, 0.9998984f, 0.9999165f, 0.9999309f, 0.9999424f, 0.9999517f, 0.9999593f, 0.9999656f, 0.9999708f, 0.9999751f, 0.9999788f, 0.9999818f, 0.9999845f, 0.9999867f, 0.9999886f, 0.9999903f, 0.9999917f, 0.9999930f, 0.9999941f, 0.9999951f, 0.9999959f, 0.9999966f, 0.9999973f, 0.9999979f, 0.9999984f, 0.9999989f, 0.9999993f, 0.9999997f, 1.0000000f,
    0.9361498f, 0.9303861f, 0.9130073f, 0.8907768f, 0.8816118f, 0.8947193f, 0.9203716f, 0.9454729f, 0.9642585f, 0.9767108f, 0.9846001f, 0.9895774f, 0.9927495f, 0.9948450f, 0.9962264f, 0.9971820f, 0.9978874f, 0.9983989f, 0.9986893f, 0.9989281f, 0.9991587f, 0.9994510f, 0.9997028f, 0.9997699f, 0.9998121f, 0.9998440f, 0.9998692f, 0.9998897f, 0.9999066f, 0.9999206f, 0.9999323f, 0.9999422f, 0.9999506f, 0.9999578f, 0.9999639f, 0.9999692f, 0.9999738f, 0.9999778f, 0.9999813f, 0.9999843f, 0.9999870f, 0.9999894f, 0.9999914f, 0.9999933f, 0.9999949f, 0.9999964f, 0.9999977f, 0.9999989f, 1.0000000f,
    0.9361346f, 0.9324719f, 0.9212103f, 0.9038151f, 0.8872933f, 0.8815947f, 0.8905932f, 0.9095501f, 0.9309634f, 0.9497739f, 0.9642462f, 0.9746372f, 0.9818715f, 0.9868588f, 0.9903208f, 0.9927484f, 0.9944739f, 0.9957235f, 0.9966848f, 0.9973256f, 0.9978907f, 0.9983308f, 0.9985875f, 0.9987925f, 0.9989788f, 0.9991653f, 0.9993851f, 0.9996689f, 0.9997523f, 0.9997968f, 0.9998294f, 0.9998555f, 0.9998771f, 0.9998953f, 0.9999107f, 0.9999239f, 0.9999353f, 0.9999452f, 0.9999538f, 0.9999614f, 0.9999680f, 0.9999738f, 0.9999789f, 0.9999835f, 0.9999875f, 0.9999912f, 0.9999944f, 0.9999973f, 0.9999999f,
    0.9360860f, 0.9335551f, 0.9257578f, 0.9129522f, 0.8976601f, 0.8853319f, 0.8815572f, 0.8881612f, 0.9025676f, 0.9203152f, 0.9376351f, 0.9524678f, 0.9642238f, 0.9731342f, 0.9797367f, 0.9845840f, 0.9881489f, 0.9907772f, 0.9927439f, 0.9942181f, 0.9953644f, 0.9962272f, 0.9969271f, 0.9974190f, 0.9978951f, 0.9982833f, 0.9985164f, 0.9986995f, 0.9988614f, 0.9990164f, 0.9991747f, 0.9993571f, 0.9996368f, 0.9997508f, 0.9998004f, 0.9998342f, 0.9998608f, 0.9998829f, 0.9999018f, 0.9999180f, 0.9999322f, 0.9999445f, 0.9999554f, 0.9999651f, 0.9999736f, 0.9999812f, 0.9999880f, 0.9999941f, 0.9999995f,
    0.9359633f, 0.9341098f, 0.9284128f, 0.9188502f, 0.9063293f, 0.8934533f, 0.8840785f, 0.8814683f, 0.8865601f, 0.8978307f, 0.9125023f, 0.9278824f, 0.9421147f, 0.9542840f, 0.9641797f, 0.9719792f, 0.9780217f, 0.9826624f, 0.9862258f, 0.9889625f, 0.9910860f, 0.9927320f, 0.9940250f, 0.9950715f, 0.9958606f, 0.9965654f, 0.9970769f, 0.9974804f, 0.9978955f, 0.9982458f, 0.9984604f, 0.9986270f, 0.9987724f, 0.9989082f, 0.9990417f, 0.9991818f, 0.9993426f, 0.9995991f, 0.9997664f, 0.9998261f, 0.9998624f, 0.9998898f, 0.9999123f, 0.9999316f, 0.9999483f, 0.9999630f, 0.9999761f, 0.9999876f, 0.9999980f,
    0.9357265f, 0.9343089f, 0.9299624f, 0.9226171f, 0.9125872f, 0.9010908f, 0.8903674f, 0.8830809f, 0.8812215f, 0.8853026f, 0.8943621f, 0.9065899f, 0.9200672f, 0.9332667f, 0.9452322f, 0.9555272f, 0.9640784f, 0.9710207f, 0.9765784f, 0.9809978f, 0.9845018f, 0.9872864f, 0.9895032f, 0.9912761f, 0.9926925f, 0.9938476f, 0.9948023f, 0.9955551f, 0.9961917f, 0.9967510f, 0.9971559f, 0.9975007f, 0.9978709f, 0.9981921f, 0.9983912f, 0.9985427f, 0.9986732f, 0.9987929f, 0.9989091f, 0.9990269f, 0.9991541f, 0.9993039f, 0.9995246f, 0.9997929f, 0.9998750f, 0.9999169f, 0.9999463f, 0.9999698f, 0.9999896f,
    0.9353646f, 0.9342429f, 0.9308137f, 0.9250028f, 0.9169062f, 0.9070930f, 0.8967978f, 0.8878056f, 0.8819611f, 0.8805345f, 0.8838269f, 0.8912055f, 0.9014480f, 0.9131736f, 0.9251686f, 0.9365553f, 0.9468137f, 0.9557204f, 0.9632566f, 0.9695223f, 0.9746772f, 0.9788889f, 0.9823188f, 0.9851196f, 0.9874029f, 0.9892747f, 0.9908168f, 0.9920777f, 0.9931382f, 0.9940215f, 0.9947633f, 0.9953585f, 0.9959319f, 0.9963714f, 0.9967152f, 0.9970226f, 0.9973651f, 0.9976686f, 0.9978530f, 0.9979871f, 0.9980962f, 0.9981906f, 0.9982752f, 0.9983539f, 0.9984301f, 0.9985091f, 0.9986036f, 0.9987697f, 0.9993496f,
    0.9348755f, 0.9339640f, 0.9311864f, 0.9264711f, 0.9198382f, 0.9115469f, 0.9022663f, 0.8931033f, 0.8854477f, 0.8807553f, 0.8797146f, 0.8825044f, 0.8886887f, 0.8974120f, 0.9076630f, 0.9184900f, 0.9291358f, 0.9390797f, 0.9480280f, 0.9558614f, 0.9625848f, 0.9682756f, 0.9730474f, 0.9770273f, 0.9803379f, 0.9830840f, 0.9853666f, 0.9872684f, 0.9888587f, 0.9901962f, 0.9913042f, 0.9922573f, 0.9930433f, 0.9937488f, 0.9943024f, 0.9948056f, 0.9952769f, 0.9956201f, 0.9959029f, 0.9961672f, 0.9964805f, 0.9967681f, 0.9969343f, 0.9970478f, 0.9971353f, 0.9972074f, 0.9972688f, 0.9973220f, 0.9973688f,
    0.9342409f, 0.9334842f, 0.9311841f, 0.9272788f, 0.9217586f, 0.9147371f, 0.9065756f, 0.8980835f, 0.8901638f, 0.8838195f, 0.8799713f, 0.8791901f, 0.8815844f, 0.8868415f, 0.8943481f, 0.9033522f, 0.9131077f, 0.9229758f, 0.9324731f, 0.9412814f, 0.9492248f, 0.9562387f, 0.9623359f, 0.9675783f, 0.9720496f, 0.9758439f, 0.9790533f, 0.9817684f, 0.9840646f, 0.9860072f, 0.9876510f, 0.9890464f, 0.9902491f, 0.9912581f, 0.9921465f, 0.9928807f, 0.9935642f, 0.9941105f, 0.9945787f, 0.9950630f, 0.9954463f, 0.9957404f, 0.9959951f, 0.9962467f, 0.9965752f, 0.9969033f, 0.9970862f, 0.9972061f, 0.9972958f,
    0.9334313f, 0.9327918f, 0.9308514f, 0.9275569f, 0.9228946f, 0.9169010f, 0.9098669f, 0.9022125f, 0.8945032f, 0.8875210f, 0.8820747f, 0.8788269f, 0.8781767f, 0.8802002f, 0.8846692f, 0.8911337f, 0.8990259f, 0.9077583f, 0.9168011f, 0.9257232f, 0.9342108f, 0.9420600f, 0.9491636f, 0.9554857f, 0.9610405f, 0.9658783f, 0.9700608f, 0.9736603f, 0.9767526f, 0.9794011f, 0.9816692f, 0.9836149f, 0.9852867f, 0.9867236f, 0.9879495f, 0.9890268f, 0.9899372f, 0.9907546f, 0.9914282f, 0.9920636f, 0.9925880f, 0.9930175f, 0.9934277f, 0.9938508f, 0.9941368f, 0.9943434f, 0.9945050f, 0.9946400f, 0.9948315f,
    0.9324150f, 0.9318659f, 0.9302026f, 0.9273802f, 0.9233828f, 0.9182337f, 0.9121512f, 0.9052897f, 0.8980574f, 0.8910110f, 0.8847853f, 0.8800090f, 0.8772820f, 0.8767937f, 0.8785902f, 0.8825023f, 0.8881923f, 0.8952238f, 0.9031286f, 0.9114669f, 0.9198629f, 0.9280223f, 0.9357344f, 0.9428663f, 0.9493480f, 0.9551618f, 0.9603209f, 0.9648646f, 0.9688427f, 0.9723136f, 0.9753291f, 0.9779497f, 0.9802259f, 0.9822007f, 0.9839131f, 0.9854081f, 0.9867158f, 0.9878414f, 0.9888469f, 0.9897050f, 0.9904947f, 0.9911483f, 0.9917648f, 0.9923150f, 0.9927520f, 0.9931437f, 0.9936275f, 0.9940379f, 0.9943113f,
    0.9311595f, 0.9306815f, 0.9292353f, 0.9267862f, 0.9233119f, 0.9188703f, 0.9135277f, 0.9073891f, 0.9007196f, 0.8939253f, 0.8875938f, 0.8821338f, 0.8780237f, 0.8756356f, 0.8751864f, 0.8767086f, 0.8800694f, 0.8850247f, 0.8912735f, 0.8984005f, 0.9060390f, 0.9138648f, 0.9216093f, 0.9290673f, 0.9360955f, 0.9426031f, 0.9485439f, 0.9539071f, 0.9587065f, 0.9629734f, 0.9667487f, 0.9700729f, 0.9729986f, 0.9755644f, 0.9778150f, 0.9797895f, 0.9815238f, 0.9830456f, 0.9843758f, 0.9855631f, 0.9865945f, 0.9875230f, 0.9883190f, 0.9890718f, 0.9896901f, 0.9902528f, 0.9908378f, 0.9912659f, 0.9915830f,
    0.9296326f, 0.9292112f, 0.9279375f, 0.9257841f, 0.9227252f, 0.9188455f, 0.9141112f, 0.9086042f, 0.9025239f, 0.8962251f, 0.8899640f, 0.8841423f, 0.8792094f, 0.8756120f, 0.8735548f, 0.8731933f, 0.8745531f, 0.8775358f, 0.8819453f, 0.8875089f, 0.8939275f, 0.9009011f, 0.9081517f, 0.9154445f, 0.9226017f, 0.9294592f, 0.9359137f, 0.9419015f, 0.9473927f, 0.9523809f, 0.9568797f, 0.9609136f, 0.9645145f, 0.9677187f, 0.9705614f, 0.9730828f, 0.9753145f, 0.9772886f, 0.9790336f, 0.9805842f, 0.9819628f, 0.9831724f, 0.9842634f, 0.9852230f, 0.9860923f, 0.9868340f, 0.9875696f, 0.9881522f, 0.9885970f,
    0.9278021f, 0.9274262f, 0.9262910f, 0.9243747f, 0.9216652f, 0.9182165f, 0.9139840f, 0.9090163f, 0.9035445f, 0.8976635f, 0.8916184f, 0.8858313f, 0.8805540f, 0.8761258f, 0.8728672f, 0.8710460f, 0.8707305f, 0.8719318f, 0.8745711f, 0.8784933f, 0.8834881f, 0.8893129f, 0.8957145f, 0.9024562f, 0.9093401f, 0.9161870f, 0.9228414f, 0.9291963f, 0.9351783f, 0.9407425f, 0.9458672f, 0.9505506f, 0.9548042f, 0.9586478f, 0.9621059f, 0.9652104f, 0.9679878f, 0.9704692f, 0.9726853f, 0.9746642f, 0.9764319f, 0.9780094f, 0.9794102f, 0.9806786f, 0.9818047f, 0.9828170f, 0.9837224f, 0.9845615f, 0.9852169f,
    0.9256360f, 0.9252971f, 0.9242741f, 0.9225493f, 0.9201251f, 0.9170224f, 0.9131995f, 0.9087431f, 0.9037337f, 0.8982686f, 0.8926117f, 0.8869123f, 0.8815072f, 0.8766837f, 0.8726711f, 0.8697375f, 0.8680964f, 0.8677896f, 0.8688339f, 0.8711626f, 0.8746517f, 0.8791325f, 0.8844456f, 0.8903556f, 0.8966500f, 0.9031420f, 0.9096669f, 0.9160907f, 0.9223052f, 0.9282329f, 0.9338197f, 0.9390326f, 0.9438551f, 0.9482862f, 0.9523346f, 0.9560160f, 0.9593534f, 0.9623678f, 0.9650888f, 0.9675397f, 0.9697442f, 0.9717260f, 0.9735045f, 0.9751089f, 0.9765582f, 0.9778503f, 0.9790246f, 0.9800923f, 0.9810641f,
    0.9231030f, 0.9227942f, 0.9218627f, 0.9202938f, 0.9181018f, 0.9152778f, 0.9117956f, 0.9077572f, 0.9031422f, 0.8981266f, 0.8927671f, 0.8873183f, 0.8819382f, 0.8769069f, 0.8724540f, 0.8687737f, 0.8661346f, 0.8646210f, 0.8642959f, 0.8651756f, 0.8672496f, 0.8704010f, 0.8744783f, 0.8793184f, 0.8847428f, 0.8905786f, 0.8966626f, 0.9028486f, 0.9090114f, 0.9150442f, 0.9208647f, 0.9264127f, 0.9316476f, 0.9365447f, 0.9410928f, 0.9452923f, 0.9491498f, 0.9526780f, 0.9558973f, 0.9588250f, 0.9614825f, 0.9638932f, 0.9660776f, 0.9680560f, 0.9698483f, 0.9714694f, 0.9729222f, 0.9742611f, 0.9754538f,
    0.9201727f, 0.9198886f, 0.9190319f, 0.9175902f, 0.9155863f, 0.9129887f, 0.9098121f, 0.9060911f, 0.9018540f, 0.8971771f, 0.8921729f, 0.8869409f, 0.8817160f, 0.8766182f, 0.8719243f, 0.8677676f, 0.8644157f, 0.8619624f, 0.8605210f, 0.8602045f, 0.8609833f, 0.8628100f, 0.8656044f, 0.8692607f, 0.8736682f, 0.8786602f, 0.8840863f, 0.8898007f, 0.8956689f, 0.9015743f, 0.9074174f, 0.9131173f, 0.9186116f, 0.9238551f, 0.9288139f, 0.9334674f, 0.9378074f, 0.9418341f, 0.9455516f, 0.9489743f, 0.9521126f, 0.9549873f, 0.9576143f, 0.9600107f, 0.9621953f, 0.9641847f, 0.9659904f, 0.9676437f, 0.9691827f,
    0.9168158f, 0.9165519f, 0.9157564f, 0.9144195f, 0.9125669f, 0.9101545f, 0.9072279f, 0.9037653f, 0.8998483f, 0.8954905f, 0.8907670f, 0.8858183f, 0.8807442f, 0.8757000f, 0.8708927f, 0.8664505f, 0.8626012f, 0.8594440f, 0.8571661f, 0.8558152f, 0.8554395f, 0.8560914f, 0.8577020f, 0.8602070f, 0.8635094f, 0.8675100f, 0.8720999f, 0.8771316f, 0.8824798f, 0.8880285f, 0.8936676f, 0.8993017f, 0.9048529f, 0.9102569f, 0.9154641f, 0.9204388f, 0.9251546f, 0.9295941f, 0.9337514f, 0.9376248f, 0.9412192f, 0.9445453f, 0.9476141f, 0.9504379f, 0.9530329f, 0.9554152f, 0.9576011f, 0.9596084f, 0.9614693f,
    0.9130047f, 0.9127572f, 0.9120117f, 0.9107643f, 0.9090300f, 0.9067695f, 0.9040443f, 0.9008213f, 0.8971382f, 0.8930602f, 0.8886170f, 0.8838943f, 0.8790001f, 0.8740651f, 0.8692320f, 0.8646365f, 0.8604466f, 0.8568127f, 0.8538486f, 0.8516833f, 0.8503756f, 0.8499633f, 0.8504807f, 0.8518782f, 0.8540964f, 0.8570786f, 0.8607339f, 0.8649476f, 0.8696064f, 0.8746000f, 0.8798267f, 0.8851879f, 0.8905946f, 0.8959719f, 0.9012640f, 0.9064160f, 0.9113848f, 0.9161366f, 0.9206508f, 0.9249150f, 0.9289231f, 0.9326723f, 0.9361705f, 0.9394204f, 0.9424337f, 0.9452246f, 0.9478019f, 0.9501791f, 0.9523309f,
    0.9087132f, 0.9084792f, 0.9077742f, 0.9065990f, 0.9049603f, 0.9028357f, 0.9002586f, 0.8972292f, 0.8937666f, 0.8899067f, 0.8856891f, 0.8811826f, 0.8764736f, 0.8716622f, 0.8668599f, 0.8621944f, 0.8577960f, 0.8538046f, 0.8503530f, 0.8475297f, 0.8454522f, 0.8441518f, 0.8436984f, 0.8440712f, 0.8452588f, 0.8472326f, 0.8499131f, 0.8532183f, 0.8570655f, 0.8613746f, 0.8660387f, 0.8709654f, 0.8760616f, 0.8812473f, 0.8864532f, 0.8916156f, 0.8966802f, 0.9016045f, 0.9063553f, 0.9109082f, 0.9152480f, 0.9193640f, 0.9232464f, 0.9268962f, 0.9303124f, 0.9335057f, 0.9364804f, 0.9392497f, 0.9418461f,
    0.9039177f, 0.9036944f, 0.9030221f, 0.9019046f, 0.9003425f, 0.8983279f, 0.8958709f, 0.8929928f, 0.8897063f, 0.8860382f, 0.8820176f, 0.8777020f, 0.8731592f, 0.8684684f, 0.8637252f, 0.8590389f, 0.8545223f, 0.8502921f, 0.8464622f, 0.8431405f, 0.8404342f, 0.8383967f, 0.8371010f, 0.8365621f, 0.8368051f, 0.8377924f, 0.8395100f, 0.8419008f, 0.8448887f, 0.8484018f, 0.8523716f, 0.8567118f, 0.8613321f, 0.8661555f, 0.8711065f, 0.8761163f, 0.8811265f, 0.8860874f, 0.8909521f, 0.8956883f, 0.9002652f, 0.9046617f, 0.9088651f, 0.9128604f, 0.9166435f, 0.9202102f, 0.9235672f, 0.9267132f, 0.9296823f,
    0.8985965f, 0.8983818f, 0.8977353f, 0.8966634f, 0.8951617f, 0.8932336f, 0.8908807f, 0.8881248f, 0.8849761f, 0.8814620f, 0.8776035f, 0.8734479f, 0.8690498f, 0.8644737f, 0.8598031f, 0.8551259f, 0.8505376f, 0.8461410f, 0.8420397f, 0.8383346f, 0.8351235f, 0.8324799f, 0.8304650f, 0.8291373f, 0.8285119f, 0.8286056f, 0.8293961f, 0.8308713f, 0.8329726f, 0.8356502f, 0.8388506f, 0.8424956f, 0.8465108f, 0.8508234f, 0.8553658f, 0.8600770f, 0.8648851f, 0.8697351f, 0.8745768f, 0.8793629f, 0.8840567f, 0.8886291f, 0.8930578f, 0.8973206f, 0.9014022f, 0.9052973f, 0.9089962f, 0.9125020f, 0.9157907f,
    0.8927310f, 0.8925229f, 0.8918965f, 0.8908597f, 0.8894046f, 0.8875428f, 0.8852704f, 0.8826121f, 0.8795725f, 0.8761782f, 0.8724495f, 0.8684271f, 0.8641514f, 0.8596794f, 0.8550812f, 0.8504257f, 0.8457960f, 0.8412840f, 0.8369826f, 0.8329809f, 0.8293655f, 0.8262174f, 0.8236094f, 0.8215915f, 0.8202025f, 0.8194759f, 0.8194099f, 0.8200036f, 0.8212266f, 0.8230562f, 0.8254340f, 0.8283104f, 0.8316354f, 0.8353372f, 0.8393503f, 0.8436126f, 0.8480676f, 0.8526564f, 0.8573241f, 0.8620187f, 0.8666994f, 0.8713281f, 0.8758701f, 0.8803008f, 0.8845998f, 0.8887506f, 0.8927387f, 0.8965536f, 0.9001849f,
    0.8863051f, 0.8861020f, 0.8854905f, 0.8844799f, 0.8830603f, 0.8812469f, 0.8790350f, 0.8764475f, 0.8734940f, 0.8701994f, 0.8665691f, 0.8626491f, 0.8584716f, 0.8540861f, 0.8495462f, 0.8449150f, 0.8402668f, 0.8356712f, 0.8312170f, 0.8269813f, 0.8230446f, 0.8194845f, 0.8163704f, 0.8137615f, 0.8117062f, 0.8102406f, 0.8093914f, 0.8091570f, 0.8095401f, 0.8105179f, 0.8120679f, 0.8141433f, 0.8167111f, 0.8197131f, 0.8230938f, 0.8268016f, 0.8307817f, 0.8349758f, 0.8393288f, 0.8437919f, 0.8483184f, 0.8528652f, 0.8574002f, 0.8618865f, 0.8662950f, 0.8706052f, 0.8747972f, 0.8788567f, 0.8827458f,
    0.8793061f, 0.8791065f, 0.8785056f, 0.8775134f, 0.8761219f, 0.8743404f, 0.8721741f, 0.8696402f, 0.8667504f, 0.8635201f, 0.8599693f, 0.8561226f, 0.8520192f, 0.8476989f, 0.8432040f, 0.8385938f, 0.8339275f, 0.8292682f, 0.8246923f, 0.8202657f, 0.8160663f, 0.8121644f, 0.8086259f, 0.8055118f, 0.8028749f, 0.8007580f, 0.7991909f, 0.7981950f, 0.7977825f, 0.7979435f, 0.7986690f, 0.7999351f, 0.8017120f, 0.8039581f, 0.8066384f, 0.8096997f, 0.8130917f, 0.8167729f, 0.8206897f, 0.8247917f, 0.8290342f, 0.8333715f, 0.8377659f, 0.8421797f, 0.8465795f, 0.8509415f, 0.8552342f, 0.8594387f, 0.8635456f,
    0.8717244f, 0.8715269f, 0.8709328f, 0.8699523f, 0.8685790f, 0.8668216f, 0.8646856f, 0.8621829f, 0.8593340f, 0.8561535f, 0.8526515f, 0.8488580f, 0.8448044f, 0.8405272f, 0.8360640f, 0.8314627f, 0.8267756f, 0.8220599f, 0.8173795f, 0.8127965f, 0.8083746f, 0.8041842f, 0.8002838f, 0.7967331f, 0.7935880f, 0.7908931f, 0.7886856f, 0.7869935f, 0.7858367f, 0.7852249f, 0.7851533f, 0.7856144f, 0.7865927f, 0.7880576f, 0.7899837f, 0.7923315f, 0.7950598f, 0.7981318f, 0.8015003f, 0.8051209f, 0.8089497f, 0.8129468f, 0.8170719f, 0.8212840f, 0.8255504f, 0.8298354f, 0.8341127f, 0.8383564f, 0.8425490f,
    0.8635539f, 0.8633575f, 0.8627667f, 0.8617920f, 0.8604283f, 0.8586836f, 0.8565636f, 0.8540774f, 0.8512557f, 0.8481008f, 0.8446289f, 0.8408661f, 0.8368410f, 0.8325856f, 0.8281338f, 0.8235284f, 0.8188146f, 0.8140429f, 0.8092665f, 0.8045434f, 0.7999320f, 0.7954909f, 0.7912770f, 0.7873498f, 0.7837568f, 0.7805497f, 0.7777672f, 0.7754422f, 0.7736010f, 0.7722609f, 0.7714300f, 0.7711116f, 0.7712979f, 0.7719728f, 0.7731203f, 0.7747100f, 0.7767169f, 0.7791020f, 0.7818344f, 0.7848739f, 0.7881805f, 0.7917145f, 0.7954414f, 0.7993230f, 0.8033207f, 0.8074040f, 0.8115381f, 0.8157023f, 0.8198654f,
    0.8547920f, 0.8545957f, 0.8540056f, 0.8530311f, 0.8516695f, 0.8499277f, 0.8478110f, 0.8453340f, 0.8425169f, 0.8393730f, 0.8359122f, 0.8321598f, 0.8281437f, 0.8238901f, 0.8194320f, 0.8148062f, 0.8100555f, 0.8052217f, 0.8003534f, 0.7955006f, 0.7907158f, 0.7860519f, 0.7815628f, 0.7772996f, 0.7733128f, 0.7696493f, 0.7663489f, 0.7634487f, 0.7609798f, 0.7589645f, 0.7574190f, 0.7563529f, 0.7557685f, 0.7556625f, 0.7560238f, 0.7568348f, 0.7580790f, 0.7597272f, 0.7617552f, 0.7641291f, 0.7668165f, 0.7697864f, 0.7730021f, 0.7764288f, 0.7800337f, 0.7837847f, 0.7876477f, 0.7915942f, 0.7956033f,
    0.8454398f, 0.8452427f, 0.8446509f, 0.8436720f, 0.8423057f, 0.8405580f, 0.8384389f, 0.8359540f, 0.8331305f, 0.8299833f, 0.8265140f, 0.8227529f, 0.8187264f, 0.8144563f, 0.8099753f, 0.8053148f, 0.8005139f, 0.7956093f, 0.7906457f, 0.7856674f, 0.7807201f, 0.7758529f, 0.7711136f, 0.7665500f, 0.7622088f, 0.7581350f, 0.7543686f, 0.7509461f, 0.7479021f, 0.7452621f, 0.7430470f, 0.7412734f, 0.7399507f, 0.7390828f, 0.7386666f, 0.7386951f, 0.7391564f, 0.7400352f, 0.7413088f, 0.7429574f, 0.7449515f, 0.7472651f, 0.7498700f, 0.7527349f, 0.7558291f, 0.7591234f, 0.7625869f, 0.7661904f, 0.7699123f,
    0.8355022f, 0.8353035f, 0.8347075f, 0.8337203f, 0.8323436f, 0.8305839f, 0.8284514f, 0.8259520f, 0.8231067f, 0.8199412f, 0.8164499f, 0.8126653f, 0.8086108f, 0.8043081f, 0.7997849f, 0.7950754f, 0.7902111f, 0.7852267f, 0.7801613f, 0.7750566f, 0.7699523f, 0.7648935f, 0.7599219f, 0.7550831f, 0.7504195f, 0.7459726f, 0.7417821f, 0.7378842f, 0.7343127f, 0.7310957f, 0.7282588f, 0.7258198f, 0.7237954f, 0.7221942f, 0.7210209f, 0.7202755f, 0.7199537f, 0.7200459f, 0.7205387f, 0.7214188f, 0.7226635f, 0.7242551f, 0.7261679f, 0.7283770f, 0.7308576f, 0.7335834f, 0.7365281f, 0.7396632f, 0.7429603f,
    0.8249874f, 0.8247865f, 0.8241843f, 0.8231854f, 0.8217933f, 0.8200168f, 0.8178601f, 0.8153369f, 0.8124607f, 0.8092637f, 0.8057393f, 0.8019155f, 0.7978173f, 0.7934657f, 0.7888880f, 0.7841128f, 0.7791725f, 0.7740977f, 0.7689239f, 0.7636883f, 0.7584287f, 0.7531832f, 0.7479924f, 0.7428966f, 0.7379341f, 0.7331452f, 0.7285661f, 0.7242323f, 0.7201759f, 0.7164267f, 0.7130114f, 0.7099512f, 0.7072652f, 0.7049659f, 0.7030643f, 0.7015656f, 0.7004706f, 0.6997771f, 0.6994795f, 0.6995669f, 0.7000275f, 0.7008476f, 0.7020074f, 0.7034892f, 0.7052718f, 0.7073335f, 0.7096528f, 0.7122037f, 0.7149673f,
    0.8139078f, 0.8137041f, 0.8130937f, 0.8120802f, 0.8106700f, 0.8088690f, 0.8066842f, 0.8041276f, 0.8012125f, 0.7979715f, 0.7944034f, 0.7905261f, 0.7863721f, 0.7819580f, 0.7773101f, 0.7724574f, 0.7674282f, 0.7622525f, 0.7569613f, 0.7515901f, 0.7461721f, 0.7407444f, 0.7353419f, 0.7300015f, 0.7247598f, 0.7196520f, 0.7147132f, 0.7099761f, 0.7054732f, 0.7012324f, 0.6972807f, 0.6936413f, 0.6903345f, 0.6873767f, 0.6847815f, 0.6825581f, 0.6807122f, 0.6792471f, 0.6781621f, 0.6774526f, 0.6771118f, 0.6771314f, 0.6774985f, 0.6782003f, 0.6792209f, 0.6805426f, 0.6821485f, 0.6840179f, 0.6861334f,
    0.8022790f, 0.8020720f, 0.8014519f, 0.8004222f, 0.7989893f, 0.7971584f, 0.7949398f, 0.7923431f, 0.7893814f, 0.7860883f, 0.7824647f, 0.7785235f, 0.7743034f, 0.7698138f, 0.7650849f, 0.7601414f, 0.7550115f, 0.7497242f, 0.7443079f, 0.7387949f, 0.7332168f, 0.7276056f, 0.7219967f, 0.7164211f, 0.7109132f, 0.7055059f, 0.7002310f, 0.6951200f, 0.6902026f, 0.6855066f, 0.6810575f, 0.6768793f, 0.6729932f, 0.6694170f, 0.6661660f, 0.6632535f, 0.6606871f, 0.6584751f, 0.6566198f, 0.6551223f, 0.6539804f, 0.6531900f, 0.6527441f, 0.6526343f, 0.6528491f, 0.6533767f, 0.6542044f, 0.6553157f, 0.6566931f,
    0.7901203f, 0.7899095f, 0.7892784f, 0.7882303f, 0.7867714f, 0.7849070f, 0.7826486f, 0.7800069f, 0.7769938f, 0.7736386f, 0.7699508f, 0.7659377f, 0.7616404f, 0.7570641f, 0.7522435f, 0.7472000f, 0.7419598f, 0.7365504f, 0.7310021f, 0.7253406f, 0.7195980f, 0.7138042f, 0.7079901f, 0.7021867f, 0.6964234f, 0.6907318f, 0.6851408f, 0.6796795f, 0.6743765f, 0.6692571f, 0.6643467f, 0.6596681f, 0.6552429f, 0.6510893f, 0.6472238f, 0.6436607f, 0.6404118f, 0.6374852f, 0.6348885f, 0.6326251f, 0.6306970f, 0.6291037f, 0.6278433f, 0.6269095f, 0.6262976f, 0.6259994f, 0.6260057f, 0.6263050f, 0.6268855f,
    0.7774541f, 0.7772392f, 0.7765960f, 0.7755276f, 0.7740401f, 0.7721403f, 0.7698384f, 0.7671453f, 0.7640749f, 0.7606512f, 0.7568921f, 0.7528018f, 0.7484157f, 0.7437455f, 0.7388251f, 0.7336720f, 0.7283121f, 0.7227741f, 0.7170846f, 0.7112702f, 0.7053601f, 0.6993823f, 0.6933653f, 0.6873375f, 0.6813282f, 0.6753641f, 0.6694741f, 0.6636839f, 0.6580199f, 0.6525065f, 0.6471677f, 0.6420254f, 0.6370996f, 0.6324096f, 0.6279716f, 0.6238004f, 0.6199089f, 0.6163077f, 0.6130061f, 0.6100094f, 0.6073230f, 0.6049492f, 0.6028891f, 0.6011395f, 0.5997012f, 0.5985667f, 0.5977320f, 0.5971897f, 0.5969331f,
    0.7643063f, 0.7640870f, 0.7634307f, 0.7623406f, 0.7608223f, 0.7588841f, 0.7565350f, 0.7537874f, 0.7506544f, 0.7471579f, 0.7433214f, 0.7391478f, 0.7346653f, 0.7298977f, 0.7248680f, 0.7195971f, 0.7141121f, 0.7084387f, 0.7026011f, 0.6966308f, 0.6905491f, 0.6843859f, 0.6781677f, 0.6719210f, 0.6656719f, 0.6594471f, 0.6532719f, 0.6471713f, 0.6411691f, 0.6352884f, 0.6295518f, 0.6239796f, 0.6185908f, 0.6134038f, 0.6084354f, 0.6037000f, 0.5992110f, 0.5949797f, 0.5910157f, 0.5873278f, 0.5839216f, 0.5808018f, 0.5779716f, 0.5754324f, 0.5731839f, 0.5712249f, 0.5695529f, 0.5681638f, 0.5670525f,
    0.7507055f, 0.7504815f, 0.7498115f, 0.7486984f, 0.7471476f, 0.7451686f, 0.7427703f, 0.7399649f, 0.7367664f, 0.7331928f, 0.7292743f, 0.7250119f, 0.7204283f, 0.7155575f, 0.7104118f, 0.7050168f, 0.6994007f, 0.6935881f, 0.6876030f, 0.6814688f, 0.6752164f, 0.6688678f, 0.6624497f, 0.6559875f, 0.6495056f, 0.6430306f, 0.6365831f, 0.6301887f, 0.6238694f, 0.6176461f, 0.6115399f, 0.6055699f, 0.5997537f, 0.5941101f, 0.5886526f, 0.5833978f, 0.5783571f, 0.5735435f, 0.5689659f, 0.5646342f, 0.5605549f, 0.5567345f, 0.5531769f, 0.5498859f, 0.5468629f, 0.5441090f, 0.5416238f, 0.5394055f, 0.5374492f,
    0.7366829f, 0.7364542f, 0.7357698f, 0.7346330f, 0.7330492f, 0.7310277f, 0.7285784f, 0.7257129f, 0.7224460f, 0.7187943f, 0.7147885f, 0.7104332f, 0.7057477f, 0.7007670f, 0.6955008f, 0.6899824f, 0.6842323f, 0.6782747f, 0.6721349f, 0.6658405f, 0.6594132f, 0.6528805f, 0.6462654f, 0.6395927f, 0.6328858f, 0.6261687f, 0.6194632f, 0.6127918f, 0.6061743f, 0.5996315f, 0.5931824f, 0.5868448f, 0.5806367f, 0.5745731f, 0.5686695f, 0.5629390f, 0.5573949f, 0.5520484f, 0.5469089f, 0.5419859f, 0.5372872f, 0.5328192f, 0.5285872f, 0.5245953f, 0.5208471f, 0.5173442f, 0.5140880f, 0.5110789f, 0.5083174f,
    0.7222723f, 0.7220387f, 0.7213397f, 0.7201785f, 0.7185613f, 0.7164965f, 0.7139939f, 0.7110670f, 0.7077302f, 0.7039998f, 0.6999041f, 0.6954538f, 0.6906650f, 0.6855701f, 0.6801844f, 0.6745379f, 0.6686504f, 0.6625477f, 0.6562550f, 0.6497964f, 0.6431968f, 0.6364813f, 0.6296730f, 0.6227953f, 0.6158710f, 0.6089221f, 0.6019711f, 0.5950369f, 0.5881405f, 0.5813008f, 0.5745349f, 0.5678604f, 0.5612930f, 0.5548472f, 0.5485378f, 0.5423771f, 0.5363770f, 0.5305485f, 0.5249009f, 0.5194430f, 0.5141827f, 0.5091264f, 0.5042800f, 0.4996481f, 0.4952344f, 0.4910418f, 0.4870726f, 0.4833279f, 0.4798063f,
    0.7075094f, 0.7072709f, 0.7065572f, 0.7053713f, 0.7037201f, 0.7016113f, 0.6990562f, 0.6960672f, 0.6926589f, 0.6888491f, 0.6846626f, 0.6801159f, 0.6752233f, 0.6700127f, 0.6645078f, 0.6587310f, 0.6527050f, 0.6464577f, 0.6400120f, 0.6333927f, 0.6266241f, 0.6197287f, 0.6127322f, 0.6056560f, 0.5985225f, 0.5913532f, 0.5841686f, 0.5769890f, 0.5698324f, 0.5627170f, 0.5556598f, 0.5486769f, 0.5417827f, 0.5349917f, 0.5283164f, 0.5217692f, 0.5153610f, 0.5091019f, 0.5030008f, 0.4970661f, 0.4913054f, 0.4857245f, 0.4803292f, 0.4751245f, 0.4701140f, 0.4653014f, 0.4606887f, 0.4562779f, 0.4520714f,
    0.6924318f, 0.6921884f, 0.6914599f, 0.6902493f, 0.6885640f, 0.6864114f, 0.6838034f, 0.6807522f, 0.6772734f, 0.6733838f, 0.6691069f, 0.6644637f, 0.6594673f, 0.6541419f, 0.6485181f, 0.6426112f, 0.6364514f, 0.6300612f, 0.6234647f, 0.6166867f, 0.6097512f, 0.6026829f, 0.5955035f, 0.5882369f, 0.5809038f, 0.5735255f, 0.5661216f, 0.5587120f, 0.5513140f, 0.5439454f, 0.5366218f, 0.5293586f, 0.5221698f, 0.5150689f, 0.5080675f, 0.5011771f, 0.4944081f, 0.4877696f, 0.4812702f, 0.4749175f, 0.4687181f, 0.4626786f, 0.4568035f, 0.4510978f, 0.4455652f, 0.4402088f, 0.4350314f, 0.4300346f, 0.4252180f,
    0.6770783f, 0.6768300f, 0.6760869f, 0.6748520f, 0.6731328f, 0.6709375f, 0.6682769f, 0.6651644f, 0.6616153f, 0.6576471f, 0.6532811f, 0.6485426f, 0.6434434f, 0.6380071f, 0.6322641f, 0.6262317f, 0.6199393f, 0.6134085f, 0.6066649f, 0.5997335f, 0.5926381f, 0.5854022f, 0.5780486f, 0.5706006f, 0.5630786f, 0.5555037f, 0.5478955f, 0.5402725f, 0.5326528f, 0.5250524f, 0.5174875f, 0.5099723f, 0.5025205f, 0.4951443f, 0.4878554f, 0.4806645f, 0.4735810f, 0.4666136f, 0.4597703f, 0.4530580f, 0.4464833f, 0.4400511f, 0.4337668f, 0.4276345f, 0.4216573f, 0.4158383f, 0.4101800f, 0.4046841f, 0.3993533f,
    0.6614889f, 0.6612359f, 0.6604785f, 0.6592200f, 0.6574676f, 0.6552300f, 0.6525180f, 0.6493454f, 0.6457277f, 0.6416825f, 0.6372307f, 0.6323979f, 0.6271986f, 0.6216548f, 0.6157950f, 0.6096418f, 0.6032199f, 0.5965534f, 0.5896692f, 0.5825905f, 0.5753416f, 0.5679463f, 0.5604285f, 0.5528089f, 0.5451104f, 0.5373526f, 0.5295554f, 0.5217370f, 0.5139149f, 0.5061057f, 0.4983244f, 0.4905851f, 0.4829010f, 0.4752841f, 0.4677455f, 0.4602955f, 0.4529428f, 0.4456960f, 0.4385620f, 0.4315478f, 0.4246592f, 0.4179010f, 0.4112776f, 0.4047928f, 0.3984498f, 0.3922510f, 0.3861987f, 0.3802943f, 0.3745384f,
    0.6457043f, 0.6454466f, 0.6446753f, 0.6433940f, 0.6416095f, 0.6393304f, 0.6365689f, 0.6333380f, 0.6296539f, 0.6255343f, 0.6209998f, 0.6160758f, 0.6107798f, 0.6051324f, 0.5991603f, 0.5928911f, 0.5863449f, 0.5795508f, 0.5725320f, 0.5653134f, 0.5579198f, 0.5503751f, 0.5427022f, 0.5349239f, 0.5270617f, 0.5191360f, 0.5111662f, 0.5031707f, 0.4951670f, 0.4871714f, 0.4791986f, 0.4712631f, 0.4633776f, 0.4555538f, 0.4478024f, 0.4401335f, 0.4325559f, 0.4250771f, 0.4177046f, 0.4104442f, 0.4033017f, 0.3962815f, 0.3893878f, 0.3826240f, 0.3759929f, 0.3694965f, 0.3631369f, 0.3569154f, 0.3508323f,
    0.6297652f, 0.6295031f, 0.6287184f, 0.6274151f, 0.6255998f, 0.6232815f, 0.6204722f, 0.6171855f, 0.6134374f, 0.6092466f, 0.6046333f, 0.5996218f, 0.5942335f, 0.5884874f, 0.5824097f, 0.5760295f, 0.5693668f, 0.5624514f, 0.5553057f, 0.5479560f, 0.5404276f, 0.5327441f, 0.5249291f, 0.5170050f, 0.5089933f, 0.5009155f, 0.4927905f, 0.4846373f, 0.4764730f, 0.4683139f, 0.4601753f, 0.4520712f, 0.4440145f, 0.4360167f, 0.4280887f, 0.4202402f, 0.4124798f, 0.4048153f, 0.3972536f, 0.3898009f, 0.3824620f, 0.3752418f, 0.3681437f, 0.3611716f, 0.3543273f, 0.3476132f, 0.3410309f, 0.3345811f, 0.3282661f,
    0.6137126f, 0.6134462f, 0.6126489f, 0.6113247f, 0.6094799f, 0.6071243f, 0.6042695f, 0.6009298f, 0.5971215f, 0.5928631f, 0.5881752f, 0.5830813f, 0.5776057f, 0.5717669f, 0.5655905f, 0.5591056f, 0.5523350f, 0.5453058f, 0.5380428f, 0.5305732f, 0.5229208f, 0.5151103f, 0.5071656f, 0.4991094f, 0.4909644f, 0.4827509f, 0.4744890f, 0.4661976f, 0.4578942f, 0.4495953f, 0.4413163f, 0.4330710f, 0.4248729f, 0.4167336f, 0.4086638f, 0.4006736f, 0.3927714f, 0.3849652f, 0.3772618f, 0.3696671f, 0.3621867f, 0.3548249f, 0.3475852f, 0.3404711f, 0.3334851f, 0.3266288f, 0.3199040f, 0.3133119f, 0.3068537f,
};

float CompensatedGgxSingleScatterAlbedo(float roughness, float NoV) {
    float x = std::clamp(roughness, 0.0f, 1.0f) * 48.0f;
    float y = std::sqrt(std::clamp(NoV, 0.0f, 1.0f)) * 48.0f;
    unsigned i = std::min(static_cast<unsigned>(x), 47u);
    unsigned j = std::min(static_cast<unsigned>(y), 47u);
    float tx = x - float(i);
    float ty = y - float(j);
    unsigned k = i * 49u + j;
    float a = kGgxSingleScatterAlbedo[k] * (1.0f - ty) + kGgxSingleScatterAlbedo[k + 1u] * ty;
    float b = kGgxSingleScatterAlbedo[k + 49u] * (1.0f - ty) + kGgxSingleScatterAlbedo[k + 50u] * ty;
    return a * (1.0f - tx) + b * tx;
}

// Mirrors specular_energy_compensation.
simd::float3 SpecularEnergyCompensation(const simd::float3& f0,
                                        float roughness,
                                        float NoV) {
    float Ess = std::clamp(CompensatedGgxSingleScatterAlbedo(roughness, NoV), 1.0e-4f, 1.0f);
    float Ems = 1.0f - Ess;
    simd::float3 one = simd_make_float3(1.0f, 1.0f, 1.0f);
    simd::float3 Favg = f0 + (one - f0) * (1.0f / 21.0f);
    return simd::clamp(one / simd::max(one - Ems * Favg,
                                     simd_make_float3(1.0e-4f, 1.0e-4f, 1.0e-4f)),
                       one, simd_make_float3(4.0f, 4.0f, 4.0f));
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
    // Mirrors fresnel_conductor (closures_eval.metal): rs and rp are reflectances.
    simd::float3 rp = rs * (term3 - term4) / (term3 + term4);

    return simd::clamp(0.5f * (rs + rp),
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

// The same D, taking sin^2(theta_h) from the half vector. In float32 the
// cosine form cannot resolve lobes near the PBR alpha floor (1e-4):
// cos(theta_h) rounds to 1 below theta_h = 2.4e-4 rad, and a2 - 1 rounds to
// -1 below alpha = 1.7e-4, so D steps between a few values or is inf there.
float GgxDistribution(float alpha, const simd::float3& normal, const simd::float3& wh) {
    float cosThetaH = simd::dot(normal, wh);
    float sin2ThetaH = simd::length_squared(simd::cross(normal, wh));
    float a2 = alpha * alpha;
    float denom = sin2ThetaH + a2 * cosThetaH * cosThetaH;
    return (denom > 0.0f) ? a2 / (kPi * denom * denom) : 0.0f;
}

float GgxPdf(float alpha, const simd::float3& normal, const simd::float3& wo, const simd::float3& wi,
             bool halfVectorD = false) {
    simd::float3 wh = simd::normalize(wo + wi);
    float cosThetaH = simd::dot(normal, wh);
    float cosThetaO = simd::dot(normal, wo);
    float dotWoWh = simd::dot(wo, wh);
    if (cosThetaO <= 0.0f || cosThetaH <= 0.0f || dotWoWh <= 0.0f) {
        return 0.0f;
    }
    float denom = 4.0f * std::max(dotWoWh, 1.0e-6f);
    float d = halfVectorD ? GgxDistribution(alpha, normal, wh)
                          : GgxDistribution(alpha, std::max(cosThetaH, 0.0f));
    float g1 = GgxG1(alpha, cosThetaO);
    return d * g1 * std::max(cosThetaH, 0.0f) / denom;
}

float GgxVndfPdf(float alpha, const simd::float3& normal, const simd::float3& wo, const simd::float3& wh,
                 bool halfVectorD = false) {
    float cosThetaO = simd::dot(normal, wo);
    float cosThetaH = simd::dot(normal, wh);
    float woDotWh = simd::dot(wo, wh);
    if (cosThetaO <= 0.0f || cosThetaH <= 0.0f || woDotWh <= 0.0f) {
        return 0.0f;
    }
    // Density of the visible-normal half vector: D * G1(wo) * (wo.wh) / cos(theta_o).
    float d = halfVectorD ? GgxDistribution(alpha, normal, wh) : GgxDistribution(alpha, cosThetaH);
    float g1 = GgxG1(alpha, cosThetaO);
    return d * g1 * woDotWh / cosThetaO;
}

simd::float3 ToLocal(const simd::float3& v, const Onb& onb) {
    return simd_make_float3(simd::dot(v, onb.tangent),
                            simd::dot(v, onb.bitangent),
                            simd::dot(v, onb.normal));
}

simd::float3 ToWorld(const simd::float3& v, const Onb& onb) {
    return v.x * onb.tangent + v.y * onb.bitangent + v.z * onb.normal;
}

// A uniform direction in a finite sun's cone, as Metal's sample_directional_cone.
// A delta sun keeps its direction and does not consume random numbers.
simd::float3 SampleDirectionalCone(const simd::float3& axis, float cosDelta, Rng& rng) {
    if (!(cosDelta > 0.0f)) {
        return axis;
    }
    const float delta = rng.nextFloat() * cosDelta;
    const float cosine = 1.0f - delta;
    const float sine = std::sqrt(std::max(delta * (2.0f - delta), 0.0f));
    const float phi = 2.0f * kPi * rng.nextFloat();
    return simd::normalize(ToWorld(simd_make_float3(sine * std::cos(phi), sine * std::sin(phi), cosine),
                                   BuildOnb(axis)));
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

// The global homogeneous medium of --volumeTransportMode=homogeneous, as Metal's volume_transport.metal builds it
// (volume_make_global_medium): coefficients clamped at 0, and the medium is on only when its largest sigma_t is finite
// and above 0 (volume_transport_enabled). It covers path segments that end at a hit; escaped paths are not attenuated.
struct EmbreeVolumeMedium {
    bool enabled = false;
    simd::float3 sigmaA = simd_make_float3(0.0f, 0.0f, 0.0f);
    simd::float3 sigmaS = simd_make_float3(0.0f, 0.0f, 0.0f);
    simd::float3 sigmaT = simd_make_float3(0.0f, 0.0f, 0.0f);
    simd::float3 emission = simd_make_float3(0.0f, 0.0f, 0.0f);
    bool scatters = false;                 // any sigma_s > 0
    bool henyeyGreenstein = false;         // --volumePhase=hg
    float anisotropy = 0.0f;               // g, clamped to [-0.98, 0.98]
    uint32_t maxScatteringEvents = 1u;     // --volumeMaxScatteringEvents, at least 1
    bool nee = true;                       // --volumeNee: rect-light NEE at scatter points
};

EmbreeVolumeMedium MakeEmbreeVolumeMedium(const PathTracer::RenderSettings& settings) {
    EmbreeVolumeMedium medium;
    const auto nonNegative = [](const simd::float3& v) {
        return simd_make_float3(std::max(v.x, 0.0f), std::max(v.y, 0.0f), std::max(v.z, 0.0f));
    };
    medium.sigmaA = nonNegative(settings.volumeSigmaA);
    medium.sigmaS = nonNegative(settings.volumeSigmaS);
    medium.sigmaT = medium.sigmaA + medium.sigmaS;
    medium.emission = nonNegative(settings.volumeEmission);
    const float majorant = std::max(std::max(medium.sigmaT.x, medium.sigmaT.y), medium.sigmaT.z);
    medium.enabled = settings.volumeTransportMode == PathTracer::RenderSettings::VolumeTransportMode::Homogeneous &&
                     std::isfinite(majorant) && majorant > 0.0f;
    medium.scatters = medium.sigmaS.x > 0.0f || medium.sigmaS.y > 0.0f || medium.sigmaS.z > 0.0f;
    medium.henyeyGreenstein =
        settings.volumePhaseFunction == PathTracer::RenderSettings::VolumePhaseFunction::HenyeyGreenstein;
    medium.anisotropy = std::clamp(settings.volumeAnisotropy, -0.98f, 0.98f);
    medium.maxScatteringEvents = std::max<uint32_t>(settings.volumeMaxScatteringEvents, 1u);
    medium.nee = settings.volumeNeeEnabled;
    return medium;
}

simd::float3 VolumeSafeNormalize(const simd::float3& v) {
    const float len = simd::length(v);
    return len > 0.0f && std::isfinite(len) ? v / len : simd_make_float3(0.0f, 0.0f, 0.0f);
}

// Henyey-Greenstein in pbrt's convention (Metal's volume_phase_hg): cosTheta = dot(wo, wi) with wo pointing back along
// the incoming ray, so g > 0 scatters forward.
float VolumePhaseHg(float cosTheta, float g) {
    const float gg = std::clamp(g, -0.98f, 0.98f);
    const float denom = std::max(1.0f + gg * gg + 2.0f * gg * cosTheta, 1.0e-4f);
    return (1.0f - gg * gg) / (4.0f * kPi * denom * std::sqrt(denom));
}

// Metal's volume_phase_eval.
float VolumePhaseEval(const EmbreeVolumeMedium& medium, const simd::float3& wo, const simd::float3& wi) {
    const float cosTheta = std::clamp(simd::dot(VolumeSafeNormalize(wo), VolumeSafeNormalize(wi)), -1.0f, 1.0f);
    return medium.henyeyGreenstein ? VolumePhaseHg(cosTheta, medium.anisotropy) : 1.0f / (4.0f * kPi);
}

// Metal's volume_phase_sample: the HG density inverted in a frame built on wo (backward), so the sign is negated.
simd::float3 VolumePhaseSample(const EmbreeVolumeMedium& medium, const simd::float3& wo, Rng& rng, float& outPdf) {
    float cosTheta = 1.0f - 2.0f * rng.nextFloat();
    if (medium.henyeyGreenstein) {
        const float g = medium.anisotropy;
        if (std::fabs(g) > 1.0e-3f) {
            const float u = rng.nextFloat();
            const float sqrTerm = (1.0f - g * g) / std::max(1.0f - g + 2.0f * g * u, 1.0e-4f);
            cosTheta = std::clamp(-(1.0f + g * g - sqrTerm * sqrTerm) / (2.0f * g), -1.0f, 1.0f);
        }
    }
    const float phi = 2.0f * kPi * rng.nextFloat();
    const float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
    const simd::float3 n = VolumeSafeNormalize(wo);
    const simd::float3 up = std::fabs(n.z) < 0.999f ? simd_make_float3(0.0f, 0.0f, 1.0f) : simd_make_float3(0.0f, 1.0f, 0.0f);
    const simd::float3 tangent = VolumeSafeNormalize(simd::cross(up, n));
    const simd::float3 bitangent = simd::cross(n, tangent);
    const simd::float3 wi = VolumeSafeNormalize(std::cos(phi) * sinTheta * tangent + std::sin(phi) * sinTheta * bitangent +
                                                cosTheta * n);
    outPdf = VolumePhaseEval(medium, wo, wi);
    return wi;
}

// Free flight along a segment of length s that ends at a hit, as Metal's volume_sample_free_flight (pbrt-v3's
// HomogeneousMedium::Sample): one colour channel chosen uniformly, the distance drawn with that channel's sigma_t, and
// the weight divided by the mixture of the three channels' densities, so the transmittance counts once:
//   scatter at t < s:  sigma_s T(t) / mean(sigma_t T(t)),    pass (t >= s):  T(s) / mean(T(s)).
struct VolumeFreeFlight {
    bool scattered = false;
    float distance = 0.0f;
    simd::float3 weight = simd_make_float3(0.0f, 0.0f, 0.0f);
};

VolumeFreeFlight VolumeSampleFreeFlight(const EmbreeVolumeMedium& medium, float segment, Rng& rng) {
    const uint32_t channel = std::min(static_cast<uint32_t>(rng.nextFloat() * 3.0f), 2u);
    const float sigmaC = medium.sigmaT[channel];
    const float u = rng.nextFloat();
    const float t = sigmaC > 0.0f ? -std::log(std::max(1.0f - u, 1.0e-30f)) / sigmaC
                                  : std::numeric_limits<float>::infinity();
    VolumeFreeFlight flight;
    flight.scattered = t < segment;
    flight.distance = flight.scattered ? t : segment;
    const simd::float3 tr = Exp3(-medium.sigmaT * std::max(flight.distance, 0.0f));
    const simd::float3 numerator = flight.scattered ? medium.sigmaS * tr : tr;
    const simd::float3 density = flight.scattered ? medium.sigmaT * tr : tr;
    const float pdf = (density.x + density.y + density.z) * (1.0f / 3.0f);
    flight.weight = pdf > 0.0f ? numerator / pdf : simd_make_float3(0.0f, 0.0f, 0.0f);
    return flight;
}

// Metal's volume_nee_mis_weight: the balance-heuristic weight of volume NEE to a rect light against the
// phase-sampled direction to the same light, clamped like the surface rect-light weights.
float VolumeNeeMisWeight(float lightPdf, float phasePdf) {
    const float denom = lightPdf + phasePdf;
    return denom > 0.0f ? std::clamp(lightPdf / denom, kMisWeightClampMin, kMisWeightClampMax) : 1.0f;
}

// exp(-sigma_t d), Metal's volume_transmittance.
simd::float3 VolumeTransmittance(const EmbreeVolumeMedium& medium, float distance) {
    return Exp3(-medium.sigmaT * std::max(distance, 0.0f));
}

// Metal's volume_surface_nee_transmittance: surface NEE to a light sample with a position is attenuated over the
// distance to the sampled point; suns and the environment are not, because escaped paths are not.
simd::float3 VolumeSurfaceNeeTransmittance(const EmbreeVolumeMedium& medium, float distance, bool directional) {
    if (directional || !medium.enabled) {
        return simd_make_float3(1.0f, 1.0f, 1.0f);
    }
    return VolumeTransmittance(medium, distance);
}

// Metal's volume_emission_integral: the medium's emission integrated along a segment, emission (1 - T) / sigma_t.
simd::float3 VolumeEmissionIntegral(const EmbreeVolumeMedium& medium, float distance) {
    const simd::float3 tr = VolumeTransmittance(medium, distance);
    const simd::float3 denom = simd_make_float3(std::max(medium.sigmaT.x, 1.0e-6f),
                                                std::max(medium.sigmaT.y, 1.0e-6f),
                                                std::max(medium.sigmaT.z, 1.0e-6f));
    return medium.emission * ((simd_make_float3(1.0f, 1.0f, 1.0f) - tr) / denom);
}

// Hero-wavelength spectral transport (--spectralMode=hero_wavelength), as Metal's spectral_transport.metal. Channel j
// of every colour inside the path (throughput, BSDF values, emission, light radiance) is the value at wavelength
//   lambda_j = min + (max - min) * frac(u + j / C),  j < C;  lambda_0 is the hero wavelength.
// RGB enters the path once per quantity (SpectralRgbToChannels); the sensor and white balance apply once when the
// path ends (SpectralChannelsToRgb). Inactive channels (C = 1) hold the hero wavelength and zero throughput.
struct EmbreeSpectralConfig {
    bool enabled = false;
    uint32_t channelCount = 3u;
    float minNm = 380.0f;
    float maxNm = 720.0f;
    simd::float3 whiteBalance = simd_make_float3(1.0f, 1.0f, 1.0f);
    float dispersionStrength = 0.0f;
};

struct EmbreeSpectralState {
    bool enabled = false;
    bool collapsed = false;     // throughput reduced to the hero channel at a dispersive vertex
    uint32_t channelCount = 0u;
    float heroNm = 0.0f;
    simd::float3 wavelengthsNm = simd_make_float3(0.0f, 0.0f, 0.0f);
    simd::float3 sensor0 = simd_make_float3(1.0f, 1.0f, 1.0f);
    simd::float3 sensor1 = simd_make_float3(1.0f, 1.0f, 1.0f);
    simd::float3 sensor2 = simd_make_float3(1.0f, 1.0f, 1.0f);
};

EmbreeSpectralConfig MakeEmbreeSpectralConfig(const PathTracer::RenderSettings& settings) {
    EmbreeSpectralConfig config;
    config.enabled =
        settings.spectralRenderingMode == PathTracer::RenderSettings::SpectralRenderingMode::HeroWavelength;
    config.channelCount = settings.spectralWavelengthsPerPath == 1u ? 1u : 3u;
    const SpectralSensor::WavelengthRange range =
        SpectralSensor::ClampRange(settings.spectralMinWavelengthNm, settings.spectralMaxWavelengthNm);
    config.minNm = range.minNm;
    config.maxNm = range.maxNm;
    config.whiteBalance = config.enabled ? SpectralSensor::WhiteBalanceCached(range) : simd_make_float3(1.0f, 1.0f, 1.0f);
    config.dispersionStrength = std::max(settings.spectralDispersionStrength, 0.0f);
    return config;
}

// s(lambda) in float, as Metal's spectral_sensor_rgb (identical to SpectralSensor::Response).
simd::float3 SpectralSensorRgb(float wavelengthNm) {
    const auto gaussian = [](float nm, float center, float width) {
        const float x = (nm - center) / std::max(width, 1.0f);
        return std::exp(-0.5f * x * x);
    };
    const simd::float3 response = simd_make_float3(gaussian(wavelengthNm, 610.0f, 48.0f),
                                                   gaussian(wavelengthNm, 545.0f, 42.0f),
                                                   gaussian(wavelengthNm, 455.0f, 36.0f));
    const float normalizer = std::max(std::max(response.x, response.y), response.z);
    if (normalizer <= 1.0e-5f) {
        return simd_make_float3(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f);
    }
    return response / normalizer;
}

float SpectralValueToSample(const simd::float3& values, float wavelengthNm, const simd::float3& sensorRgb,
                            bool explicitTriplet) {
    const simd::float3 clamped = simd::max(values, simd_make_float3(0.0f, 0.0f, 0.0f));
    if (explicitTriplet) {
        // A 450/550/650 nm triplet, interpolated linearly (spectral_triplet_to_sample).
        if (wavelengthNm <= 550.0f) {
            const float t = std::clamp((wavelengthNm - 450.0f) * (1.0f / 100.0f), 0.0f, 1.0f);
            return clamped.x + (clamped.y - clamped.x) * t;
        }
        const float t = std::clamp((wavelengthNm - 550.0f) * (1.0f / 100.0f), 0.0f, 1.0f);
        return clamped.y + (clamped.z - clamped.y) * t;
    }
    // RGB projected on the sensor (spectral_rgb_to_sample).
    const float denom = std::max(sensorRgb.x + sensorRgb.y + sensorRgb.z, 1.0e-5f);
    return std::max(simd::dot(clamped, sensorRgb) / denom, 0.0f);
}

simd::float3 SpectralRgbToChannels(const simd::float3& values, const EmbreeSpectralState& spectral, bool explicitTriplet) {
    return simd_make_float3(SpectralValueToSample(values, spectral.wavelengthsNm.x, spectral.sensor0, explicitTriplet),
                            SpectralValueToSample(values, spectral.wavelengthsNm.y, spectral.sensor1, explicitTriplet),
                            SpectralValueToSample(values, spectral.wavelengthsNm.z, spectral.sensor2, explicitTriplet));
}

// Light radiance entering the path (background, environment, light samples): RGB policy.
simd::float3 SpectralRadianceToChannels(const simd::float3& rgb, const EmbreeSpectralState& spectral) {
    return spectral.enabled ? SpectralRgbToChannels(rgb, spectral, false) : rgb;
}

simd::float3 SpectralChannelMask(const EmbreeSpectralState& spectral) {
    return simd_make_float3(1.0f, spectral.channelCount > 1u ? 1.0f : 0.0f, spectral.channelCount > 2u ? 1.0f : 0.0f);
}

// Per-channel radiance -> RGB: W * sum_j L_j s(lambda_j) / C over the active channels (spectral_channels_to_rgb).
simd::float3 SpectralChannelsToRgb(const simd::float3& channels, const EmbreeSpectralState& spectral,
                                   const EmbreeSpectralConfig& config) {
    const simd::float3 mask = SpectralChannelMask(spectral);
    const simd::float3 sensorSum = channels.x * spectral.sensor0 + (channels.y * mask.y) * spectral.sensor1 +
                                   (channels.z * mask.z) * spectral.sensor2;
    return config.whiteBalance * sensorSum / static_cast<float>(std::max(spectral.channelCount, 1u));
}

// One hero wavelength per path, uniform over the range, and the other C - 1 spread evenly from it.
EmbreeSpectralState MakeEmbreeSpectralState(const EmbreeSpectralConfig& config, float u) {
    EmbreeSpectralState state;
    if (!config.enabled) {
        return state;
    }
    state.enabled = true;
    state.channelCount = config.channelCount;
    const float span = config.maxNm - config.minNm;
    const float uu = std::clamp(u, 0.0f, 1.0f);
    state.heroNm = config.minNm + span * uu;
    const float invChannels = 1.0f / static_cast<float>(std::max(state.channelCount, 1u));
    const auto frac = [](float v) { return v - std::floor(v); };
    state.wavelengthsNm = simd_make_float3(state.heroNm, state.heroNm, state.heroNm);
    if (state.channelCount > 1u) {
        state.wavelengthsNm.y = config.minNm + span * frac(uu + invChannels);
    }
    if (state.channelCount > 2u) {
        state.wavelengthsNm.z = config.minNm + span * frac(uu + 2.0f * invChannels);
    }
    state.sensor0 = SpectralSensorRgb(state.wavelengthsNm.x);
    state.sensor1 = SpectralSensorRgb(state.wavelengthsNm.y);
    state.sensor2 = SpectralSensorRgb(state.wavelengthsNm.z);
    return state;
}

// Cauchy-style IOR offset against 550 nm (spectral_cauchy_ior_offset).
float SpectralCauchyIorOffset(float wavelengthNm, float strength) {
    const float lambdaUm = std::clamp(wavelengthNm * 0.001f, 0.3f, 0.83f);
    const float greenUm = 0.55f;
    return strength * ((1.0f / (lambdaUm * lambdaUm)) - (1.0f / (greenUm * greenUm)));
}

// Metal's transmissive primary closure (material_closure_metadata.metal): a dielectric, or a PBR material whose
// transmission weight transmission x (1 - metallic) exceeds 0.05; spectral_apply_material also counts any PBR
// transmission factor above 0.
bool SpectralMaterialTransmissive(const PathTracerShaderTypes::MaterialData& material) {
    const uint32_t type = static_cast<uint32_t>(material.typeEta.x);
    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Dielectric)) {
        return true;
    }
    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
        const float metallic = std::clamp(material.pbrParams.x, 0.0f, 1.0f);
        if (std::clamp(material.pbrExtras.z, 0.0f, 1.0f) * (1.0f - metallic) > 0.05f) {
            return true;
        }
    }
    return material.pbrExtras.z > 0.0f;
}

bool SpectralVertexIsDispersive(const PathTracerShaderTypes::MaterialData& material, const EmbreeSpectralConfig& config) {
    return SpectralMaterialTransmissive(material) && std::max(material.spectralParams.x, config.dispersionStrength) > 0.0f;
}

// Metal's spectral_apply_material: base colour, emission, glass absorption and conductor eta/k become per-channel
// samples (an explicit 450/550/650 triplet when the material's spectral policy says so), and a transmissive
// material's IOR is dispersed at the hero wavelength.
void SpectralApplyMaterial(PathTracerShaderTypes::MaterialData& material, const EmbreeSpectralConfig& config,
                           const EmbreeSpectralState& spectral) {
    if (!spectral.enabled) {
        return;
    }
    const bool explicitTriplet = static_cast<uint32_t>(std::max(material.spectralParams.z, 0.0f) + 0.5f) == 1u;
    const auto xyz = [](const simd::float4& v) { return simd_make_float3(v.x, v.y, v.z); };
    const simd::float3 base = simd::clamp(SpectralRgbToChannels(xyz(material.baseColorRoughness), spectral, explicitTriplet),
                                          simd_make_float3(0.0f, 0.0f, 0.0f), simd_make_float3(1.0f, 1.0f, 1.0f));
    material.baseColorRoughness = simd_make_float4(base.x, base.y, base.z, material.baseColorRoughness.w);
    if (material.emission.x > 0.0f || material.emission.y > 0.0f || material.emission.z > 0.0f) {
        const simd::float3 e = SpectralRgbToChannels(xyz(material.emission), spectral, explicitTriplet);
        material.emission = simd_make_float4(e.x, e.y, e.z, material.emission.w);
    }
    if (material.dielectricSigmaA.x > 0.0f || material.dielectricSigmaA.y > 0.0f || material.dielectricSigmaA.z > 0.0f) {
        const float absorptionScale = std::max(material.spectralParams.y, 0.0f);
        const simd::float3 a = SpectralRgbToChannels(xyz(material.dielectricSigmaA) * absorptionScale, spectral, explicitTriplet);
        material.dielectricSigmaA = simd_make_float4(a.x, a.y, a.z, material.dielectricSigmaA.w);
    }
    if (material.conductorEta.w > 0.0f) {
        const simd::float3 zero = simd_make_float3(0.0f, 0.0f, 0.0f);
        const simd::float3 eta = simd::max(SpectralRgbToChannels(xyz(material.conductorEta), spectral, explicitTriplet), zero);
        const simd::float3 k = simd::max(SpectralRgbToChannels(xyz(material.conductorK), spectral, explicitTriplet), zero);
        material.conductorEta = simd_make_float4(eta.x, eta.y, eta.z, material.conductorEta.w);
        material.conductorK = simd_make_float4(k.x, k.y, k.z, material.conductorK.w);
    }
    if (SpectralMaterialTransmissive(material)) {
        const float dispersionScale = std::max(material.spectralParams.x, config.dispersionStrength);
        material.typeEta.y = std::clamp(material.typeEta.y + SpectralCauchyIorOffset(spectral.heroNm, dispersionScale),
                                        1.0f, 3.0f);
    }
}

// At a dispersive vertex the sampled direction holds for the hero wavelength only: keep the hero channel scaled by C
// (the 1/C averaging at the path's end) and drop the others (spectral_collapse_at_dispersive_vertex).
bool SpectralCollapseAtDispersiveVertex(EmbreeSpectralState& spectral, const PathTracerShaderTypes::MaterialData& material,
                                        const EmbreeSpectralConfig& config, simd::float3& throughput) {
    if (!spectral.enabled || spectral.collapsed || !SpectralVertexIsDispersive(material, config)) {
        return false;
    }
    throughput = simd_make_float3(throughput.x * static_cast<float>(spectral.channelCount), 0.0f, 0.0f);
    spectral.collapsed = true;
    return true;
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
                                                        float surfaceFootprintWorld,
                                                        bool normalVarianceRoughness = true,
                                                        float textureLodBias = 0.0f,
                                                        bool disableOrmTexture = false,
                                                        float* debugMetallic = nullptr,
                                                        float* debugRoughness = nullptr,
                                                        float* debugAO = nullptr);
simd::float3 ApplyNormalMapForHit(const EmbreeSceneData& sceneData,
                                  const PathTracerShaderTypes::MaterialData& material,
                                  const HitInfo& hit,
                                  const simd::float3& shadingNormal,
                                  const simd::float3& incidentDir,
                                  float debugNormalStrengthScale,
                                  float surfaceFootprintWorld,
                                  float textureLodBias = 0.0f,
                                  bool flipNormalGreen = false);
float ApplyNormalVarianceToRoughness(const EmbreeSceneData& sceneData,
                                     const PathTracerShaderTypes::MaterialData& material,
                                     const HitInfo& hit,
                                     float debugNormalStrengthScale,
                                     float surfaceFootprintWorld,
                                     float textureLodBias = 0.0f);
simd::float3 PlasticDiffuseTransmission(const PathTracerShaderTypes::MaterialData& material,
                                        float cosThetaI,
                                        float cosThetaO);

float MaterialRoughness(const PathTracerShaderTypes::MaterialData& material) {
    float roughness = std::clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
    return std::max(roughness, 1.0e-3f);
}

float DielectricF0(float ior) {
    float eta = std::max(ior, 1.0f);
    float r = (eta - 1.0f) / (eta + 1.0f);
    return r * r;
}

// Matches Metal pbr_thin_walled_transmission: glTF transmission without a volume
// is a thin sheet (no refraction, outside IOR on both faces). Tagged reference
// materials keep their refracting convention.
bool ThinWalledPbrTransmission(const PathTracerShaderTypes::MaterialData& material) {
    return static_cast<uint32_t>(material.typeEta.x) == 7u &&
        (material.materialFlags & PathTracerShaderTypes::kMaterialFlagThinWalledTransmission) != 0u &&
        (material.materialFlags & PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular) == 0u;
}

// Matches Metal pbr_thin_walled_transmission_nee_supported: light sampling may
// connect through the rough (non-delta) thin-walled BTDF.
bool ThinWalledPbrTransmissionNee(const PathTracerShaderTypes::MaterialData& material) {
    return ThinWalledPbrTransmission(material) && material.pbrExtras.z > 0.0f &&
        material.pbrParams.x < 1.0f && material.baseColorRoughness.w > 1.0e-3f;
}

simd::float3 DielectricSigmaA(const PathTracerShaderTypes::MaterialData& material) {
    return simd::max(simd_make_float3(material.dielectricSigmaA.x,
                                      material.dielectricSigmaA.y,
                                      material.dielectricSigmaA.z),
                     simd_make_float3(0.0f, 0.0f, 0.0f));
}

// Matches Metal transmission_tint: the type-7 BTDF is tinted by the resolved
// base color per crossing, times KHR_materials_volume absorption over its thickness.
simd::float3 DielectricTransmissionTint(const PathTracerShaderTypes::MaterialData& material,
                                        float cosTheta) {
    const simd::float3 surfaceTint = MaterialBaseColor(material);
    const float thickness = std::max(material.typeEta.w, 0.0f);
    simd::float3 sigmaA = DielectricSigmaA(material);
    if (thickness <= 0.0f || (sigmaA.x <= 0.0f && sigmaA.y <= 0.0f && sigmaA.z <= 0.0f)) {
        return surfaceTint;
    }
    float distance = thickness / std::max(std::fabs(cosTheta), 1.0e-3f);
    return surfaceTint * Clamp01(Exp3(-sigmaA * distance));
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

// A vector with every component v. (simd_make_float3 with one scalar argument zero-extends it: (v, 0, 0).)
simd::float3 Splat3(float v) {
    return simd_make_float3(v, v, v);
}

// Metal's plastic_coat_crossing: one pass through the coat's absorbing layer at angle cosTheta.
simd::float3 PlasticCoatCrossing(const PathTracerShaderTypes::MaterialData& material, float cosTheta) {
    const float thickness = PlasticCoatThickness(material);
    if (thickness <= 0.0f) {
        return Splat3(1.0f);
    }
    return Exp3(-PlasticCoatAbsorption(material) * (thickness / std::max(cosTheta, 1.0e-3f)));
}

// Metal's plastic_coat_fresnel_exact: the coat's exact dielectric Fresnel (unpolarized, from air into coatIOR). A coated
// random-walk subsurface material uses it for its coat lobe and for the walk's crossing of the coat, as Arnold's
// standard_surface does; other plastics keep Schlick's approximation.
float PlasticCoatFresnelExact(const PathTracerShaderTypes::MaterialData& material, float cosTheta) {
    float cosThetaT = 0.0f;
    return FresnelDielectricExact(std::clamp(cosTheta, 0.0f, 1.0f), 1.0f, PlasticCoatIor(material), cosThetaT);
}

// Metal's subsurface_has_coat.
bool SubsurfaceHasCoat(const PathTracerShaderTypes::MaterialData& material) {
    return material.sssParams.z > 0.5f;
}

// Metal's subsurface_dense_limit_material: what a subsurface surface is as its mean free path vanishes, a diffuse
// surface of its base colour, under its coat when it has one (the plastic closure, with the scene's coatIOR, which
// the loader keeps in typeEta.z; typeEta.y is the material ior, which the walk does not use).
PathTracerShaderTypes::MaterialData SubsurfaceDenseLimitMaterial(const PathTracerShaderTypes::MaterialData& material) {
    PathTracerShaderTypes::MaterialData dense = material;
    if (SubsurfaceHasCoat(material)) {
        dense.typeEta.x = static_cast<float>(PathTracerShaderTypes::MaterialType::Plastic);
        dense.typeEta.y = std::max(material.typeEta.z, 1.0f);
    } else {
        dense.typeEta.x = static_cast<float>(PathTracerShaderTypes::MaterialType::Lambertian);
    }
    return dense;
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

// Effective flake coverage of the CarPaint under-layer (see CarpaintClosureEval).
float CarpaintFlakeCoverage(const PathTracerShaderTypes::MaterialData& material) {
    return std::clamp(material.carpaintFlakeParams.x, 0.0f, 1.0f);
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

// Path regularization for dielectrics flagged kMaterialFlagRegularizedDielectric
// (as Metal): once a path has scattered off a rough surface, the smooth interface
// is treated as rough GGX with the path's maximum roughness, so the sun and small
// lights become reachable through it (a pool floor lit through the water surface).
// The integrator applies this to its material copy: baseColorRoughness.w holds the
// regularized roughness and typeEta.y the relative IOR across the interface (far
// side over incident side).
bool MaterialIsRegularizedDielectric(const PathTracerShaderTypes::MaterialData& material) {
    return static_cast<uint32_t>(material.typeEta.x) ==
               static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Dielectric) &&
           (material.materialFlags & PathTracerShaderTypes::kMaterialFlagRegularizedDielectric) != 0u &&
           material.baseColorRoughness.w > 0.0f;
}

void RegularizeDielectricMaterial(PathTracerShaderTypes::MaterialData& material,
                                  bool frontFace,
                                  float pathRoughness) {
    if (pt_shared::source_model(material) == 3.0f) {
        pt_shared::source_regularize_dielectric(material, frontFace);  // as Metal's regularize_dielectric_material
        return;
    }
    if (static_cast<uint32_t>(material.typeEta.x) !=
            static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Dielectric) ||
        (material.materialFlags & PathTracerShaderTypes::kMaterialFlagRegularizedDielectric) == 0u ||
        material.typeEta.w > 0.5f ||
        !(pathRoughness > 0.0f)) {
        return;
    }
    const float ior = std::max(material.typeEta.y, 1.0f);
    material.typeEta.y = frontFace ? ior : 1.0f / ior;
    material.baseColorRoughness.w = std::min(pathRoughness, 1.0f);
}

// Roughness a non-delta scattering event contributes to path regularization.
float PathRegularizationRoughness(const PathTracerShaderTypes::MaterialData& material) {
    return static_cast<uint32_t>(material.typeEta.x) ==
               static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Lambertian)
        ? 1.0f
        : std::clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
}

bool MaterialIsDelta(const PathTracerShaderTypes::MaterialData& material) {
    if (pt_shared::source_model(material) > 0.0f) {
        // Metal's material_is_delta for a source closure (closures_eval.metal).
        return !pt_shared::source_never_delta(material) && pt_shared::source_has_delta_lobe(material);
    }
    const uint32_t type = static_cast<uint32_t>(material.typeEta.x);
    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Dielectric)) {
        return !MaterialIsRegularizedDielectric(material);
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

// Metal's path-ray alpha test (path_integrators.metal): MASK compares alpha with its cutoff. BLEND is stochastic
// coverage: alpha, read at mip 0 as the shadow test reads it, is the probability that the ray stops.
bool PathRayAlphaDiscards(const PathTracerShaderTypes::MaterialData& material, Rng& rng) {
    const float alphaMode = material.pbrExtras.w;
    if (!(alphaMode > 0.5f)) {
        return false;
    }
    const float alpha = std::clamp(material.pbrExtras.x, 0.0f, 1.0f);
    if (alphaMode < 1.5f) {
        return alpha < std::clamp(material.pbrExtras.y, 0.0f, 1.0f);
    }
    return rng.nextFloat() > alpha;
}

// Metal's shadow_blend_alpha_sample: a hash of the hit, so a shadow ray passes a BLEND texel with probability
// 1 - alpha, as a path ray does, without drawing from the path's RNG stream. Embree has no frame index here; the hit's
// barycentrics and distance change with every jittered sample.
float ShadowBlendAlphaSample(const HitInfo& hit) {
    uint32_t h = pt_shared::pcg_hash(hit.primitiveIndex);
    h = pt_shared::pcg_hash(h ^ hit.meshIndex);
    h = pt_shared::pcg_hash(h ^ std::bit_cast<uint32_t>(hit.position.x));
    h = pt_shared::pcg_hash(h ^ std::bit_cast<uint32_t>(hit.position.y));
    h = pt_shared::pcg_hash(h ^ std::bit_cast<uint32_t>(hit.position.z));
    return static_cast<float>(h >> 8u) * (1.0f / 16777216.0f);
}

// Metal's shadow-ray alpha test (shadow_hit_alpha_cutout_discarded): MASK compares alpha with its cutoff, which
// referencePunctualShadow raises to 1 for SingleScatterSpecular blockers of point and sun rays. BLEND lets light
// through a fully transparent texel always, and through a partly transparent one with probability 1 - alpha
// (blendSample is uniform in [0, 1)).
bool ShadowRayAlphaDiscards(const PathTracerShaderTypes::MaterialData& material, const bool referencePunctualShadow,
                            const float blendSample) {
    const float alphaMode = material.pbrExtras.w;
    if (!(alphaMode > 0.5f)) {
        return false;
    }
    const float alpha = std::clamp(material.pbrExtras.x, 0.0f, 1.0f);
    if (alphaMode < 1.5f) {
        const bool sourceMask = referencePunctualShadow &&
            (material.materialFlags & PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular) != 0u;
        return alpha < (sourceMask ? 1.0f : std::clamp(material.pbrExtras.y, 0.0f, 1.0f));
    }
    return alpha <= 0.0f || (alpha < 1.0f && blendSample >= alpha);
}

bool PavilionPunctualShadowEnabled(const PathTracerShaderTypes::MaterialData& receiver,
                                   const bool punctual) {
    return punctual &&
        (receiver.materialFlags & PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular) != 0u;
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

// EP8d: source transport (renderer sourceTransport=1; the Bitterli presets). Metal's wavefront spawns every bounce
// and shadow ray at offset_interface_origin with tMin = source_ray_epsilon (sourceRayEpsilon, or 1e-6), excludes only
// the spawning triangle, and stops a shadow ray 1e-6 short of a light (tracing/visibility.metal,
// wavefront/kernels.metal). Set once per render from the settings; read by the spawn and shadow helpers.
struct SourceTransportPolicy {
    bool enabled = false;
    float epsilon = 1.0e-6f;
    // Metal's sourceOpacityPresent (UniformBuilder): some material has source opacity. With or without source
    // transport, every wavefront shadow ray then runs the source-aware alpha trace (kernels.metal).
    bool opacityPresent = false;
};
SourceTransportPolicy gSourceTransport;

// Metal's offset_interface_origin: four float ULPs of the point (at least 0.1 micrometre) along the geometric normal,
// on the side the ray leaves, plus 1e-7 along the ray. The side times the normal does not depend on the normal's
// facing, so Embree's unfaced geometric normal gives the same point.
simd::float3 InterfaceOrigin(const HitInfo& hit, const simd::float3& direction) {
    const simd::float3 n = hit.normal;
    const float side = simd::dot(direction, n) >= 0.0f ? 1.0f : -1.0f;
    const simd::float3 step = simd::max(simd::fabs(hit.position) * (4.0f * 1.1920928955078125e-7f),
                                        simd_make_float3(1.0e-7f, 1.0e-7f, 1.0e-7f));
    return hit.position + side * step * n + direction * 1.0e-7f;
}

// Metal's offset_ray_origin: the spawn of every ray when source transport is off, and of the PBR alpha pass-through
// ray under it too (kernels.metal keeps it with kEpsilon there).
simd::float3 LegacyOffsetRayOrigin(const HitInfo& hit, const simd::float3& direction) {
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

// Metal's source_ray_epsilon (visibility.metal): sourceRayEpsilon under source transport, else kInterfaceMinT.
inline float SourceRayEpsilon() { return gSourceTransport.enabled ? gSourceTransport.epsilon : 1.0e-6f; }

// Metal's interface_precision(material) || source_transport_enabled (kernels.metal): rays leaving this material spawn
// at offset_interface_origin with the source epsilon as tMin, excluding only the spawning triangle. Without source
// transport this holds for interfacePrecision materials only.
inline bool InterfaceSpawn(const PathTracerShaderTypes::MaterialData* material) {
    return gSourceTransport.enabled ||
           (material && pt_shared::source_material_flag(*material, PathTracerShaderTypes::kMaterialFlagInterfacePrecision));
}

// Metal's offset_spawn_origin and the wavefront's bounce origin: the interface origin for an interface spawn, else
// offset_ray_origin.
simd::float3 SpawnOrigin(const HitInfo& hit, const simd::float3& direction,
                         const PathTracerShaderTypes::MaterialData* material) {
    return InterfaceSpawn(material) ? InterfaceOrigin(hit, direction) : LegacyOffsetRayOrigin(hit, direction);
}

// Metal's wavefront shadow kernels run the source-aware alpha trace under source transport or when the scene has
// source opacity (kernels.metal); it weights the light by 1 - opacity at each opacity hit.
inline bool SourceShadowActive() { return gSourceTransport.enabled || gSourceTransport.opacityPresent; }

struct RectLightSample {
    simd::float3 direction{0.0f, 0.0f, 0.0f};
    float distance = 0.0f;
    float pdf = 0.0f;
    simd::float3 emission{0.0f, 0.0f, 0.0f};
    uint32_t rectIndex = 0;
    uint32_t lightIndex = 0;
    float area = 0.0f;
    float cosLight = 0.0f;
    bool isReferencePoint = false;
};

struct RectLightHit {
    simd::float3 emission{0.0f, 0.0f, 0.0f};
    float pdf = 0.0f;
};

struct CarpaintLobeProbabilities {
    float coat = 0.0f;
    float flake = 0.0f;
    float diffuse = 0.0f;
    float metal = 0.0f;
};

// A closure (or one of its lobes) evaluated for a direction pair: its value f and the pdf its sampler draws wi with.
struct ClosureValuePdf {
    simd::float3 value{0.0f, 0.0f, 0.0f};
    float pdf = 0.0f;
};

struct PlasticLobeProbabilities {
    float coat = 0.0f;
    float diffuse = 0.0f;
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

bool ExactReferencePointLight(const RectLightInfo& light) {
    return light.isReferencePoint && light.jitterRadius == 0.0f;
}

uint32_t CountAnalyticLights(const std::vector<RectLightInfo>& lights, bool excludeExactPoints = false) {
    if (!excludeExactPoints) {
        return static_cast<uint32_t>(lights.size());
    }
    return static_cast<uint32_t>(std::count_if(lights.begin(), lights.end(), [](const RectLightInfo& light) {
        return !ExactReferencePointLight(light);
    }));
}

bool SampleReferencePointLight(const RectLightInfo& light, const HitInfo& hit,
                               uint32_t lightCount, simd::float3 diskOffset,
                               RectLightSample& outSample) {
    const simd::float3 centerDirection = light.corner - hit.position;
    if (!(simd::dot(centerDirection, centerDirection) > 1.0e-12f)) {
        return false;
    }
    const Onb frame = BuildOnb(centerDirection);
    const simd::float3 point = light.corner + light.jitterRadius *
        (diskOffset.x * frame.tangent + diskOffset.y * frame.bitangent);
    const simd::float3 toLight = point - hit.position;
    const float distanceSq = simd::dot(toLight, toLight);
    if (!(distanceSq > 0.0f)) {
        return false;
    }
    outSample.distance = std::sqrt(distanceSq);
    outSample.direction = toLight / outSample.distance;
    outSample.pdf = 1.0f / static_cast<float>(std::max(lightCount, 1u));
    const float legacyCap = light.legacyMinimumDistanceSq > 0.0f
        ? 1.0f / light.legacyMinimumDistanceSq : 1.0f;
    const float attenuation = light.inverseSquareFalloff
        ? 1.0f / distanceSq : std::min(1.0f / distanceSq, legacyCap);
    outSample.emission = light.baseEmission * attenuation;
    outSample.rectIndex = light.rectIndex;
    outSample.isReferencePoint = true;
    return IsFinite(outSample.emission) && simd::dot(outSample.emission, outSample.emission) > 0.0f;
}

bool DeterministicRectLightAuditSample(const RectLightInfo& light,
                                       uint32_t lightCount,
                                       const EnvironmentMap* env,
                                       const PathTracer::RenderSettings& settings,
                                       const HitInfo& hit,
                                       RectLightSample& outSample) {
    outSample = RectLightSample{};
    if (light.isReferencePoint) {
        return SampleReferencePointLight(light, hit, lightCount, simd_make_float3(0.0f), outSample);
    }
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
                     RectLightSample& outSample,
                     bool excludeExactPoints = false) {
    outSample = RectLightSample{};
    const uint32_t lightCount = CountAnalyticLights(rectLights, excludeExactPoints);
    if (lightCount == 0u) {
        return false;
    }

    const uint32_t ordinal = std::min(static_cast<uint32_t>(rng.nextFloat() * static_cast<float>(lightCount)), lightCount - 1u);
    uint32_t selected = 0u, current = 0u;
    for (; selected < rectLights.size(); ++selected) {
        if ((!excludeExactPoints || !ExactReferencePointLight(rectLights[selected])) && current++ == ordinal) {
            break;
        }
    }
    const RectLightInfo& light = rectLights[selected];
    outSample.lightIndex = selected;
    if (light.isReferencePoint) {
        return SampleReferencePointLight(light, hit, lightCount, SampleInUnitDisk(rng), outSample);
    }

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
    float selectionPdf = 1.0f / static_cast<float>(lightCount);
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

bool SampleRectLightNeeDraw(const std::vector<RectLightInfo>& lights, const EnvironmentMap* env,
                           const PathTracer::RenderSettings& settings, const HitInfo& hit,
                           Rng& rng, RectLightSample& sample, uint32_t draw, bool exactActive) {
    if (!exactActive) return SampleRectLight(lights, env, settings, hit, rng, sample);
    if (draw == 0u) return SampleRectLight(lights, env, settings, hit, rng, sample, true);
    const uint32_t index = draw - 1u;
    if (index >= lights.size() || !ExactReferencePointLight(lights[index])) return false;
    sample = RectLightSample{};
    sample.lightIndex = index;
    return SampleReferencePointLight(lights[index], hit, 1u, simd_make_float3(0.0f), sample);
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
    if (AreaPrimitiveIsReferencePoint(rect)) {
        return 0.0f;
    }
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
                                   settings.referencePointExactNee
                                       ? CountAnalyticLights(rectLights, true)
                                       : static_cast<uint32_t>(rectLights.size()),
                                   hit,
                                   origin);
    if (!(pdf > 0.0f) || !std::isfinite(pdf)) {
        return false;
    }

    outHit.emission = emission;
    outHit.pdf = pdf;
    return true;
}

float GgxVndfAlpha(float roughness) {
    return std::max(roughness * roughness, 1.0e-4f);  // SampleGgxVndf's alpha
}

// GGX reflection lobe around lobeNormal (mirrors ggx_reflection_lobe): value F D G2 / (4 cos_o cos_macro(wi)); pdf
// the VNDF reflection density for any reflected direction (also below the lobe plane, where the value is zero).
ClosureValuePdf GgxReflectionLobe(float alpha,
                                  const simd::float3& lobeNormal,
                                  float cosThetaMacroI,
                                  const simd::float3& wo,
                                  const simd::float3& wi,
                                  const simd::float3& F,
                                  const FireflyClampParams& clampParams) {
    ClosureValuePdf result;
    float cosThetaO = simd::dot(lobeNormal, wo);
    simd::float3 wh = simd::normalize(wo + wi);
    float cosThetaH = simd::dot(lobeNormal, wh);
    if (cosThetaO <= 0.0f || cosThetaH <= 0.0f || simd::dot(wo, wh) <= 0.0f) {
        return result;
    }
    float D = GgxDistribution(alpha, lobeNormal, wh);
    float g1o = GgxG1(alpha, cosThetaO);
    result.pdf = ClampSpecularPdf(D * g1o / (4.0f * cosThetaO), clampParams);
    float cosThetaI = simd::dot(lobeNormal, wi);
    if (cosThetaI > 0.0f && cosThetaMacroI > 0.0f) {
        float G = g1o * GgxG1(alpha, cosThetaI);
        result.value = SanitizeSpecularValue(F * (D * G / (4.0f * cosThetaO * cosThetaMacroI)));
    }
    return result;
}

// CarPaint (type 6) layered closure; mirrors carpaint_closure_eval / carpaint_lobe_probabilities and the
// case 6u sampler in shaders/bsdf (see the comment block there):
//   f = f_coat + (1 - Fc(cos_o)) (1 - Fc(cos_i)) * coat transmittance * [c f_flake + (1 - c)((1 - m) base/pi + m f_metal)]
// sampled by one-sample MIS over {coat, flake, diffuse, metal}; GGX lobe pdfs are the VNDF reflection density.
float CarpaintFlakeLobeRoughness(const PathTracerShaderTypes::MaterialData& material) {
    return std::max(CarpaintFlakeRoughness(material), 1.0e-3f);
}

float CarpaintMetalRoughness(const PathTracerShaderTypes::MaterialData& material) {
    return std::max(CarpaintBaseRoughness(material), 1.0e-3f);
}

float CarpaintCoatFresnel(const PathTracerShaderTypes::MaterialData& material, float cosTheta) {
    float f0 = DielectricF0(std::max(material.typeEta.z, 1.0f));
    return f0 + (1.0f - f0) * SchlickWeight(cosTheta);
}

simd::float3 CarpaintAverageFresnel(const simd::float3& f0) {
    return f0 + (simd_make_float3(1.0f, 1.0f, 1.0f) - f0) * (1.0f / 21.0f);
}

CarpaintLobeProbabilities CarpaintLobeProbabilitiesFor(const PathTracerShaderTypes::MaterialData& material,
                                                       const simd::float3& normal,
                                                       const simd::float3& flakeNormal,
                                                       const simd::float3& wo) {
    CarpaintLobeProbabilities probs;
    float cosThetaO = simd::dot(normal, wo);
    if (cosThetaO <= 0.0f) {
        return probs;
    }
    float coverage = CarpaintFlakeCoverage(material);
    float metallic = CarpaintBaseMetallic(material);
    float coatReflect = CarpaintCoatFresnel(material, cosThetaO);
    float transmitted = 1.0f - coatReflect;
    float underF = Luminance(CarpaintAverageFresnel(CarpaintBaseF0(material)));
    probs.coat = coatReflect;
    probs.flake = (simd::dot(flakeNormal, wo) > 0.0f) ? transmitted * coverage * underF : 0.0f;
    probs.diffuse = transmitted * (1.0f - coverage) * (1.0f - metallic) * Luminance(MaterialBaseColor(material));
    probs.metal = transmitted * (1.0f - coverage) * metallic * underF;
    float sum = probs.coat + probs.flake + probs.diffuse + probs.metal;
    if (!(sum > 0.0f)) {
        return CarpaintLobeProbabilities{};
    }
    float inv = 1.0f / sum;
    probs.coat *= inv;
    probs.flake *= inv;
    probs.diffuse *= inv;
    probs.metal *= inv;
    return probs;
}

ClosureValuePdf CarpaintClosureEval(const PathTracerShaderTypes::MaterialData& material,
                                    const simd::float3& normal,
                                    const simd::float3& flakeNormal,
                                    const simd::float3& wo,
                                    const simd::float3& wi,
                                    const CarpaintLobeProbabilities& probs,
                                    const FireflyClampParams& clampParams) {
    ClosureValuePdf result;
    float cosThetaO = simd::dot(normal, wo);
    float cosThetaI = simd::dot(normal, wi);
    if (cosThetaO <= 0.0f || cosThetaI <= 0.0f) {
        return result;
    }
    simd::float3 wh = simd::normalize(wo + wi);
    float coverage = CarpaintFlakeCoverage(material);
    float metallic = CarpaintBaseMetallic(material);
    simd::float3 baseColor = MaterialBaseColor(material);
    float coatF = CarpaintCoatFresnel(material, simd::dot(wi, wh));

    ClosureValuePdf coat = GgxReflectionLobe(GgxVndfAlpha(PlasticCoatRoughness(material)), normal, cosThetaI,
                                             wo, wi, simd_make_float3(coatF, coatF, coatF), clampParams);
    ClosureValuePdf flake = GgxReflectionLobe(GgxVndfAlpha(CarpaintFlakeLobeRoughness(material)), flakeNormal,
                                              cosThetaI, wo, wi,
                                              SchlickFresnel(CarpaintBaseF0(material), simd::dot(wi, wh)),
                                              clampParams);
    simd::float3 metalF = CarpaintHasBaseConductor(material)
        ? FresnelConductor(simd::dot(wi, wh), CarpaintBaseEta(material), CarpaintBaseK(material))
        : SchlickFresnel(baseColor, simd::dot(wi, wh));
    ClosureValuePdf metal = GgxReflectionLobe(GgxVndfAlpha(CarpaintMetalRoughness(material)), normal, cosThetaI,
                                              wo, wi, metalF, clampParams);
    simd::float3 diffuse = baseColor * (1.0f / kPi);

    simd::float3 coatTransmission = (1.0f - CarpaintCoatFresnel(material, cosThetaO)) *
                                    (1.0f - CarpaintCoatFresnel(material, cosThetaI)) *
                                    PlasticDiffuseTransmission(material, cosThetaI, cosThetaO);
    simd::float3 under = coverage * flake.value +
                         (1.0f - coverage) * ((1.0f - metallic) * diffuse + metallic * metal.value);
    result.value = simd::max(coat.value + coatTransmission * under, simd_make_float3(0.0f, 0.0f, 0.0f));
    result.pdf = probs.coat * coat.pdf + probs.flake * flake.pdf +
                 probs.diffuse * LambertPdf(normal, wi) + probs.metal * metal.pdf;
    return result;
}

// Plastic (type 4) closure; mirrors plastic_closure_eval / plastic_lobe_probabilities and the case 4u sampler in
// shaders/bsdf (see the comment block there):
//   f = f_coat + (1 - F(cos_i)) (1 - F(cos_o)) (1 - Favg) * coat transmittance * base/pi
// sampled by one-sample MIS over {coat, diffuse}; the coat pdf is the VNDF reflection density.
PlasticLobeProbabilities PlasticLobeProbabilitiesFor(const PathTracerShaderTypes::MaterialData& material,
                                                     const simd::float3& normal,
                                                     const simd::float3& wo) {
    PlasticLobeProbabilities probs;
    float cosThetaO = simd::dot(normal, wo);
    if (cosThetaO <= 0.0f) {
        return probs;
    }
    if ((material.materialFlags & PathTracerShaderTypes::kMaterialFlagSubsurfaceExit) != 0u) {
        probs.diffuse = 1.0f;  // a coated walk's exit point: the diffuse base only (Metal, plastic_lobe_probabilities)
        return probs;
    }
    float f0 = DielectricF0(PlasticCoatIor(material));
    float coatReflect = f0 + (1.0f - f0) * SchlickWeight(cosThetaO);
    probs.coat = coatReflect * Luminance(PlasticSpecularTint(material));
    // (1 - F(cos_i)) averages to 1 - (f0 + (1 - f0) / 21) over the cosine-weighted hemisphere; T at cos_i = 1.
    float entering = (1.0f - f0) * (20.0f / 21.0f);
    probs.diffuse = (1.0f - coatReflect) * entering * std::max(1.0f - PlasticCoatFresnelAverage(material), 0.0f) *
                    Luminance(MaterialBaseColor(material) * PlasticDiffuseTransmission(material, 1.0f, cosThetaO));
    float sum = probs.coat + probs.diffuse;
    if (!(sum > 0.0f)) {
        return PlasticLobeProbabilities{};
    }
    probs.coat /= sum;
    probs.diffuse /= sum;
    return probs;
}

ClosureValuePdf PlasticClosureEval(const PathTracerShaderTypes::MaterialData& material,
                                   const simd::float3& normal,
                                   const simd::float3& wo,
                                   const simd::float3& wi,
                                   const PlasticLobeProbabilities& probs,
                                   const FireflyClampParams& clampParams) {
    ClosureValuePdf result;
    float cosThetaO = simd::dot(normal, wo);
    float cosThetaI = simd::dot(normal, wi);
    if (cosThetaO <= 0.0f || cosThetaI <= 0.0f) {
        return result;
    }
    float f0 = DielectricF0(PlasticCoatIor(material));
    simd::float3 f0Color = simd_make_float3(f0, f0, f0);
    simd::float3 one = simd_make_float3(1.0f, 1.0f, 1.0f);
    if ((material.materialFlags & PathTracerShaderTypes::kMaterialFlagSubsurfaceExit) != 0u) {
        // Exit point of a coated subsurface walk (Metal, plastic_closure_eval): a white diffuse surface under the coat's
        // tint and its absorption along the light-side crossing. The coat's Fresnel weights the walk once, at its entry,
        // for the view direction, as in Arnold's layering; the exit takes no Fresnel and no (1 - Favg).
        result.value = MaterialBaseColor(material) * (1.0f / kPi) * PlasticCoatTint(material) *
                       PlasticCoatCrossing(material, cosThetaI);
        result.pdf = LambertPdf(normal, wi);
        return result;
    }
    simd::float3 wh = simd::normalize(wo + wi);
    // The coat lobe of a coated random-walk vertex takes the exact dielectric Fresnel, as Arnold's specular layer; other
    // plastics keep Schlick's approximation.
    const simd::float3 coatFresnel = (material.materialFlags & PathTracerShaderTypes::kMaterialFlagSubsurfaceCoatLobe) != 0u
                                         ? Splat3(PlasticCoatFresnelExact(material, simd::dot(wi, wh)))
                                         : SchlickFresnel(f0Color, simd::dot(wi, wh));
    ClosureValuePdf coat = GgxReflectionLobe(GgxVndfAlpha(PlasticCoatRoughness(material)), normal, cosThetaI, wo, wi,
                                             coatFresnel * PlasticSpecularTint(material), clampParams);
    simd::float3 diffuse = MaterialBaseColor(material) * (1.0f / kPi) *
                           PlasticDiffuseTransmission(material, cosThetaI, cosThetaO) *
                           (one - SchlickFresnel(f0Color, cosThetaI)) * (one - SchlickFresnel(f0Color, cosThetaO)) *
                           std::max(1.0f - PlasticCoatFresnelAverage(material), 0.0f);
    result.value = simd::max(coat.value + diffuse, simd_make_float3(0.0f, 0.0f, 0.0f));
    result.pdf = probs.coat * coat.pdf + probs.diffuse * LambertPdf(normal, wi);
    return result;
}

// Frostbite's renormalized Disney diffuse, which Falcor's StandardMaterial uses
// (kMaterialFlagFalcorStandardBsdf), as Metal.
float FrostbiteDiffuseScale(const simd::float3& normal,
                            const simd::float3& wo,
                            const simd::float3& wi,
                            float linearRoughness) {
    const float wiDotH = std::clamp(simd::dot(wi, SafeNormalize(wo + wi, simd_make_float3(0.0f, 0.0f, 0.0f))), 0.0f, 1.0f);
    const float fd90 = 0.5f * linearRoughness + 2.0f * wiDotH * wiDotH * linearRoughness;
    const float wiScatter = 1.0f + (fd90 - 1.0f) * std::pow(1.0f - std::clamp(simd::dot(normal, wi), 0.0f, 1.0f), 5.0f);
    const float woScatter = 1.0f + (fd90 - 1.0f) * std::pow(1.0f - std::clamp(simd::dot(normal, wo), 0.0f, 1.0f), 5.0f);
    return wiScatter * woScatter * (1.0f + (1.0f / 1.51f - 1.0f) * linearRoughness);
}

// Rough GGX dielectric (PBRT-v4 DielectricBxDF, radiance transport) for a
// regularized interface, as Metal. `normal` faces wo.
BsdfEval EvaluateRegularizedDielectric(const PathTracerShaderTypes::MaterialData& material,
                                       const simd::float3& normal,
                                       const simd::float3& wo,
                                       const simd::float3& wi,
                                       const bool referencePunctual) {
    BsdfEval result;
    const float cosThetaO = simd::dot(normal, wo);
    const float cosThetaI = simd::dot(normal, wi);
    if (!(cosThetaO > 0.0f) || cosThetaI == 0.0f) {
        return result;
    }
    const float eta = std::max(material.typeEta.y, 1.0e-3f);
    const float roughness = std::clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
    const float alpha = std::max(roughness * roughness, 1.0e-4f);
    const bool reflection = cosThetaI > 0.0f;
    simd::float3 wh = reflection ? (wo + wi) : (wo + wi * eta);
    if (!(simd::dot(wh, wh) > 0.0f)) {
        return result;
    }
    wh = simd::normalize(wh);
    if (simd::dot(wh, normal) < 0.0f) {
        wh = -wh;
    }
    const float woDotWh = simd::dot(wo, wh);
    const float wiDotWh = simd::dot(wi, wh);
    if (woDotWh <= 0.0f || wiDotWh * cosThetaI <= 0.0f) {
        return result;
    }

    float cosThetaT = 0.0f;
    const float F = FresnelDielectricExact(woDotWh, 1.0f, eta, cosThetaT);
    const float D = GgxDistribution(alpha, normal, wh);
    // The Pavilion source limits discrete-light values on water/glass too.
    // The actual VNDF density and ordinary continuation remain uncapped.
    const bool referenceContinuation =
        (material.materialFlags & PathTracerShaderTypes::kMaterialFlagReferenceDielectricContinuation) != 0u;
    const float valueD = referencePunctual && referenceContinuation ? std::min(D, 1.0f) : D;
    const float G = GgxG1(alpha, cosThetaO) * GgxG1(alpha, cosThetaI);
    const float pdfWh = GgxVndfPdf(alpha, normal, wo, wh, /*halfVectorD=*/true);
    if (reflection) {
        const float f = F * valueD * G / (4.0f * cosThetaO * cosThetaI);
        result.value = simd_make_float3(f, f, f);
        result.pdf = F * pdfWh / (4.0f * woDotWh);
    } else {
        const float denom = wiDotWh + woDotWh / eta;
        const float denomSq = denom * denom;
        if (!(denomSq > 1.0e-12f)) {
            return result;
        }
        // Radiance crossing into the far medium scales by eta^-2.
        const float ft = (1.0f - F) * valueD * G *
                         std::fabs(wiDotWh * woDotWh / (cosThetaI * cosThetaO * denomSq)) /
                         (eta * eta);
        const simd::float3 tint =
            (material.materialFlags & PathTracerShaderTypes::kMaterialFlagDielectricTransmissionTint) != 0u
                ? simd::clamp(MaterialBaseColor(material),
                              simd_make_float3(0.0f, 0.0f, 0.0f),
                              simd_make_float3(1.0f, 1.0f, 1.0f))
                : simd_make_float3(1.0f, 1.0f, 1.0f);
        result.value = ft * tint;
        result.pdf = (1.0f - F) * pdfWh * std::fabs(wiDotWh) / denomSq;
    }
    return result;
}

BsdfSample SampleRegularizedDielectric(const PathTracerShaderTypes::MaterialData& material,
                                       const simd::float3& normal,
                                       const simd::float3& wo,
                                       bool frontFace,
                                       Rng& rng) {
    BsdfSample result;
    result.lobeType = 1u;
    result.lobeRoughness = std::clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
    if (!(simd::dot(normal, wo) > 0.0f)) {
        return result;
    }
    const float eta = std::max(material.typeEta.y, 1.0e-3f);
    const simd::float3 wh = SampleGgxVndf(rng, result.lobeRoughness, normal, wo);
    const float woDotWh = simd::dot(wo, wh);
    if (!(woDotWh > 0.0f)) {
        return result;
    }
    float cosThetaT = 0.0f;
    const float F = FresnelDielectricExact(woDotWh, 1.0f, eta, cosThetaT);
    const bool reflection = rng.nextFloat() < F;
    const simd::float3 wi = reflection
        ? Reflect(-wo, wh)
        : SafeNormalize(-wo / eta + (woDotWh / eta - cosThetaT) * wh, simd_make_float3(0.0f, 0.0f, 0.0f));
    const float cosThetaI = simd::dot(normal, wi);
    if (reflection ? !(cosThetaI > 0.0f) : !(cosThetaI < 0.0f)) {
        return result;
    }
    const BsdfEval eval = EvaluateRegularizedDielectric(material, normal, wo, wi, false);
    if (!(eval.pdf > 0.0f)) {
        return result;
    }
    result.direction = wi;
    result.weight = eval.value * std::fabs(cosThetaI) / eval.pdf;
    if (!reflection &&
        (material.materialFlags & PathTracerShaderTypes::kMaterialFlagReferenceDielectricContinuation) != 0u) {
        result.weight *= eta * eta;
    }
    result.pdf = eval.pdf;
    result.mediumEvent = reflection ? 0 : (frontFace ? 1 : -1);  // as Metal's sample_regularized_dielectric
    return result;
}

// EP8b: Tungsten source closures through the shared header, dispatched first as Metal's evaluate_bsdf and
// sample_bsdf dispatch them (closures_eval.metal, bsdf_sample.metal). Values exclude the cosine on both sides.
BsdfEval EvaluateSourceClosure(const PathTracerShaderTypes::MaterialData& material,
                               const simd::float3& normal,
                               const simd::float3& wo,
                               const simd::float3& wi) {
    const pt_shared::BsdfEvalResult r = pt_shared::source_eval(material, normal, wo, wi, false);
    BsdfEval result;
    result.value = r.value;
    result.pdf = r.pdf;
    result.isDelta = r.isDelta;
    return result;
}

BsdfSample SampleSourceClosure(const PathTracerShaderTypes::MaterialData& material,
                               const simd::float3& normal,
                               const simd::float3& wo,
                               bool frontFace,
                               Rng& rng) {
    const pt_shared::BsdfSampleResult r = pt_shared::source_sample(material, normal, wo, frontFace, rng.state, false);
    BsdfSample result;
    result.direction = r.direction;
    result.weight = r.weight;
    result.pdf = r.pdf;
    result.lobeType = r.lobeType;
    result.lobeRoughness = r.lobeRoughness;
    result.isDelta = r.isDelta;
    result.mediumEvent = r.mediumEvent;
    return result;
}

BsdfEval EvaluateBsdf(const PathTracerShaderTypes::MaterialData& material,
                      const simd::float3& position,
                      const simd::float3& normal,
                      const simd::float3& wo,
                      const simd::float3& wi,
                      const FireflyClampParams& clampParams,
                      const bool referencePunctual = false,
                      const bool pbrFrontFace = true) {
    if (pt_shared::source_model(material) > 0.0f) {
        return EvaluateSourceClosure(material, normal, wo, wi);
    }
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
        // Delta lobes carry discrete mass, not a continuous value or PDF.
        const bool hasDeltaLobes = roughness <= 1.0e-3f;
        const bool reference = (material.materialFlags & PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular) != 0u;
        const float relativeIor = std::max(material.typeEta.y, 1.0e-3f);
        float dielectricF0 = DielectricF0FromIor(reference ? std::max(relativeIor, 1.0f / relativeIor) : material.typeEta.y);
        simd::float3 f0 = baseColor * metallic +
                          simd_make_float3(dielectricF0, dielectricF0, dielectricF0) * (1.0f - metallic);
        simd::float3 diffuseColor = baseColor * (1.0f - metallic) *
                                    std::clamp(material.pbrParams.z, 0.0f, 1.0f);

        float transmission = std::clamp(material.pbrExtras.z, 0.0f, 1.0f) * (1.0f - metallic);
        const bool glassInterface = !reference && transmission > 0.0f;
        const bool thinWalled = glassInterface && ThinWalledPbrTransmission(material);
        float reflectScale = 1.0f - transmission;
        float specWeightBase = PbrSpecularWeight(f0);
        if ((material.materialFlags & PathTracerShaderTypes::kMaterialFlagFalcorStandardBsdf) != 0u) {
            // Falcor picks lobes by albedo: Fresnel-weighted specular against diffuse, as Metal.
            const float specAlbedo = Luminance(SchlickFresnel(f0, std::clamp(simd::dot(normal, wo), 0.0f, 1.0f)));
            specWeightBase = std::clamp(specAlbedo / std::max(specAlbedo + Luminance(diffuseColor), 1.0e-6f), 0.05f, 0.95f);
        }
        const float specularScale = reference ? 1.0f : reflectScale;
        float wSpec = specWeightBase * specularScale;
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
            float D = GgxDistribution(alpha, normal, wh);
            float G = GgxG1(alpha, signedCosThetaO) * GgxG1(alpha, signedCosThetaI);
            float cosThetaT = 0.0f;
            const float referenceF = reference ? FresnelDielectricExact(simd::dot(wo, wh), 1.0f, relativeIor, cosThetaT) : 0.0f;
            simd::float3 F = reference ? simd_make_float3(referenceF, referenceF, referenceF) * (1.0f - metallic) +
                SchlickFresnel(baseColor, simd::dot(wi, wh)) * metallic : SchlickFresnel(f0, simd::dot(wi, wh));
            float denom = 4.0f * signedCosThetaO * signedCosThetaI;
            const bool referenceLeaf = reference && material.pbrExtras.w > 0.5f && material.pbrExtras.w < 1.5f;
            const float shadingD = (referenceLeaf || (reference && referencePunctual))
                ? std::min(D, 1.0f) : D;
            simd::float3 spec = hasDeltaLobes ? simd_make_float3(0.0f) : F * (shadingD * G / std::max(denom, 1.0e-6f));
            // Falcor's StandardMaterial has no multiple-scattering compensation, and Metal applies none to materials
            // with kMaterialFlagFalcorStandardBsdf.
            if ((material.materialFlags & (PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular |
                                           PathTracerShaderTypes::kMaterialFlagFalcorStandardBsdf)) == 0u) {
                spec *= SpecularEnergyCompensation(f0, roughness, absCosThetaO);
            }
            spec = SanitizeSpecularValue(spec);
            spec *= specularScale;
            simd::float3 diffuse = (diffuseColor / kPi) * reflectScale;
            if ((material.materialFlags & PathTracerShaderTypes::kMaterialFlagFalcorStandardBsdf) != 0u) {
                diffuse *= FrostbiteDiffuseScale(normal, wo, wi, roughness);
            }
            if (reference) diffuse *= 1.0f - referenceF;
            // Falcor: the VNDF density over the reflection Jacobian 4*(wo.wh), as Metal.
            const float woDotWh = simd::dot(wo, wh);
            float pdfSpec = hasDeltaLobes ? 0.0f : ((reference ||
                             (material.materialFlags & PathTracerShaderTypes::kMaterialFlagFalcorStandardBsdf) == 0u)
                ? D * GgxG1(alpha, signedCosThetaO) / (4.0f * signedCosThetaO)
                : (woDotWh > 0.0f ? GgxVndfPdf(alpha, normal, wo, wh, /*halfVectorD=*/true) / (4.0f * woDotWh) : 0.0f));
            float pdfDiffuse = LambertPdf(normal, wi);
            float pdf = pSpec * pdfSpec + pDiff * pdfDiffuse;
            if (glassInterface && !hasDeltaLobes) {
                // A thin-walled sheet has the outside medium on both faces.
                const bool entering = pbrFrontFace || thinWalled;
                const float etaI = entering ? 1.0f : std::max(material.typeEta.y, 1.0f);
                const float etaT = entering ? std::max(material.typeEta.y, 1.0f) : 1.0f;
                float glassCosThetaT = 0.0f;
                const float glassF = FresnelDielectricExact(simd::dot(wo, wh), etaI, etaT, glassCosThetaT);
                // One exact-Fresnel interface; the glass reflection is single-scattering.
                spec += simd_make_float3(1.0f, 1.0f, 1.0f) * (transmission * glassF * D * G / std::max(denom, 1.0e-6f));
                pdf += pTrans * glassF * D * GgxG1(alpha, signedCosThetaO) / (4.0f * signedCosThetaO);
            }
            if (pdf > 0.0f) {
                result.value = simd::max(spec + diffuse, simd_make_float3(0.0f, 0.0f, 0.0f));
                result.pdf = reference ? pdf : ClampSpecularPdf(pdf, clampParams);
            }
            return result;
        }

        if (hasDeltaLobes || wTrans <= 0.0f) {
            return result;
        }
        if (thinWalled) {
            // Thin-walled BTDF (Metal evaluate_pbr_metallic_roughness): the GGX
            // reflection lobe of wi mirrored through the tangent plane, times 1 - F.
            const simd::float3 n = signedCosThetaO > 0.0f ? normal : -normal;
            const simd::float3 wiMirror = wi - 2.0f * simd::dot(wi, n) * n;
            const simd::float3 wh = simd::normalize(wo + wiMirror);
            const float woDotWh = simd::dot(wo, wh);
            if (!IsFinite(wh) || simd::dot(wh, n) <= 0.0f || woDotWh <= 0.0f || simd::dot(wiMirror, wh) <= 0.0f) {
                return result;
            }
            const float D = GgxDistribution(alpha, n, wh);
            const float G = GgxG1(alpha, absCosThetaO) * GgxG1(alpha, absCosThetaI);
            float cosThetaT = 0.0f;
            const float F = FresnelDielectricExact(woDotWh, 1.0f, std::max(material.typeEta.y, 1.0f), cosThetaT);
            simd::float3 ft = simd_make_float3(1.0f, 1.0f, 1.0f) *
                (transmission * (1.0f - F) * D * G / std::max(4.0f * absCosThetaO * absCosThetaI, 1.0e-6f));
            ft *= DielectricTransmissionTint(material, absCosThetaI);
            const float pdf = pTrans * (1.0f - F) * D * GgxG1(alpha, absCosThetaO) / (4.0f * absCosThetaO);
            if (pdf > 0.0f) {
                result.value = simd::max(ft, simd_make_float3(0.0f, 0.0f, 0.0f));
                result.pdf = ClampSpecularPdf(pdf, clampParams);
            }
            return result;
        }
        float etaI = 1.0f;
        float etaT = reference ? relativeIor : std::max(material.typeEta.y, 1.0f);
        if (!reference && (glassInterface ? !pbrFrontFace : signedCosThetaO < 0.0f)) {
            std::swap(etaI, etaT);
        }
        float eta = etaI / etaT;
        // Transmission half vector eta*wo+wi, with dwh/dwi = |wi.h|/denom^2.
        simd::float3 wh = simd::normalize(wo * eta + wi);
        if (!IsFinite(wh) || simd::dot(wh, wh) <= 0.0f) {
            return result;
        }
        if (simd::dot(wh, normal) <= 0.0f) {
            wh = -wh;
        }
        float cosThetaOWh = simd::dot(wo, wh);
        float cosThetaIWh = simd::dot(wi, wh);
        if (cosThetaOWh * cosThetaIWh > 0.0f ||
            (reference && (cosThetaOWh <= 0.0f || cosThetaIWh >= 0.0f))) {
            return result;
        }
        float D = GgxDistribution(alpha, normal, wh);
        float G = GgxG1(alpha, absCosThetaO) * GgxG1(alpha, absCosThetaI);
        float cosThetaT = 0.0f;
        float F = FresnelDielectricExact(cosThetaOWh, etaI, etaT, cosThetaT);
        float denom = eta * cosThetaOWh + cosThetaIWh;
        float denomSq = denom * denom;
        if (std::fabs(denomSq) <= 1.0e-8f) {
            return result;
        }
        float factor = (eta * eta) * std::fabs(cosThetaIWh) * std::fabs(cosThetaOWh);
        factor /= std::max(absCosThetaO * absCosThetaI * denomSq, 1.0e-6f);
        const float valueD = reference && referencePunctual ? std::min(D, 1.0f) : D;
        simd::float3 ft = simd_make_float3(1.0f - F, 1.0f - F, 1.0f - F) * valueD * G * factor;
        ft *= DielectricTransmissionTint(material, absCosThetaI);
        ft *= transmission;
        float pdfWh = reference ? D * GgxG1(alpha, absCosThetaO) * std::fabs(cosThetaOWh) / absCosThetaO
                                   : GgxVndfPdf(alpha, normal, wo, wh, /*halfVectorD=*/true);
        float dwhDwi = std::fabs(cosThetaIWh) / std::max(denomSq, 1.0e-8f);
        float pdf = pTrans * pdfWh * dwhDwi;
        if (glassInterface) {
            pdf *= 1.0f - F;
        }
        if (pdf > 0.0f) {
            result.value = simd::max(ft, simd_make_float3(0.0f, 0.0f, 0.0f));
            result.pdf = reference ? pdf : ClampSpecularPdf(pdf, clampParams);
        }
        return result;
    }

    // A regularized interface also transmits: evaluate it before the reflection-only guard.
    if (MaterialIsRegularizedDielectric(material)) {
        return EvaluateRegularizedDielectric(material, normal, wo, wi, referencePunctual);
    }

    float cosThetaO = std::max(signedCosThetaO, 0.0f);
    float cosThetaI = std::max(signedCosThetaI, 0.0f);
    if (cosThetaI <= 0.0f || cosThetaO <= 0.0f) {
        return result;
    }

    // Metal evaluates subsurface as a BSSRDF with no direct-light value (closures_eval.metal, case 5u): light reaches
    // it only through the random walk or the closure sample.
    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Subsurface)) {
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Lambertian)) {
        result.value = MaterialBaseColor(material) / kPi;
        result.pdf = LambertPdf(normal, wi);
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Plastic)) {
        ClosureValuePdf closure =
            PlasticClosureEval(material, normal, wo, wi, PlasticLobeProbabilitiesFor(material, normal, wo), clampParams);
        result.value = closure.value;
        result.pdf = closure.pdf;
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Metal)) {
        float roughness = std::clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        if (roughness <= 1.0e-3f) {
            result.isDelta = true;
            return result;
        }
        float alpha = std::max(roughness * roughness, 1.0e-4f);
        simd::float3 wh = simd::normalize(wo + wi);
        if (simd::dot(wh, normal) <= 0.0f || simd::dot(wo, wh) <= 0.0f || simd::dot(wi, wh) <= 0.0f) {
            return result;
        }

        float D = GgxDistribution(alpha, normal, wh);
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
        spec *= SpecularEnergyCompensation(f0, roughness, cosThetaO);
        spec = SanitizeSpecularValue(spec);
        float pdf = D * GgxG1(alpha, cosThetaO) / (4.0f * cosThetaO);
        if (pdf > 0.0f) {
            result.value = simd::max(spec, simd_make_float3(0.0f, 0.0f, 0.0f));
            result.pdf = ClampSpecularPdf(pdf, clampParams);
        }
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::CarPaint)) {
        simd::float3 flakeNormal = CarpaintFlakeNormal(material, position, normal);
        ClosureValuePdf closure =
            CarpaintClosureEval(material, normal, flakeNormal, wo, wi,
                                CarpaintLobeProbabilitiesFor(material, normal, flakeNormal, wo), clampParams);
        result.value = closure.value;
        result.pdf = closure.pdf;
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
    uint32_t selected = std::min(static_cast<uint32_t>(rng.nextFloat() * static_cast<float>(directionalLights.size())),
                                 static_cast<uint32_t>(directionalLights.size() - 1u));
    const DirectionalLightInfo& light = directionalLights[selected];
    const simd::float3 direction = SampleDirectionalCone(light.direction, light.cosDelta, rng);
    outSample.primitiveIndex = selected;
    outSample.flags = PathTracerShaderTypes::kLightPrimitiveFlagDirectional;
    outSample.position = direction;
    outSample.normal = -direction;
    outSample.emission = light.emission;
    outSample.pdf = providerPdf / static_cast<float>(directionalLights.size());
    return outSample.pdf > 0.0f && std::isfinite(outSample.pdf) && IsFinite(outSample.emission);
}

bool CpuResolveTexturedMeshEmission(const EmbreeSceneData& sceneData,
                                    const MeshLightInfo& light,
                                    float b1,
                                    float b2,
                                    simd::float3& emission);

// Metal's tagged_emissive_selection_pdf: the power share of a plain emitter, the cumulative-power step of a two-sided
// or textured one.
float MeshLightSelectionPdf(const std::vector<MeshLightInfo>& meshLights, size_t index) {
    const MeshLightInfo& light = meshLights[index];
    return (light.twoSided || light.texturedEmission)
        ? light.cumulativePower - (index > 0u ? meshLights[index - 1u].cumulativePower : 0.0f)
        : light.selectionPdf;
}

bool CpuSampleMeshLight(const EmbreeSceneData& sceneData,
                        const std::vector<MeshLightInfo>& meshLights,
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
    const float signedCosLight = simd::dot(-direction, normal);
    const float cosLight = light.twoSided ? std::fabs(signedCosLight) : signedCosLight;
    if (!(cosLight > 0.0f) || !std::isfinite(cosLight)) {
        return false;
    }

    const float pdfArea = 1.0f / light.area;
    const float pdfDir = pdfArea * distSq / std::max(cosLight, 1.0e-6f);
    outSample.primitiveIndex = light.primitiveIndex;
    outSample.flags = light.twoSided ? PathTracerShaderTypes::kLightPrimitiveFlagTwoSided : 0u;
    if (light.texturedEmission) {
        outSample.flags |= PathTracerShaderTypes::kLightPrimitiveFlagTexturedEmission;
    }
    outSample.position = samplePoint;
    outSample.normal = normal;
    outSample.emission = light.emission;
    if (light.texturedEmission &&
        !CpuResolveTexturedMeshEmission(sceneData, light, b1, b2, outSample.emission)) {
        return false;
    }
    const size_t index = static_cast<size_t>(selectedIt - meshLights.begin());
    outSample.pdf = pdfDir * MeshLightSelectionPdf(meshLights, index) * providerPdf;
    return outSample.pdf > 0.0f && std::isfinite(outSample.pdf);
}

// Metal's mesh_emitter_light_pdf_for_hit for one baseline draw: the solid-angle density with which CpuSampleMeshLight
// reaches this hit of a mesh emitter from `origin`. 0 for a triangle outside the emitter list (Metal's inventory, in
// ascending triangle order) or seen from a side it does not emit to.
float MeshLightPdfForHit(const std::vector<MeshLightInfo>& lights, const HitInfo& hit, const simd::float3& origin) {
    if (hit.primitiveType != GeometryType::Mesh) return 0.0f;
    const simd::float3 delta = hit.position - origin;
    const float distanceSq = simd::dot(delta, delta);
    if (!(distanceSq > 0.0f)) return 0.0f;
    auto found = std::lower_bound(lights.begin(), lights.end(), hit.primitiveIndex,
        [](const MeshLightInfo& light, uint32_t triangle) { return light.primitiveIndex < triangle; });
    if (found == lights.end() || found->primitiveIndex != hit.primitiveIndex || !(found->area > 0.0f)) return 0.0f;
    const float signedCosine = simd::dot(found->normal, -delta / std::sqrt(distanceSq));
    const float cosine = found->twoSided ? std::fabs(signedCosine) : signedCosine;
    const float selectionPdf = MeshLightSelectionPdf(lights, static_cast<size_t>(found - lights.begin()));
    return cosine > 0.0f ? std::max(selectionPdf, 0.0f) * distanceSq / (found->area * cosine) : 0.0f;
}

// Whether a mesh-light triangle faces `origin`, i.e. whether the ReSTIR DI reservoir can sample it from there: its
// target takes the light's emissive cosine (Metal's emissive_light_cosine), the front one, or either side of a
// two-sided emitter.
bool MeshLightFacesOrigin(const std::vector<MeshLightInfo>& lights, const HitInfo& hit,
                          const simd::float3& origin) {
    if (hit.primitiveType != GeometryType::Mesh) return false;
    auto found = std::lower_bound(lights.begin(), lights.end(), hit.primitiveIndex,
        [](const MeshLightInfo& light, uint32_t triangle) { return light.primitiveIndex < triangle; });
    if (found == lights.end() || found->primitiveIndex != hit.primitiveIndex || !(found->area > 0.0f)) return false;
    const float facing = simd::dot(found->normal, origin - hit.position);
    return found->twoSided ? facing != 0.0f : facing > 0.0f;
}

bool ReferencePbrTransmissionNee(const PathTracerShaderTypes::MaterialData& material) {
    return static_cast<uint32_t>(material.typeEta.x) == 7u &&
        (material.materialFlags & PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular) != 0u &&
        material.pbrExtras.z > 0.0f && material.pbrParams.x < 1.0f;
}

simd::float3 ReferenceContinuationNeeValue(const PathTracerShaderTypes::MaterialData& material,
                                           simd::float3 normal, simd::float3 wo, simd::float3 wi,
                                           simd::float3 value) {
    const bool reference = (material.materialFlags & PathTracerShaderTypes::kMaterialFlagReferenceDielectricContinuation) != 0u ||
        (static_cast<uint32_t>(material.typeEta.x) == 7u &&
         (material.materialFlags & PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular) != 0u);
    if (reference && simd::dot(normal, wo) * simd::dot(normal, wi) < 0.0f) {
        const float relativeIor = std::max(material.typeEta.y, 1.0e-3f);
        value *= relativeIor * relativeIor;
    }
    return value;
}

// Falcor neither excludes the shaded triangle from the next ray nor starts it
// below the face, so light arriving from behind the geometric face (reachable
// through a tilted shading normal) is blocked by the surface itself, as Metal.
bool FalcorFaceVisible(const PathTracerShaderTypes::MaterialData& material,
                       const HitInfo& hit,
                       simd::float3 wi) {
    if ((material.materialFlags & PathTracerShaderTypes::kMaterialFlagFalcorStandardBsdf) == 0u) {
        return true;
    }
    const simd::float3 faceNormal = hit.frontFace ? hit.normal : -hit.normal;
    return simd::dot(faceNormal, wi) > 0.0f;
}

// Metal's two_sided_alpha_cutout_direct_cos: the surface cosine of a direct-light sample. Plain NEE and, since Metal
// PR #80, the RIS and ReSTIR DI targets (direct_light_target_cos) use it.
float ReferenceDirectCos(const PathTracerShaderTypes::MaterialData& material,
                          const HitInfo& hit,
                          simd::float3 normal, simd::float3 wi) {
    const float c = simd::dot(pt_shared::source_light_cos_normal(material, normal), wi);
    if ((hit.twoSidedAlphaCutout &&
         (material.materialFlags & PathTracerShaderTypes::kMaterialFlagAlphaCutoutContinuation) == 0u) ||
        MaterialIsRegularizedDielectric(material) || pt_shared::source_rough_transmission(material) ||
        ReferencePbrTransmissionNee(material) || ThinWalledPbrTransmissionNee(material)) {
        return std::fabs(c);
    }
    return FalcorFaceVisible(material, hit, wi) ? std::max(c, 0.0f) : 0.0f;
}

// Metal's apply_two_sided_alpha_cutout_diffuse_fallback: direct light from behind a two-sided cutout, which its PBR
// lobes do not reflect, shades it as Lambertian diffuse. Returns true whenever it supplied that value, a term no BSDF
// strategy produces (a transmission lobe included; see Metal's comment): the pdf it sets only keeps the callers'
// pdf > 0 tests, and NEE's MIS weight must use a BSDF pdf of 0 there (Metal's nee_mis_bsdf_pdf, EP19).
bool ApplyTwoSidedAlphaCutoutDiffuseFallback(const PathTracerShaderTypes::MaterialData& material,
                                             const HitInfo& hit,
                                             simd::float3 normal,
                                             simd::float3 wi,
                                             BsdfEval& eval) {
    if ((material.materialFlags & (PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular |
                                   PathTracerShaderTypes::kMaterialFlagAlphaCutoutContinuation)) != 0u ||
        !hit.twoSidedAlphaCutout || simd::dot(normal, wi) >= 0.0f ||
        eval.value.x > 0.0f || eval.value.y > 0.0f || eval.value.z > 0.0f) {
        return false;
    }
    // Metal's weight: the PBR diffuse lobe's (1 - metallic) and (1 - transmission), so a fully transmissive cutout
    // gets no fallback.
    const float metallic = std::clamp(material.pbrParams.x, 0.0f, 1.0f);
    const float transmission = std::clamp(material.pbrExtras.z, 0.0f, 1.0f) * (1.0f - metallic);
    const simd::float3 diffuseColor = MaterialBaseColor(material) * (1.0f - metallic) * (1.0f - transmission) *
        std::clamp(material.pbrParams.z, 0.0f, 1.0f);
    eval.value = diffuseColor / kPi;
    eval.pdf = std::max(std::fabs(simd::dot(normal, wi)) / kPi, 1.0e-6f);
    eval.isDelta = false;
    return true;
}

// Metal's nee_mis_bsdf_pdf: the BSDF pdf NEE's MIS weight uses.
float NeeMisBsdfPdf(const BsdfEval& eval, bool cutoutBackLightUnsampled) {
    return cutoutBackLightUnsampled ? 0.0f : eval.pdf;
}

// Metal's direct_light_target_bsdf (direct_light_target.metal): the material term of the RIS and ReSTIR DI targets and
// of the winner's shading is plain NEE's, with its Lambertian fallback for a two-sided cutout lit from behind and its
// eta-free continuation value on area samples.
BsdfEval DirectLightTargetBsdf(const PathTracerShaderTypes::MaterialData& material,
                               const HitInfo& hit,
                               simd::float3 normal,
                               simd::float3 wo,
                               simd::float3 wi,
                               const FireflyClampParams& clampParams,
                               bool directional,
                               bool* outCutoutBackLightUnsampled = nullptr) {
    BsdfEval eval = EvaluateBsdf(material, hit.position, normal, wo, wi, clampParams, directional);
    const bool cutoutBackLightUnsampled = ApplyTwoSidedAlphaCutoutDiffuseFallback(material, hit, normal, wi, eval);
    if (outCutoutBackLightUnsampled) {
        *outCutoutBackLightUnsampled = cutoutBackLightUnsampled;
    }
    if (!directional) {
        eval.value = ReferenceContinuationNeeValue(material, normal, wo, wi, eval.value);
    }
    return eval;
}

// Metal's emissive_light_cosine, clamped as the RIS and ReSTIR DI targets clamp it: a two-sided emitter lights both
// of its sides.
float CpuRisPayloadLightCosine(const CpuRisSamplePayload& sample, simd::float3 omegaL) {
    if (CpuRisPayloadIsDirectional(sample)) {
        return 1.0f;
    }
    const float c = simd::dot(sample.normal, -omegaL);
    return std::clamp((sample.flags & PathTracerShaderTypes::kLightPrimitiveFlagTwoSided) != 0u ? std::fabs(c) : c,
                      0.0f, 1.0f);
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
    float cosThetaSurface = ReferenceDirectCos(material, hit, shadingNormal, omegaL);
    float cosThetaLight = CpuRisPayloadLightCosine(winner, omegaL);
    BsdfEval bsdfEval = DirectLightTargetBsdf(material, hit, shadingNormal, wo, omegaL, clampParams,
        CpuRisPayloadIsDirectional(winner));
    if (bsdfEval.isDelta) {
        return 0.0f;
    }
    return CpuPHat(winner.emission, bsdfEval.value, cosThetaSurface, cosThetaLight, distSq);
}

bool CpuBuildRestirDiProviderReservoirForHit(
    const EmbreeSceneData& sceneData,
    const std::vector<DirectionalLightInfo>& directionalLights,
    const std::vector<MeshLightInfo>& meshLights,
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
    const uint32_t providerCount = (hasDirectionalProvider ? 1u : 0u) + (hasMeshProvider ? 1u : 0u);
    if (providerCount == 0u) {
        return false;
    }
    const float providerPdf = 1.0f / static_cast<float>(providerCount);
    const uint32_t risTargetM = std::max(candidateCount, 1u);
    for (uint32_t i = 0; i < risTargetM; ++i) {
        uint32_t selected = std::min(static_cast<uint32_t>(rng.nextFloat() * static_cast<float>(providerCount)),
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
            sampled = CpuSampleMeshLight(sceneData, meshLights, hit, providerPdf, rng, candidate);
        }
        if (!sampled) {
            continue;
        }
        float distSq = CpuRisPayloadDistanceSq(hit, candidate);
        if (!(distSq > 0.0f) || !std::isfinite(distSq)) {
            continue;
        }
        simd::float3 omegaL = CpuRisPayloadOmegaL(hit, candidate);
        float cosThetaSurface = ReferenceDirectCos(material, hit, shadingNormal, omegaL);
        float cosThetaLight = CpuRisPayloadLightCosine(candidate, omegaL);
        BsdfEval bsdfEval = DirectLightTargetBsdf(material, hit, shadingNormal, wo, omegaL, clampParams,
            CpuRisPayloadIsDirectional(candidate));
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
    // Metal's provider builder (restir_di.metal): RIS normalizes by every candidate drawn. CpuUpdateReservoir counts
    // only the accepted ones, which inflated W whenever a draw failed or had zero target weight.
    if (reservoir.valid) {
        reservoir.M = risTargetM;
    }
    return reservoir.valid && reservoir.M > 0u && reservoir.wSum > 0.0f;
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
           !sample.isBssrdf &&
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
           !sample.isBssrdf &&
           sample.lobeType == 0u &&
           IsFinite(sample.weight);
}

// Diagnostics only: the reason CpuRadianceCacheCanQuery refused, in radiance_cache.metal's codes
// (2 = material or depth, 3 = sample), so Embree and Metal report fallbacks the same way.
uint32_t CpuRadianceCacheCannotQueryReason(const PathTracer::RenderSettings& settings,
                                           const PathTracerShaderTypes::MaterialData& material,
                                           uint32_t depth) {
    if (depth < std::max(settings.radianceCacheMinDepth, 1u) || !CpuDiffuseRestirEligibleMaterial(material)) {
        return 2u;
    }
    return 3u;
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
           !sample.isBssrdf &&
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
    if (pt_shared::source_model(material) > 0.0f) {
        return SampleSourceClosure(material, normal, wo, frontFace, rng);
    }
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
        // Subsurface: Metal's closure sample (bsdf_sample.metal, case 5u) is this Lambert lobe with the base colour
        // unless its separable diffusion runs (sss=separable), which Embree does not port and reports.
        simd::float3 f = MaterialBaseColor(material) / kPi;
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
        const bool reference = (material.materialFlags & PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular) != 0u;
        const float relativeIor = std::max(material.typeEta.y, 1.0e-3f);
        float dielectricF0 = DielectricF0FromIor(reference ? std::max(relativeIor, 1.0f / relativeIor) : material.typeEta.y);
        simd::float3 f0 = baseColor * metallic +
                          simd_make_float3(dielectricF0, dielectricF0, dielectricF0) * (1.0f - metallic);
        simd::float3 diffuseColor = baseColor * (1.0f - metallic) *
                                    std::clamp(material.pbrParams.z, 0.0f, 1.0f);
        float transmission = std::clamp(material.pbrExtras.z, 0.0f, 1.0f) * (1.0f - metallic);
        const bool glassInterface = !reference && transmission > 0.0f;
        const bool thinWalled = glassInterface && ThinWalledPbrTransmission(material);
        float reflectScale = 1.0f - transmission;
        float specWeightBase = PbrSpecularWeight(f0);
        if ((material.materialFlags & PathTracerShaderTypes::kMaterialFlagFalcorStandardBsdf) != 0u) {
            // Falcor picks lobes by albedo: Fresnel-weighted specular against diffuse, as Metal.
            const float specAlbedo = Luminance(SchlickFresnel(f0, std::clamp(simd::dot(normal, wo), 0.0f, 1.0f)));
            specWeightBase = std::clamp(specAlbedo / std::max(specAlbedo + Luminance(diffuseColor), 1.0e-6f), 0.05f, 0.95f);
        }
        const float specularScale = reference ? 1.0f : reflectScale;
        float wSpec = specWeightBase * specularScale;
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
                // A delta BRDF carries 1/cos(theta_i), cancelling the tail's cosine.
                f = SchlickFresnel(f0, cosThetaO) * specularScale / cosThetaI;
                pdfSpec = 1.0f;
                result.isDelta = true;
            } else {
                float alpha = (reference ||
                               (material.materialFlags & PathTracerShaderTypes::kMaterialFlagFalcorStandardBsdf) != 0u)
                    ? roughness * roughness : std::max(roughness * roughness, 1.0e-4f);
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
                float D = GgxDistribution(alpha, normal, wh);
                float G = GgxG1(alpha, cosThetaO) * GgxG1(alpha, cosThetaI);
                simd::float3 F = SchlickFresnel(f0, simd::dot(wi, wh));
                float denom = 4.0f * cosThetaO * cosThetaI;
                f = F * (D * G / std::max(denom, 1.0e-6f));
                // No compensation for Falcor-standard materials, as in the evaluation above and on Metal.
                if ((material.materialFlags & (PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular |
                                               PathTracerShaderTypes::kMaterialFlagFalcorStandardBsdf)) == 0u) {
                    f *= SpecularEnergyCompensation(f0, roughness, cosThetaO);
                }
                f = SanitizeSpecularValue(f);
                f *= specularScale;
                // Falcor samples the VNDF, so its density is the VNDF pdf over the
                // reflection Jacobian 4*(wo.wh), as Metal.
                const float woDotWh = simd::dot(wo, wh);
                pdfSpec = reference
                    ? GgxPdf(alpha, normal, wo, wi, /*halfVectorD=*/true)
                    : (material.materialFlags & PathTracerShaderTypes::kMaterialFlagFalcorStandardBsdf) != 0u
                        ? (woDotWh > 0.0f ? GgxVndfPdf(alpha, normal, wo, wh, /*halfVectorD=*/true) / (4.0f * woDotWh) : 0.0f)
                        : D * GgxG1(alpha, cosThetaO) / (4.0f * cosThetaO);
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
            if ((material.materialFlags & PathTracerShaderTypes::kMaterialFlagFalcorStandardBsdf) != 0u) {
                f *= FrostbiteDiffuseScale(normal, wo, wi, roughness);
            }
        } else {
            sampledLobeType = 2u;
            sampledLobeRoughness = roughness;
            float cosThetaO = simd::dot(normal, wo);
            float absCosThetaO = std::fabs(cosThetaO);
            float etaI = 1.0f;
            float etaT = reference ? relativeIor : std::max(material.typeEta.y, 1.0f);
            // `normal` always faces wo, so only the hit's face tells an exit.
            // A thin-walled sheet is entered from outside on either face.
            if (!frontFace && !reference && !thinWalled) {
                std::swap(etaI, etaT);
            }
            float eta = etaI / etaT;
            if (glassInterface) {
                const bool deltaGlass = roughness <= 1.0e-3f;
                const simd::float3 wh = deltaGlass ? normal : SampleGgxVndf(rng, roughness, normal, wo);
                const float woDotWh = simd::dot(wo, wh);
                float cosThetaT = 0.0f;
                const float F = FresnelDielectricExact(woDotWh, etaI, etaT, cosThetaT);
                const bool reflectGlass = F >= 1.0f || rng.nextFloat() < F;
                if (reflectGlass) {
                    wi = Reflect(-wo, wh);
                } else if (thinWalled) {
                    // Straight through (eta = 1); a rough sheet mirrors the
                    // reflected microfacet direction through the tangent plane.
                    const simd::float3 reflected = Reflect(-wo, wh);
                    wi = deltaGlass ? -wo : reflected - 2.0f * simd::dot(reflected, normal) * normal;
                } else if (!Refract(-wo, wh, eta, wi)) {
                    return result;
                }
                wi = simd::normalize(wi);
                const float signedCosI = simd::dot(normal, wi);
                const float absCosI = std::fabs(signedCosI);
                if (absCosI <= 0.0f || (reflectGlass ? signedCosI <= 0.0f : signedCosI >= 0.0f)) {
                    return result;
                }
                sampledLobeType = reflectGlass ? 1u : 2u;
                result.isDelta = deltaGlass;
                if (deltaGlass) {
                    const bool etaFree = (material.materialFlags &
                        PathTracerShaderTypes::kMaterialFlagReferenceDielectricContinuation) != 0u;
                    const float etaScale = reflectGlass || etaFree || thinWalled ? 1.0f : eta * eta;
                    f = simd_make_float3(1.0f, 1.0f, 1.0f) * (transmission * (reflectGlass ? F : 1.0f - F) * etaScale / absCosI);
                    if (!reflectGlass) {
                        f *= DielectricTransmissionTint(material, absCosI);
                    }
                    pdfTrans = reflectGlass ? F : 1.0f - F;
                } else {
                    const float alpha = std::max(roughness * roughness, 1.0e-4f);
                    const float D = GgxDistribution(alpha, normal, wh);
                    const float G = GgxG1(alpha, absCosThetaO) * GgxG1(alpha, absCosI);
                    const float pdfWh = GgxVndfPdf(alpha, normal, wo, wh, /*halfVectorD=*/true);
                    if (reflectGlass) {
                        f = simd_make_float3(1.0f, 1.0f, 1.0f) * (transmission * F * D * G /
                            std::max(4.0f * absCosThetaO * absCosI, 1.0e-6f));
                        pdfTrans = F * pdfWh / (4.0f * std::max(woDotWh, 1.0e-6f));
                    } else if (thinWalled) {
                        // The mirrored reflection keeps the reflection Jacobian 1/(4 wo.wh).
                        f = simd_make_float3(1.0f, 1.0f, 1.0f) * (transmission * (1.0f - F) * D * G /
                            std::max(4.0f * absCosThetaO * absCosI, 1.0e-6f)) *
                            DielectricTransmissionTint(material, absCosI);
                        pdfTrans = (1.0f - F) * pdfWh / (4.0f * std::max(woDotWh, 1.0e-6f));
                    } else {
                        const float wiDotWh = simd::dot(wi, wh);
                        const float denom = eta * woDotWh + wiDotWh;
                        const float denomSq = denom * denom;
                        if (denomSq <= 1.0e-8f) {
                            return result;
                        }
                        const float factor = eta * eta * std::fabs(woDotWh * wiDotWh) /
                            std::max(absCosThetaO * absCosI * denomSq, 1.0e-6f);
                        f = simd_make_float3(1.0f, 1.0f, 1.0f) * (transmission * (1.0f - F) * D * G * factor) *
                            DielectricTransmissionTint(material, absCosI);
                        pdfTrans = (1.0f - F) * pdfWh * std::fabs(wiDotWh) / denomSq;
                    }
                }
            } else if (roughness <= 1.0e-3f) {
                if (!Refract(-wo, normal, eta, wi)) {
                    return result;
                }
                wi = simd::normalize(wi);
                float cosThetaI = simd::dot(normal, wi);
                float cosThetaT = 0.0f;
                float Fr = FresnelDielectricExact(cosThetaO, etaI, etaT, cosThetaT);
                // Match Metal: physical (etaI/etaT)^2 radiance unless the material
                // keeps the reference's eta-free continuation.
                const bool etaFreeContinuation = reference ||
                    (material.materialFlags &
                     PathTracerShaderTypes::kMaterialFlagReferenceDielectricContinuation) != 0u;
                float etaScale = etaFreeContinuation ? 1.0f : eta * eta;
                // A delta BTDF carries 1/|cos(theta_i)|, cancelling the tail's cosine.
                float ftScale = std::max(1.0f - Fr, 0.0f) * etaScale /
                                std::max(std::fabs(cosThetaI), 1.0e-6f);
                simd::float3 ft = simd_make_float3(ftScale, ftScale, ftScale);
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
                if (cosThetaOWh * cosThetaIWh > 0.0f ||
            (reference && (cosThetaOWh <= 0.0f || cosThetaIWh >= 0.0f))) {
                    return result;
                }
                float alpha = std::max(roughness * roughness, 1.0e-4f);
                float D = GgxDistribution(alpha, normal, wh);
                float G = GgxG1(alpha, absCosThetaO) * GgxG1(alpha, absCosThetaI);
                float cosThetaT = 0.0f;
                float F = FresnelDielectricExact(cosThetaOWh, etaI, etaT, cosThetaT);
                float denom = eta * cosThetaOWh + cosThetaIWh;
                float denomSq = denom * denom;
                if (std::fabs(denomSq) <= 1.0e-8f) {
                    return result;
                }
                float factor = (eta * eta) * std::fabs(cosThetaIWh) * std::fabs(cosThetaOWh);
                factor /= std::max(absCosThetaO * absCosThetaI * denomSq, 1.0e-6f);
                simd::float3 ft = simd_make_float3(1.0f - F, 1.0f - F, 1.0f - F) * D * G * factor;
                ft *= DielectricTransmissionTint(material, absCosThetaI);
                f = transmission * ft;
                float pdfWh = reference ? D * GgxG1(alpha, absCosThetaO) * std::fabs(cosThetaOWh) / absCosThetaO
                                   : GgxVndfPdf(alpha, normal, wo, wh, /*halfVectorD=*/true);
                float dwhDwi = std::fabs(cosThetaIWh) / std::max(denomSq, 1.0e-8f);
                pdfTrans = pdfWh * dwhDwi;
            }
        }

        float cosThetaI = std::fabs(simd::dot(normal, wi));
        if (glassInterface && result.isDelta && simd::dot(normal, wi) > 0.0f) {
            const bool entering = frontFace || thinWalled;
            const float etaI = entering ? 1.0f : std::max(material.typeEta.y, 1.0f);
            const float etaT = entering ? std::max(material.typeEta.y, 1.0f) : 1.0f;
            float cosThetaT = 0.0f;
            const float F = FresnelDielectricExact(simd::dot(normal, wo), etaI, etaT, cosThetaT);
            // Opaque and glass mirrors share the same discrete direction.
            f = ((1.0f - transmission) * SchlickFresnel(f0, simd::dot(normal, wo)) +
                 simd_make_float3(1.0f, 1.0f, 1.0f) * (transmission * F)) / cosThetaI;
            pdfSpec = 1.0f;
            pdfTrans = F;
        }
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
        if ((material.materialFlags & PathTracerShaderTypes::kMaterialFlagFalcorStandardBsdf) != 0u &&
            !result.isDelta) {
            // Falcor keeps the selected lobe's throughput, but reports the whole
            // material's marginal direction density for light/BSDF MIS, as Metal.
            result.pdf = EvaluateBsdf(material, position, normal, wo, wi, clampParams).pdf;
        }
        if (reference && !result.isDelta) {
            const BsdfEval eval = EvaluateBsdf(material, position, normal, wo, wi, clampParams);
            result.pdf = eval.pdf;
            result.weight = eval.pdf > 0.0f ? eval.value * cosThetaI / eval.pdf : simd_make_float3(0.0f);
            if (simd::dot(normal, wi) * simd::dot(normal, wo) < 0.0f) {
                result.weight *= relativeIor * relativeIor;
            }
        }
        if (!result.isDelta && roughness > 1.0e-3f && sampledLobeType < 2u &&
            simd::dot(normal, wi) * simd::dot(normal, wo) > 0.0f &&
            (material.materialFlags & (PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular |
                                       PathTracerShaderTypes::kMaterialFlagFalcorStandardBsdf)) == 0u) {
            // Export the mixture value and PDF so the BSDF-hit MIS weight uses the
            // same density as environment NEE (Metal's sample_pbr_metallic_roughness does the same).
            const BsdfEval eval = EvaluateBsdf(material, position, normal, wo, wi, clampParams);
            result.pdf = eval.pdf;
            result.weight = eval.pdf > 0.0f
                ? simd::max(eval.value * cosThetaI / eval.pdf, simd_make_float3(0.0f))
                : simd_make_float3(0.0f);
        }
        if (glassInterface && !result.isDelta && simd::dot(normal, wi) > 0.0f) {
            // Use the actual exit eta for the sampler's overlapping reflection proposals.
            const BsdfEval eval = EvaluateBsdf(material, position, normal, wo, wi, clampParams, false, frontFace);
            result.pdf = eval.pdf;
            result.weight = eval.pdf > 0.0f
                ? simd::max(eval.value * cosThetaI / eval.pdf, simd_make_float3(0.0f))
                : simd_make_float3(0.0f);
        }
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Plastic)) {
        PlasticLobeProbabilities probs = PlasticLobeProbabilitiesFor(material, normal, wo);
        if (probs.coat + probs.diffuse <= 0.0f) {
            return result;
        }
        float coatRoughness = PlasticCoatRoughness(material);
        bool sampledPlasticCoat = rng.nextFloat() < probs.coat;
        simd::float3 wi;
        if (sampledPlasticCoat) {
            simd::float3 wh = SampleGgxVndf(rng, coatRoughness, normal, wo);
            if (simd::dot(wh, normal) <= 0.0f) {
                return result;
            }
            wi = simd::normalize(Reflect(-wo, wh));
        } else {
            float cosinePdf = 0.0f;
            wi = SampleCosineHemisphere(rng, normal, cosinePdf);
        }
        float cosThetaI = simd::dot(normal, wi);
        if (!IsFinite(wi) || cosThetaI <= 0.0f) {
            return result;
        }

        ClosureValuePdf closure = PlasticClosureEval(material, normal, wo, wi, probs, clampParams);
        if (!(closure.pdf > 0.0f) ||
            !(closure.value.x > 0.0f || closure.value.y > 0.0f || closure.value.z > 0.0f)) {
            return result;
        }
        simd::float3 weight = closure.value * cosThetaI / closure.pdf;
        if (!IsFinite(weight)) {
            return result;
        }
        result.direction = wi;
        result.weight = simd::max(weight, simd_make_float3(0.0f, 0.0f, 0.0f));
        result.pdf = closure.pdf;
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

        float alpha = std::max(roughness * roughness, 1.0e-4f);
        simd::float3 wh = SampleGgxVndf(rng, roughness, normal, wo);
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
        float D = GgxDistribution(alpha, normal, wh);
        float G = GgxG1(alpha, cosThetaO) * GgxG1(alpha, cosThetaI);
        float denom = 4.0f * cosThetaO * cosThetaI;
        simd::float3 f = F * (D * G / std::max(denom, 1.0e-6f));
        f *= SpecularEnergyCompensation(f0, roughness, cosThetaO);
        f = SanitizeSpecularValue(f);
        float pdf = D * GgxG1(alpha, cosThetaO) / (4.0f * cosThetaO);
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
        if (simd::dot(normal, wo) <= 0.0f) {
            return result;
        }
        simd::float3 flakeNormal = CarpaintFlakeNormal(material, position, normal);
        CarpaintLobeProbabilities probs = CarpaintLobeProbabilitiesFor(material, normal, flakeNormal, wo);
        if (probs.coat + probs.flake + probs.diffuse + probs.metal <= 0.0f) {
            return result;
        }
        float u = rng.nextFloat();
        simd::float3 lobeNormal = normal;
        float lobeRoughness = 1.0f;
        bool diffuseLobe = false;
        if (u < probs.coat) {
            lobeRoughness = PlasticCoatRoughness(material);
        } else if (u < probs.coat + probs.flake) {
            lobeNormal = flakeNormal;
            lobeRoughness = CarpaintFlakeLobeRoughness(material);
        } else if (u < probs.coat + probs.flake + probs.diffuse || probs.metal <= 0.0f) {
            diffuseLobe = true;
        } else {
            lobeRoughness = CarpaintMetalRoughness(material);
        }

        simd::float3 wi;
        if (diffuseLobe) {
            float cosinePdf = 0.0f;
            wi = SampleCosineHemisphere(rng, normal, cosinePdf);
        } else {
            simd::float3 wh = SampleGgxVndf(rng, lobeRoughness, lobeNormal, wo);
            if (simd::dot(wh, lobeNormal) <= 0.0f) {
                return result;
            }
            wi = simd::normalize(Reflect(-wo, wh));
        }
        float cosThetaI = simd::dot(normal, wi);
        if (!IsFinite(wi) || cosThetaI <= 0.0f) {
            return result;
        }

        ClosureValuePdf closure = CarpaintClosureEval(material, normal, flakeNormal, wo, wi, probs, clampParams);
        if (!(closure.pdf > 0.0f) ||
            !(closure.value.x > 0.0f || closure.value.y > 0.0f || closure.value.z > 0.0f)) {
            return result;
        }
        simd::float3 weight = closure.value * cosThetaI / closure.pdf;
        if (!IsFinite(weight)) {
            return result;
        }
        result.direction = wi;
        result.weight = simd::max(weight, simd_make_float3(0.0f, 0.0f, 0.0f));
        result.pdf = closure.pdf;
        result.lobeType = diffuseLobe ? 0u : 1u;
        result.lobeRoughness = lobeRoughness;
        return result;
    }

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Dielectric)) {
        if (MaterialIsRegularizedDielectric(material)) {
            return SampleRegularizedDielectric(material, normal, wo, frontFace, rng);
        }
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
        if (isThin && Fr < 1.0f) {
            float Ft = 1.0f - Fr;
            Fr += (Ft * Ft * Fr) / std::max(1.0f - Fr * Fr, 1.0e-6f);
        }
        simd::float3 transmissionTint =
            (material.materialFlags & PathTracerShaderTypes::kMaterialFlagDielectricTransmissionTint) != 0u
                ? simd::clamp(MaterialBaseColor(material),
                              simd_make_float3(0.0f, 0.0f, 0.0f),
                              simd_make_float3(1.0f, 1.0f, 1.0f))
                : simd_make_float3(1.0f, 1.0f, 1.0f);

        simd::float3 direction;
        simd::float3 weight;
        if (rng.nextFloat() < Fr) {
            direction = Reflect(incidentDir, normal);
            weight = simd_make_float3(1.0f, 1.0f, 1.0f);
        } else if (isThin) {
            direction = incidentDir;
            weight = transmissionTint;
        } else {
            simd::float3 refracted;
            if (!Refract(incidentDir, normal, relativeEta, refracted) ||
                simd::dot(refracted, refracted) <= 0.0f) {
                direction = Reflect(incidentDir, normal);
                weight = simd_make_float3(1.0f, 1.0f, 1.0f);
            } else {
                direction = simd::normalize(refracted);
                result.mediumEvent = frontFace ? 1 : -1;
                // Match Metal, including the Pavilion-only reference convention.
                const bool referenceContinuation =
                    (material.materialFlags &
                     PathTracerShaderTypes::kMaterialFlagReferenceDielectricContinuation) != 0u;
                float scale = referenceContinuation ? 1.0f : relativeEta * relativeEta;
                weight = transmissionTint * scale;
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
                  rgbe.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(y) * width * 4u));
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
    return true;
}

// Loads the environment texels only; LoadEnvironmentMap builds the importance
// distribution from them.
bool LoadEnvironmentTexels(const std::string& path, EnvironmentMap& outEnv) {
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

    // MTKTextureLoader stores Radiance HDR as half float, so a sun brighter than
    // 65504 becomes inf and drops out of lighting. Re-decode such maps exactly,
    // as the Metal backend does (SceneResources::reloadEnvironmentIfNeeded).
    const bool halfOverflow = std::any_of(outEnv.rgba.begin(), outEnv.rgba.end(),
                                          [](float v) { return !std::isfinite(v); });
    if (halfOverflow) {
        EnvironmentMap exactEnv;
        if (LoadRadianceHdrCpu(path, exactEnv) && exactEnv.width == outEnv.width &&
            exactEnv.height == outEnv.height) {
            std::cerr << "[EnvRebuild] half-float overflow; reloaded " << path << " as RGBA32Float" << std::endl;
            outEnv = std::move(exactEnv);
            return true;
        }
        std::cerr << "[EnvRebuild] WARNING: " << path << " overflows half float and could not be re-decoded; "
                  << "texels above 65504 contribute no light" << std::endl;
    }

    return true;
}

// environmentSunClamp: scales every texel whose luminance exceeds the clamp down
// to it, before the importance distribution is built (as the Metal
// backend does in SceneResources::reloadEnvironmentIfNeeded).
void ApplyEnvironmentSunClamp(EnvironmentMap& env, float sunClampLuminance, const std::string& path) {
    const float sunClamp = std::isfinite(sunClampLuminance) ? std::max(sunClampLuminance, 0.0f) : 0.0f;
    if (sunClamp <= 0.0f) {
        return;
    }
    size_t clampedTexels = 0;
    for (size_t i = 0; i + 3 < env.rgba.size(); i += 4) {
        const float luminance = 0.2126f * env.rgba[i] + 0.7152f * env.rgba[i + 1] + 0.0722f * env.rgba[i + 2];
        if (luminance > sunClamp) {
            const float scale = sunClamp / luminance;
            env.rgba[i] *= scale;
            env.rgba[i + 1] *= scale;
            env.rgba[i + 2] *= scale;
            ++clampedTexels;
        }
    }
    if (clampedTexels > 0) {
        std::cerr << "[EnvRebuild] envSunClamp=" << sunClamp << " scaled " << clampedTexels << " texels of " << path
                  << std::endl;
    }
}

bool LoadEnvironmentMap(const std::string& path, float sunClampLuminance, EnvironmentMap& outEnv) {
    if (!LoadEnvironmentTexels(path, outEnv)) {
        return false;
    }
    ApplyEnvironmentSunClamp(outEnv, sunClampLuminance, path);
    BuildEnvironmentDistribution(outEnv);
    return true;
}

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

simd::float4 SampleTexturedEmitterCpuTexture(const CpuTexture& texture, simd::float2 uv) {
    if (texture.width == 0u || texture.height == 0u || texture.mipLevels.empty()) {
        return simd_make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
    uv = simd_make_float2(ApplyWrapMode(uv.x, texture.samplerDesc.wrapS),
                           ApplyWrapMode(uv.y, texture.samplerDesc.wrapT));
    auto address = [](int coordinate, uint32_t size, int32_t wrap) {
        const int dimension = static_cast<int>(size);
        if (wrap != 10497) return std::clamp(coordinate, 0, dimension - 1);
        const int index = coordinate % dimension;
        return index < 0 ? index + dimension : index;
    };
    const CpuTexture::MipLevel& level0 = texture.mipLevels.front();
    auto fetch = [&](int x, int y) {
        const size_t index = (static_cast<size_t>(address(y, level0.height, texture.samplerDesc.wrapT)) *
                             level0.width + static_cast<size_t>(address(x, level0.width, texture.samplerDesc.wrapS))) * 4u;
        return CpuTextureTexel(texture, level0, index);
    };
    if (texture.samplerDesc.magFilter == 9728) {
        return fetch(static_cast<int>(std::floor(uv.x * static_cast<float>(texture.width))),
                     static_cast<int>(std::floor(uv.y * static_cast<float>(texture.height))));
    }
    const float x = uv.x * static_cast<float>(texture.width) - 0.5f;
    const float y = uv.y * static_cast<float>(texture.height) - 0.5f;
    const int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y));
    const float tx = x - static_cast<float>(x0), ty = y - static_cast<float>(y0);
    return (fetch(x0, y0) * (1.0f - tx) + fetch(x0 + 1, y0) * tx) * (1.0f - ty) +
           (fetch(x0, y0 + 1) * (1.0f - tx) + fetch(x0 + 1, y0 + 1) * tx) * ty;
}

bool CpuResolveTexturedMeshEmission(const EmbreeSceneData& sceneData,
                                    const MeshLightInfo& light,
                                    float b1,
                                    float b2,
                                    simd::float3& emission) {
    if (!light.texturedMesh || !light.texturedMaterial) return false;
    const auto& mesh = *light.texturedMesh;
    const auto& material = *light.texturedMaterial;
    const size_t base = static_cast<size_t>(light.localTriangle) * 3u;
    if (base + 2u >= mesh.indices.size()) return false;
    const uint32_t i0 = mesh.indices[base], i1 = mesh.indices[base + 1u], i2 = mesh.indices[base + 2u];
    if (i0 >= mesh.vertices.size() || i1 >= mesh.vertices.size() || i2 >= mesh.vertices.size()) return false;
    const float b0 = 1.0f - b1 - b2;
    const simd::float2 uv0 = mesh.vertices[i0].uv * b0 + mesh.vertices[i1].uv * b1 + mesh.vertices[i2].uv * b2;
    const simd::float2 uv1 = mesh.vertices[i0].uv1 * b0 + mesh.vertices[i1].uv1 * b1 + mesh.vertices[i2].uv1 * b2;
    auto emissive = sceneData.materialTextures.find(material.textureIndices1.x);
    if (emissive == sceneData.materialTextures.end()) return false;
    const simd::float2 uv = ApplyTextureTransform(material.textureUvSet1.x == 1u ? uv1 : uv0,
                                                  material.textureTransform8, material.textureTransform9);
    const simd::float4 color = SampleTexturedEmitterCpuTexture(emissive->second, uv);
    float alpha = std::clamp(material.pbrExtras.x, 0.0f, 1.0f);
    if (material.pbrExtras.w > 0.5f) {
        auto baseColor = sceneData.materialTextures.find(material.textureIndices0.x);
        if (baseColor != sceneData.materialTextures.end()) {
            const simd::float2 baseUv = ApplyTextureTransform(material.textureUvSet0.x == 1u ? uv1 : uv0,
                                                               material.textureTransform0, material.textureTransform1);
            alpha *= SampleTexturedEmitterCpuTexture(baseColor->second, baseUv).w;
        }
    }
    if (material.pbrExtras.w > 0.5f && material.pbrExtras.w < 1.5f && alpha < material.pbrExtras.y) return false;
    const float coverage = material.pbrExtras.w >= 1.5f ? alpha : 1.0f;
    emission *= simd_make_float3(color.x, color.y, color.z) * coverage;
    return IsFinite(emission) && (emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f);
}

PathTracerShaderTypes::MaterialData ShadeMaterialForHit(const EmbreeSceneData& sceneData,
                                                        const PathTracerShaderTypes::MaterialData& material,
                                                        const HitInfo& hit,
                                                        float debugNormalStrengthScale,
                                                        float surfaceFootprintWorld,
                                                        bool normalVarianceRoughness,
                                                        float textureLodBias,
                                                        bool disableOrmTexture,
                                                        float* debugMetallic,
                                                        float* debugRoughness,
                                                        float* debugAO) {
    PathTracerShaderTypes::MaterialData shaded = material;
    const uint32_t type = static_cast<uint32_t>(shaded.typeEta.x);
    if (debugMetallic) *debugMetallic =
        type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness) &&
        hit.primitiveType == GeometryType::Mesh ? std::clamp(shaded.pbrParams.x, 0.0f, 1.0f) : 0.0f;
    if (debugRoughness) *debugRoughness = std::clamp(shaded.baseColorRoughness.w, 0.0f, 1.0f);
    if (debugAO) *debugAO = 1.0f;
    if (!hit.hasUv || sceneData.materialTextures.empty()) {
        if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness) &&
            (shaded.pbrParams.y < 0.0f || shaded.pbrParams.y > 1.0f)) {
            shaded.pbrParams.y = std::clamp(shaded.pbrParams.y, 0.0f, 1.0f);
            shaded.baseColorRoughness.w = shaded.pbrParams.y;
        }
        return shaded;
    }

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
                                         surfaceFootprintWorld,
                                         textureLodBias);
        }
        simd::float4 sample = (gradientFiltered && ctx.hasIgehyGradients)
            ? SampleCpuTextureAnisotropic(found->second, ctx.uv, ctx.dUVdx, ctx.dUVdy, lod, textureLodBias)
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
    // Metal's path_cutout_alpha: a PBR MASK cutout or BLEND coverage test reads the hit point's own alpha at mip 0
    // (bilinear, no gradients), as the shadow test does. The filtered alpha at the path footprint's LOD decides whole
    // texture regions at once, and a BLEND pass is a MIS null event whose light side NEE takes through the shadow
    // test's mip-0 alpha.
    float cutoutAlpha = baseColorSample.w;
    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness) &&
        shaded.pbrExtras.w > 0.5f) {
        cutoutAlpha = sampleTexture(shaded.textureIndices0.x,
                                    shaded.textureUvSet0.x,
                                    shaded.textureTransform0,
                                    shaded.textureTransform1,
                                    /*forceMip0=*/true,
                                    (shaded.materialFlags & PathTracerShaderTypes::kMaterialFlagBlackKeyAlphaFromRgb) != 0u,
                                    /*gradientFiltered=*/false).w;
    }
    shaded.pbrExtras.x = std::clamp(shaded.pbrExtras.x * cutoutAlpha, 0.0f, 1.0f);

    if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
        bool ormDisabled = (shaded.materialFlags & PathTracerShaderTypes::kMaterialFlagDisableOrm) != 0u;
        if (!ormDisabled && !disableOrmTexture) {
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
        shaded.pbrParams.y = std::clamp(shaded.pbrParams.y, 0.0f, 1.0f);
        if (debugMetallic) *debugMetallic = shaded.pbrParams.x;
        if (debugRoughness) *debugRoughness = shaded.pbrParams.y;
        if ((shaded.materialFlags & PathTracerShaderTypes::kMaterialFlagFalcorStandardBsdf) != 0u) {
            // Falcor's StandardMaterial shades the authored roughness unfiltered and
            // floors GGX alpha at 0.0064 (never delta), as Metal.
            shaded.pbrParams.y = std::max(std::clamp(shaded.pbrParams.y, 0.0f, 1.0f), 0.08f);
        } else if (normalVarianceRoughness) {
            shaded.pbrParams.y = ApplyNormalVarianceToRoughness(sceneData,
                                                                shaded,
                                                                hit,
                                                                debugNormalStrengthScale,
                                                                surfaceFootprintWorld,
                                                                textureLodBias);
        }
        shaded.baseColorRoughness.w = shaded.pbrParams.y;
        simd::float4 occlusionSample = sampleTexture(shaded.textureIndices0.w,
                                                     shaded.textureUvSet0.w,
                                                     shaded.textureTransform6,
                                                     shaded.textureTransform7);
        float occlusionStrength = std::clamp(shaded.pbrParams.z, 0.0f, 1.0f);
        float occlusion = std::clamp(occlusionSample.x, 0.0f, 1.0f);
        shaded.pbrParams.z = 1.0f + (occlusion - 1.0f) * occlusionStrength;
        if (debugAO && !ormDisabled) *debugAO = shaded.pbrParams.z;
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
                                  const simd::float3& incidentDir,
                                  float debugNormalStrengthScale,
                                  float surfaceFootprintWorld,
                                  float textureLodBias,
                                  bool flipNormalGreen) {
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
                                       surfaceFootprintWorld,
                                       textureLodBias);
    simd::float4 texel = ctx.hasIgehyGradients
        ? SampleCpuTextureAnisotropic(found->second, ctx.uv, ctx.dUVdx, ctx.dUVdy, lod, textureLodBias)
        : SampleCpuTexture(found->second, ctx.uv, lod);
    simd::float3 normalTs = DecodeNormalMap(simd_make_float3(texel.x, texel.y, texel.z), normalScale);
    if (flipNormalGreen) normalTs.y = -normalTs.y;

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

    simd::float3 mappedNormalTs = normalTs;
    const bool emeraldVertexNormal = hit.hasVertexTangent && std::fabs(hit.tangent.w) > 0.5f &&
        (material.materialFlags & (PathTracerShaderTypes::kMaterialFlagAlphaCutoutContinuation |
                                   PathTracerShaderTypes::kMaterialFlagSourceNormalGreen)) != 0u;
    if (emeraldVertexNormal) {
        // Per-material finding: these Emerald materials (the foliage, and Picnic_Table1 through
        // kMaterialFlagSourceNormalGreen) match the Falcor
        // references only with G negated (other ORCA normal maps
        // match with G kept). Then apply Falcor's complete-normal flip when N faced.
        mappedNormalTs.y = -mappedNormalTs.y;
        if (hit.authoredNormalWasFaced) {
            mappedNormalTs.x = -mappedNormalTs.x;
        }
    }
    simd::float3 mapped = simd::normalize(t * mappedNormalTs.x +
                                          b * mappedNormalTs.y +
                                          shadingNormal * mappedNormalTs.z);
    // Metal keeps the mapped normal on the side of the faced geometric normal (material_access.metal: rec.normal
    // faces the incoming ray). Embree stores the outward Ng, so face it here; on the back face of a double-sided
    // material the outward Ng would turn the mapped normal away from the viewer.
    const bool sourceNormalGreen =
        (material.materialFlags & PathTracerShaderTypes::kMaterialFlagSourceNormalGreen) != 0u;
    const simd::float3 facedGeometricNormal = hit.frontFace ? hit.normal : -hit.normal;
    if (simd::dot(mapped, facedGeometricNormal) < 0.0f) {
        mapped = -mapped;
    }
    if (sourceNormalGreen) {
        // Falcor 4.4 adjustShadingNormal(), as Metal applies it to Picnic_Table1 on both executors
        // (material_access.metal, path_integrators.metal): where the view grazes the mapped normal at a bevel,
        // blend it toward the faced geometric normal.
        const float nDotV = simd::dot(mapped, -incidentDir);
        if (nDotV <= 0.1f) {
            const float w = std::clamp(nDotV * 10.0f, 0.0f, 1.0f);
            mapped = SafeNormalize(facedGeometricNormal + (mapped - facedGeometricNormal) * w, facedGeometricNormal);
        }
    }
    return IsFinite(mapped) ? mapped : shadingNormal;
}

// The normal the coat of a random-walk subsurface material sees at the walk's entry point: the material's normal map
// about `normal` (the point's interpolated normal, facing out). As in Arnold, whose random walk ignores the shading
// normal, the map shapes only the coat lobe and the choice between the coat lobe and the walk. The walk's direction, its
// crossing weights and its exit keep `normal`. The walk has no ray footprint, so the map is read at mip 0,
// bilinear. The frame is the point's own: the interpolated vertex tangent when the mesh has one, else the triangle's
// UV-derivative frame, and no map with degenerate UVs (hit.hasTangent false), as at an ordinary vertex. The mapped
// normal is turned to the side of the outward geometric normal. `mapped` says whether a map was read.
simd::float3 SssCoatNormalMap(const EmbreeSceneData& sceneData,
                              const PathTracerShaderTypes::MaterialData& material,
                              const HitInfo& hit,
                              const simd::float3& normal,
                              bool& mapped) {
    mapped = false;
    const uint32_t index = material.textureIndices0.z;
    auto found = sceneData.materialTextures.find(index);
    if (index == kInvalidTextureIndex || found == sceneData.materialTextures.end() || !hit.hasTangent || !hit.hasUv) {
        return normal;
    }
    const float normalScale = std::max(material.pbrParams.w, 0.0f);
    if (normalScale <= 1.0e-4f) {
        return normal;
    }
    const CpuTextureSamplingContext ctx = MakeCpuTextureSamplingContext(hit, material.textureUvSet0.z,
                                                                        material.textureTransform4,
                                                                        material.textureTransform5);
    const simd::float4 texel = SampleCpuTexture(found->second, ctx.uv, 0.0f);
    const simd::float3 normalTs = DecodeNormalMap(simd_make_float3(texel.x, texel.y, texel.z), normalScale);
    simd::float3 t = simd_make_float3(hit.tangent.x, hit.tangent.y, hit.tangent.z);
    if (!IsFinite(t) || simd::dot(t, t) <= 1.0e-6f) {
        return normal;
    }
    t = simd::normalize(t - normal * simd::dot(normal, t));
    if (!IsFinite(t) || simd::dot(t, t) <= 1.0e-6f) {
        return normal;
    }
    const float tangentSign = hit.tangent.w < 0.0f ? -1.0f : 1.0f;
    const simd::float3 b = simd::normalize(simd::cross(normal, t)) * tangentSign;
    if (!IsFinite(b) || simd::dot(b, b) <= 1.0e-6f) {
        return normal;
    }
    simd::float3 coatNormal = simd::normalize(t * normalTs.x + b * normalTs.y + normal * normalTs.z);
    if (simd::dot(coatNormal, hit.normal) < 0.0f) {
        coatNormal = -coatNormal;
    }
    if (!IsFinite(coatNormal)) {
        return normal;
    }
    mapped = true;
    return coatNormal;
}

// Bends a mapped coat normal N toward the geometric normal Ng (facing the view direction I) so that the mirror reflection
// of I about it stays above the geometric surface, by at least min(0.9 * dot(Ng, I), 0.01); Metal's
// subsurface_coat_valid_reflection_normal (material_access.metal), line for line. This is the rule of Cycles'
// ensure_valid_reflection (Blender, Apache-2.0), rewritten. N is returned unchanged when its reflection is already above
// that threshold. Without the bend a strongly tilted coat normal reflects into the mesh.
simd::float3 SssCoatValidReflectionNormal(const simd::float3& Ng, const simd::float3& I, const simd::float3& N) {
    const float cosGeometric = simd::dot(Ng, I);
    if (!(cosGeometric > 0.0f)) {
        return N;
    }
    const simd::float3 R = 2.0f * simd::dot(N, I) * N - I;
    const float threshold = std::min(0.9f * cosGeometric, 0.01f);
    if (simd::dot(Ng, R) >= threshold) {
        return N;
    }
    // Frame with Ng as z and N in the x-z plane; solve for the normal whose reflection of I has height `threshold`.
    const simd::float3 X = simd::normalize(N - simd::dot(N, Ng) * Ng);
    const float Ix = simd::dot(I, X);
    const float Iz = cosGeometric;
    const float Ix2 = Ix * Ix;
    const float a = Ix2 + Iz * Iz;
    const float b = std::sqrt(std::max(Ix2 * (a - threshold * threshold), 0.0f));
    const float c = Iz * threshold + a;
    const float fac = 0.5f / a;
    const float n1z2 = fac * (b + c);
    const float n2z2 = fac * (c - b);
    bool valid1 = n1z2 > 1.0e-5f && n1z2 <= 1.0f + 1.0e-5f;
    bool valid2 = n2z2 > 1.0e-5f && n2z2 <= 1.0f + 1.0e-5f;
    simd::float2 bent;
    if (valid1 && valid2) {
        const simd::float2 n1 = simd_make_float2(std::sqrt(std::max(1.0f - n1z2, 0.0f)), std::sqrt(std::max(n1z2, 0.0f)));
        const simd::float2 n2 = simd_make_float2(std::sqrt(std::max(1.0f - n2z2, 0.0f)), std::sqrt(std::max(n2z2, 0.0f)));
        const float r1 = 2.0f * (n1.x * Ix + n1.y * Iz) * n1.y - Iz;
        const float r2 = 2.0f * (n2.x * Ix + n2.y * Iz) * n2.y - Iz;
        valid1 = r1 >= 1.0e-5f;
        valid2 = r2 >= 1.0e-5f;
        if (valid1 && valid2) {
            bent = (r1 < r2) ? n1 : n2;  // both reflect above: the shallower one is closer to N
        } else {
            bent = (r1 > r2) ? n1 : n2;
        }
    } else if (valid1 || valid2) {
        const float z2 = valid1 ? n1z2 : n2z2;
        bent = simd_make_float2(std::sqrt(std::max(1.0f - z2, 0.0f)), std::sqrt(std::max(z2, 0.0f)));
    } else {
        return Ng;
    }
    const simd::float3 result = bent.x * X + bent.y * Ng;
    return IsFinite(result) ? simd::normalize(result) : Ng;
}

float ApplyNormalVarianceToRoughness(const EmbreeSceneData& sceneData,
                                     const PathTracerShaderTypes::MaterialData& material,
                                     const HitInfo& hit,
                                     float debugNormalStrengthScale,
                                     float surfaceFootprintWorld,
                                     float textureLodBias) {
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
                                       surfaceFootprintWorld,
                                       textureLodBias);
    simd::float4 raw = ctx.hasIgehyGradients
        ? SampleCpuTextureAnisotropic(texture, ctx.uv, ctx.dUVdx, ctx.dUVdy, lod, textureLodBias)
        : SampleCpuTexture(texture, ctx.uv, lod);
    float directionalVariance = raw.w * raw.w;
    float meanLength = std::sqrt(1.0f - directionalVariance);
    float tok = directionalVariance / (meanLength * (1.0f + meanLength));
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
            float varianceX = 0.5f * simd::dot(nTs - nDx, nTs - nDx);
            float varianceY = 0.5f * simd::dot(nTs - nDy, nTs - nDy);
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

// A mesh vertex's world normal and tangent, with the operations the per-instance arrays used, so hits are unchanged.
simd::float3 MeshWorldNormal(const GeometryData& geom, uint32_t index) {
    simd::float3 n = TransformNormal(geom.normalMatrix, geom.vertices[index].normal);
    return simd::length(n) > 0.0f ? simd::normalize(n) : n;
}

simd::float4 MeshWorldTangent(const GeometryData& geom, uint32_t index) {
    // Metal transforms a tangent by localToWorld (interpolate_tangent_raw), not by the normal matrix, and
    // normalizes it after interpolation (interpolate_tangent); the two differ under non-uniform scale.
    const simd::float4 tangent = geom.vertices[index].tangent;
    const simd::float3 t = geom.localToWorld.columns[0].xyz * tangent.x +
                           geom.localToWorld.columns[1].xyz * tangent.y +
                           geom.localToWorld.columns[2].xyz * tangent.z;
    return simd_make_float4(t.x, t.y, t.z, tangent.w * geom.transformDetSign);
}

// Reports, once per scene, every material feature that Metal implements and Embree does not, with the
// materials that use it. A parity run must show each fallback; none may be silent.
// A source material's bump (height) texture, read by SourceBumpNormal, or kInvalidTextureIndex.
uint32_t SourceBumpTextureIndex(const PathTracerShaderTypes::MaterialData& material) {
    return pt_shared::source_material_flag(material, PathTracerShaderTypes::kMaterialFlagSourceFeatures) &&
                   material.textureControl.z != 0.0f
               ? material.textureIndices1.w
               : kInvalidTextureIndex;
}

void ReportUnsupportedEmbreeMaterialFeatures(const PathTracer::SceneResources& resources) {
    using PathTracerShaderTypes::MaterialData;
    using PathTracerShaderTypes::MaterialType;
    const MaterialData* materials = resources.materialsData();
    const uint32_t materialCount = resources.materialCount();
    if (!materials || materialCount == 0u) {
        return;
    }

    struct Feature {
        const char* description;
        bool (*uses)(const MaterialData&);
        std::vector<uint32_t> materials;
    };
    Feature features[] = {
        {"kMaterialFlagThinFoliageShadow is ignored",
         [](const MaterialData& m) {
             return (m.materialFlags & PathTracerShaderTypes::kMaterialFlagThinFoliageShadow) != 0u;
         }, {}},
        {"kMaterialFlagFoliageBeautyLift is ignored",
         [](const MaterialData& m) {
             return (m.materialFlags & PathTracerShaderTypes::kMaterialFlagFoliageBeautyLift) != 0u;
         }, {}},
    };

    for (uint32_t i = 0; i < materialCount; ++i) {
        for (Feature& feature : features) {
            if (feature.uses(materials[i])) {
                feature.materials.push_back(i);
            }
        }
    }

    constexpr size_t kListedMaterials = 8;
    for (const Feature& feature : features) {
        if (feature.materials.empty()) {
            continue;
        }
        std::cerr << "Embree warning: " << feature.description << " (" << feature.materials.size()
                  << (feature.materials.size() == 1u ? " material: " : " materials: ");
        for (size_t k = 0; k < feature.materials.size() && k < kListedMaterials; ++k) {
            const uint32_t index = feature.materials[k];
            std::cerr << (k > 0 ? ", " : "") << index << " \"" << resources.materialName(index) << "\"";
        }
        if (feature.materials.size() > kListedMaterials) {
            std::cerr << ", +" << (feature.materials.size() - kListedMaterials) << " more";
        }
        std::cerr << ")." << std::endl;
    }
}

// EP22: keep a level as its 8-bit texels when every texel's float is the table's value for its byte. The tables fill
// from the level's own (byte, float) pairs and carry over to the texture's next level; a level whose pairs disagree
// with them (a level filtered in float, or a byte that decodes to two floats) stays float and leaves them unchanged.
bool KeepCpuTextureLevelAsBytes(const PathTracer::MaterialTextureCpuData::MipLevel& level,
                                std::array<float, 256>& rgbFromByte, std::array<float, 256>& alphaFromByte,
                                std::array<bool, 256>& rgbKnown, std::array<bool, 256>& alphaKnown) {
    if (level.rgba8.empty() || level.rgba8.size() != level.rgba.size()) {
        return false;
    }
    std::array<float, 256> rgb = rgbFromByte;
    std::array<float, 256> alpha = alphaFromByte;
    std::array<bool, 256> rgbSeen = rgbKnown;
    std::array<bool, 256> alphaSeen = alphaKnown;
    for (size_t i = 0; i < level.rgba.size(); ++i) {
        const bool isAlpha = (i & 3u) == 3u;
        std::array<float, 256>& table = isAlpha ? alpha : rgb;
        std::array<bool, 256>& seen = isAlpha ? alphaSeen : rgbSeen;
        const uint8_t byte = level.rgba8[i];
        if (!seen[byte]) {
            seen[byte] = true;
            table[byte] = level.rgba[i];
        } else if (std::bit_cast<uint32_t>(table[byte]) != std::bit_cast<uint32_t>(level.rgba[i])) {
            return false;
        }
    }
    rgbFromByte = rgb;
    alphaFromByte = alpha;
    rgbKnown = rgbSeen;
    alphaKnown = alphaSeen;
    return true;
}

// Keep a level that has no source bytes as bytes when every value is an 8-bit step: the byte is the value's nearest
// step, round(value * 255), and the level's own tables map each byte back to the float it replaces, checked bit for bit
// on every texel (colour and alpha apart). A level with two floats on one step, or a value outside [0, 1], stays float.
// Normal-map chain levels qualify: BuildNormalMipChain rounds each of their values to an 8-bit step.
bool KeepFloatLevelAsByteSteps(const std::vector<float>& rgba, std::vector<uint8_t>& bytes,
                               std::array<float, 512>& tables) {
    bytes.resize(rgba.size());
    std::array<bool, 512> seen{};
    for (size_t i = 0; i < rgba.size(); ++i) {
        const float value = rgba[i];
        if (!(value >= 0.0f && value <= 1.0f)) {
            return false;
        }
        const uint8_t byte = static_cast<uint8_t>(std::lround(value * 255.0f));
        const size_t slot = ((i & 3u) == 3u ? 256u : 0u) + byte;
        if (!seen[slot]) {
            seen[slot] = true;
            tables[slot] = value;
        } else if (std::bit_cast<uint32_t>(tables[slot]) != std::bit_cast<uint32_t>(value)) {
            return false;
        }
        bytes[i] = byte;
    }
    return true;
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

    ReportUnsupportedEmbreeMaterialFeatures(resources);

    const auto* materials = resources.materialsData();
    const uint32_t materialCount = resources.materialCount();
    uint32_t footprintMipChains = 0u;
    uint32_t byteLevels = 0u;
    uint32_t floatLevels = 0u;
    uint32_t rejectedByteLevels = 0u;  // decoded from 8-bit texels, yet failed the table check: kept as floats
    uint32_t stepLevels = 0u;          // computed in float, kept as bytes by KeepFloatLevelAsByteSteps
    std::string rejectedExamples;
    // Each texture's CPU copy, in the order the materials first name it. Copies decoded from files (deferred textures:
    // KTX2 or DDS, then their mip chains) run on kTextureCopyThreads workers; a Metal-resident texture is read back
    // through the command queue on this thread. Each copy runs the same code as the serial loop did, and the results
    // are merged in job order, so the textures, the counts and the log lines do not change. The copies were the
    // longest load stage: emerald_square_day spent 18.5 s on one thread there.
    struct TextureJob {
        uint32_t material = 0u;
        uint32_t index = kInvalidTextureIndex;
        bool normalMap = false;
    };
    struct TextureCopy {
        bool ok = false;
        CpuTexture texture;
        bool footprintMipChain = false;
        uint32_t byteLevels = 0u, floatLevels = 0u, stepLevels = 0u, rejectedByteLevels = 0u;
        std::vector<std::string> rejected;  // "texture <index> level <level>" of its rejected levels, the first 3
    };
    std::vector<TextureJob> textureJobs;
    if (materials && materialCount > 0) {
        std::unordered_set<uint32_t> named;
        for (uint32_t i = 0; i < materialCount; ++i) {
            const auto& material = materials[i];
            // textureIndices1.z is a source material's opacity texture (SourceOpacity reads it); others leave it unset.
            const uint32_t indices[] = {
                material.textureIndices0.x,
                material.textureIndices0.y,
                material.textureIndices0.z,
                material.textureIndices0.w,
                material.textureIndices1.x,
                material.textureIndices1.y,
                pt_shared::source_has_opacity(material) ? material.textureIndices1.z : kInvalidTextureIndex,
                SourceBumpTextureIndex(material),
            };
            for (uint32_t index : indices) {
                if (index == kInvalidTextureIndex || !named.insert(index).second) {
                    continue;
                }
                textureJobs.push_back({i, index, index == material.textureIndices0.z});
            }
        }
    }
    std::vector<TextureCopy> textureCopies(textureJobs.size());
    auto copyTexture = [&](size_t job) {
        const uint32_t index = textureJobs[job].index;
        TextureCopy& copy = textureCopies[job];
        PathTracer::MaterialTextureCpuData cpuData;
        if (!resources.materialTextureCpuData(index, cpuData)) {
            return;
        }
        copy.ok = true;
        CpuTexture& cpuTexture = copy.texture;
        cpuTexture.normalMap = textureJobs[job].normalMap;
        cpuTexture.width = cpuData.width;
        cpuTexture.height = cpuData.height;
        // Level 0 is the full-size texture: cpuData.rgba holds the same texels (every producer copies one into
        // the other), so it is not kept a second time.
        if (cpuData.mipLevels.empty() && !cpuData.rgba.empty()) {
            cpuData.mipLevels.push_back({cpuData.width, cpuData.height, std::move(cpuData.rgba), {}});
        }
        cpuData.rgba = {};
        std::array<bool, 256> rgbKnown{};
        std::array<bool, 256> alphaKnown{};
        cpuTexture.mipLevels.reserve(cpuData.mipLevels.size());
        for (size_t levelIndex = 0; levelIndex < cpuData.mipLevels.size(); ++levelIndex) {
            auto& level = cpuData.mipLevels[levelIndex];
            CpuTexture::MipLevel mip;
            std::array<float, 512> stepTables{};
            mip.width = level.width;
            mip.height = level.height;
            if (KeepCpuTextureLevelAsBytes(level, cpuTexture.rgbFromByte, cpuTexture.alphaFromByte,
                                           rgbKnown, alphaKnown)) {
                mip.rgba8 = std::move(level.rgba8);
                ++copy.byteLevels;
            } else if (level.rgba8.empty() && KeepFloatLevelAsByteSteps(level.rgba, mip.rgba8, stepTables)) {
                mip.byteTables = std::make_unique<const std::array<float, 512>>(stepTables);
                ++copy.stepLevels;
            } else {
                mip.rgba8 = {};  // what a failed KeepFloatLevelAsByteSteps wrote
                if (!level.rgba8.empty()) {
                    if (copy.rejected.size() < 3u) {
                        copy.rejected.push_back("texture " + std::to_string(index) + " level " +
                                                std::to_string(levelIndex));
                    }
                    ++copy.rejectedByteLevels;
                }
                mip.rgba = std::move(level.rgba);
                ++copy.floatLevels;
            }
            level = {};
            cpuTexture.mipLevels.push_back(std::move(mip));
        }
        cpuTexture.samplerDesc = cpuData.samplerDesc;
        copy.footprintMipChain = cpuData.footprintMipChain;
    };
    {
        constexpr size_t kTextureCopyThreads = 6u;  // each copy also splits its per-texel loops into GCD row bands
        std::vector<size_t> fileJobs;
        for (size_t job = 0; job < textureJobs.size(); ++job) {
            if (resources.isDeferredTextureIndex(textureJobs[job].index)) {
                fileJobs.push_back(job);
            } else {
                copyTexture(job);
            }
        }
        std::atomic<size_t> next{0u};
        std::vector<std::thread> workers;
        const size_t workerCount = std::min(kTextureCopyThreads, fileJobs.size());
        for (size_t w = 0; w < workerCount; ++w) {
            workers.emplace_back([&]() {
                for (size_t k = next++; k < fileJobs.size(); k = next++) {
                    copyTexture(fileJobs[k]);
                }
            });
        }
        for (std::thread& worker : workers) {
            worker.join();
        }
    }
    for (size_t job = 0; job < textureJobs.size(); ++job) {
        TextureCopy& copy = textureCopies[job];
        if (!copy.ok) {
            std::cerr << "Embree warning: material " << textureJobs[job].material << " texture "
                      << textureJobs[job].index << " omitted: CPU texture acquisition failed." << std::endl;
            continue;
        }
        byteLevels += copy.byteLevels;
        floatLevels += copy.floatLevels;
        stepLevels += copy.stepLevels;
        for (const std::string& example : copy.rejected) {
            if (rejectedByteLevels < 3u) {
                rejectedExamples += (rejectedExamples.empty() ? "" : ", ") + example;
            }
            ++rejectedByteLevels;
        }
        rejectedByteLevels += copy.rejectedByteLevels - static_cast<uint32_t>(copy.rejected.size());
        footprintMipChains += copy.footprintMipChain ? 1u : 0u;
        outData.materialTextures.emplace(textureJobs[job].index, std::move(copy.texture));
    }
    // Metal picks the nearest texel when both filters do (CpuTextureNearestTexel), and so does Embree's CPU sampler
    // (EP8c). A texture that mixes a nearest and a linear filter (a glTF sampler; a source material's textureFilter sets
    // both) switches between them by LOD on Metal, and on Embree too (CpuTextureSampleNearest).
    {
        size_t nearestTextures = 0u, mixedTextures = 0u;
        for (const auto& [index, texture] : outData.materialTextures) {
            nearestTextures += CpuTextureNearestTexel(texture.samplerDesc) ? 1u : 0u;
            mixedTextures += CpuTextureMixesNearestAndLinear(texture.samplerDesc) ? 1u : 0u;
        }
        if (nearestTextures > 0u) {
            std::cout << "[EmbreeTexture] " << nearestTextures << " of " << outData.materialTextures.size()
                      << " material textures sampled at the nearest texel, as Metal" << std::endl;
        }
        if (mixedTextures > 0u) {
            std::cout << "[EmbreeTexture] " << mixedTextures << " material textures mix a nearest and a linear filter: "
                      << "the magnification filter at LOD 0 or below, the minification filter above, as Metal" << std::endl;
        }
    }
    // A source-opacity material whose opacity texture has no CPU copy would read as opaque, and a bump map without one
    // would be flat: refuse them instead.
    if (materials) {
        std::string missingOpacity, missingBump;
        for (uint32_t i = 0; i < materialCount; ++i) {
            const uint32_t index = materials[i].textureIndices1.z;
            if (pt_shared::source_has_opacity(materials[i]) && index != kInvalidTextureIndex &&
                outData.materialTextures.count(index) == 0u) {
                missingOpacity += (missingOpacity.empty() ? "" : ", ") + std::to_string(i);
            }
            const uint32_t bump = SourceBumpTextureIndex(materials[i]);
            if (bump != kInvalidTextureIndex && outData.materialTextures.count(bump) == 0u) {
                missingBump += (missingBump.empty() ? "" : ", ") + std::to_string(i);
            }
        }
        if (!missingOpacity.empty()) {
            error = "Embree could not copy the opacity texture of source material(s) " + missingOpacity;
            return false;
        }
        if (!missingBump.empty()) {
            error = "Embree could not copy the bump texture of source material(s) " + missingBump;
            return false;
        }
    }
    std::cout << "[MipChain] Embree textures not a power of two in size (every 8-bit colour texture decodes MipChain.h's "
                 "footprint-filtered mips; these need the odd-size weights): "
              << footprintMipChains << " of " << outData.materialTextures.size() << std::endl;
    {
        size_t byteBytes = 0u, floatBytes = 0u, texels = 0u;
        for (const auto& [index, texture] : outData.materialTextures) {
            texels += static_cast<size_t>(texture.width) * texture.height;
            for (const auto& level : texture.mipLevels) {
                byteBytes += level.rgba8.size();
                floatBytes += level.rgba.size() * sizeof(float);
            }
        }
        std::ostringstream line;
        line << std::fixed << std::setprecision(2) << "[LoadMemory] material textures: "
             << outData.materialTextures.size() << ", " << static_cast<double>(texels) / 1.0e6 << " M texels at mip 0; "
             << byteLevels + stepLevels << " levels kept as 8-bit texels (" << stepLevels
             << " of them computed in float, each value an 8-bit step; " << static_cast<double>(byteBytes) / 1073741824.0
             << " GB), " << floatLevels << " as floats (" << static_cast<double>(floatBytes) / 1073741824.0 << " GB)";
        std::cout << line.str() << std::endl;
        if (rejectedByteLevels > 0u) {
            std::cerr << "[LoadMemory] warning: " << rejectedByteLevels << " texture levels decoded from 8-bit texels "
                      << "failed the byte-to-float table check and stay float, at four times the memory (first: "
                      << rejectedExamples << ")." << std::endl;
        }
    }
    LogLoadMemory("material textures copied");

    const auto& meshes = resources.meshes();
    outData.meshPositions.reserve(meshes.size());

    // Embree shares two arrays rather than keeping its own copies: each mesh's world positions in meshPositions
    // (simd::float3, a 16-byte stride, so a 16-byte load from the last vertex stays inside the array), and the mesh's
    // indices in SceneResources, which outlives the render. Hit shading reads uv and uv1 from SceneResources' vertices.
    uint32_t meshTriangleOffset = 0u;
    for (const auto& mesh : meshes) {
        if (mesh.vertices.empty() || mesh.indices.empty()) {
            continue;
        }

        const simd::float3x3 normalMatrix = NormalMatrix(mesh.localToWorld);
        const float transformDetSign = TransformDeterminantSign(mesh.localToWorld);

        std::vector<simd::float3> positions;
        positions.reserve(mesh.vertices.size());
        for (const auto& vertex : mesh.vertices) {
            positions.push_back(TransformPoint(mesh.localToWorld, vertex.position));
        }

        outData.meshPositions.emplace_back(std::move(positions));

        auto geomData = std::make_unique<GeometryData>();
        geomData->type = GeometryType::Mesh;
        geomData->positions = outData.meshPositions.back().data();
        geomData->vertices = mesh.vertices.data();
        geomData->worldFrameAtHit = true;
        geomData->normalMatrix = normalMatrix;
        geomData->localToWorld = mesh.localToWorld;
        geomData->transformDetSign = transformDetSign;
        geomData->indices = mesh.indices.data();
        geomData->indexCount = static_cast<uint32_t>(mesh.indices.size());
        geomData->materialIndex = mesh.materialIndex;
        {
            const PathTracerShaderTypes::MaterialData* sceneMaterials = resources.materialsData();
            geomData->unitFrameForConstantUv = sceneMaterials && mesh.materialIndex < resources.materialCount() &&
                (sceneMaterials[mesh.materialIndex].materialFlags &
                 PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular) != 0u;
        }
        geomData->triangleOffset = meshTriangleOffset;
        geomData->meshIndex = static_cast<uint32_t>(&mesh - meshes.data());

        RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
        rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                   outData.meshPositions.back().data(), 0, sizeof(simd::float3),
                                   outData.meshPositions.back().size());
        unsigned int triCount = static_cast<unsigned int>(geomData->indexCount / 3);
        rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                                   mesh.indices.data(), 0, sizeof(uint32_t) * 3, triCount);

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
            if (AreaPrimitiveIsReferencePoint(rect)) {
                continue;
            }
            simd::float3 corner = {rect.corner.x, rect.corner.y, rect.corner.z};
            simd::float3 edgeU = {rect.edgeU.x, rect.edgeU.y, rect.edgeU.z};
            simd::float3 edgeV = {rect.edgeV.x, rect.edgeV.y, rect.edgeV.z};
            simd::float3 normal = {rect.normalAndPlane.x, rect.normalAndPlane.y, rect.normalAndPlane.z};
            normal = simd::normalize(normal);

            if (AreaPrimitiveIsDisk(rect)) {
                auto geomData = std::make_unique<GeometryData>();
                geomData->type = GeometryType::Rectangles;
                geomData->disk = &rect;
                RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_USER);
                rtcSetGeometryUserPrimitiveCount(geometry, 1u);
                rtcSetGeometryUserData(geometry, geomData.get());
                rtcSetGeometryBoundsFunction(geometry, [](const RTCBoundsFunctionArguments* args) {
                    const auto& disk = *static_cast<const GeometryData*>(args->geometryUserPtr)->disk;
                    simd::float3 center = simd_make_float3(disk.corner);
                    simd::float3 extent = simd::abs(simd_make_float3(disk.edgeU)) +
                                          simd::abs(simd_make_float3(disk.edgeV));
                    simd::float3 lower = center - extent, upper = center + extent;
                    args->bounds_o->lower_x = lower.x;
                    args->bounds_o->lower_y = lower.y;
                    args->bounds_o->lower_z = lower.z;
                    args->bounds_o->upper_x = upper.x;
                    args->bounds_o->upper_y = upper.y;
                    args->bounds_o->upper_z = upper.z;
                }, nullptr);
                rtcSetGeometryIntersectFunction(geometry, [](const RTCIntersectFunctionNArguments* args) {
                    const auto& disk = *static_cast<const GeometryData*>(args->geometryUserPtr)->disk;
                    RTCRayN* rays = RTCRayHitN_RayN(args->rayhit, args->N);
                    RTCHitN* hits = RTCRayHitN_HitN(args->rayhit, args->N);
                    for (unsigned int i = 0; i < args->N; ++i) {
                        if (args->valid[i] != -1) continue;
                        simd::float3 origin = {RTCRayN_org_x(rays, args->N, i),
                                              RTCRayN_org_y(rays, args->N, i),
                                              RTCRayN_org_z(rays, args->N, i)};
                        simd::float3 direction = {RTCRayN_dir_x(rays, args->N, i),
                                                 RTCRayN_dir_y(rays, args->N, i),
                                                 RTCRayN_dir_z(rays, args->N, i)};
                        simd::float3 normal = simd_make_float3(disk.normalAndPlane);
                        float denom = simd::dot(normal, direction);
                        if (std::fabs(denom) < 1e-6f) continue;
                        float t = (disk.normalAndPlane.w - simd::dot(normal, origin)) / denom;
                        if (t < RTCRayN_tnear(rays, args->N, i) || t > RTCRayN_tfar(rays, args->N, i)) continue;
                        simd::float3 relative = origin + t * direction - simd_make_float3(disk.corner);
                        float u = simd::dot(relative, simd_make_float3(disk.edgeU)) * disk.edgeU.w;
                        float v = simd::dot(relative, simd_make_float3(disk.edgeV)) * disk.edgeV.w;
                        if ((u * u + v * v) > 1.0f) continue;
                        RTCRayN_tfar(rays, args->N, i) = t;
                        RTCHitN_Ng_x(hits, args->N, i) = normal.x;
                        RTCHitN_Ng_y(hits, args->N, i) = normal.y;
                        RTCHitN_Ng_z(hits, args->N, i) = normal.z;
                        RTCHitN_u(hits, args->N, i) = 0.0f;
                        RTCHitN_v(hits, args->N, i) = 0.0f;
                        RTCHitN_geomID(hits, args->N, i) = args->geomID;
                        RTCHitN_primID(hits, args->N, i) = args->primID;
                        for (unsigned int level = 0; level < RTC_MAX_INSTANCE_LEVEL_COUNT; ++level) {
                            RTCHitN_instID(hits, args->N, i, level) = args->context->instID[level];
#if defined(RTC_GEOMETRY_INSTANCE_ARRAY)
                            RTCHitN_instPrimID(hits, args->N, i, level) = args->context->instPrimID[level];
#endif
                        }
                    }
                });
                rtcCommitGeometry(geometry);
                rtcAttachGeometry(scene, geometry);
                rtcReleaseGeometry(geometry);
                outData.geometryData.emplace_back(std::move(geomData));
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

    LogLoadMemory("geometry buffers built");
    rtcCommitScene(scene);
    LogLoadMemory("BVH committed");
    return true;
}

bool IntersectScene(const EmbreeSceneData& sceneData, const Ray& ray, HitInfo& outHit,
                    float minT = kEpsilon) {
    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);

    RTCRayHit rayHit{};
    rayHit.ray.org_x = ray.origin.x;
    rayHit.ray.org_y = ray.origin.y;
    rayHit.ray.org_z = ray.origin.z;
    rayHit.ray.dir_x = ray.direction.x;
    rayHit.ray.dir_y = ray.direction.y;
    rayHit.ray.dir_z = ray.direction.z;
    rayHit.ray.tnear = minT;
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
    outHit.meshIndex = geomData->meshIndex;
    outHit.twoSided = false;
    outHit.hasVertexTangent = false;
    outHit.authoredNormalWasFaced = false;

    if (geomData->type == GeometryType::Mesh && geomData->indices && geomData->worldFrameAtHit && geomData->vertices) {
        outHit.primitiveIndex = geomData->triangleOffset + primID;
        uint32_t base = primID * 3u;
        if (base + 2u < geomData->indexCount) {
            uint32_t i0 = geomData->indices[base + 0u];
            uint32_t i1 = geomData->indices[base + 1u];
            uint32_t i2 = geomData->indices[base + 2u];
            float u = rayHit.hit.u;
            float v = rayHit.hit.v;
            float w = 1.0f - u - v;
            simd::float3 n0 = MeshWorldNormal(*geomData, i0);
            simd::float3 n1 = MeshWorldNormal(*geomData, i1);
            simd::float3 n2 = MeshWorldNormal(*geomData, i2);
            simd::float3 interp = w * n0 + u * n1 + v * n2;
            if (simd::dot(interp, interp) > 0.0f) {
                shadingNormal = simd::normalize(interp);
                if (simd::dot(shadingNormal, adjustedNormal) < 0.0f) {
                    outHit.authoredNormalWasFaced = true;
                    shadingNormal = -shadingNormal;
                }
            }
            if (geomData->vertices) {
                simd::float2 uv0 = geomData->vertices[i0].uv;
                simd::float2 uv1 = geomData->vertices[i1].uv;
                simd::float2 uv2 = geomData->vertices[i2].uv;
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
                                                       outHit.uvPerWorld0,
                                                       geomData->unitFrameForConstantUv);
                    outHit.hasSourceBumpFrame = true;
                    outHit.sourceBumpQ1 = p1 - p0;
                    outHit.sourceBumpQ2 = p2 - p0;
                    outHit.sourceBumpE1 = uv1 - uv0;
                    outHit.sourceBumpE2 = uv2 - uv0;
                }
            }
            if (geomData->vertices) {
                simd::float2 uv10 = geomData->vertices[i0].uv1;
                simd::float2 uv11 = geomData->vertices[i1].uv1;
                simd::float2 uv12 = geomData->vertices[i2].uv1;
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
            bool vertexTangentUsable = false;
            {
                simd::float4 t0 = MeshWorldTangent(*geomData, i0);
                simd::float4 t1 = MeshWorldTangent(*geomData, i1);
                simd::float4 t2 = MeshWorldTangent(*geomData, i2);
                outHit.tangent = w * t0 + u * t1 + v * t2;
                outHit.hasTangent = IsFinite4(outHit.tangent);
                const float tangentLengthSq = simd::dot(outHit.tangent.xyz, outHit.tangent.xyz);
                if (outHit.hasTangent && tangentLengthSq > 1.0e-12f) {
                    const simd::float3 unit = outHit.tangent.xyz / std::sqrt(tangentLengthSq);
                    outHit.tangent = simd_make_float4(unit.x, unit.y, unit.z, outHit.tangent.w);
                    vertexTangentUsable = true;
                }
                // triangle_has_valid_vertex_tangents checks the stored (local) vertex tangents.
                const simd::float4 l0 = geomData->vertices ? geomData->vertices[i0].tangent : t0;
                const simd::float4 l1 = geomData->vertices ? geomData->vertices[i1].tangent : t1;
                const simd::float4 l2 = geomData->vertices ? geomData->vertices[i2].tangent : t2;
                outHit.hasVertexTangent = outHit.hasTangent &&
                    IsFinite4(l0) && IsFinite4(l1) && IsFinite4(l2) &&
                    std::fabs(l0.w) > 0.5f && std::fabs(l1.w) > 0.5f && std::fabs(l2.w) > 0.5f &&
                    simd::dot(l0.xyz, l0.xyz) > 1.0e-6f && simd::dot(l1.xyz, l1.xyz) > 1.0e-6f &&
                    simd::dot(l2.xyz, l2.xyz) > 1.0e-6f;
            }
            // No usable vertex tangent (a zero TANGENT, interpolated): the frame comes from the UVs, as Metal's
            // compute_tangent_basis_from_uv builds it (world-space dP/du, handedness against the world-space dP/dv,
            // the same limits). With degenerate UVs there is no frame and ApplyNormalMapForHit applies no normal map.
            if (!vertexTangentUsable && geomData->positions && geomData->vertices) {
                const simd::float3 p0 = geomData->positions[i0];
                const simd::float3 p1 = geomData->positions[i1];
                const simd::float3 p2 = geomData->positions[i2];
                const simd::float2 uv0 = geomData->vertices[i0].uv;
                const simd::float2 uv1 = geomData->vertices[i1].uv;
                const simd::float2 uv2 = geomData->vertices[i2].uv;
                const simd::float3 edge1 = p1 - p0;
                const simd::float3 edge2 = p2 - p0;
                const simd::float2 dUv1 = uv1 - uv0;
                const simd::float2 dUv2 = uv2 - uv0;
                const float det = dUv1.x * dUv2.y - dUv1.y * dUv2.x;
                if (std::fabs(det) >= 1.0e-8f) {
                    const float invDet = 1.0f / det;
                    simd::float3 tangent = (edge1 * dUv2.y - edge2 * dUv1.y) * invDet;
                    simd::float3 bitangent = (edge2 * dUv1.x - edge1 * dUv2.x) * invDet;
                    const float tangentLenSq = simd::dot(tangent, tangent);
                    const float bitangentLenSq = simd::dot(bitangent, bitangent);
                    if (IsFinite(tangent) && IsFinite(bitangent) && tangentLenSq > 1.0e-12f && bitangentLenSq > 1.0e-12f) {
                        tangent /= std::sqrt(tangentLenSq);
                        bitangent /= std::sqrt(bitangentLenSq);
                        tangent = tangent - shadingNormal * simd::dot(shadingNormal, tangent);
                        const float projectedLenSq = simd::dot(tangent, tangent);
                        if (IsFinite(tangent) && projectedLenSq > 1.0e-12f) {
                            tangent /= std::sqrt(projectedLenSq);
                            const float sign =
                                (simd::dot(simd::cross(shadingNormal, tangent), bitangent) < 0.0f) ? -1.0f : 1.0f;
                            outHit.tangent = simd_make_float4(tangent.x, tangent.y, tangent.z, sign);
                            outHit.hasTangent = true;
                        }
                    }
                }
            }
        }
    } else if (geomData->disk) {
        outHit.primitiveIndex = static_cast<uint32_t>(geomData->disk - sceneData.rectangles);
        outHit.twoSided = geomData->disk->materialTwoSided.y != 0u;
        materialIndex = geomData->disk->materialTwoSided.x;
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
            // Metal's outward normal is (p - c) / radius (visibility.metal): a negative radius turns it inward.
            simd::float3 normal = simd::normalize((outHit.position - center) / sphere.w);
            adjustedNormal = simd::dot(ray.direction, normal) < 0.0f ? normal : -normal;
            // The shading normal faces the incoming ray, as on meshes here and in Metal's set_face_normal
            // (rec.shadingNormal = rec.normal, the faced normal).
            shadingNormal = adjustedNormal;
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

// Metal's constants for the subsurface random walk (shaders/core/constants.metal).
constexpr float kSssRayOriginEpsilon = 1.0e-4f;          // kRayOriginEpsilon
constexpr float kSssThroughputCutoff = 1.0e-3f;          // kSssThroughputCutoff
constexpr float kSssHardwareOcclusionEpsilon = 5.0e-3f;  // kHardwareOcclusionEpsilon

// Metal's offset_surface_point (visibility.metal).
simd::float3 OffsetSurfacePoint(simd::float3 point, simd::float3 normal, simd::float3 direction) {
    const simd::float3 n = (IsFinite(normal) && simd::dot(normal, normal) > 0.0f) ? simd::normalize(normal)
                                                                                : simd_make_float3(0.0f, 1.0f, 0.0f);
    const float sign = simd::dot(direction, n) >= 0.0f ? 1.0f : -1.0f;
    return point + n * (sign * kSssRayOriginEpsilon * 4.0f) + direction * (kSssRayOriginEpsilon * 0.5f);
}

// Metal's sample_henyey_greenstein_world: a Henyey-Greenstein direction about `reference`.
simd::float3 SampleHenyeyGreenstein(simd::float3 reference, float g, Rng& rng) {
    const float u1 = rng.nextFloat();
    const float u2 = rng.nextFloat();
    float cosTheta = 1.0f - 2.0f * u1;
    if (std::fabs(g) >= 1.0e-3f) {
        const float s = (1.0f - g * g) / (1.0f - g + 2.0f * g * u1);
        cosTheta = std::clamp((1.0f + g * g - s * s) / (2.0f * g), -1.0f, 1.0f);
    }
    const float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
    const float phi = 2.0f * kPi * u2;
    const Onb onb = BuildOnb(SafeNormalize(reference, simd_make_float3(0.0f, 0.0f, 1.0f)));
    return SafeNormalize(sinTheta * std::cos(phi) * onb.tangent + sinTheta * std::sin(phi) * onb.bitangent +
                             cosTheta * onb.normal,
                         onb.normal);
}

// Metal's random-walk subsurface (#73, closures_eval.metal: sss_walk_*), the model Arnold's randomwalk uses: light
// enters the mesh as a diffuse transmission, walks through a homogeneous medium behind an index-matched boundary, and
// where it leaves the mesh the path continues from an ordinary diffuse surface vertex. `base` is the colour a dense
// medium converges to, `mfp` is 1 / sigma_t and `g` the Henyey-Greenstein anisotropy; `ior` plays no part.

// Single-scattering albedo whose semi-infinite medium reflects the fraction `colour` of diffuse light: the closed-form
// inverse of van de Hulst's fit (Chiang et al. 2016), as sss_walk_single_scatter_albedo.
simd::float3 SssWalkSingleScatterAlbedo(simd::float3 colour) {
    const simd::float3 c = simd::clamp(colour, Splat3(0.0f), Splat3(0.999f));
    const simd::float3 s = Splat3(4.09712f) + 4.20863f * c - simd::sqrt(Splat3(9.59217f) + 41.6808f * c + 17.7126f * c * c);
    return simd::clamp(Splat3(1.0f) - s * s, Splat3(0.0f), Splat3(1.0f));
}

struct SssWalkMedium {
    simd::float3 sigmaT{0.0f, 0.0f, 0.0f};
    simd::float3 sigmaS{0.0f, 0.0f, 0.0f};
    float g = 0.0f;
};

// Metal's sss_walk_medium.
SssWalkMedium SssWalkMediumFor(const PathTracerShaderTypes::MaterialData& material) {
    SssWalkMedium medium;
    medium.g = SssAnisotropy(material);
    if (SssHasSigmaOverride(material)) {
        // Explicit sigma_a and sigma_s are used as given.
        medium.sigmaS = simd::max(simd_make_float3(material.sssSigmaS.x, material.sssSigmaS.y, material.sssSigmaS.z),
                                  Splat3(0.0f));
        medium.sigmaT = simd::max(simd::max(simd_make_float3(material.sssSigmaA.x, material.sssSigmaA.y,
                                                             material.sssSigmaA.z),
                                            Splat3(0.0f)) + medium.sigmaS,
                                  Splat3(1.0e-6f));
        return medium;
    }
    // The inversion holds for isotropic scattering; the similarity relation carries it to g != 0.
    const simd::float3 reduced = SssWalkSingleScatterAlbedo(MaterialBaseColor(material));
    const simd::float3 albedo =
        reduced / simd::max(Splat3(1.0f - medium.g) + medium.g * reduced, Splat3(1.0e-6f));
    const float sigmaT = 1.0f / SssMeanFreePath(material);
    medium.sigmaT = Splat3(sigmaT);
    medium.sigmaS = Clamp01(albedo) * sigmaT;
    return medium;
}

// Metal's kSssWalk* outcome codes, which are also the per-material counter slots after the walk count.
enum class SssWalkOutcome : uint32_t { Continue = 0u, Exit = 1u, Absorbed = 2u, OpenMesh = 3u, StepLimit = 4u };
constexpr uint32_t kSssWalkMaxSkips = 8u;

// Metal's sss_walk_note_outcome: walks and endings per material over the render (slot 0 the walk count, then the
// outcome code), counted from every render thread and reported in the metadata and the [SSS] lines.
struct SssWalkStats {
    std::vector<std::atomic<uint64_t>> counts =
        std::vector<std::atomic<uint64_t>>(PathTracerShaderTypes::kMaxMaterials * PathTracerShaderTypes::kSssWalkStatCount);
    void note(uint32_t materialIndex, SssWalkOutcome outcome) {
        // Metal clamps an index past the table into the last slot, so such walks are still counted.
        const size_t base = static_cast<size_t>(std::min(materialIndex, PathTracerShaderTypes::kMaxMaterials - 1u)) *
                            PathTracerShaderTypes::kSssWalkStatCount;
        counts[base + PathTracerShaderTypes::kSssWalkStatWalks].fetch_add(1u, std::memory_order_relaxed);
        counts[base + static_cast<uint32_t>(outcome)].fetch_add(1u, std::memory_order_relaxed);
    }
};

struct SssWalkState {
    simd::float3 position{0.0f, 0.0f, 0.0f};
    simd::float3 direction{0.0f, 0.0f, 1.0f};
    simd::float3 throughput{1.0f, 1.0f, 1.0f};
    simd::float3 channelProbability{0.0f, 0.0f, 0.0f};  // probability of drawing this segment's length per channel
    float distance = 0.0f;                              // free-flight length drawn for the current segment
    float travelled = 0.0f;                             // part of it already behind the walk (other objects skipped)
    float traceMin = kSssRayOriginEpsilon;              // kRayOriginEpsilon from a surface, 0 from a scatter point
    uint32_t skips = 0u;
};

// Metal's sss_walk_sample_segment: the channel is chosen in proportion to the throughput and the weights divide by
// the density averaged over the channels.
void SssWalkSampleSegment(SssWalkState& walk, const SssWalkMedium& medium, Rng& rng) {
    const float sum = walk.throughput.x + walk.throughput.y + walk.throughput.z;
    walk.channelProbability = walk.throughput / std::max(sum, 1.0e-20f);
    const float pick = rng.nextFloat();
    float sigma = medium.sigmaT.z;
    if (pick < walk.channelProbability.x) {
        sigma = medium.sigmaT.x;
    } else if (pick < walk.channelProbability.x + walk.channelProbability.y) {
        sigma = medium.sigmaT.y;
    }
    const float xi = std::clamp(rng.nextFloat(), 1.0e-6f, 1.0f - 1.0e-6f);
    walk.distance = -std::log(1.0f - xi) / sigma;
    walk.travelled = 0.0f;
    walk.skips = 0u;
}

// Metal's sss_walk_same_object: the entry's own mesh for triangles, its own primitive otherwise.
bool SssWalkSameObject(const HitInfo& entry, const HitInfo& hit) {
    if (hit.primitiveType != entry.primitiveType) {
        return false;
    }
    return hit.primitiveType == GeometryType::Mesh ? hit.meshIndex == entry.meshIndex
                                                   : hit.primitiveIndex == entry.primitiveIndex;
}

// Metal's sss_walk_advance: consumes the closest-hit trace of the current segment.
SssWalkOutcome SssWalkAdvance(SssWalkState& walk, const SssWalkMedium& medium, const HitInfo& entry, bool hit,
                              const HitInfo& boundary, Rng& rng) {
    if (!hit) {
        return SssWalkOutcome::OpenMesh;  // nothing ahead in any direction means the walk is outside its mesh
    }
    const float remaining = walk.distance - walk.travelled;
    if (boundary.t >= remaining) {
        // Scatter inside the medium before reaching any surface. As Metal, the transmittance is over the whole drawn
        // length, which is the density the length was drawn from.
        const simd::float3 transmittance = Exp3(-medium.sigmaT * walk.distance);
        const float density = simd::dot(walk.channelProbability, medium.sigmaT * transmittance);
        walk.throughput *= medium.sigmaS * transmittance / std::max(density, 1.0e-30f);
        if (!(std::max({walk.throughput.x, walk.throughput.y, walk.throughput.z}) >= kSssThroughputCutoff)) {
            return SssWalkOutcome::Absorbed;
        }
        walk.position += walk.direction * remaining;
        walk.direction = SampleHenyeyGreenstein(walk.direction, medium.g, rng);
        if (!IsFinite(walk.direction) || simd::dot(walk.direction, walk.direction) <= 0.0f) {
            return SssWalkOutcome::Absorbed;
        }
        // A scatter point lies inside the medium, not on a surface, so the next trace needs no offset.
        walk.traceMin = 0.0f;
        SssWalkSampleSegment(walk, medium, rng);
        return SssWalkOutcome::Continue;
    }
    if (!SssWalkSameObject(entry, boundary)) {
        // Another object's surface inside this one: the medium continues behind it.
        if (walk.skips >= kSssWalkMaxSkips) {
            return SssWalkOutcome::OpenMesh;
        }
        walk.skips += 1u;
        walk.travelled += boundary.t + kSssRayOriginEpsilon;
        walk.position = boundary.position + walk.direction * kSssRayOriginEpsilon;
        walk.traceMin = kSssRayOriginEpsilon;
        return SssWalkOutcome::Continue;
    }
    if (boundary.frontFace) {
        return SssWalkOutcome::OpenMesh;  // met its own surface from outside
    }
    // Reached the boundary from inside before the drawn length: the walk leaves the mesh here.
    const simd::float3 reach = Exp3(-medium.sigmaT * (walk.travelled + boundary.t));
    walk.throughput *= reach / std::max(simd::dot(walk.channelProbability, reach), 1.0e-30f);
    return SssWalkOutcome::Exit;
}

// Metal's sss_random_walk_software / _hardware, from a front-face hit of a random-walk subsurface material. Returns
// the outcome; on Exit, `exitHit` is the front-face record a ray arriving from outside would have produced at the
// exit point (Metal's sss_walk_exit_record) and `weight` the walk's throughput.
SssWalkOutcome SssRandomWalk(const EmbreeSceneData& sceneData,
                             const PathTracerShaderTypes::MaterialData& material,
                             const HitInfo& entry,
                             uint32_t maxSteps,
                             Rng& rng,
                             HitInfo& exitHit,
                             simd::float3& weight) {
    const SssWalkMedium medium = SssWalkMediumFor(material);
    // Metal's sss_walk_begin: a diffuse transmission, cosine-weighted about the inward normal, with weight 1. The entry
    // is a front-face hit, so Embree's outward geometric normal faces the incoming ray there, as Metal's rec.normal.
    SssWalkState walk;
    const simd::float3 inward = -entry.normal;
    float cosinePdf = 0.0f;
    walk.direction = SafeNormalize(SampleCosineHemisphere(rng, inward, cosinePdf), inward);
    walk.position = OffsetSurfacePoint(entry.position, inward, walk.direction);
    walk.throughput = Splat3(1.0f);
    walk.traceMin = kSssRayOriginEpsilon;
    SssWalkSampleSegment(walk, medium, rng);
    for (uint32_t step = 0u; step < std::max(maxSteps, 1u); ++step) {
        Ray segment;
        segment.origin = walk.position;
        segment.direction = walk.direction;
        segment.unnormalizedDirection = walk.direction;
        HitInfo boundary;
        const bool hit = IntersectScene(sceneData, segment, boundary, walk.traceMin);
        const SssWalkOutcome event = SssWalkAdvance(walk, medium, entry, hit, boundary, rng);
        if (event == SssWalkOutcome::Continue) {
            continue;
        }
        if (event == SssWalkOutcome::Exit) {
            // Seen from inside, the boundary is a back-face hit: Embree's geometric normal already points out of the
            // mesh and its interpolated shading normal faces the walk, inward. The exit record turns the shading
            // normal outward, to the side of the geometric normal (Metal negates it, and SSS15 turns its software
            // exit's interpolated normal to that side), and marks the hit a front face.
            exitHit = boundary;
            simd::float3 shading = -boundary.shadingNormal;
            if (simd::dot(shading, exitHit.normal) < 0.0f) {
                shading = -shading;
            }
            exitHit.shadingNormal = SafeNormalize(shading, exitHit.normal);
            exitHit.frontFace = true;
            weight = walk.throughput;
        }
        return event;
    }
    return SssWalkOutcome::StepLimit;
}

// Metal's shadow-ray constants: spawn_ray_min(material, kEpsilon) for a material without interface precision (Embree
// ignores that flag and logs it), which is also the radius of trace_scene_hardware's same-mesh exclusion (Metal's
// kEpsilon); the step past a skipped hit (kHardwareOcclusionEpsilon, which the subsurface walk uses too); and the
// alpha-aware shadow trace's skip limits (path_integrators.metal).
constexpr float kShadowSpawnMinT = 1.0e-3f;
// A shadow ray toward a light with a position stops this far short of it, and never shorter than this, as Metal's
// shadowMax = max(distance - kEpsilon, kEpsilon) at its rect, tagged-mesh, RIS-winner and audit sites.
constexpr float kShadowLightEndMargin = 1.0e-3f;
// Under source transport (EP8d): Metal's source_ray_epsilon and its 1e-6 end margin toward a light with a position.
inline float ShadowSpawnMinT() { return gSourceTransport.enabled ? gSourceTransport.epsilon : kShadowSpawnMinT; }
inline float ShadowLightEndMargin() { return gSourceTransport.enabled ? 1.0e-6f : kShadowLightEndMargin; }
constexpr float kShadowSkipStep = kSssHardwareOcclusionEpsilon;  // Metal's kHardwareOcclusionEpsilon.
constexpr uint32_t kShadowAlphaSkips = 16u;
constexpr uint32_t kShadowAlphaSkipsContinuation = 64u;

// What a shadow ray must not report as its blocker, as Metal's compute_exclusion_indices gives it to
// trace_scene_hardware: the shaded triangle (`triangle`, a global triangle index) of mesh `mesh`. `attempts` is
// UniformBuilder's hardwareExcludeMaxAttempts, the number of traces one query may spend on skipped hits.
struct ShadowExclusion {
    uint32_t mesh = std::numeric_limits<uint32_t>::max();
    uint32_t triangle = std::numeric_limits<uint32_t>::max();
    uint32_t attempts = 1u;
};

ShadowExclusion ShadedTriangleExclusion(const HitInfo& hit, uint32_t attempts) {
    ShadowExclusion exclusion;
    exclusion.attempts = attempts;
    if (hit.primitiveType == GeometryType::Mesh) {
        exclusion.mesh = hit.meshIndex;
        exclusion.triangle = hit.primitiveIndex;
    }
    return exclusion;
}

// One query with trace_scene_hardware's exclusion (visibility.metal): a hit on the excluded mesh is skipped when it is
// the excluded triangle or, for an occlusion query (`shadowQuery`), lies within kShadowSpawnMinT of the trace origin,
// and the trace starts again kShadowSkipStep past it with no minimum distance, at most `exclusion.attempts` traces in
// all; when they run out, the query reports no hit. A path ray (shadowQuery false) skips the excluded triangle only.
// outHit.t is in units of ray.direction from ray.origin, re-traces included. Three departures from Metal, all rare
// because each needs a skipped hit first:
// - Metal tests spheres and rectangles separately, along the whole original ray. Here they share the triangle trace,
//   so one within kShadowSkipStep past a skipped hit, or behind the hits that used up the attempts, is not seen.
// - Metal's re-trace keeps the caller's tMax from the re-trace origin and returns its t from there, so it can reach
//   up to kShadowSkipStep plus the skipped distance past the light. Here every distance is from ray.origin.
// - Metal also skips a same-mesh hit within kShadowSkipStep that its software triangle test cannot reproduce
//   (hwOnlyNearMeshHit), an artefact of the hardware intersector with no Embree counterpart.
bool IntersectSceneExcludingShaded(const EmbreeSceneData& sceneData,
                                   const Ray& ray,
                                   float minT,
                                   const ShadowExclusion& exclusion,
                                   HitInfo& outHit,
                                   bool shadowQuery = true,
                                   bool preciseSpawn = gSourceTransport.enabled) {
    const float directionLength = simd::length(ray.direction);
    if (!(directionLength > 0.0f) || !std::isfinite(directionLength)) {
        return false;
    }
    const simd::float3 unitDirection = ray.direction / directionLength;
    Ray currentRay = ray;
    float currentMinT = minT;
    float base = 0.0f;
    const uint32_t attempts = std::max(exclusion.attempts, 1u);
    for (uint32_t attempt = 0u; attempt < attempts; ++attempt) {
        if (!IntersectScene(sceneData, currentRay, outHit, currentMinT)) {
            return false;
        }
        const bool excluded = exclusion.mesh != std::numeric_limits<uint32_t>::max() &&
                              outHit.primitiveType == GeometryType::Mesh && outHit.meshIndex == exclusion.mesh &&
                              (outHit.primitiveIndex == exclusion.triangle ||
                               (!preciseSpawn && shadowQuery && outHit.t * directionLength <= ShadowSpawnMinT()));
        if (!excluded) {
            outHit.t += base;
            return true;
        }
        if (preciseSpawn) {
            // Metal's precise spawn (visibility.metal, a ray whose tMin is the source epsilon): keep the ray's origin
            // and line and resume one float past the excluded triangle, so no world interval is skipped.
            currentMinT = std::nextafter(std::max(outHit.t, currentMinT), std::numeric_limits<float>::infinity());
            continue;
        }
        base += outHit.t + kShadowSkipStep / directionLength;
        currentRay.origin = outHit.position + unitDirection * kShadowSkipStep;
        currentMinT = 0.0f;
    }
    return false;
}

// Metal's source_opacity (material_access.metal): the material's opacity times the first channel of its opacity
// texture, read at level 0 at uv0 through the source texture transform, saturated; 1 without source opacity. The texture
// is read on triangles only.
float SourceOpacity(const EmbreeSceneData& sceneData, const PathTracerShaderTypes::MaterialData& material,
                    const HitInfo& hit) {
    if (!pt_shared::source_has_opacity(material)) return 1.0f;
    float alpha = material.textureControl.x;
    const auto texture = sceneData.materialTextures.find(material.textureIndices1.z);
    if (texture != sceneData.materialTextures.end() && hit.primitiveType == GeometryType::Mesh) {
        const simd::float2 uv = ApplyTextureTransform(hit.uv, material.textureTransform0, material.textureTransform1);
        alpha *= SampleCpuTextureLevel(texture->second, uv, 0u).x;
    }
    return std::clamp(alpha, 0.0f, 1.0f);
}

// Metal's source_height_derivatives (material_access.metal), after Tungsten's BitmapTexture::derivatives: two-texel
// differences of the height texture's first channel, read at level 0 without filtering and wrapped, interpolated
// bilinearly, with no division by two or by the texel size. xy truncates toward zero, as Tungsten's does, also at the
// repeat seam.
simd::float2 SourceHeightDerivatives(const CpuTexture& texture, const simd::float2& uv) {
    const CpuTexture::MipLevel& level = texture.mipLevels.front();
    const int width = static_cast<int>(level.width);
    const int height = static_cast<int>(level.height);
    const auto texel = [&](int x, int y) {
        x = (x % width + width) % width;
        y = (y % height + height) % height;
        return CpuTextureTexel(texture, level, (static_cast<size_t>(y) * static_cast<size_t>(width) +
                                                static_cast<size_t>(x)) * 4ull).x;
    };
    const float fx = uv.x * static_cast<float>(width) - 0.5f;
    const float fy = uv.y * static_cast<float>(height) - 0.5f;
    const int px = static_cast<int>(fx);
    const int py = static_cast<int>(fy);
    const float tx = fx - static_cast<float>(px);
    const float ty = fy - static_cast<float>(py);
    simd::float2 gradients[4];
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            const int qx = px + x;
            const int qy = py + y;
            gradients[y * 2 + x] = simd_make_float2(texel(qx + 1, qy) - texel(qx - 1, qy),
                                                    texel(qx, qy + 1) - texel(qx, qy - 1));
        }
    }
    const auto mix = [](simd::float2 a, simd::float2 b, float t) { return a + (b - a) * t; };
    return mix(mix(gradients[0], gradients[1], tx), mix(gradients[2], gradients[3], tx), ty);
}

// Metal's source_bump_normal (material_access.metal), after Tungsten: the height texture (textureIndices1.w, scaled by
// bumpScale, textureControl.z) perturbs the authored normal in the triangle's uv tangent frame, before the frame is
// faced. Tungsten's tangentSpace normalizes the frame's numerators without dividing by the uv determinant. Triangles only;
// a degenerate uv frame or result keeps the normal.
simd::float3 SourceBumpNormal(const EmbreeSceneData& sceneData, const PathTracerShaderTypes::MaterialData& material,
                              const HitInfo& hit, simd::float3 ns) {
    if (material.textureControl.z == 0.0f || hit.primitiveType != GeometryType::Mesh || !hit.hasSourceBumpFrame) {
        return ns;
    }
    const auto texture = sceneData.materialTextures.find(material.textureIndices1.w);
    if (texture == sceneData.materialTextures.end() || texture->second.mipLevels.empty()) {
        return ns;
    }
    const simd::float3 facedNs = ns;
    const bool backFace = !hit.frontFace;
    if (backFace) ns = -ns;
    const simd::float2 e1 = hit.sourceBumpE1;
    const simd::float2 e2 = hit.sourceBumpE2;
    if (std::fabs(e1.x * e2.y - e2.x * e1.y) < 1.0e-6f) return facedNs;
    simd::float3 t = simd::normalize(hit.sourceBumpQ1 * e2.y - e1.y * hit.sourceBumpQ2);
    simd::float3 b = simd::normalize(hit.sourceBumpQ2 * e1.x - e2.x * hit.sourceBumpQ1);
    const simd::float2 dh = SourceHeightDerivatives(
        texture->second, ApplyTextureTransform(hit.uv, material.textureTransform0, material.textureTransform1)) *
        material.textureControl.z;
    t += ns * (dh.x - simd::dot(ns, t));
    b += ns * (dh.y - simd::dot(ns, b));
    simd::float3 n = simd::cross(t, b);
    if (!std::isfinite(n.x) || !std::isfinite(n.y) || !std::isfinite(n.z) || simd::dot(n, n) < 1.0e-16f) {
        return facedNs;
    }
    n = simd::normalize(n);
    n = simd::dot(n, ns) < 0.0f ? -n : n;
    return backFace ? -n : n;
}

// Metal's source_forward_min_t (visibility.metal): the ray's tMin to resume on the same line just past a hit at t.
float SourceForwardMinT(const Ray& ray, float t, float epsilon) {
    const float scale = std::max(simd::length(ray.direction), 1.0e-20f);
    const float nextUlp = std::bit_cast<float>(std::bit_cast<uint32_t>(std::max(t, 0.0f)) + 1u);
    return std::max(t + epsilon / scale, nextUlp);
}

// Source visibility along a shadow ray under source transport (Metal's source shadow traversal,
// path_integrators.metal). Each source-opacity hit multiplies `transmission` by 1 - opacity, and each thin source sheet
// multiplies `rgb` by its forward transmission (source_thin_forward); both count a bounce from the shading vertex's
// depth + 1. At maxDepth bounces, or when nothing is left, the ray is blocked; otherwise it passes, as through an alpha
// hole. weight() is what the light's contribution is multiplied by.
struct SourceOpacityShadow {
    float transmission = 1.0f;
    uint32_t bounce = 0u;
    uint32_t maxDepth = 0u;
    simd::float3 rgb{1.0f, 1.0f, 1.0f};
    // Medium re-selections at pass-throughs of explicit-interior materials (Metal's source shadow traversal calls
    // interface_select_medium at every pass): the distance along the ray and the medium the rest of it is in.
    static constexpr uint32_t kMaxMediumEvents = 16u;
    uint32_t mediumEventCount = 0u;
    std::array<float, kMaxMediumEvents> mediumEventT{};
    std::array<simd::float3, kMaxMediumEvents> mediumEventSigma{};
    simd::float3 weight() const { return rgb * transmission; }
};

// Metal's trace_scene_hardware_alpha_aware_shadow (path_integrators.metal), the hardware megakernel's shadow trace: the
// first trace starts kShadowSpawnMinT out and skips the shaded triangle (`exclusion`); after an alpha-cutout hole the
// next trace starts kShadowSkipStep past the hole along the ray, with the same minimum distance and no exclusion. Up
// to kShadowAlphaSkips holes are skipped (kShadowAlphaSkipsContinuation once a hole belongs to an
// AlphaCutoutContinuation or SingleScatterSpecular material), and running out counts as occluded. Metal's
// specular-NEE and MNEE rays and its direct-light audit rays skip holes too (its alpha-aware tracers).
// maxT is in units of ray.direction; outHit.t is the blocker's distance from ray.origin in the same units.
// A precise spawn (Metal's tMin of the source epsilon or kInterfaceMinT) starts every trace SourceRayEpsilon() out and
// excludes only the spawning triangle: under source transport, from an interfacePrecision receiver (spawn_ray_min,
// wavefront_shadow_t_min), and on a source-aware shadow ray in a scene with source opacity (the alpha trace replaces
// its tMin by source_ray_epsilon). After a source-opacity pass the ray continues on its line; after an alpha-cutout
// hole it steps kShadowSkipStep past it, unless source transport is on.
bool IntersectSceneSkippingAlpha(const EmbreeSceneData& sceneData,
                                 const Ray& ray,
                                 float maxT,
                                 const PathTracerShaderTypes::MaterialData* materials,
                                 uint32_t materialCount,
                                 float debugNormalStrengthScale,
                                 ShadowExclusion exclusion,
                                 HitInfo& outHit,
                                 const bool referencePunctualShadow = false,
                                 SourceOpacityShadow* sourceOpacity = nullptr,
                                 const PathTracerShaderTypes::MaterialData* receiver = nullptr) {
    const bool preciseSpawn = InterfaceSpawn(receiver) || (sourceOpacity && gSourceTransport.opacityPresent);
    const float spawnMinT = preciseSpawn ? SourceRayEpsilon() : kShadowSpawnMinT;
    const float directionLength = simd::length(ray.direction);
    if (!(directionLength > 0.0f) || !std::isfinite(directionLength)) {
        return false;
    }
    const simd::float3 unitDirection = ray.direction / directionLength;
    Ray currentRay = ray;
    float traveled = 0.0f;
    uint32_t maxSkips = kShadowAlphaSkips;
    for (uint32_t skip = 0u; skip < maxSkips; ++skip) {
        HitInfo hit;
        if (!IntersectSceneExcludingShaded(sceneData, currentRay, spawnMinT, exclusion, hit, /*shadowQuery=*/true,
                                           preciseSpawn)) {
            return false;
        }
        traveled += hit.t;
        if (!(traveled < maxT)) {
            return false;
        }
        const PathTracerShaderTypes::MaterialData* hitMaterial =
            materials && hit.materialIndex < materialCount ? &materials[hit.materialIndex] : nullptr;
        bool sourcePass = false;
        // Metal tests a thin source sheet, then source opacity, before alpha cutouts, on every primitive. A thin sheet
        // passes whatever its alpha.
        if (sourceOpacity && hitMaterial && gSourceTransport.enabled && pt_shared::source_is_thin(*hitMaterial)) {
            sourceOpacity->rgb *= pt_shared::source_thin_forward(*hitMaterial, hit.shadingNormal, -unitDirection);
            const simd::float3 rgb = sourceOpacity->rgb;
            if (++sourceOpacity->bounce >= sourceOpacity->maxDepth || !(rgb.x > 0.0f || rgb.y > 0.0f || rgb.z > 0.0f)) {
                hit.t = traveled;
                outHit = hit;
                return true;
            }
            maxSkips = kShadowAlphaSkipsContinuation;
            sourcePass = true;
        }
        if (sourceOpacity && hitMaterial && pt_shared::source_has_opacity(*hitMaterial)) {
            if (++sourceOpacity->bounce >= sourceOpacity->maxDepth) {
                sourceOpacity->transmission = 0.0f;
            } else {
                sourceOpacity->transmission *= 1.0f - SourceOpacity(sceneData, *hitMaterial, hit);
            }
            if (!(sourceOpacity->transmission > 0.0f)) {
                hit.t = traveled;
                outHit = hit;
                return true;
            }
            maxSkips = kShadowAlphaSkipsContinuation;
            sourcePass = true;
        }
        if (!sourcePass) {
            // Metal alpha-tests shadow hits on triangles only.
            if (!materials || hit.materialIndex >= materialCount || hit.primitiveType != GeometryType::Mesh) {
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
            if (!ShadowRayAlphaDiscards(material, referencePunctualShadow, ShadowBlendAlphaSample(hit))) {
                hit.t = traveled;
                outHit = hit;
                return true;
            }
        }
        if ((materials[hit.materialIndex].materialFlags &
             (PathTracerShaderTypes::kMaterialFlagAlphaCutoutContinuation |
              PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular)) != 0u) {
            maxSkips = kShadowAlphaSkipsContinuation;
        }
        // Under source transport Metal re-selects the shadow ray's medium at every pass-through
        // (interface_select_medium): an explicit-interior material puts the rest of the ray inside its medium when the
        // ray goes against its outward normal, and outside otherwise. The caller attenuates piece by piece.
        if (sourceOpacity && gSourceTransport.enabled && hitMaterial &&
            pt_shared::source_material_flag(*hitMaterial, PathTracerShaderTypes::kMaterialFlagExplicitInterior) &&
            sourceOpacity->mediumEventCount < SourceOpacityShadow::kMaxMediumEvents) {
            const bool inside = simd::dot(unitDirection, hit.normal) < 0.0f;
            sourceOpacity->mediumEventT[sourceOpacity->mediumEventCount] = traveled;
            sourceOpacity->mediumEventSigma[sourceOpacity->mediumEventCount] =
                inside ? simd::max(DielectricSigmaA(*hitMaterial), simd_make_float3(0.0f, 0.0f, 0.0f))
                       : simd_make_float3(0.0f, 0.0f, 0.0f);
            ++sourceOpacity->mediumEventCount;
        }
        outHit = hit;
        outHit.t = traveled;
        // Under source transport, and past a source-opacity hit with or without it, Metal continues on the same line,
        // the source epsilon past the hit (source_forward_min_t in the source shadow traversal); otherwise it steps
        // kShadowSkipStep past the hole.
        const bool sameLine = gSourceTransport.enabled ||
                              (sourceOpacity && hitMaterial && pt_shared::source_has_opacity(*hitMaterial));
        const float skipStep = sameLine ? 0.0f : kShadowSkipStep;
        const float minRemaining = sameLine ? SourceRayEpsilon() : kShadowSkipStep;
        traveled += skipStep / directionLength;
        if (!((maxT - traveled) * directionLength > minRemaining)) {
            return false;
        }
        currentRay.origin = hit.position + unitDirection * skipStep;
        exclusion = ShadowExclusion{};
    }
    // Exhausting the bounded visibility walk must not clear an unseen blocker.
    return true;
}


#endif

}  // namespace

void EmbreeHeadlessRenderer::setMaxThreads(uint32_t maxThreads) {
    maxThreads_ = maxThreads;
}

bool EmbreeHeadlessRenderer::render(const HeadlessScene& scene,
                                    const HeadlessCamera&,
                                    const PathTracer::RenderSettings& requestedSettings,
                                    uint32_t sppTotal,
                                    bool verbose,
                                    bool statsMode,
                                    bool captureAuxBuffers,
                                    HeadlessRenderOutput& out,
                                    std::string& error) {
#if !defined(PATH_TRACER_ENABLE_EMBREE)
    (void)captureAuxBuffers;
    (void)scene;
    (void)requestedSettings;
    (void)sppTotal;
    (void)statsMode;
    (void)verbose;
    (void)out;
    error = "Embree backend not enabled (build with PATH_TRACER_ENABLE_EMBREE=ON)";
    return false;
#else
    if (!scene.resources) {
        error = "Embree backend requires scene resources";
        return false;
    }
    // Homogeneous volumes switch off the modes whose state carries no medium, as Metal's UniformBuilder does: ReSTIR DI
    // and the hybrid render as baseline_emissive (MetalDirectLightMode), and ReSTIR GI, ReSTIR PT, path guiding and the
    // radiance cache are off. The demotion follows the mode flag alone, as on Metal, even when the medium is empty.
    PathTracer::RenderSettings settings = requestedSettings;
    if (settings.volumeTransportMode == PathTracer::RenderSettings::VolumeTransportMode::Homogeneous) {
        if (PathTracer::RestirDiDemotedByVolume(settings)) {
            std::cout << "[EmbreeVolume] directLightMode "
                      << (settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RestirDi
                              ? "restir_di" : "restir_di_regir_hybrid")
                      << " renders as baseline_emissive under the homogeneous volume (its reservoirs carry no medium "
                      << "state), as on Metal" << std::endl;
        }
        settings.directLightMode = PathTracer::MetalDirectLightMode(settings);
        const bool modesOn = settings.restirGiMode != PathTracer::RenderSettings::RestirGiMode::Off ||
                             settings.restirPtMode != PathTracer::RenderSettings::RestirPtMode::Off ||
                             settings.pathGuidingMode != PathTracer::RenderSettings::PathGuidingMode::Off ||
                             settings.radianceCacheMode != PathTracer::RenderSettings::RadianceCacheMode::Off;
        if (modesOn) {
            std::cout << "[EmbreeVolume] ReSTIR GI, ReSTIR PT, path guiding and the radiance cache are off under the "
                      << "homogeneous volume, as on Metal" << std::endl;
        }
        settings.restirGiMode = PathTracer::RenderSettings::RestirGiMode::Off;
        settings.restirPtMode = PathTracer::RenderSettings::RestirPtMode::Off;
        settings.pathGuidingMode = PathTracer::RenderSettings::PathGuidingMode::Off;
        settings.radianceCacheMode = PathTracer::RenderSettings::RadianceCacheMode::Off;
    }
    gSourceTransport.enabled = settings.sourceTransport;
    gSourceTransport.epsilon = settings.sourceRayEpsilon > 0.0f ? settings.sourceRayEpsilon : 1.0e-6f;
    if (settings.sourceTransport) {
        std::ostringstream epsilonText;
        epsilonText << gSourceTransport.epsilon;
        std::cout << "[SourceTransport] Embree: interface-origin spawns, tMin " << epsilonText.str()
                  << ", shadow end margin 1e-6, shadow rays attenuated by the medium they start in, no light "
                  << "sampled at the last vertex, specular NEE off (as Metal's wavefront)" << std::endl;
        if (settings.enableMnee) {
            std::cerr << "Embree warning: enableMnee is ignored under sourceTransport, as on Metal's wavefront, which "
                      << "has no bounce-level MNEE; every delta bounce is unpaired." << std::endl;
        }
    }
    gSourceTransport.opacityPresent = false;
    uint32_t interfacePrecisionMaterials = 0u;
    if (const auto* sourceMaterials = scene.resources->materialsData()) {
        for (uint32_t i = 0; i < scene.resources->materialCount(); ++i) {
            gSourceTransport.opacityPresent |= sourceMaterials[i].textureControl.y != 0.0f;
            interfacePrecisionMaterials +=
                pt_shared::source_material_flag(sourceMaterials[i], PathTracerShaderTypes::kMaterialFlagInterfacePrecision)
                    ? 1u : 0u;
        }
    }
    if (!settings.sourceTransport && (gSourceTransport.opacityPresent || interfacePrecisionMaterials > 0u)) {
        std::cout << "[SourceSpawn] Embree without sourceTransport, as Metal's wavefront:";
        if (gSourceTransport.opacityPresent) {
            std::cout << " source opacity present, so shadow rays start 1e-6 out excluding only the shaded triangle, "
                      << "pass opacity hits on their line weighted by 1 - opacity, and path rays pass them with "
                      << "probability 1 - opacity;";
        }
        if (interfacePrecisionMaterials > 0u) {
            std::cout << " " << interfacePrecisionMaterials << " interfacePrecision material(s) spawn bounce and shadow "
                      << "rays at the interface origin with tMin 1e-6;";
        }
        std::cout << " other spawns unchanged." << std::endl;
    }

    const EmbreeVolumeMedium volumeMedium = MakeEmbreeVolumeMedium(settings);
    if (settings.volumeTransportMode == PathTracer::RenderSettings::VolumeTransportMode::Homogeneous) {
        if (!volumeMedium.enabled) {
            std::cout << "[EmbreeVolume] homogeneous volume has no extinction (sigma_a + sigma_s is 0): no medium, as "
                      << "on Metal" << std::endl;
        } else {
            std::cout << "[EmbreeVolume] homogeneous medium: sigma_a " << volumeMedium.sigmaA.x << ","
                      << volumeMedium.sigmaA.y << "," << volumeMedium.sigmaA.z << ", sigma_s " << volumeMedium.sigmaS.x
                      << "," << volumeMedium.sigmaS.y << "," << volumeMedium.sigmaS.z << ", emission "
                      << volumeMedium.emission.x << "," << volumeMedium.emission.y << "," << volumeMedium.emission.z
                      << "; segments that end at a hit are attenuated by exp(-sigma_t t) and gain the medium's "
                      << "emission, and surface NEE to lights with a position by exp(-sigma_t d)" << std::endl;
            if (volumeMedium.scatters) {
                std::cout << "[EmbreeVolume] scattering: free flights in pbrt-v3's channel mixture, at most "
                          << volumeMedium.maxScatteringEvents << " scatter events per path, phase "
                          << (volumeMedium.henyeyGreenstein ? "Henyey-Greenstein g " : "isotropic")
                          << (volumeMedium.henyeyGreenstein ? std::to_string(volumeMedium.anisotropy) : std::string())
                          << ", rect-light NEE at scatter points " << (volumeMedium.nee ? "on" : "off")
                          << " (MIS against the phase-sampled hit), as on Metal" << std::endl;
            }
        }
    }
    const EmbreeSpectralConfig spectralConfig = MakeEmbreeSpectralConfig(settings);
    if (spectralConfig.enabled) {
        std::cout << "[EmbreeSpectral] hero-wavelength transport: " << spectralConfig.channelCount
                  << " wavelength(s) per path over " << spectralConfig.minNm << "-" << spectralConfig.maxNm
                  << " nm, white balance " << spectralConfig.whiteBalance.x << "," << spectralConfig.whiteBalance.y
                  << "," << spectralConfig.whiteBalance.z << ", dispersion strength "
                  << spectralConfig.dispersionStrength << " (collapse to the hero wavelength at dispersive vertices), "
                  << "as on Metal" << std::endl;
    }
    if (settings.causticTransportMode == PathTracer::RenderSettings::CausticTransportMode::PathSpacePrototype)
        std::cerr << "Embree warning: causticTransportMode=path_space is ignored; path-space caustic connections are not implemented." << std::endl;
    if (settings.svgfDenoiseEnabled)
        std::cerr << "Embree warning: svgfDenoise is ignored; SVGF reconstruction is not implemented." << std::endl;
    if (settings.mlBackendMode == PathTracer::RenderSettings::MlBackendMode::ReconstructionTiny)
        std::cerr << "Embree warning: mlBackendMode=reconstruction_tiny is ignored; tiny ML reconstruction is not implemented." << std::endl;
    if (settings.enableMnee && settings.enableMneeSecondary)
        std::cerr << "Embree warning: enableMneeSecondary is ignored; secondary environment delta chains are not implemented." << std::endl;
    // The RIS modes run Metal's plain RIS (directLightMode=ris) at every non-delta vertex: M = risCandidateCount
    // candidates over the suns and the mesh emitters, resampled by the shared direct-light target, the winner shaded
    // with W and balanced against BSDF hits on mesh emitters as M draws of the proposal (Metal's
    // mesh_emitter_nee_mis_weight / mesh_emitter_bsdf_hit_weight for plain RIS). The reuse of ris_spatial,
    // ris_temporal, ris_world and ris_regir (neighbour, previous-frame, world-cell and ReGIR-cache reservoirs) is not
    // implemented: those modes render plain RIS, which has the same expectation, and the log says so.
    const bool risDirectMode =
        settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::Ris ||
        settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RisSpatialReuse ||
        settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RisTemporalReuse ||
        settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RisWorldReuse ||
        settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RisRegirCache;
    if (risDirectMode) {
        const char* reuse =
            settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RisSpatialReuse ? "ris_spatial"
            : settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RisTemporalReuse ? "ris_temporal"
            : settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RisWorldReuse ? "ris_world"
            : settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RisRegirCache ? "ris_regir"
                                                                                                    : nullptr;
        std::cout << "[EmbreeRis] plain RIS at every non-delta vertex, " << std::max(settings.risCandidateCount, 1u)
                  << " candidates over the suns and mesh emitters, as Metal's directLightMode=ris" << std::endl;
        if (reuse) {
            std::cerr << "Embree warning: directLightMode=" << reuse << " renders plain RIS: its reservoir reuse is not "
                      << "implemented on Embree (same expectation, more noise)." << std::endl;
        }
    }
    if (settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RestirDiRegirHybrid)
        std::cerr << "Embree warning: directLightMode=restir_di_regir_hybrid uses ordinary ReSTIR DI; the ReGIR component is not implemented." << std::endl;
    // Metal headless switches ReSTIR DI to its explicit screen-space passes when restirDebug is on, unless
    // PT_RESTIR_DI_VALIDATE_FUSED_REFERENCE is set or a homogeneous volume turns ReSTIR DI off (RenderLoop.mm,
    // ExplicitRestirDiEnabled and HeadlessFusedRestirDiReferenceEnabled). Embree implements only the fused path.
    if (settings.restirDebugEnabled &&
        (settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RestirDi ||
         settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RestirDiRegirHybrid) &&
        settings.volumeTransportMode != PathTracer::RenderSettings::VolumeTransportMode::Homogeneous &&
        !MetalEnvFlagEnabled("PT_RESTIR_DI_VALIDATE_FUSED_REFERENCE"))
        std::cerr << "Embree warning: restirDebug selects Metal's explicit screen-space ReSTIR DI passes; Embree renders "
                     "the fused per-sample path (Metal's default) instead, so the two backends can differ." << std::endl;

    if (settings.restirPtMode == PathTracer::RenderSettings::RestirPtMode::EnhancedUnbiased ||
        settings.restirPtMode == PathTracer::RenderSettings::RestirPtMode::EnhancedBiased)
        std::cerr << "Embree warning: enhanced ReSTIR PT modes use ordinary capture/reuse; enhanced feature measurements are unavailable and remain zero." << std::endl;

    if (settings.debugSpecularOnly) std::cerr << "Embree warning: debugSpecularOnly is ignored; material shading is unchanged." << std::endl;

    const uint32_t width = settings.renderWidth > 0 ? settings.renderWidth : 1280u;
    const uint32_t height = settings.renderHeight > 0 ? settings.renderHeight : 720u;

    LogLoadMemory("scene loaded, before the Embree build");
    EnvironmentMap envMap;
    EnvironmentMap* envPtr = nullptr;
    if (!settings.environmentMapPath.empty()) {
        if (!LoadEnvironmentMap(settings.environmentMapPath, settings.environmentSunClamp, envMap)) {
            error = "Failed to load environment map for Embree backend: " + settings.environmentMapPath;
            return false;
        }
        envPtr = &envMap;
    }

    LogLoadMemory("environment loaded");
    EmbreeSceneData sceneData;
    if (!BuildEmbreeScene(*scene.resources, sceneData, error)) {
        return false;
    }

    const auto* materials = scene.resources->materialsData();
    const uint32_t materialCount = scene.resources->materialCount();
    if (settings.sssMode == PathTracer::RenderSettings::SssMode::Separable && materials) {
        const uint32_t separable = static_cast<uint32_t>(std::count_if(materials, materials + materialCount,
            [](const PathTracerShaderTypes::MaterialData& m) {
                return static_cast<uint32_t>(m.typeEta.x) ==
                           static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Subsurface) &&
                       m.sssParams.y < 0.5f;
            }));
        if (separable > 0u) {
            std::cerr << "Embree warning: sss=separable: Metal's separable diffusion profile is not implemented; "
                      << separable << (separable == 1u ? " material takes" : " materials take")
                      << " its Lambert fallback instead." << std::endl;
        }
    }
    const auto* rectangles = scene.resources->rectanglesData();
    const uint32_t rectangleCount = scene.resources->rectangleCount();
    const float emissionScale = LoadEmbreeEmissionScale();
    const auto& emissivePrimitives = scene.resources->emissivePrimitives();

    std::vector<RectLightInfo> rectLights;
    std::vector<int32_t> rectLightIndexByRect;
    uint32_t rectLightsWithEnvEmissionIgnored = 0;
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
            info.isReferencePoint = AreaPrimitiveIsReferencePoint(rect);
            info.inverseSquareFalloff = info.isReferencePoint && rect.materialTwoSided.w != 0u;
            info.jitterRadius = std::max(rect.corner.w, 0.0f);
            info.legacyMinimumDistanceSq = info.isReferencePoint ? rect.edgeU.w : 0.0f;
            info.baseEmission = baseEmission;
            // Disable emitEnv for Embree rect lights to match SWRT energy during parity checks.
            if (material.emission.w > 0.5f) {
                ++rectLightsWithEnvEmissionIgnored;
            }
            info.emissionUsesEnv = false;
            info.area = AreaPrimitiveArea(rect);

            rectLightIndexByRect[i] = static_cast<int32_t>(rectLights.size());
            rectLights.push_back(info);
        }
    }
    if (rectLightsWithEnvEmissionIgnored > 0u) {
        std::cerr << "Embree warning: emitEnv=1 is ignored when sampling " << rectLightsWithEnvEmissionIgnored
                  << " rect light(s); light sampling uses the base emission, while direct hits still read the "
                     "environment map when one is loaded." << std::endl;
    }
    const uint32_t rectLightCount = static_cast<uint32_t>(rectLights.size());
    const uint32_t stochasticRectCount = settings.referencePointExactNee
        ? CountAnalyticLights(rectLights, true) : rectLightCount;
    const bool exactPointNeeActive = settings.referencePointExactNee && stochasticRectCount < rectLightCount;
    const uint32_t rectNeeLightCount = exactPointNeeActive ? stochasticRectCount : rectLightCount;


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
            info.cosDelta = primitive.directionalCosDelta;
            directionalLights.push_back(info);
        }
    }

    std::vector<MeshLightInfo> meshLights;
    {
        float totalMeshLightPower = 0.0f;
        const auto& meshes = scene.resources->meshes();
        std::vector<uint32_t> meshTriangleOffsets(meshes.size(), 0u);
        uint32_t triangleOffset = 0u;
        for (size_t i = 0u; i < meshes.size(); ++i) {
            meshTriangleOffsets[i] = triangleOffset;
            triangleOffset += static_cast<uint32_t>(meshes[i].indices.size() / 3u);
        }
        meshLights.reserve(emissivePrimitives.size());
        for (const auto& primitive : emissivePrimitives) {
            if ((primitive.flags & PathTracerShaderTypes::kLightPrimitiveFlagDirectional) != 0u) {
                continue;
            }
            simd::float3 emission = primitive.emission * emissionScale;
            const float emissionLuma = Luminance(emission);
            const bool texturedEmission =
                (primitive.flags & PathTracerShaderTypes::kLightPrimitiveFlagTexturedEmission) != 0u;
            const float power = texturedEmission ? primitive.power * emissionScale
                                                   : primitive.area * emissionLuma;
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
            info.twoSided = (primitive.flags & PathTracerShaderTypes::kLightPrimitiveFlagTwoSided) != 0u;
            info.texturedEmission = texturedEmission;
            if (texturedEmission) {
                const uint32_t meshIndex = primitive.reserved1;
                if (meshIndex >= meshes.size() || primitive.triangleIndex < meshTriangleOffsets[meshIndex] ||
                    materialCount == 0u) continue;
                info.texturedMesh = &meshes[meshIndex];
                info.texturedMaterial = &materials[std::min(info.texturedMesh->materialIndex, materialCount - 1u)];
                info.localTriangle = primitive.triangleIndex - meshTriangleOffsets[meshIndex];
            }
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

    const bool baselineDirectLight =
        settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::BaselineEmissive;
    const bool sssRandomWalk = settings.sssMode == PathTracer::RenderSettings::SssMode::RandomWalk;
    SssWalkStats sssWalkStats;
    // UniformBuilder's hardwareExcludeMaxAttempts: how often one trace may skip an excluded hit (a shadow ray's shaded
    // triangle).
    const uint32_t hardwareExcludeAttempts = std::min(settings.hardwareExcludeRetries, 3u) + 1u;
    // baseline_emissive draws one mesh emitter per non-delta vertex from every emitter, as Metal's
    // sample_direct_light_baseline does on both executors. Before EP8e Embree drew only two-sided or textured emitters
    // and reached plain one-sided ones by BSDF hits alone.
    const bool meshNeeBaseline = baselineDirectLight && !meshLights.empty();
    // Plain RIS's mesh candidates (CpuBuildRestirDiProviderReservoirForHit with M = risCandidateCount): a BSDF hit on a
    // mesh emitter pairs with M draws of the proposal, the mesh provider's share times the baseline mesh pdf.
    const bool risMeshNee = risDirectMode && !meshLights.empty();
    const uint32_t risCandidates = std::max(settings.risCandidateCount, 1u);
    const float risMeshDrawScale = risMeshNee
        ? static_cast<float>(risCandidates) /
              static_cast<float>((directionalLights.empty() ? 0u : 1u) + 1u)
        : 1.0f;
    const bool meshNeeMis = meshNeeBaseline || risMeshNee;
    if (meshNeeBaseline) {
        size_t twoSidedCount = 0u, texturedCount = 0u;
        for (const MeshLightInfo& l : meshLights) {
            twoSidedCount += l.twoSided ? 1u : 0u;
            texturedCount += l.texturedEmission ? 1u : 0u;
        }
        std::cout << "[EmbreeLights] baseline_emissive NEE over " << meshLights.size() << " mesh emitter triangles ("
                  << twoSidedCount << " two-sided, " << texturedCount << " textured)" << std::endl;
    }
    const bool envMapAvailable = envPtr && envPtr->width > 0 && envPtr->height > 0 && !envPtr->rgba.empty();
    const bool envSampling = envPtr && envPtr->hasDistribution && envMapAvailable;
    const bool restirDirectMode =
        settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RestirDi ||
        settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RestirDiRegirHybrid;
    // As in Metal, directLightMode=legacy samples rect lights and the environment only: analytic `sun` lines are lit
    // by baseline_emissive (sample_direct_light_baseline) and the RIS/ReSTIR providers. Gate on the requested mode,
    // so the RIS modes, which Embree runs through its legacy path, keep their directional NEE. Under homogeneous
    // volumes ReSTIR DI already reads baseline_emissive here (settings is demoted at the top of render, as on Metal).
    const bool directionalNeeMode =
        settings.directLightMode != PathTracer::RenderSettings::DirectLightMode::LegacyRect;
    const FireflyClampParams clampParams = MakeFireflyParams(settings);

    CameraBasis camera = BuildCamera(settings, width, height);
    const RayCone primaryCone = MakePrimaryRayCone(camera, width, height);
    const PrimaryRayDiff primaryRayDiff = MakePrimaryRayDiff(camera, width, height);
    DirectDebugConfig debugCfg = LoadDirectDebugConfig(settings, width, height);

    std::vector<float> linearRGB(width * height * 3u, 0.0f);
    const bool collectAlbedoGuide = captureAuxBuffers || (settings.denoiseEnabled && settings.denoiseUseAlbedo);
    const bool collectNormalGuide = captureAuxBuffers || (settings.denoiseEnabled && settings.denoiseUseNormal);
    const bool showMaterial = settings.debugShowBaseColor || settings.debugShowMetallic ||
                              settings.debugShowRoughness || settings.debugShowAO;
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
    std::atomic<uint64_t> volumeScatterEventCount{0};
    std::atomic<uint64_t> spectralCollapseCount{0};
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

    constexpr uint32_t kCpuPathReservoirDepthCount = 8u;
    const size_t pixelCount = static_cast<size_t>(width) * height;
    const bool restirGiActive =
        settings.restirGiMode == PathTracer::RenderSettings::RestirGiMode::ProductionDiffuse;
    const bool restirPtActive =
        settings.restirPtMode == PathTracer::RenderSettings::RestirPtMode::ProductionReservoirCapture ||
        settings.restirPtMode == PathTracer::RenderSettings::RestirPtMode::ProductionPathReuse ||
        settings.restirPtMode == PathTracer::RenderSettings::RestirPtMode::EnhancedUnbiased ||
        settings.restirPtMode == PathTracer::RenderSettings::RestirPtMode::EnhancedBiased;
    const bool restirPtReuseActive =
        settings.restirPtMode == PathTracer::RenderSettings::RestirPtMode::ProductionPathReuse ||
        settings.restirPtMode == PathTracer::RenderSettings::RestirPtMode::EnhancedUnbiased ||
        settings.restirPtMode == PathTracer::RenderSettings::RestirPtMode::EnhancedBiased;
    const bool pathGuidingActive =
        settings.pathGuidingMode == PathTracer::RenderSettings::PathGuidingMode::Prototype;
    const bool radianceCacheActive = PathTracer::RadianceCacheActive(settings);
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
    static_assert(PathTracerShaderTypes::kRadianceCacheStatCount == PathTracer::RadianceCacheTraining::kStatCount,
                  "radiance cache stat layout mismatch");
    std::array<std::atomic<uint64_t>, PathTracer::RadianceCacheTraining::kStatCount> radianceCacheStats{};
    auto cacheStat = [&radianceCacheStats](uint32_t index, uint64_t value) {
        if (value != 0u) {
            radianceCacheStats[index].fetch_add(value, std::memory_order_relaxed);
        }
    };
    // Trains the cell of one record with the radiance its path ended with (radiance_cache_train_from_record and
    // radiance_cache_store_sample on Metal).
    auto trainRadianceCache = [&](const CpuRadianceCacheTrainRecord& record,
                                  const simd::float3& radianceEnd,
                                  uint32_t sampleIndex) {
        using namespace PathTracer::RadianceCacheTraining;
        cacheStat(kStatTrainRecords, 1u);
        const TargetResult result = ComputeTarget(record.throughputAtQuery,
                                                  record.radianceBaseline,
                                                  radianceEnd,
                                                  settings.radianceCacheMaxRadiance);
        if (result.status == TargetStatus::RejectedNonFinite) {
            cacheStat(kStatTrainRejectedNonFinite, 1u);
            return;
        }
        if (result.status == TargetStatus::RejectedZeroThroughput) {
            cacheStat(kStatTrainRejectedZeroThroughput, 1u);
            return;
        }
        cacheStat(kStatTargetLumaFixed, StatLumaToFixed(result.targetLuma));
        if (result.status == TargetStatus::Clamped) {
            cacheStat(kStatTrainClamped, 1u);
            cacheStat(kStatClampedLumaFixed, StatLumaToFixed(result.clampedLuma));
        }
        const float luma = PathTracer::RadianceCacheTraining::Luminance(result.target);
        std::lock_guard<std::mutex> cacheLock(radianceCacheMutexes[record.cellIndex]);
        CpuRadianceCacheCell& cell = radianceCacheCells[record.cellIndex];
        if (!cell.valid ||
            cell.invalidated ||
            cell.domainKey != record.domainKey ||
            cell.materialType != record.materialType ||
            cell.lobeType != 0u) {
            if (cell.sampleCount > 0u && cell.domainKey != record.domainKey) {
                radianceCacheInvalidCount.fetch_add(1u, std::memory_order_relaxed);
                cacheStat(kStatCellEvictions, 1u);
            }
            cell = CpuRadianceCacheCell{};
            cell.valid = true;
            cell.domainKey = record.domainKey;
            cell.materialType = record.materialType;
            cell.lobeType = 0u;
        }
        if (cell.sampleCount >= kMaxStoredSamples) {
            radianceCacheFallbackCount.fetch_add(1u, std::memory_order_relaxed);
            cacheStat(kStatTrainSaturated, 1u);
            return;
        }
        cell.sampleCount += 1u;
        cell.radianceSum += result.target;
        cell.lumaMomentSum += luma * luma;
        const float samples = static_cast<float>(std::max(cell.sampleCount, 1u));
        cell.position = record.position;
        cell.normal = record.normal;
        cell.roughness = record.roughness;
        cell.storedVariance = samples > 1.0f ? (luma * luma) / samples : 1.0f;
        cell.confidence = std::clamp(std::log2(samples + 1.0f) / 6.0f, 0.0f, 1.0f);
        cell.frameTag = sampleIndex;
        radianceCacheTrainCount.fetch_add(1u, std::memory_order_relaxed);
        cacheStat(kStatTrainAccepted, 1u);
    };

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
        uint64_t tileVolumeScatterCount = 0;
        uint64_t tileSpectralCollapseCount = 0;
        std::vector<float> tileMaxThroughputByBounce(maxThroughputByBounce.size(), 0.0f);

        for (uint32_t localY = 0; localY < tileHeight; ++localY) {
            uint32_t y = y0 + localY;
            for (uint32_t localX = 0; localX < tileWidth; ++localX) {
                uint32_t x = x0 + localX;
                simd::float3 pixelRadiance = {0.0f, 0.0f, 0.0f};
                simd::float3 pixelAlbedo = {0.0f, 0.0f, 0.0f};
                simd::float3 pixelNormal = {0.0f, 0.0f, 0.0f};
                uint32_t pixelIndex = y * width + x;
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
                    // Spectral transport: the path's wavelengths (one RNG draw, spectral mode only); channel j of the
                    // throughput carries wavelength j and inactive channels start at zero.
                    EmbreeSpectralState spectral =
                        spectralConfig.enabled ? MakeEmbreeSpectralState(spectralConfig, rng.nextFloat()) : EmbreeSpectralState{};
                    if (spectral.enabled) {
                        throughput = SpectralChannelMask(spectral);
                    }
                    float lastBsdfPdf = 1.0f;
                    // Metal's medium stack (path_integrators.metal, kMaxMediumStack): the absorption of the solid
                    // dielectrics the path is inside, innermost last.
                    constexpr uint32_t kMaxMediumStack = 8u;
                    std::array<simd::float3, kMaxMediumStack> mediumSigmaStack{};
                    uint32_t mediumDepth = 0u;
                    bool lastScatterWasDelta = true;
                    // A delta scatter without a specular connection leaves the emitter hit
                    // or environment miss as the only estimator, so it takes full weight.
                    bool lastDeltaUnpaired = false;
                    // Where the last real scatter left from, kept across alpha holes (MIS null events): the light
                    // pdf of the next emitter hit is measured from there, as that vertex's NEE was.
                    simd::float3 holeMisOrigin = simd_make_float3(0.0f, 0.0f, 0.0f);
                    bool holeMisOriginValid = false;
                    // After a source-opacity null the ray keeps its origin: the next trace starts at sourceNullMinT,
                    // and the medium absorbs from segmentStartT, the null's distance (Metal's sourceOffset).
                    float sourceNullMinT = -1.0f;
                    float segmentStartT = 0.0f;
                    bool previousMeshNeeEligible = false;
                    // ReSTIR DI estimated the primary vertex's mesh lights along the current
                    // scatter direction (restir_di / hybrid).
                    bool previousRestirDiCovered = false;
                    bool primaryVertexScattered = false;
                    // Homogeneous medium (Metal's megakernel): scatter events so far, whether the last vertex was a
                    // medium scatter, and whether it drew rect-light NEE (only then a phase-sampled rect hit is
                    // MIS-weighted; an escape after a medium scatter keeps weight 1).
                    uint32_t volumeScatteringDepth = 0u;
                    bool lastScatterWasVolume = false;
                    bool lastVolumeRectNee = false;
                    uint32_t specularDepth = 0;
                    float pathMaxRoughness = 0.0f;
                    // Radiance cache training (RadianceCacheTraining.h): one pixel in kTrainingPathStride per sample
                    // index never loads from the cache, records its cacheable vertices, and trains them when it ends.
                    const bool cacheTrainingPath =
                        radianceCacheActive && !showMaterial &&
                        (s % std::max(settings.radianceCacheTrainingInterval, 1u)) == 0u &&
                        PathTracer::RadianceCacheTraining::IsTrainingPath(pixelIndex, s);
                    std::array<CpuRadianceCacheTrainRecord, PathTracer::RadianceCacheTraining::kTrainRecordCount>
                        cacheTrainRecords{};
                    uint32_t cacheTrainValidMask = 0u;
                    // postClampScale multiplies after the firefly clamp: Metal clamps a light sample when it queues
                    // the shadow ray and applies the source shadow transmittance in the shadow kernel.
                    auto addContribution = [&](const simd::float3& contribution,
                                               const simd::float3& postClampScale = simd_make_float3(1.0f, 1.0f, 1.0f)) {
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
                        safeContribution *= postClampScale;
                        if (!IsFinite(safeContribution)) {
                            ++tileRadianceNanInfCount;
                            return;
                        }
                        radiance += safeContribution;
                    };

                    // Metal's path trace (trace_path_hardware): every path ray starts kShadowSpawnMinT (Metal's
                    // kEpsilon, interfaceMinT) out, in world units, and skips the previous hit's triangle, a cutout
                    // hole's included (prevRec, compute_exclusion_indices), with trace_scene_hardware's re-trace.
                    HitInfo previousHit;
                    bool previousHitValid = false;
                    bool guidePending = true;  // the denoiser guides of this sample are not taken yet
                    // Metal's camera ray and its PBR alpha pass-through ray keep kEpsilon (1e-3) as tMin under source
                    // transport too (kernels.metal); bounce rays take the source epsilon.
                    bool legacyRaySpawn = true;
                    // A precise spawn (Metal's source-epsilon tMin) excludes only the previous triangle: a bounce from an
                    // interface spawn (InterfaceSpawn), and the continuation of a source null.
                    bool preciseRaySpawn = false;
                    for (uint32_t depth = 0; depth < settings.maxDepth; ++depth) {
                        ++tilePathSegmentCount;
                        HitInfo hit;
                        const float legacyPathMinT = kShadowSpawnMinT / std::max(simd::length(ray.direction), 1.0e-8f);
                        const float pathMinT = sourceNullMinT >= 0.0f ? sourceNullMinT
                                             : legacyRaySpawn ? legacyPathMinT
                                                              : (preciseRaySpawn ? SourceRayEpsilon() : kShadowSpawnMinT) /
                                                                    std::max(simd::length(ray.direction), 1.0e-8f);
                        sourceNullMinT = -1.0f;
                        bool hasHit = previousHitValid
                            ? IntersectSceneExcludingShaded(sceneData, ray, pathMinT,
                                                            ShadedTriangleExclusion(previousHit, hardwareExcludeAttempts),
                                                            hit, /*shadowQuery=*/false,
                                                            /*preciseSpawn=*/preciseRaySpawn && !legacyRaySpawn)
                            : IntersectScene(sceneData, ray, hit, pathMinT);
                        float intersectionMinT = kEpsilon;
                        for (uint32_t alphaSkip = 0u; hasHit && alphaSkip < 64u; ++alphaSkip) {
                            if (!materials || hit.primitiveType != GeometryType::Mesh ||
                                hit.materialIndex >= materialCount) break;
                            const auto& cutoutMaterial = materials[hit.materialIndex];
                            if ((cutoutMaterial.materialFlags &
                                 (PathTracerShaderTypes::kMaterialFlagAlphaCutoutContinuation |
                                  PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular)) == 0u ||
                                cutoutMaterial.pbrExtras.w <= 0.5f || cutoutMaterial.pbrExtras.w >= 1.5f) break;
                            if (depth == 0u) {
                                PopulateFirstHitUvGradients(hit, ray, primaryRayDiff,
                                    (cutoutMaterial.materialFlags &
                                     PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular) != 0u);
                            }
                            const float footprint = SurfaceFootprintFromCone(
                                RayConeWidthAtDistance(rayCone, hit.t), hit.normal, -ray.direction);
                            const auto resolvedCutout = ShadeMaterialForHit(sceneData, cutoutMaterial, hit,
                                settings.debugNormalStrengthScale, footprint, settings.normalVarianceRoughness,
                                settings.textureLodBias);
                            if (!PathRayAlphaDiscards(resolvedCutout, rng)) break;
                            // Metal traces again from the hole (advance_pavilion_alpha_visibility_ray) kEpsilon (1e-3)
                            // past it, under source transport too, so nothing within that distance is reported.
                            const float nextMinT = hit.t + legacyPathMinT;
                            if (!std::isfinite(nextMinT) || nextMinT <= intersectionMinT) break;
                            intersectionMinT = nextMinT;
                            // Preserve origin, t, cone/differentials, depth, PDF,
                            // and RNG: a MASK hole is visibility, not scattering.
                            hasHit = IntersectScene(sceneData, ray, hit, intersectionMinT);
                        }
                        if (!hasHit) {
                            if (showMaterial) break;
                            simd::float3 background =
                                SpectralRadianceToChannels(EvaluateBackground(settings, envPtr, ray.direction), spectral);
                            float misWeight = 1.0f;
                            bool useSpecularMis = depth > 0u &&
                                                  ((!lastScatterWasDelta) ||
                                                   (!lastDeltaUnpaired &&
                                                    (settings.enableSpecularNee || settings.enableMnee)));
                            // A medium scatter point never samples the environment, so its escape keeps weight 1.
                            if (useSpecularMis && envSampling && !lastScatterWasVolume) {
                                float lightPdf = EnvironmentPdf(*envPtr, settings.environmentRotation, ray.direction);
                                float denom = lastBsdfPdf + lightPdf;
                                if (denom > 0.0f) {
                                    misWeight = lastBsdfPdf / denom;
                                }
                                misWeight = std::clamp(misWeight,
                                                       kMisWeightClampMin,
                                                       kMisWeightClampMax);
                            }
                            simd::float3 contribution = background * misWeight;
                            if (depth == 0u && envMapAvailable &&
                                settings.backgroundMode == PathTracer::RenderSettings::BackgroundMode::Environment &&
                                settings.environmentBackgroundScale != 1.0f) {
                                contribution *= std::isfinite(settings.environmentBackgroundScale)
                                    ? std::max(settings.environmentBackgroundScale, 0.0f) : 1.0f;
                            }
                            addContribution(contribution);
                            break;
                        }
                        previousHit = hit;
                        previousHitValid = true;

                        if (!materials || materialCount == 0) {
                            break;
                        }

                        // The global homogeneous medium over the segment that ends at this hit (Metal's volume block
                        // in the path loop): its emission, then its transmittance. The segment starts where the last
                        // one ended (segmentStartT after a source-opacity null event), so a hole passed on the way
                        // splits it without counting any length twice.
                        if (volumeMedium.enabled) {
                            const float volumeSegment = std::max(hit.t - segmentStartT, 0.0f);
                            const simd::float3 segmentEmission = VolumeEmissionIntegral(volumeMedium, volumeSegment);
                            if (segmentEmission.x > 0.0f || segmentEmission.y > 0.0f || segmentEmission.z > 0.0f) {
                                addContribution(segmentEmission);
                            }
                            // A scattering medium draws a free flight while the path has scatter events left; past
                            // the segment it weights by T(s) / mean(T(s)), else by T(s) (Metal's volume block).
                            bool sampledFreeFlight = false;
                            simd::float3 freeFlightWeight = simd_make_float3(1.0f, 1.0f, 1.0f);
                            if (volumeMedium.scatters && volumeScatteringDepth < volumeMedium.maxScatteringEvents) {
                                const VolumeFreeFlight flight = VolumeSampleFreeFlight(volumeMedium, volumeSegment, rng);
                                sampledFreeFlight = true;
                                freeFlightWeight = flight.weight;
                                if (flight.scattered) {
                                    ++tileVolumeScatterCount;
                                    const simd::float3 scatterPoint =
                                        ray.origin + ray.direction * (segmentStartT + flight.distance);
                                    throughput *= flight.weight;
                                    // Volume NEE: rect lights only (and reference points, weight 1), MIS-weighted
                                    // against the phase-sampled hit of the same light. Shadow rays skip alpha
                                    // cutouts and exclude nothing (a scatter point is not on a surface); they leave
                                    // 1e-4 out (Metal's kRayOriginEpsilon) and stop 1e-3 short of the light.
                                    const bool volumeRectNee = !rectLights.empty() && volumeMedium.nee;
                                    if (volumeRectNee) {
                                        HitInfo volumeHit = hit;
                                        volumeHit.position = scatterPoint;
                                        const uint32_t neeDrawCount = exactPointNeeActive ? rectLightCount + 1u : 1u;
                                        for (uint32_t neeDraw = 0u; neeDraw < neeDrawCount; ++neeDraw) {
                                            RectLightSample lightSample;
                                            if (!SampleRectLightNeeDraw(rectLights, envPtr, settings, volumeHit, rng,
                                                                        lightSample, neeDraw, exactPointNeeActive)) {
                                                continue;
                                            }
                                            Ray shadowRay;
                                            shadowRay.origin = scatterPoint + lightSample.direction * 1.0e-4f;
                                            shadowRay.direction = lightSample.direction;
                                            const float shadowMax =
                                                std::max(lightSample.distance - kShadowSpawnMinT, kShadowSpawnMinT);
                                            HitInfo shadowHit;
                                            ++tileShadowRayCount;
                                            const bool occluded =
                                                IntersectSceneSkippingAlpha(sceneData, shadowRay, shadowMax, materials,
                                                                            materialCount,
                                                                            settings.debugNormalStrengthScale,
                                                                            ShadowExclusion{}, shadowHit);
                                            if (!occluded && lightSample.pdf > 0.0f) {
                                                const float phase =
                                                    VolumePhaseEval(volumeMedium, -ray.direction, lightSample.direction);
                                                const float misWeight = lightSample.isReferencePoint
                                                    ? 1.0f : VolumeNeeMisWeight(lightSample.pdf, phase);
                                                addContribution(lightSample.emission * phase *
                                                                VolumeTransmittance(volumeMedium, shadowMax) *
                                                                (misWeight / std::max(lightSample.pdf, 1.0e-6f)));
                                            }
                                        }
                                    }
                                    float phasePdf = 1.0f;
                                    const simd::float3 phaseDir = VolumePhaseSample(volumeMedium, -ray.direction, rng, phasePdf);
                                    ++volumeScatteringDepth;
                                    lastScatterWasDelta = false;
                                    lastDeltaUnpaired = false;
                                    previousMeshNeeEligible = false;
                                    previousRestirDiCovered = false;
                                    primaryVertexScattered = true;
                                    lastBsdfPdf = std::max(phasePdf, 1.0e-6f);
                                    lastScatterWasVolume = true;
                                    lastVolumeRectNee = volumeRectNee;
                                    legacyRaySpawn = false;
                                    preciseRaySpawn = false;
                                    holeMisOriginValid = false;
                                    previousHitValid = false;
                                    segmentStartT = 0.0f;
                                    ray.origin = scatterPoint + phaseDir * 1.0e-4f;
                                    ray.direction = phaseDir;
                                    continue;
                                }
                            }
                            throughput *= sampledFreeFlight ? freeFlightWeight
                                                            : VolumeTransmittance(volumeMedium, volumeSegment);
                        }

                        // Beer-Lambert absorption over the segment inside a solid dielectric, as Metal applies it
                        // at each hit.
                        if (mediumDepth > 0u) {
                            const simd::float3 sigma = mediumSigmaStack[mediumDepth - 1u];
                            if (sigma.x > 0.0f || sigma.y > 0.0f || sigma.z > 0.0f) {
                                throughput *= Exp3(-sigma * std::max(hit.t - segmentStartT, 0.0f));
                            }
                        }
                        segmentStartT = 0.0f;

                        uint32_t matIndex = std::min(hit.materialIndex, materialCount - 1);
                        if (depth == 0u) {
                            PopulateFirstHitUvGradients(hit, ray, primaryRayDiff,
                                (materials[matIndex].materialFlags &
                                 PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular) != 0u);
                        }
                        float surfaceFootprintWorld =
                            SurfaceFootprintFromCone(RayConeWidthAtDistance(rayCone, hit.t),
                                                     hit.normal,
                                                     -ray.direction);
                        float debugMetallic, debugRoughness, debugAO;
                        PathTracerShaderTypes::MaterialData material =
                            ShadeMaterialForHit(sceneData,
                                                materials[matIndex],
                                                hit,
                                                settings.debugNormalStrengthScale,
                                                surfaceFootprintWorld,
                                                settings.normalVarianceRoughness && !settings.debugDisableNormalMap,
                                                settings.textureLodBias,
                                                settings.debugDisableOrmTexture,
                                                &debugMetallic, &debugRoughness, &debugAO);
                        // Metal's source opacity on a path ray (resolve_surface_material, and the wavefront's source
                        // null in kernels.metal): the ray passes the surface with probability 1 - opacity. As an alpha
                        // hole, the pass consumes a depth and is a null event for MIS. The ray keeps its origin and line,
                        // resumes just past the hit (source_forward_min_t) without the hit triangle, and stays in its
                        // medium, which absorbs from the hit on.
                        if (pt_shared::source_has_opacity(materials[matIndex]) &&
                            rng.nextFloat() >= SourceOpacity(sceneData, materials[matIndex], hit)) {
                            if (depth == 0u) {
                                lastScatterWasDelta = true;
                                lastDeltaUnpaired = true;
                            }
                            if (!holeMisOriginValid) {
                                holeMisOrigin = ray.origin;
                                holeMisOriginValid = true;
                            }
                            sourceNullMinT = SourceForwardMinT(ray, hit.t, SourceRayEpsilon());
                            segmentStartT = hit.t;
                            legacyRaySpawn = false;
                            preciseRaySpawn = true;
                            continue;
                        }
                        // Metal alpha-tests path rays on PBR triangles only.
                        if (hit.primitiveType == GeometryType::Mesh &&
                            static_cast<uint32_t>(material.typeEta.x) ==
                                static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness) &&
                            PathRayAlphaDiscards(material, rng)) {
                            // As on Metal, the hole consumes a depth but is a null event for MIS: the last real
                            // vertex's BSDF pdf, delta flags and origin stay, so the next emitter hit or miss takes the
                            // BSDF side of the balance heuristic whose light side that vertex's NEE (its alpha-aware
                            // shadow ray passes the same MASK hole, and a BLEND texel with the path's probability
                            // 1 - alpha) took. A hole before any scatter is unpaired.
                            if (depth == 0u) {
                                lastScatterWasDelta = true;
                                lastDeltaUnpaired = true;
                            }
                            if (!holeMisOriginValid) {
                                holeMisOrigin = ray.origin;
                                holeMisOriginValid = true;
                            }
                            ray.origin = LegacyOffsetRayOrigin(hit, ray.direction);
                            legacyRaySpawn = true;
                            continue;
                        }
                        if ((material.materialFlags & PathTracerShaderTypes::kMaterialFlagBlackBackFace) != 0u &&
                            !hit.frontFace) {
                            break;  // Single-sided back face: the reference shades it black.
                        }
                        if (showMaterial) {
                            if (settings.debugShowBaseColor) radiance = MaterialBaseColor(material);
                            else if (settings.debugShowMetallic) radiance = simd_make_float3(debugMetallic, debugMetallic, debugMetallic);
                            else if (settings.debugShowRoughness) radiance = simd_make_float3(debugRoughness, debugRoughness, debugRoughness);
                            else radiance = simd_make_float3(debugAO, debugAO, debugAO);
                            break;
                        }
                        if (settings.debugDisableAO) material.pbrParams.z = 1.0f;
                        RegularizeDielectricMaterial(material, hit.frontFace, pathMaxRoughness);
                        uint32_t type = static_cast<uint32_t>(material.typeEta.x);
                        // Metal marks a double-sided PBR triangle two-sided when it shades it (path_integrators.metal,
                        // material_access.metal), so a path that hits its back face sees its emission. Light sampling
                        // stays one-sided unless the light inventory flags the emitter two-sided, as on Metal.
                        if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness) &&
                            hit.primitiveType == GeometryType::Mesh && material.typeEta.z > 0.5f) {
                            hit.twoSided = true;
                        }
                        // Metal's thin double-sided MASK cutouts (foliage, Bistro signage): direct light reaches them from
                        // both sides (material_is_two_sided_alpha_cutout, path_integrators.metal).
                        hit.twoSidedAlphaCutout = hit.twoSided && material.typeEta.z > 0.5f &&
                            material.pbrExtras.w > 0.5f && material.pbrExtras.w < 1.5f &&
                            sceneData.materialTextures.count(material.textureIndices0.x) != 0u;
                        if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness) &&
                            (material.materialFlags & PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular) != 0u) {
                            const float ior = std::max(material.typeEta.y, 1.0f);
                            material.typeEta.y = hit.frontFace ? ior : 1.0f / ior;
                        }
                        simd::float3 incidentDir = simd::normalize(ray.direction);
                        simd::float3 wo = -incidentDir;
                        simd::float3 shadingNormal = hit.shadingNormal;
                        if (simd::dot(shadingNormal, shadingNormal) <= 0.0f) {
                            shadingNormal = hit.normal;
                        }
                        // An analytic sphere's authored dielectric normal is its outward normal (hit.normal; the hit setup
                        // faces the shading normal), as Metal uses it (EP8-f3).
                        if (hit.primitiveType == GeometryType::Spheres &&
                            pt_shared::source_material_flag(
                                material, PathTracerShaderTypes::kMaterialFlagAuthoredDielectricNormal)) {
                            shadingNormal = hit.normal;
                        }
                        // Metal (path_integrators.metal): a source material with an authored dielectric normal keeps the raw
                        // interpolated normal (not faced) and skips the glass geometric-normal override;
                        // source_dielectric_frame then picks the side and the relative IOR.
                        const bool authoredDielectricNormal = pt_shared::source_material_flag(
                            material, PathTracerShaderTypes::kMaterialFlagAuthoredDielectricNormal);
                        if (authoredDielectricNormal && hit.authoredNormalWasFaced) {
                            shadingNormal = -shadingNormal;
                        }
                        if (!authoredDielectricNormal &&
                            (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Dielectric) ||
                             MaterialUsesGeometricNormal(material))) {
                            shadingNormal = hit.frontFace ? hit.normal : -hit.normal;
                        }
                        // A source bump map perturbs this normal (resolve_wavefront_surface_state, and the
                        // megakernel's shading), after the normal choices above.
                        if (pt_shared::source_material_flag(materials[matIndex],
                                                            PathTracerShaderTypes::kMaterialFlagSourceFeatures)) {
                            shadingNormal = SourceBumpNormal(sceneData, materials[matIndex], hit, shadingNormal);
                            hit.shadingNormal = shadingNormal;
                        }
                        // A thin source sheet under source transport (resolve_wavefront_surface_state): the ray passes
                        // forward with probability p, the mean of the sheet's forward transmission, weighted forward / p,
                        // as a null event like a source-opacity pass. Otherwise the sheet only reflects, weighted
                        // F / (1 - p): wavefrontShadeKernel sets sourceFresnel.z = 2, which the shared closure reads.
                        if (gSourceTransport.enabled && pt_shared::source_is_thin(materials[matIndex])) {
                            const simd::float3 forward =
                                pt_shared::source_thin_forward(materials[matIndex], shadingNormal, wo);
                            const float forwardProbability =
                                std::clamp(simd::dot(forward, simd_make_float3(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f)),
                                           0.0f, 1.0f);
                            if (rng.nextFloat() < forwardProbability) {
                                throughput *= forward / std::max(forwardProbability, 1.0e-30f);
                                if (depth == 0u) {
                                    lastScatterWasDelta = true;
                                    lastDeltaUnpaired = true;
                                }
                                if (!holeMisOriginValid) {
                                    holeMisOrigin = ray.origin;
                                    holeMisOriginValid = true;
                                }
                                sourceNullMinT = SourceForwardMinT(ray, hit.t, SourceRayEpsilon());
                                segmentStartT = hit.t;
                                legacyRaySpawn = false;
                                preciseRaySpawn = true;
                                continue;
                            }
                            material.sourceFresnel.z = 2.0f;
                        }
                        if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
                            if (!settings.debugDisableNormalMap) shadingNormal = ApplyNormalMapForHit(sceneData,
                                                                 material,
                                                                 hit,
                                                                 shadingNormal,
                                                                 incidentDir,
                                                                 settings.debugNormalStrengthScale,
                                                                 surfaceFootprintWorld,
                                                                 settings.textureLodBias,
                                                                 settings.debugFlipNormalGreen);
                            if ((material.materialFlags & PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular) != 0u) {
                                // Tagged (Pavilion) PBR takes the path's maximum roughness, as Metal.
                                pathMaxRoughness = std::max(pathMaxRoughness,
                                                            std::clamp(material.baseColorRoughness.w, 0.0f, 1.0f));
                                material.baseColorRoughness.w = std::max(pathMaxRoughness, 0.01f);
                                material.pbrParams.y = material.baseColorRoughness.w;
                            }
                        }
                        shadingNormal = simd::normalize(shadingNormal);
                        // Metal stores the shading normal it resolved (overrides, bump, normal map) in rec.shadingNormal,
                        // and offset_ray_origin offsets the NEE and bounce rays along it. LegacyOffsetRayOrigin reads
                        // hit.shadingNormal, which still held the interpolated normal.
                        hit.shadingNormal = shadingNormal;
                        pt_shared::source_mark_closure_side(material, shadingNormal, wo);  // EP8-f2, as Metal
                        // The denoiser guides come from the first vertex, except that Metal's wavefrontShadeKernel passes
                        // them on behind a smooth source mirror (denoiseMirrorGuides) and behind a delta source dielectric
                        // (denoiseTransmissionGuides), so they describe what the colour shows through it (EP8d).
                        const bool followMirrorGuide = settings.denoiseMirrorGuides &&
                            pt_shared::source_model(material) == 1.0f && material.sourceParams.z == 0.0f;
                        const bool followTransmissionGuide = settings.denoiseTransmissionGuides &&
                            pt_shared::source_model(material) == 3.0f &&
                            (material.sourceParams.z == 0.0f || material.typeEta.w > 0.5f ||
                             material.sourceFresnel.w == 1.0f);
                        if (guidePending && !followMirrorGuide && !followTransmissionGuide) {
                            guidePending = false;
                            if (collectAlbedoGuide) {
                                pixelAlbedo += MaterialBaseColor(material);
                            }
                            if (collectNormalGuide) {
                                pixelNormal += shadingNormal;
                            }
                        }
                        // Spectral transport: the material becomes per-wavelength here, after the RGB denoiser guides,
                        // and a dispersive vertex keeps the hero wavelength only (Metal's spectral_apply_material and
                        // spectral_collapse_at_dispersive_vertex, before emission, NEE and the BSDF).
                        if (spectral.enabled) {
                            SpectralApplyMaterial(material, spectralConfig, spectral);
                            if (SpectralCollapseAtDispersiveVertex(spectral, material, spectralConfig, throughput)) {
                                ++tileSpectralCollapseCount;
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
                                                                 surfaceFootprintWorld,
                                                                 settings.textureLodBias);
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
                                    shadowRay.origin = SpawnOrigin(hit, auditLight.direction, &material);
                                    shadowRay.direction = auditLight.direction;
                                    float shadowMax =
                                        std::max(auditLight.distance - ShadowLightEndMargin(), ShadowLightEndMargin());
                                    bool blocked =
                                        IntersectSceneSkippingAlpha(sceneData,
                                                                    shadowRay,
                                                                    shadowMax,
                                                                    materials,
                                                                    materialCount,
                                                                    settings.debugNormalStrengthScale,
                                                                    ShadedTriangleExclusion(hit, hardwareExcludeAttempts),
                                                                    shadowHit,
                                                                    PavilionPunctualShadowEnabled(material, auditLight.isReferencePoint),
                                                                    nullptr, &material);
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
                                                                 clampParams,
                                                                 auditLight.isReferencePoint);
                                entry.bsdfValue = bsdfEval.value;
                                entry.bsdfPdf = bsdfEval.pdf;
                                float misWeight = 1.0f;
                                if (bsdfEval.pdf > 0.0f && !auditLight.isReferencePoint) {
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
                            // The same guards as the normal-map shading (it keeps the shading normal for a zero
                            // or non-finite tangent frame): such a frame is reported as the raw tangent with a zero
                            // bitangent, not normalized into NaN.
                            const simd::float3 rawTangent = materialMrDebug.tangent;
                            if (hit.hasTangent && IsFinite(rawTangent) && simd::dot(rawTangent, rawTangent) > 1.0e-6f) {
                                simd::float3 t = simd::normalize(rawTangent - shadingNormal * simd::dot(shadingNormal, rawTangent));
                                float tangentSign = hit.tangent.w < 0.0f ? -1.0f : 1.0f;
                                simd::float3 b = simd::normalize(simd::cross(shadingNormal, t)) * tangentSign;
                                if (IsFinite(t) && simd::dot(t, t) > 1.0e-6f && IsFinite(b) && simd::dot(b, b) > 1.0e-6f) {
                                    materialMrDebug.tangent = t;
                                    materialMrDebug.bitangent = b;
                                }
                            }
                            auto normalTexIt = sceneData.materialTextures.find(material.textureIndices0.z);
                            if (normalTexIt != sceneData.materialTextures.end() && hit.hasUv &&
                                !settings.debugDisableNormalMap) {
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
                                                           surfaceFootprintWorld,
                                                           settings.textureLodBias);
                                simd::float4 raw = normalCtx.hasIgehyGradients
                                    ? SampleCpuTextureAnisotropic(normalTexIt->second, normalCtx.uv,
                                        normalCtx.dUVdx, normalCtx.dUVdy, normalLod, settings.textureLodBias)
                                    : SampleCpuTexture(normalTexIt->second, normalCtx.uv, normalLod);
                                materialMrDebug.normalSampleRaw = simd_make_float3(raw.x, raw.y, raw.z);
                                materialMrDebug.normalScale = material.pbrParams.w;
                                materialMrDebug.normalLod = normalLod;
                                materialMrDebug.normalSampleTs =
                                    DecodeNormalMap(materialMrDebug.normalSampleRaw, materialMrDebug.normalScale);
                                if (settings.debugFlipNormalGreen)
                                    materialMrDebug.normalSampleTs.y = -materialMrDebug.normalSampleTs.y;
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
                            emission *= SpectralRadianceToChannels(SampleEnvironment(*envPtr,
                                                                                     sampleDir,
                                                                                     settings.environmentRotation,
                                                                                     settings.environmentIntensity),
                                                                   spectral);
                        }
                        // Metal's camera_visible_emission (material_access.metal): seen directly by the camera, an
                        // emitter flagged CameraVisibleEmissionTint shows its base colour x 16 instead of its emission.
                        // Metal applies it only where the primary closure is not emissive, that is to every emitter
                        // except DiffuseLight, whose branch reads the raw emission (path_integrators.metal).
                        if (depth == 0u &&
                            type != static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::DiffuseLight) &&
                            (material.materialFlags & PathTracerShaderTypes::kMaterialFlagCameraVisibleEmissionTint) != 0u &&
                            (material.emission.x != 0.0f || material.emission.y != 0.0f || material.emission.z != 0.0f)) {
                            emission = MaterialBaseColor(material) * 16.0f;
                        }

                        const simd::float3 misOrigin = holeMisOriginValid ? holeMisOrigin : ray.origin;
                        if ((simd::dot(emission, emission) > 0.0f) && (hit.frontFace || hit.twoSided)) {
                            if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::DiffuseLight)) {
                                float misWeight = 1.0f;
                                bool useSpecularMis = (depth > 0) &&
                                                      ((!lastScatterWasDelta) ||
                                                       (!lastDeltaUnpaired &&
                                                        (settings.enableSpecularNee || settings.enableMnee)));
                                // After a medium scatter the hit is MIS-weighted only when volume NEE ran there.
                                if (useSpecularMis && rectLightCount > 0 && (!lastScatterWasVolume || lastVolumeRectNee)) {
                                    float lightPdf = RectLightPdfForHit(rectLightIndexByRect,
                                                                        rectangles,
                                                                        rectangleCount,
                                                                        rectNeeLightCount,
                                                                        hit,
                                                                        misOrigin);
                                    float denom = lastBsdfPdf + lightPdf;
                                    if (denom > 0.0f) {
                                        misWeight = lastBsdfPdf / denom;
                                    }
                                    misWeight = std::clamp(misWeight,
                                                           kMisWeightClampMin,
                                                           kMisWeightClampMax);
                                }
                                // Mesh emitters pair with the mesh-emitter NEE, not the rect lights (Metal's emitter
                                // branch, kernels.metal ~2516 and path_integrators.metal ~2923):
                                // mesh_emitter_bsdf_hit_weight after a vertex whose mesh NEE covered this direction.
                                if (hit.primitiveType == GeometryType::Mesh && depth > 0u && !lastScatterWasDelta &&
                                    previousMeshNeeEligible && meshNeeMis) {
                                    // Plain RIS: M draws of its proposal (risMeshDrawScale); baseline: one draw.
                                    const float lightPdf = MeshLightPdfForHit(meshLights, hit, misOrigin) * risMeshDrawScale;
                                    misWeight = (lightPdf > 0.0f && lastBsdfPdf > 0.0f)
                                        ? lastBsdfPdf / (lastBsdfPdf + lightPdf) : 1.0f;
                                }
                                // ReSTIR DI already counted this mesh light at the primary vertex.
                                if (previousRestirDiCovered && MeshLightFacesOrigin(meshLights, hit, ray.origin)) {
                                    misWeight = 0.0f;
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
                            // Metal's mesh_emitter_bsdf_hit_weight: the BSDF side of the balance heuristic, after a
                            // vertex whose mesh NEE covered this direction; 1 where NEE cannot reach the emitter.
                            if (depth > 0u && !lastScatterWasDelta && previousMeshNeeEligible && meshNeeMis) {
                                const float pdf = MeshLightPdfForHit(meshLights, hit, misOrigin) * risMeshDrawScale;
                                if (pdf > 0.0f && lastBsdfPdf > 0.0f) emission *= lastBsdfPdf / (lastBsdfPdf + pdf);
                            }
                            // ReSTIR DI already counted this mesh light at the primary vertex.
                            if (previousRestirDiCovered && MeshLightFacesOrigin(meshLights, hit, ray.origin)) {
                                emission = simd_make_float3(0.0f, 0.0f, 0.0f);
                            }
                            addContribution(emission);
                        }

                        // Subsurface, before direct light (Metal's subsurface_vertex_begin and subsurface_vertex_exit,
                        // closures_eval.metal; path_integrators.metal ~2042). At a front-face hit of a random-walk
                        // material the walk runs and the shading vertex moves to where it leaves the mesh: a white
                        // diffuse surface (under the coat's light-side crossing when coated), so direct light, MIS and
                        // the next ray are the ordinary ones there. A walk that does not leave ends the path with no
                        // light. Any other subsurface hit is shaded as its dense limit, except the separable method
                        // under sss=separable, which Embree does not port (reported at load) and shades as before.
                        // The ray cone still grows over this ray's own length, to the walk's entry point (Metal's
                        // hitDistanceWorld, taken before the walk).
                        bool subsurfaceExitVertex = false;
                        const float coneHitDistance = hit.t;
                        if (type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Subsurface) &&
                            !(settings.sssMode == PathTracer::RenderSettings::SssMode::Separable &&
                              material.sssParams.y < 0.5f)) {
                            bool runWalk = false;
                            if (!sssRandomWalk || material.sssParams.y < 0.5f || !hit.frontFace) {
                                material = SubsurfaceDenseLimitMaterial(material);
                            } else if (!SubsurfaceHasCoat(material)) {
                                runWalk = true;
                            } else {
                                // One-sample choice between the coat lobe and the walk, with the plastic closure's lobe
                                // probabilities. The coat lobe is shaded as plastic with a black base; the walk takes
                                // the view-side coat crossing here and the light-side one at its exit point. A normal
                                // map shapes only the coat (SssCoatNormalMap): the lobe probabilities and the coat lobe
                                // use the coat's normal; the walk's direction, crossings and exit do not.
                                PathTracerShaderTypes::MaterialData plastic = SubsurfaceDenseLimitMaterial(material);
                                // A coat normal whose mirror reflection of the view would point into the mesh is bent
                                // toward Ng (the walk runs at front faces only, so the outward Ng faces the view).
                                // Where the view is still below it, the coat falls back to shadingNormal.
                                bool coatMapped = false;
                                simd::float3 coatNormal =
                                    SssCoatNormalMap(sceneData, plastic, hit, shadingNormal, coatMapped);
                                if (coatMapped) {
                                    coatNormal = SssCoatValidReflectionNormal(hit.normal, wo, coatNormal);
                                }
                                if (coatMapped && !(simd::dot(coatNormal, wo) > 0.0f)) {
                                    coatNormal = shadingNormal;
                                    coatMapped = false;
                                }
                                const PlasticLobeProbabilities probs = PlasticLobeProbabilitiesFor(plastic, coatNormal, wo);
                                const float cosThetaO = simd::dot(shadingNormal, wo);
                                if (!(probs.coat + probs.diffuse > 0.0f)) {
                                    material = plastic;  // the view direction is below the shading normal
                                } else if (rng.nextFloat() < probs.coat) {
                                    // A mapped coat normal reflects part of the coat's light below the shading normal,
                                    // into the surface. As in Arnold, whose subsurface gains what a normal map takes
                                    // from its specular, that energy enters the walk: half the coat picks estimate it
                                    // with one coat sample and continue as a walk, the other half shade the coat.
                                    // Without a map the two normals are equal and the coat pick stays as it was.
                                    float coatShare = 1.0f;
                                    bool coatLostToWalk = false;
                                    bool coatPickEnds = false;
                                    if (simd::any(coatNormal != shadingNormal)) {
                                        coatShare = 0.5f;
                                        if (rng.nextFloat() < 0.5f) {
                                            const float coatRoughness = PlasticCoatRoughness(plastic);
                                            const simd::float3 wh = SampleGgxVndf(rng, coatRoughness, coatNormal, wo);
                                            const simd::float3 wi = simd::normalize(Reflect(-wo, wh));
                                            if (!(simd::dot(wh, coatNormal) > 0.0f) || !(simd::dot(wi, coatNormal) > 0.0f) ||
                                                !(simd::dot(wi, shadingNormal) <= 0.0f) || !(cosThetaO > 0.0f)) {
                                                // The coat sample stays above the surface: nothing is lost on this
                                                // pick. It ends at the coat vertex, with no walk to run or count.
                                                coatPickEnds = true;
                                            } else {
                                                const simd::float3 lost =
                                                    Splat3(PlasticCoatFresnelExact(plastic, simd::dot(wo, wh))) *
                                                    PlasticSpecularTint(plastic) *
                                                    GgxG1(GgxVndfAlpha(coatRoughness), simd::dot(coatNormal, wi));
                                                throughput *= lost * PlasticCoatCrossing(plastic, cosThetaO) /
                                                              (probs.coat * 0.5f);
                                                coatLostToWalk = true;
                                                runWalk = true;
                                            }
                                        }
                                    }
                                    if (!coatLostToWalk) {
                                        plastic.baseColorRoughness.x = 0.0f;
                                        plastic.baseColorRoughness.y = 0.0f;
                                        plastic.baseColorRoughness.z = 0.0f;
                                        plastic.materialFlags |= PathTracerShaderTypes::kMaterialFlagSubsurfaceCoatLobe;
                                        material = plastic;
                                        if (coatPickEnds) {
                                            throughput = Splat3(0.0f);
                                        } else {
                                            throughput /= probs.coat * coatShare;
                                            if (coatMapped) {
                                                // The coat lobe is an ordinary normal-mapped surface: its reflection,
                                                // NEE's cosine and the ray offset follow the coat's normal.
                                                shadingNormal = coatNormal;
                                                hit.shadingNormal = coatNormal;
                                            }
                                        }
                                    }
                                } else {
                                    // The walk is weighted once, here, by what the coat does not reflect toward the
                                    // view (exact Fresnel), and by its absorption along the view-side crossing. Both
                                    // keep the unmapped normal; only the lobe probability it divides by saw the coat's
                                    // normal.
                                    throughput *= (1.0f - PlasticCoatFresnelExact(plastic, cosThetaO)) *
                                                  PlasticCoatCrossing(plastic, cosThetaO) / probs.diffuse;
                                    runWalk = true;
                                }
                            }
                            if (runWalk) {
                                HitInfo exitHit;
                                simd::float3 walkWeight = Splat3(0.0f);
                                const SssWalkOutcome outcome =
                                    SssRandomWalk(sceneData, material, hit, settings.sssMaxSteps, rng, exitHit, walkWeight);
                                sssWalkStats.note(matIndex, outcome);
                                if (outcome != SssWalkOutcome::Exit) {
                                    break;
                                }
                                throughput *= walkWeight;
                                hit = exitHit;
                                hit.authoredNormalWasFaced = false;
                                subsurfaceExitVertex = true;
                                if (SubsurfaceHasCoat(material)) {
                                    material = SubsurfaceDenseLimitMaterial(material);
                                    material.materialFlags |= PathTracerShaderTypes::kMaterialFlagSubsurfaceExit;
                                } else {
                                    material.typeEta.x = static_cast<float>(PathTracerShaderTypes::MaterialType::Lambertian);
                                }
                                material.baseColorRoughness.x = 1.0f;
                                material.baseColorRoughness.y = 1.0f;
                                material.baseColorRoughness.z = 1.0f;
                                shadingNormal = hit.shadingNormal;
                                wo = shadingNormal;
                                incidentDir = -wo;
                                // Shadow rays and the next ray leave from the exit point.
                                previousHit = hit;
                            }
                            type = static_cast<uint32_t>(material.typeEta.x);
                        }

                        bool surfaceIsDelta = MaterialIsDelta(material);
                        // Under source transport Metal's wavefront samples no light at a path's last vertex
                        // (sourceDirectAllowed, kernels.metal): the path ends with its last emission hit, as Tungsten's
                        // bounce count does, not with one more bounce of direct light.
                        const bool sourceDirectAllowed = !gSourceTransport.enabled || depth + 1u < settings.maxDepth;
                        // Metal's source shadow medium (wavefront_prepare_source_shadow, and the source shadow
                        // traversal in path_integrators.metal): under source transport a shadow ray starts in the
                        // path's current medium, after interface_select_medium for its direction, and is attenuated
                        // over its length from the hit point. A dielectric interface on the way blocks it, so one
                        // medium covers the whole segment; an unbounded segment through an absorbing medium carries
                        // nothing. tMax is the shadow ray's own. Metal also re-selects the medium at each
                        // pass-through (alpha hole, source opacity, thin sheet): an explicit-interior material there
                        // starts a new piece in its own medium or outside it (IntersectSceneSkippingAlpha records the
                        // events in `passes`), and each piece is attenuated by its own medium.
                        const auto sourceShadowTransmittance = [&](const Ray& shadowRay, float tMax,
                                                                   const SourceOpacityShadow& passes) -> simd::float3 {
                            const simd::float3 zero = simd_make_float3(0.0f, 0.0f, 0.0f);
                            if (!gSourceTransport.enabled) {
                                return simd_make_float3(1.0f, 1.0f, 1.0f);
                            }
                            simd::float3 sigma = mediumDepth > 0u ? mediumSigmaStack[mediumDepth - 1u] : zero;
                            if (pt_shared::source_material_flag(material, PathTracerShaderTypes::kMaterialFlagExplicitInterior)) {
                                sigma = simd::dot(shadowRay.direction, hit.normal) < 0.0f
                                    ? simd::max(DielectricSigmaA(material), zero) : zero;
                            }
                            const float offset = simd::dot(shadowRay.origin - hit.position, shadowRay.direction);
                            if (passes.mediumEventCount == 0u) {
                                if (!(sigma.x > 0.0f || sigma.y > 0.0f || sigma.z > 0.0f)) {
                                    return simd_make_float3(1.0f, 1.0f, 1.0f);
                                }
                                const float segment = std::max(tMax + offset, 0.0f);
                                return Exp3(-sigma * std::min(segment, 1.0e20f));
                            }
                            simd::float3 rgb = simd_make_float3(1.0f, 1.0f, 1.0f);
                            float previous = -offset;
                            for (uint32_t i = 0u; i <= passes.mediumEventCount; ++i) {
                                const float end = i < passes.mediumEventCount ? passes.mediumEventT[i] : tMax;
                                if (sigma.x > 0.0f || sigma.y > 0.0f || sigma.z > 0.0f) {
                                    rgb *= Exp3(-sigma * std::min(std::max(end - previous, 0.0f), 1.0e20f));
                                }
                                previous = end;
                                if (i < passes.mediumEventCount) {
                                    sigma = passes.mediumEventSigma[i];
                                }
                            }
                            return rgb;
                        };

                        // ReSTIR DI at the primary surface (the first one the path scatters from, so one reached
                        // through alpha cutouts too), as Metal renders it by default: the fused shader path, which
                        // headless and GUI runs use unless restirDebug or PT_ENABLE_INTERACTIVE_EXPLICIT_RESTIR_DI
                        // selects the explicit screen-space passes (RenderLoop.mm). One reservoir over the sun and the
                        // mesh lights (Metal's emissive primitives; rect lights and the environment keep their own
                        // NEE, restir_di_make_provider_set) at this sample's own hit; its winner is shaded with
                        // W = wSum / (M * p_hat) and one shadow ray (path_integrators.metal, RIS winner shading).
                        // Plain RIS (the RIS modes) runs the same reservoir and winner shading at every non-delta
                        // vertex, with the plain-RIS MIS weight against BSDF hits on mesh emitters.
                        if (sourceDirectAllowed && ((restirDirectMode && !primaryVertexScattered) || risDirectMode) &&
                            !surfaceIsDelta &&
                            type != static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::DiffuseLight)) {
                            CpuReservoir reservoir;
                            if (CpuBuildRestirDiProviderReservoirForHit(sceneData, directionalLights, meshLights,
                                                                        hit, material, shadingNormal, wo,
                                                                        clampParams, std::max(settings.risCandidateCount, 1u),
                                                                        rng, reservoir)) {
                                const CpuRisSamplePayload& winner = reservoir.winner;
                                const bool winnerDirectional = CpuRisPayloadIsDirectional(winner);
                                const simd::float3 omegaL = CpuRisPayloadOmegaL(hit, winner);
                                const float cosThetaSurface = ReferenceDirectCos(material, hit, shadingNormal, omegaL);
                                const float phat =
                                    CpuReservoirWinnerPHatForHit(hit, material, shadingNormal, wo, clampParams, winner);
                                const float reservoirW =
                                    phat > 0.0f ? (reservoir.wSum / static_cast<float>(reservoir.M)) / phat : 0.0f;
                                bool winnerCutoutBackLightUnsampled = false;
                                const BsdfEval bsdfEval = DirectLightTargetBsdf(material, hit, shadingNormal, wo, omegaL,
                                                                                clampParams, winnerDirectional,
                                                                                &winnerCutoutBackLightUnsampled);
                                // Metal's mesh_emitter_nee_mis_weight for plain RIS: M draws of the proposal against the
                                // BSDF pdf, 1 for suns, at the last vertex and where the BSDF cannot sample the
                                // direction. ReSTIR DI carries the whole contribution (weight 1).
                                const float winnerMisBsdfPdf = NeeMisBsdfPdf(bsdfEval, winnerCutoutBackLightUnsampled);
                                const float winnerDrawPdf = static_cast<float>(risCandidates) * winner.pdf;
                                const float risMisWeight =
                                    (!risDirectMode || winnerDirectional || !(depth + 1u < settings.maxDepth) ||
                                     !(winnerMisBsdfPdf > 0.0f) || !(winnerDrawPdf > 0.0f))
                                        ? 1.0f : winnerDrawPdf / (winnerDrawPdf + winnerMisBsdfPdf);
                                if (cosThetaSurface > 0.0f && reservoirW > 0.0f && !bsdfEval.isDelta) {
                                    ++tileShadowRayCount;
                                    Ray shadowRay;
                                    shadowRay.origin = SpawnOrigin(hit, omegaL, &material);
                                    shadowRay.direction = omegaL;
                                    const float shadowMax = winnerDirectional
                                        ? std::numeric_limits<float>::infinity()
                                        : std::max(simd::length(winner.position - hit.position) - ShadowLightEndMargin(),
                                                   ShadowLightEndMargin());
                                    HitInfo shadowHit;
                                    SourceOpacityShadow opacityShadow{.transmission = 1.0f, .bounce = depth + 1u, .maxDepth = settings.maxDepth};
                                    bool occluded = IntersectSceneSkippingAlpha(sceneData, shadowRay, shadowMax, materials,
                                                                                materialCount, settings.debugNormalStrengthScale,
                                                                                ShadedTriangleExclusion(hit, hardwareExcludeAttempts),
                                                                                shadowHit,
                                                                                PavilionPunctualShadowEnabled(material,
                                                                                                              winnerDirectional),
                                SourceShadowActive() ? &opacityShadow : nullptr, &material);
                                    if (occluded && CpuShadowHitIsSelectedLight(shadowHit, winner)) {
                                        occluded = false;
                                    }
                                    if (!occluded) {
                                        // Medium transmittance over the shadow segment, as on Metal's fused winner
                                        // (unreachable under volumes, which demote ReSTIR DI; kept for parity).
                                        const simd::float3 winnerVolumeTr = VolumeSurfaceNeeTransmittance(
                                            volumeMedium, simd::length(winner.position - hit.position), winnerDirectional);
                                        addContribution(bsdfEval.value * winner.emission *
                                                            (cosThetaSurface * reservoirW * risMisWeight) * winnerVolumeTr,
                                                        sourceShadowTransmittance(shadowRay, shadowMax, opacityShadow) * opacityShadow.weight());
                                    }
                                }
                            }
                        }

                        // Every vertex after the first scatter samples the sun, as Metal's RIS NEE does there for
                        // any non-delta material (path_integrators.metal, non-explicit stage). Plain RIS already
                        // draws the suns among its candidates at every vertex.
                        if (sourceDirectAllowed && directionalNeeMode && !risDirectMode &&
                            (!restirDirectMode || primaryVertexScattered) &&
                            !surfaceIsDelta &&
                            !directionalLights.empty()) {
                            for (const DirectionalLightInfo& light : directionalLights) {
                                const simd::float3 lightDirection = SampleDirectionalCone(light.direction, light.cosDelta, rng);
                                // Plain NEE's cosine in every mode: Metal's reservoir modes draw the sun with
                                // direct_light_target_cos, which lights a two-sided cutout from behind (PR #80).
                                float nDotL = ReferenceDirectCos(material, hit, shadingNormal, lightDirection);
                                if (!(nDotL > 0.0f)) {
                                    continue;
                                }

                                ++tileShadowRayCount;
                                Ray shadowRay;
                                shadowRay.origin = SpawnOrigin(hit, lightDirection, &material);
                                shadowRay.direction = lightDirection;
                                HitInfo shadowHit;
                                SourceOpacityShadow opacityShadow{.transmission = 1.0f, .bounce = depth + 1u, .maxDepth = settings.maxDepth};
                                const bool occluded =
                                    IntersectSceneSkippingAlpha(sceneData,
                                                                shadowRay,
                                                                1.0e20f,
                                                                materials,
                                                                materialCount,
                                                                settings.debugNormalStrengthScale,
                                                                ShadedTriangleExclusion(hit, hardwareExcludeAttempts),
                                                                shadowHit,
                                                                PavilionPunctualShadowEnabled(material, true),
                                SourceShadowActive() ? &opacityShadow : nullptr, &material);
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
                                                                 lightDirection,
                                                                 clampParams,
                                                                 true);
                                ApplyTwoSidedAlphaCutoutDiffuseFallback(material, hit, shadingNormal, lightDirection, bsdfEval);
                                if (debugSample && depth == 0) {
                                    std::cerr << "[EmbreeDirectDebug] directional_bsdf"
                                              << " bsdfPdf=" << bsdfEval.pdf
                                              << " bsdfLum=" << Luminance(bsdfEval.value)
                                              << " isDelta=" << (bsdfEval.isDelta ? 1 : 0)
                                              << std::endl;
                                }
                                if (!bsdfEval.isDelta && bsdfEval.pdf > 0.0f) {
                                    simd::float3 contribution =
                                        SpectralRadianceToChannels(light.emission, spectral) * bsdfEval.value * nDotL;
                                    if (debugSample && depth == 0) {
                                        std::cerr << "[EmbreeDirectDebug] directional_contrib"
                                                  << " contribLum=" << Luminance(contribution)
                                                  << std::endl;
                                    }
                                    addContribution(contribution, sourceShadowTransmittance(shadowRay, 1.0e20f, opacityShadow) * opacityShadow.weight());
                                }
                            }
                        }

                        // Baseline mesh NEE (Metal's sample_direct_light_baseline): one draw by power over every
                        // emitter. CpuSampleMeshLight rejects a point seen from a side its emitter does not emit to
                        // (emissive_light_cosine), so one-sided emitters stay one-sided.
                        if (sourceDirectAllowed && meshNeeBaseline && !surfaceIsDelta) {
                            CpuRisSamplePayload light;
                            if (CpuSampleMeshLight(sceneData, meshLights, hit, 1.0f, rng, light)) {
                                const simd::float3 toLight = light.position - hit.position;
                                const float distance = simd::length(toLight);
                                const simd::float3 direction = toLight / distance;
                                const float cosine = ReferenceDirectCos(material, hit, shadingNormal, direction);
                                if (cosine > 0.0f && light.pdf > 0.0f) {
                                    Ray shadow;
                                    shadow.origin = SpawnOrigin(hit, direction, &material);
                                    // Metal aims the shadow ray of a two-sided or textured emitter at the sampled point
                                    // from the offset origin, and every one under source transport
                                    // (wavefront_prepare_source_shadow); a plain emitter's leaves along the sampled
                                    // direction and stops the end margin short of the sampled distance.
                                    const bool aimFromOrigin = gSourceTransport.enabled ||
                                        (light.flags & (PathTracerShaderTypes::kLightPrimitiveFlagTwoSided |
                                                        PathTracerShaderTypes::kLightPrimitiveFlagTexturedEmission)) != 0u;
                                    float shadowMax = 0.0f;
                                    if (aimFromOrigin) {
                                        const simd::float3 visibilityDelta = light.position - shadow.origin;
                                        const float visibilityDistance = simd::length(visibilityDelta);
                                        shadow.direction = visibilityDelta / visibilityDistance;
                                        shadowMax = std::max(visibilityDistance - ShadowLightEndMargin(), ShadowLightEndMargin());
                                    } else {
                                        shadow.direction = direction;
                                        shadowMax = std::max(distance - ShadowLightEndMargin(), ShadowLightEndMargin());
                                    }
                                    HitInfo blocker;
                                    ++tileShadowRayCount;
                                    SourceOpacityShadow opacityShadow{.transmission = 1.0f, .bounce = depth + 1u, .maxDepth = settings.maxDepth};
                                    bool occluded = IntersectSceneSkippingAlpha(sceneData, shadow, shadowMax,
                                                                                materials, materialCount, settings.debugNormalStrengthScale,
                                                                                ShadedTriangleExclusion(hit, hardwareExcludeAttempts),
                                                                                blocker, false,
                                SourceShadowActive() ? &opacityShadow : nullptr, &material);
                                    // A hit on the sampled light itself does not shadow it, as on Metal's wavefront
                                    // (kWavefrontShadowFlagEmissivePrimitive with primitiveIndex == lightId). Under
                                    // source transport the end margin is 1e-6, within float error of the light's own t.
                                    if (occluded && CpuShadowHitIsSelectedLight(blocker, light)) {
                                        occluded = false;
                                    }
                                    if (!occluded) {
                                        BsdfEval eval = EvaluateBsdf(material, hit.position, shadingNormal, wo, direction, clampParams);
                                        const bool cutoutBackLightUnsampled =
                                            ApplyTwoSidedAlphaCutoutDiffuseFallback(material, hit, shadingNormal, direction, eval);
                                        if (!eval.isDelta) {
                                            // Metal's mesh_emitter_nee_mis_weight: the light side of the balance
                                            // heuristic; 1 at the last vertex or where the BSDF cannot sample the
                                            // direction.
                                            const float misBsdfPdf = NeeMisBsdfPdf(eval, cutoutBackLightUnsampled);
                                            const float weight = (depth + 1u < settings.maxDepth && misBsdfPdf > 0.0f)
                                                ? light.pdf / (light.pdf + misBsdfPdf) : 1.0f;
                                            const simd::float3 value = ReferenceContinuationNeeValue(material, shadingNormal, wo, direction, eval.value);
                                            // Medium transmittance over the distance to the sampled point
                                            // (VolumeSurfaceNeeTransmittance), before the firefly clamp as on Metal.
                                            addContribution(SpectralRadianceToChannels(light.emission, spectral) * value * cosine *
                                                                (weight / std::max(light.pdf, 1.0e-6f)) *
                                                                VolumeSurfaceNeeTransmittance(volumeMedium, distance, false),
                                                            sourceShadowTransmittance(shadow, shadowMax, opacityShadow) * opacityShadow.weight());
                                        }
                                    }
                                }
                            }
                        }

                        // Rect lights keep their own NEE at every non-delta vertex in every mode, as on Metal,
                        // where the ReSTIR DI reservoir holds emissive primitives only.
                        if (sourceDirectAllowed && !surfaceIsDelta && !rectLights.empty()) {
                            const uint32_t neeDrawCount = exactPointNeeActive ? rectLightCount + 1u : 1u;
                            for (uint32_t neeDraw = 0u; neeDraw < neeDrawCount; ++neeDraw) {
                                RectLightSample lightSample;
                                bool sampled = SampleRectLightNeeDraw(rectLights, envPtr, settings, hit, rng, lightSample,
                                                               neeDraw, exactPointNeeActive);
                                if (debugSample && depth == 0 && !sampled) {
                                    std::cerr << "[EmbreeDirectDebug] rect_sample_failed"
                                              << " pixel=(" << x << "," << y << ")"
                                              << std::endl;
                                }
                                if (sampled) {
                                    float nDotL = ReferenceDirectCos(material, hit, shadingNormal, lightSample.direction);
                                    if (lightSample.pdf > 0.0f && nDotL > 0.0f) {
                                        float shadowMax =
                                            std::max(lightSample.distance - ShadowLightEndMargin(), ShadowLightEndMargin());
                                        ++tileShadowRayCount;
                                        HitInfo shadowHit;
                                        Ray shadowRay;
                                        shadowRay.origin = SpawnOrigin(hit, lightSample.direction, &material);
                                        shadowRay.direction = lightSample.direction;
                                        bool occluded = false;
                                        SourceOpacityShadow opacityShadow{.transmission = 1.0f, .bounce = depth + 1u, .maxDepth = settings.maxDepth};
                                        if (IntersectSceneSkippingAlpha(sceneData,
                                                                        shadowRay,
                                                                        shadowMax,
                                                                        materials,
                                                                        materialCount,
                                                                        settings.debugNormalStrengthScale,
                                                                        ShadedTriangleExclusion(hit, hardwareExcludeAttempts),
                                                                        shadowHit,
                                                                        PavilionPunctualShadowEnabled(material, lightSample.isReferencePoint),
                                SourceShadowActive() ? &opacityShadow : nullptr, &material)) {
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
                                                                         clampParams,
                                                                         lightSample.isReferencePoint);
                                        const bool cutoutBackLightUnsampled =
                                            ApplyTwoSidedAlphaCutoutDiffuseFallback(material, hit, shadingNormal,
                                                                                    lightSample.direction, bsdfEval);
                                            if (debugSample && depth == 0) {
                                                std::cerr << "[EmbreeDirectDebug] rect_bsdf"
                                                          << " bsdfPdf=" << bsdfEval.pdf
                                                          << " bsdfLum=" << Luminance(bsdfEval.value)
                                                          << " isDelta=" << (bsdfEval.isDelta ? 1 : 0)
                                                          << std::endl;
                                            }
                                            if (!bsdfEval.isDelta && bsdfEval.pdf > 0.0f) {
                                                const float misBsdfPdf = NeeMisBsdfPdf(bsdfEval, cutoutBackLightUnsampled);
                                                float weight = lightSample.isReferencePoint ? 1.0f : lightSample.pdf /
                                                               (lightSample.pdf + misBsdfPdf);
                                                const simd::float3 value = lightSample.isReferencePoint ? bsdfEval.value :
                                                    ReferenceContinuationNeeValue(material, shadingNormal, wo, lightSample.direction, bsdfEval.value);
                                                if ((material.materialFlags & (PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular |
                                                      PathTracerShaderTypes::kMaterialFlagReferenceDielectricContinuation)) != 0u &&
                                                    depth + 1u >= settings.maxDepth) weight = 1.0f;
                                                simd::float3 contribution =
                                                    SpectralRadianceToChannels(lightSample.emission, spectral) * value * nDotL;
                                                contribution *= weight / lightSample.pdf;
                                                contribution *= VolumeSurfaceNeeTransmittance(volumeMedium, lightSample.distance, false);
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
                                                addContribution(contribution, sourceShadowTransmittance(shadowRay, shadowMax, opacityShadow) * opacityShadow.weight());
                                            }
                                        }
                                    }
                                }
                            } // analytic NEE draw loop
                        }

                        if (sourceDirectAllowed && !surfaceIsDelta && envSampling) {
                            PathTracer::EnvImportanceSample envSample =
                                PathTracer::SampleEnvironmentCpu(envPtr->distribution,
                                                                 rng.nextFloat(),
                                                                 rng.nextFloat(),
                                                                 rng.nextFloat(),
                                                                 settings.environmentRotation,
                                                                 settings.environmentIntensity,
                                                                 envPtr->rgba.data());
                            float nDotL = ReferenceDirectCos(material, hit, shadingNormal, envSample.direction);
                            if (envSample.pdf > 0.0f && nDotL > 0.0f) {
                                Ray envShadowRay;
                                envShadowRay.origin = SpawnOrigin(hit, envSample.direction, &material);
                                envShadowRay.direction = envSample.direction;
                                HitInfo envShadowHit;
                                SourceOpacityShadow opacityShadow{.transmission = 1.0f, .bounce = depth + 1u, .maxDepth = settings.maxDepth};
                                if (!IntersectSceneSkippingAlpha(sceneData,
                                                                 envShadowRay,
                                                                 std::numeric_limits<float>::infinity(),
                                                                 materials,
                                                                 materialCount,
                                                                 settings.debugNormalStrengthScale,
                                                                 ShadedTriangleExclusion(hit, hardwareExcludeAttempts),
                                                                 envShadowHit, false,
                                SourceShadowActive() ? &opacityShadow : nullptr, &material)) {
                                    simd::float3 envRadiance = SpectralRadianceToChannels(
                                        SampleEnvironment(*envPtr,
                                                          envSample.direction,
                                                          settings.environmentRotation,
                                                          settings.environmentIntensity),
                                        spectral);
                                BsdfEval bsdfEval = EvaluateBsdf(material,
                                                                 hit.position,
                                                                 shadingNormal,
                                                                 wo,
                                                                 envSample.direction,
                                                                 clampParams);
                                const bool cutoutBackLightUnsampled =
                                    ApplyTwoSidedAlphaCutoutDiffuseFallback(material, hit, shadingNormal, envSample.direction,
                                                                            bsdfEval);
                                    if (!bsdfEval.isDelta && bsdfEval.pdf > 0.0f) {
                                        float weight = envSample.pdf / (envSample.pdf + NeeMisBsdfPdf(bsdfEval, cutoutBackLightUnsampled));
                                        if ((material.materialFlags & (PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular |
                                              PathTracerShaderTypes::kMaterialFlagReferenceDielectricContinuation)) != 0u &&
                                            depth + 1u >= settings.maxDepth) weight = 1.0f;
                                        simd::float3 contribution = envRadiance *
                                            ReferenceContinuationNeeValue(material, shadingNormal, wo, envSample.direction, bsdfEval.value) * nDotL;
                                        contribution *= weight / envSample.pdf;
                                        contribution *= sourceShadowTransmittance(envShadowRay, std::numeric_limits<float>::infinity(), opacityShadow) * opacityShadow.weight();
                                        if (materialMrDebugCaptured &&
                                            debugSample &&
                                            depth == 0 &&
                                            type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness)) {
                                            materialMrDebug.directContribution = contribution;
                                            materialMrDebug.directContributionKind = 1.0f;
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
                    bsdfSample.isBssrdf = subsurfaceExitVertex;
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
                        !IsFinite(bsdfSample.weight) ||
                        !FalcorFaceVisible(material, hit, bsdfSample.direction)) {
                        break;
                    }
                    if (!bsdfSample.isDelta) {
                        pathMaxRoughness = std::max(pathMaxRoughness, PathRegularizationRoughness(material));
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
                    if (cacheTrainingPath) {
                        // A training path never loads from the cache: its suffix is the unbiased sample the cache
                        // learns from. Its records train when it ends.
                        const uint32_t slot =
                            PathTracer::RadianceCacheTraining::TrainSlot(depth, settings.radianceCacheMinDepth);
                        if (slot < PathTracer::RadianceCacheTraining::kTrainRecordCount &&
                            CpuRadianceCacheCanQuery(settings, material, bsdfSample, depth)) {
                            CpuRadianceCacheTrainRecord& record = cacheTrainRecords[slot];
                            record.materialType = static_cast<uint32_t>(material.typeEta.x);
                            record.domainKey =
                                CpuRadianceCacheDomainKey(settings, hit.position, depth, record.materialType, 0u);
                            record.cellIndex =
                                radianceCacheCapacity > 0u ? (record.domainKey % radianceCacheCapacity) : 0u;
                            record.position = hit.position;
                            record.normal = simd::normalize(shadingNormal);
                            record.roughness = MaterialRoughness(material);
                            record.throughputAtQuery = throughput;
                            record.radianceBaseline = radiance;
                            cacheTrainValidMask |= 1u << slot;
                        }
                    } else if (radianceCacheActive) {
                        if (!CpuRadianceCacheCanQuery(settings, material, bsdfSample, depth)) {
                            radianceCacheFallbackCount.fetch_add(1u, std::memory_order_relaxed);
                            cacheStat(PathTracer::RadianceCacheTraining::kStatFallbackReasonBase +
                                          CpuRadianceCacheCannotQueryReason(settings, material, depth),
                                      1u);
                        } else {
                            radianceCacheQueryCount.fetch_add(1u, std::memory_order_relaxed);
                            const uint32_t cacheDepthBin = std::min(depth, kCpuPathReservoirDepthCount - 1u);
                            cacheStat(PathTracer::RadianceCacheTraining::kStatDepthQueryBase + cacheDepthBin, 1u);
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
                            // Miss reason for the diagnostics, in radiance_cache_load's order and codes.
                            uint32_t cacheMissReason = 4u;  // empty cell
                            if (cell.valid && cell.sampleCount > 0u) {
                                if (cell.invalidated || cell.frameTag >= s) {
                                    cacheMissReason = 6u;
                                } else if (cell.domainKey != domainKey || cell.materialType != materialType ||
                                           cell.lobeType != 0u) {
                                    cacheMissReason = 7u;
                                }
                            }
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
                                    cacheStat(PathTracer::RadianceCacheTraining::kStatDepthHitBase + cacheDepthBin, 1u);
                                    cacheStat(PathTracer::RadianceCacheTraining::kStatDepthHitLumaFixedBase +
                                                  cacheDepthBin,
                                              PathTracer::RadianceCacheTraining::StatLumaToFixed(luma));
                                } else if (normalDot < 0.65f || roughnessDelta > 0.35f) {
                                    cacheMissReason = 7u;
                                } else if (!IsFinite(estimate)) {
                                    cacheMissReason = 6u;
                                } else if (biasRisk > 0.85f) {
                                    cacheMissReason = 8u;
                                } else {
                                    cacheMissReason = 5u;
                                }
                            }
                            if (!cacheHit) {
                                radianceCacheMissCount.fetch_add(1u, std::memory_order_relaxed);
                                radianceCacheFallbackCount.fetch_add(1u, std::memory_order_relaxed);
                                cacheStat(PathTracer::RadianceCacheTraining::kStatFallbackReasonBase + cacheMissReason,
                                          1u);
                            }
                        }
                    }
                    if (radianceCacheTerminated) {
                        break;
                    }

                    uint32_t nextSpecularDepth = bsdfSample.isDelta ? (specularDepth + 1u) : 0u;
                    specularDepth = nextSpecularDepth;

                    bool useMnee = settings.enableMnee && !settings.sourceTransport;  // Metal's wavefront has no bounce MNEE
                    bool specNeeEnabled = settings.enableSpecularNee && !settings.sourceTransport;  // off under source transport, as Metal
                    float dirLenSq = simd::dot(bsdfSample.direction, bsdfSample.direction);
                    bool specDirectionValid = (dirLenSq > 0.0f) && IsFinite(bsdfSample.direction);
                    bool mneeEligible = useMnee &&
                                        bsdfSample.isDelta &&
                                        specDirectionValid &&
                                        type == static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::Dielectric) &&
                                        nextSpecularDepth == 1u;
                    // As on Metal, no specular NEE along a refraction into a medium.
                    bool specNeeEligible = specNeeEnabled &&
                                           bsdfSample.isDelta &&
                                           bsdfSample.mediumEvent <= 0 &&
                                           specDirectionValid &&
                                           !mneeEligible;

                    // The four delta-direction visibility rays below pass no shadow exclusion. After a transmission
                    // Metal excludes the shaded mesh without its triangle, which skips at most a hit within kEpsilon
                    // (1e-3) of the origin, and its 1e-3 minimum distance leaves no such hit.
                    if (specNeeEligible && envSampling) {
                        simd::float3 neeDir = simd::normalize(bsdfSample.direction);
                        Ray neeVisibilityRay;
                        neeVisibilityRay.origin = SpawnOrigin(hit, neeDir, &material);
                        neeVisibilityRay.direction = neeDir;
                        HitInfo neeVisibilityHit;
                        if (!IntersectSceneSkippingAlpha(sceneData,
                                                         neeVisibilityRay,
                                                         std::numeric_limits<float>::infinity(),
                                                         materials,
                                                         materialCount,
                                                         settings.debugNormalStrengthScale,
                                                         ShadowExclusion{},
                                                         neeVisibilityHit, false, nullptr, &material)) {
                            float envPdf = std::max(EnvironmentPdf(*envPtr,
                                                                   settings.environmentRotation,
                                                                   neeDir),
                                                    kSpecularNeePdfFloor);
                            float bsdfPdf = std::max(bsdfSample.pdf, kSpecularNeePdfFloor);
                            float denom = envPdf + bsdfPdf;
                            float misWeight = denom > 0.0f ? (envPdf / denom) : 0.0f;
                            misWeight = std::clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                            simd::float3 envColor = SampleEnvironment(*envPtr,
                                                                      neeDir,
                                                                      settings.environmentRotation,
                                                                      settings.environmentIntensity);
                            // The delta sample fixed this direction; it was not drawn from the
                            // environment, so there is no 1/envPdf. This connection takes the
                            // light strategy's MIS share and the continuation's environment miss
                            // takes bsdfPdf / (bsdfPdf + envPdf), so the two sum to one.
                            simd::float3 neeContribution = bsdfSample.weight * envColor * misWeight;
                            addContribution(neeContribution);
                        }
                    }

                    if (specNeeEligible && rectLightCount > 0) {
                        simd::float3 neeDir = simd::normalize(bsdfSample.direction);
                        Ray neeRay;
                        neeRay.origin = SpawnOrigin(hit, neeDir, &material);
                        neeRay.direction = neeDir;
                        HitInfo lightHit;
                        if (IntersectSceneSkippingAlpha(sceneData,
                                                        neeRay,
                                                        std::numeric_limits<float>::infinity(),
                                                        materials,
                                                        materialCount,
                                                        settings.debugNormalStrengthScale,
                                                        ShadowExclusion{},
                                                        lightHit, false, nullptr, &material)) {
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
                                float bsdfPdf = std::max(bsdfSample.pdf, kSpecularNeePdfFloor);
                                float denom = lightPdf + bsdfPdf;
                                float misWeight = denom > 0.0f ? (lightPdf / denom) : 0.0f;
                                misWeight = std::clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                                // Delta direction, not a light sample: MIS share only, no 1/lightPdf
                                // (the continuation's emitter hit takes the complement).
                                simd::float3 neeContribution = bsdfSample.weight * rectHit.emission * misWeight;
                                // Delta-direction connection to a rect light: transmittance over the connection.
                                neeContribution *= VolumeSurfaceNeeTransmittance(volumeMedium, lightHit.t, false);
                                addContribution(neeContribution);
                            }
                        }
                    }

                    if (mneeEligible && envSampling) {
                        simd::float3 mneeDir = simd::normalize(bsdfSample.direction);
                        ++tileShadowRayCount;
	                        Ray mneeVisibilityRay;
	                        mneeVisibilityRay.origin = SpawnOrigin(hit, mneeDir, &material);
	                        mneeVisibilityRay.direction = mneeDir;
	                        HitInfo mneeVisibilityHit;
	                        if (!IntersectSceneSkippingAlpha(sceneData,
	                                                         mneeVisibilityRay,
	                                                         std::numeric_limits<float>::infinity(),
	                                                         materials,
	                                                         materialCount,
	                                                         settings.debugNormalStrengthScale,
	                                                         ShadowExclusion{},
	                                                         mneeVisibilityHit, false, nullptr, &material)) {
                            float envPdf = std::max(EnvironmentPdf(*envPtr,
                                                                   settings.environmentRotation,
                                                                   mneeDir),
                                                    kSpecularNeePdfFloor);
                            float bsdfPdf = std::max(bsdfSample.pdf, kSpecularNeePdfFloor);
                            float denom = envPdf + bsdfPdf;
                            float misWeight = denom > 0.0f ? (envPdf / denom) : 0.0f;
                            misWeight = std::clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                            simd::float3 envColor = SampleEnvironment(*envPtr,
                                                                      mneeDir,
                                                                      settings.environmentRotation,
                                                                      settings.environmentIntensity);
                            // Delta direction, not an environment sample: MIS share only, no
                            // 1/envPdf (the continuation's environment miss takes the complement).
                            simd::float3 mneeContribution = bsdfSample.weight * envColor * misWeight;
                            addContribution(mneeContribution);
                        }
                    }

                    if (mneeEligible && rectLightCount > 0) {
                        simd::float3 mneeDir = simd::normalize(bsdfSample.direction);
                        Ray mneeRay;
                        mneeRay.origin = SpawnOrigin(hit, mneeDir, &material);
                        mneeRay.direction = mneeDir;
                        HitInfo lightHit;
                        ++tileShadowRayCount;
                        if (IntersectSceneSkippingAlpha(sceneData,
                                                        mneeRay,
                                                        std::numeric_limits<float>::infinity(),
                                                        materials,
                                                        materialCount,
                                                        settings.debugNormalStrengthScale,
                                                        ShadowExclusion{},
                                                        lightHit, false, nullptr, &material)) {
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
                                float bsdfPdf = std::max(bsdfSample.pdf, kSpecularNeePdfFloor);
                                float denom = lightPdf + bsdfPdf;
                                float misWeight = denom > 0.0f ? (lightPdf / denom) : 0.0f;
                                misWeight = std::clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                                // Delta direction, not a light sample: MIS share only, no 1/lightPdf
                                // (the continuation's emitter hit takes the complement).
                                simd::float3 mneeContribution = bsdfSample.weight * rectHit.emission * misWeight;
                                mneeContribution *= VolumeSurfaceNeeTransmittance(volumeMedium, lightHit.t, false);
                                addContribution(mneeContribution);
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

                        rayCone.width = RayConeWidthAtDistance(rayCone, coneHitDistance);
                        rayCone.spread = std::min(rayCone.spread +
                                                      BsdfConeSpreadIncrement(bsdfSample.lobeType,
                                                                              bsdfSample.lobeRoughness,
                                                                              bsdfSample.isDelta),
                                                  1.5f);

                        lastBsdfPdf = bsdfSample.pdf > 0.0f ? bsdfSample.pdf : lastBsdfPdf;
                        // Metal's mesh_emitter_nee_covers_direction (baseline): this vertex's mesh NEE could have
                        // produced the scattered direction only where its direct-light cosine is positive.
                        // Plain RIS covers the same directions at every vertex where its reservoir ran (not on a
                        // DiffuseLight, whose hit ends the path).
                        previousMeshNeeEligible = meshNeeMis && !surfaceIsDelta && !bsdfSample.isDelta &&
                            ReferenceDirectCos(material, hit, shadingNormal, bsdfSample.direction) > 0.0f;
                        // ReSTIR DI shades every non-light primary surface with a non-delta BSDF where its target's
                        // cosine is positive (Metal's mesh_emitter_nee_covers_direction, direct_light_target_cos). A
                        // subsurface walk's exit point is an ordinary surface vertex here, as Metal's
                        // (!isBssrdf || subsurfaceExitVertex).
                        previousRestirDiCovered = restirDirectMode && !primaryVertexScattered && !meshLights.empty() &&
                            type != static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::DiffuseLight) &&
                            !surfaceIsDelta && !bsdfSample.isDelta &&
                            ReferenceDirectCos(material, hit, shadingNormal, bsdfSample.direction) > 0.0f;
                        primaryVertexScattered = true;
                        lastScatterWasDelta = bsdfSample.isDelta;
                        lastDeltaUnpaired = bsdfSample.isDelta && !(specNeeEligible || mneeEligible);

                        // Metal pushes the dielectric's absorption when the path refracts in and pops it when the
                        // path refracts out.
                        // Metal's interface_select_medium (visibility.metal) comes first: an explicit interior selects the
                        // medium by the outgoing side of the outward (unfaced) normal; an inheriting interface keeps the
                        // current medium.
                        if (pt_shared::source_material_flag(material, PathTracerShaderTypes::kMaterialFlagExplicitInterior)) {
                            const bool inside = simd::dot(bsdfSample.direction, hit.normal) < 0.0f;
                            mediumSigmaStack[0] = inside ? DielectricSigmaA(material) : simd_make_float3(0.0f, 0.0f, 0.0f);
                            mediumDepth = inside ? 1u : 0u;
                        } else if (pt_shared::source_material_flag(material, PathTracerShaderTypes::kMaterialFlagMediumInherit)) {
                        } else if (bsdfSample.mediumEvent == 1) {
                            mediumSigmaStack[std::min(mediumDepth, kMaxMediumStack - 1u)] = DielectricSigmaA(material);
                            mediumDepth = std::min(mediumDepth + 1u, kMaxMediumStack);
                        } else if (bsdfSample.mediumEvent == -1 && mediumDepth > 0u) {
                            --mediumDepth;
                        }

                        legacyRaySpawn = false;
                        preciseRaySpawn = InterfaceSpawn(&material);
                        holeMisOriginValid = false;
                        lastScatterWasVolume = false;
                        ray.origin = SpawnOrigin(hit, bsdfSample.direction, &material);
                        ray.direction = bsdfSample.direction;

                        if (settings.enableRussianRoulette && depth >= 5) {
                            float rrProb = std::clamp(maxComp, 0.05f, 0.95f);
                            if (rng.nextFloat() > rrProb) {
                                break;
                            }
                            throughput /= rrProb;
                        }
                    }

                    for (uint32_t slot = 0u; slot < PathTracer::RadianceCacheTraining::kTrainRecordCount; ++slot) {
                        if ((cacheTrainValidMask & (1u << slot)) != 0u) {
                            trainRadianceCache(cacheTrainRecords[slot], radiance, s);
                        }
                    }
                    if (spectral.enabled) {
                        radiance = SpectralChannelsToRgb(radiance, spectral, spectralConfig);
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
            volumeScatterEventCount.fetch_add(tileVolumeScatterCount, std::memory_order_relaxed);
            spectralCollapseCount.fetch_add(tileSpectralCollapseCount, std::memory_order_relaxed);
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
    if (spectralConfig.enabled) {
        std::cout << "[EmbreeSpectral] " << spectralCollapseCount.load(std::memory_order_relaxed)
                  << " paths collapsed to the hero wavelength at a dispersive vertex" << std::endl;
    }
    if (volumeMedium.enabled && volumeMedium.scatters) {
        std::cout << "[EmbreeVolume] " << volumeScatterEventCount.load(std::memory_order_relaxed)
                  << " medium scatter events (max " << volumeMedium.maxScatteringEvents << " per path, phase "
                  << (volumeMedium.henyeyGreenstein ? "Henyey-Greenstein" : "isotropic") << ", volume NEE "
                  << (volumeMedium.nee ? "on" : "off") << ")" << std::endl;
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
    // ReSTIR DI runs inside the path loop, as on Metal's fused path, which dispatches no screen-space passes.
    out.performance.integrationDispatchCount = targetSamples;
    out.performance.totalComputeDispatchCount = out.performance.integrationDispatchCount;
    out.performance.sssMaxSteps = settings.sssMaxSteps;
    for (uint32_t m = 0; m < PathTracerShaderTypes::kMaxMaterials; ++m) {
        const size_t base = static_cast<size_t>(m) * PathTracerShaderTypes::kSssWalkStatCount;
        if (sssWalkStats.counts[base + PathTracerShaderTypes::kSssWalkStatWalks].load() == 0u) {
            continue;
        }
        HeadlessRenderOutput::PerformanceSummary::SssWalkMaterialStats entry;
        entry.material = m < materialCount ? scene.resources->materialName(m) : std::string();
        for (uint32_t k = 0; k < PathTracerShaderTypes::kSssWalkStatCount; ++k) {
            entry.stats[k] = sssWalkStats.counts[base + k].load();
            out.performance.sssWalkStats[k] += entry.stats[k];
        }
        out.performance.sssWalkMaterials.push_back(std::move(entry));
    }
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
    for (uint32_t i = 0u; i < PathTracer::RadianceCacheTraining::kStatCount; ++i) {
        out.pbrMetrics.radianceCacheStats[i] = radianceCacheStats[i].load(std::memory_order_relaxed);
    }
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
