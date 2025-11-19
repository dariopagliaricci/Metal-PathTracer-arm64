#pragma once

#include <cstdint>
#include <simd/simd.h>

#ifdef __METAL_VERSION__
#include <metal_atomic>
using PathtraceStatsAtomic = atomic_uint;
#else
using PathtraceStatsAtomic = uint32_t;
#endif

namespace PathTracerShaderTypes {

constexpr uint32_t kMaxSpheres = 512;
constexpr uint32_t kMaxMaterials = 512;
constexpr uint32_t kMaxRectangles = 128;

enum class IntersectionMode : uint32_t {
    SoftwareBVH = 0,
    HardwareRayTracing = 1,
};

enum class SoftwareBvhType : uint32_t {
    None = 0,
    Spheres = 1,
    Triangles = 2,
};

enum class MaterialType : uint32_t {
    Lambertian = 0,
    Metal = 1,
    Dielectric = 2,
    DiffuseLight = 3,
    Plastic = 4,
    Subsurface = 5,
    CarPaint = 6,
};

struct SphereData {
    simd::float4 centerRadius;   // xyz = center, w = radius
    simd::uint4 materialIndex;   // x = index
};

struct RectData {
    simd::float4 corner;          // xyz = rectangle corner, w unused
    simd::float4 edgeU;           // xyz = first edge vector, w = inverse length squared
    simd::float4 edgeV;           // xyz = second edge vector, w = inverse length squared
    simd::float4 normalAndPlane;  // xyz = normalized normal, w = plane constant (dot(normal, corner))
    simd::uint4 materialTwoSided; // x = material index, y = two-sided flag
};

struct MaterialData {
    simd::float4 baseColorRoughness;  // xyz = base color/F0 tint, w = roughness
    simd::float4 typeEta;             // x = type, y = base IOR, z = coat IOR, w unused
    simd::float4 emission;            // xyz = emission radiance, w = env-sampled flag (1 = sample environment)
    simd::float4 conductorEta;        // xyz = conductor eta, w = flag (>0 when valid)
    simd::float4 conductorK;          // xyz = conductor k, w = flag (>0 when valid)
    simd::float4 coatParams;          // x = coat roughness, y = coat thickness, z = coat sample weight, w = coat Fresnel average
    simd::float4 coatTint;            // xyz = coat tint, w unused
    simd::float4 coatAbsorption;      // xyz = coat absorption coefficient, w unused
    simd::float4 dielectricSigmaA;    // xyz = glass absorption (per meter), w unused
    simd::float4 sssSigmaA;           // xyz = sigma_a, w = override flag (1 = explicit sigma)
    simd::float4 sssSigmaS;           // xyz = sigma_s, w = anisotropy g
    simd::float4 sssParams;           // x = mean free path, y = method, z = coat enabled flag, w unused
    simd::float4 carpaintBaseParams;  // x = base metallic, y = base roughness, z = flake scale, w = flake reflectance scale
    simd::float4 carpaintFlakeParams; // x = flake sample weight, y = flake roughness, z = flake anisotropy, w = flake normal strength
    simd::float4 carpaintBaseEta;     // xyz = carpaint base eta, w = flag (>0 when valid)
    simd::float4 carpaintBaseK;       // xyz = carpaint base k, w = flag (>0 when valid)
    simd::float4 carpaintBaseTint;    // xyz = conductor tint multiplier, w unused
};

struct EnvironmentAliasEntry {
    float threshold = 1.0f;
    uint32_t alias = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
};

struct alignas(16) BvhNode {
    simd::float4 boundsMin = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 boundsMax = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t leftChild = 0;
    uint32_t rightChild = 0;
    uint32_t primitiveOffset = 0;
    uint32_t primitiveCount = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
};

struct PathtraceUniforms {
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t frameIndex = 0;
    uint32_t sampleCount = 0;

    simd::float3 cameraOrigin = {0.0f, 0.0f, 0.0f};
    float cameraPad0 = 0.0f;
    simd::float3 lowerLeftCorner = {0.0f, 0.0f, -1.0f};
    float cameraPad1 = 0.0f;
    simd::float3 horizontal = {1.0f, 0.0f, 0.0f};
    float cameraPad2 = 0.0f;
    simd::float3 vertical = {0.0f, 1.0f, 0.0f};
    float cameraPad3 = 0.0f;
    simd::float3 cameraU = {1.0f, 0.0f, 0.0f};
    float lensRadius = 0.0f;
    simd::float3 cameraV = {0.0f, 1.0f, 0.0f};
    float cameraPad4 = 0.0f;

    uint32_t sphereCount = 0;
    uint32_t rectangleCount = 0;
    uint32_t materialCount = 0;
    uint32_t maxDepth = 5;
    uint32_t useRussianRoulette = 1;
    uint32_t intersectionMode = static_cast<uint32_t>(IntersectionMode::SoftwareBVH);
    uint32_t softwareBvhType = static_cast<uint32_t>(SoftwareBvhType::None);
    uint32_t primitiveCount = 0;
    uint32_t meshCount = 0;
    uint32_t triangleCount = 0;
    uint32_t fixedRngSeed = 0;  // If non-zero, use as deterministic RNG seed
    uint32_t backgroundMode = 0;
    float environmentRotation = 0.0f;   // Radians, rotates env map around world Y
    float environmentIntensity = 1.0f;  // Multiplier applied to environment sampling
    float padding0 = 0.0f;
    simd::float3 backgroundColor = {0.0f, 0.0f, 0.0f};
    uint32_t environmentAliasCount = 0;
    uint32_t environmentMapWidth = 0;
    uint32_t environmentMapHeight = 0;
    uint32_t environmentHasDistribution = 0;
    uint32_t fireflyClampEnabled = 0;
    float fireflyClampFactor = 0.0f;
    float fireflyClampFloor = 0.0f;
    float throughputClamp = 0.0f;
    float specularTailClampBase = 0.0f;
    float specularTailClampRoughnessScale = 0.0f;
    float minSpecularPdf = 0.0f;
    float paddingFirefly = 0.0f;
    uint32_t sssMode = 0;
    uint32_t sssMaxSteps = 0;
    uint32_t paddingSss0 = 0;
    uint32_t enableSpecularNee = 1;
    uint32_t debugPathActive = 0;
    uint32_t debugPixelX = 0;
    uint32_t debugPixelY = 0;
    uint32_t debugMaxEntries = 0;
};

struct PathtraceStats {
    PathtraceStatsAtomic primaryRayCount;
    PathtraceStatsAtomic nodesVisited;
    PathtraceStatsAtomic leafPrimTests;
    PathtraceStatsAtomic internalNodeVisits;
    PathtraceStatsAtomic internalBothVisited;
    PathtraceStatsAtomic shadowRayCount;
    PathtraceStatsAtomic shadowRayEarlyExitCount;
    PathtraceStatsAtomic hardwareRayCount;
    PathtraceStatsAtomic hardwareHitCount;
    PathtraceStatsAtomic hardwareMissCount;
    // Extended diagnostics for hardware intersector results
    PathtraceStatsAtomic hardwareResultNoneCount;
    PathtraceStatsAtomic hardwareRejectedCount;
    PathtraceStatsAtomic hardwareUnavailableCount;
    PathtraceStatsAtomic hardwareLastResultType;
    PathtraceStatsAtomic hardwareLastInstanceId;
    PathtraceStatsAtomic hardwareLastPrimitiveId;
    PathtraceStatsAtomic hardwareLastDistanceBits;
    PathtraceStatsAtomic specularNeeOcclusionHitCount;
    PathtraceStatsAtomic hardwareSelfHitRejectedCount;
    PathtraceStatsAtomic hardwareMissDistanceBins[32];
    PathtraceStatsAtomic hardwareMissLastDistanceBits;
    PathtraceStatsAtomic hardwareSelfHitLastDistanceBits;
};

constexpr uint32_t kPathtraceDebugMaxEntries = 512;

struct PathtraceDebugEntry {
    uint32_t integrator = 0;         // 0 = SWRT, 1 = HWRT
    uint32_t pixelX = 0;
    uint32_t pixelY = 0;
    uint32_t sampleIndex = 0;
    uint32_t depth = 0;
    uint32_t mediumDepthBefore = 0;
    uint32_t mediumDepthAfter = 0;
    int32_t mediumEvent = 0;         // -1 exit, 0 none, 1 enter
    uint32_t frontFace = 0;
    uint32_t scatterIsDelta = 0;
    uint32_t materialIndex = 0;
    uint32_t reserved0 = 0;
    simd::float4 throughput = {0.0f, 0.0f, 0.0f, 0.0f};
};

struct PathtraceDebugBuffer {
    PathtraceStatsAtomic writeIndex;
    uint32_t maxEntries = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    PathtraceDebugEntry entries[kPathtraceDebugMaxEntries];
};

struct MeshInfo {
    uint32_t materialIndex = 0;
    uint32_t triangleOffset = 0;
    uint32_t triangleCount = 0;
    uint32_t vertexOffset = 0;
    uint32_t vertexCount = 0;
    uint32_t indexOffset = 0;
    uint32_t indexCount = 0;
    simd::float4x4 localToWorld = matrix_identity_float4x4;
    simd::float4x4 worldToLocal = matrix_identity_float4x4;
};

struct SceneVertex {
    simd::float4 position = {0.0f, 0.0f, 0.0f, 1.0f};
    simd::float4 normal = {0.0f, 1.0f, 0.0f, 0.0f};
    simd::float4 tangent = {1.0f, 0.0f, 0.0f, 1.0f};
    simd::float4 uv = {0.0f, 0.0f, 0.0f, 0.0f};
};

struct TriangleData {
    simd::float4 v0 = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 v1 = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 v2 = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::uint4 metadata = {0u, 0u, 0u, 0u};  // x = material index
};

struct SoftwareInstanceInfo {
    uint32_t blasRootNodeOffset = 0;      // Root node index in packed BLAS nodes
    uint32_t blasNodeCount = 0;           // Number of nodes for this instance's BLAS
    uint32_t blasPrimIndexOffset = 0;     // Start in packed BLAS primitiveIndices
    uint32_t blasPrimIndexCount = 0;      // Count in packed BLAS primitiveIndices
    uint32_t triangleBaseOffset = 0;      // Start in global triangleData buffer
    uint32_t triangleCount = 0;           // Number of triangles in this instance
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
    simd::float4x4 localToWorld = matrix_identity_float4x4;
    simd::float4x4 worldToLocal = matrix_identity_float4x4;
};

struct DisplayUniforms {
    uint32_t tonemapMode = 1;   // 1 = Linear, 2 = ACES, 3 = Reinhard, 4 = Hable
    uint32_t acesVariant = 0;   // 0 = Fitted, 1 = Simple
    float exposure = 0.0f;      // Exposure adjustment in stops
    float reinhardWhite = 1.5f; // White point for Reinhard operator
};

}  // namespace PathTracerShaderTypes
