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
// Full-scene Bistro content exceeds 256 unique runtime texture slots even after
// payload deduplication. Raise the cap so the combined asset can load cleanly
// and defer memory enforcement to recommendedMaxWorkingSetSize checks.
constexpr uint32_t kMaxMaterialTextures = 512;
constexpr uint32_t kMaxMaterialSamplers = 14;
constexpr uint32_t kMaterialResourcePathGuidingStateIndex =
    kMaxMaterialTextures + kMaxMaterialSamplers;
constexpr uint32_t kMaterialResourceRestirPtStateIndex =
    kMaterialResourcePathGuidingStateIndex + 1u;
constexpr uint32_t kMaterialResourceRadianceCacheStateIndex =
    kMaterialResourceRestirPtStateIndex + 1u;
constexpr uint32_t kWavefrontShadowCapacityMultiplier = 4;
constexpr uint32_t kRestirPathReservoirDepthCount = 8;
constexpr uint32_t kMaterialFlagDisableOrm = 1u << 0;
constexpr uint32_t kMaterialFlagForceBaseColorMip0 = 1u << 1;
constexpr uint32_t kMaterialFlagBlackKeyAlphaFromRgb = 1u << 2;
constexpr uint32_t kMaterialFlagCameraVisibleEmissionTint = 1u << 3;
constexpr uint32_t kLightPrimitiveFlagDirectional = 1u << 0;

enum class IntersectionMode : uint32_t {
    SoftwareBVH = 0,
    HardwareRayTracing = 1,
};

enum class SoftwareBvhType : uint32_t {
    None = 0,
    Spheres = 1,
    Triangles = 2,
};

enum class RenderExecutionMode : uint32_t {
    Megakernel = 0,
    Wavefront = 1,
};

enum class WavefrontSchedulingPolicy : uint32_t {
    Preview = 0,
    Final = 1,
    Offline = 2,
    Research = 3,
};

enum class MaterialType : uint32_t {
    Lambertian = 0,
    Metal = 1,
    Dielectric = 2,
    DiffuseLight = 3,
    Plastic = 4,
    Subsurface = 5,
    CarPaint = 6,
    PbrMetallicRoughness = 7,
};

enum class MaterialClosureType : uint32_t {
    Diffuse = 0,
    Conductor = 1,
    Dielectric = 2,
    Transmission = 3,
    Clearcoat = 4,
    Sheen = 5,
    CarPaint = 6,
    Subsurface = 7,
    VolumeBoundary = 8,
    Emission = 9,
    PbrDiffuse = 10,
    PbrSpecular = 11,
};

constexpr uint32_t kMaterialClosureFlagDelta = 1u << 0;
constexpr uint32_t kMaterialClosureFlagConnectable = 1u << 1;
constexpr uint32_t kMaterialClosureFlagGuideable = 1u << 2;
constexpr uint32_t kMaterialClosureFlagCacheable = 1u << 3;
constexpr uint32_t kMaterialClosureFlagEmissive = 1u << 4;
constexpr uint32_t kMaterialClosureFlagTransmissive = 1u << 5;
constexpr uint32_t kMaterialClosureFlagVolumetric = 1u << 6;
constexpr uint32_t kMaterialClosureFlagGlossy = 1u << 7;
constexpr uint32_t kMaterialClosureFlagDiffuse = 1u << 8;
constexpr uint32_t kMaterialClosureFlagSpecular = 1u << 9;

struct MaterialClosureMetadata {
    uint32_t type = static_cast<uint32_t>(MaterialClosureType::Diffuse);
    uint32_t flags = 0;
    float weight = 1.0f;
    float roughness = 1.0f;
    float eta = 1.0f;
    uint32_t lobeIndex = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    simd::float4 aovAlbedoRoughness = {0.0f, 0.0f, 0.0f, 1.0f};
    simd::float4 aovSpecularTransmission = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 aovEmission = {0.0f, 0.0f, 0.0f, 0.0f};
};

enum class AreaPrimitiveShape : uint32_t {
    Rectangle = 0,
    Disk = 1,
};

struct SphereData {
    simd::float4 centerRadius;   // xyz = center, w = radius
    simd::uint4 materialIndex;   // x = index
};

struct RectData {
    simd::float4 corner;          // rectangle: xyz = corner; disk: xyz = center
    simd::float4 edgeU;           // rectangle: edge vector; disk: scaled tangent/radius axis
    simd::float4 edgeV;           // rectangle: edge vector; disk: scaled bitangent/radius axis
    simd::float4 normalAndPlane;  // xyz = normalized normal, w = plane constant
    simd::uint4 materialTwoSided; // x = material index, y = two-sided flag, z = AreaPrimitiveShape
};

struct MaterialData {
    simd::float4 baseColorRoughness;  // xyz = base color/F0 tint, w = roughness
    simd::float4 typeEta;             // x = type, y = base IOR, z = coat IOR / PBR double-sided, w = thin dielectric flag / PBR thickness
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
    simd::uint4 textureIndices0;      // x = baseColor, y = metallicRoughness, z = normal, w = occlusion
    simd::uint4 textureIndices1;      // x = emissive, yzw unused
    uint32_t materialFlags;           // bitfield (kMaterialFlag*)
    uint32_t interiorMediumId;        // 0 = vacuum; front-face transmission enters this medium
    uint32_t exteriorMediumId;        // 0 = vacuum; back-face transmission exits to this medium
    uint32_t materialPad2;
    simd::float4 spectralParams;      // x = dispersion scale, y = spectral absorption scale, z/w reserved
    simd::float4 pbrParams;           // x = metallic, y = roughness, z = occlusion strength, w = normal scale
    simd::float4 pbrExtras;           // x = alpha factor, y = alpha cutoff, z = transmission factor, w = alpha mode
    simd::uint4 textureUvSet0;        // x = baseColor, y = metallicRoughness, z = normal, w = occlusion
    simd::uint4 textureUvSet1;        // x = emissive, y = transmission, zw unused
    simd::float4 textureTransform0;   // baseColor: row0 = (m00, m01, m02)
    simd::float4 textureTransform1;   // baseColor: row1 = (m10, m11, m12)
    simd::float4 textureTransform2;   // metallicRoughness: row0
    simd::float4 textureTransform3;   // metallicRoughness: row1
    simd::float4 textureTransform4;   // normal: row0
    simd::float4 textureTransform5;   // normal: row1
    simd::float4 textureTransform6;   // occlusion: row0
    simd::float4 textureTransform7;   // occlusion: row1
    simd::float4 textureTransform8;   // emissive: row0
    simd::float4 textureTransform9;   // emissive: row1
    simd::float4 textureTransform10;  // transmission: row0
    simd::float4 textureTransform11;  // transmission: row1
};

struct EnvironmentAliasEntry {
    float threshold = 1.0f;
    uint32_t alias = 0;
    uint32_t padding0 = 0;
    uint32_t padding1 = 0;
};

struct MediumDescriptor {
    simd::float4 sigmaA;              // xyz absorption, w sigma_t majorant
    simd::float4 sigmaSAnisotropy;    // xyz scattering, w Henyey-Greenstein g
    simd::float4 emissionPhase;       // xyz emission, w phase function id
    simd::uint4 metadata;             // x medium id, y max scattering events, z NEE enabled, w flags
};

struct MediumState {
    uint32_t mediumId = 0;
    uint32_t scatteringDepth = 0;
    uint32_t phaseFunction = 0;
    uint32_t flags = 0;
    simd::float4 transmittance = {1.0f, 1.0f, 1.0f, 0.0f};
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
    uint32_t executionMode = static_cast<uint32_t>(RenderExecutionMode::Megakernel);
    uint32_t wavefrontSchedulingPolicy = static_cast<uint32_t>(WavefrontSchedulingPolicy::Preview);
    uint32_t wavefrontCompactionEnabled = 0;
    uint32_t wavefrontDebugStageFailures = 0;

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
    uint32_t materialTextureCount = 0;
    uint32_t maxDepth = 5;
    uint32_t useRussianRoulette = 1;
    uint32_t intersectionMode = static_cast<uint32_t>(IntersectionMode::SoftwareBVH);
    uint32_t softwareBvhType = static_cast<uint32_t>(SoftwareBvhType::None);
    uint32_t primitiveCount = 0;
    uint32_t meshCount = 0;
    uint32_t triangleCount = 0;
    uint32_t emissivePrimitiveCount = 0;
    uint32_t fixedRngSeed = 0;  // If non-zero, use as deterministic RNG seed
    uint32_t backgroundMode = 0;
    uint32_t workingColorSpace = 0;
    float environmentRotation = 0.0f;   // Radians, rotates env map around world Y
    float environmentIntensity = 1.0f;  // Multiplier applied to environment sampling
    float emissiveTotalPower = 0.0f;
    simd::float3 backgroundColor = {0.0f, 0.0f, 0.0f};
    uint32_t environmentAliasCount = 0;
    uint32_t environmentMapWidth = 0;
    uint32_t environmentMapHeight = 0;
    uint32_t environmentHasDistribution = 0;
    uint32_t fireflyClampEnabled = 0;
    uint32_t fireflyClampMode = 0;
    float fireflyClampFactor = 0.0f;
    float fireflyClampFloor = 0.0f;
    float throughputClamp = 0.0f;
    float specularTailClampBase = 0.0f;
    float specularTailClampRoughnessScale = 0.0f;
    float minSpecularPdf = 0.0f;
    float fireflyClampMaxContribution = 0.0f;
    float paddingFirefly = 0.0f;
    uint32_t sssMode = 0;
    uint32_t sssMaxSteps = 0;
    uint32_t directLightMode = 0;
    uint32_t enableSpecularNee = 1;
    uint32_t enableMnee = 0;
    uint32_t enableMneeSecondary = 1;
    uint32_t risCandidateCount = 8;
    uint32_t spatialReuseNeighborCount = 4;
    float worldReuseCellSize = 0.5f;
    uint32_t restirGiMode = 0;
    uint32_t pathGuidingMode = 0;
    float pathGuidingStrength = 0.0f;
    float pathGuidingCellSize = 1.0f;
    uint32_t pathGuidingTrainingInterval = 1;
    uint32_t restirPtMode = 0;
    uint32_t restirPtMaxReservoirs = 4096;
    uint32_t restirPtDebug = 0;
    float restirPtReuseStrength = 0.0f;
    uint32_t radianceCacheMode = 0;
    float radianceCacheCellSize = 1.0f;
    uint32_t radianceCacheMinDepth = 2;
    float radianceCacheMinConfidence = 0.35f;
    uint32_t radianceCacheTrainingInterval = 1;
    float radianceCacheMaxRadiance = 16.0f;
    uint32_t radianceCacheDebug = 0;
    uint32_t radianceCachePadding0 = 0;
    uint32_t causticTransportMode = 0;
    uint32_t causticMaxVertices = 4096;
    uint32_t volumeTransportMode = 0;
    uint32_t volumePhaseFunction = 0;
    simd::float4 volumeSigmaA = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 volumeSigmaSAnisotropy = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 volumeEmissionMaxEvents = {0.0f, 0.0f, 0.0f, 8.0f};
    uint32_t volumeNeeEnabled = 1;
    uint32_t volumeDebug = 0;
    uint32_t spectralRenderingMode = 0;
    uint32_t spectralTexturePolicy = 0;
    simd::float4 spectralWavelengthRange = {380.0f, 720.0f, 0.012f, 0.0f};
    uint32_t spectralDebug = 0;
    uint32_t spectralPadding0 = 0;
    uint32_t spectralPadding1 = 0;
    uint32_t spectralPadding2 = 0;
    uint32_t restirDebugEnabled = 0;
    uint32_t restirDebugView = 0;
    uint32_t restirDebugCounters = 0;
    uint32_t restirDebugFrameIndex = 0;
    uint32_t debugPathActive = 0;
    uint32_t debugPixelX = 0;
    uint32_t debugPixelY = 0;
    uint32_t debugMaxEntries = 0;
    uint32_t hardwareExcludeMaxAttempts = 4;
    float hardwareExitNormalBias = 0.0f;
    float hardwareExitDirectionalBias = 0.0f;
    uint32_t enableHardwareMissFallback = 0;
    uint32_t enableHardwareFirstHitFromSoftware = 0;
    uint32_t enableHardwareForceSoftware = 0;
    uint32_t paddingHardware1 = 0;
    uint32_t parityAssertEnabled = 0;
    uint32_t parityAssertMode = 0;
    uint32_t parityPixelX = 0;
    uint32_t parityPixelY = 0;
    uint32_t parityOncePerFrame = 1;
    uint32_t parityPadding0 = 0;
    uint32_t explicitRestirDiLightingStage = 0;
    uint32_t parityPadding2 = 0;
    uint32_t forcePureHWRTForGlass = 0;
    // When non-zero, shadow rays treat near-origin hits found only via the
    // tMin=0 retry (t < kEpsilon, different mesh) as false-positive occlusion
    // and skip them — matching SWRT behaviour.
    uint32_t enableHardwareShadowNearRetryReject = 0;
    uint32_t parityPadding4 = 0;
    uint32_t parityPadding5 = 0;
    uint32_t debugViewMode = 0;
    uint32_t debugDisableAO = 0;
    uint32_t debugAoIndirectOnly = 1;
    uint32_t debugDisableNormalMap = 0;
    uint32_t debugDisableOrmTexture = 0;
    uint32_t debugFlipNormalGreen = 0;
    uint32_t debugSpecularOnly = 0;
    uint32_t debugDirectLightAudit = 0;
    float debugNormalStrengthScale = 1.0f;
    float debugNormalLodBias = 0.0f;
    float debugOrmLodBias = 0.0f;
    float debugEnvMipOverride = -1.0f;
    uint32_t debugEnableVisorOverride = 0;
    int32_t debugVisorOverrideMaterialId = -1;
    float debugVisorOverrideRoughness = 0.15f;
    float debugVisorOverrideF0 = 0.04f;
    uint32_t debugEnvNearest = 0;
    uint32_t debugPadding0 = 0;
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
    PathtraceStatsAtomic specNeeEnvAddedCount;
    PathtraceStatsAtomic specNeeRectAddedCount;
    PathtraceStatsAtomic mneeEnvHwOccludedCount;
    PathtraceStatsAtomic mneeEnvSwOccludedCount;
    PathtraceStatsAtomic mneeEnvHwSwMismatchCount;
    PathtraceStatsAtomic mneeRectHwOccludedCount;
    PathtraceStatsAtomic mneeRectSwOccludedCount;
    PathtraceStatsAtomic mneeRectHwSwMismatchCount;
    PathtraceStatsAtomic mneeHitHwSwHitMissCount;
    PathtraceStatsAtomic mneeHitHwSwNormalMismatchCount;
    PathtraceStatsAtomic mneeHitHwSwIdMismatchCount;
    PathtraceStatsAtomic mneeHitHwSwTDiffCount;
    PathtraceStatsAtomic mneeChainHwSwHitMissCount;
    PathtraceStatsAtomic mneeChainHwSwNormalMismatchCount;
    PathtraceStatsAtomic mneeChainHwSwIdMismatchCount;
    PathtraceStatsAtomic mneeChainHwSwTDiffCount;
    PathtraceStatsAtomic mneeEligibleCount;
    PathtraceStatsAtomic mneeEnvAttemptCount;
    PathtraceStatsAtomic mneeEnvAddedCount;
    PathtraceStatsAtomic mneeEnvChainUnsupportedCount;
    PathtraceStatsAtomic mneeEnvChainNonDeltaBlockCount;
    PathtraceStatsAtomic mneeEnvChainSampleRejectCount;
    PathtraceStatsAtomic mneeEnvChainWeightRejectCount;
    PathtraceStatsAtomic mneeEnvChainMaxBounceCount;
    PathtraceStatsAtomic mneeRectAttemptCount;
    PathtraceStatsAtomic mneeRectAddedCount;
    PathtraceStatsAtomic mneeContributionCount;
    PathtraceStatsAtomic mneeContributionLumaSumLo;
    PathtraceStatsAtomic mneeContributionLumaSumHi;
    PathtraceStatsAtomic hardwareSelfHitRejectedCount;
    PathtraceStatsAtomic hardwareMissDistanceBins[32];
    PathtraceStatsAtomic hardwareMissLastDistanceBits;
    PathtraceStatsAtomic hardwareSelfHitLastDistanceBits;
    PathtraceStatsAtomic hardwareExcludeRetryHistogram[4];
    PathtraceStatsAtomic hardwareMissLastInstanceId;
    PathtraceStatsAtomic hardwareMissLastPrimitiveId;
    PathtraceStatsAtomic hardwareFallbackHitCount;
    PathtraceStatsAtomic hardwareFirstHitFallbackCount;
    PathtraceStatsAtomic throughputNanInfCount;
    PathtraceStatsAtomic radianceNanInfCount;
    PathtraceStatsAtomic clampEventCount;
    PathtraceStatsAtomic restirGiCandidateCount;
    PathtraceStatsAtomic restirGiReservoirUpdateCount;
    PathtraceStatsAtomic restirGiAcceptCount;
    PathtraceStatsAtomic restirGiRejectCount;
    PathtraceStatsAtomic restirGiFallbackCount;
    PathtraceStatsAtomic pathGuidingCandidateCount;
    PathtraceStatsAtomic pathGuidingUsedCount;
    PathtraceStatsAtomic pathGuidingFallbackCount;
    PathtraceStatsAtomic pathGuidingInvalidCount;
    PathtraceStatsAtomic restirPtCandidateCount;
    PathtraceStatsAtomic restirPtReservoirUpdateCount;
    PathtraceStatsAtomic restirPtRejectedInvalidCount;
    PathtraceStatsAtomic restirPtRejectedUnsupportedCount;
    PathtraceStatsAtomic restirPtDebugRecordCount;
    PathtraceStatsAtomic restirPtReuseCandidateCount;
    PathtraceStatsAtomic restirPtReuseAppliedCount;
    PathtraceStatsAtomic restirPtReuseRejectedInvalidCount;
    PathtraceStatsAtomic restirPtReuseFallbackCount;
    PathtraceStatsAtomic radianceCacheQueryCount;
    PathtraceStatsAtomic radianceCacheHitCount;
    PathtraceStatsAtomic radianceCacheMissCount;
    PathtraceStatsAtomic radianceCacheTrainCount;
    PathtraceStatsAtomic radianceCacheFallbackCount;
    PathtraceStatsAtomic radianceCacheInvalidCount;
    PathtraceStatsAtomic causticEyeVertexCount;
    PathtraceStatsAtomic causticLightVertexCount;
    PathtraceStatsAtomic causticConnectionCandidateCount;
    PathtraceStatsAtomic causticDeltaChainCount;
    PathtraceStatsAtomic causticDebugClassificationCount;
    PathtraceStatsAtomic volumeTransmittanceQueryCount;
    PathtraceStatsAtomic volumeScatterEventCount;
    PathtraceStatsAtomic volumeNeeRayCount;
    PathtraceStatsAtomic volumePhaseSampleCount;
    PathtraceStatsAtomic volumeMediumBoundaryEventCount;
    PathtraceStatsAtomic volumeFallbackCount;
    PathtraceStatsAtomic spectralWavelengthSampleCount;
    PathtraceStatsAtomic spectralMaterialLookupCount;
    PathtraceStatsAtomic spectralDispersionEventCount;
    PathtraceStatsAtomic spectralFallbackCount;
    // Shadow-ray tMin=0 retry diagnostics.
    // Incremented each time the none→tMin=0 retry fires for a shadow (anyHit) ray.
    PathtraceStatsAtomic hardwareShadowTMinRetryCount;
    // Incremented when the tMin=0 retry finds a hit at t < kEpsilon on a mesh that
    // is NOT the origin surface (not caught by selfHit or the excluded-mesh filter).
    // A high count here indicates false-positive occlusion diverging from SWRT.
    PathtraceStatsAtomic hardwareShadowNearForeignHitCount;
    PathtraceStatsAtomic wavefrontActiveRayCount;
    PathtraceStatsAtomic wavefrontNextRayCount;
    PathtraceStatsAtomic wavefrontShadowRayCount;
    PathtraceStatsAtomic wavefrontTerminatedCount;
    PathtraceStatsAtomic wavefrontOverflowFlag;
    PathtraceStatsAtomic wavefrontTerminateMissCount;
    PathtraceStatsAtomic wavefrontTerminateEmissiveCount;
    PathtraceStatsAtomic wavefrontTerminateMaxDepthCount;
    PathtraceStatsAtomic wavefrontTerminateOverflowCount;
    PathtraceStatsAtomic wavefrontEnvShadowRayCount;
    PathtraceStatsAtomic wavefrontAovAlbedoValidCount;
    PathtraceStatsAtomic wavefrontAovNormalValidCount;
    PathtraceStatsAtomic wavefrontPrimaryGeneratedCount;
    PathtraceStatsAtomic wavefrontIntersectProcessedCount;
    PathtraceStatsAtomic wavefrontShadeProcessedCount;
    PathtraceStatsAtomic wavefrontShadowProcessedCount;
    PathtraceStatsAtomic wavefrontNextRayEnqueueCount;
    PathtraceStatsAtomic wavefrontShadowRayEnqueueCount;
};

constexpr uint32_t kWavefrontMediumStackSize = 8u;

struct WavefrontRay {
    simd::float4 originMinT = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 directionMaxT = {0.0f, 0.0f, -1.0f, 1.0e20f};
    simd::float4 rayDiffDOdx = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 rayDiffDOdy = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 rayDiffDDdx = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 rayDiffDDdy = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float2 cone = {0.0f, 0.0f};  // x = width, y = spread
    uint32_t excludeMeshIndex = 0;
    uint32_t excludePrimitiveIndex = 0;
    uint32_t pixelIndex = 0;
    uint32_t pathIndex = 0;
    uint32_t depth = 0;
    uint32_t flags = 0;
};

// Wavefront path state layout: AoS, one PathState per path index.
// Large float4 fields stay together today to preserve simple indexing across
// megakernel parity checks and future typed-queue compaction checkpoints.
struct PathState {
    simd::float4 throughput = {1.0f, 1.0f, 1.0f, 0.0f};
    simd::float4 radiance = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 firstHitAlbedo = {0.0f, 0.0f, 0.0f, 0.0f};  // w = valid
    simd::float4 firstHitNormal = {0.0f, 0.0f, 0.0f, 0.0f};  // xyz in [-1, 1]
    simd::float4 firstHitPosition = {0.0f, 0.0f, 0.0f, 0.0f}; // xyz = world position, w = valid
    simd::float4 firstHitMaterial = {0.0f, 0.0f, 0.0f, 0.0f}; // roughness, transmission, specular luma, material type
    float envLod = 0.0f;
    float lastBsdfPdf = 1.0f;
    uint32_t rngState = 0;
    uint32_t mediumDepth = 0;
    uint32_t alive = 0;
    uint32_t flags = 0;  // bit0: envLodActive, bit1: lastScatterWasDelta
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    uint32_t currentMediumId = 0;
    uint32_t volumeScatteringDepth = 0;
    float spectralWavelengthNm = 0.0f;
    uint32_t spectralSampleIndex = 0;
    uint32_t spectralPadding0 = 0;
    uint32_t spectralPadding1 = 0;
    simd::float4 mediumStack[kWavefrontMediumStackSize] = {};
};

struct PathGuidingReservoirState {
    simd::float4 directionConfidence = {0.0f, 1.0f, 0.0f, 0.0f}; // xyz = bin mean direction, w = confidence
    simd::float4 weightMoment = {0.0f, 0.0f, 0.0f, 0.0f};        // debug: x = luma, y = last luma, z = guiding pdf, w = guided choice
    uint32_t sampleCount = 0;                                     // integer training count
    uint32_t frameTag = 0;                                        // last update frame
    uint32_t flags = 0;                                           // bit0 = valid, bit1 = last update guided
    uint32_t reserved0 = 0;                                       // quantized luminance accumulator
};

struct RestirPtReservoirState {
    simd::float4 positionPdf = {0.0f, 0.0f, 0.0f, 0.0f};          // xyz = hit position, w = pdf
    simd::float4 normalDepth = {0.0f, 1.0f, 0.0f, 0.0f};          // xyz = normal, w = path depth
    simd::float4 directionWeight = {0.0f, 1.0f, 0.0f, 0.0f};      // xyz = sampled direction, w = sample luma
    simd::float4 throughputLuma = {0.0f, 0.0f, 0.0f, 0.0f};       // xyz = throughput, w = throughput luma
    uint32_t materialType = 0;
    uint32_t lobeType = 0;
    uint32_t frameTag = 0;
    uint32_t flags = 0;                                           // bit0 = valid
};

struct RadianceQuery {
    simd::float4 positionRoughness = {0.0f, 0.0f, 0.0f, 1.0f};
    simd::float4 normalDepth = {0.0f, 1.0f, 0.0f, 0.0f};
    simd::float4 outgoingBiasRisk = {0.0f, 1.0f, 0.0f, 1.0f};
    simd::uint4 materialLobeMeta = {0u, 0u, 0u, 0u};
};

struct RadianceEstimate {
    simd::float4 radianceConfidence = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 varianceBiasRisk = {1.0f, 1.0f, 0.0f, 0.0f};
    simd::uint4 sampleAgeFallback = {0u, 0u, 0u, 0u};
};

struct RadianceCacheState {
    simd::float4 positionRoughness = {0.0f, 0.0f, 0.0f, 1.0f};
    simd::float4 normalVariance = {0.0f, 1.0f, 0.0f, 1.0f};
    uint32_t radianceR = 0;    // quantized accumulated RGB training radiance
    uint32_t radianceG = 0;
    uint32_t radianceB = 0;
    uint32_t sampleCount = 0;
    uint32_t lumaMoment = 0;   // quantized accumulated luma^2 proxy
    uint32_t frameTag = 0;
    uint32_t flags = 0;        // bit0 = valid, bit1 = low-confidence last query, bit2 = trained
    uint32_t domainKey = 0;
    uint32_t materialType = 0;
    uint32_t lobeType = 0;
    uint32_t confidenceQ = 0;
    uint32_t invalidationState = 0;
};

struct PathVertex {
    simd::float4 positionPdfFwd = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 normalPdfRev = {0.0f, 1.0f, 0.0f, 0.0f};
    simd::float4 throughputEta = {1.0f, 1.0f, 1.0f, 1.0f};
    simd::uint4 metadata = {0u, 0u, 0u, 0u}; // closure flags, depth, connectable, material type
};

struct EyeSubpathVertex {
    PathVertex pathVertex;
};

struct LightSubpathVertex {
    PathVertex pathVertex;
};

struct PathConnection {
    simd::uint4 vertexIndices = {0u, 0u, 0u, 0u}; // eye index, light index, class, MIS flags
    simd::float4 contributionMis = {0.0f, 0.0f, 0.0f, 0.0f}; // xyz contribution estimate, w MIS weight
    simd::float4 pdfs = {0.0f, 0.0f, 0.0f, 0.0f}; // eye fwd/rev, light fwd/rev
};

struct ConnectionResult {
    simd::float4 radianceMis = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::uint4 classification = {0u, 0u, 0u, 0u};
};

struct RestirDIReservoirState {
    simd::float4 positionPdf = {0.0f, 0.0f, 0.0f, 0.0f};          // xyz = selected light/sample position, w = pdf
    simd::float4 normalTargetPdf = {0.0f, 1.0f, 0.0f, 0.0f};      // xyz = selected normal, w = target pdf
    simd::float4 emissionWeight = {0.0f, 0.0f, 0.0f, 0.0f};       // xyz = emission, w = reservoir W
    simd::uint4 sampleMeta = {0u, 0u, 0u, 0u};                    // primitive/light id, kind, flags, frame tag
    uint32_t candidateCount = 0;
    uint32_t temporalAccepted = 0;
    uint32_t spatialAccepted = 0;
    uint32_t reserved0 = 0;
};

struct RestirDIVisibilityState {
    simd::float4 contributionVisibility = {0.0f, 0.0f, 0.0f, 0.0f}; // xyz = contribution, w = visibility
    simd::uint4 sampleMeta = {0u, 0u, 0u, 0u};
};

struct RestirDIInitUniforms {
    uint32_t pixelCount = 0;
    uint32_t frameIndex = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
};

struct WavefrontHitRecord {
    simd::float4 pointT = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 normalFrontFace = {0.0f, 1.0f, 0.0f, 1.0f};
    simd::float4 shadingNormalPrimitiveType = {0.0f, 1.0f, 0.0f, 0.0f};
    simd::uint4 ids = {0u, 0u, 0u, 0u};  // material, primitive index, mesh index, path index
    simd::float2 barycentric = {0.0f, 0.0f};
    uint32_t hit = 0;
    uint32_t padding = 0;
};

struct ShadowRay {
    simd::float4 originTMax = {0.0f, 0.0f, 0.0f, 1.0e20f};
    simd::float4 direction = {0.0f, 1.0f, 0.0f, 0.0f};
    simd::float4 contribution = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 throughput = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t pathIndex = 0;
    uint32_t lightId = 0;
    uint32_t flags = 0;
    uint32_t excludeMeshIndex = 0;
    uint32_t excludePrimitiveIndex = 0;
    uint32_t padding = 0;
};

enum class WavefrontQueueType : uint32_t {
    PrimaryRay = 1,
    DiffuseRay = 2,
    GlossyRay = 3,
    SpecularRay = 4,
    TransmissionRay = 5,
    ShadowRay = 6,
    MaterialEval = 7,
    ReservoirUpdate = 8,
    Medium = 9,
    CacheQuery = 10,
    CacheTraining = 11,
    GuidingTraining = 12,
};

struct WavefrontQueueHeader {
    PathtraceStatsAtomic activeCount;
    uint32_t capacity = 0;
    PathtraceStatsAtomic overflowCount;
    uint32_t queueType = 0;
    uint32_t frameIndex = 0;
    uint32_t debugLabel = 0;
    uint32_t producerPass = 0;
    uint32_t consumerPass = 0;
    uint32_t lastResetFrame = 0;
    PathtraceStatsAtomic peakActiveCount;
    uint32_t storagePolicy = 0;
    uint32_t reserved2 = 0;
};

struct QueueCounters {
    PathtraceStatsAtomic activeRayCount;
    PathtraceStatsAtomic nextRayCount;
    PathtraceStatsAtomic shadowRayCount;
    PathtraceStatsAtomic terminatedCount;
    PathtraceStatsAtomic overflowFlag;
    PathtraceStatsAtomic iterationIndex;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    WavefrontQueueHeader primaryRayQueue;
    WavefrontQueueHeader diffuseRayQueue;
    WavefrontQueueHeader glossyRayQueue;
    WavefrontQueueHeader specularRayQueue;
    WavefrontQueueHeader transmissionRayQueue;
    WavefrontQueueHeader shadowRayQueue;
    WavefrontQueueHeader materialEvalQueue;
    WavefrontQueueHeader reservoirUpdateQueue;
    WavefrontQueueHeader mediumQueue;
    WavefrontQueueHeader cacheQueryQueue;
    WavefrontQueueHeader cacheTrainingQueue;
    WavefrontQueueHeader guidingTrainingQueue;
};

constexpr uint32_t kPathtraceDebugMaxEntries = 512;
constexpr uint32_t kPathtraceParityMaxEntries = 16;

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
    uint32_t eventKind = 0;
    simd::float4 scalars = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 throughput = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 aux = {0.0f, 0.0f, 0.0f, 0.0f};
};

struct PathtraceParityEntry {
    uint32_t frameIndex = 0;
    uint32_t pixelX = 0;
    uint32_t pixelY = 0;
    uint32_t depth = 0;
    uint32_t probeKind = 0;
    uint32_t reasonMask = 0;
    uint32_t hwHit = 0;
    uint32_t swHit = 0;
    uint32_t hwFrontFace = 0;
    uint32_t swFrontFace = 0;
    uint32_t hwMaterialIndex = 0;
    uint32_t swMaterialIndex = 0;
    uint32_t hwMeshIndex = 0;
    uint32_t swMeshIndex = 0;
    uint32_t hwPrimitiveIndex = 0;
    uint32_t swPrimitiveIndex = 0;
    float hwT = 0.0f;
    float swT = 0.0f;
    float tMin = 0.0f;
    float tMax = 0.0f;
    simd::float2 hwBarycentric = {0.0f, 0.0f};
    simd::float2 swBarycentric = {0.0f, 0.0f};
    simd::float4 rayOrigin = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 rayDirection = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 hwNormal = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 swNormal = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 hwShadingNormal = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 swShadingNormal = {0.0f, 0.0f, 0.0f, 0.0f};
};

struct PathtraceDebugBuffer {
    PathtraceStatsAtomic writeIndex;
    uint32_t maxEntries = 0;
    uint32_t reserved0 = 0;
    uint32_t reserved1 = 0;
    PathtraceDebugEntry entries[kPathtraceDebugMaxEntries];
    PathtraceStatsAtomic parityWriteIndex;
    uint32_t parityMaxEntries = 0;
    uint32_t parityReserved0 = 0;
    uint32_t parityReserved1 = 0;
    PathtraceParityEntry parityEntries[kPathtraceParityMaxEntries];
    PathtraceStatsAtomic parityChecksPerformed;
    PathtraceStatsAtomic parityChecksInMedium;
    uint32_t parityCountsReserved0 = 0;
    uint32_t parityCountsReserved1 = 0;
};

struct MaterialTextureInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipCount = 0;
    uint32_t flags = 0;
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

struct alignas(16) LightPrimitive {
    simd::float3 centroid = {0.0f, 0.0f, 0.0f};
    float area = 0.0f;
    simd::float3 emission = {0.0f, 0.0f, 0.0f};
    float power = 0.0f;
    simd::float3 normal = {0.0f, 1.0f, 0.0f};
    uint32_t triangleIndex = 0u;
    uint32_t flags = 0u;
    float cumulativePower = 0.0f;
    uint32_t reserved1 = 0u;
    simd::float3 edge1 = {0.0f, 0.0f, 0.0f};
    float reserved2 = 0.0f;
    simd::float3 edge2 = {0.0f, 0.0f, 0.0f};
    float reserved3 = 0.0f;
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
    uint32_t bloomEnabled = 0;  // 0 = off, 1 = on
    float bloomThreshold = 1.0f;
    float bloomIntensity = 0.12f;
    float bloomRadius = 1.5f;
};

struct SvgfDenoiseUniforms {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t step = 1;
    uint32_t reprojectionEnabled = 0;
    float normalSigma = 64.0f;
    float albedoSigma = 16.0f;
    float luminanceSigma = 4.0f;
    uint32_t historyValid = 0;
    simd::float4 currentCameraOrigin = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 previousCameraOrigin = {0.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 previousLowerLeftCorner = {0.0f, 0.0f, -1.0f, 0.0f};
    simd::float4 previousHorizontal = {1.0f, 0.0f, 0.0f, 0.0f};
    simd::float4 previousVertical = {0.0f, 1.0f, 0.0f, 0.0f};
};

}  // namespace PathTracerShaderTypes
