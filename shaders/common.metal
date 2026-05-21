#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

#ifndef ENABLE_MNEE_CAUSTICS
#define ENABLE_MNEE_CAUSTICS 1
#endif
#ifndef ENABLE_MNEE
#define ENABLE_MNEE ENABLE_MNEE_CAUSTICS
#endif

constant uint kMaxSpheres = 512;
constant uint kMaxMaterials = 512;
constant uint kMaxRectangles = 128;
// Match the CPU-side cap in MetalShaderTypes.h.
constant uint kMaxMaterialTextures = 512;
constant uint kMaxMaterialSamplers = 14;
constant uint kMaterialResourcePathGuidingStateIndex = kMaxMaterialTextures + kMaxMaterialSamplers;
constant uint kMaterialResourceRestirPtStateIndex = kMaterialResourcePathGuidingStateIndex + 1u;
constant uint kMaterialResourceRadianceCacheStateIndex = kMaterialResourceRestirPtStateIndex + 1u;
constant uint kWavefrontShadowCapacityMultiplier = 4u;
constant uint kRestirPathReservoirDepthCount = 8u;
constant uint kMaterialFlagDisableOrm = 1u << 0;
constant uint kMaterialFlagForceBaseColorMip0 = 1u << 1;
constant uint kMaterialFlagBlackKeyAlphaFromRgb = 1u << 2;
constant uint kMaterialFlagCameraVisibleEmissionTint = 1u << 3;
constant uint kLightPrimitiveFlagDirectional = 1u << 0;
constant uint kMaterialClosureTypeDiffuse = 0u;
constant uint kMaterialClosureTypeConductor = 1u;
constant uint kMaterialClosureTypeDielectric = 2u;
constant uint kMaterialClosureTypeTransmission = 3u;
constant uint kMaterialClosureTypeClearcoat = 4u;
constant uint kMaterialClosureTypeSheen = 5u;
constant uint kMaterialClosureTypeCarPaint = 6u;
constant uint kMaterialClosureTypeSubsurface = 7u;
constant uint kMaterialClosureTypeVolumeBoundary = 8u;
constant uint kMaterialClosureTypeEmission = 9u;
constant uint kMaterialClosureTypePbrDiffuse = 10u;
constant uint kMaterialClosureTypePbrSpecular = 11u;
constant uint kMaterialClosureFlagDelta = 1u << 0;
constant uint kMaterialClosureFlagConnectable = 1u << 1;
constant uint kMaterialClosureFlagGuideable = 1u << 2;
constant uint kMaterialClosureFlagCacheable = 1u << 3;
constant uint kMaterialClosureFlagEmissive = 1u << 4;
constant uint kMaterialClosureFlagTransmissive = 1u << 5;
constant uint kMaterialClosureFlagVolumetric = 1u << 6;
constant uint kMaterialClosureFlagGlossy = 1u << 7;
constant uint kMaterialClosureFlagDiffuse = 1u << 8;
constant uint kMaterialClosureFlagSpecular = 1u << 9;

struct PathGuidingReservoirState {
    float4 directionConfidence;
    float4 weightMoment;
    uint sampleCount;
    uint frameTag;
    uint flags;
    uint reserved0;
};

struct RestirPtReservoirState {
    float4 positionPdf;
    float4 normalDepth;
    float4 directionWeight;
    float4 throughputLuma;
    uint materialType;
    uint lobeType;
    uint frameTag;
    uint flags;
};

struct RadianceQuery {
    float4 positionRoughness;
    float4 normalDepth;
    float4 outgoingBiasRisk;
    uint4 materialLobeMeta;
};

struct RadianceEstimate {
    float4 radianceConfidence;
    float4 varianceBiasRisk;
    uint4 sampleAgeFallback;
};

struct RadianceCacheState {
    float4 positionRoughness;
    float4 normalVariance;
    uint radianceR;
    uint radianceG;
    uint radianceB;
    uint sampleCount;
    uint lumaMoment;
    uint frameTag;
    uint flags;
    uint domainKey;
    uint materialType;
    uint lobeType;
    uint confidenceQ;
    uint invalidationState;
};

struct PathVertex {
    float4 positionPdfFwd;
    float4 normalPdfRev;
    float4 throughputEta;
    uint4 metadata;
};

struct EyeSubpathVertex {
    PathVertex pathVertex;
};

struct LightSubpathVertex {
    PathVertex pathVertex;
};

struct PathConnection {
    uint4 vertexIndices;
    float4 contributionMis;
    float4 pdfs;
};

struct ConnectionResult {
    float4 radianceMis;
    uint4 classification;
};

struct RestirDIReservoirState {
    float4 positionPdf;
    float4 normalTargetPdf;
    float4 emissionWeight;
    uint4 sampleMeta;
    uint candidateCount;
    uint temporalAccepted;
    uint spatialAccepted;
    uint reserved0;
};

struct RestirDIVisibilityState {
    float4 contributionVisibility;
    uint4 sampleMeta;
};

struct RestirDIInitUniforms {
    uint pixelCount;
    uint frameIndex;
    uint reserved0;
    uint reserved1;
};

struct MaterialTextureArgumentBuffer {
    array<texture2d<float, access::sample>, kMaxMaterialTextures> textures [[id(0)]];
    array<sampler, kMaxMaterialSamplers> samplers [[id(kMaxMaterialTextures)]];
    device PathGuidingReservoirState* pathGuidingStates [[id(kMaterialResourcePathGuidingStateIndex)]];
    device RestirPtReservoirState* restirPtReservoirs [[id(kMaterialResourceRestirPtStateIndex)]];
    device RadianceCacheState* radianceCacheStates [[id(kMaterialResourceRadianceCacheStateIndex)]];
};

struct SphereData {
    float4 centerRadius; // xyz center, w radius
    uint4 materialIndex; // x material index
};

struct RectData {
    float4 corner;          // rectangle: xyz corner; disk: xyz center
    float4 edgeU;           // rectangle: edge vector; disk: scaled tangent/radius axis
    float4 edgeV;           // rectangle: edge vector; disk: scaled bitangent/radius axis
    float4 normalAndPlane;  // xyz normal, w plane constant
    uint4 materialTwoSided; // x material index, y two-sided, z area shape
};

struct MaterialData {
    float4 baseColorRoughness;  // xyz base color/F0 tint, w roughness
    float4 typeEta;             // x type, y base IOR, z coat IOR / PBR double-sided, w thin dielectric flag / PBR thickness
    float4 emission;            // xyz emission radiance, w env-sampled flag
    float4 conductorEta;        // xyz conductor eta, w flag (>0 when valid)
    float4 conductorK;          // xyz conductor k, w flag (>0 when valid)
    float4 coatParams;          // x coat roughness, y coat thickness, z coat sample weight, w coat Fresnel average
    float4 coatTint;            // xyz coat tint, w unused
    float4 coatAbsorption;      // xyz coat absorption coefficient, w unused
    float4 dielectricSigmaA;    // xyz glass absorption coefficient, w unused
    float4 sssSigmaA;           // xyz sigma_a, w override flag
    float4 sssSigmaS;           // xyz sigma_s, w anisotropy g
    float4 sssParams;           // x mean free path, y method, z coat enabled flag, w unused
    float4 carpaintBaseParams;  // x base metallic, y base roughness, z flake scale, w flake reflectance scale
    float4 carpaintFlakeParams; // x flake sample weight, y flake roughness, z flake anisotropy, w flake normal strength
    float4 carpaintBaseEta;     // xyz carpaint base eta, w flag (>0 when valid)
    float4 carpaintBaseK;       // xyz carpaint base k, w flag (>0 when valid)
    float4 carpaintBaseTint;    // xyz conductor tint multiplier, w unused
    uint4 textureIndices0;      // x = baseColor, y = metallicRoughness, z = normal, w = occlusion
    uint4 textureIndices1;      // x = emissive, yzw unused
    uint materialFlags;         // bitfield (kMaterialFlag*)
    uint interiorMediumId;      // 0 = vacuum; front-face transmission enters this medium
    uint exteriorMediumId;      // 0 = vacuum; back-face transmission exits to this medium
    uint materialPad2;
    float4 spectralParams;      // x dispersion scale, y spectral absorption scale, z/w reserved
    float4 pbrParams;           // x = metallic, y = roughness, z = occlusion strength, w = normal scale
    float4 pbrExtras;           // x = alpha factor, y = alpha cutoff, z = transmission factor, w = alpha mode
    uint4 textureUvSet0;        // x = baseColor, y = metallicRoughness, z = normal, w = occlusion
    uint4 textureUvSet1;        // x = emissive, y = transmission, zw unused
    float4 textureTransform0;   // baseColor: row0 = (m00, m01, m02)
    float4 textureTransform1;   // baseColor: row1 = (m10, m11, m12)
    float4 textureTransform2;   // metallicRoughness: row0
    float4 textureTransform3;   // metallicRoughness: row1
    float4 textureTransform4;   // normal: row0
    float4 textureTransform5;   // normal: row1
    float4 textureTransform6;   // occlusion: row0
    float4 textureTransform7;   // occlusion: row1
    float4 textureTransform8;   // emissive: row0
    float4 textureTransform9;   // emissive: row1
    float4 textureTransform10;  // transmission: row0
    float4 textureTransform11;  // transmission: row1
};

struct MaterialClosureMetadata {
    uint type;
    uint flags;
    float weight;
    float roughness;
    float eta;
    uint lobeIndex;
    uint reserved0;
    uint reserved1;
    float4 aovAlbedoRoughness;
    float4 aovSpecularTransmission;
    float4 aovEmission;
};

struct EnvironmentAliasEntry {
    float threshold;
    uint alias;
    uint padding0;
    uint padding1;
};

struct MediumDescriptor {
    float4 sigmaA;              // xyz absorption, w sigma_t majorant
    float4 sigmaSAnisotropy;    // xyz scattering, w Henyey-Greenstein g
    float4 emissionPhase;       // xyz emission, w phase function id
    uint4 metadata;             // x medium id, y max scattering events, z NEE enabled, w flags
};

struct MediumState {
    uint mediumId;
    uint scatteringDepth;
    uint phaseFunction;
    uint flags;
    float4 transmittance;
};

struct PathtraceUniforms {
    uint width;
    uint height;
    uint frameIndex;
    uint sampleCount;
    uint executionMode;
    uint wavefrontSchedulingPolicy;
    uint wavefrontCompactionEnabled;
    uint wavefrontDebugStageFailures;

    float3 cameraOrigin;
    float cameraPad0;
    float3 lowerLeftCorner;
    float cameraPad1;
    float3 horizontal;
    float cameraPad2;
    float3 vertical;
    float cameraPad3;
    float3 cameraU;
    float lensRadius;
    float3 cameraV;
    float cameraPad4;

    uint sphereCount;
    uint rectangleCount;
    uint materialCount;
    uint materialTextureCount;
    uint maxDepth;
    uint useRussianRoulette;
    uint intersectionMode;
    uint softwareBvhType;
    uint primitiveCount;
    uint meshCount;
    uint triangleCount;
    uint emissivePrimitiveCount;
    uint fixedRngSeed;  // If non-zero, use as deterministic RNG seed
    uint backgroundMode;
    uint workingColorSpace;
    float environmentRotation;
    float environmentIntensity;
    float emissiveTotalPower;
    float3 backgroundColor;
    uint environmentAliasCount;
    uint environmentMapWidth;
    uint environmentMapHeight;
    uint environmentHasDistribution;
    uint fireflyClampEnabled;
    uint fireflyClampMode;
    float fireflyClampFactor;
    float fireflyClampFloor;
    float throughputClamp;
    float specularTailClampBase;
    float specularTailClampRoughnessScale;
    float minSpecularPdf;
    float fireflyClampMaxContribution;
    float paddingFirefly;
    uint sssMode;
    uint sssMaxSteps;
    uint directLightMode;
    uint enableSpecularNee;
    uint enableMnee;
    uint enableMneeSecondary;
    uint risCandidateCount;
    uint spatialReuseNeighborCount;
    float worldReuseCellSize;
    uint restirGiMode;
    uint pathGuidingMode;
    float pathGuidingStrength;
    float pathGuidingCellSize;
    uint pathGuidingTrainingInterval;
    uint restirPtMode;
    uint restirPtMaxReservoirs;
    uint restirPtDebug;
    float restirPtReuseStrength;
    uint radianceCacheMode;
    float radianceCacheCellSize;
    uint radianceCacheMinDepth;
    float radianceCacheMinConfidence;
    uint radianceCacheTrainingInterval;
    float radianceCacheMaxRadiance;
    uint radianceCacheDebug;
    uint radianceCachePadding0;
    uint causticTransportMode;
    uint causticMaxVertices;
    uint volumeTransportMode;
    uint volumePhaseFunction;
    float4 volumeSigmaA;
    float4 volumeSigmaSAnisotropy;
    float4 volumeEmissionMaxEvents;
    uint volumeNeeEnabled;
    uint volumeDebug;
    uint spectralRenderingMode;
    uint spectralTexturePolicy;
    float4 spectralWavelengthRange;
    uint spectralDebug;
    uint spectralPadding0;
    uint spectralPadding1;
    uint spectralPadding2;
    uint restirDebugEnabled;
    uint restirDebugView;
    uint restirDebugCounters;
    uint restirDebugFrameIndex;
    uint debugPathActive;
    uint debugPixelX;
    uint debugPixelY;
    uint debugMaxEntries;
    uint hardwareExcludeMaxAttempts;
    float hardwareExitNormalBias;
    float hardwareExitDirectionalBias;
    uint enableHardwareMissFallback;
    uint enableHardwareFirstHitFromSoftware;
    uint enableHardwareForceSoftware;
    uint paddingHardware1;
    uint parityAssertEnabled;
    uint parityAssertMode;
    uint parityPixelX;
    uint parityPixelY;
    uint parityOncePerFrame;
    uint parityPadding0;
    uint explicitRestirDiLightingStage;
    uint parityPadding2;
    uint forcePureHWRTForGlass;
    uint enableHardwareShadowNearRetryReject;
    uint parityPadding4;
    uint parityPadding5;
    uint debugViewMode;
    uint debugDisableAO;
    uint debugAoIndirectOnly;
    uint debugDisableNormalMap;
    uint debugDisableOrmTexture;
    uint debugFlipNormalGreen;
    uint debugSpecularOnly;
    uint debugDirectLightAudit;
    float debugNormalStrengthScale;
    float debugNormalLodBias;
    float debugOrmLodBias;
    float debugEnvMipOverride;
    uint debugEnableVisorOverride;
    int debugVisorOverrideMaterialId;
    float debugVisorOverrideRoughness;
    float debugVisorOverrideF0;
    uint debugEnvNearest;
    uint debugPadding0;
};

struct DisplayUniforms {
    uint tonemapMode;
    uint acesVariant;
    float exposure;
    float reinhardWhite;
    uint bloomEnabled;
    float bloomThreshold;
    float bloomIntensity;
    float bloomRadius;
};

struct SvgfDenoiseUniforms {
    uint width;
    uint height;
    uint step;
    uint reprojectionEnabled;
    float normalSigma;
    float albedoSigma;
    float luminanceSigma;
    uint historyValid;
    float4 currentCameraOrigin;
    float4 previousCameraOrigin;
    float4 previousLowerLeftCorner;
    float4 previousHorizontal;
    float4 previousVertical;
};

struct DisplayVertexOut {
    float4 position [[position]];
    float2 uv;
};

struct Ray {
    float3 origin;
    float3 direction;
};

struct BvhNode {
    float4 boundsMin;
    float4 boundsMax;
    uint leftChild;
    uint rightChild;
    uint primitiveOffset;
    uint primitiveCount;
    uint padding0;
    uint padding1;
};

constant uint kIntersectionModeSoftwareBVH = 0u;
constant uint kIntersectionModeHardwareRT = 1u;
constant uint kRenderExecutionMegakernel = 0u;
constant uint kRenderExecutionWavefront = 1u;

constant uint kSoftwareBvhNone = 0u;
constant uint kSoftwareBvhSpheres = 1u;
constant uint kSoftwareBvhTriangles = 2u;
constant uint kPathtraceDebugMaxEntries = 512u;
constant uint kPathtraceParityMaxEntries = 16u;

struct PathtraceStats {
    atomic_uint primaryRayCount;
    atomic_uint nodesVisited;
    atomic_uint leafPrimTests;
    atomic_uint internalNodeVisits;
    atomic_uint internalBothVisited;
    atomic_uint shadowRayCount;
    atomic_uint shadowRayEarlyExitCount;
    atomic_uint hardwareRayCount;
    atomic_uint hardwareHitCount;
    atomic_uint hardwareMissCount;
    // Extended diagnostics mirroring CPU-side PathtraceStats
    atomic_uint hardwareResultNoneCount;
    atomic_uint hardwareRejectedCount;
    atomic_uint hardwareUnavailableCount;
    atomic_uint hardwareLastResultType;
    atomic_uint hardwareLastInstanceId;
    atomic_uint hardwareLastPrimitiveId;
    atomic_uint hardwareLastDistanceBits;
    atomic_uint specularNeeOcclusionHitCount;
    atomic_uint specNeeEnvAddedCount;
    atomic_uint specNeeRectAddedCount;
    atomic_uint mneeEnvHwOccludedCount;
    atomic_uint mneeEnvSwOccludedCount;
    atomic_uint mneeEnvHwSwMismatchCount;
    atomic_uint mneeRectHwOccludedCount;
    atomic_uint mneeRectSwOccludedCount;
    atomic_uint mneeRectHwSwMismatchCount;
    atomic_uint mneeHitHwSwHitMissCount;
    atomic_uint mneeHitHwSwNormalMismatchCount;
    atomic_uint mneeHitHwSwIdMismatchCount;
    atomic_uint mneeHitHwSwTDiffCount;
    atomic_uint mneeChainHwSwHitMissCount;
    atomic_uint mneeChainHwSwNormalMismatchCount;
    atomic_uint mneeChainHwSwIdMismatchCount;
    atomic_uint mneeChainHwSwTDiffCount;
    atomic_uint mneeEligibleCount;
    atomic_uint mneeEnvAttemptCount;
    atomic_uint mneeEnvAddedCount;
    atomic_uint mneeEnvChainUnsupportedCount;
    atomic_uint mneeEnvChainNonDeltaBlockCount;
    atomic_uint mneeEnvChainSampleRejectCount;
    atomic_uint mneeEnvChainWeightRejectCount;
    atomic_uint mneeEnvChainMaxBounceCount;
    atomic_uint mneeRectAttemptCount;
    atomic_uint mneeRectAddedCount;
    atomic_uint mneeContributionCount;
    atomic_uint mneeContributionLumaSumLo;
    atomic_uint mneeContributionLumaSumHi;
    atomic_uint hardwareSelfHitRejectedCount;
    atomic_uint hardwareMissDistanceBins[32];
    atomic_uint hardwareMissLastDistanceBits;
    atomic_uint hardwareSelfHitLastDistanceBits;
    atomic_uint hardwareExcludeRetryHistogram[4];
    atomic_uint hardwareMissLastInstanceId;
    atomic_uint hardwareMissLastPrimitiveId;
    atomic_uint hardwareFallbackHitCount;
    atomic_uint hardwareFirstHitFallbackCount;
    atomic_uint throughputNanInfCount;
    atomic_uint radianceNanInfCount;
    atomic_uint clampEventCount;
    atomic_uint restirGiCandidateCount;
    atomic_uint restirGiReservoirUpdateCount;
    atomic_uint restirGiAcceptCount;
    atomic_uint restirGiRejectCount;
    atomic_uint restirGiFallbackCount;
    atomic_uint pathGuidingCandidateCount;
    atomic_uint pathGuidingUsedCount;
    atomic_uint pathGuidingFallbackCount;
    atomic_uint pathGuidingInvalidCount;
    atomic_uint restirPtCandidateCount;
    atomic_uint restirPtReservoirUpdateCount;
    atomic_uint restirPtRejectedInvalidCount;
    atomic_uint restirPtRejectedUnsupportedCount;
    atomic_uint restirPtDebugRecordCount;
    atomic_uint restirPtReuseCandidateCount;
    atomic_uint restirPtReuseAppliedCount;
    atomic_uint restirPtReuseRejectedInvalidCount;
    atomic_uint restirPtReuseFallbackCount;
    atomic_uint radianceCacheQueryCount;
    atomic_uint radianceCacheHitCount;
    atomic_uint radianceCacheMissCount;
    atomic_uint radianceCacheTrainCount;
    atomic_uint radianceCacheFallbackCount;
    atomic_uint radianceCacheInvalidCount;
    atomic_uint causticEyeVertexCount;
    atomic_uint causticLightVertexCount;
    atomic_uint causticConnectionCandidateCount;
    atomic_uint causticDeltaChainCount;
    atomic_uint causticDebugClassificationCount;
    atomic_uint volumeTransmittanceQueryCount;
    atomic_uint volumeScatterEventCount;
    atomic_uint volumeNeeRayCount;
    atomic_uint volumePhaseSampleCount;
    atomic_uint volumeMediumBoundaryEventCount;
    atomic_uint volumeFallbackCount;
    atomic_uint spectralWavelengthSampleCount;
    atomic_uint spectralMaterialLookupCount;
    atomic_uint spectralDispersionEventCount;
    atomic_uint spectralFallbackCount;
    atomic_uint hardwareShadowTMinRetryCount;
    atomic_uint hardwareShadowNearForeignHitCount;
    atomic_uint wavefrontActiveRayCount;
    atomic_uint wavefrontNextRayCount;
    atomic_uint wavefrontShadowRayCount;
    atomic_uint wavefrontTerminatedCount;
    atomic_uint wavefrontOverflowFlag;
    atomic_uint wavefrontTerminateMissCount;
    atomic_uint wavefrontTerminateEmissiveCount;
    atomic_uint wavefrontTerminateMaxDepthCount;
    atomic_uint wavefrontTerminateOverflowCount;
    atomic_uint wavefrontEnvShadowRayCount;
    atomic_uint wavefrontAovAlbedoValidCount;
    atomic_uint wavefrontAovNormalValidCount;
    atomic_uint wavefrontPrimaryGeneratedCount;
    atomic_uint wavefrontIntersectProcessedCount;
    atomic_uint wavefrontShadeProcessedCount;
    atomic_uint wavefrontShadowProcessedCount;
    atomic_uint wavefrontNextRayEnqueueCount;
    atomic_uint wavefrontShadowRayEnqueueCount;
};

struct WavefrontRay {
    float4 originMinT;
    float4 directionMaxT;
    float4 rayDiffDOdx;
    float4 rayDiffDOdy;
    float4 rayDiffDDdx;
    float4 rayDiffDDdy;
    float2 cone;
    uint excludeMeshIndex;
    uint excludePrimitiveIndex;
    uint pixelIndex;
    uint pathIndex;
    uint depth;
    uint flags;
};

constant uint kWavefrontMediumStackSize = 8u;

// Wavefront path state layout: AoS, one PathState per path index.
// Large float4 fields stay together today to preserve simple indexing across
// megakernel parity checks and future typed-queue compaction checkpoints.
struct PathState {
    float4 throughput;
    float4 radiance;
    float4 firstHitAlbedo;
    float4 firstHitNormal;
    float4 firstHitPosition;
    float4 firstHitMaterial;
    float envLod;
    float lastBsdfPdf;
    uint rngState;
    uint mediumDepth;
    uint alive;
    uint flags;
    uint reserved0;
    uint reserved1;
    uint currentMediumId;
    uint volumeScatteringDepth;
    float spectralWavelengthNm;
    uint spectralSampleIndex;
    uint spectralPadding0;
    uint spectralPadding1;
    float4 mediumStack[kWavefrontMediumStackSize];
};

struct WavefrontHitRecord {
    float4 pointT;
    float4 normalFrontFace;
    float4 shadingNormalPrimitiveType;
    uint4 ids;
    float2 barycentric;
    uint hit;
    uint padding;
};

struct ShadowRay {
    float4 originTMax;
    float4 direction;
    float4 contribution;
    float4 throughput;
    uint pathIndex;
    uint lightId;
    uint flags;
    uint excludeMeshIndex;
    uint excludePrimitiveIndex;
    uint padding;
};

constant uint kWavefrontQueuePrimaryRay = 1u;
constant uint kWavefrontQueueDiffuseRay = 2u;
constant uint kWavefrontQueueGlossyRay = 3u;
constant uint kWavefrontQueueSpecularRay = 4u;
constant uint kWavefrontQueueTransmissionRay = 5u;
constant uint kWavefrontQueueShadowRay = 6u;
constant uint kWavefrontQueueMaterialEval = 7u;
constant uint kWavefrontQueueReservoirUpdate = 8u;
constant uint kWavefrontQueueMedium = 9u;
constant uint kWavefrontQueueCacheQuery = 10u;
constant uint kWavefrontQueueCacheTraining = 11u;
constant uint kWavefrontQueueGuidingTraining = 12u;
constant uint kWavefrontQueueStoragePhysical = 1u;
constant uint kWavefrontQueueStorageMetadataOnly = 2u;

struct WavefrontQueueHeader {
    atomic_uint activeCount;
    uint capacity;
    atomic_uint overflowCount;
    uint queueType;
    uint frameIndex;
    uint debugLabel;
    uint producerPass;
    uint consumerPass;
    uint lastResetFrame;
    atomic_uint peakActiveCount;
    uint storagePolicy;
    uint reserved2;
};

struct QueueCounters {
    atomic_uint activeRayCount;
    atomic_uint nextRayCount;
    atomic_uint shadowRayCount;
    atomic_uint terminatedCount;
    atomic_uint overflowFlag;
    atomic_uint iterationIndex;
    uint reserved0;
    uint reserved1;
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

struct PathtraceDebugEntry {
    uint integrator;
    uint pixelX;
    uint pixelY;
    uint sampleIndex;
    uint depth;
    uint mediumDepthBefore;
    uint mediumDepthAfter;
    int mediumEvent;
    uint frontFace;
    uint scatterIsDelta;
    uint materialIndex;
    uint eventKind;
    float4 scalars;
    float4 throughput;
    float4 aux;
};

struct PathtraceParityEntry {
    uint frameIndex;
    uint pixelX;
    uint pixelY;
    uint depth;
    uint probeKind;
    uint reasonMask;
    uint hwHit;
    uint swHit;
    uint hwFrontFace;
    uint swFrontFace;
    uint hwMaterialIndex;
    uint swMaterialIndex;
    uint hwMeshIndex;
    uint swMeshIndex;
    uint hwPrimitiveIndex;
    uint swPrimitiveIndex;
    float hwT;
    float swT;
    float tMin;
    float tMax;
    float2 hwBarycentric;
    float2 swBarycentric;
    float4 rayOrigin;
    float4 rayDirection;
    float4 hwNormal;
    float4 swNormal;
    float4 hwShadingNormal;
    float4 swShadingNormal;
};

struct PathtraceDebugBuffer {
    atomic_uint writeIndex;
    uint maxEntries;
    uint reserved0;
    uint reserved1;
    PathtraceDebugEntry entries[kPathtraceDebugMaxEntries];
    atomic_uint parityWriteIndex;
    uint parityMaxEntries;
    uint parityReserved0;
    uint parityReserved1;
    PathtraceParityEntry parityEntries[kPathtraceParityMaxEntries];
    atomic_uint parityChecksPerformed;
    atomic_uint parityChecksInMedium;
    uint parityCountsReserved0;
    uint parityCountsReserved1;
};

struct MaterialTextureInfo {
    uint width;
    uint height;
    uint mipCount;
    uint flags;
};

struct RayHit {
    float3 normal;
    float t;
    uint primitiveIndex;
    uint materialIndex;
    uint frontFace;
    uint hit;
    uint padding0;
    uint padding1;
};

constant uint kPrimitiveTypeNone = 0u;
constant uint kPrimitiveTypeSphere = 1u;
constant uint kPrimitiveTypeRectangle = 2u;
constant uint kPrimitiveTypeTriangle = 3u;

struct MeshInfo {
    uint materialIndex;
    uint triangleOffset;
    uint triangleCount;
    uint vertexOffset;
    uint vertexCount;
    uint indexOffset;
    uint indexCount;
    float4x4 localToWorld;
    float4x4 worldToLocal;
};

struct SceneVertex {
    float4 position;
    float4 normal;
    float4 tangent;
    float4 uv;
};

struct TriangleData {
    float4 v0;
    float4 v1;
    float4 v2;
    uint4 metadata; // x material index
};

struct LightPrimitive {
    float3 centroid;
    float area;
    float3 emission;
    float power;
    float3 normal;
    uint triangleIndex;
    uint flags;
    float cumulativePower;
    uint reserved1;
    float3 edge1;
    float reserved2;
    float3 edge2;
    float reserved3;
};

struct SoftwareInstanceInfo {
    uint blasRootNodeOffset;
    uint blasNodeCount;
    uint blasPrimIndexOffset;
    uint blasPrimIndexCount;
    uint triangleBaseOffset;
    uint triangleCount;
    uint padding0;
    uint padding1;
    float4x4 localToWorld;
    float4x4 worldToLocal;
};

inline float3 ray_at(const Ray ray, float t) {
    return ray.origin + t * ray.direction;
}
