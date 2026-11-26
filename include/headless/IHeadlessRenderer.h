#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <simd/simd.h>

#include "renderer/AccumulationState.h"
#include "renderer/RadianceCacheTraining.h"
#include "renderer/RenderSettings.h"
#include "renderer/SceneResources.h"

enum class HeadlessBackend {
    Metal = 0,
    Embree = 1,
};

struct HeadlessScene {
    std::string source;
    bool isPath = false;
    PathTracer::SceneResources* resources = nullptr;
};

struct HeadlessCamera {
    simd::float3 target{0.0f, 0.0f, 0.0f};
    float distance = 0.0f;
    float yaw = 0.0f;
    float pitch = 0.0f;
    float verticalFov = 0.0f;
    float defocusAngle = 0.0f;
    float focusDistance = 0.0f;
};

struct HeadlessRenderOutput {
    static constexpr uint32_t kWavefrontQueueMetricCount = 12;

    struct DirectLightShadowPathAudit {
        bool hwRawResultAvailable = false;
        bool hwTraceUsed = false;
        uint32_t hwRawResultType = 0xFFFFFFFFu;
        uint32_t hwRawInstanceIndex = 0xFFFFFFFFu;
        uint32_t hwRawPrimitiveIndex = 0xFFFFFFFFu;
        float hwRawDistance = 0.0f;
        bool hwResolvedHit = false;
        uint32_t hwResolvedMeshIndex = 0xFFFFFFFFu;
        uint32_t hwResolvedMaterialIndex = 0xFFFFFFFFu;
        uint32_t hwResolvedPrimitiveIndex = 0xFFFFFFFFu;
        uint32_t hwResolvedPrimitiveType = 0xFFFFFFFFu;
        bool hwResolvedFrontFace = false;
        float hwResolvedDistance = 0.0f;
        bool embeddedSwConsulted = false;
        bool embeddedSwHit = false;
        uint32_t embeddedSwInstanceIndex = 0xFFFFFFFFu;
        uint32_t embeddedSwMeshIndex = 0xFFFFFFFFu;
        uint32_t embeddedSwMaterialIndex = 0xFFFFFFFFu;
        uint32_t embeddedSwPrimitiveIndex = 0xFFFFFFFFu;
        uint32_t embeddedSwPrimitiveType = 0xFFFFFFFFu;
        bool embeddedSwFrontFace = false;
        float embeddedSwDistance = 0.0f;
        simd::float3 embeddedSwHitPosition = {0.0f, 0.0f, 0.0f};
        uint32_t softwareBvhType = 0xFFFFFFFFu;
        bool softwareTlasAvailable = false;
        bool softwareBlasAvailable = false;
        bool softwareInstanceInfoAvailable = false;
        uint32_t swBlasRootNodeOffset = 0u;
        uint32_t swBlasPrimIndexOffset = 0u;
        uint32_t swTriangleBaseOffset = 0u;
        float shadowTMin = 0.0f;
        float shadowTMax = 0.0f;
        float rayOriginEpsilon = 0.0f;
        float hardwareOcclusionEpsilon = 0.0f;
        uint32_t rejectionMask = 0u;
        bool lightTwoSided = false;
        bool finalVisible = false;
    };

    struct DirectLightAuditEntry {
        uint32_t lightIndex = 0;
        uint32_t rectIndex = 0;
        simd::float3 lightDirection = {0.0f, 0.0f, 0.0f};
        float distance = 0.0f;
        float nDotL = 0.0f;
        bool visible = false;
        simd::float3 bsdfValue = {0.0f, 0.0f, 0.0f};
        float lightPdf = 0.0f;
        float bsdfPdf = 0.0f;
        float misWeight = 0.0f;
        simd::float3 contribution = {0.0f, 0.0f, 0.0f};
        uint32_t occluderMaterialIndex = 0xFFFFFFFFu;
        uint32_t occluderPrimitiveIndex = 0xFFFFFFFFu;
        uint32_t occluderPrimitiveType = 0xFFFFFFFFu;
        bool occluderFrontFace = false;
        float occluderDistance = 0.0f;
        DirectLightShadowPathAudit shadowPath;
    };

    struct DirectLightBsdfAuditEntry {
        uint32_t depth = 0;
        uint32_t materialIndex = 0;
        bool frontFace = false;
        bool isDelta = false;
        uint32_t closureType = 0;
        bool usedRandomWalk = false;
        uint32_t rngBefore = 0;
        uint32_t rngAfter = 0;
        simd::float3 shadingNormal = {0.0f, 0.0f, 0.0f};
        simd::float3 incidentDirection = {0.0f, 0.0f, 0.0f};
        simd::float3 wo = {0.0f, 0.0f, 0.0f};
        simd::float3 sampledDirection = {0.0f, 0.0f, 0.0f};
        float pdf = 0.0f;
        float directionalPdf = 0.0f;
        uint32_t lobeType = 0;
    };

    struct RisAuditState {
        bool available = false;
        bool valid = false;
        bool visible = false;
        uint32_t winnerPrimitiveIndex = 0xFFFFFFFFu;
        uint32_t M = 0u;
        simd::float2 winnerBary = {0.0f, 0.0f};
        simd::float3 winnerPosition = {0.0f, 0.0f, 0.0f};
        float winnerPdf = 0.0f;
        simd::float3 winnerNormal = {0.0f, 0.0f, 0.0f};
        float winnerDistance = 0.0f;
        simd::float3 winnerEmission = {0.0f, 0.0f, 0.0f};
        float winnerNDotL = 0.0f;
        simd::float3 winnerBsdfValue = {0.0f, 0.0f, 0.0f};
        float winnerBsdfPdf = 0.0f;
        simd::float3 winnerContribution = {0.0f, 0.0f, 0.0f};
        float wSum = 0.0f;
        float W = 0.0f;
        uint32_t rngBefore = 0u;
        uint32_t rngAfter = 0u;
        uint32_t rngTemporalBefore = 0u;
        uint32_t rngTemporalAfter = 0u;
        bool spatialReuseAttempted = false;
        uint32_t spatialNeighborTarget = 0u;
        uint32_t spatialNeighborsConsidered = 0u;
        uint32_t spatialNeighborsAccepted = 0u;
        uint32_t spatialRejectedDepth = 0u;
        uint32_t spatialRejectedNormal = 0u;
        uint32_t spatialRejectedInvalid = 0u;
        int32_t spatialWinnerOffsetX = 0;
        int32_t spatialWinnerOffsetY = 0;
        float spatialLastMisWeight = 0.0f;
        float spatialLastMergeWeight = 0.0f;
        bool temporalReuseAttempted = false;
        bool temporalPreviousAvailable = false;
        bool temporalReuseAccepted = false;
        uint32_t temporalRejectedDepth = 0u;
        uint32_t temporalRejectedNormal = 0u;
        uint32_t temporalRejectedInvalid = 0u;
        uint32_t temporalPreviousPrimitiveIndex = 0xFFFFFFFFu;
        float temporalLastMisWeight = 0.0f;
        float temporalLastMergeWeight = 0.0f;
        bool worldReuseAttempted = false;
        bool worldReuseAccepted = false;
        uint32_t worldCellHash = 0u;
        int32_t worldCellX = 0;
        int32_t worldCellY = 0;
        int32_t worldCellZ = 0;
        uint32_t worldCandidatesConsidered = 0u;
        uint32_t worldCandidatesAccepted = 0u;
        uint32_t worldCandidateTarget = 0u;
        uint32_t worldQueryRadius = 0u;
        uint32_t worldRejectedDepth = 0u;
        uint32_t worldRejectedNormal = 0u;
        uint32_t worldRejectedInvalid = 0u;
        uint32_t worldRejectedCell = 0u;
        uint32_t worldCandidatePrimitiveIndex = 0xFFFFFFFFu;
        float worldLastMisWeight = 0.0f;
        float worldLastMergeWeight = 0.0f;
        bool cacheReuseAttempted = false;
        bool cacheStateAvailable = false;
        bool cacheReuseAccepted = false;
        bool cacheFallbackUsed = false;
        uint32_t cacheCellHash = 0u;
        int32_t cacheCellX = 0;
        int32_t cacheCellY = 0;
        int32_t cacheCellZ = 0;
        uint32_t cacheCandidatesConsidered = 0u;
        uint32_t cacheEntriesAvailable = 0u;
        uint32_t cacheCandidatesAccepted = 0u;
        uint32_t cacheCandidateTarget = 0u;
        uint32_t cacheRejectedDepth = 0u;
        uint32_t cacheRejectedNormal = 0u;
        uint32_t cacheRejectedInvalid = 0u;
        uint32_t cacheRejectedCell = 0u;
        uint32_t cacheCandidatePrimitiveIndex = 0xFFFFFFFFu;
        uint32_t cacheSourceFrameIndex = 0u;
        float cacheLastMisWeight = 0.0f;
        float cacheLastMergeWeight = 0.0f;
    };

    struct RisCandidateAuditEntry {
        uint32_t candidateIndex = 0u;
        uint32_t primitiveIndex = 0xFFFFFFFFu;
        uint32_t rngBefore = 0u;
        uint32_t rngAfter = 0u;
        uint32_t reservoirM = 0u;
        bool valid = false;
        float pdf = 0.0f;
        float phat = 0.0f;
        float weight = 0.0f;
        float reservoirWSum = 0.0f;
    };

    struct DirectLightAuditDump {
        bool available = false;
        uint32_t pixelX = 0;
        uint32_t pixelY = 0;
        uint32_t sampleIndex = 0;
        uint32_t depth = 0;
        uint32_t materialIndex = 0;
        uint32_t materialType = 0;
        uint32_t primitiveIndex = 0;
        uint32_t rawMaterialType = 0;
        simd::float3 hitPosition = {0.0f, 0.0f, 0.0f};
        simd::float3 shadingNormal = {0.0f, 0.0f, 0.0f};
        simd::float3 wo = {0.0f, 0.0f, 0.0f};
        simd::float2 uv0 = {0.0f, 0.0f};
        simd::float3 baseColor = {0.0f, 0.0f, 0.0f};
        float roughness = 0.0f;
        float baseColorLod = 0.0f;
        uint32_t baseColorTextureIndex = 0xFFFFFFFFu;
        simd::float4 pbrParams = {0.0f, 0.0f, 0.0f, 0.0f};
        simd::float4 pbrExtras = {0.0f, 0.0f, 0.0f, 0.0f};
        simd::float4 typeEta = {0.0f, 0.0f, 0.0f, 0.0f};
        float diffuseOcclusion = 1.0f;
        RisAuditState ris;
        std::vector<RisCandidateAuditEntry> risCandidates;
        std::vector<DirectLightBsdfAuditEntry> bsdfPath;
        std::vector<DirectLightAuditEntry> entries;
    };

    struct MaterialMrDebugDump {
        bool available = false;
        uint32_t pixelX = 0;
        uint32_t pixelY = 0;
        uint32_t sampleIndex = 0;
        uint32_t depth = 0;
        uint32_t materialIndex = 0;
        uint32_t primitiveIndex = 0;
        simd::float2 uv0 = {0.0f, 0.0f};
        simd::float2 uv1 = {0.0f, 0.0f};
        simd::float3 baseColor = {0.0f, 0.0f, 0.0f};
        float metallic = 0.0f;
        float roughness = 0.0f;
        float ao = 1.0f;
        simd::float3 emissive = {0.0f, 0.0f, 0.0f};
        simd::float3 geometricNormal = {0.0f, 0.0f, 0.0f};
        simd::float3 shadingNormal = {0.0f, 0.0f, 0.0f};
        simd::float3 tangent = {0.0f, 0.0f, 0.0f};
        simd::float3 bitangent = {0.0f, 0.0f, 0.0f};
        simd::float3 normalSampleRaw = {0.0f, 0.0f, 0.0f};
        simd::float3 normalSampleTs = {0.0f, 0.0f, 0.0f};
        simd::float3 perturbedNormal = {0.0f, 0.0f, 0.0f};
        simd::float3 f0 = {0.0f, 0.0f, 0.0f};
        float diffuseWeight = 0.0f;
        float specularWeight = 0.0f;
        float normalScale = 0.0f;
        float transmission = 0.0f;
        float alpha = 1.0f;
        float alphaCutoff = 0.0f;
        float alphaMode = 0.0f;
        float roughnessInput = 0.0f;
        float roughnessBeforeNormalVariance = 0.0f;
        float normalLod = 0.0f;
        float ndotv = 0.0f;
        simd::float3 sampledOutgoingDirection = {0.0f, 0.0f, 0.0f};
        simd::float3 brdfSpecularValue = {0.0f, 0.0f, 0.0f};
        float specularPdf = 0.0f;
        float dTerm = 0.0f;
        float gTerm = 0.0f;
        simd::float3 fTerm = {0.0f, 0.0f, 0.0f};
        simd::float3 finalSpecularContribution = {0.0f, 0.0f, 0.0f};
        simd::float3 directContribution = {0.0f, 0.0f, 0.0f};
        float directContributionKind = 0.0f; // 0 none, 1 env, 2 rect
    };

    struct PbrMetrics {
        static constexpr uint32_t kRestirPtEnhancedHistogramBins = 8;

        bool available = false;
        uint64_t throughputNanInfCount = 0;
        uint64_t radianceNanInfCount = 0;
        uint64_t clampEventCount = 0;
        uint64_t specNeeEnvAddedCount = 0;
        uint64_t specNeeRectAddedCount = 0;
        uint64_t mneeEligibleCount = 0;
        uint64_t mneeEnvAttemptCount = 0;
        uint64_t mneeEnvAddedCount = 0;
        uint64_t mneeEnvChainUnsupportedCount = 0;
        uint64_t mneeEnvChainNonDeltaBlockCount = 0;
        uint64_t mneeEnvChainSampleRejectCount = 0;
        uint64_t mneeEnvChainWeightRejectCount = 0;
        uint64_t mneeEnvChainMaxBounceCount = 0;
        uint64_t mneeRectAddedCount = 0;
        uint64_t mneeContributionCount = 0;
        uint64_t restirDiRegirCandidateCount = 0;
        uint64_t restirDiRegirAcceptedCount = 0;
        uint64_t restirDiRegirFallbackCount = 0;
        uint64_t restirGiCandidateCount = 0;
        uint64_t restirGiReservoirUpdateCount = 0;
        uint64_t restirGiAcceptCount = 0;
        uint64_t restirGiRejectCount = 0;
        uint64_t restirGiFallbackCount = 0;
        uint64_t pathGuidingCandidateCount = 0;
        uint64_t pathGuidingUsedCount = 0;
        uint64_t pathGuidingFallbackCount = 0;
        uint64_t pathGuidingInvalidCount = 0;
        uint64_t restirPtCandidateCount = 0;
        uint64_t restirPtReservoirUpdateCount = 0;
        uint64_t restirPtRejectedInvalidCount = 0;
        uint64_t restirPtRejectedUnsupportedCount = 0;
        uint64_t restirPtDebugRecordCount = 0;
        uint64_t restirPtReuseCandidateCount = 0;
        uint64_t restirPtReuseAppliedCount = 0;
        uint64_t restirPtReuseRejectedInvalidCount = 0;
        uint64_t restirPtReuseFallbackCount = 0;
        uint64_t restirPtEnhancedCandidateCount = 0;
        uint64_t restirPtEnhancedInitialSelectedCount = 0;
        uint64_t restirPtEnhancedTemporalAcceptedCount = 0;
        uint64_t restirPtEnhancedTemporalRejectedCount = 0;
        uint64_t restirPtEnhancedSpatialAcceptedCount = 0;
        uint64_t restirPtEnhancedSpatialRejectedCount = 0;
        uint64_t restirPtEnhancedInvalidShiftGeometryCount = 0;
        uint64_t restirPtEnhancedInvalidShiftMaterialCount = 0;
        uint64_t restirPtEnhancedInvalidShiftPdfCount = 0;
        uint64_t restirPtEnhancedInvalidShiftVisibilityCount = 0;
        uint64_t restirPtEnhancedInvalidShiftInvertibilityCount = 0;
        uint64_t restirPtEnhancedInvalidShiftUnsupportedCount = 0;
        uint64_t restirPtEnhancedDuplicateScoreSum = 0;
        uint64_t restirPtEnhancedBiasedCCapApplicationCount = 0;
        uint64_t restirPtEnhancedVectorWeightUseCount = 0;
        uint64_t restirPtEnhancedReplayCompactionInputCount = 0;
        uint64_t restirPtEnhancedReplayCompactionOutputCount = 0;
        uint64_t restirPtEnhancedFallbackCount = 0;
        uint64_t restirPtEnhancedCanonicalSampleCount = 0;
        uint64_t restirPtEnhancedPathLengthHistogram[kRestirPtEnhancedHistogramBins] = {};
        uint64_t restirPtEnhancedReconnectionDepthHistogram[kRestirPtEnhancedHistogramBins] = {};
        uint64_t radianceCacheQueryCount = 0;
        uint64_t radianceCacheHitCount = 0;
        uint64_t radianceCacheMissCount = 0;
        uint64_t radianceCacheTrainCount = 0;
        uint64_t radianceCacheFallbackCount = 0;
        uint64_t radianceCacheInvalidCount = 0;
        // Whole-render radiance cache counters, RadianceCacheTraining::RadianceCacheStat layout.
        uint64_t radianceCacheStats[PathTracer::RadianceCacheTraining::kStatCount] = {};
        uint64_t causticEyeVertexCount = 0;
        uint64_t causticLightVertexCount = 0;
        uint64_t causticConnectionCandidateCount = 0;
        uint64_t causticDeltaChainCount = 0;
        uint64_t causticDebugClassificationCount = 0;
        uint64_t volumeTransmittanceQueryCount = 0;
        uint64_t volumeScatterEventCount = 0;
        uint64_t volumeNeeRayCount = 0;
        uint64_t volumePhaseSampleCount = 0;
        uint64_t volumeMediumBoundaryEventCount = 0;
        uint64_t volumeFallbackCount = 0;
        uint64_t spectralWavelengthSampleCount = 0;
        uint64_t spectralMaterialLookupCount = 0;
        uint64_t spectralDispersionEventCount = 0;
        uint64_t spectralFallbackCount = 0;
        uint64_t spectralCollapseCount = 0;
        uint64_t spectralWavelengthChannelSampleCount = 0;
        std::vector<float> maxThroughputByBounce;
    };

    struct PerformanceSummary {
        bool available = false;
        uint32_t renderExecutionMode = 0;
        uint64_t frameCount = 0;
        double totalGpuSeconds = 0.0;
        double avgGpuMsPerSample = 0.0;
        double totalCpuEncodeSeconds = 0.0;
        double avgCpuEncodeMsPerSample = 0.0;
        double avgFrameGpuMs = 0.0;
        double avgFrameCpuEncodeMs = 0.0;
        bool gpuBucketTimingAvailable = false;
        double gpuBucketIntersectSeconds = 0.0;
        double gpuBucketShadeSeconds = 0.0;
        double gpuBucketShadowQueueSeconds = 0.0;
        double gpuBucketFusedIntegrateSeconds = 0.0;
        double gpuBucketPresentationSeconds = 0.0;
        uint64_t currentAllocatedBytes = 0;
        uint64_t peakAllocatedBytes = 0;
        uint64_t recommendedMaxWorkingSetBytes = 0;
        bool hardwareRaytracingAvailable = false;
        bool hardwareRaytracingActive = false;
        bool hardwareRaytracingFallbackActive = false;
        std::string hardwareRaytracingFallbackReason;
        std::string megakernelIntegratePipeline;
        // Random-walk subsurface over the whole render: walks, exits, absorbed, open mesh, step limit.
        uint64_t sssWalkStats[5] = {};
        // The same counts per material that walked, with the material's name and the render's step limit.
        struct SssWalkMaterialStats {
            std::string material;
            uint64_t stats[5] = {};
        };
        std::vector<SssWalkMaterialStats> sssWalkMaterials;
        uint32_t sssMaxSteps = 0;
        std::string wavefrontDirectLightFallback = "none";
        std::string wavefrontMegakernelFallback = "none";
        std::string deviceName;
        uint64_t deviceRecommendedMaxWorkingSetBytes = 0;
        bool deviceSupportsRaytracing = false;
        bool deviceSupportsCounterSampling = false;
        bool deviceSupportsArgumentBuffers = false;
        bool deviceSupportsSparseResources = false;
        bool deviceSupportsMetalFX = false;
        bool deviceSupportsMLBackend = false;
        std::string accelerationBuildPolicy;
        std::string accelerationBuildPolicyReason;
        std::string accelerationCompactionPolicy;
        std::string accelerationIdentityStrategy;
        std::string accelerationIntersectionFunctionStrategy;
        bool accelerationIdentityStable = false;
        bool accelerationCustomIntersectionFunctionsSupported = false;
        bool accelerationCustomIntersectionFunctionsEnabled = false;
        double accelerationBlasCompactionRatio = 0.0;
        double accelerationTlasCompactionRatio = 0.0;
        std::string accelerationTraversalCoverage;
        uint32_t intersectionMode = 0;
        uint32_t accelerationPrimitiveCount = 0;
        uint32_t accelerationInstanceCount = 0;
        uint32_t accelerationBlasCount = 0;
        uint64_t primaryRayCount = 0;
        uint64_t pathSegmentCount = 0;
        uint64_t shadowRayCount = 0;
        uint64_t tracedRayCount = 0;
        double tracedRaysPerSecond = 0.0;
        double tracedRaysPerGpuSecond = 0.0;
        uint32_t wavefrontSchedulingPolicy = 0;
        uint32_t wavefrontCompactionEnabled = 0;
        uint64_t totalComputeDispatchCount = 0;
        uint64_t restirDiInitializeDispatchCount = 0;
        uint64_t restirDiInitialCandidateDispatchCount = 0;
        uint64_t restirDiTemporalReuseDispatchCount = 0;
        uint64_t restirDiSpatialReuseDispatchCount = 0;
        uint64_t restirDiVisibilityDispatchCount = 0;
        uint64_t restirDiApplyLightingDispatchCount = 0;
        uint64_t restirDiDebugAovDispatchCount = 0;
        uint64_t integrationDispatchCount = 0;
        uint64_t presentationDispatchCount = 0;
        uint64_t wavefrontResetDispatchCount = 0;
        uint64_t wavefrontGenerateDispatchCount = 0;
        uint64_t wavefrontBuildDispatchArgsDispatchCount = 0;
        uint64_t wavefrontCompactQueueDispatchCount = 0;
        uint64_t wavefrontPrepareDispatchCount = 0;
        uint64_t wavefrontIntersectDispatchCount = 0;
        uint64_t wavefrontShadeDispatchCount = 0;
        uint64_t wavefrontShadowDispatchCount = 0;
        uint64_t wavefrontSwapDispatchCount = 0;
        uint64_t wavefrontAccumulateDispatchCount = 0;
        uint64_t wavefrontQueueBytesTotal = 0;
        uint64_t wavefrontBudgetBytes = 0;
        double wavefrontBudgetUsagePercent = 0.0;
        uint64_t restirDiResourceBytesTotal = 0;
        uint64_t restirDiReservoirBytesEach = 0;
        uint64_t restirDiVisibilityBytes = 0;
        uint64_t restirDiDebugBytes = 0;
        std::string restirDiPathStatus;
        uint64_t transportMemoryBytesTotal = 0;
        uint64_t transportMemoryBytesPrivate = 0;
        uint64_t transportMemoryBytesShared = 0;
        uint64_t transportMemoryBytesStaging = 0;
        uint32_t transportMemoryRecordCount = 0;
        uint32_t transportMemoryAllocatorBackedRecordCount = 0;
        uint32_t transportMemoryFrameGraphBackedRecordCount = 0;
        uint32_t transportMemoryOwnershipGapRecordCount = 0;
        uint32_t transportMemoryInvalidatedRecordCount = 0;
        uint64_t wavefrontPrimaryGeneratedCount = 0;
        uint64_t wavefrontIntersectProcessedCount = 0;
        uint64_t wavefrontShadeProcessedCount = 0;
        uint64_t wavefrontShadowProcessedCount = 0;
        uint64_t wavefrontNextRayEnqueueCount = 0;
        uint64_t wavefrontShadowRayEnqueueCount = 0;
        uint32_t wavefrontQueueActiveCounts[kWavefrontQueueMetricCount] = {};
        uint32_t wavefrontQueuePeakActiveCounts[kWavefrontQueueMetricCount] = {};
        uint32_t wavefrontQueueCapacities[kWavefrontQueueMetricCount] = {};
        uint32_t wavefrontQueueOverflowCounts[kWavefrontQueueMetricCount] = {};
        double wavefrontQueueOccupancyRatios[kWavefrontQueueMetricCount] = {};
        double wavefrontQueuePeakOccupancyRatios[kWavefrontQueueMetricCount] = {};
        double wavefrontQueueCompactionCostMs = 0.0;
        std::string wavefrontIntersectDispatchMode;
        std::string wavefrontShadeDispatchMode;
        std::string wavefrontShadowDispatchMode;
        std::string wavefrontQueueStoragePolicySummary;
        std::string wavefrontM1BridgeStatus;
        uint32_t passGraphVersion = 0;
        bool passGraphCompiled = false;
        uint32_t passGraphCompiledExecutablePassCount = 0;
        uint32_t passGraphCompiledSkippedPassCount = 0;
        uint32_t passGraphCompiledUsedResourceCount = 0;
        uint32_t passGraphAliasableTransientResourceCount = 0;
        uint32_t passGraphTransientAliasPoolCount = 0;
        uint32_t passGraphResourceTransitionCount = 0;
        uint32_t passGraphRequiredBarrierCount = 0;
        uint32_t passGraphWriteTransitionCount = 0;
        std::string historyResetReason;
        std::vector<std::string> passGraphCompileWarnings;
        std::vector<std::string> passGraphCompileErrors;
        std::vector<std::string> passGraphExecutedPasses;
        std::vector<std::string> passGraphSkippedPasses;
        std::vector<std::string> passGraphFailedPasses;
        bool passGraphTimingAvailable = false;
        std::vector<std::string> passGraphTimingNames;
        std::vector<double> passGraphTimingMs;
        std::vector<std::string> passGraphNames;
        std::vector<std::string> passGraphTypes;
        std::vector<std::string> passGraphExecutions;
        std::vector<std::string> passGraphPipelines;
        std::vector<std::string> passGraphDispatches;
        std::vector<std::string> passGraphDebugLabels;
        std::vector<std::string> passGraphTimingLabels;
        std::vector<std::string> passGraphDebugPolicies;
        std::vector<std::string> passGraphTimingPolicies;
        std::vector<std::string> passGraphResourceNames;
        std::vector<std::string> passGraphResourceLifetimes;
        std::vector<std::string> passGraphResourceStorage;
        std::vector<std::string> passGraphResourceTargetStorage;
        std::vector<std::string> passGraphResourceSizeFormulas;
        std::vector<std::string> passGraphResourceDebugLabels;
        std::vector<std::string> passGraphResourceCurrentOwners;
        std::vector<std::string> passGraphResourceTargetOwners;
        std::vector<std::string> passGraphResourceFeatureGates;
        std::vector<std::string> passGraphResourceDebugReadbackPolicies;
        std::vector<std::string> passGraphResourceClearLoadPolicies;
        std::vector<std::string> passGraphResourceMigrationRisks;
        std::vector<int32_t> passGraphResourceFirstPasses;
        std::vector<int32_t> passGraphResourceLastPasses;
        std::vector<int32_t> passGraphResourceAliasPools;
        std::vector<uint32_t> passGraphResourceUsed;
        std::vector<uint32_t> passGraphResourceAliasable;
        std::vector<uint32_t> passGraphResourceHotPath;
        std::vector<uint32_t> passGraphResourceTemporaryShared;
        std::vector<uint32_t> passGraphTransitionPassIndices;
        std::vector<std::string> passGraphTransitionPassNames;
        std::vector<std::string> passGraphTransitionResourceNames;
        std::vector<std::string> passGraphTransitionPreviousAccesses;
        std::vector<std::string> passGraphTransitionAccesses;
        std::vector<uint32_t> passGraphTransitionFirstUse;
        std::vector<uint32_t> passGraphTransitionRequiresBarrier;
        std::vector<uint32_t> passGraphTransitionWritesResource;
        std::vector<std::string> transportMemoryNames;
        std::vector<std::string> transportMemoryKinds;
        std::vector<std::string> transportMemoryOwners;
        std::vector<std::string> transportMemoryTargetOwners;
        std::vector<std::string> transportMemoryStorage;
        std::vector<std::string> transportMemoryTargetStorage;
        std::vector<std::string> transportMemoryHistory;
        std::vector<std::string> transportMemoryDebugReadback;
        std::vector<std::string> transportMemoryLastInvalidations;
        std::vector<std::string> transportMemoryByteSizeFormulas;
        std::vector<std::string> transportMemoryFeatureGates;
        std::vector<std::string> transportMemoryMigrationRisks;
        std::vector<uint64_t> transportMemoryBytes;
        std::vector<uint32_t> transportMemoryWidths;
        std::vector<uint32_t> transportMemoryHeights;
        std::vector<uint32_t> transportMemoryDepths;
        std::vector<uint32_t> transportMemoryInvalidationCounts;
        std::vector<uint32_t> transportMemoryHotPath;
        std::vector<uint32_t> transportMemoryTemporaryShared;
        std::vector<uint32_t> transportMemoryFrameGraphOwned;
        std::vector<uint32_t> transportMemoryAllocatorOwned;
        uint32_t mlBackendMode = 0;
        bool mlBackendRequested = false;
        bool mlBackendActive = false;
        bool mlBackendNativeAccelerationAvailable = false;
        uint64_t mlBackendPassCount = 0;
        std::string mlBackendProvider;
        std::string mlBackendProviderVersion;
        std::string mlBackendTensorContract;
        std::string mlBackendFallbackReason;
        float mlBackendConfidence = 0.0f;
        float mlBackendBiasRisk = 0.0f;
        std::string mlModelPackagePath;
        bool pipelineCompilationAvailable = false;
        bool pipelineRuntimeSourceCompilation = true;
        bool pipelineFunctionConstantsUsed = false;
        bool pipelineMetalBinaryArchiveUsed = false;
        bool pipelineMetalBinaryArchiveSerialized = false;
        bool pipelineHarvestedCompilationReady = false;
        uint64_t pipelineShaderSourceBytes = 0;
        double pipelineLibraryCompileMs = 0.0;
        uint32_t pipelineComputePipelineCount = 0;
        uint32_t pipelineRenderPipelineCount = 0;
        uint32_t pipelineFailedPipelineCount = 0;
        std::string pipelineCacheStrategy;
        std::string pipelineSpecializationStrategy;
        std::string pipelineBinaryArchivePath;
        std::vector<std::string> pipelineCompileKeys;
        std::vector<std::string> pipelineCompileFunctions;
        std::vector<std::string> pipelineCompileStages;
        std::vector<double> pipelineCompileMs;
        std::vector<uint32_t> pipelineCompileSuccess;
        std::vector<uint32_t> pipelineCompileRayTracingFlags;
        std::vector<std::string> pipelineCompileSpecializations;
    };

    std::vector<float> linearRGB;
    std::vector<float> albedoLinearRGB;
    std::vector<float> normalLinearRGB; // Encoded [0,1] normal guide buffer matching accumulation output
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t samples = 0;
    double totalSeconds = 0.0;
    double avgMsPerSample = 0.0;
    PbrMetrics pbrMetrics;
    PerformanceSummary performance;
    MaterialMrDebugDump materialMrDebug;
    DirectLightAuditDump directLightAudit;
};

// Headless sample checkpoints (--checkpointEverySamples, --resumeCheckpointJson on a partial checkpoint).
struct HeadlessSampleCheckpointOptions {
    // Export the accumulation every this many samples (0: never). The last batch before each multiple is shortened
    // so that exports land on the multiples.
    uint32_t everySamples = 0;
    std::function<bool(const PathTracer::AccumulationState&, std::string&)> write;
    // Restore this accumulation before the first sample and render only the samples it does not have yet.
    const PathTracer::AccumulationState* resumeFrom = nullptr;
};

class IHeadlessRenderer {
public:
    virtual ~IHeadlessRenderer() = default;
    // Returns false, with a reason, if the backend cannot honour the options.
    virtual bool setSampleCheckpointOptions(const HeadlessSampleCheckpointOptions& options, std::string& error) {
        if (options.everySamples == 0u && options.resumeFrom == nullptr) {
            return true;
        }
        error = "this backend does not support sample checkpoints";
        return false;
    }
    virtual bool render(const HeadlessScene& scene,
                        const HeadlessCamera& camera,
                        const PathTracer::RenderSettings& settings,
                        uint32_t sppTotal,
                        bool verbose,
                        bool statsMode,
                        bool captureAuxBuffers,
                        HeadlessRenderOutput& out,
                        std::string& error) = 0;
};
