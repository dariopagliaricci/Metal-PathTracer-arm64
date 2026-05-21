#pragma once

// ⚠️ INTERNAL HEADER - Subject to change without notice
// This header is part of PathTracer's internal implementation.
// Use only the public API from MetalRenderer.h

#include <cstdint>

namespace PathTracer {

/// Performance statistics for display in UI
struct PerformanceStats {
    static constexpr uint32_t kWavefrontQueueMetricCount = 12;

    // Timing
    double frameTimeMs = 0.0;
    double gpuTimeMs = 0.0;
    double cpuEncodeMs = 0.0;
    double drawableWaitMs = 0.0;
    double samplesPerMinute = 0.0;
    bool gpuBucketTimingAvailable = false;
    double gpuBucketIntersectMs = 0.0;
    double gpuBucketShadeMs = 0.0;
    double gpuBucketShadowQueueMs = 0.0;
    double gpuBucketFusedIntegrateMs = 0.0;
    double gpuBucketPresentationMs = 0.0;
    uint64_t currentAllocatedBytes = 0;
    uint64_t recommendedMaxWorkingSetBytes = 0;
    
    // Progress
    uint32_t sampleCount = 0;
    uint32_t activeSamplesPerFrame = 0;
    uint32_t renderExecutionMode = 0;
    uint32_t wavefrontSchedulingPolicy = 0;
    uint32_t wavefrontCompactionEnabled = 0;

    // Scene stats
    uint32_t sphereCount = 0;
    uint32_t triangleCount = 0;
    
    // BVH stats
    uint32_t bvhNodeCount = 0;
    uint32_t bvhPrimitiveCount = 0;
    double avgNodesVisited = 0.0;
    double avgLeafPrimTests = 0.0;
    double shadowRayEarlyExitPct = 0.0;
    double bothChildrenVisitedPct = 0.0;
    double hardwareRayPctHit = 0.0;
    double hardwareRayPctMiss = 0.0;
    uint64_t hardwareRayCount = 0;
    uint64_t hardwareHitCount = 0;
    uint64_t hardwareMissCount = 0;
    uint64_t hardwareResultNoneCount = 0;
    uint64_t hardwareRejectedCount = 0;
    uint64_t hardwareUnavailableCount = 0;
    uint64_t specularNeeOcclusionHitCount = 0;
    uint64_t specNeeEnvAddedCount = 0;
    uint64_t specNeeRectAddedCount = 0;
    uint64_t mneeEnvHwOccludedCount = 0;
    uint64_t mneeEnvSwOccludedCount = 0;
    uint64_t mneeEnvHwSwMismatchCount = 0;
    uint64_t mneeRectHwOccludedCount = 0;
    uint64_t mneeRectSwOccludedCount = 0;
    uint64_t mneeRectHwSwMismatchCount = 0;
    uint64_t mneeHitHwSwHitMissCount = 0;
    uint64_t mneeHitHwSwNormalMismatchCount = 0;
    uint64_t mneeHitHwSwIdMismatchCount = 0;
    uint64_t mneeHitHwSwTDiffCount = 0;
    uint64_t mneeChainHwSwHitMissCount = 0;
    uint64_t mneeChainHwSwNormalMismatchCount = 0;
    uint64_t mneeChainHwSwIdMismatchCount = 0;
    uint64_t mneeChainHwSwTDiffCount = 0;
    uint64_t mneeEligibleCount = 0;
    uint64_t mneeEnvAttemptCount = 0;
    uint64_t mneeEnvAddedCount = 0;
    uint64_t mneeRectAttemptCount = 0;
    uint64_t mneeRectAddedCount = 0;
    uint64_t mneeContributionCount = 0;
    uint64_t mneeContributionLumaSumFixed = 0;
    uint64_t hardwareSelfHitRejectedCount = 0;
    uint64_t hardwareMissDistanceBins[32] = {0};
    uint64_t hardwareExcludeRetryHistogram[4] = {0};
    float hardwareMissLastDistance = 0.0f;
    float hardwareSelfHitLastDistance = 0.0f;
    uint32_t hardwareMissLastInstanceId = 0;
    uint32_t hardwareMissLastPrimitiveId = 0;
    uint64_t hardwareFallbackHitCount = 0;
    uint64_t hardwareFirstHitFallbackCount = 0;
    uint64_t hardwareShadowTMinRetryCount = 0;
    uint64_t hardwareShadowNearForeignHitCount = 0;
    uint64_t throughputNanInfCount = 0;
    uint64_t radianceNanInfCount = 0;
    uint64_t clampEventCount = 0;
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
    uint64_t radianceCacheQueryCount = 0;
    uint64_t radianceCacheHitCount = 0;
    uint64_t radianceCacheMissCount = 0;
    uint64_t radianceCacheTrainCount = 0;
    uint64_t radianceCacheFallbackCount = 0;
    uint64_t radianceCacheInvalidCount = 0;
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
    uint32_t debugPathEntryCount = 0;
    uint32_t debugPathMaxEntries = 0;
    uint32_t parityEntryCount = 0;
    uint32_t parityMaxEntries = 0;
    uint32_t parityLastReasonMask = 0;
    uint32_t parityLastPixelX = 0;
    uint32_t parityLastPixelY = 0;
    uint32_t parityLastDepth = 0;
    uint32_t parityLastHwHit = 0;
    uint32_t parityLastSwHit = 0;
    uint32_t parityLastHwFrontFace = 0;
    uint32_t parityLastSwFrontFace = 0;
    uint32_t parityLastHwMaterialIndex = 0;
    uint32_t parityLastSwMaterialIndex = 0;
    uint32_t parityLastHwMeshIndex = 0;
    uint32_t parityLastSwMeshIndex = 0;
    uint32_t parityLastHwPrimitiveIndex = 0;
    uint32_t parityLastSwPrimitiveIndex = 0;
    float parityLastHwT = 0.0f;
    float parityLastSwT = 0.0f;
    uint32_t parityChecksPerformed = 0;
    uint32_t parityChecksInMedium = 0;
    uint32_t hardwareLastResultType = 0;
    uint32_t hardwareLastInstanceId = 0;
    uint32_t hardwareLastPrimitiveId = 0;
    float hardwareLastDistance = 0.0f;
    uint32_t wavefrontActiveRayCount = 0;
    uint32_t wavefrontNextRayCount = 0;
    uint32_t wavefrontShadowRayCount = 0;
    uint32_t wavefrontTerminatedCount = 0;
    uint32_t wavefrontOverflowFlag = 0;
    uint64_t wavefrontTerminateMissCount = 0;
    uint64_t wavefrontTerminateEmissiveCount = 0;
    uint64_t wavefrontTerminateMaxDepthCount = 0;
    uint64_t wavefrontTerminateOverflowCount = 0;
    uint32_t wavefrontEnvShadowRayCount = 0;
    uint32_t wavefrontAovAlbedoValidCount = 0;
    uint32_t wavefrontAovNormalValidCount = 0;
    uint64_t wavefrontQueueBytesActive = 0;
    uint64_t wavefrontQueueBytesNext = 0;
    uint64_t wavefrontQueueBytesHit = 0;
    uint64_t wavefrontQueueBytesShadow = 0;
    uint64_t wavefrontQueueBytesPathState = 0;
    uint64_t wavefrontQueueBytesTotal = 0;
    uint64_t wavefrontBudgetBytes = 0;
    double wavefrontBudgetUsagePercent = 0.0;
    uint64_t transportMemoryBytesTotal = 0;
    uint64_t transportMemoryBytesPrivate = 0;
    uint64_t transportMemoryBytesShared = 0;
    uint64_t transportMemoryBytesStaging = 0;
    uint32_t transportMemoryRecordCount = 0;
    uint32_t transportMemoryInvalidatedRecordCount = 0;
    uint32_t transportMemoryLastInvalidationReason = 0;
    uint64_t primaryRayCount = 0;
    uint64_t shadowRayCount = 0;
    uint64_t pathSegmentCount = 0;
    uint64_t tracedRayCount = 0;
    uint64_t totalComputeDispatchCount = 0;
    uint64_t restirDiInitializeDispatchCount = 0;
    uint64_t restirDiInitialCandidateDispatchCount = 0;
    uint64_t restirDiTemporalReuseDispatchCount = 0;
    uint64_t restirDiSpatialReuseDispatchCount = 0;
    uint64_t restirDiVisibilityDispatchCount = 0;
    uint64_t restirDiApplyLightingDispatchCount = 0;
    uint64_t restirDiDebugAovDispatchCount = 0;
    bool restirDiHardwareTraceActive = false;
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

    // Resolution
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    double renderScale = 1.0;

    // Intersection / hardware status
    uint32_t intersectionMode = 0;
    bool hardwareRaytracingAvailable = false;
    bool hardwareRaytracingActive = false;
    bool cameraMotionActive = false;
    uint32_t envRadianceMipCount = 0;
};

}  // namespace PathTracer
