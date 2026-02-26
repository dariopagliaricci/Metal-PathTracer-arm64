#pragma once

// ⚠️ INTERNAL HEADER - Subject to change without notice
// This header is part of PathTracer's internal implementation.
// Use only the public API from MetalRenderer.h

#include <cstdint>

namespace PathTracer {

/// Performance statistics for display in UI
struct PerformanceStats {
    // Timing
    double frameTimeMs = 0.0;
    double gpuTimeMs = 0.0;
    double cpuEncodeMs = 0.0;
    double drawableWaitMs = 0.0;
    double samplesPerMinute = 0.0;
    
    // Progress
    uint32_t sampleCount = 0;
    uint32_t activeSamplesPerFrame = 0;
    
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
