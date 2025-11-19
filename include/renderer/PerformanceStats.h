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
    uint64_t hardwareSelfHitRejectedCount = 0;
    uint64_t hardwareMissDistanceBins[32] = {0};
    float hardwareMissLastDistance = 0.0f;
    float hardwareSelfHitLastDistance = 0.0f;
    uint32_t debugPathEntryCount = 0;
    uint32_t debugPathMaxEntries = 0;
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
};

}  // namespace PathTracer
