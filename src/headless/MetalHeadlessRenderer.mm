#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>

#include "headless/MetalHeadlessRenderer.h"
#include "MetalRenderer.h"

namespace {

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

}  // namespace

bool MetalHeadlessRenderer::render(const HeadlessScene& scene,
                                   const HeadlessCamera&,
                                   const PathTracer::RenderSettings& settings,
                                   uint32_t sppTotal,
                                   bool verbose,
                                   bool statsMode,
                                   bool captureAuxBuffers,
                                   HeadlessRenderOutput& out,
                                   std::string& error) {
    PathTracer::RenderSettings rendererSettings = settings;
    rendererSettings.denoiseEnabled = false;

    MetalRendererOptions rendererOptions;
    rendererOptions.headless = true;
    rendererOptions.width = rendererSettings.renderWidth > 0 ? static_cast<int>(rendererSettings.renderWidth) : 1280;
    rendererOptions.height = rendererSettings.renderHeight > 0 ? static_cast<int>(rendererSettings.renderHeight) : 720;
    rendererOptions.windowTitle = "PathTracerCLI";
    rendererOptions.fixedRngSeed = rendererSettings.fixedRngSeed;
    rendererOptions.enableSoftwareRayTracing = rendererSettings.enableSoftwareRayTracing;
    rendererOptions.verbose = verbose;

    MetalRenderer renderer;
    if (!renderer.init(nullptr, rendererOptions)) {
        error = "Failed to initialize Metal renderer";
        return false;
    }

    bool sceneLoaded = false;
    if (scene.resources) {
        sceneLoaded = renderer.useSceneResources(*scene.resources, scene.source.c_str());
    } else if (scene.isPath) {
        sceneLoaded = renderer.loadSceneFromPath(scene.source.c_str());
    } else {
        sceneLoaded = renderer.setScene(scene.source.c_str());
    }

    if (!sceneLoaded) {
        error = "Failed to load scene: " + scene.source;
        return false;
    }

    renderer.applySettings(rendererSettings, true);

    const uint32_t targetSamples = std::max<uint32_t>(1u, sppTotal);
    uint32_t accumulatedSamples = renderer.sampleCount();
    uint32_t maxBatch = rendererSettings.enableSoftwareRayTracing ? 1u : 16u;
#if PT_MNEE_SWRT_RAYS
    if (verbose) {
        NSLog(@"[Warning] PT_MNEE_SWRT_RAYS enabled: MNEE rays use SWRT (slow diagnostic mode)");
    }
    if (!rendererSettings.enableSoftwareRayTracing && rendererSettings.enableMnee) {
        // Temporary: Keep headless progress responsive when SWRT is used for MNEE rays.
        // This is symptom management; revisit once timing logs reveal actual bottlenecks.
        maxBatch = 1u;
    }
#endif

    CFAbsoluteTime renderStart = CFAbsoluteTimeGetCurrent();
    double accumulatedFrameTime = 0.0;
    double accumulatedGpuTimeMs = 0.0;
    double accumulatedCpuEncodeMs = 0.0;
    double accumulatedBucketIntersectMs = 0.0;
    double accumulatedBucketShadeMs = 0.0;
    double accumulatedBucketShadowQueueMs = 0.0;
    double accumulatedBucketFusedIntegrateMs = 0.0;
    double accumulatedBucketPresentationMs = 0.0;
    bool gpuBucketTimingAvailable = false;
    uint64_t frameCount = 0;
    uint64_t totalPrimaryRayCount = 0;
    uint64_t totalPathSegmentCount = 0;
    uint64_t totalShadowRayCount = 0;
    uint64_t totalTracedRayCount = 0;
    uint64_t totalComputeDispatchCount = 0;
    uint64_t totalRestirDiInitializeDispatchCount = 0;
    uint64_t totalRestirDiInitialCandidateDispatchCount = 0;
    uint64_t totalRestirDiTemporalReuseDispatchCount = 0;
    uint64_t totalRestirDiSpatialReuseDispatchCount = 0;
    uint64_t totalRestirDiVisibilityDispatchCount = 0;
    uint64_t totalRestirDiApplyLightingDispatchCount = 0;
    uint64_t totalRestirDiDebugAovDispatchCount = 0;
    uint64_t totalIntegrationDispatchCount = 0;
    uint64_t totalPresentationDispatchCount = 0;
    uint64_t totalWavefrontResetDispatchCount = 0;
    uint64_t totalWavefrontGenerateDispatchCount = 0;
    uint64_t totalWavefrontBuildDispatchArgsDispatchCount = 0;
    uint64_t totalWavefrontCompactQueueDispatchCount = 0;
    uint64_t totalWavefrontPrepareDispatchCount = 0;
    uint64_t totalWavefrontIntersectDispatchCount = 0;
    uint64_t totalWavefrontShadeDispatchCount = 0;
    uint64_t totalWavefrontShadowDispatchCount = 0;
    uint64_t totalWavefrontSwapDispatchCount = 0;
    uint64_t totalWavefrontAccumulateDispatchCount = 0;
    uint64_t totalWavefrontPrimaryGeneratedCount = 0;
    uint64_t totalWavefrontIntersectProcessedCount = 0;
    uint64_t totalWavefrontShadeProcessedCount = 0;
    uint64_t totalWavefrontShadowProcessedCount = 0;
    uint64_t totalWavefrontNextRayEnqueueCount = 0;
    uint64_t totalWavefrontShadowRayEnqueueCount = 0;
    uint32_t wavefrontQueueActiveCounts[HeadlessRenderOutput::kWavefrontQueueMetricCount] = {};
    uint32_t wavefrontQueuePeakActiveCounts[HeadlessRenderOutput::kWavefrontQueueMetricCount] = {};
    uint32_t wavefrontQueueCapacities[HeadlessRenderOutput::kWavefrontQueueMetricCount] = {};
    uint32_t wavefrontQueueOverflowCounts[HeadlessRenderOutput::kWavefrontQueueMetricCount] = {};
    double wavefrontQueueOccupancyRatios[HeadlessRenderOutput::kWavefrontQueueMetricCount] = {};
    double wavefrontQueuePeakOccupancyRatios[HeadlessRenderOutput::kWavefrontQueueMetricCount] = {};
    double wavefrontQueueCompactionCostMs = 0.0;
    std::string wavefrontIntersectDispatchMode;
    std::string wavefrontShadeDispatchMode;
    std::string wavefrontShadowDispatchMode;
    std::string wavefrontQueueStoragePolicySummary;
    std::string wavefrontM1BridgeStatus;
    uint64_t lastAllocatedBytes = 0;
    uint64_t peakAllocatedBytes = 0;
    uint64_t recommendedMaxWorkingSetBytes = 0;
    uint64_t wavefrontQueueBytesTotal = 0;
    uint64_t wavefrontBudgetBytes = 0;
    double wavefrontBudgetUsagePercent = 0.0;
    uint64_t restirDiResourceBytesTotal = 0;
    uint64_t restirDiReservoirBytesEach = 0;
    uint64_t restirDiVisibilityBytes = 0;
    uint64_t restirDiDebugBytes = 0;
    uint64_t transportMemoryBytesTotal = 0;
    uint64_t transportMemoryBytesPrivate = 0;
    uint64_t transportMemoryBytesShared = 0;
    uint64_t transportMemoryBytesStaging = 0;
    uint32_t transportMemoryRecordCount = 0;
    uint32_t transportMemoryInvalidatedRecordCount = 0;
    uint32_t renderExecutionMode = 0;
    uint32_t wavefrontSchedulingPolicy = 0;
    uint32_t wavefrontCompactionEnabled = 0;
    bool hardwareRaytracingAvailable = false;
    bool hardwareRaytracingActive = false;
    bool hardwareRaytracingFallbackActive = false;
    std::string hardwareRaytracingFallbackReason;
    std::string deviceName;
    uint64_t deviceRecommendedMaxWorkingSetBytes = 0;
    bool deviceSupportsRaytracing = false;
    bool deviceSupportsCounterSampling = false;
    bool deviceSupportsArgumentBuffers = false;
    bool deviceSupportsSparseResources = false;
    bool deviceSupportsMetalFX = false;
    bool deviceSupportsMLBackend = false;
    std::string accelerationBuildPolicy;
    std::string accelerationIdentityStrategy;
    std::string accelerationIntersectionFunctionStrategy;
    bool accelerationIdentityStable = false;
    bool accelerationCustomIntersectionFunctionsSupported = false;
    bool accelerationCustomIntersectionFunctionsEnabled = false;
    uint32_t intersectionMode = 0;
    uint32_t accelerationPrimitiveCount = 0;
    uint32_t accelerationInstanceCount = 0;
    uint32_t accelerationBlasCount = 0;
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
    bool pipelineHarvestedCompilationReady = false;
    uint64_t pipelineShaderSourceBytes = 0;
    double pipelineLibraryCompileMs = 0.0;
    uint32_t pipelineComputePipelineCount = 0;
    uint32_t pipelineRenderPipelineCount = 0;
    uint32_t pipelineFailedPipelineCount = 0;
    std::string pipelineCacheStrategy;
    std::string pipelineSpecializationStrategy;
    std::vector<std::string> pipelineCompileKeys;
    std::vector<std::string> pipelineCompileFunctions;
    std::vector<std::string> pipelineCompileStages;
    std::vector<double> pipelineCompileMs;
    std::vector<uint32_t> pipelineCompileSuccess;
    std::vector<uint32_t> pipelineCompileRayTracingFlags;
    std::vector<std::string> pipelineCompileSpecializations;
    CFAbsoluteTime lastProgress = renderStart;
    CFAbsoluteTime lastStatsPrint = renderStart;
    uint64_t lastStatsRayCount = 0;

    while (accumulatedSamples < targetSamples) {
        uint32_t remaining = targetSamples - accumulatedSamples;
        uint32_t request = (maxBatch == 0) ? remaining : std::min(maxBatch, remaining);
        renderer.setSamplesPerFrame(request);

        uint32_t before = renderer.sampleCount();
        CFAbsoluteTime frameStart = CFAbsoluteTimeGetCurrent();
        renderer.drawFrame();
        CFAbsoluteTime frameEnd = CFAbsoluteTimeGetCurrent();

        uint32_t after = renderer.sampleCount();
        uint32_t produced = (after > before) ? (after - before) : std::min(request, remaining);
        if (produced == 0) {
            produced = 1;
        }
        MetalRenderer::RuntimeStatsSnapshot statsSnapshot{};
        if (renderer.captureRuntimeStats(statsSnapshot) && statsSnapshot.available) {
            frameCount += 1u;
            accumulatedGpuTimeMs += statsSnapshot.gpuTimeMs;
            accumulatedCpuEncodeMs += statsSnapshot.cpuEncodeMs;
            totalPrimaryRayCount += statsSnapshot.primaryRayCount;
            totalPathSegmentCount += statsSnapshot.pathSegmentCount;
            totalShadowRayCount += statsSnapshot.shadowRayCount;
            totalTracedRayCount += statsSnapshot.tracedRayCount;
            totalComputeDispatchCount += statsSnapshot.totalComputeDispatchCount;
            totalRestirDiInitializeDispatchCount +=
                statsSnapshot.restirDiInitializeDispatchCount;
            totalRestirDiInitialCandidateDispatchCount +=
                statsSnapshot.restirDiInitialCandidateDispatchCount;
            totalRestirDiTemporalReuseDispatchCount +=
                statsSnapshot.restirDiTemporalReuseDispatchCount;
            totalRestirDiSpatialReuseDispatchCount +=
                statsSnapshot.restirDiSpatialReuseDispatchCount;
            totalRestirDiVisibilityDispatchCount +=
                statsSnapshot.restirDiVisibilityDispatchCount;
            totalRestirDiApplyLightingDispatchCount +=
                statsSnapshot.restirDiApplyLightingDispatchCount;
            totalRestirDiDebugAovDispatchCount +=
                statsSnapshot.restirDiDebugAovDispatchCount;
            totalIntegrationDispatchCount += statsSnapshot.integrationDispatchCount;
            totalPresentationDispatchCount += statsSnapshot.presentationDispatchCount;
            totalWavefrontResetDispatchCount += statsSnapshot.wavefrontResetDispatchCount;
            totalWavefrontGenerateDispatchCount += statsSnapshot.wavefrontGenerateDispatchCount;
            totalWavefrontBuildDispatchArgsDispatchCount +=
                statsSnapshot.wavefrontBuildDispatchArgsDispatchCount;
            totalWavefrontCompactQueueDispatchCount +=
                statsSnapshot.wavefrontCompactQueueDispatchCount;
            totalWavefrontPrepareDispatchCount += statsSnapshot.wavefrontPrepareDispatchCount;
            totalWavefrontIntersectDispatchCount += statsSnapshot.wavefrontIntersectDispatchCount;
            totalWavefrontShadeDispatchCount += statsSnapshot.wavefrontShadeDispatchCount;
            totalWavefrontShadowDispatchCount += statsSnapshot.wavefrontShadowDispatchCount;
            totalWavefrontSwapDispatchCount += statsSnapshot.wavefrontSwapDispatchCount;
            totalWavefrontAccumulateDispatchCount += statsSnapshot.wavefrontAccumulateDispatchCount;
            totalWavefrontPrimaryGeneratedCount += statsSnapshot.wavefrontPrimaryGeneratedCount;
            totalWavefrontIntersectProcessedCount += statsSnapshot.wavefrontIntersectProcessedCount;
            totalWavefrontShadeProcessedCount += statsSnapshot.wavefrontShadeProcessedCount;
            totalWavefrontShadowProcessedCount += statsSnapshot.wavefrontShadowProcessedCount;
            totalWavefrontNextRayEnqueueCount += statsSnapshot.wavefrontNextRayEnqueueCount;
            totalWavefrontShadowRayEnqueueCount += statsSnapshot.wavefrontShadowRayEnqueueCount;
            for (uint32_t i = 0u; i < HeadlessRenderOutput::kWavefrontQueueMetricCount; ++i) {
                wavefrontQueueActiveCounts[i] = statsSnapshot.wavefrontQueueActiveCounts[i];
                wavefrontQueuePeakActiveCounts[i] =
                    std::max(wavefrontQueuePeakActiveCounts[i],
                             statsSnapshot.wavefrontQueuePeakActiveCounts[i]);
                wavefrontQueueCapacities[i] = statsSnapshot.wavefrontQueueCapacities[i];
                wavefrontQueueOverflowCounts[i] += statsSnapshot.wavefrontQueueOverflowCounts[i];
                wavefrontQueueOccupancyRatios[i] = statsSnapshot.wavefrontQueueOccupancyRatios[i];
                wavefrontQueuePeakOccupancyRatios[i] =
                    std::max(wavefrontQueuePeakOccupancyRatios[i],
                             statsSnapshot.wavefrontQueuePeakOccupancyRatios[i]);
            }
            wavefrontQueueCompactionCostMs += statsSnapshot.wavefrontQueueCompactionCostMs;
            wavefrontIntersectDispatchMode = statsSnapshot.wavefrontIntersectDispatchMode;
            wavefrontShadeDispatchMode = statsSnapshot.wavefrontShadeDispatchMode;
            wavefrontShadowDispatchMode = statsSnapshot.wavefrontShadowDispatchMode;
            wavefrontQueueStoragePolicySummary = statsSnapshot.wavefrontQueueStoragePolicySummary;
            wavefrontM1BridgeStatus = statsSnapshot.wavefrontM1BridgeStatus;
            gpuBucketTimingAvailable = gpuBucketTimingAvailable || statsSnapshot.gpuBucketTimingAvailable;
            accumulatedBucketIntersectMs += statsSnapshot.gpuBucketIntersectMs;
            accumulatedBucketShadeMs += statsSnapshot.gpuBucketShadeMs;
            accumulatedBucketShadowQueueMs += statsSnapshot.gpuBucketShadowQueueMs;
            accumulatedBucketFusedIntegrateMs += statsSnapshot.gpuBucketFusedIntegrateMs;
            accumulatedBucketPresentationMs += statsSnapshot.gpuBucketPresentationMs;
            lastAllocatedBytes = statsSnapshot.currentAllocatedBytes;
            peakAllocatedBytes = std::max(peakAllocatedBytes, statsSnapshot.currentAllocatedBytes);
            recommendedMaxWorkingSetBytes = statsSnapshot.recommendedMaxWorkingSetBytes;
            wavefrontQueueBytesTotal = statsSnapshot.wavefrontQueueBytesTotal;
            wavefrontBudgetBytes = statsSnapshot.wavefrontBudgetBytes;
            wavefrontBudgetUsagePercent = statsSnapshot.wavefrontBudgetUsagePercent;
            restirDiResourceBytesTotal = statsSnapshot.restirDiResourceBytesTotal;
            restirDiReservoirBytesEach = statsSnapshot.restirDiReservoirBytesEach;
            restirDiVisibilityBytes = statsSnapshot.restirDiVisibilityBytes;
            restirDiDebugBytes = statsSnapshot.restirDiDebugBytes;
            transportMemoryBytesTotal = statsSnapshot.transportMemoryBytesTotal;
            transportMemoryBytesPrivate = statsSnapshot.transportMemoryBytesPrivate;
            transportMemoryBytesShared = statsSnapshot.transportMemoryBytesShared;
            transportMemoryBytesStaging = statsSnapshot.transportMemoryBytesStaging;
            transportMemoryRecordCount = statsSnapshot.transportMemoryRecordCount;
            transportMemoryInvalidatedRecordCount =
                statsSnapshot.transportMemoryInvalidatedRecordCount;
            renderExecutionMode = statsSnapshot.renderExecutionMode;
            wavefrontSchedulingPolicy = statsSnapshot.wavefrontSchedulingPolicy;
            wavefrontCompactionEnabled = statsSnapshot.wavefrontCompactionEnabled;
            hardwareRaytracingAvailable = statsSnapshot.hardwareRaytracingAvailable;
            hardwareRaytracingActive = statsSnapshot.hardwareRaytracingActive;
            hardwareRaytracingFallbackActive =
                statsSnapshot.hardwareRaytracingFallbackActive;
            hardwareRaytracingFallbackReason =
                statsSnapshot.hardwareRaytracingFallbackReason;
            deviceName = statsSnapshot.deviceName;
            deviceRecommendedMaxWorkingSetBytes =
                statsSnapshot.deviceRecommendedMaxWorkingSetBytes;
            deviceSupportsRaytracing = statsSnapshot.deviceSupportsRaytracing;
            deviceSupportsCounterSampling = statsSnapshot.deviceSupportsCounterSampling;
            deviceSupportsArgumentBuffers = statsSnapshot.deviceSupportsArgumentBuffers;
            deviceSupportsSparseResources = statsSnapshot.deviceSupportsSparseResources;
            deviceSupportsMetalFX = statsSnapshot.deviceSupportsMetalFX;
            deviceSupportsMLBackend = statsSnapshot.deviceSupportsMLBackend;
            accelerationBuildPolicy = statsSnapshot.accelerationBuildPolicy;
            accelerationIdentityStrategy = statsSnapshot.accelerationIdentityStrategy;
            accelerationIntersectionFunctionStrategy =
                statsSnapshot.accelerationIntersectionFunctionStrategy;
            accelerationIdentityStable = statsSnapshot.accelerationIdentityStable;
            accelerationCustomIntersectionFunctionsSupported =
                statsSnapshot.accelerationCustomIntersectionFunctionsSupported;
            accelerationCustomIntersectionFunctionsEnabled =
                statsSnapshot.accelerationCustomIntersectionFunctionsEnabled;
            intersectionMode = statsSnapshot.intersectionMode;
            accelerationPrimitiveCount = statsSnapshot.accelerationPrimitiveCount;
            accelerationInstanceCount = statsSnapshot.accelerationInstanceCount;
            accelerationBlasCount = statsSnapshot.accelerationBlasCount;
            passGraphVersion = statsSnapshot.passGraphVersion;
            passGraphCompiled = statsSnapshot.passGraphCompiled;
            passGraphCompiledExecutablePassCount =
                statsSnapshot.passGraphCompiledExecutablePassCount;
            passGraphCompiledSkippedPassCount =
                statsSnapshot.passGraphCompiledSkippedPassCount;
            passGraphCompiledUsedResourceCount =
                statsSnapshot.passGraphCompiledUsedResourceCount;
            passGraphAliasableTransientResourceCount =
                statsSnapshot.passGraphAliasableTransientResourceCount;
            passGraphTransientAliasPoolCount =
                statsSnapshot.passGraphTransientAliasPoolCount;
            passGraphResourceTransitionCount =
                statsSnapshot.passGraphResourceTransitionCount;
            passGraphRequiredBarrierCount =
                statsSnapshot.passGraphRequiredBarrierCount;
            passGraphWriteTransitionCount =
                statsSnapshot.passGraphWriteTransitionCount;
            historyResetReason = statsSnapshot.historyResetReason;
            passGraphCompileWarnings = statsSnapshot.passGraphCompileWarnings;
            passGraphCompileErrors = statsSnapshot.passGraphCompileErrors;
            passGraphExecutedPasses = statsSnapshot.passGraphExecutedPasses;
            passGraphSkippedPasses = statsSnapshot.passGraphSkippedPasses;
            passGraphFailedPasses = statsSnapshot.passGraphFailedPasses;
            passGraphNames = statsSnapshot.passGraphNames;
            passGraphTypes = statsSnapshot.passGraphTypes;
            passGraphExecutions = statsSnapshot.passGraphExecutions;
            passGraphPipelines = statsSnapshot.passGraphPipelines;
            passGraphDispatches = statsSnapshot.passGraphDispatches;
            passGraphDebugLabels = statsSnapshot.passGraphDebugLabels;
            passGraphTimingLabels = statsSnapshot.passGraphTimingLabels;
            passGraphDebugPolicies = statsSnapshot.passGraphDebugPolicies;
            passGraphTimingPolicies = statsSnapshot.passGraphTimingPolicies;
            passGraphResourceNames = statsSnapshot.passGraphResourceNames;
            passGraphResourceLifetimes = statsSnapshot.passGraphResourceLifetimes;
            passGraphResourceStorage = statsSnapshot.passGraphResourceStorage;
            passGraphResourceTargetStorage = statsSnapshot.passGraphResourceTargetStorage;
            passGraphResourceSizeFormulas = statsSnapshot.passGraphResourceSizeFormulas;
            passGraphResourceDebugLabels = statsSnapshot.passGraphResourceDebugLabels;
            passGraphResourceCurrentOwners = statsSnapshot.passGraphResourceCurrentOwners;
            passGraphResourceTargetOwners = statsSnapshot.passGraphResourceTargetOwners;
            passGraphResourceFeatureGates = statsSnapshot.passGraphResourceFeatureGates;
            passGraphResourceDebugReadbackPolicies =
                statsSnapshot.passGraphResourceDebugReadbackPolicies;
            passGraphResourceMigrationRisks = statsSnapshot.passGraphResourceMigrationRisks;
            passGraphResourceFirstPasses = statsSnapshot.passGraphResourceFirstPasses;
            passGraphResourceLastPasses = statsSnapshot.passGraphResourceLastPasses;
            passGraphResourceAliasPools = statsSnapshot.passGraphResourceAliasPools;
            passGraphResourceUsed = statsSnapshot.passGraphResourceUsed;
            passGraphResourceAliasable = statsSnapshot.passGraphResourceAliasable;
            passGraphResourceHotPath = statsSnapshot.passGraphResourceHotPath;
            passGraphResourceTemporaryShared = statsSnapshot.passGraphResourceTemporaryShared;
            passGraphTransitionPassIndices = statsSnapshot.passGraphTransitionPassIndices;
            passGraphTransitionPassNames = statsSnapshot.passGraphTransitionPassNames;
            passGraphTransitionResourceNames = statsSnapshot.passGraphTransitionResourceNames;
            passGraphTransitionPreviousAccesses =
                statsSnapshot.passGraphTransitionPreviousAccesses;
            passGraphTransitionAccesses = statsSnapshot.passGraphTransitionAccesses;
            passGraphTransitionFirstUse = statsSnapshot.passGraphTransitionFirstUse;
            passGraphTransitionRequiresBarrier =
                statsSnapshot.passGraphTransitionRequiresBarrier;
            passGraphTransitionWritesResource = statsSnapshot.passGraphTransitionWritesResource;
            transportMemoryNames = statsSnapshot.transportMemoryNames;
            transportMemoryKinds = statsSnapshot.transportMemoryKinds;
            transportMemoryOwners = statsSnapshot.transportMemoryOwners;
            transportMemoryTargetOwners = statsSnapshot.transportMemoryTargetOwners;
            transportMemoryStorage = statsSnapshot.transportMemoryStorage;
            transportMemoryTargetStorage = statsSnapshot.transportMemoryTargetStorage;
            transportMemoryHistory = statsSnapshot.transportMemoryHistory;
            transportMemoryDebugReadback = statsSnapshot.transportMemoryDebugReadback;
            transportMemoryLastInvalidations = statsSnapshot.transportMemoryLastInvalidations;
            transportMemoryByteSizeFormulas = statsSnapshot.transportMemoryByteSizeFormulas;
            transportMemoryFeatureGates = statsSnapshot.transportMemoryFeatureGates;
            transportMemoryMigrationRisks = statsSnapshot.transportMemoryMigrationRisks;
            transportMemoryBytes = statsSnapshot.transportMemoryBytes;
            transportMemoryWidths = statsSnapshot.transportMemoryWidths;
            transportMemoryHeights = statsSnapshot.transportMemoryHeights;
            transportMemoryDepths = statsSnapshot.transportMemoryDepths;
            transportMemoryInvalidationCounts = statsSnapshot.transportMemoryInvalidationCounts;
            transportMemoryHotPath = statsSnapshot.transportMemoryHotPath;
            transportMemoryTemporaryShared = statsSnapshot.transportMemoryTemporaryShared;
            mlBackendMode = statsSnapshot.mlBackendMode;
            mlBackendRequested = statsSnapshot.mlBackendRequested;
            mlBackendActive = statsSnapshot.mlBackendActive;
            mlBackendNativeAccelerationAvailable =
                statsSnapshot.mlBackendNativeAccelerationAvailable;
            mlBackendPassCount = statsSnapshot.mlBackendPassCount;
            mlBackendProvider = statsSnapshot.mlBackendProvider;
            mlBackendProviderVersion = statsSnapshot.mlBackendProviderVersion;
            mlBackendTensorContract = statsSnapshot.mlBackendTensorContract;
            mlBackendFallbackReason = statsSnapshot.mlBackendFallbackReason;
            mlBackendConfidence = statsSnapshot.mlBackendConfidence;
            mlBackendBiasRisk = statsSnapshot.mlBackendBiasRisk;
            mlModelPackagePath = statsSnapshot.mlModelPackagePath;
            pipelineCompilationAvailable = statsSnapshot.pipelineCompilationAvailable;
            pipelineRuntimeSourceCompilation = statsSnapshot.pipelineRuntimeSourceCompilation;
            pipelineFunctionConstantsUsed = statsSnapshot.pipelineFunctionConstantsUsed;
            pipelineMetalBinaryArchiveUsed = statsSnapshot.pipelineMetalBinaryArchiveUsed;
            pipelineHarvestedCompilationReady = statsSnapshot.pipelineHarvestedCompilationReady;
            pipelineShaderSourceBytes = statsSnapshot.pipelineShaderSourceBytes;
            pipelineLibraryCompileMs = statsSnapshot.pipelineLibraryCompileMs;
            pipelineComputePipelineCount = statsSnapshot.pipelineComputePipelineCount;
            pipelineRenderPipelineCount = statsSnapshot.pipelineRenderPipelineCount;
            pipelineFailedPipelineCount = statsSnapshot.pipelineFailedPipelineCount;
            pipelineCacheStrategy = statsSnapshot.pipelineCacheStrategy;
            pipelineSpecializationStrategy = statsSnapshot.pipelineSpecializationStrategy;
            pipelineCompileKeys = statsSnapshot.pipelineCompileKeys;
            pipelineCompileFunctions = statsSnapshot.pipelineCompileFunctions;
            pipelineCompileStages = statsSnapshot.pipelineCompileStages;
            pipelineCompileMs = statsSnapshot.pipelineCompileMs;
            pipelineCompileSuccess = statsSnapshot.pipelineCompileSuccess;
            pipelineCompileRayTracingFlags = statsSnapshot.pipelineCompileRayTracingFlags;
            pipelineCompileSpecializations = statsSnapshot.pipelineCompileSpecializations;
        }
        accumulatedSamples += produced;
        accumulatedFrameTime += (frameEnd - frameStart);

        if (verbose) {
            CFAbsoluteTime now = frameEnd;
            if ((now - lastProgress) >= 0.5 || accumulatedSamples >= targetSamples) {
                uint32_t completedSamples = std::min(accumulatedSamples, targetSamples);
                double pct = (static_cast<double>(completedSamples) / targetSamples) * 100.0;
                double elapsed = now - renderStart;
                double etaSeconds = 0.0;
                if (completedSamples > 0u && elapsed > 0.0) {
                    etaSeconds = (elapsed / static_cast<double>(completedSamples)) *
                                 static_cast<double>(targetSamples - completedSamples);
                }
                std::cout << "Progress: " << completedSamples << "/" << targetSamples
                          << " spp (" << std::fixed << std::setprecision(1) << pct << "%)"
                          << " ETA " << FormatEta(etaSeconds) << "\r";
                std::cout.flush();
                lastProgress = now;
            }
        }

        if (statsMode) {
            if (renderer.captureRuntimeStats(statsSnapshot)) {
                const CFAbsoluteTime now = frameEnd;
                if ((now - lastStatsPrint) >= 1.0 || accumulatedSamples >= targetSamples) {
                    const double deltaSeconds = std::max(now - lastStatsPrint, 1.0e-6);
                    const uint64_t deltaRays =
                        (statsSnapshot.tracedRayCount >= lastStatsRayCount)
                            ? (statsSnapshot.tracedRayCount - lastStatsRayCount)
                            : statsSnapshot.tracedRayCount;
                    const double raysPerSecond =
                        static_cast<double>(deltaRays) / deltaSeconds;
                    if (verbose) {
                        std::cout << std::endl;
                    }
                    std::cout << "[Stats] spp=" << std::min(statsSnapshot.sampleCount, targetSamples)
                              << "/" << targetSamples
                              << " frameMs=" << std::fixed << std::setprecision(2)
                              << statsSnapshot.frameTimeMs
                              << " rays=" << statsSnapshot.tracedRayCount
                              << " raysPerSec=" << static_cast<uint64_t>(raysPerSecond + 0.5)
                              << " avgBounces=" << std::setprecision(3)
                              << statsSnapshot.averageBounceDepth
                              << std::endl;
                    lastStatsPrint = now;
                    lastStatsRayCount = statsSnapshot.tracedRayCount;
                }
            }
        }
    }

    if (verbose) {
        std::cout << std::endl;
    }

    CFAbsoluteTime renderEnd = CFAbsoluteTimeGetCurrent();

    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<float> linearRGB;
    const bool useRendererDenoisedOutput = rendererSettings.svgfDenoiseEnabled;
    const bool captured = useRendererDenoisedOutput
        ? renderer.captureAverageImage(linearRGB, width, height, nullptr)
        : renderer.captureRawImage(linearRGB, width, height, nullptr);
    if (!captured) {
        error = "Failed to capture rendered image";
        return false;
    }

    out.linearRGB = std::move(linearRGB);
    out.width = width;
    out.height = height;
    if (captureAuxBuffers) {
        uint32_t auxWidth = 0;
        uint32_t auxHeight = 0;
        if (!renderer.captureAuxiliaryImages(&out.albedoLinearRGB,
                                             &out.normalLinearRGB,
                                             auxWidth,
                                             auxHeight)) {
            if (verbose) {
                std::cerr << "[Headless] Warning: failed to capture auxiliary AOVs; continuing without guides."
                          << std::endl;
            }
            out.albedoLinearRGB.clear();
            out.normalLinearRGB.clear();
        } else if (auxWidth != width || auxHeight != height) {
            if (verbose) {
                std::cerr << "[Headless] Warning: auxiliary AOV dimensions do not match beauty buffer; "
                             "continuing without guides."
                          << std::endl;
            }
            out.albedoLinearRGB.clear();
            out.normalLinearRGB.clear();
        }
    }
    out.samples = accumulatedSamples;
    out.totalSeconds = renderEnd - renderStart;
    out.avgMsPerSample = (accumulatedSamples > 0)
                             ? (accumulatedFrameTime * 1000.0 / accumulatedSamples)
                             : 0.0;
    out.performance.available = (frameCount > 0u);
    out.performance.renderExecutionMode = renderExecutionMode;
    out.performance.wavefrontSchedulingPolicy = wavefrontSchedulingPolicy;
    out.performance.wavefrontCompactionEnabled = wavefrontCompactionEnabled;
    out.performance.frameCount = frameCount;
    out.performance.totalGpuSeconds = accumulatedGpuTimeMs / 1000.0;
    out.performance.avgGpuMsPerSample =
        (accumulatedSamples > 0u) ? (accumulatedGpuTimeMs / static_cast<double>(accumulatedSamples)) : 0.0;
    out.performance.totalCpuEncodeSeconds = accumulatedCpuEncodeMs / 1000.0;
    out.performance.avgCpuEncodeMsPerSample =
        (accumulatedSamples > 0u) ? (accumulatedCpuEncodeMs / static_cast<double>(accumulatedSamples)) : 0.0;
    out.performance.avgFrameGpuMs =
        (frameCount > 0u) ? (accumulatedGpuTimeMs / static_cast<double>(frameCount)) : 0.0;
    out.performance.avgFrameCpuEncodeMs =
        (frameCount > 0u) ? (accumulatedCpuEncodeMs / static_cast<double>(frameCount)) : 0.0;
    out.performance.gpuBucketTimingAvailable = gpuBucketTimingAvailable;
    out.performance.gpuBucketIntersectSeconds = accumulatedBucketIntersectMs / 1000.0;
    out.performance.gpuBucketShadeSeconds = accumulatedBucketShadeMs / 1000.0;
    out.performance.gpuBucketShadowQueueSeconds = accumulatedBucketShadowQueueMs / 1000.0;
    out.performance.gpuBucketFusedIntegrateSeconds = accumulatedBucketFusedIntegrateMs / 1000.0;
    out.performance.gpuBucketPresentationSeconds = accumulatedBucketPresentationMs / 1000.0;
    out.performance.currentAllocatedBytes = lastAllocatedBytes;
    out.performance.peakAllocatedBytes = peakAllocatedBytes;
    out.performance.recommendedMaxWorkingSetBytes = recommendedMaxWorkingSetBytes;
    out.performance.hardwareRaytracingAvailable = hardwareRaytracingAvailable;
    out.performance.hardwareRaytracingActive = hardwareRaytracingActive;
    out.performance.hardwareRaytracingFallbackActive = hardwareRaytracingFallbackActive;
    out.performance.hardwareRaytracingFallbackReason = hardwareRaytracingFallbackReason;
    out.performance.deviceName = std::move(deviceName);
    out.performance.deviceRecommendedMaxWorkingSetBytes = deviceRecommendedMaxWorkingSetBytes;
    out.performance.deviceSupportsRaytracing = deviceSupportsRaytracing;
    out.performance.deviceSupportsCounterSampling = deviceSupportsCounterSampling;
    out.performance.deviceSupportsArgumentBuffers = deviceSupportsArgumentBuffers;
    out.performance.deviceSupportsSparseResources = deviceSupportsSparseResources;
    out.performance.deviceSupportsMetalFX = deviceSupportsMetalFX;
    out.performance.deviceSupportsMLBackend = deviceSupportsMLBackend;
    out.performance.accelerationBuildPolicy = accelerationBuildPolicy;
    out.performance.accelerationIdentityStrategy = accelerationIdentityStrategy;
    out.performance.accelerationIntersectionFunctionStrategy =
        accelerationIntersectionFunctionStrategy;
    out.performance.accelerationIdentityStable = accelerationIdentityStable;
    out.performance.accelerationCustomIntersectionFunctionsSupported =
        accelerationCustomIntersectionFunctionsSupported;
    out.performance.accelerationCustomIntersectionFunctionsEnabled =
        accelerationCustomIntersectionFunctionsEnabled;
    out.performance.intersectionMode = intersectionMode;
    out.performance.accelerationPrimitiveCount = accelerationPrimitiveCount;
    out.performance.accelerationInstanceCount = accelerationInstanceCount;
    out.performance.accelerationBlasCount = accelerationBlasCount;
    out.performance.primaryRayCount = totalPrimaryRayCount;
    out.performance.pathSegmentCount = totalPathSegmentCount;
    out.performance.shadowRayCount = totalShadowRayCount;
    out.performance.tracedRayCount = totalTracedRayCount;
    out.performance.tracedRaysPerSecond =
        (out.totalSeconds > 0.0) ? (static_cast<double>(totalTracedRayCount) / out.totalSeconds) : 0.0;
    out.performance.tracedRaysPerGpuSecond =
        (accumulatedGpuTimeMs > 0.0)
            ? (static_cast<double>(totalTracedRayCount) / (accumulatedGpuTimeMs / 1000.0))
            : 0.0;
    out.performance.totalComputeDispatchCount = totalComputeDispatchCount;
    out.performance.restirDiInitializeDispatchCount =
        totalRestirDiInitializeDispatchCount;
    out.performance.restirDiInitialCandidateDispatchCount =
        totalRestirDiInitialCandidateDispatchCount;
    out.performance.restirDiTemporalReuseDispatchCount =
        totalRestirDiTemporalReuseDispatchCount;
    out.performance.restirDiSpatialReuseDispatchCount =
        totalRestirDiSpatialReuseDispatchCount;
    out.performance.restirDiVisibilityDispatchCount =
        totalRestirDiVisibilityDispatchCount;
    out.performance.restirDiApplyLightingDispatchCount =
        totalRestirDiApplyLightingDispatchCount;
    out.performance.restirDiDebugAovDispatchCount =
        totalRestirDiDebugAovDispatchCount;
    out.performance.integrationDispatchCount = totalIntegrationDispatchCount;
    out.performance.presentationDispatchCount = totalPresentationDispatchCount;
    out.performance.wavefrontResetDispatchCount = totalWavefrontResetDispatchCount;
    out.performance.wavefrontGenerateDispatchCount = totalWavefrontGenerateDispatchCount;
    out.performance.wavefrontBuildDispatchArgsDispatchCount =
        totalWavefrontBuildDispatchArgsDispatchCount;
    out.performance.wavefrontCompactQueueDispatchCount =
        totalWavefrontCompactQueueDispatchCount;
    out.performance.wavefrontPrepareDispatchCount = totalWavefrontPrepareDispatchCount;
    out.performance.wavefrontIntersectDispatchCount = totalWavefrontIntersectDispatchCount;
    out.performance.wavefrontShadeDispatchCount = totalWavefrontShadeDispatchCount;
    out.performance.wavefrontShadowDispatchCount = totalWavefrontShadowDispatchCount;
    out.performance.wavefrontSwapDispatchCount = totalWavefrontSwapDispatchCount;
    out.performance.wavefrontAccumulateDispatchCount = totalWavefrontAccumulateDispatchCount;
    out.performance.wavefrontQueueBytesTotal = wavefrontQueueBytesTotal;
    out.performance.wavefrontBudgetBytes = wavefrontBudgetBytes;
    out.performance.wavefrontBudgetUsagePercent = wavefrontBudgetUsagePercent;
    out.performance.restirDiResourceBytesTotal = restirDiResourceBytesTotal;
    out.performance.restirDiReservoirBytesEach = restirDiReservoirBytesEach;
    out.performance.restirDiVisibilityBytes = restirDiVisibilityBytes;
    out.performance.restirDiDebugBytes = restirDiDebugBytes;
    out.performance.transportMemoryBytesTotal = transportMemoryBytesTotal;
    out.performance.transportMemoryBytesPrivate = transportMemoryBytesPrivate;
    out.performance.transportMemoryBytesShared = transportMemoryBytesShared;
    out.performance.transportMemoryBytesStaging = transportMemoryBytesStaging;
    out.performance.transportMemoryRecordCount = transportMemoryRecordCount;
    out.performance.transportMemoryInvalidatedRecordCount =
        transportMemoryInvalidatedRecordCount;
    out.performance.wavefrontPrimaryGeneratedCount = totalWavefrontPrimaryGeneratedCount;
    out.performance.wavefrontIntersectProcessedCount = totalWavefrontIntersectProcessedCount;
    out.performance.wavefrontShadeProcessedCount = totalWavefrontShadeProcessedCount;
    out.performance.wavefrontShadowProcessedCount = totalWavefrontShadowProcessedCount;
    out.performance.wavefrontNextRayEnqueueCount = totalWavefrontNextRayEnqueueCount;
    out.performance.wavefrontShadowRayEnqueueCount = totalWavefrontShadowRayEnqueueCount;
    for (uint32_t i = 0u; i < HeadlessRenderOutput::kWavefrontQueueMetricCount; ++i) {
        out.performance.wavefrontQueueActiveCounts[i] = wavefrontQueueActiveCounts[i];
        out.performance.wavefrontQueuePeakActiveCounts[i] = wavefrontQueuePeakActiveCounts[i];
        out.performance.wavefrontQueueCapacities[i] = wavefrontQueueCapacities[i];
        out.performance.wavefrontQueueOverflowCounts[i] = wavefrontQueueOverflowCounts[i];
        out.performance.wavefrontQueueOccupancyRatios[i] = wavefrontQueueOccupancyRatios[i];
        out.performance.wavefrontQueuePeakOccupancyRatios[i] =
            wavefrontQueuePeakOccupancyRatios[i];
    }
    out.performance.wavefrontQueueCompactionCostMs = wavefrontQueueCompactionCostMs;
    out.performance.wavefrontIntersectDispatchMode = std::move(wavefrontIntersectDispatchMode);
    out.performance.wavefrontShadeDispatchMode = std::move(wavefrontShadeDispatchMode);
    out.performance.wavefrontShadowDispatchMode = std::move(wavefrontShadowDispatchMode);
    out.performance.wavefrontQueueStoragePolicySummary =
        std::move(wavefrontQueueStoragePolicySummary);
    out.performance.wavefrontM1BridgeStatus = std::move(wavefrontM1BridgeStatus);
    out.performance.passGraphVersion = passGraphVersion;
    out.performance.passGraphCompiled = passGraphCompiled;
    out.performance.passGraphCompiledExecutablePassCount =
        passGraphCompiledExecutablePassCount;
    out.performance.passGraphCompiledSkippedPassCount =
        passGraphCompiledSkippedPassCount;
    out.performance.passGraphCompiledUsedResourceCount =
        passGraphCompiledUsedResourceCount;
    out.performance.passGraphAliasableTransientResourceCount =
        passGraphAliasableTransientResourceCount;
    out.performance.passGraphTransientAliasPoolCount =
        passGraphTransientAliasPoolCount;
    out.performance.passGraphResourceTransitionCount =
        passGraphResourceTransitionCount;
    out.performance.passGraphRequiredBarrierCount =
        passGraphRequiredBarrierCount;
    out.performance.passGraphWriteTransitionCount =
        passGraphWriteTransitionCount;
    out.performance.historyResetReason = historyResetReason;
    out.performance.passGraphCompileWarnings = std::move(passGraphCompileWarnings);
    out.performance.passGraphCompileErrors = std::move(passGraphCompileErrors);
    out.performance.passGraphExecutedPasses = std::move(passGraphExecutedPasses);
    out.performance.passGraphSkippedPasses = std::move(passGraphSkippedPasses);
    out.performance.passGraphFailedPasses = std::move(passGraphFailedPasses);
    out.performance.passGraphNames = std::move(passGraphNames);
    out.performance.passGraphTypes = std::move(passGraphTypes);
    out.performance.passGraphExecutions = std::move(passGraphExecutions);
    out.performance.passGraphPipelines = std::move(passGraphPipelines);
    out.performance.passGraphDispatches = std::move(passGraphDispatches);
    out.performance.passGraphDebugLabels = std::move(passGraphDebugLabels);
    out.performance.passGraphTimingLabels = std::move(passGraphTimingLabels);
    out.performance.passGraphDebugPolicies = std::move(passGraphDebugPolicies);
    out.performance.passGraphTimingPolicies = std::move(passGraphTimingPolicies);
    out.performance.passGraphResourceNames = std::move(passGraphResourceNames);
    out.performance.passGraphResourceLifetimes = std::move(passGraphResourceLifetimes);
    out.performance.passGraphResourceStorage = std::move(passGraphResourceStorage);
    out.performance.passGraphResourceTargetStorage = std::move(passGraphResourceTargetStorage);
    out.performance.passGraphResourceSizeFormulas = std::move(passGraphResourceSizeFormulas);
    out.performance.passGraphResourceDebugLabels = std::move(passGraphResourceDebugLabels);
    out.performance.passGraphResourceCurrentOwners = std::move(passGraphResourceCurrentOwners);
    out.performance.passGraphResourceTargetOwners = std::move(passGraphResourceTargetOwners);
    out.performance.passGraphResourceFeatureGates = std::move(passGraphResourceFeatureGates);
    out.performance.passGraphResourceDebugReadbackPolicies =
        std::move(passGraphResourceDebugReadbackPolicies);
    out.performance.passGraphResourceMigrationRisks = std::move(passGraphResourceMigrationRisks);
    out.performance.passGraphResourceFirstPasses = std::move(passGraphResourceFirstPasses);
    out.performance.passGraphResourceLastPasses = std::move(passGraphResourceLastPasses);
    out.performance.passGraphResourceAliasPools = std::move(passGraphResourceAliasPools);
    out.performance.passGraphResourceUsed = std::move(passGraphResourceUsed);
    out.performance.passGraphResourceAliasable = std::move(passGraphResourceAliasable);
    out.performance.passGraphResourceHotPath = std::move(passGraphResourceHotPath);
    out.performance.passGraphResourceTemporaryShared = std::move(passGraphResourceTemporaryShared);
    out.performance.passGraphTransitionPassIndices = std::move(passGraphTransitionPassIndices);
    out.performance.passGraphTransitionPassNames = std::move(passGraphTransitionPassNames);
    out.performance.passGraphTransitionResourceNames = std::move(passGraphTransitionResourceNames);
    out.performance.passGraphTransitionPreviousAccesses =
        std::move(passGraphTransitionPreviousAccesses);
    out.performance.passGraphTransitionAccesses = std::move(passGraphTransitionAccesses);
    out.performance.passGraphTransitionFirstUse = std::move(passGraphTransitionFirstUse);
    out.performance.passGraphTransitionRequiresBarrier =
        std::move(passGraphTransitionRequiresBarrier);
    out.performance.passGraphTransitionWritesResource =
        std::move(passGraphTransitionWritesResource);
    out.performance.transportMemoryNames = std::move(transportMemoryNames);
    out.performance.transportMemoryKinds = std::move(transportMemoryKinds);
    out.performance.transportMemoryOwners = std::move(transportMemoryOwners);
    out.performance.transportMemoryTargetOwners = std::move(transportMemoryTargetOwners);
    out.performance.transportMemoryStorage = std::move(transportMemoryStorage);
    out.performance.transportMemoryTargetStorage = std::move(transportMemoryTargetStorage);
    out.performance.transportMemoryHistory = std::move(transportMemoryHistory);
    out.performance.transportMemoryDebugReadback = std::move(transportMemoryDebugReadback);
    out.performance.transportMemoryLastInvalidations =
        std::move(transportMemoryLastInvalidations);
    out.performance.transportMemoryByteSizeFormulas = std::move(transportMemoryByteSizeFormulas);
    out.performance.transportMemoryFeatureGates = std::move(transportMemoryFeatureGates);
    out.performance.transportMemoryMigrationRisks = std::move(transportMemoryMigrationRisks);
    out.performance.transportMemoryBytes = std::move(transportMemoryBytes);
    out.performance.transportMemoryWidths = std::move(transportMemoryWidths);
    out.performance.transportMemoryHeights = std::move(transportMemoryHeights);
    out.performance.transportMemoryDepths = std::move(transportMemoryDepths);
    out.performance.transportMemoryInvalidationCounts =
        std::move(transportMemoryInvalidationCounts);
    out.performance.transportMemoryHotPath = std::move(transportMemoryHotPath);
    out.performance.transportMemoryTemporaryShared = std::move(transportMemoryTemporaryShared);
    out.performance.mlBackendMode = mlBackendMode;
    out.performance.mlBackendRequested = mlBackendRequested;
    out.performance.mlBackendActive = mlBackendActive;
    out.performance.mlBackendNativeAccelerationAvailable = mlBackendNativeAccelerationAvailable;
    out.performance.mlBackendPassCount = mlBackendPassCount;
    out.performance.mlBackendProvider = std::move(mlBackendProvider);
    out.performance.mlBackendProviderVersion = std::move(mlBackendProviderVersion);
    out.performance.mlBackendTensorContract = std::move(mlBackendTensorContract);
    out.performance.mlBackendFallbackReason = std::move(mlBackendFallbackReason);
    out.performance.mlBackendConfidence = mlBackendConfidence;
    out.performance.mlBackendBiasRisk = mlBackendBiasRisk;
    out.performance.mlModelPackagePath = std::move(mlModelPackagePath);
    out.performance.pipelineCompilationAvailable = pipelineCompilationAvailable;
    out.performance.pipelineRuntimeSourceCompilation = pipelineRuntimeSourceCompilation;
    out.performance.pipelineFunctionConstantsUsed = pipelineFunctionConstantsUsed;
    out.performance.pipelineMetalBinaryArchiveUsed = pipelineMetalBinaryArchiveUsed;
    out.performance.pipelineHarvestedCompilationReady = pipelineHarvestedCompilationReady;
    out.performance.pipelineShaderSourceBytes = pipelineShaderSourceBytes;
    out.performance.pipelineLibraryCompileMs = pipelineLibraryCompileMs;
    out.performance.pipelineComputePipelineCount = pipelineComputePipelineCount;
    out.performance.pipelineRenderPipelineCount = pipelineRenderPipelineCount;
    out.performance.pipelineFailedPipelineCount = pipelineFailedPipelineCount;
    out.performance.pipelineCacheStrategy = std::move(pipelineCacheStrategy);
    out.performance.pipelineSpecializationStrategy = std::move(pipelineSpecializationStrategy);
    out.performance.pipelineCompileKeys = std::move(pipelineCompileKeys);
    out.performance.pipelineCompileFunctions = std::move(pipelineCompileFunctions);
    out.performance.pipelineCompileStages = std::move(pipelineCompileStages);
    out.performance.pipelineCompileMs = std::move(pipelineCompileMs);
    out.performance.pipelineCompileSuccess = std::move(pipelineCompileSuccess);
    out.performance.pipelineCompileRayTracingFlags = std::move(pipelineCompileRayTracingFlags);
    out.performance.pipelineCompileSpecializations = std::move(pipelineCompileSpecializations);
    MetalRenderer::PbrMetricsSnapshot pbrSnapshot{};
    if (renderer.capturePbrMetrics(pbrSnapshot)) {
        out.pbrMetrics.available = pbrSnapshot.available;
        out.pbrMetrics.throughputNanInfCount = pbrSnapshot.throughputNanInfCount;
        out.pbrMetrics.radianceNanInfCount = pbrSnapshot.radianceNanInfCount;
        out.pbrMetrics.clampEventCount = pbrSnapshot.clampEventCount;
        out.pbrMetrics.specNeeEnvAddedCount = pbrSnapshot.specNeeEnvAddedCount;
        out.pbrMetrics.specNeeRectAddedCount = pbrSnapshot.specNeeRectAddedCount;
        out.pbrMetrics.mneeEligibleCount = pbrSnapshot.mneeEligibleCount;
        out.pbrMetrics.mneeEnvAttemptCount = pbrSnapshot.mneeEnvAttemptCount;
        out.pbrMetrics.mneeEnvAddedCount = pbrSnapshot.mneeEnvAddedCount;
        out.pbrMetrics.mneeEnvChainUnsupportedCount = pbrSnapshot.mneeEnvChainUnsupportedCount;
        out.pbrMetrics.mneeEnvChainNonDeltaBlockCount = pbrSnapshot.mneeEnvChainNonDeltaBlockCount;
        out.pbrMetrics.mneeEnvChainSampleRejectCount = pbrSnapshot.mneeEnvChainSampleRejectCount;
        out.pbrMetrics.mneeEnvChainWeightRejectCount = pbrSnapshot.mneeEnvChainWeightRejectCount;
        out.pbrMetrics.mneeEnvChainMaxBounceCount = pbrSnapshot.mneeEnvChainMaxBounceCount;
        out.pbrMetrics.mneeRectAddedCount = pbrSnapshot.mneeRectAddedCount;
        out.pbrMetrics.mneeContributionCount = pbrSnapshot.mneeContributionCount;
        out.pbrMetrics.restirGiCandidateCount = pbrSnapshot.restirGiCandidateCount;
        out.pbrMetrics.restirGiReservoirUpdateCount = pbrSnapshot.restirGiReservoirUpdateCount;
        out.pbrMetrics.restirGiAcceptCount = pbrSnapshot.restirGiAcceptCount;
        out.pbrMetrics.restirGiRejectCount = pbrSnapshot.restirGiRejectCount;
        out.pbrMetrics.restirGiFallbackCount = pbrSnapshot.restirGiFallbackCount;
        out.pbrMetrics.pathGuidingCandidateCount = pbrSnapshot.pathGuidingCandidateCount;
        out.pbrMetrics.pathGuidingUsedCount = pbrSnapshot.pathGuidingUsedCount;
        out.pbrMetrics.pathGuidingFallbackCount = pbrSnapshot.pathGuidingFallbackCount;
        out.pbrMetrics.pathGuidingInvalidCount = pbrSnapshot.pathGuidingInvalidCount;
        out.pbrMetrics.restirPtCandidateCount = pbrSnapshot.restirPtCandidateCount;
        out.pbrMetrics.restirPtReservoirUpdateCount = pbrSnapshot.restirPtReservoirUpdateCount;
        out.pbrMetrics.restirPtRejectedInvalidCount = pbrSnapshot.restirPtRejectedInvalidCount;
        out.pbrMetrics.restirPtRejectedUnsupportedCount = pbrSnapshot.restirPtRejectedUnsupportedCount;
        out.pbrMetrics.restirPtDebugRecordCount = pbrSnapshot.restirPtDebugRecordCount;
        out.pbrMetrics.restirPtReuseCandidateCount = pbrSnapshot.restirPtReuseCandidateCount;
        out.pbrMetrics.restirPtReuseAppliedCount = pbrSnapshot.restirPtReuseAppliedCount;
        out.pbrMetrics.restirPtReuseRejectedInvalidCount = pbrSnapshot.restirPtReuseRejectedInvalidCount;
        out.pbrMetrics.restirPtReuseFallbackCount = pbrSnapshot.restirPtReuseFallbackCount;
        out.pbrMetrics.radianceCacheQueryCount = pbrSnapshot.radianceCacheQueryCount;
        out.pbrMetrics.radianceCacheHitCount = pbrSnapshot.radianceCacheHitCount;
        out.pbrMetrics.radianceCacheMissCount = pbrSnapshot.radianceCacheMissCount;
        out.pbrMetrics.radianceCacheTrainCount = pbrSnapshot.radianceCacheTrainCount;
        out.pbrMetrics.radianceCacheFallbackCount = pbrSnapshot.radianceCacheFallbackCount;
        out.pbrMetrics.radianceCacheInvalidCount = pbrSnapshot.radianceCacheInvalidCount;
        out.pbrMetrics.causticEyeVertexCount = pbrSnapshot.causticEyeVertexCount;
        out.pbrMetrics.causticLightVertexCount = pbrSnapshot.causticLightVertexCount;
        out.pbrMetrics.causticConnectionCandidateCount = pbrSnapshot.causticConnectionCandidateCount;
        out.pbrMetrics.causticDeltaChainCount = pbrSnapshot.causticDeltaChainCount;
        out.pbrMetrics.causticDebugClassificationCount = pbrSnapshot.causticDebugClassificationCount;
        out.pbrMetrics.volumeTransmittanceQueryCount = pbrSnapshot.volumeTransmittanceQueryCount;
        out.pbrMetrics.volumeScatterEventCount = pbrSnapshot.volumeScatterEventCount;
        out.pbrMetrics.volumeNeeRayCount = pbrSnapshot.volumeNeeRayCount;
        out.pbrMetrics.volumePhaseSampleCount = pbrSnapshot.volumePhaseSampleCount;
        out.pbrMetrics.volumeMediumBoundaryEventCount = pbrSnapshot.volumeMediumBoundaryEventCount;
        out.pbrMetrics.volumeFallbackCount = pbrSnapshot.volumeFallbackCount;
        out.pbrMetrics.spectralWavelengthSampleCount = pbrSnapshot.spectralWavelengthSampleCount;
        out.pbrMetrics.spectralMaterialLookupCount = pbrSnapshot.spectralMaterialLookupCount;
        out.pbrMetrics.spectralDispersionEventCount = pbrSnapshot.spectralDispersionEventCount;
        out.pbrMetrics.spectralFallbackCount = pbrSnapshot.spectralFallbackCount;
        out.pbrMetrics.maxThroughputByBounce = std::move(pbrSnapshot.maxThroughputByBounce);
    }
    renderer.captureMaterialMrDebugDump(out.materialMrDebug);
    renderer.captureDirectLightAuditDump(out.directLightAudit);
    return true;
}
