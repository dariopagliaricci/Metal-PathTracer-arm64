#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "MetalRendererTypes.h"
#include "headless/IHeadlessRenderer.h"

namespace PathTracer {
struct RenderSettings;
class SceneResources;
}

class MetalRenderer {
public:
    struct RuntimeStatsSnapshot {
        bool available = false;
        uint32_t sampleCount = 0;
        double frameTimeMs = 0.0;
        uint64_t pathSegmentCount = 0;
        uint64_t shadowRayCount = 0;
        uint64_t tracedRayCount = 0;
        double averageBounceDepth = 0.0;
        double gpuTimeMs = 0.0;
        double cpuEncodeMs = 0.0;
        double drawableWaitMs = 0.0;
        bool gpuBucketTimingAvailable = false;
        double gpuBucketIntersectMs = 0.0;
        double gpuBucketShadeMs = 0.0;
        double gpuBucketShadowQueueMs = 0.0;
        double gpuBucketFusedIntegrateMs = 0.0;
        double gpuBucketPresentationMs = 0.0;
        bool hardwareRaytracingActive = false;
        bool hardwareRaytracingAvailable = false;
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
        uint32_t renderExecutionMode = 0;
        uint32_t wavefrontSchedulingPolicy = 0;
        uint32_t wavefrontCompactionEnabled = 0;
        uint64_t currentAllocatedBytes = 0;
        uint64_t recommendedMaxWorkingSetBytes = 0;
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
        uint64_t transportMemoryBytesTotal = 0;
        uint64_t transportMemoryBytesPrivate = 0;
        uint64_t transportMemoryBytesShared = 0;
        uint64_t transportMemoryBytesStaging = 0;
        uint32_t transportMemoryRecordCount = 0;
        uint32_t transportMemoryInvalidatedRecordCount = 0;
        uint64_t primaryRayCount = 0;
        uint64_t hardwareRayCount = 0;
        uint64_t wavefrontPrimaryGeneratedCount = 0;
        uint64_t wavefrontIntersectProcessedCount = 0;
        uint64_t wavefrontShadeProcessedCount = 0;
        uint64_t wavefrontShadowProcessedCount = 0;
        uint64_t wavefrontNextRayEnqueueCount = 0;
        uint64_t wavefrontShadowRayEnqueueCount = 0;
        uint32_t wavefrontQueueActiveCounts[12] = {};
        uint32_t wavefrontQueuePeakActiveCounts[12] = {};
        uint32_t wavefrontQueueCapacities[12] = {};
        uint32_t wavefrontQueueOverflowCounts[12] = {};
        double wavefrontQueueOccupancyRatios[12] = {};
        double wavefrontQueuePeakOccupancyRatios[12] = {};
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
    };

    struct PbrMetricsSnapshot {
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
        std::vector<float> maxThroughputByBounce;
    };

    MetalRenderer();
    ~MetalRenderer();

    bool init(void* windowHandle, const MetalRendererOptions& options);
    void drawFrame();
    void resize(int width, int height);
    void resetAccumulation();
    void setTonemapMode(uint32_t mode);
    void shutdown();

    // Headless rendering support
    bool exportToPPM(const char* filepath);

    // Scene control
    bool setScene(const char* identifier);
    bool loadSceneFromPath(const char* path);
    bool useSceneResources(PathTracer::SceneResources& resources, const char* identifier = nullptr);
    std::vector<std::string> sceneIdentifiers() const;

    // Settings accessors
    PathTracer::RenderSettings settings() const;
    void applySettings(const PathTracer::RenderSettings& settings, bool resetAccumulation = true);
    void setSamplesPerFrame(uint32_t samples);
    uint32_t sampleCount() const;

    // Presentation Mode controls
    void setPresentationEnabled(bool enabled);
    bool isPresentationEnabled() const;
    void togglePresentationUIPanels();

    bool captureAverageImage(std::vector<float>& outLinearRGB,
                             uint32_t& width,
                             uint32_t& height,
                             std::vector<float>* outSampleCounts = nullptr);
    bool captureRawImage(std::vector<float>& outLinearRGB,
                         uint32_t& width,
                         uint32_t& height,
                         std::vector<float>* outSampleCounts = nullptr);
    bool captureAuxiliaryImages(std::vector<float>* outAlbedoRGB,
                                std::vector<float>* outNormalRGB,
                                uint32_t& width,
                                uint32_t& height);
    bool captureRuntimeStats(RuntimeStatsSnapshot& outStats) const;
    bool capturePbrMetrics(PbrMetricsSnapshot& outMetrics) const;
    bool captureMaterialMrDebugDump(HeadlessRenderOutput::MaterialMrDebugDump& outDump) const;
    bool captureDirectLightAuditDump(HeadlessRenderOutput::DirectLightAuditDump& outDump) const;

private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};
