#pragma once

// ⚠️ INTERNAL HEADER - Subject to change without notice
// This header is part of PathTracer's internal implementation.
// Use only the public API from MetalRenderer.h

#include "renderer/MetalHandles.h"
#include "MetalShaderTypes.h"
#include "PerformanceStats.h"
#include "renderer/RenderPassGraph.h"
#include "renderer/TransportMemoryAllocator.h"
#include "renderer/TransportMemoryRegistry.h"
#include "RenderSettings.h"
#include <array>
#include <string>
#include <vector>

namespace PathTracer {

// Forward declarations
class MetalContext;
class Accumulation;
class Pipelines;
class SceneResources;
class UIOverlay;
class DenoiserContext;

struct MlReconstructionTinyModel {
    uint32_t magic = 0x4d4c5231u;
    uint32_t version = 1u;
    uint32_t featureCount = 8u;
    uint32_t outputCount = 4u;
    float confidenceBias = 0.0f;
    float varianceScale = 0.05f;
    uint32_t flags = 0u;
    uint32_t reserved = 0u;
    std::array<float, 36> weights{};
};

/// Result from encoding a frame
struct FrameResult {
    double gpuTimeMs = 0.0;
    double avgNodesVisited = 0.0;
    double avgLeafPrimTests = 0.0;
    double shadowRayEarlyExitPct = 0.0;
    double bothChildrenVisitedPct = 0.0;
    uint32_t samplesDispatched = 0;
    uint32_t totalComputeDispatchCount = 0;
    uint32_t restirDiInitializeDispatchCount = 0;
    uint32_t restirDiInitialCandidateDispatchCount = 0;
    uint32_t restirDiTemporalReuseDispatchCount = 0;
    uint32_t restirDiSpatialReuseDispatchCount = 0;
    uint32_t restirDiVisibilityDispatchCount = 0;
    uint32_t restirDiApplyLightingDispatchCount = 0;
    uint32_t restirDiDebugAovDispatchCount = 0;
    uint32_t integrationDispatchCount = 0;
    uint32_t presentationDispatchCount = 0;
    uint32_t wavefrontResetDispatchCount = 0;
    uint32_t wavefrontGenerateDispatchCount = 0;
    uint32_t wavefrontBuildDispatchArgsDispatchCount = 0;
    uint32_t wavefrontCompactQueueDispatchCount = 0;
    uint32_t wavefrontPrepareDispatchCount = 0;
    uint32_t wavefrontIntersectDispatchCount = 0;
    uint32_t wavefrontShadeDispatchCount = 0;
    uint32_t wavefrontShadowDispatchCount = 0;
    uint32_t wavefrontSwapDispatchCount = 0;
    uint32_t wavefrontAccumulateDispatchCount = 0;
    RenderPassGraphDesc passGraph;
    RenderPassExecutionResult passGraphExecution;
    HistoryResetReason historyResetReason = HistoryResetReason::None;
};

/// Orchestrates GPU command encoding for path tracing
/// Handles integration, presentation, and display passes
class RenderLoop {
public:
    RenderLoop() = default;
    ~RenderLoop() = default;
    
    // Non-copyable
    RenderLoop(const RenderLoop&) = delete;
    RenderLoop& operator=(const RenderLoop&) = delete;
    
    /// Initialize buffers and denoiser
    /// @param context Metal context
    /// @param denoiser Optional DenoiserContext for OIDN (can be nullptr)
    /// @return true if initialization succeeded
    bool initialize(const MetalContext& context, DenoiserContext* denoiser = nullptr);
    
    /// Encode a complete frame
    /// @return Frame statistics
    FrameResult encodeFrame(MTLCommandBufferHandle commandBuffer,
                            MTLRenderPassDescriptorHandle renderPassDescriptor,
                            MTLCAMetalDrawableHandle drawable,
                            const MetalContext& context,
                            Accumulation& accumulation,
                            const Pipelines& pipelines,
                            SceneResources& scene,
                            UIOverlay& overlay,
                            const RenderSettings& settings);

    /// Encode only the display pass and ImGui overlay.
    /// Used by the interactive window path when path tracing is paused/throttled.
    void encodeDisplayOnlyFrame(MTLCommandBufferHandle commandBuffer,
                                MTLRenderPassDescriptorHandle renderPassDescriptor,
                                MTLCAMetalDrawableHandle drawable,
                                const Accumulation& accumulation,
                                const Pipelines& pipelines,
                                UIOverlay& overlay,
                                const RenderSettings& settings);
    
    /// Cleanup resources
    void shutdown();

    /// Access the statistics buffer used for GPU counters
    MTLBufferHandle statsBuffer() const { return m_statsBuffer; }
    MTLBufferHandle debugBuffer() const { return m_debugBuffer; }
    MTLBufferHandle queueCountersBuffer() const { return m_wavefront.queueCounters; }
    bool hasDenoisedOutput() const { return m_hasDenoisedOutput; }
    uint64_t wavefrontQueueBytesActive() const { return m_wavefront.activeBytes; }
    uint64_t wavefrontQueueBytesNext() const { return m_wavefront.nextBytes; }
    uint64_t wavefrontQueueBytesHit() const { return m_wavefront.hitBytes; }
    uint64_t wavefrontQueueBytesShadow() const { return m_wavefront.shadowBytes; }
    uint64_t wavefrontQueueBytesPathState() const { return m_wavefront.pathStateBytes; }
    uint64_t wavefrontQueueBytesIndirectArgs() const { return m_wavefront.indirectDispatchArgsBytes; }
    uint64_t wavefrontQueueBytesTotal() const { return m_wavefront.totalBytes; }
    uint64_t wavefrontBudgetBytes() const { return m_wavefront.budgetBytes; }
    uint64_t restirDiResourceBytesTotal() const {
        return m_restirState.restirDiReservoirBytes * 4u +
               m_restirState.restirDiVisibilityBytes +
               m_restirState.restirDiNeighborPatternBytes +
               m_restirState.restirDiDebugBytes;
    }
    uint64_t restirDiReservoirBytesEach() const { return m_restirState.restirDiReservoirBytes; }
    uint64_t restirDiVisibilityBytes() const { return m_restirState.restirDiVisibilityBytes; }
    uint64_t restirDiDebugBytes() const { return m_restirState.restirDiDebugBytes; }
    bool restirDiHardwareTraceActive() const { return m_restirDiHardwareTraceActiveThisFrame; }
    const std::string& restirDiPathStatus() const { return m_restirDiPathStatus; }
    bool mlBackendActive() const { return m_mlState.activeThisFrame; }
    bool mlBackendNativeAccelerationAvailable() const { return m_mlState.modelValid; }
    uint64_t mlBackendDispatchCount() const { return m_mlState.dispatchCount; }
    uint64_t mlBackendModelBytes() const { return m_mlState.modelBytes; }
    float mlBackendConfidence() const { return m_mlState.lastConfidence; }
    float mlBackendBiasRisk() const { return m_mlState.biasRisk; }
    const std::string& mlBackendProvider() const { return m_mlState.provider; }
    const std::string& mlBackendProviderVersion() const { return m_mlState.providerVersion; }
    const std::string& mlBackendTensorContract() const { return m_mlState.tensorContract; }
    const std::string& mlBackendFallbackReason() const { return m_mlState.fallbackReason; }
    uint64_t transportMemoryBytesTotal() const { return m_transportMemory.totalBytes(); }
    uint64_t transportMemoryBytesPrivate() const { return m_transportMemory.privateBytes(); }
    uint64_t transportMemoryBytesShared() const { return m_transportMemory.sharedBytes(); }
    uint64_t transportMemoryBytesStaging() const { return m_transportMemory.stagingBytes(); }
    uint32_t transportMemoryRecordCount() const { return m_transportMemory.recordCount(); }
    uint32_t transportMemoryAllocatorBackedRecordCount() const {
        return m_transportMemory.allocatorBackedRecordCount();
    }
    uint32_t transportMemoryFrameGraphBackedRecordCount() const {
        return m_transportMemory.frameGraphBackedRecordCount();
    }
    uint32_t transportMemoryOwnershipGapRecordCount() const {
        return m_transportMemory.ownershipGapRecordCount();
    }
    uint32_t transportMemoryInvalidatedRecordCount() const {
        return m_transportMemory.invalidatedRecordCount();
    }
    HistoryResetReason transportMemoryLastInvalidationReason() const {
        return m_transportMemory.lastInvalidationReason();
    }
    const TransportMemoryRegistry& transportMemoryRegistry() const { return m_transportMemory; }
    bool gpuTimingAvailable() const { return m_gpuTiming.supported && !m_gpuTiming.ranges.empty(); }
    double gpuTimingIntersectMs() const { return m_gpuTiming.lastIntersectMs; }
    double gpuTimingShadeMs() const { return m_gpuTiming.lastShadeMs; }
    double gpuTimingShadowQueueMs() const { return m_gpuTiming.lastShadowQueueMs; }
    double gpuTimingCompactionMs() const { return m_gpuTiming.lastCompactionMs; }
    double gpuTimingFusedIntegrateMs() const { return m_gpuTiming.lastFusedIntegrateMs; }
    double gpuTimingPresentationMs() const { return m_gpuTiming.lastPresentationMs; }
    void resolveGpuTiming(double totalGpuTimeMs);
    void resetHistory(HistoryResetReason reason, bool preserveSvgfHistory = false);
    const RenderPassGraphDesc& lastPassGraph() const { return m_lastPassGraph; }
    const RenderPassExecutionResult& lastPassGraphExecution() const { return m_lastPassGraphExecution; }
    HistoryResetReason lastFrameHistoryResetReason() const { return m_lastFrameHistoryResetReason; }
    
    /// Megakernel integrate pipeline of the last dispatch, for logs and render metadata:
    /// "not_used", "software", "hardware_general" or "hardware_default_features".
    const char* megakernelIntegratePipeline() const {
        switch (m_lastIntegrateVariant) {
            case 0: return "software";
            case 1: return "hardware_general";
            case 2: return "hardware_default_features";
            default: return "not_used";
        }
    }

    /// Direct-light mode the wavefront could not run as requested, for logs and render metadata: "none",
    /// "ris_spatial ran as plain ris" (the wavefront has no same-frame spatial reuse), or the explicit restir_di stage
    /// replaced by the megakernel's fused direct light when the frame fell back to the megakernel.
    const char* wavefrontDirectLightFallback() const {
        if (m_wavefrontExplicitRestirDiRanFused) {
            return "explicit restir_di stage ran as the megakernel's fused direct light";
        }
        return m_wavefrontRisSpatialRanAsRis ? "ris_spatial ran as plain ris" : "none";
    }

    /// Why the wavefront executor handed the last frame to the megakernel, for logs and render metadata: "none", or
    /// the reason (missing pipelines, a failed allocation, or the PT_WAVEFRONT_FORCE_MEGAKERNEL_FALLBACK test hook).
    const char* wavefrontMegakernelFallback() const { return m_wavefrontMegakernelFallback; }

private:
    struct WavefrontResources {
        MTLBufferHandle activeRays[2] = {nullptr, nullptr};
        MTLBufferHandle hitRecords = nullptr;
        MTLBufferHandle shadowRays = nullptr;
        MTLBufferHandle shadowIndices = nullptr;
        MTLBufferHandle pathStates = nullptr;
        MTLBufferHandle queueCounters = nullptr;
        MTLBufferHandle cacheQueryItems = nullptr;
        MTLBufferHandle cacheTrainRecords = nullptr;
        MTLBufferHandle resolvedSurfaces = nullptr;
        MTLBufferHandle indirectDispatchArgs = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t capacity = 0;
        uint32_t shadowCapacity = 0;
        bool cacheQueuesEnabled = false;
        bool cacheTrainRecordsNeedClear = false;
        bool resolvedSurfacesEnabled = false;
        uint64_t activeBytes = 0;
        uint64_t nextBytes = 0;
        uint64_t hitBytes = 0;
        uint64_t shadowBytes = 0;
        uint64_t shadowIndexBytes = 0;
        uint64_t pathStateBytes = 0;
        uint64_t cacheQueryBytes = 0;
        uint64_t cacheTrainRecordBytes = 0;
        uint64_t resolvedSurfaceBytes = 0;
        uint64_t indirectDispatchArgsBytes = 0;
        uint64_t totalBytes = 0;
        uint64_t budgetBytes = 0;
    };

    struct RestirStateResources {
        MTLBufferHandle pathGuiding = nullptr;
        MTLBufferHandle restirPt = nullptr;
        MTLBufferHandle radianceCache = nullptr;
        MTLBufferHandle restirDiCurrent = nullptr;
        MTLBufferHandle restirDiPrevious = nullptr;
        MTLBufferHandle restirDiTemporal = nullptr;
        MTLBufferHandle restirDiSpatial = nullptr;
        MTLBufferHandle restirDiVisibility = nullptr;
        MTLBufferHandle restirDiNeighborPattern = nullptr;
        MTLBufferHandle restirDiDebug = nullptr;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t capacity = 0;
        uint64_t pathGuidingBytes = 0;
        uint64_t restirPtBytes = 0;
        uint64_t radianceCacheBytes = 0;
        uint64_t restirDiReservoirBytes = 0;
        uint64_t restirDiVisibilityBytes = 0;
        uint64_t restirDiNeighborPatternBytes = 0;
        uint64_t restirDiDebugBytes = 0;
        uint64_t causticTransportBytes = 0;
        uint32_t causticMaxVertices = 0;
        bool pathGuidingPrivateStorage = false;
        bool restirPtPrivateStorage = false;
        bool restirDiPrivateStorage = false;
        bool radianceCachePrivateStorage = false;
        bool causticTransportActive = false;
        bool pendingGpuClear = false;
    };

    struct MlBackendResources {
        MTLBufferHandle modelBuffer = nullptr;
        uint64_t modelBytes = 0;
        uint32_t width = 0;
        uint32_t height = 0;
        bool modelValid = false;
        bool activeThisFrame = false;
        uint64_t dispatchCount = 0;
        float lastConfidence = 0.0f;
        float biasRisk = 0.0f;
        std::string modelPath;
        std::string provider = "disabled";
        std::string providerVersion = "disabled";
        std::string tensorContract =
            "radiance,albedo,normal,depth,motion,variance,confidence -> reconstructed_radiance,confidence,validity";
        std::string fallbackReason = "ml backend disabled";
    };

    struct FrameGraphOwnedBuffer {
        RenderResourceGroup group = RenderResourceGroup::FrameConstants;
        MTLBufferHandle buffer = nullptr;
        uint64_t bytes = 0;
        StorageClass storage = StorageClass::Unknown;
        ResourceLifetime lifetime = ResourceLifetime::Transient;
    };

    enum class GpuTimingBucket : uint32_t {
        Intersect = 0,
        Shade = 1,
        ShadowQueue = 2,
        FusedIntegrate = 3,
        Presentation = 4,
        Compaction = 5,
    };

    struct GpuTimingSampleRange {
        GpuTimingBucket bucket = GpuTimingBucket::ShadowQueue;
        std::string passName;
        uint32_t startSample = 0;
        uint32_t endSample = 0;
    };

    struct GpuTimingState {
        bool supported = false;
        bool active = false;
        uint64_t timestampFrequency = 0;
        uint32_t sampleCapacity = 0;
        uint32_t nextSample = 0;
        MTLCounterSetHandle timestampCounterSet = nullptr;
        MTLCounterSampleBufferHandle sampleBuffer = nullptr;
        std::vector<GpuTimingSampleRange> ranges;
        double lastIntersectMs = 0.0;
        double lastShadeMs = 0.0;
        double lastShadowQueueMs = 0.0;
        double lastCompactionMs = 0.0;
        double lastFusedIntegrateMs = 0.0;
        double lastPresentationMs = 0.0;
    };

    MTLDeviceHandle m_device = nullptr;
    DenoiserContext* m_denoiser = nullptr;

    // Uniform buffers
    MTLBufferHandle m_uniformBuffer = nullptr;
    MTLBufferHandle m_displayUniformBuffer = nullptr;
    MTLBufferHandle m_statsBuffer = nullptr;
    MTLBufferHandle m_debugBuffer = nullptr;
    MTLBufferHandle m_materialResourcesArgumentBuffer = nullptr;
    MTLBufferHandle m_dummyBuffer = nullptr;

    // Frame counter for progressive denoising
    uint32_t m_frameCounter = 0;
    uint32_t m_lastDenoisedFrame = 0;
    uint32_t m_nextDenoiseAttemptFrame = 0;
    uint32_t m_denoiseFailureStreak = 0;
    bool m_hasDenoisedOutput = false;
    WavefrontResources m_wavefront{};
    RestirStateResources m_restirState{};
    MlBackendResources m_mlState{};
    TransportMemoryAllocator m_transportAllocator{};
    TransportMemoryRegistry m_transportMemory{};
    std::vector<FrameGraphOwnedBuffer> m_frameGraphOwnedBuffers{};
    GpuTimingState m_gpuTiming{};
    RenderPassGraphDesc m_lastPassGraph{};
    RenderPassExecutionResult m_lastPassGraphExecution{};
    HistoryResetReason m_lastHistoryResetReason = HistoryResetReason::None;
    HistoryResetReason m_lastFrameHistoryResetReason = HistoryResetReason::None;
    PathTracerShaderTypes::PathtraceUniforms m_currentPathtraceUniforms{};
    PathTracerShaderTypes::PathtraceUniforms m_previousSvgfCameraUniforms{};
    bool m_hasCurrentPathtraceUniforms = false;
    bool m_hasPreviousSvgfCameraUniforms = false;
    // Megakernel pipeline of the last integrate dispatch: -1 none yet, 0 software,
    // 1 hardware general, 2 hardware default features.
    int m_lastIntegrateVariant = -1;
    // The last wavefront dispatch was asked for ris_spatial, which it runs as plain RIS.
    bool m_wavefrontRisSpatialRanAsRis = false;
    bool m_preserveSvgfHistoryForNextClear = false;
    bool m_explicitRestirDiActiveThisFrame = false;
    bool m_loggedMegakernelExplicitRestirDiRefusal = false;
    // The last frame ran a ReSTIR DI mode as baseline_emissive because homogeneous volumes are on.
    bool m_restirDiDemotedByVolume = false;
    bool m_restirDiWavefrontStageActiveThisFrame = false;
    bool m_restirDiMegakernelIntegratedStageActiveThisFrame = false;
    bool m_restirDiHardwareTraceActiveThisFrame = false;
    std::string m_restirDiPathStatus = "off";
    const char* m_wavefrontMegakernelFallback = "none";
    const char* m_loggedWavefrontMegakernelFallback = nullptr;  // logs each new reason once, not every frame
    bool m_wavefrontMegakernelFallbackThisFrame = false;
    bool m_wavefrontExplicitRestirDiRanFused = false;
    bool m_loggedWavefrontExplicitRestirDiRanFused = false;

    // Logs the reason, records it for the render metadata and encodes the frame on the megakernel instead.
    uint32_t encodeWavefrontMegakernelFallback(const char* reason,
                                               MTLCommandBufferHandle commandBuffer,
                                               Accumulation& accumulation,
                                               const Pipelines& pipelines,
                                               const SceneResources& scene,
                                               const RenderSettings& settings);

    uint32_t encodeIntegration(MTLCommandBufferHandle commandBuffer,
                               Accumulation& accumulation,
                               const Pipelines& pipelines,
                               const SceneResources& scene,
                               const RenderSettings& settings);

    uint32_t encodeWavefront(MTLCommandBufferHandle commandBuffer,
                             Accumulation& accumulation,
                             const Pipelines& pipelines,
                             const SceneResources& scene,
                             const RenderSettings& settings);

    bool ensureWavefrontResources(const SceneResources& scene,
                                  const RenderSettings& settings,
                                  uint32_t width,
                                  uint32_t height);
    bool ensureRestirStateResources(uint32_t width,
                                    uint32_t height,
                                    const RenderSettings& settings,
                                    MTLCommandBufferHandle commandBuffer = nullptr);
    void rebuildTransportMemoryRegistry(const RenderSettings& settings);
    bool ensureMlBackendResources(uint32_t width,
                                  uint32_t height,
                                  const RenderSettings& settings,
                                  MTLCommandBufferHandle commandBuffer);
    bool encodeMlReconstructionTiny(MTLCommandBufferHandle commandBuffer,
                                    Accumulation& accumulation,
                                    const Pipelines& pipelines,
                                    const RenderSettings& settings);

    bool encodeRestirDiInitializeReservoirs(MTLCommandBufferHandle commandBuffer,
                                            const Accumulation& accumulation,
                                            const Pipelines& pipelines,
                                            const RenderSettings& settings);

    bool encodeRestirDiInitialCandidates(MTLCommandBufferHandle commandBuffer,
                                         const Accumulation& accumulation,
                                         const Pipelines& pipelines,
                                         const SceneResources& scene,
                                         const RenderSettings& settings);

    bool encodeRestirDiTemporalReuse(MTLCommandBufferHandle commandBuffer,
                                     const Accumulation& accumulation,
                                     const Pipelines& pipelines,
                                     const RenderSettings& settings);

    bool encodeRestirDiSpatialReuse(MTLCommandBufferHandle commandBuffer,
                                    const Accumulation& accumulation,
                                    const Pipelines& pipelines,
                                    const RenderSettings& settings);

    bool encodeRestirDiVisibility(MTLCommandBufferHandle commandBuffer,
                                  const Accumulation& accumulation,
                                  const Pipelines& pipelines,
                                  const SceneResources& scene,
                                  const RenderSettings& settings);

    bool encodeRestirDiApplyLighting(MTLCommandBufferHandle commandBuffer,
                                     const Accumulation& accumulation,
                                     const Pipelines& pipelines,
                                     const RenderSettings& settings);

    bool encodeRestirDiDebugAov(MTLCommandBufferHandle commandBuffer,
                                const Accumulation& accumulation,
                                const Pipelines& pipelines,
                                const RenderSettings& settings);

    void encodePresentation(MTLCommandBufferHandle commandBuffer,
                            const Accumulation& accumulation,
                            const Pipelines& pipelines);

    void encodeDenoising(MTLCommandBufferHandle commandBuffer,
                         Accumulation& accumulation,
                         const RenderSettings& settings,
                         uint32_t frameIndex);

    void encodeSvgfDenoising(MTLCommandBufferHandle commandBuffer,
                             Accumulation& accumulation,
                             const Pipelines& pipelines,
                             const RenderSettings& settings);

    void encodeDisplay(MTLRenderCommandEncoderHandle renderEncoder,
                       const Accumulation& accumulation,
                       const Pipelines& pipelines,
                       const RenderSettings& settings);
};

}  // namespace PathTracer
