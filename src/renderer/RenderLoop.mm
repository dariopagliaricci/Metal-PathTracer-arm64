#import "renderer/RenderLoop.h"
#import "renderer/MetalContext.h"
#import "renderer/Accumulation.h"
#import "renderer/Pipelines.h"
#import "renderer/SceneResources.h"
#import "renderer/UIOverlay.h"
#import "renderer/UniformBuilder.h"
#import "renderer/DenoiserContext.h"

#include "MetalShaderTypes.h"
#include "renderer/TraversalService.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>

using PathTracerShaderTypes::DisplayUniforms;
using PathTracerShaderTypes::SvgfDenoiseUniforms;
using PathTracerShaderTypes::PathtraceUniforms;
using PathTracerShaderTypes::PathtraceStats;
using PathTracerShaderTypes::PathtraceDebugBuffer;
using PathTracerShaderTypes::WavefrontRay;
using PathTracerShaderTypes::PathState;
using PathTracerShaderTypes::PathGuidingReservoirState;
using PathTracerShaderTypes::RestirPtReservoirState;
using PathTracerShaderTypes::RadianceCacheState;
using PathTracerShaderTypes::RestirDIReservoirState;
using PathTracerShaderTypes::RestirDIVisibilityState;
using PathTracerShaderTypes::RestirDIInitUniforms;
using PathTracerShaderTypes::WavefrontHitRecord;
using PathTracerShaderTypes::ShadowRay;
using PathTracerShaderTypes::QueueCounters;
using PathTracerShaderTypes::kMaxMaterialSamplers;
using PathTracerShaderTypes::kWavefrontShadowCapacityMultiplier;

namespace PathTracer {

namespace {

bool EnvFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return false;
    }
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "no") != 0 &&
           std::strcmp(value, "NO") != 0;
}

bool SkipBlasUseResources() {
    static const bool enabled = EnvFlagEnabled("PT_SKIP_BLAS_USE_RESOURCES");
    return enabled;
}

bool SkipTextureUseResources() {
    static const bool enabled = EnvFlagEnabled("PT_SKIP_TEXTURE_USE_RESOURCES");
    return enabled;
}

uint64_t CurrentAllocatedSizeBytes(MTLDeviceHandle device) {
    if (device && [device respondsToSelector:@selector(currentAllocatedSize)]) {
        return static_cast<uint64_t>(device.currentAllocatedSize);
    }
    return 0;
}

uint64_t RecommendedMaxWorkingSetBytes(MTLDeviceHandle device) {
    if (device && [device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
        return static_cast<uint64_t>(device.recommendedMaxWorkingSetSize);
    }
    return 0;
}

uint64_t ApplyPercent(uint64_t value, uint64_t numerator, uint64_t denominator) {
    if (denominator == 0u || value == 0u) {
        return 0u;
    }
    const long double scaled =
        static_cast<long double>(value) * static_cast<long double>(numerator) /
        static_cast<long double>(denominator);
    return static_cast<uint64_t>(scaled);
}

void LogIsolationTogglesIfNeeded() {
    static const bool logged = []() {
        const bool skipBlas = SkipBlasUseResources();
        const bool skipTextures = SkipTextureUseResources();
        if (skipBlas || skipTextures) {
            NSLog(@"[Isolation] PT_SKIP_BLAS_USE_RESOURCES=%@ PT_SKIP_TEXTURE_USE_RESOURCES=%@",
                  skipBlas ? @"1" : @"0",
                  skipTextures ? @"1" : @"0");
        }
        return true;
    }();
    (void)logged;
}

inline MTLBufferHandle BufferOrDummy(MTLBufferHandle buffer, MTLBufferHandle dummyBuffer) {
    return buffer ? buffer : dummyBuffer;
}

inline bool WavefrontCompactionEnabled(const RenderSettings& settings) {
    return settings.wavefrontSchedulingPolicy != RenderSettings::WavefrontSchedulingPolicy::Preview &&
           settings.wavefrontCompactionEnabled;
}

inline void LogPipelineDispatchDiagnostics(id<MTLComputePipelineState> pipeline,
                                           MTLSize threadsPerThreadgroup) {
    if (!pipeline) {
        return;
    }

    NSLog(@"[Pipeline] threadExecutionWidth=%lu",
          static_cast<unsigned long>(pipeline.threadExecutionWidth));
    NSLog(@"[Pipeline] maxTotalThreadsPerThreadgroup=%lu",
          static_cast<unsigned long>(pipeline.maxTotalThreadsPerThreadgroup));
    NSLog(@"[Dispatch] threadsPerThreadgroup=%lu x %lu x %lu",
          static_cast<unsigned long>(threadsPerThreadgroup.width),
          static_cast<unsigned long>(threadsPerThreadgroup.height),
          static_cast<unsigned long>(threadsPerThreadgroup.depth));

    if ([(id)pipeline respondsToSelector:@selector(staticThreadgroupMemoryLength)]) {
        NSUInteger staticLength = 0u;
        @try {
            staticLength = [[(id)pipeline valueForKey:@"staticThreadgroupMemoryLength"] unsignedIntegerValue];
        } @catch (__unused NSException* ex) {}
        NSLog(@"[Pipeline] staticThreadgroupMemoryLength=%lu",
              static_cast<unsigned long>(staticLength));
    }
    if ([(id)pipeline respondsToSelector:@selector(threadgroupMemoryLength)]) {
        NSUInteger dynamicLength = 0u;
        @try {
            dynamicLength = [[(id)pipeline valueForKey:@"threadgroupMemoryLength"] unsignedIntegerValue];
        } @catch (__unused NSException* ex) {}
        NSLog(@"[Pipeline] threadgroupMemoryLength=%lu",
              static_cast<unsigned long>(dynamicLength));
    }
}

inline NSUInteger EnvUnsignedOrZero(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return 0u;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value) {
        return 0u;
    }
    return static_cast<NSUInteger>(parsed);
}

inline MTLSize OverrideThreadgroupSizeIfRequested(id<MTLComputePipelineState> pipeline,
                                                  MTLSize fallback) {
    if (!pipeline) {
        return fallback;
    }
    const NSUInteger forcedWidth = EnvUnsignedOrZero("PT_FORCE_TG_WIDTH");
    const NSUInteger forcedHeight = EnvUnsignedOrZero("PT_FORCE_TG_HEIGHT");
    if (forcedWidth == 0u || forcedHeight == 0u) {
        return fallback;
    }

    const NSUInteger maxThreads = std::max<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 1u);
    const NSUInteger clampedWidth = std::max<NSUInteger>(1u, forcedWidth);
    const NSUInteger maxAllowedHeight = std::max<NSUInteger>(1u, maxThreads / clampedWidth);
    const NSUInteger clampedHeight = std::max<NSUInteger>(1u, std::min<NSUInteger>(forcedHeight, maxAllowedHeight));
    return MTLSizeMake(clampedWidth, clampedHeight, 1u);
}

inline MTLSize PreferredIntegrateThreadgroupSize(id<MTLComputePipelineState> pipeline,
                                                 MTLSize fallback) {
    if (!pipeline) {
        return fallback;
    }
    const MTLSize overrideSize = OverrideThreadgroupSizeIfRequested(pipeline, fallback);
    if (overrideSize.width != fallback.width || overrideSize.height != fallback.height) {
        return overrideSize;
    }

    const NSUInteger maxThreads = std::max<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 1u);
    const NSUInteger preferredWidth = std::min<NSUInteger>(16u, maxThreads);
    const NSUInteger preferredHeight =
        std::max<NSUInteger>(1u, std::min<NSUInteger>(16u, maxThreads / std::max<NSUInteger>(preferredWidth, 1u)));
    if (preferredWidth * preferredHeight > maxThreads) {
        return fallback;
    }
    return MTLSizeMake(preferredWidth, preferredHeight, 1u);
}

inline MTLSize PreferredLinearThreadgroupSize(id<MTLComputePipelineState> pipeline) {
    if (!pipeline) {
        return MTLSizeMake(64u, 1u, 1u);
    }
    NSUInteger width = std::max<NSUInteger>(pipeline.threadExecutionWidth, 1u);
    width = std::min<NSUInteger>(width, pipeline.maxTotalThreadsPerThreadgroup);
    return OverrideThreadgroupSizeIfRequested(pipeline, MTLSizeMake(width, 1u, 1u));
}

inline bool ExplicitRestirDiEnabled(const RenderSettings& settings) {
    return settings.directLightMode == RenderSettings::DirectLightMode::RestirDi ||
           settings.directLightMode == RenderSettings::DirectLightMode::RestirDiRegirHybrid;
}

inline bool InteractiveExplicitRestirDiEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_INTERACTIVE_EXPLICIT_RESTIR_DI");
    return enabled;
}

inline bool HeadlessFusedRestirDiReferenceEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_RESTIR_DI_VALIDATE_FUSED_REFERENCE");
    return enabled;
}

TransportMemoryInvalidationMask AllTransportInvalidations() {
    return TransportInvalidationMaskFor({
        HistoryResetReason::ResolutionChanged,
        HistoryResetReason::SceneReloaded,
        HistoryResetReason::CameraCut,
        HistoryResetReason::LightTableChanged,
        HistoryResetReason::FeatureToggleChanged,
        HistoryResetReason::AccelerationStructureRebuilt,
        HistoryResetReason::EmissiveTableChanged,
        HistoryResetReason::MaterialTableChanged,
        HistoryResetReason::FrameAgeExpired,
        HistoryResetReason::ManualDebugReset,
    });
}

TransportMemoryInvalidationMask FrameHistoryInvalidationMask() {
    return TransportInvalidationMaskFor({
        HistoryResetReason::ResolutionChanged,
        HistoryResetReason::SceneReloaded,
        HistoryResetReason::CameraCut,
        HistoryResetReason::FeatureToggleChanged,
        HistoryResetReason::ManualDebugReset,
    });
}

TransportMemoryInvalidationMask LightTransportInvalidationMask() {
    return TransportInvalidationMaskFor({
        HistoryResetReason::ResolutionChanged,
        HistoryResetReason::SceneReloaded,
        HistoryResetReason::CameraCut,
        HistoryResetReason::LightTableChanged,
        HistoryResetReason::EmissiveTableChanged,
        HistoryResetReason::MaterialTableChanged,
        HistoryResetReason::AccelerationStructureRebuilt,
        HistoryResetReason::FeatureToggleChanged,
        HistoryResetReason::FrameAgeExpired,
        HistoryResetReason::ManualDebugReset,
    });
}

TransportConfidenceState TransportConfidencePlaceholder(HistoryResetReason reason) {
    TransportConfidenceState confidence{};
    confidence.lastInvalidationReason = reason;
    return confidence;
}

id<MTLCounterSet> FindTimestampCounterSet(id<MTLDevice> device) {
    if (!device || ![device respondsToSelector:@selector(counterSets)]) {
        return nil;
    }
    for (id<MTLCounterSet> counterSet in device.counterSets) {
        if ([counterSet.name isEqualToString:MTLCommonCounterSetTimestamp]) {
            return counterSet;
        }
    }
    return nil;
}

inline double BytesToMiB(uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

MTLResourceOptions MetalStorageOptions(StorageClass storage) {
    switch (storage) {
        case StorageClass::Private:
            return MTLResourceStorageModePrivate;
        case StorageClass::Managed:
#if TARGET_OS_OSX
            return MTLResourceStorageModeManaged;
#else
            return MTLResourceStorageModeShared;
#endif
        case StorageClass::Shared:
        case StorageClass::Unknown:
        case StorageClass::External:
        case StorageClass::Memoryless:
            return MTLResourceStorageModeShared;
    }
    return MTLResourceStorageModeShared;
}

}  // namespace

bool RenderLoop::initialize(const MetalContext& context, DenoiserContext* denoiser) {
    m_device = context.device();
    m_denoiser = denoiser;
    if (!m_device) {
        return false;
    }

    // Create uniform buffers
    m_uniformBuffer = [m_device newBufferWithLength:sizeof(PathtraceUniforms)
                                            options:MTLResourceStorageModeShared];
    if (!m_uniformBuffer) {
        NSLog(@"Failed to create uniform buffer");
        return false;
    }

    m_displayUniformBuffer = [m_device newBufferWithLength:sizeof(DisplayUniforms)
                                                   options:MTLResourceStorageModeShared];
    if (!m_displayUniformBuffer) {
        NSLog(@"Failed to create display uniform buffer");
        return false;
    }

    m_statsBuffer = [m_device newBufferWithLength:sizeof(PathtraceStats)
                                          options:MTLResourceStorageModeShared];
    if (!m_statsBuffer) {
        NSLog(@"Failed to create stats buffer");
        return false;
    }

#if PT_DEBUG_TOOLS
    m_debugBuffer = [m_device newBufferWithLength:sizeof(PathtraceDebugBuffer)
                                          options:MTLResourceStorageModeShared];
    if (!m_debugBuffer) {
        NSLog(@"Failed to create path debug buffer");
        return false;
    }
#endif

    m_dummyBuffer = [m_device newBufferWithLength:256 options:MTLResourceStorageModeShared];
    if (!m_dummyBuffer) {
        NSLog(@"Failed to create dummy validation buffer");
        return false;
    }

    m_gpuTiming.supported = false;
    m_gpuTiming.active = false;
    if (context.headless() &&
        [m_device respondsToSelector:@selector(supportsCounterSampling:)] &&
        [m_device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
        m_gpuTiming.timestampCounterSet = FindTimestampCounterSet(m_device);
        if (m_gpuTiming.timestampCounterSet) {
            m_gpuTiming.supported = true;
        }
    }

    return true;
}

void RenderLoop::resetHistory(HistoryResetReason reason, bool preserveSvgfHistory) {
    if (!preserveSvgfHistory) {
        m_frameCounter = 0;
    }
    m_lastDenoisedFrame = 0;
    m_hasDenoisedOutput = false;
    m_lastHistoryResetReason = reason;
    m_transportMemory.noteInvalidation(reason);
    if (!preserveSvgfHistory) {
        m_hasPreviousSvgfCameraUniforms = false;
    }
    m_preserveSvgfHistoryForNextClear = preserveSvgfHistory;
    if (m_restirState.pathGuiding && [m_restirState.pathGuiding contents]) {
        std::memset([m_restirState.pathGuiding contents], 0, m_restirState.pathGuidingBytes);
    }
    if (m_restirState.restirPt && [m_restirState.restirPt contents]) {
        std::memset([m_restirState.restirPt contents], 0, m_restirState.restirPtBytes);
    }
    if (m_restirState.radianceCache && [m_restirState.radianceCache contents]) {
        std::memset([m_restirState.radianceCache contents], 0, m_restirState.radianceCacheBytes);
    }
    if (m_restirState.restirDiCurrent && [m_restirState.restirDiCurrent contents]) {
        std::memset([m_restirState.restirDiCurrent contents], 0, m_restirState.restirDiReservoirBytes);
    }
    if (m_restirState.restirDiPrevious && [m_restirState.restirDiPrevious contents]) {
        std::memset([m_restirState.restirDiPrevious contents], 0, m_restirState.restirDiReservoirBytes);
    }
    if (m_restirState.restirDiTemporal && [m_restirState.restirDiTemporal contents]) {
        std::memset([m_restirState.restirDiTemporal contents], 0, m_restirState.restirDiReservoirBytes);
    }
    if (m_restirState.restirDiSpatial && [m_restirState.restirDiSpatial contents]) {
        std::memset([m_restirState.restirDiSpatial contents], 0, m_restirState.restirDiReservoirBytes);
    }
    if (m_restirState.restirDiVisibility && [m_restirState.restirDiVisibility contents]) {
        std::memset([m_restirState.restirDiVisibility contents], 0, m_restirState.restirDiVisibilityBytes);
    }
    if (m_restirState.restirDiDebug && [m_restirState.restirDiDebug contents]) {
        std::memset([m_restirState.restirDiDebug contents], 0, m_restirState.restirDiDebugBytes);
    }
    if (m_restirState.restirDiPrivateStorage || m_restirState.radianceCachePrivateStorage) {
        m_restirState.pendingGpuClear = true;
    }
}

bool RenderLoop::ensureRestirStateResources(uint32_t width,
                                            uint32_t height,
                                            const RenderSettings& settings,
                                            MTLCommandBufferHandle commandBuffer) {
    const uint64_t capacity64 =
        std::max<uint64_t>(1u, static_cast<uint64_t>(width) * static_cast<uint64_t>(height));
    if (capacity64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        NSLog(@"[ReSTIR State] render target too large for per-pixel reservoir state");
        return false;
    }

    const bool restirDiEnabled =
        ExplicitRestirDiEnabled(settings) && m_explicitRestirDiActiveThisFrame;
    const bool pathGuidingEnabled =
        settings.pathGuidingMode == RenderSettings::PathGuidingMode::Prototype;
    const bool restirPtEnabled =
        settings.restirPtMode != RenderSettings::RestirPtMode::Off ||
        settings.restirGiMode == RenderSettings::RestirGiMode::DiffusePrototype;
    const bool radianceCacheEnabled =
        settings.radianceCacheMode == RenderSettings::RadianceCacheMode::Prototype;
    const uint32_t capacity = static_cast<uint32_t>(capacity64);
    const bool hasAnyRestirDiBuffer =
        m_restirState.restirDiCurrent ||
        m_restirState.restirDiPrevious ||
        m_restirState.restirDiTemporal ||
        m_restirState.restirDiSpatial ||
        m_restirState.restirDiVisibility ||
        m_restirState.restirDiNeighborPattern ||
        m_restirState.restirDiDebug;
    const bool restirDiBuffersMatch =
        restirDiEnabled
            ? (m_restirState.restirDiCurrent &&
               m_restirState.restirDiPrevious &&
               m_restirState.restirDiTemporal &&
               m_restirState.restirDiSpatial &&
               m_restirState.restirDiVisibility &&
               m_restirState.restirDiNeighborPattern &&
               (!settings.restirDebugEnabled || m_restirState.restirDiDebug) &&
               (settings.restirDebugEnabled || !m_restirState.restirDiDebug))
            : !hasAnyRestirDiBuffer;
    auto clearGpuPrivateTransportBuffers = [&](MTLCommandBufferHandle clearCommandBuffer) -> bool {
        if (!m_restirState.pendingGpuClear ||
            (!m_restirState.restirDiPrivateStorage && !m_restirState.radianceCachePrivateStorage)) {
            return true;
        }
        if (!clearCommandBuffer) {
            return false;
        }
        id<MTLBlitCommandEncoder> blit = [clearCommandBuffer blitCommandEncoder];
        if (!blit) {
            return false;
        }
        auto fillBuffer = ^(MTLBufferHandle buffer, uint64_t bytes) {
            if (buffer && bytes > 0u) {
                [blit fillBuffer:buffer range:NSMakeRange(0u, static_cast<NSUInteger>(bytes)) value:0u];
            }
        };
        fillBuffer(m_restirState.restirDiCurrent, m_restirState.restirDiReservoirBytes);
        fillBuffer(m_restirState.restirDiPrevious, m_restirState.restirDiReservoirBytes);
        fillBuffer(m_restirState.restirDiTemporal, m_restirState.restirDiReservoirBytes);
        fillBuffer(m_restirState.restirDiSpatial, m_restirState.restirDiReservoirBytes);
        fillBuffer(m_restirState.restirDiVisibility, m_restirState.restirDiVisibilityBytes);
        fillBuffer(m_restirState.restirDiNeighborPattern, m_restirState.restirDiNeighborPatternBytes);
        fillBuffer(m_restirState.restirDiDebug, m_restirState.restirDiDebugBytes);
        fillBuffer(m_restirState.radianceCache, m_restirState.radianceCacheBytes);
        [blit endEncoding];
        m_restirState.pendingGpuClear = false;
        return true;
    };
    if (m_restirState.capacity == capacity &&
        m_restirState.width == width &&
        m_restirState.height == height &&
        (pathGuidingEnabled == (m_restirState.pathGuiding != nullptr)) &&
        (restirPtEnabled == (m_restirState.restirPt != nullptr)) &&
        (radianceCacheEnabled == (m_restirState.radianceCache != nullptr)) &&
        restirDiBuffersMatch) {
        if (!clearGpuPrivateTransportBuffers(commandBuffer)) {
            return false;
        }
        rebuildTransportMemoryRegistry(settings);
        return true;
    }

    const uint64_t reservoirCount =
        capacity64 * static_cast<uint64_t>(PathTracerShaderTypes::kRestirPathReservoirDepthCount);
    const NSUInteger pathGuidingBytes =
        static_cast<NSUInteger>(reservoirCount * sizeof(PathGuidingReservoirState));
    const NSUInteger restirPtBytes =
        static_cast<NSUInteger>(reservoirCount * sizeof(RestirPtReservoirState));
    const NSUInteger radianceCacheBytes =
        static_cast<NSUInteger>(reservoirCount * sizeof(RadianceCacheState));
    const NSUInteger restirDiReservoirBytes =
        static_cast<NSUInteger>(capacity64 * sizeof(RestirDIReservoirState));
    const NSUInteger restirDiVisibilityBytes =
        static_cast<NSUInteger>(capacity64 * sizeof(RestirDIVisibilityState));
    const NSUInteger restirDiNeighborPatternBytes =
        static_cast<NSUInteger>(
            std::max<uint32_t>(1u, settings.spatialReuseNeighborCount) * sizeof(simd::int2));
    const NSUInteger restirDiDebugBytes =
        static_cast<NSUInteger>(capacity64 * sizeof(RestirDIReservoirState));

    const TransportMemoryInvalidationMask allTransport = AllTransportInvalidations();
    const TransportMemoryInvalidationMask lightTransport = LightTransportInvalidationMask();
    m_transportAllocator.clear();
    m_frameGraphOwnedBuffers.clear();

    auto allocateTransportBuffer =
        [&](const char* name,
            TransportMemoryKind kind,
            TransportMemoryOwner owner,
            RenderResourceGroup group,
            uint32_t recordWidth,
            uint32_t recordHeight,
            uint32_t depthOrCells,
            NSUInteger length,
            StorageClass storage,
            ResourceLifetime lifetime,
            TransportMemoryHistoryPolicy history,
            TransportMemoryDebugReadbackPolicy debug,
            TransportMemoryInvalidationMask invalidatesOn,
            const char* label) -> MTLBufferHandle {
        if (length == 0u) {
            return nil;
        }
        TransportMemoryRecord record{};
        record.name = name;
        record.kind = kind;
        record.owner = owner;
        record.targetOwner = owner;
        record.frameGraphGroup = group;
        record.lifetime = lifetime;
        record.width = recordWidth;
        record.height = recordHeight;
        record.depthOrCellCount = depthOrCells;
        record.bytes = static_cast<uint64_t>(length);
        record.history = history;
        record.debug = debug;
        record.invalidatesOn = invalidatesOn;
        record.frameGraphOwned = true;
        record.transportMemoryOwned = true;
        MTLBufferHandle buffer = m_transportAllocator.allocateBuffer(m_device, record, storage, label);
        if (buffer) {
            m_frameGraphOwnedBuffers.push_back({group,
                                                buffer,
                                                static_cast<uint64_t>(length),
                                                storage,
                                                lifetime});
        }
        return buffer;
    };

    id<MTLBuffer> pathGuiding = pathGuidingEnabled
        ? allocateTransportBuffer("PathGuiding.WorldGridBins",
                                  TransportMemoryKind::PathGuidingField,
                                  TransportMemoryOwner::PathGuiding,
                                  RenderResourceGroup::PathGuidingResources,
                                  64u,
                                  1u,
                                  static_cast<uint32_t>(std::min<uint64_t>(
                                      std::max<uint64_t>(1u, reservoirCount / 64u),
                                      std::numeric_limits<uint32_t>::max())),
                                  pathGuidingBytes,
                                  StorageClass::Shared,
                                  ResourceLifetime::History,
                                  TransportMemoryHistoryPolicy::PersistentHistory,
                                  TransportMemoryDebugReadbackPolicy::DirectCpuVisible,
                                  allTransport,
                                  "TransportMemory.PathGuiding.WorldGridBins")
        : nil;
    id<MTLBuffer> restirPt = restirPtEnabled
        ? allocateTransportBuffer("RestirPTPathReservoirs",
                                  TransportMemoryKind::RestirPTReservoir,
                                  TransportMemoryOwner::RestirPT,
                                  RenderResourceGroup::ReSTIRPTResources,
                                  width,
                                  height,
                                  PathTracerShaderTypes::kRestirPathReservoirDepthCount,
                                  restirPtBytes,
                                  StorageClass::Shared,
                                  ResourceLifetime::History,
                                  TransportMemoryHistoryPolicy::PersistentHistory,
                                  TransportMemoryDebugReadbackPolicy::DirectCpuVisible,
                                  allTransport,
                                  "TransportMemory.ReSTIRPT.PathReservoirs")
        : nil;
    id<MTLBuffer> radianceCache = radianceCacheEnabled
        ? allocateTransportBuffer("RadianceCache.WorldGrid",
                                  TransportMemoryKind::RadianceCache,
                                  TransportMemoryOwner::RadianceCache,
                                  RenderResourceGroup::RadianceCacheResources,
                                  width,
                                  height,
                                  PathTracerShaderTypes::kRestirPathReservoirDepthCount,
                                  radianceCacheBytes,
                                  StorageClass::Shared,
                                  ResourceLifetime::History,
                                  TransportMemoryHistoryPolicy::PersistentHistory,
                                  TransportMemoryDebugReadbackPolicy::DirectCpuVisible,
                                  allTransport,
                                  "TransportMemory.RadianceCache.WorldGrid")
        : nil;
    if ((pathGuidingEnabled && !pathGuiding) ||
        (restirPtEnabled && !restirPt) ||
        (radianceCacheEnabled && !radianceCache)) {
        NSLog(@"[ReSTIR State] failed to allocate production reservoir buffers");
        return false;
    }

    id<MTLBuffer> restirDiCurrent = nil;
    id<MTLBuffer> restirDiPrevious = nil;
    id<MTLBuffer> restirDiTemporal = nil;
    id<MTLBuffer> restirDiSpatial = nil;
    id<MTLBuffer> restirDiVisibility = nil;
    id<MTLBuffer> restirDiNeighborPattern = nil;
    id<MTLBuffer> restirDiDebug = nil;
    if (restirDiEnabled) {
        const StorageClass restirDiStorage = StorageClass::Private;
        restirDiCurrent = allocateTransportBuffer("ReSTIRDIReservoirCurrent",
                                                  TransportMemoryKind::RestirDIReservoir,
                                                  TransportMemoryOwner::RestirDI,
                                                  RenderResourceGroup::ReSTIRDIReservoirCurrent,
                                                  width,
                                                  height,
                                                  1u,
                                                  restirDiReservoirBytes,
                                                  restirDiStorage,
                                                  ResourceLifetime::History,
                                                  TransportMemoryHistoryPolicy::CurrentFrame,
                                                  TransportMemoryDebugReadbackPolicy::StagingCopy,
                                                  lightTransport,
                                                  "FrameGraph.ReSTIRDI.Reservoir.Current");
        restirDiPrevious = allocateTransportBuffer("ReSTIRDIReservoirPrevious",
                                                   TransportMemoryKind::RestirDIReservoir,
                                                   TransportMemoryOwner::RestirDI,
                                                   RenderResourceGroup::ReSTIRDIReservoirPrevious,
                                                   width,
                                                   height,
                                                   1u,
                                                   restirDiReservoirBytes,
                                                   restirDiStorage,
                                                   ResourceLifetime::History,
                                                   TransportMemoryHistoryPolicy::PersistentHistory,
                                                   TransportMemoryDebugReadbackPolicy::StagingCopy,
                                                   lightTransport,
                                                   "TransportMemory.ReSTIRDI.Reservoir.Previous");
        restirDiTemporal = allocateTransportBuffer("ReSTIRDIReservoirTemporal",
                                                   TransportMemoryKind::RestirDIReservoir,
                                                   TransportMemoryOwner::RestirDI,
                                                   RenderResourceGroup::ReSTIRDIReservoirTemporal,
                                                   width,
                                                   height,
                                                   1u,
                                                   restirDiReservoirBytes,
                                                   restirDiStorage,
                                                   ResourceLifetime::Transient,
                                                   TransportMemoryHistoryPolicy::CurrentFrame,
                                                   TransportMemoryDebugReadbackPolicy::StagingCopy,
                                                   lightTransport,
                                                   "FrameGraph.ReSTIRDI.Reservoir.Temporal");
        restirDiSpatial = allocateTransportBuffer("ReSTIRDIReservoirSpatial",
                                                  TransportMemoryKind::RestirDIReservoir,
                                                  TransportMemoryOwner::RestirDI,
                                                  RenderResourceGroup::ReSTIRDIReservoirSpatial,
                                                  width,
                                                  height,
                                                  1u,
                                                  restirDiReservoirBytes,
                                                  restirDiStorage,
                                                  ResourceLifetime::Transient,
                                                  TransportMemoryHistoryPolicy::CurrentFrame,
                                                  TransportMemoryDebugReadbackPolicy::StagingCopy,
                                                  lightTransport,
                                                  "FrameGraph.ReSTIRDI.Reservoir.Spatial");
        restirDiVisibility = allocateTransportBuffer("ReSTIRDIVisibility",
                                                     TransportMemoryKind::VisibilityCache,
                                                     TransportMemoryOwner::RestirDI,
                                                     RenderResourceGroup::ReSTIRDIVisibility,
                                                     width,
                                                     height,
                                                     1u,
                                                     restirDiVisibilityBytes,
                                                     restirDiStorage,
                                                     ResourceLifetime::Transient,
                                                     TransportMemoryHistoryPolicy::CurrentFrame,
                                                     TransportMemoryDebugReadbackPolicy::StagingCopy,
                                                     lightTransport,
                                                     "FrameGraph.ReSTIRDI.Visibility");
        restirDiNeighborPattern = allocateTransportBuffer("ReSTIRDINeighborPattern",
                                                          TransportMemoryKind::ReprojectionValidityMask,
                                                          TransportMemoryOwner::RestirDI,
                                                          RenderResourceGroup::ReSTIRDINeighborPattern,
                                                          std::max<uint32_t>(1u, settings.spatialReuseNeighborCount),
                                                          1u,
                                                          1u,
                                                          restirDiNeighborPatternBytes,
                                                          restirDiStorage,
                                                          ResourceLifetime::Persistent,
                                                          TransportMemoryHistoryPolicy::CurrentFrame,
                                                          TransportMemoryDebugReadbackPolicy::StagingCopy,
                                                          lightTransport,
                                                          "TransportMemory.ReSTIRDI.NeighborPattern");
        if (settings.restirDebugEnabled) {
            restirDiDebug = allocateTransportBuffer("ReSTIRDIDebugAOVReadback",
                                                    TransportMemoryKind::DebugReadback,
                                                    TransportMemoryOwner::Debug,
                                                    RenderResourceGroup::ReSTIRDIDebugAOVs,
                                                    width,
                                                    height,
                                                    1u,
                                                    restirDiDebugBytes,
                                                    restirDiStorage,
                                                    ResourceLifetime::DebugReadback,
                                                    TransportMemoryHistoryPolicy::CurrentFrame,
                                                    TransportMemoryDebugReadbackPolicy::StagingCopy,
                                                    lightTransport,
                                                    "Debug.ReSTIRDI.AOVState");
        }
        if (!restirDiCurrent || !restirDiPrevious || !restirDiTemporal || !restirDiSpatial ||
            !restirDiVisibility || !restirDiNeighborPattern ||
            (settings.restirDebugEnabled && !restirDiDebug)) {
            NSLog(@"[ReSTIR DI] failed to allocate explicit reservoir resource set");
            return false;
        }
    }

    if (pathGuiding && [pathGuiding contents]) {
        std::memset([pathGuiding contents], 0, pathGuidingBytes);
    }
    if (restirPt && [restirPt contents]) {
        std::memset([restirPt contents], 0, restirPtBytes);
    }
    if (radianceCache && [radianceCache contents]) {
        std::memset([radianceCache contents], 0, radianceCacheBytes);
    }

    m_restirState.pathGuiding = pathGuiding;
    m_restirState.restirPt = restirPt;
    m_restirState.radianceCache = radianceCache;
    m_restirState.restirDiCurrent = restirDiCurrent;
    m_restirState.restirDiPrevious = restirDiPrevious;
    m_restirState.restirDiTemporal = restirDiTemporal;
    m_restirState.restirDiSpatial = restirDiSpatial;
    m_restirState.restirDiVisibility = restirDiVisibility;
    m_restirState.restirDiNeighborPattern = restirDiNeighborPattern;
    m_restirState.restirDiDebug = restirDiDebug;
    m_restirState.width = width;
    m_restirState.height = height;
    m_restirState.capacity = capacity;
    m_restirState.pathGuidingBytes = pathGuidingEnabled ? pathGuidingBytes : 0u;
    m_restirState.restirPtBytes = restirPtEnabled ? restirPtBytes : 0u;
    m_restirState.radianceCacheBytes = radianceCacheEnabled ? radianceCacheBytes : 0u;
    m_restirState.restirDiReservoirBytes = restirDiEnabled ? restirDiReservoirBytes : 0u;
    m_restirState.restirDiVisibilityBytes = restirDiEnabled ? restirDiVisibilityBytes : 0u;
    m_restirState.restirDiNeighborPatternBytes = restirDiEnabled ? restirDiNeighborPatternBytes : 0u;
    m_restirState.restirDiDebugBytes =
        (restirDiEnabled && settings.restirDebugEnabled) ? restirDiDebugBytes : 0u;
    m_restirState.restirDiPrivateStorage = restirDiEnabled;
    m_restirState.radianceCachePrivateStorage = false;
    m_restirState.pendingGpuClear = restirDiEnabled;
    if (!clearGpuPrivateTransportBuffers(commandBuffer) && (restirDiEnabled || radianceCacheEnabled)) {
        NSLog(@"[TransportMemory] private resources require a command buffer for initialization");
        return false;
    }

    NSLog(@"[ReSTIR State] production reservoir buffers pathGuidingGrid=%.2f MiB restirPt=%.2f MiB radianceCache=%.2f MiB pixels=%u",
          BytesToMiB(m_restirState.pathGuidingBytes),
          BytesToMiB(m_restirState.restirPtBytes),
          BytesToMiB(m_restirState.radianceCacheBytes),
          m_restirState.capacity);
    if (restirDiEnabled) {
        NSLog(@"[ReSTIR DI] explicit resource set reservoir=%.2f MiB each visibility=%.2f MiB neighbor=%.2f KiB debug=%.2f MiB",
              BytesToMiB(m_restirState.restirDiReservoirBytes),
              BytesToMiB(m_restirState.restirDiVisibilityBytes),
              static_cast<double>(m_restirState.restirDiNeighborPatternBytes) / 1024.0,
              BytesToMiB(m_restirState.restirDiDebugBytes));
    }
    rebuildTransportMemoryRegistry(settings);
    return true;
}

void RenderLoop::rebuildTransportMemoryRegistry(const RenderSettings& settings) {
    const uint32_t width = m_restirState.width;
    const uint32_t height = m_restirState.height;
    const uint32_t capacity = m_restirState.capacity;
    const HistoryResetReason lastInvalidation = m_transportMemory.lastInvalidationReason();
    const TransportMemoryInvalidationMask allTransport = AllTransportInvalidations();
    const TransportMemoryInvalidationMask frameHistory = FrameHistoryInvalidationMask();
    const TransportMemoryInvalidationMask lightTransport = LightTransportInvalidationMask();
    const uint32_t pathDepth = PathTracerShaderTypes::kRestirPathReservoirDepthCount;

    m_transportMemory.clear();

    auto applyBridgeMetadata = [&](TransportMemoryRecord& record) {
        record.targetOwner = record.owner;
        record.targetStorage = record.storage;
        record.byteSizeFormula = "registered byte count";
        record.featureGate = "always";
        record.migrationRisk = "low";
        record.hotPath = false;
        record.temporaryShared = false;

        if (record.name == "PathGuiding.WorldGridBins") {
            record.targetOwner = TransportMemoryOwner::PathGuiding;
            record.targetStorage = TransportMemoryStorageClass::Private;
            record.byteSizeFormula =
                "capacity * kRestirPathReservoirDepthCount / 64 * 64 * PathGuidingReservoirState";
            record.featureGate = "pathGuidingMode=prototype";
            record.migrationRisk = "high";
            record.hotPath = true;
            record.temporaryShared = record.storage == TransportMemoryStorageClass::Shared;
        } else if (record.name == "RestirPTPathReservoirs") {
            record.targetStorage = TransportMemoryStorageClass::Private;
            record.byteSizeFormula =
                "width * height * kRestirPathReservoirDepthCount * RestirPtReservoirState";
            record.featureGate = "restirPtMode!=off";
            record.migrationRisk = "high";
            record.hotPath = true;
            record.temporaryShared = record.storage == TransportMemoryStorageClass::Shared;
        } else if (record.name == "RadianceCache.WorldGrid") {
            record.targetOwner = TransportMemoryOwner::RadianceCache;
            record.targetStorage = TransportMemoryStorageClass::Private;
            record.byteSizeFormula =
                "hashed world-grid cells * kRestirPathReservoirDepthCount * RadianceCacheState with query metadata";
            record.featureGate = "radianceCacheMode=prototype";
            record.migrationRisk = "low";
            record.hotPath = true;
            record.temporaryShared = record.storage == TransportMemoryStorageClass::Shared;
        } else if (record.name == "CausticTransport.PathSpacePrototype") {
            record.targetOwner = TransportMemoryOwner::CausticTransport;
            record.targetStorage = TransportMemoryStorageClass::Private;
            record.byteSizeFormula =
                "causticMaxVertices * (EyeSubpathVertex + LightSubpathVertex + PathConnection)";
            record.featureGate = "causticTransportMode=path_space";
            record.migrationRisk = "medium";
            record.hotPath = true;
            record.temporaryShared = false;
        } else if (record.name == "MLBackend.TensorMetadata") {
            record.targetOwner = TransportMemoryOwner::MLBackend;
            record.targetStorage = TransportMemoryStorageClass::Private;
            record.byteSizeFormula =
                "tensor descriptor table + optional scratch tensors; 0 bytes for NoOp backend";
            record.featureGate = "mlBackendMode!=off";
            record.migrationRisk = "medium";
            record.hotPath = false;
            record.temporaryShared = false;
        } else if (record.owner == TransportMemoryOwner::RestirDI) {
            record.targetStorage = TransportMemoryStorageClass::Private;
            record.byteSizeFormula =
                record.kind == TransportMemoryKind::RestirDIReservoir
                    ? "width * height * RestirDIReservoirState"
                    : (record.name == "ReSTIRDINeighborPattern"
                           ? "spatialReuseNeighborCount * RestirDINeighborOffset"
                           : "width * height * RestirDIVisibilityState");
            record.featureGate = "directLightMode=restir_di* && explicitRestirDi";
            record.migrationRisk = "high";
            record.hotPath = true;
            record.temporaryShared = record.storage == TransportMemoryStorageClass::Shared;
        } else if (record.name == "ReSTIRDIDebugAOVReadback") {
            record.targetOwner = TransportMemoryOwner::Debug;
            record.targetStorage = TransportMemoryStorageClass::Staging;
            record.byteSizeFormula = "width * height * RestirDIDebugAOVState";
            record.featureGate = "restirDebugEnabled && explicitRestirDi";
            record.migrationRisk = "low";
        } else if (record.owner == TransportMemoryOwner::Denoiser ||
                   record.owner == TransportMemoryOwner::FrameHistory) {
            record.byteSizeFormula = "width * height * format bytes * history count";
            record.featureGate = record.owner == TransportMemoryOwner::Denoiser
                                     ? "svgfDenoiseEnabled || oidn_export"
                                     : "accumulation_history";
            record.migrationRisk = "medium";
        } else if (record.name.find("Placeholder") != std::string::npos) {
            record.byteSizeFormula = "0 until production backend is enabled";
            record.featureGate = "prototype_or_noop";
            record.migrationRisk = "low";
        }
    };

    auto registerRecord = [&](std::string name,
                              TransportMemoryKind kind,
                              TransportMemoryOwner owner,
                              uint32_t recordWidth,
                              uint32_t recordHeight,
                              uint32_t depthOrCells,
                              uint64_t bytes,
                              TransportMemoryStorageClass storage,
                              TransportMemoryHistoryPolicy history,
                              TransportMemoryDebugReadbackPolicy debug,
                              TransportMemoryInvalidationMask invalidatesOn) {
        TransportMemoryRecord record{};
        record.name = std::move(name);
        record.kind = kind;
        record.owner = owner;
        record.width = recordWidth;
        record.height = recordHeight;
        record.depthOrCellCount = depthOrCells;
        record.bytes = bytes;
        record.storage = storage;
        record.history = history;
        record.debug = debug;
        record.invalidatesOn = invalidatesOn;
        record.confidence = TransportConfidencePlaceholder(lastInvalidation);
        if (const TransportMemoryRecord* allocation = m_transportAllocator.recordForName(record.name)) {
            record.ownedBuffer = allocation->ownedBuffer;
            record.frameGraphGroup = allocation->frameGraphGroup;
            record.lifetime = allocation->lifetime;
            record.storage = allocation->storage;
            record.targetStorage = allocation->targetStorage;
            record.frameGraphOwned = allocation->frameGraphOwned;
            record.transportMemoryOwned = allocation->transportMemoryOwned;
            record.allocationGeneration = allocation->allocationGeneration;
        } else if (record.name == "PathGuiding.WorldGridBins") {
            record.ownedBuffer = m_restirState.pathGuiding;
        } else if (record.name == "RestirPTPathReservoirs") {
            record.ownedBuffer = m_restirState.restirPt;
        } else if (record.name == "RadianceCache.WorldGrid") {
            record.ownedBuffer = m_restirState.radianceCache;
        } else if (record.name == "ReSTIRDIReservoirCurrent") {
            record.ownedBuffer = m_restirState.restirDiCurrent;
        } else if (record.name == "ReSTIRDIReservoirPrevious") {
            record.ownedBuffer = m_restirState.restirDiPrevious;
        } else if (record.name == "ReSTIRDIReservoirTemporal") {
            record.ownedBuffer = m_restirState.restirDiTemporal;
        } else if (record.name == "ReSTIRDIReservoirSpatial") {
            record.ownedBuffer = m_restirState.restirDiSpatial;
        } else if (record.name == "ReSTIRDIVisibility") {
            record.ownedBuffer = m_restirState.restirDiVisibility;
        } else if (record.name == "ReSTIRDINeighborPattern") {
            record.ownedBuffer = m_restirState.restirDiNeighborPattern;
        } else if (record.name == "ReSTIRDIDebugAOVReadback") {
            record.ownedBuffer = m_restirState.restirDiDebug;
        }
        applyBridgeMetadata(record);
        m_transportMemory.registerResource(std::move(record));
    };

    if (m_restirState.pathGuidingBytes > 0u) {
        const uint64_t guidingBinCount =
            static_cast<uint64_t>(capacity) * static_cast<uint64_t>(pathDepth);
        const uint64_t guidingCellCount =
            std::max<uint64_t>(1u, guidingBinCount / 64u);
        registerRecord("PathGuiding.WorldGridBins",
                       TransportMemoryKind::PathGuidingField,
                       TransportMemoryOwner::PathGuiding,
                       64u,
                       1u,
                       static_cast<uint32_t>(std::min<uint64_t>(
                           guidingCellCount, std::numeric_limits<uint32_t>::max())),
                       m_restirState.pathGuidingBytes,
                       TransportMemoryStorageClass::Shared,
                       TransportMemoryHistoryPolicy::PersistentHistory,
                       TransportMemoryDebugReadbackPolicy::DirectCpuVisible,
                       allTransport);
    }

    if (m_restirState.restirPtBytes > 0u) {
        registerRecord("RestirPTPathReservoirs",
                       TransportMemoryKind::RestirPTReservoir,
                       TransportMemoryOwner::RestirPT,
                       width,
                       height,
                       pathDepth,
                       m_restirState.restirPtBytes,
                       TransportMemoryStorageClass::Shared,
                       TransportMemoryHistoryPolicy::PersistentHistory,
                       TransportMemoryDebugReadbackPolicy::DirectCpuVisible,
                       allTransport);
    }

    if (m_restirState.radianceCacheBytes > 0u) {
        registerRecord("RadianceCache.WorldGrid",
                       TransportMemoryKind::RadianceCache,
                       TransportMemoryOwner::RadianceCache,
                       width,
                       height,
                       pathDepth,
                       m_restirState.radianceCacheBytes,
                       TransportMemoryStorageClass::Shared,
                       TransportMemoryHistoryPolicy::PersistentHistory,
                       TransportMemoryDebugReadbackPolicy::DirectCpuVisible,
                       allTransport);
    }

    if (settings.causticTransportMode == RenderSettings::CausticTransportMode::PathSpacePrototype) {
        const uint64_t vertexBudget = std::max<uint32_t>(settings.causticMaxVertices, 1u);
        const uint64_t causticBytes =
            vertexBudget * (sizeof(PathTracerShaderTypes::EyeSubpathVertex) +
                            sizeof(PathTracerShaderTypes::LightSubpathVertex) +
                            sizeof(PathTracerShaderTypes::PathConnection));
        registerRecord("CausticTransport.PathSpacePrototype",
                       TransportMemoryKind::CausticPathSpace,
                       TransportMemoryOwner::CausticTransport,
                       width,
                       height,
                       static_cast<uint32_t>(std::min<uint64_t>(
                           vertexBudget, std::numeric_limits<uint32_t>::max())),
                       causticBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::PersistentHistory,
                       TransportMemoryDebugReadbackPolicy::MetadataOnly,
                       allTransport);
    }

    const bool restirDiEnabled =
        ExplicitRestirDiEnabled(settings) && m_explicitRestirDiActiveThisFrame;
    if (restirDiEnabled && m_restirState.restirDiReservoirBytes > 0u) {
        registerRecord("ReSTIRDIReservoirCurrent",
                       TransportMemoryKind::RestirDIReservoir,
                       TransportMemoryOwner::RestirDI,
                       width,
                       height,
                       1u,
                       m_restirState.restirDiReservoirBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::CurrentFrame,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       lightTransport);
        registerRecord("ReSTIRDIReservoirPrevious",
                       TransportMemoryKind::RestirDIReservoir,
                       TransportMemoryOwner::RestirDI,
                       width,
                       height,
                       1u,
                       m_restirState.restirDiReservoirBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::PersistentHistory,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       lightTransport);
        registerRecord("ReSTIRDIReservoirTemporal",
                       TransportMemoryKind::RestirDIReservoir,
                       TransportMemoryOwner::RestirDI,
                       width,
                       height,
                       1u,
                       m_restirState.restirDiReservoirBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::CurrentFrame,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       lightTransport);
        registerRecord("ReSTIRDIReservoirSpatial",
                       TransportMemoryKind::RestirDIReservoir,
                       TransportMemoryOwner::RestirDI,
                       width,
                       height,
                       1u,
                       m_restirState.restirDiReservoirBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::CurrentFrame,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       lightTransport);
    }

    if (restirDiEnabled && m_restirState.restirDiVisibilityBytes > 0u) {
        registerRecord("ReSTIRDIVisibility",
                       TransportMemoryKind::VisibilityCache,
                       TransportMemoryOwner::RestirDI,
                       width,
                       height,
                       1u,
                       m_restirState.restirDiVisibilityBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::CurrentFrame,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       lightTransport);
    }

    if (restirDiEnabled && m_restirState.restirDiNeighborPatternBytes > 0u) {
        registerRecord("ReSTIRDINeighborPattern",
                       TransportMemoryKind::ReprojectionValidityMask,
                       TransportMemoryOwner::RestirDI,
                       std::max<uint32_t>(1u, settings.spatialReuseNeighborCount),
                       1u,
                       1u,
                       m_restirState.restirDiNeighborPatternBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::CurrentFrame,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       lightTransport);
    }

    if (restirDiEnabled && m_restirState.restirDiDebugBytes > 0u) {
        registerRecord("ReSTIRDIDebugAOVReadback",
                       TransportMemoryKind::DebugReadback,
                       TransportMemoryOwner::Debug,
                       width,
                       height,
                       1u,
                       m_restirState.restirDiDebugBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::CurrentFrame,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       lightTransport);
    }

    const uint64_t pixelCount = static_cast<uint64_t>(capacity);
    if (pixelCount > 0u) {
        const uint64_t rgba32fBytes = pixelCount * sizeof(float) * 4u;
        const uint64_t rg32fBytes = pixelCount * sizeof(float) * 2u;
        const uint64_t r32fBytes = pixelCount * sizeof(float);
        registerRecord("DenoiserRadianceHistory",
                       TransportMemoryKind::DenoiserHistory,
                       TransportMemoryOwner::Denoiser,
                       width,
                       height,
                       1u,
                       rgba32fBytes * 2u,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::PersistentHistory,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       frameHistory);
        registerRecord("DenoiserMomentsVariance",
                       TransportMemoryKind::VarianceConfidence,
                       TransportMemoryOwner::Denoiser,
                       width,
                       height,
                       1u,
                       rg32fBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::PersistentHistory,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       frameHistory);
        registerRecord("TransportConfidenceField",
                       TransportMemoryKind::VarianceConfidence,
                       TransportMemoryOwner::FrameHistory,
                       width,
                       height,
                       1u,
                       r32fBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::PersistentHistory,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       allTransport);
        registerRecord("ReprojectionValidityMask",
                       TransportMemoryKind::ReprojectionValidityMask,
                       TransportMemoryOwner::FrameHistory,
                       width,
                       height,
                       1u,
                       pixelCount,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::PersistentHistory,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       frameHistory);
    }

    if (settings.mlBackendMode != RenderSettings::MlBackendMode::Off) {
        registerRecord("MLBackend.TensorMetadata",
                       TransportMemoryKind::MLBackendTensor,
                       TransportMemoryOwner::MLBackend,
                       width,
                       height,
                       1u,
                       0u,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::Transient,
                       TransportMemoryDebugReadbackPolicy::MetadataOnly,
                       frameHistory);
    }

    registerRecord("ReSTIRGIReservoirPlaceholder",
                   TransportMemoryKind::RestirGIReservoir,
                   TransportMemoryOwner::RestirGI,
                   width,
                   height,
                   1u,
                   0u,
                   TransportMemoryStorageClass::Private,
                   TransportMemoryHistoryPolicy::Placeholder,
                   TransportMemoryDebugReadbackPolicy::MetadataOnly,
                   allTransport);
    if (m_restirState.radianceCacheBytes == 0u) {
        registerRecord("RadianceCachePlaceholder",
                       TransportMemoryKind::RadianceCache,
                       TransportMemoryOwner::RadianceCache,
                       0u,
                       0u,
                       0u,
                       0u,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::Placeholder,
                       TransportMemoryDebugReadbackPolicy::MetadataOnly,
                       allTransport);
    }
    registerRecord("WorldSpaceVisibilityCachePlaceholder",
                   TransportMemoryKind::VisibilityCache,
                   TransportMemoryOwner::VisibilityCache,
                   0u,
                   0u,
                   0u,
                   0u,
                   TransportMemoryStorageClass::Private,
                   TransportMemoryHistoryPolicy::Placeholder,
                   TransportMemoryDebugReadbackPolicy::MetadataOnly,
                   allTransport);
}

bool RenderLoop::encodeRestirDiInitializeReservoirs(MTLCommandBufferHandle commandBuffer,
                                                    const Accumulation& accumulation,
                                                    const Pipelines& pipelines,
                                                    const RenderSettings& settings) {
    const bool restirDiEnabled =
        settings.directLightMode == RenderSettings::DirectLightMode::RestirDi ||
        settings.directLightMode == RenderSettings::DirectLightMode::RestirDiRegirHybrid;
    if (!restirDiEnabled) {
        return true;
    }
    if (!commandBuffer || !pipelines.restirDiInitializeReservoirs()) {
        NSLog(@"[ReSTIR DI] initialize reservoirs pipeline unavailable");
        return false;
    }

    const CGSize currentSize = accumulation.currentSize();
    const uint32_t width = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.width));
    const uint32_t height = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.height));
    if (!ensureRestirStateResources(width, height, settings, commandBuffer)) {
        return false;
    }
    if (!m_restirState.restirDiCurrent ||
        !m_restirState.restirDiTemporal ||
        !m_restirState.restirDiSpatial ||
        !m_restirState.restirDiVisibility) {
        NSLog(@"[ReSTIR DI] initialize reservoirs missing explicit resources");
        return false;
    }

    id<MTLComputePipelineState> pipeline = pipelines.restirDiInitializeReservoirs();
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"ReSTIR DI Initialize Reservoirs";
    [encoder setComputePipelineState:pipeline];

    RestirDIInitUniforms uniforms{};
    uniforms.pixelCount = m_restirState.capacity;
    uniforms.frameIndex = m_frameCounter;

    [encoder setBuffer:m_restirState.restirDiCurrent offset:0 atIndex:0];
    [encoder setBuffer:m_restirState.restirDiTemporal offset:0 atIndex:1];
    [encoder setBuffer:m_restirState.restirDiSpatial offset:0 atIndex:2];
    [encoder setBuffer:m_restirState.restirDiVisibility offset:0 atIndex:3];
    [encoder setBytes:&uniforms length:sizeof(RestirDIInitUniforms) atIndex:4];

    if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
        [encoder useResource:m_restirState.restirDiCurrent usage:MTLResourceUsageWrite];
        [encoder useResource:m_restirState.restirDiTemporal usage:MTLResourceUsageWrite];
        [encoder useResource:m_restirState.restirDiSpatial usage:MTLResourceUsageWrite];
        [encoder useResource:m_restirState.restirDiVisibility usage:MTLResourceUsageWrite];
    }

    const MTLSize threadsPerGrid = MTLSizeMake(m_restirState.capacity, 1u, 1u);
    const MTLSize threadsPerThreadgroup = PreferredLinearThreadgroupSize(pipeline);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    return true;
}

bool RenderLoop::encodeRestirDiInitialCandidates(MTLCommandBufferHandle commandBuffer,
                                                 const Accumulation& accumulation,
                                                 const Pipelines& pipelines,
                                                 const SceneResources& scene,
                                                 const RenderSettings& settings) {
    const bool restirDiEnabled =
        settings.directLightMode == RenderSettings::DirectLightMode::RestirDi ||
        settings.directLightMode == RenderSettings::DirectLightMode::RestirDiRegirHybrid;
    if (!restirDiEnabled) {
        return true;
    }
    if (!commandBuffer || !pipelines.restirDiInitialCandidates()) {
        NSLog(@"[ReSTIR DI] initial candidates pipeline unavailable");
        return false;
    }

    const CGSize currentSize = accumulation.currentSize();
    const uint32_t width = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.width));
    const uint32_t height = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.height));
    if (!ensureRestirStateResources(width, height, settings, commandBuffer) || !m_restirState.restirDiCurrent) {
        return false;
    }

    PathtraceUniforms uniforms = UniformBuilder::buildPathtraceUniforms(
        settings, accumulation, scene, accumulation.currentSize());
    const auto& provider = scene.intersectionProvider();
    const TraversalServiceDecision traversal = SelectTraversalBackend(
        provider,
        TraversalServiceRequest{
            .kind = TraversalQueryKind::RestirDiVisibility,
            .softwareOverride = settings.enableSoftwareRayTracing,
            .hardwarePipelineAvailable = pipelines.restirDiInitialCandidatesHardware() != nil,
            .hardwareTraversalResourceAvailable = provider.hardware.tlas != nil,
            .sceneIdentityBuffersAvailable = scene.meshInfoBuffer() != nil &&
                                             scene.meshVertexBuffer() != nil &&
                                             scene.meshIndexBuffer() != nil,
            .encoderSupportsHardwareRaytracing = true,
        });
    const bool useHardware = traversal.usesHardware;
    m_restirDiHardwareTraceActiveThisFrame =
        m_restirDiHardwareTraceActiveThisFrame || useHardware;
    uniforms.intersectionMode = static_cast<uint32_t>(traversal.mode);
    uniforms.frameIndex = m_frameCounter;
    m_currentPathtraceUniforms = uniforms;
    m_hasCurrentPathtraceUniforms = true;

    id<MTLBuffer> uniformsBuffer =
        [m_device newBufferWithBytes:&uniforms
                              length:sizeof(PathtraceUniforms)
                             options:MTLResourceStorageModeShared];
    if (!uniformsBuffer) {
        NSLog(@"[ReSTIR DI] initial candidates uniform allocation failed");
        return false;
    }

    id<MTLComputePipelineState> pipeline = useHardware
        ? pipelines.restirDiInitialCandidatesHardware()
        : pipelines.restirDiInitialCandidates();
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = useHardware
        ? @"ReSTIR DI Initial Candidates (HW)"
        : @"ReSTIR DI Initial Candidates";
    [encoder setComputePipelineState:pipeline];

    [encoder setBuffer:uniformsBuffer offset:0 atIndex:0];
    if (useHardware) {
        if ([encoder respondsToSelector:@selector(setAccelerationStructure:atBufferIndex:)]) {
            [encoder setAccelerationStructure:provider.hardware.tlas atBufferIndex:1];
        }
        [encoder setBuffer:BufferOrDummy(scene.meshInfoBuffer(), m_dummyBuffer) offset:0 atIndex:2];
        [encoder setBuffer:BufferOrDummy(scene.triangleBuffer(), m_dummyBuffer) offset:0 atIndex:3];
        [encoder setBuffer:BufferOrDummy(scene.meshVertexBuffer(), m_dummyBuffer) offset:0 atIndex:4];
        [encoder setBuffer:BufferOrDummy(scene.meshIndexBuffer(), m_dummyBuffer) offset:0 atIndex:5];
        [encoder setBuffer:BufferOrDummy(provider.hardware.instanceUserIDBuffer, m_dummyBuffer) offset:0 atIndex:6];
        [encoder setBuffer:BufferOrDummy(scene.sphereBuffer(), m_dummyBuffer) offset:0 atIndex:7];
        [encoder setBuffer:BufferOrDummy(scene.rectangleBuffer(), m_dummyBuffer) offset:0 atIndex:8];
        [encoder setBuffer:BufferOrDummy(scene.materialBuffer(), m_dummyBuffer) offset:0 atIndex:9];
        [encoder setBuffer:BufferOrDummy(m_statsBuffer, m_dummyBuffer) offset:0 atIndex:10];
        [encoder setBuffer:BufferOrDummy(provider.software.nodes, m_dummyBuffer) offset:0 atIndex:11];
        [encoder setBuffer:BufferOrDummy(provider.software.primitiveIndices, m_dummyBuffer) offset:0 atIndex:12];
        [encoder setBuffer:BufferOrDummy(scene.emissivePrimitivesBuffer(), m_dummyBuffer) offset:0 atIndex:13];
        [encoder setBuffer:m_restirState.restirDiCurrent offset:0 atIndex:14];
    } else {
        [encoder setBuffer:BufferOrDummy(provider.software.nodes, m_dummyBuffer) offset:0 atIndex:1];
        [encoder setBuffer:BufferOrDummy(provider.software.primitiveIndices, m_dummyBuffer) offset:0 atIndex:2];
        [encoder setBuffer:BufferOrDummy(scene.sphereBuffer(), m_dummyBuffer) offset:0 atIndex:3];
        [encoder setBuffer:BufferOrDummy(scene.materialBuffer(), m_dummyBuffer) offset:0 atIndex:4];
        [encoder setBuffer:BufferOrDummy(m_statsBuffer, m_dummyBuffer) offset:0 atIndex:5];
        [encoder setBuffer:BufferOrDummy(scene.rectangleBuffer(), m_dummyBuffer) offset:0 atIndex:6];
        [encoder setBuffer:BufferOrDummy(scene.triangleBuffer(), m_dummyBuffer) offset:0 atIndex:7];
        [encoder setBuffer:BufferOrDummy(provider.software.tlasNodes, m_dummyBuffer) offset:0 atIndex:8];
        [encoder setBuffer:BufferOrDummy(provider.software.tlasPrimitiveIndices, m_dummyBuffer) offset:0 atIndex:9];
        [encoder setBuffer:BufferOrDummy(provider.software.blasNodes, m_dummyBuffer) offset:0 atIndex:10];
        [encoder setBuffer:BufferOrDummy(provider.software.blasPrimitiveIndices, m_dummyBuffer) offset:0 atIndex:11];
        [encoder setBuffer:BufferOrDummy(provider.software.instanceInfoBuffer, m_dummyBuffer) offset:0 atIndex:12];
        [encoder setBuffer:BufferOrDummy(scene.emissivePrimitivesBuffer(), m_dummyBuffer) offset:0 atIndex:13];
        [encoder setBuffer:m_restirState.restirDiCurrent offset:0 atIndex:14];
    }
    [encoder setTexture:scene.environmentTexture() atIndex:0];

    if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
        [encoder useResource:m_restirState.restirDiCurrent usage:MTLResourceUsageWrite];
        if (useHardware && provider.hardware.tlas) {
            [encoder useResource:provider.hardware.tlas usage:MTLResourceUsageRead];
        }
        if (scene.emissivePrimitivesBuffer()) {
            [encoder useResource:scene.emissivePrimitivesBuffer() usage:MTLResourceUsageRead];
        }
        if (scene.rectangleBuffer()) {
            [encoder useResource:scene.rectangleBuffer() usage:MTLResourceUsageRead];
        }
        if (scene.materialBuffer()) {
            [encoder useResource:scene.materialBuffer() usage:MTLResourceUsageRead];
        }
        if (scene.environmentTexture()) {
            [encoder useResource:scene.environmentTexture() usage:MTLResourceUsageRead];
        }
    }

    const MTLSize threadsPerGrid = MTLSizeMake(width, height, 1u);
    NSUInteger threadWidth = std::max<NSUInteger>(pipeline.threadExecutionWidth, 1u);
    NSUInteger threadHeight = std::max<NSUInteger>(
        1u, pipeline.maxTotalThreadsPerThreadgroup / threadWidth);
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadWidth, threadHeight, 1u);
    threadsPerThreadgroup = PreferredIntegrateThreadgroupSize(pipeline, threadsPerThreadgroup);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    return true;
}

bool RenderLoop::encodeRestirDiTemporalReuse(MTLCommandBufferHandle commandBuffer,
                                             const Accumulation& accumulation,
                                             const Pipelines& pipelines,
                                             const RenderSettings& settings) {
    const bool restirDiEnabled =
        settings.directLightMode == RenderSettings::DirectLightMode::RestirDi ||
        settings.directLightMode == RenderSettings::DirectLightMode::RestirDiRegirHybrid;
    if (!restirDiEnabled) {
        return true;
    }
    if (!commandBuffer || !pipelines.restirDiTemporalReuse()) {
        NSLog(@"[ReSTIR DI] temporal reuse pipeline unavailable");
        return false;
    }

    const CGSize currentSize = accumulation.currentSize();
    const uint32_t width = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.width));
    const uint32_t height = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.height));
    if (!ensureRestirStateResources(width, height, settings, commandBuffer) ||
        !m_restirState.restirDiCurrent ||
        !m_restirState.restirDiPrevious ||
        !m_restirState.restirDiTemporal) {
        NSLog(@"[ReSTIR DI] temporal reuse missing explicit resources");
        return false;
    }

    id<MTLComputePipelineState> pipeline = pipelines.restirDiTemporalReuse();
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"ReSTIR DI Temporal Reuse";
    [encoder setComputePipelineState:pipeline];

    RestirDIInitUniforms uniforms{};
    uniforms.pixelCount = m_restirState.capacity;
    uniforms.frameIndex = m_frameCounter;

    [encoder setBuffer:m_restirState.restirDiCurrent offset:0 atIndex:0];
    [encoder setBuffer:m_restirState.restirDiPrevious offset:0 atIndex:1];
    [encoder setBuffer:m_restirState.restirDiTemporal offset:0 atIndex:2];
    [encoder setBytes:&uniforms length:sizeof(RestirDIInitUniforms) atIndex:3];

    if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
        [encoder useResource:m_restirState.restirDiCurrent usage:MTLResourceUsageRead];
        [encoder useResource:m_restirState.restirDiPrevious usage:MTLResourceUsageRead];
        [encoder useResource:m_restirState.restirDiTemporal usage:MTLResourceUsageWrite];
    }

    const MTLSize threadsPerGrid = MTLSizeMake(m_restirState.capacity, 1u, 1u);
    const MTLSize threadsPerThreadgroup = PreferredLinearThreadgroupSize(pipeline);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    return true;
}

bool RenderLoop::encodeRestirDiSpatialReuse(MTLCommandBufferHandle commandBuffer,
                                            const Accumulation& accumulation,
                                            const Pipelines& pipelines,
                                            const RenderSettings& settings) {
    const bool restirDiEnabled =
        settings.directLightMode == RenderSettings::DirectLightMode::RestirDi ||
        settings.directLightMode == RenderSettings::DirectLightMode::RestirDiRegirHybrid;
    if (!restirDiEnabled) {
        return true;
    }
    if (!commandBuffer || !pipelines.restirDiSpatialReuse()) {
        NSLog(@"[ReSTIR DI] spatial reuse pipeline unavailable");
        return false;
    }

    const CGSize currentSize = accumulation.currentSize();
    const uint32_t width = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.width));
    const uint32_t height = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.height));
    if (!ensureRestirStateResources(width, height, settings, commandBuffer) ||
        !m_restirState.restirDiTemporal ||
        !m_restirState.restirDiSpatial) {
        NSLog(@"[ReSTIR DI] spatial reuse missing explicit resources");
        return false;
    }

    id<MTLComputePipelineState> pipeline = pipelines.restirDiSpatialReuse();
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"ReSTIR DI Spatial Reuse";
    [encoder setComputePipelineState:pipeline];

    RestirDIInitUniforms uniforms{};
    uniforms.pixelCount = m_restirState.capacity;
    uniforms.frameIndex = m_frameCounter;
    uniforms.reserved0 = width;
    uniforms.reserved1 = height;
    uint32_t neighborCount = std::max<uint32_t>(settings.spatialReuseNeighborCount, 1u);

    [encoder setBuffer:m_restirState.restirDiTemporal offset:0 atIndex:0];
    [encoder setBuffer:m_restirState.restirDiSpatial offset:0 atIndex:1];
    [encoder setBytes:&uniforms length:sizeof(RestirDIInitUniforms) atIndex:2];
    [encoder setBytes:&neighborCount length:sizeof(uint32_t) atIndex:3];

    if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
        [encoder useResource:m_restirState.restirDiTemporal usage:MTLResourceUsageRead];
        [encoder useResource:m_restirState.restirDiSpatial usage:MTLResourceUsageWrite];
    }

    const MTLSize threadsPerGrid = MTLSizeMake(width, height, 1u);
    NSUInteger threadWidth = std::max<NSUInteger>(pipeline.threadExecutionWidth, 1u);
    NSUInteger threadHeight = std::max<NSUInteger>(
        1u, pipeline.maxTotalThreadsPerThreadgroup / threadWidth);
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadWidth, threadHeight, 1u);
    threadsPerThreadgroup = PreferredIntegrateThreadgroupSize(pipeline, threadsPerThreadgroup);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    return true;
}

bool RenderLoop::encodeRestirDiVisibility(MTLCommandBufferHandle commandBuffer,
                                          const Accumulation& accumulation,
                                          const Pipelines& pipelines,
                                          const SceneResources& scene,
                                          const RenderSettings& settings) {
    const bool restirDiEnabled =
        settings.directLightMode == RenderSettings::DirectLightMode::RestirDi ||
        settings.directLightMode == RenderSettings::DirectLightMode::RestirDiRegirHybrid;
    if (!restirDiEnabled) {
        return true;
    }
    if (!commandBuffer || !pipelines.restirDiVisibility()) {
        NSLog(@"[ReSTIR DI] visibility pipeline unavailable");
        return false;
    }

    const CGSize currentSize = accumulation.currentSize();
    const uint32_t width = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.width));
    const uint32_t height = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.height));
    if (!ensureRestirStateResources(width, height, settings, commandBuffer) ||
        !m_restirState.restirDiSpatial ||
        !m_restirState.restirDiVisibility ||
        !m_restirState.restirDiPrevious) {
        NSLog(@"[ReSTIR DI] visibility missing explicit resources");
        return false;
    }

    PathtraceUniforms uniforms = UniformBuilder::buildPathtraceUniforms(
        settings, accumulation, scene, accumulation.currentSize());
    const auto& provider = scene.intersectionProvider();
    const TraversalServiceDecision traversal = SelectTraversalBackend(
        provider,
        TraversalServiceRequest{
            .kind = TraversalQueryKind::RestirDiVisibility,
            .softwareOverride = settings.enableSoftwareRayTracing,
            .hardwarePipelineAvailable = pipelines.restirDiVisibilityHardware() != nil,
            .hardwareTraversalResourceAvailable = provider.hardware.tlas != nil,
            .sceneIdentityBuffersAvailable = scene.meshInfoBuffer() != nil &&
                                             scene.meshVertexBuffer() != nil &&
                                             scene.meshIndexBuffer() != nil,
            .encoderSupportsHardwareRaytracing = true,
        });
    const bool useHardware = traversal.usesHardware;
    m_restirDiHardwareTraceActiveThisFrame =
        m_restirDiHardwareTraceActiveThisFrame || useHardware;
    uniforms.intersectionMode = static_cast<uint32_t>(traversal.mode);
    uniforms.frameIndex = m_frameCounter;
    id<MTLBuffer> uniformsBuffer =
        [m_device newBufferWithBytes:&uniforms
                              length:sizeof(PathtraceUniforms)
                             options:MTLResourceStorageModeShared];
    if (!uniformsBuffer) {
        NSLog(@"[ReSTIR DI] visibility uniform allocation failed");
        return false;
    }

    id<MTLComputePipelineState> pipeline = useHardware
        ? pipelines.restirDiVisibilityHardware()
        : pipelines.restirDiVisibility();
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = useHardware ? @"ReSTIR DI Visibility (HW)" : @"ReSTIR DI Visibility";
    [encoder setComputePipelineState:pipeline];

    [encoder setBuffer:uniformsBuffer offset:0 atIndex:0];
    if (useHardware) {
        if ([encoder respondsToSelector:@selector(setAccelerationStructure:atBufferIndex:)]) {
            [encoder setAccelerationStructure:provider.hardware.tlas atBufferIndex:1];
        }
        [encoder setBuffer:BufferOrDummy(scene.meshInfoBuffer(), m_dummyBuffer) offset:0 atIndex:2];
        [encoder setBuffer:BufferOrDummy(scene.triangleBuffer(), m_dummyBuffer) offset:0 atIndex:3];
        [encoder setBuffer:BufferOrDummy(scene.meshVertexBuffer(), m_dummyBuffer) offset:0 atIndex:4];
        [encoder setBuffer:BufferOrDummy(scene.meshIndexBuffer(), m_dummyBuffer) offset:0 atIndex:5];
        [encoder setBuffer:BufferOrDummy(provider.hardware.instanceUserIDBuffer, m_dummyBuffer) offset:0 atIndex:6];
        [encoder setBuffer:BufferOrDummy(scene.sphereBuffer(), m_dummyBuffer) offset:0 atIndex:7];
        [encoder setBuffer:BufferOrDummy(scene.rectangleBuffer(), m_dummyBuffer) offset:0 atIndex:8];
        [encoder setBuffer:BufferOrDummy(scene.materialBuffer(), m_dummyBuffer) offset:0 atIndex:9];
        [encoder setBuffer:BufferOrDummy(m_statsBuffer, m_dummyBuffer) offset:0 atIndex:10];
        [encoder setBuffer:BufferOrDummy(provider.software.nodes, m_dummyBuffer) offset:0 atIndex:11];
        [encoder setBuffer:BufferOrDummy(provider.software.primitiveIndices, m_dummyBuffer) offset:0 atIndex:12];
        [encoder setBuffer:m_restirState.restirDiSpatial offset:0 atIndex:13];
        [encoder setBuffer:m_restirState.restirDiVisibility offset:0 atIndex:14];
        [encoder setBuffer:m_restirState.restirDiPrevious offset:0 atIndex:15];
    } else {
        [encoder setBuffer:BufferOrDummy(provider.software.nodes, m_dummyBuffer) offset:0 atIndex:1];
        [encoder setBuffer:BufferOrDummy(provider.software.primitiveIndices, m_dummyBuffer) offset:0 atIndex:2];
        [encoder setBuffer:BufferOrDummy(scene.sphereBuffer(), m_dummyBuffer) offset:0 atIndex:3];
        [encoder setBuffer:BufferOrDummy(scene.materialBuffer(), m_dummyBuffer) offset:0 atIndex:4];
        [encoder setBuffer:BufferOrDummy(m_statsBuffer, m_dummyBuffer) offset:0 atIndex:5];
        [encoder setBuffer:BufferOrDummy(scene.rectangleBuffer(), m_dummyBuffer) offset:0 atIndex:6];
        [encoder setBuffer:BufferOrDummy(scene.triangleBuffer(), m_dummyBuffer) offset:0 atIndex:7];
        [encoder setBuffer:BufferOrDummy(provider.software.tlasNodes, m_dummyBuffer) offset:0 atIndex:8];
        [encoder setBuffer:BufferOrDummy(provider.software.tlasPrimitiveIndices, m_dummyBuffer) offset:0 atIndex:9];
        [encoder setBuffer:BufferOrDummy(provider.software.blasNodes, m_dummyBuffer) offset:0 atIndex:10];
        [encoder setBuffer:BufferOrDummy(provider.software.blasPrimitiveIndices, m_dummyBuffer) offset:0 atIndex:11];
        [encoder setBuffer:BufferOrDummy(provider.software.instanceInfoBuffer, m_dummyBuffer) offset:0 atIndex:12];
        [encoder setBuffer:m_restirState.restirDiSpatial offset:0 atIndex:13];
        [encoder setBuffer:m_restirState.restirDiVisibility offset:0 atIndex:14];
        [encoder setBuffer:m_restirState.restirDiPrevious offset:0 atIndex:15];
    }

    if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
        [encoder useResource:m_restirState.restirDiSpatial usage:MTLResourceUsageRead];
        [encoder useResource:m_restirState.restirDiVisibility usage:MTLResourceUsageWrite];
        [encoder useResource:m_restirState.restirDiPrevious usage:MTLResourceUsageWrite];
        if (useHardware && provider.hardware.tlas) {
            [encoder useResource:provider.hardware.tlas usage:MTLResourceUsageRead];
        }
    }

    const MTLSize threadsPerGrid = MTLSizeMake(width, height, 1u);
    NSUInteger threadWidth = std::max<NSUInteger>(pipeline.threadExecutionWidth, 1u);
    NSUInteger threadHeight = std::max<NSUInteger>(
        1u, pipeline.maxTotalThreadsPerThreadgroup / threadWidth);
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadWidth, threadHeight, 1u);
    threadsPerThreadgroup = PreferredIntegrateThreadgroupSize(pipeline, threadsPerThreadgroup);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    return true;
}

bool RenderLoop::encodeRestirDiApplyLighting(MTLCommandBufferHandle commandBuffer,
                                             const Accumulation& accumulation,
                                             const Pipelines& pipelines,
                                             const RenderSettings& settings) {
    if (!ExplicitRestirDiEnabled(settings)) {
        return true;
    }
    if (!commandBuffer || !pipelines.restirDiApplyLighting()) {
        NSLog(@"[ReSTIR DI] apply lighting pipeline unavailable");
        return false;
    }

    MTLTextureHandle radianceSum = accumulation.radianceSum();
    if (!radianceSum || !m_restirState.restirDiVisibility) {
        NSLog(@"[ReSTIR DI] apply lighting missing explicit visibility or radiance target");
        return false;
    }

    const CGSize currentSize = accumulation.currentSize();
    const uint32_t width = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.width));
    const uint32_t height = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.height));
    if (!ensureRestirStateResources(width, height, settings, commandBuffer)) {
        return false;
    }

    RestirDIInitUniforms uniforms{};
    uniforms.pixelCount = m_restirState.capacity;
    uniforms.frameIndex = m_frameCounter;
    uniforms.reserved0 = width;
    uniforms.reserved1 = height;

    id<MTLComputePipelineState> pipeline = pipelines.restirDiApplyLighting();
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"ReSTIR DI Apply Lighting";
    [encoder setComputePipelineState:pipeline];
    [encoder setTexture:radianceSum atIndex:0];
    [encoder setBuffer:m_restirState.restirDiVisibility offset:0 atIndex:0];
    [encoder setBytes:&uniforms length:sizeof(RestirDIInitUniforms) atIndex:1];
    if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
        [encoder useResource:m_restirState.restirDiVisibility usage:MTLResourceUsageRead];
        [encoder useResource:radianceSum usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
    }

    const MTLSize threadsPerGrid = MTLSizeMake(width, height, 1u);
    NSUInteger threadWidth = std::max<NSUInteger>(pipeline.threadExecutionWidth, 1u);
    NSUInteger threadHeight = std::max<NSUInteger>(
        1u, pipeline.maxTotalThreadsPerThreadgroup / threadWidth);
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadWidth, threadHeight, 1u);
    threadsPerThreadgroup = PreferredIntegrateThreadgroupSize(pipeline, threadsPerThreadgroup);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    return true;
}

bool RenderLoop::encodeRestirDiDebugAov(MTLCommandBufferHandle commandBuffer,
                                        const Accumulation& accumulation,
                                        const Pipelines& pipelines,
                                        const RenderSettings& settings) {
    if (!ExplicitRestirDiEnabled(settings) || !settings.restirDebugEnabled) {
        return true;
    }
    if (!commandBuffer || !pipelines.restirDiDebugAov()) {
        NSLog(@"[ReSTIR DI] debug AOV pipeline unavailable");
        return false;
    }

    const CGSize currentSize = accumulation.currentSize();
    const uint32_t width = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.width));
    const uint32_t height = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.height));
    if (!ensureRestirStateResources(width, height, settings, commandBuffer) ||
        !m_restirState.restirDiCurrent ||
        !m_restirState.restirDiTemporal ||
        !m_restirState.restirDiSpatial ||
        !m_restirState.restirDiVisibility ||
        !m_restirState.restirDiDebug) {
        NSLog(@"[ReSTIR DI] debug AOV missing explicit resources");
        return false;
    }

    RestirDIInitUniforms uniforms{};
    uniforms.pixelCount = m_restirState.capacity;
    uniforms.frameIndex = m_frameCounter;
    uniforms.reserved0 = width;
    uniforms.reserved1 = height;
    uint32_t debugView = static_cast<uint32_t>(settings.restirDebugView);

    id<MTLComputePipelineState> pipeline = pipelines.restirDiDebugAov();
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"ReSTIR DI Debug AOV";
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:m_restirState.restirDiCurrent offset:0 atIndex:0];
    [encoder setBuffer:m_restirState.restirDiTemporal offset:0 atIndex:1];
    [encoder setBuffer:m_restirState.restirDiSpatial offset:0 atIndex:2];
    [encoder setBuffer:m_restirState.restirDiVisibility offset:0 atIndex:3];
    [encoder setBuffer:m_restirState.restirDiDebug offset:0 atIndex:4];
    [encoder setBytes:&uniforms length:sizeof(RestirDIInitUniforms) atIndex:5];
    [encoder setBytes:&debugView length:sizeof(uint32_t) atIndex:6];

    if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
        [encoder useResource:m_restirState.restirDiCurrent usage:MTLResourceUsageRead];
        [encoder useResource:m_restirState.restirDiTemporal usage:MTLResourceUsageRead];
        [encoder useResource:m_restirState.restirDiSpatial usage:MTLResourceUsageRead];
        [encoder useResource:m_restirState.restirDiVisibility usage:MTLResourceUsageRead];
        [encoder useResource:m_restirState.restirDiDebug usage:MTLResourceUsageWrite];
    }

    const MTLSize threadsPerGrid = MTLSizeMake(m_restirState.capacity, 1u, 1u);
    const MTLSize threadsPerThreadgroup = PreferredLinearThreadgroupSize(pipeline);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
    return true;
}

uint32_t RenderLoop::encodeIntegration(MTLCommandBufferHandle commandBuffer,
                                       Accumulation& accumulation,
                                       const Pipelines& pipelines,
                                       const SceneResources& scene,
                                       const RenderSettings& settings) {
    LogIsolationTogglesIfNeeded();
    
    // Throttle samples per frame when falling back to software to keep UI responsive
    const uint32_t requestedSamples = std::max<uint32_t>(1u, settings.samplesPerFrame);
    uint32_t samplesThisFrame = requestedSamples;

    const CGSize currentSize = accumulation.currentSize();
    const uint32_t width = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.width));
    const uint32_t height = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.height));
    if (!ensureRestirStateResources(width, height, settings, commandBuffer)) {
        return 0u;
    }

    id<MTLComputeCommandEncoder> encoder = nil;
    if (m_gpuTiming.active &&
        m_gpuTiming.nextSample + 2u <= m_gpuTiming.sampleCapacity &&
        [commandBuffer respondsToSelector:@selector(computeCommandEncoderWithDescriptor:)]) {
        MTLComputePassDescriptor* passDescriptor = [MTLComputePassDescriptor computePassDescriptor];
        MTLComputePassSampleBufferAttachmentDescriptor* attachment =
            passDescriptor.sampleBufferAttachments[0];
        attachment.sampleBuffer = (id<MTLCounterSampleBuffer>)m_gpuTiming.sampleBuffer;
        const uint32_t startSample = m_gpuTiming.nextSample++;
        const uint32_t endSample = m_gpuTiming.nextSample++;
        attachment.startOfEncoderSampleIndex = startSample;
        attachment.endOfEncoderSampleIndex = endSample;
        m_gpuTiming.ranges.push_back({GpuTimingBucket::FusedIntegrate, startSample, endSample});
        encoder = [commandBuffer computeCommandEncoderWithDescriptor:passDescriptor];
    } else {
        encoder = [commandBuffer computeCommandEncoder];
    }
    encoder.label = @"Path Tracer Integrate";

    const auto& provider = scene.intersectionProvider();
    bool encoderSupportsRt =
        [encoder respondsToSelector:@selector(setAccelerationStructure:atBufferIndex:)];
    const TraversalServiceDecision traversal = SelectTraversalBackend(
        provider,
        TraversalServiceRequest{
            .kind = TraversalQueryKind::ContinuationRay,
            .softwareOverride = settings.enableSoftwareRayTracing,
            .hardwarePipelineAvailable = pipelines.integrateHardware() != nil,
            .hardwareTraversalResourceAvailable = provider.hardware.tlas != nil,
            .sceneIdentityBuffersAvailable = scene.meshInfoBuffer() != nil &&
                                             scene.meshVertexBuffer() != nil &&
                                             scene.meshIndexBuffer() != nil,
            .encoderSupportsHardwareRaytracing = encoderSupportsRt,
        });
    bool useHardware = traversal.usesHardware;

    if (!useHardware) {
        samplesThisFrame = 1u;
    }

    id<MTLComputePipelineState> selectedPipeline = useHardware
        ? pipelines.integrateHardware()
        : pipelines.integrate();
    if (!selectedPipeline) {
        selectedPipeline = pipelines.integrate();
        useHardware = false;
    }
    [encoder setComputePipelineState:selectedPipeline];

    NSUInteger threadWidth = selectedPipeline.threadExecutionWidth;
    NSUInteger threadHeight = std::max<NSUInteger>(
        1, selectedPipeline.maxTotalThreadsPerThreadgroup / std::max<NSUInteger>(threadWidth, 1));
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadWidth, threadHeight, 1);
    threadsPerThreadgroup = PreferredIntegrateThreadgroupSize(selectedPipeline, threadsPerThreadgroup);
    static bool sLoggedIntegrateDispatchDiagnostics = false;
    if (!sLoggedIntegrateDispatchDiagnostics) {
        sLoggedIntegrateDispatchDiagnostics = true;
        LogPipelineDispatchDiagnostics(selectedPipeline, threadsPerThreadgroup);
    }

    id<MTLTexture> environmentTexture = scene.environmentTexture();
    [encoder setTexture:environmentTexture atIndex:2];
    const auto& materialTextures = scene.materialTextures();
    const auto& materialSamplers = scene.materialSamplers();

    if (auto argumentEncoder = pipelines.materialResourcesArgumentEncoder();
        argumentEncoder) {
        NSUInteger encodedLength = argumentEncoder.encodedLength;
        if (encodedLength > 0u) {
            if (!m_materialResourcesArgumentBuffer ||
                m_materialResourcesArgumentBuffer.length < encodedLength) {
                m_materialResourcesArgumentBuffer =
                    [m_device newBufferWithLength:encodedLength options:MTLResourceStorageModeShared];
            }
            if (m_materialResourcesArgumentBuffer) {
                std::memset([m_materialResourcesArgumentBuffer contents], 0, encodedLength);
                [argumentEncoder setArgumentBuffer:m_materialResourcesArgumentBuffer offset:0];
                if (!materialTextures.empty()) {
                    NSUInteger textureCount =
                        std::min<NSUInteger>(materialTextures.size(), PathTracerShaderTypes::kMaxMaterialTextures);
                    [argumentEncoder setTextures:materialTextures.data()
                                       withRange:NSMakeRange(0, textureCount)];
                }
                if (!materialSamplers.empty()) {
                    NSUInteger samplerCount =
                        std::min<NSUInteger>(materialSamplers.size(), kMaxMaterialSamplers);
                    [argumentEncoder setSamplerStates:materialSamplers.data()
                                            withRange:NSMakeRange(PathTracerShaderTypes::kMaxMaterialTextures,
                                                                  samplerCount)];
                }
                [argumentEncoder setBuffer:BufferOrDummy(m_restirState.pathGuiding, m_dummyBuffer)
                                    offset:0
                                   atIndex:PathTracerShaderTypes::kMaterialResourcePathGuidingStateIndex];
                [argumentEncoder setBuffer:BufferOrDummy(m_restirState.restirPt, m_dummyBuffer)
                                    offset:0
                                   atIndex:PathTracerShaderTypes::kMaterialResourceRestirPtStateIndex];
                [argumentEncoder setBuffer:BufferOrDummy(m_restirState.radianceCache, m_dummyBuffer)
                                    offset:0
                                   atIndex:PathTracerShaderTypes::kMaterialResourceRadianceCacheStateIndex];
                [encoder setBuffer:m_materialResourcesArgumentBuffer offset:0 atIndex:23];
                if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                    if (m_restirState.pathGuiding) {
                        [encoder useResource:m_restirState.pathGuiding
                                       usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                    }
                    if (m_restirState.restirPt) {
                        [encoder useResource:m_restirState.restirPt
                                       usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                    }
                    if (m_restirState.radianceCache) {
                        [encoder useResource:m_restirState.radianceCache
                                       usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                    }
                }
                if (!SkipTextureUseResources() &&
                    [encoder respondsToSelector:@selector(useResource:usage:)]) {
                    for (id<MTLTexture> texture : materialTextures) {
                        if (texture) {
                            [encoder useResource:texture usage:MTLResourceUsageRead];
                        }
                    }
                }
            }
        }
    } else {
        [encoder setBuffer:nil offset:0 atIndex:23];
    }

    NSMutableArray<id<MTLBuffer>>* sampleUniformBuffers =
        [NSMutableArray arrayWithCapacity:samplesThisFrame];

    if (useHardware) {
        if (encoderSupportsRt) {
            [encoder setAccelerationStructure:provider.hardware.tlas atBufferIndex:1];
        }

        if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
            if (provider.hardware.tlas) {
                [encoder useResource:provider.hardware.tlas usage:MTLResourceUsageRead];
            }
            if (!SkipBlasUseResources() && provider.hardware.blas) {
                [encoder useResource:provider.hardware.blas usage:MTLResourceUsageRead];
            }
            if (!SkipBlasUseResources() && !provider.hardware.blasHandles.empty()) {
                for (auto h : provider.hardware.blasHandles) {
                    id<MTLAccelerationStructure> blasObj = (__bridge id<MTLAccelerationStructure>)h;
                    if (blasObj) {
                        [encoder useResource:blasObj usage:MTLResourceUsageRead];
                    }
                }
            }
            if (provider.hardware.instanceBuffer) {
                [encoder useResource:provider.hardware.instanceBuffer usage:MTLResourceUsageRead];
            }
            if (provider.hardware.instanceUserIDBuffer) {
                [encoder useResource:provider.hardware.instanceUserIDBuffer usage:MTLResourceUsageRead];
            }
        }

        [encoder setBuffer:BufferOrDummy(scene.meshInfoBuffer(), m_dummyBuffer) offset:0 atIndex:2];
        [encoder setBuffer:scene.triangleBuffer() offset:0 atIndex:3];
        [encoder setBuffer:BufferOrDummy(provider.software.nodes, m_dummyBuffer) offset:0 atIndex:4];
        [encoder setBuffer:BufferOrDummy(provider.software.primitiveIndices, m_dummyBuffer) offset:0 atIndex:5];
        [encoder setBuffer:BufferOrDummy(scene.sphereBuffer(), m_dummyBuffer) offset:0 atIndex:6];
        [encoder setBuffer:BufferOrDummy(scene.materialBuffer(), m_dummyBuffer) offset:0 atIndex:7];
        [encoder setBuffer:BufferOrDummy(m_statsBuffer, m_dummyBuffer) offset:0 atIndex:8];
        [encoder setBuffer:BufferOrDummy(scene.rectangleBuffer(), m_dummyBuffer) offset:0 atIndex:9];
        [encoder setBuffer:BufferOrDummy(scene.environmentConditionalAliasBuffer(), m_dummyBuffer) offset:0 atIndex:10];
        [encoder setBuffer:BufferOrDummy(scene.environmentMarginalAliasBuffer(), m_dummyBuffer) offset:0 atIndex:11];
        [encoder setBuffer:BufferOrDummy(scene.environmentPdfBuffer(), m_dummyBuffer) offset:0 atIndex:12];
        [encoder setBuffer:BufferOrDummy(provider.hardware.instanceUserIDBuffer, m_dummyBuffer) offset:0 atIndex:13];
        [encoder setBuffer:BufferOrDummy(scene.meshVertexBuffer(), m_dummyBuffer) offset:0 atIndex:14];
        [encoder setBuffer:BufferOrDummy(scene.meshIndexBuffer(), m_dummyBuffer) offset:0 atIndex:15];
        [encoder setBuffer:BufferOrDummy(m_debugBuffer, m_dummyBuffer) offset:0 atIndex:16];
        [encoder setBuffer:BufferOrDummy(provider.software.tlasNodes, m_dummyBuffer) offset:0 atIndex:17];
        [encoder setBuffer:BufferOrDummy(provider.software.tlasPrimitiveIndices, m_dummyBuffer) offset:0 atIndex:18];
        [encoder setBuffer:BufferOrDummy(provider.software.blasNodes, m_dummyBuffer) offset:0 atIndex:19];
        [encoder setBuffer:BufferOrDummy(provider.software.blasPrimitiveIndices, m_dummyBuffer) offset:0 atIndex:20];
        [encoder setBuffer:BufferOrDummy(provider.software.instanceInfoBuffer, m_dummyBuffer) offset:0 atIndex:21];
        [encoder setBuffer:BufferOrDummy(scene.materialTextureInfoBuffer(), m_dummyBuffer) offset:0 atIndex:22];
        [encoder setBuffer:BufferOrDummy(scene.emissivePrimitivesBuffer(), m_dummyBuffer) offset:0 atIndex:24];
        [encoder setBuffer:BufferOrDummy(m_restirState.pathGuiding, m_dummyBuffer) offset:0 atIndex:25];
        [encoder setBuffer:BufferOrDummy(m_restirState.restirPt, m_dummyBuffer) offset:0 atIndex:26];
    } else {
        if (encoderSupportsRt) {
            [encoder setAccelerationStructure:nil atBufferIndex:1];
        }
        [encoder setBuffer:BufferOrDummy(provider.software.nodes, m_dummyBuffer) offset:0 atIndex:1];
        [encoder setBuffer:BufferOrDummy(provider.software.primitiveIndices, m_dummyBuffer) offset:0 atIndex:2];
        [encoder setBuffer:BufferOrDummy(scene.sphereBuffer(), m_dummyBuffer) offset:0 atIndex:3];
        [encoder setBuffer:BufferOrDummy(scene.materialBuffer(), m_dummyBuffer) offset:0 atIndex:4];
        [encoder setBuffer:BufferOrDummy(m_statsBuffer, m_dummyBuffer) offset:0 atIndex:5];
        [encoder setBuffer:BufferOrDummy(scene.rectangleBuffer(), m_dummyBuffer) offset:0 atIndex:6];
        [encoder setBuffer:BufferOrDummy(scene.environmentConditionalAliasBuffer(), m_dummyBuffer) offset:0 atIndex:7];
        [encoder setBuffer:BufferOrDummy(scene.environmentMarginalAliasBuffer(), m_dummyBuffer) offset:0 atIndex:8];
        [encoder setBuffer:BufferOrDummy(scene.environmentPdfBuffer(), m_dummyBuffer) offset:0 atIndex:9];
        [encoder setBuffer:BufferOrDummy(scene.triangleBuffer(), m_dummyBuffer) offset:0 atIndex:10];
        [encoder setBuffer:BufferOrDummy(provider.software.tlasNodes, m_dummyBuffer) offset:0 atIndex:11];
        [encoder setBuffer:BufferOrDummy(provider.software.tlasPrimitiveIndices, m_dummyBuffer) offset:0 atIndex:12];
        [encoder setBuffer:BufferOrDummy(provider.software.blasNodes, m_dummyBuffer) offset:0 atIndex:13];
        [encoder setBuffer:BufferOrDummy(provider.software.blasPrimitiveIndices, m_dummyBuffer) offset:0 atIndex:14];
        [encoder setBuffer:BufferOrDummy(provider.software.instanceInfoBuffer, m_dummyBuffer) offset:0 atIndex:15];
        [encoder setBuffer:BufferOrDummy(scene.meshInfoBuffer(), m_dummyBuffer) offset:0 atIndex:16];
        [encoder setBuffer:BufferOrDummy(scene.meshVertexBuffer(), m_dummyBuffer) offset:0 atIndex:17];
        if (scene.meshIndexBuffer()) {
            [encoder setBuffer:scene.meshIndexBuffer() offset:0 atIndex:18];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:18];
        }
        if (m_debugBuffer) {
            [encoder setBuffer:m_debugBuffer offset:0 atIndex:19];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:19];
        }
        if (scene.materialTextureInfoBuffer()) {
            [encoder setBuffer:scene.materialTextureInfoBuffer() offset:0 atIndex:20];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:20];
        }
        [encoder setBuffer:BufferOrDummy(scene.emissivePrimitivesBuffer(), m_dummyBuffer) offset:0 atIndex:21];
        [encoder setBuffer:BufferOrDummy(m_restirState.pathGuiding, m_dummyBuffer) offset:0 atIndex:22];
        [encoder setBuffer:BufferOrDummy(m_restirState.restirPt, m_dummyBuffer) offset:0 atIndex:24];
    }

    // Dispatch multiple samples
    for (uint32_t sample = 0; sample < samplesThisFrame; ++sample) {
        PathtraceUniforms uniforms = UniformBuilder::buildPathtraceUniforms(
            settings, accumulation, scene, accumulation.currentSize());
        if (m_explicitRestirDiActiveThisFrame) {
            uniforms.emissivePrimitiveCount = 0u;
        }
        PathTracerShaderTypes::IntersectionMode modeOverride =
            useHardware ? PathTracerShaderTypes::IntersectionMode::HardwareRayTracing
                        : PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
        uniforms.intersectionMode = static_cast<uint32_t>(modeOverride);
        id<MTLBuffer> sampleUniformBuffer =
            [m_device newBufferWithBytes:&uniforms
                                  length:sizeof(PathtraceUniforms)
                                 options:MTLResourceStorageModeShared];
        if (!sampleUniformBuffer) {
            [encoder endEncoding];
            NSLog(@"[Megakernel] per-sample uniform allocation failed");
            return sample;
        }
        [sampleUniformBuffers addObject:sampleUniformBuffer];

        [encoder setTexture:accumulation.radianceSum() atIndex:0];
        [encoder setTexture:accumulation.sampleCountTexture() atIndex:1];
        [encoder setTexture:accumulation.albedoBuffer() atIndex:3];
        [encoder setTexture:accumulation.normalBuffer() atIndex:4];
        [encoder setTexture:accumulation.positionBuffer() atIndex:5];
        [encoder setTexture:accumulation.materialFeatureBuffer() atIndex:6];
        [encoder setTexture:accumulation.motionVectorBuffer() atIndex:7];
        [encoder setBuffer:sampleUniformBuffer offset:0 atIndex:0];
        m_currentPathtraceUniforms = uniforms;
        m_hasCurrentPathtraceUniforms = true;

        MTLSize threadsPerGrid = MTLSizeMake(uniforms.width, uniforms.height, 1);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

        accumulation.incrementFrame();
    }
    
    [encoder endEncoding];

    return samplesThisFrame;
}

bool RenderLoop::ensureWavefrontResources(const SceneResources& scene,
                                          uint32_t width,
                                          uint32_t height) {
    const uint64_t capacity64 = std::max<uint64_t>(1u, static_cast<uint64_t>(width) * static_cast<uint64_t>(height));
    if (capacity64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        NSLog(@"[Wavefront] render target too large for fixed queue capacity");
        return false;
    }
    const uint64_t shadowCapacity64 = capacity64 * static_cast<uint64_t>(kWavefrontShadowCapacityMultiplier);
    if (shadowCapacity64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        NSLog(@"[Wavefront] render target too large for fixed shadow queue capacity");
        return false;
    }
    const uint32_t capacity = static_cast<uint32_t>(capacity64);
    const uint32_t shadowCapacity = static_cast<uint32_t>(shadowCapacity64);
    if (m_wavefront.capacity == capacity &&
        m_wavefront.shadowCapacity == shadowCapacity &&
        m_wavefront.width == width &&
        m_wavefront.height == height &&
        m_wavefront.activeRays[0] &&
        m_wavefront.activeRays[1] &&
        m_wavefront.hitRecords &&
        m_wavefront.shadowRays &&
        m_wavefront.pathStates &&
        m_wavefront.queueCounters &&
        m_wavefront.indirectDispatchArgs) {
        m_wavefront.budgetBytes = scene.recommendedMaxWorkingSetSizeBytes();
        return true;
    }

    auto makeBuffer = ^MTLBufferHandle(NSUInteger length, NSString* label) {
        id<MTLBuffer> buffer = [m_device newBufferWithLength:length options:MTLResourceStorageModeShared];
        if (buffer) {
            buffer.label = label;
        }
        return buffer;
    };

    const NSUInteger activeBytes = static_cast<NSUInteger>(capacity64 * sizeof(WavefrontRay));
    const NSUInteger hitBytes = static_cast<NSUInteger>(capacity64 * sizeof(WavefrontHitRecord));
    const NSUInteger shadowBytes = static_cast<NSUInteger>(shadowCapacity64 * sizeof(ShadowRay));
    const NSUInteger pathStateBytes = static_cast<NSUInteger>(capacity64 * sizeof(PathState));
    const NSUInteger counterBytes = sizeof(QueueCounters);
    const NSUInteger indirectDispatchArgsBytes = static_cast<NSUInteger>(3u * sizeof(uint32_t));

    MTLBufferHandle active0 = makeBuffer(activeBytes, @"Wavefront Active Rays A");
    MTLBufferHandle active1 = makeBuffer(activeBytes, @"Wavefront Active Rays B");
    MTLBufferHandle hits = makeBuffer(hitBytes, @"Wavefront Hit Records");
    MTLBufferHandle shadows = makeBuffer(shadowBytes, @"Wavefront Shadow Rays");
    MTLBufferHandle pathStates = makeBuffer(pathStateBytes, @"Wavefront Path States");
    MTLBufferHandle counters = makeBuffer(counterBytes, @"Wavefront Queue Counters");
    MTLBufferHandle indirectArgs = makeBuffer(indirectDispatchArgsBytes, @"Wavefront Indirect Dispatch Args");
    if (!active0 || !active1 || !hits || !shadows || !pathStates || !counters || !indirectArgs) {
        NSLog(@"[Wavefront] failed to allocate queue buffers");
        return false;
    }

    m_wavefront.activeRays[0] = active0;
    m_wavefront.activeRays[1] = active1;
    m_wavefront.hitRecords = hits;
    m_wavefront.shadowRays = shadows;
    m_wavefront.pathStates = pathStates;
    m_wavefront.queueCounters = counters;
    m_wavefront.indirectDispatchArgs = indirectArgs;
    m_wavefront.width = width;
    m_wavefront.height = height;
    m_wavefront.capacity = capacity;
    m_wavefront.shadowCapacity = shadowCapacity;
    m_wavefront.activeBytes = activeBytes;
    m_wavefront.nextBytes = activeBytes;
    m_wavefront.hitBytes = hitBytes;
    m_wavefront.shadowBytes = shadowBytes;
    m_wavefront.pathStateBytes = pathStateBytes;
    m_wavefront.indirectDispatchArgsBytes = indirectDispatchArgsBytes;
    m_wavefront.totalBytes =
        static_cast<uint64_t>(activeBytes) * 2u +
        static_cast<uint64_t>(hitBytes) +
        static_cast<uint64_t>(shadowBytes) +
        static_cast<uint64_t>(pathStateBytes) +
        static_cast<uint64_t>(counterBytes) +
        static_cast<uint64_t>(indirectDispatchArgsBytes);
    m_wavefront.budgetBytes = scene.recommendedMaxWorkingSetSizeBytes();

    const double budgetUsagePct =
        (m_wavefront.budgetBytes > 0u)
            ? (static_cast<double>(m_wavefront.totalBytes) /
               static_cast<double>(m_wavefront.budgetBytes)) * 100.0
            : 0.0;
    NSLog(@"[Wavefront] queueFootprint active=%.2f MiB next=%.2f MiB hit=%.2f MiB shadow=%.2f MiB "
          "path=%.2f MiB indirect=%.2f KiB total=%.2f MiB recommendedMaxWorkingSetSize=%.2f MiB usage=%.2f%%",
          BytesToMiB(m_wavefront.activeBytes),
          BytesToMiB(m_wavefront.nextBytes),
          BytesToMiB(m_wavefront.hitBytes),
          BytesToMiB(m_wavefront.shadowBytes),
          BytesToMiB(m_wavefront.pathStateBytes),
          static_cast<double>(m_wavefront.indirectDispatchArgsBytes) / 1024.0,
          BytesToMiB(m_wavefront.totalBytes),
          BytesToMiB(m_wavefront.budgetBytes),
          budgetUsagePct);
    return true;
}

uint32_t RenderLoop::encodeWavefront(MTLCommandBufferHandle commandBuffer,
                                     Accumulation& accumulation,
                                     const Pipelines& pipelines,
                                     const SceneResources& scene,
                                     const RenderSettings& settings) {
    const auto makeEncoder = [&](GpuTimingBucket bucket,
                                 NSString* label) -> id<MTLComputeCommandEncoder> {
        id<MTLComputeCommandEncoder> encoder = nil;
        if (m_gpuTiming.active &&
            m_gpuTiming.nextSample + 2u <= m_gpuTiming.sampleCapacity &&
            [commandBuffer respondsToSelector:@selector(computeCommandEncoderWithDescriptor:)]) {
            MTLComputePassDescriptor* passDescriptor = [MTLComputePassDescriptor computePassDescriptor];
            MTLComputePassSampleBufferAttachmentDescriptor* attachment =
                passDescriptor.sampleBufferAttachments[0];
            attachment.sampleBuffer = (id<MTLCounterSampleBuffer>)m_gpuTiming.sampleBuffer;
            const uint32_t startSample = m_gpuTiming.nextSample++;
            const uint32_t endSample = m_gpuTiming.nextSample++;
            attachment.startOfEncoderSampleIndex = startSample;
            attachment.endOfEncoderSampleIndex = endSample;
            m_gpuTiming.ranges.push_back({bucket, startSample, endSample});
            encoder = [commandBuffer computeCommandEncoderWithDescriptor:passDescriptor];
        } else {
            encoder = [commandBuffer computeCommandEncoder];
        }
        encoder.label = label;
        return encoder;
    };

    if (!pipelines.wavefrontReset() ||
        !pipelines.wavefrontBuildDispatchArgs() ||
        !pipelines.wavefrontCompactQueue() ||
        !pipelines.wavefrontPrepareIteration() ||
        !pipelines.wavefrontGeneratePrimary() ||
        !pipelines.wavefrontIntersect() ||
        !pipelines.wavefrontShade() ||
        !pipelines.wavefrontShadow() ||
        !pipelines.wavefrontSwapQueues() ||
        !pipelines.wavefrontAccumulate()) {
        NSLog(@"[Wavefront] required pipelines missing; falling back to megakernel");
        return encodeIntegration(commandBuffer, accumulation, pipelines, scene, settings);
    }
    if (m_explicitRestirDiActiveThisFrame && !pipelines.wavefrontRestirDiDirectLighting()) {
        NSLog(@"[Wavefront] explicit ReSTIR DI direct-lighting pipeline missing; frame not dispatched");
        return 0u;
    }

    const CGSize currentSize = accumulation.currentSize();
    const uint32_t width = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.width));
    const uint32_t height = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.height));
    if (!ensureRestirStateResources(width, height, settings, commandBuffer)) {
        NSLog(@"[Wavefront] ReSTIR state allocation failed; falling back to megakernel");
        return encodeIntegration(commandBuffer, accumulation, pipelines, scene, settings);
    }
    if (!ensureWavefrontResources(scene, width, height)) {
        NSLog(@"[Wavefront] queue allocation failed; falling back to megakernel");
        return encodeIntegration(commandBuffer, accumulation, pipelines, scene, settings);
    }

    const auto& provider = scene.intersectionProvider();
    const auto& materialTextures = scene.materialTextures();
    const auto& materialSamplers = scene.materialSamplers();
    const bool hasMeshIdentityBuffers = scene.meshInfoBuffer() != nil &&
                                        scene.meshVertexBuffer() != nil &&
                                        scene.meshIndexBuffer() != nil;
    const TraversalServiceDecision intersectTraversal = SelectTraversalBackend(
        provider,
        TraversalServiceRequest{
            .kind = TraversalQueryKind::ContinuationRay,
            .softwareOverride = settings.enableSoftwareRayTracing,
            .hardwarePipelineAvailable = pipelines.wavefrontIntersectHardware() != nil,
            .hardwareTraversalResourceAvailable = provider.hardware.tlas != nil,
            .sceneIdentityBuffersAvailable = hasMeshIdentityBuffers,
            .encoderSupportsHardwareRaytracing = true,
        });
    const TraversalServiceDecision shadowTraversal = SelectTraversalBackend(
        provider,
        TraversalServiceRequest{
            .kind = TraversalQueryKind::ShadowRay,
            .softwareOverride = settings.enableSoftwareRayTracing,
            .hardwarePipelineAvailable = pipelines.wavefrontShadowHardware() != nil,
            .hardwareTraversalResourceAvailable = provider.hardware.tlas != nil,
            .sceneIdentityBuffersAvailable = hasMeshIdentityBuffers,
            .encoderSupportsHardwareRaytracing = true,
        });
    const bool canHardwareIntersect = intersectTraversal.usesHardware;
    const bool canHardwareShadow = shadowTraversal.usesHardware;
    const uint32_t requestedSamples = std::max<uint32_t>(1u, settings.samplesPerFrame);
    const bool wavefrontCompactionEnabled = WavefrontCompactionEnabled(settings);
    uint32_t samplesThisFrame = requestedSamples;
    if (!canHardwareIntersect) {
        samplesThisFrame = 1u;
    }

    const MTLSize resetThreads = MTLSizeMake(1u, 1u, 1u);
    const MTLSize fullGrid = MTLSizeMake(m_wavefront.capacity, 1u, 1u);
    auto encodeWavefrontDispatchArgs = [&](uint32_t queueType,
                                           MTLSize threadsPerThreadgroup) -> bool {
        if (!pipelines.wavefrontBuildDispatchArgs() || !m_wavefront.indirectDispatchArgs) {
            return false;
        }
        id<MTLComputeCommandEncoder> encoder =
            makeEncoder(GpuTimingBucket::ShadowQueue, @"Wavefront Build Dispatch Args");
        [encoder setComputePipelineState:pipelines.wavefrontBuildDispatchArgs()];
        uint32_t args[2] = {
            queueType,
            static_cast<uint32_t>(std::max<NSUInteger>(threadsPerThreadgroup.width, 1u)),
        };
        [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:0];
        [encoder setBuffer:m_wavefront.indirectDispatchArgs offset:0 atIndex:1];
        [encoder setBytes:args length:sizeof(args) atIndex:2];
        if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
            [encoder useResource:m_wavefront.queueCounters usage:MTLResourceUsageRead];
            [encoder useResource:m_wavefront.indirectDispatchArgs usage:MTLResourceUsageWrite];
        }
        [encoder dispatchThreads:resetThreads threadsPerThreadgroup:resetThreads];
        [encoder endEncoding];
        return true;
    };
    auto dispatchWavefrontStage = [&](id<MTLComputeCommandEncoder> encoder,
                                      MTLSize threadsPerThreadgroup,
                                      bool useIndirect) {
        if (useIndirect &&
            m_wavefront.indirectDispatchArgs &&
            [encoder respondsToSelector:@selector(dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:)]) {
            if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                [encoder useResource:m_wavefront.indirectDispatchArgs usage:MTLResourceUsageRead];
            }
            [encoder dispatchThreadgroupsWithIndirectBuffer:m_wavefront.indirectDispatchArgs
                                       indirectBufferOffset:0
                                      threadsPerThreadgroup:threadsPerThreadgroup];
        } else {
            [encoder dispatchThreads:fullGrid threadsPerThreadgroup:threadsPerThreadgroup];
        }
    };
    NSMutableArray<id<MTLBuffer>>* sampleUniformBuffers =
        [NSMutableArray arrayWithCapacity:samplesThisFrame];

    for (uint32_t sample = 0; sample < samplesThisFrame; ++sample) {
        PathtraceUniforms uniforms = UniformBuilder::buildPathtraceUniforms(
            settings, accumulation, scene, accumulation.currentSize());
        uniforms.explicitRestirDiLightingStage =
            m_explicitRestirDiActiveThisFrame ? 1u : 0u;
        uniforms.executionMode =
            static_cast<uint32_t>(PathTracerShaderTypes::RenderExecutionMode::Wavefront);
        uniforms.intersectionMode = static_cast<uint32_t>(
            canHardwareIntersect
                ? PathTracerShaderTypes::IntersectionMode::HardwareRayTracing
                : PathTracerShaderTypes::IntersectionMode::SoftwareBVH);
        m_currentPathtraceUniforms = uniforms;
        m_hasCurrentPathtraceUniforms = true;
        id<MTLBuffer> sampleUniformBuffer =
            [m_device newBufferWithBytes:&uniforms
                                  length:sizeof(PathtraceUniforms)
                                 options:MTLResourceStorageModeShared];
        if (!sampleUniformBuffer) {
            NSLog(@"[Wavefront] per-sample uniform allocation failed; falling back to megakernel");
            return encodeIntegration(commandBuffer, accumulation, pipelines, scene, settings);
        }
        [sampleUniformBuffers addObject:sampleUniformBuffer];

        int currentQueueIndex = 0;
        int nextQueueIndex = 1;

        {
            id<MTLComputeCommandEncoder> encoder =
                makeEncoder(GpuTimingBucket::ShadowQueue, @"Wavefront Reset");
            [encoder setComputePipelineState:pipelines.wavefrontReset()];
            [encoder setBuffer:sampleUniformBuffer offset:0 atIndex:0];
            [encoder setBuffer:m_wavefront.activeRays[0] offset:0 atIndex:1];
            [encoder setBuffer:m_wavefront.activeRays[1] offset:0 atIndex:2];
            [encoder setBuffer:m_wavefront.hitRecords offset:0 atIndex:3];
            [encoder setBuffer:m_wavefront.shadowRays offset:0 atIndex:4];
            [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:5];
            [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:6];
            [encoder setBuffer:m_statsBuffer offset:0 atIndex:7];
            [encoder setBuffer:BufferOrDummy(m_debugBuffer, m_dummyBuffer) offset:0 atIndex:8];
            [encoder dispatchThreads:resetThreads threadsPerThreadgroup:resetThreads];
            [encoder endEncoding];
        }

        {
            id<MTLComputeCommandEncoder> encoder =
                makeEncoder(GpuTimingBucket::ShadowQueue, @"Wavefront Generate Primary");
            [encoder setComputePipelineState:pipelines.wavefrontGeneratePrimary()];
            [encoder setBuffer:sampleUniformBuffer offset:0 atIndex:0];
            [encoder setBuffer:m_wavefront.activeRays[currentQueueIndex] offset:0 atIndex:1];
            [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:2];
            [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:3];
            [encoder setBuffer:m_statsBuffer offset:0 atIndex:4];
            [encoder setTexture:accumulation.sampleCountTexture() atIndex:0];
            const MTLSize threadsPerThreadgroup =
                PreferredLinearThreadgroupSize(pipelines.wavefrontGeneratePrimary());
            [encoder dispatchThreads:fullGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

        for (uint32_t depth = 0; depth < std::max<uint32_t>(1u, settings.maxDepth); ++depth) {
            {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::ShadowQueue, @"Wavefront Prepare Iteration");
                [encoder setComputePipelineState:pipelines.wavefrontPrepareIteration()];
                [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:0];
                [encoder setBuffer:m_statsBuffer offset:0 atIndex:1];
                [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:2];
                [encoder setBuffer:sampleUniformBuffer offset:0 atIndex:3];
                const MTLSize threadsPerThreadgroup =
                    PreferredLinearThreadgroupSize(pipelines.wavefrontPrepareIteration());
                [encoder dispatchThreads:fullGrid threadsPerThreadgroup:threadsPerThreadgroup];
                [encoder endEncoding];
            }

            if (wavefrontCompactionEnabled) {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::Compaction, @"Wavefront Compact Queue");
                [encoder setComputePipelineState:pipelines.wavefrontCompactQueue()];
                [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:0];
                [encoder setBuffer:m_statsBuffer offset:0 atIndex:1];
                [encoder setBuffer:sampleUniformBuffer offset:0 atIndex:2];
                [encoder dispatchThreads:resetThreads threadsPerThreadgroup:resetThreads];
                [encoder endEncoding];
            }

            id<MTLComputePipelineState> intersectPipeline = canHardwareIntersect
                ? pipelines.wavefrontIntersectHardware()
                : pipelines.wavefrontIntersect();
            const MTLSize intersectThreadsPerThreadgroup =
                PreferredLinearThreadgroupSize(intersectPipeline);
            const bool useIndirectIntersect =
                encodeWavefrontDispatchArgs(
                    static_cast<uint32_t>(PathTracerShaderTypes::WavefrontQueueType::PrimaryRay),
                    intersectThreadsPerThreadgroup);

            if (canHardwareIntersect) {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::Intersect, @"Wavefront Intersect (HW)");
                [encoder setComputePipelineState:pipelines.wavefrontIntersectHardware()];
                if ([encoder respondsToSelector:@selector(setAccelerationStructure:atBufferIndex:)]) {
                    [encoder setAccelerationStructure:provider.hardware.tlas atBufferIndex:1];
                }
                if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                    [encoder useResource:provider.hardware.tlas usage:MTLResourceUsageRead];
                    if (!SkipBlasUseResources() && provider.hardware.blas) {
                        [encoder useResource:provider.hardware.blas usage:MTLResourceUsageRead];
                    }
                    if (!SkipBlasUseResources() && !provider.hardware.blasHandles.empty()) {
                        for (auto h : provider.hardware.blasHandles) {
                            id<MTLAccelerationStructure> blasObj = (__bridge id<MTLAccelerationStructure>)h;
                            if (blasObj) {
                                [encoder useResource:blasObj usage:MTLResourceUsageRead];
                            }
                        }
                    }
                    if (provider.hardware.instanceBuffer) {
                        [encoder useResource:provider.hardware.instanceBuffer usage:MTLResourceUsageRead];
                    }
                    if (provider.hardware.instanceUserIDBuffer) {
                        [encoder useResource:provider.hardware.instanceUserIDBuffer usage:MTLResourceUsageRead];
                    }
                }
                [encoder setBuffer:sampleUniformBuffer offset:0 atIndex:0];
                [encoder setBuffer:m_wavefront.activeRays[currentQueueIndex] offset:0 atIndex:2];
                [encoder setBuffer:m_wavefront.hitRecords offset:0 atIndex:3];
                [encoder setBuffer:BufferOrDummy(scene.meshInfoBuffer(), m_dummyBuffer) offset:0 atIndex:4];
                [encoder setBuffer:BufferOrDummy(scene.triangleBuffer(), m_dummyBuffer) offset:0 atIndex:5];
                [encoder setBuffer:BufferOrDummy(scene.meshVertexBuffer(), m_dummyBuffer) offset:0 atIndex:6];
                [encoder setBuffer:BufferOrDummy(scene.meshIndexBuffer(), m_dummyBuffer) offset:0 atIndex:7];
                [encoder setBuffer:BufferOrDummy(provider.hardware.instanceUserIDBuffer, m_dummyBuffer) offset:0 atIndex:8];
                [encoder setBuffer:BufferOrDummy(scene.sphereBuffer(), m_dummyBuffer) offset:0 atIndex:9];
                [encoder setBuffer:BufferOrDummy(scene.rectangleBuffer(), m_dummyBuffer) offset:0 atIndex:10];
                [encoder setBuffer:BufferOrDummy(provider.software.nodes, m_dummyBuffer) offset:0 atIndex:11];
                [encoder setBuffer:BufferOrDummy(provider.software.primitiveIndices, m_dummyBuffer) offset:0 atIndex:12];
                [encoder setBuffer:BufferOrDummy(m_statsBuffer, m_dummyBuffer) offset:0 atIndex:13];
                [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:14];
                dispatchWavefrontStage(encoder, intersectThreadsPerThreadgroup, useIndirectIntersect);
                [encoder endEncoding];
            } else {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::Intersect, @"Wavefront Intersect");
                [encoder setComputePipelineState:pipelines.wavefrontIntersect()];
                [encoder setBuffer:sampleUniformBuffer offset:0 atIndex:0];
                [encoder setBuffer:m_wavefront.activeRays[currentQueueIndex] offset:0 atIndex:1];
                [encoder setBuffer:m_wavefront.hitRecords offset:0 atIndex:2];
                [encoder setBuffer:BufferOrDummy(m_statsBuffer, m_dummyBuffer) offset:0 atIndex:3];
                [encoder setBuffer:BufferOrDummy(scene.sphereBuffer(), m_dummyBuffer) offset:0 atIndex:4];
                [encoder setBuffer:BufferOrDummy(scene.rectangleBuffer(), m_dummyBuffer) offset:0 atIndex:5];
                [encoder setBuffer:BufferOrDummy(scene.triangleBuffer(), m_dummyBuffer) offset:0 atIndex:6];
                [encoder setBuffer:BufferOrDummy(provider.software.tlasNodes, m_dummyBuffer) offset:0 atIndex:7];
                [encoder setBuffer:BufferOrDummy(provider.software.tlasPrimitiveIndices, m_dummyBuffer) offset:0 atIndex:8];
                [encoder setBuffer:BufferOrDummy(provider.software.instanceInfoBuffer, m_dummyBuffer) offset:0 atIndex:9];
                [encoder setBuffer:BufferOrDummy(provider.software.blasNodes, m_dummyBuffer) offset:0 atIndex:10];
                [encoder setBuffer:BufferOrDummy(provider.software.blasPrimitiveIndices, m_dummyBuffer) offset:0 atIndex:11];
                [encoder setBuffer:BufferOrDummy(provider.software.nodes, m_dummyBuffer) offset:0 atIndex:12];
                [encoder setBuffer:BufferOrDummy(provider.software.primitiveIndices, m_dummyBuffer) offset:0 atIndex:13];
                [encoder setBuffer:BufferOrDummy(scene.meshInfoBuffer(), m_dummyBuffer) offset:0 atIndex:14];
                [encoder setBuffer:BufferOrDummy(scene.meshVertexBuffer(), m_dummyBuffer) offset:0 atIndex:15];
                [encoder setBuffer:BufferOrDummy(scene.meshIndexBuffer(), m_dummyBuffer) offset:0 atIndex:16];
                [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:17];
                dispatchWavefrontStage(encoder, intersectThreadsPerThreadgroup, useIndirectIntersect);
                [encoder endEncoding];
            }

            if (m_explicitRestirDiActiveThisFrame) {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::Shade, @"Wavefront ReSTIR DI Direct Lighting");
                [encoder setComputePipelineState:pipelines.wavefrontRestirDiDirectLighting()];
                if ([encoder respondsToSelector:@selector(setAccelerationStructure:atBufferIndex:)]) {
                    [encoder setAccelerationStructure:(canHardwareIntersect ? provider.hardware.tlas : nil)
                                        atBufferIndex:18];
                }
                if (canHardwareIntersect &&
                    [encoder respondsToSelector:@selector(useResource:usage:)]) {
                    if (provider.hardware.tlas) {
                        [encoder useResource:provider.hardware.tlas usage:MTLResourceUsageRead];
                    }
                    if (!SkipBlasUseResources() && provider.hardware.blas) {
                        [encoder useResource:provider.hardware.blas usage:MTLResourceUsageRead];
                    }
                    if (!SkipBlasUseResources() && !provider.hardware.blasHandles.empty()) {
                        for (auto h : provider.hardware.blasHandles) {
                            id<MTLAccelerationStructure> blasObj = (__bridge id<MTLAccelerationStructure>)h;
                            if (blasObj) {
                                [encoder useResource:blasObj usage:MTLResourceUsageRead];
                            }
                        }
                    }
                    if (provider.hardware.instanceBuffer) {
                        [encoder useResource:provider.hardware.instanceBuffer usage:MTLResourceUsageRead];
                    }
                    if (provider.hardware.instanceUserIDBuffer) {
                        [encoder useResource:provider.hardware.instanceUserIDBuffer usage:MTLResourceUsageRead];
                    }
                }
                if (auto argumentEncoder = pipelines.materialResourcesArgumentEncoder();
                    argumentEncoder) {
                    NSUInteger encodedLength = argumentEncoder.encodedLength;
                    if (encodedLength > 0u) {
                        if (!m_materialResourcesArgumentBuffer ||
                            m_materialResourcesArgumentBuffer.length < encodedLength) {
                            m_materialResourcesArgumentBuffer =
                                [m_device newBufferWithLength:encodedLength
                                                      options:MTLResourceStorageModeShared];
                        }
                        if (m_materialResourcesArgumentBuffer) {
                            std::memset([m_materialResourcesArgumentBuffer contents], 0, encodedLength);
                            [argumentEncoder setArgumentBuffer:m_materialResourcesArgumentBuffer offset:0];
                            if (!materialTextures.empty()) {
                                NSUInteger textureCount =
                                    std::min<NSUInteger>(materialTextures.size(),
                                                         PathTracerShaderTypes::kMaxMaterialTextures);
                                [argumentEncoder setTextures:materialTextures.data()
                                                   withRange:NSMakeRange(0, textureCount)];
                            }
                            if (!materialSamplers.empty()) {
                                NSUInteger samplerCount =
                                    std::min<NSUInteger>(materialSamplers.size(), kMaxMaterialSamplers);
                                [argumentEncoder setSamplerStates:materialSamplers.data()
                                                        withRange:NSMakeRange(PathTracerShaderTypes::kMaxMaterialTextures,
                                                                              samplerCount)];
                            }
                            [argumentEncoder setBuffer:BufferOrDummy(m_restirState.pathGuiding, m_dummyBuffer)
                                                offset:0
                                               atIndex:PathTracerShaderTypes::kMaterialResourcePathGuidingStateIndex];
                            [argumentEncoder setBuffer:BufferOrDummy(m_restirState.restirPt, m_dummyBuffer)
                                                offset:0
                                               atIndex:PathTracerShaderTypes::kMaterialResourceRestirPtStateIndex];
                            [argumentEncoder setBuffer:BufferOrDummy(m_restirState.radianceCache, m_dummyBuffer)
                                                offset:0
                                               atIndex:PathTracerShaderTypes::kMaterialResourceRadianceCacheStateIndex];
                            [encoder setBuffer:m_materialResourcesArgumentBuffer offset:0 atIndex:23];
                            if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                                if (m_restirState.pathGuiding) {
                                    [encoder useResource:m_restirState.pathGuiding
                                                   usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                                }
                                if (m_restirState.restirPt) {
                                    [encoder useResource:m_restirState.restirPt
                                                   usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                                }
                                if (m_restirState.radianceCache) {
                                    [encoder useResource:m_restirState.radianceCache
                                                   usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                                }
                            }
                            if (!SkipTextureUseResources() &&
                                [encoder respondsToSelector:@selector(useResource:usage:)]) {
                                for (id<MTLTexture> texture : materialTextures) {
                                    if (texture) {
                                        [encoder useResource:texture usage:MTLResourceUsageRead];
                                    }
                                }
                            }
                        }
                    }
                } else {
                    [encoder setBuffer:nil offset:0 atIndex:23];
                }
                [encoder setBuffer:sampleUniformBuffer offset:0 atIndex:0];
                [encoder setBuffer:m_wavefront.activeRays[currentQueueIndex] offset:0 atIndex:1];
                [encoder setBuffer:m_wavefront.hitRecords offset:0 atIndex:2];
                [encoder setBuffer:m_wavefront.activeRays[nextQueueIndex] offset:0 atIndex:3];
                [encoder setBuffer:m_wavefront.shadowRays offset:0 atIndex:4];
                [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:5];
                [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:6];
                [encoder setBuffer:BufferOrDummy(m_statsBuffer, m_dummyBuffer) offset:0 atIndex:7];
                [encoder setTexture:scene.environmentTexture() atIndex:0];
                [encoder setBuffer:BufferOrDummy(scene.materialBuffer(), m_dummyBuffer) offset:0 atIndex:8];
                [encoder setBuffer:BufferOrDummy(scene.rectangleBuffer(), m_dummyBuffer) offset:0 atIndex:9];
                [encoder setBuffer:BufferOrDummy(m_debugBuffer, m_dummyBuffer) offset:0 atIndex:10];
                [encoder setBuffer:BufferOrDummy(scene.environmentConditionalAliasBuffer(), m_dummyBuffer)
                           offset:0
                          atIndex:11];
                [encoder setBuffer:BufferOrDummy(scene.environmentMarginalAliasBuffer(), m_dummyBuffer)
                           offset:0
                          atIndex:12];
                [encoder setBuffer:BufferOrDummy(scene.environmentPdfBuffer(), m_dummyBuffer)
                           offset:0
                          atIndex:13];
                [encoder setBuffer:BufferOrDummy(scene.meshInfoBuffer(), m_dummyBuffer) offset:0 atIndex:14];
                [encoder setBuffer:BufferOrDummy(scene.meshVertexBuffer(), m_dummyBuffer) offset:0 atIndex:15];
                [encoder setBuffer:BufferOrDummy(scene.meshIndexBuffer(), m_dummyBuffer) offset:0 atIndex:16];
                [encoder setBuffer:BufferOrDummy(scene.materialTextureInfoBuffer(), m_dummyBuffer)
                           offset:0
                          atIndex:17];
                [encoder setBuffer:BufferOrDummy(scene.sphereBuffer(), m_dummyBuffer) offset:0 atIndex:19];
                [encoder setBuffer:BufferOrDummy(scene.triangleBuffer(), m_dummyBuffer) offset:0 atIndex:20];
                [encoder setBuffer:BufferOrDummy(provider.software.tlasNodes, m_dummyBuffer) offset:0 atIndex:21];
                [encoder setBuffer:BufferOrDummy(provider.software.tlasPrimitiveIndices, m_dummyBuffer)
                           offset:0
                          atIndex:22];
                [encoder setBuffer:BufferOrDummy(provider.software.blasNodes, m_dummyBuffer) offset:0 atIndex:24];
                [encoder setBuffer:BufferOrDummy(provider.software.blasPrimitiveIndices, m_dummyBuffer)
                           offset:0
                          atIndex:25];
                [encoder setBuffer:BufferOrDummy(provider.software.instanceInfoBuffer, m_dummyBuffer)
                           offset:0
                          atIndex:26];
                [encoder setBuffer:BufferOrDummy(provider.software.nodes, m_dummyBuffer) offset:0 atIndex:27];
                [encoder setBuffer:BufferOrDummy(provider.software.primitiveIndices, m_dummyBuffer)
                           offset:0
                          atIndex:28];
                [encoder setBuffer:BufferOrDummy(provider.hardware.instanceUserIDBuffer, m_dummyBuffer)
                           offset:0
                          atIndex:29];
                [encoder setBuffer:BufferOrDummy(scene.emissivePrimitivesBuffer(), m_dummyBuffer)
                           offset:0
                          atIndex:30];
                const MTLSize threadsPerThreadgroup =
                    PreferredLinearThreadgroupSize(pipelines.wavefrontRestirDiDirectLighting());
                [encoder dispatchThreads:fullGrid threadsPerThreadgroup:threadsPerThreadgroup];
                [encoder endEncoding];
            }

            {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::Shade, @"Wavefront Shade");
                [encoder setComputePipelineState:pipelines.wavefrontShade()];
                if ([encoder respondsToSelector:@selector(setAccelerationStructure:atBufferIndex:)]) {
                    [encoder setAccelerationStructure:(canHardwareIntersect ? provider.hardware.tlas : nil)
                                        atBufferIndex:18];
                }
                if (canHardwareIntersect &&
                    [encoder respondsToSelector:@selector(useResource:usage:)]) {
                    if (provider.hardware.tlas) {
                        [encoder useResource:provider.hardware.tlas usage:MTLResourceUsageRead];
                    }
                    if (!SkipBlasUseResources() && provider.hardware.blas) {
                        [encoder useResource:provider.hardware.blas usage:MTLResourceUsageRead];
                    }
                    if (!SkipBlasUseResources() && !provider.hardware.blasHandles.empty()) {
                        for (auto h : provider.hardware.blasHandles) {
                            id<MTLAccelerationStructure> blasObj = (__bridge id<MTLAccelerationStructure>)h;
                            if (blasObj) {
                                [encoder useResource:blasObj usage:MTLResourceUsageRead];
                            }
                        }
                    }
                    if (provider.hardware.instanceBuffer) {
                        [encoder useResource:provider.hardware.instanceBuffer usage:MTLResourceUsageRead];
                    }
                    if (provider.hardware.instanceUserIDBuffer) {
                        [encoder useResource:provider.hardware.instanceUserIDBuffer usage:MTLResourceUsageRead];
                    }
                }
                if (auto argumentEncoder = pipelines.materialResourcesArgumentEncoder();
                    argumentEncoder) {
                    NSUInteger encodedLength = argumentEncoder.encodedLength;
                    if (encodedLength > 0u) {
                        if (!m_materialResourcesArgumentBuffer ||
                            m_materialResourcesArgumentBuffer.length < encodedLength) {
                            m_materialResourcesArgumentBuffer =
                                [m_device newBufferWithLength:encodedLength
                                                      options:MTLResourceStorageModeShared];
                        }
                        if (m_materialResourcesArgumentBuffer) {
                            std::memset([m_materialResourcesArgumentBuffer contents], 0, encodedLength);
                            [argumentEncoder setArgumentBuffer:m_materialResourcesArgumentBuffer offset:0];
                            if (!materialTextures.empty()) {
                                NSUInteger textureCount =
                                    std::min<NSUInteger>(materialTextures.size(),
                                                         PathTracerShaderTypes::kMaxMaterialTextures);
                                [argumentEncoder setTextures:materialTextures.data()
                                                   withRange:NSMakeRange(0, textureCount)];
                            }
                            if (!materialSamplers.empty()) {
                                NSUInteger samplerCount =
                                    std::min<NSUInteger>(materialSamplers.size(), kMaxMaterialSamplers);
                                [argumentEncoder setSamplerStates:materialSamplers.data()
                                                        withRange:NSMakeRange(PathTracerShaderTypes::kMaxMaterialTextures,
                                                                              samplerCount)];
                            }
                            [argumentEncoder setBuffer:BufferOrDummy(m_restirState.pathGuiding, m_dummyBuffer)
                                                offset:0
                                               atIndex:PathTracerShaderTypes::kMaterialResourcePathGuidingStateIndex];
                            [argumentEncoder setBuffer:BufferOrDummy(m_restirState.restirPt, m_dummyBuffer)
                                                offset:0
                                               atIndex:PathTracerShaderTypes::kMaterialResourceRestirPtStateIndex];
                            [argumentEncoder setBuffer:BufferOrDummy(m_restirState.radianceCache, m_dummyBuffer)
                                                offset:0
                                               atIndex:PathTracerShaderTypes::kMaterialResourceRadianceCacheStateIndex];
                            [encoder setBuffer:m_materialResourcesArgumentBuffer offset:0 atIndex:23];
                            if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                                if (m_restirState.pathGuiding) {
                                    [encoder useResource:m_restirState.pathGuiding
                                                   usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                                }
                                if (m_restirState.restirPt) {
                                    [encoder useResource:m_restirState.restirPt
                                                   usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                                }
                                if (m_restirState.radianceCache) {
                                    [encoder useResource:m_restirState.radianceCache
                                                   usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                                }
                            }
                            if (!SkipTextureUseResources() &&
                                [encoder respondsToSelector:@selector(useResource:usage:)]) {
                                for (id<MTLTexture> texture : materialTextures) {
                                    if (texture) {
                                        [encoder useResource:texture usage:MTLResourceUsageRead];
                                    }
                                }
                            }
                        }
                    }
                } else {
                    [encoder setBuffer:nil offset:0 atIndex:23];
                }
                [encoder setBuffer:sampleUniformBuffer offset:0 atIndex:0];
                [encoder setBuffer:m_wavefront.activeRays[currentQueueIndex] offset:0 atIndex:1];
                [encoder setBuffer:m_wavefront.hitRecords offset:0 atIndex:2];
                [encoder setBuffer:m_wavefront.activeRays[nextQueueIndex] offset:0 atIndex:3];
                [encoder setBuffer:m_wavefront.shadowRays offset:0 atIndex:4];
                [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:5];
                [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:6];
                [encoder setBuffer:BufferOrDummy(m_statsBuffer, m_dummyBuffer) offset:0 atIndex:7];
                [encoder setTexture:scene.environmentTexture() atIndex:0];
                [encoder setBuffer:BufferOrDummy(scene.materialBuffer(), m_dummyBuffer) offset:0 atIndex:8];
                [encoder setBuffer:BufferOrDummy(scene.rectangleBuffer(), m_dummyBuffer) offset:0 atIndex:9];
                [encoder setBuffer:BufferOrDummy(m_debugBuffer, m_dummyBuffer) offset:0 atIndex:10];
                [encoder setBuffer:BufferOrDummy(scene.environmentConditionalAliasBuffer(), m_dummyBuffer)
                           offset:0
                          atIndex:11];
                [encoder setBuffer:BufferOrDummy(scene.environmentMarginalAliasBuffer(), m_dummyBuffer)
                           offset:0
                          atIndex:12];
                [encoder setBuffer:BufferOrDummy(scene.environmentPdfBuffer(), m_dummyBuffer)
                           offset:0
                          atIndex:13];
                [encoder setBuffer:BufferOrDummy(scene.meshInfoBuffer(), m_dummyBuffer) offset:0 atIndex:14];
                [encoder setBuffer:BufferOrDummy(scene.meshVertexBuffer(), m_dummyBuffer) offset:0 atIndex:15];
                [encoder setBuffer:BufferOrDummy(scene.meshIndexBuffer(), m_dummyBuffer) offset:0 atIndex:16];
                [encoder setBuffer:BufferOrDummy(scene.materialTextureInfoBuffer(), m_dummyBuffer)
                           offset:0
                          atIndex:17];
                [encoder setBuffer:BufferOrDummy(scene.sphereBuffer(), m_dummyBuffer) offset:0 atIndex:19];
                [encoder setBuffer:BufferOrDummy(scene.triangleBuffer(), m_dummyBuffer) offset:0 atIndex:20];
                [encoder setBuffer:BufferOrDummy(provider.software.tlasNodes, m_dummyBuffer) offset:0 atIndex:21];
                [encoder setBuffer:BufferOrDummy(provider.software.tlasPrimitiveIndices, m_dummyBuffer)
                           offset:0
                          atIndex:22];
                [encoder setBuffer:BufferOrDummy(provider.software.blasNodes, m_dummyBuffer) offset:0 atIndex:24];
                [encoder setBuffer:BufferOrDummy(provider.software.blasPrimitiveIndices, m_dummyBuffer)
                           offset:0
                          atIndex:25];
                [encoder setBuffer:BufferOrDummy(provider.software.instanceInfoBuffer, m_dummyBuffer)
                           offset:0
                          atIndex:26];
                [encoder setBuffer:BufferOrDummy(provider.software.nodes, m_dummyBuffer) offset:0 atIndex:27];
                [encoder setBuffer:BufferOrDummy(provider.software.primitiveIndices, m_dummyBuffer)
                           offset:0
                          atIndex:28];
                [encoder setBuffer:BufferOrDummy(provider.hardware.instanceUserIDBuffer, m_dummyBuffer)
                           offset:0
                          atIndex:29];
                [encoder setBuffer:BufferOrDummy(scene.emissivePrimitivesBuffer(), m_dummyBuffer)
                           offset:0
                          atIndex:30];
                const MTLSize threadsPerThreadgroup =
                    PreferredLinearThreadgroupSize(pipelines.wavefrontShade());
                [encoder dispatchThreads:fullGrid threadsPerThreadgroup:threadsPerThreadgroup];
                [encoder endEncoding];
            }

            if (canHardwareShadow) {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::ShadowQueue, @"Wavefront Shadow (HW)");
                [encoder setComputePipelineState:pipelines.wavefrontShadowHardware()];
                if ([encoder respondsToSelector:@selector(setAccelerationStructure:atBufferIndex:)]) {
                    [encoder setAccelerationStructure:provider.hardware.tlas atBufferIndex:1];
                }
                if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                    [encoder useResource:provider.hardware.tlas usage:MTLResourceUsageRead];
                    if (!SkipBlasUseResources() && provider.hardware.blas) {
                        [encoder useResource:provider.hardware.blas usage:MTLResourceUsageRead];
                    }
                    if (!SkipBlasUseResources() && !provider.hardware.blasHandles.empty()) {
                        for (auto h : provider.hardware.blasHandles) {
                            id<MTLAccelerationStructure> blasObj = (__bridge id<MTLAccelerationStructure>)h;
                            if (blasObj) {
                                [encoder useResource:blasObj usage:MTLResourceUsageRead];
                            }
                        }
                    }
                    if (provider.hardware.instanceBuffer) {
                        [encoder useResource:provider.hardware.instanceBuffer usage:MTLResourceUsageRead];
                    }
                    if (provider.hardware.instanceUserIDBuffer) {
                        [encoder useResource:provider.hardware.instanceUserIDBuffer usage:MTLResourceUsageRead];
                    }
                }
                [encoder setBuffer:sampleUniformBuffer offset:0 atIndex:0];
                [encoder setTexture:scene.environmentTexture() atIndex:0];
                [encoder setBuffer:m_wavefront.shadowRays offset:0 atIndex:2];
                [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:3];
                [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:4];
                [encoder setBuffer:BufferOrDummy(m_statsBuffer, m_dummyBuffer) offset:0 atIndex:5];
                [encoder setBuffer:BufferOrDummy(scene.meshInfoBuffer(), m_dummyBuffer) offset:0 atIndex:6];
                [encoder setBuffer:BufferOrDummy(scene.triangleBuffer(), m_dummyBuffer) offset:0 atIndex:7];
                [encoder setBuffer:BufferOrDummy(scene.meshVertexBuffer(), m_dummyBuffer) offset:0 atIndex:8];
                [encoder setBuffer:BufferOrDummy(scene.meshIndexBuffer(), m_dummyBuffer) offset:0 atIndex:9];
                [encoder setBuffer:BufferOrDummy(provider.hardware.instanceUserIDBuffer, m_dummyBuffer) offset:0 atIndex:10];
                [encoder setBuffer:BufferOrDummy(scene.sphereBuffer(), m_dummyBuffer) offset:0 atIndex:11];
                [encoder setBuffer:BufferOrDummy(scene.rectangleBuffer(), m_dummyBuffer) offset:0 atIndex:12];
                [encoder setBuffer:BufferOrDummy(provider.software.nodes, m_dummyBuffer) offset:0 atIndex:13];
                [encoder setBuffer:BufferOrDummy(provider.software.primitiveIndices, m_dummyBuffer) offset:0 atIndex:14];
                [encoder setBuffer:BufferOrDummy(scene.materialBuffer(), m_dummyBuffer) offset:0 atIndex:15];
                [encoder setBuffer:BufferOrDummy(provider.software.tlasNodes, m_dummyBuffer) offset:0 atIndex:16];
                [encoder setBuffer:BufferOrDummy(provider.software.tlasPrimitiveIndices, m_dummyBuffer)
                           offset:0
                          atIndex:17];
                [encoder setBuffer:BufferOrDummy(provider.software.blasNodes, m_dummyBuffer) offset:0 atIndex:18];
                [encoder setBuffer:BufferOrDummy(provider.software.blasPrimitiveIndices, m_dummyBuffer)
                           offset:0
                          atIndex:19];
                [encoder setBuffer:BufferOrDummy(provider.software.instanceInfoBuffer, m_dummyBuffer)
                           offset:0
                          atIndex:20];
                const MTLSize threadsPerThreadgroup =
                    PreferredLinearThreadgroupSize(pipelines.wavefrontShadowHardware());
                [encoder dispatchThreads:fullGrid threadsPerThreadgroup:threadsPerThreadgroup];
                [encoder endEncoding];
            } else {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::ShadowQueue, @"Wavefront Shadow");
                [encoder setComputePipelineState:pipelines.wavefrontShadow()];
                [encoder setBuffer:sampleUniformBuffer offset:0 atIndex:0];
                [encoder setTexture:scene.environmentTexture() atIndex:0];
                [encoder setBuffer:m_wavefront.shadowRays offset:0 atIndex:1];
                [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:2];
                [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:3];
                [encoder setBuffer:BufferOrDummy(m_statsBuffer, m_dummyBuffer) offset:0 atIndex:4];
                [encoder setBuffer:BufferOrDummy(scene.sphereBuffer(), m_dummyBuffer) offset:0 atIndex:5];
                [encoder setBuffer:BufferOrDummy(scene.rectangleBuffer(), m_dummyBuffer) offset:0 atIndex:6];
                [encoder setBuffer:BufferOrDummy(scene.triangleBuffer(), m_dummyBuffer) offset:0 atIndex:7];
                [encoder setBuffer:BufferOrDummy(provider.software.tlasNodes, m_dummyBuffer) offset:0 atIndex:8];
                [encoder setBuffer:BufferOrDummy(provider.software.tlasPrimitiveIndices, m_dummyBuffer) offset:0 atIndex:9];
                [encoder setBuffer:BufferOrDummy(provider.software.instanceInfoBuffer, m_dummyBuffer) offset:0 atIndex:10];
                [encoder setBuffer:BufferOrDummy(provider.software.blasNodes, m_dummyBuffer) offset:0 atIndex:11];
                [encoder setBuffer:BufferOrDummy(provider.software.blasPrimitiveIndices, m_dummyBuffer) offset:0 atIndex:12];
                [encoder setBuffer:BufferOrDummy(provider.software.nodes, m_dummyBuffer) offset:0 atIndex:13];
                [encoder setBuffer:BufferOrDummy(provider.software.primitiveIndices, m_dummyBuffer) offset:0 atIndex:14];
                [encoder setBuffer:BufferOrDummy(scene.materialBuffer(), m_dummyBuffer) offset:0 atIndex:15];
                const MTLSize threadsPerThreadgroup =
                    PreferredLinearThreadgroupSize(pipelines.wavefrontShadow());
                [encoder dispatchThreads:fullGrid threadsPerThreadgroup:threadsPerThreadgroup];
                [encoder endEncoding];
            }

            {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::ShadowQueue, @"Wavefront Swap Queues");
                [encoder setComputePipelineState:pipelines.wavefrontSwapQueues()];
                [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:0];
                [encoder setBuffer:BufferOrDummy(m_statsBuffer, m_dummyBuffer) offset:0 atIndex:1];
                [encoder dispatchThreads:resetThreads threadsPerThreadgroup:resetThreads];
                [encoder endEncoding];
            }

            std::swap(currentQueueIndex, nextQueueIndex);
        }

        {
            id<MTLComputeCommandEncoder> encoder =
                makeEncoder(GpuTimingBucket::ShadowQueue, @"Wavefront Accumulate");
            [encoder setComputePipelineState:pipelines.wavefrontAccumulate()];
            [encoder setTexture:accumulation.radianceSum() atIndex:0];
            [encoder setTexture:accumulation.sampleCountTexture() atIndex:1];
            [encoder setTexture:accumulation.albedoBuffer() atIndex:2];
            [encoder setTexture:accumulation.normalBuffer() atIndex:3];
            [encoder setTexture:accumulation.positionBuffer() atIndex:4];
            [encoder setTexture:accumulation.materialFeatureBuffer() atIndex:5];
            [encoder setTexture:accumulation.motionVectorBuffer() atIndex:6];
            [encoder setBuffer:sampleUniformBuffer offset:0 atIndex:0];
            [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:1];
            [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:2];
            [encoder setBuffer:BufferOrDummy(m_statsBuffer, m_dummyBuffer) offset:0 atIndex:3];
            const MTLSize threadsPerThreadgroup =
                PreferredLinearThreadgroupSize(pipelines.wavefrontAccumulate());
            [encoder dispatchThreads:fullGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

        accumulation.incrementFrame();
    }

    return samplesThisFrame;
}

void RenderLoop::encodeDenoising(MTLCommandBufferHandle commandBuffer,
                                 Accumulation& accumulation,
                                 const RenderSettings& settings,
                                 uint32_t frameIndex) {
    (void)commandBuffer;
    // Denoising is only performed if:
    // 1. Denoiser context is available (not nullptr)
    // 2. Denoising is enabled in settings
    // 3. Denoiser is ready for use
    // 4. Current frame index matches denoising frequency
    if (!m_denoiser || !settings.denoiseEnabled || !m_denoiser->isReady()) {
        m_hasDenoisedOutput = false;
        return;
    }

    // Progressive denoising: only denoise every N frames
    uint32_t frequency = std::max(1u, settings.denoiseFrequency);
    if ((frameIndex % frequency) != 0) {
        return;
    }

    if (frameIndex < m_nextDenoiseAttemptFrame) {
        return;
    }

    const uint64_t recommendedWorkingSet = RecommendedMaxWorkingSetBytes(m_device);
    const uint64_t currentAllocated = CurrentAllocatedSizeBytes(m_device);
    if (recommendedWorkingSet > 0u) {
        const uint64_t suspendThreshold = ApplyPercent(recommendedWorkingSet, 102u, 100u);
        const uint64_t resumeThreshold = ApplyPercent(recommendedWorkingSet, 98u, 100u);
        if (currentAllocated > suspendThreshold) {
            if (m_denoiseFailureStreak == 0u) {
                NSLog(@"[Denoise] Deferring OIDN while Metal residency exceeds denoise pressure threshold "
                      "(currentAllocatedSize=%.2f MiB threshold=%.2f MiB recommendedMaxWorkingSetSize=%.2f MiB)",
                      static_cast<double>(currentAllocated) / (1024.0 * 1024.0),
                      static_cast<double>(suspendThreshold) / (1024.0 * 1024.0),
                      static_cast<double>(recommendedWorkingSet) / (1024.0 * 1024.0));
            }
            m_hasDenoisedOutput = false;
            m_denoiseFailureStreak = std::min(m_denoiseFailureStreak + 1u, 6u);
            m_nextDenoiseAttemptFrame = frameIndex + std::max(30u, frequency * (1u << m_denoiseFailureStreak));
            return;
        }
        if (m_denoiseFailureStreak > 0u && currentAllocated > resumeThreshold) {
            m_hasDenoisedOutput = false;
            m_nextDenoiseAttemptFrame = frameIndex + std::max(30u, frequency);
            return;
        }
        if (m_denoiseFailureStreak > 0u && currentAllocated <= resumeThreshold) {
            NSLog(@"[Denoise] Resuming OIDN after Metal residency dropped below denoise resume threshold "
                  "(currentAllocatedSize=%.2f MiB threshold=%.2f MiB recommendedMaxWorkingSetSize=%.2f MiB)",
                  static_cast<double>(currentAllocated) / (1024.0 * 1024.0),
                  static_cast<double>(resumeThreshold) / (1024.0 * 1024.0),
                  static_cast<double>(recommendedWorkingSet) / (1024.0 * 1024.0));
            m_nextDenoiseAttemptFrame = 0;
            m_denoiseFailureStreak = 0;
        }
    }

    // Get the accumulated noisy image from presentation buffer
    MTLTextureHandle colorInput = accumulation.present();
    if (!colorInput) {
        NSLog(@"[Denoise] No presentation buffer available");
        return;
    }

    // Get optional auxiliary buffers (albedo and normal)
    MTLTextureHandle albedoInput = settings.denoiseUseAlbedo ? accumulation.albedoBuffer() : nullptr;
    MTLTextureHandle normalInput = settings.denoiseUseNormal ? accumulation.normalBuffer() : nullptr;

    // Output buffer for denoised result
    MTLTextureHandle colorOutput = accumulation.denoisedBuffer();
    if (!colorOutput) {
        NSLog(@"[Denoise] No denoised output buffer available");
        m_hasDenoisedOutput = false;
        return;
    }

    // Determine filter type
    DenoiserContext::FilterType filterType = (settings.denoiseFilterType == 1)
        ? DenoiserContext::FilterType::RTLightmap
        : DenoiserContext::FilterType::RT;

    // Execute denoising
    if (!m_denoiser->denoise(colorInput, albedoInput, normalInput, colorOutput, filterType)) {
        NSLog(@"[Denoise] Denoising failed: %s", m_denoiser->lastError().c_str());
        m_hasDenoisedOutput = false;
        m_denoiseFailureStreak = std::min(m_denoiseFailureStreak + 1u, 6u);
        m_nextDenoiseAttemptFrame = frameIndex + std::max(30u, frequency * (1u << m_denoiseFailureStreak));
        return;
    }

    m_lastDenoisedFrame = frameIndex;
    m_nextDenoiseAttemptFrame = 0;
    m_denoiseFailureStreak = 0;
    m_hasDenoisedOutput = true;
}

void RenderLoop::encodeSvgfDenoising(MTLCommandBufferHandle commandBuffer,
                                     Accumulation& accumulation,
                                     const Pipelines& pipelines,
                                     const RenderSettings& settings) {
    if (!settings.svgfDenoiseEnabled || !pipelines.svgfTemporal() || !pipelines.svgfAtrous()) {
        return;
    }
    MTLTextureHandle present = accumulation.present();
    MTLTextureHandle albedo = accumulation.albedoBuffer();
    MTLTextureHandle normal = accumulation.normalBuffer();
    MTLTextureHandle position = accumulation.positionBuffer();
    MTLTextureHandle materialFeatures = accumulation.materialFeatureBuffer();
    MTLTextureHandle motionVectors = accumulation.motionVectorBuffer();
    MTLTextureHandle denoised = accumulation.denoisedBuffer();
    MTLTextureHandle scratch = accumulation.svgfScratchBuffer();
    MTLTextureHandle momentsA = accumulation.svgfMomentsBufferA();
    MTLTextureHandle momentsB = accumulation.svgfMomentsBufferB();
    MTLTextureHandle guideA = accumulation.svgfGuideBufferA();
    MTLTextureHandle guideB = accumulation.svgfGuideBufferB();
    MTLTextureHandle colorA = accumulation.svgfColorBufferA();
    MTLTextureHandle colorB = accumulation.svgfColorBufferB();
    if (!commandBuffer || !present || !albedo || !normal || !position ||
        !materialFeatures || !motionVectors || !denoised || !scratch ||
        !momentsA || !momentsB || !guideA || !guideB || !colorA || !colorB ||
        !m_hasCurrentPathtraceUniforms) {
        m_hasDenoisedOutput = false;
        return;
    }

    const uint32_t width = static_cast<uint32_t>(
        std::min({present.width, albedo.width, normal.width, position.width,
                  materialFeatures.width, motionVectors.width, denoised.width,
                  scratch.width, momentsA.width, momentsB.width, guideA.width, guideB.width,
                  colorA.width, colorB.width}));
    const uint32_t height = static_cast<uint32_t>(
        std::min({present.height, albedo.height, normal.height, position.height,
                  materialFeatures.height, motionVectors.height, denoised.height,
                  scratch.height, momentsA.height, momentsB.height, guideA.height, guideB.height,
                  colorA.height, colorB.height}));
    if (width == 0u || height == 0u) {
        m_hasDenoisedOutput = false;
        return;
    }

    MTLTextureHandle previousMoments = (m_frameCounter & 1u) ? momentsB : momentsA;
    MTLTextureHandle currentMoments = (m_frameCounter & 1u) ? momentsA : momentsB;
    MTLTextureHandle previousGuide = (m_frameCounter & 1u) ? guideB : guideA;
    MTLTextureHandle currentGuide = (m_frameCounter & 1u) ? guideA : guideB;
    MTLTextureHandle previousColor = (m_frameCounter & 1u) ? colorB : colorA;
    MTLTextureHandle currentColor = (m_frameCounter & 1u) ? colorA : colorB;
    const PathtraceUniforms& previousCamera =
        m_hasPreviousSvgfCameraUniforms ? m_previousSvgfCameraUniforms : m_currentPathtraceUniforms;
    {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            m_hasDenoisedOutput = false;
            return;
        }
        encoder.label = @"SVGF Temporal Moments";
        [encoder setComputePipelineState:pipelines.svgfTemporal()];
        SvgfDenoiseUniforms uniforms{};
        uniforms.width = width;
        uniforms.height = height;
        uniforms.step = 1u;
        uniforms.reprojectionEnabled = m_hasPreviousSvgfCameraUniforms ? 1u : 0u;
        uniforms.normalSigma = std::max(settings.svgfNormalSigma, 1.0e-4f);
        uniforms.albedoSigma = std::max(settings.svgfAlbedoSigma, 1.0e-4f);
        uniforms.luminanceSigma = std::max(settings.svgfLuminanceSigma, 1.0e-4f);
        uniforms.historyValid = m_hasPreviousSvgfCameraUniforms ? 1u : 0u;
        uniforms.currentCameraOrigin = simd_make_float4(m_currentPathtraceUniforms.cameraOrigin, 0.0f);
        uniforms.previousCameraOrigin = simd_make_float4(previousCamera.cameraOrigin, 0.0f);
        uniforms.previousLowerLeftCorner = simd_make_float4(previousCamera.lowerLeftCorner, 0.0f);
        uniforms.previousHorizontal = simd_make_float4(previousCamera.horizontal, 0.0f);
        uniforms.previousVertical = simd_make_float4(previousCamera.vertical, 0.0f);
        [encoder setBytes:&uniforms length:sizeof(SvgfDenoiseUniforms) atIndex:0];
        [encoder setTexture:present atIndex:0];
        [encoder setTexture:previousMoments atIndex:1];
        [encoder setTexture:currentMoments atIndex:2];
        [encoder setTexture:position atIndex:3];
        [encoder setTexture:normal atIndex:4];
        [encoder setTexture:previousGuide atIndex:5];
        [encoder setTexture:currentGuide atIndex:6];
        [encoder setTexture:previousColor atIndex:7];
        [encoder setTexture:currentColor atIndex:8];
        [encoder setTexture:materialFeatures atIndex:9];
        [encoder setTexture:motionVectors atIndex:10];

        const MTLSize threadsPerThreadgroup =
            PreferredLinearThreadgroupSize(pipelines.svgfTemporal());
        MTLSize threadsPerGrid = MTLSizeMake(width * height, 1u, 1u);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
    }

    const uint32_t iterations = std::clamp(settings.svgfAtrousIterations, 1u, 5u);
    MTLTextureHandle source = currentColor;

    for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const bool finalIteration = (iteration + 1u == iterations);
        MTLTextureHandle target = finalIteration
            ? denoised
            : ((source == denoised) ? scratch : denoised);
        if (target == source) {
            m_hasDenoisedOutput = false;
            return;
        }

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            m_hasDenoisedOutput = false;
            return;
        }
        encoder.label = [NSString stringWithFormat:@"SVGF A-Trous Filter %u", iteration];
        [encoder setComputePipelineState:pipelines.svgfAtrous()];
        SvgfDenoiseUniforms uniforms{};
        uniforms.width = width;
        uniforms.height = height;
        uniforms.step = 1u << iteration;
        uniforms.reprojectionEnabled = m_hasPreviousSvgfCameraUniforms ? 1u : 0u;
        uniforms.normalSigma = std::max(settings.svgfNormalSigma, 1.0e-4f);
        uniforms.albedoSigma = std::max(settings.svgfAlbedoSigma, 1.0e-4f);
        uniforms.luminanceSigma = std::max(settings.svgfLuminanceSigma, 1.0e-4f);
        uniforms.historyValid = m_hasPreviousSvgfCameraUniforms ? 1u : 0u;
        uniforms.currentCameraOrigin = simd_make_float4(m_currentPathtraceUniforms.cameraOrigin, 0.0f);
        uniforms.previousCameraOrigin = simd_make_float4(previousCamera.cameraOrigin, 0.0f);
        uniforms.previousLowerLeftCorner = simd_make_float4(previousCamera.lowerLeftCorner, 0.0f);
        uniforms.previousHorizontal = simd_make_float4(previousCamera.horizontal, 0.0f);
        uniforms.previousVertical = simd_make_float4(previousCamera.vertical, 0.0f);
        [encoder setBytes:&uniforms length:sizeof(SvgfDenoiseUniforms) atIndex:0];
        [encoder setTexture:source atIndex:0];
        [encoder setTexture:albedo atIndex:1];
        [encoder setTexture:normal atIndex:2];
        [encoder setTexture:target atIndex:3];
        [encoder setTexture:currentMoments atIndex:4];

        const MTLSize threadsPerThreadgroup =
            PreferredLinearThreadgroupSize(pipelines.svgfAtrous());
        MTLSize threadsPerGrid = MTLSizeMake(width * height, 1u, 1u);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];

        source = target;
    }

    m_lastDenoisedFrame = m_frameCounter;
    m_hasDenoisedOutput = true;
    m_previousSvgfCameraUniforms = m_currentPathtraceUniforms;
    m_hasPreviousSvgfCameraUniforms = true;
}

void RenderLoop::encodePresentation(MTLCommandBufferHandle commandBuffer,
                                    const Accumulation& accumulation,
                                    const Pipelines& pipelines) {
    
    id<MTLComputeCommandEncoder> encoder = nil;
    if (m_gpuTiming.active &&
        m_gpuTiming.nextSample + 2u <= m_gpuTiming.sampleCapacity &&
        [commandBuffer respondsToSelector:@selector(computeCommandEncoderWithDescriptor:)]) {
        MTLComputePassDescriptor* passDescriptor = [MTLComputePassDescriptor computePassDescriptor];
        MTLComputePassSampleBufferAttachmentDescriptor* attachment =
            passDescriptor.sampleBufferAttachments[0];
        attachment.sampleBuffer = (id<MTLCounterSampleBuffer>)m_gpuTiming.sampleBuffer;
        const uint32_t startSample = m_gpuTiming.nextSample++;
        const uint32_t endSample = m_gpuTiming.nextSample++;
        attachment.startOfEncoderSampleIndex = startSample;
        attachment.endOfEncoderSampleIndex = endSample;
        m_gpuTiming.ranges.push_back({GpuTimingBucket::Presentation, startSample, endSample});
        encoder = [commandBuffer computeCommandEncoderWithDescriptor:passDescriptor];
    } else {
        encoder = [commandBuffer computeCommandEncoder];
    }
    encoder.label = @"Path Tracer Present";
    [encoder setComputePipelineState:pipelines.present()];
    [encoder setTexture:accumulation.radianceSum() atIndex:0];
    [encoder setTexture:accumulation.sampleCountTexture() atIndex:1];
    [encoder setTexture:accumulation.present() atIndex:2];

    NSUInteger threadWidth = pipelines.present().threadExecutionWidth;
    NSUInteger threadHeight = std::max<NSUInteger>(
        1, pipelines.present().maxTotalThreadsPerThreadgroup / std::max<NSUInteger>(threadWidth, 1));
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadWidth, threadHeight, 1);
    MTLSize threadsPerGrid = MTLSizeMake(accumulation.present().width,
                                        accumulation.present().height, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
}

void RenderLoop::encodeDisplay(MTLRenderCommandEncoderHandle renderEncoder,
                               const Accumulation& accumulation,
                               const Pipelines& pipelines,
                               const RenderSettings& settings) {
    
    // Update display uniforms
    DisplayUniforms displayUniforms = UniformBuilder::buildDisplayUniforms(settings);
    if (m_displayUniformBuffer) {
        memcpy([m_displayUniformBuffer contents], &displayUniforms, sizeof(DisplayUniforms));
    }

    renderEncoder.label = @"Display Quad";
    [renderEncoder setRenderPipelineState:pipelines.display()];
    if (m_displayUniformBuffer) {
        [renderEncoder setFragmentBuffer:m_displayUniformBuffer offset:0 atIndex:0];
    }
    MTLTextureHandle displayTexture = accumulation.present();
    if ((settings.denoiseEnabled || settings.svgfDenoiseEnabled) && m_hasDenoisedOutput) {
        MTLTextureHandle denoised = accumulation.denoisedBuffer();
        if (denoised) {
            displayTexture = denoised;
        }
    }
    [renderEncoder setFragmentTexture:displayTexture atIndex:0];
    [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
}

FrameResult RenderLoop::encodeFrame(MTLCommandBufferHandle commandBuffer,
                                    MTLRenderPassDescriptorHandle renderPassDescriptor,
                                    MTLCAMetalDrawableHandle drawable,
                                    const MetalContext& context,
                                    Accumulation& accumulation,
                                    const Pipelines& pipelines,
                                    SceneResources& scene,
                                    UIOverlay& overlay,
                                    const RenderSettings& settings) {

    FrameResult result{};
    const bool willClearAccumulation = accumulation.needsClear() && pipelines.clear();
    const bool willRebuildScene = scene.isDirty();
    const bool hasDisplayPass = (renderPassDescriptor && drawable && pipelines.display());
    const bool restirDiRequested = ExplicitRestirDiEnabled(settings);
    const bool fusedRestirDiReference =
        context.headless() && restirDiRequested && HeadlessFusedRestirDiReferenceEnabled();
    const bool restirDiEnabled =
        restirDiRequested && !fusedRestirDiReference &&
        (context.headless() || InteractiveExplicitRestirDiEnabled());
    const bool restirDiWavefrontStageEnabled =
        restirDiEnabled && settings.executionMode == RenderSettings::ExecutionMode::Wavefront;
    const bool restirDiScreenPassesEnabled = restirDiEnabled && !restirDiWavefrontStageEnabled;
    m_explicitRestirDiActiveThisFrame = restirDiEnabled;
    m_restirDiHardwareTraceActiveThisFrame = false;
    static bool sLoggedInteractiveExplicitRestirFallback = false;
    if (restirDiRequested && !restirDiEnabled && !sLoggedInteractiveExplicitRestirFallback) {
        sLoggedInteractiveExplicitRestirFallback = true;
        if (fusedRestirDiReference) {
            NSLog(@"[ReSTIR DI] headless validation is rendering the fused/reference shader path "
                  "because PT_RESTIR_DI_VALIDATE_FUSED_REFERENCE is set.");
        } else {
            NSLog(@"[ReSTIR DI] interactive explicit D passes disabled for responsiveness; "
                  "set PT_ENABLE_INTERACTIVE_EXPLICIT_RESTIR_DI=1 to profile the explicit pass graph in the GUI.");
        }
    }
    m_lastPassGraph =
        BuildRenderPassGraph(settings,
                             willClearAccumulation,
                             willRebuildScene,
                             hasDisplayPass,
                             restirDiEnabled);
    if (commandBuffer) {
        commandBuffer.label = @"PathTracer FrameGraph";
        [commandBuffer pushDebugGroup:@"FrameGraph"];
    }
    HistoryResetReason frameHistoryResetReason = m_lastHistoryResetReason;
    m_lastHistoryResetReason = HistoryResetReason::None;

    m_gpuTiming.active = false;
    m_gpuTiming.nextSample = 0u;
    m_gpuTiming.ranges.clear();
    m_gpuTiming.lastIntersectMs = 0.0;
    m_gpuTiming.lastShadeMs = 0.0;
    m_gpuTiming.lastShadowQueueMs = 0.0;
    m_gpuTiming.lastCompactionMs = 0.0;
    m_gpuTiming.lastFusedIntegrateMs = 0.0;
    m_gpuTiming.lastPresentationMs = 0.0;
    if (m_gpuTiming.supported && m_gpuTiming.timestampCounterSet) {
        const uint32_t requestedSamples = std::max<uint32_t>(1u, settings.samplesPerFrame);
        uint32_t dispatchCountEstimate = 0u;
        if (settings.executionMode == RenderSettings::ExecutionMode::Wavefront) {
            const uint32_t depthCount = std::max<uint32_t>(1u, settings.maxDepth);
            const uint32_t compactionDispatches =
                WavefrontCompactionEnabled(settings) ? depthCount : 0u;
            dispatchCountEstimate =
                requestedSamples *
                    (3u + (restirDiWavefrontStageEnabled ? 7u : 6u) * depthCount +
                     compactionDispatches) +
                1u;
        } else {
            dispatchCountEstimate = requestedSamples + 1u;
        }
        dispatchCountEstimate += restirDiScreenPassesEnabled ? 6u : 0u;
        dispatchCountEstimate +=
            (restirDiScreenPassesEnabled && settings.restirDebugEnabled) ? 1u : 0u;
        const uint32_t sampleCountNeeded = dispatchCountEstimate * 2u + 8u;
        if (!m_gpuTiming.sampleBuffer || m_gpuTiming.sampleCapacity < sampleCountNeeded) {
            MTLCounterSampleBufferDescriptor* descriptor = [[MTLCounterSampleBufferDescriptor alloc] init];
            descriptor.counterSet = (id<MTLCounterSet>)m_gpuTiming.timestampCounterSet;
            descriptor.storageMode = MTLStorageModeShared;
            descriptor.sampleCount = sampleCountNeeded;
            descriptor.label = @"Headless GPU Timing";
            NSError* error = nil;
            m_gpuTiming.sampleBuffer = [m_device newCounterSampleBufferWithDescriptor:descriptor error:&error];
            if (!m_gpuTiming.sampleBuffer) {
                if (error) {
                    NSLog(@"[PerfTiming] Failed to create counter sample buffer: %@", error);
                }
                m_gpuTiming.sampleCapacity = 0u;
            } else {
                m_gpuTiming.sampleCapacity = sampleCountNeeded;
            }
        }
        m_gpuTiming.active = (m_gpuTiming.sampleBuffer != nil);
    }

    // Clear stats buffer
    if (m_statsBuffer) {
        memset([m_statsBuffer contents], 0, m_statsBuffer.length);
    }
#if PT_DEBUG_TOOLS
    if (m_debugBuffer) {
        auto* debugData =
            reinterpret_cast<PathtraceDebugBuffer*>([m_debugBuffer contents]);
        if (debugData) {
            debugData->writeIndex = 0;
            uint32_t allowed = 0;
            if (settings.enablePathDebug && settings.debugMaxEntries > 0) {
                allowed = std::min(settings.debugMaxEntries,
                                   PathTracerShaderTypes::kPathtraceDebugMaxEntries);
                allowed = std::max<uint32_t>(1u, allowed);
            }
            debugData->maxEntries = allowed;
            debugData->parityWriteIndex = 0;
            uint32_t parityAllowed = 0;
            if (settings.parityAssertEnabled &&
                settings.parityAssertMode != RenderSettings::ParityAssertMode::Off) {
                parityAllowed = settings.parityAssertOncePerFrame
                    ? 1u
                    : PathTracerShaderTypes::kPathtraceParityMaxEntries;
            }
            debugData->parityMaxEntries =
                std::min(parityAllowed, PathTracerShaderTypes::kPathtraceParityMaxEntries);
            debugData->parityChecksPerformed = 0;
            debugData->parityChecksInMedium = 0;
        }
    }
#endif

    uint32_t samplesDispatched = 0u;
    uint32_t restirDiInitializeDispatchCount = 0u;
    uint32_t restirDiInitialCandidateDispatchCount = 0u;
    uint32_t restirDiTemporalReuseDispatchCount = 0u;
    uint32_t restirDiSpatialReuseDispatchCount = 0u;
    uint32_t restirDiVisibilityDispatchCount = 0u;
    uint32_t restirDiApplyLightingDispatchCount = 0u;
    uint32_t restirDiDebugAovDispatchCount = 0u;
    m_lastPassGraphExecution = ExecuteRenderPassGraph(
        m_lastPassGraph,
        [&](const RenderPassDesc& pass) -> bool {
            const bool useDebugScope =
                commandBuffer &&
                pass.debugPolicy == DebugCapturePolicy::Scoped &&
                !pass.debugLabel.empty();
            if (useDebugScope) {
                [commandBuffer pushDebugGroup:[NSString stringWithUTF8String:pass.debugLabel.c_str()]];
            }
            bool passOk = true;
            switch (pass.executionId) {
                case RenderPassExecutionId::UpdateAccelerationStructures:
                    if (scene.isDirty()) {
                        scene.rebuildAccelerationStructures();
                        frameHistoryResetReason = HistoryResetReason::AccelerationStructureRebuilt;
                    }
                    break;
                case RenderPassExecutionId::ClearAccumulation:
                    if (accumulation.needsClear() && pipelines.clear()) {
                        accumulation.clear(commandBuffer, pipelines.clear());
                        const bool preserveSvgfHistory = m_preserveSvgfHistoryForNextClear;
                        resetHistory(HistoryResetReason::ResolutionChanged, preserveSvgfHistory);
                        m_preserveSvgfHistoryForNextClear = false;
                        frameHistoryResetReason = m_lastHistoryResetReason;
                    }
                    break;
                case RenderPassExecutionId::MegakernelIntegrate:
                    samplesDispatched =
                        encodeIntegration(commandBuffer, accumulation, pipelines, scene, settings);
                    break;
                case RenderPassExecutionId::WavefrontWorkGraph:
                    samplesDispatched =
                        encodeWavefront(commandBuffer, accumulation, pipelines, scene, settings);
                    break;
                case RenderPassExecutionId::Present:
                    encodePresentation(commandBuffer, accumulation, pipelines);
                    break;
                case RenderPassExecutionId::SvgfDenoise:
                    encodeSvgfDenoising(commandBuffer, accumulation, pipelines, settings);
                    break;
                case RenderPassExecutionId::RestirDiInitializeReservoirs:
                    passOk =
                        encodeRestirDiInitializeReservoirs(commandBuffer, accumulation, pipelines, settings);
                    if (passOk) {
                        restirDiInitializeDispatchCount += 1u;
                    }
                    break;
                case RenderPassExecutionId::RestirDiInitialCandidates:
                    passOk =
                        encodeRestirDiInitialCandidates(commandBuffer, accumulation, pipelines, scene, settings);
                    if (passOk) {
                        restirDiInitialCandidateDispatchCount += 1u;
                    }
                    break;
                case RenderPassExecutionId::RestirDiTemporalReuse:
                    passOk =
                        encodeRestirDiTemporalReuse(commandBuffer, accumulation, pipelines, settings);
                    if (passOk) {
                        restirDiTemporalReuseDispatchCount += 1u;
                    }
                    break;
                case RenderPassExecutionId::RestirDiSpatialReuse:
                    passOk =
                        encodeRestirDiSpatialReuse(commandBuffer, accumulation, pipelines, settings);
                    if (passOk) {
                        restirDiSpatialReuseDispatchCount += 1u;
                    }
                    break;
                case RenderPassExecutionId::RestirDiVisibility:
                    passOk =
                        encodeRestirDiVisibility(commandBuffer, accumulation, pipelines, scene, settings);
                    if (passOk) {
                        restirDiVisibilityDispatchCount += 1u;
                    }
                    break;
                case RenderPassExecutionId::RestirDiDebugAov:
                    passOk =
                        encodeRestirDiDebugAov(commandBuffer, accumulation, pipelines, settings);
                    if (passOk) {
                        restirDiDebugAovDispatchCount += 1u;
                    }
                    break;
                case RenderPassExecutionId::RestirDiApplyLighting:
                    passOk =
                        encodeRestirDiApplyLighting(commandBuffer, accumulation, pipelines, settings);
                    if (passOk) {
                        restirDiApplyLightingDispatchCount += 1u;
                    }
                    break;
                case RenderPassExecutionId::OidnDenoise:
                    if (!settings.svgfDenoiseEnabled) {
                        encodeDenoising(commandBuffer, accumulation, settings, m_frameCounter);
                    }
                    break;
                case RenderPassExecutionId::Display:
                    if (renderPassDescriptor && drawable && pipelines.display()) {
                        id<MTLRenderCommandEncoder> renderEncoder =
                            [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
                        encodeDisplay(renderEncoder, accumulation, pipelines, settings);
                        overlay.render(commandBuffer, renderEncoder);
                        [renderEncoder endEncoding];
                    }
                    break;
                case RenderPassExecutionId::NoOp:
                case RenderPassExecutionId::WavefrontStageMarker:
                    break;
                default:
                    passOk = false;
                    break;
            }
            if (useDebugScope) {
                [commandBuffer popDebugGroup];
            }
            return passOk;
        });
    if (commandBuffer) {
        [commandBuffer popDebugGroup];
    }

    // Increment frame counter for next frame
    m_frameCounter++;

    result.samplesDispatched = samplesDispatched;
    result.restirDiInitializeDispatchCount = restirDiInitializeDispatchCount;
    result.restirDiInitialCandidateDispatchCount = restirDiInitialCandidateDispatchCount;
    result.restirDiTemporalReuseDispatchCount = restirDiTemporalReuseDispatchCount;
    result.restirDiSpatialReuseDispatchCount = restirDiSpatialReuseDispatchCount;
    result.restirDiVisibilityDispatchCount = restirDiVisibilityDispatchCount;
    result.restirDiApplyLightingDispatchCount = restirDiApplyLightingDispatchCount;
    result.restirDiDebugAovDispatchCount = restirDiDebugAovDispatchCount;
    result.presentationDispatchCount = 1u;
    if (settings.executionMode == RenderSettings::ExecutionMode::Wavefront) {
        const uint32_t depthCount = std::max<uint32_t>(1u, settings.maxDepth);
        result.wavefrontResetDispatchCount = samplesDispatched;
        result.wavefrontGenerateDispatchCount = samplesDispatched;
        result.wavefrontBuildDispatchArgsDispatchCount = samplesDispatched * depthCount;
        result.wavefrontCompactQueueDispatchCount =
            WavefrontCompactionEnabled(settings) ? (samplesDispatched * depthCount) : 0u;
        result.wavefrontPrepareDispatchCount = samplesDispatched * depthCount;
        result.wavefrontIntersectDispatchCount = samplesDispatched * depthCount;
        result.wavefrontShadeDispatchCount =
            samplesDispatched * depthCount * (restirDiWavefrontStageEnabled ? 2u : 1u);
        result.wavefrontShadowDispatchCount = samplesDispatched * depthCount;
        result.wavefrontSwapDispatchCount = samplesDispatched * depthCount;
        result.wavefrontAccumulateDispatchCount = samplesDispatched;
    } else {
        result.integrationDispatchCount = samplesDispatched;
    }
    result.totalComputeDispatchCount =
        result.restirDiInitializeDispatchCount +
        result.restirDiInitialCandidateDispatchCount +
        result.restirDiTemporalReuseDispatchCount +
        result.restirDiSpatialReuseDispatchCount +
        result.restirDiVisibilityDispatchCount +
        result.restirDiApplyLightingDispatchCount +
        result.restirDiDebugAovDispatchCount +
        result.integrationDispatchCount +
        result.presentationDispatchCount +
        result.wavefrontResetDispatchCount +
        result.wavefrontGenerateDispatchCount +
        result.wavefrontBuildDispatchArgsDispatchCount +
        result.wavefrontCompactQueueDispatchCount +
        result.wavefrontPrepareDispatchCount +
        result.wavefrontIntersectDispatchCount +
        result.wavefrontShadeDispatchCount +
        result.wavefrontShadowDispatchCount +
        result.wavefrontSwapDispatchCount +
        result.wavefrontAccumulateDispatchCount +
        (settings.svgfDenoiseEnabled ? (std::clamp(settings.svgfAtrousIterations, 1u, 5u) + 1u) : 0u);
    result.passGraph = m_lastPassGraph;
    result.passGraphExecution = m_lastPassGraphExecution;
    result.historyResetReason = frameHistoryResetReason;
    m_lastFrameHistoryResetReason = frameHistoryResetReason;
    m_lastHistoryResetReason = HistoryResetReason::None;
    return result;
}

void RenderLoop::encodeDisplayOnlyFrame(MTLCommandBufferHandle commandBuffer,
                                        MTLRenderPassDescriptorHandle renderPassDescriptor,
                                        MTLCAMetalDrawableHandle drawable,
                                        const Accumulation& accumulation,
                                        const Pipelines& pipelines,
                                        UIOverlay& overlay,
                                        const RenderSettings& settings) {
    if (!commandBuffer || !renderPassDescriptor || !drawable || !pipelines.display()) {
        return;
    }

    id<MTLRenderCommandEncoder> renderEncoder =
        [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
    if (!renderEncoder) {
        return;
    }
    encodeDisplay(renderEncoder, accumulation, pipelines, settings);
    overlay.render(commandBuffer, renderEncoder);
    [renderEncoder endEncoding];
}

void RenderLoop::resolveGpuTiming(double totalGpuTimeMs) {
    m_gpuTiming.lastIntersectMs = 0.0;
    m_gpuTiming.lastShadeMs = 0.0;
    m_gpuTiming.lastShadowQueueMs = 0.0;
    m_gpuTiming.lastCompactionMs = 0.0;
    m_gpuTiming.lastFusedIntegrateMs = 0.0;
    m_gpuTiming.lastPresentationMs = 0.0;
    if (!m_gpuTiming.active ||
        !m_gpuTiming.sampleBuffer ||
        m_gpuTiming.nextSample == 0u ||
        totalGpuTimeMs <= 0.0) {
        return;
    }
    NSData* resolved =
        [(id<MTLCounterSampleBuffer>)m_gpuTiming.sampleBuffer resolveCounterRange:NSMakeRange(0u, m_gpuTiming.nextSample)];
    if (!resolved || resolved.length < (sizeof(MTLCounterResultTimestamp) * m_gpuTiming.nextSample)) {
        return;
    }
    const auto* timestamps =
        reinterpret_cast<const MTLCounterResultTimestamp*>(resolved.bytes);
    uint64_t intersectTicks = 0u;
    uint64_t shadeTicks = 0u;
    uint64_t shadowQueueTicks = 0u;
    uint64_t compactionTicks = 0u;
    uint64_t fusedIntegrateTicks = 0u;
    uint64_t presentationTicks = 0u;
    uint64_t totalTicks = 0u;
    for (const GpuTimingSampleRange& range : m_gpuTiming.ranges) {
        if (range.endSample >= m_gpuTiming.nextSample) {
            continue;
        }
        const uint64_t start = timestamps[range.startSample].timestamp;
        const uint64_t end = timestamps[range.endSample].timestamp;
        if (start == MTLCounterErrorValue || end == MTLCounterErrorValue || end <= start) {
            continue;
        }
        const uint64_t deltaTicks = end - start;
        totalTicks += deltaTicks;
        switch (range.bucket) {
            case GpuTimingBucket::Intersect:
                intersectTicks += deltaTicks;
                break;
            case GpuTimingBucket::Shade:
                shadeTicks += deltaTicks;
                break;
            case GpuTimingBucket::ShadowQueue:
                shadowQueueTicks += deltaTicks;
                break;
            case GpuTimingBucket::Compaction:
                compactionTicks += deltaTicks;
                break;
            case GpuTimingBucket::FusedIntegrate:
                fusedIntegrateTicks += deltaTicks;
                break;
            case GpuTimingBucket::Presentation:
                presentationTicks += deltaTicks;
                break;
        }
    }
    if (totalTicks == 0u) {
        return;
    }
    const double scale = totalGpuTimeMs / static_cast<double>(totalTicks);
    m_gpuTiming.lastIntersectMs = static_cast<double>(intersectTicks) * scale;
    m_gpuTiming.lastShadeMs = static_cast<double>(shadeTicks) * scale;
    m_gpuTiming.lastShadowQueueMs = static_cast<double>(shadowQueueTicks) * scale;
    m_gpuTiming.lastCompactionMs = static_cast<double>(compactionTicks) * scale;
    m_gpuTiming.lastFusedIntegrateMs = static_cast<double>(fusedIntegrateTicks) * scale;
    m_gpuTiming.lastPresentationMs = static_cast<double>(presentationTicks) * scale;
}

void RenderLoop::shutdown() {
    resetHistory(HistoryResetReason::None);
    m_lastPassGraph = RenderPassGraphDesc{};
    m_lastPassGraphExecution = RenderPassExecutionResult{};
    m_lastFrameHistoryResetReason = HistoryResetReason::None;
    m_gpuTiming.sampleBuffer = nullptr;
    m_gpuTiming.timestampCounterSet = nullptr;
    m_gpuTiming.ranges.clear();
    m_wavefront.queueCounters = nullptr;
    m_wavefront.pathStates = nullptr;
    m_wavefront.shadowRays = nullptr;
    m_wavefront.hitRecords = nullptr;
    m_wavefront.activeRays[1] = nullptr;
    m_wavefront.activeRays[0] = nullptr;
    m_frameGraphOwnedBuffers.clear();
    m_restirState.restirPt = nullptr;
    m_restirState.pathGuiding = nullptr;
    m_restirState = RestirStateResources{};
    m_transportMemory.clear();
    m_statsBuffer = nullptr;
    m_debugBuffer = nullptr;
    m_displayUniformBuffer = nullptr;
    m_uniformBuffer = nullptr;
    m_device = nullptr;
}

}  // namespace PathTracer
