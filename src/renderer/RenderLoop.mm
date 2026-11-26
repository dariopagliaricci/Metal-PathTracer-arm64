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
#include <array>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

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
using PathTracerShaderTypes::WavefrontResolvedSurface;
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

bool WavefrontShadeIndirectEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_WAVEFRONT_SHADE_INDIRECT");
    return enabled;
}

bool WavefrontShadeIndirectProbeEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_WAVEFRONT_SHADE_INDIRECT_PROBE");
    return enabled;
}

bool WavefrontShadeSurfaceProbeEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_WAVEFRONT_SHADE_SURFACE_PROBE");
    return enabled;
}

bool WavefrontShadeMetadataProbeEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_WAVEFRONT_SHADE_METADATA_PROBE");
    return enabled;
}

bool WavefrontShadeScalarProbeEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_WAVEFRONT_SHADE_SCALAR_PROBE");
    return enabled;
}

bool WavefrontShadeBsdfProbeEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_WAVEFRONT_SHADE_BSDF_PROBE");
    return enabled;
}

bool WavefrontShadeClosureProbeEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_WAVEFRONT_SHADE_CLOSURE_PROBE");
    return enabled;
}

constexpr uint32_t kMlTinyModelMagic = 0x4d4c5231u;
constexpr uint32_t kMlTinyModelVersion = 1u;
constexpr uint32_t kMlTinyFeatureCount = 8u;
constexpr uint32_t kMlTinyOutputCount = 4u;
constexpr uint32_t kMlTinyWeightCount =
    (kMlTinyFeatureCount + 1u) * kMlTinyOutputCount;

struct MlModelLoadResult {
    bool valid = false;
    MlReconstructionTinyModel model{};
    std::string providerVersion = "invalid";
    std::string fallbackReason;
};

MlModelLoadResult LoadMlReconstructionTinyModel(const std::string& path) {
    MlModelLoadResult result{};
    if (path.empty()) {
        result.fallbackReason =
            "custom Metal ML backend requires --mlModelPackage PTMLRECONv1 package";
        return result;
    }

    std::ifstream input(path);
    if (!input) {
        result.fallbackReason = "model package unavailable: " + path;
        return result;
    }

    std::string token;
    input >> token;
    if (token != "PTMLRECONv1") {
        result.fallbackReason =
            "invalid model package magic/version; expected PTMLRECONv1";
        return result;
    }

    uint32_t version = 0u;
    uint32_t featureCount = 0u;
    uint32_t outputCount = 0u;
    float confidenceBias = 0.0f;
    float varianceScale = 0.05f;
    std::array<float, kMlTinyWeightCount> weights{};
    bool sawVersion = false;
    bool sawFeatures = false;
    bool sawOutputs = false;
    bool sawWeights = false;

    while (input >> token) {
        if (token == "version") {
            input >> version;
            sawVersion = true;
        } else if (token == "feature_count") {
            input >> featureCount;
            sawFeatures = true;
        } else if (token == "output_count") {
            input >> outputCount;
            sawOutputs = true;
        } else if (token == "confidence_bias") {
            input >> confidenceBias;
        } else if (token == "variance_scale") {
            input >> varianceScale;
        } else if (token == "weights") {
            for (float& value : weights) {
                if (!(input >> value)) {
                    result.fallbackReason =
                        "invalid model package: expected 36 float weights";
                    return result;
                }
            }
            sawWeights = true;
        } else {
            std::string discard;
            std::getline(input, discard);
        }
    }

    if (!sawVersion || version != kMlTinyModelVersion) {
        result.fallbackReason =
            "unsupported model version; expected PTMLRECONv1 version 1";
        return result;
    }
    if (!sawFeatures || featureCount != kMlTinyFeatureCount ||
        !sawOutputs || outputCount != kMlTinyOutputCount) {
        result.fallbackReason =
            "invalid model tensor shape; expected 8 features and 4 outputs";
        return result;
    }
    if (!sawWeights) {
        result.fallbackReason =
            "invalid model package: missing weights block";
        return result;
    }

    result.model.magic = kMlTinyModelMagic;
    result.model.version = version;
    result.model.featureCount = featureCount;
    result.model.outputCount = outputCount;
    result.model.confidenceBias = confidenceBias;
    result.model.varianceScale = std::max(varianceScale, 0.0f);
    result.model.weights = weights;
    result.providerVersion = "ptmlrecon-v1";
    result.valid = true;
    return result;
}

bool WavefrontShadowIndirectEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_WAVEFRONT_SHADOW_INDIRECT");
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
        const bool shadeIndirect = WavefrontShadeIndirectEnabled();
        const bool shadeIndirectProbe = WavefrontShadeIndirectProbeEnabled();
        const bool shadeSurfaceProbe = WavefrontShadeSurfaceProbeEnabled();
        const bool shadeMetadataProbe = WavefrontShadeMetadataProbeEnabled();
        const bool shadeScalarProbe = WavefrontShadeScalarProbeEnabled();
        const bool shadeBsdfProbe = WavefrontShadeBsdfProbeEnabled();
        const bool shadeClosureProbe = WavefrontShadeClosureProbeEnabled();
        const bool shadowIndirect = WavefrontShadowIndirectEnabled();
        if (skipBlas || skipTextures || shadeIndirect || shadeIndirectProbe || shadeSurfaceProbe || shadeMetadataProbe || shadeScalarProbe || shadeBsdfProbe || shadeClosureProbe || shadowIndirect) {
            NSLog(@"[Isolation] PT_SKIP_BLAS_USE_RESOURCES=%@ PT_SKIP_TEXTURE_USE_RESOURCES=%@ "
                  "PT_ENABLE_WAVEFRONT_SHADE_INDIRECT=%@ "
                  "PT_ENABLE_WAVEFRONT_SHADE_INDIRECT_PROBE=%@ "
                  "PT_ENABLE_WAVEFRONT_SHADE_SURFACE_PROBE=%@ "
                  "PT_ENABLE_WAVEFRONT_SHADE_METADATA_PROBE=%@ "
                  "PT_ENABLE_WAVEFRONT_SHADE_SCALAR_PROBE=%@ "
                  "PT_ENABLE_WAVEFRONT_SHADE_CLOSURE_PROBE=%@ "
                  "PT_ENABLE_WAVEFRONT_SHADE_BSDF_PROBE=%@ "
                  "PT_ENABLE_WAVEFRONT_SHADOW_INDIRECT=%@",
                  skipBlas ? @"1" : @"0",
                  skipTextures ? @"1" : @"0",
                  shadeIndirect ? @"1" : @"0",
                  shadeIndirectProbe ? @"1" : @"0",
                  shadeSurfaceProbe ? @"1" : @"0",
                  shadeMetadataProbe ? @"1" : @"0",
                  shadeScalarProbe ? @"1" : @"0",
                  shadeClosureProbe ? @"1" : @"0",
                  shadeBsdfProbe ? @"1" : @"0",
                  shadowIndirect ? @"1" : @"0");
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
    if (RestirDiDemotedByVolume(settings)) {
        return false;
    }
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

inline bool ForceSoftwareRestirDiInitialCandidates() {
    static const bool enabled = EnvFlagEnabled("PT_RESTIR_DI_FORCE_SW_INITIAL");
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
    std::memset([m_statsBuffer contents], 0, m_statsBuffer.length);

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

bool RenderLoop::ensureMlBackendResources(uint32_t width,
                                          uint32_t height,
                                          const RenderSettings& settings,
                                          MTLCommandBufferHandle commandBuffer) {
    m_mlState.activeThisFrame = false;
    m_mlState.dispatchCount = 0u;
    m_mlState.lastConfidence = 0.0f;
    m_mlState.biasRisk = 0.0f;
    m_mlState.tensorContract =
        "radiance,albedo,normal,depth,motion,variance,confidence -> reconstructed_radiance,confidence,validity";

    if (settings.mlBackendMode == RenderSettings::MlBackendMode::Off) {
        m_mlState = {};
        m_mlState.provider = "disabled";
        m_mlState.providerVersion = "disabled";
        m_mlState.tensorContract =
            "radiance,albedo,normal,depth,motion,variance,confidence -> reconstructed_radiance,confidence,validity";
        m_mlState.fallbackReason = "ml backend disabled";
        m_transportAllocator.releaseOwner(TransportMemoryOwner::MLBackend);
        return true;
    }

    if (settings.mlBackendMode == RenderSettings::MlBackendMode::ReconstructionNoOp) {
        m_mlState.modelBuffer = nullptr;
        m_mlState.modelBytes = 0u;
        m_mlState.width = width;
        m_mlState.height = height;
        m_mlState.modelValid = false;
        m_mlState.modelPath = settings.mlModelPackagePath;
        m_mlState.provider = "MLBackend.NoOp";
        m_mlState.providerVersion = "noop-contract-v1";
        m_mlState.fallbackReason =
            "no-op ML backend selected; renderer preserves non-neural reconstruction output";
        m_transportAllocator.releaseOwner(TransportMemoryOwner::MLBackend);
        return true;
    }

    m_mlState.provider = "MLBackend.CustomMetal.TinyReconstruction";
    if (settings.mlBackendMode != RenderSettings::MlBackendMode::ReconstructionTiny) {
        m_mlState.modelValid = false;
        m_mlState.fallbackReason = "unsupported ML backend mode";
        m_transportAllocator.releaseOwner(TransportMemoryOwner::MLBackend);
        return true;
    }

    MlModelLoadResult loaded = LoadMlReconstructionTinyModel(settings.mlModelPackagePath);
    if (!loaded.valid) {
        m_mlState.modelBuffer = nullptr;
        m_mlState.modelBytes = 0u;
        m_mlState.width = width;
        m_mlState.height = height;
        m_mlState.modelValid = false;
        m_mlState.modelPath = settings.mlModelPackagePath;
        m_mlState.providerVersion = "invalid";
        m_mlState.fallbackReason = loaded.fallbackReason;
        m_transportAllocator.releaseOwner(TransportMemoryOwner::MLBackend);
        return true;
    }

    const uint64_t modelBytes = sizeof(MlReconstructionTinyModel);
    const bool reuse =
        m_mlState.modelBuffer &&
        m_mlState.modelValid &&
        m_mlState.modelPath == settings.mlModelPackagePath &&
        m_mlState.modelBytes == modelBytes &&
        m_mlState.width == width &&
        m_mlState.height == height;
    if (!reuse) {
        TransportMemoryRecord record{};
        record.name = "MLBackend.TensorMetadata";
        record.kind = TransportMemoryKind::MLBackendTensor;
        record.owner = TransportMemoryOwner::MLBackend;
        record.targetOwner = TransportMemoryOwner::MLBackend;
        record.frameGraphGroup = RenderResourceGroup::MLBackendResources;
        record.lifetime = ResourceLifetime::Persistent;
        record.width = width;
        record.height = height;
        record.depthOrCellCount = 1u;
        record.bytes = modelBytes;
        record.history = TransportMemoryHistoryPolicy::PersistentHistory;
        record.debug = TransportMemoryDebugReadbackPolicy::MetadataOnly;
        record.invalidatesOn = FrameHistoryInvalidationMask();
        record.frameGraphOwned = true;
        record.transportMemoryOwned = true;
        record.hotPath = true;
        record.byteSizeFormula =
            "sizeof(PTMLRECONv1 header + 36 affine reconstruction weights)";
        record.featureGate = "mlBackendMode=reconstruction_tiny";
        record.migrationRisk = "low";
        MTLBufferHandle modelBuffer =
            m_transportAllocator.allocateBuffer(m_device,
                                                record,
                                                StorageClass::Private,
                                                "TransportMemory.MLBackend.TinyReconstructionV1");
        if (!modelBuffer) {
            m_mlState.modelBuffer = nullptr;
            m_mlState.modelBytes = 0u;
            m_mlState.modelValid = false;
            m_mlState.fallbackReason = "failed to allocate private ML model buffer";
            return true;
        }

        id<MTLBuffer> upload =
            [m_device newBufferWithBytes:&loaded.model
                                  length:sizeof(loaded.model)
                                 options:MTLResourceStorageModeShared];
        if (!upload || !commandBuffer) {
            m_transportAllocator.releaseOwner(TransportMemoryOwner::MLBackend);
            m_mlState.modelBuffer = nullptr;
            m_mlState.modelBytes = 0u;
            m_mlState.modelValid = false;
            m_mlState.fallbackReason =
                "failed to stage ML model upload without CPU-hot synchronization";
            return true;
        }
        upload.label = @"Upload.MLBackend.TinyReconstructionV1";
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        blit.label = @"Upload ML Reconstruction Model";
        [blit copyFromBuffer:upload
                sourceOffset:0
                    toBuffer:modelBuffer
           destinationOffset:0
                        size:sizeof(loaded.model)];
        [blit endEncoding];
        m_mlState.modelBuffer = modelBuffer;
    }

    m_mlState.modelBytes = modelBytes;
    m_mlState.width = width;
    m_mlState.height = height;
    m_mlState.modelValid = true;
    m_mlState.modelPath = settings.mlModelPackagePath;
    m_mlState.providerVersion = loaded.providerVersion;
    m_mlState.fallbackReason.clear();
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
    if (m_statsBuffer && [m_statsBuffer contents]) {
        auto* stats = reinterpret_cast<PathtraceStats*>([m_statsBuffer contents]);
        std::memset(stats->radianceCacheStats, 0, sizeof(stats->radianceCacheStats));
        std::memset(stats->sssWalkStats, 0, sizeof(stats->sssWalkStats));
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
    if (m_restirState.pathGuidingPrivateStorage ||
        m_restirState.restirPtPrivateStorage ||
        m_restirState.restirDiPrivateStorage ||
        m_restirState.radianceCachePrivateStorage ||
        m_restirState.causticTransportActive) {
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
        settings.restirGiMode == RenderSettings::RestirGiMode::ProductionDiffuse;
    const bool radianceCacheEnabled = RadianceCacheActive(settings);
    const bool causticTransportEnabled =
        settings.causticTransportMode == RenderSettings::CausticTransportMode::PathSpacePrototype;
    const uint32_t causticMaxVertices = std::max<uint32_t>(settings.causticMaxVertices, 1u);
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
            (!m_restirState.pathGuidingPrivateStorage &&
             !m_restirState.restirPtPrivateStorage &&
             !m_restirState.restirDiPrivateStorage &&
             !m_restirState.radianceCachePrivateStorage &&
             !m_restirState.causticTransportActive)) {
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
        fillBuffer(m_restirState.pathGuiding, m_restirState.pathGuidingBytes);
        fillBuffer(m_restirState.restirPt, m_restirState.restirPtBytes);
        fillBuffer(m_restirState.radianceCache, m_restirState.radianceCacheBytes);
        fillBuffer(m_transportAllocator.bufferForName("CausticTransport.PathSpacePrototype"),
                   m_restirState.causticTransportBytes);
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
        (causticTransportEnabled == m_restirState.causticTransportActive) &&
        (!causticTransportEnabled || m_restirState.causticMaxVertices == causticMaxVertices) &&
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
    const NSUInteger causticTransportBytes =
        causticTransportEnabled
            ? static_cast<NSUInteger>(
                  static_cast<uint64_t>(causticMaxVertices) *
                  (sizeof(PathTracerShaderTypes::EyeSubpathVertex) +
                   sizeof(PathTracerShaderTypes::LightSubpathVertex) +
                   sizeof(PathTracerShaderTypes::PathConnection) +
                   sizeof(PathTracerShaderTypes::ConnectionResult)))
            : 0u;

    const TransportMemoryInvalidationMask allTransport = AllTransportInvalidations();
    const TransportMemoryInvalidationMask lightTransport = LightTransportInvalidationMask();
    m_transportAllocator.releaseOwners({
        TransportMemoryOwner::RestirDI,
        TransportMemoryOwner::RestirGI,
        TransportMemoryOwner::RestirPT,
        TransportMemoryOwner::PathGuiding,
        TransportMemoryOwner::RadianceCache,
        TransportMemoryOwner::CausticTransport,
        TransportMemoryOwner::MLBackend,
        TransportMemoryOwner::Debug,
    });
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
                                  StorageClass::Private,
                                  ResourceLifetime::History,
                                  TransportMemoryHistoryPolicy::PersistentHistory,
                                  TransportMemoryDebugReadbackPolicy::StagingCopy,
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
                                  StorageClass::Private,
                                  ResourceLifetime::History,
                                  TransportMemoryHistoryPolicy::PersistentHistory,
                                  TransportMemoryDebugReadbackPolicy::StagingCopy,
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
                                  StorageClass::Private,
                                  ResourceLifetime::History,
                                  TransportMemoryHistoryPolicy::PersistentHistory,
                                  TransportMemoryDebugReadbackPolicy::StagingCopy,
                                  allTransport,
                                  "TransportMemory.RadianceCache.WorldGrid")
        : nil;
    if ((pathGuidingEnabled && !pathGuiding) ||
        (restirPtEnabled && !restirPt) ||
        (radianceCacheEnabled && !radianceCache)) {
        NSLog(@"[ReSTIR State] failed to allocate production reservoir buffers");
        return false;
    }
    MTLBufferHandle causticTransport = causticTransportEnabled
        ? allocateTransportBuffer("CausticTransport.PathSpacePrototype",
                                  TransportMemoryKind::CausticPathSpace,
                                  TransportMemoryOwner::CausticTransport,
                                  RenderResourceGroup::CausticTransportResources,
                                  width,
                                  height,
                                  causticMaxVertices,
                                  causticTransportBytes,
                                  StorageClass::Private,
                                  ResourceLifetime::History,
                                  TransportMemoryHistoryPolicy::PersistentHistory,
                                  TransportMemoryDebugReadbackPolicy::MetadataOnly,
                                  allTransport,
                                  "TransportMemory.CausticTransport.PathSpacePrototype")
        : nil;
    if (causticTransportEnabled && !causticTransport) {
        NSLog(@"[CausticTransport] failed to allocate path-space transport buffer");
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
    m_restirState.causticTransportBytes = causticTransportEnabled ? causticTransportBytes : 0u;
    m_restirState.causticMaxVertices = causticTransportEnabled ? causticMaxVertices : 0u;
    m_restirState.pathGuidingPrivateStorage = pathGuidingEnabled;
    m_restirState.restirPtPrivateStorage = restirPtEnabled;
    m_restirState.restirDiPrivateStorage = restirDiEnabled;
    m_restirState.radianceCachePrivateStorage = radianceCacheEnabled;
    m_restirState.causticTransportActive = causticTransportEnabled;
    m_restirState.pendingGpuClear =
        pathGuidingEnabled || restirPtEnabled || restirDiEnabled || radianceCacheEnabled;
    m_restirState.pendingGpuClear = m_restirState.pendingGpuClear || causticTransportEnabled;
    if (!clearGpuPrivateTransportBuffers(commandBuffer) &&
        (pathGuidingEnabled || restirPtEnabled || restirDiEnabled || radianceCacheEnabled ||
         causticTransportEnabled)) {
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
    const bool restirGiActive =
        settings.restirGiMode == RenderSettings::RestirGiMode::ProductionDiffuse;
    const bool restirPtActive =
        settings.restirPtMode != RenderSettings::RestirPtMode::Off;

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
            record.featureGate = "pathGuidingMode=path_guiding";
            record.migrationRisk = "medium";
            record.hotPath = true;
            record.temporaryShared = record.storage == TransportMemoryStorageClass::Shared;
        } else if (record.name == "ReSTIRGIProductionReservoirs") {
            record.targetStorage = TransportMemoryStorageClass::Private;
            record.byteSizeFormula =
                "width * height * kRestirPathReservoirDepthCount * RestirPtReservoirState";
            record.featureGate = "restirGiMode=production_diffuse";
            record.migrationRisk = "low";
            record.hotPath = true;
            record.temporaryShared = false;
        } else if (record.name == "RestirPTPathReservoirs") {
            record.targetStorage = TransportMemoryStorageClass::Private;
            record.byteSizeFormula =
                "width * height * kRestirPathReservoirDepthCount * RestirPtReservoirState";
            record.featureGate = "restirPtMode!=off";
            record.migrationRisk = "low";
            record.hotPath = true;
            record.temporaryShared = false;
        } else if (record.name == "RadianceCache.WorldGrid") {
            record.targetOwner = TransportMemoryOwner::RadianceCache;
            record.targetStorage = TransportMemoryStorageClass::Private;
            record.byteSizeFormula =
                "hashed world-grid cells * kRestirPathReservoirDepthCount * RadianceCacheState with query metadata";
            record.featureGate = "radianceCacheMode=radiance_cache";
            record.migrationRisk = "medium";
            record.hotPath = true;
            record.temporaryShared = record.storage == TransportMemoryStorageClass::Shared;
        } else if (record.name == "CausticTransport.PathSpacePrototype") {
            record.targetOwner = TransportMemoryOwner::CausticTransport;
            record.targetStorage = TransportMemoryStorageClass::Private;
            record.byteSizeFormula =
                "causticMaxVertices * (EyeSubpathVertex + LightSubpathVertex + PathConnection + ConnectionResult)";
            record.featureGate = "causticTransportMode=path_space";
            record.migrationRisk = "medium";
            record.hotPath = true;
            record.temporaryShared = false;
        } else if (record.name == "MLBackend.TensorMetadata") {
            record.targetOwner = TransportMemoryOwner::MLBackend;
            record.targetStorage = TransportMemoryStorageClass::Private;
            record.byteSizeFormula =
                settings.mlBackendMode == RenderSettings::MlBackendMode::ReconstructionTiny
                    ? "sizeof(PTMLRECONv1 header + 36 affine reconstruction weights); graph textures own tensor IO"
                    : "0 bytes for no-op backend contract";
            record.featureGate = "mlBackendMode!=off";
            record.migrationRisk =
                settings.mlBackendMode == RenderSettings::MlBackendMode::ReconstructionTiny
                    ? "low"
                    : "medium";
            record.hotPath =
                settings.mlBackendMode == RenderSettings::MlBackendMode::ReconstructionTiny;
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
        } else if (record.owner == TransportMemoryOwner::Wavefront) {
            record.targetOwner = TransportMemoryOwner::Wavefront;
            record.targetStorage = record.name == "Wavefront.QueueCounters"
                                       ? TransportMemoryStorageClass::Shared
                                       : TransportMemoryStorageClass::Private;
            record.byteSizeFormula = "queue capacity * queue element stride";
            if (record.name == "Wavefront.QueueCounters") {
                record.byteSizeFormula = "sizeof(QueueCounters)";
            } else if (record.name == "Wavefront.IndirectDispatchArgs") {
                record.byteSizeFormula = "3 * sizeof(uint32_t)";
            } else if (record.name == "Wavefront.RadianceCacheQueryItems") {
                record.byteSizeFormula = "width * height * RadianceCacheWorkItem";
            } else if (record.name == "Wavefront.RadianceCacheTrainRecords") {
                record.byteSizeFormula = "ceil(width * height / 8) * 6 * RadianceCacheTrainRecord";
            }
            record.featureGate = "executionMode=wavefront";
            record.migrationRisk =
                record.storage == record.targetStorage ? "medium" : "high";
            record.hotPath = record.name != "Wavefront.QueueCounters";
            record.temporaryShared =
                record.hotPath && record.storage == TransportMemoryStorageClass::Shared;
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
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "ReSTIRGIProductionReservoirs" ||
                   record.name == "RestirPTPathReservoirs") {
            record.ownedBuffer = m_restirState.restirPt;
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "RadianceCache.WorldGrid") {
            record.ownedBuffer = m_restirState.radianceCache;
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "ReSTIRDIReservoirCurrent") {
            record.ownedBuffer = m_restirState.restirDiCurrent;
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "ReSTIRDIReservoirPrevious") {
            record.ownedBuffer = m_restirState.restirDiPrevious;
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "ReSTIRDIReservoirTemporal") {
            record.ownedBuffer = m_restirState.restirDiTemporal;
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "ReSTIRDIReservoirSpatial") {
            record.ownedBuffer = m_restirState.restirDiSpatial;
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "ReSTIRDIVisibility") {
            record.ownedBuffer = m_restirState.restirDiVisibility;
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "ReSTIRDINeighborPattern") {
            record.ownedBuffer = m_restirState.restirDiNeighborPattern;
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "ReSTIRDIDebugAOVReadback") {
            record.ownedBuffer = m_restirState.restirDiDebug;
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "Wavefront.ActiveRays.Current") {
            record.ownedBuffer = m_wavefront.activeRays[0];
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "Wavefront.ActiveRays.Next") {
            record.ownedBuffer = m_wavefront.activeRays[1];
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "Wavefront.HitRecords") {
            record.ownedBuffer = m_wavefront.hitRecords;
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "Wavefront.ShadowRays") {
            record.ownedBuffer = m_wavefront.shadowRays;
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "Wavefront.ShadowIndices") {
            record.ownedBuffer = m_wavefront.shadowIndices;
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "Wavefront.PathStates") {
            record.ownedBuffer = m_wavefront.pathStates;
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "Wavefront.RadianceCacheQueryItems") {
            record.ownedBuffer = m_wavefront.cacheQueryItems;
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "Wavefront.RadianceCacheTrainRecords") {
            record.ownedBuffer = m_wavefront.cacheTrainRecords;
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "Wavefront.QueueCounters") {
            record.ownedBuffer = m_wavefront.queueCounters;
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        } else if (record.name == "Wavefront.IndirectDispatchArgs") {
            record.ownedBuffer = m_wavefront.indirectDispatchArgs;
            record.frameGraphOwned = record.ownedBuffer != nullptr;
            record.transportMemoryOwned = record.ownedBuffer != nullptr;
        }
        if (record.owner == TransportMemoryOwner::Denoiser ||
            record.owner == TransportMemoryOwner::FrameHistory) {
            record.frameGraphOwned = record.bytes > 0u;
            record.transportMemoryOwned = false;
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
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::PersistentHistory,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       allTransport);
    }

    if (restirGiActive && m_restirState.restirPtBytes > 0u) {
        registerRecord("ReSTIRGIProductionReservoirs",
                       TransportMemoryKind::RestirGIReservoir,
                       TransportMemoryOwner::RestirGI,
                       width,
                       height,
                       pathDepth,
                       m_restirState.restirPtBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::PersistentHistory,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       allTransport);
    }

    if (restirPtActive && m_restirState.restirPtBytes > 0u) {
        registerRecord("RestirPTPathReservoirs",
                       TransportMemoryKind::RestirPTReservoir,
                       TransportMemoryOwner::RestirPT,
                       width,
                       height,
                       pathDepth,
                       m_restirState.restirPtBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::PersistentHistory,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
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
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::PersistentHistory,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       allTransport);
    }

    if (settings.causticTransportMode == RenderSettings::CausticTransportMode::PathSpacePrototype &&
        m_restirState.causticTransportBytes > 0u) {
        registerRecord("CausticTransport.PathSpacePrototype",
                       TransportMemoryKind::CausticPathSpace,
                       TransportMemoryOwner::CausticTransport,
                       width,
                       height,
                       m_restirState.causticMaxVertices,
                       m_restirState.causticTransportBytes,
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

    if (m_wavefront.totalBytes > 0u) {
        registerRecord("Wavefront.ActiveRays.Current",
                       TransportMemoryKind::WavefrontQueue,
                       TransportMemoryOwner::Wavefront,
                       m_wavefront.width,
                       m_wavefront.height,
                       m_wavefront.capacity,
                       m_wavefront.activeBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::Transient,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       allTransport);
        registerRecord("Wavefront.ActiveRays.Next",
                       TransportMemoryKind::WavefrontQueue,
                       TransportMemoryOwner::Wavefront,
                       m_wavefront.width,
                       m_wavefront.height,
                       m_wavefront.capacity,
                       m_wavefront.nextBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::Transient,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       allTransport);
        registerRecord("Wavefront.HitRecords",
                       TransportMemoryKind::WavefrontQueue,
                       TransportMemoryOwner::Wavefront,
                       m_wavefront.width,
                       m_wavefront.height,
                       m_wavefront.capacity,
                       m_wavefront.hitBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::Transient,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       allTransport);
        registerRecord("Wavefront.ShadowRays",
                       TransportMemoryKind::WavefrontQueue,
                       TransportMemoryOwner::Wavefront,
                       m_wavefront.width,
                       m_wavefront.height,
                       m_wavefront.shadowCapacity,
                       m_wavefront.shadowBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::Transient,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       allTransport);
        registerRecord("Wavefront.ShadowIndices",
                       TransportMemoryKind::WavefrontQueue,
                       TransportMemoryOwner::Wavefront,
                       m_wavefront.width,
                       m_wavefront.height,
                       m_wavefront.shadowCapacity,
                       m_wavefront.shadowIndexBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::Transient,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       allTransport);
        registerRecord("Wavefront.PathStates",
                       TransportMemoryKind::WavefrontQueue,
                       TransportMemoryOwner::Wavefront,
                       m_wavefront.width,
                       m_wavefront.height,
                       m_wavefront.capacity,
                       m_wavefront.pathStateBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::Transient,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       allTransport);
        if (m_wavefront.cacheQueryBytes > 0u) {
            registerRecord("Wavefront.RadianceCacheQueryItems",
                           TransportMemoryKind::WavefrontQueue,
                           TransportMemoryOwner::Wavefront,
                           m_wavefront.width,
                           m_wavefront.height,
                           m_wavefront.capacity,
                           m_wavefront.cacheQueryBytes,
                           TransportMemoryStorageClass::Private,
                           TransportMemoryHistoryPolicy::Transient,
                           TransportMemoryDebugReadbackPolicy::StagingCopy,
                           allTransport);
        }
        if (m_wavefront.cacheTrainRecordBytes > 0u) {
            registerRecord("Wavefront.RadianceCacheTrainRecords",
                           TransportMemoryKind::WavefrontQueue,
                           TransportMemoryOwner::Wavefront,
                           m_wavefront.width,
                           m_wavefront.height,
                           m_wavefront.capacity,
                           m_wavefront.cacheTrainRecordBytes,
                           TransportMemoryStorageClass::Private,
                           TransportMemoryHistoryPolicy::Transient,
                           TransportMemoryDebugReadbackPolicy::StagingCopy,
                           allTransport);
        }
        registerRecord("Wavefront.QueueCounters",
                       TransportMemoryKind::WavefrontQueue,
                       TransportMemoryOwner::Wavefront,
                       1u,
                       1u,
                       1u,
                       sizeof(QueueCounters),
                       TransportMemoryStorageClass::Shared,
                       TransportMemoryHistoryPolicy::Transient,
                       TransportMemoryDebugReadbackPolicy::DirectCpuVisible,
                       allTransport);
        registerRecord("Wavefront.IndirectDispatchArgs",
                       TransportMemoryKind::WavefrontQueue,
                       TransportMemoryOwner::Wavefront,
                       1u,
                       1u,
                       1u,
                       m_wavefront.indirectDispatchArgsBytes,
                       TransportMemoryStorageClass::Private,
                       TransportMemoryHistoryPolicy::Transient,
                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                       allTransport);
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
                       m_mlState.modelBytes,
                       TransportMemoryStorageClass::Private,
                       settings.mlBackendMode == RenderSettings::MlBackendMode::ReconstructionTiny
                           ? TransportMemoryHistoryPolicy::PersistentHistory
                           : TransportMemoryHistoryPolicy::Transient,
                       TransportMemoryDebugReadbackPolicy::MetadataOnly,
                       frameHistory);
    }

    if (!restirGiActive) {
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
    }
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
    const uint32_t ownershipGaps = m_transportMemory.ownershipGapRecordCount();
    if (ownershipGaps > 0u) {
        NSLog(@"[TransportMemory] ownership validation found %u nonzero records without "
              "FrameGraph or allocator ownership",
              ownershipGaps);
    }
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
    const bool forceSoftwareInitialCandidates =
        settings.enableSoftwareRayTracing || ForceSoftwareRestirDiInitialCandidates();
    static bool sLoggedRestirDiInitialSoftwareTraversal = false;
    if (forceSoftwareInitialCandidates &&
        provider.mode == PathTracerShaderTypes::IntersectionMode::HardwareRayTracing &&
        !sLoggedRestirDiInitialSoftwareTraversal) {
        sLoggedRestirDiInitialSoftwareTraversal = true;
        NSLog(@"[ReSTIR DI] explicit initial candidates use software traversal by request while HWRT remains active for the frame");
    }
    const TraversalServiceDecision traversal = SelectTraversalBackend(
        provider,
        TraversalServiceRequest{
            .kind = TraversalQueryKind::RestirDiVisibility,
            .softwareOverride = forceSoftwareInitialCandidates,
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
        [encoder setBuffer:BufferOrDummy(scene.environmentConditionalAliasBuffer(), m_dummyBuffer)
                    offset:0
                   atIndex:15];
        [encoder setBuffer:BufferOrDummy(scene.environmentMarginalAliasBuffer(), m_dummyBuffer)
                    offset:0
                   atIndex:16];
        [encoder setBuffer:BufferOrDummy(scene.environmentPdfBuffer(), m_dummyBuffer) offset:0 atIndex:17];
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
        [encoder setBuffer:BufferOrDummy(scene.environmentConditionalAliasBuffer(), m_dummyBuffer)
                    offset:0
                   atIndex:15];
        [encoder setBuffer:BufferOrDummy(scene.environmentMarginalAliasBuffer(), m_dummyBuffer)
                    offset:0
                   atIndex:16];
        [encoder setBuffer:BufferOrDummy(scene.environmentPdfBuffer(), m_dummyBuffer) offset:0 atIndex:17];
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
        if (scene.environmentConditionalAliasBuffer()) {
            [encoder useResource:scene.environmentConditionalAliasBuffer() usage:MTLResourceUsageRead];
        }
        if (scene.environmentMarginalAliasBuffer()) {
            [encoder useResource:scene.environmentMarginalAliasBuffer() usage:MTLResourceUsageRead];
        }
        if (scene.environmentPdfBuffer()) {
            [encoder useResource:scene.environmentPdfBuffer() usage:MTLResourceUsageRead];
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
        [encoder setBuffer:BufferOrDummy(provider.software.tlasNodes, m_dummyBuffer) offset:0 atIndex:16];
        [encoder setBuffer:BufferOrDummy(provider.software.tlasPrimitiveIndices, m_dummyBuffer) offset:0 atIndex:17];
        [encoder setBuffer:BufferOrDummy(provider.software.blasNodes, m_dummyBuffer) offset:0 atIndex:18];
        [encoder setBuffer:BufferOrDummy(provider.software.blasPrimitiveIndices, m_dummyBuffer) offset:0 atIndex:19];
        [encoder setBuffer:BufferOrDummy(provider.software.instanceInfoBuffer, m_dummyBuffer) offset:0 atIndex:20];
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
        static bool sLoggedMegakernelRestirStateFailure = false;
        if (!sLoggedMegakernelRestirStateFailure) {
            sLoggedMegakernelRestirStateFailure = true;
            NSLog(@"[Megakernel] ReSTIR state allocation failed; frame not dispatched");
        }
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
        m_gpuTiming.ranges.push_back(
            {GpuTimingBucket::FusedIntegrate, "Megakernel Integrate", startSample, endSample});
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
            .hardwarePipelineAvailable = pipelines.integrateHardwareAvailable(),
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

    // The hardware megakernel has a default-features specialization with every research and
    // debug gate folded to 0 (the direct-light mode stays live). Use it when all those gates are
    // off. The gates come from the
    // settings and do not change between the samples dispatched below, so one build decides.
    bool useDefaultFeatures = false;
    std::string liveGates;
    if (useHardware && pipelines.integrateHardwareAvailable()) {
        const PathtraceUniforms gateProbe = UniformBuilder::buildPathtraceUniforms(
            settings, accumulation, scene, accumulation.currentSize());
        useDefaultFeatures = UniformBuilder::usesOnlyDefaultFeatures(gateProbe);
        if (!useDefaultFeatures) liveGates = UniformBuilder::liveFeatureGates(gateProbe);
    }
    const int integrateVariant = !useHardware ? 0 : (useDefaultFeatures ? 2 : 1);
    if (integrateVariant != m_lastIntegrateVariant || liveGates != m_lastLiveGates) {
        m_lastIntegrateVariant = integrateVariant;
        m_lastLiveGates = liveGates;
        if (integrateVariant == 1) {
            NSLog(@"[Megakernel] integrate pipeline: hardware, general (gates on: %s)", liveGates.c_str());
        } else {
            NSLog(@"[Megakernel] integrate pipeline: %s%s%s",
                  integrateVariant == 2 ? "hardware, default features (research and debug gates off, direct-light mode live)"
                                        : "software (",
                  integrateVariant == 0 ? traversal.fallbackReason.c_str() : "",
                  integrateVariant == 0 ? ")" : "");
        }
    }
    logPathSpaceCausticsOnce(settings);

    id<MTLComputePipelineState> selectedPipeline = useHardware
        ? (useDefaultFeatures ? pipelines.integrateHardwareDefaultFeatures()
                              : pipelines.integrateHardware())
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
                [argumentEncoder setBuffer:BufferOrDummy(m_wavefront.cacheQueryItems, m_dummyBuffer)
                                    offset:0
                                   atIndex:PathTracerShaderTypes::kMaterialResourceRadianceCacheQueryItemIndex];
                [argumentEncoder setBuffer:BufferOrDummy(m_wavefront.cacheTrainRecords, m_dummyBuffer)
                                    offset:0
                                   atIndex:PathTracerShaderTypes::kMaterialResourceRadianceCacheTrainRecordIndex];
                MTLBufferHandle causticTransport =
                    m_transportAllocator.bufferForName("CausticTransport.PathSpacePrototype");
                const NSUInteger causticVertexCount =
                    causticTransport ? static_cast<NSUInteger>(m_restirState.causticMaxVertices) : 0u;
                const NSUInteger causticLightOffset =
                    causticVertexCount * sizeof(PathTracerShaderTypes::EyeSubpathVertex);
                const NSUInteger causticConnectionOffset =
                    causticLightOffset +
                    causticVertexCount * sizeof(PathTracerShaderTypes::LightSubpathVertex);
                const NSUInteger causticResultOffset =
                    causticConnectionOffset +
                    causticVertexCount * sizeof(PathTracerShaderTypes::PathConnection);
                [argumentEncoder setBuffer:BufferOrDummy(causticTransport, m_dummyBuffer)
                                    offset:0
                                   atIndex:PathTracerShaderTypes::kMaterialResourceCausticEyeVertexIndex];
                [argumentEncoder setBuffer:BufferOrDummy(causticTransport, m_dummyBuffer)
                                    offset:(causticTransport ? causticLightOffset : 0u)
                                   atIndex:PathTracerShaderTypes::kMaterialResourceCausticLightVertexIndex];
                [argumentEncoder setBuffer:BufferOrDummy(causticTransport, m_dummyBuffer)
                                    offset:(causticTransport ? causticConnectionOffset : 0u)
                                   atIndex:PathTracerShaderTypes::kMaterialResourceCausticConnectionIndex];
                [argumentEncoder setBuffer:BufferOrDummy(causticTransport, m_dummyBuffer)
                                    offset:(causticTransport ? causticResultOffset : 0u)
                                   atIndex:PathTracerShaderTypes::kMaterialResourceCausticConnectionResultIndex];
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
                    if (m_wavefront.cacheQueryItems) {
                        [encoder useResource:m_wavefront.cacheQueryItems
                                       usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                    }
                    if (m_wavefront.cacheTrainRecords) {
                        [encoder useResource:m_wavefront.cacheTrainRecords
                                       usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                    }
                    if (causticTransport) {
                        [encoder useResource:causticTransport
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
        // The megakernel leaves direct light to the explicit ReSTIR DI stage only when that stage is its screen
        // passes. A wavefront frame handed to the megakernel (encodeWavefrontMegakernelFallback) never ran the
        // wavefront's explicit stage, so it shades its own direct light, the fused restir_di or hybrid.
        const bool explicitRestirDiScreenLightingStage =
            m_explicitRestirDiActiveThisFrame &&
            !m_restirDiMegakernelIntegratedStageActiveThisFrame &&
            !m_restirDiWavefrontStageActiveThisFrame;
        uniforms.explicitRestirDiLightingStage =
            explicitRestirDiScreenLightingStage ? 1u : 0u;
        if (explicitRestirDiScreenLightingStage) {
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
        [encoder setTexture:accumulation.directRadianceSum() atIndex:8];
        [encoder setTexture:accumulation.indirectRadianceSum() atIndex:9];
        [encoder setTexture:accumulation.radianceMomentSum() atIndex:10];
        [encoder setTexture:accumulation.reconstructionMetricBuffer() atIndex:11];
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

// causticTransportMode=path_space is a prototype that adds no light transport: on both executors its hooks
// (caustic_transport_connect_light_sample) record environment and rect-light next-event connections, up to
// causticMaxVertices, and return the plain next-event contribution with the same balance-heuristic weight. Say so once
// per enable, since the mode's name promises caustics and it costs the megakernel its default-features pipeline.
void RenderLoop::logPathSpaceCausticsOnce(const RenderSettings& settings) {
    const bool pathSpaceCaustics =
        settings.causticTransportMode == RenderSettings::CausticTransportMode::PathSpacePrototype;
    if (pathSpaceCaustics && !m_loggedPathSpaceCausticsInert) {
        NSLog(@"[Caustics] causticTransportMode=path_space adds no caustic paths: it records environment and rect-light "
              @"next-event connections (up to causticMaxVertices=%u) and returns the same contribution as plain "
              @"next-event estimation, so the image is the one causticTransportMode=off gives (on the hardware megakernel, up "
              @"to the arithmetic order of its two pipelines). On the hardware megakernel it also selects the general pipeline.", settings.causticMaxVertices);
    }
    m_loggedPathSpaceCausticsInert = pathSpaceCaustics;
}

bool RenderLoop::ensureWavefrontResources(const SceneResources& scene,
                                          const RenderSettings& settings,
                                          uint32_t width,
                                          uint32_t height) {
    const uint64_t capacity64 = std::max<uint64_t>(1u, static_cast<uint64_t>(width) * static_cast<uint64_t>(height));
    if (capacity64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        NSLog(@"[Wavefront] render target too large for fixed queue capacity");
        return false;
    }
    const uint64_t shadowStride = static_cast<uint64_t>(kWavefrontShadowCapacityMultiplier) +
        (settings.referencePointExactNee ? scene.zeroRadiusReferencePointLightCount() : 0u);
    const uint64_t shadowCapacity64 = capacity64 * shadowStride;
    if (shadowCapacity64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        NSLog(@"[Wavefront] render target too large for fixed shadow queue capacity");
        return false;
    }
    const uint32_t capacity = static_cast<uint32_t>(capacity64);
    const uint32_t shadowCapacity = static_cast<uint32_t>(shadowCapacity64);
    const bool cacheQueuesEnabled = RadianceCacheActive(settings);
    const bool resolvedSurfacesEnabled =
        settings.wavefrontShadeIndirectEnabled || WavefrontShadeIndirectEnabled();
    if (m_wavefront.capacity == capacity &&
        m_wavefront.shadowCapacity == shadowCapacity &&
        m_wavefront.width == width &&
        m_wavefront.height == height &&
        m_wavefront.cacheQueuesEnabled == cacheQueuesEnabled &&
        m_wavefront.resolvedSurfacesEnabled == resolvedSurfacesEnabled &&
        m_wavefront.activeRays[0] &&
        m_wavefront.activeRays[1] &&
        m_wavefront.hitRecords &&
        m_wavefront.shadowRays &&
        m_wavefront.shadowIndices &&
        m_wavefront.pathStates &&
        m_wavefront.queueCounters &&
        (!cacheQueuesEnabled || (m_wavefront.cacheQueryItems && m_wavefront.cacheTrainRecords)) &&
        (!resolvedSurfacesEnabled || m_wavefront.resolvedSurfaces) &&
        m_wavefront.indirectDispatchArgs) {
        m_wavefront.budgetBytes = scene.recommendedMaxWorkingSetSizeBytes();
        return true;
    }

    const NSUInteger activeBytes = static_cast<NSUInteger>(capacity64 * sizeof(WavefrontRay));
    const NSUInteger hitBytes = static_cast<NSUInteger>(capacity64 * sizeof(WavefrontHitRecord));
    const NSUInteger shadowBytes = static_cast<NSUInteger>(shadowCapacity64 * sizeof(ShadowRay));
    const NSUInteger shadowIndexBytes = static_cast<NSUInteger>(shadowCapacity64 * sizeof(uint32_t));
    const NSUInteger pathStateBytes = static_cast<NSUInteger>(capacity64 * sizeof(PathState));
    const NSUInteger resolvedSurfaceBytes =
        resolvedSurfacesEnabled
            ? static_cast<NSUInteger>(capacity64 * sizeof(WavefrontResolvedSurface))
            : 0u;
    const NSUInteger cacheWorkBytes =
        cacheQueuesEnabled
            ? static_cast<NSUInteger>(capacity64 *
                                      sizeof(PathTracerShaderTypes::RadianceCacheWorkItem))
            : 0u;
    // Radiance cache training records: kRadianceCacheTrainRecordCount per training pixel, and one pixel in
    // kRadianceCacheTrainingPathStride trains in a frame (shaders/wavefront/kernels.metal).
    const uint64_t cacheTrainGroups64 =
        (capacity64 + PathTracerShaderTypes::kRadianceCacheTrainingPathStride - 1u) /
        PathTracerShaderTypes::kRadianceCacheTrainingPathStride;
    const NSUInteger cacheTrainRecordBytes =
        cacheQueuesEnabled
            ? static_cast<NSUInteger>(cacheTrainGroups64 * PathTracerShaderTypes::kRadianceCacheTrainRecordCount *
                                      sizeof(PathTracerShaderTypes::RadianceCacheTrainRecord))
            : 0u;
    const NSUInteger counterBytes = sizeof(QueueCounters);
    const NSUInteger indirectDispatchArgsBytes = static_cast<NSUInteger>(3u * sizeof(uint32_t));

    auto makeWavefrontBuffer =
        [&](const char* name,
            NSUInteger length,
            StorageClass storage,
            ResourceLifetime lifetime,
            TransportMemoryDebugReadbackPolicy debug,
            const char* label) -> MTLBufferHandle {
        TransportMemoryRecord record{};
        record.name = name;
        record.kind = TransportMemoryKind::WavefrontQueue;
        record.owner = TransportMemoryOwner::Wavefront;
        record.targetOwner = TransportMemoryOwner::Wavefront;
        record.frameGraphGroup = RenderResourceGroup::WavefrontResources;
        record.lifetime = lifetime;
        record.width = width;
        record.height = height;
        record.depthOrCellCount = capacity;
        record.bytes = static_cast<uint64_t>(length);
        record.history = TransportMemoryHistoryPolicy::Transient;
        record.debug = debug;
        record.invalidatesOn = AllTransportInvalidations();
        record.frameGraphOwned = true;
        record.transportMemoryOwned = true;
        return m_transportAllocator.allocateBuffer(m_device, record, storage, label);
    };

    MTLBufferHandle active0 = makeWavefrontBuffer("Wavefront.ActiveRays.Current",
                                                  activeBytes,
                                                  StorageClass::Private,
                                                  ResourceLifetime::Transient,
                                                  TransportMemoryDebugReadbackPolicy::StagingCopy,
                                                  "TransportMemory.Wavefront.ActiveRays.Current");
    MTLBufferHandle active1 = makeWavefrontBuffer("Wavefront.ActiveRays.Next",
                                                  activeBytes,
                                                  StorageClass::Private,
                                                  ResourceLifetime::Transient,
                                                  TransportMemoryDebugReadbackPolicy::StagingCopy,
                                                  "TransportMemory.Wavefront.ActiveRays.Next");
    MTLBufferHandle hits = makeWavefrontBuffer("Wavefront.HitRecords",
                                               hitBytes,
                                               StorageClass::Private,
                                               ResourceLifetime::Transient,
                                               TransportMemoryDebugReadbackPolicy::StagingCopy,
                                               "TransportMemory.Wavefront.HitRecords");
    MTLBufferHandle shadows = makeWavefrontBuffer("Wavefront.ShadowRays",
                                                  shadowBytes,
                                                  StorageClass::Private,
                                                  ResourceLifetime::Transient,
                                                  TransportMemoryDebugReadbackPolicy::StagingCopy,
                                                  "TransportMemory.Wavefront.ShadowRays");
    MTLBufferHandle shadowIndices = makeWavefrontBuffer("Wavefront.ShadowIndices",
                                                        shadowIndexBytes,
                                                        StorageClass::Private,
                                                        ResourceLifetime::Transient,
                                                        TransportMemoryDebugReadbackPolicy::StagingCopy,
                                                        "TransportMemory.Wavefront.ShadowIndices");
    MTLBufferHandle pathStates = makeWavefrontBuffer("Wavefront.PathStates",
                                                     pathStateBytes,
                                                     StorageClass::Private,
                                                     ResourceLifetime::Transient,
                                                     TransportMemoryDebugReadbackPolicy::StagingCopy,
                                                     "TransportMemory.Wavefront.PathStates");
    MTLBufferHandle cacheQueryItems = cacheQueuesEnabled
        ? makeWavefrontBuffer("Wavefront.RadianceCacheQueryItems",
                              cacheWorkBytes,
                              StorageClass::Private,
                              ResourceLifetime::Transient,
                              TransportMemoryDebugReadbackPolicy::StagingCopy,
                              "TransportMemory.Wavefront.RadianceCacheQueryItems")
        : nil;
    MTLBufferHandle cacheTrainRecords = cacheQueuesEnabled
        ? makeWavefrontBuffer("Wavefront.RadianceCacheTrainRecords",
                              cacheTrainRecordBytes,
                              StorageClass::Private,
                              ResourceLifetime::Transient,
                              TransportMemoryDebugReadbackPolicy::StagingCopy,
                              "TransportMemory.Wavefront.RadianceCacheTrainRecords")
        : nil;
    MTLBufferHandle resolvedSurfaces =
        resolvedSurfacesEnabled
            ? makeWavefrontBuffer("Wavefront.ResolvedSurfaces",
                                  resolvedSurfaceBytes,
                                  StorageClass::Private,
                                  ResourceLifetime::Transient,
                                  TransportMemoryDebugReadbackPolicy::StagingCopy,
                                  "TransportMemory.Wavefront.ResolvedSurfaces")
            : nil;
    MTLBufferHandle counters = makeWavefrontBuffer("Wavefront.QueueCounters",
                                                   counterBytes,
                                                   StorageClass::Shared,
                                                   ResourceLifetime::DebugReadback,
                                                   TransportMemoryDebugReadbackPolicy::DirectCpuVisible,
                                                   "TransportMemory.Wavefront.QueueCounters");
    MTLBufferHandle indirectArgs = makeWavefrontBuffer("Wavefront.IndirectDispatchArgs",
                                                       indirectDispatchArgsBytes,
                                                       StorageClass::Private,
                                                       ResourceLifetime::Transient,
                                                       TransportMemoryDebugReadbackPolicy::StagingCopy,
                                                       "TransportMemory.Wavefront.IndirectDispatchArgs");
    if (!active0 || !active1 || !hits || !shadows || !shadowIndices || !pathStates ||
        (cacheQueuesEnabled && (!cacheQueryItems || !cacheTrainRecords)) || !counters ||
        (resolvedSurfacesEnabled && !resolvedSurfaces) || !indirectArgs) {
        NSLog(@"[Wavefront] failed to allocate queue buffers");
        return false;
    }

    m_wavefront.activeRays[0] = active0;
    m_wavefront.activeRays[1] = active1;
    m_wavefront.hitRecords = hits;
    m_wavefront.shadowRays = shadows;
    m_wavefront.shadowIndices = shadowIndices;
    m_wavefront.pathStates = pathStates;
    m_wavefront.queueCounters = counters;
    m_wavefront.cacheQueryItems = cacheQueryItems;
    m_wavefront.cacheTrainRecords = cacheTrainRecords;
    // Training records are valid only between a training path's shade and the train kernel, which invalidates
    // them; a new buffer is zeroed before its first use (encodeWavefront...).
    m_wavefront.cacheTrainRecordsNeedClear = cacheTrainRecords != nullptr;
    m_wavefront.resolvedSurfaces = resolvedSurfaces;
    m_wavefront.indirectDispatchArgs = indirectArgs;
    m_wavefront.width = width;
    m_wavefront.height = height;
    m_wavefront.capacity = capacity;
    m_wavefront.shadowCapacity = shadowCapacity;
    m_wavefront.cacheQueuesEnabled = cacheQueuesEnabled;
    m_wavefront.resolvedSurfacesEnabled = resolvedSurfacesEnabled;
    m_wavefront.activeBytes = activeBytes;
    m_wavefront.nextBytes = activeBytes;
    m_wavefront.hitBytes = hitBytes;
    m_wavefront.shadowBytes = shadowBytes;
    m_wavefront.shadowIndexBytes = shadowIndexBytes;
    m_wavefront.pathStateBytes = pathStateBytes;
    m_wavefront.cacheQueryBytes = cacheWorkBytes;
    m_wavefront.cacheTrainRecordBytes = cacheTrainRecordBytes;
    m_wavefront.resolvedSurfaceBytes = resolvedSurfaceBytes;
    m_wavefront.indirectDispatchArgsBytes = indirectDispatchArgsBytes;
    m_wavefront.totalBytes =
        static_cast<uint64_t>(activeBytes) * 2u +
        static_cast<uint64_t>(hitBytes) +
        static_cast<uint64_t>(shadowBytes) +
        static_cast<uint64_t>(shadowIndexBytes) +
        static_cast<uint64_t>(pathStateBytes) +
        static_cast<uint64_t>(cacheWorkBytes) * 2u +
        static_cast<uint64_t>(resolvedSurfaceBytes) +
        static_cast<uint64_t>(counterBytes) +
        static_cast<uint64_t>(indirectDispatchArgsBytes);
    m_wavefront.budgetBytes = scene.recommendedMaxWorkingSetSizeBytes();

    const double budgetUsagePct =
        (m_wavefront.budgetBytes > 0u)
            ? (static_cast<double>(m_wavefront.totalBytes) /
               static_cast<double>(m_wavefront.budgetBytes)) * 100.0
            : 0.0;
    NSLog(@"[Wavefront] queueFootprint active=%.2f MiB next=%.2f MiB hit=%.2f MiB shadow=%.2f MiB "
          "shadowIndex=%.2f MiB path=%.2f MiB cache=%.2f MiB resolved=%.2f MiB indirect=%.2f KiB total=%.2f MiB "
          "recommendedMaxWorkingSetSize=%.2f MiB usage=%.2f%%",
          BytesToMiB(m_wavefront.activeBytes),
          BytesToMiB(m_wavefront.nextBytes),
          BytesToMiB(m_wavefront.hitBytes),
          BytesToMiB(m_wavefront.shadowBytes),
          BytesToMiB(m_wavefront.shadowIndexBytes),
          BytesToMiB(m_wavefront.pathStateBytes),
          BytesToMiB(m_wavefront.cacheQueryBytes + m_wavefront.cacheTrainRecordBytes),
          BytesToMiB(m_wavefront.resolvedSurfaceBytes),
          static_cast<double>(m_wavefront.indirectDispatchArgsBytes) / 1024.0,
          BytesToMiB(m_wavefront.totalBytes),
          BytesToMiB(m_wavefront.budgetBytes),
          budgetUsagePct);
    return true;
}

uint32_t RenderLoop::encodeWavefrontMegakernelFallback(const char* reason,
                                                      MTLCommandBufferHandle commandBuffer,
                                                      Accumulation& accumulation,
                                                      const Pipelines& pipelines,
                                                      const SceneResources& scene,
                                                      const RenderSettings& settings) {
    if (m_loggedWavefrontMegakernelFallback != reason) {
        m_loggedWavefrontMegakernelFallback = reason;
        NSLog(@"[Wavefront] %s; falling back to megakernel", reason);
    }
    m_wavefrontMegakernelFallback = reason;
    m_wavefrontMegakernelFallbackThisFrame = true;
    // The wavefront's explicit ReSTIR DI stage does not run on the megakernel; encodeIntegration shades the fused
    // direct light instead (an unbiased estimate of the same light). Say so in the log and the render metadata.
    m_wavefrontExplicitRestirDiRanFused = m_restirDiWavefrontStageActiveThisFrame;
    if (m_wavefrontExplicitRestirDiRanFused) {
        // No explicit stage runs this frame: encodeIntegration must not ask ensureRestirStateResources for its
        // reservoirs (their allocation may be what failed) or leave direct light to it.
        m_explicitRestirDiActiveThisFrame = false;
        m_restirDiPathStatus = "fused_megakernel_wavefront_fallback";
        if (!m_loggedWavefrontExplicitRestirDiRanFused) {
            m_loggedWavefrontExplicitRestirDiRanFused = true;
            NSLog(@"[Wavefront] the explicit ReSTIR DI stage does not run on the megakernel fallback; rendering "
                  @"the megakernel's fused direct light for this mode");
        }
    }
    return encodeIntegration(commandBuffer, accumulation, pipelines, scene, settings);
}

uint32_t RenderLoop::encodeWavefront(MTLCommandBufferHandle commandBuffer,
                                     Accumulation& accumulation,
                                     const Pipelines& pipelines,
                                     const SceneResources& scene,
                                     const RenderSettings& settings) {
    const auto makeEncoder = [&](GpuTimingBucket bucket,
                                 NSString* label,
                                 const char* passName = nullptr) -> id<MTLComputeCommandEncoder> {
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
            std::string timingPassName;
            if (passName && passName[0] != '\0') {
                timingPassName = passName;
            } else if (label) {
                timingPassName = std::string([label UTF8String]);
            }
            m_gpuTiming.ranges.push_back({bucket, timingPassName, startSample, endSample});
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
        !pipelines.wavefrontShadeResolveSurface() ||
        !pipelines.wavefrontShadeResolved() ||
        !pipelines.wavefrontShadeIndirectProbe() ||
        !pipelines.wavefrontShadeSurfaceProbe() ||
        !pipelines.wavefrontShadeMetadataProbe() ||
        !pipelines.wavefrontShadeScalarProbe() ||
        !pipelines.wavefrontShadeClosureProbe() ||
        !pipelines.wavefrontShadeBsdfProbe() ||
        !pipelines.wavefrontBuildShadowIndices() ||
        !pipelines.wavefrontShadow() ||
        !pipelines.wavefrontShadowIndexed() ||
        !pipelines.wavefrontSwapQueues() ||
        !pipelines.wavefrontRadianceCacheTrain() ||
        !pipelines.wavefrontAccumulate()) {
        return encodeWavefrontMegakernelFallback("required pipelines missing", commandBuffer, accumulation,
                                                 pipelines, scene, settings);
    }
    // Test hook: takes the fallback above with every pipeline present, so the megakernel's handling of a
    // wavefront frame can be measured.
    static const bool forceMegakernelFallback = EnvFlagEnabled("PT_WAVEFRONT_FORCE_MEGAKERNEL_FALLBACK");
    if (forceMegakernelFallback) {
        return encodeWavefrontMegakernelFallback("forced by PT_WAVEFRONT_FORCE_MEGAKERNEL_FALLBACK", commandBuffer,
                                                 accumulation, pipelines, scene, settings);
    }
    if (m_restirDiWavefrontStageActiveThisFrame && !pipelines.wavefrontRestirDiDirectLighting()) {
        return encodeWavefrontMegakernelFallback("explicit ReSTIR DI direct-lighting pipeline missing", commandBuffer,
                                                 accumulation, pipelines, scene, settings);
    }

    const CGSize currentSize = accumulation.currentSize();
    const uint32_t width = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.width));
    const uint32_t height = static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.height));
    if (!ensureRestirStateResources(width, height, settings, commandBuffer)) {
        return encodeWavefrontMegakernelFallback("ReSTIR state allocation failed", commandBuffer, accumulation,
                                                 pipelines, scene, settings);
    }
    if (!ensureWavefrontResources(scene, settings, width, height)) {
        return encodeWavefrontMegakernelFallback("queue allocation failed", commandBuffer, accumulation,
                                                 pipelines, scene, settings);
    }
    logPathSpaceCausticsOnce(settings);
    if (m_wavefront.cacheTrainRecordsNeedClear && m_wavefront.cacheTrainRecords) {
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        if (!blit) {
            NSLog(@"[Wavefront] radiance cache training records not cleared; frame not dispatched");
            return 0u;
        }
        blit.label = @"Wavefront Clear Radiance Cache Train Records";
        [blit fillBuffer:m_wavefront.cacheTrainRecords
                   range:NSMakeRange(0u, static_cast<NSUInteger>(m_wavefront.cacheTrainRecordBytes))
                   value:0u];
        [blit endEncoding];
        m_wavefront.cacheTrainRecordsNeedClear = false;
    }
    rebuildTransportMemoryRegistry(settings);

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
    const bool guardedShadeIndirect =
        settings.wavefrontShadeIndirectEnabled || WavefrontShadeIndirectEnabled();
    const bool guardedShadeIndirectProbe =
        settings.wavefrontShadeIndirectProbeEnabled || WavefrontShadeIndirectProbeEnabled();
    const bool guardedShadeSurfaceProbe =
        settings.wavefrontShadeSurfaceProbeEnabled || WavefrontShadeSurfaceProbeEnabled();
    const bool guardedShadeMetadataProbe =
        settings.wavefrontShadeMetadataProbeEnabled || WavefrontShadeMetadataProbeEnabled();
    const bool guardedShadeScalarProbe =
        settings.wavefrontShadeScalarProbeEnabled || WavefrontShadeScalarProbeEnabled();
    const bool guardedShadeBsdfProbe =
        settings.wavefrontShadeBsdfProbeEnabled || WavefrontShadeBsdfProbeEnabled();
    const bool guardedShadeClosureProbe =
        settings.wavefrontShadeClosureProbeEnabled || WavefrontShadeClosureProbeEnabled();
    // Indexed shadow threads write whole PathState objects and can race when
    // several exact points target one path. Use the existing serial per-path
    // shadow kernel for this option; default/no-point dispatch is unchanged.
    const bool guardedShadowIndirect =
        (settings.wavefrontShadowIndirectEnabled || WavefrontShadowIndirectEnabled()) &&
        !(settings.referencePointExactNee && scene.zeroRadiusReferencePointLightCount() > 0u);
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
            makeEncoder(GpuTimingBucket::ShadowQueue,
                        @"Wavefront Build Dispatch Args",
                        "Wavefront Prepare Intersections");
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
        if (m_explicitRestirDiActiveThisFrame && !m_restirDiWavefrontStageActiveThisFrame) {
            uniforms.emissivePrimitiveCount = 0u;
        }
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
            if (sample == 0u) {
                return encodeWavefrontMegakernelFallback("per-sample uniform allocation failed", commandBuffer,
                                                         accumulation, pipelines, scene, settings);
            }
            // Samples 0..sample-1 are fully encoded (each ends with its accumulate dispatch); handing the whole
            // frame to the megakernel now would render them twice.
            NSLog(@"[Wavefront] per-sample uniform allocation failed after %u of %u samples; the frame ends early",
                  sample, samplesThisFrame);
            return sample;
        }
        [sampleUniformBuffers addObject:sampleUniformBuffer];

        int currentQueueIndex = 0;
        int nextQueueIndex = 1;

        {
            id<MTLComputeCommandEncoder> encoder =
                makeEncoder(GpuTimingBucket::ShadowQueue,
                            @"Wavefront Reset",
                            "Wavefront Reset Queues");
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
                makeEncoder(GpuTimingBucket::ShadowQueue,
                            @"Wavefront Generate Primary",
                            "Wavefront Generate Primary");
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
                    makeEncoder(GpuTimingBucket::ShadowQueue,
                                @"Wavefront Prepare Iteration",
                                "Wavefront Prepare Intersections");
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
                    makeEncoder(GpuTimingBucket::Compaction,
                                @"Wavefront Compact Queue",
                                "Wavefront Compact Queue");
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

            auto bindIntersectionAlphaResources = [&](id<MTLComputeCommandEncoder> encoder) {
                if (auto argumentEncoder = pipelines.materialResourcesArgumentEncoder();
                    argumentEncoder && argumentEncoder.encodedLength > 0u) {
                    const NSUInteger encodedLength = argumentEncoder.encodedLength;
                    if (!m_materialResourcesArgumentBuffer ||
                        m_materialResourcesArgumentBuffer.length < encodedLength) {
                        m_materialResourcesArgumentBuffer =
                            [m_device newBufferWithLength:encodedLength
                                                 options:MTLResourceStorageModeShared];
                        if (m_materialResourcesArgumentBuffer) {
                            std::memset([m_materialResourcesArgumentBuffer contents], 0, encodedLength);
                        }
                    }
                    if (m_materialResourcesArgumentBuffer) {
                        [argumentEncoder setArgumentBuffer:m_materialResourcesArgumentBuffer offset:0];
                        if (!materialTextures.empty()) {
                            const NSUInteger textureCount =
                                std::min<NSUInteger>(materialTextures.size(),
                                                     PathTracerShaderTypes::kMaxMaterialTextures);
                            [argumentEncoder setTextures:materialTextures.data()
                                               withRange:NSMakeRange(0, textureCount)];
                        }
                        if (!materialSamplers.empty()) {
                            const NSUInteger samplerCount =
                                std::min<NSUInteger>(materialSamplers.size(), kMaxMaterialSamplers);
                            [argumentEncoder setSamplerStates:materialSamplers.data()
                                                    withRange:NSMakeRange(PathTracerShaderTypes::kMaxMaterialTextures,
                                                                          samplerCount)];
                        }
                    }
                }
                [encoder setBuffer:BufferOrDummy(scene.materialBuffer(), m_dummyBuffer) offset:0 atIndex:18];
                [encoder setBuffer:BufferOrDummy(scene.materialTextureInfoBuffer(), m_dummyBuffer)
                           offset:0 atIndex:19];
                [encoder setBuffer:m_materialResourcesArgumentBuffer offset:0 atIndex:23];
                if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                    if (m_materialResourcesArgumentBuffer) {
                        [encoder useResource:m_materialResourcesArgumentBuffer usage:MTLResourceUsageRead];
                    }
                    if (!SkipTextureUseResources()) {
                        for (id<MTLTexture> texture : materialTextures) {
                            if (texture) {
                                [encoder useResource:texture usage:MTLResourceUsageRead];
                            }
                        }
                    }
                }
            };

            if (canHardwareIntersect) {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::Intersect,
                                @"Wavefront Intersect (HW)",
                                "Wavefront Intersect");
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
                bindIntersectionAlphaResources(encoder);
                dispatchWavefrontStage(encoder, intersectThreadsPerThreadgroup, useIndirectIntersect);
                [encoder endEncoding];
            } else {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::Intersect,
                                @"Wavefront Intersect",
                                "Wavefront Intersect");
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
                bindIntersectionAlphaResources(encoder);
                dispatchWavefrontStage(encoder, intersectThreadsPerThreadgroup, useIndirectIntersect);
                [encoder endEncoding];
            }

            if (m_restirDiWavefrontStageActiveThisFrame) {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::Shade,
                                @"Wavefront ReSTIR DI Direct Lighting",
                                "Wavefront ReSTIR DI Direct Lighting");
                [encoder setComputePipelineState:pipelines.wavefrontRestirDiDirectLighting()];
                if (canHardwareIntersect &&
                    provider.hardware.tlas &&
                    [encoder respondsToSelector:@selector(setAccelerationStructure:atBufferIndex:)]) {
                    [encoder setAccelerationStructure:provider.hardware.tlas atBufferIndex:18];
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
                            [argumentEncoder setBuffer:BufferOrDummy(m_wavefront.cacheQueryItems, m_dummyBuffer)
                                                offset:0
                                               atIndex:PathTracerShaderTypes::kMaterialResourceRadianceCacheQueryItemIndex];
                            [argumentEncoder setBuffer:BufferOrDummy(m_wavefront.cacheTrainRecords, m_dummyBuffer)
                                                offset:0
                                               atIndex:PathTracerShaderTypes::kMaterialResourceRadianceCacheTrainRecordIndex];
                            MTLBufferHandle causticTransport =
                                m_transportAllocator.bufferForName("CausticTransport.PathSpacePrototype");
                            const NSUInteger causticVertexCount =
                                causticTransport ? static_cast<NSUInteger>(m_restirState.causticMaxVertices) : 0u;
                            const NSUInteger causticLightOffset =
                                causticVertexCount * sizeof(PathTracerShaderTypes::EyeSubpathVertex);
                            const NSUInteger causticConnectionOffset =
                                causticLightOffset +
                                causticVertexCount * sizeof(PathTracerShaderTypes::LightSubpathVertex);
                            const NSUInteger causticResultOffset =
                                causticConnectionOffset +
                                causticVertexCount * sizeof(PathTracerShaderTypes::PathConnection);
                            [argumentEncoder setBuffer:BufferOrDummy(causticTransport, m_dummyBuffer)
                                                offset:0
                                               atIndex:PathTracerShaderTypes::kMaterialResourceCausticEyeVertexIndex];
                            [argumentEncoder setBuffer:BufferOrDummy(causticTransport, m_dummyBuffer)
                                                offset:(causticTransport ? causticLightOffset : 0u)
                                               atIndex:PathTracerShaderTypes::kMaterialResourceCausticLightVertexIndex];
                            [argumentEncoder setBuffer:BufferOrDummy(causticTransport, m_dummyBuffer)
                                                offset:(causticTransport ? causticConnectionOffset : 0u)
                                               atIndex:PathTracerShaderTypes::kMaterialResourceCausticConnectionIndex];
                            [argumentEncoder setBuffer:BufferOrDummy(causticTransport, m_dummyBuffer)
                                                offset:(causticTransport ? causticResultOffset : 0u)
                                               atIndex:PathTracerShaderTypes::kMaterialResourceCausticConnectionResultIndex];
                            [encoder setBuffer:m_materialResourcesArgumentBuffer offset:0 atIndex:23];
                            if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                                [encoder useResource:m_materialResourcesArgumentBuffer
                                               usage:MTLResourceUsageRead];
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
                                if (m_wavefront.cacheQueryItems) {
                                    [encoder useResource:m_wavefront.cacheQueryItems
                                                   usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                                }
                                if (m_wavefront.cacheTrainRecords) {
                                    [encoder useResource:m_wavefront.cacheTrainRecords
                                                   usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                                }
                                if (causticTransport) {
                                    [encoder useResource:causticTransport
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

            // The shade kernel has the hardware megakernel's default-features specialization (function constant 2:
            // every research and debug gate folded to 0, the direct-light mode live). Use it when all those gates are
            // off in this frame's uniforms; it is built on first use, and the general pipeline stays if that fails.
            id<MTLComputePipelineState> shadePipeline = pipelines.wavefrontShade();
            id<MTLComputePipelineState> shadeResolvedPipeline = pipelines.wavefrontShadeResolved();
            const bool shadeDefaultFeatures =
                m_hasCurrentPathtraceUniforms && UniformBuilder::usesOnlyDefaultFeatures(m_currentPathtraceUniforms);
            bool shadeDefaultFeaturesUsed = false;
            if (shadeDefaultFeatures) {
                // Only the variant this frame dispatches is fetched, so a run builds one of the two.
                const bool shadeResolvedNeeded = guardedShadeIndirect && m_wavefront.resolvedSurfaces != nil;
                id<MTLComputePipelineState> defaultShade =
                    shadeResolvedNeeded ? nil : pipelines.wavefrontShadeDefaultFeatures();
                id<MTLComputePipelineState> defaultShadeResolved =
                    shadeResolvedNeeded ? pipelines.wavefrontShadeResolvedDefaultFeatures() : nil;
                if (defaultShade) {
                    shadePipeline = defaultShade;
                    shadeDefaultFeaturesUsed = true;
                }
                if (defaultShadeResolved) {
                    shadeResolvedPipeline = defaultShadeResolved;
                    shadeDefaultFeaturesUsed = true;
                }
            }
            const std::string shadeLiveGates = shadeDefaultFeatures || !m_hasCurrentPathtraceUniforms
                ? std::string() : UniformBuilder::liveFeatureGates(m_currentPathtraceUniforms);
            const int shadeVariant = shadeDefaultFeaturesUsed ? 1 : 0;
            if (shadeVariant != m_lastWavefrontShadeVariant || shadeLiveGates != m_lastWavefrontShadeLiveGates) {
                m_lastWavefrontShadeVariant = shadeVariant;
                m_lastWavefrontShadeLiveGates = shadeLiveGates;
                if (shadeDefaultFeaturesUsed) {
                    NSLog(@"[Wavefront] shade pipeline: default features (research and debug gates off, direct-light mode live)");
                } else if (shadeDefaultFeatures) {
                    NSLog(@"[Wavefront] shade pipeline: general (the default-features pipeline failed to build)");
                } else {
                    NSLog(@"[Wavefront] shade pipeline: general (gates on: %s)", shadeLiveGates.c_str());
                }
            }
            const MTLSize shadeThreadsPerThreadgroup =
                PreferredLinearThreadgroupSize(shadePipeline);
            const MTLSize shadeResolveSurfaceThreadsPerThreadgroup =
                PreferredLinearThreadgroupSize(pipelines.wavefrontShadeResolveSurface());
            const MTLSize shadeResolvedThreadsPerThreadgroup =
                PreferredLinearThreadgroupSize(shadeResolvedPipeline);
            const bool useIndirectShadeProbe =
                guardedShadeIndirectProbe &&
                encodeWavefrontDispatchArgs(
                    static_cast<uint32_t>(PathTracerShaderTypes::WavefrontQueueType::PrimaryRay),
                    shadeThreadsPerThreadgroup);
            const bool useIndirectShadeSurfaceProbe =
                guardedShadeSurfaceProbe &&
                encodeWavefrontDispatchArgs(
                    static_cast<uint32_t>(PathTracerShaderTypes::WavefrontQueueType::PrimaryRay),
                    shadeThreadsPerThreadgroup);
            const bool useIndirectShadeMetadataProbe =
                guardedShadeMetadataProbe &&
                encodeWavefrontDispatchArgs(
                    static_cast<uint32_t>(PathTracerShaderTypes::WavefrontQueueType::PrimaryRay),
                    shadeThreadsPerThreadgroup);
            const bool useIndirectShadeScalarProbe =
                guardedShadeScalarProbe &&
                encodeWavefrontDispatchArgs(
                    static_cast<uint32_t>(PathTracerShaderTypes::WavefrontQueueType::PrimaryRay),
                    shadeThreadsPerThreadgroup);
            const bool useIndirectShadeBsdfProbe =
                guardedShadeBsdfProbe &&
                encodeWavefrontDispatchArgs(
                    static_cast<uint32_t>(PathTracerShaderTypes::WavefrontQueueType::PrimaryRay),
                    shadeThreadsPerThreadgroup);
            const bool useIndirectShadeClosureProbe =
                guardedShadeClosureProbe &&
                encodeWavefrontDispatchArgs(
                    static_cast<uint32_t>(PathTracerShaderTypes::WavefrontQueueType::PrimaryRay),
                    shadeThreadsPerThreadgroup);
            const bool useIndirectShade =
                guardedShadeIndirect && m_wavefront.resolvedSurfaces != nil;

            if (useIndirectShadeProbe) {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::Shade,
                                @"Wavefront Shade Indirect Probe",
                                "Wavefront Shade");
                [encoder setComputePipelineState:pipelines.wavefrontShadeIndirectProbe()];
                [encoder setBuffer:sampleUniformBuffer offset:0 atIndex:0];
                [encoder setBuffer:m_wavefront.activeRays[currentQueueIndex] offset:0 atIndex:1];
                [encoder setBuffer:m_wavefront.hitRecords offset:0 atIndex:2];
                [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:5];
                [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:6];
                dispatchWavefrontStage(encoder, shadeThreadsPerThreadgroup, true);
                [encoder endEncoding];
            }

            if (useIndirectShadeSurfaceProbe) {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::Shade,
                                @"Wavefront Shade Surface Probe",
                                "Wavefront Shade");
                [encoder setComputePipelineState:pipelines.wavefrontShadeSurfaceProbe()];
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
                            [encoder setBuffer:m_materialResourcesArgumentBuffer offset:0 atIndex:23];
                            if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                                [encoder useResource:m_materialResourcesArgumentBuffer
                                               usage:MTLResourceUsageRead];
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
                [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:5];
                [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:6];
                [encoder setBuffer:BufferOrDummy(scene.materialBuffer(), m_dummyBuffer) offset:0 atIndex:8];
                [encoder setBuffer:BufferOrDummy(scene.meshInfoBuffer(), m_dummyBuffer) offset:0 atIndex:14];
                [encoder setBuffer:BufferOrDummy(scene.meshVertexBuffer(), m_dummyBuffer) offset:0 atIndex:15];
                [encoder setBuffer:BufferOrDummy(scene.meshIndexBuffer(), m_dummyBuffer) offset:0 atIndex:16];
                [encoder setBuffer:BufferOrDummy(scene.materialTextureInfoBuffer(), m_dummyBuffer)
                           offset:0
                          atIndex:17];
                dispatchWavefrontStage(encoder, shadeThreadsPerThreadgroup, true);
                [encoder endEncoding];
            }

            if (useIndirectShadeMetadataProbe) {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::Shade,
                                @"Wavefront Shade Metadata Probe",
                                "Wavefront Shade");
                [encoder setComputePipelineState:pipelines.wavefrontShadeMetadataProbe()];
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
                            [encoder setBuffer:m_materialResourcesArgumentBuffer offset:0 atIndex:23];
                            if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                                [encoder useResource:m_materialResourcesArgumentBuffer
                                               usage:MTLResourceUsageRead];
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
                [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:5];
                [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:6];
                [encoder setBuffer:BufferOrDummy(scene.materialBuffer(), m_dummyBuffer) offset:0 atIndex:8];
                [encoder setBuffer:BufferOrDummy(scene.meshInfoBuffer(), m_dummyBuffer) offset:0 atIndex:14];
                [encoder setBuffer:BufferOrDummy(scene.meshVertexBuffer(), m_dummyBuffer) offset:0 atIndex:15];
                [encoder setBuffer:BufferOrDummy(scene.meshIndexBuffer(), m_dummyBuffer) offset:0 atIndex:16];
                [encoder setBuffer:BufferOrDummy(scene.materialTextureInfoBuffer(), m_dummyBuffer)
                           offset:0
                          atIndex:17];
                dispatchWavefrontStage(encoder, shadeThreadsPerThreadgroup, true);
                [encoder endEncoding];
            }

            if (useIndirectShadeScalarProbe) {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::Shade,
                                @"Wavefront Shade Scalar Probe",
                                "Wavefront Shade");
                [encoder setComputePipelineState:pipelines.wavefrontShadeScalarProbe()];
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
                            [encoder setBuffer:m_materialResourcesArgumentBuffer offset:0 atIndex:23];
                            if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                                [encoder useResource:m_materialResourcesArgumentBuffer
                                               usage:MTLResourceUsageRead];
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
                [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:5];
                [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:6];
                [encoder setBuffer:BufferOrDummy(scene.materialBuffer(), m_dummyBuffer) offset:0 atIndex:8];
                [encoder setBuffer:BufferOrDummy(scene.meshInfoBuffer(), m_dummyBuffer) offset:0 atIndex:14];
                [encoder setBuffer:BufferOrDummy(scene.meshVertexBuffer(), m_dummyBuffer) offset:0 atIndex:15];
                [encoder setBuffer:BufferOrDummy(scene.meshIndexBuffer(), m_dummyBuffer) offset:0 atIndex:16];
                [encoder setBuffer:BufferOrDummy(scene.materialTextureInfoBuffer(), m_dummyBuffer)
                           offset:0
                          atIndex:17];
                dispatchWavefrontStage(encoder, shadeThreadsPerThreadgroup, true);
                [encoder endEncoding];
            }

            if (useIndirectShadeBsdfProbe) {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::Shade,
                                @"Wavefront Shade BSDF Probe",
                                "Wavefront Shade");
                [encoder setComputePipelineState:pipelines.wavefrontShadeBsdfProbe()];
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
                            [encoder setBuffer:m_materialResourcesArgumentBuffer offset:0 atIndex:23];
                            if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                                [encoder useResource:m_materialResourcesArgumentBuffer
                                               usage:MTLResourceUsageRead];
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
                [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:5];
                [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:6];
                [encoder setBuffer:BufferOrDummy(scene.materialBuffer(), m_dummyBuffer) offset:0 atIndex:8];
                [encoder setBuffer:BufferOrDummy(scene.meshInfoBuffer(), m_dummyBuffer) offset:0 atIndex:14];
                [encoder setBuffer:BufferOrDummy(scene.meshVertexBuffer(), m_dummyBuffer) offset:0 atIndex:15];
                [encoder setBuffer:BufferOrDummy(scene.meshIndexBuffer(), m_dummyBuffer) offset:0 atIndex:16];
                [encoder setBuffer:BufferOrDummy(scene.materialTextureInfoBuffer(), m_dummyBuffer)
                           offset:0
                          atIndex:17];
                dispatchWavefrontStage(encoder, shadeThreadsPerThreadgroup, true);
                [encoder endEncoding];
            }

            if (useIndirectShadeClosureProbe) {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::Shade,
                                @"Wavefront Shade Closure Probe",
                                "Wavefront Shade");
                [encoder setComputePipelineState:pipelines.wavefrontShadeClosureProbe()];
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
                            [encoder setBuffer:m_materialResourcesArgumentBuffer offset:0 atIndex:23];
                            if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                                [encoder useResource:m_materialResourcesArgumentBuffer
                                               usage:MTLResourceUsageRead];
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
                [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:5];
                [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:6];
                [encoder setBuffer:BufferOrDummy(scene.materialBuffer(), m_dummyBuffer) offset:0 atIndex:8];
                [encoder setBuffer:BufferOrDummy(scene.meshInfoBuffer(), m_dummyBuffer) offset:0 atIndex:14];
                [encoder setBuffer:BufferOrDummy(scene.meshVertexBuffer(), m_dummyBuffer) offset:0 atIndex:15];
                [encoder setBuffer:BufferOrDummy(scene.meshIndexBuffer(), m_dummyBuffer) offset:0 atIndex:16];
                [encoder setBuffer:BufferOrDummy(scene.materialTextureInfoBuffer(), m_dummyBuffer)
                           offset:0
                          atIndex:17];
                dispatchWavefrontStage(encoder, shadeThreadsPerThreadgroup, true);
                [encoder endEncoding];
            }

            if (useIndirectShade) {
                if (!encodeWavefrontDispatchArgs(
                        static_cast<uint32_t>(PathTracerShaderTypes::WavefrontQueueType::PrimaryRay),
                        shadeResolveSurfaceThreadsPerThreadgroup)) {
                    NSLog(@"[Wavefront] guarded shade indirect requested but resolver dispatch args unavailable");
                    return sample;
                }
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::Shade,
                                @"Wavefront Shade Resolve Surface",
                                "Wavefront Shade Resolve Surface");
                [encoder setComputePipelineState:pipelines.wavefrontShadeResolveSurface()];
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
                            [encoder setBuffer:m_materialResourcesArgumentBuffer offset:0 atIndex:23];
                            if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                                [encoder useResource:m_materialResourcesArgumentBuffer
                                               usage:MTLResourceUsageRead];
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
                [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:5];
                [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:6];
                [encoder setBuffer:BufferOrDummy(scene.materialBuffer(), m_dummyBuffer) offset:0 atIndex:8];
                [encoder setBuffer:BufferOrDummy(scene.meshInfoBuffer(), m_dummyBuffer) offset:0 atIndex:14];
                [encoder setBuffer:BufferOrDummy(scene.meshVertexBuffer(), m_dummyBuffer) offset:0 atIndex:15];
                [encoder setBuffer:BufferOrDummy(scene.meshIndexBuffer(), m_dummyBuffer) offset:0 atIndex:16];
                [encoder setBuffer:BufferOrDummy(scene.materialTextureInfoBuffer(), m_dummyBuffer)
                           offset:0
                          atIndex:17];
                [encoder setBuffer:m_wavefront.resolvedSurfaces offset:0 atIndex:30];
                if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                    [encoder useResource:m_wavefront.resolvedSurfaces usage:MTLResourceUsageWrite];
                }
                dispatchWavefrontStage(encoder, shadeResolveSurfaceThreadsPerThreadgroup, true);
                [encoder endEncoding];
            }

            {
                const bool useResolvedShade = useIndirectShade;
                if (useResolvedShade &&
                    !encodeWavefrontDispatchArgs(
                        static_cast<uint32_t>(PathTracerShaderTypes::WavefrontQueueType::PrimaryRay),
                        shadeResolvedThreadsPerThreadgroup)) {
                    NSLog(@"[Wavefront] guarded shade indirect requested but continuation dispatch args unavailable");
                    return sample;
                }
                id<MTLComputeCommandEncoder> encoder =
                makeEncoder(GpuTimingBucket::Shade,
                            useResolvedShade ? @"Wavefront Shade Resolved" : @"Wavefront Shade",
                            "Wavefront Shade");
                [encoder setComputePipelineState:(useResolvedShade
                                                      ? shadeResolvedPipeline
                                                      : shadePipeline)];
                if (canHardwareIntersect &&
                    provider.hardware.tlas &&
                    [encoder respondsToSelector:@selector(setAccelerationStructure:atBufferIndex:)]) {
                    [encoder setAccelerationStructure:provider.hardware.tlas atBufferIndex:18];
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
                            [argumentEncoder setBuffer:BufferOrDummy(m_wavefront.cacheQueryItems, m_dummyBuffer)
                                                offset:0
                                               atIndex:PathTracerShaderTypes::kMaterialResourceRadianceCacheQueryItemIndex];
                            [argumentEncoder setBuffer:BufferOrDummy(m_wavefront.cacheTrainRecords, m_dummyBuffer)
                                                offset:0
                                               atIndex:PathTracerShaderTypes::kMaterialResourceRadianceCacheTrainRecordIndex];
                            MTLBufferHandle causticTransport =
                                m_transportAllocator.bufferForName("CausticTransport.PathSpacePrototype");
                            const NSUInteger causticVertexCount =
                                causticTransport ? static_cast<NSUInteger>(m_restirState.causticMaxVertices) : 0u;
                            const NSUInteger causticLightOffset =
                                causticVertexCount * sizeof(PathTracerShaderTypes::EyeSubpathVertex);
                            const NSUInteger causticConnectionOffset =
                                causticLightOffset +
                                causticVertexCount * sizeof(PathTracerShaderTypes::LightSubpathVertex);
                            const NSUInteger causticResultOffset =
                                causticConnectionOffset +
                                causticVertexCount * sizeof(PathTracerShaderTypes::PathConnection);
                            [argumentEncoder setBuffer:BufferOrDummy(causticTransport, m_dummyBuffer)
                                                offset:0
                                               atIndex:PathTracerShaderTypes::kMaterialResourceCausticEyeVertexIndex];
                            [argumentEncoder setBuffer:BufferOrDummy(causticTransport, m_dummyBuffer)
                                                offset:(causticTransport ? causticLightOffset : 0u)
                                               atIndex:PathTracerShaderTypes::kMaterialResourceCausticLightVertexIndex];
                            [argumentEncoder setBuffer:BufferOrDummy(causticTransport, m_dummyBuffer)
                                                offset:(causticTransport ? causticConnectionOffset : 0u)
                                               atIndex:PathTracerShaderTypes::kMaterialResourceCausticConnectionIndex];
                            [argumentEncoder setBuffer:BufferOrDummy(causticTransport, m_dummyBuffer)
                                                offset:(causticTransport ? causticResultOffset : 0u)
                                               atIndex:PathTracerShaderTypes::kMaterialResourceCausticConnectionResultIndex];
                            [encoder setBuffer:m_materialResourcesArgumentBuffer offset:0 atIndex:23];
                            if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                                [encoder useResource:m_materialResourcesArgumentBuffer
                                               usage:MTLResourceUsageRead];
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
                                if (m_wavefront.cacheQueryItems) {
                                    [encoder useResource:m_wavefront.cacheQueryItems
                                                   usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                                }
                                if (m_wavefront.cacheTrainRecords) {
                                    [encoder useResource:m_wavefront.cacheTrainRecords
                                                   usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                                }
                                if (causticTransport) {
                                    [encoder useResource:causticTransport
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
                [encoder setBuffer:(useResolvedShade
                                        ? m_wavefront.resolvedSurfaces
                                        : BufferOrDummy(scene.materialTextureInfoBuffer(), m_dummyBuffer))
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
                if (useResolvedShade && [encoder respondsToSelector:@selector(useResource:usage:)]) {
                    [encoder useResource:m_wavefront.resolvedSurfaces usage:MTLResourceUsageRead];
                }
                dispatchWavefrontStage(encoder,
                                       useResolvedShade
                                           ? shadeResolvedThreadsPerThreadgroup
                                           : shadeThreadsPerThreadgroup,
                                       useResolvedShade);
                [encoder endEncoding];
            }

            bool useIndirectShadow = false;
            // Reuse the material resources already encoded for this shade pass.
            auto bindShadowAlphaResources = [&](id<MTLComputeCommandEncoder> encoder, bool hardware) {
                [encoder setBuffer:m_materialResourcesArgumentBuffer offset:0 atIndex:23];
                if (m_wavefront.cacheTrainRecords) {
                    // The shadow kernels add each training vertex's NEE to its record.
                    [encoder useResource:m_wavefront.cacheTrainRecords
                                   usage:MTLResourceUsageRead | MTLResourceUsageWrite];
                }
                [encoder setBuffer:BufferOrDummy(scene.materialTextureInfoBuffer(), m_dummyBuffer)
                           offset:0 atIndex:24];
                if (!hardware) {
                    [encoder setBuffer:BufferOrDummy(scene.meshInfoBuffer(), m_dummyBuffer) offset:0 atIndex:19];
                    [encoder setBuffer:BufferOrDummy(scene.meshVertexBuffer(), m_dummyBuffer) offset:0 atIndex:20];
                    [encoder setBuffer:BufferOrDummy(scene.meshIndexBuffer(), m_dummyBuffer) offset:0 atIndex:21];
                }
                if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
                    if (m_materialResourcesArgumentBuffer) {
                        [encoder useResource:m_materialResourcesArgumentBuffer usage:MTLResourceUsageRead];
                    }
                    if (!SkipTextureUseResources()) {
                        for (id<MTLTexture> texture : materialTextures) {
                            if (texture) [encoder useResource:texture usage:MTLResourceUsageRead];
                        }
                    }
                }
            };
            const MTLSize shadowThreadsPerThreadgroup =
                PreferredLinearThreadgroupSize(canHardwareShadow
                                                   ? (guardedShadowIndirect
                                                          ? pipelines.wavefrontShadowIndexedHardware()
                                                          : pipelines.wavefrontShadowHardware())
                                                   : (guardedShadowIndirect
                                                          ? pipelines.wavefrontShadowIndexed()
                                                          : pipelines.wavefrontShadow()));
            if (guardedShadowIndirect) {
                {
                    id<MTLComputeCommandEncoder> encoder =
                        makeEncoder(GpuTimingBucket::ShadowQueue,
                                    @"Wavefront Build Shadow Indices",
                                    "Wavefront Shadow");
                    [encoder setComputePipelineState:pipelines.wavefrontBuildShadowIndices()];
                    [encoder setBuffer:sampleUniformBuffer offset:0 atIndex:0];
                    [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:1];
                    [encoder setBuffer:m_wavefront.shadowIndices offset:0 atIndex:2];
                    [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:3];
                    const MTLSize threadsPerThreadgroup =
                        PreferredLinearThreadgroupSize(pipelines.wavefrontBuildShadowIndices());
                    [encoder dispatchThreads:fullGrid threadsPerThreadgroup:threadsPerThreadgroup];
                    [encoder endEncoding];
                }
                useIndirectShadow =
                    encodeWavefrontDispatchArgs(
                        static_cast<uint32_t>(PathTracerShaderTypes::WavefrontQueueType::ShadowRay),
                        shadowThreadsPerThreadgroup);
            }

            if (canHardwareShadow) {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::ShadowQueue,
                                @"Wavefront Shadow (HW)",
                                "Wavefront Shadow");
                [encoder setComputePipelineState:guardedShadowIndirect
                                                  ? pipelines.wavefrontShadowIndexedHardware()
                                                  : pipelines.wavefrontShadowHardware()];
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
                if (guardedShadowIndirect) {
                    [encoder setBuffer:m_wavefront.shadowIndices offset:0 atIndex:21];
                }
                bindShadowAlphaResources(encoder, true);
                dispatchWavefrontStage(encoder, shadowThreadsPerThreadgroup, useIndirectShadow);
                [encoder endEncoding];
            } else {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::ShadowQueue,
                                @"Wavefront Shadow",
                                "Wavefront Shadow");
                [encoder setComputePipelineState:guardedShadowIndirect
                                                  ? pipelines.wavefrontShadowIndexed()
                                                  : pipelines.wavefrontShadow()];
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
                if (guardedShadowIndirect) {
                    [encoder setBuffer:m_wavefront.shadowIndices offset:0 atIndex:16];
                }
                bindShadowAlphaResources(encoder, false);
                dispatchWavefrontStage(encoder, shadowThreadsPerThreadgroup, useIndirectShadow);
                [encoder endEncoding];
            }

            {
                id<MTLComputeCommandEncoder> encoder =
                    makeEncoder(GpuTimingBucket::ShadowQueue,
                                @"Wavefront Swap Queues",
                                "Wavefront Swap Queues");
                [encoder setComputePipelineState:pipelines.wavefrontSwapQueues()];
                [encoder setBuffer:m_wavefront.queueCounters offset:0 atIndex:0];
                [encoder setBuffer:BufferOrDummy(m_statsBuffer, m_dummyBuffer) offset:0 atIndex:1];
                [encoder setBuffer:sampleUniformBuffer offset:0 atIndex:2];
                [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:3];
                [encoder dispatchThreads:resetThreads threadsPerThreadgroup:resetThreads];
                [encoder endEncoding];
            }

            std::swap(currentQueueIndex, nextQueueIndex);
        }

        if (m_wavefront.cacheTrainRecords && m_restirState.radianceCache) {
            // Every shadow ray has resolved: train the radiance cache from this sample's training paths.
            id<MTLComputeCommandEncoder> encoder =
                makeEncoder(GpuTimingBucket::ShadowQueue,
                            @"Wavefront Radiance Cache Train",
                            "Wavefront Radiance Cache Train");
            [encoder setComputePipelineState:pipelines.wavefrontRadianceCacheTrain()];
            [encoder setBuffer:sampleUniformBuffer offset:0 atIndex:0];
            [encoder setBuffer:m_wavefront.pathStates offset:0 atIndex:1];
            [encoder setBuffer:m_restirState.radianceCache offset:0 atIndex:2];
            [encoder setBuffer:m_wavefront.cacheTrainRecords offset:0 atIndex:3];
            [encoder setBuffer:BufferOrDummy(m_statsBuffer, m_dummyBuffer) offset:0 atIndex:4];
            const NSUInteger groups =
                (static_cast<NSUInteger>(m_wavefront.capacity) + PathTracerShaderTypes::kRadianceCacheTrainingPathStride - 1u) /
                PathTracerShaderTypes::kRadianceCacheTrainingPathStride;
            const MTLSize threadsPerThreadgroup =
                PreferredLinearThreadgroupSize(pipelines.wavefrontRadianceCacheTrain());
            [encoder dispatchThreads:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:threadsPerThreadgroup];
            [encoder endEncoding];
        }

        {
            id<MTLComputeCommandEncoder> encoder =
                makeEncoder(GpuTimingBucket::ShadowQueue,
                            @"Wavefront Accumulate",
                            "Wavefront Accumulate");
            [encoder setComputePipelineState:pipelines.wavefrontAccumulate()];
            [encoder setTexture:accumulation.radianceSum() atIndex:0];
            [encoder setTexture:accumulation.sampleCountTexture() atIndex:1];
            [encoder setTexture:accumulation.albedoBuffer() atIndex:2];
            [encoder setTexture:accumulation.normalBuffer() atIndex:3];
            [encoder setTexture:accumulation.positionBuffer() atIndex:4];
            [encoder setTexture:accumulation.materialFeatureBuffer() atIndex:5];
            [encoder setTexture:accumulation.motionVectorBuffer() atIndex:6];
            [encoder setTexture:accumulation.directRadianceSum() atIndex:7];
            [encoder setTexture:accumulation.indirectRadianceSum() atIndex:8];
            [encoder setTexture:accumulation.radianceMomentSum() atIndex:9];
            [encoder setTexture:accumulation.reconstructionMetricBuffer() atIndex:10];
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

    // The wavefront rendered this frame, so a fallback that comes back later is logged again.
    m_loggedWavefrontMegakernelFallback = nullptr;
    m_loggedWavefrontExplicitRestirDiRanFused = false;
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
    if (!settings.svgfDenoiseEnabled || !pipelines.svgfTemporal() || !pipelines.svgfAtrous() ||
        !pipelines.svgfCompositeSplit()) {
        return;
    }
    MTLTextureHandle present = accumulation.present();
    MTLTextureHandle directPresent = accumulation.directPresent();
    MTLTextureHandle indirectPresent = accumulation.indirectPresent();
    MTLTextureHandle albedo = accumulation.albedoBuffer();
    MTLTextureHandle normal = accumulation.normalBuffer();
    MTLTextureHandle position = accumulation.positionBuffer();
    MTLTextureHandle materialFeatures = accumulation.materialFeatureBuffer();
    MTLTextureHandle motionVectors = accumulation.motionVectorBuffer();
    MTLTextureHandle reconstructionMetrics = accumulation.reconstructionMetricBuffer();
    MTLTextureHandle denoised = accumulation.denoisedBuffer();
    MTLTextureHandle scratch = accumulation.svgfScratchBuffer();
    MTLTextureHandle momentsA = accumulation.svgfMomentsBufferA();
    MTLTextureHandle momentsB = accumulation.svgfMomentsBufferB();
    MTLTextureHandle indirectMomentsA = accumulation.svgfIndirectMomentsBufferA();
    MTLTextureHandle indirectMomentsB = accumulation.svgfIndirectMomentsBufferB();
    MTLTextureHandle guideA = accumulation.svgfGuideBufferA();
    MTLTextureHandle guideB = accumulation.svgfGuideBufferB();
    MTLTextureHandle colorA = accumulation.svgfColorBufferA();
    MTLTextureHandle colorB = accumulation.svgfColorBufferB();
    MTLTextureHandle indirectColorA = accumulation.svgfIndirectColorBufferA();
    MTLTextureHandle indirectColorB = accumulation.svgfIndirectColorBufferB();
    MTLTextureHandle directComponent = accumulation.svgfDirectComponentBuffer();
    MTLTextureHandle indirectComponent = accumulation.svgfIndirectComponentBuffer();
    MTLTextureHandle materialHistoryA = accumulation.svgfMaterialHistoryBufferA();
    MTLTextureHandle materialHistoryB = accumulation.svgfMaterialHistoryBufferB();
    if (!commandBuffer || !present || !directPresent || !indirectPresent ||
        !albedo || !normal || !position ||
        !materialFeatures || !motionVectors || !reconstructionMetrics || !denoised || !scratch ||
        !momentsA || !momentsB || !indirectMomentsA || !indirectMomentsB ||
        !guideA || !guideB || !colorA || !colorB || !indirectColorA || !indirectColorB ||
        !directComponent || !indirectComponent ||
        !materialHistoryA || !materialHistoryB ||
        !m_hasCurrentPathtraceUniforms) {
        m_hasDenoisedOutput = false;
        return;
    }

    auto matchesPresentExtent = ^BOOL(MTLTextureHandle texture) {
        return texture && texture.width == present.width && texture.height == present.height;
    };
    if (!matchesPresentExtent(directPresent) || !matchesPresentExtent(indirectPresent) ||
        !matchesPresentExtent(albedo) || !matchesPresentExtent(normal) ||
        !matchesPresentExtent(position) || !matchesPresentExtent(materialFeatures) ||
        !matchesPresentExtent(motionVectors) || !matchesPresentExtent(reconstructionMetrics) ||
        !matchesPresentExtent(denoised) ||
        !matchesPresentExtent(scratch) || !matchesPresentExtent(momentsA) ||
        !matchesPresentExtent(momentsB) || !matchesPresentExtent(indirectMomentsA) ||
        !matchesPresentExtent(indirectMomentsB) || !matchesPresentExtent(guideA) ||
        !matchesPresentExtent(guideB) || !matchesPresentExtent(colorA) ||
        !matchesPresentExtent(colorB) || !matchesPresentExtent(indirectColorA) ||
        !matchesPresentExtent(indirectColorB) || !matchesPresentExtent(directComponent) ||
        !matchesPresentExtent(indirectComponent) || !matchesPresentExtent(materialHistoryA) ||
        !matchesPresentExtent(materialHistoryB)) {
        static bool sLoggedSvgfExtentMismatch = false;
        if (!sLoggedSvgfExtentMismatch) {
            sLoggedSvgfExtentMismatch = true;
            NSLog(@"[SVGF] Required textures do not match present extent %zux%zu; "
                  "skipping denoised output until accumulation resources are rebuilt.",
                  static_cast<size_t>(present.width),
                  static_cast<size_t>(present.height));
        }
        m_hasDenoisedOutput = false;
        return;
    }

    const uint32_t width = static_cast<uint32_t>(present.width);
    const uint32_t height = static_cast<uint32_t>(present.height);
    if (width == 0u || height == 0u) {
        m_hasDenoisedOutput = false;
        return;
    }

    MTLTextureHandle previousGuide = (m_frameCounter & 1u) ? guideB : guideA;
    MTLTextureHandle currentGuide = (m_frameCounter & 1u) ? guideA : guideB;
    MTLTextureHandle previousMaterialHistory = (m_frameCounter & 1u) ? materialHistoryB : materialHistoryA;
    MTLTextureHandle currentMaterialHistory = (m_frameCounter & 1u) ? materialHistoryA : materialHistoryB;
    const PathtraceUniforms& previousCamera =
        m_hasPreviousSvgfCameraUniforms ? m_previousSvgfCameraUniforms : m_currentPathtraceUniforms;

    auto makeSvgfUniforms = [&](uint32_t step) {
        SvgfDenoiseUniforms uniforms{};
        uniforms.width = width;
        uniforms.height = height;
        uniforms.step = step;
        uniforms.reprojectionEnabled = m_hasPreviousSvgfCameraUniforms ? 1u : 0u;
        uniforms.normalSigma = std::max(settings.svgfNormalSigma, 1.0e-4f);
        uniforms.albedoSigma = std::max(settings.svgfAlbedoSigma, 1.0e-4f);
        uniforms.luminanceSigma = std::max(settings.svgfLuminanceSigma, 1.0e-4f);
        uniforms.historyValid = m_hasPreviousSvgfCameraUniforms ? 1u : 0u;
        uniforms.denoiseDirect = settings.svgfDenoiseDirect ? 1u : 0u;
        uniforms.denoiseIndirect = settings.svgfDenoiseIndirect ? 1u : 0u;
        uniforms.currentCameraOrigin = simd_make_float4(m_currentPathtraceUniforms.cameraOrigin, 0.0f);
        uniforms.previousCameraOrigin = simd_make_float4(previousCamera.cameraOrigin, 0.0f);
        uniforms.previousLowerLeftCorner = simd_make_float4(previousCamera.lowerLeftCorner, 0.0f);
        uniforms.previousHorizontal = simd_make_float4(previousCamera.horizontal, 0.0f);
        uniforms.previousVertical = simd_make_float4(previousCamera.vertical, 0.0f);
        return uniforms;
    };

    auto encodeSvgfComponent = [&](NSString* label,
                                   MTLTextureHandle sourceTexture,
                                   MTLTextureHandle componentOutput,
                                   MTLTextureHandle momentsTextureA,
                                   MTLTextureHandle momentsTextureB,
                                   MTLTextureHandle colorTextureA,
                                   MTLTextureHandle colorTextureB,
                                   bool enabled) -> bool {
        if (!enabled) {
            return true;
        }
        MTLTextureHandle previousMoments = (m_frameCounter & 1u) ? momentsTextureB : momentsTextureA;
        MTLTextureHandle currentMoments = (m_frameCounter & 1u) ? momentsTextureA : momentsTextureB;
        MTLTextureHandle previousColor = (m_frameCounter & 1u) ? colorTextureB : colorTextureA;
        MTLTextureHandle currentColor = (m_frameCounter & 1u) ? colorTextureA : colorTextureB;

        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            return false;
        }
        encoder.label = [NSString stringWithFormat:@"SVGF Temporal Moments %@", label];
        [encoder setComputePipelineState:pipelines.svgfTemporal()];
        SvgfDenoiseUniforms uniforms = makeSvgfUniforms(1u);
        [encoder setBytes:&uniforms length:sizeof(SvgfDenoiseUniforms) atIndex:0];
        [encoder setTexture:sourceTexture atIndex:0];
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
        [encoder setTexture:previousMaterialHistory atIndex:11];
        [encoder setTexture:currentMaterialHistory atIndex:12];
        [encoder setTexture:reconstructionMetrics atIndex:13];

        const MTLSize threadsPerThreadgroup =
            PreferredLinearThreadgroupSize(pipelines.svgfTemporal());
        MTLSize threadsPerGrid = MTLSizeMake(width * height, 1u, 1u);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];

        const uint32_t iterations = std::clamp(settings.svgfAtrousIterations, 1u, 5u);
        MTLTextureHandle source = currentColor;

        for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
            const bool finalIteration = (iteration + 1u == iterations);
            MTLTextureHandle target = finalIteration
                ? componentOutput
                : (((iterations - iteration) % 2u == 0u) ? scratch : componentOutput);
            if (target == source) {
                return false;
            }

            id<MTLComputeCommandEncoder> atrousEncoder = [commandBuffer computeCommandEncoder];
            if (!atrousEncoder) {
                return false;
            }
            atrousEncoder.label = [NSString stringWithFormat:@"SVGF A-Trous %@ %u", label, iteration];
            [atrousEncoder setComputePipelineState:pipelines.svgfAtrous()];
            SvgfDenoiseUniforms atrousUniforms = makeSvgfUniforms(1u << iteration);
            [atrousEncoder setBytes:&atrousUniforms length:sizeof(SvgfDenoiseUniforms) atIndex:0];
            [atrousEncoder setTexture:source atIndex:0];
            [atrousEncoder setTexture:albedo atIndex:1];
            [atrousEncoder setTexture:normal atIndex:2];
            [atrousEncoder setTexture:target atIndex:3];
            [atrousEncoder setTexture:currentMoments atIndex:4];
            [atrousEncoder setTexture:position atIndex:5];
            [atrousEncoder setTexture:materialFeatures atIndex:6];

            const MTLSize atrousThreadsPerThreadgroup =
                PreferredLinearThreadgroupSize(pipelines.svgfAtrous());
            [atrousEncoder dispatchThreads:threadsPerGrid
                     threadsPerThreadgroup:atrousThreadsPerThreadgroup];
            [atrousEncoder endEncoding];

            source = target;
        }
        return true;
    };

    if (!encodeSvgfComponent(@"Direct",
                             directPresent,
                             directComponent,
                             momentsA,
                             momentsB,
                             colorA,
                             colorB,
                             settings.svgfDenoiseDirect) ||
        !encodeSvgfComponent(@"Indirect",
                             indirectPresent,
                             indirectComponent,
                             indirectMomentsA,
                             indirectMomentsB,
                             indirectColorA,
                             indirectColorB,
                             settings.svgfDenoiseIndirect)) {
        m_hasDenoisedOutput = false;
        return;
    }

    {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (!encoder) {
            m_hasDenoisedOutput = false;
            return;
        }
        encoder.label = @"SVGF Direct/Indirect Composite";
        [encoder setComputePipelineState:pipelines.svgfCompositeSplit()];
        SvgfDenoiseUniforms uniforms = makeSvgfUniforms(1u);
        [encoder setBytes:&uniforms length:sizeof(SvgfDenoiseUniforms) atIndex:0];
        [encoder setTexture:directPresent atIndex:0];
        [encoder setTexture:indirectPresent atIndex:1];
        [encoder setTexture:directComponent atIndex:2];
        [encoder setTexture:indirectComponent atIndex:3];
        [encoder setTexture:denoised atIndex:4];
        const MTLSize threadsPerThreadgroup =
            PreferredLinearThreadgroupSize(pipelines.svgfCompositeSplit());
        MTLSize threadsPerGrid = MTLSizeMake(width * height, 1u, 1u);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
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
        m_gpuTiming.ranges.push_back(
            {GpuTimingBucket::Presentation, "Path Tracer Present", startSample, endSample});
        encoder = [commandBuffer computeCommandEncoderWithDescriptor:passDescriptor];
    } else {
        encoder = [commandBuffer computeCommandEncoder];
    }
    encoder.label = @"Path Tracer Present";
    [encoder setComputePipelineState:pipelines.present()];
    [encoder setTexture:accumulation.radianceSum() atIndex:0];
    [encoder setTexture:accumulation.sampleCountTexture() atIndex:1];
    [encoder setTexture:accumulation.present() atIndex:2];
    [encoder setTexture:accumulation.directRadianceSum() atIndex:3];
    [encoder setTexture:accumulation.indirectRadianceSum() atIndex:4];
    [encoder setTexture:accumulation.radianceMomentSum() atIndex:5];
    [encoder setTexture:accumulation.directPresent() atIndex:6];
    [encoder setTexture:accumulation.indirectPresent() atIndex:7];
    [encoder setTexture:accumulation.reconstructionMetricBuffer() atIndex:8];

    NSUInteger threadWidth = pipelines.present().threadExecutionWidth;
    NSUInteger threadHeight = std::max<NSUInteger>(
        1, pipelines.present().maxTotalThreadsPerThreadgroup / std::max<NSUInteger>(threadWidth, 1));
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadWidth, threadHeight, 1);
    MTLSize threadsPerGrid = MTLSizeMake(accumulation.present().width,
                                        accumulation.present().height, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
}

bool RenderLoop::encodeMlReconstructionTiny(MTLCommandBufferHandle commandBuffer,
                                            Accumulation& accumulation,
                                            const Pipelines& pipelines,
                                            const RenderSettings& settings) {
    m_mlState.activeThisFrame = false;
    m_mlState.dispatchCount = 0u;
    m_mlState.lastConfidence = 0.0f;

    if (settings.mlBackendMode != RenderSettings::MlBackendMode::ReconstructionTiny) {
        return true;
    }
    if (!pipelines.mlReconstructionTiny()) {
        m_mlState.modelValid = false;
        m_mlState.fallbackReason = "ML reconstruction pipeline unavailable";
        return true;
    }

    MTLTextureHandle source = accumulation.present();
    MTLTextureHandle albedo = accumulation.albedoBuffer();
    MTLTextureHandle normal = accumulation.normalBuffer();
    MTLTextureHandle position = accumulation.positionBuffer();
    MTLTextureHandle motion = accumulation.motionVectorBuffer();
    MTLTextureHandle metrics = accumulation.reconstructionMetricBuffer();
    MTLTextureHandle output = accumulation.denoisedBuffer();
    if (!source || !albedo || !normal || !position || !motion || !metrics || !output) {
        m_mlState.modelValid = false;
        m_mlState.fallbackReason = "ML reconstruction tensor textures unavailable";
        return true;
    }

    const uint32_t width = static_cast<uint32_t>(source.width);
    const uint32_t height = static_cast<uint32_t>(source.height);
    if (!ensureMlBackendResources(width, height, settings, commandBuffer) ||
        !m_mlState.modelValid || !m_mlState.modelBuffer) {
        return true;
    }

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"ML Reconstruction Tiny";
    [encoder setComputePipelineState:pipelines.mlReconstructionTiny()];
    [encoder setBuffer:m_mlState.modelBuffer offset:0 atIndex:0];
    [encoder setTexture:source atIndex:0];
    [encoder setTexture:albedo atIndex:1];
    [encoder setTexture:normal atIndex:2];
    [encoder setTexture:position atIndex:3];
    [encoder setTexture:motion atIndex:4];
    [encoder setTexture:metrics atIndex:5];
    [encoder setTexture:output atIndex:6];
    MTLSize threadsPerThreadgroup = PreferredLinearThreadgroupSize(pipelines.mlReconstructionTiny());
    MTLSize threadsPerGrid = MTLSizeMake(source.width, source.height, 1u);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    m_mlState.activeThisFrame = true;
    m_mlState.dispatchCount = 1u;
    m_mlState.lastConfidence = 1.0f;
    m_mlState.biasRisk = 0.05f;
    m_mlState.fallbackReason.clear();
    m_hasDenoisedOutput = true;
    return true;
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
    if ((settings.denoiseEnabled || settings.svgfDenoiseEnabled ||
         settings.mlBackendMode == RenderSettings::MlBackendMode::ReconstructionTiny) &&
        m_hasDenoisedOutput) {
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
    m_wavefrontMegakernelFallback = "none";
    m_wavefrontMegakernelFallbackThisFrame = false;
    m_wavefrontExplicitRestirDiRanFused = false;
    bool hardwareDebugSoftwareFallbackEnabled = false;
#if PT_DEBUG_TOOLS
    hardwareDebugSoftwareFallbackEnabled =
        settings.enablePathDebug || settings.debugDirectLightAudit || settings.parityAssertEnabled;
#endif
    scene.setHardwareDebugSoftwareFallbackEnabled(hardwareDebugSoftwareFallbackEnabled);
    const bool willClearAccumulation = accumulation.needsClear() && pipelines.clear();
    const bool willRebuildScene = scene.isDirty();
    const bool hasDisplayPass = (renderPassDescriptor && drawable && pipelines.display());
    const bool restirDiRequested = ExplicitRestirDiEnabled(settings);
    const bool fusedRestirDiReference =
        context.headless() && restirDiRequested &&
        (!settings.restirDebugEnabled || HeadlessFusedRestirDiReferenceEnabled());
    const bool explicitRestirDiRequested =
        restirDiRequested && !fusedRestirDiReference &&
        (context.headless() || InteractiveExplicitRestirDiEnabled());
    // On the megakernel, restir_di's explicit passes would run as screen passes, which read no material textures,
    // normal maps or occlusion and give wrong light (0.0625 of the exact value on an analytic case). They are refused:
    // headless runs stop before rendering (main_headless.mm), and the window logs it once and keeps the fused
    // restir_di. The hybrid's explicit stage on the megakernel and the wavefront's explicit stage are exact.
    const bool megakernelExplicitRestirDiRefused =
        explicitRestirDiRequested &&
        settings.executionMode == RenderSettings::ExecutionMode::Megakernel &&
        settings.directLightMode == RenderSettings::DirectLightMode::RestirDi;
    if (megakernelExplicitRestirDiRefused && !m_loggedMegakernelExplicitRestirDiRefusal) {
        m_loggedMegakernelExplicitRestirDiRefusal = true;
        NSLog(@"[ReSTIR DI] Explicit passes refused on the megakernel (its screen passes ignore material textures, "
              @"normal maps and occlusion); rendering the fused restir_di. The wavefront runs the explicit passes.");
    }
    const bool restirDiEnabled = explicitRestirDiRequested && !megakernelExplicitRestirDiRefused;
    const bool restirDiWavefrontStageEnabled =
        restirDiEnabled &&
        settings.executionMode == RenderSettings::ExecutionMode::Wavefront;
    const bool restirDiMegakernelIntegratedStageEnabled =
        restirDiEnabled &&
        settings.executionMode == RenderSettings::ExecutionMode::Megakernel &&
        settings.directLightMode == RenderSettings::DirectLightMode::RestirDiRegirHybrid;
    const bool restirDiScreenPassesEnabled =
        restirDiEnabled &&
        !restirDiWavefrontStageEnabled &&
        !restirDiMegakernelIntegratedStageEnabled;
    m_explicitRestirDiActiveThisFrame = restirDiEnabled;
    m_restirDiWavefrontStageActiveThisFrame = restirDiWavefrontStageEnabled;
    m_restirDiMegakernelIntegratedStageActiveThisFrame =
        restirDiMegakernelIntegratedStageEnabled;
    m_restirDiHardwareTraceActiveThisFrame = false;
    // Homogeneous volumes run restir_di and the hybrid as baseline_emissive (UniformBuilder); say so when that starts.
    const bool restirDiDemotedByVolume = RestirDiDemotedByVolume(settings);
    if (restirDiDemotedByVolume && !m_restirDiDemotedByVolume) {
        NSLog(@"[ReSTIR DI] directLightMode=%s does not run with volumeTransportMode=homogeneous (its reservoirs "
              @"carry no medium state); rendering baseline_emissive instead.",
              settings.directLightMode == RenderSettings::DirectLightMode::RestirDi ? "restir_di"
                                                                                     : "restir_di_regir_hybrid");
    }
    m_restirDiDemotedByVolume = restirDiDemotedByVolume;
    if (restirDiDemotedByVolume) {
        m_restirDiPathStatus = "volume_baseline_emissive";
    } else if (!restirDiRequested) {
        m_restirDiPathStatus = "off";
    } else if (restirDiEnabled) {
        m_restirDiPathStatus = restirDiWavefrontStageEnabled
            ? "explicit_wavefront"
            : (restirDiMegakernelIntegratedStageEnabled ? "explicit_megakernel_integrated"
                                                        : "explicit_graph");
    } else if (fusedRestirDiReference) {
        m_restirDiPathStatus = "fused_reference";
    } else if (megakernelExplicitRestirDiRefused) {
        m_restirDiPathStatus = "fused_megakernel_explicit_refused";
    } else {
        m_restirDiPathStatus = "fused_default";
    }
    if (settings.executionMode != RenderSettings::ExecutionMode::Wavefront &&
        m_wavefront.totalBytes > 0u) {
        m_wavefront = {};
        m_transportAllocator.releaseOwner(TransportMemoryOwner::Wavefront);
    }
    static bool sLoggedInteractiveExplicitRestirFallback = false;
    if (restirDiRequested && !restirDiEnabled && !megakernelExplicitRestirDiRefused &&
        !sLoggedInteractiveExplicitRestirFallback) {
        sLoggedInteractiveExplicitRestirFallback = true;
        if (fusedRestirDiReference) {
            NSLog(@"[ReSTIR DI] headless validation is rendering the fused/reference shader path "
                  "because ReSTIR debug is disabled or PT_RESTIR_DI_VALIDATE_FUSED_REFERENCE is set.");
        } else {
            NSLog(@"[ReSTIR DI] interactive explicit DI passes disabled for responsiveness; "
                  "set PT_ENABLE_INTERACTIVE_EXPLICIT_RESTIR_DI=1 to profile the explicit pass graph in the GUI.");
        }
    }
    m_lastPassGraph =
        BuildRenderPassGraph(settings,
                             willClearAccumulation,
                             willRebuildScene,
                             hasDisplayPass,
                             restirDiScreenPassesEnabled);
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

    // Clear the per-frame stats. The radiance cache counters at the end of PathtraceStats accumulate over
    // the render and are cleared in resetHistory().
    if (m_statsBuffer) {
        memset([m_statsBuffer contents], 0, offsetof(PathtraceStats, radianceCacheStats));
    }
#if PT_DEBUG_TOOLS
    if (m_debugBuffer) {
        auto* debugData =
            reinterpret_cast<PathtraceDebugBuffer*>([m_debugBuffer contents]);
        if (debugData) {
            debugData->writeIndex = 0;
            uint32_t allowed = 0;
            if ((settings.enablePathDebug || settings.debugDirectLightAudit) &&
                settings.debugMaxEntries > 0) {
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
                case RenderPassExecutionId::MlBackendNoOp: {
                    const CGSize currentSize = accumulation.currentSize();
                    const uint32_t width =
                        static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.width));
                    const uint32_t height =
                        static_cast<uint32_t>(std::max<CGFloat>(1.0, currentSize.height));
                    passOk = ensureMlBackendResources(width, height, settings, commandBuffer);
                    rebuildTransportMemoryRegistry(settings);
                    break;
                }
                case RenderPassExecutionId::MlBackendReconstructionTiny:
                    passOk = encodeMlReconstructionTiny(commandBuffer, accumulation, pipelines, settings);
                    rebuildTransportMemoryRegistry(settings);
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
    // A frame the wavefront handed to the megakernel counts as megakernel dispatches.
    if (settings.executionMode == RenderSettings::ExecutionMode::Wavefront && !m_wavefrontMegakernelFallbackThisFrame) {
        const uint32_t depthCount = std::max<uint32_t>(1u, settings.maxDepth);
        const bool shadeIndirectActive =
            settings.wavefrontShadeIndirectEnabled || WavefrontShadeIndirectEnabled();
        const uint32_t buildArgsPerDepth =
            1u +
            (shadeIndirectActive ? 2u : 0u) +
            ((settings.wavefrontShadeIndirectProbeEnabled || WavefrontShadeIndirectProbeEnabled()) ? 1u : 0u) +
            ((settings.wavefrontShadeSurfaceProbeEnabled || WavefrontShadeSurfaceProbeEnabled()) ? 1u : 0u) +
            ((settings.wavefrontShadeMetadataProbeEnabled || WavefrontShadeMetadataProbeEnabled()) ? 1u : 0u) +
            ((settings.wavefrontShadeScalarProbeEnabled || WavefrontShadeScalarProbeEnabled()) ? 1u : 0u) +
            ((settings.wavefrontShadeClosureProbeEnabled || WavefrontShadeClosureProbeEnabled()) ? 1u : 0u) +
            ((settings.wavefrontShadeBsdfProbeEnabled || WavefrontShadeBsdfProbeEnabled()) ? 1u : 0u) +
            (((settings.wavefrontShadowIndirectEnabled || WavefrontShadowIndirectEnabled()) &&
        !(settings.referencePointExactNee && scene.zeroRadiusReferencePointLightCount() > 0u)) ? 1u : 0u);
        result.wavefrontResetDispatchCount = samplesDispatched;
        result.wavefrontGenerateDispatchCount = samplesDispatched;
        result.wavefrontBuildDispatchArgsDispatchCount = samplesDispatched * depthCount * buildArgsPerDepth;
        result.wavefrontCompactQueueDispatchCount =
            WavefrontCompactionEnabled(settings) ? (samplesDispatched * depthCount) : 0u;
        result.wavefrontPrepareDispatchCount = samplesDispatched * depthCount;
        result.wavefrontIntersectDispatchCount = samplesDispatched * depthCount;
        result.wavefrontShadeDispatchCount =
            samplesDispatched * depthCount *
            ((shadeIndirectActive ? 2u : 1u) +
             (restirDiWavefrontStageEnabled ? 1u : 0u) +
             ((settings.wavefrontShadeIndirectProbeEnabled || WavefrontShadeIndirectProbeEnabled()) ? 1u : 0u) +
             ((settings.wavefrontShadeSurfaceProbeEnabled || WavefrontShadeSurfaceProbeEnabled()) ? 1u : 0u) +
             ((settings.wavefrontShadeMetadataProbeEnabled || WavefrontShadeMetadataProbeEnabled()) ? 1u : 0u) +
             ((settings.wavefrontShadeScalarProbeEnabled || WavefrontShadeScalarProbeEnabled()) ? 1u : 0u) +
             ((settings.wavefrontShadeClosureProbeEnabled || WavefrontShadeClosureProbeEnabled()) ? 1u : 0u) +
             ((settings.wavefrontShadeBsdfProbeEnabled || WavefrontShadeBsdfProbeEnabled()) ? 1u : 0u));
        result.wavefrontShadowDispatchCount =
            samplesDispatched * depthCount *
            (1u + (((settings.wavefrontShadowIndirectEnabled || WavefrontShadowIndirectEnabled()) &&
        !(settings.referencePointExactNee && scene.zeroRadiusReferencePointLightCount() > 0u)) ? 1u : 0u));
        result.wavefrontSwapDispatchCount = samplesDispatched * depthCount;
        result.wavefrontAccumulateDispatchCount = samplesDispatched;
    } else {
        result.integrationDispatchCount = samplesDispatched;
    }
    const uint64_t svgfComponentCount =
        settings.svgfDenoiseEnabled
            ? static_cast<uint64_t>((settings.svgfDenoiseDirect ? 1u : 0u) +
                                    (settings.svgfDenoiseIndirect ? 1u : 0u))
            : 0u;
    const uint64_t svgfDispatchCount =
        settings.svgfDenoiseEnabled
            ? (svgfComponentCount * (static_cast<uint64_t>(std::clamp(settings.svgfAtrousIterations, 1u, 5u)) + 1u) + 1u)
            : 0u;
    // Per-frame counts: the 64-bit SVGF and ML terms are small (the ML term is 0 or 1).
    result.totalComputeDispatchCount = static_cast<uint32_t>(
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
        svgfDispatchCount +
        m_mlState.dispatchCount);
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
    m_lastPassGraphExecution.passTimingAvailable = false;
    m_lastPassGraphExecution.passTimingNames.clear();
    m_lastPassGraphExecution.passTimingMs.clear();
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
    std::vector<std::pair<std::string, uint64_t>> passTicks;
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
        if (!range.passName.empty()) {
            auto passIt = std::find_if(
                passTicks.begin(),
                passTicks.end(),
                [&range](const std::pair<std::string, uint64_t>& candidate) {
                    return candidate.first == range.passName;
                });
            if (passIt == passTicks.end()) {
                passTicks.emplace_back(range.passName, deltaTicks);
            } else {
                passIt->second += deltaTicks;
            }
        }
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
    m_lastPassGraphExecution.passTimingAvailable = !passTicks.empty();
    m_lastPassGraphExecution.passTimingNames.reserve(passTicks.size());
    m_lastPassGraphExecution.passTimingMs.reserve(passTicks.size());
    for (const auto& passTick : passTicks) {
        m_lastPassGraphExecution.passTimingNames.push_back(passTick.first);
        m_lastPassGraphExecution.passTimingMs.push_back(static_cast<double>(passTick.second) * scale);
    }
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
    m_wavefront.cacheTrainRecords = nullptr;
    m_wavefront.cacheQueryItems = nullptr;
    m_wavefront.pathStates = nullptr;
    m_wavefront.shadowRays = nullptr;
    m_wavefront.shadowIndices = nullptr;
    m_wavefront.hitRecords = nullptr;
    m_wavefront.activeRays[1] = nullptr;
    m_wavefront.activeRays[0] = nullptr;
    m_frameGraphOwnedBuffers.clear();
    m_restirState.restirPt = nullptr;
    m_restirState.pathGuiding = nullptr;
    m_restirState = RestirStateResources{};
    m_transportAllocator.clear();
    m_transportMemory.clear();
    m_statsBuffer = nullptr;
    m_debugBuffer = nullptr;
    m_displayUniformBuffer = nullptr;
    m_uniformBuffer = nullptr;
    m_device = nullptr;
}

}  // namespace PathTracer
