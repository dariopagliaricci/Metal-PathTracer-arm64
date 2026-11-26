#pragma once

#include "renderer/MetalHandles.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace PathTracer {

// Forward declaration
class MetalContext;

/// The directory this process last read the Metal shader sources from (logged once as "[Shaders] Reading the shader
/// sources from ..."), or empty before the first shader load.
std::string ResolvedShaderDirectory();

/// Manages all Metal pipeline states and shader compilation
/// Handles shader source loading, compilation, and pipeline state object creation
class Pipelines {
public:
    struct CompilationRecord {
        std::string key;
        std::string functionName;
        std::string stage;
        double compileMs = 0.0;
        bool success = false;
        bool rayTracingSupportFlag = false;
        std::string specialization;
    };

    struct CompilationReport {
        bool available = false;
        bool runtimeSourceCompilation = true;
        bool functionConstantsUsed = false;
        bool metalBinaryArchiveUsed = false;
        bool metalBinaryArchiveSerialized = false;
        bool harvestedCompilationReady = false;
        uint64_t shaderSourceBytes = 0;
        double libraryCompileMs = 0.0;
        uint32_t computePipelineCount = 0;
        uint32_t renderPipelineCount = 0;
        uint32_t failedPipelineCount = 0;
        std::string cacheStrategy = "runtime_source_library_no_binary_archive";
        std::string specializationStrategy =
            "separate pipeline keys for execution/backend features; function constants not yet used";
        std::string binaryArchivePath;
        std::vector<CompilationRecord> records;
    };

    Pipelines() = default;
    ~Pipelines() = default;
    
    // Non-copyable
    Pipelines(const Pipelines&) = delete;
    Pipelines& operator=(const Pipelines&) = delete;
    
    /// Initialize pipelines with Metal context and display format
    /// @param context Metal context providing device
    /// @param displayFormat Pixel format for final display render pipeline
    /// @return true if all pipelines compiled successfully
    bool initialize(const MetalContext& context, MTLPixelFormat displayFormat);

    /// Enable verbose timing logs (headless/diagnostic)
    void setVerboseTiming(bool enabled) { m_verboseTiming = enabled; }

    /// Compile the shaders with PT_SOURCE_FEATURES=1 (Tungsten-source closures, textures and transport). Call before
    /// initialize(); the default set compiles none of that code, so it is the code main compiles.
    void setSourceFeatures(bool enabled) { m_sourceFeatures = enabled; }
    
    /// Reload shaders and recompile pipelines (useful for hot-reloading)
    /// @param context Metal context
    /// @param displayFormat Display pixel format
    /// @return true if reload succeeded
    bool reload(const MetalContext& context, MTLPixelFormat displayFormat);
    
    /// Release all pipeline state objects
    void shutdown();
    
    // Pipeline state accessors
    // The megakernel integrate pipelines (software, hardware, hardware with default features) are built on first
    // use: each is a cold compile of minutes, and a run dispatches one of them, or none on the wavefront. A failed
    // build logs the pipeline error and stops the run.
    MTLComputePipelineStateHandle integrate() const;
    MTLComputePipelineStateHandle integrateHardware() const;
    MTLComputePipelineStateHandle integrateHardwareDefaultFeatures() const;
    /// Whether the integrate pipelines can be built, without building them.
    bool integrateAvailable() const { return m_library != nullptr; }
    bool integrateHardwareAvailable() const;
    /// Builds the integrate pipelines not built yet on a background queue, one after another (the window calls it
    /// once a frame is on screen), so a later executor or feature switch finds them ready. Starts once.
    void buildIntegrateVariantsInBackground() const;
    MTLComputePipelineStateHandle wavefrontReset() const { return m_wavefrontResetPipeline; }
    MTLComputePipelineStateHandle wavefrontBuildDispatchArgs() const { return m_wavefrontBuildDispatchArgsPipeline; }
    MTLComputePipelineStateHandle wavefrontCompactQueue() const { return m_wavefrontCompactQueuePipeline; }
    MTLComputePipelineStateHandle wavefrontPrepareIteration() const { return m_wavefrontPrepareIterationPipeline; }
    MTLComputePipelineStateHandle wavefrontGeneratePrimary() const { return m_wavefrontGeneratePrimaryPipeline; }
    MTLComputePipelineStateHandle wavefrontIntersect() const { return m_wavefrontIntersectPipeline; }
    MTLComputePipelineStateHandle wavefrontIntersectHardware() const { return m_wavefrontIntersectHardwarePipeline; }
    MTLComputePipelineStateHandle wavefrontShade() const { return m_wavefrontShadePipeline; }
    MTLComputePipelineStateHandle wavefrontShadeResolveSurface() const { return m_wavefrontShadeResolveSurfacePipeline; }
    MTLComputePipelineStateHandle wavefrontShadeResolved() const { return m_wavefrontShadeResolvedPipeline; }
    /// The shade kernel's default-features specializations (function constant 2, every research and debug gate
    /// folded to 0, as the hardware megakernel's): built on first use without waiting on another build; nil when that
    /// build is busy or failed (the caller keeps the general shade pipeline).
    MTLComputePipelineStateHandle wavefrontShadeDefaultFeatures() const;
    MTLComputePipelineStateHandle wavefrontShadeResolvedDefaultFeatures() const;
    MTLComputePipelineStateHandle wavefrontShadeIndirectProbe() const { return m_wavefrontShadeIndirectProbePipeline; }
    MTLComputePipelineStateHandle wavefrontShadeSurfaceProbe() const { return m_wavefrontShadeSurfaceProbePipeline; }
    MTLComputePipelineStateHandle wavefrontShadeMetadataProbe() const { return m_wavefrontShadeMetadataProbePipeline; }
    MTLComputePipelineStateHandle wavefrontShadeScalarProbe() const { return m_wavefrontShadeScalarProbePipeline; }
    MTLComputePipelineStateHandle wavefrontShadeClosureProbe() const { return m_wavefrontShadeClosureProbePipeline; }
    MTLComputePipelineStateHandle wavefrontShadeBsdfProbe() const { return m_wavefrontShadeBsdfProbePipeline; }
    MTLComputePipelineStateHandle wavefrontRestirDiDirectLighting() const { return m_wavefrontRestirDiDirectLightingPipeline; }
    MTLComputePipelineStateHandle wavefrontBuildShadowIndices() const { return m_wavefrontBuildShadowIndicesPipeline; }
    MTLComputePipelineStateHandle wavefrontShadow() const { return m_wavefrontShadowPipeline; }
    MTLComputePipelineStateHandle wavefrontShadowIndexed() const { return m_wavefrontShadowIndexedPipeline; }
    MTLComputePipelineStateHandle wavefrontShadowHardware() const { return m_wavefrontShadowHardwarePipeline; }
    MTLComputePipelineStateHandle wavefrontShadowIndexedHardware() const { return m_wavefrontShadowIndexedHardwarePipeline; }
    MTLComputePipelineStateHandle wavefrontSwapQueues() const { return m_wavefrontSwapQueuesPipeline; }
    MTLComputePipelineStateHandle wavefrontAccumulate() const { return m_wavefrontAccumulatePipeline; }
    MTLComputePipelineStateHandle wavefrontRadianceCacheTrain() const { return m_wavefrontRadianceCacheTrainPipeline; }
    MTLComputePipelineStateHandle present() const { return m_presentPipeline; }
    MTLComputePipelineStateHandle svgfTemporal() const { return m_svgfTemporalPipeline; }
    MTLComputePipelineStateHandle svgfAtrous() const { return m_svgfAtrousPipeline; }
    MTLComputePipelineStateHandle svgfCompositeSplit() const { return m_svgfCompositeSplitPipeline; }
    MTLComputePipelineStateHandle mlReconstructionTiny() const { return m_mlReconstructionTinyPipeline; }
    MTLComputePipelineStateHandle restirDiInitializeReservoirs() const { return m_restirDiInitializeReservoirsPipeline; }
    MTLComputePipelineStateHandle restirDiInitialCandidates() const { return m_restirDiInitialCandidatesPipeline; }
    MTLComputePipelineStateHandle restirDiInitialCandidatesHardware() const { return m_restirDiInitialCandidatesHardwarePipeline; }
    MTLComputePipelineStateHandle restirDiTemporalReuse() const { return m_restirDiTemporalReusePipeline; }
    MTLComputePipelineStateHandle restirDiSpatialReuse() const { return m_restirDiSpatialReusePipeline; }
    MTLComputePipelineStateHandle restirDiVisibility() const { return m_restirDiVisibilityPipeline; }
    MTLComputePipelineStateHandle restirDiVisibilityHardware() const { return m_restirDiVisibilityHardwarePipeline; }
    MTLComputePipelineStateHandle restirDiApplyLighting() const { return m_restirDiApplyLightingPipeline; }
    MTLComputePipelineStateHandle restirDiDebugAov() const { return m_restirDiDebugAovPipeline; }
    MTLComputePipelineStateHandle clear() const { return m_clearPipeline; }
    MTLRenderPipelineStateHandle display() const { return m_displayPipeline; }
    MTLLibraryHandle library() const { return m_library; }
    MTLArgumentEncoderHandle materialResourcesArgumentEncoder() const { return m_materialResourcesArgumentEncoder; }
    /// The integrate pipelines' build state, shared with a background build (Pipelines.mm).
    struct LazyIntegrateState;

    /// The compile records, with those of integrate pipelines built since initialization.
    CompilationReport compilationReport() const;
    
    /// Check if pipelines are valid and ready to use
    bool isValid() const {
        return m_presentPipeline != nullptr && 
               m_clearPipeline != nullptr && 
               m_displayPipeline != nullptr &&
               m_library != nullptr;
    }
    
private:
    MTLDeviceHandle m_device = nullptr;
    MTLLibraryHandle m_library = nullptr;
    
    // Compute pipelines
    // Built on first use; shared with a background build, which may outlive a reload (Pipelines.mm).
    std::shared_ptr<LazyIntegrateState> m_lazyIntegrate;
    static MTLComputePipelineStateHandle integrateVariant(const std::shared_ptr<LazyIntegrateState>& state,
                                                          int variant,
                                                          bool foreground);
    static MTLComputePipelineStateHandle lazyVariantWithoutWaiting(const std::shared_ptr<LazyIntegrateState>& state,
                                                                   int variant);
    MTLComputePipelineStateHandle m_wavefrontResetPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontBuildDispatchArgsPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontCompactQueuePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontPrepareIterationPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontGeneratePrimaryPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontIntersectPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontIntersectHardwarePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontShadePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontShadeResolveSurfacePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontShadeResolvedPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontShadeIndirectProbePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontShadeSurfaceProbePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontShadeMetadataProbePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontShadeScalarProbePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontShadeClosureProbePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontShadeBsdfProbePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontRestirDiDirectLightingPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontBuildShadowIndicesPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontShadowPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontShadowIndexedPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontShadowHardwarePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontShadowIndexedHardwarePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontSwapQueuesPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontAccumulatePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontRadianceCacheTrainPipeline = nullptr;
    MTLComputePipelineStateHandle m_presentPipeline = nullptr;
    MTLComputePipelineStateHandle m_svgfTemporalPipeline = nullptr;
    MTLComputePipelineStateHandle m_svgfAtrousPipeline = nullptr;
    MTLComputePipelineStateHandle m_svgfCompositeSplitPipeline = nullptr;
    MTLComputePipelineStateHandle m_mlReconstructionTinyPipeline = nullptr;
    MTLComputePipelineStateHandle m_restirDiInitializeReservoirsPipeline = nullptr;
    MTLComputePipelineStateHandle m_restirDiInitialCandidatesPipeline = nullptr;
    MTLComputePipelineStateHandle m_restirDiInitialCandidatesHardwarePipeline = nullptr;
    MTLComputePipelineStateHandle m_restirDiTemporalReusePipeline = nullptr;
    MTLComputePipelineStateHandle m_restirDiSpatialReusePipeline = nullptr;
    MTLComputePipelineStateHandle m_restirDiVisibilityPipeline = nullptr;
    MTLComputePipelineStateHandle m_restirDiVisibilityHardwarePipeline = nullptr;
    MTLComputePipelineStateHandle m_restirDiApplyLightingPipeline = nullptr;
    MTLComputePipelineStateHandle m_restirDiDebugAovPipeline = nullptr;
    MTLComputePipelineStateHandle m_clearPipeline = nullptr;
    MTLArgumentEncoderHandle m_materialResourcesArgumentEncoder = nullptr;
    MTLBinaryArchiveHandle m_binaryArchive = nullptr;
    std::string m_binaryArchivePath;
    
    // Render pipeline
    MTLRenderPipelineStateHandle m_displayPipeline = nullptr;
    bool m_verboseTiming = false;
    bool m_sourceFeatures = false;
    CompilationReport m_compilationReport{};
    
    bool compileShaders(const MetalContext& context);
    bool createComputePipelines();
    bool createDisplayPipeline(MTLPixelFormat displayFormat);
};

}  // namespace PathTracer
