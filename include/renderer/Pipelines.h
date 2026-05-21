#pragma once

#include "renderer/MetalHandles.h"

#include <cstdint>
#include <string>
#include <vector>

namespace PathTracer {

// Forward declaration
class MetalContext;

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
        bool harvestedCompilationReady = false;
        uint64_t shaderSourceBytes = 0;
        double libraryCompileMs = 0.0;
        uint32_t computePipelineCount = 0;
        uint32_t renderPipelineCount = 0;
        uint32_t failedPipelineCount = 0;
        std::string cacheStrategy = "runtime_source_library_no_binary_archive";
        std::string specializationStrategy =
            "separate pipeline keys for execution/backend features; function constants not yet used";
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
    
    /// Reload shaders and recompile pipelines (useful for hot-reloading)
    /// @param context Metal context
    /// @param displayFormat Display pixel format
    /// @return true if reload succeeded
    bool reload(const MetalContext& context, MTLPixelFormat displayFormat);
    
    /// Release all pipeline state objects
    void shutdown();
    
    // Pipeline state accessors
    MTLComputePipelineStateHandle integrate() const { return m_integratePipeline; }
    MTLComputePipelineStateHandle integrateHardware() const { return m_integrateHardwarePipeline; }
    MTLComputePipelineStateHandle wavefrontReset() const { return m_wavefrontResetPipeline; }
    MTLComputePipelineStateHandle wavefrontBuildDispatchArgs() const { return m_wavefrontBuildDispatchArgsPipeline; }
    MTLComputePipelineStateHandle wavefrontCompactQueue() const { return m_wavefrontCompactQueuePipeline; }
    MTLComputePipelineStateHandle wavefrontPrepareIteration() const { return m_wavefrontPrepareIterationPipeline; }
    MTLComputePipelineStateHandle wavefrontGeneratePrimary() const { return m_wavefrontGeneratePrimaryPipeline; }
    MTLComputePipelineStateHandle wavefrontIntersect() const { return m_wavefrontIntersectPipeline; }
    MTLComputePipelineStateHandle wavefrontIntersectHardware() const { return m_wavefrontIntersectHardwarePipeline; }
    MTLComputePipelineStateHandle wavefrontShade() const { return m_wavefrontShadePipeline; }
    MTLComputePipelineStateHandle wavefrontRestirDiDirectLighting() const { return m_wavefrontRestirDiDirectLightingPipeline; }
    MTLComputePipelineStateHandle wavefrontShadow() const { return m_wavefrontShadowPipeline; }
    MTLComputePipelineStateHandle wavefrontShadowHardware() const { return m_wavefrontShadowHardwarePipeline; }
    MTLComputePipelineStateHandle wavefrontSwapQueues() const { return m_wavefrontSwapQueuesPipeline; }
    MTLComputePipelineStateHandle wavefrontAccumulate() const { return m_wavefrontAccumulatePipeline; }
    MTLComputePipelineStateHandle present() const { return m_presentPipeline; }
    MTLComputePipelineStateHandle svgfTemporal() const { return m_svgfTemporalPipeline; }
    MTLComputePipelineStateHandle svgfAtrous() const { return m_svgfAtrousPipeline; }
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
    const CompilationReport& compilationReport() const { return m_compilationReport; }
    
    /// Check if pipelines are valid and ready to use
    bool isValid() const {
        return m_integratePipeline != nullptr && 
               m_presentPipeline != nullptr && 
               m_clearPipeline != nullptr && 
               m_displayPipeline != nullptr &&
               m_library != nullptr;
    }
    
private:
    MTLDeviceHandle m_device = nullptr;
    MTLLibraryHandle m_library = nullptr;
    
    // Compute pipelines
    MTLComputePipelineStateHandle m_integratePipeline = nullptr;
    MTLComputePipelineStateHandle m_integrateHardwarePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontResetPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontBuildDispatchArgsPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontCompactQueuePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontPrepareIterationPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontGeneratePrimaryPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontIntersectPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontIntersectHardwarePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontShadePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontRestirDiDirectLightingPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontShadowPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontShadowHardwarePipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontSwapQueuesPipeline = nullptr;
    MTLComputePipelineStateHandle m_wavefrontAccumulatePipeline = nullptr;
    MTLComputePipelineStateHandle m_presentPipeline = nullptr;
    MTLComputePipelineStateHandle m_svgfTemporalPipeline = nullptr;
    MTLComputePipelineStateHandle m_svgfAtrousPipeline = nullptr;
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
    
    // Render pipeline
    MTLRenderPipelineStateHandle m_displayPipeline = nullptr;
    bool m_verboseTiming = false;
    CompilationReport m_compilationReport{};
    
    bool compileShaders(const MetalContext& context);
    bool createComputePipelines();
    bool createDisplayPipeline(MTLPixelFormat displayFormat);
};

}  // namespace PathTracer
