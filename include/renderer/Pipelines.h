#pragma once

#include "renderer/MetalHandles.h"

namespace PathTracer {

// Forward declaration
class MetalContext;

/// Manages all Metal pipeline states and shader compilation
/// Handles shader source loading, compilation, and pipeline state object creation
class Pipelines {
public:
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
    MTLComputePipelineStateHandle present() const { return m_presentPipeline; }
    MTLComputePipelineStateHandle clear() const { return m_clearPipeline; }
    MTLRenderPipelineStateHandle display() const { return m_displayPipeline; }
    MTLLibraryHandle library() const { return m_library; }
    
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
    MTLComputePipelineStateHandle m_presentPipeline = nullptr;
    MTLComputePipelineStateHandle m_clearPipeline = nullptr;
    
    // Render pipeline
    MTLRenderPipelineStateHandle m_displayPipeline = nullptr;
    
    bool compileShaders(const MetalContext& context);
    bool createComputePipelines();
    bool createDisplayPipeline(MTLPixelFormat displayFormat);
};

}  // namespace PathTracer
