#pragma once

// ⚠️ INTERNAL HEADER - Subject to change without notice
// This header is part of PathTracer's internal implementation.
// Use only the public API from MetalRenderer.h

#include "renderer/MetalHandles.h"
#include "PerformanceStats.h"
#include "RenderSettings.h"

namespace PathTracer {

// Forward declarations
class MetalContext;
class Accumulation;
class Pipelines;
class SceneResources;
class UIOverlay;
class DenoiserContext;

/// Result from encoding a frame
struct FrameResult {
    double gpuTimeMs = 0.0;
    double avgNodesVisited = 0.0;
    double avgLeafPrimTests = 0.0;
    double shadowRayEarlyExitPct = 0.0;
    double bothChildrenVisitedPct = 0.0;
    uint32_t samplesDispatched = 0;
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
    
    /// Cleanup resources
    void shutdown();

    /// Access the statistics buffer used for GPU counters
    MTLBufferHandle statsBuffer() const { return m_statsBuffer; }
    MTLBufferHandle debugBuffer() const { return m_debugBuffer; }
    
private:
    MTLDeviceHandle m_device = nullptr;
    DenoiserContext* m_denoiser = nullptr;

    // Uniform buffers
    MTLBufferHandle m_uniformBuffer = nullptr;
    MTLBufferHandle m_displayUniformBuffer = nullptr;
    MTLBufferHandle m_statsBuffer = nullptr;
    MTLBufferHandle m_debugBuffer = nullptr;

    // Frame counter for progressive denoising
    uint32_t m_frameCounter = 0;

    uint32_t encodeIntegration(MTLCommandBufferHandle commandBuffer,
                               Accumulation& accumulation,
                               const Pipelines& pipelines,
                               const SceneResources& scene,
                               const RenderSettings& settings);

    void encodePresentation(MTLCommandBufferHandle commandBuffer,
                            const Accumulation& accumulation,
                            const Pipelines& pipelines);

    void encodeDenoising(MTLCommandBufferHandle commandBuffer,
                         Accumulation& accumulation,
                         const RenderSettings& settings,
                         uint32_t frameIndex);

    void encodeDisplay(MTLRenderCommandEncoderHandle renderEncoder,
                       const Accumulation& accumulation,
                       const Pipelines& pipelines,
                       const RenderSettings& settings);
};

}  // namespace PathTracer
