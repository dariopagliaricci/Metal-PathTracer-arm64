#pragma once

#include <CoreGraphics/CoreGraphics.h>
#include <cstddef>
#include <cstdint>
#include "renderer/MetalHandles.h"

namespace PathTracer {

/// Manages progressive accumulation of path-traced samples
/// Handles accumulation textures, frame counting, and clearing
class Accumulation {
public:
    Accumulation() = default;
    ~Accumulation() = default;
    
    // Non-copyable
    Accumulation(const Accumulation&) = delete;
    Accumulation& operator=(const Accumulation&) = delete;
    
    /// Initialize with Metal device
    void initialize(MTLDeviceHandle device);
    
    /// Ensure accumulation textures exist for the given size
    /// @param drawableSize The target drawable size
    /// @param force Force recreation even if size matches
    void ensureTextures(CGSize drawableSize, bool force = false);
    
    /// Clear accumulation textures using compute shader
    /// @param commandBuffer Command buffer for encoding
    /// @param clearPipeline Pipeline state for clear kernel
    void clear(MTLCommandBufferHandle commandBuffer,
               MTLComputePipelineStateHandle clearPipeline);
    
    /// Reset accumulation state (frame/sample counters) and optionally clear textures
    void reset(MTLCommandQueueHandle commandQueue,
               MTLComputePipelineStateHandle clearPipeline,
               MTKViewHandle view);
    
    /// Update drawable size for the view and resize textures if needed.
    /// @return Updated content scale factor used for UI scaling.
    float updateDrawableSize(NSWindowHandle window, MTKViewHandle view);
    
    /// Release owned GPU resources and reset internal state.
    void teardown();
    
    /// Mark that textures need clearing on next frame
    void markNeedsClear() { m_needsClear = true; }
    
    /// Check if textures need clearing
    bool needsClear() const { return m_needsClear; }
    
    /// Increment frame and sample counters
    void incrementFrame() { 
        m_frameIndex++; 
        m_sampleCount++; 
    }
    
    // Texture accessors
    MTLTextureHandle radianceSum() const { return m_radianceSumTexture; }
    MTLTextureHandle sampleCountTexture() const { return m_sampleCountTexture; }
    MTLTextureHandle present() const { return m_presentTexture; }
    MTLTextureHandle albedoBuffer() const { return m_albedoTexture; }
    MTLTextureHandle normalBuffer() const { return m_normalTexture; }
    MTLTextureHandle denoisedBuffer() const { return m_denoisedTexture; }
    
    // State accessors
    uint32_t frameIndex() const { return m_frameIndex; }
    uint32_t sampleCount() const { return m_sampleCount; }
    CGSize currentSize() const { return m_currentSize; }
    
private:
    MTLDeviceHandle m_device = nullptr;

    // Accumulation textures
    MTLTextureHandle m_radianceSumTexture = nullptr;
    MTLTextureHandle m_sampleCountTexture = nullptr;
    MTLTextureHandle m_presentTexture = nullptr;
    MTLTextureHandle m_albedoTexture = nullptr;
    MTLTextureHandle m_normalTexture = nullptr;
    MTLTextureHandle m_denoisedTexture = nullptr;

    // Accumulation state
    uint32_t m_frameIndex = 0;
    uint32_t m_sampleCount = 0;
    CGSize m_currentSize = CGSizeZero;
    bool m_needsClear = false;
};

}  // namespace PathTracer
