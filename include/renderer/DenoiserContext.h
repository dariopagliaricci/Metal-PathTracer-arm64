#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include "renderer/MetalHandles.h"

namespace PathTracer {

/// Manages OIDN denoising device and filter lifecycle
/// Handles:
/// - OIDN device creation and initialization
/// - Filter creation and configuration
/// - Metal device/buffer interoperability
/// - Error tracking and logging
class DenoiserContext {
public:
    enum class FilterType : uint32_t {
        RT = 0,           // High-quality ray tracing denoiser
        RTLightmap = 1,   // Specialized for light map denoising
    };

    enum class DeviceStatus : uint32_t {
        Uninitialized = 0,
        Initializing = 1,
        Ready = 2,
        Failed = 3,
    };

    DenoiserContext();
    ~DenoiserContext();

    // Non-copyable
    DenoiserContext(const DenoiserContext&) = delete;
    DenoiserContext& operator=(const DenoiserContext&) = delete;

    /// Initialize OIDN device with Metal backend
    /// Must be called once before any denoising operations
    /// @param metalDevice Metal device to use for denoising
    /// @return true if initialization succeeded
    bool initialize(MTLDeviceHandle metalDevice, MTLCommandQueueHandle commandQueue = nullptr);

    /// Check if denoiser is ready for use
    bool isReady() const { return m_status == DeviceStatus::Ready; }

    /// Get current device status
    DeviceStatus status() const { return m_status; }

    /// Get last error message
    const std::string& lastError() const { return m_lastError; }

    /// Execute denoising on input texture
    /// Requires:
    /// - Color (beauty) texture with noise
    /// - Optional albedo texture for edge preservation
    /// - Optional normal texture for detail preservation
    /// @param colorInput Input noisy color buffer (RGBA, float)
    /// @param albedoInput Optional albedo buffer for edge preservation
    /// @param normalInput Optional normal buffer for detail preservation
    /// @param colorOutput Output denoised color buffer (same format as input)
    /// @param filterType Which OIDN filter to use
    /// @return true if denoising succeeded
    bool denoise(MTLTextureHandle colorInput,
                 MTLTextureHandle albedoInput,
                 MTLTextureHandle normalInput,
                 MTLTextureHandle colorOutput,
                 FilterType filterType = FilterType::RT);

    /// Execute denoising directly on linear RGB buffers.
    /// Optional albedo is linear RGB. Optional normal is expected in [0,1] encoding
    /// matching the accumulation normal buffer; it will be remapped to [-1,1].
    bool denoiseLinearBuffers(const float* colorInputRGB,
                              const float* albedoInputRGB,
                              const float* normalInputRGB,
                              uint32_t width,
                              uint32_t height,
                              std::vector<float>& outDenoisedRGB,
                              FilterType filterType = FilterType::RT,
                              bool hdr = true);

    /// Shutdown and release all OIDN resources
    void shutdown();

    /// Release reusable Metal transfer buffers while keeping the OIDN CPU device alive.
    void releaseMetalStagingBuffers();

private:
    // OIDN device and filter (stored as opaque void* to avoid exposing OIDN headers)
    // These are actually OIDNDevice and OIDNFilter pointers, but stored as void*
    void* m_device = nullptr;
    void* m_filter = nullptr;

    // State tracking
    DeviceStatus m_status = DeviceStatus::Uninitialized;
    std::string m_lastError;
    MTLDeviceHandle m_metalDevice = nullptr;
    MTLCommandQueueHandle m_commandQueue = nullptr;
    FilterType m_currentFilterType = FilterType::RT;

    MTLBufferHandle m_colorReadbackBuffer = nullptr;
    MTLBufferHandle m_albedoReadbackBuffer = nullptr;
    MTLBufferHandle m_normalReadbackBuffer = nullptr;
    MTLBufferHandle m_writebackBuffer = nullptr;
    size_t m_colorReadbackBufferBytes = 0;
    size_t m_albedoReadbackBufferBytes = 0;
    size_t m_normalReadbackBufferBytes = 0;
    size_t m_writebackBufferBytes = 0;

    // CPU buffers for Metal↔OIDN data transfer
    std::vector<float> m_colorBuffer;     // Input color (beauty) buffer
    std::vector<float> m_albedoBuffer;    // Optional albedo buffer
    std::vector<float> m_normalBuffer;    // Optional normal buffer
    std::vector<float> m_outputBuffer;    // Output denoised buffer

    /// Create and configure OIDN filter for specified type
    /// @return true if filter creation/configuration succeeded
    bool createFilter(FilterType type);

    /// Update filter for new input dimensions
    bool updateFilterDimensions(uint32_t width, uint32_t height);
};

}  // namespace PathTracer
