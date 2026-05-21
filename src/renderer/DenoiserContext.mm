#import "renderer/DenoiserContext.h"

#include <sstream>
#include <cstring>
#include <vector>

#ifndef PATH_TRACER_ENABLE_OIDN
#define PATH_TRACER_ENABLE_OIDN 0
#endif

#if PATH_TRACER_ENABLE_OIDN
#include <OpenImageDenoise/oidn.h>
#endif

#if PATH_TRACER_ENABLE_OIDN
namespace PathTracer {

namespace {
    /// Helper to convert Metal pixel format to bytes per pixel
    size_t bytesPerPixelForFormat(MTLPixelFormat format) {
        switch (format) {
            case MTLPixelFormatRGBA32Float:
                return 16;  // 4 x float32
            case MTLPixelFormatRGBA16Float:
                return 8;   // 4 x float16
            default:
                return 0;
        }
    }

    bool ensureStagingBuffer(id<MTLDevice> device,
                             MTLBufferHandle __strong& buffer,
                             size_t& bufferBytes,
                             size_t requiredBytes,
                             const char* purpose,
                             std::string& outError) {
        if (!device) {
            outError = "No Metal device available for staging buffer";
            return false;
        }
        if (buffer && bufferBytes >= requiredBytes) {
            return true;
        }

        buffer = [device newBufferWithLength:requiredBytes options:MTLResourceStorageModeShared];
        if (!buffer) {
            std::ostringstream oss;
            oss << "Failed to allocate " << purpose << " staging buffer (" << requiredBytes << " bytes)";
            outError = oss.str();
            bufferBytes = 0;
            return false;
        }
        bufferBytes = requiredBytes;
        return true;
    }

    /// Readback Metal texture to CPU buffer
    /// Returns true if successful, false otherwise
    bool readMetalTextureToCPU(MTLTextureHandle texture,
                               MTLCommandQueueHandle commandQueue,
                               MTLBufferHandle __strong& stagingBuffer,
                               size_t& stagingBufferBytes,
                               std::vector<float>& outBuffer,
                               size_t& outWidth,
                               size_t& outHeight,
                               std::string& outError) {
        if (!texture) {
            outError = "Invalid texture handle";
            return false;
        }

        outWidth = texture.width;
        outHeight = texture.height;
        size_t pixelCount = outWidth * outHeight;

        // Validate format
        if (texture.pixelFormat != MTLPixelFormatRGBA32Float &&
            texture.pixelFormat != MTLPixelFormatRGBA16Float) {
            outError = "Unsupported texture format for readback";
            return false;
        }

        size_t bytesPerPixel = bytesPerPixelForFormat(texture.pixelFormat);
        if (bytesPerPixel == 0) {
            outError = "Unknown texture format";
            return false;
        }

        size_t totalBytes = pixelCount * bytesPerPixel;

        // Allocate buffer (always as float32 for OIDN compatibility)
        try {
            outBuffer.resize(pixelCount * 4);  // RGBA as 4 floats
        } catch (const std::bad_alloc& e) {
            std::ostringstream oss;
            oss << "Failed to allocate CPU buffer (" << totalBytes << " bytes)";
            outError = oss.str();
            return false;
        }

        // Create temporary buffer for readback
        @autoreleasepool {
            id<MTLDevice> device = texture.device;
            if (!commandQueue) {
                outError = "No command queue available for texture readback";
                return false;
            }

            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            if (!commandBuffer) {
                outError = "Failed to create command buffer for texture readback";
                return false;
            }

            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
            if (!blitEncoder) {
                outError = "Failed to create blit encoder for texture readback";
                return false;
            }

            if (!ensureStagingBuffer(device,
                                     stagingBuffer,
                                     stagingBufferBytes,
                                     totalBytes,
                                     "readback",
                                     outError)) {
                [blitEncoder endEncoding];
                return false;
            }

            // Blit texture to buffer
            MTLOrigin origin = MTLOriginMake(0, 0, 0);
            MTLSize size = MTLSizeMake(outWidth, outHeight, 1);
            [blitEncoder copyFromTexture:texture
                             sourceSlice:0
                             sourceLevel:0
                            sourceOrigin:origin
                              sourceSize:size
                                toBuffer:stagingBuffer
                       destinationOffset:0
                  destinationBytesPerRow:outWidth * bytesPerPixel
                destinationBytesPerImage:outWidth * outHeight * bytesPerPixel];

            [blitEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            // Convert texture data to float32 if needed
            const void* srcData = [stagingBuffer contents];
            if (texture.pixelFormat == MTLPixelFormatRGBA32Float) {
                // Direct copy
                std::memcpy(outBuffer.data(), srcData, totalBytes);
            } else if (texture.pixelFormat == MTLPixelFormatRGBA16Float) {
                // Convert float16 to float32
                // This is a simplified conversion - proper implementation would use Apple's simd library
                const uint16_t* srcHalf = (const uint16_t*)srcData;
                for (size_t i = 0; i < pixelCount * 4; ++i) {
                    // Simplified float16 to float32 conversion
                    // In production, use proper half-precision conversion
                    outBuffer[i] = float(srcHalf[i]) / 65504.0f;
                }
            }
        }

        return true;
    }

    /// Writeback CPU buffer to Metal texture
    bool writeMetalTextureFromCPU(MTLTextureHandle texture,
                                  MTLCommandQueueHandle commandQueue,
                                  MTLBufferHandle __strong& stagingBuffer,
                                  size_t& stagingBufferBytes,
                                  const std::vector<float>& buffer,
                                  std::string& outError) {
        if (!texture) {
            outError = "Invalid texture handle";
            return false;
        }

        if (buffer.empty()) {
            outError = "Empty buffer for writeback";
            return false;
        }

        size_t width = texture.width;
        size_t height = texture.height;
        size_t pixelCount = width * height;

        if (buffer.size() < pixelCount * 4) {
            outError = "Buffer too small for texture";
            return false;
        }

        size_t bytesPerPixel = bytesPerPixelForFormat(texture.pixelFormat);
        if (bytesPerPixel == 0) {
            outError = "Unsupported texture format for writeback";
            return false;
        }

        size_t totalBytes = pixelCount * bytesPerPixel;

        @autoreleasepool {
            id<MTLDevice> device = texture.device;
            if (!commandQueue) {
                outError = "No command queue available for texture writeback";
                return false;
            }

            if (!ensureStagingBuffer(device,
                                     stagingBuffer,
                                     stagingBufferBytes,
                                     totalBytes,
                                     "writeback",
                                     outError)) {
                return false;
            }

            // Convert float32 to texture format if needed
            void* dstData = [stagingBuffer contents];
            if (texture.pixelFormat == MTLPixelFormatRGBA32Float) {
                std::memcpy(dstData, buffer.data(), totalBytes);
            } else if (texture.pixelFormat == MTLPixelFormatRGBA16Float) {
                // Convert float32 to float16
                uint16_t* dstHalf = (uint16_t*)dstData;
                for (size_t i = 0; i < pixelCount * 4; ++i) {
                    // Simplified float32 to float16 conversion
                    dstHalf[i] = (uint16_t)(buffer[i] * 65504.0f);
                }
            }

            // Blit buffer to texture
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            if (!commandBuffer) {
                outError = "Failed to create command buffer for texture writeback";
                return false;
            }
            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
            if (!blitEncoder) {
                outError = "Failed to create blit encoder for texture writeback";
                return false;
            }

            MTLOrigin origin = MTLOriginMake(0, 0, 0);
            MTLSize size = MTLSizeMake(width, height, 1);
            [blitEncoder copyFromBuffer:stagingBuffer
                           sourceOffset:0
                      sourceBytesPerRow:width * bytesPerPixel
                    sourceBytesPerImage:width * height * bytesPerPixel
                             sourceSize:size
                              toTexture:texture
                       destinationSlice:0
                       destinationLevel:0
                      destinationOrigin:origin];

            [blitEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }

        return true;
    }
}

DenoiserContext::DenoiserContext()
    : m_device(nullptr),
      m_filter(nullptr),
      m_status(DeviceStatus::Uninitialized),
      m_lastError("Not initialized"),
      m_metalDevice(nullptr),
      m_commandQueue(nullptr),
      m_currentFilterType(FilterType::RT),
      m_colorBuffer(),
      m_albedoBuffer(),
      m_normalBuffer(),
      m_outputBuffer() {
}

DenoiserContext::~DenoiserContext() {
    shutdown();
}

bool DenoiserContext::initialize(MTLDeviceHandle metalDevice, MTLCommandQueueHandle commandQueue) {
    m_status = DeviceStatus::Initializing;
    m_metalDevice = metalDevice;
    m_commandQueue = commandQueue;

    try {
        // Create OIDN device with Metal backend using C API
        // NOTE: This creates a CPU-based OIDN device. Metal GPU denoising would require
        // Metal compute shaders or OIDN's experimental Metal support.
        OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_CPU);
        if (!device) {
            m_lastError = "Failed to create OIDN CPU device";
            m_status = DeviceStatus::Failed;
            return false;
        }

        // Commit device
        oidnCommitDevice(device);

        // Check for errors
        const char* errorMsg = nullptr;
        OIDNError error = oidnGetDeviceError(device, &errorMsg);
        if (error != OIDN_ERROR_NONE) {
            m_lastError = std::string("OIDN device setup failed");
            if (errorMsg) {
                m_lastError += std::string(": ") + errorMsg;
            }
            oidnReleaseDevice(device);
            m_status = DeviceStatus::Failed;
            return false;
        }

        // Store device pointer
        m_device = (void*)device;

        // Create initial filter
        if (!createFilter(FilterType::RT)) {
            m_lastError = "Failed to create initial denoising filter";
            m_status = DeviceStatus::Failed;
            oidnReleaseDevice(device);
            m_device = nullptr;
            return false;
        }

        m_status = DeviceStatus::Ready;
        m_lastError = "";
        return true;

    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "OIDN initialization exception: " << e.what();
        m_lastError = oss.str();
        m_status = DeviceStatus::Failed;
        return false;
    } catch (...) {
        m_lastError = "Unknown exception during OIDN initialization";
        m_status = DeviceStatus::Failed;
        return false;
    }
}

bool DenoiserContext::createFilter(FilterType type) {
    if (!m_device) {
        m_lastError = "Device not initialized";
        return false;
    }

    try {
        OIDNDevice device = (OIDNDevice)m_device;

        // Select filter type
        const char* filterType = (type == FilterType::RTLightmap) ? "RTLightmap" : "RT";

        // Create filter
        OIDNFilter filter = oidnNewFilter(device, filterType);
        if (!filter) {
            m_lastError = std::string("Failed to create OIDN ") + filterType + " filter";
            return false;
        }

        // Release old filter if it exists
        if (m_filter) {
            oidnReleaseFilter((OIDNFilter)m_filter);
        }

        // Store filter pointer
        m_filter = (void*)filter;
        m_currentFilterType = type;
        m_lastError = "";
        return true;

    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "OIDN filter creation exception: " << e.what();
        m_lastError = oss.str();
        return false;
    } catch (...) {
        m_lastError = "Unknown exception during filter creation";
        return false;
    }
}

bool DenoiserContext::denoise(MTLTextureHandle colorInput,
                              MTLTextureHandle albedoInput,
                              MTLTextureHandle normalInput,
                              MTLTextureHandle colorOutput,
                              FilterType filterType) {
    if (!isReady()) {
        m_lastError = "Denoiser not ready (status: " + std::to_string(static_cast<int>(m_status)) + ")";
        return false;
    }

    if (!colorInput || !colorOutput) {
        m_lastError = "Invalid input/output texture handles";
        return false;
    }

    try {
        // Validate texture formats
        if (colorInput.pixelFormat != MTLPixelFormatRGBA32Float &&
            colorInput.pixelFormat != MTLPixelFormatRGBA16Float) {
            m_lastError = "Color input must be RGBA32Float or RGBA16Float";
            return false;
        }

        if (colorOutput.pixelFormat != colorInput.pixelFormat) {
            m_lastError = "Output format must match input format";
            return false;
        }

        // Validate dimensions match
        if (colorInput.width != colorOutput.width || colorInput.height != colorOutput.height) {
            m_lastError = "Input and output dimensions must match";
            return false;
        }

        // Recreate filter if type changed
        if (m_currentFilterType != filterType) {
            if (!createFilter(filterType)) {
                return false;
            }
        }

        OIDNDevice device = (OIDNDevice)m_device;
        OIDNFilter filter = (OIDNFilter)m_filter;

        // STEP 1: Readback color texture to CPU buffer
        size_t colorWidth, colorHeight;
        if (!readMetalTextureToCPU(colorInput,
                                   m_commandQueue,
                                   m_colorReadbackBuffer,
                                   m_colorReadbackBufferBytes,
                                   m_colorBuffer,
                                   colorWidth,
                                   colorHeight,
                                   m_lastError)) {
            m_lastError = std::string("Failed to readback color texture: ") + m_lastError;
            return false;
        }

        // STEP 2: Readback optional auxiliary buffers
        size_t albedoWidth = 0, albedoHeight = 0;
        if (albedoInput) {
            if (!readMetalTextureToCPU(albedoInput,
                                       m_commandQueue,
                                       m_albedoReadbackBuffer,
                                       m_albedoReadbackBufferBytes,
                                       m_albedoBuffer,
                                       albedoWidth,
                                       albedoHeight,
                                       m_lastError)) {
                NSLog(@"[OIDN] Warning: Failed to readback albedo texture, continuing without it");
                m_albedoBuffer.clear();
            } else if (albedoWidth != colorWidth || albedoHeight != colorHeight) {
                NSLog(@"[OIDN] Warning: Albedo dimensions (%zu x %zu) do not match color buffer (%zu x %zu); ignoring albedo AOV",
                      albedoWidth, albedoHeight, colorWidth, colorHeight);
                m_albedoBuffer.clear();
            }
        }

        size_t normalWidth = 0, normalHeight = 0;
        if (normalInput) {
            if (!readMetalTextureToCPU(normalInput,
                                       m_commandQueue,
                                       m_normalReadbackBuffer,
                                       m_normalReadbackBufferBytes,
                                       m_normalBuffer,
                                       normalWidth,
                                       normalHeight,
                                       m_lastError)) {
                NSLog(@"[OIDN] Warning: Failed to readback normal texture, continuing without it");
                m_normalBuffer.clear();
            } else if (normalWidth != colorWidth || normalHeight != colorHeight) {
                NSLog(@"[OIDN] Warning: Normal dimensions (%zu x %zu) do not match color buffer (%zu x %zu); ignoring normal AOV",
                      normalWidth, normalHeight, colorWidth, colorHeight);
                m_normalBuffer.clear();
            } else {
                const size_t pixelCount = normalWidth * normalHeight;
                for (size_t i = 0; i < pixelCount; ++i) {
                    float* n = m_normalBuffer.data() + i * 4;
                    n[0] = n[0] * 2.0f - 1.0f;
                    n[1] = n[1] * 2.0f - 1.0f;
                    n[2] = n[2] * 2.0f - 1.0f;
                }
            }
        }

        // STEP 3: Allocate output buffer
        try {
            m_outputBuffer.resize(colorWidth * colorHeight * 4);
        } catch (const std::bad_alloc&) {
            m_lastError = "Failed to allocate output buffer for denoising";
            return false;
        }

        // STEP 4: Set images in OIDN filter
        // Using oidnSetSharedFilterImage to provide CPU buffers to OIDN
        const size_t colorPixelStride = sizeof(float) * 4;
        const size_t colorRowStride = colorPixelStride * colorWidth;
        oidnSetSharedFilterImage(filter,
                                "color",
                                m_colorBuffer.data(),
                                OIDN_FORMAT_FLOAT3,
                                colorWidth,
                                colorHeight,
                                0,  // byteOffset
                                colorPixelStride,
                                colorRowStride);

        if (!m_albedoBuffer.empty() && albedoInput) {
            const size_t albedoPixelStride = sizeof(float) * 4;
            const size_t albedoRowStride = albedoPixelStride * albedoWidth;
            oidnSetSharedFilterImage(filter,
                                    "albedo",
                                    m_albedoBuffer.data(),
                                    OIDN_FORMAT_FLOAT3,
                                    albedoWidth,
                                    albedoHeight,
                                    0,
                                    albedoPixelStride,
                                    albedoRowStride);
        }

        if (!m_normalBuffer.empty() && normalInput) {
            const size_t normalPixelStride = sizeof(float) * 4;
            const size_t normalRowStride = normalPixelStride * normalWidth;
            oidnSetSharedFilterImage(filter,
                                    "normal",
                                    m_normalBuffer.data(),
                                    OIDN_FORMAT_FLOAT3,
                                    normalWidth,
                                    normalHeight,
                                    0,
                                    normalPixelStride,
                                    normalRowStride);
        }

        const size_t outputPixelStride = sizeof(float) * 4;
        const size_t outputRowStride = outputPixelStride * colorWidth;
        // Output buffer
        oidnSetSharedFilterImage(filter,
                                "output",
                                m_outputBuffer.data(),
                                OIDN_FORMAT_FLOAT3,
                                colorWidth,
                                colorHeight,
                                0,
                                outputPixelStride,
                                outputRowStride);

        oidnSetFilterBool(filter, "hdr", true);
        oidnSetFilterBool(filter, "srgb", false);
        const bool usingAux = (!m_albedoBuffer.empty() && albedoInput) ||
                              (!m_normalBuffer.empty() && normalInput);
        oidnSetFilterBool(filter, "cleanAux", usingAux);

        // STEP 5: Execute filter
        oidnCommitFilter(filter);

        // Check for commit errors
        const char* errorMsg = nullptr;
        OIDNError error = oidnGetDeviceError(device, &errorMsg);
        if (error != OIDN_ERROR_NONE) {
            m_lastError = std::string("OIDN filter commit failed");
            if (errorMsg) {
                m_lastError += std::string(": ") + errorMsg;
            }
            return false;
        }

        // Execute denoising
        oidnExecuteFilter(filter);

        // Check for execution errors
        error = oidnGetDeviceError(device, &errorMsg);
        if (error != OIDN_ERROR_NONE) {
            m_lastError = std::string("OIDN filter execution failed");
            if (errorMsg) {
                m_lastError += std::string(": ") + errorMsg;
            }
            return false;
        }

        const size_t pixelCount = colorWidth * colorHeight;
        for (size_t i = 0; i < pixelCount; ++i) {
            m_outputBuffer[i * 4 + 3] = m_colorBuffer[i * 4 + 3];
        }

        // STEP 6: Writeback denoised output to GPU texture
        if (!writeMetalTextureFromCPU(colorOutput,
                                      m_commandQueue,
                                      m_writebackBuffer,
                                      m_writebackBufferBytes,
                                      m_outputBuffer,
                                      m_lastError)) {
            m_lastError = std::string("Failed to writeback denoised texture: ") + m_lastError;
            return false;
        }

        NSLog(@"[OIDN] Denoising completed successfully for %ux%u image", (unsigned)colorWidth, (unsigned)colorHeight);
        m_lastError = "";
        return true;

    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "OIDN denoising exception: " << e.what();
        m_lastError = oss.str();
        return false;
    } catch (...) {
        m_lastError = "Unknown exception during denoising";
        return false;
    }
}

bool DenoiserContext::denoiseLinearBuffers(const float* colorInputRGB,
                                           const float* albedoInputRGB,
                                           const float* normalInputRGB,
                                           uint32_t width,
                                           uint32_t height,
                                           std::vector<float>& outDenoisedRGB,
                                           FilterType filterType,
                                           bool hdr) {
    if (!isReady()) {
        m_lastError = "Denoiser not ready (status: " + std::to_string(static_cast<int>(m_status)) + ")";
        return false;
    }
    if (!colorInputRGB || width == 0u || height == 0u) {
        m_lastError = "Invalid color buffer or dimensions";
        return false;
    }

    try {
        if (m_currentFilterType != filterType) {
            if (!createFilter(filterType)) {
                return false;
            }
        }

        const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
        const size_t rgbaCount = pixelCount * 4u;
        m_colorBuffer.assign(rgbaCount, 1.0f);
        for (size_t i = 0; i < pixelCount; ++i) {
            m_colorBuffer[i * 4 + 0] = colorInputRGB[i * 3 + 0];
            m_colorBuffer[i * 4 + 1] = colorInputRGB[i * 3 + 1];
            m_colorBuffer[i * 4 + 2] = colorInputRGB[i * 3 + 2];
        }

        m_albedoBuffer.clear();
        if (albedoInputRGB) {
            m_albedoBuffer.assign(rgbaCount, 1.0f);
            for (size_t i = 0; i < pixelCount; ++i) {
                m_albedoBuffer[i * 4 + 0] = albedoInputRGB[i * 3 + 0];
                m_albedoBuffer[i * 4 + 1] = albedoInputRGB[i * 3 + 1];
                m_albedoBuffer[i * 4 + 2] = albedoInputRGB[i * 3 + 2];
            }
        }

        m_normalBuffer.clear();
        if (normalInputRGB) {
            m_normalBuffer.assign(rgbaCount, 1.0f);
            for (size_t i = 0; i < pixelCount; ++i) {
                m_normalBuffer[i * 4 + 0] = normalInputRGB[i * 3 + 0] * 2.0f - 1.0f;
                m_normalBuffer[i * 4 + 1] = normalInputRGB[i * 3 + 1] * 2.0f - 1.0f;
                m_normalBuffer[i * 4 + 2] = normalInputRGB[i * 3 + 2] * 2.0f - 1.0f;
            }
        }

        m_outputBuffer.assign(rgbaCount, 0.0f);
        OIDNDevice device = (OIDNDevice)m_device;
        OIDNFilter filter = (OIDNFilter)m_filter;

        const size_t pixelStride = sizeof(float) * 4u;
        const size_t rowStride = pixelStride * width;

        oidnSetSharedFilterImage(filter,
                                 "color",
                                 m_colorBuffer.data(),
                                 OIDN_FORMAT_FLOAT3,
                                 width,
                                 height,
                                 0,
                                 pixelStride,
                                 rowStride);
        if (!m_albedoBuffer.empty()) {
            oidnSetSharedFilterImage(filter,
                                     "albedo",
                                     m_albedoBuffer.data(),
                                     OIDN_FORMAT_FLOAT3,
                                     width,
                                     height,
                                     0,
                                     pixelStride,
                                     rowStride);
        }
        if (!m_normalBuffer.empty()) {
            oidnSetSharedFilterImage(filter,
                                     "normal",
                                     m_normalBuffer.data(),
                                     OIDN_FORMAT_FLOAT3,
                                     width,
                                     height,
                                     0,
                                     pixelStride,
                                     rowStride);
        }
        oidnSetSharedFilterImage(filter,
                                 "output",
                                 m_outputBuffer.data(),
                                 OIDN_FORMAT_FLOAT3,
                                 width,
                                 height,
                                 0,
                                 pixelStride,
                                 rowStride);

        oidnSetFilterBool(filter, "hdr", hdr);
        oidnSetFilterBool(filter, "srgb", false);
        oidnSetFilterBool(filter, "cleanAux", !m_albedoBuffer.empty() || !m_normalBuffer.empty());
        oidnCommitFilter(filter);

        const char* errorMsg = nullptr;
        OIDNError error = oidnGetDeviceError(device, &errorMsg);
        if (error != OIDN_ERROR_NONE) {
            m_lastError = std::string("OIDN filter commit failed");
            if (errorMsg) {
                m_lastError += std::string(": ") + errorMsg;
            }
            return false;
        }

        oidnExecuteFilter(filter);
        error = oidnGetDeviceError(device, &errorMsg);
        if (error != OIDN_ERROR_NONE) {
            m_lastError = std::string("OIDN filter execution failed");
            if (errorMsg) {
                m_lastError += std::string(": ") + errorMsg;
            }
            return false;
        }

        outDenoisedRGB.resize(pixelCount * 3u);
        for (size_t i = 0; i < pixelCount; ++i) {
            outDenoisedRGB[i * 3 + 0] = m_outputBuffer[i * 4 + 0];
            outDenoisedRGB[i * 3 + 1] = m_outputBuffer[i * 4 + 1];
            outDenoisedRGB[i * 3 + 2] = m_outputBuffer[i * 4 + 2];
        }

        m_lastError.clear();
        return true;
    } catch (const std::exception& e) {
        m_lastError = std::string("OIDN denoiseLinearBuffers exception: ") + e.what();
        return false;
    } catch (...) {
        m_lastError = "Unknown exception during buffer denoising";
        return false;
    }
}

void DenoiserContext::shutdown() {
    if (m_filter) {
        oidnReleaseFilter((OIDNFilter)m_filter);
        m_filter = nullptr;
    }
    if (m_device) {
        oidnReleaseDevice((OIDNDevice)m_device);
        m_device = nullptr;
    }

    // Release CPU buffers
    m_colorBuffer.clear();
    m_albedoBuffer.clear();
    m_normalBuffer.clear();
    m_outputBuffer.clear();

    m_status = DeviceStatus::Uninitialized;
    m_lastError = "Shut down";
    m_metalDevice = nullptr;
    m_commandQueue = nullptr;
    releaseMetalStagingBuffers();
}

void DenoiserContext::releaseMetalStagingBuffers() {
    m_colorReadbackBuffer = nullptr;
    m_albedoReadbackBuffer = nullptr;
    m_normalReadbackBuffer = nullptr;
    m_writebackBuffer = nullptr;
    m_colorReadbackBufferBytes = 0;
    m_albedoReadbackBufferBytes = 0;
    m_normalReadbackBufferBytes = 0;
    m_writebackBufferBytes = 0;
}

}  // namespace PathTracer
#else
namespace PathTracer {

DenoiserContext::DenoiserContext()
    : m_device(nullptr),
      m_filter(nullptr),
      m_status(DeviceStatus::Uninitialized),
      m_lastError("OIDN support is disabled in this build"),
      m_metalDevice(nullptr),
      m_commandQueue(nullptr),
      m_currentFilterType(FilterType::RT),
      m_colorBuffer(),
      m_albedoBuffer(),
      m_normalBuffer(),
      m_outputBuffer() {
}

DenoiserContext::~DenoiserContext() {
    shutdown();
}

bool DenoiserContext::initialize(MTLDeviceHandle metalDevice, MTLCommandQueueHandle commandQueue) {
    m_metalDevice = metalDevice;
    m_commandQueue = commandQueue;
    m_status = DeviceStatus::Failed;
    m_lastError = "OIDN support is disabled in this build";
    return false;
}

bool DenoiserContext::createFilter(FilterType type) {
    m_currentFilterType = type;
    m_lastError = "OIDN support is disabled in this build";
    return false;
}

bool DenoiserContext::denoise(MTLTextureHandle,
                              MTLTextureHandle,
                              MTLTextureHandle,
                              MTLTextureHandle,
                              FilterType) {
    m_status = DeviceStatus::Failed;
    m_lastError = "OIDN support is disabled in this build";
    return false;
}

bool DenoiserContext::denoiseLinearBuffers(const float*,
                                           const float*,
                                           const float*,
                                           uint32_t,
                                           uint32_t,
                                           std::vector<float>& outDenoisedRGB,
                                           FilterType,
                                           bool) {
    outDenoisedRGB.clear();
    m_status = DeviceStatus::Failed;
    m_lastError = "OIDN support is disabled in this build";
    return false;
}

bool DenoiserContext::updateFilterDimensions(uint32_t, uint32_t) {
    m_lastError = "OIDN support is disabled in this build";
    return false;
}

void DenoiserContext::shutdown() {
    m_device = nullptr;
    m_filter = nullptr;
    m_colorBuffer.clear();
    m_albedoBuffer.clear();
    m_normalBuffer.clear();
    m_outputBuffer.clear();
    m_status = DeviceStatus::Uninitialized;
    m_lastError = "OIDN support is disabled in this build";
    m_metalDevice = nullptr;
    m_commandQueue = nullptr;
    releaseMetalStagingBuffers();
}

void DenoiserContext::releaseMetalStagingBuffers() {
    m_colorReadbackBuffer = nullptr;
    m_albedoReadbackBuffer = nullptr;
    m_normalReadbackBuffer = nullptr;
    m_writebackBuffer = nullptr;
    m_colorReadbackBufferBytes = 0;
    m_albedoReadbackBufferBytes = 0;
    m_normalReadbackBufferBytes = 0;
    m_writebackBufferBytes = 0;
}

}  // namespace PathTracer
#endif
