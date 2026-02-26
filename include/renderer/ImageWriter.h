#pragma once

#include <cstdint>
#include <string>

namespace PathTracer {

enum class ImageFileFormat {
    EXR,
    PNG,
    PFM,
    PPM,
};

struct TonemapSettings {
    uint32_t tonemapMode = 1;       // 1=Linear, 2=ACES, 3=Reinhard, 4=Hable
    uint32_t acesVariant = 0;       // 0=ACES Fitted, 1=ACES Simple
    float exposure = 0.0f;          // Exposure in stops
    float reinhardWhitePoint = 1.5f;
};

/// Write an image to disk in the requested format.
/// For LDR formats (PNG/PPM) the image is tone mapped and gamma corrected.
/// For HDR formats (EXR/PFM) the input is written as linear HDR.
/// @param path Output file path (UTF-8 encoded)
/// @param format Image file format to write
/// @param linearRGB Pointer to RGB float data (linear) with size width * height * 3
/// @param width Image width in pixels
/// @param height Image height in pixels
/// @param tonemap Tone mapping parameters (used for LDR formats)
/// @param errorMessage Optional pointer to receive an error message on failure
/// @return true on success, false otherwise
bool WriteImage(const std::string& path,
                ImageFileFormat format,
                const float* linearRGB,
                uint32_t width,
                uint32_t height,
                const TonemapSettings& tonemap,
                std::string* errorMessage = nullptr);

/// Convert a lowercase format string ("exr", "png", etc.) to ImageFileFormat.
/// Returns true on success.
bool ParseImageFileFormat(const std::string& value, ImageFileFormat& outFormat);

/// Default file extension (without dot) for the given format.
const char* FormatExtension(ImageFileFormat format);

namespace ImageWriter {

/// Write linear RGBA float32 buffer to EXR (scanline, uncompressed).
bool WriteEXR(const char* path,
              const float* rgba,
              int width,
              int height,
              const char* colorspace = nullptr);

/// Write linear RGBA float32 buffer to EXR and optionally append a sample-count channel.
bool WriteEXR_Multilayer(const char* path,
                         const float* rgba,
                         int width,
                         int height,
                         const float* sampleCount = nullptr,
                         const char* colorspace = nullptr);

}  // namespace ImageWriter

}  // namespace PathTracer
