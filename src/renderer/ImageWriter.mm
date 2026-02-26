#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

#if __has_include(<UniformTypeIdentifiers/UniformTypeIdentifiers.h>)
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#define HAS_UNIFORM_TYPE_IDENTIFIERS 1
#else
#import <MobileCoreServices/MobileCoreServices.h>
#define HAS_UNIFORM_TYPE_IDENTIFIERS 0
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "renderer/ImageWriter.h"

namespace PathTracer {

namespace {

struct Vec3 {
    float x;
    float y;
    float z;

    Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    Vec3(float vx, float vy, float vz) : x(vx), y(vy), z(vz) {}
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline Vec3 operator*(const Vec3& a, const Vec3& b) {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline Vec3 operator*(const Vec3& a, float s) {
    return Vec3(a.x * s, a.y * s, a.z * s);
}

inline Vec3 operator*(float s, const Vec3& a) {
    return a * s;
}

inline Vec3 operator/(const Vec3& a, float s) {
    return Vec3(a.x / s, a.y / s, a.z / s);
}

inline Vec3 operator/(const Vec3& a, const Vec3& b) {
    return Vec3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 clampVec(const Vec3& v, float minValue, float maxValue) {
    auto clampScalar = [&](float value) {
        return std::max(minValue, std::min(maxValue, value));
    };
    return Vec3(clampScalar(v.x), clampScalar(v.y), clampScalar(v.z));
}

inline Vec3 applyMatrix(const float m[3][3], const Vec3& v) {
    Vec3 r;
    r.x = m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z;
    r.y = m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z;
    r.z = m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z;
    return r;
}

inline Vec3 ACESFitted(const Vec3& color) {
    static constexpr float inputMat[3][3] = {
        {0.59719f, 0.07600f, 0.02840f},
        {0.35458f, 0.90834f, 0.13383f},
        {0.04823f, 0.01566f, 0.83777f},
    };
    static constexpr float outputMat[3][3] = {
        {1.60475f, -0.10208f, -0.00327f},
        {-0.53108f, 1.10813f, -0.07276f},
        {-0.07367f, -0.00605f, 1.07602f},
    };

    Vec3 c = applyMatrix(inputMat, color);
    Vec3 a = c * (c + Vec3(0.0245786f, 0.0245786f, 0.0245786f)) - Vec3(0.000090537f, 0.000090537f, 0.000090537f);
    Vec3 b = c * (Vec3(0.983729f, 0.983729f, 0.983729f) * c + Vec3(0.4329510f, 0.4329510f, 0.4329510f)) + Vec3(0.238081f, 0.238081f, 0.238081f);
    c = Vec3(a.x / b.x, a.y / b.y, a.z / b.z);
    c = applyMatrix(outputMat, c);
    return clampVec(c, 0.0f, 1.0f);
}

inline Vec3 ACESSimple(const Vec3& color) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    Vec3 numerator = color * (a * color + Vec3(b, b, b));
    Vec3 denominator = color * (c * color + Vec3(d, d, d)) + Vec3(e, e, e);
    return clampVec(Vec3(numerator.x / denominator.x,
                         numerator.y / denominator.y,
                         numerator.z / denominator.z), 0.0f, 1.0f);
}

inline Vec3 tonemapReinhard(const Vec3& c, float whitePoint) {
    const Vec3 lumWeights(0.2126f, 0.7152f, 0.0722f);
    float L = dot(c, lumWeights);
    float denom = 1.0f + L / std::max(whitePoint, 1e-4f);
    return clampVec(c / denom, 0.0f, 1.0f);
}

inline Vec3 tonemapHable(const Vec3& color) {
    const float A = 0.15f;
    const float B = 0.50f;
    const float C = 0.10f;
    const float D = 0.20f;
    const float E = 0.02f;
    const float F = 0.30f;
    const float W = 11.2f;

    Vec3 numerator = (color * (A * color + Vec3(B, B, B))) + C * color + Vec3(D, D, D);
    Vec3 denominator = (color * (A * color + Vec3(B, B, B))) + E * color + Vec3(F, F, F);
    Vec3 mapped = numerator / denominator - Vec3(D / F, D / F, D / F);
    float white = ((W * (A * W + B)) + C * W + D) / ((W * (A * W + B)) + E * W + F) - D / F;
    return clampVec(mapped / white, 0.0f, 1.0f);
}

inline Vec3 applyTonemap(const Vec3& linearColor, const TonemapSettings& tonemap) {
    Vec3 color = linearColor * std::pow(2.0f, tonemap.exposure);

    switch (tonemap.tonemapMode) {
        case 2:
            color = (tonemap.acesVariant == 0) ? ACESFitted(color) : ACESSimple(color);
            break;
        case 3:
            color = tonemapReinhard(color, tonemap.reinhardWhitePoint);
            break;
        case 4:
            color = tonemapHable(color);
            break;
        default:
            color = clampVec(color, 0.0f, 1.0f);
            break;
    }

    const float gamma = 1.0f / 2.2f;
    color.x = std::pow(std::max(color.x, 0.0f), gamma);
    color.y = std::pow(std::max(color.y, 0.0f), gamma);
    color.z = std::pow(std::max(color.z, 0.0f), gamma);
    return clampVec(color, 0.0f, 1.0f);
}

bool WritePPM(const std::string& path,
              const float* linearRGB,
              uint32_t width,
              uint32_t height,
              const TonemapSettings& tonemap,
              std::string* error) {
    std::vector<uint8_t> ldr(width * height * 3);
    for (uint32_t i = 0; i < width * height; ++i) {
        Vec3 color(linearRGB[3 * i + 0], linearRGB[3 * i + 1], linearRGB[3 * i + 2]);
        color = applyTonemap(color, tonemap);
        ldr[3 * i + 0] = static_cast<uint8_t>(std::clamp(std::lround(color.x * 255.0f), 0l, 255l));
        ldr[3 * i + 1] = static_cast<uint8_t>(std::clamp(std::lround(color.y * 255.0f), 0l, 255l));
        ldr[3 * i + 2] = static_cast<uint8_t>(std::clamp(std::lround(color.z * 255.0f), 0l, 255l));
    }

    FILE* file = fopen(path.c_str(), "wb");
    if (!file) {
        if (error) {
            *error = "Failed to open output file: " + path;
        }
        return false;
    }

    fprintf(file, "P6\n%u %u\n255\n", width, height);
    fwrite(ldr.data(), 1, ldr.size(), file);
    fclose(file);
    return true;
}

bool WritePFM(const std::string& path,
              const float* linearRGB,
              uint32_t width,
              uint32_t height,
              std::string* error) {
    FILE* file = fopen(path.c_str(), "wb");
    if (!file) {
        if (error) {
            *error = "Failed to open output file: " + path;
        }
        return false;
    }

    fprintf(file, "PF\n%u %u\n-1.0\n", width, height);
    const size_t rowSize = static_cast<size_t>(width) * 3;
    for (int y = static_cast<int>(height) - 1; y >= 0; --y) {
        const float* row = linearRGB + static_cast<size_t>(y) * rowSize;
        fwrite(row, sizeof(float), rowSize, file);
    }
    fclose(file);
    return true;
}

CFStringRef PNGUTI() {
#if HAS_UNIFORM_TYPE_IDENTIFIERS
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 110000 && defined(UTTypePNG)
    if (@available(macOS 11.0, *)) {
        return (__bridge CFStringRef)UTTypePNG.identifier;
    }
#endif
#endif
    return CFSTR("public.png");
}

enum class ChannelSourceType {
    Interleaved,
    Planar,
};

struct ChannelDescriptor {
    const char* name = nullptr;
    ChannelSourceType type = ChannelSourceType::Interleaved;
    uint32_t componentIndex = 0;
    const float* planarData = nullptr;
};

bool WriteScanlineEXR(const std::string& path,
                      const float* interleaved,
                      uint32_t interleavedStride,
                      uint32_t width,
                      uint32_t height,
                      const std::vector<ChannelDescriptor>& channels,
                      const char* colorspace,
                      std::string* error) {
    if (width == 0 || height == 0 || channels.empty()) {
        if (error) {
            *error = "Invalid EXR parameters";
        }
        return false;
    }

    FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) {
        if (error) {
            *error = "Failed to open output file: " + path;
        }
        return false;
    }

    auto writeBytes = [&](const void* ptr, size_t size) -> bool {
        if (std::fwrite(ptr, 1, size, file) != size) {
            if (error) {
                *error = "Failed writing to EXR file";
            }
            return false;
        }
        return true;
    };

    auto writeUint32 = [&](uint32_t value) -> bool {
        return writeBytes(&value, sizeof(uint32_t));
    };

    auto writeUint64 = [&](uint64_t value) -> bool {
        return writeBytes(&value, sizeof(uint64_t));
    };

    auto writeInt32 = [&](int32_t value) -> bool {
        return writeBytes(&value, sizeof(int32_t));
    };

    auto writeFloat = [&](float value) -> bool {
        return writeBytes(&value, sizeof(float));
    };

    auto writeCStr = [&](const char* str) -> bool {
        size_t len = std::strlen(str) + 1;
        return writeBytes(str, len);
    };

    // Magic number and version (scanline image, no additional features)
    if (!writeUint32(20000630u) || !writeUint32(2u)) {
        std::fclose(file);
        return false;
    }

    auto beginAttribute = [&](const char* name, const char* type, uint32_t size) -> bool {
        return writeCStr(name) && writeCStr(type) && writeUint32(size);
    };

    auto writeChannelsAttribute = [&]() -> bool {
        std::vector<uint8_t> payload;
        auto append = [&](const void* data, size_t sz) {
            const uint8_t* bytes = static_cast<const uint8_t*>(data);
            payload.insert(payload.end(), bytes, bytes + sz);
        };
        auto appendCStr = [&](const char* str) {
            append(str, std::strlen(str) + 1);
        };

        for (const ChannelDescriptor& channel : channels) {
            appendCStr(channel.name ? channel.name : "");
            int32_t pixelType = 2;  // FLOAT
            append(&pixelType, sizeof(int32_t));
            unsigned char pLinear = 0;
            append(&pLinear, sizeof(unsigned char));
            unsigned char reserved[3] = {0, 0, 0};
            append(reserved, sizeof(reserved));
            int32_t sampling = 1;
            append(&sampling, sizeof(int32_t));
            append(&sampling, sizeof(int32_t));
        }
        unsigned char terminator = 0;
        append(&terminator, sizeof(unsigned char));

        return beginAttribute("channels", "chlist", static_cast<uint32_t>(payload.size())) &&
               writeBytes(payload.data(), payload.size());
    };

    if (!writeChannelsAttribute()) {
        std::fclose(file);
        return false;
    }

    unsigned char compression = 0;  // NO_COMPRESSION
    if (!beginAttribute("compression", "compression", 1) ||
        !writeBytes(&compression, sizeof(unsigned char))) {
        std::fclose(file);
        return false;
    }

    auto writeBox2i = [&](const char* name) -> bool {
        int32_t values[4] = {0, 0, static_cast<int32_t>(width) - 1, static_cast<int32_t>(height) - 1};
        return beginAttribute(name, "box2i", sizeof(values)) && writeBytes(values, sizeof(values));
    };

    if (!writeBox2i("dataWindow") || !writeBox2i("displayWindow")) {
        std::fclose(file);
        return false;
    }

    float pixelAspect = 1.0f;
    if (!beginAttribute("pixelAspectRatio", "float", sizeof(pixelAspect)) || !writeFloat(pixelAspect)) {
        std::fclose(file);
        return false;
    }

    float screenCenter[2] = {0.0f, 0.0f};
    if (!beginAttribute("screenWindowCenter", "v2f", sizeof(screenCenter)) ||
        !writeBytes(screenCenter, sizeof(screenCenter))) {
        std::fclose(file);
        return false;
    }

    float screenWidth = 1.0f;
    if (!beginAttribute("screenWindowWidth", "float", sizeof(screenWidth)) || !writeFloat(screenWidth)) {
        std::fclose(file);
        return false;
    }

    unsigned char lineOrder = 0;  // INCREASING_Y
    if (!beginAttribute("lineOrder", "lineOrder", 1) || !writeBytes(&lineOrder, sizeof(unsigned char))) {
        std::fclose(file);
        return false;
    }

    if (colorspace && colorspace[0] != '\0') {
        uint32_t size = static_cast<uint32_t>(std::strlen(colorspace) + 1);
        if (!beginAttribute("colorspace", "string", size) ||
            !writeBytes(colorspace, size)) {
            std::fclose(file);
            return false;
        }
    }

    // End of header (empty attribute name)
    unsigned char headerEnd = 0;
    if (!writeBytes(&headerEnd, sizeof(unsigned char))) {
        std::fclose(file);
        return false;
    }

    long headerEndPos = std::ftell(file);
    if (headerEndPos < 0) {
        if (error) {
            *error = "ftell failed";
        }
        std::fclose(file);
        return false;
    }

    const uint64_t channelCount = static_cast<uint64_t>(channels.size());
    const uint64_t blockSize = 8ull + static_cast<uint64_t>(width) * channelCount * sizeof(float);
    std::vector<uint64_t> lineOffsets(height);
    uint64_t nextOffset = static_cast<uint64_t>(headerEndPos) + static_cast<uint64_t>(height) * sizeof(uint64_t);
    for (uint32_t y = 0; y < height; ++y) {
        lineOffsets[y] = nextOffset;
        nextOffset += blockSize;
    }

    for (uint64_t offset : lineOffsets) {
        if (!writeUint64(offset)) {
            std::fclose(file);
            return false;
        }
    }

    std::vector<float> channelBuffer(width);
    auto writeChannel = [&](const ChannelDescriptor& channel, uint32_t y) -> bool {
        for (uint32_t x = 0; x < width; ++x) {
            size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            float value = 0.0f;
            if (channel.type == ChannelSourceType::Interleaved) {
                if (!interleaved) {
                    return false;
                }
                size_t channelOffset = idx * static_cast<size_t>(interleavedStride) +
                                       static_cast<size_t>(channel.componentIndex);
                value = interleaved[channelOffset];
            } else {
                if (!channel.planarData) {
                    return false;
                }
                value = channel.planarData[idx];
            }
            channelBuffer[x] = value;
        }
        return writeBytes(channelBuffer.data(), static_cast<size_t>(width) * sizeof(float));
    };

    for (uint32_t y = 0; y < height; ++y) {
        if (!writeInt32(static_cast<int32_t>(y))) {
            std::fclose(file);
            return false;
        }
        uint32_t packedSize = width * static_cast<uint32_t>(channelCount) * sizeof(float);
        if (!writeUint32(packedSize)) {
            std::fclose(file);
            return false;
        }

        for (const ChannelDescriptor& channel : channels) {
            if (!writeChannel(channel, y)) {
                std::fclose(file);
                return false;
            }
        }
    }

    std::fclose(file);
    return true;
}

bool WriteEXR(const std::string& path,
              const float* linearRGB,
              uint32_t width,
              uint32_t height,
              const char* colorspace,
              std::string* error) {
    std::vector<ChannelDescriptor> channels = {
        {"B", ChannelSourceType::Interleaved, 2, nullptr},
        {"G", ChannelSourceType::Interleaved, 1, nullptr},
        {"R", ChannelSourceType::Interleaved, 0, nullptr},
    };
    return WriteScanlineEXR(path, linearRGB, 3, width, height, channels, colorspace, error);
}

bool WritePNG(const std::string& path,
              const float* linearRGB,
              uint32_t width,
              uint32_t height,
              const TonemapSettings& tonemap,
              std::string* error) {
    std::vector<uint8_t> ldr(width * height * 4);
    for (uint32_t i = 0; i < width * height; ++i) {
        Vec3 color(linearRGB[3 * i + 0], linearRGB[3 * i + 1], linearRGB[3 * i + 2]);
        color = applyTonemap(color, tonemap);
        ldr[4 * i + 0] = static_cast<uint8_t>(std::clamp(std::lround(color.x * 255.0f), 0l, 255l));
        ldr[4 * i + 1] = static_cast<uint8_t>(std::clamp(std::lround(color.y * 255.0f), 0l, 255l));
        ldr[4 * i + 2] = static_cast<uint8_t>(std::clamp(std::lround(color.z * 255.0f), 0l, 255l));
        ldr[4 * i + 3] = 255;
    }

    CFURLRef url = (__bridge CFURLRef)[NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    if (!url) {
        if (error) {
            *error = "Failed to create URL for output file";
        }
        return false;
    }

    CGColorSpaceRef colorSpace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    if (!colorSpace) {
        if (error) {
            *error = "Failed to create color space";
        }
        return false;
    }

    const size_t bytesPerRow = static_cast<size_t>(width) * 4;
    CGDataProviderRef provider = CGDataProviderCreateWithData(nullptr,
                                                              ldr.data(),
                                                              bytesPerRow * height,
                                                              nullptr);
    if (!provider) {
        CGColorSpaceRelease(colorSpace);
        if (error) {
            *error = "Failed to create data provider";
        }
        return false;
    }

    CGBitmapInfo bitmapInfo = kCGImageByteOrder32Big | kCGImageAlphaPremultipliedLast;
    CGImageRef image = CGImageCreate(width,
                                     height,
                                     8,
                                     32,
                                     bytesPerRow,
                                     colorSpace,
                                     bitmapInfo,
                                     provider,
                                     nullptr,
                                     false,
                                     kCGRenderingIntentDefault);
    CGColorSpaceRelease(colorSpace);
    CGDataProviderRelease(provider);

    if (!image) {
        if (error) {
            *error = "Failed to create CGImage for PNG";
        }
        return false;
    }

    CGImageDestinationRef destination = CGImageDestinationCreateWithURL(url, PNGUTI(), 1, nullptr);
    if (!destination) {
        CGImageRelease(image);
        if (error) {
            *error = "Failed to create PNG destination";
        }
        return false;
    }

    CGImageDestinationAddImage(destination, image, nullptr);
    bool success = CGImageDestinationFinalize(destination);
    CFRelease(destination);
    CGImageRelease(image);

    if (!success && error) {
        *error = "Failed to finalize PNG file";
    }
    return success;
}

}  // namespace

bool ParseImageFileFormat(const std::string& value, ImageFileFormat& outFormat) {
    std::string lower;
    lower.reserve(value.size());
    for (char c : value) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    if (lower == "exr") {
        outFormat = ImageFileFormat::EXR;
        return true;
    }
    if (lower == "png") {
        outFormat = ImageFileFormat::PNG;
        return true;
    }
    if (lower == "pfm") {
        outFormat = ImageFileFormat::PFM;
        return true;
    }
    if (lower == "ppm") {
        outFormat = ImageFileFormat::PPM;
        return true;
    }
    return false;
}

const char* FormatExtension(ImageFileFormat format) {
    switch (format) {
        case ImageFileFormat::EXR:
            return "exr";
        case ImageFileFormat::PNG:
            return "png";
        case ImageFileFormat::PFM:
            return "pfm";
        case ImageFileFormat::PPM:
        default:
            return "ppm";
    }
}

bool WriteImage(const std::string& path,
                ImageFileFormat format,
                const float* linearRGB,
                uint32_t width,
                uint32_t height,
                const TonemapSettings& tonemap,
                std::string* errorMessage) {
    switch (format) {
        case ImageFileFormat::EXR:
            return WriteEXR(path, linearRGB, width, height, nullptr, errorMessage);
        case ImageFileFormat::PFM:
            return WritePFM(path, linearRGB, width, height, errorMessage);
        case ImageFileFormat::PNG:
            return WritePNG(path, linearRGB, width, height, tonemap, errorMessage);
        case ImageFileFormat::PPM:
            return WritePPM(path, linearRGB, width, height, tonemap, errorMessage);
    }
    return false;
}

}  // namespace PathTracer

namespace PathTracer::ImageWriter {

bool WriteEXR(const char* path,
              const float* rgba,
              int width,
              int height,
              const char* colorspace) {
    if (!path || !rgba || width <= 0 || height <= 0) {
        return false;
    }
    std::vector<ChannelDescriptor> channels = {
        {"B", ChannelSourceType::Interleaved, 2, nullptr},
        {"G", ChannelSourceType::Interleaved, 1, nullptr},
        {"R", ChannelSourceType::Interleaved, 0, nullptr},
        {"A", ChannelSourceType::Interleaved, 3, nullptr},
    };
    return WriteScanlineEXR(std::string(path),
                            rgba,
                            4,
                            static_cast<uint32_t>(width),
                            static_cast<uint32_t>(height),
                            channels,
                            colorspace,
                            nullptr);
}

bool WriteEXR_Multilayer(const char* path,
                         const float* rgba,
                         int width,
                         int height,
                         const float* sampleCount,
                         const char* colorspace) {
    if (!sampleCount) {
        return WriteEXR(path, rgba, width, height, colorspace);
    }
    if (!path || !rgba || width <= 0 || height <= 0) {
        return false;
    }
    std::vector<ChannelDescriptor> channels = {
        {"B", ChannelSourceType::Interleaved, 2, nullptr},
        {"G", ChannelSourceType::Interleaved, 1, nullptr},
        {"R", ChannelSourceType::Interleaved, 0, nullptr},
        {"A", ChannelSourceType::Interleaved, 3, nullptr},
        {"SAMPLES", ChannelSourceType::Planar, 0, sampleCount},
    };
    return WriteScanlineEXR(std::string(path),
                            rgba,
                            4,
                            static_cast<uint32_t>(width),
                            static_cast<uint32_t>(height),
                            channels,
                            colorspace,
                            nullptr);
}

}  // namespace PathTracer::ImageWriter
