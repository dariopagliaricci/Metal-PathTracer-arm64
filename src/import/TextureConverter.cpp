#include "import/TextureConverter.h"

#include "shared/Ktx2Texture.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_TGA
#define STBI_ONLY_BMP
#include "stb_image.h"

namespace fs = std::filesystem;

namespace PathTracer::Import {

namespace {

constexpr const char* kTranscodeTool = "pathtracer_ktx2";
constexpr const char* kDdsAuditComment =
    "Bistro DDS audit: DXT1/BC1, DXT5/BC3, ATI2/BC5";

std::string ToLowerAscii(std::string value) {
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string SanitizeName(std::string value) {
    std::string out;
    out.reserve(value.size());
    for (unsigned char ch : value) {
        if (std::isalnum(ch)) {
            out.push_back(static_cast<char>(std::tolower(ch)));
        } else if (ch == '_' || ch == '-' || ch == '.') {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('_');
        }
    }
    while (!out.empty() && out.front() == '_') {
        out.erase(out.begin());
    }
    while (!out.empty() && out.back() == '_') {
        out.pop_back();
    }
    return out.empty() ? "texture" : out;
}

std::vector<uint8_t> ReadFileBytes(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }
    stream.seekg(0, std::ios::end);
    const std::streamsize size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    if (size < 0) {
        throw std::runtime_error("Failed to read file size: " + path.string());
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (size > 0) {
        stream.read(reinterpret_cast<char*>(bytes.data()), size);
        if (!stream) {
            throw std::runtime_error("Failed to read file: " + path.string());
        }
    }
    return bytes;
}

void WriteFileBytes(const fs::path& path, const std::vector<uint8_t>& bytes) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }
    if (!bytes.empty()) {
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    if (!stream) {
        throw std::runtime_error("Failed to write output file: " + path.string());
    }
}

std::string Sha256Hex(const std::vector<uint8_t>& bytes) {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(bytes.empty() ? nullptr : bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(digest.size() * 2u, '0');
    for (size_t i = 0; i < digest.size(); ++i) {
        out[i * 2u] = kHex[digest[i] >> 4u];
        out[i * 2u + 1u] = kHex[digest[i] & 0x0Fu];
    }
    return out;
}

std::string ShortPathHash(const fs::path& path) {
    const std::string normalized = ToLowerAscii(path.lexically_normal().generic_string());
    const std::vector<uint8_t> bytes(normalized.begin(), normalized.end());
    return Sha256Hex(bytes).substr(0, 8);
}

uint32_t ReadU32(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8u) |
           (static_cast<uint32_t>(data[2]) << 16u) |
           (static_cast<uint32_t>(data[3]) << 24u);
}

uint16_t ReadU16(const uint8_t* data) {
    return static_cast<uint16_t>(data[0]) |
           static_cast<uint16_t>(static_cast<uint16_t>(data[1]) << 8u);
}

uint8_t Expand5(uint8_t value) {
    return static_cast<uint8_t>((value * 255u + 15u) / 31u);
}

uint8_t Expand6(uint8_t value) {
    return static_cast<uint8_t>((value * 255u + 31u) / 63u);
}

struct Color32 {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    uint8_t a = 255;
};

Color32 DecodeRgb565(uint16_t value) {
    Color32 color;
    color.r = Expand5(static_cast<uint8_t>((value >> 11u) & 0x1Fu));
    color.g = Expand6(static_cast<uint8_t>((value >> 5u) & 0x3Fu));
    color.b = Expand5(static_cast<uint8_t>(value & 0x1Fu));
    return color;
}

void WritePixel(std::vector<uint8_t>& rgba,
                uint32_t width,
                uint32_t x,
                uint32_t y,
                const Color32& color) {
    const size_t offset = (static_cast<size_t>(y) * width + x) * 4u;
    rgba[offset + 0] = color.r;
    rgba[offset + 1] = color.g;
    rgba[offset + 2] = color.b;
    rgba[offset + 3] = color.a;
}

void DecompressBc1Block(const uint8_t* block,
                        uint32_t width,
                        uint32_t height,
                        uint32_t blockX,
                        uint32_t blockY,
                        bool allowTransparentIndex,
                        std::vector<uint8_t>& rgba) {
    Color32 colors[4];
    const uint16_t c0 = ReadU16(block + 0);
    const uint16_t c1 = ReadU16(block + 2);
    colors[0] = DecodeRgb565(c0);
    colors[1] = DecodeRgb565(c1);
    if (!allowTransparentIndex || c0 > c1) {
        colors[2] = {
            static_cast<uint8_t>((2u * colors[0].r + colors[1].r) / 3u),
            static_cast<uint8_t>((2u * colors[0].g + colors[1].g) / 3u),
            static_cast<uint8_t>((2u * colors[0].b + colors[1].b) / 3u),
            255
        };
        colors[3] = {
            static_cast<uint8_t>((colors[0].r + 2u * colors[1].r) / 3u),
            static_cast<uint8_t>((colors[0].g + 2u * colors[1].g) / 3u),
            static_cast<uint8_t>((colors[0].b + 2u * colors[1].b) / 3u),
            255
        };
    } else {
        colors[2] = {
            static_cast<uint8_t>((colors[0].r + colors[1].r) / 2u),
            static_cast<uint8_t>((colors[0].g + colors[1].g) / 2u),
            static_cast<uint8_t>((colors[0].b + colors[1].b) / 2u),
            255
        };
        colors[3] = {0, 0, 0, 0};
    }

    const uint32_t indices = ReadU32(block + 4);
    for (uint32_t py = 0; py < 4; ++py) {
        for (uint32_t px = 0; px < 4; ++px) {
            const uint32_t x = blockX * 4u + px;
            const uint32_t y = blockY * 4u + py;
            if (x >= width || y >= height) {
                continue;
            }
            const uint32_t index = (indices >> (2u * (py * 4u + px))) & 0x3u;
            WritePixel(rgba, width, x, y, colors[index]);
        }
    }
}

std::array<uint8_t, 8> DecodeBc4Values(const uint8_t* block) {
    std::array<uint8_t, 8> values{};
    values[0] = block[0];
    values[1] = block[1];
    if (values[0] > values[1]) {
        for (int i = 1; i <= 6; ++i) {
            values[static_cast<size_t>(i) + 1u] =
                static_cast<uint8_t>(((7 - i) * values[0] + i * values[1]) / 7);
        }
    } else {
        for (int i = 1; i <= 4; ++i) {
            values[static_cast<size_t>(i) + 1u] =
                static_cast<uint8_t>(((5 - i) * values[0] + i * values[1]) / 5);
        }
        values[6] = 0;
        values[7] = 255;
    }
    return values;
}

uint64_t DecodeBc4IndexBits(const uint8_t* block) {
    uint64_t bits = 0u;
    for (int i = 0; i < 6; ++i) {
        bits |= static_cast<uint64_t>(block[2 + i]) << (8u * static_cast<uint32_t>(i));
    }
    return bits;
}

void DecompressDxt5Block(const uint8_t* block,
                         uint32_t width,
                         uint32_t height,
                         uint32_t blockX,
                         uint32_t blockY,
                         std::vector<uint8_t>& rgba) {
    const auto alphaValues = DecodeBc4Values(block);
    const uint64_t alphaBits = DecodeBc4IndexBits(block);

    Color32 colors[4];
    const uint16_t c0 = ReadU16(block + 8);
    const uint16_t c1 = ReadU16(block + 10);
    colors[0] = DecodeRgb565(c0);
    colors[1] = DecodeRgb565(c1);
    colors[2] = {
        static_cast<uint8_t>((2u * colors[0].r + colors[1].r) / 3u),
        static_cast<uint8_t>((2u * colors[0].g + colors[1].g) / 3u),
        static_cast<uint8_t>((2u * colors[0].b + colors[1].b) / 3u),
        255
    };
    colors[3] = {
        static_cast<uint8_t>((colors[0].r + 2u * colors[1].r) / 3u),
        static_cast<uint8_t>((colors[0].g + 2u * colors[1].g) / 3u),
        static_cast<uint8_t>((colors[0].b + 2u * colors[1].b) / 3u),
        255
    };

    const uint32_t colorBits = ReadU32(block + 12);
    for (uint32_t py = 0; py < 4; ++py) {
        for (uint32_t px = 0; px < 4; ++px) {
            const uint32_t x = blockX * 4u + px;
            const uint32_t y = blockY * 4u + py;
            if (x >= width || y >= height) {
                continue;
            }
            const uint32_t pixelIndex = py * 4u + px;
            Color32 color = colors[(colorBits >> (2u * pixelIndex)) & 0x3u];
            color.a = alphaValues[(alphaBits >> (3u * pixelIndex)) & 0x7u];
            WritePixel(rgba, width, x, y, color);
        }
    }
}

void DecompressBc5Block(const uint8_t* block,
                        uint32_t width,
                        uint32_t height,
                        uint32_t blockX,
                        uint32_t blockY,
                        std::vector<uint8_t>& rgba) {
    const auto redValues = DecodeBc4Values(block + 0);
    const auto greenValues = DecodeBc4Values(block + 8);
    const uint64_t redBits = DecodeBc4IndexBits(block + 0);
    const uint64_t greenBits = DecodeBc4IndexBits(block + 8);

    for (uint32_t py = 0; py < 4; ++py) {
        for (uint32_t px = 0; px < 4; ++px) {
            const uint32_t x = blockX * 4u + px;
            const uint32_t y = blockY * 4u + py;
            if (x >= width || y >= height) {
                continue;
            }
            const uint32_t pixelIndex = py * 4u + px;
            const uint8_t red = redValues[(redBits >> (3u * pixelIndex)) & 0x7u];
            const uint8_t green = greenValues[(greenBits >> (3u * pixelIndex)) & 0x7u];
            const float nx = static_cast<float>(red) / 127.5f - 1.0f;
            const float ny = static_cast<float>(green) / 127.5f - 1.0f;
            const float nz = std::sqrt(std::max(0.0f, 1.0f - nx * nx - ny * ny));
            Color32 color;
            color.r = red;
            color.g = green;
            color.b = static_cast<uint8_t>(std::clamp((nz * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
            color.a = 255;
            WritePixel(rgba, width, x, y, color);
        }
    }
}

bool DecodeDds(const std::vector<uint8_t>& bytes,
               uint32_t& width,
               uint32_t& height,
               std::vector<uint8_t>& rgba) {
    (void)kDdsAuditComment;
    if (bytes.size() < 128u || std::memcmp(bytes.data(), "DDS ", 4) != 0) {
        return false;
    }

    height = ReadU32(bytes.data() + 12);
    width = ReadU32(bytes.data() + 16);
    if (width == 0 || height == 0) {
        return false;
    }

    std::string format;
    size_t offset = 128u;
    const std::string fourcc(reinterpret_cast<const char*>(bytes.data() + 84), 4);
    if (fourcc == "DXT1") {
        format = "DXT1";
    } else if (fourcc == "DXT5") {
        format = "DXT5";
    } else if (fourcc == "ATI2") {
        format = "ATI2";
    } else if (fourcc == "DX10") {
        if (bytes.size() < 148u) {
            return false;
        }
        offset = 148u;
        const uint32_t dxgiFormat = ReadU32(bytes.data() + 128);
        switch (dxgiFormat) {
            case 71u:
            case 72u:
                format = "DXT1";
                break;
            case 77u:
            case 78u:
                format = "DXT5";
                break;
            case 83u:
            case 84u:
                format = "ATI2";
                break;
            default:
                return false;
        }
    } else {
        return false;
    }

    const uint32_t blockWidth = (width + 3u) / 4u;
    const uint32_t blockHeight = (height + 3u) / 4u;
    const size_t blockBytes = (format == "DXT1") ? 8u : 16u;
    const uint64_t requiredBytes = static_cast<uint64_t>(blockWidth) * static_cast<uint64_t>(blockHeight) * blockBytes;
    if (offset + requiredBytes > bytes.size()) {
        return false;
    }

    rgba.assign(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u, 0);
    const uint8_t* blocks = bytes.data() + offset;
    for (uint32_t by = 0; by < blockHeight; ++by) {
        for (uint32_t bx = 0; bx < blockWidth; ++bx) {
            const uint8_t* block = blocks + (static_cast<size_t>(by) * blockWidth + bx) * blockBytes;
            if (format == "DXT1") {
                DecompressBc1Block(block, width, height, bx, by, true, rgba);
            } else if (format == "DXT5") {
                DecompressDxt5Block(block, width, height, bx, by, rgba);
            } else {
                DecompressBc5Block(block, width, height, bx, by, rgba);
            }
        }
    }
    return true;
}

bool LoadStandardImage(const fs::path& path,
                       uint32_t& width,
                       uint32_t& height,
                       std::vector<uint8_t>& rgba) {
    int loadedWidth = 0;
    int loadedHeight = 0;
    int channels = 0;
    stbi_uc* pixels = stbi_load(path.string().c_str(), &loadedWidth, &loadedHeight, &channels, 4);
    if (!pixels) {
        return false;
    }
    width = static_cast<uint32_t>(loadedWidth);
    height = static_cast<uint32_t>(loadedHeight);
    rgba.assign(pixels, pixels + static_cast<size_t>(width) * static_cast<size_t>(height) * 4u);
    stbi_image_free(pixels);
    return true;
}

std::string SourceFormatFromExtension(const fs::path& path) {
    std::string extension = ToLowerAscii(path.extension().string());
    if (!extension.empty() && extension.front() == '.') {
        extension.erase(extension.begin());
    }
    return extension.empty() ? "bin" : extension;
}

void AnalyzeAlphaChannel(const std::vector<uint8_t>& rgba, bool& hasAlpha, bool& binaryAlpha) {
    hasAlpha = false;
    binaryAlpha = true;
    for (size_t i = 3; i < rgba.size(); i += 4u) {
        const uint8_t alpha = rgba[i];
        if (alpha < 250u) {
            hasAlpha = true;
        }
        if (alpha > 8u && alpha < 247u) {
            binaryAlpha = false;
        }
    }
    if (!hasAlpha) {
        binaryAlpha = false;
    }
}

}  // namespace

DecodedTexture decodeTextureFile(const std::string& source_path) {
    const fs::path source = fs::absolute(source_path).lexically_normal();
    DecodedTexture result;
    result.source_format = SourceFormatFromExtension(source);
    const std::vector<uint8_t> sourceBytes = ReadFileBytes(source);
    if (result.source_format == "dds") {
        if (!DecodeDds(sourceBytes, result.width, result.height, result.rgba)) {
            throw std::runtime_error("Failed to decode DDS image: " + source.string());
        }
    } else if (result.source_format == "png" ||
               result.source_format == "jpg" ||
               result.source_format == "jpeg" ||
               result.source_format == "tga" ||
               result.source_format == "bmp") {
        if (!LoadStandardImage(source, result.width, result.height, result.rgba)) {
            throw std::runtime_error("Failed to decode image: " + source.string());
        }
    } else {
        throw std::runtime_error("Unsupported texture decode format: " + source.string());
    }
    AnalyzeAlphaChannel(result.rgba, result.has_alpha, result.binary_alpha);
    return result;
}

ConvertResult convertTexture(const std::string& source_path, const std::string& output_dir) {
    const fs::path source = fs::absolute(source_path).lexically_normal();
    const fs::path outputRoot = fs::absolute(output_dir).lexically_normal();
    std::error_code ec;
    fs::create_directories(outputRoot, ec);
    if (ec) {
        throw std::runtime_error("Failed to create output directory: " + outputRoot.string());
    }

    ConvertResult result;
    result.source_format = SourceFormatFromExtension(source);
    result.transcode_tool = kTranscodeTool;
    result.transcode_version = PATH_TRACER_IMPORT_VERSION;

    const std::string stem = SanitizeName(source.stem().string());
    const std::string suffix = ShortPathHash(source);
    const fs::path convertedPath = outputRoot / (stem + "_" + suffix + ".ktx2");

    std::vector<uint8_t> sourceBytes = ReadFileBytes(source);
    std::vector<uint8_t> rgba;
    if (result.source_format == "dds") {
        if (!DecodeDds(sourceBytes, result.width, result.height, rgba)) {
            const fs::path fallbackPath = outputRoot / (stem + "_" + suffix + source.extension().string());
            WriteFileBytes(fallbackPath, sourceBytes);
            result.output_path = fallbackPath.string();
            result.output_format = result.source_format;
            result.output_bytes = sourceBytes.size();
            result.output_sha256 = Sha256Hex(sourceBytes);
            result.fell_back_to_copy = true;
            std::fprintf(stderr,
                         "[TextureConverter] unsupported format %s, falling back to copy: %s\n",
                         result.source_format.c_str(),
                         source.filename().string().c_str());
            return result;
        }
    } else if (result.source_format == "png" ||
               result.source_format == "jpg" ||
               result.source_format == "jpeg" ||
               result.source_format == "tga" ||
               result.source_format == "bmp") {
        if (!LoadStandardImage(source, result.width, result.height, rgba)) {
            throw std::runtime_error("Failed to decode image: " + source.string());
        }
    } else {
        const fs::path fallbackPath = outputRoot / (stem + "_" + suffix + source.extension().string());
        WriteFileBytes(fallbackPath, sourceBytes);
        result.output_path = fallbackPath.string();
        result.output_format = result.source_format;
        result.output_bytes = sourceBytes.size();
        result.output_sha256 = Sha256Hex(sourceBytes);
        result.fell_back_to_copy = true;
        std::fprintf(stderr,
                     "[TextureConverter] unsupported format %s, falling back to copy: %s\n",
                     result.source_format.c_str(),
                     source.filename().string().c_str());
        return result;
    }

    AnalyzeAlphaChannel(rgba, result.has_alpha, result.binary_alpha);

    std::string error;
    if (!Ktx2::WriteRgba8File(convertedPath, result.width, result.height, false, rgba, error)) {
        throw std::runtime_error(error);
    }
    const std::vector<uint8_t> outputBytes = ReadFileBytes(convertedPath);
    result.output_path = convertedPath.string();
    result.output_format = "ktx2";
    result.output_bytes = outputBytes.size();
    result.output_sha256 = Sha256Hex(outputBytes);
    return result;
}

}  // namespace PathTracer::Import
