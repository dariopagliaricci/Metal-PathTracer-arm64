#include "import/TextureConverter.h"

#include "shared/DdsTexture.h"
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

// DDS sources (Zero Day, Bistro) are decoded with the shared decoder (shared/DdsTexture.h) that Embree also uses. Since
// PR #131 (2026-11-23) it applies the GPU's block rules (bit-replicated 565 endpoints, 6-bit palette weights), so the
// KTX2 an import writes holds the texels Metal would sample from the DDS itself. Imports made before that differ by up to
// 2-3 8-bit steps in colour and in a normal map's X and Y, and by up to 15 in a BC5 normal map's reconstructed Z, whose
// square root amplifies the X and Y change. Zero Day's installed packs keep their Nov 19 KTX2 files (older rules).
bool DecodeDds(const std::vector<uint8_t>& bytes,
               uint32_t& width,
               uint32_t& height,
               std::vector<uint8_t>& rgba) {
    (void)kDdsAuditComment;
    Dds::Header header;
    const bool parsed = Dds::ParseHeader(bytes, header);
    // The size is reported even for a format the importer cannot decode; it then copies the file.
    width = header.width;
    height = header.height;
    std::vector<Dds::Level> levels;
    if (!parsed || !Dds::DecodeLevels(bytes, header, 1u, /*reconstructBc5Z=*/true, levels)) {
        return false;
    }
    rgba = std::move(levels.front().rgba);
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
