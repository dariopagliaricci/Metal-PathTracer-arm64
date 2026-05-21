#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace PathTracer::Ktx2 {

struct ImageData {
    uint32_t width = 0;
    uint32_t height = 0;
    bool srgb = false;
    std::vector<uint8_t> rgba;
};

namespace detail {

inline uint32_t ReadU32(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8u) |
           (static_cast<uint32_t>(data[2]) << 16u) |
           (static_cast<uint32_t>(data[3]) << 24u);
}

inline uint64_t ReadU64(const uint8_t* data) {
    return static_cast<uint64_t>(ReadU32(data)) |
           (static_cast<uint64_t>(ReadU32(data + 4)) << 32u);
}

inline void AppendU32(std::vector<uint8_t>& bytes, uint32_t value) {
    bytes.push_back(static_cast<uint8_t>(value & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((value >> 8u) & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((value >> 16u) & 0xFFu));
    bytes.push_back(static_cast<uint8_t>((value >> 24u) & 0xFFu));
}

inline void AppendU64(std::vector<uint8_t>& bytes, uint64_t value) {
    AppendU32(bytes, static_cast<uint32_t>(value & 0xFFFFFFFFu));
    AppendU32(bytes, static_cast<uint32_t>((value >> 32u) & 0xFFFFFFFFu));
}

}  // namespace detail

inline constexpr uint8_t kIdentifier[12] = {
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A
};

inline constexpr uint32_t kVkFormatR8G8B8A8Unorm = 37u;
inline constexpr uint32_t kVkFormatR8G8B8A8Srgb = 43u;
inline constexpr uint32_t kHeaderSize = 80u;
inline constexpr uint32_t kLevelIndexSize = 24u;

inline bool ParseRgba8(const uint8_t* data, size_t size, ImageData& out, std::string& error) {
    if (!data || size < (kHeaderSize + kLevelIndexSize)) {
        error = "KTX2 payload is truncated";
        return false;
    }
    if (std::memcmp(data, kIdentifier, sizeof(kIdentifier)) != 0) {
        error = "KTX2 identifier mismatch";
        return false;
    }

    const uint32_t vkFormat = detail::ReadU32(data + 12);
    if (vkFormat != kVkFormatR8G8B8A8Unorm && vkFormat != kVkFormatR8G8B8A8Srgb) {
        error = "KTX2 vkFormat is unsupported";
        return false;
    }

    const uint32_t width = detail::ReadU32(data + 20);
    const uint32_t height = detail::ReadU32(data + 24);
    const uint32_t pixelDepth = detail::ReadU32(data + 28);
    const uint32_t layerCount = detail::ReadU32(data + 32);
    const uint32_t faceCount = detail::ReadU32(data + 36);
    const uint32_t levelCount = detail::ReadU32(data + 40);
    const uint64_t levelOffset = detail::ReadU64(data + kHeaderSize);
    const uint64_t levelLength = detail::ReadU64(data + kHeaderSize + 8);
    const uint64_t uncompressedLength = detail::ReadU64(data + kHeaderSize + 16);

    if (width == 0 || height == 0) {
        error = "KTX2 image dimensions are invalid";
        return false;
    }
    if (pixelDepth != 0 || layerCount != 0 || faceCount != 1 || levelCount != 1) {
        error = "KTX2 texture layout is unsupported";
        return false;
    }

    const uint64_t expectedLength = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 4u;
    if (levelLength != expectedLength || uncompressedLength != expectedLength) {
        error = "KTX2 payload size does not match RGBA8 image dimensions";
        return false;
    }
    if (levelOffset > size || levelLength > size || (levelOffset + levelLength) > size) {
        error = "KTX2 level data is out of range";
        return false;
    }

    out.width = width;
    out.height = height;
    out.srgb = (vkFormat == kVkFormatR8G8B8A8Srgb);
    out.rgba.assign(data + levelOffset, data + levelOffset + levelLength);
    return true;
}

inline bool ParseRgba8File(const std::filesystem::path& path, ImageData& out, std::string& error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        error = "Failed to open KTX2 file: " + path.string();
        return false;
    }
    stream.seekg(0, std::ios::end);
    const std::streamsize size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    if (size <= 0) {
        error = "KTX2 file is empty: " + path.string();
        return false;
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    stream.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!stream) {
        error = "Failed to read KTX2 file: " + path.string();
        return false;
    }
    return ParseRgba8(bytes.data(), bytes.size(), out, error);
}

inline bool WriteRgba8File(const std::filesystem::path& path,
                           uint32_t width,
                           uint32_t height,
                           bool srgb,
                           const std::vector<uint8_t>& rgba,
                           std::string& error) {
    const uint64_t expectedLength = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 4u;
    if (width == 0 || height == 0 || rgba.size() != expectedLength) {
        error = "KTX2 write failed: RGBA payload does not match image dimensions";
        return false;
    }

    std::vector<uint8_t> bytes;
    bytes.reserve(static_cast<size_t>(kHeaderSize + kLevelIndexSize + rgba.size()));
    bytes.insert(bytes.end(), std::begin(kIdentifier), std::end(kIdentifier));
    detail::AppendU32(bytes, srgb ? kVkFormatR8G8B8A8Srgb : kVkFormatR8G8B8A8Unorm);
    detail::AppendU32(bytes, 1u);
    detail::AppendU32(bytes, width);
    detail::AppendU32(bytes, height);
    detail::AppendU32(bytes, 0u);
    detail::AppendU32(bytes, 0u);
    detail::AppendU32(bytes, 1u);
    detail::AppendU32(bytes, 1u);
    detail::AppendU32(bytes, 0u);
    detail::AppendU32(bytes, 0u);
    detail::AppendU32(bytes, 0u);
    detail::AppendU32(bytes, 0u);
    detail::AppendU32(bytes, 0u);
    detail::AppendU64(bytes, 0u);
    detail::AppendU64(bytes, 0u);

    const uint64_t levelOffset = kHeaderSize + kLevelIndexSize;
    detail::AppendU64(bytes, levelOffset);
    detail::AppendU64(bytes, expectedLength);
    detail::AppendU64(bytes, expectedLength);
    bytes.insert(bytes.end(), rgba.begin(), rgba.end());

    std::ofstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        error = "Failed to open output file: " + path.string();
        return false;
    }
    stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream) {
        error = "Failed to write output file: " + path.string();
        return false;
    }
    return true;
}

}  // namespace PathTracer::Ktx2
