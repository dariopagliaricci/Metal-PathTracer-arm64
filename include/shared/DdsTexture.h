#pragma once

// DDS header parsing and block decoding (BC1/DXT1, BC3/DXT5, BC5/ATI2) to RGBA8. Shared by the importer, which writes
// level 0 into the canonical KTX2, and SceneResources: its Metal loader parses DDS headers with it, and the deferred
// CPU copy of a DDS-backed texture (the one Embree reads) decodes the DDS's own mip levels, the levels Metal samples.
// They need not equal a box filter of level 0.
//
// The block rules are the ones Apple GPUs apply when Metal samples BC1/BC3/BC5, measured by reading every texel of
// every level back from the GPU and comparing it with this decoder. 565 endpoints are widened by bit replication, and
// each palette entry is a 6-bit weighted blend (BC7's weights). These are this hardware's measured results, not the
// D3D spec's rules, which allow some error in the blends. On every level of the DDS that Sun Temple, Emerald Square
// and Zero Day load, BC1 and BC3 colour match the GPU within 1.2e-4 in linear light (the GPU's own sRGB decode), and
// BC3 alpha matches byte for byte. BC5 (BC5_RGUnorm) keeps more than 8 bits on the GPU: it returns the blend
// unrounded. This decoder rounds it, so a value exactly halfway between two bytes can land 1 byte off (0.2% of
// interpolated texels on a Sun Temple normal map). The earlier rounded endpoints and floored thirds made Embree's
// copy up to 0.9% darker than the texture Metal samples.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace PathTracer::Dds {

enum class Format { Bc1, Bc3, Bc5 };

struct Header {
    Format format = Format::Bc1;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipCount = 1;
    size_t dataOffset = 128;
    bool forcedSrgb = false;  // DX10 BC1/BC3 *_UNORM_SRGB (DXGI 72, 78).
};

struct Level {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> rgba;
};

namespace detail {

inline uint32_t ReadU32(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8u) |
           (static_cast<uint32_t>(data[2]) << 16u) |
           (static_cast<uint32_t>(data[3]) << 24u);
}

inline uint16_t ReadU16(const uint8_t* data) {
    return static_cast<uint16_t>(data[0]) |
           static_cast<uint16_t>(static_cast<uint16_t>(data[1]) << 8u);
}

// 565 endpoints widen to 8 bits by bit replication, as the GPU does (not round(v * 255 / 31), which differs by 1 on
// some values).
inline uint8_t Expand5(uint8_t value) {
    return static_cast<uint8_t>((value << 3u) | (value >> 2u));
}

inline uint8_t Expand6(uint8_t value) {
    return static_cast<uint8_t>((value << 2u) | (value >> 4u));
}

// A palette entry between two 8-bit endpoints, weight / 64 of the way to b, rounded: the GPU's blend. The weights are
// BC7's: 21 and 43 for the thirds, 32 for the half, 9..55 and 13..51 for BC4's sevenths and fifths.
inline uint8_t Blend64(uint32_t a, uint32_t b, uint32_t weight) {
    return static_cast<uint8_t>(((64u - weight) * a + weight * b + 32u) >> 6u);
}

struct Color32 {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    uint8_t a = 255;
};

inline Color32 DecodeRgb565(uint16_t value) {
    Color32 color;
    color.r = Expand5(static_cast<uint8_t>((value >> 11u) & 0x1Fu));
    color.g = Expand6(static_cast<uint8_t>((value >> 5u) & 0x3Fu));
    color.b = Expand5(static_cast<uint8_t>(value & 0x1Fu));
    return color;
}

inline Color32 BlendColor(const Color32& a, const Color32& b, uint32_t weight) {
    return {Blend64(a.r, b.r, weight), Blend64(a.g, b.g, weight), Blend64(a.b, b.b, weight), 255};
}

inline void WritePixel(std::vector<uint8_t>& rgba,
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

inline void DecompressBc1Block(const uint8_t* block,
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
        colors[2] = BlendColor(colors[0], colors[1], 21u);
        colors[3] = BlendColor(colors[0], colors[1], 43u);
    } else {
        colors[2] = BlendColor(colors[0], colors[1], 32u);
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

inline std::array<uint8_t, 8> DecodeBc4Values(const uint8_t* block) {
    std::array<uint8_t, 8> values{};
    values[0] = block[0];
    values[1] = block[1];
    // i/7 and i/5 of the way to values[1], as 6-bit weights (round(64 i / 7), round(64 i / 5)).
    static constexpr uint32_t kSevenths[6] = {9u, 18u, 27u, 37u, 46u, 55u};
    static constexpr uint32_t kFifths[4] = {13u, 26u, 38u, 51u};
    if (values[0] > values[1]) {
        for (int i = 1; i <= 6; ++i) {
            values[static_cast<size_t>(i) + 1u] = Blend64(values[0], values[1], kSevenths[i - 1]);
        }
    } else {
        for (int i = 1; i <= 4; ++i) {
            values[static_cast<size_t>(i) + 1u] = Blend64(values[0], values[1], kFifths[i - 1]);
        }
        values[6] = 0;
        values[7] = 255;
    }
    return values;
}

inline uint64_t DecodeBc4IndexBits(const uint8_t* block) {
    uint64_t bits = 0u;
    for (int i = 0; i < 6; ++i) {
        bits |= static_cast<uint64_t>(block[2 + i]) << (8u * static_cast<uint32_t>(i));
    }
    return bits;
}

inline void DecompressDxt5Block(const uint8_t* block,
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
    colors[2] = BlendColor(colors[0], colors[1], 21u);
    colors[3] = BlendColor(colors[0], colors[1], 43u);

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

// reconstructZ fills blue with the unit normal's z, as the importer stores normal maps. Without it the block decodes
// as the GPU samples BC5_RGUnorm: blue 0, alpha 1.
inline void DecompressBc5Block(const uint8_t* block,
                               uint32_t width,
                               uint32_t height,
                               uint32_t blockX,
                               uint32_t blockY,
                               bool reconstructZ,
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
            Color32 color;
            color.r = red;
            color.g = green;
            color.b = 0;
            if (reconstructZ) {
                const float nx = static_cast<float>(red) / 127.5f - 1.0f;
                const float ny = static_cast<float>(green) / 127.5f - 1.0f;
                const float nz = std::sqrt(std::max(0.0f, 1.0f - nx * nx - ny * ny));
                color.b = static_cast<uint8_t>(std::clamp((nz * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
            }
            color.a = 255;
            WritePixel(rgba, width, x, y, color);
        }
    }
}

}  // namespace detail

inline size_t BlockBytes(Format format) {
    return format == Format::Bc1 ? 8u : 16u;
}

// The byte size of a mip level when it fits in `available` bytes, else 0. The block count is computed in 64 bits, so a
// corrupt header cannot wrap it to a size that passes.
inline size_t LevelBytes(Format format, uint32_t width, uint32_t height, size_t available) {
    const uint64_t blocks = ((static_cast<uint64_t>(width) + 3u) / 4u) * ((static_cast<uint64_t>(height) + 3u) / 4u);
    const size_t blockBytes = BlockBytes(format);
    return blocks <= available / blockBytes ? static_cast<size_t>(blocks) * blockBytes : 0u;
}

// Reads the header of a DDS in a format both the importer and Metal can use. On failure, `error` (when given) says why.
inline bool ParseHeader(const std::vector<uint8_t>& bytes, Header& header, std::string* error = nullptr) {
    auto fail = [error](const char* reason) {
        if (error) {
            *error = reason;
        }
        return false;
    };
    if (bytes.size() < 128u || std::memcmp(bytes.data(), "DDS ", 4) != 0) {
        return fail("DDS header is invalid");
    }
    header = Header{};
    header.height = detail::ReadU32(bytes.data() + 12);
    header.width = detail::ReadU32(bytes.data() + 16);
    header.mipCount = std::max<uint32_t>(1u, detail::ReadU32(bytes.data() + 28));
    if (header.width == 0 || header.height == 0) {
        return fail("DDS dimensions are invalid");
    }
    const uint8_t* fourcc = bytes.data() + 84;
    if (std::memcmp(fourcc, "DXT1", 4) == 0) {
        header.format = Format::Bc1;
    } else if (std::memcmp(fourcc, "DXT5", 4) == 0) {
        header.format = Format::Bc3;
    } else if (std::memcmp(fourcc, "ATI2", 4) == 0) {
        header.format = Format::Bc5;
    } else if (std::memcmp(fourcc, "DX10", 4) == 0) {
        if (bytes.size() < 148u) {
            return fail("DDS DX10 header is truncated");
        }
        header.dataOffset = 148u;
        switch (detail::ReadU32(bytes.data() + 128)) {
            case 71u: header.format = Format::Bc1; break;
            case 72u: header.format = Format::Bc1; header.forcedSrgb = true; break;
            case 77u: header.format = Format::Bc3; break;
            case 78u: header.format = Format::Bc3; header.forcedSrgb = true; break;
            case 83u:
            case 84u: header.format = Format::Bc5; break;
            default: return fail("DDS DXGI format is unsupported");
        }
    } else {
        return fail("DDS compression format is unsupported");
    }
    return true;
}

// Whether a texture loaded with the requested sRGB flag decodes its colour as sRGB, as Metal picks the pixel format.
inline bool DecodesAsSrgb(const Header& header, bool requestedSrgb) {
    return header.format != Format::Bc5 && (requestedSrgb || header.forcedSrgb);
}

// Decodes the header's level count capped at a full chain, the levels Metal's loader uploads, or at most maxLevels
// of them (0 = no further cap). Fails when a decoded level's data is truncated. (Metal's loader also refuses a DDS
// that lacks any declared level; SceneResources checks that before calling this.)
inline bool DecodeLevels(const std::vector<uint8_t>& bytes,
                         const Header& header,
                         uint32_t maxLevels,
                         bool reconstructBc5Z,
                         std::vector<Level>& levels) {
    levels.clear();
    uint32_t fullChain = 1u;
    for (uint32_t extent = std::max(header.width, header.height); extent > 1u; extent >>= 1u) {
        ++fullChain;
    }
    uint32_t count = std::min(header.mipCount, fullChain);
    if (maxLevels != 0u) {
        count = std::min(count, maxLevels);
    }
    if (header.dataOffset > bytes.size()) {
        return false;
    }
    size_t offset = header.dataOffset;
    uint32_t width = header.width;
    uint32_t height = header.height;
    const size_t blockBytes = BlockBytes(header.format);
    for (uint32_t level = 0; level < count; ++level) {
        const size_t levelBytes = LevelBytes(header.format, width, height, bytes.size() - offset);
        if (levelBytes == 0u) {
            levels.clear();
            return false;
        }
        Level out;
        out.width = width;
        out.height = height;
        out.rgba.assign(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u, 0);
        const uint32_t blockWidth = (width + 3u) / 4u;
        const uint32_t blockHeight = (height + 3u) / 4u;
        const uint8_t* blocks = bytes.data() + offset;
        for (uint32_t by = 0; by < blockHeight; ++by) {
            for (uint32_t bx = 0; bx < blockWidth; ++bx) {
                const uint8_t* block = blocks + (static_cast<size_t>(by) * blockWidth + bx) * blockBytes;
                if (header.format == Format::Bc1) {
                    detail::DecompressBc1Block(block, width, height, bx, by, true, out.rgba);
                } else if (header.format == Format::Bc3) {
                    detail::DecompressDxt5Block(block, width, height, bx, by, out.rgba);
                } else {
                    detail::DecompressBc5Block(block, width, height, bx, by, reconstructBc5Z, out.rgba);
                }
            }
        }
        levels.push_back(std::move(out));
        offset += levelBytes;
        width = std::max(width >> 1u, 1u);
        height = std::max(height >> 1u, 1u);
    }
    return true;
}

}  // namespace PathTracer::Dds
