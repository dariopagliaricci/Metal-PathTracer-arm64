#pragma once

// Mean-preserving mip levels for 8-bit colour textures.
//
// A 2x2 box keeps a texture's mean at every level only when each level halves exactly. With an odd size, a 2x2 box
// drops the last row or column, and Metal's generateMipmapsForTexture takes a bilinear sample at each texel's
// centre: neither keeps the mean, and at the coarse levels both are far off (an 894x894 tile texture's 1x1 level was
// 1.20 / 1.34 / 1.50 times the mip-0 mean in R / G / B). This filter weights every source texel by the area it shares
// with the destination texel's true footprint, so each level keeps the mean of the one above it.
//
// Power-of-two sizes take the same chain. On an RGBA8Unorm_sRGB texture the blit builds each level from the previous
// level's 8-bit texels and rounds each to the nearest sRGB code. Rounding to the nearest code in sRGB space raises the
// linear value on average, and the rise adds up level by level: over Bistro's 193 base-colour images the 1x1 level was
// 1.0106 times the mip-0 mean, and up to 1.04 on dark images. Here the chain runs in float and each level is quantized
// once. Metal uploads the 8-bit levels built here, and Embree's CPU copy decodes the same levels, so both backends
// sample the same texels.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace PathTracer::MipChain {

inline bool IsPowerOfTwo(uint32_t value) {
    return value != 0u && (value & (value - 1u)) == 0u;
}

// Every level of a texture halves exactly, so a 2x2 box keeps its mean, only when both sizes are powers of two.
inline bool BoxChainKeepsMean(uint32_t width, uint32_t height) {
    return IsPowerOfTwo(width) && IsPowerOfTwo(height);
}

// One level to the next along one axis: an exact halving, or a size of 1 that stays 1.
inline bool AxisHalvesExactly(uint32_t srcSize, uint32_t dstSize) {
    return srcSize == 2u * dstSize || (srcSize == 1u && dstSize == 1u);
}

inline uint32_t NextLevelSize(uint32_t size) {
    return std::max(size / 2u, 1u);
}

// The source texels under one destination texel along one axis, with their weights.
struct AxisFootprint {
    uint32_t first = 0u;
    uint32_t count = 0u;
    std::array<float, 4> weight{};
};

// Destination texel d of a level dstSize wide covers source texels [d * srcSize / dstSize, (d + 1) * srcSize /
// dstSize). Each source texel it touches is weighted by the overlap, computed exactly in units of 1 / dstSize source
// texels. Every source texel's weights over the level sum to dstSize / srcSize, so the level keeps the source mean.
// The footprint is at most 3 texels long (srcSize 2k + 1 to dstSize k), so it touches at most 4.
inline std::vector<AxisFootprint> AxisFootprints(uint32_t srcSize, uint32_t dstSize) {
    std::vector<AxisFootprint> footprints(dstSize);
    for (uint32_t d = 0u; d < dstSize; ++d) {
        const uint64_t lo = static_cast<uint64_t>(d) * srcSize;
        const uint64_t hi = static_cast<uint64_t>(d + 1u) * srcSize;
        AxisFootprint& footprint = footprints[d];
        footprint.first = static_cast<uint32_t>(lo / dstSize);
        footprint.count = static_cast<uint32_t>((hi - 1u) / dstSize) - footprint.first + 1u;
        for (uint32_t i = 0u; i < footprint.count && i < 4u; ++i) {
            const uint64_t texelLo = static_cast<uint64_t>(footprint.first + i) * dstSize;
            const uint64_t overlap = std::min(hi, texelLo + dstSize) - std::max(lo, texelLo);
            footprint.weight[i] = static_cast<float>(static_cast<double>(overlap) / static_cast<double>(srcSize));
        }
    }
    return footprints;
}

// The next level of a linear RGBA float image (4 floats per texel), filtered over each texel's true footprint.
inline std::vector<float> DownsampleFootprint(const std::vector<float>& src,
                                              uint32_t srcWidth,
                                              uint32_t srcHeight,
                                              uint32_t dstWidth,
                                              uint32_t dstHeight) {
    const std::vector<AxisFootprint> xs = AxisFootprints(srcWidth, dstWidth);
    const std::vector<AxisFootprint> ys = AxisFootprints(srcHeight, dstHeight);
    std::vector<float> dst(static_cast<size_t>(dstWidth) * dstHeight * 4u, 0.0f);
    for (uint32_t y = 0u; y < dstHeight; ++y) {
        const AxisFootprint& fy = ys[y];
        for (uint32_t x = 0u; x < dstWidth; ++x) {
            const AxisFootprint& fx = xs[x];
            float sum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint32_t j = 0u; j < fy.count; ++j) {
                const size_t row = static_cast<size_t>(fy.first + j) * srcWidth;
                for (uint32_t i = 0u; i < fx.count; ++i) {
                    const float w = fx.weight[i] * fy.weight[j];
                    const size_t s = (row + fx.first + i) * 4u;
                    for (uint32_t c = 0u; c < 4u; ++c) sum[c] += w * src[s + c];
                }
            }
            const size_t d = (static_cast<size_t>(y) * dstWidth + x) * 4u;
            for (uint32_t c = 0u; c < 4u; ++c) dst[d + c] = sum[c];
        }
    }
    return dst;
}

inline float SrgbEncode(float linear) {
    const double c = std::clamp(static_cast<double>(linear), 0.0, 1.0);
    return static_cast<float>(c <= 0.0031308 ? c * 12.92 : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055);
}

inline uint8_t QuantizeUnorm8(float value) {
    const float v = std::isfinite(value) ? std::clamp(value, 0.0f, 1.0f) : 0.0f;
    return static_cast<uint8_t>(std::lround(v * 255.0f));
}

// QuantizeUnorm8(SrgbEncode(linear)) without a pow per texel. That function never decreases as linear grows, so it is
// fixed by the 255 smallest floats at which it reaches codes 1 to 255; each is found by a search over the float bit
// patterns of [0, 1] that evaluates the function itself, so the table gives the same byte for every float.
inline const std::array<float, 255>& SrgbQuantizeThresholds() {
    static const std::array<float, 255> thresholds = [] {
        std::array<float, 255> table{};
        for (uint32_t code = 1u; code < 256u; ++code) {
            uint32_t lo = 0u;
            uint32_t hi = 0x3f800000u;  // 1.0f, which gives code 255
            while (lo < hi) {
                const uint32_t mid = lo + (hi - lo) / 2u;
                float x = 0.0f;
                std::memcpy(&x, &mid, sizeof(x));
                if (QuantizeUnorm8(SrgbEncode(x)) >= code) {
                    hi = mid;
                } else {
                    lo = mid + 1u;
                }
            }
            std::memcpy(&table[code - 1u], &lo, sizeof(float));
        }
        return table;
    }();
    return thresholds;
}

// For each bucket of 2^15 float bit patterns in [0, 1), the code of the bucket's smallest float: a lower bound on the
// code of every float in the bucket, which QuantizeSrgb8 then walks up through the thresholds (at most a step or two).
inline const std::array<uint8_t, (0x3f800000u >> 15u)>& SrgbQuantizeBuckets() {
    static const std::array<uint8_t, (0x3f800000u >> 15u)> buckets = [] {
        const std::array<float, 255>& thresholds = SrgbQuantizeThresholds();
        std::array<uint8_t, (0x3f800000u >> 15u)> table{};
        for (uint32_t bucket = 0u; bucket < table.size(); ++bucket) {
            const uint32_t bits = bucket << 15u;
            float x = 0.0f;
            std::memcpy(&x, &bits, sizeof(x));
            table[bucket] = static_cast<uint8_t>(std::upper_bound(thresholds.begin(), thresholds.end(), x) -
                                                 thresholds.begin());
        }
        return table;
    }();
    return buckets;
}

inline uint8_t QuantizeSrgb8(float linear) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &linear, sizeof(bits));
    if (bits >= 0x3f800000u) {
        // Negative (sign bit) and NaN give 0, as QuantizeUnorm8(SrgbEncode(x)) does; 1 and above, infinity included, 255.
        return ((bits & 0x80000000u) != 0u || bits > 0x7f800000u) ? 0u : 255u;
    }
    const std::array<float, 255>& thresholds = SrgbQuantizeThresholds();
    uint32_t code = SrgbQuantizeBuckets()[bits >> 15u];
    while (code < 255u && linear >= thresholds[code]) {
        ++code;
    }
    return static_cast<uint8_t>(code);
}

struct Rgba8Level {
    uint32_t width = 0u;
    uint32_t height = 0u;
    std::vector<uint8_t> rgba;
};

// Levels 1 and up of an 8-bit RGBA texture (channels 0-2 colour, in either order; channel 3 alpha). Colour is decoded
// from sRGB when srgb is set and filtered in linear light; alpha is linear. The chain runs in float and each level is
// quantized once, so the error does not accumulate down the chain.
inline std::vector<Rgba8Level> BuildRgba8MipLevels(const uint8_t* rgba0, uint32_t width, uint32_t height, bool srgb) {
    std::array<float, 256> decode{};
    for (uint32_t i = 0u; i < 256u; ++i) {
        const double c = static_cast<double>(i) / 255.0;
        decode[i] = static_cast<float>(!srgb ? c : (c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4)));
    }
    std::vector<float> level(static_cast<size_t>(width) * height * 4u);
    for (size_t i = 0u; i < level.size(); ++i) {
        level[i] = (i % 4u == 3u) ? static_cast<float>(rgba0[i]) / 255.0f : decode[rgba0[i]];
    }
    std::vector<Rgba8Level> out;
    while (width > 1u || height > 1u) {
        const uint32_t nextWidth = NextLevelSize(width);
        const uint32_t nextHeight = NextLevelSize(height);
        level = DownsampleFootprint(level, width, height, nextWidth, nextHeight);
        width = nextWidth;
        height = nextHeight;
        Rgba8Level encoded{width, height, std::vector<uint8_t>(level.size())};
        for (size_t i = 0u; i < level.size(); ++i) {
            encoded.rgba[i] = (srgb && i % 4u != 3u) ? QuantizeSrgb8(level[i]) : QuantizeUnorm8(level[i]);
        }
        out.push_back(std::move(encoded));
    }
    return out;
}

}  // namespace PathTracer::MipChain
