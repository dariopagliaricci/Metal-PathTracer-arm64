#import "renderer/EnvImportanceSampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace PathTracer {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr double kPiD = 3.14159265358979323846;

inline float Luminance(const float* rgba) {
    return 0.2126f * rgba[0] + 0.7152f * rgba[1] + 0.0722f * rgba[2];
}

// Vose alias table, built in double and normalised by the actual sum of the
// probabilities, so it samples in proportion to them.
void BuildAliasTable(const std::vector<double>& probabilities,
                     std::vector<uint32_t>& aliasOut,
                     std::vector<float>& thresholdOut) {
    const size_t count = probabilities.size();
    aliasOut.assign(count, 0u);
    thresholdOut.assign(count, 0.0f);
    if (count == 0) {
        return;
    }

    const double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    const double scale = sum > 0.0 ? static_cast<double>(count) / sum : 0.0;

    std::vector<double> scaled(count, 0.0);
    std::vector<size_t> small;
    std::vector<size_t> large;
    small.reserve(count);
    large.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        const double scaledValue = probabilities[i] * scale;
        scaled[i] = scaledValue;
        if (scaledValue < 1.0) {
            small.push_back(i);
        } else {
            large.push_back(i);
        }
    }

    while (!small.empty() && !large.empty()) {
        const size_t smallIndex = small.back();
        small.pop_back();
        size_t& largeIndex = large.back();

        thresholdOut[smallIndex] = static_cast<float>(std::clamp(scaled[smallIndex], 0.0, 1.0));
        aliasOut[smallIndex] = static_cast<uint32_t>(largeIndex);
        scaled[largeIndex] = (scaled[largeIndex] + scaled[smallIndex]) - 1.0;

        if (scaled[largeIndex] < 1.0) {
            small.push_back(largeIndex);
            large.pop_back();
        }
    }

    const auto finalize = [&](const std::vector<size_t>& entries) {
        for (size_t index : entries) {
            thresholdOut[index] = 1.0f;
            aliasOut[index] = static_cast<uint32_t>(index);
        }
    };

    finalize(small);
    finalize(large);
}

}  // namespace

bool BuildEnvImportanceDistribution(const float* rgba32,
                                    uint32_t width,
                                    uint32_t height,
                                    EnvImportanceDistribution* outDist,
                                    std::string* error) {
    if (!outDist) {
        if (error) {
            *error = "Output distribution pointer was null";
        }
        return false;
    }

    *outDist = EnvImportanceDistribution{};

    if (!rgba32 || width == 0 || height == 0) {
        if (error) {
            *error = "Invalid environment texture data";
        }
        return false;
    }

    const size_t texelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    const double dTheta = kPiD / static_cast<double>(height);
    const double dPhi = (2.0 * kPiD) / static_cast<double>(width);

    // Weights, row sums and the total accumulate in double: a float32 running
    // total over the 8.4M texels of a 4k map comes out ~5% low, which made
    // texelPdf integrate to ~1.05 while the alias tables sample by row sums.
    std::vector<double> cellSolidAngles(height, 0.0);
    std::vector<float> weights(texelCount, 0.0f);
    std::vector<double> rowWeights(height, 0.0);
    double totalWeight = 0.0;

    for (uint32_t y = 0; y < height; ++y) {
        const double theta = (static_cast<double>(y) + 0.5) * dTheta;
        const double cellSolidAngle = std::max(std::sin(theta), 0.0) * dTheta * dPhi;
        cellSolidAngles[y] = cellSolidAngle;
        double rowWeight = 0.0;
        for (uint32_t x = 0; x < width; ++x) {
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const size_t rgbaIndex = index * 4ull;
            const double luminance = Luminance(&rgba32[rgbaIndex]);
            const double weight = std::isfinite(luminance) ? std::max(luminance, 0.0) * cellSolidAngle : 0.0;
            weights[index] = static_cast<float>(weight);
            rowWeight += static_cast<double>(weights[index]);
        }
        rowWeights[y] = rowWeight;
        totalWeight += rowWeight;
    }

    if (!(totalWeight > 0.0) || !std::isfinite(totalWeight)) {
        if (error) {
            *error = "Environment map contains no positive radiance";
        }
        return false;
    }

    outDist->width = width;
    outDist->height = height;
    outDist->aliasCount = static_cast<uint32_t>(texelCount);
    outDist->totalWeight = static_cast<float>(totalWeight);

    std::vector<double> marginalProb(height, 0.0);
    for (uint32_t y = 0; y < height; ++y) {
        marginalProb[y] = rowWeights[y] / totalWeight;
    }
    BuildAliasTable(marginalProb, outDist->marginalAlias, outDist->marginalThreshold);

    outDist->conditionalAlias.assign(texelCount, 0u);
    outDist->conditionalThreshold.assign(texelCount, 0.0f);

    std::vector<double> conditionalProb(width, 0.0);
    std::vector<uint32_t> aliasRow;
    std::vector<float> thresholdRow;
    for (uint32_t y = 0; y < height; ++y) {
        const size_t rowOffset = static_cast<size_t>(y) * static_cast<size_t>(width);
        if (rowWeights[y] > 0.0) {
            for (uint32_t x = 0; x < width; ++x) {
                conditionalProb[x] = static_cast<double>(weights[rowOffset + x]) / rowWeights[y];
            }
        } else {
            std::fill(conditionalProb.begin(), conditionalProb.end(), 1.0 / static_cast<double>(width));
        }

        BuildAliasTable(conditionalProb, aliasRow, thresholdRow);
        for (uint32_t x = 0; x < width; ++x) {
            const size_t index = rowOffset + x;
            outDist->conditionalAlias[index] = aliasRow[x];
            outDist->conditionalThreshold[index] = thresholdRow[x];
        }
    }

    // texelPdf is the solid-angle density of the alias sampler (a texel drawn
    // with probability weight / totalWeight, uniform over its cell).
    outDist->texelPdf.assign(texelCount, 0.0f);
    for (uint32_t y = 0; y < height; ++y) {
        const double cellSolidAngle = cellSolidAngles[y];
        if (!(cellSolidAngle > 0.0)) {
            continue;
        }
        for (uint32_t x = 0; x < width; ++x) {
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const double probability = static_cast<double>(weights[index]) / totalWeight;
            outDist->texelPdf[index] = static_cast<float>(probability / cellSolidAngle);
        }
    }

    return true;
}

EnvImportanceSample SampleEnvironmentCpu(const EnvImportanceDistribution& dist,
                                         float uMarginal,
                                         float uConditional,
                                         float uJitter,
                                         float rotation,
                                         float intensity,
                                         const float* rgba32) {
    EnvImportanceSample sample;

    if (!rgba32 || dist.width == 0 || dist.height == 0 || dist.aliasCount == 0) {
        return sample;
    }

    uMarginal = std::clamp(uMarginal, 0.0f, 0.99999994f);
    uConditional = std::clamp(uConditional, 0.0f, 0.99999994f);
    uJitter = std::clamp(uJitter, 0.0f, 0.99999994f);

    float rowChoice = uMarginal * static_cast<float>(dist.height);
    uint32_t row = std::min(static_cast<uint32_t>(rowChoice), dist.height - 1u);
    float rowFrac = rowChoice - static_cast<float>(row);
    float rowThreshold = dist.marginalThreshold[row];
    uint32_t rowAlias = dist.marginalAlias[row];
    if (rowFrac >= rowThreshold) {
        row = std::min(rowAlias, dist.height - 1u);
    }

    float colChoice = uConditional * static_cast<float>(dist.width);
    uint32_t col = std::min(static_cast<uint32_t>(colChoice), dist.width - 1u);
    float colFrac = colChoice - static_cast<float>(col);
    const size_t rowOffset = static_cast<size_t>(row) * static_cast<size_t>(dist.width);
    float colThreshold = dist.conditionalThreshold[rowOffset + col];
    uint32_t colAlias = dist.conditionalAlias[rowOffset + col];
    // Reuse the selection draw by remapping its chosen alias interval, leaving
    // a uniform texel offset without consuming another random number.
    float jitterX = colFrac / std::max(colThreshold, 1.0e-7f);
    if (colFrac >= colThreshold) {
        jitterX = (colFrac - colThreshold) / std::max(1.0f - colThreshold, 1.0e-7f);
        col = std::min(colAlias, dist.width - 1u);
    }

    jitterX = std::clamp(jitterX, 0.0f, 0.99999994f);
    float jitterY = uJitter;
    float fx = (static_cast<float>(col) + jitterX) / static_cast<float>(dist.width);
    float fy = (static_cast<float>(row) + jitterY) / static_cast<float>(dist.height);

    float theta = fy * kPi;
    // Inverse of the unchanged radiance/PDF longitude lookup.
    float phi = (fx - 0.5f) * (2.0f * kPi);

    float sinTheta = std::sin(theta);
    float cosTheta = std::cos(theta);
    simd::float3 mapDir = {sinTheta * std::cos(phi), cosTheta, sinTheta * std::sin(phi)};

    float cosRot = std::cos(rotation);
    float sinRot = std::sin(rotation);
    simd::float3 worldDir = {mapDir.x * cosRot + mapDir.z * sinRot,
                             mapDir.y,
                             -mapDir.x * sinRot + mapDir.z * cosRot};

    const size_t texelIndex = static_cast<size_t>(row) * static_cast<size_t>(dist.width) + static_cast<size_t>(col);
    sample.direction = worldDir;
    sample.pdf = (texelIndex < dist.texelPdf.size()) ? dist.texelPdf[texelIndex] : 0.0f;

    const size_t rgbaIndex = texelIndex * 4ull;
    simd::float3 color = {rgba32[rgbaIndex + 0], rgba32[rgbaIndex + 1], rgba32[rgbaIndex + 2]};
    sample.radiance = color * intensity;

    return sample;
}

}  // namespace PathTracer
