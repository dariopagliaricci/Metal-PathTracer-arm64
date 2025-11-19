#import "renderer/EnvImportanceSampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace PathTracer {
namespace {

constexpr float kPi = 3.14159265358979323846f;

inline float Luminance(const float* rgba) {
    return 0.2126f * rgba[0] + 0.7152f * rgba[1] + 0.0722f * rgba[2];
}

void BuildAliasTable(const std::vector<float>& probabilities,
                     std::vector<uint32_t>& aliasOut,
                     std::vector<float>& thresholdOut) {
    const size_t count = probabilities.size();
    aliasOut.assign(count, 0u);
    thresholdOut.assign(count, 0.0f);
    if (count == 0) {
        return;
    }

    std::vector<float> scaled(count, 0.0f);
    std::vector<size_t> small;
    std::vector<size_t> large;
    small.reserve(count);
    large.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        const float scaledValue = probabilities[i] * static_cast<float>(count);
        scaled[i] = scaledValue;
        if (scaledValue < 1.0f) {
            small.push_back(i);
        } else {
            large.push_back(i);
        }
    }

    while (!small.empty() && !large.empty()) {
        const size_t smallIndex = small.back();
        small.pop_back();
        size_t& largeIndex = large.back();

        thresholdOut[smallIndex] = std::clamp(scaled[smallIndex], 0.0f, 1.0f);
        aliasOut[smallIndex] = static_cast<uint32_t>(largeIndex);
        scaled[largeIndex] = (scaled[largeIndex] + scaled[smallIndex]) - 1.0f;

        if (scaled[largeIndex] < 1.0f - 1e-7f) {
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
    const float dTheta = kPi / static_cast<float>(height);
    const float dPhi = (2.0f * kPi) / static_cast<float>(width);

    std::vector<float> weights(texelCount, 0.0f);
    std::vector<float> rowWeights(height, 0.0f);
    float totalWeight = 0.0f;

    for (uint32_t y = 0; y < height; ++y) {
        const float theta = (static_cast<float>(y) + 0.5f) * dTheta;
        const float sinTheta = std::sin(theta);
        const float cellSolidAngle = std::max(sinTheta, 0.0f) * dTheta * dPhi;
        for (uint32_t x = 0; x < width; ++x) {
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const size_t rgbaIndex = index * 4ull;
            const float luminance = Luminance(&rgba32[rgbaIndex]);
            const float weight = std::max(luminance, 0.0f) * cellSolidAngle;
            weights[index] = weight;
            rowWeights[y] += weight;
            totalWeight += weight;
        }
    }

    if (totalWeight <= 0.0f) {
        if (error) {
            *error = "Environment map contains no positive radiance";
        }
        return false;
    }

    outDist->width = width;
    outDist->height = height;
    outDist->aliasCount = static_cast<uint32_t>(texelCount);
    outDist->totalWeight = totalWeight;

    std::vector<float> marginalProb(height, 0.0f);
    for (uint32_t y = 0; y < height; ++y) {
        marginalProb[y] = rowWeights[y] > 0.0f ? (rowWeights[y] / totalWeight) : 0.0f;
    }
    BuildAliasTable(marginalProb, outDist->marginalAlias, outDist->marginalThreshold);

    outDist->conditionalAlias.assign(texelCount, 0u);
    outDist->conditionalThreshold.assign(texelCount, 0.0f);

    for (uint32_t y = 0; y < height; ++y) {
        const size_t rowOffset = static_cast<size_t>(y) * static_cast<size_t>(width);
        std::vector<float> conditionalProb(width, 0.0f);
        if (rowWeights[y] > 0.0f) {
            const float invRowWeight = 1.0f / rowWeights[y];
            for (uint32_t x = 0; x < width; ++x) {
                conditionalProb[x] = weights[rowOffset + x] * invRowWeight;
            }
        } else {
            const float uniform = width > 0 ? 1.0f / static_cast<float>(width) : 0.0f;
            std::fill(conditionalProb.begin(), conditionalProb.end(), uniform);
        }

        std::vector<uint32_t> aliasRow;
        std::vector<float> thresholdRow;
        BuildAliasTable(conditionalProb, aliasRow, thresholdRow);
        for (uint32_t x = 0; x < width; ++x) {
            const size_t index = rowOffset + x;
            outDist->conditionalAlias[index] = aliasRow[x];
            outDist->conditionalThreshold[index] = thresholdRow[x];
        }
    }

    outDist->texelPdf.assign(texelCount, 0.0f);
    for (uint32_t y = 0; y < height; ++y) {
        const float theta = (static_cast<float>(y) + 0.5f) * dTheta;
        const float sinTheta = std::sin(theta);
        const float cellSolidAngle = std::max(sinTheta, 0.0f) * dTheta * dPhi;
        for (uint32_t x = 0; x < width; ++x) {
            const size_t index = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            const float probability = weights[index] / totalWeight;
            outDist->texelPdf[index] = (cellSolidAngle > 0.0f) ? (probability / cellSolidAngle) : 0.0f;
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
    if (colFrac >= colThreshold) {
        col = std::min(colAlias, dist.width - 1u);
    }

    float jitterX = uConditional - std::floor(uConditional);
    float jitterY = uJitter;
    float fx = (static_cast<float>(col) + jitterX) / static_cast<float>(dist.width);
    float fy = (static_cast<float>(row) + jitterY) / static_cast<float>(dist.height);

    float theta = fy * kPi;
    float phi = fx * (2.0f * kPi);

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
