#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <simd/simd.h>

namespace PathTracer {

struct EnvImportanceDistribution {
    std::vector<float> texelPdf;                 // Per-texel PDF in solid-angle domain
    std::vector<uint32_t> conditionalAlias;      // Width * height entries
    std::vector<float> conditionalThreshold;     // Width * height entries
    std::vector<uint32_t> marginalAlias;         // Height entries
    std::vector<float> marginalThreshold;        // Height entries
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t aliasCount = 0;
    float totalWeight = 0.0f;
};

struct EnvImportanceSample {
    simd::float3 direction{0.0f, 0.0f, 0.0f};
    simd::float3 radiance{0.0f, 0.0f, 0.0f};
    float pdf = 0.0f;
};

bool BuildEnvImportanceDistribution(const float* rgba32,
                                    uint32_t width,
                                    uint32_t height,
                                    EnvImportanceDistribution* outDist,
                                    std::string* error = nullptr);

EnvImportanceSample SampleEnvironmentCpu(const EnvImportanceDistribution& dist,
                                         float uMarginal,
                                         float uConditional,
                                         float uJitter,
                                         float rotation,
                                         float intensity,
                                         const float* rgba32);

}  // namespace PathTracer
