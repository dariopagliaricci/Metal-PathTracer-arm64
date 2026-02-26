#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <simd/simd.h>

#include "renderer/RenderSettings.h"
#include "renderer/SceneResources.h"

enum class HeadlessBackend {
    Metal = 0,
    Embree = 1,
};

struct HeadlessScene {
    std::string source;
    bool isPath = false;
    const PathTracer::SceneResources* resources = nullptr;
};

struct HeadlessCamera {
    simd::float3 target{0.0f, 0.0f, 0.0f};
    float distance = 0.0f;
    float yaw = 0.0f;
    float pitch = 0.0f;
    float verticalFov = 0.0f;
    float defocusAngle = 0.0f;
    float focusDistance = 0.0f;
};

struct HeadlessRenderOutput {
    std::vector<float> linearRGB;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t samples = 0;
    double totalSeconds = 0.0;
    double avgMsPerSample = 0.0;
};

class IHeadlessRenderer {
public:
    virtual ~IHeadlessRenderer() = default;
    virtual bool render(const HeadlessScene& scene,
                        const HeadlessCamera& camera,
                        const PathTracer::RenderSettings& settings,
                        uint32_t sppTotal,
                        bool verbose,
                        HeadlessRenderOutput& out,
                        std::string& error) = 0;
};
