#pragma once

#include <memory>
#include <string>
#include <vector>

#include "MetalRendererTypes.h"

namespace PathTracer {
struct RenderSettings;
}

class MetalRenderer {
public:
    MetalRenderer();
    ~MetalRenderer();

    bool init(void* windowHandle, const MetalRendererOptions& options);
    void drawFrame();
    void resize(int width, int height);
    void resetAccumulation();
    void setTonemapMode(uint32_t mode);
    void shutdown();

    // Headless rendering support
    bool exportToPPM(const char* filepath);

    // Scene control
    bool setScene(const char* identifier);
    bool loadSceneFromPath(const char* path);
    std::vector<std::string> sceneIdentifiers() const;

    // Settings accessors
    PathTracer::RenderSettings settings() const;
    void applySettings(const PathTracer::RenderSettings& settings, bool resetAccumulation = true);
    void setSamplesPerFrame(uint32_t samples);
    uint32_t sampleCount() const;

    bool captureAverageImage(std::vector<float>& outLinearRGB,
                             uint32_t& width,
                             uint32_t& height,
                             std::vector<float>* outSampleCounts = nullptr);

private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};
