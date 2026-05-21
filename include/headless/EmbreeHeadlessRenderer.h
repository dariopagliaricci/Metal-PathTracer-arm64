#pragma once

#include "headless/IHeadlessRenderer.h"

class EmbreeHeadlessRenderer : public IHeadlessRenderer {
public:
    void setMaxThreads(uint32_t maxThreads);
    bool render(const HeadlessScene& scene,
                const HeadlessCamera& camera,
                const PathTracer::RenderSettings& settings,
                uint32_t sppTotal,
                bool verbose,
                bool statsMode,
                bool captureAuxBuffers,
                HeadlessRenderOutput& out,
                std::string& error) override;

private:
    uint32_t maxThreads_ = 0;
};
