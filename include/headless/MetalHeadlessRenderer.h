#pragma once

#include "headless/IHeadlessRenderer.h"

class MetalHeadlessRenderer : public IHeadlessRenderer {
public:
    bool render(const HeadlessScene& scene,
                const HeadlessCamera& camera,
                const PathTracer::RenderSettings& settings,
                uint32_t sppTotal,
                bool verbose,
                bool statsMode,
                bool captureAuxBuffers,
                HeadlessRenderOutput& out,
                std::string& error) override;
    bool setSampleCheckpointOptions(const HeadlessSampleCheckpointOptions& options, std::string& error) override {
        (void)error;
        m_checkpoint = options;
        return true;
    }

private:
    HeadlessSampleCheckpointOptions m_checkpoint;
};
