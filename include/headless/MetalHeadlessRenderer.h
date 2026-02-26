#pragma once

#include "headless/IHeadlessRenderer.h"

class MetalHeadlessRenderer : public IHeadlessRenderer {
public:
    bool render(const HeadlessScene& scene,
                const HeadlessCamera& camera,
                const PathTracer::RenderSettings& settings,
                uint32_t sppTotal,
                bool verbose,
                HeadlessRenderOutput& out,
                std::string& error) override;
};
