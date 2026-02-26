#import <Foundation/Foundation.h>

#include <algorithm>
#include <iomanip>
#include <iostream>

#include "headless/MetalHeadlessRenderer.h"
#include "MetalRenderer.h"

bool MetalHeadlessRenderer::render(const HeadlessScene& scene,
                                   const HeadlessCamera&,
                                   const PathTracer::RenderSettings& settings,
                                   uint32_t sppTotal,
                                   bool verbose,
                                   HeadlessRenderOutput& out,
                                   std::string& error) {
    MetalRendererOptions rendererOptions;
    rendererOptions.headless = true;
    rendererOptions.width = settings.renderWidth > 0 ? static_cast<int>(settings.renderWidth) : 1280;
    rendererOptions.height = settings.renderHeight > 0 ? static_cast<int>(settings.renderHeight) : 720;
    rendererOptions.windowTitle = "PathTracerCLI";
    rendererOptions.fixedRngSeed = settings.fixedRngSeed;
    rendererOptions.enableSoftwareRayTracing = settings.enableSoftwareRayTracing;
    rendererOptions.verbose = verbose;

    MetalRenderer renderer;
    if (!renderer.init(nullptr, rendererOptions)) {
        error = "Failed to initialize Metal renderer";
        return false;
    }

    bool sceneLoaded = false;
    if (scene.isPath) {
        sceneLoaded = renderer.loadSceneFromPath(scene.source.c_str());
    } else {
        sceneLoaded = renderer.setScene(scene.source.c_str());
    }

    if (!sceneLoaded) {
        error = "Failed to load scene: " + scene.source;
        return false;
    }

    renderer.applySettings(settings, true);

    const uint32_t targetSamples = std::max<uint32_t>(1u, sppTotal);
    uint32_t accumulatedSamples = renderer.sampleCount();
    uint32_t maxBatch = settings.enableSoftwareRayTracing ? 1u : 16u;
#if PT_MNEE_SWRT_RAYS
    if (verbose) {
        NSLog(@"[Warning] PT_MNEE_SWRT_RAYS enabled: MNEE rays use SWRT (slow diagnostic mode)");
    }
    if (!settings.enableSoftwareRayTracing && settings.enableMnee) {
        // Temporary: Keep headless progress responsive when SWRT is used for MNEE rays.
        // This is symptom management; revisit once timing logs reveal actual bottlenecks.
        maxBatch = 1u;
    }
#endif

    CFAbsoluteTime renderStart = CFAbsoluteTimeGetCurrent();
    double accumulatedFrameTime = 0.0;
    CFAbsoluteTime lastProgress = renderStart;

    while (accumulatedSamples < targetSamples) {
        uint32_t remaining = targetSamples - accumulatedSamples;
        uint32_t request = (maxBatch == 0) ? remaining : std::min(maxBatch, remaining);
        renderer.setSamplesPerFrame(request);

        uint32_t before = renderer.sampleCount();
        CFAbsoluteTime frameStart = CFAbsoluteTimeGetCurrent();
        renderer.drawFrame();
        CFAbsoluteTime frameEnd = CFAbsoluteTimeGetCurrent();

        uint32_t after = renderer.sampleCount();
        uint32_t produced = (after > before) ? (after - before) : std::min(request, remaining);
        if (produced == 0) {
            produced = 1;
        }
        accumulatedSamples += produced;
        accumulatedFrameTime += (frameEnd - frameStart);

        if (verbose) {
            CFAbsoluteTime now = frameEnd;
            if ((now - lastProgress) >= 0.5 || accumulatedSamples >= targetSamples) {
                double pct = (static_cast<double>(accumulatedSamples) / targetSamples) * 100.0;
                std::cout << "Progress: " << accumulatedSamples << "/" << targetSamples
                          << " spp (" << std::fixed << std::setprecision(1) << pct << "%)\r";
                std::cout.flush();
                lastProgress = now;
            }
        }
    }

    if (verbose) {
        std::cout << std::endl;
    }

    CFAbsoluteTime renderEnd = CFAbsoluteTimeGetCurrent();

    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<float> linearRGB;
    if (!renderer.captureAverageImage(linearRGB, width, height, nullptr)) {
        error = "Failed to capture rendered image";
        return false;
    }

    out.linearRGB = std::move(linearRGB);
    out.width = width;
    out.height = height;
    out.samples = accumulatedSamples;
    out.totalSeconds = renderEnd - renderStart;
    out.avgMsPerSample = (accumulatedSamples > 0)
                             ? (accumulatedFrameTime * 1000.0 / accumulatedSamples)
                             : 0.0;
    return true;
}
