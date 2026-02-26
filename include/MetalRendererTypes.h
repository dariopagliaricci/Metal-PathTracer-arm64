#pragma once

#include <cstdint>
#include <string>

struct MetalRendererOptions {
    int width = 1280;
    int height = 720;
    std::string windowTitle = "Path Tracer Metal";
    bool headless = false;           // If true, render without window/ImGui
    bool verbose = false;            // Headless timing/progress logs when true
    uint32_t fixedRngSeed = 0;       // If non-zero, use fixed RNG seed for deterministic output
    std::string initialScene;        // Optional scene identifier to load at startup
    bool enableSoftwareRayTracing = false;   // If true, force software path tracing kernels
    bool presentationMode = false;   // If true, start in Presentation Mode
};
