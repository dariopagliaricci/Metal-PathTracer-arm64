#pragma once

// ⚠️ INTERNAL HEADER - Subject to change without notice
// This header is part of PathTracer's internal implementation.
// Use only the public API from MetalRenderer.h

#include <cstdint>
#include <simd/simd.h>
#include <cmath>
#include <string>

namespace PathTracer {

/// Render configuration and settings
/// Consolidates all path tracing and tonemapping parameters
struct RenderSettings {
    enum class BackgroundMode : uint32_t {
        Gradient = 0,
        Solid = 1,
        Environment = 2,
    };

    enum class SssMode : uint32_t {
        Off = 0,
        Separable = 1,
        RandomWalk = 2,
    };

    enum class ParityAssertMode : uint32_t {
        Off = 0,
        ProbePixelOnly = 1,
        FirstInMediumBoundary = 2,
    };

    enum class WorkingColorSpace : uint32_t {
        LinearSRGB = 0,
        ACEScg = 1,
    };

    // Path tracing settings
    uint32_t samplesPerFrame = 1;
    uint32_t maxDepth = 50;
    bool enableRussianRoulette = true;
    uint32_t fixedRngSeed = 0;
    uint32_t renderWidth = 0;   // 0 => use default/view size
    uint32_t renderHeight = 0;  // 0 => use default/view size
    float renderScale = 1.0f;   // Internal render resolution multiplier (0.5x - 2.0x)
    bool enableSoftwareRayTracing = false;
    SssMode sssMode = SssMode::Off;
    uint32_t sssMaxSteps = 32;
    bool enableSpecularNee = true;
    bool enableMnee = false;
    bool enableMneeSecondary = true;
#if PT_DEBUG_TOOLS
    bool enablePathDebug = false;
    uint32_t debugPixelX = 0;
    uint32_t debugPixelY = 0;
    uint32_t debugMaxEntries = 128;
    bool parityAssertEnabled = false;
    ParityAssertMode parityAssertMode = ParityAssertMode::ProbePixelOnly;
    uint32_t parityPixelX = 0;
    uint32_t parityPixelY = 0;
    bool parityAssertOncePerFrame = true;
    bool forcePureHWRTForGlass = false;
#endif
    uint32_t hardwareExcludeRetries = 3;  // Number of HWRT exclusion retries (0 = no retries)
    float hardwareExitNormalBias = 0.0f;  // Extra HWRT exit bias along normal (meters)
    float hardwareExitDirectionalBias = 0.0f;  // Extra HWRT exit bias along ray direction (meters)
#if PT_DEBUG_TOOLS
    bool enableHardwareMissFallback = false;  // Allow SWRT triangle fallback when HWRT misses
    bool enableHardwareFirstHitFromSoftware = false;  // Use SWRT for depth 0, then resume HWRT
    bool enableHardwareForceSoftware = false;  // Debug: use SWRT intersections for all HWRT hits
#endif

    // Tonemapping settings
    uint32_t tonemapMode = 1;        // 1=Linear, 2=ACES, 3=Reinhard, 4=Hable
    uint32_t acesVariant = 0;        // 0=Fitted, 1=Simple
    float exposure = 0.0f;           // Exposure in stops
    float reinhardWhitePoint = 1.5f; // White point for Reinhard
    bool bloomEnabled = false;       // Optional post-tonemap bloom
    float bloomThreshold = 1.0f;     // HDR luminance threshold for bloom extraction
    float bloomIntensity = 0.12f;    // Bloom add strength
    float bloomRadius = 1.5f;        // Bloom tap radius in pixels
    WorkingColorSpace workingColorSpace = WorkingColorSpace::LinearSRGB;
    bool gltfViewerCompatibilityMode = false;   // Opt-in glTF viewer parity behavior
    bool gltfThinWalledFallback = true;         // If transmission exists without volume, force thin dielectric
    float gltfEmissiveScale = 1.0f;             // Extra emissive scale for viewer-oriented readability
    bool gltfCompatForceLinearBaseColor = false; // Treat baseColor textures as linear (debug/compat)
    bool gltfCompatForceLinearEmissive = false;  // Treat emissive textures as linear (debug/compat)

    // PBR debug toggles
    bool debugShowBaseColor = false;
    bool debugShowMetallic = false;
    bool debugShowRoughness = false;
    bool debugShowAO = false;
    bool debugDisableAO = false;
    bool debugAoIndirectOnly = true;
    bool debugDisableNormalMap = false;
    bool debugDisableOrmTexture = false;
    bool debugFlipNormalGreen = false;
    bool debugSpecularOnly = false;
    float debugNormalStrengthScale = 1.0f;
    float debugNormalLodBias = 0.0f;
    float debugOrmLodBias = 0.0f;
    float debugEnvMipOverride = -1.0f;
    bool debugEnableVisorOverride = false;
    int32_t debugVisorOverrideMaterialId = -1;  // -1 = auto visor region mask
    float debugVisorOverrideRoughness = 0.15f;
    float debugVisorOverrideF0 = 0.04f;
    bool debugEnvNearest = false;

    // Camera (orbit) settings
    simd::float3 cameraTarget = {0.0f, 0.0f, 0.0f};
    float cameraDistance = 13.490737f;
    float cameraYaw = 0.226799f;     // radians
    float cameraPitch = 0.149000f;   // radians
    float cameraVerticalFov = 20.0f; // degrees
    float cameraDefocusAngle = 0.0f; // degrees (0 disables depth of field blur)
    float cameraFocusDistance = 0.0f; // 0 => auto (matches cameraDistance)

    // Background / environment
    BackgroundMode backgroundMode = BackgroundMode::Gradient;
    simd::float3 backgroundColor = {0.0f, 0.0f, 0.0f};
    std::string environmentMapPath{};
    float environmentRotation = 0.0f;   // Radians, rotates env map around world Y
    float environmentIntensity = 1.0f;  // Multiplier applied to sampled env radiance
    bool environmentMapDirty = false;   // UI flag to request an environment reload

    // Firefly clamping / variance control
    bool fireflyClampEnabled = true;
    float fireflyClampFactor = 32.0f;
    float fireflyClampFloor = 4.0f;
    float throughputClamp = 32.0f;
    float specularTailClampBase = 0.0f;
    float specularTailClampRoughnessScale = 0.0f;
    float minSpecularPdf = 0.0f;
    float fireflyClampMaxContribution = 1000.0f;

    // Denoising settings
    bool denoiseEnabled = false;        // Enable OIDN denoising post-processing
    uint32_t denoiseFilterType = 0;     // 0=RT, 1=RTLightmap
    bool denoiseUseAlbedo = true;       // Use albedo auxiliary buffer for denoising (future)
    bool denoiseUseNormal = true;       // Use normal auxiliary buffer for denoising (future)
    uint32_t denoiseFrequency = 4;      // Denoise every N frames (1=every frame, 4=every 4th frame, etc)
};

}  // namespace PathTracer
