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
    enum class ExecutionMode : uint32_t {
        Megakernel = 0,
        Wavefront = 1,
    };

    enum class WavefrontSchedulingPolicy : uint32_t {
        Preview = 0,
        Final = 1,
        Offline = 2,
        Research = 3,
    };

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

    enum class FireflyClampMode : uint32_t {
        Luminance = 0,
        MaxComponent = 1,
    };

    enum class CameraControlMode : uint32_t {
        Orbit = 0,
        FirstPerson = 1,
    };

    enum class DirectLightMode : uint32_t {
        LegacyRect = 0,
        BaselineEmissive = 1,
        Ris = 2,
        RisSpatialReuse = 3,
        RisTemporalReuse = 4,
        RisWorldReuse = 5,
        RisRegirCache = 6,
        RestirDi = 7,
        RestirDiRegirHybrid = 8,
    };

    enum class RestirGiMode : uint32_t {
        Off = 0,
        DiffusePrototype = 1,
    };

    enum class PathGuidingMode : uint32_t {
        Off = 0,
        Prototype = 1,
    };

    enum class RestirPtMode : uint32_t {
        Off = 0,
        Research = 1,
        ExperimentalPathReuse = 2,
    };

    enum class RadianceCacheMode : uint32_t {
        Off = 0,
        Prototype = 1,
    };

    enum class CausticTransportMode : uint32_t {
        Off = 0,
        PathSpacePrototype = 1,
    };

    enum class VolumeTransportMode : uint32_t {
        Off = 0,
        Homogeneous = 1,
    };

    enum class VolumePhaseFunction : uint32_t {
        Isotropic = 0,
        HenyeyGreenstein = 1,
    };

    enum class SpectralRenderingMode : uint32_t {
        Off = 0,
        HeroWavelength = 1,
    };

    enum class SpectralTexturePolicy : uint32_t {
        RGBReflectanceD65 = 0,
    };

    enum class MlBackendMode : uint32_t {
        Off = 0,
        ReconstructionNoOp = 1,
    };

    enum class RestirDebugView : uint32_t {
        Beauty = 0,
        CandidateSourceId = 1,
        ReservoirConfidence = 2,
        ReGIRCellId = 3,
        PathGuidingUsedMask = 4,
        RestirPTReuseMask = 5,
        SVGFVariance = 6,
        NaNInfMask = 7,
        QueueStageFailure = 8,
        RadianceCache = 9,
    };

    // Path tracing settings
    uint32_t samplesPerFrame = 1;
    uint32_t maxDepth = 50;
    bool enableRussianRoulette = true;
    uint32_t fixedRngSeed = 0;
    uint32_t renderWidth = 0;   // 0 => use default/view size
    uint32_t renderHeight = 0;  // 0 => use default/view size
    float renderScale = 1.0f;   // Internal render resolution multiplier (0.5x - 2.0x)
    ExecutionMode executionMode = ExecutionMode::Megakernel;
    WavefrontSchedulingPolicy wavefrontSchedulingPolicy = WavefrontSchedulingPolicy::Preview;
    bool wavefrontCompactionEnabled = true;
    bool enableSoftwareRayTracing = false;
    SssMode sssMode = SssMode::Off;
    uint32_t sssMaxSteps = 32;
    bool enableSpecularNee = true;
    bool enableMnee = false;
    bool enableMneeSecondary = true;
    DirectLightMode directLightMode = DirectLightMode::LegacyRect;
    uint32_t risCandidateCount = 8;
    uint32_t spatialReuseNeighborCount = 4;
    float worldReuseCellSize = 0.5f;
    RestirGiMode restirGiMode = RestirGiMode::Off;
    PathGuidingMode pathGuidingMode = PathGuidingMode::Off;
    float pathGuidingStrength = 0.35f;
    float pathGuidingCellSize = 1.0f;
    uint32_t pathGuidingTrainingInterval = 1;
    RestirPtMode restirPtMode = RestirPtMode::Off;
    uint32_t restirPtMaxReservoirs = 4096;
    bool restirPtDebug = false;
    float restirPtReuseStrength = 0.25f;
    RadianceCacheMode radianceCacheMode = RadianceCacheMode::Off;
    float radianceCacheCellSize = 1.0f;
    uint32_t radianceCacheMinDepth = 2;
    float radianceCacheMinConfidence = 0.35f;
    uint32_t radianceCacheTrainingInterval = 1;
    float radianceCacheMaxRadiance = 16.0f;
    bool radianceCacheDebug = false;
    CausticTransportMode causticTransportMode = CausticTransportMode::Off;
    uint32_t causticMaxVertices = 4096;
    VolumeTransportMode volumeTransportMode = VolumeTransportMode::Off;
    VolumePhaseFunction volumePhaseFunction = VolumePhaseFunction::Isotropic;
    simd::float3 volumeSigmaA = {0.0f, 0.0f, 0.0f};
    simd::float3 volumeSigmaS = {0.0f, 0.0f, 0.0f};
    simd::float3 volumeEmission = {0.0f, 0.0f, 0.0f};
    float volumeAnisotropy = 0.0f;
    uint32_t volumeMaxScatteringEvents = 8;
    bool volumeNeeEnabled = true;
    bool volumeDebug = false;
    SpectralRenderingMode spectralRenderingMode = SpectralRenderingMode::Off;
    SpectralTexturePolicy spectralTexturePolicy = SpectralTexturePolicy::RGBReflectanceD65;
    float spectralMinWavelengthNm = 380.0f;
    float spectralMaxWavelengthNm = 720.0f;
    float spectralDispersionStrength = 0.012f;
    bool spectralDebug = false;
    MlBackendMode mlBackendMode = MlBackendMode::Off;
    std::string mlModelPackagePath;
    bool mlDebug = false;
    bool restirDebugEnabled = false;
    RestirDebugView restirDebugView = RestirDebugView::Beauty;
    bool restirDebugCounters = false;
    bool restirDebugSanityAlarms = true;
#if PT_DEBUG_TOOLS
    bool enablePathDebug = false;
    uint32_t debugPixelX = 0;
    uint32_t debugPixelY = 0;
    uint32_t debugMaxEntries = 128;
    bool parityAssertEnabled = false;
    ParityAssertMode parityAssertMode = ParityAssertMode::ProbePixelOnly;
    uint32_t parityPixelX = 0;
    uint32_t parityPixelY = 0;
    uint32_t parityTargetDepth = 0;
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
    // Reject shadow-ray hits that were found only via the tMin=0 retry and
    // land at t < kEpsilon on a different mesh — matching SWRT behaviour.
    // Fixes false-positive occlusion in diffuse rectNEE (H4).
    bool enableHardwareShadowNearRetryReject = true;

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
    bool gltfCompatFlipTexcoordV = false;        // Flip glTF V coordinates for image-origin compatibility.

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
    bool debugDirectLightAudit = false;
    float debugNormalStrengthScale = 1.0f;
    float debugNormalLodBias = 0.0f;
    float debugOrmLodBias = 0.0f;
    float debugEnvMipOverride = -1.0f;
    bool debugEnableVisorOverride = false;
    int32_t debugVisorOverrideMaterialId = -1;  // -1 = auto visor region mask
    float debugVisorOverrideRoughness = 0.15f;
    float debugVisorOverrideF0 = 0.04f;
    bool debugEnvNearest = false;

    // Camera settings
    CameraControlMode cameraControlMode = CameraControlMode::Orbit;
    simd::float3 cameraTarget = {0.0f, 0.0f, 0.0f};
    float cameraDistance = 13.490737f;
    float cameraYaw = 0.226799f;     // radians
    float cameraPitch = 0.149000f;   // radians
    simd::float3 firstPersonCameraPosition = {0.0f, 0.0f, 0.0f};
    float firstPersonMoveSpeed = 3.0f;
    float cameraVerticalFov = 20.0f; // degrees
    float cameraDefocusAngle = 0.0f; // degrees (0 disables depth of field blur)
    float cameraFocusDistance = 0.0f; // 0 => auto (matches cameraDistance)
    bool explicitCameraEnabled = false;
    simd::float3 explicitCameraPosition = {0.0f, 0.0f, 0.0f};
    simd::float3 explicitCameraForward = {0.0f, 0.0f, -1.0f};
    simd::float3 explicitCameraUp = {0.0f, 1.0f, 0.0f};

    // Background / environment
    BackgroundMode backgroundMode = BackgroundMode::Gradient;
    simd::float3 backgroundColor = {0.0f, 0.0f, 0.0f};
    std::string environmentMapPath{};
    float environmentRotation = 0.0f;   // Radians, rotates env map around world Y
    float environmentIntensity = 1.0f;  // Multiplier applied to sampled env radiance
    bool environmentMapDirty = false;   // UI flag to request an environment reload

    // Firefly clamping / variance control
    bool fireflyClampEnabled = true;
    FireflyClampMode fireflyClampMode = FireflyClampMode::Luminance;
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
    bool denoiseHdr = true;             // Treat denoise input as linear HDR radiance
    uint32_t denoiseFrequency = 4;      // Denoise every N frames (1=every frame, 4=every 4th frame, etc)

    // GPU SVGF denoising with temporal luminance moments, variance-guided
    // a-trous filtering, and normal/albedo edge stops. This is disabled by
    // default and intentionally separate from the OIDN CPU postprocess.
    bool svgfDenoiseEnabled = false;
    uint32_t svgfAtrousIterations = 1;
    float svgfNormalSigma = 64.0f;
    float svgfAlbedoSigma = 16.0f;
    float svgfLuminanceSigma = 4.0f;
};

}  // namespace PathTracer
