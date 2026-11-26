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

    enum class FireflyClampMode : uint32_t {
        Luminance = 0,
        MaxComponent = 1,
    };

    enum class CameraControlMode : uint32_t {
        Orbit = 0,
        FirstPerson = 1,
    };

    enum class ViewportQualityPreset : uint32_t {
        Custom = 0,
        Interactive = 1,
        Balanced = 2,
        FinalPreview = 3,
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
        ProductionDiffuse = 1,
    };

    enum class PathGuidingMode : uint32_t {
        Off = 0,
        Prototype = 1,
    };

    enum class RestirPtMode : uint32_t {
        Off = 0,
        ProductionReservoirCapture = 1,
        ProductionPathReuse = 2,
        EnhancedUnbiased = 3,
        EnhancedBiased = 4,
    };

    enum class RadianceCacheMode : uint32_t {
        Off = 0,
        Production = 1,
        Prototype = Production,
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
        ReconstructionTiny = 2,
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
        CausticTransport = 10,
    };

    // Path tracing settings
    uint32_t samplesPerFrame = 1;
    uint32_t maxDepth = 50;
    ViewportQualityPreset viewportQualityPreset = ViewportQualityPreset::Interactive;
    bool viewportQualityPresetOverridden = false;
    bool enableRussianRoulette = true;
    // Add filtered normal-map variance to PBR roughness. Disable only for an
    // explicitly calibrated scene; texture filtering and mapped normals remain active.
    bool normalVarianceRoughness = true;
    // Enumerate zero-radius reference points at every existing analytic NEE vertex.
    bool referencePointExactNee = false;
    // Shared material-texture sampling bias; negative values sharpen the
    // existing filtered lookups without changing UVs or ray footprints.
    float textureLodBias = 0.0f;
    uint32_t fixedRngSeed = 0;
    uint32_t renderWidth = 0;   // 0 => use default/view size
    uint32_t renderHeight = 0;  // 0 => use default/view size
    float renderScale = 1.0f;   // Internal render resolution multiplier (0.5x - 2.0x)
    ExecutionMode executionMode = ExecutionMode::Megakernel;
    bool sourceTransport = false; // Opt-in Tungsten forward visibility and depth semantics.
    // Tungsten TraceableScene::DefaultEpsilon (5e-4): scattered, forward-continuation and shadow rays ignore hits
    // closer than this world distance. 0 = off (1 um precise spawning). Requires sourceTransport.
    float sourceRayEpsilon = 0.0f;
    bool denoiseMirrorGuides = false; // OIDN guides follow smooth source-conductor reflections (wavefront only).
    bool denoiseTransmissionGuides = false; // OIDN guides follow delta source-dielectric interfaces (wavefront only).
    // Set by ResolveSourceCapabilities, not a user setting: the scene or settings use Tungsten-source closures, textures
    // or transport. Selects the compute pipeline set specialized with kSourceFeatures (Pipelines::selectSourceFeatures).
    bool sourceFeatures = false;
    // Set by ResolveSourceCapabilities when it switched a source scene to wavefront, not a user setting. The GUI's next
    // scene load restores megakernel instead of carrying the forced mode over as the viewport's own choice.
    bool sourceForcedWavefront = false;
    WavefrontSchedulingPolicy wavefrontSchedulingPolicy = WavefrontSchedulingPolicy::Preview;
    bool wavefrontCompactionEnabled = true;
    bool wavefrontShadeIndirectEnabled = false;
    bool wavefrontShadeIndirectProbeEnabled = false;
    bool wavefrontShadeSurfaceProbeEnabled = false;
    bool wavefrontShadeMetadataProbeEnabled = false;
    bool wavefrontShadeScalarProbeEnabled = false;
    bool wavefrontShadeClosureProbeEnabled = false;
    bool wavefrontShadeBsdfProbeEnabled = false;
    bool wavefrontShadowIndirectEnabled = false;
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
    // Wavelengths carried per path in spectral mode: 3 (hero plus two rotated) or 1 (single-wavelength baseline).
    uint32_t spectralWavelengthsPerPath = 3;
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
    uint32_t tonemapMode = 1;        // 1=Linear, 2=ACES, 3=Reinhard, 4=Hable, 5=Exponential, 6=Filmic (Tungsten)
    uint32_t acesVariant = 0;        // 0=Fitted, 1=Simple, 2=Falcor (0.6 pre-exposure, sRGB)
    float exposure = 0.0f;           // Exposure in stops
    float reinhardWhitePoint = 1.5f; // White point for Reinhard
    bool bloomEnabled = false;       // Optional bloom before tonemapping
    float bloomThreshold = 1.0f;     // HDR luminance threshold for bloom extraction
    float bloomIntensity = 0.12f;    // Bloom add strength
    float bloomRadius = 1.5f;        // Bloom tap radius in pixels
    bool srgbOutputEncoding = true;        // LDR outputs and display: piecewise sRGB encode; false = gamma 2.2
    float referenceBloomIntensity = 0.0f;  // LDR outputs: Path-Tracing reference bloom; 0 disables
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
    // Reference capture calibration for depth-zero HDR misses only. Emerald
    // MASK visibility holes retain depth zero; reflection/refraction stays physical.
    float environmentBackgroundScale = 1.0f;
    // >0: HDR texels brighter than this luminance are scaled down to it at load,
    // so a scene can replace an HDR's blurred sun with an explicit sun light.
    float environmentSunClamp = 0.0f;
    bool environmentMapDirty = false;   // UI flag to request an environment reload

    // Firefly clamping / variance control
    bool fireflyClampEnabled = true;
    FireflyClampMode fireflyClampMode = FireflyClampMode::Luminance;
    float fireflyClampFactor = 32.0f;
    float fireflyClampFloor = 4.0f;
    float throughputClamp = 32.0f;
    float minSpecularPdf = 0.0f;
    float fireflyClampMaxContribution = 1000.0f;

    // Denoising settings
    bool denoiseEnabled = false;        // Enable OIDN denoising post-processing
    uint32_t denoiseFilterType = 0;     // 0=RT, 1=RTLightmap
    bool denoiseUseAlbedo = true;       // Use albedo auxiliary buffer for denoising
    bool denoiseUseNormal = true;       // Use normal auxiliary buffer for denoising
    bool denoiseHdr = true;             // Treat denoise input as linear HDR radiance
    uint32_t denoiseFrequency = 4;      // Denoise every N frames (1=every frame, 4=every 4th frame, etc)

    // GPU SVGF denoising with temporal luminance moments, variance-guided
    // a-trous filtering, and normal/albedo edge stops. This is disabled by
    // default and intentionally separate from the OIDN CPU postprocess.
    bool svgfDenoiseEnabled = false;
    bool svgfDenoiseDirect = true;
    bool svgfDenoiseIndirect = true;
    uint32_t svgfAtrousIterations = 1;
    float svgfNormalSigma = 64.0f;
    float svgfAlbedoSigma = 16.0f;
    float svgfLuminanceSigma = 4.0f;

    void markViewportQualityPresetOverride() {
        if (viewportQualityPreset != ViewportQualityPreset::Custom) {
            viewportQualityPresetOverridden = true;
            viewportQualityPreset = ViewportQualityPreset::Custom;
        }
    }

    void applyViewportQualityPreset(ViewportQualityPreset preset) {
        viewportQualityPreset = preset;
        viewportQualityPresetOverridden = false;
        switch (preset) {
            case ViewportQualityPreset::Interactive:
                samplesPerFrame = 1u;
                maxDepth = 32u;
                renderScale = 0.5f;
                executionMode = ExecutionMode::Megakernel;
                wavefrontSchedulingPolicy = WavefrontSchedulingPolicy::Preview;
                denoiseEnabled = true;
                denoiseFrequency = 8u;
                svgfDenoiseEnabled = false;
                svgfDenoiseDirect = true;
                svgfDenoiseIndirect = true;
                svgfAtrousIterations = 1u;
                break;
            case ViewportQualityPreset::Balanced:
                samplesPerFrame = 2u;
                maxDepth = 50u;
                renderScale = 0.75f;
                executionMode = ExecutionMode::Megakernel;
                wavefrontSchedulingPolicy = WavefrontSchedulingPolicy::Final;
                denoiseEnabled = true;
                denoiseFrequency = 4u;
                svgfDenoiseEnabled = false;
                svgfDenoiseDirect = true;
                svgfDenoiseIndirect = true;
                svgfAtrousIterations = 1u;
                break;
            case ViewportQualityPreset::FinalPreview:
                samplesPerFrame = 4u;
                maxDepth = 50u;
                renderScale = 1.0f;
                executionMode = ExecutionMode::Megakernel;
                wavefrontSchedulingPolicy = WavefrontSchedulingPolicy::Offline;
                denoiseEnabled = true;
                denoiseFrequency = 2u;
                svgfDenoiseEnabled = false;
                svgfDenoiseDirect = true;
                svgfDenoiseIndirect = true;
                svgfAtrousIterations = 1u;
                break;
            case ViewportQualityPreset::Custom:
            default:
                viewportQualityPresetOverridden = true;
                break;
        }
    }
};

// Radiance cache support matrix (see the README's radiance cache notes). The cache trains on the radiance a
// plain path sample adds after a cached vertex; a feature that replaces that suffix with something else makes the
// training target meaningless, so the cache refuses it. Returns nullptr when the cache may run with these settings
// (or is off), otherwise a short reason code. A refused cache is off for the whole render on every backend; the
// reason is printed and written to the metadata (radiance_cache_refusal_reason).
inline const char* RadianceCacheRefusalReason(const RenderSettings& settings) {
    if (settings.radianceCacheMode != RenderSettings::RadianceCacheMode::Prototype) {
        return nullptr;
    }
    if (settings.restirGiMode != RenderSettings::RestirGiMode::Off) {
        return "restir_gi";
    }
    if (settings.restirPtMode == RenderSettings::RestirPtMode::ProductionPathReuse ||
        settings.restirPtMode == RenderSettings::RestirPtMode::EnhancedUnbiased ||
        settings.restirPtMode == RenderSettings::RestirPtMode::EnhancedBiased) {
        return "restir_pt_path_reuse";
    }
    return nullptr;
}

inline const char* RadianceCacheRefusalMessage(const char* reason) {
    if (!reason) {
        return "";
    }
    const std::string code(reason);
    if (code == "restir_gi") {
        return "ReSTIR GI replaces the BSDF sample before the cache query, so a training path's suffix is not a "
               "plain path sample";
    }
    if (code == "restir_pt_path_reuse") {
        return "ReSTIR PT path reuse substitutes reused path samples, so a training path's suffix is not a plain "
               "path sample";
    }
    return "unsupported combination";
}

// The cache mode the renderers run: the requested mode unless the combination is refused.
inline bool RadianceCacheActive(const RenderSettings& settings) {
    return settings.radianceCacheMode == RenderSettings::RadianceCacheMode::Prototype &&
           RadianceCacheRefusalReason(settings) == nullptr;
}

// ReSTIR DI reservoirs carry no medium state, so with volumeTransportMode=homogeneous the Metal renderers do not run
// restir_di or restir_di_regir_hybrid: they render baseline_emissive, the plain surface NEE of every light (suns, mesh
// emitters, rect lights; sample_direct_light_baseline). Volume NEE at scatter points does not depend on the mode.
// Embree ignores volumes and runs the requested mode. The renderers log the demotion, and headless runs write it to
// the render metadata (volume_direct_light_fallback), the ReSTIR DI path status (volume_baseline_emissive) and the
// metrics JSON fallback report (volume_restir_di_interaction).
inline bool RestirDiDemotedByVolume(const RenderSettings& settings) {
    return settings.volumeTransportMode == RenderSettings::VolumeTransportMode::Homogeneous &&
           (settings.directLightMode == RenderSettings::DirectLightMode::RestirDi ||
            settings.directLightMode == RenderSettings::DirectLightMode::RestirDiRegirHybrid);
}

// The direct-light mode the Metal shaders run: the requested mode, or baseline_emissive for a ReSTIR DI mode that
// homogeneous volumes demote.
inline RenderSettings::DirectLightMode MetalDirectLightMode(const RenderSettings& settings) {
    return RestirDiDemotedByVolume(settings) ? RenderSettings::DirectLightMode::BaselineEmissive
                                             : settings.directLightMode;
}

}  // namespace PathTracer
