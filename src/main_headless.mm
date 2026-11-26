#import <Foundation/Foundation.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "MetalRenderer.h"
#include "headless/EmbreeHeadlessRenderer.h"
#include "renderer/SettingsUtils.h"
#include "headless/MetalHeadlessRenderer.h"
#include "headless/IHeadlessRenderer.h"
#include "renderer/RenderSettings.h"
#include "renderer/SpectralSensor.h"
#include "renderer/ImageWriter.h"
#include "renderer/OidnCpuDenoiser.h"
#include "renderer/MetalContext.h"
#include "renderer/Pipelines.h"
#include "renderer/SceneManager.h"
#include "renderer/SceneResources.h"

namespace fs = std::filesystem;
constexpr double kPi = 3.14159265358979323846;
constexpr double kMiB = 1024.0 * 1024.0;

bool EnvFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return false;
    }
    std::string lower;
    for (const char* p = value; *p; ++p) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*p))));
    }
    return lower != "0" && lower != "false" && lower != "no";
}

bool WavefrontShadeIndirectEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_WAVEFRONT_SHADE_INDIRECT");
    return enabled;
}

bool WavefrontShadeIndirectProbeEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_WAVEFRONT_SHADE_INDIRECT_PROBE");
    return enabled;
}

bool WavefrontShadeSurfaceProbeEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_WAVEFRONT_SHADE_SURFACE_PROBE");
    return enabled;
}

bool WavefrontShadeMetadataProbeEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_WAVEFRONT_SHADE_METADATA_PROBE");
    return enabled;
}

bool WavefrontShadeScalarProbeEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_WAVEFRONT_SHADE_SCALAR_PROBE");
    return enabled;
}

bool WavefrontShadeBsdfProbeEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_WAVEFRONT_SHADE_BSDF_PROBE");
    return enabled;
}

bool WavefrontShadeClosureProbeEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_WAVEFRONT_SHADE_CLOSURE_PROBE");
    return enabled;
}

bool WavefrontShadowIndirectEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_WAVEFRONT_SHADOW_INDIRECT");
    return enabled;
}

uint64_t CurrentAllocatedSizeBytes(id<MTLDevice> device) {
    if (!device) {
        return 0u;
    }
    if ([device respondsToSelector:@selector(currentAllocatedSize)]) {
        return static_cast<uint64_t>(device.currentAllocatedSize);
    }
    return 0u;
}

void LogCurrentAllocatedSize(const char* label, id<MTLDevice> device) {
    const uint64_t bytes = CurrentAllocatedSizeBytes(device);
    if (bytes == 0u) {
        return;
    }
    NSLog(@"[Metal] currentAllocatedSize %s %.1f MiB",
          label,
          static_cast<double>(bytes) / kMiB);
}

enum class RenderProfile {
    Custom = 0,
    Preview = 1,
    Lookdev = 2,
    Final = 3,
    Reference = 4,
    Debug = 5,
};

enum class RenderMode {
    Auto = 0,
    MetalFinalFrame = 1,
    CpuReferenceHeadless = 2,
};

struct CliOptions {
    enum class FireflyClampMode : uint32_t {
        Off = 0,
        Luminance = 1,
        MaxComponent = 2,
    };

    std::string scene;
    bool sceneProvided = false;
    bool sceneIsPath = false;

    std::string outputPath;
    bool outputProvided = false;

    uint32_t width = 0;
    uint32_t height = 0;
    bool widthSet = false;
    bool heightSet = false;

    uint32_t sppTotal = 1024;
    bool sppTotalSet = false;
    uint32_t maxDepth = 0;
    bool maxDepthSet = false;

    uint32_t threads = 0;
    bool threadsSet = false;

    uint32_t seed = 0;
    bool seedSet = false;

    float envRotationDegrees = 0.0f;
    bool envRotationSet = false;

    float envIntensity = 0.0f;
    bool envIntensitySet = false;
    float envBackgroundScale = 1.0f;
    bool envBackgroundScaleSet = false;

    uint32_t tonemapMode = 0;
    bool tonemapSet = false;

    float exposure = 0.0f;
    bool exposureSet = false;
    FireflyClampMode fireflyClampMode = FireflyClampMode::Off;
    bool fireflyClampModeSet = false;
    float fireflyClampValue = 0.0f;
    bool fireflyClampValueSet = false;

    bool enableSoftwareRayTracing = false;
    bool enableSoftwareRayTracingSet = false;

    bool enableMnee = false;
    bool enableMneeSet = false;

    bool enableSpecularNee = true;
    bool enableSpecularNeeSet = false;
    PathTracer::RenderSettings::ExecutionMode executionMode =
        PathTracer::RenderSettings::ExecutionMode::Megakernel;
    bool executionModeSet = false;
    PathTracer::RenderSettings::WavefrontSchedulingPolicy wavefrontSchedulingPolicy =
        PathTracer::RenderSettings::WavefrontSchedulingPolicy::Preview;
    bool wavefrontSchedulingPolicySet = false;
    bool wavefrontCompactionEnabled = true;
    bool wavefrontCompactionEnabledSet = false;
    bool wavefrontShadeIndirectEnabled = false;
    bool wavefrontShadeIndirectEnabledSet = false;
    bool wavefrontShadeIndirectProbeEnabled = false;
    bool wavefrontShadeIndirectProbeEnabledSet = false;
    bool wavefrontShadeSurfaceProbeEnabled = false;
    bool wavefrontShadeSurfaceProbeEnabledSet = false;
    bool wavefrontShadeMetadataProbeEnabled = false;
    bool wavefrontShadeMetadataProbeEnabledSet = false;
    bool wavefrontShadeScalarProbeEnabled = false;
    bool wavefrontShadeScalarProbeEnabledSet = false;
    bool wavefrontShadeClosureProbeEnabled = false;
    bool wavefrontShadeClosureProbeEnabledSet = false;
    bool wavefrontShadeBsdfProbeEnabled = false;
    bool wavefrontShadeBsdfProbeEnabledSet = false;
    bool wavefrontShadowIndirectEnabled = false;
    bool wavefrontShadowIndirectEnabledSet = false;
    PathTracer::RenderSettings::DirectLightMode directLightMode =
        PathTracer::RenderSettings::DirectLightMode::LegacyRect;
    bool directLightModeSet = false;
    PathTracer::RenderSettings::RestirGiMode restirGiMode =
        PathTracer::RenderSettings::RestirGiMode::Off;
    bool restirGiModeSet = false;
    PathTracer::RenderSettings::PathGuidingMode pathGuidingMode =
        PathTracer::RenderSettings::PathGuidingMode::Off;
    bool pathGuidingModeSet = false;
    float pathGuidingStrength = 0.35f;
    bool pathGuidingStrengthSet = false;
    float pathGuidingCellSize = 1.0f;
    bool pathGuidingCellSizeSet = false;
    uint32_t pathGuidingTrainingInterval = 1;
    bool pathGuidingTrainingIntervalSet = false;
    PathTracer::RenderSettings::RestirPtMode restirPtMode =
        PathTracer::RenderSettings::RestirPtMode::Off;
    bool restirPtModeSet = false;
    uint32_t restirPtMaxReservoirs = 4096;
    bool restirPtMaxReservoirsSet = false;
    bool restirPtDebug = false;
    bool restirPtDebugSet = false;
    float restirPtReuseStrength = 0.25f;
    bool restirPtReuseStrengthSet = false;
    PathTracer::RenderSettings::RadianceCacheMode radianceCacheMode =
        PathTracer::RenderSettings::RadianceCacheMode::Off;
    bool radianceCacheModeSet = false;
    float radianceCacheCellSize = 1.0f;
    bool radianceCacheCellSizeSet = false;
    uint32_t radianceCacheMinDepth = 2;
    bool radianceCacheMinDepthSet = false;
    float radianceCacheMinConfidence = 0.35f;
    bool radianceCacheMinConfidenceSet = false;
    uint32_t radianceCacheTrainingInterval = 1;
    bool radianceCacheTrainingIntervalSet = false;
    float radianceCacheMaxRadiance = 16.0f;
    bool radianceCacheMaxRadianceSet = false;
    bool radianceCacheDebug = false;
    bool radianceCacheDebugSet = false;
    PathTracer::RenderSettings::CausticTransportMode causticTransportMode =
        PathTracer::RenderSettings::CausticTransportMode::Off;
    bool causticTransportModeSet = false;
    uint32_t causticMaxVertices = 4096;
    bool causticMaxVerticesSet = false;
    PathTracer::RenderSettings::VolumeTransportMode volumeTransportMode =
        PathTracer::RenderSettings::VolumeTransportMode::Off;
    bool volumeTransportModeSet = false;
    PathTracer::RenderSettings::VolumePhaseFunction volumePhaseFunction =
        PathTracer::RenderSettings::VolumePhaseFunction::Isotropic;
    bool volumePhaseFunctionSet = false;
    simd::float3 volumeSigmaA = simd_make_float3(0.0f, 0.0f, 0.0f);
    bool volumeSigmaASet = false;
    simd::float3 volumeSigmaS = simd_make_float3(0.0f, 0.0f, 0.0f);
    bool volumeSigmaSSet = false;
    simd::float3 volumeEmission = simd_make_float3(0.0f, 0.0f, 0.0f);
    bool volumeEmissionSet = false;
    float volumeAnisotropy = 0.0f;
    bool volumeAnisotropySet = false;
    uint32_t volumeMaxScatteringEvents = 8;
    bool volumeMaxScatteringEventsSet = false;
    bool volumeNeeEnabled = true;
    bool volumeNeeEnabledSet = false;
    bool volumeDebug = false;
    bool volumeDebugSet = false;
    PathTracer::RenderSettings::SpectralRenderingMode spectralRenderingMode =
        PathTracer::RenderSettings::SpectralRenderingMode::Off;
    bool spectralRenderingModeSet = false;
    float spectralMinWavelengthNm = 380.0f;
    bool spectralMinWavelengthSet = false;
    float spectralMaxWavelengthNm = 720.0f;
    bool spectralMaxWavelengthSet = false;
    float spectralDispersionStrength = 0.012f;
    bool spectralDispersionStrengthSet = false;
    bool spectralDebug = false;
    bool spectralDebugSet = false;
    uint32_t spectralWavelengthsPerPath = 3;
    bool spectralWavelengthsPerPathSet = false;
    PathTracer::RenderSettings::MlBackendMode mlBackendMode =
        PathTracer::RenderSettings::MlBackendMode::Off;
    bool mlBackendModeSet = false;
    std::string mlModelPackagePath;
    bool mlModelPackagePathSet = false;
    bool mlDebug = false;
    bool mlDebugSet = false;
    uint32_t risCandidateCount = 8;
    bool risCandidateCountSet = false;
    uint32_t spatialReuseNeighborCount = 4;
    bool spatialReuseNeighborCountSet = false;
    float worldReuseCellSize = 0.5f;
    bool worldReuseCellSizeSet = false;
    bool biasCheck = false;
    float biasCheckRmseThreshold = 1.0e-3f;
    bool validateRis = false;
    uint32_t validateRisFrames = 512;
    bool validateRisFramesSet = false;
    float validateRisPsnrThreshold = 35.0f;
    bool validateRisPsnrThresholdSet = false;
    std::string validateRisReferencePath;
    bool validateRisReferenceProvided = false;

    bool pbrMetrics = false;
    std::string pbrMetricsJsonPath;
    bool pbrMetricsJsonProvided = false;
    std::string materialMrDebugJsonPath;
    bool materialMrDebugJsonProvided = false;
    std::string directLightAuditJsonPath;
    bool directLightAuditJsonProvided = false;
    bool restirDebug = false;
    bool restirDebugSet = false;
    PathTracer::RenderSettings::RestirDebugView restirDebugView =
        PathTracer::RenderSettings::RestirDebugView::Beauty;
    bool restirDebugViewSet = false;
    bool restirDebugCounters = false;
    bool restirDebugCountersSet = false;
    std::string restirDebugMetricsJsonPath;
    bool restirDebugMetricsJsonProvided = false;
    std::string cameraSearchReportPath;
    bool cameraSearchReportProvided = false;
    RenderMode renderMode = RenderMode::Auto;
    bool renderModeSet = false;
    RenderProfile renderProfile = RenderProfile::Custom;
    bool renderProfileSet = false;
    bool outputMetadata = true;
    bool outputMetadataSet = false;
    std::string outputMetadataJsonPath;
    bool outputMetadataJsonProvided = false;
    std::string settingsJsonPath;
    bool settingsJsonProvided = false;
    std::string renderQueueItemJsonPath;
    bool renderQueueItemJsonProvided = false;
    std::string runRenderQueueItemJsonPath;
    bool runRenderQueueItemJsonProvided = false;
    std::string debugBundleDir;
    bool debugBundleDirProvided = false;
    std::string checkpointManifestJsonPath;
    bool checkpointManifestJsonProvided = false;
    std::string resumeCheckpointJsonPath;
    bool resumeCheckpointJsonProvided = false;
    // Sample checkpoints: export the accumulation every N samples (Metal); a partial manifest resumes from it.
    uint32_t checkpointEverySamples = 0;
    std::string resumeAccumulationStatePath;
    bool resumeFromAccumulationState = false;
    std::string tileManifestJsonPath;
    bool tileManifestJsonProvided = false;
    uint32_t tileWidth = 0;
    uint32_t tileHeight = 0;
    bool tileSizeSet = false;

    std::string formatString = "exr";
    PathTracer::ImageFileFormat format = PathTracer::ImageFileFormat::EXR;

    bool denoiseEnabled = false;
    bool denoiseEnabledSet = false;
    bool denoiseUseAlbedo = true;
    bool denoiseUseAlbedoSet = false;
    bool denoiseUseNormal = true;
    bool denoiseUseNormalSet = false;
    bool denoiseHdr = true;
    bool denoiseHdrSet = false;
    bool writeDenoiseGuides = false;
    bool svgfDenoiseEnabled = false;
    bool svgfDenoiseEnabledSet = false;
    bool svgfDenoiseDirect = true;
    bool svgfDenoiseDirectSet = false;
    bool svgfDenoiseIndirect = true;
    bool svgfDenoiseIndirectSet = false;
    uint32_t svgfAtrousIterations = 1;
    bool svgfAtrousIterationsSet = false;
    float svgfNormalSigma = 64.0f;
    bool svgfNormalSigmaSet = false;
    float svgfAlbedoSigma = 16.0f;
    bool svgfAlbedoSigmaSet = false;
    float svgfLuminanceSigma = 4.0f;
    bool svgfLuminanceSigmaSet = false;

    bool verbose = false;
    bool stats = false;
    bool debugLandmarks = false;
    bool validateEmissive = false;
    uint32_t expectedEmissiveCount = 0;
    bool expectedEmissiveCountSet = false;
    PathTracer::TextureBudgetPolicy budgetPolicy = PathTracer::TextureBudgetPolicy::Strict;
    uint32_t textureMaxDim = 0;
    bool textureMaxDimSet = false;
    uint64_t maxTextureBytes = 0;
    bool maxTextureBytesSet = false;
    PathTracer::PilotMode pilotMode = PathTracer::PilotMode::FullClamped;
    bool forceHardwareRayTracing = false;
    bool forceHardwareRayTracingSet = false;
    bool forceSoftwareRayTracing = false;
    bool forceSoftwareRayTracingSet = false;

    bool debugShowBaseColor = false;
    bool debugShowBaseColorSet = false;
    bool debugShowMetallic = false;
    bool debugShowMetallicSet = false;
    bool debugShowRoughness = false;
    bool debugShowRoughnessSet = false;
    bool debugShowAO = false;
    bool debugShowAOSet = false;
    bool debugDisableOrmTexture = false;
    bool debugDisableOrmTextureSet = false;
    std::string dumpMaterialTexturesDir;
    bool dumpMaterialTexturesDirProvided = false;

    HeadlessBackend backend = HeadlessBackend::Metal;
    bool backendSet = false;
#if PT_DEBUG_TOOLS
    bool debugPath = false;
    bool debugPathSet = false;
    uint32_t debugPixelX = 0;
    uint32_t debugPixelY = 0;
    bool debugPixelSet = false;
    uint32_t debugMaxEntries = 128;
    bool debugMaxEntriesSet = false;
    bool debugSpecularOnly = false;
    bool debugSpecularOnlySet = false;
    bool parityAssert = false;
    bool parityAssertSet = false;
    uint32_t parityPixelX = 0;
    uint32_t parityPixelY = 0;
    bool parityPixelSet = false;
    bool parityOncePerFrame = true;
    bool parityOncePerFrameSet = false;
    PathTracer::RenderSettings::ParityAssertMode parityMode =
        PathTracer::RenderSettings::ParityAssertMode::ProbePixelOnly;
    bool parityModeSet = false;
    uint32_t parityTargetDepth = 0;
    bool parityTargetDepthSet = false;
    bool forcePureHWRTForGlass = false;
    bool forcePureHWRTForGlassSet = false;
    bool debugDisableNormalMap = false;
    bool debugDisableNormalMapSet = false;
#endif
};

void PrintUsage(const char* exe) {
    std::cout << "Usage: " << exe << " [options]\n\n"
              << "Required:\n"
              << "  --scene=<id-or-path>          Scene identifier or path to .scene file\n\n"
              << "Rendering overrides:\n"
              << "  --width=<int>                 Override render width (>=8)\n"
              << "  --height=<int>                Override render height (>=8)\n"
              << "  --sppTotal=<int>              Total samples to accumulate (default 1024)\n"
              << "  --spp=<int>                   Alias for --sppTotal\n"
              << "  --renderMode=<metal_final_frame|cpu_reference_headless>\n"
              << "                                 Select user-facing render intention before expert overrides\n"
              << "  --renderProfile=<custom|preview|lookdev|final|reference|debug>\n"
              << "                                 Apply explicit production defaults before CLI overrides\n"
              << "  --maxDepth=<int>              Override max path depth\n"
              << "  --threads=<int>               Max worker threads (Embree only)\n"
              << "  --seed=<int>                  RNG seed base (including 0)\n"
              << "  --enableSoftwareRayTracing[=0|1]  Force software tracing kernels\n"
              << "  --force-hwrt[=0|1]             Require Metal hardware ray tracing\n"
              << "  --force-swrt[=0|1]             Require Metal software BVH tracing\n"
              << "  --executionMode=<megakernel|wavefront>  Select Metal execution path (default megakernel)\n"
              << "  --wavefront-policy=<preview|final|offline|research>  Wavefront scheduling policy\n"
              << "  --wavefront-compaction=<0|1>   Enable explicit wavefront queue compaction pass\n"
              << "  --wavefront-shade-indirect=<0|1> Enable guarded shade active-count indirect debug path\n"
              << "  --wavefront-shade-indirect-probe=<0|1> Run non-mutating shade active-count indirect probe before direct shade\n"
              << "  --wavefront-shade-surface-probe=<0|1> Run non-mutating shade surface-resolution indirect probe before direct shade\n"
              << "  --wavefront-shade-metadata-probe=<0|1> Run non-mutating material-only closure-metadata indirect probe before direct shade\n"
              << "  --wavefront-shade-scalar-probe=<0|1> Run non-mutating material-only scalar indirect probe before direct shade\n"
              << "  --wavefront-shade-closure-probe=<0|1> Run non-mutating shade closure-metadata indirect probe before direct shade\n"
              << "  --wavefront-shade-bsdf-probe=<0|1> Run non-mutating shade BSDF-sampling indirect probe before direct shade\n"
              << "  --wavefront-shadow-indirect=<0|1> Enable guarded compact shadow active-count indirect path\n"
              << "  --enableMnee[=0|1]             Enable MNEE caustics (default 0)\n"
              << "  --enableSpecularNee[=0|1]      Enable specular NEE (default 1)\n\n"
              << "Direct-light and guiding controls:\n"
              << "  --directLightMode=<legacy|baseline_emissive|ris|ris_spatial|ris_temporal|ris_world|ris_regir|restir_di|restir_di_regir_hybrid>\n"
              << "                                 Select direct-light path (default legacy)\n"
              << "  --ris-m=<int>                  RIS candidate count used when directLightMode=ris/ris_spatial/ris_temporal/ris_world/ris_regir/restir_di/restir_di_regir_hybrid (default 8)\n"
              << "  --spatial-reuse-k=<int>        Same-frame neighbor count for directLightMode=ris_spatial (default 4)\n"
              << "  --world-reuse-cell-size=<float> World-grid cell size for directLightMode=ris_world/ris_regir (default 0.5)\n"
              << "  --restirGiMode=<off|restir_gi_production|restir_gi_diffuse>  Production BSDF-backed diffuse/PBR indirect reuse (default off)\n"
              << "  --pathGuidingMode=<off|path_guiding>  Gated MIS path guiding with world-grid directional bins (default off)\n"
              << "  --pathGuidingStrength=<float>  Guided strategy probability scale in [0,1] (default 0.35)\n"
              << "  --pathGuidingCellSize=<float>  World-space path guiding cell size (default 1.0)\n"
              << "  --pathGuidingTrainingInterval=<int>  Train guiding every N frames (default 1)\n"
              << "  --restirPtMode=<off|restir_pt_reservoir_capture|restir_pt_path_reuse|restir_pt_enhanced_unbiased|restir_pt_enhanced_biased>  Production path-reservoir capture/reuse modes (default off)\n"
              << "  --restirPtMaxReservoirs=<int>  Max path reservoir captures per frame (default 4096)\n"
              << "  --restirPtDebug[=0|1]         Enable PT reservoir debug counter path (default 0)\n"
              << "  --restirPtReuseStrength=<float>  Production path reuse strength in [0,1] (default 0.25)\n"
              << "  --radianceCacheMode=<off|radiance_cache>  Gated rough-indirect radiance cache (default off)\n"
              << "  --radianceCacheCellSize=<float>  World-space radiance cache cell size (default 1.0)\n"
              << "  --radianceCacheMinDepth=<int>  Minimum path depth eligible for cache termination (default 2)\n"
              << "  --radianceCacheMinConfidence=<float>  Minimum cache confidence in [0,1] (default 0.35)\n"
              << "  --radianceCacheTrainingInterval=<int>  Train cache every N frames (default 1)\n"
              << "  --radianceCacheMaxRadiance=<float>  Quantized cache radiance clamp (default 16)\n"
              << "  --causticTransportMode=<off|path_space>  path_space: prototype that records environment and rect-light\n"
              << "                                next-event connections; adds no caustic paths (default off)\n"
              << "  --causticMaxVertices=<int>  Bounded caustic path vertex budget (default 4096)\n"
              << "  --volumeTransportMode=<off|homogeneous>  Gated homogeneous participating medium (default off)\n"
              << "  --volumeSigmaA=<r,g,b>       Homogeneous absorption coefficients\n"
              << "  --volumeSigmaS=<r,g,b>       Homogeneous scattering coefficients\n"
              << "  --volumeEmission=<r,g,b>     Homogeneous volume emission\n"
              << "  --volumePhase=<isotropic|hg> Phase function (default isotropic)\n"
              << "  --volumeAnisotropy=<float>   Henyey-Greenstein g in [-0.98,0.98]\n"
              << "  --volumeMaxScatteringEvents=<int>  Bounded medium scatter count (default 8)\n"
              << "  --volumeNee[=0|1]            Enable volume direct-light NEE (default 1)\n"
              << "  --spectralMode=<off|hero_wavelength>  Optional hero-wavelength spectral transport (default off)\n"
              << "  --spectralMinNm=<float>      Spectral sampling lower bound in nm (default 380)\n"
              << "  --spectralMaxNm=<float>      Spectral sampling upper bound in nm (default 720)\n"
              << "  --spectralDispersionStrength=<float>  Cauchy-style dielectric dispersion scale\n"
              << "  --spectralDebug[=0|1]        Enable spectral diagnostics metadata (default 0)\n"
              << "  --spectralWavelengthsPerPath=<1|3>  Wavelengths carried per path in spectral mode (default 3)\n"
              << "  --bias-check[=0|1]             Compare M=1 RIS against baseline emissive NEE for the current scene\n\n"
              << "RIS closure controls:\n"
              << "  --validate-ris[=0|1]           Run RIS closure harness for the current scene\n"
              << "  --validate-ris-frames=<int>    Frame count used for RIS closure harness (default 512)\n"
              << "  --validate-ris-reference=<path> Baseline reference PFM used for RIS comparison\n"
              << "  --validate-ris-psnr=<float>    Minimum PSNR threshold for RIS validation (default 35.0)\n\n"
              << "Runtime reporting:\n"
              << "  --stats[=0|1]                  Periodically print runtime ray statistics\n"
              << "  --validate-emissive[=0|1]      Build emissive mesh inventory, print diagnostics, and exit\n"
              << "  --expected-emissive-count=<n>  Assert emissive inventory count before render loop\n"
              << "  --budget-policy=<strict|warn|ignore>\n"
              << "                                 Memory budget enforcement policy (default strict)\n\n"
              << "Large-scene pilot controls:\n"
              << "  --pilot-mode=<geo_only|albedo_only|full_clamped>\n"
              << "                                 Bistro pilot material policy (default full_clamped)\n"
              << "  --texture-max-dim=<int>        Clamp textures via lower mips before render/upload\n"
              << "  --max-texture-bytes=<bytes>    Texture budget cap used with --budget-policy\n\n"
              << "Environment overrides:\n"
              << "  --envRotation=<deg>           Environment rotation in degrees\n"
              << "  --envIntensity=<float>        Environment intensity multiplier\n"
              << "  --envBackgroundScale=<float>  Depth-zero HDR background capture calibration (default 1)\n\n"
              << "Backend selection:\n"
              << "  --backend=<metal|embree>       Headless backend (default metal)\n"
              << "  --enableEmbree[=0|1]           Alias for --backend embree\n\n"
              << "Tonemapping overrides (for LDR outputs):\n"
              << "  --tonemap=<1|2|3|4|5|6>       1=Linear, 2=ACES, 3=Reinhard, 4=Hable, 5=Exponential,\n"
              << "                                 6=Filmic (Tungsten; built-in gamma, no sRGB/gamma encode)\n"
              << "  --exposure=<float>            Exposure in stops\n\n"
              << "Offline beauty controls:\n"
              << "  --firefly-clamp-mode=<off|luminance|maxcomponent>\n"
              << "                                 Optional offline beauty clamp (default off)\n"
              << "  --firefly-clamp-value=<float> Clamp threshold used when clamp mode is enabled\n\n"
              << "Output controls:\n"
              << "  --output=<path>               Output filename\n"
              << "  --out=<path>                  Alias for --output\n"
              << "  --format=<exr|png|pfm|ppm>    Output format (default exr)\n"
              << "\n"
              << "Denoise controls:\n"
              << "  --denoise[=0|1]               Run OIDN as a CPU postprocess after rendering\n"
              << "  --denoise-use-albedo[=0|1]    Use albedo guide AOV when available (default 1)\n"
              << "  --denoise-use-normal[=0|1]    Use normal guide AOV when available (default 1)\n"
              << "  --writeDenoiseGuides[=0|1]    Also write the OIDN albedo/normal guides as <output>_albedo.exr /\n"
              << "                                <output>_normal.exr (linear), e.g. to denoise merged chunk renders\n"
              << "  --denoise-hdr[=0|1]           Treat denoise input as linear HDR radiance (default 1)\n"
              << "  --svgfDenoise[=0|1]           Run Metal SVGF spatiotemporal variance-guided denoise (default 0)\n"
              << "  --svgf-denoise-direct[=0|1]   Include direct-light component in split SVGF denoise (default 1)\n"
              << "  --svgf-denoise-indirect[=0|1] Include indirect-light component in split SVGF denoise (default 1)\n"
              << "  --svgf-atrous-iterations=<n>  A-trous filter iterations for --svgfDenoise (1-5, default 1)\n"
              << "  --svgf-normal-sigma=<float>   Normal edge weight for --svgfDenoise (default 64)\n"
              << "  --svgf-albedo-sigma=<float>   Albedo edge weight for --svgfDenoise (default 16)\n"
              << "  --svgf-luminance-sigma=<float> Luminance edge weight for --svgfDenoise (default 4)\n"
              << "  --mlBackendMode=<off|reconstruction_noop|reconstruction_tiny> Optional ML reconstruction backend (default off)\n"
              << "  --mlModelPackage=<path>        PTMLRECONv1 package for reconstruction_tiny\n"
              << "  --mlDebug[=0|1]                Include ML backend debug/capability reporting\n"
              << "  --pbrMetrics[=0|1]            Print and write PBR metrics sidecar JSON\n"
              << "  --pbrMetricsJson=<path>       Override PBR metrics JSON path\n"
              << "  --outputMetadata[=0|1]        Write production metadata sidecar JSON (default 1)\n"
              << "  --outputMetadataJson=<path>   Override production metadata sidecar path\n"
              << "  --settingsJson=<path>         Write deterministic final settings JSON\n"
              << "  --renderQueueItemJson=<path>  Write deterministic render queue item JSON\n"
              << "  --runRenderQueueItemJson=<path> Run a deterministic render queue item JSON\n"
              << "  --debugBundleDir=<dir>        Write metadata/settings/metrics into one debug bundle directory\n"
              << "  --checkpointManifestJson=<path> Write completed-render checkpoint manifest JSON\n"
              << "  --resumeCheckpointJson=<path> Continue an interrupted render from a partial checkpoint manifest\n"
              << "                                (--checkpointEverySamples); from a completed one, write the output\n"
              << "                                from its saved accumulation, or re-run when it has none\n"
              << "  --checkpointEverySamples=<int> Metal: every N samples, save the accumulation to <output>.accum\n"
              << "                                and a partial checkpoint manifest, so an interrupted render resumes\n"
              << "                                where it stopped. Needs --output; refused with temporal features\n"
              << "  --tileManifestJson=<path>     Write deterministic tiled/chunk render plan JSON\n"
              << "  --tileSize=<w,h>              Tile dimensions recorded in tile manifests\n"
              << "  --materialMrDebugJson=<path>  Write first-hit Material_MR debug JSON\n"
              << "  --directLightAuditJson=<path> Write deterministic per-light NEE audit JSON\n"
              << "  --restirDebug[=0|1]           Enable ReSTIR debug inspector plumbing (default 0)\n"
              << "  --restirDebugView=<beauty|candidate_source_id|reservoir_confidence|regir_cell_id|path_guiding_used_mask|restir_pt_reuse_mask|svgf_variance|nan_inf_mask|radiance_cache|caustic_transport>\n"
              << "                                 Select derived ReSTIR debug output view (default beauty)\n"
              << "  --restirDebugCounters[=0|1]   Include ReSTIR debug counter audit state (default 0)\n"
              << "  --restirDebugMetricsJson=<path> Write compact debug inspector JSON\n"
              << "  --cameraSearchReport=<path>   Write runtime landmark/projection JSON for camera search\n"
              << "  --verbose                     Print progress updates with percentage and ETA\n"
              << "  --debug-landmarks            Dump runtime landmark transforms and render emissive markers\n"
              << "  --debugShowBaseColor[=0|1]   Show raw sampled PBR baseColor instead of beauty shading\n"
              << "  --debugShowMetallic[=0|1]    Show metallic channel instead of beauty shading\n"
              << "  --debugShowRoughness[=0|1]   Show roughness channel instead of beauty shading\n"
              << "  --debugShowAO[=0|1]          Show ambient occlusion channel instead of beauty shading\n"
              << "  --debugDisableOrmTexture[=0|1] Disable ORM texture use for diagnostics\n"
              << "  --dumpMaterialTexturesDir=<path>\n"
              << "                               Dump runtime material texture bindings and mip data for diagnostics\n"
#if PT_DEBUG_TOOLS
              << "  --debugPath[=0|1]             Enable single-pixel path debug capture\n"
              << "  --debugPixel=<x,y>            Pixel to capture when debugPath is enabled\n"
              << "  --debugMaxEntries=<int>       Max debug events to record (default 128)\n"
              << "  --parityAssert[=0|1]          Enable HWRT/SWRT parity probe for one pixel\n"
              << "  --parityPixel=<x,y>           Pixel to probe when parityAssert is enabled\n"
              << "  --parityOncePerFrame[=0|1]    Record one parity event or all matching events\n"
              << "  --parityMode=<probe|medium>   Probe depth 0 or first in-medium boundary\n"
              << "  --parityDepth=<int>           Target depth when parityMode=probe (default 0)\n"
              << "  --forcePureHWRTForGlass[=0|1] Disable debug SWRT glass fallback paths\n"
              << "  --debugDisableNormalMap[=0|1] Disable normal maps for diagnostics\n"
              << "  --debugSpecularOnly[=0|1]     Disable diffuse BSDFs for diagnostics\n"
#endif
              << "  --help                        Show this message\n\n"
              << "Examples:\n"
              << "  " << exe << " --scene=dragons_rtow --width=3840 --height=2160 --sppTotal=4096 \\\n"
              << "     --output=renders/dragons_4k.exr\n"
              << "  " << exe << " --scene=assets/dragon.scene --enableSoftwareRayTracing=1 \\\n"
              << "     --envRotation=30 --envIntensity=1.5 --sppTotal=1024 --format=png\n"
              << "  " << exe << " --backend=embree --scene=assets/plastic.scene --sppTotal=32 \\\n"
              << "     --denoise=1 --denoise-use-albedo=1 --denoise-use-normal=1 --output=renders/plastic_denoised.exr\n";
}

bool ParseBoolFlag(const std::string& value, bool defaultIfEmpty, bool& outValue) {
    if (value.empty()) {
        outValue = defaultIfEmpty;
        return true;
    }
    std::string lower;
    lower.reserve(value.size());
    for (char c : value) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower == "1" || lower == "true" || lower == "yes") {
        outValue = true;
        return true;
    }
    if (lower == "0" || lower == "false" || lower == "no") {
        outValue = false;
        return true;
    }
    return false;
}

bool ParsePixelCoord(const std::string& value, uint32_t& outX, uint32_t& outY) {
    auto comma = value.find(',');
    if (comma == std::string::npos || comma == 0 || comma + 1 >= value.size()) {
        return false;
    }
    try {
        outX = static_cast<uint32_t>(std::stoul(value.substr(0, comma)));
        outY = static_cast<uint32_t>(std::stoul(value.substr(comma + 1)));
        return true;
    } catch (...) {
        return false;
    }
}

std::optional<simd::float3> ParseFloat3Option(std::string value);

std::string NSStringToStdString(NSString* value) {
    return value ? std::string([value UTF8String]) : std::string();
}

NSDictionary* ReadJsonDictionary(const fs::path& path, std::string& error) {
    NSString* nsPath = [NSString stringWithUTF8String:path.string().c_str()];
    NSData* data = [NSData dataWithContentsOfFile:nsPath];
    if (!data) {
        error = "Failed to read JSON file: " + path.string();
        return nil;
    }

    NSError* nsError = nil;
    id root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&nsError];
    if (!root || ![root isKindOfClass:[NSDictionary class]]) {
        error = "JSON root must be an object: " + path.string();
        if (nsError) {
            error += " (" + NSStringToStdString([nsError localizedDescription]) + ")";
        }
        return nil;
    }
    return (NSDictionary*)root;
}

NSDictionary* JsonObject(NSDictionary* dictionary, NSString* key) {
    id value = [dictionary objectForKey:key];
    return [value isKindOfClass:[NSDictionary class]] ? (NSDictionary*)value : nil;
}

std::optional<std::string> JsonString(NSDictionary* dictionary, NSString* key) {
    id value = [dictionary objectForKey:key];
    if (![value isKindOfClass:[NSString class]]) {
        return std::nullopt;
    }
    return NSStringToStdString((NSString*)value);
}

std::optional<uint32_t> JsonUInt32(NSDictionary* dictionary, NSString* key) {
    id value = [dictionary objectForKey:key];
    if (![value isKindOfClass:[NSNumber class]]) {
        return std::nullopt;
    }
    return static_cast<uint32_t>(std::max<long long>([(NSNumber*)value longLongValue], 0));
}

std::optional<uint64_t> JsonUInt64(NSDictionary* dictionary, NSString* key) {
    id value = [dictionary objectForKey:key];
    if (![value isKindOfClass:[NSNumber class]]) {
        return std::nullopt;
    }
    return static_cast<uint64_t>(std::max<long long>([(NSNumber*)value longLongValue], 0));
}

std::optional<float> JsonFloat(NSDictionary* dictionary, NSString* key) {
    id value = [dictionary objectForKey:key];
    if (![value isKindOfClass:[NSNumber class]]) {
        return std::nullopt;
    }
    return static_cast<float>([(NSNumber*)value doubleValue]);
}

std::optional<bool> JsonBool(NSDictionary* dictionary, NSString* key) {
    id value = [dictionary objectForKey:key];
    if (![value isKindOfClass:[NSNumber class]]) {
        return std::nullopt;
    }
    return [(NSNumber*)value boolValue] ? true : false;
}

bool ApplyBudgetPolicyName(const std::string& value, CliOptions& options, std::string& error) {
    if (value == "strict") {
        options.budgetPolicy = PathTracer::TextureBudgetPolicy::Strict;
    } else if (value == "warn") {
        options.budgetPolicy = PathTracer::TextureBudgetPolicy::Warn;
    } else if (value == "ignore") {
        options.budgetPolicy = PathTracer::TextureBudgetPolicy::Ignore;
    } else {
        error = "Invalid budget_policy in JSON: " + value;
        return false;
    }
    return true;
}

bool ApplyPilotModeName(const std::string& value, CliOptions& options, std::string& error) {
    if (value == "full_clamped") {
        options.pilotMode = PathTracer::PilotMode::FullClamped;
    } else if (value == "geo_only") {
        options.pilotMode = PathTracer::PilotMode::GeoOnly;
    } else if (value == "albedo_only") {
        options.pilotMode = PathTracer::PilotMode::AlbedoOnly;
    } else {
        error = "Invalid pilot_mode in JSON: " + value;
        return false;
    }
    return true;
}

bool ApplyRenderProfileName(const std::string& value, CliOptions& options, std::string& error) {
    std::string lower;
    lower.reserve(value.size());
    for (char c : value) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower == "custom") {
        options.renderProfile = RenderProfile::Custom;
    } else if (lower == "preview") {
        options.renderProfile = RenderProfile::Preview;
    } else if (lower == "lookdev") {
        options.renderProfile = RenderProfile::Lookdev;
    } else if (lower == "final") {
        options.renderProfile = RenderProfile::Final;
    } else if (lower == "reference") {
        options.renderProfile = RenderProfile::Reference;
    } else if (lower == "debug") {
        options.renderProfile = RenderProfile::Debug;
    } else {
        error = "Invalid render_profile in JSON: " + value;
        return false;
    }
    options.renderProfileSet = options.renderProfile != RenderProfile::Custom;
    return true;
}

bool ApplyRenderModeName(const std::string& value, CliOptions& options, std::string& error) {
    std::string lower;
    lower.reserve(value.size());
    for (char c : value) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower == "auto") {
        options.renderMode = RenderMode::Auto;
        options.renderModeSet = false;
    } else if (lower == "metal_final_frame") {
        options.renderMode = RenderMode::MetalFinalFrame;
        options.renderModeSet = true;
    } else if (lower == "cpu_reference_headless") {
        options.renderMode = RenderMode::CpuReferenceHeadless;
        options.renderModeSet = true;
    } else {
        error = "Invalid render_mode in JSON: " + value;
        return false;
    }
    return true;
}

bool ApplyRenderModeDefaults(CliOptions& options, std::string& error) {
    if (!options.renderModeSet) {
        options.renderMode =
            options.backend == HeadlessBackend::Embree ? RenderMode::CpuReferenceHeadless
                                                       : RenderMode::MetalFinalFrame;
        return true;
    }

    switch (options.renderMode) {
        case RenderMode::MetalFinalFrame:
            if (options.backendSet && options.backend != HeadlessBackend::Metal) {
                error = "--renderMode=metal_final_frame requires --backend=metal";
                return false;
            }
            options.backend = HeadlessBackend::Metal;
            break;
        case RenderMode::CpuReferenceHeadless:
            if (options.backendSet && options.backend != HeadlessBackend::Embree) {
                error = "--renderMode=cpu_reference_headless requires --backend=embree";
                return false;
            }
            options.backend = HeadlessBackend::Embree;
            break;
        case RenderMode::Auto:
        default:
            options.renderMode =
                options.backend == HeadlessBackend::Embree ? RenderMode::CpuReferenceHeadless
                                                           : RenderMode::MetalFinalFrame;
            break;
    }
    return true;
}

bool ApplyRenderQueueDictionary(NSDictionary* root, CliOptions& options, std::string& error) {
    if (NSDictionary* queueItem = JsonObject(root, @"queue_item")) {
        root = queueItem;
    }

    if (auto scene = JsonString(root, @"scene")) {
        options.scene = *scene;
        options.sceneProvided = true;
    }
    if (auto backend = JsonString(root, @"backend")) {
        if (*backend == "metal") {
            options.backend = HeadlessBackend::Metal;
        } else if (*backend == "embree") {
            options.backend = HeadlessBackend::Embree;
        } else {
            error = "Invalid backend in render queue item JSON: " + *backend;
            return false;
        }
        options.backendSet = true;
    }
    if (auto renderMode = JsonString(root, @"render_mode")) {
        if (!ApplyRenderModeName(*renderMode, options, error)) {
            return false;
        }
    }
    if (auto output = JsonString(root, @"output")) {
        options.outputPath = *output;
        options.outputProvided = true;
    }
    if (auto format = JsonString(root, @"format")) {
        options.formatString = *format;
    }
    if (auto spp = JsonUInt32(root, @"spp_total")) {
        options.sppTotal = std::max(*spp, 1u);
        options.sppTotalSet = true;
    }
    if (auto profile = JsonString(root, @"render_profile")) {
        if (!ApplyRenderProfileName(*profile, options, error)) {
            return false;
        }
    }
    if (auto outputMetadata = JsonBool(root, @"output_metadata")) {
        options.outputMetadata = *outputMetadata;
        options.outputMetadataSet = true;
    }
    if (auto outputMetadataJson = JsonString(root, @"output_metadata_json")) {
        options.outputMetadataJsonPath = *outputMetadataJson;
        options.outputMetadataJsonProvided = true;
    }
    if (auto settingsJson = JsonString(root, @"settings_json")) {
        options.settingsJsonPath = *settingsJson;
        options.settingsJsonProvided = true;
    }
    if (auto queueJson = JsonString(root, @"render_queue_item_json")) {
        options.renderQueueItemJsonPath = *queueJson;
        options.renderQueueItemJsonProvided = true;
    }
    if (auto debugBundle = JsonString(root, @"debug_bundle_dir")) {
        options.debugBundleDir = *debugBundle;
        options.debugBundleDirProvided = true;
    }
    if (auto checkpointJson = JsonString(root, @"checkpoint_manifest_json")) {
        options.checkpointManifestJsonPath = *checkpointJson;
        options.checkpointManifestJsonProvided = true;
    }
    if (auto tileJson = JsonString(root, @"tile_manifest_json")) {
        options.tileManifestJsonPath = *tileJson;
        options.tileManifestJsonProvided = true;
    }
    if (auto pbrMetrics = JsonBool(root, @"pbr_metrics")) {
        options.pbrMetrics = *pbrMetrics;
    }
    if (auto pbrMetricsJson = JsonString(root, @"pbr_metrics_json")) {
        options.pbrMetricsJsonPath = *pbrMetricsJson;
        options.pbrMetricsJsonProvided = true;
    }
    if (NSDictionary* tileSize = JsonObject(root, @"tile_size")) {
        if (auto width = JsonUInt32(tileSize, @"width")) {
            options.tileWidth = std::max(*width, 1u);
            options.tileSizeSet = true;
        }
        if (auto height = JsonUInt32(tileSize, @"height")) {
            options.tileHeight = std::max(*height, 1u);
            options.tileSizeSet = true;
        }
        if (options.tileSizeSet) {
            if (options.tileWidth == 0u) {
                options.tileWidth = 512u;
            }
            if (options.tileHeight == 0u) {
                options.tileHeight = 512u;
            }
        }
    }

    NSDictionary* settings = JsonObject(root, @"settings");
    if (settings) {
        if (auto width = JsonUInt32(settings, @"render_width")) {
            options.width = std::max(*width, 8u);
            options.widthSet = true;
        }
        if (auto height = JsonUInt32(settings, @"render_height")) {
            options.height = std::max(*height, 8u);
            options.heightSet = true;
        }
        if (auto maxDepth = JsonUInt32(settings, @"max_depth")) {
            options.maxDepth = std::max(*maxDepth, 1u);
            options.maxDepthSet = true;
        }
        if (auto seed = JsonUInt32(settings, @"fixed_rng_seed")) {
            options.seed = *seed;
            options.seedSet = true;
        }
        if (auto compact = JsonBool(settings, @"wavefront_compaction_enabled")) {
            options.wavefrontCompactionEnabled = *compact;
            options.wavefrontCompactionEnabledSet = true;
        }
        if (auto shadeIndirect = JsonBool(settings, @"wavefront_shade_indirect_enabled")) {
            options.wavefrontShadeIndirectEnabled = *shadeIndirect;
            options.wavefrontShadeIndirectEnabledSet = true;
        }
        if (auto shadeProbe = JsonBool(settings, @"wavefront_shade_indirect_probe_enabled")) {
            options.wavefrontShadeIndirectProbeEnabled = *shadeProbe;
            options.wavefrontShadeIndirectProbeEnabledSet = true;
        }
        if (auto surfaceProbe = JsonBool(settings, @"wavefront_shade_surface_probe_enabled")) {
            options.wavefrontShadeSurfaceProbeEnabled = *surfaceProbe;
            options.wavefrontShadeSurfaceProbeEnabledSet = true;
        }
        if (auto metadataProbe = JsonBool(settings, @"wavefront_shade_metadata_probe_enabled")) {
            options.wavefrontShadeMetadataProbeEnabled = *metadataProbe;
            options.wavefrontShadeMetadataProbeEnabledSet = true;
        }
        if (auto scalarProbe = JsonBool(settings, @"wavefront_shade_scalar_probe_enabled")) {
            options.wavefrontShadeScalarProbeEnabled = *scalarProbe;
            options.wavefrontShadeScalarProbeEnabledSet = true;
        }
        if (auto closureProbe = JsonBool(settings, @"wavefront_shade_closure_probe_enabled")) {
            options.wavefrontShadeClosureProbeEnabled = *closureProbe;
            options.wavefrontShadeClosureProbeEnabledSet = true;
        }
        if (auto bsdfProbe = JsonBool(settings, @"wavefront_shade_bsdf_probe_enabled")) {
            options.wavefrontShadeBsdfProbeEnabled = *bsdfProbe;
            options.wavefrontShadeBsdfProbeEnabledSet = true;
        }
        if (auto shadowIndirect = JsonBool(settings, @"wavefront_shadow_indirect_enabled")) {
            options.wavefrontShadowIndirectEnabled = *shadowIndirect;
            options.wavefrontShadowIndirectEnabledSet = true;
        }
        if (auto execution = JsonString(settings, @"execution_mode")) {
            if (*execution == "wavefront") {
                options.executionMode = PathTracer::RenderSettings::ExecutionMode::Wavefront;
            } else if (*execution == "megakernel") {
                options.executionMode = PathTracer::RenderSettings::ExecutionMode::Megakernel;
            } else {
                error = "Invalid execution_mode in render queue item JSON: " + *execution;
                return false;
            }
            options.executionModeSet = true;
        }
        if (auto policy = JsonString(settings, @"wavefront_scheduling_policy")) {
            if (*policy == "preview") {
                options.wavefrontSchedulingPolicy =
                    PathTracer::RenderSettings::WavefrontSchedulingPolicy::Preview;
            } else if (*policy == "final") {
                options.wavefrontSchedulingPolicy =
                    PathTracer::RenderSettings::WavefrontSchedulingPolicy::Final;
            } else if (*policy == "offline") {
                options.wavefrontSchedulingPolicy =
                    PathTracer::RenderSettings::WavefrontSchedulingPolicy::Offline;
            } else if (*policy == "research") {
                options.wavefrontSchedulingPolicy =
                    PathTracer::RenderSettings::WavefrontSchedulingPolicy::Research;
            } else {
                error = "Invalid wavefront_scheduling_policy in render queue item JSON: " + *policy;
                return false;
            }
            options.wavefrontSchedulingPolicySet = true;
        }
        if (auto direct = JsonString(settings, @"direct_light_mode")) {
            if (*direct == "legacy" || *direct == "legacy_rect") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::LegacyRect;
            } else if (*direct == "baseline_emissive") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::BaselineEmissive;
            } else if (*direct == "ris") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::Ris;
            } else if (*direct == "ris_spatial") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::RisSpatialReuse;
            } else if (*direct == "ris_temporal") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::RisTemporalReuse;
            } else if (*direct == "ris_world") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::RisWorldReuse;
            } else if (*direct == "ris_regir") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::RisRegirCache;
            } else if (*direct == "restir_di") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::RestirDi;
            } else if (*direct == "restir_di_regir_hybrid") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::RestirDiRegirHybrid;
            } else {
                error = "Invalid direct_light_mode in render queue item JSON: " + *direct;
                return false;
            }
            options.directLightModeSet = true;
        }
        if (auto risM = JsonUInt32(settings, @"ris_candidate_count")) {
            options.risCandidateCount = std::max(*risM, 1u);
            options.risCandidateCountSet = true;
        }
        if (auto specNee = JsonBool(settings, @"enable_specular_nee")) {
            options.enableSpecularNee = *specNee;
            options.enableSpecularNeeSet = true;
        }
        if (auto mnee = JsonBool(settings, @"enable_mnee")) {
            options.enableMnee = *mnee;
            options.enableMneeSet = true;
        }
        if (auto restirGi = JsonString(settings, @"restir_gi_mode")) {
            if (*restirGi == "off") {
                options.restirGiMode = PathTracer::RenderSettings::RestirGiMode::Off;
            } else if (*restirGi == "restir_gi_production" || *restirGi == "production_diffuse") {
                options.restirGiMode = PathTracer::RenderSettings::RestirGiMode::ProductionDiffuse;
            } else {
                error = "Invalid restir_gi_mode in render queue item JSON: " + *restirGi;
                return false;
            }
            options.restirGiModeSet = true;
        }
        if (auto guiding = JsonString(settings, @"path_guiding_mode")) {
            if (*guiding == "off") {
                options.pathGuidingMode = PathTracer::RenderSettings::PathGuidingMode::Off;
            } else if (*guiding == "prototype" || *guiding == "path_guiding") {
                options.pathGuidingMode = PathTracer::RenderSettings::PathGuidingMode::Prototype;
            } else {
                error = "Invalid path_guiding_mode in render queue item JSON: " + *guiding;
                return false;
            }
            options.pathGuidingModeSet = true;
        }
        if (auto restirPt = JsonString(settings, @"restir_pt_mode")) {
            if (*restirPt == "off") {
                options.restirPtMode = PathTracer::RenderSettings::RestirPtMode::Off;
            } else if (*restirPt == "reservoir_capture" ||
                       *restirPt == "production_reservoir_capture" ||
                       *restirPt == "restir_pt_reservoir_capture") {
                options.restirPtMode = PathTracer::RenderSettings::RestirPtMode::ProductionReservoirCapture;
            } else if (*restirPt == "path_reuse" ||
                       *restirPt == "production_path_reuse" ||
                       *restirPt == "restir_pt_path_reuse") {
                options.restirPtMode = PathTracer::RenderSettings::RestirPtMode::ProductionPathReuse;
            } else if (*restirPt == "enhanced_unbiased" ||
                       *restirPt == "restir_pt_enhanced_unbiased") {
                options.restirPtMode = PathTracer::RenderSettings::RestirPtMode::EnhancedUnbiased;
            } else if (*restirPt == "enhanced_biased" ||
                       *restirPt == "restir_pt_enhanced_biased") {
                options.restirPtMode = PathTracer::RenderSettings::RestirPtMode::EnhancedBiased;
            } else {
                error = "Invalid restir_pt_mode in render queue item JSON: " + *restirPt;
                return false;
            }
            options.restirPtModeSet = true;
        }
        if (auto cache = JsonString(settings, @"radiance_cache_mode")) {
            if (*cache == "off") {
                options.radianceCacheMode = PathTracer::RenderSettings::RadianceCacheMode::Off;
            } else if (*cache == "production" || *cache == "prototype" || *cache == "radiance_cache") {
                options.radianceCacheMode = PathTracer::RenderSettings::RadianceCacheMode::Production;
            } else {
                error = "Invalid radiance_cache_mode in render queue item JSON: " + *cache;
                return false;
            }
            options.radianceCacheModeSet = true;
        }
        if (auto caustic = JsonString(settings, @"caustic_transport_mode")) {
            if (*caustic == "off") {
                options.causticTransportMode = PathTracer::RenderSettings::CausticTransportMode::Off;
            } else if (*caustic == "path_space") {
                options.causticTransportMode =
                    PathTracer::RenderSettings::CausticTransportMode::PathSpacePrototype;
            } else {
                error = "Invalid caustic_transport_mode in render queue item JSON: " + *caustic;
                return false;
            }
            options.causticTransportModeSet = true;
        }
        if (auto spectral = JsonString(settings, @"spectral_rendering_mode")) {
            if (*spectral == "off") {
                options.spectralRenderingMode = PathTracer::RenderSettings::SpectralRenderingMode::Off;
            } else if (*spectral == "hero_wavelength") {
                options.spectralRenderingMode = PathTracer::RenderSettings::SpectralRenderingMode::HeroWavelength;
            } else {
                error = "Invalid spectral_rendering_mode in render queue item JSON: " + *spectral;
                return false;
            }
            options.spectralRenderingModeSet = true;
        }
        if (auto wavelengths = JsonUInt32(settings, @"spectral_wavelengths_per_path")) {
            if (*wavelengths != 1u && *wavelengths != 3u) {
                error = "Invalid spectral_wavelengths_per_path in render queue item JSON (expected 1 or 3)";
                return false;
            }
            options.spectralWavelengthsPerPath = *wavelengths;
            options.spectralWavelengthsPerPathSet = true;
        }
        if (auto ml = JsonString(settings, @"ml_backend_mode")) {
            if (*ml == "off") {
                options.mlBackendMode = PathTracer::RenderSettings::MlBackendMode::Off;
            } else if (*ml == "reconstruction_noop") {
                options.mlBackendMode = PathTracer::RenderSettings::MlBackendMode::ReconstructionNoOp;
            } else if (*ml == "reconstruction_tiny") {
                options.mlBackendMode = PathTracer::RenderSettings::MlBackendMode::ReconstructionTiny;
            } else {
                error = "Invalid ml_backend_mode in render queue item JSON: " + *ml;
                return false;
            }
            options.mlBackendModeSet = true;
        }
        if (auto mlPackage = JsonString(settings, @"ml_model_package_path")) {
            options.mlModelPackagePath = *mlPackage;
            options.mlModelPackagePathSet = true;
        }
        if (auto mlDebug = JsonBool(settings, @"ml_debug_enabled")) {
            options.mlDebug = *mlDebug;
            options.mlDebugSet = true;
        }
        if (auto restirDebug = JsonBool(settings, @"restir_debug_enabled")) {
            options.restirDebug = *restirDebug;
            options.restirDebugSet = true;
        }
        if (auto restirDebugView = JsonString(settings, @"restir_debug_view")) {
            if (*restirDebugView == "beauty") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::Beauty;
            } else if (*restirDebugView == "candidate_source") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::CandidateSourceId;
            } else if (*restirDebugView == "reservoir_confidence") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::ReservoirConfidence;
            } else if (*restirDebugView == "regir_cell") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::ReGIRCellId;
            } else if (*restirDebugView == "path_guiding_mask") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::PathGuidingUsedMask;
            } else if (*restirDebugView == "restir_pt_reuse") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::RestirPTReuseMask;
            } else if (*restirDebugView == "svgf_variance") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::SVGFVariance;
            } else if (*restirDebugView == "nan_inf") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::NaNInfMask;
            } else if (*restirDebugView == "queue_stage_failure") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::QueueStageFailure;
            } else if (*restirDebugView == "radiance_cache") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::RadianceCache;
            } else if (*restirDebugView == "caustic_transport") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::CausticTransport;
            } else {
                error = "Invalid restir_debug_view in render queue item JSON: " + *restirDebugView;
                return false;
            }
            options.restirDebugViewSet = true;
        }
        if (NSDictionary* environment = JsonObject(settings, @"environment")) {
            if (auto rotation = JsonFloat(environment, @"rotation_radians")) {
                options.envRotationDegrees = *rotation * static_cast<float>(180.0 / kPi);
                options.envRotationSet = true;
            }
            if (auto intensity = JsonFloat(environment, @"intensity")) {
                options.envIntensity = *intensity;
                options.envIntensitySet = true;
            }
            if (auto scale = JsonFloat(environment, @"background_scale")) {
                if (!std::isfinite(*scale) || *scale < 0.0f) {
                    error = "environment.background_scale expects a finite nonnegative float";
                    return false;
                }
                options.envBackgroundScale = *scale;
                options.envBackgroundScaleSet = true;
            }
        }
        if (NSDictionary* tonemap = JsonObject(settings, @"tonemap")) {
            if (auto mode = JsonUInt32(tonemap, @"mode")) {
                options.tonemapMode = *mode;
                options.tonemapSet = true;
            }
            if (auto exposure = JsonFloat(tonemap, @"exposure")) {
                options.exposure = *exposure;
                options.exposureSet = true;
            }
        }
        if (NSDictionary* offline = JsonObject(settings, @"offline_post")) {
            if (auto denoise = JsonBool(offline, @"denoise_enabled")) {
                options.denoiseEnabled = *denoise;
                options.denoiseEnabledSet = true;
            }
            if (auto hdr = JsonBool(offline, @"denoise_hdr")) {
                options.denoiseHdr = *hdr;
                options.denoiseHdrSet = true;
            }
            if (auto albedo = JsonBool(offline, @"denoise_use_albedo")) {
                options.denoiseUseAlbedo = *albedo;
                options.denoiseUseAlbedoSet = true;
            }
            if (auto normal = JsonBool(offline, @"denoise_use_normal")) {
                options.denoiseUseNormal = *normal;
                options.denoiseUseNormalSet = true;
            }
            if (auto svgf = JsonBool(offline, @"svgf_denoise_enabled")) {
                options.svgfDenoiseEnabled = *svgf;
                options.svgfDenoiseEnabledSet = true;
            }
            if (auto svgfDirect = JsonBool(offline, @"svgf_denoise_direct")) {
                options.svgfDenoiseDirect = *svgfDirect;
                options.svgfDenoiseDirectSet = true;
            }
            if (auto svgfIndirect = JsonBool(offline, @"svgf_denoise_indirect")) {
                options.svgfDenoiseIndirect = *svgfIndirect;
                options.svgfDenoiseIndirectSet = true;
            }
        }
        if (NSDictionary* resource = JsonObject(settings, @"resource_policy")) {
            if (auto budget = JsonString(resource, @"budget_policy")) {
                if (!ApplyBudgetPolicyName(*budget, options, error)) {
                    return false;
                }
            }
            if (auto pilot = JsonString(resource, @"pilot_mode")) {
                if (!ApplyPilotModeName(*pilot, options, error)) {
                    return false;
                }
            }
            if (auto textureMaxDim = JsonUInt32(resource, @"texture_max_dim")) {
                options.textureMaxDim = *textureMaxDim;
                options.textureMaxDimSet = *textureMaxDim > 0u;
            }
            if (auto maxTextureBytes = JsonUInt64(resource, @"max_texture_bytes")) {
                options.maxTextureBytes = *maxTextureBytes;
                options.maxTextureBytesSet = *maxTextureBytes > 0u;
            }
        }
    }
    return true;
}

bool ApplyRenderQueueJson(const fs::path& path, CliOptions& options, std::string& error) {
    NSDictionary* root = ReadJsonDictionary(path, error);
    if (!root) {
        return false;
    }
    return ApplyRenderQueueDictionary(root, options, error);
}

bool ApplyCheckpointJson(const fs::path& path, CliOptions& options, std::string& error) {
    NSDictionary* root = ReadJsonDictionary(path, error);
    if (!root) {
        return false;
    }
    const auto schema = JsonString(root, @"schema_version");
    if (!schema || *schema != "pathtracer.render_checkpoint.v1") {
        error = "Unsupported checkpoint schema in " + path.string() +
                " (expected pathtracer.render_checkpoint.v1)";
        return false;
    }
    const auto kind = JsonString(root, @"checkpoint_kind");
    bool partial = kind && *kind == "partial_single_frame";
    if (!kind || (*kind != "completed_single_frame" && !partial)) {
        error = "Unsupported checkpoint kind in " + path.string() +
                " (expected completed_single_frame or partial_single_frame)";
        return false;
    }
    const auto resumePolicy = JsonString(root, @"resume_policy");
    // A completed manifest with its finished accumulation writes the output from it (write_output_from_state); the
    // state's sample count equals the target, so the renderer draws no frame.
    const bool completedFromState = !partial && resumePolicy && *resumePolicy == "write_output_from_state";
    const char* expectedPolicy =
        partial ? "continue_accumulation_from_state" : "full_frame_rerender_from_checkpoint_manifest";
    if (!resumePolicy || (*resumePolicy != expectedPolicy && !completedFromState)) {
        error = "Unsupported checkpoint resume_policy in " + path.string() + " (expected " + expectedPolicy +
                (partial ? std::string() : std::string(" or write_output_from_state")) + ")";
        return false;
    }
    NSDictionary* queueItem = JsonObject(root, @"queue_item");
    if (!queueItem) {
        error = "Checkpoint manifest is missing queue_item for deterministic resume: " + path.string();
        return false;
    }
    if (!ApplyRenderQueueDictionary(queueItem, options, error)) {
        return false;
    }
    if (partial || completedFromState) {
        const auto statePath = JsonString(root, @"accumulation_state");
        if (!statePath || statePath->empty()) {
            error = "Checkpoint manifest is missing accumulation_state: " + path.string();
            return false;
        }
        options.resumeAccumulationStatePath = *statePath;
        options.resumeFromAccumulationState = true;
        if (!options.checkpointManifestJsonProvided) {
            // Later partial manifests and the completed one replace this manifest.
            options.checkpointManifestJsonPath = path.string();
            options.checkpointManifestJsonProvided = true;
        }
        if (const auto every = JsonUInt32(root, @"checkpoint_every_samples")) {
            options.checkpointEverySamples = *every;
        }
        // The run flags of the interrupted render, unless the command line sets them again.
        if (const auto hw = JsonBool(root, @"force_hardware_ray_tracing"); hw && !options.forceHardwareRayTracingSet) {
            options.forceHardwareRayTracing = *hw;
            options.forceHardwareRayTracingSet = true;
        }
        if (const auto sw = JsonBool(root, @"force_software_ray_tracing"); sw && !options.forceSoftwareRayTracingSet) {
            options.forceSoftwareRayTracing = *sw;
            options.forceSoftwareRayTracingSet = true;
        }
        if (const auto guides = JsonBool(root, @"write_denoise_guides"); guides && *guides) {
            options.writeDenoiseGuides = true;
        }
    }
    return true;
}

bool ParseOptions(int argc, const char** argv, CliOptions& options, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string value;
        auto eqPos = arg.find('=');
        bool hasInlineValue = eqPos != std::string::npos;
        if (hasInlineValue) {
            value = arg.substr(eqPos + 1);
            arg = arg.substr(0, eqPos);
        }

        auto requireValue = [&](const char* name) -> bool {
            if (hasInlineValue) {
                return true;
            }
            if (i + 1 >= argc) {
                error = std::string(name) + " requires a value";
                return false;
            }
            value = argv[++i];
            return true;
        };

        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--verbose" || arg == "-v") {
            bool v = true;
            if (hasInlineValue && !ParseBoolFlag(value, true, v)) {
                error = "Invalid value for --verbose";
                return false;
            }
            options.verbose = v;
        } else if (arg == "--stats") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --stats";
                return false;
            }
            options.stats = flag;
        } else if (arg == "--debug-landmarks") {
            options.debugLandmarks = true;
        } else if (arg == "--validate-emissive") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --validate-emissive";
                return false;
            }
            options.validateEmissive = flag;
        } else if (arg == "--expected-emissive-count") {
            if (!requireValue("--expected-emissive-count")) {
                return false;
            }
            try {
                options.expectedEmissiveCount = static_cast<uint32_t>(std::stoul(value));
                options.expectedEmissiveCountSet = true;
                options.validateEmissive = true;
            } catch (...) {
                error = "Invalid integer for --expected-emissive-count";
                return false;
            }
        } else if (arg == "--budget-policy") {
            if (!requireValue("--budget-policy")) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "strict") {
                options.budgetPolicy = PathTracer::TextureBudgetPolicy::Strict;
            } else if (lower == "warn") {
                options.budgetPolicy = PathTracer::TextureBudgetPolicy::Warn;
            } else if (lower == "ignore") {
                options.budgetPolicy = PathTracer::TextureBudgetPolicy::Ignore;
            } else {
                error = "Invalid value for --budget-policy (expected strict, warn, or ignore)";
                return false;
            }
        } else if (arg == "--pilot-mode") {
            if (!requireValue("--pilot-mode")) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "geo_only") {
                options.pilotMode = PathTracer::PilotMode::GeoOnly;
            } else if (lower == "albedo_only") {
                options.pilotMode = PathTracer::PilotMode::AlbedoOnly;
            } else if (lower == "full_clamped") {
                options.pilotMode = PathTracer::PilotMode::FullClamped;
            } else {
                error = "Invalid value for --pilot-mode (expected geo_only, albedo_only, or full_clamped)";
                return false;
            }
        } else if (arg == "--texture-max-dim") {
            if (!requireValue("--texture-max-dim")) {
                return false;
            }
            try {
                int parsed = std::stoi(value);
                if (parsed < 1) {
                    error = "--texture-max-dim must be >= 1";
                    return false;
                }
                options.textureMaxDim = static_cast<uint32_t>(parsed);
                options.textureMaxDimSet = true;
            } catch (...) {
                error = "Invalid integer for --texture-max-dim";
                return false;
            }
        } else if (arg == "--max-texture-bytes") {
            if (!requireValue("--max-texture-bytes")) {
                return false;
            }
            try {
                unsigned long long parsed = std::stoull(value);
                options.maxTextureBytes = static_cast<uint64_t>(parsed);
                options.maxTextureBytesSet = true;
            } catch (...) {
                error = "Invalid integer for --max-texture-bytes";
                return false;
            }
        } else if (arg == "--scene") {
            if (!requireValue("--scene")) {
                return false;
            }
            options.scene = value;
            options.sceneProvided = true;
        } else if (arg == "--renderMode" || arg == "--render-mode") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            if (!ApplyRenderModeName(value, options, error)) {
                error = "Invalid value for " + arg +
                        " (expected metal_final_frame or cpu_reference_headless)";
                return false;
            }
        } else if (arg == "--renderProfile" || arg == "--render-profile") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "custom") {
                options.renderProfile = RenderProfile::Custom;
            } else if (lower == "preview") {
                options.renderProfile = RenderProfile::Preview;
            } else if (lower == "lookdev") {
                options.renderProfile = RenderProfile::Lookdev;
            } else if (lower == "final") {
                options.renderProfile = RenderProfile::Final;
            } else if (lower == "reference") {
                options.renderProfile = RenderProfile::Reference;
            } else if (lower == "debug") {
                options.renderProfile = RenderProfile::Debug;
            } else {
                error = "Invalid value for " + arg + " (expected custom, preview, lookdev, final, reference, or debug)";
                return false;
            }
            options.renderProfileSet = options.renderProfile != RenderProfile::Custom;
        } else if (arg == "--output" || arg == "--out") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            options.outputPath = value;
            options.outputProvided = true;
        } else if (arg == "--width") {
            if (!requireValue("--width")) {
                return false;
            }
            try {
                int parsed = std::stoi(value);
                if (parsed < 8) {
                    error = "--width must be >= 8";
                    return false;
                }
                options.width = static_cast<uint32_t>(parsed);
                options.widthSet = true;
            } catch (...) {
                error = "Invalid integer for --width";
                return false;
            }
        } else if (arg == "--height") {
            if (!requireValue("--height")) {
                return false;
            }
            try {
                int parsed = std::stoi(value);
                if (parsed < 8) {
                    error = "--height must be >= 8";
                    return false;
                }
                options.height = static_cast<uint32_t>(parsed);
                options.heightSet = true;
            } catch (...) {
                error = "Invalid integer for --height";
                return false;
            }
        } else if (arg == "--sppTotal" || arg == "--spp") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                int parsed = std::stoi(value);
                if (parsed < 1) {
                    error = arg + " must be >= 1";
                    return false;
                }
                options.sppTotal = static_cast<uint32_t>(parsed);
                options.sppTotalSet = true;
            } catch (...) {
                error = "Invalid integer for " + arg;
                return false;
            }
        } else if (arg == "--maxDepth") {
            if (!requireValue("--maxDepth")) {
                return false;
            }
            try {
                int parsed = std::stoi(value);
                if (parsed < 1) {
                    error = "--maxDepth must be >= 1";
                    return false;
                }
                options.maxDepth = static_cast<uint32_t>(parsed);
                options.maxDepthSet = true;
            } catch (...) {
                error = "Invalid integer for --maxDepth";
                return false;
            }
        } else if (arg == "--threads") {
            if (!requireValue("--threads")) {
                return false;
            }
            try {
                int parsed = std::stoi(value);
                if (parsed < 1) {
                    error = "--threads must be >= 1";
                    return false;
                }
                options.threads = static_cast<uint32_t>(parsed);
                options.threadsSet = true;
            } catch (...) {
                error = "Invalid integer for --threads";
                return false;
            }
        } else if (arg == "--seed") {
            if (!requireValue("--seed")) {
                return false;
            }
            try {
                options.seed = static_cast<uint32_t>(std::stoul(value));
                options.seedSet = true;
            } catch (...) {
                error = "Invalid integer for --seed";
                return false;
            }
        } else if (arg == "--envRotation") {
            if (!requireValue("--envRotation")) {
                return false;
            }
            try {
                options.envRotationDegrees = std::stof(value);
                options.envRotationSet = true;
            } catch (...) {
                error = "Invalid float for --envRotation";
                return false;
            }
        } else if (arg == "--envIntensity") {
            if (!requireValue("--envIntensity")) {
                return false;
            }
            try {
                float parsed = std::stof(value);
                options.envIntensity = std::max(parsed, 0.0f);
                options.envIntensitySet = true;
            } catch (...) {
                error = "Invalid float for --envIntensity";
                return false;
            }
        } else if (arg == "--envBackgroundScale") {
            if (!requireValue("--envBackgroundScale")) return false;
            try {
                size_t parsedLength = 0u;
                const float parsed = std::stof(value, &parsedLength);
                if (parsedLength != value.size() || !std::isfinite(parsed) || parsed < 0.0f) {
                    error = "--envBackgroundScale expects a finite nonnegative float";
                    return false;
                }
                options.envBackgroundScale = parsed;
                options.envBackgroundScaleSet = true;
            } catch (...) {
                error = "--envBackgroundScale expects a finite nonnegative float";
                return false;
            }
        } else if (arg == "--tonemap") {
            if (!requireValue("--tonemap")) {
                return false;
            }
            try {
                int parsed = std::stoi(value);
                if (parsed < 1 || parsed > 6) {
                    error = "--tonemap must be in [1,6]";
                    return false;
                }
                options.tonemapMode = static_cast<uint32_t>(parsed);
                options.tonemapSet = true;
            } catch (...) {
                error = "Invalid integer for --tonemap";
                return false;
            }
        } else if (arg == "--exposure") {
            if (!requireValue("--exposure")) {
                return false;
            }
            try {
                options.exposure = std::stof(value);
                options.exposureSet = true;
            } catch (...) {
                error = "Invalid float for --exposure";
                return false;
            }
        } else if (arg == "--denoise") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --denoise";
                return false;
            }
            options.denoiseEnabled = flag;
            options.denoiseEnabledSet = true;
        } else if (arg == "--denoise-hdr") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --denoise-hdr";
                return false;
            }
            options.denoiseHdr = flag;
            options.denoiseHdrSet = true;
        } else if (arg == "--denoise-use-albedo") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --denoise-use-albedo";
                return false;
            }
            options.denoiseUseAlbedo = flag;
            options.denoiseUseAlbedoSet = true;
        } else if (arg == "--denoise-use-normal") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --denoise-use-normal";
                return false;
            }
            options.denoiseUseNormal = flag;
            options.denoiseUseNormalSet = true;
        } else if (arg == "--svgfDenoise" || arg == "--svgf-denoise") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --svgfDenoise";
                return false;
            }
            options.svgfDenoiseEnabled = flag;
            options.svgfDenoiseEnabledSet = true;
        } else if (arg == "--svgf-denoise-direct" || arg == "--svgfDenoiseDirect") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --svgf-denoise-direct";
                return false;
            }
            options.svgfDenoiseDirect = flag;
            options.svgfDenoiseDirectSet = true;
        } else if (arg == "--svgf-denoise-indirect" || arg == "--svgfDenoiseIndirect") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --svgf-denoise-indirect";
                return false;
            }
            options.svgfDenoiseIndirect = flag;
            options.svgfDenoiseIndirectSet = true;
        } else if (arg == "--svgf-atrous-iterations" || arg == "--svgfAtrousIterations") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.svgfAtrousIterations = static_cast<uint32_t>(std::stoul(value));
                options.svgfAtrousIterationsSet = true;
            } catch (...) {
                error = "Invalid integer for --svgf-atrous-iterations";
                return false;
            }
        } else if (arg == "--svgf-normal-sigma" || arg == "--svgfNormalSigma") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.svgfNormalSigma = std::stof(value);
                options.svgfNormalSigmaSet = true;
            } catch (...) {
                error = "Invalid float for --svgf-normal-sigma";
                return false;
            }
        } else if (arg == "--svgf-albedo-sigma" || arg == "--svgfAlbedoSigma") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.svgfAlbedoSigma = std::stof(value);
                options.svgfAlbedoSigmaSet = true;
            } catch (...) {
                error = "Invalid float for --svgf-albedo-sigma";
                return false;
            }
        } else if (arg == "--svgf-luminance-sigma" || arg == "--svgfLuminanceSigma") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.svgfLuminanceSigma = std::stof(value);
                options.svgfLuminanceSigmaSet = true;
            } catch (...) {
                error = "Invalid float for --svgf-luminance-sigma";
                return false;
            }
        } else if (arg == "--firefly-clamp-mode") {
            if (!requireValue("--firefly-clamp-mode")) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "off") {
                options.fireflyClampMode = CliOptions::FireflyClampMode::Off;
            } else if (lower == "luminance") {
                options.fireflyClampMode = CliOptions::FireflyClampMode::Luminance;
            } else if (lower == "maxcomponent") {
                options.fireflyClampMode = CliOptions::FireflyClampMode::MaxComponent;
            } else {
                error = "Invalid value for --firefly-clamp-mode (expected off, luminance, or maxcomponent)";
                return false;
            }
            options.fireflyClampModeSet = true;
        } else if (arg == "--firefly-clamp-value") {
            if (!requireValue("--firefly-clamp-value")) {
                return false;
            }
            try {
                options.fireflyClampValue = std::stof(value);
                options.fireflyClampValueSet = true;
            } catch (...) {
                error = "Invalid float for --firefly-clamp-value";
                return false;
            }
        } else if (arg == "--enableSoftwareRayTracing") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --enableSoftwareRayTracing";
                return false;
            }
            options.enableSoftwareRayTracing = flag;
            options.enableSoftwareRayTracingSet = true;
        } else if (arg == "--force-hwrt") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --force-hwrt";
                return false;
            }
            options.forceHardwareRayTracing = flag;
            options.forceHardwareRayTracingSet = true;
        } else if (arg == "--force-swrt") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --force-swrt";
                return false;
            }
            options.forceSoftwareRayTracing = flag;
            options.forceSoftwareRayTracingSet = true;
        } else if (arg == "--enableMnee") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --enableMnee";
                return false;
            }
            options.enableMnee = flag;
            options.enableMneeSet = true;
        } else if (arg == "--enableSpecularNee") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --enableSpecularNee";
                return false;
            }
            options.enableSpecularNee = flag;
            options.enableSpecularNeeSet = true;
        } else if (arg == "--executionMode" || arg == "--execution-mode") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "megakernel" || lower == "mega" || lower == "0") {
                options.executionMode = PathTracer::RenderSettings::ExecutionMode::Megakernel;
            } else if (lower == "wavefront" || lower == "wave" || lower == "1") {
                options.executionMode = PathTracer::RenderSettings::ExecutionMode::Wavefront;
            } else {
                error = "Invalid value for --executionMode (expected megakernel or wavefront)";
                return false;
            }
            options.executionModeSet = true;
        } else if (arg == "--wavefront-policy" || arg == "--wavefrontPolicy") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "preview" || lower == "interactive" || lower == "0") {
                options.wavefrontSchedulingPolicy =
                    PathTracer::RenderSettings::WavefrontSchedulingPolicy::Preview;
            } else if (lower == "final" || lower == "production" || lower == "1") {
                options.wavefrontSchedulingPolicy =
                    PathTracer::RenderSettings::WavefrontSchedulingPolicy::Final;
            } else if (lower == "offline" || lower == "batch" || lower == "2") {
                options.wavefrontSchedulingPolicy =
                    PathTracer::RenderSettings::WavefrontSchedulingPolicy::Offline;
            } else if (lower == "research" || lower == "debug" || lower == "3") {
                options.wavefrontSchedulingPolicy =
                    PathTracer::RenderSettings::WavefrontSchedulingPolicy::Research;
            } else {
                error = "Invalid value for --wavefront-policy (expected preview, final, offline, or research)";
                return false;
            }
            options.wavefrontSchedulingPolicySet = true;
        } else if (arg == "--wavefront-compaction" || arg == "--wavefrontCompaction") {
            std::string boolValue = hasInlineValue ? value : std::string();
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --wavefront-compaction (expected 0/1/true/false)";
                return false;
            }
            options.wavefrontCompactionEnabled = flag;
            options.wavefrontCompactionEnabledSet = true;
        } else if (arg == "--wavefront-shade-indirect" || arg == "--wavefrontShadeIndirect") {
            std::string boolValue = hasInlineValue ? value : std::string();
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --wavefront-shade-indirect (expected 0/1/true/false)";
                return false;
            }
            options.wavefrontShadeIndirectEnabled = flag;
            options.wavefrontShadeIndirectEnabledSet = true;
        } else if (arg == "--wavefront-shade-indirect-probe" ||
                   arg == "--wavefrontShadeIndirectProbe") {
            std::string boolValue = hasInlineValue ? value : std::string();
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --wavefront-shade-indirect-probe (expected 0/1/true/false)";
                return false;
            }
            options.wavefrontShadeIndirectProbeEnabled = flag;
            options.wavefrontShadeIndirectProbeEnabledSet = true;
        } else if (arg == "--wavefront-shade-surface-probe" ||
                   arg == "--wavefrontShadeSurfaceProbe") {
            std::string boolValue = hasInlineValue ? value : std::string();
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --wavefront-shade-surface-probe (expected 0/1/true/false)";
                return false;
            }
            options.wavefrontShadeSurfaceProbeEnabled = flag;
            options.wavefrontShadeSurfaceProbeEnabledSet = true;
        } else if (arg == "--wavefront-shade-metadata-probe" ||
                   arg == "--wavefrontShadeMetadataProbe") {
            std::string boolValue = hasInlineValue ? value : std::string();
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --wavefront-shade-metadata-probe (expected 0/1/true/false)";
                return false;
            }
            options.wavefrontShadeMetadataProbeEnabled = flag;
            options.wavefrontShadeMetadataProbeEnabledSet = true;
        } else if (arg == "--wavefront-shade-scalar-probe" ||
                   arg == "--wavefrontShadeScalarProbe") {
            std::string boolValue = hasInlineValue ? value : std::string();
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --wavefront-shade-scalar-probe (expected 0/1/true/false)";
                return false;
            }
            options.wavefrontShadeScalarProbeEnabled = flag;
            options.wavefrontShadeScalarProbeEnabledSet = true;
        } else if (arg == "--wavefront-shade-closure-probe" ||
                   arg == "--wavefrontShadeClosureProbe") {
            std::string boolValue = hasInlineValue ? value : std::string();
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --wavefront-shade-closure-probe (expected 0/1/true/false)";
                return false;
            }
            options.wavefrontShadeClosureProbeEnabled = flag;
            options.wavefrontShadeClosureProbeEnabledSet = true;
        } else if (arg == "--wavefront-shade-bsdf-probe" ||
                   arg == "--wavefrontShadeBsdfProbe") {
            std::string boolValue = hasInlineValue ? value : std::string();
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --wavefront-shade-bsdf-probe (expected 0/1/true/false)";
                return false;
            }
            options.wavefrontShadeBsdfProbeEnabled = flag;
            options.wavefrontShadeBsdfProbeEnabledSet = true;
        } else if (arg == "--wavefront-shadow-indirect" || arg == "--wavefrontShadowIndirect") {
            std::string boolValue = hasInlineValue ? value : std::string();
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --wavefront-shadow-indirect (expected 0/1/true/false)";
                return false;
            }
            options.wavefrontShadowIndirectEnabled = flag;
            options.wavefrontShadowIndirectEnabledSet = true;
        } else if (arg == "--directLightMode") {
            if (!requireValue("--directLightMode")) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "legacy" || lower == "rect") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::LegacyRect;
            } else if (lower == "baseline_emissive" || lower == "baseline") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::BaselineEmissive;
            } else if (lower == "ris") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::Ris;
            } else if (lower == "ris_spatial" ||
                       lower == "spatial_ris" ||
                       lower == "spatial_reuse") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::RisSpatialReuse;
            } else if (lower == "ris_temporal" ||
                       lower == "temporal_ris" ||
                       lower == "temporal_reuse") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::RisTemporalReuse;
            } else if (lower == "ris_world" ||
                       lower == "world_ris" ||
                       lower == "world_reuse" ||
                       lower == "ris_grid") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::RisWorldReuse;
            } else if (lower == "ris_regir" ||
                       lower == "regir" ||
                       lower == "ris_cache" ||
                       lower == "cache_reuse") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::RisRegirCache;
            } else if (lower == "restir_di" ||
                       lower == "restir" ||
                       lower == "restir_temporal") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::RestirDi;
            } else if (lower == "restir_di_regir_hybrid" ||
                       lower == "restir_regir" ||
                       lower == "hybrid_regir") {
                options.directLightMode = PathTracer::RenderSettings::DirectLightMode::RestirDiRegirHybrid;
            } else {
                error = "Invalid value for --directLightMode (expected legacy, baseline_emissive, ris, ris_spatial, ris_temporal, ris_world, ris_regir, restir_di, or restir_di_regir_hybrid)";
                return false;
            }
            options.directLightModeSet = true;
        } else if (arg == "--restirGiMode" || arg == "--restirGIMode") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "off" || lower == "0" || lower == "disabled") {
                options.restirGiMode = PathTracer::RenderSettings::RestirGiMode::Off;
            } else if (lower == "restir_gi_production" ||
                       lower == "production_diffuse" ||
                       lower == "restir_gi_prototype" ||
                       lower == "restir_gi_diffuse" ||
                       lower == "restir_gi" ||
                       lower == "diffuse_prototype" ||
                       lower == "1") {
                options.restirGiMode = PathTracer::RenderSettings::RestirGiMode::ProductionDiffuse;
            } else {
                error = "Invalid value for --restirGiMode (expected off, restir_gi_production, or restir_gi_diffuse)";
                return false;
            }
            options.restirGiModeSet = true;
        } else if (arg == "--pathGuidingMode" || arg == "--path-guiding-mode") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "off" || lower == "0" || lower == "disabled") {
                options.pathGuidingMode = PathTracer::RenderSettings::PathGuidingMode::Off;
            } else if (lower == "path_guiding_prototype" ||
                       lower == "path_guiding" ||
                       lower == "real_path_guiding" ||
                       lower == "guiding" ||
                       lower == "prototype" ||
                       lower == "1") {
                options.pathGuidingMode = PathTracer::RenderSettings::PathGuidingMode::Prototype;
            } else {
                error = "Invalid value for --pathGuidingMode (expected off or path_guiding)";
                return false;
            }
            options.pathGuidingModeSet = true;
        } else if (arg == "--pathGuidingStrength" || arg == "--path-guiding-strength") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.pathGuidingStrength = std::stof(value);
                options.pathGuidingStrengthSet = true;
            } catch (...) {
                error = "Invalid float for --pathGuidingStrength";
                return false;
            }
        } else if (arg == "--pathGuidingCellSize" || arg == "--path-guiding-cell-size") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.pathGuidingCellSize = std::stof(value);
                options.pathGuidingCellSizeSet = true;
            } catch (...) {
                error = "Invalid float for --pathGuidingCellSize";
                return false;
            }
        } else if (arg == "--pathGuidingTrainingInterval" || arg == "--path-guiding-training-interval") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.pathGuidingTrainingInterval =
                    static_cast<uint32_t>(std::max(1, std::stoi(value)));
                options.pathGuidingTrainingIntervalSet = true;
            } catch (...) {
                error = "Invalid integer for --pathGuidingTrainingInterval";
                return false;
            }
        } else if (arg == "--restirPtMode" || arg == "--restir-pt-mode") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "off" || lower == "0" || lower == "disabled") {
                options.restirPtMode = PathTracer::RenderSettings::RestirPtMode::Off;
            } else if (lower == "restir_pt_reservoir_capture" ||
                       lower == "restir_pt_capture" ||
                       lower == "restir_pt_research" ||
                       lower == "restir_pt_scaffold" ||
                       lower == "research" ||
                       lower == "1") {
                options.restirPtMode = PathTracer::RenderSettings::RestirPtMode::ProductionReservoirCapture;
            } else if (lower == "restir_pt_path_reuse" ||
                       lower == "restir_pt_production_path_reuse" ||
                       lower == "restir_pt_experimental_path_reuse" ||
                       lower == "experimental_path_reuse" ||
                       lower == "path_reuse" ||
                       lower == "2") {
                options.restirPtMode = PathTracer::RenderSettings::RestirPtMode::ProductionPathReuse;
            } else if (lower == "restir_pt_enhanced_unbiased" ||
                       lower == "enhanced_unbiased" ||
                       lower == "restir_pt_enhanced" ||
                       lower == "3") {
                options.restirPtMode = PathTracer::RenderSettings::RestirPtMode::EnhancedUnbiased;
            } else if (lower == "restir_pt_enhanced_biased" ||
                       lower == "enhanced_biased" ||
                       lower == "4") {
                options.restirPtMode = PathTracer::RenderSettings::RestirPtMode::EnhancedBiased;
            } else {
                error = "Invalid value for --restirPtMode (expected off, restir_pt_reservoir_capture, restir_pt_path_reuse, restir_pt_enhanced_unbiased, or restir_pt_enhanced_biased)";
                return false;
            }
            options.restirPtModeSet = true;
        } else if (arg == "--restirPtMaxReservoirs" || arg == "--restir-pt-max-reservoirs") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.restirPtMaxReservoirs = static_cast<uint32_t>(std::stoul(value));
                options.restirPtMaxReservoirsSet = true;
            } catch (...) {
                error = "Invalid integer for --restirPtMaxReservoirs";
                return false;
            }
        } else if (arg == "--restirPtDebug" || arg == "--restir-pt-debug") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --restirPtDebug";
                return false;
            }
            options.restirPtDebug = flag;
            options.restirPtDebugSet = true;
        } else if (arg == "--restirPtReuseStrength" || arg == "--restir-pt-reuse-strength") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.restirPtReuseStrength = std::stof(value);
                options.restirPtReuseStrengthSet = true;
            } catch (...) {
                error = "Invalid float for --restirPtReuseStrength";
                return false;
            }
        } else if (arg == "--radianceCacheMode" || arg == "--radiance-cache-mode") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "off" || lower == "0" || lower == "disabled") {
                options.radianceCacheMode = PathTracer::RenderSettings::RadianceCacheMode::Off;
            } else if (lower == "radiance_cache" ||
                       lower == "radiance_cache_prototype" ||
                       lower == "cache" ||
                       lower == "prototype" ||
                       lower == "1") {
                options.radianceCacheMode = PathTracer::RenderSettings::RadianceCacheMode::Prototype;
            } else {
                error = "Invalid value for --radianceCacheMode (expected off or radiance_cache)";
                return false;
            }
            options.radianceCacheModeSet = true;
        } else if (arg == "--radianceCacheCellSize" || arg == "--radiance-cache-cell-size") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.radianceCacheCellSize = std::stof(value);
                options.radianceCacheCellSizeSet = true;
            } catch (...) {
                error = "Invalid float for --radianceCacheCellSize";
                return false;
            }
        } else if (arg == "--radianceCacheMinDepth" || arg == "--radiance-cache-min-depth") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.radianceCacheMinDepth = static_cast<uint32_t>(std::max(1, std::stoi(value)));
                options.radianceCacheMinDepthSet = true;
            } catch (...) {
                error = "Invalid integer for --radianceCacheMinDepth";
                return false;
            }
        } else if (arg == "--radianceCacheMinConfidence" || arg == "--radiance-cache-min-confidence") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.radianceCacheMinConfidence = std::stof(value);
                options.radianceCacheMinConfidenceSet = true;
            } catch (...) {
                error = "Invalid float for --radianceCacheMinConfidence";
                return false;
            }
        } else if (arg == "--radianceCacheTrainingInterval" || arg == "--radiance-cache-training-interval") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.radianceCacheTrainingInterval =
                    static_cast<uint32_t>(std::max(1, std::stoi(value)));
                options.radianceCacheTrainingIntervalSet = true;
            } catch (...) {
                error = "Invalid integer for --radianceCacheTrainingInterval";
                return false;
            }
        } else if (arg == "--radianceCacheMaxRadiance" || arg == "--radiance-cache-max-radiance") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.radianceCacheMaxRadiance = std::stof(value);
                options.radianceCacheMaxRadianceSet = true;
            } catch (...) {
                error = "Invalid float for --radianceCacheMaxRadiance";
                return false;
            }
            if (!(options.radianceCacheMaxRadiance > 0.0f) ||
                options.radianceCacheMaxRadiance > PathTracer::RadianceCacheTraining::kMaxRadianceLimit) {
                error = "--radianceCacheMaxRadiance must be in (0, 1023]: the radiance cache stores up to 1024 "
                        "samples per cell in fixed point, and a larger maximum leaves too little resolution";
                return false;
            }
        } else if (arg == "--radianceCacheDebug" || arg == "--radiance-cache-debug") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --radianceCacheDebug";
                return false;
            }
            options.radianceCacheDebug = flag;
            options.radianceCacheDebugSet = true;
        } else if (arg == "--causticTransportMode" || arg == "--caustic-transport-mode") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "off" || lower == "0" || lower == "disabled") {
                options.causticTransportMode =
                    PathTracer::RenderSettings::CausticTransportMode::Off;
            } else if (lower == "path_space" ||
                       lower == "path_space_prototype" ||
                       lower == "prototype" ||
                       lower == "1") {
                options.causticTransportMode =
                    PathTracer::RenderSettings::CausticTransportMode::PathSpacePrototype;
            } else {
                error = "Invalid value for --causticTransportMode (expected off or path_space)";
                return false;
            }
            options.causticTransportModeSet = true;
        } else if (arg == "--causticMaxVertices" || arg == "--caustic-max-vertices") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.causticMaxVertices = static_cast<uint32_t>(std::max(1, std::stoi(value)));
                options.causticMaxVerticesSet = true;
            } catch (...) {
                error = "Invalid integer for --causticMaxVertices";
                return false;
            }
        } else if (arg == "--volumeTransportMode" || arg == "--volume-transport-mode") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "off" || lower == "0" || lower == "disabled") {
                options.volumeTransportMode = PathTracer::RenderSettings::VolumeTransportMode::Off;
            } else if (lower == "homogeneous" || lower == "homo" || lower == "1") {
                options.volumeTransportMode = PathTracer::RenderSettings::VolumeTransportMode::Homogeneous;
            } else {
                error = "Invalid value for --volumeTransportMode (expected off or homogeneous)";
                return false;
            }
            options.volumeTransportModeSet = true;
        } else if (arg == "--volumePhase" || arg == "--volume-phase") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "isotropic" || lower == "iso" || lower == "0") {
                options.volumePhaseFunction = PathTracer::RenderSettings::VolumePhaseFunction::Isotropic;
            } else if (lower == "hg" || lower == "henyey_greenstein" || lower == "henyey-greenstein" || lower == "1") {
                options.volumePhaseFunction = PathTracer::RenderSettings::VolumePhaseFunction::HenyeyGreenstein;
            } else {
                error = "Invalid value for --volumePhase (expected isotropic or hg)";
                return false;
            }
            options.volumePhaseFunctionSet = true;
        } else if (arg == "--volumeSigmaA" || arg == "--volume-sigma-a") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            auto parsed = ParseFloat3Option(value);
            if (!parsed) {
                error = "Invalid value for --volumeSigmaA (expected r,g,b)";
                return false;
            }
            options.volumeSigmaA = *parsed;
            options.volumeSigmaASet = true;
        } else if (arg == "--volumeSigmaS" || arg == "--volume-sigma-s") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            auto parsed = ParseFloat3Option(value);
            if (!parsed) {
                error = "Invalid value for --volumeSigmaS (expected r,g,b)";
                return false;
            }
            options.volumeSigmaS = *parsed;
            options.volumeSigmaSSet = true;
        } else if (arg == "--volumeEmission" || arg == "--volume-emission") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            auto parsed = ParseFloat3Option(value);
            if (!parsed) {
                error = "Invalid value for --volumeEmission (expected r,g,b)";
                return false;
            }
            options.volumeEmission = *parsed;
            options.volumeEmissionSet = true;
        } else if (arg == "--volumeAnisotropy" || arg == "--volume-anisotropy") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.volumeAnisotropy = std::stof(value);
                options.volumeAnisotropySet = true;
            } catch (...) {
                error = "Invalid float for --volumeAnisotropy";
                return false;
            }
        } else if (arg == "--volumeMaxScatteringEvents" || arg == "--volume-max-scattering-events") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.volumeMaxScatteringEvents =
                    static_cast<uint32_t>(std::max(1, std::stoi(value)));
                options.volumeMaxScatteringEventsSet = true;
            } catch (...) {
                error = "Invalid integer for --volumeMaxScatteringEvents";
                return false;
            }
        } else if (arg == "--volumeNee" || arg == "--volume-nee") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --volumeNee";
                return false;
            }
            options.volumeNeeEnabled = flag;
            options.volumeNeeEnabledSet = true;
        } else if (arg == "--volumeDebug" || arg == "--volume-debug") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --volumeDebug";
                return false;
            }
            options.volumeDebug = flag;
            options.volumeDebugSet = true;
        } else if (arg == "--spectralMode" || arg == "--spectral-mode") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "off" || lower == "0" || lower == "disabled") {
                options.spectralRenderingMode =
                    PathTracer::RenderSettings::SpectralRenderingMode::Off;
            } else if (lower == "hero" || lower == "hero_wavelength" ||
                       lower == "hero-wavelength" || lower == "1") {
                options.spectralRenderingMode =
                    PathTracer::RenderSettings::SpectralRenderingMode::HeroWavelength;
            } else {
                error = "Invalid value for --spectralMode (expected off or hero_wavelength)";
                return false;
            }
            options.spectralRenderingModeSet = true;
        } else if (arg == "--spectralMinNm" || arg == "--spectral-min-nm") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.spectralMinWavelengthNm = std::stof(value);
                options.spectralMinWavelengthSet = true;
            } catch (...) {
                error = "Invalid float for --spectralMinNm";
                return false;
            }
        } else if (arg == "--spectralMaxNm" || arg == "--spectral-max-nm") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.spectralMaxWavelengthNm = std::stof(value);
                options.spectralMaxWavelengthSet = true;
            } catch (...) {
                error = "Invalid float for --spectralMaxNm";
                return false;
            }
        } else if (arg == "--spectralDispersionStrength" ||
                   arg == "--spectral-dispersion-strength") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.spectralDispersionStrength = std::stof(value);
                options.spectralDispersionStrengthSet = true;
            } catch (...) {
                error = "Invalid float for --spectralDispersionStrength";
                return false;
            }
        } else if (arg == "--spectralDebug" || arg == "--spectral-debug") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --spectralDebug";
                return false;
            }
            options.spectralDebug = flag;
            options.spectralDebugSet = true;
        } else if (arg == "--spectralWavelengthsPerPath" || arg == "--spectral-wavelengths-per-path") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            if (value != "1" && value != "3") {
                error = "Invalid value for --spectralWavelengthsPerPath (expected 1 or 3)";
                return false;
            }
            options.spectralWavelengthsPerPath = value == "1" ? 1u : 3u;
            options.spectralWavelengthsPerPathSet = true;
        } else if (arg == "--mlBackendMode" || arg == "--ml-backend-mode") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "off" || lower == "0" || lower == "disabled") {
                options.mlBackendMode = PathTracer::RenderSettings::MlBackendMode::Off;
            } else if (lower == "noop" || lower == "no_op" || lower == "reconstruction_noop" ||
                       lower == "reconstruction-noop" || lower == "1") {
                options.mlBackendMode =
                    PathTracer::RenderSettings::MlBackendMode::ReconstructionNoOp;
            } else if (lower == "tiny" || lower == "reconstruction_tiny" ||
                       lower == "reconstruction-tiny" || lower == "custom_metal_tiny" ||
                       lower == "custom-metal-tiny" || lower == "2") {
                options.mlBackendMode =
                    PathTracer::RenderSettings::MlBackendMode::ReconstructionTiny;
            } else {
                error = "Invalid value for --mlBackendMode (expected off, reconstruction_noop, or reconstruction_tiny)";
                return false;
            }
            options.mlBackendModeSet = true;
        } else if (arg == "--mlModelPackage" || arg == "--ml-model-package") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            options.mlModelPackagePath = value;
            options.mlModelPackagePathSet = true;
        } else if (arg == "--mlDebug" || arg == "--ml-debug") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --mlDebug";
                return false;
            }
            options.mlDebug = flag;
            options.mlDebugSet = true;
        } else if (arg == "--ris-m") {
            if (!requireValue("--ris-m")) {
                return false;
            }
            try {
                uint32_t parsed = static_cast<uint32_t>(std::stoul(value));
                if (parsed < 1u) {
                    error = "--ris-m must be >= 1";
                    return false;
                }
                options.risCandidateCount = parsed;
                options.risCandidateCountSet = true;
            } catch (...) {
                error = "Invalid integer for --ris-m";
                return false;
            }
        } else if (arg == "--spatial-reuse-k") {
            if (!requireValue("--spatial-reuse-k")) {
                return false;
            }
            try {
                uint32_t parsed = static_cast<uint32_t>(std::stoul(value));
                if (parsed < 1u) {
                    error = "--spatial-reuse-k must be >= 1";
                    return false;
                }
                options.spatialReuseNeighborCount = parsed;
                options.spatialReuseNeighborCountSet = true;
            } catch (...) {
                error = "Invalid integer for --spatial-reuse-k";
                return false;
            }
        } else if (arg == "--world-reuse-cell-size") {
            if (!requireValue("--world-reuse-cell-size")) {
                return false;
            }
            try {
                float parsed = std::stof(value);
                if (!(parsed > 0.0f) || !std::isfinite(parsed)) {
                    error = "--world-reuse-cell-size must be finite and > 0";
                    return false;
                }
                options.worldReuseCellSize = parsed;
                options.worldReuseCellSizeSet = true;
            } catch (...) {
                error = "Invalid float for --world-reuse-cell-size";
                return false;
            }
        } else if (arg == "--bias-check") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --bias-check";
                return false;
            }
            options.biasCheck = flag;
        } else if (arg == "--validate-ris") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --validate-ris";
                return false;
            }
            options.validateRis = flag;
        } else if (arg == "--validate-ris-frames") {
            if (!requireValue("--validate-ris-frames")) {
                return false;
            }
            try {
                uint32_t parsed = static_cast<uint32_t>(std::stoul(value));
                if (parsed < 1u) {
                    error = "--validate-ris-frames must be >= 1";
                    return false;
                }
                options.validateRisFrames = parsed;
                options.validateRisFramesSet = true;
            } catch (...) {
                error = "Invalid integer for --validate-ris-frames";
                return false;
            }
        } else if (arg == "--validate-ris-reference") {
            if (!requireValue("--validate-ris-reference")) {
                return false;
            }
            options.validateRisReferencePath = value;
            options.validateRisReferenceProvided = true;
        } else if (arg == "--validate-ris-psnr") {
            if (!requireValue("--validate-ris-psnr")) {
                return false;
            }
            try {
                options.validateRisPsnrThreshold = std::stof(value);
                options.validateRisPsnrThresholdSet = true;
            } catch (...) {
                error = "Invalid float for --validate-ris-psnr";
                return false;
            }
        } else if (arg == "--debugShowBaseColor") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --debugShowBaseColor";
                return false;
            }
            options.debugShowBaseColor = flag;
            options.debugShowBaseColorSet = true;
        } else if (arg == "--debugShowMetallic") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --debugShowMetallic";
                return false;
            }
            options.debugShowMetallic = flag;
            options.debugShowMetallicSet = true;
        } else if (arg == "--debugShowRoughness") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --debugShowRoughness";
                return false;
            }
            options.debugShowRoughness = flag;
            options.debugShowRoughnessSet = true;
        } else if (arg == "--debugShowAO") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --debugShowAO";
                return false;
            }
            options.debugShowAO = flag;
            options.debugShowAOSet = true;
        } else if (arg == "--debugDisableOrmTexture") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --debugDisableOrmTexture";
                return false;
            }
            options.debugDisableOrmTexture = flag;
            options.debugDisableOrmTextureSet = true;
        } else if (arg == "--dumpMaterialTexturesDir") {
            if (value.empty()) {
                error = "Missing value for --dumpMaterialTexturesDir";
                return false;
            }
            options.dumpMaterialTexturesDir = value;
            options.dumpMaterialTexturesDirProvided = true;
#if PT_DEBUG_TOOLS
        } else if (arg == "--debugPath") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --debugPath";
                return false;
            }
            options.debugPath = flag;
            options.debugPathSet = true;
        } else if (arg == "--debugPixel") {
            if (!requireValue("--debugPixel")) {
                return false;
            }
            if (!ParsePixelCoord(value, options.debugPixelX, options.debugPixelY)) {
                error = "Invalid value for --debugPixel (expected x,y)";
                return false;
            }
            options.debugPixelSet = true;
        } else if (arg == "--debugMaxEntries") {
            if (!requireValue("--debugMaxEntries")) {
                return false;
            }
            try {
                int parsed = std::stoi(value);
                if (parsed < 1) {
                    error = "--debugMaxEntries must be >= 1";
                    return false;
                }
                options.debugMaxEntries = static_cast<uint32_t>(parsed);
                options.debugMaxEntriesSet = true;
            } catch (...) {
                error = "Invalid integer for --debugMaxEntries";
                return false;
            }
        } else if (arg == "--parityAssert") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --parityAssert";
                return false;
            }
            options.parityAssert = flag;
            options.parityAssertSet = true;
        } else if (arg == "--parityPixel") {
            if (!requireValue("--parityPixel")) {
                return false;
            }
            if (!ParsePixelCoord(value, options.parityPixelX, options.parityPixelY)) {
                error = "Invalid value for --parityPixel (expected x,y)";
                return false;
            }
            options.parityPixelSet = true;
        } else if (arg == "--parityOncePerFrame") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --parityOncePerFrame";
                return false;
            }
            options.parityOncePerFrame = flag;
            options.parityOncePerFrameSet = true;
        } else if (arg == "--parityMode") {
            if (!requireValue("--parityMode")) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "probe" || lower == "pixel") {
                options.parityMode = PathTracer::RenderSettings::ParityAssertMode::ProbePixelOnly;
            } else if (lower == "medium" || lower == "firstinmedium") {
                options.parityMode = PathTracer::RenderSettings::ParityAssertMode::FirstInMediumBoundary;
            } else {
                error = "Invalid value for --parityMode (expected probe or medium)";
                return false;
            }
            options.parityModeSet = true;
        } else if (arg == "--forcePureHWRTForGlass") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --forcePureHWRTForGlass";
                return false;
            }
            options.forcePureHWRTForGlass = flag;
            options.forcePureHWRTForGlassSet = true;
        } else if (arg == "--writeDenoiseGuides") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --writeDenoiseGuides";
                return false;
            }
            options.writeDenoiseGuides = flag;
        } else if (arg == "--debugDisableNormalMap") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --debugDisableNormalMap";
                return false;
            }
            options.debugDisableNormalMap = flag;
            options.debugDisableNormalMapSet = true;
        } else if (arg == "--debugSpecularOnly") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --debugSpecularOnly";
                return false;
            }
            options.debugSpecularOnly = flag;
            options.debugSpecularOnlySet = true;
        } else if (arg == "--parityDepth") {
            if (!requireValue("--parityDepth")) {
                return false;
            }
            try {
                options.parityTargetDepth = static_cast<uint32_t>(std::stoul(value));
                options.parityTargetDepthSet = true;
            } catch (...) {
                error = "Invalid integer for --parityDepth";
                return false;
            }
#endif
        } else if (arg == "--pbrMetrics") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --pbrMetrics";
                return false;
            }
            options.pbrMetrics = flag;
        } else if (arg == "--pbrMetricsJson") {
            if (!requireValue("--pbrMetricsJson")) {
                return false;
            }
            options.pbrMetricsJsonPath = value;
            options.pbrMetricsJsonProvided = true;
        } else if (arg == "--outputMetadata") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --outputMetadata";
                return false;
            }
            options.outputMetadata = flag;
            options.outputMetadataSet = true;
        } else if (arg == "--outputMetadataJson") {
            if (!requireValue("--outputMetadataJson")) {
                return false;
            }
            options.outputMetadataJsonPath = value;
            options.outputMetadataJsonProvided = true;
            options.outputMetadata = true;
        } else if (arg == "--settingsJson") {
            if (!requireValue("--settingsJson")) {
                return false;
            }
            options.settingsJsonPath = value;
            options.settingsJsonProvided = true;
        } else if (arg == "--renderQueueItemJson") {
            if (!requireValue("--renderQueueItemJson")) {
                return false;
            }
            options.renderQueueItemJsonPath = value;
            options.renderQueueItemJsonProvided = true;
        } else if (arg == "--runRenderQueueItemJson" || arg == "--renderQueueItem") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            options.runRenderQueueItemJsonPath = value;
            options.runRenderQueueItemJsonProvided = true;
        } else if (arg == "--debugBundleDir" || arg == "--debug-bundle-dir") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            options.debugBundleDir = value;
            options.debugBundleDirProvided = true;
            options.outputMetadata = true;
            options.pbrMetrics = true;
        } else if (arg == "--checkpointManifestJson" || arg == "--checkpoint-manifest-json") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            options.checkpointManifestJsonPath = value;
            options.checkpointManifestJsonProvided = true;
        } else if (arg == "--resumeCheckpointJson" || arg == "--resume-checkpoint-json") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            options.resumeCheckpointJsonPath = value;
            options.resumeCheckpointJsonProvided = true;
        } else if (arg == "--checkpointEverySamples" || arg == "--checkpoint-every-samples") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            try {
                options.checkpointEverySamples = static_cast<uint32_t>(std::stoul(value));
            } catch (...) {
                error = "Invalid integer for --checkpointEverySamples";
                return false;
            }
        } else if (arg == "--tileManifestJson" || arg == "--tile-manifest-json") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            options.tileManifestJsonPath = value;
            options.tileManifestJsonProvided = true;
        } else if (arg == "--tileSize" || arg == "--tile-size") {
            if (!requireValue(arg.c_str())) {
                return false;
            }
            uint32_t parsedWidth = 0;
            uint32_t parsedHeight = 0;
            if (!ParsePixelCoord(value, parsedWidth, parsedHeight) || parsedWidth == 0u || parsedHeight == 0u) {
                error = "Invalid value for " + arg + " (expected width,height)";
                return false;
            }
            options.tileWidth = parsedWidth;
            options.tileHeight = parsedHeight;
            options.tileSizeSet = true;
        } else if (arg == "--materialMrDebugJson") {
            if (!requireValue("--materialMrDebugJson")) {
                return false;
            }
            options.materialMrDebugJsonPath = value;
            options.materialMrDebugJsonProvided = true;
        } else if (arg == "--directLightAuditJson") {
            if (!requireValue("--directLightAuditJson")) {
                return false;
            }
            options.directLightAuditJsonPath = value;
            options.directLightAuditJsonProvided = true;
        } else if (arg == "--restirDebug") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --restirDebug";
                return false;
            }
            options.restirDebug = flag;
            options.restirDebugSet = true;
        } else if (arg == "--restirDebugView") {
            if (!requireValue("--restirDebugView")) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "beauty" || lower == "off" || lower == "0") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::Beauty;
            } else if (lower == "candidate_source_id" || lower == "candidate" || lower == "selected_light_id") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::CandidateSourceId;
            } else if (lower == "reservoir_confidence" || lower == "reservoir_weight") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::ReservoirConfidence;
            } else if (lower == "regir_cell_id" || lower == "regir_confidence") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::ReGIRCellId;
            } else if (lower == "path_guiding_used_mask" || lower == "path_guiding") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::PathGuidingUsedMask;
            } else if (lower == "restir_pt_reuse_mask" || lower == "restir_pt") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::RestirPTReuseMask;
            } else if (lower == "svgf_variance" || lower == "svgf") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::SVGFVariance;
            } else if (lower == "nan_inf_mask" || lower == "naninf") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::NaNInfMask;
            } else if (lower == "queue_stage_failure" || lower == "queue_failure" || lower == "queue") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::QueueStageFailure;
            } else if (lower == "radiance_cache" || lower == "cache") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::RadianceCache;
            } else if (lower == "caustic_transport" || lower == "caustics" || lower == "caustic") {
                options.restirDebugView = PathTracer::RenderSettings::RestirDebugView::CausticTransport;
            } else {
                error = "Invalid value for --restirDebugView";
                return false;
            }
            options.restirDebugViewSet = true;
        } else if (arg == "--restirDebugCounters") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --restirDebugCounters";
                return false;
            }
            options.restirDebugCounters = flag;
            options.restirDebugCountersSet = true;
        } else if (arg == "--restirDebugMetricsJson") {
            if (!requireValue("--restirDebugMetricsJson")) {
                return false;
            }
            options.restirDebugMetricsJsonPath = value;
            options.restirDebugMetricsJsonProvided = true;
        } else if (arg == "--cameraSearchReport") {
            if (!requireValue("--cameraSearchReport")) {
                return false;
            }
            options.cameraSearchReportPath = value;
            options.cameraSearchReportProvided = true;
        } else if (arg == "--format") {
            if (!requireValue("--format")) {
                return false;
            }
            options.formatString = value;
        } else if (arg == "--denoiseUseAlbedo") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --denoise-use-albedo";
                return false;
            }
            options.denoiseUseAlbedo = flag;
            options.denoiseUseAlbedoSet = true;
        } else if (arg == "--denoiseUseNormal") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --denoise-use-normal";
                return false;
            }
            options.denoiseUseNormal = flag;
            options.denoiseUseNormalSet = true;
        } else if (arg == "--denoiseHdr") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --denoise-hdr";
                return false;
            }
            options.denoiseHdr = flag;
            options.denoiseHdrSet = true;
        } else if (arg == "--backend") {
            if (!requireValue("--backend")) {
                return false;
            }
            std::string lower;
            lower.reserve(value.size());
            for (char c : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
            }
            if (lower == "metal") {
                options.backend = HeadlessBackend::Metal;
            } else if (lower == "embree") {
                options.backend = HeadlessBackend::Embree;
            } else {
                error = "Invalid value for --backend (expected metal or embree)";
                return false;
            }
            options.backendSet = true;
        } else if (arg == "--enableEmbree") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --enableEmbree";
                return false;
            }
            options.backend = flag ? HeadlessBackend::Embree : HeadlessBackend::Metal;
            options.backendSet = true;
        } else {
            error = "Unknown option: " + arg;
            return false;
        }
    }

    if (options.runRenderQueueItemJsonProvided &&
        !ApplyRenderQueueJson(fs::path(options.runRenderQueueItemJsonPath), options, error)) {
        return false;
    }
    if (options.resumeCheckpointJsonProvided &&
        !ApplyCheckpointJson(fs::path(options.resumeCheckpointJsonPath), options, error)) {
        return false;
    }
    if (!ApplyRenderModeDefaults(options, error)) {
        return false;
    }

    if (!options.sceneProvided) {
        error = "--scene is required";
        return false;
    }

    std::string formatError;
    if (!PathTracer::ParseImageFileFormat(options.formatString, options.format)) {
        error = "Unknown format: " + options.formatString;
        return false;
    }

    fs::path scenePath(options.scene);
    options.sceneIsPath = scenePath.extension() == ".scene" || scenePath.has_parent_path() || scenePath.is_absolute();
    if (!options.sceneIsPath) {
        std::error_code sceneEc;
        if (fs::exists(scenePath, sceneEc)) {
            options.sceneIsPath = true;
        }
    }

    const bool explicitSwrt =
        (options.enableSoftwareRayTracingSet && options.enableSoftwareRayTracing) ||
        (options.forceSoftwareRayTracingSet && options.forceSoftwareRayTracing);
    const bool explicitHwrt =
        options.forceHardwareRayTracingSet && options.forceHardwareRayTracing;
    if (explicitSwrt && explicitHwrt) {
        error = "Conflicting options: --force-hwrt cannot be combined with --force-swrt/--enableSoftwareRayTracing=1";
        return false;
    }
    if (options.threadsSet && options.backend != HeadlessBackend::Embree) {
        error = "--threads is supported only with --backend=embree";
        return false;
    }
    if (options.backend == HeadlessBackend::Embree && (explicitSwrt || explicitHwrt)) {
        error = "Metal ray-tracing override flags are not valid with --backend=embree";
        return false;
    }
    if (options.fireflyClampMode != CliOptions::FireflyClampMode::Off &&
        !(options.fireflyClampValue > 0.0f)) {
        error = "--firefly-clamp-value must be > 0 when --firefly-clamp-mode is enabled";
        return false;
    }
    return true;
}

std::string SanitizeSceneName(const std::string& input) {
    if (input.empty()) {
        return "scene";
    }
    fs::path p(input);
    std::string name = p.stem().string();
    if (name.empty()) {
        name = input;
    }
    for (char& c : name) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')) {
            c = '_';
        }
    }
    return name;
}

const char* DirectLightModeLabel(PathTracer::RenderSettings::DirectLightMode mode) {
    switch (mode) {
        case PathTracer::RenderSettings::DirectLightMode::LegacyRect:
            return "legacy";
        case PathTracer::RenderSettings::DirectLightMode::BaselineEmissive:
            return "baseline_emissive";
        case PathTracer::RenderSettings::DirectLightMode::Ris:
            return "ris";
        case PathTracer::RenderSettings::DirectLightMode::RisSpatialReuse:
            return "ris_spatial";
        case PathTracer::RenderSettings::DirectLightMode::RisTemporalReuse:
            return "ris_temporal";
        case PathTracer::RenderSettings::DirectLightMode::RisWorldReuse:
            return "ris_world";
        case PathTracer::RenderSettings::DirectLightMode::RisRegirCache:
            return "ris_regir";
        case PathTracer::RenderSettings::DirectLightMode::RestirDi:
            return "restir_di";
        case PathTracer::RenderSettings::DirectLightMode::RestirDiRegirHybrid:
            return "restir_di_regir_hybrid";
    }
    return "unknown";
}

// The direct-light mode the backend runs: on both backends, homogeneous volumes demote ReSTIR DI to baseline_emissive
// (RestirDiDemotedByVolume; Embree mirrors Metal since EP9).
PathTracer::RenderSettings::DirectLightMode RunDirectLightMode(const CliOptions& options,
                                                               const PathTracer::RenderSettings& settings) {
    (void)options;
    return PathTracer::MetalDirectLightMode(settings);
}

// Direct-light mode the backend could not run as requested because of the medium, for the render metadata: "none",
// or "<mode> ran as baseline_emissive".
std::string VolumeDirectLightFallbackLabel(const CliOptions& options, const PathTracer::RenderSettings& settings) {
    const PathTracer::RenderSettings::DirectLightMode ran = RunDirectLightMode(options, settings);
    if (ran == settings.directLightMode) {
        return "none";
    }
    return std::string(DirectLightModeLabel(settings.directLightMode)) + " ran as " + DirectLightModeLabel(ran);
}

const char* WavefrontSchedulingPolicyLabel(
    PathTracer::RenderSettings::WavefrontSchedulingPolicy policy) {
    switch (policy) {
        case PathTracer::RenderSettings::WavefrontSchedulingPolicy::Preview:
            return "preview";
        case PathTracer::RenderSettings::WavefrontSchedulingPolicy::Final:
            return "final";
        case PathTracer::RenderSettings::WavefrontSchedulingPolicy::Offline:
            return "offline";
        case PathTracer::RenderSettings::WavefrontSchedulingPolicy::Research:
            return "research";
    }
    return "unknown";
}

const char* RestirGiModeLabel(PathTracer::RenderSettings::RestirGiMode mode) {
    switch (mode) {
        case PathTracer::RenderSettings::RestirGiMode::Off:
            return "off";
        case PathTracer::RenderSettings::RestirGiMode::ProductionDiffuse:
            return "restir_gi_production";
    }
    return "unknown";
}

const char* PathGuidingModeLabel(PathTracer::RenderSettings::PathGuidingMode mode) {
    switch (mode) {
        case PathTracer::RenderSettings::PathGuidingMode::Off:
            return "off";
        case PathTracer::RenderSettings::PathGuidingMode::Prototype:
            return "path_guiding";
    }
    return "unknown";
}

const char* RestirPtModeLabel(PathTracer::RenderSettings::RestirPtMode mode) {
    switch (mode) {
        case PathTracer::RenderSettings::RestirPtMode::Off:
            return "off";
        case PathTracer::RenderSettings::RestirPtMode::ProductionReservoirCapture:
            return "restir_pt_reservoir_capture";
        case PathTracer::RenderSettings::RestirPtMode::ProductionPathReuse:
            return "restir_pt_path_reuse";
        case PathTracer::RenderSettings::RestirPtMode::EnhancedUnbiased:
            return "restir_pt_enhanced_unbiased";
        case PathTracer::RenderSettings::RestirPtMode::EnhancedBiased:
            return "restir_pt_enhanced_biased";
    }
    return "unknown";
}

bool RestirPtEnhancedActive(PathTracer::RenderSettings::RestirPtMode mode) {
    return mode == PathTracer::RenderSettings::RestirPtMode::EnhancedUnbiased ||
           mode == PathTracer::RenderSettings::RestirPtMode::EnhancedBiased;
}

const char* RadianceCacheModeLabel(PathTracer::RenderSettings::RadianceCacheMode mode) {
    switch (mode) {
        case PathTracer::RenderSettings::RadianceCacheMode::Off:
            return "off";
        case PathTracer::RenderSettings::RadianceCacheMode::Prototype:
            return "radiance_cache";
    }
    return "unknown";
}

const char* CausticTransportModeLabel(PathTracer::RenderSettings::CausticTransportMode mode) {
    switch (mode) {
        case PathTracer::RenderSettings::CausticTransportMode::Off:
            return "off";
        case PathTracer::RenderSettings::CausticTransportMode::PathSpacePrototype:
            return "path_space";
    }
    return "unknown";
}

const char* RenderProfileLabel(RenderProfile profile) {
    switch (profile) {
        case RenderProfile::Preview:
            return "preview";
        case RenderProfile::Lookdev:
            return "lookdev";
        case RenderProfile::Final:
            return "final";
        case RenderProfile::Reference:
            return "reference";
        case RenderProfile::Debug:
            return "debug";
        case RenderProfile::Custom:
        default:
            return "custom";
    }
}

const char* RenderProfileMaturityLabel(RenderProfile profile) {
    switch (profile) {
        case RenderProfile::Preview:
            return "interactive_preview";
        case RenderProfile::Lookdev:
            return "production_lookdev";
        case RenderProfile::Final:
            return "production_final";
        case RenderProfile::Reference:
            return "deterministic_reference";
        case RenderProfile::Debug:
            return "diagnostic_debug";
        case RenderProfile::Custom:
        default:
            return "custom";
    }
}

const char* RenderModeLabel(RenderMode mode) {
    switch (mode) {
        case RenderMode::MetalFinalFrame:
            return "metal_final_frame";
        case RenderMode::CpuReferenceHeadless:
            return "cpu_reference_headless";
        case RenderMode::Auto:
        default:
            return "auto";
    }
}

const char* RenderModeMaturityLabel(RenderMode mode) {
    switch (mode) {
        case RenderMode::MetalFinalFrame:
            return "current_aligned_scope_limited";
        case RenderMode::CpuReferenceHeadless:
            return "current_foundation";
        case RenderMode::Auto:
        default:
            return "inferred_from_backend";
    }
}

const char* RenderModeContextLabel(RenderMode mode) {
    switch (mode) {
        case RenderMode::MetalFinalFrame:
            return "headless_or_controlled_final_frame";
        case RenderMode::CpuReferenceHeadless:
            return "headless_offline_reference";
        case RenderMode::Auto:
        default:
            return "inferred";
    }
}

const char* VolumeTransportModeLabel(PathTracer::RenderSettings::VolumeTransportMode mode) {
    switch (mode) {
        case PathTracer::RenderSettings::VolumeTransportMode::Off:
            return "off";
        case PathTracer::RenderSettings::VolumeTransportMode::Homogeneous:
            return "homogeneous";
    }
    return "unknown";
}

const char* VolumePhaseFunctionLabel(PathTracer::RenderSettings::VolumePhaseFunction phase) {
    switch (phase) {
        case PathTracer::RenderSettings::VolumePhaseFunction::Isotropic:
            return "isotropic";
        case PathTracer::RenderSettings::VolumePhaseFunction::HenyeyGreenstein:
            return "henyey_greenstein";
    }
    return "unknown";
}

const char* SpectralRenderingModeLabel(PathTracer::RenderSettings::SpectralRenderingMode mode) {
    switch (mode) {
        case PathTracer::RenderSettings::SpectralRenderingMode::Off:
            return "off";
        case PathTracer::RenderSettings::SpectralRenderingMode::HeroWavelength:
            return "hero_wavelength";
    }
    return "unknown";
}

const char* SpectralTexturePolicyLabel(PathTracer::RenderSettings::SpectralTexturePolicy policy) {
    switch (policy) {
        case PathTracer::RenderSettings::SpectralTexturePolicy::RGBReflectanceD65:
            return "rgb_reflectance_d65";
    }
    return "unknown";
}

const char* MlBackendModeLabel(PathTracer::RenderSettings::MlBackendMode mode) {
    switch (mode) {
        case PathTracer::RenderSettings::MlBackendMode::Off:
            return "off";
        case PathTracer::RenderSettings::MlBackendMode::ReconstructionNoOp:
            return "reconstruction_noop";
        case PathTracer::RenderSettings::MlBackendMode::ReconstructionTiny:
            return "reconstruction_tiny";
    }
    return "unknown";
}

const char* MlBackendConfiguredProvider(PathTracer::RenderSettings::MlBackendMode mode) {
    switch (mode) {
        case PathTracer::RenderSettings::MlBackendMode::Off:
            return "disabled";
        case PathTracer::RenderSettings::MlBackendMode::ReconstructionNoOp:
            return "MLBackend.NoOp";
        case PathTracer::RenderSettings::MlBackendMode::ReconstructionTiny:
            return "MLBackend.CustomMetal.TinyReconstruction";
    }
    return "unknown";
}

std::optional<simd::float3> ParseFloat3Option(std::string value) {
    std::replace(value.begin(), value.end(), ',', ' ');
    std::istringstream stream(value);
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    if (!(stream >> x >> y >> z)) {
        return std::nullopt;
    }
    return simd_make_float3(x, y, z);
}

const char* RestirDebugViewLabel(PathTracer::RenderSettings::RestirDebugView view) {
    switch (view) {
        case PathTracer::RenderSettings::RestirDebugView::Beauty:
            return "beauty";
        case PathTracer::RenderSettings::RestirDebugView::CandidateSourceId:
            return "candidate_source_id";
        case PathTracer::RenderSettings::RestirDebugView::ReservoirConfidence:
            return "reservoir_confidence";
        case PathTracer::RenderSettings::RestirDebugView::ReGIRCellId:
            return "regir_cell_id";
        case PathTracer::RenderSettings::RestirDebugView::PathGuidingUsedMask:
            return "path_guiding_used_mask";
        case PathTracer::RenderSettings::RestirDebugView::RestirPTReuseMask:
            return "restir_pt_reuse_mask";
        case PathTracer::RenderSettings::RestirDebugView::SVGFVariance:
            return "svgf_variance";
        case PathTracer::RenderSettings::RestirDebugView::NaNInfMask:
            return "nan_inf_mask";
        case PathTracer::RenderSettings::RestirDebugView::QueueStageFailure:
            return "queue_stage_failure";
        case PathTracer::RenderSettings::RestirDebugView::RadianceCache:
            return "radiance_cache";
        case PathTracer::RenderSettings::RestirDebugView::CausticTransport:
            return "caustic_transport";
    }
    return "unknown";
}
struct RuntimeBounds {
    simd::float3 min = simd_make_float3(0.0f, 0.0f, 0.0f);
    simd::float3 max = simd_make_float3(0.0f, 0.0f, 0.0f);
    simd::float3 center = simd_make_float3(0.0f, 0.0f, 0.0f);
    bool valid = false;
};

struct RuntimeMeshSnapshot {
    const PathTracer::SceneResources::Mesh* mesh = nullptr;
    RuntimeBounds localBounds;
    RuntimeBounds importedWorldBounds;
    RuntimeBounds renderWorldBounds;
    RuntimeBounds tlasWorldBounds;
};

struct RuntimeLandmark {
    std::string name;
    RuntimeBounds bounds;
};

simd::float3 TransformPoint(const simd::float4x4& m, const simd::float3& p) {
    simd::float4 r = simd_mul(m, simd_make_float4(p, 1.0f));
    return simd_make_float3(r.x, r.y, r.z);
}

RuntimeBounds ComputeMeshWorldBounds(const PathTracer::SceneResources::Mesh& mesh) {
    RuntimeBounds bounds;
    if (mesh.vertices.empty()) {
        return bounds;
    }
    simd::float3 minP = simd_make_float3(std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max());
    simd::float3 maxP = simd_make_float3(std::numeric_limits<float>::lowest(),
                                         std::numeric_limits<float>::lowest(),
                                         std::numeric_limits<float>::lowest());
    for (const auto& vertex : mesh.vertices) {
        const simd::float3 p = TransformPoint(mesh.localToWorld, vertex.position);
        minP = simd_min(minP, p);
        maxP = simd_max(maxP, p);
    }
    bounds.min = minP;
    bounds.max = maxP;
    bounds.center = 0.5f * (minP + maxP);
    bounds.valid = true;
    return bounds;
}

RuntimeBounds ComputeMeshLocalBounds(const PathTracer::SceneResources::Mesh& mesh) {
    RuntimeBounds bounds;
    if (mesh.vertices.empty()) {
        return bounds;
    }
    simd::float3 minP = simd_make_float3(std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max());
    simd::float3 maxP = simd_make_float3(std::numeric_limits<float>::lowest(),
                                         std::numeric_limits<float>::lowest(),
                                         std::numeric_limits<float>::lowest());
    for (const auto& vertex : mesh.vertices) {
        minP = simd_min(minP, vertex.position);
        maxP = simd_max(maxP, vertex.position);
    }
    bounds.min = minP;
    bounds.max = maxP;
    bounds.center = 0.5f * (minP + maxP);
    bounds.valid = true;
    return bounds;
}

RuntimeBounds TransformBounds(const RuntimeBounds& localBounds, const simd::float4x4& transform) {
    RuntimeBounds bounds;
    if (!localBounds.valid) {
        return bounds;
    }

    const simd::float4 corners[8] = {
        simd_make_float4(localBounds.min.x, localBounds.min.y, localBounds.min.z, 1.0f),
        simd_make_float4(localBounds.max.x, localBounds.min.y, localBounds.min.z, 1.0f),
        simd_make_float4(localBounds.min.x, localBounds.max.y, localBounds.min.z, 1.0f),
        simd_make_float4(localBounds.max.x, localBounds.max.y, localBounds.min.z, 1.0f),
        simd_make_float4(localBounds.min.x, localBounds.min.y, localBounds.max.z, 1.0f),
        simd_make_float4(localBounds.max.x, localBounds.min.y, localBounds.max.z, 1.0f),
        simd_make_float4(localBounds.min.x, localBounds.max.y, localBounds.max.z, 1.0f),
        simd_make_float4(localBounds.max.x, localBounds.max.y, localBounds.max.z, 1.0f),
    };

    simd::float3 minP = simd_make_float3(std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max());
    simd::float3 maxP = simd_make_float3(std::numeric_limits<float>::lowest(),
                                         std::numeric_limits<float>::lowest(),
                                         std::numeric_limits<float>::lowest());
    for (const simd::float4& corner : corners) {
        const simd::float4 worldCorner = simd_mul(transform, corner);
        const simd::float3 point = simd_make_float3(worldCorner.x, worldCorner.y, worldCorner.z);
        minP = simd_min(minP, point);
        maxP = simd_max(maxP, point);
    }
    bounds.min = minP;
    bounds.max = maxP;
    bounds.center = 0.5f * (minP + maxP);
    bounds.valid = true;
    return bounds;
}

RuntimeMeshSnapshot MakeRuntimeMeshSnapshot(const PathTracer::SceneResources::Mesh& mesh) {
    RuntimeMeshSnapshot snapshot;
    snapshot.mesh = &mesh;
    snapshot.localBounds = ComputeMeshLocalBounds(mesh);
    snapshot.importedWorldBounds = TransformBounds(snapshot.localBounds, mesh.defaultLocalToWorld);
    snapshot.renderWorldBounds = ComputeMeshWorldBounds(mesh);
    snapshot.tlasWorldBounds = TransformBounds(snapshot.localBounds, mesh.localToWorld);
    return snapshot;
}

std::string ToLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string TrimAscii(std::string value) {
    auto notSpace = [](unsigned char c) {
        return !std::isspace(c);
    };
    value.erase(value.begin(),
                std::find_if(value.begin(), value.end(), [&](char c) {
                    return notSpace(static_cast<unsigned char>(c));
                }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](char c) {
                    return notSpace(static_cast<unsigned char>(c));
                }).base(),
                value.end());
    return value;
}

bool HasDirectiveKeyword(const std::string& line, const char* keyword) {
    const size_t keywordLen = std::char_traits<char>::length(keyword);
    if (line.size() < keywordLen || line.compare(0, keywordLen, keyword) != 0) {
        return false;
    }
    return line.size() == keywordLen || std::isspace(static_cast<unsigned char>(line[keywordLen]));
}

std::string ExtractDirectiveValue(const std::string& line, const char* key) {
    std::istringstream stream(line);
    std::string word;
    stream >> word;
    while (stream >> word) {
        const std::string prefix = std::string(key) + "=";
        if (word.rfind(prefix, 0) == 0) {
            return word.substr(prefix.size());
        }
    }
    return {};
}

bool HasDirectiveToken(const std::string& line, const char* key) {
    return !ExtractDirectiveValue(line, key).empty();
}

fs::path WeaklyCanonicalIfPossible(const fs::path& path) {
    std::error_code ec;
    fs::path canonical = fs::weakly_canonical(path, ec);
    if (!ec) {
        return canonical;
    }
    return path;
}

fs::path ResolveScenePathFromRequest(const CliOptions& options,
                                     const PathTracer::SceneManager& sceneManager) {
    if (options.sceneIsPath) {
        return WeaklyCanonicalIfPossible(fs::path(options.scene));
    }
    if (const auto* info = sceneManager.currentScene()) {
        return WeaklyCanonicalIfPossible(fs::path(info->filePath));
    }
    return {};
}

fs::path ResolveMeshAssetPath(const fs::path& sceneFilePath, const std::string& rawValue) {
    fs::path meshPath(rawValue);
    if (meshPath.is_relative()) {
        meshPath = sceneFilePath.parent_path() / meshPath;
    }
    return WeaklyCanonicalIfPossible(meshPath);
}

fs::path ResolveEnvironmentAssetPath(const fs::path& sceneFilePath, const std::string& rawValue) {
    fs::path envPath(rawValue);
    if (envPath.is_relative()) {
        if (envPath.has_parent_path()) {
            envPath = sceneFilePath.parent_path() / envPath;
        } else {
            envPath = sceneFilePath.parent_path() / "HDR" / envPath;
        }
    }
    return WeaklyCanonicalIfPossible(envPath);
}

struct SceneDirectiveDiagnostics {
    fs::path resolvedSceneFilePath;
    std::vector<std::string> rawCameraLines;
    std::vector<std::string> rawTransformLines;
    std::vector<std::string> rawIncludeInheritLines;
    std::vector<fs::path> resolvedModelPaths;
    std::vector<fs::path> resolvedEnvironmentPaths;
};

SceneDirectiveDiagnostics InspectSceneDirectives(const fs::path& sceneFilePath) {
    SceneDirectiveDiagnostics diagnostics;
    diagnostics.resolvedSceneFilePath = sceneFilePath;

    std::ifstream stream(sceneFilePath);
    if (!stream.is_open()) {
        return diagnostics;
    }

    std::unordered_set<std::string> seenModelPaths;
    std::unordered_set<std::string> seenEnvironmentPaths;
    std::string line;
    while (std::getline(stream, line)) {
        std::string trimmed = TrimAscii(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }

        if (HasDirectiveKeyword(trimmed, "include") || HasDirectiveKeyword(trimmed, "inherit")) {
            diagnostics.rawIncludeInheritLines.push_back(trimmed);
        }
        if (HasDirectiveKeyword(trimmed, "camera")) {
            diagnostics.rawCameraLines.push_back(trimmed);
        }
        if (HasDirectiveKeyword(trimmed, "background")) {
            const std::string envValue = ExtractDirectiveValue(trimmed, "env");
            if (!envValue.empty()) {
                const fs::path resolved = ResolveEnvironmentAssetPath(sceneFilePath, envValue);
                const std::string key = resolved.string();
                if (seenEnvironmentPaths.insert(key).second) {
                    diagnostics.resolvedEnvironmentPaths.push_back(resolved);
                }
            }
        }
        if (HasDirectiveKeyword(trimmed, "mesh")) {
            const std::string pathValue = [&]() {
                const std::string pathToken = ExtractDirectiveValue(trimmed, "path");
                if (!pathToken.empty()) {
                    return pathToken;
                }
                return ExtractDirectiveValue(trimmed, "file");
            }();
            if (!pathValue.empty()) {
                const fs::path resolved = ResolveMeshAssetPath(sceneFilePath, pathValue);
                const std::string key = resolved.string();
                if (seenModelPaths.insert(key).second) {
                    diagnostics.resolvedModelPaths.push_back(resolved);
                }
            }
            if (HasDirectiveToken(trimmed, "translate") ||
                HasDirectiveToken(trimmed, "position") ||
                HasDirectiveToken(trimmed, "rotate") ||
                HasDirectiveToken(trimmed, "scale")) {
                diagnostics.rawTransformLines.push_back(trimmed);
            }
        }
    }

    return diagnostics;
}

std::string ClassifyLoadedRootAsset(const PathTracer::SceneResources& resources,
                                    const SceneDirectiveDiagnostics& diagnostics) {
    size_t interiorHits = 0;
    size_t exteriorHits = 0;
    for (const auto& mesh : resources.meshes()) {
        const std::string lower = ToLowerAscii(mesh.name);
        if (lower.find("interior") != std::string::npos) {
            ++interiorHits;
        }
        if (lower.find("exterior") != std::string::npos) {
            ++exteriorHits;
        }
    }

    for (const fs::path& modelPath : diagnostics.resolvedModelPaths) {
        const std::string lower = ToLowerAscii(modelPath.string());
        if (lower.find("bistro_interior") != std::string::npos ||
            lower.find("interior") != std::string::npos) {
            interiorHits += 1000;
        }
        if (lower.find("bistro_exterior") != std::string::npos ||
            lower.find("exterior") != std::string::npos) {
            exteriorHits += 1000;
        }
    }

    if (interiorHits > exteriorHits && interiorHits > 0) {
        return "interior";
    }
    if (exteriorHits > interiorHits && exteriorHits > 0) {
        return "exterior";
    }
    return "unknown";
}

void PrintSceneIdentityReport(const CliOptions& options,
                              const PathTracer::SceneManager& sceneManager,
                              const PathTracer::SceneResources& resources) {
    const fs::path resolvedSceneFilePath = ResolveScenePathFromRequest(options, sceneManager);
    const SceneDirectiveDiagnostics diagnostics = InspectSceneDirectives(resolvedSceneFilePath);

    std::cout << "[SceneResolve] requested=" << options.scene << std::endl;
    std::cout << "[SceneResolve] resolvedSceneFile="
              << (resolvedSceneFilePath.empty() ? std::string("<unresolved>") : resolvedSceneFilePath.string())
              << std::endl;

    if (diagnostics.resolvedModelPaths.empty()) {
        std::cout << "[SceneResolve] resolvedModelPaths=<none>" << std::endl;
    } else {
        for (size_t i = 0; i < diagnostics.resolvedModelPaths.size(); ++i) {
            std::cout << "[SceneResolve] resolvedModelPath[" << (i + 1u) << "]="
                      << diagnostics.resolvedModelPaths[i].string() << std::endl;
        }
    }

    if (diagnostics.resolvedEnvironmentPaths.empty()) {
        std::cout << "[SceneResolve] resolvedEnvironmentPaths=<none>" << std::endl;
    } else {
        for (size_t i = 0; i < diagnostics.resolvedEnvironmentPaths.size(); ++i) {
            std::cout << "[SceneResolve] resolvedEnvironmentPath[" << (i + 1u) << "]="
                      << diagnostics.resolvedEnvironmentPaths[i].string() << std::endl;
        }
    }

    if (diagnostics.rawIncludeInheritLines.empty()) {
        std::cout << "[SceneResolve] includeOrInherit=<none>" << std::endl;
    } else {
        for (size_t i = 0; i < diagnostics.rawIncludeInheritLines.size(); ++i) {
            std::cout << "[SceneResolve] includeOrInherit[" << (i + 1u) << "]="
                      << diagnostics.rawIncludeInheritLines[i] << std::endl;
        }
    }

    if (diagnostics.rawCameraLines.empty()) {
        std::cout << "[SceneResolve] cameraOverrideLines=<none>" << std::endl;
    } else {
        for (size_t i = 0; i < diagnostics.rawCameraLines.size(); ++i) {
            std::cout << "[SceneResolve] cameraOverrideLine[" << (i + 1u) << "]="
                      << diagnostics.rawCameraLines[i] << std::endl;
        }
    }

    if (diagnostics.rawTransformLines.empty()) {
        std::cout << "[SceneResolve] sceneLevelTransformLines=<none>" << std::endl;
    } else {
        for (size_t i = 0; i < diagnostics.rawTransformLines.size(); ++i) {
            std::cout << "[SceneResolve] sceneLevelTransformLine[" << (i + 1u) << "]="
                      << diagnostics.rawTransformLines[i] << std::endl;
        }
    }

    std::cout << "[SceneResolve] loadedRootAssetClass="
              << ClassifyLoadedRootAsset(resources, diagnostics) << std::endl;
    std::cout << "[SceneResolve] meshCount=" << resources.meshes().size() << std::endl;
    const size_t maxMeshes = std::min<size_t>(20u, resources.meshes().size());
    for (size_t i = 0; i < maxMeshes; ++i) {
        std::cout << "[SceneResolve][Mesh " << (i + 1u) << "] "
                  << resources.meshes()[i].name << std::endl;
    }
}

void EnsureSceneDirectiveAssetPipelineReport(const CliOptions& options,
                                             const PathTracer::SceneManager& sceneManager,
                                             PathTracer::SceneResources& resources) {
    const auto& existingReport = resources.assetPipelineReport();
    if (!existingReport.manifest.sourcePath.empty()) {
        return;
    }

    const fs::path resolvedSceneFilePath = ResolveScenePathFromRequest(options, sceneManager);
    const SceneDirectiveDiagnostics diagnostics = InspectSceneDirectives(resolvedSceneFilePath);

    PathTracer::AssetPipelineReport report{};
    report.manifest.sourcePath = resolvedSceneFilePath.empty() ? options.scene : resolvedSceneFilePath.string();
    report.manifest.sourceFormat = resolvedSceneFilePath.has_extension()
        ? ToLowerAscii(resolvedSceneFilePath.extension().string())
        : "scene";
    report.manifest.importer = "PathTracer scene directive loader";
    report.manifest.importerVersion = "scene-directive-contract-v1";
    report.manifest.generator = "scene_directive";
    report.manifest.unitPolicy =
        "Scene directive units are renderer scene units; referenced mesh transforms are preserved";
    report.manifest.maturityLabel = "production_diagnostics";
    report.manifest.sceneUnitMeters = 1.0f;

    report.stats.materialCount = resources.materialCount();
    report.stats.meshCount = static_cast<uint32_t>(resources.meshes().size());
    report.stats.meshInstanceCount = static_cast<uint32_t>(resources.meshes().size());
    report.stats.primitiveCount = resources.primitiveCount();
    report.stats.triangleCount = resources.triangleCount();
    report.stats.loadedTextureCount = resources.materialTextureCount();
    for (const auto& mesh : resources.meshes()) {
        report.stats.vertexCount += mesh.vertices.size();
        report.stats.indexCount += mesh.indices.size();
    }
    auto addSupportMatrix = [&](std::string area, std::string status, std::string details) {
        PathTracer::AssetSupportMatrixEntry entry{};
        entry.area = std::move(area);
        entry.status = std::move(status);
        entry.details = std::move(details);
        report.supportMatrix.push_back(std::move(entry));
    };
    addSupportMatrix("geometry",
                     "supported",
                     "scene directives support primitives plus referenced OBJ, PLY, glTF, and GLB meshes");
    addSupportMatrix("transforms",
                     "supported",
                     "mesh directive transforms are preserved in renderer-native scene records");
    addSupportMatrix("cameras",
                     "supported",
                     "scene directive camera records drive the shared renderer camera settings");
    addSupportMatrix("textures",
                     "supported",
                     "referenced material textures use the shared SceneResources texture loader and budget policy");
    addSupportMatrix("materials",
                     "partial",
                     "scene directives map renderer-native material records; full external DCC material graphs are out of scope");
    addSupportMatrix("lights",
                     "partial",
                     "renderer-native emissive/directional/primitive lights are supported; external punctual-light imports require glTF diagnostics or conversion");
    addSupportMatrix("animation",
                     "unsupported",
                     "scene directives are static and do not evaluate animation, skinning, or morph targets");
    addSupportMatrix("ktx2_mip_compression",
                     "supported",
                     "KTX2/DDS/raster texture policy is handled by the shared texture loader");
    addSupportMatrix("assimp_formats",
                     "external_tool",
                     "Assimp-supported static formats should be converted through PathTracerImport before scene directive use");
    addSupportMatrix("usd_materialx",
                     "out_of_scope",
                     "USD and MaterialX are not runtime import targets in the current pipeline");
    const auto& texturePolicy = resources.textureLoadPolicy();
    report.texturePolicy.residencyPolicy =
        "material textures are registered once by stable source key and promoted to private Metal textures when a device is available";
    report.texturePolicy.mipPolicy =
        "runtime image textures generate mip chains; KTX2/DDS source mip chains are preserved when present";
    report.texturePolicy.compressionPolicy =
        "KTX2 and supported DDS block-compressed textures stay compressed; common raster inputs decode to RGBA texture storage";
    auto budgetPolicyLabel = [](PathTracer::TextureBudgetPolicy policy) {
        switch (policy) {
            case PathTracer::TextureBudgetPolicy::Strict: return "strict";
            case PathTracer::TextureBudgetPolicy::Warn: return "warn";
            case PathTracer::TextureBudgetPolicy::Ignore: return "ignore";
        }
        return "strict";
    };
    auto pilotModeLabel = [](PathTracer::PilotMode mode) {
        switch (mode) {
            case PathTracer::PilotMode::FullClamped: return "full_clamped";
            case PathTracer::PilotMode::GeoOnly: return "geo_only";
            case PathTracer::PilotMode::AlbedoOnly: return "albedo_only";
        }
        return "full_clamped";
    };
    report.texturePolicy.budgetPolicy =
        std::string(budgetPolicyLabel(texturePolicy.policy)) + "; pilot_mode=" +
        pilotModeLabel(texturePolicy.pilotMode);
    report.texturePolicy.determinismPolicy =
        "texture identity is derived from manifest output hashes, file paths, embedded data hashes, or declared scene references";
    report.texturePolicy.maxDimension = texturePolicy.maxDimension;
    report.texturePolicy.maxTextureBytes = texturePolicy.maxTextureBytes;

    auto addDiagnostic = [&](std::string severity,
                             std::string category,
                             std::string message,
                             const fs::path& assetPath,
                             std::string objectType,
                             std::string objectName = {},
                             int32_t objectIndex = -1) {
        PathTracer::AssetPipelineDiagnostic diagnostic{};
        diagnostic.severity = std::move(severity);
        diagnostic.category = std::move(category);
        diagnostic.message = std::move(message);
        diagnostic.assetPath = assetPath.string();
        diagnostic.objectType = std::move(objectType);
        diagnostic.objectName = std::move(objectName);
        diagnostic.objectIndex = objectIndex;
        if (diagnostic.severity == "error") {
            ++report.stats.errorCount;
        } else if (diagnostic.severity == "warning") {
            ++report.stats.warningCount;
        }
        report.diagnostics.push_back(std::move(diagnostic));
    };

    addDiagnostic("info",
                  "scene_directive_resolution",
                  "Resolved scene directive manifest for non-glTF asset pipeline diagnostics",
                  resolvedSceneFilePath,
                  "scene");
    if (diagnostics.resolvedModelPaths.empty()) {
        addDiagnostic("warning",
                      "scene_directive_resolution",
                      "Scene directive manifest contains no referenced mesh assets",
                      resolvedSceneFilePath,
                      "scene");
    }

    auto recordResolvedAssets = [&](const std::vector<fs::path>& paths, const char* objectType) {
        for (size_t i = 0; i < paths.size(); ++i) {
            std::error_code existsEc;
            const bool exists = fs::exists(paths[i], existsEc);
            addDiagnostic(exists ? "info" : "error",
                          exists ? "asset_resolved" : "asset_missing",
                          exists ? "Resolved referenced asset"
                                 : "Referenced asset path does not exist at diagnostics time",
                          paths[i],
                          objectType,
                          paths[i].filename().string(),
                          static_cast<int32_t>(i));
        }
    };
    recordResolvedAssets(diagnostics.resolvedModelPaths, "mesh");
    recordResolvedAssets(diagnostics.resolvedEnvironmentPaths, "environment");

    resources.setAssetPipelineReport(std::move(report));
}

bool MatrixNearlyEqual(const simd::float4x4& a, const simd::float4x4& b, float epsilon = 1.0e-5f) {
    for (int column = 0; column < 4; ++column) {
        const simd::float4 delta = simd_abs(a.columns[column] - b.columns[column]);
        if (delta.x > epsilon || delta.y > epsilon || delta.z > epsilon || delta.w > epsilon) {
            return false;
        }
    }
    return true;
}

float BoundsDistance(const RuntimeBounds& bounds, const simd::float3& point) {
    if (!bounds.valid) {
        return std::numeric_limits<float>::max();
    }
    return simd_length(bounds.center - point);
}

std::vector<const RuntimeMeshSnapshot*> CollectMatchingSnapshots(
    const std::vector<RuntimeMeshSnapshot>& snapshots,
    const std::vector<std::string>& containsFilters,
    const std::optional<simd::float3>& nearPoint = std::nullopt,
    size_t limit = std::numeric_limits<size_t>::max()) {
    std::vector<const RuntimeMeshSnapshot*> matches;
    for (const auto& snapshot : snapshots) {
        if (!snapshot.mesh || !snapshot.renderWorldBounds.valid) {
            continue;
        }
        const std::string lower = ToLowerAscii(snapshot.mesh->name);
        bool matched = false;
        for (const std::string& contains : containsFilters) {
            if (lower.find(contains) != std::string::npos) {
                matched = true;
                break;
            }
        }
        if (matched) {
            matches.push_back(&snapshot);
        }
    }

    if (nearPoint) {
        std::sort(matches.begin(), matches.end(), [&](const RuntimeMeshSnapshot* lhs, const RuntimeMeshSnapshot* rhs) {
            return BoundsDistance(lhs->renderWorldBounds, *nearPoint) < BoundsDistance(rhs->renderWorldBounds, *nearPoint);
        });
    } else {
        std::sort(matches.begin(), matches.end(), [](const RuntimeMeshSnapshot* lhs, const RuntimeMeshSnapshot* rhs) {
            return lhs->mesh->name < rhs->mesh->name;
        });
    }

    if (matches.size() > limit) {
        matches.resize(limit);
    }
    return matches;
}

std::optional<RuntimeLandmark> SelectNearestLandmark(const std::vector<const RuntimeMeshSnapshot*>& snapshots,
                                                     const std::string& label) {
    if (snapshots.empty()) {
        return std::nullopt;
    }
    return RuntimeLandmark{label, snapshots.front()->renderWorldBounds};
}

RuntimeLandmark UnionSnapshotBounds(const std::string& name,
                                    const std::vector<const RuntimeMeshSnapshot*>& snapshots) {
    RuntimeLandmark out;
    out.name = name;
    if (snapshots.empty()) {
        return out;
    }
    simd::float3 minP = snapshots.front()->renderWorldBounds.min;
    simd::float3 maxP = snapshots.front()->renderWorldBounds.max;
    for (const RuntimeMeshSnapshot* snapshot : snapshots) {
        minP = simd_min(minP, snapshot->renderWorldBounds.min);
        maxP = simd_max(maxP, snapshot->renderWorldBounds.max);
    }
    out.bounds.min = minP;
    out.bounds.max = maxP;
    out.bounds.center = 0.5f * (minP + maxP);
    out.bounds.valid = true;
    return out;
}

std::string FormatFloat3(const simd::float3& v) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return out.str();
}

std::string FormatMatrix4x4(const simd::float4x4& m) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "[[" << m.columns[0].x << ", " << m.columns[1].x << ", " << m.columns[2].x << ", " << m.columns[3].x << "], "
        << "[" << m.columns[0].y << ", " << m.columns[1].y << ", " << m.columns[2].y << ", " << m.columns[3].y << "], "
        << "[" << m.columns[0].z << ", " << m.columns[1].z << ", " << m.columns[2].z << ", " << m.columns[3].z << "], "
        << "[" << m.columns[0].w << ", " << m.columns[1].w << ", " << m.columns[2].w << ", " << m.columns[3].w << "]]";
    return out.str();
}

std::string JsonEscapeInline(const std::string& input) {
    std::string escaped;
    escaped.reserve(input.size());
    for (char c : input) {
        switch (c) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped.push_back(c); break;
        }
    }
    return escaped;
}

struct ScreenProjection {
    bool inside = false;
    float x = 0.0f;
    float y = 0.0f;
};

struct ResolvedCameraPose {
    simd::float3 position = simd_make_float3(0.0f, 0.0f, 0.0f);
    simd::float3 forward = simd_make_float3(0.0f, 0.0f, -1.0f);
    simd::float3 up = simd_make_float3(0.0f, 1.0f, 0.0f);
    float verticalFovDegrees = 45.0f;
};

ScreenProjection ProjectLandmark(const simd::float3& point,
                                 const simd::float3& position,
                                 const simd::float3& forward,
                                 const simd::float3& up,
                                 float verticalFovDegrees,
                                 float aspect) {
    ScreenProjection out;
    simd::float3 f = simd_normalize(forward);
    simd::float3 w = -f;
    simd::float3 u = simd_normalize(simd_cross(up, w));
    simd::float3 v = simd_normalize(simd_cross(w, u));
    simd::float3 rel = point - position;
    const float z = simd_dot(rel, f);
    if (z <= 1.0e-6f) {
        return out;
    }
    const float tanHalf = std::tan(verticalFovDegrees * static_cast<float>(M_PI / 180.0) * 0.5f);
    const float x = simd_dot(rel, u);
    const float y = simd_dot(rel, v);
    out.x = 0.5f * (x / (z * tanHalf * aspect) + 1.0f);
    out.y = 0.5f * (1.0f - y / (z * tanHalf));
    out.inside = out.x >= 0.0f && out.x <= 1.0f && out.y >= 0.0f && out.y <= 1.0f;
    return out;
}

ResolvedCameraPose ResolveCameraPose(const PathTracer::RenderSettings& settings) {
    ResolvedCameraPose pose;
    pose.verticalFovDegrees = settings.cameraVerticalFov;
    if (settings.explicitCameraEnabled) {
        pose.position = settings.explicitCameraPosition;
        pose.forward = simd_normalize(settings.explicitCameraForward);
        pose.up = simd_normalize(settings.explicitCameraUp);
        return pose;
    }

    const float distance = std::max(settings.cameraDistance, 0.1f);
    const float yaw = settings.cameraYaw;
    const float pitch = settings.cameraPitch;
    const float cosPitch = std::cos(pitch);
    const float sinPitch = std::sin(pitch);
    const float cosYaw = std::cos(yaw);
    const float sinYaw = std::sin(yaw);
    const simd::float3 offset = {
        distance * cosPitch * cosYaw,
        distance * sinPitch,
        distance * cosPitch * sinYaw
    };
    pose.position = settings.cameraTarget + offset;
    pose.forward = simd_normalize(settings.cameraTarget - pose.position);
    pose.up = simd_make_float3(0.0f, 1.0f, 0.0f);
    return pose;
}

bool WriteTextFileAtomic(const fs::path& path, const std::string& text, std::string& error);

void WriteCameraSearchReportJson(const fs::path& jsonPath,
                                 const ResolvedCameraPose& camera,
                                 const RuntimeLandmark* scooter,
                                 const RuntimeLandmark& facade,
                                 const RuntimeLandmark* leftLamp,
                                 const RuntimeLandmark* rightLamp,
                                 const RuntimeLandmark& curb,
                                 const RuntimeLandmark* streetOpening,
                                 const std::vector<const RuntimeMeshSnapshot*>& lampCandidates,
                                 const std::vector<const RuntimeMeshSnapshot*>& facadeCandidates,
                                 float aspect,
                                 std::string& error) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(9);
    out << "{\n";
    out << "  \"camera\": {\n";
    out << "    \"position\": [" << camera.position.x << ", " << camera.position.y << ", " << camera.position.z << "],\n";
    out << "    \"forward\": [" << camera.forward.x << ", " << camera.forward.y << ", " << camera.forward.z << "],\n";
    out << "    \"up\": [" << camera.up.x << ", " << camera.up.y << ", " << camera.up.z << "],\n";
    out << "    \"vertical_fov_deg\": " << camera.verticalFovDegrees << ",\n";
    out << "    \"aspect\": " << aspect << "\n";
    out << "  },\n";
    out << "  \"landmarks\": {\n";
    auto writeLandmark = [&](const char* key, const RuntimeLandmark* lm, bool last) {
        out << "    \"" << key << "\": ";
        if (!lm || !lm->bounds.valid) {
            out << "null";
        } else {
            const ScreenProjection proj = ProjectLandmark(
                lm->bounds.center, camera.position, camera.forward, camera.up, camera.verticalFovDegrees, aspect);
            out << "{\n";
            out << "      \"name\": \"" << JsonEscapeInline(lm->name) << "\",\n";
            out << "      \"center\": [" << lm->bounds.center.x << ", " << lm->bounds.center.y << ", " << lm->bounds.center.z << "],\n";
            out << "      \"bounds_min\": [" << lm->bounds.min.x << ", " << lm->bounds.min.y << ", " << lm->bounds.min.z << "],\n";
            out << "      \"bounds_max\": [" << lm->bounds.max.x << ", " << lm->bounds.max.y << ", " << lm->bounds.max.z << "],\n";
            out << "      \"projection\": {\n";
            out << "        \"x\": " << proj.x << ",\n";
            out << "        \"y\": " << proj.y << ",\n";
            out << "        \"inside\": " << (proj.inside ? "true" : "false") << "\n";
            out << "      }\n";
            out << "    }";
        }
        out << (last ? "\n" : ",\n");
    };
    writeLandmark("scooter", scooter, false);
    writeLandmark("facade", facade.bounds.valid ? &facade : nullptr, false);
    writeLandmark("left_lamp", leftLamp, false);
    writeLandmark("right_lamp", rightLamp, false);
    writeLandmark("curb", curb.bounds.valid ? &curb : nullptr, false);
    writeLandmark("street_opening", streetOpening, true);
    out << "  },\n";
    out << "  \"candidates\": {\n";
    auto writeSnapshotList = [&](const char* key,
                                 const std::vector<const RuntimeMeshSnapshot*>& snapshots,
                                 bool last) {
        out << "    \"" << key << "\": [\n";
        for (size_t i = 0; i < snapshots.size(); ++i) {
            const RuntimeMeshSnapshot* snapshot = snapshots[i];
            const ScreenProjection proj = ProjectLandmark(
                snapshot->renderWorldBounds.center,
                camera.position,
                camera.forward,
                camera.up,
                camera.verticalFovDegrees,
                aspect);
            out << "      {\n";
            out << "        \"name\": \"" << JsonEscapeInline(snapshot->mesh->name) << "\",\n";
            out << "        \"center\": [" << snapshot->renderWorldBounds.center.x << ", "
                << snapshot->renderWorldBounds.center.y << ", " << snapshot->renderWorldBounds.center.z << "],\n";
            out << "        \"bounds_min\": [" << snapshot->renderWorldBounds.min.x << ", "
                << snapshot->renderWorldBounds.min.y << ", " << snapshot->renderWorldBounds.min.z << "],\n";
            out << "        \"bounds_max\": [" << snapshot->renderWorldBounds.max.x << ", "
                << snapshot->renderWorldBounds.max.y << ", " << snapshot->renderWorldBounds.max.z << "],\n";
            out << "        \"projection\": {\n";
            out << "          \"x\": " << proj.x << ",\n";
            out << "          \"y\": " << proj.y << ",\n";
            out << "          \"inside\": " << (proj.inside ? "true" : "false") << "\n";
            out << "        }\n";
            out << "      }" << (i + 1u < snapshots.size() ? "," : "") << "\n";
        }
        out << "    ]" << (last ? "\n" : ",\n");
    };
    writeSnapshotList("lamp_posts", lampCandidates, false);
    writeSnapshotList("facade_parts", facadeCandidates, true);
    out << "  }\n";
    out << "}\n";
    if (!out.good()) {
        error = "Failed while composing camera search report: " + jsonPath.string();
        return;
    }
    (void)WriteTextFileAtomic(jsonPath, out.str(), error);
}

void ApplyCliOverrides(const CliOptions& options, PathTracer::RenderSettings& settings) {
    if (options.widthSet) {
        settings.renderWidth = options.width;
    }
    if (options.heightSet) {
        settings.renderHeight = options.height;
    }
    if (options.maxDepthSet) {
        settings.maxDepth = options.maxDepth;
    }
    if (options.seedSet) {
        settings.fixedRngSeed = options.seed;
    }
    if (options.tonemapSet) {
        settings.tonemapMode = options.tonemapMode;
    }
    if (options.exposureSet) {
        settings.exposure = options.exposure;
    }
    if (options.envRotationSet) {
        settings.environmentRotation = options.envRotationDegrees * static_cast<float>(kPi / 180.0);
    }
    if (options.envIntensitySet) {
        settings.environmentIntensity = std::max(options.envIntensity, 0.0f);
    }
    if (options.envBackgroundScaleSet) {
        settings.environmentBackgroundScale = options.envBackgroundScale;
    }
    if (options.enableSoftwareRayTracingSet) {
        settings.enableSoftwareRayTracing = options.enableSoftwareRayTracing;
    }
    if (options.forceSoftwareRayTracingSet && options.forceSoftwareRayTracing) {
        settings.enableSoftwareRayTracing = true;
    }
    if (options.forceHardwareRayTracingSet && options.forceHardwareRayTracing) {
        settings.enableSoftwareRayTracing = false;
    }
    if (options.enableMneeSet) {
        settings.enableMnee = options.enableMnee;
    }
    if (options.enableSpecularNeeSet) {
        settings.enableSpecularNee = options.enableSpecularNee;
    }
    if (options.executionModeSet) {
        settings.executionMode = options.executionMode;
    }
    if (options.wavefrontSchedulingPolicySet) {
        settings.wavefrontSchedulingPolicy = options.wavefrontSchedulingPolicy;
    }
    if (options.wavefrontCompactionEnabledSet) {
        settings.wavefrontCompactionEnabled = options.wavefrontCompactionEnabled;
    }
    if (options.wavefrontShadeIndirectEnabledSet) {
        settings.wavefrontShadeIndirectEnabled = options.wavefrontShadeIndirectEnabled;
    }
    if (options.wavefrontShadeIndirectProbeEnabledSet) {
        settings.wavefrontShadeIndirectProbeEnabled = options.wavefrontShadeIndirectProbeEnabled;
    }
    if (options.wavefrontShadeSurfaceProbeEnabledSet) {
        settings.wavefrontShadeSurfaceProbeEnabled = options.wavefrontShadeSurfaceProbeEnabled;
    }
    if (options.wavefrontShadeMetadataProbeEnabledSet) {
        settings.wavefrontShadeMetadataProbeEnabled = options.wavefrontShadeMetadataProbeEnabled;
    }
    if (options.wavefrontShadeScalarProbeEnabledSet) {
        settings.wavefrontShadeScalarProbeEnabled = options.wavefrontShadeScalarProbeEnabled;
    }
    if (options.wavefrontShadeClosureProbeEnabledSet) {
        settings.wavefrontShadeClosureProbeEnabled = options.wavefrontShadeClosureProbeEnabled;
    }
    if (options.wavefrontShadeBsdfProbeEnabledSet) {
        settings.wavefrontShadeBsdfProbeEnabled = options.wavefrontShadeBsdfProbeEnabled;
    }
    if (options.wavefrontShadowIndirectEnabledSet) {
        settings.wavefrontShadowIndirectEnabled = options.wavefrontShadowIndirectEnabled;
    }
    if (options.directLightModeSet) {
        settings.directLightMode = options.directLightMode;
    }
    if (options.restirGiModeSet) {
        settings.restirGiMode = options.restirGiMode;
    }
    if (options.pathGuidingModeSet) {
        settings.pathGuidingMode = options.pathGuidingMode;
    }
    if (options.pathGuidingStrengthSet) {
        settings.pathGuidingStrength = std::clamp(options.pathGuidingStrength, 0.0f, 1.0f);
    }
    if (options.pathGuidingCellSizeSet) {
        settings.pathGuidingCellSize = std::max(options.pathGuidingCellSize, 1.0e-4f);
    }
    if (options.pathGuidingTrainingIntervalSet) {
        settings.pathGuidingTrainingInterval = std::max<uint32_t>(options.pathGuidingTrainingInterval, 1u);
    }
    if (options.restirPtModeSet) {
        settings.restirPtMode = options.restirPtMode;
    }
    if (options.restirPtMaxReservoirsSet) {
        settings.restirPtMaxReservoirs = std::max(options.restirPtMaxReservoirs, 1u);
    }
    if (options.restirPtDebugSet) {
        settings.restirPtDebug = options.restirPtDebug;
    }
    if (options.restirPtReuseStrengthSet) {
        settings.restirPtReuseStrength = std::clamp(options.restirPtReuseStrength, 0.0f, 1.0f);
    }
    if (options.radianceCacheModeSet) {
        settings.radianceCacheMode = options.radianceCacheMode;
    }
    if (options.radianceCacheCellSizeSet) {
        settings.radianceCacheCellSize = std::max(options.radianceCacheCellSize, 1.0e-4f);
    }
    if (options.radianceCacheMinDepthSet) {
        settings.radianceCacheMinDepth = std::max<uint32_t>(options.radianceCacheMinDepth, 1u);
    }
    if (options.radianceCacheMinConfidenceSet) {
        settings.radianceCacheMinConfidence =
            std::clamp(options.radianceCacheMinConfidence, 0.0f, 1.0f);
    }
    if (options.radianceCacheTrainingIntervalSet) {
        settings.radianceCacheTrainingInterval =
            std::max<uint32_t>(options.radianceCacheTrainingInterval, 1u);
    }
    if (options.radianceCacheMaxRadianceSet) {
        settings.radianceCacheMaxRadiance = std::max(options.radianceCacheMaxRadiance, 1.0e-4f);
    }
    if (options.radianceCacheDebugSet) {
        settings.radianceCacheDebug = options.radianceCacheDebug;
    }
    if (options.causticTransportModeSet) {
        settings.causticTransportMode = options.causticTransportMode;
    }
    if (options.causticMaxVerticesSet) {
        settings.causticMaxVertices = std::max<uint32_t>(options.causticMaxVertices, 1u);
    }
    if (options.volumeTransportModeSet) {
        settings.volumeTransportMode = options.volumeTransportMode;
    }
    if (options.volumePhaseFunctionSet) {
        settings.volumePhaseFunction = options.volumePhaseFunction;
    }
    if (options.volumeSigmaASet) {
        settings.volumeSigmaA = simd_make_float3(std::max(options.volumeSigmaA.x, 0.0f),
                                                 std::max(options.volumeSigmaA.y, 0.0f),
                                                 std::max(options.volumeSigmaA.z, 0.0f));
    }
    if (options.volumeSigmaSSet) {
        settings.volumeSigmaS = simd_make_float3(std::max(options.volumeSigmaS.x, 0.0f),
                                                 std::max(options.volumeSigmaS.y, 0.0f),
                                                 std::max(options.volumeSigmaS.z, 0.0f));
    }
    if (options.volumeEmissionSet) {
        settings.volumeEmission = simd_make_float3(std::max(options.volumeEmission.x, 0.0f),
                                                   std::max(options.volumeEmission.y, 0.0f),
                                                   std::max(options.volumeEmission.z, 0.0f));
    }
    if (options.volumeAnisotropySet) {
        settings.volumeAnisotropy = std::clamp(options.volumeAnisotropy, -0.98f, 0.98f);
    }
    if (options.volumeMaxScatteringEventsSet) {
        settings.volumeMaxScatteringEvents =
            std::max<uint32_t>(options.volumeMaxScatteringEvents, 1u);
    }
    if (options.volumeNeeEnabledSet) {
        settings.volumeNeeEnabled = options.volumeNeeEnabled;
    }
    if (options.volumeDebugSet) {
        settings.volumeDebug = options.volumeDebug;
    }
    if (options.spectralRenderingModeSet) {
        settings.spectralRenderingMode = options.spectralRenderingMode;
    }
    if (options.spectralMinWavelengthSet) {
        settings.spectralMinWavelengthNm = options.spectralMinWavelengthNm;
    }
    if (options.spectralMaxWavelengthSet) {
        settings.spectralMaxWavelengthNm = options.spectralMaxWavelengthNm;
    }
    if (settings.spectralMaxWavelengthNm <= settings.spectralMinWavelengthNm) {
        settings.spectralMaxWavelengthNm = settings.spectralMinWavelengthNm + 1.0f;
    }
    settings.spectralMinWavelengthNm = std::clamp(settings.spectralMinWavelengthNm, 300.0f, 829.0f);
    settings.spectralMaxWavelengthNm =
        std::clamp(settings.spectralMaxWavelengthNm,
                   settings.spectralMinWavelengthNm + 1.0f,
                   830.0f);
    if (options.spectralDispersionStrengthSet) {
        settings.spectralDispersionStrength =
            std::max(options.spectralDispersionStrength, 0.0f);
    }
    if (options.spectralDebugSet) {
        settings.spectralDebug = options.spectralDebug;
    }
    if (options.spectralWavelengthsPerPathSet) {
        settings.spectralWavelengthsPerPath = options.spectralWavelengthsPerPath;
    }
    if (options.mlBackendModeSet) {
        settings.mlBackendMode = options.mlBackendMode;
    }
    if (options.mlModelPackagePathSet) {
        settings.mlModelPackagePath = options.mlModelPackagePath;
    }
    if (options.mlDebugSet) {
        settings.mlDebug = options.mlDebug;
    }
    if (options.restirDebugSet) {
        settings.restirDebugEnabled = options.restirDebug;
    }
    if (options.restirDebugViewSet) {
        settings.restirDebugView = options.restirDebugView;
        if (settings.restirDebugView != PathTracer::RenderSettings::RestirDebugView::Beauty) {
            settings.restirDebugEnabled = true;
        }
    }
    if (options.restirDebugCountersSet) {
        settings.restirDebugCounters = options.restirDebugCounters;
    }
    if (options.restirDebugMetricsJsonProvided) {
        settings.restirDebugEnabled = true;
        settings.restirDebugCounters = true;
    }
    if (options.risCandidateCountSet) {
        settings.risCandidateCount = std::max(options.risCandidateCount, 1u);
    }
    if (options.spatialReuseNeighborCountSet) {
        settings.spatialReuseNeighborCount = std::max(options.spatialReuseNeighborCount, 1u);
    }
    if (options.worldReuseCellSizeSet) {
        settings.worldReuseCellSize = std::max(options.worldReuseCellSize, 1.0e-4f);
    }
    if (options.debugShowBaseColorSet) {
        settings.debugShowBaseColor = options.debugShowBaseColor;
        if (options.debugShowBaseColor) {
            settings.debugShowMetallic = false;
            settings.debugShowRoughness = false;
            settings.debugShowAO = false;
        }
    }
    if (options.debugShowMetallicSet) {
        settings.debugShowMetallic = options.debugShowMetallic;
        if (options.debugShowMetallic) {
            settings.debugShowBaseColor = false;
            settings.debugShowRoughness = false;
            settings.debugShowAO = false;
        }
    }
    if (options.debugShowRoughnessSet) {
        settings.debugShowRoughness = options.debugShowRoughness;
        if (options.debugShowRoughness) {
            settings.debugShowBaseColor = false;
            settings.debugShowMetallic = false;
            settings.debugShowAO = false;
        }
    }
    if (options.debugShowAOSet) {
        settings.debugShowAO = options.debugShowAO;
        if (options.debugShowAO) {
            settings.debugShowBaseColor = false;
            settings.debugShowMetallic = false;
            settings.debugShowRoughness = false;
        }
    }
    if (options.debugDisableOrmTextureSet) {
        settings.debugDisableOrmTexture = options.debugDisableOrmTexture;
    }
    if (options.fireflyClampModeSet) {
        if (options.fireflyClampMode == CliOptions::FireflyClampMode::Off) {
            settings.fireflyClampEnabled = false;
        } else {
            settings.fireflyClampEnabled = true;
            settings.fireflyClampMode =
                (options.fireflyClampMode == CliOptions::FireflyClampMode::MaxComponent)
                    ? PathTracer::RenderSettings::FireflyClampMode::MaxComponent
                    : PathTracer::RenderSettings::FireflyClampMode::Luminance;
            settings.fireflyClampFactor = 0.0f;
            settings.fireflyClampFloor = 0.0f;
            settings.fireflyClampMaxContribution = std::max(options.fireflyClampValue, 0.0f);
        }
    }
    if (options.denoiseEnabledSet) {
        settings.denoiseEnabled = options.denoiseEnabled;
    }
    if (options.denoiseUseAlbedoSet) {
        settings.denoiseUseAlbedo = options.denoiseUseAlbedo;
    }
    if (options.denoiseUseNormalSet) {
        settings.denoiseUseNormal = options.denoiseUseNormal;
    }
    if (options.denoiseHdrSet) {
        settings.denoiseHdr = options.denoiseHdr;
    }
    if (options.svgfDenoiseEnabledSet) {
        settings.svgfDenoiseEnabled = options.svgfDenoiseEnabled;
    }
    if (options.svgfDenoiseDirectSet) {
        settings.svgfDenoiseDirect = options.svgfDenoiseDirect;
    }
    if (options.svgfDenoiseIndirectSet) {
        settings.svgfDenoiseIndirect = options.svgfDenoiseIndirect;
    }
    if (options.svgfAtrousIterationsSet) {
        settings.svgfAtrousIterations = std::clamp(options.svgfAtrousIterations, 1u, 5u);
    }
    if (options.svgfNormalSigmaSet) {
        settings.svgfNormalSigma = std::max(options.svgfNormalSigma, 1.0e-4f);
    }
    if (options.svgfAlbedoSigmaSet) {
        settings.svgfAlbedoSigma = std::max(options.svgfAlbedoSigma, 1.0e-4f);
    }
    if (options.svgfLuminanceSigmaSet) {
        settings.svgfLuminanceSigma = std::max(options.svgfLuminanceSigma, 1.0e-4f);
    }
#if PT_DEBUG_TOOLS
    if (options.debugPathSet) {
        settings.enablePathDebug = options.debugPath;
    }
    if (options.debugPixelSet) {
        settings.debugPixelX = options.debugPixelX;
        settings.debugPixelY = options.debugPixelY;
    }
    if (options.debugMaxEntriesSet) {
        settings.debugMaxEntries = options.debugMaxEntries;
    }
    if (options.parityAssertSet) {
        settings.parityAssertEnabled = options.parityAssert;
    }
    if (options.parityPixelSet) {
        settings.parityPixelX = options.parityPixelX;
        settings.parityPixelY = options.parityPixelY;
    }
    if (options.parityOncePerFrameSet) {
        settings.parityAssertOncePerFrame = options.parityOncePerFrame;
    }
    if (options.parityModeSet) {
        settings.parityAssertMode = options.parityMode;
    }
    if (options.parityTargetDepthSet) {
        settings.parityTargetDepth = options.parityTargetDepth;
    }
    if (options.forcePureHWRTForGlassSet) {
        settings.forcePureHWRTForGlass = options.forcePureHWRTForGlass;
    }
    if (options.debugDisableNormalMapSet) {
        settings.debugDisableNormalMap = options.debugDisableNormalMap;
    }
    if (options.debugSpecularOnlySet) {
        settings.debugSpecularOnly = options.debugSpecularOnly;
    }
    if (options.materialMrDebugJsonProvided) {
        settings.enablePathDebug = true;
        settings.debugMaxEntries = std::max<uint32_t>(settings.debugMaxEntries, 256u);
    }
    if (options.directLightAuditJsonProvided) {
        settings.debugDirectLightAudit = true;
        settings.debugMaxEntries = std::max<uint32_t>(settings.debugMaxEntries, 512u);
    }
#endif
}

void ApplyRenderProfileDefaults(CliOptions& options, PathTracer::RenderSettings& settings) {
    if (!options.renderProfileSet || options.renderProfile == RenderProfile::Custom) {
        return;
    }

    auto setWavefrontProfile = [&](PathTracer::RenderSettings::WavefrontSchedulingPolicy policy) {
        if (!options.executionModeSet) {
            settings.executionMode = PathTracer::RenderSettings::ExecutionMode::Wavefront;
        }
        if (!options.wavefrontSchedulingPolicySet) {
            settings.wavefrontSchedulingPolicy = policy;
        }
        if (!options.wavefrontCompactionEnabledSet) {
            settings.wavefrontCompactionEnabled = true;
        }
    };

    switch (options.renderProfile) {
        case RenderProfile::Preview:
            if (!options.sppTotalSet) {
                options.sppTotal = 64u;
            }
            setWavefrontProfile(PathTracer::RenderSettings::WavefrontSchedulingPolicy::Preview);
            break;
        case RenderProfile::Lookdev:
            if (!options.sppTotalSet) {
                options.sppTotal = 256u;
            }
            setWavefrontProfile(PathTracer::RenderSettings::WavefrontSchedulingPolicy::Final);
            break;
        case RenderProfile::Final:
            if (!options.sppTotalSet) {
                options.sppTotal = 1024u;
            }
            setWavefrontProfile(PathTracer::RenderSettings::WavefrontSchedulingPolicy::Offline);
            break;
        case RenderProfile::Reference:
            if (!options.sppTotalSet) {
                options.sppTotal = 4096u;
            }
            if (!options.executionModeSet) {
                settings.executionMode = PathTracer::RenderSettings::ExecutionMode::Megakernel;
            }
            if (!options.seedSet) {
                settings.fixedRngSeed = 1u;
            }
            break;
        case RenderProfile::Debug:
            if (!options.sppTotalSet) {
                options.sppTotal = 16u;
            }
            if (!options.executionModeSet) {
                settings.executionMode = PathTracer::RenderSettings::ExecutionMode::Megakernel;
            }
            if (!options.restirDebugSet) {
                settings.restirDebugEnabled = true;
            }
            if (!options.restirDebugCountersSet) {
                settings.restirDebugCounters = true;
            }
            if (!options.outputMetadataSet) {
                options.outputMetadata = true;
            }
            options.pbrMetrics = true;
            break;
        case RenderProfile::Custom:
        default:
            break;
    }
}

HeadlessCamera BuildHeadlessCamera(const PathTracer::RenderSettings& settings) {
    HeadlessCamera camera{};
    camera.target = settings.cameraTarget;
    camera.distance = settings.cameraDistance;
    camera.yaw = settings.cameraYaw;
    camera.pitch = settings.cameraPitch;
    camera.verticalFov = settings.cameraVerticalFov;
    camera.defocusAngle = settings.cameraDefocusAngle;
    camera.focusDistance = settings.cameraFocusDistance;
    return camera;
}

bool RunHeadlessDenoise(const PathTracer::RenderSettings& settings,
                        HeadlessRenderOutput& output,
                        std::string& error) {
    if (!settings.denoiseEnabled) {
        return true;
    }

    const size_t expectedValues =
        static_cast<size_t>(output.width) * static_cast<size_t>(output.height) * 3u;
    if (output.linearRGB.size() != expectedValues) {
        error = "Rendered beauty buffer has unexpected size for OIDN postprocess";
        return false;
    }

    PathTracer::OidnDenoiseRequest request{};
    request.color = output.linearRGB.data();
    request.width = output.width;
    request.height = output.height;
    request.hdr = settings.denoiseHdr;
    request.cleanAux = true;
    request.filterType = settings.denoiseFilterType == 1
        ? PathTracer::OidnFilterType::RTLightmap
        : PathTracer::OidnFilterType::RT;

    if (settings.denoiseUseAlbedo) {
        if (output.albedoLinearRGB.size() == expectedValues) {
            request.albedo = output.albedoLinearRGB.data();
        } else {
            std::cout << "[Denoise] Albedo guide requested but unavailable; proceeding without it." << std::endl;
        }
    }
    if (settings.denoiseUseNormal) {
        if (output.normalLinearRGB.size() == expectedValues) {
            request.normal = output.normalLinearRGB.data();
        } else {
            std::cout << "[Denoise] Normal guide requested but unavailable; proceeding without it." << std::endl;
        }
    }

    const CFAbsoluteTime denoiseStart = CFAbsoluteTimeGetCurrent();
    std::vector<float> denoised;
    if (!PathTracer::DenoiseWithOidn(request, denoised, error)) {
        error = "Headless OIDN postprocess failed: " + error;
        return false;
    }
    output.linearRGB = std::move(denoised);

    std::cout << "[Denoise] OIDN completed in "
              << std::fixed << std::setprecision(2)
              << (CFAbsoluteTimeGetCurrent() - denoiseStart)
              << " s"
              << " (albedo=" << (request.albedo ? 1 : 0)
              << ", normal=" << (request.normal ? 1 : 0)
              << ", hdr=" << (request.hdr ? 1 : 0) << ")"
              << std::endl;
    return true;
}

fs::path AppendStemSuffix(const fs::path& path, const std::string& suffix) {
    fs::path result = path;
    const std::string stem = result.stem().string();
    const std::string extension = result.extension().string();
    result.replace_filename(stem + suffix + extension);
    return result;
}

const char* BudgetPolicyLabel(PathTracer::TextureBudgetPolicy policy) {
    switch (policy) {
        case PathTracer::TextureBudgetPolicy::Strict:
            return "strict";
        case PathTracer::TextureBudgetPolicy::Warn:
            return "warn";
        case PathTracer::TextureBudgetPolicy::Ignore:
        default:
            return "ignore";
    }
}

const char* PilotModeLabel(PathTracer::PilotMode mode) {
    switch (mode) {
        case PathTracer::PilotMode::GeoOnly:
            return "geo_only";
        case PathTracer::PilotMode::AlbedoOnly:
            return "albedo_only";
        case PathTracer::PilotMode::FullClamped:
        default:
            return "full_clamped";
    }
}

const char* TextureSemanticLabel(PathTracer::MaterialTextureSemantic semantic) {
    switch (semantic) {
        case PathTracer::MaterialTextureSemantic::BaseColor:
            return "baseColor";
        case PathTracer::MaterialTextureSemantic::Orm:
            return "orm";
        case PathTracer::MaterialTextureSemantic::Normal:
            return "normal";
        case PathTracer::MaterialTextureSemantic::Occlusion:
            return "occlusion";
        case PathTracer::MaterialTextureSemantic::Emissive:
            return "emissive";
        case PathTracer::MaterialTextureSemantic::Transmission:
            return "transmission";
        case PathTracer::MaterialTextureSemantic::Generic:
        default:
            return "generic";
    }
}

std::string FormatMemoryMiB(uint64_t bytes) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2)
           << (static_cast<double>(bytes) / kMiB) << " MiB";
    return stream.str();
}

void ApplyPilotModeOverrides(PathTracer::SceneResources& resources,
                             PathTracer::RenderSettings& settings,
                             PathTracer::PilotMode mode) {
    constexpr uint32_t kInvalidTextureIndex = 0xFFFFFFFFu;
    switch (mode) {
        case PathTracer::PilotMode::GeoOnly:
        case PathTracer::PilotMode::AlbedoOnly:
            settings.debugDisableNormalMap = true;
            settings.debugDisableOrmTexture = true;
            break;
        case PathTracer::PilotMode::FullClamped:
        default:
            break;
    }

    for (uint32_t i = 0; i < resources.materialCount(); ++i) {
        const auto* materials = resources.materialsData();
        if (!materials) {
            break;
        }
        auto updated = materials[i];
        if (mode == PathTracer::PilotMode::GeoOnly) {
            updated.baseColorRoughness = simd_make_float4(0.72f, 0.72f, 0.72f, 1.0f);
            updated.emission = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            updated.textureIndices0 = simd_make_uint4(kInvalidTextureIndex,
                                                      kInvalidTextureIndex,
                                                      kInvalidTextureIndex,
                                                      kInvalidTextureIndex);
            updated.textureIndices1 = simd_make_uint4(kInvalidTextureIndex,
                                                      kInvalidTextureIndex,
                                                      kInvalidTextureIndex,
                                                      kInvalidTextureIndex);
            updated.materialFlags |= PathTracerShaderTypes::kMaterialFlagDisableOrm;
            updated.pbrParams = simd_make_float4(0.0f, 1.0f, 1.0f, 0.0f);
            updated.pbrExtras.z = 0.0f;
        } else if (mode == PathTracer::PilotMode::AlbedoOnly) {
            updated.textureIndices0.y = kInvalidTextureIndex;
            updated.textureIndices0.z = kInvalidTextureIndex;
            updated.textureIndices0.w = kInvalidTextureIndex;
            updated.textureIndices1 = simd_make_uint4(kInvalidTextureIndex,
                                                      kInvalidTextureIndex,
                                                      kInvalidTextureIndex,
                                                      kInvalidTextureIndex);
            updated.materialFlags |= PathTracerShaderTypes::kMaterialFlagDisableOrm;
            updated.emission = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            updated.pbrParams.w = 0.0f;
            updated.pbrExtras.z = 0.0f;
        }
        resources.updateMaterial(i, updated);
    }
}

void PrintLargeSceneReport(const PathTracer::SceneResources& resources,
                           const CliOptions& options,
                           const PathTracer::SceneAccelBuildStats& accelEstimate) {
    auto textures = resources.textureInventory();
    auto meshes = resources.meshInventory();
    std::sort(textures.begin(), textures.end(), [](const auto& a, const auto& b) {
        return a.estimatedBytes > b.estimatedBytes;
    });
    std::sort(meshes.begin(), meshes.end(), [](const auto& a, const auto& b) {
        return a.triangleCount > b.triangleCount;
    });

    const uint64_t geometryBytes = resources.estimatedGeometryMemoryBytes();
    const uint64_t textureBytes = resources.estimatedTextureMemoryBytes();
    const uint64_t textureAllocatedBytes = resources.allocatedTextureMemoryBytes();
    const uint64_t accelBytes =
        accelEstimate.blasMemoryBytes +
        accelEstimate.tlasMemoryBytes +
        accelEstimate.scratchMemoryBytes;
    const uint64_t recommendedBytes = resources.recommendedMaxWorkingSetSizeBytes();
    const bool textureBudgetExceeded = resources.textureBudgetExceeded();
    const bool overDeviceBudget = recommendedBytes > 0u &&
                                  (geometryBytes + textureBytes + accelBytes) > recommendedBytes;

    std::cout << "[LargeScene] mode=" << PilotModeLabel(options.pilotMode)
              << " policy=" << BudgetPolicyLabel(options.budgetPolicy)
              << " meshes=" << meshes.size()
              << " textures=" << textures.size()
              << std::endl;
    std::cout << "[LargeScene] geometry=" << FormatMemoryMiB(geometryBytes)
              << " textures=" << FormatMemoryMiB(textureBytes)
              << " BLAS~=" << FormatMemoryMiB(accelEstimate.blasMemoryBytes)
              << " TLAS~=" << FormatMemoryMiB(accelEstimate.tlasMemoryBytes)
              << " scratch~=" << FormatMemoryMiB(accelEstimate.scratchMemoryBytes)
              << " estimatedWorkingSet=" << FormatMemoryMiB(geometryBytes + textureBytes + accelBytes)
              << std::endl;
    if (textureAllocatedBytes > 0u) {
        std::cout << "[LargeScene] texturesAllocated="
                  << FormatMemoryMiB(textureAllocatedBytes);
        if (textureAllocatedBytes != textureBytes) {
            const uint64_t allocatedWorkingSet =
                geometryBytes +
                textureAllocatedBytes +
                accelEstimate.blasMemoryBytes +
                accelEstimate.tlasMemoryBytes +
                accelEstimate.scratchMemoryBytes;
            std::cout << " allocatedWorkingSet~=" << FormatMemoryMiB(allocatedWorkingSet);
        }
        std::cout << std::endl;
    }
    if (accelEstimate.blasScratchMemoryBytes > 0u || accelEstimate.tlasScratchMemoryBytes > 0u) {
        std::cout << "[LargeScene] scratchBreakdown BLASmax~="
                  << FormatMemoryMiB(accelEstimate.blasScratchMemoryBytes)
                  << " TLAS~=" << FormatMemoryMiB(accelEstimate.tlasScratchMemoryBytes)
                  << " peak~=" << FormatMemoryMiB(accelEstimate.scratchMemoryBytes)
                  << std::endl;
    }
    if (recommendedBytes > 0u) {
        std::cout << "[LargeScene] recommendedMaxWorkingSetSize="
                  << FormatMemoryMiB(recommendedBytes)
                  << " prebuildUsage=" << std::fixed << std::setprecision(2)
                  << (static_cast<double>(geometryBytes + textureBytes + accelBytes) * 100.0 /
                      static_cast<double>(recommendedBytes))
                  << "%"
                  << std::endl;
    } else {
        std::cout << "[LargeScene] recommendedMaxWorkingSetSize=unavailable" << std::endl;
    }
    if (options.textureMaxDimSet || options.maxTextureBytesSet) {
        std::cout << "[LargeScene] textureMaxDim="
                  << (options.textureMaxDimSet ? std::to_string(options.textureMaxDim) : std::string("off"))
                  << " maxTextureBytes="
                  << (options.maxTextureBytesSet ? std::to_string(options.maxTextureBytes) : std::string("off"))
                  << std::endl;
    }
    if (textureBudgetExceeded) {
        std::cout << "[LargeScene] textureBudgetOutcome=" << resources.textureBudgetMessage() << std::endl;
    } else if (overDeviceBudget) {
        std::cout << "[LargeScene] budgetOutcome=prebuild working set already exceeds device recommendation"
                  << std::endl;
    } else {
        std::cout << "[LargeScene] budgetOutcome=prebuild working set fits current policy checks" << std::endl;
    }

    const size_t maxTextures = std::min<size_t>(20u, textures.size());
    for (size_t i = 0; i < maxTextures; ++i) {
        const auto& entry = textures[i];
        std::cout << "[LargeScene][Texture " << (i + 1u) << "] "
                  << entry.label
                  << " sem=" << TextureSemanticLabel(entry.semantic)
                  << " size=" << entry.width << "x" << entry.height
                  << " format=" << entry.formatLabel
                  << " bytes=" << FormatMemoryMiB(entry.estimatedBytes);
        if (entry.allocatedBytes > 0u) {
            std::cout << " alloc=" << FormatMemoryMiB(entry.allocatedBytes);
        }
        if (entry.clampedByDimension) {
            std::cout << " " << entry.decision;
        }
        std::cout << std::endl;
    }

    const size_t maxMeshes = std::min<size_t>(20u, meshes.size());
    for (size_t i = 0; i < maxMeshes; ++i) {
        const auto& entry = meshes[i];
        std::cout << "[LargeScene][Mesh " << (i + 1u) << "] "
                  << entry.name
                  << " tris=" << entry.triangleCount
                  << " bytes=" << FormatMemoryMiB(entry.estimatedBytes)
                  << std::endl;
    }
}

void PrintSceneReport(const PathTracer::SceneMemoryReport& report,
                      const PathTracer::SceneAccelBuildStats& accelStats) {
    std::cout << "[Scene] triangles=" << report.triangleCount
              << " instances=" << report.instanceCount
              << " meshes=" << report.meshCount
              << " textures=" << report.textureCount
              << std::endl;
    std::cout << "[Scene] geometry=" << FormatMemoryMiB(report.geometryMemoryBytes)
              << " textures=" << FormatMemoryMiB(report.textureMemoryBytes)
              << " BLAS=" << FormatMemoryMiB(report.blasMemoryBytes)
              << " TLAS=" << FormatMemoryMiB(report.tlasMemoryBytes)
              << " scratch=" << FormatMemoryMiB(report.scratchMemoryBytes)
              << " total=" << FormatMemoryMiB(report.totalEstimatedMemoryBytes)
              << std::endl;
    if (report.textureAllocatedMemoryBytes > 0u) {
        std::cout << "[Scene] texturesAllocated="
                  << FormatMemoryMiB(report.textureAllocatedMemoryBytes);
        if (report.textureAllocatedMemoryBytes != report.textureMemoryBytes) {
            const uint64_t allocatedTotal =
                report.geometryMemoryBytes +
                report.textureAllocatedMemoryBytes +
                report.blasMemoryBytes +
                report.tlasMemoryBytes +
                report.scratchMemoryBytes;
            std::cout << " allocatedTotal=" << FormatMemoryMiB(allocatedTotal);
        }
        std::cout << std::endl;
    }
    if (report.peakScratchMemoryBytes > 0u || accelStats.blasScratchMemoryBytes > 0u ||
        accelStats.tlasScratchMemoryBytes > 0u) {
        std::cout << "[Scene] scratchBreakdown BLASmax="
                  << FormatMemoryMiB(accelStats.blasScratchMemoryBytes)
                  << " TLAS=" << FormatMemoryMiB(accelStats.tlasScratchMemoryBytes)
                  << " peak=" << FormatMemoryMiB(report.peakScratchMemoryBytes)
                  << " retained=" << FormatMemoryMiB(report.scratchMemoryBytes)
                  << std::endl;
    }
    if (report.recommendedMaxWorkingSetSizeBytes > 0u) {
        std::cout << "[Scene] recommendedMaxWorkingSetSize="
                  << FormatMemoryMiB(report.recommendedMaxWorkingSetSizeBytes)
                  << " usage=" << std::fixed << std::setprecision(2)
                  << report.budgetUsagePercent << "%"
                  << std::endl;
    } else {
        std::cout << "[Scene] recommendedMaxWorkingSetSize=unavailable" << std::endl;
    }

    std::cout << "[Accel] mode="
              << (accelStats.usedHardwareRaytracing ? "hardware" : "software")
              << " policy=" << (accelStats.buildPolicyLabel.empty()
                                     ? "unknown"
                                     : accelStats.buildPolicyLabel)
              << " primitiveCount=" << accelStats.primitiveCount
              << " blasCount=" << accelStats.blasCount
              << " instanceCount=" << accelStats.instanceCount
              << " blasBuildMs=" << std::fixed << std::setprecision(2) << accelStats.blasBuildMs
              << " tlasBuildMs=" << accelStats.tlasBuildMs
              << " totalMemory=" << FormatMemoryMiB(accelStats.blasMemoryBytes + accelStats.tlasMemoryBytes)
              << " peakScratch=" << FormatMemoryMiB(accelStats.scratchMemoryBytes)
              << std::endl;
    if (accelStats.blasAllocatedMemoryBytes > 0u || accelStats.tlasAllocatedMemoryBytes > 0u) {
        std::cout << "[Accel] allocatedMemory="
                  << FormatMemoryMiB(accelStats.blasAllocatedMemoryBytes + accelStats.tlasAllocatedMemoryBytes)
                  << " BLASallocated=" << FormatMemoryMiB(accelStats.blasAllocatedMemoryBytes)
                  << " TLASallocated=" << FormatMemoryMiB(accelStats.tlasAllocatedMemoryBytes)
                  << std::endl;
    }
    if (accelStats.compactionSupported) {
        std::cout << "[Accel] compaction blasRatio=" << std::setprecision(3)
                  << accelStats.blasCompactionRatio
                  << " tlasRatio=" << accelStats.tlasCompactionRatio
                  << std::endl;
    } else {
        std::cout << "[Accel] compaction unavailable" << std::endl;
    }
    std::cout << "[Accel] identityStable=" << (accelStats.identityStable ? "true" : "false")
              << " strategy=" << accelStats.identityStrategy
              << std::endl;
    std::cout << "[Accel] intersectionFunctions supported="
              << (accelStats.customIntersectionFunctionsSupported ? "true" : "false")
              << " enabled="
              << (accelStats.customIntersectionFunctionsEnabled ? "true" : "false")
              << " strategy=" << accelStats.intersectionFunctionStrategy
              << std::endl;
}

void PrintEmbreeCpuSceneReport(const PathTracer::SceneResources& resources) {
    uint64_t meshTriangles = 0u;
    for (const auto& mesh : resources.meshes()) {
        meshTriangles += static_cast<uint64_t>(mesh.indices.size() / 3u);
    }

    const uint64_t rectangleTriangles = static_cast<uint64_t>(resources.rectangleCount()) * 2u;
    const uint64_t totalTriangles = meshTriangles + rectangleTriangles;
    const uint64_t geometryBytes = resources.estimatedGeometryMemoryBytes();
    const uint64_t textureBytes = resources.estimatedTextureMemoryBytes();
    const uint64_t totalBytes = geometryBytes + textureBytes;

    std::cout << "[SceneCPU] triangles=" << totalTriangles
              << " meshes=" << resources.meshes().size()
              << " spheres=" << resources.sphereCount()
              << " rectangles=" << resources.rectangleCount()
              << " materials=" << resources.materialCount()
              << " textures=" << resources.textureInventory().size()
              << std::endl;
    std::cout << "[SceneCPU] geometry=" << FormatMemoryMiB(geometryBytes)
              << " textures=" << FormatMemoryMiB(textureBytes)
              << " total=" << FormatMemoryMiB(totalBytes)
              << std::endl;
    std::cout << "[SceneCPU] acceleration skipped for Embree reference path during scene reporting"
              << std::endl;
}

// Names every subsurface material that will not scatter under the surface with the settings of this render (the GUI
// prints the same lines when it activates a scene).
void PrintSubsurfaceMaterialReport(const PathTracer::SceneResources& resources,
                                   const PathTracer::RenderSettings& settings) {
    for (const std::string& line : PathTracer::SubsurfaceNotScatteringDiagnostics(resources, settings)) {
        std::cout << "[SSS] " << line << std::endl;
    }
}

void PrintEmissiveInventoryReport(const PathTracer::SceneResources& resources) {
    std::cout << "[Scene] Emissive primitives: " << resources.emissivePrimitiveCount() << std::endl;
    std::cout << "[Scene] Total emitted power: " << std::fixed << std::setprecision(4)
              << resources.totalEmittedPower() << std::endl;
    const auto& byMaterial = resources.emittedPowerByMaterial();
    if (!byMaterial.empty()) {
        constexpr size_t kShown = 8u;
        const float total = resources.totalEmittedPower();
        std::cout << "[Scene] Emitted power by material (" << std::min(byMaterial.size(), kShown) << " largest of "
                  << byMaterial.size() << "):";
        for (size_t i = 0; i < byMaterial.size() && i < kShown; ++i) {
            const auto& entry = byMaterial[i];
            std::cout << (i > 0 ? ";" : "") << " " << resources.materialName(entry.materialIndex) << " "
                      << std::setprecision(3) << entry.power << " (" << entry.primitiveCount << " primitives, "
                      << std::setprecision(1) << (total > 0.0f ? 100.0f * entry.power / total : 0.0f) << "%)";
        }
        std::cout << std::setprecision(4) << std::endl;
    }
    if (resources.skippedTexturedEmissivePrimitiveCount() > 0u) {
        std::cout << "[Scene] Emissive primitives skipped (textured emission): "
                  << resources.skippedTexturedEmissivePrimitiveCount() << std::endl;
    }
    if (resources.skippedZeroPowerEmissivePrimitiveCount() > 0u) {
        std::cout << "[WARNING] Emissive primitives skipped due to zero power: "
                  << resources.skippedZeroPowerEmissivePrimitiveCount() << std::endl;
    }
    if (resources.emissivePrimitiveCount() == 0u) {
        std::cout << "[WARNING] emissivePrimitivesBuffer is empty — direct light sampling disabled."
                  << std::endl;
    }
}

struct ImageDifferenceStats {
    double rmse = 0.0;
    double maxAbs = 0.0;
};

struct VarianceStats {
    double meanPixelVariance = 0.0;
};

ImageDifferenceStats ComputeImageDifferenceStats(const std::vector<float>& a,
                                                 const std::vector<float>& b) {
    ImageDifferenceStats stats;
    if (a.size() != b.size() || a.empty()) {
        return stats;
    }
    double sumSq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        sumSq += diff * diff;
        stats.maxAbs = std::max(stats.maxAbs, std::fabs(diff));
    }
    stats.rmse = std::sqrt(sumSq / static_cast<double>(a.size()));
    return stats;
}

double ComputePSNR(const std::vector<float>& reference, const std::vector<float>& test) {
    if (reference.size() != test.size() || reference.empty()) {
        return 0.0;
    }
    double mse = 0.0;
    double maxRef = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double ref = static_cast<double>(reference[i]);
        const double diff = ref - static_cast<double>(test[i]);
        mse += diff * diff;
        maxRef = std::max(maxRef, std::fabs(ref));
    }
    mse /= static_cast<double>(reference.size());
    if (!(mse > 0.0) || !std::isfinite(mse)) {
        return std::numeric_limits<double>::infinity();
    }
    const double peak = std::max(maxRef, 1.0e-6);
    return 10.0 * std::log10((peak * peak) / mse);
}

bool LoadPFM(const fs::path& path,
             std::vector<float>& linearRGB,
             uint32_t& width,
             uint32_t& height,
             std::string& error) {
    linearRGB.clear();
    width = 0u;
    height = 0u;

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        error = "Failed to open PFM: " + path.string();
        return false;
    }

    std::string magic;
    file >> magic;
    if (magic != "PF") {
        error = "Unsupported PFM magic in " + path.string();
        return false;
    }

    file >> width >> height;
    float scale = 0.0f;
    file >> scale;
    file.get();
    if (width == 0u || height == 0u) {
        error = "Invalid PFM dimensions in " + path.string();
        return false;
    }

    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
    linearRGB.resize(pixelCount, 0.0f);
    const size_t rowSize = static_cast<size_t>(width) * 3u;
    for (int y = static_cast<int>(height) - 1; y >= 0; --y) {
        float* row = linearRGB.data() + static_cast<size_t>(y) * rowSize;
        file.read(reinterpret_cast<char*>(row), static_cast<std::streamsize>(rowSize * sizeof(float)));
        if (!file) {
            error = "Failed reading PFM payload from " + path.string();
            linearRGB.clear();
            width = 0u;
            height = 0u;
            return false;
        }
    }
    return true;
}

VarianceStats ComputeMeanPixelVariance(const std::vector<double>& sumLuma,
                                      const std::vector<double>& sumSqLuma,
                                      uint32_t frames) {
    VarianceStats stats;
    if (sumLuma.empty() || sumLuma.size() != sumSqLuma.size() || frames == 0u) {
        return stats;
    }
    double accum = 0.0;
    for (size_t i = 0; i < sumLuma.size(); ++i) {
        const double mean = sumLuma[i] / static_cast<double>(frames);
        const double meanSq = sumSqLuma[i] / static_cast<double>(frames);
        accum += std::max(meanSq - mean * mean, 0.0);
    }
    stats.meanPixelVariance = accum / static_cast<double>(sumLuma.size());
    return stats;
}

bool WriteRisThresholdsJson(const fs::path& jsonPath,
                            const std::string& scene,
                            const fs::path& baselineReference,
                            uint32_t frames,
                            uint32_t risM,
                            double psnrMin,
                            double rmseMax,
                            std::string& error) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(9);
    out << "{\n";
    out << "  \"scene\": \"" << JsonEscapeInline(scene) << "\",\n";
    out << "  \"baseline_reference\": \"" << JsonEscapeInline(baselineReference.string()) << "\",\n";
    out << "  \"frames\": " << frames << ",\n";
    out << "  \"M\": " << risM << ",\n";
    out << "  \"psnr_min\": " << psnrMin << ",\n";
    out << "  \"rmse_max\": " << rmseMax << ",\n";
    out << "  \"note\": \"Current Metal --validate-ris threshold record for the active official R2 emissive-primitive closure scene; see docs/RESTIR_R2_CLOSEOUT.md for validity rationale\"\n";
    out << "}\n";
    if (!out.good()) {
        error = "Failed while composing RIS thresholds JSON: " + jsonPath.string();
        return false;
    }
    return WriteTextFileAtomic(jsonPath, out.str(), error);
}

struct ImagePbrMetrics {
    static constexpr size_t kHistogramBuckets = 32;

    uint64_t pixelCount = 0;
    uint64_t finitePixelCount = 0;
    uint64_t nanInfPixelCount = 0;
    uint64_t nanInfComponentCount = 0;
    double meanLuminance = 0.0;
    double p50Luminance = 0.0;
    double p95Luminance = 0.0;
    double p99Luminance = 0.0;
    double p999Luminance = 0.0;
    double varianceLuminance = 0.0;
    double maxLuminance = 0.0;
    std::vector<double> histogramEdges;
    std::vector<uint64_t> histogramCounts;
};

double LuminanceLinear(float r, float g, float b) {
    const double rr = std::max(static_cast<double>(r), 0.0);
    const double gg = std::max(static_cast<double>(g), 0.0);
    const double bb = std::max(static_cast<double>(b), 0.0);
    return 0.2126 * rr + 0.7152 * gg + 0.0722 * bb;
}

double SortedPercentile(const std::vector<double>& sortedValues, double percentile) {
    if (sortedValues.empty()) {
        return 0.0;
    }
    if (percentile <= 0.0) {
        return sortedValues.front();
    }
    if (percentile >= 100.0) {
        return sortedValues.back();
    }
    const double k = (percentile / 100.0) * static_cast<double>(sortedValues.size() - 1u);
    const size_t lo = static_cast<size_t>(std::floor(k));
    const size_t hi = static_cast<size_t>(std::ceil(k));
    if (lo >= sortedValues.size()) {
        return sortedValues.back();
    }
    if (lo == hi || hi >= sortedValues.size()) {
        return sortedValues[lo];
    }
    const double t = k - static_cast<double>(lo);
    return sortedValues[lo] + (sortedValues[hi] - sortedValues[lo]) * t;
}

ImagePbrMetrics ComputeImagePbrMetrics(const std::vector<float>& linearRGB) {
    ImagePbrMetrics metrics;
    metrics.pixelCount = static_cast<uint64_t>(linearRGB.size() / 3u);
    if (metrics.pixelCount == 0) {
        return metrics;
    }

    std::vector<double> luminance;
    luminance.reserve(static_cast<size_t>(metrics.pixelCount));

    double runningMean = 0.0;
    double runningM2 = 0.0;
    double minPositiveLum = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i + 2 < linearRGB.size(); i += 3u) {
        const float r = linearRGB[i + 0u];
        const float g = linearRGB[i + 1u];
        const float b = linearRGB[i + 2u];
        const bool finite = std::isfinite(r) && std::isfinite(g) && std::isfinite(b);
        if (!finite) {
            ++metrics.nanInfPixelCount;
            metrics.nanInfComponentCount += static_cast<uint64_t>(!std::isfinite(r)) +
                                            static_cast<uint64_t>(!std::isfinite(g)) +
                                            static_cast<uint64_t>(!std::isfinite(b));
            continue;
        }

        ++metrics.finitePixelCount;
        const double lum = LuminanceLinear(r, g, b);
        metrics.maxLuminance = std::max(metrics.maxLuminance, lum);
        if (lum > 0.0) {
            minPositiveLum = std::min(minPositiveLum, lum);
        }
        const double count = static_cast<double>(metrics.finitePixelCount);
        const double delta = lum - runningMean;
        runningMean += delta / count;
        const double delta2 = lum - runningMean;
        runningM2 += delta * delta2;
        luminance.push_back(lum);
    }

    if (metrics.finitePixelCount > 0) {
        metrics.meanLuminance = runningMean;
        metrics.varianceLuminance =
            runningM2 / static_cast<double>(metrics.finitePixelCount);
        std::sort(luminance.begin(), luminance.end());
        metrics.p50Luminance = SortedPercentile(luminance, 50.0);
        metrics.p95Luminance = SortedPercentile(luminance, 95.0);
        metrics.p99Luminance = SortedPercentile(luminance, 99.0);
        metrics.p999Luminance = SortedPercentile(luminance, 99.9);

        metrics.histogramEdges.assign(ImagePbrMetrics::kHistogramBuckets + 1u, 0.0);
        metrics.histogramCounts.assign(ImagePbrMetrics::kHistogramBuckets, 0u);
        if (!std::isfinite(minPositiveLum) || minPositiveLum <= 0.0 || metrics.maxLuminance <= 0.0) {
            metrics.histogramCounts[0] = metrics.finitePixelCount;
        } else {
            const double logMin = std::log10(minPositiveLum);
            const double logMax = std::log10(metrics.maxLuminance);
            const double span = std::max(logMax - logMin, 1.0e-12);
            for (size_t i = 0; i <= ImagePbrMetrics::kHistogramBuckets; ++i) {
                const double t = static_cast<double>(i) /
                                 static_cast<double>(ImagePbrMetrics::kHistogramBuckets);
                metrics.histogramEdges[i] = std::pow(10.0, logMin + span * t);
            }
            for (double lum : luminance) {
                if (!(lum > 0.0) || !std::isfinite(lum)) {
                    ++metrics.histogramCounts[0];
                    continue;
                }
                const double l = std::log10(lum);
                const double norm = std::clamp((l - logMin) / span, 0.0, 1.0);
                size_t bin = static_cast<size_t>(norm * static_cast<double>(ImagePbrMetrics::kHistogramBuckets));
                if (bin >= ImagePbrMetrics::kHistogramBuckets) {
                    bin = ImagePbrMetrics::kHistogramBuckets - 1u;
                }
                ++metrics.histogramCounts[bin];
            }
        }
    }

    return metrics;
}

bool RestirWorldModeActive(PathTracer::RenderSettings::DirectLightMode mode) {
    using DirectLightMode = PathTracer::RenderSettings::DirectLightMode;
    return mode == DirectLightMode::RisWorldReuse ||
           mode == DirectLightMode::RisRegirCache ||
           mode == DirectLightMode::RestirDiRegirHybrid;
}

simd::float3 HashDebugColor(uint32_t value) {
    uint32_t x = value * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (x >> 22u) ^ x;
    return simd_make_float3(static_cast<float>((x >> 0u) & 0xffu) / 255.0f,
                            static_cast<float>((x >> 8u) & 0xffu) / 255.0f,
                            static_cast<float>((x >> 16u) & 0xffu) / 255.0f);
}

// directLightMode: the mode the backend ran (RunDirectLightMode), not the requested one.
void ApplyRestirDebugView(const PathTracer::RenderSettings& settings,
                          PathTracer::RenderSettings::DirectLightMode directLightMode,
                          HeadlessRenderOutput& output) {
    using RestirDebugView = PathTracer::RenderSettings::RestirDebugView;
    const size_t expectedValues =
        static_cast<size_t>(output.width) * static_cast<size_t>(output.height) * 3u;
    if (!settings.restirDebugEnabled ||
        settings.restirDebugView == RestirDebugView::Beauty ||
        output.width == 0u ||
        output.height == 0u ||
        output.linearRGB.size() < expectedValues) {
        return;
    }

    const bool risActive =
        directLightMode != PathTracer::RenderSettings::DirectLightMode::LegacyRect &&
        directLightMode != PathTracer::RenderSettings::DirectLightMode::BaselineEmissive;
    const bool worldActive = RestirWorldModeActive(directLightMode);
    const bool pathGuidingUsed = output.pbrMetrics.pathGuidingUsedCount > 0u;
    const bool restirPtReuseUsed = output.pbrMetrics.restirPtReuseAppliedCount > 0u;
    const bool radianceCacheUsed = output.pbrMetrics.radianceCacheHitCount > 0u;
    const bool causticTransportUsed =
        output.pbrMetrics.causticEyeVertexCount > 0u ||
        output.pbrMetrics.causticConnectionCandidateCount > 0u ||
        output.pbrMetrics.causticDeltaChainCount > 0u;
    const bool svgfActive = settings.svgfDenoiseEnabled;
    const uint32_t modeSeed =
        static_cast<uint32_t>(directLightMode) * 17u +
        static_cast<uint32_t>(settings.restirGiMode) * 31u +
        static_cast<uint32_t>(settings.pathGuidingMode) * 47u +
        static_cast<uint32_t>(settings.restirPtMode) * 59u +
        settings.risCandidateCount;

    const std::vector<float> source = output.linearRGB;
    const uint32_t width = output.width;
    const uint32_t height = output.height;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const size_t idx = (static_cast<size_t>(y) * width + x) * 3u;
            simd::float3 color = simd_make_float3(0.0f, 0.0f, 0.0f);
            switch (settings.restirDebugView) {
                case RestirDebugView::CandidateSourceId: {
                    const uint32_t cell = ((x / 8u) + (y / 8u) * 13u + modeSeed) & 31u;
                    color = risActive ? HashDebugColor(cell + 1u)
                                      : simd_make_float3(0.06f, 0.06f, 0.06f);
                    break;
                }
                case RestirDebugView::ReservoirConfidence: {
                    const float confidence = risActive
                        ? std::clamp(static_cast<float>(settings.risCandidateCount) / 16.0f, 0.05f, 1.0f)
                        : 0.0f;
                    color = simd_make_float3(confidence, confidence * 0.85f, confidence * 0.25f);
                    break;
                }
                case RestirDebugView::ReGIRCellId: {
                    const uint32_t cellX = x / 12u;
                    const uint32_t cellY = y / 12u;
                    color = worldActive
                        ? HashDebugColor(cellX * 73856093u ^ cellY * 19349663u ^ modeSeed)
                        : simd_make_float3(0.0f, 0.0f, 0.0f);
                    break;
                }
                case RestirDebugView::PathGuidingUsedMask: {
                    const float mask = pathGuidingUsed ? 1.0f : 0.0f;
                    color = simd_make_float3(0.1f * mask, mask, 0.25f * mask);
                    break;
                }
                case RestirDebugView::RestirPTReuseMask: {
                    const float mask = restirPtReuseUsed ? 1.0f : 0.0f;
                    color = simd_make_float3(mask, 0.45f * mask, 0.05f * mask);
                    break;
                }
                case RestirDebugView::SVGFVariance: {
                    const size_t right =
                        (static_cast<size_t>(y) * width + std::min(x + 1u, width - 1u)) * 3u;
                    const size_t down =
                        (static_cast<size_t>(std::min(y + 1u, height - 1u)) * width + x) * 3u;
                    const double l = LuminanceLinear(source[idx], source[idx + 1u], source[idx + 2u]);
                    const double lr = LuminanceLinear(source[right], source[right + 1u], source[right + 2u]);
                    const double ld = LuminanceLinear(source[down], source[down + 1u], source[down + 2u]);
                    const float variance = svgfActive
                        ? std::clamp(static_cast<float>(std::abs(l - lr) + std::abs(l - ld)), 0.0f, 1.0f)
                        : 0.0f;
                    color = simd_make_float3(variance, variance, variance);
                    break;
                }
                case RestirDebugView::NaNInfMask: {
                    const bool invalid = !std::isfinite(source[idx]) ||
                                         !std::isfinite(source[idx + 1u]) ||
                                         !std::isfinite(source[idx + 2u]);
                    color = invalid ? simd_make_float3(1.0f, 0.0f, 0.0f)
                                    : simd_make_float3(0.0f, 0.0f, 0.0f);
                    break;
                }
                case RestirDebugView::QueueStageFailure:
                    color = simd_make_float3(0.0f, 0.0f, 0.0f);
                    break;
                case RestirDebugView::RadianceCache: {
                    const float confidence =
                        output.pbrMetrics.radianceCacheQueryCount > 0u
                            ? std::clamp(static_cast<float>(output.pbrMetrics.radianceCacheHitCount) /
                                             static_cast<float>(output.pbrMetrics.radianceCacheQueryCount),
                                         0.0f,
                                         1.0f)
                            : 0.0f;
                    const float train =
                        output.pbrMetrics.radianceCacheTrainCount > 0u ? 0.65f : 0.0f;
                    color = radianceCacheUsed ? simd_make_float3(confidence, train, 0.25f)
                                              : simd_make_float3(0.35f, 0.08f, train);
                    break;
                }
                case RestirDebugView::CausticTransport: {
                    const float connection =
                        output.pbrMetrics.causticEyeVertexCount > 0u
                            ? std::clamp(static_cast<float>(output.pbrMetrics.causticConnectionCandidateCount) /
                                             static_cast<float>(output.pbrMetrics.causticEyeVertexCount),
                                         0.0f,
                                         1.0f)
                            : 0.0f;
                    const float delta =
                        output.pbrMetrics.causticEyeVertexCount > 0u
                            ? std::clamp(static_cast<float>(output.pbrMetrics.causticDeltaChainCount) /
                                             static_cast<float>(output.pbrMetrics.causticEyeVertexCount),
                                         0.0f,
                                         1.0f)
                            : 0.0f;
                    const float classified =
                        output.pbrMetrics.causticDebugClassificationCount > 0u ? 0.75f : 0.0f;
                    color = causticTransportUsed ? simd_make_float3(connection, classified, delta)
                                                 : simd_make_float3(0.10f, 0.10f, 0.10f);
                    break;
                }
                case RestirDebugView::Beauty:
                    break;
            }
            output.linearRGB[idx] = color.x;
            output.linearRGB[idx + 1u] = color.y;
            output.linearRGB[idx + 2u] = color.z;
        }
    }
}

std::string JsonEscape(const std::string& input) {
    std::string escaped;
    escaped.reserve(input.size());
    for (char c : input) {
        switch (c) {
            case '\\':
                escaped += "\\\\";
                break;
            case '"':
                escaped += "\\\"";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped.push_back(c);
                break;
        }
    }
    return escaped;
}

void WriteJsonStringArray(std::ofstream& outFile, const std::vector<std::string>& values) {
    outFile << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            outFile << ", ";
        }
        outFile << "\"" << JsonEscape(values[i]) << "\"";
    }
    outFile << "]";
}

void WriteAssetPipelineReportJson(std::ostream& outFile,
                                  const PathTracer::SceneResources& resources) {
    const auto& report = resources.assetPipelineReport();
    outFile << "  \"asset_pipeline\": {\n";
    outFile << "    \"manifest\": {\n";
    outFile << "      \"source_path\": \"" << JsonEscape(report.manifest.sourcePath) << "\",\n";
    outFile << "      \"source_format\": \"" << JsonEscape(report.manifest.sourceFormat) << "\",\n";
    outFile << "      \"importer\": \"" << JsonEscape(report.manifest.importer) << "\",\n";
    outFile << "      \"importer_version\": \"" << JsonEscape(report.manifest.importerVersion) << "\",\n";
    outFile << "      \"generator\": \"" << JsonEscape(report.manifest.generator) << "\",\n";
    outFile << "      \"unit_policy\": \"" << JsonEscape(report.manifest.unitPolicy) << "\",\n";
    outFile << "      \"scene_unit_meters\": " << report.manifest.sceneUnitMeters << ",\n";
    outFile << "      \"maturity_label\": \"" << JsonEscape(report.manifest.maturityLabel) << "\"\n";
    outFile << "    },\n";
    outFile << "    \"statistics\": {\n";
    outFile << "      \"materials\": " << report.stats.materialCount << ",\n";
    outFile << "      \"meshes\": " << report.stats.meshCount << ",\n";
    outFile << "      \"mesh_instances\": " << report.stats.meshInstanceCount << ",\n";
    outFile << "      \"primitives\": " << report.stats.primitiveCount << ",\n";
    outFile << "      \"vertices\": " << report.stats.vertexCount << ",\n";
    outFile << "      \"indices\": " << report.stats.indexCount << ",\n";
    outFile << "      \"triangles\": " << report.stats.triangleCount << ",\n";
    outFile << "      \"degenerate_triangles\": " << report.stats.degenerateTriangleCount << ",\n";
    outFile << "      \"images_declared\": " << report.stats.imageCount << ",\n";
    outFile << "      \"textures_declared\": " << report.stats.textureCount << ",\n";
    outFile << "      \"textures_loaded\": " << report.stats.loadedTextureCount << ",\n";
    outFile << "      \"missing_normal_primitives\": " << report.stats.missingNormalPrimitiveCount << ",\n";
    outFile << "      \"missing_tangent_primitives\": " << report.stats.missingTangentPrimitiveCount << ",\n";
    outFile << "      \"unsupported_extensions\": " << report.stats.unsupportedExtensionCount << ",\n";
    outFile << "      \"warnings\": " << report.stats.warningCount << ",\n";
    outFile << "      \"errors\": " << report.stats.errorCount << "\n";
    outFile << "    },\n";
    outFile << "    \"support_matrix\": [";
    for (size_t i = 0; i < report.supportMatrix.size(); ++i) {
        if (i > 0) {
            outFile << ", ";
        }
        const auto& entry = report.supportMatrix[i];
        outFile << "{\"area\":\"" << JsonEscape(entry.area)
                << "\",\"status\":\"" << JsonEscape(entry.status)
                << "\",\"details\":\"" << JsonEscape(entry.details) << "\"}";
    }
    outFile << "],\n";
    outFile << "    \"stable_resource_ids\": [";
    for (size_t i = 0; i < report.stableResourceIds.size(); ++i) {
        if (i > 0) {
            outFile << ", ";
        }
        const auto& entry = report.stableResourceIds[i];
        outFile << "{\"resource_type\":\"" << JsonEscape(entry.resourceType)
                << "\",\"name\":\"" << JsonEscape(entry.name)
                << "\",\"source_index\":" << entry.sourceIndex
                << ",\"stable_id\":\"" << JsonEscape(entry.stableId) << "\"}";
    }
    outFile << "],\n";
    outFile << "    \"texture_policy\": {"
            << "\"residency_policy\":\"" << JsonEscape(report.texturePolicy.residencyPolicy)
            << "\",\"mip_policy\":\"" << JsonEscape(report.texturePolicy.mipPolicy)
            << "\",\"compression_policy\":\"" << JsonEscape(report.texturePolicy.compressionPolicy)
            << "\",\"budget_policy\":\"" << JsonEscape(report.texturePolicy.budgetPolicy)
            << "\",\"determinism_policy\":\"" << JsonEscape(report.texturePolicy.determinismPolicy)
            << "\",\"max_dimension\":" << report.texturePolicy.maxDimension
            << ",\"max_texture_bytes\":" << report.texturePolicy.maxTextureBytes
            << "},\n";
    outFile << "    \"material_overrides\": [";
    for (size_t i = 0; i < report.materialOverrides.size(); ++i) {
        if (i > 0) {
            outFile << ", ";
        }
        const auto& entry = report.materialOverrides[i];
        outFile << "{\"target\":\"" << JsonEscape(entry.target)
                << "\",\"action\":\"" << JsonEscape(entry.action)
                << "\",\"reason\":\"" << JsonEscape(entry.reason) << "\"}";
    }
    outFile << "],\n";
    outFile << "    \"texture_color_spaces\": [";
    for (size_t i = 0; i < report.textureColorSpaces.size(); ++i) {
        if (i > 0) {
            outFile << ", ";
        }
        const auto& entry = report.textureColorSpaces[i];
        outFile << "{\"texture_index\":" << entry.textureIndex
                << ",\"image_index\":" << entry.imageIndex
                << ",\"semantic\":\"" << JsonEscape(entry.semantic)
                << "\",\"color_space\":\"" << JsonEscape(entry.colorSpace)
                << "\",\"source\":\"" << JsonEscape(entry.source)
                << "\",\"status\":\"" << JsonEscape(entry.status) << "\"}";
    }
    outFile << "],\n";
    outFile << "    \"diagnostics\": [";
    for (size_t i = 0; i < report.diagnostics.size(); ++i) {
        if (i > 0) {
            outFile << ", ";
        }
        const auto& diagnostic = report.diagnostics[i];
        outFile << "{\"severity\":\"" << JsonEscape(diagnostic.severity)
                << "\",\"category\":\"" << JsonEscape(diagnostic.category)
                << "\",\"message\":\"" << JsonEscape(diagnostic.message)
                << "\",\"asset_path\":\"" << JsonEscape(diagnostic.assetPath)
                << "\",\"object_type\":\"" << JsonEscape(diagnostic.objectType)
                << "\",\"object_name\":\"" << JsonEscape(diagnostic.objectName)
                << "\",\"object_index\":" << diagnostic.objectIndex << "}";
    }
    outFile << "]\n";
    outFile << "  },\n";
}

void WriteRenderSettingsObjectJson(std::ostream& out,
                                   const CliOptions& options,
                                   const PathTracer::RenderSettings& settings,
                                   uint32_t sppTotal,
                                   const fs::path& outputPath) {
    out << "  \"render_mode\": \"" << RenderModeLabel(options.renderMode) << "\",\n";
    out << "  \"render_mode_maturity\": \"" << RenderModeMaturityLabel(options.renderMode) << "\",\n";
    out << "  \"render_mode_context\": \"" << RenderModeContextLabel(options.renderMode) << "\",\n";
    out << "  \"render_mode_source\": \"" << (options.renderModeSet ? "explicit" : "inferred_from_backend") << "\",\n";
    out << "  \"render_profile\": \"" << RenderProfileLabel(options.renderProfile) << "\",\n";
    out << "  \"render_profile_maturity\": \"" << RenderProfileMaturityLabel(options.renderProfile) << "\",\n";
    out << "  \"scene\": \"" << JsonEscape(options.scene) << "\",\n";
    out << "  \"backend\": \"" << (options.backend == HeadlessBackend::Embree ? "embree" : "metal") << "\",\n";
    out << "  \"output\": \"" << JsonEscape(outputPath.string()) << "\",\n";
    out << "  \"format\": \"" << PathTracer::FormatExtension(options.format) << "\",\n";
    out << "  \"spp_total\": " << sppTotal << ",\n";
    out << "  \"settings\": {\n";
    out << "    \"schema_version\": \"pathtracer.render_settings.v1\",\n";
    out << "    \"serialization_scope\": \"documented_headless_product_contract_v1\",\n";
    out << "    \"serialization_policy\": "
        << "\"deterministic headless workflow settings, not a private RenderSettings ABI dump\",\n";
    out << "    \"render_width\": " << settings.renderWidth << ",\n";
    out << "    \"render_height\": " << settings.renderHeight << ",\n";
    out << "    \"max_depth\": " << settings.maxDepth << ",\n";
    out << "    \"fixed_rng_seed\": " << settings.fixedRngSeed << ",\n";
    out << "    \"execution_mode\": \""
        << (settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront
                ? "wavefront"
                : "megakernel")
        << "\",\n";
    out << "    \"wavefront_scheduling_policy\": \""
        << WavefrontSchedulingPolicyLabel(settings.wavefrontSchedulingPolicy) << "\",\n";
    out << "    \"wavefront_compaction_enabled\": "
        << (settings.wavefrontCompactionEnabled ? "true" : "false") << ",\n";
    out << "    \"wavefront_shade_indirect_enabled\": "
        << (settings.wavefrontShadeIndirectEnabled ? "true" : "false") << ",\n";
    out << "    \"wavefront_shade_indirect_probe_enabled\": "
        << (settings.wavefrontShadeIndirectProbeEnabled ? "true" : "false") << ",\n";
    out << "    \"wavefront_shade_surface_probe_enabled\": "
        << (settings.wavefrontShadeSurfaceProbeEnabled ? "true" : "false") << ",\n";
    out << "    \"wavefront_shade_metadata_probe_enabled\": "
        << (settings.wavefrontShadeMetadataProbeEnabled ? "true" : "false") << ",\n";
    out << "    \"wavefront_shade_scalar_probe_enabled\": "
        << (settings.wavefrontShadeScalarProbeEnabled ? "true" : "false") << ",\n";
    out << "    \"wavefront_shade_closure_probe_enabled\": "
        << (settings.wavefrontShadeClosureProbeEnabled ? "true" : "false") << ",\n";
    out << "    \"wavefront_shade_bsdf_probe_enabled\": "
        << (settings.wavefrontShadeBsdfProbeEnabled ? "true" : "false") << ",\n";
    out << "    \"wavefront_shadow_indirect_enabled\": "
        << (settings.wavefrontShadowIndirectEnabled ? "true" : "false") << ",\n";
    out << "    \"direct_light_mode\": \"" << DirectLightModeLabel(settings.directLightMode) << "\",\n";
    out << "    \"ris_candidate_count\": " << settings.risCandidateCount << ",\n";
    out << "    \"enable_specular_nee\": " << (settings.enableSpecularNee ? "true" : "false") << ",\n";
    out << "    \"reference_point_exact_nee\": " << (settings.referencePointExactNee ? "true" : "false") << ",\n";
    out << "    \"enable_mnee\": " << (settings.enableMnee ? "true" : "false") << ",\n";
    out << "    \"restir_gi_mode\": \"" << RestirGiModeLabel(settings.restirGiMode) << "\",\n";
    out << "    \"path_guiding_mode\": \"" << PathGuidingModeLabel(settings.pathGuidingMode) << "\",\n";
    out << "    \"restir_pt_mode\": \"" << RestirPtModeLabel(settings.restirPtMode) << "\",\n";
    const bool restirPtEnhancedRequested = RestirPtEnhancedActive(settings.restirPtMode);
    const bool restirPtEnhancedActive = restirPtEnhancedRequested && options.backend == HeadlessBackend::Metal &&
        settings.volumeTransportMode != PathTracer::RenderSettings::VolumeTransportMode::Homogeneous;
    out << "    \"restir_pt_enhanced\": {\n";
    out << "      \"active\": " << (restirPtEnhancedActive ? "true" : "false") << ",\n";
    out << "      \"bias_mode\": \""
        << (restirPtEnhancedActive ? "directional_reuse_heuristic_no_unbiased_guarantee_or_cCap"
                : restirPtEnhancedRequested && options.backend == HeadlessBackend::Embree
                    ? "ordinary_cpu_capture_reuse_no_enhanced_implementation" : "inactive")
        << "\",\n";
    out << "      \"paired_spatial_reuse\": false,\n";
    out << "      \"footprint_reconnection\": false,\n";
    out << "      \"duplication_decorrelation\": false,\n";
    out << "      \"unified_di_gi\": false,\n";
    out << "      \"vector_shading_weights\": false,\n";
    out << "      \"dual_motion_vectors\": false,\n";
    out << "      \"replay_compaction\": false,\n";
    out << "      \"dispatch\": \"" << (!restirPtEnhancedRequested ? "inactive"
        : options.backend == HeadlessBackend::Embree ? "ordinary_cpu_capture_reuse"
        : restirPtEnhancedActive ? "fused_megakernel_or_wavefront_integrator" : "disabled_by_volume_fallback") << "\",\n";
    out << "      \"implementation_scope\": \"" << (options.backend == HeadlessBackend::Embree
        ? "ordinary_cpu_capture_reuse_no_enhanced_implementation"
        : "fused_directional_reuse_heuristics_no_GRIS_equivalence_unbiased_guarantee_or_cCap") << "\",\n";
    out << "      \"unsupported_paper_features\": [\"paired_spatial_reuse\", \"footprint_reconnection\", \"duplication_decorrelation\", \"unified_di_gi\", \"vector_shading_weights\", \"dual_motion_vectors\", \"replay_compaction\"]\n";
    out << "    },\n";
    out << "    \"radiance_cache_mode\": \"" << RadianceCacheModeLabel(settings.radianceCacheMode) << "\",\n";
    out << "    \"caustic_transport_mode\": \"" << CausticTransportModeLabel(settings.causticTransportMode) << "\",\n";
    out << "    \"spectral_rendering_mode\": \""
        << SpectralRenderingModeLabel(settings.spectralRenderingMode) << "\",\n";
    out << "    \"spectral_texture_policy\": \""
        << SpectralTexturePolicyLabel(settings.spectralTexturePolicy) << "\",\n";
    {
        // White balance carried by the uniforms (UniformBuilder): 1 / E[s_c] over the clamped range, 0 when off.
        const simd::float3 whiteBalance =
            settings.spectralRenderingMode != PathTracer::RenderSettings::SpectralRenderingMode::Off
                ? SpectralSensor::WhiteBalance(SpectralSensor::ClampRange(settings.spectralMinWavelengthNm,
                                                                          settings.spectralMaxWavelengthNm))
                : simd_make_float3(0.0f, 0.0f, 0.0f);
        out << "    \"spectral_white_balance\": [" << whiteBalance.x << ", " << whiteBalance.y << ", "
            << whiteBalance.z << "],\n";
    }
    out << "    \"spectral_wavelengths_per_path\": " << settings.spectralWavelengthsPerPath << ",\n";
    out << "    \"spectral_sampling_strategy\": \""
        << (settings.spectralWavelengthsPerPath == 1u ? "hero_wavelength_rotation_uniform_nm_c1"
                                                      : "hero_wavelength_rotation_uniform_nm_c3") << "\",\n";
    out << "    \"spectral_material_policy_contract\": "
        << "\"explicit scene spectral* tokens use sampled 450/550/650nm triplets; "
           "ordinary RGB assets are reported through spectral_fallback_count when spectral mode is enabled\",\n";
    out << "    \"spectral_texture_conversion_policy\": "
        << "\"RGB textures use rgb_reflectance_d65 approximation in spectral mode and are counted "
           "as spectral fallbacks after texture sampling\",\n";
    out << "    \"spectral_display_conversion\": "
        << "\"sampled_reflectance_times_approx_rgb_sensor_basis_to_existing_output_transform\",\n";
    out << "    \"ml_backend_mode\": \"" << MlBackendModeLabel(settings.mlBackendMode) << "\",\n";
    out << "    \"ml_backend_provider\": \""
        << MlBackendConfiguredProvider(settings.mlBackendMode) << "\",\n";
    out << "    \"ml_model_package_path\": \"" << JsonEscape(settings.mlModelPackagePath) << "\",\n";
    out << "    \"ml_debug_enabled\": " << (settings.mlDebug ? "true" : "false") << ",\n";
    out << "    \"restir_debug_enabled\": " << (settings.restirDebugEnabled ? "true" : "false") << ",\n";
    out << "    \"restir_debug_view\": \"" << RestirDebugViewLabel(settings.restirDebugView) << "\",\n";
    out << "    \"reconstruction_aov_contract_version\": "
        << PathTracerShaderTypes::kReconstructionAovContractVersion << ",\n";
    out << "    \"reconstruction_aov_contract_channels\": \""
        << PathTracerShaderTypes::kReconstructionAovContractChannels << "\",\n";
    out << "    \"camera\": {\n";
    out << "      \"target\": [" << settings.cameraTarget.x << ", " << settings.cameraTarget.y << ", " << settings.cameraTarget.z << "],\n";
    out << "      \"distance\": " << settings.cameraDistance << ",\n";
    out << "      \"yaw\": " << settings.cameraYaw << ",\n";
    out << "      \"pitch\": " << settings.cameraPitch << ",\n";
    out << "      \"vertical_fov\": " << settings.cameraVerticalFov << ",\n";
    out << "      \"defocus_angle\": " << settings.cameraDefocusAngle << ",\n";
    out << "      \"focus_distance\": " << settings.cameraFocusDistance << "\n";
    out << "    },\n";
    out << "    \"environment\": {\n";
    out << "      \"path\": \"" << JsonEscape(settings.environmentMapPath) << "\",\n";
    out << "      \"rotation_radians\": " << settings.environmentRotation << ",\n";
    out << "      \"intensity\": " << settings.environmentIntensity << ",\n";
    out << "      \"background_scale\": " << settings.environmentBackgroundScale << "\n";
    out << "    },\n";
    out << "    \"tonemap\": {\n";
    out << "      \"mode\": " << settings.tonemapMode << ",\n";
    out << "      \"aces_variant\": " << settings.acesVariant << ",\n";
    out << "      \"exposure\": " << settings.exposure << ",\n";
    out << "      \"srgb_output_encoding\": " << (settings.srgbOutputEncoding ? "true" : "false") << ",\n";
    const bool srgbTransfer = settings.srgbOutputEncoding ||
                              (settings.tonemapMode == 2 && settings.acesVariant == 2);
    out << "      \"output_transfer\": \""
        << (settings.tonemapMode == 6 ? "filmic" : (srgbTransfer ? "srgb" : "gamma2.2")) << "\",\n";
    out << "      \"reference_bloom_intensity\": " << settings.referenceBloomIntensity << ",\n";
    out << "      \"working_color_space\": \"Linear sRGB\"\n";
    out << "    },\n";
    out << "    \"offline_post\": {\n";
    out << "      \"denoise_enabled\": " << (settings.denoiseEnabled ? "true" : "false") << ",\n";
    out << "      \"denoise_hdr\": " << (settings.denoiseHdr ? "true" : "false") << ",\n";
    out << "      \"denoise_use_albedo\": " << (settings.denoiseUseAlbedo ? "true" : "false") << ",\n";
    out << "      \"denoise_use_normal\": " << (settings.denoiseUseNormal ? "true" : "false") << ",\n";
    out << "      \"svgf_denoise_enabled\": " << (settings.svgfDenoiseEnabled ? "true" : "false") << ",\n";
    out << "      \"svgf_denoise_direct\": " << (settings.svgfDenoiseDirect ? "true" : "false") << ",\n";
    out << "      \"svgf_denoise_indirect\": " << (settings.svgfDenoiseIndirect ? "true" : "false") << "\n";
    out << "    },\n";
    out << "    \"resource_policy\": {\n";
    out << "      \"budget_policy\": \"" << BudgetPolicyLabel(options.budgetPolicy) << "\",\n";
    out << "      \"pilot_mode\": \"" << PilotModeLabel(options.pilotMode) << "\",\n";
    out << "      \"texture_max_dim\": " << (options.textureMaxDimSet ? options.textureMaxDim : 0u) << ",\n";
    out << "      \"max_texture_bytes\": " << (options.maxTextureBytesSet ? options.maxTextureBytes : 0u) << "\n";
    out << "    }\n";
    out << "  }\n";
}

bool FsyncPath(const fs::path& path, std::string& error, const char* role) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        error = std::string("Failed to open ") + role + " for fsync: " + path.string();
        return false;
    }
    const bool ok = (::fsync(fd) == 0);
    ::close(fd);
    if (!ok) {
        error = std::string("Failed to fsync ") + role + ": " + path.string();
        return false;
    }
    return true;
}

bool PublishTempFileAtomic(const fs::path& tmpPath,
                           const fs::path& finalPath,
                           std::string& error) {
    if (!FsyncPath(tmpPath, error, "sidecar temp file")) {
        return false;
    }

    std::error_code renameEc;
    fs::rename(tmpPath, finalPath, renameEc);
    if (renameEc) {
        error = "Failed to publish sidecar file '" + finalPath.string() + "': " + renameEc.message();
        return false;
    }

    const fs::path parent = finalPath.parent_path().empty() ? fs::path(".") : finalPath.parent_path();
    std::string dirError;
    (void)FsyncPath(parent, dirError, "sidecar directory");
    return true;
}

bool WriteTextFileAtomic(const fs::path& path, const std::string& text, std::string& error) {
    if (!path.parent_path().empty()) {
        std::error_code dirEc;
        fs::create_directories(path.parent_path(), dirEc);
        if (dirEc) {
            error = "Failed to create sidecar directory '" + path.parent_path().string() + "': " + dirEc.message();
            return false;
        }
    }

    fs::path tmpPath = path;
    tmpPath += ".tmp";
    {
        std::ofstream out(tmpPath, std::ios::binary);
        if (!out.is_open()) {
            error = "Failed to open sidecar temp file: " + tmpPath.string();
            return false;
        }
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        if (!out) {
            error = "Failed while writing sidecar temp file: " + tmpPath.string();
            return false;
        }
    }

    return PublishTempFileAtomic(tmpPath, path, error);
}

bool OpenAtomicSidecarStream(const fs::path& path,
                             fs::path& tmpPath,
                             std::ofstream& outFile,
                             std::string& error,
                             const char* label) {
    if (!path.parent_path().empty()) {
        std::error_code dirEc;
        fs::create_directories(path.parent_path(), dirEc);
        if (dirEc) {
            error = std::string("Failed to create ") + label + " directory '" +
                    path.parent_path().string() + "': " + dirEc.message();
            return false;
        }
    }

    tmpPath = path;
    tmpPath += ".tmp";
    outFile.open(tmpPath, std::ios::binary);
    if (!outFile.is_open()) {
        error = std::string("Failed to open ") + label + " temp file for writing: " +
                tmpPath.string();
        return false;
    }
    return true;
}

bool PublishAtomicSidecarStream(std::ofstream& outFile,
                                const fs::path& tmpPath,
                                const fs::path& path,
                                std::string& error,
                                const char* label) {
    if (!outFile.good()) {
        error = std::string("Failed while writing ") + label + " temp file: " + tmpPath.string();
        return false;
    }
    outFile.close();
    if (!outFile) {
        error = std::string("Failed while closing ") + label + " temp file: " + tmpPath.string();
        return false;
    }
    return PublishTempFileAtomic(tmpPath, path, error);
}

std::string BuildRenderSettingsJson(const CliOptions& options,
                                    const PathTracer::RenderSettings& settings,
                                    const fs::path& outputPath) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(9);
    json << "{\n";
    WriteRenderSettingsObjectJson(json, options, settings, options.sppTotal, outputPath);
    json << "}\n";
    return json.str();
}

std::string BuildRenderQueueItemJson(const CliOptions& options,
                                     const PathTracer::RenderSettings& settings,
                                     const fs::path& outputPath) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(9);
    json << "{\n";
    json << "  \"schema_version\": \"pathtracer.render_queue_item.v1\",\n";
    json << "  \"kind\": \"single_frame_render\",\n";
    json << "  \"status\": \"ready\",\n";
    WriteRenderSettingsObjectJson(json, options, settings, options.sppTotal, outputPath);
    json << "}\n";
    return json.str();
}

// Random-walk subsurface outcomes over the whole render, in total and per material. Written on every render, so a walk
// that ended because its mesh is not closed, or because it hit the step limit, is never only a debug counter.
void WriteSssWalkJson(std::ostream& out, const HeadlessRenderOutput& output) {
    namespace st = PathTracerShaderTypes;
    const auto& perf = output.performance;
    const auto* v = perf.sssWalkStats;
    auto percent = [](uint64_t count, uint64_t walks) {
        return walks > 0u ? 100.0 * static_cast<double>(count) / static_cast<double>(walks) : 0.0;
    };
    out << "    \"sss_walk\": {\"walks\": " << v[st::kSssWalkStatWalks] << ", \"exits\": " << v[st::kSssWalkStatExits]
        << ", \"open_mesh\": " << v[st::kSssWalkStatOpenMesh] << ", \"absorbed\": " << v[st::kSssWalkStatAbsorbed]
        << ", \"step_limit\": " << v[st::kSssWalkStatStepLimit]
        << ", \"open_mesh_percent\": " << percent(v[st::kSssWalkStatOpenMesh], v[st::kSssWalkStatWalks])
        << ", \"step_limit_percent\": " << percent(v[st::kSssWalkStatStepLimit], v[st::kSssWalkStatWalks])
        << ", \"sss_max_steps\": " << perf.sssMaxSteps << ", \"materials\": [";
    for (size_t i = 0; i < perf.sssWalkMaterials.size(); ++i) {
        const auto& m = perf.sssWalkMaterials[i];
        out << (i > 0 ? ", " : "") << "{\"name\": \"" << JsonEscape(m.material)
            << "\", \"walks\": " << m.stats[st::kSssWalkStatWalks] << ", \"exits\": " << m.stats[st::kSssWalkStatExits]
            << ", \"open_mesh\": " << m.stats[st::kSssWalkStatOpenMesh]
            << ", \"absorbed\": " << m.stats[st::kSssWalkStatAbsorbed]
            << ", \"step_limit\": " << m.stats[st::kSssWalkStatStepLimit] << "}";
    }
    out << "]},\n";
}

void PrintSssWalkLine(const HeadlessRenderOutput& output) {
    namespace st = PathTracerShaderTypes;
    const auto& perf = output.performance;
    const auto* v = perf.sssWalkStats;
    if (v[st::kSssWalkStatWalks] == 0u) {
        return;
    }
    std::cout << "[SSS] random walks=" << v[st::kSssWalkStatWalks] << " exits=" << v[st::kSssWalkStatExits]
              << " openMesh=" << v[st::kSssWalkStatOpenMesh] << " absorbed=" << v[st::kSssWalkStatAbsorbed]
              << " stepLimit=" << v[st::kSssWalkStatStepLimit] << " (sssMaxSteps=" << perf.sssMaxSteps
              << "; openMesh and stepLimit walks end their path with no light)" << std::endl;
    for (const auto& m : perf.sssWalkMaterials) {
        const std::string line = PathTracer::SubsurfaceWalkLossDiagnostic(
            m.material, m.stats[st::kSssWalkStatWalks], m.stats[st::kSssWalkStatOpenMesh],
            m.stats[st::kSssWalkStatStepLimit], perf.sssMaxSteps);
        if (!line.empty()) {
            std::cout << "[SSS] " << line << std::endl;
        }
    }
}

std::string BuildOutputMetadataJson(const CliOptions& options,
                                    const PathTracer::RenderSettings& settings,
                                    const PathTracer::SceneResources& resources,
                                    const HeadlessRenderOutput& output,
                                    const fs::path& outputPath,
                                    const fs::path* rawOutputPath,
                                    const fs::path* denoisedOutputPath) {
    const PathTracer::SceneMemoryReport memory = resources.sceneMemoryReport();
    std::ostringstream json;
    json << std::fixed << std::setprecision(9);
    json << "{\n";
    json << "  \"schema_version\": \"pathtracer.output_metadata.v1\",\n";
    json << "  \"milestone\": \"T_renderer_productization\",\n";
    json << "  \"sidecar_write_policy\": \"atomic_temp_rename\",\n";
    WriteRenderSettingsObjectJson(json, options, settings, output.samples, outputPath);
    json << ",\n";
    const uint32_t exactPointCount = resources.zeroRadiusReferencePointLightCount();
    const uint32_t stochasticAnalyticCount = resources.analyticLightCount(true);
    json << "  \"reference_point_nee_population\": {\"zero_radius_reference_points\": " << exactPointCount
         << ", \"stochastic_analytic_lights\": " << stochasticAnalyticCount
         << ", \"exact_active\": " << (settings.referencePointExactNee && exactPointCount > 0u ? "true" : "false")
         << ", \"wavefront_shadow_stride\": " << (PathTracerShaderTypes::kWavefrontShadowCapacityMultiplier +
             (settings.referencePointExactNee ? exactPointCount : 0u)) << "},\n";

    if (rawOutputPath) {
        json << "  \"raw_output\": \"" << JsonEscape(rawOutputPath->string()) << "\",\n";
    }
    if (denoisedOutputPath) {
        json << "  \"denoised_output\": \"" << JsonEscape(denoisedOutputPath->string()) << "\",\n";
    }
    json << "  \"resolution\": {\"width\": " << output.width << ", \"height\": " << output.height << "},\n";
    json << "  \"spectral_last_frame_stats\": {\"wavelength_sample_count\": "
         << output.pbrMetrics.spectralWavelengthSampleCount
         << ", \"spectral_wavelength_channel_sample_count\": "
         << output.pbrMetrics.spectralWavelengthChannelSampleCount
         << ", \"spectral_collapse_count\": " << output.pbrMetrics.spectralCollapseCount
         << ", \"dispersion_event_count\": " << output.pbrMetrics.spectralDispersionEventCount << "},\n";
    json << "  \"timing\": {\"total_seconds\": " << output.totalSeconds
         << ", \"avg_ms_per_sample\": " << output.avgMsPerSample << "},\n";
    json << "  \"device_capabilities\": {\n";
    json << "    \"available\": " << (output.performance.available ? "true" : "false") << ",\n";
    json << "    \"hardware_raytracing_available\": "
         << (output.performance.hardwareRaytracingAvailable ? "true" : "false") << ",\n";
    json << "    \"hardware_raytracing_active\": "
         << (output.performance.hardwareRaytracingActive ? "true" : "false") << ",\n";
    json << "    \"hardware_raytracing_fallback_active\": "
         << (output.performance.hardwareRaytracingFallbackActive ? "true" : "false") << ",\n";
    json << "    \"hardware_raytracing_fallback_reason\": \""
         << JsonEscape(output.performance.hardwareRaytracingFallbackReason) << "\",\n";
    json << "    \"megakernel_integrate_pipeline\": \""
         << JsonEscape(output.performance.megakernelIntegratePipeline) << "\",\n";
    json << "    \"shader_directory\": \""
         << JsonEscape(PathTracer::ResolvedShaderDirectory()) << "\",\n";
    WriteSssWalkJson(json, output);
    json << "    \"wavefront_direct_light_fallback\": \""
         << JsonEscape(output.performance.wavefrontDirectLightFallback) << "\",\n";
    json << "    \"volume_direct_light_fallback\": \""
         << JsonEscape(VolumeDirectLightFallbackLabel(options, settings)) << "\",\n";
    json << "    \"wavefront_megakernel_fallback\": \""
         << JsonEscape(output.performance.wavefrontMegakernelFallback) << "\",\n";
    json << "    \"acceleration_build_policy\": \""
         << JsonEscape(output.performance.accelerationBuildPolicy) << "\",\n";
    json << "    \"acceleration_identity_stable\": "
         << (output.performance.accelerationIdentityStable ? "true" : "false") << "\n";
    json << "  },\n";
    json << "  \"resource_summary\": {\n";
    json << "    \"triangles\": " << memory.triangleCount << ",\n";
    json << "    \"meshes\": " << memory.meshCount << ",\n";
    json << "    \"instances\": " << memory.instanceCount << ",\n";
    json << "    \"textures\": " << memory.textureCount << ",\n";
    json << "    \"geometry_memory_bytes\": " << memory.geometryMemoryBytes << ",\n";
    json << "    \"texture_memory_bytes\": " << memory.textureMemoryBytes << ",\n";
    json << "    \"texture_allocated_memory_bytes\": " << memory.textureAllocatedMemoryBytes << ",\n";
    json << "    \"blas_memory_bytes\": " << memory.blasMemoryBytes << ",\n";
    json << "    \"tlas_memory_bytes\": " << memory.tlasMemoryBytes << ",\n";
    json << "    \"scratch_memory_bytes\": " << memory.scratchMemoryBytes << ",\n";
    json << "    \"total_estimated_memory_bytes\": " << memory.totalEstimatedMemoryBytes << ",\n";
    json << "    \"recommended_max_working_set_bytes\": " << memory.recommendedMaxWorkingSetSizeBytes << ",\n";
    json << "    \"budget_usage_percent\": " << memory.budgetUsagePercent << "\n";
    json << "  },\n";
    WriteAssetPipelineReportJson(json, resources);
    json << "  \"asset_pipeline_metadata_policy\": \"shared_scene_manifest_available_in_output_metadata_and_pbr_metrics\"\n";
    json << "}\n";
    return json.str();
}

std::string BuildCheckpointManifestJson(const CliOptions& options,
                                        const PathTracer::RenderSettings& settings,
                                        const HeadlessRenderOutput& output,
                                        const fs::path& outputPath,
                                        const fs::path* metadataPath,
                                        const fs::path* settingsPath,
                                        const fs::path* queuePath,
                                        const fs::path* statePath) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(9);
    json << "{\n";
    json << "  \"schema_version\": \"pathtracer.render_checkpoint.v1\",\n";
    json << "  \"checkpoint_kind\": \"completed_single_frame\",\n";
    // With the finished accumulation kept beside the output, a resume writes the output from it without rendering;
    // without it, a resume renders the frame again.
    json << "  \"resume_policy\": \""
         << (statePath ? "write_output_from_state" : "full_frame_rerender_from_checkpoint_manifest") << "\",\n";
    json << "  \"sample_accumulation_resume_supported\": " << (statePath ? "true" : "false") << ",\n";
    if (statePath) {
        json << "  \"accumulation_state\": \"" << JsonEscape(statePath->string()) << "\",\n";
        json << "  \"checkpoint_every_samples\": " << options.checkpointEverySamples << ",\n";
        if (options.forceHardwareRayTracingSet) {
            json << "  \"force_hardware_ray_tracing\": " << (options.forceHardwareRayTracing ? "true" : "false") << ",\n";
        }
        if (options.forceSoftwareRayTracingSet) {
            json << "  \"force_software_ray_tracing\": " << (options.forceSoftwareRayTracing ? "true" : "false") << ",\n";
        }
        json << "  \"write_denoise_guides\": " << (options.writeDenoiseGuides ? "true" : "false") << ",\n";
    }
    json << "  \"output\": \"" << JsonEscape(outputPath.string()) << "\",\n";
    json << "  \"completed_samples\": " << output.samples << ",\n";
    json << "  \"resolution\": {\"width\": " << output.width << ", \"height\": " << output.height << "},\n";
    if (metadataPath) {
        json << "  \"metadata_json\": \"" << JsonEscape(metadataPath->string()) << "\",\n";
    }
    if (settingsPath) {
        json << "  \"settings_json\": \"" << JsonEscape(settingsPath->string()) << "\",\n";
    }
    if (queuePath) {
        json << "  \"render_queue_item_json\": \"" << JsonEscape(queuePath->string()) << "\",\n";
    }
    json << "  \"queue_item\": {\n";
    json << "  \"schema_version\": \"pathtracer.render_queue_item.v1\",\n";
    json << "  \"kind\": \"single_frame_render\",\n";
    json << "  \"status\": \"ready\",\n";
    WriteRenderSettingsObjectJson(json, options, settings, options.sppTotal, outputPath);
    json << "  }\n";
    json << "}\n";
    return json.str();
}

std::string BuildPartialCheckpointManifestJson(const CliOptions& options,
                                               const PathTracer::RenderSettings& settings,
                                               const fs::path& outputPath,
                                               const fs::path& statePath,
                                               uint32_t completedSamples) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(9);
    json << "{\n";
    json << "  \"schema_version\": \"pathtracer.render_checkpoint.v1\",\n";
    json << "  \"checkpoint_kind\": \"partial_single_frame\",\n";
    json << "  \"resume_policy\": \"continue_accumulation_from_state\",\n";
    json << "  \"sample_accumulation_resume_supported\": true,\n";
    json << "  \"output\": \"" << JsonEscape(outputPath.string()) << "\",\n";
    json << "  \"completed_samples\": " << completedSamples << ",\n";
    json << "  \"target_samples\": " << options.sppTotal << ",\n";
    json << "  \"checkpoint_every_samples\": " << options.checkpointEverySamples << ",\n";
    json << "  \"accumulation_state\": \"" << JsonEscape(statePath.string()) << "\",\n";
    // Run flags that the queue item does not carry but that change the samples or the outputs: a resume must trace
    // with the same ray tracer and write the same guides.
    if (options.forceHardwareRayTracingSet) {
        json << "  \"force_hardware_ray_tracing\": " << (options.forceHardwareRayTracing ? "true" : "false") << ",\n";
    }
    if (options.forceSoftwareRayTracingSet) {
        json << "  \"force_software_ray_tracing\": " << (options.forceSoftwareRayTracing ? "true" : "false") << ",\n";
    }
    json << "  \"write_denoise_guides\": " << (options.writeDenoiseGuides ? "true" : "false") << ",\n";
    json << "  \"queue_item\": {\n";
    json << "  \"schema_version\": \"pathtracer.render_queue_item.v1\",\n";
    json << "  \"kind\": \"single_frame_render\",\n";
    json << "  \"status\": \"ready\",\n";
    WriteRenderSettingsObjectJson(json, options, settings, options.sppTotal, outputPath);
    json << "  }\n";
    json << "}\n";
    return json.str();
}

// Sample checkpoints continue one accumulation, so every sample must depend only on its own counters and pixel.
// Features that carry state from earlier frames (reservoirs, caches, guiding, SVGF history) are refused by name.
std::optional<std::string> SampleCheckpointRefusal(const CliOptions& options, const PathTracer::RenderSettings& s) {
    using DL = PathTracer::RenderSettings::DirectLightMode;
    if (options.backend != HeadlessBackend::Metal) return std::string("the Embree backend");
    if (options.outputPath.empty()) return std::string("no --output path");
    if (options.validateRis) return std::string("--validate-ris");
    if (s.directLightMode != DL::LegacyRect && s.directLightMode != DL::BaselineEmissive && s.directLightMode != DL::Ris) {
        return std::string("directLightMode ") + DirectLightModeLabel(s.directLightMode) +
               " (only legacy, baseline_emissive and ris keep no state between frames)";
    }
    if (s.svgfDenoiseEnabled) return std::string("SVGF denoising (temporal history)");
    if (s.restirGiMode != PathTracer::RenderSettings::RestirGiMode::Off) return std::string("ReSTIR GI");
    if (s.restirPtMode != PathTracer::RenderSettings::RestirPtMode::Off) return std::string("ReSTIR PT");
    if (s.pathGuidingMode != PathTracer::RenderSettings::PathGuidingMode::Off) return std::string("path guiding");
    if (s.radianceCacheMode != PathTracer::RenderSettings::RadianceCacheMode::Off) return std::string("the radiance cache");
    if (s.causticTransportMode != PathTracer::RenderSettings::CausticTransportMode::Off) return std::string("caustic transport");
    if (s.mlBackendMode != PathTracer::RenderSettings::MlBackendMode::Off) return std::string("the ML backend");
    return std::nullopt;
}

std::string BuildTileManifestJson(const CliOptions& options,
                                  const HeadlessRenderOutput& output,
                                  const fs::path& outputPath) {
    const uint32_t tileWidth = options.tileSizeSet ? options.tileWidth : 512u;
    const uint32_t tileHeight = options.tileSizeSet ? options.tileHeight : 512u;
    const uint32_t tilesX = (output.width + tileWidth - 1u) / tileWidth;
    const uint32_t tilesY = (output.height + tileHeight - 1u) / tileHeight;
    std::ostringstream json;
    json << "{\n";
    json << "  \"schema_version\": \"pathtracer.tile_manifest.v1\",\n";
    json << "  \"chunk_policy\": \"deterministic_full_frame_tile_plan\",\n";
    json << "  \"tile_rendering_backend\": \"not_split_at_dispatch_boundary\",\n";
    json << "  \"output\": \"" << JsonEscape(outputPath.string()) << "\",\n";
    json << "  \"resolution\": {\"width\": " << output.width << ", \"height\": " << output.height << "},\n";
    json << "  \"tile_size\": {\"width\": " << tileWidth << ", \"height\": " << tileHeight << "},\n";
    json << "  \"tile_count\": " << (tilesX * tilesY) << ",\n";
    json << "  \"tiles\": [\n";
    for (uint32_t tileY = 0; tileY < tilesY; ++tileY) {
        for (uint32_t tileX = 0; tileX < tilesX; ++tileX) {
            const uint32_t x0 = tileX * tileWidth;
            const uint32_t y0 = tileY * tileHeight;
            const uint32_t x1 = std::min(x0 + tileWidth, output.width);
            const uint32_t y1 = std::min(y0 + tileHeight, output.height);
            const uint32_t tileIndex = tileY * tilesX + tileX;
            json << "    {\"index\": " << tileIndex
                 << ", \"x\": " << x0
                 << ", \"y\": " << y0
                 << ", \"width\": " << (x1 - x0)
                 << ", \"height\": " << (y1 - y0)
                 << "}";
            if (tileIndex + 1u < tilesX * tilesY) {
                json << ",";
            }
            json << "\n";
        }
    }
    json << "  ]\n";
    json << "}\n";
    return json.str();
}

std::string BuildDebugBundleManifestJson(const fs::path& bundleDir,
                                         const fs::path& outputPath,
                                         const fs::path* metadataPath,
                                         const fs::path* settingsPath,
                                         const fs::path* queuePath,
                                         const fs::path* metricsPath,
                                         const fs::path* checkpointPath,
                                         const fs::path* tileManifestPath) {
    std::vector<std::pair<std::string, fs::path>> artifacts;
    auto addArtifact = [&](const char* role, const fs::path* path) {
        if (path) {
            artifacts.emplace_back(role, *path);
        }
    };
    addArtifact("metadata", metadataPath);
    addArtifact("settings", settingsPath);
    addArtifact("queue_item", queuePath);
    addArtifact("pbr_metrics", metricsPath);
    addArtifact("checkpoint", checkpointPath);
    addArtifact("tile_manifest", tileManifestPath);

    std::ostringstream json;
    json << "{\n";
    json << "  \"schema_version\": \"pathtracer.debug_bundle.v1\",\n";
    json << "  \"bundle_scope\": \"completed_or_late_render_product_artifacts\",\n";
    json << "  \"failure_context_policy\": "
         << "\"early setup failures report through process status and batch summaries; "
            "this manifest records artifacts published by the completed render path\",\n";
    json << "  \"bundle_dir\": \"" << JsonEscape(bundleDir.string()) << "\",\n";
    json << "  \"output\": \"" << JsonEscape(outputPath.string()) << "\",\n";
    json << "  \"artifact_integrity_policy\": "
         << "\"record file existence and byte size after atomic sidecar publication\",\n";
    json << "  \"artifact_count\": " << artifacts.size() << ",\n";
    json << "  \"files\": {\n";
    for (size_t i = 0; i < artifacts.size(); ++i) {
        json << "    \"" << JsonEscape(artifacts[i].first) << "\": \""
             << JsonEscape(artifacts[i].second.string()) << "\"";
        if (i + 1u < artifacts.size()) {
            json << ",\n";
        } else {
            json << "\n";
        }
    }
    json << "  },\n";
    json << "  \"artifacts\": [\n";
    for (size_t i = 0; i < artifacts.size(); ++i) {
        std::error_code existsEc;
        const bool exists = fs::exists(artifacts[i].second, existsEc);
        std::error_code sizeEc;
        const uint64_t byteSize =
            exists ? static_cast<uint64_t>(fs::file_size(artifacts[i].second, sizeEc)) : 0u;
        json << "    {\"role\": \"" << JsonEscape(artifacts[i].first)
             << "\", \"path\": \"" << JsonEscape(artifacts[i].second.string())
             << "\", \"exists\": " << (exists ? "true" : "false")
             << ", \"byte_size\": " << (sizeEc ? 0u : byteSize) << "}";
        if (i + 1u < artifacts.size()) {
            json << ",";
        }
        json << "\n";
    }
    json << "  ]\n";
    json << "}\n";
    return json.str();
}

// Radiance cache counters (whole render, RadianceCacheTraining::RadianceCacheStat layout) as one JSON object.
void WriteRadianceCacheStatsJson(std::ostream& out, const HeadlessRenderOutput::PbrMetrics& metrics) {
    namespace rct = PathTracer::RadianceCacheTraining;
    const auto* v = metrics.radianceCacheStats;
    auto luma = [](uint64_t fixed) { return static_cast<double>(fixed) / static_cast<double>(rct::kStatLumaScale); };
    out << "{";
    for (uint32_t i = 0u; i < rct::kStatFallbackReasonBase; ++i) {
        out << "\"" << rct::StatName(i) << "\": ";
        if (rct::StatIsLumaFixed(i)) {
            out << std::setprecision(9) << luma(v[i]);
        } else {
            out << v[i];
        }
        out << ", ";
    }
    const double targetLuma = luma(v[rct::kStatTargetLumaFixed]);
    out << "\"clamped_energy_fraction\": "
        << (targetLuma > 0.0 ? luma(v[rct::kStatClampedLumaFixed]) / targetLuma : 0.0) << ", ";
    out << "\"fallback_reasons\": {";
    for (uint32_t r = 0u; r < rct::kStatFallbackReasonCount; ++r) {
        out << (r ? ", " : "") << "\"" << rct::FallbackReasonName(r) << "\": " << v[rct::kStatFallbackReasonBase + r];
    }
    out << "}, \"depth\": [";
    for (uint32_t d = 0u; d < rct::kStatDepthBins; ++d) {
        const uint64_t queries = v[rct::kStatDepthQueryBase + d];
        const uint64_t hits = v[rct::kStatDepthHitBase + d];
        out << (d ? ", " : "") << "{\"depth\": " << d << ", \"queries\": " << queries << ", \"hits\": " << hits
            << ", \"hit_rate\": " << (queries ? static_cast<double>(hits) / static_cast<double>(queries) : 0.0)
            << ", \"mean_returned_luma\": "
            << (hits ? luma(v[rct::kStatDepthHitLumaFixedBase + d]) / static_cast<double>(hits) : 0.0) << "}";
    }
    out << "], \"max_stored_sum\": {";
    for (uint32_t c = 0u; c < rct::kStatMaxStoredSumCount; ++c) {
        const uint64_t m = v[rct::kStatMaxStoredSumBase + c];
        out << (c ? ", " : "") << "\"" << rct::MaxStoredSumName(c) << "\": {\"value\": " << m
            << ", \"fraction_of_uint32_max\": " << static_cast<double>(m) / 4294967295.0 << "}";
    }
    out << "}, \"wrapped_sums\": " << v[rct::kStatWrappedSums] << "}";
}

void PrintRadianceCacheStatsLine(const HeadlessRenderOutput::PbrMetrics& metrics) {
    namespace rct = PathTracer::RadianceCacheTraining;
    const auto* v = metrics.radianceCacheStats;
    uint64_t any = 0u;
    for (uint32_t i = 0u; i < rct::kStatCount; ++i) {
        any |= v[i];
    }
    if (any == 0u) {
        return;
    }
    const double targetLuma = static_cast<double>(v[rct::kStatTargetLumaFixed]);
    std::cout << "[PBR] radianceCacheStats(records=" << v[rct::kStatTrainRecords]
              << ",accepted=" << v[rct::kStatTrainAccepted] << ",clamped=" << v[rct::kStatTrainClamped]
              << ",rejectedNonFinite=" << v[rct::kStatTrainRejectedNonFinite]
              << ",rejectedZeroThroughput=" << v[rct::kStatTrainRejectedZeroThroughput]
              << ",saturated=" << v[rct::kStatTrainSaturated] << ",evictions=" << v[rct::kStatCellEvictions]
              << ",clampedEnergyFraction="
              << (targetLuma > 0.0 ? static_cast<double>(v[rct::kStatClampedLumaFixed]) / targetLuma : 0.0) << ")";
    std::cout << " fallbackReasons(";
    for (uint32_t r = 1u; r < rct::kStatFallbackReasonCount; ++r) {
        std::cout << (r > 1u ? "," : "") << rct::FallbackReasonName(r) << "=" << v[rct::kStatFallbackReasonBase + r];
    }
    std::cout << ") depthQueriesHits(";
    for (uint32_t d = 0u; d < rct::kStatDepthBins; ++d) {
        std::cout << (d ? "," : "") << "d" << d << "=" << v[rct::kStatDepthQueryBase + d] << "/"
                  << v[rct::kStatDepthHitBase + d];
    }
    std::cout << ") maxStoredSumOfUint32Max(";
    for (uint32_t c = 0u; c < rct::kStatMaxStoredSumCount; ++c) {
        std::cout << (c ? "," : "") << rct::MaxStoredSumName(c) << "="
                  << static_cast<double>(v[rct::kStatMaxStoredSumBase + c]) / 4294967295.0;
    }
    std::cout << ",wrapped=" << v[rct::kStatWrappedSums] << ")" << std::endl;
}

bool WritePbrMetricsJson(const fs::path& jsonPath,
                         const CliOptions& options,
                         const PathTracer::RenderSettings& settings,
                         const PathTracer::SceneResources& resources,
                         const fs::path& outputPath,
                         const fs::path* rawOutputPath,
                         const fs::path* denoisedOutputPath,
                         const HeadlessRenderOutput& output,
                         const ImagePbrMetrics& imageMetrics,
                         std::string& error) {
    if (!jsonPath.parent_path().empty()) {
        std::error_code dirEc;
        fs::create_directories(jsonPath.parent_path(), dirEc);
        if (dirEc) {
            error = "Failed to create metrics directory '" + jsonPath.parent_path().string() + "': " + dirEc.message();
            return false;
        }
    }

    fs::path tmpPath = jsonPath;
    tmpPath += ".tmp";
    std::ofstream outFile(tmpPath);
    if (!outFile.is_open()) {
        error = "Failed to open metrics temp file for writing: " + tmpPath.string();
        return false;
    }

    const bool guardedShadeIndirect =
        settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront &&
        (settings.wavefrontShadeIndirectEnabled || WavefrontShadeIndirectEnabled());
    const bool guardedShadeIndirectProbe =
        settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront &&
        (settings.wavefrontShadeIndirectProbeEnabled || WavefrontShadeIndirectProbeEnabled());
    const bool guardedShadeSurfaceProbe =
        settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront &&
        (settings.wavefrontShadeSurfaceProbeEnabled || WavefrontShadeSurfaceProbeEnabled());
    const bool guardedShadeMetadataProbe =
        settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront &&
        (settings.wavefrontShadeMetadataProbeEnabled || WavefrontShadeMetadataProbeEnabled());
    const bool guardedShadeScalarProbe =
        settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront &&
        (settings.wavefrontShadeScalarProbeEnabled || WavefrontShadeScalarProbeEnabled());
    const bool guardedShadeBsdfProbe =
        settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront &&
        (settings.wavefrontShadeBsdfProbeEnabled || WavefrontShadeBsdfProbeEnabled());
    const bool guardedShadeClosureProbe =
        settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront &&
        (settings.wavefrontShadeClosureProbeEnabled || WavefrontShadeClosureProbeEnabled());
    const bool guardedShadowIndirect =
        settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront &&
        (settings.wavefrontShadowIndirectEnabled || WavefrontShadowIndirectEnabled());
    const PathTracer::SceneMemoryReport sceneMemory = resources.sceneMemoryReport();
    const bool wavefrontExecution =
        settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront;
    const char* shadeDispatchMode =
        guardedShadeIndirect
            ? "split_resolved_surface_active_count_indirect"
            : (wavefrontExecution ? "direct_grid_exact_platform_limited" : "disabled_non_wavefront");
    const char* shadowDispatchMode =
        guardedShadowIndirect
            ? "guarded_active_count_indexed_indirect"
            : (wavefrontExecution ? "direct_grid_sparse_storage" : "disabled_non_wavefront");
    const uint64_t renderPixelCount =
        static_cast<uint64_t>(output.width) * static_cast<uint64_t>(output.height);
    const uint64_t shadeDispatchedThreadCount =
        guardedShadeIndirect
            ? output.performance.wavefrontShadeProcessedCount
            : renderPixelCount * output.performance.wavefrontShadeDispatchCount;
    const uint64_t shadowDispatchedThreadCount =
        guardedShadowIndirect
            ? std::max(output.performance.wavefrontShadowProcessedCount,
                       output.performance.wavefrontShadowRayEnqueueCount)
            : renderPixelCount * output.performance.wavefrontShadowDispatchCount;
    const uint64_t shadeInactiveLaneEstimate =
        shadeDispatchedThreadCount > output.performance.wavefrontShadeProcessedCount
            ? shadeDispatchedThreadCount - output.performance.wavefrontShadeProcessedCount
            : 0u;
    const uint64_t shadowInactiveLaneEstimate =
        shadowDispatchedThreadCount > output.performance.wavefrontShadowProcessedCount
            ? shadowDispatchedThreadCount - output.performance.wavefrontShadowProcessedCount
            : 0u;

    outFile << std::fixed << std::setprecision(9);
    outFile << "{\n";
    outFile << "  \"scene\": \"" << JsonEscape(options.scene) << "\",\n";
    outFile << "  \"backend\": \"" << (options.backend == HeadlessBackend::Embree ? "embree" : "metal")
            << "\",\n";
    outFile << "  \"render_mode\": \"" << RenderModeLabel(options.renderMode) << "\",\n";
    outFile << "  \"render_mode_maturity\": \"" << RenderModeMaturityLabel(options.renderMode) << "\",\n";
    outFile << "  \"render_mode_context\": \"" << RenderModeContextLabel(options.renderMode) << "\",\n";
    outFile << "  \"render_profile\": \"" << RenderProfileLabel(options.renderProfile) << "\",\n";
    outFile << "  \"render_profile_maturity\": \"" << RenderProfileMaturityLabel(options.renderProfile) << "\",\n";
    outFile << "  \"output\": \"" << JsonEscape(outputPath.string()) << "\",\n";
    if (rawOutputPath) {
        outFile << "  \"raw_output\": \"" << JsonEscape(rawOutputPath->string()) << "\",\n";
    }
    if (denoisedOutputPath) {
        outFile << "  \"denoised_output\": \"" << JsonEscape(denoisedOutputPath->string()) << "\",\n";
    }
    outFile << "  \"seed\": " << settings.fixedRngSeed << ",\n";
    outFile << "  \"settings\": {\n";
    outFile << "    \"environment_intensity\": " << settings.environmentIntensity << ",\n";
    outFile << "    \"environment_background_scale\": " << settings.environmentBackgroundScale << ",\n";
    outFile << "    \"environment_background_scale_scope\": \"depth_zero_hdr_miss_only\",\n";
    outFile << "    \"execution_mode\": \""
            << (settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront
                    ? "wavefront"
                    : "megakernel")
            << "\",\n";
    outFile << "    \"wavefront_scheduling_policy\": \""
            << WavefrontSchedulingPolicyLabel(settings.wavefrontSchedulingPolicy) << "\",\n";
    outFile << "    \"wavefront_compaction_enabled\": "
            << (settings.wavefrontCompactionEnabled ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_shade_indirect_enabled\": "
            << (guardedShadeIndirect ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_shade_indirect_probe_enabled\": "
            << (guardedShadeIndirectProbe ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_shade_surface_probe_enabled\": "
            << (guardedShadeSurfaceProbe ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_shade_metadata_probe_enabled\": "
            << (guardedShadeMetadataProbe ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_shade_scalar_probe_enabled\": "
            << (guardedShadeScalarProbe ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_shade_closure_probe_enabled\": "
            << (guardedShadeClosureProbe ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_shade_bsdf_probe_enabled\": "
            << (guardedShadeBsdfProbe ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_shadow_indirect_enabled\": "
            << (guardedShadowIndirect ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_overflow_policy\": \"clamp_and_report\",\n";
    outFile << "    \"wavefront_material_ray_sorting\": \"experimental_disabled\",\n";
    outFile << "    \"wavefront_active_count_indirect_stages\": ";
    {
        std::vector<std::string> stages{"intersect"};
        if (guardedShadeIndirectProbe) stages.emplace_back("shade_probe");
        if (guardedShadeSurfaceProbe) stages.emplace_back("shade_surface_probe");
        if (guardedShadeMetadataProbe) stages.emplace_back("shade_metadata_probe");
        if (guardedShadeScalarProbe) stages.emplace_back("shade_scalar_probe");
        if (guardedShadeClosureProbe) stages.emplace_back("shade_closure_probe");
        if (guardedShadeBsdfProbe) stages.emplace_back("shade_bsdf_probe");
        if (guardedShadeIndirect) {
            stages.emplace_back("shade_resolve_surface_guarded");
            stages.emplace_back("shade_resolved_guarded");
        }
        if (guardedShadowIndirect) stages.emplace_back("shadow_guarded");
        WriteJsonStringArray(outFile, stages);
    }
    outFile << ",\n";
    outFile << "    \"wavefront_direct_grid_stages\": ";
    if (guardedShadeIndirect && guardedShadowIndirect) {
        outFile << "[]";
    } else if (guardedShadeIndirect) {
        outFile << "[\"shadow\"]";
    } else if (guardedShadowIndirect) {
        outFile << "[\"shade\"]";
    } else {
        outFile << "[\"shade\", \"shadow\"]";
    }
    outFile << ",\n";
    outFile << "    \"wavefront_guarded_only_stages\": ["
            << "\"shade_indirect_negative_repro\","
            << "\"shade_indirect_probe\","
            << "\"shade_surface_probe\","
            << "\"shade_metadata_probe\","
            << "\"shade_scalar_probe\","
            << "\"shade_closure_probe\","
            << "\"shade_bsdf_probe\","
            << "\"shadow_indirect_guard\"],\n";
    outFile << "    \"wavefront_platform_limited_stages\": ";
    outFile << (wavefrontExecution && !guardedShadeIndirect ? "[\"shade\"]" : "[]");
    outFile << ",\n";
    outFile << "    \"wavefront_stage_dispatch_modes\": {"
            << "\"intersect\":\"active_count_indirect\","
            << "\"shade\":\"" << shadeDispatchMode << "\","
            << "\"shadow\":\"" << shadowDispatchMode << "\"},\n";
    outFile << "    \"wavefront_stage_fallback_reasons\": {"
            << "\"intersect\":\"none\","
            << "\"shade\":\""
            << (guardedShadeIndirect
                    ? "explicit_diagnostic_guard_requested_not_production_closure"
                    : "exact_shade_active_count_indirect_reproduces_code_8_default_exact_direct_grid_platform_limited")
            << "\",\"shadow\":\""
            << (guardedShadowIndirect
                    ? "explicit_guarded_indexed_shadow_dispatch_requested_not_default_promoted"
                    : "sparse_shadow_storage_requires_indexed_sidecar_until_promotion_matrix_passes")
            << "\"},\n";
    outFile << "    \"wavefront_m1_shade_indirect_guard_available\": true,\n";
    outFile << "    \"wavefront_m1_shade_indirect_guard_default_enabled\": false,\n";
    outFile << "    \"wavefront_m1_shadow_indirect_guard_available\": true,\n";
    outFile << "    \"wavefront_m1_shadow_indirect_promoted\": false,\n";
    outFile << "    \"wavefront_m1_bridge_status\": \""
            << (guardedShadeIndirect
                    ? "M.1 reframed closure diagnostic active: shade is split into active-count indirect surface/material resolution plus active-count indirect resolved continuation by --wavefront-shade-indirect=1 or PT_ENABLE_WAVEFRONT_SHADE_INDIRECT=1; this remains a default-off Code 8 diagnostic and is not original shade active-count closure"
                    : (guardedShadeIndirectProbe || guardedShadeSurfaceProbe || guardedShadeMetadataProbe || guardedShadeScalarProbe || guardedShadeClosureProbe || guardedShadeBsdfProbe)
                        ? "M.1 reframed closure probe active: a non-mutating shade active-count indirect probe is enabled before the production direct-grid shade path; production shade remains exact direct-grid platform-limited"
                    : guardedShadowIndirect
                        ? "M.1 reframed closure shadow guard active: compact shadow index sidecar drives guarded active-count indexed shadow dispatch; guarded shadow remains default-off/not promoted and original shade indirect remains diagnostic"
                    : "Milestone M.1 closed under the 2026-06-07 reframed stage-specific scheduling contract: intersect is active-count indirect, production shade is exact direct-grid platform-limited with residual metrics, default shadow is direct-grid sparse storage, and original shade active-count indirect remains a default-off Code 8 diagnostic")
            << "\",\n";
    outFile << "    \"direct_light_mode\": \"" << DirectLightModeLabel(settings.directLightMode) << "\",\n";
    outFile << "    \"restir_gi_mode\": \"" << RestirGiModeLabel(settings.restirGiMode) << "\",\n";
    outFile << "    \"path_guiding_mode\": \"" << PathGuidingModeLabel(settings.pathGuidingMode) << "\",\n";
    outFile << "    \"path_guiding_strength\": " << settings.pathGuidingStrength << ",\n";
    outFile << "    \"path_guiding_cell_size\": " << settings.pathGuidingCellSize << ",\n";
    outFile << "    \"path_guiding_training_interval\": " << settings.pathGuidingTrainingInterval << ",\n";
    outFile << "    \"path_guiding_provider\": \""
            << (settings.pathGuidingMode == PathTracer::RenderSettings::PathGuidingMode::Prototype
                    ? "PathGuiding.WorldGrid"
                    : "PathGuiding.NoOp")
            << "\",\n";
    outFile << "    \"path_guiding_provider_production\": "
            << (settings.pathGuidingMode == PathTracer::RenderSettings::PathGuidingMode::Prototype
                    ? "true"
                    : "false")
            << ",\n";
    outFile << "    \"path_guiding_storage\": \""
            << (options.backend == HeadlessBackend::Embree ? "cpu_host_memory" : "private_transport_memory") << "\",\n";
    outFile << "    \"path_guiding_query_contract\": [\"canGuide\", \"sampleDirection\", \"pdfDirection\", \"update\", \"confidence\", \"debugRecord\"],\n";
    outFile << "    \"path_guiding_fallback_policy\": \"low_confidence_invalid_bin_or_unsupported_surface_fallback_to_bsdf_sampling\",\n";
    outFile << "    \"restir_pt_mode\": \"" << RestirPtModeLabel(settings.restirPtMode) << "\",\n";
    outFile << "    \"restir_pt_max_reservoirs\": " << settings.restirPtMaxReservoirs << ",\n";
    outFile << "    \"restir_pt_debug\": " << (settings.restirPtDebug ? "true" : "false") << ",\n";
    outFile << "    \"restir_pt_reuse_strength\": " << settings.restirPtReuseStrength << ",\n";
    outFile << "    \"radiance_cache_mode\": \"" << RadianceCacheModeLabel(settings.radianceCacheMode) << "\",\n";
    outFile << "    \"radiance_cache_cell_size\": " << settings.radianceCacheCellSize << ",\n";
    outFile << "    \"radiance_cache_min_depth\": " << settings.radianceCacheMinDepth << ",\n";
    outFile << "    \"radiance_cache_min_confidence\": " << settings.radianceCacheMinConfidence << ",\n";
    outFile << "    \"radiance_cache_training_interval\": " << settings.radianceCacheTrainingInterval << ",\n";
    outFile << "    \"radiance_cache_max_radiance\": " << settings.radianceCacheMaxRadiance << ",\n";
    const bool volumeHomogeneous =
        settings.volumeTransportMode == PathTracer::RenderSettings::VolumeTransportMode::Homogeneous;
    // Both backends switch off the modes whose state carries no medium (Metal's UniformBuilder; Embree since EP9).
    const bool volumeModesDemoted = volumeHomogeneous;
    const char* radianceCacheRefusal = PathTracer::RadianceCacheRefusalReason(settings);
    const bool volumeRadianceCacheDisabled =
        volumeModesDemoted && PathTracer::RadianceCacheActive(settings);
    outFile << "    \"radiance_cache_provider\": \""
            << (settings.radianceCacheMode == PathTracer::RenderSettings::RadianceCacheMode::Off
                    ? "RadianceCache.NoOp"
                    : radianceCacheRefusal ? "RadianceCache.Refused"
                    : (volumeRadianceCacheDisabled ? "RadianceCache.DisabledByVolumeFallback"
                                                   : "RadianceCache.WorldGrid"))
            << "\",\n";
    outFile << "    \"radiance_cache_refusal_reason\": \""
            << (radianceCacheRefusal ? radianceCacheRefusal : "none") << "\",\n";
    outFile << "    \"radiance_cache_provider_production\": "
            << (settings.radianceCacheMode != PathTracer::RenderSettings::RadianceCacheMode::Off &&
                        !radianceCacheRefusal && !volumeRadianceCacheDisabled
                    ? "true"
                    : "false")
            << ",\n";
    outFile << "    \"radiance_cache_storage\": \""
            << (options.backend == HeadlessBackend::Embree ? "cpu_host_memory" : "private_transport_memory") << "\",\n";
    outFile << "    \"radiance_cache_query_contract\": [\"canQuery\", \"query\", \"train\", \"confidence\", \"variance\", \"cacheability\", \"invalidation\", \"debugRecord\"],\n";
    outFile << "    \"radiance_cache_fallback_policy\": \"confidence_or_bias_risk_fallback_to_path_tracing\",\n";
    outFile << "    \"radiance_cache_reconstruction_confidence_source\": \"RadianceEstimate.radianceConfidence.w -> ReconstructionAOV confidence channel on cache hits\",\n";
    outFile << "    \"caustic_transport_mode\": \"" << CausticTransportModeLabel(settings.causticTransportMode) << "\",\n";
    outFile << "    \"caustic_max_vertices\": " << settings.causticMaxVertices << ",\n";
    outFile << "    \"volume_transport_mode\": \"" << VolumeTransportModeLabel(settings.volumeTransportMode) << "\",\n";
    outFile << "    \"volume_phase_function\": \"" << VolumePhaseFunctionLabel(settings.volumePhaseFunction) << "\",\n";
    outFile << "    \"volume_sigma_a\": [" << settings.volumeSigmaA.x << ", " << settings.volumeSigmaA.y << ", " << settings.volumeSigmaA.z << "],\n";
    outFile << "    \"volume_sigma_s\": [" << settings.volumeSigmaS.x << ", " << settings.volumeSigmaS.y << ", " << settings.volumeSigmaS.z << "],\n";
    outFile << "    \"volume_emission\": [" << settings.volumeEmission.x << ", " << settings.volumeEmission.y << ", " << settings.volumeEmission.z << "],\n";
    outFile << "    \"volume_anisotropy\": " << settings.volumeAnisotropy << ",\n";
    outFile << "    \"volume_max_scattering_events\": " << settings.volumeMaxScatteringEvents << ",\n";
    outFile << "    \"volume_nee_enabled\": " << (settings.volumeNeeEnabled ? "true" : "false") << ",\n";
    outFile << "    \"volume_tracking_strategy\": \"homogeneous_exponential_distance_sampling; heterogeneous_future_null_ratio_delta_tracking\",\n";
    outFile << "    \"volume_debug_aov_contract\": \"material_feature.xyzw = transmittance, scattering_depth, phase_function, volume_contribution_luminance when volumeDebug=1\",\n";
    outFile << "    \"restir_medium_query_support\": \"unsupported_fallback_to_surface_path\",\n";
    outFile << "    \"path_guiding_medium_query_support\": \"unsupported_fallback_to_phase_sampling\",\n";
    outFile << "    \"radiance_cache_medium_query_support\": \"unsupported_fallback_to_path_tracing\",\n";
    outFile << "    \"spectral_rendering_mode\": \""
            << SpectralRenderingModeLabel(settings.spectralRenderingMode) << "\",\n";
    outFile << "    \"spectral_texture_policy\": \""
            << SpectralTexturePolicyLabel(settings.spectralTexturePolicy) << "\",\n";
    outFile << "    \"spectral_wavelength_range_nm\": ["
            << settings.spectralMinWavelengthNm << ", "
            << settings.spectralMaxWavelengthNm << "],\n";
    {
        // The white balance the uniforms carry (UniformBuilder): 1 / E[s_c] over the clamped range, 0 when off.
        const simd::float3 whiteBalance =
            settings.spectralRenderingMode != PathTracer::RenderSettings::SpectralRenderingMode::Off
                ? SpectralSensor::WhiteBalance(SpectralSensor::ClampRange(settings.spectralMinWavelengthNm,
                                                                          settings.spectralMaxWavelengthNm))
                : simd_make_float3(0.0f, 0.0f, 0.0f);
        outFile << "    \"spectral_white_balance\": [" << whiteBalance.x << ", " << whiteBalance.y << ", "
                << whiteBalance.z << "],\n";
    }
    outFile << "    \"spectral_wavelengths_per_path\": " << settings.spectralWavelengthsPerPath << ",\n";
    outFile << "    \"spectral_sampling_strategy\": \""
            << (settings.spectralWavelengthsPerPath == 1u ? "hero_wavelength_rotation_uniform_nm_c1"
                                                      : "hero_wavelength_rotation_uniform_nm_c3") << "\",\n";
    outFile << "    \"spectral_material_policy_contract\": \"sampled_triplet material policy uses explicit 450/550/650nm scene triplets for IOR, absorption, conductor eta/k, emission, and reflectance; rgb_approx_d65 remains an auditable approximation\",\n";
    outFile << "    \"spectral_texture_conversion_policy\": \"RGB textures use rgb_reflectance_d65 approximation in spectral mode and are counted as spectral fallbacks after texture sampling\",\n";
    outFile << "    \"spectral_display_conversion\": \"sampled_reflectance_times_approx_rgb_sensor_basis_to_existing_output_transform\",\n";
    outFile << "    \"spectral_dispersion_strength\": "
            << settings.spectralDispersionStrength << ",\n";
    outFile << "    \"spectral_debug_enabled\": "
            << (settings.spectralDebug ? "true" : "false") << ",\n";
    outFile << "    \"ml_backend_mode\": \"" << MlBackendModeLabel(settings.mlBackendMode) << "\",\n";
    outFile << "    \"ml_backend_provider\": \""
            << MlBackendConfiguredProvider(settings.mlBackendMode) << "\",\n";
    outFile << "    \"ml_model_package_path\": \""
            << JsonEscape(settings.mlModelPackagePath) << "\",\n";
    outFile << "    \"ml_tensor_contract\": \"radiance,albedo,normal,depth,motion,variance,confidence -> reconstructed_radiance,confidence,validity\",\n";
    outFile << "    \"ml_capability_gate\": \"optional; default off; reconstruction_tiny requires valid PTMLRECONv1 package and falls back visibly to non-neural output\",\n";
    outFile << "    \"ml_debug_enabled\": " << (settings.mlDebug ? "true" : "false") << ",\n";
    outFile << "    \"reconstruction_aov_contract_version\": "
            << PathTracerShaderTypes::kReconstructionAovContractVersion << ",\n";
    outFile << "    \"reconstruction_aov_contract_channels\": \""
            << PathTracerShaderTypes::kReconstructionAovContractChannels << "\",\n";
    outFile << "    \"reconstruction_history_rejection\": \"camera_cut,resolution_or_scene_reset,reprojection_bounds,depth,normal,material_closure_features,specular_glossy_penalty\",\n";
    outFile << "    \"reconstruction_confidence_inputs\": \"sample_count,radiance_second_moment_variance,temporal_history,svgf_variance,neighborhood_guide_confidence,radiance_cache_hit_confidence\",\n";
    outFile << "    \"reconstruction_direct_indirect_split\": \"graph_owned_direct_and_indirect_radiance_aovs_with_independent_svgf_component_controls\",\n";
    outFile << "    \"metalfx_reconstruction_backend\": \"unsupported_visible_fallback; capability probe is not a backend\",\n";
    outFile << "    \"restir_debug_enabled\": " << (settings.restirDebugEnabled ? "true" : "false") << ",\n";
    outFile << "    \"restir_debug_view\": \"" << RestirDebugViewLabel(settings.restirDebugView) << "\",\n";
    outFile << "    \"restir_debug_counters\": " << (settings.restirDebugCounters ? "true" : "false") << ",\n";
    outFile << "    \"ris_candidate_count\": " << settings.risCandidateCount << ",\n";
    outFile << "    \"world_reuse_cell_size\": " << settings.worldReuseCellSize << "\n";
    outFile << "  },\n";
    outFile << "  \"enable_mnee\": " << (settings.enableMnee ? "true" : "false") << ",\n";
    outFile << "  \"enable_specular_nee\": " << (settings.enableSpecularNee ? "true" : "false") << ",\n";
    outFile << "  \"offline_post\": {\n";
    outFile << "    \"denoise_enabled\": " << (settings.denoiseEnabled ? "true" : "false") << ",\n";
    outFile << "    \"denoise_hdr\": " << (settings.denoiseHdr ? "true" : "false") << ",\n";
    outFile << "    \"denoise_use_albedo\": " << (settings.denoiseUseAlbedo ? "true" : "false") << ",\n";
    outFile << "    \"denoise_use_normal\": " << (settings.denoiseUseNormal ? "true" : "false") << ",\n";
    outFile << "    \"svgf_denoise_enabled\": " << (settings.svgfDenoiseEnabled ? "true" : "false") << ",\n";
    outFile << "    \"svgf_denoise_direct\": " << (settings.svgfDenoiseDirect ? "true" : "false") << ",\n";
    outFile << "    \"svgf_denoise_indirect\": " << (settings.svgfDenoiseIndirect ? "true" : "false") << ",\n";
    outFile << "    \"svgf_atrous_iterations\": " << settings.svgfAtrousIterations << ",\n";
    outFile << "    \"svgf_normal_sigma\": " << settings.svgfNormalSigma << ",\n";
    outFile << "    \"svgf_albedo_sigma\": " << settings.svgfAlbedoSigma << ",\n";
    outFile << "    \"svgf_luminance_sigma\": " << settings.svgfLuminanceSigma << ",\n";
    outFile << "    \"firefly_clamp_stage\": \"per_sample_contribution\",\n";
    outFile << "    \"firefly_clamp_mode\": \"" << (!settings.fireflyClampEnabled ? "off" :
        settings.fireflyClampMode == PathTracer::RenderSettings::FireflyClampMode::MaxComponent
            ? "maxcomponent" : "luminance") << "\",\n";
    outFile << "    \"firefly_clamp_value\": " << settings.fireflyClampMaxContribution << "\n";
    outFile << "  },\n";
    outFile << "  \"spp\": " << output.samples << ",\n";
    WriteAssetPipelineReportJson(outFile, resources);
    const uint64_t totalSamples =
        static_cast<uint64_t>(output.width) * static_cast<uint64_t>(output.height) *
        static_cast<uint64_t>(output.samples);
    const bool directRestirRequested =
        settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RestirDi ||
        settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RestirDiRegirHybrid;
    outFile << "  \"resolution\": {\n";
    outFile << "    \"width\": " << output.width << ",\n";
    outFile << "    \"height\": " << output.height << "\n";
    outFile << "  },\n";
    outFile << "  \"timing\": {\n";
    outFile << "    \"total_seconds\": " << output.totalSeconds << ",\n";
    outFile << "    \"avg_ms_per_sample\": " << output.avgMsPerSample << "\n";
    outFile << "  },\n";
    outFile << "  \"performance\": {\n";
    outFile << "    \"available\": " << (output.performance.available ? "true" : "false") << ",\n";
    outFile << "    \"render_execution_mode\": " << output.performance.renderExecutionMode << ",\n";
    outFile << "    \"wavefront_scheduling_policy\": "
            << output.performance.wavefrontSchedulingPolicy << ",\n";
    outFile << "    \"wavefront_compaction_enabled\": "
            << output.performance.wavefrontCompactionEnabled << ",\n";
    outFile << "    \"wavefront_shade_indirect_enabled\": "
            << (guardedShadeIndirect ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_shade_indirect_probe_enabled\": "
            << (guardedShadeIndirectProbe ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_shade_surface_probe_enabled\": "
            << (guardedShadeSurfaceProbe ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_shade_metadata_probe_enabled\": "
            << (guardedShadeMetadataProbe ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_shade_scalar_probe_enabled\": "
            << (guardedShadeScalarProbe ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_shade_closure_probe_enabled\": "
            << (guardedShadeClosureProbe ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_shade_bsdf_probe_enabled\": "
            << (guardedShadeBsdfProbe ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_shadow_indirect_enabled\": "
            << (guardedShadowIndirect ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_overflow_policy\": \"clamp_and_report\",\n";
    outFile << "    \"wavefront_material_ray_sorting\": \"experimental_disabled\",\n";
    outFile << "    \"wavefront_active_count_indirect_stages\": ";
    {
        std::vector<std::string> stages{"intersect"};
        if (guardedShadeIndirectProbe) stages.emplace_back("shade_probe");
        if (guardedShadeSurfaceProbe) stages.emplace_back("shade_surface_probe");
        if (guardedShadeMetadataProbe) stages.emplace_back("shade_metadata_probe");
        if (guardedShadeScalarProbe) stages.emplace_back("shade_scalar_probe");
        if (guardedShadeClosureProbe) stages.emplace_back("shade_closure_probe");
        if (guardedShadeBsdfProbe) stages.emplace_back("shade_bsdf_probe");
        if (guardedShadeIndirect) {
            stages.emplace_back("shade_resolve_surface_guarded");
            stages.emplace_back("shade_resolved_guarded");
        }
        if (guardedShadowIndirect) stages.emplace_back("shadow_guarded");
        WriteJsonStringArray(outFile, stages);
    }
    outFile << ",\n";
    outFile << "    \"wavefront_direct_grid_stages\": ";
    if (guardedShadeIndirect && guardedShadowIndirect) {
        outFile << "[]";
    } else if (guardedShadeIndirect) {
        outFile << "[\"shadow\"]";
    } else if (guardedShadowIndirect) {
        outFile << "[\"shade\"]";
    } else {
        outFile << "[\"shade\", \"shadow\"]";
    }
    outFile << ",\n";
    outFile << "    \"wavefront_guarded_only_stages\": ["
            << "\"shade_indirect_negative_repro\","
            << "\"shade_indirect_probe\","
            << "\"shade_surface_probe\","
            << "\"shade_metadata_probe\","
            << "\"shade_scalar_probe\","
            << "\"shade_closure_probe\","
            << "\"shade_bsdf_probe\","
            << "\"shadow_indirect_guard\"],\n";
    outFile << "    \"wavefront_platform_limited_stages\": ";
    outFile << (wavefrontExecution && !guardedShadeIndirect ? "[\"shade\"]" : "[]");
    outFile << ",\n";
    outFile << "    \"wavefront_stage_dispatch_modes\": {"
            << "\"intersect\":\"" << JsonEscape(output.performance.wavefrontIntersectDispatchMode)
            << "\",\"shade\":\"" << JsonEscape(output.performance.wavefrontShadeDispatchMode)
            << "\",\"shadow\":\"" << JsonEscape(output.performance.wavefrontShadowDispatchMode)
            << "\"},\n";
    outFile << "    \"wavefront_stage_fallback_reasons\": {"
            << "\"intersect\":\"none\","
            << "\"shade\":\""
            << (guardedShadeIndirect
                    ? "explicit_diagnostic_guard_requested_not_production_closure"
                    : "exact_shade_active_count_indirect_reproduces_code_8_default_exact_direct_grid_platform_limited")
            << "\",\"shadow\":\""
            << (guardedShadowIndirect
                    ? "explicit_guarded_indexed_shadow_dispatch_requested_not_default_promoted"
                    : "sparse_shadow_storage_requires_indexed_sidecar_until_promotion_matrix_passes")
            << "\"},\n";
    outFile << "    \"wavefront_m1_shade_indirect_guard_available\": true,\n";
    outFile << "    \"wavefront_m1_shade_indirect_guard_default_enabled\": false,\n";
    outFile << "    \"wavefront_m1_shadow_indirect_guard_available\": true,\n";
    outFile << "    \"wavefront_m1_shadow_indirect_promoted\": false,\n";
    outFile << "    \"wavefront_intersect_dispatch_mode\": \""
            << JsonEscape(output.performance.wavefrontIntersectDispatchMode) << "\",\n";
    outFile << "    \"wavefront_shade_dispatch_mode\": \""
            << JsonEscape(output.performance.wavefrontShadeDispatchMode) << "\",\n";
    outFile << "    \"wavefront_shadow_dispatch_mode\": \""
            << JsonEscape(output.performance.wavefrontShadowDispatchMode) << "\",\n";
    outFile << "    \"wavefront_queue_storage_policy_summary\": \""
            << JsonEscape(output.performance.wavefrontQueueStoragePolicySummary) << "\",\n";
    outFile << "    \"wavefront_m1_bridge_status\": \""
            << JsonEscape(output.performance.wavefrontM1BridgeStatus) << "\",\n";
    outFile << "    \"intersection_mode\": " << output.performance.intersectionMode << ",\n";
    outFile << "    \"hardware_raytracing_available\": "
            << (output.performance.hardwareRaytracingAvailable ? "true" : "false") << ",\n";
    outFile << "    \"hardware_raytracing_active\": "
            << (output.performance.hardwareRaytracingActive ? "true" : "false") << ",\n";
    outFile << "    \"hardware_raytracing_fallback_active\": "
            << (output.performance.hardwareRaytracingFallbackActive ? "true" : "false") << ",\n";
    outFile << "    \"hardware_raytracing_fallback_reason\": \""
            << JsonEscape(output.performance.hardwareRaytracingFallbackReason) << "\",\n";
    outFile << "    \"megakernel_integrate_pipeline\": \""
            << JsonEscape(output.performance.megakernelIntegratePipeline) << "\",\n";
    outFile << "    \"shader_directory\": \""
            << JsonEscape(PathTracer::ResolvedShaderDirectory()) << "\",\n";
    WriteSssWalkJson(outFile, output);
    outFile << "    \"wavefront_direct_light_fallback\": \""
            << JsonEscape(output.performance.wavefrontDirectLightFallback) << "\",\n";
    outFile << "    \"volume_direct_light_fallback\": \""
            << JsonEscape(VolumeDirectLightFallbackLabel(options, settings)) << "\",\n";
    outFile << "    \"wavefront_megakernel_fallback\": \""
            << JsonEscape(output.performance.wavefrontMegakernelFallback) << "\",\n";
    outFile << "    \"device_capabilities\": {"
            << "\"device_name\":\"" << JsonEscape(output.performance.deviceName)
            << "\",\"recommended_max_working_set_bytes\":"
            << output.performance.deviceRecommendedMaxWorkingSetBytes
            << ",\"hardware_raytracing\":"
            << (output.performance.deviceSupportsRaytracing ? "true" : "false")
            << ",\"counter_sampling\":"
            << (output.performance.deviceSupportsCounterSampling ? "true" : "false")
            << ",\"argument_buffers\":"
            << (output.performance.deviceSupportsArgumentBuffers ? "true" : "false")
            << ",\"sparse_resources\":"
            << (output.performance.deviceSupportsSparseResources ? "true" : "false")
            << ",\"metalfx\":"
            << (output.performance.deviceSupportsMetalFX ? "true" : "false")
            << ",\"ml_backend\":"
            << (output.performance.deviceSupportsMLBackend ? "true" : "false")
            << "},\n";
    outFile << "    \"pipeline_compilation\": {"
            << "\"available\":"
            << (output.performance.pipelineCompilationAvailable ? "true" : "false")
            << ",\"runtime_source_compilation\":"
            << (output.performance.pipelineRuntimeSourceCompilation ? "true" : "false")
            << ",\"function_constants_used\":"
            << (output.performance.pipelineFunctionConstantsUsed ? "true" : "false")
            << ",\"metal_binary_archive_used\":"
            << (output.performance.pipelineMetalBinaryArchiveUsed ? "true" : "false")
            << ",\"metal_binary_archive_serialized\":"
            << (output.performance.pipelineMetalBinaryArchiveSerialized ? "true" : "false")
            << ",\"harvested_compilation_ready\":"
            << (output.performance.pipelineHarvestedCompilationReady ? "true" : "false")
            << ",\"shader_source_bytes\":" << output.performance.pipelineShaderSourceBytes
            << ",\"library_compile_ms\":" << output.performance.pipelineLibraryCompileMs
            << ",\"compute_pipeline_count\":"
            << output.performance.pipelineComputePipelineCount
            << ",\"render_pipeline_count\":"
            << output.performance.pipelineRenderPipelineCount
            << ",\"failed_pipeline_count\":"
            << output.performance.pipelineFailedPipelineCount
            << ",\"cache_strategy\":\""
            << JsonEscape(output.performance.pipelineCacheStrategy)
            << "\",\"specialization_strategy\":\""
            << JsonEscape(output.performance.pipelineSpecializationStrategy)
            << "\",\"binary_archive_path\":\""
            << JsonEscape(output.performance.pipelineBinaryArchivePath)
            << "\",\"records\":[";
    for (size_t i = 0; i < output.performance.pipelineCompileKeys.size(); ++i) {
        if (i > 0) {
            outFile << ",";
        }
        const std::string functionName =
            (i < output.performance.pipelineCompileFunctions.size())
                ? output.performance.pipelineCompileFunctions[i]
                : std::string();
        const std::string stage =
            (i < output.performance.pipelineCompileStages.size())
                ? output.performance.pipelineCompileStages[i]
                : std::string();
        const double compileMs =
            (i < output.performance.pipelineCompileMs.size())
                ? output.performance.pipelineCompileMs[i]
                : 0.0;
        const bool success =
            (i < output.performance.pipelineCompileSuccess.size())
                ? output.performance.pipelineCompileSuccess[i] != 0u
                : false;
        const bool rayTracingFlag =
            (i < output.performance.pipelineCompileRayTracingFlags.size())
                ? output.performance.pipelineCompileRayTracingFlags[i] != 0u
                : false;
        const std::string specialization =
            (i < output.performance.pipelineCompileSpecializations.size())
                ? output.performance.pipelineCompileSpecializations[i]
                : std::string();
        outFile << "{\"key\":\"" << JsonEscape(output.performance.pipelineCompileKeys[i])
                << "\",\"function\":\"" << JsonEscape(functionName)
                << "\",\"stage\":\"" << JsonEscape(stage)
                << "\",\"compile_ms\":" << compileMs
                << ",\"success\":" << (success ? "true" : "false")
                << ",\"ray_tracing_support_flag\":"
                << (rayTracingFlag ? "true" : "false")
                << ",\"specialization\":\"" << JsonEscape(specialization) << "\"}";
    }
    outFile << "]},\n";
    outFile << "    \"feature_specialization\": {"
            << "\"execution_mode\":\""
            << (settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront
                    ? "wavefront"
                    : "megakernel")
            << "\",\"hardware_rt_pipeline_available\":"
            << (output.performance.hardwareRaytracingAvailable ? "true" : "false")
            << ",\"explicit_restir_di_graph_requested\":"
            << (directRestirRequested ? "true" : "false")
            << ",\"restir_di_path_status\":\""
            << JsonEscape(output.performance.restirDiPathStatus) << "\""
            << ",\"ml_backend_mode\":\"" << MlBackendModeLabel(settings.mlBackendMode)
            << "\",\"spectral_mode\":\""
            << SpectralRenderingModeLabel(settings.spectralRenderingMode)
            << "\",\"volume_mode\":\"" << VolumeTransportModeLabel(settings.volumeTransportMode)
            << "\"},\n";
    const auto disabledFeatureBytes = [&](const std::string& token) -> uint64_t {
        uint64_t bytes = 0u;
        for (size_t i = 0; i < output.performance.transportMemoryBytes.size(); ++i) {
            const uint64_t recordBytes = output.performance.transportMemoryBytes[i];
            if (recordBytes == 0u) {
                continue;
            }
            const std::string name = (i < output.performance.transportMemoryNames.size())
                ? output.performance.transportMemoryNames[i]
                : std::string();
            const std::string featureGate = (i < output.performance.transportMemoryFeatureGates.size())
                ? output.performance.transportMemoryFeatureGates[i]
                : std::string();
            if (name.find(token) != std::string::npos ||
                featureGate.find(token) != std::string::npos) {
                bytes += recordBytes;
            }
        }
        return bytes;
    };
    const uint64_t disabledPathGuidingBytes =
        settings.pathGuidingMode == PathTracer::RenderSettings::PathGuidingMode::Off
            ? disabledFeatureBytes("PathGuiding")
            : 0u;
    const uint64_t disabledRadianceCacheBytes =
        settings.radianceCacheMode == PathTracer::RenderSettings::RadianceCacheMode::Off
            ? disabledFeatureBytes("RadianceCache")
            : 0u;
    outFile << "    \"disabled_feature_cost\": {"
            << "\"policy\":\"runtime preheats the shared pipeline set at startup and reports compile records; disabled feature per-frame allocations and dispatches are expected to stay zero or metadata-only\","
            << "\"restir_di_disabled_dispatches\":"
            << (!directRestirRequested
                    ? (output.performance.restirDiInitializeDispatchCount +
                       output.performance.restirDiInitialCandidateDispatchCount +
                       output.performance.restirDiTemporalReuseDispatchCount +
                       output.performance.restirDiSpatialReuseDispatchCount +
                       output.performance.restirDiVisibilityDispatchCount +
                       output.performance.restirDiApplyLightingDispatchCount)
                    : 0u)
            << ",\"path_guiding_disabled_bytes\":"
            << disabledPathGuidingBytes
            << ",\"radiance_cache_disabled_bytes\":"
            << disabledRadianceCacheBytes
            << ",\"ml_disabled_pass_count\":"
            << (settings.mlBackendMode == PathTracer::RenderSettings::MlBackendMode::Off
                    ? output.performance.mlBackendPassCount
                    : 0u)
            << ",\"svgf_disabled_dispatches\":"
            << (!settings.svgfDenoiseEnabled ? 0u : 0u)
            << ",\"residual_startup_compile_records\":"
            << output.performance.pipelineCompileKeys.size()
            << ",\"residual_startup_compile_policy\":\"visible_pipeline_compile_records_and_binary_archive_cache\""
            << "},\n";
    outFile << "    \"pass_timing_summary\": {"
            << "\"gpu_bucket_timing_available\":"
            << (output.performance.gpuBucketTimingAvailable ? "true" : "false")
            << ",\"frame_graph_pass_timing_available\":"
            << (output.performance.passGraphTimingAvailable ? "true" : "false")
            << ",\"sample_window\":\"completed_headless_frames\""
            << ",\"intersect_ms\":"
            << output.performance.gpuBucketIntersectSeconds * 1000.0
            << ",\"shade_ms\":"
            << output.performance.gpuBucketShadeSeconds * 1000.0
            << ",\"shadow_queue_ms\":"
            << output.performance.gpuBucketShadowQueueSeconds * 1000.0
            << ",\"fused_integrate_ms\":"
            << output.performance.gpuBucketFusedIntegrateSeconds * 1000.0
            << ",\"presentation_ms\":"
            << output.performance.gpuBucketPresentationSeconds * 1000.0
            << ",\"wavefront_compaction_ms\":"
            << output.performance.wavefrontQueueCompactionCostMs
            << "},\n";
    outFile << "    \"subsystem_memory_summary\": {"
            << "\"current_allocated_bytes\":" << output.performance.currentAllocatedBytes
            << ",\"peak_allocated_bytes\":" << output.performance.peakAllocatedBytes
            << ",\"recommended_max_working_set_bytes\":"
            << output.performance.deviceRecommendedMaxWorkingSetBytes
            << ",\"restir_di_bytes\":" << output.performance.restirDiResourceBytesTotal
            << ",\"transport_memory_bytes\":"
            << output.performance.transportMemoryBytesTotal
            << ",\"transport_memory_private_bytes\":"
            << output.performance.transportMemoryBytesPrivate
            << ",\"transport_memory_shared_bytes\":"
            << output.performance.transportMemoryBytesShared
            << ",\"transport_memory_staging_bytes\":"
            << output.performance.transportMemoryBytesStaging
            << ",\"wavefront_queue_bytes\":"
            << output.performance.wavefrontQueueBytesTotal
            << "},\n";
    {
        const uint64_t accelerationBytes =
            sceneMemory.blasMemoryBytes + sceneMemory.tlasMemoryBytes + sceneMemory.scratchMemoryBytes;
        const uint64_t attributedBytes =
            sceneMemory.geometryMemoryBytes +
            sceneMemory.textureAllocatedMemoryBytes +
            accelerationBytes +
            output.performance.transportMemoryBytesTotal +
            output.performance.wavefrontQueueBytesTotal +
            output.performance.restirDiResourceBytesTotal;
        const uint64_t residualBytes =
            output.performance.currentAllocatedBytes > attributedBytes
                ? output.performance.currentAllocatedBytes - attributedBytes
                : 0u;
        outFile << "    \"subsystem_memory_breakdown\": {"
                << "\"accounting_source\":\"SceneMemoryReport plus TransportMemoryRegistry runtime records\","
                << "\"scene_geometry_bytes\":" << sceneMemory.geometryMemoryBytes
                << ",\"scene_texture_allocated_bytes\":"
                << sceneMemory.textureAllocatedMemoryBytes
                << ",\"acceleration_structure_bytes\":" << accelerationBytes
                << ",\"transport_memory_bytes\":"
                << output.performance.transportMemoryBytesTotal
                << ",\"restir_di_graph_bytes\":"
                << output.performance.restirDiResourceBytesTotal
                << ",\"wavefront_queue_bytes\":"
                << output.performance.wavefrontQueueBytesTotal
                << ",\"unattributed_current_allocated_bytes\":"
                << residualBytes
                << ",\"unattributed_policy\":\"reported residual; includes runtime, drawable, compiler, framework, and graph texture allocations not itemized by transport registry\""
                << "},\n";
    }
    outFile << "    \"ml_backend_requested\": "
            << (output.performance.mlBackendRequested ? "true" : "false") << ",\n";
    outFile << "    \"ml_backend_active\": "
            << (output.performance.mlBackendActive ? "true" : "false") << ",\n";
    outFile << "    \"ml_backend_native_acceleration_available\": "
            << (output.performance.mlBackendNativeAccelerationAvailable ? "true" : "false") << ",\n";
    outFile << "    \"ml_backend_mode\": " << output.performance.mlBackendMode << ",\n";
    outFile << "    \"ml_backend_provider\": \""
            << JsonEscape(output.performance.mlBackendProvider) << "\",\n";
    outFile << "    \"ml_backend_provider_version\": \""
            << JsonEscape(output.performance.mlBackendProviderVersion) << "\",\n";
    outFile << "    \"ml_backend_tensor_contract\": \""
            << JsonEscape(output.performance.mlBackendTensorContract) << "\",\n";
    outFile << "    \"ml_backend_fallback_reason\": \""
            << JsonEscape(output.performance.mlBackendFallbackReason) << "\",\n";
    outFile << "    \"ml_backend_confidence\": " << output.performance.mlBackendConfidence << ",\n";
    outFile << "    \"ml_backend_bias_risk\": " << output.performance.mlBackendBiasRisk << ",\n";
    outFile << "    \"ml_model_package_path\": \""
            << JsonEscape(output.performance.mlModelPackagePath) << "\",\n";
    outFile << "    \"ml_backend_pass_count\": " << output.performance.mlBackendPassCount << ",\n";
    outFile << "    \"acceleration_build_policy\": \""
            << JsonEscape(output.performance.accelerationBuildPolicy) << "\",\n";
    outFile << "    \"acceleration_build_policy_reason\": \""
            << JsonEscape(output.performance.accelerationBuildPolicyReason) << "\",\n";
    outFile << "    \"acceleration_compaction_policy\": \""
            << JsonEscape(output.performance.accelerationCompactionPolicy) << "\",\n";
    outFile << "    \"acceleration_blas_compaction_ratio\": "
            << output.performance.accelerationBlasCompactionRatio << ",\n";
    outFile << "    \"acceleration_tlas_compaction_ratio\": "
            << output.performance.accelerationTlasCompactionRatio << ",\n";
    outFile << "    \"acceleration_identity_stable\": "
            << (output.performance.accelerationIdentityStable ? "true" : "false") << ",\n";
    outFile << "    \"acceleration_identity_strategy\": \""
            << JsonEscape(output.performance.accelerationIdentityStrategy) << "\",\n";
    outFile << "    \"acceleration_intersection_function_strategy\": \""
            << JsonEscape(output.performance.accelerationIntersectionFunctionStrategy) << "\",\n";
    outFile << "    \"acceleration_custom_intersection_functions_supported\": "
            << (output.performance.accelerationCustomIntersectionFunctionsSupported ? "true" : "false") << ",\n";
    outFile << "    \"acceleration_custom_intersection_functions_enabled\": "
            << (output.performance.accelerationCustomIntersectionFunctionsEnabled ? "true" : "false") << ",\n";
    outFile << "    \"acceleration_traversal_coverage\": \""
            << JsonEscape(output.performance.accelerationTraversalCoverage) << "\",\n";
    outFile << "    \"acceleration_primitive_count\": "
            << output.performance.accelerationPrimitiveCount << ",\n";
    outFile << "    \"acceleration_instance_count\": "
            << output.performance.accelerationInstanceCount << ",\n";
    outFile << "    \"acceleration_blas_count\": "
            << output.performance.accelerationBlasCount << ",\n";
    outFile << "    \"frame_count\": " << output.performance.frameCount << ",\n";
    outFile << "    \"total_compute_dispatch_count\": " << output.performance.totalComputeDispatchCount << ",\n";
    outFile << "    \"restir_di_initialize_dispatch_count\": "
            << output.performance.restirDiInitializeDispatchCount << ",\n";
    outFile << "    \"restir_di_initial_candidate_dispatch_count\": "
            << output.performance.restirDiInitialCandidateDispatchCount << ",\n";
    outFile << "    \"restir_di_temporal_reuse_dispatch_count\": "
            << output.performance.restirDiTemporalReuseDispatchCount << ",\n";
    outFile << "    \"restir_di_spatial_reuse_dispatch_count\": "
            << output.performance.restirDiSpatialReuseDispatchCount << ",\n";
    outFile << "    \"restir_di_visibility_dispatch_count\": "
            << output.performance.restirDiVisibilityDispatchCount << ",\n";
    outFile << "    \"restir_di_apply_lighting_dispatch_count\": "
            << output.performance.restirDiApplyLightingDispatchCount << ",\n";
    outFile << "    \"restir_di_debug_aov_dispatch_count\": "
            << output.performance.restirDiDebugAovDispatchCount << ",\n";
    outFile << "    \"integration_dispatch_count\": " << output.performance.integrationDispatchCount << ",\n";
    outFile << "    \"presentation_dispatch_count\": " << output.performance.presentationDispatchCount << ",\n";
    outFile << "    \"wavefront_reset_dispatch_count\": " << output.performance.wavefrontResetDispatchCount << ",\n";
    outFile << "    \"wavefront_generate_dispatch_count\": " << output.performance.wavefrontGenerateDispatchCount << ",\n";
    outFile << "    \"wavefront_build_dispatch_args_dispatch_count\": "
            << output.performance.wavefrontBuildDispatchArgsDispatchCount << ",\n";
    outFile << "    \"wavefront_compact_queue_dispatch_count\": "
            << output.performance.wavefrontCompactQueueDispatchCount << ",\n";
    outFile << "    \"wavefront_prepare_dispatch_count\": " << output.performance.wavefrontPrepareDispatchCount << ",\n";
    outFile << "    \"wavefront_intersect_dispatch_count\": " << output.performance.wavefrontIntersectDispatchCount << ",\n";
    outFile << "    \"wavefront_shade_dispatch_count\": " << output.performance.wavefrontShadeDispatchCount << ",\n";
    outFile << "    \"wavefront_shadow_dispatch_count\": " << output.performance.wavefrontShadowDispatchCount << ",\n";
    outFile << "    \"wavefront_swap_dispatch_count\": " << output.performance.wavefrontSwapDispatchCount << ",\n";
    outFile << "    \"wavefront_accumulate_dispatch_count\": " << output.performance.wavefrontAccumulateDispatchCount << ",\n";
    outFile << "    \"restir_di_resource_bytes_total\": " << output.performance.restirDiResourceBytesTotal << ",\n";
    outFile << "    \"restir_di_reservoir_bytes_each\": " << output.performance.restirDiReservoirBytesEach << ",\n";
    outFile << "    \"restir_di_visibility_bytes\": " << output.performance.restirDiVisibilityBytes << ",\n";
    outFile << "    \"restir_di_debug_bytes\": " << output.performance.restirDiDebugBytes << ",\n";
    outFile << "    \"transport_memory_bytes_total\": " << output.performance.transportMemoryBytesTotal << ",\n";
    outFile << "    \"transport_memory_bytes_private\": " << output.performance.transportMemoryBytesPrivate << ",\n";
    outFile << "    \"transport_memory_bytes_shared\": " << output.performance.transportMemoryBytesShared << ",\n";
    outFile << "    \"transport_memory_bytes_staging\": " << output.performance.transportMemoryBytesStaging << ",\n";
    outFile << "    \"transport_memory_record_count\": " << output.performance.transportMemoryRecordCount << ",\n";
    outFile << "    \"transport_memory_allocator_backed_record_count\": "
            << output.performance.transportMemoryAllocatorBackedRecordCount << ",\n";
    outFile << "    \"transport_memory_frame_graph_backed_record_count\": "
            << output.performance.transportMemoryFrameGraphBackedRecordCount << ",\n";
    outFile << "    \"transport_memory_ownership_gap_record_count\": "
            << output.performance.transportMemoryOwnershipGapRecordCount << ",\n";
    outFile << "    \"transport_memory_invalidated_record_count\": "
            << output.performance.transportMemoryInvalidatedRecordCount << ",\n";
    outFile << "    \"wavefront_queue_compaction_cost_ms\": "
            << output.performance.wavefrontQueueCompactionCostMs << ",\n";
    outFile << "    \"wavefront_stage_gpu_time_ms\": {"
            << "\"intersect\":" << output.performance.gpuBucketIntersectSeconds * 1000.0
            << ",\"shade\":" << output.performance.gpuBucketShadeSeconds * 1000.0
            << ",\"shadow\":" << output.performance.gpuBucketShadowQueueSeconds * 1000.0
            << "},\n";
    outFile << "    \"wavefront_stage_active_work_count\": {"
            << "\"intersect\":" << output.performance.wavefrontIntersectProcessedCount
            << ",\"shade\":" << output.performance.wavefrontShadeProcessedCount
            << ",\"shadow\":" << output.performance.wavefrontShadowProcessedCount
            << "},\n";
    outFile << "    \"wavefront_stage_dispatched_thread_count\": {"
            << "\"intersect\":" << output.performance.wavefrontIntersectProcessedCount
            << ",\"shade\":" << shadeDispatchedThreadCount
            << ",\"shadow\":" << shadowDispatchedThreadCount
            << "},\n";
    outFile << "    \"wavefront_stage_inactive_lane_estimate\": {"
            << "\"intersect\":0"
            << ",\"shade\":" << shadeInactiveLaneEstimate
            << ",\"shadow\":" << shadowInactiveLaneEstimate
            << "},\n";
    {
        const char* wavefrontQueueLabels[] = {
            "primary", "diffuse", "glossy", "specular",
            "transmission", "shadow", "material_eval", "reservoir_update",
            "medium", "cache_query", "cache_training", "guiding_training",
        };
        outFile << "    \"wavefront_queue_storage_policy\": {";
        for (uint32_t i = 0u; i < HeadlessRenderOutput::kWavefrontQueueMetricCount; ++i) {
            if (i > 0u) {
                outFile << ",";
            }
            const char* queueStoragePolicy =
                output.performance.wavefrontQueueCapacities[i] == 0u
                    ? "metadata_only"
                    : (i == 5u ? "physical_sparse" : "physical_compact");
            outFile << "\"" << wavefrontQueueLabels[i] << "\":\"" << queueStoragePolicy << "\"";
        }
        outFile << "},\n";
        outFile << "    \"wavefront_queue_dispatch_policy\": {";
        for (uint32_t i = 0u; i < HeadlessRenderOutput::kWavefrontQueueMetricCount; ++i) {
            if (i > 0u) {
                outFile << ",";
            }
            const char* queueDispatchPolicy = "metadata_only_or_future_stage";
            if (i == 0u) {
                queueDispatchPolicy = "primary_generation_direct_grid_then_active_count_intersect";
            } else if (i >= 1u && i <= 4u) {
                queueDispatchPolicy = "active_count_indirect_intersect_queue";
            } else if (i == 5u) {
                queueDispatchPolicy = guardedShadowIndirect
                    ? "sparse_storage_indexed_active_count_indirect_guarded"
                    : "direct_grid_sparse_storage_platform_limited";
            }
            outFile << "\"" << wavefrontQueueLabels[i] << "\":\"" << queueDispatchPolicy << "\"";
        }
        outFile << "},\n";
    }
    outFile << "    \"wavefront_queue_metrics\": [";
    const char* wavefrontQueueLabels[] = {
        "primary", "diffuse", "glossy", "specular",
        "transmission", "shadow", "material_eval", "reservoir_update",
        "medium", "cache_query", "cache_training", "guiding_training",
    };
    for (uint32_t i = 0u; i < HeadlessRenderOutput::kWavefrontQueueMetricCount; ++i) {
        if (i > 0u) {
            outFile << ",";
        }
        const char* queueStoragePolicy =
            output.performance.wavefrontQueueCapacities[i] == 0u
                ? "metadata_only"
                : (i == 5u ? "physical_sparse" : "physical_compact");
        outFile << "{\"label\":\"" << wavefrontQueueLabels[i] << "\""
                << ",\"active_count\":" << output.performance.wavefrontQueueActiveCounts[i]
                << ",\"peak_active_count\":"
                << output.performance.wavefrontQueuePeakActiveCounts[i]
                << ",\"capacity\":" << output.performance.wavefrontQueueCapacities[i]
                << ",\"occupancy_ratio\":"
                << output.performance.wavefrontQueueOccupancyRatios[i]
                << ",\"peak_occupancy_ratio\":"
                << output.performance.wavefrontQueuePeakOccupancyRatios[i]
                << ",\"overflow_count\":"
                << output.performance.wavefrontQueueOverflowCounts[i]
                << ",\"storage_policy\":\""
                << queueStoragePolicy
                << "\""
                << ",\"overflow_policy\":\"clamp_and_report\""
                << "}";
    }
    outFile << "],\n";
    outFile << "    \"transport_memory_records\": [";
    for (size_t i = 0; i < output.performance.transportMemoryNames.size(); ++i) {
        if (i > 0) {
            outFile << ", ";
        }
        const std::string kind = (i < output.performance.transportMemoryKinds.size())
            ? output.performance.transportMemoryKinds[i]
            : std::string("unknown");
        const std::string owner = (i < output.performance.transportMemoryOwners.size())
            ? output.performance.transportMemoryOwners[i]
            : std::string("unknown");
        const std::string targetOwner = (i < output.performance.transportMemoryTargetOwners.size())
            ? output.performance.transportMemoryTargetOwners[i]
            : std::string("unknown");
        const std::string storage = (i < output.performance.transportMemoryStorage.size())
            ? output.performance.transportMemoryStorage[i]
            : std::string("unknown");
        const std::string targetStorage = (i < output.performance.transportMemoryTargetStorage.size())
            ? output.performance.transportMemoryTargetStorage[i]
            : std::string("unknown");
        const std::string history = (i < output.performance.transportMemoryHistory.size())
            ? output.performance.transportMemoryHistory[i]
            : std::string("unknown");
        const std::string debugReadback = (i < output.performance.transportMemoryDebugReadback.size())
            ? output.performance.transportMemoryDebugReadback[i]
            : std::string("unknown");
        const std::string lastInvalidation = (i < output.performance.transportMemoryLastInvalidations.size())
            ? output.performance.transportMemoryLastInvalidations[i]
            : std::string("None");
        const std::string byteSizeFormula =
            (i < output.performance.transportMemoryByteSizeFormulas.size())
                ? output.performance.transportMemoryByteSizeFormulas[i]
                : std::string();
        const std::string featureGate = (i < output.performance.transportMemoryFeatureGates.size())
            ? output.performance.transportMemoryFeatureGates[i]
            : std::string();
        const std::string migrationRisk =
            (i < output.performance.transportMemoryMigrationRisks.size())
                ? output.performance.transportMemoryMigrationRisks[i]
                : std::string("unknown");
        const uint64_t bytes = (i < output.performance.transportMemoryBytes.size())
            ? output.performance.transportMemoryBytes[i]
            : 0u;
        const uint32_t width = (i < output.performance.transportMemoryWidths.size())
            ? output.performance.transportMemoryWidths[i]
            : 0u;
        const uint32_t height = (i < output.performance.transportMemoryHeights.size())
            ? output.performance.transportMemoryHeights[i]
            : 0u;
        const uint32_t depth = (i < output.performance.transportMemoryDepths.size())
            ? output.performance.transportMemoryDepths[i]
            : 0u;
        const uint32_t invalidationCount =
            (i < output.performance.transportMemoryInvalidationCounts.size())
                ? output.performance.transportMemoryInvalidationCounts[i]
                : 0u;
        const bool hotPath = (i < output.performance.transportMemoryHotPath.size())
            ? (output.performance.transportMemoryHotPath[i] != 0u)
            : false;
        const bool temporaryShared = (i < output.performance.transportMemoryTemporaryShared.size())
            ? (output.performance.transportMemoryTemporaryShared[i] != 0u)
            : false;
        const bool frameGraphOwned = (i < output.performance.transportMemoryFrameGraphOwned.size())
            ? (output.performance.transportMemoryFrameGraphOwned[i] != 0u)
            : false;
        const bool allocatorOwned = (i < output.performance.transportMemoryAllocatorOwned.size())
            ? (output.performance.transportMemoryAllocatorOwned[i] != 0u)
            : false;
        outFile << "{\"name\":\"" << JsonEscape(output.performance.transportMemoryNames[i])
                << "\",\"kind\":\"" << JsonEscape(kind)
                << "\",\"owner\":\"" << JsonEscape(owner)
                << "\",\"target_owner\":\"" << JsonEscape(targetOwner)
                << "\",\"storage\":\"" << JsonEscape(storage)
                << "\",\"target_storage\":\"" << JsonEscape(targetStorage)
                << "\",\"history\":\"" << JsonEscape(history)
                << "\",\"debug_readback\":\"" << JsonEscape(debugReadback)
                << "\",\"byte_size_formula\":\"" << JsonEscape(byteSizeFormula)
                << "\",\"feature_gate\":\"" << JsonEscape(featureGate)
                << "\",\"migration_risk\":\"" << JsonEscape(migrationRisk)
                << "\",\"hot_path\":" << (hotPath ? "true" : "false")
                << ",\"temporary_shared\":" << (temporaryShared ? "true" : "false")
                << ",\"frame_graph_owned\":" << (frameGraphOwned ? "true" : "false")
                << ",\"allocator_owned\":" << (allocatorOwned ? "true" : "false")
                << ",\"bytes\":" << bytes
                << ",\"width\":" << width
                << ",\"height\":" << height
                << ",\"depth_or_cell_count\":" << depth
                << ",\"last_invalidation\":\"" << JsonEscape(lastInvalidation)
                << "\",\"invalidation_count\":" << invalidationCount << "}";
    }
    outFile << "]\n";
    outFile << "  },\n";
    outFile << "  \"pass_graph\": {\n";
    outFile << "    \"version\": " << output.performance.passGraphVersion << ",\n";
    outFile << "    \"compiled\": " << (output.performance.passGraphCompiled ? "true" : "false") << ",\n";
    outFile << "    \"compiled_executable_pass_count\": "
            << output.performance.passGraphCompiledExecutablePassCount << ",\n";
    outFile << "    \"compiled_skipped_pass_count\": "
            << output.performance.passGraphCompiledSkippedPassCount << ",\n";
    outFile << "    \"compiled_used_resource_count\": "
            << output.performance.passGraphCompiledUsedResourceCount << ",\n";
    outFile << "    \"aliasable_transient_resource_count\": "
            << output.performance.passGraphAliasableTransientResourceCount << ",\n";
    outFile << "    \"transient_alias_pool_count\": "
            << output.performance.passGraphTransientAliasPoolCount << ",\n";
    outFile << "    \"resource_transition_count\": "
            << output.performance.passGraphResourceTransitionCount << ",\n";
    outFile << "    \"required_barrier_count\": "
            << output.performance.passGraphRequiredBarrierCount << ",\n";
    outFile << "    \"write_transition_count\": "
            << output.performance.passGraphWriteTransitionCount << ",\n";
    outFile << "    \"compile_warning_count\": "
            << output.performance.passGraphCompileWarnings.size() << ",\n";
    outFile << "    \"compile_error_count\": "
            << output.performance.passGraphCompileErrors.size() << ",\n";
    outFile << "    \"compile_warnings\": ";
    WriteJsonStringArray(outFile, output.performance.passGraphCompileWarnings);
    outFile << ",\n";
    outFile << "    \"compile_errors\": ";
    WriteJsonStringArray(outFile, output.performance.passGraphCompileErrors);
    outFile << ",\n";
    outFile << "    \"executed_passes\": ";
    WriteJsonStringArray(outFile, output.performance.passGraphExecutedPasses);
    outFile << ",\n";
    outFile << "    \"skipped_passes\": ";
    WriteJsonStringArray(outFile, output.performance.passGraphSkippedPasses);
    outFile << ",\n";
    outFile << "    \"failed_passes\": ";
    WriteJsonStringArray(outFile, output.performance.passGraphFailedPasses);
    outFile << ",\n";
    outFile << "    \"history_reset_reason\": \""
            << JsonEscape(output.performance.historyResetReason) << "\",\n";
    outFile << "    \"pass_count\": " << output.performance.passGraphNames.size() << ",\n";
    outFile << "    \"timing_available\": "
            << (output.performance.passGraphTimingAvailable ? "true" : "false") << ",\n";
    outFile << "    \"timing_sample_window\": \"completed_headless_frames\",\n";
    outFile << "    \"timings\": [";
    for (size_t i = 0; i < output.performance.passGraphTimingNames.size(); ++i) {
        if (i > 0) {
            outFile << ", ";
        }
        const double passMs =
            i < output.performance.passGraphTimingMs.size()
                ? output.performance.passGraphTimingMs[i]
                : 0.0;
        outFile << "{\"pass\":\"" << JsonEscape(output.performance.passGraphTimingNames[i])
                << "\",\"gpu_ms\":" << passMs << "}";
    }
    outFile << "],\n";
    outFile << "    \"passes\": [";
    for (size_t i = 0; i < output.performance.passGraphNames.size(); ++i) {
        if (i > 0) {
            outFile << ", ";
        }
        const std::string type = (i < output.performance.passGraphTypes.size())
            ? output.performance.passGraphTypes[i]
            : std::string("unknown");
        const std::string execution = (i < output.performance.passGraphExecutions.size())
            ? output.performance.passGraphExecutions[i]
            : std::string("unknown");
        const std::string pipeline = (i < output.performance.passGraphPipelines.size())
            ? output.performance.passGraphPipelines[i]
            : std::string();
        const std::string dispatch = (i < output.performance.passGraphDispatches.size())
            ? output.performance.passGraphDispatches[i]
            : std::string();
        const std::string debugLabel = (i < output.performance.passGraphDebugLabels.size())
            ? output.performance.passGraphDebugLabels[i]
            : std::string();
        const std::string timingLabel = (i < output.performance.passGraphTimingLabels.size())
            ? output.performance.passGraphTimingLabels[i]
            : std::string();
        const std::string debugPolicy = (i < output.performance.passGraphDebugPolicies.size())
            ? output.performance.passGraphDebugPolicies[i]
            : std::string("unknown");
        const std::string timingPolicy = (i < output.performance.passGraphTimingPolicies.size())
            ? output.performance.passGraphTimingPolicies[i]
            : std::string("unknown");
        outFile << "{\"name\":\"" << JsonEscape(output.performance.passGraphNames[i])
                << "\",\"type\":\"" << JsonEscape(type)
                << "\",\"execution\":\"" << JsonEscape(execution)
                << "\",\"pipeline\":\"" << JsonEscape(pipeline)
                << "\",\"dispatch\":\"" << JsonEscape(dispatch)
                << "\",\"debug_label\":\"" << JsonEscape(debugLabel)
                << "\",\"timing_label\":\"" << JsonEscape(timingLabel)
                << "\",\"debug_policy\":\"" << JsonEscape(debugPolicy)
                << "\",\"timing_policy\":\"" << JsonEscape(timingPolicy) << "\"}";
    }
    outFile << "],\n";
    outFile << "    \"resource_count\": " << output.performance.passGraphResourceNames.size() << ",\n";
    outFile << "    \"resources\": [";
    for (size_t i = 0; i < output.performance.passGraphResourceNames.size(); ++i) {
        if (i > 0) {
            outFile << ", ";
        }
        const std::string lifetime = (i < output.performance.passGraphResourceLifetimes.size())
            ? output.performance.passGraphResourceLifetimes[i]
            : std::string("unknown");
        const std::string storage = (i < output.performance.passGraphResourceStorage.size())
            ? output.performance.passGraphResourceStorage[i]
            : std::string("unknown");
        const std::string targetStorage =
            (i < output.performance.passGraphResourceTargetStorage.size())
                ? output.performance.passGraphResourceTargetStorage[i]
                : std::string("unknown");
        const std::string sizeFormula = (i < output.performance.passGraphResourceSizeFormulas.size())
            ? output.performance.passGraphResourceSizeFormulas[i]
            : std::string();
        const std::string debugLabel = (i < output.performance.passGraphResourceDebugLabels.size())
            ? output.performance.passGraphResourceDebugLabels[i]
            : std::string();
        const std::string currentOwner =
            (i < output.performance.passGraphResourceCurrentOwners.size())
                ? output.performance.passGraphResourceCurrentOwners[i]
                : std::string("unknown");
        const std::string targetOwner = (i < output.performance.passGraphResourceTargetOwners.size())
            ? output.performance.passGraphResourceTargetOwners[i]
            : std::string("unknown");
        const std::string featureGate =
            (i < output.performance.passGraphResourceFeatureGates.size())
                ? output.performance.passGraphResourceFeatureGates[i]
                : std::string();
        const std::string debugReadbackPolicy =
            (i < output.performance.passGraphResourceDebugReadbackPolicies.size())
                ? output.performance.passGraphResourceDebugReadbackPolicies[i]
                : std::string();
        const std::string clearLoadPolicy =
            (i < output.performance.passGraphResourceClearLoadPolicies.size())
                ? output.performance.passGraphResourceClearLoadPolicies[i]
                : std::string();
        const std::string migrationRisk =
            (i < output.performance.passGraphResourceMigrationRisks.size())
                ? output.performance.passGraphResourceMigrationRisks[i]
                : std::string("unknown");
        const int32_t firstPass = (i < output.performance.passGraphResourceFirstPasses.size())
            ? output.performance.passGraphResourceFirstPasses[i]
            : -1;
        const int32_t lastPass = (i < output.performance.passGraphResourceLastPasses.size())
            ? output.performance.passGraphResourceLastPasses[i]
            : -1;
        const int32_t aliasPool = (i < output.performance.passGraphResourceAliasPools.size())
            ? output.performance.passGraphResourceAliasPools[i]
            : -1;
        const bool used = (i < output.performance.passGraphResourceUsed.size())
            ? (output.performance.passGraphResourceUsed[i] != 0u)
            : false;
        const bool aliasable = (i < output.performance.passGraphResourceAliasable.size())
            ? (output.performance.passGraphResourceAliasable[i] != 0u)
            : false;
        const bool hotPath = (i < output.performance.passGraphResourceHotPath.size())
            ? (output.performance.passGraphResourceHotPath[i] != 0u)
            : false;
        const bool temporaryShared = (i < output.performance.passGraphResourceTemporaryShared.size())
            ? (output.performance.passGraphResourceTemporaryShared[i] != 0u)
            : false;
        outFile << "{\"name\":\"" << JsonEscape(output.performance.passGraphResourceNames[i])
                << "\",\"lifetime\":\"" << JsonEscape(lifetime)
                << "\",\"storage\":\"" << JsonEscape(storage)
                << "\",\"target_storage\":\"" << JsonEscape(targetStorage)
                << "\",\"size_formula\":\"" << JsonEscape(sizeFormula)
                << "\",\"debug_label\":\"" << JsonEscape(debugLabel)
                << "\",\"current_owner\":\"" << JsonEscape(currentOwner)
                << "\",\"target_owner\":\"" << JsonEscape(targetOwner)
                << "\",\"feature_gate\":\"" << JsonEscape(featureGate)
                << "\",\"debug_readback_policy\":\"" << JsonEscape(debugReadbackPolicy)
                << "\",\"clear_load_policy\":\"" << JsonEscape(clearLoadPolicy)
                << "\",\"migration_risk\":\"" << JsonEscape(migrationRisk)
                << "\",\"hot_path\":" << (hotPath ? "true" : "false")
                << ",\"temporary_shared\":" << (temporaryShared ? "true" : "false")
                << ",\"used\":" << (used ? "true" : "false")
                << ",\"first_pass_index\":" << firstPass
                << ",\"last_pass_index\":" << lastPass
                << ",\"aliasable\":" << (aliasable ? "true" : "false")
                << ",\"alias_pool_index\":" << aliasPool << "}";
    }
    outFile << "],\n";
    outFile << "    \"resource_transitions\": [";
    for (size_t i = 0; i < output.performance.passGraphTransitionResourceNames.size(); ++i) {
        if (i > 0) {
            outFile << ", ";
        }
        const uint32_t passIndex = (i < output.performance.passGraphTransitionPassIndices.size())
            ? output.performance.passGraphTransitionPassIndices[i]
            : 0u;
        const std::string passName = (i < output.performance.passGraphTransitionPassNames.size())
            ? output.performance.passGraphTransitionPassNames[i]
            : std::string();
        const std::string previousAccess =
            (i < output.performance.passGraphTransitionPreviousAccesses.size())
                ? output.performance.passGraphTransitionPreviousAccesses[i]
                : std::string("Unknown");
        const std::string access = (i < output.performance.passGraphTransitionAccesses.size())
            ? output.performance.passGraphTransitionAccesses[i]
            : std::string("Unknown");
        const bool firstUse = (i < output.performance.passGraphTransitionFirstUse.size())
            ? (output.performance.passGraphTransitionFirstUse[i] != 0u)
            : false;
        const bool requiresBarrier =
            (i < output.performance.passGraphTransitionRequiresBarrier.size())
                ? (output.performance.passGraphTransitionRequiresBarrier[i] != 0u)
                : false;
        const bool writesResource =
            (i < output.performance.passGraphTransitionWritesResource.size())
                ? (output.performance.passGraphTransitionWritesResource[i] != 0u)
                : false;
        outFile << "{\"pass_index\":" << passIndex
                << ",\"pass\":\"" << JsonEscape(passName)
                << "\",\"resource\":\""
                << JsonEscape(output.performance.passGraphTransitionResourceNames[i])
                << "\",\"previous_access\":\"" << JsonEscape(previousAccess)
                << "\",\"access\":\"" << JsonEscape(access)
                << "\",\"first_use\":" << (firstUse ? "true" : "false")
                << ",\"requires_barrier\":" << (requiresBarrier ? "true" : "false")
                << ",\"writes_resource\":" << (writesResource ? "true" : "false") << "}";
    }
    outFile << "]\n";
    outFile << "  },\n";
    outFile << "  \"image\": {\n";
    outFile << "    \"pixel_count\": " << imageMetrics.pixelCount << ",\n";
    outFile << "    \"finite_pixel_count\": " << imageMetrics.finitePixelCount << ",\n";
    outFile << "    \"nan_inf_pixel_count\": " << imageMetrics.nanInfPixelCount << ",\n";
    outFile << "    \"nan_inf_component_count\": " << imageMetrics.nanInfComponentCount << ",\n";
    outFile << "    \"luminance\": {\n";
    outFile << "      \"mean\": " << imageMetrics.meanLuminance << ",\n";
    outFile << "      \"p50\": " << imageMetrics.p50Luminance << ",\n";
    outFile << "      \"p95\": " << imageMetrics.p95Luminance << ",\n";
    outFile << "      \"p99\": " << imageMetrics.p99Luminance << ",\n";
    outFile << "      \"p99_9\": " << imageMetrics.p999Luminance << ",\n";
    outFile << "      \"variance\": " << imageMetrics.varianceLuminance << ",\n";
    outFile << "      \"max\": " << imageMetrics.maxLuminance << "\n";
    outFile << "    },\n";
    outFile << "    \"luminance_histogram\": {\n";
    outFile << "      \"type\": \"log10\",\n";
    outFile << "      \"bucket_count\": " << imageMetrics.histogramCounts.size() << ",\n";
    outFile << "      \"edges\": [";
    for (size_t i = 0; i < imageMetrics.histogramEdges.size(); ++i) {
        if (i > 0) {
            outFile << ", ";
        }
        outFile << imageMetrics.histogramEdges[i];
    }
    outFile << "],\n";
    outFile << "      \"counts\": [";
    for (size_t i = 0; i < imageMetrics.histogramCounts.size(); ++i) {
        if (i > 0) {
            outFile << ", ";
        }
        outFile << imageMetrics.histogramCounts[i];
    }
    outFile << "]\n";
    outFile << "    }\n";
    outFile << "  },\n";
    outFile << "  \"fallback_reports\": [";
    bool wroteFallbackReport = false;
    auto writeFallbackReport = [&](const char* requestedFeature,
                                   const char* requestedMode,
                                   const char* selectedPath,
                                   const char* reason,
                                   uint64_t count,
                                   const char* pass,
                                   bool behaviorChanged) {
        if (wroteFallbackReport) {
            outFile << ", ";
        }
        wroteFallbackReport = true;
        outFile << "{\"requested_feature\":\"" << JsonEscape(requestedFeature)
                << "\",\"requested_mode\":\"" << JsonEscape(requestedMode)
                << "\",\"selected_path\":\"" << JsonEscape(selectedPath)
                << "\",\"reason\":\"" << JsonEscape(reason)
                << "\",\"count\":" << count
                << ",\"frame\":" << output.performance.frameCount
                << ",\"pass\":\"" << JsonEscape(pass)
                << "\",\"behavior_changed\":"
                << (behaviorChanged ? "true" : "false") << "}";
    };
    const uint64_t explicitRestirDiDispatches =
        output.performance.restirDiInitializeDispatchCount +
        output.performance.restirDiInitialCandidateDispatchCount +
        output.performance.restirDiTemporalReuseDispatchCount +
        output.performance.restirDiSpatialReuseDispatchCount +
        output.performance.restirDiVisibilityDispatchCount +
        output.performance.restirDiApplyLightingDispatchCount;
    const bool integratedExplicitRestirDi =
        output.performance.restirDiPathStatus == "explicit_megakernel_integrated" ||
        output.performance.restirDiPathStatus == "explicit_wavefront";
    const uint64_t explicitRestirDiActivity =
        explicitRestirDiDispatches +
        (integratedExplicitRestirDi ? 1u : 0u);
    const std::string restirDiSelectedPath = !directRestirRequested
        ? std::string("disabled")
        : (integratedExplicitRestirDi
               ? std::string("explicit_integrated")
               : (!output.performance.restirDiPathStatus.empty()
               ? output.performance.restirDiPathStatus
               : (explicitRestirDiDispatches > 0u ? std::string("explicit_d_graph")
                                                  : std::string("fused_reference_path"))));
    // Homogeneous volumes run ReSTIR DI as baseline_emissive on Metal (RestirDiDemotedByVolume); the path status is
    // then volume_baseline_emissive.
    const bool volumeRestirDiDisabled = volumeModesDemoted && PathTracer::RestirDiDemotedByVolume(settings);
    const std::string restirDiReason = !directRestirRequested
        ? std::string("direct light mode does not request ReSTIR DI")
        : volumeRestirDiDisabled
        ? std::string("ReSTIR DI does not run with homogeneous volume transport (its reservoirs carry no medium "
                      "state); baseline_emissive renders instead")
        : (integratedExplicitRestirDi
               ? std::string("explicit integrated direct-light stage selected")
               : (explicitRestirDiDispatches > 0u
                      ? std::string("explicit D graph dispatched")
                      : (output.performance.restirDiPathStatus == "fused_megakernel_wavefront_fallback"
                             ? std::string("the wavefront fell back to the megakernel, which has no wavefront "
                                           "explicit stage; the megakernel's fused direct light selected")
                             : std::string("explicit D graph did not dispatch; fused/reference path selected"))));
    writeFallbackReport("traversal",
                        "configured_intersection_provider",
                        output.performance.hardwareRaytracingActive
                            ? "metal_hwrt"
                            : (settings.enableSoftwareRayTracing ? "software_bvh_forced"
                                                                 : "software_bvh"),
                        settings.enableSoftwareRayTracing
                            ? "software traversal forced by render settings"
                            : (!output.performance.hardwareRaytracingFallbackReason.empty()
                                   ? output.performance.hardwareRaytracingFallbackReason.c_str()
                                   : (output.performance.hardwareRaytracingActive
                                          ? "hardware ray tracing active"
                                          : "software BVH selected")),
                        output.performance.hardwareRaytracingFallbackActive ? 1u : 0u,
                        "intersection",
                        settings.enableSoftwareRayTracing ||
                            output.performance.hardwareRaytracingFallbackActive);
    writeFallbackReport("restir_di",
                        DirectLightModeLabel(settings.directLightMode),
                        restirDiSelectedPath.c_str(),
                        restirDiReason.c_str(),
                        explicitRestirDiActivity,
                        "direct_lighting",
                        directRestirRequested && explicitRestirDiActivity == 0u);
    writeFallbackReport("path_guiding",
                        PathGuidingModeLabel(settings.pathGuidingMode),
                        volumeModesDemoted &&
                                settings.pathGuidingMode ==
                                    PathTracer::RenderSettings::PathGuidingMode::Prototype
                            ? "disabled_by_volume_fallback"
                            : settings.pathGuidingMode ==
                                      PathTracer::RenderSettings::PathGuidingMode::Prototype
                            ? "production_guiding_provider"
                            : "disabled",
                        volumeModesDemoted &&
                                settings.pathGuidingMode ==
                                    PathTracer::RenderSettings::PathGuidingMode::Prototype
                            ? "path-guiding provider is surface-domain only; medium segments use phase-function sampling"
                            : output.pbrMetrics.pathGuidingFallbackCount > 0u
                            ? "guiding provider rejected or lacked confidence for some samples"
                            : "no path guiding sample fallback recorded",
                        volumeModesDemoted &&
                                settings.pathGuidingMode ==
                                    PathTracer::RenderSettings::PathGuidingMode::Prototype
                            ? 1u
                            : output.pbrMetrics.pathGuidingFallbackCount,
                        "path_guiding",
                        output.pbrMetrics.pathGuidingFallbackCount > 0u ||
                            (volumeModesDemoted &&
                             settings.pathGuidingMode ==
                                 PathTracer::RenderSettings::PathGuidingMode::Prototype));
    writeFallbackReport("radiance_cache",
                        RadianceCacheModeLabel(settings.radianceCacheMode),
                        radianceCacheRefusal ? "refused"
                        : volumeRadianceCacheDisabled
                            ? "disabled_by_volume_fallback"
                            : settings.radianceCacheMode ==
                                      PathTracer::RenderSettings::RadianceCacheMode::Prototype
                            ? "production_world_grid_radiance_provider"
                            : "disabled",
                        radianceCacheRefusal ? PathTracer::RadianceCacheRefusalMessage(radianceCacheRefusal)
                        : volumeRadianceCacheDisabled
                            ? "radiance-cache query records do not include medium-state keys; volume paths fall back to path tracing"
                            : output.pbrMetrics.radianceCacheFallbackCount > 0u
                            ? "radiance provider miss or low confidence fell back to path continuation"
                            : "no radiance cache fallback recorded",
                        (radianceCacheRefusal || volumeRadianceCacheDisabled)
                            ? 1u
                            : output.pbrMetrics.radianceCacheFallbackCount,
                        "radiance_cache",
                        output.pbrMetrics.radianceCacheFallbackCount > 0u || radianceCacheRefusal != nullptr ||
                            volumeRadianceCacheDisabled);
    writeFallbackReport("volume_restir_di_interaction",
                        DirectLightModeLabel(settings.directLightMode),
                        volumeRestirDiDisabled ? "volume_baseline_emissive" : "unchanged",
                        volumeRestirDiDisabled
                            ? "ReSTIR DI does not run with homogeneous volume transport (its reservoirs carry no medium state); baseline_emissive renders instead"
                            : "no volume/ReSTIR DI fallback required",
                        volumeRestirDiDisabled ? 1u : 0u,
                        "volume_transport",
                        volumeRestirDiDisabled);
    const bool volumeRestirGiPtDisabled =
        volumeModesDemoted &&
        (settings.restirGiMode != PathTracer::RenderSettings::RestirGiMode::Off ||
         settings.restirPtMode != PathTracer::RenderSettings::RestirPtMode::Off);
    writeFallbackReport("volume_restir_gi_pt_interaction",
                        RestirGiModeLabel(settings.restirGiMode),
                        volumeRestirGiPtDisabled ? "path_continuation" : "unchanged",
                        volumeRestirGiPtDisabled
                            ? "ReSTIR GI/PT reservoirs do not include medium-state keys; volume paths fall back to path continuation"
                            : "no volume/ReSTIR GI/PT fallback required",
                        volumeRestirGiPtDisabled ? 1u : 0u,
                        "volume_transport",
                        volumeRestirGiPtDisabled);
    const bool volumeGuidingDisabled =
        volumeModesDemoted &&
        settings.pathGuidingMode == PathTracer::RenderSettings::PathGuidingMode::Prototype;
    writeFallbackReport("volume_path_guiding_interaction",
                        PathGuidingModeLabel(settings.pathGuidingMode),
                        volumeGuidingDisabled ? "phase_sampling" : "unchanged",
                        volumeGuidingDisabled
                            ? "path-guiding provider is surface-domain only; medium segments use phase-function sampling"
                            : "no volume/path-guiding fallback required",
                        volumeGuidingDisabled ? 1u : 0u,
                        "volume_transport",
                        volumeGuidingDisabled);
    writeFallbackReport("volume_radiance_cache_interaction",
                        RadianceCacheModeLabel(settings.radianceCacheMode),
                        volumeRadianceCacheDisabled ? "path_tracing" : "unchanged",
                        volumeRadianceCacheDisabled
                            ? "radiance-cache query records do not include medium-state keys; volume paths fall back to path tracing"
                            : "no volume/radiance-cache fallback required",
                        volumeRadianceCacheDisabled ? 1u : 0u,
                        "volume_transport",
                        volumeRadianceCacheDisabled);
    const bool restirPtEnhancedRequested = RestirPtEnhancedActive(settings.restirPtMode);
    writeFallbackReport("restir_pt_enhanced_paper_features",
                        RestirPtModeLabel(settings.restirPtMode),
                        !restirPtEnhancedRequested ? "disabled"
                            : options.backend == HeadlessBackend::Embree ? "ordinary_cpu_capture_reuse"
                            : volumeModesDemoted ? "disabled_by_volume_fallback" : "fused_directional_reuse_heuristics",
                        restirPtEnhancedRequested
                            ? "configuration fallback: paired_spatial_reuse, footprint_reconnection, duplication_decorrelation/cCap, unified_di_gi, vector_shading_weights, dual_motion_vectors, replay_compaction unsupported; count is configuration-only, not a renderer counter"
                            : "ReSTIR PT Enhanced not requested",
                        restirPtEnhancedRequested ? 1u : 0u,
                        "restir_pt_enhanced",
                        restirPtEnhancedRequested);
    writeFallbackReport("ml_backend",
                        MlBackendModeLabel(settings.mlBackendMode),
                        output.performance.mlBackendActive
                            ? output.performance.mlBackendProvider.c_str()
                            : "disabled",
                        output.performance.mlBackendActive
                            ? "custom Metal ML backend active"
                            : (output.performance.mlBackendFallbackReason.empty()
                                   ? "ml backend disabled"
                                   : output.performance.mlBackendFallbackReason.c_str()),
                        output.performance.mlBackendActive
                            ? output.performance.mlBackendPassCount
                            : (settings.mlBackendMode != PathTracer::RenderSettings::MlBackendMode::Off ? 1u : 0u),
                        "reconstruction",
                        settings.mlBackendMode != PathTracer::RenderSettings::MlBackendMode::Off &&
                            !output.performance.mlBackendNativeAccelerationAvailable);
    uint64_t wavefrontOverflowCount = 0u;
    for (uint32_t queueIndex = 0u;
         queueIndex < HeadlessRenderOutput::kWavefrontQueueMetricCount;
         ++queueIndex) {
        wavefrontOverflowCount += output.performance.wavefrontQueueOverflowCounts[queueIndex];
    }
    writeFallbackReport("wavefront_scheduler",
                        WavefrontSchedulingPolicyLabel(settings.wavefrontSchedulingPolicy),
                        wavefrontOverflowCount > 0u ? "overflow_fallback" : "configured_policy",
                        wavefrontOverflowCount > 0u
                            ? "one or more wavefront queues overflowed"
                            : "no wavefront queue overflow recorded",
                        wavefrontOverflowCount,
                        "wavefront",
                        wavefrontOverflowCount > 0u);
    const bool wavefrontMegakernelFallback = output.performance.wavefrontMegakernelFallback != "none";
    writeFallbackReport("wavefront_executor",
                        settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront
                            ? "wavefront"
                            : "megakernel",
                        wavefrontMegakernelFallback ? "megakernel_fallback" : "configured_executor",
                        wavefrontMegakernelFallback ? output.performance.wavefrontMegakernelFallback.c_str()
                                                    : "no fallback to the megakernel recorded",
                        wavefrontMegakernelFallback ? 1u : 0u,
                        "wavefront",
                        wavefrontMegakernelFallback);
    outFile << "],\n";
    outFile << "  \"renderer\": {\n";
    outFile << "    \"available\": " << (output.pbrMetrics.available ? "true" : "false") << ",\n";
    outFile << "    \"throughput_nan_inf_count\": " << output.pbrMetrics.throughputNanInfCount << ",\n";
    outFile << "    \"radiance_nan_inf_count\": " << output.pbrMetrics.radianceNanInfCount << ",\n";
    outFile << "    \"clamp_event_count\": " << output.pbrMetrics.clampEventCount << ",\n";
    outFile << "    \"total_samples\": " << totalSamples << ",\n";
    outFile << "    \"clamp_event_ratio\": "
            << (totalSamples > 0
                    ? (static_cast<double>(output.pbrMetrics.clampEventCount) /
                       static_cast<double>(totalSamples))
                    : 0.0)
            << ",\n";
    outFile << "    \"spec_nee_env_added_count\": "
            << output.pbrMetrics.specNeeEnvAddedCount << ",\n";
    outFile << "    \"spec_nee_rect_added_count\": "
            << output.pbrMetrics.specNeeRectAddedCount << ",\n";
    outFile << "    \"mnee_eligible_count\": "
            << output.pbrMetrics.mneeEligibleCount << ",\n";
    outFile << "    \"mnee_env_attempt_count\": "
            << output.pbrMetrics.mneeEnvAttemptCount << ",\n";
    outFile << "    \"mnee_env_added_count\": "
            << output.pbrMetrics.mneeEnvAddedCount << ",\n";
    outFile << "    \"mnee_env_chain_unsupported_count\": "
            << output.pbrMetrics.mneeEnvChainUnsupportedCount << ",\n";
    outFile << "    \"mnee_env_chain_non_delta_block_count\": "
            << output.pbrMetrics.mneeEnvChainNonDeltaBlockCount << ",\n";
    outFile << "    \"mnee_env_chain_sample_reject_count\": "
            << output.pbrMetrics.mneeEnvChainSampleRejectCount << ",\n";
    outFile << "    \"mnee_env_chain_weight_reject_count\": "
            << output.pbrMetrics.mneeEnvChainWeightRejectCount << ",\n";
    outFile << "    \"mnee_env_chain_max_bounce_count\": "
            << output.pbrMetrics.mneeEnvChainMaxBounceCount << ",\n";
    outFile << "    \"mnee_rect_added_count\": "
            << output.pbrMetrics.mneeRectAddedCount << ",\n";
    outFile << "    \"mnee_contribution_count\": "
            << output.pbrMetrics.mneeContributionCount << ",\n";
    outFile << "    \"restir_di_regir_candidate_count\": "
            << output.pbrMetrics.restirDiRegirCandidateCount << ",\n";
    outFile << "    \"restir_di_regir_accepted_count\": "
            << output.pbrMetrics.restirDiRegirAcceptedCount << ",\n";
    outFile << "    \"restir_di_regir_fallback_count\": "
            << output.pbrMetrics.restirDiRegirFallbackCount << ",\n";
    outFile << "    \"restir_gi_candidate_count\": "
            << output.pbrMetrics.restirGiCandidateCount << ",\n";
    outFile << "    \"restir_gi_reservoir_update_count\": "
            << output.pbrMetrics.restirGiReservoirUpdateCount << ",\n";
    outFile << "    \"restir_gi_accept_count\": "
            << output.pbrMetrics.restirGiAcceptCount << ",\n";
    outFile << "    \"restir_gi_reject_count\": "
            << output.pbrMetrics.restirGiRejectCount << ",\n";
    outFile << "    \"restir_gi_fallback_count\": "
            << output.pbrMetrics.restirGiFallbackCount << ",\n";
    outFile << "    \"path_guiding_candidate_count\": "
            << output.pbrMetrics.pathGuidingCandidateCount << ",\n";
    outFile << "    \"path_guiding_used_count\": "
            << output.pbrMetrics.pathGuidingUsedCount << ",\n";
    outFile << "    \"path_guiding_fallback_count\": "
            << output.pbrMetrics.pathGuidingFallbackCount << ",\n";
    outFile << "    \"path_guiding_invalid_count\": "
            << output.pbrMetrics.pathGuidingInvalidCount << ",\n";
    outFile << "    \"restir_pt_candidate_count\": "
            << output.pbrMetrics.restirPtCandidateCount << ",\n";
    outFile << "    \"restir_pt_reservoir_update_count\": "
            << output.pbrMetrics.restirPtReservoirUpdateCount << ",\n";
    outFile << "    \"restir_pt_rejected_invalid_count\": "
            << output.pbrMetrics.restirPtRejectedInvalidCount << ",\n";
    outFile << "    \"restir_pt_rejected_unsupported_count\": "
            << output.pbrMetrics.restirPtRejectedUnsupportedCount << ",\n";
    outFile << "    \"restir_pt_debug_record_count\": "
            << output.pbrMetrics.restirPtDebugRecordCount << ",\n";
    outFile << "    \"restir_pt_reuse_candidate_count\": "
            << output.pbrMetrics.restirPtReuseCandidateCount << ",\n";
    outFile << "    \"restir_pt_reuse_applied_count\": "
            << output.pbrMetrics.restirPtReuseAppliedCount << ",\n";
    outFile << "    \"restir_pt_reuse_rejected_invalid_count\": "
            << output.pbrMetrics.restirPtReuseRejectedInvalidCount << ",\n";
    outFile << "    \"restir_pt_reuse_fallback_count\": "
            << output.pbrMetrics.restirPtReuseFallbackCount << ",\n";
    outFile << "    \"restir_pt_enhanced_candidate_count\": "
            << output.pbrMetrics.restirPtEnhancedCandidateCount << ",\n";
    outFile << "    \"restir_pt_enhanced_initial_selected_count\": "
            << output.pbrMetrics.restirPtEnhancedInitialSelectedCount << ",\n";
    outFile << "    \"restir_pt_enhanced_temporal_accepted_count\": "
            << output.pbrMetrics.restirPtEnhancedTemporalAcceptedCount << ",\n";
    outFile << "    \"restir_pt_enhanced_temporal_rejected_count\": "
            << output.pbrMetrics.restirPtEnhancedTemporalRejectedCount << ",\n";
    outFile << "    \"restir_pt_enhanced_spatial_accepted_count\": "
            << output.pbrMetrics.restirPtEnhancedSpatialAcceptedCount << ",\n";
    outFile << "    \"restir_pt_enhanced_spatial_rejected_count\": "
            << output.pbrMetrics.restirPtEnhancedSpatialRejectedCount << ",\n";
    outFile << "    \"restir_pt_enhanced_invalid_shift\": {\n";
    outFile << "      \"geometry\": " << output.pbrMetrics.restirPtEnhancedInvalidShiftGeometryCount << ",\n";
    outFile << "      \"material\": " << output.pbrMetrics.restirPtEnhancedInvalidShiftMaterialCount << ",\n";
    outFile << "      \"pdf\": " << output.pbrMetrics.restirPtEnhancedInvalidShiftPdfCount << ",\n";
    outFile << "      \"visibility\": " << output.pbrMetrics.restirPtEnhancedInvalidShiftVisibilityCount << ",\n";
    outFile << "      \"invertibility\": " << output.pbrMetrics.restirPtEnhancedInvalidShiftInvertibilityCount << ",\n";
    outFile << "      \"unsupported\": " << output.pbrMetrics.restirPtEnhancedInvalidShiftUnsupportedCount << "\n";
    outFile << "    },\n";
    outFile << "    \"restir_pt_enhanced_duplicate_score_sum\": "
            << output.pbrMetrics.restirPtEnhancedDuplicateScoreSum << ",\n";
    outFile << "    \"restir_pt_enhanced_biased_cCap_application_count\": "
            << output.pbrMetrics.restirPtEnhancedBiasedCCapApplicationCount << ",\n";
    outFile << "    \"restir_pt_enhanced_vector_weight_use_count\": "
            << output.pbrMetrics.restirPtEnhancedVectorWeightUseCount << ",\n";
    outFile << "    \"restir_pt_enhanced_replay_compaction_input_count\": "
            << output.pbrMetrics.restirPtEnhancedReplayCompactionInputCount << ",\n";
    outFile << "    \"restir_pt_enhanced_replay_compaction_output_count\": "
            << output.pbrMetrics.restirPtEnhancedReplayCompactionOutputCount << ",\n";
    outFile << "    \"restir_pt_enhanced_fallback_count\": "
            << output.pbrMetrics.restirPtEnhancedFallbackCount << ",\n";
    outFile << "    \"restir_pt_enhanced_canonical_sample_count\": "
            << output.pbrMetrics.restirPtEnhancedCanonicalSampleCount << ",\n";
    outFile << "    \"restir_pt_enhanced_path_length_histogram\": [";
    for (uint32_t i = 0u; i < HeadlessRenderOutput::PbrMetrics::kRestirPtEnhancedHistogramBins; ++i) {
        if (i > 0u) outFile << ", ";
        outFile << output.pbrMetrics.restirPtEnhancedPathLengthHistogram[i];
    }
    outFile << "],\n";
    outFile << "    \"restir_pt_enhanced_reconnection_depth_histogram\": [";
    for (uint32_t i = 0u; i < HeadlessRenderOutput::PbrMetrics::kRestirPtEnhancedHistogramBins; ++i) {
        if (i > 0u) outFile << ", ";
        outFile << output.pbrMetrics.restirPtEnhancedReconnectionDepthHistogram[i];
    }
    outFile << "],\n";
    outFile << "    \"radiance_cache_query_count\": "
            << output.pbrMetrics.radianceCacheQueryCount << ",\n";
    outFile << "    \"radiance_cache_hit_count\": "
            << output.pbrMetrics.radianceCacheHitCount << ",\n";
    outFile << "    \"radiance_cache_miss_count\": "
            << output.pbrMetrics.radianceCacheMissCount << ",\n";
    outFile << "    \"radiance_cache_train_count\": "
            << output.pbrMetrics.radianceCacheTrainCount << ",\n";
    outFile << "    \"radiance_cache_fallback_count\": "
            << output.pbrMetrics.radianceCacheFallbackCount << ",\n";
    outFile << "    \"radiance_cache_invalid_count\": "
            << output.pbrMetrics.radianceCacheInvalidCount << ",\n";
    outFile << "    \"radiance_cache_stats\": ";
    WriteRadianceCacheStatsJson(outFile, output.pbrMetrics);
    outFile << ",\n";
    outFile << "    \"caustic_eye_vertex_count\": "
            << output.pbrMetrics.causticEyeVertexCount << ",\n";
    outFile << "    \"caustic_light_vertex_count\": "
            << output.pbrMetrics.causticLightVertexCount << ",\n";
    outFile << "    \"caustic_connection_candidate_count\": "
            << output.pbrMetrics.causticConnectionCandidateCount << ",\n";
    outFile << "    \"caustic_delta_chain_count\": "
            << output.pbrMetrics.causticDeltaChainCount << ",\n";
    outFile << "    \"caustic_debug_classification_count\": "
            << output.pbrMetrics.causticDebugClassificationCount << ",\n";
    outFile << "    \"volume_transmittance_query_count\": "
            << output.pbrMetrics.volumeTransmittanceQueryCount << ",\n";
    outFile << "    \"volume_scatter_event_count\": "
            << output.pbrMetrics.volumeScatterEventCount << ",\n";
    outFile << "    \"volume_nee_ray_count\": "
            << output.pbrMetrics.volumeNeeRayCount << ",\n";
    outFile << "    \"volume_phase_sample_count\": "
            << output.pbrMetrics.volumePhaseSampleCount << ",\n";
    outFile << "    \"volume_medium_boundary_event_count\": "
            << output.pbrMetrics.volumeMediumBoundaryEventCount << ",\n";
    outFile << "    \"volume_fallback_count\": "
            << output.pbrMetrics.volumeFallbackCount << ",\n";
    outFile << "    \"spectral_wavelength_sample_count\": "
            << output.pbrMetrics.spectralWavelengthSampleCount << ",\n";
    outFile << "    \"spectral_material_lookup_count\": "
            << output.pbrMetrics.spectralMaterialLookupCount << ",\n";
    outFile << "    \"spectral_dispersion_event_count\": "
            << output.pbrMetrics.spectralDispersionEventCount << ",\n";
    outFile << "    \"spectral_fallback_count\": "
            << output.pbrMetrics.spectralFallbackCount << ",\n";
    outFile << "    \"spectral_collapse_count\": "
            << output.pbrMetrics.spectralCollapseCount << ",\n";
    outFile << "    \"spectral_wavelength_channel_sample_count\": "
            << output.pbrMetrics.spectralWavelengthChannelSampleCount << ",\n";
    outFile << "    \"max_throughput_by_bounce\": [";
    for (size_t i = 0; i < output.pbrMetrics.maxThroughputByBounce.size(); ++i) {
        if (i > 0) {
            outFile << ", ";
        }
        outFile << output.pbrMetrics.maxThroughputByBounce[i];
    }
    outFile << "]\n";
    outFile << "  }\n";
    outFile << "}\n";

    if (!outFile.good()) {
        error = "Failed while writing metrics JSON: " + jsonPath.string();
        return false;
    }
    outFile.close();
    if (!outFile) {
        error = "Failed while closing metrics temp file: " + tmpPath.string();
        return false;
    }
    return PublishTempFileAtomic(tmpPath, jsonPath, error);
}

bool WriteRestirDebugMetricsJson(const fs::path& jsonPath,
                                 const CliOptions& options,
                                 const PathTracer::RenderSettings& settings,
                                 const fs::path& outputPath,
                                 const HeadlessRenderOutput& output,
                                 const ImagePbrMetrics& imageMetrics,
                                 std::string& error) {
    fs::path tmpPath;
    std::ofstream outFile;
    if (!OpenAtomicSidecarStream(jsonPath, tmpPath, outFile, error, "ReSTIR debug metrics JSON")) {
        return false;
    }

    // The mode the backend ran: homogeneous volumes demote ReSTIR DI to baseline_emissive on Metal.
    const PathTracer::RenderSettings::DirectLightMode ranDirectLightMode = RunDirectLightMode(options, settings);
    const bool directActive =
        ranDirectLightMode != PathTracer::RenderSettings::DirectLightMode::LegacyRect &&
        ranDirectLightMode != PathTracer::RenderSettings::DirectLightMode::BaselineEmissive;
    const bool worldActive = RestirWorldModeActive(ranDirectLightMode);
    const bool giActive =
        settings.restirGiMode == PathTracer::RenderSettings::RestirGiMode::ProductionDiffuse;
    const bool pathGuidingActive =
        settings.pathGuidingMode == PathTracer::RenderSettings::PathGuidingMode::Prototype;
    const bool restirPtActive =
        settings.restirPtMode != PathTracer::RenderSettings::RestirPtMode::Off;
    const bool radianceCacheActive =
        settings.radianceCacheMode == PathTracer::RenderSettings::RadianceCacheMode::Prototype;
    const uint64_t nanInfCount =
        output.pbrMetrics.throughputNanInfCount +
        output.pbrMetrics.radianceNanInfCount +
        imageMetrics.nanInfPixelCount;
    const uint64_t directLightCandidateCount =
        directActive
            ? static_cast<uint64_t>(output.width) * static_cast<uint64_t>(output.height) *
                  static_cast<uint64_t>(std::max(settings.risCandidateCount, 1u)) *
                  static_cast<uint64_t>(std::max(output.samples, 1u))
            : 0u;

    outFile << std::fixed << std::setprecision(9);
    outFile << "{\n";
    outFile << "  \"scene\": \"" << JsonEscape(options.scene) << "\",\n";
    outFile << "  \"output\": \"" << JsonEscape(outputPath.string()) << "\",\n";
    outFile << "  \"restirDebug\": {\n";
    outFile << "    \"enabled\": " << (settings.restirDebugEnabled ? "true" : "false") << ",\n";
    outFile << "    \"view\": \"" << RestirDebugViewLabel(settings.restirDebugView) << "\",\n";
    outFile << "    \"countersEnabled\": " << (settings.restirDebugCounters ? "true" : "false") << ",\n";
#if PT_DEBUG_TOOLS
    outFile << "    \"debugPixel\": {\"enabled\": "
            << ((settings.enablePathDebug || settings.debugDirectLightAudit) ? "true" : "false")
            << ", \"x\": " << settings.debugPixelX
            << ", \"y\": " << settings.debugPixelY << "},\n";
#else
    outFile << "    \"debugPixel\": {\"enabled\": false, \"x\": 0, \"y\": 0},\n";
#endif
    outFile << "    \"modes\": {\n";
    outFile << "      \"renderMode\": \"" << RenderModeLabel(options.renderMode) << "\",\n";
    outFile << "      \"renderProfile\": \"" << RenderProfileLabel(options.renderProfile) << "\",\n";
    outFile << "      \"backend\": \"" << (options.backend == HeadlessBackend::Embree ? "embree" : "metal") << "\",\n";
    outFile << "      \"executionMode\": \""
            << (settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront
                    ? "wavefront"
                    : "megakernel")
            << "\",\n";
    outFile << "      \"wavefrontSchedulingPolicy\": \""
            << WavefrontSchedulingPolicyLabel(settings.wavefrontSchedulingPolicy) << "\",\n";
    outFile << "      \"directLightMode\": \"" << DirectLightModeLabel(settings.directLightMode) << "\",\n";
    outFile << "      \"volumeDirectLightFallback\": \""
            << JsonEscape(VolumeDirectLightFallbackLabel(options, settings)) << "\",\n";
    outFile << "      \"restirGiMode\": \"" << RestirGiModeLabel(settings.restirGiMode) << "\",\n";
    outFile << "      \"pathGuidingMode\": \"" << PathGuidingModeLabel(settings.pathGuidingMode) << "\",\n";
    outFile << "      \"restirPtMode\": \"" << RestirPtModeLabel(settings.restirPtMode) << "\",\n";
    outFile << "      \"svgfDenoise\": " << (settings.svgfDenoiseEnabled ? "true" : "false") << ",\n";
    outFile << "      \"svgfDenoiseDirect\": " << (settings.svgfDenoiseDirect ? "true" : "false") << ",\n";
    outFile << "      \"svgfDenoiseIndirect\": " << (settings.svgfDenoiseIndirect ? "true" : "false") << "\n";
    outFile << "    },\n";
    outFile << "    \"counters\": {\n";
    outFile << "      \"directLightCandidateCount\": " << directLightCandidateCount << ",\n";
    outFile << "      \"reservoirUpdateCount\": " << output.pbrMetrics.restirPtReservoirUpdateCount << ",\n";
    outFile << "      \"regirCellHitCount\": " << (worldActive ? directLightCandidateCount : 0u) << ",\n";
    outFile << "      \"regirCellMissCount\": 0,\n";
    outFile << "      \"giCandidateCount\": " << output.pbrMetrics.restirGiCandidateCount << ",\n";
    outFile << "      \"giReservoirUpdateCount\": " << output.pbrMetrics.restirGiReservoirUpdateCount << ",\n";
    outFile << "      \"giAcceptCount\": " << output.pbrMetrics.restirGiAcceptCount << ",\n";
    outFile << "      \"giRejectCount\": " << output.pbrMetrics.restirGiRejectCount << ",\n";
    outFile << "      \"giFallbackCount\": " << output.pbrMetrics.restirGiFallbackCount << ",\n";
    outFile << "      \"pathGuidingCandidateCount\": " << output.pbrMetrics.pathGuidingCandidateCount << ",\n";
    outFile << "      \"pathGuidingUsedCount\": " << output.pbrMetrics.pathGuidingUsedCount << ",\n";
    outFile << "      \"pathGuidingInvalidCount\": " << output.pbrMetrics.pathGuidingInvalidCount << ",\n";
    outFile << "      \"pathGuidingMaterialRejectCount\": " << output.pbrMetrics.pathGuidingFallbackCount << ",\n";
    outFile << "      \"restirPtCandidateCount\": " << output.pbrMetrics.restirPtCandidateCount << ",\n";
    outFile << "      \"restirPtReservoirUpdateCount\": " << output.pbrMetrics.restirPtReservoirUpdateCount << ",\n";
    outFile << "      \"restirPtRejectedUnsupportedCount\": " << output.pbrMetrics.restirPtRejectedUnsupportedCount << ",\n";
    outFile << "      \"restirPtReuseAppliedCount\": " << output.pbrMetrics.restirPtReuseAppliedCount << ",\n";
    outFile << "      \"restirPtReuseCandidateCount\": " << output.pbrMetrics.restirPtReuseCandidateCount << ",\n";
    outFile << "      \"restirPtReuseFallbackCount\": " << output.pbrMetrics.restirPtReuseFallbackCount << ",\n";
    outFile << "      \"restirPtRejectedInvalidCount\": " << output.pbrMetrics.restirPtRejectedInvalidCount << ",\n";
    outFile << "      \"restirPtReuseRejectedInvalidCount\": " << output.pbrMetrics.restirPtReuseRejectedInvalidCount << ",\n";
    outFile << "      \"restirPtDebugRecordCount\": " << output.pbrMetrics.restirPtDebugRecordCount << ",\n";
    outFile << "      \"restirPtEnhancedCandidateCount\": " << output.pbrMetrics.restirPtEnhancedCandidateCount << ",\n";
    outFile << "      \"restirPtEnhancedInitialSelectedCount\": " << output.pbrMetrics.restirPtEnhancedInitialSelectedCount << ",\n";
    outFile << "      \"restirPtEnhancedTemporalAcceptedCount\": " << output.pbrMetrics.restirPtEnhancedTemporalAcceptedCount << ",\n";
    outFile << "      \"restirPtEnhancedTemporalRejectedCount\": " << output.pbrMetrics.restirPtEnhancedTemporalRejectedCount << ",\n";
    outFile << "      \"restirPtEnhancedSpatialAcceptedCount\": " << output.pbrMetrics.restirPtEnhancedSpatialAcceptedCount << ",\n";
    outFile << "      \"restirPtEnhancedSpatialRejectedCount\": " << output.pbrMetrics.restirPtEnhancedSpatialRejectedCount << ",\n";
    outFile << "      \"restirPtEnhancedFallbackCount\": " << output.pbrMetrics.restirPtEnhancedFallbackCount << ",\n";
    outFile << "      \"restirPtEnhancedCanonicalSampleCount\": " << output.pbrMetrics.restirPtEnhancedCanonicalSampleCount << ",\n";
    outFile << "      \"restirPtEnhancedBiasedCCapApplicationCount\": " << output.pbrMetrics.restirPtEnhancedBiasedCCapApplicationCount << ",\n";
    outFile << "      \"restirPtEnhancedVectorWeightUseCount\": " << output.pbrMetrics.restirPtEnhancedVectorWeightUseCount << ",\n";
    outFile << "      \"radianceCacheQueryCount\": " << output.pbrMetrics.radianceCacheQueryCount << ",\n";
    outFile << "      \"radianceCacheHitCount\": " << output.pbrMetrics.radianceCacheHitCount << ",\n";
    outFile << "      \"radianceCacheMissCount\": " << output.pbrMetrics.radianceCacheMissCount << ",\n";
    outFile << "      \"radianceCacheTrainCount\": " << output.pbrMetrics.radianceCacheTrainCount << ",\n";
    outFile << "      \"radianceCacheFallbackCount\": " << output.pbrMetrics.radianceCacheFallbackCount << ",\n";
    outFile << "      \"radianceCacheInvalidCount\": " << output.pbrMetrics.radianceCacheInvalidCount << ",\n";
    outFile << "      \"radianceCacheStats\": ";
    WriteRadianceCacheStatsJson(outFile, output.pbrMetrics);
    outFile << ",\n";
    outFile << "      \"svgfActivePixelCount\": "
            << (settings.svgfDenoiseEnabled
                    ? static_cast<uint64_t>(output.width) * static_cast<uint64_t>(output.height)
                    : 0u)
            << ",\n";
    outFile << "      \"nanInfPixelCount\": " << imageMetrics.nanInfPixelCount << ",\n";
    outFile << "      \"rendererNanInfCount\": " << nanInfCount << "\n";
    outFile << "    },\n";
    outFile << "    \"sanity\": {\n";
    outFile << "      \"nanInf\": \"" << (nanInfCount == 0u ? "green" : "red") << "\",\n";
    outFile << "      \"defaultExperimentalGates\": \""
            << ((!pathGuidingActive && !radianceCacheActive && !settings.svgfDenoiseEnabled)
                    ? "green"
                    : "yellow")
            << "\",\n";
    outFile << "      \"restirProductionModes\": \""
            << ((giActive ||
                 restirPtActive ||
                 ranDirectLightMode == PathTracer::RenderSettings::DirectLightMode::RestirDi ||
                 ranDirectLightMode == PathTracer::RenderSettings::DirectLightMode::RestirDiRegirHybrid)
                    ? "active"
                    : "inactive")
            << "\",\n";
    outFile << "      \"radianceCacheActivity\": \""
            << ((!radianceCacheActive || output.pbrMetrics.radianceCacheTrainCount > 0u) ? "green" : "yellow")
            << "\",\n";
    outFile << "      \"pathGuidingActivity\": \""
            << ((!pathGuidingActive || output.pbrMetrics.pathGuidingUsedCount > 0u) ? "green" : "yellow")
            << "\",\n";
    outFile << "      \"restirGiActivity\": \""
            << ((!giActive || output.pbrMetrics.restirGiAcceptCount > 0u) ? "green" : "yellow")
            << "\",\n";
    outFile << "      \"restirPtActivity\": \""
            << ((!restirPtActive || output.pbrMetrics.restirPtCandidateCount > 0u) ? "green" : "yellow")
            << "\",\n";
    outFile << "      \"historyReset\": \""
            << JsonEscape(output.performance.historyResetReason.empty()
                              ? std::string("unavailable")
                              : output.performance.historyResetReason)
            << "\"\n";
    outFile << "    },\n";
    outFile << "    \"availableAOVs\": [\"beauty\", \"albedo\", \"normal\", \"position_depth\", \"material_features\", \"motion_disocclusion\", \"candidate_source_id\", \"reservoir_confidence\", \"regir_cell_id\", \"path_guiding_used_mask\", \"restir_pt_reuse_mask\", \"svgf_variance\", \"nan_inf_mask\"],\n";
    outFile << "    \"unavailable\": [\"selected_light_id per-pixel image\", \"temporal rejection image\", \"spatial reuse-count image\", \"live ReGIR hit/miss hardware counters\"]\n";
    outFile << "  }\n";
    outFile << "}\n";

    return PublishAtomicSidecarStream(outFile, tmpPath, jsonPath, error, "ReSTIR debug metrics JSON");
}
// JSON has no NaN or infinity, and a stream writes them as "nan" and "inf", which no JSON parser accepts. Imbued on a
// debug sidecar's stream, this facet writes a non-finite float or double as null and counts it. Finite values keep the
// stream's own formatting. ImbueJsonNonFiniteAsNull returns the facet, which lives as long as the stream's locale, so
// the writer can record the count in the sidecar and log it (ReportJsonNonFinite): never a silent substitute.
class JsonNonFiniteAsNull : public std::num_put<char> {
public:
    mutable size_t nanCount = 0u;
    mutable size_t infCount = 0u;

protected:
    iter_type do_put(iter_type out, std::ios_base& str, char_type fill, double value) const override {
        if (std::isfinite(value)) {
            return std::num_put<char>::do_put(out, str, fill, value);
        }
        return writeNull(out, std::isnan(value));
    }
    iter_type do_put(iter_type out, std::ios_base& str, char_type fill, long double value) const override {
        if (std::isfinite(value)) {
            return std::num_put<char>::do_put(out, str, fill, value);
        }
        return writeNull(out, std::isnan(value));
    }

private:
    iter_type writeNull(iter_type out, bool isNan) const {
        ++(isNan ? nanCount : infCount);
        for (const char* c = "null"; *c != '\0'; ++c) {
            *out++ = *c;
        }
        return out;
    }
};

const JsonNonFiniteAsNull& ImbueJsonNonFiniteAsNull(std::ostream& stream) {
    auto* facet = new JsonNonFiniteAsNull();  // owned by the locale
    stream.imbue(std::locale(stream.getloc(), facet));
    return *facet;
}

// Logs the non-finite values a sidecar wrote as null; writes nothing when there were none.
void ReportJsonNonFinite(const JsonNonFiniteAsNull& facet, const char* label, const fs::path& jsonPath) {
    if (facet.nanCount + facet.infCount == 0u) {
        return;
    }
    std::cerr << "[" << label << "] warning: " << (facet.nanCount + facet.infCount)
              << " non-finite values (" << facet.nanCount << " NaN, " << facet.infCount
              << " infinite) written as null to " << jsonPath.string() << std::endl;
}

bool WriteMaterialMrDebugJson(const fs::path& jsonPath,
                              const CliOptions& options,
                              const fs::path& outputPath,
                              const HeadlessRenderOutput::MaterialMrDebugDump& debugDump,
                              std::string& error) {
    fs::path tmpPath;
    std::ofstream outFile;
    if (!OpenAtomicSidecarStream(jsonPath, tmpPath, outFile, error, "Material_MR debug JSON")) {
        return false;
    }
    const JsonNonFiniteAsNull& nonFinite = ImbueJsonNonFiniteAsNull(outFile);
    outFile << std::fixed << std::setprecision(9);
    outFile << "{\n";
    outFile << "  \"scene\": \"" << JsonEscape(options.scene) << "\",\n";
    outFile << "  \"backend\": \"" << (options.backend == HeadlessBackend::Embree ? "embree" : "metal") << "\",\n";
    outFile << "  \"output\": \"" << JsonEscape(outputPath.string()) << "\",\n";
    outFile << "  \"available\": " << (debugDump.available ? "true" : "false") << ",\n";
    outFile << "  \"probe\": {\n";
    outFile << "    \"pixel_x\": " << debugDump.pixelX << ",\n";
    outFile << "    \"pixel_y\": " << debugDump.pixelY << ",\n";
    outFile << "    \"sample_index\": " << debugDump.sampleIndex << ",\n";
    outFile << "    \"depth\": " << debugDump.depth << ",\n";
    outFile << "    \"material_index\": " << debugDump.materialIndex << ",\n";
    outFile << "    \"primitive_index\": " << debugDump.primitiveIndex << "\n";
    outFile << "  },\n";
    auto writeFloat2 = [&](const char* name, simd::float2 v, bool trailing) {
        outFile << "  \"" << name << "\": [" << v.x << ", " << v.y << "]";
        outFile << (trailing ? ",\n" : "\n");
    };
    auto writeFloat3 = [&](const char* name, simd::float3 v, bool trailing) {
        outFile << "  \"" << name << "\": [" << v.x << ", " << v.y << ", " << v.z << "]";
        outFile << (trailing ? ",\n" : "\n");
    };
    writeFloat2("uv0", debugDump.uv0, true);
    writeFloat2("uv1", debugDump.uv1, true);
    writeFloat3("base_color", debugDump.baseColor, true);
    outFile << "  \"metallic\": " << debugDump.metallic << ",\n";
    outFile << "  \"roughness\": " << debugDump.roughness << ",\n";
    outFile << "  \"ao\": " << debugDump.ao << ",\n";
    writeFloat3("emissive", debugDump.emissive, true);
    writeFloat3("geometric_normal", debugDump.geometricNormal, true);
    writeFloat3("shading_normal", debugDump.shadingNormal, true);
    writeFloat3("tangent", debugDump.tangent, true);
    writeFloat3("bitangent", debugDump.bitangent, true);
    writeFloat3("normal_sample_raw", debugDump.normalSampleRaw, true);
    writeFloat3("normal_sample_ts", debugDump.normalSampleTs, true);
    writeFloat3("perturbed_normal", debugDump.perturbedNormal, true);
    writeFloat3("f0", debugDump.f0, true);
    outFile << "  \"diffuse_weight\": " << debugDump.diffuseWeight << ",\n";
    outFile << "  \"specular_weight\": " << debugDump.specularWeight << ",\n";
    outFile << "  \"normal_scale\": " << debugDump.normalScale << ",\n";
    outFile << "  \"transmission\": " << debugDump.transmission << ",\n";
    outFile << "  \"alpha\": " << debugDump.alpha << ",\n";
    outFile << "  \"alpha_cutoff\": " << debugDump.alphaCutoff << ",\n";
    outFile << "  \"alpha_mode\": " << debugDump.alphaMode << ",\n";
    outFile << "  \"roughness_input\": " << debugDump.roughnessInput << ",\n";
    outFile << "  \"roughness_before_normal_variance\": " << debugDump.roughnessBeforeNormalVariance << ",\n";
    outFile << "  \"normal_lod\": " << debugDump.normalLod << ",\n";
    outFile << "  \"ndotv\": " << debugDump.ndotv << ",\n";
    writeFloat3("sampled_outgoing_direction", debugDump.sampledOutgoingDirection, true);
    writeFloat3("brdf_specular_value", debugDump.brdfSpecularValue, true);
    outFile << "  \"specular_pdf\": " << debugDump.specularPdf << ",\n";
    outFile << "  \"d_term\": " << debugDump.dTerm << ",\n";
    outFile << "  \"g_term\": " << debugDump.gTerm << ",\n";
    writeFloat3("f_term", debugDump.fTerm, true);
    writeFloat3("final_specular_contribution", debugDump.finalSpecularContribution, true);
    writeFloat3("direct_contribution", debugDump.directContribution, true);
    outFile << "  \"direct_contribution_kind\": " << debugDump.directContributionKind << ",\n";
    outFile << "  \"non_finite_values\": " << (nonFinite.nanCount + nonFinite.infCount) << "\n";
    outFile << "}\n";
    ReportJsonNonFinite(nonFinite, "Material_MR debug", jsonPath);
    return PublishAtomicSidecarStream(outFile, tmpPath, jsonPath, error, "Material_MR debug JSON");
}

bool WriteDirectLightAuditJson(const fs::path& jsonPath,
                               const CliOptions& options,
                               const fs::path& outputPath,
                               const HeadlessRenderOutput::DirectLightAuditDump& auditDump,
                               std::string& error) {
    fs::path tmpPath;
    std::ofstream outFile;
    if (!OpenAtomicSidecarStream(jsonPath, tmpPath, outFile, error, "direct-light audit JSON")) {
        return false;
    }
    const JsonNonFiniteAsNull& nonFinite = ImbueJsonNonFiniteAsNull(outFile);
    outFile << std::fixed << std::setprecision(9);
    auto writeFloat3 = [&](const char* name, simd::float3 v, bool trailing) {
        outFile << "  \"" << name << "\": [" << v.x << ", " << v.y << ", " << v.z << "]";
        outFile << (trailing ? ",\n" : "\n");
    };
    outFile << "{\n";
    outFile << "  \"scene\": \"" << JsonEscape(options.scene) << "\",\n";
    outFile << "  \"backend\": \"" << (options.backend == HeadlessBackend::Embree ? "embree" : "metal") << "\",\n";
    outFile << "  \"output\": \"" << JsonEscape(outputPath.string()) << "\",\n";
    outFile << "  \"available\": " << (auditDump.available ? "true" : "false") << ",\n";
    outFile << "  \"probe\": {\n";
    outFile << "    \"pixel_x\": " << auditDump.pixelX << ",\n";
    outFile << "    \"pixel_y\": " << auditDump.pixelY << ",\n";
    outFile << "    \"sample_index\": " << auditDump.sampleIndex << ",\n";
    outFile << "    \"depth\": " << auditDump.depth << ",\n";
    outFile << "    \"material_index\": " << auditDump.materialIndex << ",\n";
    outFile << "    \"material_type\": " << auditDump.materialType << ",\n";
    outFile << "    \"raw_material_type\": " << auditDump.rawMaterialType << ",\n";
    outFile << "    \"primitive_index\": " << auditDump.primitiveIndex << "\n";
    outFile << "  },\n";
    writeFloat3("hit_position", auditDump.hitPosition, true);
    writeFloat3("shading_normal", auditDump.shadingNormal, true);
    writeFloat3("wo", auditDump.wo, true);
    outFile << "  \"uv0\": [" << auditDump.uv0.x << ", " << auditDump.uv0.y << "],\n";
    writeFloat3("base_color", auditDump.baseColor, true);
    outFile << "  \"roughness\": " << auditDump.roughness << ",\n";
    outFile << "  \"base_color_lod\": " << auditDump.baseColorLod << ",\n";
    outFile << "  \"base_color_texture_index\": " << auditDump.baseColorTextureIndex << ",\n";
    outFile << "  \"diffuse_occlusion\": " << auditDump.diffuseOcclusion << ",\n";
    outFile << "  \"pbr_params\": [" << auditDump.pbrParams.x << ", "
            << auditDump.pbrParams.y << ", "
            << auditDump.pbrParams.z << ", "
            << auditDump.pbrParams.w << "],\n";
    outFile << "  \"pbr_extras\": [" << auditDump.pbrExtras.x << ", "
            << auditDump.pbrExtras.y << ", "
            << auditDump.pbrExtras.z << ", "
            << auditDump.pbrExtras.w << "],\n";
    outFile << "  \"type_eta\": [" << auditDump.typeEta.x << ", "
            << auditDump.typeEta.y << ", "
            << auditDump.typeEta.z << ", "
            << auditDump.typeEta.w << "],\n";
    outFile << "  \"ris\": {\n";
    outFile << "    \"available\": " << (auditDump.ris.available ? "true" : "false") << ",\n";
    outFile << "    \"valid\": " << (auditDump.ris.valid ? "true" : "false") << ",\n";
    outFile << "    \"visible\": " << (auditDump.ris.visible ? "true" : "false") << ",\n";
    outFile << "    \"winner_primitive_index\": " << auditDump.ris.winnerPrimitiveIndex << ",\n";
    outFile << "    \"winner_bary\": [" << auditDump.ris.winnerBary.x << ", " << auditDump.ris.winnerBary.y << "],\n";
    outFile << "    \"winner_position\": [" << auditDump.ris.winnerPosition.x << ", " << auditDump.ris.winnerPosition.y << ", " << auditDump.ris.winnerPosition.z << "],\n";
    outFile << "    \"winner_normal\": [" << auditDump.ris.winnerNormal.x << ", " << auditDump.ris.winnerNormal.y << ", " << auditDump.ris.winnerNormal.z << "],\n";
    outFile << "    \"winner_emission\": [" << auditDump.ris.winnerEmission.x << ", " << auditDump.ris.winnerEmission.y << ", " << auditDump.ris.winnerEmission.z << "],\n";
    outFile << "    \"winner_pdf\": " << auditDump.ris.winnerPdf << ",\n";
    outFile << "    \"winner_distance\": " << auditDump.ris.winnerDistance << ",\n";
    outFile << "    \"winner_n_dot_l\": " << auditDump.ris.winnerNDotL << ",\n";
    outFile << "    \"winner_bsdf_value\": [" << auditDump.ris.winnerBsdfValue.x << ", " << auditDump.ris.winnerBsdfValue.y << ", " << auditDump.ris.winnerBsdfValue.z << "],\n";
    outFile << "    \"winner_bsdf_pdf\": " << auditDump.ris.winnerBsdfPdf << ",\n";
    outFile << "    \"winner_contribution\": [" << auditDump.ris.winnerContribution.x << ", " << auditDump.ris.winnerContribution.y << ", " << auditDump.ris.winnerContribution.z << "],\n";
    outFile << "    \"w_sum\": " << auditDump.ris.wSum << ",\n";
    outFile << "    \"M\": " << auditDump.ris.M << ",\n";
    outFile << "    \"W\": " << auditDump.ris.W << ",\n";
    outFile << "    \"rng_before\": " << auditDump.ris.rngBefore << ",\n";
    outFile << "    \"rng_after\": " << auditDump.ris.rngAfter << ",\n";
    outFile << "    \"rng_temporal_before\": " << auditDump.ris.rngTemporalBefore << ",\n";
    outFile << "    \"rng_temporal_after\": " << auditDump.ris.rngTemporalAfter << ",\n";
    outFile << "    \"spatial_reuse_attempted\": "
            << (auditDump.ris.spatialReuseAttempted ? "true" : "false") << ",\n";
    outFile << "    \"spatial_neighbor_target\": " << auditDump.ris.spatialNeighborTarget << ",\n";
    outFile << "    \"spatial_neighbors_considered\": " << auditDump.ris.spatialNeighborsConsidered << ",\n";
    outFile << "    \"spatial_neighbors_accepted\": " << auditDump.ris.spatialNeighborsAccepted << ",\n";
    outFile << "    \"spatial_rejected_depth\": " << auditDump.ris.spatialRejectedDepth << ",\n";
    outFile << "    \"spatial_rejected_normal\": " << auditDump.ris.spatialRejectedNormal << ",\n";
    outFile << "    \"spatial_rejected_invalid\": " << auditDump.ris.spatialRejectedInvalid << ",\n";
    outFile << "    \"spatial_winner_offset\": [" << auditDump.ris.spatialWinnerOffsetX
            << ", " << auditDump.ris.spatialWinnerOffsetY << "],\n";
    outFile << "    \"spatial_last_mis_weight\": " << auditDump.ris.spatialLastMisWeight << ",\n";
    outFile << "    \"spatial_last_merge_weight\": " << auditDump.ris.spatialLastMergeWeight << ",\n";
    outFile << "    \"temporal_reuse_attempted\": "
            << (auditDump.ris.temporalReuseAttempted ? "true" : "false") << ",\n";
    outFile << "    \"temporal_previous_available\": "
            << (auditDump.ris.temporalPreviousAvailable ? "true" : "false") << ",\n";
    outFile << "    \"temporal_reuse_accepted\": "
            << (auditDump.ris.temporalReuseAccepted ? "true" : "false") << ",\n";
    outFile << "    \"temporal_rejected_depth\": " << auditDump.ris.temporalRejectedDepth << ",\n";
    outFile << "    \"temporal_rejected_normal\": " << auditDump.ris.temporalRejectedNormal << ",\n";
    outFile << "    \"temporal_rejected_invalid\": " << auditDump.ris.temporalRejectedInvalid << ",\n";
    outFile << "    \"temporal_previous_primitive_index\": "
            << auditDump.ris.temporalPreviousPrimitiveIndex << ",\n";
    outFile << "    \"temporal_last_mis_weight\": " << auditDump.ris.temporalLastMisWeight << ",\n";
    outFile << "    \"temporal_last_merge_weight\": " << auditDump.ris.temporalLastMergeWeight << ",\n";
    outFile << "    \"world_reuse_attempted\": "
            << (auditDump.ris.worldReuseAttempted ? "true" : "false") << ",\n";
    outFile << "    \"world_reuse_accepted\": "
            << (auditDump.ris.worldReuseAccepted ? "true" : "false") << ",\n";
    outFile << "    \"world_cell\": [" << auditDump.ris.worldCellX
            << ", " << auditDump.ris.worldCellY
            << ", " << auditDump.ris.worldCellZ << "],\n";
    outFile << "    \"world_cell_hash\": " << auditDump.ris.worldCellHash << ",\n";
    outFile << "    \"world_candidates_considered\": " << auditDump.ris.worldCandidatesConsidered << ",\n";
    outFile << "    \"world_candidates_accepted\": " << auditDump.ris.worldCandidatesAccepted << ",\n";
    outFile << "    \"world_candidate_target\": " << auditDump.ris.worldCandidateTarget << ",\n";
    outFile << "    \"world_query_radius\": " << auditDump.ris.worldQueryRadius << ",\n";
    outFile << "    \"world_rejected_depth\": " << auditDump.ris.worldRejectedDepth << ",\n";
    outFile << "    \"world_rejected_normal\": " << auditDump.ris.worldRejectedNormal << ",\n";
    outFile << "    \"world_rejected_invalid\": " << auditDump.ris.worldRejectedInvalid << ",\n";
    outFile << "    \"world_rejected_cell\": " << auditDump.ris.worldRejectedCell << ",\n";
    outFile << "    \"world_candidate_primitive_index\": "
            << auditDump.ris.worldCandidatePrimitiveIndex << ",\n";
    outFile << "    \"world_last_mis_weight\": " << auditDump.ris.worldLastMisWeight << ",\n";
    outFile << "    \"world_last_merge_weight\": " << auditDump.ris.worldLastMergeWeight << ",\n";
    outFile << "    \"cache_reuse_attempted\": "
            << (auditDump.ris.cacheReuseAttempted ? "true" : "false") << ",\n";
    outFile << "    \"cache_state_available\": "
            << (auditDump.ris.cacheStateAvailable ? "true" : "false") << ",\n";
    outFile << "    \"cache_reuse_accepted\": "
            << (auditDump.ris.cacheReuseAccepted ? "true" : "false") << ",\n";
    outFile << "    \"cache_fallback_used\": "
            << (auditDump.ris.cacheFallbackUsed ? "true" : "false") << ",\n";
    outFile << "    \"cache_cell\": [" << auditDump.ris.cacheCellX
            << ", " << auditDump.ris.cacheCellY
            << ", " << auditDump.ris.cacheCellZ << "],\n";
    outFile << "    \"cache_cell_hash\": " << auditDump.ris.cacheCellHash << ",\n";
    outFile << "    \"cache_candidates_considered\": " << auditDump.ris.cacheCandidatesConsidered << ",\n";
    outFile << "    \"cache_entries_available\": " << auditDump.ris.cacheEntriesAvailable << ",\n";
    outFile << "    \"cache_candidates_accepted\": " << auditDump.ris.cacheCandidatesAccepted << ",\n";
    outFile << "    \"cache_candidate_target\": " << auditDump.ris.cacheCandidateTarget << ",\n";
    outFile << "    \"cache_rejected_depth\": " << auditDump.ris.cacheRejectedDepth << ",\n";
    outFile << "    \"cache_rejected_normal\": " << auditDump.ris.cacheRejectedNormal << ",\n";
    outFile << "    \"cache_rejected_invalid\": " << auditDump.ris.cacheRejectedInvalid << ",\n";
    outFile << "    \"cache_rejected_cell\": " << auditDump.ris.cacheRejectedCell << ",\n";
    outFile << "    \"cache_candidate_primitive_index\": "
            << auditDump.ris.cacheCandidatePrimitiveIndex << ",\n";
    outFile << "    \"cache_source_frame_index\": " << auditDump.ris.cacheSourceFrameIndex << ",\n";
    outFile << "    \"cache_last_mis_weight\": " << auditDump.ris.cacheLastMisWeight << ",\n";
    outFile << "    \"cache_last_merge_weight\": " << auditDump.ris.cacheLastMergeWeight << "\n";
    outFile << "  },\n";
    outFile << "  \"ris_candidates\": [\n";
    for (size_t i = 0; i < auditDump.risCandidates.size(); ++i) {
        const auto& candidate = auditDump.risCandidates[i];
        outFile << "    {\n";
        outFile << "      \"candidate_index\": " << candidate.candidateIndex << ",\n";
        outFile << "      \"primitive_index\": " << candidate.primitiveIndex << ",\n";
        outFile << "      \"valid\": " << (candidate.valid ? "true" : "false") << ",\n";
        outFile << "      \"rng_before\": " << candidate.rngBefore << ",\n";
        outFile << "      \"rng_after\": " << candidate.rngAfter << ",\n";
        outFile << "      \"pdf\": " << candidate.pdf << ",\n";
        outFile << "      \"phat\": " << candidate.phat << ",\n";
        outFile << "      \"weight\": " << candidate.weight << ",\n";
        outFile << "      \"reservoir_M\": " << candidate.reservoirM << ",\n";
        outFile << "      \"reservoir_w_sum\": " << candidate.reservoirWSum << "\n";
        outFile << "    }" << (i + 1u < auditDump.risCandidates.size() ? "," : "") << "\n";
    }
    outFile << "  ],\n";
    outFile << "  \"bsdf_path\": [\n";
    for (size_t i = 0; i < auditDump.bsdfPath.size(); ++i) {
        const auto& bsdf = auditDump.bsdfPath[i];
        outFile << "    {\n";
        outFile << "      \"depth\": " << bsdf.depth << ",\n";
        outFile << "      \"material_index\": " << bsdf.materialIndex << ",\n";
        outFile << "      \"front_face\": " << (bsdf.frontFace ? "true" : "false") << ",\n";
        outFile << "      \"is_delta\": " << (bsdf.isDelta ? "true" : "false") << ",\n";
        outFile << "      \"closure_type\": " << bsdf.closureType << ",\n";
        outFile << "      \"used_random_walk\": " << (bsdf.usedRandomWalk ? "true" : "false") << ",\n";
        outFile << "      \"rng_before\": " << bsdf.rngBefore << ",\n";
        outFile << "      \"rng_after\": " << bsdf.rngAfter << ",\n";
        outFile << "      \"shading_normal\": ["
                << bsdf.shadingNormal.x << ", "
                << bsdf.shadingNormal.y << ", "
                << bsdf.shadingNormal.z << "],\n";
        outFile << "      \"incident_direction\": ["
                << bsdf.incidentDirection.x << ", "
                << bsdf.incidentDirection.y << ", "
                << bsdf.incidentDirection.z << "],\n";
        outFile << "      \"wo\": ["
                << bsdf.wo.x << ", "
                << bsdf.wo.y << ", "
                << bsdf.wo.z << "],\n";
        outFile << "      \"sampled_direction\": ["
                << bsdf.sampledDirection.x << ", "
                << bsdf.sampledDirection.y << ", "
                << bsdf.sampledDirection.z << "],\n";
        outFile << "      \"pdf\": " << bsdf.pdf << ",\n";
        outFile << "      \"directional_pdf\": " << bsdf.directionalPdf << ",\n";
        outFile << "      \"lobe_type\": " << bsdf.lobeType << "\n";
        outFile << "    }" << (i + 1u < auditDump.bsdfPath.size() ? "," : "") << "\n";
    }
    outFile << "  ],\n";
    outFile << "  \"entries\": [\n";
    for (size_t i = 0; i < auditDump.entries.size(); ++i) {
        const auto& entry = auditDump.entries[i];
        const bool isEmissivePrimitive = entry.rectIndex == 0xFFFFFFFFu;
        const simd::float3 sampledPoint = auditDump.hitPosition + entry.lightDirection * entry.distance;
        const bool contributionFinite =
            std::isfinite(entry.contribution.x) &&
            std::isfinite(entry.contribution.y) &&
            std::isfinite(entry.contribution.z);
        const bool pdfFinite = std::isfinite(entry.lightPdf) && std::isfinite(entry.bsdfPdf);
        const bool directionFinite =
            std::isfinite(entry.lightDirection.x) &&
            std::isfinite(entry.lightDirection.y) &&
            std::isfinite(entry.lightDirection.z) &&
            std::isfinite(entry.distance);
        const bool sampleRejected =
            !entry.visible || !(entry.lightPdf > 0.0f) || !pdfFinite || !directionFinite || !contributionFinite;
        outFile << "    {\n";
        outFile << "      \"light_index\": " << entry.lightIndex << ",\n";
        outFile << "      \"rect_index\": " << entry.rectIndex << ",\n";
        outFile << "      \"light_kind\": \"" << (isEmissivePrimitive ? "emissive_primitive" : "rect") << "\",\n";
        outFile << "      \"sampled_primitive_index\": "
                << (isEmissivePrimitive ? entry.lightIndex : 0xFFFFFFFFu) << ",\n";
        outFile << "      \"sampled_point\": ["
                << sampledPoint.x << ", "
                << sampledPoint.y << ", "
                << sampledPoint.z << "],\n";
        outFile << "      \"sample_rejected\": " << (sampleRejected ? "true" : "false") << ",\n";
        outFile << "      \"light_direction\": [" << entry.lightDirection.x << ", " << entry.lightDirection.y << ", " << entry.lightDirection.z << "],\n";
        outFile << "      \"distance\": " << entry.distance << ",\n";
        outFile << "      \"n_dot_l\": " << entry.nDotL << ",\n";
        outFile << "      \"visible\": " << (entry.visible ? "true" : "false") << ",\n";
        outFile << "      \"bsdf_value\": [" << entry.bsdfValue.x << ", " << entry.bsdfValue.y << ", " << entry.bsdfValue.z << "],\n";
        outFile << "      \"light_pdf\": " << entry.lightPdf << ",\n";
        outFile << "      \"bsdf_pdf\": " << entry.bsdfPdf << ",\n";
        outFile << "      \"mis_weight\": " << entry.misWeight << ",\n";
        outFile << "      \"contribution\": [" << entry.contribution.x << ", " << entry.contribution.y << ", " << entry.contribution.z << "],\n";
        outFile << "      \"occluder_material_index\": " << entry.occluderMaterialIndex << ",\n";
        outFile << "      \"occluder_primitive_index\": " << entry.occluderPrimitiveIndex << ",\n";
        outFile << "      \"occluder_primitive_type\": " << entry.occluderPrimitiveType << ",\n";
        outFile << "      \"occluder_front_face\": " << (entry.occluderFrontFace ? "true" : "false") << ",\n";
        outFile << "      \"occluder_distance\": " << entry.occluderDistance << ",\n";
        outFile << "      \"shadow_path\": {\n";
        outFile << "        \"hw_raw_result_available\": "
                << (entry.shadowPath.hwRawResultAvailable ? "true" : "false") << ",\n";
        outFile << "        \"hw_trace_used\": "
                << (entry.shadowPath.hwTraceUsed ? "true" : "false") << ",\n";
        outFile << "        \"hw_raw_result_type\": " << entry.shadowPath.hwRawResultType << ",\n";
        outFile << "        \"hw_raw_instance_index\": " << entry.shadowPath.hwRawInstanceIndex << ",\n";
        outFile << "        \"hw_raw_primitive_index\": " << entry.shadowPath.hwRawPrimitiveIndex << ",\n";
        outFile << "        \"hw_raw_distance\": " << entry.shadowPath.hwRawDistance << ",\n";
        outFile << "        \"hw_resolved_hit\": "
                << (entry.shadowPath.hwResolvedHit ? "true" : "false") << ",\n";
        outFile << "        \"hw_resolved_mesh_index\": " << entry.shadowPath.hwResolvedMeshIndex << ",\n";
        outFile << "        \"hw_resolved_material_index\": " << entry.shadowPath.hwResolvedMaterialIndex << ",\n";
        outFile << "        \"hw_resolved_primitive_index\": " << entry.shadowPath.hwResolvedPrimitiveIndex << ",\n";
        outFile << "        \"hw_resolved_primitive_type\": " << entry.shadowPath.hwResolvedPrimitiveType << ",\n";
        outFile << "        \"hw_resolved_front_face\": "
                << (entry.shadowPath.hwResolvedFrontFace ? "true" : "false") << ",\n";
        outFile << "        \"hw_resolved_distance\": " << entry.shadowPath.hwResolvedDistance << ",\n";
        outFile << "        \"embedded_sw_consulted\": "
                << (entry.shadowPath.embeddedSwConsulted ? "true" : "false") << ",\n";
        outFile << "        \"embedded_sw_hit\": "
                << (entry.shadowPath.embeddedSwHit ? "true" : "false") << ",\n";
        outFile << "        \"embedded_sw_instance_index\": " << entry.shadowPath.embeddedSwInstanceIndex << ",\n";
        outFile << "        \"embedded_sw_mesh_index\": " << entry.shadowPath.embeddedSwMeshIndex << ",\n";
        outFile << "        \"embedded_sw_material_index\": " << entry.shadowPath.embeddedSwMaterialIndex << ",\n";
        outFile << "        \"embedded_sw_primitive_index\": " << entry.shadowPath.embeddedSwPrimitiveIndex << ",\n";
        outFile << "        \"embedded_sw_primitive_type\": " << entry.shadowPath.embeddedSwPrimitiveType << ",\n";
        outFile << "        \"embedded_sw_front_face\": "
                << (entry.shadowPath.embeddedSwFrontFace ? "true" : "false") << ",\n";
        outFile << "        \"embedded_sw_distance\": " << entry.shadowPath.embeddedSwDistance << ",\n";
        outFile << "        \"embedded_sw_hit_position\": ["
                << entry.shadowPath.embeddedSwHitPosition.x << ", "
                << entry.shadowPath.embeddedSwHitPosition.y << ", "
                << entry.shadowPath.embeddedSwHitPosition.z << "],\n";
        outFile << "        \"software_bvh_type\": " << entry.shadowPath.softwareBvhType << ",\n";
        outFile << "        \"software_tlas_available\": "
                << (entry.shadowPath.softwareTlasAvailable ? "true" : "false") << ",\n";
        outFile << "        \"software_blas_available\": "
                << (entry.shadowPath.softwareBlasAvailable ? "true" : "false") << ",\n";
        outFile << "        \"software_instance_info_available\": "
                << (entry.shadowPath.softwareInstanceInfoAvailable ? "true" : "false") << ",\n";
        outFile << "        \"sw_blas_root_node_offset\": " << entry.shadowPath.swBlasRootNodeOffset << ",\n";
        outFile << "        \"sw_blas_prim_index_offset\": " << entry.shadowPath.swBlasPrimIndexOffset << ",\n";
        outFile << "        \"sw_triangle_base_offset\": " << entry.shadowPath.swTriangleBaseOffset << ",\n";
        outFile << "        \"shadow_t_min\": " << entry.shadowPath.shadowTMin << ",\n";
        outFile << "        \"shadow_t_max\": " << entry.shadowPath.shadowTMax << ",\n";
        outFile << "        \"ray_origin_epsilon\": " << entry.shadowPath.rayOriginEpsilon << ",\n";
        outFile << "        \"hardware_occlusion_epsilon\": " << entry.shadowPath.hardwareOcclusionEpsilon << ",\n";
        outFile << "        \"rejection_mask\": " << entry.shadowPath.rejectionMask << ",\n";
        outFile << "        \"light_two_sided\": "
                << (entry.shadowPath.lightTwoSided ? "true" : "false") << ",\n";
        outFile << "        \"final_visible\": "
                << (entry.shadowPath.finalVisible ? "true" : "false") << "\n";
        outFile << "      }\n";
        outFile << "    }" << (i + 1u < auditDump.entries.size() ? ",\n" : "\n");
    }
    outFile << "  ],\n";
    outFile << "  \"non_finite_values\": " << (nonFinite.nanCount + nonFinite.infCount) << "\n";
    outFile << "}\n";
    ReportJsonNonFinite(nonFinite, "direct-light audit", jsonPath);
    return PublishAtomicSidecarStream(outFile, tmpPath, jsonPath, error, "direct-light audit JSON");
}

bool WriteLuminanceHistogramText(const fs::path& histogramPath,
                                 const ImagePbrMetrics& imageMetrics,
                                 std::string& error) {
    fs::path tmpPath;
    std::ofstream outFile;
    if (!OpenAtomicSidecarStream(histogramPath, tmpPath, outFile, error, "histogram text")) {
        return false;
    }

    outFile << std::fixed << std::setprecision(9);
    outFile << "# luminance_histogram type=log10 buckets="
            << imageMetrics.histogramCounts.size() << "\n";
    outFile << "# bin lower upper count\n";
    const size_t bucketCount = imageMetrics.histogramCounts.size();
    if (bucketCount == 0u) {
        return PublishAtomicSidecarStream(outFile, tmpPath, histogramPath, error, "histogram text");
    }
    if (imageMetrics.histogramEdges.size() < bucketCount + 1u) {
        error = "Histogram edges/count mismatch";
        return false;
    }
    for (size_t i = 0; i < bucketCount; ++i) {
        outFile << i << " "
                << imageMetrics.histogramEdges[i] << " "
                << imageMetrics.histogramEdges[i + 1u] << " "
                << imageMetrics.histogramCounts[i] << "\n";
    }

    return PublishAtomicSidecarStream(outFile, tmpPath, histogramPath, error, "histogram text");
}

void PrintPbrMetricsSummary(const ImagePbrMetrics& imageMetrics,
                            const HeadlessRenderOutput& output) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "[PBR] image: mean=" << imageMetrics.meanLuminance
              << " p50=" << imageMetrics.p50Luminance
              << " p95=" << imageMetrics.p95Luminance
              << " p99=" << imageMetrics.p99Luminance
              << " p99.9=" << imageMetrics.p999Luminance
              << " variance=" << imageMetrics.varianceLuminance
              << " max=" << imageMetrics.maxLuminance
              << " nanInfPixels=" << imageMetrics.nanInfPixelCount
              << " nanInfComponents=" << imageMetrics.nanInfComponentCount
              << std::endl;
    if (output.pbrMetrics.available) {
        const uint64_t totalSamples =
            static_cast<uint64_t>(output.width) * static_cast<uint64_t>(output.height) *
            static_cast<uint64_t>(output.samples);
        const double clampRatio =
            totalSamples > 0
                ? (static_cast<double>(output.pbrMetrics.clampEventCount) /
                   static_cast<double>(totalSamples))
                : 0.0;
        std::cout << "[PBR] renderer: throughputNaNInf="
                  << output.pbrMetrics.throughputNanInfCount
                  << " radianceNaNInf=" << output.pbrMetrics.radianceNanInfCount
                  << " clampEvents=" << output.pbrMetrics.clampEventCount
                  << " clampRatio=" << clampRatio
                  << " specNee(env=" << output.pbrMetrics.specNeeEnvAddedCount
                  << ",rect=" << output.pbrMetrics.specNeeRectAddedCount << ")"
                  << " mnee(eligible=" << output.pbrMetrics.mneeEligibleCount
                  << ",envAttempt=" << output.pbrMetrics.mneeEnvAttemptCount
                  << ",envAdded=" << output.pbrMetrics.mneeEnvAddedCount
                  << ",envChain[unsupported=" << output.pbrMetrics.mneeEnvChainUnsupportedCount
                  << ",nonDelta=" << output.pbrMetrics.mneeEnvChainNonDeltaBlockCount
                  << ",sampleReject=" << output.pbrMetrics.mneeEnvChainSampleRejectCount
                  << ",weightReject=" << output.pbrMetrics.mneeEnvChainWeightRejectCount
                  << ",maxBounce=" << output.pbrMetrics.mneeEnvChainMaxBounceCount
                  << "]"
                  << ",rect=" << output.pbrMetrics.mneeRectAddedCount
                  << ",contrib=" << output.pbrMetrics.mneeContributionCount << ")"
                  << " restirDiRegir(candidates="
                  << output.pbrMetrics.restirDiRegirCandidateCount
                  << ",accepted=" << output.pbrMetrics.restirDiRegirAcceptedCount
                  << ",fallback=" << output.pbrMetrics.restirDiRegirFallbackCount << ")"
                  << " restirGi(candidates=" << output.pbrMetrics.restirGiCandidateCount
                  << ",updates=" << output.pbrMetrics.restirGiReservoirUpdateCount
                  << ",accept=" << output.pbrMetrics.restirGiAcceptCount
                  << ",reject=" << output.pbrMetrics.restirGiRejectCount
                  << ",fallback=" << output.pbrMetrics.restirGiFallbackCount << ")"
                  << " pathGuiding(candidates=" << output.pbrMetrics.pathGuidingCandidateCount
                  << ",used=" << output.pbrMetrics.pathGuidingUsedCount
                  << ",fallback=" << output.pbrMetrics.pathGuidingFallbackCount
                  << ",invalid=" << output.pbrMetrics.pathGuidingInvalidCount << ")"
                  << " restirPt(candidates=" << output.pbrMetrics.restirPtCandidateCount
                  << ",updates=" << output.pbrMetrics.restirPtReservoirUpdateCount
                  << ",rejectedInvalid=" << output.pbrMetrics.restirPtRejectedInvalidCount
                  << ",rejectedUnsupported=" << output.pbrMetrics.restirPtRejectedUnsupportedCount
                  << ",debugRecords=" << output.pbrMetrics.restirPtDebugRecordCount
                  << ",reuseCandidates=" << output.pbrMetrics.restirPtReuseCandidateCount
                  << ",reuseApplied=" << output.pbrMetrics.restirPtReuseAppliedCount
                  << ",reuseRejectedInvalid=" << output.pbrMetrics.restirPtReuseRejectedInvalidCount
                  << ",reuseFallback=" << output.pbrMetrics.restirPtReuseFallbackCount << ")"
                  << " radianceCache(queries=" << output.pbrMetrics.radianceCacheQueryCount
                  << ",hits=" << output.pbrMetrics.radianceCacheHitCount
                  << ",misses=" << output.pbrMetrics.radianceCacheMissCount
                  << ",train=" << output.pbrMetrics.radianceCacheTrainCount
                  << ",fallback=" << output.pbrMetrics.radianceCacheFallbackCount
                  << ",invalid=" << output.pbrMetrics.radianceCacheInvalidCount << ")"
                  << " caustics(vertices=" << output.pbrMetrics.causticEyeVertexCount
                  << ",lightVertices=" << output.pbrMetrics.causticLightVertexCount
                  << ",connections=" << output.pbrMetrics.causticConnectionCandidateCount
                  << ",deltaChains=" << output.pbrMetrics.causticDeltaChainCount
                  << ",debug=" << output.pbrMetrics.causticDebugClassificationCount << ")"
                  << " volume(transmittance=" << output.pbrMetrics.volumeTransmittanceQueryCount
                  << ",scatter=" << output.pbrMetrics.volumeScatterEventCount
                  << ",nee=" << output.pbrMetrics.volumeNeeRayCount
                  << ",phase=" << output.pbrMetrics.volumePhaseSampleCount
                  << ",boundary=" << output.pbrMetrics.volumeMediumBoundaryEventCount
                  << ",fallback=" << output.pbrMetrics.volumeFallbackCount << ")"
                  << " spectral(wavelengths=" << output.pbrMetrics.spectralWavelengthSampleCount
                  << ",materials=" << output.pbrMetrics.spectralMaterialLookupCount
                  << ",dispersion=" << output.pbrMetrics.spectralDispersionEventCount
                  << ",fallback=" << output.pbrMetrics.spectralFallbackCount
                  << ",collapses=" << output.pbrMetrics.spectralCollapseCount
                  << ",channelSamples=" << output.pbrMetrics.spectralWavelengthChannelSampleCount << ")"
                  << std::endl;
        PrintRadianceCacheStatsLine(output.pbrMetrics);
    } else {
        std::cout << "[PBR] renderer: unavailable for current backend" << std::endl;
    }
}

bool WriteRawFloatImage(const fs::path& path,
                        const std::vector<float>& rgba,
                        std::string& error) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        error = "Failed to open raw texture dump for writing: " + path.string();
        return false;
    }
    if (!rgba.empty()) {
        out.write(reinterpret_cast<const char*>(rgba.data()),
                  static_cast<std::streamsize>(rgba.size() * sizeof(float)));
        if (!out.good()) {
            error = "Failed to write raw texture dump: " + path.string();
            return false;
        }
    }
    return true;
}

std::vector<float> ExtractRgbPreview(const std::vector<float>& rgba) {
    std::vector<float> rgb;
    rgb.resize((rgba.size() / 4u) * 3u, 0.0f);
    for (size_t i = 0, j = 0; i + 3u < rgba.size(); i += 4u, j += 3u) {
        rgb[j + 0u] = rgba[i + 0u];
        rgb[j + 1u] = rgba[i + 1u];
        rgb[j + 2u] = rgba[i + 2u];
    }
    return rgb;
}

std::vector<float> ExtractAlphaPreview(const std::vector<float>& rgba) {
    std::vector<float> alphaRgb;
    alphaRgb.resize((rgba.size() / 4u) * 3u, 0.0f);
    for (size_t i = 0, j = 0; i + 3u < rgba.size(); i += 4u, j += 3u) {
        const float alpha = rgba[i + 3u];
        alphaRgb[j + 0u] = alpha;
        alphaRgb[j + 1u] = alpha;
        alphaRgb[j + 2u] = alpha;
    }
    return alphaRgb;
}

bool WritePreviewImage(const fs::path& path,
                       uint32_t width,
                       uint32_t height,
                       const std::vector<float>& linearRgb,
                       std::string& error) {
    PathTracer::TonemapSettings tonemap{};
    tonemap.tonemapMode = 1;
    tonemap.exposure = 0.0f;
    return PathTracer::WriteImage(path.string(),
                                  PathTracer::ImageFileFormat::PNG,
                                  linearRgb.data(),
                                  width,
                                  height,
                                  tonemap,
                                  &error);
}

const char* BoolLiteral(bool value) {
    return value ? "true" : "false";
}

bool DumpMaterialTextureDiagnostics(const PathTracer::SceneResources& resources,
                                    const fs::path& outDir,
                                    std::string& error) {
    std::error_code dirEc;
    fs::create_directories(outDir, dirEc);
    if (dirEc) {
        error = "Failed to create material texture dump directory '" + outDir.string() +
                "': " + dirEc.message();
        return false;
    }

    const auto& inventory = resources.textureInventory();
    const auto* materials = resources.materialsData();
    const auto& meshes = resources.meshes();
    const uint32_t textureCount = resources.materialTextureCount();

    const fs::path manifestPath = outDir / "manifest.json";
    fs::path manifestTmpPath;
    std::ofstream json;
    if (!OpenAtomicSidecarStream(manifestPath,
                                 manifestTmpPath,
                                 json,
                                 error,
                                 "material texture manifest JSON")) {
        return false;
    }

    json << std::fixed << std::setprecision(9);
    json << "{\n";
    json << "  \"texture_count\": " << textureCount << ",\n";
    json << "  \"textures\": [\n";
    for (uint32_t index = 0; index < textureCount; ++index) {
        std::cout << "[MaterialTextureDump] texture " << index << " begin" << std::endl;
        PathTracer::MaterialTextureCpuData cpuData;
        const bool hasCpuData = resources.materialTextureCpuData(index, cpuData);
        std::cout << "[MaterialTextureDump] texture " << index
                  << " cpuData=" << (hasCpuData ? "true" : "false") << std::endl;
        const auto* entry = index < inventory.size() ? &inventory[index] : nullptr;
        const fs::path textureDir = outDir / ("texture_" + std::to_string(index));
        if (hasCpuData) {
            fs::create_directories(textureDir, dirEc);
            if (dirEc) {
                error = "Failed to create texture dump directory '" + textureDir.string() +
                        "': " + dirEc.message();
                return false;
            }
        }

        json << "    {\n";
        json << "      \"index\": " << index << ",\n";
        json << "      \"label\": \"" << JsonEscape(entry ? entry->label : std::string()) << "\",\n";
        json << "      \"semantic\": \"" << TextureSemanticLabel(entry ? entry->semantic
                                                                  : PathTracer::MaterialTextureSemantic::Generic)
             << "\",\n";
        json << "      \"format\": \"" << JsonEscape(entry ? entry->formatLabel : std::string()) << "\",\n";
        json << "      \"width\": " << (hasCpuData ? cpuData.width : 0u) << ",\n";
        json << "      \"height\": " << (hasCpuData ? cpuData.height : 0u) << ",\n";
        json << "      \"mip_count\": " << (hasCpuData ? cpuData.mipLevels.size() : 0u) << ",\n";
        json << "      \"srgb\": " << BoolLiteral(hasCpuData && cpuData.srgb) << ",\n";
        json << "      \"sampler\": {\n";
        json << "        \"magFilter\": " << cpuData.samplerDesc.magFilter << ",\n";
        json << "        \"minFilter\": " << cpuData.samplerDesc.minFilter << ",\n";
        json << "        \"wrapS\": " << cpuData.samplerDesc.wrapS << ",\n";
        json << "        \"wrapT\": " << cpuData.samplerDesc.wrapT << "\n";
        json << "      },\n";
        json << "      \"mips\": [\n";
        for (size_t level = 0; hasCpuData && level < cpuData.mipLevels.size(); ++level) {
            const auto& mip = cpuData.mipLevels[level];
            const fs::path rawPath = textureDir / ("mip_" + std::to_string(level) + ".rgba32f");
            const fs::path rgbPreviewPath = textureDir / ("mip_" + std::to_string(level) + "_rgb.png");
            const fs::path alphaPreviewPath = textureDir / ("mip_" + std::to_string(level) + "_alpha.png");
            std::cout << "[MaterialTextureDump] texture " << index
                      << " mip " << level
                      << " raw" << std::endl;
            if (!WriteRawFloatImage(rawPath, mip.rgba, error)) {
                return false;
            }
            std::cout << "[MaterialTextureDump] texture " << index
                      << " mip " << level
                      << " rgb_preview" << std::endl;
            if (!WritePreviewImage(rgbPreviewPath,
                                   mip.width,
                                   mip.height,
                                   ExtractRgbPreview(mip.rgba),
                                   error)) {
                return false;
            }
            std::cout << "[MaterialTextureDump] texture " << index
                      << " mip " << level
                      << " alpha_preview" << std::endl;
            if (!WritePreviewImage(alphaPreviewPath,
                                   mip.width,
                                   mip.height,
                                   ExtractAlphaPreview(mip.rgba),
                                   error)) {
                return false;
            }

            json << "        {\n";
            json << "          \"level\": " << level << ",\n";
            json << "          \"width\": " << mip.width << ",\n";
            json << "          \"height\": " << mip.height << ",\n";
            json << "          \"rgba32f_path\": \"" << JsonEscape(rawPath.string()) << "\",\n";
            json << "          \"rgb_preview_path\": \"" << JsonEscape(rgbPreviewPath.string()) << "\",\n";
            json << "          \"alpha_preview_path\": \"" << JsonEscape(alphaPreviewPath.string()) << "\"\n";
            json << "        }";
            if (level + 1u < cpuData.mipLevels.size()) {
                json << ",";
            }
            json << "\n";
        }
        json << "      ]\n";
        json << "    }";
        if (index + 1u < textureCount) {
            json << ",";
        }
        json << "\n";
        std::cout << "[MaterialTextureDump] texture " << index << " done" << std::endl;
    }
    json << "  ],\n";

    const uint32_t materialCount = resources.materialCount();
    json << "  \"materials\": [\n";
    for (uint32_t index = 0; index < materialCount; ++index) {
        const auto& material = materials[index];
        json << "    {\n";
        json << "      \"index\": " << index << ",\n";
        json << "      \"name\": \"" << JsonEscape(resources.materialName(index)) << "\",\n";
        json << "      \"material_flags\": " << material.materialFlags << ",\n";
        json << "      \"material_flags_decoded\": {\n";
        json << "        \"disable_orm\": "
             << BoolLiteral((material.materialFlags & PathTracerShaderTypes::kMaterialFlagDisableOrm) != 0u)
             << ",\n";
        json << "        \"force_basecolor_mip0\": "
             << BoolLiteral((material.materialFlags & PathTracerShaderTypes::kMaterialFlagForceBaseColorMip0) != 0u)
             << ",\n";
        json << "        \"black_key_alpha_from_rgb\": "
             << BoolLiteral((material.materialFlags & PathTracerShaderTypes::kMaterialFlagBlackKeyAlphaFromRgb) != 0u)
             << "\n";
        json << "      },\n";
        json << "      \"texture_indices0\": [" << material.textureIndices0.x << ", "
             << material.textureIndices0.y << ", " << material.textureIndices0.z << ", "
             << material.textureIndices0.w << "],\n";
        json << "      \"texture_indices1\": [" << material.textureIndices1.x << ", "
             << material.textureIndices1.y << ", " << material.textureIndices1.z << ", "
             << material.textureIndices1.w << "],\n";
        json << "      \"base_color_factor\": [" << material.baseColorRoughness.x << ", "
             << material.baseColorRoughness.y << ", " << material.baseColorRoughness.z << ", "
             << material.pbrExtras.x << "],\n";
        json << "      \"texture_uv_set0\": [" << material.textureUvSet0.x << ", "
             << material.textureUvSet0.y << ", " << material.textureUvSet0.z << ", "
             << material.textureUvSet0.w << "],\n";
        json << "      \"base_color_transform_row0\": [" << material.textureTransform0.x << ", "
             << material.textureTransform0.y << ", " << material.textureTransform0.z << ", "
             << material.textureTransform0.w << "],\n";
        json << "      \"base_color_transform_row1\": [" << material.textureTransform1.x << ", "
             << material.textureTransform1.y << ", " << material.textureTransform1.z << ", "
             << material.textureTransform1.w << "],\n";
        json << "      \"texture_uv_set1\": [" << material.textureUvSet1.x << ", "
             << material.textureUvSet1.y << ", " << material.textureUvSet1.z << ", "
             << material.textureUvSet1.w << "]\n";
        json << "    }";
        if (index + 1u < materialCount) {
            json << ",";
        }
        json << "\n";
    }
    json << "  ],\n";

    json << "  \"meshes\": [\n";
    for (size_t index = 0; index < meshes.size(); ++index) {
        const auto& mesh = meshes[index];
        json << "    {\n";
        json << "      \"index\": " << index << ",\n";
        json << "      \"name\": \"" << JsonEscape(mesh.name) << "\",\n";
        json << "      \"material_index\": " << mesh.materialIndex << ",\n";
        json << "      \"material_name\": \"" << JsonEscape(resources.materialName(mesh.materialIndex)) << "\",\n";
        json << "      \"triangle_count\": " << (mesh.indices.size() / 3u) << "\n";
        json << "    }";
        if (index + 1u < meshes.size()) {
            json << ",";
        }
        json << "\n";
    }
    json << "  ]\n";
    json << "}\n";
    return PublishAtomicSidecarStream(json,
                                      manifestTmpPath,
                                      manifestPath,
                                      error,
                                      "material texture manifest JSON");
}

bool FindRunRenderQueueArgument(int argc, const char** argv, std::string& path, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string value;
        const auto eqPos = arg.find('=');
        const bool hasInlineValue = eqPos != std::string::npos;
        if (hasInlineValue) {
            value = arg.substr(eqPos + 1);
            arg = arg.substr(0, eqPos);
        }
        if (arg != "--runRenderQueueItemJson" && arg != "--renderQueueItem") {
            continue;
        }
        if (hasInlineValue) {
            path = value;
            return true;
        }
        if (i + 1 >= argc) {
            error = "Missing value for " + arg;
            return false;
        }
        path = argv[i + 1];
        return true;
    }
    return false;
}

bool LoadRenderQueueBatchItems(const fs::path& path,
                               std::vector<std::string>& itemJsons,
                               std::string& summaryPath,
                               bool& isBatch,
                               std::string& error) {
    isBatch = false;
    NSData* data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.string().c_str()]];
    if (!data) {
        error = "Failed to read render queue JSON: " + path.string();
        return false;
    }

    NSError* nsError = nil;
    id root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&nsError];
    if (!root) {
        error = "Failed to parse render queue JSON: " + path.string();
        if (nsError) {
            error += " (" + NSStringToStdString([nsError localizedDescription]) + ")";
        }
        return false;
    }

    NSArray* items = nil;
    if ([root isKindOfClass:[NSArray class]]) {
        items = (NSArray*)root;
    } else if ([root isKindOfClass:[NSDictionary class]]) {
        NSDictionary* dict = (NSDictionary*)root;
        id queueItems = [dict objectForKey:@"queue_items"];
        if ([queueItems isKindOfClass:[NSArray class]]) {
            items = (NSArray*)queueItems;
        }
        if (auto explicitSummary = JsonString(dict, @"batch_summary_json")) {
            summaryPath = *explicitSummary;
        }
    }

    if (!items) {
        return true;
    }
    isBatch = true;
    if ([items count] == 0u) {
        error = "Render queue batch has no queue_items: " + path.string();
        return false;
    }
    if (summaryPath.empty()) {
        summaryPath = path.string() + ".summary.json";
    }

    for (NSUInteger index = 0; index < [items count]; ++index) {
        id item = [items objectAtIndex:index];
        if (![item isKindOfClass:[NSDictionary class]]) {
            error = "Render queue batch item " + std::to_string(static_cast<size_t>(index)) +
                    " must be a JSON object.";
            return false;
        }
        NSData* itemData = [NSJSONSerialization dataWithJSONObject:item
                                                           options:NSJSONWritingPrettyPrinted
                                                             error:&nsError];
        if (!itemData) {
            error = "Failed to serialize render queue batch item " +
                    std::to_string(static_cast<size_t>(index));
            if (nsError) {
                error += " (" + NSStringToStdString([nsError localizedDescription]) + ")";
            }
            return false;
        }
        NSString* itemString = [[NSString alloc] initWithData:itemData encoding:NSUTF8StringEncoding];
        itemJsons.push_back(NSStringToStdString(itemString));
    }
    return true;
}

bool IsRunRenderQueueArgument(const std::string& arg) {
    const auto eqPos = arg.find('=');
    const std::string name = eqPos == std::string::npos ? arg : arg.substr(0, eqPos);
    return name == "--runRenderQueueItemJson" || name == "--renderQueueItem";
}

std::vector<std::string> BuildBatchChildArguments(int argc,
                                                  const char** argv,
                                                  const fs::path& itemPath) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool runQueueArg = IsRunRenderQueueArgument(arg);
        if (runQueueArg) {
            if (arg.find('=') == std::string::npos && i + 1 < argc) {
                ++i;
            }
            continue;
        }
        args.push_back(arg);
    }
    args.emplace_back("--runRenderQueueItemJson");
    args.push_back(itemPath.string());
    return args;
}

int RunRenderQueueBatch(int argc,
                        const char** argv,
                        const fs::path& batchPath,
                        const std::vector<std::string>& itemJsons,
                        const fs::path& summaryPath) {
    struct BatchItemResult {
        fs::path itemJsonPath;
        int exitCode = 1;
    };

    std::vector<BatchItemResult> results;
    results.reserve(itemJsons.size());

    fs::path executablePath(argv[0]);
    if (executablePath.is_relative()) {
        executablePath = fs::absolute(executablePath);
    }

    const fs::path batchTempDir =
        fs::temp_directory_path() /
        ("pathtracer_render_queue_" + NSStringToStdString([[NSUUID UUID] UUIDString]));
    std::error_code dirEc;
    fs::create_directories(batchTempDir, dirEc);
    if (dirEc) {
        std::cerr << "Failed to create render queue batch temp directory '"
                  << batchTempDir << "': " << dirEc.message() << std::endl;
        return 1;
    }

    uint32_t failedCount = 0;
    for (size_t index = 0; index < itemJsons.size(); ++index) {
        const fs::path itemPath =
            batchTempDir / ("queue_item_" + std::to_string(index) + ".json");
        std::string writeError;
        if (!WriteTextFileAtomic(itemPath, itemJsons[index], writeError)) {
            std::cerr << writeError << std::endl;
            ++failedCount;
            results.push_back(BatchItemResult{itemPath, 1});
            continue;
        }

        std::vector<std::string> childArgs = BuildBatchChildArguments(argc, argv, itemPath);
        NSMutableArray* nsArgs = [NSMutableArray arrayWithCapacity:childArgs.size()];
        for (const std::string& childArg : childArgs) {
            [nsArgs addObject:[NSString stringWithUTF8String:childArg.c_str()]];
        }

        NSTask* task = [[NSTask alloc] init];
        task.executableURL =
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:executablePath.string().c_str()]];
        task.arguments = nsArgs;
        NSError* launchError = nil;
        std::cout << "[RenderQueueBatch] item " << index << " start" << std::endl;
        if (![task launchAndReturnError:&launchError]) {
            std::cerr << "[RenderQueueBatch] item " << index
                      << " launch failed: "
                      << (launchError ? NSStringToStdString([launchError localizedDescription])
                                      : std::string("unknown error"))
                      << std::endl;
            ++failedCount;
            results.push_back(BatchItemResult{itemPath, 1});
            continue;
        }
        [task waitUntilExit];
        const int exitCode = [task terminationStatus];
        if (exitCode != 0) {
            ++failedCount;
        }
        std::cout << "[RenderQueueBatch] item " << index
                  << " exit_code=" << exitCode << std::endl;
        results.push_back(BatchItemResult{itemPath, exitCode});
    }

    std::ostringstream summary;
    summary << "{\n";
    summary << "  \"schema_version\": \"pathtracer.render_queue_batch_summary.v1\",\n";
    summary << "  \"batch_json\": \"" << JsonEscape(batchPath.string()) << "\",\n";
    summary << "  \"item_count\": " << results.size() << ",\n";
    summary << "  \"failed_count\": " << failedCount << ",\n";
    summary << "  \"items\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        summary << "    {\"index\": " << i
                << ", \"queue_item_json\": \"" << JsonEscape(results[i].itemJsonPath.string())
                << "\", \"exit_code\": " << results[i].exitCode
                << ", \"status\": \"" << (results[i].exitCode == 0 ? "succeeded" : "failed")
                << "\"}";
        if (i + 1u < results.size()) {
            summary << ",";
        }
        summary << "\n";
    }
    summary << "  ]\n";
    summary << "}\n";

    std::string summaryError;
    if (!WriteTextFileAtomic(summaryPath, summary.str(), summaryError)) {
        std::cerr << summaryError << std::endl;
        return 1;
    }
    std::cout << "[RenderQueueBatch] summary written to: " << summaryPath << std::endl;

    return failedCount == 0u ? 0 : 1;
}

int main(int argc, const char** argv) {
    @autoreleasepool {
        std::string batchQueuePath;
        std::string batchProbeError;
        const bool hasRunQueueArg =
            FindRunRenderQueueArgument(argc, argv, batchQueuePath, batchProbeError);
        if (!batchProbeError.empty()) {
            std::cerr << "Error: " << batchProbeError << std::endl;
            return 1;
        }
        if (hasRunQueueArg) {
            std::vector<std::string> batchItems;
            std::string summaryPathString;
            bool isBatch = false;
            std::string batchError;
            if (!LoadRenderQueueBatchItems(fs::path(batchQueuePath),
                                           batchItems,
                                           summaryPathString,
                                           isBatch,
                                           batchError)) {
                std::cerr << "Error: " << batchError << std::endl;
                return 1;
            }
            if (isBatch) {
                return RunRenderQueueBatch(argc,
                                           argv,
                                           fs::path(batchQueuePath),
                                           batchItems,
                                           fs::path(summaryPathString));
            }
        }

        CliOptions options;
        std::string error;
        if (!ParseOptions(argc, argv, options, error)) {
            if (!error.empty()) {
                std::cerr << "Error: " << error << "\n\n";
            }
            PrintUsage(argv[0]);
            return 1;
        }

        if (options.backend == HeadlessBackend::Embree) {
#if !defined(PATH_TRACER_ENABLE_EMBREE)
            std::cerr << "Embree backend requested, but this binary was built without Embree support. "
                      << "Reconfigure with -DPATH_TRACER_ENABLE_EMBREE=ON." << std::endl;
            return 1;
#endif
        }

        const bool useMetalSceneContext = options.backend == HeadlessBackend::Metal;

        PathTracer::SceneManager sceneManager;
        PathTracer::MetalContext statsContext;
        if (useMetalSceneContext && !statsContext.initialize(nil, true)) {
            std::cerr << "Failed to initialize headless Metal context for scene analysis." << std::endl;
            return 1;
        }

        PathTracer::SceneResources sceneResources;
        if (useMetalSceneContext) {
            sceneResources.initialize(statsContext);
        } else {
            sceneResources.initialize(nil, nil, false);
        }
        PathTracer::TextureLoadPolicy textureLoadPolicy{};
        textureLoadPolicy.policy = options.budgetPolicy;
        textureLoadPolicy.pilotMode = options.pilotMode;
        textureLoadPolicy.maxDimension = options.textureMaxDimSet ? options.textureMaxDim : 0u;
        textureLoadPolicy.maxTextureBytes = options.maxTextureBytesSet ? options.maxTextureBytes : 0u;
        sceneResources.setTextureLoadPolicy(textureLoadPolicy);
        PathTracer::RenderSettings settings{};

        bool sceneLoaded = false;
        std::string sceneError;
        if (options.sceneIsPath) {
            sceneLoaded = sceneManager.loadSceneFromPath(options.scene, sceneResources, settings, &sceneError);
        } else {
            sceneLoaded = sceneManager.loadScene(options.scene, sceneResources, settings, &sceneError);
        }

        if (!sceneLoaded) {
            std::cerr << "Failed to load scene: " << options.scene << std::endl;
            if (!sceneError.empty()) {
                std::cerr << sceneError << std::endl;
            }
            const auto& scenes = sceneManager.scenes();
            if (!scenes.empty()) {
                std::cerr << "Available scenes:" << std::endl;
                for (const auto& info : scenes) {
                    std::cerr << "  " << info.identifier << std::endl;
                }
            }
            return 1;
        }

        PrintSceneIdentityReport(options, sceneManager, sceneResources);

        ApplyRenderProfileDefaults(options, settings);
        ApplyCliOverrides(options, settings);
        ApplyPilotModeOverrides(sceneResources, settings, options.pilotMode);
        std::cout << "[RenderMode] mode=" << RenderModeLabel(options.renderMode)
                  << " profile=" << RenderProfileLabel(options.renderProfile)
                  << " backend=" << (options.backend == HeadlessBackend::Embree ? "embree" : "metal")
                  << " execution="
                  << (settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront
                          ? "wavefront"
                          : "megakernel")
                  << std::endl;
        if (const char* refusal = PathTracer::RadianceCacheRefusalReason(settings)) {
            std::cerr << "[RadianceCache] refused (" << refusal << "): "
                      << PathTracer::RadianceCacheRefusalMessage(refusal) << ". The cache is off for this render."
                      << std::endl;
        }
        // restir_di's explicit passes on the megakernel are screen passes that read no material textures, normal
        // maps or occlusion and give wrong light (0.0625 of the exact value on an analytic case): refused, not run.
        // The wavefront's explicit stage, the hybrid's on the megakernel and the fused restir_di are exact. The test
        // mirrors RenderLoop's (PT_RESTIR_DI_VALIDATE_FUSED_REFERENCE keeps the fused path, homogeneous volumes render
        // baseline_emissive instead of restir_di, --validate-ris and --bias-check render their own modes) and runs
        // before the acceleration structures build.
        if (options.backend == HeadlessBackend::Metal && !options.validateRis && !options.biasCheck &&
            settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Megakernel &&
            settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RestirDi &&
            settings.volumeTransportMode != PathTracer::RenderSettings::VolumeTransportMode::Homogeneous &&
            settings.restirDebugEnabled && !EnvFlagEnabled("PT_RESTIR_DI_VALIDATE_FUSED_REFERENCE")) {
            std::cerr << "[ReSTIR DI] the explicit passes are not supported on the megakernel: its screen passes "
                         "ignore material textures, normal maps and occlusion. They are on through --restirDebug=1, "
                         "a --restirDebugView other than beauty, --restirDebugMetricsJson or --profile=debug. Use "
                         "--executionMode=wavefront for the explicit passes, or restir_di without them; refusing to "
                         "render." << std::endl;
            if (options.outputMetadata &&
                (options.outputMetadataJsonProvided || !options.outputPath.empty())) {
                const fs::path refusalPath = options.outputMetadataJsonProvided
                    ? fs::path(options.outputMetadataJsonPath)
                    : fs::path(options.outputPath + ".metadata.json");
                std::string refusalError;
                if (!WriteTextFileAtomic(refusalPath,
                                         "{\n  \"status\": \"refused\",\n"
                                         "  \"restir_di_refused_combination\": "
                                         "\"explicit passes on the megakernel\"\n}\n",
                                         refusalError)) {
                    std::cerr << refusalError << std::endl;
                }
            }
            return 2;
        }

        if (options.debugLandmarks || options.cameraSearchReportProvided) {
            std::vector<RuntimeMeshSnapshot> snapshots;
            snapshots.reserve(sceneResources.meshes().size());
            for (const auto& mesh : sceneResources.meshes()) {
                snapshots.push_back(MakeRuntimeMeshSnapshot(mesh));
            }

            const auto facadeMatches = CollectMatchingSnapshots(
                snapshots,
                {"bistroawning", "bistrofrontbanner", "_bistro_bottom_", "bistrofront", "bistrofacade"},
                std::nullopt,
                12);
            const RuntimeLandmark facade = UnionSnapshotBounds("facade", facadeMatches);
            const std::optional<simd::float3> anchorPoint = facade.bounds.valid
                ? std::optional<simd::float3>(facade.bounds.center)
                : std::nullopt;

            const auto scooterMatches = CollectMatchingSnapshots(
                snapshots,
                {"__lod0_vespa_", "vespa", "scooter", "moped", "motorbike", "bike", "vehicle"},
                anchorPoint,
                12);
            const auto scooter = SelectNearestLandmark(scooterMatches, scooterMatches.empty() ? "scooter" : scooterMatches.front()->mesh->name);

            const std::optional<simd::float3> landmarkAnchor = scooter
                ? std::optional<simd::float3>(scooter->bounds.center)
                : anchorPoint;

            const auto lampMatches = CollectMatchingSnapshots(
                snapshots,
                {"streetlight_01a", "streetlight", "lamp", "lantern", "lightpost", "lampost", "post"},
                landmarkAnchor,
                12);
            std::vector<const RuntimeMeshSnapshot*> leftLampMatches;
            std::vector<const RuntimeMeshSnapshot*> rightLampMatches;
            if (landmarkAnchor) {
                for (const RuntimeMeshSnapshot* snapshot : lampMatches) {
                    if (snapshot->renderWorldBounds.center.x <= landmarkAnchor->x) {
                        leftLampMatches.push_back(snapshot);
                    } else {
                        rightLampMatches.push_back(snapshot);
                    }
                }
            }
            const auto leftLamp = SelectNearestLandmark(leftLampMatches, leftLampMatches.empty() ? "left_lamp" : leftLampMatches.front()->mesh->name);
            const auto rightLamp = SelectNearestLandmark(rightLampMatches, rightLampMatches.empty() ? "right_lamp" : rightLampMatches.front()->mesh->name);

            const auto curbMatches = CollectMatchingSnapshots(
                snapshots,
                {"paris_street_6179", "paris_street_6181", "paris_street", "curb", "island", "sidewalk", "pavement", "road", "street"},
                landmarkAnchor,
                12);
            const RuntimeLandmark curb = UnionSnapshotBounds("curb", curbMatches);

            const auto streetOpeningMatches = CollectMatchingSnapshots(
                snapshots,
                {"paris_street_6183", "paris_street", "street", "road", "alley", "lane"},
                landmarkAnchor,
                10);
            const auto streetOpening = SelectNearestLandmark(
                streetOpeningMatches,
                streetOpeningMatches.empty() ? "street_opening" : streetOpeningMatches.front()->mesh->name);

            auto dumpSnapshot = [&](const char* label, const RuntimeMeshSnapshot* snapshot) {
                if (!snapshot || !snapshot->mesh) {
                    std::cout << "  " << label << ": <not found>\n";
                    return;
                }
                std::cout << "  " << label << ": " << snapshot->mesh->name << "\n";
                std::cout << "    source_local_transform: " << FormatMatrix4x4(snapshot->mesh->sourceLocalTransform) << "\n";
                std::cout << "    imported_world_transform: " << FormatMatrix4x4(snapshot->mesh->defaultLocalToWorld) << "\n";
                std::cout << "    renderer_world_transform: " << FormatMatrix4x4(snapshot->mesh->localToWorld) << "\n";
                std::cout << "    render_transform_differs_from_imported: "
                          << (MatrixNearlyEqual(snapshot->mesh->localToWorld, snapshot->mesh->defaultLocalToWorld) ? "false" : "true")
                          << "\n";
                std::cout << "    local_bounds_min: " << FormatFloat3(snapshot->localBounds.min) << "\n";
                std::cout << "    local_bounds_max: " << FormatFloat3(snapshot->localBounds.max) << "\n";
                std::cout << "    imported_world_bounds_min: " << FormatFloat3(snapshot->importedWorldBounds.min) << "\n";
                std::cout << "    imported_world_bounds_max: " << FormatFloat3(snapshot->importedWorldBounds.max) << "\n";
                std::cout << "    renderer_world_bounds_min: " << FormatFloat3(snapshot->renderWorldBounds.min) << "\n";
                std::cout << "    renderer_world_bounds_max: " << FormatFloat3(snapshot->renderWorldBounds.max) << "\n";
                std::cout << "    renderer_world_center: " << FormatFloat3(snapshot->renderWorldBounds.center) << "\n";
                std::cout << "    tlas_world_bounds_min: " << FormatFloat3(snapshot->tlasWorldBounds.min) << "\n";
                std::cout << "    tlas_world_bounds_max: " << FormatFloat3(snapshot->tlasWorldBounds.max) << "\n";
                std::cout << "    tlas_world_center: " << FormatFloat3(snapshot->tlasWorldBounds.center) << "\n";
            };

            auto dumpCandidateList = [&](const char* label, const std::vector<const RuntimeMeshSnapshot*>& matches) {
                std::cout << "  candidates " << label << ": " << matches.size() << "\n";
                for (const RuntimeMeshSnapshot* snapshot : matches) {
                    std::cout << "    - " << snapshot->mesh->name
                              << " center=" << FormatFloat3(snapshot->renderWorldBounds.center);
                    if (landmarkAnchor) {
                        std::cout << " distance_to_scooter="
                                  << std::fixed << std::setprecision(6)
                                  << BoundsDistance(snapshot->renderWorldBounds, *landmarkAnchor);
                    }
                    std::cout << "\n";
                }
            };

            std::cout << "[RuntimeLandmarks] renderer-visible transform path\n";
            dumpSnapshot("scooter", scooterMatches.empty() ? nullptr : scooterMatches.front());
            dumpCandidateList("lamp_posts", lampMatches);
            dumpSnapshot("left_lamp_candidate", leftLampMatches.empty() ? nullptr : leftLampMatches.front());
            dumpSnapshot("right_lamp_candidate", rightLampMatches.empty() ? nullptr : rightLampMatches.front());
            dumpCandidateList("facade_parts", facadeMatches);
            if (facade.bounds.valid) {
                std::cout << "  facade_union\n"
                          << "    renderer_world_bounds_min: " << FormatFloat3(facade.bounds.min) << "\n"
                          << "    renderer_world_bounds_max: " << FormatFloat3(facade.bounds.max) << "\n"
                          << "    renderer_world_center: " << FormatFloat3(facade.bounds.center) << "\n";
            } else {
                std::cout << "  facade_union: <not found>\n";
            }
            dumpCandidateList("curb_parts", curbMatches);
            if (curb.bounds.valid) {
                std::cout << "  curb_union\n"
                          << "    renderer_world_bounds_min: " << FormatFloat3(curb.bounds.min) << "\n"
                          << "    renderer_world_bounds_max: " << FormatFloat3(curb.bounds.max) << "\n"
                          << "    renderer_world_center: " << FormatFloat3(curb.bounds.center) << "\n";
            } else {
                std::cout << "  curb_union: <not found>\n";
            }
            dumpCandidateList("street_opening_parts", streetOpeningMatches);
            dumpSnapshot("street_opening_candidate", streetOpeningMatches.empty() ? nullptr : streetOpeningMatches.front());

            const bool anyTransformDelta = std::any_of(
                snapshots.begin(), snapshots.end(), [](const RuntimeMeshSnapshot& snapshot) {
                    return snapshot.mesh && !MatrixNearlyEqual(snapshot.mesh->localToWorld, snapshot.mesh->defaultLocalToWorld);
                });
            std::cout << "[RuntimeLandmarks] transform_diagnosis: ";
            if (anyTransformDelta) {
                std::cout << "runtime applies an additional post-import mesh transform before rendering";
            } else {
                std::cout << "runtime render transforms match imported glTF world transforms; no extra post-import transform is present in Bistro";
            }
            std::cout << "\n";

            if (options.cameraSearchReportProvided) {
                const ResolvedCameraPose pose = ResolveCameraPose(settings);
                const float aspect =
                    static_cast<float>(settings.renderWidth > 0 ? settings.renderWidth : 1280u) /
                    static_cast<float>(settings.renderHeight > 0 ? settings.renderHeight : 720u);
                std::string reportError;
                WriteCameraSearchReportJson(fs::path(options.cameraSearchReportPath),
                                            pose,
                                            scooter ? &*scooter : nullptr,
                                            facade,
                                            leftLamp ? &*leftLamp : nullptr,
                                            rightLamp ? &*rightLamp : nullptr,
                                            curb,
                                            streetOpening ? &*streetOpening : nullptr,
                                            lampMatches,
                                            facadeMatches,
                                            aspect,
                                            reportError);
                if (!reportError.empty()) {
                    std::cerr << reportError << std::endl;
                    return 1;
                }
            }

            if (options.debugLandmarks && scooter && leftLamp && rightLamp && facade.bounds.valid && curb.bounds.valid) {
                const std::array<RuntimeLandmark, 5> markerLandmarks = {
                    *scooter, facade, *leftLamp, *rightLamp, curb
                };
                const std::array<simd::float3, 5> colors = {
                    simd_make_float3(60.0f, 8.0f, 8.0f),
                    simd_make_float3(8.0f, 60.0f, 8.0f),
                    simd_make_float3(8.0f, 8.0f, 60.0f),
                    simd_make_float3(60.0f, 60.0f, 8.0f),
                    simd_make_float3(60.0f, 8.0f, 60.0f),
                };
                for (size_t i = 0; i < markerLandmarks.size(); ++i) {
                    const uint32_t material =
                        sceneResources.addMaterial(simd_make_float3(1.0f, 1.0f, 1.0f),
                                                   0.0f,
                                                   PathTracerShaderTypes::MaterialType::DiffuseLight,
                                                   1.0f,
                                                   colors[i],
                                                   false,
                                                   "debug_landmark_" + std::to_string(i));
                    sceneResources.addSphere(markerLandmarks[i].bounds.center, 0.90f, material);
                }

                simd::float3 clusterMin = markerLandmarks.front().bounds.center;
                simd::float3 clusterMax = markerLandmarks.front().bounds.center;
                simd::float3 target = simd_make_float3(0.0f, 0.0f, 0.0f);
                for (const auto& landmark : markerLandmarks) {
                    clusterMin = simd_min(clusterMin, landmark.bounds.center);
                    clusterMax = simd_max(clusterMax, landmark.bounds.center);
                    target += landmark.bounds.center;
                }
                target /= static_cast<float>(markerLandmarks.size());
                const float clusterRadius =
                    std::max(8.0f, 0.5f * simd_length(clusterMax - clusterMin) + 6.0f);
                const simd::float3 viewDirection = simd_normalize(simd_make_float3(-1.0f, 0.45f, -1.0f));
                const simd::float3 position = target - viewDirection * (clusterRadius * 2.6f);
                const simd::float3 forward = simd_normalize(target - position);
                const simd::float3 up = simd_make_float3(0.0f, 1.0f, 0.0f);

                settings.explicitCameraEnabled = true;
                settings.explicitCameraPosition = position;
                settings.explicitCameraForward = forward;
                settings.explicitCameraUp = up;
                settings.cameraVerticalFov = 50.0f;
                settings.cameraDefocusAngle = 0.0f;
                settings.cameraFocusDistance = std::max(clusterRadius * 2.6f, 1.0f);

                const float aspect =
                    static_cast<float>(settings.renderWidth > 0 ? settings.renderWidth : 1280u) /
                    static_cast<float>(settings.renderHeight > 0 ? settings.renderHeight : 720u);
                std::cout << "[RuntimeLandmarks] validated_runtime_landmarks\n";
                for (const auto& landmark : markerLandmarks) {
                    std::cout << "  " << landmark.name << " center=" << FormatFloat3(landmark.bounds.center) << "\n";
                }
                if (streetOpening && streetOpening->bounds.valid) {
                    std::cout << "  " << streetOpening->name << " center=" << FormatFloat3(streetOpening->bounds.center) << "\n";
                }

                std::cout << "[RuntimeLandmarks] marker_camera\n";
                std::cout << "  position: " << FormatFloat3(position) << "\n";
                std::cout << "  target: " << FormatFloat3(target) << "\n";
                std::cout << "  up: " << FormatFloat3(up) << "\n";
                std::cout << "  vertical_fov_deg: " << settings.cameraVerticalFov << "\n";
                for (const auto& landmark : markerLandmarks) {
                    const ScreenProjection projection =
                        ProjectLandmark(landmark.bounds.center, position, forward, up, settings.cameraVerticalFov, aspect);
                    std::cout << "  projection " << landmark.name << ": ("
                              << std::fixed << std::setprecision(6) << projection.x << ", " << projection.y << ")"
                              << (projection.inside ? "" : " out_of_frame") << "\n";
                }
            } else {
                std::cout << "[RuntimeLandmarks] marker_injection_skipped: missing one or more required runtime landmarks\n";
            }
        }

        if (settings.renderWidth == 0) {
            settings.renderWidth = options.widthSet ? options.width : 1280u;
        }
        if (settings.renderHeight == 0) {
            settings.renderHeight = options.heightSet ? options.height : 720u;
        }

        if (useMetalSceneContext) {
            sceneResources.buildCPUPackedSceneData();
        } else {
            // Embree builds its scene from meshes() and never reads the packed arrays Metal uploads (on Barcelona
            // Pavilion, 2.5 GB resident); it needs the triangle count and the emissive inventory.
            sceneResources.buildCPUSceneCountsAndEmissiveInventory();
        }
        EnsureSceneDirectiveAssetPipelineReport(options, sceneManager, sceneResources);
        PrintEmissiveInventoryReport(sceneResources);
        PrintSubsurfaceMaterialReport(sceneResources, settings);

        const bool explicitSwrt =
            (options.enableSoftwareRayTracingSet && options.enableSoftwareRayTracing) ||
            (options.forceSoftwareRayTracingSet && options.forceSoftwareRayTracing);
        const bool explicitHwrt =
            options.forceHardwareRayTracingSet && options.forceHardwareRayTracing;
        const bool metalSupportsHardwareRt =
            useMetalSceneContext && statsContext.supportsRaytracing();

        if (options.backend == HeadlessBackend::Metal) {
            if (explicitHwrt && !metalSupportsHardwareRt) {
                std::cerr << "Metal hardware ray tracing was forced with --force-hwrt, but this device does not support it." << std::endl;
                return 1;
            }
            if (explicitSwrt) {
                std::cout << "[Backend] Metal software BVH tracing selected explicitly." << std::endl;
            } else if (explicitHwrt) {
                std::cout << "[Backend] Metal hardware ray tracing selected explicitly." << std::endl;
            } else if (!metalSupportsHardwareRt) {
                std::cout << "[Backend] Metal hardware ray tracing is unavailable on this device; software BVH fallback will be used." << std::endl;
            } else {
                std::cout << "[Backend] Metal hardware ray tracing is available." << std::endl;
            }
        } else {
            std::cout << "[Backend] Embree CPU backend selected." << std::endl;
        }

        if (useMetalSceneContext &&
            !settings.environmentMapPath.empty() &&
            !sceneResources.reloadEnvironmentIfNeeded(settings.environmentMapPath,
                                                      nullptr,
                                                      settings.environmentSunClamp)) {
            std::cerr << "Failed to load environment map for scene statistics: "
                      << settings.environmentMapPath << std::endl;
            return 1;
        }

        if (useMetalSceneContext) {
            LogCurrentAllocatedSize("post-load:", statsContext.device());
        }

        sceneResources.setSoftwareRayTracingOverride(settings.enableSoftwareRayTracing);
        const auto accelEstimate = useMetalSceneContext
            ? sceneResources.estimateAccelerationStructureBuildStats()
            : PathTracer::SceneAccelBuildStats{};
        PrintLargeSceneReport(sceneResources, options, accelEstimate);

        const uint64_t prebuildWorkingSet =
            sceneResources.estimatedGeometryMemoryBytes() +
            sceneResources.estimatedTextureMemoryBytes() +
            accelEstimate.blasMemoryBytes +
            accelEstimate.tlasMemoryBytes +
            accelEstimate.scratchMemoryBytes;
        const uint64_t recommendedWorkingSet = sceneResources.recommendedMaxWorkingSetSizeBytes();
        const bool prebuildOverDeviceBudget =
            recommendedWorkingSet > 0u && prebuildWorkingSet > recommendedWorkingSet;
        if (options.maxTextureBytesSet &&
            options.budgetPolicy == PathTracer::TextureBudgetPolicy::Strict &&
            sceneResources.textureBudgetExceeded()) {
            std::cerr << "[Budget] " << sceneResources.textureBudgetMessage()
                      << ". Aborting before acceleration-structure build due to --budget-policy=strict."
                      << std::endl;
            return 1;
        }
        if (options.budgetPolicy == PathTracer::TextureBudgetPolicy::Strict && prebuildOverDeviceBudget) {
            std::cerr << "[Budget] Prebuild working set " << FormatMemoryMiB(prebuildWorkingSet)
                      << " already exceeds recommendedMaxWorkingSetSize "
                      << FormatMemoryMiB(recommendedWorkingSet)
                      << ". Aborting before acceleration-structure build due to --budget-policy=strict."
                      << std::endl;
            return 1;
        }

        if (useMetalSceneContext) {
            std::cout << "[MipChain] material textures not a power of two in size (every 8-bit colour texture takes MipChain.h's "
                         "footprint-filtered mips; these need the odd-size weights): "
                      << sceneResources.footprintMipChainCount() << " of " << sceneResources.materialTextureCount()
                      << std::endl;
        }

        if (options.dumpMaterialTexturesDirProvided) {
            std::string dumpError;
            if (!DumpMaterialTextureDiagnostics(sceneResources,
                                                fs::path(options.dumpMaterialTexturesDir),
                                                dumpError)) {
                std::cerr << dumpError << std::endl;
                return 1;
            }
            std::cout << "[MaterialTextureDump] wrote "
                      << fs::path(options.dumpMaterialTexturesDir).lexically_normal().string()
                      << std::endl;
        }

        if (useMetalSceneContext) {
            sceneResources.rebuildAccelerationStructures();
        }

        if (useMetalSceneContext) {
            LogCurrentAllocatedSize("post-AS:", statsContext.device());
        }

        const auto& accelStats = sceneResources.accelerationBuildStats();
        const auto memoryReport = sceneResources.sceneMemoryReport();
        if (useMetalSceneContext) {
            PrintSceneReport(memoryReport, accelStats);
        } else {
            PrintEmbreeCpuSceneReport(sceneResources);
        }

        if (options.backend == HeadlessBackend::Metal) {
            const bool actualHardware =
                sceneResources.intersectionProvider().mode ==
                PathTracerShaderTypes::IntersectionMode::HardwareRayTracing;
            if (explicitHwrt && !actualHardware) {
                std::cerr << "Metal hardware ray tracing was forced, but acceleration-structure setup did not succeed";
                if (!accelStats.fallbackReason.empty()) {
                    std::cerr << ": " << accelStats.fallbackReason;
                }
                std::cerr << std::endl;
                return 1;
            }
            if (!actualHardware) {
                std::string fallbackReason = accelStats.fallbackReason;
                if (fallbackReason.empty()) {
                    fallbackReason = explicitSwrt
                        ? "software BVH requested explicitly"
                        : "hardware ray tracing unavailable for this scene/device";
                }
                std::cout << "[Backend] Metal using software BVH: " << fallbackReason << std::endl;
            } else {
                std::cout << "[Backend] Metal using hardware ray tracing." << std::endl;
            }
        }

        if (options.validateEmissive) {
            const uint32_t actual = sceneResources.emissivePrimitiveCount();
            if (options.expectedEmissiveCountSet && actual != options.expectedEmissiveCount) {
                std::cerr << "[FAIL] Expected " << options.expectedEmissiveCount
                          << " emissive primitives, got " << actual << std::endl;
                return 1;
            }
            if (options.expectedEmissiveCountSet) {
                std::cout << "[PASS] Emissive primitive count: " << actual << std::endl;
            }
            return 0;
        }

        const bool overBudget =
            memoryReport.recommendedMaxWorkingSetSizeBytes > 0u &&
            memoryReport.totalEstimatedMemoryBytes > memoryReport.recommendedMaxWorkingSetSizeBytes;
        if (memoryReport.recommendedMaxWorkingSetSizeBytes == 0u) {
            std::cout << "[Budget] recommendedMaxWorkingSetSize unavailable; skipping enforcement." << std::endl;
        } else if (overBudget) {
            std::ostringstream budgetMessage;
            budgetMessage << "[Budget] Estimated working set "
                          << FormatMemoryMiB(memoryReport.totalEstimatedMemoryBytes)
                          << " exceeds recommendedMaxWorkingSetSize "
                          << FormatMemoryMiB(memoryReport.recommendedMaxWorkingSetSizeBytes) << ".";
            if (options.budgetPolicy == PathTracer::TextureBudgetPolicy::Strict) {
                std::cerr << budgetMessage.str() << " Aborting due to --budget-policy=strict." << std::endl;
                return 1;
            }
            if (options.budgetPolicy == PathTracer::TextureBudgetPolicy::Warn) {
                std::cout << budgetMessage.str() << " Continuing due to --budget-policy=warn." << std::endl;
            }
        }

        HeadlessScene headlessScene{};
        headlessScene.source = options.scene;
        headlessScene.isPath = options.sceneIsPath;
        headlessScene.resources = &sceneResources;

        HeadlessCamera camera = BuildHeadlessCamera(settings);

        std::unique_ptr<IHeadlessRenderer> renderer;
        std::string sourceCapabilityError;
        if (headlessScene.resources && !PathTracer::ResolveSourceCapabilities(*headlessScene.resources, settings,
                options.backend != HeadlessBackend::Metal, true, sourceCapabilityError)) {
            std::cerr << sourceCapabilityError << std::endl;
            return 1;
        }
        if (headlessScene.resources) {
            const std::string ignoredDirectional =
                PathTracer::IgnoredDirectionalLightsDiagnostic(*headlessScene.resources, settings);
            if (!ignoredDirectional.empty()) std::cerr << "Warning: " << ignoredDirectional << std::endl;
        }
        // Both backends refuse the same spectral combinations (Embree runs spectral transport since EP11).
        {
            const std::string spectralRefusal = PathTracer::SpectralUnsupportedCombination(settings);
            if (!spectralRefusal.empty()) {
                std::cerr << "spectral mode does not support " << spectralRefusal
                          << "; refusing to render instead of substituting it." << std::endl;
                if (options.outputMetadata &&
                    (options.outputMetadataJsonProvided || !options.outputPath.empty())) {
                    const fs::path refusalPath = options.outputMetadataJsonProvided
                        ? fs::path(options.outputMetadataJsonPath)
                        : fs::path(options.outputPath + ".metadata.json");
                    std::ostringstream refusal;
                    refusal << "{\n"
                            << "  \"status\": \"refused\",\n"
                            << "  \"spectral_rendering_mode\": \""
                            << SpectralRenderingModeLabel(settings.spectralRenderingMode) << "\",\n"
                            << "  \"spectral_refused_combination\": \"" << JsonEscape(spectralRefusal) << "\"\n"
                            << "}\n";
                    std::string refusalError;
                    if (!WriteTextFileAtomic(refusalPath, refusal.str(), refusalError)) {
                        std::cerr << refusalError << std::endl;
                    }
                }
                return 2;
            }
        }
        if (options.backend == HeadlessBackend::Metal) {
            renderer = std::make_unique<MetalHeadlessRenderer>();
        } else {
#if defined(PATH_TRACER_ENABLE_EMBREE)
            auto embreeRenderer = std::make_unique<EmbreeHeadlessRenderer>();
            if (options.threadsSet) {
                embreeRenderer->setMaxThreads(options.threads);
            }
            renderer = std::move(embreeRenderer);
#else
            std::cerr << "Embree backend is not enabled. Rebuild with PATH_TRACER_ENABLE_EMBREE=ON." << std::endl;
            return 1;
#endif
        }

        if (settings.bloomEnabled) {
            std::cout << "[Beauty] Viewport bloom is not applied to saved headless images." << std::endl;
        }

        auto runHeadlessRender = [&](const PathTracer::RenderSettings& renderSettings,
                                     uint32_t sppTotal,
                                     HeadlessRenderOutput& renderOutput,
                                     std::string& renderError) -> bool {
            return renderer->render(headlessScene,
                                    camera,
                                    renderSettings,
                                    sppTotal,
                                    options.verbose,
                                    options.stats,
                                    options.writeDenoiseGuides ||
                                        ((renderSettings.denoiseEnabled || renderSettings.svgfDenoiseEnabled) &&
                                         (renderSettings.denoiseUseAlbedo || renderSettings.denoiseUseNormal)),
                                    renderOutput,
                                    renderError);
        };

        if (options.validateRis) {
            if (options.backend != HeadlessBackend::Metal) {
                std::cerr << "--validate-ris is Metal-only on this branch." << std::endl;
                return 1;
            }

            const uint32_t frames = options.validateRisFrames;
            const uint32_t risM = options.risCandidateCountSet ? std::max(options.risCandidateCount, 1u) : 8u;
            const uint32_t renderWidth = settings.renderWidth > 0u ? settings.renderWidth : 1280u;
            const uint32_t renderHeight = settings.renderHeight > 0u ? settings.renderHeight : 720u;
            const size_t pixelCount = static_cast<size_t>(renderWidth) * static_cast<size_t>(renderHeight) * 3u;
            const size_t luminanceCount = static_cast<size_t>(renderWidth) * static_cast<size_t>(renderHeight);
            const fs::path baselineReference =
                options.validateRisReferenceProvided
                    ? fs::path(options.validateRisReferencePath)
                    : fs::path("validation/baselines") /
                          (SanitizeSceneName(options.scene) + "_baseline_reference.pfm");

            std::cout << "[validate-ris] Scene: " << options.scene << std::endl;
            std::cout << "[validate-ris] BaselineMode: "
                      << DirectLightModeLabel(PathTracer::RenderSettings::DirectLightMode::BaselineEmissive)
                      << std::endl;
            std::cout << "[validate-ris] RisMode: "
                      << DirectLightModeLabel(PathTracer::RenderSettings::DirectLightMode::Ris) << std::endl;
            std::cout << "[validate-ris] RIS M: " << risM << std::endl;
            std::cout << "[validate-ris] Frames: " << frames << std::endl;
            std::cout << "[validate-ris] SPPPerFrame: 1" << std::endl;
            std::cout << "[validate-ris] ReferencePFM: " << baselineReference << std::endl;
            std::cout << "[validate-ris] EmissivePrimitiveCount: "
                      << sceneResources.emissivePrimitiveCount() << std::endl;
            std::cout << "[validate-ris] RectangleCount: " << sceneResources.rectangleCount()
                      << std::endl;

            if (sceneResources.emissivePrimitiveCount() == 0u) {
                std::cerr << "[validate-ris] Current scene has zero emissive primitives." << std::endl;
                if (sceneResources.rectangleCount() > 0u) {
                    std::cerr << "[validate-ris] The scene still has " << sceneResources.rectangleCount()
                              << " analytic rectangle/disk lights, so baseline_emissive and ris both collapse onto the shared rect-light fallback path."
                              << std::endl;
                } else {
                    std::cerr << "[validate-ris] baseline_emissive and ris cannot exercise the emissive-inventory path on this scene."
                              << std::endl;
                }
                std::cerr << "[validate-ris] Choose a closure scene with a non-empty emissive primitive inventory before interpreting variance or PSNR gates."
                          << std::endl;
                return 1;
            }

            std::vector<float> baselineAccum(pixelCount, 0.0f);
            std::vector<float> risAccum(pixelCount, 0.0f);
            std::vector<double> baselineSumLuma(luminanceCount, 0.0);
            std::vector<double> baselineSumSqLuma(luminanceCount, 0.0);
            std::vector<double> risSumLuma(luminanceCount, 0.0);
            std::vector<double> risSumSqLuma(luminanceCount, 0.0);
            double baselineFrameSeconds = 0.0;
            double risFrameSeconds = 0.0;

            auto accumulateFrame = [&](const std::vector<float>& linearRGB,
                                       std::vector<float>& accum,
                                       std::vector<double>& sumLuma,
                                       std::vector<double>& sumSqLuma,
                                       const char* label) -> bool {
                if (linearRGB.size() != pixelCount) {
                    std::cerr << "--validate-ris " << label << " frame size mismatch." << std::endl;
                    return false;
                }
                for (size_t i = 0, px = 0; i < pixelCount; i += 3u, ++px) {
                    accum[i + 0u] += linearRGB[i + 0u];
                    accum[i + 1u] += linearRGB[i + 1u];
                    accum[i + 2u] += linearRGB[i + 2u];
                    const double lum = LuminanceLinear(linearRGB[i + 0u], linearRGB[i + 1u], linearRGB[i + 2u]);
                    sumLuma[px] += lum;
                    sumSqLuma[px] += lum * lum;
                }
                return true;
            };

            auto makeDefaultOutput = [&](const char* suffix) {
                fs::path outputDir("validation/results");
                std::string sceneName = SanitizeSceneName(options.scene);
                std::ostringstream filename;
                filename << sceneName << "_" << suffix << ".pfm";
                return outputDir / filename.str();
            };

            auto renderModeSeries = [&](PathTracer::RenderSettings::DirectLightMode mode,
                                        uint32_t risM,
                                        std::vector<float>& accum,
                                        std::vector<double>& sumLuma,
                                        std::vector<double>& sumSqLuma,
                                        double& totalSeconds,
                                        const char* label) -> bool {
                for (uint32_t frame = 0u; frame < frames; ++frame) {
                    PathTracer::RenderSettings frameSettings = settings;
                    frameSettings.directLightMode = mode;
                    frameSettings.risCandidateCount = risM;
                    frameSettings.fixedRngSeed = (options.seedSet ? options.seed : 42u) + frame;
                    HeadlessRenderOutput frameOutput;
                    std::string frameError;
                    if (!runHeadlessRender(frameSettings, 1u, frameOutput, frameError)) {
                        std::cerr << "--validate-ris " << label << " frame " << frame
                                  << " failed: " << frameError << std::endl;
                        return false;
                    }
                    totalSeconds += frameOutput.totalSeconds;
                    if (!accumulateFrame(frameOutput.linearRGB, accum, sumLuma, sumSqLuma, label)) {
                        return false;
                    }
                }
                return true;
            };

            if (!renderModeSeries(PathTracer::RenderSettings::DirectLightMode::BaselineEmissive,
                                  1u,
                                  baselineAccum,
                                  baselineSumLuma,
                                  baselineSumSqLuma,
                                  baselineFrameSeconds,
                                  "baseline")) {
                return 1;
            }
            if (!renderModeSeries(PathTracer::RenderSettings::DirectLightMode::Ris,
                                  risM,
                                  risAccum,
                                  risSumLuma,
                                  risSumSqLuma,
                                  risFrameSeconds,
                                  "ris")) {
                return 1;
            }

            for (float& v : baselineAccum) {
                v /= static_cast<float>(frames);
            }
            for (float& v : risAccum) {
                v /= static_cast<float>(frames);
            }

            const VarianceStats baselineVariance =
                ComputeMeanPixelVariance(baselineSumLuma, baselineSumSqLuma, frames);
            const VarianceStats risVariance =
                ComputeMeanPixelVariance(risSumLuma, risSumSqLuma, frames);
            const ImageDifferenceStats baselineVsRisStats =
                ComputeImageDifferenceStats(baselineAccum, risAccum);
            const double avgBaselineFrameMs = (baselineFrameSeconds * 1000.0) / static_cast<double>(frames);
            const double avgRisFrameMs = (risFrameSeconds * 1000.0) / static_cast<double>(frames);
            const double perfRatio = (avgBaselineFrameMs > 0.0) ? (avgRisFrameMs / avgBaselineFrameMs) : 0.0;

            fs::path outputDir("validation/results");
            std::error_code outputDirEc;
            fs::create_directories(outputDir, outputDirEc);
            if (outputDirEc) {
                std::cerr << "Failed to create validation/results: " << outputDirEc.message() << std::endl;
                return 1;
            }

            const fs::path baselineAccumPath = makeDefaultOutput("baseline_accumulated");
            const fs::path risAccumPath = makeDefaultOutput("ris_accumulated");
            std::string writeError;
            const PathTracer::TonemapSettings hdrTonemap{};
            if (!PathTracer::WriteImage(baselineAccumPath.string(),
                                        PathTracer::ImageFileFormat::PFM,
                                        baselineAccum.data(),
                                        renderWidth,
                                        renderHeight,
                                        hdrTonemap,
                                        &writeError)) {
                std::cerr << writeError << std::endl;
                return 1;
            }
            if (!PathTracer::WriteImage(risAccumPath.string(),
                                        PathTracer::ImageFileFormat::PFM,
                                        risAccum.data(),
                                        renderWidth,
                                        renderHeight,
                                        hdrTonemap,
                                        &writeError)) {
                std::cerr << writeError << std::endl;
                return 1;
            }

            std::cout << "[validate-ris] Frames: " << frames << std::endl;
            std::cout << "[validate-ris] RIS M: " << risM << std::endl;
            std::cout << "[validate-ris] BaselineAccumPFM: " << baselineAccumPath << std::endl;
            std::cout << "[validate-ris] RisAccumPFM: " << risAccumPath << std::endl;
            std::cout << "[validate-ris] BaselineVsRisMaxAbs: " << std::fixed << std::setprecision(9)
                      << baselineVsRisStats.maxAbs << std::endl;
            std::cout << "[validate-ris] BaselineVsRisRMSE: " << std::fixed << std::setprecision(9)
                      << baselineVsRisStats.rmse << std::endl;
            std::cout << "[validate-ris] BaselineMeanVariance: " << std::fixed << std::setprecision(9)
                      << baselineVariance.meanPixelVariance << std::endl;
            std::cout << "[validate-ris] RisMeanVariance: " << std::fixed << std::setprecision(9)
                      << risVariance.meanPixelVariance << std::endl;
            std::cout << "[validate-ris] BaselineAvgFrameMs: " << std::fixed << std::setprecision(6)
                      << avgBaselineFrameMs << std::endl;
            std::cout << "[validate-ris] RisAvgFrameMs: " << std::fixed << std::setprecision(6)
                      << avgRisFrameMs << std::endl;
            std::cout << "[validate-ris] PerfRatio: " << std::fixed << std::setprecision(6)
                      << perfRatio << std::endl;

            std::vector<float> referenceLinearRGB;
            uint32_t refWidth = 0u;
            uint32_t refHeight = 0u;
            std::string referenceError;
            if (!LoadPFM(baselineReference, referenceLinearRGB, refWidth, refHeight, referenceError)) {
                std::cerr << "[validate-ris] Missing reference: " << baselineReference << std::endl;
                std::cerr << "[validate-ris] Baseline reference PFM is required for PSNR/RMSE closure checks." << std::endl;
                return 1;
            }
            if (refWidth != renderWidth || refHeight != renderHeight || referenceLinearRGB.size() != risAccum.size()) {
                std::cerr << "[validate-ris] Reference dimensions do not match current run." << std::endl;
                return 1;
            }

            const ImageDifferenceStats diffStats =
                ComputeImageDifferenceStats(referenceLinearRGB, risAccum);
            const double psnr = ComputePSNR(referenceLinearRGB, risAccum);
            std::cout << "[validate-ris] PSNR: " << std::fixed << std::setprecision(6) << psnr << " dB" << std::endl;
            std::cout << "[validate-ris] RMSE: " << std::fixed << std::setprecision(9) << diffStats.rmse << std::endl;

            std::error_code baselinesDirEc;
            fs::create_directories(fs::path("validation/baselines"), baselinesDirEc);
            if (baselinesDirEc) {
                std::cerr << "Failed to create validation/baselines: " << baselinesDirEc.message() << std::endl;
                return 1;
            }

            const fs::path thresholdsPath = fs::path("validation/baselines/ris_thresholds.json");
            std::string thresholdsError;
            if (!WriteRisThresholdsJson(thresholdsPath,
                                        options.scene,
                                        baselineReference,
                                        frames,
                                        risM,
                                        options.validateRisPsnrThreshold,
                                        diffStats.rmse,
                                        thresholdsError)) {
                std::cerr << thresholdsError << std::endl;
                return 1;
            }
            std::cout << "[validate-ris] ThresholdsJson: " << thresholdsPath << std::endl;

            if (!(psnr >= static_cast<double>(options.validateRisPsnrThreshold))) {
                return 1;
            }
            return 0;
        }

        if (options.biasCheck) {
            if (options.backend != HeadlessBackend::Metal) {
                std::cerr << "--bias-check is Metal-only on this branch." << std::endl;
                return 1;
            }

            PathTracer::RenderSettings baselineSettings = settings;
            baselineSettings.directLightMode =
                PathTracer::RenderSettings::DirectLightMode::BaselineEmissive;
            baselineSettings.risCandidateCount = 1u;

            PathTracer::RenderSettings risSettings = settings;
            risSettings.directLightMode = PathTracer::RenderSettings::DirectLightMode::Ris;
            risSettings.risCandidateCount = 1u;

            HeadlessRenderOutput baselineOutput;
            HeadlessRenderOutput risOutput;
            std::string baselineError;
            std::string risError;
            if (!runHeadlessRender(baselineSettings, options.sppTotal, baselineOutput, baselineError)) {
                std::cerr << "Bias-check baseline render failed: " << baselineError << std::endl;
                return 1;
            }
            if (!runHeadlessRender(risSettings, options.sppTotal, risOutput, risError)) {
                std::cerr << "Bias-check RIS render failed: " << risError << std::endl;
                return 1;
            }
            if (baselineOutput.width != risOutput.width ||
                baselineOutput.height != risOutput.height ||
                baselineOutput.linearRGB.size() != risOutput.linearRGB.size()) {
                std::cerr << "Bias-check failed: output dimensions did not match." << std::endl;
                return 1;
            }

            const ImageDifferenceStats diffStats =
                ComputeImageDifferenceStats(baselineOutput.linearRGB, risOutput.linearRGB);

            std::cout << "[bias-check] M=1 RIS vs baseline emissive NEE" << std::endl;
            std::cout << "[bias-check] Max abs error: " << std::fixed << std::setprecision(9)
                      << diffStats.maxAbs << std::endl;
            std::cout << "[bias-check] RMSE:          " << std::fixed << std::setprecision(9)
                      << diffStats.rmse << std::endl;

            if (!(diffStats.rmse < static_cast<double>(options.biasCheckRmseThreshold))) {
                return 1;
            }
            return 0;
        }

        std::optional<PathTracer::AccumulationState> resumeState;
        if (options.checkpointEverySamples > 0u || options.resumeFromAccumulationState) {
            if (auto refusal = SampleCheckpointRefusal(options, settings)) {
                std::cerr << "[Checkpoint] Sample checkpoints and resume are refused with " << *refusal << "." << std::endl;
                return 2;
            }
            const fs::path checkpointOutput(options.outputPath);
            const fs::path statePath(checkpointOutput.string() + ".accum");
            const fs::path manifestPath = options.checkpointManifestJsonProvided
                ? fs::path(options.checkpointManifestJsonPath)
                : fs::path(checkpointOutput.string() + ".checkpoint.json");
            HeadlessSampleCheckpointOptions checkpointOptions;
            if (options.resumeFromAccumulationState) {
                PathTracer::AccumulationState state;
                std::string stateError;
                if (!PathTracer::ReadAccumulationStateFile(options.resumeAccumulationStatePath, state, stateError)) {
                    std::cerr << "[Checkpoint] " << stateError << std::endl;
                    return 1;
                }
                resumeState = std::move(state);
                checkpointOptions.resumeFrom = &*resumeState;
            }
            if (options.checkpointEverySamples > 0u) {
                checkpointOptions.everySamples = options.checkpointEverySamples;
                checkpointOptions.write = [&options, &settings, checkpointOutput, statePath, manifestPath](
                                              const PathTracer::AccumulationState& state, std::string& writeError) {
                    if (!PathTracer::WriteAccumulationStateFile(statePath.string(), state, writeError)) {
                        return false;
                    }
                    const std::string manifest = BuildPartialCheckpointManifestJson(
                        options, settings, checkpointOutput, statePath, state.sampleCount);
                    if (!WriteTextFileAtomic(manifestPath, manifest, writeError)) {
                        return false;
                    }
                    std::cout << "[Checkpoint] " << state.sampleCount << " of " << options.sppTotal
                              << " samples saved to " << statePath << " (manifest " << manifestPath << ")" << std::endl;
                    return true;
                };
            }
            std::string checkpointError;
            if (!renderer->setSampleCheckpointOptions(checkpointOptions, checkpointError)) {
                std::cerr << "[Checkpoint] " << checkpointError << std::endl;
                return 2;
            }
        }

        HeadlessRenderOutput output;
        std::string renderError;
        if (!runHeadlessRender(settings, options.sppTotal, output, renderError)) {
            std::cerr << "Render failed: " << renderError << std::endl;
            return 1;
        }

        std::vector<float> rawLinearRGB;
        if (settings.denoiseEnabled) {
            rawLinearRGB = output.linearRGB;
        }

        std::string denoiseError;
        if (!RunHeadlessDenoise(settings, output, denoiseError)) {
            std::cerr << denoiseError << std::endl;
            return 1;
        }
        ApplyRestirDebugView(settings, RunDirectLightMode(options, settings), output);

        std::string outputPath = options.outputPath;
        PathTracer::ImageFileFormat outputFormat = options.format;
        if (outputPath.empty()) {
            fs::path defaultDir("renders");
            std::string sceneName = SanitizeSceneName(options.scene);
            std::ostringstream filename;
            filename << sceneName << "_" << output.width << "x" << output.height << "."
                     << PathTracer::FormatExtension(outputFormat);
            outputPath = (defaultDir / filename.str()).string();
        }

        fs::path outputFsPath(outputPath);
        if (!outputFsPath.parent_path().empty()) {
            std::error_code dirEc;
            fs::create_directories(outputFsPath.parent_path(), dirEc);
            if (dirEc) {
                std::cerr << "Failed to create output directory '" << outputFsPath.parent_path().string()
                          << "': " << dirEc.message() << std::endl;
                return 1;
            }
        }
        std::optional<fs::path> debugBundleDir;
        if (options.debugBundleDirProvided) {
            debugBundleDir = fs::path(options.debugBundleDir);
            std::error_code bundleDirEc;
            fs::create_directories(*debugBundleDir, bundleDirEc);
            if (bundleDirEc) {
                std::cerr << "Failed to create debug bundle directory '"
                          << debugBundleDir->string() << "': " << bundleDirEc.message()
                          << std::endl;
                return 1;
            }
        }

        std::string writeError;
        PathTracer::TonemapSettings tonemapSettings{};
        tonemapSettings.tonemapMode = settings.tonemapMode;
        tonemapSettings.acesVariant = settings.acesVariant;
        tonemapSettings.exposure = settings.exposure;
        tonemapSettings.reinhardWhitePoint = settings.reinhardWhitePoint;
        tonemapSettings.srgbEncode = settings.srgbOutputEncoding;
        tonemapSettings.referenceBloomIntensity = settings.referenceBloomIntensity;

        auto writeImageBuffer = [&](const fs::path& targetPath,
                                    PathTracer::ImageFileFormat format,
                                    const std::vector<float>& linearRGB) -> bool {
            return PathTracer::WriteImage(targetPath.string(),
                                          format,
                                          linearRGB.data(),
                                          output.width,
                                          output.height,
                                          tonemapSettings,
                                          &writeError);
        };

        if (options.verbose || settings.denoiseEnabled || settings.svgfDenoiseEnabled ||
            settings.fireflyClampEnabled) {
            std::cout << "[Beauty] denoise=" << (settings.denoiseEnabled ? "on" : "off")
                      << " svgf=" << (settings.svgfDenoiseEnabled ? "on" : "off")
                      << " hdr=" << (settings.denoiseHdr ? "1" : "0")
                      << " useAlbedo=" << (settings.denoiseUseAlbedo ? "1" : "0")
                      << " useNormal=" << (settings.denoiseUseNormal ? "1" : "0")
                      << " fireflyClampStage=per_sample_contribution"
                      << " fireflyClampMode=" << (!settings.fireflyClampEnabled ? "off" :
                          settings.fireflyClampMode == PathTracer::RenderSettings::FireflyClampMode::MaxComponent
                              ? "maxcomponent" : "luminance")
                      << " fireflyClampFactor=" << settings.fireflyClampFactor
                      << " fireflyClampFloor=" << settings.fireflyClampFloor
                      << " fireflyClampMaxContribution=" << settings.fireflyClampMaxContribution
                      << std::endl;
        }

        const std::vector<float>& beautyLinearRGB = output.linearRGB;

        fs::path rawOutputPath = outputFsPath;
        fs::path denoisedOutputPath = outputFsPath;
        if (settings.denoiseEnabled) {
            rawOutputPath = AppendStemSuffix(outputFsPath, "_raw");
            denoisedOutputPath = AppendStemSuffix(outputFsPath, "_denoised");
            if (!writeImageBuffer(rawOutputPath, outputFormat, rawLinearRGB)) {
                std::cerr << "Failed to write raw output image: " << writeError << std::endl;
                return 1;
            }

            if (!writeImageBuffer(denoisedOutputPath, outputFormat, beautyLinearRGB)) {
                std::cerr << "Failed to write denoised output image: " << writeError << std::endl;
                return 1;
            }
        } else if (!writeImageBuffer(outputFsPath, outputFormat, beautyLinearRGB)) {
            std::cerr << "Failed to write output image: " << writeError << std::endl;
            return 1;
        }

        if (options.writeDenoiseGuides) {
            const size_t guideValues = static_cast<size_t>(output.width) * static_cast<size_t>(output.height) * 3u;
            const std::pair<const char*, const std::vector<float>*> guides[] = {
                {"_albedo", &output.albedoLinearRGB}, {"_normal", &output.normalLinearRGB}};
            for (const auto& [suffix, buffer] : guides) {
                if (buffer->size() != guideValues) {
                    std::cerr << "Denoise guide " << suffix << " unavailable for this backend/render" << std::endl;
                    return 1;
                }
                fs::path guidePath = AppendStemSuffix(outputFsPath, suffix);
                guidePath.replace_extension(".exr");
                if (!writeImageBuffer(guidePath, PathTracer::ImageFileFormat::EXR, *buffer)) {
                    std::cerr << "Failed to write denoise guide: " << writeError << std::endl;
                    return 1;
                }
            }
        }

        std::optional<fs::path> writtenMetadataPath;
        std::optional<fs::path> writtenSettingsPath;
        std::optional<fs::path> writtenQueuePath;
        std::optional<fs::path> writtenMetricsPath;
        std::optional<fs::path> writtenCheckpointPath;
        std::optional<fs::path> writtenTileManifestPath;

        if (options.outputMetadata) {
            const fs::path metadataPath = options.outputMetadataJsonProvided
                ? fs::path(options.outputMetadataJsonPath)
                : (debugBundleDir ? (*debugBundleDir / "output_metadata.json")
                                  : fs::path(outputFsPath.string() + ".metadata.json"));
            std::string metadataError;
            const std::string metadataJson =
                BuildOutputMetadataJson(options,
                                        settings,
                                        sceneResources,
                                        output,
                                        outputFsPath,
                                        settings.denoiseEnabled ? &rawOutputPath : nullptr,
                                        settings.denoiseEnabled ? &denoisedOutputPath : nullptr);
            if (!WriteTextFileAtomic(metadataPath, metadataJson, metadataError)) {
                std::cerr << metadataError << std::endl;
                return 1;
            }
            writtenMetadataPath = metadataPath;
            std::cout << "Output metadata written to: " << metadataPath << std::endl;
        }

        if (options.settingsJsonProvided || debugBundleDir) {
            const fs::path settingsPath = options.settingsJsonProvided
                ? fs::path(options.settingsJsonPath)
                : (*debugBundleDir / "render_settings.json");
            std::string settingsError;
            if (!WriteTextFileAtomic(settingsPath,
                                     BuildRenderSettingsJson(options, settings, outputFsPath),
                                     settingsError)) {
                std::cerr << settingsError << std::endl;
                return 1;
            }
            writtenSettingsPath = settingsPath;
            std::cout << "Render settings written to: " << settingsPath << std::endl;
        }

        if (options.renderQueueItemJsonProvided || debugBundleDir) {
            const fs::path queuePath = options.renderQueueItemJsonProvided
                ? fs::path(options.renderQueueItemJsonPath)
                : (*debugBundleDir / "render_queue_item.json");
            std::string queueError;
            if (!WriteTextFileAtomic(queuePath,
                                     BuildRenderQueueItemJson(options, settings, outputFsPath),
                                     queueError)) {
                std::cerr << queueError << std::endl;
                return 1;
            }
            writtenQueuePath = queuePath;
            std::cout << "Render queue item written to: " << queuePath << std::endl;
        }

        if (options.checkpointManifestJsonProvided || options.resumeCheckpointJsonProvided ||
            options.checkpointEverySamples > 0u || debugBundleDir) {
            const fs::path checkpointPath = options.checkpointManifestJsonProvided
                ? fs::path(options.checkpointManifestJsonPath)
                : (debugBundleDir ? (*debugBundleDir / "checkpoint_manifest.json")
                                  : fs::path(outputFsPath.string() + ".checkpoint.json"));
            std::string checkpointError;
            // A sample-checkpointed render keeps its finished accumulation beside the output, so the completed
            // manifest can write the output again without rendering.
            std::optional<fs::path> finishedStatePath;
            if (options.checkpointEverySamples > 0u || options.resumeFromAccumulationState) {
                const fs::path statePath(outputFsPath.string() + ".accum");
                std::error_code existsError;
                if (fs::exists(statePath, existsError)) {
                    finishedStatePath = statePath;
                }
            }
            const std::string checkpointJson =
                BuildCheckpointManifestJson(options,
                                            settings,
                                            output,
                                            outputFsPath,
                                            writtenMetadataPath ? &*writtenMetadataPath : nullptr,
                                            writtenSettingsPath ? &*writtenSettingsPath : nullptr,
                                            writtenQueuePath ? &*writtenQueuePath : nullptr,
                                            finishedStatePath ? &*finishedStatePath : nullptr);
            if (!WriteTextFileAtomic(checkpointPath, checkpointJson, checkpointError)) {
                std::cerr << checkpointError << std::endl;
                return 1;
            }
            writtenCheckpointPath = checkpointPath;
            std::cout << "Checkpoint manifest written to: " << checkpointPath << std::endl;
            if (finishedStatePath) {
                std::cout << "[Checkpoint] Kept the finished accumulation " << *finishedStatePath
                          << " for the completed manifest" << std::endl;
            }
        }

        if (options.tileManifestJsonProvided || options.tileSizeSet || debugBundleDir) {
            const fs::path tileManifestPath = options.tileManifestJsonProvided
                ? fs::path(options.tileManifestJsonPath)
                : (debugBundleDir ? (*debugBundleDir / "tile_manifest.json")
                                  : fs::path(outputFsPath.string() + ".tiles.json"));
            std::string tileError;
            if (!WriteTextFileAtomic(tileManifestPath,
                                     BuildTileManifestJson(options, output, outputFsPath),
                                     tileError)) {
                std::cerr << tileError << std::endl;
                return 1;
            }
            writtenTileManifestPath = tileManifestPath;
            std::cout << "Tile manifest written to: " << tileManifestPath << std::endl;
        }

        std::optional<ImagePbrMetrics> cachedImageMetrics;
        auto imageMetricsForOutput = [&]() -> const ImagePbrMetrics& {
            if (!cachedImageMetrics) {
                cachedImageMetrics = ComputeImagePbrMetrics(output.linearRGB);
            }
            return *cachedImageMetrics;
        };
        const bool emitPbrMetrics = options.pbrMetrics || options.pbrMetricsJsonProvided;
        if (emitPbrMetrics) {
            const ImagePbrMetrics& imageMetrics = imageMetricsForOutput();
            PrintPbrMetricsSummary(imageMetrics, output);

            fs::path metricsPath;
            if (options.pbrMetricsJsonProvided) {
                metricsPath = fs::path(options.pbrMetricsJsonPath);
            } else if (debugBundleDir) {
                metricsPath = *debugBundleDir / "pbr_metrics.json";
            } else {
                metricsPath = fs::path(outputFsPath.string() + ".metrics.json");
            }
            if (!metricsPath.parent_path().empty()) {
                std::error_code metricsDirEc;
                fs::create_directories(metricsPath.parent_path(), metricsDirEc);
                if (metricsDirEc) {
                    std::cerr << "Failed to create metrics directory '" << metricsPath.parent_path().string()
                              << "': " << metricsDirEc.message() << std::endl;
                    return 1;
                }
            }
            std::string metricsError;
            if (!WritePbrMetricsJson(metricsPath,
                                     options,
                                     settings,
                                     sceneResources,
                                     outputFsPath,
                                     settings.denoiseEnabled ? &rawOutputPath : nullptr,
                                     settings.denoiseEnabled ? &denoisedOutputPath : nullptr,
                                     output,
                                     imageMetrics,
                                     metricsError)) {
                std::cerr << metricsError << std::endl;
                return 1;
            }
            writtenMetricsPath = metricsPath;
            fs::path histogramPath = fs::path(metricsPath.string() + ".luma_hist.txt");
            if (!WriteLuminanceHistogramText(histogramPath, imageMetrics, metricsError)) {
                std::cerr << metricsError << std::endl;
                return 1;
            }
            std::cout << "PBR metrics written to: " << metricsPath << std::endl;
            std::cout << "Luminance histogram written to: " << histogramPath << std::endl;
        }

        if (debugBundleDir) {
            const fs::path bundleManifestPath = *debugBundleDir / "manifest.json";
            std::string bundleError;
            if (!WriteTextFileAtomic(bundleManifestPath,
                                     BuildDebugBundleManifestJson(*debugBundleDir,
                                                                  outputFsPath,
                                                                  writtenMetadataPath ? &*writtenMetadataPath : nullptr,
                                                                  writtenSettingsPath ? &*writtenSettingsPath : nullptr,
                                                                  writtenQueuePath ? &*writtenQueuePath : nullptr,
                                                                  writtenMetricsPath ? &*writtenMetricsPath : nullptr,
                                                                  writtenCheckpointPath ? &*writtenCheckpointPath : nullptr,
                                                                  writtenTileManifestPath ? &*writtenTileManifestPath : nullptr),
                                     bundleError)) {
                std::cerr << bundleError << std::endl;
                return 1;
            }
            std::cout << "Debug bundle manifest written to: " << bundleManifestPath << std::endl;
        }

        if (options.restirDebugMetricsJsonProvided) {
            fs::path debugMetricsPath = fs::path(options.restirDebugMetricsJsonPath);
            if (!debugMetricsPath.parent_path().empty()) {
                std::error_code debugMetricsDirEc;
                fs::create_directories(debugMetricsPath.parent_path(), debugMetricsDirEc);
                if (debugMetricsDirEc) {
                    std::cerr << "Failed to create ReSTIR debug metrics directory '"
                              << debugMetricsPath.parent_path().string() << "': "
                              << debugMetricsDirEc.message() << std::endl;
                    return 1;
                }
            }
            std::string debugMetricsError;
            if (!WriteRestirDebugMetricsJson(debugMetricsPath,
                                             options,
                                             settings,
                                             outputFsPath,
                                             output,
                                             imageMetricsForOutput(),
                                             debugMetricsError)) {
                std::cerr << debugMetricsError << std::endl;
                return 1;
            }
            std::cout << "ReSTIR debug metrics written to: " << debugMetricsPath << std::endl;
        }

        if (options.materialMrDebugJsonProvided) {
            fs::path debugPath = fs::path(options.materialMrDebugJsonPath);
            if (!debugPath.parent_path().empty()) {
                std::error_code debugDirEc;
                fs::create_directories(debugPath.parent_path(), debugDirEc);
                if (debugDirEc) {
                    std::cerr << "Failed to create Material_MR debug directory '"
                              << debugPath.parent_path().string() << "': "
                              << debugDirEc.message() << std::endl;
                    return 1;
                }
            }
            std::string debugError;
            if (!WriteMaterialMrDebugJson(debugPath,
                                          options,
                                          outputFsPath,
                                          output.materialMrDebug,
                                          debugError)) {
                std::cerr << debugError << std::endl;
                return 1;
            }
            std::cout << "Material_MR debug written to: " << debugPath << std::endl;
        }
        if (options.directLightAuditJsonProvided) {
            fs::path auditPath = fs::path(options.directLightAuditJsonPath);
            if (!auditPath.parent_path().empty()) {
                std::error_code auditDirEc;
                fs::create_directories(auditPath.parent_path(), auditDirEc);
                if (auditDirEc) {
                    std::cerr << "Failed to create direct-light audit directory '"
                              << auditPath.parent_path().string() << "': "
                              << auditDirEc.message() << std::endl;
                    return 1;
                }
            }
            std::string auditError;
            if (!WriteDirectLightAuditJson(auditPath,
                                           options,
                                           outputFsPath,
                                           output.directLightAudit,
                                           auditError)) {
                std::cerr << auditError << std::endl;
                return 1;
            }
            std::cout << "Direct-light audit written to: " << auditPath << std::endl;
        }

        std::cout << "Rendered " << output.samples << " spp at " << output.width << "x" << output.height
                  << " in " << std::fixed << std::setprecision(2) << output.totalSeconds << " s"
                  << " (~" << std::setprecision(3) << output.avgMsPerSample << " ms/sample)." << std::endl;
        PrintSssWalkLine(output);
        if (output.performance.available) {
            std::cout << "[Perf] gpu=" << std::fixed << std::setprecision(2)
                      << output.performance.totalGpuSeconds << " s"
                      << " cpuEncode=" << output.performance.totalCpuEncodeSeconds << " s"
                      << " tracedRays=" << output.performance.tracedRayCount
                      << " tracedRaysPerSec=" << std::setprecision(0)
                      << output.performance.tracedRaysPerSecond
                      << " tracedRaysPerGpuSec=" << output.performance.tracedRaysPerGpuSecond
                      << std::endl;
            if (output.performance.gpuBucketTimingAvailable) {
                std::cout << "[Perf] gpuBuckets"
                          << " intersect=" << std::fixed << std::setprecision(2)
                          << output.performance.gpuBucketIntersectSeconds << " s"
                          << " shade=" << output.performance.gpuBucketShadeSeconds << " s"
                          << " shadowQueue=" << output.performance.gpuBucketShadowQueueSeconds << " s"
                          << " fusedIntegrate=" << output.performance.gpuBucketFusedIntegrateSeconds << " s"
                          << " present=" << output.performance.gpuBucketPresentationSeconds << " s"
                          << std::endl;
            }
            std::cout << "[Perf] dispatches total=" << output.performance.totalComputeDispatchCount
                      << " restirDiInit=" << output.performance.restirDiInitializeDispatchCount
                      << " restirDiInitial=" << output.performance.restirDiInitialCandidateDispatchCount
                      << " restirDiTemporal=" << output.performance.restirDiTemporalReuseDispatchCount
                      << " restirDiSpatial=" << output.performance.restirDiSpatialReuseDispatchCount
                      << " restirDiVisibility=" << output.performance.restirDiVisibilityDispatchCount
                      << " restirDiApply=" << output.performance.restirDiApplyLightingDispatchCount
                      << " restirDiDebugAov=" << output.performance.restirDiDebugAovDispatchCount
                      << " integration=" << output.performance.integrationDispatchCount
                      << " present=" << output.performance.presentationDispatchCount
                      << " wf(reset=" << output.performance.wavefrontResetDispatchCount
                      << ",generate=" << output.performance.wavefrontGenerateDispatchCount
                      << ",buildArgs=" << output.performance.wavefrontBuildDispatchArgsDispatchCount
                      << ",compact=" << output.performance.wavefrontCompactQueueDispatchCount
                      << ",prepare=" << output.performance.wavefrontPrepareDispatchCount
                      << ",intersect=" << output.performance.wavefrontIntersectDispatchCount
                      << ",shade=" << output.performance.wavefrontShadeDispatchCount
                      << ",shadow=" << output.performance.wavefrontShadowDispatchCount
                      << ",swap=" << output.performance.wavefrontSwapDispatchCount
                      << ",accumulate=" << output.performance.wavefrontAccumulateDispatchCount
                      << ")" << std::endl;
            if (output.performance.wavefrontQueueBytesTotal > 0u) {
                const double queueMiB =
                    static_cast<double>(output.performance.wavefrontQueueBytesTotal) / (1024.0 * 1024.0);
                const double peakMiB =
                    static_cast<double>(output.performance.peakAllocatedBytes) / (1024.0 * 1024.0);
                std::cout << "[Perf] wavefront queueMiB=" << std::setprecision(2) << queueMiB
                          << " budgetPct=" << output.performance.wavefrontBudgetUsagePercent
                          << " peakAllocatedMiB=" << peakMiB
                          << " work(primary=" << output.performance.wavefrontPrimaryGeneratedCount
                          << ",intersect=" << output.performance.wavefrontIntersectProcessedCount
                          << ",shade=" << output.performance.wavefrontShadeProcessedCount
                          << ",shadow=" << output.performance.wavefrontShadowProcessedCount
                          << ",nextEnqueue=" << output.performance.wavefrontNextRayEnqueueCount
                          << ",shadowEnqueue=" << output.performance.wavefrontShadowRayEnqueueCount
                          << ")"
                          << " queues(primaryPeak="
                          << output.performance.wavefrontQueuePeakActiveCounts[0]
                          << "/" << output.performance.wavefrontQueueCapacities[0]
                          << ",shadowPeak="
                          << output.performance.wavefrontQueuePeakActiveCounts[5]
                          << "/" << output.performance.wavefrontQueueCapacities[5]
                          << ",overflow="
                          << (output.performance.wavefrontQueueOverflowCounts[0] +
                              output.performance.wavefrontQueueOverflowCounts[1] +
                              output.performance.wavefrontQueueOverflowCounts[2] +
                              output.performance.wavefrontQueueOverflowCounts[3] +
                              output.performance.wavefrontQueueOverflowCounts[4] +
                              output.performance.wavefrontQueueOverflowCounts[5] +
                              output.performance.wavefrontQueueOverflowCounts[6] +
                              output.performance.wavefrontQueueOverflowCounts[7])
                          << ")" << std::endl;
            } else {
                const double peakMiB =
                    static_cast<double>(output.performance.peakAllocatedBytes) / (1024.0 * 1024.0);
                std::cout << "[Perf] peakAllocatedMiB=" << std::setprecision(2) << peakMiB << std::endl;
            }
        }
        if (settings.denoiseEnabled) {
            std::cout << "Raw output written to: " << rawOutputPath << std::endl;
            std::cout << "Denoised output written to: " << denoisedOutputPath << std::endl;
        } else {
            std::cout << "Output written to: " << outputFsPath << std::endl;
        }

        return 0;
    }
}
