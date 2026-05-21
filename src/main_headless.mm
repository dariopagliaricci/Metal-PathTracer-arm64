#import <Foundation/Foundation.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "MetalRenderer.h"
#include "headless/EmbreeHeadlessRenderer.h"
#include "headless/MetalHeadlessRenderer.h"
#include "headless/IHeadlessRenderer.h"
#include "renderer/RenderSettings.h"
#include "renderer/ImageWriter.h"
#include "renderer/OidnCpuDenoiser.h"
#include "renderer/MetalContext.h"
#include "renderer/SceneManager.h"
#include "renderer/SceneResources.h"

namespace fs = std::filesystem;
constexpr double kPi = 3.14159265358979323846;
constexpr double kMiB = 1024.0 * 1024.0;

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
    bool svgfDenoiseEnabled = false;
    bool svgfDenoiseEnabledSet = false;
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
              << "  --renderProfile=<custom|preview|lookdev|final|reference|debug>\n"
              << "                                 Apply explicit production defaults before CLI overrides\n"
              << "  --maxDepth=<int>              Override max path depth\n"
              << "  --threads=<int>               Max worker threads (Embree only)\n"
              << "  --seed=<int>                  Fixed RNG seed (0 = random)\n"
              << "  --enableSoftwareRayTracing[=0|1]  Force software tracing kernels\n"
              << "  --force-hwrt[=0|1]             Require Metal hardware ray tracing\n"
              << "  --force-swrt[=0|1]             Require Metal software BVH tracing\n"
              << "  --executionMode=<megakernel|wavefront>  Select Metal execution path (default megakernel)\n"
              << "  --wavefront-policy=<preview|final|offline|research>  Wavefront scheduling policy\n"
              << "  --wavefront-compaction=<0|1>   Enable explicit wavefront queue compaction pass\n"
              << "  --enableMnee[=0|1]             Enable MNEE caustics (default 0)\n"
              << "  --enableSpecularNee[=0|1]      Enable specular NEE (default 1)\n\n"
              << "R2/R3/R4/R5/R6/R7 direct-light and guiding controls:\n"
              << "  --directLightMode=<legacy|baseline_emissive|ris|ris_spatial|ris_temporal|ris_world|ris_regir|restir_di|restir_di_regir_hybrid>\n"
              << "                                 Select direct-light path (default legacy)\n"
              << "  --ris-m=<int>                  RIS candidate count used when directLightMode=ris/ris_spatial/ris_temporal/ris_world/ris_regir/restir_di/restir_di_regir_hybrid (default 8)\n"
              << "  --spatial-reuse-k=<int>        R3 same-frame neighbor count for directLightMode=ris_spatial (default 4)\n"
              << "  --world-reuse-cell-size=<float> R5/R6 checkpoint world-grid cell size for directLightMode=ris_world/ris_regir (default 0.5)\n"
              << "  --restirGiMode=<off|restir_gi_prototype|restir_gi_diffuse>  Gated production BSDF-backed diffuse/PBR indirect reuse (default off)\n"
              << "  --pathGuidingMode=<off|path_guiding>  Gated MIS path guiding with world-grid directional bins (default off)\n"
              << "  --pathGuidingStrength=<float>  Guided strategy probability scale in [0,1] (default 0.35)\n"
              << "  --pathGuidingCellSize=<float>  World-space path guiding cell size (default 1.0)\n"
              << "  --pathGuidingTrainingInterval=<int>  Train guiding every N frames (default 1)\n"
              << "  --restirPtMode=<off|restir_pt_research|restir_pt_experimental_path_reuse>  Gated production path-reservoir capture/reuse modes (default off)\n"
              << "  --restirPtMaxReservoirs=<int>  Max R8 path reservoir captures per frame (default 4096)\n"
              << "  --restirPtDebug[=0|1]         Enable PT reservoir debug counter path (default 0)\n"
              << "  --restirPtReuseStrength=<float>  Production path reuse strength in [0,1] (default 0.25)\n"
              << "  --radianceCacheMode=<off|radiance_cache>  Gated rough-indirect radiance cache (default off)\n"
              << "  --radianceCacheCellSize=<float>  World-space radiance cache cell size (default 1.0)\n"
              << "  --radianceCacheMinDepth=<int>  Minimum path depth eligible for cache termination (default 2)\n"
              << "  --radianceCacheMinConfidence=<float>  Minimum cache confidence in [0,1] (default 0.35)\n"
              << "  --radianceCacheTrainingInterval=<int>  Train cache every N frames (default 1)\n"
              << "  --radianceCacheMaxRadiance=<float>  Quantized cache radiance clamp (default 16)\n"
              << "  --causticTransportMode=<off|path_space>  Gated path-space caustic transport prototype (default off)\n"
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
              << "  --bias-check[=0|1]             Compare M=1 RIS against baseline emissive NEE for the current scene\n\n"
              << "R2 closure controls:\n"
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
              << "  --envIntensity=<float>        Environment intensity multiplier\n\n"
              << "Backend selection:\n"
              << "  --backend=<metal|embree>       Headless backend (default metal)\n"
              << "  --enableEmbree[=0|1]           Alias for --backend embree\n\n"
              << "Tonemapping overrides (for LDR outputs):\n"
              << "  --tonemap=<1|2|3|4>           1=Linear, 2=ACES, 3=Reinhard, 4=Hable\n"
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
              << "  --denoise-hdr[=0|1]           Treat denoise input as linear HDR radiance (default 1)\n"
              << "  --svgfDenoise[=0|1]           Run Metal SVGF spatiotemporal variance-guided denoise (default 0)\n"
              << "  --svgf-atrous-iterations=<n>  A-trous filter iterations for --svgfDenoise (1-5, default 1)\n"
              << "  --svgf-normal-sigma=<float>   Normal edge weight for --svgfDenoise (default 64)\n"
              << "  --svgf-albedo-sigma=<float>   Albedo edge weight for --svgfDenoise (default 16)\n"
              << "  --svgf-luminance-sigma=<float> Luminance edge weight for --svgfDenoise (default 4)\n"
              << "  --mlBackendMode=<off|reconstruction_noop> Optional ML reconstruction backend contract (default off)\n"
              << "  --mlModelPackage=<path>        Optional model package path recorded in metrics\n"
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
              << "  --resumeCheckpointJson=<path> Re-run from a completed checkpoint manifest\n"
              << "  --tileManifestJson=<path>     Write deterministic tiled/chunk render plan JSON\n"
              << "  --tileSize=<w,h>              Tile dimensions recorded in tile manifests\n"
              << "  --materialMrDebugJson=<path>  Write first-hit Material_MR debug JSON\n"
              << "  --directLightAuditJson=<path> Write deterministic per-light NEE audit JSON\n"
              << "  --restirDebug[=0|1]           Enable R16 ReSTIR debug inspector plumbing (default 0)\n"
              << "  --restirDebugView=<beauty|candidate_source_id|reservoir_confidence|regir_cell_id|path_guiding_used_mask|restir_pt_reuse_mask|svgf_variance|nan_inf_mask>\n"
              << "                                 Select derived ReSTIR debug output view (default beauty)\n"
              << "  --restirDebugCounters[=0|1]   Include ReSTIR debug counter audit state (default 0)\n"
              << "  --restirDebugMetricsJson=<path> Write compact R16 debug inspector JSON\n"
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
        } else if (arg == "--tonemap") {
            if (!requireValue("--tonemap")) {
                return false;
            }
            try {
                int parsed = std::stoi(value);
                if (parsed < 1 || parsed > 4) {
                    error = "--tonemap must be in [1,4]";
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
            } else if (lower == "restir_gi_prototype" ||
                       lower == "restir_gi_diffuse" ||
                       lower == "restir_gi" ||
                       lower == "diffuse_prototype" ||
                       lower == "1") {
                options.restirGiMode = PathTracer::RenderSettings::RestirGiMode::DiffusePrototype;
            } else {
                error = "Invalid value for --restirGiMode (expected off, restir_gi_prototype, or restir_gi_diffuse)";
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
            } else if (lower == "restir_pt_research" ||
                       lower == "restir_pt_scaffold" ||
                       lower == "research" ||
                       lower == "1") {
                options.restirPtMode = PathTracer::RenderSettings::RestirPtMode::Research;
            } else if (lower == "restir_pt_experimental_path_reuse" ||
                       lower == "restir_pt_path_reuse" ||
                       lower == "experimental_path_reuse" ||
                       lower == "path_reuse" ||
                       lower == "2") {
                options.restirPtMode = PathTracer::RenderSettings::RestirPtMode::ExperimentalPathReuse;
            } else {
                error = "Invalid value for --restirPtMode (expected off, restir_pt_research, or restir_pt_experimental_path_reuse)";
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
            } else {
                error = "Invalid value for --mlBackendMode (expected off or reconstruction_noop)";
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
        !ApplyRenderQueueJson(fs::path(options.resumeCheckpointJsonPath), options, error)) {
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
        case PathTracer::RenderSettings::RestirGiMode::DiffusePrototype:
            return "restir_gi_prototype";
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
        case PathTracer::RenderSettings::RestirPtMode::Research:
            return "restir_pt_research";
        case PathTracer::RenderSettings::RestirPtMode::ExperimentalPathReuse:
            return "restir_pt_experimental_path_reuse";
    }
    return "unknown";
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
    std::ofstream out(jsonPath);
    if (!out.is_open()) {
        error = "Failed to open camera search report: " + jsonPath.string();
        return;
    }
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
        error = "Failed while writing camera search report: " + jsonPath.string();
    }
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
        settings.enablePathDebug = true;
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

const char* WorkingColorSpaceLabel(const PathTracer::RenderSettings& settings) {
    return (settings.workingColorSpace == PathTracer::RenderSettings::WorkingColorSpace::ACEScg)
        ? "ACEScg"
        : "Linear sRGB";
}

const char* FireflyClampModeLabel(CliOptions::FireflyClampMode mode) {
    switch (mode) {
        case CliOptions::FireflyClampMode::Luminance:
            return "luminance";
        case CliOptions::FireflyClampMode::MaxComponent:
            return "maxcomponent";
        case CliOptions::FireflyClampMode::Off:
        default:
            return "off";
    }
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
                  << ((geometryBytes + textureBytes + accelBytes) * 100.0 /
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

void PrintEmissiveInventoryReport(const PathTracer::SceneResources& resources) {
    std::cout << "[Scene] Emissive primitives: " << resources.emissivePrimitiveCount() << std::endl;
    std::cout << "[Scene] Total emitted power: " << std::fixed << std::setprecision(4)
              << resources.totalEmittedPower() << std::endl;
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
    std::ofstream out(jsonPath);
    if (!out.is_open()) {
        error = "Failed to open RIS thresholds JSON for writing: " + jsonPath.string();
        return false;
    }
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
        error = "Failed while writing RIS thresholds JSON: " + jsonPath.string();
        return false;
    }
    return true;
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

void ApplyRestirDebugView(const PathTracer::RenderSettings& settings,
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
        settings.directLightMode != PathTracer::RenderSettings::DirectLightMode::LegacyRect &&
        settings.directLightMode != PathTracer::RenderSettings::DirectLightMode::BaselineEmissive;
    const bool worldActive = RestirWorldModeActive(settings.directLightMode);
    const bool pathGuidingUsed = output.pbrMetrics.pathGuidingUsedCount > 0u;
    const bool restirPtReuseUsed = output.pbrMetrics.restirPtReuseAppliedCount > 0u;
    const bool radianceCacheUsed = output.pbrMetrics.radianceCacheHitCount > 0u;
    const bool svgfActive = settings.svgfDenoiseEnabled;
    const uint32_t modeSeed =
        static_cast<uint32_t>(settings.directLightMode) * 17u +
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

void WriteAssetPipelineReportJson(std::ofstream& outFile,
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
    out << "  \"render_profile\": \"" << RenderProfileLabel(options.renderProfile) << "\",\n";
    out << "  \"render_profile_maturity\": \"" << RenderProfileMaturityLabel(options.renderProfile) << "\",\n";
    out << "  \"scene\": \"" << JsonEscape(options.scene) << "\",\n";
    out << "  \"backend\": \"" << (options.backend == HeadlessBackend::Embree ? "embree" : "metal") << "\",\n";
    out << "  \"output\": \"" << JsonEscape(outputPath.string()) << "\",\n";
    out << "  \"format\": \"" << PathTracer::FormatExtension(options.format) << "\",\n";
    out << "  \"spp_total\": " << sppTotal << ",\n";
    out << "  \"settings\": {\n";
    out << "    \"schema_version\": \"pathtracer.render_settings.v1\",\n";
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
    out << "    \"direct_light_mode\": \"" << DirectLightModeLabel(settings.directLightMode) << "\",\n";
    out << "    \"ris_candidate_count\": " << settings.risCandidateCount << ",\n";
    out << "    \"enable_specular_nee\": " << (settings.enableSpecularNee ? "true" : "false") << ",\n";
    out << "    \"enable_mnee\": " << (settings.enableMnee ? "true" : "false") << ",\n";
    out << "    \"restir_gi_mode\": \"" << RestirGiModeLabel(settings.restirGiMode) << "\",\n";
    out << "    \"path_guiding_mode\": \"" << PathGuidingModeLabel(settings.pathGuidingMode) << "\",\n";
    out << "    \"restir_pt_mode\": \"" << RestirPtModeLabel(settings.restirPtMode) << "\",\n";
    out << "    \"radiance_cache_mode\": \"" << RadianceCacheModeLabel(settings.radianceCacheMode) << "\",\n";
    out << "    \"caustic_transport_mode\": \"" << CausticTransportModeLabel(settings.causticTransportMode) << "\",\n";
    out << "    \"spectral_rendering_mode\": \""
        << SpectralRenderingModeLabel(settings.spectralRenderingMode) << "\",\n";
    out << "    \"spectral_texture_policy\": \""
        << SpectralTexturePolicyLabel(settings.spectralTexturePolicy) << "\",\n";
    out << "    \"ml_backend_mode\": \"" << MlBackendModeLabel(settings.mlBackendMode) << "\",\n";
    out << "    \"ml_backend_provider\": \""
        << (settings.mlBackendMode == PathTracer::RenderSettings::MlBackendMode::Off
                ? "disabled"
                : "MLBackend.NoOp")
        << "\",\n";
    out << "    \"ml_model_package_path\": \"" << JsonEscape(settings.mlModelPackagePath) << "\",\n";
    out << "    \"ml_debug_enabled\": " << (settings.mlDebug ? "true" : "false") << ",\n";
    out << "    \"restir_debug_enabled\": " << (settings.restirDebugEnabled ? "true" : "false") << ",\n";
    out << "    \"restir_debug_view\": \"" << RestirDebugViewLabel(settings.restirDebugView) << "\",\n";
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
    out << "      \"intensity\": " << settings.environmentIntensity << "\n";
    out << "    },\n";
    out << "    \"tonemap\": {\n";
    out << "      \"mode\": " << settings.tonemapMode << ",\n";
    out << "      \"aces_variant\": " << settings.acesVariant << ",\n";
    out << "      \"exposure\": " << settings.exposure << ",\n";
    out << "      \"working_color_space\": \"" << WorkingColorSpaceLabel(settings) << "\"\n";
    out << "    },\n";
    out << "    \"offline_post\": {\n";
    out << "      \"denoise_enabled\": " << (settings.denoiseEnabled ? "true" : "false") << ",\n";
    out << "      \"denoise_hdr\": " << (settings.denoiseHdr ? "true" : "false") << ",\n";
    out << "      \"denoise_use_albedo\": " << (settings.denoiseUseAlbedo ? "true" : "false") << ",\n";
    out << "      \"denoise_use_normal\": " << (settings.denoiseUseNormal ? "true" : "false") << ",\n";
    out << "      \"svgf_denoise_enabled\": " << (settings.svgfDenoiseEnabled ? "true" : "false") << "\n";
    out << "    },\n";
    out << "    \"resource_policy\": {\n";
    out << "      \"budget_policy\": \"" << BudgetPolicyLabel(options.budgetPolicy) << "\",\n";
    out << "      \"pilot_mode\": \"" << PilotModeLabel(options.pilotMode) << "\",\n";
    out << "      \"texture_max_dim\": " << (options.textureMaxDimSet ? options.textureMaxDim : 0u) << ",\n";
    out << "      \"max_texture_bytes\": " << (options.maxTextureBytesSet ? options.maxTextureBytes : 0u) << "\n";
    out << "    }\n";
    out << "  }\n";
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

    std::error_code removeEc;
    fs::remove(path, removeEc);
    std::error_code renameEc;
    fs::rename(tmpPath, path, renameEc);
    if (renameEc) {
        error = "Failed to publish sidecar file '" + path.string() + "': " + renameEc.message();
        return false;
    }
    return true;
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
    if (rawOutputPath) {
        json << "  \"raw_output\": \"" << JsonEscape(rawOutputPath->string()) << "\",\n";
    }
    if (denoisedOutputPath) {
        json << "  \"denoised_output\": \"" << JsonEscape(denoisedOutputPath->string()) << "\",\n";
    }
    json << "  \"resolution\": {\"width\": " << output.width << ", \"height\": " << output.height << "},\n";
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
    json << "  }\n";
    json << "}\n";
    return json.str();
}

std::string BuildCheckpointManifestJson(const CliOptions& options,
                                        const PathTracer::RenderSettings& settings,
                                        const HeadlessRenderOutput& output,
                                        const fs::path& outputPath,
                                        const fs::path* metadataPath,
                                        const fs::path* settingsPath,
                                        const fs::path* queuePath) {
    std::ostringstream json;
    json << std::fixed << std::setprecision(9);
    json << "{\n";
    json << "  \"schema_version\": \"pathtracer.render_checkpoint.v1\",\n";
    json << "  \"checkpoint_kind\": \"completed_single_frame\",\n";
    json << "  \"resume_policy\": \"full_frame_rerender_from_checkpoint_manifest\",\n";
    json << "  \"sample_accumulation_resume_supported\": false,\n";
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
    std::ofstream outFile(jsonPath);
    if (!outFile.is_open()) {
        error = "Failed to open metrics file for writing: " + jsonPath.string();
        return false;
    }

    outFile << std::fixed << std::setprecision(9);
    outFile << "{\n";
    outFile << "  \"scene\": \"" << JsonEscape(options.scene) << "\",\n";
    outFile << "  \"backend\": \"" << (options.backend == HeadlessBackend::Embree ? "embree" : "metal")
            << "\",\n";
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
    outFile << "    \"execution_mode\": \""
            << (settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront
                    ? "wavefront"
                    : "megakernel")
            << "\",\n";
    outFile << "    \"wavefront_scheduling_policy\": \""
            << WavefrontSchedulingPolicyLabel(settings.wavefrontSchedulingPolicy) << "\",\n";
    outFile << "    \"wavefront_compaction_enabled\": "
            << (settings.wavefrontCompactionEnabled ? "true" : "false") << ",\n";
    outFile << "    \"wavefront_overflow_policy\": \"clamp_and_report\",\n";
    outFile << "    \"wavefront_material_ray_sorting\": \"experimental_disabled\",\n";
    outFile << "    \"wavefront_active_count_indirect_stages\": [\"intersect\"],\n";
    outFile << "    \"wavefront_direct_grid_stages\": [\"shade\", \"shadow\"],\n";
    outFile << "    \"wavefront_stage_dispatch_modes\": {"
            << "\"intersect\":\"active_count_indirect\","
            << "\"shade\":\"direct_grid_fallback\","
            << "\"shadow\":\"direct_grid_sparse_storage\"},\n";
    outFile << "    \"wavefront_m1_bridge_status\": \"intersect active-count indirect is live; shade and sparse shadow storage remain explicitly reported until dense shade/shadow queues replace the direct-grid paths\",\n";
    outFile << "    \"direct_light_mode\": \"" << DirectLightModeLabel(settings.directLightMode) << "\",\n";
    outFile << "    \"restir_gi_mode\": \"" << RestirGiModeLabel(settings.restirGiMode) << "\",\n";
    outFile << "    \"path_guiding_mode\": \"" << PathGuidingModeLabel(settings.pathGuidingMode) << "\",\n";
    outFile << "    \"path_guiding_strength\": " << settings.pathGuidingStrength << ",\n";
    outFile << "    \"path_guiding_cell_size\": " << settings.pathGuidingCellSize << ",\n";
    outFile << "    \"path_guiding_training_interval\": " << settings.pathGuidingTrainingInterval << ",\n";
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
    outFile << "    \"radiance_cache_provider\": \"RadianceCache.WorldGrid\",\n";
    outFile << "    \"radiance_cache_storage\": \"private_transport_memory\",\n";
    outFile << "    \"radiance_cache_query_contract\": [\"canQuery\", \"query\", \"train\", \"confidence\", \"variance\", \"cacheability\", \"invalidation\", \"debugRecord\"],\n";
    outFile << "    \"radiance_cache_fallback_policy\": \"confidence_or_bias_risk_fallback_to_path_tracing\",\n";
    outFile << "    \"radiance_cache_reconstruction_confidence_source\": \"RadianceEstimate.radianceConfidence.w\",\n";
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
    outFile << "    \"spectral_sampling_strategy\": \"hero_wavelength_uniform_nm_deterministic_per_path\",\n";
    outFile << "    \"spectral_display_conversion\": \"sampled_reflectance_times_approx_rgb_sensor_basis_to_existing_output_transform\",\n";
    outFile << "    \"spectral_dispersion_strength\": "
            << settings.spectralDispersionStrength << ",\n";
    outFile << "    \"spectral_debug_enabled\": "
            << (settings.spectralDebug ? "true" : "false") << ",\n";
    outFile << "    \"ml_backend_mode\": \"" << MlBackendModeLabel(settings.mlBackendMode) << "\",\n";
    outFile << "    \"ml_backend_provider\": \""
            << (settings.mlBackendMode == PathTracer::RenderSettings::MlBackendMode::Off
                    ? "disabled"
                    : "MLBackend.NoOp")
            << "\",\n";
    outFile << "    \"ml_model_package_path\": \""
            << JsonEscape(settings.mlModelPackagePath) << "\",\n";
    outFile << "    \"ml_tensor_contract\": \"radiance,albedo,normal,depth,motion,variance,confidence -> reconstructed_radiance,confidence\",\n";
    outFile << "    \"ml_capability_gate\": \"optional; default off; no-op backend preserves transport output\",\n";
    outFile << "    \"ml_debug_enabled\": " << (settings.mlDebug ? "true" : "false") << ",\n";
    outFile << "    \"restir_debug_enabled\": " << (settings.restirDebugEnabled ? "true" : "false") << ",\n";
    outFile << "    \"restir_debug_view\": \"" << RestirDebugViewLabel(settings.restirDebugView) << "\",\n";
    outFile << "    \"restir_debug_counters\": " << (settings.restirDebugCounters ? "true" : "false") << ",\n";
    outFile << "    \"ris_candidate_count\": " << settings.risCandidateCount << ",\n";
    outFile << "    \"world_reuse_cell_size\": " << settings.worldReuseCellSize << "\n";
    outFile << "  },\n";
    outFile << "  \"enable_mnee\": " << (settings.enableMnee ? "true" : "false") << ",\n";
    outFile << "  \"enable_specular_nee\": " << (settings.enableSpecularNee ? "true" : "false") << ",\n";
    outFile << "  \"offline_post\": {\n";
    outFile << "    \"denoise_enabled\": " << (options.denoiseEnabled ? "true" : "false") << ",\n";
    outFile << "    \"denoise_hdr\": " << (options.denoiseHdr ? "true" : "false") << ",\n";
    outFile << "    \"denoise_use_albedo\": " << (options.denoiseUseAlbedo ? "true" : "false") << ",\n";
    outFile << "    \"denoise_use_normal\": " << (options.denoiseUseNormal ? "true" : "false") << ",\n";
    outFile << "    \"svgf_denoise_enabled\": " << (settings.svgfDenoiseEnabled ? "true" : "false") << ",\n";
    outFile << "    \"svgf_atrous_iterations\": " << settings.svgfAtrousIterations << ",\n";
    outFile << "    \"svgf_normal_sigma\": " << settings.svgfNormalSigma << ",\n";
    outFile << "    \"svgf_albedo_sigma\": " << settings.svgfAlbedoSigma << ",\n";
    outFile << "    \"svgf_luminance_sigma\": " << settings.svgfLuminanceSigma << ",\n";
    outFile << "    \"firefly_clamp_stage\": \"per_sample_contribution\",\n";
    outFile << "    \"firefly_clamp_mode\": \"" << FireflyClampModeLabel(options.fireflyClampMode) << "\",\n";
    outFile << "    \"firefly_clamp_value\": " << options.fireflyClampValue << "\n";
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
    outFile << "    \"wavefront_overflow_policy\": \"clamp_and_report\",\n";
    outFile << "    \"wavefront_material_ray_sorting\": \"experimental_disabled\",\n";
    outFile << "    \"wavefront_active_count_indirect_stages\": [\"intersect\"],\n";
    outFile << "    \"wavefront_direct_grid_stages\": [\"shade\", \"shadow\"],\n";
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
            << ",\"ml_backend_mode\":\"" << MlBackendModeLabel(settings.mlBackendMode)
            << "\",\"spectral_mode\":\""
            << SpectralRenderingModeLabel(settings.spectralRenderingMode)
            << "\",\"volume_mode\":\"" << VolumeTransportModeLabel(settings.volumeTransportMode)
            << "\"},\n";
    outFile << "    \"pass_timing_summary\": {"
            << "\"gpu_bucket_timing_available\":"
            << (output.performance.gpuBucketTimingAvailable ? "true" : "false")
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
    outFile << "    \"transport_memory_invalidated_record_count\": "
            << output.performance.transportMemoryInvalidatedRecordCount << ",\n";
    outFile << "    \"wavefront_queue_compaction_cost_ms\": "
            << output.performance.wavefrontQueueCompactionCostMs << ",\n";
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
                        !directRestirRequested
                            ? "disabled"
                            : (explicitRestirDiDispatches > 0u ? "explicit_d_graph"
                                                                : "fused_reference_path"),
                        !directRestirRequested
                            ? "direct light mode does not request ReSTIR DI"
                            : (explicitRestirDiDispatches > 0u
                                   ? "explicit D graph dispatched"
                                   : "explicit D graph did not dispatch; fused/reference path selected"),
                        explicitRestirDiDispatches,
                        "direct_lighting",
                        directRestirRequested && explicitRestirDiDispatches == 0u);
    writeFallbackReport("path_guiding",
                        PathGuidingModeLabel(settings.pathGuidingMode),
                        settings.pathGuidingMode == PathTracer::RenderSettings::PathGuidingMode::Prototype
                            ? "prototype_guiding_provider"
                            : "disabled",
                        output.pbrMetrics.pathGuidingFallbackCount > 0u
                            ? "guiding provider rejected or lacked confidence for some samples"
                            : "no path guiding sample fallback recorded",
                        output.pbrMetrics.pathGuidingFallbackCount,
                        "path_guiding",
                        output.pbrMetrics.pathGuidingFallbackCount > 0u);
    writeFallbackReport("radiance_cache",
                        RadianceCacheModeLabel(settings.radianceCacheMode),
                        settings.radianceCacheMode == PathTracer::RenderSettings::RadianceCacheMode::Prototype
                            ? "prototype_radiance_provider"
                            : "disabled",
                        output.pbrMetrics.radianceCacheFallbackCount > 0u
                            ? "radiance provider miss or low confidence fell back to path continuation"
                            : "no radiance cache fallback recorded",
                        output.pbrMetrics.radianceCacheFallbackCount,
                        "radiance_cache",
                        output.pbrMetrics.radianceCacheFallbackCount > 0u);
    writeFallbackReport("ml_backend",
                        MlBackendModeLabel(settings.mlBackendMode),
                        output.performance.mlBackendActive
                            ? output.performance.mlBackendProvider.c_str()
                            : "disabled",
                        output.performance.mlBackendFallbackReason.empty()
                            ? "ml backend disabled"
                            : output.performance.mlBackendFallbackReason.c_str(),
                        output.performance.mlBackendPassCount,
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
    return true;
}

bool WriteRestirDebugMetricsJson(const fs::path& jsonPath,
                                 const CliOptions& options,
                                 const PathTracer::RenderSettings& settings,
                                 const fs::path& outputPath,
                                 const HeadlessRenderOutput& output,
                                 const ImagePbrMetrics& imageMetrics,
                                 std::string& error) {
    std::ofstream outFile(jsonPath);
    if (!outFile.is_open()) {
        error = "Failed to open ReSTIR debug metrics file for writing: " + jsonPath.string();
        return false;
    }

    const bool directActive =
        settings.directLightMode != PathTracer::RenderSettings::DirectLightMode::LegacyRect &&
        settings.directLightMode != PathTracer::RenderSettings::DirectLightMode::BaselineEmissive;
    const bool worldActive = RestirWorldModeActive(settings.directLightMode);
    const bool giActive =
        settings.restirGiMode == PathTracer::RenderSettings::RestirGiMode::DiffusePrototype;
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
    outFile << "      \"executionMode\": \""
            << (settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront
                    ? "wavefront"
                    : "megakernel")
            << "\",\n";
    outFile << "      \"wavefrontSchedulingPolicy\": \""
            << WavefrontSchedulingPolicyLabel(settings.wavefrontSchedulingPolicy) << "\",\n";
    outFile << "      \"directLightMode\": \"" << DirectLightModeLabel(settings.directLightMode) << "\",\n";
    outFile << "      \"restirGiMode\": \"" << RestirGiModeLabel(settings.restirGiMode) << "\",\n";
    outFile << "      \"pathGuidingMode\": \"" << PathGuidingModeLabel(settings.pathGuidingMode) << "\",\n";
    outFile << "      \"restirPtMode\": \"" << RestirPtModeLabel(settings.restirPtMode) << "\",\n";
    outFile << "      \"svgfDenoise\": " << (settings.svgfDenoiseEnabled ? "true" : "false") << "\n";
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
    outFile << "      \"radianceCacheQueryCount\": " << output.pbrMetrics.radianceCacheQueryCount << ",\n";
    outFile << "      \"radianceCacheHitCount\": " << output.pbrMetrics.radianceCacheHitCount << ",\n";
    outFile << "      \"radianceCacheMissCount\": " << output.pbrMetrics.radianceCacheMissCount << ",\n";
    outFile << "      \"radianceCacheTrainCount\": " << output.pbrMetrics.radianceCacheTrainCount << ",\n";
    outFile << "      \"radianceCacheFallbackCount\": " << output.pbrMetrics.radianceCacheFallbackCount << ",\n";
    outFile << "      \"radianceCacheInvalidCount\": " << output.pbrMetrics.radianceCacheInvalidCount << ",\n";
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
            << ((!giActive && !pathGuidingActive && !restirPtActive && !radianceCacheActive && !settings.svgfDenoiseEnabled)
                    ? "green"
                    : "yellow")
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
    outFile << "    \"availableAOVs\": [\"beauty\", \"candidate_source_id\", \"reservoir_confidence\", \"regir_cell_id\", \"path_guiding_used_mask\", \"restir_pt_reuse_mask\", \"svgf_variance\", \"nan_inf_mask\"],\n";
    outFile << "    \"unavailable\": [\"selected_light_id per-pixel image\", \"temporal rejection image\", \"spatial reuse-count image\", \"live ReGIR hit/miss hardware counters\"]\n";
    outFile << "  }\n";
    outFile << "}\n";

    if (!outFile.good()) {
        error = "Failed while writing ReSTIR debug metrics JSON: " + jsonPath.string();
        return false;
    }
    return true;
}
bool WriteMaterialMrDebugJson(const fs::path& jsonPath,
                              const CliOptions& options,
                              const fs::path& outputPath,
                              const HeadlessRenderOutput::MaterialMrDebugDump& debugDump,
                              std::string& error) {
    std::ofstream outFile(jsonPath);
    if (!outFile.is_open()) {
        error = "Failed to open Material_MR debug file for writing: " + jsonPath.string();
        return false;
    }
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
    auto jsonFloat = [&](float value) {
        return std::isfinite(value) ? value : 0.0f;
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
    outFile << "  \"env_lod\": " << debugDump.envLod << "\n";
    outFile << "}\n";
    if (!outFile.good()) {
        error = "Failed while writing Material_MR debug JSON: " + jsonPath.string();
        return false;
    }
    return true;
}

bool WriteDirectLightAuditJson(const fs::path& jsonPath,
                               const CliOptions& options,
                               const fs::path& outputPath,
                               const HeadlessRenderOutput::DirectLightAuditDump& auditDump,
                               std::string& error) {
    std::ofstream outFile(jsonPath);
    if (!outFile.is_open()) {
        error = "Failed to open direct-light audit file for writing: " + jsonPath.string();
        return false;
    }
    outFile << std::fixed << std::setprecision(9);
    auto writeFloat3 = [&](const char* name, simd::float3 v, bool trailing) {
        outFile << "  \"" << name << "\": [" << v.x << ", " << v.y << ", " << v.z << "]";
        outFile << (trailing ? ",\n" : "\n");
    };
    auto jsonFloat = [&](float value) {
        return std::isfinite(value) ? value : 0.0f;
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
                << jsonFloat(sampledPoint.x) << ", "
                << jsonFloat(sampledPoint.y) << ", "
                << jsonFloat(sampledPoint.z) << "],\n";
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
        outFile << "        \"hw_raw_distance\": " << jsonFloat(entry.shadowPath.hwRawDistance) << ",\n";
        outFile << "        \"hw_resolved_hit\": "
                << (entry.shadowPath.hwResolvedHit ? "true" : "false") << ",\n";
        outFile << "        \"hw_resolved_mesh_index\": " << entry.shadowPath.hwResolvedMeshIndex << ",\n";
        outFile << "        \"hw_resolved_material_index\": " << entry.shadowPath.hwResolvedMaterialIndex << ",\n";
        outFile << "        \"hw_resolved_primitive_index\": " << entry.shadowPath.hwResolvedPrimitiveIndex << ",\n";
        outFile << "        \"hw_resolved_primitive_type\": " << entry.shadowPath.hwResolvedPrimitiveType << ",\n";
        outFile << "        \"hw_resolved_front_face\": "
                << (entry.shadowPath.hwResolvedFrontFace ? "true" : "false") << ",\n";
        outFile << "        \"hw_resolved_distance\": " << jsonFloat(entry.shadowPath.hwResolvedDistance) << ",\n";
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
        outFile << "        \"embedded_sw_distance\": " << jsonFloat(entry.shadowPath.embeddedSwDistance) << ",\n";
        outFile << "        \"embedded_sw_hit_position\": ["
                << jsonFloat(entry.shadowPath.embeddedSwHitPosition.x) << ", "
                << jsonFloat(entry.shadowPath.embeddedSwHitPosition.y) << ", "
                << jsonFloat(entry.shadowPath.embeddedSwHitPosition.z) << "],\n";
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
        outFile << "        \"shadow_t_min\": " << jsonFloat(entry.shadowPath.shadowTMin) << ",\n";
        outFile << "        \"shadow_t_max\": " << jsonFloat(entry.shadowPath.shadowTMax) << ",\n";
        outFile << "        \"ray_origin_epsilon\": " << jsonFloat(entry.shadowPath.rayOriginEpsilon) << ",\n";
        outFile << "        \"hardware_occlusion_epsilon\": " << jsonFloat(entry.shadowPath.hardwareOcclusionEpsilon) << ",\n";
        outFile << "        \"rejection_mask\": " << entry.shadowPath.rejectionMask << ",\n";
        outFile << "        \"light_two_sided\": "
                << (entry.shadowPath.lightTwoSided ? "true" : "false") << ",\n";
        outFile << "        \"final_visible\": "
                << (entry.shadowPath.finalVisible ? "true" : "false") << "\n";
        outFile << "      }\n";
        outFile << "    }" << (i + 1u < auditDump.entries.size() ? ",\n" : "\n");
    }
    outFile << "  ]\n";
    outFile << "}\n";
    if (!outFile.good()) {
        error = "Failed while writing direct-light audit JSON: " + jsonPath.string();
        return false;
    }
    return true;
}

bool WriteLuminanceHistogramText(const fs::path& histogramPath,
                                 const ImagePbrMetrics& imageMetrics,
                                 std::string& error) {
    std::ofstream outFile(histogramPath);
    if (!outFile.is_open()) {
        error = "Failed to open histogram file for writing: " + histogramPath.string();
        return false;
    }

    outFile << std::fixed << std::setprecision(9);
    outFile << "# luminance_histogram type=log10 buckets="
            << imageMetrics.histogramCounts.size() << "\n";
    outFile << "# bin lower upper count\n";
    const size_t bucketCount = imageMetrics.histogramCounts.size();
    if (bucketCount == 0u) {
        return true;
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

    if (!outFile.good()) {
        error = "Failed while writing histogram file: " + histogramPath.string();
        return false;
    }
    return true;
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
                  << ",fallback=" << output.pbrMetrics.spectralFallbackCount << ")"
                  << std::endl;
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

    std::ofstream json(outDir / "manifest.json");
    if (!json.is_open()) {
        error = "Failed to open material texture manifest for writing: " +
                (outDir / "manifest.json").string();
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
    return true;
}

int main(int argc, const char** argv) {
    @autoreleasepool {
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

        sceneResources.buildCPUPackedSceneData();
        EnsureSceneDirectiveAssetPipelineReport(options, sceneManager, sceneResources);
        PrintEmissiveInventoryReport(sceneResources);

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
            !sceneResources.reloadEnvironmentIfNeeded(settings.environmentMapPath)) {
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
                                    (renderSettings.denoiseEnabled || renderSettings.svgfDenoiseEnabled) &&
                                        (renderSettings.denoiseUseAlbedo || renderSettings.denoiseUseNormal),
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
        ApplyRestirDebugView(settings, output);

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
            options.fireflyClampMode != CliOptions::FireflyClampMode::Off) {
            std::cout << "[Beauty] denoise=" << (settings.denoiseEnabled ? "on" : "off")
                      << " svgf=" << (settings.svgfDenoiseEnabled ? "on" : "off")
                      << " hdr=" << (options.denoiseHdr ? "1" : "0")
                      << " useAlbedo=" << (options.denoiseUseAlbedo ? "1" : "0")
                      << " useNormal=" << (options.denoiseUseNormal ? "1" : "0")
                      << " fireflyClampStage=per_sample_contribution"
                      << " fireflyClampMode=" << FireflyClampModeLabel(options.fireflyClampMode)
                      << " fireflyClampValue=" << options.fireflyClampValue
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

        if (options.checkpointManifestJsonProvided || options.resumeCheckpointJsonProvided || debugBundleDir) {
            const fs::path checkpointPath = options.checkpointManifestJsonProvided
                ? fs::path(options.checkpointManifestJsonPath)
                : (debugBundleDir ? (*debugBundleDir / "checkpoint_manifest.json")
                                  : fs::path(outputFsPath.string() + ".checkpoint.json"));
            std::string checkpointError;
            const std::string checkpointJson =
                BuildCheckpointManifestJson(options,
                                            settings,
                                            output,
                                            outputFsPath,
                                            writtenMetadataPath ? &*writtenMetadataPath : nullptr,
                                            writtenSettingsPath ? &*writtenSettingsPath : nullptr,
                                            writtenQueuePath ? &*writtenQueuePath : nullptr);
            if (!WriteTextFileAtomic(checkpointPath, checkpointJson, checkpointError)) {
                std::cerr << checkpointError << std::endl;
                return 1;
            }
            writtenCheckpointPath = checkpointPath;
            std::cout << "Checkpoint manifest written to: " << checkpointPath << std::endl;
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
