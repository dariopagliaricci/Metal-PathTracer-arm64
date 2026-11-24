#include "renderer/SettingsUtils.h"

#include <iomanip>
#include <sstream>
#include "renderer/SceneResources.h"
#include <simd/simd.h>
#include <string>

namespace PathTracer {

bool ResolveSourceCapabilities(const SceneResources& scene, RenderSettings& settings,
                               bool embree, bool strict, std::string& diagnostic) {
    bool opacity = false, source = settings.sourceTransport || settings.denoiseMirrorGuides ||
        settings.denoiseTransmissionGuides;
    const auto* materials = scene.materialsData();
    for (uint32_t i = 0; materials && i < scene.materialCount(); ++i) {
        opacity |= materials[i].textureControl.y != 0.0f;
        source |= (materials[i].materialFlags & PathTracerShaderTypes::kMaterialFlagSourceFeatures) != 0u;
    }
    settings.sourceFeatures = source || opacity;
    if (settings.sourceRayEpsilon > 0.0f && !settings.sourceTransport) {
        diagnostic = "renderer sourceRayEpsilon requires sourceTransport=1.";
        if (strict) return false;
        settings.sourceRayEpsilon = 0.0f;
        diagnostic += " Ignored.";
    }
    if (embree && source) {
        // EP8: Embree runs the source closures and the authored dielectric normal from the shared Metal/C++ header
        // (shaders/shared/source_closures.h), and source transport with interface precision, medium boundaries and
        // the mirror and transmission denoiser guides (EP8d). It refuses, by name, the source features it does not port yet.
        std::string missing;
        auto add = [&missing](const std::string& feature) { missing += (missing.empty() ? "" : ", ") + feature; };
        // Bump maps are ported (EP8c): Embree perturbs the authored normal as Metal's source_bump_normal does.
        // Thin source sheets under source transport are ported (EP8c): stochastic forward events on path rays, the forward
        // transmission on shadow rays, as Metal's wavefront.
        uint32_t opacityCount = 0, interfaceCount = 0, explicitHoleCount = 0;
        for (uint32_t i = 0; materials && i < scene.materialCount(); ++i) {
            const auto& m = materials[i];
            if ((m.materialFlags & PathTracerShaderTypes::kMaterialFlagSourceFeatures) == 0u) continue;
            opacityCount += m.textureControl.y != 0.0f ? 1u : 0u;
            interfaceCount += (m.materialFlags & PathTracerShaderTypes::kMaterialFlagInterfacePrecision) != 0u ? 1u : 0u;
            explicitHoleCount += ((m.materialFlags & PathTracerShaderTypes::kMaterialFlagExplicitInterior) != 0u &&
                                  m.pbrExtras.w > 0.5f) ? 1u : 0u;
        }
        // Source opacity is ported under source transport (EP8c): path rays pass with probability 1 - opacity, and shadow
        // rays are weighted by 1 - opacity at each hit. Without transport Metal's opacity spawns use the source epsilon,
        // which Embree takes only under transport.
        if (opacityCount && !settings.sourceTransport)
            add("opacity without sourceTransport (" + std::to_string(opacityCount) + " materials)");
        // Without source transport, interfacePrecision changes only that material's spawns on Metal (not ported).
        if (interfaceCount && !settings.sourceTransport)
            add("interfacePrecision without sourceTransport (" + std::to_string(interfaceCount) + " materials)");
        // Explicit-interior and inherited medium boundaries are ported (EP8d). Under source transport Metal's shadow
        // rays re-select the medium at each alpha hole they pass; Embree's take one medium per segment.
        if (explicitHoleCount && settings.sourceTransport)
            add("alpha cutouts on explicit-interior materials under sourceTransport (" +
                std::to_string(explicitHoleCount) + " materials)");
        if (!missing.empty()) {
            diagnostic = "Embree does not support these source features yet: " + missing + ".";
            return false;
        }
    }
    if ((opacity || settings.sourceTransport || settings.denoiseMirrorGuides ||
         settings.denoiseTransmissionGuides) && settings.executionMode != RenderSettings::ExecutionMode::Wavefront) {
        diagnostic = "Source opacity/forward visibility or mirror/transmission denoise guides require wavefront execution.";
        if (strict) return false;
        settings.executionMode = RenderSettings::ExecutionMode::Wavefront;
        settings.sourceForcedWavefront = true;
        diagnostic += " Switched to wavefront.";
    }
    if (source && static_cast<uint32_t>(settings.directLightMode) > 1u) {
        diagnostic = "Source closures/visibility support legacy or baseline_emissive lighting; RIS/ReSTIR is unsupported.";
        if (strict) return false;
        settings.directLightMode = RenderSettings::DirectLightMode::BaselineEmissive;
        diagnostic += " Switched to baseline_emissive.";
    }
    return true;
}

std::string SpectralUnsupportedCombination(const RenderSettings& settings) {
    if (settings.spectralRenderingMode == RenderSettings::SpectralRenderingMode::Off) {
        return {};
    }
    // Per-pixel NEE through the shared light-sample conversion. RIS, ReGIR and ReSTIR DI score or reuse light
    // samples with unconverted RGB radiance and materials across pixels and frames.
    switch (settings.directLightMode) {
        case RenderSettings::DirectLightMode::LegacyRect:
        case RenderSettings::DirectLightMode::BaselineEmissive:
            break;
        case RenderSettings::DirectLightMode::Ris: return "directLightMode=ris";
        case RenderSettings::DirectLightMode::RisSpatialReuse: return "directLightMode=ris_spatial";
        case RenderSettings::DirectLightMode::RisTemporalReuse: return "directLightMode=ris_temporal";
        case RenderSettings::DirectLightMode::RisWorldReuse: return "directLightMode=ris_world";
        case RenderSettings::DirectLightMode::RisRegirCache: return "directLightMode=ris_regir";
        case RenderSettings::DirectLightMode::RestirDi: return "directLightMode=restir_di";
        case RenderSettings::DirectLightMode::RestirDiRegirHybrid: return "directLightMode=restir_di_regir_hybrid";
    }
    if (settings.restirGiMode != RenderSettings::RestirGiMode::Off) return "restirGiMode (not off)";
    if (settings.restirPtMode != RenderSettings::RestirPtMode::Off) return "restirPtMode (not off)";
    if (settings.pathGuidingMode != RenderSettings::PathGuidingMode::Off) return "pathGuidingMode (not off)";
    if (settings.radianceCacheMode != RenderSettings::RadianceCacheMode::Off) return "radianceCacheMode (not off)";
    // Medium coefficients are RGB and would need per-wavelength conversion.
    if (settings.volumeTransportMode != RenderSettings::VolumeTransportMode::Off) return "volumeTransportMode=homogeneous";
    // Delta chains (specular NEE, MNEE) sample unconverted materials with an undispersed IOR.
    if (settings.enableSpecularNee) return "enableSpecularNee=1 (pass --enableSpecularNee=0)";
    if (settings.enableMnee) return "enableMnee=1";
    if (settings.svgfDenoiseEnabled) return "svgfDenoise=1";
    if (settings.mlBackendMode != RenderSettings::MlBackendMode::Off) return "mlBackendMode (not off)";
    // Tungsten source closures use unconverted source reflectance and Fresnel parameters.
    if (settings.sourceTransport || settings.sourceFeatures) return "Tungsten source closures or sourceTransport";
    // Debug views write colors as radiance, which the accumulation would treat as per-wavelength values.
    if (settings.debugShowBaseColor || settings.debugShowMetallic || settings.debugShowRoughness ||
        settings.debugShowAO ||
        (settings.restirDebugEnabled && settings.restirDebugView != RenderSettings::RestirDebugView::Beauty)) {
        return "debug view (debugShow* or restirDebugView not beauty)";
    }
    return {};
}

std::vector<std::string> SubsurfaceNotScatteringDiagnostics(const SceneResources& scene,
                                                            const RenderSettings& settings) {
    using SssMode = RenderSettings::SssMode;
    std::vector<std::string> lines;
    const auto* materials = scene.materialsData();
    if (!materials) {
        return lines;
    }
    const char* modeName = settings.sssMode == SssMode::Off
                               ? "off"
                               : (settings.sssMode == SssMode::Separable ? "separable" : "randomwalk");
    for (uint32_t i = 0u; i < scene.materialCount(); ++i) {
        const auto& material = materials[i];
        if (static_cast<uint32_t>(material.typeEta.x) != 5u) {
            continue;
        }
        // Mirrors sss_use_separable and sss_use_random_walk in the shaders.
        const bool randomWalk = material.sssParams.y >= 0.5f;
        const bool scatters = randomWalk ? settings.sssMode == SssMode::RandomWalk
                                         : settings.sssMode == SssMode::Separable;
        if (scatters) {
            continue;
        }
        lines.push_back("material '" + scene.materialName(i) + "' (method=" +
                        (randomWalk ? "randomwalk" : "separable") +
                        ") does not scatter under the surface: the renderer's sss mode is " + modeName +
                        ". It is shaded as plain diffuse in its base colour" +
                        (material.sssParams.z > 0.5f ? ", under its coat." : "."));
    }
    return lines;
}

std::string SubsurfaceWalkLossDiagnostic(const std::string& material, uint64_t walks, uint64_t openMesh,
                                         uint64_t stepLimit, uint32_t sssMaxSteps) {
    if (walks == 0u || (openMesh == 0u && stepLimit == 0u)) {
        return {};
    }
    auto percent = [walks](uint64_t count) {
        std::ostringstream text;
        text << std::fixed << std::setprecision(2) << 100.0 * static_cast<double>(count) / static_cast<double>(walks)
             << "%";
        return text.str();
    };
    std::ostringstream line;
    line << "material '" << material << "': ";
    if (stepLimit > 0u) {
        line << stepLimit << " of " << walks << " random walks (" << percent(stepLimit)
             << ") ended at the step limit: sssMaxSteps=" << sssMaxSteps
             << " is cutting walks, and a cut walk ends its path with no light.";
    }
    if (openMesh > 0u) {
        line << (stepLimit > 0u ? " " : "") << openMesh << " of " << walks << " random walks (" << percent(openMesh)
             << ") found themselves outside the mesh, which is not closed there, and ended with no light.";
    }
    return line.str();
}

std::string IgnoredDirectionalLightsDiagnostic(const SceneResources& scene, const RenderSettings& settings) {
    // Only legacy leaves suns out. Homogeneous volumes demote ReSTIR DI to baseline_emissive (RestirDiDemotedByVolume),
    // which samples them.
    if (settings.directLightMode != RenderSettings::DirectLightMode::LegacyRect) {
        return {};
    }
    size_t directional = 0;
    for (const auto& primitive : scene.emissivePrimitives()) {
        const bool emits = primitive.emission.x > 0.0f || primitive.emission.y > 0.0f || primitive.emission.z > 0.0f;
        if ((primitive.flags & PathTracerShaderTypes::kLightPrimitiveFlagDirectional) != 0u && emits) {
            ++directional;
        }
    }
    if (directional == 0) {
        return {};
    }
    return std::to_string(directional) + " directional (sun) light(s) are not sampled in directLightMode=legacy" +
           "; use baseline_emissive or a RIS/ReSTIR mode to light the scene with them.";
}

static inline bool vecNearlyEqual(simd::float3 a, simd::float3 b, float eps = 1e-5f) {
    return nearlyEqual(a.x, b.x, eps) &&
           nearlyEqual(a.y, b.y, eps) &&
           nearlyEqual(a.z, b.z, eps);
}

RadiometricChangeResult DetectRadiometricChange(const RenderSettings& prev,
                                                const RenderSettings& next) {
    RadiometricChangeResult result{};
    auto change = [&](const char* reason) -> RadiometricChangeResult {
        return RadiometricChangeResult{true, reason};
    };

    if (prev.sourceTransport != next.sourceTransport) return change("SOURCE_TRANSPORT");
    if (prev.sourceRayEpsilon != next.sourceRayEpsilon) return change("SOURCE_RAY_EPSILON");
    if (prev.denoiseMirrorGuides != next.denoiseMirrorGuides) return change("DENOISE_MIRROR_GUIDES");
    if (prev.denoiseTransmissionGuides != next.denoiseTransmissionGuides) return change("DENOISE_TRANSMISSION_GUIDES");
    // --- Camera (orbit) ---
    if (prev.cameraControlMode != next.cameraControlMode)                   return change("CAMERA_MODE");
    if (!vecNearlyEqual(prev.cameraTarget, next.cameraTarget))               return change("CAMERA_TARGET");
    if (!nearlyEqual(prev.cameraDistance,     next.cameraDistance))          return change("CAMERA_DIST");
    if (!nearlyEqual(prev.cameraYaw,          next.cameraYaw))               return change("CAMERA_YAW");
    if (!nearlyEqual(prev.cameraPitch,        next.cameraPitch))             return change("CAMERA_PITCH");
    if (!vecNearlyEqual(prev.firstPersonCameraPosition, next.firstPersonCameraPosition)) return change("CAMERA_FPS_POS");
    if (!nearlyEqual(prev.cameraVerticalFov,  next.cameraVerticalFov))       return change("CAMERA_FOV");
    if (!nearlyEqual(prev.cameraDefocusAngle, next.cameraDefocusAngle))      return change("CAMERA_DOF");
    if (!nearlyEqual(prev.cameraFocusDistance,next.cameraFocusDistance))     return change("CAMERA_FOCUS");

    // --- Background / environment ---
    if (prev.backgroundMode != next.backgroundMode) return change("BG_MODE");
    if (!vecNearlyEqual(prev.backgroundColor, next.backgroundColor))        return change("BG_COLOR");
    if (prev.environmentMapPath != next.environmentMapPath)                 return change("ENV_MAP");
    if (!nearlyEqual(prev.environmentRotation,  next.environmentRotation))  return change("ENV_ROT");
    if (!nearlyEqual(prev.environmentIntensity, next.environmentIntensity)) return change("ENV_INTENSITY");
    if (!nearlyEqual(prev.environmentBackgroundScale, next.environmentBackgroundScale)) return change("ENV_BACKGROUND_SCALE");

    // --- Tonemapping/exposure ---
    if (prev.tonemapMode != next.tonemapMode) return change("TONEMAP_MODE");
    if (prev.acesVariant != next.acesVariant) return change("ACES_VARIANT");
    if (!nearlyEqual(prev.exposure,           next.exposure))           return change("EXPOSURE");
    if (!nearlyEqual(prev.reinhardWhitePoint, next.reinhardWhitePoint)) return change("REINHARD_WHITE");
    if (prev.bloomEnabled != next.bloomEnabled) return change("BLOOM_ENABLE");
    if (!nearlyEqual(prev.bloomThreshold, next.bloomThreshold)) return change("BLOOM_THRESHOLD");
    if (!nearlyEqual(prev.bloomIntensity, next.bloomIntensity)) return change("BLOOM_INTENSITY");
    if (!nearlyEqual(prev.bloomRadius, next.bloomRadius)) return change("BLOOM_RADIUS");
    if (prev.workingColorSpace != next.workingColorSpace) return change("COLORSPACE");
    if (prev.gltfViewerCompatibilityMode != next.gltfViewerCompatibilityMode) return change("GLTF_COMPAT");
    if (prev.gltfThinWalledFallback != next.gltfThinWalledFallback) return change("GLTF_THIN");
    if (!nearlyEqual(prev.gltfEmissiveScale, next.gltfEmissiveScale)) return change("GLTF_EMISSIVE");
    if (prev.gltfCompatForceLinearBaseColor != next.gltfCompatForceLinearBaseColor) return change("GLTF_BASECOLOR_LINEAR");
    if (prev.gltfCompatForceLinearEmissive != next.gltfCompatForceLinearEmissive) return change("GLTF_EMISSIVE_LINEAR");
    if (prev.debugShowBaseColor != next.debugShowBaseColor) return change("DEBUG_SHOW_BASECOLOR");
    if (prev.debugShowMetallic != next.debugShowMetallic) return change("DEBUG_SHOW_METALLIC");
    if (prev.debugShowRoughness != next.debugShowRoughness) return change("DEBUG_SHOW_ROUGHNESS");
    if (prev.debugShowAO != next.debugShowAO) return change("DEBUG_SHOW_AO");
    if (prev.debugDisableAO != next.debugDisableAO) return change("DEBUG_DISABLE_AO");
    if (prev.debugAoIndirectOnly != next.debugAoIndirectOnly) return change("DEBUG_AO_INDIRECT_ONLY");
    if (prev.debugDisableNormalMap != next.debugDisableNormalMap) return change("DEBUG_DISABLE_NORMAL");
    if (prev.debugDisableOrmTexture != next.debugDisableOrmTexture) return change("DEBUG_DISABLE_ORM");
    if (prev.debugFlipNormalGreen != next.debugFlipNormalGreen) return change("DEBUG_FLIP_NORMAL_GREEN");
    if (prev.debugSpecularOnly != next.debugSpecularOnly) return change("DEBUG_SPECULAR_ONLY");
    if (!nearlyEqual(prev.debugNormalStrengthScale, next.debugNormalStrengthScale)) return change("DEBUG_NORMAL_SCALE");
    if (prev.normalVarianceRoughness != next.normalVarianceRoughness) return change("NORMAL_VARIANCE_ROUGHNESS");
    if (!nearlyEqual(prev.textureLodBias, next.textureLodBias)) return change("TEXTURE_LOD_BIAS");
    if (!nearlyEqual(prev.debugNormalLodBias, next.debugNormalLodBias)) return change("DEBUG_NORMAL_BIAS");
    if (!nearlyEqual(prev.debugOrmLodBias, next.debugOrmLodBias)) return change("DEBUG_ORM_BIAS");
    if (!nearlyEqual(prev.debugEnvMipOverride, next.debugEnvMipOverride)) return change("DEBUG_ENV_MIP");
    if (prev.debugEnableVisorOverride != next.debugEnableVisorOverride) return change("DEBUG_VISOR_OVERRIDE");
    if (prev.debugVisorOverrideMaterialId != next.debugVisorOverrideMaterialId) return change("DEBUG_VISOR_MAT");
    if (!nearlyEqual(prev.debugVisorOverrideRoughness, next.debugVisorOverrideRoughness)) return change("DEBUG_VISOR_ROUGH");
    if (!nearlyEqual(prev.debugVisorOverrideF0, next.debugVisorOverrideF0)) return change("DEBUG_VISOR_F0");
    if (prev.debugEnvNearest != next.debugEnvNearest) return change("DEBUG_ENV_NEAREST");

    // --- Sampling/variance controls ---
    // samplesPerFrame only changes batching/progress cadence. It must not reset
    // accumulation, especially in headless mode where the renderer adjusts it
    // per dispatch batch while preserving the same image integral.
    if (!nearlyEqual(prev.renderScale, next.renderScale, 1e-4f)) return change("RENDER_SCALE");
    if (prev.executionMode != next.executionMode) return change("RENDER_MODE");
    if (prev.wavefrontSchedulingPolicy != next.wavefrontSchedulingPolicy) return change("WAVEFRONT_POLICY");
    if (prev.wavefrontCompactionEnabled != next.wavefrontCompactionEnabled) return change("WAVEFRONT_COMPACTION");
    if (prev.wavefrontShadeIndirectEnabled != next.wavefrontShadeIndirectEnabled) return change("WAVEFRONT_SHADE_INDIRECT");
    if (prev.wavefrontShadeIndirectProbeEnabled != next.wavefrontShadeIndirectProbeEnabled) return change("WAVEFRONT_SHADE_INDIRECT_PROBE");
    if (prev.wavefrontShadeSurfaceProbeEnabled != next.wavefrontShadeSurfaceProbeEnabled) return change("WAVEFRONT_SHADE_SURFACE_PROBE");
    if (prev.wavefrontShadeMetadataProbeEnabled != next.wavefrontShadeMetadataProbeEnabled) return change("WAVEFRONT_SHADE_METADATA_PROBE");
    if (prev.wavefrontShadeScalarProbeEnabled != next.wavefrontShadeScalarProbeEnabled) return change("WAVEFRONT_SHADE_SCALAR_PROBE");
    if (prev.wavefrontShadeClosureProbeEnabled != next.wavefrontShadeClosureProbeEnabled) return change("WAVEFRONT_SHADE_CLOSURE_PROBE");
    if (prev.wavefrontShadeBsdfProbeEnabled != next.wavefrontShadeBsdfProbeEnabled) return change("WAVEFRONT_SHADE_BSDF_PROBE");
    if (prev.wavefrontShadowIndirectEnabled != next.wavefrontShadowIndirectEnabled) return change("WAVEFRONT_SHADOW_INDIRECT");

    if (prev.fireflyClampEnabled    != next.fireflyClampEnabled)    return change("FIREFLY_ENABLE");
    if (prev.fireflyClampMode       != next.fireflyClampMode)       return change("FIREFLY_MODE");
    if (!nearlyEqual(prev.fireflyClampFactor,          next.fireflyClampFactor))          return change("FIREFLY_FACTOR");
    if (!nearlyEqual(prev.fireflyClampFloor,           next.fireflyClampFloor))           return change("FIREFLY_FLOOR");
    if (!nearlyEqual(prev.throughputClamp,             next.throughputClamp))             return change("THROUGHPUT_CLAMP");
    if (!nearlyEqual(prev.minSpecularPdf,              next.minSpecularPdf))              return change("SPECULAR_PDF");
    if (!nearlyEqual(prev.fireflyClampMaxContribution, next.fireflyClampMaxContribution)) return change("FIREFLY_MAX");

    // --- SSS mode (integrator behavior) ---
    if (prev.sssMode     != next.sssMode)     return change("SSS_MODE");
    if (prev.sssMaxSteps != next.sssMaxSteps) return change("SSS_STEPS");
    if (prev.enableMnee  != next.enableMnee)  return change("MNEE");
    if (prev.enableMneeSecondary != next.enableMneeSecondary) return change("MNEE_SECONDARY");
    if (prev.directLightMode != next.directLightMode) return change("DIRECT_LIGHT_MODE");
    if (prev.risCandidateCount != next.risCandidateCount) return change("RIS_CANDIDATE_COUNT");
    if (prev.spatialReuseNeighborCount != next.spatialReuseNeighborCount) return change("SPATIAL_REUSE_NEIGHBOR_COUNT");
    if (prev.worldReuseCellSize != next.worldReuseCellSize) return change("WORLD_REUSE_CELL_SIZE");
    if (prev.restirGiMode != next.restirGiMode) return change("RESTIR_GI_MODE");
    if (prev.pathGuidingMode != next.pathGuidingMode) return change("PATH_GUIDING_MODE");
    if (!nearlyEqual(prev.pathGuidingStrength, next.pathGuidingStrength)) return change("PATH_GUIDING_STRENGTH");
    if (!nearlyEqual(prev.pathGuidingCellSize, next.pathGuidingCellSize)) return change("PATH_GUIDING_CELL_SIZE");
    if (prev.pathGuidingTrainingInterval != next.pathGuidingTrainingInterval) return change("PATH_GUIDING_TRAINING_INTERVAL");
    if (prev.restirPtMode != next.restirPtMode) return change("RESTIR_PT_MODE");
    if (prev.restirPtMaxReservoirs != next.restirPtMaxReservoirs) return change("RESTIR_PT_MAX_RESERVOIRS");
    if (prev.restirPtDebug != next.restirPtDebug) return change("RESTIR_PT_DEBUG");
    if (!nearlyEqual(prev.restirPtReuseStrength, next.restirPtReuseStrength)) return change("RESTIR_PT_REUSE_STRENGTH");
    if (prev.svgfDenoiseEnabled != next.svgfDenoiseEnabled) return change("SVGF_DENOISE_ENABLED");
    if (prev.svgfDenoiseDirect != next.svgfDenoiseDirect) return change("SVGF_DENOISE_DIRECT");
    if (prev.svgfDenoiseIndirect != next.svgfDenoiseIndirect) return change("SVGF_DENOISE_INDIRECT");
    if (prev.svgfAtrousIterations != next.svgfAtrousIterations) return change("SVGF_ATROUS_ITERATIONS");
    if (!nearlyEqual(prev.svgfNormalSigma, next.svgfNormalSigma)) return change("SVGF_NORMAL_SIGMA");
    if (!nearlyEqual(prev.svgfAlbedoSigma, next.svgfAlbedoSigma)) return change("SVGF_ALBEDO_SIGMA");
    if (!nearlyEqual(prev.svgfLuminanceSigma, next.svgfLuminanceSigma)) return change("SVGF_LUMINANCE_SIGMA");

    return result;
}

bool MarkRadiometricChange(const RenderSettings& prev,
                           const RenderSettings& next) {
    return DetectRadiometricChange(prev, next).changed;
}

} // namespace PathTracer
