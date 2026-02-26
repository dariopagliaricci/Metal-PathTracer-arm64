#include "renderer/SettingsUtils.h"
#include <simd/simd.h>
#include <string>

namespace PathTracer {

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

    // --- Camera (orbit) ---
    if (!vecNearlyEqual(prev.cameraTarget, next.cameraTarget))               return change("CAMERA_TARGET");
    if (!nearlyEqual(prev.cameraDistance,     next.cameraDistance))          return change("CAMERA_DIST");
    if (!nearlyEqual(prev.cameraYaw,          next.cameraYaw))               return change("CAMERA_YAW");
    if (!nearlyEqual(prev.cameraPitch,        next.cameraPitch))             return change("CAMERA_PITCH");
    if (!nearlyEqual(prev.cameraVerticalFov,  next.cameraVerticalFov))       return change("CAMERA_FOV");
    if (!nearlyEqual(prev.cameraDefocusAngle, next.cameraDefocusAngle))      return change("CAMERA_DOF");
    if (!nearlyEqual(prev.cameraFocusDistance,next.cameraFocusDistance))     return change("CAMERA_FOCUS");

    // --- Background / environment ---
    if (prev.backgroundMode != next.backgroundMode) return change("BG_MODE");
    if (!vecNearlyEqual(prev.backgroundColor, next.backgroundColor))        return change("BG_COLOR");
    if (prev.environmentMapPath != next.environmentMapPath)                 return change("ENV_MAP");
    if (!nearlyEqual(prev.environmentRotation,  next.environmentRotation))  return change("ENV_ROT");
    if (!nearlyEqual(prev.environmentIntensity, next.environmentIntensity)) return change("ENV_INTENSITY");

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
    if (!nearlyEqual(prev.debugNormalLodBias, next.debugNormalLodBias)) return change("DEBUG_NORMAL_BIAS");
    if (!nearlyEqual(prev.debugOrmLodBias, next.debugOrmLodBias)) return change("DEBUG_ORM_BIAS");
    if (!nearlyEqual(prev.debugEnvMipOverride, next.debugEnvMipOverride)) return change("DEBUG_ENV_MIP");
    if (prev.debugEnableVisorOverride != next.debugEnableVisorOverride) return change("DEBUG_VISOR_OVERRIDE");
    if (prev.debugVisorOverrideMaterialId != next.debugVisorOverrideMaterialId) return change("DEBUG_VISOR_MAT");
    if (!nearlyEqual(prev.debugVisorOverrideRoughness, next.debugVisorOverrideRoughness)) return change("DEBUG_VISOR_ROUGH");
    if (!nearlyEqual(prev.debugVisorOverrideF0, next.debugVisorOverrideF0)) return change("DEBUG_VISOR_F0");
    if (prev.debugEnvNearest != next.debugEnvNearest) return change("DEBUG_ENV_NEAREST");

    // --- Sampling/variance controls ---
    if (prev.samplesPerFrame != next.samplesPerFrame) return change("SPP");
    if (!nearlyEqual(prev.renderScale, next.renderScale, 1e-4f)) return change("RENDER_SCALE");

    if (prev.fireflyClampEnabled    != next.fireflyClampEnabled)    return change("FIREFLY_ENABLE");
    if (!nearlyEqual(prev.fireflyClampFactor,          next.fireflyClampFactor))          return change("FIREFLY_FACTOR");
    if (!nearlyEqual(prev.fireflyClampFloor,           next.fireflyClampFloor))           return change("FIREFLY_FLOOR");
    if (!nearlyEqual(prev.throughputClamp,             next.throughputClamp))             return change("THROUGHPUT_CLAMP");
    if (!nearlyEqual(prev.specularTailClampBase,       next.specularTailClampBase))       return change("SPECULAR_BASE");
    if (!nearlyEqual(prev.specularTailClampRoughnessScale, next.specularTailClampRoughnessScale)) return change("SPECULAR_SCALE");
    if (!nearlyEqual(prev.minSpecularPdf,              next.minSpecularPdf))              return change("SPECULAR_PDF");
    if (!nearlyEqual(prev.fireflyClampMaxContribution, next.fireflyClampMaxContribution)) return change("FIREFLY_MAX");

    // --- SSS mode (integrator behavior) ---
    if (prev.sssMode     != next.sssMode)     return change("SSS_MODE");
    if (prev.sssMaxSteps != next.sssMaxSteps) return change("SSS_STEPS");
    if (prev.enableMnee  != next.enableMnee)  return change("MNEE");
    if (prev.enableMneeSecondary != next.enableMneeSecondary) return change("MNEE_SECONDARY");

    return result;
}

bool MarkRadiometricChange(const RenderSettings& prev,
                           const RenderSettings& next) {
    return DetectRadiometricChange(prev, next).changed;
}

} // namespace PathTracer
