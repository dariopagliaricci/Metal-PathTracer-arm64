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

    // --- SSS mode (integrator behavior) ---
    if (prev.sssMode     != next.sssMode)     return change("SSS_MODE");
    if (prev.sssMaxSteps != next.sssMaxSteps) return change("SSS_STEPS");

    return result;
}

bool MarkRadiometricChange(const RenderSettings& prev,
                           const RenderSettings& next) {
    return DetectRadiometricChange(prev, next).changed;
}

} // namespace PathTracer
