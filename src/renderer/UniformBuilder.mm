#import "renderer/UniformBuilder.h"
#import "renderer/Accumulation.h"
#import "renderer/SceneResources.h"

#include <algorithm>
#include <cmath>

namespace PathTracer {

namespace {

constexpr float kPi = 3.14159265358979323846f;

constexpr float DegreesToRadians(float degrees) {
    return degrees * (kPi / 180.0f);
}

simd::float3 SafeNormalize(simd::float3 v, simd::float3 fallback) {
    const float lenSq = simd::dot(v, v);
    if (!(lenSq > 1.0e-12f) || !std::isfinite(lenSq)) {
        return fallback;
    }
    return v / std::sqrt(lenSq);
}

simd::float3 CameraForwardFromYawPitch(float yaw, float pitch) {
    const float cosPitch = std::cos(pitch);
    return simd_make_float3(-cosPitch * std::cos(yaw),
                            -std::sin(pitch),
                            -cosPitch * std::sin(yaw));
}

}  // namespace

PathTracerShaderTypes::PathtraceUniforms UniformBuilder::buildPathtraceUniforms(
    const RenderSettings& settings,
    const Accumulation& accumulation,
    const SceneResources& scene,
    CGSize renderSize) {
    
    PathTracerShaderTypes::PathtraceUniforms uniforms{};
    
    // Render resolution
    uniforms.width = static_cast<uint32_t>(std::max(1.0, renderSize.width));
    uniforms.height = static_cast<uint32_t>(std::max(1.0, renderSize.height));
    uniforms.frameIndex = accumulation.frameIndex();
    uniforms.sampleCount = accumulation.sampleCount();
    uniforms.executionMode = static_cast<uint32_t>(settings.executionMode);
    uniforms.wavefrontSchedulingPolicy =
        static_cast<uint32_t>(settings.wavefrontSchedulingPolicy);
    uniforms.wavefrontCompactionEnabled =
        (settings.wavefrontSchedulingPolicy != RenderSettings::WavefrontSchedulingPolicy::Preview &&
         settings.wavefrontCompactionEnabled) ? 1u : 0u;
    uniforms.wavefrontDebugStageFailures =
        (settings.wavefrontSchedulingPolicy == RenderSettings::WavefrontSchedulingPolicy::Research)
            ? 1u
            : 0u;
    
    // Camera setup
    float aspectRatio = static_cast<float>(uniforms.width) / static_cast<float>(uniforms.height);
    float vfov = std::clamp(settings.cameraVerticalFov, 1.0f, 179.0f);
    float defocusAngle = std::max(settings.cameraDefocusAngle, 0.0f);

    const float theta = DegreesToRadians(vfov);
    const float h = std::tan(theta * 0.5f);
    const float viewportHeight = 2.0f * h;
    const float viewportWidth = aspectRatio * viewportHeight;

    simd::float3 lookFrom{};
    simd::float3 w{};
    simd::float3 u{};
    simd::float3 v{};
    float focusDist = settings.cameraFocusDistance;

    if (settings.explicitCameraEnabled) {
        lookFrom = settings.explicitCameraPosition;
        const simd::float3 forward =
            SafeNormalize(settings.explicitCameraForward, simd_make_float3(0.0f, 0.0f, -1.0f));
        const simd::float3 upHint =
            SafeNormalize(settings.explicitCameraUp, simd_make_float3(0.0f, 1.0f, 0.0f));
        w = -forward;
        u = SafeNormalize(simd::cross(upHint, w), simd_make_float3(1.0f, 0.0f, 0.0f));
        v = SafeNormalize(simd::cross(w, u), simd_make_float3(0.0f, 1.0f, 0.0f));
        if (focusDist <= 0.0f) {
            focusDist = 1.0f;
        }
    } else if (settings.cameraControlMode == RenderSettings::CameraControlMode::FirstPerson) {
        lookFrom = settings.firstPersonCameraPosition;
        const simd::float3 forward =
            SafeNormalize(CameraForwardFromYawPitch(settings.cameraYaw, settings.cameraPitch),
                          simd_make_float3(0.0f, 0.0f, -1.0f));
        const simd::float3 upHint = simd_make_float3(0.0f, 1.0f, 0.0f);
        w = -forward;
        u = SafeNormalize(simd::cross(upHint, w), simd_make_float3(1.0f, 0.0f, 0.0f));
        v = SafeNormalize(simd::cross(w, u), simd_make_float3(0.0f, 1.0f, 0.0f));
        if (focusDist <= 0.0f) {
            focusDist = 1.0f;
        }
    } else {
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

        const simd::float3 lookAt = settings.cameraTarget;
        lookFrom = lookAt + offset;
        const simd::float3 vup = {0.0f, 1.0f, 0.0f};

        w = simd::normalize(lookFrom - lookAt);
        u = simd::normalize(simd::cross(vup, w));
        v = simd::cross(w, u);

        if (focusDist <= 0.0f) {
            focusDist = distance;
        }
    }

    const simd::float3 horizontal = focusDist * viewportWidth * u;
    const simd::float3 vertical = focusDist * viewportHeight * v;
    const simd::float3 lowerLeftCorner =
        lookFrom - 0.5f * horizontal - 0.5f * vertical - focusDist * w;
    const float lensRadius = focusDist * std::tan(DegreesToRadians(defocusAngle * 0.5f));

    uniforms.cameraOrigin = lookFrom;
    uniforms.horizontal = horizontal;
    uniforms.vertical = vertical;
    uniforms.lowerLeftCorner = lowerLeftCorner;
    uniforms.cameraU = u;
    uniforms.lensRadius = lensRadius;
    uniforms.cameraV = v;

    // Scene data
    uniforms.sphereCount = scene.sphereCount();
    uniforms.rectangleCount = scene.rectangleCount();
    uniforms.materialCount = scene.materialCount();
    uniforms.materialTextureCount = scene.materialTextureCount();
    uniforms.maxDepth = settings.maxDepth;
    uniforms.useRussianRoulette = settings.enableRussianRoulette ? 1u : 0u;
    
    // Intersection mode
    uniforms.intersectionMode = static_cast<uint32_t>(scene.intersectionProvider().mode);
    // Decide software BVH type: when in software mode and triangles exist, prefer triangle BVH
    if (scene.intersectionProvider().mode == PathTracerShaderTypes::IntersectionMode::SoftwareBVH) {
        uniforms.softwareBvhType = (scene.triangleCount() > 0)
            ? static_cast<uint32_t>(PathTracerShaderTypes::SoftwareBvhType::Triangles)
            : static_cast<uint32_t>(PathTracerShaderTypes::SoftwareBvhType::Spheres);
    } else {
#if PT_DEBUG_TOOLS
        // HWRT debug tools may rely on SWRT intersections.
        uniforms.softwareBvhType = (scene.triangleCount() > 0)
            ? static_cast<uint32_t>(PathTracerShaderTypes::SoftwareBvhType::Triangles)
            : static_cast<uint32_t>(PathTracerShaderTypes::SoftwareBvhType::None);
#elif PT_MNEE_SWRT_RAYS || PT_MNEE_OCCLUSION_PARITY
        // MNEE diagnostics need software BVH bindings when triangles exist.
        uniforms.softwareBvhType = (scene.triangleCount() > 0)
            ? static_cast<uint32_t>(PathTracerShaderTypes::SoftwareBvhType::Triangles)
            : static_cast<uint32_t>(PathTracerShaderTypes::SoftwareBvhType::None);
#else
        uniforms.softwareBvhType = static_cast<uint32_t>(PathTracerShaderTypes::SoftwareBvhType::None);
#endif
    }
    uniforms.primitiveCount = scene.primitiveCount();
    uniforms.meshCount = static_cast<uint32_t>(scene.meshes().size());
    uniforms.triangleCount = scene.triangleCount();
    uniforms.emissivePrimitiveCount = scene.emissivePrimitiveCount();
    uniforms.fixedRngSeed = settings.fixedRngSeed;
    uniforms.backgroundMode = static_cast<uint32_t>(settings.backgroundMode);
    uniforms.workingColorSpace = static_cast<uint32_t>(settings.workingColorSpace);
    uniforms.environmentRotation = settings.environmentRotation;
    uniforms.environmentIntensity = std::max(settings.environmentIntensity, 0.0f);
    uniforms.emissiveTotalPower = std::max(scene.totalEmittedPower(), 0.0f);
    uniforms.backgroundColor = settings.backgroundColor;
    uniforms.environmentAliasCount = scene.environmentAliasCount();
    uniforms.environmentMapWidth = scene.environmentMapWidth();
    uniforms.environmentMapHeight = scene.environmentMapHeight();
    uniforms.environmentHasDistribution = scene.hasEnvironmentDistribution() ? 1u : 0u;

    // Variance reduction / clamping controls
    uniforms.fireflyClampEnabled = settings.fireflyClampEnabled ? 1u : 0u;
    uniforms.fireflyClampMode = static_cast<uint32_t>(settings.fireflyClampMode);
    uniforms.fireflyClampFactor = std::max(settings.fireflyClampFactor, 0.0f);
    uniforms.fireflyClampFloor = std::max(settings.fireflyClampFloor, 0.0f);
    uniforms.throughputClamp = std::max(settings.throughputClamp, 0.0f);
    uniforms.specularTailClampBase = std::max(settings.specularTailClampBase, 0.0f);
    uniforms.specularTailClampRoughnessScale = std::max(settings.specularTailClampRoughnessScale, 0.0f);
    uniforms.minSpecularPdf = std::max(settings.minSpecularPdf, 0.0f);
    uniforms.fireflyClampMaxContribution = std::max(settings.fireflyClampMaxContribution, 0.0f);

    // Subsurface scattering configuration
    uniforms.sssMode = static_cast<uint32_t>(settings.sssMode);
    uniforms.sssMaxSteps = settings.sssMaxSteps;
    uniforms.directLightMode = static_cast<uint32_t>(settings.directLightMode);
    uniforms.enableSpecularNee = settings.enableSpecularNee ? 1u : 0u;
    uniforms.enableMnee = settings.enableMnee ? 1u : 0u;
    uniforms.enableMneeSecondary = settings.enableMneeSecondary ? 1u : 0u;
    uniforms.risCandidateCount = std::max(settings.risCandidateCount, 1u);
    uniforms.spatialReuseNeighborCount = std::max(settings.spatialReuseNeighborCount, 1u);
    uniforms.worldReuseCellSize = std::max(settings.worldReuseCellSize, 1.0e-4f);
    uniforms.restirGiMode = static_cast<uint32_t>(settings.restirGiMode);
    uniforms.pathGuidingMode = static_cast<uint32_t>(settings.pathGuidingMode);
    uniforms.pathGuidingStrength = std::clamp(settings.pathGuidingStrength, 0.0f, 1.0f);
    uniforms.pathGuidingCellSize = std::max(settings.pathGuidingCellSize, 1.0e-4f);
    uniforms.pathGuidingTrainingInterval = std::max<uint32_t>(settings.pathGuidingTrainingInterval, 1u);
    uniforms.restirPtMode = static_cast<uint32_t>(settings.restirPtMode);
    uniforms.restirPtMaxReservoirs = std::max(settings.restirPtMaxReservoirs, 1u);
    uniforms.restirPtDebug = settings.restirPtDebug ? 1u : 0u;
    uniforms.restirPtReuseStrength = std::clamp(settings.restirPtReuseStrength, 0.0f, 1.0f);
    uniforms.radianceCacheMode = static_cast<uint32_t>(settings.radianceCacheMode);
    uniforms.radianceCacheCellSize = std::max(settings.radianceCacheCellSize, 1.0e-4f);
    uniforms.radianceCacheMinDepth = std::max<uint32_t>(settings.radianceCacheMinDepth, 1u);
    uniforms.radianceCacheMinConfidence =
        std::clamp(settings.radianceCacheMinConfidence, 0.0f, 1.0f);
    uniforms.radianceCacheTrainingInterval =
        std::max<uint32_t>(settings.radianceCacheTrainingInterval, 1u);
    uniforms.radianceCacheMaxRadiance = std::max(settings.radianceCacheMaxRadiance, 1.0e-4f);
    uniforms.radianceCacheDebug = settings.radianceCacheDebug ? 1u : 0u;
    uniforms.radianceCachePadding0 = 0u;
    uniforms.causticTransportMode = static_cast<uint32_t>(settings.causticTransportMode);
    uniforms.causticMaxVertices = std::max<uint32_t>(settings.causticMaxVertices, 1u);
    uniforms.volumeTransportMode = static_cast<uint32_t>(settings.volumeTransportMode);
    uniforms.volumePhaseFunction = static_cast<uint32_t>(settings.volumePhaseFunction);
    const simd::float3 sigmaA = simd_make_float3(std::max(settings.volumeSigmaA.x, 0.0f),
                                                 std::max(settings.volumeSigmaA.y, 0.0f),
                                                 std::max(settings.volumeSigmaA.z, 0.0f));
    const simd::float3 sigmaS = simd_make_float3(std::max(settings.volumeSigmaS.x, 0.0f),
                                                 std::max(settings.volumeSigmaS.y, 0.0f),
                                                 std::max(settings.volumeSigmaS.z, 0.0f));
    const simd::float3 sigmaT = sigmaA + sigmaS;
    const float majorant = std::max(std::max(sigmaT.x, sigmaT.y), sigmaT.z);
    uniforms.volumeSigmaA = simd_make_float4(sigmaA.x, sigmaA.y, sigmaA.z, majorant);
    uniforms.volumeSigmaSAnisotropy =
        simd_make_float4(sigmaS.x, sigmaS.y, sigmaS.z,
                         std::clamp(settings.volumeAnisotropy, -0.98f, 0.98f));
    const simd::float3 volumeEmission =
        simd_make_float3(std::max(settings.volumeEmission.x, 0.0f),
                         std::max(settings.volumeEmission.y, 0.0f),
                         std::max(settings.volumeEmission.z, 0.0f));
    uniforms.volumeEmissionMaxEvents =
        simd_make_float4(volumeEmission.x,
                         volumeEmission.y,
                         volumeEmission.z,
                         static_cast<float>(std::max<uint32_t>(settings.volumeMaxScatteringEvents, 1u)));
    uniforms.volumeNeeEnabled = settings.volumeNeeEnabled ? 1u : 0u;
    uniforms.volumeDebug = settings.volumeDebug ? 1u : 0u;
    uniforms.spectralRenderingMode = static_cast<uint32_t>(settings.spectralRenderingMode);
    uniforms.spectralTexturePolicy = static_cast<uint32_t>(settings.spectralTexturePolicy);
    const float spectralMin =
        std::clamp(std::min(settings.spectralMinWavelengthNm, settings.spectralMaxWavelengthNm - 1.0f),
                   300.0f,
                   829.0f);
    const float spectralMax =
        std::clamp(std::max(settings.spectralMaxWavelengthNm, spectralMin + 1.0f),
                   spectralMin + 1.0f,
                   830.0f);
    uniforms.spectralWavelengthRange =
        simd_make_float4(spectralMin,
                         spectralMax,
                         std::max(settings.spectralDispersionStrength, 0.0f),
                         0.0f);
    uniforms.spectralDebug = settings.spectralDebug ? 1u : 0u;
    uniforms.spectralPadding0 = 0u;
    uniforms.spectralPadding1 = 0u;
    uniforms.spectralPadding2 = 0u;
    uniforms.restirDebugEnabled = settings.restirDebugEnabled ? 1u : 0u;
    uniforms.restirDebugView = static_cast<uint32_t>(settings.restirDebugView);
    uniforms.restirDebugCounters = settings.restirDebugCounters ? 1u : 0u;
    uniforms.restirDebugFrameIndex = uniforms.frameIndex;

    const uint32_t maxHardwareAttempts = 4u;
    uint32_t retries = settings.hardwareExcludeRetries;
    if (retries > maxHardwareAttempts - 1u) {
        retries = maxHardwareAttempts - 1u;
    }
    uniforms.hardwareExcludeMaxAttempts = std::max(1u, retries + 1u);
    uniforms.hardwareExitNormalBias = std::max(settings.hardwareExitNormalBias, 0.0f);
    uniforms.hardwareExitDirectionalBias = std::max(settings.hardwareExitDirectionalBias, 0.0f);
    uniforms.enableHardwareMissFallback = 0u;
    uniforms.enableHardwareFirstHitFromSoftware = 0u;
    uniforms.enableHardwareForceSoftware = 0u;
#if PT_DEBUG_TOOLS
    uniforms.enableHardwareMissFallback = settings.enableHardwareMissFallback ? 1u : 0u;
    uniforms.enableHardwareFirstHitFromSoftware =
        settings.enableHardwareFirstHitFromSoftware ? 1u : 0u;
    uniforms.enableHardwareForceSoftware = settings.enableHardwareForceSoftware ? 1u : 0u;
#endif
    uniforms.paddingHardware1 = 0u;

    uniforms.debugPathActive = 0u;
    uniforms.debugPixelX = 0u;
    uniforms.debugPixelY = 0u;
    uniforms.debugMaxEntries = 0u;
#if PT_DEBUG_TOOLS
    if (settings.enablePathDebug && settings.debugMaxEntries > 0 &&
        uniforms.width > 0u && uniforms.height > 0u) {
        uniforms.debugPathActive = 1u;
        uint32_t clampedX = settings.debugPixelX;
        uint32_t clampedY = settings.debugPixelY;
        if (clampedX >= uniforms.width) {
            clampedX = uniforms.width - 1u;
        }
        if (clampedY >= uniforms.height) {
            clampedY = uniforms.height - 1u;
        }
        uniforms.debugPixelX = clampedX;
        uniforms.debugPixelY = clampedY;
        uint32_t maxEntries =
            std::min(settings.debugMaxEntries, PathTracerShaderTypes::kPathtraceDebugMaxEntries);
        uniforms.debugMaxEntries = std::max<uint32_t>(1u, maxEntries);
    }
#endif

    uniforms.parityAssertEnabled = 0u;
    uniforms.parityAssertMode = 0u;
    uniforms.parityPixelX = 0u;
    uniforms.parityPixelY = 0u;
    uniforms.parityOncePerFrame = 0u;
    uniforms.parityPadding0 = 0u;
    uniforms.explicitRestirDiLightingStage = 0u;
    uniforms.parityPadding2 = 0u;
    uniforms.forcePureHWRTForGlass = 0u;
    uniforms.enableHardwareShadowNearRetryReject =
        settings.enableHardwareShadowNearRetryReject ? 1u : 0u;
    uniforms.parityPadding4 = 0u;
    uniforms.parityPadding5 = 0u;
    uniforms.debugViewMode = 0u;
    uniforms.debugDisableAO = settings.debugDisableAO ? 1u : 0u;
    uniforms.debugAoIndirectOnly = settings.debugAoIndirectOnly ? 1u : 0u;
    uniforms.debugDisableNormalMap = settings.debugDisableNormalMap ? 1u : 0u;
    uniforms.debugDisableOrmTexture = settings.debugDisableOrmTexture ? 1u : 0u;
    uniforms.debugFlipNormalGreen = settings.debugFlipNormalGreen ? 1u : 0u;
    uniforms.debugSpecularOnly = settings.debugSpecularOnly ? 1u : 0u;
    uniforms.debugDirectLightAudit = settings.debugDirectLightAudit ? 1u : 0u;
    uniforms.debugNormalStrengthScale = std::max(settings.debugNormalStrengthScale, 0.0f);
    uniforms.debugNormalLodBias = std::max(settings.debugNormalLodBias, 0.0f);
    uniforms.debugOrmLodBias = std::max(settings.debugOrmLodBias, 0.0f);
    uniforms.debugEnvMipOverride = settings.debugEnvMipOverride;
    uniforms.debugEnableVisorOverride = settings.debugEnableVisorOverride ? 1u : 0u;
    uniforms.debugVisorOverrideMaterialId = settings.debugVisorOverrideMaterialId;
    uniforms.debugVisorOverrideRoughness =
        std::clamp(settings.debugVisorOverrideRoughness, 0.0f, 1.0f);
    uniforms.debugVisorOverrideF0 = std::clamp(settings.debugVisorOverrideF0, 0.0f, 0.12f);
    uniforms.debugEnvNearest = settings.debugEnvNearest ? 1u : 0u;
    uniforms.debugPadding0 = 0u;
#if PT_DEBUG_TOOLS
    uniforms.parityOncePerFrame = settings.parityAssertOncePerFrame ? 1u : 0u;
    uniforms.forcePureHWRTForGlass = settings.forcePureHWRTForGlass ? 1u : 0u;
    uniforms.parityPadding0 = settings.parityTargetDepth;
    const bool parityEnabled =
        settings.parityAssertEnabled &&
        settings.parityAssertMode != RenderSettings::ParityAssertMode::Off;
    if (parityEnabled && uniforms.width > 0u && uniforms.height > 0u) {
        uniforms.parityAssertEnabled = 1u;
        uniforms.parityAssertMode = static_cast<uint32_t>(settings.parityAssertMode);
        uint32_t clampedX = settings.parityPixelX;
        uint32_t clampedY = settings.parityPixelY;
        if (clampedX >= uniforms.width) {
            clampedX = uniforms.width - 1u;
        }
        if (clampedY >= uniforms.height) {
            clampedY = uniforms.height - 1u;
        }
        uniforms.parityPixelX = clampedX;
        uniforms.parityPixelY = clampedY;
    }
#else
    // Preserve prior release behavior: force pure HWRT for glass unless debug tools are enabled.
    uniforms.forcePureHWRTForGlass = 1u;
#endif

    if (settings.debugShowBaseColor) {
        uniforms.debugViewMode = 1u;
    } else if (settings.debugShowMetallic) {
        uniforms.debugViewMode = 2u;
    } else if (settings.debugShowRoughness) {
        uniforms.debugViewMode = 3u;
    } else if (settings.debugShowAO) {
        uniforms.debugViewMode = 4u;
    } else if (settings.restirDebugEnabled &&
               settings.restirDebugView != RenderSettings::RestirDebugView::Beauty) {
        switch (settings.restirDebugView) {
            case RenderSettings::RestirDebugView::CandidateSourceId:
                uniforms.debugViewMode = 10u;
                break;
            case RenderSettings::RestirDebugView::ReservoirConfidence:
                uniforms.debugViewMode = 11u;
                break;
            case RenderSettings::RestirDebugView::ReGIRCellId:
                uniforms.debugViewMode = 12u;
                break;
            case RenderSettings::RestirDebugView::PathGuidingUsedMask:
                uniforms.debugViewMode = 13u;
                break;
            case RenderSettings::RestirDebugView::RestirPTReuseMask:
                uniforms.debugViewMode = 14u;
                break;
            case RenderSettings::RestirDebugView::SVGFVariance:
                uniforms.debugViewMode = 15u;
                break;
            case RenderSettings::RestirDebugView::NaNInfMask:
                uniforms.debugViewMode = 16u;
                break;
            case RenderSettings::RestirDebugView::QueueStageFailure:
                uniforms.debugViewMode = 17u;
                break;
            case RenderSettings::RestirDebugView::RadianceCache:
                uniforms.debugViewMode = 18u;
                break;
            case RenderSettings::RestirDebugView::Beauty:
                break;
        }
    }

    return uniforms;
}

PathTracerShaderTypes::DisplayUniforms UniformBuilder::buildDisplayUniforms(
    const RenderSettings& settings) {
    
    PathTracerShaderTypes::DisplayUniforms uniforms{};
    uniforms.tonemapMode = settings.tonemapMode;
    uniforms.acesVariant = settings.acesVariant;
    uniforms.exposure = settings.exposure;
    uniforms.reinhardWhite = settings.reinhardWhitePoint;
    uniforms.bloomEnabled = settings.bloomEnabled ? 1u : 0u;
    uniforms.bloomThreshold = std::max(settings.bloomThreshold, 0.0f);
    uniforms.bloomIntensity = std::max(settings.bloomIntensity, 0.0f);
    uniforms.bloomRadius = std::max(settings.bloomRadius, 0.0f);
    
    return uniforms;
}

}  // namespace PathTracer
