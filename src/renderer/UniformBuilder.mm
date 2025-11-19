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
    
    // Camera setup
    float aspectRatio = static_cast<float>(uniforms.width) / static_cast<float>(uniforms.height);
    float vfov = std::clamp(settings.cameraVerticalFov, 1.0f, 179.0f);
    float defocusAngle = std::max(settings.cameraDefocusAngle, 0.0f);

    const float theta = DegreesToRadians(vfov);
    const float h = std::tan(theta * 0.5f);
    const float viewportHeight = 2.0f * h;
    const float viewportWidth = aspectRatio * viewportHeight;

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
    const simd::float3 lookFrom = lookAt + offset;
    const simd::float3 vup = {0.0f, 1.0f, 0.0f};

    const simd::float3 w = simd::normalize(lookFrom - lookAt);
    const simd::float3 u = simd::normalize(simd::cross(vup, w));
    const simd::float3 v = simd::cross(w, u);

    float focusDist = settings.cameraFocusDistance;
    if (focusDist <= 0.0f) {
        focusDist = distance;
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
        uniforms.softwareBvhType = static_cast<uint32_t>(PathTracerShaderTypes::SoftwareBvhType::None);
    }
    uniforms.primitiveCount = scene.primitiveCount();
    uniforms.meshCount = static_cast<uint32_t>(scene.meshes().size());
    uniforms.triangleCount = scene.triangleCount();
    uniforms.fixedRngSeed = settings.fixedRngSeed;
    uniforms.backgroundMode = static_cast<uint32_t>(settings.backgroundMode);
    uniforms.environmentRotation = settings.environmentRotation;
    uniforms.environmentIntensity = std::max(settings.environmentIntensity, 0.0f);
    uniforms.padding0 = 0.0f;
    uniforms.backgroundColor = settings.backgroundColor;
    uniforms.environmentAliasCount = scene.environmentAliasCount();
    uniforms.environmentMapWidth = scene.environmentMapWidth();
    uniforms.environmentMapHeight = scene.environmentMapHeight();
    uniforms.environmentHasDistribution = scene.hasEnvironmentDistribution() ? 1u : 0u;

    // Variance reduction / clamping controls
    uniforms.fireflyClampEnabled = settings.fireflyClampEnabled ? 1u : 0u;
    uniforms.fireflyClampFactor = std::max(settings.fireflyClampFactor, 0.0f);
    uniforms.fireflyClampFloor = std::max(settings.fireflyClampFloor, 0.0f);
    uniforms.throughputClamp = std::max(settings.throughputClamp, 0.0f);
    uniforms.specularTailClampBase = std::max(settings.specularTailClampBase, 0.0f);
    uniforms.specularTailClampRoughnessScale = std::max(settings.specularTailClampRoughnessScale, 0.0f);
    uniforms.minSpecularPdf = std::max(settings.minSpecularPdf, 0.0f);

    // Subsurface scattering configuration
    uniforms.sssMode = static_cast<uint32_t>(settings.sssMode);
    uniforms.sssMaxSteps = settings.sssMaxSteps;
    uniforms.enableSpecularNee = settings.enableSpecularNee ? 1u : 0u;

    uniforms.debugPathActive = 0u;
    uniforms.debugPixelX = 0u;
    uniforms.debugPixelY = 0u;
    uniforms.debugMaxEntries = 0u;
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

    return uniforms;
}

PathTracerShaderTypes::DisplayUniforms UniformBuilder::buildDisplayUniforms(
    const RenderSettings& settings) {
    
    PathTracerShaderTypes::DisplayUniforms uniforms{};
    uniforms.tonemapMode = settings.tonemapMode;
    uniforms.acesVariant = settings.acesVariant;
    uniforms.exposure = settings.exposure;
    uniforms.reinhardWhite = settings.reinhardWhitePoint;
    
    return uniforms;
}

}  // namespace PathTracer
