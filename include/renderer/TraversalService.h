#pragma once

#include <string>

#include "IntersectionProvider.h"
#include "MetalShaderTypes.h"

namespace PathTracer {

enum class TraversalQueryKind : uint32_t {
    PrimaryVisibility = 0,
    ContinuationRay,
    ShadowRay,
    RestirDiVisibility,
    RadianceCacheQuery,
    DebugProbe,
};

struct TraversalServiceRequest {
    TraversalQueryKind kind = TraversalQueryKind::ContinuationRay;
    bool softwareOverride = false;
    bool hardwarePipelineAvailable = false;
    bool hardwareTraversalResourceAvailable = false;
    bool sceneIdentityBuffersAvailable = false;
    bool encoderSupportsHardwareRaytracing = true;
};

struct TraversalServiceDecision {
    PathTracerShaderTypes::IntersectionMode mode =
        PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
    bool usesHardware = false;
    bool fallback = false;
    std::string fallbackReason;
};

inline const char* TraversalQueryKindLabel(TraversalQueryKind kind) {
    switch (kind) {
        case TraversalQueryKind::PrimaryVisibility:
            return "primary_visibility";
        case TraversalQueryKind::ContinuationRay:
            return "continuation_ray";
        case TraversalQueryKind::ShadowRay:
            return "shadow_ray";
        case TraversalQueryKind::RestirDiVisibility:
            return "restir_di_visibility";
        case TraversalQueryKind::RadianceCacheQuery:
            return "radiance_cache_query";
        case TraversalQueryKind::DebugProbe:
            return "debug_probe";
    }
    return "unknown";
}

inline TraversalServiceDecision SelectTraversalBackend(
    const IntersectionProvider& provider,
    const TraversalServiceRequest& request) {
    TraversalServiceDecision decision{};
    const bool sceneHasHardwareProvider =
        provider.mode == PathTracerShaderTypes::IntersectionMode::HardwareRayTracing;

    if (request.softwareOverride) {
        decision.fallback = sceneHasHardwareProvider;
        decision.fallbackReason = "software traversal forced by render settings";
        return decision;
    }
    if (!sceneHasHardwareProvider) {
        decision.fallbackReason = "scene intersection provider is software BVH";
        return decision;
    }
    if (!request.hardwarePipelineAvailable) {
        decision.fallback = true;
        decision.fallbackReason = "hardware traversal pipeline unavailable for query";
        return decision;
    }
    if (!request.hardwareTraversalResourceAvailable) {
        decision.fallback = true;
        decision.fallbackReason = "hardware TLAS unavailable";
        return decision;
    }
    if (!request.sceneIdentityBuffersAvailable) {
        decision.fallback = true;
        decision.fallbackReason = "mesh identity buffers unavailable";
        return decision;
    }
    if (!request.encoderSupportsHardwareRaytracing) {
        decision.fallback = true;
        decision.fallbackReason = "compute encoder cannot bind Metal acceleration structures";
        return decision;
    }

    decision.mode = PathTracerShaderTypes::IntersectionMode::HardwareRayTracing;
    decision.usesHardware = true;
    decision.fallback = false;
    return decision;
}

}  // namespace PathTracer
