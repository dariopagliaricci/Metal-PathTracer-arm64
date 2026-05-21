#pragma once

#include <string>
#include <vector>

namespace PathTracer {

enum class LightCandidateProviderKind {
    AnalyticLights,
    EmissivePrimitives,
    Environment,
    ReGIR,
};

struct LightCandidateProviderContract {
    LightCandidateProviderKind kind = LightCandidateProviderKind::AnalyticLights;
    std::string name;
    std::string featureGate;
    bool producesCandidate = true;
    bool providesPdf = true;
    bool providesConfidence = false;
    bool productionBackend = true;
};

struct GuidingProviderContract {
    std::string name;
    std::string featureGate;
    bool supportsQuery = true;
    bool supportsSample = true;
    bool supportsPdf = true;
    bool supportsUpdateTrain = true;
    bool supportsConfidence = true;
    bool productionBackend = false;
    std::string fallbackReason;
};

struct RadianceProviderContract {
    std::string name;
    std::string featureGate;
    bool supportsCanQuery = true;
    bool supportsQuery = true;
    bool supportsTrain = true;
    bool supportsVariance = true;
    bool supportsCacheability = true;
    bool supportsInvalidation = true;
    bool supportsDebugRecord = true;
    bool supportsConfidence = true;
    bool exposesBiasRisk = true;
    bool productionBackend = false;
    std::string fallbackReason;
};

inline std::vector<LightCandidateProviderContract> DefaultLightCandidateProviderContracts() {
    return {
        {LightCandidateProviderKind::AnalyticLights,
         "AnalyticLights",
         "analytic light count > 0",
         true,
         true,
         false,
         true},
        {LightCandidateProviderKind::EmissivePrimitives,
         "EmissivePrimitives",
         "emissive primitive table non-empty",
         true,
         true,
         false,
         true},
        {LightCandidateProviderKind::Environment,
         "Environment",
         "environment map or sky enabled",
         true,
         true,
         false,
         true},
        {LightCandidateProviderKind::ReGIR,
         "ReGIR",
         "directLightMode=restir_di_regir_hybrid",
         true,
         true,
         true,
         true},
    };
}

inline GuidingProviderContract DefaultGuidingProviderContract() {
    return {"PathGuiding.WorldGrid",
            "pathGuidingMode=prototype",
            true,
            true,
            true,
            true,
            true,
            false,
            "fallback to BSDF/path sampling when confidence is low or the bin is invalid"};
}

inline RadianceProviderContract DefaultRadianceProviderContract() {
    return {"RadianceCache.WorldGrid",
            "radianceCacheMode=prototype",
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            false,
            "fallback to path continuation when cache confidence is low, invalid, biased, or unavailable"};
}

inline RadianceProviderContract NoOpRadianceProviderContract() {
    return {"RadianceCache.NoOp",
            "radianceCacheMode=off",
            true,
            false,
            false,
            false,
            false,
            true,
            true,
            true,
            false,
            false,
            "radiance cache disabled; path continuation remains authoritative"};
}

}  // namespace PathTracer
