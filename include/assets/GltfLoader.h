#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <simd/simd.h>

namespace PathTracer {

class SceneResources;

struct GltfCameraInfo {
    bool valid = false;
    bool hasPerspective = false;
    float yfov = 45.0f;   // radians
    float znear = 0.01f;
    float zfar = 0.0f;
    simd::float3 position = {0.0f, 0.0f, 0.0f};
    simd::float3 forward = {0.0f, 0.0f, -1.0f};
    simd::float3 up = {0.0f, 1.0f, 0.0f};
    bool hasSceneBounds = false;
    simd::float3 sceneCenter = {0.0f, 0.0f, 0.0f};
    float sceneRadius = 1.0f;
};

// Explicit scene calibration multipliers; original texture pixels are unchanged.
struct GltfMaterialFactors {
    float roughnessMultiplier = 1.0f;  // [0,1.5], applied before sampled roughness is clamped.
    simd::float3 baseColorMultiplier = {1.0f, 1.0f, 1.0f};  // Linear RGB [0,4]; also tints metallic F0.
    float metallicMultiplier = 1.0f;  // [0,1], multiplied with authored metallic.
};

struct GltfLoadOptions {
    bool enableViewerCompatibilityMode = false;
    bool thinWalledTransmissionFallback = true;
    float emissiveScale = 1.0f;
    bool forceLinearBaseColor = false;  // Treat baseColor as linear (debug/compat)
    bool forceLinearEmissive = false;   // Treat emissive as linear (debug/compat)
    bool flipTexcoordV = false;         // Compatibility path for assets authored against opposite image origin.
    bool enableAnimationSampling = false;
    int animationIndex = 0;
    float animationTimeSeconds = 0.0f;
    bool loopAnimation = false;
    bool importPunctualLights = false;
    bool texturedEmissionNee = false;   // Scene opt-in: textured emitters join direct-light sampling.
    bool referenceDielectricContinuation = false;  // Scene opt-in: eta-free refraction (Path-Tracing references).
    // Scene opt-in (mesh gltfBarcelonaPavilionRules=1, set by the Barcelona Pavilion presets): the rules that match its
    // reference renderer, which the loader chose by the file path before: sampler anisotropy 16, SingleScatterSpecular on
    // every material, and the water and architectural-glass material rules.
    bool barcelonaPavilionRules = false;
    float punctualLightIntensityScale = 1.0f;
    float transmissionScale = 1.0f;
    float roughnessOverride = -1.0f;
    std::unordered_map<std::string, GltfMaterialFactors> materialFactorsByName;  // Exact material names.
    float alphaOverride = -1.0f;
    std::vector<std::string> disableOrmMaterialNameSubstrings;  // Case-insensitive material-name matches.
    float disableOrmRoughnessOverride = -1.0f;  // <0 disables override; otherwise [0,1].
};


/// Load a glTF 2.0 asset (static, core) and append meshes/materials into SceneResources.
bool LoadGltfScene(const std::string& path,
                   SceneResources& resources,
                   std::string& errorMessage,
                   GltfCameraInfo* outCamera = nullptr,
                   const GltfLoadOptions* options = nullptr);


}  // namespace PathTracer
