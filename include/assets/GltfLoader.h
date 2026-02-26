#pragma once

#include <string>
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

struct GltfLoadOptions {
    bool enableViewerCompatibilityMode = false;
    bool thinWalledTransmissionFallback = true;
    float emissiveScale = 1.0f;
    bool forceLinearBaseColor = false;  // Treat baseColor as linear (debug/compat)
    bool forceLinearEmissive = false;   // Treat emissive as linear (debug/compat)
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
