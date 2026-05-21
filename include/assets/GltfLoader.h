#pragma once

#include <string>
#include <vector>
#include <simd/simd.h>

namespace PathTracer {

class SceneResources;
struct GltfStagedLoadState;

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
    bool flipTexcoordV = false;         // Compatibility path for assets authored against opposite image origin.
    std::vector<std::string> disableOrmMaterialNameSubstrings;  // Case-insensitive material-name matches.
    float disableOrmRoughnessOverride = -1.0f;  // <0 disables override; otherwise [0,1].
};

enum class GltfStagedLoadStatus : uint32_t {
    InProgress = 0,
    Complete = 1,
    Failed = 2,
};

/// Load a glTF 2.0 asset (static, core) and append meshes/materials into SceneResources.
bool LoadGltfScene(const std::string& path,
                   SceneResources& resources,
                   std::string& errorMessage,
                   GltfCameraInfo* outCamera = nullptr,
                   const GltfLoadOptions* options = nullptr);

GltfStagedLoadState* CreateGltfStagedLoadState();
void DestroyGltfStagedLoadState(GltfStagedLoadState* state);
bool BeginLoadGltfSceneStaged(const std::string& path,
                              GltfStagedLoadState* state,
                              std::string& errorMessage,
                              const GltfLoadOptions* options = nullptr);
GltfStagedLoadStatus ContinueLoadGltfSceneStaged(GltfStagedLoadState* state,
                                                 SceneResources& resources,
                                                 std::string& errorMessage,
                                                 size_t primitiveBudget,
                                                 GltfCameraInfo* outCamera = nullptr);

}  // namespace PathTracer
