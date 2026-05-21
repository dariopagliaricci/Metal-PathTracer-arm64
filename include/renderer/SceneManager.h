#pragma once

#include <cstdint>
#include <istream>
#include <string>
#include <unordered_map>
#include <vector>

#include <simd/simd.h>

#include "MetalShaderTypes.h"
#include "renderer/RenderSettings.h"

namespace PathTracer {

class SceneResources;
struct RenderSettings;

/// Discovers available scene descriptions and loads them into SceneResources.
class SceneManager {
public:
    enum class StagedLoadStatus : uint32_t {
        InProgress = 0,
        Complete = 1,
        Failed = 2,
    };

    struct SceneInfo {
        std::string identifier;                  // File stem, used as unique key
        std::string displayName;                 // Human readable title, fallback to identifier
        std::string filePath;                    // Absolute path to the scene file
    };

    struct StagedLoadState {
        std::vector<std::pair<size_t, std::string>> directives;
        size_t nextDirective = 0;
        bool sawExplicitCameraDirective = false;
        std::unordered_map<std::string, uint32_t> materialIndicesByName;
        RenderSettings parsedSettings{};
        std::string scenePath;
        std::string sceneDirectoryOverride;
    };

    SceneManager();
    explicit SceneManager(std::string scenesDirectory);

    /// Set the directory that contains .scene files and rescan it.
    bool setSceneDirectory(const std::string& directory, std::string* errorMessage = nullptr);
    /// Absolute directory that is currently scanned for scenes.
    const std::string& sceneDirectory() const { return m_sceneDirectory; }

    /// Refresh the list of available scenes from disk.
    bool refresh(std::string* errorMessage = nullptr);
    /// Immutable view of the known scenes.
    const std::vector<SceneInfo>& scenes() const { return m_scenes; }

    /// Load a scene by identifier (file stem), updating resources and render settings.
    bool loadScene(const std::string& identifier,
                   SceneResources& resources,
                   RenderSettings& inOutSettings,
                   std::string* errorMessage = nullptr);

    /// Load a scene directly from the specified path.
    bool loadSceneFromPath(const std::string& path,
                           SceneResources& resources,
                           RenderSettings& inOutSettings,
                           std::string* errorMessage = nullptr);

    bool beginStagedLoadSceneFromPath(const std::string& path,
                                      SceneResources& resources,
                                      RenderSettings& inOutSettings,
                                      StagedLoadState& state,
                                      std::string* errorMessage = nullptr);

    StagedLoadStatus continueStagedLoadScene(StagedLoadState& state,
                                             SceneResources& resources,
                                             RenderSettings& inOutSettings,
                                             size_t maxDirectives,
                                             std::string* errorMessage = nullptr) const;

    /// Information about the most recently loaded scene, if any.
    const SceneInfo* currentScene() const;

private:
    bool discoverScenes(std::string* errorMessage);
    bool parseScene(std::istream& stream,
                    SceneResources& resources,
                    RenderSettings& inOutSettings,
                    std::string& errorMessage) const;
    static bool parseCamera(const std::unordered_map<std::string, std::string>& tokens,
                            RenderSettings& inOutSettings,
                            std::string& errorMessage);
    static bool parseRenderer(const std::unordered_map<std::string, std::string>& tokens,
                              RenderSettings& inOutSettings,
                              std::string& errorMessage);
    static bool parseMaterial(const std::unordered_map<std::string, std::string>& tokens,
                              SceneResources& resources,
                              std::string& errorMessage,
                              std::unordered_map<std::string, uint32_t>& materialIndicesByName,
                              const std::string& sceneDirectory);
    static bool parseSphere(const std::unordered_map<std::string, std::string>& tokens,
                            SceneResources& resources,
                            std::string& errorMessage);
    static bool parseBox(const std::unordered_map<std::string, std::string>& tokens,
                         SceneResources& resources,
                         std::string& errorMessage);
    static bool parseRectangle(const std::unordered_map<std::string, std::string>& tokens,
                               SceneResources& resources,
                               std::string& errorMessage);
    static bool parseDisk(const std::unordered_map<std::string, std::string>& tokens,
                          SceneResources& resources,
                          std::string& errorMessage);
    static bool parseDirectionalLight(const std::unordered_map<std::string, std::string>& tokens,
                                      SceneResources& resources,
                                      std::string& errorMessage);
    static bool parseMesh(const std::unordered_map<std::string, std::string>& tokens,
                          SceneResources& resources,
                          std::string& errorMessage,
                          RenderSettings& inOutSettings,
                          bool allowEmbeddedCameraOverride,
                          const std::string& sceneDirectory,
                          const std::unordered_map<std::string, uint32_t>& materialIndicesByName);
    static bool parseBackground(const std::unordered_map<std::string, std::string>& tokens,
                                RenderSettings& inOutSettings,
                                std::string& errorMessage,
                                const std::string& sceneDirectory);

    static std::unordered_map<std::string, std::string> tokenize(const std::string& line);
    static std::string trim(const std::string& value);
    static bool parseFloat(const std::string& value, float& out);
    static bool parseUInt(const std::string& value, uint32_t& out);
    static bool parseFloat2(const std::string& value, simd::float2& out);
    static bool parseFloat3(const std::string& value, simd::float3& out);
    static bool parseFloatRange(const std::string& value,
                                float& outMin,
                                float& outMax,
                                bool& outIsFixed);
    static bool parseMaterialType(const std::string& value,
                                  PathTracerShaderTypes::MaterialType& out);
    static std::string readDisplayName(const std::string& filePath);

    const SceneInfo* findScene(const std::string& identifier) const;

    std::string m_sceneDirectory;
    std::vector<SceneInfo> m_scenes;
    std::unordered_map<std::string, size_t> m_sceneIndexById;
    std::string m_currentSceneId;
};

}  // namespace PathTracer
