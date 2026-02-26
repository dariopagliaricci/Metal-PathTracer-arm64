#pragma once

#include <cstdint>
#include <istream>
#include <string>
#include <unordered_map>
#include <vector>

#include <simd/simd.h>

#include "MetalShaderTypes.h"

namespace PathTracer {

class SceneResources;
struct RenderSettings;

/// Discovers available scene descriptions and loads them into SceneResources.
class SceneManager {
public:
    struct SceneInfo {
        std::string identifier;                  // File stem, used as unique key
        std::string displayName;                 // Human readable title, fallback to identifier
        std::string filePath;                    // Absolute path to the scene file
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
                              std::unordered_map<std::string, uint32_t>& materialIndicesByName);
    static bool parseSphere(const std::unordered_map<std::string, std::string>& tokens,
                            SceneResources& resources,
                            std::string& errorMessage);
    static bool parseBox(const std::unordered_map<std::string, std::string>& tokens,
                         SceneResources& resources,
                         std::string& errorMessage);
    static bool parseRectangle(const std::unordered_map<std::string, std::string>& tokens,
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
