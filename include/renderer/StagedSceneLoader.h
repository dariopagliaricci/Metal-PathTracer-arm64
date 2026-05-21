#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "renderer/RenderSettings.h"
#include "renderer/SceneManager.h"

namespace PathTracer {

class SceneResources;

enum class ScenePhase : uint8_t {
    Idle = 0,
    ParseSceneDesc,
    ImportGeometry,
    BuildCPUMaterialTables,
    AwaitingMainThread,
    UploadBuffers,
    BuildBLASBatch,
    AssembleTLAS,
    FinalizeRenderState,
    Complete,
    Failed,
};

struct SceneLoadProgress {
    std::atomic<ScenePhase> phase{ScenePhase::Idle};
    std::atomic<uint32_t> blasBatchesDone{0};
    std::atomic<uint32_t> blasBatchesTotal{0};
    std::atomic<float> fractionComplete{0.0f};
    char phaseLabel[64]{};
};

class StagedSceneLoader {
public:
    explicit StagedSceneLoader(SceneManager& sceneManager);
    ~StagedSceneLoader();

    bool beginLoad(const std::string& scenePath,
                   SceneResources& targetResources,
                   const RenderSettings& startingSettings,
                   std::string* errorMessage = nullptr);
    bool pollMainThreadWork();
    bool pumpGPUPhase(double budgetMs);

    const SceneLoadProgress& progress() const { return m_progress; }
    const RenderSettings& loadedSettings() const { return m_loadedSettings; }
    const std::string& errorMessage() const { return m_errorMessage; }
    bool isComplete() const;
    bool hasFailed() const;
    bool isActive() const;

private:
    void importThreadEntry();
    void setPhase(ScenePhase phase, const char* label, float fractionComplete);
    void fail(const std::string& message);
    void logPhase(const char* phase, const char* event, long durationMs = -1) const;
    void joinImportThread();
    bool pumpGPUPhaseStep();

    SceneManager& m_sceneManager;
    SceneResources* m_targetResources = nullptr;
    std::unique_ptr<SceneResources> m_cpuScene;
    SceneManager::StagedLoadState m_loadState{};
    RenderSettings m_loadedSettings{};
    std::thread m_importThread;
    std::atomic<bool> m_importComplete{false};
    std::atomic<bool> m_phaseFailed{false};
    SceneLoadProgress m_progress{};
    std::string m_scenePath;
    std::string m_errorMessage;
    bool m_uploadStarted = false;
    bool m_accelBuildStarted = false;
    double m_longestMainThreadBlockMs = 0.0;
};

}  // namespace PathTracer
