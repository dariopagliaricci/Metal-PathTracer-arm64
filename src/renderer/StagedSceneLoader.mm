#import "renderer/StagedSceneLoader.h"

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <pthread.h>

#include "renderer/SceneResources.h"

namespace PathTracer {

namespace {

const char* PhaseName(ScenePhase phase) {
    switch (phase) {
        case ScenePhase::Idle: return "Idle";
        case ScenePhase::ParseSceneDesc: return "ParseSceneDesc";
        case ScenePhase::ImportGeometry: return "ImportGeometry";
        case ScenePhase::BuildCPUMaterialTables: return "BuildCPUMaterialTables";
        case ScenePhase::AwaitingMainThread: return "AwaitingMainThread";
        case ScenePhase::UploadBuffers: return "UploadBuffers";
        case ScenePhase::BuildBLASBatch: return "BuildBLASBatch";
        case ScenePhase::AssembleTLAS: return "AssembleTLAS";
        case ScenePhase::FinalizeRenderState: return "FinalizeRenderState";
        case ScenePhase::Complete: return "Complete";
        case ScenePhase::Failed: return "Failed";
    }
    return "Unknown";
}

bool SceneLoadDiagnosticsEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("PT_SCENE_LOAD_DIAGNOSTICS");
        if (!value || !*value) {
            return false;
        }
        return std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0 &&
               std::strcmp(value, "no") != 0 &&
               std::strcmp(value, "NO") != 0;
    }();
    return enabled;
}

struct ScenePayloadStats {
    uint32_t meshCount = 0u;
    uint64_t vertexCount = 0u;
    uint64_t indexCount = 0u;
    uint64_t triangleCount = 0u;
    uint32_t materialCount = 0u;
};

ScenePayloadStats CollectPayloadStats(const SceneResources& resources) {
    ScenePayloadStats stats{};
    stats.meshCount = static_cast<uint32_t>(resources.meshes().size());
    stats.materialCount = resources.materialCount();
    for (const auto& mesh : resources.meshes()) {
        stats.vertexCount += static_cast<uint64_t>(mesh.vertices.size());
        stats.indexCount += static_cast<uint64_t>(mesh.indices.size());
    }
    stats.triangleCount = stats.indexCount / 3u;
    return stats;
}

}  // namespace

StagedSceneLoader::StagedSceneLoader(SceneManager& sceneManager)
    : m_sceneManager(sceneManager) {
    std::snprintf(m_progress.phaseLabel, sizeof(m_progress.phaseLabel), "%s", "Idle");
}

StagedSceneLoader::~StagedSceneLoader() {
    joinImportThread();
}

void StagedSceneLoader::joinImportThread() {
    if (m_importThread.joinable()) {
        m_importThread.join();
    }
}

void StagedSceneLoader::logPhase(const char* phase, const char* event, long durationMs) const {
    if (durationMs >= 0) {
        std::fprintf(stderr,
                     "[StagedLoader] phase=%s thread=%p %s duration=%ldms\n",
                     phase,
                     reinterpret_cast<void*>(pthread_self()),
                     event,
                     durationMs);
    } else {
        std::fprintf(stderr,
                     "[StagedLoader] phase=%s thread=%p %s\n",
                     phase,
                     reinterpret_cast<void*>(pthread_self()),
                     event);
    }
}

void StagedSceneLoader::setPhase(ScenePhase phase, const char* label, float fractionComplete) {
    std::snprintf(m_progress.phaseLabel, sizeof(m_progress.phaseLabel), "%s", label ? label : "Loading");
    m_progress.fractionComplete.store(fractionComplete);
    m_progress.phase.store(phase);
}

void StagedSceneLoader::fail(const std::string& message) {
    m_errorMessage = message;
    m_phaseFailed.store(true);
    setPhase(ScenePhase::Failed, "Failed", 1.0f);
}

bool StagedSceneLoader::beginLoad(const std::string& scenePath,
                                  SceneResources& targetResources,
                                  const RenderSettings& startingSettings,
                                  std::string* errorMessage) {
    joinImportThread();
    m_targetResources = &targetResources;
    m_cpuScene = std::make_unique<SceneResources>();
    m_loadState = SceneManager::StagedLoadState{};
    m_loadedSettings = startingSettings;
    m_scenePath = scenePath;
    m_errorMessage.clear();
    m_importComplete.store(false, std::memory_order_relaxed);
    m_phaseFailed.store(false, std::memory_order_relaxed);
    m_progress.blasBatchesDone.store(0u);
    m_progress.blasBatchesTotal.store(0u);
    m_uploadStarted = false;
    m_accelBuildStarted = false;
    m_longestMainThreadBlockMs = 0.0;

    using Clock = std::chrono::steady_clock;
    setPhase(ScenePhase::ParseSceneDesc, "Parse scene", 0.05f);
    logPhase("ParseSceneDesc", "begin");
    const auto t0 = Clock::now();

    std::string beginError;
    if (!m_sceneManager.beginStagedLoadSceneFromPath(scenePath,
                                                     *m_cpuScene,
                                                     m_loadedSettings,
                                                     m_loadState,
                                                     &beginError)) {
        fail(beginError);
        if (errorMessage) {
            *errorMessage = m_errorMessage;
        }
        return false;
    }

    const auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
    logPhase("ParseSceneDesc", "end", dur);

    try {
        m_importThread = std::thread(&StagedSceneLoader::importThreadEntry, this);
    } catch (const std::exception& e) {
        fail(std::string("Failed to create import thread: ") + e.what());
        if (errorMessage) {
            *errorMessage = m_errorMessage;
        }
        return false;
    }

    return true;
}

void StagedSceneLoader::importThreadEntry() {
    using Clock = std::chrono::steady_clock;

    logPhase("ImportGeometry", "begin");
    auto t0 = Clock::now();
    setPhase(ScenePhase::ImportGeometry, "Import geometry", 0.45f);

    std::string continueError;
    const auto status =
        m_sceneManager.continueStagedLoadScene(m_loadState,
                                               *m_cpuScene,
                                               m_loadedSettings,
                                               std::numeric_limits<size_t>::max(),
                                               &continueError);
    if (status == SceneManager::StagedLoadStatus::Failed) {
        fail(continueError.empty() ? "Import failed" : continueError);
        return;
    }
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
    logPhase("ImportGeometry", "end", dur);

    logPhase("BuildCPUMaterialTables", "begin");
    t0 = Clock::now();
    setPhase(ScenePhase::BuildCPUMaterialTables, "Build CPU materials", 0.60f);
    m_cpuScene->buildCPUPackedSceneData();
    dur = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
    logPhase("BuildCPUMaterialTables", "end", dur);

    if (SceneLoadDiagnosticsEnabled()) {
        const ScenePayloadStats stats = CollectPayloadStats(*m_cpuScene);
        std::fprintf(stderr,
                     "[SceneDiag] staged importPublish scene='%s' materials=%u meshes=%u vertices=%llu indices=%llu triangles=%llu\n",
                     m_scenePath.c_str(),
                     stats.materialCount,
                     stats.meshCount,
                     static_cast<unsigned long long>(stats.vertexCount),
                     static_cast<unsigned long long>(stats.indexCount),
                     static_cast<unsigned long long>(stats.triangleCount));
    }

    m_importComplete.store(true, std::memory_order_release);
    setPhase(ScenePhase::AwaitingMainThread, "Await main thread", 0.65f);
}

bool StagedSceneLoader::pollMainThreadWork() {
    const ScenePhase phase = m_progress.phase.load();
    if (phase == ScenePhase::AwaitingMainThread &&
        m_importComplete.load(std::memory_order_acquire)) {
        if (SceneLoadDiagnosticsEnabled() && m_cpuScene) {
            const ScenePayloadStats stats = CollectPayloadStats(*m_cpuScene);
            std::fprintf(stderr,
                         "[SceneDiag] staged importObserved scene='%s' materials=%u meshes=%u vertices=%llu indices=%llu triangles=%llu\n",
                         m_scenePath.c_str(),
                         stats.materialCount,
                         stats.meshCount,
                         static_cast<unsigned long long>(stats.vertexCount),
                         static_cast<unsigned long long>(stats.indexCount),
                         static_cast<unsigned long long>(stats.triangleCount));
        }
        setPhase(ScenePhase::UploadBuffers, "Upload buffers", 0.72f);
        return true;
    }

    return phase == ScenePhase::UploadBuffers ||
           phase == ScenePhase::BuildBLASBatch ||
           phase == ScenePhase::AssembleTLAS ||
           phase == ScenePhase::FinalizeRenderState;
}

bool StagedSceneLoader::pumpGPUPhase(double budgetMs) {
    if (budgetMs <= 0.0) {
        return true;
    }

    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const double clampedBudgetMs = std::max(0.05, budgetMs);

    while (true) {
        const ScenePhase phaseBefore = m_progress.phase.load();
        const float fractionBefore = m_progress.fractionComplete.load();
        const uint32_t blasDoneBefore = m_progress.blasBatchesDone.load();
        const uint32_t blasTotalBefore = m_progress.blasBatchesTotal.load();

        if (!pumpGPUPhaseStep()) {
            return false;
        }

        const ScenePhase phaseAfter = m_progress.phase.load();
        const float fractionAfter = m_progress.fractionComplete.load();
        const uint32_t blasDoneAfter = m_progress.blasBatchesDone.load();
        const uint32_t blasTotalAfter = m_progress.blasBatchesTotal.load();

        const bool progressed =
            (phaseAfter != phaseBefore) ||
            (blasDoneAfter != blasDoneBefore) ||
            (blasTotalAfter != blasTotalBefore) ||
            (fractionAfter > (fractionBefore + 1.0e-6f));

        const double elapsedMs =
            static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                                   Clock::now() - start)
                                   .count()) /
            1000.0;

        if (elapsedMs >= clampedBudgetMs) {
            break;
        }

        if (!progressed ||
            phaseAfter == ScenePhase::Idle ||
            phaseAfter == ScenePhase::Complete ||
            phaseAfter == ScenePhase::Failed) {
            break;
        }
    }

    return !hasFailed();
}

bool StagedSceneLoader::pumpGPUPhaseStep() {
    if (!m_targetResources || !m_cpuScene) {
        fail("Staged loader is missing scene resources");
        return false;
    }

    using Clock = std::chrono::steady_clock;
    const ScenePhase phase = m_progress.phase.load();
    const auto phaseStart = Clock::now();

    auto finishMainBlock = [&]() {
        const double blockMs =
            static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                    Clock::now() - phaseStart)
                                    .count());
        m_longestMainThreadBlockMs = std::max(m_longestMainThreadBlockMs, blockMs);
    };

    switch (phase) {
        case ScenePhase::UploadBuffers: {
            logPhase("UploadBuffers", "begin");
            std::string applyError;
            if (!m_uploadStarted) {
                if (SceneLoadDiagnosticsEnabled() && m_cpuScene) {
                    const ScenePayloadStats cpuStats = CollectPayloadStats(*m_cpuScene);
                    const SceneMemoryReport targetBefore = m_targetResources->sceneMemoryReport();
                    std::fprintf(stderr,
                                 "[SceneDiag] staged uploadBegin scene='%s' cpu{materials=%u meshes=%u vertices=%llu indices=%llu triangles=%llu} "
                                 "targetBefore{meshes=%u triangles=%u geomMiB=%.2f texMiB=%.2f}\n",
                                 m_scenePath.c_str(),
                                 cpuStats.materialCount,
                                 cpuStats.meshCount,
                                 static_cast<unsigned long long>(cpuStats.vertexCount),
                                 static_cast<unsigned long long>(cpuStats.indexCount),
                                 static_cast<unsigned long long>(cpuStats.triangleCount),
                                 targetBefore.meshCount,
                                 targetBefore.triangleCount,
                                 static_cast<double>(targetBefore.geometryMemoryBytes) / (1024.0 * 1024.0),
                                 static_cast<double>(targetBefore.textureAllocatedMemoryBytes) / (1024.0 * 1024.0));
                }
                if (!m_targetResources->adoptCPUScene(*m_cpuScene, &applyError)) {
                    fail(applyError.empty() ? "Failed to apply CPU scene" : applyError);
                    finishMainBlock();
                    return false;
                }
                if (!m_targetResources->beginIncrementalUpload()) {
                    fail("Failed to begin GPU buffer upload");
                    finishMainBlock();
                    return false;
                }
                m_uploadStarted = true;
            }

            bool uploadComplete = false;
            if (!m_targetResources->stepIncrementalUpload(&uploadComplete)) {
                fail("Failed to upload GPU buffers");
                finishMainBlock();
                return false;
            }
            if (uploadComplete) {
                const auto dur =
                    std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - phaseStart).count();
                logPhase("UploadBuffers", "end", dur);
                setPhase(ScenePhase::BuildBLASBatch, "Build BLAS", 0.78f);
            } else {
                m_progress.fractionComplete.store(0.70f);
            }
            break;
        }
        case ScenePhase::BuildBLASBatch: {
            if (!m_accelBuildStarted) {
                uint32_t total = 0u;
                if (!m_targetResources->beginAccelerationStructureBuild(&total)) {
                    fail("Failed to begin acceleration build");
                    finishMainBlock();
                    return false;
                }
                m_progress.blasBatchesTotal.store(total);
                m_accelBuildStarted = true;
            }

            uint32_t done = 0u;
            uint32_t total = 0u;
            bool complete = false;
            const auto batchStart = Clock::now();
            if (!m_targetResources->stepAccelerationStructureBuild(&done, &total, &complete)) {
                fail("Failed to build acceleration structures");
                finishMainBlock();
                return false;
            }
            m_progress.blasBatchesDone.store(done);
            m_progress.blasBatchesTotal.store(total);
            if (total > 0u && done > 0u) {
                const auto batchDur =
                    std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - batchStart)
                        .count();
                std::fprintf(stderr,
                             "[StagedLoader] BLAS batch %u/%u duration=%lldms\n",
                             done,
                             total,
                             static_cast<long long>(batchDur));
            }
            if (complete) {
                setPhase(ScenePhase::FinalizeRenderState, "Finalize render state", 0.96f);
            } else if (total > 0u && done >= total) {
                setPhase(ScenePhase::AssembleTLAS, "Assemble TLAS", 0.92f);
            } else {
                const float batchFrac =
                    (total > 0u) ? (static_cast<float>(done) / static_cast<float>(total)) : 1.0f;
                m_progress.fractionComplete.store(0.78f + batchFrac * 0.14f);
            }
            break;
        }
        case ScenePhase::AssembleTLAS: {
            logPhase("AssembleTLAS", "begin");
            uint32_t done = 0u;
            uint32_t total = 0u;
            bool complete = false;
            if (!m_targetResources->stepAccelerationStructureBuild(&done, &total, &complete)) {
                fail("Failed to assemble TLAS");
                finishMainBlock();
                return false;
            }
            const auto dur =
                std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - phaseStart).count();
            logPhase("AssembleTLAS", "end", dur);
            if (complete) {
                setPhase(ScenePhase::FinalizeRenderState, "Finalize render state", 0.96f);
            }
            break;
        }
        case ScenePhase::FinalizeRenderState: {
            logPhase("FinalizeRenderState", "begin");
            joinImportThread();
            const auto dur =
                std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - phaseStart).count();
            logPhase("FinalizeRenderState", "end", dur);
            setPhase(ScenePhase::Complete, "Complete", 1.0f);
            std::fprintf(stderr,
                         "[StagedLoader] longest_main_thread_block=%ldms\n",
                         static_cast<long>(m_longestMainThreadBlockMs));
            break;
        }
        default:
            break;
    }

    finishMainBlock();
    return !hasFailed();
}

bool StagedSceneLoader::isComplete() const {
    return m_progress.phase.load() == ScenePhase::Complete;
}

bool StagedSceneLoader::hasFailed() const {
    return m_progress.phase.load() == ScenePhase::Failed || m_phaseFailed.load();
}

bool StagedSceneLoader::isActive() const {
    const ScenePhase phase = m_progress.phase.load();
    return phase != ScenePhase::Idle &&
           phase != ScenePhase::Complete &&
           phase != ScenePhase::Failed;
}

}  // namespace PathTracer
