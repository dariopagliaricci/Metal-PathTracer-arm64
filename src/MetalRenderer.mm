#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <filesystem>

#include <simd/simd.h>

#include "MetalRenderer.h"
#include "MetalShaderTypes.h"
#include "IntersectionProvider.h"
#include "renderer/Accumulation.h"
#include "renderer/ImageWriter.h"
#include "renderer/MetalContext.h"
#include "renderer/Pipelines.h"
#include "renderer/SceneResources.h"
#include "renderer/SceneManager.h"
#include "renderer/MetalHandles.h"
#include "renderer/PerformanceStats.h"
#include "renderer/RenderLoop.h"
#include "renderer/RenderSettings.h"
#include "renderer/SettingsUtils.h"
#include "renderer/UIOverlay.h"

// ImGui is only needed for windowed mode (not headless)
#if __has_include("imgui.h")
#include "imgui.h"
#include "imgui_internal.h"
#include "backends/imgui_impl_metal.h"
#include "backends/imgui_impl_osx.h"
#define HAS_IMGUI 1
#else
#define HAS_IMGUI 0
#endif

namespace {
constexpr float kMinRenderScale = 0.5f;
constexpr float kMaxRenderScale = 2.0f;
constexpr CGFloat kMaxRenderDimension = 8192.0;
constexpr double kMaxRenderPixels = 16.0 * 1024.0 * 1024.0;  // ~16 MP cap for windowed mode
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;
constexpr float kCameraPitchLimitRadians = 89.9f * (kPi / 180.0f);
constexpr float kCameraSmoothingCutoffHz = 12.0f;

inline float ClampRenderScale(float value) {
    return std::clamp(value, kMinRenderScale, kMaxRenderScale);
}

inline float WrapRadians(float radians) {
    if (!std::isfinite(radians)) {
        return 0.0f;
    }
    float wrapped = std::fmod(radians + kPi, kTwoPi);
    if (wrapped < 0.0f) {
        wrapped += kTwoPi;
    }
    return wrapped - kPi;
}

inline float ShortestAngleDelta(float target, float current) {
    return WrapRadians(target - current);
}
}  // namespace

// DEBUG: Render path tracer at fixed internal resolution, independent of drawable size
// Set to 1 to enable fixed 1280x720 internal rendering while keeping UI at full resolution
#ifndef DEBUG_FIXED_RENDER_RESOLUTION
#define DEBUG_FIXED_RENDER_RESOLUTION 0
#endif

#if DEBUG_FIXED_RENDER_RESOLUTION
constexpr uint32_t kDebugRenderWidth = 1280;
constexpr uint32_t kDebugRenderHeight = 720;
#endif

using PathTracerShaderTypes::MaterialType;
using PathTracerShaderTypes::kMaxMaterials;
using PathTracerShaderTypes::kMaxSpheres;
using PathTracerShaderTypes::PathtraceStats;
using PathTracerShaderTypes::PathtraceDebugBuffer;
using PathTracer::RenderSettings;
using PathTracer::ScenePanelProvider;
using PathTracer::MaterialDisplayInfo;
using PathTracer::ObjectDisplayInfo;
using PathTracer::FrameResult;
using PathTracer::SceneManager;
using PathTracer::EnvGpuHandles;

class MetalRenderer::Impl : public std::enable_shared_from_this<Impl> {
public:
    bool initialize(NSWindow* window, const MetalRendererOptions& options);
    void drawFrame();
    void resize(int width, int height);
    void resetAccumulation();
    void setTonemapMode(uint32_t mode);
    void shutdown();
    bool exportToPPM(const char* filepath);
    bool loadScene(const std::string& identifier);
    bool loadSceneFromPath(const std::string& path);
    std::vector<std::string> sceneIdentifiers() const;
    RenderSettings currentSettings() const { return m_settings; }
    void applySettings(const RenderSettings& settings, bool resetAccumulation);
    void setSamplesPerFrame(uint32_t samples) { m_settings.samplesPerFrame = std::max<uint32_t>(1u, samples); }
    uint32_t sampleCount() const { return m_accumulation.sampleCount(); }
    bool captureAverageImage(std::vector<float>& outLinearRGB,
                             uint32_t& width,
                             uint32_t& height,
                             std::vector<float>* outSampleCounts);
    ~Impl();

private:
    CGSize targetRenderSize();
    bool initializeOverlay();
    void updatePerformanceStats(MTLCommandBufferHandle commandBuffer,
                                MTLBufferHandle statsBuffer,
                                MTLBufferHandle debugBuffer,
                                CGSize renderSize,
                                uint32_t finalSampleCount);
    void handleCameraInput();
    void setupDefaultScene();
    void buildProceduralScene();
    uint32_t addMaterial(const simd::float3& baseColor,
                         float roughness,
                         MaterialType type,
                         float indexOfRefraction,
                         const simd::float3& conductorEta = simd_make_float3(0.0f, 0.0f, 0.0f),
                         const simd::float3& conductorK = simd_make_float3(0.0f, 0.0f, 0.0f),
                         bool hasConductorParameters = false);
    void addSphere(const simd::float3& center, float radius, uint32_t materialIndex);
    void handleSaveExrRequest(const std::string& path);
    void updateDrawableSize();
    void noteWindowInteraction();
    bool isMouseInsideRenderView() const;
    void applyBackgroundSettings();
    bool syncEnvironmentMap(bool* outChanged = nullptr);
    void syncCameraSmoothingToTarget();
    void updateCameraSmoothing(float deltaSeconds);
    void evaluateAccumulationState();
    void serviceAccumulationReset();
    void syncRadiometricBaseline();
    MetalRendererOptions m_options{};

    PathTracer::MetalContext m_context{};
    PathTracer::Pipelines m_pipelines{};
    PathTracer::Accumulation m_accumulation{};
    PathTracer::SceneResources m_sceneResources{};
    PathTracer::SceneManager m_sceneManager{};
    PathTracer::RenderLoop m_renderLoop{};
    PathTracer::RenderSettings m_settings{};
    PathTracer::UIOverlay m_overlay{};
    PathTracer::PerformanceStats m_performanceStats{};
    uint32_t m_lastDebugLogCount = 0;
    std::string m_activeSceneId{};
    std::vector<std::string> m_sceneLabels{};
    std::vector<const char*> m_sceneLabelPointers{};
    std::vector<std::string> m_sceneIdentifiers{};

    bool m_overlayInitialized = false;
    bool m_shutdown = false;

    float m_contentScale = 1.0f;
    id m_backingChangeObserver = nil;
    id m_windowMoveObserver = nil;
    id m_windowWillMoveObserver = nil;
    double m_cameraInputSuppressionDeadline = 0.0;
    bool m_cameraOrbitActive = false;
    bool m_cameraZoomActive = false;
    bool m_cameraMotionActive = false;
    float m_smoothYaw = 0.0f;
    float m_smoothPitch = 0.0f;
    bool m_cameraSmoothingInitialized = false;
    CFAbsoluteTime m_lastCameraInteractionTime = 0.0;

    CFAbsoluteTime m_lastFrameTimestamp = 0.0;
    double m_lastFrameTimeMs = 0.0;
    uint32_t m_previousSampleCount = 0;
    uint32_t m_activeSamplesPerFrame = 0;
    double m_lastDrawableWaitMs = 0.0;
    double m_lastCpuEncodeMs = 0.0;
    float m_currentRenderScale = 1.0f;
    bool m_hasRadiometricBaseline = false;
    bool m_accumDirty = false;
    std::string m_accumDirtyReason;
    RenderSettings m_radiometricBaseline{};
    CGSize m_viewSizePoints = CGSizeMake(1.0, 1.0);
    CGSize m_drawablePixelSize = CGSizeMake(1.0, 1.0);
    bool m_renderClampWarningLogged = false;
    bool m_renderPixelClampLogged = false;
};

bool MetalRenderer::Impl::initialize(NSWindow* window, const MetalRendererOptions& options) {
    m_options = options;
    m_settings.fixedRngSeed = options.fixedRngSeed;
    m_settings.enableSoftwareRayTracing = options.enableSoftwareRayTracing;

    if (!options.headless && !window) {
        return false;
    }

    if (!m_context.initialize(window, options.headless)) {
        return false;
    }

    m_accumulation.initialize(m_context.device());
    m_sceneResources.initialize(m_context);
    m_sceneResources.setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);

    if (!m_options.headless) {
        Impl* selfPtr = this;
        m_backingChangeObserver = [[NSNotificationCenter defaultCenter]
            addObserverForName:NSWindowDidChangeBackingPropertiesNotification
                        object:m_context.window()
                         queue:[NSOperationQueue mainQueue]
                    usingBlock:^(__unused NSNotification* note) {
                        if (selfPtr) {
                            selfPtr->updateDrawableSize();
                        }
                    }];
        m_context.setBackingChangeObserver(m_backingChangeObserver);

        m_windowMoveObserver = [[NSNotificationCenter defaultCenter]
            addObserverForName:NSWindowDidMoveNotification
                        object:m_context.window()
                         queue:[NSOperationQueue mainQueue]
                    usingBlock:^(__unused NSNotification* note) {
                        if (selfPtr) {
                            selfPtr->noteWindowInteraction();
                        }
                    }];

        m_windowWillMoveObserver = [[NSNotificationCenter defaultCenter]
            addObserverForName:NSWindowWillMoveNotification
                        object:m_context.window()
                         queue:[NSOperationQueue mainQueue]
                    usingBlock:^(__unused NSNotification* note) {
                        if (selfPtr) {
                            selfPtr->noteWindowInteraction();
                        }
                    }];
    }

    updateDrawableSize();

    MTLPixelFormat displayFormat = MTLPixelFormatBGRA8Unorm;
    if (!m_options.headless) {
        MTKView* view = m_context.view();
        if (!view) {
            return false;
        }
        displayFormat = view.colorPixelFormat;
    }

    MTLDeviceHandle device = m_context.device();
    if (!device) {
        return false;
    }

    if (!m_renderLoop.initialize(m_context)) {
        return false;
    }

    if (!m_pipelines.initialize(m_context, displayFormat)) {
        return false;
    }

#if HAS_IMGUI
    if (!m_options.headless) {
        m_overlayInitialized = initializeOverlay();
        if (!m_overlayInitialized) {
            return false;
        }
    }
#endif

    setupDefaultScene();
    resetAccumulation();
    return true;
}

void MetalRenderer::Impl::resize(int width, int height) {
    if (m_options.headless) {
        return;
    }

    MTKView* view = m_context.view();
    if (!view) {
        return;
    }

    NSRect frame = view.frame;
    frame.size.width = std::max(width, 1);
    frame.size.height = std::max(height, 1);
    [view setFrame:frame];
    updateDrawableSize();
}

void MetalRenderer::Impl::updateDrawableSize() {
    if (m_options.headless) {
        return;
    }

    m_context.updateDrawableSize();
    m_contentScale = m_context.contentScale();

    MTKView* view = m_context.view();
    if (!view) {
        return;
    }

    m_viewSizePoints = view.bounds.size;
    CGSize drawable = m_context.drawableSize();
    if (drawable.width >= 1.0f && drawable.height >= 1.0f) {
        m_drawablePixelSize = drawable;
    }

#if HAS_IMGUI
    m_overlay.updateDisplayMetrics(m_viewSizePoints, m_contentScale);
#endif
}

void MetalRenderer::Impl::noteWindowInteraction() {
    constexpr double kCameraSuppressionSeconds = 0.35;
    const double deadline = CFAbsoluteTimeGetCurrent() + kCameraSuppressionSeconds;
    m_cameraInputSuppressionDeadline = std::max(m_cameraInputSuppressionDeadline, deadline);
    m_cameraOrbitActive = false;
    m_cameraZoomActive = false;
}

bool MetalRenderer::Impl::isMouseInsideRenderView() const {
    if (m_options.headless) {
        return false;
    }
    NSWindow* window = m_context.window();
    MTKView* view = m_context.view();
    if (!window || !view) {
        return false;
    }

    NSRect viewBounds = [view bounds];
    NSRect viewRectInWindow = [view convertRect:viewBounds toView:nil];
    NSRect viewRectOnScreen = [window convertRectToScreen:viewRectInWindow];
    NSPoint screenLocation = [NSEvent mouseLocation];
    return NSPointInRect(screenLocation, viewRectOnScreen);
}

void MetalRenderer::Impl::drawFrame() {
    id<MTLCommandQueue> commandQueue = m_context.commandQueue();
    if (!commandQueue) {
        return;
    }

    if (!m_pipelines.integrate() || !m_pipelines.present()) {
        return;
    }

    // Render pipeline only needed for windowed mode
    if (!m_options.headless) {
        MTKView* view = m_context.view();
        if (!view || !m_pipelines.display()) {
            return;
        }
    }

    CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
    if (m_lastFrameTimestamp > 0.0) {
        m_lastFrameTimeMs = (now - m_lastFrameTimestamp) * 1000.0;
    }
    m_lastFrameTimestamp = now;

    MTLRenderPassDescriptor* renderPassDescriptor = nil;
    id<CAMetalDrawable> drawable = nil;
    CGSize renderSize = CGSizeZero;

    if (!m_options.headless) {
        MTKView* view = m_context.view();
        if (!view) {
            return;
        }

        CFAbsoluteTime drawableWaitStart = CFAbsoluteTimeGetCurrent();
        [view setNeedsDisplay:YES];
        [view displayIfNeeded];

        renderPassDescriptor = view.currentRenderPassDescriptor;
        drawable = view.currentDrawable;
        CFAbsoluteTime drawableWaitEnd = CFAbsoluteTimeGetCurrent();
        m_lastDrawableWaitMs = (drawableWaitEnd - drawableWaitStart) * 1000.0;
        if (!renderPassDescriptor || !drawable) {
            return;
        }

#if HAS_IMGUI
        if (!m_overlayInitialized) {
            m_overlayInitialized = initializeOverlay();
        }
        if (m_overlayInitialized) {
            m_overlay.beginFrame(renderPassDescriptor);

            bool uiRequestedReset = false;

            m_sceneLabels.clear();
            m_sceneLabelPointers.clear();
            m_sceneIdentifiers.clear();
            int currentSceneIndex = 0;

            if (m_sceneManager.refresh(nullptr)) {
                const auto& scenes = m_sceneManager.scenes();
                m_sceneLabels.reserve(scenes.size() + 1);
                m_sceneIdentifiers.reserve(scenes.size() + 1);
                m_sceneLabelPointers.reserve(scenes.size() + 1);

                m_sceneLabels.emplace_back("Procedural (Randomized RTOW)");
                m_sceneLabelPointers.push_back(m_sceneLabels.back().c_str());
                m_sceneIdentifiers.emplace_back("");

                currentSceneIndex = 0;
                for (size_t i = 0; i < scenes.size(); ++i) {
                    const auto& info = scenes[i];
                    std::string label;
                    if (!info.displayName.empty()) {
                        label = info.displayName + " [" + info.identifier + "]";
                    } else {
                        label = info.identifier;
                    }
                    m_sceneLabels.push_back(std::move(label));
                    m_sceneLabelPointers.push_back(m_sceneLabels.back().c_str());
                    m_sceneIdentifiers.push_back(info.identifier);
                    if (!m_activeSceneId.empty() && info.identifier == m_activeSceneId) {
                        currentSceneIndex = static_cast<int>(m_sceneIdentifiers.size() - 1);
                    }
                }
            } else {
                m_sceneLabels.emplace_back("Procedural (Randomized RTOW)");
                m_sceneLabelPointers.push_back(m_sceneLabels.back().c_str());
                m_sceneIdentifiers.emplace_back("");
            }

            ScenePanelProvider panelProvider{};
            panelProvider.materialCount = [this]() -> uint32_t {
                return m_sceneResources.materialCount();
            };
            panelProvider.readMaterial = [this](uint32_t index, MaterialDisplayInfo& info) -> bool {
                if (index >= m_sceneResources.materialCount()) {
                    return false;
                }
                info.name = m_sceneResources.materialName(index);
                const PathTracerShaderTypes::MaterialData* materials = m_sceneResources.materialsData();
                if (!materials) {
                    return false;
                }
                info.data = materials[index];
                return true;
            };
            panelProvider.applyMaterial = [this](uint32_t index, const MaterialDisplayInfo& info) {
                if (m_sceneResources.updateMaterial(index, info.data)) {
                    m_accumDirty = true;
                    m_accumDirtyReason = "MATERIAL_EDIT";
                }
            };
            panelProvider.resetMaterial = [this](uint32_t index) {
                if (m_sceneResources.resetMaterial(index)) {
                    m_accumDirty = true;
                    m_accumDirtyReason = "MATERIAL_RESET";
                }
            };

            panelProvider.objectCount = [this]() -> uint32_t {
                return static_cast<uint32_t>(m_sceneResources.meshes().size());
            };
            panelProvider.readObject = [this](uint32_t index, ObjectDisplayInfo& info) -> bool {
                const auto& meshes = m_sceneResources.meshes();
                if (index >= meshes.size()) {
                    return false;
                }
                const auto& mesh = meshes[index];
                info.name = mesh.name;
                info.meshIndex = index;
                info.materialIndex = mesh.materialIndex;
                info.transform = mesh.localToWorld;
                return true;
            };
            panelProvider.applyObjectTransform = [this](uint32_t meshIndex, const ObjectDisplayInfo& info) {
                if (m_sceneResources.setMeshTransform(meshIndex, info.transform)) {
                    m_accumDirty = true;
                    m_accumDirtyReason = "OBJECT_TRANSFORM";
                }
            };
            panelProvider.resetObjectTransform = [this](uint32_t meshIndex) {
                if (m_sceneResources.resetMeshTransform(meshIndex)) {
                    m_accumDirty = true;
                    m_accumDirtyReason = "OBJECT_RESET";
                }
            };
            m_overlay.setScenePanelProvider(std::move(panelProvider));

            int selectedSceneIndex = currentSceneIndex;
            bool previousSoftwareOverride = m_settings.enableSoftwareRayTracing;
            m_overlay.buildUI(m_performanceStats,
                              m_settings,
                              uiRequestedReset,
                              m_sceneLabelPointers,
                              selectedSceneIndex);

            if (selectedSceneIndex != currentSceneIndex) {
                if (selectedSceneIndex <= 0 ||
                    selectedSceneIndex >= static_cast<int>(m_sceneIdentifiers.size())) {
                    // Procedural scene should always start from the canonical camera pose.
                    const RenderSettings cameraDefaults{};
                    m_settings.cameraTarget = cameraDefaults.cameraTarget;
                    m_settings.cameraDistance = cameraDefaults.cameraDistance;
                    m_settings.cameraYaw = cameraDefaults.cameraYaw;
                    m_settings.cameraPitch = cameraDefaults.cameraPitch;
                    m_settings.cameraVerticalFov = cameraDefaults.cameraVerticalFov;
                    m_settings.cameraDefocusAngle = cameraDefaults.cameraDefocusAngle;
                    m_settings.cameraFocusDistance = cameraDefaults.cameraFocusDistance;
                    buildProceduralScene();
                    resetAccumulation();
                } else {
                    const size_t sceneIdx = static_cast<size_t>(selectedSceneIndex);
                    const std::string& chosenId = m_sceneIdentifiers[sceneIdx];
                    if (!loadScene(chosenId)) {
                        NSLog(@"Failed to switch to scene '%s'", chosenId.c_str());
                    }
                }
            } else if (uiRequestedReset) {
                resetAccumulation();
            }

            if (m_settings.enableSoftwareRayTracing != previousSoftwareOverride) {
                m_sceneResources.setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);
                resetAccumulation();
            }

            std::string pendingSavePath;
            if (m_overlay.consumeSaveExrRequest(pendingSavePath)) {
                handleSaveExrRequest(pendingSavePath);
            }
        }
#endif

        handleCameraInput();

        const CFAbsoluteTime motionNow = CFAbsoluteTimeGetCurrent();
        bool recentMotion = false;
        if (m_lastCameraInteractionTime > 0.0) {
            constexpr double kMotionHoldSeconds = 0.25;
            recentMotion = (motionNow - m_lastCameraInteractionTime) < kMotionHoldSeconds;
        }
        m_cameraMotionActive = recentMotion;
    }
    else {
        m_lastDrawableWaitMs = 0.0;
        m_cameraMotionActive = false;
    }

    bool environmentChanged = false;
    if (!syncEnvironmentMap(&environmentChanged)) {
        if (!m_settings.environmentMapPath.empty()) {
            NSLog(@"Failed to reload environment map: %s", m_settings.environmentMapPath.c_str());
        }
    } else if (environmentChanged) {
        resetAccumulation();
    }

    evaluateAccumulationState();
    serviceAccumulationReset();

    renderSize = targetRenderSize();
    m_accumulation.ensureTextures(renderSize);

    m_performanceStats.renderWidth = static_cast<uint32_t>(std::max<CGFloat>(renderSize.width, 0.0f));
    m_performanceStats.renderHeight = static_cast<uint32_t>(std::max<CGFloat>(renderSize.height, 0.0f));
    m_performanceStats.frameTimeMs = m_lastFrameTimeMs;

    const float deltaSeconds =
        (m_lastFrameTimeMs > 0.0) ? static_cast<float>(m_lastFrameTimeMs * 0.001) : (1.0f / 60.0f);
    updateCameraSmoothing(deltaSeconds);

    RenderSettings frameSettings = m_settings;
    frameSettings.cameraYaw = m_smoothYaw;
    frameSettings.cameraPitch = m_smoothPitch;
    if (m_cameraMotionActive) {
        frameSettings.samplesPerFrame = 1u;
    }
    frameSettings.samplesPerFrame = std::max<uint32_t>(1u, frameSettings.samplesPerFrame);

    CFAbsoluteTime encodeCpuStart = CFAbsoluteTimeGetCurrent();
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    commandBuffer.label = @"Pathtracer Frame";

    FrameResult frameResult = m_renderLoop.encodeFrame(
        commandBuffer,
        renderPassDescriptor,
        drawable,
        m_context,
        m_accumulation,
        m_pipelines,
        m_sceneResources,
        m_overlay,
        frameSettings);

    m_lastCpuEncodeMs = (CFAbsoluteTimeGetCurrent() - encodeCpuStart) * 1000.0;
    if (frameResult.samplesDispatched > 0) {
        m_activeSamplesPerFrame = frameResult.samplesDispatched;
    } else {
        m_activeSamplesPerFrame = frameSettings.samplesPerFrame;
    }

    MTLBufferHandle statsBuffer = m_renderLoop.statsBuffer();
    MTLBufferHandle debugBuffer = m_renderLoop.debugBuffer();
    const uint32_t finalSampleCount = m_accumulation.sampleCount();

    std::weak_ptr<Impl> weakSelf = shared_from_this();
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        if (auto strongSelf = weakSelf.lock()) {
            strongSelf->updatePerformanceStats(cb,
                                               statsBuffer,
                                               debugBuffer,
                                               renderSize,
                                               finalSampleCount);
        }
    }];

    if (!m_options.headless && drawable) {
        [commandBuffer presentDrawable:drawable];
    }

    [commandBuffer commit];

    if (m_options.headless) {
        [commandBuffer waitUntilCompleted];
    }
}

CGSize MetalRenderer::Impl::targetRenderSize() {
    const float requestedScale = ClampRenderScale(m_settings.renderScale);
    float appliedScale = requestedScale;

    auto clampDimension = [](uint32_t overrideValue, CGFloat fallback) -> CGFloat {
        if (overrideValue >= 8u) {
            return static_cast<CGFloat>(overrideValue);
        }
        return std::max<CGFloat>(fallback, 1.0f);
    };

    CGFloat fallbackWidth = 1.0f;
    CGFloat fallbackHeight = 1.0f;

    if (m_options.headless) {
        fallbackWidth = std::max<CGFloat>(m_options.width, 8);
        fallbackHeight = std::max<CGFloat>(m_options.height, 8);
    }
#if !DEBUG_FIXED_RENDER_RESOLUTION
    else {
        if (m_drawablePixelSize.width >= 1.0f && m_drawablePixelSize.height >= 1.0f) {
            fallbackWidth = m_drawablePixelSize.width;
            fallbackHeight = m_drawablePixelSize.height;
        } else if (MTKView* view = m_context.view()) {
            CGSize drawable = view.drawableSize;
            fallbackWidth = std::max<CGFloat>(drawable.width, 1.0f);
            fallbackHeight = std::max<CGFloat>(drawable.height, 1.0f);
        }
    }
#endif

    CGFloat baseWidth = clampDimension(m_settings.renderWidth, fallbackWidth);
    CGFloat baseHeight = clampDimension(m_settings.renderHeight, fallbackHeight);

#if DEBUG_FIXED_RENDER_RESOLUTION
    baseWidth = clampDimension(static_cast<uint32_t>(kDebugRenderWidth), baseWidth);
    baseHeight = clampDimension(static_cast<uint32_t>(kDebugRenderHeight), baseHeight);
#endif

    baseWidth = std::max<CGFloat>(baseWidth, 1.0f);
    baseHeight = std::max<CGFloat>(baseHeight, 1.0f);

    bool pixelClamped = false;
    if (!m_options.headless) {
        const double requestedPixels =
            static_cast<double>(baseWidth) * static_cast<double>(baseHeight) *
            static_cast<double>(appliedScale) * static_cast<double>(appliedScale);
        if (requestedPixels > kMaxRenderPixels && kMaxRenderPixels > 0.0) {
            const double reduction = std::sqrt(kMaxRenderPixels / requestedPixels);
            appliedScale *= static_cast<float>(reduction);
            pixelClamped = true;
        }
    }

    CGFloat scaledWidth = std::max<CGFloat>(static_cast<CGFloat>(std::round(baseWidth * appliedScale)), 1.0f);
    CGFloat scaledHeight = std::max<CGFloat>(static_cast<CGFloat>(std::round(baseHeight * appliedScale)), 1.0f);

    bool clamped = false;
    if (scaledWidth > kMaxRenderDimension) {
        scaledWidth = kMaxRenderDimension;
        clamped = true;
    }
    if (scaledHeight > kMaxRenderDimension) {
        scaledHeight = kMaxRenderDimension;
        clamped = true;
    }

    if (clamped && !m_renderClampWarningLogged) {
        m_renderClampWarningLogged = true;
        NSLog(@"[Renderer] Internal render size clamped to %.0f x %.0f (max %.0f)",
              scaledWidth,
              scaledHeight,
              static_cast<double>(kMaxRenderDimension));
    }

    if (pixelClamped && !m_renderPixelClampLogged) {
        m_renderPixelClampLogged = true;
        const double megaPixels = (static_cast<double>(scaledWidth) * static_cast<double>(scaledHeight)) / 1.0e6;
        NSLog(@"[Renderer] Internal render resolution reduced to %.0f x %.0f (~%.1f MP) for responsiveness",
              scaledWidth,
              scaledHeight,
              megaPixels);
    }

    const float widthScale =
        baseWidth > 0.0f ? static_cast<float>(scaledWidth / baseWidth) : appliedScale;
    const float heightScale =
        baseHeight > 0.0f ? static_cast<float>(scaledHeight / baseHeight) : appliedScale;
    appliedScale = std::max(widthScale, heightScale);

    m_currentRenderScale = appliedScale;
    m_settings.renderScale = requestedScale;
    return CGSizeMake(scaledWidth, scaledHeight);
}

bool MetalRenderer::Impl::initializeOverlay() {
#if HAS_IMGUI
    if (m_options.headless) {
        return false;
    }
    MTKView* view = m_context.view();
    MTLDeviceHandle device = m_context.device();
    if (!view || !device) {
        return false;
    }
    if (!m_overlay.initialize(view, device, m_contentScale)) {
        return false;
    }
    m_overlay.updateDisplayMetrics(m_viewSizePoints, m_contentScale);
    return true;
#else
    return false;
#endif
}

void MetalRenderer::Impl::updatePerformanceStats(MTLCommandBufferHandle commandBuffer,
                                                 MTLBufferHandle statsBuffer,
                                                 MTLBufferHandle debugBuffer,
                                                 CGSize renderSize,
                                                 uint32_t finalSampleCount) {
    if (!commandBuffer) {
        return;
    }

    if (commandBuffer.GPUStartTime != 0 && commandBuffer.GPUEndTime != 0 &&
        commandBuffer.GPUEndTime > commandBuffer.GPUStartTime) {
        m_performanceStats.gpuTimeMs = (commandBuffer.GPUEndTime - commandBuffer.GPUStartTime) * 1000.0;
    } else {
        m_performanceStats.gpuTimeMs = 0.0;
    }

    m_performanceStats.cpuEncodeMs = m_lastCpuEncodeMs;
    m_performanceStats.drawableWaitMs = m_lastDrawableWaitMs;
    m_performanceStats.activeSamplesPerFrame = m_activeSamplesPerFrame;
    m_performanceStats.renderScale =
        static_cast<double>(std::clamp(m_currentRenderScale, kMinRenderScale, kMaxRenderScale));
    m_performanceStats.cameraMotionActive = m_cameraMotionActive;

    if (statsBuffer) {
        const auto* stats = reinterpret_cast<const PathtraceStats*>([statsBuffer contents]);
        if (stats) {
            const double totalRays = static_cast<double>(stats->primaryRayCount);
            const double shadowRays = static_cast<double>(stats->shadowRayCount);
            const double internalVisits = static_cast<double>(stats->internalNodeVisits);
            const double hardwareRays = static_cast<double>(stats->hardwareRayCount);
            const double hardwareHits = static_cast<double>(stats->hardwareHitCount);
            const double hardwareMisses = static_cast<double>(stats->hardwareMissCount);

            m_performanceStats.avgNodesVisited =
                totalRays > 0.0 ? static_cast<double>(stats->nodesVisited) / totalRays : 0.0;
            m_performanceStats.avgLeafPrimTests =
                totalRays > 0.0 ? static_cast<double>(stats->leafPrimTests) / totalRays : 0.0;
            m_performanceStats.shadowRayEarlyExitPct =
                shadowRays > 0.0
                    ? (static_cast<double>(stats->shadowRayEarlyExitCount) / shadowRays) * 100.0
                    : 0.0;
            m_performanceStats.bothChildrenVisitedPct =
                internalVisits > 0.0
                    ? (static_cast<double>(stats->internalBothVisited) / internalVisits) * 100.0
                    : 0.0;
            m_performanceStats.hardwareRayCount = static_cast<uint64_t>(stats->hardwareRayCount);
            m_performanceStats.hardwareHitCount = static_cast<uint64_t>(stats->hardwareHitCount);
            m_performanceStats.hardwareMissCount = static_cast<uint64_t>(stats->hardwareMissCount);
            m_performanceStats.hardwareResultNoneCount =
                static_cast<uint64_t>(stats->hardwareResultNoneCount);
            m_performanceStats.hardwareRejectedCount =
                static_cast<uint64_t>(stats->hardwareRejectedCount);
            m_performanceStats.hardwareUnavailableCount =
                static_cast<uint64_t>(stats->hardwareUnavailableCount);
            m_performanceStats.specularNeeOcclusionHitCount =
                static_cast<uint64_t>(stats->specularNeeOcclusionHitCount);
            m_performanceStats.hardwareSelfHitRejectedCount =
                static_cast<uint64_t>(stats->hardwareSelfHitRejectedCount);
            for (int i = 0; i < 32; ++i) {
                m_performanceStats.hardwareMissDistanceBins[i] =
                    static_cast<uint64_t>(stats->hardwareMissDistanceBins[i]);
            }
            float missDist = 0.0f;
            std::memcpy(&missDist, &stats->hardwareMissLastDistanceBits, sizeof(float));
            m_performanceStats.hardwareMissLastDistance = missDist;
            float selfHitDist = 0.0f;
            std::memcpy(&selfHitDist, &stats->hardwareSelfHitLastDistanceBits, sizeof(float));
            m_performanceStats.hardwareSelfHitLastDistance = selfHitDist;
            m_performanceStats.hardwareLastResultType = stats->hardwareLastResultType;
            m_performanceStats.hardwareLastInstanceId = stats->hardwareLastInstanceId;
            m_performanceStats.hardwareLastPrimitiveId = stats->hardwareLastPrimitiveId;
            float lastDistance = 0.0f;
            static_assert(sizeof(uint32_t) == sizeof(float), "uint32_t must match float");
            std::memcpy(&lastDistance, &stats->hardwareLastDistanceBits, sizeof(float));
            m_performanceStats.hardwareLastDistance = lastDistance;
            if (hardwareRays > 0.0) {
                m_performanceStats.hardwareRayPctHit = (hardwareHits / hardwareRays) * 100.0;
                m_performanceStats.hardwareRayPctMiss = (hardwareMisses / hardwareRays) * 100.0;
            } else {
                m_performanceStats.hardwareRayPctHit = 0.0;
                m_performanceStats.hardwareRayPctMiss = 0.0;
            }
            if (stats->hardwareRayCount > 0 && stats->hardwareHitCount == 0) {
                NSLog(@"Warning: Hardware RT rays=%u resultNone=%u rejected=%u unavailable=%u type=%u dist=%f inst=%u prim=%u",
                      stats->hardwareRayCount,
                      stats->hardwareResultNoneCount,
                      stats->hardwareRejectedCount,
                      stats->hardwareUnavailableCount,
                      stats->hardwareLastResultType,
                      lastDistance,
                      stats->hardwareLastInstanceId,
                      stats->hardwareLastPrimitiveId);
            } else if (stats->hardwareHitCount > 0) {
                static bool loggedHardwareSuccess = false;
                if (!loggedHardwareSuccess) {
                    loggedHardwareSuccess = true;
                    NSLog(@"Hardware RT active: rays=%u hits=%u misses=%u none=%u",
                          stats->hardwareRayCount,
                          stats->hardwareHitCount,
                          stats->hardwareMissCount,
                          stats->hardwareResultNoneCount);
                }
            }
        } else {
            m_performanceStats.avgNodesVisited = 0.0;
            m_performanceStats.avgLeafPrimTests = 0.0;
            m_performanceStats.shadowRayEarlyExitPct = 0.0;
            m_performanceStats.bothChildrenVisitedPct = 0.0;
            m_performanceStats.hardwareRayCount = 0;
            m_performanceStats.hardwareHitCount = 0;
            m_performanceStats.hardwareMissCount = 0;
            m_performanceStats.hardwareResultNoneCount = 0;
            m_performanceStats.hardwareRejectedCount = 0;
            m_performanceStats.hardwareUnavailableCount = 0;
            m_performanceStats.specularNeeOcclusionHitCount = 0;
            m_performanceStats.hardwareSelfHitRejectedCount = 0;
            std::fill(std::begin(m_performanceStats.hardwareMissDistanceBins),
                      std::end(m_performanceStats.hardwareMissDistanceBins),
                      0ull);
            m_performanceStats.hardwareMissLastDistance = 0.0f;
            m_performanceStats.hardwareSelfHitLastDistance = 0.0f;
            m_performanceStats.hardwareLastResultType = 0;
            m_performanceStats.hardwareLastInstanceId = 0;
            m_performanceStats.hardwareLastPrimitiveId = 0;
            m_performanceStats.hardwareLastDistance = 0.0f;
            m_performanceStats.hardwareRayPctHit = 0.0;
            m_performanceStats.hardwareRayPctMiss = 0.0;
        }
    } else {
        m_performanceStats.avgNodesVisited = 0.0;
        m_performanceStats.avgLeafPrimTests = 0.0;
        m_performanceStats.shadowRayEarlyExitPct = 0.0;
        m_performanceStats.bothChildrenVisitedPct = 0.0;
        m_performanceStats.hardwareRayCount = 0;
        m_performanceStats.hardwareHitCount = 0;
        m_performanceStats.hardwareMissCount = 0;
        m_performanceStats.hardwareResultNoneCount = 0;
        m_performanceStats.hardwareRejectedCount = 0;
        m_performanceStats.hardwareUnavailableCount = 0;
        m_performanceStats.specularNeeOcclusionHitCount = 0;
        m_performanceStats.hardwareSelfHitRejectedCount = 0;
        std::fill(std::begin(m_performanceStats.hardwareMissDistanceBins),
                  std::end(m_performanceStats.hardwareMissDistanceBins),
                  0ull);
        m_performanceStats.hardwareMissLastDistance = 0.0f;
        m_performanceStats.hardwareSelfHitLastDistance = 0.0f;
        m_performanceStats.hardwareLastResultType = 0;
        m_performanceStats.hardwareLastInstanceId = 0;
        m_performanceStats.hardwareLastPrimitiveId = 0;
        m_performanceStats.hardwareLastDistance = 0.0f;
        m_performanceStats.hardwareRayPctHit = 0.0;
        m_performanceStats.hardwareRayPctMiss = 0.0;
    }

    uint32_t debugEntryCount = 0;
    uint32_t debugMaxEntries = 0;
    if (debugBuffer) {
        const auto* debugData =
            reinterpret_cast<const PathtraceDebugBuffer*>([debugBuffer contents]);
        if (debugData) {
            debugMaxEntries = debugData->maxEntries;
            uint32_t recorded = debugData->writeIndex;
            debugEntryCount = std::min(recorded, debugMaxEntries);
            if (debugEntryCount > 0 && m_settings.enablePathDebug &&
                debugEntryCount != m_lastDebugLogCount) {
                const uint32_t entriesToLog = std::min(debugEntryCount, 8u);
                for (uint32_t i = 0; i < entriesToLog; ++i) {
                    const auto& entry = debugData->entries[i];
                    const char* integratorLabel = (entry.integrator == 0u) ? "SWRT" : "HWRT";
                    NSLog(@"[PathDebug] %s sample=%u depth=%u medium=%u->%u event=%d front=%u mat=%u throughput=(%.4f %.4f %.4f)",
                          integratorLabel,
                          entry.sampleIndex,
                          entry.depth,
                          entry.mediumDepthBefore,
                          entry.mediumDepthAfter,
                          entry.mediumEvent,
                          entry.frontFace,
                          entry.materialIndex,
                          entry.throughput.x,
                          entry.throughput.y,
                          entry.throughput.z);
                }
                m_lastDebugLogCount = debugEntryCount;
            } else if (debugEntryCount == 0) {
                m_lastDebugLogCount = 0;
            }
        } else {
            m_lastDebugLogCount = 0;
        }
    } else {
        m_lastDebugLogCount = 0;
    }
    m_performanceStats.debugPathEntryCount = debugEntryCount;
    m_performanceStats.debugPathMaxEntries = debugMaxEntries;

    const uint32_t previousSampleCount = m_previousSampleCount;
    m_previousSampleCount = finalSampleCount;
    uint32_t completedSamples = 0;
    if (finalSampleCount >= previousSampleCount) {
        completedSamples = finalSampleCount - previousSampleCount;
    } else {
        completedSamples = finalSampleCount;
    }

    if (m_lastFrameTimeMs > 0.0) {
        const double seconds = m_lastFrameTimeMs / 1000.0;
        m_performanceStats.samplesPerMinute =
            (seconds > 0.0 && completedSamples > 0)
                ? (static_cast<double>(completedSamples) / seconds) * 60.0
                : 0.0;
    } else {
        m_performanceStats.samplesPerMinute = 0.0;
    }

    m_performanceStats.sampleCount = finalSampleCount;
    m_performanceStats.sphereCount = m_sceneResources.sphereCount();
    m_performanceStats.triangleCount = m_sceneResources.triangleCount();

    const auto& provider = m_sceneResources.intersectionProvider();
    m_performanceStats.bvhNodeCount = provider.software.nodeCount;
    m_performanceStats.bvhPrimitiveCount = provider.software.primitiveCount;
    m_performanceStats.intersectionMode = static_cast<uint32_t>(provider.mode);
    m_performanceStats.hardwareRaytracingAvailable = m_sceneResources.supportsRaytracing();
    m_performanceStats.hardwareRaytracingActive =
        provider.mode == PathTracerShaderTypes::IntersectionMode::HardwareRayTracing;

    m_performanceStats.renderWidth =
        static_cast<uint32_t>(std::max<CGFloat>(renderSize.width, 0.0f));
    m_performanceStats.renderHeight =
        static_cast<uint32_t>(std::max<CGFloat>(renderSize.height, 0.0f));
}

void MetalRenderer::Impl::handleCameraInput() {
#if HAS_IMGUI
    if (m_options.headless) {
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        return;
    }

    const CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
    const bool suppressionActive = now < m_cameraInputSuppressionDeadline;
    const bool pointerInsideView = isMouseInsideRenderView();

    if (suppressionActive) {
        m_cameraOrbitActive = false;
        m_cameraZoomActive = false;
        io.MouseDelta = ImVec2(0.0f, 0.0f);
        return;
    }

    if (!pointerInsideView) {
        m_cameraOrbitActive = false;
        m_cameraZoomActive = false;
    }

    if (!pointerInsideView) {
        io.MouseDelta = ImVec2(0.0f, 0.0f);
        return;
    }

    m_cameraInputSuppressionDeadline = 0.0;

    if (!io.MouseDown[0]) {
        m_cameraOrbitActive = false;
    } else if (io.MouseClicked[0]) {
        m_cameraOrbitActive = pointerInsideView;
    }

    if (!io.MouseDown[1]) {
        m_cameraZoomActive = false;
    } else if (io.MouseClicked[1]) {
        m_cameraZoomActive = pointerInsideView;
    }

    if (!m_cameraOrbitActive && !m_cameraZoomActive) {
        io.MouseDelta = ImVec2(0.0f, 0.0f);
        return;
    }

    const float rotateSpeed = 0.005f;
    const float zoomSpeed = 0.01f;
    constexpr float kMinPitch = -kCameraPitchLimitRadians;
    constexpr float kMaxPitch = kCameraPitchLimitRadians;
    constexpr float kMinDistance = 0.5f;
    constexpr float kMaxDistance = 2000.0f;

    bool cameraChanged = false;

    if (m_cameraOrbitActive) {
        const float dx = io.MouseDelta.x;
        const float dy = io.MouseDelta.y;
        if (dx != 0.0f || dy != 0.0f) {
            m_settings.cameraYaw -= dx * rotateSpeed;
            m_settings.cameraPitch -= dy * rotateSpeed;

            m_settings.cameraYaw = WrapRadians(m_settings.cameraYaw);

            m_settings.cameraPitch = std::clamp(m_settings.cameraPitch, kMinPitch, kMaxPitch);
            cameraChanged = true;
        }
    }

    if (m_cameraZoomActive) {
        const float dy = io.MouseDelta.y;
        if (dy != 0.0f) {
            const float factor = std::exp(dy * zoomSpeed);
            m_settings.cameraDistance = std::clamp(m_settings.cameraDistance * factor,
                                                   kMinDistance,
                                                   kMaxDistance);
            cameraChanged = true;
        }
    }

    if (cameraChanged) {
        m_lastCameraInteractionTime = now;
    }
#else
    (void)this;
#endif
}

void MetalRenderer::Impl::syncCameraSmoothingToTarget() {
    m_settings.cameraYaw = WrapRadians(m_settings.cameraYaw);
    m_settings.cameraPitch = std::clamp(m_settings.cameraPitch,
                                        -kCameraPitchLimitRadians,
                                        kCameraPitchLimitRadians);
    m_smoothYaw = m_settings.cameraYaw;
    m_smoothPitch = m_settings.cameraPitch;
    m_cameraSmoothingInitialized = true;
}

void MetalRenderer::Impl::updateCameraSmoothing(float deltaSeconds) {
    if (!m_cameraSmoothingInitialized) {
        syncCameraSmoothingToTarget();
        return;
    }

    if (!std::isfinite(deltaSeconds) || deltaSeconds <= 0.0f) {
        deltaSeconds = 1.0f / 60.0f;
    }
    deltaSeconds = std::clamp(deltaSeconds, 1.0f / 240.0f, 0.25f);

    const float alpha = std::clamp(1.0f - std::exp(-deltaSeconds * kCameraSmoothingCutoffHz),
                                   0.0f,
                                   1.0f);

    float yawTarget = WrapRadians(m_settings.cameraYaw);
    float yawDelta = ShortestAngleDelta(yawTarget, m_smoothYaw);
    m_smoothYaw = WrapRadians(m_smoothYaw + yawDelta * alpha);

    float pitchTarget = std::clamp(m_settings.cameraPitch,
                                   -kCameraPitchLimitRadians,
                                   kCameraPitchLimitRadians);
    m_settings.cameraPitch = pitchTarget;
    m_smoothPitch += (pitchTarget - m_smoothPitch) * alpha;
    m_smoothPitch = std::clamp(m_smoothPitch,
                               -kCameraPitchLimitRadians,
                               kCameraPitchLimitRadians);
}
uint32_t MetalRenderer::Impl::addMaterial(const simd::float3& baseColor,
                                          float roughness,
                                          MaterialType type,
                                          float indexOfRefraction,
                                          const simd::float3& conductorEta,
                                          const simd::float3& conductorK,
                                          bool hasConductorParameters) {
    return m_sceneResources.addMaterial(baseColor,
                                        roughness,
                                        type,
                                        indexOfRefraction,
                                        simd_make_float3(0.0f, 0.0f, 0.0f),
                                        /*emissionUsesEnvironment=*/false,
                                        conductorEta,
                                        conductorK,
                                        hasConductorParameters,
                                        /*coatRoughness=*/0.0f,
                                        /*coatThickness=*/0.0f,
                                        simd_make_float3(1.0f, 1.0f, 1.0f),
                                        simd_make_float3(0.0f, 0.0f, 0.0f),
                                        /*coatIndexOfRefraction=*/1.5f,
                                        simd_make_float3(0.0f, 0.0f, 0.0f),
                                        simd_make_float3(0.0f, 0.0f, 0.0f),
                                        simd_make_float3(0.0f, 0.0f, 0.0f),
                                        /*sssMeanFreePath=*/0.0f,
                                        /*sssAnisotropy=*/0.0f,
                                        /*sssMethod=*/0u,
                                        /*sssCoatEnabled=*/false,
                                        /*sssSigmaOverride=*/false,
                                        /*carpaintBaseMetallic=*/0.0f,
                                        /*carpaintBaseRoughness=*/0.0f,
                                        /*carpaintFlakeSampleWeight=*/0.0f,
                                        /*carpaintFlakeRoughness=*/0.0f,
                                        /*carpaintFlakeAnisotropy=*/0.0f,
                                        /*carpaintFlakeNormalStrength=*/0.0f,
                                        /*carpaintFlakeScale=*/1.0f,
                                        /*carpaintFlakeReflectanceScale=*/1.0f,
                                        simd_make_float3(0.0f, 0.0f, 0.0f),
                                        simd_make_float3(0.0f, 0.0f, 0.0f),
                                        /*carpaintHasBaseConductor=*/false,
                                        simd_make_float3(1.0f, 1.0f, 1.0f));
}

void MetalRenderer::Impl::addSphere(const simd::float3& center, float radius, uint32_t materialIndex) {
    m_sceneResources.addSphere(center, radius, materialIndex);
}

void MetalRenderer::Impl::applyBackgroundSettings() {
    using BackgroundMode = RenderSettings::BackgroundMode;
    switch (m_settings.backgroundMode) {
        case BackgroundMode::Gradient:
            m_sceneResources.clearEnvironmentMap();
            m_settings.environmentMapPath.clear();
            m_settings.environmentMapDirty = false;
            break;
        case BackgroundMode::Solid:
            m_sceneResources.clearEnvironmentMap();
            m_settings.environmentMapPath.clear();
            m_settings.environmentMapDirty = false;
            break;
        case BackgroundMode::Environment: {
            bool envChanged = false;
            if (!syncEnvironmentMap(&envChanged)) {
                if (!m_settings.environmentMapPath.empty()) {
                    NSLog(@"Failed to load environment map: %s", m_settings.environmentMapPath.c_str());
                }
                m_sceneResources.clearEnvironmentMap();
                m_settings.backgroundMode = BackgroundMode::Gradient;
                m_settings.environmentMapPath.clear();
            } else if (envChanged) {
                resetAccumulation();
            }
            break;
        }
    }
}

bool MetalRenderer::Impl::syncEnvironmentMap(bool* outChanged) {
    using BackgroundMode = RenderSettings::BackgroundMode;
    bool changed = false;
    const bool wantsEnvironment =
        (m_settings.backgroundMode == BackgroundMode::Environment) &&
        !m_settings.environmentMapPath.empty();

    if (!wantsEnvironment) {
        const bool hadEnvironment = !m_sceneResources.environmentPath().empty();
        if (hadEnvironment) {
            m_sceneResources.clearEnvironmentMap();
            changed = true;
        }
        if (m_settings.backgroundMode != BackgroundMode::Environment) {
            m_settings.environmentMapPath.clear();
        }
        m_settings.environmentMapDirty = false;
        if (outChanged) {
            *outChanged = changed;
        }
        return true;
    }

    const std::string currentPath = m_sceneResources.environmentPath();
    if (!m_settings.environmentMapDirty &&
        !currentPath.empty() &&
        currentPath == m_settings.environmentMapPath) {
        if (outChanged) {
            *outChanged = false;
        }
        return true;
    }

    m_settings.environmentMapDirty = false;
    EnvGpuHandles handles;
    if (!m_sceneResources.reloadEnvironmentIfNeeded(m_settings.environmentMapPath, &handles)) {
        if (!currentPath.empty()) {
            m_settings.environmentMapPath = currentPath;
        } else {
            m_settings.environmentMapPath.clear();
        }
        return false;
    }

    changed = (currentPath != m_sceneResources.environmentPath());
    if (outChanged) {
        *outChanged = changed;
    }
    return true;
}

void MetalRenderer::Impl::syncRadiometricBaseline() {
    m_radiometricBaseline = m_settings;
    m_hasRadiometricBaseline = true;
}

void MetalRenderer::Impl::evaluateAccumulationState() {
    if (!m_hasRadiometricBaseline) {
        syncRadiometricBaseline();
        return;
    }
    if (m_accumDirty) {
        return;
    }
    const auto change = PathTracer::DetectRadiometricChange(m_radiometricBaseline, m_settings);
    if (change.changed) {
        m_accumDirty = true;
        m_accumDirtyReason = change.reason ? change.reason : "RADIOMETRIC";
    }
}

void MetalRenderer::Impl::serviceAccumulationReset() {
    if (!m_hasRadiometricBaseline) {
        syncRadiometricBaseline();
    }
    if (!m_accumDirty) {
        return;
    }
    const char* reason = m_accumDirtyReason.empty() ? "RADIOMETRIC" : m_accumDirtyReason.c_str();
    NSLog(@"[I] ResetAccum(%s)", reason);
    resetAccumulation();
    m_accumDirty = false;
    m_accumDirtyReason.clear();
    syncRadiometricBaseline();
}

void MetalRenderer::Impl::setupDefaultScene() {
    m_sceneResources.setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);

    std::string sceneError;
    if (!m_sceneManager.refresh(&sceneError) && !sceneError.empty()) {
        NSLog(@"SceneManager refresh failed: %s", sceneError.c_str());
    }
    const auto& available = m_sceneManager.scenes();

    std::string targetIdentifier;
    if (!m_options.initialScene.empty()) {
        targetIdentifier = m_options.initialScene;
    } else if (!available.empty()) {
        targetIdentifier = available.front().identifier;
    }

    if (!targetIdentifier.empty()) {
        namespace fs = std::filesystem;
        bool loadFromPath = false;
        fs::path candidate(targetIdentifier);
        if (candidate.is_absolute() || candidate.has_parent_path() || candidate.extension() == ".scene") {
            loadFromPath = true;
        } else {
            std::error_code probeEc;
            if (fs::exists(candidate, probeEc)) {
                loadFromPath = true;
            }
        }

        if (loadFromPath) {
            if (loadSceneFromPath(targetIdentifier)) {
                return;
            }
        } else {
            RenderSettings updatedSettings = m_settings;
            sceneError.clear();
            if (m_sceneManager.loadScene(targetIdentifier, m_sceneResources, updatedSettings, &sceneError)) {
                if (m_options.enableSoftwareRayTracing) {
                    updatedSettings.enableSoftwareRayTracing = true;
                }
                m_settings = updatedSettings;
                m_settings.renderScale = ClampRenderScale(m_settings.renderScale);
                m_sceneResources.setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);
                applyBackgroundSettings();
                m_sceneResources.rebuildAccelerationStructures();
                m_activeSceneId = targetIdentifier;
                return;
            }
            if (!sceneError.empty()) {
                NSLog(@"Failed to load scene '%s': %s", targetIdentifier.c_str(), sceneError.c_str());
            }
        }
    }

    buildProceduralScene();
}

void MetalRenderer::Impl::buildProceduralScene() {
    m_settings.backgroundMode = RenderSettings::BackgroundMode::Gradient;
    m_settings.backgroundColor = simd_make_float3(0.0f, 0.0f, 0.0f);
    m_settings.environmentMapPath.clear();
    m_settings.environmentRotation = 0.0f;
    m_settings.environmentIntensity = 1.0f;
    m_sceneResources.clearEnvironmentMap();

    m_sceneResources.clear();
    m_sceneResources.setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);

    struct ReservedSphere {
        simd::float3 center;
        float radius;
    };

    std::vector<ReservedSphere> placedSpheres;
    placedSpheres.reserve(kMaxSpheres);

    auto addSceneSphere = [&](const simd::float3& center, float radius, uint32_t materialIndex) {
        addSphere(center, radius, materialIndex);
        placedSpheres.push_back(ReservedSphere{center, radius});
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    auto randomFloat = [&]() { return dist(rng); };
    auto randomFloatRange = [&](float min, float max) {
        return min + (max - min) * randomFloat();
    };
    auto randomColor = [&]() -> simd::float3 {
        return simd_make_float3(randomFloat(), randomFloat(), randomFloat());
    };
    auto randomColorRange = [&](float min, float max) -> simd::float3 {
        return simd_make_float3(randomFloatRange(min, max),
                                randomFloatRange(min, max),
                                randomFloatRange(min, max));
    };

    uint32_t groundMaterial =
        addMaterial(simd_make_float3(0.5f, 0.5f, 0.5f), 0.0f, MaterialType::Lambertian, 1.0f);
    addSceneSphere(simd_make_float3(0.0f, -1000.0f, 0.0f), 1000.0f, groundMaterial);

    const std::array<ReservedSphere, 3> reservedLargeSpheres = {{
        {simd_make_float3(0.0f, 1.0f, 0.0f), 1.0f},
        {simd_make_float3(-4.0f, 1.0f, 0.0f), 1.0f},
        {simd_make_float3(4.0f, 1.0f, 0.0f), 1.0f},
    }};

    auto intersectsExisting = [&](const simd::float3& center, float radius) {
        constexpr float kSeparationEpsilon = 1e-3f;
        for (const auto& placed : placedSpheres) {
            if (placed.radius > 900.0f) {
                continue;  // skip ground
            }
            if (simd::length(center - placed.center) <
                radius + placed.radius + kSeparationEpsilon) {
                return true;
            }
        }
        for (const auto& reserved : reservedLargeSpheres) {
            if (simd::length(center - reserved.center) <
                radius + reserved.radius + kSeparationEpsilon) {
                return true;
            }
        }
        return false;
    };

    const uint32_t sharedGlassMaterial =
        addMaterial(simd_make_float3(1.0f, 1.0f, 1.0f), 0.0f, MaterialType::Dielectric, 1.50f);

    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            if (m_sceneResources.sphereCount() >= kMaxSpheres - 3 ||
                m_sceneResources.materialCount() >= kMaxMaterials - 3) {
                break;
            }

            simd::float3 center = simd_make_float3(static_cast<float>(a) + 0.9f * randomFloat(),
                                                   0.2f,
                                                   static_cast<float>(b) + 0.9f * randomFloat());

            constexpr float smallRadius = 0.2f;
            if (intersectsExisting(center, smallRadius)) {
                continue;
            }

            float normalizedZ = std::clamp((center.z + 11.0f) / 22.0f, 0.0f, 1.0f);
            constexpr float kNearOccupancy = 0.9f;
            constexpr float kFarOccupancy = 0.6f;
            float occupancyProbability =
                kNearOccupancy - (kNearOccupancy - kFarOccupancy) * normalizedZ;
            if (randomFloat() > occupancyProbability) {
                continue;
            }

            float chooseMaterial = randomFloat();
            uint32_t materialIndex = 0;

            if (chooseMaterial < 0.8f) {
                simd::float3 albedo = randomColor() * randomColor();
                materialIndex = addMaterial(albedo, 0.0f, MaterialType::Lambertian, 1.0f);
            } else if (chooseMaterial < 0.95f) {
                simd::float3 albedo = randomColorRange(0.5f, 1.0f);
                float roughness = randomFloatRange(0.0f, 0.5f);
                materialIndex = addMaterial(albedo, roughness, MaterialType::Metal, 1.0f);
            } else {
                materialIndex = sharedGlassMaterial;
            }

            addSceneSphere(center, smallRadius, materialIndex);
        }
    }

    uint32_t bigGlass = sharedGlassMaterial;
    uint32_t bigLambertian =
        addMaterial(simd_make_float3(0.4f, 0.2f, 0.1f), 0.0f, MaterialType::Lambertian, 1.0f);
    uint32_t bigMetal =
        addMaterial(simd_make_float3(0.7f, 0.6f, 0.5f), 0.0f, MaterialType::Metal, 1.0f);

    addSceneSphere(simd_make_float3(0.0f, 1.0f, 0.0f), 1.0f, bigGlass);
    addSceneSphere(simd_make_float3(-4.0f, 1.0f, 0.0f), 1.0f, bigLambertian);
    addSceneSphere(simd_make_float3(4.0f, 1.0f, 0.0f), 1.0f, bigMetal);

    m_sceneResources.setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);
    m_sceneResources.rebuildAccelerationStructures();
    m_activeSceneId.clear();
    syncCameraSmoothingToTarget();
}

void MetalRenderer::Impl::resetAccumulation() {
    m_performanceStats.samplesPerMinute = 0.0;
    m_performanceStats.avgNodesVisited = 0.0;
    m_performanceStats.avgLeafPrimTests = 0.0;
    m_performanceStats.shadowRayEarlyExitPct = 0.0;
    m_performanceStats.bothChildrenVisitedPct = 0.0;
    m_performanceStats.gpuTimeMs = 0.0;
    m_performanceStats.sampleCount = 0;
    m_performanceStats.sampleCount = 0;

    m_previousSampleCount = 0;

    CGSize targetSize = targetRenderSize();
    m_accumulation.ensureTextures(targetSize);
    m_accumulation.reset(m_context.commandQueue(), m_pipelines.clear(), nullptr);

    m_accumDirty = false;
    m_accumDirtyReason.clear();
    syncRadiometricBaseline();
}

void MetalRenderer::Impl::setTonemapMode(uint32_t mode) {
    uint32_t clamped = std::max<uint32_t>(1u, std::min<uint32_t>(mode, 4u));
    m_settings.tonemapMode = clamped;
}

bool MetalRenderer::Impl::loadScene(const std::string& identifier) {
    std::string refreshError;
    if (!m_sceneManager.refresh(&refreshError) && !refreshError.empty()) {
        NSLog(@"SceneManager refresh failed: %s", refreshError.c_str());
    }

    RenderSettings updatedSettings = m_settings;
    {
        // Reset camera to scene defaults on every switch so each scene starts from a known view.
        const RenderSettings cameraDefaults{};
        updatedSettings.cameraTarget = cameraDefaults.cameraTarget;
        updatedSettings.cameraDistance = cameraDefaults.cameraDistance;
        updatedSettings.cameraYaw = cameraDefaults.cameraYaw;
        updatedSettings.cameraPitch = cameraDefaults.cameraPitch;
        updatedSettings.cameraVerticalFov = cameraDefaults.cameraVerticalFov;
        updatedSettings.cameraDefocusAngle = cameraDefaults.cameraDefocusAngle;
        updatedSettings.cameraFocusDistance = cameraDefaults.cameraFocusDistance;
    }
    std::string errorMessage;
    if (!m_sceneManager.loadScene(identifier, m_sceneResources, updatedSettings, &errorMessage)) {
        if (!errorMessage.empty()) {
            NSLog(@"Failed to load scene '%s': %s", identifier.c_str(), errorMessage.c_str());
        }
        return false;
    }

    if (m_options.enableSoftwareRayTracing) {
        updatedSettings.enableSoftwareRayTracing = true;
    }
    m_settings = updatedSettings;
    m_settings.renderScale = ClampRenderScale(m_settings.renderScale);
    m_sceneResources.setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);
    applyBackgroundSettings();
    syncCameraSmoothingToTarget();
    m_activeSceneId = identifier;
    resetAccumulation();
    return true;
}

bool MetalRenderer::Impl::loadSceneFromPath(const std::string& path) {
    if (path.empty()) {
        NSLog(@"Invalid scene path (empty)");
        return false;
    }

    RenderSettings updatedSettings = m_settings;
    std::string errorMessage;
    if (!m_sceneManager.loadSceneFromPath(path, m_sceneResources, updatedSettings, &errorMessage)) {
        if (!errorMessage.empty()) {
            NSLog(@"Failed to load scene from '%s': %s", path.c_str(), errorMessage.c_str());
        }
        return false;
    }

    m_settings = updatedSettings;
    m_settings.renderScale = ClampRenderScale(m_settings.renderScale);
    m_sceneResources.setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);
    applyBackgroundSettings();
    m_sceneResources.rebuildAccelerationStructures();
    m_activeSceneId.clear();
    syncCameraSmoothingToTarget();
    resetAccumulation();
    return true;
}

std::vector<std::string> MetalRenderer::Impl::sceneIdentifiers() const {
    const_cast<SceneManager&>(m_sceneManager).refresh(nullptr);
    std::vector<std::string> identifiers;
    const auto& scenes = m_sceneManager.scenes();
    identifiers.reserve(scenes.size());
    for (const auto& info : scenes) {
        identifiers.push_back(info.identifier);
    }
    return identifiers;
}

void MetalRenderer::Impl::applySettings(const RenderSettings& settings, bool resetAccumulationFlag) {
    RenderSettings sanitized = settings;
    sanitized.renderScale = ClampRenderScale(sanitized.renderScale);
    m_settings = sanitized;
    m_sceneResources.setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);
    applyBackgroundSettings();
    syncCameraSmoothingToTarget();
    if (resetAccumulationFlag) {
        resetAccumulation();
    } else {
        m_accumulation.ensureTextures(targetRenderSize());
    }
}

bool MetalRenderer::Impl::captureAverageImage(std::vector<float>& outLinearRGB,
                                              uint32_t& width,
                                              uint32_t& height,
                                              std::vector<float>* outSampleCounts) {
    id<MTLTexture> present = m_accumulation.present();
    id<MTLDevice> device = m_context.device();
    id<MTLCommandQueue> queue = m_context.commandQueue();
    if (!present || !device || !queue) {
        return false;
    }

    width = static_cast<uint32_t>(present.width);
    height = static_cast<uint32_t>(present.height);

    MTLTextureDescriptor* descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                           width:present.width
                                                          height:present.height
                                                       mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderWrite;

    id<MTLTexture> readable = [device newTextureWithDescriptor:descriptor];
    if (!readable) {
        return false;
    }

    id<MTLCommandBuffer> blitCmd = [queue commandBuffer];
    blitCmd.label = @"Capture Image Blit";
    id<MTLBlitCommandEncoder> blitEncoder = [blitCmd blitCommandEncoder];
    [blitEncoder copyFromTexture:present toTexture:readable];
    [blitEncoder endEncoding];
    [blitCmd commit];
    [blitCmd waitUntilCompleted];

    std::vector<float> rgba(static_cast<size_t>(width) * height * 4u);
    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    [readable getBytes:rgba.data()
           bytesPerRow:static_cast<NSUInteger>(width) * 4u * sizeof(float)
            fromRegion:region
           mipmapLevel:0];

    outLinearRGB.resize(static_cast<size_t>(width) * height * 3u);
    if (outSampleCounts) {
        outSampleCounts->resize(static_cast<size_t>(width) * height);
    }

    for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
        outLinearRGB[3 * i + 0] = rgba[4 * i + 0];
        outLinearRGB[3 * i + 1] = rgba[4 * i + 1];
        outLinearRGB[3 * i + 2] = rgba[4 * i + 2];
        if (outSampleCounts) {
            (*outSampleCounts)[i] = rgba[4 * i + 3];
        }
    }

    return true;
}

void MetalRenderer::Impl::handleSaveExrRequest(const std::string& path) {
    if (path.empty()) {
        m_overlay.notifySaveExrResult(false, "Invalid save path");
        NSLog(@"[Output] Save EXR aborted: empty path");
        return;
    }

    std::vector<float> linearRGB;
    std::vector<float> sampleCounts;
    uint32_t width = 0;
    uint32_t height = 0;
    if (!captureAverageImage(linearRGB, width, height, &sampleCounts) || width == 0 || height == 0) {
        m_overlay.notifySaveExrResult(false, "No accumulation data");
        NSLog(@"[Output] Save EXR failed: no accumulation data available");
        return;
    }

    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (linearRGB.size() != pixelCount * 3ull) {
        m_overlay.notifySaveExrResult(false, "Unexpected buffer size");
        NSLog(@"[Output] Save EXR failed: unexpected RGB buffer size");
        return;
    }

    std::vector<float> rgba(pixelCount * 4ull);
    for (size_t i = 0; i < pixelCount; ++i) {
        rgba[4 * i + 0] = linearRGB[3 * i + 0];
        rgba[4 * i + 1] = linearRGB[3 * i + 1];
        rgba[4 * i + 2] = linearRGB[3 * i + 2];
        const float alpha = sampleCounts.empty() ? 1.0f : sampleCounts[i];
        rgba[4 * i + 3] = alpha;
    }

    const bool hasSamples = !sampleCounts.empty();
    bool success = false;
    namespace fs = std::filesystem;
    fs::path targetPath(path);
    std::error_code dirEc;
    if (!targetPath.has_parent_path()) {
        targetPath = fs::path("renders") / targetPath;
    }
    fs::path parent = targetPath.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent, dirEc);
    }

    if (hasSamples) {
        success = PathTracer::ImageWriter::WriteEXR_Multilayer(targetPath.string().c_str(),
                                                               rgba.data(),
                                                               static_cast<int>(width),
                                                               static_cast<int>(height),
                                                               sampleCounts.data());
    } else {
        success = PathTracer::ImageWriter::WriteEXR(targetPath.string().c_str(),
                                                    rgba.data(),
                                                    static_cast<int>(width),
                                                    static_cast<int>(height));
    }

    if (success) {
        NSLog(@"[Output] Saved EXR to %s (%ux%u%s)",
              targetPath.string().c_str(),
              width,
              height,
              hasSamples ? " + SAMPLES" : "");
        std::string status = "Saved " + std::to_string(width) + "x" + std::to_string(height) + " EXR to ";
        status += targetPath.string();
        if (hasSamples) {
            status += " (+SAMPLES)";
        }
        m_overlay.notifySaveExrResult(true, status);
    } else {
        NSLog(@"[Output] Failed to save EXR to %s", targetPath.string().c_str());
        m_overlay.notifySaveExrResult(false, "Failed to save EXR");
    }
}

bool MetalRenderer::Impl::exportToPPM(const char* filepath) {
    if (!filepath) {
        NSLog(@"Invalid filepath for PPM export");
        return false;
    }

    std::vector<float> linearRGB;
    uint32_t width = 0;
    uint32_t height = 0;
    if (!captureAverageImage(linearRGB, width, height, nullptr)) {
        NSLog(@"No render data to export");
        return false;
    }

    PathTracer::TonemapSettings tonemap{};
    tonemap.tonemapMode = 1;
    tonemap.acesVariant = 0;
    tonemap.exposure = 0.0f;
    tonemap.reinhardWhitePoint = 1.5f;

    std::string error;
    if (!PathTracer::WriteImage(std::string(filepath), PathTracer::ImageFileFormat::PPM,
                                linearRGB.data(), width, height, tonemap, &error)) {
        if (!error.empty()) {
            NSLog(@"PPM export failed: %s", error.c_str());
        }
        return false;
    }

    NSLog(@"Successfully exported PPM to: %s (%ux%u, %u samples)",
          filepath, width, height, m_accumulation.sampleCount());
    return true;
}

void MetalRenderer::Impl::shutdown() {
    if (m_shutdown) {
        return;
    }
    m_shutdown = true;

    if (m_backingChangeObserver) {
        [[NSNotificationCenter defaultCenter] removeObserver:m_backingChangeObserver];
        m_backingChangeObserver = nil;
        m_context.setBackingChangeObserver(nil);
    }
    if (m_windowMoveObserver) {
        [[NSNotificationCenter defaultCenter] removeObserver:m_windowMoveObserver];
        m_windowMoveObserver = nil;
    }
    if (m_windowWillMoveObserver) {
        [[NSNotificationCenter defaultCenter] removeObserver:m_windowWillMoveObserver];
        m_windowWillMoveObserver = nil;
    }

    if (auto queue = m_context.commandQueue()) {
        id<MTLCommandBuffer> drain = [queue commandBuffer];
        drain.label = @"Renderer Shutdown Drain";
        [drain commit];
        [drain waitUntilCompleted];
    }

    m_pipelines.shutdown();
#if HAS_IMGUI
    if (m_overlayInitialized) {
        m_overlay.shutdown();
        m_overlayInitialized = false;
    }
#endif

    m_renderLoop.shutdown();
    m_sceneResources.clear();
    m_accumulation.teardown();

    m_context.shutdown();
}

MetalRenderer::Impl::~Impl() {
    shutdown();
}

MetalRenderer::MetalRenderer() : m_impl(std::make_shared<Impl>()) {}
MetalRenderer::~MetalRenderer() = default;

bool MetalRenderer::init(void* windowHandle, const MetalRendererOptions& options) {
    // Headless mode doesn't require a window
    if (!options.headless && !windowHandle) {
        return false;
    }

    NSWindow* window = nullptr;
    if (windowHandle) {
        window = (__bridge NSWindow*)windowHandle;
        [window setTitle:[NSString stringWithUTF8String:options.windowTitle.c_str()]];
    }
    return m_impl->initialize(window, options);
}

void MetalRenderer::drawFrame() {
    if (m_impl) {
        m_impl->drawFrame();
    }
}

void MetalRenderer::resize(int width, int height) {
    if (m_impl) {
        m_impl->resize(width, height);
    }
}

void MetalRenderer::resetAccumulation() {
    if (m_impl) {
        m_impl->resetAccumulation();
    }
}

void MetalRenderer::setTonemapMode(uint32_t mode) {
    if (m_impl) {
        m_impl->setTonemapMode(mode);
    }
}

void MetalRenderer::shutdown() {
    if (m_impl) {
        m_impl->shutdown();
    }
}

bool MetalRenderer::exportToPPM(const char* filepath) {
    if (m_impl) {
        return m_impl->exportToPPM(filepath);
    }
    return false;
}

bool MetalRenderer::loadSceneFromPath(const char* path) {
    if (!m_impl || !path) {
        return false;
    }
    return m_impl->loadSceneFromPath(path);
}

bool MetalRenderer::setScene(const char* identifier) {
    if (!m_impl || !identifier) {
        return false;
    }
    std::string idStr(identifier);
    if (idStr.find('/') != std::string::npos || idStr.find(".scene") != std::string::npos) {
        return m_impl->loadSceneFromPath(idStr);
    }
    return m_impl->loadScene(idStr);
}

std::vector<std::string> MetalRenderer::sceneIdentifiers() const {
    if (!m_impl) {
        return {};
    }
    return m_impl->sceneIdentifiers();
}

PathTracer::RenderSettings MetalRenderer::settings() const {
    if (!m_impl) {
        return PathTracer::RenderSettings{};
    }
    return m_impl->currentSettings();
}

void MetalRenderer::applySettings(const PathTracer::RenderSettings& settings, bool resetAccumulation) {
    if (m_impl) {
        m_impl->applySettings(settings, resetAccumulation);
    }
}

void MetalRenderer::setSamplesPerFrame(uint32_t samples) {
    if (m_impl) {
        m_impl->setSamplesPerFrame(samples);
    }
}

uint32_t MetalRenderer::sampleCount() const {
    if (!m_impl) {
        return 0;
    }
    return m_impl->sampleCount();
}

bool MetalRenderer::captureAverageImage(std::vector<float>& outLinearRGB,
                                        uint32_t& width,
                                        uint32_t& height,
                                        std::vector<float>* outSampleCounts) {
    if (!m_impl) {
        return false;
    }
    return m_impl->captureAverageImage(outLinearRGB, width, height, outSampleCounts);
}
