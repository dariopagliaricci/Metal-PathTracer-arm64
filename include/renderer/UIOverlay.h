#pragma once

// ⚠️ INTERNAL HEADER - Subject to change without notice
// This header is part of PathTracer's internal implementation.
// Use only the public API from MetalRenderer.h

#include <vector>
#include <string>
#include <functional>
#include <cstdint>
#include <CoreGraphics/CoreGraphics.h>
#include <simd/simd.h>

#include "renderer/MetalHandles.h"
#include "PerformanceStats.h"
#include "RenderSettings.h"
#include "MetalShaderTypes.h"

namespace PathTracer {

struct MaterialDisplayInfo {
    std::string name;
    PathTracerShaderTypes::MaterialData data{};
};

struct ObjectDisplayInfo {
    std::string name;
    uint32_t meshIndex = 0;
    uint32_t materialIndex = 0;
    simd::float4x4 transform = matrix_identity_float4x4;
};

struct ScenePanelProvider {
    std::function<uint32_t()> materialCount;
    std::function<bool(uint32_t, MaterialDisplayInfo&)> readMaterial;
    std::function<void(uint32_t, const MaterialDisplayInfo&)> applyMaterial;
    std::function<void(uint32_t)> resetMaterial;

    std::function<uint32_t()> objectCount;
    std::function<bool(uint32_t, ObjectDisplayInfo&)> readObject;
    std::function<void(uint32_t, const ObjectDisplayInfo&)> applyObjectTransform;
    std::function<void(uint32_t)> resetObjectTransform;
};

struct PresentationSettings {
    enum class TargetScreen : uint32_t {
        Auto = 0,
        Primary = 1,
        External = 2,
    };

    enum class WindowMode : uint32_t {
        BorderlessFullscreen = 0,
        Maximized = 1,
    };

    enum class RenderResolutionLock : uint32_t {
        Off = 0,
        Lock1280x720 = 1,
        Lock1920x1080 = 2,
    };

    enum class ContentScaleMode : uint32_t {
        Auto = 0,
        Scale1x = 1,
        Scale2x = 2,
    };

    bool enabled = false;
    bool hideUIPanels = true;
    bool minimalOverlay = true;
    bool resetAccumulationOnToggle = true;
    TargetScreen targetScreen = TargetScreen::Auto;
    WindowMode windowMode = WindowMode::BorderlessFullscreen;
    RenderResolutionLock resolutionLock = RenderResolutionLock::Lock1920x1080;
    ContentScaleMode contentScale = ContentScaleMode::Auto;
};

/// ImGui-based performance overlay
/// Manages ImGui initialization, frame lifecycle, and UI building
class UIOverlay {
public:
    UIOverlay() = default;
    ~UIOverlay();
    
    // Non-copyable
    UIOverlay(const UIOverlay&) = delete;
    UIOverlay& operator=(const UIOverlay&) = delete;
    
    /// Initialize ImGui for Metal + macOS
    /// @param view NSView for ImGui input handling
    /// @param device Metal device for ImGui rendering
    /// @param contentScale Display scale factor for HiDPI
    /// @return true if initialization succeeded
    bool initialize(NSViewHandle view, MTLDeviceHandle device, float contentScale);
    
    /// Begin new ImGui frame
    /// @param renderPassDescriptor Render pass for ImGui to target
    void beginFrame(MTLRenderPassDescriptorHandle renderPassDescriptor);
    
    /// Build UI with current stats and settings
    /// @param stats Current performance statistics
    /// @param settings Render settings (will be modified by UI)
    /// @param outNeedsReset Set to true if user requests reset
    void buildUI(const PerformanceStats& stats,
                 RenderSettings& settings,
                 bool& outNeedsReset,
                 const std::vector<const char*>& sceneNames,
                 int& inOutSelectedScene,
                 PresentationSettings& presentation);
    
    /// Render ImGui to command encoder
    /// @param commandBuffer Command buffer for completed handler
    /// @param renderEncoder Render encoder to draw into
    void render(MTLCommandBufferHandle commandBuffer,
                MTLRenderCommandEncoderHandle renderEncoder);
    
    /// Cleanup ImGui resources
    void shutdown();
    
    /// Check if ImGui wants keyboard/mouse input
    bool wantsInput() const;

    /// Check for pending EXR save path and consume it.
    bool consumeSaveExrRequest(std::string& outPath);

    /// Report status of the last EXR save operation for UI feedback.
    void notifySaveExrResult(bool success, const std::string& message);
    
    /// Check if overlay is visible
    bool isVisible() const { return m_visible; }
    
    /// Toggle overlay visibility
    void setVisible(bool visible) { m_visible = visible; }

    /// Toggle main UI panel visibility (Presentation Mode).
    void setMainPanelVisible(bool visible) { m_mainPanelVisible = visible; }
    bool isMainPanelVisible() const { return m_mainPanelVisible; }

    /// Toggle minimal overlay visibility (Presentation Mode).
    void setMinimalOverlayVisible(bool visible) { m_minimalOverlayVisible = visible; }
    bool isMinimalOverlayVisible() const { return m_minimalOverlayVisible; }

    /// Check initialization state
    bool isInitialized() const { return m_initialized; }

    /// Update ImGui display metrics (logical size in points and backing scale)
    void updateDisplayMetrics(CGSize logicalSizePoints, float backingScale);

    /// Force a specific content scale (e.g., 1x or 2x). When disabled, trust backend metrics.
    void setForcedContentScale(float scale, bool enabled);

    /// Provide callbacks for scene/object inspection & editing.
    void setScenePanelProvider(ScenePanelProvider provider);
    
private:
    struct MaterialEditorState {
        bool initialized = false;
        std::string name;
        PathTracerShaderTypes::MaterialData workingData{};
        PathTracerShaderTypes::MaterialData sourceData{};
    };

    struct ObjectEditorState {
        bool initialized = false;
        std::string name;
        uint32_t meshIndex = 0;
        uint32_t materialIndex = 0;
        simd::float4x4 workingTransform = matrix_identity_float4x4;
        simd::float4x4 sourceTransform = matrix_identity_float4x4;
    };

    void promptSaveExr();
    void drawSaveStatus();

    void rebuildFontsIfNeeded();
    void rebuildFonts(float scale);
    void requestFontRebuild(float scale);
    void applyDisplayState();

    bool m_initialized = false;
    bool m_visible = true;
    bool m_mainPanelVisible = true;
    bool m_minimalOverlayVisible = false;
    bool m_showDemo = false;
    NSViewHandle m_view = nullptr;
    MTLDeviceHandle m_device = nullptr;
    CGSize m_displaySizePoints = CGSizeMake(1.0, 1.0);
    float m_backingScale = 1.0f;
    float m_displayScale = 1.0f;
    bool m_hasForcedContentScale = false;
    float m_forcedContentScale = 1.0f;
    float m_lastFontScale = 1.0f;
    float m_pendingFontScale = 1.0f;
    bool m_fontScaleInitialized = false;
    bool m_pendingFontRebuild = false;
    std::string m_pendingExrPath;
    bool m_lastSaveExrSuccess = false;
    std::string m_lastSaveExrMessage;
    bool m_savePanelActive = false;
    ScenePanelProvider m_scenePanelProvider{};
    bool m_hasScenePanelProvider = false;
    std::vector<MaterialEditorState> m_materialStates;
    std::vector<ObjectEditorState> m_objectStates;
    int m_selectedMaterial = 0;
    int m_selectedObject = 0;
    int m_gizmoOperation = 0;  // 0=translate,1=rotate,2=scale
    int m_gizmoMode = 1;       // 0=local,1=world
    bool m_gizmoUseSnap = false;
    float m_gizmoSnapTranslation[3] = {0.5f, 0.5f, 0.5f};
    float m_gizmoSnapAngle = 5.0f;
    float m_gizmoSnapScale = 0.1f;
    bool m_gizmoWasActive = false;
    int m_gizmoActiveObject = -1;
};

}  // namespace PathTracer
