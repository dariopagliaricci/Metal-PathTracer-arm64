#pragma once

#include <CoreGraphics/CoreGraphics.h>
#include <cstdint>
#include "renderer/MetalHandles.h"

namespace PathTracer {

/// Manages Metal device, command queue, and view lifecycle.
class MetalContext {
public:
    MetalContext() = default;
    ~MetalContext() = default;

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;
    MetalContext(MetalContext&&) = default;
    MetalContext& operator=(MetalContext&&) = default;

    /// Initialize the Metal context.
    /// @param window Source window for non-headless rendering.
    /// @param headless If true, skip MTKView creation.
    bool initialize(NSWindowHandle window, bool headless);

    /// Update drawable size using current backing scale.
    void updateDrawableSize();

    /// Release all associated resources.
    void shutdown();

    // Accessors
    MTLDeviceHandle device() const { return m_device; }
    MTLCommandQueueHandle commandQueue() const { return m_queue; }
    MTKViewHandle view() const { return m_view; }
    NSWindowHandle window() const { return m_window; }
    bool supportsRaytracing() const { return m_supportsRaytracing; }
    bool headless() const { return m_headless; }
    float contentScale() const { return m_contentScale; }
    CGSize drawableSize() const;

    void setBackingChangeObserver(ObjCObserverHandle observer) { m_backingChangeObserver = observer; }
    ObjCObserverHandle backingChangeObserver() const { return m_backingChangeObserver; }

    /// Force a specific content scale (e.g., disable Retina scaling).
    void setForcedContentScale(float scale, bool enabled);
    bool hasForcedContentScale() const { return m_hasForcedContentScale; }

private:
    MTLDeviceHandle m_device = nullptr;
    MTLCommandQueueHandle m_queue = nullptr;
    MTKViewHandle m_view = nullptr;
    NSWindowHandle m_window = nullptr;

    bool m_supportsRaytracing = false;
    bool m_headless = false;
    float m_contentScale = 1.0f;

    ObjCObserverHandle m_backingChangeObserver = nullptr;
    bool m_hasForcedContentScale = false;
    float m_forcedContentScale = 1.0f;
};

}  // namespace PathTracer
