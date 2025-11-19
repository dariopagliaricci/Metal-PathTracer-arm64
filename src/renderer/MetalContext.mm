#import "renderer/MetalContext.h"

#import <algorithm>
#import <QuartzCore/CAMetalLayer.h>

namespace PathTracer {

bool MetalContext::initialize(NSWindowHandle window, bool headless) {
    m_headless = headless;
    m_window = window;

    // Create Metal device
    m_device = MTLCreateSystemDefaultDevice();
    if (!m_device) {
        NSLog(@"Metal is not supported on this device");
        return false;
    }
    
    // Check raytracing support
    if ([m_device respondsToSelector:@selector(supportsRaytracing)]) {
        m_supportsRaytracing = [m_device supportsRaytracing];
    } else {
        m_supportsRaytracing = false;
    }
    
    // Create command queue
    m_queue = [m_device newCommandQueue];
    if (!m_queue) {
        NSLog(@"Failed to create Metal command queue");
        return false;
    }

    if (!m_headless) {
        if (!window) {
            NSLog(@"MetalContext::initialize - window is nil in windowed mode");
            return false;
        }

        NSRect bounds = [[window contentView] bounds];
        m_view = [[MTKView alloc] initWithFrame:bounds device:m_device];
        if (!m_view) {
            NSLog(@"Failed to create MTKView");
            return false;
        }

        m_view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
        m_view.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
        m_view.framebufferOnly = NO;
        m_view.autoResizeDrawable = NO;
        m_view.paused = YES;
        m_view.enableSetNeedsDisplay = YES;
        m_view.preferredFramesPerSecond = 60;
        m_view.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

        [window setContentView:m_view];
        updateDrawableSize();
    }
    
    return true;
}

void MetalContext::updateDrawableSize() {
    if (m_headless || !m_view || !m_window) {
        return;
    }
    
    // Get screen backing scale
    NSScreen* screen = m_window.screen ?: [NSScreen mainScreen];
    CGFloat scale = 1.0f;
    if (m_hasForcedContentScale) {
        scale = std::max<CGFloat>(m_forcedContentScale, 1.0f);
    } else {
        const CGFloat backing = screen.backingScaleFactor;
        scale = backing > 0.0 ? backing : 1.0f;
    }
    scale = std::max<CGFloat>(scale, 1.0);
    m_contentScale = static_cast<float>(scale);
    
    // Update layer scale
    if (m_view.layer) {
        m_view.layer.contentsScale = scale;
    }
    
    // Calculate drawable size
    CGSize bounds = m_view.bounds.size;
    CGSize drawable = CGSizeMake(std::max<CGFloat>(bounds.width * scale, 1.0f),
                                 std::max<CGFloat>(bounds.height * scale, 1.0f));
    m_view.drawableSize = drawable;
}

void MetalContext::shutdown() {
    if (m_backingChangeObserver) {
        [[NSNotificationCenter defaultCenter] removeObserver:m_backingChangeObserver];
        m_backingChangeObserver = nil;
    }

    if (m_view) {
        m_view.paused = YES;
        m_view.enableSetNeedsDisplay = NO;
        m_view.delegate = nil;

        if (m_view.layer && [m_view.layer isKindOfClass:[CAMetalLayer class]]) {
            CAMetalLayer* metalLayer = (CAMetalLayer*)m_view.layer;
            metalLayer.device = nil;
            metalLayer.drawableSize = CGSizeZero;
        }

        m_view.device = nil;
        [m_view removeFromSuperview];
        m_view = nil;
    }

    m_queue = nil;
    m_device = nil;
    m_window = nil;
    m_supportsRaytracing = false;
    m_headless = false;
    m_contentScale = 1.0f;
}

CGSize MetalContext::drawableSize() const {
    if (m_view) {
        return m_view.drawableSize;
    }
    return CGSizeZero;
}

void MetalContext::setForcedContentScale(float scale, bool enabled) {
    if (enabled) {
        m_hasForcedContentScale = true;
        m_forcedContentScale = std::max(scale, 1.0f);
    } else {
        m_hasForcedContentScale = false;
    }
    updateDrawableSize();
}

}  // namespace PathTracer
