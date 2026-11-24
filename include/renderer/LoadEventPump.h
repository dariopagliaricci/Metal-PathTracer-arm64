#pragma once

// AppKit event pumping during scene loads, shared by the loaders (glTF, OBJ/PLY, scene resources, acceleration
// structures).
//
// A load that runs on the GUI's main thread drains pending events from inside its loops, so the app keeps
// answering the window server. Until 2026-11-20 every pump waited up to 1 ms for an event in
// nextEventMatchingMask (about 4.6 ms when sampled, consistent with timer coalescing) and ran every few meshes or
// accessor elements, in headless too. On barcelona_pavilion_day that was 67% of the main thread's 82 s load.
// The rules now:
// - Only a process with an NSApplication pumps. The GUI creates one before it loads anything. Headless and the
//   other command-line tools never create one, and nothing here does, so they never pump.
// - A load off the main thread (the GUI's scene switch) neither pumps nor sleeps.
// - The GUI's main thread drains only the events already queued, without waiting, at most once per
//   kMinPumpIntervalSeconds of wall time.
// - Waiting for a command buffer keeps the 1 ms event wait on the GUI's main thread, where the GPU is the
//   bottleneck. Everywhere else it blocks in waitUntilCompleted.

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#include <cstddef>

namespace PathTracer::LoadEventPump {

constexpr CFTimeInterval kMinPumpIntervalSeconds = 1.0 / 60.0;

inline bool CanPump() {
    return NSApp != nil && [NSThread isMainThread];
}

// Sends the queued events of the modes the loaders have always drained. untilDate decides how long the first
// mode may wait for an event; the later modes see the same deadline.
inline void DrainEvents(NSDate* untilDate) {
    NSApplication* app = NSApp;
    for (NSString* mode in @[ NSDefaultRunLoopMode, NSEventTrackingRunLoopMode, NSModalPanelRunLoopMode ]) {
        while (NSEvent* event = [app nextEventMatchingMask:NSEventMaskAny
                                                 untilDate:untilDate
                                                    inMode:mode
                                                   dequeue:YES]) {
            [app sendEvent:event];
        }
    }
}

// Drains the events already queued, if this is the GUI's main thread and the last pump is at least
// kMinPumpIntervalSeconds old. Never waits.
inline void PumpIfDue() {
    if (!CanPump()) {
        return;
    }
    static CFTimeInterval lastPump = 0.0;  // main thread only
    const CFTimeInterval now = CACurrentMediaTime();
    if (now - lastPump < kMinPumpIntervalSeconds) {
        return;
    }
    lastPump = now;
    @autoreleasepool {
        DrainEvents([NSDate distantPast]);
    }
}

// Called inside load loops: considers a pump every `interval` iterations.
inline void PumpEventsEvery(size_t counter, size_t interval = 16384u) {
    if (counter > 0u && (counter % interval) == 0u) {
        PumpIfDue();
    }
}

// Waits for a committed command buffer to complete or fail.
inline void WaitForCommandBufferResponsive(id<MTLCommandBuffer> commandBuffer) {
    if (!commandBuffer) {
        return;
    }
    if (!CanPump()) {
        [commandBuffer waitUntilCompleted];
        return;
    }
    while (true) {
        const MTLCommandBufferStatus status = commandBuffer.status;
        if (status == MTLCommandBufferStatusCompleted ||
            status == MTLCommandBufferStatusError) {
            break;
        }
        @autoreleasepool {
            DrainEvents([NSDate dateWithTimeIntervalSinceNow:0.001]);
        }
    }
}

}  // namespace PathTracer::LoadEventPump
