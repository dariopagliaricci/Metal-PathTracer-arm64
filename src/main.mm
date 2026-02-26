#import <Cocoa/Cocoa.h>
#import <Carbon/Carbon.h>

#include "MetalRenderer.h"
#include "backends/imgui_impl_osx.h"

#include <cstring>
#include <string>

namespace {
constexpr int kInitialWidth = 1280;
constexpr int kInitialHeight = 720;
}

@interface PathTracerAppDelegate : NSObject<NSApplicationDelegate>
@property(nonatomic, assign) bool* shouldQuitFlag;
@property(nonatomic, assign) bool* pendingTerminationFlag;
- (void)handleQuitAppleEvent:(NSAppleEventDescriptor*)event
             withReplyEvent:(NSAppleEventDescriptor*)replyEvent;
@end

@implementation PathTracerAppDelegate
- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication*)sender {
    if (self.shouldQuitFlag) {
        *self.shouldQuitFlag = true;
    }
    if (self.pendingTerminationFlag) {
        *self.pendingTerminationFlag = true;
    }
    return NSTerminateLater;
}

- (void)handleQuitAppleEvent:(NSAppleEventDescriptor*)event
             withReplyEvent:(NSAppleEventDescriptor*)replyEvent {
    if (self.shouldQuitFlag) {
        *self.shouldQuitFlag = true;
    }
    if (self.pendingTerminationFlag) {
        *self.pendingTerminationFlag = true;
    }
}
@end

@interface PathTracerWindowDelegate : NSObject<NSWindowDelegate>
@property(nonatomic, assign) bool* shouldQuitFlag;
@end

@implementation PathTracerWindowDelegate
- (void)windowWillClose:(NSNotification*)notification {
    if (self.shouldQuitFlag) {
        *self.shouldQuitFlag = true;
    }
}
@end

int main(int argc, const char** argv) {
    bool presentationMode = false;
    for (int i = 1; i < argc; ++i) {
        if (!argv[i]) {
            continue;
        }
        std::string arg(argv[i]);
        constexpr const char* kPresentationPrefix = "--presentation=";
        if (arg.rfind(kPresentationPrefix, 0) == 0) {
            std::string value = arg.substr(std::strlen(kPresentationPrefix));
            if (value == "1" || value == "true" || value == "on") {
                presentationMode = true;
            } else if (value == "0" || value == "false" || value == "off") {
                presentationMode = false;
            }
        }
    }
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        NSRect frame = NSMakeRect(100, 100, kInitialWidth, kInitialHeight);
        NSWindowStyleMask style = NSWindowStyleMaskTitled |
                                  NSWindowStyleMaskClosable |
                                  NSWindowStyleMaskMiniaturizable |
                                  NSWindowStyleMaskResizable;
        NSWindow* window = [[NSWindow alloc] initWithContentRect:frame
                                                       styleMask:style
                                                         backing:NSBackingStoreBuffered
                                                           defer:NO];
        [window center];

        MetalRendererOptions options;
        options.width = kInitialWidth;
        options.height = kInitialHeight;
        options.windowTitle = "Path Tracer Metal";
        options.presentationMode = presentationMode;
        [window setTitle:[NSString stringWithUTF8String:options.windowTitle.c_str()]];

        __block bool shouldQuit = false;
        __strong PathTracerWindowDelegate* windowDelegate = [[PathTracerWindowDelegate alloc] init];
        windowDelegate.shouldQuitFlag = &shouldQuit;
        window.delegate = windowDelegate;

        __block bool terminationRequested = false;
        __strong PathTracerAppDelegate* appDelegate = [[PathTracerAppDelegate alloc] init];
        appDelegate.shouldQuitFlag = &shouldQuit;
        appDelegate.pendingTerminationFlag = &terminationRequested;
        [NSApp setDelegate:appDelegate];
        [NSApp finishLaunching];
        [[NSAppleEventManager sharedAppleEventManager] setEventHandler:appDelegate
                                                           andSelector:@selector(handleQuitAppleEvent:withReplyEvent:)
                                                         forEventClass:kCoreEventClass
                                                        andEventID:kAEQuitApplication];

        {
            MetalRenderer renderer;
            if (!renderer.init((__bridge void*)window, options)) {
                NSLog(@"Failed to initialize Metal renderer");
                renderer.shutdown();
                window.delegate = nil;
                windowDelegate.shouldQuitFlag = nullptr;
                appDelegate.shouldQuitFlag = nullptr;
                appDelegate.pendingTerminationFlag = nullptr;
                [NSApp setDelegate:nil];
                return -1;
            }

            [NSApp activateIgnoringOtherApps:YES];
            [window makeKeyAndOrderFront:nil];

            while (!shouldQuit) {
                @autoreleasepool {
                    NSEvent* event = nil;
                    do {
                        event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                                   untilDate:[NSDate dateWithTimeIntervalSinceNow:1.0 / 120.0]
                                                      inMode:NSDefaultRunLoopMode
                                                     dequeue:YES];
                        if (event) {
                            if (event.type == NSEventTypeKeyDown) {
                                if (event.keyCode == kVK_Escape && renderer.isPresentationEnabled()) {
                                    renderer.setPresentationEnabled(false);
                                    event = nil;
                                    continue;
                                }
                                if (event.keyCode == kVK_F1 && renderer.isPresentationEnabled()) {
                                    renderer.togglePresentationUIPanels();
                                    event = nil;
                                    continue;
                                }
                                if ((event.modifierFlags & NSEventModifierFlagCommand)) {
                                    NSString* chars = event.charactersIgnoringModifiers.lowercaseString;
                                    if ([chars isEqualToString:@"q"]) {
                                        shouldQuit = true;
                                        terminationRequested = true;
                                        event = nil;
                                        continue;
                                    }
                                }
                            }
                            [NSApp sendEvent:event];
                        }
                    } while (event != nil);

                    renderer.drawFrame();
                }
            }

            windowDelegate.shouldQuitFlag = nullptr;
            renderer.shutdown();
        }

        if (terminationRequested) {
            [NSApp replyToApplicationShouldTerminate:YES];
            terminationRequested = false;
        }

        appDelegate.shouldQuitFlag = nullptr;
        appDelegate.pendingTerminationFlag = nullptr;
        [[NSAppleEventManager sharedAppleEventManager] removeEventHandlerForEventClass:kCoreEventClass
                                                                           andEventID:kAEQuitApplication];
        [NSApp setDelegate:nil];

        NSView* placeholderView = [[NSView alloc] initWithFrame:NSZeroRect];
        [window setContentView:placeholderView];
        window.delegate = nil;
        [window orderOut:nil];
    }
    return 0;
}
