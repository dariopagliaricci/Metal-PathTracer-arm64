#pragma once

#ifdef __OBJC__
#include <AppKit/AppKit.h>
#include <Metal/Metal.h>
#include <MetalKit/MetalKit.h>

@protocol CAMetalDrawable;

using MTLDeviceHandle = id<MTLDevice>;
using MTLCommandQueueHandle = id<MTLCommandQueue>;
using MTLCommandBufferHandle = id<MTLCommandBuffer>;
using MTLComputePipelineStateHandle = id<MTLComputePipelineState>;
using MTLRenderPipelineStateHandle = id<MTLRenderPipelineState>;
using MTLRenderCommandEncoderHandle = id<MTLRenderCommandEncoder>;
using MTLLibraryHandle = id<MTLLibrary>;
using MTLBufferHandle = id<MTLBuffer>;
using MTLAccelerationStructureHandle = id<MTLAccelerationStructure>;
using MTLTextureHandle = id<MTLTexture>;
using MTLSamplerStateHandle = id<MTLSamplerState>;
using MTLRenderPassDescriptorHandle = MTLRenderPassDescriptor*;
using MTLCAMetalDrawableHandle = id<CAMetalDrawable>;
using MTKViewHandle = MTKView*;
using NSWindowHandle = NSWindow*;
using NSViewHandle = NSView*;
using ObjCObserverHandle = id;

#else

class MTKView;
class NSWindow;
class NSView;
struct MTLRenderPassDescriptor;
class CAMetalDrawable;
class MTLRenderCommandEncoder;

using MTLDeviceHandle = void*;
using MTLCommandQueueHandle = void*;
using MTLCommandBufferHandle = void*;
using MTLComputePipelineStateHandle = void*;
using MTLRenderPipelineStateHandle = void*;
using MTLRenderCommandEncoderHandle = void*;
using MTLLibraryHandle = void*;
using MTLBufferHandle = void*;
using MTLAccelerationStructureHandle = void*;
using MTLTextureHandle = void*;
using MTLSamplerStateHandle = void*;
using MTLRenderPassDescriptorHandle = void*;
using MTLCAMetalDrawableHandle = void*;
using MTKViewHandle = MTKView*;
using NSWindowHandle = NSWindow*;
using NSViewHandle = void*;
using ObjCObserverHandle = void*;

using MTLPixelFormat = unsigned long;

#endif
