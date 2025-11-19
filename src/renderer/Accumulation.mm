#import "renderer/Accumulation.h"

#import <AppKit/AppKit.h>
#import <MetalKit/MetalKit.h>

#include <algorithm>
#include <cmath>

#ifndef DEBUG_FIXED_RENDER_RESOLUTION
#define DEBUG_FIXED_RENDER_RESOLUTION 0
#endif

#if DEBUG_FIXED_RENDER_RESOLUTION
constexpr uint32_t kDebugRenderWidth = 1280;
constexpr uint32_t kDebugRenderHeight = 720;
#endif

namespace {

constexpr MTLPixelFormat kRadianceSumFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kSampleCountFormat = MTLPixelFormatR32Uint;
constexpr MTLPixelFormat kPresentFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kAlbedoFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kNormalFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kDenoisedFormat = MTLPixelFormatRGBA32Float;

}  // namespace

namespace PathTracer {

void Accumulation::initialize(MTLDeviceHandle device) {
    m_device = device;
}

void Accumulation::ensureTextures(CGSize drawableSize, bool force) {
    if (!m_device) {
        return;
    }
    if (drawableSize.width < 1.0 || drawableSize.height < 1.0) {
        return;
    }

#if DEBUG_FIXED_RENDER_RESOLUTION
    const CGFloat targetWidth = static_cast<CGFloat>(kDebugRenderWidth);
    const CGFloat targetHeight = static_cast<CGFloat>(kDebugRenderHeight);
#else
    const CGFloat targetWidth = std::max<CGFloat>(drawableSize.width, 1.0f);
    const CGFloat targetHeight = std::max<CGFloat>(drawableSize.height, 1.0f);
#endif

    const bool hasTextures =
        (m_radianceSumTexture && m_sampleCountTexture && m_presentTexture &&
         m_albedoTexture && m_normalTexture && m_denoisedTexture);
    if (!force && hasTextures) {
        constexpr CGFloat kSizeTolerance = 0.5f;
        const CGFloat deltaWidth = std::fabs(targetWidth - m_currentSize.width);
        const CGFloat deltaHeight = std::fabs(targetHeight - m_currentSize.height);
        if (deltaWidth < kSizeTolerance && deltaHeight < kSizeTolerance) {
            return;
        }
    }

#if DEBUG_FIXED_RENDER_RESOLUTION
    const NSUInteger width = kDebugRenderWidth;
    const NSUInteger height = kDebugRenderHeight;
#else
    const CGFloat roundedWidth = std::max<CGFloat>(static_cast<CGFloat>(std::round(targetWidth)), 1.0f);
    const CGFloat roundedHeight =
        std::max<CGFloat>(static_cast<CGFloat>(std::round(targetHeight)), 1.0f);
    const NSUInteger width = static_cast<NSUInteger>(roundedWidth);
    const NSUInteger height = static_cast<NSUInteger>(roundedHeight);
#endif

    auto needsTexture = ^BOOL(id<MTLTexture> texture, MTLPixelFormat expectedFormat) {
        if (!texture) {
            return YES;
        }
        if (texture.width != width || texture.height != height) {
            return YES;
        }
        if (texture.pixelFormat != expectedFormat) {
            return YES;
        }
        return NO;
    };

    BOOL needsRebuild = force ? YES : NO;
    if (!needsRebuild) {
        if (needsTexture(m_radianceSumTexture, kRadianceSumFormat) ||
            needsTexture(m_sampleCountTexture, kSampleCountFormat) ||
            needsTexture(m_presentTexture, kPresentFormat) ||
            needsTexture(m_albedoTexture, kAlbedoFormat) ||
            needsTexture(m_normalTexture, kNormalFormat) ||
            needsTexture(m_denoisedTexture, kDenoisedFormat)) {
            needsRebuild = YES;
        }
    }

    if (!needsRebuild) {
        return;
    }

    MTLTextureDescriptor* radianceDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kRadianceSumFormat
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    radianceDescriptor.storageMode = MTLStorageModePrivate;
    radianceDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    m_radianceSumTexture = [m_device newTextureWithDescriptor:radianceDescriptor];

    MTLTextureDescriptor* countDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kSampleCountFormat
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    countDescriptor.storageMode = MTLStorageModePrivate;
    countDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    m_sampleCountTexture = [m_device newTextureWithDescriptor:countDescriptor];

    MTLTextureDescriptor* presentDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kPresentFormat
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    presentDescriptor.storageMode = MTLStorageModePrivate;
    presentDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    m_presentTexture = [m_device newTextureWithDescriptor:presentDescriptor];

    // Create AOV textures for denoising
    MTLTextureDescriptor* albedoDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kAlbedoFormat
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    albedoDescriptor.storageMode = MTLStorageModePrivate;
    albedoDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    m_albedoTexture = [m_device newTextureWithDescriptor:albedoDescriptor];

    MTLTextureDescriptor* normalDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kNormalFormat
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    normalDescriptor.storageMode = MTLStorageModePrivate;
    normalDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    m_normalTexture = [m_device newTextureWithDescriptor:normalDescriptor];

    MTLTextureDescriptor* denoisedDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kDenoisedFormat
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    denoisedDescriptor.storageMode = MTLStorageModePrivate;
    denoisedDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    m_denoisedTexture = [m_device newTextureWithDescriptor:denoisedDescriptor];

    m_currentSize = CGSizeMake(static_cast<CGFloat>(width), static_cast<CGFloat>(height));
    m_frameIndex = 0;
    m_sampleCount = 0;
    m_needsClear = true;
}

void Accumulation::clear(MTLCommandBufferHandle commandBuffer,
                         MTLComputePipelineStateHandle clearPipeline) {
    if (!commandBuffer || !clearPipeline || !m_radianceSumTexture || !m_sampleCountTexture ||
        !m_presentTexture) {
        return;
    }

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (!encoder) {
        return;
    }

    encoder.label = @"Clear Accumulation Kernel";
    [encoder setComputePipelineState:clearPipeline];
    [encoder setTexture:m_radianceSumTexture atIndex:0];
    [encoder setTexture:m_sampleCountTexture atIndex:1];
    [encoder setTexture:m_presentTexture atIndex:2];

    NSUInteger threadWidth = clearPipeline.threadExecutionWidth;
    NSUInteger threadHeight = std::max<NSUInteger>(
        1, clearPipeline.maxTotalThreadsPerThreadgroup / std::max<NSUInteger>(threadWidth, 1));
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadWidth, threadHeight, 1);
    MTLSize threadsPerGrid = MTLSizeMake(m_radianceSumTexture.width,
                                         m_radianceSumTexture.height,
                                         1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];

    m_needsClear = false;
}

void Accumulation::reset(MTLCommandQueueHandle commandQueue,
                         MTLComputePipelineStateHandle clearPipeline,
                         MTKViewHandle view) {
    m_sampleCount = 0;
    m_frameIndex = 0;
    m_needsClear = true;

    // Update size from view if provided (windowed mode)
    // In headless mode, view is nil and size is already set by ensureTextures()
    if (view) {
        // Reuse existing textures when dimensions are unchanged to avoid churn.
        ensureTextures(view.drawableSize);
    }
    (void)commandQueue;
    (void)clearPipeline;
}

float Accumulation::updateDrawableSize(NSWindowHandle window, MTKViewHandle view) {
    if (!view) {
        return 1.0f;
    }

    NSScreen* screen = window ? window.screen : nil;
    if (!screen) {
        screen = [NSScreen mainScreen];
    }

    CGFloat scale = screen ? screen.backingScaleFactor : 1.0;
    scale = std::max<CGFloat>(scale, 1.0);

    if (view.layer) {
        view.layer.contentsScale = scale;
    }

    CGSize bounds = view.bounds.size;
    CGSize drawable = CGSizeMake(std::max<CGFloat>(bounds.width * scale, 1.0f),
                                 std::max<CGFloat>(bounds.height * scale, 1.0f));
    view.drawableSize = drawable;

    ensureTextures(drawable);
    return static_cast<float>(scale);
}

void Accumulation::teardown() {
    m_radianceSumTexture = nil;
    m_sampleCountTexture = nil;
    m_presentTexture = nil;
    m_albedoTexture = nil;
    m_normalTexture = nil;
    m_denoisedTexture = nil;
    m_currentSize = CGSizeZero;
    m_frameIndex = 0;
    m_sampleCount = 0;
    m_needsClear = false;
    m_device = nil;
}

}  // namespace PathTracer
