#import "renderer/Accumulation.h"

#import <AppKit/AppKit.h>
#import <MetalKit/MetalKit.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <iterator>

#ifndef DEBUG_FIXED_RENDER_RESOLUTION
#define DEBUG_FIXED_RENDER_RESOLUTION 0
#endif

#if DEBUG_FIXED_RENDER_RESOLUTION
constexpr uint32_t kDebugRenderWidth = 1280;
constexpr uint32_t kDebugRenderHeight = 720;
#endif

namespace {

constexpr MTLPixelFormat kRadianceSumFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kRadianceMomentSumFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kSampleCountFormat = MTLPixelFormatR32Uint;
constexpr MTLPixelFormat kPresentFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kAlbedoFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kNormalFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kPositionFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kMaterialFeatureFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kMotionVectorFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kReconstructionMetricFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kDenoisedFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kSvgfScratchFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kSvgfMomentsFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kSvgfGuideFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kSvgfColorFormat = MTLPixelFormatRGBA32Float;
constexpr MTLPixelFormat kSvgfMaterialHistoryFormat = MTLPixelFormatRGBA32Float;

bool GraphUsesGroup(const PathTracer::RenderPassGraphCompileResult& compileResult,
                    PathTracer::RenderResourceGroup group) {
    const auto it = std::find_if(
        compileResult.resourcePlans.begin(),
        compileResult.resourcePlans.end(),
        [group](const PathTracer::FrameResourceLifetimePlan& plan) {
            return plan.group == group;
        });
    return it != compileResult.resourcePlans.end() && it->used;
}

}  // namespace

namespace PathTracer {

void Accumulation::initialize(MTLDeviceHandle device) {
    m_device = device;
}

void Accumulation::ensureTextures(CGSize drawableSize, bool force) {
    if (m_hasLastAllocationGraph) {
        ensureTextures(drawableSize, m_lastAllocationGraph, force);
        return;
    }
    RenderPassGraphDesc compatibilityGraph{};
    auto addCompatibilityResource = [&](const char* name, RenderResourceGroup group) {
        FrameResourceDesc resource{};
        resource.name = name;
        resource.group = group;
        compatibilityGraph.resources.push_back(std::move(resource));
    };
    addCompatibilityResource("GBufferResources", RenderResourceGroup::GBufferResources);
    addCompatibilityResource("ReconstructionAOVResources", RenderResourceGroup::ReconstructionAOVResources);
    addCompatibilityResource("ReconstructionHistoryResources", RenderResourceGroup::ReconstructionHistoryResources);
    addCompatibilityResource("DenoiserResources", RenderResourceGroup::DenoiserResources);
    addCompatibilityResource("DenoiserScratchResources", RenderResourceGroup::DenoiserScratchResources);
    addCompatibilityResource("FinalOutputResources", RenderResourceGroup::FinalOutputResources);
    RenderPassDesc pass{};
    pass.name = "Compatibility Accumulation Allocation";
    pass.outputs = {RenderResourceGroup::GBufferResources,
                    RenderResourceGroup::ReconstructionAOVResources,
                    RenderResourceGroup::ReconstructionHistoryResources,
                    RenderResourceGroup::DenoiserResources,
                    RenderResourceGroup::DenoiserScratchResources,
                    RenderResourceGroup::FinalOutputResources};
    pass.executionId = RenderPassExecutionId::NoOp;
    pass.pipelineKey = "compatibility";
    pass.dispatch.kind = DispatchKind::External;
    pass.dispatch.dimensions = "current drawable";
    compatibilityGraph.passes.push_back(std::move(pass));
    ensureTextures(drawableSize, compatibilityGraph, force);
}

void Accumulation::ensureTextures(CGSize drawableSize, const RenderPassGraphDesc& graph, bool force) {
    if (!m_device) {
        return;
    }
    if (drawableSize.width < 1.0 || drawableSize.height < 1.0) {
        return;
    }
    m_lastAllocationGraph = graph;
    m_hasLastAllocationGraph = true;

#if DEBUG_FIXED_RENDER_RESOLUTION
    const CGFloat targetWidth = static_cast<CGFloat>(kDebugRenderWidth);
    const CGFloat targetHeight = static_cast<CGFloat>(kDebugRenderHeight);
#else
    const CGFloat targetWidth = std::max<CGFloat>(drawableSize.width, 1.0f);
    const CGFloat targetHeight = std::max<CGFloat>(drawableSize.height, 1.0f);
#endif

    const RenderPassGraphCompileResult compileResult = CompileRenderPassGraph(graph);
    const bool useGBuffer = GraphUsesGroup(compileResult, RenderResourceGroup::GBufferResources) ||
                            GraphUsesGroup(compileResult, RenderResourceGroup::ReconstructionAOVResources);
    const bool useReconstructionHistory =
        GraphUsesGroup(compileResult, RenderResourceGroup::ReconstructionHistoryResources);
    const bool useDenoiser = GraphUsesGroup(compileResult, RenderResourceGroup::DenoiserResources);
    const bool useDenoiserScratch =
        GraphUsesGroup(compileResult, RenderResourceGroup::DenoiserScratchResources);
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
            needsTexture(m_directRadianceSumTexture, kRadianceSumFormat) ||
            needsTexture(m_indirectRadianceSumTexture, kRadianceSumFormat) ||
            needsTexture(m_radianceMomentSumTexture, kRadianceMomentSumFormat) ||
            needsTexture(m_sampleCountTexture, kSampleCountFormat) ||
            needsTexture(m_presentTexture, kPresentFormat) ||
            needsTexture(m_directPresentTexture, kPresentFormat) ||
            needsTexture(m_indirectPresentTexture, kPresentFormat) ||
            (useGBuffer && (needsTexture(m_albedoTexture, kAlbedoFormat) ||
                            needsTexture(m_normalTexture, kNormalFormat) ||
                            needsTexture(m_positionTexture, kPositionFormat) ||
                            needsTexture(m_materialFeatureTexture, kMaterialFeatureFormat) ||
                            needsTexture(m_motionVectorTexture, kMotionVectorFormat) ||
                            needsTexture(m_reconstructionMetricTexture, kReconstructionMetricFormat))) ||
            (useDenoiser && needsTexture(m_denoisedTexture, kDenoisedFormat)) ||
            (useDenoiserScratch && needsTexture(m_svgfScratchTexture, kSvgfScratchFormat)) ||
            (useReconstructionHistory &&
             (needsTexture(m_svgfMomentsTextureA, kSvgfMomentsFormat) ||
              needsTexture(m_svgfMomentsTextureB, kSvgfMomentsFormat) ||
              needsTexture(m_svgfIndirectMomentsTextureA, kSvgfMomentsFormat) ||
              needsTexture(m_svgfIndirectMomentsTextureB, kSvgfMomentsFormat) ||
              needsTexture(m_svgfGuideTextureA, kSvgfGuideFormat) ||
              needsTexture(m_svgfGuideTextureB, kSvgfGuideFormat) ||
              needsTexture(m_svgfColorTextureA, kSvgfColorFormat) ||
              needsTexture(m_svgfColorTextureB, kSvgfColorFormat) ||
              needsTexture(m_svgfIndirectColorTextureA, kSvgfColorFormat) ||
              needsTexture(m_svgfIndirectColorTextureB, kSvgfColorFormat) ||
              needsTexture(m_svgfDirectComponentTexture, kSvgfColorFormat) ||
              needsTexture(m_svgfIndirectComponentTexture, kSvgfColorFormat) ||
              needsTexture(m_svgfMaterialHistoryTextureA, kSvgfMaterialHistoryFormat) ||
              needsTexture(m_svgfMaterialHistoryTextureB, kSvgfMaterialHistoryFormat)))) {
            needsRebuild = YES;
        }
    }

    if (!needsRebuild) {
        return;
    }

    auto makeTexture = ^MTLTextureHandle(NSString* label,
                                         MTLPixelFormat format,
                                         NSUInteger textureWidth,
                                         NSUInteger textureHeight,
                                         MTLTextureUsage usage) {
        MTLTextureDescriptor* descriptor =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                               width:textureWidth
                                                              height:textureHeight
                                                           mipmapped:NO];
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = usage;
        id<MTLTexture> texture = [m_device newTextureWithDescriptor:descriptor];
        if (texture) {
            texture.label = label;
        }
        return texture;
    };
    auto makeFullTexture = ^MTLTextureHandle(NSString* label, MTLPixelFormat format) {
        return makeTexture(label,
                           format,
                           width,
                           height,
                           MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    };
    auto makePlaceholderTexture = ^MTLTextureHandle(NSString* label, MTLPixelFormat format) {
        return makeTexture(label,
                           format,
                           1u,
                           1u,
                           MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    };

    MTLTextureDescriptor* radianceDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kRadianceSumFormat
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    radianceDescriptor.storageMode = MTLStorageModePrivate;
    radianceDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    m_radianceSumTexture = [m_device newTextureWithDescriptor:radianceDescriptor];
    if (m_radianceSumTexture) {
        m_radianceSumTexture.label = @"FrameGraph.FinalOutput.RadianceSum";
    }
    m_directRadianceSumTexture =
        makeFullTexture(@"FrameGraph.ReconstructionAOV.DirectRadianceSum", kRadianceSumFormat);
    m_indirectRadianceSumTexture =
        makeFullTexture(@"FrameGraph.ReconstructionAOV.IndirectRadianceSum", kRadianceSumFormat);
    m_radianceMomentSumTexture =
        makeFullTexture(@"FrameGraph.ReconstructionAOV.RadianceSecondMomentSum",
                        kRadianceMomentSumFormat);

    MTLTextureDescriptor* countDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kSampleCountFormat
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    countDescriptor.storageMode = MTLStorageModePrivate;
    countDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    m_sampleCountTexture = [m_device newTextureWithDescriptor:countDescriptor];
    if (m_sampleCountTexture) {
        m_sampleCountTexture.label = @"FrameGraph.FinalOutput.SampleCount";
    }

    MTLTextureDescriptor* presentDescriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:kPresentFormat
                                                           width:width
                                                          height:height
                                                       mipmapped:NO];
    presentDescriptor.storageMode = MTLStorageModePrivate;
    presentDescriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    m_presentTexture = [m_device newTextureWithDescriptor:presentDescriptor];
    if (m_presentTexture) {
        m_presentTexture.label = @"FrameGraph.FinalOutput.Present";
    }
    m_directPresentTexture =
        makeFullTexture(@"FrameGraph.ReconstructionAOV.DirectPresent", kPresentFormat);
    m_indirectPresentTexture =
        makeFullTexture(@"FrameGraph.ReconstructionAOV.IndirectPresent", kPresentFormat);

    m_albedoTexture = useGBuffer ? makeFullTexture(@"FrameGraph.GBuffer.Albedo", kAlbedoFormat)
                                 : makePlaceholderTexture(@"FrameGraph.Placeholder.Albedo", kAlbedoFormat);
    m_normalTexture = useGBuffer ? makeFullTexture(@"FrameGraph.GBuffer.Normal", kNormalFormat)
                                 : makePlaceholderTexture(@"FrameGraph.Placeholder.Normal", kNormalFormat);
    m_positionTexture = useGBuffer ? makeFullTexture(@"FrameGraph.GBuffer.Position", kPositionFormat)
                                   : makePlaceholderTexture(@"FrameGraph.Placeholder.Position", kPositionFormat);
    m_materialFeatureTexture =
        useGBuffer ? makeFullTexture(@"FrameGraph.ReconstructionAOV.MaterialFeatures", kMaterialFeatureFormat)
                   : makePlaceholderTexture(@"FrameGraph.Placeholder.MaterialFeatures", kMaterialFeatureFormat);
    m_motionVectorTexture =
        useGBuffer ? makeFullTexture(@"FrameGraph.ReconstructionAOV.MotionVectors", kMotionVectorFormat)
                   : makePlaceholderTexture(@"FrameGraph.Placeholder.MotionVectors", kMotionVectorFormat);
    m_reconstructionMetricTexture =
        useGBuffer
            ? makeFullTexture(@"FrameGraph.ReconstructionAOV.VarianceConfidenceDisocclusion",
                              kReconstructionMetricFormat)
            : makePlaceholderTexture(@"FrameGraph.Placeholder.VarianceConfidenceDisocclusion",
                                     kReconstructionMetricFormat);

    m_denoisedTexture =
        useDenoiser ? makeFullTexture(@"FrameGraph.Denoiser.Output", kDenoisedFormat)
                    : makePlaceholderTexture(@"FrameGraph.Placeholder.DenoiserOutput", kDenoisedFormat);
    m_svgfScratchTexture =
        useDenoiserScratch ? makeFullTexture(@"FrameGraph.DenoiserScratch.SVGF", kSvgfScratchFormat)
                           : makePlaceholderTexture(@"FrameGraph.Placeholder.SVGF.Scratch", kSvgfScratchFormat);
    m_svgfMomentsTextureA =
        useReconstructionHistory ? makeFullTexture(@"FrameGraph.ReconstructionHistory.SVGF.MomentsA", kSvgfMomentsFormat)
                                 : makePlaceholderTexture(@"FrameGraph.Placeholder.SVGF.MomentsA", kSvgfMomentsFormat);
    m_svgfMomentsTextureB =
        useReconstructionHistory ? makeFullTexture(@"FrameGraph.ReconstructionHistory.SVGF.MomentsB", kSvgfMomentsFormat)
                                 : makePlaceholderTexture(@"FrameGraph.Placeholder.SVGF.MomentsB", kSvgfMomentsFormat);
    m_svgfIndirectMomentsTextureA =
        useReconstructionHistory ? makeFullTexture(@"FrameGraph.ReconstructionHistory.SVGF.IndirectMomentsA", kSvgfMomentsFormat)
                                 : makePlaceholderTexture(@"FrameGraph.Placeholder.SVGF.IndirectMomentsA", kSvgfMomentsFormat);
    m_svgfIndirectMomentsTextureB =
        useReconstructionHistory ? makeFullTexture(@"FrameGraph.ReconstructionHistory.SVGF.IndirectMomentsB", kSvgfMomentsFormat)
                                 : makePlaceholderTexture(@"FrameGraph.Placeholder.SVGF.IndirectMomentsB", kSvgfMomentsFormat);
    m_svgfGuideTextureA =
        useReconstructionHistory ? makeFullTexture(@"FrameGraph.ReconstructionHistory.SVGF.GuideA", kSvgfGuideFormat)
                                 : makePlaceholderTexture(@"FrameGraph.Placeholder.SVGF.GuideA", kSvgfGuideFormat);
    m_svgfGuideTextureB =
        useReconstructionHistory ? makeFullTexture(@"FrameGraph.ReconstructionHistory.SVGF.GuideB", kSvgfGuideFormat)
                                 : makePlaceholderTexture(@"FrameGraph.Placeholder.SVGF.GuideB", kSvgfGuideFormat);
    m_svgfColorTextureA =
        useReconstructionHistory ? makeFullTexture(@"FrameGraph.ReconstructionHistory.SVGF.ColorA", kSvgfColorFormat)
                                 : makePlaceholderTexture(@"FrameGraph.Placeholder.SVGF.ColorA", kSvgfColorFormat);
    m_svgfColorTextureB =
        useReconstructionHistory ? makeFullTexture(@"FrameGraph.ReconstructionHistory.SVGF.ColorB", kSvgfColorFormat)
                                 : makePlaceholderTexture(@"FrameGraph.Placeholder.SVGF.ColorB", kSvgfColorFormat);
    m_svgfIndirectColorTextureA =
        useReconstructionHistory ? makeFullTexture(@"FrameGraph.ReconstructionHistory.SVGF.IndirectColorA", kSvgfColorFormat)
                                 : makePlaceholderTexture(@"FrameGraph.Placeholder.SVGF.IndirectColorA", kSvgfColorFormat);
    m_svgfIndirectColorTextureB =
        useReconstructionHistory ? makeFullTexture(@"FrameGraph.ReconstructionHistory.SVGF.IndirectColorB", kSvgfColorFormat)
                                 : makePlaceholderTexture(@"FrameGraph.Placeholder.SVGF.IndirectColorB", kSvgfColorFormat);
    m_svgfDirectComponentTexture =
        useDenoiser ? makeFullTexture(@"FrameGraph.Denoiser.SVGF.DirectComponent", kSvgfColorFormat)
                    : makePlaceholderTexture(@"FrameGraph.Placeholder.SVGF.DirectComponent", kSvgfColorFormat);
    m_svgfIndirectComponentTexture =
        useDenoiser ? makeFullTexture(@"FrameGraph.Denoiser.SVGF.IndirectComponent", kSvgfColorFormat)
                    : makePlaceholderTexture(@"FrameGraph.Placeholder.SVGF.IndirectComponent", kSvgfColorFormat);
    m_svgfMaterialHistoryTextureA =
        useReconstructionHistory
            ? makeFullTexture(@"FrameGraph.ReconstructionHistory.SVGF.MaterialFeaturesA",
                              kSvgfMaterialHistoryFormat)
            : makePlaceholderTexture(@"FrameGraph.Placeholder.SVGF.MaterialFeaturesA",
                                     kSvgfMaterialHistoryFormat);
    m_svgfMaterialHistoryTextureB =
        useReconstructionHistory
            ? makeFullTexture(@"FrameGraph.ReconstructionHistory.SVGF.MaterialFeaturesB",
                              kSvgfMaterialHistoryFormat)
            : makePlaceholderTexture(@"FrameGraph.Placeholder.SVGF.MaterialFeaturesB",
                                     kSvgfMaterialHistoryFormat);

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
    [encoder setTexture:m_svgfMomentsTextureA atIndex:3];
    [encoder setTexture:m_svgfMomentsTextureB atIndex:4];
    [encoder setTexture:m_positionTexture atIndex:5];
    [encoder setTexture:m_svgfGuideTextureA atIndex:6];
    [encoder setTexture:m_svgfGuideTextureB atIndex:7];
    [encoder setTexture:m_svgfColorTextureA atIndex:8];
    [encoder setTexture:m_svgfColorTextureB atIndex:9];
    [encoder setTexture:m_materialFeatureTexture atIndex:10];
    [encoder setTexture:m_motionVectorTexture atIndex:11];
    [encoder setTexture:m_svgfMaterialHistoryTextureA atIndex:12];
    [encoder setTexture:m_svgfMaterialHistoryTextureB atIndex:13];
    [encoder setTexture:m_directRadianceSumTexture atIndex:14];
    [encoder setTexture:m_indirectRadianceSumTexture atIndex:15];
    [encoder setTexture:m_radianceMomentSumTexture atIndex:16];
    [encoder setTexture:m_directPresentTexture atIndex:17];
    [encoder setTexture:m_indirectPresentTexture atIndex:18];
    [encoder setTexture:m_reconstructionMetricTexture atIndex:19];
    [encoder setTexture:m_svgfIndirectMomentsTextureA atIndex:20];
    [encoder setTexture:m_svgfIndirectMomentsTextureB atIndex:21];
    [encoder setTexture:m_svgfIndirectColorTextureA atIndex:22];
    [encoder setTexture:m_svgfIndirectColorTextureB atIndex:23];
    [encoder setTexture:m_svgfDirectComponentTexture atIndex:24];
    [encoder setTexture:m_svgfIndirectComponentTexture atIndex:25];

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
    m_directRadianceSumTexture = nil;
    m_indirectRadianceSumTexture = nil;
    m_radianceMomentSumTexture = nil;
    m_sampleCountTexture = nil;
    m_presentTexture = nil;
    m_directPresentTexture = nil;
    m_indirectPresentTexture = nil;
    m_albedoTexture = nil;
    m_normalTexture = nil;
    m_positionTexture = nil;
    m_materialFeatureTexture = nil;
    m_motionVectorTexture = nil;
    m_reconstructionMetricTexture = nil;
    m_denoisedTexture = nil;
    m_svgfScratchTexture = nil;
    m_svgfMomentsTextureA = nil;
    m_svgfMomentsTextureB = nil;
    m_svgfIndirectMomentsTextureA = nil;
    m_svgfIndirectMomentsTextureB = nil;
    m_svgfGuideTextureA = nil;
    m_svgfGuideTextureB = nil;
    m_svgfColorTextureA = nil;
    m_svgfColorTextureB = nil;
    m_svgfIndirectColorTextureA = nil;
    m_svgfIndirectColorTextureB = nil;
    m_svgfDirectComponentTexture = nil;
    m_svgfIndirectComponentTexture = nil;
    m_svgfMaterialHistoryTextureA = nil;
    m_svgfMaterialHistoryTextureB = nil;
    m_currentSize = CGSizeZero;
    m_frameIndex = 0;
    m_sampleCount = 0;
    m_needsClear = false;
    m_device = nil;
    m_lastAllocationGraph = {};
    m_hasLastAllocationGraph = false;
}

namespace {

// The textures the integrators read and add to on every sample, and the present textures the next sample reads.
// The denoised and SVGF textures belong to denoisers that run after the last sample or that a checkpoint refuses.
struct ResumableTexture {
    const char* name;
    id<MTLTexture> texture;
};

uint32_t BytesPerPixelForResume(MTLPixelFormat format) {
    switch (format) {
        case MTLPixelFormatRGBA32Float: return 16u;
        case MTLPixelFormatR32Uint: return 4u;
        default: return 0u;
    }
}

constexpr uint32_t kAccumulationStateMagic = 0x53434150u;  // "PACS"
constexpr uint32_t kAccumulationStateVersion = 1u;

}  // namespace

bool Accumulation::exportState(MTLCommandQueueHandle commandQueue, AccumulationState& state, std::string& error) const {
    const ResumableTexture textures[] = {
        {"radiance_sum", m_radianceSumTexture},
        {"direct_radiance_sum", m_directRadianceSumTexture},
        {"indirect_radiance_sum", m_indirectRadianceSumTexture},
        {"radiance_moment_sum", m_radianceMomentSumTexture},
        {"sample_count", m_sampleCountTexture},
        {"albedo", m_albedoTexture},
        {"normal", m_normalTexture},
        {"position", m_positionTexture},
        {"material_feature", m_materialFeatureTexture},
        {"motion_vector", m_motionVectorTexture},
        {"reconstruction_metric", m_reconstructionMetricTexture},
        {"present", m_presentTexture},
        {"direct_present", m_directPresentTexture},
        {"indirect_present", m_indirectPresentTexture},
    };
    if (!m_device || !commandQueue) {
        error = "Accumulation state export: no Metal device or command queue";
        return false;
    }
    state = AccumulationState{};
    state.frameIndex = m_frameIndex;
    state.sampleCount = m_sampleCount;
    std::vector<id<MTLBuffer>> buffers;
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    commandBuffer.label = @"Accumulation State Export";
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    for (const ResumableTexture& entry : textures) {
        if (!entry.texture) {
            error = std::string("Accumulation state export: texture ") + entry.name + " does not exist";
            [blit endEncoding];
            return false;
        }
        const uint32_t bytesPerPixel = BytesPerPixelForResume(entry.texture.pixelFormat);
        if (bytesPerPixel == 0u) {
            error = std::string("Accumulation state export: texture ") + entry.name + " has an unsupported format";
            [blit endEncoding];
            return false;
        }
        AccumulationTextureState textureState;
        textureState.name = entry.name;
        textureState.width = static_cast<uint32_t>(entry.texture.width);
        textureState.height = static_cast<uint32_t>(entry.texture.height);
        textureState.bytesPerPixel = bytesPerPixel;
        const NSUInteger bytesPerRow = static_cast<NSUInteger>(textureState.width) * bytesPerPixel;
        const NSUInteger length = bytesPerRow * textureState.height;
        id<MTLBuffer> buffer = [m_device newBufferWithLength:length options:MTLResourceStorageModeShared];
        if (!buffer) {
            error = std::string("Accumulation state export: cannot allocate a buffer for ") + entry.name;
            [blit endEncoding];
            return false;
        }
        [blit copyFromTexture:entry.texture
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(textureState.width, textureState.height, 1)
                        toBuffer:buffer
               destinationOffset:0
          destinationBytesPerRow:bytesPerRow
        destinationBytesPerImage:length];
        buffers.push_back(buffer);
        state.textures.push_back(std::move(textureState));
    }
    [blit endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        error = "Accumulation state export: the copy command buffer failed";
        return false;
    }
    for (size_t i = 0; i < buffers.size(); ++i) {
        const uint8_t* source = static_cast<const uint8_t*>(buffers[i].contents);
        state.textures[i].bytes.assign(source, source + buffers[i].length);
    }
    return true;
}

bool Accumulation::importState(MTLCommandQueueHandle commandQueue, const AccumulationState& state, std::string& error) {
    const ResumableTexture textures[] = {
        {"radiance_sum", m_radianceSumTexture},
        {"direct_radiance_sum", m_directRadianceSumTexture},
        {"indirect_radiance_sum", m_indirectRadianceSumTexture},
        {"radiance_moment_sum", m_radianceMomentSumTexture},
        {"sample_count", m_sampleCountTexture},
        {"albedo", m_albedoTexture},
        {"normal", m_normalTexture},
        {"position", m_positionTexture},
        {"material_feature", m_materialFeatureTexture},
        {"motion_vector", m_motionVectorTexture},
        {"reconstruction_metric", m_reconstructionMetricTexture},
        {"present", m_presentTexture},
        {"direct_present", m_directPresentTexture},
        {"indirect_present", m_indirectPresentTexture},
    };
    if (!m_device || !commandQueue) {
        error = "Accumulation state import: no Metal device or command queue";
        return false;
    }
    if (state.textures.size() != std::size(textures)) {
        error = "Accumulation state import: the state has " + std::to_string(state.textures.size()) +
                " textures, the renderer " + std::to_string(std::size(textures));
        return false;
    }
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    commandBuffer.label = @"Accumulation State Import";
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    for (size_t i = 0; i < std::size(textures); ++i) {
        const ResumableTexture& entry = textures[i];
        const AccumulationTextureState& textureState = state.textures[i];
        const uint32_t bytesPerPixel = entry.texture ? BytesPerPixelForResume(entry.texture.pixelFormat) : 0u;
        const size_t expectedBytes =
            static_cast<size_t>(textureState.width) * textureState.height * textureState.bytesPerPixel;
        if (!entry.texture || textureState.name != entry.name ||
            textureState.width != entry.texture.width || textureState.height != entry.texture.height ||
            textureState.bytesPerPixel != bytesPerPixel || bytesPerPixel == 0u ||
            textureState.bytes.size() != expectedBytes) {
            error = std::string("Accumulation state import: texture ") + entry.name +
                    " does not match the state's " + textureState.name + " (" +
                    std::to_string(textureState.width) + "x" + std::to_string(textureState.height) + ", " +
                    std::to_string(textureState.bytesPerPixel) + " bytes per pixel)";
            [blit endEncoding];
            return false;
        }
        id<MTLBuffer> buffer = [m_device newBufferWithBytes:textureState.bytes.data()
                                                     length:textureState.bytes.size()
                                                    options:MTLResourceStorageModeShared];
        if (!buffer) {
            error = std::string("Accumulation state import: cannot allocate a buffer for ") + entry.name;
            [blit endEncoding];
            return false;
        }
        const NSUInteger bytesPerRow = static_cast<NSUInteger>(textureState.width) * bytesPerPixel;
        [blit copyFromBuffer:buffer
                   sourceOffset:0
              sourceBytesPerRow:bytesPerRow
            sourceBytesPerImage:bytesPerRow * textureState.height
                     sourceSize:MTLSizeMake(textureState.width, textureState.height, 1)
                      toTexture:entry.texture
               destinationSlice:0
               destinationLevel:0
              destinationOrigin:MTLOriginMake(0, 0, 0)];
    }
    [blit endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        error = "Accumulation state import: the copy command buffer failed";
        return false;
    }
    m_frameIndex = state.frameIndex;
    m_sampleCount = state.sampleCount;
    m_needsClear = false;
    return true;
}

bool WriteAccumulationStateFile(const std::string& path, const AccumulationState& state, std::string& error) {
    const std::string temporaryPath = path + ".tmp";
    FILE* file = std::fopen(temporaryPath.c_str(), "wb");
    if (!file) {
        error = "Cannot write accumulation state " + temporaryPath;
        return false;
    }
    bool ok = true;
    auto put32 = [&](uint32_t value) { ok = ok && std::fwrite(&value, sizeof(value), 1, file) == 1; };
    put32(kAccumulationStateMagic);
    put32(kAccumulationStateVersion);
    put32(state.frameIndex);
    put32(state.sampleCount);
    put32(static_cast<uint32_t>(state.textures.size()));
    for (const AccumulationTextureState& texture : state.textures) {
        put32(static_cast<uint32_t>(texture.name.size()));
        ok = ok && std::fwrite(texture.name.data(), 1, texture.name.size(), file) == texture.name.size();
        put32(texture.width);
        put32(texture.height);
        put32(texture.bytesPerPixel);
        ok = ok && std::fwrite(texture.bytes.data(), 1, texture.bytes.size(), file) == texture.bytes.size();
    }
    ok = (std::fclose(file) == 0) && ok;
    if (!ok || std::rename(temporaryPath.c_str(), path.c_str()) != 0) {
        std::remove(temporaryPath.c_str());
        error = "Cannot write accumulation state " + path;
        return false;
    }
    return true;
}

bool ReadAccumulationStateFile(const std::string& path, AccumulationState& state, std::string& error) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (!file) {
        error = "Cannot read accumulation state " + path;
        return false;
    }
    bool ok = true;
    auto get32 = [&]() -> uint32_t {
        uint32_t value = 0u;
        ok = ok && std::fread(&value, sizeof(value), 1, file) == 1;
        return value;
    };
    state = AccumulationState{};
    const uint32_t magic = get32();
    const uint32_t version = get32();
    if (!ok || magic != kAccumulationStateMagic || version != kAccumulationStateVersion) {
        std::fclose(file);
        error = "Not an accumulation state file (version " + std::to_string(kAccumulationStateVersion) + "): " + path;
        return false;
    }
    state.frameIndex = get32();
    state.sampleCount = get32();
    const uint32_t textureCount = get32();
    for (uint32_t i = 0; ok && i < textureCount && i < 64u; ++i) {
        AccumulationTextureState texture;
        const uint32_t nameLength = get32();
        if (!ok || nameLength > 256u) {
            ok = false;
            break;
        }
        texture.name.resize(nameLength);
        ok = ok && std::fread(texture.name.data(), 1, nameLength, file) == nameLength;
        texture.width = get32();
        texture.height = get32();
        texture.bytesPerPixel = get32();
        const uint64_t byteCount = static_cast<uint64_t>(texture.width) * texture.height * texture.bytesPerPixel;
        if (!ok || byteCount > (1ull << 32)) {
            ok = false;
            break;
        }
        texture.bytes.resize(static_cast<size_t>(byteCount));
        ok = ok && std::fread(texture.bytes.data(), 1, texture.bytes.size(), file) == texture.bytes.size();
        state.textures.push_back(std::move(texture));
    }
    std::fclose(file);
    if (!ok || state.textures.size() != textureCount) {
        error = "Truncated or corrupt accumulation state: " + path;
        return false;
    }
    return true;
}

}  // namespace PathTracer
