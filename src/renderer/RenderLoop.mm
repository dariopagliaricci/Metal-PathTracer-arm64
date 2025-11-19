#import "renderer/RenderLoop.h"
#import "renderer/MetalContext.h"
#import "renderer/Accumulation.h"
#import "renderer/Pipelines.h"
#import "renderer/SceneResources.h"
#import "renderer/UIOverlay.h"
#import "renderer/UniformBuilder.h"
#import "renderer/DenoiserContext.h"

#include "MetalShaderTypes.h"
#include <algorithm>
#include <cstring>

using PathTracerShaderTypes::DisplayUniforms;
using PathTracerShaderTypes::PathtraceUniforms;
using PathTracerShaderTypes::PathtraceStats;
using PathTracerShaderTypes::PathtraceDebugBuffer;

namespace PathTracer {

bool RenderLoop::initialize(const MetalContext& context, DenoiserContext* denoiser) {
    m_device = context.device();
    m_denoiser = denoiser;
    if (!m_device) {
        return false;
    }

    // Create uniform buffers
    m_uniformBuffer = [m_device newBufferWithLength:sizeof(PathtraceUniforms)
                                            options:MTLResourceStorageModeShared];
    if (!m_uniformBuffer) {
        NSLog(@"Failed to create uniform buffer");
        return false;
    }

    m_displayUniformBuffer = [m_device newBufferWithLength:sizeof(DisplayUniforms)
                                                   options:MTLResourceStorageModeShared];
    if (!m_displayUniformBuffer) {
        NSLog(@"Failed to create display uniform buffer");
        return false;
    }

    m_statsBuffer = [m_device newBufferWithLength:sizeof(PathtraceStats)
                                          options:MTLResourceStorageModeShared];
    if (!m_statsBuffer) {
        NSLog(@"Failed to create stats buffer");
        return false;
    }

    m_debugBuffer = [m_device newBufferWithLength:sizeof(PathtraceDebugBuffer)
                                          options:MTLResourceStorageModeShared];
    if (!m_debugBuffer) {
        NSLog(@"Failed to create path debug buffer");
        return false;
    }

    return true;
}

uint32_t RenderLoop::encodeIntegration(MTLCommandBufferHandle commandBuffer,
                                       Accumulation& accumulation,
                                       const Pipelines& pipelines,
                                       const SceneResources& scene,
                                       const RenderSettings& settings) {
    
    // Throttle samples per frame when falling back to software to keep UI responsive
    const uint32_t requestedSamples = std::max<uint32_t>(1u, settings.samplesPerFrame);
    uint32_t samplesThisFrame = requestedSamples;

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"Path Tracer Integrate";

    const auto& provider = scene.intersectionProvider();
    bool hardwareMode =
        provider.mode == PathTracerShaderTypes::IntersectionMode::HardwareRayTracing;
    bool hasHardwarePipeline = (pipelines.integrateHardware() != nil);
    bool hasTlas = provider.hardware.tlas != nil;
    bool hasMeshBuffers = scene.meshInfoBuffer() != nil && scene.triangleBuffer() != nil;
    bool encoderSupportsRt =
        [encoder respondsToSelector:@selector(setAccelerationStructure:atBufferIndex:)];
    bool useHardware = hardwareMode && hasHardwarePipeline && hasTlas && hasMeshBuffers && encoderSupportsRt;

    if (settings.enableSoftwareRayTracing) {
        useHardware = false;
    }

    if (!useHardware) {
        samplesThisFrame = 1u;
    }

    id<MTLComputePipelineState> selectedPipeline = useHardware
        ? pipelines.integrateHardware()
        : pipelines.integrate();
    if (!selectedPipeline) {
        selectedPipeline = pipelines.integrate();
        useHardware = false;
    }
    [encoder setComputePipelineState:selectedPipeline];

    NSUInteger threadWidth = selectedPipeline.threadExecutionWidth;
    NSUInteger threadHeight = std::max<NSUInteger>(
        1, selectedPipeline.maxTotalThreadsPerThreadgroup / std::max<NSUInteger>(threadWidth, 1));
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadWidth, threadHeight, 1);

    id<MTLTexture> environmentTexture = scene.environmentTexture();
    [encoder setTexture:environmentTexture atIndex:2];

    if (useHardware) {
        if (encoderSupportsRt) {
            [encoder setAccelerationStructure:provider.hardware.tlas atBufferIndex:1];
        }

        if ([encoder respondsToSelector:@selector(useResource:usage:)]) {
            if (provider.hardware.tlas) {
                [encoder useResource:provider.hardware.tlas usage:MTLResourceUsageRead];
            }
            if (provider.hardware.blas) {
                [encoder useResource:provider.hardware.blas usage:MTLResourceUsageRead];
            }
            if (!provider.hardware.blasHandles.empty()) {
                for (auto h : provider.hardware.blasHandles) {
                    id<MTLAccelerationStructure> blasObj = (id<MTLAccelerationStructure>)h;
                    if (blasObj) {
                        [encoder useResource:blasObj usage:MTLResourceUsageRead];
                    }
                }
            }
            if (provider.hardware.instanceBuffer) {
                [encoder useResource:provider.hardware.instanceBuffer usage:MTLResourceUsageRead];
            }
            if (provider.hardware.instanceUserIDBuffer) {
                [encoder useResource:provider.hardware.instanceUserIDBuffer usage:MTLResourceUsageRead];
            }
        }

        if (scene.meshInfoBuffer()) {
            [encoder setBuffer:scene.meshInfoBuffer() offset:0 atIndex:2];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:2];
        }
        if (scene.triangleBuffer()) {
            [encoder setBuffer:scene.triangleBuffer() offset:0 atIndex:3];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:3];
        }
        if (provider.software.nodes) {
            [encoder setBuffer:provider.software.nodes offset:0 atIndex:4];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:4];
        }
        if (provider.software.primitiveIndices) {
            [encoder setBuffer:provider.software.primitiveIndices offset:0 atIndex:5];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:5];
        }
        if (scene.sphereBuffer()) {
            [encoder setBuffer:scene.sphereBuffer() offset:0 atIndex:6];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:6];
        }
        if (scene.materialBuffer()) {
            [encoder setBuffer:scene.materialBuffer() offset:0 atIndex:7];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:7];
        }
        if (m_statsBuffer) {
            [encoder setBuffer:m_statsBuffer offset:0 atIndex:8];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:8];
        }
        if (scene.rectangleBuffer()) {
            [encoder setBuffer:scene.rectangleBuffer() offset:0 atIndex:9];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:9];
        }
        if (scene.environmentConditionalAliasBuffer()) {
            [encoder setBuffer:scene.environmentConditionalAliasBuffer() offset:0 atIndex:10];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:10];
        }
        if (scene.environmentMarginalAliasBuffer()) {
            [encoder setBuffer:scene.environmentMarginalAliasBuffer() offset:0 atIndex:11];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:11];
        }
        if (scene.environmentPdfBuffer()) {
            [encoder setBuffer:scene.environmentPdfBuffer() offset:0 atIndex:12];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:12];
        }
        if (provider.hardware.instanceUserIDBuffer) {
            [encoder setBuffer:provider.hardware.instanceUserIDBuffer offset:0 atIndex:13];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:13];
        }
        if (scene.meshVertexBuffer()) {
            [encoder setBuffer:scene.meshVertexBuffer() offset:0 atIndex:14];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:14];
        }
        if (scene.meshIndexBuffer()) {
            [encoder setBuffer:scene.meshIndexBuffer() offset:0 atIndex:15];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:15];
        }
        if (m_debugBuffer) {
            [encoder setBuffer:m_debugBuffer offset:0 atIndex:16];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:16];
        }
    } else {
        if (encoderSupportsRt) {
            [encoder setAccelerationStructure:nil atBufferIndex:1];
        }
        [encoder setBuffer:nil offset:0 atIndex:2];
        [encoder setBuffer:nil offset:0 atIndex:3];

        if (scene.sphereBuffer()) {
            [encoder setBuffer:scene.sphereBuffer() offset:0 atIndex:3];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:3];
        }
        if (scene.materialBuffer()) {
            [encoder setBuffer:scene.materialBuffer() offset:0 atIndex:4];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:4];
        }
        if (m_statsBuffer) {
            [encoder setBuffer:m_statsBuffer offset:0 atIndex:5];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:5];
        }
        if (scene.rectangleBuffer()) {
            [encoder setBuffer:scene.rectangleBuffer() offset:0 atIndex:6];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:6];
        }
        if (scene.environmentConditionalAliasBuffer()) {
            [encoder setBuffer:scene.environmentConditionalAliasBuffer() offset:0 atIndex:7];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:7];
        }
        if (scene.environmentMarginalAliasBuffer()) {
            [encoder setBuffer:scene.environmentMarginalAliasBuffer() offset:0 atIndex:8];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:8];
        }
        if (scene.environmentPdfBuffer()) {
            [encoder setBuffer:scene.environmentPdfBuffer() offset:0 atIndex:9];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:9];
        }
        if (scene.triangleBuffer()) {
            [encoder setBuffer:scene.triangleBuffer() offset:0 atIndex:10];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:10];
        }
        [encoder setBuffer:nil offset:0 atIndex:13];

        if (provider.software.nodes && provider.software.primitiveIndices) {
            [encoder setBuffer:provider.software.nodes offset:0 atIndex:1];
            [encoder setBuffer:provider.software.primitiveIndices offset:0 atIndex:2];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:1];
            [encoder setBuffer:nil offset:0 atIndex:2];
        }

        // Bind software TLAS/BLAS buffers if available
        if (provider.software.tlasNodes) {
            [encoder setBuffer:provider.software.tlasNodes offset:0 atIndex:11];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:11];
        }
        if (provider.software.tlasPrimitiveIndices) {
            [encoder setBuffer:provider.software.tlasPrimitiveIndices offset:0 atIndex:12];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:12];
        }
        if (provider.software.blasNodes) {
            [encoder setBuffer:provider.software.blasNodes offset:0 atIndex:13];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:13];
        }
        if (provider.software.blasPrimitiveIndices) {
            [encoder setBuffer:provider.software.blasPrimitiveIndices offset:0 atIndex:14];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:14];
        }
        if (provider.software.instanceInfoBuffer) {
            [encoder setBuffer:provider.software.instanceInfoBuffer offset:0 atIndex:15];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:15];
        }
        if (scene.meshInfoBuffer()) {
            [encoder setBuffer:scene.meshInfoBuffer() offset:0 atIndex:16];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:16];
        }
        if (scene.meshVertexBuffer()) {
            [encoder setBuffer:scene.meshVertexBuffer() offset:0 atIndex:17];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:17];
        }
        if (scene.meshIndexBuffer()) {
            [encoder setBuffer:scene.meshIndexBuffer() offset:0 atIndex:18];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:18];
        }
        if (m_debugBuffer) {
            [encoder setBuffer:m_debugBuffer offset:0 atIndex:19];
        } else {
            [encoder setBuffer:nil offset:0 atIndex:19];
        }
    }

    // Dispatch multiple samples
    for (uint32_t sample = 0; sample < samplesThisFrame; ++sample) {
        PathtraceUniforms uniforms = UniformBuilder::buildPathtraceUniforms(
            settings, accumulation, scene, accumulation.currentSize());
        PathTracerShaderTypes::IntersectionMode modeOverride =
            useHardware ? PathTracerShaderTypes::IntersectionMode::HardwareRayTracing
                        : PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
        uniforms.intersectionMode = static_cast<uint32_t>(modeOverride);
        memcpy([m_uniformBuffer contents], &uniforms, sizeof(PathtraceUniforms));

        [encoder setTexture:accumulation.radianceSum() atIndex:0];
        [encoder setTexture:accumulation.sampleCountTexture() atIndex:1];
        [encoder setTexture:accumulation.albedoBuffer() atIndex:3];
        [encoder setTexture:accumulation.normalBuffer() atIndex:4];
        [encoder setBuffer:m_uniformBuffer offset:0 atIndex:0];

        MTLSize threadsPerGrid = MTLSizeMake(uniforms.width, uniforms.height, 1);
        [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

        accumulation.incrementFrame();
    }
    
    [encoder endEncoding];

    return samplesThisFrame;
}

void RenderLoop::encodeDenoising(MTLCommandBufferHandle commandBuffer,
                                 Accumulation& accumulation,
                                 const RenderSettings& settings,
                                 uint32_t frameIndex) {
    (void)commandBuffer;
    // Denoising is only performed if:
    // 1. Denoiser context is available (not nullptr)
    // 2. Denoising is enabled in settings
    // 3. Denoiser is ready for use
    // 4. Current frame index matches denoising frequency
    if (!m_denoiser || !settings.denoiseEnabled || !m_denoiser->isReady()) {
        return;
    }

    // Progressive denoising: only denoise every N frames
    uint32_t frequency = std::max(1u, settings.denoiseFrequency);
    if ((frameIndex % frequency) != 0) {
        return;
    }

    // Get the accumulated noisy image from presentation buffer
    MTLTextureHandle colorInput = accumulation.present();
    if (!colorInput) {
        NSLog(@"[Denoise] No presentation buffer available");
        return;
    }

    // Get optional auxiliary buffers (albedo and normal)
    MTLTextureHandle albedoInput = settings.denoiseUseAlbedo ? accumulation.albedoBuffer() : nullptr;
    MTLTextureHandle normalInput = settings.denoiseUseNormal ? accumulation.normalBuffer() : nullptr;

    // Output buffer for denoised result
    MTLTextureHandle colorOutput = accumulation.denoisedBuffer();
    if (!colorOutput) {
        NSLog(@"[Denoise] No denoised output buffer available");
        return;
    }

    // Determine filter type
    DenoiserContext::FilterType filterType = (settings.denoiseFilterType == 1)
        ? DenoiserContext::FilterType::RTLightmap
        : DenoiserContext::FilterType::RT;

    // Execute denoising
    if (!m_denoiser->denoise(colorInput, albedoInput, normalInput, colorOutput, filterType)) {
        NSLog(@"[Denoise] Denoising failed: %s", m_denoiser->lastError().c_str());
        return;
    }

    // After successful denoising, we would normally swap the buffers so that
    // denoisedBuffer becomes the new presentation buffer. However, this requires
    // careful synchronization and state management. For now, the denoised output
    // is available in denoisedBuffer() for the next frame's display.
}

void RenderLoop::encodePresentation(MTLCommandBufferHandle commandBuffer,
                                    const Accumulation& accumulation,
                                    const Pipelines& pipelines) {
    
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    encoder.label = @"Path Tracer Present";
    [encoder setComputePipelineState:pipelines.present()];
    [encoder setTexture:accumulation.radianceSum() atIndex:0];
    [encoder setTexture:accumulation.sampleCountTexture() atIndex:1];
    [encoder setTexture:accumulation.present() atIndex:2];

    NSUInteger threadWidth = pipelines.present().threadExecutionWidth;
    NSUInteger threadHeight = std::max<NSUInteger>(
        1, pipelines.present().maxTotalThreadsPerThreadgroup / std::max<NSUInteger>(threadWidth, 1));
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadWidth, threadHeight, 1);
    MTLSize threadsPerGrid = MTLSizeMake(accumulation.present().width,
                                        accumulation.present().height, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
    [encoder endEncoding];
}

void RenderLoop::encodeDisplay(MTLRenderCommandEncoderHandle renderEncoder,
                               const Accumulation& accumulation,
                               const Pipelines& pipelines,
                               const RenderSettings& settings) {
    
    // Update display uniforms
    DisplayUniforms displayUniforms = UniformBuilder::buildDisplayUniforms(settings);
    if (m_displayUniformBuffer) {
        memcpy([m_displayUniformBuffer contents], &displayUniforms, sizeof(DisplayUniforms));
    }

    renderEncoder.label = @"Display Quad";
    [renderEncoder setRenderPipelineState:pipelines.display()];
    if (m_displayUniformBuffer) {
        [renderEncoder setFragmentBuffer:m_displayUniformBuffer offset:0 atIndex:0];
    }
    [renderEncoder setFragmentTexture:accumulation.present() atIndex:0];
    [renderEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
}

FrameResult RenderLoop::encodeFrame(MTLCommandBufferHandle commandBuffer,
                                    MTLRenderPassDescriptorHandle renderPassDescriptor,
                                    MTLCAMetalDrawableHandle drawable,
                                    const MetalContext& context,
                                    Accumulation& accumulation,
                                    const Pipelines& pipelines,
                                    SceneResources& scene,
                                    UIOverlay& overlay,
                                    const RenderSettings& settings) {

    FrameResult result{};
    (void)context;

    // Clear stats buffer
    if (m_statsBuffer) {
        memset([m_statsBuffer contents], 0, m_statsBuffer.length);
    }
    if (m_debugBuffer) {
        auto* debugData =
            reinterpret_cast<PathtraceDebugBuffer*>([m_debugBuffer contents]);
        if (debugData) {
            debugData->writeIndex = 0;
            uint32_t allowed = 0;
            if (settings.enablePathDebug && settings.debugMaxEntries > 0) {
                allowed = std::min(settings.debugMaxEntries,
                                   PathTracerShaderTypes::kPathtraceDebugMaxEntries);
                allowed = std::max<uint32_t>(1u, allowed);
            }
            debugData->maxEntries = allowed;
        }
    }

    // Clear accumulation if needed
    if (accumulation.needsClear() && pipelines.clear()) {
        accumulation.clear(commandBuffer, pipelines.clear());
        m_frameCounter = 0;  // Reset frame counter on accumulation clear
    }

    // Rebuild scene if dirty
    if (scene.isDirty()) {
        scene.rebuildAccelerationStructures();
    }

    // Encode path tracing integration
    const uint32_t samplesDispatched =
        encodeIntegration(commandBuffer, accumulation, pipelines, scene, settings);

    // Encode presentation (averaging)
    encodePresentation(commandBuffer, accumulation, pipelines);

    // Encode denoising (optional OIDN post-processing with progressive frequency)
    encodeDenoising(commandBuffer, accumulation, settings, m_frameCounter);

    // Encode display pass
    if (renderPassDescriptor && drawable && pipelines.display()) {
        id<MTLRenderCommandEncoder> renderEncoder =
            [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
        encodeDisplay(renderEncoder, accumulation, pipelines, settings);
        overlay.render(commandBuffer, renderEncoder);
        [renderEncoder endEncoding];
    }

    // Increment frame counter for next frame
    m_frameCounter++;

    result.samplesDispatched = samplesDispatched;
    return result;
}

void RenderLoop::shutdown() {
    m_statsBuffer = nullptr;
    m_debugBuffer = nullptr;
    m_displayUniformBuffer = nullptr;
    m_uniformBuffer = nullptr;
    m_device = nullptr;
}

}  // namespace PathTracer
