#import "renderer/Pipelines.h"
#import "renderer/MetalContext.h"
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#if __has_include(<objc/message.h>)
#import <objc/message.h>
#endif
#include <string>
#include <utility>

namespace {

bool EnableRayTracingIfAvailable(MTLComputePipelineDescriptor* descriptor) {
#if __has_include(<objc/message.h>)
    if (!descriptor) {
        return false;
    }
    if ([descriptor respondsToSelector:@selector(setSupportRayTracing:)]) {
        auto setter = reinterpret_cast<void (*)(id, SEL, BOOL)>(objc_msgSend);
        setter(descriptor, @selector(setSupportRayTracing:), YES);
        return true;
    }
    return false;
#else
    (void)descriptor;
    return false;
#endif
}

NSString* FindShaderPath(NSString* filename) {
    NSFileManager* fileManager = [NSFileManager defaultManager];
    NSBundle* bundle = [NSBundle mainBundle];

    // Look inside the app bundle (Resources/shaders by CMake).
    NSString* bundlePath = [bundle pathForResource:[filename stringByDeletingPathExtension]
                                            ofType:[filename pathExtension]
                                       inDirectory:@"shaders"];
    if (bundlePath && [fileManager fileExistsAtPath:bundlePath]) {
        return bundlePath;
    }

    // Prefer the current working directory so repo-root runs use source-of-truth shaders.
    NSString* cwd = [fileManager currentDirectoryPath];
    if (cwd) {
        NSString* candidate = [[cwd stringByAppendingPathComponent:@"shaders"]
                              stringByAppendingPathComponent:filename];
        if ([fileManager fileExistsAtPath:candidate]) {
            return candidate;
        }
    }

    // Fallback to a shaders directory next to the executable (useful when running from build tree).
    NSString* executableDir = [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent];
    if (executableDir) {
        NSString* candidate = [[executableDir stringByAppendingPathComponent:@"shaders"]
                              stringByAppendingPathComponent:filename];
        if ([fileManager fileExistsAtPath:candidate]) {
            return candidate;
        }
    }

    return nil;
}

std::string ReadFileToString(NSString* path) {
    if (!path) {
        return {};
    }

    NSError* error = nil;
    NSString* contents = [NSString stringWithContentsOfFile:path
                                                   encoding:NSUTF8StringEncoding
                                                      error:&error];
    if (!contents) {
        if (error) {
            NSLog(@"Failed to read shader %@: %@", path, error);
        }
        return {};
    }

    return std::string([contents UTF8String]);
}

std::pair<bool, std::string> ReadFileToStringWithIncludes(NSString* path,
                                                          NSMutableSet<NSString*>* includeStack) {
    if (!path || !includeStack) {
        return {false, {}};
    }

    NSString* normalizedPath = [path stringByStandardizingPath];
    if ([includeStack containsObject:normalizedPath]) {
        NSLog(@"Shader include cycle detected at %@", normalizedPath);
        return {false, {}};
    }

    NSError* error = nil;
    NSString* contents = [NSString stringWithContentsOfFile:normalizedPath
                                                   encoding:NSUTF8StringEncoding
                                                      error:&error];
    if (!contents) {
        if (error) {
            NSLog(@"Failed to read shader %@: %@", normalizedPath, error);
        }
        return {false, {}};
    }

    [includeStack addObject:normalizedPath];

    NSFileManager* fileManager = [NSFileManager defaultManager];
    NSCharacterSet* whitespace = [NSCharacterSet whitespaceCharacterSet];
    NSString* baseDir = [normalizedPath stringByDeletingLastPathComponent];
    NSArray<NSString*>* lines = [contents componentsSeparatedByString:@"\n"];

    std::string expanded;
    expanded.reserve([contents lengthOfBytesUsingEncoding:NSUTF8StringEncoding]);
    for (NSString* line in lines) {
        NSString* trimmed = [line stringByTrimmingCharactersInSet:whitespace];
        if ([trimmed hasPrefix:@"#include \""]) {
            NSRange firstQuote = [trimmed rangeOfString:@"\""];
            if (firstQuote.location == NSNotFound) {
                NSLog(@"Malformed shader include in %@: %@", normalizedPath, line);
                [includeStack removeObject:normalizedPath];
                return {false, {}};
            }

            NSRange searchRange = NSMakeRange(firstQuote.location + 1u,
                                              trimmed.length - firstQuote.location - 1u);
            NSRange secondQuote = [trimmed rangeOfString:@"\"" options:0 range:searchRange];
            if (secondQuote.location == NSNotFound) {
                NSLog(@"Malformed shader include in %@: %@", normalizedPath, line);
                [includeStack removeObject:normalizedPath];
                return {false, {}};
            }

            NSString* relativePath =
                [trimmed substringWithRange:NSMakeRange(firstQuote.location + 1u,
                                                        secondQuote.location - firstQuote.location - 1u)];
            NSString* includePath =
                [[baseDir stringByAppendingPathComponent:relativePath] stringByStandardizingPath];
            if (![fileManager fileExistsAtPath:includePath]) {
                NSLog(@"Shader include %@ not found from %@", relativePath, normalizedPath);
                [includeStack removeObject:normalizedPath];
                return {false, {}};
            }

            auto included = ReadFileToStringWithIncludes(includePath, includeStack);
            if (!included.first) {
                [includeStack removeObject:normalizedPath];
                return {false, {}};
            }
            expanded.append(included.second);
            if (!expanded.empty() && expanded.back() != '\n') {
                expanded.push_back('\n');
            }
            continue;
        }

        expanded.append([line UTF8String]);
        expanded.push_back('\n');
    }

    [includeStack removeObject:normalizedPath];
    return {true, expanded};
}

std::string LoadShaderSource() {
    NSString* commonPath = FindShaderPath(@"common.metal");
    NSString* pathtracePath = FindShaderPath(@"pathtrace.metal");
    NSString* mneePath = FindShaderPath(@"mnee.metal");
    NSString* displayPath = FindShaderPath(@"display.metal");

    if (!commonPath || !pathtracePath || !mneePath || !displayPath) {
        NSLog(@"Unable to locate shader sources");
        return {};
    }

    NSMutableSet<NSString*>* includeStack = [NSMutableSet set];
    std::string shaderSource = ReadFileToString(commonPath);
    shaderSource.append("\n");
    auto expandedPathtrace = ReadFileToStringWithIncludes(pathtracePath, includeStack);
    if (!expandedPathtrace.first) {
        return {};
    }
    shaderSource.append(expandedPathtrace.second);
    shaderSource.append("\n");
    shaderSource.append(ReadFileToString(mneePath));
    shaderSource.append("\n");
    shaderSource.append(ReadFileToString(displayPath));

    if (shaderSource.empty()) {
        NSLog(@"Shader source is empty");
        return {};
    }

    return shaderSource;
}

}  // namespace

namespace PathTracer {

bool Pipelines::compileShaders(const MetalContext& context) {
    m_device = context.device();
    if (!m_device) {
        NSLog(@"Pipelines::compileShaders - device is nil");
        return false;
    }

    std::string shaderSource = LoadShaderSource();
    if (shaderSource.empty()) {
        return false;
    }
    m_compilationReport.shaderSourceBytes = shaderSource.size();

    NSString* sourceNSString = [[NSString alloc] initWithBytes:shaderSource.data()
                                                        length:shaderSource.size()
                                                      encoding:NSUTF8StringEncoding];
    
    NSError* error = nil;
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#if PT_DEBUG_TOOLS
    NSNumber* debugTools = @1;
#else
    NSNumber* debugTools = @0;
#endif
#if PT_MNEE_SWRT_RAYS
    NSNumber* mneeSwrt = @1;
#else
    NSNumber* mneeSwrt = @0;
#endif
#if PT_MNEE_OCCLUSION_PARITY
    NSNumber* mneeParity = @1;
#else
    NSNumber* mneeParity = @0;
#endif
    options.preprocessorMacros = @{
        @"PT_DEBUG_TOOLS": debugTools,
        @"PT_MNEE_SWRT_RAYS": mneeSwrt,
        @"PT_MNEE_OCCLUSION_PARITY": mneeParity
    };
    CFAbsoluteTime compileStart = CFAbsoluteTimeGetCurrent();
    m_library = [m_device newLibraryWithSource:sourceNSString options:options error:&error];
    m_compilationReport.libraryCompileMs = (CFAbsoluteTimeGetCurrent() - compileStart) * 1000.0;
    if (m_verboseTiming) {
        NSLog(@"[Timing] PipelineLibraryCompileMs=%.2f", m_compilationReport.libraryCompileMs);
    }
    if (!m_library) {
        if (error) {
            NSLog(@"Failed to compile shaders: %@", error);
        }
        return false;
    }

    m_compilationReport.available = true;
    return true;
}

bool Pipelines::createComputePipelines() {
    if (!m_library) {
        NSLog(@"Cannot create compute pipelines - library is nil");
        return false;
    }

    auto recordPipeline = ^(NSString* key,
                            NSString* functionName,
                            NSString* stage,
                            double compileMs,
                            bool success,
                            bool rayTracingSupportFlag,
                            NSString* specialization) {
        CompilationRecord record{};
        record.key = key ? std::string([key UTF8String]) : std::string();
        record.functionName = functionName ? std::string([functionName UTF8String]) : std::string();
        record.stage = stage ? std::string([stage UTF8String]) : std::string();
        record.compileMs = compileMs;
        record.success = success;
        record.rayTracingSupportFlag = rayTracingSupportFlag;
        record.specialization =
            specialization ? std::string([specialization UTF8String]) : std::string();
        if (success) {
            if ([stage isEqualToString:@"compute"]) {
                ++m_compilationReport.computePipelineCount;
            } else if ([stage isEqualToString:@"render"]) {
                ++m_compilationReport.renderPipelineCount;
            }
        } else {
            ++m_compilationReport.failedPipelineCount;
        }
        m_compilationReport.records.push_back(std::move(record));
    };

    auto createComputePipeline = ^id<MTLComputePipelineState>(NSString* functionName, NSString* label) {
        id<MTLFunction> function = [m_library newFunctionWithName:functionName];
        if (!function) {
            NSLog(@"Unable to find compute function '%@'", functionName);
            recordPipeline(label, functionName, @"compute", 0.0, false, false, @"function_missing");
            return (id<MTLComputePipelineState>)nil;
        }

        if (!m_materialResourcesArgumentEncoder) {
            if ([function respondsToSelector:@selector(newArgumentEncoderWithBufferIndex:)]) {
                m_materialResourcesArgumentEncoder = [function newArgumentEncoderWithBufferIndex:23];
            }
        }

        MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
        const bool rayTracingFlag = EnableRayTracingIfAvailable(descriptor);
        descriptor.computeFunction = function;
        descriptor.label = label;

        NSError* localError = nil;
        CFAbsoluteTime pipelineStart = CFAbsoluteTimeGetCurrent();
        id<MTLComputePipelineState> pipeline =
            [m_device newComputePipelineStateWithDescriptor:descriptor
                                                    options:0
                                                 reflection:nil
                                                      error:&localError];
        double pipelineMs = (CFAbsoluteTimeGetCurrent() - pipelineStart) * 1000.0;
        recordPipeline(label,
                       functionName,
                       @"compute",
                       pipelineMs,
                       pipeline != nil,
                       rayTracingFlag,
                       rayTracingFlag ? @"raytracing_descriptor_enabled"
                                      : @"default_descriptor");
        if (m_verboseTiming) {
            NSLog(@"[Timing] ComputePipelineCompileMs %@=%.2f", label, pipelineMs);
        }
        if (!pipeline && localError) {
            NSLog(@"Failed to create compute pipeline '%@': %@", functionName, localError);
        }
        return pipeline;
    };

    m_integratePipeline = createComputePipeline(@"pathtraceIntegrateKernel", @"Path Trace Integrate");
    if (!m_integratePipeline) {
        return false;
    }

    bool deviceSupportsRaytracing = false;
    if ([m_device respondsToSelector:@selector(supportsRaytracing)]) {
        deviceSupportsRaytracing = [m_device supportsRaytracing];
    }

    if (deviceSupportsRaytracing) {
        m_integrateHardwarePipeline = createComputePipeline(@"pathtraceIntegrateHardwareKernel",
                                                            @"Path Trace Integrate (HW)");
    } else {
        m_integrateHardwarePipeline = nullptr;
    }

    m_wavefrontResetPipeline =
        createComputePipeline(@"wavefrontResetKernel", @"Wavefront Reset");
    if (!m_wavefrontResetPipeline) {
        return false;
    }

    m_wavefrontBuildDispatchArgsPipeline =
        createComputePipeline(@"wavefrontBuildDispatchArgsKernel",
                              @"Wavefront Build Dispatch Args");
    if (!m_wavefrontBuildDispatchArgsPipeline) {
        return false;
    }

    m_wavefrontCompactQueuePipeline =
        createComputePipeline(@"wavefrontCompactQueueKernel",
                              @"Wavefront Compact Queue");
    if (!m_wavefrontCompactQueuePipeline) {
        return false;
    }

    m_wavefrontPrepareIterationPipeline =
        createComputePipeline(@"wavefrontPrepareIterationKernel", @"Wavefront Prepare Iteration");
    if (!m_wavefrontPrepareIterationPipeline) {
        return false;
    }

    m_wavefrontGeneratePrimaryPipeline =
        createComputePipeline(@"wavefrontGeneratePrimaryKernel", @"Wavefront Generate Primary");
    if (!m_wavefrontGeneratePrimaryPipeline) {
        return false;
    }

    m_wavefrontIntersectPipeline =
        createComputePipeline(@"wavefrontIntersectKernel", @"Wavefront Intersect");
    if (!m_wavefrontIntersectPipeline) {
        return false;
    }

    if (deviceSupportsRaytracing) {
        m_wavefrontIntersectHardwarePipeline =
            createComputePipeline(@"wavefrontIntersectHardwareKernel",
                                  @"Wavefront Intersect (HW)");
        m_wavefrontShadowHardwarePipeline =
            createComputePipeline(@"wavefrontShadowHardwareKernel",
                                  @"Wavefront Shadow (HW)");
    } else {
        m_wavefrontIntersectHardwarePipeline = nullptr;
        m_wavefrontShadowHardwarePipeline = nullptr;
    }

    m_wavefrontShadePipeline =
        createComputePipeline(@"wavefrontShadeKernel", @"Wavefront Shade");
    if (!m_wavefrontShadePipeline) {
        return false;
    }

    m_wavefrontRestirDiDirectLightingPipeline =
        createComputePipeline(@"wavefrontRestirDiDirectLightingKernel",
                              @"Wavefront ReSTIR DI Direct Lighting");
    if (!m_wavefrontRestirDiDirectLightingPipeline) {
        return false;
    }

    m_wavefrontShadowPipeline =
        createComputePipeline(@"wavefrontShadowKernel", @"Wavefront Shadow");
    if (!m_wavefrontShadowPipeline) {
        return false;
    }

    m_wavefrontSwapQueuesPipeline =
        createComputePipeline(@"wavefrontSwapQueuesKernel", @"Wavefront Swap Queues");
    if (!m_wavefrontSwapQueuesPipeline) {
        return false;
    }

    m_wavefrontAccumulatePipeline =
        createComputePipeline(@"wavefrontAccumulateKernel", @"Wavefront Accumulate");
    if (!m_wavefrontAccumulatePipeline) {
        return false;
    }

    // Present kernel
    id<MTLFunction> presentFunction = [m_library newFunctionWithName:@"pathtracePresentKernel"];
    if (!presentFunction) {
        NSLog(@"Unable to find compute function 'pathtracePresentKernel'");
        return false;
    }

    MTLComputePipelineDescriptor* presentDesc = [[MTLComputePipelineDescriptor alloc] init];
    const bool presentRayTracingFlag = EnableRayTracingIfAvailable(presentDesc);
    presentDesc.computeFunction = presentFunction;
    presentDesc.label = @"Path Trace Present";
    NSError* presentError = nil;
    CFAbsoluteTime presentStart = CFAbsoluteTimeGetCurrent();
    m_presentPipeline = [m_device newComputePipelineStateWithDescriptor:presentDesc
                                                                options:0
                                                             reflection:nil
                                                                  error:&presentError];
    double presentMs = (CFAbsoluteTimeGetCurrent() - presentStart) * 1000.0;
    recordPipeline(@"Present",
                   @"pathtracePresentKernel",
                   @"compute",
                   presentMs,
                   m_presentPipeline != nil,
                   presentRayTracingFlag,
                   presentRayTracingFlag ? @"raytracing_descriptor_enabled"
                                         : @"default_descriptor");
    if (m_verboseTiming) {
        NSLog(@"[Timing] ComputePipelineCompileMs Present=%.2f", presentMs);
    }
    if (!m_presentPipeline) {
        if (presentError) {
            NSLog(@"Failed to create present pipeline: %@", presentError);
        }
        return false;
    }

    m_svgfTemporalPipeline =
        createComputePipeline(@"svgfTemporalMomentsKernel", @"SVGF Temporal Moments");
    if (!m_svgfTemporalPipeline) {
        return false;
    }

    m_svgfAtrousPipeline =
        createComputePipeline(@"svgfAtrousFilterKernel", @"SVGF A-Trous Filter");
    if (!m_svgfAtrousPipeline) {
        return false;
    }

    m_restirDiInitializeReservoirsPipeline =
        createComputePipeline(@"restirDiInitializeReservoirsKernel",
                              @"ReSTIR DI Initialize Reservoirs");
    if (!m_restirDiInitializeReservoirsPipeline) {
        return false;
    }

    m_restirDiInitialCandidatesPipeline =
        createComputePipeline(@"restirDiInitialCandidatesKernel",
                              @"ReSTIR DI Initial Candidates");
    if (!m_restirDiInitialCandidatesPipeline) {
        return false;
    }
    if (deviceSupportsRaytracing) {
        m_restirDiInitialCandidatesHardwarePipeline =
            createComputePipeline(@"restirDiInitialCandidatesHardwareKernel",
                                  @"ReSTIR DI Initial Candidates (HW)");
    } else {
        m_restirDiInitialCandidatesHardwarePipeline = nullptr;
    }

    m_restirDiTemporalReusePipeline =
        createComputePipeline(@"restirDiTemporalReuseKernel",
                              @"ReSTIR DI Temporal Reuse");
    if (!m_restirDiTemporalReusePipeline) {
        return false;
    }

    m_restirDiSpatialReusePipeline =
        createComputePipeline(@"restirDiSpatialReuseKernel",
                              @"ReSTIR DI Spatial Reuse");
    if (!m_restirDiSpatialReusePipeline) {
        return false;
    }

    m_restirDiVisibilityPipeline =
        createComputePipeline(@"restirDiVisibilityKernel",
                              @"ReSTIR DI Visibility");
    if (!m_restirDiVisibilityPipeline) {
        return false;
    }
    if (deviceSupportsRaytracing) {
        m_restirDiVisibilityHardwarePipeline =
            createComputePipeline(@"restirDiVisibilityHardwareKernel",
                                  @"ReSTIR DI Visibility (HW)");
    } else {
        m_restirDiVisibilityHardwarePipeline = nullptr;
    }

    m_restirDiApplyLightingPipeline =
        createComputePipeline(@"restirDiApplyLightingKernel",
                              @"ReSTIR DI Apply Lighting");
    if (!m_restirDiApplyLightingPipeline) {
        return false;
    }

    m_restirDiDebugAovPipeline =
        createComputePipeline(@"restirDiDebugAovKernel",
                              @"ReSTIR DI Debug AOV");
    if (!m_restirDiDebugAovPipeline) {
        return false;
    }

    // Clear kernel
    id<MTLFunction> clearFunction = [m_library newFunctionWithName:@"pathtraceClearKernel"];
    if (!clearFunction) {
        NSLog(@"Unable to find compute function 'pathtraceClearKernel'");
        return false;
    }
    MTLComputePipelineDescriptor* clearDesc = [[MTLComputePipelineDescriptor alloc] init];
    const bool clearRayTracingFlag = EnableRayTracingIfAvailable(clearDesc);
    clearDesc.computeFunction = clearFunction;
    clearDesc.label = @"Path Trace Clear";
    NSError* clearError = nil;
    CFAbsoluteTime clearStart = CFAbsoluteTimeGetCurrent();
    m_clearPipeline = [m_device newComputePipelineStateWithDescriptor:clearDesc
                                                              options:0
                                                           reflection:nil
                                                                error:&clearError];
    double clearMs = (CFAbsoluteTimeGetCurrent() - clearStart) * 1000.0;
    recordPipeline(@"Clear",
                   @"pathtraceClearKernel",
                   @"compute",
                   clearMs,
                   m_clearPipeline != nil,
                   clearRayTracingFlag,
                   clearRayTracingFlag ? @"raytracing_descriptor_enabled"
                                       : @"default_descriptor");
    if (m_verboseTiming) {
        NSLog(@"[Timing] ComputePipelineCompileMs Clear=%.2f", clearMs);
    }
    if (!m_clearPipeline) {
        if (clearError) {
            NSLog(@"Failed to create clear pipeline: %@", clearError);
        }
        return false;
    }

    return true;
}

bool Pipelines::createDisplayPipeline(MTLPixelFormat displayFormat) {
    if (!m_library) {
        NSLog(@"Cannot create display pipeline - library is nil");
        return false;
    }

    id<MTLFunction> vertexFunction = [m_library newFunctionWithName:@"displayVertex"];
    id<MTLFunction> fragmentFunction = [m_library newFunctionWithName:@"displayFragment"];
    if (!vertexFunction || !fragmentFunction) {
        NSLog(@"Unable to find display shader functions");
        return false;
    }

    MTLRenderPipelineDescriptor* descriptor = [[MTLRenderPipelineDescriptor alloc] init];
    descriptor.label = @"Display Pipeline";
    descriptor.vertexFunction = vertexFunction;
    descriptor.fragmentFunction = fragmentFunction;
    descriptor.colorAttachments[0].pixelFormat = displayFormat;

    NSError* error = nil;
    CFAbsoluteTime displayStart = CFAbsoluteTimeGetCurrent();
    m_displayPipeline = [m_device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    double displayMs = (CFAbsoluteTimeGetCurrent() - displayStart) * 1000.0;
    CompilationRecord record{};
    record.key = "Display";
    record.functionName = "displayVertex+displayFragment";
    record.stage = "render";
    record.compileMs = displayMs;
    record.success = m_displayPipeline != nil;
    record.rayTracingSupportFlag = false;
    record.specialization = "display_pixel_format";
    if (record.success) {
        ++m_compilationReport.renderPipelineCount;
    } else {
        ++m_compilationReport.failedPipelineCount;
    }
    m_compilationReport.records.push_back(std::move(record));
    if (m_verboseTiming) {
        NSLog(@"[Timing] RenderPipelineCompileMs=%.2f", displayMs);
    }
    if (!m_displayPipeline) {
        NSLog(@"Failed to create render pipeline: %@", error);
        return false;
    }

    return true;
}

bool Pipelines::initialize(const MetalContext& context, MTLPixelFormat displayFormat) {
    m_compilationReport = CompilationReport{};
    // Compile shaders
    if (!compileShaders(context)) {
        return false;
    }

    // Create compute pipelines
    if (!createComputePipelines()) {
        return false;
    }

    // Create display pipeline
    if (!createDisplayPipeline(displayFormat)) {
        return false;
    }

    return true;
}

bool Pipelines::reload(const MetalContext& context, MTLPixelFormat displayFormat) {
    NSLog(@"Reloading shaders and pipelines...");
    
    // Release old pipelines
    shutdown();
    
    // Reinitialize
    bool success = initialize(context, displayFormat);
    
    if (success) {
        NSLog(@"Shader reload successful");
    } else {
        NSLog(@"Shader reload failed");
    }
    
    return success;
}

void Pipelines::shutdown() {
    m_displayPipeline = nil;
    m_clearPipeline = nil;
    m_restirDiDebugAovPipeline = nil;
    m_restirDiApplyLightingPipeline = nil;
    m_restirDiVisibilityHardwarePipeline = nil;
    m_restirDiVisibilityPipeline = nil;
    m_restirDiSpatialReusePipeline = nil;
    m_restirDiTemporalReusePipeline = nil;
    m_restirDiInitialCandidatesHardwarePipeline = nil;
    m_restirDiInitialCandidatesPipeline = nil;
    m_restirDiInitializeReservoirsPipeline = nil;
    m_svgfAtrousPipeline = nil;
    m_svgfTemporalPipeline = nil;
    m_presentPipeline = nil;
    m_wavefrontAccumulatePipeline = nil;
    m_wavefrontSwapQueuesPipeline = nil;
    m_wavefrontShadowHardwarePipeline = nil;
    m_wavefrontShadowPipeline = nil;
    m_wavefrontRestirDiDirectLightingPipeline = nil;
    m_wavefrontShadePipeline = nil;
    m_wavefrontIntersectHardwarePipeline = nil;
    m_wavefrontIntersectPipeline = nil;
    m_wavefrontGeneratePrimaryPipeline = nil;
    m_wavefrontPrepareIterationPipeline = nil;
    m_wavefrontCompactQueuePipeline = nil;
    m_wavefrontBuildDispatchArgsPipeline = nil;
    m_wavefrontResetPipeline = nil;
    m_integrateHardwarePipeline = nil;
    m_integratePipeline = nil;
    m_library = nil;
    m_device = nil;
}

}  // namespace PathTracer
