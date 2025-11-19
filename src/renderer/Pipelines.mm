#import "renderer/Pipelines.h"
#import "renderer/MetalContext.h"
#import <Foundation/Foundation.h>
#if __has_include(<objc/message.h>)
#import <objc/message.h>
#endif
#include <string>

namespace {

void EnableRayTracingIfAvailable(MTLComputePipelineDescriptor* descriptor) {
#if __has_include(<objc/message.h>)
    if (!descriptor) {
        return;
    }
    if ([descriptor respondsToSelector:@selector(setSupportRayTracing:)]) {
        auto setter = reinterpret_cast<void (*)(id, SEL, BOOL)>(objc_msgSend);
        setter(descriptor, @selector(setSupportRayTracing:), YES);
    }
#else
    (void)descriptor;
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

    // Check a shaders directory next to the executable (useful when running from build tree).
    NSString* executableDir = [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent];
    if (executableDir) {
        NSString* candidate = [[executableDir stringByAppendingPathComponent:@"shaders"] 
                              stringByAppendingPathComponent:filename];
        if ([fileManager fileExistsAtPath:candidate]) {
            return candidate;
        }
    }

    // Fallback to the current working directory.
    NSString* cwd = [fileManager currentDirectoryPath];
    if (cwd) {
        NSString* candidate = [[cwd stringByAppendingPathComponent:@"shaders"] 
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

std::string LoadShaderSource() {
    NSString* commonPath = FindShaderPath(@"common.metal");
    NSString* pathtracePath = FindShaderPath(@"pathtrace.metal");
    NSString* displayPath = FindShaderPath(@"display.metal");

    if (!commonPath || !pathtracePath || !displayPath) {
        NSLog(@"Unable to locate shader sources");
        return {};
    }

    std::string shaderSource = ReadFileToString(commonPath);
    shaderSource.append("\n");
    shaderSource.append(ReadFileToString(pathtracePath));
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

    NSString* sourceNSString = [[NSString alloc] initWithBytes:shaderSource.data()
                                                        length:shaderSource.size()
                                                      encoding:NSUTF8StringEncoding];
    
    NSError* error = nil;
    m_library = [m_device newLibraryWithSource:sourceNSString options:nil error:&error];
    if (!m_library) {
        if (error) {
            NSLog(@"Failed to compile shaders: %@", error);
        }
        return false;
    }

    return true;
}

bool Pipelines::createComputePipelines() {
    if (!m_library) {
        NSLog(@"Cannot create compute pipelines - library is nil");
        return false;
    }

    auto createComputePipeline = ^id<MTLComputePipelineState>(NSString* functionName, NSString* label) {
        id<MTLFunction> function = [m_library newFunctionWithName:functionName];
        if (!function) {
            NSLog(@"Unable to find compute function '%@'", functionName);
            return (id<MTLComputePipelineState>)nil;
        }

    MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
    EnableRayTracingIfAvailable(descriptor);
        descriptor.computeFunction = function;
        descriptor.label = label;

        NSError* localError = nil;
        id<MTLComputePipelineState> pipeline =
            [m_device newComputePipelineStateWithDescriptor:descriptor
                                                    options:0
                                                 reflection:nil
                                                      error:&localError];
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

    // Present kernel
    id<MTLFunction> presentFunction = [m_library newFunctionWithName:@"pathtracePresentKernel"];
    if (!presentFunction) {
        NSLog(@"Unable to find compute function 'pathtracePresentKernel'");
        return false;
    }

    MTLComputePipelineDescriptor* presentDesc = [[MTLComputePipelineDescriptor alloc] init];
    EnableRayTracingIfAvailable(presentDesc);
    presentDesc.computeFunction = presentFunction;
    presentDesc.label = @"Path Trace Present";
    NSError* presentError = nil;
    m_presentPipeline = [m_device newComputePipelineStateWithDescriptor:presentDesc
                                                                options:0
                                                             reflection:nil
                                                                  error:&presentError];
    if (!m_presentPipeline) {
        if (presentError) {
            NSLog(@"Failed to create present pipeline: %@", presentError);
        }
        return false;
    }

    // Clear kernel
    id<MTLFunction> clearFunction = [m_library newFunctionWithName:@"pathtraceClearKernel"];
    if (!clearFunction) {
        NSLog(@"Unable to find compute function 'pathtraceClearKernel'");
        return false;
    }
    MTLComputePipelineDescriptor* clearDesc = [[MTLComputePipelineDescriptor alloc] init];
    EnableRayTracingIfAvailable(clearDesc);
    clearDesc.computeFunction = clearFunction;
    clearDesc.label = @"Path Trace Clear";
    NSError* clearError = nil;
    m_clearPipeline = [m_device newComputePipelineStateWithDescriptor:clearDesc
                                                              options:0
                                                           reflection:nil
                                                                error:&clearError];
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
    m_displayPipeline = [m_device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    if (!m_displayPipeline) {
        NSLog(@"Failed to create render pipeline: %@", error);
        return false;
    }

    return true;
}

bool Pipelines::initialize(const MetalContext& context, MTLPixelFormat displayFormat) {
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
    m_presentPipeline = nil;
    m_integrateHardwarePipeline = nil;
    m_integratePipeline = nil;
    m_library = nil;
    m_device = nil;
}

}  // namespace PathTracer
