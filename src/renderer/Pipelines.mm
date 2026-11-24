#import "renderer/Pipelines.h"
#import "renderer/MetalContext.h"
#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#if __has_include(<objc/message.h>)
#import <objc/message.h>
#endif
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <mutex>
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

// Finds a shader source file. `via` names the rule that found it. The order is:
// 1. [NSBundle mainBundle]'s shaders/ directory. For the app, that is Contents/Resources/shaders, which CMake fills. For
//    a bare CLI binary such as PathTracerHeadless, the main bundle is the executable's directory, so this finds the
//    shaders/ copy next to the binary, and that copy wins over the cwd's shaders/.
// 2. shaders/ in the current working directory, reached only when there are no shaders in the bundle or next to a CLI
//    binary.
// 3. shaders/ next to the executable (for a CLI binary, rule 1 has already found it).
NSString* FindShaderPath(NSString* filename, const char** via = nullptr) {
    NSFileManager* fileManager = [NSFileManager defaultManager];
    NSBundle* bundle = [NSBundle mainBundle];

    NSString* bundlePath = [bundle pathForResource:[filename stringByDeletingPathExtension]
                                            ofType:[filename pathExtension]
                                       inDirectory:@"shaders"];
    if (bundlePath && [fileManager fileExistsAtPath:bundlePath]) {
        if (via) *via = "main bundle (the app's Resources, or the directory of a CLI binary)";
        return bundlePath;
    }

    NSString* cwd = [fileManager currentDirectoryPath];
    if (cwd) {
        NSString* candidate = [[cwd stringByAppendingPathComponent:@"shaders"]
                              stringByAppendingPathComponent:filename];
        if ([fileManager fileExistsAtPath:candidate]) {
            if (via) *via = "current working directory";
            return candidate;
        }
    }

    NSString* executableDir = [[[NSBundle mainBundle] executablePath] stringByDeletingLastPathComponent];
    if (executableDir) {
        NSString* candidate = [[executableDir stringByAppendingPathComponent:@"shaders"]
                              stringByAppendingPathComponent:filename];
        if ([fileManager fileExistsAtPath:candidate]) {
            if (via) *via = "executable directory";
            return candidate;
        }
    }

    return nil;
}

std::mutex g_shaderDirectoryMutex;
std::string g_shaderDirectory;  // where LoadShaderSource last read common.metal
bool g_shaderDirectoryLogged = false;

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
    const char* commonVia = "";
    NSString* commonPath = FindShaderPath(@"common.metal", &commonVia);
    NSString* pathtracePath = FindShaderPath(@"pathtrace.metal");
    NSString* mneePath = FindShaderPath(@"mnee.metal");
    NSString* displayPath = FindShaderPath(@"display.metal");

    if (!commonPath || !pathtracePath || !mneePath || !displayPath) {
        NSLog(@"Unable to locate shader sources");
        return {};
    }

    {
        NSString* directory = [commonPath stringByDeletingLastPathComponent];
        std::lock_guard<std::mutex> lock(g_shaderDirectoryMutex);
        g_shaderDirectory = directory.UTF8String;
        if (!g_shaderDirectoryLogged) {
            g_shaderDirectoryLogged = true;
            NSLog(@"[Shaders] Reading the shader sources from %@ (found through the %s)", directory, commonVia);
            for (NSString* path : @[ pathtracePath, mneePath, displayPath ]) {
                if (![[path stringByDeletingLastPathComponent] isEqualToString:directory]) {
                    NSLog(@"[Shaders] warning: %@ comes from another directory than common.metal (%@)", path,
                          directory);
                }
            }
        }
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

uint64_t HashShaderSource(const std::string& source) {
    uint64_t hash = 1469598103934665603ull;
    for (const char rawC : source) {
        const unsigned char c = static_cast<unsigned char>(rawC);
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ull;
    }
    return hash;
}

NSString* BinaryArchivePathForSource(const std::string& shaderSource, bool sourceFeatures) {
    NSArray<NSURL*>* cacheUrls =
        [[NSFileManager defaultManager] URLsForDirectory:NSCachesDirectory
                                               inDomains:NSUserDomainMask];
    NSURL* cacheUrl = cacheUrls.count > 0 ? cacheUrls[0] : nil;
    if (!cacheUrl) {
        return nil;
    }
    NSString* directory =
        [[cacheUrl path] stringByAppendingPathComponent:@"MetalPathTracer/PipelineArchives"];
    NSError* createError = nil;
    [[NSFileManager defaultManager] createDirectoryAtPath:directory
                              withIntermediateDirectories:YES
                                               attributes:nil
                                                    error:&createError];
    if (createError) {
        NSLog(@"[PipelineCache] Failed to create archive directory %@: %@", directory, createError);
        return nil;
    }
    NSString* filename =
        [NSString stringWithFormat:@"pathtracer_%016llx%@.metallibarchive",
                                   static_cast<unsigned long long>(HashShaderSource(shaderSource)),
                                   sourceFeatures ? @"_source" : @""];
    return [directory stringByAppendingPathComponent:filename];
}

}  // namespace

namespace PathTracer {

std::string ResolvedShaderDirectory() {
    std::lock_guard<std::mutex> lock(g_shaderDirectoryMutex);
    return g_shaderDirectory;
}

// The megakernel integrate pipelines (software, hardware, hardware with default features). Each is a cold compile of
// minutes and a run dispatches one of them, or none on the wavefront, so a variant missing from the archives is built
// on first use, into its own archive file (written whole and renamed into place, so parallel runs never overwrite one
// another's variants). A variant already in an archive is created at initialization, as before, without compiling.
// The window builds the missing ones in the background once a frame is on screen. The background build holds its own
// reference to this state, so a reload or a quit during it frees nothing under it. This file is built without ARC.
struct Pipelines::LazyIntegrateState {
    std::mutex mutex;  // serializes builds; a built variant is read through `ready` without it
    id<MTLDevice> device = nil;
    id<MTLLibrary> library = nil;
    id<MTLBinaryArchive> mainArchive = nil;
    std::string archiveBase;  // main archive path without its extension; empty without archives
    bool deviceSupportsRaytracing = false;
    bool verboseTiming = false;
    id<MTLComputePipelineState> pipelines[3] = {nil, nil, nil};
    std::atomic<bool> ready[3] = {false, false, false};
    std::atomic<bool> failed[3] = {false, false, false};
    std::atomic<bool> cancelled{false};
    std::atomic<bool> backgroundStarted{false};
    std::mutex recordsMutex;
    std::vector<CompilationRecord> records;

    ~LazyIntegrateState() {
        for (id<MTLComputePipelineState> pipeline : pipelines) {
            [pipeline release];
        }
        [mainArchive release];
        [library release];
        [device release];
    }
};

namespace {

constexpr int kIntegrateSoftware = 0;
constexpr int kIntegrateHardware = 1;
constexpr int kIntegrateHardwareDefaultFeatures = 2;

NSString* IntegrateVariantFunction(int variant) {
    return variant == kIntegrateSoftware ? @"pathtraceIntegrateKernel" : @"pathtraceIntegrateHardwareKernel";
}

NSString* IntegrateVariantLabel(int variant) {
    return variant == kIntegrateSoftware ? @"Path Trace Integrate"
         : variant == kIntegrateHardware ? @"Path Trace Integrate (HW)"
                                         : @"Path Trace Integrate (HW, default features)";
}

}  // namespace

// Builds or loads one variant; the caller holds state.mutex. With archiveHitOnly, only an archive hit creates it
// (initialization); otherwise a miss compiles it into its own archive. A failure is recorded and logged with the
// pipeline error; stopping the run is the caller's.
static bool BuildIntegrateVariantLocked(Pipelines::LazyIntegrateState& state, int variant, bool archiveHitOnly) {
    NSString* label = IntegrateVariantLabel(variant);
    NSString* functionName = IntegrateVariantFunction(variant);
    const bool defaultFeatures = variant == kIntegrateHardwareDefaultFeatures;
    // Shared shader code reads the default-features function constant (index 2), and Metal only builds pipelines
    // from such functions once specialized: unset, kMegakernelDefaultFeatures is false (shaders/common.metal).
    MTLFunctionConstantValues* constants = [[[MTLFunctionConstantValues alloc] init] autorelease];
    if (defaultFeatures) {
        bool enabled = true;
        [constants setConstantValue:&enabled type:MTLDataTypeBool atIndex:2];
    }
    NSError* functionError = nil;
    id<MTLFunction> function = [[state.library newFunctionWithName:functionName
                                                   constantValues:constants
                                                            error:&functionError] autorelease];
    if (!function) {
        if (!archiveHitOnly) {
            NSLog(defaultFeatures ? @"Unable to specialize compute function '%@': %@"
                                  : @"Unable to find compute function '%@': %@", functionName, functionError);
        }
        return false;
    }
    MTLComputePipelineDescriptor* descriptor = [[[MTLComputePipelineDescriptor alloc] init] autorelease];
    const bool rayTracingFlag = EnableRayTracingIfAvailable(descriptor);
    descriptor.computeFunction = function;
    descriptor.label = label;

    NSString* variantPath = state.archiveBase.empty() ? nil
        : [NSString stringWithFormat:@"%s_integrate%d.metallibarchive", state.archiveBase.c_str(), variant];
    const bool archivesSupported = variantPath && [descriptor respondsToSelector:@selector(setBinaryArchives:)] &&
        [state.device respondsToSelector:@selector(newBinaryArchiveWithDescriptor:error:)];
    CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
    id<MTLComputePipelineState> pipeline = nil;
    NSError* localError = nil;
    bool compiled = false;
    if (archivesSupported) {
        // Its own archive first, then the main one (which held every variant before they were built lazily).
        NSMutableArray* archives = [NSMutableArray array];
        if ([[NSFileManager defaultManager] fileExistsAtPath:variantPath]) {
            MTLBinaryArchiveDescriptor* existing = [[[MTLBinaryArchiveDescriptor alloc] init] autorelease];
            existing.url = [NSURL fileURLWithPath:variantPath];
            id<MTLBinaryArchive> archive = [[state.device newBinaryArchiveWithDescriptor:existing error:nil] autorelease];
            if (archive) {
                [archives addObject:archive];
            }
        }
        if (state.mainArchive) {
            [archives addObject:state.mainArchive];
        }
        if (archives.count > 0) {
            descriptor.binaryArchives = archives;
            pipeline = [state.device newComputePipelineStateWithDescriptor:descriptor
                                                                   options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                                reflection:nil
                                                                     error:nil];
        }
        if (!pipeline && !archiveHitOnly) {
            MTLBinaryArchiveDescriptor* fresh = [[[MTLBinaryArchiveDescriptor alloc] init] autorelease];
            id<MTLBinaryArchive> archive = [[state.device newBinaryArchiveWithDescriptor:fresh error:nil] autorelease];
            if (archive) {
                NSError* addError = nil;
                [archive addComputePipelineFunctionsWithDescriptor:descriptor error:&addError];
                if (addError && state.verboseTiming) {
                    NSLog(@"[PipelineCache] Failed to add compute descriptor %@ to archive: %@", label, addError);
                }
                descriptor.binaryArchives = @[archive];
            }
            pipeline = [state.device newComputePipelineStateWithDescriptor:descriptor
                                                                   options:0
                                                                reflection:nil
                                                                     error:&localError];
            compiled = true;
            if (pipeline && archive) {
                NSString* tmp = [variantPath stringByAppendingFormat:@".%d.tmp", getpid()];
                NSError* serializeError = nil;
                if ([archive serializeToURL:[NSURL fileURLWithPath:tmp] error:&serializeError]) {
                    rename([tmp fileSystemRepresentation], [variantPath fileSystemRepresentation]);
                } else if (state.verboseTiming) {
                    NSLog(@"[PipelineCache] Failed to serialize the %@ archive: %@", label, serializeError);
                }
            }
        }
    } else if (!archiveHitOnly) {
        pipeline = [state.device newComputePipelineStateWithDescriptor:descriptor
                                                               options:0
                                                            reflection:nil
                                                                 error:&localError];
        compiled = true;
    }
    const double ms = (CFAbsoluteTimeGetCurrent() - start) * 1000.0;
    if (!pipeline && archiveHitOnly) {
        return false;  // not in an archive: built on first use
    }
    {
        Pipelines::CompilationRecord record{};
        record.key = std::string([label UTF8String]);
        record.functionName = std::string([functionName UTF8String]);
        record.stage = "compute";
        record.compileMs = ms;
        record.success = pipeline != nil;
        record.rayTracingSupportFlag = rayTracingFlag;
        record.specialization = defaultFeatures ? "function_constant:default_features"
                              : rayTracingFlag ? "raytracing_descriptor_enabled" : "default_descriptor";
        std::lock_guard<std::mutex> recordsLock(state.recordsMutex);
        state.records.push_back(std::move(record));
    }
    if (state.verboseTiming) {
        NSLog(@"[Timing] ComputePipelineCompileMs %@=%.2f", label, ms);
    }
    if (!pipeline) {
        NSLog(defaultFeatures ? @"Failed to create specialized compute pipeline '%@': %@"
                              : @"Failed to create compute pipeline '%@': %@", functionName, localError);
        state.failed[variant].store(true);
        return false;
    }
    if (compiled) {
        NSLog(@"[Pipelines] Built %@ on first use in %.0f ms", label, ms);
    }
    state.pipelines[variant] = pipeline;  // +1 from new..., released with the state
    state.ready[variant].store(true, std::memory_order_release);
    return true;
}

MTLComputePipelineStateHandle Pipelines::integrateVariant(const std::shared_ptr<LazyIntegrateState>& state,
                                                          int variant, bool foreground) {
    if (!state || (variant != kIntegrateSoftware && !state->deviceSupportsRaytracing)) {
        return nil;
    }
    if (state->ready[variant].load(std::memory_order_acquire)) {
        return state->pipelines[variant];
    }
    if (!state->failed[variant].load()) {
        std::unique_lock<std::mutex> lock(state->mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            if (foreground) {
                NSLog(@"[Pipelines] Waiting for %@: another integrate pipeline is compiling",
                      IntegrateVariantLabel(variant));
            }
            lock.lock();
        }
        if (!state->ready[variant].load() && !state->failed[variant].load()) {
            BuildIntegrateVariantLocked(*state, variant, /*archiveHitOnly=*/false);
        }
    }
    if (state->ready[variant].load(std::memory_order_acquire)) {
        return state->pipelines[variant];
    }
    if (foreground) {
        // As when this pipeline failed at initialization: the run stops (the error is logged above).
        NSLog(@"[Pipelines] %@ failed to build; stopping.", IntegrateVariantLabel(variant));
        std::exit(EXIT_FAILURE);
    }
    return nil;
}

MTLComputePipelineStateHandle Pipelines::integrate() const {
    return integrateVariant(m_lazyIntegrate, kIntegrateSoftware, true);
}

MTLComputePipelineStateHandle Pipelines::integrateHardware() const {
    return integrateVariant(m_lazyIntegrate, kIntegrateHardware, true);
}

MTLComputePipelineStateHandle Pipelines::integrateHardwareDefaultFeatures() const {
    return integrateVariant(m_lazyIntegrate, kIntegrateHardwareDefaultFeatures, true);
}

bool Pipelines::integrateHardwareAvailable() const {
    return m_lazyIntegrate && m_lazyIntegrate->deviceSupportsRaytracing;
}

void Pipelines::buildIntegrateVariantsInBackground() const {
    std::shared_ptr<LazyIntegrateState> state = m_lazyIntegrate;
    if (!state || state->backgroundStarted.exchange(true)) {
        return;
    }
    bool missing = false;
    for (int variant = 0; variant < 3; ++variant) {
        missing = missing || (!state->ready[variant].load() &&
                              (variant == kIntegrateSoftware || state->deviceSupportsRaytracing));
    }
    if (!missing) {
        return;
    }
    NSLog(@"[Pipelines] Building the integrate pipelines not built yet in the background");
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
        for (int variant : {kIntegrateHardwareDefaultFeatures, kIntegrateHardware, kIntegrateSoftware}) {
            if (state->cancelled.load()) {
                return;
            }
            integrateVariant(state, variant, /*foreground=*/false);
        }
    });
}

Pipelines::CompilationReport Pipelines::compilationReport() const {
    CompilationReport report = m_compilationReport;
    if (m_lazyIntegrate) {
        std::lock_guard<std::mutex> lock(m_lazyIntegrate->recordsMutex);
        for (const CompilationRecord& record : m_lazyIntegrate->records) {
            if (record.success) {
                ++report.computePipelineCount;
            } else {
                ++report.failedPipelineCount;
            }
            report.records.push_back(record);
        }
    }
    return report;
}

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
    m_binaryArchive = nil;
    m_binaryArchivePath.clear();
    NSString* archivePath = BinaryArchivePathForSource(shaderSource, m_sourceFeatures);
    if (archivePath && [m_device respondsToSelector:@selector(newBinaryArchiveWithDescriptor:error:)]) {
        MTLBinaryArchiveDescriptor* archiveDescriptor = [[MTLBinaryArchiveDescriptor alloc] init];
        if ([[NSFileManager defaultManager] fileExistsAtPath:archivePath]) {
            archiveDescriptor.url = [NSURL fileURLWithPath:archivePath];
        }
        NSError* archiveError = nil;
        m_binaryArchive = [m_device newBinaryArchiveWithDescriptor:archiveDescriptor
                                                             error:&archiveError];
        if (m_binaryArchive) {
            m_binaryArchivePath = std::string([archivePath UTF8String]);
            m_compilationReport.metalBinaryArchiveUsed = true;
            m_compilationReport.binaryArchivePath = m_binaryArchivePath;
            m_compilationReport.cacheStrategy =
                "runtime_source_library_with_persistent_metal_binary_archive";
        } else {
            std::string reason = archiveError ? std::string([[archiveError localizedDescription] UTF8String])
                                              : std::string("unknown archive creation failure");
            m_compilationReport.cacheStrategy =
                "runtime_source_library_binary_archive_unavailable:" + reason;
            if (m_verboseTiming) {
                NSLog(@"[PipelineCache] Binary archive unavailable: %@", archiveError);
            }
        }
    } else {
        m_compilationReport.cacheStrategy =
            "runtime_source_library_binary_archive_api_unavailable";
    }

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
        @"PT_MNEE_OCCLUSION_PARITY": mneeParity,
        @"PT_SOURCE_FEATURES": m_sourceFeatures ? @1 : @0
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
        // Shared shader code reads the default-features function constant (index 2), and Metal
        // only builds pipelines from such functions once specialized. No values are set, so the
        // constant stays undefined and kMegakernelDefaultFeatures is false (shaders/common.metal).
        MTLFunctionConstantValues* noConstants = [[MTLFunctionConstantValues alloc] init];
        NSError* functionError = nil;
        id<MTLFunction> function = [m_library newFunctionWithName:functionName
                                                   constantValues:noConstants
                                                            error:&functionError];
        if (!function) {
            NSLog(@"Unable to find compute function '%@': %@", functionName, functionError);
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
        if (m_binaryArchive &&
            [descriptor respondsToSelector:@selector(setBinaryArchives:)]) {
            descriptor.binaryArchives = @[(id<MTLBinaryArchive>)m_binaryArchive];
            NSError* archiveAddError = nil;
            [(id<MTLBinaryArchive>)m_binaryArchive addComputePipelineFunctionsWithDescriptor:descriptor
                                                                                       error:&archiveAddError];
            if (archiveAddError && m_verboseTiming) {
                NSLog(@"[PipelineCache] Failed to add compute descriptor %@ to archive: %@",
                      label,
                      archiveAddError);
            }
        }

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

    // A compute pipeline with one bool function constant set (wavefront shade surface
    // resolve at index 1, megakernel default features at index 2).
    auto createSpecializedComputePipeline =
        ^id<MTLComputePipelineState>(NSString* functionName,
                                     NSString* label,
                                     NSUInteger constantIndex,
                                     bool constantValue,
                                     NSString* specialization) {
        MTLFunctionConstantValues* constants = [[MTLFunctionConstantValues alloc] init];
        bool enabled = constantValue;
        [constants setConstantValue:&enabled type:MTLDataTypeBool atIndex:constantIndex];
        NSError* functionError = nil;
        id<MTLFunction> function =
            [m_library newFunctionWithName:functionName
                            constantValues:constants
                                     error:&functionError];
        if (!function) {
            NSLog(@"Unable to specialize compute function '%@': %@", functionName, functionError);
            recordPipeline(label, functionName, @"compute", 0.0, false, false, @"function_constant_missing");
            return (id<MTLComputePipelineState>)nil;
        }
        MTLComputePipelineDescriptor* descriptor = [[MTLComputePipelineDescriptor alloc] init];
        const bool rayTracingFlag = EnableRayTracingIfAvailable(descriptor);
        descriptor.computeFunction = function;
        descriptor.label = label;
        if (m_binaryArchive &&
            [descriptor respondsToSelector:@selector(setBinaryArchives:)]) {
            descriptor.binaryArchives = @[(id<MTLBinaryArchive>)m_binaryArchive];
            NSError* archiveAddError = nil;
            [(id<MTLBinaryArchive>)m_binaryArchive addComputePipelineFunctionsWithDescriptor:descriptor
                                                                                       error:&archiveAddError];
            if (archiveAddError && m_verboseTiming) {
                NSLog(@"[PipelineCache] Failed to add compute descriptor %@ to archive: %@",
                      label,
                      archiveAddError);
            }
        }
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
                       specialization);
        if (m_verboseTiming) {
            NSLog(@"[Timing] ComputePipelineCompileMs %@=%.2f", label, pipelineMs);
        }
        if (!pipeline && localError) {
            NSLog(@"Failed to create specialized compute pipeline '%@': %@", functionName, localError);
        }
        return pipeline;
    };

    // The integrate pipelines are built on first use (integrateVariant). The material-resources argument encoder
    // comes from the integrate function, as when that pipeline was built here first.
    {
        MTLFunctionConstantValues* noConstants = [[MTLFunctionConstantValues alloc] init];
        NSError* functionError = nil;
        id<MTLFunction> integrateFunction = [m_library newFunctionWithName:@"pathtraceIntegrateKernel"
                                                            constantValues:noConstants
                                                                     error:&functionError];
        if (!integrateFunction) {
            NSLog(@"Unable to find compute function '%@': %@", @"pathtraceIntegrateKernel", functionError);
            recordPipeline(@"Path Trace Integrate", @"pathtraceIntegrateKernel", @"compute", 0.0, false, false,
                           @"function_missing");
            return false;
        }
        if (!m_materialResourcesArgumentEncoder &&
            [integrateFunction respondsToSelector:@selector(newArgumentEncoderWithBufferIndex:)]) {
            m_materialResourcesArgumentEncoder = [integrateFunction newArgumentEncoderWithBufferIndex:23];
        }
    }
    if (m_lazyIntegrate) {
        m_lazyIntegrate->cancelled.store(true);
    }
    m_lazyIntegrate = std::make_shared<LazyIntegrateState>();
    m_lazyIntegrate->device = [m_device retain];
    m_lazyIntegrate->library = [m_library retain];
    m_lazyIntegrate->mainArchive = [(id<MTLBinaryArchive>)m_binaryArchive retain];
    if (!m_binaryArchivePath.empty()) {
        const std::string& path = m_binaryArchivePath;
        const size_t dot = path.rfind('.');
        m_lazyIntegrate->archiveBase = dot == std::string::npos ? path : path.substr(0, dot);
    }
    m_lazyIntegrate->verboseTiming = m_verboseTiming;
    m_lazyIntegrate->deviceSupportsRaytracing =
        [m_device respondsToSelector:@selector(supportsRaytracing)] && [m_device supportsRaytracing];
    // Variants already in an archive are created now, as before, without compiling; the others on first use.
    {
        std::lock_guard<std::mutex> lock(m_lazyIntegrate->mutex);
        for (int variant = 0; variant < 3; ++variant) {
            if (variant == 0 || m_lazyIntegrate->deviceSupportsRaytracing) {
                BuildIntegrateVariantLocked(*m_lazyIntegrate, variant, /*archiveHitOnly=*/true);
            }
        }
    }
    const bool deviceSupportsRaytracing = m_lazyIntegrate->deviceSupportsRaytracing;

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
        m_wavefrontShadowIndexedHardwarePipeline =
            createComputePipeline(@"wavefrontShadowIndexedHardwareKernel",
                                  @"Wavefront Shadow Indexed (HW)");
    } else {
        m_wavefrontIntersectHardwarePipeline = nullptr;
        m_wavefrontShadowHardwarePipeline = nullptr;
        m_wavefrontShadowIndexedHardwarePipeline = nullptr;
    }

    m_wavefrontShadePipeline =
        createSpecializedComputePipeline(@"wavefrontShadeKernel",
                                         @"Wavefront Shade",
                                         1,
                                         false,
                                         @"function_constant:direct_surface");
    if (!m_wavefrontShadePipeline) {
        return false;
    }
    m_wavefrontShadeResolveSurfacePipeline =
        createComputePipeline(@"wavefrontShadeResolveSurfaceKernel",
                              @"Wavefront Shade Resolve Surface");
    if (!m_wavefrontShadeResolveSurfacePipeline) {
        return false;
    }
    m_wavefrontShadeResolvedPipeline =
        createSpecializedComputePipeline(@"wavefrontShadeKernel",
                                         @"Wavefront Shade Resolved",
                                         1,
                                         true,
                                         @"function_constant:resolved_surface");
    if (!m_wavefrontShadeResolvedPipeline) {
        return false;
    }

    m_wavefrontShadeIndirectProbePipeline =
        createComputePipeline(@"wavefrontShadeIndirectProbeKernel",
                              @"Wavefront Shade Indirect Probe");
    if (!m_wavefrontShadeIndirectProbePipeline) {
        return false;
    }
    m_wavefrontShadeSurfaceProbePipeline =
        createComputePipeline(@"wavefrontShadeSurfaceProbeKernel",
                              @"Wavefront Shade Surface Probe");
    if (!m_wavefrontShadeSurfaceProbePipeline) {
        return false;
    }
    m_wavefrontShadeMetadataProbePipeline =
        createComputePipeline(@"wavefrontShadeMetadataProbeKernel",
                              @"Wavefront Shade Metadata Probe");
    if (!m_wavefrontShadeMetadataProbePipeline) {
        return false;
    }
    m_wavefrontShadeScalarProbePipeline =
        createComputePipeline(@"wavefrontShadeScalarProbeKernel",
                              @"Wavefront Shade Scalar Probe");
    if (!m_wavefrontShadeScalarProbePipeline) {
        return false;
    }
    m_wavefrontShadeBsdfProbePipeline =
        createComputePipeline(@"wavefrontShadeBsdfProbeKernel",
                              @"Wavefront Shade BSDF Probe");
    if (!m_wavefrontShadeBsdfProbePipeline) {
        return false;
    }
    m_wavefrontShadeClosureProbePipeline =
        createComputePipeline(@"wavefrontShadeClosureProbeKernel",
                              @"Wavefront Shade Closure Probe");
    if (!m_wavefrontShadeClosureProbePipeline) {
        return false;
    }

    m_wavefrontRestirDiDirectLightingPipeline =
        createComputePipeline(@"wavefrontRestirDiDirectLightingKernel",
                              @"Wavefront ReSTIR DI Direct Lighting");
    if (!m_wavefrontRestirDiDirectLightingPipeline) {
        return false;
    }

    m_wavefrontBuildShadowIndicesPipeline =
        createComputePipeline(@"wavefrontBuildShadowIndicesKernel",
                              @"Wavefront Build Shadow Indices");
    if (!m_wavefrontBuildShadowIndicesPipeline) {
        return false;
    }

    m_wavefrontShadowPipeline =
        createComputePipeline(@"wavefrontShadowKernel", @"Wavefront Shadow");
    if (!m_wavefrontShadowPipeline) {
        return false;
    }

    m_wavefrontShadowIndexedPipeline =
        createComputePipeline(@"wavefrontShadowIndexedKernel",
                              @"Wavefront Shadow Indexed");
    if (!m_wavefrontShadowIndexedPipeline) {
        return false;
    }

    m_wavefrontSwapQueuesPipeline =
        createComputePipeline(@"wavefrontSwapQueuesKernel", @"Wavefront Swap Queues");
    if (!m_wavefrontSwapQueuesPipeline) {
        return false;
    }

    m_wavefrontRadianceCacheTrainPipeline =
        createComputePipeline(@"wavefrontRadianceCacheTrainKernel", @"Wavefront Radiance Cache Train");
    if (!m_wavefrontRadianceCacheTrainPipeline) {
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
    if (m_binaryArchive &&
        [presentDesc respondsToSelector:@selector(setBinaryArchives:)]) {
        presentDesc.binaryArchives = @[(id<MTLBinaryArchive>)m_binaryArchive];
        NSError* archiveAddError = nil;
        [(id<MTLBinaryArchive>)m_binaryArchive addComputePipelineFunctionsWithDescriptor:presentDesc
                                                                                   error:&archiveAddError];
        if (archiveAddError && m_verboseTiming) {
            NSLog(@"[PipelineCache] Failed to add present descriptor to archive: %@", archiveAddError);
        }
    }
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

    m_svgfCompositeSplitPipeline =
        createComputePipeline(@"svgfCompositeSplitKernel", @"SVGF Direct/Indirect Composite");
    if (!m_svgfCompositeSplitPipeline) {
        return false;
    }

    m_mlReconstructionTinyPipeline =
        createComputePipeline(@"mlReconstructionTinyKernel", @"ML Reconstruction Tiny");
    if (!m_mlReconstructionTinyPipeline) {
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
    if (m_binaryArchive &&
        [clearDesc respondsToSelector:@selector(setBinaryArchives:)]) {
        clearDesc.binaryArchives = @[(id<MTLBinaryArchive>)m_binaryArchive];
        NSError* archiveAddError = nil;
        [(id<MTLBinaryArchive>)m_binaryArchive addComputePipelineFunctionsWithDescriptor:clearDesc
                                                                                   error:&archiveAddError];
        if (archiveAddError && m_verboseTiming) {
            NSLog(@"[PipelineCache] Failed to add clear descriptor to archive: %@", archiveAddError);
        }
    }
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
    MTLFunctionConstantValues* displayConstants = [[MTLFunctionConstantValues alloc] init];
    BOOL displayBloomSpecializedEnabled = YES;
    [displayConstants setConstantValue:&displayBloomSpecializedEnabled
                                  type:MTLDataTypeBool
                               atIndex:0];
    NSError* fragmentFunctionError = nil;
    id<MTLFunction> fragmentFunction =
        [m_library newFunctionWithName:@"displayFragment"
                        constantValues:displayConstants
                                 error:&fragmentFunctionError];
    if (!vertexFunction || !fragmentFunction) {
        if (fragmentFunctionError) {
            NSLog(@"Unable to specialize display fragment shader: %@", fragmentFunctionError);
        } else {
            NSLog(@"Unable to find display shader functions");
        }
        return false;
    }
    m_compilationReport.functionConstantsUsed = true;
    m_compilationReport.specializationStrategy =
        "pipeline keys split backend/execution features; display fragment uses Metal function constant kDisplayBloomSpecializedEnabled";

    MTLRenderPipelineDescriptor* descriptor = [[MTLRenderPipelineDescriptor alloc] init];
    descriptor.label = @"Display Pipeline";
    descriptor.vertexFunction = vertexFunction;
    descriptor.fragmentFunction = fragmentFunction;
    descriptor.colorAttachments[0].pixelFormat = displayFormat;
    if (m_binaryArchive &&
        [descriptor respondsToSelector:@selector(setBinaryArchives:)]) {
        descriptor.binaryArchives = @[(id<MTLBinaryArchive>)m_binaryArchive];
        NSError* archiveAddError = nil;
        [(id<MTLBinaryArchive>)m_binaryArchive addRenderPipelineFunctionsWithDescriptor:descriptor
                                                                                  error:&archiveAddError];
        if (archiveAddError && m_verboseTiming) {
            NSLog(@"[PipelineCache] Failed to add display descriptor to archive: %@", archiveAddError);
        }
    }

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
    record.specialization = "display_pixel_format,function_constant:bloom_enabled";
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

    if (m_binaryArchive && !m_binaryArchivePath.empty() &&
        [(id<MTLBinaryArchive>)m_binaryArchive respondsToSelector:@selector(serializeToURL:error:)]) {
        NSError* serializeError = nil;
        NSURL* archiveUrl =
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:m_binaryArchivePath.c_str()]];
        const BOOL serialized =
            [(id<MTLBinaryArchive>)m_binaryArchive serializeToURL:archiveUrl error:&serializeError];
        m_compilationReport.metalBinaryArchiveSerialized = serialized == YES;
        m_compilationReport.harvestedCompilationReady = serialized == YES;
        if (!serialized && serializeError) {
            m_compilationReport.cacheStrategy += ";serialize_failed:" +
                std::string([[serializeError localizedDescription] UTF8String]);
            if (m_verboseTiming) {
                NSLog(@"[PipelineCache] Failed to serialize binary archive: %@", serializeError);
            }
        }
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
    m_svgfCompositeSplitPipeline = nil;
    m_presentPipeline = nil;
    m_wavefrontAccumulatePipeline = nil;
    m_wavefrontRadianceCacheTrainPipeline = nil;
    m_wavefrontSwapQueuesPipeline = nil;
    m_wavefrontShadowHardwarePipeline = nil;
    m_wavefrontShadowPipeline = nil;
    m_wavefrontRestirDiDirectLightingPipeline = nil;
    m_wavefrontShadeBsdfProbePipeline = nil;
    m_wavefrontShadeClosureProbePipeline = nil;
    m_wavefrontShadeScalarProbePipeline = nil;
    m_wavefrontShadeMetadataProbePipeline = nil;
    m_wavefrontShadeSurfaceProbePipeline = nil;
    m_wavefrontShadeIndirectProbePipeline = nil;
    m_wavefrontShadeResolvedPipeline = nil;
    m_wavefrontShadeResolveSurfacePipeline = nil;
    m_wavefrontShadePipeline = nil;
    m_wavefrontIntersectHardwarePipeline = nil;
    m_wavefrontIntersectPipeline = nil;
    m_wavefrontGeneratePrimaryPipeline = nil;
    m_wavefrontPrepareIterationPipeline = nil;
    m_wavefrontCompactQueuePipeline = nil;
    m_wavefrontBuildDispatchArgsPipeline = nil;
    m_wavefrontResetPipeline = nil;
    if (m_lazyIntegrate) {
        m_lazyIntegrate->cancelled.store(true);
        m_lazyIntegrate.reset();
    }
    m_binaryArchive = nil;
    m_binaryArchivePath.clear();
    m_library = nil;
    m_device = nil;
}

}  // namespace PathTracer
