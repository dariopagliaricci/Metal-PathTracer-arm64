#import <AppKit/AppKit.h>
#import <dispatch/dispatch.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <filesystem>

#include <simd/simd.h>

#include "MetalRenderer.h"
#include "MetalShaderTypes.h"
#include "IntersectionProvider.h"
#include "renderer/Accumulation.h"
#include "renderer/BvhBuilder.h"
#include "renderer/DenoiserContext.h"
#include "renderer/ImageWriter.h"
#include "renderer/MetalContext.h"
#include "renderer/Pipelines.h"
#include "renderer/SceneResources.h"
#include "renderer/SceneManager.h"
#include "renderer/MetalHandles.h"
#include "renderer/PerformanceStats.h"
#include "renderer/RenderLoop.h"
#include "renderer/RenderSettings.h"
#include "renderer/SettingsUtils.h"
#include "renderer/UIOverlay.h"

#ifndef PATH_TRACER_ENABLE_OIDN
#define PATH_TRACER_ENABLE_OIDN 0
#endif

// ImGui is only needed for windowed mode (not headless)
#if __has_include("imgui.h")
#include "imgui.h"
#include "imgui_internal.h"
#include "backends/imgui_impl_metal.h"
#include "backends/imgui_impl_osx.h"
#define HAS_IMGUI 1
#else
#define HAS_IMGUI 0
#endif

namespace {
constexpr float kMinRenderScale = 0.5f;
constexpr float kMaxRenderScale = 2.0f;
constexpr CGFloat kMaxRenderDimension = 8192.0;
constexpr double kMaxRenderPixels = 16.0 * 1024.0 * 1024.0;  // ~16 MP cap for windowed mode
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;
constexpr float kCameraPitchLimitRadians = 89.9f * (kPi / 180.0f);
constexpr float kCameraSmoothingCutoffHz = 12.0f;
constexpr uint32_t kInteractiveTextureMaxDimension = 1024u;
constexpr uint64_t kInteractiveTextureBudgetMinBytes = 512ull * 1024ull * 1024ull;
constexpr uint64_t kInteractiveTextureBudgetFallbackBytes = 1024ull * 1024ull * 1024ull;
constexpr uint64_t kInteractiveTextureBudgetMaxBytes = 2ull * 1024ull * 1024ull * 1024ull;
constexpr double kInteractiveTextureBudgetFraction = 0.08;
constexpr double kInteractiveSceneWorkingSetFraction = 0.80;
constexpr uint64_t kExclusiveActiveSceneBytes = 768ull * 1024ull * 1024ull;
constexpr uint64_t kExclusiveActiveGeometryBytes = 384ull * 1024ull * 1024ull;
constexpr uint64_t kExclusiveActiveTextureBytes = 384ull * 1024ull * 1024ull;
constexpr uint32_t kExclusiveActiveTriangleCount = 2'000'000u;
constexpr uint64_t kExclusiveIncomingAssetBytes = 256ull * 1024ull * 1024ull;
constexpr uint64_t kPreflightDirectorySizeCapBytes = 2ull * 1024ull * 1024ull * 1024ull;

inline uint64_t CurrentAllocatedSizeBytes(id<MTLDevice> device) {
    if (!device) {
        return 0u;
    }
    if ([device respondsToSelector:@selector(currentAllocatedSize)]) {
        return static_cast<uint64_t>(device.currentAllocatedSize);
    }
    return 0u;
}

inline double BytesToMiB(uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

inline std::string FormatMiB(uint64_t bytes) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << BytesToMiB(bytes) << " MiB";
    return out.str();
}

inline uint64_t SaturatingAdd(uint64_t a, uint64_t b) {
    if (a > std::numeric_limits<uint64_t>::max() - b) {
        return std::numeric_limits<uint64_t>::max();
    }
    return a + b;
}

inline uint64_t DeviceRecommendedWorkingSetBytes(id<MTLDevice> device) {
    if (device && [device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
        return static_cast<uint64_t>(device.recommendedMaxWorkingSetSize);
    }
    return 0u;
}

inline uint64_t InteractiveWorkingSetLimitBytes(uint64_t recommendedBytes) {
    if (recommendedBytes == 0u) {
        return 0u;
    }
    return static_cast<uint64_t>(static_cast<double>(recommendedBytes) *
                                 kInteractiveSceneWorkingSetFraction);
}

inline PathTracer::TextureLoadPolicy MakeInteractiveTextureLoadPolicy(id<MTLDevice> device) {
    PathTracer::TextureLoadPolicy policy{};
    policy.policy = PathTracer::TextureBudgetPolicy::Warn;
    policy.pilotMode = PathTracer::PilotMode::FullClamped;
    policy.maxDimension = kInteractiveTextureMaxDimension;

    const uint64_t recommendedBytes = DeviceRecommendedWorkingSetBytes(device);
    if (recommendedBytes > 0u) {
        const uint64_t scaledBudget =
            static_cast<uint64_t>(static_cast<double>(recommendedBytes) *
                                  kInteractiveTextureBudgetFraction);
        policy.maxTextureBytes = std::clamp(scaledBudget,
                                            kInteractiveTextureBudgetMinBytes,
                                            kInteractiveTextureBudgetMaxBytes);
    } else {
        policy.maxTextureBytes = kInteractiveTextureBudgetFallbackBytes;
    }
    return policy;
}

inline bool EnvFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return false;
    }
    return std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "FALSE") != 0 &&
           std::strcmp(value, "no") != 0 &&
           std::strcmp(value, "NO") != 0;
}

inline bool SceneLoadDiagnosticsEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_SCENE_LOAD_DIAGNOSTICS");
    return enabled;
}

inline bool InteractiveExplicitRestirDiEnabled() {
    static const bool enabled = EnvFlagEnabled("PT_ENABLE_INTERACTIVE_EXPLICIT_RESTIR_DI");
    return enabled;
}

inline bool InteractiveTransportHeavySceneSwitch(const PathTracer::RenderSettings& settings) {
    const bool restirDiLighting =
        settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RestirDi ||
        settings.directLightMode == PathTracer::RenderSettings::DirectLightMode::RestirDiRegirHybrid;
    return (restirDiLighting && InteractiveExplicitRestirDiEnabled()) ||
           settings.pathGuidingMode == PathTracer::RenderSettings::PathGuidingMode::Prototype ||
           settings.radianceCacheMode == PathTracer::RenderSettings::RadianceCacheMode::Prototype ||
           settings.restirPtMode != PathTracer::RenderSettings::RestirPtMode::Off ||
           settings.restirDebugEnabled ||
           settings.svgfDenoiseEnabled;
}

std::string ToLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool TextContainsCaseInsensitive(const std::string& text, const char* needle) {
    if (!needle || !*needle) {
        return false;
    }
    return ToLowerAscii(text).find(ToLowerAscii(needle)) != std::string::npos;
}

std::string TrimAscii(std::string value) {
    auto isSpace = [](unsigned char c) {
        return std::isspace(c) != 0;
    };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [&](char ch) {
        return !isSpace(static_cast<unsigned char>(ch));
    }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](char ch) {
        return !isSpace(static_cast<unsigned char>(ch));
    }).base(), value.end());
    return value;
}

bool HasExtensionCaseInsensitive(const std::filesystem::path& path, const char* extension) {
    return TextContainsCaseInsensitive(ToLowerAscii(path.extension().string()), extension);
}

uint64_t FileSizeBytes(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || !std::filesystem::is_regular_file(path, ec)) {
        return 0u;
    }
    const uintmax_t bytes = std::filesystem::file_size(path, ec);
    if (ec) {
        return 0u;
    }
    return static_cast<uint64_t>(std::min<uintmax_t>(bytes, std::numeric_limits<uint64_t>::max()));
}

uint64_t DirectorySizeBytesCapped(const std::filesystem::path& directory, uint64_t capBytes) {
    std::error_code ec;
    if (!std::filesystem::exists(directory, ec) || !std::filesystem::is_directory(directory, ec)) {
        return 0u;
    }

    uint64_t total = 0u;
    std::filesystem::recursive_directory_iterator it(
        directory,
        std::filesystem::directory_options::skip_permission_denied,
        ec);
    std::filesystem::recursive_directory_iterator end;
    while (!ec && it != end) {
        const auto& entry = *it;
        if (entry.is_regular_file(ec)) {
            total = SaturatingAdd(total, FileSizeBytes(entry.path()));
            if (total >= capBytes) {
                return capBytes;
            }
        } else {
            ec.clear();
        }
        it.increment(ec);
    }
    return total;
}

std::string ExtractDirectiveValue(const std::string& line, const char* key) {
    const std::string keyText = key ? key : "";
    if (keyText.empty()) {
        return {};
    }
    const size_t keyPos = line.find(keyText);
    if (keyPos == std::string::npos) {
        return {};
    }
    size_t valueStart = keyPos + keyText.size();
    if (valueStart >= line.size()) {
        return {};
    }
    if (line[valueStart] == '"' || line[valueStart] == '\'') {
        const char quote = line[valueStart++];
        const size_t valueEnd = line.find(quote, valueStart);
        return line.substr(valueStart, valueEnd == std::string::npos ? std::string::npos : valueEnd - valueStart);
    }
    const size_t valueEnd = line.find_first_of(" \t\r\n", valueStart);
    return line.substr(valueStart, valueEnd == std::string::npos ? std::string::npos : valueEnd - valueStart);
}

std::filesystem::path ResolveSceneAssetPath(const std::filesystem::path& scenePath,
                                            const std::string& sceneDirectory,
                                            const std::string& assetPath) {
    namespace fs = std::filesystem;
    fs::path candidate(assetPath);
    if (candidate.is_absolute()) {
        return candidate;
    }

    std::error_code ec;
    if (!sceneDirectory.empty()) {
        fs::path fromSceneDirectory = fs::path(sceneDirectory) / candidate;
        if (fs::exists(fromSceneDirectory, ec)) {
            return fromSceneDirectory;
        }
        ec.clear();
    }

    fs::path fromSceneFile = scenePath.parent_path() / candidate;
    if (fs::exists(fromSceneFile, ec)) {
        return fromSceneFile;
    }
    return !sceneDirectory.empty() ? (fs::path(sceneDirectory) / candidate) : fromSceneFile;
}

uint64_t EstimateReferencedAssetBytes(const std::filesystem::path& assetPath) {
    const uint64_t fileBytes = FileSizeBytes(assetPath);
    if (HasExtensionCaseInsensitive(assetPath, ".gltf")) {
        return std::max(fileBytes,
                        DirectorySizeBytesCapped(assetPath.parent_path(), kPreflightDirectorySizeCapBytes));
    }
    return fileBytes;
}

struct SceneSwitchPreflight {
    uint64_t referencedAssetBytes = 0u;
    uint32_t meshReferenceCount = 0u;
    bool metadataExclusive = false;
    bool metadataLarge = false;
    bool knownLargeName = false;
};

bool SceneNameLooksLarge(const std::string& scenePath, const std::string& sceneId) {
    const std::string combined = scenePath + " " + sceneId;
    return TextContainsCaseInsensitive(combined, "bistro") ||
           TextContainsCaseInsensitive(combined, "san_miguel") ||
           TextContainsCaseInsensitive(combined, "san-miguel") ||
           TextContainsCaseInsensitive(combined, "sponza") ||
           TextContainsCaseInsensitive(combined, "hygieia") ||
           TextContainsCaseInsensitive(combined, "lucy");
}

SceneSwitchPreflight AnalyzeSceneSwitchPreflight(const std::string& scenePathString,
                                                 const std::string& sceneId,
                                                 const std::string& sceneDirectory) {
    namespace fs = std::filesystem;
    SceneSwitchPreflight preflight{};
    preflight.knownLargeName = SceneNameLooksLarge(scenePathString, sceneId);

    const fs::path scenePath(scenePathString);
    if (!HasExtensionCaseInsensitive(scenePath, ".scene")) {
        preflight.referencedAssetBytes = EstimateReferencedAssetBytes(scenePath);
        return preflight;
    }

    std::ifstream stream(scenePathString);
    if (!stream.is_open()) {
        return preflight;
    }

    std::unordered_set<std::string> seenAssets;
    std::string line;
    while (std::getline(stream, line)) {
        std::string trimmed = TrimAscii(line);
        if (trimmed.empty()) {
            continue;
        }

        const std::string lower = ToLowerAscii(trimmed);
        if (trimmed.front() == '#') {
            if (lower.find("loadpolicy=exclusive") != std::string::npos ||
                lower.find("load_policy=exclusive") != std::string::npos ||
                lower.find("scene-load=exclusive") != std::string::npos) {
                preflight.metadataExclusive = true;
            }
            if (lower.find("loadclass=large") != std::string::npos ||
                lower.find("load_class=large") != std::string::npos ||
                lower.find("large-scene") != std::string::npos) {
                preflight.metadataLarge = true;
            }
            continue;
        }

        if (lower.rfind("mesh", 0) != 0) {
            continue;
        }

        const std::string meshPath = ExtractDirectiveValue(trimmed, "path=");
        if (meshPath.empty()) {
            continue;
        }
        ++preflight.meshReferenceCount;

        const fs::path resolved = ResolveSceneAssetPath(scenePath, sceneDirectory, meshPath);
        std::error_code ec;
        const fs::path canonical = fs::weakly_canonical(resolved, ec);
        const std::string key = ec ? resolved.string() : canonical.string();
        ec.clear();
        if (!seenAssets.insert(key).second) {
            continue;
        }
        preflight.referencedAssetBytes =
            SaturatingAdd(preflight.referencedAssetBytes, EstimateReferencedAssetBytes(resolved));
    }

    return preflight;
}

struct MeshAggregateStats {
    uint32_t meshCount = 0u;
    uint64_t vertexCount = 0u;
    uint64_t indexCount = 0u;
    uint64_t triangleCount = 0u;
};

inline MeshAggregateStats CollectMeshAggregateStats(const PathTracer::SceneResources& resources) {
    MeshAggregateStats stats{};
    const auto& meshes = resources.meshes();
    stats.meshCount = static_cast<uint32_t>(meshes.size());
    for (const auto& mesh : meshes) {
        stats.vertexCount += static_cast<uint64_t>(mesh.vertices.size());
        stats.indexCount += static_cast<uint64_t>(mesh.indices.size());
    }
    stats.triangleCount = stats.indexCount / 3u;
    return stats;
}

struct GpuBufferSummary {
    uint32_t coreCount = 0u;
    uint32_t meshCount = 0u;
    uint64_t coreBytes = 0u;
    uint64_t meshBytes = 0u;
};

inline GpuBufferSummary CollectGpuBufferSummary(const PathTracer::SceneResources& resources) {
    GpuBufferSummary summary{};
    auto addCoreBuffer = [&](id<MTLBuffer> buffer) {
        if (!buffer) {
            return;
        }
        ++summary.coreCount;
        summary.coreBytes += static_cast<uint64_t>(buffer.length);
    };
    addCoreBuffer(resources.sphereBuffer());
    addCoreBuffer(resources.materialBuffer());
    addCoreBuffer(resources.rectangleBuffer());
    addCoreBuffer(resources.meshInfoBuffer());
    addCoreBuffer(resources.triangleBuffer());
    addCoreBuffer(resources.meshVertexBuffer());
    addCoreBuffer(resources.meshIndexBuffer());

    std::unordered_set<uintptr_t> seenMeshVertexBuffers;
    std::unordered_set<uintptr_t> seenMeshIndexBuffers;
    for (const auto& mesh : resources.meshes()) {
        if (mesh.vertexBuffer) {
            const uintptr_t key = reinterpret_cast<uintptr_t>(mesh.vertexBuffer);
            if (seenMeshVertexBuffers.insert(key).second) {
                ++summary.meshCount;
                summary.meshBytes += static_cast<uint64_t>(mesh.vertexBuffer.length);
            }
        }
        if (mesh.indexBuffer) {
            const uintptr_t key = reinterpret_cast<uintptr_t>(mesh.indexBuffer);
            if (seenMeshIndexBuffers.insert(key).second) {
                ++summary.meshCount;
                summary.meshBytes += static_cast<uint64_t>(mesh.indexBuffer.length);
            }
        }
    }
    return summary;
}

inline void LogCurrentAllocatedSize(const char* label, id<MTLDevice> device) {
    const uint64_t bytes = CurrentAllocatedSizeBytes(device);
    if (bytes == 0u) {
        return;
    }
    NSLog(@"[Metal] currentAllocatedSize %s %.1f MiB",
          label,
          BytesToMiB(bytes));
}

inline double CaptureDelaySeconds() {
    static const double seconds = []() {
        const char* value = std::getenv("PT_CAPTURE_DELAY_SECONDS");
        if (!value || !*value) {
            return 0.0;
        }
        char* end = nullptr;
        const double parsed = std::strtod(value, &end);
        if (end == value || !std::isfinite(parsed)) {
            return 0.0;
        }
        return std::max(0.0, parsed);
    }();
    return seconds;
}

inline void LogCommandBufferDiagnostics(id<MTLCommandBuffer> commandBuffer) {
    if (!commandBuffer) {
        return;
    }

    NSError* error = commandBuffer.error;
    if (error) {
        NSLog(@"[CB] error: %@", error);
        NSLog(@"[CB] error domain: %@", error.domain);
        NSLog(@"[CB] error code: %ld", static_cast<long>(error.code));
        NSLog(@"[CB] error userInfo: %@", error.userInfo);
    }

    if ([commandBuffer respondsToSelector:@selector(logs)]) {
        id logs = [(id)commandBuffer valueForKey:@"logs"];
        NSLog(@"[CB] logs: %@", logs);
        if ([logs respondsToSelector:@selector(count)]) {
            for (id log in logs) {
                id function = nil;
                id debugLocation = nil;
                @try {
                    function = [log valueForKey:@"function"];
                } @catch (__unused NSException* ex) {}
                @try {
                    debugLocation = [log valueForKey:@"debugLocation"];
                } @catch (__unused NSException* ex) {}
                NSLog(@"[GPULog] entry=%@", log);
                if (function || debugLocation) {
                    NSLog(@"[GPULog] function=%@ debugLocation=%@",
                          function ?: @"<nil>",
                          debugLocation ?: @"<nil>");
                }
            }
        }
    }
}

inline float ClampRenderScale(float value) {
    return std::clamp(value, kMinRenderScale, kMaxRenderScale);
}

inline float WrapRadians(float radians) {
    if (!std::isfinite(radians)) {
        return 0.0f;
    }
    float wrapped = std::fmod(radians + kPi, kTwoPi);
    if (wrapped < 0.0f) {
        wrapped += kTwoPi;
    }
    return wrapped - kPi;
}

inline float ShortestAngleDelta(float target, float current) {
    return WrapRadians(target - current);
}

inline bool IsFiniteFloat3(const simd::float3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

inline simd::float3 SafeNormalize(const simd::float3& v, const simd::float3& fallback) {
    const float lenSq = simd::dot(v, v);
    if (!(lenSq > 1.0e-12f) || !std::isfinite(lenSq)) {
        return fallback;
    }
    return v / std::sqrt(lenSq);
}

inline simd::float3 CameraForwardFromYawPitch(float yaw, float pitch) {
    const float cosPitch = std::cos(pitch);
    return simd_make_float3(-cosPitch * std::cos(yaw),
                            -std::sin(pitch),
                            -cosPitch * std::sin(yaw));
}

inline simd::float3 CameraOrbitEye(const PathTracer::RenderSettings& settings) {
    const float distance = std::max(settings.cameraDistance, 0.1f);
    const float cosPitch = std::cos(settings.cameraPitch);
    const simd::float3 offset = simd_make_float3(
        distance * cosPitch * std::cos(settings.cameraYaw),
        distance * std::sin(settings.cameraPitch),
        distance * cosPitch * std::sin(settings.cameraYaw));
    return settings.cameraTarget + offset;
}

inline bool IntersectAabb(const simd::float3& origin,
                          const simd::float3& direction,
                          const simd::float3& boundsMin,
                          const simd::float3& boundsMax,
                          float maxDistance) {
    float tMin = 0.0f;
    float tMax = maxDistance;
    for (int axis = 0; axis < 3; ++axis) {
        const float o = origin[axis];
        const float d = direction[axis];
        const float b0 = boundsMin[axis];
        const float b1 = boundsMax[axis];
        if (std::fabs(d) < 1.0e-8f) {
            if (o < b0 || o > b1) {
                return false;
            }
            continue;
        }
        float invD = 1.0f / d;
        float t0 = (b0 - o) * invD;
        float t1 = (b1 - o) * invD;
        if (t0 > t1) {
            std::swap(t0, t1);
        }
        tMin = std::max(tMin, t0);
        tMax = std::min(tMax, t1);
        if (tMax < tMin) {
            return false;
        }
    }
    return tMin <= maxDistance && tMax >= 0.0f;
}

inline bool IntersectTriangle(const simd::float3& origin,
                              const simd::float3& direction,
                              const simd::float3& v0,
                              const simd::float3& v1,
                              const simd::float3& v2,
                              float maxDistance,
                              float& outT,
                              simd::float3& outNormal) {
    const simd::float3 edge1 = v1 - v0;
    const simd::float3 edge2 = v2 - v0;
    const simd::float3 pvec = simd::cross(direction, edge2);
    const float det = simd::dot(edge1, pvec);
    if (std::fabs(det) < 1.0e-8f) {
        return false;
    }
    const float invDet = 1.0f / det;
    const simd::float3 tvec = origin - v0;
    const float u = simd::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    const simd::float3 qvec = simd::cross(tvec, edge1);
    const float v = simd::dot(direction, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    const float t = simd::dot(edge2, qvec) * invDet;
    if (!(t >= 0.0f) || t > maxDistance || !std::isfinite(t)) {
        return false;
    }
    outT = t;
    outNormal = SafeNormalize(simd::cross(edge1, edge2), simd_make_float3(0.0f, 1.0f, 0.0f));
    if (simd::dot(outNormal, direction) > 0.0f) {
        outNormal = -outNormal;
    }
    return true;
}

inline bool MakeExplicitCameraOrbitable(PathTracer::RenderSettings& settings) {
    if (!settings.explicitCameraEnabled ||
        !IsFiniteFloat3(settings.explicitCameraPosition)) {
        return false;
    }

    const simd::float3 position = settings.explicitCameraPosition;
    simd::float3 target = settings.cameraTarget;
    simd::float3 offset = position - target;
    float distanceSq = simd::dot(offset, offset);
    float distance = (distanceSq > 1.0e-12f && std::isfinite(distanceSq))
        ? std::sqrt(distanceSq)
        : 0.0f;
    bool useSceneTarget = distance > 1.0e-4f && IsFiniteFloat3(target);
    if (useSceneTarget) {
        const simd::float3 targetForward = SafeNormalize(target - position,
                                                         simd_make_float3(0.0f, 0.0f, -1.0f));
        const simd::float3 explicitForward = SafeNormalize(settings.explicitCameraForward,
                                                           targetForward);
        useSceneTarget = simd::dot(targetForward, explicitForward) > 0.999f;
    }

    if (!useSceneTarget) {
        distance = settings.cameraFocusDistance;
        if (!(distance > 1.0e-4f) || !std::isfinite(distance)) {
            distance = settings.cameraDistance;
        }
        if (!(distance > 1.0e-4f) || !std::isfinite(distance)) {
            distance = 1.0f;
        }

        const simd::float3 forward =
            SafeNormalize(settings.explicitCameraForward, simd_make_float3(0.0f, 0.0f, -1.0f));
        target = position + forward * distance;
        offset = position - target;
    }

    const float horizontalDistance =
        std::sqrt(std::max(offset.x * offset.x + offset.z * offset.z, 0.0f));
    settings.cameraTarget = target;
    settings.cameraDistance = std::max(distance, 0.1f);
    settings.cameraYaw = WrapRadians(std::atan2(offset.z, offset.x));
    settings.cameraPitch = std::clamp(std::atan2(offset.y, horizontalDistance),
                                      -kCameraPitchLimitRadians,
                                      kCameraPitchLimitRadians);
    settings.firstPersonCameraPosition = position;
    settings.explicitCameraEnabled = false;
    return true;
}

inline const char* WorkingColorSpaceLabel(const PathTracer::RenderSettings& settings) {
    return (settings.workingColorSpace == PathTracer::RenderSettings::WorkingColorSpace::ACEScg)
        ? "ACEScg"
        : "Linear sRGB";
}

float CpuGgxG1(float alpha, float cosTheta) {
    float a2 = alpha * alpha;
    float c2 = cosTheta * cosTheta;
    return 2.0f * cosTheta /
           (cosTheta + std::sqrt(std::max(a2 + (1.0f - a2) * c2, 1.0e-6f)));
}

float CpuGgxDistribution(float alpha, float cosThetaH) {
    float a2 = alpha * alpha;
    float c2 = cosThetaH * cosThetaH;
    float denom = c2 * (a2 - 1.0f) + 1.0f;
    return a2 / (static_cast<float>(M_PI) * denom * denom + 1.0e-6f);
}

simd::float3 CpuSchlickFresnel(const simd::float3& f0, float cosTheta) {
    float m = std::clamp(1.0f - cosTheta, 0.0f, 1.0f);
    float m2 = m * m;
    float weight = m2 * m2 * m;
    return f0 + (simd_make_float3(1.0f, 1.0f, 1.0f) - f0) * weight;
}
}  // namespace

// DEBUG: Render path tracer at fixed internal resolution, independent of drawable size
// Set to 1 to enable fixed 1280x720 internal rendering while keeping UI at full resolution
#ifndef DEBUG_FIXED_RENDER_RESOLUTION
#define DEBUG_FIXED_RENDER_RESOLUTION 0
#endif

#if DEBUG_FIXED_RENDER_RESOLUTION
constexpr uint32_t kDebugRenderWidth = 1280;
constexpr uint32_t kDebugRenderHeight = 720;
#endif

using PathTracerShaderTypes::MaterialType;
using PathTracerShaderTypes::kMaxMaterials;
using PathTracerShaderTypes::kMaxSpheres;
using PathTracerShaderTypes::PathtraceStats;
using PathTracerShaderTypes::PathtraceDebugBuffer;
using PathTracerShaderTypes::QueueCounters;
using PathTracerShaderTypes::WavefrontQueueHeader;
using PathTracer::RenderSettings;
using PathTracer::ScenePanelProvider;
using PathTracer::MaterialDisplayInfo;
using PathTracer::ObjectDisplayInfo;
using PathTracer::FrameResult;
using PathTracer::PerformanceStats;
using PathTracer::SceneManager;
using PathTracer::EnvGpuHandles;
using PathTracer::HistoryResetReason;

class MetalRenderer::Impl : public std::enable_shared_from_this<Impl> {
public:
    bool initialize(NSWindow* window, const MetalRendererOptions& options);
    void drawFrame();
    void resize(int width, int height);
    void resetAccumulation();
    void setTonemapMode(uint32_t mode);
    void setPresentationEnabled(bool enabled);
    bool presentationEnabled() const { return m_presentationSettings.enabled; }
    void togglePresentationUIPanels();
    void shutdown();
    bool exportToPPM(const char* filepath);
    bool loadScene(const std::string& identifier);
    bool loadSceneFromPath(const std::string& path);
    bool useSceneResources(PathTracer::SceneResources& resources, const std::string& identifier);
    std::vector<std::string> sceneIdentifiers() const;
    RenderSettings currentSettings() const { return m_settings; }
    void applySettings(const RenderSettings& settings, bool resetAccumulation);
    void setSamplesPerFrame(uint32_t samples) { m_settings.samplesPerFrame = std::max<uint32_t>(1u, samples); }
    uint32_t sampleCount() const { return m_accumulation.sampleCount(); }
    bool captureAverageImage(std::vector<float>& outLinearRGB,
                             uint32_t& width,
                             uint32_t& height,
                             std::vector<float>* outSampleCounts);
    bool captureRawImage(std::vector<float>& outLinearRGB,
                         uint32_t& width,
                         uint32_t& height,
                         std::vector<float>* outSampleCounts);
    bool captureAuxiliaryImages(std::vector<float>* outAlbedoRGB,
                                std::vector<float>* outNormalRGB,
                                uint32_t& width,
                                uint32_t& height);
    RuntimeStatsSnapshot runtimeStatsSnapshot() const;
    PbrMetricsSnapshot pbrMetricsSnapshot() const { return m_pbrMetricsSnapshot; }
    bool captureMaterialMrDebugDump(HeadlessRenderOutput::MaterialMrDebugDump& outDump) const;
    bool captureDirectLightAuditDump(HeadlessRenderOutput::DirectLightAuditDump& outDump) const;
    ~Impl();

private:
    struct SceneLoadRequest;
    struct SceneLoadCompleted;

    PathTracer::SceneResources& activeSceneResources();
    const PathTracer::SceneResources& activeSceneResources() const;
    CGSize targetRenderSize();
    bool initializeOverlay();
    void updatePerformanceStats(MTLCommandBufferHandle commandBuffer,
                                MTLBufferHandle statsBuffer,
                                MTLBufferHandle queueCountersBuffer,
                                MTLBufferHandle debugBuffer,
                                CGSize renderSize,
                                uint32_t finalSampleCount,
                                const FrameResult& frameResult);
    void handleCameraInput();
    void rebuildCameraCollisionCacheIfNeeded();
    bool traceCameraCollision(const simd::float3& origin,
                              const simd::float3& direction,
                              float maxDistance,
                              float& outDistance,
                              simd::float3& outNormal) const;
    bool moveFirstPersonCamera(const simd::float3& displacement, bool snapToWalkableGround = false);
    bool moveFirstPersonVertical(const simd::float3& displacement);
    void resetFirstPersonMotion();
    void setupDefaultScene();
    void buildProceduralScene();
    uint32_t addMaterial(const simd::float3& baseColor,
                         float roughness,
                         MaterialType type,
                         float indexOfRefraction,
                         const simd::float3& conductorEta = simd_make_float3(0.0f, 0.0f, 0.0f),
                         const simd::float3& conductorK = simd_make_float3(0.0f, 0.0f, 0.0f),
                         bool hasConductorParameters = false);
    void addSphere(const simd::float3& center, float radius, uint32_t materialIndex);
    void handleSaveExrRequest(const std::string& path);
    void updateDrawableSize();
    void applyPresentationSettings(bool force);
    NSScreen* resolvePresentationScreen(const PathTracer::PresentationSettings& settings) const;
    void noteWindowInteraction();
    bool isMouseInsideRenderView() const;
    void applyBackgroundSettings();
    bool syncEnvironmentMap(bool* outChanged = nullptr);
    bool captureTextureImage(MTLTextureHandle sourceTexture,
                             std::vector<float>& outLinearRGB,
                             uint32_t& width,
                             uint32_t& height,
                             std::vector<float>* outAlpha = nullptr,
                             bool decodeNormal = false);
    void syncCameraSmoothingToTarget();
    void updateCameraSmoothing(float deltaSeconds);
    void evaluateAccumulationState();
    void serviceAccumulationReset();
    bool pathTraceFrameInFlight() const;
    void syncRadiometricBaseline();
    void serviceSceneLoadState(bool allowGpuWork);
    bool sceneLoadUiOnlyInProgress() const;
    bool shouldUseExclusiveSceneSwitch(const std::string& scenePath,
                                       const std::string& activeSceneId,
                                       const RenderSettings& startingSettings,
                                       std::string* reason) const;
    void releaseActiveSceneForExclusiveLoad(const std::string& nextSceneId);
    bool queueSceneLoadFromResolvedPath(const std::string& scenePath,
                                        const std::string& activeSceneId,
                                        const RenderSettings& startingSettings);
    void startQueuedSceneLoadIfPossible();
    void launchSceneLoadWorker(SceneLoadRequest request, uint64_t generation);
    void handleSceneLoadWorkerCompletion(uint64_t generation,
                                         std::shared_ptr<SceneLoadCompleted> completion);
    void cancelPendingSceneLoad();
    MetalRendererOptions m_options{};

    PathTracer::MetalContext m_context{};
    PathTracer::Pipelines m_pipelines{};
    PathTracer::Accumulation m_accumulation{};
    PathTracer::SceneResources m_sceneResources{};
    std::unique_ptr<PathTracer::SceneResources> m_loadedSceneResources;
    PathTracer::SceneResources* m_externalSceneResources = nullptr;
    PathTracer::SceneManager m_sceneManager{};
    PathTracer::DenoiserContext m_denoiser{};
    PathTracer::RenderLoop m_renderLoop{};
    PathTracer::RenderSettings m_settings{};
    PathTracer::UIOverlay m_overlay{};
    PathTracer::PresentationSettings m_presentationSettings{};
    PathTracer::PerformanceStats m_performanceStats{};
    uint32_t m_lastDebugLogCount = 0;
    uint32_t m_lastParityLogCount = 0;
    std::string m_activeSceneId{};
    std::vector<std::string> m_sceneLabels{};
    std::vector<const char*> m_sceneLabelPointers{};
    std::vector<std::string> m_sceneIdentifiers{};

    bool m_overlayInitialized = false;
    bool m_shutdown = false;

    float m_contentScale = 1.0f;
    id m_backingChangeObserver = nil;
    id m_windowResizeObserver = nil;
    id m_windowMoveObserver = nil;
    id m_windowWillMoveObserver = nil;
    double m_cameraInputSuppressionDeadline = 0.0;
    bool m_cameraOrbitActive = false;
    bool m_cameraZoomActive = false;
    bool m_firstPersonLookActive = false;
    bool m_cameraMotionActive = false;
    float m_smoothYaw = 0.0f;
    float m_smoothPitch = 0.0f;
    simd::float3 m_firstPersonVelocity = {0.0f, 0.0f, 0.0f};
    float m_firstPersonEyeHeight = 1.6f;
    bool m_firstPersonEyeHeightInitialized = false;
    bool m_cameraSmoothingInitialized = false;
    CFAbsoluteTime m_lastCameraInteractionTime = 0.0;

    CFAbsoluteTime m_lastFrameTimestamp = 0.0;
    double m_lastFrameTimeMs = 0.0;
    uint32_t m_previousSampleCount = 0;
    uint32_t m_activeSamplesPerFrame = 0;
    double m_lastDrawableWaitMs = 0.0;
    double m_lastCpuEncodeMs = 0.0;
    float m_currentRenderScale = 1.0f;
    bool m_hasRadiometricBaseline = false;
    bool m_accumDirty = false;
    std::string m_accumDirtyReason;
    bool m_deferredAccumulationReset = false;
    RenderSettings m_radiometricBaseline{};
    struct PresentationRestoreState {
        bool valid = false;
        NSRect frame = NSZeroRect;
        NSWindowStyleMask styleMask = 0;
        NSInteger level = 0;
        NSApplicationPresentationOptions appOptions = 0;
        bool overlayMainVisible = true;
        bool overlayMinimalVisible = false;
        bool forcedContentScaleEnabled = false;
        float forcedContentScale = 1.0f;
        uint32_t renderWidth = 0;
        uint32_t renderHeight = 0;
        float renderScale = 1.0f;
    };
    PresentationRestoreState m_presentationRestore{};
    PathTracer::PresentationSettings m_presentationAppliedSettings{};
    bool m_presentationActive = false;
    bool m_presentationRenderLockApplied = false;
    CGSize m_viewSizePoints = CGSizeMake(1.0, 1.0);
    CGSize m_drawablePixelSize = CGSizeMake(1.0, 1.0);
    bool m_renderClampWarningLogged = false;
    bool m_renderPixelClampLogged = false;
    bool m_verboseTiming = false;
    bool m_firstDispatchTimingLogged = false;
    bool m_firstStatsTimingLogged = false;
    bool m_mneeAccountingLogged = false;
    PbrMetricsSnapshot m_pbrMetricsSnapshot{};
    CFAbsoluteTime m_lastCommandBufferFailureLogTime = 0.0;
    std::atomic_bool m_pathTraceFrameInFlight{false};
    std::atomic_bool m_interactiveUiFrameInFlight{false};
    struct CameraCollisionMesh {
        const PathTracer::SceneResources::Mesh* mesh = nullptr;
        simd::float3 localBoundsMin = {0.0f, 0.0f, 0.0f};
        simd::float3 localBoundsMax = {0.0f, 0.0f, 0.0f};
    };
    const PathTracer::SceneResources* m_cameraCollisionSource = nullptr;
    size_t m_cameraCollisionMeshCount = 0;
    std::vector<CameraCollisionMesh> m_cameraCollisionMeshes;
    std::vector<PathTracerShaderTypes::TriangleData> m_cameraCollisionTriangles;
    PathTracer::BvhBuildOutput m_cameraCollisionBvh{};

    struct SceneLoadRequest {
        std::string scenePath;
        std::string activeSceneId;
        RenderSettings startingSettings{};
        bool exclusive = false;
    };

    struct SceneLoadCompleted {
        bool success = false;
        std::unique_ptr<PathTracer::SceneResources> resources;
        RenderSettings settings{};
        std::string activeSceneId;
        std::string errorMessage;
        bool exclusive = false;
    };

    enum class SceneLoadStage : uint32_t {
        Idle = 0,
        LoadScene = 1,
        Activate = 2,
        Failed = 3,
    };

    std::optional<SceneLoadRequest> m_queuedSceneLoad;
    std::optional<SceneLoadCompleted> m_pendingSceneLoadResult;
    SceneLoadStage m_sceneLoadStage = SceneLoadStage::Idle;
    std::shared_ptr<std::atomic<uint64_t>> m_sceneLoadGeneration =
        std::make_shared<std::atomic<uint64_t>>(0u);
    dispatch_queue_t m_sceneLoadWorkerQueue = nullptr;
    bool m_sceneLoadWorkerActive = false;
    bool m_sceneLoadExclusiveActive = false;
    MTLCommandBufferHandle m_sceneSwapDrainCommandBuffer = nil;
    CFAbsoluteTime m_sceneSwapDrainStartTime = 0.0;
    bool m_sceneSwapDrainWarningLogged = false;
};

bool MetalRenderer::Impl::initialize(NSWindow* window, const MetalRendererOptions& options) {
    m_options = options;
    m_settings.fixedRngSeed = options.fixedRngSeed;
    m_settings.enableSoftwareRayTracing = options.enableSoftwareRayTracing;
    m_verboseTiming = options.headless && options.verbose;
    m_presentationSettings.enabled = options.presentationMode;

#if PT_MNEE_SWRT_RAYS
    NSLog(@"[Warning] PT_MNEE_SWRT_RAYS enabled: MNEE rays use SWRT (slow diagnostic mode)");
#endif

    if (!options.headless && !window) {
        return false;
    }

    if (!m_context.initialize(window, options.headless)) {
        return false;
    }

    if (!m_sceneLoadWorkerQueue) {
        dispatch_queue_attr_t attr =
            dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL,
                                                    QOS_CLASS_USER_INITIATED,
                                                    0);
        m_sceneLoadWorkerQueue =
            dispatch_queue_create("com.dariopagliaricci.PathTracer.SceneLoadWorker", attr);
    }

    m_accumulation.initialize(m_context.device());
    m_sceneResources.initialize(m_context);
    m_sceneResources.setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);

    if (!m_options.headless) {
        Impl* selfPtr = this;
        m_backingChangeObserver = [[NSNotificationCenter defaultCenter]
            addObserverForName:NSWindowDidChangeBackingPropertiesNotification
                        object:m_context.window()
                         queue:[NSOperationQueue mainQueue]
                    usingBlock:^(__unused NSNotification* note) {
                        if (selfPtr) {
                            selfPtr->updateDrawableSize();
                        }
                    }];
        m_context.setBackingChangeObserver(m_backingChangeObserver);

        // Update drawable size when the window resizes (autoResizeDrawable is disabled).
        m_windowResizeObserver = [[NSNotificationCenter defaultCenter]
            addObserverForName:NSWindowDidResizeNotification
                        object:m_context.window()
                         queue:[NSOperationQueue mainQueue]
                    usingBlock:^(__unused NSNotification* note) {
                        if (selfPtr) {
                            selfPtr->updateDrawableSize();
                        }
                    }];

        m_windowMoveObserver = [[NSNotificationCenter defaultCenter]
            addObserverForName:NSWindowDidMoveNotification
                        object:m_context.window()
                         queue:[NSOperationQueue mainQueue]
                    usingBlock:^(__unused NSNotification* note) {
                        if (selfPtr) {
                            selfPtr->noteWindowInteraction();
                        }
                    }];

        m_windowWillMoveObserver = [[NSNotificationCenter defaultCenter]
            addObserverForName:NSWindowWillMoveNotification
                        object:m_context.window()
                         queue:[NSOperationQueue mainQueue]
                    usingBlock:^(__unused NSNotification* note) {
                        if (selfPtr) {
                            selfPtr->noteWindowInteraction();
                        }
                    }];
    }

    updateDrawableSize();

    MTLPixelFormat displayFormat = MTLPixelFormatBGRA8Unorm;
    if (!m_options.headless) {
        MTKView* view = m_context.view();
        if (!view) {
            return false;
        }
        displayFormat = view.colorPixelFormat;
    }

    MTLDeviceHandle device = m_context.device();
    if (!device) {
        return false;
    }

#if PATH_TRACER_ENABLE_OIDN
    if (!m_denoiser.initialize(device, m_context.commandQueue())) {
        NSLog(@"[Denoise] OIDN init failed: %s", m_denoiser.lastError().c_str());
    }
#endif

    if (!m_renderLoop.initialize(m_context, &m_denoiser)) {
        return false;
    }

    m_pipelines.setVerboseTiming(m_verboseTiming);
    if (!m_pipelines.initialize(m_context, displayFormat)) {
        return false;
    }

#if HAS_IMGUI
    if (!m_options.headless) {
        m_overlayInitialized = initializeOverlay();
        if (!m_overlayInitialized) {
            return false;
        }
    }
#endif

    if (!m_options.headless) {
        setupDefaultScene();
        applyPresentationSettings(true);
        resetAccumulation();
    }
    return true;
}

void MetalRenderer::Impl::resize(int width, int height) {
    if (m_options.headless) {
        return;
    }

    MTKView* view = m_context.view();
    if (!view) {
        return;
    }

    NSRect frame = view.frame;
    frame.size.width = std::max(width, 1);
    frame.size.height = std::max(height, 1);
    [view setFrame:frame];
    updateDrawableSize();
}

void MetalRenderer::Impl::updateDrawableSize() {
    if (m_options.headless) {
        return;
    }

    m_context.updateDrawableSize();
    m_contentScale = m_context.contentScale();

    MTKView* view = m_context.view();
    if (!view) {
        return;
    }

    m_viewSizePoints = view.bounds.size;
    CGSize drawable = m_context.drawableSize();
    if (drawable.width >= 1.0f && drawable.height >= 1.0f) {
        m_drawablePixelSize = drawable;
    }

#if HAS_IMGUI
    m_overlay.updateDisplayMetrics(m_viewSizePoints, m_contentScale);
#endif
}

NSScreen* MetalRenderer::Impl::resolvePresentationScreen(
    const PathTracer::PresentationSettings& settings) const {
    NSWindow* window = m_context.window();
    NSScreen* current = window ? window.screen : nil;
    NSScreen* main = [NSScreen mainScreen];
    NSArray<NSScreen*>* screens = [NSScreen screens];

    auto findExternal = ^NSScreen*() {
        for (NSScreen* screen in screens) {
            if (screen && screen != main) {
                return screen;
            }
        }
        return (NSScreen*)nil;
    };

    switch (settings.targetScreen) {
        case PathTracer::PresentationSettings::TargetScreen::Primary:
            return main ?: current;
        case PathTracer::PresentationSettings::TargetScreen::External: {
            NSScreen* external = findExternal();
            return external ?: (current ?: main);
        }
        case PathTracer::PresentationSettings::TargetScreen::Auto:
        default: {
            NSScreen* external = findExternal();
            return external ?: (current ?: main);
        }
    }
}

void MetalRenderer::Impl::applyPresentationSettings(bool force) {
    if (m_options.headless) {
        return;
    }

    NSWindow* window = m_context.window();
    if (!window) {
        return;
    }

    const PathTracer::PresentationSettings& settings = m_presentationSettings;

    auto applyOverlayVisibility = [&]() {
#if HAS_IMGUI
        if (m_overlayInitialized) {
            m_overlay.setMainPanelVisible(!settings.hideUIPanels);
            m_overlay.setMinimalOverlayVisible(settings.minimalOverlay);
        }
#endif
    };

    auto applyContentScale = [&]() {
        switch (settings.contentScale) {
            case PathTracer::PresentationSettings::ContentScaleMode::Scale1x:
                m_context.setForcedContentScale(1.0f, true);
#if HAS_IMGUI
                if (m_overlayInitialized) {
                    m_overlay.setForcedContentScale(1.0f, true);
                }
#endif
                break;
            case PathTracer::PresentationSettings::ContentScaleMode::Scale2x:
                m_context.setForcedContentScale(2.0f, true);
#if HAS_IMGUI
                if (m_overlayInitialized) {
                    m_overlay.setForcedContentScale(2.0f, true);
                }
#endif
                break;
            case PathTracer::PresentationSettings::ContentScaleMode::Auto:
            default:
                m_context.setForcedContentScale(1.0f, false);
#if HAS_IMGUI
                if (m_overlayInitialized) {
                    m_overlay.setForcedContentScale(1.0f, false);
                }
#endif
                break;
        }
    };

    auto applyWindowPresentation = [&]() {
        NSScreen* target = resolvePresentationScreen(settings);
        if (!target) {
            target = window.screen ?: [NSScreen mainScreen];
        }
        if (!target) {
            return;
        }

        bool borderless =
            settings.windowMode == PathTracer::PresentationSettings::WindowMode::BorderlessFullscreen;
        NSRect targetFrame = borderless ? target.frame : target.visibleFrame;
        if (targetFrame.size.width < 1.0 || targetFrame.size.height < 1.0) {
            borderless = false;
            targetFrame = target.visibleFrame;
        }

        if (borderless) {
            [window setStyleMask:NSWindowStyleMaskBorderless];
            [window setLevel:NSMainMenuWindowLevel + 1];
            const NSApplicationPresentationOptions options =
                m_presentationRestore.appOptions |
                NSApplicationPresentationHideDock |
                NSApplicationPresentationHideMenuBar;
            [NSApp setPresentationOptions:options];
        } else {
            NSWindowStyleMask restoredMask = m_presentationRestore.styleMask;
            if ((restoredMask & NSWindowStyleMaskTitled) == 0) {
                restoredMask = NSWindowStyleMaskTitled |
                               NSWindowStyleMaskClosable |
                               NSWindowStyleMaskMiniaturizable |
                               NSWindowStyleMaskResizable;
            }
            [window setStyleMask:restoredMask];
            [window setLevel:m_presentationRestore.level];
            [NSApp setPresentationOptions:m_presentationRestore.appOptions];
        }

        [window setFrame:targetFrame display:YES];
        updateDrawableSize();
    };

    auto applyResolutionLock = [&](PathTracer::PresentationSettings::RenderResolutionLock lockMode) {
        switch (lockMode) {
            case PathTracer::PresentationSettings::RenderResolutionLock::Lock1280x720:
                m_settings.renderWidth = 1280u;
                m_settings.renderHeight = 720u;
                m_settings.renderScale = 1.0f;
                break;
            case PathTracer::PresentationSettings::RenderResolutionLock::Lock1920x1080:
                m_settings.renderWidth = 1920u;
                m_settings.renderHeight = 1080u;
                m_settings.renderScale = 1.0f;
                break;
            case PathTracer::PresentationSettings::RenderResolutionLock::Off:
            default:
                break;
        }
    };

    if (!m_presentationActive && !settings.enabled) {
        m_presentationAppliedSettings = settings;
        return;
    }

    if (!m_presentationActive && settings.enabled) {
        m_presentationRestore.valid = true;
        m_presentationRestore.frame = window.frame;
        m_presentationRestore.styleMask = window.styleMask;
        m_presentationRestore.level = window.level;
        m_presentationRestore.appOptions = [NSApp presentationOptions];
        m_presentationRestore.forcedContentScaleEnabled = m_context.hasForcedContentScale();
        m_presentationRestore.forcedContentScale = m_context.forcedContentScale();
        m_presentationRestore.renderWidth = m_settings.renderWidth;
        m_presentationRestore.renderHeight = m_settings.renderHeight;
        m_presentationRestore.renderScale = m_settings.renderScale;
#if HAS_IMGUI
        if (m_overlayInitialized) {
            m_presentationRestore.overlayMainVisible = m_overlay.isMainPanelVisible();
            m_presentationRestore.overlayMinimalVisible = m_overlay.isMinimalOverlayVisible();
        }
#endif

        m_presentationActive = true;
        m_presentationRenderLockApplied = false;
        applyOverlayVisibility();
        applyContentScale();
        applyWindowPresentation();

        bool resetIssued = false;
        if (settings.resolutionLock !=
            PathTracer::PresentationSettings::RenderResolutionLock::Off) {
            applyResolutionLock(settings.resolutionLock);
            m_presentationRenderLockApplied = true;
            resetAccumulation();
            resetIssued = true;
        }

        if (settings.resetAccumulationOnToggle && !resetIssued) {
            resetAccumulation();
        }

        m_presentationAppliedSettings = settings;
        return;
    }

    if (m_presentationActive && !settings.enabled) {
        if (m_presentationRestore.valid) {
            [window setStyleMask:m_presentationRestore.styleMask];
            [window setLevel:m_presentationRestore.level];
            [NSApp setPresentationOptions:m_presentationRestore.appOptions];
            [window setFrame:m_presentationRestore.frame display:YES];
        }

#if HAS_IMGUI
        if (m_overlayInitialized) {
            m_overlay.setMainPanelVisible(m_presentationRestore.overlayMainVisible);
            m_overlay.setMinimalOverlayVisible(m_presentationRestore.overlayMinimalVisible);
        }
#endif

        if (m_presentationRestore.forcedContentScaleEnabled) {
            m_context.setForcedContentScale(m_presentationRestore.forcedContentScale, true);
#if HAS_IMGUI
            if (m_overlayInitialized) {
                m_overlay.setForcedContentScale(m_presentationRestore.forcedContentScale, true);
            }
#endif
        } else {
            m_context.setForcedContentScale(1.0f, false);
#if HAS_IMGUI
            if (m_overlayInitialized) {
                m_overlay.setForcedContentScale(1.0f, false);
            }
#endif
        }

        if (m_presentationRenderLockApplied && m_presentationRestore.valid) {
            m_settings.renderWidth = m_presentationRestore.renderWidth;
            m_settings.renderHeight = m_presentationRestore.renderHeight;
            m_settings.renderScale = m_presentationRestore.renderScale;
            m_presentationRenderLockApplied = false;
        }

        updateDrawableSize();

        if (settings.resetAccumulationOnToggle) {
            resetAccumulation();
        }

        m_presentationActive = false;
        m_presentationAppliedSettings = settings;
        return;
    }

    bool needsWindowUpdate =
        force ||
        settings.windowMode != m_presentationAppliedSettings.windowMode ||
        settings.targetScreen != m_presentationAppliedSettings.targetScreen;
    if (needsWindowUpdate) {
        applyWindowPresentation();
    }

    if (force ||
        settings.hideUIPanels != m_presentationAppliedSettings.hideUIPanels ||
        settings.minimalOverlay != m_presentationAppliedSettings.minimalOverlay) {
        applyOverlayVisibility();
    }

    if (force || settings.contentScale != m_presentationAppliedSettings.contentScale) {
        applyContentScale();
    }

    if (force || settings.resolutionLock != m_presentationAppliedSettings.resolutionLock) {
        if (settings.resolutionLock ==
            PathTracer::PresentationSettings::RenderResolutionLock::Off) {
            if (m_presentationRenderLockApplied && m_presentationRestore.valid) {
                m_settings.renderWidth = m_presentationRestore.renderWidth;
                m_settings.renderHeight = m_presentationRestore.renderHeight;
                m_settings.renderScale = m_presentationRestore.renderScale;
                m_presentationRenderLockApplied = false;
            }
        } else {
            if (!m_presentationRenderLockApplied) {
                m_presentationRestore.renderWidth = m_settings.renderWidth;
                m_presentationRestore.renderHeight = m_settings.renderHeight;
                m_presentationRestore.renderScale = m_settings.renderScale;
            }
            applyResolutionLock(settings.resolutionLock);
            m_presentationRenderLockApplied = true;
        }
        resetAccumulation();
    }

    m_presentationAppliedSettings = settings;
}

void MetalRenderer::Impl::noteWindowInteraction() {
    constexpr double kCameraSuppressionSeconds = 0.35;
    const double deadline = CFAbsoluteTimeGetCurrent() + kCameraSuppressionSeconds;
    m_cameraInputSuppressionDeadline = std::max(m_cameraInputSuppressionDeadline, deadline);
    m_cameraOrbitActive = false;
    m_cameraZoomActive = false;
    m_firstPersonLookActive = false;
}

void MetalRenderer::Impl::resetFirstPersonMotion() {
    m_firstPersonVelocity = simd_make_float3(0.0f, 0.0f, 0.0f);
    m_firstPersonEyeHeightInitialized = false;
    m_firstPersonLookActive = false;
}

bool MetalRenderer::Impl::isMouseInsideRenderView() const {
    if (m_options.headless) {
        return false;
    }
    NSWindow* window = m_context.window();
    MTKView* view = m_context.view();
    if (!window || !view) {
        return false;
    }

    NSRect viewBounds = [view bounds];
    NSRect viewRectInWindow = [view convertRect:viewBounds toView:nil];
    NSRect viewRectOnScreen = [window convertRectToScreen:viewRectInWindow];
    NSPoint screenLocation = [NSEvent mouseLocation];
    return NSPointInRect(screenLocation, viewRectOnScreen);
}

void MetalRenderer::Impl::drawFrame() {
    serviceSceneLoadState(false);

    id<MTLCommandQueue> commandQueue = m_context.commandQueue();
    if (!commandQueue) {
        return;
    }

    if (!m_pipelines.integrate() || !m_pipelines.present()) {
        return;
    }

    // Render pipeline only needed for windowed mode
    if (!m_options.headless) {
        MTKView* view = m_context.view();
        if (!view || !m_pipelines.display()) {
            return;
        }
    }

    if (!m_options.headless &&
        m_pathTraceFrameInFlight.load(std::memory_order_acquire) &&
        m_interactiveUiFrameInFlight.load(std::memory_order_acquire)) {
        return;
    }

    CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
    if (m_lastFrameTimestamp > 0.0) {
        m_lastFrameTimeMs = (now - m_lastFrameTimestamp) * 1000.0;
    }
    m_lastFrameTimestamp = now;

    MTLRenderPassDescriptor* renderPassDescriptor = nil;
    id<CAMetalDrawable> drawable = nil;
    CGSize renderSize = CGSizeZero;
    std::string pendingSavePath;

    if (!m_options.headless) {
        MTKView* view = m_context.view();
        if (!view) {
            return;
        }

        CFAbsoluteTime drawableWaitStart = CFAbsoluteTimeGetCurrent();
        CAMetalLayer* metalLayer = (CAMetalLayer*)view.layer;
        if (!metalLayer || ![metalLayer isKindOfClass:[CAMetalLayer class]]) {
            return;
        }

        drawable = [metalLayer nextDrawable];
        CFAbsoluteTime drawableWaitEnd = CFAbsoluteTimeGetCurrent();
        m_lastDrawableWaitMs = (drawableWaitEnd - drawableWaitStart) * 1000.0;
        if (!drawable) {
            return;
        }

        renderPassDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
        renderPassDescriptor.colorAttachments[0].texture = drawable.texture;
        renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
        renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
        renderPassDescriptor.colorAttachments[0].clearColor = view.clearColor;

#if HAS_IMGUI
        if (!m_overlayInitialized) {
            m_overlayInitialized = initializeOverlay();
        }
        if (m_overlayInitialized) {
            m_overlay.beginFrame(renderPassDescriptor);

            bool uiRequestedReset = false;

            m_sceneLabels.clear();
            m_sceneLabelPointers.clear();
            m_sceneIdentifiers.clear();
            int currentSceneIndex = 0;

            if (m_sceneManager.refresh(nullptr)) {
                const auto& scenes = m_sceneManager.scenes();
                m_sceneLabels.reserve(scenes.size() + 1);
                m_sceneIdentifiers.reserve(scenes.size() + 1);
                m_sceneLabelPointers.reserve(scenes.size() + 1);

                m_sceneLabels.emplace_back("Procedural (Randomized RTOW)");
                m_sceneLabelPointers.push_back(m_sceneLabels.back().c_str());
                m_sceneIdentifiers.emplace_back("");

                currentSceneIndex = 0;
                bool matchedActiveScene = false;
                for (size_t i = 0; i < scenes.size(); ++i) {
                    const auto& info = scenes[i];
                    std::string label;
                    if (!info.displayName.empty()) {
                        label = info.displayName + " [" + info.identifier + "]";
                    } else {
                        label = info.identifier;
                    }
                    m_sceneLabels.push_back(std::move(label));
                    m_sceneLabelPointers.push_back(m_sceneLabels.back().c_str());
                    m_sceneIdentifiers.push_back(info.identifier);
                    if (!m_activeSceneId.empty() && info.identifier == m_activeSceneId) {
                        currentSceneIndex = static_cast<int>(m_sceneIdentifiers.size() - 1);
                        matchedActiveScene = true;
                    }
                }
                if (!m_activeSceneId.empty() && !matchedActiveScene) {
                    m_sceneLabels.emplace_back("External Scene [" + m_activeSceneId + "]");
                    m_sceneLabelPointers.push_back(m_sceneLabels.back().c_str());
                    m_sceneIdentifiers.push_back(m_activeSceneId);
                    currentSceneIndex = static_cast<int>(m_sceneIdentifiers.size() - 1);
                }
            } else {
                m_sceneLabels.emplace_back("Procedural (Randomized RTOW)");
                m_sceneLabelPointers.push_back(m_sceneLabels.back().c_str());
                m_sceneIdentifiers.emplace_back("");
            }

            ScenePanelProvider panelProvider{};
            panelProvider.materialCount = [this]() -> uint32_t {
                return activeSceneResources().materialCount();
            };
            panelProvider.readMaterial = [this](uint32_t index, MaterialDisplayInfo& info) -> bool {
                auto& sceneResources = activeSceneResources();
                if (index >= sceneResources.materialCount()) {
                    return false;
                }
                info.name = sceneResources.materialName(index);
                const PathTracerShaderTypes::MaterialData* materials = sceneResources.materialsData();
                if (!materials) {
                    return false;
                }
                info.data = materials[index];
                return true;
            };
            panelProvider.applyMaterial = [this](uint32_t index, const MaterialDisplayInfo& info) {
                if (activeSceneResources().updateMaterial(index, info.data)) {
                    m_accumDirty = true;
                    m_accumDirtyReason = "MATERIAL_EDIT";
                }
            };
            panelProvider.resetMaterial = [this](uint32_t index) {
                if (activeSceneResources().resetMaterial(index)) {
                    m_accumDirty = true;
                    m_accumDirtyReason = "MATERIAL_RESET";
                }
            };

            panelProvider.objectCount = [this]() -> uint32_t {
                return static_cast<uint32_t>(activeSceneResources().meshes().size());
            };
            panelProvider.readObject = [this](uint32_t index, ObjectDisplayInfo& info) -> bool {
                const auto& meshes = activeSceneResources().meshes();
                if (index >= meshes.size()) {
                    return false;
                }
                const auto& mesh = meshes[index];
                info.name = mesh.name;
                info.meshIndex = index;
                info.materialIndex = mesh.materialIndex;
                info.transform = mesh.localToWorld;
                return true;
            };
            panelProvider.applyObjectTransform = [this](uint32_t meshIndex, const ObjectDisplayInfo& info) {
                if (activeSceneResources().setMeshTransform(meshIndex, info.transform)) {
                    m_cameraCollisionSource = nullptr;
                    m_accumDirty = true;
                    m_accumDirtyReason = "OBJECT_TRANSFORM";
                }
            };
            panelProvider.resetObjectTransform = [this](uint32_t meshIndex) {
                if (activeSceneResources().resetMeshTransform(meshIndex)) {
                    m_cameraCollisionSource = nullptr;
                    m_accumDirty = true;
                    m_accumDirtyReason = "OBJECT_RESET";
                }
            };
            m_overlay.setScenePanelProvider(std::move(panelProvider));

            int selectedSceneIndex = currentSceneIndex;
            bool previousSoftwareOverride = m_settings.enableSoftwareRayTracing;
            const RenderSettings::CameraControlMode previousCameraControlMode =
                m_settings.cameraControlMode;
            m_overlay.buildUI(m_performanceStats,
                              m_settings,
                              uiRequestedReset,
                              m_sceneLabelPointers,
                              selectedSceneIndex,
                              m_presentationSettings);
            if (m_settings.cameraControlMode != previousCameraControlMode) {
                if (m_settings.cameraControlMode == RenderSettings::CameraControlMode::FirstPerson &&
                    previousCameraControlMode == RenderSettings::CameraControlMode::Orbit) {
                    m_settings.firstPersonCameraPosition = CameraOrbitEye(m_settings);
                }
                resetFirstPersonMotion();
                noteWindowInteraction();
            }

            if (m_sceneLoadStage == SceneLoadStage::LoadScene ||
                m_sceneLoadStage == SceneLoadStage::Activate) {
                const char* phaseLabel = (m_sceneLoadStage == SceneLoadStage::Activate)
                    ? "Activating scene"
                    : (m_sceneLoadExclusiveActive ? "Preparing scene (exclusive)" : "Preparing scene");
                const float progress = (m_sceneLoadStage == SceneLoadStage::Activate) ? 0.95f : 0.55f;
                ImGui::SetNextWindowPos(ImVec2(10.0f, 10.0f), ImGuiCond_Always);
                ImGui::Begin("Loading",
                             nullptr,
                             ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize);
                ImGui::Text("%s", phaseLabel);
                ImGui::ProgressBar(progress, ImVec2(240.0f, 0.0f));
                ImGui::End();
            }

            if (selectedSceneIndex != currentSceneIndex) {
                if (selectedSceneIndex <= 0 ||
                    selectedSceneIndex >= static_cast<int>(m_sceneIdentifiers.size())) {
                    // Procedural scene should always start from the canonical camera pose.
                    cancelPendingSceneLoad();
                    m_loadedSceneResources.reset();
                    m_externalSceneResources = nullptr;
                    const RenderSettings cameraDefaults{};
                    m_settings.cameraControlMode = cameraDefaults.cameraControlMode;
                    m_settings.cameraTarget = cameraDefaults.cameraTarget;
                    m_settings.cameraDistance = cameraDefaults.cameraDistance;
                    m_settings.cameraYaw = cameraDefaults.cameraYaw;
                    m_settings.cameraPitch = cameraDefaults.cameraPitch;
                    m_settings.firstPersonCameraPosition = cameraDefaults.firstPersonCameraPosition;
                    m_settings.firstPersonMoveSpeed = cameraDefaults.firstPersonMoveSpeed;
                    m_settings.cameraVerticalFov = cameraDefaults.cameraVerticalFov;
                    m_settings.cameraDefocusAngle = cameraDefaults.cameraDefocusAngle;
                    m_settings.cameraFocusDistance = cameraDefaults.cameraFocusDistance;
                    resetFirstPersonMotion();
                    buildProceduralScene();
                    resetAccumulation();
                } else {
                    const size_t sceneIdx = static_cast<size_t>(selectedSceneIndex);
                    const std::string& chosenId = m_sceneIdentifiers[sceneIdx];
                    if (!loadScene(chosenId)) {
                        NSLog(@"Failed to switch to scene '%s'", chosenId.c_str());
                    }
                }
            } else if (uiRequestedReset) {
                resetAccumulation();
            }

            if (m_settings.enableSoftwareRayTracing != previousSoftwareOverride) {
                activeSceneResources().setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);
                resetAccumulation();
            }

            if (m_overlay.consumeSaveExrRequest(pendingSavePath)) {
                // Defer save until after this frame is committed so capture includes
                // the latest presentation/denoise results.
            }
        }
#endif

        applyPresentationSettings(false);

        if (sceneLoadUiOnlyInProgress()) {
            id<MTLCommandBuffer> loadingUiCommandBuffer = [commandQueue commandBuffer];
            if (!loadingUiCommandBuffer) {
                return;
            }
            loadingUiCommandBuffer.label = @"Scene Loading UI";

            if (renderPassDescriptor && drawable) {
                id<MTLRenderCommandEncoder> renderEncoder =
                    [loadingUiCommandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
                if (renderEncoder) {
#if HAS_IMGUI
                    if (m_overlayInitialized) {
                        m_overlay.render(loadingUiCommandBuffer, renderEncoder);
                    }
#endif
                    [renderEncoder endEncoding];
                }
                [loadingUiCommandBuffer presentDrawable:drawable];
            }

            [loadingUiCommandBuffer commit];
            if (!pendingSavePath.empty()) {
                handleSaveExrRequest(pendingSavePath);
            }
            serviceSceneLoadState(true);
            return;
        }

        handleCameraInput();

        const CFAbsoluteTime motionNow = CFAbsoluteTimeGetCurrent();
        bool recentMotion = false;
        if (m_lastCameraInteractionTime > 0.0) {
            constexpr double kMotionHoldSeconds = 0.25;
            recentMotion = (motionNow - m_lastCameraInteractionTime) < kMotionHoldSeconds;
        }
        m_cameraMotionActive = recentMotion;
    }
    else {
        m_lastDrawableWaitMs = 0.0;
        m_cameraMotionActive = false;
    }

    bool environmentChanged = false;
    if (!syncEnvironmentMap(&environmentChanged)) {
        if (!m_settings.environmentMapPath.empty()) {
            NSLog(@"Failed to reload environment map: %s", m_settings.environmentMapPath.c_str());
        }
    } else if (environmentChanged) {
        resetAccumulation();
    }

    evaluateAccumulationState();
    serviceAccumulationReset();

    renderSize = targetRenderSize();
    m_accumulation.ensureTextures(renderSize);

    m_performanceStats.renderWidth = static_cast<uint32_t>(std::max<CGFloat>(renderSize.width, 0.0f));
    m_performanceStats.renderHeight = static_cast<uint32_t>(std::max<CGFloat>(renderSize.height, 0.0f));
    m_performanceStats.frameTimeMs = m_lastFrameTimeMs;
    if (auto envTex = activeSceneResources().environmentTexture()) {
        m_performanceStats.envRadianceMipCount =
            static_cast<uint32_t>(envTex.mipmapLevelCount);
    } else {
        m_performanceStats.envRadianceMipCount = 0;
    }

    const bool pathTraceFrameInFlight =
        !m_options.headless && m_pathTraceFrameInFlight.load(std::memory_order_acquire);
    if (!m_options.headless && pathTraceFrameInFlight) {
        CFAbsoluteTime encodeCpuStart = CFAbsoluteTimeGetCurrent();
        id<MTLCommandBuffer> uiCommandBuffer = [commandQueue commandBuffer];
        if (!uiCommandBuffer) {
            return;
        }
        uiCommandBuffer.label = @"Interactive UI Frame (Path Trace In Flight)";

        m_renderLoop.encodeDisplayOnlyFrame(uiCommandBuffer,
                                            renderPassDescriptor,
                                            drawable,
                                            m_accumulation,
                                            m_pipelines,
                                            m_overlay,
                                            m_settings);
        if (drawable) {
            [uiCommandBuffer presentDrawable:drawable];
        }
        m_lastCpuEncodeMs = (CFAbsoluteTimeGetCurrent() - encodeCpuStart) * 1000.0;
        std::weak_ptr<Impl> weakSelf = shared_from_this();
        m_interactiveUiFrameInFlight.store(true, std::memory_order_release);
        [uiCommandBuffer addCompletedHandler:^(__unused id<MTLCommandBuffer> cb) {
            if (auto strongSelf = weakSelf.lock()) {
                strongSelf->m_interactiveUiFrameInFlight.store(false, std::memory_order_release);
            }
        }];
        [uiCommandBuffer commit];

        if (!pendingSavePath.empty()) {
            handleSaveExrRequest(pendingSavePath);
        }
        return;
    }

    const float deltaSeconds =
        (m_lastFrameTimeMs > 0.0) ? static_cast<float>(m_lastFrameTimeMs * 0.001) : (1.0f / 60.0f);
    updateCameraSmoothing(deltaSeconds);

    RenderSettings frameSettings = m_settings;
    frameSettings.cameraYaw = m_smoothYaw;
    frameSettings.cameraPitch = m_smoothPitch;
    if (m_cameraMotionActive) {
        frameSettings.samplesPerFrame = 1u;
    }
    frameSettings.samplesPerFrame = std::max<uint32_t>(1u, frameSettings.samplesPerFrame);
    m_performanceStats.renderExecutionMode = static_cast<uint32_t>(frameSettings.executionMode);
    m_performanceStats.wavefrontSchedulingPolicy =
        static_cast<uint32_t>(frameSettings.wavefrontSchedulingPolicy);
    m_performanceStats.wavefrontCompactionEnabled =
        (frameSettings.wavefrontSchedulingPolicy !=
             PathTracer::RenderSettings::WavefrontSchedulingPolicy::Preview &&
         frameSettings.wavefrontCompactionEnabled)
            ? 1u
            : 0u;

    CFAbsoluteTime encodeCpuStart = CFAbsoluteTimeGetCurrent();
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    commandBuffer.label = @"Pathtracer Frame";

    FrameResult frameResult = m_renderLoop.encodeFrame(
        commandBuffer,
        renderPassDescriptor,
        drawable,
        m_context,
        m_accumulation,
        m_pipelines,
        activeSceneResources(),
        m_overlay,
        frameSettings);

    m_lastCpuEncodeMs = (CFAbsoluteTimeGetCurrent() - encodeCpuStart) * 1000.0;
    if (m_verboseTiming && !m_firstDispatchTimingLogged) {
        NSLog(@"[Timing] FirstDispatchEncodeMs=%.2f", m_lastCpuEncodeMs);
    }
    if (frameResult.samplesDispatched > 0) {
        m_activeSamplesPerFrame = frameResult.samplesDispatched;
    } else {
        m_activeSamplesPerFrame = frameSettings.samplesPerFrame;
    }

    MTLBufferHandle statsBuffer = m_renderLoop.statsBuffer();
    MTLBufferHandle queueCountersBuffer = m_renderLoop.queueCountersBuffer();
    MTLBufferHandle debugBuffer = nullptr;
#if PT_DEBUG_TOOLS
    debugBuffer = m_renderLoop.debugBuffer();
#endif
    const uint32_t finalSampleCount = m_accumulation.sampleCount();
    id<MTLDevice> allocatedSizeDevice = m_context.device();
    const bool logAllocatedSizeThisDispatch = m_options.headless && !m_firstDispatchTimingLogged;

    std::weak_ptr<Impl> weakSelf = shared_from_this();
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        if (logAllocatedSizeThisDispatch) {
            LogCurrentAllocatedSize("at-completion:", allocatedSizeDevice);
        }
        if (auto strongSelf = weakSelf.lock()) {
            if (!strongSelf->m_options.headless) {
                strongSelf->m_pathTraceFrameInFlight.store(false, std::memory_order_release);
            }
            strongSelf->updatePerformanceStats(cb,
                                               statsBuffer,
                                               queueCountersBuffer,
                                               debugBuffer,
                                               renderSize,
                                               finalSampleCount,
                                               frameResult);
        }
    }];

    if (!m_options.headless && drawable) {
        [commandBuffer presentDrawable:drawable];
    }

    if (logAllocatedSizeThisDispatch) {
        LogCurrentAllocatedSize("pre-dispatch:", allocatedSizeDevice);
    }

    if (m_options.headless && logAllocatedSizeThisDispatch) {
        const double captureDelay = CaptureDelaySeconds();
        if (captureDelay > 0.0) {
            NSLog(@"[Capture] delaying first headless command buffer commit for %.2f seconds", captureDelay);
            [NSThread sleepForTimeInterval:captureDelay];
        }
    }

    CFAbsoluteTime commitStart = CFAbsoluteTimeGetCurrent();
    if (!m_options.headless) {
        m_pathTraceFrameInFlight.store(true, std::memory_order_release);
    }
    [commandBuffer commit];
    CFAbsoluteTime commitEnd = CFAbsoluteTimeGetCurrent();
    if (m_verboseTiming && !m_firstDispatchTimingLogged) {
        double commitMs = (commitEnd - commitStart) * 1000.0;
        NSLog(@"[Timing] CommandBufferCommitMs=%.2f", commitMs);
    }

    if (m_options.headless) {
        CFAbsoluteTime waitStart = CFAbsoluteTimeGetCurrent();
        [commandBuffer waitUntilCompleted];
        CFAbsoluteTime waitEnd = CFAbsoluteTimeGetCurrent();
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            NSError* commandError = commandBuffer.error;
            if (commandError) {
                NSLog(@"[Headless] Command buffer failed status=%ld error=%@",
                      static_cast<long>(commandBuffer.status),
                      commandError);
            } else {
                NSLog(@"[Headless] Command buffer failed status=%ld with no NSError",
                      static_cast<long>(commandBuffer.status));
            }
            LogCommandBufferDiagnostics(commandBuffer);
        }
        if (m_verboseTiming && !m_firstDispatchTimingLogged) {
            double waitMs = (waitEnd - waitStart) * 1000.0;
            NSLog(@"[Timing] FirstDispatchToCompletionMs=%.2f", waitMs);
            m_firstDispatchTimingLogged = true;
        }
    }

    if (!pendingSavePath.empty()) {
        handleSaveExrRequest(pendingSavePath);
    }

    // Submit scene-load GPU work after the frame command buffer has been queued so
    // interactive rendering keeps priority and UI remains responsive.
    serviceSceneLoadState(true);
}

CGSize MetalRenderer::Impl::targetRenderSize() {
    const float requestedScale = ClampRenderScale(m_settings.renderScale);
    float appliedScale = requestedScale;

    auto clampDimension = [](uint32_t overrideValue, CGFloat fallback) -> CGFloat {
        if (overrideValue >= 8u) {
            return static_cast<CGFloat>(overrideValue);
        }
        return std::max<CGFloat>(fallback, 1.0f);
    };

    CGFloat fallbackWidth = 1.0f;
    CGFloat fallbackHeight = 1.0f;

    if (m_options.headless) {
        fallbackWidth = std::max<CGFloat>(m_options.width, 8);
        fallbackHeight = std::max<CGFloat>(m_options.height, 8);
    }
#if !DEBUG_FIXED_RENDER_RESOLUTION
    else {
        if (m_drawablePixelSize.width >= 1.0f && m_drawablePixelSize.height >= 1.0f) {
            fallbackWidth = m_drawablePixelSize.width;
            fallbackHeight = m_drawablePixelSize.height;
        } else if (MTKView* view = m_context.view()) {
            CGSize drawable = view.drawableSize;
            fallbackWidth = std::max<CGFloat>(drawable.width, 1.0f);
            fallbackHeight = std::max<CGFloat>(drawable.height, 1.0f);
        }
    }
#endif

    CGFloat baseWidth = clampDimension(m_settings.renderWidth, fallbackWidth);
    CGFloat baseHeight = clampDimension(m_settings.renderHeight, fallbackHeight);

#if DEBUG_FIXED_RENDER_RESOLUTION
    baseWidth = clampDimension(static_cast<uint32_t>(kDebugRenderWidth), baseWidth);
    baseHeight = clampDimension(static_cast<uint32_t>(kDebugRenderHeight), baseHeight);
#endif

    baseWidth = std::max<CGFloat>(baseWidth, 1.0f);
    baseHeight = std::max<CGFloat>(baseHeight, 1.0f);

    bool pixelClamped = false;
    if (!m_options.headless) {
        const double requestedPixels =
            static_cast<double>(baseWidth) * static_cast<double>(baseHeight) *
            static_cast<double>(appliedScale) * static_cast<double>(appliedScale);
        if (requestedPixels > kMaxRenderPixels && kMaxRenderPixels > 0.0) {
            const double reduction = std::sqrt(kMaxRenderPixels / requestedPixels);
            appliedScale *= static_cast<float>(reduction);
            pixelClamped = true;
        }
    }

    CGFloat scaledWidth = std::max<CGFloat>(static_cast<CGFloat>(std::round(baseWidth * appliedScale)), 1.0f);
    CGFloat scaledHeight = std::max<CGFloat>(static_cast<CGFloat>(std::round(baseHeight * appliedScale)), 1.0f);

    bool clamped = false;
    if (scaledWidth > kMaxRenderDimension) {
        scaledWidth = kMaxRenderDimension;
        clamped = true;
    }
    if (scaledHeight > kMaxRenderDimension) {
        scaledHeight = kMaxRenderDimension;
        clamped = true;
    }

    if (clamped && !m_renderClampWarningLogged) {
        m_renderClampWarningLogged = true;
        NSLog(@"[Renderer] Internal render size clamped to %.0f x %.0f (max %.0f)",
              scaledWidth,
              scaledHeight,
              static_cast<double>(kMaxRenderDimension));
    }

    if (pixelClamped && !m_renderPixelClampLogged) {
        m_renderPixelClampLogged = true;
        const double megaPixels = (static_cast<double>(scaledWidth) * static_cast<double>(scaledHeight)) / 1.0e6;
        NSLog(@"[Renderer] Internal render resolution reduced to %.0f x %.0f (~%.1f MP) for responsiveness",
              scaledWidth,
              scaledHeight,
              megaPixels);
    }

    const float widthScale =
        baseWidth > 0.0f ? static_cast<float>(scaledWidth / baseWidth) : appliedScale;
    const float heightScale =
        baseHeight > 0.0f ? static_cast<float>(scaledHeight / baseHeight) : appliedScale;
    appliedScale = std::max(widthScale, heightScale);

    m_currentRenderScale = appliedScale;
    m_settings.renderScale = requestedScale;
    return CGSizeMake(scaledWidth, scaledHeight);
}

bool MetalRenderer::Impl::initializeOverlay() {
#if HAS_IMGUI
    if (m_options.headless) {
        return false;
    }
    MTKView* view = m_context.view();
    MTLDeviceHandle device = m_context.device();
    if (!view || !device) {
        return false;
    }
    if (!m_overlay.initialize(view, device, m_contentScale)) {
        return false;
    }
    m_overlay.updateDisplayMetrics(m_viewSizePoints, m_contentScale);
    return true;
#else
    return false;
#endif
}

void MetalRenderer::Impl::updatePerformanceStats(MTLCommandBufferHandle commandBuffer,
                                                 MTLBufferHandle statsBuffer,
                                                 MTLBufferHandle queueCountersBuffer,
                                                 MTLBufferHandle debugBuffer,
                                                 CGSize renderSize,
                                                 uint32_t finalSampleCount,
                                                 const FrameResult& frameResult) {
    if (!commandBuffer) {
        return;
    }
    if (!m_options.headless && commandBuffer.status != MTLCommandBufferStatusCompleted) {
        const CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
        constexpr double kFailureLogIntervalSeconds = 1.0;
        if (m_lastCommandBufferFailureLogTime <= 0.0 ||
            (now - m_lastCommandBufferFailureLogTime) >= kFailureLogIntervalSeconds) {
            const char* sceneLabel = m_activeSceneId.empty() ? "<none>" : m_activeSceneId.c_str();
            NSError* commandError = commandBuffer.error;
            if (commandError) {
                NSLog(@"[Render] Command buffer failed scene='%s' status=%ld error=%@",
                      sceneLabel,
                      static_cast<long>(commandBuffer.status),
                      commandError);
            } else {
                NSLog(@"[Render] Command buffer failed scene='%s' status=%ld (no NSError)",
                      sceneLabel,
                      static_cast<long>(commandBuffer.status));
            }
            if (id<MTLDevice> device = m_context.device()) {
                const uint64_t allocatedWorkingSet = CurrentAllocatedSizeBytes(device);
                uint64_t recommendedWorkingSet = 0u;
                if ([device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
                    recommendedWorkingSet = static_cast<uint64_t>(device.recommendedMaxWorkingSetSize);
                }
                if (allocatedWorkingSet > 0u || recommendedWorkingSet > 0u) {
                    NSLog(@"[Render] memory currentAllocatedSize=%.2f MiB recommendedMaxWorkingSetSize=%.2f MiB",
                          BytesToMiB(allocatedWorkingSet),
                          BytesToMiB(recommendedWorkingSet));
                }
            }
            LogCommandBufferDiagnostics(commandBuffer);
            m_lastCommandBufferFailureLogTime = now;
        }
    }
    CFAbsoluteTime statsStart = CFAbsoluteTimeGetCurrent();

    if (commandBuffer.GPUStartTime != 0 && commandBuffer.GPUEndTime != 0 &&
        commandBuffer.GPUEndTime > commandBuffer.GPUStartTime) {
        m_performanceStats.gpuTimeMs = (commandBuffer.GPUEndTime - commandBuffer.GPUStartTime) * 1000.0;
    } else {
        m_performanceStats.gpuTimeMs = 0.0;
    }

    m_performanceStats.cpuEncodeMs = m_lastCpuEncodeMs;
    m_performanceStats.drawableWaitMs = m_lastDrawableWaitMs;
    m_performanceStats.activeSamplesPerFrame = m_activeSamplesPerFrame;
    m_performanceStats.currentAllocatedBytes = CurrentAllocatedSizeBytes(m_context.device());
    if (id<MTLDevice> device = m_context.device();
        device && [device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
        m_performanceStats.recommendedMaxWorkingSetBytes =
            static_cast<uint64_t>(device.recommendedMaxWorkingSetSize);
    } else {
        m_performanceStats.recommendedMaxWorkingSetBytes = 0u;
    }
    m_performanceStats.totalComputeDispatchCount = frameResult.totalComputeDispatchCount;
    m_performanceStats.restirDiInitializeDispatchCount =
        frameResult.restirDiInitializeDispatchCount;
    m_performanceStats.restirDiInitialCandidateDispatchCount =
        frameResult.restirDiInitialCandidateDispatchCount;
    m_performanceStats.restirDiTemporalReuseDispatchCount =
        frameResult.restirDiTemporalReuseDispatchCount;
    m_performanceStats.restirDiSpatialReuseDispatchCount =
        frameResult.restirDiSpatialReuseDispatchCount;
    m_performanceStats.restirDiVisibilityDispatchCount =
        frameResult.restirDiVisibilityDispatchCount;
    m_performanceStats.restirDiApplyLightingDispatchCount =
        frameResult.restirDiApplyLightingDispatchCount;
    m_performanceStats.restirDiDebugAovDispatchCount =
        frameResult.restirDiDebugAovDispatchCount;
    m_performanceStats.restirDiHardwareTraceActive =
        m_renderLoop.restirDiHardwareTraceActive();
    m_performanceStats.integrationDispatchCount = frameResult.integrationDispatchCount;
    m_performanceStats.presentationDispatchCount = frameResult.presentationDispatchCount;
    m_performanceStats.wavefrontResetDispatchCount = frameResult.wavefrontResetDispatchCount;
    m_performanceStats.wavefrontGenerateDispatchCount = frameResult.wavefrontGenerateDispatchCount;
    m_performanceStats.wavefrontBuildDispatchArgsDispatchCount =
        frameResult.wavefrontBuildDispatchArgsDispatchCount;
    m_performanceStats.wavefrontCompactQueueDispatchCount =
        frameResult.wavefrontCompactQueueDispatchCount;
    m_performanceStats.wavefrontPrepareDispatchCount = frameResult.wavefrontPrepareDispatchCount;
    m_performanceStats.wavefrontIntersectDispatchCount = frameResult.wavefrontIntersectDispatchCount;
    m_performanceStats.wavefrontShadeDispatchCount = frameResult.wavefrontShadeDispatchCount;
    m_performanceStats.wavefrontShadowDispatchCount = frameResult.wavefrontShadowDispatchCount;
    m_performanceStats.wavefrontSwapDispatchCount = frameResult.wavefrontSwapDispatchCount;
    m_performanceStats.wavefrontAccumulateDispatchCount = frameResult.wavefrontAccumulateDispatchCount;
    m_renderLoop.resolveGpuTiming(m_performanceStats.gpuTimeMs);
    m_performanceStats.gpuBucketTimingAvailable = m_renderLoop.gpuTimingAvailable();
    m_performanceStats.gpuBucketIntersectMs = m_renderLoop.gpuTimingIntersectMs();
    m_performanceStats.gpuBucketShadeMs = m_renderLoop.gpuTimingShadeMs();
    m_performanceStats.gpuBucketShadowQueueMs = m_renderLoop.gpuTimingShadowQueueMs();
    m_performanceStats.gpuBucketFusedIntegrateMs = m_renderLoop.gpuTimingFusedIntegrateMs();
    m_performanceStats.gpuBucketPresentationMs = m_renderLoop.gpuTimingPresentationMs();
    m_performanceStats.wavefrontQueueCompactionCostMs = m_renderLoop.gpuTimingCompactionMs();
    m_performanceStats.transportMemoryBytesTotal = m_renderLoop.transportMemoryBytesTotal();
    m_performanceStats.transportMemoryBytesPrivate = m_renderLoop.transportMemoryBytesPrivate();
    m_performanceStats.transportMemoryBytesShared = m_renderLoop.transportMemoryBytesShared();
    m_performanceStats.transportMemoryBytesStaging = m_renderLoop.transportMemoryBytesStaging();
    m_performanceStats.transportMemoryRecordCount = m_renderLoop.transportMemoryRecordCount();
    m_performanceStats.transportMemoryInvalidatedRecordCount =
        m_renderLoop.transportMemoryInvalidatedRecordCount();
    m_performanceStats.transportMemoryLastInvalidationReason =
        static_cast<uint32_t>(m_renderLoop.transportMemoryLastInvalidationReason());
    m_performanceStats.renderScale =
        static_cast<double>(std::clamp(m_currentRenderScale, kMinRenderScale, kMaxRenderScale));
    m_performanceStats.cameraMotionActive = m_cameraMotionActive;

    if (statsBuffer) {
        const auto* stats = reinterpret_cast<const PathtraceStats*>([statsBuffer contents]);
        if (stats) {
            const double totalRays = static_cast<double>(stats->primaryRayCount);
            const double shadowRays = static_cast<double>(stats->shadowRayCount);
            const double internalVisits = static_cast<double>(stats->internalNodeVisits);
            const double hardwareRays = static_cast<double>(stats->hardwareRayCount);
            const double hardwareHits = static_cast<double>(stats->hardwareHitCount);
            const double hardwareMisses = static_cast<double>(stats->hardwareMissCount);
            m_performanceStats.primaryRayCount = static_cast<uint64_t>(stats->primaryRayCount);
            m_performanceStats.shadowRayCount = static_cast<uint64_t>(stats->shadowRayCount);
            m_performanceStats.pathSegmentCount =
                static_cast<uint64_t>(stats->primaryRayCount) +
                static_cast<uint64_t>(stats->hardwareRayCount);
            m_performanceStats.tracedRayCount =
                m_performanceStats.pathSegmentCount + m_performanceStats.shadowRayCount;

            m_performanceStats.avgNodesVisited =
                totalRays > 0.0 ? static_cast<double>(stats->nodesVisited) / totalRays : 0.0;
            m_performanceStats.avgLeafPrimTests =
                totalRays > 0.0 ? static_cast<double>(stats->leafPrimTests) / totalRays : 0.0;
            m_performanceStats.shadowRayEarlyExitPct =
                shadowRays > 0.0
                    ? (static_cast<double>(stats->shadowRayEarlyExitCount) / shadowRays) * 100.0
                    : 0.0;
            m_performanceStats.bothChildrenVisitedPct =
                internalVisits > 0.0
                    ? (static_cast<double>(stats->internalBothVisited) / internalVisits) * 100.0
                    : 0.0;
            m_performanceStats.hardwareRayCount = static_cast<uint64_t>(stats->hardwareRayCount);
            m_performanceStats.hardwareHitCount = static_cast<uint64_t>(stats->hardwareHitCount);
            m_performanceStats.hardwareMissCount = static_cast<uint64_t>(stats->hardwareMissCount);
            m_performanceStats.hardwareResultNoneCount =
                static_cast<uint64_t>(stats->hardwareResultNoneCount);
            m_performanceStats.hardwareRejectedCount =
                static_cast<uint64_t>(stats->hardwareRejectedCount);
            m_performanceStats.hardwareUnavailableCount =
                static_cast<uint64_t>(stats->hardwareUnavailableCount);
            m_performanceStats.specularNeeOcclusionHitCount =
                static_cast<uint64_t>(stats->specularNeeOcclusionHitCount);
            m_performanceStats.specNeeEnvAddedCount =
                static_cast<uint64_t>(stats->specNeeEnvAddedCount);
            m_performanceStats.specNeeRectAddedCount =
                static_cast<uint64_t>(stats->specNeeRectAddedCount);
            m_performanceStats.mneeEnvHwOccludedCount =
                static_cast<uint64_t>(stats->mneeEnvHwOccludedCount);
            m_performanceStats.mneeEnvSwOccludedCount =
                static_cast<uint64_t>(stats->mneeEnvSwOccludedCount);
            m_performanceStats.mneeEnvHwSwMismatchCount =
                static_cast<uint64_t>(stats->mneeEnvHwSwMismatchCount);
            m_performanceStats.mneeRectHwOccludedCount =
                static_cast<uint64_t>(stats->mneeRectHwOccludedCount);
            m_performanceStats.mneeRectSwOccludedCount =
                static_cast<uint64_t>(stats->mneeRectSwOccludedCount);
            m_performanceStats.mneeRectHwSwMismatchCount =
                static_cast<uint64_t>(stats->mneeRectHwSwMismatchCount);
            m_performanceStats.mneeHitHwSwHitMissCount =
                static_cast<uint64_t>(stats->mneeHitHwSwHitMissCount);
            m_performanceStats.mneeHitHwSwNormalMismatchCount =
                static_cast<uint64_t>(stats->mneeHitHwSwNormalMismatchCount);
            m_performanceStats.mneeHitHwSwIdMismatchCount =
                static_cast<uint64_t>(stats->mneeHitHwSwIdMismatchCount);
            m_performanceStats.mneeHitHwSwTDiffCount =
                static_cast<uint64_t>(stats->mneeHitHwSwTDiffCount);
            m_performanceStats.mneeChainHwSwHitMissCount =
                static_cast<uint64_t>(stats->mneeChainHwSwHitMissCount);
            m_performanceStats.mneeChainHwSwNormalMismatchCount =
                static_cast<uint64_t>(stats->mneeChainHwSwNormalMismatchCount);
            m_performanceStats.mneeChainHwSwIdMismatchCount =
                static_cast<uint64_t>(stats->mneeChainHwSwIdMismatchCount);
            m_performanceStats.mneeChainHwSwTDiffCount =
                static_cast<uint64_t>(stats->mneeChainHwSwTDiffCount);
            m_performanceStats.mneeEligibleCount =
                static_cast<uint64_t>(stats->mneeEligibleCount);
            m_performanceStats.mneeEnvAttemptCount =
                static_cast<uint64_t>(stats->mneeEnvAttemptCount);
            m_performanceStats.mneeEnvAddedCount =
                static_cast<uint64_t>(stats->mneeEnvAddedCount);
            m_performanceStats.mneeRectAttemptCount =
                static_cast<uint64_t>(stats->mneeRectAttemptCount);
            m_performanceStats.mneeRectAddedCount =
                static_cast<uint64_t>(stats->mneeRectAddedCount);
            m_performanceStats.mneeContributionCount =
                static_cast<uint64_t>(stats->mneeContributionCount);
            const uint64_t mneeLumaSumFixed =
                (static_cast<uint64_t>(stats->mneeContributionLumaSumHi) << 32) |
                static_cast<uint64_t>(stats->mneeContributionLumaSumLo);
            m_performanceStats.mneeContributionLumaSumFixed = mneeLumaSumFixed;
            m_performanceStats.hardwareSelfHitRejectedCount =
                static_cast<uint64_t>(stats->hardwareSelfHitRejectedCount);
            for (int i = 0; i < 32; ++i) {
                m_performanceStats.hardwareMissDistanceBins[i] =
                    static_cast<uint64_t>(stats->hardwareMissDistanceBins[i]);
            }
            for (int i = 0; i < 4; ++i) {
                m_performanceStats.hardwareExcludeRetryHistogram[i] =
                    static_cast<uint64_t>(stats->hardwareExcludeRetryHistogram[i]);
            }
            float missDist = 0.0f;
            std::memcpy(&missDist, &stats->hardwareMissLastDistanceBits, sizeof(float));
            m_performanceStats.hardwareMissLastDistance = missDist;
            float selfHitDist = 0.0f;
            std::memcpy(&selfHitDist, &stats->hardwareSelfHitLastDistanceBits, sizeof(float));
            m_performanceStats.hardwareSelfHitLastDistance = selfHitDist;
            m_performanceStats.hardwareMissLastInstanceId = stats->hardwareMissLastInstanceId;
            m_performanceStats.hardwareMissLastPrimitiveId = stats->hardwareMissLastPrimitiveId;
            m_performanceStats.hardwareFallbackHitCount =
                static_cast<uint64_t>(stats->hardwareFallbackHitCount);
            m_performanceStats.hardwareFirstHitFallbackCount =
                static_cast<uint64_t>(stats->hardwareFirstHitFallbackCount);
            m_performanceStats.hardwareShadowTMinRetryCount =
                static_cast<uint64_t>(stats->hardwareShadowTMinRetryCount);
            m_performanceStats.hardwareShadowNearForeignHitCount =
                static_cast<uint64_t>(stats->hardwareShadowNearForeignHitCount);
            m_performanceStats.throughputNanInfCount =
                static_cast<uint64_t>(stats->throughputNanInfCount);
            m_performanceStats.radianceNanInfCount =
                static_cast<uint64_t>(stats->radianceNanInfCount);
            m_performanceStats.clampEventCount =
                static_cast<uint64_t>(stats->clampEventCount);
            m_performanceStats.restirGiCandidateCount =
                static_cast<uint64_t>(stats->restirGiCandidateCount);
            m_performanceStats.restirGiReservoirUpdateCount =
                static_cast<uint64_t>(stats->restirGiReservoirUpdateCount);
            m_performanceStats.restirGiAcceptCount =
                static_cast<uint64_t>(stats->restirGiAcceptCount);
            m_performanceStats.restirGiRejectCount =
                static_cast<uint64_t>(stats->restirGiRejectCount);
            m_performanceStats.restirGiFallbackCount =
                static_cast<uint64_t>(stats->restirGiFallbackCount);
            m_performanceStats.pathGuidingCandidateCount =
                static_cast<uint64_t>(stats->pathGuidingCandidateCount);
            m_performanceStats.pathGuidingUsedCount =
                static_cast<uint64_t>(stats->pathGuidingUsedCount);
            m_performanceStats.pathGuidingFallbackCount =
                static_cast<uint64_t>(stats->pathGuidingFallbackCount);
            m_performanceStats.pathGuidingInvalidCount =
                static_cast<uint64_t>(stats->pathGuidingInvalidCount);
            m_performanceStats.restirPtCandidateCount =
                static_cast<uint64_t>(stats->restirPtCandidateCount);
            m_performanceStats.restirPtReservoirUpdateCount =
                static_cast<uint64_t>(stats->restirPtReservoirUpdateCount);
            m_performanceStats.restirPtRejectedInvalidCount =
                static_cast<uint64_t>(stats->restirPtRejectedInvalidCount);
            m_performanceStats.restirPtRejectedUnsupportedCount =
                static_cast<uint64_t>(stats->restirPtRejectedUnsupportedCount);
            m_performanceStats.restirPtDebugRecordCount =
                static_cast<uint64_t>(stats->restirPtDebugRecordCount);
            m_performanceStats.restirPtReuseCandidateCount =
                static_cast<uint64_t>(stats->restirPtReuseCandidateCount);
            m_performanceStats.restirPtReuseAppliedCount =
                static_cast<uint64_t>(stats->restirPtReuseAppliedCount);
            m_performanceStats.restirPtReuseRejectedInvalidCount =
                static_cast<uint64_t>(stats->restirPtReuseRejectedInvalidCount);
            m_performanceStats.restirPtReuseFallbackCount =
                static_cast<uint64_t>(stats->restirPtReuseFallbackCount);
            m_performanceStats.radianceCacheQueryCount =
                static_cast<uint64_t>(stats->radianceCacheQueryCount);
            m_performanceStats.radianceCacheHitCount =
                static_cast<uint64_t>(stats->radianceCacheHitCount);
            m_performanceStats.radianceCacheMissCount =
                static_cast<uint64_t>(stats->radianceCacheMissCount);
            m_performanceStats.radianceCacheTrainCount =
                static_cast<uint64_t>(stats->radianceCacheTrainCount);
            m_performanceStats.radianceCacheFallbackCount =
                static_cast<uint64_t>(stats->radianceCacheFallbackCount);
            m_performanceStats.radianceCacheInvalidCount =
                static_cast<uint64_t>(stats->radianceCacheInvalidCount);
            m_performanceStats.causticEyeVertexCount =
                static_cast<uint64_t>(stats->causticEyeVertexCount);
            m_performanceStats.causticLightVertexCount =
                static_cast<uint64_t>(stats->causticLightVertexCount);
            m_performanceStats.causticConnectionCandidateCount =
                static_cast<uint64_t>(stats->causticConnectionCandidateCount);
            m_performanceStats.causticDeltaChainCount =
                static_cast<uint64_t>(stats->causticDeltaChainCount);
            m_performanceStats.causticDebugClassificationCount =
                static_cast<uint64_t>(stats->causticDebugClassificationCount);
            m_performanceStats.volumeTransmittanceQueryCount =
                static_cast<uint64_t>(stats->volumeTransmittanceQueryCount);
            m_performanceStats.volumeScatterEventCount =
                static_cast<uint64_t>(stats->volumeScatterEventCount);
            m_performanceStats.volumeNeeRayCount =
                static_cast<uint64_t>(stats->volumeNeeRayCount);
            m_performanceStats.volumePhaseSampleCount =
                static_cast<uint64_t>(stats->volumePhaseSampleCount);
            m_performanceStats.volumeMediumBoundaryEventCount =
                static_cast<uint64_t>(stats->volumeMediumBoundaryEventCount);
            m_performanceStats.volumeFallbackCount =
                static_cast<uint64_t>(stats->volumeFallbackCount);
            m_performanceStats.spectralWavelengthSampleCount =
                static_cast<uint64_t>(stats->spectralWavelengthSampleCount);
            m_performanceStats.spectralMaterialLookupCount =
                static_cast<uint64_t>(stats->spectralMaterialLookupCount);
            m_performanceStats.spectralDispersionEventCount =
                static_cast<uint64_t>(stats->spectralDispersionEventCount);
            m_performanceStats.spectralFallbackCount =
                static_cast<uint64_t>(stats->spectralFallbackCount);
            m_performanceStats.wavefrontActiveRayCount =
                static_cast<uint32_t>(stats->wavefrontActiveRayCount);
            m_performanceStats.wavefrontNextRayCount =
                static_cast<uint32_t>(stats->wavefrontNextRayCount);
            m_performanceStats.wavefrontShadowRayCount =
                static_cast<uint32_t>(stats->wavefrontShadowRayCount);
            m_performanceStats.wavefrontTerminatedCount =
                static_cast<uint32_t>(stats->wavefrontTerminatedCount);
            m_performanceStats.wavefrontOverflowFlag =
                static_cast<uint32_t>(stats->wavefrontOverflowFlag);
            m_performanceStats.wavefrontTerminateMissCount =
                static_cast<uint64_t>(stats->wavefrontTerminateMissCount);
            m_performanceStats.wavefrontTerminateEmissiveCount =
                static_cast<uint64_t>(stats->wavefrontTerminateEmissiveCount);
            m_performanceStats.wavefrontTerminateMaxDepthCount =
                static_cast<uint64_t>(stats->wavefrontTerminateMaxDepthCount);
            m_performanceStats.wavefrontTerminateOverflowCount =
                static_cast<uint64_t>(stats->wavefrontTerminateOverflowCount);
            m_performanceStats.wavefrontEnvShadowRayCount =
                static_cast<uint32_t>(stats->wavefrontEnvShadowRayCount);
            m_performanceStats.wavefrontAovAlbedoValidCount =
                static_cast<uint32_t>(stats->wavefrontAovAlbedoValidCount);
            m_performanceStats.wavefrontAovNormalValidCount =
                static_cast<uint32_t>(stats->wavefrontAovNormalValidCount);
            m_performanceStats.wavefrontPrimaryGeneratedCount =
                static_cast<uint64_t>(stats->wavefrontPrimaryGeneratedCount);
            m_performanceStats.wavefrontIntersectProcessedCount =
                static_cast<uint64_t>(stats->wavefrontIntersectProcessedCount);
            m_performanceStats.wavefrontShadeProcessedCount =
                static_cast<uint64_t>(stats->wavefrontShadeProcessedCount);
            m_performanceStats.wavefrontShadowProcessedCount =
                static_cast<uint64_t>(stats->wavefrontShadowProcessedCount);
            m_performanceStats.wavefrontNextRayEnqueueCount =
                static_cast<uint64_t>(stats->wavefrontNextRayEnqueueCount);
            m_performanceStats.wavefrontShadowRayEnqueueCount =
                static_cast<uint64_t>(stats->wavefrontShadowRayEnqueueCount);
            if (queueCountersBuffer) {
                const auto* queueCounters =
                    reinterpret_cast<const QueueCounters*>([queueCountersBuffer contents]);
                const WavefrontQueueHeader* headers[] = {
                    &queueCounters->primaryRayQueue,
                    &queueCounters->diffuseRayQueue,
                    &queueCounters->glossyRayQueue,
                    &queueCounters->specularRayQueue,
                    &queueCounters->transmissionRayQueue,
                    &queueCounters->shadowRayQueue,
                    &queueCounters->materialEvalQueue,
                    &queueCounters->reservoirUpdateQueue,
                    &queueCounters->mediumQueue,
                    &queueCounters->cacheQueryQueue,
                    &queueCounters->cacheTrainingQueue,
                    &queueCounters->guidingTrainingQueue,
                };
                for (uint32_t i = 0u; i < PerformanceStats::kWavefrontQueueMetricCount; ++i) {
                    const uint32_t active = static_cast<uint32_t>(headers[i]->activeCount);
                    const uint32_t peakActive = static_cast<uint32_t>(headers[i]->peakActiveCount);
                    const uint32_t capacity = headers[i]->capacity;
                    m_performanceStats.wavefrontQueueActiveCounts[i] = active;
                    m_performanceStats.wavefrontQueuePeakActiveCounts[i] = peakActive;
                    m_performanceStats.wavefrontQueueCapacities[i] = capacity;
                    m_performanceStats.wavefrontQueueOverflowCounts[i] =
                        static_cast<uint32_t>(headers[i]->overflowCount);
                    m_performanceStats.wavefrontQueueOccupancyRatios[i] =
                        (capacity > 0u) ? static_cast<double>(active) / static_cast<double>(capacity) : 0.0;
                    m_performanceStats.wavefrontQueuePeakOccupancyRatios[i] =
                        (capacity > 0u) ? static_cast<double>(peakActive) / static_cast<double>(capacity) : 0.0;
                }
            }
            if (stats->hardwareShadowNearForeignHitCount > 0u) {
                static uint32_t s_lastNearForeignLogCount = 0u;
                if (stats->hardwareShadowNearForeignHitCount != s_lastNearForeignLogCount) {
                    s_lastNearForeignLogCount = stats->hardwareShadowNearForeignHitCount;
                    NSLog(@"[ShadowRetry] tMinRetry=%u nearForeignHit=%u "
                          "(HWRT shadow rays hitting t<kEpsilon foreign mesh — "
                          "kParityReasonBelowTMin=0x20 in parity log)",
                          stats->hardwareShadowTMinRetryCount,
                          stats->hardwareShadowNearForeignHitCount);
                }
            }
            m_pbrMetricsSnapshot.available = true;
            m_pbrMetricsSnapshot.throughputNanInfCount =
                static_cast<uint64_t>(stats->throughputNanInfCount);
            m_pbrMetricsSnapshot.radianceNanInfCount =
                static_cast<uint64_t>(stats->radianceNanInfCount);
            m_pbrMetricsSnapshot.clampEventCount =
                static_cast<uint64_t>(stats->clampEventCount);
            m_pbrMetricsSnapshot.specNeeEnvAddedCount =
                static_cast<uint64_t>(stats->specNeeEnvAddedCount);
            m_pbrMetricsSnapshot.specNeeRectAddedCount =
                static_cast<uint64_t>(stats->specNeeRectAddedCount);
            m_pbrMetricsSnapshot.mneeEligibleCount =
                static_cast<uint64_t>(stats->mneeEligibleCount);
            m_pbrMetricsSnapshot.mneeEnvAttemptCount =
                static_cast<uint64_t>(stats->mneeEnvAttemptCount);
            m_pbrMetricsSnapshot.mneeEnvAddedCount =
                static_cast<uint64_t>(stats->mneeEnvAddedCount);
            m_pbrMetricsSnapshot.mneeEnvChainUnsupportedCount =
                static_cast<uint64_t>(stats->mneeEnvChainUnsupportedCount);
            m_pbrMetricsSnapshot.mneeEnvChainNonDeltaBlockCount =
                static_cast<uint64_t>(stats->mneeEnvChainNonDeltaBlockCount);
            m_pbrMetricsSnapshot.mneeEnvChainSampleRejectCount =
                static_cast<uint64_t>(stats->mneeEnvChainSampleRejectCount);
            m_pbrMetricsSnapshot.mneeEnvChainWeightRejectCount =
                static_cast<uint64_t>(stats->mneeEnvChainWeightRejectCount);
            m_pbrMetricsSnapshot.mneeEnvChainMaxBounceCount =
                static_cast<uint64_t>(stats->mneeEnvChainMaxBounceCount);
            m_pbrMetricsSnapshot.mneeRectAddedCount =
                static_cast<uint64_t>(stats->mneeRectAddedCount);
            m_pbrMetricsSnapshot.mneeContributionCount =
                static_cast<uint64_t>(stats->mneeContributionCount);
            m_pbrMetricsSnapshot.restirGiCandidateCount =
                static_cast<uint64_t>(stats->restirGiCandidateCount);
            m_pbrMetricsSnapshot.restirGiReservoirUpdateCount =
                static_cast<uint64_t>(stats->restirGiReservoirUpdateCount);
            m_pbrMetricsSnapshot.restirGiAcceptCount =
                static_cast<uint64_t>(stats->restirGiAcceptCount);
            m_pbrMetricsSnapshot.restirGiRejectCount =
                static_cast<uint64_t>(stats->restirGiRejectCount);
            m_pbrMetricsSnapshot.restirGiFallbackCount =
                static_cast<uint64_t>(stats->restirGiFallbackCount);
            m_pbrMetricsSnapshot.pathGuidingCandidateCount =
                static_cast<uint64_t>(stats->pathGuidingCandidateCount);
            m_pbrMetricsSnapshot.pathGuidingUsedCount =
                static_cast<uint64_t>(stats->pathGuidingUsedCount);
            m_pbrMetricsSnapshot.pathGuidingFallbackCount =
                static_cast<uint64_t>(stats->pathGuidingFallbackCount);
            m_pbrMetricsSnapshot.pathGuidingInvalidCount =
                static_cast<uint64_t>(stats->pathGuidingInvalidCount);
            m_pbrMetricsSnapshot.restirPtCandidateCount =
                static_cast<uint64_t>(stats->restirPtCandidateCount);
            m_pbrMetricsSnapshot.restirPtReservoirUpdateCount =
                static_cast<uint64_t>(stats->restirPtReservoirUpdateCount);
            m_pbrMetricsSnapshot.restirPtRejectedInvalidCount =
                static_cast<uint64_t>(stats->restirPtRejectedInvalidCount);
            m_pbrMetricsSnapshot.restirPtRejectedUnsupportedCount =
                static_cast<uint64_t>(stats->restirPtRejectedUnsupportedCount);
            m_pbrMetricsSnapshot.restirPtDebugRecordCount =
                static_cast<uint64_t>(stats->restirPtDebugRecordCount);
            m_pbrMetricsSnapshot.restirPtReuseCandidateCount =
                static_cast<uint64_t>(stats->restirPtReuseCandidateCount);
            m_pbrMetricsSnapshot.restirPtReuseAppliedCount =
                static_cast<uint64_t>(stats->restirPtReuseAppliedCount);
            m_pbrMetricsSnapshot.restirPtReuseRejectedInvalidCount =
                static_cast<uint64_t>(stats->restirPtReuseRejectedInvalidCount);
            m_pbrMetricsSnapshot.restirPtReuseFallbackCount =
                static_cast<uint64_t>(stats->restirPtReuseFallbackCount);
            m_pbrMetricsSnapshot.radianceCacheQueryCount =
                static_cast<uint64_t>(stats->radianceCacheQueryCount);
            m_pbrMetricsSnapshot.radianceCacheHitCount =
                static_cast<uint64_t>(stats->radianceCacheHitCount);
            m_pbrMetricsSnapshot.radianceCacheMissCount =
                static_cast<uint64_t>(stats->radianceCacheMissCount);
            m_pbrMetricsSnapshot.radianceCacheTrainCount =
                static_cast<uint64_t>(stats->radianceCacheTrainCount);
            m_pbrMetricsSnapshot.radianceCacheFallbackCount =
                static_cast<uint64_t>(stats->radianceCacheFallbackCount);
            m_pbrMetricsSnapshot.radianceCacheInvalidCount =
                static_cast<uint64_t>(stats->radianceCacheInvalidCount);
            m_pbrMetricsSnapshot.causticEyeVertexCount =
                static_cast<uint64_t>(stats->causticEyeVertexCount);
            m_pbrMetricsSnapshot.causticLightVertexCount =
                static_cast<uint64_t>(stats->causticLightVertexCount);
            m_pbrMetricsSnapshot.causticConnectionCandidateCount =
                static_cast<uint64_t>(stats->causticConnectionCandidateCount);
            m_pbrMetricsSnapshot.causticDeltaChainCount =
                static_cast<uint64_t>(stats->causticDeltaChainCount);
            m_pbrMetricsSnapshot.causticDebugClassificationCount =
                static_cast<uint64_t>(stats->causticDebugClassificationCount);
            m_pbrMetricsSnapshot.volumeTransmittanceQueryCount =
                static_cast<uint64_t>(stats->volumeTransmittanceQueryCount);
            m_pbrMetricsSnapshot.volumeScatterEventCount =
                static_cast<uint64_t>(stats->volumeScatterEventCount);
            m_pbrMetricsSnapshot.volumeNeeRayCount =
                static_cast<uint64_t>(stats->volumeNeeRayCount);
            m_pbrMetricsSnapshot.volumePhaseSampleCount =
                static_cast<uint64_t>(stats->volumePhaseSampleCount);
            m_pbrMetricsSnapshot.volumeMediumBoundaryEventCount =
                static_cast<uint64_t>(stats->volumeMediumBoundaryEventCount);
            m_pbrMetricsSnapshot.volumeFallbackCount =
                static_cast<uint64_t>(stats->volumeFallbackCount);
            m_pbrMetricsSnapshot.spectralWavelengthSampleCount =
                static_cast<uint64_t>(stats->spectralWavelengthSampleCount);
            m_pbrMetricsSnapshot.spectralMaterialLookupCount =
                static_cast<uint64_t>(stats->spectralMaterialLookupCount);
            m_pbrMetricsSnapshot.spectralDispersionEventCount =
                static_cast<uint64_t>(stats->spectralDispersionEventCount);
            m_pbrMetricsSnapshot.spectralFallbackCount =
                static_cast<uint64_t>(stats->spectralFallbackCount);
            m_pbrMetricsSnapshot.maxThroughputByBounce.clear();
            m_performanceStats.hardwareLastResultType = stats->hardwareLastResultType;
            m_performanceStats.hardwareLastInstanceId = stats->hardwareLastInstanceId;
            m_performanceStats.hardwareLastPrimitiveId = stats->hardwareLastPrimitiveId;
            float lastDistance = 0.0f;
            static_assert(sizeof(uint32_t) == sizeof(float), "uint32_t must match float");
            std::memcpy(&lastDistance, &stats->hardwareLastDistanceBits, sizeof(float));
            m_performanceStats.hardwareLastDistance = lastDistance;
            if (hardwareRays > 0.0) {
                m_performanceStats.hardwareRayPctHit = (hardwareHits / hardwareRays) * 100.0;
                m_performanceStats.hardwareRayPctMiss = (hardwareMisses / hardwareRays) * 100.0;
            } else {
                m_performanceStats.hardwareRayPctHit = 0.0;
                m_performanceStats.hardwareRayPctMiss = 0.0;
            }
            m_performanceStats.wavefrontQueueBytesActive =
                m_renderLoop.wavefrontQueueBytesActive();
            m_performanceStats.wavefrontQueueBytesNext =
                m_renderLoop.wavefrontQueueBytesNext();
            m_performanceStats.wavefrontQueueBytesHit =
                m_renderLoop.wavefrontQueueBytesHit();
            m_performanceStats.wavefrontQueueBytesShadow =
                m_renderLoop.wavefrontQueueBytesShadow();
            m_performanceStats.wavefrontQueueBytesPathState =
                m_renderLoop.wavefrontQueueBytesPathState();
            m_performanceStats.wavefrontQueueBytesTotal =
                m_renderLoop.wavefrontQueueBytesTotal();
            m_performanceStats.wavefrontBudgetBytes =
                m_renderLoop.wavefrontBudgetBytes();
            m_performanceStats.wavefrontBudgetUsagePercent =
                (m_performanceStats.wavefrontBudgetBytes > 0u)
                    ? (static_cast<double>(m_performanceStats.wavefrontQueueBytesTotal) /
                       static_cast<double>(m_performanceStats.wavefrontBudgetBytes)) * 100.0
                    : 0.0;
            if (stats->hardwareRayCount > 0 && stats->hardwareHitCount == 0) {
                NSLog(@"Warning: Hardware RT rays=%u resultNone=%u rejected=%u unavailable=%u type=%u dist=%f inst=%u prim=%u",
                      stats->hardwareRayCount,
                      stats->hardwareResultNoneCount,
                      stats->hardwareRejectedCount,
                      stats->hardwareUnavailableCount,
                      stats->hardwareLastResultType,
                      lastDistance,
                      stats->hardwareLastInstanceId,
                      stats->hardwareLastPrimitiveId);
            } else if (stats->hardwareHitCount > 0) {
                static bool loggedHardwareSuccess = false;
                if (!loggedHardwareSuccess) {
                    loggedHardwareSuccess = true;
                    NSLog(@"Hardware RT active: rays=%u hits=%u misses=%u none=%u",
                          stats->hardwareRayCount,
                          stats->hardwareHitCount,
                          stats->hardwareMissCount,
                          stats->hardwareResultNoneCount);
#if PT_MNEE_OCCLUSION_PARITY
                    static bool loggedMneeParity = false;
                    if (!loggedMneeParity) {
                        loggedMneeParity = true;
                        NSLog(@"MNEE HW/SW visibility: env(hw=%u sw=%u mismatch=%u) rect(hw=%u sw=%u mismatch=%u)",
                              stats->mneeEnvHwOccludedCount,
                              stats->mneeEnvSwOccludedCount,
                              stats->mneeEnvHwSwMismatchCount,
                              stats->mneeRectHwOccludedCount,
                              stats->mneeRectSwOccludedCount,
                              stats->mneeRectHwSwMismatchCount);
                        NSLog(@"MNEE HW/SW hit parity: hitMiss=%u normal=%u id=%u tdiff=%u",
                              stats->mneeHitHwSwHitMissCount,
                              stats->mneeHitHwSwNormalMismatchCount,
                              stats->mneeHitHwSwIdMismatchCount,
                              stats->mneeHitHwSwTDiffCount);
                    }
#endif
                }
            }
            if (m_verboseTiming && !m_mneeAccountingLogged) {
                m_mneeAccountingLogged = true;
                constexpr double kMneeLumaScale = 1024.0;
                const uint64_t lumaFixed =
                    (static_cast<uint64_t>(stats->mneeContributionLumaSumHi) << 32) |
                    static_cast<uint64_t>(stats->mneeContributionLumaSumLo);
                const double lumaSum = static_cast<double>(lumaFixed) / kMneeLumaScale;
                const double lumaAvg = (stats->mneeContributionCount > 0)
                                           ? (lumaSum / static_cast<double>(stats->mneeContributionCount))
                                           : 0.0;
                NSLog(@"MNEE accounting: eligible=%u env(attempt=%u added=%u chain[unsupported=%u nonDelta=%u sampleReject=%u weightReject=%u maxBounce=%u]) rect(attempt=%u added=%u) contrib(count=%u lumaSum=%.6f avg=%.6f)",
                      stats->mneeEligibleCount,
                      stats->mneeEnvAttemptCount,
                      stats->mneeEnvAddedCount,
                      stats->mneeEnvChainUnsupportedCount,
                      stats->mneeEnvChainNonDeltaBlockCount,
                      stats->mneeEnvChainSampleRejectCount,
                      stats->mneeEnvChainWeightRejectCount,
                      stats->mneeEnvChainMaxBounceCount,
                      stats->mneeRectAttemptCount,
                      stats->mneeRectAddedCount,
                      stats->mneeContributionCount,
                      lumaSum,
                      lumaAvg);
                NSLog(@"Spec-NEE accounting: env(added=%u) rect(added=%u) occlusionHits=%u",
                      stats->specNeeEnvAddedCount,
                      stats->specNeeRectAddedCount,
                      stats->specularNeeOcclusionHitCount);
            }
        } else {
            m_performanceStats.avgNodesVisited = 0.0;
            m_performanceStats.avgLeafPrimTests = 0.0;
            m_performanceStats.shadowRayEarlyExitPct = 0.0;
            m_performanceStats.bothChildrenVisitedPct = 0.0;
            m_performanceStats.hardwareRayCount = 0;
            m_performanceStats.hardwareHitCount = 0;
            m_performanceStats.hardwareMissCount = 0;
            m_performanceStats.hardwareResultNoneCount = 0;
            m_performanceStats.hardwareRejectedCount = 0;
            m_performanceStats.hardwareUnavailableCount = 0;
            m_performanceStats.specularNeeOcclusionHitCount = 0;
            m_performanceStats.specNeeEnvAddedCount = 0;
            m_performanceStats.specNeeRectAddedCount = 0;
            m_performanceStats.mneeEnvHwOccludedCount = 0;
            m_performanceStats.mneeEnvSwOccludedCount = 0;
            m_performanceStats.mneeEnvHwSwMismatchCount = 0;
            m_performanceStats.mneeRectHwOccludedCount = 0;
            m_performanceStats.mneeRectSwOccludedCount = 0;
            m_performanceStats.mneeRectHwSwMismatchCount = 0;
            m_performanceStats.mneeHitHwSwHitMissCount = 0;
            m_performanceStats.mneeHitHwSwNormalMismatchCount = 0;
            m_performanceStats.mneeHitHwSwIdMismatchCount = 0;
            m_performanceStats.mneeHitHwSwTDiffCount = 0;
            m_performanceStats.mneeChainHwSwHitMissCount = 0;
            m_performanceStats.mneeChainHwSwNormalMismatchCount = 0;
            m_performanceStats.mneeChainHwSwIdMismatchCount = 0;
            m_performanceStats.mneeChainHwSwTDiffCount = 0;
            m_performanceStats.mneeEligibleCount = 0;
            m_performanceStats.mneeEnvAttemptCount = 0;
            m_performanceStats.mneeEnvAddedCount = 0;
            m_performanceStats.mneeRectAttemptCount = 0;
            m_performanceStats.mneeRectAddedCount = 0;
            m_performanceStats.mneeContributionCount = 0;
            m_performanceStats.mneeContributionLumaSumFixed = 0;
            m_performanceStats.hardwareSelfHitRejectedCount = 0;
            std::fill(std::begin(m_performanceStats.hardwareMissDistanceBins),
                      std::end(m_performanceStats.hardwareMissDistanceBins),
                      0ull);
            std::fill(std::begin(m_performanceStats.hardwareExcludeRetryHistogram),
                      std::end(m_performanceStats.hardwareExcludeRetryHistogram),
                      0ull);
            m_performanceStats.hardwareMissLastDistance = 0.0f;
            m_performanceStats.hardwareSelfHitLastDistance = 0.0f;
            m_performanceStats.hardwareMissLastInstanceId = 0;
            m_performanceStats.hardwareMissLastPrimitiveId = 0;
            m_performanceStats.hardwareFallbackHitCount = 0;
            m_performanceStats.hardwareFirstHitFallbackCount = 0;
            m_performanceStats.hardwareShadowTMinRetryCount = 0;
            m_performanceStats.hardwareShadowNearForeignHitCount = 0;
            m_pbrMetricsSnapshot = PbrMetricsSnapshot{};
            m_performanceStats.hardwareLastResultType = 0;
            m_performanceStats.hardwareLastInstanceId = 0;
            m_performanceStats.hardwareLastPrimitiveId = 0;
            m_performanceStats.hardwareLastDistance = 0.0f;
            m_performanceStats.hardwareRayPctHit = 0.0;
            m_performanceStats.hardwareRayPctMiss = 0.0;
        }
    } else {
        m_performanceStats.avgNodesVisited = 0.0;
        m_performanceStats.avgLeafPrimTests = 0.0;
        m_performanceStats.shadowRayEarlyExitPct = 0.0;
        m_performanceStats.bothChildrenVisitedPct = 0.0;
        m_performanceStats.hardwareRayCount = 0;
        m_performanceStats.hardwareHitCount = 0;
        m_performanceStats.hardwareMissCount = 0;
        m_performanceStats.hardwareResultNoneCount = 0;
        m_performanceStats.hardwareRejectedCount = 0;
        m_performanceStats.hardwareUnavailableCount = 0;
        m_performanceStats.specularNeeOcclusionHitCount = 0;
        m_performanceStats.specNeeEnvAddedCount = 0;
        m_performanceStats.specNeeRectAddedCount = 0;
        m_performanceStats.mneeEnvHwOccludedCount = 0;
        m_performanceStats.mneeEnvSwOccludedCount = 0;
        m_performanceStats.mneeEnvHwSwMismatchCount = 0;
        m_performanceStats.mneeRectHwOccludedCount = 0;
        m_performanceStats.mneeRectSwOccludedCount = 0;
        m_performanceStats.mneeRectHwSwMismatchCount = 0;
        m_performanceStats.mneeHitHwSwHitMissCount = 0;
        m_performanceStats.mneeHitHwSwNormalMismatchCount = 0;
        m_performanceStats.mneeHitHwSwIdMismatchCount = 0;
        m_performanceStats.mneeHitHwSwTDiffCount = 0;
        m_performanceStats.mneeChainHwSwHitMissCount = 0;
        m_performanceStats.mneeChainHwSwNormalMismatchCount = 0;
        m_performanceStats.mneeChainHwSwIdMismatchCount = 0;
        m_performanceStats.mneeChainHwSwTDiffCount = 0;
        m_performanceStats.mneeEligibleCount = 0;
        m_performanceStats.mneeEnvAttemptCount = 0;
        m_performanceStats.mneeEnvAddedCount = 0;
        m_performanceStats.mneeRectAttemptCount = 0;
        m_performanceStats.mneeRectAddedCount = 0;
        m_performanceStats.mneeContributionCount = 0;
        m_performanceStats.mneeContributionLumaSumFixed = 0;
        m_performanceStats.hardwareSelfHitRejectedCount = 0;
        std::fill(std::begin(m_performanceStats.hardwareMissDistanceBins),
                  std::end(m_performanceStats.hardwareMissDistanceBins),
                  0ull);
        std::fill(std::begin(m_performanceStats.hardwareExcludeRetryHistogram),
                  std::end(m_performanceStats.hardwareExcludeRetryHistogram),
                  0ull);
        m_performanceStats.hardwareMissLastDistance = 0.0f;
        m_performanceStats.hardwareSelfHitLastDistance = 0.0f;
        m_performanceStats.hardwareMissLastInstanceId = 0;
        m_performanceStats.hardwareMissLastPrimitiveId = 0;
        m_performanceStats.hardwareFallbackHitCount = 0;
        m_performanceStats.hardwareFirstHitFallbackCount = 0;
        m_performanceStats.hardwareShadowTMinRetryCount = 0;
        m_performanceStats.hardwareShadowNearForeignHitCount = 0;
        m_pbrMetricsSnapshot = PbrMetricsSnapshot{};
        m_performanceStats.hardwareLastResultType = 0;
        m_performanceStats.hardwareLastInstanceId = 0;
        m_performanceStats.hardwareLastPrimitiveId = 0;
        m_performanceStats.hardwareLastDistance = 0.0f;
        m_performanceStats.hardwareRayPctHit = 0.0;
        m_performanceStats.hardwareRayPctMiss = 0.0;
    }

    uint32_t debugEntryCount = 0;
    uint32_t debugMaxEntries = 0;
    uint32_t parityEntryCount = 0;
    uint32_t parityMaxEntries = 0;
    if (debugBuffer) {
#if PT_DEBUG_TOOLS
        const auto* debugData =
            reinterpret_cast<const PathtraceDebugBuffer*>([debugBuffer contents]);
        if (debugData) {
            debugMaxEntries = debugData->maxEntries;
            uint32_t recorded = debugData->writeIndex;
            debugEntryCount = std::min(recorded, debugMaxEntries);
            if (debugEntryCount > 0 && m_settings.enablePathDebug &&
                debugEntryCount != m_lastDebugLogCount) {
                const uint32_t entriesToLog = std::min(debugEntryCount, 32u);
                for (uint32_t i = 0; i < entriesToLog; ++i) {
                    const auto& entry = debugData->entries[i];
                    const char* integratorLabel = (entry.integrator == 0u) ? "SWRT" : "HWRT";
                    const char* eventLabel = "scatter";
                    switch (entry.eventKind) {
                        case 1u: eventLabel = "background"; break;
                        case 2u: eventLabel = "emitter"; break;
                        case 3u: eventLabel = "rectNee"; break;
                        case 4u: eventLabel = "envNee"; break;
                        case 5u: eventLabel = "specEnvNee"; break;
                        case 6u: eventLabel = "specRectNee"; break;
                        case 7u: eventLabel = "throughput"; break;
                        case 8u: eventLabel = "rectSample"; break;
                        case 9u: eventLabel = "rectEval"; break;
                        case 10u: eventLabel = "rectShadow"; break;
                        case 11u: eventLabel = "ray"; break;
                        case 12u: eventLabel = "shadingNormal"; break;
                        case 13u: eventLabel = "tbnBasis0"; break;
                        case 14u: eventLabel = "tbnBasis1"; break;
                        case 15u: eventLabel = "tbnBasis2"; break;
                        case 16u: eventLabel = "tbnBasis3"; break;
                        case 17u: eventLabel = "tbnBasis4"; break;
                        case 18u: eventLabel = "bsdfState"; break;
                        case 19u: eventLabel = "bsdfRng"; break;
                        case 20u: eventLabel = "bsdfMaterial0"; break;
                        case 21u: eventLabel = "bsdfMaterial1"; break;
                        case 22u: eventLabel = "bsdfMaterial2"; break;
                        case 23u: eventLabel = "bsdfGradients"; break;
                        default: break;
                    }
                    if (entry.eventKind >= 13u && entry.eventKind <= 16u) {
                        switch (entry.eventKind) {
                            case 13u:
                                NSLog(@"[PathDebug] %s sample=%u depth=%u kind=tbnBasis0 mat=%u "
                                      "uv=(%.6f %.6f) tangent_w=%.6f "
                                      "vtx_normal_raw=(%.4f %.4f %.4f) "
                                      "vtx_normal=(%.4f %.4f %.4f)",
                                      integratorLabel,
                                      entry.sampleIndex,
                                      entry.depth,
                                      entry.materialIndex,
                                      entry.scalars.x,
                                      entry.scalars.y,
                                      entry.scalars.z,
                                      entry.throughput.x,
                                      entry.throughput.y,
                                      entry.throughput.z,
                                      entry.aux.x,
                                      entry.aux.y,
                                      entry.aux.z);
                                break;
                            case 14u:
                                NSLog(@"[PathDebug] %s sample=%u depth=%u kind=tbnBasis1 mat=%u "
                                      "tangent_raw=(%.4f %.4f %.4f) "
                                      "tangent=(%.4f %.4f %.4f) "
                                      "bitangent=(%.4f %.4f %.4f)",
                                      integratorLabel,
                                      entry.sampleIndex,
                                      entry.depth,
                                      entry.materialIndex,
                                      entry.throughput.x,
                                      entry.throughput.y,
                                      entry.throughput.z,
                                      entry.scalars.x,
                                      entry.scalars.y,
                                      entry.scalars.z,
                                      entry.aux.x,
                                      entry.aux.y,
                                      entry.aux.z);
                                break;
                            case 15u:
                                NSLog(@"[PathDebug] %s sample=%u depth=%u kind=tbnBasis2 mat=%u "
                                      "texel_raw=(%.4f %.4f %.4f) "
                                      "texel_decoded=(%.4f %.4f %.4f) "
                                      "sn_tangent_space=(%.4f %.4f %.4f)",
                                      integratorLabel,
                                      entry.sampleIndex,
                                      entry.depth,
                                      entry.materialIndex,
                                      entry.throughput.x,
                                      entry.throughput.y,
                                      entry.throughput.z,
                                      entry.scalars.x,
                                      entry.scalars.y,
                                      entry.scalars.z,
                                      entry.aux.x,
                                      entry.aux.y,
                                      entry.aux.z);
                                break;
                            case 16u:
                                NSLog(@"[PathDebug] %s sample=%u depth=%u kind=tbnBasis3 mat=%u "
                                      "sn_world=(%.4f %.4f %.4f)",
                                      integratorLabel,
                                      entry.sampleIndex,
                                      entry.depth,
                                      entry.materialIndex,
                                      entry.aux.x,
                                      entry.aux.y,
                                      entry.aux.z);
                                break;
                            default:
                                break;
                        }
                        continue;
                    }
                    if (entry.eventKind == 17u) {
                        NSLog(@"[PathDebug] %s sample=%u depth=%u kind=tbnBasis4 mat=%u "
                              "bary=(%.6f %.6f %.6f) tri=(%.0f %.0f %.0f)",
                              integratorLabel,
                              entry.sampleIndex,
                              entry.depth,
                              entry.materialIndex,
                              entry.throughput.x,
                              entry.throughput.y,
                              entry.throughput.z,
                              entry.aux.x,
                              entry.aux.y,
                              entry.aux.z);
                        continue;
                    }
                    if (entry.eventKind == 18u) {
                        NSLog(@"[PathDebug] %s sample=%u depth=%u kind=bsdfState mat=%u "
                              "incident=(%.8f %.8f %.8f) "
                              "pdf=%.8f dirPdf=%.8f lobe=%.0f isDelta=%.0f "
                              "outgoing=(%.8f %.8f %.8f)",
                              integratorLabel,
                              entry.sampleIndex,
                              entry.depth,
                              entry.materialIndex,
                              entry.throughput.x,
                              entry.throughput.y,
                              entry.throughput.z,
                              entry.scalars.x,
                              entry.scalars.y,
                              entry.scalars.z,
                              entry.scalars.w,
                              entry.aux.x,
                              entry.aux.y,
                              entry.aux.z);
                        continue;
                    }
                    if (entry.eventKind == 19u) {
                        const uint32_t stateIn = std::bit_cast<uint32_t>(entry.scalars.x);
                        const uint32_t stateOut = std::bit_cast<uint32_t>(entry.scalars.y);
                        NSLog(@"[PathDebug] %s sample=%u depth=%u kind=bsdfRng mat=%u "
                              "rngIn=0x%08x rngOut=0x%08x type=%.0f usedRandomWalk=%.0f "
                              "shadingNormal=(%.8f %.8f %.8f) wo=(%.8f %.8f %.8f)",
                              integratorLabel,
                              entry.sampleIndex,
                              entry.depth,
                              entry.materialIndex,
                              stateIn,
                              stateOut,
                              entry.scalars.z,
                              entry.scalars.w,
                              entry.throughput.x,
                              entry.throughput.y,
                              entry.throughput.z,
                              entry.aux.x,
                              entry.aux.y,
                              entry.aux.z);
                        continue;
                    }
                    if (entry.eventKind == 20u) {
                        NSLog(@"[PathDebug] %s sample=%u depth=%u kind=bsdfMaterial0 mat=%u "
                              "uv=(%.6f %.6f) baseColor=(%.8f %.8f %.8f) "
                              "metallic=%.8f roughness=%.8f normalScale=%.8f transmission=%.8f",
                              integratorLabel,
                              entry.sampleIndex,
                              entry.depth,
                              entry.materialIndex,
                              entry.scalars.x,
                              entry.scalars.y,
                              entry.throughput.x,
                              entry.throughput.y,
                              entry.throughput.z,
                              entry.scalars.z,
                              entry.scalars.w,
                              entry.aux.x,
                              entry.aux.y);
                        continue;
                    }
                    if (entry.eventKind == 21u) {
                        NSLog(@"[PathDebug] %s sample=%u depth=%u kind=bsdfMaterial1 mat=%u "
                              "alpha=%.8f alphaCutoff=%.8f alphaMode=%.8f "
                              "emissive=(%.8f %.8f %.8f) "
                              "packedBaseColorRoughness=(%.8f %.8f %.8f %.8f)",
                              integratorLabel,
                              entry.sampleIndex,
                              entry.depth,
                              entry.materialIndex,
                              entry.scalars.x,
                              entry.scalars.y,
                              entry.scalars.z,
                              entry.throughput.x,
                              entry.throughput.y,
                              entry.throughput.z,
                              entry.aux.x,
                              entry.aux.y,
                              entry.aux.z,
                              entry.scalars.w);
                        continue;
                    }
                    if (entry.eventKind == 22u) {
                        NSLog(@"[PathDebug] %s sample=%u depth=%u kind=bsdfMaterial2 mat=%u "
                              "pbrParams=(%.8f %.8f %.8f %.8f) "
                              "pbrExtras=(%.8f %.8f %.8f %.8f)",
                              integratorLabel,
                              entry.sampleIndex,
                              entry.depth,
                              entry.materialIndex,
                              entry.scalars.x,
                              entry.scalars.y,
                              entry.scalars.z,
                              entry.scalars.w,
                              entry.throughput.x,
                              entry.throughput.y,
                              entry.throughput.z,
                              entry.aux.x);
                        continue;
                    }
                    if (entry.eventKind == 23u) {
                        NSLog(@"[PathDebug] %s sample=%u depth=%u kind=bsdfGradients mat=%u "
                              "hasIgehy=%.0f dUVdx=(%.8f %.8f) dUVdy=(%.8f %.8f) "
                              "uvPerWorld=%.8f normalLod=%.8f",
                              integratorLabel,
                              entry.sampleIndex,
                              entry.depth,
                              entry.materialIndex,
                              entry.scalars.x,
                              entry.throughput.x,
                              entry.throughput.y,
                              entry.throughput.z,
                              entry.scalars.y,
                              entry.scalars.z,
                              entry.scalars.w);
                        continue;
                    }
                    NSLog(@"[PathDebug] %s sample=%u depth=%u kind=%s medium=%u->%u event=%d "
                          "front=%u mat=%u throughput=(%.4f %.4f %.4f) "
                          "scalars=(%.6f %.6f %.6f %.6f) aux=(%.4f %.4f %.4f)",
                          integratorLabel,
                          entry.sampleIndex,
                          entry.depth,
                          eventLabel,
                          entry.mediumDepthBefore,
                          entry.mediumDepthAfter,
                          entry.mediumEvent,
                          entry.frontFace,
                          entry.materialIndex,
                          entry.throughput.x,
                          entry.throughput.y,
                          entry.throughput.z,
                          entry.scalars.x,
                          entry.scalars.y,
                          entry.scalars.z,
                          entry.scalars.w,
                          entry.aux.x,
                          entry.aux.y,
                          entry.aux.z);
                }
                m_lastDebugLogCount = debugEntryCount;
            } else if (debugEntryCount == 0) {
                m_lastDebugLogCount = 0;
            }

            parityMaxEntries = debugData->parityMaxEntries;
            uint32_t parityRecorded = debugData->parityWriteIndex;
            parityEntryCount = std::min(parityRecorded, parityMaxEntries);
            if (parityEntryCount > 0 && m_settings.parityAssertEnabled &&
                parityEntryCount != m_lastParityLogCount) {
                const uint32_t entriesToLog = std::min(parityEntryCount, 8u);
                for (uint32_t i = 0; i < entriesToLog; ++i) {
                    const auto& entry = debugData->parityEntries[i];
                    NSLog(@"[ParityAssert] idx=%u frame=%u pixel=(%u,%u) depth=%u kind=%u reason=0x%02x "
                          "tRange=[%.8f, %.8f] "
                          "hw(hit=%u t=%.8f bary=(%.8f %.8f) mat=%u mesh=%u prim=%u front=%u n=(%.8f %.8f %.8f) sn=(%.8f %.8f %.8f)) "
                          "sw(hit=%u t=%.8f bary=(%.8f %.8f) mat=%u mesh=%u prim=%u front=%u n=(%.8f %.8f %.8f) sn=(%.8f %.8f %.8f))",
                          i,
                          entry.frameIndex,
                          entry.pixelX,
                          entry.pixelY,
                          entry.depth,
                          entry.probeKind,
                          entry.reasonMask,
                          entry.tMin,
                          entry.tMax,
                          entry.hwHit,
                          entry.hwT,
                          entry.hwBarycentric.x,
                          entry.hwBarycentric.y,
                          entry.hwMaterialIndex,
                          entry.hwMeshIndex,
                          entry.hwPrimitiveIndex,
                          entry.hwFrontFace,
                          entry.hwNormal.x,
                          entry.hwNormal.y,
                          entry.hwNormal.z,
                          entry.hwShadingNormal.x,
                          entry.hwShadingNormal.y,
                          entry.hwShadingNormal.z,
                          entry.swHit,
                          entry.swT,
                          entry.swBarycentric.x,
                          entry.swBarycentric.y,
                          entry.swMaterialIndex,
                          entry.swMeshIndex,
                          entry.swPrimitiveIndex,
                          entry.swFrontFace,
                          entry.swNormal.x,
                          entry.swNormal.y,
                          entry.swNormal.z,
                          entry.swShadingNormal.x,
                          entry.swShadingNormal.y,
                          entry.swShadingNormal.z);
                }
                m_lastParityLogCount = parityEntryCount;
            } else if (parityEntryCount == 0) {
                m_lastParityLogCount = 0;
            }
        } else {
            m_lastDebugLogCount = 0;
            m_lastParityLogCount = 0;
        }
#else
        (void)debugBuffer;
        m_lastDebugLogCount = 0;
        m_lastParityLogCount = 0;
#endif
    } else {
        m_lastDebugLogCount = 0;
        m_lastParityLogCount = 0;
    }
    m_performanceStats.debugPathEntryCount = debugEntryCount;
    m_performanceStats.debugPathMaxEntries = debugMaxEntries;
    m_performanceStats.parityEntryCount = parityEntryCount;
    m_performanceStats.parityMaxEntries = parityMaxEntries;
    if (debugBuffer) {
#if PT_DEBUG_TOOLS
        const auto* debugData =
            reinterpret_cast<const PathtraceDebugBuffer*>([debugBuffer contents]);
        if (debugData) {
            m_performanceStats.parityChecksPerformed = debugData->parityChecksPerformed;
            m_performanceStats.parityChecksInMedium = debugData->parityChecksInMedium;
        } else {
            m_performanceStats.parityChecksPerformed = 0;
            m_performanceStats.parityChecksInMedium = 0;
        }
#else
        m_performanceStats.parityChecksPerformed = 0;
        m_performanceStats.parityChecksInMedium = 0;
#endif
    } else {
        m_performanceStats.parityChecksPerformed = 0;
        m_performanceStats.parityChecksInMedium = 0;
    }

    if (parityEntryCount > 0 && debugBuffer) {
#if PT_DEBUG_TOOLS
        const auto* debugData =
            reinterpret_cast<const PathtraceDebugBuffer*>([debugBuffer contents]);
        const auto& entry = debugData->parityEntries[parityEntryCount - 1u];
        m_performanceStats.parityLastReasonMask = entry.reasonMask;
        m_performanceStats.parityLastPixelX = entry.pixelX;
        m_performanceStats.parityLastPixelY = entry.pixelY;
        m_performanceStats.parityLastDepth = entry.depth;
        m_performanceStats.parityLastHwHit = entry.hwHit;
        m_performanceStats.parityLastSwHit = entry.swHit;
        m_performanceStats.parityLastHwFrontFace = entry.hwFrontFace;
        m_performanceStats.parityLastSwFrontFace = entry.swFrontFace;
        m_performanceStats.parityLastHwMaterialIndex = entry.hwMaterialIndex;
        m_performanceStats.parityLastSwMaterialIndex = entry.swMaterialIndex;
        m_performanceStats.parityLastHwMeshIndex = entry.hwMeshIndex;
        m_performanceStats.parityLastSwMeshIndex = entry.swMeshIndex;
        m_performanceStats.parityLastHwPrimitiveIndex = entry.hwPrimitiveIndex;
        m_performanceStats.parityLastSwPrimitiveIndex = entry.swPrimitiveIndex;
        m_performanceStats.parityLastHwT = entry.hwT;
        m_performanceStats.parityLastSwT = entry.swT;
#else
        (void)debugBuffer;
#endif
    } else {
        m_performanceStats.parityLastReasonMask = 0;
        m_performanceStats.parityLastPixelX = 0;
        m_performanceStats.parityLastPixelY = 0;
        m_performanceStats.parityLastDepth = 0;
        m_performanceStats.parityLastHwHit = 0;
        m_performanceStats.parityLastSwHit = 0;
        m_performanceStats.parityLastHwFrontFace = 0;
        m_performanceStats.parityLastSwFrontFace = 0;
        m_performanceStats.parityLastHwMaterialIndex = 0;
        m_performanceStats.parityLastSwMaterialIndex = 0;
        m_performanceStats.parityLastHwMeshIndex = 0;
        m_performanceStats.parityLastSwMeshIndex = 0;
        m_performanceStats.parityLastHwPrimitiveIndex = 0;
        m_performanceStats.parityLastSwPrimitiveIndex = 0;
        m_performanceStats.parityLastHwT = 0.0f;
        m_performanceStats.parityLastSwT = 0.0f;
    }

    const uint32_t previousSampleCount = m_previousSampleCount;
    m_previousSampleCount = finalSampleCount;
    uint32_t completedSamples = 0;
    if (finalSampleCount >= previousSampleCount) {
        completedSamples = finalSampleCount - previousSampleCount;
    } else {
        completedSamples = finalSampleCount;
    }

    if (m_lastFrameTimeMs > 0.0) {
        const double seconds = m_lastFrameTimeMs / 1000.0;
        m_performanceStats.samplesPerMinute =
            (seconds > 0.0 && completedSamples > 0)
                ? (static_cast<double>(completedSamples) / seconds) * 60.0
                : 0.0;
    } else {
        m_performanceStats.samplesPerMinute = 0.0;
    }

    const auto& sceneResources = activeSceneResources();
    m_performanceStats.sampleCount = finalSampleCount;
    m_performanceStats.sphereCount = sceneResources.sphereCount();
    m_performanceStats.triangleCount = sceneResources.triangleCount();

    const auto& provider = sceneResources.intersectionProvider();
    m_performanceStats.bvhNodeCount = provider.software.nodeCount;
    m_performanceStats.bvhPrimitiveCount = provider.software.primitiveCount;
    m_performanceStats.intersectionMode = static_cast<uint32_t>(provider.mode);
    m_performanceStats.hardwareRaytracingAvailable = sceneResources.supportsRaytracing();
    m_performanceStats.hardwareRaytracingActive =
        provider.mode == PathTracerShaderTypes::IntersectionMode::HardwareRayTracing;

    m_performanceStats.renderWidth =
        static_cast<uint32_t>(std::max<CGFloat>(renderSize.width, 0.0f));
    m_performanceStats.renderHeight =
        static_cast<uint32_t>(std::max<CGFloat>(renderSize.height, 0.0f));

    if (m_verboseTiming && !m_firstStatsTimingLogged) {
        double statsMs = (CFAbsoluteTimeGetCurrent() - statsStart) * 1000.0;
        NSLog(@"[Timing] StatsReadbackMs=%.2f", statsMs);
        m_firstStatsTimingLogged = true;
    }
}

void MetalRenderer::Impl::handleCameraInput() {
#if HAS_IMGUI
    if (m_options.headless) {
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    const CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
    const bool suppressionActive = now < m_cameraInputSuppressionDeadline;
    const bool pointerInsideView = isMouseInsideRenderView();

    if (suppressionActive) {
        m_cameraOrbitActive = false;
        m_cameraZoomActive = false;
        io.MouseDelta = ImVec2(0.0f, 0.0f);
        return;
    }

    if (!pointerInsideView) {
        m_cameraOrbitActive = false;
        m_cameraZoomActive = false;
        m_firstPersonLookActive = false;
    }

    if (!pointerInsideView) {
        io.MouseDelta = ImVec2(0.0f, 0.0f);
        return;
    }

    m_cameraInputSuppressionDeadline = 0.0;

    if (m_settings.cameraControlMode == RenderSettings::CameraControlMode::FirstPerson) {
        constexpr float kLookSpeed = 0.0035f;
        constexpr float kKeyboardLookSpeed = 1.75f;
        constexpr float kWheelDollyScale = 0.75f;
        bool cameraChanged = false;
        const bool keyboardAvailable = !io.WantCaptureKeyboard && !io.WantCaptureMouse;

        if (io.WantCaptureMouse || !io.MouseDown[0]) {
            m_firstPersonLookActive = false;
        } else if (io.MouseClicked[0]) {
            m_firstPersonLookActive = pointerInsideView;
            io.MouseDelta = ImVec2(0.0f, 0.0f);
        } else if (m_firstPersonLookActive) {
            const float dx = io.MouseDelta.x;
            const float dy = io.MouseDelta.y;
            if (dx != 0.0f || dy != 0.0f) {
                m_settings.cameraYaw -= dx * kLookSpeed;
                m_settings.cameraPitch -= dy * kLookSpeed;
                m_settings.cameraYaw = WrapRadians(m_settings.cameraYaw);
                m_settings.cameraPitch = std::clamp(m_settings.cameraPitch,
                                                    -kCameraPitchLimitRadians,
                                                    kCameraPitchLimitRadians);
                cameraChanged = true;
            }
        }

        const simd::float3 viewForward =
            SafeNormalize(CameraForwardFromYawPitch(m_settings.cameraYaw, m_settings.cameraPitch),
                          simd_make_float3(0.0f, 0.0f, -1.0f));
        const simd::float3 moveForward =
            SafeNormalize(simd_make_float3(viewForward.x, 0.0f, viewForward.z),
                          simd_make_float3(-1.0f, 0.0f, 0.0f));
        const simd::float3 up = simd_make_float3(0.0f, 1.0f, 0.0f);

        simd::float3 movement = simd_make_float3(0.0f, 0.0f, 0.0f);
        float verticalInput = 0.0f;
        float speed = std::max(m_settings.firstPersonMoveSpeed, 0.01f);
        if (ImGui::IsKeyDown(ImGuiKey_LeftShift) || ImGui::IsKeyDown(ImGuiKey_RightShift)) {
            speed *= 4.0f;
        }
        if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_RightCtrl)) {
            speed *= 0.25f;
        }
        const float dt = std::clamp(io.DeltaTime, 1.0f / 240.0f, 0.25f);
        if (keyboardAvailable) {
            float yawInput = 0.0f;
            if (ImGui::IsKeyDown(ImGuiKey_A)) yawInput -= 1.0f;
            if (ImGui::IsKeyDown(ImGuiKey_D)) yawInput += 1.0f;
            if (yawInput != 0.0f) {
                m_settings.cameraYaw = WrapRadians(m_settings.cameraYaw +
                                                   yawInput * kKeyboardLookSpeed * dt);
                cameraChanged = true;
            }
            if (ImGui::IsKeyDown(ImGuiKey_W)) movement += moveForward;
            if (ImGui::IsKeyDown(ImGuiKey_S)) movement -= moveForward;
            if (ImGui::IsKeyDown(ImGuiKey_Q)) verticalInput += 1.0f;
            if (ImGui::IsKeyDown(ImGuiKey_E)) verticalInput -= 1.0f;
        }
        if (!io.WantCaptureMouse && io.MouseWheel != 0.0f) {
            cameraChanged |=
                moveFirstPersonCamera(viewForward * (io.MouseWheel * speed * kWheelDollyScale));
        }

        simd::float3 desiredVelocity = simd_make_float3(0.0f, 0.0f, 0.0f);
        const float moveLenSq = simd::dot(movement, movement);
        if (moveLenSq > 1.0e-12f && std::isfinite(moveLenSq)) {
            movement /= std::sqrt(moveLenSq);
            desiredVelocity = movement * speed;
        }

        const bool manualVerticalActive = std::fabs(verticalInput) > 1.0e-4f;
        if (manualVerticalActive) {
            cameraChanged |= moveFirstPersonVertical(up * (verticalInput * speed * dt));
        }

        constexpr float kMoveResponseHz = 14.0f;
        const float velocityAlpha =
            std::clamp(1.0f - std::exp(-dt * kMoveResponseHz), 0.0f, 1.0f);
        m_firstPersonVelocity += (desiredVelocity - m_firstPersonVelocity) * velocityAlpha;

        const float velocityLenSq = simd::dot(m_firstPersonVelocity, m_firstPersonVelocity);
        if (velocityLenSq > 1.0e-8f && std::isfinite(velocityLenSq)) {
            const bool snapToWalkableGround = !manualVerticalActive;
            cameraChanged |= moveFirstPersonCamera(m_firstPersonVelocity * dt,
                                                   snapToWalkableGround);
        } else {
            m_firstPersonVelocity = simd_make_float3(0.0f, 0.0f, 0.0f);
        }

        if (simd::dot(desiredVelocity, desiredVelocity) <= 1.0e-12f &&
            simd::dot(m_firstPersonVelocity, m_firstPersonVelocity) < 1.0e-4f) {
            m_firstPersonVelocity = simd_make_float3(0.0f, 0.0f, 0.0f);
        }

        if (cameraChanged) {
            m_lastCameraInteractionTime = now;
            m_smoothYaw = m_settings.cameraYaw;
            m_smoothPitch = m_settings.cameraPitch;
            m_cameraSmoothingInitialized = true;
        }
        io.MouseDelta = ImVec2(0.0f, 0.0f);
        return;
    }

    if (io.WantCaptureMouse) {
        return;
    }

    if (!io.MouseDown[0]) {
        m_cameraOrbitActive = false;
    } else if (io.MouseClicked[0]) {
        m_cameraOrbitActive = pointerInsideView;
    }

    if (!io.MouseDown[1]) {
        m_cameraZoomActive = false;
    } else if (io.MouseClicked[1]) {
        m_cameraZoomActive = pointerInsideView;
    }

    if (!m_cameraOrbitActive && !m_cameraZoomActive) {
        if (io.MouseWheel != 0.0f) {
            constexpr float kWheelZoomSpeed = 0.12f;
            constexpr float kMinDistance = 0.5f;
            constexpr float kMaxDistance = 2000.0f;
            const float factor = std::exp(-io.MouseWheel * kWheelZoomSpeed);
            m_settings.cameraDistance = std::clamp(m_settings.cameraDistance * factor,
                                                   kMinDistance,
                                                   kMaxDistance);
            m_lastCameraInteractionTime = now;
        }
        io.MouseDelta = ImVec2(0.0f, 0.0f);
        return;
    }

    const float rotateSpeed = 0.005f;
    const float zoomSpeed = 0.01f;
    constexpr float kMinPitch = -kCameraPitchLimitRadians;
    constexpr float kMaxPitch = kCameraPitchLimitRadians;
    constexpr float kMinDistance = 0.5f;
    constexpr float kMaxDistance = 2000.0f;

    bool cameraChanged = false;

    if (io.MouseWheel != 0.0f) {
        constexpr float kWheelZoomSpeed = 0.12f;
        const float factor = std::exp(-io.MouseWheel * kWheelZoomSpeed);
        m_settings.cameraDistance = std::clamp(m_settings.cameraDistance * factor,
                                               kMinDistance,
                                               kMaxDistance);
        cameraChanged = true;
    }

    if (m_cameraOrbitActive) {
        const float dx = io.MouseDelta.x;
        const float dy = io.MouseDelta.y;
        if (dx != 0.0f || dy != 0.0f) {
            m_settings.cameraYaw -= dx * rotateSpeed;
            m_settings.cameraPitch -= dy * rotateSpeed;

            m_settings.cameraYaw = WrapRadians(m_settings.cameraYaw);

            m_settings.cameraPitch = std::clamp(m_settings.cameraPitch, kMinPitch, kMaxPitch);
            cameraChanged = true;
        }
    }

    if (m_cameraZoomActive) {
        const float dy = io.MouseDelta.y;
        if (dy != 0.0f) {
            const float factor = std::exp(dy * zoomSpeed);
            m_settings.cameraDistance = std::clamp(m_settings.cameraDistance * factor,
                                                   kMinDistance,
                                                   kMaxDistance);
            cameraChanged = true;
        }
    }

    if (cameraChanged) {
        m_lastCameraInteractionTime = now;
    }
#else
    (void)this;
#endif
}

void MetalRenderer::Impl::rebuildCameraCollisionCacheIfNeeded() {
    const PathTracer::SceneResources& resources = activeSceneResources();
    const auto& meshes = resources.meshes();
    if (m_cameraCollisionSource == &resources &&
        m_cameraCollisionMeshCount == meshes.size()) {
        return;
    }

    m_cameraCollisionSource = &resources;
    m_cameraCollisionMeshCount = meshes.size();
    m_cameraCollisionMeshes.clear();
    m_cameraCollisionMeshes.reserve(meshes.size());
    m_cameraCollisionTriangles.clear();
    m_cameraCollisionBvh = {};

    size_t totalTriangleCount = 0;
    for (const auto& mesh : meshes) {
        totalTriangleCount += mesh.indices.size() / 3u;
    }
    m_cameraCollisionTriangles.reserve(totalTriangleCount);

    for (const auto& mesh : meshes) {
        if (mesh.vertices.empty() || mesh.indices.size() < 3) {
            continue;
        }
        simd::float3 bmin{std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::max()};
        simd::float3 bmax{-std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max(),
                          -std::numeric_limits<float>::max()};
        for (const auto& vertex : mesh.vertices) {
            bmin = simd::min(bmin, vertex.position);
            bmax = simd::max(bmax, vertex.position);
        }
        CameraCollisionMesh entry{};
        entry.mesh = &mesh;
        entry.localBoundsMin = bmin;
        entry.localBoundsMax = bmax;
        m_cameraCollisionMeshes.push_back(entry);

        const uint32_t triangleCount = static_cast<uint32_t>(mesh.indices.size() / 3u);
        for (uint32_t tri = 0; tri < triangleCount; ++tri) {
            const uint32_t base = tri * 3u;
            const uint32_t i0 = mesh.indices[base + 0u];
            const uint32_t i1 = mesh.indices[base + 1u];
            const uint32_t i2 = mesh.indices[base + 2u];
            if (i0 >= mesh.vertices.size() ||
                i1 >= mesh.vertices.size() ||
                i2 >= mesh.vertices.size()) {
                continue;
            }
            PathTracerShaderTypes::TriangleData collisionTri{};
            collisionTri.v0 = simd_mul(mesh.localToWorld,
                                       simd_make_float4(mesh.vertices[i0].position, 1.0f));
            collisionTri.v1 = simd_mul(mesh.localToWorld,
                                       simd_make_float4(mesh.vertices[i1].position, 1.0f));
            collisionTri.v2 = simd_mul(mesh.localToWorld,
                                       simd_make_float4(mesh.vertices[i2].position, 1.0f));
            m_cameraCollisionTriangles.push_back(collisionTri);
        }
    }

    if (!m_cameraCollisionTriangles.empty()) {
        PathTracer::BvhBuildInput input{};
        input.triangles = m_cameraCollisionTriangles.data();
        input.triangleCount = static_cast<uint32_t>(m_cameraCollisionTriangles.size());
        PathTracer::BvhBuilder builder(8);
        m_cameraCollisionBvh = builder.build(input);
    }
}

bool MetalRenderer::Impl::traceCameraCollision(const simd::float3& origin,
                                               const simd::float3& direction,
                                               float maxDistance,
                                               float& outDistance,
                                               simd::float3& outNormal) const {
    if (!(maxDistance > 0.0f) || !std::isfinite(maxDistance)) {
        return false;
    }

    bool hit = false;
    float closest = maxDistance;
    simd::float3 closestNormal = simd_make_float3(0.0f, 1.0f, 0.0f);
    if (m_cameraCollisionBvh.nodes.empty() ||
        m_cameraCollisionBvh.primitiveIndices.empty() ||
        m_cameraCollisionTriangles.empty()) {
        return false;
    }

    std::array<uint32_t, 96> stack{};
    uint32_t stackSize = 0;
    stack[stackSize++] = 0u;
    while (stackSize > 0) {
        const uint32_t nodeIndex = stack[--stackSize];
        if (nodeIndex >= m_cameraCollisionBvh.nodes.size()) {
            continue;
        }
        const PathTracerShaderTypes::BvhNode& node = m_cameraCollisionBvh.nodes[nodeIndex];
        if (!IntersectAabb(origin,
                           direction,
                           simd_make_float3(node.boundsMin),
                           simd_make_float3(node.boundsMax),
                           closest)) {
            continue;
        }

        constexpr uint32_t kInvalidIndex = std::numeric_limits<uint32_t>::max();
        if (node.leftChild == kInvalidIndex && node.rightChild == kInvalidIndex) {
            for (uint32_t i = 0; i < node.primitiveCount; ++i) {
                const uint32_t primOffset = node.primitiveOffset + i;
                if (primOffset >= m_cameraCollisionBvh.primitiveIndices.size()) {
                    continue;
                }
                const uint32_t triIndex = m_cameraCollisionBvh.primitiveIndices[primOffset];
                if (triIndex >= m_cameraCollisionTriangles.size()) {
                    continue;
                }
                const PathTracerShaderTypes::TriangleData& tri = m_cameraCollisionTriangles[triIndex];
                float t = closest;
                simd::float3 normal{};
                if (IntersectTriangle(origin,
                                      direction,
                                      simd_make_float3(tri.v0),
                                      simd_make_float3(tri.v1),
                                      simd_make_float3(tri.v2),
                                      closest,
                                      t,
                                      normal)) {
                    closest = t;
                    closestNormal = normal;
                    hit = true;
                }
            }
        } else {
            if (node.leftChild != kInvalidIndex && stackSize < stack.size()) {
                stack[stackSize++] = node.leftChild;
            }
            if (node.rightChild != kInvalidIndex && stackSize < stack.size()) {
                stack[stackSize++] = node.rightChild;
            }
        }
    }

    if (!hit) {
        return false;
    }
    outDistance = closest;
    outNormal = closestNormal;
    return true;
}

bool MetalRenderer::Impl::moveFirstPersonCamera(const simd::float3& displacement,
                                                bool snapToWalkableGround) {
    constexpr float kCameraRadius = 0.20f;
    constexpr float kSkin = 0.03f;
    constexpr float kMaxStepUp = 0.45f;
    constexpr float kMaxStepDown = 0.75f;
    constexpr float kMinWalkableNormalY = 0.45f;
    constexpr float kMinEyeHeight = 0.7f;
    constexpr float kMaxEyeHeight = 2.2f;
    const float distanceSq = simd::dot(displacement, displacement);
    if (!(distanceSq > 1.0e-12f) || !std::isfinite(distanceSq)) {
        return false;
    }

    rebuildCameraCollisionCacheIfNeeded();
    const float distance = std::sqrt(distanceSq);
    const simd::float3 direction = displacement / distance;
    const simd::float3 origin = m_settings.firstPersonCameraPosition;
    float allowedDistance = distance;
    simd::float3 hitNormal = simd_make_float3(0.0f, 1.0f, 0.0f);
    bool blocked = false;

    auto testSweepRay = [&](const simd::float3& rayOrigin, float clearance) {
        float hitDistance = distance + clearance + kSkin;
        simd::float3 normal{};
        if (!traceCameraCollision(rayOrigin,
                                  direction,
                                  distance + clearance + kSkin,
                                  hitDistance,
                                  normal)) {
            return;
        }
        allowedDistance = std::min(allowedDistance,
                                   std::max(hitDistance - clearance - kSkin, 0.0f));
        hitNormal = normal;
        blocked = true;
    };

    testSweepRay(origin, kCameraRadius);

    const simd::float3 worldUp = simd_make_float3(0.0f, 1.0f, 0.0f);
    simd::float3 side = SafeNormalize(simd::cross(direction, worldUp),
                                      simd_make_float3(1.0f, 0.0f, 0.0f));
    simd::float3 up = SafeNormalize(simd::cross(side, direction), worldUp);
    testSweepRay(origin + side * kCameraRadius, 0.0f);
    testSweepRay(origin - side * kCameraRadius, 0.0f);
    testSweepRay(origin + up * kCameraRadius, 0.0f);
    testSweepRay(origin - up * kCameraRadius, 0.0f);

    bool moved = false;
    if (blocked) {
        const float appliedDistance = std::min(allowedDistance, distance);
        m_settings.firstPersonCameraPosition += direction * appliedDistance;
        moved = appliedDistance > 1.0e-5f;
        const float intoWall = simd::dot(m_firstPersonVelocity, hitNormal);
        if (intoWall < 0.0f) {
            m_firstPersonVelocity -= hitNormal * intoWall;
        }
    } else {
        m_settings.firstPersonCameraPosition += displacement;
        moved = true;
    }

    if (snapToWalkableGround && moved) {
        float floorDistance = 0.0f;
        simd::float3 floorNormal{};
        const simd::float3 down = simd_make_float3(0.0f, -1.0f, 0.0f);
        if (!m_firstPersonEyeHeightInitialized &&
            traceCameraCollision(m_settings.firstPersonCameraPosition,
                                 down,
                                 3.0f,
                                 floorDistance,
                                 floorNormal) &&
            floorDistance >= kMinEyeHeight &&
            floorDistance <= kMaxEyeHeight &&
            floorNormal.y >= kMinWalkableNormalY) {
            m_firstPersonEyeHeight = floorDistance;
            m_firstPersonEyeHeightInitialized = true;
        }

        if (m_firstPersonEyeHeightInitialized &&
            traceCameraCollision(m_settings.firstPersonCameraPosition,
                                 down,
                                 m_firstPersonEyeHeight + kMaxStepDown,
                                 floorDistance,
                                 floorNormal) &&
            floorNormal.y >= kMinWalkableNormalY) {
            const float verticalCorrection = m_firstPersonEyeHeight - floorDistance;
            if (verticalCorrection <= kMaxStepUp &&
                verticalCorrection >= -kMaxStepDown) {
                const float maxSnapThisMove =
                    std::clamp(distance * 1.25f, 0.02f, 0.08f);
                const float clampedCorrection =
                    std::clamp(verticalCorrection, -maxSnapThisMove, maxSnapThisMove);
                m_settings.firstPersonCameraPosition.y += clampedCorrection;
                moved = moved || std::fabs(clampedCorrection) > 1.0e-5f;
            }
        }
    }

    return moved;
}

bool MetalRenderer::Impl::moveFirstPersonVertical(const simd::float3& displacement) {
    const float dy = displacement.y;
    if (std::fabs(dy) <= 1.0e-6f || !std::isfinite(dy)) {
        return false;
    }

    m_settings.firstPersonCameraPosition.y += dy;
    m_firstPersonVelocity.y = 0.0f;
    m_firstPersonEyeHeightInitialized = false;
    return true;
}

void MetalRenderer::Impl::syncCameraSmoothingToTarget() {
    if (!m_options.headless) {
        MakeExplicitCameraOrbitable(m_settings);
    }
    m_settings.cameraYaw = WrapRadians(m_settings.cameraYaw);
    m_settings.cameraPitch = std::clamp(m_settings.cameraPitch,
                                        -kCameraPitchLimitRadians,
                                        kCameraPitchLimitRadians);
    m_smoothYaw = m_settings.cameraYaw;
    m_smoothPitch = m_settings.cameraPitch;
    m_cameraSmoothingInitialized = true;
}

void MetalRenderer::Impl::updateCameraSmoothing(float deltaSeconds) {
    if (!m_cameraSmoothingInitialized) {
        syncCameraSmoothingToTarget();
        return;
    }

    if (!std::isfinite(deltaSeconds) || deltaSeconds <= 0.0f) {
        deltaSeconds = 1.0f / 60.0f;
    }
    deltaSeconds = std::clamp(deltaSeconds, 1.0f / 240.0f, 0.25f);

    const float alpha = std::clamp(1.0f - std::exp(-deltaSeconds * kCameraSmoothingCutoffHz),
                                   0.0f,
                                   1.0f);

    float yawTarget = WrapRadians(m_settings.cameraYaw);
    float yawDelta = ShortestAngleDelta(yawTarget, m_smoothYaw);
    m_smoothYaw = WrapRadians(m_smoothYaw + yawDelta * alpha);

    float pitchTarget = std::clamp(m_settings.cameraPitch,
                                   -kCameraPitchLimitRadians,
                                   kCameraPitchLimitRadians);
    m_settings.cameraPitch = pitchTarget;
    m_smoothPitch += (pitchTarget - m_smoothPitch) * alpha;
    m_smoothPitch = std::clamp(m_smoothPitch,
                               -kCameraPitchLimitRadians,
                               kCameraPitchLimitRadians);
}
uint32_t MetalRenderer::Impl::addMaterial(const simd::float3& baseColor,
                                          float roughness,
                                          MaterialType type,
                                          float indexOfRefraction,
                                          const simd::float3& conductorEta,
                                          const simd::float3& conductorK,
                                          bool hasConductorParameters) {
    return m_sceneResources.addMaterial(baseColor,
                                        roughness,
                                        type,
                                        indexOfRefraction,
                                        simd_make_float3(0.0f, 0.0f, 0.0f),
                                        /*emissionUsesEnvironment=*/false,
                                        conductorEta,
                                        conductorK,
                                        hasConductorParameters,
                                        /*coatRoughness=*/0.0f,
                                        /*coatThickness=*/0.0f,
                                        simd_make_float3(1.0f, 1.0f, 1.0f),
                                        simd_make_float3(0.0f, 0.0f, 0.0f),
                                        /*coatIndexOfRefraction=*/1.5f,
                                        simd_make_float3(0.0f, 0.0f, 0.0f),
                                        simd_make_float3(0.0f, 0.0f, 0.0f),
                                        simd_make_float3(0.0f, 0.0f, 0.0f),
                                        /*sssMeanFreePath=*/0.0f,
                                        /*sssAnisotropy=*/0.0f,
                                        /*sssMethod=*/0u,
                                        /*sssCoatEnabled=*/false,
                                        /*sssSigmaOverride=*/false,
                                        /*carpaintBaseMetallic=*/0.0f,
                                        /*carpaintBaseRoughness=*/0.0f,
                                        /*carpaintFlakeSampleWeight=*/0.0f,
                                        /*carpaintFlakeRoughness=*/0.0f,
                                        /*carpaintFlakeAnisotropy=*/0.0f,
                                        /*carpaintFlakeNormalStrength=*/0.0f,
                                        /*carpaintFlakeScale=*/1.0f,
                                        /*carpaintFlakeReflectanceScale=*/1.0f,
                                        simd_make_float3(0.0f, 0.0f, 0.0f),
                                        simd_make_float3(0.0f, 0.0f, 0.0f),
                                        /*carpaintHasBaseConductor=*/false,
                                        simd_make_float3(1.0f, 1.0f, 1.0f));
}

void MetalRenderer::Impl::addSphere(const simd::float3& center, float radius, uint32_t materialIndex) {
    m_sceneResources.addSphere(center, radius, materialIndex);
}

PathTracer::SceneResources& MetalRenderer::Impl::activeSceneResources() {
    if (m_externalSceneResources) {
        return *m_externalSceneResources;
    }
    if (m_loadedSceneResources) {
        return *m_loadedSceneResources;
    }
    return m_sceneResources;
}

const PathTracer::SceneResources& MetalRenderer::Impl::activeSceneResources() const {
    if (m_externalSceneResources) {
        return *m_externalSceneResources;
    }
    if (m_loadedSceneResources) {
        return *m_loadedSceneResources;
    }
    return m_sceneResources;
}

void MetalRenderer::Impl::cancelPendingSceneLoad() {
    if (m_sceneLoadGeneration) {
        m_sceneLoadGeneration->fetch_add(1u, std::memory_order_acq_rel);
    }
    m_queuedSceneLoad.reset();
    m_pendingSceneLoadResult.reset();
    m_sceneLoadExclusiveActive = false;
    m_sceneSwapDrainCommandBuffer = nil;
    m_sceneSwapDrainStartTime = 0.0;
    m_sceneSwapDrainWarningLogged = false;
    m_sceneLoadStage = SceneLoadStage::Idle;
}

bool MetalRenderer::Impl::sceneLoadUiOnlyInProgress() const {
    if (m_options.headless) {
        return false;
    }
    if (m_sceneLoadStage == SceneLoadStage::Activate) {
        return true;
    }
    return m_sceneLoadExclusiveActive && m_sceneLoadStage == SceneLoadStage::LoadScene;
}

bool MetalRenderer::Impl::shouldUseExclusiveSceneSwitch(const std::string& scenePath,
                                                        const std::string& activeSceneId,
                                                        const RenderSettings& startingSettings,
                                                        std::string* reason) const {
    if (m_options.headless) {
        return false;
    }

    const auto& active = activeSceneResources();
    const auto activeReport = active.sceneMemoryReport();
    const bool hasActiveScene =
        !m_activeSceneId.empty() &&
        (activeReport.triangleCount > 0u ||
         activeReport.meshCount > 0u ||
         activeReport.textureCount > 0u ||
         activeReport.totalEstimatedMemoryBytes > 0u);

    if (hasActiveScene) {
        if (reason) {
            std::ostringstream out;
            out << "scene switch is an explicit memory-release point"
                << " active='" << m_activeSceneId << "'"
                << " estimated=" << FormatMiB(activeReport.totalEstimatedMemoryBytes)
                << " geometry=" << FormatMiB(activeReport.geometryMemoryBytes)
                << " textures=" << FormatMiB(activeReport.textureAllocatedMemoryBytes)
                << " triangles=" << activeReport.triangleCount;
            *reason = out.str();
        }
        return true;
    }

    const SceneSwitchPreflight incoming =
        AnalyzeSceneSwitchPreflight(scenePath, activeSceneId, m_sceneManager.sceneDirectory());

    const bool activeLargeByReport =
        activeReport.totalEstimatedMemoryBytes >= kExclusiveActiveSceneBytes ||
        activeReport.geometryMemoryBytes >= kExclusiveActiveGeometryBytes ||
        activeReport.textureAllocatedMemoryBytes >= kExclusiveActiveTextureBytes ||
        activeReport.triangleCount >= kExclusiveActiveTriangleCount;
    const bool activeLargeByName = SceneNameLooksLarge(m_activeSceneId, m_activeSceneId);
    if (hasActiveScene && (activeLargeByReport || activeLargeByName)) {
        if (reason) {
            std::ostringstream out;
            out << "active scene is large"
                << " id='" << m_activeSceneId << "'"
                << " estimated=" << FormatMiB(activeReport.totalEstimatedMemoryBytes)
                << " geometry=" << FormatMiB(activeReport.geometryMemoryBytes)
                << " textures=" << FormatMiB(activeReport.textureAllocatedMemoryBytes)
                << " triangles=" << activeReport.triangleCount;
            *reason = out.str();
        }
        return true;
    }

    if (incoming.metadataExclusive ||
        (incoming.metadataLarge && incoming.referencedAssetBytes >= kExclusiveIncomingAssetBytes)) {
        if (reason) {
            std::ostringstream out;
            out << "incoming scene metadata requests exclusive/large load"
                << " referencedAssets=" << FormatMiB(incoming.referencedAssetBytes);
            *reason = out.str();
        }
        return true;
    }

    uint64_t recommendedWorkingSet = activeReport.recommendedMaxWorkingSetSizeBytes;
    if (recommendedWorkingSet == 0u) {
        recommendedWorkingSet = DeviceRecommendedWorkingSetBytes(m_context.device());
    }
    const uint64_t workingSetLimit = InteractiveWorkingSetLimitBytes(recommendedWorkingSet);
    const uint64_t currentAllocated = CurrentAllocatedSizeBytes(m_context.device());
    const uint64_t activeResidentEstimate =
        std::max(activeReport.totalEstimatedMemoryBytes, currentAllocated);
    const uint64_t incomingProxyBytes = incoming.referencedAssetBytes;
    const uint64_t projectedOverlap = SaturatingAdd(activeResidentEstimate, incomingProxyBytes);

    if (hasActiveScene &&
        workingSetLimit > 0u &&
        activeResidentEstimate > workingSetLimit) {
        if (reason) {
            std::ostringstream out;
            out << "active Metal residency already exceeds interactive working-set limit"
                << " active=" << FormatMiB(activeResidentEstimate)
                << " activeEstimated=" << FormatMiB(activeReport.totalEstimatedMemoryBytes)
                << " currentAllocated=" << FormatMiB(currentAllocated)
                << " limit=" << FormatMiB(workingSetLimit);
            *reason = out.str();
        }
        return true;
    }

    if (hasActiveScene &&
        incomingProxyBytes >= kExclusiveIncomingAssetBytes &&
        (InteractiveTransportHeavySceneSwitch(startingSettings) ||
         activeResidentEstimate >= kExclusiveActiveSceneBytes)) {
        if (reason) {
            std::ostringstream out;
            out << "heavy incoming scene requires exclusive load while interactive "
                   "transport/explicit resources are resident"
                << " activeResident=" << FormatMiB(activeResidentEstimate)
                << " activeEstimated=" << FormatMiB(activeReport.totalEstimatedMemoryBytes)
                << " currentAllocated=" << FormatMiB(currentAllocated)
                << " incomingProxy=" << FormatMiB(incomingProxyBytes);
            *reason = out.str();
        }
        return true;
    }

    if (hasActiveScene &&
        workingSetLimit > 0u &&
        incomingProxyBytes >= kExclusiveIncomingAssetBytes &&
        projectedOverlap > workingSetLimit) {
        if (reason) {
            std::ostringstream out;
            out << "projected overlap exceeds interactive working-set limit"
                << " active=" << FormatMiB(activeResidentEstimate)
                << " incomingProxy=" << FormatMiB(incomingProxyBytes)
                << " limit=" << FormatMiB(workingSetLimit);
            *reason = out.str();
        }
        return true;
    }

    if (hasActiveScene &&
        incoming.knownLargeName &&
        incomingProxyBytes >= kExclusiveIncomingAssetBytes &&
        activeReport.totalEstimatedMemoryBytes >= (kExclusiveActiveSceneBytes / 2u)) {
        if (reason) {
            std::ostringstream out;
            out << "large incoming scene while active scene has nontrivial residency"
                << " activeEstimated=" << FormatMiB(activeReport.totalEstimatedMemoryBytes)
                << " incomingProxy=" << FormatMiB(incomingProxyBytes);
            *reason = out.str();
        }
        return true;
    }

    if (reason) {
        std::ostringstream out;
        out << "async load allowed"
            << " active=" << FormatMiB(activeResidentEstimate)
            << " incomingProxy=" << FormatMiB(incomingProxyBytes)
            << " meshes=" << incoming.meshReferenceCount;
        *reason = out.str();
    }
    return false;
}

void MetalRenderer::Impl::releaseActiveSceneForExclusiveLoad(const std::string& nextSceneId) {
    if (m_options.headless || m_sceneLoadExclusiveActive) {
        return;
    }

    NSLog(@"[SceneLoad] exclusive load for '%s': releasing active scene '%s'",
          nextSceneId.c_str(),
          m_activeSceneId.empty() ? "<none>" : m_activeSceneId.c_str());

    if (id<MTLCommandQueue> queue = m_context.commandQueue()) {
        id<MTLCommandBuffer> drain = [queue commandBuffer];
        if (drain) {
            drain.label = @"Exclusive Scene Load Drain";
            [drain commit];
            [drain waitUntilCompleted];
        }
    }
    m_pathTraceFrameInFlight.store(false, std::memory_order_release);
    m_interactiveUiFrameInFlight.store(false, std::memory_order_release);

    m_denoiser.releaseMetalStagingBuffers();
    m_externalSceneResources = nullptr;
    if (m_loadedSceneResources) {
        m_loadedSceneResources->clear();
        m_loadedSceneResources.reset();
    }
    m_sceneResources.clear();
    m_cameraCollisionSource = nullptr;
    m_cameraCollisionMeshes.clear();
    m_cameraCollisionTriangles.clear();
    m_cameraCollisionBvh = PathTracer::BvhBuildOutput{};
    resetFirstPersonMotion();
    resetAccumulation();
    if (id<MTLCommandQueue> queue = m_context.commandQueue()) {
        id<MTLCommandBuffer> drain = [queue commandBuffer];
        if (drain) {
            drain.label = @"Exclusive Scene Release Drain";
            [drain commit];
            [drain waitUntilCompleted];
        }
    }
    m_sceneLoadExclusiveActive = true;
}

bool MetalRenderer::Impl::queueSceneLoadFromResolvedPath(const std::string& scenePath,
                                                         const std::string& activeSceneId,
                                                         const RenderSettings& startingSettings) {
    if (scenePath.empty()) {
        return false;
    }

    if (m_sceneLoadStage != SceneLoadStage::Idle) {
        NSLog(@"[SceneLoad] cancelling staged load in favor of '%s'", activeSceneId.c_str());
        cancelPendingSceneLoad();
    }

    std::string exclusiveReason;
    const bool exclusiveLoad =
        shouldUseExclusiveSceneSwitch(scenePath, activeSceneId, startingSettings, &exclusiveReason);
    if (exclusiveLoad) {
        NSLog(@"[SceneLoad] exclusive switch selected for '%s': %s",
              activeSceneId.c_str(),
              exclusiveReason.c_str());
        releaseActiveSceneForExclusiveLoad(activeSceneId);
    } else if (!exclusiveReason.empty()) {
        NSLog(@"[SceneLoad] async switch selected for '%s': %s",
              activeSceneId.c_str(),
              exclusiveReason.c_str());
    }

    if (SceneLoadDiagnosticsEnabled()) {
        const auto& active = activeSceneResources();
        const MeshAggregateStats activeMeshStats = CollectMeshAggregateStats(active);
        const GpuBufferSummary activeBuffers = CollectGpuBufferSummary(active);
        const auto activeReport = active.sceneMemoryReport();
        const uint64_t allocatedBytes = CurrentAllocatedSizeBytes(m_context.device());
        NSLog(@"[SceneDiag] queueLoad id='%s' path='%s' route=worker-sync-parser "
              "active{materials=%u meshes=%u vertices=%llu indices=%llu triangles=%u "
              "coreBuffers=%u meshBuffers=%u coreMiB=%.2f meshMiB=%.2f geomMiB=%.2f texMiB=%.2f asMiB=%.2f} "
              "metalAllocatedMiB=%.2f",
              activeSceneId.c_str(),
              scenePath.c_str(),
              active.materialCount(),
              activeMeshStats.meshCount,
              static_cast<unsigned long long>(activeMeshStats.vertexCount),
              static_cast<unsigned long long>(activeMeshStats.indexCount),
              active.triangleCount(),
              activeBuffers.coreCount,
              activeBuffers.meshCount,
              BytesToMiB(activeBuffers.coreBytes),
              BytesToMiB(activeBuffers.meshBytes),
              BytesToMiB(activeReport.geometryMemoryBytes),
              BytesToMiB(activeReport.textureAllocatedMemoryBytes),
              BytesToMiB(activeReport.blasMemoryBytes + activeReport.tlasMemoryBytes + activeReport.scratchMemoryBytes),
              BytesToMiB(allocatedBytes));
    }

    m_queuedSceneLoad = SceneLoadRequest{scenePath, activeSceneId, startingSettings, exclusiveLoad};
    startQueuedSceneLoadIfPossible();
    return true;
}

void MetalRenderer::Impl::startQueuedSceneLoadIfPossible() {
    if (m_shutdown ||
        m_sceneLoadWorkerActive ||
        m_sceneLoadStage != SceneLoadStage::Idle ||
        !m_queuedSceneLoad.has_value()) {
        return;
    }

    const SceneLoadRequest request = *m_queuedSceneLoad;
    const uint64_t generation =
        m_sceneLoadGeneration ? (m_sceneLoadGeneration->fetch_add(1u, std::memory_order_acq_rel) + 1u) : 1u;
    m_pendingSceneLoadResult.reset();
    m_sceneSwapDrainCommandBuffer = nil;
    m_sceneSwapDrainStartTime = 0.0;
    m_sceneSwapDrainWarningLogged = false;
    m_sceneLoadStage = SceneLoadStage::LoadScene;
    m_sceneLoadWorkerActive = true;
    launchSceneLoadWorker(request, generation);

    NSLog(@"[SceneLoad] staged load id='%s' path='%s' exclusive=%d",
          request.activeSceneId.c_str(),
          request.scenePath.c_str(),
          request.exclusive ? 1 : 0);
    if (SceneLoadDiagnosticsEnabled()) {
        NSLog(@"[SceneDiag] loadPath id='%s' parser=SceneManager.sync workerThread=1 activation=mainThread",
              request.activeSceneId.c_str());
    }
}

void MetalRenderer::Impl::launchSceneLoadWorker(SceneLoadRequest request, uint64_t generation) {
    std::weak_ptr<Impl> weakSelf = shared_from_this();
    id<MTLDevice> device = m_context.device();
    const bool supportsRaytracing = m_context.supportsRaytracing();
    const bool forceSoftwareRayTracing = m_options.enableSoftwareRayTracing;
    const std::string sceneDirectory = m_sceneManager.sceneDirectory();
    std::shared_ptr<std::atomic<uint64_t>> generationToken = m_sceneLoadGeneration;

    dispatch_queue_t workerQueue = m_sceneLoadWorkerQueue
        ? m_sceneLoadWorkerQueue
        : dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

    dispatch_async(workerQueue, ^{
        auto completion = std::make_shared<SceneLoadCompleted>();
        completion->activeSceneId = request.activeSceneId;
        completion->settings = request.startingSettings;
        completion->success = false;
        completion->exclusive = request.exclusive;

        auto isCancelled = [&]() -> bool {
            return generationToken &&
                   generationToken->load(std::memory_order_acquire) != generation;
        };

        if (isCancelled()) {
            return;
        }

        if (!device) {
            completion->errorMessage = "Metal device unavailable";
        } else {
            completion->resources = std::make_unique<PathTracer::SceneResources>();
            id<MTLCommandQueue> loadQueue = [device newCommandQueue];
            if (loadQueue) {
                loadQueue.label = @"Scene Load Queue";
            }
            completion->resources->initialize(device, loadQueue, supportsRaytracing);
            const PathTracer::TextureLoadPolicy interactiveTexturePolicy =
                MakeInteractiveTextureLoadPolicy(device);
            completion->resources->setTextureLoadPolicy(interactiveTexturePolicy);
            NSLog(@"[SceneLoad] interactive texture policy maxDimension=%u maxTextureBytes=%.2f MiB",
                  interactiveTexturePolicy.maxDimension,
                  BytesToMiB(interactiveTexturePolicy.maxTextureBytes));

            SceneManager workerSceneManager(sceneDirectory);
            RenderSettings loadedSettings = request.startingSettings;
            std::string loadError;
            if (!workerSceneManager.loadSceneFromPath(request.scenePath,
                                                      *completion->resources,
                                                      loadedSettings,
                                                      &loadError)) {
                completion->errorMessage =
                    loadError.empty() ? "Failed to load scene" : loadError;
            } else if (isCancelled()) {
                return;
            } else {
                if (SceneLoadDiagnosticsEnabled()) {
                    const MeshAggregateStats parsedMeshStats = CollectMeshAggregateStats(*completion->resources);
                    NSLog(@"[SceneDiag] workerParseComplete id='%s' path='%s' materials=%u meshes=%u vertices=%llu indices=%llu triangles=%u",
                          request.activeSceneId.c_str(),
                          request.scenePath.c_str(),
                          completion->resources->materialCount(),
                          parsedMeshStats.meshCount,
                          static_cast<unsigned long long>(parsedMeshStats.vertexCount),
                          static_cast<unsigned long long>(parsedMeshStats.indexCount),
                          completion->resources->triangleCount());
                }
                if (forceSoftwareRayTracing) {
                    loadedSettings.enableSoftwareRayTracing = true;
                }
                loadedSettings.renderScale = ClampRenderScale(loadedSettings.renderScale);
                completion->resources->setSoftwareRayTracingOverride(loadedSettings.enableSoftwareRayTracing);

                using BackgroundMode = RenderSettings::BackgroundMode;
                if (loadedSettings.backgroundMode == BackgroundMode::Environment &&
                    !loadedSettings.environmentMapPath.empty()) {
                    EnvGpuHandles handles;
                    if (!completion->resources->reloadEnvironmentIfNeeded(loadedSettings.environmentMapPath,
                                                                          &handles)) {
                        completion->resources->clearEnvironmentMap();
                        loadedSettings.backgroundMode = BackgroundMode::Gradient;
                        loadedSettings.environmentMapPath.clear();
                    }
                    loadedSettings.environmentMapDirty = false;
                } else {
                    completion->resources->clearEnvironmentMap();
                    loadedSettings.environmentMapPath.clear();
                    loadedSettings.environmentMapDirty = false;
                }

                if (isCancelled()) {
                    return;
                }

                PathTracer::SceneAccelBuildStats prebuildAccelEstimate =
                    completion->resources->estimateAccelerationStructureBuildStats();
                auto prebuildMemoryReport = completion->resources->sceneMemoryReport();
                const uint64_t recommendedWorkingSet =
                    prebuildMemoryReport.recommendedMaxWorkingSetSizeBytes;
                const uint64_t workingSetLimit =
                    InteractiveWorkingSetLimitBytes(recommendedWorkingSet);
                auto projectedResidentWorkingSet = [&]() -> uint64_t {
                    uint64_t projected = CurrentAllocatedSizeBytes(device);
                    if (projected == 0u) {
                        projected = SaturatingAdd(prebuildMemoryReport.geometryMemoryBytes,
                                                  prebuildMemoryReport.textureAllocatedMemoryBytes);
                    }
                    projected = SaturatingAdd(projected, prebuildAccelEstimate.blasMemoryBytes);
                    projected = SaturatingAdd(projected, prebuildAccelEstimate.tlasMemoryBytes);
                    projected = SaturatingAdd(projected, prebuildAccelEstimate.scratchMemoryBytes);
                    return projected;
                };
                uint64_t projectedWorkingSet = projectedResidentWorkingSet();
                if (workingSetLimit > 0u && projectedWorkingSet > workingSetLimit) {
                    const bool prebuildHardwareRt = completion->resources->hardwareRaytracingEnabled();
                    NSLog(@"[SceneLoad] projected %s working set exceeds interactive soft limit; "
                          "continuing requested build "
                          "(projected=%.2f MiB softLimit=%.2f MiB recommendedMaxWorkingSetSize=%.2f MiB)",
                          prebuildHardwareRt ? "hardware RT" : "software BVH",
                          BytesToMiB(projectedWorkingSet),
                          BytesToMiB(workingSetLimit),
                          BytesToMiB(recommendedWorkingSet));
                }

                if (completion->resources && SceneLoadDiagnosticsEnabled()) {
                    const MeshAggregateStats uploadMeshStats = CollectMeshAggregateStats(*completion->resources);
                    NSLog(@"[SceneDiag] workerRebuildBegin id='%s' materials=%u meshes=%u vertices=%llu indices=%llu triangles=%u",
                          request.activeSceneId.c_str(),
                          completion->resources->materialCount(),
                          uploadMeshStats.meshCount,
                          static_cast<unsigned long long>(uploadMeshStats.vertexCount),
                          static_cast<unsigned long long>(uploadMeshStats.indexCount),
                          completion->resources->triangleCount());
                }
                if (completion->resources) {
                    completion->resources->rebuildAccelerationStructures();
                }
                if (isCancelled()) {
                    return;
                }
                if (completion->resources && SceneLoadDiagnosticsEnabled()) {
                    const auto report = completion->resources->sceneMemoryReport();
                    const auto accelStats = completion->resources->accelerationBuildStats();
                    const GpuBufferSummary gpuSummary = CollectGpuBufferSummary(*completion->resources);
                    NSLog(@"[SceneDiag] workerRebuildDone id='%s' dirty=%d mode=%s coreBuffers=%u meshBuffers=%u "
                          "coreMiB=%.2f meshMiB=%.2f geomMiB=%.2f texMiB=%.2f blasMiB=%.2f tlasMiB=%.2f scratchMiB=%.2f "
                          "blasCount=%u tlasInstances=%u primitives=%u",
                          request.activeSceneId.c_str(),
                          completion->resources->isDirty() ? 1 : 0,
                          (completion->resources->intersectionProvider().mode ==
                           PathTracerShaderTypes::IntersectionMode::HardwareRayTracing)
                              ? "hardware"
                              : "software",
                          gpuSummary.coreCount,
                          gpuSummary.meshCount,
                          BytesToMiB(gpuSummary.coreBytes),
                          BytesToMiB(gpuSummary.meshBytes),
                          BytesToMiB(report.geometryMemoryBytes),
                          BytesToMiB(report.textureAllocatedMemoryBytes),
                          BytesToMiB(accelStats.blasAllocatedMemoryBytes),
                          BytesToMiB(accelStats.tlasAllocatedMemoryBytes),
                          BytesToMiB(accelStats.retainedScratchMemoryBytes),
                          accelStats.blasCount,
                          accelStats.instanceCount,
                          accelStats.primitiveCount);
                }
                if (!completion->resources) {
                    // Error already recorded by an earlier load/build stage.
                } else if (completion->resources->isDirty()) {
                    completion->errorMessage = "Failed to build scene acceleration structures";
                    completion->resources.reset();
                } else {
                    auto memoryReport = completion->resources->sceneMemoryReport();
                    const uint64_t recommendedWorkingSet = memoryReport.recommendedMaxWorkingSetSizeBytes;
                    uint64_t allocatedWorkingSet = CurrentAllocatedSizeBytes(device);
                    const uint64_t postBuildLimit =
                        InteractiveWorkingSetLimitBytes(recommendedWorkingSet);
                    const bool overRecommended =
                        (recommendedWorkingSet > 0u) && (allocatedWorkingSet > recommendedWorkingSet);
                    bool builtHardwareRt =
                        completion->resources->intersectionProvider().mode ==
                        PathTracerShaderTypes::IntersectionMode::HardwareRayTracing;

                    if (overRecommended && builtHardwareRt && !forceSoftwareRayTracing) {
                        NSLog(@"[SceneLoad] retrying in software BVH due to memory pressure "
                              "(currentAllocatedSize=%.2f MiB recommendedMaxWorkingSetSize=%.2f MiB)",
                              BytesToMiB(allocatedWorkingSet),
                              BytesToMiB(recommendedWorkingSet));
                        completion->resources->setSoftwareRayTracingOverride(true);
                        completion->resources->rebuildAccelerationStructures();
                        if (completion->resources->isDirty()) {
                            completion->errorMessage =
                                "Hardware RT exceeded memory budget and software BVH fallback failed";
                            completion->resources.reset();
                        } else {
                            loadedSettings.enableSoftwareRayTracing = true;
                            memoryReport = completion->resources->sceneMemoryReport();
                            allocatedWorkingSet = CurrentAllocatedSizeBytes(device);
                            builtHardwareRt =
                                completion->resources->intersectionProvider().mode ==
                                PathTracerShaderTypes::IntersectionMode::HardwareRayTracing;
                        }
                    }

                    if (completion->resources &&
                        postBuildLimit > 0u &&
                        allocatedWorkingSet > postBuildLimit) {
                        NSLog(@"[SceneLoad] warning: scene exceeds interactive soft memory limit after AS build "
                              "(currentAllocatedSize=%.2f MiB softLimit=%.2f MiB recommendedMaxWorkingSetSize=%.2f MiB "
                              "estimatedSceneWorkingSet=%.2f MiB mode=%s)",
                              BytesToMiB(allocatedWorkingSet),
                              BytesToMiB(postBuildLimit),
                              BytesToMiB(recommendedWorkingSet),
                              BytesToMiB(memoryReport.totalEstimatedMemoryBytes),
                              builtHardwareRt ? "hardware" : "software");
                    }

                    if (completion->resources) {
                        completion->settings = loadedSettings;
                        completion->success = true;
                    }
                }
            }
        }

        dispatch_async(dispatch_get_main_queue(), ^{
            if (auto self = weakSelf.lock()) {
                self->handleSceneLoadWorkerCompletion(generation, completion);
            }
        });
    });
}

void MetalRenderer::Impl::handleSceneLoadWorkerCompletion(
    uint64_t generation,
    std::shared_ptr<SceneLoadCompleted> completion) {
    if (m_shutdown || !completion || !m_sceneLoadGeneration) {
        return;
    }
    m_sceneLoadWorkerActive = false;
    if (m_sceneLoadGeneration->load(std::memory_order_acquire) != generation) {
        if (m_sceneLoadStage == SceneLoadStage::LoadScene) {
            m_sceneLoadStage = SceneLoadStage::Idle;
        }
        startQueuedSceneLoadIfPossible();
        return;
    }

    SceneLoadCompleted pending{};
    pending.success = completion->success;
    pending.settings = completion->settings;
    pending.activeSceneId = completion->activeSceneId;
    pending.errorMessage = completion->errorMessage;
    pending.exclusive = completion->exclusive;
    pending.resources = std::move(completion->resources);
    m_pendingSceneLoadResult = std::move(pending);
    m_sceneLoadStage = completion->success ? SceneLoadStage::Activate : SceneLoadStage::Failed;

    if (completion->success) {
        NSLog(@"[SceneLoad] worker complete id='%s'", completion->activeSceneId.c_str());
    } else {
        NSLog(@"[SceneLoad] worker failed id='%s': %s",
              completion->activeSceneId.c_str(),
              completion->errorMessage.empty() ? "unknown error" : completion->errorMessage.c_str());
    }
    startQueuedSceneLoadIfPossible();
}

void MetalRenderer::Impl::serviceSceneLoadState(bool allowGpuWork) {
    startQueuedSceneLoadIfPossible();
    if (m_sceneLoadStage == SceneLoadStage::Idle) {
        return;
    }

    if (m_sceneLoadStage == SceneLoadStage::LoadScene) {
        return;
    }

    if (!m_pendingSceneLoadResult.has_value()) {
        if (m_sceneLoadStage == SceneLoadStage::Failed) {
            m_sceneLoadStage = SceneLoadStage::Idle;
            m_queuedSceneLoad.reset();
        }
        return;
    }

    SceneLoadCompleted& pending = *m_pendingSceneLoadResult;
    switch (m_sceneLoadStage) {
        case SceneLoadStage::Activate: {
            if (!allowGpuWork) {
                break;
            }

            if (!m_sceneSwapDrainCommandBuffer) {
                NSLog(@"[SceneLoad] stage=Activate id='%s'", pending.activeSceneId.c_str());
                if (id<MTLCommandQueue> queue = m_context.commandQueue()) {
                    id<MTLCommandBuffer> drain = [queue commandBuffer];
                    if (!drain) {
                        pending.errorMessage = "Failed to allocate scene swap drain command buffer";
                        m_sceneLoadStage = SceneLoadStage::Failed;
                        break;
                    }
                    drain.label = @"Scene Swap Drain";
                    [drain commit];
                    m_sceneSwapDrainCommandBuffer = drain;
                    m_sceneSwapDrainStartTime = CFAbsoluteTimeGetCurrent();
                    m_sceneSwapDrainWarningLogged = false;
                }
            }

            if (m_sceneSwapDrainCommandBuffer) {
                const MTLCommandBufferStatus status = m_sceneSwapDrainCommandBuffer.status;
                if (status == MTLCommandBufferStatusError) {
                    pending.errorMessage = "Scene swap drain command buffer failed";
                    m_sceneLoadStage = SceneLoadStage::Failed;
                    break;
                }
                if (status != MTLCommandBufferStatusCompleted) {
                    if (!m_sceneSwapDrainWarningLogged && m_sceneSwapDrainStartTime > 0.0) {
                        const double elapsedMs =
                            (CFAbsoluteTimeGetCurrent() - m_sceneSwapDrainStartTime) * 1000.0;
                        if (elapsedMs >= 2000.0) {
                            m_sceneSwapDrainWarningLogged = true;
                            NSLog(@"[SceneLoad] waiting for scene swap drain %.0fms", elapsedMs);
                        }
                    }
                    break;
                }
            }

            if (SceneLoadDiagnosticsEnabled()) {
                const auto& outgoing = activeSceneResources();
                const MeshAggregateStats outgoingMeshStats = CollectMeshAggregateStats(outgoing);
                const GpuBufferSummary outgoingBuffers = CollectGpuBufferSummary(outgoing);
                const auto outgoingReport = outgoing.sceneMemoryReport();
                const uint64_t allocatedBeforeSwap = CurrentAllocatedSizeBytes(m_context.device());
                NSLog(@"[SceneDiag] activateSwap outgoing id='%s' materials=%u meshes=%u vertices=%llu indices=%llu triangles=%u "
                      "coreBuffers=%u meshBuffers=%u coreMiB=%.2f meshMiB=%.2f geomMiB=%.2f texMiB=%.2f asMiB=%.2f "
                      "metalAllocatedMiB=%.2f",
                      m_activeSceneId.c_str(),
                      outgoing.materialCount(),
                      outgoingMeshStats.meshCount,
                      static_cast<unsigned long long>(outgoingMeshStats.vertexCount),
                      static_cast<unsigned long long>(outgoingMeshStats.indexCount),
                      outgoing.triangleCount(),
                      outgoingBuffers.coreCount,
                      outgoingBuffers.meshCount,
                      BytesToMiB(outgoingBuffers.coreBytes),
                      BytesToMiB(outgoingBuffers.meshBytes),
                      BytesToMiB(outgoingReport.geometryMemoryBytes),
                      BytesToMiB(outgoingReport.textureAllocatedMemoryBytes),
                      BytesToMiB(outgoingReport.blasMemoryBytes + outgoingReport.tlasMemoryBytes +
                                 outgoingReport.scratchMemoryBytes),
                      BytesToMiB(allocatedBeforeSwap));
            }

            m_externalSceneResources = nullptr;
            if (m_loadedSceneResources) {
                m_loadedSceneResources->clear();
                m_loadedSceneResources.reset();
            } else {
                m_sceneResources.clear();
            }
            m_loadedSceneResources = std::move(pending.resources);
            m_settings = pending.settings;
            m_settings.renderScale = ClampRenderScale(m_settings.renderScale);
            m_cameraCollisionSource = nullptr;
            resetFirstPersonMotion();
            activeSceneResources().setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);
            applyBackgroundSettings();
            syncCameraSmoothingToTarget();
            m_activeSceneId = pending.activeSceneId;
            resetAccumulation();
            m_queuedSceneLoad.reset();
            m_pendingSceneLoadResult.reset();
            m_sceneSwapDrainCommandBuffer = nil;
            m_sceneSwapDrainStartTime = 0.0;
            m_sceneSwapDrainWarningLogged = false;
            m_sceneLoadExclusiveActive = false;
            m_sceneLoadStage = SceneLoadStage::Idle;
            NSLog(@"[SceneLoad] activated scene '%s'", m_activeSceneId.c_str());
            if (SceneLoadDiagnosticsEnabled()) {
                const auto& active = activeSceneResources();
                const MeshAggregateStats activeMeshStats = CollectMeshAggregateStats(active);
                const GpuBufferSummary activeBuffers = CollectGpuBufferSummary(active);
                const auto activeReport = active.sceneMemoryReport();
                const auto accelStats = active.accelerationBuildStats();
                const uint64_t allocatedAfterSwap = CurrentAllocatedSizeBytes(m_context.device());
                NSLog(@"[SceneDiag] activateSwap complete id='%s' materials=%u meshes=%u vertices=%llu indices=%llu triangles=%u "
                      "coreBuffers=%u meshBuffers=%u coreMiB=%.2f meshMiB=%.2f geomMiB=%.2f texMiB=%.2f "
                      "blasCount=%u tlasInstances=%u asMiB=%.2f metalAllocatedMiB=%.2f",
                      m_activeSceneId.c_str(),
                      active.materialCount(),
                      activeMeshStats.meshCount,
                      static_cast<unsigned long long>(activeMeshStats.vertexCount),
                      static_cast<unsigned long long>(activeMeshStats.indexCount),
                      active.triangleCount(),
                      activeBuffers.coreCount,
                      activeBuffers.meshCount,
                      BytesToMiB(activeBuffers.coreBytes),
                      BytesToMiB(activeBuffers.meshBytes),
                      BytesToMiB(activeReport.geometryMemoryBytes),
                      BytesToMiB(activeReport.textureAllocatedMemoryBytes),
                      accelStats.blasCount,
                      accelStats.instanceCount,
                      BytesToMiB(activeReport.blasMemoryBytes + activeReport.tlasMemoryBytes +
                                 activeReport.scratchMemoryBytes),
                      BytesToMiB(allocatedAfterSwap));
            }
            break;
        }
        case SceneLoadStage::Failed:
            NSLog(@"[SceneLoad] failed id='%s': %s",
                  pending.activeSceneId.c_str(),
                  pending.errorMessage.empty() ? "unknown error" : pending.errorMessage.c_str());
            m_queuedSceneLoad.reset();
            m_pendingSceneLoadResult.reset();
            m_sceneSwapDrainCommandBuffer = nil;
            m_sceneSwapDrainStartTime = 0.0;
            m_sceneSwapDrainWarningLogged = false;
            m_sceneLoadExclusiveActive = false;
            m_sceneLoadStage = SceneLoadStage::Idle;
            break;
        case SceneLoadStage::LoadScene:
        case SceneLoadStage::Idle:
        default:
            break;
    }
}

void MetalRenderer::Impl::applyBackgroundSettings() {
    using BackgroundMode = RenderSettings::BackgroundMode;
    auto& sceneResources = activeSceneResources();
    switch (m_settings.backgroundMode) {
        case BackgroundMode::Gradient:
            sceneResources.clearEnvironmentMap();
            m_settings.environmentMapPath.clear();
            m_settings.environmentMapDirty = false;
            break;
        case BackgroundMode::Solid:
            sceneResources.clearEnvironmentMap();
            m_settings.environmentMapPath.clear();
            m_settings.environmentMapDirty = false;
            break;
        case BackgroundMode::Environment: {
            bool envChanged = false;
            if (!syncEnvironmentMap(&envChanged)) {
                if (!m_settings.environmentMapPath.empty()) {
                    NSLog(@"Failed to load environment map: %s", m_settings.environmentMapPath.c_str());
                }
                sceneResources.clearEnvironmentMap();
                m_settings.backgroundMode = BackgroundMode::Gradient;
                m_settings.environmentMapPath.clear();
            } else if (envChanged) {
                resetAccumulation();
            }
            break;
        }
    }
}

bool MetalRenderer::Impl::syncEnvironmentMap(bool* outChanged) {
    using BackgroundMode = RenderSettings::BackgroundMode;
    bool changed = false;
    auto& sceneResources = activeSceneResources();
    const bool wantsEnvironment =
        (m_settings.backgroundMode == BackgroundMode::Environment) &&
        !m_settings.environmentMapPath.empty();

    if (!wantsEnvironment) {
        const bool hadEnvironment = !sceneResources.environmentPath().empty();
        if (hadEnvironment) {
            sceneResources.clearEnvironmentMap();
            changed = true;
        }
        if (m_settings.backgroundMode != BackgroundMode::Environment) {
            m_settings.environmentMapPath.clear();
        }
        m_settings.environmentMapDirty = false;
        if (outChanged) {
            *outChanged = changed;
        }
        return true;
    }

    const std::string currentPath = sceneResources.environmentPath();
    if (!m_settings.environmentMapDirty &&
        !currentPath.empty() &&
        currentPath == m_settings.environmentMapPath) {
        if (outChanged) {
            *outChanged = false;
        }
        return true;
    }

    m_settings.environmentMapDirty = false;
    EnvGpuHandles handles;
    if (!sceneResources.reloadEnvironmentIfNeeded(m_settings.environmentMapPath, &handles)) {
        if (!currentPath.empty()) {
            m_settings.environmentMapPath = currentPath;
        } else {
            m_settings.environmentMapPath.clear();
        }
        return false;
    }

    changed = (currentPath != sceneResources.environmentPath());
    if (outChanged) {
        *outChanged = changed;
    }
    return true;
}

void MetalRenderer::Impl::syncRadiometricBaseline() {
    m_radiometricBaseline = m_settings;
    m_hasRadiometricBaseline = true;
}

void MetalRenderer::Impl::evaluateAccumulationState() {
    if (!m_hasRadiometricBaseline) {
        syncRadiometricBaseline();
        return;
    }
    if (m_accumDirty) {
        return;
    }
    const auto change = PathTracer::DetectRadiometricChange(m_radiometricBaseline, m_settings);
    if (change.changed) {
        m_accumDirty = true;
        m_accumDirtyReason = change.reason ? change.reason : "RADIOMETRIC";
    }
}

void MetalRenderer::Impl::serviceAccumulationReset() {
    if (!m_hasRadiometricBaseline) {
        syncRadiometricBaseline();
    }
    if (!m_accumDirty && !m_deferredAccumulationReset) {
        return;
    }
    if (pathTraceFrameInFlight()) {
        m_deferredAccumulationReset = true;
        return;
    }
    const char* reason = m_accumDirtyReason.empty() ? "RADIOMETRIC" : m_accumDirtyReason.c_str();
    NSLog(@"[I] ResetAccum(%s)", reason);
    resetAccumulation();
    m_accumDirty = false;
    m_accumDirtyReason.clear();
    syncRadiometricBaseline();
}

bool MetalRenderer::Impl::pathTraceFrameInFlight() const {
    return !m_options.headless && m_pathTraceFrameInFlight.load(std::memory_order_acquire);
}

void MetalRenderer::Impl::setupDefaultScene() {
    m_sceneResources.setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);

    std::string sceneError;
    if (!m_sceneManager.refresh(&sceneError) && !sceneError.empty()) {
        NSLog(@"SceneManager refresh failed: %s", sceneError.c_str());
    }
    const auto& available = m_sceneManager.scenes();

    std::string targetIdentifier;
    if (!m_options.initialScene.empty()) {
        targetIdentifier = m_options.initialScene;
    } else if (!available.empty()) {
        targetIdentifier = available.front().identifier;
    }

    if (m_options.initialScene.empty() && !available.empty()) {
        RenderSettings updatedSettings = m_settings;
        const RenderSettings cameraDefaults{};
        updatedSettings.cameraControlMode = cameraDefaults.cameraControlMode;
        updatedSettings.cameraTarget = cameraDefaults.cameraTarget;
        updatedSettings.cameraDistance = cameraDefaults.cameraDistance;
        updatedSettings.cameraYaw = cameraDefaults.cameraYaw;
        updatedSettings.cameraPitch = cameraDefaults.cameraPitch;
        updatedSettings.firstPersonCameraPosition = cameraDefaults.firstPersonCameraPosition;
        updatedSettings.firstPersonMoveSpeed = cameraDefaults.firstPersonMoveSpeed;
        updatedSettings.cameraVerticalFov = cameraDefaults.cameraVerticalFov;
        updatedSettings.cameraDefocusAngle = cameraDefaults.cameraDefocusAngle;
        updatedSettings.cameraFocusDistance = cameraDefaults.cameraFocusDistance;

        sceneError.clear();
        if (m_sceneManager.loadScene(targetIdentifier, m_sceneResources, updatedSettings, &sceneError)) {
            if (m_options.enableSoftwareRayTracing) {
                updatedSettings.enableSoftwareRayTracing = true;
            }
            m_settings = updatedSettings;
            m_settings.renderScale = ClampRenderScale(m_settings.renderScale);
            m_sceneResources.setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);
            applyBackgroundSettings();
            m_sceneResources.rebuildAccelerationStructures();
            m_activeSceneId = targetIdentifier;
            syncCameraSmoothingToTarget();
            return;
        }
        if (!sceneError.empty()) {
            NSLog(@"Failed to load default scene '%s': %s", targetIdentifier.c_str(), sceneError.c_str());
        }
    }

    if (!targetIdentifier.empty()) {
        namespace fs = std::filesystem;
        bool loadFromPath = false;
        fs::path candidate(targetIdentifier);
        if (candidate.is_absolute() || candidate.has_parent_path() || candidate.extension() == ".scene") {
            loadFromPath = true;
        } else {
            std::error_code probeEc;
            if (fs::exists(candidate, probeEc)) {
                loadFromPath = true;
            }
        }

        if (loadFromPath) {
            if (queueSceneLoadFromResolvedPath(targetIdentifier, targetIdentifier, m_settings)) {
                return;
            }
        } else {
            sceneError.clear();
            const auto& scenes = m_sceneManager.scenes();
            for (const auto& info : scenes) {
                if (info.identifier == targetIdentifier) {
                    RenderSettings updatedSettings = m_settings;
                    const RenderSettings cameraDefaults{};
                    updatedSettings.cameraControlMode = cameraDefaults.cameraControlMode;
                    updatedSettings.cameraTarget = cameraDefaults.cameraTarget;
                    updatedSettings.cameraDistance = cameraDefaults.cameraDistance;
                    updatedSettings.cameraYaw = cameraDefaults.cameraYaw;
                    updatedSettings.cameraPitch = cameraDefaults.cameraPitch;
                    updatedSettings.firstPersonCameraPosition = cameraDefaults.firstPersonCameraPosition;
                    updatedSettings.firstPersonMoveSpeed = cameraDefaults.firstPersonMoveSpeed;
                    updatedSettings.cameraVerticalFov = cameraDefaults.cameraVerticalFov;
                    updatedSettings.cameraDefocusAngle = cameraDefaults.cameraDefocusAngle;
                    updatedSettings.cameraFocusDistance = cameraDefaults.cameraFocusDistance;
                    if (queueSceneLoadFromResolvedPath(info.filePath, info.identifier, updatedSettings)) {
                        return;
                    }
                    break;
                }
            }
        }
    }

    buildProceduralScene();
}

void MetalRenderer::Impl::buildProceduralScene() {
    m_settings.backgroundMode = RenderSettings::BackgroundMode::Gradient;
    m_settings.backgroundColor = simd_make_float3(0.0f, 0.0f, 0.0f);
    m_settings.environmentMapPath.clear();
    m_settings.environmentRotation = 0.0f;
    m_settings.environmentIntensity = 1.0f;
    m_sceneResources.clearEnvironmentMap();

    m_sceneResources.clear();
    m_sceneResources.setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);

    struct ReservedSphere {
        simd::float3 center;
        float radius;
    };

    std::vector<ReservedSphere> placedSpheres;
    placedSpheres.reserve(kMaxSpheres);

    auto addSceneSphere = [&](const simd::float3& center, float radius, uint32_t materialIndex) {
        addSphere(center, radius, materialIndex);
        placedSpheres.push_back(ReservedSphere{center, radius});
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    auto randomFloat = [&]() { return dist(rng); };
    auto randomFloatRange = [&](float min, float max) {
        return min + (max - min) * randomFloat();
    };
    auto randomColor = [&]() -> simd::float3 {
        return simd_make_float3(randomFloat(), randomFloat(), randomFloat());
    };
    auto randomColorRange = [&](float min, float max) -> simd::float3 {
        return simd_make_float3(randomFloatRange(min, max),
                                randomFloatRange(min, max),
                                randomFloatRange(min, max));
    };

    uint32_t groundMaterial =
        addMaterial(simd_make_float3(0.5f, 0.5f, 0.5f), 0.0f, MaterialType::Lambertian, 1.0f);
    addSceneSphere(simd_make_float3(0.0f, -1000.0f, 0.0f), 1000.0f, groundMaterial);

    const std::array<ReservedSphere, 3> reservedLargeSpheres = {{
        {simd_make_float3(0.0f, 1.0f, 0.0f), 1.0f},
        {simd_make_float3(-4.0f, 1.0f, 0.0f), 1.0f},
        {simd_make_float3(4.0f, 1.0f, 0.0f), 1.0f},
    }};

    auto intersectsExisting = [&](const simd::float3& center, float radius) {
        constexpr float kSeparationEpsilon = 1e-3f;
        for (const auto& placed : placedSpheres) {
            if (placed.radius > 900.0f) {
                continue;  // skip ground
            }
            if (simd::length(center - placed.center) <
                radius + placed.radius + kSeparationEpsilon) {
                return true;
            }
        }
        for (const auto& reserved : reservedLargeSpheres) {
            if (simd::length(center - reserved.center) <
                radius + reserved.radius + kSeparationEpsilon) {
                return true;
            }
        }
        return false;
    };

    const uint32_t sharedGlassMaterial =
        addMaterial(simd_make_float3(1.0f, 1.0f, 1.0f), 0.0f, MaterialType::Dielectric, 1.50f);

    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            if (m_sceneResources.sphereCount() >= kMaxSpheres - 3 ||
                m_sceneResources.materialCount() >= kMaxMaterials - 3) {
                break;
            }

            simd::float3 center = simd_make_float3(static_cast<float>(a) + 0.9f * randomFloat(),
                                                   0.2f,
                                                   static_cast<float>(b) + 0.9f * randomFloat());

            constexpr float smallRadius = 0.2f;
            if (intersectsExisting(center, smallRadius)) {
                continue;
            }

            float normalizedZ = std::clamp((center.z + 11.0f) / 22.0f, 0.0f, 1.0f);
            constexpr float kNearOccupancy = 0.9f;
            constexpr float kFarOccupancy = 0.6f;
            float occupancyProbability =
                kNearOccupancy - (kNearOccupancy - kFarOccupancy) * normalizedZ;
            if (randomFloat() > occupancyProbability) {
                continue;
            }

            float chooseMaterial = randomFloat();
            uint32_t materialIndex = 0;

            if (chooseMaterial < 0.8f) {
                simd::float3 albedo = randomColor() * randomColor();
                materialIndex = addMaterial(albedo, 0.0f, MaterialType::Lambertian, 1.0f);
            } else if (chooseMaterial < 0.95f) {
                simd::float3 albedo = randomColorRange(0.5f, 1.0f);
                float roughness = randomFloatRange(0.0f, 0.5f);
                materialIndex = addMaterial(albedo, roughness, MaterialType::Metal, 1.0f);
            } else {
                materialIndex = sharedGlassMaterial;
            }

            addSceneSphere(center, smallRadius, materialIndex);
        }
    }

    uint32_t bigGlass = sharedGlassMaterial;
    uint32_t bigLambertian =
        addMaterial(simd_make_float3(0.4f, 0.2f, 0.1f), 0.0f, MaterialType::Lambertian, 1.0f);
    uint32_t bigMetal =
        addMaterial(simd_make_float3(0.7f, 0.6f, 0.5f), 0.0f, MaterialType::Metal, 1.0f);

    addSceneSphere(simd_make_float3(0.0f, 1.0f, 0.0f), 1.0f, bigGlass);
    addSceneSphere(simd_make_float3(-4.0f, 1.0f, 0.0f), 1.0f, bigLambertian);
    addSceneSphere(simd_make_float3(4.0f, 1.0f, 0.0f), 1.0f, bigMetal);

    m_sceneResources.setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);
    m_sceneResources.rebuildAccelerationStructures();
    m_activeSceneId.clear();
    m_cameraCollisionSource = nullptr;
    resetFirstPersonMotion();
    syncCameraSmoothingToTarget();
}

void MetalRenderer::Impl::resetAccumulation() {
    if (pathTraceFrameInFlight()) {
        m_accumDirty = true;
        m_deferredAccumulationReset = true;
        if (m_accumDirtyReason.empty()) {
            m_accumDirtyReason = "GPU_FRAME_IN_FLIGHT";
        }
        return;
    }

    m_performanceStats.samplesPerMinute = 0.0;
    m_performanceStats.avgNodesVisited = 0.0;
    m_performanceStats.avgLeafPrimTests = 0.0;
    m_performanceStats.shadowRayEarlyExitPct = 0.0;
    m_performanceStats.bothChildrenVisitedPct = 0.0;
    m_performanceStats.gpuTimeMs = 0.0;
    m_performanceStats.cpuEncodeMs = 0.0;
    m_performanceStats.drawableWaitMs = 0.0;
    m_performanceStats.gpuBucketTimingAvailable = false;
    m_performanceStats.gpuBucketIntersectMs = 0.0;
    m_performanceStats.gpuBucketShadeMs = 0.0;
    m_performanceStats.gpuBucketShadowQueueMs = 0.0;
    m_performanceStats.gpuBucketFusedIntegrateMs = 0.0;
    m_performanceStats.gpuBucketPresentationMs = 0.0;
    m_performanceStats.currentAllocatedBytes = 0;
    m_performanceStats.recommendedMaxWorkingSetBytes = 0;
    m_performanceStats.sampleCount = 0;
    m_performanceStats.sampleCount = 0;
    m_performanceStats.totalComputeDispatchCount = 0;
    m_performanceStats.integrationDispatchCount = 0;
    m_performanceStats.presentationDispatchCount = 0;
    m_performanceStats.wavefrontResetDispatchCount = 0;
    m_performanceStats.wavefrontGenerateDispatchCount = 0;
    m_performanceStats.wavefrontBuildDispatchArgsDispatchCount = 0;
    m_performanceStats.wavefrontCompactQueueDispatchCount = 0;
    m_performanceStats.wavefrontPrepareDispatchCount = 0;
    m_performanceStats.wavefrontIntersectDispatchCount = 0;
    m_performanceStats.wavefrontShadeDispatchCount = 0;
    m_performanceStats.wavefrontShadowDispatchCount = 0;
    m_performanceStats.wavefrontSwapDispatchCount = 0;
    m_performanceStats.wavefrontAccumulateDispatchCount = 0;
    m_performanceStats.primaryRayCount = 0;
    m_performanceStats.shadowRayCount = 0;
    m_performanceStats.pathSegmentCount = 0;
    m_performanceStats.tracedRayCount = 0;
    m_performanceStats.restirGiCandidateCount = 0;
    m_performanceStats.restirGiReservoirUpdateCount = 0;
    m_performanceStats.restirGiAcceptCount = 0;
    m_performanceStats.restirGiRejectCount = 0;
    m_performanceStats.restirGiFallbackCount = 0;
    m_performanceStats.pathGuidingCandidateCount = 0;
    m_performanceStats.pathGuidingUsedCount = 0;
    m_performanceStats.pathGuidingFallbackCount = 0;
    m_performanceStats.pathGuidingInvalidCount = 0;
    m_performanceStats.restirPtCandidateCount = 0;
    m_performanceStats.restirPtReservoirUpdateCount = 0;
    m_performanceStats.restirPtRejectedInvalidCount = 0;
    m_performanceStats.restirPtRejectedUnsupportedCount = 0;
    m_performanceStats.restirPtDebugRecordCount = 0;
    m_performanceStats.restirPtReuseCandidateCount = 0;
    m_performanceStats.restirPtReuseAppliedCount = 0;
    m_performanceStats.restirPtReuseRejectedInvalidCount = 0;
    m_performanceStats.restirPtReuseFallbackCount = 0;
    m_performanceStats.radianceCacheQueryCount = 0;
    m_performanceStats.radianceCacheHitCount = 0;
    m_performanceStats.radianceCacheMissCount = 0;
    m_performanceStats.radianceCacheTrainCount = 0;
    m_performanceStats.radianceCacheFallbackCount = 0;
    m_performanceStats.radianceCacheInvalidCount = 0;
    m_performanceStats.causticEyeVertexCount = 0;
    m_performanceStats.causticLightVertexCount = 0;
    m_performanceStats.causticConnectionCandidateCount = 0;
    m_performanceStats.causticDeltaChainCount = 0;
    m_performanceStats.causticDebugClassificationCount = 0;
    m_performanceStats.volumeTransmittanceQueryCount = 0;
    m_performanceStats.volumeScatterEventCount = 0;
    m_performanceStats.volumeNeeRayCount = 0;
    m_performanceStats.volumePhaseSampleCount = 0;
    m_performanceStats.volumeMediumBoundaryEventCount = 0;
    m_performanceStats.volumeFallbackCount = 0;
    m_performanceStats.spectralWavelengthSampleCount = 0;
    m_performanceStats.spectralMaterialLookupCount = 0;
    m_performanceStats.spectralDispersionEventCount = 0;
    m_performanceStats.spectralFallbackCount = 0;
    m_performanceStats.wavefrontPrimaryGeneratedCount = 0;
    m_performanceStats.wavefrontIntersectProcessedCount = 0;
    m_performanceStats.wavefrontShadeProcessedCount = 0;
    m_performanceStats.wavefrontShadowProcessedCount = 0;
    m_performanceStats.wavefrontNextRayEnqueueCount = 0;
    m_performanceStats.wavefrontShadowRayEnqueueCount = 0;
    for (uint32_t i = 0u; i < PerformanceStats::kWavefrontQueueMetricCount; ++i) {
        m_performanceStats.wavefrontQueueActiveCounts[i] = 0u;
        m_performanceStats.wavefrontQueuePeakActiveCounts[i] = 0u;
        m_performanceStats.wavefrontQueueCapacities[i] = 0u;
        m_performanceStats.wavefrontQueueOverflowCounts[i] = 0u;
        m_performanceStats.wavefrontQueueOccupancyRatios[i] = 0.0;
        m_performanceStats.wavefrontQueuePeakOccupancyRatios[i] = 0.0;
    }
    m_performanceStats.wavefrontQueueCompactionCostMs = 0.0;

    m_previousSampleCount = 0;

    CGSize targetSize = targetRenderSize();
    m_accumulation.ensureTextures(targetSize);
    m_accumulation.reset(m_context.commandQueue(), m_pipelines.clear(), nullptr);
    const bool preserveSvgfHistory =
        m_settings.svgfDenoiseEnabled &&
        m_accumDirtyReason.rfind("CAMERA_", 0) == 0;
    m_renderLoop.resetHistory(HistoryResetReason::FeatureToggleChanged, preserveSvgfHistory);

    m_deferredAccumulationReset = false;
    m_accumDirty = false;
    m_accumDirtyReason.clear();
    syncRadiometricBaseline();
}

void MetalRenderer::Impl::setTonemapMode(uint32_t mode) {
    uint32_t clamped = std::max<uint32_t>(1u, std::min<uint32_t>(mode, 4u));
    m_settings.tonemapMode = clamped;
}

void MetalRenderer::Impl::setPresentationEnabled(bool enabled) {
    if (m_options.headless) {
        return;
    }
    if (m_presentationSettings.enabled == enabled) {
        return;
    }
    m_presentationSettings.enabled = enabled;
    applyPresentationSettings(true);
}

void MetalRenderer::Impl::togglePresentationUIPanels() {
    if (m_options.headless) {
        return;
    }
    if (!m_presentationSettings.enabled) {
        return;
    }
    m_presentationSettings.hideUIPanels = !m_presentationSettings.hideUIPanels;
    applyPresentationSettings(false);
}

bool MetalRenderer::Impl::loadScene(const std::string& identifier) {
    std::string refreshError;
    if (!m_sceneManager.refresh(&refreshError) && !refreshError.empty()) {
        NSLog(@"SceneManager refresh failed: %s", refreshError.c_str());
    }

    RenderSettings updatedSettings = m_settings;
    {
        // Reset camera to scene defaults on every switch so each scene starts from a known view.
        const RenderSettings cameraDefaults{};
        updatedSettings.cameraControlMode = cameraDefaults.cameraControlMode;
        updatedSettings.cameraTarget = cameraDefaults.cameraTarget;
        updatedSettings.cameraDistance = cameraDefaults.cameraDistance;
        updatedSettings.cameraYaw = cameraDefaults.cameraYaw;
        updatedSettings.cameraPitch = cameraDefaults.cameraPitch;
        updatedSettings.firstPersonCameraPosition = cameraDefaults.firstPersonCameraPosition;
        updatedSettings.firstPersonMoveSpeed = cameraDefaults.firstPersonMoveSpeed;
        updatedSettings.cameraVerticalFov = cameraDefaults.cameraVerticalFov;
        updatedSettings.cameraDefocusAngle = cameraDefaults.cameraDefocusAngle;
        updatedSettings.cameraFocusDistance = cameraDefaults.cameraFocusDistance;
    }
    if (m_options.headless) {
        m_externalSceneResources = nullptr;
        std::string errorMessage;
        if (!m_sceneManager.loadScene(identifier, m_sceneResources, updatedSettings, &errorMessage)) {
            if (!errorMessage.empty()) {
                NSLog(@"Failed to load scene '%s': %s", identifier.c_str(), errorMessage.c_str());
            }
            return false;
        }

        if (m_options.enableSoftwareRayTracing) {
            updatedSettings.enableSoftwareRayTracing = true;
        }
        m_settings = updatedSettings;
        m_settings.renderScale = ClampRenderScale(m_settings.renderScale);
        m_sceneResources.setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);
        applyBackgroundSettings();
        syncCameraSmoothingToTarget();
        m_activeSceneId = identifier;
        resetAccumulation();
        return true;
    }

    const auto& scenes = m_sceneManager.scenes();
    for (const auto& info : scenes) {
        if (info.identifier == identifier) {
            return queueSceneLoadFromResolvedPath(info.filePath, identifier, updatedSettings);
        }
    }

    NSLog(@"Failed to load scene '%s': identifier not found", identifier.c_str());
    return false;
}

bool MetalRenderer::Impl::loadSceneFromPath(const std::string& path) {
    if (path.empty()) {
        NSLog(@"Invalid scene path (empty)");
        return false;
    }

    RenderSettings updatedSettings = m_settings;
    if (m_options.headless) {
        m_externalSceneResources = nullptr;
        std::string errorMessage;
        if (!m_sceneManager.loadSceneFromPath(path, m_sceneResources, updatedSettings, &errorMessage)) {
            if (!errorMessage.empty()) {
                NSLog(@"Failed to load scene from '%s': %s", path.c_str(), errorMessage.c_str());
            }
            return false;
        }

        m_settings = updatedSettings;
        m_settings.renderScale = ClampRenderScale(m_settings.renderScale);
        m_sceneResources.setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);
        applyBackgroundSettings();
        m_sceneResources.rebuildAccelerationStructures();
        m_activeSceneId = path;
        syncCameraSmoothingToTarget();
        resetAccumulation();
        return true;
    }

    return queueSceneLoadFromResolvedPath(path, path, updatedSettings);
}

bool MetalRenderer::Impl::useSceneResources(PathTracer::SceneResources& resources,
                                            const std::string& identifier) {
    cancelPendingSceneLoad();
    m_loadedSceneResources.reset();
    m_externalSceneResources = &resources;
    m_cameraCollisionSource = nullptr;
    resetFirstPersonMotion();
    activeSceneResources().setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);
    applyBackgroundSettings();
    m_activeSceneId = identifier;
    syncCameraSmoothingToTarget();
    resetAccumulation();
    return true;
}

std::vector<std::string> MetalRenderer::Impl::sceneIdentifiers() const {
    const_cast<SceneManager&>(m_sceneManager).refresh(nullptr);
    std::vector<std::string> identifiers;
    const auto& scenes = m_sceneManager.scenes();
    identifiers.reserve(scenes.size());
    for (const auto& info : scenes) {
        identifiers.push_back(info.identifier);
    }
    return identifiers;
}

void MetalRenderer::Impl::applySettings(const RenderSettings& settings, bool resetAccumulationFlag) {
    RenderSettings sanitized = settings;
    sanitized.renderScale = ClampRenderScale(sanitized.renderScale);
    m_settings = sanitized;
    activeSceneResources().setSoftwareRayTracingOverride(m_settings.enableSoftwareRayTracing);
    applyBackgroundSettings();
    syncCameraSmoothingToTarget();
    if (resetAccumulationFlag) {
        resetAccumulation();
    } else {
        m_accumulation.ensureTextures(targetRenderSize());
    }
}

bool MetalRenderer::Impl::captureTextureImage(MTLTextureHandle sourceTexture,
                                              std::vector<float>& outLinearRGB,
                                              uint32_t& width,
                                              uint32_t& height,
                                              std::vector<float>* outAlpha,
                                              bool decodeNormal) {
    id<MTLDevice> device = m_context.device();
    id<MTLCommandQueue> queue = m_context.commandQueue();
    if (!sourceTexture || !device || !queue) {
        return false;
    }

    width = static_cast<uint32_t>(sourceTexture.width);
    height = static_cast<uint32_t>(sourceTexture.height);

    MTLTextureDescriptor* descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                           width:sourceTexture.width
                                                          height:sourceTexture.height
                                                       mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderWrite;

    id<MTLTexture> readable = [device newTextureWithDescriptor:descriptor];
    if (!readable) {
        return false;
    }

    id<MTLCommandBuffer> blitCmd = [queue commandBuffer];
    blitCmd.label = @"Capture Image Blit";
    id<MTLBlitCommandEncoder> blitEncoder = [blitCmd blitCommandEncoder];
    [blitEncoder copyFromTexture:sourceTexture toTexture:readable];
    [blitEncoder endEncoding];
    [blitCmd commit];
    [blitCmd waitUntilCompleted];

    std::vector<float> rgba(static_cast<size_t>(width) * height * 4u);
    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    [readable getBytes:rgba.data()
           bytesPerRow:static_cast<NSUInteger>(width) * 4u * sizeof(float)
            fromRegion:region
           mipmapLevel:0];

    outLinearRGB.resize(static_cast<size_t>(width) * height * 3u);
    if (outAlpha) {
        outAlpha->resize(static_cast<size_t>(width) * height);
    }

    for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
        float r = rgba[4 * i + 0];
        float g = rgba[4 * i + 1];
        float b = rgba[4 * i + 2];
        if (decodeNormal) {
            r = r * 2.0f - 1.0f;
            g = g * 2.0f - 1.0f;
            b = b * 2.0f - 1.0f;
        }
        outLinearRGB[3 * i + 0] = r;
        outLinearRGB[3 * i + 1] = g;
        outLinearRGB[3 * i + 2] = b;
        if (outAlpha) {
            (*outAlpha)[i] = rgba[4 * i + 3];
        }
    }

    return true;
}

bool MetalRenderer::Impl::captureAverageImage(std::vector<float>& outLinearRGB,
                                              uint32_t& width,
                                              uint32_t& height,
                                              std::vector<float>* outSampleCounts) {
    id<MTLTexture> sourceTexture = m_accumulation.present();
    if ((m_settings.denoiseEnabled || m_settings.svgfDenoiseEnabled) &&
        m_renderLoop.hasDenoisedOutput()) {
        if (id<MTLTexture> denoised = m_accumulation.denoisedBuffer()) {
            sourceTexture = denoised;
        }
    }
    return captureTextureImage(sourceTexture, outLinearRGB, width, height, outSampleCounts, false);
}

bool MetalRenderer::Impl::captureRawImage(std::vector<float>& outLinearRGB,
                                          uint32_t& width,
                                          uint32_t& height,
                                          std::vector<float>* outSampleCounts) {
    return captureTextureImage(m_accumulation.present(), outLinearRGB, width, height, outSampleCounts, false);
}

bool MetalRenderer::Impl::captureAuxiliaryImages(std::vector<float>* outAlbedoRGB,
                                                 std::vector<float>* outNormalRGB,
                                                 uint32_t& width,
                                                 uint32_t& height) {
    if (!outAlbedoRGB && !outNormalRGB) {
        return false;
    }
    bool captured = false;
    uint32_t localWidth = 0;
    uint32_t localHeight = 0;
    if (outAlbedoRGB) {
        std::vector<float> dummyAlpha;
        if (!captureTextureImage(m_accumulation.albedoBuffer(),
                                 *outAlbedoRGB,
                                 localWidth,
                                 localHeight,
                                 &dummyAlpha,
                                 false)) {
            outAlbedoRGB->clear();
        } else {
            captured = true;
        }
    }
    if (outNormalRGB) {
        uint32_t normalWidth = 0;
        uint32_t normalHeight = 0;
        std::vector<float> dummyAlpha;
        if (!captureTextureImage(m_accumulation.normalBuffer(),
                                 *outNormalRGB,
                                 normalWidth,
                                 normalHeight,
                                 &dummyAlpha,
                                 false)) {
            outNormalRGB->clear();
        } else {
            if (captured &&
                (normalWidth != localWidth || normalHeight != localHeight)) {
                if (outAlbedoRGB) {
                    outAlbedoRGB->clear();
                }
                outNormalRGB->clear();
                return false;
            }
            localWidth = normalWidth;
            localHeight = normalHeight;
            captured = true;
        }
    }
    width = localWidth;
    height = localHeight;
    return captured;
}

MetalRenderer::RuntimeStatsSnapshot MetalRenderer::Impl::runtimeStatsSnapshot() const {
    RuntimeStatsSnapshot snapshot{};
    snapshot.sampleCount = m_accumulation.sampleCount();
    snapshot.frameTimeMs = m_lastFrameTimeMs;
    snapshot.gpuTimeMs = m_performanceStats.gpuTimeMs;
    snapshot.cpuEncodeMs = m_performanceStats.cpuEncodeMs;
    snapshot.drawableWaitMs = m_performanceStats.drawableWaitMs;
    snapshot.gpuBucketTimingAvailable = m_performanceStats.gpuBucketTimingAvailable;
    snapshot.gpuBucketIntersectMs = m_performanceStats.gpuBucketIntersectMs;
    snapshot.gpuBucketShadeMs = m_performanceStats.gpuBucketShadeMs;
    snapshot.gpuBucketShadowQueueMs = m_performanceStats.gpuBucketShadowQueueMs;
    snapshot.gpuBucketFusedIntegrateMs = m_performanceStats.gpuBucketFusedIntegrateMs;
    snapshot.gpuBucketPresentationMs = m_performanceStats.gpuBucketPresentationMs;
    snapshot.hardwareRaytracingActive = m_performanceStats.hardwareRaytracingActive;
    snapshot.hardwareRaytracingAvailable = m_performanceStats.hardwareRaytracingAvailable;
    snapshot.intersectionMode = m_performanceStats.intersectionMode;
    const auto& activeScene = activeSceneResources();
    const auto& accelStats = activeScene.accelerationBuildStats();
    snapshot.hardwareRaytracingFallbackActive =
        activeScene.supportsRaytracing() &&
        !accelStats.usedHardwareRaytracing &&
        (m_settings.enableSoftwareRayTracing || !accelStats.fallbackReason.empty());
    snapshot.hardwareRaytracingFallbackReason =
        !accelStats.fallbackReason.empty()
            ? accelStats.fallbackReason
            : (m_settings.enableSoftwareRayTracing ? "software BVH requested explicitly"
                                                   : std::string());
    if (id<MTLDevice> device = m_context.device()) {
        snapshot.deviceName = device.name ? std::string([device.name UTF8String]) : std::string();
        snapshot.deviceRecommendedMaxWorkingSetBytes = DeviceRecommendedWorkingSetBytes(device);
        snapshot.deviceSupportsRaytracing = m_context.supportsRaytracing();
        snapshot.deviceSupportsCounterSampling =
            [device respondsToSelector:@selector(supportsCounterSampling:)] &&
            [device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary];
        snapshot.deviceSupportsArgumentBuffers = true;
        snapshot.deviceSupportsSparseResources = false;
        snapshot.deviceSupportsMetalFX = false;
        snapshot.deviceSupportsMLBackend =
            m_settings.mlBackendMode != PathTracer::RenderSettings::MlBackendMode::Off;
    }
    snapshot.accelerationBuildPolicy = accelStats.buildPolicyLabel;
    snapshot.accelerationIdentityStable = accelStats.identityStable;
    snapshot.accelerationIdentityStrategy = accelStats.identityStrategy;
    snapshot.accelerationIntersectionFunctionStrategy =
        accelStats.intersectionFunctionStrategy;
    snapshot.accelerationCustomIntersectionFunctionsSupported =
        accelStats.customIntersectionFunctionsSupported;
    snapshot.accelerationCustomIntersectionFunctionsEnabled =
        accelStats.customIntersectionFunctionsEnabled;
    snapshot.accelerationPrimitiveCount = accelStats.primitiveCount;
    snapshot.accelerationInstanceCount = accelStats.instanceCount;
    snapshot.accelerationBlasCount = accelStats.blasCount;
    snapshot.renderExecutionMode = m_performanceStats.renderExecutionMode;
    snapshot.wavefrontSchedulingPolicy = m_performanceStats.wavefrontSchedulingPolicy;
    snapshot.wavefrontCompactionEnabled = m_performanceStats.wavefrontCompactionEnabled;
    snapshot.currentAllocatedBytes = m_performanceStats.currentAllocatedBytes;
    snapshot.recommendedMaxWorkingSetBytes = m_performanceStats.recommendedMaxWorkingSetBytes;
    snapshot.totalComputeDispatchCount = m_performanceStats.totalComputeDispatchCount;
    snapshot.restirDiInitializeDispatchCount =
        m_performanceStats.restirDiInitializeDispatchCount;
    snapshot.restirDiInitialCandidateDispatchCount =
        m_performanceStats.restirDiInitialCandidateDispatchCount;
    snapshot.restirDiTemporalReuseDispatchCount =
        m_performanceStats.restirDiTemporalReuseDispatchCount;
    snapshot.restirDiSpatialReuseDispatchCount =
        m_performanceStats.restirDiSpatialReuseDispatchCount;
    snapshot.restirDiVisibilityDispatchCount =
        m_performanceStats.restirDiVisibilityDispatchCount;
    snapshot.restirDiApplyLightingDispatchCount =
        m_performanceStats.restirDiApplyLightingDispatchCount;
    snapshot.restirDiDebugAovDispatchCount =
        m_performanceStats.restirDiDebugAovDispatchCount;
    snapshot.integrationDispatchCount = m_performanceStats.integrationDispatchCount;
    snapshot.presentationDispatchCount = m_performanceStats.presentationDispatchCount;
    snapshot.wavefrontResetDispatchCount = m_performanceStats.wavefrontResetDispatchCount;
    snapshot.wavefrontGenerateDispatchCount = m_performanceStats.wavefrontGenerateDispatchCount;
    snapshot.wavefrontBuildDispatchArgsDispatchCount =
        m_performanceStats.wavefrontBuildDispatchArgsDispatchCount;
    snapshot.wavefrontCompactQueueDispatchCount =
        m_performanceStats.wavefrontCompactQueueDispatchCount;
    snapshot.wavefrontPrepareDispatchCount = m_performanceStats.wavefrontPrepareDispatchCount;
    snapshot.wavefrontIntersectDispatchCount = m_performanceStats.wavefrontIntersectDispatchCount;
    snapshot.wavefrontShadeDispatchCount = m_performanceStats.wavefrontShadeDispatchCount;
    snapshot.wavefrontShadowDispatchCount = m_performanceStats.wavefrontShadowDispatchCount;
    snapshot.wavefrontSwapDispatchCount = m_performanceStats.wavefrontSwapDispatchCount;
    snapshot.wavefrontAccumulateDispatchCount = m_performanceStats.wavefrontAccumulateDispatchCount;
    snapshot.wavefrontQueueBytesTotal = m_performanceStats.wavefrontQueueBytesTotal;
    snapshot.wavefrontBudgetBytes = m_performanceStats.wavefrontBudgetBytes;
    snapshot.wavefrontBudgetUsagePercent = m_performanceStats.wavefrontBudgetUsagePercent;
    const bool wavefrontExecution =
        m_settings.executionMode == PathTracer::RenderSettings::ExecutionMode::Wavefront;
    snapshot.wavefrontIntersectDispatchMode =
        wavefrontExecution ? "active_count_indirect" : "disabled_non_wavefront";
    snapshot.wavefrontShadeDispatchMode =
        wavefrontExecution ? "direct_grid_fallback" : "disabled_non_wavefront";
    snapshot.wavefrontShadowDispatchMode =
        wavefrontExecution ? "direct_grid_sparse_storage" : "disabled_non_wavefront";
    snapshot.wavefrontQueueStoragePolicySummary =
        wavefrontExecution
            ? "active/path queues physical_compact when compaction is enabled; shadow queue physical_sparse"
            : "metadata_only";
    snapshot.wavefrontM1BridgeStatus =
        wavefrontExecution
            ? "M1 bridge partial: intersect active-count indirect is live; shade remains direct-grid fallback; shadow remains direct-grid over sparse path-indexed storage pending dense shadow index queue"
            : "wavefront bridge inactive for megakernel execution";
    snapshot.restirDiResourceBytesTotal = m_renderLoop.restirDiResourceBytesTotal();
    snapshot.restirDiReservoirBytesEach = m_renderLoop.restirDiReservoirBytesEach();
    snapshot.restirDiVisibilityBytes = m_renderLoop.restirDiVisibilityBytes();
    snapshot.restirDiDebugBytes = m_renderLoop.restirDiDebugBytes();
    snapshot.transportMemoryBytesTotal = m_renderLoop.transportMemoryBytesTotal();
    snapshot.transportMemoryBytesPrivate = m_renderLoop.transportMemoryBytesPrivate();
    snapshot.transportMemoryBytesShared = m_renderLoop.transportMemoryBytesShared();
    snapshot.transportMemoryBytesStaging = m_renderLoop.transportMemoryBytesStaging();
    snapshot.transportMemoryRecordCount = m_renderLoop.transportMemoryRecordCount();
    snapshot.transportMemoryInvalidatedRecordCount =
        m_renderLoop.transportMemoryInvalidatedRecordCount();
    const auto& transportMemory = m_renderLoop.transportMemoryRegistry().records();
    snapshot.transportMemoryNames.reserve(transportMemory.size());
    snapshot.transportMemoryKinds.reserve(transportMemory.size());
    snapshot.transportMemoryOwners.reserve(transportMemory.size());
    snapshot.transportMemoryTargetOwners.reserve(transportMemory.size());
    snapshot.transportMemoryStorage.reserve(transportMemory.size());
    snapshot.transportMemoryTargetStorage.reserve(transportMemory.size());
    snapshot.transportMemoryHistory.reserve(transportMemory.size());
    snapshot.transportMemoryDebugReadback.reserve(transportMemory.size());
    snapshot.transportMemoryLastInvalidations.reserve(transportMemory.size());
    snapshot.transportMemoryByteSizeFormulas.reserve(transportMemory.size());
    snapshot.transportMemoryFeatureGates.reserve(transportMemory.size());
    snapshot.transportMemoryMigrationRisks.reserve(transportMemory.size());
    snapshot.transportMemoryBytes.reserve(transportMemory.size());
    snapshot.transportMemoryWidths.reserve(transportMemory.size());
    snapshot.transportMemoryHeights.reserve(transportMemory.size());
    snapshot.transportMemoryDepths.reserve(transportMemory.size());
    snapshot.transportMemoryInvalidationCounts.reserve(transportMemory.size());
    snapshot.transportMemoryHotPath.reserve(transportMemory.size());
    snapshot.transportMemoryTemporaryShared.reserve(transportMemory.size());
    for (const auto& record : transportMemory) {
        snapshot.transportMemoryNames.push_back(record.name);
        snapshot.transportMemoryKinds.push_back(PathTracer::TransportMemoryKindLabel(record.kind));
        snapshot.transportMemoryOwners.push_back(PathTracer::TransportMemoryOwnerLabel(record.owner));
        snapshot.transportMemoryTargetOwners.push_back(
            PathTracer::TransportMemoryOwnerLabel(record.targetOwner));
        snapshot.transportMemoryStorage.push_back(
            PathTracer::TransportMemoryStorageClassLabel(record.storage));
        snapshot.transportMemoryTargetStorage.push_back(
            PathTracer::TransportMemoryStorageClassLabel(record.targetStorage));
        snapshot.transportMemoryHistory.push_back(
            PathTracer::TransportMemoryHistoryPolicyLabel(record.history));
        snapshot.transportMemoryDebugReadback.push_back(
            PathTracer::TransportMemoryDebugReadbackPolicyLabel(record.debug));
        snapshot.transportMemoryLastInvalidations.push_back(
            PathTracer::HistoryResetReasonLabel(record.lastInvalidationReason));
        snapshot.transportMemoryByteSizeFormulas.push_back(record.byteSizeFormula);
        snapshot.transportMemoryFeatureGates.push_back(record.featureGate);
        snapshot.transportMemoryMigrationRisks.push_back(record.migrationRisk);
        snapshot.transportMemoryBytes.push_back(record.bytes);
        snapshot.transportMemoryWidths.push_back(record.width);
        snapshot.transportMemoryHeights.push_back(record.height);
        snapshot.transportMemoryDepths.push_back(record.depthOrCellCount);
        snapshot.transportMemoryInvalidationCounts.push_back(record.invalidationCount);
        snapshot.transportMemoryHotPath.push_back(record.hotPath ? 1u : 0u);
        snapshot.transportMemoryTemporaryShared.push_back(record.temporaryShared ? 1u : 0u);
    }
    const auto& passGraph = m_renderLoop.lastPassGraph();
    const auto& passGraphExecution = m_renderLoop.lastPassGraphExecution();
    snapshot.passGraphVersion = passGraph.version;
    snapshot.passGraphCompiled = passGraphExecution.compiled;
    snapshot.passGraphCompiledExecutablePassCount =
        passGraphExecution.compiledExecutablePassCount;
    snapshot.passGraphCompiledSkippedPassCount =
        passGraphExecution.compiledSkippedPassCount;
    snapshot.passGraphCompiledUsedResourceCount =
        passGraphExecution.compiledUsedResourceCount;
    snapshot.passGraphAliasableTransientResourceCount =
        passGraphExecution.compiledAliasableTransientResourceCount;
    snapshot.passGraphTransientAliasPoolCount =
        passGraphExecution.compiledTransientAliasPoolCount;
    snapshot.passGraphResourceTransitionCount =
        passGraphExecution.compiledResourceTransitionCount;
    snapshot.passGraphRequiredBarrierCount =
        passGraphExecution.compiledRequiredBarrierCount;
    snapshot.passGraphWriteTransitionCount =
        passGraphExecution.compiledWriteTransitionCount;
    snapshot.historyResetReason =
        PathTracer::HistoryResetReasonLabel(m_renderLoop.lastFrameHistoryResetReason());
    snapshot.passGraphCompileWarnings = passGraphExecution.compileWarnings;
    snapshot.passGraphCompileErrors = passGraphExecution.compileErrors;
    snapshot.passGraphExecutedPasses = passGraphExecution.executedPasses;
    snapshot.passGraphSkippedPasses = passGraphExecution.skippedPasses;
    snapshot.passGraphFailedPasses = passGraphExecution.failedPasses;
    snapshot.passGraphNames.reserve(passGraph.passes.size());
    snapshot.passGraphTypes.reserve(passGraph.passes.size());
    snapshot.passGraphExecutions.reserve(passGraph.passes.size());
    snapshot.passGraphPipelines.reserve(passGraph.passes.size());
    snapshot.passGraphDispatches.reserve(passGraph.passes.size());
    snapshot.passGraphDebugLabels.reserve(passGraph.passes.size());
    snapshot.passGraphTimingLabels.reserve(passGraph.passes.size());
    snapshot.passGraphDebugPolicies.reserve(passGraph.passes.size());
    snapshot.passGraphTimingPolicies.reserve(passGraph.passes.size());
    for (const auto& pass : passGraph.passes) {
        snapshot.passGraphNames.push_back(pass.name);
        snapshot.passGraphTypes.push_back(PathTracer::RenderPassTypeLabel(pass.type));
        snapshot.passGraphExecutions.push_back(PathTracer::RenderPassExecutionLabel(pass.executionId));
        snapshot.passGraphPipelines.push_back(pass.pipelineKey);
        snapshot.passGraphDispatches.push_back(pass.dispatch.dimensions);
        snapshot.passGraphDebugLabels.push_back(pass.debugLabel);
        snapshot.passGraphTimingLabels.push_back(pass.timingLabel);
        snapshot.passGraphDebugPolicies.push_back(PathTracer::DebugCapturePolicyLabel(pass.debugPolicy));
        snapshot.passGraphTimingPolicies.push_back(PathTracer::TimingPolicyLabel(pass.timingPolicy));
        if (pass.executionId == PathTracer::RenderPassExecutionId::MlBackendNoOp) {
            ++snapshot.mlBackendPassCount;
        }
    }
    snapshot.mlBackendMode = static_cast<uint32_t>(m_settings.mlBackendMode);
    snapshot.mlBackendRequested =
        m_settings.mlBackendMode != PathTracer::RenderSettings::MlBackendMode::Off;
    snapshot.mlBackendActive = snapshot.mlBackendPassCount > 0u;
    snapshot.mlBackendNativeAccelerationAvailable = false;
    snapshot.mlBackendProvider = snapshot.mlBackendRequested ? "MLBackend.NoOp" : "disabled";
    snapshot.mlBackendProviderVersion = snapshot.mlBackendRequested ? "noop-contract-v1" : "disabled";
    snapshot.mlBackendTensorContract =
        "radiance,albedo,normal,depth,motion,variance,confidence -> reconstructed_radiance,confidence";
    snapshot.mlBackendFallbackReason =
        snapshot.mlBackendRequested
            ? "native ML acceleration backend is not selected; no-op backend preserves renderer output"
            : "ml backend disabled";
    snapshot.mlBackendConfidence = snapshot.mlBackendActive ? 1.0f : 0.0f;
    snapshot.mlBackendBiasRisk = 0.0f;
    snapshot.mlModelPackagePath = m_settings.mlModelPackagePath;
    const auto& pipelineReport = m_pipelines.compilationReport();
    snapshot.pipelineCompilationAvailable = pipelineReport.available;
    snapshot.pipelineRuntimeSourceCompilation = pipelineReport.runtimeSourceCompilation;
    snapshot.pipelineFunctionConstantsUsed = pipelineReport.functionConstantsUsed;
    snapshot.pipelineMetalBinaryArchiveUsed = pipelineReport.metalBinaryArchiveUsed;
    snapshot.pipelineHarvestedCompilationReady = pipelineReport.harvestedCompilationReady;
    snapshot.pipelineShaderSourceBytes = pipelineReport.shaderSourceBytes;
    snapshot.pipelineLibraryCompileMs = pipelineReport.libraryCompileMs;
    snapshot.pipelineComputePipelineCount = pipelineReport.computePipelineCount;
    snapshot.pipelineRenderPipelineCount = pipelineReport.renderPipelineCount;
    snapshot.pipelineFailedPipelineCount = pipelineReport.failedPipelineCount;
    snapshot.pipelineCacheStrategy = pipelineReport.cacheStrategy;
    snapshot.pipelineSpecializationStrategy = pipelineReport.specializationStrategy;
    snapshot.pipelineCompileKeys.reserve(pipelineReport.records.size());
    snapshot.pipelineCompileFunctions.reserve(pipelineReport.records.size());
    snapshot.pipelineCompileStages.reserve(pipelineReport.records.size());
    snapshot.pipelineCompileMs.reserve(pipelineReport.records.size());
    snapshot.pipelineCompileSuccess.reserve(pipelineReport.records.size());
    snapshot.pipelineCompileRayTracingFlags.reserve(pipelineReport.records.size());
    snapshot.pipelineCompileSpecializations.reserve(pipelineReport.records.size());
    for (const auto& record : pipelineReport.records) {
        snapshot.pipelineCompileKeys.push_back(record.key);
        snapshot.pipelineCompileFunctions.push_back(record.functionName);
        snapshot.pipelineCompileStages.push_back(record.stage);
        snapshot.pipelineCompileMs.push_back(record.compileMs);
        snapshot.pipelineCompileSuccess.push_back(record.success ? 1u : 0u);
        snapshot.pipelineCompileRayTracingFlags.push_back(record.rayTracingSupportFlag ? 1u : 0u);
        snapshot.pipelineCompileSpecializations.push_back(record.specialization);
    }
    snapshot.passGraphResourceNames.reserve(passGraph.resources.size());
    snapshot.passGraphResourceLifetimes.reserve(passGraph.resources.size());
    snapshot.passGraphResourceStorage.reserve(passGraph.resources.size());
    snapshot.passGraphResourceTargetStorage.reserve(passGraph.resources.size());
    snapshot.passGraphResourceSizeFormulas.reserve(passGraph.resources.size());
    snapshot.passGraphResourceDebugLabels.reserve(passGraph.resources.size());
    snapshot.passGraphResourceCurrentOwners.reserve(passGraph.resources.size());
    snapshot.passGraphResourceTargetOwners.reserve(passGraph.resources.size());
    snapshot.passGraphResourceFeatureGates.reserve(passGraph.resources.size());
    snapshot.passGraphResourceDebugReadbackPolicies.reserve(passGraph.resources.size());
    snapshot.passGraphResourceMigrationRisks.reserve(passGraph.resources.size());
    snapshot.passGraphResourceFirstPasses.reserve(passGraph.resources.size());
    snapshot.passGraphResourceLastPasses.reserve(passGraph.resources.size());
    snapshot.passGraphResourceAliasPools.reserve(passGraph.resources.size());
    snapshot.passGraphResourceUsed.reserve(passGraph.resources.size());
    snapshot.passGraphResourceAliasable.reserve(passGraph.resources.size());
    snapshot.passGraphResourceHotPath.reserve(passGraph.resources.size());
    snapshot.passGraphResourceTemporaryShared.reserve(passGraph.resources.size());
    for (size_t i = 0; i < passGraph.resources.size(); ++i) {
        const auto& resource = passGraph.resources[i];
        snapshot.passGraphResourceNames.push_back(resource.name);
        snapshot.passGraphResourceLifetimes.push_back(PathTracer::ResourceLifetimeLabel(resource.lifetime));
        snapshot.passGraphResourceStorage.push_back(PathTracer::StorageClassLabel(resource.storageClass));
        snapshot.passGraphResourceTargetStorage.push_back(
            PathTracer::StorageClassLabel(resource.targetStorageClass));
        snapshot.passGraphResourceSizeFormulas.push_back(resource.sizeFormula);
        snapshot.passGraphResourceDebugLabels.push_back(resource.debugLabel);
        snapshot.passGraphResourceCurrentOwners.push_back(resource.currentOwner);
        snapshot.passGraphResourceTargetOwners.push_back(resource.targetOwner);
        snapshot.passGraphResourceFeatureGates.push_back(resource.featureGate);
        snapshot.passGraphResourceDebugReadbackPolicies.push_back(resource.debugReadbackPolicy);
        snapshot.passGraphResourceMigrationRisks.push_back(resource.migrationRisk);
        snapshot.passGraphResourceHotPath.push_back(resource.hotPath ? 1u : 0u);
        snapshot.passGraphResourceTemporaryShared.push_back(resource.temporaryShared ? 1u : 0u);
        if (i < passGraphExecution.resourcePlans.size()) {
            const auto& plan = passGraphExecution.resourcePlans[i];
            snapshot.passGraphResourceFirstPasses.push_back(plan.firstPassIndex);
            snapshot.passGraphResourceLastPasses.push_back(plan.lastPassIndex);
            snapshot.passGraphResourceAliasPools.push_back(plan.aliasPoolIndex);
            snapshot.passGraphResourceUsed.push_back(plan.used ? 1u : 0u);
            snapshot.passGraphResourceAliasable.push_back(plan.aliasable ? 1u : 0u);
        } else {
            snapshot.passGraphResourceFirstPasses.push_back(-1);
            snapshot.passGraphResourceLastPasses.push_back(-1);
            snapshot.passGraphResourceAliasPools.push_back(-1);
            snapshot.passGraphResourceUsed.push_back(0u);
            snapshot.passGraphResourceAliasable.push_back(0u);
        }
    }
    snapshot.passGraphTransitionPassIndices.reserve(passGraphExecution.resourceTransitions.size());
    snapshot.passGraphTransitionPassNames.reserve(passGraphExecution.resourceTransitions.size());
    snapshot.passGraphTransitionResourceNames.reserve(passGraphExecution.resourceTransitions.size());
    snapshot.passGraphTransitionPreviousAccesses.reserve(passGraphExecution.resourceTransitions.size());
    snapshot.passGraphTransitionAccesses.reserve(passGraphExecution.resourceTransitions.size());
    snapshot.passGraphTransitionFirstUse.reserve(passGraphExecution.resourceTransitions.size());
    snapshot.passGraphTransitionRequiresBarrier.reserve(passGraphExecution.resourceTransitions.size());
    snapshot.passGraphTransitionWritesResource.reserve(passGraphExecution.resourceTransitions.size());
    for (const auto& transition : passGraphExecution.resourceTransitions) {
        snapshot.passGraphTransitionPassIndices.push_back(transition.passIndex);
        snapshot.passGraphTransitionPassNames.push_back(transition.passName);
        snapshot.passGraphTransitionResourceNames.push_back(transition.resourceName);
        snapshot.passGraphTransitionPreviousAccesses.push_back(
            PathTracer::ResourceAccessLabel(transition.previousAccess));
        snapshot.passGraphTransitionAccesses.push_back(
            PathTracer::ResourceAccessLabel(transition.access));
        snapshot.passGraphTransitionFirstUse.push_back(transition.firstUse ? 1u : 0u);
        snapshot.passGraphTransitionRequiresBarrier.push_back(transition.requiresBarrier ? 1u : 0u);
        snapshot.passGraphTransitionWritesResource.push_back(transition.writesResource ? 1u : 0u);
    }

    id<MTLBuffer> statsBuffer = m_renderLoop.statsBuffer();
    if (!statsBuffer || ![statsBuffer contents]) {
        return snapshot;
    }

    const auto* stats = reinterpret_cast<const PathtraceStats*>([statsBuffer contents]);
    const uint64_t primaryRayCount = static_cast<uint64_t>(stats->primaryRayCount);
    const uint64_t hardwareRayCount = static_cast<uint64_t>(stats->hardwareRayCount);
    const uint64_t shadowRayCount = static_cast<uint64_t>(stats->shadowRayCount);
    snapshot.primaryRayCount = primaryRayCount;
    snapshot.hardwareRayCount = hardwareRayCount;
    snapshot.pathSegmentCount = primaryRayCount + hardwareRayCount;
    snapshot.shadowRayCount = shadowRayCount;
    snapshot.tracedRayCount = snapshot.pathSegmentCount + shadowRayCount;
    snapshot.wavefrontPrimaryGeneratedCount =
        static_cast<uint64_t>(stats->wavefrontPrimaryGeneratedCount);
    snapshot.wavefrontIntersectProcessedCount =
        static_cast<uint64_t>(stats->wavefrontIntersectProcessedCount);
    snapshot.wavefrontShadeProcessedCount =
        static_cast<uint64_t>(stats->wavefrontShadeProcessedCount);
    snapshot.wavefrontShadowProcessedCount =
        static_cast<uint64_t>(stats->wavefrontShadowProcessedCount);
    snapshot.wavefrontNextRayEnqueueCount =
        static_cast<uint64_t>(stats->wavefrontNextRayEnqueueCount);
    snapshot.wavefrontShadowRayEnqueueCount =
        static_cast<uint64_t>(stats->wavefrontShadowRayEnqueueCount);
    for (uint32_t i = 0u; i < PerformanceStats::kWavefrontQueueMetricCount; ++i) {
        snapshot.wavefrontQueueActiveCounts[i] = m_performanceStats.wavefrontQueueActiveCounts[i];
        snapshot.wavefrontQueuePeakActiveCounts[i] =
            m_performanceStats.wavefrontQueuePeakActiveCounts[i];
        snapshot.wavefrontQueueCapacities[i] = m_performanceStats.wavefrontQueueCapacities[i];
        snapshot.wavefrontQueueOverflowCounts[i] =
            m_performanceStats.wavefrontQueueOverflowCounts[i];
        snapshot.wavefrontQueueOccupancyRatios[i] =
            m_performanceStats.wavefrontQueueOccupancyRatios[i];
        snapshot.wavefrontQueuePeakOccupancyRatios[i] =
            m_performanceStats.wavefrontQueuePeakOccupancyRatios[i];
    }
    snapshot.wavefrontQueueCompactionCostMs = m_performanceStats.wavefrontQueueCompactionCostMs;

    const uint64_t pathCount =
        static_cast<uint64_t>(m_performanceStats.renderWidth) *
        static_cast<uint64_t>(m_performanceStats.renderHeight) *
        static_cast<uint64_t>(snapshot.sampleCount);
    if (pathCount > 0u) {
        snapshot.averageBounceDepth =
            static_cast<double>(snapshot.pathSegmentCount) / static_cast<double>(pathCount);
    }
    snapshot.available = true;
    return snapshot;
}

void MetalRenderer::Impl::handleSaveExrRequest(const std::string& path) {
    if (path.empty()) {
        m_overlay.notifySaveExrResult(false, "Invalid save path");
        NSLog(@"[Output] Save EXR aborted: empty path");
        return;
    }

    std::vector<float> linearRGB;
    std::vector<float> sampleCounts;
    uint32_t width = 0;
    uint32_t height = 0;
    if (!captureAverageImage(linearRGB, width, height, &sampleCounts) || width == 0 || height == 0) {
        m_overlay.notifySaveExrResult(false, "No accumulation data");
        NSLog(@"[Output] Save EXR failed: no accumulation data available");
        return;
    }

    const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (linearRGB.size() != pixelCount * 3ull) {
        m_overlay.notifySaveExrResult(false, "Unexpected buffer size");
        NSLog(@"[Output] Save EXR failed: unexpected RGB buffer size");
        return;
    }

    std::vector<float> rgba(pixelCount * 4ull);
    for (size_t i = 0; i < pixelCount; ++i) {
        rgba[4 * i + 0] = linearRGB[3 * i + 0];
        rgba[4 * i + 1] = linearRGB[3 * i + 1];
        rgba[4 * i + 2] = linearRGB[3 * i + 2];
        const float alpha = sampleCounts.empty() ? 1.0f : sampleCounts[i];
        rgba[4 * i + 3] = alpha;
    }

    const bool hasSamples = !sampleCounts.empty();
    bool success = false;
    namespace fs = std::filesystem;
    fs::path targetPath(path);
    std::error_code dirEc;
    if (!targetPath.has_parent_path()) {
        targetPath = fs::path("renders") / targetPath;
    }
    fs::path parent = targetPath.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent, dirEc);
    }

    if (hasSamples) {
        success = PathTracer::ImageWriter::WriteEXR_Multilayer(targetPath.string().c_str(),
                                                               rgba.data(),
                                                               static_cast<int>(width),
                                                               static_cast<int>(height),
                                                               sampleCounts.data(),
                                                               WorkingColorSpaceLabel(m_settings));
    } else {
        success = PathTracer::ImageWriter::WriteEXR(targetPath.string().c_str(),
                                                    rgba.data(),
                                                    static_cast<int>(width),
                                                    static_cast<int>(height),
                                                    WorkingColorSpaceLabel(m_settings));
    }

    if (success) {
        NSLog(@"[Output] Saved EXR to %s (%ux%u%s)",
              targetPath.string().c_str(),
              width,
              height,
              hasSamples ? " + SAMPLES" : "");
        std::string status = "Saved " + std::to_string(width) + "x" + std::to_string(height) + " EXR to ";
        status += targetPath.string();
        if (hasSamples) {
            status += " (+SAMPLES)";
        }
        m_overlay.notifySaveExrResult(true, status);
    } else {
        NSLog(@"[Output] Failed to save EXR to %s", targetPath.string().c_str());
        m_overlay.notifySaveExrResult(false, "Failed to save EXR");
    }
}

bool MetalRenderer::Impl::exportToPPM(const char* filepath) {
    if (!filepath) {
        NSLog(@"Invalid filepath for PPM export");
        return false;
    }

    std::vector<float> linearRGB;
    uint32_t width = 0;
    uint32_t height = 0;
    if (!captureAverageImage(linearRGB, width, height, nullptr)) {
        NSLog(@"No render data to export");
        return false;
    }

    PathTracer::TonemapSettings tonemap{};
    tonemap.tonemapMode = 1;
    tonemap.acesVariant = 0;
    tonemap.exposure = 0.0f;
    tonemap.reinhardWhitePoint = 1.5f;

    std::string error;
    if (!PathTracer::WriteImage(std::string(filepath), PathTracer::ImageFileFormat::PPM,
                                linearRGB.data(), width, height, tonemap, &error)) {
        if (!error.empty()) {
            NSLog(@"PPM export failed: %s", error.c_str());
        }
        return false;
    }

    NSLog(@"Successfully exported PPM to: %s (%ux%u, %u samples)",
          filepath, width, height, m_accumulation.sampleCount());
    return true;
}

void MetalRenderer::Impl::shutdown() {
    if (m_shutdown) {
        return;
    }
    m_shutdown = true;
    cancelPendingSceneLoad();
    m_sceneLoadWorkerActive = false;

    if (m_backingChangeObserver) {
        [[NSNotificationCenter defaultCenter] removeObserver:m_backingChangeObserver];
        m_backingChangeObserver = nil;
        m_context.setBackingChangeObserver(nil);
    }
    if (m_windowResizeObserver) {
        [[NSNotificationCenter defaultCenter] removeObserver:m_windowResizeObserver];
        m_windowResizeObserver = nil;
    }
    if (m_windowMoveObserver) {
        [[NSNotificationCenter defaultCenter] removeObserver:m_windowMoveObserver];
        m_windowMoveObserver = nil;
    }
    if (m_windowWillMoveObserver) {
        [[NSNotificationCenter defaultCenter] removeObserver:m_windowWillMoveObserver];
        m_windowWillMoveObserver = nil;
    }

    if (auto queue = m_context.commandQueue()) {
        id<MTLCommandBuffer> drain = [queue commandBuffer];
        if (drain) {
            drain.label = @"Renderer Shutdown Drain";
            [drain commit];

            const double waitStart = CFAbsoluteTimeGetCurrent();
            while (true) {
                const MTLCommandBufferStatus status = drain.status;
                if (status == MTLCommandBufferStatusCompleted ||
                    status == MTLCommandBufferStatusError) {
                    break;
                }
                const double elapsedMs = (CFAbsoluteTimeGetCurrent() - waitStart) * 1000.0;
                if (elapsedMs > 500.0) {
                    NSLog(@"[Shutdown] skipping drain wait after %.0fms (status=%ld)",
                          elapsedMs,
                          static_cast<long>(status));
                    break;
                }
                [NSThread sleepForTimeInterval:0.001];
            }
        }
    }

    m_pipelines.shutdown();
#if HAS_IMGUI
    if (m_overlayInitialized) {
        m_overlay.shutdown();
        m_overlayInitialized = false;
    }
#endif

    m_renderLoop.shutdown();
    m_loadedSceneResources.reset();
    m_sceneResources.clear();
    m_accumulation.teardown();
    m_sceneLoadWorkerQueue = nullptr;

    m_context.shutdown();
}

MetalRenderer::Impl::~Impl() {
    shutdown();
}

MetalRenderer::MetalRenderer() : m_impl(std::make_shared<Impl>()) {}
MetalRenderer::~MetalRenderer() = default;

bool MetalRenderer::init(void* windowHandle, const MetalRendererOptions& options) {
    // Headless mode doesn't require a window
    if (!options.headless && !windowHandle) {
        return false;
    }

    NSWindow* window = nullptr;
    if (windowHandle) {
        window = (__bridge NSWindow*)windowHandle;
        [window setTitle:[NSString stringWithUTF8String:options.windowTitle.c_str()]];
    }
    return m_impl->initialize(window, options);
}

void MetalRenderer::drawFrame() {
    if (m_impl) {
        @autoreleasepool {
            m_impl->drawFrame();
        }
    }
}

void MetalRenderer::resize(int width, int height) {
    if (m_impl) {
        m_impl->resize(width, height);
    }
}

void MetalRenderer::resetAccumulation() {
    if (m_impl) {
        m_impl->resetAccumulation();
    }
}

void MetalRenderer::setTonemapMode(uint32_t mode) {
    if (m_impl) {
        m_impl->setTonemapMode(mode);
    }
}

void MetalRenderer::setPresentationEnabled(bool enabled) {
    if (m_impl) {
        m_impl->setPresentationEnabled(enabled);
    }
}

bool MetalRenderer::isPresentationEnabled() const {
    if (!m_impl) {
        return false;
    }
    return m_impl->presentationEnabled();
}

void MetalRenderer::togglePresentationUIPanels() {
    if (m_impl) {
        m_impl->togglePresentationUIPanels();
    }
}

void MetalRenderer::shutdown() {
    if (m_impl) {
        m_impl->shutdown();
    }
}

bool MetalRenderer::exportToPPM(const char* filepath) {
    if (m_impl) {
        return m_impl->exportToPPM(filepath);
    }
    return false;
}

bool MetalRenderer::loadSceneFromPath(const char* path) {
    if (!m_impl || !path) {
        return false;
    }
    return m_impl->loadSceneFromPath(path);
}

bool MetalRenderer::useSceneResources(PathTracer::SceneResources& resources, const char* identifier) {
    if (!m_impl) {
        return false;
    }
    return m_impl->useSceneResources(resources, identifier ? std::string(identifier) : std::string{});
}

bool MetalRenderer::setScene(const char* identifier) {
    if (!m_impl || !identifier) {
        return false;
    }
    std::string idStr(identifier);
    if (idStr.find('/') != std::string::npos || idStr.find(".scene") != std::string::npos) {
        return m_impl->loadSceneFromPath(idStr);
    }
    return m_impl->loadScene(idStr);
}

std::vector<std::string> MetalRenderer::sceneIdentifiers() const {
    if (!m_impl) {
        return {};
    }
    return m_impl->sceneIdentifiers();
}

PathTracer::RenderSettings MetalRenderer::settings() const {
    if (!m_impl) {
        return PathTracer::RenderSettings{};
    }
    return m_impl->currentSettings();
}

void MetalRenderer::applySettings(const PathTracer::RenderSettings& settings, bool resetAccumulation) {
    if (m_impl) {
        m_impl->applySettings(settings, resetAccumulation);
    }
}

void MetalRenderer::setSamplesPerFrame(uint32_t samples) {
    if (m_impl) {
        m_impl->setSamplesPerFrame(samples);
    }
}

uint32_t MetalRenderer::sampleCount() const {
    if (!m_impl) {
        return 0;
    }
    return m_impl->sampleCount();
}

bool MetalRenderer::captureAverageImage(std::vector<float>& outLinearRGB,
                                        uint32_t& width,
                                        uint32_t& height,
                                        std::vector<float>* outSampleCounts) {
    if (!m_impl) {
        return false;
    }
    return m_impl->captureAverageImage(outLinearRGB, width, height, outSampleCounts);
}

bool MetalRenderer::captureRawImage(std::vector<float>& outLinearRGB,
                                    uint32_t& width,
                                    uint32_t& height,
                                    std::vector<float>* outSampleCounts) {
    if (!m_impl) {
        return false;
    }
    return m_impl->captureRawImage(outLinearRGB, width, height, outSampleCounts);
}

bool MetalRenderer::captureAuxiliaryImages(std::vector<float>* outAlbedoRGB,
                                           std::vector<float>* outNormalRGB,
                                           uint32_t& width,
                                           uint32_t& height) {
    if (!m_impl) {
        return false;
    }
    return m_impl->captureAuxiliaryImages(outAlbedoRGB, outNormalRGB, width, height);
}

bool MetalRenderer::captureRuntimeStats(RuntimeStatsSnapshot& outStats) const {
    if (!m_impl) {
        outStats = RuntimeStatsSnapshot{};
        return false;
    }
    outStats = m_impl->runtimeStatsSnapshot();
    return outStats.available;
}

bool MetalRenderer::capturePbrMetrics(PbrMetricsSnapshot& outMetrics) const {
    if (!m_impl) {
        outMetrics = PbrMetricsSnapshot{};
        return false;
    }
    outMetrics = m_impl->pbrMetricsSnapshot();
    return outMetrics.available;
}

bool MetalRenderer::Impl::captureMaterialMrDebugDump(HeadlessRenderOutput::MaterialMrDebugDump& outDump) const {
    outDump = HeadlessRenderOutput::MaterialMrDebugDump{};
#if !PT_DEBUG_TOOLS
    return false;
#else
    MTLBufferHandle debugBuffer = m_renderLoop.debugBuffer();
    if (!debugBuffer || !m_settings.enablePathDebug) {
        return false;
    }
    const auto* debugData =
        reinterpret_cast<const PathtraceDebugBuffer*>([debugBuffer contents]);
    if (!debugData) {
        return false;
    }

    const uint32_t debugCount = std::min(debugData->writeIndex, debugData->maxEntries);
    if (debugCount == 0u) {
        return false;
    }

    constexpr uint32_t kDebugEventShadingNormal = 12u;
    constexpr uint32_t kDebugEventTbnBasis0 = 13u;
    constexpr uint32_t kDebugEventTbnBasis1 = 14u;
    constexpr uint32_t kDebugEventTbnBasis2 = 15u;
    constexpr uint32_t kDebugEventTbnBasis3 = 16u;
    constexpr uint32_t kDebugEventBsdfState = 18u;
    constexpr uint32_t kDebugEventBsdfRng = 19u;
    constexpr uint32_t kDebugEventBsdfMaterial0 = 20u;
    constexpr uint32_t kDebugEventBsdfMaterial1 = 21u;
    constexpr uint32_t kDebugEventBsdfMaterial2 = 22u;
    constexpr uint32_t kDebugEventRectNee = 3u;
    constexpr uint32_t kDebugEventEnvNee = 4u;

    bool foundAny = false;
    simd::float3 wo = {0.0f, 0.0f, 0.0f};
    for (uint32_t i = 0; i < debugCount; ++i) {
        const auto& entry = debugData->entries[i];
        if (entry.sampleIndex != 0u || entry.depth != 0u) {
            continue;
        }
        if (entry.eventKind != kDebugEventBsdfMaterial0 &&
            entry.eventKind != kDebugEventBsdfMaterial1 &&
            entry.eventKind != kDebugEventBsdfMaterial2 &&
            entry.eventKind != kDebugEventBsdfState &&
            entry.eventKind != kDebugEventBsdfRng &&
            entry.eventKind != kDebugEventTbnBasis0 &&
            entry.eventKind != kDebugEventTbnBasis1 &&
            entry.eventKind != kDebugEventTbnBasis2 &&
            entry.eventKind != kDebugEventTbnBasis3 &&
            entry.eventKind != kDebugEventShadingNormal &&
            entry.eventKind != kDebugEventRectNee &&
            entry.eventKind != kDebugEventEnvNee) {
            continue;
        }
        if (!foundAny) {
            outDump.available = true;
            outDump.pixelX = entry.pixelX;
            outDump.pixelY = entry.pixelY;
            outDump.sampleIndex = entry.sampleIndex;
            outDump.depth = entry.depth;
            outDump.materialIndex = entry.materialIndex;
            foundAny = true;
        }
        switch (entry.eventKind) {
            case kDebugEventTbnBasis0:
                outDump.geometricNormal = simd_make_float3(entry.aux.x, entry.aux.y, entry.aux.z);
                outDump.uv0 = simd_make_float2(entry.scalars.x, entry.scalars.y);
                break;
            case kDebugEventTbnBasis1:
                outDump.tangent = simd_make_float3(entry.scalars.x, entry.scalars.y, entry.scalars.z);
                outDump.bitangent = simd_make_float3(entry.aux.x, entry.aux.y, entry.aux.z);
                break;
            case kDebugEventTbnBasis2:
                outDump.normalSampleRaw = simd_make_float3(entry.throughput.x, entry.throughput.y, entry.throughput.z);
                outDump.normalSampleTs = simd_make_float3(entry.aux.x, entry.aux.y, entry.aux.z);
                break;
            case kDebugEventTbnBasis3:
                outDump.perturbedNormal = simd_make_float3(entry.aux.x, entry.aux.y, entry.aux.z);
                break;
            case kDebugEventShadingNormal:
                outDump.shadingNormal = simd_make_float3(entry.aux.x, entry.aux.y, entry.aux.z);
                outDump.ndotv = entry.scalars.x != 0.0f ? 0.0f : outDump.ndotv;
                break;
            case kDebugEventBsdfRng:
                wo = simd_make_float3(entry.aux.x, entry.aux.y, entry.aux.z);
                break;
            case kDebugEventBsdfState:
                outDump.sampledOutgoingDirection = simd_make_float3(entry.aux.x, entry.aux.y, entry.aux.z);
                outDump.specularPdf = entry.scalars.y;
                break;
            case kDebugEventBsdfMaterial0:
                outDump.baseColor = simd_make_float3(entry.throughput.x, entry.throughput.y, entry.throughput.z);
                outDump.uv0 = simd_make_float2(entry.scalars.x, entry.scalars.y);
                outDump.metallic = entry.scalars.z;
                outDump.roughness = entry.scalars.w;
                outDump.normalScale = entry.aux.x;
                break;
            case kDebugEventBsdfMaterial1:
                outDump.emissive = simd_make_float3(entry.throughput.x, entry.throughput.y, entry.throughput.z);
                outDump.alpha = entry.scalars.x;
                outDump.alphaCutoff = entry.scalars.y;
                outDump.alphaMode = entry.scalars.z;
                outDump.roughnessInput = entry.scalars.w;
                outDump.f0 = simd_make_float3(entry.aux.x, entry.aux.y, entry.aux.z);
                break;
            case kDebugEventBsdfMaterial2:
                outDump.transmission = entry.throughput.x;
                outDump.metallic = entry.scalars.x;
                outDump.roughnessBeforeNormalVariance = entry.scalars.y;
                outDump.ao = entry.scalars.z;
                outDump.normalScale = entry.scalars.w;
                break;
            case kDebugEventEnvNee:
                outDump.directContribution = simd_make_float3(entry.aux.x, entry.aux.y, entry.aux.z);
                outDump.directContributionKind = 1.0f;
                break;
            case kDebugEventRectNee:
                outDump.directContribution = simd_make_float3(entry.aux.x, entry.aux.y, entry.aux.z);
                outDump.directContributionKind = 2.0f;
                break;
            default:
                break;
        }
    }

    if (!foundAny) {
        outDump = HeadlessRenderOutput::MaterialMrDebugDump{};
        return false;
    }
    if (outDump.shadingNormal.x == 0.0f && outDump.shadingNormal.y == 0.0f && outDump.shadingNormal.z == 0.0f) {
        outDump.shadingNormal = outDump.perturbedNormal;
    }
    outDump.ndotv = std::max(simd::dot(outDump.perturbedNormal, wo), 0.0f);
    outDump.specularWeight = std::clamp(std::max({outDump.f0.x, outDump.f0.y, outDump.f0.z}), 0.05f, 0.95f);
    outDump.diffuseWeight = std::max(1.0f - outDump.specularWeight, 0.0f);
    if (simd::dot(outDump.sampledOutgoingDirection, outDump.sampledOutgoingDirection) > 0.0f) {
        float NoV = std::max(simd::dot(outDump.perturbedNormal, wo), 0.0f);
        float NoL = std::max(simd::dot(outDump.perturbedNormal, outDump.sampledOutgoingDirection), 0.0f);
        simd::float3 wh = simd::normalize(wo + outDump.sampledOutgoingDirection);
        float alpha = std::max(outDump.roughness * outDump.roughness, 1.0e-4f);
        outDump.dTerm = CpuGgxDistribution(alpha, std::max(simd::dot(outDump.perturbedNormal, wh), 0.0f));
        outDump.gTerm = CpuGgxG1(alpha, NoV) * CpuGgxG1(alpha, NoL);
        outDump.fTerm = CpuSchlickFresnel(outDump.f0, std::max(simd::dot(outDump.sampledOutgoingDirection, wh), 0.0f));
        float denom = std::max(4.0f * NoV * NoL, 1.0e-6f);
        outDump.brdfSpecularValue = outDump.fTerm * (outDump.dTerm * outDump.gTerm / denom);
        if (outDump.specularPdf > 0.0f) {
            outDump.finalSpecularContribution = outDump.brdfSpecularValue * (NoL / outDump.specularPdf);
        }
    }
    return true;
#endif
}

bool MetalRenderer::Impl::captureDirectLightAuditDump(HeadlessRenderOutput::DirectLightAuditDump& outDump) const {
    outDump = HeadlessRenderOutput::DirectLightAuditDump{};
#if !PT_DEBUG_TOOLS
    return false;
#else
    MTLBufferHandle debugBuffer = m_renderLoop.debugBuffer();
    if (!debugBuffer || !m_settings.enablePathDebug || !m_settings.debugDirectLightAudit) {
        return false;
    }
    const auto* debugData =
        reinterpret_cast<const PathtraceDebugBuffer*>([debugBuffer contents]);
    if (!debugData) {
        return false;
    }

    const uint32_t debugCount = std::min(debugData->writeIndex, debugData->maxEntries);
    if (debugCount == 0u) {
        return false;
    }

    constexpr uint32_t kDebugEventShadingNormal = 12u;
    constexpr uint32_t kDebugEventBsdfRng = 19u;
    constexpr uint32_t kDebugEventDirectLightAuditMeta = 26u;
    constexpr uint32_t kDebugEventDirectLightAuditMaterial = 29u;
    constexpr uint32_t kDebugEventDirectLightAuditEval = 27u;
    constexpr uint32_t kDebugEventDirectLightAuditContrib = 28u;
    constexpr uint32_t kDebugEventDirectLightAuditShadowRaw = 30u;
    constexpr uint32_t kDebugEventDirectLightAuditShadowMapped = 31u;
    constexpr uint32_t kDebugEventDirectLightAuditShadowEmbeddedSw = 32u;
    constexpr uint32_t kDebugEventDirectLightAuditShadowBinding = 33u;
    constexpr uint32_t kDebugEventDirectLightAuditShadowReject = 34u;
    constexpr uint32_t kDebugEventRisAuditState = 35u;
    constexpr uint32_t kDebugEventRisAuditWinnerA = 36u;
    constexpr uint32_t kDebugEventRisAuditWinnerB = 37u;
    constexpr uint32_t kDebugEventRisAuditWinnerC = 38u;
    constexpr uint32_t kDebugEventSpatialReuseAudit = 39u;
    constexpr uint32_t kDebugEventSpatialReuseWeights = 40u;
    constexpr uint32_t kDebugEventTemporalReuseAudit = 41u;
    constexpr uint32_t kDebugEventTemporalReuseWeights = 42u;
    constexpr uint32_t kDebugEventWorldReuseAudit = 43u;
    constexpr uint32_t kDebugEventWorldReuseWeights = 44u;
    constexpr uint32_t kDebugEventCacheReuseAudit = 45u;
    constexpr uint32_t kDebugEventCacheReuseWeights = 46u;
    constexpr int32_t kDirectLightAuditKindEmissivePrimitive = 1;
    constexpr uint32_t kAuditEntryKindBit = 0x80000000u;

    auto makeAuditKey = [&](uint32_t index, int32_t mediumEvent) {
        return (mediumEvent == kDirectLightAuditKindEmissivePrimitive)
            ? (kAuditEntryKindBit | index)
            : index;
    };

    struct TempEntry {
        HeadlessRenderOutput::DirectLightAuditEntry value{};
        bool sawEval = false;
        bool sawContrib = false;
    };

    std::unordered_map<uint32_t, TempEntry> entries;
    bool foundAny = false;
    uint32_t selectedSampleIndex = 0u;
    bool foundDepthZeroSample = false;
    for (uint32_t i = 0; i < debugCount; ++i) {
        const auto& entry = debugData->entries[i];
        if (entry.depth == 0u) {
            selectedSampleIndex = foundDepthZeroSample
                ? std::max(selectedSampleIndex, entry.sampleIndex)
                : entry.sampleIndex;
            foundDepthZeroSample = true;
        }
    }

    for (uint32_t i = 0; i < debugCount; ++i) {
        const auto& entry = debugData->entries[i];
        if (entry.sampleIndex != selectedSampleIndex || entry.depth != 0u) {
            continue;
        }
        switch (entry.eventKind) {
            case kDebugEventDirectLightAuditMeta:
                outDump.available = true;
                outDump.pixelX = entry.pixelX;
                outDump.pixelY = entry.pixelY;
                outDump.sampleIndex = entry.sampleIndex;
                outDump.depth = entry.depth;
                outDump.materialIndex = entry.materialIndex;
                outDump.materialType = static_cast<uint32_t>(entry.scalars.x);
                outDump.primitiveIndex = static_cast<uint32_t>(entry.scalars.y);
                outDump.hitPosition = simd_make_float3(entry.throughput.x, entry.throughput.y, entry.throughput.z);
                outDump.shadingNormal = simd_make_float3(entry.aux.x, entry.aux.y, entry.aux.z);
                foundAny = true;
                break;
            case kDebugEventDirectLightAuditMaterial:
                outDump.baseColor = simd_make_float3(entry.throughput.x, entry.throughput.y, entry.throughput.z);
                outDump.roughness = entry.scalars.x;
                outDump.uv0 = simd_make_float2(entry.scalars.y, entry.scalars.z);
                outDump.baseColorLod = entry.scalars.w;
                outDump.baseColorTextureIndex = static_cast<uint32_t>(entry.aux.x);
                break;
            case kDebugEventShadingNormal:
                if (!foundAny) {
                    outDump.pixelX = entry.pixelX;
                    outDump.pixelY = entry.pixelY;
                    outDump.sampleIndex = entry.sampleIndex;
                    outDump.depth = entry.depth;
                    outDump.materialIndex = entry.materialIndex;
                }
                outDump.shadingNormal = simd_make_float3(entry.aux.x, entry.aux.y, entry.aux.z);
                break;
            case kDebugEventBsdfRng:
                outDump.wo = simd_make_float3(entry.aux.x, entry.aux.y, entry.aux.z);
                break;
            case kDebugEventDirectLightAuditEval: {
                uint32_t lightIndex = static_cast<uint32_t>(entry.scalars.x);
                const bool isEmissive = entry.mediumEvent == kDirectLightAuditKindEmissivePrimitive;
                auto& dst = entries[makeAuditKey(lightIndex, entry.mediumEvent)];
                dst.value.lightIndex = lightIndex;
                dst.value.rectIndex = isEmissive ? 0xFFFFFFFFu : lightIndex;
                dst.value.distance = entry.scalars.y;
                dst.value.nDotL = entry.scalars.z;
                dst.value.lightDirection = simd_make_float3(entry.throughput.x, entry.throughput.y, entry.throughput.z);
                dst.value.bsdfValue = simd_make_float3(entry.aux.x, entry.aux.y, entry.aux.z);
                dst.sawEval = true;
                break;
            }
            case kDebugEventDirectLightAuditContrib: {
                uint32_t lightIndex = static_cast<uint32_t>(entry.scalars.x);
                const bool isEmissive = entry.mediumEvent == kDirectLightAuditKindEmissivePrimitive;
                auto& dst = entries[makeAuditKey(lightIndex, entry.mediumEvent)];
                dst.value.lightIndex = lightIndex;
                dst.value.rectIndex = isEmissive ? 0xFFFFFFFFu : lightIndex;
                dst.value.visible = entry.scalars.y > 0.5f;
                dst.value.misWeight = entry.scalars.z;
                dst.value.lightPdf = entry.scalars.w;
                dst.value.bsdfPdf = entry.aux.x;
                dst.value.contribution = simd_make_float3(entry.throughput.x, entry.throughput.y, entry.throughput.z);
                dst.value.shadowPath.finalVisible = dst.value.visible;
                dst.sawContrib = true;
                break;
            }
            case kDebugEventDirectLightAuditShadowRaw: {
                uint32_t lightIndex = static_cast<uint32_t>(entry.scalars.x);
                const bool isEmissive = entry.mediumEvent == kDirectLightAuditKindEmissivePrimitive;
                auto& dst = entries[makeAuditKey(lightIndex, entry.mediumEvent)];
                dst.value.lightIndex = lightIndex;
                dst.value.rectIndex = isEmissive ? 0xFFFFFFFFu : lightIndex;
                dst.value.shadowPath.shadowTMin = entry.throughput.x;
                dst.value.shadowPath.shadowTMax = entry.throughput.y;
                dst.value.shadowPath.hwRawDistance = entry.throughput.z;
                dst.value.shadowPath.hwRawResultAvailable = entry.scalars.y > 0.5f;
                dst.value.shadowPath.hwRawResultType = static_cast<uint32_t>(entry.scalars.z);
                dst.value.shadowPath.hwTraceUsed = entry.scalars.w > 0.5f;
                dst.value.shadowPath.hwRawInstanceIndex = static_cast<uint32_t>(entry.aux.x);
                dst.value.shadowPath.hwRawPrimitiveIndex = static_cast<uint32_t>(entry.aux.y);
                dst.value.shadowPath.lightTwoSided = entry.aux.z > 0.5f;
                break;
            }
            case kDebugEventDirectLightAuditShadowMapped: {
                uint32_t lightIndex = static_cast<uint32_t>(entry.scalars.x);
                const bool isEmissive = entry.mediumEvent == kDirectLightAuditKindEmissivePrimitive;
                auto& dst = entries[makeAuditKey(lightIndex, entry.mediumEvent)];
                dst.value.lightIndex = lightIndex;
                dst.value.rectIndex = isEmissive ? 0xFFFFFFFFu : lightIndex;
                dst.value.shadowPath.hwResolvedDistance = entry.throughput.x;
                dst.value.shadowPath.rayOriginEpsilon = entry.throughput.y;
                dst.value.shadowPath.hardwareOcclusionEpsilon = entry.throughput.z;
                dst.value.shadowPath.hwResolvedHit = entry.scalars.y > 0.5f;
                dst.value.shadowPath.hwResolvedMeshIndex = static_cast<uint32_t>(entry.scalars.z);
                dst.value.shadowPath.hwResolvedPrimitiveIndex = static_cast<uint32_t>(entry.scalars.w);
                dst.value.shadowPath.hwResolvedMaterialIndex = static_cast<uint32_t>(entry.aux.x);
                dst.value.shadowPath.hwResolvedPrimitiveType = static_cast<uint32_t>(entry.aux.y);
                dst.value.shadowPath.hwResolvedFrontFace = entry.aux.z > 0.5f;
                break;
            }
            case kDebugEventDirectLightAuditShadowEmbeddedSw: {
                uint32_t lightIndex = static_cast<uint32_t>(entry.scalars.x);
                const bool isEmissive = entry.mediumEvent == kDirectLightAuditKindEmissivePrimitive;
                auto& dst = entries[makeAuditKey(lightIndex, entry.mediumEvent)];
                dst.value.lightIndex = lightIndex;
                dst.value.rectIndex = isEmissive ? 0xFFFFFFFFu : lightIndex;
                dst.value.shadowPath.embeddedSwDistance = entry.throughput.x;
                dst.value.shadowPath.embeddedSwHit = entry.throughput.y > 0.5f;
                dst.value.shadowPath.softwareBvhType = static_cast<uint32_t>(entry.throughput.z);
                dst.value.shadowPath.embeddedSwConsulted = entry.scalars.y > 0.5f;
                dst.value.shadowPath.embeddedSwInstanceIndex = static_cast<uint32_t>(entry.scalars.z);
                dst.value.shadowPath.embeddedSwMeshIndex = static_cast<uint32_t>(entry.scalars.w);
                dst.value.shadowPath.embeddedSwMaterialIndex = static_cast<uint32_t>(entry.aux.x);
                dst.value.shadowPath.embeddedSwPrimitiveIndex = static_cast<uint32_t>(entry.aux.y);
                dst.value.shadowPath.embeddedSwPrimitiveType = static_cast<uint32_t>(entry.aux.z);
                break;
            }
            case kDebugEventDirectLightAuditShadowBinding: {
                uint32_t lightIndex = static_cast<uint32_t>(entry.scalars.x);
                const bool isEmissive = entry.mediumEvent == kDirectLightAuditKindEmissivePrimitive;
                auto& dst = entries[makeAuditKey(lightIndex, entry.mediumEvent)];
                dst.value.lightIndex = lightIndex;
                dst.value.rectIndex = isEmissive ? 0xFFFFFFFFu : lightIndex;
                dst.value.shadowPath.softwareTlasAvailable = entry.scalars.y > 0.5f;
                dst.value.shadowPath.softwareBlasAvailable = entry.scalars.z > 0.5f;
                dst.value.shadowPath.softwareInstanceInfoAvailable = entry.scalars.w > 0.5f;
                dst.value.shadowPath.swBlasRootNodeOffset = static_cast<uint32_t>(entry.throughput.x);
                dst.value.shadowPath.swBlasPrimIndexOffset = static_cast<uint32_t>(entry.throughput.y);
                dst.value.shadowPath.swTriangleBaseOffset = static_cast<uint32_t>(entry.throughput.z);
                dst.value.shadowPath.embeddedSwHitPosition =
                    simd_make_float3(entry.aux.x, entry.aux.y, entry.aux.z);
                break;
            }
            case kDebugEventDirectLightAuditShadowReject: {
                uint32_t lightIndex = static_cast<uint32_t>(entry.scalars.x);
                const bool isEmissive = entry.mediumEvent == kDirectLightAuditKindEmissivePrimitive;
                auto& dst = entries[makeAuditKey(lightIndex, entry.mediumEvent)];
                dst.value.lightIndex = lightIndex;
                dst.value.rectIndex = isEmissive ? 0xFFFFFFFFu : lightIndex;
                dst.value.shadowPath.rejectionMask = static_cast<uint32_t>(entry.throughput.x);
                dst.value.shadowPath.finalVisible = entry.throughput.y > 0.5f;
                dst.value.shadowPath.embeddedSwFrontFace = entry.throughput.z > 0.5f;
                break;
            }
            case kDebugEventRisAuditState:
                outDump.ris.available = true;
                outDump.ris.wSum = entry.throughput.x;
                outDump.ris.W = entry.throughput.y;
                outDump.ris.winnerPdf = entry.throughput.z;
                outDump.ris.winnerPrimitiveIndex = static_cast<uint32_t>(entry.scalars.x);
                outDump.ris.M = static_cast<uint32_t>(entry.scalars.y);
                outDump.ris.valid = entry.scalars.z > 0.5f;
                outDump.ris.visible = entry.scalars.w > 0.5f;
                outDump.ris.winnerDistance = entry.aux.x;
                outDump.ris.winnerNDotL = entry.aux.y;
                outDump.ris.winnerBsdfPdf = entry.aux.z;
                break;
            case kDebugEventRisAuditWinnerA:
                outDump.ris.available = true;
                outDump.ris.winnerPosition =
                    simd_make_float3(entry.throughput.x, entry.throughput.y, entry.throughput.z);
                outDump.ris.winnerBary = simd_make_float2(entry.scalars.x, entry.scalars.y);
                outDump.ris.winnerNormal = simd_make_float3(entry.aux.x, entry.aux.y, entry.aux.z);
                break;
            case kDebugEventRisAuditWinnerB:
                outDump.ris.available = true;
                outDump.ris.winnerEmission =
                    simd_make_float3(entry.throughput.x, entry.throughput.y, entry.throughput.z);
                outDump.ris.winnerBsdfValue = simd_make_float3(entry.aux.x, entry.aux.y, entry.aux.z);
                break;
            case kDebugEventRisAuditWinnerC:
                outDump.ris.available = true;
                outDump.ris.winnerContribution =
                    simd_make_float3(entry.throughput.x, entry.throughput.y, entry.throughput.z);
                break;
            case kDebugEventSpatialReuseAudit:
                outDump.ris.available = true;
                outDump.ris.spatialNeighborTarget = static_cast<uint32_t>(entry.throughput.x);
                outDump.ris.spatialNeighborsConsidered = static_cast<uint32_t>(entry.throughput.y);
                outDump.ris.spatialNeighborsAccepted = static_cast<uint32_t>(entry.throughput.z);
                outDump.ris.spatialRejectedDepth = static_cast<uint32_t>(entry.scalars.x);
                outDump.ris.spatialRejectedNormal = static_cast<uint32_t>(entry.scalars.y);
                outDump.ris.spatialRejectedInvalid = static_cast<uint32_t>(entry.scalars.z);
                outDump.ris.spatialReuseAttempted = entry.scalars.w > 0.5f;
                outDump.ris.spatialWinnerOffsetX = static_cast<int32_t>(entry.aux.x);
                outDump.ris.spatialWinnerOffsetY = static_cast<int32_t>(entry.aux.y);
                break;
            case kDebugEventSpatialReuseWeights:
                outDump.ris.available = true;
                outDump.ris.spatialLastMisWeight = entry.throughput.x;
                outDump.ris.spatialLastMergeWeight = entry.throughput.y;
                break;
            case kDebugEventTemporalReuseAudit:
                outDump.ris.available = true;
                outDump.ris.temporalPreviousAvailable = entry.throughput.x > 0.5f;
                outDump.ris.temporalReuseAccepted = entry.throughput.y > 0.5f;
                outDump.ris.temporalPreviousPrimitiveIndex = static_cast<uint32_t>(entry.throughput.z);
                outDump.ris.temporalRejectedDepth = static_cast<uint32_t>(entry.scalars.x);
                outDump.ris.temporalRejectedNormal = static_cast<uint32_t>(entry.scalars.y);
                outDump.ris.temporalRejectedInvalid = static_cast<uint32_t>(entry.scalars.z);
                outDump.ris.temporalReuseAttempted = entry.scalars.w > 0.5f;
                break;
            case kDebugEventTemporalReuseWeights:
                outDump.ris.available = true;
                outDump.ris.temporalLastMisWeight = entry.throughput.x;
                outDump.ris.temporalLastMergeWeight = entry.throughput.y;
                break;
            case kDebugEventWorldReuseAudit:
                outDump.ris.available = true;
                outDump.ris.worldCandidatesConsidered = static_cast<uint32_t>(entry.throughput.x);
                outDump.ris.worldCandidatesAccepted = static_cast<uint32_t>(entry.throughput.y);
                outDump.ris.worldCellHash = static_cast<uint32_t>(entry.throughput.z);
                outDump.ris.worldRejectedDepth = static_cast<uint32_t>(entry.scalars.x);
                outDump.ris.worldRejectedNormal = static_cast<uint32_t>(entry.scalars.y);
                outDump.ris.worldRejectedInvalid = static_cast<uint32_t>(entry.scalars.z);
                outDump.ris.worldReuseAttempted = entry.scalars.w > 0.5f;
                outDump.ris.worldCellX = static_cast<int32_t>(entry.aux.x);
                outDump.ris.worldCellY = static_cast<int32_t>(entry.aux.y);
                outDump.ris.worldCellZ = static_cast<int32_t>(entry.aux.z);
                break;
            case kDebugEventWorldReuseWeights:
                outDump.ris.available = true;
                outDump.ris.worldLastMisWeight = entry.throughput.x;
                outDump.ris.worldLastMergeWeight = entry.throughput.y;
                outDump.ris.worldRejectedCell = static_cast<uint32_t>(entry.throughput.z);
                outDump.ris.worldCandidatePrimitiveIndex = static_cast<uint32_t>(entry.scalars.x);
                outDump.ris.worldReuseAccepted = entry.scalars.y > 0.5f;
                outDump.ris.worldQueryRadius = static_cast<uint32_t>(entry.scalars.z);
                outDump.ris.worldCandidateTarget = static_cast<uint32_t>(entry.scalars.w);
                break;
            case kDebugEventCacheReuseAudit:
                outDump.ris.available = true;
                outDump.ris.cacheCandidatesConsidered = static_cast<uint32_t>(entry.throughput.x);
                outDump.ris.cacheCandidatesAccepted = static_cast<uint32_t>(entry.throughput.y);
                outDump.ris.cacheCellHash = static_cast<uint32_t>(entry.throughput.z);
                outDump.ris.cacheRejectedDepth = static_cast<uint32_t>(entry.scalars.x);
                outDump.ris.cacheRejectedNormal = static_cast<uint32_t>(entry.scalars.y);
                outDump.ris.cacheRejectedInvalid = static_cast<uint32_t>(entry.scalars.z);
                outDump.ris.cacheReuseAttempted = entry.scalars.w > 0.5f;
                outDump.ris.cacheCellX = static_cast<int32_t>(entry.aux.x);
                outDump.ris.cacheCellY = static_cast<int32_t>(entry.aux.y);
                outDump.ris.cacheCellZ = static_cast<int32_t>(entry.aux.z);
                break;
            case kDebugEventCacheReuseWeights:
                outDump.ris.available = true;
                outDump.ris.cacheLastMisWeight = entry.throughput.x;
                outDump.ris.cacheLastMergeWeight = entry.throughput.y;
                outDump.ris.cacheRejectedCell = static_cast<uint32_t>(entry.throughput.z);
                outDump.ris.cacheCandidatePrimitiveIndex = static_cast<uint32_t>(entry.scalars.x);
                outDump.ris.cacheReuseAccepted = entry.scalars.y > 0.5f;
                outDump.ris.cacheStateAvailable = entry.scalars.z > 0.5f;
                outDump.ris.cacheCandidateTarget = static_cast<uint32_t>(entry.scalars.w);
                outDump.ris.cacheFallbackUsed = entry.aux.x > 0.5f;
                outDump.ris.cacheSourceFrameIndex = static_cast<uint32_t>(entry.aux.y);
                outDump.ris.cacheEntriesAvailable = static_cast<uint32_t>(entry.aux.z);
                break;
            default:
                break;
        }
    }

    if (!foundAny) {
        outDump = HeadlessRenderOutput::DirectLightAuditDump{};
        return false;
    }

    std::vector<uint32_t> ordered;
    ordered.reserve(entries.size());
    for (const auto& [lightIndex, _] : entries) {
        ordered.push_back(lightIndex);
    }
    std::sort(ordered.begin(), ordered.end());
    outDump.entries.clear();
    outDump.entries.reserve(ordered.size());
    for (uint32_t lightIndex : ordered) {
        const auto& tmp = entries[lightIndex];
        if (tmp.sawEval || tmp.sawContrib) {
            outDump.entries.push_back(tmp.value);
        }
    }
    return true;
#endif
}

bool MetalRenderer::captureMaterialMrDebugDump(HeadlessRenderOutput::MaterialMrDebugDump& outDump) const {
    if (!m_impl) {
        outDump = HeadlessRenderOutput::MaterialMrDebugDump{};
        return false;
    }
    return m_impl->captureMaterialMrDebugDump(outDump);
}

bool MetalRenderer::captureDirectLightAuditDump(HeadlessRenderOutput::DirectLightAuditDump& outDump) const {
    if (!m_impl) {
        outDump = HeadlessRenderOutput::DirectLightAuditDump{};
        return false;
    }
    return m_impl->captureDirectLightAuditDump(outDump);
}
