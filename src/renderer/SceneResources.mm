#import "renderer/SceneResources.h"
#import "renderer/MetalContext.h"
#import "renderer/SceneAccel.h"
#import "renderer/EnvImportanceSampler.h"
#import "renderer/ImageWriter.h"
#include "shared/Ktx2Texture.h"

#import <AppKit/AppKit.h>
#import <CommonCrypto/CommonDigest.h>
#import <ImageIO/ImageIO.h>
#import <MetalKit/MetalKit.h>
#import <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using PathTracerShaderTypes::EnvironmentAliasEntry;
using PathTracerShaderTypes::LightPrimitive;
using PathTracerShaderTypes::MaterialData;
using PathTracerShaderTypes::MaterialTextureInfo;
using PathTracerShaderTypes::MaterialType;
using PathTracerShaderTypes::RectData;
using PathTracerShaderTypes::SphereData;
using PathTracerShaderTypes::kMaxMaterials;
using PathTracerShaderTypes::kMaxMaterialSamplers;
using PathTracerShaderTypes::kMaxMaterialTextures;
using PathTracerShaderTypes::kMaxRectangles;
using PathTracerShaderTypes::kMaxSpheres;

namespace PathTracer {

namespace {

inline float SrgbToLinear(float value) {
    if (value <= 0.04045f) {
        return value / 12.92f;
    }
    return powf((value + 0.055f) / 1.055f, 2.4f);
}

constexpr uint32_t kInvalidTextureIndex = 0xFFFFFFFFu;
constexpr int32_t kGltfFilterNearest = 9728;
constexpr int32_t kGltfFilterLinear = 9729;
constexpr int32_t kGltfFilterNearestMipmapNearest = 9984;
constexpr int32_t kGltfFilterLinearMipmapNearest = 9985;
constexpr int32_t kGltfFilterNearestMipmapLinear = 9986;
constexpr int32_t kGltfFilterLinearMipmapLinear = 9987;
constexpr int32_t kGltfWrapClampToEdge = 33071;
constexpr int32_t kGltfWrapMirroredRepeat = 33648;
constexpr int32_t kGltfWrapRepeat = 10497;
constexpr NSUInteger kDefaultMaterialAnisotropy = 8u;
constexpr NSUInteger kIncrementalBlasBatchSize = 1u;

bool SceneLoadDiagnosticsEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("PT_SCENE_LOAD_DIAGNOSTICS");
        if (!value || !*value) {
            return false;
        }
        return std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0 &&
               std::strcmp(value, "no") != 0 &&
               std::strcmp(value, "NO") != 0;
    }();
    return enabled;
}

bool SceneLoadDiagnosticsVerboseEnabled() {
    static const bool enabled = []() {
        const char* value = std::getenv("PT_SCENE_LOAD_DIAGNOSTICS_VERBOSE");
        if (!value || !*value) {
            return false;
        }
        return std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0 &&
               std::strcmp(value, "no") != 0 &&
               std::strcmp(value, "NO") != 0;
    }();
    return SceneLoadDiagnosticsEnabled() && enabled;
}

bool KeepTriangleBufferForHardware() {
    static const bool keep = []() {
        const char* value = std::getenv("PT_HWRT_KEEP_TRIANGLE_BUFFER");
        if (!value || !*value) {
            return false;
        }
        return std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "FALSE") != 0 &&
               std::strcmp(value, "no") != 0 &&
               std::strcmp(value, "NO") != 0;
    }();
    return keep;
}

double BytesToMiB(uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

inline simd::float3 TransformPoint(const simd::float4x4& transform, const simd::float3& point) {
    const simd::float4 transformed = simd_mul(transform, simd_make_float4(point, 1.0f));
    return simd_make_float3(transformed.x, transformed.y, transformed.z);
}

inline float RelativeLuminance(const simd::float3& rgb) {
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

struct MeshAggregateStats {
    uint32_t meshCount = 0u;
    uint64_t vertexCount = 0u;
    uint64_t indexCount = 0u;
    uint64_t triangleCount = 0u;
    uint64_t vertexBytes = 0u;
    uint64_t indexBytes = 0u;
    uint64_t meshVertexBufferBytes = 0u;
    uint64_t meshIndexBufferBytes = 0u;
    uint32_t meshGpuBufferCount = 0u;
};

MeshAggregateStats CollectMeshAggregateStats(const std::vector<PathTracer::SceneResources::Mesh>& meshes) {
    MeshAggregateStats stats{};
    stats.meshCount = static_cast<uint32_t>(meshes.size());
    std::unordered_set<uintptr_t> seenVertexBuffers;
    std::unordered_set<uintptr_t> seenIndexBuffers;
    for (const auto& mesh : meshes) {
        stats.vertexCount += static_cast<uint64_t>(mesh.vertices.size());
        stats.indexCount += static_cast<uint64_t>(mesh.indices.size());
        stats.vertexBytes += static_cast<uint64_t>(mesh.vertices.size()) *
                             static_cast<uint64_t>(sizeof(PathTracer::SceneResources::MeshVertex));
        stats.indexBytes += static_cast<uint64_t>(mesh.indices.size()) * sizeof(uint32_t);
        if (mesh.vertexBuffer) {
            const uintptr_t key = reinterpret_cast<uintptr_t>(mesh.vertexBuffer);
            if (seenVertexBuffers.insert(key).second) {
                stats.meshVertexBufferBytes += static_cast<uint64_t>(mesh.vertexBuffer.length);
                ++stats.meshGpuBufferCount;
            }
        }
        if (mesh.indexBuffer) {
            const uintptr_t key = reinterpret_cast<uintptr_t>(mesh.indexBuffer);
            if (seenIndexBuffers.insert(key).second) {
                stats.meshIndexBufferBytes += static_cast<uint64_t>(mesh.indexBuffer.length);
                ++stats.meshGpuBufferCount;
            }
        }
    }
    stats.triangleCount = stats.indexCount / 3u;
    return stats;
}

inline simd::float4 MakeTextureTransformRow0Identity() {
    return simd_make_float4(1.0f, 0.0f, 0.0f, 0.0f);
}

inline simd::float4 MakeTextureTransformRow1Identity() {
    return simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f);
}

inline bool TextureTransformRowsAreUsable(const simd::float4& row0, const simd::float4& row1) {
    if (!std::isfinite(row0.x) || !std::isfinite(row0.y) || !std::isfinite(row0.z) ||
        !std::isfinite(row1.x) || !std::isfinite(row1.y) || !std::isfinite(row1.z)) {
        return false;
    }
    const float linearSum = std::fabs(row0.x) + std::fabs(row0.y) + std::fabs(row1.x) + std::fabs(row1.y);
    return linearSum > 1.0e-8f;
}

inline void SetMaterialTextureTransformIdentity(MaterialData& material) {
    material.textureUvSet0 = simd_make_uint4(0u, 0u, 0u, 0u);
    material.textureUvSet1 = simd_make_uint4(0u, 0u, 0u, 0u);
    material.textureTransform0 = MakeTextureTransformRow0Identity();
    material.textureTransform1 = MakeTextureTransformRow1Identity();
    material.textureTransform2 = MakeTextureTransformRow0Identity();
    material.textureTransform3 = MakeTextureTransformRow1Identity();
    material.textureTransform4 = MakeTextureTransformRow0Identity();
    material.textureTransform5 = MakeTextureTransformRow1Identity();
    material.textureTransform6 = MakeTextureTransformRow0Identity();
    material.textureTransform7 = MakeTextureTransformRow1Identity();
    material.textureTransform8 = MakeTextureTransformRow0Identity();
    material.textureTransform9 = MakeTextureTransformRow1Identity();
    material.textureTransform10 = MakeTextureTransformRow0Identity();
    material.textureTransform11 = MakeTextureTransformRow1Identity();
}

inline void SanitizeMaterialTextureMapping(MaterialData& material) {
    material.textureUvSet0.x = std::min<uint32_t>(material.textureUvSet0.x, 1u);
    material.textureUvSet0.y = std::min<uint32_t>(material.textureUvSet0.y, 1u);
    material.textureUvSet0.z = std::min<uint32_t>(material.textureUvSet0.z, 1u);
    material.textureUvSet0.w = std::min<uint32_t>(material.textureUvSet0.w, 1u);
    material.textureUvSet1.x = std::min<uint32_t>(material.textureUvSet1.x, 1u);
    material.textureUvSet1.y = std::min<uint32_t>(material.textureUvSet1.y, 1u);

    if (!TextureTransformRowsAreUsable(material.textureTransform0, material.textureTransform1)) {
        material.textureTransform0 = MakeTextureTransformRow0Identity();
        material.textureTransform1 = MakeTextureTransformRow1Identity();
    }
    if (!TextureTransformRowsAreUsable(material.textureTransform2, material.textureTransform3)) {
        material.textureTransform2 = MakeTextureTransformRow0Identity();
        material.textureTransform3 = MakeTextureTransformRow1Identity();
    }
    if (!TextureTransformRowsAreUsable(material.textureTransform4, material.textureTransform5)) {
        material.textureTransform4 = MakeTextureTransformRow0Identity();
        material.textureTransform5 = MakeTextureTransformRow1Identity();
    }
    if (!TextureTransformRowsAreUsable(material.textureTransform6, material.textureTransform7)) {
        material.textureTransform6 = MakeTextureTransformRow0Identity();
        material.textureTransform7 = MakeTextureTransformRow1Identity();
    }
    if (!TextureTransformRowsAreUsable(material.textureTransform8, material.textureTransform9)) {
        material.textureTransform8 = MakeTextureTransformRow0Identity();
        material.textureTransform9 = MakeTextureTransformRow1Identity();
    }
    if (!TextureTransformRowsAreUsable(material.textureTransform10, material.textureTransform11)) {
        material.textureTransform10 = MakeTextureTransformRow0Identity();
        material.textureTransform11 = MakeTextureTransformRow1Identity();
    }
}

MaterialTextureSamplerDesc EffectiveSamplerDesc(const MaterialTextureSamplerDesc* desc) {
    MaterialTextureSamplerDesc effective{};
    if (desc) {
        effective = *desc;
    }
    if (effective.magFilter < 0) {
        effective.magFilter = kGltfFilterLinear;
    }
    if (effective.minFilter < 0) {
        effective.minFilter = kGltfFilterLinearMipmapLinear;
    }
    if (effective.wrapS != kGltfWrapClampToEdge &&
        effective.wrapS != kGltfWrapMirroredRepeat &&
        effective.wrapS != kGltfWrapRepeat) {
        effective.wrapS = kGltfWrapRepeat;
    }
    if (effective.wrapT != kGltfWrapClampToEdge &&
        effective.wrapT != kGltfWrapMirroredRepeat &&
        effective.wrapT != kGltfWrapRepeat) {
        effective.wrapT = kGltfWrapRepeat;
    }
    return effective;
}

uint64_t SamplerDescKey(const MaterialTextureSamplerDesc& desc) {
    uint64_t key = 0u;
    key |= static_cast<uint64_t>(static_cast<uint16_t>(desc.magFilter));
    key |= static_cast<uint64_t>(static_cast<uint16_t>(desc.minFilter)) << 16u;
    key |= static_cast<uint64_t>(static_cast<uint16_t>(desc.wrapS)) << 32u;
    key |= static_cast<uint64_t>(static_cast<uint16_t>(desc.wrapT)) << 48u;
    return key;
}

std::string SamplerKeySuffix(const MaterialTextureSamplerDesc* desc) {
    MaterialTextureSamplerDesc effective = EffectiveSamplerDesc(desc);
    return "|samp:" + std::to_string(effective.magFilter) + ":" +
           std::to_string(effective.minFilter) + ":" +
           std::to_string(effective.wrapS) + ":" +
           std::to_string(effective.wrapT);
}

std::string TextureSemanticKeySuffix(MaterialTextureSemantic semantic) {
    return "|sem:" + std::to_string(static_cast<uint32_t>(semantic));
}

const char* TextureSemanticLabel(MaterialTextureSemantic semantic) {
    switch (semantic) {
        case MaterialTextureSemantic::BaseColor:
            return "baseColor";
        case MaterialTextureSemantic::Orm:
            return "orm";
        case MaterialTextureSemantic::Normal:
            return "normal";
        case MaterialTextureSemantic::Occlusion:
            return "occlusion";
        case MaterialTextureSemantic::Emissive:
            return "emissive";
        case MaterialTextureSemantic::Transmission:
            return "transmission";
        case MaterialTextureSemantic::Generic:
        default:
            return "generic";
    }
}

const char* PixelFormatLabel(MTLPixelFormat format) {
    switch (format) {
        case MTLPixelFormatRGBA32Float:
            return "RGBA32Float";
        case MTLPixelFormatRGBA16Float:
            return "RGBA16Float";
        case MTLPixelFormatRGBA8Unorm:
            return "RGBA8Unorm";
        case MTLPixelFormatRGBA8Unorm_sRGB:
            return "RGBA8Unorm_sRGB";
        case MTLPixelFormatBGRA8Unorm:
            return "BGRA8Unorm";
        case MTLPixelFormatBGRA8Unorm_sRGB:
            return "BGRA8Unorm_sRGB";
        case MTLPixelFormatBC1_RGBA:
            return "BC1_RGBA";
        case MTLPixelFormatBC1_RGBA_sRGB:
            return "BC1_RGBA_sRGB";
        case MTLPixelFormatBC3_RGBA:
            return "BC3_RGBA";
        case MTLPixelFormatBC3_RGBA_sRGB:
            return "BC3_RGBA_sRGB";
        case MTLPixelFormatBC5_RGUnorm:
            return "BC5_RGUnorm";
        default:
            return "Other";
    }
}

std::string ToLowerAscii(std::string value) {
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool HasExtensionCaseInsensitive(const std::string& path, const char* extension) {
    const std::string lowerPath = ToLowerAscii(path);
    const std::string lowerExtension = ToLowerAscii(extension);
    if (lowerPath.size() < lowerExtension.size()) {
        return false;
    }
    return lowerPath.compare(lowerPath.size() - lowerExtension.size(), lowerExtension.size(), lowerExtension) == 0;
}

bool IsKtx2Data(const uint8_t* data, size_t size) {
    return data && size >= sizeof(Ktx2::kIdentifier) &&
           std::memcmp(data, Ktx2::kIdentifier, sizeof(Ktx2::kIdentifier)) == 0;
}

void PumpMainThreadEventsBriefly() {
    if (![NSThread isMainThread]) {
        [NSThread sleepForTimeInterval:0.001];
        return;
    }
    NSApplication* app = [NSApplication sharedApplication];
    if (!app) {
        [NSThread sleepForTimeInterval:0.001];
        return;
    }
    NSDate* deadline = [NSDate dateWithTimeIntervalSinceNow:0.001];
    NSArray<NSString*>* modes = @[
        NSDefaultRunLoopMode,
        NSEventTrackingRunLoopMode,
        NSModalPanelRunLoopMode
    ];
    for (NSString* mode in modes) {
        while (true) {
            NSEvent* event = [app nextEventMatchingMask:NSEventMaskAny
                                              untilDate:deadline
                                                 inMode:mode
                                                dequeue:YES];
            if (!event) {
                break;
            }
            [app sendEvent:event];
        }
    }
}

void WaitForCommandBufferResponsive(id<MTLCommandBuffer> commandBuffer) {
    if (!commandBuffer) {
        return;
    }
    while (true) {
        const MTLCommandBufferStatus status = commandBuffer.status;
        if (status == MTLCommandBufferStatusCompleted ||
            status == MTLCommandBufferStatusError) {
            break;
        }
        PumpMainThreadEventsBriefly();
    }
}

inline void PumpEventsEvery(size_t counter, size_t interval = 16384u) {
    if (counter > 0u && (counter % interval) == 0u) {
        PumpMainThreadEventsBriefly();
    }
}

id<MTLTexture> PromoteTextureToPrivate(id<MTLDevice> device,
                                       id<MTLCommandQueue> commandQueue,
                                       id<MTLTexture> source);

struct ManifestTextureSourceCache {
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> manifestToTextureSources;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> manifestToTextureHashes;
    std::unordered_set<std::string> failedManifests;
};

ManifestTextureSourceCache& TextureSourceCache() {
    static ManifestTextureSourceCache cache;
    return cache;
}

std::string RelativePathForManifest(const std::filesystem::path& path,
                                    const std::filesystem::path& manifestPath) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path rel = fs::relative(path, manifestPath.parent_path(), ec);
    if (!ec) {
        return rel.generic_string();
    }
    return path.filename().generic_string();
}

bool LoadManifestTextureMetadata(const std::string& manifestPath) {
    auto& cache = TextureSourceCache();
    auto existingSources = cache.manifestToTextureSources.find(manifestPath);
    auto existingHashes = cache.manifestToTextureHashes.find(manifestPath);
    if (existingSources != cache.manifestToTextureSources.end() &&
        existingHashes != cache.manifestToTextureHashes.end()) {
        return true;
    }
    if (cache.failedManifests.contains(manifestPath)) {
        return false;
    }

    NSData* manifestData =
        [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:manifestPath.c_str()]];
    if (!manifestData) {
        cache.failedManifests.insert(manifestPath);
        return false;
    }

    NSError* jsonError = nil;
    id root = [NSJSONSerialization JSONObjectWithData:manifestData options:0 error:&jsonError];
    if (![root isKindOfClass:[NSDictionary class]]) {
        cache.failedManifests.insert(manifestPath);
        return false;
    }

    NSDictionary* dict = static_cast<NSDictionary*>(root);
    NSArray* textures = dict[@"textures"];
    if (![textures isKindOfClass:[NSArray class]]) {
        cache.failedManifests.insert(manifestPath);
        return false;
    }

    std::unordered_map<std::string, std::string> mapping;
    std::unordered_map<std::string, std::string> hashes;
    for (id entry : textures) {
        if (![entry isKindOfClass:[NSDictionary class]]) {
            continue;
        }
        NSDictionary* textureEntry = static_cast<NSDictionary*>(entry);
        NSString* sourcePath = textureEntry[@"source_path"];
        NSString* sourceFormat = textureEntry[@"source_format"];
        NSString* outputPath = textureEntry[@"output_path"];
        NSString* outputSha256 = textureEntry[@"output_sha256"];
        NSNumber* fellBackToCopy = textureEntry[@"fell_back_to_copy"];
        if (![outputPath isKindOfClass:[NSString class]]) {
            continue;
        }
        if ([outputSha256 isKindOfClass:[NSString class]]) {
            hashes.emplace(outputPath.UTF8String, outputSha256.UTF8String);
        }
        if (![sourcePath isKindOfClass:[NSString class]] ||
            ![sourceFormat isKindOfClass:[NSString class]]) {
            continue;
        }
        const std::string sourceFormatLower = ToLowerAscii(sourceFormat.UTF8String);
        if (sourceFormatLower != "dds") {
            continue;
        }
        if ([fellBackToCopy isKindOfClass:[NSNumber class]] && fellBackToCopy.boolValue) {
            continue;
        }
        mapping.emplace(outputPath.UTF8String, sourcePath.UTF8String);
    }

    cache.manifestToTextureSources[manifestPath] = std::move(mapping);
    cache.manifestToTextureHashes[manifestPath] = std::move(hashes);
    return true;
}

const std::unordered_map<std::string, std::string>* LoadManifestTextureSources(const std::string& manifestPath) {
    auto& cache = TextureSourceCache();
    if (!LoadManifestTextureMetadata(manifestPath)) {
        return nullptr;
    }
    auto found = cache.manifestToTextureSources.find(manifestPath);
    return found != cache.manifestToTextureSources.end() ? &found->second : nullptr;
}

const std::unordered_map<std::string, std::string>* LoadManifestTextureOutputHashes(const std::string& manifestPath) {
    auto& cache = TextureSourceCache();
    if (!LoadManifestTextureMetadata(manifestPath)) {
        return nullptr;
    }
    auto found = cache.manifestToTextureHashes.find(manifestPath);
    return found != cache.manifestToTextureHashes.end() ? &found->second : nullptr;
}

std::string ResolveCompressedSourceTexturePath(const std::string& canonicalPath) {
    namespace fs = std::filesystem;
    if (!HasExtensionCaseInsensitive(canonicalPath, ".ktx2")) {
        return {};
    }

    auto canonicalize = [](const fs::path& path) -> std::string {
        std::error_code ec;
        fs::path canonical = fs::weakly_canonical(path, ec);
        if (!ec) {
            return canonical.string();
        }
        return path.lexically_normal().string();
    };

    const fs::path texturePath(canonicalPath);
    std::vector<fs::path> manifestCandidates;
    if (texturePath.has_parent_path()) {
        manifestCandidates.push_back(texturePath.parent_path() / "import_manifest.json");
        manifestCandidates.push_back(texturePath.parent_path().parent_path() / "import_manifest.json");
    }

    for (const fs::path& manifestCandidate : manifestCandidates) {
        std::error_code ec;
        if (!fs::exists(manifestCandidate, ec) || ec) {
            continue;
        }
        const std::string manifestCanonical = canonicalize(manifestCandidate);
        const auto* mapping = LoadManifestTextureSources(manifestCanonical);
        if (!mapping) {
            continue;
        }
        const std::string relativeTexturePath = RelativePathForManifest(texturePath, manifestCandidate);
        auto found = mapping->find(relativeTexturePath);
        if (found == mapping->end()) {
            continue;
        }
        const std::string sourceCanonical = canonicalize(fs::path(found->second));
        if (!sourceCanonical.empty() && HasExtensionCaseInsensitive(sourceCanonical, ".dds")) {
            return sourceCanonical;
        }
    }

    return {};
}

std::string ResolveTextureOutputHash(const std::string& canonicalPath) {
    namespace fs = std::filesystem;

    auto canonicalize = [](const fs::path& path) -> std::string {
        std::error_code ec;
        fs::path canonical = fs::weakly_canonical(path, ec);
        if (!ec) {
            return canonical.string();
        }
        return path.lexically_normal().string();
    };

    const fs::path texturePath(canonicalPath);
    std::vector<fs::path> manifestCandidates;
    if (texturePath.has_parent_path()) {
        manifestCandidates.push_back(texturePath.parent_path() / "import_manifest.json");
        manifestCandidates.push_back(texturePath.parent_path().parent_path() / "import_manifest.json");
    }

    for (const fs::path& manifestCandidate : manifestCandidates) {
        std::error_code ec;
        if (!fs::exists(manifestCandidate, ec) || ec) {
            continue;
        }
        const std::string manifestCanonical = canonicalize(manifestCandidate);
        const auto* hashes = LoadManifestTextureOutputHashes(manifestCanonical);
        if (!hashes) {
            continue;
        }
        const std::string relativeTexturePath = RelativePathForManifest(texturePath, manifestCandidate);
        auto found = hashes->find(relativeTexturePath);
        if (found != hashes->end()) {
            return found->second;
        }
    }

    return {};
}

uint32_t ReadU32LE(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8u) |
           (static_cast<uint32_t>(data[2]) << 16u) |
           (static_cast<uint32_t>(data[3]) << 24u);
}

struct DdsCompressedInfo {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t mipCount = 0;
    uint32_t blockBytes = 0;
    size_t dataOffset = 0;
    MTLPixelFormat pixelFormat = MTLPixelFormatInvalid;
};

bool ParseCompressedDdsHeader(const std::vector<uint8_t>& bytes,
                              bool srgb,
                              DdsCompressedInfo& outInfo,
                              std::string& errorMessage) {
    if (bytes.size() < 128u || std::memcmp(bytes.data(), "DDS ", 4) != 0) {
        errorMessage = "DDS header is invalid";
        return false;
    }

    outInfo.height = ReadU32LE(bytes.data() + 12u);
    outInfo.width = ReadU32LE(bytes.data() + 16u);
    outInfo.mipCount = std::max<uint32_t>(1u, ReadU32LE(bytes.data() + 28u));
    outInfo.dataOffset = 128u;

    const std::string fourcc(reinterpret_cast<const char*>(bytes.data() + 84u), 4u);
    if (fourcc == "DXT1") {
        outInfo.blockBytes = 8u;
        outInfo.pixelFormat = srgb ? MTLPixelFormatBC1_RGBA_sRGB : MTLPixelFormatBC1_RGBA;
    } else if (fourcc == "DXT5") {
        outInfo.blockBytes = 16u;
        outInfo.pixelFormat = srgb ? MTLPixelFormatBC3_RGBA_sRGB : MTLPixelFormatBC3_RGBA;
    } else if (fourcc == "ATI2") {
        outInfo.blockBytes = 16u;
        outInfo.pixelFormat = MTLPixelFormatBC5_RGUnorm;
    } else if (fourcc == "DX10") {
        if (bytes.size() < 148u) {
            errorMessage = "DDS DX10 header is truncated";
            return false;
        }
        outInfo.dataOffset = 148u;
        const uint32_t dxgiFormat = ReadU32LE(bytes.data() + 128u);
        switch (dxgiFormat) {
            case 71u:
                outInfo.blockBytes = 8u;
                outInfo.pixelFormat = srgb ? MTLPixelFormatBC1_RGBA_sRGB : MTLPixelFormatBC1_RGBA;
                break;
            case 72u:
                outInfo.blockBytes = 8u;
                outInfo.pixelFormat = MTLPixelFormatBC1_RGBA_sRGB;
                break;
            case 77u:
                outInfo.blockBytes = 16u;
                outInfo.pixelFormat = srgb ? MTLPixelFormatBC3_RGBA_sRGB : MTLPixelFormatBC3_RGBA;
                break;
            case 78u:
                outInfo.blockBytes = 16u;
                outInfo.pixelFormat = MTLPixelFormatBC3_RGBA_sRGB;
                break;
            case 83u:
            case 84u:
                outInfo.blockBytes = 16u;
                outInfo.pixelFormat = MTLPixelFormatBC5_RGUnorm;
                break;
            default:
                errorMessage = "DDS DXGI format is unsupported";
                return false;
        }
    } else {
        errorMessage = "DDS compression format is unsupported";
        return false;
    }

    if (outInfo.width == 0u || outInfo.height == 0u || outInfo.pixelFormat == MTLPixelFormatInvalid) {
        errorMessage = "DDS dimensions or pixel format are invalid";
        return false;
    }

    size_t offset = outInfo.dataOffset;
    uint32_t mipWidth = outInfo.width;
    uint32_t mipHeight = outInfo.height;
    for (uint32_t level = 0u; level < outInfo.mipCount; ++level) {
        const uint32_t blocksWide = std::max<uint32_t>(1u, (mipWidth + 3u) / 4u);
        const uint32_t blocksHigh = std::max<uint32_t>(1u, (mipHeight + 3u) / 4u);
        const size_t levelBytes =
            static_cast<size_t>(blocksWide) * static_cast<size_t>(blocksHigh) * outInfo.blockBytes;
        if (offset + levelBytes > bytes.size()) {
            errorMessage = "DDS mip chain is truncated";
            return false;
        }
        offset += levelBytes;
        mipWidth = std::max<uint32_t>(1u, mipWidth >> 1u);
        mipHeight = std::max<uint32_t>(1u, mipHeight >> 1u);
    }

    return true;
}

struct DecodedImageRgba8 {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> rgba;
};

bool DecodeImageSourceToRgba8(CGImageSourceRef source,
                              DecodedImageRgba8& outImage,
                              std::string& errorMessage) {
    if (!source) {
        errorMessage = "Material texture load failed: invalid image source";
        return false;
    }

    CGImageRef image = CGImageSourceCreateImageAtIndex(source, 0, nullptr);
    if (!image) {
        errorMessage = "Material texture load failed: unable to decode image";
        return false;
    }

    const size_t width = CGImageGetWidth(image);
    const size_t height = CGImageGetHeight(image);
    if (width == 0 || height == 0) {
        CGImageRelease(image);
        errorMessage = "Material texture load failed: decoded image has invalid size";
        return false;
    }

    outImage.width = static_cast<uint32_t>(width);
    outImage.height = static_cast<uint32_t>(height);
    outImage.rgba.assign(width * height * 4u, 0u);

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    if (!colorSpace) {
        CGImageRelease(image);
        errorMessage = "Material texture load failed: unable to create RGB color space";
        return false;
    }

    CGContextRef context =
        CGBitmapContextCreate(outImage.rgba.data(),
                              width,
                              height,
                              8,
                              width * 4u,
                              colorSpace,
                              static_cast<uint32_t>(kCGImageAlphaPremultipliedLast) |
                                  static_cast<uint32_t>(kCGBitmapByteOrder32Big));
    CGColorSpaceRelease(colorSpace);
    if (!context) {
        CGImageRelease(image);
        errorMessage = "Material texture load failed: unable to create bitmap context";
        return false;
    }

    CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
    CGContextRelease(context);
    CGImageRelease(image);

    const size_t pixelCount = width * height;
    for (size_t i = 0; i < pixelCount; ++i) {
        uint8_t* px = outImage.rgba.data() + i * 4u;
        const uint8_t alpha = px[3];
        if (alpha == 0u || alpha == 255u) {
            continue;
        }
        const float invAlpha = 255.0f / static_cast<float>(alpha);
        px[0] = static_cast<uint8_t>(std::clamp(std::lround(px[0] * invAlpha), 0l, 255l));
        px[1] = static_cast<uint8_t>(std::clamp(std::lround(px[1] * invAlpha), 0l, 255l));
        px[2] = static_cast<uint8_t>(std::clamp(std::lround(px[2] * invAlpha), 0l, 255l));
    }

    return true;
}

bool DecodeImageFileToRgba8(const std::string& path,
                            DecodedImageRgba8& outImage,
                            std::string& errorMessage) {
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    if (!url) {
        errorMessage = "Material texture load failed: invalid path";
        return false;
    }
    CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
    if (!source) {
        errorMessage = "Material texture load failed: unable to open image";
        return false;
    }
    const bool ok = DecodeImageSourceToRgba8(source, outImage, errorMessage);
    CFRelease(source);
    return ok;
}

bool DecodeImageDataToRgba8(const uint8_t* data,
                            size_t size,
                            DecodedImageRgba8& outImage,
                            std::string& errorMessage) {
    NSData* nsData = [NSData dataWithBytes:data length:size];
    if (!nsData) {
        errorMessage = "Material texture load failed: NSData allocation failed";
        return false;
    }
    CGImageSourceRef source = CGImageSourceCreateWithData((__bridge CFDataRef)nsData, nullptr);
    if (!source) {
        errorMessage = "Material texture load failed: unable to open image data";
        return false;
    }
    const bool ok = DecodeImageSourceToRgba8(source, outImage, errorMessage);
    CFRelease(source);
    return ok;
}

bool ImageHasNonOpaqueAlpha(const DecodedImageRgba8& image) {
    for (size_t i = 3; i < image.rgba.size(); i += 4u) {
        if (image.rgba[i] < 250u) {
            return true;
        }
    }
    return false;
}

bool LabelSuggestsBlackKeyLetters(const std::string& label, MaterialTextureSemantic semantic) {
    if (semantic != MaterialTextureSemantic::BaseColor) {
        return false;
    }
    const std::string lower = ToLowerAscii(label);
    return lower.find("master_side_letters") != std::string::npos;
}

bool LabelSuggestsBorderBlackKeyCutout(const std::string& label, MaterialTextureSemantic semantic) {
    if (semantic != MaterialTextureSemantic::BaseColor) {
        return false;
    }
    const std::string lower = ToLowerAscii(label);
    return lower.find("8750083169368950601") != std::string::npos ||
           lower.find("6772804448157695701") != std::string::npos;
}

bool LabelSuggestsFerrariF14TextureOriginFlip(const std::string& label) {
    const std::string lower = ToLowerAscii(label);
    return lower.find("ferrari_f14_t_2014") != std::string::npos;
}

void FlipDecodedImageRows(DecodedImageRgba8& image) {
    if (image.rgba.empty() || image.width == 0u || image.height < 2u) {
        return;
    }
    const size_t rowBytes = static_cast<size_t>(image.width) * 4u;
    std::vector<uint8_t> scratch(rowBytes);
    for (uint32_t y = 0; y < image.height / 2u; ++y) {
        uint8_t* top = image.rgba.data() + static_cast<size_t>(y) * rowBytes;
        uint8_t* bottom = image.rgba.data() + static_cast<size_t>(image.height - 1u - y) * rowBytes;
        std::copy_n(top, rowBytes, scratch.data());
        std::copy_n(bottom, rowBytes, top);
        std::copy_n(scratch.data(), rowBytes, bottom);
    }
}

void DeriveBlackKeyAlpha(DecodedImageRgba8& image) {
    if (image.rgba.empty() || ImageHasNonOpaqueAlpha(image)) {
        return;
    }
    constexpr uint8_t kThreshold = 20u;
    for (size_t i = 0; i < image.rgba.size(); i += 4u) {
        const uint8_t mx = std::max({image.rgba[i + 0], image.rgba[i + 1], image.rgba[i + 2]});
        image.rgba[i + 3] = (mx <= kThreshold) ? 0u : 255u;
    }
}

void DeriveBorderBlackKeyAlpha(DecodedImageRgba8& image) {
    if (image.rgba.empty() || ImageHasNonOpaqueAlpha(image) ||
        image.width == 0u || image.height == 0u) {
        return;
    }
    constexpr uint8_t kThreshold = 20u;
    auto isBlackKey = [&](size_t pixel) {
        const size_t i = pixel * 4u;
        const uint8_t mx = std::max({image.rgba[i + 0], image.rgba[i + 1], image.rgba[i + 2]});
        return mx <= kThreshold;
    };
    const size_t width = image.width;
    const size_t height = image.height;
    std::vector<uint8_t> visited(width * height, 0u);
    std::vector<size_t> stack;
    stack.reserve(width + height);
    auto pushIfKey = [&](size_t x, size_t y) {
        const size_t pixel = y * width + x;
        if (visited[pixel] || !isBlackKey(pixel)) {
            return;
        }
        visited[pixel] = 1u;
        stack.push_back(pixel);
    };
    for (size_t x = 0; x < width; ++x) {
        pushIfKey(x, 0u);
        pushIfKey(x, height - 1u);
    }
    for (size_t y = 0; y < height; ++y) {
        pushIfKey(0u, y);
        pushIfKey(width - 1u, y);
    }
    while (!stack.empty()) {
        const size_t pixel = stack.back();
        stack.pop_back();
        const size_t x = pixel % width;
        const size_t y = pixel / width;
        if (x > 0u) {
            pushIfKey(x - 1u, y);
        }
        if (x + 1u < width) {
            pushIfKey(x + 1u, y);
        }
        if (y > 0u) {
            pushIfKey(x, y - 1u);
        }
        if (y + 1u < height) {
            pushIfKey(x, y + 1u);
        }
    }
    for (size_t pixel = 0u; pixel < visited.size(); ++pixel) {
        if (visited[pixel]) {
            image.rgba[pixel * 4u + 3u] = 0u;
        }
    }
}

id<MTLTexture> CreateTextureFromDecodedRgba8(id<MTLDevice> device,
                                             id<MTLCommandQueue> commandQueue,
                                             id<MTLBuffer> __strong& reusableUploadBuffer,
                                             const DecodedImageRgba8& image,
                                             MTLPixelFormat pixelFormat,
                                             bool buildMipmaps,
                                             MTLStorageMode storageMode,
                                             std::string& errorMessage) {
    if (!device || image.width == 0 || image.height == 0 || image.rgba.empty()) {
        errorMessage = "Material texture load failed: decoded image is empty";
        return nil;
    }

    const bool usePrivateUpload =
        (storageMode == MTLStorageModePrivate) && commandQueue;

    MTLTextureDescriptor* descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixelFormat
                                                          width:image.width
                                                         height:image.height
                                                      mipmapped:buildMipmaps];
    descriptor.usage = MTLTextureUsageShaderRead | (buildMipmaps ? MTLTextureUsageRenderTarget : 0u);
    descriptor.storageMode = usePrivateUpload ? MTLStorageModePrivate : storageMode;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    if (!texture) {
        errorMessage = "Material texture load failed: unable to allocate texture";
        return nil;
    }

    const NSUInteger bytesPerRow = static_cast<NSUInteger>(image.width) * 4u;
    if (usePrivateUpload) {
        const NSUInteger uploadBytes = bytesPerRow * static_cast<NSUInteger>(image.height);
        if (!reusableUploadBuffer || reusableUploadBuffer.length < uploadBytes) {
            reusableUploadBuffer =
                [device newBufferWithLength:uploadBytes options:MTLResourceStorageModeShared];
            if (reusableUploadBuffer) {
                reusableUploadBuffer.label = @"Texture Upload Buffer";
            }
        }
        id<MTLBuffer> stagingBuffer = reusableUploadBuffer;
        if (!stagingBuffer) {
            errorMessage = "Material texture load failed: unable to allocate upload buffer";
            return nil;
        }
        std::memcpy([stagingBuffer contents], image.rgba.data(), uploadBytes);

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        if (!blit) {
            errorMessage = "Material texture load failed: unable to create upload blit encoder";
            return nil;
        }
        [blit copyFromBuffer:stagingBuffer
                sourceOffset:0
           sourceBytesPerRow:bytesPerRow
         sourceBytesPerImage:uploadBytes
                  sourceSize:MTLSizeMake(image.width, image.height, 1)
                   toTexture:texture
            destinationSlice:0
            destinationLevel:0
           destinationOrigin:MTLOriginMake(0, 0, 0)];
        if (buildMipmaps && texture.mipmapLevelCount > 1u) {
            [blit generateMipmapsForTexture:texture];
        }
        [blit endEncoding];
        [commandBuffer commit];
        WaitForCommandBufferResponsive(commandBuffer);
        if (commandBuffer.error) {
            errorMessage = std::string("Material texture load failed: ") +
                           commandBuffer.error.localizedDescription.UTF8String;
            return nil;
        }
        return texture;
    }

    const MTLRegion region = MTLRegionMake2D(0, 0, image.width, image.height);
    [texture replaceRegion:region
               mipmapLevel:0
                 withBytes:image.rgba.data()
               bytesPerRow:bytesPerRow];
    return texture;
}

id<MTLTexture> LoadCompressedDdsTextureFromPath(id<MTLDevice> device,
                                                id<MTLCommandQueue> commandQueue,
                                                const std::string& path,
                                                bool srgb,
                                                std::string& errorMessage) {
    if (!device) {
        errorMessage = "Compressed DDS load failed: Metal device not ready";
        return nil;
    }

    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        errorMessage = "Compressed DDS load failed: unable to open file";
        return nil;
    }
    stream.seekg(0, std::ios::end);
    const std::streamsize size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    if (size <= 0) {
        errorMessage = "Compressed DDS load failed: file is empty";
        return nil;
    }

    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    stream.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!stream) {
        errorMessage = "Compressed DDS load failed: unable to read file";
        return nil;
    }

    DdsCompressedInfo info{};
    if (!ParseCompressedDdsHeader(bytes, srgb, info, errorMessage)) {
        return nil;
    }

    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:info.pixelFormat
                                                           width:info.width
                                                          height:info.height
                                                       mipmapped:(info.mipCount > 1u)];
    desc.textureType = MTLTextureType2D;
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
    if (!texture) {
        errorMessage = "Compressed DDS load failed: unable to allocate texture";
        return nil;
    }

    size_t offset = info.dataOffset;
    uint32_t mipWidth = info.width;
    uint32_t mipHeight = info.height;
    const uint32_t mipCount = std::min<uint32_t>(info.mipCount, static_cast<uint32_t>(texture.mipmapLevelCount));
    for (uint32_t level = 0u; level < mipCount; ++level) {
        const uint32_t blocksWide = std::max<uint32_t>(1u, (mipWidth + 3u) / 4u);
        const uint32_t blocksHigh = std::max<uint32_t>(1u, (mipHeight + 3u) / 4u);
        const NSUInteger bytesPerRow = static_cast<NSUInteger>(blocksWide * info.blockBytes);
        const size_t levelBytes =
            static_cast<size_t>(blocksWide) * static_cast<size_t>(blocksHigh) * info.blockBytes;
        MTLRegion region = MTLRegionMake2D(0, 0, mipWidth, mipHeight);
        [texture replaceRegion:region
                   mipmapLevel:level
                     withBytes:bytes.data() + offset
                   bytesPerRow:bytesPerRow];
        offset += levelBytes;
        mipWidth = std::max<uint32_t>(1u, mipWidth >> 1u);
        mipHeight = std::max<uint32_t>(1u, mipHeight >> 1u);
    }

    if (commandQueue) {
        id<MTLTexture> privateTexture = PromoteTextureToPrivate(device, commandQueue, texture);
        if (privateTexture) {
            return privateTexture;
        }
    }
    return texture;
}

id<MTLTexture> CreateTextureFromRgba8(id<MTLDevice> device,
                                      id<MTLCommandQueue> commandQueue,
                                      const Ktx2::ImageData& image,
                                      bool srgb,
                                      std::string& errorMessage) {
    if (!device) {
        errorMessage = "Material texture load failed: Metal device not ready";
        return nil;
    }
    if (image.width == 0 || image.height == 0 || image.rgba.empty()) {
        errorMessage = "Material texture load failed: KTX2 image is empty";
        return nil;
    }

    MTLTextureDescriptor* descriptor =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:(srgb ? MTLPixelFormatRGBA8Unorm_sRGB
                                                                      : MTLPixelFormatRGBA8Unorm)
                                                          width:image.width
                                                         height:image.height
                                                      mipmapped:NO];
    descriptor.usage = MTLTextureUsageShaderRead;
    descriptor.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [device newTextureWithDescriptor:descriptor];
    if (!texture) {
        errorMessage = "Material texture load failed: unable to allocate KTX2 texture";
        return nil;
    }

    const MTLRegion region = MTLRegionMake2D(0, 0, image.width, image.height);
    [texture replaceRegion:region
               mipmapLevel:0
                 withBytes:image.rgba.data()
               bytesPerRow:static_cast<NSUInteger>(image.width) * 4u];

    return texture;
}

id<MTLTexture> LoadKtx2TextureFromPath(id<MTLDevice> device,
                                       id<MTLCommandQueue> commandQueue,
                                       const std::string& path,
                                       bool srgb,
                                       std::string& errorMessage) {
    Ktx2::ImageData image;
    if (!Ktx2::ParseRgba8File(path, image, errorMessage)) {
        errorMessage = "Material texture load failed: " + errorMessage;
        return nil;
    }
    return CreateTextureFromRgba8(device, commandQueue, image, srgb, errorMessage);
}

id<MTLTexture> LoadKtx2TextureFromMemory(id<MTLDevice> device,
                                         id<MTLCommandQueue> commandQueue,
                                         const uint8_t* data,
                                         size_t size,
                                         bool srgb,
                                         std::string& errorMessage) {
    Ktx2::ImageData image;
    if (!Ktx2::ParseRgba8(data, size, image, errorMessage)) {
        errorMessage = "Material texture load failed: " + errorMessage;
        return nil;
    }
    return CreateTextureFromRgba8(device, commandQueue, image, srgb, errorMessage);
}

MTLSamplerAddressMode AddressModeFromGltfWrap(int32_t wrap) {
    switch (wrap) {
        case kGltfWrapClampToEdge:
            return MTLSamplerAddressModeClampToEdge;
        case kGltfWrapMirroredRepeat:
            return MTLSamplerAddressModeMirrorRepeat;
        case kGltfWrapRepeat:
        default:
            return MTLSamplerAddressModeRepeat;
    }
}

NSUInteger BytesPerPixel(MTLPixelFormat format) {
    switch (format) {
        case MTLPixelFormatRGBA32Float:
            return sizeof(float) * 4u;
        case MTLPixelFormatRGBA16Float:
            return sizeof(uint16_t) * 4u;
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGRA8Unorm_sRGB:
            return sizeof(uint8_t) * 4u;
        default:
            return 0u;
    }
}

NSUInteger BytesPerCompressedBlock(MTLPixelFormat format) {
    switch (format) {
        case MTLPixelFormatBC1_RGBA:
        case MTLPixelFormatBC1_RGBA_sRGB:
            return 8u;
        case MTLPixelFormatBC3_RGBA:
        case MTLPixelFormatBC3_RGBA_sRGB:
        case MTLPixelFormatBC5_RGUnorm:
            return 16u;
        default:
            return 0u;
    }
}

bool IsSrgbPixelFormat(MTLPixelFormat format) {
    return format == MTLPixelFormatRGBA8Unorm_sRGB ||
           format == MTLPixelFormatBGRA8Unorm_sRGB;
}

MTLPixelFormat LinearEquivalentPixelFormat(MTLPixelFormat format) {
    switch (format) {
        case MTLPixelFormatRGBA8Unorm_sRGB:
            return MTLPixelFormatRGBA8Unorm;
        case MTLPixelFormatBGRA8Unorm_sRGB:
            return MTLPixelFormatRGBA8Unorm;
        case MTLPixelFormatBGRA8Unorm:
            return MTLPixelFormatRGBA8Unorm;
        default:
            return format;
    }
}

NSUInteger MipDimension(NSUInteger base, NSUInteger level) {
    return std::max<NSUInteger>(1u, base >> level);
}

uint64_t EstimateTextureMemoryBytes(id<MTLTexture> texture) {
    if (!texture) {
        return 0;
    }

    const NSUInteger bytesPerPixel = BytesPerPixel(texture.pixelFormat);
    const NSUInteger bytesPerCompressedBlock = BytesPerCompressedBlock(texture.pixelFormat);
    if (bytesPerPixel == 0u && bytesPerCompressedBlock == 0u) {
        return 0u;
    }

    uint64_t totalBytes = 0;
    const NSUInteger mipCount = std::max<NSUInteger>(texture.mipmapLevelCount, 1u);
    const NSUInteger arrayLength = std::max<NSUInteger>(texture.arrayLength, 1u);
    const NSUInteger depth = std::max<NSUInteger>(texture.depth, 1u);
    for (NSUInteger level = 0; level < mipCount; ++level) {
        const uint64_t width = static_cast<uint64_t>(MipDimension(texture.width, level));
        const uint64_t height = static_cast<uint64_t>(MipDimension(texture.height, level));
        const uint64_t levelDepth = static_cast<uint64_t>(std::max<NSUInteger>(1u, depth >> level));
        if (bytesPerPixel > 0u) {
            totalBytes += width * height * levelDepth *
                          static_cast<uint64_t>(bytesPerPixel) *
                          static_cast<uint64_t>(arrayLength);
        } else {
            const uint64_t blocksWide = std::max<uint64_t>(1u, (width + 3u) / 4u);
            const uint64_t blocksHigh = std::max<uint64_t>(1u, (height + 3u) / 4u);
            totalBytes += blocksWide * blocksHigh * levelDepth *
                          static_cast<uint64_t>(bytesPerCompressedBlock) *
                          static_cast<uint64_t>(arrayLength);
        }
    }
    return totalBytes;
}

uint64_t TextureAllocatedSizeBytes(id<MTLTexture> texture) {
    if (!texture) {
        return 0u;
    }
    if ([texture respondsToSelector:@selector(allocatedSize)]) {
        return static_cast<uint64_t>(texture.allocatedSize);
    }
    return EstimateTextureMemoryBytes(texture);
}

bool CopyTextureLevelToFloat(id<MTLTexture> texture,
                             NSUInteger level,
                             bool decodeSrgb,
                             std::vector<float>& outFloats) {
    if (!texture) {
        return false;
    }

    const NSUInteger width = MipDimension(texture.width, level);
    const NSUInteger height = MipDimension(texture.height, level);
    if (width == 0 || height == 0) {
        return false;
    }

    const MTLPixelFormat format = texture.pixelFormat;
    const NSUInteger bytesPerPixel = BytesPerPixel(format);
    if (bytesPerPixel == 0) {
        return false;
    }

    const NSUInteger pixelCount = width * height;
    const NSUInteger bytesPerRow = width * bytesPerPixel;
    std::vector<uint8_t> rawData(static_cast<size_t>(bytesPerRow) * height);

    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    [texture getBytes:rawData.data()
          bytesPerRow:bytesPerRow
           fromRegion:region
          mipmapLevel:level];

    outFloats.assign(pixelCount * 4u, 0.0f);

    switch (format) {
        case MTLPixelFormatRGBA32Float: {
            const float* src = reinterpret_cast<const float*>(rawData.data());
            std::copy(src, src + outFloats.size(), outFloats.begin());
            return true;
        }
        case MTLPixelFormatRGBA16Float: {
            const __fp16* src = reinterpret_cast<const __fp16*>(rawData.data());
            for (size_t i = 0; i < outFloats.size(); ++i) {
                outFloats[i] = static_cast<float>(src[i]);
            }
            return true;
        }
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatBGRA8Unorm_sRGB: {
            const bool isBgra = (format == MTLPixelFormatBGRA8Unorm ||
                                 format == MTLPixelFormatBGRA8Unorm_sRGB);
            const bool isSrgbFormat = (format == MTLPixelFormatRGBA8Unorm_sRGB ||
                                       format == MTLPixelFormatBGRA8Unorm_sRGB);
            const uint8_t* src = rawData.data();
            for (NSUInteger i = 0; i < pixelCount; ++i) {
                uint8_t r8 = src[i * 4u + 0u];
                uint8_t g8 = src[i * 4u + 1u];
                uint8_t b8 = src[i * 4u + 2u];
                uint8_t a8 = src[i * 4u + 3u];
                if (isBgra) {
                    std::swap(r8, b8);
                }

                float r = static_cast<float>(r8) / 255.0f;
                float g = static_cast<float>(g8) / 255.0f;
                float b = static_cast<float>(b8) / 255.0f;
                if (decodeSrgb && isSrgbFormat) {
                    r = SrgbToLinear(r);
                    g = SrgbToLinear(g);
                    b = SrgbToLinear(b);
                }

                outFloats[i * 4u + 0u] = r;
                outFloats[i * 4u + 1u] = g;
                outFloats[i * 4u + 2u] = b;
                outFloats[i * 4u + 3u] = static_cast<float>(a8) / 255.0f;
            }
            return true;
        }
        default:
            return false;
    }
}

bool WriteTextureLevelFromFloat(id<MTLTexture> texture,
                                NSUInteger level,
                                const std::vector<float>& inFloats) {
    if (!texture) {
        return false;
    }

    const NSUInteger width = MipDimension(texture.width, level);
    const NSUInteger height = MipDimension(texture.height, level);
    if (width == 0 || height == 0) {
        return false;
    }

    const NSUInteger pixelCount = width * height;
    if (inFloats.size() != static_cast<size_t>(pixelCount) * 4u) {
        return false;
    }

    const MTLPixelFormat format = texture.pixelFormat;
    const NSUInteger bytesPerPixel = BytesPerPixel(format);
    if (bytesPerPixel == 0) {
        return false;
    }

    const NSUInteger bytesPerRow = width * bytesPerPixel;
    std::vector<uint8_t> rawData(static_cast<size_t>(bytesPerRow) * height, 0u);

    switch (format) {
        case MTLPixelFormatRGBA32Float: {
            float* dst = reinterpret_cast<float*>(rawData.data());
            std::copy(inFloats.begin(), inFloats.end(), dst);
            break;
        }
        case MTLPixelFormatRGBA16Float: {
            __fp16* dst = reinterpret_cast<__fp16*>(rawData.data());
            for (size_t i = 0; i < inFloats.size(); ++i) {
                float value = std::isfinite(inFloats[i]) ? inFloats[i] : 0.0f;
                dst[i] = static_cast<__fp16>(value);
            }
            break;
        }
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatBGRA8Unorm_sRGB: {
            const bool isBgra = (format == MTLPixelFormatBGRA8Unorm ||
                                 format == MTLPixelFormatBGRA8Unorm_sRGB);
            uint8_t* dst = rawData.data();
            for (NSUInteger i = 0; i < pixelCount; ++i) {
                float r = std::isfinite(inFloats[i * 4u + 0u]) ? inFloats[i * 4u + 0u] : 0.0f;
                float g = std::isfinite(inFloats[i * 4u + 1u]) ? inFloats[i * 4u + 1u] : 0.0f;
                float b = std::isfinite(inFloats[i * 4u + 2u]) ? inFloats[i * 4u + 2u] : 0.0f;
                float a = std::isfinite(inFloats[i * 4u + 3u]) ? inFloats[i * 4u + 3u] : 1.0f;
                uint8_t r8 = static_cast<uint8_t>(std::lround(std::clamp(r, 0.0f, 1.0f) * 255.0f));
                uint8_t g8 = static_cast<uint8_t>(std::lround(std::clamp(g, 0.0f, 1.0f) * 255.0f));
                uint8_t b8 = static_cast<uint8_t>(std::lround(std::clamp(b, 0.0f, 1.0f) * 255.0f));
                uint8_t a8 = static_cast<uint8_t>(std::lround(std::clamp(a, 0.0f, 1.0f) * 255.0f));
                if (isBgra) {
                    std::swap(r8, b8);
                }
                dst[i * 4u + 0u] = r8;
                dst[i * 4u + 1u] = g8;
                dst[i * 4u + 2u] = b8;
                dst[i * 4u + 3u] = a8;
            }
            break;
        }
        default:
            return false;
    }

    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    [texture replaceRegion:region
               mipmapLevel:level
                 withBytes:rawData.data()
               bytesPerRow:bytesPerRow];
    return true;
}

void DownsampleOrmMipLevel(const std::vector<float>& src,
                           NSUInteger srcWidth,
                           NSUInteger srcHeight,
                           std::vector<float>& dst,
                           NSUInteger dstWidth,
                           NSUInteger dstHeight,
                           int32_t wrapS,
                           int32_t wrapT) {
    dst.assign(static_cast<size_t>(dstWidth * dstHeight) * 4u, 0.0f);
    for (NSUInteger y = 0; y < dstHeight; ++y) {
        for (NSUInteger x = 0; x < dstWidth; ++x) {
            float sumAo = 0.0f;
            float sumAlpha = 0.0f;
            float sumMetal = 0.0f;
            float sumA = 0.0f;
            for (NSUInteger ky = 0; ky < 2; ++ky) {
                for (NSUInteger kx = 0; kx < 2; ++kx) {
                    auto wrap_coord = [](NSInteger coord, NSUInteger extent, int32_t mode) -> NSUInteger {
                        if (extent == 0u) {
                            return 0u;
                        }
                        const NSInteger maxIndex = static_cast<NSInteger>(extent) - 1;
                        switch (mode) {
                            case kGltfWrapRepeat: {
                                const NSInteger period = static_cast<NSInteger>(extent);
                                NSInteger wrapped = coord % period;
                                if (wrapped < 0) {
                                    wrapped += period;
                                }
                                return static_cast<NSUInteger>(wrapped);
                            }
                            case kGltfWrapMirroredRepeat: {
                                const NSInteger period = static_cast<NSInteger>(extent) * 2;
                                NSInteger wrapped = coord % period;
                                if (wrapped < 0) {
                                    wrapped += period;
                                }
                                if (wrapped >= static_cast<NSInteger>(extent)) {
                                    wrapped = period - wrapped - 1;
                                }
                                return static_cast<NSUInteger>(wrapped);
                            }
                            case kGltfWrapClampToEdge:
                            default:
                                if (coord <= 0) {
                                    return 0u;
                                }
                                if (coord >= maxIndex) {
                                    return static_cast<NSUInteger>(maxIndex);
                                }
                                return static_cast<NSUInteger>(coord);
                        }
                    };
                    NSInteger sxCoord = static_cast<NSInteger>(x * 2u + kx);
                    NSInteger syCoord = static_cast<NSInteger>(y * 2u + ky);
                    NSUInteger sx = wrap_coord(sxCoord, srcWidth, wrapS);
                    NSUInteger sy = wrap_coord(syCoord, srcHeight, wrapT);
                    size_t srcIndex = static_cast<size_t>((sy * srcWidth + sx) * 4u);
                    float ao = std::isfinite(src[srcIndex + 0u]) ? src[srcIndex + 0u] : 1.0f;
                    float rough = std::isfinite(src[srcIndex + 1u]) ? src[srcIndex + 1u] : 1.0f;
                    float metal = std::isfinite(src[srcIndex + 2u]) ? src[srcIndex + 2u] : 0.0f;
                    float alpha = std::isfinite(src[srcIndex + 3u]) ? src[srcIndex + 3u] : 1.0f;
                    rough = std::clamp(rough, 0.0f, 1.0f);
                    sumAo += ao;
                    sumAlpha += rough * rough;
                    sumMetal += metal;
                    sumA += alpha;
                }
            }

            const float invSampleCount = 0.25f;
            size_t dstIndex = static_cast<size_t>((y * dstWidth + x) * 4u);
            dst[dstIndex + 0u] = sumAo * invSampleCount;
            dst[dstIndex + 1u] = std::sqrt(std::max(sumAlpha * invSampleCount, 0.0f));
            dst[dstIndex + 2u] = sumMetal * invSampleCount;
            dst[dstIndex + 3u] = sumA * invSampleCount;
        }
    }
}

void PrefilterOrmBaseLevel(const std::vector<float>& src,
                           NSUInteger width,
                           NSUInteger height,
                           std::vector<float>& dst,
                           int32_t wrapS,
                           int32_t wrapT) {
    dst.assign(src.size(), 0.0f);
    if (width == 0 || height == 0) {
        return;
    }

    for (NSUInteger y = 0; y < height; ++y) {
        for (NSUInteger x = 0; x < width; ++x) {
            float sumAo = 0.0f;
            float sumAlpha = 0.0f;
            float sumMetal = 0.0f;
            float sumA = 0.0f;
            float totalW = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    auto wrap_coord = [](NSInteger coord, NSUInteger extent, int32_t mode) -> NSUInteger {
                        if (extent == 0u) {
                            return 0u;
                        }
                        const NSInteger maxIndex = static_cast<NSInteger>(extent) - 1;
                        switch (mode) {
                            case kGltfWrapRepeat: {
                                const NSInteger period = static_cast<NSInteger>(extent);
                                NSInteger wrapped = coord % period;
                                if (wrapped < 0) {
                                    wrapped += period;
                                }
                                return static_cast<NSUInteger>(wrapped);
                            }
                            case kGltfWrapMirroredRepeat: {
                                const NSInteger period = static_cast<NSInteger>(extent) * 2;
                                NSInteger wrapped = coord % period;
                                if (wrapped < 0) {
                                    wrapped += period;
                                }
                                if (wrapped >= static_cast<NSInteger>(extent)) {
                                    wrapped = period - wrapped - 1;
                                }
                                return static_cast<NSUInteger>(wrapped);
                            }
                            case kGltfWrapClampToEdge:
                            default:
                                if (coord <= 0) {
                                    return 0u;
                                }
                                if (coord >= maxIndex) {
                                    return static_cast<NSUInteger>(maxIndex);
                                }
                                return static_cast<NSUInteger>(coord);
                        }
                    };
                    const NSUInteger sx = wrap_coord(static_cast<NSInteger>(x) + kx, width, wrapS);
                    const NSUInteger sy = wrap_coord(static_cast<NSInteger>(y) + ky, height, wrapT);
                    const float weight = (kx == 0 && ky == 0) ? 4.0f : ((kx == 0 || ky == 0) ? 2.0f : 1.0f);
                    const size_t srcIndex = static_cast<size_t>((sy * width + sx) * 4u);
                    const float ao = std::isfinite(src[srcIndex + 0u]) ? src[srcIndex + 0u] : 1.0f;
                    const float rough = std::isfinite(src[srcIndex + 1u]) ? src[srcIndex + 1u] : 1.0f;
                    const float metal = std::isfinite(src[srcIndex + 2u]) ? src[srcIndex + 2u] : 0.0f;
                    const float alpha = std::isfinite(src[srcIndex + 3u]) ? src[srcIndex + 3u] : 1.0f;
                    const float rClamped = std::clamp(rough, 0.0f, 1.0f);
                    sumAo += weight * ao;
                    sumAlpha += weight * (rClamped * rClamped);
                    sumMetal += weight * metal;
                    sumA += weight * alpha;
                    totalW += weight;
                }
            }

            const float invW = (totalW > 0.0f) ? (1.0f / totalW) : 0.0f;
            const size_t dstIndex = static_cast<size_t>((y * width + x) * 4u);
            dst[dstIndex + 0u] = sumAo * invW;
            dst[dstIndex + 1u] = std::sqrt(std::max(sumAlpha * invW, 0.0f));
            dst[dstIndex + 2u] = sumMetal * invW;
            dst[dstIndex + 3u] = sumA * invW;
        }
    }
}

id<MTLTexture> BuildTextureWithBlitMips(id<MTLDevice> device,
                                        id<MTLCommandQueue> commandQueue,
                                        id<MTLTexture> source,
                                        MTLStorageMode storageMode = MTLStorageModeShared) {
    if (!device || !commandQueue || !source) {
        return nil;
    }
    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat
                                                           width:source.width
                                                          height:source.height
                                                       mipmapped:YES];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    desc.storageMode = storageMode;
    id<MTLTexture> mipTexture = [device newTextureWithDescriptor:desc];
    if (!mipTexture) {
        return nil;
    }
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!blit) {
        return nil;
    }
    MTLOrigin origin = MTLOriginMake(0, 0, 0);
    MTLSize size = MTLSizeMake(source.width, source.height, 1);
    [blit copyFromTexture:source
             sourceSlice:0
             sourceLevel:0
            sourceOrigin:origin
              sourceSize:size
               toTexture:mipTexture
        destinationSlice:0
        destinationLevel:0
       destinationOrigin:origin];
    if (mipTexture.mipmapLevelCount > 1u) {
        [blit generateMipmapsForTexture:mipTexture];
    }
    [blit endEncoding];
    [commandBuffer commit];
    WaitForCommandBufferResponsive(commandBuffer);
    return mipTexture;
}

id<MTLTexture> CloneTextureFromSourceLevel(id<MTLDevice> device,
                                           id<MTLCommandQueue> commandQueue,
                                           id<MTLTexture> source,
                                           NSUInteger sourceLevel) {
    if (!device || !commandQueue || !source || sourceLevel >= source.mipmapLevelCount) {
        return nil;
    }

    const NSUInteger width = MipDimension(source.width, sourceLevel);
    const NSUInteger height = MipDimension(source.height, sourceLevel);
    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat
                                                           width:width
                                                          height:height
                                                       mipmapped:YES];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> reducedTexture = [device newTextureWithDescriptor:desc];
    if (!reducedTexture) {
        return nil;
    }

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!blit) {
        return nil;
    }
    MTLOrigin origin = MTLOriginMake(0, 0, 0);
    MTLSize size = MTLSizeMake(width, height, 1);
    [blit copyFromTexture:source
             sourceSlice:0
             sourceLevel:sourceLevel
            sourceOrigin:origin
              sourceSize:size
               toTexture:reducedTexture
        destinationSlice:0
        destinationLevel:0
       destinationOrigin:origin];
    if (reducedTexture.mipmapLevelCount > 1u) {
        [blit generateMipmapsForTexture:reducedTexture];
    }
    [blit endEncoding];
    [commandBuffer commit];
    WaitForCommandBufferResponsive(commandBuffer);
    return reducedTexture;
}

id<MTLTexture> PromoteTextureToPrivate(id<MTLDevice> device,
                                       id<MTLCommandQueue> commandQueue,
                                       id<MTLTexture> source) {
    if (!device || !commandQueue || !source) {
        return nil;
    }
    if (source.storageMode == MTLStorageModePrivate) {
        return source;
    }

    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat
                                                           width:source.width
                                                          height:source.height
                                                       mipmapped:(source.mipmapLevelCount > 1u)];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModePrivate;
    id<MTLTexture> privateTexture = [device newTextureWithDescriptor:desc];
    if (!privateTexture) {
        return nil;
    }

    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!blit) {
        return nil;
    }

    MTLOrigin origin = MTLOriginMake(0, 0, 0);
    const NSUInteger mipCount = std::max<NSUInteger>(source.mipmapLevelCount, 1u);
    for (NSUInteger level = 0u; level < mipCount; ++level) {
        MTLSize size = MTLSizeMake(MipDimension(source.width, level),
                                   MipDimension(source.height, level),
                                   1);
        [blit copyFromTexture:source
                 sourceSlice:0
                 sourceLevel:level
                sourceOrigin:origin
                  sourceSize:size
                   toTexture:privateTexture
            destinationSlice:0
            destinationLevel:level
           destinationOrigin:origin];
    }
    [blit endEncoding];
    [commandBuffer commit];
    WaitForCommandBufferResponsive(commandBuffer);
    if (commandBuffer.error) {
        return nil;
    }

    return privateTexture;
}

id<MTLTexture> CloneTextureWithFormat(id<MTLDevice> device,
                                      id<MTLTexture> source,
                                      MTLPixelFormat pixelFormat) {
    if (!device || !source) {
        return nil;
    }
    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixelFormat
                                                           width:source.width
                                                          height:source.height
                                                       mipmapped:(source.mipmapLevelCount > 1u)];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> outTexture = [device newTextureWithDescriptor:desc];
    if (!outTexture) {
        return nil;
    }

    for (NSUInteger level = 0u; level < source.mipmapLevelCount; ++level) {
        std::vector<float> levelData;
        if (!CopyTextureLevelToFloat(source, level, /*decodeSrgb=*/false, levelData)) {
            return nil;
        }
        if (!WriteTextureLevelFromFloat(outTexture, level, levelData)) {
            return nil;
        }
    }
    return outTexture;
}

id<MTLTexture> BuildTextureWithOrmMips(id<MTLDevice> device,
                                       id<MTLCommandQueue> commandQueue,
                                       id<MTLTexture> source,
                                       const MaterialTextureSamplerDesc* samplerDesc,
                                       MTLStorageMode storageMode = MTLStorageModeShared) {
    if (!device || !source || source.width == 0 || source.height == 0) {
        return nil;
    }

    if (BytesPerPixel(source.pixelFormat) == 0) {
        return nil;
    }

    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat
                                                           width:source.width
                                                          height:source.height
                                                       mipmapped:YES];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> mipTexture = [device newTextureWithDescriptor:desc];
    if (!mipTexture) {
        return nil;
    }

    MaterialTextureSamplerDesc effectiveSampler = EffectiveSamplerDesc(samplerDesc);
    std::vector<float> prevLevel;
    if (!CopyTextureLevelToFloat(source, 0u, /*decodeSrgb=*/false, prevLevel)) {
        return nil;
    }
    if (!WriteTextureLevelFromFloat(mipTexture, 0u, prevLevel)) {
        return nil;
    }

    NSUInteger prevWidth = source.width;
    NSUInteger prevHeight = source.height;
    for (NSUInteger level = 1u; level < mipTexture.mipmapLevelCount; ++level) {
        NSUInteger dstWidth = MipDimension(source.width, level);
        NSUInteger dstHeight = MipDimension(source.height, level);
        std::vector<float> nextLevel;
        DownsampleOrmMipLevel(prevLevel,
                              prevWidth,
                              prevHeight,
                              nextLevel,
                              dstWidth,
                              dstHeight,
                              effectiveSampler.wrapS,
                              effectiveSampler.wrapT);
        if (!WriteTextureLevelFromFloat(mipTexture, level, nextLevel)) {
            return nil;
        }
        prevLevel.swap(nextLevel);
        prevWidth = dstWidth;
        prevHeight = dstHeight;
    }
#if !defined(NDEBUG)
    static bool sDumpedOrmMipChain = false;
    if (!sDumpedOrmMipChain) {
        sDumpedOrmMipChain = true;
        std::error_code ec;
        namespace fs = std::filesystem;
        fs::path dumpDir = fs::path("renders") / "debug" / "orm_mips";
        fs::create_directories(dumpDir, ec);
        for (NSUInteger level = 0u; level < mipTexture.mipmapLevelCount; ++level) {
            std::vector<float> levelData;
            if (!CopyTextureLevelToFloat(mipTexture, level, /*decodeSrgb=*/false, levelData)) {
                continue;
            }
            fs::path outPath = dumpDir / ("orm_mip_" + std::to_string(level) + ".exr");
            bool wrote = ImageWriter::WriteEXR(outPath.string().c_str(),
                                               levelData.data(),
                                               static_cast<int>(MipDimension(mipTexture.width, level)),
                                               static_cast<int>(MipDimension(mipTexture.height, level)),
                                               "Linear");
            if (!wrote) {
                NSLog(@"[glTF] Failed to dump ORM mip %lu", static_cast<unsigned long>(level));
            }
        }
        NSLog(@"[glTF] Dumped ORM mip chain to %s", dumpDir.string().c_str());
    }
#endif
    if (storageMode == MTLStorageModePrivate && commandQueue) {
        id<MTLTexture> privateTexture = PromoteTextureToPrivate(device, commandQueue, mipTexture);
        if (privateTexture) {
            return privateTexture;
        }
    }
    return mipTexture;
}

bool CopyTextureToFloat(id<MTLTexture> texture, std::vector<float>& outFloats) {
    if (!texture) {
        return false;
    }

    const NSUInteger width = texture.width;
    const NSUInteger height = texture.height;
    if (width == 0 || height == 0) {
        return false;
    }

    const MTLPixelFormat format = texture.pixelFormat;
    const NSUInteger bytesPerPixel = BytesPerPixel(format);
    if (bytesPerPixel == 0) {
        NSLog(@"Environment map pixel format %lu not supported for importance sampling", static_cast<unsigned long>(format));
        return false;
    }

    const NSUInteger pixelCount = width * height;
    const NSUInteger bytesPerRow = width * bytesPerPixel;
    std::vector<uint8_t> rawData(static_cast<size_t>(bytesPerRow) * height);

    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    [texture getBytes:rawData.data()
          bytesPerRow:bytesPerRow
           fromRegion:region
          mipmapLevel:0];

    outFloats.assign(pixelCount * 4u, 0.0f);

    switch (format) {
        case MTLPixelFormatRGBA32Float: {
            const float* src = reinterpret_cast<const float*>(rawData.data());
            std::copy(src, src + outFloats.size(), outFloats.begin());
            break;
        }
        case MTLPixelFormatRGBA16Float: {
            const __fp16* src = reinterpret_cast<const __fp16*>(rawData.data());
            for (size_t i = 0; i < outFloats.size(); ++i) {
                outFloats[i] = static_cast<float>(src[i]);
            }
            break;
        }
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatBGRA8Unorm: {
            const uint8_t* src = rawData.data();
            for (NSUInteger i = 0; i < pixelCount; ++i) {
                uint8_t r = src[i * 4u + 0u];
                uint8_t g = src[i * 4u + 1u];
                uint8_t b = src[i * 4u + 2u];
                uint8_t a = src[i * 4u + 3u];
                if (format == MTLPixelFormatBGRA8Unorm) {
                    std::swap(r, b);
                }
                outFloats[i * 4u + 0u] = static_cast<float>(r) / 255.0f;
                outFloats[i * 4u + 1u] = static_cast<float>(g) / 255.0f;
                outFloats[i * 4u + 2u] = static_cast<float>(b) / 255.0f;
                outFloats[i * 4u + 3u] = static_cast<float>(a) / 255.0f;
            }
            break;
        }
        case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatBGRA8Unorm_sRGB: {
            const uint8_t* src = rawData.data();
            for (NSUInteger i = 0; i < pixelCount; ++i) {
                uint8_t r = src[i * 4u + 0u];
                uint8_t g = src[i * 4u + 1u];
                uint8_t b = src[i * 4u + 2u];
                uint8_t a = src[i * 4u + 3u];
                if (format == MTLPixelFormatBGRA8Unorm_sRGB) {
                    std::swap(r, b);
                }
                outFloats[i * 4u + 0u] = SrgbToLinear(static_cast<float>(r) / 255.0f);
                outFloats[i * 4u + 1u] = SrgbToLinear(static_cast<float>(g) / 255.0f);
                outFloats[i * 4u + 2u] = SrgbToLinear(static_cast<float>(b) / 255.0f);
                outFloats[i * 4u + 3u] = static_cast<float>(a) / 255.0f;
            }
            break;
        }
        default:
            return false;
    }

    return true;
}

std::string CanonicalizePath(const std::string& path) {
    if (path.empty()) {
        return {};
    }
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path input(path);
    fs::path canonical = fs::weakly_canonical(input, ec);
    if (!ec) {
        return canonical.string();
    }
    fs::path normalized = input.lexically_normal();
    return normalized.string();
}

std::string DefaultMaterialName(uint32_t index) {
    return "Material " + std::to_string(index);
}

std::string DefaultMeshName(uint32_t index) {
    return "Mesh " + std::to_string(index);
}

std::string HashTexturePayload(const uint8_t* data, size_t size) {
    if (!data || size == 0u) {
        return "empty";
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data, static_cast<CC_LONG>(size), digest);
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(CC_SHA256_DIGEST_LENGTH * 2u);
    for (unsigned char byte : digest) {
        out.push_back(kHex[(byte >> 4u) & 0x0Fu]);
        out.push_back(kHex[byte & 0x0Fu]);
    }
    return out;
}

struct ThresholdChecksums {
    double head = 0.0;
    double total = 0.0;
};

ThresholdChecksums ComputeThresholdChecksums(const std::vector<float>& values) {
    constexpr size_t kHeadCount = 1024;
    ThresholdChecksums sums{};
    const size_t headLimit = std::min(values.size(), kHeadCount);
    for (size_t i = 0; i < values.size(); ++i) {
        const double value = static_cast<double>(values[i]);
        sums.total += value;
        if (i < headLimit) {
            sums.head += value;
        }
    }
    return sums;
}

constexpr float kSchlickAverageFactor = 1.0f / 21.0f;

inline float ComputeCoatAverage(float coatIor) {
    float eta = std::max(coatIor, 1.0f);
    float numerator = eta - 1.0f;
    float denominator = std::max(eta + 1.0f, 1.0e-6f);
    float ratio = numerator / denominator;
    float f0 = ratio * ratio;
    float average = f0 + (1.0f - f0) * kSchlickAverageFactor;
    return std::clamp(average, 0.0f, 0.999f);
}

inline float ComputeCoatSampleWeight(MaterialType type,
                                     float coatRoughness,
                                     float coatThickness,
                                     float coatAverage) {
    bool hasLayer = (coatThickness > 1.0e-4f) || (coatRoughness > 1.0e-4f) ||
                    type == MaterialType::Plastic || type == MaterialType::CarPaint;
    if (!hasLayer) {
        return 0.0f;
    }

    float weight = coatAverage * 2.5f + coatRoughness * 0.5f;
    if (type == MaterialType::CarPaint) {
        weight = std::max(weight, 0.35f);
    } else if (type == MaterialType::Plastic) {
        weight = std::max(weight, 0.25f);
    }
    return std::clamp(weight, 0.0f, 0.95f);
}

inline simd::float3 ClampColor01(const simd::float3& value) {
    return simd_make_float3(std::clamp(value.x, 0.0f, 1.0f),
                            std::clamp(value.y, 0.0f, 1.0f),
                            std::clamp(value.z, 0.0f, 1.0f));
}

inline simd::float3 ClampPositive(const simd::float3& value) {
    return simd_make_float3(std::max(value.x, 0.0f),
                            std::max(value.y, 0.0f),
                            std::max(value.z, 0.0f));
}

}  // namespace

SceneResources::SceneResources() = default;

SceneResources::~SceneResources() = default;

void SceneResources::initialize(const MetalContext& context) {
    initialize(context.device(), context.commandQueue(), context.supportsRaytracing());
}

void SceneResources::initialize(MTLDeviceHandle device,
                                MTLCommandQueueHandle commandQueue,
                                bool supportsRaytracing) {
    m_device = device;
    m_commandQueue = commandQueue;
    m_supportsRaytracing = supportsRaytracing;
    m_sphereCount = 0;
    m_rectangleCount = 0;
    m_materialCount = 0;
    m_triangleCount = 0;
    m_primitiveCount = 0;
    m_dirty = true;
    m_meshes.clear();
    m_assetPipelineReport = AssetPipelineReport{};
    m_environmentTexture = nil;
    m_meshInfoBuffer = nil;
    m_triangleBuffer = nil;
    m_materialTextures.clear();
    m_materialSamplers.clear();
    m_materialTextureSamplerIndices.clear();
    m_materialTextureCpuSamplers.clear();
    m_materialTextureLabels.clear();
    m_textureInventory.clear();
    m_materialTextureIndex.clear();
    m_materialSamplerIndices.clear();
    m_materialTextureInfoBuffer = nil;
    m_textureUploadBuffer = nil;
    m_forceSoftwareOverride = false;
    resetTextureBudgetState();
    clearEnvironmentDistribution();

    SceneAccelConfig accelConfig{};
    accelConfig.hardwareRaytracingSupported = m_supportsRaytracing;
    accelConfig.commandQueue = m_commandQueue;
    m_sceneAccel = CreateSceneAccel(accelConfig);
}

void SceneResources::setTextureLoadPolicy(const TextureLoadPolicy& policy) {
    m_textureLoadPolicy = policy;
    resetTextureBudgetState();
}

void SceneResources::setAssetPipelineReport(AssetPipelineReport report) {
    m_assetPipelineReport = std::move(report);
}

void SceneResources::resetTextureBudgetState() {
    m_textureBytesLoaded = 0;
    m_textureBudgetExceeded = false;
    m_textureBudgetMessage.clear();
}

uint32_t SceneResources::addMaterial(const simd::float3& baseColor,
                                     float roughness,
                                     MaterialType type,
                                     float indexOfRefraction,
                                     const simd::float3& emission,
                                     bool emissionUsesEnvironment,
                                     const simd::float3& conductorEta,
                                     const simd::float3& conductorK,
                                     bool hasConductorParameters,
                                     float coatRoughness,
                                     float coatThickness,
                                     const simd::float3& coatTint,
                                     const simd::float3& coatAbsorption,
                                     float coatIor,
                                     const simd::float3& dielectricSigmaA,
                                     const simd::float3& sssSigmaA,
                                     const simd::float3& sssSigmaS,
                                     float sssMeanFreePath,
                                     float sssAnisotropy,
                                     uint32_t sssMethod,
                                     bool sssCoatEnabled,
                                     bool sssSigmaOverride,
                                     float carpaintBaseMetallic,
                                     float carpaintBaseRoughness,
                                     float carpaintFlakeSampleWeight,
                                     float carpaintFlakeRoughness,
                                     float carpaintFlakeAnisotropy,
                                     float carpaintFlakeNormalStrength,
                                     float carpaintFlakeScale,
                                     float carpaintFlakeReflectanceScale,
                                     simd::float3 carpaintBaseEta,
                                     simd::float3 carpaintBaseK,
                                     bool carpaintHasBaseConductor,
                                     simd::float3 carpaintBaseTint,
                                     bool thinDielectric,
                                     std::string name) {
    if (m_materialCount >= kMaxMaterials) {
        return static_cast<uint32_t>(kMaxMaterials - 1);
    }

    uint32_t index = m_materialCount++;
    MaterialData material{};
    SetMaterialTextureTransformIdentity(material);

    simd::float3 clampedBaseColor = ClampColor01(baseColor);
    float clampedRoughness = std::clamp(roughness, 0.0f, 1.0f);
    material.baseColorRoughness = simd_make_float4(clampedBaseColor, clampedRoughness);

    float clampedIor = std::max(indexOfRefraction, 0.0f);
    float clampedCoatIor = std::max(coatIor, 0.0f);
    material.typeEta = simd_make_float4(static_cast<float>(type),
                                        clampedIor,
                                        clampedCoatIor,
                                        thinDielectric ? 1.0f : 0.0f);

    material.emission = simd_make_float4(emission, emissionUsesEnvironment ? 1.0f : 0.0f);

    simd::float3 etaClamped = ClampPositive(conductorEta);
    simd::float3 kClamped = ClampPositive(conductorK);
    float conductorFlag = hasConductorParameters ? 1.0f : 0.0f;
    material.conductorEta = simd_make_float4(etaClamped, conductorFlag);
    material.conductorK = simd_make_float4(kClamped, conductorFlag);

    float clampedCoatRoughness = std::clamp(coatRoughness, 0.0f, 1.0f);
    float clampedCoatThickness = std::max(coatThickness, 0.0f);
    float coatAverage = ComputeCoatAverage(clampedCoatIor);
    float coatSampleWeight = ComputeCoatSampleWeight(type,
                                                    clampedCoatRoughness,
                                                    clampedCoatThickness,
                                                    coatAverage);
    coatSampleWeight = std::clamp(coatSampleWeight, 0.0f, 0.95f);
    material.coatParams = simd_make_float4(clampedCoatRoughness,
                                           clampedCoatThickness,
                                           coatSampleWeight,
                                           coatAverage);

    simd::float3 coatTintClamped = ClampColor01(coatTint);
    simd::float3 coatAbsorptionClamped = ClampPositive(coatAbsorption);
    material.coatTint = simd_make_float4(coatTintClamped, 0.0f);
    material.coatAbsorption = simd_make_float4(coatAbsorptionClamped, 0.0f);
    simd::float3 dielectricSigmaAClamped = ClampPositive(dielectricSigmaA);
    material.dielectricSigmaA = simd_make_float4(dielectricSigmaAClamped, 0.0f);

    simd::float3 sigmaAClamped = ClampPositive(sssSigmaA);
    simd::float3 sigmaSClamped = ClampPositive(sssSigmaS);
    float sssFlag = sssSigmaOverride ? 1.0f : 0.0f;
    material.sssSigmaA = simd_make_float4(sigmaAClamped, sssFlag);
    float clampedAnisotropy = std::clamp(sssAnisotropy, -0.99f, 0.99f);
    material.sssSigmaS = simd_make_float4(sigmaSClamped, clampedAnisotropy);
    material.sssParams = simd_make_float4(std::max(sssMeanFreePath, 0.0f),
                                          static_cast<float>(sssMethod),
                                          sssCoatEnabled ? 1.0f : 0.0f,
                                          0.0f);

    float baseMetallicClamped = std::clamp(carpaintBaseMetallic, 0.0f, 1.0f);
    float baseRoughnessClamped = std::clamp(carpaintBaseRoughness, 0.0f, 1.0f);
    float flakeReflectanceClamped = std::clamp(carpaintFlakeReflectanceScale, 0.0f, 1.0f);
    float flakeSampleWeightClamped = std::clamp(carpaintFlakeSampleWeight, 0.0f, 0.95f);
    // Keep flake sampling weight aligned with the energy contributed by the lobe
    flakeSampleWeightClamped = std::clamp(flakeSampleWeightClamped * std::max(flakeReflectanceClamped, 0.01f), 0.0f, 0.95f);
    float flakeRoughnessClamped = std::clamp(carpaintFlakeRoughness, 0.0f, 1.0f);
    float flakeAnisotropyClamped = std::clamp(carpaintFlakeAnisotropy, -0.99f, 0.99f);
    float flakeNormalStrengthClamped = std::clamp(carpaintFlakeNormalStrength, 0.0f, 1.0f);
    float flakeScaleClamped = std::max(carpaintFlakeScale, 1.0e-4f);
    material.carpaintBaseParams = simd_make_float4(baseMetallicClamped,
                                                   baseRoughnessClamped,
                                                   flakeScaleClamped,
                                                   flakeReflectanceClamped);
    material.carpaintFlakeParams = simd_make_float4(flakeSampleWeightClamped,
                                                    flakeRoughnessClamped,
                                                    flakeAnisotropyClamped,
                                                    flakeNormalStrengthClamped);

    simd::float3 baseEtaClamped = ClampPositive(carpaintBaseEta);
    simd::float3 baseKClamped = ClampPositive(carpaintBaseK);
    float baseConductorFlag = carpaintHasBaseConductor ? 1.0f : 0.0f;
    if (!carpaintHasBaseConductor) {
        baseEtaClamped = simd_make_float3(0.0f, 0.0f, 0.0f);
        baseKClamped = simd_make_float3(0.0f, 0.0f, 0.0f);
    }
    material.carpaintBaseEta = simd_make_float4(baseEtaClamped, baseConductorFlag);
    material.carpaintBaseK = simd_make_float4(baseKClamped, baseConductorFlag);
    material.carpaintBaseTint = simd_make_float4(ClampColor01(carpaintBaseTint), 0.0f);
    material.spectralParams = simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    material.textureIndices0 =
        simd_make_uint4(kInvalidTextureIndex, kInvalidTextureIndex, kInvalidTextureIndex, kInvalidTextureIndex);
    material.textureIndices1 =
        simd_make_uint4(kInvalidTextureIndex, kInvalidTextureIndex, kInvalidTextureIndex, kInvalidTextureIndex);
    material.pbrParams = simd_make_float4(0.0f, clampedRoughness, 1.0f, 1.0f);
    SanitizeMaterialTextureMapping(material);

    m_materials[index] = material;
    m_materialDefaults[index] = material;
    m_materialNames[index] = name.empty() ? DefaultMaterialName(index) : std::move(name);
    m_dirty = true;
    return index;
}

uint32_t SceneResources::addMaterial(const simd::float3& albedo,
                                     float fuzz,
                                     MaterialType type,
                                     float indexOfRefraction,
                                     const simd::float3& emission,
                                     bool emissionUsesEnvironment,
                                     std::string name) {
    float clampedRoughness = std::clamp(fuzz, 0.0f, 1.0f);
    simd::float3 zero = simd_make_float3(0.0f, 0.0f, 0.0f);
    simd::float3 coatTint = simd_make_float3(1.0f, 1.0f, 1.0f);

    return addMaterial(albedo,
                       clampedRoughness,
                       type,
                       indexOfRefraction,
                       emission,
                       emissionUsesEnvironment,
                       zero,
                       zero,
                       false,
                       0.0f,
                       0.0f,
                       coatTint,
                       zero,
                       1.5f,
                       zero,
                       zero,
                       zero,
                       0.0f,
                       0.0f,
                       0u,
                       false,
                       false,
                       0.0f,
                       clampedRoughness,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       1.0f,
                       1.0f,
                       zero,
                       zero,
                       false,
                       ClampColor01(simd_make_float3(1.0f, 1.0f, 1.0f)),
                       /*thinDielectric=*/false,
                       std::move(name));
}

uint32_t SceneResources::addMaterialData(const MaterialData& material,
                                         std::string name) {
    if (m_materialCount >= kMaxMaterials) {
        return static_cast<uint32_t>(kMaxMaterials - 1);
    }

    uint32_t index = m_materialCount++;
    MaterialData sanitized = material;
    SanitizeMaterialTextureMapping(sanitized);
    m_materials[index] = sanitized;
    m_materialDefaults[index] = sanitized;
    m_materialNames[index] = name.empty() ? DefaultMaterialName(index) : std::move(name);
    m_dirty = true;
    return index;
}

bool SceneResources::isDeferredTextureIndex(uint32_t index) const {
    return index >= kDeferredTextureIndexBase &&
           index < (kDeferredTextureIndexBase + static_cast<uint32_t>(m_deferredMaterialTextures.size()));
}

uint32_t SceneResources::registerDeferredMaterialTexture(const std::string& key,
                                                         const std::string& label,
                                                         bool fromFile,
                                                         const std::string& path,
                                                         const uint8_t* data,
                                                         size_t size,
                                                         bool srgb,
                                                         const MaterialTextureSamplerDesc* samplerDesc,
                                                         MaterialTextureSemantic semantic) {
    auto cached = m_materialTextureIndex.find(key);
    if (cached != m_materialTextureIndex.end()) {
        return cached->second;
    }

    if (m_deferredMaterialTextures.size() >=
        (static_cast<size_t>(std::numeric_limits<uint32_t>::max()) - kDeferredTextureIndexBase)) {
        return kInvalidTextureIndex;
    }

    DeferredMaterialTexture deferred{};
    deferred.placeholderIndex =
        kDeferredTextureIndexBase + static_cast<uint32_t>(m_deferredMaterialTextures.size());
    deferred.fromFile = fromFile;
    deferred.path = path;
    deferred.label = label;
    deferred.srgb = srgb;
    deferred.semantic = semantic;
    if (samplerDesc) {
        deferred.hasSamplerDesc = true;
        deferred.samplerDesc = *samplerDesc;
    }
    if (!fromFile && data && size > 0u) {
        deferred.data.assign(data, data + size);
    }

    m_deferredMaterialTextures.push_back(std::move(deferred));
    m_materialTextureIndex.emplace(key, m_deferredMaterialTextures.back().placeholderIndex);
    return m_deferredMaterialTextures.back().placeholderIndex;
}

uint32_t SceneResources::resolveDeferredTextureIndex(const SceneResources& src,
                                                     uint32_t index,
                                                     std::unordered_map<uint32_t, uint32_t>& cache,
                                                     std::string* errorMessage) {
    if (index == kInvalidTextureIndex) {
        return kInvalidTextureIndex;
    }
    if (index < kDeferredTextureIndexBase) {
        return index;
    }

    auto cached = cache.find(index);
    if (cached != cache.end()) {
        return cached->second;
    }

    if (index < kDeferredTextureIndexBase) {
        return index;
    }
    const uint32_t deferredOffset = index - kDeferredTextureIndexBase;
    if (deferredOffset >= src.m_deferredMaterialTextures.size()) {
        if (errorMessage) {
            *errorMessage = "Deferred texture placeholder is out of range";
        }
        return kInvalidTextureIndex;
    }

    const DeferredMaterialTexture& deferred = src.m_deferredMaterialTextures[deferredOffset];
    std::string localError;
    TextureLoadStatus loadStatus = TextureLoadStatus::Failed;
    const MaterialTextureSamplerDesc* samplerDesc =
        deferred.hasSamplerDesc ? &deferred.samplerDesc : nullptr;
    uint32_t resolved = kInvalidTextureIndex;
    if (!deferred.decodedRgba.empty() &&
        deferred.decodedWidth > 0u &&
        deferred.decodedHeight > 0u) {
        DecodedImageRgba8 decoded{};
        decoded.width = deferred.decodedWidth;
        decoded.height = deferred.decodedHeight;
        decoded.rgba = deferred.decodedRgba;
        const bool buildMipmaps = m_stagedUploadMode ? false : (deferred.semantic != MaterialTextureSemantic::Orm);
        const MTLStorageMode finalStorage =
            (m_stagedUploadMode || !buildMipmaps || !m_commandQueue) ? MTLStorageModeShared
                                                                     : MTLStorageModePrivate;
        id<MTLTexture> texture = CreateTextureFromDecodedRgba8(m_device,
                                                               m_commandQueue,
                                                               m_textureUploadBuffer,
                                                               decoded,
                                                               deferred.srgb ? MTLPixelFormatRGBA8Unorm_sRGB
                                                                             : MTLPixelFormatRGBA8Unorm,
                                                               buildMipmaps,
                                                               finalStorage,
                                                               localError);
        if (texture) {
            TextureInventoryEntry inventory{};
            inventory.label = deferred.label;
            inventory.formatLabel = PixelFormatLabel(texture.pixelFormat);
            inventory.semantic = deferred.semantic;
            inventory.originalWidth = static_cast<uint32_t>(texture.width);
            inventory.originalHeight = static_cast<uint32_t>(texture.height);
            inventory.originalEstimatedBytes = EstimateTextureMemoryBytes(texture);
            inventory.originalAllocatedBytes = TextureAllocatedSizeBytes(texture);

            std::string budgetReason;
            const uint64_t estimatedBytes = EstimateTextureMemoryBytes(texture);
            if (shouldSkipTextureForBudget(deferred.semantic, estimatedBytes, budgetReason)) {
                localError = budgetReason;
                loadStatus = TextureLoadStatus::SkippedByPolicy;
            } else {
                const std::string deferredKey = "deferred:" + std::to_string(index);
                resolved = registerMaterialTexture(texture,
                                                   deferredKey,
                                                   deferred.label,
                                                   samplerDesc,
                                                   deferred.semantic,
                                                   &inventory);
                loadStatus = (resolved == kInvalidTextureIndex) ? TextureLoadStatus::Failed
                                                                : TextureLoadStatus::Loaded;
            }
        }
    } else if (deferred.fromFile) {
        resolved = addMaterialTextureFromFile(deferred.path,
                                              deferred.srgb,
                                              &localError,
                                              samplerDesc,
                                              deferred.semantic,
                                              &loadStatus);
    } else {
        resolved = addMaterialTextureFromData(deferred.data.data(),
                                              deferred.data.size(),
                                              deferred.label,
                                              deferred.srgb,
                                              &localError,
                                              samplerDesc,
                                              deferred.semantic,
                                              &loadStatus);
    }

    if (resolved == kInvalidTextureIndex &&
        loadStatus != TextureLoadStatus::SkippedByPolicy &&
        errorMessage) {
        *errorMessage = localError.empty()
            ? ("Failed to resolve deferred texture: " + deferred.label)
            : localError;
    }

    cache.emplace(index, resolved);
    return resolved;
}

uint32_t SceneResources::materialSamplerIndexForDesc(
    const MaterialTextureSamplerDesc* samplerDesc) {
    if (!m_device) {
        return 0u;
    }

    MaterialTextureSamplerDesc effective = EffectiveSamplerDesc(samplerDesc);
    uint64_t key = SamplerDescKey(effective);
    auto cached = m_materialSamplerIndices.find(key);
    if (cached != m_materialSamplerIndices.end()) {
        return cached->second;
    }

    MTLSamplerMinMagFilter minFilter = MTLSamplerMinMagFilterLinear;
    MTLSamplerMinMagFilter magFilter = MTLSamplerMinMagFilterLinear;
    MTLSamplerMipFilter mipFilter = MTLSamplerMipFilterLinear;
    switch (effective.minFilter) {
        case kGltfFilterNearest:
            minFilter = MTLSamplerMinMagFilterNearest;
            mipFilter = MTLSamplerMipFilterNotMipmapped;
            break;
        case kGltfFilterLinear:
            minFilter = MTLSamplerMinMagFilterLinear;
            mipFilter = MTLSamplerMipFilterNotMipmapped;
            break;
        case kGltfFilterNearestMipmapNearest:
            minFilter = MTLSamplerMinMagFilterNearest;
            mipFilter = MTLSamplerMipFilterNearest;
            break;
        case kGltfFilterLinearMipmapNearest:
            minFilter = MTLSamplerMinMagFilterLinear;
            mipFilter = MTLSamplerMipFilterNearest;
            break;
        case kGltfFilterNearestMipmapLinear:
            minFilter = MTLSamplerMinMagFilterNearest;
            mipFilter = MTLSamplerMipFilterLinear;
            break;
        case kGltfFilterLinearMipmapLinear:
        default:
            minFilter = MTLSamplerMinMagFilterLinear;
            mipFilter = MTLSamplerMipFilterLinear;
            break;
    }

    switch (effective.magFilter) {
        case kGltfFilterNearest:
            magFilter = MTLSamplerMinMagFilterNearest;
            break;
        case kGltfFilterLinear:
        default:
            magFilter = MTLSamplerMinMagFilterLinear;
            break;
    }

    MTLSamplerDescriptor* descriptor = [[MTLSamplerDescriptor alloc] init];
    descriptor.minFilter = minFilter;
    descriptor.magFilter = magFilter;
    descriptor.mipFilter = mipFilter;
    descriptor.sAddressMode = AddressModeFromGltfWrap(effective.wrapS);
    descriptor.tAddressMode = AddressModeFromGltfWrap(effective.wrapT);
    descriptor.normalizedCoordinates = YES;
    descriptor.lodMinClamp = 0.0f;
    descriptor.lodMaxClamp = FLT_MAX;
    if ([descriptor respondsToSelector:@selector(setSupportArgumentBuffers:)]) {
        descriptor.supportArgumentBuffers = YES;
    }
    if (minFilter == MTLSamplerMinMagFilterLinear &&
        magFilter == MTLSamplerMinMagFilterLinear &&
        mipFilter != MTLSamplerMipFilterNotMipmapped) {
        descriptor.maxAnisotropy = kDefaultMaterialAnisotropy;
    } else {
        descriptor.maxAnisotropy = 1;
    }

    MTLSamplerStateHandle sampler = [m_device newSamplerStateWithDescriptor:descriptor];
    if (!sampler || m_materialSamplers.size() >= kMaxMaterialSamplers) {
        MaterialTextureSamplerDesc fallback = EffectiveSamplerDesc(nullptr);
        uint64_t fallbackKey = SamplerDescKey(fallback);
        auto fallbackIt = m_materialSamplerIndices.find(fallbackKey);
        if (fallbackIt != m_materialSamplerIndices.end()) {
            return fallbackIt->second;
        }
        return 0u;
    }
    uint32_t samplerIndex = static_cast<uint32_t>(m_materialSamplers.size());
    m_materialSamplers.push_back(sampler);
    m_materialSamplerIndices.emplace(key, samplerIndex);
    return samplerIndex;
}

bool SceneResources::isTextureAllowedByPilotMode(MaterialTextureSemantic semantic,
                                                 std::string& reason) const {
    switch (m_textureLoadPolicy.pilotMode) {
        case PilotMode::GeoOnly:
            reason = "geo_only disables all material textures";
            return false;
        case PilotMode::AlbedoOnly:
            if (semantic == MaterialTextureSemantic::BaseColor ||
                semantic == MaterialTextureSemantic::Generic) {
                return true;
            }
            reason = "albedo_only keeps only baseColor textures";
            return false;
        case PilotMode::FullClamped:
        default:
            return true;
    }
}

bool SceneResources::shouldSkipTextureForBudget(MaterialTextureSemantic semantic,
                                                uint64_t estimatedBytes,
                                                std::string& reason) {
    if (m_textureLoadPolicy.maxTextureBytes == 0u) {
        return false;
    }

    const uint64_t projected = m_textureBytesLoaded + estimatedBytes;
    if (projected <= m_textureLoadPolicy.maxTextureBytes) {
        return false;
    }

    std::ostringstream message;
    message << "texture budget " << m_textureLoadPolicy.maxTextureBytes
            << " bytes exceeded by " << TextureSemanticLabel(semantic)
            << " texture (" << projected << " projected bytes)";
    m_textureBudgetExceeded = true;
    m_textureBudgetMessage = message.str();

    if (m_textureLoadPolicy.policy == TextureBudgetPolicy::Warn &&
        semantic != MaterialTextureSemantic::BaseColor) {
        reason = "warn policy skipped texture after exceeding --max-texture-bytes";
        return true;
    }

    reason = m_textureBudgetMessage;
    return false;
}

void SceneResources::rebuildMaterialTextureInfoBuffer() {
    if (!m_device || m_materialTextures.empty()) {
        m_materialTextureInfoBuffer = nil;
        return;
    }
    const NSUInteger infoCount = static_cast<NSUInteger>(m_materialTextures.size());
    const NSUInteger infoBytes = infoCount * sizeof(MaterialTextureInfo);
    if (!m_materialTextureInfoBuffer || m_materialTextureInfoBuffer.length < infoBytes) {
        m_materialTextureInfoBuffer = [m_device newBufferWithLength:infoBytes
                                                            options:MTLResourceStorageModeShared];
    }
    if (!m_materialTextureInfoBuffer) {
        return;
    }

    auto* infos = reinterpret_cast<MaterialTextureInfo*>([m_materialTextureInfoBuffer contents]);
    if (!infos) {
        return;
    }
    for (NSUInteger i = 0; i < infoCount; ++i) {
        MaterialTextureInfo info{};
        id<MTLTexture> texture = m_materialTextures[i];
        if (texture) {
            info.width = static_cast<uint32_t>(texture.width);
            info.height = static_cast<uint32_t>(texture.height);
            info.mipCount = static_cast<uint32_t>(texture.mipmapLevelCount);
            if (i < m_materialTextureSamplerIndices.size()) {
                info.flags = m_materialTextureSamplerIndices[i];
            }
        }
        infos[i] = info;
    }
}

uint32_t SceneResources::registerMaterialTexture(MTLTextureHandle texture,
                                                 const std::string& key,
                                                 const std::string& label,
                                                 const MaterialTextureSamplerDesc* samplerDesc,
                                                 MaterialTextureSemantic semantic,
                                                 const TextureInventoryEntry* inventoryTemplate) {
    if (!texture) {
        return kInvalidTextureIndex;
    }
    auto found = m_materialTextureIndex.find(key);
    if (found != m_materialTextureIndex.end()) {
        return found->second;
    }
    if (m_materialTextures.size() >= kMaxMaterialTextures) {
        std::cerr << "[MaterialTexture] cap reached (" << kMaxMaterialTextures
                  << ") while registering " << label << " ("
                  << TextureSemanticLabel(semantic) << ")" << std::endl;
        return kInvalidTextureIndex;
    }
    if (m_device && !m_stagedUploadMode) {
        if (semantic == MaterialTextureSemantic::Orm) {
            id<MTLTexture> ormMipTexture =
                BuildTextureWithOrmMips(m_device,
                                        m_commandQueue,
                                        texture,
                                        samplerDesc,
                                        MTLStorageModePrivate);
            if (ormMipTexture) {
                texture = ormMipTexture;
            } else if (texture.mipmapLevelCount <= 1 &&
                       BytesPerPixel(texture.pixelFormat) > 0u &&
                       m_commandQueue) {
                id<MTLTexture> mipTexture =
                    BuildTextureWithBlitMips(m_device, m_commandQueue, texture, MTLStorageModePrivate);
                if (mipTexture) {
                    texture = mipTexture;
                }
            }
        } else if (texture.mipmapLevelCount <= 1 &&
                   BytesPerPixel(texture.pixelFormat) > 0u &&
                   m_commandQueue) {
            id<MTLTexture> mipTexture =
                BuildTextureWithBlitMips(m_device, m_commandQueue, texture, MTLStorageModePrivate);
            if (mipTexture) {
                texture = mipTexture;
            }
        }
    }
    if (texture) {
        if (!m_stagedUploadMode && m_device && m_commandQueue) {
            id<MTLTexture> privateTexture = PromoteTextureToPrivate(m_device, m_commandQueue, texture);
            if (privateTexture) {
                texture = privateTexture;
            }
        }
        texture.label = [NSString stringWithFormat:@"Material Texture (%s)", label.c_str()];
    }
    if (m_materialSamplers.empty()) {
        (void)materialSamplerIndexForDesc(nullptr);
    }
    uint32_t samplerIndex = materialSamplerIndexForDesc(samplerDesc);
    uint32_t index = static_cast<uint32_t>(m_materialTextures.size());
    m_materialTextures.push_back(texture);
    m_materialTextureSamplerIndices.push_back(samplerIndex);
    m_materialTextureCpuSamplers.push_back(EffectiveSamplerDesc(samplerDesc));
    m_materialTextureLabels.push_back(label);
    TextureInventoryEntry inventory{};
    if (inventoryTemplate) {
        inventory = *inventoryTemplate;
    }
    inventory.label = label;
    inventory.semantic = semantic;
    inventory.width = static_cast<uint32_t>(texture.width);
    inventory.height = static_cast<uint32_t>(texture.height);
    inventory.mipCount = static_cast<uint32_t>(texture.mipmapLevelCount);
    inventory.estimatedBytes = EstimateTextureMemoryBytes(texture);
    inventory.allocatedBytes = TextureAllocatedSizeBytes(texture);
    if (inventory.originalWidth == 0u) {
        inventory.originalWidth = inventory.width;
    }
    if (inventory.originalHeight == 0u) {
        inventory.originalHeight = inventory.height;
    }
    if (inventory.originalEstimatedBytes == 0u) {
        inventory.originalEstimatedBytes = inventory.estimatedBytes;
    }
    if (inventory.originalAllocatedBytes == 0u) {
        inventory.originalAllocatedBytes = inventory.allocatedBytes;
    }
    if (inventory.formatLabel.empty()) {
        inventory.formatLabel = PixelFormatLabel(texture.pixelFormat);
    }
    m_textureInventory.push_back(std::move(inventory));
    m_textureBytesLoaded += m_textureInventory.back().estimatedBytes;
    m_materialTextureIndex.emplace(key, index);
    rebuildMaterialTextureInfoBuffer();
    return index;
}

uint32_t SceneResources::addMaterialTextureFromFile(const std::string& path,
                                                    bool srgb,
                                                    std::string* errorMessage,
                                                    const MaterialTextureSamplerDesc* samplerDesc,
                                                    MaterialTextureSemantic semantic,
                                                    TextureLoadStatus* outStatus) {
    if (outStatus) {
        *outStatus = TextureLoadStatus::Failed;
    }
    if (path.empty()) {
        if (errorMessage) {
            *errorMessage = "Material texture load failed: empty path";
        }
        return kInvalidTextureIndex;
    }

    std::string policyReason;
    if (!isTextureAllowedByPilotMode(semantic, policyReason)) {
        std::cout << "[TexturePolicy] skip " << path << " (" << TextureSemanticLabel(semantic)
                  << "): " << policyReason << std::endl;
        if (errorMessage) {
            *errorMessage = policyReason;
        }
        if (outStatus) {
            *outStatus = TextureLoadStatus::SkippedByPolicy;
        }
        return kInvalidTextureIndex;
    }

    const std::string canonicalPath = CanonicalizePath(path);
    const bool deriveBlackKeyAlpha = LabelSuggestsBlackKeyLetters(canonicalPath, semantic);
    const bool deriveBorderBlackKeyAlpha =
        LabelSuggestsBorderBlackKeyCutout(canonicalPath, semantic);
    const bool flipFerrariF14TextureOrigin = LabelSuggestsFerrariF14TextureOriginFlip(canonicalPath);
    const std::string dedupeIdentity = [&]() {
        const std::string outputHash = ResolveTextureOutputHash(canonicalPath);
        if (!outputHash.empty()) {
            return std::string("hash:") + outputHash;
        }
        return canonicalPath;
    }();
    const std::string key =
        dedupeIdentity + (srgb ? "|srgb" : "|linear") + SamplerKeySuffix(samplerDesc) +
        TextureSemanticKeySuffix(semantic) +
        (deriveBlackKeyAlpha ? "|alpha:black-key" : "") +
        (deriveBorderBlackKeyAlpha ? "|alpha:border-black-key" : "") +
        (flipFerrariF14TextureOrigin ? "|origin:ferrari-f14-flip" : "") +
        "|maxDim:" + std::to_string(m_textureLoadPolicy.maxDimension);
    auto cached = m_materialTextureIndex.find(key);
    if (cached != m_materialTextureIndex.end()) {
        if (outStatus) {
            *outStatus = TextureLoadStatus::Loaded;
        }
        return cached->second;
    }

    if (!m_device) {
        uint32_t deferred = kInvalidTextureIndex;
        if (HasExtensionCaseInsensitive(canonicalPath, ".ktx2")) {
            std::vector<uint8_t> bytes;
            std::string stageError;
            std::ifstream stream(canonicalPath, std::ios::binary);
            if (!stream.is_open()) {
                stageError = "Material texture staging failed: unable to open file";
            } else {
                stream.seekg(0, std::ios::end);
                const std::streamsize fileSize = stream.tellg();
                stream.seekg(0, std::ios::beg);
                if (fileSize < 0) {
                    stageError = "Material texture staging failed: invalid file size";
                } else {
                    bytes.resize(static_cast<size_t>(fileSize));
                    if (fileSize > 0) {
                        stream.read(reinterpret_cast<char*>(bytes.data()), fileSize);
                    }
                    if (!stream) {
                        stageError = "Material texture staging failed: unable to read file";
                    }
                }
            }
            if (!stageError.empty()) {
                if (errorMessage) {
                    *errorMessage = stageError;
                }
                return kInvalidTextureIndex;
            }
            deferred = registerDeferredMaterialTexture(key,
                                                       canonicalPath,
                                                       /*fromFile=*/false,
                                                       {},
                                                       bytes.data(),
                                                       bytes.size(),
                                                       srgb,
                                                       samplerDesc,
                                                       semantic);
        } else {
            DecodedImageRgba8 decoded{};
            std::string stageError;
            if (!DecodeImageFileToRgba8(canonicalPath, decoded, stageError)) {
                if (errorMessage) {
                    *errorMessage = stageError;
                }
                return kInvalidTextureIndex;
            }
            if (flipFerrariF14TextureOrigin) {
                FlipDecodedImageRows(decoded);
            }
            if (deriveBlackKeyAlpha) {
                DeriveBlackKeyAlpha(decoded);
            }
            if (deriveBorderBlackKeyAlpha) {
                DeriveBorderBlackKeyAlpha(decoded);
            }
            deferred = registerDeferredMaterialTexture(key,
                                                       canonicalPath,
                                                       /*fromFile=*/false,
                                                       {},
                                                       nullptr,
                                                       0u,
                                                       srgb,
                                                       samplerDesc,
                                                       semantic);
            if (deferred != kInvalidTextureIndex) {
                DeferredMaterialTexture& entry =
                    m_deferredMaterialTextures[deferred - kDeferredTextureIndexBase];
                entry.label = canonicalPath;
                entry.decodedWidth = decoded.width;
                entry.decodedHeight = decoded.height;
                entry.decodedRgba = std::move(decoded.rgba);
            }
        }
        if (outStatus) {
            *outStatus = (deferred == kInvalidTextureIndex) ? TextureLoadStatus::Failed
                                                            : TextureLoadStatus::Loaded;
        }
        if (deferred == kInvalidTextureIndex && errorMessage) {
            *errorMessage = "Material texture staging failed";
        }
        return deferred;
    }

    id<MTLTexture> texture = nil;
    if (HasExtensionCaseInsensitive(canonicalPath, ".ktx2")) {
        std::string loadError;
        const std::string compressedSourcePath = ResolveCompressedSourceTexturePath(canonicalPath);
        if (!compressedSourcePath.empty()) {
            texture = LoadCompressedDdsTextureFromPath(m_device, m_commandQueue, compressedSourcePath, srgb, loadError);
            if (texture) {
                loadError.clear();
            }
        }
        if (!texture) {
            texture = LoadKtx2TextureFromPath(m_device, m_commandQueue, canonicalPath, srgb, loadError);
        }
        if (!texture) {
            if (errorMessage) {
                *errorMessage = loadError;
            }
            return kInvalidTextureIndex;
        }
    } else {
        DecodedImageRgba8 decoded{};
        std::string loadError;
        if (!DecodeImageFileToRgba8(canonicalPath, decoded, loadError)) {
            if (errorMessage) {
                *errorMessage = loadError;
            }
            return kInvalidTextureIndex;
        }
        if (flipFerrariF14TextureOrigin) {
            FlipDecodedImageRows(decoded);
        }
        if (deriveBlackKeyAlpha) {
            DeriveBlackKeyAlpha(decoded);
        }
        if (deriveBorderBlackKeyAlpha) {
            DeriveBorderBlackKeyAlpha(decoded);
        }
        const bool buildMipmaps = m_stagedUploadMode ? false : (semantic != MaterialTextureSemantic::Orm);
        const MTLStorageMode finalStorage =
            (m_stagedUploadMode || !buildMipmaps || !m_commandQueue) ? MTLStorageModeShared
                                                                     : MTLStorageModePrivate;
        texture = CreateTextureFromDecodedRgba8(m_device,
                                               m_commandQueue,
                                               m_textureUploadBuffer,
                                               decoded,
                                               srgb ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm,
                                               buildMipmaps,
                                               finalStorage,
                                               loadError);
        if (!texture) {
            if (errorMessage) {
                *errorMessage = loadError;
            }
            return kInvalidTextureIndex;
        }
    }

    TextureInventoryEntry inventory{};
    inventory.label = canonicalPath;
    inventory.formatLabel = PixelFormatLabel(texture.pixelFormat);
    inventory.semantic = semantic;
    inventory.originalWidth = static_cast<uint32_t>(texture.width);
    inventory.originalHeight = static_cast<uint32_t>(texture.height);
    inventory.originalEstimatedBytes = EstimateTextureMemoryBytes(texture);
    inventory.originalAllocatedBytes = TextureAllocatedSizeBytes(texture);

    if (m_textureLoadPolicy.maxDimension > 0u &&
        (texture.width > m_textureLoadPolicy.maxDimension ||
         texture.height > m_textureLoadPolicy.maxDimension)) {
        NSUInteger sourceLevel = 0u;
        while (sourceLevel + 1u < texture.mipmapLevelCount &&
               (MipDimension(texture.width, sourceLevel) > m_textureLoadPolicy.maxDimension ||
                MipDimension(texture.height, sourceLevel) > m_textureLoadPolicy.maxDimension)) {
            ++sourceLevel;
        }
        if (sourceLevel > 0u && m_commandQueue) {
            id<MTLTexture> reducedTexture =
                CloneTextureFromSourceLevel(m_device, m_commandQueue, texture, sourceLevel);
            if (reducedTexture) {
                texture = reducedTexture;
                inventory.clampedByDimension = true;
                std::ostringstream decision;
                decision << "clamped " << inventory.originalWidth << "x" << inventory.originalHeight
                         << " -> " << texture.width << "x" << texture.height
                         << " via mip level " << sourceLevel;
                inventory.decision = decision.str();
                std::cout << "[TexturePolicy] " << canonicalPath << " (" << TextureSemanticLabel(semantic)
                          << "): " << inventory.decision << std::endl;
            }
        }
    }

    std::string budgetReason;
    const uint64_t estimatedBytes = EstimateTextureMemoryBytes(texture);
    if (shouldSkipTextureForBudget(semantic, estimatedBytes, budgetReason)) {
        std::cout << "[TexturePolicy] skip " << canonicalPath << " (" << TextureSemanticLabel(semantic)
                  << "): " << budgetReason << std::endl;
        if (errorMessage) {
            *errorMessage = budgetReason;
        }
        if (outStatus) {
            *outStatus = TextureLoadStatus::SkippedByPolicy;
        }
        return kInvalidTextureIndex;
    }

    texture.label = [NSString stringWithFormat:@"Material Texture (%s)", canonicalPath.c_str()];
    const uint32_t index =
        registerMaterialTexture(texture, key, canonicalPath, samplerDesc, semantic, &inventory);
    if (outStatus) {
        *outStatus = (index == kInvalidTextureIndex) ? TextureLoadStatus::Failed
                                                     : TextureLoadStatus::Loaded;
    }
    return index;
}

uint32_t SceneResources::addMaterialTextureFromData(const uint8_t* data,
                                                    size_t size,
                                                    const std::string& label,
                                                    bool srgb,
                                                    std::string* errorMessage,
                                                    const MaterialTextureSamplerDesc* samplerDesc,
                                                    MaterialTextureSemantic semantic,
                                                    TextureLoadStatus* outStatus) {
    if (outStatus) {
        *outStatus = TextureLoadStatus::Failed;
    }
    if (!data || size == 0) {
        if (errorMessage) {
            *errorMessage = "Material texture load failed: empty data";
        }
        return kInvalidTextureIndex;
    }

    std::string policyReason;
    if (!isTextureAllowedByPilotMode(semantic, policyReason)) {
        std::cout << "[TexturePolicy] skip " << label << " (" << TextureSemanticLabel(semantic)
                  << "): " << policyReason << std::endl;
        if (errorMessage) {
            *errorMessage = policyReason;
        }
        if (outStatus) {
            *outStatus = TextureLoadStatus::SkippedByPolicy;
        }
        return kInvalidTextureIndex;
    }

    const bool deriveBlackKeyAlpha = LabelSuggestsBlackKeyLetters(label, semantic);
    const bool deriveBorderBlackKeyAlpha = LabelSuggestsBorderBlackKeyCutout(label, semantic);
    const bool flipFerrariF14TextureOrigin = LabelSuggestsFerrariF14TextureOriginFlip(label);
    const std::string dedupeIdentity =
        std::string("datahash:") + HashTexturePayload(data, size);
    const std::string key =
        dedupeIdentity + (srgb ? "|srgb" : "|linear") + SamplerKeySuffix(samplerDesc) +
        TextureSemanticKeySuffix(semantic) +
        (deriveBlackKeyAlpha ? "|alpha:black-key" : "") +
        (deriveBorderBlackKeyAlpha ? "|alpha:border-black-key" : "") +
        (flipFerrariF14TextureOrigin ? "|origin:ferrari-f14-flip" : "") +
        "|maxDim:" + std::to_string(m_textureLoadPolicy.maxDimension);
    auto cached = m_materialTextureIndex.find(key);
    if (cached != m_materialTextureIndex.end()) {
        if (outStatus) {
            *outStatus = TextureLoadStatus::Loaded;
        }
        return cached->second;
    }

    if (!m_device) {
        uint32_t deferred = kInvalidTextureIndex;
        if (IsKtx2Data(data, size)) {
            deferred = registerDeferredMaterialTexture(key,
                                                       label,
                                                       /*fromFile=*/false,
                                                       {},
                                                       data,
                                                       size,
                                                       srgb,
                                                       samplerDesc,
                                                       semantic);
        } else {
            DecodedImageRgba8 decoded{};
            std::string stageError;
            if (!DecodeImageDataToRgba8(data, size, decoded, stageError)) {
                if (errorMessage) {
                    *errorMessage = stageError;
                }
                return kInvalidTextureIndex;
            }
            if (flipFerrariF14TextureOrigin) {
                FlipDecodedImageRows(decoded);
            }
            if (deriveBlackKeyAlpha) {
                DeriveBlackKeyAlpha(decoded);
            }
            if (deriveBorderBlackKeyAlpha) {
                DeriveBorderBlackKeyAlpha(decoded);
            }
            deferred = registerDeferredMaterialTexture(key,
                                                       label,
                                                       /*fromFile=*/false,
                                                       {},
                                                       nullptr,
                                                       0u,
                                                       srgb,
                                                       samplerDesc,
                                                       semantic);
            if (deferred != kInvalidTextureIndex) {
                DeferredMaterialTexture& entry =
                    m_deferredMaterialTextures[deferred - kDeferredTextureIndexBase];
                entry.label = label;
                entry.decodedWidth = decoded.width;
                entry.decodedHeight = decoded.height;
                entry.decodedRgba = std::move(decoded.rgba);
            }
        }
        if (outStatus) {
            *outStatus = (deferred == kInvalidTextureIndex) ? TextureLoadStatus::Failed
                                                            : TextureLoadStatus::Loaded;
        }
        if (deferred == kInvalidTextureIndex && errorMessage) {
            *errorMessage = "Material texture staging failed";
        }
        return deferred;
    }

    id<MTLTexture> texture = nil;
    if (IsKtx2Data(data, size)) {
        std::string loadError;
        texture = LoadKtx2TextureFromMemory(m_device, m_commandQueue, data, size, srgb, loadError);
        if (!texture) {
            if (errorMessage) {
                *errorMessage = loadError;
            }
            return kInvalidTextureIndex;
        }
    } else {
        DecodedImageRgba8 decoded{};
        std::string loadError;
        if (!DecodeImageDataToRgba8(data, size, decoded, loadError)) {
            if (errorMessage) {
                *errorMessage = loadError;
            }
            return kInvalidTextureIndex;
        }
        if (flipFerrariF14TextureOrigin) {
            FlipDecodedImageRows(decoded);
        }
        if (deriveBlackKeyAlpha) {
            DeriveBlackKeyAlpha(decoded);
        }
        if (deriveBorderBlackKeyAlpha) {
            DeriveBorderBlackKeyAlpha(decoded);
        }
        const bool buildMipmaps = m_stagedUploadMode ? false : (semantic != MaterialTextureSemantic::Orm);
        const MTLStorageMode finalStorage =
            (m_stagedUploadMode || !buildMipmaps || !m_commandQueue) ? MTLStorageModeShared
                                                                     : MTLStorageModePrivate;
        texture = CreateTextureFromDecodedRgba8(m_device,
                                               m_commandQueue,
                                               m_textureUploadBuffer,
                                               decoded,
                                               srgb ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm,
                                               buildMipmaps,
                                               finalStorage,
                                               loadError);
        if (!texture) {
            if (errorMessage) {
                *errorMessage = loadError;
            }
            return kInvalidTextureIndex;
        }
    }

    TextureInventoryEntry inventory{};
    inventory.label = label;
    inventory.formatLabel = PixelFormatLabel(texture.pixelFormat);
    inventory.semantic = semantic;
    inventory.originalWidth = static_cast<uint32_t>(texture.width);
    inventory.originalHeight = static_cast<uint32_t>(texture.height);
    inventory.originalEstimatedBytes = EstimateTextureMemoryBytes(texture);
    inventory.originalAllocatedBytes = TextureAllocatedSizeBytes(texture);

    if (m_textureLoadPolicy.maxDimension > 0u &&
        (texture.width > m_textureLoadPolicy.maxDimension ||
         texture.height > m_textureLoadPolicy.maxDimension)) {
        NSUInteger sourceLevel = 0u;
        while (sourceLevel + 1u < texture.mipmapLevelCount &&
               (MipDimension(texture.width, sourceLevel) > m_textureLoadPolicy.maxDimension ||
                MipDimension(texture.height, sourceLevel) > m_textureLoadPolicy.maxDimension)) {
            ++sourceLevel;
        }
        if (sourceLevel > 0u && m_commandQueue) {
            id<MTLTexture> reducedTexture =
                CloneTextureFromSourceLevel(m_device, m_commandQueue, texture, sourceLevel);
            if (reducedTexture) {
                texture = reducedTexture;
                inventory.clampedByDimension = true;
                std::ostringstream decision;
                decision << "clamped " << inventory.originalWidth << "x" << inventory.originalHeight
                         << " -> " << texture.width << "x" << texture.height
                         << " via mip level " << sourceLevel;
                inventory.decision = decision.str();
                std::cout << "[TexturePolicy] " << label << " (" << TextureSemanticLabel(semantic)
                          << "): " << inventory.decision << std::endl;
            }
        }
    }

    std::string budgetReason;
    const uint64_t estimatedBytes = EstimateTextureMemoryBytes(texture);
    if (shouldSkipTextureForBudget(semantic, estimatedBytes, budgetReason)) {
        std::cout << "[TexturePolicy] skip " << label << " (" << TextureSemanticLabel(semantic)
                  << "): " << budgetReason << std::endl;
        if (errorMessage) {
            *errorMessage = budgetReason;
        }
        if (outStatus) {
            *outStatus = TextureLoadStatus::SkippedByPolicy;
        }
        return kInvalidTextureIndex;
    }

    texture.label = [NSString stringWithFormat:@"Material Texture (%s)", label.c_str()];
    const uint32_t index =
        registerMaterialTexture(texture, key, label, samplerDesc, semantic, &inventory);
    if (outStatus) {
        *outStatus = (index == kInvalidTextureIndex) ? TextureLoadStatus::Failed
                                                     : TextureLoadStatus::Loaded;
    }
    return index;
}

bool SceneResources::materialTextureCpuData(uint32_t index, MaterialTextureCpuData& out) const {
    out = MaterialTextureCpuData{};
    if (index == kInvalidTextureIndex) {
        return false;
    }

    auto flipRowsInPlace = [](std::vector<float>& rgba, uint32_t width, uint32_t height) {
        if (width == 0u || height == 0u) {
            return;
        }
        const size_t rowFloats = static_cast<size_t>(width) * 4u;
        std::vector<float> flipped(rgba.size(), 0.0f);
        for (uint32_t y = 0; y < height; ++y) {
            const size_t srcOffset = static_cast<size_t>(y) * rowFloats;
            const size_t dstOffset = static_cast<size_t>(height - 1u - y) * rowFloats;
            std::copy_n(rgba.data() + srcOffset, rowFloats, flipped.data() + dstOffset);
        }
        rgba.swap(flipped);
    };

    if (index < m_materialTextures.size()) {
        id<MTLTexture> texture = m_materialTextures[index];
        if (!texture) {
            return false;
        }
        id<MTLTexture> readableTexture = texture;
        if (texture.storageMode == MTLStorageModePrivate && m_device && m_commandQueue) {
            readableTexture = BuildTextureWithBlitMips(m_device,
                                                       m_commandQueue,
                                                       texture,
                                                       MTLStorageModeShared);
            if (!readableTexture) {
                return false;
            }
        }
        const NSUInteger mipCount = std::max<NSUInteger>(texture.mipmapLevelCount, 1u);
        out.mipLevels.resize(mipCount);
        for (NSUInteger level = 0; level < mipCount; ++level) {
            std::vector<float> rgba;
            if (!CopyTextureLevelToFloat(readableTexture, level, /*decodeSrgb=*/true, rgba)) {
                return false;
            }
            out.mipLevels[level].width = static_cast<uint32_t>(std::max<NSUInteger>(readableTexture.width >> level, 1u));
            out.mipLevels[level].height = static_cast<uint32_t>(std::max<NSUInteger>(readableTexture.height >> level, 1u));
            flipRowsInPlace(rgba, out.mipLevels[level].width, out.mipLevels[level].height);
            out.mipLevels[level].rgba = std::move(rgba);
        }
        out.width = static_cast<uint32_t>(readableTexture.width);
        out.height = static_cast<uint32_t>(readableTexture.height);
        out.srgb = IsSrgbPixelFormat(readableTexture.pixelFormat);
        out.rgba = out.mipLevels.front().rgba;
        if (index < m_materialTextureCpuSamplers.size()) {
            out.samplerDesc = m_materialTextureCpuSamplers[index];
        }
        return true;
    }

    if (!isDeferredTextureIndex(index)) {
        return false;
    }
    const uint32_t deferredOffset = index - kDeferredTextureIndexBase;
    if (deferredOffset >= m_deferredMaterialTextures.size()) {
        return false;
    }

    const DeferredMaterialTexture& deferred = m_deferredMaterialTextures[deferredOffset];
    if (deferred.decodedWidth == 0 || deferred.decodedHeight == 0 || deferred.decodedRgba.empty()) {
        return false;
    }

    out.width = deferred.decodedWidth;
    out.height = deferred.decodedHeight;
    out.srgb = deferred.srgb;
    out.samplerDesc = deferred.hasSamplerDesc ? deferred.samplerDesc : MaterialTextureSamplerDesc{};
    const size_t pixelCount = static_cast<size_t>(out.width) * static_cast<size_t>(out.height);
    out.rgba.resize(pixelCount * 4u, 0.0f);
    for (size_t i = 0; i < pixelCount; ++i) {
        float r = static_cast<float>(deferred.decodedRgba[i * 4u + 0u]) / 255.0f;
        float g = static_cast<float>(deferred.decodedRgba[i * 4u + 1u]) / 255.0f;
        float b = static_cast<float>(deferred.decodedRgba[i * 4u + 2u]) / 255.0f;
        if (deferred.srgb) {
            r = SrgbToLinear(r);
            g = SrgbToLinear(g);
            b = SrgbToLinear(b);
        }
        out.rgba[i * 4u + 0u] = r;
        out.rgba[i * 4u + 1u] = g;
        out.rgba[i * 4u + 2u] = b;
        out.rgba[i * 4u + 3u] = static_cast<float>(deferred.decodedRgba[i * 4u + 3u]) / 255.0f;
    }
    out.mipLevels.resize(1u);
    out.mipLevels[0].width = out.width;
    out.mipLevels[0].height = out.height;
    out.mipLevels[0].rgba = out.rgba;

    while (out.mipLevels.back().width > 1u || out.mipLevels.back().height > 1u) {
        const auto& src = out.mipLevels.back();
        MaterialTextureCpuData::MipLevel next{};
        next.width = std::max(src.width / 2u, 1u);
        next.height = std::max(src.height / 2u, 1u);
        next.rgba.resize(static_cast<size_t>(next.width) * static_cast<size_t>(next.height) * 4u, 0.0f);
        for (uint32_t y = 0; y < next.height; ++y) {
            for (uint32_t x = 0; x < next.width; ++x) {
                simd::float4 accum = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                uint32_t taps = 0u;
                for (uint32_t oy = 0; oy < 2u; ++oy) {
                    for (uint32_t ox = 0; ox < 2u; ++ox) {
                        uint32_t sx = std::min(x * 2u + ox, src.width - 1u);
                        uint32_t sy = std::min(y * 2u + oy, src.height - 1u);
                        size_t idx = (static_cast<size_t>(sy) * src.width + static_cast<size_t>(sx)) * 4u;
                        accum += simd_make_float4(src.rgba[idx + 0u],
                                                  src.rgba[idx + 1u],
                                                  src.rgba[idx + 2u],
                                                  src.rgba[idx + 3u]);
                        ++taps;
                    }
                }
                accum /= std::max(static_cast<float>(taps), 1.0f);
                size_t dst = (static_cast<size_t>(y) * next.width + static_cast<size_t>(x)) * 4u;
                next.rgba[dst + 0u] = accum.x;
                next.rgba[dst + 1u] = accum.y;
                next.rgba[dst + 2u] = accum.z;
                next.rgba[dst + 3u] = accum.w;
            }
        }
        out.mipLevels.push_back(std::move(next));
    }
    return true;
}

const std::string& SceneResources::materialName(uint32_t index) const {
    static const std::string kEmptyName;
    if (index >= m_materialCount) {
        return kEmptyName;
    }
    return m_materialNames[index];
}

bool SceneResources::updateMaterial(uint32_t index, const MaterialData& material) {
    if (index >= m_materialCount) {
        return false;
    }
    m_materials[index] = material;
    uploadMaterialToGpu(index);
    return true;
}

bool SceneResources::resetMaterial(uint32_t index) {
    if (index >= m_materialCount) {
        return false;
    }
    m_materials[index] = m_materialDefaults[index];
    uploadMaterialToGpu(index);
    return true;
}

SceneMemoryReport SceneResources::sceneMemoryReport() const {
    SceneMemoryReport report{};
    report.triangleCount = m_triangleCount;
    report.meshCount = static_cast<uint32_t>(m_meshes.size());
    report.instanceCount = static_cast<uint32_t>(m_meshes.size());
    report.textureCount = static_cast<uint32_t>(m_materialTextures.size()) +
                          ((m_environmentTexture != nil) ? 1u : 0u);

    report.geometryMemoryBytes += static_cast<uint64_t>(m_sphereBuffer ? m_sphereBuffer.length : 0u);
    report.geometryMemoryBytes += static_cast<uint64_t>(m_materialBuffer ? m_materialBuffer.length : 0u);
    report.geometryMemoryBytes += static_cast<uint64_t>(m_rectangleBuffer ? m_rectangleBuffer.length : 0u);
    report.geometryMemoryBytes += static_cast<uint64_t>(m_meshInfoBuffer ? m_meshInfoBuffer.length : 0u);
    report.geometryMemoryBytes += static_cast<uint64_t>(m_triangleBuffer ? m_triangleBuffer.length : 0u);
    report.geometryMemoryBytes += static_cast<uint64_t>(m_meshVertexBuffer ? m_meshVertexBuffer.length : 0u);
    report.geometryMemoryBytes += static_cast<uint64_t>(m_meshIndexBuffer ? m_meshIndexBuffer.length : 0u);
    std::unordered_set<uintptr_t> seenMeshVertexBuffers;
    std::unordered_set<uintptr_t> seenMeshIndexBuffers;
    for (const auto& mesh : m_meshes) {
        if (mesh.vertexBuffer) {
            const uintptr_t key = reinterpret_cast<uintptr_t>(mesh.vertexBuffer);
            if (seenMeshVertexBuffers.insert(key).second) {
                report.geometryMemoryBytes += static_cast<uint64_t>(mesh.vertexBuffer.length);
            }
        }
        if (mesh.indexBuffer) {
            const uintptr_t key = reinterpret_cast<uintptr_t>(mesh.indexBuffer);
            if (seenMeshIndexBuffers.insert(key).second) {
                report.geometryMemoryBytes += static_cast<uint64_t>(mesh.indexBuffer.length);
            }
        }
    }

    for (MTLTextureHandle texture : m_materialTextures) {
        report.textureMemoryBytes += EstimateTextureMemoryBytes(texture);
        report.textureAllocatedMemoryBytes += TextureAllocatedSizeBytes(texture);
    }
    report.textureMemoryBytes += EstimateTextureMemoryBytes(m_environmentTexture);
    report.textureAllocatedMemoryBytes += TextureAllocatedSizeBytes(m_environmentTexture);

    report.blasMemoryBytes = m_lastBuildStats.blasMemoryBytes;
    report.tlasMemoryBytes = m_lastBuildStats.tlasMemoryBytes;
    report.scratchMemoryBytes = m_lastBuildStats.retainedScratchMemoryBytes;
    report.peakScratchMemoryBytes = m_lastBuildStats.scratchMemoryBytes;
    report.totalEstimatedMemoryBytes =
        report.geometryMemoryBytes +
        report.textureMemoryBytes +
        report.blasMemoryBytes +
        report.tlasMemoryBytes +
        report.scratchMemoryBytes;

    report.recommendedMaxWorkingSetSizeBytes = recommendedMaxWorkingSetSizeBytes();
    if (report.recommendedMaxWorkingSetSizeBytes > 0u) {
        report.budgetUsagePercent =
            (static_cast<double>(report.totalEstimatedMemoryBytes) /
             static_cast<double>(report.recommendedMaxWorkingSetSizeBytes)) * 100.0;
    }

    return report;
}

uint64_t SceneResources::estimatedGeometryMemoryBytes() const {
    const bool includePackedTriangles =
        !hardwareRaytracingEnabled() || KeepTriangleBufferForHardware();

    uint64_t totalBytes = 0;
    totalBytes += static_cast<uint64_t>(m_sphereCount) * sizeof(PathTracerShaderTypes::SphereData);
    totalBytes += static_cast<uint64_t>(m_materialCount) * sizeof(PathTracerShaderTypes::MaterialData);
    totalBytes += static_cast<uint64_t>(m_rectangleCount) * sizeof(PathTracerShaderTypes::RectData);
    totalBytes += static_cast<uint64_t>(m_meshes.size()) * sizeof(PathTracerShaderTypes::MeshInfo);

    // Estimate deduplicated GPU geometry payload (shared mesh buffers + shared
    // packed vertices/indices), matching current runtime upload behavior.
    std::unordered_set<std::string> seenVertexBufferKeys;
    std::unordered_set<std::string> seenIndexBufferKeys;
    for (const auto& mesh : m_meshes) {
        std::string vertexKey;
        std::string indexKey;
        if (!mesh.geometryCacheKey.empty()) {
            vertexKey = "geo-v:" + mesh.geometryCacheKey;
            indexKey = "geo-i:" + mesh.geometryCacheKey;
        } else {
            // Fallback for meshes without a geometry cache key (procedural).
            vertexKey = "ptr-v:" + std::to_string(reinterpret_cast<uintptr_t>(mesh.vertices.data()));
            indexKey = "ptr-i:" + std::to_string(reinterpret_cast<uintptr_t>(mesh.indices.data()));
        }

        if (seenVertexBufferKeys.insert(vertexKey).second) {
            totalBytes += static_cast<uint64_t>(mesh.vertices.size()) * sizeof(MeshVertex);
            totalBytes += static_cast<uint64_t>(mesh.vertices.size()) *
                          sizeof(PathTracerShaderTypes::SceneVertex);
        }
        if (seenIndexBufferKeys.insert(indexKey).second) {
            totalBytes += static_cast<uint64_t>(mesh.indices.size()) * sizeof(uint32_t);
            totalBytes += static_cast<uint64_t>(mesh.indices.size() / 3u) * sizeof(simd::uint3);
        }

        // Packed triangle data remains per-instance metadata today.
        if (includePackedTriangles) {
            totalBytes += static_cast<uint64_t>(mesh.indices.size() / 3u) *
                          sizeof(PathTracerShaderTypes::TriangleData);
        }

    }
    return totalBytes;
}

uint64_t SceneResources::estimatedTextureMemoryBytes() const {
    uint64_t totalBytes = 0;
    for (const auto& entry : m_textureInventory) {
        totalBytes += entry.estimatedBytes;
    }
    totalBytes += EstimateTextureMemoryBytes(m_environmentTexture);
    return totalBytes;
}

uint64_t SceneResources::allocatedTextureMemoryBytes() const {
    uint64_t totalBytes = 0;
    for (MTLTextureHandle texture : m_materialTextures) {
        totalBytes += TextureAllocatedSizeBytes(texture);
    }
    totalBytes += TextureAllocatedSizeBytes(m_environmentTexture);
    return totalBytes;
}

uint64_t SceneResources::recommendedMaxWorkingSetSizeBytes() const {
    if (m_device && [m_device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
        return static_cast<uint64_t>(m_device.recommendedMaxWorkingSetSize);
    }
    return 0u;
}

std::vector<MeshInventoryEntry> SceneResources::meshInventory() const {
    const bool includePackedTriangles =
        !hardwareRaytracingEnabled() || KeepTriangleBufferForHardware();

    std::vector<MeshInventoryEntry> entries;
    entries.reserve(m_meshes.size());
    std::unordered_set<std::string> seenGeometryKeys;
    for (size_t i = 0; i < m_meshes.size(); ++i) {
        const auto& mesh = m_meshes[i];
        MeshInventoryEntry entry{};
        entry.name = mesh.name.empty() ? ("mesh_" + std::to_string(i)) : mesh.name;
        entry.triangleCount = static_cast<uint32_t>(mesh.indices.size() / 3u);
        const bool sharedDuplicate =
            !mesh.geometryCacheKey.empty() &&
            !seenGeometryKeys.insert(mesh.geometryCacheKey).second;
        if (!sharedDuplicate) {
            entry.estimatedBytes =
                static_cast<uint64_t>(mesh.vertices.size()) * sizeof(PathTracerShaderTypes::SceneVertex) +
                static_cast<uint64_t>(mesh.indices.size() / 3u) * sizeof(simd::uint3);
            if (includePackedTriangles) {
                entry.estimatedBytes +=
                    static_cast<uint64_t>(mesh.indices.size() / 3u) *
                    sizeof(PathTracerShaderTypes::TriangleData);
            }
        } else {
            entry.estimatedBytes = 0u;
        }
        entries.push_back(std::move(entry));
    }
    return entries;
}

void SceneResources::setSoftwareRayTracingOverride(bool force) {
    setForceSoftwareBvh(force);
}

void SceneResources::addSphere(const simd::float3& center, float radius, uint32_t materialIndex) {
    if (m_sphereCount >= kMaxSpheres) {
        return;
    }

    if (materialIndex >= m_materialCount) {
        materialIndex = (m_materialCount == 0) ? 0u : (m_materialCount - 1);
    }

    SphereData sphere{};
    sphere.centerRadius = simd_make_float4(center, radius);
    sphere.materialIndex = simd_make_uint4(materialIndex, 0u, 0u, 0u);
    m_spheres[m_sphereCount++] = sphere;
    m_dirty = true;
}

bool SceneResources::setEnvironmentMap(const std::string& path) {
    if (!m_device) {
        return false;
    }

    if (path.empty()) {
        clearEnvironmentMap();
        return false;
    }

    EnvGpuHandles handles;
    bool success = reloadEnvironmentIfNeeded(path, &handles);
    if (!success) {
        return false;
    }

    return m_environmentTexture != nil;
}

bool SceneResources::reloadEnvironmentIfNeeded(const std::string& path, EnvGpuHandles* outHandles) {
    if (!m_device) {
        return false;
    }

    if (path.empty()) {
        const bool hadEnvironment = (m_environmentTexture != nil) || (m_environmentAliasCount > 0);
        if (hadEnvironment) {
            clearEnvironmentMap();
        }
        if (outHandles) {
            *outHandles = EnvGpuHandles{};
        }
        return true;
    }

    const std::string canonicalPath = CanonicalizePath(path);
    const bool needsReload =
        (m_environmentTexture == nil) || (canonicalPath != m_environmentPath);
    if (!needsReload) {
        if (outHandles) {
            outHandles->texture = m_environmentTexture;
            outHandles->conditionalAlias = m_environmentConditionalAliasBuffer;
            outHandles->marginalAlias = m_environmentMarginalAliasBuffer;
            outHandles->pdf = m_environmentPdfBuffer;
            outHandles->aliasCount = m_environmentAliasCount;
            outHandles->width = m_environmentWidth;
            outHandles->height = m_environmentHeight;
            outHandles->thresholdHeadSum = 0.0;
            outHandles->thresholdTotalSum = 0.0;
        }
        return true;
    }

    CFAbsoluteTime rebuildStart = CFAbsoluteTimeGetCurrent();
    NSString* nsPath = [NSString stringWithUTF8String:path.c_str()];
    NSURL* url = [NSURL fileURLWithPath:nsPath];
    MTKTextureLoader* loader = [[MTKTextureLoader alloc] initWithDevice:m_device];
    NSDictionary* options = @{MTKTextureLoaderOptionSRGB : @NO,
                              MTKTextureLoaderOptionTextureUsage :
                                  @(MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget),
                              MTKTextureLoaderOptionTextureStorageMode : @(MTLStorageModeShared),
                              MTKTextureLoaderOptionAllocateMipmaps : @NO,
                              MTKTextureLoaderOptionGenerateMipmaps : @NO};
    NSError* error = nil;
    id<MTLTexture> texture = [loader newTextureWithContentsOfURL:url options:options error:&error];
    if (!texture) {
        if (error) {
            NSLog(@"Failed to load environment map %@: %@", nsPath, error);
        } else {
            NSLog(@"Failed to load environment map %@", nsPath);
        }
        return false;
    }

    const NSUInteger width = texture.width;
    const NSUInteger height = texture.height;
    if (width == 0 || height == 0) {
        NSLog(@"Environment map %@ has invalid dimensions (%lu x %lu)",
              nsPath,
              static_cast<unsigned long>(width),
              static_cast<unsigned long>(height));
        return false;
    }

    id<MTLTexture> finalTexture = texture;
    if (texture.mipmapLevelCount <= 1 && m_commandQueue) {
        id<MTLTexture> mipTexture =
            BuildTextureWithBlitMips(m_device, m_commandQueue, texture, MTLStorageModePrivate);
        if (mipTexture) {
            finalTexture = mipTexture;
        }
    } else if (texture.storageMode != MTLStorageModePrivate && m_commandQueue) {
        id<MTLTexture> privateTexture = PromoteTextureToPrivate(m_device, m_commandQueue, texture);
        if (privateTexture) {
            finalTexture = privateTexture;
        }
    }

    std::vector<float> rgbaFloats;
    if (!CopyTextureToFloat(texture, rgbaFloats)) {
        NSLog(@"Failed to extract texels for environment map %@", nsPath);
        return false;
    }

    EnvGpuHandles tempHandles;
    if (!buildEnvironmentDistribution(rgbaFloats.data(),
                                      static_cast<uint32_t>(width),
                                      static_cast<uint32_t>(height),
                                      tempHandles)) {
        return false;
    }

    finalTexture.label = @"Environment Map";
    tempHandles.texture = finalTexture;

    const double elapsedMs = (CFAbsoluteTimeGetCurrent() - rebuildStart) * 1000.0;
    std::string fileName = std::filesystem::path(path).filename().string();
    NSLog(@"[EnvRebuild] file=\"%s\" size=%lux%lu mips=%lu aliasN=%u thrHead=%.5f thrSum=%.5f elapsedMs=%.2f",
          fileName.c_str(),
          static_cast<unsigned long>(width),
          static_cast<unsigned long>(height),
          static_cast<unsigned long>(finalTexture.mipmapLevelCount),
          tempHandles.aliasCount,
          tempHandles.thresholdHeadSum,
          tempHandles.thresholdTotalSum,
          elapsedMs);

    m_environmentTexture = tempHandles.texture;
    m_environmentConditionalAliasBuffer = tempHandles.conditionalAlias;
    m_environmentMarginalAliasBuffer = tempHandles.marginalAlias;
    m_environmentPdfBuffer = tempHandles.pdf;
    m_environmentAliasCount = tempHandles.aliasCount;
    m_environmentWidth = tempHandles.width;
    m_environmentHeight = tempHandles.height;
    m_environmentPath = canonicalPath;

    if (outHandles) {
        *outHandles = tempHandles;
    }
    return true;
}

void SceneResources::clearEnvironmentMap() {
    m_environmentTexture = nil;
    clearEnvironmentDistribution();
    m_environmentPath.clear();
}

void SceneResources::clearEnvironmentDistribution() {
    m_environmentConditionalAliasBuffer = nil;
    m_environmentMarginalAliasBuffer = nil;
    m_environmentPdfBuffer = nil;
    m_environmentAliasCount = 0;
    m_environmentWidth = 0;
    m_environmentHeight = 0;
}

bool SceneResources::buildEnvironmentDistribution(const float* rgba32,
                                                  uint32_t width,
                                                  uint32_t height,
                                                  EnvGpuHandles& outHandles) {
    outHandles = EnvGpuHandles{};

    if (!rgba32 || width == 0 || height == 0 || !m_device) {
        return false;
    }

    EnvImportanceDistribution distribution;
    std::string errorMessage;
    if (!BuildEnvImportanceDistribution(rgba32,
                                        width,
                                        height,
                                        &distribution,
                                        &errorMessage)) {
        if (!errorMessage.empty()) {
            NSLog(@"Environment importance sampler build failed: %s", errorMessage.c_str());
        }
        return false;
    }

    const size_t texelCount = static_cast<size_t>(distribution.aliasCount);
    if (texelCount == 0 || distribution.marginalAlias.empty() || distribution.texelPdf.empty()) {
        return false;
    }

    const NSUInteger conditionalBytes =
        static_cast<NSUInteger>(texelCount * sizeof(EnvironmentAliasEntry));
    const NSUInteger marginalBytes = static_cast<NSUInteger>(
        distribution.marginalAlias.size() * sizeof(EnvironmentAliasEntry));
    const NSUInteger pdfBytes =
        static_cast<NSUInteger>(distribution.texelPdf.size() * sizeof(float));
    if (conditionalBytes == 0 || marginalBytes == 0 || pdfBytes == 0) {
        return false;
    }

    std::vector<EnvironmentAliasEntry> conditionalEntries(texelCount);
    for (size_t i = 0; i < texelCount; ++i) {
        conditionalEntries[i].threshold =
            std::clamp(distribution.conditionalThreshold[i], 0.0f, 1.0f);
        conditionalEntries[i].alias = distribution.conditionalAlias[i];
        conditionalEntries[i].padding0 = 0;
        conditionalEntries[i].padding1 = 0;
    }

    std::vector<EnvironmentAliasEntry> marginalEntries(distribution.marginalAlias.size());
    for (size_t i = 0; i < distribution.marginalAlias.size(); ++i) {
        marginalEntries[i].threshold = std::clamp(distribution.marginalThreshold[i], 0.0f, 1.0f);
        marginalEntries[i].alias = distribution.marginalAlias[i];
        marginalEntries[i].padding0 = 0;
        marginalEntries[i].padding1 = 0;
    }

    MTLBufferHandle conditionalBuffer =
        [m_device newBufferWithLength:conditionalBytes options:MTLResourceStorageModeShared];
    MTLBufferHandle marginalBuffer =
        [m_device newBufferWithLength:marginalBytes options:MTLResourceStorageModeShared];
    MTLBufferHandle pdfBuffer =
        [m_device newBufferWithLength:pdfBytes options:MTLResourceStorageModeShared];

    if (!conditionalBuffer || !marginalBuffer || !pdfBuffer) {
        return false;
    }

    auto* conditionalPtr =
        reinterpret_cast<EnvironmentAliasEntry*>([conditionalBuffer contents]);
    memcpy(conditionalPtr, conditionalEntries.data(), conditionalBytes);

    auto* marginalPtr = reinterpret_cast<EnvironmentAliasEntry*>([marginalBuffer contents]);
    memcpy(marginalPtr, marginalEntries.data(), marginalBytes);

    memcpy([pdfBuffer contents], distribution.texelPdf.data(), pdfBytes);

    const ThresholdChecksums sums = ComputeThresholdChecksums(distribution.conditionalThreshold);
    outHandles.conditionalAlias = conditionalBuffer;
    outHandles.marginalAlias = marginalBuffer;
    outHandles.pdf = pdfBuffer;
    outHandles.aliasCount = distribution.aliasCount;
    outHandles.width = distribution.width;
    outHandles.height = distribution.height;
    outHandles.thresholdHeadSum = sums.head;
    outHandles.thresholdTotalSum = sums.total;
    return true;
}

void SceneResources::setForceSoftwareBvh(bool force) {
    if (m_forceSoftwareOverride == force) {
        return;
    }

    m_forceSoftwareOverride = force;

    if (m_sceneAccel) {
        m_sceneAccel->clear();
        m_sceneAccel.reset();
    }

    SceneAccelConfig accelConfig{};
    accelConfig.hardwareRaytracingSupported = m_supportsRaytracing && !m_forceSoftwareOverride;
    accelConfig.commandQueue = m_commandQueue;
    m_sceneAccel = CreateSceneAccel(accelConfig);

    m_intersectionProvider = IntersectionProvider{};
    m_meshInfoBuffer = nil;
    m_triangleBuffer = nil;
    m_dirty = true;
}

void SceneResources::addRectangle(const simd::float3& boundsMin,
                                  const simd::float3& boundsMax,
                                  uint32_t normalAxis,
                                  bool normalPositive,
                                  bool twoSided,
                                  uint32_t materialIndex) {
    if (m_rectangleCount >= kMaxRectangles) {
        return;
    }
    if (materialIndex >= m_materialCount) {
        materialIndex = (m_materialCount == 0) ? 0u : (m_materialCount - 1);
    }
    if (normalAxis > 2u) {
        normalAxis = 2u;
    }

    simd::float3 min = simd_make_float3(std::min(boundsMin.x, boundsMax.x),
                                        std::min(boundsMin.y, boundsMax.y),
                                        std::min(boundsMin.z, boundsMax.z));
    simd::float3 max = simd_make_float3(std::max(boundsMin.x, boundsMax.x),
                                        std::max(boundsMin.y, boundsMax.y),
                                        std::max(boundsMin.z, boundsMax.z));

    simd::float3 corner{};
    simd::float3 edgeU{};
    simd::float3 edgeV{};

    switch (normalAxis) {
        case 0: { // X constant
            float y0 = min.y;
            float y1 = max.y;
            float z0 = min.z;
            float z1 = max.z;
            edgeU = simd_make_float3(0.0f, y1 - y0, 0.0f);
            if (normalPositive) {
                corner = simd_make_float3(max.x, y0, z0);
                edgeV = simd_make_float3(0.0f, 0.0f, z1 - z0);
            } else {
                corner = simd_make_float3(min.x, y0, z1);
                edgeV = simd_make_float3(0.0f, 0.0f, z0 - z1);
            }
            break;
        }
        case 1: { // Y constant
            float x0 = min.x;
            float x1 = max.x;
            float z0 = min.z;
            float z1 = max.z;
            edgeU = simd_make_float3(x1 - x0, 0.0f, 0.0f);
            if (normalPositive) {
                corner = simd_make_float3(x0, max.y, z0);
                edgeV = simd_make_float3(0.0f, 0.0f, z1 - z0);
            } else {
                corner = simd_make_float3(x0, min.y, z1);
                edgeV = simd_make_float3(0.0f, 0.0f, z0 - z1);
            }
            break;
        }
        default: { // Z constant
            float x0 = min.x;
            float x1 = max.x;
            float y0 = min.y;
            float y1 = max.y;
            edgeU = simd_make_float3(x1 - x0, 0.0f, 0.0f);
            if (normalPositive) {
                corner = simd_make_float3(x0, y0, max.z);
                edgeV = simd_make_float3(0.0f, y1 - y0, 0.0f);
            } else {
                corner = simd_make_float3(x1, y0, min.z);
                edgeV = simd_make_float3(0.0f, y1 - y0, 0.0f);
                edgeU = simd_make_float3(x0 - x1, 0.0f, 0.0f);
            }
            break;
        }
    }

    simd::float3 desiredNormal = simd_make_float3(0.0f, 0.0f, 0.0f);
    switch (normalAxis) {
        case 0:
            desiredNormal = simd_make_float3(normalPositive ? 1.0f : -1.0f, 0.0f, 0.0f);
            break;
        case 1:
            desiredNormal = simd_make_float3(0.0f, normalPositive ? 1.0f : -1.0f, 0.0f);
            break;
        default:
            desiredNormal = simd_make_float3(0.0f, 0.0f, normalPositive ? 1.0f : -1.0f);
            break;
    }

    storeRectangleOriented(corner, edgeU, edgeV, twoSided, materialIndex, desiredNormal);
}

void SceneResources::addDisk(const simd::float3& center,
                             uint32_t normalAxis,
                             bool normalPositive,
                             float radius,
                             bool twoSided,
                             uint32_t materialIndex) {
    if (m_rectangleCount >= kMaxRectangles) {
        return;
    }
    if (materialIndex >= m_materialCount) {
        materialIndex = (m_materialCount == 0) ? 0u : (m_materialCount - 1);
    }
    if (normalAxis > 2u) {
        normalAxis = 2u;
    }

    radius = std::max(radius, 0.0f);
    if (radius <= std::numeric_limits<float>::min()) {
        return;
    }

    simd::float3 radiusAxisU{};
    simd::float3 radiusAxisV{};
    simd::float3 desiredNormal{};

    switch (normalAxis) {
        case 0:
            radiusAxisU = simd_make_float3(0.0f, radius, 0.0f);
            radiusAxisV = simd_make_float3(0.0f, 0.0f, normalPositive ? radius : -radius);
            desiredNormal = simd_make_float3(normalPositive ? 1.0f : -1.0f, 0.0f, 0.0f);
            break;
        case 1:
            radiusAxisU = simd_make_float3(radius, 0.0f, 0.0f);
            radiusAxisV = simd_make_float3(0.0f, 0.0f, normalPositive ? radius : -radius);
            desiredNormal = simd_make_float3(0.0f, normalPositive ? 1.0f : -1.0f, 0.0f);
            break;
        default:
            radiusAxisU = simd_make_float3(radius, 0.0f, 0.0f);
            radiusAxisV = simd_make_float3(0.0f, normalPositive ? radius : -radius, 0.0f);
            desiredNormal = simd_make_float3(0.0f, 0.0f, normalPositive ? 1.0f : -1.0f);
            break;
    }

    storeDiskOriented(center, radiusAxisU, radiusAxisV, twoSided, materialIndex, desiredNormal);
}

void SceneResources::addDirectionalLight(const simd::float3& directionToLight,
                                         const simd::float3& radiance,
                                         float selectionWeight) {
    const float dirLen = simd_length(directionToLight);
    const float emissionLuminance = RelativeLuminance(radiance);
    if (!(dirLen > 1.0e-6f) ||
        !std::isfinite(dirLen) ||
        !(emissionLuminance > 1.0e-6f) ||
        !std::isfinite(emissionLuminance)) {
        return;
    }

    PathTracerShaderTypes::LightPrimitive primitive{};
    const simd::float3 omegaL = directionToLight / dirLen;
    primitive.centroid = omegaL;
    primitive.area = 1.0f;
    primitive.emission = radiance;
    primitive.power = std::max(selectionWeight, 0.0f) * emissionLuminance;
    primitive.normal = -omegaL;
    primitive.triangleIndex = std::numeric_limits<uint32_t>::max();
    primitive.flags = PathTracerShaderTypes::kLightPrimitiveFlagDirectional;
    m_analyticLightPrimitives.push_back(primitive);
    m_dirty = true;
}

void SceneResources::addBox(const simd::float3& minCorner,
                            const simd::float3& maxCorner,
                            uint32_t materialIndex,
                            bool includeBottomFace,
                            bool twoSided) {
    addBoxTransformed(minCorner, maxCorner, materialIndex, matrix_identity_float4x4, includeBottomFace, twoSided);
}

void SceneResources::addBoxTransformed(const simd::float3& minCorner,
                                       const simd::float3& maxCorner,
                                       uint32_t materialIndex,
                                       const simd::float4x4& transform,
                                       bool includeBottomFace,
                                       bool twoSided) {
    if (materialIndex >= m_materialCount) {
        materialIndex = (m_materialCount == 0) ? 0u : (m_materialCount - 1);
    }

    simd::float3 min = simd_make_float3(std::min(minCorner.x, maxCorner.x),
                                        std::min(minCorner.y, maxCorner.y),
                                        std::min(minCorner.z, maxCorner.z));
    simd::float3 max = simd_make_float3(std::max(minCorner.x, maxCorner.x),
                                        std::max(minCorner.y, maxCorner.y),
                                        std::max(minCorner.z, maxCorner.z));

    struct Face {
        simd::float3 corner;
        simd::float3 edgeU;
        simd::float3 edgeV;
        simd::float3 normal;
        bool include;
    } faces[6] = {
        {simd_make_float3(max.x, min.y, min.z), simd_make_float3(0.0f, max.y - min.y, 0.0f), simd_make_float3(0.0f, 0.0f, max.z - min.z), simd_make_float3(1.0f, 0.0f, 0.0f), true},   // +X
        {simd_make_float3(min.x, min.y, max.z), simd_make_float3(0.0f, max.y - min.y, 0.0f), simd_make_float3(0.0f, 0.0f, min.z - max.z), simd_make_float3(-1.0f, 0.0f, 0.0f), true},   // -X
        {simd_make_float3(min.x, max.y, min.z), simd_make_float3(max.x - min.x, 0.0f, 0.0f), simd_make_float3(0.0f, 0.0f, max.z - min.z), simd_make_float3(0.0f, 1.0f, 0.0f), true},   // +Y
        {simd_make_float3(min.x, min.y, max.z), simd_make_float3(max.x - min.x, 0.0f, 0.0f), simd_make_float3(0.0f, 0.0f, min.z - max.z), simd_make_float3(0.0f, -1.0f, 0.0f), includeBottomFace}, // -Y
        {simd_make_float3(min.x, min.y, max.z), simd_make_float3(max.x - min.x, 0.0f, 0.0f), simd_make_float3(0.0f, max.y - min.y, 0.0f), simd_make_float3(0.0f, 0.0f, 1.0f), true},   // +Z
        {simd_make_float3(max.x, min.y, min.z), simd_make_float3(min.x - max.x, 0.0f, 0.0f), simd_make_float3(0.0f, max.y - min.y, 0.0f), simd_make_float3(0.0f, 0.0f, -1.0f), true},   // -Z
    };

    auto transformPoint = [&](const simd::float3& p) -> simd::float3 {
        simd::float4 result = simd_mul(transform, simd_make_float4(p, 1.0f));
        return simd_make_float3(result.x, result.y, result.z);
    };

    auto transformVector = [&](const simd::float3& v) -> simd::float3 {
        simd::float4 result = simd_mul(transform, simd_make_float4(v, 0.0f));
        return simd_make_float3(result.x, result.y, result.z);
    };

    for (const Face& face : faces) {
        if (!face.include) {
            continue;
        }
        simd::float3 corner = transformPoint(face.corner);
        simd::float3 edgeU = transformVector(face.edgeU);
        simd::float3 edgeV = transformVector(face.edgeV);
        simd::float3 desiredNormal = transformVector(face.normal);
        storeRectangleOriented(corner, edgeU, edgeV, twoSided, materialIndex, desiredNormal);
    }
}

uint32_t SceneResources::addMesh(const MeshVertex* vertices,
                                 uint32_t vertexCount,
                                 const uint32_t* indices,
                                 uint32_t indexCount,
                                 const simd::float4x4& localToWorld,
                                 const simd::float4x4& sourceLocalTransform,
                                 uint32_t materialIndex,
                                 std::string name,
                                 std::string geometryCacheKey) {
    if (!vertices || vertexCount == 0 || !indices || indexCount == 0) {
        NSLog(@"SceneResources::addMesh received empty mesh data");
        return std::numeric_limits<uint32_t>::max();
    }
    if (indexCount % 3 != 0) {
        NSLog(@"SceneResources::addMesh requires triangle indices (count must be divisible by 3)");
        return std::numeric_limits<uint32_t>::max();
    }

    uint32_t meshIndex = static_cast<uint32_t>(m_meshes.size());
    Mesh mesh{};
    mesh.vertices.assign(vertices, vertices + vertexCount);
    mesh.indices.assign(indices, indices + indexCount);
    mesh.geometryCacheKey = std::move(geometryCacheKey);
    mesh.localToWorld = localToWorld;
    mesh.defaultLocalToWorld = localToWorld;
    mesh.sourceLocalTransform = sourceLocalTransform;
    mesh.materialIndex = materialIndex;
    mesh.name = name.empty() ? DefaultMeshName(meshIndex) : std::move(name);
    m_meshes.push_back(std::move(mesh));
    m_dirty = true;
    return static_cast<uint32_t>(m_meshes.size() - 1);
}

void SceneResources::clear() {
    if (SceneLoadDiagnosticsVerboseEnabled()) {
        const MeshAggregateStats meshStats = CollectMeshAggregateStats(m_meshes);
        const SceneMemoryReport report = sceneMemoryReport();
        const uint64_t coreGpuBufferBytes =
            static_cast<uint64_t>(m_sphereBuffer ? m_sphereBuffer.length : 0u) +
            static_cast<uint64_t>(m_materialBuffer ? m_materialBuffer.length : 0u) +
            static_cast<uint64_t>(m_rectangleBuffer ? m_rectangleBuffer.length : 0u) +
            static_cast<uint64_t>(m_meshInfoBuffer ? m_meshInfoBuffer.length : 0u) +
            static_cast<uint64_t>(m_triangleBuffer ? m_triangleBuffer.length : 0u) +
            static_cast<uint64_t>(m_meshVertexBuffer ? m_meshVertexBuffer.length : 0u) +
            static_cast<uint64_t>(m_meshIndexBuffer ? m_meshIndexBuffer.length : 0u);
        uint32_t coreGpuBufferCount = 0u;
        coreGpuBufferCount += (m_sphereBuffer ? 1u : 0u);
        coreGpuBufferCount += (m_materialBuffer ? 1u : 0u);
        coreGpuBufferCount += (m_rectangleBuffer ? 1u : 0u);
        coreGpuBufferCount += (m_meshInfoBuffer ? 1u : 0u);
        coreGpuBufferCount += (m_triangleBuffer ? 1u : 0u);
        coreGpuBufferCount += (m_meshVertexBuffer ? 1u : 0u);
        coreGpuBufferCount += (m_meshIndexBuffer ? 1u : 0u);

        NSLog(@"[SceneDiag] clear begin materials=%u meshes=%u vertices=%llu indices=%llu triangles=%u "
              "coreGpuBuffers=%u meshGpuBuffers=%u coreGpuMiB=%.2f meshGpuMiB=%.2f "
              "geomMiB=%.2f texMiB=%.2f asMiB=%.2f",
              m_materialCount,
              meshStats.meshCount,
              static_cast<unsigned long long>(meshStats.vertexCount),
              static_cast<unsigned long long>(meshStats.indexCount),
              m_triangleCount,
              coreGpuBufferCount,
              meshStats.meshGpuBufferCount,
              BytesToMiB(coreGpuBufferBytes),
              BytesToMiB(meshStats.meshVertexBufferBytes + meshStats.meshIndexBufferBytes),
              BytesToMiB(report.geometryMemoryBytes),
              BytesToMiB(report.textureAllocatedMemoryBytes),
              BytesToMiB(m_lastBuildStats.blasAllocatedMemoryBytes + m_lastBuildStats.tlasAllocatedMemoryBytes));
    }

    m_sphereCount = 0;
    m_rectangleCount = 0;
    m_materialCount = 0;
    m_triangleCount = 0;
    m_primitiveCount = 0;
    m_materialNames.fill(std::string{});
    m_materialDefaults.fill(MaterialData{});
    m_sphereBuffer = nil;
    m_materialBuffer = nil;
    m_rectangleBuffer = nil;
    m_meshInfoBuffer = nil;
    m_triangleBuffer = nil;
    m_meshVertexBuffer = nil;
    m_meshIndexBuffer = nil;
    m_emissivePrimitivesBuffer = nil;
    m_intersectionProvider = IntersectionProvider{};
    m_assetPipelineReport = AssetPipelineReport{};
    for (auto& mesh : m_meshes) {
        mesh.vertexBuffer = nil;
        mesh.indexBuffer = nil;
    }
    m_meshes.clear();
    m_materialTextures.clear();
    m_materialSamplers.clear();
    m_materialTextureSamplerIndices.clear();
    m_materialTextureLabels.clear();
    m_textureInventory.clear();
    m_materialTextureIndex.clear();
    m_materialSamplerIndices.clear();
    m_deferredMaterialTextures.clear();
    m_packedMeshInfos.clear();
    m_packedTriangleData.clear();
    m_emissivePrimitives.clear();
    m_analyticLightPrimitives.clear();
    m_packedSceneVertices.clear();
    m_packedSceneIndices.clear();
    m_emissivePrimitiveCount = 0u;
    m_totalEmittedPower = 0.0f;
    m_skippedTexturedEmissivePrimitiveCount = 0u;
    m_skippedZeroPowerEmissivePrimitiveCount = 0u;
    m_incrementalUploadPhase = UploadPhase::Idle;
    m_incrementalUploadInProgress = false;
    m_incrementalUploadDeferredTextureIndex = 0u;
    m_incrementalUploadMeshIndex = 0u;
    m_incrementalUploadPackedOffset = 0u;
    m_stagedUploadMode = false;
    m_buffersUploadedViaStagedLoad = false;
    m_deferTriangleBufferForHardware = false;
    m_incrementalSharedMeshBuffers.clear();
    m_resolvedDeferredTextureIndices.clear();
    m_materialTextureInfoBuffer = nil;
    if (m_sceneAccel) {
        m_sceneAccel->clear();
    }
    m_stagedAccelBuildInput = SceneAccelBuildInput{};
    m_stagedAccelMeshInputs.clear();
    m_accelBuildInProgress = false;
    m_lastBuildStats = SceneAccelBuildStats{};
    resetTextureBudgetState();
    m_dirty = true;
    clearEnvironmentMap();
}

bool SceneResources::adoptCPUScene(SceneResources& src, std::string* errorMessage) {
    if (&src == this) {
        return true;
    }

    if (SceneLoadDiagnosticsVerboseEnabled()) {
        const MeshAggregateStats srcMeshStats = CollectMeshAggregateStats(src.m_meshes);
        const uint64_t srcPackedMeshInfoBytes =
            static_cast<uint64_t>(src.m_packedMeshInfos.size()) * sizeof(PathTracerShaderTypes::MeshInfo);
        const uint64_t srcPackedTriangleBytes =
            static_cast<uint64_t>(src.m_packedTriangleData.size()) * sizeof(PathTracerShaderTypes::TriangleData);
        const uint64_t srcPackedVertexBytes =
            static_cast<uint64_t>(src.m_packedSceneVertices.size()) * sizeof(PathTracerShaderTypes::SceneVertex);
        const uint64_t srcPackedIndexBytes =
            static_cast<uint64_t>(src.m_packedSceneIndices.size()) * sizeof(simd::uint3);

        NSLog(@"[SceneDiag] adoptCPUScene source materials=%u meshes=%u vertices=%llu indices=%llu triangles=%llu "
              "packedMeshInfoMiB=%.2f packedTriMiB=%.2f packedVertMiB=%.2f packedIdxMiB=%.2f deferredTextures=%zu",
              src.m_materialCount,
              srcMeshStats.meshCount,
              static_cast<unsigned long long>(srcMeshStats.vertexCount),
              static_cast<unsigned long long>(srcMeshStats.indexCount),
              static_cast<unsigned long long>(srcMeshStats.triangleCount),
              BytesToMiB(srcPackedMeshInfoBytes),
              BytesToMiB(srcPackedTriangleBytes),
              BytesToMiB(srcPackedVertexBytes),
              BytesToMiB(srcPackedIndexBytes),
              src.m_deferredMaterialTextures.size());
    }

    clear();

    m_spheres = src.m_spheres;
    m_materialCount = src.m_materialCount;
    m_sphereCount = src.m_sphereCount;
    m_rectangleCount = src.m_rectangleCount;
    m_rectangles = src.m_rectangles;
    m_materialNames = std::move(src.m_materialNames);
    m_meshes = std::move(src.m_meshes);
    for (auto& mesh : m_meshes) {
        mesh.vertexBuffer = nil;
        mesh.indexBuffer = nil;
    }
    m_packedMeshInfos = std::move(src.m_packedMeshInfos);
    m_packedTriangleData = std::move(src.m_packedTriangleData);
    m_emissivePrimitives = std::move(src.m_emissivePrimitives);
    m_analyticLightPrimitives = std::move(src.m_analyticLightPrimitives);
    m_packedSceneVertices = std::move(src.m_packedSceneVertices);
    m_packedSceneIndices = std::move(src.m_packedSceneIndices);
    m_assetPipelineReport = std::move(src.m_assetPipelineReport);
    m_emissivePrimitiveCount = src.m_emissivePrimitiveCount;
    m_totalEmittedPower = src.m_totalEmittedPower;
    m_skippedTexturedEmissivePrimitiveCount = src.m_skippedTexturedEmissivePrimitiveCount;
    m_skippedZeroPowerEmissivePrimitiveCount = src.m_skippedZeroPowerEmissivePrimitiveCount;
    m_emissivePrimitivesBuffer = nil;
    for (uint32_t i = 0; i < m_materialCount; ++i) {
        m_materials[i] = src.m_materials[i];
        m_materialDefaults[i] = src.m_materialDefaults[i];
    }
    m_deferredMaterialTextures = std::move(src.m_deferredMaterialTextures);
    m_resolvedDeferredTextureIndices.clear();

    m_triangleCount = 0;
    m_primitiveCount = 0;
    for (const auto& mesh : m_meshes) {
        m_triangleCount += static_cast<uint32_t>(mesh.indices.size() / 3u);
        m_primitiveCount += static_cast<uint32_t>(mesh.indices.size() / 3u);
    }
    if (!m_packedTriangleData.empty()) {
        m_triangleCount = static_cast<uint32_t>(m_packedTriangleData.size());
    }
    if (m_emissivePrimitives.empty() && !m_meshes.empty()) {
        rebuildEmissivePrimitiveInventory();
    } else {
        uploadEmissivePrimitiveBuffer();
    }

    if (SceneLoadDiagnosticsVerboseEnabled()) {
        const MeshAggregateStats dstMeshStats = CollectMeshAggregateStats(m_meshes);
        const uint64_t dstPackedMeshInfoBytes =
            static_cast<uint64_t>(m_packedMeshInfos.size()) * sizeof(PathTracerShaderTypes::MeshInfo);
        const uint64_t dstPackedTriangleBytes =
            static_cast<uint64_t>(m_packedTriangleData.size()) * sizeof(PathTracerShaderTypes::TriangleData);
        const uint64_t dstPackedVertexBytes =
            static_cast<uint64_t>(m_packedSceneVertices.size()) * sizeof(PathTracerShaderTypes::SceneVertex);
        const uint64_t dstPackedIndexBytes =
            static_cast<uint64_t>(m_packedSceneIndices.size()) * sizeof(simd::uint3);

        NSLog(@"[SceneDiag] adoptCPUScene ready materials=%u meshes=%u vertices=%llu indices=%llu triangles=%u "
              "packedMeshInfoMiB=%.2f packedTriMiB=%.2f packedVertMiB=%.2f packedIdxMiB=%.2f deferredTextures=%zu",
              m_materialCount,
              dstMeshStats.meshCount,
              static_cast<unsigned long long>(dstMeshStats.vertexCount),
              static_cast<unsigned long long>(dstMeshStats.indexCount),
              m_triangleCount,
              BytesToMiB(dstPackedMeshInfoBytes),
              BytesToMiB(dstPackedTriangleBytes),
              BytesToMiB(dstPackedVertexBytes),
              BytesToMiB(dstPackedIndexBytes),
              m_deferredMaterialTextures.size());
    }

    m_dirty = true;
    return true;
}

bool SceneResources::setMeshTransform(uint32_t meshIndex, const simd::float4x4& localToWorld) {
    if (meshIndex >= m_meshes.size()) {
        return false;
    }
    m_meshes[meshIndex].localToWorld = localToWorld;
    m_dirty = true;
    return true;
}

bool SceneResources::resetMeshTransform(uint32_t meshIndex) {
    if (meshIndex >= m_meshes.size()) {
        return false;
    }
    m_meshes[meshIndex].localToWorld = m_meshes[meshIndex].defaultLocalToWorld;
    m_dirty = true;
    return true;
}

const simd::float4x4& SceneResources::meshTransform(uint32_t meshIndex) const {
    static const simd::float4x4 kIdentity = matrix_identity_float4x4;
    if (meshIndex >= m_meshes.size()) {
        return kIdentity;
    }
    return m_meshes[meshIndex].localToWorld;
}

const std::string& SceneResources::meshName(uint32_t meshIndex) const {
    static const std::string kEmptyName;
    if (meshIndex >= m_meshes.size()) {
        return kEmptyName;
    }
    return m_meshes[meshIndex].name;
}

void SceneResources::uploadMaterialToGpu(uint32_t index) {
    if (!m_device || index >= m_materialCount) {
        return;
    }
    const NSUInteger materialBytes =
        static_cast<NSUInteger>(m_materialCount) * sizeof(MaterialData);
    if (materialBytes == 0) {
        return;
    }
    if (!m_materialBuffer || m_materialBuffer.length < materialBytes) {
        m_materialBuffer = [m_device newBufferWithLength:materialBytes
                                                 options:MTLResourceStorageModeShared];
    }
    if (!m_materialBuffer) {
        return;
    }
    uint8_t* dst = reinterpret_cast<uint8_t*>([m_materialBuffer contents]);
    if (!dst) {
        return;
    }
    memcpy(dst + static_cast<NSUInteger>(index) * sizeof(MaterialData),
           &m_materials[index],
           sizeof(MaterialData));
}

void SceneResources::uploadBuffers() {
    if (!m_device) {
        return;
    }

    const NSUInteger sphereBytes = static_cast<NSUInteger>(m_sphereCount) * sizeof(SphereData);
    const NSUInteger materialBytes = static_cast<NSUInteger>(m_materialCount) * sizeof(MaterialData);
    const NSUInteger rectangleBytes = static_cast<NSUInteger>(m_rectangleCount) * sizeof(RectData);
    const MeshAggregateStats meshStats = CollectMeshAggregateStats(m_meshes);
    const uint64_t packedMeshInfoBytes =
        static_cast<uint64_t>(m_packedMeshInfos.size()) * sizeof(PathTracerShaderTypes::MeshInfo);
    const uint64_t packedTriangleBytes =
        static_cast<uint64_t>(m_packedTriangleData.size()) * sizeof(PathTracerShaderTypes::TriangleData);
    const uint64_t packedVertexBytes =
        static_cast<uint64_t>(m_packedSceneVertices.size()) * sizeof(PathTracerShaderTypes::SceneVertex);
    const uint64_t packedIndexBytes =
        static_cast<uint64_t>(m_packedSceneIndices.size()) * sizeof(simd::uint3);

    if (SceneLoadDiagnosticsVerboseEnabled()) {
        const uint64_t totalRequestedBytes =
            static_cast<uint64_t>(sphereBytes) +
            static_cast<uint64_t>(materialBytes) +
            static_cast<uint64_t>(rectangleBytes) +
            meshStats.vertexBytes +
            meshStats.indexBytes +
            packedMeshInfoBytes +
            packedTriangleBytes +
            packedVertexBytes +
            packedIndexBytes;
        NSLog(@"[SceneDiag] uploadBuffers request materials=%u meshes=%u vertices=%llu indices=%llu triangles=%u "
              "bytes{spheres=%llu materials=%llu rectangles=%llu meshVertices=%llu meshIndices=%llu "
              "packedMeshInfo=%llu packedTriangles=%llu packedVertices=%llu packedIndices=%llu total=%llu} "
              "deferredTextures=%zu",
              m_materialCount,
              meshStats.meshCount,
              static_cast<unsigned long long>(meshStats.vertexCount),
              static_cast<unsigned long long>(meshStats.indexCount),
              m_triangleCount,
              static_cast<unsigned long long>(sphereBytes),
              static_cast<unsigned long long>(materialBytes),
              static_cast<unsigned long long>(rectangleBytes),
              static_cast<unsigned long long>(meshStats.vertexBytes),
              static_cast<unsigned long long>(meshStats.indexBytes),
              static_cast<unsigned long long>(packedMeshInfoBytes),
              static_cast<unsigned long long>(packedTriangleBytes),
              static_cast<unsigned long long>(packedVertexBytes),
              static_cast<unsigned long long>(packedIndexBytes),
              static_cast<unsigned long long>(totalRequestedBytes),
              m_deferredMaterialTextures.size());
    }

    if (sphereBytes > 0) {
        if (!m_sphereBuffer || m_sphereBuffer.length < sphereBytes) {
            m_sphereBuffer = [m_device newBufferWithLength:sphereBytes
                                                   options:MTLResourceStorageModeShared];
        }
        if (m_sphereBuffer) {
            memcpy([m_sphereBuffer contents], m_spheres.data(), sphereBytes);
        }
    } else {
        m_sphereBuffer = nil;
    }

    if (materialBytes > 0) {
        if (!m_materialBuffer || m_materialBuffer.length < materialBytes) {
            m_materialBuffer = [m_device newBufferWithLength:materialBytes
                                                     options:MTLResourceStorageModeShared];
        }
        if (m_materialBuffer) {
            memcpy([m_materialBuffer contents], m_materials.data(), materialBytes);
        }
    } else {
        m_materialBuffer = nil;
    }

    uploadRectangles();
    uploadMeshes();
    uploadEmissivePrimitiveBuffer();
}

void SceneResources::populateSceneAccelBuildInput(SceneAccelBuildInput& buildInput,
                                                  std::vector<SceneAccelMeshInput>& meshInputs) const {
    buildInput.device = m_device;
    buildInput.commandQueue = m_commandQueue;
    buildInput.spheres = m_spheres.data();
    buildInput.sphereCount = m_sphereCount;
    meshInputs.clear();
    meshInputs.reserve(m_meshes.size());
    for (const auto& mesh : m_meshes) {
        SceneAccelMeshInput meshInput{};
        meshInput.vertexBuffer = mesh.vertexBuffer;
        meshInput.indexBuffer = mesh.indexBuffer;
        meshInput.vertexStride = sizeof(MeshVertex);
        meshInput.vertexCount = static_cast<uint32_t>(mesh.vertices.size());
        meshInput.indexCount = static_cast<uint32_t>(mesh.indices.size());
        meshInput.localToWorldTransform = mesh.localToWorld;
        meshInput.materialIndex = mesh.materialIndex;
        meshInputs.push_back(meshInput);
    }
    buildInput.meshes = meshInputs.empty() ? nullptr : meshInputs.data();
    buildInput.meshCount = static_cast<uint32_t>(meshInputs.size());
}

void SceneResources::buildCPUPackedSceneData() {
    buildPackedSceneCpuData();
}

void SceneResources::rebuildEmissivePrimitiveInventory() {
    m_emissivePrimitives.clear();
    m_emissivePrimitiveCount = 0u;
    m_totalEmittedPower = 0.0f;
    m_skippedTexturedEmissivePrimitiveCount = 0u;
    m_skippedZeroPowerEmissivePrimitiveCount = 0u;

    uint32_t triangleOffset = 0u;
    for (const auto& mesh : m_meshes) {
        const uint32_t triangleCount = static_cast<uint32_t>(mesh.indices.size() / 3u);
        if (triangleCount == 0u || m_materialCount == 0u) {
            triangleOffset += triangleCount;
            continue;
        }

        const uint32_t materialIndex = std::min<uint32_t>(mesh.materialIndex, m_materialCount - 1u);
        const MaterialData& material = m_materials[materialIndex];
        const simd::float3 emission = simd_make_float3(material.emission.x,
                                                       material.emission.y,
                                                       material.emission.z);
        const float emissionLuminance = RelativeLuminance(emission);
        const bool hasEmissiveTexture = material.textureIndices1.x != kInvalidTextureIndex;
        if (hasEmissiveTexture) {
            if (emissionLuminance > 1.0e-6f) {
                m_skippedTexturedEmissivePrimitiveCount += triangleCount;
            }
            triangleOffset += triangleCount;
            continue;
        }
        if (!(emissionLuminance > 1.0e-6f) || !std::isfinite(emissionLuminance)) {
            triangleOffset += triangleCount;
            continue;
        }

        for (uint32_t tri = 0u; tri < triangleCount; ++tri) {
            const uint32_t base = tri * 3u;
            if (base + 2u >= mesh.indices.size()) {
                continue;
            }

            const uint32_t i0 = mesh.indices[base + 0u];
            const uint32_t i1 = mesh.indices[base + 1u];
            const uint32_t i2 = mesh.indices[base + 2u];
            if (i0 >= mesh.vertices.size() || i1 >= mesh.vertices.size() || i2 >= mesh.vertices.size()) {
                continue;
            }

            const simd::float3 v0 = TransformPoint(mesh.localToWorld, mesh.vertices[i0].position);
            const simd::float3 v1 = TransformPoint(mesh.localToWorld, mesh.vertices[i1].position);
            const simd::float3 v2 = TransformPoint(mesh.localToWorld, mesh.vertices[i2].position);
            const simd::float3 faceNormalRaw = simd_cross(v1 - v0, v2 - v0);
            const float faceNormalLen = simd_length(faceNormalRaw);
            const float area = 0.5f * faceNormalLen;
            const float power = area * emissionLuminance;
            if (!(faceNormalLen > 1.0e-6f) ||
                !std::isfinite(faceNormalLen) ||
                !(power > 1.0e-6f) ||
                !std::isfinite(power)) {
                ++m_skippedZeroPowerEmissivePrimitiveCount;
                continue;
            }

            LightPrimitive primitive{};
            primitive.centroid = (v0 + v1 + v2) / 3.0f;
            primitive.area = area;
            primitive.emission = emission;
            primitive.power = power;
            primitive.normal = faceNormalRaw / faceNormalLen;
            primitive.triangleIndex = triangleOffset + tri;
            primitive.edge1 = v1 - v0;
            primitive.edge2 = v2 - v0;
            m_emissivePrimitives.push_back(primitive);
            m_totalEmittedPower += power;
        }

        triangleOffset += triangleCount;
    }

    for (const auto& primitive : m_analyticLightPrimitives) {
        if (!(primitive.power > 0.0f) || !std::isfinite(primitive.power)) {
            continue;
        }
        m_emissivePrimitives.push_back(primitive);
        m_totalEmittedPower += primitive.power;
    }

    m_emissivePrimitiveCount = static_cast<uint32_t>(m_emissivePrimitives.size());
    if (m_totalEmittedPower > 0.0f && std::isfinite(m_totalEmittedPower)) {
        float cumulativePower = 0.0f;
        for (auto& primitive : m_emissivePrimitives) {
            cumulativePower += std::max(primitive.power, 0.0f);
            primitive.cumulativePower = std::min(cumulativePower / m_totalEmittedPower, 1.0f);
        }
        if (!m_emissivePrimitives.empty()) {
            m_emissivePrimitives.back().cumulativePower = 1.0f;
        }
    }
    uploadEmissivePrimitiveBuffer();
}

void SceneResources::uploadEmissivePrimitiveBuffer() {
    if (!m_device || m_emissivePrimitives.empty()) {
        m_emissivePrimitivesBuffer = nil;
        return;
    }

    const NSUInteger primitiveBytes =
        static_cast<NSUInteger>(m_emissivePrimitives.size() * sizeof(LightPrimitive));
    if (!m_emissivePrimitivesBuffer || m_emissivePrimitivesBuffer.length < primitiveBytes) {
        m_emissivePrimitivesBuffer =
            [m_device newBufferWithLength:primitiveBytes options:MTLResourceStorageModeShared];
    }
    if (m_emissivePrimitivesBuffer) {
        std::memcpy([m_emissivePrimitivesBuffer contents],
                    m_emissivePrimitives.data(),
                    primitiveBytes);
    }
}

void SceneResources::buildPackedSceneCpuData() {
    m_packedMeshInfos.clear();
    m_packedTriangleData.clear();
    m_packedSceneVertices.clear();
    m_packedSceneIndices.clear();
    m_packedMeshInfos.reserve(m_meshes.size());
    m_packedTriangleData.reserve(m_triangleCount);
    size_t totalVertexCount = 0;
    size_t totalTriangleCount = 0;
    for (const auto& mesh : m_meshes) {
        totalVertexCount += mesh.vertices.size();
        totalTriangleCount += mesh.indices.size() / 3u;
    }
    m_packedSceneVertices.reserve(totalVertexCount);
    m_packedSceneIndices.reserve(totalTriangleCount);
    uint32_t triangleOffset = 0;
    struct SharedPackedGeometryRange {
        uint32_t vertexOffset = 0u;
        uint32_t vertexCount = 0u;
        uint32_t indexOffset = 0u;
        uint32_t indexCount = 0u;
    };
    std::unordered_map<std::string, SharedPackedGeometryRange> sharedPackedGeometry;
    sharedPackedGeometry.reserve(m_meshes.size());

    for (size_t meshListIndex = 0; meshListIndex < m_meshes.size(); ++meshListIndex) {
        PumpEventsEvery(meshListIndex, 4u);
        const auto& mesh = m_meshes[meshListIndex];
        PathTracerShaderTypes::MeshInfo info{};
        info.materialIndex = mesh.materialIndex;
        info.triangleOffset = triangleOffset;
        uint32_t triangleCount = static_cast<uint32_t>(mesh.indices.size() / 3u);
        info.triangleCount = triangleCount;
        const bool canSharePackedGeometry = !mesh.geometryCacheKey.empty();
        bool reusedPackedGeometry = false;
        SharedPackedGeometryRange sharedRange{};
        if (canSharePackedGeometry) {
            auto sharedIt = sharedPackedGeometry.find(mesh.geometryCacheKey);
            if (sharedIt != sharedPackedGeometry.end()) {
                sharedRange = sharedIt->second;
                reusedPackedGeometry = true;
            }
        }
        if (reusedPackedGeometry) {
            info.vertexOffset = sharedRange.vertexOffset;
            info.vertexCount = sharedRange.vertexCount;
            info.indexOffset = sharedRange.indexOffset;
            info.indexCount = sharedRange.indexCount;
        } else {
            info.vertexOffset = static_cast<uint32_t>(m_packedSceneVertices.size());
            info.vertexCount = static_cast<uint32_t>(mesh.vertices.size());
            info.indexOffset = static_cast<uint32_t>(m_packedSceneIndices.size());
            info.indexCount = triangleCount;
        }
        info.localToWorld = mesh.localToWorld;
        info.worldToLocal = simd_inverse(mesh.localToWorld);
        m_packedMeshInfos.push_back(info);

        if (!reusedPackedGeometry) {
            m_packedSceneVertices.reserve(m_packedSceneVertices.size() + mesh.vertices.size());
            for (size_t vertexIndex = 0; vertexIndex < mesh.vertices.size(); ++vertexIndex) {
                PumpEventsEvery(vertexIndex);
                const auto& vertex = mesh.vertices[vertexIndex];
                PathTracerShaderTypes::SceneVertex packed{};
                packed.position = simd_make_float4(vertex.position, 1.0f);
                packed.normal = simd_make_float4(vertex.normal, 0.0f);
                packed.tangent = vertex.tangent;
                packed.uv = simd_make_float4(vertex.uv.x, vertex.uv.y, vertex.uv1.x, vertex.uv1.y);
                m_packedSceneVertices.push_back(packed);
            }

            m_packedSceneIndices.reserve(m_packedSceneIndices.size() + triangleCount);
        }

        for (uint32_t tri = 0; tri < triangleCount; ++tri) {
            PumpEventsEvery(tri, 8192u);
            uint32_t base = tri * 3u;
            PathTracerShaderTypes::TriangleData triData{};
            if (base + 2u >= mesh.indices.size()) {
                m_packedTriangleData.push_back(triData);
                if (!reusedPackedGeometry) {
                    simd::uint3 packed = simd_make_uint3(info.vertexOffset, info.vertexOffset, info.vertexOffset);
                    m_packedSceneIndices.push_back(packed);
                }
                continue;
            }
            uint32_t i0 = mesh.indices[base + 0u];
            uint32_t i1 = mesh.indices[base + 1u];
            uint32_t i2 = mesh.indices[base + 2u];
            if (i0 >= mesh.vertices.size() ||
                i1 >= mesh.vertices.size() ||
                i2 >= mesh.vertices.size()) {
                m_packedTriangleData.push_back(triData);
                if (!reusedPackedGeometry) {
                    simd::uint3 packed = simd_make_uint3(info.vertexOffset, info.vertexOffset, info.vertexOffset);
                    m_packedSceneIndices.push_back(packed);
                }
                continue;
            }

            const MeshVertex& v0 = mesh.vertices[i0];
            const MeshVertex& v1 = mesh.vertices[i1];
            const MeshVertex& v2 = mesh.vertices[i2];

            triData.v0 = simd_make_float4(v0.position, 0.0f);
            triData.v1 = simd_make_float4(v1.position, 0.0f);
            triData.v2 = simd_make_float4(v2.position, 0.0f);
            uint32_t matIndex = mesh.materialIndex;
            if (m_materialCount > 0) {
                matIndex = std::min<uint32_t>(matIndex, m_materialCount - 1);
            } else {
                matIndex = 0u;
            }
            triData.metadata = simd_make_uint4(matIndex,
                                               static_cast<uint32_t>(meshListIndex),
                                               0u,
                                               0u);
            m_packedTriangleData.push_back(triData);

            if (!reusedPackedGeometry) {
                simd::uint3 packedIdx = simd_make_uint3(info.vertexOffset + i0,
                                                        info.vertexOffset + i1,
                                                        info.vertexOffset + i2);
                m_packedSceneIndices.push_back(packedIdx);
            }
        }
        if (!reusedPackedGeometry && canSharePackedGeometry) {
            sharedPackedGeometry.emplace(mesh.geometryCacheKey,
                                         SharedPackedGeometryRange{
                                             info.vertexOffset,
                                             info.vertexCount,
                                             info.indexOffset,
                                             info.indexCount,
                                         });
        }
        triangleOffset += triangleCount;
    }

    m_triangleCount = triangleOffset;
    rebuildEmissivePrimitiveInventory();
}

void SceneResources::uploadPackedSceneBuffers(bool includeTriangleData) {
    if (!m_device) {
        return;
    }

    if (SceneLoadDiagnosticsEnabled()) {
        const uint64_t infoBytes =
            static_cast<uint64_t>(m_packedMeshInfos.size()) * sizeof(PathTracerShaderTypes::MeshInfo);
        const uint64_t triangleBytes =
            static_cast<uint64_t>(m_packedTriangleData.size()) * sizeof(PathTracerShaderTypes::TriangleData);
        const uint64_t vertexBytes =
            static_cast<uint64_t>(m_packedSceneVertices.size()) * sizeof(PathTracerShaderTypes::SceneVertex);
        const uint64_t indexBytes =
            static_cast<uint64_t>(m_packedSceneIndices.size()) * sizeof(simd::uint3);
        NSLog(@"[SceneDiag] uploadPackedSceneBuffers request meshInfos=%zu triangles=%zu vertices=%zu packedIndices=%zu includeTriangleData=%d "
              "bytes{meshInfo=%llu triangles=%llu vertices=%llu indices=%llu}",
              m_packedMeshInfos.size(),
              m_packedTriangleData.size(),
              m_packedSceneVertices.size(),
              m_packedSceneIndices.size(),
              includeTriangleData ? 1 : 0,
              static_cast<unsigned long long>(infoBytes),
              static_cast<unsigned long long>(triangleBytes),
              static_cast<unsigned long long>(vertexBytes),
              static_cast<unsigned long long>(indexBytes));
    }

    if (!m_packedMeshInfos.empty()) {
        const NSUInteger infoBytes =
            static_cast<NSUInteger>(m_packedMeshInfos.size() * sizeof(PathTracerShaderTypes::MeshInfo));
        if (!m_meshInfoBuffer || m_meshInfoBuffer.length < infoBytes) {
            m_meshInfoBuffer =
                [m_device newBufferWithLength:infoBytes options:MTLResourceStorageModeShared];
        }
        if (m_meshInfoBuffer) {
            memcpy([m_meshInfoBuffer contents], m_packedMeshInfos.data(), infoBytes);
        }
    } else {
        m_meshInfoBuffer = nil;
    }

    if (includeTriangleData && !m_packedTriangleData.empty()) {
        const NSUInteger triangleBytes =
            static_cast<NSUInteger>(m_packedTriangleData.size() * sizeof(PathTracerShaderTypes::TriangleData));
        if (!m_triangleBuffer || m_triangleBuffer.length < triangleBytes) {
            m_triangleBuffer =
                [m_device newBufferWithLength:triangleBytes options:MTLResourceStorageModeShared];
        }
        if (m_triangleBuffer) {
            memcpy([m_triangleBuffer contents], m_packedTriangleData.data(), triangleBytes);
        }
    } else {
        m_triangleBuffer = nil;
    }

    if (!m_packedSceneVertices.empty()) {
        const NSUInteger vertexBytes =
            static_cast<NSUInteger>(m_packedSceneVertices.size() * sizeof(PathTracerShaderTypes::SceneVertex));
        if (!m_meshVertexBuffer || m_meshVertexBuffer.length < vertexBytes) {
            m_meshVertexBuffer =
                [m_device newBufferWithLength:vertexBytes options:MTLResourceStorageModeShared];
        }
        if (m_meshVertexBuffer) {
            memcpy([m_meshVertexBuffer contents], m_packedSceneVertices.data(), vertexBytes);
        }
    } else {
        m_meshVertexBuffer = nil;
    }

    if (!m_packedSceneIndices.empty()) {
        const NSUInteger indexBytes =
            static_cast<NSUInteger>(m_packedSceneIndices.size() * sizeof(simd::uint3));
        if (!m_meshIndexBuffer || m_meshIndexBuffer.length < indexBytes) {
            m_meshIndexBuffer =
                [m_device newBufferWithLength:indexBytes options:MTLResourceStorageModeShared];
        }
        if (m_meshIndexBuffer) {
            memcpy([m_meshIndexBuffer contents], m_packedSceneIndices.data(), indexBytes);
        }
    } else {
        m_meshIndexBuffer = nil;
    }
}

bool SceneResources::replaceDeferredTexturePlaceholder(uint32_t placeholderIndex, uint32_t resolvedIndex) {
    auto patchMaterial = [&](MaterialData& material) {
        if (material.textureIndices0.x == placeholderIndex) material.textureIndices0.x = resolvedIndex;
        if (material.textureIndices0.y == placeholderIndex) material.textureIndices0.y = resolvedIndex;
        if (material.textureIndices0.z == placeholderIndex) material.textureIndices0.z = resolvedIndex;
        if (material.textureIndices0.w == placeholderIndex) material.textureIndices0.w = resolvedIndex;
        if (material.textureIndices1.x == placeholderIndex) material.textureIndices1.x = resolvedIndex;
        if (material.textureIndices1.y == placeholderIndex) material.textureIndices1.y = resolvedIndex;
        if (material.textureIndices1.z == placeholderIndex) material.textureIndices1.z = resolvedIndex;
        if (material.textureIndices1.w == placeholderIndex) material.textureIndices1.w = resolvedIndex;
    };
    for (uint32_t i = 0; i < m_materialCount; ++i) {
        patchMaterial(m_materials[i]);
        patchMaterial(m_materialDefaults[i]);
    }
    return true;
}

SceneAccelBuildStats SceneResources::estimateAccelerationStructureBuildStats() {
    if (!m_sceneAccel || !m_device) {
        return SceneAccelBuildStats{};
    }
    uploadBuffers();
    SceneAccelBuildInput buildInput{};
    std::vector<SceneAccelMeshInput> meshInputs;
    populateSceneAccelBuildInput(buildInput, meshInputs);
    return m_sceneAccel->estimateBuildStats(buildInput);
}

bool SceneResources::beginIncrementalUpload() {
    if (!m_device) {
        return false;
    }
    if (SceneLoadDiagnosticsVerboseEnabled()) {
        const MeshAggregateStats meshStats = CollectMeshAggregateStats(m_meshes);
        NSLog(@"[SceneDiag] beginIncrementalUpload materials=%u meshes=%u vertices=%llu indices=%llu triangles=%u deferredTextures=%zu",
              m_materialCount,
              meshStats.meshCount,
              static_cast<unsigned long long>(meshStats.vertexCount),
              static_cast<unsigned long long>(meshStats.indexCount),
              m_triangleCount,
              m_deferredMaterialTextures.size());
    }
    m_incrementalUploadPhase = UploadPhase::ResolveTextures;
    m_incrementalUploadInProgress = true;
    m_incrementalUploadDeferredTextureIndex = 0u;
    m_incrementalUploadMeshIndex = 0u;
    m_incrementalUploadPackedOffset = 0u;
    m_stagedUploadMode = true;
    m_deferTriangleBufferForHardware =
        hardwareRaytracingEnabled() && !KeepTriangleBufferForHardware();
    m_incrementalSharedMeshBuffers.clear();
    return true;
}

bool SceneResources::stepIncrementalUpload(bool* outComplete) {
    if (!m_device) {
        return false;
    }
    if (outComplete) {
        *outComplete = false;
    }
    if (!m_incrementalUploadInProgress) {
        if (outComplete) {
            *outComplete = true;
        }
        return true;
    }

    constexpr size_t kMeshesPerStep = 1u;
    constexpr NSUInteger kPackedChunkBytes = 2u * 1024u * 1024u;

    auto advancePackedBuffer = [&](id<MTLBuffer> __strong* buffer,
                                   const void* src,
                                   NSUInteger totalBytes,
                                   UploadPhase nextPhase) {
        if (!buffer) {
            return;
        }
        if (!*buffer || (*buffer).length < totalBytes) {
            *buffer = totalBytes > 0u
                ? [m_device newBufferWithLength:totalBytes options:MTLResourceStorageModeShared]
                : nil;
        }
        if (totalBytes == 0u || !*buffer || !src) {
            m_incrementalUploadPackedOffset = 0u;
            m_incrementalUploadPhase = nextPhase;
            return;
        }

        const NSUInteger remaining = totalBytes - m_incrementalUploadPackedOffset;
        const NSUInteger chunkBytes = std::min<NSUInteger>(remaining, kPackedChunkBytes);
        uint8_t* dst = reinterpret_cast<uint8_t*>((*buffer).contents);
        const uint8_t* srcBytes = reinterpret_cast<const uint8_t*>(src);
        if (dst) {
            std::memcpy(dst + m_incrementalUploadPackedOffset,
                        srcBytes + m_incrementalUploadPackedOffset,
                        chunkBytes);
        }
        m_incrementalUploadPackedOffset += chunkBytes;
        if (m_incrementalUploadPackedOffset >= totalBytes) {
            m_incrementalUploadPackedOffset = 0u;
            m_incrementalUploadPhase = nextPhase;
        }
    };

    switch (m_incrementalUploadPhase) {
        case UploadPhase::ResolveTextures: {
            constexpr size_t kTexturesPerStep = 1u;
            size_t processed = 0u;
            while (m_incrementalUploadDeferredTextureIndex < m_deferredMaterialTextures.size() &&
                   processed < kTexturesPerStep) {
                const uint32_t placeholder =
                    kDeferredTextureIndexBase + static_cast<uint32_t>(m_incrementalUploadDeferredTextureIndex);
                std::string resolveError;
                const uint32_t resolved =
                    resolveDeferredTextureIndex(*this,
                                                placeholder,
                                                m_resolvedDeferredTextureIndices,
                                                &resolveError);
                if (resolved == kInvalidTextureIndex && !resolveError.empty()) {
                    return false;
                }
                replaceDeferredTexturePlaceholder(placeholder, resolved);
                ++m_incrementalUploadDeferredTextureIndex;
                ++processed;
            }
            if (m_incrementalUploadDeferredTextureIndex >= m_deferredMaterialTextures.size()) {
                m_deferredMaterialTextures.clear();
                m_resolvedDeferredTextureIndices.clear();
                m_incrementalUploadPhase = UploadPhase::ScalarsSpheres;
            }
            break;
        }
        case UploadPhase::ScalarsSpheres: {
            const NSUInteger sphereBytes = static_cast<NSUInteger>(m_sphereCount) * sizeof(SphereData);
            if (sphereBytes > 0) {
                if (!m_sphereBuffer || m_sphereBuffer.length < sphereBytes) {
                    m_sphereBuffer = [m_device newBufferWithLength:sphereBytes
                                                           options:MTLResourceStorageModeShared];
                }
                if (m_sphereBuffer) {
                    std::memcpy([m_sphereBuffer contents], m_spheres.data(), sphereBytes);
                }
            } else {
                m_sphereBuffer = nil;
            }

            m_incrementalUploadPhase = UploadPhase::ScalarsMaterials;
            break;
        }

        case UploadPhase::ScalarsMaterials: {

            const NSUInteger materialBytes = static_cast<NSUInteger>(m_materialCount) * sizeof(MaterialData);
            if (materialBytes > 0) {
                if (!m_materialBuffer || m_materialBuffer.length < materialBytes) {
                    m_materialBuffer = [m_device newBufferWithLength:materialBytes
                                                             options:MTLResourceStorageModeShared];
                }
                if (m_materialBuffer) {
                    std::memcpy([m_materialBuffer contents], m_materials.data(), materialBytes);
                }
            } else {
                m_materialBuffer = nil;
            }

            m_incrementalUploadPhase = UploadPhase::ScalarsRectangles;
            break;
        }

        case UploadPhase::ScalarsRectangles: {

            uploadRectangles();
            m_incrementalUploadPhase = UploadPhase::Meshes;
            break;
        }
        case UploadPhase::Meshes: {
            size_t processed = 0u;
            while (m_incrementalUploadMeshIndex < m_meshes.size() && processed < kMeshesPerStep) {
                auto& mesh = m_meshes[m_incrementalUploadMeshIndex];
                if (!mesh.geometryCacheKey.empty()) {
                    auto sharedIt = m_incrementalSharedMeshBuffers.find(mesh.geometryCacheKey);
                    if (sharedIt != m_incrementalSharedMeshBuffers.end() &&
                        sharedIt->second.vertexBuffer &&
                        sharedIt->second.indexBuffer) {
                        mesh.vertexBuffer = sharedIt->second.vertexBuffer;
                        mesh.indexBuffer = sharedIt->second.indexBuffer;
                        ++m_incrementalUploadMeshIndex;
                        ++processed;
                        continue;
                    }
                }
                const NSUInteger vertexBytes =
                    static_cast<NSUInteger>(mesh.vertices.size()) * sizeof(MeshVertex);
                if (vertexBytes > 0) {
                    if (!mesh.vertexBuffer || mesh.vertexBuffer.length < vertexBytes) {
                        mesh.vertexBuffer = [m_device newBufferWithLength:vertexBytes
                                                                  options:MTLResourceStorageModeShared];
                    }
                    if (mesh.vertexBuffer && !mesh.vertices.empty()) {
                        std::memcpy([mesh.vertexBuffer contents], mesh.vertices.data(), vertexBytes);
                    }
                } else {
                    mesh.vertexBuffer = nil;
                }

                const NSUInteger indexBytes =
                    static_cast<NSUInteger>(mesh.indices.size()) * sizeof(uint32_t);
                if (indexBytes > 0) {
                    if (!mesh.indexBuffer || mesh.indexBuffer.length < indexBytes) {
                        mesh.indexBuffer = [m_device newBufferWithLength:indexBytes
                                                                 options:MTLResourceStorageModeShared];
                    }
                    if (mesh.indexBuffer && !mesh.indices.empty()) {
                        std::memcpy([mesh.indexBuffer contents], mesh.indices.data(), indexBytes);
                    }
                } else {
                    mesh.indexBuffer = nil;
                }

                if (!mesh.geometryCacheKey.empty() &&
                    mesh.vertexBuffer &&
                    mesh.indexBuffer) {
                    m_incrementalSharedMeshBuffers.emplace(
                        mesh.geometryCacheKey,
                        SharedMeshGpuBuffers{mesh.vertexBuffer, mesh.indexBuffer});
                }

                ++m_incrementalUploadMeshIndex;
                ++processed;
            }
            if (m_incrementalUploadMeshIndex >= m_meshes.size()) {
                m_incrementalUploadPhase = UploadPhase::PackedMeshInfos;
            }
            break;
        }
        case UploadPhase::PackedMeshInfos:
            advancePackedBuffer(&m_meshInfoBuffer,
                                m_packedMeshInfos.data(),
                                static_cast<NSUInteger>(m_packedMeshInfos.size() * sizeof(PathTracerShaderTypes::MeshInfo)),
                                m_deferTriangleBufferForHardware
                                    ? UploadPhase::PackedVertices
                                    : UploadPhase::PackedTriangles);
            break;
        case UploadPhase::PackedTriangles:
            if (m_deferTriangleBufferForHardware) {
                m_triangleBuffer = nil;
                m_incrementalUploadPhase = UploadPhase::PackedVertices;
                break;
            }
            advancePackedBuffer(&m_triangleBuffer,
                                m_packedTriangleData.data(),
                                static_cast<NSUInteger>(m_packedTriangleData.size() * sizeof(PathTracerShaderTypes::TriangleData)),
                                UploadPhase::PackedVertices);
            break;
        case UploadPhase::PackedVertices:
            advancePackedBuffer(&m_meshVertexBuffer,
                                m_packedSceneVertices.data(),
                                static_cast<NSUInteger>(m_packedSceneVertices.size() * sizeof(PathTracerShaderTypes::SceneVertex)),
                                UploadPhase::PackedIndices);
            break;
        case UploadPhase::PackedIndices:
            advancePackedBuffer(&m_meshIndexBuffer,
                                m_packedSceneIndices.data(),
                                static_cast<NSUInteger>(m_packedSceneIndices.size() * sizeof(simd::uint3)),
                                UploadPhase::Complete);
            break;
        case UploadPhase::Complete:
            m_incrementalUploadInProgress = false;
            m_incrementalUploadPhase = UploadPhase::Idle;
            m_stagedUploadMode = false;
            m_buffersUploadedViaStagedLoad = true;
            if (outComplete) {
                *outComplete = true;
            }
            return true;
        case UploadPhase::Idle:
        default:
            m_stagedUploadMode = false;
            if (outComplete) {
                *outComplete = true;
            }
            return true;
    }

    if (m_incrementalUploadPhase == UploadPhase::Complete) {
        m_incrementalUploadInProgress = false;
        m_incrementalUploadPhase = UploadPhase::Idle;
        m_stagedUploadMode = false;
        m_buffersUploadedViaStagedLoad = true;
        m_incrementalSharedMeshBuffers.clear();
        if (SceneLoadDiagnosticsVerboseEnabled()) {
            const MeshAggregateStats meshStats = CollectMeshAggregateStats(m_meshes);
            uint32_t coreGpuBufferCount = 0u;
            coreGpuBufferCount += (m_sphereBuffer ? 1u : 0u);
            coreGpuBufferCount += (m_materialBuffer ? 1u : 0u);
            coreGpuBufferCount += (m_rectangleBuffer ? 1u : 0u);
            coreGpuBufferCount += (m_meshInfoBuffer ? 1u : 0u);
            coreGpuBufferCount += (m_triangleBuffer ? 1u : 0u);
            coreGpuBufferCount += (m_meshVertexBuffer ? 1u : 0u);
            coreGpuBufferCount += (m_meshIndexBuffer ? 1u : 0u);
            NSLog(@"[SceneDiag] incrementalUpload complete coreGpuBuffers=%u meshGpuBuffers=%u meshes=%u triangles=%u",
                  coreGpuBufferCount,
                  meshStats.meshGpuBufferCount,
                  meshStats.meshCount,
                  m_triangleCount);
        }
        if (outComplete) {
            *outComplete = true;
        }
    }
    return true;
}

bool SceneResources::beginAccelerationStructureBuild(uint32_t* outBlasBatchesTotal) {
    if (!m_dirty) {
        if (outBlasBatchesTotal) {
            *outBlasBatchesTotal = 0u;
        }
        return true;
    }
    if (!m_sceneAccel) {
        return false;
    }

    m_deferTriangleBufferForHardware =
        hardwareRaytracingEnabled() && !KeepTriangleBufferForHardware();

    if (!m_buffersUploadedViaStagedLoad) {
        // Non-staged (synchronous) path: upload all buffers on the main thread.
        // When loading via StagedSceneLoader these uploads were already done
        // incrementally by stepIncrementalUpload(), so we skip them here to
        // avoid blocking the main thread for potentially seconds on large scenes.
        uploadBuffers();
        if (m_packedTriangleData.empty() && !m_meshes.empty()) {
            buildPackedSceneCpuData();
        }
        uploadPackedSceneBuffers(!m_deferTriangleBufferForHardware);
    }
    m_stagedAccelBuildInput = SceneAccelBuildInput{};
    populateSceneAccelBuildInput(m_stagedAccelBuildInput, m_stagedAccelMeshInputs);
    struct BlasKey {
        const void* vertexBuffer = nullptr;
        const void* indexBuffer = nullptr;
        uint32_t vertexStride = 0u;
        uint32_t vertexCount = 0u;
        uint32_t indexCount = 0u;
        bool operator==(const BlasKey& other) const noexcept {
            return vertexBuffer == other.vertexBuffer &&
                   indexBuffer == other.indexBuffer &&
                   vertexStride == other.vertexStride &&
                   vertexCount == other.vertexCount &&
                   indexCount == other.indexCount;
        }
    };
    struct BlasKeyHasher {
        size_t operator()(const BlasKey& key) const noexcept {
            const size_t h0 = std::hash<const void*>{}(key.vertexBuffer);
            const size_t h1 = std::hash<const void*>{}(key.indexBuffer);
            const size_t h2 = std::hash<uint32_t>{}(key.vertexStride);
            const size_t h3 = std::hash<uint32_t>{}(key.vertexCount);
            const size_t h4 = std::hash<uint32_t>{}(key.indexCount);
            return h0 ^ (h1 << 1u) ^ (h2 << 2u) ^ (h3 << 3u) ^ (h4 << 4u);
        }
    };
    std::unordered_set<BlasKey, BlasKeyHasher> uniqueBlasKeys;
    uniqueBlasKeys.reserve(m_stagedAccelMeshInputs.size());
    for (const auto& meshInput : m_stagedAccelMeshInputs) {
        uniqueBlasKeys.insert(BlasKey{
            .vertexBuffer = (__bridge const void*)meshInput.vertexBuffer,
            .indexBuffer = (__bridge const void*)meshInput.indexBuffer,
            .vertexStride = meshInput.vertexStride,
            .vertexCount = meshInput.vertexCount,
            .indexCount = meshInput.indexCount,
        });
    }
    if (SceneLoadDiagnosticsEnabled()) {
        uint64_t accelInputVertexCount = 0u;
        uint64_t accelInputIndexCount = 0u;
        for (const auto& meshInput : m_stagedAccelMeshInputs) {
            accelInputVertexCount += static_cast<uint64_t>(meshInput.vertexCount);
            accelInputIndexCount += static_cast<uint64_t>(meshInput.indexCount);
        }
        NSLog(@"[SceneDiag] beginASBuild stagedUpload=%d meshInputs=%u uniqueBlasInputs=%zu accelInputVertices=%llu accelInputIndices=%llu sphereInputs=%u",
              m_buffersUploadedViaStagedLoad ? 1 : 0,
              static_cast<uint32_t>(m_stagedAccelMeshInputs.size()),
              uniqueBlasKeys.size(),
              static_cast<unsigned long long>(accelInputVertexCount),
              static_cast<unsigned long long>(accelInputIndexCount),
              m_stagedAccelBuildInput.sphereCount);
    }
    if (!m_sceneAccel->beginIncrementalRebuild(m_stagedAccelBuildInput)) {
        return false;
    }

    m_accelBuildInProgress = m_sceneAccel->incrementalRebuildActive();
    if (outBlasBatchesTotal) {
        *outBlasBatchesTotal = static_cast<uint32_t>(
            (uniqueBlasKeys.size() + (kIncrementalBlasBatchSize - 1u)) / kIncrementalBlasBatchSize);
    }
    return true;
}

bool SceneResources::stepAccelerationStructureBuild(uint32_t* outBlasBatchesDone,
                                                    uint32_t* outBlasBatchesTotal,
                                                    bool* outComplete) {
    if (!m_sceneAccel) {
        return false;
    }

    SceneAccelBuildProgress progress{};
    if (!m_sceneAccel->stepIncrementalRebuild(m_intersectionProvider, progress)) {
        return false;
    }

    m_lastBuildStats = m_sceneAccel->buildStats();
    m_accelBuildInProgress = m_sceneAccel->incrementalRebuildActive();
    m_primitiveCount = m_sceneAccel->primitiveCount() + m_rectangleCount;
    if (m_sceneAccel->mode() == PathTracerShaderTypes::IntersectionMode::HardwareRayTracing) {
        m_primitiveCount += m_sphereCount;
        for (auto& mesh : m_meshes) {
            mesh.vertexBuffer = nil;
            mesh.indexBuffer = nil;
        }
    }

    if (outBlasBatchesDone) {
        *outBlasBatchesDone = progress.blasBatchesDone;
    }
    if (outBlasBatchesTotal) {
        *outBlasBatchesTotal = progress.blasBatchesTotal;
    }
    if (outComplete) {
        *outComplete = progress.complete;
    }

    if (progress.complete) {
        if (m_sceneAccel->mode() == PathTracerShaderTypes::IntersectionMode::SoftwareBVH &&
            m_triangleBuffer == nil &&
            !m_packedTriangleData.empty()) {
            // Hardware build can be configured to skip triangle-buffer upload to
            // save memory; if we ended up in software fallback, triangle data is
            // required for shading, so upload it now.
            uploadPackedSceneBuffers(true);
        }
        if (SceneLoadDiagnosticsEnabled()) {
            NSLog(@"[SceneDiag] asBuild complete mode=%s blasCount=%u instanceCount=%u primitiveCount=%u "
                  "blasMiB=%.2f tlasMiB=%.2f scratchMiB=%.2f fallback='%s'",
                  (m_sceneAccel->mode() == PathTracerShaderTypes::IntersectionMode::HardwareRayTracing)
                      ? "hardware"
                      : "software",
                  m_lastBuildStats.blasCount,
                  m_lastBuildStats.instanceCount,
                  m_lastBuildStats.primitiveCount,
                  BytesToMiB(m_lastBuildStats.blasAllocatedMemoryBytes),
                  BytesToMiB(m_lastBuildStats.tlasAllocatedMemoryBytes),
                  BytesToMiB(m_lastBuildStats.retainedScratchMemoryBytes),
                  m_lastBuildStats.fallbackReason.c_str());
        }
        m_stagedAccelBuildInput = SceneAccelBuildInput{};
        m_stagedAccelMeshInputs.clear();
        m_accelBuildInProgress = false;
        m_dirty = false;
    }

    return true;
}

void SceneResources::rebuildAccelerationStructures() {
    if (!m_dirty) {
        return;
    }

    uploadBuffers();
    if (m_packedTriangleData.empty() && !m_meshes.empty()) {
        buildPackedSceneCpuData();
    }
    m_deferTriangleBufferForHardware =
        hardwareRaytracingEnabled() && !KeepTriangleBufferForHardware();
    uploadPackedSceneBuffers(!m_deferTriangleBufferForHardware);

    if (m_sceneAccel) {
        uint32_t ignoredTotal = 0u;
        if (!beginAccelerationStructureBuild(&ignoredTotal)) {
            return;
        }
        bool complete = false;
        while (!complete) {
            if (!stepAccelerationStructureBuild(nullptr, nullptr, &complete)) {
                return;
            }
        }
    } else {
        m_intersectionProvider = IntersectionProvider{};
        m_primitiveCount = m_rectangleCount;
        m_lastBuildStats = SceneAccelBuildStats{};
        m_dirty = false;
    }
    NSString* modeString = (m_intersectionProvider.mode ==
                            PathTracerShaderTypes::IntersectionMode::HardwareRayTracing)
                               ? @"HardwareRT"
                               : @"SoftwareBVH";
    NSLog(@"SceneResources::rebuildAccelerationStructures - mode=%@ meshes=%zu triangles=%u spheres=%u primitives=%u",
          modeString,
          m_meshes.size(),
          m_triangleCount,
          m_sphereCount,
          m_primitiveCount);
}

void SceneResources::uploadMeshes() {
    if (!m_device) {
        return;
    }

    std::unordered_map<std::string, SharedMeshGpuBuffers> sharedBuffersByKey;
    sharedBuffersByKey.reserve(m_meshes.size());

    for (size_t meshIndex = 0; meshIndex < m_meshes.size(); ++meshIndex) {
        PumpEventsEvery(meshIndex, 4u);
        auto& mesh = m_meshes[meshIndex];
        if (!mesh.geometryCacheKey.empty()) {
            auto sharedIt = sharedBuffersByKey.find(mesh.geometryCacheKey);
            if (sharedIt != sharedBuffersByKey.end() &&
                sharedIt->second.vertexBuffer &&
                sharedIt->second.indexBuffer) {
                mesh.vertexBuffer = sharedIt->second.vertexBuffer;
                mesh.indexBuffer = sharedIt->second.indexBuffer;
                continue;
            }
        }
        const NSUInteger vertexBytes =
            static_cast<NSUInteger>(mesh.vertices.size()) * sizeof(MeshVertex);
        if (vertexBytes > 0) {
            if (!mesh.vertexBuffer || mesh.vertexBuffer.length < vertexBytes) {
                mesh.vertexBuffer = [m_device newBufferWithLength:vertexBytes
                                                          options:MTLResourceStorageModeShared];
            }
            if (mesh.vertexBuffer && !mesh.vertices.empty()) {
                memcpy([mesh.vertexBuffer contents], mesh.vertices.data(), vertexBytes);
            }
        } else {
            mesh.vertexBuffer = nil;
        }

        const NSUInteger indexBytes =
            static_cast<NSUInteger>(mesh.indices.size()) * sizeof(uint32_t);
        if (indexBytes > 0) {
            if (!mesh.indexBuffer || mesh.indexBuffer.length < indexBytes) {
                mesh.indexBuffer = [m_device newBufferWithLength:indexBytes
                                                         options:MTLResourceStorageModeShared];
            }
            if (mesh.indexBuffer && !mesh.indices.empty()) {
                memcpy([mesh.indexBuffer contents], mesh.indices.data(), indexBytes);
            }
        } else {
            mesh.indexBuffer = nil;
        }

        if (!mesh.geometryCacheKey.empty() &&
            mesh.vertexBuffer &&
            mesh.indexBuffer) {
            sharedBuffersByKey.emplace(
                mesh.geometryCacheKey,
                SharedMeshGpuBuffers{mesh.vertexBuffer, mesh.indexBuffer});
        }
    }
}

void SceneResources::uploadRectangles() {
    if (!m_device) {
        return;
    }

    const NSUInteger rectBytes =
        static_cast<NSUInteger>(m_rectangleCount) * sizeof(RectData);
    if (rectBytes > 0) {
        if (!m_rectangleBuffer || m_rectangleBuffer.length < rectBytes) {
            m_rectangleBuffer =
                [m_device newBufferWithLength:rectBytes options:MTLResourceStorageModeShared];
        }
        if (m_rectangleBuffer) {
            memcpy([m_rectangleBuffer contents], m_rectangles.data(), rectBytes);
        }
    } else {
        m_rectangleBuffer = nil;
    }
}

void SceneResources::storeRectangleOriented(const simd::float3& corner,
                                            const simd::float3& edgeU,
                                            const simd::float3& edgeV,
                                            bool twoSided,
                                            uint32_t materialIndex,
                                            const simd::float3& desiredNormal) {
    if (m_rectangleCount >= kMaxRectangles) {
        return;
    }

    float uLenSq = simd::dot(edgeU, edgeU);
    float vLenSq = simd::dot(edgeV, edgeV);
    if (uLenSq <= std::numeric_limits<float>::min() ||
        vLenSq <= std::numeric_limits<float>::min()) {
        return;
    }

    simd::float3 normal = simd::cross(edgeU, edgeV);
    float normalLenSq = simd::dot(normal, normal);
    if (normalLenSq <= std::numeric_limits<float>::min()) {
        return;
    }
    simd::float3 unitNormal = normal / std::sqrt(normalLenSq);

    float desiredLenSq = simd::dot(desiredNormal, desiredNormal);
    simd::float3 targetNormal = unitNormal;
    if (desiredLenSq > std::numeric_limits<float>::min()) {
        targetNormal = desiredNormal / std::sqrt(desiredLenSq);
    }
    if (simd::dot(unitNormal, targetNormal) < 0.0f) {
        unitNormal = -unitNormal;
    }

    if (!std::isfinite(unitNormal.x) || !std::isfinite(unitNormal.y) || !std::isfinite(unitNormal.z)) {
        return;
    }

    float planeConstant = simd::dot(unitNormal, corner);

    RectData rect{};
    rect.corner = simd_make_float4(corner, 0.0f);
    rect.edgeU = simd_make_float4(edgeU, (uLenSq > 0.0f) ? (1.0f / uLenSq) : 0.0f);
    rect.edgeV = simd_make_float4(edgeV, (vLenSq > 0.0f) ? (1.0f / vLenSq) : 0.0f);
    rect.normalAndPlane = simd_make_float4(unitNormal, planeConstant);
    rect.materialTwoSided = simd_make_uint4(materialIndex, twoSided ? 1u : 0u, 0u, 0u);

    m_rectangles[m_rectangleCount++] = rect;
    m_dirty = true;
}

void SceneResources::storeDiskOriented(const simd::float3& center,
                                       const simd::float3& radiusAxisU,
                                       const simd::float3& radiusAxisV,
                                       bool twoSided,
                                       uint32_t materialIndex,
                                       const simd::float3& desiredNormal) {
    if (m_rectangleCount >= kMaxRectangles) {
        return;
    }

    float uLenSq = simd::dot(radiusAxisU, radiusAxisU);
    float vLenSq = simd::dot(radiusAxisV, radiusAxisV);
    if (uLenSq <= std::numeric_limits<float>::min() ||
        vLenSq <= std::numeric_limits<float>::min()) {
        return;
    }

    simd::float3 normal = simd::cross(radiusAxisU, radiusAxisV);
    float normalLenSq = simd::dot(normal, normal);
    if (normalLenSq <= std::numeric_limits<float>::min()) {
        return;
    }
    simd::float3 unitNormal = normal / std::sqrt(normalLenSq);

    float desiredLenSq = simd::dot(desiredNormal, desiredNormal);
    simd::float3 targetNormal = unitNormal;
    if (desiredLenSq > std::numeric_limits<float>::min()) {
        targetNormal = desiredNormal / std::sqrt(desiredLenSq);
    }
    if (simd::dot(unitNormal, targetNormal) < 0.0f) {
        unitNormal = -unitNormal;
    }

    if (!std::isfinite(unitNormal.x) || !std::isfinite(unitNormal.y) || !std::isfinite(unitNormal.z)) {
        return;
    }

    float planeConstant = simd::dot(unitNormal, center);

    RectData rect{};
    rect.corner = simd_make_float4(center, 0.0f);
    rect.edgeU = simd_make_float4(radiusAxisU, (uLenSq > 0.0f) ? (1.0f / uLenSq) : 0.0f);
    rect.edgeV = simd_make_float4(radiusAxisV, (vLenSq > 0.0f) ? (1.0f / vLenSq) : 0.0f);
    rect.normalAndPlane = simd_make_float4(unitNormal, planeConstant);
    rect.materialTwoSided = simd_make_uint4(materialIndex,
                                            twoSided ? 1u : 0u,
                                            static_cast<uint32_t>(PathTracerShaderTypes::AreaPrimitiveShape::Disk),
                                            0u);

    m_rectangles[m_rectangleCount++] = rect;
    m_dirty = true;
}

}  // namespace PathTracer
