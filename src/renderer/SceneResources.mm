#import "renderer/SceneResources.h"
#import "renderer/MetalContext.h"
#import "renderer/SceneAccel.h"
#import "renderer/EnvImportanceSampler.h"
#import "renderer/ImageWriter.h"
#import "renderer/LoadEventPump.h"
#include "shared/DdsTexture.h"
#include "shared/Ktx2Texture.h"
#include "shared/MipChain.h"

// Source scalar data must bypass CoreGraphics color management and alpha premultiplication.
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb_image.h"  // external/stb, a SYSTEM include directory

#import <AppKit/AppKit.h>
#import <CommonCrypto/CommonDigest.h>
#import <ImageIO/ImageIO.h>
#import <MetalKit/MetalKit.h>
#import <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <condition_variable>
#include <iostream>
#include <limits>
#include <chrono>
#include <map>
#include <memory>
#include <set>
#include <mutex>
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

bool TexturePolicyVerboseLoggingEnabled() {
    return SceneLoadDiagnosticsVerboseEnabled();
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
    effective.maxAnisotropy = std::clamp(effective.maxAnisotropy, 1u, 16u);
    return effective;
}

uint64_t SamplerDescKey(const MaterialTextureSamplerDesc& desc) {
    uint64_t key = 14695981039346656037ull;
    for (uint32_t value : {static_cast<uint32_t>(desc.magFilter),
                           static_cast<uint32_t>(desc.minFilter),
                           static_cast<uint32_t>(desc.wrapS),
                           static_cast<uint32_t>(desc.wrapT), desc.maxAnisotropy}) {
        key = (key ^ value) * 1099511628211ull;
    }
    return key;
}

std::string SamplerKeySuffix(const MaterialTextureSamplerDesc* desc) {
    MaterialTextureSamplerDesc effective = EffectiveSamplerDesc(desc);
    return "|samp:" + std::to_string(effective.magFilter) + ":" +
           std::to_string(effective.minFilter) + ":" +
           std::to_string(effective.wrapS) + ":" +
           std::to_string(effective.wrapT) + ":" +
           std::to_string(effective.maxAnisotropy);
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
        case MaterialTextureSemantic::ScalarAverage:
            return "scalarAverage";
        case MaterialTextureSemantic::ScalarAlpha:
            return "scalarAlpha";
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

using LoadEventPump::PumpEventsEvery;
using LoadEventPump::WaitForCommandBufferResponsive;

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

// Metal's checks on a DDS source: a header the shared parser accepts, and every mip level the header declares
// present in the file. A DDS that fails them is not loaded; the texture falls back to its KTX2.
bool ParseCompressedDdsHeader(const std::vector<uint8_t>& bytes,
                              bool srgb,
                              Dds::Header& header,
                              MTLPixelFormat& pixelFormat,
                              std::string& errorMessage) {
    if (!Dds::ParseHeader(bytes, header, &errorMessage)) {
        return false;
    }

    size_t offset = header.dataOffset;
    uint32_t mipWidth = header.width;
    uint32_t mipHeight = header.height;
    for (uint32_t level = 0u; level < header.mipCount; ++level) {
        const size_t levelBytes = Dds::LevelBytes(header.format, mipWidth, mipHeight, bytes.size() - offset);
        if (levelBytes == 0u) {
            errorMessage = "DDS mip chain is truncated";
            return false;
        }
        offset += levelBytes;
        mipWidth = std::max<uint32_t>(1u, mipWidth >> 1u);
        mipHeight = std::max<uint32_t>(1u, mipHeight >> 1u);
    }

    const bool srgbFormat = Dds::DecodesAsSrgb(header, srgb);
    switch (header.format) {
        case Dds::Format::Bc1:
            pixelFormat = srgbFormat ? MTLPixelFormatBC1_RGBA_sRGB : MTLPixelFormatBC1_RGBA;
            break;
        case Dds::Format::Bc3:
            pixelFormat = srgbFormat ? MTLPixelFormatBC3_RGBA_sRGB : MTLPixelFormatBC3_RGBA;
            break;
        case Dds::Format::Bc5:
            pixelFormat = MTLPixelFormatBC5_RGUnorm;
            break;
    }
    return true;
}

bool ReadCompressedDdsFile(const std::string& path, std::vector<uint8_t>& bytes, std::string& errorMessage) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        errorMessage = "Compressed DDS load failed: unable to open file";
        return false;
    }
    stream.seekg(0, std::ios::end);
    const std::streamsize size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    if (size <= 0) {
        errorMessage = "Compressed DDS load failed: file is empty";
        return false;
    }

    bytes.resize(static_cast<size_t>(size));
    stream.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!stream) {
        errorMessage = "Compressed DDS load failed: unable to read file";
        return false;
    }
    return true;
}

// Decodes the mip levels LoadCompressedDdsTextureFromPath uploads from a DDS, for a CPU copy of the texture. Fails,
// with Metal's reason, on a DDS Metal's loader refuses.
bool DecodeCompressedDdsLevels(const std::string& path,
                               bool srgb,
                               std::vector<Dds::Level>& levels,
                               bool& decodesAsSrgb,
                               std::string& errorMessage) {
    std::vector<uint8_t> bytes;
    Dds::Header header;
    MTLPixelFormat pixelFormat = MTLPixelFormatInvalid;
    if (!ReadCompressedDdsFile(path, bytes, errorMessage) ||
        !ParseCompressedDdsHeader(bytes, srgb, header, pixelFormat, errorMessage)) {
        return false;
    }
    decodesAsSrgb = Dds::DecodesAsSrgb(header, srgb);
    if (!Dds::DecodeLevels(bytes, header, 0u, /*reconstructBc5Z=*/false, levels)) {
        errorMessage = "DDS mip chain is truncated";
        return false;
    }
    return true;
}

// Texture preparation runs its per-texel loops in bands of output rows on up to kTexturePrepThreads GCD workers. Each
// row runs the same code as the serial loop, so the results are bit-identical. A level under
// kTexturePrepMinParallelTexels stays on the calling thread. The cap and the image prefetch's decode window
// (MaterialImagePrefetch::kWindow) bound the loader's GCD work at 14 tasks.
constexpr size_t kTexturePrepThreads = 8u;
constexpr size_t kTexturePrepMinParallelTexels = size_t{1} << 16;

template <typename Body>
void ForEachRowBand(size_t rows, size_t texelsPerRow, const Body& body) {
    size_t bands = std::min(kTexturePrepThreads, rows);
    if (bands <= 1u || rows * texelsPerRow < kTexturePrepMinParallelTexels) {
        body(size_t{0}, rows);
        return;
    }
    const size_t rowsPerBand = (rows + bands - 1u) / bands;
    bands = (rows + rowsPerBand - 1u) / rowsPerBand;
    const Body* bodyPtr = &body;
    dispatch_apply(bands, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t band) {
        const size_t firstRow = band * rowsPerBand;
        (*bodyPtr)(firstRow, std::min(rows, firstRow + rowsPerBand));
    });
}

struct DecodedImageRgba8 {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> rgba;
};

bool IsNativeScalarTexture(MaterialTextureSemantic semantic) {
    return semantic == MaterialTextureSemantic::ScalarAverage ||
           semantic == MaterialTextureSemantic::ScalarAlpha;
}

void ConvertNativeScalarTexture(DecodedImageRgba8& image, MaterialTextureSemantic semantic) {
    if (!IsNativeScalarTexture(semantic)) return;
    for (size_t i = 0; i < image.rgba.size(); i += 4u) {
        const uint8_t value = semantic == MaterialTextureSemantic::ScalarAlpha
            ? image.rgba[i + 3u]
            : static_cast<uint8_t>((uint32_t(image.rgba[i]) + image.rgba[i + 1u] + image.rgba[i + 2u]) / 3u);
        image.rgba[i] = image.rgba[i + 1u] = image.rgba[i + 2u] = image.rgba[i + 3u] = value;
    }
}

uint8_t MinAlphaOfRgba8(const DecodedImageRgba8& image) {
    uint8_t minAlpha = 255u;
    for (size_t i = 3; i < image.rgba.size(); i += 4u) {
        minAlpha = std::min(minAlpha, image.rgba[i]);
    }
    return minAlpha;
}

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

    CGContextDrawImage(context, CGRectMake(0, 0, static_cast<CGFloat>(width), static_cast<CGFloat>(height)), image);
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

bool DecodeRawScalarData(const uint8_t* data, size_t size, DecodedImageRgba8& image,
                         std::string& error) {
    if (size > size_t(std::numeric_limits<int>::max())) {
        error = "Scalar texture exceeds decoder size limit";
        return false;
    }
    int width, height, channels;
    stbi_uc* pixels = stbi_load_from_memory(data, int(size), &width, &height, &channels, 4);
    if (!pixels) {
        error = std::string("Scalar texture decode failed: ") + stbi_failure_reason();
        return false;
    }
    image.width = uint32_t(width);
    image.height = uint32_t(height);
    image.rgba.assign(pixels, pixels + size_t(width) * size_t(height) * 4u);
    stbi_image_free(pixels);
    return true;
}

bool DecodeRawScalarFile(const std::string& path, DecodedImageRgba8& image, std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { error = "Scalar texture cannot be opened: " + path; return false; }
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    return DecodeRawScalarData(data.data(), data.size(), image, error);
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
        id<MTLBuffer> mipStagingBuffer = nil;
        if (buildMipmaps && texture.mipmapLevelCount > 1u) {
            if (MipChain::BoxChainKeepsMean(image.width, image.height)) {
                [blit generateMipmapsForTexture:texture];
            } else {
                // Not a power of two: the blit's filter does not keep the texture mean at the coarse levels. Upload
                // the footprint-filtered levels (MipChain.h); Embree's CPU copy decodes the same levels.
                const std::vector<MipChain::Rgba8Level> levels = MipChain::BuildRgba8MipLevels(
                    image.rgba.data(), image.width, image.height, pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB);
                size_t levelBytes = 0u;
                for (const auto& level : levels) levelBytes += level.rgba.size();
                mipStagingBuffer = levelBytes > 0u
                    ? [device newBufferWithLength:levelBytes options:MTLResourceStorageModeShared] : nil;
                if (!mipStagingBuffer || levels.size() + 1u != texture.mipmapLevelCount) {
                    [blit endEncoding];
                    errorMessage = "Material texture load failed: unable to stage the mip levels";
                    return nil;
                }
                size_t offset = 0u;
                for (size_t level = 0u; level < levels.size(); ++level) {
                    const MipChain::Rgba8Level& mip = levels[level];
                    std::memcpy(static_cast<uint8_t*>([mipStagingBuffer contents]) + offset, mip.rgba.data(),
                                mip.rgba.size());
                    [blit copyFromBuffer:mipStagingBuffer
                            sourceOffset:offset
                       sourceBytesPerRow:static_cast<NSUInteger>(mip.width) * 4u
                     sourceBytesPerImage:mip.rgba.size()
                              sourceSize:MTLSizeMake(mip.width, mip.height, 1)
                               toTexture:texture
                        destinationSlice:0
                        destinationLevel:level + 1u
                       destinationOrigin:MTLOriginMake(0, 0, 0)];
                    offset += mip.rgba.size();
                }
            }
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

    std::vector<uint8_t> bytes;
    Dds::Header header;
    MTLPixelFormat pixelFormat = MTLPixelFormatInvalid;
    if (!ReadCompressedDdsFile(path, bytes, errorMessage) ||
        !ParseCompressedDdsHeader(bytes, srgb, header, pixelFormat, errorMessage)) {
        return nil;
    }

    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:pixelFormat
                                                           width:header.width
                                                          height:header.height
                                                       mipmapped:(header.mipCount > 1u)];
    desc.textureType = MTLTextureType2D;
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
    if (!texture) {
        errorMessage = "Compressed DDS load failed: unable to allocate texture";
        return nil;
    }

    const size_t blockBytes = Dds::BlockBytes(header.format);
    size_t offset = header.dataOffset;
    uint32_t mipWidth = header.width;
    uint32_t mipHeight = header.height;
    const uint32_t mipCount = std::min<uint32_t>(header.mipCount, static_cast<uint32_t>(texture.mipmapLevelCount));
    for (uint32_t level = 0u; level < mipCount; ++level) {
        const uint32_t blocksWide = std::max<uint32_t>(1u, (mipWidth + 3u) / 4u);
        const uint32_t blocksHigh = std::max<uint32_t>(1u, (mipHeight + 3u) / 4u);
        const NSUInteger bytesPerRow = static_cast<NSUInteger>(blocksWide * blockBytes);
        const size_t levelBytes = static_cast<size_t>(blocksWide) * static_cast<size_t>(blocksHigh) * blockBytes;
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
                                      id<MTLCommandQueue> /*commandQueue*/,
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

// 8-bit RGBA (or BGRA) texels to RGBA floats in [0, 1], as CopyTextureLevelToFloat reads an 8-bit level. decodeSrgb
// linearizes the colour channels.
void Rgba8TexelsToFloats(const uint8_t* src, size_t width, size_t height, bool isBgra, bool decodeSrgb, float* out) {
ForEachRowBand(height, width, [&](size_t firstRow, size_t endRow) {
        for (size_t i = firstRow * width; i < endRow * width; ++i) {
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
            if (decodeSrgb) {
                r = SrgbToLinear(r);
                g = SrgbToLinear(g);
                b = SrgbToLinear(b);
            }

            out[i * 4u + 0u] = r;
            out[i * 4u + 1u] = g;
            out[i * 4u + 2u] = b;
            out[i * 4u + 3u] = static_cast<float>(a8) / 255.0f;
        }
    });
}

// RGBA floats to 8-bit RGBA (or BGRA) texels, as WriteTextureLevelFromFloat writes an 8-bit level: clamped to [0, 1]
// and rounded; a non-finite colour channel becomes 0 and a non-finite alpha 1.
void FloatsToRgba8Texels(const float* in, size_t width, size_t height, bool isBgra, uint8_t* dst) {
ForEachRowBand(height, width, [&](size_t firstRow, size_t endRow) {
        for (size_t i = firstRow * width; i < endRow * width; ++i) {
            float r = std::isfinite(in[i * 4u + 0u]) ? in[i * 4u + 0u] : 0.0f;
            float g = std::isfinite(in[i * 4u + 1u]) ? in[i * 4u + 1u] : 0.0f;
            float b = std::isfinite(in[i * 4u + 2u]) ? in[i * 4u + 2u] : 0.0f;
            float a = std::isfinite(in[i * 4u + 3u]) ? in[i * 4u + 3u] : 1.0f;
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
    });
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
            Rgba8TexelsToFloats(rawData.data(), width, height, isBgra, decodeSrgb && isSrgbFormat, outFloats.data());
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
            FloatsToRgba8Texels(inFloats.data(), width, height, isBgra, rawData.data());
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

// The next level of a power-of-two texture's CPU copy: the 2x2 box, which Metal's generateMipmapsForTexture also runs on
// exact halvings. Textures of other sizes take MipChain.h's footprint-filtered levels on both backends instead
// (populateRgba8 in materialTextureCpuData); they replace the emulation of the blit's filter for odd sizes that this
// function held before.
void DownsampleMipLevelBox(const std::vector<float>& src,
                           uint32_t srcWidth,
                           uint32_t srcHeight,
                           std::vector<float>& dst,
                           uint32_t dstWidth,
                           uint32_t dstHeight) {
    dst.assign(static_cast<size_t>(dstWidth) * static_cast<size_t>(dstHeight) * 4u, 0.0f);
    for (uint32_t y = 0; y < dstHeight; ++y) {
        for (uint32_t x = 0; x < dstWidth; ++x) {
            simd::float4 accum = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (uint32_t oy = 0; oy < 2u; ++oy) {
                for (uint32_t ox = 0; ox < 2u; ++ox) {
                    const uint32_t sx = std::min(x * 2u + ox, srcWidth - 1u);
                    const uint32_t sy = std::min(y * 2u + oy, srcHeight - 1u);
                    const size_t idx = (static_cast<size_t>(sy) * srcWidth + static_cast<size_t>(sx)) * 4u;
                    accum += simd_make_float4(src[idx + 0u], src[idx + 1u], src[idx + 2u], src[idx + 3u]);
                }
            }
            accum /= 4.0f;
            const size_t out = (static_cast<size_t>(y) * dstWidth + static_cast<size_t>(x)) * 4u;
            dst[out + 0u] = accum.x;
            dst[out + 1u] = accum.y;
            dst[out + 2u] = accum.z;
            dst[out + 3u] = accum.w;
        }
    }
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
    // A level with an odd size: weight each source texel by its share of the destination texel's true footprint
    // (MipChain.h), in the same channel spaces as the 2x2 box below, so the level keeps the means of AO, metalness,
    // alpha and squared roughness. The box drops the odd row or column. Exact halvings keep the box, so every level
    // of a power-of-two texture stays bit-identical.
    if (!MipChain::AxisHalvesExactly(static_cast<uint32_t>(srcWidth), static_cast<uint32_t>(dstWidth)) ||
        !MipChain::AxisHalvesExactly(static_cast<uint32_t>(srcHeight), static_cast<uint32_t>(dstHeight))) {
        const auto xs = MipChain::AxisFootprints(static_cast<uint32_t>(srcWidth), static_cast<uint32_t>(dstWidth));
        const auto ys = MipChain::AxisFootprints(static_cast<uint32_t>(srcHeight), static_cast<uint32_t>(dstHeight));
ForEachRowBand(dstHeight, dstWidth, [&](size_t firstRow, size_t endRow) {
            for (NSUInteger y = firstRow; y < endRow; ++y) {
                for (NSUInteger x = 0; x < dstWidth; ++x) {
                    float sumAo = 0.0f, sumAlpha = 0.0f, sumMetal = 0.0f, sumA = 0.0f;
                    for (uint32_t j = 0u; j < ys[y].count; ++j) {
                        for (uint32_t i = 0u; i < xs[x].count; ++i) {
                            const float w = xs[x].weight[i] * ys[y].weight[j];
                            const size_t srcIndex =
                                (static_cast<size_t>(ys[y].first + j) * srcWidth + xs[x].first + i) * 4u;
                            const float ao = std::isfinite(src[srcIndex + 0u]) ? src[srcIndex + 0u] : 1.0f;
                            const float rough = std::clamp(std::isfinite(src[srcIndex + 1u]) ? src[srcIndex + 1u] : 1.0f,
                                                           0.0f, 1.0f);
                            const float metal = std::isfinite(src[srcIndex + 2u]) ? src[srcIndex + 2u] : 0.0f;
                            const float alpha = std::isfinite(src[srcIndex + 3u]) ? src[srcIndex + 3u] : 1.0f;
                            sumAo += w * ao;
                            sumAlpha += w * rough * rough;
                            sumMetal += w * metal;
                            sumA += w * alpha;
                        }
                    }
                    const size_t dstIndex = static_cast<size_t>((y * dstWidth + x) * 4u);
                    dst[dstIndex + 0u] = sumAo;
                    dst[dstIndex + 1u] = std::sqrt(std::max(sumAlpha, 0.0f));
                    dst[dstIndex + 2u] = sumMetal;
                    dst[dstIndex + 3u] = sumA;
                }
            }
        });
        return;
    }
ForEachRowBand(dstHeight, dstWidth, [&](size_t firstRow, size_t endRow) {
        for (NSUInteger y = firstRow; y < endRow; ++y) {
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
    });
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
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
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
    [blit endEncoding];
    [commandBuffer commit];
    WaitForCommandBufferResponsive(commandBuffer);
    return reducedTexture;
}

// A copy of every level of a texture, as it is, in the given storage mode.
id<MTLTexture> CopyTextureLevels(id<MTLDevice> device,
                                 id<MTLCommandQueue> commandQueue,
                                 id<MTLTexture> source,
                                 MTLStorageMode storageMode) {
    if (!device || !commandQueue || !source) {
        return nil;
    }
    MTLTextureDescriptor* desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:source.pixelFormat
                                                           width:source.width
                                                          height:source.height
                                                       mipmapped:(source.mipmapLevelCount > 1u)];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = storageMode;
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

id<MTLTexture> PromoteTextureToPrivate(id<MTLDevice> device,
                                       id<MTLCommandQueue> commandQueue,
                                       id<MTLTexture> source) {
    if (source && source.storageMode == MTLStorageModePrivate) {
        return source;
    }
    return CopyTextureLevels(device, commandQueue, source, MTLStorageModePrivate);
}

// Mips for a single-level material texture, such as a KTX2 image. Power-of-two textures take the blit. Other sizes take
// MipChain.h's footprint-filtered levels, which keep the texture mean (the blit's filter does not); decoded images
// get the same levels in CreateTextureFromDecodedRgba8, and Embree's CPU copy decodes them.
id<MTLTexture> BuildMaterialTextureWithMips(id<MTLDevice> device,
                                            id<MTLCommandQueue> commandQueue,
                                            id<MTLTexture> source,
                                            MTLStorageMode storageMode) {
    if (!device || !commandQueue || !source) {
        return nil;
    }
    const uint32_t width = static_cast<uint32_t>(source.width);
    const uint32_t height = static_cast<uint32_t>(source.height);
    if (MipChain::BoxChainKeepsMean(width, height)) {
        return BuildTextureWithBlitMips(device, commandQueue, source, storageMode);
    }
    const MTLPixelFormat format = source.pixelFormat;
    const bool rgba8 = format == MTLPixelFormatRGBA8Unorm || format == MTLPixelFormatRGBA8Unorm_sRGB ||
                       format == MTLPixelFormatBGRA8Unorm || format == MTLPixelFormatBGRA8Unorm_sRGB;
    if (!rgba8) {
        std::cerr << "[MaterialTexture] warning: " << width << "x" << height << " texture in pixel format "
                  << static_cast<unsigned long>(format)
                  << " has no mean-preserving mip path; its levels come from the blit, whose coarse levels do not"
                  << " keep the texture mean at this size." << std::endl;
        return BuildTextureWithBlitMips(device, commandQueue, source, storageMode);
    }
    id<MTLTexture> level0 = source.storageMode == MTLStorageModePrivate
        ? CloneTextureFromSourceLevel(device, commandQueue, source, 0u) : source;
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                                    width:width
                                                                                   height:height
                                                                                mipmapped:YES];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = level0 ? [device newTextureWithDescriptor:desc] : nil;
    if (!texture) {
        return nil;
    }
    std::vector<uint8_t> texels(static_cast<size_t>(width) * height * 4u);
    [level0 getBytes:texels.data()
         bytesPerRow:width * 4u
          fromRegion:MTLRegionMake2D(0, 0, width, height)
         mipmapLevel:0];
    [texture replaceRegion:MTLRegionMake2D(0, 0, width, height)
               mipmapLevel:0
                 withBytes:texels.data()
               bytesPerRow:width * 4u];
    const bool srgb = format == MTLPixelFormatRGBA8Unorm_sRGB || format == MTLPixelFormatBGRA8Unorm_sRGB;
    const std::vector<MipChain::Rgba8Level> levels = MipChain::BuildRgba8MipLevels(texels.data(), width, height, srgb);
    for (size_t level = 0u; level < levels.size() && level + 1u < texture.mipmapLevelCount; ++level) {
        [texture replaceRegion:MTLRegionMake2D(0, 0, levels[level].width, levels[level].height)
                   mipmapLevel:level + 1u
                     withBytes:levels[level].rgba.data()
                   bytesPerRow:levels[level].width * 4u];
    }
    if (storageMode == MTLStorageModePrivate) {
        id<MTLTexture> privateTexture = PromoteTextureToPrivate(device, commandQueue, texture);
        return privateTexture ? privateTexture : texture;
    }
    return texture;
}

// A material texture with its mip levels, or the texture as it came, with a warning, when the levels cannot be built.
// The texture is then sampled at mip 0 at every LOD, so the failure must not be silent.
id<MTLTexture> WithMaterialMipsOrWarn(id<MTLDevice> device,
                                      id<MTLCommandQueue> commandQueue,
                                      id<MTLTexture> source,
                                      const std::string& label) {
    id<MTLTexture> mipTexture = BuildMaterialTextureWithMips(device, commandQueue, source, MTLStorageModePrivate);
    if (!mipTexture) {
        std::cerr << "[MaterialTexture] warning: could not build the mip levels of " << label << " ("
                  << source.width << "x" << source.height
                  << "); it keeps one level and is sampled at mip 0 at every LOD." << std::endl;
        return source;
    }
    return mipTexture;
}

// A normal-map chain level's value as the chain stores it: quantized to an 8-bit step.
inline float QuantizeNormalChainValue(float value) {
    return std::round(value * 255.0f) / 255.0f;
}

// Alpha holds the standard deviation of the normalized base directions under each texel.
// `base` becomes level 0. quantizeLevel0 = false leaves level 0's colour unquantized, for a caller that does not read
// level 0's floats (BuildNormalMipLevelsRgba8); the other levels never read level 0's quantized values.
// The moment buffers are allocated uninitialized: every element is written before it is read.
std::vector<MaterialTextureCpuData::MipLevel> BuildNormalMipChain(std::vector<float>&& base, uint32_t width,
                                                                  uint32_t height, bool quantizeLevel0 = true) {
    std::vector<MaterialTextureCpuData::MipLevel> levels;
    levels.push_back(MaterialTextureCpuData::MipLevel{width, height, std::move(base), {}});
    std::unique_ptr<simd::double4[]> moments(new simd::double4[static_cast<size_t>(width) * height]);
    {
        float* level0 = levels[0].rgba.data();
ForEachRowBand(height, width, [&](size_t firstRow, size_t endRow) {
            for (size_t i = firstRow * width; i < endRow * width; ++i) {
                simd::double3 n = simd::normalize(simd_make_double3(level0[i * 4u] * 2.0 - 1.0,
                                                                    level0[i * 4u + 1u] * 2.0 - 1.0,
                                                                    level0[i * 4u + 2u] * 2.0 - 1.0));
                moments[i] = simd_make_double4(n.x, n.y, n.z, 0.0);
                level0[i * 4u + 3u] = 0.0f;
            }
        });
    }
    while (width > 1u || height > 1u) {
        const auto& prev = levels.back();
        MaterialTextureCpuData::MipLevel next{std::max(width / 2u, 1u), std::max(height / 2u, 1u), {}, {}};
        next.rgba.resize(static_cast<size_t>(next.width) * next.height * 4u);
        std::unique_ptr<simd::double4[]> nextMoments(new simd::double4[next.rgba.size() / 4u]);
        // An exact halving takes the 2x2 box. A level with an odd size weights each source texel by its share of the
        // destination texel's true footprint (MipChain.h); the box would drop the odd row or column. Both run the same
        // sums, so the box levels (every level of a power-of-two texture) stay bit-identical.
        const bool box = MipChain::AxisHalvesExactly(width, next.width) &&
                         MipChain::AxisHalvesExactly(height, next.height);
        const auto xs = MipChain::AxisFootprints(width, next.width);
        const auto ys = MipChain::AxisFootprints(height, next.height);
ForEachRowBand(next.height, next.width, [&](size_t firstRow, size_t endRow) {
            for (uint32_t y = static_cast<uint32_t>(firstRow); y < endRow; ++y) {
                for (uint32_t x = 0; x < next.width; ++x) {
                    size_t ids[16];
                    double weights[16];
                    uint32_t taps = 0u;
                    if (box) {
                        for (uint32_t k = 0; k < 4u; ++k, ++taps) {
                            ids[k] = static_cast<size_t>(std::min(y * 2u + k / 2u, height - 1u)) * width +
                                     std::min(x * 2u + k % 2u, width - 1u);
                            weights[k] = 0.25;
                        }
                    } else {
                        for (uint32_t j = 0u; j < ys[y].count; ++j) {
                            for (uint32_t i = 0u; i < xs[x].count; ++i, ++taps) {
                                ids[taps] = static_cast<size_t>(ys[y].first + j) * width + xs[x].first + i;
                                weights[taps] = static_cast<double>(xs[x].weight[i]) * ys[y].weight[j];
                            }
                        }
                    }
                    simd::double3 mean = moments[ids[0]].xyz;
                    double variance = 0.0;
                    for (uint32_t k = 1; k < taps; ++k) mean += (moments[ids[k]].xyz - moments[ids[0]].xyz) * weights[k];
                    for (uint32_t k = 0; k < taps; ++k) {
                        simd::double3 delta = moments[ids[k]].xyz - mean;
                        variance += (moments[ids[k]].w + simd::dot(delta, delta)) * weights[k];
                    }
                    const size_t dst = static_cast<size_t>(y) * next.width + x;
                    nextMoments[dst] = simd_make_double4(mean.x, mean.y, mean.z, variance);
                    for (uint32_t c = 0; c < 3u; ++c) {
                        double value = prev.rgba[ids[0] * 4u + c];
                        for (uint32_t k = 1; k < taps; ++k) value += (prev.rgba[ids[k] * 4u + c] - prev.rgba[ids[0] * 4u + c]) * weights[k];
                        next.rgba[dst * 4u + c] = static_cast<float>(value);
                    }
                    next.rgba[dst * 4u + 3u] = static_cast<float>(std::round(std::sqrt(variance) * 255.0) / 255.0);
                }
            }
        });
        width = next.width;
        height = next.height;
        moments.swap(nextMoments);
        levels.push_back(std::move(next));
    }
    for (size_t index = quantizeLevel0 ? 0u : 1u; index < levels.size(); ++index) {
        MaterialTextureCpuData::MipLevel& level = levels[index];
        float* texels = level.rgba.data();
        const size_t rowFloats = static_cast<size_t>(level.width) * 4u;
        ForEachRowBand(level.height, level.width, [&](size_t firstRow, size_t endRow) {
            for (size_t i = firstRow * rowFloats; i < endRow * rowFloats; ++i) texels[i] = QuantizeNormalChainValue(texels[i]);
        });
    }
    return levels;
}

std::vector<MaterialTextureCpuData::MipLevel> BuildNormalMipChain(const std::vector<float>& base,
                                                                  uint32_t width, uint32_t height) {
    return BuildNormalMipChain(std::vector<float>(base), width, height);
}

// The byte that FloatsToRgba8Texels writes for a level-0 colour byte of an 8-bit normal map after the chain quantizes
// it (index 0-255), and the byte it writes for level 0's alpha (index 256). Built once by running those same functions
// (Rgba8TexelsToFloats, QuantizeNormalChainValue, FloatsToRgba8Texels) on every byte value.
const std::array<uint8_t, 257>& NormalLevel0ByteTable() {
    static const std::array<uint8_t, 257> table = [] {
        std::array<uint8_t, 256u * 4u> in{};
        for (size_t b = 0; b < 256u; ++b) in[b * 4u] = in[b * 4u + 1u] = in[b * 4u + 2u] = in[b * 4u + 3u] = uint8_t(b);
        std::array<float, 256u * 4u> floats{};
        Rgba8TexelsToFloats(in.data(), 256u, 1u, /*isBgra=*/false, /*decodeSrgb=*/false, floats.data());
        for (size_t b = 0; b < 256u; ++b) {
            for (size_t c = 0; c < 3u; ++c) floats[b * 4u + c] = QuantizeNormalChainValue(floats[b * 4u + c]);
            floats[b * 4u + 3u] = QuantizeNormalChainValue(0.0f);
        }
        std::array<uint8_t, 256u * 4u> out{};
        FloatsToRgba8Texels(floats.data(), 256u, 1u, /*isBgra=*/false, out.data());
        std::array<uint8_t, 257> result{};
        for (size_t b = 0; b < 256u; ++b) result[b] = out[b * 4u];
        result[256] = out[3];
        return result;
    }();
    return table;
}

// A normal map's mip levels as 8-bit RGBA bytes, from its level 0 bytes: the levels that BuildTextureWithNormalMips
// writes for an 8-bit RGBA source, by the same conversions (Rgba8TexelsToFloats as CopyTextureLevelToFloat reads it,
// BuildNormalMipChain, FloatsToRgba8Texels as WriteTextureLevelFromFloat writes it).
// Level 0's bytes come from NormalLevel0ByteTable, the bytes those conversions write for each source byte, so level 0
// is not quantized or converted back texel by texel.
std::vector<std::vector<uint8_t>> BuildNormalMipLevelsRgba8(const uint8_t* rgba, uint32_t width, uint32_t height) {
    std::vector<MaterialTextureCpuData::MipLevel> levels;
    {
        std::vector<float> base(static_cast<size_t>(width) * height * 4u);
        Rgba8TexelsToFloats(rgba, width, height, /*isBgra=*/false, /*decodeSrgb=*/false, base.data());
        levels = BuildNormalMipChain(std::move(base), width, height, /*quantizeLevel0=*/false);
    }
    std::vector<std::vector<uint8_t>> bytes;
    bytes.reserve(levels.size());
    {
        const std::array<uint8_t, 257>& table = NormalLevel0ByteTable();
        const size_t texels = static_cast<size_t>(width) * height;
        bytes.emplace_back(rgba, rgba + texels * 4u);
        uint8_t* level0 = bytes.back().data();
        ForEachRowBand(height, width, [&](size_t firstRow, size_t endRow) {
            for (size_t i = firstRow * width; i < endRow * width; ++i) {
                level0[i * 4u] = table[level0[i * 4u]];
                level0[i * 4u + 1u] = table[level0[i * 4u + 1u]];
                level0[i * 4u + 2u] = table[level0[i * 4u + 2u]];
                level0[i * 4u + 3u] = table[256];
            }
        });
        std::vector<float>().swap(levels[0].rgba);
    }
    for (size_t index = 1u; index < levels.size(); ++index) {
        MaterialTextureCpuData::MipLevel& level = levels[index];
        bytes.emplace_back(static_cast<size_t>(level.width) * level.height * 4u);
        FloatsToRgba8Texels(level.rgba.data(), level.width, level.height, /*isBgra=*/false, bytes.back().data());
        std::vector<float>().swap(level.rgba);
    }
    return bytes;
}

// The texture BuildTextureWithNormalMips makes when mipmapped, from levels that BuildNormalMipLevelsRgba8 built.
id<MTLTexture> CreateNormalMipTexture(id<MTLDevice> device, NSUInteger width, NSUInteger height,
                                      const std::vector<std::vector<uint8_t>>& levels) {
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:width height:height mipmapped:YES];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
    if (!texture || levels.size() != texture.mipmapLevelCount) return nil;
    for (NSUInteger level = 0u; level < levels.size(); ++level) {
        const NSUInteger levelWidth = MipDimension(width, level);
        const NSUInteger levelHeight = MipDimension(height, level);
        if (levels[level].size() != static_cast<size_t>(levelWidth) * levelHeight * 4u) return nil;
        [texture replaceRegion:MTLRegionMake2D(0, 0, levelWidth, levelHeight)
                   mipmapLevel:level
                     withBytes:levels[level].data()
                   bytesPerRow:levelWidth * 4u];
    }
    return texture;
}

id<MTLTexture> BuildTextureWithNormalMips(id<MTLDevice> device, id<MTLTexture> source, bool mipmapped) {
    if (!mipmapped) {
        std::vector<uint8_t> texels(static_cast<size_t>(source.width) * source.height * 4u);
        const MTLRegion region = MTLRegionMake2D(0, 0, source.width, source.height);
        [source getBytes:texels.data() bytesPerRow:source.width * 4u fromRegion:region mipmapLevel:0];
        for (size_t i = 3u; i < texels.size(); i += 4u) texels[i] = 0u;
        [source replaceRegion:region mipmapLevel:0 withBytes:texels.data() bytesPerRow:source.width * 4u];
        return source;
    }
    std::vector<float> base;
    if (!CopyTextureLevelToFloat(source, 0u, /*decodeSrgb=*/false, base)) return nil;
    const auto levels =
        BuildNormalMipChain(std::move(base), static_cast<uint32_t>(source.width), static_cast<uint32_t>(source.height));
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:source.width height:source.height mipmapped:YES];
    desc.usage = MTLTextureUsageShaderRead;
    desc.storageMode = MTLStorageModeShared;
    id<MTLTexture> texture = [device newTextureWithDescriptor:desc];
    for (NSUInteger level = 0u; level < levels.size(); ++level)
        if (!WriteTextureLevelFromFloat(texture, level, levels[level].rgba)) return nil;
    return texture;
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

// Decodes a Radiance RLE RGBE file to RGBA32 floats (alpha 1).
bool DecodeRadianceHdrFloat(const std::string& path,
                            uint32_t expectedWidth,
                            uint32_t expectedHeight,
                            std::vector<float>& outRgba) {
    std::ifstream file(path, std::ios::binary);
    std::string line;
    bool rgbe = false;
    uint32_t width = 0;
    uint32_t height = 0;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line == "FORMAT=32-bit_rle_rgbe") {
            rgbe = true;
        } else if (line.rfind("-Y ", 0) == 0) {
            std::istringstream stream(line);
            std::string yToken;
            std::string xToken;
            stream >> yToken >> height >> xToken >> width;
            if (xToken != "+X") {
                return false;
            }
            break;
        }
    }
    if (!rgbe || width != expectedWidth || height != expectedHeight || width == 0u || width > 32767u) {
        return false;
    }
    outRgba.assign(static_cast<size_t>(width) * height * 4u, 1.0f);
    std::vector<uint8_t> scanline(static_cast<size_t>(width) * 4u);
    for (uint32_t y = 0; y < height; ++y) {
        uint8_t header[4] = {};
        if (!file.read(reinterpret_cast<char*>(header), 4) || header[0] != 2u || header[1] != 2u ||
            ((static_cast<uint32_t>(header[2]) << 8u) | header[3]) != width) {
            return false;
        }
        for (uint32_t channel = 0; channel < 4u; ++channel) {
            for (uint32_t x = 0; x < width;) {
                int count = file.get();
                if (count <= 0) {
                    return false;
                }
                const bool run = count > 128;
                const uint32_t length = static_cast<uint32_t>(run ? count - 128 : count);
                if (x + length > width) {
                    return false;
                }
                int value = run ? file.get() : 0;
                for (uint32_t i = 0; i < length; ++i, ++x) {
                    if (!run) {
                        value = file.get();
                    }
                    if (value < 0) {
                        return false;
                    }
                    scanline[x * 4u + channel] = static_cast<uint8_t>(value);
                }
            }
        }
        for (uint32_t x = 0; x < width; ++x) {
            const uint8_t* texel = &scanline[x * 4u];
            float* dst = &outRgba[(static_cast<size_t>(y) * width + x) * 4u];
            for (uint32_t c = 0; c < 3u; ++c) {
                dst[c] = texel[3] == 0u ? 0.0f : std::ldexp(static_cast<float>(texel[c]), texel[3] - 136);
            }
        }
    }
    return true;
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

// Material image prefetch (SceneResources::beginMaterialImagePrefetch). Worker threads decode the listed images in
// order: at most kWindow of them in flight or decoded and not yet taken, and no new decode starts while the decoded,
// untaken images hold kMaxHeldBytes or more. That bounds the extra memory to a few decoded images (a 4096^2 RGBA8 image
// is 64 MiB, an 8192^2 one 256 MiB) plus ImageIO's buffers for the decodes in flight. A request takes a finished decode, waits for one in flight, or (for an image
// not listed or not started yet) decodes it on the calling thread as before. Either way the decoder and its input are
// the same, so the decoded bytes are too. A request also drops every listed image before it that was not taken: the
// loader skipped it (a texture policy, a cache hit on another key), and it must not hold a window slot.
// Only the first image of each decode identity is listed: the SHA-256 of a payload (the loader's `datahash:` key) or a
// file's import-manifest output hash (`hash:`). A later image with the same identity is a cache hit at its request, so
// decoding it would be wasted (Bistro: 314 of 591 embedded images are byte-identical copies). The payload digests are
// computed once, in parallel, and handed to the requests (payloadHash), which would otherwise hash each payload again.
// A KTX2 file is listed when its request parses it (a normal map, or a file with no DDS source): the workers run
// Ktx2::ParseRgba8File, and the request uploads the parsed level (takeKtx2) as LoadKtx2TextureFromPath would.
// When the loader builds mipmapped normal maps (buildNormalChains), a normal map's mip chain is also built ahead from its
// parsed KTX2 or decoded level 0 (BuildNormalMipLevelsRgba8), one at a time on a serial queue: its float and moment
// buffers take about 75 bytes per texel at their peak, so chains in parallel would multiply that. The request uploads the
// levels (CreateNormalMipTexture) instead of running BuildTextureWithNormalMips on the loading thread. A request that
// changes the decoded bytes first (FlipDecodedImageRows), or that is not a mipmapped normal map, does not use the chain.
struct MaterialImagePrefetch {
    static constexpr size_t kWindow = 6u;
    static constexpr size_t kMaxHeldBytes = size_t{384} << 20;

    enum class Stage { Pending, Running, Done, Taken };
    struct Item {
        MaterialImagePrefetchSource source;
        Stage stage = Stage::Pending;
        bool abandoned = false;  // dropped while its decode was running: freed when the decode finishes
        bool ok = false;
        bool ktx2 = false;  // parsed by Ktx2::ParseRgba8File into ktx2Image, not decoded into image
        DecodedImageRgba8 image;
        Ktx2::ImageData ktx2Image;
        std::vector<std::vector<uint8_t>> normalLevels;  // built ahead for a normal map, or empty
        std::string error;
        size_t bytes() const {
            size_t total = image.rgba.size() + ktx2Image.rgba.size();
            for (const std::vector<uint8_t>& level : normalLevels) total += level.size();
            return total;
        }
        void clearResults() {
            image = DecodedImageRgba8{};
            ktx2Image = Ktx2::ImageData{};
            normalLevels = {};
        }
    };
    struct State {
        std::mutex mutex;
        std::condition_variable changed;
        std::vector<Item> items;
        std::unordered_map<std::string, size_t> byPath;
        std::map<std::pair<const uint8_t*, size_t>, size_t> byData;
        size_t next = 0u;
        size_t requested = 0u;  // items before this were taken or dropped
        size_t outstanding = 0u;  // running, or done and not taken
        size_t running = 0u;
        size_t heldBytes = 0u;  // decoded and not taken
        size_t decodedAhead = 0u;
        size_t taken = 0u;
        double waitedSeconds = 0.0;  // time requests spent waiting for decodes in flight
        size_t normalChainsAhead = 0u;
        size_t normalChainsUnused = 0u;  // built ahead, then not used by their request
        bool buildNormalChains = false;
        dispatch_queue_t chainQueue = nil;  // serial: the normal-map chains built ahead
        bool stopped = false;
    };
    std::shared_ptr<State> state = std::make_shared<State>();
    // Written by the constructor only, then read by requests on the loading thread.
    std::map<std::pair<const uint8_t*, size_t>, std::string> payloadHashes;
    size_t duplicatesSkipped = 0u;
    size_t ktx2Listed = 0u;
    double listingSeconds = 0.0;  // the constructor: hashing, identities, KTX2 rules

    MaterialImagePrefetch(const std::vector<MaterialImagePrefetchSource>& sources, bool buildNormalChains) {
        const auto listingStart = std::chrono::steady_clock::now();
        state->buildNormalChains = buildNormalChains;
        if (buildNormalChains) {
            state->chainQueue = dispatch_queue_create(
                "PathTracer.MaterialImagePrefetch.NormalChains",
                dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0));
        }
        std::vector<std::pair<const uint8_t*, size_t>> payloads;
        for (const MaterialImagePrefetchSource& source : sources) {
            if (source.data && source.size != 0u && !IsKtx2Data(source.data, source.size) &&
                payloadHashes.emplace(std::make_pair(source.data, source.size), std::string()).second) {
                payloads.emplace_back(source.data, source.size);
            }
        }
        if (!payloads.empty()) {
            std::vector<std::string> digests(payloads.size());
            std::string* digestOut = digests.data();
            const std::pair<const uint8_t*, size_t>* payloadIn = payloads.data();
            dispatch_apply(payloads.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
                digestOut[i] = HashTexturePayload(payloadIn[i].first, payloadIn[i].second);
            });
            for (size_t i = 0; i < payloads.size(); ++i) {
                payloadHashes[payloads[i]] = std::move(digests[i]);
            }
        }
        std::unordered_set<std::string> listedIdentities;
        std::set<std::pair<const uint8_t*, size_t>> duplicateData;
        std::unordered_set<std::string> duplicatePaths;
        for (const MaterialImagePrefetchSource& source : sources) {
            Item item;
            if (source.data) {
                if (source.size == 0u || IsKtx2Data(source.data, source.size) ||
                    state->byData.count({source.data, source.size})) {
                    continue;
                }
                if (!listedIdentities.insert("datahash:" + payloadHashes.at({source.data, source.size})).second) {
                    duplicateData.insert({source.data, source.size});
                    continue;
                }
                item.source = source;
                state->byData[{source.data, source.size}] = state->items.size();
            } else {
                if (source.path.empty()) {
                    continue;
                }
                const std::string canonicalPath = CanonicalizePath(source.path);
                // A KTX2 file is parsed by its request only for a normal map or when it has no DDS source; otherwise
                // the request uploads the DDS (addMaterialTextureFromFile).
                const bool ktx2 = HasExtensionCaseInsensitive(canonicalPath, ".ktx2");
                if (ktx2 && !source.normalMap && !ResolveCompressedSourceTexturePath(canonicalPath).empty()) {
                    continue;
                }
                if (state->byPath.count(canonicalPath)) {
                    continue;
                }
                item.ktx2 = ktx2;
                const std::string outputHash = ResolveTextureOutputHash(canonicalPath);
                if (!listedIdentities.insert(outputHash.empty() ? canonicalPath : "hash:" + outputHash).second) {
                    duplicatePaths.insert(canonicalPath);
                    continue;
                }
                item.source.path = canonicalPath;
                item.source.normalMap = source.normalMap;
                state->byPath[canonicalPath] = state->items.size();
            }
            ktx2Listed += item.ktx2 ? 1u : 0u;
            state->items.push_back(std::move(item));
        }
        duplicatesSkipped = duplicateData.size() + duplicatePaths.size();
        listingSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - listingStart).count();
        std::lock_guard<std::mutex> lock(state->mutex);
        LaunchLocked(state);
    }

    ~MaterialImagePrefetch() { stop(); }

    // The SHA-256 hex digest of a listed payload (HashTexturePayload), or null for a payload the prefetch did not hash.
    const std::string* payloadHash(const uint8_t* data, size_t size) const {
        auto found = payloadHashes.find({data, size});
        return found == payloadHashes.end() ? nullptr : &found->second;
    }

    // Starts decodes until the window is full. Called with the mutex held.
    static void LaunchLocked(const std::shared_ptr<State>& s) {
        while (!s->stopped && s->outstanding < kWindow && s->heldBytes < kMaxHeldBytes && s->next < s->items.size()) {
            const size_t index = s->next++;
            if (s->items[index].stage != Stage::Pending) {
                continue;
            }
            s->items[index].stage = Stage::Running;
            ++s->outstanding;
            ++s->running;
            const std::shared_ptr<State> shared = s;
            const MaterialImagePrefetchSource source = s->items[index].source;
            const bool ktx2 = s->items[index].ktx2;
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
                DecodedImageRgba8 image{};
                Ktx2::ImageData ktx2Image;
                std::string error;
                bool ok = false;
                @autoreleasepool {
                    ok = ktx2          ? Ktx2::ParseRgba8File(source.path, ktx2Image, error)
                         : source.data ? DecodeImageDataToRgba8(source.data, source.size, image, error)
                                       : DecodeImageFileToRgba8(source.path, image, error);
                }
                if (ok && source.normalMap && shared->buildNormalChains) {
                    auto parsed = std::make_shared<Ktx2::ImageData>(std::move(ktx2Image));
                    auto decoded = std::make_shared<DecodedImageRgba8>(std::move(image));
                    dispatch_async(shared->chainQueue, ^{
                        bool abandoned = false;
                        {
                            std::lock_guard<std::mutex> lock(shared->mutex);
                            abandoned = shared->items[index].abandoned;
                        }
                        std::vector<std::vector<uint8_t>> levels;
                        if (!abandoned) {
                            levels = ktx2 ? BuildNormalMipLevelsRgba8(parsed->rgba.data(), parsed->width, parsed->height)
                                          : BuildNormalMipLevelsRgba8(decoded->rgba.data(), decoded->width,
                                                                      decoded->height);
                        }
                        Finish(shared, index, true, std::move(*decoded), std::move(*parsed), std::move(levels),
                               std::string{});
                    });
                    return;
                }
                Finish(shared, index, ok, std::move(image), std::move(ktx2Image), {}, std::move(error));
            });
        }
    }

    // Stores a finished item's results, or frees them if the item was dropped while it ran.
    static void Finish(const std::shared_ptr<State>& shared, size_t index, bool ok, DecodedImageRgba8&& image,
                       Ktx2::ImageData&& ktx2Image, std::vector<std::vector<uint8_t>>&& normalLevels,
                       std::string&& error) {
        std::lock_guard<std::mutex> lock(shared->mutex);
        Item& item = shared->items[index];
        ++shared->decodedAhead;
        --shared->running;
        if (item.abandoned) {
            item.stage = Stage::Taken;
            --shared->outstanding;
            LaunchLocked(shared);
        } else {
            item.ok = ok;
            item.image = std::move(image);
            item.ktx2Image = std::move(ktx2Image);
            item.normalLevels = std::move(normalLevels);
            shared->heldBytes += item.bytes();
            shared->normalChainsAhead += item.normalLevels.empty() ? 0u : 1u;
            item.error = std::move(error);
            item.stage = Stage::Done;
        }
        shared->changed.notify_all();
    }

    // True when the prefetch owns this image: then `ok`, `out` and `error` are the decoder's results. False when the
    // caller must decode it itself. `normalLevels`, when given, receives the mip chain built ahead for a normal map
    // (BuildNormalMipLevelsRgba8 of `out`), or is left empty.
    bool take(const std::string& path, const uint8_t* data, size_t size,
              DecodedImageRgba8& out, bool& ok, std::string& error,
              std::vector<std::vector<uint8_t>>* normalLevels = nullptr) {
        std::unique_lock<std::mutex> lock(state->mutex);
        Item* item = claim(lock, path, data, size, /*ktx2=*/false);
        if (!item) return false;
        const size_t bytes = item->bytes();
        out = std::move(item->image);
        if (normalLevels) *normalLevels = std::move(item->normalLevels);
        ok = item->ok;
        error = std::move(item->error);
        release(*item, bytes);
        return true;
    }

    // The same for a KTX2 file: `out` and `error` are Ktx2::ParseRgba8File's results. `normalLevels` receives the mip
    // chain built ahead for a normal map (BuildNormalMipLevelsRgba8), or is left empty.
    bool takeKtx2(const std::string& path, Ktx2::ImageData& out, bool& ok, std::string& error,
                  std::vector<std::vector<uint8_t>>& normalLevels) {
        std::unique_lock<std::mutex> lock(state->mutex);
        Item* item = claim(lock, path, nullptr, 0u, /*ktx2=*/true);
        if (!item) return false;
        const size_t bytes = item->bytes();
        out = std::move(item->ktx2Image);
        normalLevels = std::move(item->normalLevels);
        ok = item->ok;
        error = std::move(item->error);
        release(*item, bytes);
        return true;
    }

    // Counts a chain built ahead that its request did not use, for the [TexturePrefetch] line.
    void noteNormalChainUnused() {
        std::lock_guard<std::mutex> lock(state->mutex);
        ++state->normalChainsUnused;
    }

private:
    // The listed item a request owns, its decode finished, or null when the caller must decode the image itself.
    // Called with the mutex held.
    Item* claim(std::unique_lock<std::mutex>& lock, const std::string& path, const uint8_t* data, size_t size,
                bool ktx2) {
        size_t index = 0u;
        if (data) {
            auto found = state->byData.find({data, size});
            if (found == state->byData.end()) return nullptr;
            index = found->second;
        } else {
            auto found = state->byPath.find(path);
            if (found == state->byPath.end()) return nullptr;
            index = found->second;
        }
        // Drop the listed images before this one that no request took.
        for (; state->requested < index; ++state->requested) {
            Item& skipped = state->items[state->requested];
            if (skipped.stage == Stage::Pending) {
                skipped.stage = Stage::Taken;
            } else if (skipped.stage == Stage::Done) {
                state->heldBytes -= skipped.bytes();
                skipped.clearResults();
                skipped.stage = Stage::Taken;
                --state->outstanding;
            } else if (skipped.stage == Stage::Running) {
                skipped.abandoned = true;
            }
        }
        if (state->requested == index) {
            ++state->requested;
        }
        LaunchLocked(state);
        Item& item = state->items[index];
        if (item.stage == Stage::Taken || item.abandoned || item.ktx2 != ktx2) {
            return nullptr;
        }
        if (item.stage == Stage::Pending) {
            item.stage = Stage::Taken;
            return nullptr;
        }
        if (item.stage != Stage::Done) {
            const auto waitStart = std::chrono::steady_clock::now();
            state->changed.wait(lock, [&] { return item.stage == Stage::Done; });
            state->waitedSeconds +=
                std::chrono::duration<double>(std::chrono::steady_clock::now() - waitStart).count();
        }
        return &item;
    }

    // Frees a claimed item's slot after its results were moved out; `bytes` is item.bytes() before the move. Called with
    // the mutex held.
    void release(Item& item, size_t bytes) {
        state->heldBytes -= bytes;
        item.clearResults();
        item.stage = Stage::Taken;
        ++state->taken;
        --state->outstanding;
        LaunchLocked(state);
    }

public:

    // Starts nothing more and waits for the decodes in flight: their inputs (a glTF buffer) must outlive them. No request
    // takes an image after this, so the decodes in flight are dropped, and a normal-map chain not started yet is skipped.
    void stop() {
        std::unique_lock<std::mutex> lock(state->mutex);
        state->stopped = true;
        for (Item& item : state->items) {
            if (item.stage == Stage::Running) item.abandoned = true;
        }
        state->changed.wait(lock, [&] { return state->running == 0u; });
    }
};

namespace {

bool DecodeMaterialImageFile(MaterialImagePrefetch* prefetch, const std::string& canonicalPath,
                             DecodedImageRgba8& out, std::string& error,
                             std::vector<std::vector<uint8_t>>* normalLevels = nullptr) {
    bool ok = false;
    if (prefetch && prefetch->take(canonicalPath, nullptr, 0u, out, ok, error, normalLevels)) {
        return ok;
    }
    return DecodeImageFileToRgba8(canonicalPath, out, error);
}

bool DecodeMaterialImageData(MaterialImagePrefetch* prefetch, const uint8_t* data, size_t size,
                             DecodedImageRgba8& out, std::string& error,
                             std::vector<std::vector<uint8_t>>* normalLevels = nullptr) {
    bool ok = false;
    if (prefetch && prefetch->take(std::string(), data, size, out, ok, error, normalLevels)) {
        return ok;
    }
    return DecodeImageDataToRgba8(data, size, out, error);
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
    m_materialTextureMinAlphaByKey.clear();
    m_materialSamplerIndices.clear();
    m_materialTextureInfoBuffer = nil;
    m_textureUploadBuffer = nil;
    m_forceSoftwareOverride = false;
    m_hardwareDebugSoftwareFallbackEnabled = false;
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
    m_texturePolicySkippedCount = 0;
    m_footprintMipChainCount = 0;
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
                                     float spectralDispersionScale,
                                     float spectralAbsorptionScale,
                                     uint32_t spectralMaterialPolicy,
                                     uint32_t spectralTexturePolicy,
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
    // Effective flake coverage of the CarPaint under-layer: the density-derived coverage scaled by the flake
    // reflectance; the closure weights flakes by it and the base by its complement.
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
    material.spectralParams =
        simd_make_float4(std::max(spectralDispersionScale, 0.0f),
                         std::max(spectralAbsorptionScale, 0.0f),
                         static_cast<float>(spectralMaterialPolicy),
                         static_cast<float>(spectralTexturePolicy));
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
                       0.0f,
                       1.0f,
                       0u,
                       0u,
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
        const bool buildMipmaps = m_stagedUploadMode
            ? false
            : (m_textureLoadPolicy.maxDimension == 0u &&
               deferred.semantic != MaterialTextureSemantic::Normal &&
               deferred.semantic != MaterialTextureSemantic::Orm && !IsNativeScalarTexture(deferred.semantic));
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
        if (deferred.semantic == MaterialTextureSemantic::Normal) texture = BuildTextureWithNormalMips(m_device, texture, !m_stagedUploadMode && m_textureLoadPolicy.maxDimension == 0u);
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
        descriptor.maxAnisotropy = effective.maxAnisotropy;
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
    const bool allowGeneratedMipmaps = (m_textureLoadPolicy.maxDimension == 0u) && !IsNativeScalarTexture(semantic);
    if (m_device && !m_stagedUploadMode && allowGeneratedMipmaps) {
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
                texture = WithMaterialMipsOrWarn(m_device, m_commandQueue, texture, label);
            }
        } else if (texture.mipmapLevelCount <= 1 &&
                   BytesPerPixel(texture.pixelFormat) > 0u &&
                   m_commandQueue) {
            texture = WithMaterialMipsOrWarn(m_device, m_commandQueue, texture, label);
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
        if (texture.mipmapLevelCount > 1u && BytesPerPixel(texture.pixelFormat) > 0u &&
            !MipChain::BoxChainKeepsMean(static_cast<uint32_t>(texture.width), static_cast<uint32_t>(texture.height))) {
            ++m_footprintMipChainCount;
        }
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

int SceneResources::materialTextureMinAlpha(uint32_t index) const {
    for (const auto& entry : m_materialTextureIndex) {
        if (entry.second == index) {
            const auto it = m_materialTextureMinAlphaByKey.find(entry.first);
            return it == m_materialTextureMinAlphaByKey.end() ? -1 : static_cast<int>(it->second);
        }
    }
    return -1;
}

void SceneResources::beginMaterialImagePrefetch(const std::vector<MaterialImagePrefetchSource>& sources) {
    endMaterialImagePrefetch();
    if (!sources.empty()) {
        // The mipmapped normal-map case of addMaterialTextureFromFile, the one a chain built ahead serves.
        const bool buildNormalChains = m_device && !m_stagedUploadMode && m_textureLoadPolicy.maxDimension == 0u;
        m_materialImagePrefetch = std::make_shared<MaterialImagePrefetch>(sources, buildNormalChains);
    }
}

void SceneResources::endMaterialImagePrefetch() {
    if (!m_materialImagePrefetch) {
        return;
    }
    m_materialImagePrefetch->stop();
    const MaterialImagePrefetch::State& state = *m_materialImagePrefetch->state;
    NSLog(@"[TexturePrefetch] %zu images listed (%zu of them KTX2 files to parse; %zu duplicates of a listed image not "
          @"listed), %zu decoded ahead (window %zu, %zu MiB held at most), %zu taken by requests; the rest were skipped "
          @"by the loader or decoded by their request; %zu normal-map mip chains built ahead (%zu not used by their "
          @"request); listing took %.0f ms, requests waited %.0f ms for decodes",
          state.items.size(), m_materialImagePrefetch->ktx2Listed, m_materialImagePrefetch->duplicatesSkipped,
          state.decodedAhead,
          MaterialImagePrefetch::kWindow, MaterialImagePrefetch::kMaxHeldBytes >> 20, state.taken,
          state.normalChainsAhead, state.normalChainsUnused, m_materialImagePrefetch->listingSeconds * 1000.0,
          state.waitedSeconds * 1000.0);
    m_materialImagePrefetch.reset();
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
        ++m_texturePolicySkippedCount;
        if (TexturePolicyVerboseLoggingEnabled()) {
            std::cout << "[TexturePolicy] skip " << path << " (" << TextureSemanticLabel(semantic)
                      << "): " << policyReason << std::endl;
        }
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
            if (!(IsNativeScalarTexture(semantic) ? DecodeRawScalarFile(canonicalPath, decoded, stageError) : DecodeMaterialImageFile(m_materialImagePrefetch.get(), canonicalPath, decoded, stageError))) {
                if (errorMessage) {
                    *errorMessage = stageError;
                }
                return kInvalidTextureIndex;
            }
            ConvertNativeScalarTexture(decoded, semantic);
            if (flipFerrariF14TextureOrigin) {
                FlipDecodedImageRows(decoded);
            }
            if (deriveBlackKeyAlpha) {
                DeriveBlackKeyAlpha(decoded);
            }
            if (deriveBorderBlackKeyAlpha) {
                DeriveBorderBlackKeyAlpha(decoded);
            }
            m_materialTextureMinAlphaByKey[key] = MinAlphaOfRgba8(decoded);
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
    std::vector<std::vector<uint8_t>> normalLevels;  // a normal map's mip chain, built ahead by the prefetch
    bool levelZeroTransformed = false;  // the decoded bytes were changed after the chain was built from them
    if (HasExtensionCaseInsensitive(canonicalPath, ".ktx2")) {
        std::string loadError;
        const std::string compressedSourcePath = ResolveCompressedSourceTexturePath(canonicalPath);
        if (!compressedSourcePath.empty() && semantic != MaterialTextureSemantic::Normal) {
            texture = LoadCompressedDdsTextureFromPath(m_device, m_commandQueue, compressedSourcePath, srgb, loadError);
            if (texture) {
                loadError.clear();
            } else {
                std::cerr << "[SceneResources] warning: " << compressedSourcePath << ", the DDS source of "
                          << canonicalPath << ": " << loadError << ". Loading the KTX2 instead." << std::endl;
            }
        }
        if (!texture) {
            Ktx2::ImageData parsed;
            bool parsedOk = false;
            if (m_materialImagePrefetch && m_materialImagePrefetch->takeKtx2(canonicalPath, parsed, parsedOk, loadError,
                                                                                  normalLevels)) {
                // Parsed ahead by the prefetch: the same parser and upload as LoadKtx2TextureFromPath.
                if (parsedOk) {
                    texture = CreateTextureFromRgba8(m_device, m_commandQueue, parsed, srgb, loadError);
                } else {
                    loadError = "Material texture load failed: " + loadError;
                }
            } else {
                texture = LoadKtx2TextureFromPath(m_device, m_commandQueue, canonicalPath, srgb, loadError);
            }
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
        if (!(IsNativeScalarTexture(semantic) ? DecodeRawScalarFile(canonicalPath, decoded, loadError) : DecodeMaterialImageFile(m_materialImagePrefetch.get(), canonicalPath, decoded, loadError, &normalLevels))) {
            if (errorMessage) {
                *errorMessage = loadError;
            }
            return kInvalidTextureIndex;
        }
        ConvertNativeScalarTexture(decoded, semantic);
        if (flipFerrariF14TextureOrigin) {
            FlipDecodedImageRows(decoded);
        }
        if (deriveBlackKeyAlpha) {
            DeriveBlackKeyAlpha(decoded);
        }
        if (deriveBorderBlackKeyAlpha) {
            DeriveBorderBlackKeyAlpha(decoded);
        }
        levelZeroTransformed = flipFerrariF14TextureOrigin || deriveBlackKeyAlpha || deriveBorderBlackKeyAlpha;
        m_materialTextureMinAlphaByKey[key] = MinAlphaOfRgba8(decoded);
        const bool buildMipmaps = m_stagedUploadMode
            ? false
            : (m_textureLoadPolicy.maxDimension == 0u &&
               semantic != MaterialTextureSemantic::Normal &&
               semantic != MaterialTextureSemantic::Orm && !IsNativeScalarTexture(semantic));
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

    if (!IsNativeScalarTexture(semantic) && m_textureLoadPolicy.maxDimension > 0u &&
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
                if (TexturePolicyVerboseLoggingEnabled()) {
                    std::cout << "[TexturePolicy] " << canonicalPath << " ("
                              << TextureSemanticLabel(semantic)
                              << "): " << inventory.decision << std::endl;
                }
            }
        }
    }

    if (semantic == MaterialTextureSemantic::Normal) {
        const bool mipmapped = !m_stagedUploadMode && m_textureLoadPolicy.maxDimension == 0u;
        const bool chainAhead = mipmapped && !normalLevels.empty() && !levelZeroTransformed;
        texture = chainAhead ? CreateNormalMipTexture(m_device, texture.width, texture.height, normalLevels)
                             : BuildTextureWithNormalMips(m_device, texture, mipmapped);
        if (!texture) return kInvalidTextureIndex;
        if (chainAhead) normalLevels.clear();
    }
    if (!normalLevels.empty() && m_materialImagePrefetch) m_materialImagePrefetch->noteNormalChainUnused();
    std::string budgetReason;
    const uint64_t estimatedBytes = EstimateTextureMemoryBytes(texture);
    if (shouldSkipTextureForBudget(semantic, estimatedBytes, budgetReason)) {
        ++m_texturePolicySkippedCount;
        if (TexturePolicyVerboseLoggingEnabled()) {
            std::cout << "[TexturePolicy] skip " << canonicalPath << " ("
                      << TextureSemanticLabel(semantic) << "): " << budgetReason << std::endl;
        }
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
        ++m_texturePolicySkippedCount;
        if (TexturePolicyVerboseLoggingEnabled()) {
            std::cout << "[TexturePolicy] skip " << label << " (" << TextureSemanticLabel(semantic)
                      << "): " << policyReason << std::endl;
        }
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
    const std::string* prefetchedHash =
        m_materialImagePrefetch ? m_materialImagePrefetch->payloadHash(data, size) : nullptr;
    const std::string dedupeIdentity =
        std::string("datahash:") + (prefetchedHash ? *prefetchedHash : HashTexturePayload(data, size));
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
            if (!(IsNativeScalarTexture(semantic) ? DecodeRawScalarData(data, size, decoded, stageError) : DecodeMaterialImageData(m_materialImagePrefetch.get(), data, size, decoded, stageError))) {
                if (errorMessage) {
                    *errorMessage = stageError;
                }
                return kInvalidTextureIndex;
            }
            ConvertNativeScalarTexture(decoded, semantic);
            if (flipFerrariF14TextureOrigin) {
                FlipDecodedImageRows(decoded);
            }
            if (deriveBlackKeyAlpha) {
                DeriveBlackKeyAlpha(decoded);
            }
            if (deriveBorderBlackKeyAlpha) {
                DeriveBorderBlackKeyAlpha(decoded);
            }
            m_materialTextureMinAlphaByKey[key] = MinAlphaOfRgba8(decoded);
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
    std::vector<std::vector<uint8_t>> normalLevels;  // a normal map's mip chain, built ahead by the prefetch
    bool levelZeroTransformed = false;  // the decoded bytes were changed after the chain was built from them
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
        if (!(IsNativeScalarTexture(semantic) ? DecodeRawScalarData(data, size, decoded, loadError) : DecodeMaterialImageData(m_materialImagePrefetch.get(), data, size, decoded, loadError, &normalLevels))) {
            if (errorMessage) {
                *errorMessage = loadError;
            }
            return kInvalidTextureIndex;
        }
        ConvertNativeScalarTexture(decoded, semantic);
        if (flipFerrariF14TextureOrigin) {
            FlipDecodedImageRows(decoded);
        }
        if (deriveBlackKeyAlpha) {
            DeriveBlackKeyAlpha(decoded);
        }
        if (deriveBorderBlackKeyAlpha) {
            DeriveBorderBlackKeyAlpha(decoded);
        }
        levelZeroTransformed = flipFerrariF14TextureOrigin || deriveBlackKeyAlpha || deriveBorderBlackKeyAlpha;
        m_materialTextureMinAlphaByKey[key] = MinAlphaOfRgba8(decoded);
        const bool buildMipmaps = m_stagedUploadMode
            ? false
            : (m_textureLoadPolicy.maxDimension == 0u &&
               semantic != MaterialTextureSemantic::Normal &&
               semantic != MaterialTextureSemantic::Orm && !IsNativeScalarTexture(semantic));
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

    if (!IsNativeScalarTexture(semantic) && m_textureLoadPolicy.maxDimension > 0u &&
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
                if (TexturePolicyVerboseLoggingEnabled()) {
                    std::cout << "[TexturePolicy] " << label << " ("
                              << TextureSemanticLabel(semantic)
                              << "): " << inventory.decision << std::endl;
                }
            }
        }
    }

    if (semantic == MaterialTextureSemantic::Normal) {
        const bool mipmapped = !m_stagedUploadMode && m_textureLoadPolicy.maxDimension == 0u;
        const bool chainAhead = mipmapped && !normalLevels.empty() && !levelZeroTransformed;
        texture = chainAhead ? CreateNormalMipTexture(m_device, texture.width, texture.height, normalLevels)
                             : BuildTextureWithNormalMips(m_device, texture, mipmapped);
        if (!texture) return kInvalidTextureIndex;
        if (chainAhead) normalLevels.clear();
    }
    if (!normalLevels.empty() && m_materialImagePrefetch) m_materialImagePrefetch->noteNormalChainUnused();
    std::string budgetReason;
    const uint64_t estimatedBytes = EstimateTextureMemoryBytes(texture);
    if (shouldSkipTextureForBudget(semantic, estimatedBytes, budgetReason)) {
        ++m_texturePolicySkippedCount;
        if (TexturePolicyVerboseLoggingEnabled()) {
            std::cout << "[TexturePolicy] skip " << label << " ("
                      << TextureSemanticLabel(semantic) << "): " << budgetReason << std::endl;
        }
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

    auto flipRowsInPlace = [](auto& rgba, uint32_t width, uint32_t height) {
        if (width == 0u || height == 0u) {
            return;
        }
        const size_t rowValues = static_cast<size_t>(width) * 4u;
        std::decay_t<decltype(rgba)> flipped(rgba.size());
        for (uint32_t y = 0; y < height; ++y) {
            const size_t srcOffset = static_cast<size_t>(y) * rowValues;
            const size_t dstOffset = static_cast<size_t>(height - 1u - y) * rowValues;
            std::copy_n(rgba.data() + srcOffset, rowValues, flipped.data() + dstOffset);
        }
        rgba.swap(flipped);
    };

    // 8-bit RGBA to float, with sRGB colour decoded to linear.
    auto decodeTexels = [](const std::vector<uint8_t>& rgba, size_t pixelCount, bool srgb, std::vector<float>& texels) {
        texels.resize(pixelCount * 4u, 0.0f);
        for (size_t i = 0; i < pixelCount; ++i) {
            for (size_t c = 0; c < 3u; ++c) {
                const float value = static_cast<float>(rgba[i * 4u + c]) / 255.0f;
                texels[i * 4u + c] = srgb ? SrgbToLinear(value) : value;
            }
            texels[i * 4u + 3u] = static_cast<float>(rgba[i * 4u + 3u]) / 255.0f;
        }
    };

    auto populateRgba8 = [&](const std::vector<uint8_t>& rgba, uint32_t width, uint32_t height,
                             bool srgb, const MaterialTextureSamplerDesc& sampler, bool flipRows, bool orm,
                             bool normal) {
        out.width = width;
        out.height = height;
        out.srgb = srgb;
        out.samplerDesc = sampler;
        out.footprintMipChain = !MipChain::BoxChainKeepsMean(width, height);
        decodeTexels(rgba, static_cast<size_t>(out.width) * static_cast<size_t>(out.height), srgb, out.rgba);
        std::vector<uint8_t> level0Bytes = rgba;
        if (flipRows) {
            flipRowsInPlace(out.rgba, out.width, out.height);
            flipRowsInPlace(level0Bytes, out.width, out.height);
        }
        if (normal) {
            out.mipLevels = BuildNormalMipChain(out.rgba, out.width, out.height);
            out.rgba = out.mipLevels.front().rgba;
            out.mipLevels.front().rgba8 = std::move(level0Bytes);
            return true;
        }
        out.mipLevels.resize(1u);
        out.mipLevels[0].width = out.width;
        out.mipLevels[0].height = out.height;
        out.mipLevels[0].rgba = out.rgba;
        out.mipLevels[0].rgba8 = std::move(level0Bytes);
        if (!orm && !MipChain::BoxChainKeepsMean(width, height)) {
            // Not a power of two: decode the same 8-bit levels Metal uploads (CreateTextureFromDecodedRgba8,
            // BuildMaterialTextureWithMips). Row flips commute with the filter: footprints are symmetric.
            for (MipChain::Rgba8Level& level : MipChain::BuildRgba8MipLevels(rgba.data(), width, height, srgb)) {
                MaterialTextureCpuData::MipLevel next{level.width, level.height, {}, {}};
                decodeTexels(level.rgba, static_cast<size_t>(level.width) * level.height, srgb, next.rgba);
                next.rgba8 = std::move(level.rgba);
                if (flipRows) {
                    flipRowsInPlace(next.rgba, next.width, next.height);
                    flipRowsInPlace(next.rgba8, next.width, next.height);
                }
                out.mipLevels.push_back(std::move(next));
            }
            return true;
        }

        while (out.mipLevels.back().width > 1u || out.mipLevels.back().height > 1u) {
            const auto& src = out.mipLevels.back();
            MaterialTextureCpuData::MipLevel next{};
            next.width = std::max(src.width / 2u, 1u);
            next.height = std::max(src.height / 2u, 1u);
            if (orm) {
                const MaterialTextureSamplerDesc effectiveSampler = EffectiveSamplerDesc(&sampler);
                DownsampleOrmMipLevel(src.rgba, src.width, src.height,
                                      next.rgba, next.width, next.height,
                                      effectiveSampler.wrapS, effectiveSampler.wrapT);
                out.mipLevels.push_back(std::move(next));
                continue;
            }
            DownsampleMipLevelBox(src.rgba, src.width, src.height, next.rgba, next.width, next.height);
            out.mipLevels.push_back(std::move(next));
        }
        return true;
    };

    if (index < m_materialTextures.size()) {
        id<MTLTexture> texture = m_materialTextures[index];
        if (!texture) {
            return false;
        }
        auto sourceKtxFallback = [&]() {
            if (index >= m_materialTextureLabels.size()) return false;
            Ktx2::ImageData source;
            std::string error;
            if (!Ktx2::ParseRgba8File(m_materialTextureLabels[index], source, error)) return false;
            const MaterialTextureSamplerDesc sampler = index < m_materialTextureCpuSamplers.size()
                ? m_materialTextureCpuSamplers[index] : MaterialTextureSamplerDesc{};
            return populateRgba8(source.rgba, source.width, source.height,
                                 IsSrgbPixelFormat(texture.pixelFormat), sampler, true, false,
                                 m_textureInventory[index].semantic == MaterialTextureSemantic::Normal);
        };
        // Compressed resident formats cannot be read as individual texels by
        // CopyTextureLevelToFloat. Decode their converted RGBA8 source instead.
        if (BytesPerPixel(texture.pixelFormat) == 0u) return sourceKtxFallback();
        const bool normalMap = m_textureInventory[index].semantic == MaterialTextureSemantic::Normal;
        id<MTLTexture> readableTexture = texture;
        if (!normalMap && texture.storageMode == MTLStorageModePrivate && m_device && m_commandQueue) {
            // Copy the levels as they are: regenerating them with the blit would replace the levels Metal samples
            // (ORM, normal and non-power-of-two chains are built on the CPU) with the blit's.
            readableTexture = CopyTextureLevels(m_device, m_commandQueue, texture, MTLStorageModeShared);
            if (!readableTexture) {
                return sourceKtxFallback();
            }
        }
        const NSUInteger mipCount = std::max<NSUInteger>(texture.mipmapLevelCount, 1u);
        out.mipLevels.resize(mipCount);
        for (NSUInteger level = 0; level < mipCount; ++level) {
            std::vector<float> rgba;
            id<MTLTexture> levelTexture = readableTexture;
            if (normalMap && texture.storageMode == MTLStorageModePrivate)
                levelTexture = CloneTextureFromSourceLevel(m_device, m_commandQueue, texture, level);
            if (!CopyTextureLevelToFloat(levelTexture, levelTexture == readableTexture ? level : 0u,
                                         /*decodeSrgb=*/true, rgba)) {
                return sourceKtxFallback();
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
    const MaterialTextureSamplerDesc sampler = deferred.hasSamplerDesc
        ? deferred.samplerDesc : MaterialTextureSamplerDesc{};
    if (deferred.decodedWidth != 0u && deferred.decodedHeight != 0u && !deferred.decodedRgba.empty()) {
        return populateRgba8(deferred.decodedRgba, deferred.decodedWidth, deferred.decodedHeight,
                             deferred.srgb, sampler, false, deferred.semantic == MaterialTextureSemantic::Orm,
                             deferred.semantic == MaterialTextureSemantic::Normal);
    }
    // Metal samples a non-normal texture that has a DDS source (named in import_manifest.json) through that DDS and its
    // own mip levels; see the texture loader. This copy, which Embree reads, decodes the same levels: they can be far
    // from a chain filtered down from the KTX2 level 0 (Sun Temple's emissive levels are up to 4x darker at coarse
    // levels).
    const std::string ddsPath = deferred.semantic == MaterialTextureSemantic::Normal
        ? std::string() : ResolveCompressedSourceTexturePath(deferred.label);
    if (!ddsPath.empty()) {
        std::vector<Dds::Level> levels;
        bool srgb = false;
        std::string ddsError;
        if (DecodeCompressedDdsLevels(ddsPath, deferred.srgb, levels, srgb, ddsError)) {
            out.width = levels.front().width;
            out.height = levels.front().height;
            out.srgb = srgb;
            out.samplerDesc = sampler;
            out.mipLevels.resize(levels.size());
            for (size_t level = 0; level < levels.size(); ++level) {
                out.mipLevels[level].width = levels[level].width;
                out.mipLevels[level].height = levels[level].height;
                decodeTexels(levels[level].rgba,
                             static_cast<size_t>(levels[level].width) * static_cast<size_t>(levels[level].height),
                             srgb, out.mipLevels[level].rgba);
                out.mipLevels[level].rgba8 = std::move(levels[level].rgba);
            }
            out.rgba = out.mipLevels.front().rgba;
            return true;
        }
        std::cerr << "[SceneResources] warning: " << ddsPath << ", the DDS source of " << deferred.label << ": "
                  << ddsError << ". Metal's loader refuses it and samples the KTX2, so this CPU copy does too."
                  << std::endl;
    }
    Ktx2::ImageData source;
    std::string error;
    const bool parsed = deferred.fromFile
        ? Ktx2::ParseRgba8File(deferred.path, source, error)
        : Ktx2::ParseRgba8(deferred.data.data(), deferred.data.size(), source, error);
    if (!parsed) return false;
    return populateRgba8(source.rgba, source.width, source.height, deferred.srgb, sampler, false,
                         deferred.semantic == MaterialTextureSemantic::Orm,
                         deferred.semantic == MaterialTextureSemantic::Normal);
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

bool SceneResources::reloadEnvironmentIfNeeded(const std::string& path,
                                               EnvGpuHandles* outHandles,
                                               float sunClampLuminance) {
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
    const float sunClamp = std::isfinite(sunClampLuminance) ? std::max(sunClampLuminance, 0.0f) : 0.0f;
    const bool needsReload =
        (m_environmentTexture == nil) || (canonicalPath != m_environmentPath) ||
        (sunClamp != m_environmentSunClamp);
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

    std::vector<float> rgbaFloats;
    if (!CopyTextureToFloat(texture, rgbaFloats)) {
        NSLog(@"Failed to extract texels for environment map %@", nsPath);
        return false;
    }
    // MTKTextureLoader stores Radiance HDR as half float, so a sun brighter than
    // 65504 becomes inf and drops out of lighting. Re-decode such maps exactly.
    const bool halfOverflow = std::any_of(rgbaFloats.begin(), rgbaFloats.end(),
                                          [](float v) { return !std::isfinite(v); });
    bool reloadedExact = false;
    std::vector<float> exactRgba;
    if (halfOverflow && m_device.supports32BitFloatFiltering &&
        DecodeRadianceHdrFloat(path, static_cast<uint32_t>(width), static_cast<uint32_t>(height), exactRgba)) {
        MTLTextureDescriptor* desc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                               width:width
                                                              height:height
                                                           mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        desc.storageMode = MTLStorageModeShared;
        id<MTLTexture> exactTexture = [m_device newTextureWithDescriptor:desc];
        if (exactTexture) {
            [exactTexture replaceRegion:MTLRegionMake2D(0, 0, width, height)
                            mipmapLevel:0
                              withBytes:exactRgba.data()
                            bytesPerRow:width * 4u * sizeof(float)];
            texture = exactTexture;
            rgbaFloats.swap(exactRgba);
            reloadedExact = true;
            NSLog(@"[EnvRebuild] half-float overflow; reloaded %@ as RGBA32Float", nsPath);
        }
    }
    if (halfOverflow && !reloadedExact) {
        // e.g. flat (non-RLE) RGBE or no fp32 filtering: the overflowed texels stay dark.
        NSLog(@"[EnvRebuild] WARNING: %@ overflows half float and could not be re-decoded; "
              @"texels above 65504 contribute no light",
              nsPath);
    }
    if (sunClamp > 0.0f && m_device.supports32BitFloatFiltering) {
        size_t clampedTexels = 0;
        for (size_t i = 0; i + 3 < rgbaFloats.size(); i += 4) {
            const float luminance = 0.2126f * rgbaFloats[i] + 0.7152f * rgbaFloats[i + 1] +
                                    0.0722f * rgbaFloats[i + 2];
            if (luminance > sunClamp) {
                const float scale = sunClamp / luminance;
                rgbaFloats[i] *= scale;
                rgbaFloats[i + 1] *= scale;
                rgbaFloats[i + 2] *= scale;
                ++clampedTexels;
            }
        }
        if (clampedTexels > 0) {
            MTLTextureDescriptor* desc =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                   width:width
                                                                  height:height
                                                               mipmapped:NO];
            desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
            desc.storageMode = MTLStorageModeShared;
            id<MTLTexture> clampedTexture = [m_device newTextureWithDescriptor:desc];
            if (!clampedTexture) {
                return false;
            }
            [clampedTexture replaceRegion:MTLRegionMake2D(0, 0, width, height)
                              mipmapLevel:0
                                withBytes:rgbaFloats.data()
                              bytesPerRow:width * 4u * sizeof(float)];
            texture = clampedTexture;
            NSLog(@"[EnvRebuild] envSunClamp=%.3f scaled %zu texels of %@", sunClamp, clampedTexels, nsPath);
        }
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
    m_environmentSunClamp = sunClamp;

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

void SceneResources::setHardwareDebugSoftwareFallbackEnabled(bool enabled) {
    if (m_hardwareDebugSoftwareFallbackEnabled == enabled) {
        return;
    }

    m_hardwareDebugSoftwareFallbackEnabled = enabled;
    if (m_intersectionProvider.mode ==
        PathTracerShaderTypes::IntersectionMode::HardwareRayTracing) {
        m_dirty = true;
    }
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

void SceneResources::addReferencePointLight(const simd::float3& position,
                                           float jitterRadius,
                                           uint32_t materialIndex,
                                           bool inverseSquareFalloff,
                                           float minimumDistance) {
    if (m_rectangleCount >= kMaxRectangles || materialIndex >= m_materialCount) {
        return;
    }
    RectData light{};
    if (!inverseSquareFalloff && minimumDistance != 1.0f) {
        const float minimumDistanceSq = minimumDistance * minimumDistance;
        if (!std::isfinite(minimumDistance) || !(minimumDistance > 0.0f) ||
            !std::isnormal(minimumDistanceSq)) {
            return;
        }
        // Zero-initialized default storage remains byte-identical to the old cap.
        light.edgeU.w = minimumDistanceSq;
    }
    light.corner = simd_make_float4(position, jitterRadius);
    light.normalAndPlane = simd_make_float4(0.0f, 0.0f, 1.0f, 0.0f);
    light.materialTwoSided = simd_make_uint4(materialIndex, 1u,
        static_cast<uint32_t>(PathTracerShaderTypes::AreaPrimitiveShape::ReferencePoint),
        inverseSquareFalloff ? 1u : 0u);
    m_rectangles[m_rectangleCount++] = light;
    m_dirty = true;
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
                                         float selectionWeight,
                                         float angularRadiusDegrees) {
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
    if (angularRadiusDegrees > 0.0f && std::isfinite(angularRadiusDegrees)) {
        // Double precision preserves small solar cones before conversion to float.
        const double radians = std::min(double(angularRadiusDegrees), 90.0) * M_PI / 180.0;
        primitive.directionalCosDelta = float(1.0 - std::cos(radians));
    }
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
    m_materialTextureMinAlphaByKey.clear();
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
    m_emittedPowerByMaterial.clear();
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

bool SceneResources::adoptCPUScene(SceneResources& src, std::string* /*errorMessage*/) {
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
    m_texturePolicySkippedCount = src.m_texturePolicySkippedCount;
    m_footprintMipChainCount = src.m_footprintMipChainCount;
    m_textureBudgetExceeded = src.m_textureBudgetExceeded;
    m_textureBudgetMessage = std::move(src.m_textureBudgetMessage);
    m_emissivePrimitiveCount = src.m_emissivePrimitiveCount;
    m_totalEmittedPower = src.m_totalEmittedPower;
    m_skippedTexturedEmissivePrimitiveCount = src.m_skippedTexturedEmissivePrimitiveCount;
    m_skippedZeroPowerEmissivePrimitiveCount = src.m_skippedZeroPowerEmissivePrimitiveCount;
    m_emittedPowerByMaterial = std::move(src.m_emittedPowerByMaterial);
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
    buildInput.buildSoftwareFallbackForHardwareDebug =
        m_hardwareDebugSoftwareFallbackEnabled;
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

void SceneResources::buildCPUSceneCountsAndEmissiveInventory() {
    // The same count buildPackedSceneCpuData() ends with: every mesh's indices.size() / 3.
    uint32_t triangleCount = 0u;
    for (const auto& mesh : m_meshes) {
        triangleCount += static_cast<uint32_t>(mesh.indices.size() / 3u);
    }
    m_triangleCount = triangleCount;
    rebuildEmissivePrimitiveInventory();
}

void SceneResources::rebuildEmissivePrimitiveInventory() {
    m_emissivePrimitives.clear();
    m_emissivePrimitiveCount = 0u;
    m_totalEmittedPower = 0.0f;
    m_skippedTexturedEmissivePrimitiveCount = 0u;
    m_skippedZeroPowerEmissivePrimitiveCount = 0u;
    m_emittedPowerByMaterial.clear();
    std::vector<MaterialEmittedPower> powerByMaterial(m_materialCount);
    for (uint32_t i = 0u; i < m_materialCount; ++i) {
        powerByMaterial[i].materialIndex = i;
    }

    struct EmissionProposalTexture {
        uint32_t width = 0u;
        uint32_t height = 0u;
        MaterialTextureSamplerDesc sampler{};
        std::vector<float> luminance;
    };
    std::unordered_map<uint32_t, EmissionProposalTexture> proposalTextures;
    auto emissionProposalTexture = [&](uint32_t textureIndex) -> const EmissionProposalTexture& {
        auto [entry, inserted] = proposalTextures.try_emplace(textureIndex);
        auto& proposal = entry->second;
        if (!inserted) {
            return proposal;
        }
        MaterialTextureCpuData cpu;
        if (materialTextureCpuData(textureIndex, cpu)) {
            proposal.width = cpu.width;
            proposal.height = cpu.height;
            proposal.sampler = cpu.samplerDesc;
            proposal.luminance.resize(static_cast<size_t>(cpu.width) * cpu.height);
            for (uint32_t y = 0u; y < cpu.height; ++y) {
                // CPU readback flips resident rows; undo that for Metal UVs.
                const uint32_t sourceY = textureIndex < m_materialTextures.size()
                    ? cpu.height - 1u - y : y;
                for (uint32_t x = 0u; x < cpu.width; ++x) {
                    const size_t src = (static_cast<size_t>(sourceY) * cpu.width + x) * 4u;
                    proposal.luminance[static_cast<size_t>(y) * cpu.width + x] =
                        std::max(RelativeLuminance(simd_make_float3(cpu.rgba[src], cpu.rgba[src + 1u],
                                                                     cpu.rgba[src + 2u])), 0.0f);
                }
            }
        }
        return proposal;
    };
    auto sampleEmissionProposal = [](const EmissionProposalTexture& texture, simd::float2 uv) {
        if (texture.luminance.empty() || !std::isfinite(uv.x) || !std::isfinite(uv.y)) {
            return 1.0f;
        }
        auto wrap = [](float value, int32_t mode) {
            if (mode == kGltfWrapClampToEdge) {
                return std::clamp(value, 0.0f, 1.0f);
            }
            if (mode == kGltfWrapMirroredRepeat) {
                const float period = value - 2.0f * std::floor(value * 0.5f);
                return period <= 1.0f ? period : 2.0f - period;
            }
            return value - std::floor(value);
        };
        auto address = [](int value, uint32_t size, int32_t mode) {
            const int dimension = static_cast<int>(size);
            if (mode != kGltfWrapRepeat) {
                return std::clamp(value, 0, dimension - 1);
            }
            const int index = value % dimension;
            return index < 0 ? index + dimension : index;
        };
        const float x = wrap(uv.x, texture.sampler.wrapS) * static_cast<float>(texture.width) - 0.5f;
        const float y = wrap(uv.y, texture.sampler.wrapT) * static_cast<float>(texture.height) - 0.5f;
        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const float tx = x - static_cast<float>(x0);
        const float ty = y - static_cast<float>(y0);
        auto fetch = [&](int px, int py) {
            return texture.luminance[static_cast<size_t>(address(py, texture.height, texture.sampler.wrapT)) *
                                         texture.width + static_cast<size_t>(address(px, texture.width, texture.sampler.wrapS))];
        };
        return (fetch(x0, y0) * (1.0f - tx) + fetch(x0 + 1, y0) * tx) * (1.0f - ty) +
               (fetch(x0, y0 + 1) * (1.0f - tx) + fetch(x0 + 1, y0 + 1) * tx) * ty;
    };

    uint32_t triangleOffset = 0u;
    for (size_t meshIndex = 0u; meshIndex < m_meshes.size(); ++meshIndex) {
        const auto& mesh = m_meshes[meshIndex];
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
        const bool texturedEmissionNee =
            (material.materialFlags & PathTracerShaderTypes::kMaterialFlagTexturedEmissionNee) != 0u;
        if (hasEmissiveTexture && !texturedEmissionNee) {
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
        const EmissionProposalTexture* proposalTexture = hasEmissiveTexture
            ? &emissionProposalTexture(material.textureIndices1.x) : nullptr;

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
            float texturePowerEstimate = 1.0f;
            if (proposalTexture) {
                auto uvAtVertex = [&](uint32_t vertexIndex) {
                    const auto& vertex = mesh.vertices[vertexIndex];
                    return material.textureUvSet1.x == 1u ? vertex.uv1 : vertex.uv;
                };
                const simd::float2 uv0 = uvAtVertex(i0);
                const simd::float2 uv1 = uvAtVertex(i1);
                const simd::float2 uv2 = uvAtVertex(i2);
                auto sample = [&](simd::float2 uv) {
                    const simd::float4 row0 = material.textureTransform8;
                    const simd::float4 row1 = material.textureTransform9;
                    uv = simd_make_float2(row0.x * uv.x + row0.y * uv.y + row0.z,
                                           row1.x * uv.x + row1.y * uv.y + row1.z);
                    return sampleEmissionProposal(*proposalTexture, uv);
                };
                texturePowerEstimate = 0.25f * (sample((uv0 + uv1 + uv2) / 3.0f) +
                    sample(uv0 * (2.0f / 3.0f) + (uv1 + uv2) / 6.0f) +
                    sample(uv1 * (2.0f / 3.0f) + (uv0 + uv2) / 6.0f) +
                    sample(uv2 * (2.0f / 3.0f) + (uv0 + uv1) / 6.0f));
                // Proposal weights never replace emitted radiance. Keep
                // support for bright texels missed by the coarse quadrature.
                texturePowerEstimate = std::max(texturePowerEstimate, 1.0e-4f);
            }
            const float power = area * emissionLuminance * texturePowerEstimate;
            if (!(faceNormalLen > 1.0e-6f) ||
                !std::isfinite(faceNormalLen) ||
                !(power > (proposalTexture ? 0.0f : 1.0e-6f)) ||
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
            if (static_cast<uint32_t>(material.typeEta.x) ==
                    static_cast<uint32_t>(PathTracerShaderTypes::MaterialType::PbrMetallicRoughness) &&
                ((material.materialFlags & PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular) != 0u ||
                 texturedEmissionNee) &&
                material.typeEta.z > 0.5f) {
                primitive.flags |= PathTracerShaderTypes::kLightPrimitiveFlagTwoSided;
            }
            if (proposalTexture) {
                primitive.flags |= PathTracerShaderTypes::kLightPrimitiveFlagTexturedEmission;
                primitive.reserved1 = static_cast<uint32_t>(meshIndex);
            }
            primitive.triangleIndex = triangleOffset + tri;
            primitive.edge1 = v1 - v0;
            primitive.edge2 = v2 - v0;
            m_emissivePrimitives.push_back(primitive);
            m_totalEmittedPower += power;
            powerByMaterial[materialIndex].power += power;
            ++powerByMaterial[materialIndex].primitiveCount;
        }

        triangleOffset += triangleCount;
    }
    for (const MaterialEmittedPower& entry : powerByMaterial) {
        if (entry.primitiveCount > 0u) {
            m_emittedPowerByMaterial.push_back(entry);
        }
    }
    std::stable_sort(m_emittedPowerByMaterial.begin(), m_emittedPowerByMaterial.end(),
                     [](const MaterialEmittedPower& a, const MaterialEmittedPower& b) { return a.power > b.power; });

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
        hardwareRaytracingEnabled() &&
        !m_hardwareDebugSoftwareFallbackEnabled &&
        !KeepTriangleBufferForHardware();
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
        hardwareRaytracingEnabled() &&
        !m_hardwareDebugSoftwareFallbackEnabled &&
        !KeepTriangleBufferForHardware();

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
