#include "import/ImportPipeline.h"
#include "import/TextureConverter.h"
#include "shared/Ktx2Texture.h"

#import <CommonCrypto/CommonDigest.h>
#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if PATH_TRACER_ENABLE_ASSIMP
#include <assimp/DefaultIOSystem.h>
#include <assimp/IOStream.hpp>
#include <assimp/IOSystem.hpp>
#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <assimp/version.h>
#endif

namespace fs = std::filesystem;

namespace PathTracer {

namespace {

constexpr const char* kSemanticVersion = PATH_TRACER_IMPORT_VERSION;
constexpr const char* kGitHash = PATH_TRACER_GIT_HASH;

NSString* StringToNSString(const std::string& value) {
    return [NSString stringWithUTF8String:value.c_str()];
}

std::string ReplaceBackslashes(std::string value) {
    std::replace(value.begin(), value.end(), '\\', '/');
    return value;
}

std::string ToLowerAscii(std::string value) {
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

bool ContainsCaseInsensitive(std::string_view text, std::string_view needle) {
    if (needle.empty()) {
        return false;
    }
    std::string lowerText(text);
    std::string lowerNeedle(needle);
    lowerText = ToLowerAscii(std::move(lowerText));
    lowerNeedle = ToLowerAscii(std::move(lowerNeedle));
    return lowerText.find(lowerNeedle) != std::string::npos;
}

std::string SanitizeName(std::string value) {
    value = ReplaceBackslashes(std::move(value));
    std::string out;
    out.reserve(value.size());
    for (unsigned char ch : value) {
        if (std::isalnum(ch)) {
            out.push_back(static_cast<char>(std::tolower(ch)));
        } else if (ch == '.' || ch == '_' || ch == '-') {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('_');
        }
    }
    while (!out.empty() && out.front() == '_') {
        out.erase(out.begin());
    }
    while (!out.empty() && out.back() == '_') {
        out.pop_back();
    }
    return out.empty() ? "unnamed" : out;
}

std::string JsonEscape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (unsigned char ch : value) {
        switch (ch) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (ch < 0x20) {
                    char buffer[7];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", ch);
                    out += buffer;
                } else {
                    out.push_back(static_cast<char>(ch));
                }
                break;
        }
    }
    return out;
}

bool ReadFileBytes(const fs::path& path, std::vector<uint8_t>& outBytes, std::string& error) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        error = "Failed to open file: " + path.string();
        return false;
    }
    stream.seekg(0, std::ios::end);
    const std::streamsize size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    if (size < 0) {
        error = "Failed to read file size: " + path.string();
        return false;
    }
    outBytes.resize(static_cast<size_t>(size));
    if (size > 0) {
        stream.read(reinterpret_cast<char*>(outBytes.data()), size);
        if (!stream) {
            error = "Failed to read file: " + path.string();
            return false;
        }
    }
    return true;
}

bool WriteFileBytes(const fs::path& path, const std::vector<uint8_t>& bytes, std::string& error) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        error = "Failed to open output file: " + path.string();
        return false;
    }
    if (!bytes.empty()) {
        stream.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
    }
    if (!stream) {
        error = "Failed to write file: " + path.string();
        return false;
    }
    return true;
}

bool WriteTextFile(const fs::path& path, const std::string& text, std::string& error) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        error = "Failed to open output file: " + path.string();
        return false;
    }
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!stream) {
        error = "Failed to write file: " + path.string();
        return false;
    }
    return true;
}

std::string Sha256Hex(const std::vector<uint8_t>& bytes) {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    const uint8_t* data = bytes.empty() ? nullptr : bytes.data();
    CC_SHA256(data, static_cast<CC_LONG>(bytes.size()), digest.data());
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.resize(digest.size() * 2);
    for (size_t i = 0; i < digest.size(); ++i) {
        out[i * 2] = kHex[digest[i] >> 4];
        out[i * 2 + 1] = kHex[digest[i] & 0x0F];
    }
    return out;
}

std::string Sha256HexString(const std::string& text) {
    const uint8_t* data = text.empty() ? nullptr : reinterpret_cast<const uint8_t*>(text.data());
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(data, static_cast<CC_LONG>(text.size()), digest.data());
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.resize(digest.size() * 2);
    for (size_t i = 0; i < digest.size(); ++i) {
        out[i * 2] = kHex[digest[i] >> 4];
        out[i * 2 + 1] = kHex[digest[i] & 0x0F];
    }
    return out;
}

std::string StableImportId(const std::string& type, const std::string& source) {
    return "assimp:" + type + ":" + Sha256HexString(source).substr(0, 16);
}

bool QueryImageInfoFromData(NSData* data, int& width, int& height) {
    CGImageSourceRef source = CGImageSourceCreateWithData((__bridge CFDataRef)data, nullptr);
    if (!source) {
        return false;
    }
    CFDictionaryRef properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nullptr);
    if (!properties) {
        CFRelease(source);
        return false;
    }
    CFNumberRef widthNumber = (CFNumberRef)CFDictionaryGetValue(properties, kCGImagePropertyPixelWidth);
    CFNumberRef heightNumber = (CFNumberRef)CFDictionaryGetValue(properties, kCGImagePropertyPixelHeight);
    const bool ok = widthNumber && heightNumber &&
                    CFNumberGetValue(widthNumber, kCFNumberIntType, &width) &&
                    CFNumberGetValue(heightNumber, kCFNumberIntType, &height);
    CFRelease(properties);
    CFRelease(source);
    return ok;
}

bool QueryImageInfoFromPath(const fs::path& path, int& width, int& height) {
    NSURL* url = [NSURL fileURLWithPath:StringToNSString(path.string())];
    CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
    if (!source) {
        return false;
    }
    CFDictionaryRef properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nullptr);
    if (!properties) {
        CFRelease(source);
        return false;
    }
    CFNumberRef widthNumber = (CFNumberRef)CFDictionaryGetValue(properties, kCGImagePropertyPixelWidth);
    CFNumberRef heightNumber = (CFNumberRef)CFDictionaryGetValue(properties, kCGImagePropertyPixelHeight);
    const bool ok = widthNumber && heightNumber &&
                    CFNumberGetValue(widthNumber, kCFNumberIntType, &width) &&
                    CFNumberGetValue(heightNumber, kCFNumberIntType, &height);
    CFRelease(properties);
    CFRelease(source);
    return ok;
}

bool EncodeRawTextureToPng(const uint8_t* rgbaBytes,
                           size_t width,
                           size_t height,
                           std::vector<uint8_t>& outBytes) {
    if (!rgbaBytes || width == 0 || height == 0) {
        return false;
    }

    NSMutableData* pngData = [NSMutableData data];
    CGImageDestinationRef destination =
        CGImageDestinationCreateWithData((__bridge CFMutableDataRef)pngData, CFSTR("public.png"), 1, nullptr);
    if (!destination) {
        return false;
    }

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    if (!colorSpace) {
        CFRelease(destination);
        return false;
    }

    CGDataProviderRef provider =
        CGDataProviderCreateWithData(nullptr, rgbaBytes, width * height * 4, nullptr);
    if (!provider) {
        CGColorSpaceRelease(colorSpace);
        CFRelease(destination);
        return false;
    }

    CGImageRef image = CGImageCreate(width,
                                     height,
                                     8,
                                     32,
                                     width * 4,
                                     colorSpace,
                                     static_cast<CGBitmapInfo>(kCGBitmapByteOrderDefault) |
                                         static_cast<CGBitmapInfo>(kCGImageAlphaPremultipliedLast),
                                     provider,
                                     nullptr,
                                     false,
                                     kCGRenderingIntentDefault);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    if (!image) {
        CFRelease(destination);
        return false;
    }

    CGImageDestinationAddImage(destination, image, nullptr);
    const bool ok = CGImageDestinationFinalize(destination);
    CGImageRelease(image);
    CFRelease(destination);
    if (!ok) {
        return false;
    }

    outBytes.resize([pngData length]);
    memcpy(outBytes.data(), [pngData bytes], outBytes.size());
    return true;
}

struct BufferViewInfo {
    size_t offset = 0;
    size_t length = 0;
    int target = -1;
};

struct AccessorInfo {
    int bufferView = -1;
    size_t offset = 0;
    size_t count = 0;
    int componentType = 0;
    std::string type;
    bool normalized = false;
    bool hasMin = false;
    bool hasMax = false;
    std::vector<double> minValues;
    std::vector<double> maxValues;
};

struct PrimitiveRef {
    int positionAccessor = -1;
    int normalAccessor = -1;
    int texcoord0Accessor = -1;
    int texcoord1Accessor = -1;
    int tangentAccessor = -1;
    int indicesAccessor = -1;
    int materialIndex = -1;
};

struct MaterialInfo {
    std::string name;
    std::array<float, 4> baseColorFactor{1.0f, 1.0f, 1.0f, 1.0f};
    float metallicFactor = 0.0f;
    float roughnessFactor = 1.0f;
    std::array<float, 3> emissiveFactor{0.0f, 0.0f, 0.0f};
    bool doubleSided = false;
    std::string alphaMode = "OPAQUE";
    float alphaCutoff = 0.5f;
    int baseColorTexture = -1;
    int baseColorTexCoord = 0;
    int normalTexture = -1;
    int normalTexCoord = 0;
    int emissiveTexture = -1;
    int emissiveTexCoord = 0;
    int metallicRoughnessTexture = -1;
    int metallicRoughnessTexCoord = 0;
    int occlusionTexture = -1;
    int occlusionTexCoord = 0;
};

struct TextureInfo {
    int image = -1;
    int sampler = -1;
};

struct ImageInfo {
    std::string name;
    std::string mimeType;
    std::string uri;
    int bufferView = -1;
};

struct SamplerInfo {
    int wrapS = 10497;
    int wrapT = 10497;
    int magFilter = 9729;
    int minFilter = 9987;
};

struct MeshInfo {
    std::string name;
    std::vector<int> primitiveIndices;
};

struct CameraInfo {
    std::string name;
    float yfov = 0.7853982f;
    float aspect = 1.7777778f;
    float znear = 0.01f;
    float zfar = 1000000.0f;
    std::array<double, 3> translation{0.0, 0.0, 0.0};
    std::array<double, 4> rotation{0.0, 0.0, 0.0, 1.0};
};

struct NodeInfo {
    std::string name;
    int mesh = -1;
    int camera = -1;
    std::vector<int> children;
    std::array<double, 3> translation{0.0, 0.0, 0.0};
    std::array<double, 4> rotation{0.0, 0.0, 0.0, 1.0};
    std::array<double, 3> scale{1.0, 1.0, 1.0};
};

struct TextureInventoryEntry {
    std::string path;
    int width = 0;
    int height = 0;
    uint64_t estimatedBytes = 0;
};

struct ManifestTextureEntry {
    std::string sourcePath;
    std::string sourceFormat;
    std::string outputPath;
    std::string outputFormat;
    std::string outputSha256;
    std::string transcodeTool;
    std::string transcodeVersion;
    bool fellBackToCopy = false;
};

struct TextureAlphaInfo {
    bool hasAlpha = false;
    bool binaryAlpha = false;
};

struct CanonicalManifestEntry {
    std::string materialName;
    std::string sourceDiffuse;
    std::string sourceNormal;
    int sourceNormalChannels = 0;
    float normalAlphaMin = 0.0f;
    float normalAlphaMax = 0.0f;
    float normalAlphaStd = 0.0f;
    std::string alphaInterpretation = "absent";
    std::string sourceSpecular;
    std::string specularClassification = "absent";
    std::string specularAction = "absent";
    std::string roughnessSource = "scalar_fallback";
    std::string roughnessValueOrTarget;
    std::string metallicSource = "dielectric_default";
    bool normalYFlipped = false;
    std::string emissiveSource;
    std::vector<std::string> unresolved;
};

struct ZeroDayManifestEntry {
    std::string materialName;
    std::string sourceBaseColor;
    std::string sourceNormal;
    std::string sourceSpecular;
    std::string sourceEmissive;
    bool baseColorAlphaUsed = false;
    bool normalYFlipped = false;
    std::string specularAction = "absent";
    std::string ormTarget = "absent";
    std::vector<std::string> unresolved;
};

struct OrmSource {
    enum class Kind {
        Texture,
        Scalar,
        Fallback,
        GlossTextureInverted,
        GlossScalarInverted,
        SpecularFallback,
    };

    Kind kind = Kind::Fallback;
    std::string label;
    float scalarValue = 0.0f;
    bool varying = false;
};

struct DecodedTextureCacheEntry {
    Import::DecodedTexture decoded;
    std::string assetLabel;
};

struct OrmResolution {
    uint32_t width = 0;
    uint32_t height = 0;
};

struct ResolvedOrmChannel {
    OrmSource source;
    std::optional<DecodedTextureCacheEntry> decodedTexture;
    uint32_t channel = 0;
    bool invert = false;
    float constantValue = 0.0f;
};

struct OutputHashEntry {
    std::string path;
    std::string sha256;
};

struct SceneStats {
    uint32_t meshCount = 0;
    uint64_t triangleCount = 0;
    uint32_t nodeCount = 0;
};

// Result of the OBJ material check (see ScanObjMaterials / CheckObjMaterialAssignment).
struct ObjMaterialReport {
    uint64_t faceCount = 0;
    size_t materialCount = 0;
    size_t mismatchedMaterials = 0;
    uint64_t mtllibDirectives = 0;
    std::vector<std::string> mtllibLibraries;
    bool mtllibHoisted = false;
};

class BinaryWriter {
public:
    void align(size_t alignment = 4) {
        while (m_bytes.size() % alignment != 0) {
            m_bytes.push_back(0);
        }
    }

    template <typename T>
    size_t append(const std::vector<T>& values, size_t alignment = alignof(T)) {
        align(alignment);
        const size_t offset = m_bytes.size();
        if (!values.empty()) {
            const uint8_t* data = reinterpret_cast<const uint8_t*>(values.data());
            m_bytes.insert(m_bytes.end(), data, data + values.size() * sizeof(T));
        }
        return offset;
    }

    const std::vector<uint8_t>& bytes() const { return m_bytes; }

private:
    std::vector<uint8_t> m_bytes;
};

#if PATH_TRACER_ENABLE_ASSIMP

struct TextureAsset {
    std::string key;
    std::string debugName;
    std::string extension;
    std::string mimeType;
    std::vector<uint8_t> bytes;
    int width = 0;
    int height = 0;
    bool allowLink = false;
    fs::path sourcePath;
};

struct ImportContext {
    const aiScene* scene = nullptr;
    fs::path sourcePath;
    fs::path outputDirectory;
    std::string sceneBaseName;
    ImportOptions options;
    SceneStats stats;
    BinaryWriter buffer;
    std::vector<BufferViewInfo> bufferViews;
    std::vector<AccessorInfo> accessors;
    std::vector<PrimitiveRef> primitives;
    std::vector<MaterialInfo> materials;
    std::vector<TextureInfo> textures;
    std::vector<ImageInfo> images;
    std::vector<SamplerInfo> samplers;
    std::vector<MeshInfo> meshes;
    std::vector<CameraInfo> cameras;
    std::vector<NodeInfo> nodes;
    std::vector<int> rootNodes;
    std::vector<TextureInventoryEntry> textureInventory;
    std::vector<ManifestTextureEntry> manifestTextures;
    std::vector<TextureAlphaInfo> textureAlphaInfo;
    std::vector<CanonicalManifestEntry> canonicalManifestEntries;
    std::vector<ZeroDayManifestEntry> zeroDayManifestEntries;
    std::vector<std::string> warnings;
    std::optional<ObjMaterialReport> objMaterialReport;
    uint32_t ormTextureCount = 0;
    std::unordered_map<std::string, int> textureByKey;
    std::unordered_map<std::string, int> cameraByName;
    std::map<std::vector<unsigned int>, int> meshByPrimitiveSet;
    // FBX Model visibility, read from the file itself (ReadFbxModelVisibility): Assimp does not expose it.
    struct FbxModel {
        std::string name;
        bool ownHidden = false;
        bool inheritVisibility = true;
        int64_t parent = 0;
    };
    bool fbxVisibilityRead = false;
    std::unordered_map<int64_t, FbxModel> fbxModels;
    std::unordered_map<std::string, std::vector<int64_t>> fbxModelIdsByName;
    uint32_t fbxHiddenModelCount = 0;
    uint32_t fbxHiddenNodesSkipped = 0;
    uint64_t fbxHiddenTrianglesSkipped = 0;
    std::vector<std::string> fbxAmbiguousNodeNames;
    std::unordered_map<const aiNode*, bool> fbxNodeHidden;  // FbxNodeHidden's answer per node, computed once
    std::string fbxApplicationName;                         // FBXHeaderExtension SceneInfo Original|ApplicationName
    // Cinema 4D polygon selections (ApplyC4dPolygonSelectionRule): per node and mesh slot, the faces it keeps.
    std::unordered_map<const aiNode*, std::unordered_map<unsigned int, std::vector<uint32_t>>> c4dKeptFaces;
};

std::string OrmSourceDescription(const OrmSource& source) {
    std::ostringstream out;
    switch (source.kind) {
        case OrmSource::Kind::Texture:
            out << source.label;
            break;
        case OrmSource::Kind::GlossTextureInverted:
            out << "gloss-inverted: " << source.label;
            break;
        case OrmSource::Kind::Scalar:
            out << "scalar " << std::fixed << std::setprecision(3) << source.scalarValue;
            break;
        case OrmSource::Kind::GlossScalarInverted:
            out << "gloss-inverted-scalar " << std::fixed << std::setprecision(3) << source.scalarValue;
            break;
        case OrmSource::Kind::SpecularFallback:
            out << "specular-fallback";
            break;
        case OrmSource::Kind::Fallback:
        default:
            out << "fallback";
            break;
    }
    return out.str();
}

void LogMaterialInfo(const std::string& materialName,
                     const OrmSource& roughness,
                     const OrmSource& ao,
                     const OrmSource& metallic,
                     bool ormEmitted,
                     const std::string& ormReason) {
    std::cout << "[ImportMaterial] " << materialName
              << " roughness source: " << OrmSourceDescription(roughness)
              << " | AO source: " << OrmSourceDescription(ao)
              << " | metallic source: " << OrmSourceDescription(metallic)
              << " | ORM emitted: " << (ormEmitted ? ormReason : ("no: " + ormReason))
              << "\n";
}

void LogMaterialWarning(ImportContext& ctx,
                        const std::string& materialName,
                        const std::string& message) {
    const std::string full = "[ImportMaterial][warn] " + materialName + ": " + message;
    ctx.warnings.push_back(full);
    std::cerr << full << "\n";
}

void LogMaterialDecision(const std::string& materialName,
                         const std::string& slot,
                         const std::string& action,
                         const std::string& reason) {
    std::cout << "[material: " << materialName
              << "] [slot: " << slot
              << "] [action: " << action
              << "] [reason: " << reason
              << "]\n";
}

bool UseCanonicalImport(const ImportContext& ctx) {
    return ctx.options.importMode == ImportOptions::ImportMode::Canonical;
}

std::string GuessMimeType(const std::string& extension) {
    const std::string lower = ToLowerAscii(extension);
    if (lower == ".png") {
        return "image/png";
    }
    if (lower == ".jpg" || lower == ".jpeg") {
        return "image/jpeg";
    }
    if (lower == ".bmp") {
        return "image/bmp";
    }
    if (lower == ".tga") {
        return "image/x-tga";
    }
    if (lower == ".ktx2") {
        return "image/ktx2";
    }
    return "application/octet-stream";
}

std::string EmbeddedTextureExtension(const aiTexture* texture) {
    if (texture && texture->mHeight == 0) {
        std::string hint = ToLowerAscii(texture->achFormatHint);
        if (!hint.empty()) {
            if (hint[0] != '.') {
                hint.insert(hint.begin(), '.');
            }
            return hint;
        }
    }
    return ".png";
}

bool BuildEmbeddedTexture(const aiTexture* texture,
                          const std::string& key,
                          TextureAsset& out,
                          std::string& error) {
    if (!texture) {
        error = "Embedded texture not found: " + key;
        return false;
    }
    out.key = key;
    out.debugName = key;
    out.allowLink = false;
    if (texture->mHeight == 0) {
        out.extension = EmbeddedTextureExtension(texture);
        out.mimeType = GuessMimeType(out.extension);
        out.bytes.resize(static_cast<size_t>(texture->mWidth));
        memcpy(out.bytes.data(), texture->pcData, out.bytes.size());
        NSData* data = [NSData dataWithBytes:out.bytes.data() length:out.bytes.size()];
        QueryImageInfoFromData(data, out.width, out.height);
        return true;
    }

    std::vector<uint8_t> rgba;
    rgba.resize(static_cast<size_t>(texture->mWidth) * static_cast<size_t>(texture->mHeight) * 4);
    for (unsigned int y = 0; y < texture->mHeight; ++y) {
        for (unsigned int x = 0; x < texture->mWidth; ++x) {
            const aiTexel texel = texture->pcData[y * texture->mWidth + x];
            const size_t base = (static_cast<size_t>(y) * texture->mWidth + x) * 4;
            rgba[base + 0] = texel.r;
            rgba[base + 1] = texel.g;
            rgba[base + 2] = texel.b;
            rgba[base + 3] = texel.a;
        }
    }
    out.extension = ".png";
    out.mimeType = "image/png";
    out.width = static_cast<int>(texture->mWidth);
    out.height = static_cast<int>(texture->mHeight);
    if (!EncodeRawTextureToPng(rgba.data(), texture->mWidth, texture->mHeight, out.bytes)) {
        error = "Failed to encode embedded texture: " + key;
        return false;
    }
    return true;
}

bool BuildFileTexture(const fs::path& path, const std::string& key, TextureAsset& out, std::string& error) {
    out.key = key;
    out.debugName = path.filename().string();
    out.allowLink = true;
    out.sourcePath = path;
    out.extension = ToLowerAscii(path.extension().string());
    if (out.extension.empty()) {
        out.extension = ".bin";
    }
    out.mimeType = GuessMimeType(out.extension);
    if (!ReadFileBytes(path, out.bytes, error)) {
        return false;
    }
    QueryImageInfoFromPath(path, out.width, out.height);
    return true;
}

std::optional<TextureAsset> ResolveTextureAsset(const ImportContext& ctx,
                                                const aiString& texturePath,
                                                std::string& error) {
    const std::string rawPath = ReplaceBackslashes(texturePath.C_Str());
    if (rawPath.empty()) {
        return std::nullopt;
    }
    if (rawPath[0] == '*') {
        TextureAsset asset;
        if (!BuildEmbeddedTexture(ctx.scene->GetEmbeddedTexture(rawPath.c_str()), rawPath, asset, error)) {
            return std::nullopt;
        }
        return asset;
    }

    const fs::path sourceDir = ctx.sourcePath.parent_path();
    const fs::path sourceRoot = sourceDir.parent_path();
    std::vector<fs::path> candidates;
    auto pushCandidate = [&candidates](const fs::path& candidate) {
        if (candidate.empty()) {
            return;
        }
        if (std::find(candidates.begin(), candidates.end(), candidate) == candidates.end()) {
            candidates.push_back(candidate.lexically_normal());
        }
    };

    const fs::path rawFsPath(rawPath);
    pushCandidate(rawFsPath);
    pushCandidate(sourceDir / rawFsPath);
    pushCandidate(sourceRoot / rawFsPath);
    pushCandidate(sourceDir / rawFsPath.filename());
    pushCandidate(sourceRoot / rawFsPath.filename());

    const std::string lowerRawPath = ToLowerAscii(rawPath);
    const size_t texturesPos = lowerRawPath.rfind("textures/");
    if (texturesPos != std::string::npos) {
        pushCandidate(sourceRoot / fs::path(rawPath.substr(texturesPos)));
    }

    if (rawPath.size() > 3 &&
        std::isalpha(static_cast<unsigned char>(rawPath[0])) &&
        rawPath[1] == ':' &&
        rawPath[2] == '/') {
        const std::string withoutDrive = rawPath.substr(3);
        const fs::path driveRelative(withoutDrive);
        pushCandidate(sourceDir / driveRelative);
        pushCandidate(sourceRoot / driveRelative);
        pushCandidate(sourceRoot / driveRelative.filename());
    }
    // Last resort: archives often ship the scene next to a textures/ folder while
    // the scene still references the author's absolute path.
    pushCandidate(sourceDir / "textures" / rawFsPath.filename());

    TextureAsset asset;
    for (const fs::path& candidate : candidates) {
        std::error_code ec;
        if (!fs::exists(candidate, ec) || ec) {
            continue;
        }
        if (BuildFileTexture(candidate, rawPath, asset, error)) {
            return asset;
        }
    }

    error = "Failed to resolve texture path: " + rawPath;
    return std::nullopt;
}

size_t CountNodes(const aiNode* node) {
    if (!node) {
        return 0;
    }
    size_t count = 1;
    for (unsigned int i = 0; i < node->mNumChildren; ++i) {
        count += CountNodes(node->mChildren[i]);
    }
    return count;
}

template <typename T>
int PushAccessor(ImportContext& ctx,
                 const std::vector<T>& values,
                 size_t componentCount,
                 int componentType,
                 const std::string& type,
                 int target,
                 bool computeMinMax = false) {
    const size_t offset = ctx.buffer.append(values);
    ctx.bufferViews.push_back(BufferViewInfo{offset, values.size() * sizeof(T), target});
    AccessorInfo accessor;
    accessor.bufferView = static_cast<int>(ctx.bufferViews.size() - 1);
    accessor.count = componentCount == 0 ? 0 : values.size() / componentCount;
    accessor.componentType = componentType;
    accessor.type = type;
    if (computeMinMax && componentType == 5126 && componentCount > 0 && !values.empty()) {
        accessor.hasMin = true;
        accessor.hasMax = true;
        accessor.minValues.assign(componentCount, std::numeric_limits<double>::max());
        accessor.maxValues.assign(componentCount, std::numeric_limits<double>::lowest());
        for (size_t i = 0; i < accessor.count; ++i) {
            for (size_t c = 0; c < componentCount; ++c) {
                const double v = static_cast<double>(values[i * componentCount + c]);
                accessor.minValues[c] = std::min(accessor.minValues[c], v);
                accessor.maxValues[c] = std::max(accessor.maxValues[c], v);
            }
        }
    }
    ctx.accessors.push_back(std::move(accessor));
    return static_cast<int>(ctx.accessors.size() - 1);
}

std::array<float, 4> MakeTangent(const aiMesh* mesh, unsigned int vertexIndex) {
    const aiVector3D tangent = mesh->mTangents[vertexIndex];
    float handedness = 1.0f;
    if (mesh->HasNormals() && mesh->mBitangents) {
        const aiVector3D normal = mesh->mNormals[vertexIndex];
        const aiVector3D bitangent = mesh->mBitangents[vertexIndex];
        const aiVector3D cross = normal ^ tangent;
        handedness = (cross * bitangent) < 0.0f ? -1.0f : 1.0f;
    }
    return {tangent.x, tangent.y, tangent.z, handedness};
}

int RegisterTexture(ImportContext& ctx, const TextureAsset& asset) {
    const auto found = ctx.textureByKey.find(asset.key);
    if (found != ctx.textureByKey.end()) {
        return found->second;
    }

    const int textureIndex = static_cast<int>(ctx.textures.size());
    const int samplerIndex = static_cast<int>(ctx.samplers.size());
    ctx.samplers.push_back(SamplerInfo{});
    if (ctx.textureAlphaInfo.size() <= static_cast<size_t>(textureIndex)) {
        ctx.textureAlphaInfo.resize(static_cast<size_t>(textureIndex) + 1u);
    }

    ImageInfo image;
    image.name = asset.debugName;
    std::string inventoryPath = asset.key;
    int inventoryWidth = asset.width;
    int inventoryHeight = asset.height;
    uint64_t inventoryBytes = static_cast<uint64_t>(asset.bytes.size());

    if (ctx.options.textureMode == ImportOptions::TextureMode::Embed) {
        const size_t offset = ctx.buffer.append(asset.bytes, 1);
        ctx.bufferViews.push_back(BufferViewInfo{offset, asset.bytes.size(), -1});
        image.bufferView = static_cast<int>(ctx.bufferViews.size() - 1);
        image.mimeType = asset.mimeType;
    } else if (ctx.options.textureMode == ImportOptions::TextureMode::Convert) {
        const fs::path textureDir = ctx.outputDirectory / "textures";
        auto writeCopyFallback = [&](const std::string& reason,
                                     const std::string& sourceFormat,
                                     const std::string& transcodeTool,
                                     const std::string& transcodeVersion) {
            const std::string fileName =
                "tex_" + std::to_string(textureIndex) + "_" +
                SanitizeName(fs::path(asset.debugName).stem().string()) + asset.extension;
            const fs::path dstPath = textureDir / fileName;
            std::string fallbackError;
            if (!WriteFileBytes(dstPath, asset.bytes, fallbackError)) {
                ctx.warnings.push_back(fallbackError);
                return;
            }

            ManifestTextureEntry manifestEntry;
            manifestEntry.sourcePath = asset.sourcePath.empty()
                                           ? asset.key
                                           : ReplaceBackslashes(asset.sourcePath.string());
            manifestEntry.sourceFormat = sourceFormat;
            manifestEntry.outputPath = ReplaceBackslashes((fs::path("textures") / fileName).generic_string());
            manifestEntry.outputFormat = sourceFormat;
            manifestEntry.outputSha256 = Sha256Hex(asset.bytes);
            manifestEntry.transcodeTool = transcodeTool;
            manifestEntry.transcodeVersion = transcodeVersion;
            manifestEntry.fellBackToCopy = true;
            image.uri = manifestEntry.outputPath;
            inventoryPath = image.uri;
            inventoryWidth = asset.width;
            inventoryHeight = asset.height;
            inventoryBytes = static_cast<uint64_t>(asset.bytes.size());
            ctx.manifestTextures.push_back(std::move(manifestEntry));
            if (!reason.empty()) {
                ctx.warnings.push_back(reason);
            }
        };

        std::error_code ec;
        fs::create_directories(textureDir, ec);
        if (ec) {
            ctx.warnings.push_back("Failed to create textures directory: " + textureDir.string());
            std::string sourceFormat = ToLowerAscii(asset.extension);
            if (!sourceFormat.empty() && sourceFormat.front() == '.') {
                sourceFormat.erase(sourceFormat.begin());
            }
            writeCopyFallback("Convert mode fell back to copy for " + asset.debugName,
                              sourceFormat,
                              "pathtracer_ktx2",
                              kSemanticVersion);
        } else {
            try {
                ManifestTextureEntry manifestEntry;
                manifestEntry.sourcePath = asset.sourcePath.empty()
                                               ? asset.key
                                               : ReplaceBackslashes(asset.sourcePath.string());

                if (asset.sourcePath.empty()) {
                    std::string sourceFormat = ToLowerAscii(asset.extension);
                    if (!sourceFormat.empty() && sourceFormat.front() == '.') {
                        sourceFormat.erase(sourceFormat.begin());
                    }
                    writeCopyFallback("Embedded texture converted via copy fallback: " + asset.debugName,
                                      sourceFormat,
                                      "pathtracer_ktx2",
                                      kSemanticVersion);
                } else {
                    const Import::ConvertResult converted =
                        Import::convertTexture(asset.sourcePath.string(), textureDir.string());
                    const fs::path outputPath(converted.output_path);
                    std::error_code relEc;
                    const fs::path relativePath = fs::relative(outputPath, ctx.outputDirectory, relEc);
                    image.uri = ReplaceBackslashes(relEc ? outputPath.filename().generic_string()
                                                         : relativePath.generic_string());
                    image.mimeType = GuessMimeType("." + converted.output_format);
                    inventoryPath = image.uri;
                    inventoryWidth = static_cast<int>(converted.width);
                    inventoryHeight = static_cast<int>(converted.height);
                    inventoryBytes = converted.output_bytes;
                    ctx.textureAlphaInfo[static_cast<size_t>(textureIndex)] =
                        TextureAlphaInfo{converted.has_alpha, converted.binary_alpha};
                    manifestEntry.sourceFormat = converted.source_format;
                    manifestEntry.outputPath = image.uri;
                    manifestEntry.outputFormat = converted.output_format;
                    manifestEntry.outputSha256 = converted.output_sha256;
                    manifestEntry.transcodeTool = converted.transcode_tool;
                    manifestEntry.transcodeVersion = converted.transcode_version;
                    manifestEntry.fellBackToCopy = converted.fell_back_to_copy;
                    if (converted.fell_back_to_copy) {
                        ctx.warnings.push_back("Texture convert fallback copied " + asset.debugName +
                                               " (" + converted.source_format + ")");
                    }
                    ctx.manifestTextures.push_back(std::move(manifestEntry));
                }
            } catch (const std::exception& exception) {
                std::string sourceFormat = ToLowerAscii(asset.extension);
                if (!sourceFormat.empty() && sourceFormat.front() == '.') {
                    sourceFormat.erase(sourceFormat.begin());
                }
                writeCopyFallback(exception.what(),
                                  sourceFormat,
                                  "pathtracer_ktx2",
                                  kSemanticVersion);
            }
        }
    } else if (ctx.options.textureMode == ImportOptions::TextureMode::Link && asset.allowLink) {
        std::error_code ec;
        const fs::path relative = fs::relative(asset.sourcePath, ctx.outputDirectory, ec);
        image.uri = ec ? ReplaceBackslashes(asset.sourcePath.string())
                       : ReplaceBackslashes(relative.generic_string());
        inventoryPath = image.uri;
    } else {
        const fs::path textureDir = ctx.outputDirectory / "textures";
        std::error_code ec;
        fs::create_directories(textureDir, ec);
        const std::string fileName =
            "tex_" + std::to_string(textureIndex) + "_" +
            SanitizeName(fs::path(asset.debugName).stem().string()) + asset.extension;
        const fs::path dstPath = textureDir / fileName;
        std::string error;
        if (!WriteFileBytes(dstPath, asset.bytes, error)) {
            ctx.warnings.push_back(error);
        }
        image.uri = ReplaceBackslashes((fs::path("textures") / fileName).generic_string());
        inventoryPath = image.uri;
    }

    ctx.images.push_back(std::move(image));
    ctx.textures.push_back(TextureInfo{static_cast<int>(ctx.images.size() - 1), samplerIndex});
    ctx.textureByKey.emplace(asset.key, textureIndex);
    ctx.textureInventory.push_back(TextureInventoryEntry{
        inventoryPath,
        inventoryWidth,
        inventoryHeight,
        inventoryBytes,
    });
    return textureIndex;
}

int RegisterGeneratedTexture(ImportContext& ctx,
                             const std::string& key,
                             const std::string& debugName,
                             const std::vector<uint8_t>& rgbaBytes,
                             uint32_t width,
                             uint32_t height) {
    const auto found = ctx.textureByKey.find(key);
    if (found != ctx.textureByKey.end()) {
        return found->second;
    }

    const int textureIndex = static_cast<int>(ctx.textures.size());
    const int samplerIndex = static_cast<int>(ctx.samplers.size());
    ctx.samplers.push_back(SamplerInfo{});
    if (ctx.textureAlphaInfo.size() <= static_cast<size_t>(textureIndex)) {
        ctx.textureAlphaInfo.resize(static_cast<size_t>(textureIndex) + 1u);
    }

    const fs::path textureDir = ctx.outputDirectory / "textures";
    std::error_code ec;
    fs::create_directories(textureDir, ec);
    if (ec) {
        throw std::runtime_error("Failed to create textures directory: " + textureDir.string());
    }

    const std::string baseName =
        "orm_" + std::to_string(textureIndex) + "_" + SanitizeName(fs::path(debugName).stem().string());
    fs::path outputPath;
    uint64_t outputBytes = 0;
    if (ctx.options.textureMode == ImportOptions::TextureMode::Convert) {
        outputPath = textureDir / (baseName + ".ktx2");
        std::string error;
        if (!Ktx2::WriteRgba8File(outputPath, width, height, false, rgbaBytes, error)) {
            throw std::runtime_error(error);
        }
        std::vector<uint8_t> written;
        std::string readError;
        if (!ReadFileBytes(outputPath, written, readError)) {
            throw std::runtime_error(readError);
        }
        outputBytes = written.size();
    } else {
        outputPath = textureDir / (baseName + ".png");
        std::vector<uint8_t> pngBytes;
        if (!EncodeRawTextureToPng(rgbaBytes.data(), width, height, pngBytes)) {
            throw std::runtime_error("Failed to encode ORM PNG: " + debugName);
        }
        std::string error;
        if (!WriteFileBytes(outputPath, pngBytes, error)) {
            throw std::runtime_error(error);
        }
        outputBytes = pngBytes.size();
    }

    ImageInfo image;
    image.name = debugName;
    image.uri = ReplaceBackslashes((fs::path("textures") / outputPath.filename()).generic_string());
    image.mimeType = GuessMimeType(outputPath.extension().string());
    ctx.images.push_back(std::move(image));
    ctx.textures.push_back(TextureInfo{static_cast<int>(ctx.images.size() - 1), samplerIndex});
    ctx.textureByKey.emplace(key, textureIndex);
    ctx.textureInventory.push_back(TextureInventoryEntry{
        ReplaceBackslashes((fs::path("textures") / outputPath.filename()).generic_string()),
        static_cast<int>(width),
        static_cast<int>(height),
        outputBytes,
    });
    ctx.textureAlphaInfo[static_cast<size_t>(textureIndex)] = TextureAlphaInfo{};
    return textureIndex;
}

std::optional<TextureAsset> ResolveMaterialTextureAsset(const ImportContext& ctx,
                                                        aiMaterial* material,
                                                        aiTextureType type,
                                                        std::string& error) {
    aiString texturePath;
    if (material->GetTexture(type, 0, &texturePath) != aiReturn_SUCCESS) {
        return std::nullopt;
    }
    return ResolveTextureAsset(ctx, texturePath, error);
}

std::array<float, 4> ReadBaseColor(aiMaterial* material) {
    aiColor4D color(1.0f, 1.0f, 1.0f, 1.0f);
#ifdef AI_MATKEY_BASE_COLOR
    if (AI_SUCCESS != aiGetMaterialColor(material, AI_MATKEY_BASE_COLOR, &color)) {
        aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &color);
    }
#else
    aiGetMaterialColor(material, AI_MATKEY_COLOR_DIFFUSE, &color);
#endif
    float opacity = 1.0f;
    material->Get(AI_MATKEY_OPACITY, opacity);
    return {color.r, color.g, color.b, opacity};
}

float ReadRoughness(aiMaterial* material) {
#ifdef AI_MATKEY_ROUGHNESS_FACTOR
    float roughness = 1.0f;
    if (AI_SUCCESS == material->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness)) {
        return std::clamp(roughness, 0.0f, 1.0f);
    }
#endif
    float shininess = 0.0f;
    if (AI_SUCCESS == material->Get(AI_MATKEY_SHININESS, shininess)) {
        return std::clamp(1.0f - shininess / 256.0f, 0.04f, 1.0f);
    }
    return 1.0f;
}

float ReadMetallic(aiMaterial* material) {
#ifdef AI_MATKEY_METALLIC_FACTOR
    float metallic = 0.0f;
    if (AI_SUCCESS == material->Get(AI_MATKEY_METALLIC_FACTOR, metallic)) {
        return std::clamp(metallic, 0.0f, 1.0f);
    }
#endif
    return 0.0f;
}

std::array<float, 3> ReadEmissive(aiMaterial* material) {
    aiColor4D emissive(0.0f, 0.0f, 0.0f, 1.0f);
    aiGetMaterialColor(material, AI_MATKEY_COLOR_EMISSIVE, &emissive);
    return {emissive.r, emissive.g, emissive.b};
}

const TextureAlphaInfo& GetTextureAlphaInfo(const ImportContext& ctx, int textureIndex) {
    static const TextureAlphaInfo kDefaultInfo{};
    if (textureIndex < 0) {
        return kDefaultInfo;
    }
    const size_t index = static_cast<size_t>(textureIndex);
    if (index >= ctx.textureAlphaInfo.size()) {
        return kDefaultInfo;
    }
    return ctx.textureAlphaInfo[index];
}

std::optional<DecodedTextureCacheEntry> DecodeTextureForOrm(
    const TextureAsset& asset,
    std::unordered_map<std::string, DecodedTextureCacheEntry>& cache,
    std::string& error) {
    const std::string cacheKey = asset.sourcePath.empty() ? asset.key : asset.sourcePath.string();
    const auto found = cache.find(cacheKey);
    if (found != cache.end()) {
        return found->second;
    }
    try {
        if (asset.sourcePath.empty()) {
            error = "Embedded texture decode for ORM is unsupported: " + asset.debugName;
            return std::nullopt;
        }
        DecodedTextureCacheEntry entry;
        entry.decoded = Import::decodeTextureFile(asset.sourcePath.string());
        entry.assetLabel = asset.debugName;
        cache.emplace(cacheKey, entry);
        return entry;
    } catch (const std::exception& ex) {
        error = ex.what();
        return std::nullopt;
    }
}

float SampleLinearChannel(const Import::DecodedTexture& texture,
                          uint32_t channel,
                          uint32_t x,
                          uint32_t y) {
    const uint32_t clampedX = std::min(x, texture.width - 1u);
    const uint32_t clampedY = std::min(y, texture.height - 1u);
    const size_t index = (static_cast<size_t>(clampedY) * texture.width + clampedX) * 4u + channel;
    return static_cast<float>(texture.rgba[index]) / 255.0f;
}

float SampleLinearChannelResampled(const Import::DecodedTexture& texture,
                                   uint32_t channel,
                                   uint32_t outX,
                                   uint32_t outY,
                                   uint32_t outWidth,
                                   uint32_t outHeight) {
    if (texture.width == 0u || texture.height == 0u || outWidth == 0u || outHeight == 0u) {
        return 0.0f;
    }
    if (texture.width == outWidth && texture.height == outHeight) {
        return SampleLinearChannel(texture, channel, outX, outY);
    }
    const float srcX = ((static_cast<float>(outX) + 0.5f) / static_cast<float>(outWidth)) * texture.width - 0.5f;
    const float srcY = ((static_cast<float>(outY) + 0.5f) / static_cast<float>(outHeight)) * texture.height - 0.5f;
    const int x0 = std::clamp(static_cast<int>(std::floor(srcX)), 0, static_cast<int>(texture.width) - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(srcY)), 0, static_cast<int>(texture.height) - 1);
    const int x1 = std::clamp(x0 + 1, 0, static_cast<int>(texture.width) - 1);
    const int y1 = std::clamp(y0 + 1, 0, static_cast<int>(texture.height) - 1);
    const float tx = std::clamp(srcX - static_cast<float>(x0), 0.0f, 1.0f);
    const float ty = std::clamp(srcY - static_cast<float>(y0), 0.0f, 1.0f);
    const float c00 = SampleLinearChannel(texture, channel, static_cast<uint32_t>(x0), static_cast<uint32_t>(y0));
    const float c10 = SampleLinearChannel(texture, channel, static_cast<uint32_t>(x1), static_cast<uint32_t>(y0));
    const float c01 = SampleLinearChannel(texture, channel, static_cast<uint32_t>(x0), static_cast<uint32_t>(y1));
    const float c11 = SampleLinearChannel(texture, channel, static_cast<uint32_t>(x1), static_cast<uint32_t>(y1));
    const float c0 = c00 + (c10 - c00) * tx;
    const float c1 = c01 + (c11 - c01) * tx;
    return c0 + (c1 - c0) * ty;
}

bool MaterialHasSpecularWorkflow(aiMaterial* material) {
    aiString texturePath;
    if (material->GetTexture(aiTextureType_SPECULAR, 0, &texturePath) == aiReturn_SUCCESS) {
        return true;
    }
    if (material->GetTexture(aiTextureType_SHININESS, 0, &texturePath) == aiReturn_SUCCESS) {
        return true;
    }
#ifdef AI_MATKEY_GLOSSINESS_FACTOR
    float glossiness = 0.0f;
    if (AI_SUCCESS == material->Get(AI_MATKEY_GLOSSINESS_FACTOR, glossiness)) {
        return true;
    }
#endif
    return false;
}

bool IsBistroImport(const ImportContext& ctx) {
    const std::string lower = ToLowerAscii(ctx.sourcePath.string());
    const std::string file = ToLowerAscii(ctx.sourcePath.filename().string());
    return lower.find("/bistro/") != std::string::npos ||
           lower.find("/canonical/bistro") != std::string::npos ||
           file == "bistroexterior.fbx" ||
           file == "bistrointerior.fbx" ||
           file == "bistrointerior_wine.fbx";
}

bool IsZeroDayImport(const ImportContext& ctx) {
    const std::string lower = ToLowerAscii(ctx.sourcePath.string());
    const std::string file = ToLowerAscii(ctx.sourcePath.filename().string());
    return lower.find("/zeroday_v1/") != std::string::npos ||
           lower.find("/zero_day_v1/") != std::string::npos ||
           file == "measure_one.fbx" ||
           file == "measure_seven.fbx" ||
           file == "measure_seven_colored_lights.fbx";
}

bool IsSunTempleImport(const ImportContext& ctx) {
    const std::string lower = ToLowerAscii(ctx.sourcePath.string());
    const std::string file = ToLowerAscii(ctx.sourcePath.filename().string());
    return lower.find("/suntemple_v4/") != std::string::npos ||
           lower.find("/suntemple/") != std::string::npos ||
           file == "suntemple.fbx";
}

bool IsEmeraldSquareImport(const ImportContext& ctx) {
    const std::string lower = ToLowerAscii(ctx.sourcePath.string());
    const std::string file = ToLowerAscii(ctx.sourcePath.filename().string());
    return lower.find("/emeraldsquare_v4_1/") != std::string::npos ||
           lower.find("/emerald_square_v4_1/") != std::string::npos ||
           file == "emeraldsquare_day.fbx" ||
           file == "emeraldsquare_dusk.fbx";
}

bool UsePackedOrcaMaterialImport(const ImportContext& ctx) {
    return IsZeroDayImport(ctx) ||
           IsEmeraldSquareImport(ctx) ||
           (IsSunTempleImport(ctx) && UseCanonicalImport(ctx));
}


const char* PackedOrcaImportLabel(const ImportContext& ctx) {
    if (IsSunTempleImport(ctx)) {
        return "SunTemple";
    }
    if (IsEmeraldSquareImport(ctx)) {
        return "EmeraldSquare";
    }
    return "ZeroDay";
}

bool IsEmeraldSquareFoliageMaterial(const ImportContext& ctx, std::string_view materialName) {
    if (!IsEmeraldSquareImport(ctx)) {
        return false;
    }
    const std::string lower = ToLowerAscii(std::string(materialName));
    return lower.find("leaf") != std::string::npos ||
           lower.find("leaves") != std::string::npos ||
           lower.find("foliage") != std::string::npos ||
           lower.find("fern_fiddlehead") != std::string::npos ||
           lower.find("fern_leaflets") != std::string::npos;
}

bool IsEmeraldSquareWorldGridMaterial(const ImportContext& ctx, std::string_view materialName) {
    if (!IsEmeraldSquareImport(ctx)) {
        return false;
    }
    const std::string lower = ToLowerAscii(std::string(materialName));
    return lower.rfind("worldgridmaterial", 0) == 0;
}

namespace canonical {

enum class AssetFamily {
    Generic,
    CryEngine,
};

struct ChannelStats {
    float minValue = 1.0f;
    float maxValue = 0.0f;
    float mean = 0.0f;
    float stddev = 0.0f;
};

struct LoadedTexture {
    TextureAsset asset;
    DecodedTextureCacheEntry decoded;
    int sourceChannels = 0;
    ChannelStats alphaStats;
};

struct SourceMaterialBundle {
    std::string materialName;
    AssetFamily assetFamily = AssetFamily::Generic;
    std::optional<LoadedTexture> diffuse;
    std::optional<LoadedTexture> normal;
    std::optional<LoadedTexture> specular;
    std::optional<LoadedTexture> emissive;
    float shininessScalar = 0.0f;
    std::vector<std::string> unresolved;
};

uint32_t ReadU32Le(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8u) |
           (static_cast<uint32_t>(data[2]) << 16u) |
           (static_cast<uint32_t>(data[3]) << 24u);
}

int InferSourceChannelCount(const fs::path& path) {
    const std::string extension = ToLowerAscii(path.extension().string());
    if (extension != ".dds") {
        return 4;
    }
    std::vector<uint8_t> bytes;
    std::string error;
    if (!ReadFileBytes(path, bytes, error) || bytes.size() < 128u || std::memcmp(bytes.data(), "DDS ", 4) != 0) {
        return 4;
    }
    const std::string fourcc(reinterpret_cast<const char*>(bytes.data() + 84), 4);
    if (fourcc == "DXT1") {
        return 3;
    }
    if (fourcc == "DXT5") {
        return 4;
    }
    if (fourcc == "ATI2") {
        return 2;
    }
    if (fourcc == "DX10" && bytes.size() >= 148u) {
        const uint32_t dxgiFormat = ReadU32Le(bytes.data() + 128);
        switch (dxgiFormat) {
            case 71u:
            case 72u:
                return 3;
            case 77u:
            case 78u:
                return 4;
            case 83u:
            case 84u:
                return 2;
            default:
                break;
        }
    }
    return 4;
}

ChannelStats ComputeAlphaStats(const Import::DecodedTexture& decoded) {
    ChannelStats stats;
    if (decoded.rgba.empty()) {
        stats.minValue = 0.0f;
        return stats;
    }
    const size_t pixelCount = static_cast<size_t>(decoded.width) * decoded.height;
    double sum = 0.0;
    double sumSquares = 0.0;
    for (size_t i = 3; i < decoded.rgba.size(); i += 4u) {
        const float value = static_cast<float>(decoded.rgba[i]) / 255.0f;
        stats.minValue = std::min(stats.minValue, value);
        stats.maxValue = std::max(stats.maxValue, value);
        sum += value;
        sumSquares += static_cast<double>(value) * value;
    }
    if (pixelCount > 0) {
        stats.mean = static_cast<float>(sum / static_cast<double>(pixelCount));
        const double meanSquare = sumSquares / static_cast<double>(pixelCount);
        stats.stddev = static_cast<float>(std::sqrt(std::max(0.0, meanSquare - stats.mean * stats.mean)));
    }
    return stats;
}

std::string StatsString(const ChannelStats& stats) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(4)
        << "min=" << stats.minValue
        << " max=" << stats.maxValue
        << " std=" << stats.stddev;
    return out.str();
}

std::vector<std::string> MaterialNameCandidates(std::string materialName) {
    std::vector<std::string> candidates;
    auto push = [&](std::string value) {
        if (value.empty()) {
            return;
        }
        if (std::find(candidates.begin(), candidates.end(), value) == candidates.end()) {
            candidates.push_back(std::move(value));
        }
    };

    push(materialName);
    const std::string doubleSidedSuffix = ".DoubleSided";
    if (materialName.size() > doubleSidedSuffix.size() &&
        materialName.rfind(doubleSidedSuffix) == materialName.size() - doubleSidedSuffix.size()) {
        push(materialName.substr(0, materialName.size() - doubleSidedSuffix.size()));
    }
    return candidates;
}

std::optional<TextureAsset> ResolveSourceTextureByConvention(const ImportContext& ctx,
                                                             const std::string& materialName,
                                                             const char* suffix,
                                                             std::string& error) {
    const fs::path textureRoot = ctx.sourcePath.parent_path().parent_path() / "Textures";
    const std::array<std::string, 10> extensions{
        ".dds", ".DDS", ".tga", ".TGA", ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"};
    for (const std::string& candidateName : MaterialNameCandidates(materialName)) {
        for (const std::string& extension : extensions) {
            const fs::path candidate = textureRoot / (candidateName + suffix + extension);
            std::error_code ec;
            if (!fs::exists(candidate, ec) || ec) {
                continue;
            }
            TextureAsset asset;
            if (BuildFileTexture(candidate, candidate.generic_string(), asset, error)) {
                return asset;
            }
        }
    }
    error = "No source texture matched " + materialName + suffix;
    return std::nullopt;
}

std::optional<LoadedTexture> LoadTexture(const TextureAsset& asset, std::string& error) {
    try {
        LoadedTexture loaded;
        loaded.asset = asset;
        loaded.decoded.decoded = Import::decodeTextureFile(asset.sourcePath.string());
        loaded.decoded.assetLabel = asset.debugName;
        loaded.sourceChannels = InferSourceChannelCount(asset.sourcePath);
        loaded.alphaStats = ComputeAlphaStats(loaded.decoded.decoded);
        return loaded;
    } catch (const std::exception& ex) {
        error = ex.what();
        return std::nullopt;
    }
}

std::optional<LoadedTexture> LoadSourceTexture(ImportContext& ctx,
                                               aiMaterial* material,
                                               const std::string& materialName,
                                               aiTextureType fallbackType,
                                               const char* suffix,
                                               std::string& error) {
    if (const auto asset = ResolveSourceTextureByConvention(ctx, materialName, suffix, error)) {
        return LoadTexture(*asset, error);
    }
    if (const auto asset = ResolveMaterialTextureAsset(ctx, material, fallbackType, error)) {
        return LoadTexture(*asset, error);
    }
    return std::nullopt;
}

int GetTextureUvIndex(aiMaterial* material, aiTextureType type) {
    if (!material) {
        return 0;
    }
    aiString texturePath;
    unsigned int uvIndex = 0u;
    if (material->GetTexture(type, 0, &texturePath, nullptr, &uvIndex) == aiReturn_SUCCESS) {
        return static_cast<int>(uvIndex);
    }
    return 0;
}

bool IsBistroSourceAsset(const ImportContext& ctx) {
    const std::string lower = ToLowerAscii(ctx.sourcePath.string());
    return lower.find("bistro") != std::string::npos;
}

std::string ClassifySpecular(const LoadedTexture& specular) {
    const auto& rgba = specular.decoded.decoded.rgba;
    if (rgba.empty()) {
        return "constant";
    }
    std::array<float, 4> mean{0, 0, 0, 0};
    std::array<float, 4> stddev{0, 0, 0, 0};
    const size_t pixelCount = static_cast<size_t>(specular.decoded.decoded.width) * specular.decoded.decoded.height;
    for (size_t channel = 0; channel < 4; ++channel) {
        double sum = 0.0;
        double sumSquares = 0.0;
        for (size_t i = channel; i < rgba.size(); i += 4u) {
            const float v = static_cast<float>(rgba[i]) / 255.0f;
            sum += v;
            sumSquares += static_cast<double>(v) * v;
        }
        mean[channel] = static_cast<float>(sum / static_cast<double>(pixelCount));
        const double meanSquare = sumSquares / static_cast<double>(pixelCount);
        stddev[channel] = static_cast<float>(std::sqrt(std::max(0.0, meanSquare - mean[channel] * mean[channel])));
    }
    const float rgbSpread = std::max({mean[0], mean[1], mean[2]}) - std::min({mean[0], mean[1], mean[2]});
    const float rgbStdMax = std::max({stddev[0], stddev[1], stddev[2]});
    if (rgbStdMax < 0.02f && stddev[3] < 0.02f) {
        return "constant";
    }
    if (stddev[3] > 0.08f) {
        return "gloss_channel_A";
    }
    if (rgbSpread > 0.08f || std::abs(stddev[0] - stddev[1]) > 0.08f || std::abs(stddev[1] - stddev[2]) > 0.08f) {
        return "reflectance_color";
    }
    return "monochrome_intensity";
}

SourceMaterialBundle BuildSourceMaterialBundle(ImportContext& ctx,
                                               aiMaterial* material,
                                               const std::string& materialName) {
    SourceMaterialBundle bundle;
    bundle.materialName = materialName;
    bundle.assetFamily = AssetFamily::CryEngine;
    material->Get(AI_MATKEY_SHININESS, bundle.shininessScalar);

    std::string error;
    if (const auto loaded = LoadSourceTexture(ctx, material, materialName, aiTextureType_DIFFUSE, "_BaseColor", error)) {
        bundle.diffuse = loaded;
    } else if (!error.empty()) {
        bundle.unresolved.push_back("diffuse: " + error);
    }

    error.clear();
    if (const auto loaded = LoadSourceTexture(ctx, material, materialName, aiTextureType_NORMALS, "_Normal", error)) {
        bundle.normal = loaded;
        std::cout << "[canonical] [material: " << materialName << "] normal loaded: "
                  << loaded->decoded.decoded.width << "x" << loaded->decoded.decoded.height
                  << " channels=" << loaded->sourceChannels << "\n";
    } else if (!error.empty()) {
        bundle.unresolved.push_back("normal: " + error);
    }

    error.clear();
    if (const auto loaded = LoadSourceTexture(ctx, material, materialName, aiTextureType_SPECULAR, "_Specular", error)) {
        bundle.specular = loaded;
    } else if (!error.empty()) {
        bundle.unresolved.push_back("specular: " + error);
    }

    error.clear();
    if (const auto loaded = LoadSourceTexture(ctx, material, materialName, aiTextureType_EMISSIVE, "_Emissive", error)) {
        bundle.emissive = loaded;
    } else if (!error.empty()) {
        bundle.unresolved.push_back("emissive: " + error);
    }

    return bundle;
}

}  // namespace canonical

std::vector<std::string> ZeroDayMaterialNameCandidates(const std::string& materialName) {
    std::vector<std::string> candidates = canonical::MaterialNameCandidates(materialName);
    auto push = [&](std::string value) {
        if (value.empty()) {
            return;
        }
        if (std::find(candidates.begin(), candidates.end(), value) == candidates.end()) {
            candidates.push_back(std::move(value));
        }
    };

    const size_t c4dPos = materialName.find("_c4d");
    if (c4dPos != std::string::npos) {
        push(materialName.substr(0, c4dPos));
    }
    const size_t slashPos = materialName.find('/');
    if (slashPos != std::string::npos) {
        push(materialName.substr(0, slashPos));
    }
    return candidates;
}

std::optional<TextureAsset> ResolveZeroDayTextureByConvention(const ImportContext& ctx,
                                                              const std::string& materialName,
                                                              const char* suffix,
                                                              std::string& error) {
    const fs::path textureRoot = ctx.sourcePath.parent_path() /
                                 ((IsSunTempleImport(ctx) || IsEmeraldSquareImport(ctx)) ? "Textures" : "tex");
    const std::array<std::string, 10> extensions{
        ".dds", ".DDS", ".tga", ".TGA", ".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"};
    for (const std::string& candidateName : ZeroDayMaterialNameCandidates(materialName)) {
        for (const std::string& extension : extensions) {
            const fs::path candidate = textureRoot / (candidateName + suffix + extension);
            std::error_code ec;
            if (!fs::exists(candidate, ec) || ec) {
                continue;
            }
            TextureAsset asset;
            if (BuildFileTexture(candidate, candidate.generic_string(), asset, error)) {
                return asset;
            }
        }
    }
    error = std::string("No ") + PackedOrcaImportLabel(ctx) + " source texture matched " +
            materialName + suffix;
    return std::nullopt;
}

// The texture the FBX material connects to the slot wins: Falcor, which the
// packed ORCA scenes target, binds exactly that file. Some connections name a
// file other than <material>_<Slot> (Zero Day's "side - white three" emits
// through "..._Emissive2.dds", the colored-lights floor clickers through
// "new_light_orange.dds"), so the file-name convention is only the fallback for
// materials whose FBX connects no texture to the slot. `fromConnection` (when
// given) reports whether the returned texture is the FBX-connected one.
// A slot with neither a connection nor a convention file has nothing to bind:
// it returns nullopt with an empty `error`, which callers do not report.
std::optional<canonical::LoadedTexture> LoadZeroDayTexture(ImportContext& ctx,
                                                           aiMaterial* material,
                                                           const std::string& materialName,
                                                           std::initializer_list<aiTextureType> connectedTypes,
                                                           const char* suffix,
                                                           std::string& error,
                                                           bool* fromConnection = nullptr) {
    const std::string slot = std::string(suffix + (suffix[0] == '_' ? 1 : 0));
    if (fromConnection) {
        *fromConnection = false;
    }
    std::string connectionError;
    for (aiTextureType type : connectedTypes) {
        aiString connectedPath;
        if (material->GetTexture(type, 0, &connectedPath) != aiReturn_SUCCESS) {
            continue;
        }
        std::string resolveError;
        if (const auto asset = ResolveMaterialTextureAsset(ctx, material, type, resolveError)) {
            error.clear();
            if (auto loaded = canonical::LoadTexture(*asset, error)) {
                LogMaterialDecision(materialName,
                                    slot,
                                    "fbx_connection",
                                    std::string(PackedOrcaImportLabel(ctx)) + " texture the FBX connects: " +
                                        ReplaceBackslashes(connectedPath.C_Str()));
                if (fromConnection) {
                    *fromConnection = true;
                }
                return loaded;
            }
            resolveError = error;
        }
        connectionError = "FBX connection " + ReplaceBackslashes(connectedPath.C_Str()) +
                          " did not load" + (resolveError.empty() ? std::string() : ": " + resolveError);
        LogMaterialWarning(ctx, materialName, std::string(PackedOrcaImportLabel(ctx)) + " " + slot + " " +
                                                  connectionError + "; trying the next source");
    }
    std::string conventionError;
    const auto conventionAsset = ResolveZeroDayTextureByConvention(ctx, materialName, suffix, conventionError);
    if (conventionAsset) {
        error.clear();
        if (auto loaded = canonical::LoadTexture(*conventionAsset, error)) {
            LogMaterialDecision(materialName,
                                slot,
                                connectionError.empty() ? "file_name_convention" : "file_name_convention_after_failed_connection",
                                std::string(PackedOrcaImportLabel(ctx)) +
                                    (connectionError.empty() ? " FBX connects no texture; " : " " + connectionError + "; ") +
                                    "using " + ReplaceBackslashes(conventionAsset->sourcePath.string()));
            return loaded;
        }
        conventionError = error;
    } else if (connectionError.empty()) {
        // No connection and no convention file: nothing to bind, nothing to report.
        error.clear();
        return std::nullopt;
    }
    error = connectionError;
    if (!conventionError.empty()) {
        error += (error.empty() ? std::string() : "; ") + conventionError;
    }
    return std::nullopt;
}

OrmResolution ChooseOrmResolution(const std::vector<const Import::DecodedTexture*>& varyingTextures,
                                  uint32_t baseColorWidth,
                                  uint32_t baseColorHeight) {
    OrmResolution resolution;
    for (const Import::DecodedTexture* texture : varyingTextures) {
        if (!texture) {
            continue;
        }
        if (texture->width * texture->height > resolution.width * resolution.height) {
            resolution.width = texture->width;
            resolution.height = texture->height;
        }
    }
    if (baseColorWidth > 0u && baseColorHeight > 0u && resolution.width > 0u && resolution.height > 0u) {
        resolution.width = std::min(resolution.width, baseColorWidth);
        resolution.height = std::min(resolution.height, baseColorHeight);
    }
    return resolution;
}

float EvaluateOrmChannel(const ResolvedOrmChannel& channel,
                         uint32_t x,
                         uint32_t y,
                         uint32_t width,
                         uint32_t height) {
    float value = channel.constantValue;
    if (channel.decodedTexture && !channel.decodedTexture->decoded.rgba.empty()) {
        value = SampleLinearChannelResampled(channel.decodedTexture->decoded,
                                             channel.channel,
                                             x,
                                             y,
                                             width,
                                             height);
    }
    if (channel.invert) {
        value = 1.0f - value;
    }
    return std::clamp(value, 0.0f, 1.0f);
}

void ConvertGenericMaterials(ImportContext& ctx) {
    ctx.materials.reserve(ctx.scene->mNumMaterials);
    for (unsigned int i = 0; i < ctx.scene->mNumMaterials; ++i) {
        aiMaterial* material = ctx.scene->mMaterials[i];
        MaterialInfo info;
        std::unordered_map<int, std::optional<TextureAsset>> assetCache;
        std::unordered_map<std::string, DecodedTextureCacheEntry> decodeCache;
        aiString name;
        if (AI_SUCCESS == material->Get(AI_MATKEY_NAME, name)) {
            info.name = name.C_Str();
        } else {
            info.name = "material_" + std::to_string(i);
        }
        info.baseColorFactor = ReadBaseColor(material);
        info.roughnessFactor = 0.5f;
        info.metallicFactor = 0.0f;
        info.emissiveFactor = ReadEmissive(material);
        int twoSided = 0;
        material->Get(AI_MATKEY_TWOSIDED, twoSided);
        info.doubleSided = (twoSided != 0);
        if (ContainsCaseInsensitive(info.name, ".doublesided") ||
            ContainsCaseInsensitive(info.name, "_doublesided")) {
            info.doubleSided = true;
        }
        if (info.baseColorFactor[3] < 0.999f) {
            info.alphaMode = "BLEND";
        }
        const bool bistroPolicy = canonical::IsBistroSourceAsset(ctx);

        auto fetchAsset = [&](aiTextureType type) -> std::optional<TextureAsset> {
            const auto found = assetCache.find(static_cast<int>(type));
            if (found != assetCache.end()) {
                return found->second;
            }

            std::string error;
            auto asset = ResolveMaterialTextureAsset(ctx, material, type, error);
            if (!asset && !error.empty()) {
                LogMaterialWarning(ctx, info.name, error);
            }
            assetCache.emplace(static_cast<int>(type), asset);
            return asset;
        };

        auto registerResolvedAsset = [&](aiTextureType type) -> std::optional<int> {
            const auto asset = fetchAsset(type);
            if (!asset) {
                return std::nullopt;
            }
            return RegisterTexture(ctx, *asset);
        };

        auto decodeResolvedAsset = [&](aiTextureType type,
                                       const std::string& failureMessage) -> std::optional<DecodedTextureCacheEntry> {
            const auto asset = fetchAsset(type);
            if (!asset) {
                return std::nullopt;
            }
            std::string error;
            const auto decoded = DecodeTextureForOrm(*asset, decodeCache, error);
            if (!decoded && !error.empty()) {
                LogMaterialWarning(ctx, info.name, failureMessage + ": " + error);
            }
            return decoded;
        };

        std::optional<TextureAsset> baseColorAsset = fetchAsset(aiTextureType_DIFFUSE);
        if (baseColorAsset) {
            const int tex = RegisterTexture(ctx, *baseColorAsset);
            info.baseColorTexture = tex;
            info.baseColorTexCoord = canonical::GetTextureUvIndex(material, aiTextureType_DIFFUSE);
            const TextureAlphaInfo& alphaInfo = GetTextureAlphaInfo(ctx, tex);
            if (alphaInfo.hasAlpha) {
                info.alphaMode = alphaInfo.binaryAlpha ? "MASK" : "BLEND";
                info.alphaCutoff = alphaInfo.binaryAlpha ? 0.5f : info.alphaCutoff;
            }
        }

        if (const auto tex = registerResolvedAsset(aiTextureType_NORMAL_CAMERA)) {
            info.normalTexture = *tex;
            info.normalTexCoord = canonical::GetTextureUvIndex(material, aiTextureType_NORMAL_CAMERA);
        } else if (const auto tex = registerResolvedAsset(aiTextureType_NORMALS)) {
            info.normalTexture = *tex;
            info.normalTexCoord = canonical::GetTextureUvIndex(material, aiTextureType_NORMALS);
        }
        if (const auto tex = registerResolvedAsset(aiTextureType_EMISSIVE)) {
            info.emissiveTexture = *tex;
            info.emissiveTexCoord = canonical::GetTextureUvIndex(material, aiTextureType_EMISSIVE);
        }

        uint32_t baseColorWidth = 0u;
        uint32_t baseColorHeight = 0u;
        if (baseColorAsset) {
            baseColorWidth = std::max(0, baseColorAsset->width);
            baseColorHeight = std::max(0, baseColorAsset->height);
            if ((baseColorWidth == 0u || baseColorHeight == 0u) && baseColorAsset->sourcePath.has_filename()) {
                if (const auto decoded = decodeResolvedAsset(aiTextureType_DIFFUSE, "baseColor decode failed")) {
                    baseColorWidth = decoded->decoded.width;
                    baseColorHeight = decoded->decoded.height;
                }
            }
        }

        ResolvedOrmChannel roughness{};
        ResolvedOrmChannel ao{};
        ResolvedOrmChannel metallic{};

        const auto specularAsset = fetchAsset(aiTextureType_SPECULAR);
        if (bistroPolicy && specularAsset) {
            LogMaterialDecision(info.name,
                                "SPECULAR",
                                "discarded",
                                "Bistro specular texture is not mapped into glTF roughness/AO; discarding " +
                                    specularAsset->debugName);
        }

        if (const auto decoded = decodeResolvedAsset(aiTextureType_DIFFUSE_ROUGHNESS, "roughness texture decode failed")) {
            roughness.source.kind = OrmSource::Kind::Texture;
            roughness.source.label = decoded->assetLabel;
            roughness.source.varying = true;
            roughness.decodedTexture = decoded;
            roughness.channel = 0u;
        } else if (const auto decoded = decodeResolvedAsset(aiTextureType_SHININESS, "glossiness texture decode failed")) {
            roughness.source.kind = OrmSource::Kind::GlossTextureInverted;
            roughness.source.label = decoded->assetLabel;
            roughness.source.varying = true;
            roughness.decodedTexture = decoded;
            roughness.channel = 0u;
            roughness.invert = true;
        } else {
            if (!bistroPolicy) {
                if (const auto decoded = decodeResolvedAsset(aiTextureType_SPECULAR, "specular-workflow glossiness texture decode failed")) {
                    roughness.source.kind = OrmSource::Kind::GlossTextureInverted;
                    roughness.source.label = decoded->assetLabel;
                    roughness.source.varying = true;
                    roughness.decodedTexture = decoded;
                    roughness.channel = 0u;
                    roughness.invert = true;
                }
            }
            if (roughness.source.kind == OrmSource::Kind::GlossTextureInverted) {
                // already resolved above
            } else {
            float scalar = 0.0f;
            if (AI_SUCCESS == material->Get(AI_MATKEY_ROUGHNESS_FACTOR, scalar)) {
                roughness.source.kind = OrmSource::Kind::Scalar;
                roughness.source.scalarValue = std::clamp(scalar, 0.0f, 1.0f);
                roughness.constantValue = roughness.source.scalarValue;
                if (bistroPolicy) {
                    std::ostringstream reason;
                    reason << "no varying Bistro roughness texture confirmed; using scalar roughness "
                           << std::fixed << std::setprecision(3) << roughness.constantValue;
                    LogMaterialDecision(info.name, "ROUGHNESS_FACTOR", "scalar_fallback", reason.str());
                }
            } else if (AI_SUCCESS == material->Get(AI_MATKEY_GLOSSINESS_FACTOR, scalar)) {
                const float gloss = std::clamp(scalar, 0.0f, 1.0f);
                roughness.source.kind = OrmSource::Kind::GlossScalarInverted;
                roughness.source.scalarValue = gloss;
                roughness.constantValue = 1.0f - gloss;
                if (bistroPolicy) {
                    std::ostringstream reason;
                    reason << "no varying Bistro roughness texture confirmed; using inverted glossiness scalar "
                           << std::fixed << std::setprecision(3) << gloss
                           << " -> roughness " << roughness.constantValue;
                    LogMaterialDecision(info.name, "GLOSSINESS_FACTOR", "scalar_fallback", reason.str());
                }
            } else {
                roughness.source.kind = OrmSource::Kind::Fallback;
                roughness.constantValue = 0.5f;
                LogMaterialWarning(ctx, info.name, "no roughness source found");
            }
            }
        }

        if (const auto decoded = decodeResolvedAsset(aiTextureType_AMBIENT_OCCLUSION, "occlusion texture decode failed")) {
            ao.source.kind = OrmSource::Kind::Texture;
            ao.source.label = decoded->assetLabel;
            ao.source.varying = true;
            ao.decodedTexture = decoded;
            ao.channel = 0u;
        } else {
            ao.source.kind = OrmSource::Kind::Fallback;
            ao.constantValue = 1.0f;
        }

        if (const auto decoded = decodeResolvedAsset(aiTextureType_METALNESS, "metallic texture decode failed")) {
            metallic.source.kind = OrmSource::Kind::Texture;
            metallic.source.label = decoded->assetLabel;
            metallic.source.varying = true;
            metallic.decodedTexture = decoded;
            metallic.channel = 0u;
        } else {
            float scalar = 0.0f;
            if (AI_SUCCESS == material->Get(AI_MATKEY_METALLIC_FACTOR, scalar)) {
                metallic.source.kind = OrmSource::Kind::Scalar;
                metallic.source.scalarValue = std::clamp(scalar, 0.0f, 1.0f);
                metallic.constantValue = metallic.source.scalarValue;
            } else if (!bistroPolicy && MaterialHasSpecularWorkflow(material)) {
                metallic.source.kind = OrmSource::Kind::SpecularFallback;
                metallic.constantValue = 0.0f;
                LogMaterialWarning(ctx, info.name, "specular workflow reduced to dielectric fallback");
            } else {
                metallic.source.kind = OrmSource::Kind::Fallback;
                metallic.constantValue = 0.0f;
            }
        }

        std::vector<const Import::DecodedTexture*> varyingTextures;
        if (roughness.source.varying && roughness.decodedTexture) {
            varyingTextures.push_back(&roughness.decodedTexture->decoded);
        }
        if (ao.source.varying && ao.decodedTexture) {
            varyingTextures.push_back(&ao.decodedTexture->decoded);
        }
        if (metallic.source.varying && metallic.decodedTexture) {
            varyingTextures.push_back(&metallic.decodedTexture->decoded);
        }

        bool ormEmitted = false;
        std::string ormReason = "ORM omitted: no varying source data";
        if (!varyingTextures.empty()) {
            const OrmResolution resolution = ChooseOrmResolution(varyingTextures, baseColorWidth, baseColorHeight);
            if (resolution.width > 0u && resolution.height > 0u) {
                std::vector<uint8_t> packed(static_cast<size_t>(resolution.width) * resolution.height * 4u, 255u);
                for (uint32_t y = 0; y < resolution.height; ++y) {
                    for (uint32_t x = 0; x < resolution.width; ++x) {
                        const size_t base = (static_cast<size_t>(y) * resolution.width + x) * 4u;
                        packed[base + 0] = static_cast<uint8_t>(std::round(EvaluateOrmChannel(ao, x, y, resolution.width, resolution.height) * 255.0f));
                        packed[base + 1] = static_cast<uint8_t>(std::round(EvaluateOrmChannel(roughness, x, y, resolution.width, resolution.height) * 255.0f));
                        packed[base + 2] = static_cast<uint8_t>(std::round(EvaluateOrmChannel(metallic, x, y, resolution.width, resolution.height) * 255.0f));
                    }
                }
                const int ormTexture = RegisterGeneratedTexture(ctx,
                                                                "orm:" + info.name,
                                                                info.name + "_orm",
                                                                packed,
                                                                resolution.width,
                                                                resolution.height);
                info.metallicRoughnessTexture = ormTexture;
                info.occlusionTexture = ormTexture;
                info.metallicRoughnessTexCoord = canonical::GetTextureUvIndex(material, aiTextureType_DIFFUSE_ROUGHNESS);
                info.occlusionTexCoord = canonical::GetTextureUvIndex(material, aiTextureType_AMBIENT_OCCLUSION);
                info.roughnessFactor = 1.0f;
                info.metallicFactor = 1.0f;
                ormEmitted = true;
                ++ctx.ormTextureCount;
                std::ostringstream ormInfo;
                ormInfo << "yes, " << resolution.width << "x" << resolution.height;
                ormReason = ormInfo.str();
            }
        } else {
            info.roughnessFactor = std::clamp(roughness.constantValue, 0.0f, 1.0f);
            info.metallicFactor = std::clamp(metallic.constantValue, 0.0f, 1.0f);
        }

        LogMaterialInfo(info.name, roughness.source, ao.source, metallic.source, ormEmitted, ormReason);
        ctx.materials.push_back(std::move(info));
    }
    if (ctx.materials.empty()) {
        ctx.materials.push_back(MaterialInfo{});
    }
}

void ConvertCanonicalMaterials(ImportContext& ctx) {
    ctx.materials.reserve(ctx.scene->mNumMaterials);
    ctx.canonicalManifestEntries.reserve(ctx.scene->mNumMaterials);
    for (unsigned int i = 0; i < ctx.scene->mNumMaterials; ++i) {
        aiMaterial* material = ctx.scene->mMaterials[i];
        MaterialInfo info;
        aiString name;
        if (AI_SUCCESS == material->Get(AI_MATKEY_NAME, name)) {
            info.name = name.C_Str();
        } else {
            info.name = "material_" + std::to_string(i);
        }

        info.baseColorFactor = ReadBaseColor(material);
        info.roughnessFactor = 0.5f;
        info.metallicFactor = 0.0f;
        info.emissiveFactor = ReadEmissive(material);
        int twoSided = 0;
        material->Get(AI_MATKEY_TWOSIDED, twoSided);
        info.doubleSided = (twoSided != 0);
        if (ContainsCaseInsensitive(info.name, ".doublesided") ||
            ContainsCaseInsensitive(info.name, "_doublesided")) {
            info.doubleSided = true;
        }

        canonical::SourceMaterialBundle bundle = canonical::BuildSourceMaterialBundle(ctx, material, info.name);
        CanonicalManifestEntry manifest;
        manifest.materialName = info.name;

        auto pathString = [](const std::optional<canonical::LoadedTexture>& texture) -> std::string {
            if (!texture) {
                return "";
            }
            return ReplaceBackslashes(texture->asset.sourcePath.string());
        };

        manifest.sourceDiffuse = pathString(bundle.diffuse);
        manifest.sourceNormal = pathString(bundle.normal);
        manifest.sourceSpecular = pathString(bundle.specular);
        manifest.emissiveSource = pathString(bundle.emissive);
        manifest.unresolved = bundle.unresolved;

        if (bundle.diffuse) {
            const int tex = RegisterTexture(ctx, bundle.diffuse->asset);
            info.baseColorTexture = tex;
            info.baseColorTexCoord = canonical::GetTextureUvIndex(material, aiTextureType_DIFFUSE);
            const TextureAlphaInfo& alphaInfo = GetTextureAlphaInfo(ctx, tex);
            if (alphaInfo.hasAlpha) {
                info.alphaMode = alphaInfo.binaryAlpha ? "MASK" : "BLEND";
                info.alphaCutoff = alphaInfo.binaryAlpha ? 0.5f : info.alphaCutoff;
            }
        }

        if (bundle.emissive) {
            info.emissiveTexture = RegisterTexture(ctx, bundle.emissive->asset);
            info.emissiveTexCoord = canonical::GetTextureUvIndex(material, aiTextureType_EMISSIVE);
        }

        if (bundle.normal) {
            manifest.sourceNormalChannels = bundle.normal->sourceChannels;
            manifest.normalAlphaMin = bundle.normal->alphaStats.minValue;
            manifest.normalAlphaMax = bundle.normal->alphaStats.maxValue;
            manifest.normalAlphaStd = bundle.normal->alphaStats.stddev;

            // Keep the source G: the reflected V and tangent.w negate the
            // bitangent, which keeps the world-space normal that the earlier
            // unreflected frame with a G flip produced.
            const Import::DecodedTexture& normalDecoded = bundle.normal->decoded.decoded;
            manifest.normalYFlipped = false;
            LogMaterialDecision(info.name,
                                "NORMALS",
                                "used",
                                "canonical CryEngine normal G preserved for the reflected V/bitangent frame");
            info.normalTexture = RegisterGeneratedTexture(ctx,
                                                          "canonical-normal:" + info.name,
                                                          info.name + "_canonical_normal",
                                                          normalDecoded.rgba,
                                                          normalDecoded.width,
                                                          normalDecoded.height);
            info.normalTexCoord = canonical::GetTextureUvIndex(material, aiTextureType_NORMALS);

            if (bundle.normal->sourceChannels >= 4) {
                manifest.alphaInterpretation = bundle.normal->alphaStats.stddev > 0.05f ? "ddna_gloss" : "constant";
                std::vector<uint8_t> orm(static_cast<size_t>(normalDecoded.width) * normalDecoded.height * 4u, 255u);
                for (size_t px = 0; px < static_cast<size_t>(normalDecoded.width) * normalDecoded.height; ++px) {
                    const float gloss = static_cast<float>(bundle.normal->decoded.decoded.rgba[px * 4u + 3u]) / 255.0f;
                    const float roughness = 1.0f - gloss;
                    orm[px * 4u + 0u] = 255u;
                    orm[px * 4u + 1u] = static_cast<uint8_t>(std::round(std::clamp(roughness, 0.0f, 1.0f) * 255.0f));
                    orm[px * 4u + 2u] = 0u;
                    orm[px * 4u + 3u] = 255u;
                }
                const int ormTexture = RegisterGeneratedTexture(ctx,
                                                                "canonical-orm:" + info.name,
                                                                info.name + "_canonical_orm",
                                                                orm,
                                                                normalDecoded.width,
                                                                normalDecoded.height);
                info.metallicRoughnessTexture = ormTexture;
                info.occlusionTexture = ormTexture;
                info.metallicRoughnessTexCoord = canonical::GetTextureUvIndex(material, aiTextureType_SPECULAR);
                info.occlusionTexCoord = canonical::GetTextureUvIndex(material, aiTextureType_SPECULAR);
                info.roughnessFactor = 1.0f;
                info.metallicFactor = 1.0f;
                manifest.roughnessSource = "ddna_alpha";
                manifest.roughnessValueOrTarget = "ORM_G";
                LogMaterialDecision(info.name,
                                    "NORMAL_ALPHA",
                                    "used",
                                    "_ddna gloss converted to roughness and written to ORM G");
                ++ctx.ormTextureCount;
            } else {
                const float roughness = ReadRoughness(material);
                info.roughnessFactor = roughness;
                manifest.alphaInterpretation = "absent";
                manifest.roughnessSource = "scalar_fallback";
                std::ostringstream value;
                value << std::fixed << std::setprecision(3) << roughness;
                manifest.roughnessValueOrTarget = value.str();
                LogMaterialDecision(info.name,
                                    "NORMAL_ALPHA",
                                    "scalar_fallback",
                                    "canonical source normal has no alpha; roughness=" + value.str() + " from source material scalar");
            }
            std::cout << "[canonical] [material: " << info.name << "] normal_channels="
                      << manifest.sourceNormalChannels << " alpha_" << canonical::StatsString(bundle.normal->alphaStats)
                      << "\n";
        } else {
            const float roughness = ReadRoughness(material);
            info.roughnessFactor = roughness;
            manifest.roughnessSource = "scalar_fallback";
            std::ostringstream value;
            value << std::fixed << std::setprecision(3) << roughness;
            manifest.roughnessValueOrTarget = value.str();
            manifest.unresolved.push_back("normal: absent");
        }

        if (bundle.specular) {
            manifest.specularClassification = canonical::ClassifySpecular(*bundle.specular);
            manifest.specularAction = "discarded";
            LogMaterialDecision(info.name,
                                "SPECULAR",
                                "discarded",
                                "canonical source classification=" + manifest.specularClassification +
                                    "; no valid glTF mapping");
        }

        manifest.metallicSource = "dielectric_default";
        ctx.canonicalManifestEntries.push_back(manifest);
        ctx.materials.push_back(std::move(info));
    }
    if (ctx.materials.empty()) {
        ctx.materials.push_back(MaterialInfo{});
    }
}

void ConvertZeroDayMaterials(ImportContext& ctx) {
    ctx.materials.reserve(ctx.scene->mNumMaterials);
    ctx.zeroDayManifestEntries.reserve(ctx.scene->mNumMaterials);
    for (unsigned int i = 0; i < ctx.scene->mNumMaterials; ++i) {
        aiMaterial* material = ctx.scene->mMaterials[i];
        MaterialInfo info;
        aiString name;
        if (AI_SUCCESS == material->Get(AI_MATKEY_NAME, name)) {
            info.name = name.C_Str();
        } else {
            info.name = "material_" + std::to_string(i);
        }

        info.baseColorFactor = ReadBaseColor(material);
        info.roughnessFactor = 1.0f;
        info.metallicFactor = 0.0f;
        info.emissiveFactor = ReadEmissive(material);
        int twoSided = 0;
        material->Get(AI_MATKEY_TWOSIDED, twoSided);
        info.doubleSided = (twoSided != 0);
        if (ContainsCaseInsensitive(info.name, ".doublesided") ||
            ContainsCaseInsensitive(info.name, "_doublesided")) {
            info.doubleSided = true;
        }

        ZeroDayManifestEntry manifest;
        manifest.materialName = info.name;
        auto pathString = [](const std::optional<canonical::LoadedTexture>& texture) -> std::string {
            if (!texture) {
                return "";
            }
            return ReplaceBackslashes(texture->asset.sourcePath.string());
        };

        std::string error;
        const auto baseColor = LoadZeroDayTexture(ctx,
                                                  material,
                                                  info.name,
                                                  {aiTextureType_DIFFUSE, aiTextureType_BASE_COLOR},
                                                  "_BaseColor",
                                                  error);
        if (baseColor) {
            const int tex = RegisterTexture(ctx, baseColor->asset);
            info.baseColorTexture = tex;
            info.baseColorTexCoord = canonical::GetTextureUvIndex(material, aiTextureType_DIFFUSE);
            const TextureAlphaInfo& alphaInfo = GetTextureAlphaInfo(ctx, tex);
            if (alphaInfo.hasAlpha) {
                info.alphaMode = alphaInfo.binaryAlpha ? "MASK" : "BLEND";
                info.alphaCutoff = alphaInfo.binaryAlpha ? 0.5f : info.alphaCutoff;
                if (IsEmeraldSquareFoliageMaterial(ctx, info.name)) {
                    info.alphaMode = "MASK";
                    info.alphaCutoff = 0.5f;
                }
                manifest.baseColorAlphaUsed = true;
            }
            LogMaterialDecision(info.name,
                                "BASE_COLOR_ALPHA",
                                alphaInfo.hasAlpha ? "used" : "used_rgb_alpha_opaque",
                                std::string(PackedOrcaImportLabel(ctx)) +
                                    " BaseColor RGB plus opacity alpha imported through source texture");
        } else if (!error.empty()) {
            manifest.unresolved.push_back("baseColor: " + error);
            LogMaterialWarning(ctx,
                               info.name,
                               std::string(PackedOrcaImportLabel(ctx)) + " baseColor unresolved: " + error);
        }
        manifest.sourceBaseColor = pathString(baseColor);

        error.clear();
        bool emissiveFromConnection = false;
        const auto emissive = LoadZeroDayTexture(ctx,
                                                 material,
                                                 info.name,
                                                 {aiTextureType_EMISSIVE},
                                                 "_Emissive",
                                                 error,
                                                 &emissiveFromConnection);
        if (emissive) {
            info.emissiveTexture = RegisterTexture(ctx, emissive->asset);
            info.emissiveTexCoord = canonical::GetTextureUvIndex(material, aiTextureType_EMISSIVE);
            // Falcor's textured emissive channel replaces the FBX color; its
            // separate scene emissiveFactor scales the texture later. Every
            // packed ORCA pack targets Falcor, Zero Day ("optimized for Falcor
            // 4.0") included, whose FBX EmissiveColor 0.5 would otherwise halve
            // each connected emitter. A texture found only by file name (no FBX
            // connection) keeps the earlier per-pack rule.
            const bool textureReplacesEmissiveColor =
                emissiveFromConnection || IsEmeraldSquareImport(ctx) || IsSunTempleImport(ctx);
            const std::array<float, 3> fbxEmissiveColor = info.emissiveFactor;
            if (textureReplacesEmissiveColor) {
                info.emissiveFactor = {1.0f, 1.0f, 1.0f};
            }
            const std::string fbxColorText = std::to_string(fbxEmissiveColor[0]) + "," +
                                             std::to_string(fbxEmissiveColor[1]) + "," +
                                             std::to_string(fbxEmissiveColor[2]);
            LogMaterialDecision(info.name,
                                "EMISSIVE",
                                textureReplacesEmissiveColor ? "texture_replaces_fbx_emissive_color" : "used",
                                textureReplacesEmissiveColor
                                    ? std::string(PackedOrcaImportLabel(ctx)) +
                                          " emissive RGB texture replaces the FBX fallback color " + fbxColorText +
                                          ", matching Falcor"
                                    : std::string(PackedOrcaImportLabel(ctx)) +
                                          " emissive texture found by file name only (no FBX connection); FBX color " +
                                          fbxColorText + " kept as its factor");
        } else if (!error.empty()) {
            manifest.unresolved.push_back("emissive: " + error);
        }
        manifest.sourceEmissive = pathString(emissive);

        error.clear();
        const auto normal = LoadZeroDayTexture(ctx,
                                               material,
                                               info.name,
                                               {aiTextureType_NORMALS, aiTextureType_NORMAL_CAMERA},
                                               "_Normal",
                                               error);
        if (normal) {
            // Keep the source G. ConvertMeshes reflects V and tangent.w, which
            // negates the bitangent; together they keep the world-space normal
            // the earlier unreflected frame with a DirectX->GL G flip produced.
            const Import::DecodedTexture& normalDecoded = normal->decoded.decoded;
            info.normalTexture = RegisterGeneratedTexture(ctx,
                                                          "zeroday-normal:" + normal->asset.key,
                                                          info.name + "_zeroday_normal",
                                                          normalDecoded.rgba,
                                                          normalDecoded.width,
                                                          normalDecoded.height);
            info.normalTexCoord = canonical::GetTextureUvIndex(material, aiTextureType_NORMALS);
            manifest.normalYFlipped = false;
            LogMaterialDecision(info.name,
                                "NORMAL",
                                "used",
                                std::string(PackedOrcaImportLabel(ctx)) +
                                    " source normal G preserved for the reflected V/bitangent frame");
        } else if (!error.empty()) {
            manifest.unresolved.push_back("normal: " + error);
        }
        manifest.sourceNormal = pathString(normal);

        error.clear();
        const auto specular = LoadZeroDayTexture(ctx,
                                                 material,
                                                 info.name,
                                                 {aiTextureType_SPECULAR},
                                                 "_Specular",
                                                 error);
        if (specular) {
            const Import::DecodedTexture& src = specular->decoded.decoded;
            std::vector<uint8_t> orm(static_cast<size_t>(src.width) * src.height * 4u, 255u);
            bool sourceOcclusionUnused = IsEmeraldSquareImport(ctx) || IsSunTempleImport(ctx);
            for (size_t px = 0; px < static_cast<size_t>(src.width) * src.height; ++px) {
                sourceOcclusionUnused = sourceOcclusionUnused && src.rgba[px * 4u] == 0u;
                orm[px * 4u + 0u] = src.rgba[px * 4u + 0u];
                orm[px * 4u + 1u] = src.rgba[px * 4u + 1u];
                orm[px * 4u + 2u] = src.rgba[px * 4u + 2u];
                orm[px * 4u + 3u] = 255u;
            }
            const int ormTexture = RegisterGeneratedTexture(ctx,
                                                            "zeroday-orm:" + specular->asset.key,
                                                            info.name + "_zeroday_orm",
                                                            orm,
                                                            src.width,
                                                            src.height);
            info.metallicRoughnessTexture = ormTexture;
            // EmeraldSquare and SunTemple source packs use constant-zero red
            // for absent occlusion, rather than complete occlusion.
            // Keep the authored texture bytes while omitting the AO binding,
            // which would otherwise extinguish diffuse reflection.
            info.occlusionTexture = sourceOcclusionUnused ? -1 : ormTexture;
            info.metallicRoughnessTexCoord = canonical::GetTextureUvIndex(material, aiTextureType_SPECULAR);
            info.occlusionTexCoord = canonical::GetTextureUvIndex(material, aiTextureType_SPECULAR);
            info.roughnessFactor = 1.0f;
            info.metallicFactor = 1.0f;
            manifest.specularAction = "packed_rgb_to_orm";
            manifest.ormTarget = sourceOcclusionUnused
                                     ? "Specular R unused (constant zero; no AO binding), G->roughness, B->metalness"
                                     : "Specular R->AO, G->roughness, B->metalness";
            ++ctx.ormTextureCount;
            LogMaterialDecision(info.name,
                                "SPECULAR",
                                manifest.specularAction,
                                sourceOcclusionUnused
                                    ? std::string(PackedOrcaImportLabel(ctx)) +
                                          " source G=roughness and B=metalness preserved; unused constant-zero R is not bound as AO"
                                    : std::string(PackedOrcaImportLabel(ctx)) +
                                          " Specular R=AO, G=roughness, B=metalness preserved in glTF ORM texture");
        } else {
            info.roughnessFactor = ReadRoughness(material);
            info.metallicFactor = ReadMetallic(material);
            if (!error.empty()) {
                manifest.unresolved.push_back("specular: " + error);
                LogMaterialWarning(ctx,
                                   info.name,
                                   std::string(PackedOrcaImportLabel(ctx)) + " specular unresolved: " + error);
            }
        }
        manifest.sourceSpecular = pathString(specular);

        ctx.zeroDayManifestEntries.push_back(std::move(manifest));
        ctx.materials.push_back(std::move(info));
    }
    if (ctx.materials.empty()) {
        ctx.materials.push_back(MaterialInfo{});
    }
}

void ConvertMaterials(ImportContext& ctx) {
    if (UsePackedOrcaMaterialImport(ctx)) {
        ConvertZeroDayMaterials(ctx);
        return;
    }
    if (UseCanonicalImport(ctx) && IsBistroImport(ctx)) {
        ConvertCanonicalMaterials(ctx);
        return;
    }
    ConvertGenericMaterials(ctx);
}

void ConvertMeshes(ImportContext& ctx) {
    // Assimp keeps texture coordinates with a bottom-left V origin for every
    // source format: OBJ, FBX and DAE store them that way, and its glTF importer
    // converts glTF's top-left V on load. glTF TEXCOORD uses a top-left origin,
    // so V is written as 1 - v, as Assimp's own glTF exporter writes it.
    // v' = 1 - v reverses dP/dv, so tangent.w is negated with it to keep the
    // TANGENT handedness consistent with the emitted TEXCOORD: an upright quad
    // gets the glTF w = +1.
    ctx.primitives.resize(ctx.scene->mNumMeshes);
    ctx.stats.meshCount = ctx.scene->mNumMeshes;
    for (unsigned int meshIndex = 0; meshIndex < ctx.scene->mNumMeshes; ++meshIndex) {
        const aiMesh* mesh = ctx.scene->mMeshes[meshIndex];
        PrimitiveRef primitive;
        if (mesh->mNumBones > 0u) {
            ctx.warnings.push_back("Unsupported skinning omitted from mesh " +
                                   std::string(mesh->mName.C_Str()) +
                                   ": bones=" + std::to_string(mesh->mNumBones));
        }
        if (mesh->mNumAnimMeshes > 0u) {
            ctx.warnings.push_back("Unsupported morph/animated mesh data omitted from mesh " +
                                   std::string(mesh->mName.C_Str()) +
                                   ": anim_meshes=" + std::to_string(mesh->mNumAnimMeshes));
        }

        std::vector<float> positions;
        std::vector<float> normals;
        std::vector<float> texcoord0;
        std::vector<float> texcoord1;
        std::vector<float> tangents;
        std::vector<uint32_t> indices;

        positions.reserve(static_cast<size_t>(mesh->mNumVertices) * 3);
        if (mesh->HasNormals()) {
            normals.reserve(static_cast<size_t>(mesh->mNumVertices) * 3);
        }
        if (mesh->HasTextureCoords(0)) {
            texcoord0.reserve(static_cast<size_t>(mesh->mNumVertices) * 2);
        }
        if (mesh->HasTextureCoords(1)) {
            texcoord1.reserve(static_cast<size_t>(mesh->mNumVertices) * 2);
        }
        if (mesh->HasTangentsAndBitangents()) {
            tangents.reserve(static_cast<size_t>(mesh->mNumVertices) * 4);
        }

        for (unsigned int vertex = 0; vertex < mesh->mNumVertices; ++vertex) {
            const aiVector3D p = mesh->mVertices[vertex];
            positions.push_back(p.x);
            positions.push_back(p.y);
            positions.push_back(p.z);
            if (mesh->HasNormals()) {
                const aiVector3D n = mesh->mNormals[vertex];
                normals.push_back(n.x);
                normals.push_back(n.y);
                normals.push_back(n.z);
            }
            if (mesh->HasTextureCoords(0)) {
                const aiVector3D uv = mesh->mTextureCoords[0][vertex];
                texcoord0.push_back(uv.x);
                texcoord0.push_back(1.0f - uv.y);
            }
            if (mesh->HasTextureCoords(1)) {
                const aiVector3D uv = mesh->mTextureCoords[1][vertex];
                texcoord1.push_back(uv.x);
                texcoord1.push_back(1.0f - uv.y);
            }
            if (mesh->HasTangentsAndBitangents()) {
                auto tangent = MakeTangent(mesh, vertex);
                tangent[3] = -tangent[3];
                tangents.insert(tangents.end(), tangent.begin(), tangent.end());
            }
        }

        indices.reserve(static_cast<size_t>(mesh->mNumFaces) * 3);
        for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex) {
            const aiFace& face = mesh->mFaces[faceIndex];
            if (face.mNumIndices != 3) {
                continue;
            }
            indices.push_back(face.mIndices[0]);
            indices.push_back(face.mIndices[1]);
            indices.push_back(face.mIndices[2]);
        }
        ctx.stats.triangleCount += indices.size() / 3;

        primitive.positionAccessor = PushAccessor(ctx, positions, 3, 5126, "VEC3", 34962, true);
        if (!normals.empty()) {
            primitive.normalAccessor = PushAccessor(ctx, normals, 3, 5126, "VEC3", 34962);
        }
        if (!texcoord0.empty()) {
            primitive.texcoord0Accessor = PushAccessor(ctx, texcoord0, 2, 5126, "VEC2", 34962);
        }
        if (!texcoord1.empty()) {
            primitive.texcoord1Accessor = PushAccessor(ctx, texcoord1, 2, 5126, "VEC2", 34962);
        }
        if (!tangents.empty()) {
            primitive.tangentAccessor = PushAccessor(ctx, tangents, 4, 5126, "VEC4", 34962);
        }
        primitive.indicesAccessor = PushAccessor(ctx, indices, 1, 5125, "SCALAR", 34963);
        primitive.materialIndex = std::min<int>(mesh->mMaterialIndex,
                                                static_cast<int>(ctx.materials.size() - 1));
        ctx.primitives[meshIndex] = std::move(primitive);
    }
}

int RegisterMesh(ImportContext& ctx, const std::vector<unsigned int>& meshIndices, const std::string& name) {
    if (meshIndices.empty()) {
        return -1;
    }
    const auto found = ctx.meshByPrimitiveSet.find(meshIndices);
    if (found != ctx.meshByPrimitiveSet.end()) {
        return found->second;
    }
    MeshInfo meshInfo;
    meshInfo.name = name;
    for (unsigned int meshIndex : meshIndices) {
        meshInfo.primitiveIndices.push_back(static_cast<int>(meshIndex));
    }
    const int index = static_cast<int>(ctx.meshes.size());
    ctx.meshes.push_back(std::move(meshInfo));
    ctx.meshByPrimitiveSet.emplace(meshIndices, index);
    return index;
}

bool ShouldSkipPrimitiveForImport(const ImportContext& ctx, unsigned int meshIndex) {
    if (meshIndex >= ctx.primitives.size()) {
        return false;
    }
    const int materialIndex = ctx.primitives[meshIndex].materialIndex;
    if (materialIndex < 0 || materialIndex >= static_cast<int>(ctx.materials.size())) {
        return false;
    }
    return IsEmeraldSquareWorldGridMaterial(ctx, ctx.materials[materialIndex].name);
}

aiVector3D NormalizeCameraVector(aiVector3D v, const aiVector3D& fallback) {
    const float lenSq = v.x * v.x + v.y * v.y + v.z * v.z;
    if (!(lenSq > 1.0e-12f) || !std::isfinite(lenSq)) {
        return fallback;
    }
    const float invLen = 1.0f / std::sqrt(lenSq);
    return aiVector3D(v.x * invLen, v.y * invLen, v.z * invLen);
}

std::array<double, 4> QuaternionFromBasis(const aiVector3D& x,
                                          const aiVector3D& y,
                                          const aiVector3D& z) {
    const double m00 = x.x;
    const double m01 = y.x;
    const double m02 = z.x;
    const double m10 = x.y;
    const double m11 = y.y;
    const double m12 = z.y;
    const double m20 = x.z;
    const double m21 = y.z;
    const double m22 = z.z;
    const double trace = m00 + m11 + m22;
    double qx = 0.0;
    double qy = 0.0;
    double qz = 0.0;
    double qw = 1.0;
    if (trace > 0.0) {
        const double s = std::sqrt(trace + 1.0) * 2.0;
        qw = 0.25 * s;
        qx = (m21 - m12) / s;
        qy = (m02 - m20) / s;
        qz = (m10 - m01) / s;
    } else if (m00 > m11 && m00 > m22) {
        const double s = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;
        qw = (m21 - m12) / s;
        qx = 0.25 * s;
        qy = (m01 + m10) / s;
        qz = (m02 + m20) / s;
    } else if (m11 > m22) {
        const double s = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;
        qw = (m02 - m20) / s;
        qx = (m01 + m10) / s;
        qy = 0.25 * s;
        qz = (m12 + m21) / s;
    } else {
        const double s = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;
        qw = (m10 - m01) / s;
        qx = (m02 + m20) / s;
        qy = (m12 + m21) / s;
        qz = 0.25 * s;
    }
    const double len = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (!(len > 0.0) || !std::isfinite(len)) {
        return {0.0, 0.0, 0.0, 1.0};
    }
    return {qx / len, qy / len, qz / len, qw / len};
}

std::array<double, 4> CameraRotationFromLookUp(aiVector3D lookAt, aiVector3D up) {
    const aiVector3D forward = NormalizeCameraVector(lookAt, aiVector3D(0.0f, 0.0f, -1.0f));
    aiVector3D upN = NormalizeCameraVector(up, aiVector3D(0.0f, 1.0f, 0.0f));
    aiVector3D right = forward ^ upN;
    right = NormalizeCameraVector(right, aiVector3D(1.0f, 0.0f, 0.0f));
    const aiVector3D localZ(-forward.x, -forward.y, -forward.z);
    upN = NormalizeCameraVector(localZ ^ right, aiVector3D(0.0f, 1.0f, 0.0f));
    return QuaternionFromBasis(right, upN, localZ);
}

void ConvertCameras(ImportContext& ctx) {
    ctx.cameras.reserve(ctx.scene->mNumCameras);
    for (unsigned int cameraIndex = 0; cameraIndex < ctx.scene->mNumCameras; ++cameraIndex) {
        const aiCamera* camera = ctx.scene->mCameras[cameraIndex];
        if (!camera) {
            continue;
        }
        CameraInfo out;
        out.name = camera->mName.length > 0 ? std::string(camera->mName.C_Str())
                                            : "camera_" + std::to_string(ctx.cameras.size());
        const float aspect = camera->mAspect > 0.0f ? camera->mAspect : (16.0f / 9.0f);
        out.aspect = aspect;
        const float horizontalFov = camera->mHorizontalFOV > 0.0f ? camera->mHorizontalFOV : 0.7853982f;
        out.yfov = 2.0f * std::atan(std::tan(horizontalFov * 0.5f) / aspect);
        out.znear = camera->mClipPlaneNear > 0.0f ? camera->mClipPlaneNear : 0.01f;
        out.zfar = camera->mClipPlaneFar > out.znear ? camera->mClipPlaneFar : 1000000.0f;
        out.translation = {camera->mPosition.x, camera->mPosition.y, camera->mPosition.z};
        out.rotation = CameraRotationFromLookUp(camera->mLookAt, camera->mUp);

        const int index = static_cast<int>(ctx.cameras.size());
        ctx.cameraByName.emplace(out.name, index);
        ctx.cameras.push_back(std::move(out));
    }
}

// Reads each Model's Visibility and Visibility Inheritance and the Model parent links from a binary FBX file.
// Assimp 6 drops Visibility, so a hidden Model (an alternate state an artist switched off) would be imported and
// rendered on top of the visible one. Returns false with `error` set when the file is not a readable binary FBX.
bool ReadFbxModelVisibility(const fs::path& path, ImportContext& ctx, std::string& error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "cannot open " + path.string();
        return false;
    }
    const std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    static const char kMagic[] = "Kaydara FBX Binary  ";
    if (data.size() < 27 || std::memcmp(data.data(), kMagic, 20) != 0) {
        error = "not a binary FBX file (ASCII FBX visibility is not read)";
        return false;
    }
    uint32_t version = 0;
    std::memcpy(&version, data.data() + 23, 4);
    const bool wide = version >= 7500u;
    const size_t headerSize = wide ? 25u : 13u;
    bool ok = true;
    auto readU = [&](size_t offset, size_t bytes) -> uint64_t {
        if (offset + bytes > data.size()) {
            ok = false;
            return 0;
        }
        uint64_t value = 0;
        std::memcpy(&value, data.data() + offset, bytes);
        return value;
    };
    struct Prop {
        char type = 0;
        int64_t integer = 0;
        double real = 0.0;
        std::string text;
    };
    // Reads the properties of the node record at `offset`; returns the record's end offset (0 for a null record).
    auto readRecord = [&](size_t offset, std::string& name, std::vector<Prop>* props, size_t& childrenBegin) -> uint64_t {
        const uint64_t end = readU(offset, wide ? 8 : 4);
        const uint64_t propCount = readU(offset + (wide ? 8 : 4), wide ? 8 : 4);
        const uint64_t propBytes = readU(offset + (wide ? 16 : 8), wide ? 8 : 4);
        size_t cursor = offset + (wide ? 24 : 12);
        const uint64_t nameLength = readU(cursor, 1);
        cursor += 1;
        if (!ok || cursor + nameLength > data.size()) {
            ok = false;
            return 0;
        }
        name.assign(reinterpret_cast<const char*>(data.data() + cursor), nameLength);
        cursor += nameLength;
        childrenBegin = cursor + propBytes;
        if (end == 0) {
            return 0;
        }
        for (uint64_t i = 0; props && i < propCount && ok; ++i) {
            Prop prop;
            prop.type = static_cast<char>(readU(cursor, 1));
            cursor += 1;
            switch (prop.type) {
                case 'Y': prop.integer = static_cast<int16_t>(readU(cursor, 2)); cursor += 2; break;
                case 'C': prop.integer = static_cast<int64_t>(readU(cursor, 1)); cursor += 1; break;
                case 'I': prop.integer = static_cast<int32_t>(readU(cursor, 4)); cursor += 4; break;
                case 'L': prop.integer = static_cast<int64_t>(readU(cursor, 8)); cursor += 8; break;
                case 'F': {
                    const uint32_t bits = static_cast<uint32_t>(readU(cursor, 4));
                    float value = 0.0f;
                    std::memcpy(&value, &bits, 4);
                    prop.real = value;
                    cursor += 4;
                    break;
                }
                case 'D': {
                    const uint64_t bits = readU(cursor, 8);
                    std::memcpy(&prop.real, &bits, 8);
                    cursor += 8;
                    break;
                }
                case 'S':
                case 'R': {
                    const uint64_t length = readU(cursor, 4);
                    cursor += 4;
                    if (cursor + length > data.size()) {
                        ok = false;
                        break;
                    }
                    prop.text.assign(reinterpret_cast<const char*>(data.data() + cursor), length);
                    cursor += length;
                    break;
                }
                case 'f': case 'd': case 'l': case 'i': case 'b':
                    cursor += 12 + readU(cursor + 8, 4);
                    break;
                default:
                    ok = false;
                    break;
            }
            if (prop.type == 'Y' || prop.type == 'C' || prop.type == 'I' || prop.type == 'L') {
                prop.real = static_cast<double>(prop.integer);
            }
            props->push_back(std::move(prop));
        }
        return end;
    };
    // Visits the child records of [begin, end).
    auto forEachChild = [&](size_t begin, uint64_t end, const std::function<void(size_t, const std::string&, uint64_t, size_t)>& visit) {
        size_t cursor = begin;
        while (ok && cursor + headerSize <= end) {
            std::string name;
            size_t childrenBegin = 0;
            const uint64_t childEnd = readRecord(cursor, name, nullptr, childrenBegin);
            if (!ok || childEnd == 0 || childEnd <= cursor || childEnd > end) {
                break;
            }
            visit(cursor, name, childEnd, childrenBegin);
            cursor = childEnd;
        }
    };
    std::vector<std::array<int64_t, 2>> links;
    forEachChild(27, data.size(), [&](size_t, const std::string& name, uint64_t end, size_t childrenBegin) {
        if (name == "Objects") {
            forEachChild(childrenBegin, end, [&](size_t modelOffset, const std::string& objectName, uint64_t modelEnd, size_t) {
                if (objectName != "Model") {
                    return;
                }
                std::string recordName;
                std::vector<Prop> props;
                size_t modelChildren = 0;
                readRecord(modelOffset, recordName, &props, modelChildren);
                if (props.size() < 2 || props[0].type != 'L' || props[1].type != 'S') {
                    return;
                }
                ImportContext::FbxModel model;
                model.name = props[1].text.substr(0, props[1].text.find('\0'));
                forEachChild(modelChildren, modelEnd, [&](size_t, const std::string& childName, uint64_t childEnd, size_t p70Children) {
                    if (childName != "Properties70") {
                        return;
                    }
                    forEachChild(p70Children, childEnd, [&](size_t pOffset, const std::string& pName, uint64_t, size_t) {
                        if (pName != "P") {
                            return;
                        }
                        std::string unused;
                        std::vector<Prop> p;
                        size_t pChildren = 0;
                        readRecord(pOffset, unused, &p, pChildren);
                        if (p.size() < 5 || p[0].type != 'S') {
                            return;
                        }
                        if (p[0].text == "Visibility") {
                            model.ownHidden = p[4].real == 0.0;
                        } else if (p[0].text == "Visibility Inheritance") {
                            model.inheritVisibility = p[4].real != 0.0;
                        }
                    });
                });
                ctx.fbxModelIdsByName[model.name].push_back(props[0].integer);
                ctx.fbxModels.emplace(props[0].integer, std::move(model));
            });
        } else if (name == "FBXHeaderExtension") {
            forEachChild(childrenBegin, end, [&](size_t, const std::string& hName, uint64_t hEnd, size_t hChildren) {
                if (hName != "SceneInfo") {
                    return;
                }
                forEachChild(hChildren, hEnd, [&](size_t, const std::string& sName, uint64_t sEnd, size_t sChildren) {
                    if (sName != "Properties70") {
                        return;
                    }
                    forEachChild(sChildren, sEnd, [&](size_t pOffset, const std::string& pName, uint64_t, size_t) {
                        if (pName != "P") {
                            return;
                        }
                        std::string unused;
                        std::vector<Prop> p;
                        size_t pChildren = 0;
                        readRecord(pOffset, unused, &p, pChildren);
                        if (p.size() >= 5 && p[0].type == 'S' && p[0].text == "Original|ApplicationName" && p[4].type == 'S') {
                            ctx.fbxApplicationName = p[4].text;
                        }
                    });
                });
            });
        } else if (name == "Connections") {
            forEachChild(childrenBegin, end, [&](size_t cOffset, const std::string& cName, uint64_t, size_t) {
                if (cName != "C") {
                    return;
                }
                std::string unused;
                std::vector<Prop> p;
                size_t cChildren = 0;
                readRecord(cOffset, unused, &p, cChildren);
                if (p.size() >= 3 && p[0].type == 'S' && p[0].text == "OO" && p[1].type == 'L' && p[2].type == 'L') {
                    links.push_back({p[1].integer, p[2].integer});
                }
            });
        }
    });
    if (!ok) {
        error = "malformed binary FBX record";
        ctx.fbxModels.clear();
        ctx.fbxModelIdsByName.clear();
        return false;
    }
    for (const auto& link : links) {
        auto child = ctx.fbxModels.find(link[0]);
        if (child != ctx.fbxModels.end() && ctx.fbxModels.count(link[1]) != 0) {
            child->second.parent = link[1];
        }
    }
    return true;
}

// A Model is hidden when its own Visibility is 0, or when it inherits visibility from a hidden parent Model.
bool FbxModelHidden(const ImportContext& ctx, int64_t id) {
    for (int depth = 0; depth < 4096; ++depth) {
        auto it = ctx.fbxModels.find(id);
        if (it == ctx.fbxModels.end()) {
            return false;
        }
        if (it->second.ownHidden) {
            return true;
        }
        if (!it->second.inheritVisibility || it->second.parent == 0) {
            return false;
        }
        id = it->second.parent;
    }
    return false;
}

// Whether the FBX Model an Assimp node came from is hidden. Assimp names a node after its Model. When several Models
// share the name and disagree, the parent's name decides (Assimp's "$AssimpFbx$" pivot nodes are skipped); a node
// that stays ambiguous is kept visible and counted.
bool FbxNodeHidden(ImportContext& ctx, const aiNode* node) {
    auto found = ctx.fbxModelIdsByName.find(node->mName.C_Str());
    if (found == ctx.fbxModelIdsByName.end()) {
        return false;
    }
    size_t hiddenCount = 0;
    for (int64_t id : found->second) {
        hiddenCount += FbxModelHidden(ctx, id) ? 1u : 0u;
    }
    if (hiddenCount == 0u || hiddenCount == found->second.size()) {
        return hiddenCount != 0u;
    }
    const aiNode* parent = node->mParent;
    while (parent && std::strstr(parent->mName.C_Str(), "$AssimpFbx$")) {
        parent = parent->mParent;
    }
    std::optional<int64_t> match;
    for (int64_t id : found->second) {
        auto parentModel = ctx.fbxModels.find(ctx.fbxModels[id].parent);
        const bool parentMatches = parentModel != ctx.fbxModels.end()
            ? (parent && parentModel->second.name == parent->mName.C_Str())
            : (!parent || !parent->mParent);
        if (parentMatches) {
            if (match) {
                match.reset();
                break;
            }
            match = id;
        }
    }
    if (!match) {
        ctx.fbxAmbiguousNodeNames.push_back(node->mName.C_Str());
        return false;
    }
    return FbxModelHidden(ctx, *match);
}

// FbxNodeHidden, asked once per node (the selection prepass and ConvertNode both need it).
bool FbxNodeHiddenCached(ImportContext& ctx, const aiNode* node) {
    if (!ctx.fbxVisibilityRead || node->mNumMeshes == 0) {
        return false;
    }
    auto found = ctx.fbxNodeHidden.find(node);
    if (found != ctx.fbxNodeHidden.end()) {
        return found->second;
    }
    const bool hidden = FbxNodeHidden(ctx, node);
    ctx.fbxNodeHidden.emplace(node, hidden);
    return hidden;
}

// The Cinema 4D polygon-selection rule's gate (Dario, 2026-11-24): the file has at least one sibling overlap between a
// "*_Polygon_Selection*" mesh and another mesh under the same parent, and at least one material named "*_c4d*".
// C4D exports a material on a polygon selection as the whole object plus a coincident sub-mesh of the selected
// polygons; the header may name another application (Zero Day's FBX files were re-saved by Maya).
bool C4dPolygonSelectionRuleApplies(const ImportContext& ctx, size_t overlapPairs, size_t& c4dMaterials) {
    c4dMaterials = 0;
    for (const MaterialInfo& material : ctx.materials) {
        c4dMaterials += (ToLowerAscii(material.name).find("_c4d") != std::string::npos) ? 1u : 0u;
    }
    return overlapPairs > 0 && c4dMaterials > 0;
}

namespace {
constexpr double kC4dSelectionEpsilon = 1.0e-4;  // world units; a triangle overlaps when its 3 quantized positions match
using TriangleKey = std::array<int64_t, 9>;
struct TriangleKeyHash {
    size_t operator()(const TriangleKey& k) const {
        size_t h = 1469598103934665603ull;
        for (int64_t v : k) h = (h ^ static_cast<size_t>(v)) * 1099511628211ull;
        return h;
    }
};
using TriangleKeySet = std::unordered_set<TriangleKey, TriangleKeyHash>;

TriangleKey KeyOfTriangle(const aiMesh* mesh, const aiFace& face, const aiMatrix4x4& world) {
    std::array<std::array<int64_t, 3>, 3> v{};
    for (int i = 0; i < 3; ++i) {
        const aiVector3D p = world * mesh->mVertices[face.mIndices[i]];
        v[i] = {static_cast<int64_t>(std::llround(p.x / kC4dSelectionEpsilon)),
                static_cast<int64_t>(std::llround(p.y / kC4dSelectionEpsilon)),
                static_cast<int64_t>(std::llround(p.z / kC4dSelectionEpsilon))};
    }
    std::sort(v.begin(), v.end());
    return {v[0][0], v[0][1], v[0][2], v[1][0], v[1][1], v[1][2], v[2][0], v[2][1], v[2][2]};
}

struct SelectionChild {
    const aiNode* node = nullptr;
    aiMatrix4x4 world;
    bool selection = false;
};
}  // namespace

// Cinema 4D polygon selections (gate: C4dPolygonSelectionRuleApplies). Under each parent, a "*_Polygon_Selection*"
// mesh's material wins over a sibling's coincident triangles: those are dropped from the sibling (the base mesh), and
// its other triangles stay. Of two selections that coincide with the same material, the first keeps the triangles.
// Selections that coincide with different materials are logged as unresolved and both kept: the file does not record
// C4D's tag order. Fills ctx.c4dKeptFaces; ConvertNode writes a filtered primitive for those meshes.
void ApplyC4dPolygonSelectionRule(ImportContext& ctx) {
    struct Drop { std::string parent, base, selection; size_t dropped = 0, total = 0; bool duplicate = false; };
    struct Unresolved { std::string parent, a, b, materialA, materialB; size_t count = 0; };
    std::vector<Drop> drops;
    std::vector<Unresolved> unresolved;
    std::unordered_map<const aiNode*, std::unordered_map<unsigned int, std::vector<uint32_t>>> kept;
    auto materialName = [&](const aiMesh* mesh) {
        const int index = std::min<int>(mesh->mMaterialIndex, static_cast<int>(ctx.materials.size()) - 1);
        return index >= 0 ? ctx.materials[index].name : std::string();
    };
    std::function<void(const aiNode*, const aiMatrix4x4&)> visit = [&](const aiNode* parent, const aiMatrix4x4& parentWorld) {
        std::vector<SelectionChild> children;
        for (unsigned int c = 0; c < parent->mNumChildren; ++c) {
            const aiNode* child = parent->mChildren[c];
            SelectionChild entry{child, parentWorld * child->mTransformation,
                                 std::strstr(child->mName.C_Str(), "_Polygon_Selection") != nullptr};
            if (child->mNumMeshes > 0 && !FbxNodeHiddenCached(ctx, child)) {
                children.push_back(entry);
            }
            visit(child, entry.world);
        }
        bool anySelection = false;
        for (const SelectionChild& c : children) anySelection |= c.selection;
        if (!anySelection) {
            return;
        }
        // Each selection's triangles, after its own earlier duplicates are removed.
        std::vector<std::pair<const SelectionChild*, std::vector<std::pair<std::string, TriangleKeySet>>>> selections;
        for (const SelectionChild& c : children) {
            if (!c.selection) continue;
            std::vector<std::pair<std::string, TriangleKeySet>> perMaterial;
            for (unsigned int m = 0; m < c.node->mNumMeshes; ++m) {
                const aiMesh* mesh = ctx.scene->mMeshes[c.node->mMeshes[m]];
                std::vector<uint32_t> keepFaces;
                size_t duplicateDrops = 0;
                TriangleKeySet keys;
                for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
                    const aiFace& face = mesh->mFaces[f];
                    if (face.mNumIndices != 3) continue;
                    const TriangleKey key = KeyOfTriangle(mesh, face, c.world);
                    bool duplicate = false;
                    for (const auto& [earlier, earlierSets] : selections) {
                        for (const auto& [mat, set] : earlierSets) {
                            if (mat == materialName(mesh) && set.count(key)) { duplicate = true; break; }
                        }
                        if (duplicate) break;
                    }
                    if (duplicate) { ++duplicateDrops; continue; }
                    keepFaces.push_back(f);
                    keys.insert(key);
                }
                if (duplicateDrops > 0) {
                    kept[c.node][m] = keepFaces;
                    drops.push_back({parent->mName.C_Str(), c.node->mName.C_Str(), "(earlier same-material selection)",
                                     duplicateDrops, mesh->mNumFaces, true});
                }
                perMaterial.emplace_back(materialName(mesh), std::move(keys));
            }
            // Coincidence with an earlier selection of a different material: logged, both kept.
            for (const auto& [earlier, earlierSets] : selections) {
                for (const auto& [matA, setA] : earlierSets) {
                    for (const auto& [matB, setB] : perMaterial) {
                        if (matA == matB) continue;
                        size_t count = 0;
                        for (const TriangleKey& k : setB) count += setA.count(k);
                        if (count) unresolved.push_back({parent->mName.C_Str(), earlier->node->mName.C_Str(),
                                                         c.node->mName.C_Str(), matA, matB, count});
                    }
                }
            }
            selections.emplace_back(&c, std::move(perMaterial));
        }
        // Base meshes: drop the triangles a selection covers.
        for (const SelectionChild& c : children) {
            if (c.selection) continue;
            for (unsigned int m = 0; m < c.node->mNumMeshes; ++m) {
                const aiMesh* mesh = ctx.scene->mMeshes[c.node->mMeshes[m]];
                std::unordered_map<std::string, size_t> droppedBy;
                std::vector<uint32_t> keepFaces;
                for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
                    const aiFace& face = mesh->mFaces[f];
                    if (face.mNumIndices != 3) continue;
                    const TriangleKey key = KeyOfTriangle(mesh, face, c.world);
                    const char* by = nullptr;
                    for (const auto& [sel, sets] : selections) {
                        for (const auto& [mat, set] : sets) {
                            if (set.count(key)) { by = sel->node->mName.C_Str(); break; }
                        }
                        if (by) break;
                    }
                    if (by) droppedBy[by] += 1; else keepFaces.push_back(f);
                }
                if (!droppedBy.empty()) {
                    kept[c.node][m] = keepFaces;
                    for (const auto& [sel, n] : droppedBy) {
                        drops.push_back({parent->mName.C_Str(), c.node->mName.C_Str(), sel, n, mesh->mNumFaces, false});
                    }
                }
            }
        }
    };
    visit(ctx.scene->mRootNode, ctx.scene->mRootNode->mTransformation);
    size_t c4dMaterials = 0;
    const size_t basePairs = static_cast<size_t>(std::count_if(drops.begin(), drops.end(), [](const Drop& d) { return !d.duplicate; }));
    const bool applies = C4dPolygonSelectionRuleApplies(ctx, basePairs, c4dMaterials);
    std::cout << "[Import] C4D polygon selections: file ApplicationName \"" << ctx.fbxApplicationName << "\"; "
              << c4dMaterials << " _c4d materials; " << basePairs << " base/selection overlap pairs; epsilon "
              << kC4dSelectionEpsilon << "; rule " << (applies ? "applied" : "not applied") << std::endl;
    if (!applies) {
        return;
    }
    size_t totalDropped = 0;
    for (const Drop& d : drops) {
        std::cout << "[Import] C4D selection: parent " << d.parent << " | "
                  << (d.duplicate ? "duplicate selection " : "base ") << d.base << " | selection " << d.selection
                  << " | dropped " << d.dropped << " of " << d.total << " triangles" << std::endl;
        totalDropped += d.dropped;
    }
    size_t unresolvedTriangles = 0;
    for (const Unresolved& u : unresolved) {
        std::cout << "[Import] C4D selection: parent " << u.parent << " | selections " << u.a << " (" << u.materialA
                  << ") and " << u.b << " (" << u.materialB << ") coincide on " << u.count
                  << " triangles; unresolved, both kept" << std::endl;
        unresolvedTriangles += u.count;
    }
    std::cout << "[Import] C4D polygon selections: dropped " << totalDropped << " triangles in " << drops.size()
              << " pairs; " << unresolved.size() << " different-material selection overlaps (" << unresolvedTriangles
              << " triangles) left unresolved" << std::endl;
    if (!unresolved.empty()) {
        ctx.warnings.push_back(std::to_string(unresolved.size()) +
                               " C4D polygon selections coincide with a different material; both kept");
    }
    ctx.c4dKeptFaces = std::move(kept);
}

// A primitive for `meshIndex` with only `faces`, sharing the mesh's vertex accessors.
int FilteredPrimitive(ImportContext& ctx, unsigned int meshIndex, const std::vector<uint32_t>& faces) {
    const aiMesh* mesh = ctx.scene->mMeshes[meshIndex];
    std::vector<uint32_t> indices;
    indices.reserve(faces.size() * 3u);
    for (uint32_t f : faces) {
        const aiFace& face = mesh->mFaces[f];
        indices.insert(indices.end(), {face.mIndices[0], face.mIndices[1], face.mIndices[2]});
    }
    PrimitiveRef primitive = ctx.primitives[meshIndex];
    primitive.indicesAccessor = PushAccessor(ctx, indices, 1, 5125, "SCALAR", 34963);
    ctx.primitives.push_back(primitive);
    return static_cast<int>(ctx.primitives.size() - 1);
}

int ConvertNode(ImportContext& ctx, const aiNode* node) {
    NodeInfo out;
    out.name = node->mName.length > 0 ? std::string(node->mName.C_Str())
                                      : "node_" + std::to_string(ctx.nodes.size());
    aiVector3D scaling(1.0f, 1.0f, 1.0f);
    aiQuaternion rotation;
    aiVector3D position;
    node->mTransformation.Decompose(scaling, rotation, position);
    out.translation = {position.x, position.y, position.z};
    out.rotation = {rotation.x, rotation.y, rotation.z, rotation.w};
    out.scale = {scaling.x, scaling.y, scaling.z};
    std::vector<unsigned int> primitiveSet;
    primitiveSet.reserve(node->mNumMeshes);
    const bool hiddenInSource = FbxNodeHiddenCached(ctx, node);
    if (hiddenInSource) {
        // Hidden in the FBX file: keep the node and its transform for its children, drop its meshes.
        ctx.fbxHiddenNodesSkipped += 1;
        for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
            ctx.fbxHiddenTrianglesSkipped += ctx.scene->mMeshes[node->mMeshes[i]]->mNumFaces;
        }
    }
    const auto c4dKept = ctx.c4dKeptFaces.find(node);
    for (unsigned int i = 0; !hiddenInSource && i < node->mNumMeshes; ++i) {
        const unsigned int meshIndex = node->mMeshes[i];
        if (ShouldSkipPrimitiveForImport(ctx, meshIndex)) {
            continue;
        }
        if (c4dKept != ctx.c4dKeptFaces.end()) {
            if (auto faces = c4dKept->second.find(i); faces != c4dKept->second.end()) {
                // A C4D polygon selection covers some of these triangles (ApplyC4dPolygonSelectionRule).
                if (!faces->second.empty()) {
                    primitiveSet.push_back(static_cast<unsigned int>(FilteredPrimitive(ctx, meshIndex, faces->second)));
                }
                continue;
            }
        }
        primitiveSet.push_back(meshIndex);
    }
    out.mesh = RegisterMesh(ctx, primitiveSet, out.name);
    if (auto cameraIt = ctx.cameraByName.find(out.name); cameraIt != ctx.cameraByName.end()) {
        out.camera = cameraIt->second;
    }
    const int nodeIndex = static_cast<int>(ctx.nodes.size());
    ctx.nodes.push_back(std::move(out));
    ctx.nodes[nodeIndex].children.reserve(node->mNumChildren);
    for (unsigned int childIndex = 0; childIndex < node->mNumChildren; ++childIndex) {
        const int convertedChild = ConvertNode(ctx, node->mChildren[childIndex]);
        ctx.nodes[nodeIndex].children.push_back(convertedChild);
    }
    return nodeIndex;
}

std::string VectorToJson(const std::vector<double>& values) {
    std::ostringstream stream;
    stream << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            stream << ",";
        }
        stream << values[i];
    }
    stream << "]";
    return stream.str();
}

template <size_t N>
std::string ArrayToJson(const std::array<float, N>& values) {
    std::ostringstream stream;
    stream << "[";
    for (size_t i = 0; i < N; ++i) {
        if (i > 0) {
            stream << ",";
        }
        stream << values[i];
    }
    stream << "]";
    return stream.str();
}

template <size_t N>
std::string ArrayToJson(const std::array<double, N>& values) {
    std::ostringstream stream;
    stream << "[";
    for (size_t i = 0; i < N; ++i) {
        if (i > 0) {
            stream << ",";
        }
        stream << values[i];
    }
    stream << "]";
    return stream.str();
}

std::string BuildGltfJson(const ImportContext& ctx, const std::string& binaryUri) {
    std::ostringstream json;
    json << "{";
    json << "\"asset\":{\"generator\":\"PathTracerImport " << JsonEscape(kSemanticVersion)
         << " (" << JsonEscape(kGitHash) << ")\",\"version\":\"2.0\"},";
    json << "\"buffers\":[{";
    if (ctx.options.outputFormat == ImportOptions::OutputFormat::Gltf) {
        json << "\"uri\":\"" << JsonEscape(binaryUri) << "\",";
    }
    json << "\"byteLength\":" << ctx.buffer.bytes().size() << "}],";

    json << "\"bufferViews\":[";
    for (size_t i = 0; i < ctx.bufferViews.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        const auto& view = ctx.bufferViews[i];
        json << "{"
             << "\"buffer\":0,"
             << "\"byteOffset\":" << view.offset << ","
             << "\"byteLength\":" << view.length;
        if (view.target >= 0) {
            json << ",\"target\":" << view.target;
        }
        json << "}";
    }
    json << "],";

    json << "\"accessors\":[";
    for (size_t i = 0; i < ctx.accessors.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        const auto& accessor = ctx.accessors[i];
        json << "{"
             << "\"bufferView\":" << accessor.bufferView << ","
             << "\"byteOffset\":" << accessor.offset << ","
             << "\"componentType\":" << accessor.componentType << ","
             << "\"count\":" << accessor.count << ","
             << "\"type\":\"" << accessor.type << "\"";
        if (accessor.normalized) {
            json << ",\"normalized\":true";
        }
        if (accessor.hasMin) {
            json << ",\"min\":" << VectorToJson(accessor.minValues);
        }
        if (accessor.hasMax) {
            json << ",\"max\":" << VectorToJson(accessor.maxValues);
        }
        json << "}";
    }
    json << "],";

    if (!ctx.images.empty()) {
        json << "\"images\":[";
        for (size_t i = 0; i < ctx.images.size(); ++i) {
            if (i > 0) {
                json << ",";
            }
            const auto& image = ctx.images[i];
            json << "{";
            bool firstField = true;
            if (!image.name.empty()) {
                json << "\"name\":\"" << JsonEscape(image.name) << "\"";
                firstField = false;
            }
            if (!image.uri.empty()) {
                if (!firstField) {
                    json << ",";
                }
                json << "\"uri\":\"" << JsonEscape(image.uri) << "\"";
                firstField = false;
                if (!image.mimeType.empty()) {
                    json << ",\"mimeType\":\"" << JsonEscape(image.mimeType) << "\"";
                }
            }
            if (image.bufferView >= 0) {
                if (!firstField) {
                    json << ",";
                }
                json << "\"bufferView\":" << image.bufferView
                     << ",\"mimeType\":\"" << JsonEscape(image.mimeType) << "\"";
            }
            json << "}";
        }
        json << "],";

        json << "\"samplers\":[";
        for (size_t i = 0; i < ctx.samplers.size(); ++i) {
            if (i > 0) {
                json << ",";
            }
            const auto& sampler = ctx.samplers[i];
            json << "{"
                 << "\"magFilter\":" << sampler.magFilter << ","
                 << "\"minFilter\":" << sampler.minFilter << ","
                 << "\"wrapS\":" << sampler.wrapS << ","
                 << "\"wrapT\":" << sampler.wrapT << "}";
        }
        json << "],";

        json << "\"textures\":[";
        for (size_t i = 0; i < ctx.textures.size(); ++i) {
            if (i > 0) {
                json << ",";
            }
            const auto& texture = ctx.textures[i];
            json << "{"
                 << "\"sampler\":" << texture.sampler << ","
                 << "\"source\":" << texture.image << "}";
        }
        json << "],";
    }

    json << "\"materials\":[";
    for (size_t i = 0; i < ctx.materials.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        const auto& material = ctx.materials[i];
        json << "{"
             << "\"name\":\"" << JsonEscape(material.name) << "\","
             << "\"pbrMetallicRoughness\":{"
             << "\"baseColorFactor\":" << ArrayToJson(material.baseColorFactor) << ","
             << "\"metallicFactor\":" << material.metallicFactor << ","
             << "\"roughnessFactor\":" << material.roughnessFactor;
        if (material.baseColorTexture >= 0) {
            json << ",\"baseColorTexture\":{\"index\":" << material.baseColorTexture;
            if (material.baseColorTexCoord > 0) {
                json << ",\"texCoord\":" << material.baseColorTexCoord;
            }
            json << "}";
        }
        if (material.metallicRoughnessTexture >= 0) {
            json << ",\"metallicRoughnessTexture\":{\"index\":" << material.metallicRoughnessTexture;
            if (material.metallicRoughnessTexCoord > 0) {
                json << ",\"texCoord\":" << material.metallicRoughnessTexCoord;
            }
            json << "}";
        }
        json << "},"
             << "\"emissiveFactor\":" << ArrayToJson(material.emissiveFactor);
        if (material.normalTexture >= 0) {
            json << ",\"normalTexture\":{\"index\":" << material.normalTexture;
            if (material.normalTexCoord > 0) {
                json << ",\"texCoord\":" << material.normalTexCoord;
            }
            json << "}";
        }
        if (material.occlusionTexture >= 0) {
            json << ",\"occlusionTexture\":{\"index\":" << material.occlusionTexture;
            if (material.occlusionTexCoord > 0) {
                json << ",\"texCoord\":" << material.occlusionTexCoord;
            }
            json << "}";
        }
        if (material.emissiveTexture >= 0) {
            json << ",\"emissiveTexture\":{\"index\":" << material.emissiveTexture;
            if (material.emissiveTexCoord > 0) {
                json << ",\"texCoord\":" << material.emissiveTexCoord;
            }
            json << "}";
        }
        if (material.doubleSided) {
            json << ",\"doubleSided\":true";
        }
        if (material.alphaMode != "OPAQUE") {
            json << ",\"alphaMode\":\"" << material.alphaMode << "\"";
            if (material.alphaMode == "MASK") {
                json << ",\"alphaCutoff\":" << material.alphaCutoff;
            }
        }
        json << "}";
    }
    json << "],";

    json << "\"meshes\":[";
    for (size_t i = 0; i < ctx.meshes.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        const auto& mesh = ctx.meshes[i];
        json << "{"
             << "\"name\":\"" << JsonEscape(mesh.name) << "\","
             << "\"primitives\":[";
        for (size_t p = 0; p < mesh.primitiveIndices.size(); ++p) {
            if (p > 0) {
                json << ",";
            }
            const auto& primitive = ctx.primitives[mesh.primitiveIndices[p]];
            json << "{"
                 << "\"attributes\":{"
                 << "\"POSITION\":" << primitive.positionAccessor;
            if (primitive.normalAccessor >= 0) {
                json << ",\"NORMAL\":" << primitive.normalAccessor;
            }
            if (primitive.texcoord0Accessor >= 0) {
                json << ",\"TEXCOORD_0\":" << primitive.texcoord0Accessor;
            }
            if (primitive.texcoord1Accessor >= 0) {
                json << ",\"TEXCOORD_1\":" << primitive.texcoord1Accessor;
            }
            if (primitive.tangentAccessor >= 0) {
                json << ",\"TANGENT\":" << primitive.tangentAccessor;
            }
            json << "},"
                 << "\"indices\":" << primitive.indicesAccessor << ","
                 << "\"material\":" << primitive.materialIndex << ","
                 << "\"mode\":4}";
        }
        json << "]}";
    }
    json << "],";

    if (!ctx.cameras.empty()) {
        json << "\"cameras\":[";
        for (size_t i = 0; i < ctx.cameras.size(); ++i) {
            if (i > 0) {
                json << ",";
            }
            const auto& camera = ctx.cameras[i];
            json << "{"
                 << "\"name\":\"" << JsonEscape(camera.name) << "\","
                 << "\"type\":\"perspective\","
                 << "\"perspective\":{"
                 << "\"yfov\":" << camera.yfov << ","
                 << "\"znear\":" << camera.znear;
            if (camera.aspect > 0.0f) {
                json << ",\"aspectRatio\":" << camera.aspect;
            }
            if (camera.zfar > camera.znear) {
                json << ",\"zfar\":" << camera.zfar;
            }
            json << "}}";
        }
        json << "],";
    }

    json << "\"nodes\":[";
    for (size_t i = 0; i < ctx.nodes.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        const auto& node = ctx.nodes[i];
        json << "{"
             << "\"name\":\"" << JsonEscape(node.name) << "\"";
        if (node.mesh >= 0) {
            json << ",\"mesh\":" << node.mesh;
        }
        if (node.camera >= 0) {
            json << ",\"camera\":" << node.camera;
        }
        if (!node.children.empty()) {
            json << ",\"children\":[";
            for (size_t c = 0; c < node.children.size(); ++c) {
                if (c > 0) {
                    json << ",";
                }
                json << node.children[c];
            }
            json << "]";
        }
        json << ",\"translation\":" << ArrayToJson(node.translation)
             << ",\"rotation\":" << ArrayToJson(node.rotation)
             << ",\"scale\":" << ArrayToJson(node.scale)
             << "}";
    }
    json << "],";

    json << "\"scenes\":[{\"nodes\":[";
    for (size_t i = 0; i < ctx.rootNodes.size(); ++i) {
        if (i > 0) {
            json << ",";
        }
        json << ctx.rootNodes[i];
    }
    json << "]}],\"scene\":0}";
    return json.str();
}

bool WriteGlb(const fs::path& path,
              const std::string& jsonText,
              const std::vector<uint8_t>& binBytes,
              std::string& error) {
    std::vector<uint8_t> jsonChunk(jsonText.begin(), jsonText.end());
    while (jsonChunk.size() % 4 != 0) {
        jsonChunk.push_back(' ');
    }
    std::vector<uint8_t> binChunk = binBytes;
    while (binChunk.size() % 4 != 0) {
        binChunk.push_back(0);
    }

    auto appendU32 = [](std::vector<uint8_t>& bytes, uint32_t value) {
        bytes.push_back(static_cast<uint8_t>(value & 0xFFu));
        bytes.push_back(static_cast<uint8_t>((value >> 8) & 0xFFu));
        bytes.push_back(static_cast<uint8_t>((value >> 16) & 0xFFu));
        bytes.push_back(static_cast<uint8_t>((value >> 24) & 0xFFu));
    };

    std::vector<uint8_t> bytes;
    const uint32_t totalLength =
        12u + 8u + static_cast<uint32_t>(jsonChunk.size()) + 8u + static_cast<uint32_t>(binChunk.size());
    appendU32(bytes, 0x46546C67u);
    appendU32(bytes, 2u);
    appendU32(bytes, totalLength);
    appendU32(bytes, static_cast<uint32_t>(jsonChunk.size()));
    appendU32(bytes, 0x4E4F534Au);
    bytes.insert(bytes.end(), jsonChunk.begin(), jsonChunk.end());
    appendU32(bytes, static_cast<uint32_t>(binChunk.size()));
    appendU32(bytes, 0x004E4942u);
    bytes.insert(bytes.end(), binChunk.begin(), binChunk.end());
    return WriteFileBytes(path, bytes, error);
}

std::vector<OutputHashEntry> HashOutputs(const fs::path& directory) {
    std::vector<OutputHashEntry> outputs;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file()) {
            continue;
        }
        std::vector<uint8_t> bytes;
        std::string error;
        if (!ReadFileBytes(it->path(), bytes, error)) {
            continue;
        }
        outputs.push_back(OutputHashEntry{
            ReplaceBackslashes(fs::relative(it->path(), directory, ec).generic_string()),
            Sha256Hex(bytes),
        });
    }
    std::sort(outputs.begin(),
              outputs.end(),
              [](const OutputHashEntry& a, const OutputHashEntry& b) { return a.path < b.path; });
    return outputs;
}

bool WriteManifest(const ImportContext& ctx,
                   const fs::path& manifestPath,
                   const std::vector<OutputHashEntry>& outputs,
                   std::string& error) {
    std::ostringstream json;
    json << "{\n";
    json << "  \"schema_version\": \"1.0\",\n";
    json << "  \"tool_version\": \"" << JsonEscape(kSemanticVersion) << "\",\n";
    json << "  \"git_hash\": \"" << JsonEscape(kGitHash) << "\",\n";
    json << "  \"assimp_version\": \"" << aiGetVersionMajor() << "." << aiGetVersionMinor() << "."
         << aiGetVersionPatch() << "\",\n";
    {
        std::ostringstream revision;
        revision << std::hex << aiGetVersionRevision();
        json << "  \"assimp_revision\": \"" << revision.str() << "\",\n";
    }
    json << "  \"source_path\": \"" << JsonEscape(ctx.sourcePath.string()) << "\",\n";
    json << "  \"texcoord_v_origin\": \"top_left\",\n";
    if (ctx.objMaterialReport) {
        const ObjMaterialReport& report = *ctx.objMaterialReport;
        json << "  \"obj_material_check\": {\"faces\": " << report.faceCount
             << ", \"materials\": " << report.materialCount
             << ", \"mismatched_materials\": " << report.mismatchedMaterials
             << ", \"mtllib_directives\": " << report.mtllibDirectives
             << ", \"mtllib_hoisted\": " << (report.mtllibHoisted ? "true" : "false")
             << ", \"mtllib_libraries\": [";
        for (size_t i = 0; i < report.mtllibLibraries.size(); ++i) {
            json << (i > 0 ? ", " : "") << "\"" << JsonEscape(report.mtllibLibraries[i]) << "\"";
        }
        json << "]},\n";
    }
    json << "  \"mesh_count\": " << ctx.stats.meshCount << ",\n";
    json << "  \"triangle_count\": " << ctx.stats.triangleCount << ",\n";
    json << "  \"node_count\": " << ctx.stats.nodeCount << ",\n";
    json << "  \"support_matrix\": [\n";
    json << "    {\"area\":\"geometry\",\"status\":\"supported\",\"details\":\"Assimp meshes are triangulated into deterministic glTF primitive records\"},\n";
    json << "    {\"area\":\"transforms\",\"status\":\"supported\",\"details\":\"Assimp node transforms are preserved in the emitted glTF node hierarchy\"},\n";
    json << "    {\"area\":\"cameras\",\"status\":\"supported\",\"details\":\"Assimp perspective cameras are emitted as glTF camera records and attached to matching scene nodes\"},\n";
    json << "    {\"area\":\"textures\",\"status\":\"supported\",\"details\":\"copy, convert, embed, and link policies are explicit in the manifest texture table\"},\n";
    json << "    {\"area\":\"materials\",\"status\":\"partial\",\"details\":\"PBR-compatible scalar and texture slots are converted; full DCC material graphs are not preserved\"},\n";
    json << "    {\"area\":\"lights\",\"status\":\"unsupported\",\"details\":\"Assimp lights are diagnosed and not converted to renderer light primitives\"},\n";
    json << "    {\"area\":\"animation\",\"status\":\"unsupported\",\"details\":\"animation, skinning, and morph data are diagnosed and omitted from the deterministic static package\"},\n";
    json << "    {\"area\":\"alpha_transmission_emission\",\"status\":\"partial\",\"details\":\"alpha/emissive/material texture records are converted where Assimp exposes compatible data\"},\n";
    json << "    {\"area\":\"ktx2_mip_compression\",\"status\":\"supported\",\"details\":\"convert mode emits deterministic KTX2 outputs where supported and records copy fallbacks\"},\n";
    json << "    {\"area\":\"assimp_formats\",\"status\":\"supported\",\"details\":\"static Assimp-supported scene formats are accepted by PathTracerImport\"},\n";
    json << "    {\"area\":\"usd_materialx\",\"status\":\"out_of_scope\",\"details\":\"USD and MaterialX are not importer targets in this pipeline\"}\n";
    json << "  ],\n";
    json << "  \"texture_policy\": {\n";
    json << "    \"mode\": \"";
    switch (ctx.options.textureMode) {
        case ImportOptions::TextureMode::Copy: json << "copy"; break;
        case ImportOptions::TextureMode::Convert: json << "convert"; break;
        case ImportOptions::TextureMode::Embed: json << "embed"; break;
        case ImportOptions::TextureMode::Link: json << "link"; break;
    }
    json << "\",\n";
    json << "    \"residency_policy\": \"runtime SceneResources owns uploaded texture residency after loading emitted glTF/GLB\",\n";
    json << "    \"mip_policy\": \"converted KTX2/DDS source mip metadata is preserved when available; runtime loader generates mips for decoded raster inputs\",\n";
    json << "    \"compression_policy\": \"convert mode prefers deterministic KTX2 for supported inputs and records fall_back_to_copy otherwise\",\n";
    json << "    \"budget_policy\": \"runtime budget policy is selected by renderer CLI and reported in headless metadata\",\n";
    json << "    \"determinism_policy\": \"output file hashes, texture output hashes, and stable resource IDs are emitted in import_manifest.json\"\n";
    json << "  },\n";
    json << "  \"stable_resource_ids\": [\n";
    bool wroteStableId = false;
    auto writeStableId = [&](const char* resourceType,
                             const std::string& name,
                             int sourceIndex,
                             const std::string& stableId) {
        if (wroteStableId) {
            json << ",\n";
        }
        wroteStableId = true;
        json << "    {\"resource_type\":\"" << JsonEscape(resourceType)
             << "\", \"name\":\"" << JsonEscape(name)
             << "\", \"source_index\":" << sourceIndex
             << ", \"stable_id\":\"" << JsonEscape(stableId) << "\"}";
    };
    for (size_t i = 0; i < ctx.materials.size(); ++i) {
        writeStableId("material",
                      ctx.materials[i].name,
                      static_cast<int>(i),
                      StableImportId("material",
                                     ctx.sourcePath.string() + "|material|" + std::to_string(i) +
                                         "|" + ctx.materials[i].name));
    }
    for (size_t i = 0; i < ctx.meshes.size(); ++i) {
        writeStableId("geometry",
                      ctx.meshes[i].name,
                      static_cast<int>(i),
                      StableImportId("geometry",
                                     ctx.sourcePath.string() + "|geometry|" + std::to_string(i) +
                                         "|" + ctx.meshes[i].name));
    }
    for (size_t i = 0; i < ctx.manifestTextures.size(); ++i) {
        writeStableId("texture",
                      ctx.manifestTextures[i].outputPath,
                      static_cast<int>(i),
                      StableImportId("texture",
                                     ctx.manifestTextures[i].sourcePath + "|" +
                                         ctx.manifestTextures[i].outputPath + "|" +
                                         ctx.manifestTextures[i].outputSha256));
    }
    json << "\n  ],\n";

    json << "  \"texture_inventory\": [\n";
    for (size_t i = 0; i < ctx.textureInventory.size(); ++i) {
        const TextureInventoryEntry& entry = ctx.textureInventory[i];
        json << "    {\"path\": \"" << JsonEscape(entry.path) << "\", "
             << "\"width\": " << entry.width << ", "
             << "\"height\": " << entry.height << ", "
             << "\"estimated_bytes\": " << entry.estimatedBytes << "}";
        if (i + 1 < ctx.textureInventory.size()) {
            json << ",";
        }
        json << "\n";
    }
    json << "  ]";

    if (!ctx.manifestTextures.empty()) {
        json << ",\n  \"textures\": [\n";
        for (size_t i = 0; i < ctx.manifestTextures.size(); ++i) {
            const ManifestTextureEntry& entry = ctx.manifestTextures[i];
            json << "    {"
                 << "\"source_path\": \"" << JsonEscape(entry.sourcePath) << "\", "
                 << "\"source_format\": \"" << JsonEscape(entry.sourceFormat) << "\", "
                 << "\"output_path\": \"" << JsonEscape(entry.outputPath) << "\", "
                 << "\"output_format\": \"" << JsonEscape(entry.outputFormat) << "\", "
                 << "\"output_sha256\": \"" << JsonEscape(entry.outputSha256) << "\", "
                 << "\"transcode_tool\": \"" << JsonEscape(entry.transcodeTool) << "\", "
                 << "\"transcode_version\": \"" << JsonEscape(entry.transcodeVersion) << "\", "
                 << "\"fell_back_to_copy\": " << (entry.fellBackToCopy ? "true" : "false")
                 << "}";
            if (i + 1 < ctx.manifestTextures.size()) {
                json << ",";
            }
            json << "\n";
        }
        json << "  ]";
    }

    if (!ctx.warnings.empty()) {
        json << ",\n  \"warnings\": [\n";
        for (size_t i = 0; i < ctx.warnings.size(); ++i) {
            json << "    \"" << JsonEscape(ctx.warnings[i]) << "\"";
            if (i + 1 < ctx.warnings.size()) {
                json << ",";
            }
            json << "\n";
        }
        json << "  ]";
    }

    json << ",\n  \"output_files\": [\n";
    for (size_t i = 0; i < outputs.size(); ++i) {
        const OutputHashEntry& output = outputs[i];
        json << "    {\"path\": \"" << JsonEscape(output.path) << "\", "
             << "\"sha256\": \"" << JsonEscape(output.sha256) << "\"}";
        if (i + 1 < outputs.size()) {
            json << ",";
        }
        json << "\n";
    }
    json << "  ]\n";
    json << "}\n";
    return WriteTextFile(manifestPath, json.str(), error);
}

bool WriteCanonicalManifest(const ImportContext& ctx, std::string& error) {
    if (ctx.canonicalManifestEntries.empty()) {
        return true;
    }

    const fs::path reportDir = fs::absolute("validation/reports").lexically_normal();
    std::error_code ec;
    fs::create_directories(reportDir, ec);
    if (ec) {
        error = "Failed to create canonical report directory: " + reportDir.string();
        return false;
    }

    const fs::path manifestPath = reportDir / (ctx.sceneBaseName + "_canonical_import_manifest.json");
    std::ostringstream json;
    json << "{\n";
    json << "  \"scene\": \"" << JsonEscape(ctx.sceneBaseName) << "\",\n";
    json << "  \"import_mode\": \"canonical\",\n";
    json << "  \"source_path\": \"" << JsonEscape(ctx.sourcePath.string()) << "\",\n";
    json << "  \"materials\": [\n";
    for (size_t i = 0; i < ctx.canonicalManifestEntries.size(); ++i) {
        const CanonicalManifestEntry& entry = ctx.canonicalManifestEntries[i];
        json << "    {\n";
        json << "      \"material\": \"" << JsonEscape(entry.materialName) << "\",\n";
        json << "      \"source_diffuse\": \"" << JsonEscape(entry.sourceDiffuse) << "\",\n";
        json << "      \"source_normal\": \"" << JsonEscape(entry.sourceNormal) << "\",\n";
        json << "      \"source_normal_channels\": " << entry.sourceNormalChannels << ",\n";
        json << "      \"normal_alpha_min\": " << std::fixed << std::setprecision(6) << entry.normalAlphaMin << ",\n";
        json << "      \"normal_alpha_max\": " << std::fixed << std::setprecision(6) << entry.normalAlphaMax << ",\n";
        json << "      \"normal_alpha_std\": " << std::fixed << std::setprecision(6) << entry.normalAlphaStd << ",\n";
        json << "      \"alpha_interpretation\": \"" << JsonEscape(entry.alphaInterpretation) << "\",\n";
        json << "      \"source_specular\": \"" << JsonEscape(entry.sourceSpecular) << "\",\n";
        json << "      \"specular_classification\": \"" << JsonEscape(entry.specularClassification) << "\",\n";
        json << "      \"specular_action\": \"" << JsonEscape(entry.specularAction) << "\",\n";
        json << "      \"roughness_source\": \"" << JsonEscape(entry.roughnessSource) << "\",\n";
        json << "      \"roughness_target\": \"" << JsonEscape(entry.roughnessValueOrTarget) << "\",\n";
        json << "      \"metallic_source\": \"" << JsonEscape(entry.metallicSource) << "\",\n";
        json << "      \"normal_y_flipped\": " << (entry.normalYFlipped ? "true" : "false") << ",\n";
        json << "      \"emissive_source\": \"" << JsonEscape(entry.emissiveSource) << "\",\n";
        json << "      \"unresolved\": [";
        for (size_t u = 0; u < entry.unresolved.size(); ++u) {
            json << "\"" << JsonEscape(entry.unresolved[u]) << "\"";
            if (u + 1 < entry.unresolved.size()) {
                json << ", ";
            }
        }
        json << "]\n";
        json << "    }";
        if (i + 1 < ctx.canonicalManifestEntries.size()) {
            json << ",";
        }
        json << "\n";
    }
    json << "  ]\n";
    json << "}\n";
    return WriteTextFile(manifestPath, json.str(), error);
}

bool WriteZeroDayManifest(const ImportContext& ctx, std::string& error) {
    if (ctx.zeroDayManifestEntries.empty()) {
        return true;
    }

    const fs::path manifestPath = ctx.outputDirectory /
                                  (IsSunTempleImport(ctx)
                                       ? "sun_temple_material_manifest.json"
                                       : (IsEmeraldSquareImport(ctx)
                                              ? "emerald_square_material_manifest.json"
                                              : "zero_day_material_manifest.json"));
    std::ostringstream json;
    json << "{\n";
    json << "  \"schema_version\": \"1.0\",\n";
    json << "  \"asset_family\": \"" << JsonEscape(PackedOrcaImportLabel(ctx)) << "\",\n";
    json << "  \"scene\": \"" << JsonEscape(ctx.sceneBaseName) << "\",\n";
    json << "  \"source_path\": \"" << JsonEscape(ctx.sourcePath.string()) << "\",\n";
    json << "  \"material_convention\": {\n";
    json << "    \"base_color\": \"RGB base color, alpha opacity\",\n";
    json << "    \"specular\": \""
         << (IsEmeraldSquareImport(ctx)
                 ? "R source occlusion (constant-zero channel unused; no AO binding), G roughness, B metalness"
                 : "R ambient occlusion, G roughness, B metalness")
         << "\",\n";
    json << "    \"normal\": \"DirectX source normal, green preserved with reflected V/bitangent frame\",\n";
    json << "    \"emissive\": \""
         << (IsEmeraldSquareImport(ctx)
                 ? "RGB emissive texture replaces FBX fallback color; glTF factor 1 preserves Falcor's separate scene multiplier"
                 : "RGB emissive color")
         << "\"\n";
    json << "  },\n";
    json << "  \"materials\": [\n";
    for (size_t i = 0; i < ctx.zeroDayManifestEntries.size(); ++i) {
        const ZeroDayManifestEntry& entry = ctx.zeroDayManifestEntries[i];
        json << "    {\n";
        json << "      \"material\": \"" << JsonEscape(entry.materialName) << "\",\n";
        json << "      \"source_base_color\": \"" << JsonEscape(entry.sourceBaseColor) << "\",\n";
        json << "      \"base_color_alpha_used\": " << (entry.baseColorAlphaUsed ? "true" : "false") << ",\n";
        json << "      \"source_normal\": \"" << JsonEscape(entry.sourceNormal) << "\",\n";
        json << "      \"normal_y_flipped\": " << (entry.normalYFlipped ? "true" : "false") << ",\n";
        json << "      \"source_specular\": \"" << JsonEscape(entry.sourceSpecular) << "\",\n";
        json << "      \"specular_action\": \"" << JsonEscape(entry.specularAction) << "\",\n";
        json << "      \"orm_target\": \"" << JsonEscape(entry.ormTarget) << "\",\n";
        json << "      \"source_emissive\": \"" << JsonEscape(entry.sourceEmissive) << "\",\n";
        json << "      \"unresolved\": [";
        for (size_t u = 0; u < entry.unresolved.size(); ++u) {
            json << "\"" << JsonEscape(entry.unresolved[u]) << "\"";
            if (u + 1 < entry.unresolved.size()) {
                json << ", ";
            }
        }
        json << "]\n";
        json << "    }";
        if (i + 1 < ctx.zeroDayManifestEntries.size()) {
            json << ",";
        }
        json << "\n";
    }
    json << "  ]\n";
    json << "}\n";
    return WriteTextFile(manifestPath, json.str(), error);
}

// ---------------------------------------------------------------------------------------------
// OBJ material libraries
//
// Assimp's OBJ importer (same logic in 5.4.3 through 6.0.6 and master, checked 2026-11) mis-assigns
// materials when an `mtllib` directive is read after a mesh exists. Re-reading a known library leaves the
// parser's current material on that library's last `newmtl`, so a following `usemtl` naming it is dropped
// as a repeat and the mesh keeps the previous object's material. Loading a new library relabels the mesh
// being filled. living_room.obj repeats `mtllib` before every object and lost WhitePaint that way (it
// became the emissive BackLight).
//
// For OBJ sources the importer therefore (1) serves Assimp the file with each distinct `mtllib` once,
// ahead of all geometry, and the original directives turned into comments, and (2) checks Assimp's
// per-material face counts against an independent scan of the file before post-processing. The scan
// mirrors Assimp's line handling: every '\n', '\r', '\f' or '\0' ends a line, '\\' before a line end
// joins the next line, keywords are read at column 0, `cstype` ... `end` blocks are skipped, and faces
// without indices are ignored.
// ---------------------------------------------------------------------------------------------

bool IsObjLineEnd(char c) {
    return c == '\n' || c == '\r' || c == '\f' || c == '\0';
}

bool IsObjBlank(char c) {
    return c == ' ' || c == '\t';
}

// Splits a file into logical lines the way Assimp's IOStreamBuffer::getNextDataLine does.
class ObjLineReader {
public:
    explicit ObjLineReader(std::FILE* file) : m_file(file), m_buffer(size_t(1) << 22) {}

    // Returns false at end of file. `offset` is the file offset of the line's first byte.
    bool next(std::string& line, uint64_t& offset) {
        line.clear();
        if (m_pos == m_end && !fill()) {
            return false;
        }
        offset = m_bufferOffset + m_pos;
        for (;;) {
            if (m_pos == m_end && !fill()) {
                return true;
            }
            const char c = m_buffer[m_pos];
            if (c == '\\') {
                if (m_pos + 1 == m_end && !fill()) {
                    line.push_back(c);
                    ++m_pos;
                    return true;
                }
                if (IsObjLineEnd(m_buffer[m_pos + 1])) {
                    // Continuation: skip to just after the next '\n' and keep reading the same line.
                    ++m_pos;
                    for (;;) {
                        if (m_pos == m_end && !fill()) {
                            return true;
                        }
                        if (m_buffer[m_pos++] == '\n') {
                            break;
                        }
                    }
                    continue;
                }
            }
            ++m_pos;
            if (IsObjLineEnd(c)) {
                return true;
            }
            line.push_back(c);
        }
    }

private:
    // Keeps unread bytes and appends more from the file; returns false when nothing new was read.
    bool fill() {
        const size_t remaining = m_end - m_pos;
        if (remaining > 0 && m_pos > 0) {
            std::memmove(m_buffer.data(), m_buffer.data() + m_pos, remaining);
        }
        m_bufferOffset += m_pos;
        m_pos = 0;
        m_end = remaining;
        const size_t read = std::fread(m_buffer.data() + m_end, 1, m_buffer.size() - m_end, m_file);
        m_end += read;
        return read > 0;
    }

    std::FILE* m_file = nullptr;
    std::vector<char> m_buffer;
    size_t m_pos = 0;
    size_t m_end = 0;
    uint64_t m_bufferOffset = 0;
};

// First token of an OBJ line (up to a blank), as Assimp's getNameNoSpace reads keywords.
std::string_view ObjKeyword(std::string_view line) {
    size_t end = 0;
    while (end < line.size() && !IsObjBlank(line[end])) {
        ++end;
    }
    return line.substr(0, end);
}

// Text after the keyword with leading blanks skipped, as Assimp's getNextToken leaves it.
std::string_view ObjArgument(std::string_view line) {
    size_t pos = ObjKeyword(line).size();
    while (pos < line.size() && IsObjBlank(line[pos])) {
        ++pos;
    }
    return line.substr(pos);
}

std::string_view TrimObjText(std::string_view text) {
    const auto isSpace = [](char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; };
    while (!text.empty() && isSpace(text.front())) {
        text.remove_prefix(1);
    }
    while (!text.empty() && isSpace(text.back())) {
        text.remove_suffix(1);
    }
    return text;
}

// Number of vertex references on an f/l/p line (one per blank-separated group that starts like an index).
uint64_t CountObjFaceVertices(std::string_view argument) {
    uint64_t count = 0;
    size_t pos = 0;
    while (pos < argument.size()) {
        while (pos < argument.size() && IsObjBlank(argument[pos])) {
            ++pos;
        }
        if (pos >= argument.size()) {
            break;
        }
        const char c = argument[pos];
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+') {
            ++count;
        }
        while (pos < argument.size() && !IsObjBlank(argument[pos])) {
            ++pos;
        }
    }
    return count;
}

struct ObjMaterialScan {
    std::vector<std::string> libraries;    // distinct `mtllib` arguments, first-appearance order
    std::vector<uint64_t> mtllibOffsets;  // file offset of the 'm' of every `mtllib` directive
    uint64_t mtllibDirectives = 0;
    bool libraryAfterMeshDirective = false;  // an `mtllib` follows o/g/usemtl/f/l/p
    uint64_t bomBytes = 0;
    uint64_t fileSize = 0;
    std::map<std::string, uint64_t> facesByUsemtl;
    uint64_t facesBeforeUsemtl = 0;

    bool needsHoist() const {
        return mtllibDirectives > libraries.size() || libraryAfterMeshDirective;
    }
};

bool ScanObjMaterials(const fs::path& path, ObjMaterialScan& scan, std::string& error) {
    std::error_code ec;
    scan.fileSize = fs::file_size(path, ec);
    if (ec) {
        error = "Failed to stat OBJ source: " + path.string();
        return false;
    }
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        error = "Failed to open OBJ source: " + path.string();
        return false;
    }

    ObjLineReader reader(file);
    std::string line;
    uint64_t offset = 0;
    bool firstLine = true;
    bool insideCstype = false;
    bool meshDirectiveSeen = false;
    std::optional<std::string> currentUsemtl;
    while (reader.next(line, offset)) {
        std::string_view view(line);
        if (firstLine) {
            firstLine = false;
            if (view.size() >= 3 && static_cast<unsigned char>(view[0]) == 0xEF &&
                static_cast<unsigned char>(view[1]) == 0xBB && static_cast<unsigned char>(view[2]) == 0xBF) {
                view.remove_prefix(3);
                offset += 3;
                scan.bomBytes = 3;
            }
        }
        if (view.empty()) {
            continue;
        }
        if (insideCstype) {
            if (view[0] == 'e' && ObjKeyword(view) == "end") {
                insideCstype = false;
            }
            continue;
        }
        switch (view[0]) {
            case 'f':
            case 'l':
            case 'p': {
                const uint64_t vertices = CountObjFaceVertices(ObjArgument(view));
                if (vertices == 0) {
                    break;  // Assimp ignores faces without vertices.
                }
                // Assimp stores a polyline of n vertices as n-1 line faces and n points as n faces.
                const uint64_t faces = view[0] == 'f' ? 1 : (view[0] == 'l' ? vertices - 1 : vertices);
                if (currentUsemtl) {
                    scan.facesByUsemtl[*currentUsemtl] += faces;
                } else {
                    scan.facesBeforeUsemtl += faces;
                }
                meshDirectiveSeen = true;
                break;
            }
            case 'u':
                if (ObjKeyword(view) == "usemtl") {
                    const std::string_view name = TrimObjText(ObjArgument(view));
                    if (!name.empty()) {
                        currentUsemtl = std::string(name);
                    }
                    meshDirectiveSeen = true;
                }
                break;
            case 'm':
                if (ObjKeyword(view) == "mtllib") {
                    const std::string_view library = ObjArgument(view);
                    if (library.empty()) {
                        break;  // Assimp warns and loads nothing.
                    }
                    ++scan.mtllibDirectives;
                    scan.mtllibOffsets.push_back(offset);
                    if (meshDirectiveSeen) {
                        scan.libraryAfterMeshDirective = true;
                    }
                    if (std::find(scan.libraries.begin(), scan.libraries.end(), library) == scan.libraries.end()) {
                        scan.libraries.emplace_back(library);
                    }
                }
                break;
            case 'o':
            case 'g':
                meshDirectiveSeen = true;
                break;
            case 'c':
                insideCstype = ObjKeyword(view) == "cstype";
                break;
            default:
                break;
        }
    }
    const bool readError = std::ferror(file) != 0;
    std::fclose(file);
    if (readError) {
        error = "Failed to read OBJ source: " + path.string();
        return false;
    }
    return true;
}

// Name of the last `newmtl` in an MTL file, read the way Assimp's ObjFileMtlImporter does (leading blanks
// skipped, a line starting with "ne" is `newmtl`, a missing name means AI_DEFAULT_MATERIAL_NAME).
std::optional<std::string> LastMtlMaterialName(const fs::path& path) {
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        return std::nullopt;
    }
    ObjLineReader reader(file);
    std::string line;
    uint64_t offset = 0;
    std::optional<std::string> last;
    while (reader.next(line, offset)) {
        std::string_view view(line);
        if (view.size() >= 3 && static_cast<unsigned char>(view[0]) == 0xEF &&
            static_cast<unsigned char>(view[1]) == 0xBB && static_cast<unsigned char>(view[2]) == 0xBF) {
            view.remove_prefix(3);
        }
        while (!view.empty() && IsObjBlank(view.front())) {
            view.remove_prefix(1);
        }
        if (view.size() < 2 || view[0] != 'n' || view[1] != 'e') {
            continue;
        }
        const std::string_view name = TrimObjText(ObjArgument(view));
        last = name.empty() ? std::string(AI_DEFAULT_MATERIAL_NAME) : std::string(name);
    }
    std::fclose(file);
    return last;
}

// The material Assimp gives faces that precede every `usemtl`: the last `newmtl` of the last library that
// loaded (libraries resolve against the OBJ's folder, then fall back to <obj>.mtl), else the default.
std::string ObjMaterialBeforeUsemtl(const fs::path& objPath, const ObjMaterialScan& scan) {
    std::string material = AI_DEFAULT_MATERIAL_NAME;
    const std::string folder = objPath.parent_path().string();
    const std::string objString = objPath.string();
    const fs::path fallback = objString.substr(0, objString.size() - 3) + "mtl";
    for (const std::string& library : scan.libraries) {
        std::string slashed = library;
        std::replace(slashed.begin(), slashed.end(), '\\', '/');
        std::optional<std::string> last = LastMtlMaterialName(fs::path(folder + "/" + library));
        if (!last && slashed != library) {
            last = LastMtlMaterialName(fs::path(folder + "/" + slashed));
        }
        if (!last) {
            last = LastMtlMaterialName(fallback);
        }
        if (last) {
            material = *last;
        }
    }
    return material;
}

// Serves an OBJ as one `mtllib` line per distinct library, then the original bytes (minus a UTF-8 BOM)
// with the first byte of every original `mtllib` directive replaced by '#'. Offsets stay put and the
// directives become comments.
class ObjHoistedMtllibStream final : public Assimp::IOStream {
public:
    ObjHoistedMtllibStream(std::FILE* file,
                           std::string prefix,
                           uint64_t skipBytes,
                           uint64_t fileSize,
                           std::vector<uint64_t> commentOffsets)
        : m_file(file),
          m_prefix(std::move(prefix)),
          m_skipBytes(skipBytes),
          m_total(m_prefix.size() + (fileSize - skipBytes)),
          m_commentOffsets(std::move(commentOffsets)) {}

    ~ObjHoistedMtllibStream() override {
        if (m_file != nullptr) {
            std::fclose(m_file);
        }
    }

    size_t Read(void* buffer, size_t size, size_t count) override {
        if (size == 0 || count == 0 || m_position >= m_total) {
            return 0;
        }
        uint64_t bytes = std::min<uint64_t>(static_cast<uint64_t>(size) * count, m_total - m_position);
        bytes -= bytes % size;
        char* out = static_cast<char*>(buffer);
        uint64_t done = 0;
        if (m_position < m_prefix.size()) {
            const uint64_t fromPrefix = std::min<uint64_t>(bytes, m_prefix.size() - m_position);
            std::memcpy(out, m_prefix.data() + m_position, static_cast<size_t>(fromPrefix));
            done = fromPrefix;
        }
        if (done < bytes) {
            const uint64_t fileOffset = m_position + done - m_prefix.size() + m_skipBytes;
            if (fseeko(m_file, static_cast<off_t>(fileOffset), SEEK_SET) == 0) {
                const size_t read = std::fread(out + done, 1, static_cast<size_t>(bytes - done), m_file);
                for (auto it = std::lower_bound(m_commentOffsets.begin(), m_commentOffsets.end(), fileOffset);
                     it != m_commentOffsets.end() && *it < fileOffset + read;
                     ++it) {
                    out[done + (*it - fileOffset)] = '#';
                }
                done += read;
            }
        }
        m_position += done;
        return static_cast<size_t>(done / size);
    }

    size_t Write(const void*, size_t, size_t) override { return 0; }

    aiReturn Seek(size_t offset, aiOrigin origin) override {
        int64_t base = 0;
        switch (origin) {
            case aiOrigin_SET:
                base = 0;
                break;
            case aiOrigin_CUR:
                base = static_cast<int64_t>(m_position);
                break;
            case aiOrigin_END:
                base = static_cast<int64_t>(m_total);
                break;
            default:
                return aiReturn_FAILURE;
        }
        const int64_t target = base + static_cast<int64_t>(offset);
        if (target < 0 || static_cast<uint64_t>(target) > m_total) {
            return aiReturn_FAILURE;
        }
        m_position = static_cast<uint64_t>(target);
        return aiReturn_SUCCESS;
    }

    size_t Tell() const override { return static_cast<size_t>(m_position); }
    size_t FileSize() const override { return static_cast<size_t>(m_total); }
    void Flush() override {}

private:
    std::FILE* m_file = nullptr;
    std::string m_prefix;
    uint64_t m_skipBytes = 0;
    uint64_t m_total = 0;
    uint64_t m_position = 0;
    std::vector<uint64_t> m_commentOffsets;
};

// Opens the source OBJ through ObjHoistedMtllibStream and every other path (the MTL files) as usual.
class ObjHoistedMtllibIOSystem final : public Assimp::IOSystem {
public:
    ObjHoistedMtllibIOSystem(const fs::path& source, const ObjMaterialScan& scan)
        : m_source(fs::absolute(source).lexically_normal()),
          m_skipBytes(scan.bomBytes),
          m_fileSize(scan.fileSize),
          m_commentOffsets(scan.mtllibOffsets) {
        for (const std::string& library : scan.libraries) {
            m_prefix += "mtllib " + library + "\n";
        }
        std::sort(m_commentOffsets.begin(), m_commentOffsets.end());
    }

    bool Exists(const char* file) const override { return m_default.Exists(file); }
    char getOsSeparator() const override { return m_default.getOsSeparator(); }
    bool ComparePaths(const char* one, const char* second) const override {
        return m_default.ComparePaths(one, second);
    }

    Assimp::IOStream* Open(const char* file, const char* mode = "rb") override {
        if (file != nullptr && IsSource(file) && (mode == nullptr || std::strchr(mode, 'w') == nullptr)) {
            std::FILE* handle = std::fopen(m_source.c_str(), "rb");
            if (handle == nullptr) {
                return nullptr;
            }
            return new ObjHoistedMtllibStream(handle, m_prefix, m_skipBytes, m_fileSize, m_commentOffsets);
        }
        return m_default.Open(file, mode);
    }

    void Close(Assimp::IOStream* stream) override { delete stream; }

private:
    bool IsSource(const char* file) const {
        std::error_code ec;
        const fs::path candidate = fs::absolute(fs::path(file), ec).lexically_normal();
        return !ec && candidate == m_source;
    }

    Assimp::DefaultIOSystem m_default;
    fs::path m_source;
    std::string m_prefix;
    uint64_t m_skipBytes = 0;
    uint64_t m_fileSize = 0;
    std::vector<uint64_t> m_commentOffsets;
};

// Compares Assimp's per-material face counts (raw import, before post-processing) with the scan.
// Returns an empty string when they agree, else a description of the differences.
std::string CheckObjMaterialAssignment(const aiScene* scene,
                                       const fs::path& objPath,
                                       const ObjMaterialScan& scan,
                                       ObjMaterialReport& report) {
    std::map<std::string, uint64_t> expected = scan.facesByUsemtl;
    if (scan.facesBeforeUsemtl > 0) {
        expected[ObjMaterialBeforeUsemtl(objPath, scan)] += scan.facesBeforeUsemtl;
    }
    std::map<std::string, uint64_t> actual;
    for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
        const aiMesh* mesh = scene->mMeshes[i];
        aiString name;
        if (mesh->mMaterialIndex < scene->mNumMaterials) {
            scene->mMaterials[mesh->mMaterialIndex]->Get(AI_MATKEY_NAME, name);
        }
        actual[name.C_Str()] += mesh->mNumFaces;
    }

    report.faceCount = 0;
    for (const auto& [name, faces] : expected) {
        report.faceCount += faces;
    }
    report.materialCount = expected.size();

    std::vector<std::string> differences;
    std::map<std::string, uint64_t> all = expected;
    all.insert(actual.begin(), actual.end());
    for (const auto& entry : all) {
        const auto expectedIt = expected.find(entry.first);
        const auto actualIt = actual.find(entry.first);
        const uint64_t want = expectedIt == expected.end() ? 0 : expectedIt->second;
        const uint64_t got = actualIt == actual.end() ? 0 : actualIt->second;
        if (want != got) {
            differences.push_back(entry.first + " expected " + std::to_string(want) + " got " + std::to_string(got));
        }
    }
    report.mismatchedMaterials = differences.size();
    if (differences.empty()) {
        return {};
    }
    std::ostringstream message;
    message << "OBJ material check failed for " << objPath.filename().string()
            << ": Assimp's per-material face counts differ from the file's usemtl directives in "
            << differences.size() << " of " << all.size() << " materials (";
    for (size_t i = 0; i < differences.size() && i < 8; ++i) {
        message << (i > 0 ? "; " : "") << differences[i];
    }
    if (differences.size() > 8) {
        message << "; ...";
    }
    message << ")";
    return message.str();
}

ImportResult RunImportPipelineImpl(const ImportOptions& options) {
    ImportResult result;
    ImportContext ctx;
    ctx.options = options;
    ctx.sourcePath = fs::absolute(options.inputPath).lexically_normal();
    ctx.outputDirectory = fs::absolute(options.outputDirectory).lexically_normal();
    ctx.sceneBaseName = ctx.outputDirectory.filename().empty()
                            ? SanitizeName(ctx.sourcePath.stem().string())
                            : SanitizeName(ctx.outputDirectory.filename().string());

    Assimp::Importer importer;
    const unsigned int postprocess =
        aiProcess_Triangulate |
        aiProcess_JoinIdenticalVertices |
        aiProcess_ImproveCacheLocality |
        aiProcess_SortByPType |
        aiProcess_FindInvalidData |
        aiProcess_GenSmoothNormals |
        aiProcess_CalcTangentSpace |
        aiProcess_ValidateDataStructure;

    const bool isObjSource = ToLowerAscii(ctx.sourcePath.extension().string()) == ".obj";
    if (isObjSource) {
        ObjMaterialScan objScan;
        std::string scanError;
        if (!ScanObjMaterials(ctx.sourcePath, objScan, scanError)) {
            result.message = scanError;
            return result;
        }
        ObjMaterialReport report;
        report.mtllibDirectives = objScan.mtllibDirectives;
        report.mtllibLibraries = objScan.libraries;
        if (objScan.needsHoist()) {
            if (options.objHoistMaterialLibraries) {
                importer.SetIOHandler(new ObjHoistedMtllibIOSystem(ctx.sourcePath, objScan));
                report.mtllibHoisted = true;
                std::cout << "[PathTracerImport] OBJ: serving " << objScan.libraries.size()
                          << " material librar" << (objScan.libraries.size() == 1 ? "y" : "ies")
                          << " ahead of geometry (" << objScan.mtllibDirectives
                          << " mtllib directives in the file)\n";
            } else {
                ctx.warnings.push_back("OBJ mtllib directives repeat or follow geometry and were not hoisted "
                                       "(--no-obj-mtllib-hoist)");
            }
        }

        // Raw import first so the material check sees Assimp's faces before post-processing.
        // Equivalent to ReadFile(path, postprocess): ReadFile validates, preprocesses and then applies
        // the remaining steps through ApplyPostProcessing.
        ctx.scene = importer.ReadFile(ctx.sourcePath.string(), aiProcess_ValidateDataStructure);
        if (ctx.scene) {
            const std::string mismatch = CheckObjMaterialAssignment(ctx.scene, ctx.sourcePath, objScan, report);
            if (!mismatch.empty()) {
                if (!options.allowObjMaterialMismatch) {
                    result.message = mismatch + ". Pass --allow-obj-material-mismatch to import anyway.";
                    return result;
                }
                ctx.warnings.push_back(mismatch);
            }
            ctx.scene = importer.ApplyPostProcessing(postprocess & ~aiProcess_ValidateDataStructure);
        }
        ctx.objMaterialReport = report;
    } else {
        ctx.scene = importer.ReadFile(ctx.sourcePath.string(), postprocess);
    }
    if (!ctx.scene) {
        result.message = std::string("Assimp import failed: ") + importer.GetErrorString();
        return result;
    }
    if (!ctx.scene->mRootNode) {
        result.message = "Imported scene has no root node.";
        return result;
    }
    if (ctx.scene->mNumLights > 0u) {
        ctx.warnings.push_back("Unsupported Assimp lights omitted: lights=" +
                               std::to_string(ctx.scene->mNumLights));
    }
    if (ctx.scene->mNumAnimations > 0u) {
        ctx.warnings.push_back("Unsupported Assimp animations omitted: animations=" +
                               std::to_string(ctx.scene->mNumAnimations));
    }

    std::error_code ec;
    fs::create_directories(ctx.outputDirectory, ec);
    if (ec) {
        result.message = "Failed to create output directory: " + ctx.outputDirectory.string();
        return result;
    }
    if (options.textureMode == ImportOptions::TextureMode::Copy ||
        options.textureMode == ImportOptions::TextureMode::Convert) {
        fs::create_directories(ctx.outputDirectory / "textures", ec);
        if (ec) {
            result.message = "Failed to create textures directory.";
            return result;
        }
    }

    ctx.stats.nodeCount = static_cast<uint32_t>(CountNodes(ctx.scene->mRootNode));
    ConvertCameras(ctx);
    ConvertMaterials(ctx);
    ConvertMeshes(ctx);
    if (ToLowerAscii(ctx.sourcePath.extension().string()) == ".fbx") {
        std::string fbxError;
        ctx.fbxVisibilityRead = ReadFbxModelVisibility(ctx.sourcePath, ctx, fbxError);
        if (!ctx.fbxVisibilityRead) {
            ctx.warnings.push_back("FBX Model visibility not read (" + fbxError + "); hidden Models are imported");
            std::cerr << "[Import] FBX visibility: not read (" << fbxError << "); hidden Models are imported" << std::endl;
        } else {
            for (const auto& [id, model] : ctx.fbxModels) {
                ctx.fbxHiddenModelCount += FbxModelHidden(ctx, id) ? 1u : 0u;
            }
        }
    }
    if (ctx.fbxVisibilityRead) {
        ApplyC4dPolygonSelectionRule(ctx);
    }
    ctx.rootNodes.push_back(ConvertNode(ctx, ctx.scene->mRootNode));
    if (ctx.fbxVisibilityRead) {
        std::cout << "[Import] FBX visibility: " << ctx.fbxModels.size() << " Models, " << ctx.fbxHiddenModelCount
                  << " hidden (own Visibility 0 or a hidden parent); skipped the meshes of " << ctx.fbxHiddenNodesSkipped
                  << " nodes (" << ctx.fbxHiddenTrianglesSkipped << " triangles)";
        if (!ctx.fbxAmbiguousNodeNames.empty()) {
            std::cout << "; " << ctx.fbxAmbiguousNodeNames.size()
                      << " nodes kept visible: their Model name is shared by hidden and visible Models and the parent does not decide (first: "
                      << ctx.fbxAmbiguousNodeNames.front() << ")";
            ctx.warnings.push_back(std::to_string(ctx.fbxAmbiguousNodeNames.size()) +
                                   " FBX nodes with an ambiguous Model name were kept visible");
        }
        std::cout << std::endl;
    }

    const std::string sceneFileName =
        ctx.sceneBaseName + (options.outputFormat == ImportOptions::OutputFormat::Glb ? ".glb" : ".gltf");
    const std::string binFileName = ctx.sceneBaseName + ".bin";
    const fs::path scenePath = ctx.outputDirectory / sceneFileName;
    const std::string gltfJson = BuildGltfJson(ctx, binFileName);

    std::string error;
    if (options.outputFormat == ImportOptions::OutputFormat::Glb) {
        if (!WriteGlb(scenePath, gltfJson, ctx.buffer.bytes(), error)) {
            result.message = error;
            return result;
        }
    } else {
        if (!WriteTextFile(scenePath, gltfJson + "\n", error)) {
            result.message = error;
            return result;
        }
        if (!WriteFileBytes(ctx.outputDirectory / binFileName, ctx.buffer.bytes(), error)) {
            result.message = error;
            return result;
        }
    }

    if (UsePackedOrcaMaterialImport(ctx) && !WriteZeroDayManifest(ctx, error)) {
        result.message = error;
        return result;
    }

    std::vector<OutputHashEntry> outputs = HashOutputs(ctx.outputDirectory);
    if (!WriteManifest(ctx, ctx.outputDirectory / "import_manifest.json", outputs, error)) {
        result.message = error;
        return result;
    }
    if (UseCanonicalImport(ctx) && IsBistroImport(ctx) && !WriteCanonicalManifest(ctx, error)) {
        result.message = error;
        return result;
    }

    std::ostringstream message;
    message << "Import completed: meshes=" << ctx.stats.meshCount
            << " triangles=" << ctx.stats.triangleCount
            << " nodes=" << ctx.stats.nodeCount
            << " textures=" << ctx.textureInventory.size()
            << " orm=" << ctx.ormTextureCount;
    if (!ctx.warnings.empty()) {
        message << " warnings=" << ctx.warnings.size();
    }

    result.success = true;
    result.message = message.str();
    result.scenePath = scenePath.string();
    return result;
}

#endif  // PATH_TRACER_ENABLE_ASSIMP

}  // namespace

bool ImportBackendAvailable() {
#if PATH_TRACER_ENABLE_ASSIMP
    return true;
#else
    return false;
#endif
}

std::string PathTracerImportUsage() {
    std::ostringstream usage;
    usage << "PathTracerImport\n"
          << "Usage:\n"
          << "  PathTracerImport --in <file> --out <directory> "
             "[--format glb|gltf] [--textures copy|convert|embed|link] [--import-mode generic|canonical]\n"
          << "  PathTracerImport --input <file> --output <directory> "
             "[--format glb|gltf] [--textures copy|convert|embed|link] [--import-mode generic|canonical]\n\n"
          << "Flags:\n"
          << "  --in, --input <file>    Source scene path (.fbx, .obj, and other Assimp-supported static formats)\n"
          << "  --out, --output <dir>   Output scene directory\n"
          << "  --format <fmt>    glb or gltf (default glb)\n"
          << "  --textures <mode> copy, convert, embed, or link (default copy)\n"
          << "  --import-mode <mode>    generic or canonical (default generic)\n"
          << "  --allow-obj-material-mismatch  OBJ: import even if Assimp's per-material face counts\n"
          << "                          differ from the file's usemtl directives (recorded as a warning)\n"
          << "  --no-obj-mtllib-hoist   OBJ: read mtllib directives where they are (diagnostics only;\n"
          << "                          Assimp then mis-assigns materials after a repeated or late mtllib)\n"
          << "  --help            Show this help text\n\n"
          << "Notes:\n"
          << "  Runtime rendering remains glTF-first; FBX is importer-only.\n";
    if (!ImportBackendAvailable()) {
        usage << "  This build does not include Assimp.\n";
    }
    return usage.str();
}

ImportResult RunImportPipeline(const ImportOptions& options) {
#if PATH_TRACER_ENABLE_ASSIMP
    return RunImportPipelineImpl(options);
#else
    ImportResult result;
    result.success = false;
    result.message =
        "[PathTracerImport] Assimp not available in this build. Rebuild with -DWITH_ASSIMP=ON.";
    (void)options;
    return result;
#endif
}

}  // namespace PathTracer
