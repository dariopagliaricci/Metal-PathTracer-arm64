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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if PATH_TRACER_ENABLE_ASSIMP
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

void WriteTextureBindingJson(std::ostringstream& json,
                             const char* key,
                             int textureIndex,
                             int texCoord) {
    if (textureIndex < 0) {
        return;
    }
    json << ",\"" << key << "\":{\"index\":" << textureIndex;
    if (texCoord > 0) {
        json << ",\"texCoord\":" << texCoord;
    }
    json << "}";
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

bool IsLinearDataExtension(const std::string& extension) {
    const std::string lower = ToLowerAscii(extension);
    return lower == ".dds" || lower == ".png" || lower == ".jpg" || lower == ".jpeg" ||
           lower == ".tga" || lower == ".bmp" || lower == ".ktx2";
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

struct NodeInfo {
    std::string name;
    int mesh = -1;
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
    std::vector<NodeInfo> nodes;
    std::vector<int> rootNodes;
    std::vector<TextureInventoryEntry> textureInventory;
    std::vector<ManifestTextureEntry> manifestTextures;
    std::vector<TextureAlphaInfo> textureAlphaInfo;
    std::vector<CanonicalManifestEntry> canonicalManifestEntries;
    std::vector<std::string> warnings;
    uint32_t ormTextureCount = 0;
    std::unordered_map<std::string, int> textureByKey;
    std::map<std::vector<unsigned int>, int> meshByPrimitiveSet;
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

void LogMaterialNote(const std::string& materialName, const std::string& message) {
    std::cout << "[ImportMaterial][note] " << materialName << ": " << message << "\n";
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

std::optional<int> RegisterMaterialTexture(ImportContext& ctx,
                                           aiMaterial* material,
                                           aiTextureType type) {
    aiString texturePath;
    if (material->GetTexture(type, 0, &texturePath) != aiReturn_SUCCESS) {
        return std::nullopt;
    }
    std::string error;
    const auto asset = ResolveTextureAsset(ctx, texturePath, error);
    if (!asset) {
        ctx.warnings.push_back(error);
        return std::nullopt;
    }
    return RegisterTexture(ctx, *asset);
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

std::optional<int> RegisterBistroNormalTexture(ImportContext& ctx,
                                               const TextureAsset& asset,
                                               const std::string& materialName) {
    if (asset.sourcePath.empty()) {
        LogMaterialDecision(materialName,
                            "NORMALS",
                            "used",
                            "embedded normal texture cannot be transformed offline; keeping original texture");
        return RegisterTexture(ctx, asset);
    }

    try {
        Import::DecodedTexture decoded = Import::decodeTextureFile(asset.sourcePath.string());
        for (size_t i = 1; i < decoded.rgba.size(); i += 4u) {
            decoded.rgba[i] = static_cast<uint8_t>(255u - decoded.rgba[i]);
        }
        LogMaterialDecision(materialName,
                            "NORMALS",
                            "used",
                            "Bistro/CryEngine normal Y flipped to OpenGL convention during import");
        return RegisterGeneratedTexture(ctx,
                                        "bistro-normal-yflip:" + asset.key,
                                        asset.debugName + "_ogl",
                                        decoded.rgba,
                                        decoded.width,
                                        decoded.height);
    } catch (const std::exception& ex) {
        LogMaterialWarning(ctx,
                           materialName,
                           "failed to flip Bistro normal texture, keeping original: " + std::string(ex.what()));
        return RegisterTexture(ctx, asset);
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

            Import::DecodedTexture normalDecoded = bundle.normal->decoded.decoded;
            for (size_t px = 1; px < normalDecoded.rgba.size(); px += 4u) {
                normalDecoded.rgba[px] = static_cast<uint8_t>(255u - normalDecoded.rgba[px]);
            }
            manifest.normalYFlipped = true;
            LogMaterialDecision(info.name,
                                "NORMALS",
                                "used",
                                "canonical CryEngine normal G flipped during decode");
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

void ConvertMaterials(ImportContext& ctx) {
    if (UseCanonicalImport(ctx) && IsBistroImport(ctx)) {
        ConvertCanonicalMaterials(ctx);
        return;
    }
    ConvertGenericMaterials(ctx);
}

void ConvertMeshes(ImportContext& ctx) {
    ctx.primitives.resize(ctx.scene->mNumMeshes);
    ctx.stats.meshCount = ctx.scene->mNumMeshes;
    for (unsigned int meshIndex = 0; meshIndex < ctx.scene->mNumMeshes; ++meshIndex) {
        const aiMesh* mesh = ctx.scene->mMeshes[meshIndex];
        PrimitiveRef primitive;

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
                texcoord0.push_back(uv.y);
            }
            if (mesh->HasTextureCoords(1)) {
                const aiVector3D uv = mesh->mTextureCoords[1][vertex];
                texcoord1.push_back(uv.x);
                texcoord1.push_back(uv.y);
            }
            if (mesh->HasTangentsAndBitangents()) {
                const auto tangent = MakeTangent(mesh, vertex);
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
    for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
        primitiveSet.push_back(node->mMeshes[i]);
    }
    out.mesh = RegisterMesh(ctx, primitiveSet, out.name);
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
         << aiGetVersionRevision() << "\",\n";
    json << "  \"source_path\": \"" << JsonEscape(ctx.sourcePath.string()) << "\",\n";
    json << "  \"mesh_count\": " << ctx.stats.meshCount << ",\n";
    json << "  \"triangle_count\": " << ctx.stats.triangleCount << ",\n";
    json << "  \"node_count\": " << ctx.stats.nodeCount << ",\n";

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

    ctx.scene = importer.ReadFile(ctx.sourcePath.string(), postprocess);
    if (!ctx.scene) {
        result.message = std::string("Assimp import failed: ") + importer.GetErrorString();
        return result;
    }
    if (!ctx.scene->mRootNode) {
        result.message = "Imported scene has no root node.";
        return result;
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
    ConvertMaterials(ctx);
    ConvertMeshes(ctx);
    ctx.rootNodes.push_back(ConvertNode(ctx, ctx.scene->mRootNode));

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
