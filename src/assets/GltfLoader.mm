#import "assets/GltfLoader.h"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdint>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include <simd/simd.h>

#include "MetalShaderTypes.h"
#include "assets/TangentGen.h"
#include "renderer/SceneResources.h"

namespace fs = std::filesystem;

namespace PathTracer {

namespace {

using PathTracerShaderTypes::MaterialType;
using PathTracerShaderTypes::MaterialData;

constexpr uint32_t kInvalidTextureIndex = 0xFFFFFFFFu;

struct GltfBuffer {
    std::vector<uint8_t> data;
};

struct GltfBufferView {
    int buffer = -1;
    size_t offset = 0;
    size_t length = 0;
    size_t stride = 0;
};

struct GltfAccessor {
    int bufferView = -1;
    size_t offset = 0;
    size_t count = 0;
    int componentType = 0;
    std::string type;
    bool normalized = false;
};

struct GltfImage {
    std::string name;
    std::string uri;
    std::string mimeType;
    int bufferView = -1;
};

struct GltfTexture {
    int source = -1;
    int sampler = -1;
};

struct GltfSampler {
    int magFilter = -1;
    int minFilter = -1;
    int wrapS = 10497;
    int wrapT = 10497;
};

struct GltfTextureBinding {
    int index = -1;
    int texCoord = 0;
    simd::float2 offset = simd_make_float2(0.0f, 0.0f);
    simd::float2 scale = simd_make_float2(1.0f, 1.0f);
    float rotation = 0.0f;
};

struct GltfMaterial {
    std::string name;
    simd::float4 baseColorFactor = simd_make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;
    std::string alphaMode = "OPAQUE";
    float alphaCutoff = 0.5f;
    bool doubleSided = false;
    GltfTextureBinding baseColorTexture;
    GltfTextureBinding metallicRoughnessTexture;
    GltfTextureBinding normalTexture;
    float normalScale = 1.0f;
    GltfTextureBinding occlusionTexture;
    float occlusionStrength = 1.0f;
    GltfTextureBinding emissiveTexture;
    simd::float3 emissiveFactor = simd_make_float3(0.0f, 0.0f, 0.0f);
    float emissiveStrength = 1.0f;
    bool hasTransmission = false;
    float transmissionFactor = 0.0f;
    GltfTextureBinding transmissionTexture;
    bool hasVolume = false;
    float thicknessFactor = 0.0f;
    simd::float3 attenuationColor = simd_make_float3(1.0f, 1.0f, 1.0f);
    float attenuationDistance = std::numeric_limits<float>::infinity();
    bool hasIor = false;
    float ior = 1.5f;
    bool disableOrmTexture = false;
};

struct GltfPrimitive {
    int material = -1;
    int positionAccessor = -1;
    int normalAccessor = -1;
    int texcoordAccessor = -1;
    int texcoord1Accessor = -1;
    int tangentAccessor = -1;
    int indexAccessor = -1;
    int mode = 4;
};

struct GltfMesh {
    std::string name;
    std::vector<GltfPrimitive> primitives;
};

struct GltfCamera {
    bool isPerspective = false;
    float yfov = 0.0f;
    float znear = 0.01f;
    float zfar = 0.0f;
    float aspect = 0.0f;
};

struct GltfNode {
    std::string name;
    int mesh = -1;
    int camera = -1;
    std::vector<int> children;
    simd::float3 translation = simd_make_float3(0.0f, 0.0f, 0.0f);
    simd::float4 rotation = simd_make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    simd::float3 scale = simd_make_float3(1.0f, 1.0f, 1.0f);
    bool hasMatrix = false;
    simd::float4x4 matrix = matrix_identity_float4x4;
};

bool ReadFileBytes(const fs::path& path, std::vector<uint8_t>& outBytes, std::string& error) {
    std::error_code ec;
    if (!fs::exists(path, ec)) {
        error = "glTF file not found: " + path.string();
        return false;
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        error = "Failed to open glTF file: " + path.string();
        return false;
    }
    stream.seekg(0, std::ios::end);
    std::streamsize size = stream.tellg();
    if (size <= 0) {
        error = "glTF file is empty: " + path.string();
        return false;
    }
    stream.seekg(0, std::ios::beg);
    outBytes.resize(static_cast<size_t>(size));
    stream.read(reinterpret_cast<char*>(outBytes.data()), size);
    if (!stream) {
        error = "Failed to read glTF file: " + path.string();
        return false;
    }
    return true;
}

bool DecodeDataUri(const std::string& uri, std::vector<uint8_t>& outBytes, std::string& mimeType) {
    constexpr const char* kPrefix = "data:";
    if (uri.rfind(kPrefix, 0) != 0) {
        return false;
    }
    size_t comma = uri.find(',');
    if (comma == std::string::npos) {
        return false;
    }
    std::string header = uri.substr(5, comma - 5);
    std::string dataPart = uri.substr(comma + 1);
    bool isBase64 = header.find(";base64") != std::string::npos;
    size_t semi = header.find(';');
    mimeType = (semi == std::string::npos) ? header : header.substr(0, semi);
    if (!isBase64) {
        return false;
    }
    NSString* dataStr = [NSString stringWithUTF8String:dataPart.c_str()];
    NSData* decoded = [[NSData alloc] initWithBase64EncodedString:dataStr options:0];
    if (!decoded) {
        return false;
    }
    outBytes.resize(decoded.length);
    memcpy(outBytes.data(), decoded.bytes, decoded.length);
    return true;
}

std::string ToLowerAscii(std::string value) {
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return value;
}

bool ContainsCaseInsensitive(const std::string& text, const std::string& needle) {
    if (needle.empty()) {
        return false;
    }
    const std::string lowerText = ToLowerAscii(text);
    const std::string lowerNeedle = ToLowerAscii(needle);
    return lowerText.find(lowerNeedle) != std::string::npos;
}

simd::float4x4 MakeTranslation(const simd::float3& t) {
    simd::float4x4 m = matrix_identity_float4x4;
    m.columns[3] = simd_make_float4(t, 1.0f);
    return m;
}

simd::float4x4 MakeScale(const simd::float3& s) {
    simd::float4x4 m = matrix_identity_float4x4;
    m.columns[0].x = s.x;
    m.columns[1].y = s.y;
    m.columns[2].z = s.z;
    return m;
}

simd::float4x4 MakeRotation(const simd::float4& q) {
    float x = q.x;
    float y = q.y;
    float z = q.z;
    float w = q.w;

    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float xz = x * z;
    float yz = y * z;
    float wx = w * x;
    float wy = w * y;
    float wz = w * z;

    simd::float4x4 m;
    m.columns[0] = simd_make_float4(1.0f - 2.0f * (yy + zz),
                                    2.0f * (xy + wz),
                                    2.0f * (xz - wy),
                                    0.0f);
    m.columns[1] = simd_make_float4(2.0f * (xy - wz),
                                    1.0f - 2.0f * (xx + zz),
                                    2.0f * (yz + wx),
                                    0.0f);
    m.columns[2] = simd_make_float4(2.0f * (xz + wy),
                                    2.0f * (yz - wx),
                                    1.0f - 2.0f * (xx + yy),
                                    0.0f);
    m.columns[3] = simd_make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    return m;
}

simd::float4x4 ComposeTrs(const simd::float3& t, const simd::float4& r, const simd::float3& s) {
    return simd_mul(MakeTranslation(t), simd_mul(MakeRotation(r), MakeScale(s)));
}

bool GetFloatArray(NSDictionary* dict, NSString* key, std::vector<float>& out) {
    id val = dict[key];
    if (![val isKindOfClass:[NSArray class]]) {
        return false;
    }
    NSArray* arr = (NSArray*)val;
    out.resize(arr.count);
    for (NSUInteger i = 0; i < arr.count; ++i) {
        id num = arr[i];
        out[i] = [num floatValue];
    }
    return true;
}

int GetInt(NSDictionary* dict, NSString* key, int defaultValue = -1) {
    id val = dict[key];
    if ([val isKindOfClass:[NSNumber class]]) {
        return [(NSNumber*)val intValue];
    }
    return defaultValue;
}

float GetFloat(NSDictionary* dict, NSString* key, float defaultValue = 0.0f) {
    id val = dict[key];
    if ([val isKindOfClass:[NSNumber class]]) {
        return [(NSNumber*)val floatValue];
    }
    return defaultValue;
}

bool GetBool(NSDictionary* dict, NSString* key, bool defaultValue = false) {
    id val = dict[key];
    if ([val isKindOfClass:[NSNumber class]]) {
        return [(NSNumber*)val boolValue];
    }
    return defaultValue;
}

std::string GetString(NSDictionary* dict, NSString* key) {
    id val = dict[key];
    if ([val isKindOfClass:[NSString class]]) {
        return std::string([(NSString*)val UTF8String]);
    }
    return {};
}

int ClampTexCoordSet(int texCoord) {
    if (texCoord < 0) {
        return 0;
    }
    return std::min(texCoord, 1);
}

void ParseTextureBinding(NSDictionary* textureInfo, GltfTextureBinding& out) {
    if (![textureInfo isKindOfClass:[NSDictionary class]]) {
        return;
    }
    out.index = GetInt(textureInfo, @"index", -1);
    out.texCoord = ClampTexCoordSet(GetInt(textureInfo, @"texCoord", 0));

    NSDictionary* extensions = textureInfo[@"extensions"];
    if (![extensions isKindOfClass:[NSDictionary class]]) {
        return;
    }
    NSDictionary* transform = extensions[@"KHR_texture_transform"];
    if (![transform isKindOfClass:[NSDictionary class]]) {
        return;
    }

    std::vector<float> offset;
    if (GetFloatArray(transform, @"offset", offset) && offset.size() >= 2) {
        out.offset = simd_make_float2(offset[0], offset[1]);
    }
    std::vector<float> scale;
    if (GetFloatArray(transform, @"scale", scale) && scale.size() >= 2) {
        out.scale = simd_make_float2(scale[0], scale[1]);
    }
    out.rotation = GetFloat(transform, @"rotation", 0.0f);
    out.texCoord = ClampTexCoordSet(GetInt(transform, @"texCoord", out.texCoord));
}

size_t ComponentCountForType(const std::string& type) {
    if (type == "SCALAR") return 1;
    if (type == "VEC2") return 2;
    if (type == "VEC3") return 3;
    if (type == "VEC4") return 4;
    return 0;
}

bool ReadAccessorBytes(const std::vector<GltfBuffer>& buffers,
                       const std::vector<GltfBufferView>& views,
                       const GltfAccessor& accessor,
                       const uint8_t*& outBase,
                       size_t& outStride) {
    if (accessor.bufferView < 0 || accessor.bufferView >= static_cast<int>(views.size())) {
        return false;
    }
    const auto& view = views[accessor.bufferView];
    if (view.buffer < 0 || view.buffer >= static_cast<int>(buffers.size())) {
        return false;
    }
    const auto& buffer = buffers[view.buffer];
    size_t offset = view.offset + accessor.offset;
    if (offset >= buffer.data.size()) {
        return false;
    }
    outBase = buffer.data.data() + offset;
    size_t elementSize = 0;
    switch (accessor.componentType) {
        case 5126: elementSize = sizeof(float); break;
        case 5125: elementSize = sizeof(uint32_t); break;
        case 5123: elementSize = sizeof(uint16_t); break;
        case 5121: elementSize = sizeof(uint8_t); break;
        case 5122: elementSize = sizeof(int16_t); break;
        case 5120: elementSize = sizeof(int8_t); break;
        default: return false;
    }
    size_t compCount = ComponentCountForType(accessor.type);
    if (compCount == 0) {
        return false;
    }
    outStride = view.stride != 0 ? view.stride : (elementSize * compCount);
    return true;
}

float ReadComponentAsFloat(const uint8_t* base,
                           int componentType,
                           bool normalized) {
    switch (componentType) {
        case 5126: {
            float v;
            memcpy(&v, base, sizeof(float));
            return v;
        }
        case 5125: {
            uint32_t v;
            memcpy(&v, base, sizeof(uint32_t));
            return normalized ? static_cast<float>(v) / 4294967295.0f : static_cast<float>(v);
        }
        case 5123: {
            uint16_t v;
            memcpy(&v, base, sizeof(uint16_t));
            return normalized ? static_cast<float>(v) / 65535.0f : static_cast<float>(v);
        }
        case 5121: {
            uint8_t v;
            memcpy(&v, base, sizeof(uint8_t));
            return normalized ? static_cast<float>(v) / 255.0f : static_cast<float>(v);
        }
        case 5122: {
            int16_t v;
            memcpy(&v, base, sizeof(int16_t));
            if (normalized) {
                return std::max(-1.0f, static_cast<float>(v) / 32767.0f);
            }
            return static_cast<float>(v);
        }
        case 5120: {
            int8_t v;
            memcpy(&v, base, sizeof(int8_t));
            if (normalized) {
                return std::max(-1.0f, static_cast<float>(v) / 127.0f);
            }
            return static_cast<float>(v);
        }
        default:
            break;
    }
    return 0.0f;
}

bool ReadAccessorFloatN(const std::vector<GltfBuffer>& buffers,
                        const std::vector<GltfBufferView>& views,
                        const GltfAccessor& accessor,
                        size_t components,
                        std::vector<float>& out) {
    if (ComponentCountForType(accessor.type) != components) {
        return false;
    }
    const uint8_t* base = nullptr;
    size_t stride = 0;
    if (!ReadAccessorBytes(buffers, views, accessor, base, stride)) {
        return false;
    }
    out.resize(accessor.count * components);
    for (size_t i = 0; i < accessor.count; ++i) {
        const uint8_t* ptr = base + i * stride;
        for (size_t c = 0; c < components; ++c) {
            size_t offset = 0;
            switch (accessor.componentType) {
                case 5126: offset = sizeof(float) * c; break;
                case 5125: offset = sizeof(uint32_t) * c; break;
                case 5123: offset = sizeof(uint16_t) * c; break;
                case 5121: offset = sizeof(uint8_t) * c; break;
                case 5122: offset = sizeof(int16_t) * c; break;
                case 5120: offset = sizeof(int8_t) * c; break;
                default: return false;
            }
            out[i * components + c] = ReadComponentAsFloat(ptr + offset,
                                                           accessor.componentType,
                                                           accessor.normalized);
        }
    }
    return true;
}

bool ReadAccessorIndices(const std::vector<GltfBuffer>& buffers,
                         const std::vector<GltfBufferView>& views,
                         const GltfAccessor& accessor,
                         std::vector<uint32_t>& out) {
    const uint8_t* base = nullptr;
    size_t stride = 0;
    if (!ReadAccessorBytes(buffers, views, accessor, base, stride)) {
        return false;
    }
    out.resize(accessor.count);
    for (size_t i = 0; i < accessor.count; ++i) {
        const uint8_t* ptr = base + i * stride;
        switch (accessor.componentType) {
            case 5125: {
                uint32_t v;
                memcpy(&v, ptr, sizeof(uint32_t));
                out[i] = v;
                break;
            }
            case 5123: {
                uint16_t v;
                memcpy(&v, ptr, sizeof(uint16_t));
                out[i] = static_cast<uint32_t>(v);
                break;
            }
            case 5121: {
                uint8_t v;
                memcpy(&v, ptr, sizeof(uint8_t));
                out[i] = static_cast<uint32_t>(v);
                break;
            }
            default:
                return false;
        }
    }
    return true;
}

uint32_t ResolveTextureIndex(const std::string& gltfPath,
                             const std::vector<GltfTexture>& textures,
                             const std::vector<GltfSampler>& samplers,
                             const std::vector<GltfImage>& images,
                             const std::vector<GltfBuffer>& buffers,
                             const std::vector<GltfBufferView>& views,
                             int textureIndex,
                             bool srgb,
                             MaterialTextureSemantic semantic,
                             SceneResources& resources) {
    if (textureIndex < 0 || textureIndex >= static_cast<int>(textures.size())) {
        return kInvalidTextureIndex;
    }
    int imageIndex = textures[textureIndex].source;
    if (imageIndex < 0 || imageIndex >= static_cast<int>(images.size())) {
        return kInvalidTextureIndex;
    }
    const GltfTexture& texture = textures[textureIndex];
    MaterialTextureSamplerDesc samplerDesc{};
    const MaterialTextureSamplerDesc* samplerDescPtr = nullptr;
    bool hasGltfSampler = false;
    if (texture.sampler >= 0 && texture.sampler < static_cast<int>(samplers.size())) {
        const GltfSampler& gltfSampler = samplers[texture.sampler];
        samplerDesc.magFilter = gltfSampler.magFilter;
        samplerDesc.minFilter = gltfSampler.minFilter;
        samplerDesc.wrapS = gltfSampler.wrapS;
        samplerDesc.wrapT = gltfSampler.wrapT;
        hasGltfSampler = true;
    }

    // Non-color maps need bandwidth-limited sampling in the path tracer.
    // Keep wrap modes from glTF, but force trilinear mip filtering policy.
    if (!srgb) {
        samplerDesc.magFilter = 9729;  // LINEAR
        samplerDesc.minFilter = 9987;  // LINEAR_MIPMAP_LINEAR
        samplerDescPtr = &samplerDesc;
    } else if (hasGltfSampler) {
        samplerDescPtr = &samplerDesc;
    }
    const auto& image = images[imageIndex];
    std::string error;

    if (!image.uri.empty()) {
        std::vector<uint8_t> data;
        std::string mime;
        if (DecodeDataUri(image.uri, data, mime)) {
            std::string label = gltfPath + ":image:" + std::to_string(imageIndex);
            return resources.addMaterialTextureFromData(data.data(),
                                                        data.size(),
                                                        label,
                                                        srgb,
                                                        &error,
                                                        samplerDescPtr,
                                                        semantic);
        }
        fs::path imagePath = fs::path(gltfPath).parent_path() / image.uri;
        return resources.addMaterialTextureFromFile(imagePath.string(),
                                                    srgb,
                                                    &error,
                                                    samplerDescPtr,
                                                    semantic);
    }

    if (image.bufferView >= 0 && image.bufferView < static_cast<int>(views.size())) {
        const auto& view = views[image.bufferView];
        if (view.buffer >= 0 && view.buffer < static_cast<int>(buffers.size())) {
            const auto& buffer = buffers[view.buffer];
            size_t offset = view.offset;
            size_t length = view.length;
            if (offset + length <= buffer.data.size() && length > 0) {
                std::string label = gltfPath + ":image:" + std::to_string(imageIndex);
                return resources.addMaterialTextureFromData(buffer.data.data() + offset,
                                                           length,
                                                           label,
                                                           srgb,
                                                           &error,
                                                           samplerDescPtr,
                                                           semantic);
            }
        }
    }

    return kInvalidTextureIndex;
}

simd::float3 ComputeVolumeSigmaA(const GltfMaterial& src) {
    if (!src.hasVolume) {
        return simd_make_float3(0.0f, 0.0f, 0.0f);
    }
    if (!std::isfinite(src.attenuationDistance) || src.attenuationDistance <= 0.0f) {
        return simd_make_float3(0.0f, 0.0f, 0.0f);
    }
    simd::float3 safeColor = simd_clamp(src.attenuationColor,
                                        simd_make_float3(1.0e-6f, 1.0e-6f, 1.0e-6f),
                                        simd_make_float3(1.0f, 1.0f, 1.0f));
    simd::float3 sigmaA = simd_make_float3(-std::log(safeColor.x),
                                           -std::log(safeColor.y),
                                           -std::log(safeColor.z)) / src.attenuationDistance;
    return simd_max(sigmaA, simd_make_float3(0.0f, 0.0f, 0.0f));
}

simd::float4 TextureTransformRow0FromBinding(const GltfTextureBinding& binding) {
    float c = std::cos(binding.rotation);
    float s = std::sin(binding.rotation);
    return simd_make_float4(c * binding.scale.x,
                            -s * binding.scale.y,
                            binding.offset.x,
                            0.0f);
}

simd::float4 TextureTransformRow1FromBinding(const GltfTextureBinding& binding) {
    float c = std::cos(binding.rotation);
    float s = std::sin(binding.rotation);
    return simd_make_float4(s * binding.scale.x,
                            c * binding.scale.y,
                            binding.offset.y,
                            0.0f);
}

void InitializeMaterialTextureDefaults(MaterialData& material) {
    material.textureUvSet0 = simd_make_uint4(0u, 0u, 0u, 0u);
    material.textureUvSet1 = simd_make_uint4(0u, 0u, 0u, 0u);
    material.textureTransform0 = simd_make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    material.textureTransform1 = simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    material.textureTransform2 = simd_make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    material.textureTransform3 = simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    material.textureTransform4 = simd_make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    material.textureTransform5 = simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    material.textureTransform6 = simd_make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    material.textureTransform7 = simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    material.textureTransform8 = simd_make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    material.textureTransform9 = simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    material.textureTransform10 = simd_make_float4(1.0f, 0.0f, 0.0f, 0.0f);
    material.textureTransform11 = simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f);
}

MaterialData BuildGltfMaterial(const GltfMaterial& src,
                               const std::string& gltfPath,
                               const std::vector<GltfTexture>& textures,
                               const std::vector<GltfSampler>& samplers,
                               const std::vector<GltfImage>& images,
                               const std::vector<GltfBuffer>& buffers,
                               const std::vector<GltfBufferView>& views,
                               const GltfLoadOptions& loadOptions,
                               SceneResources& resources) {
    const float roughnessScale = 1.0f;
    const float normalScale = 1.0f;
    const float pbrIor = std::clamp(src.hasIor ? src.ior : 1.5f, 1.0f, 3.0f);

    MaterialData material{};
    InitializeMaterialTextureDefaults(material);
    material.baseColorRoughness =
        simd_make_float4(simd_make_float3(src.baseColorFactor.x,
                                          src.baseColorFactor.y,
                                          src.baseColorFactor.z),
                         std::clamp(src.roughnessFactor * roughnessScale, 0.0f, 1.0f));
    float thickness = src.hasVolume ? std::max(src.thicknessFactor, 0.0f) : 0.0f;
    material.typeEta = simd_make_float4(static_cast<float>(MaterialType::PbrMetallicRoughness),
                                        pbrIor,
                                        src.doubleSided ? 1.0f : 0.0f,
                                        thickness);
    float emissiveScale = std::max(src.emissiveStrength * loadOptions.emissiveScale, 0.0f);
    material.emission = simd_make_float4(src.emissiveFactor * emissiveScale, 0.0f);
    material.conductorEta = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    material.conductorK = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    material.coatParams = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    material.coatTint = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    material.coatAbsorption = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (src.hasVolume) {
        material.dielectricSigmaA = simd_make_float4(ComputeVolumeSigmaA(src), 0.0f);
    } else {
        material.dielectricSigmaA = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    material.sssSigmaA = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    material.sssSigmaS = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    material.sssParams = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    material.carpaintBaseParams = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    material.carpaintFlakeParams = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    material.carpaintBaseEta = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    material.carpaintBaseK = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    material.carpaintBaseTint = simd_make_float4(1.0f, 1.0f, 1.0f, 0.0f);

    const bool baseColorSrgb = !loadOptions.forceLinearBaseColor;
    const bool emissiveSrgb = !loadOptions.forceLinearEmissive;
    uint32_t baseColorIdx = ResolveTextureIndex(gltfPath, textures, samplers, images, buffers, views,
                                                src.baseColorTexture.index, baseColorSrgb,
                                                MaterialTextureSemantic::Generic, resources);
    uint32_t mrIdx = ResolveTextureIndex(gltfPath, textures, samplers, images, buffers, views,
                                         src.metallicRoughnessTexture.index, false,
                                         MaterialTextureSemantic::Orm, resources);
    uint32_t normalIdx = ResolveTextureIndex(gltfPath, textures, samplers, images, buffers, views,
                                             src.normalTexture.index, false,
                                             MaterialTextureSemantic::Generic, resources);
    MaterialTextureSemantic occlusionSemantic = MaterialTextureSemantic::Generic;
    if (src.occlusionTexture.index >= 0 &&
        src.occlusionTexture.index == src.metallicRoughnessTexture.index) {
        occlusionSemantic = MaterialTextureSemantic::Orm;
    }
    uint32_t occlusionIdx = ResolveTextureIndex(gltfPath, textures, samplers, images, buffers, views,
                                                src.occlusionTexture.index, false,
                                                occlusionSemantic, resources);
    uint32_t emissiveIdx = ResolveTextureIndex(gltfPath, textures, samplers, images, buffers, views,
                                               src.emissiveTexture.index, emissiveSrgb,
                                               MaterialTextureSemantic::Generic, resources);
    uint32_t transmissionIdx = ResolveTextureIndex(gltfPath, textures, samplers, images, buffers, views,
                                                   src.transmissionTexture.index, false,
                                                   MaterialTextureSemantic::Generic, resources);
    if (mrIdx != kInvalidTextureIndex) {
        const auto& loadedTextures = resources.materialTextures();
        if (mrIdx < loadedTextures.size()) {
            id<MTLTexture> mrTexture = loadedTextures[mrIdx];
            if (mrTexture && mrTexture.mipmapLevelCount <= 1) {
                NSLog(@"[glTF] Warning: ORM texture %u has mipCount=%lu (expected > 1)",
                      mrIdx,
                      static_cast<unsigned long>(mrTexture.mipmapLevelCount));
            } else if (mrTexture) {
                NSLog(@"[glTF] ORM texture %u format=%lu mips=%lu",
                      mrIdx,
                      static_cast<unsigned long>(mrTexture.pixelFormat),
                      static_cast<unsigned long>(mrTexture.mipmapLevelCount));
            }
        }
    }

    material.textureIndices0 = simd_make_uint4(baseColorIdx, mrIdx, normalIdx, occlusionIdx);
    material.textureIndices1 =
        simd_make_uint4(emissiveIdx, transmissionIdx, kInvalidTextureIndex, kInvalidTextureIndex);
    material.textureUvSet0 = simd_make_uint4(static_cast<uint32_t>(ClampTexCoordSet(src.baseColorTexture.texCoord)),
                                             static_cast<uint32_t>(ClampTexCoordSet(src.metallicRoughnessTexture.texCoord)),
                                             static_cast<uint32_t>(ClampTexCoordSet(src.normalTexture.texCoord)),
                                             static_cast<uint32_t>(ClampTexCoordSet(src.occlusionTexture.texCoord)));
    material.textureUvSet1 = simd_make_uint4(static_cast<uint32_t>(ClampTexCoordSet(src.emissiveTexture.texCoord)),
                                             static_cast<uint32_t>(ClampTexCoordSet(src.transmissionTexture.texCoord)),
                                             0u,
                                             0u);
    material.textureTransform0 = TextureTransformRow0FromBinding(src.baseColorTexture);
    material.textureTransform1 = TextureTransformRow1FromBinding(src.baseColorTexture);
    material.textureTransform2 = TextureTransformRow0FromBinding(src.metallicRoughnessTexture);
    material.textureTransform3 = TextureTransformRow1FromBinding(src.metallicRoughnessTexture);
    material.textureTransform4 = TextureTransformRow0FromBinding(src.normalTexture);
    material.textureTransform5 = TextureTransformRow1FromBinding(src.normalTexture);
    material.textureTransform6 = TextureTransformRow0FromBinding(src.occlusionTexture);
    material.textureTransform7 = TextureTransformRow1FromBinding(src.occlusionTexture);
    material.textureTransform8 = TextureTransformRow0FromBinding(src.emissiveTexture);
    material.textureTransform9 = TextureTransformRow1FromBinding(src.emissiveTexture);
    material.textureTransform10 = TextureTransformRow0FromBinding(src.transmissionTexture);
    material.textureTransform11 = TextureTransformRow1FromBinding(src.transmissionTexture);
    float occlusionStrength = std::clamp(src.occlusionStrength, 0.0f, 1.0f);
    material.pbrParams = simd_make_float4(std::clamp(src.metallicFactor, 0.0f, 1.0f),
                                          std::clamp(src.roughnessFactor * roughnessScale, 0.0f, 1.0f),
                                          occlusionStrength,
                                          std::max(src.normalScale * normalScale, 0.0f));
    material.materialFlags = src.disableOrmTexture ? PathTracerShaderTypes::kMaterialFlagDisableOrm : 0u;
    if (src.disableOrmTexture && loadOptions.disableOrmRoughnessOverride >= 0.0f) {
        material.pbrParams.y = std::clamp(loadOptions.disableOrmRoughnessOverride, 0.0f, 1.0f);
    }
    material.materialPad0 = 0u;
    material.materialPad1 = 0u;
    material.materialPad2 = 0u;
    int alphaMode = 0;
    if (src.alphaMode == "MASK") {
        alphaMode = 1;
    } else if (src.alphaMode == "BLEND") {
        alphaMode = 2;
    }
    float alphaFactor = std::clamp(src.baseColorFactor.w, 0.0f, 1.0f);
    float alphaCutoff = std::clamp(src.alphaCutoff, 0.0f, 1.0f);
    float transmissionFactor = std::clamp(src.transmissionFactor, 0.0f, 1.0f);
    material.pbrExtras = simd_make_float4(alphaFactor,
                                          alphaCutoff,
                                          transmissionFactor,
                                          static_cast<float>(alphaMode));

    return material;
}

}  // namespace

bool LoadGltfScene(const std::string& path,
                   SceneResources& resources,
                   std::string& errorMessage,
                   GltfCameraInfo* outCamera,
                   const GltfLoadOptions* options) {
    GltfLoadOptions loadOptions{};
    if (options) {
        loadOptions = *options;
    }
    loadOptions.emissiveScale = std::max(loadOptions.emissiveScale, 0.0f);

    fs::path gltfPath(path);
    std::vector<uint8_t> fileBytes;
    if (!ReadFileBytes(gltfPath, fileBytes, errorMessage)) {
        return false;
    }

    NSData* jsonData = nil;
    std::vector<uint8_t> binChunk;

    if (gltfPath.extension() == ".glb") {
        if (fileBytes.size() < 12) {
            errorMessage = "Invalid .glb header";
            return false;
        }
        uint32_t magic = 0;
        uint32_t version = 0;
        uint32_t length = 0;
        memcpy(&magic, fileBytes.data(), 4);
        memcpy(&version, fileBytes.data() + 4, 4);
        memcpy(&length, fileBytes.data() + 8, 4);
        (void)length;
        if (magic != 0x46546C67) {
            errorMessage = "Invalid .glb magic";
            return false;
        }
        if (version != 2) {
            errorMessage = "Unsupported .glb version";
            return false;
        }
        size_t offset = 12;
        while (offset + 8 <= fileBytes.size()) {
            uint32_t chunkLen = 0;
            uint32_t chunkType = 0;
            memcpy(&chunkLen, fileBytes.data() + offset, 4);
            memcpy(&chunkType, fileBytes.data() + offset + 4, 4);
            offset += 8;
            if (offset + chunkLen > fileBytes.size()) {
                errorMessage = "Invalid .glb chunk length";
                return false;
            }
            if (chunkType == 0x4E4F534A) { // JSON
                jsonData = [NSData dataWithBytes:fileBytes.data() + offset length:chunkLen];
            } else if (chunkType == 0x004E4942) { // BIN
                binChunk.assign(fileBytes.begin() + static_cast<long>(offset),
                                fileBytes.begin() + static_cast<long>(offset + chunkLen));
            }
            offset += chunkLen;
        }
        if (!jsonData) {
            errorMessage = "Missing JSON chunk in .glb";
            return false;
        }
    } else {
        jsonData = [NSData dataWithBytes:fileBytes.data() length:fileBytes.size()];
    }

    NSError* nsError = nil;
    id jsonObj = [NSJSONSerialization JSONObjectWithData:jsonData options:0 error:&nsError];
    if (!jsonObj || ![jsonObj isKindOfClass:[NSDictionary class]]) {
        errorMessage = "Failed to parse glTF JSON";
        return false;
    }
    NSDictionary* root = (NSDictionary*)jsonObj;

    std::vector<GltfBuffer> buffers;
    std::vector<GltfBufferView> views;
    std::vector<GltfAccessor> accessors;
    std::vector<GltfImage> images;
    std::vector<GltfSampler> samplers;
    std::vector<GltfTexture> textures;
    std::vector<GltfMaterial> materials;
    std::vector<GltfMesh> meshes;
    std::vector<GltfCamera> cameras;
    std::vector<GltfNode> nodes;

    NSArray* bufferArray = root[@"buffers"];
    if (bufferArray && [bufferArray isKindOfClass:[NSArray class]]) {
        buffers.resize(bufferArray.count);
        for (NSUInteger i = 0; i < bufferArray.count; ++i) {
            NSDictionary* buf = bufferArray[i];
            std::string uri = GetString(buf, @"uri");
            if (!uri.empty()) {
                std::vector<uint8_t> data;
                std::string mime;
                if (DecodeDataUri(uri, data, mime)) {
                    buffers[i].data = std::move(data);
                } else {
                    fs::path binPath = gltfPath.parent_path() / uri;
                    std::string err;
                    if (!ReadFileBytes(binPath, buffers[i].data, err)) {
                        errorMessage = err;
                        return false;
                    }
                }
            } else if (!binChunk.empty()) {
                buffers[i].data = binChunk;
            } else {
                errorMessage = "glTF buffer missing uri and no .glb BIN chunk";
                return false;
            }
        }
    }

    NSArray* viewArray = root[@"bufferViews"];
    if (viewArray && [viewArray isKindOfClass:[NSArray class]]) {
        views.resize(viewArray.count);
        for (NSUInteger i = 0; i < viewArray.count; ++i) {
            NSDictionary* view = viewArray[i];
            views[i].buffer = GetInt(view, @"buffer", -1);
            views[i].offset = static_cast<size_t>(GetInt(view, @"byteOffset", 0));
            views[i].length = static_cast<size_t>(GetInt(view, @"byteLength", 0));
            views[i].stride = static_cast<size_t>(GetInt(view, @"byteStride", 0));
        }
    }

    NSArray* accessorArray = root[@"accessors"];
    if (accessorArray && [accessorArray isKindOfClass:[NSArray class]]) {
        accessors.resize(accessorArray.count);
        for (NSUInteger i = 0; i < accessorArray.count; ++i) {
            NSDictionary* acc = accessorArray[i];
            accessors[i].bufferView = GetInt(acc, @"bufferView", -1);
            accessors[i].offset = static_cast<size_t>(GetInt(acc, @"byteOffset", 0));
            accessors[i].count = static_cast<size_t>(GetInt(acc, @"count", 0));
            accessors[i].componentType = GetInt(acc, @"componentType", 0);
            accessors[i].type = GetString(acc, @"type");
            accessors[i].normalized = GetBool(acc, @"normalized", false);
        }
    }

    NSArray* imageArray = root[@"images"];
    if (imageArray && [imageArray isKindOfClass:[NSArray class]]) {
        images.resize(imageArray.count);
        for (NSUInteger i = 0; i < imageArray.count; ++i) {
            NSDictionary* img = imageArray[i];
            images[i].name = GetString(img, @"name");
            images[i].uri = GetString(img, @"uri");
            images[i].mimeType = GetString(img, @"mimeType");
            images[i].bufferView = GetInt(img, @"bufferView", -1);
        }
    }

    NSArray* samplerArray = root[@"samplers"];
    if (samplerArray && [samplerArray isKindOfClass:[NSArray class]]) {
        samplers.resize(samplerArray.count);
        for (NSUInteger i = 0; i < samplerArray.count; ++i) {
            NSDictionary* sampler = samplerArray[i];
            samplers[i].magFilter = GetInt(sampler, @"magFilter", -1);
            samplers[i].minFilter = GetInt(sampler, @"minFilter", -1);
            samplers[i].wrapS = GetInt(sampler, @"wrapS", 10497);
            samplers[i].wrapT = GetInt(sampler, @"wrapT", 10497);
        }
    }

    NSArray* textureArray = root[@"textures"];
    if (textureArray && [textureArray isKindOfClass:[NSArray class]]) {
        textures.resize(textureArray.count);
        for (NSUInteger i = 0; i < textureArray.count; ++i) {
            NSDictionary* tex = textureArray[i];
            textures[i].source = GetInt(tex, @"source", -1);
            textures[i].sampler = GetInt(tex, @"sampler", -1);
        }
    }

    NSArray* materialArray = root[@"materials"];
    if (materialArray && [materialArray isKindOfClass:[NSArray class]]) {
        materials.resize(materialArray.count);
        for (NSUInteger i = 0; i < materialArray.count; ++i) {
            NSDictionary* mat = materialArray[i];
            GltfMaterial& dst = materials[i];
            dst.name = GetString(mat, @"name");
            std::string alphaMode = GetString(mat, @"alphaMode");
            if (!alphaMode.empty()) {
                dst.alphaMode = alphaMode;
            }
            dst.alphaCutoff = GetFloat(mat, @"alphaCutoff", 0.5f);
            dst.doubleSided = GetBool(mat, @"doubleSided", false);

            NSDictionary* pbr = mat[@"pbrMetallicRoughness"];
            if ([pbr isKindOfClass:[NSDictionary class]]) {
                std::vector<float> baseColor;
                if (GetFloatArray(pbr, @"baseColorFactor", baseColor) && baseColor.size() >= 4) {
                    dst.baseColorFactor = simd_make_float4(baseColor[0], baseColor[1], baseColor[2], baseColor[3]);
                }
                dst.metallicFactor = GetFloat(pbr, @"metallicFactor", 1.0f);
                dst.roughnessFactor = GetFloat(pbr, @"roughnessFactor", 1.0f);
                NSDictionary* baseTex = pbr[@"baseColorTexture"];
                if ([baseTex isKindOfClass:[NSDictionary class]]) {
                    ParseTextureBinding(baseTex, dst.baseColorTexture);
                }
                NSDictionary* mrTex = pbr[@"metallicRoughnessTexture"];
                if ([mrTex isKindOfClass:[NSDictionary class]]) {
                    ParseTextureBinding(mrTex, dst.metallicRoughnessTexture);
                }
            }

            NSDictionary* normalTex = mat[@"normalTexture"];
            if ([normalTex isKindOfClass:[NSDictionary class]]) {
                ParseTextureBinding(normalTex, dst.normalTexture);
                dst.normalScale = GetFloat(normalTex, @"scale", 1.0f);
            }

            NSDictionary* occTex = mat[@"occlusionTexture"];
            if ([occTex isKindOfClass:[NSDictionary class]]) {
                ParseTextureBinding(occTex, dst.occlusionTexture);
                dst.occlusionStrength = GetFloat(occTex, @"strength", 1.0f);
            }

            NSDictionary* emissiveTex = mat[@"emissiveTexture"];
            if ([emissiveTex isKindOfClass:[NSDictionary class]]) {
                ParseTextureBinding(emissiveTex, dst.emissiveTexture);
            }

            std::vector<float> emissiveFactor;
            if (GetFloatArray(mat, @"emissiveFactor", emissiveFactor) && emissiveFactor.size() >= 3) {
                dst.emissiveFactor = simd_make_float3(emissiveFactor[0], emissiveFactor[1], emissiveFactor[2]);
            }

            NSDictionary* extensions = mat[@"extensions"];
            if ([extensions isKindOfClass:[NSDictionary class]]) {
                NSDictionary* transmission = extensions[@"KHR_materials_transmission"];
                if ([transmission isKindOfClass:[NSDictionary class]]) {
                    dst.hasTransmission = true;
                    dst.transmissionFactor = std::max(GetFloat(transmission, @"transmissionFactor", 0.0f), 0.0f);
                    NSDictionary* transmissionTexture = transmission[@"transmissionTexture"];
                    if ([transmissionTexture isKindOfClass:[NSDictionary class]]) {
                        ParseTextureBinding(transmissionTexture, dst.transmissionTexture);
                    }
                }

                NSDictionary* volume = extensions[@"KHR_materials_volume"];
                if ([volume isKindOfClass:[NSDictionary class]]) {
                    dst.hasVolume = true;
                    dst.thicknessFactor = std::max(GetFloat(volume, @"thicknessFactor", 0.0f), 0.0f);
                    std::vector<float> attenuationColor;
                    if (GetFloatArray(volume, @"attenuationColor", attenuationColor) &&
                        attenuationColor.size() >= 3) {
                        dst.attenuationColor = simd_make_float3(attenuationColor[0],
                                                                attenuationColor[1],
                                                                attenuationColor[2]);
                    }
                    dst.attenuationDistance = GetFloat(volume,
                                                       @"attenuationDistance",
                                                       std::numeric_limits<float>::infinity());
                }

                NSDictionary* ior = extensions[@"KHR_materials_ior"];
                if ([ior isKindOfClass:[NSDictionary class]]) {
                    dst.hasIor = true;
                    dst.ior = GetFloat(ior, @"ior", 1.5f);
                }

                NSDictionary* emissiveStrength = extensions[@"KHR_materials_emissive_strength"];
                if ([emissiveStrength isKindOfClass:[NSDictionary class]]) {
                    dst.emissiveStrength =
                        std::max(GetFloat(emissiveStrength, @"emissiveStrength", 1.0f), 0.0f);
                }
            }
        }
    }

    if (materials.empty()) {
        materials.emplace_back();
    }

    for (size_t i = 0; i < materials.size(); ++i) {
        bool disableOrm = ContainsCaseInsensitive(materials[i].name, "visor");
        if (!disableOrm) {
            for (const std::string& pattern : loadOptions.disableOrmMaterialNameSubstrings) {
                if (ContainsCaseInsensitive(materials[i].name, pattern)) {
                    disableOrm = true;
                    break;
                }
            }
        }
        materials[i].disableOrmTexture = disableOrm;
        if (disableOrm) {
            NSLog(@"[glTF] ORM bypass enabled for material '%s'", materials[i].name.c_str());
        }
    }

    std::vector<uint32_t> materialMap(materials.size(), 0u);
    for (size_t i = 0; i < materials.size(); ++i) {
        MaterialData material = BuildGltfMaterial(materials[i],
                                                 path,
                                                 textures,
                                                 samplers,
                                                 images,
                                                 buffers,
                                                 views,
                                                 loadOptions,
                                                 resources);
        materialMap[i] = resources.addMaterialData(material, materials[i].name);
    }

    NSArray* meshArray = root[@"meshes"];
    if (meshArray && [meshArray isKindOfClass:[NSArray class]]) {
        meshes.resize(meshArray.count);
        for (NSUInteger i = 0; i < meshArray.count; ++i) {
            NSDictionary* mesh = meshArray[i];
            meshes[i].name = GetString(mesh, @"name");
            NSArray* prims = mesh[@"primitives"];
            if (![prims isKindOfClass:[NSArray class]]) {
                continue;
            }
            for (NSDictionary* prim in prims) {
                GltfPrimitive p{};
                p.material = GetInt(prim, @"material", -1);
                p.indexAccessor = GetInt(prim, @"indices", -1);
                p.mode = GetInt(prim, @"mode", 4);

                NSDictionary* attrs = prim[@"attributes"];
                if ([attrs isKindOfClass:[NSDictionary class]]) {
                    id pos = attrs[@"POSITION"];
                    if ([pos isKindOfClass:[NSNumber class]]) {
                        p.positionAccessor = [(NSNumber*)pos intValue];
                    }
                    id nrm = attrs[@"NORMAL"];
                    if ([nrm isKindOfClass:[NSNumber class]]) {
                        p.normalAccessor = [(NSNumber*)nrm intValue];
                    }
                    id uv = attrs[@"TEXCOORD_0"];
                    if ([uv isKindOfClass:[NSNumber class]]) {
                        p.texcoordAccessor = [(NSNumber*)uv intValue];
                    }
                    id uv1 = attrs[@"TEXCOORD_1"];
                    if ([uv1 isKindOfClass:[NSNumber class]]) {
                        p.texcoord1Accessor = [(NSNumber*)uv1 intValue];
                    }
                    id tan = attrs[@"TANGENT"];
                    if ([tan isKindOfClass:[NSNumber class]]) {
                        p.tangentAccessor = [(NSNumber*)tan intValue];
                    }
                }
                meshes[i].primitives.push_back(p);
            }
        }
    }

    NSArray* cameraArray = root[@"cameras"];
    if (cameraArray && [cameraArray isKindOfClass:[NSArray class]]) {
        cameras.resize(cameraArray.count);
        for (NSUInteger i = 0; i < cameraArray.count; ++i) {
            NSDictionary* cam = cameraArray[i];
            NSString* typeStr = cam[@"type"];
            if (![typeStr isKindOfClass:[NSString class]]) {
                continue;
            }
            if ([(NSString*)typeStr isEqualToString:@"perspective"]) {
                cameras[i].isPerspective = true;
                NSDictionary* persp = cam[@"perspective"];
                if ([persp isKindOfClass:[NSDictionary class]]) {
                    id yfov = persp[@"yfov"];
                    if ([yfov isKindOfClass:[NSNumber class]]) {
                        cameras[i].yfov = [(NSNumber*)yfov floatValue];
                    }
                    id znear = persp[@"znear"];
                    if ([znear isKindOfClass:[NSNumber class]]) {
                        cameras[i].znear = [(NSNumber*)znear floatValue];
                    }
                    id zfar = persp[@"zfar"];
                    if ([zfar isKindOfClass:[NSNumber class]]) {
                        cameras[i].zfar = [(NSNumber*)zfar floatValue];
                    }
                    id aspect = persp[@"aspectRatio"];
                    if ([aspect isKindOfClass:[NSNumber class]]) {
                        cameras[i].aspect = [(NSNumber*)aspect floatValue];
                    }
                }
            } else if ([(NSString*)typeStr isEqualToString:@"orthographic"]) {
                cameras[i].isPerspective = false;
                NSDictionary* ortho = cam[@"orthographic"];
                if ([ortho isKindOfClass:[NSDictionary class]]) {
                    id znear = ortho[@"znear"];
                    if ([znear isKindOfClass:[NSNumber class]]) {
                        cameras[i].znear = [(NSNumber*)znear floatValue];
                    }
                    id zfar = ortho[@"zfar"];
                    if ([zfar isKindOfClass:[NSNumber class]]) {
                        cameras[i].zfar = [(NSNumber*)zfar floatValue];
                    }
                }
            }
        }
    }

    NSArray* nodeArray = root[@"nodes"];
    if (nodeArray && [nodeArray isKindOfClass:[NSArray class]]) {
        nodes.resize(nodeArray.count);
        for (NSUInteger i = 0; i < nodeArray.count; ++i) {
            NSDictionary* node = nodeArray[i];
            nodes[i].name = GetString(node, @"name");
            nodes[i].mesh = GetInt(node, @"mesh", -1);
            nodes[i].camera = GetInt(node, @"camera", -1);

            std::vector<float> translation;
            if (GetFloatArray(node, @"translation", translation) && translation.size() >= 3) {
                nodes[i].translation = simd_make_float3(translation[0], translation[1], translation[2]);
            }
            std::vector<float> rotation;
            if (GetFloatArray(node, @"rotation", rotation) && rotation.size() >= 4) {
                nodes[i].rotation = simd_make_float4(rotation[0], rotation[1], rotation[2], rotation[3]);
            }
            std::vector<float> scale;
            if (GetFloatArray(node, @"scale", scale) && scale.size() >= 3) {
                nodes[i].scale = simd_make_float3(scale[0], scale[1], scale[2]);
            }
            std::vector<float> matrix;
            if (GetFloatArray(node, @"matrix", matrix) && matrix.size() >= 16) {
                nodes[i].hasMatrix = true;
                nodes[i].matrix.columns[0] = simd_make_float4(matrix[0], matrix[1], matrix[2], matrix[3]);
                nodes[i].matrix.columns[1] = simd_make_float4(matrix[4], matrix[5], matrix[6], matrix[7]);
                nodes[i].matrix.columns[2] = simd_make_float4(matrix[8], matrix[9], matrix[10], matrix[11]);
                nodes[i].matrix.columns[3] = simd_make_float4(matrix[12], matrix[13], matrix[14], matrix[15]);
            }

            NSArray* children = node[@"children"];
            if ([children isKindOfClass:[NSArray class]]) {
                for (NSNumber* childIdx in children) {
                    nodes[i].children.push_back(childIdx.intValue);
                }
            }
        }
    }

    NSArray* scenesArray = root[@"scenes"];
    int sceneIndex = GetInt(root, @"scene", 0);
    std::vector<int> sceneNodes;
    if (scenesArray && [scenesArray isKindOfClass:[NSArray class]] &&
        sceneIndex >= 0 && sceneIndex < static_cast<int>(scenesArray.count)) {
        NSDictionary* scene = scenesArray[sceneIndex];
        NSArray* nodeList = scene[@"nodes"];
        if ([nodeList isKindOfClass:[NSArray class]]) {
            for (NSNumber* idx in nodeList) {
                sceneNodes.push_back(idx.intValue);
            }
        }
    } else if (!nodes.empty()) {
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            sceneNodes.push_back(i);
        }
    }

    auto loadPrimitive = [&](const GltfPrimitive& prim,
                             const simd::float4x4& localToWorld,
                             const std::string& baseName) -> bool {
        if (prim.mode != 4) {
            return true;
        }
        if (prim.positionAccessor < 0 ||
            prim.positionAccessor >= static_cast<int>(accessors.size())) {
            errorMessage = "glTF primitive missing POSITION accessor";
            return false;
        }

        const GltfAccessor& posAcc = accessors[prim.positionAccessor];
        std::vector<float> positions;
        if (!ReadAccessorFloatN(buffers, views, posAcc, 3, positions)) {
            errorMessage = "Failed reading POSITION accessor";
            return false;
        }
        size_t vertexCount = posAcc.count;
        if (vertexCount == 0) {
            return true;
        }

        if (outCamera) {
            simd::float3 sceneMin = simd_make_float3(0.0f, 0.0f, 0.0f);
            simd::float3 sceneMax = simd_make_float3(0.0f, 0.0f, 0.0f);
            bool hasBounds = false;
            for (size_t i = 0; i < vertexCount; ++i) {
                simd::float3 localPos = simd_make_float3(positions[i * 3 + 0],
                                                        positions[i * 3 + 1],
                                                        positions[i * 3 + 2]);
                simd::float4 worldPos4 = simd_mul(localToWorld, simd_make_float4(localPos, 1.0f));
                simd::float3 worldPos = simd_make_float3(worldPos4.x, worldPos4.y, worldPos4.z);
                if (!hasBounds) {
                    sceneMin = worldPos;
                    sceneMax = worldPos;
                    hasBounds = true;
                } else {
                    sceneMin = simd_min(sceneMin, worldPos);
                    sceneMax = simd_max(sceneMax, worldPos);
                }
            }
            if (hasBounds) {
                if (!outCamera->hasSceneBounds) {
                    outCamera->sceneCenter = (sceneMin + sceneMax) * 0.5f;
                    simd::float3 ext = sceneMax - sceneMin;
                    outCamera->sceneRadius = simd::length(ext) * 0.5f;
                    outCamera->hasSceneBounds = true;
                } else {
                    simd::float3 minC = outCamera->sceneCenter - simd_make_float3(outCamera->sceneRadius);
                    simd::float3 maxC = outCamera->sceneCenter + simd_make_float3(outCamera->sceneRadius);
                    minC = simd_min(minC, sceneMin);
                    maxC = simd_max(maxC, sceneMax);
                    outCamera->sceneCenter = (minC + maxC) * 0.5f;
                    simd::float3 ext = maxC - minC;
                    outCamera->sceneRadius = simd::length(ext) * 0.5f;
                }
            }
        }

        std::vector<float> normals;
        bool hasNormals = false;
        if (prim.normalAccessor >= 0 && prim.normalAccessor < static_cast<int>(accessors.size())) {
            const GltfAccessor& nAcc = accessors[prim.normalAccessor];
            hasNormals = ReadAccessorFloatN(buffers, views, nAcc, 3, normals);
        }

        std::vector<float> uvs;
        bool hasUvs = false;
        if (prim.texcoordAccessor >= 0 && prim.texcoordAccessor < static_cast<int>(accessors.size())) {
            const GltfAccessor& uvAcc = accessors[prim.texcoordAccessor];
            hasUvs = ReadAccessorFloatN(buffers, views, uvAcc, 2, uvs);
        }
        std::vector<float> uvs1;
        bool hasUvs1 = false;
        if (prim.texcoord1Accessor >= 0 && prim.texcoord1Accessor < static_cast<int>(accessors.size())) {
            const GltfAccessor& uv1Acc = accessors[prim.texcoord1Accessor];
            hasUvs1 = ReadAccessorFloatN(buffers, views, uv1Acc, 2, uvs1);
        }
        static bool sLoggedUvProbe = false;
        if (!sLoggedUvProbe && hasUvs && !uvs.empty()) {
            float minU = std::numeric_limits<float>::infinity();
            float minV = std::numeric_limits<float>::infinity();
            float maxU = -std::numeric_limits<float>::infinity();
            float maxV = -std::numeric_limits<float>::infinity();
            for (size_t uvIdx = 0; uvIdx + 1 < uvs.size(); uvIdx += 2) {
                float u = uvs[uvIdx + 0];
                float v = uvs[uvIdx + 1];
                if (!std::isfinite(u) || !std::isfinite(v)) {
                    continue;
                }
                minU = std::min(minU, u);
                minV = std::min(minV, v);
                maxU = std::max(maxU, u);
                maxV = std::max(maxV, v);
            }
            int mrTexIndex = -1;
            int mrTexCoord = 0;
            int wrapS = 10497;
            int wrapT = 10497;
            if (prim.material >= 0 && prim.material < static_cast<int>(materials.size())) {
                const GltfMaterial& primMaterial = materials[prim.material];
                mrTexIndex = primMaterial.metallicRoughnessTexture.index;
                mrTexCoord = primMaterial.metallicRoughnessTexture.texCoord;
                if (mrTexIndex >= 0 && mrTexIndex < static_cast<int>(textures.size())) {
                    int samplerIndex = textures[mrTexIndex].sampler;
                    if (samplerIndex >= 0 && samplerIndex < static_cast<int>(samplers.size())) {
                        wrapS = samplers[samplerIndex].wrapS;
                        wrapT = samplers[samplerIndex].wrapT;
                    }
                }
            }
            NSLog(@"[glTF][Probe] UV0 bounds u=[%.6f, %.6f] v=[%.6f, %.6f] "
                  @"mrTex=%d texCoord=%d wrapS=%d wrapT=%d",
                  minU,
                  maxU,
                  minV,
                  maxV,
                  mrTexIndex,
                  mrTexCoord,
                  wrapS,
                  wrapT);
            sLoggedUvProbe = true;
        }

        std::vector<float> tangents;
        bool hasTangents = false;
        if (prim.tangentAccessor >= 0 && prim.tangentAccessor < static_cast<int>(accessors.size())) {
            const GltfAccessor& tAcc = accessors[prim.tangentAccessor];
            hasTangents = ReadAccessorFloatN(buffers, views, tAcc, 4, tangents);
        }

        std::vector<uint32_t> indices;
        if (prim.indexAccessor >= 0 && prim.indexAccessor < static_cast<int>(accessors.size())) {
            const GltfAccessor& iAcc = accessors[prim.indexAccessor];
            if (!ReadAccessorIndices(buffers, views, iAcc, indices)) {
                errorMessage = "Failed reading indices accessor";
                return false;
            }
        } else {
            indices.resize(vertexCount);
            for (size_t i = 0; i < vertexCount; ++i) {
                indices[i] = static_cast<uint32_t>(i);
            }
        }

        std::vector<SceneResources::MeshVertex> vertices(vertexCount);
        for (size_t i = 0; i < vertexCount; ++i) {
            SceneResources::MeshVertex v{};
            v.position = simd_make_float3(positions[i * 3 + 0],
                                          positions[i * 3 + 1],
                                          positions[i * 3 + 2]);
            if (hasNormals) {
                v.normal = simd_make_float3(normals[i * 3 + 0],
                                            normals[i * 3 + 1],
                                            normals[i * 3 + 2]);
            }
            if (hasUvs) {
                v.uv = simd_make_float2(uvs[i * 2 + 0],
                                        uvs[i * 2 + 1]);
            }
            if (hasUvs1) {
                v.uv1 = simd_make_float2(uvs1[i * 2 + 0],
                                         uvs1[i * 2 + 1]);
            } else {
                v.uv1 = v.uv;
            }
            if (hasTangents) {
                v.tangent = simd_make_float4(tangents[i * 4 + 0],
                                             tangents[i * 4 + 1],
                                             tangents[i * 4 + 2],
                                             tangents[i * 4 + 3]);
            }
            vertices[i] = v;
        }

        if (!hasNormals) {
            std::vector<simd::float3> normalAccum(vertexCount, simd_make_float3(0.0f, 0.0f, 0.0f));
            for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                uint32_t i0 = indices[i + 0];
                uint32_t i1 = indices[i + 1];
                uint32_t i2 = indices[i + 2];
                if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) {
                    continue;
                }
                simd::float3 p0 = vertices[i0].position;
                simd::float3 p1 = vertices[i1].position;
                simd::float3 p2 = vertices[i2].position;
                simd::float3 n = simd::cross(p1 - p0, p2 - p0);
                if (simd::length(n) > 0.0f) {
                    normalAccum[i0] += n;
                    normalAccum[i1] += n;
                    normalAccum[i2] += n;
                }
            }
            for (size_t i = 0; i < vertexCount; ++i) {
                if (simd::length(normalAccum[i]) > 0.0f) {
                    vertices[i].normal = simd::normalize(normalAccum[i]);
                }
            }
        }

        bool generatedTangents = false;
        if (!hasTangents && hasUvs) {
            GenerateTangents(vertices, indices);
            generatedTangents = true;
        }
        uint32_t materialIndex = 0u;
        if (prim.material >= 0 && prim.material < static_cast<int>(materialMap.size())) {
            materialIndex = materialMap[prim.material];
        }

        resources.addMesh(vertices.data(),
                          static_cast<uint32_t>(vertices.size()),
                          indices.data(),
                          static_cast<uint32_t>(indices.size()),
                          localToWorld,
                          materialIndex,
                          baseName);
        return true;
    };

    std::function<bool(int, const simd::float4x4&, const std::string&)> traverse;
    traverse = [&](int nodeIndex, const simd::float4x4& parent, const std::string& prefix) -> bool {
        if (nodeIndex < 0 || nodeIndex >= static_cast<int>(nodes.size())) {
            return true;
        }
        const GltfNode& node = nodes[nodeIndex];
        simd::float4x4 local = node.hasMatrix ? node.matrix : ComposeTrs(node.translation, node.rotation, node.scale);
        simd::float4x4 world = simd_mul(parent, local);
        std::string nodeName = prefix;
        if (!node.name.empty()) {
            if (!nodeName.empty()) {
                nodeName += "/";
            }
            nodeName += node.name;
        }
        if (node.camera >= 0 && node.camera < static_cast<int>(cameras.size()) &&
            outCamera && !outCamera->valid) {
            simd::float4 worldPos4 = simd_mul(world, simd_make_float4(0.0f, 0.0f, 0.0f, 1.0f));
            simd::float3 worldPos = simd_make_float3(worldPos4.x, worldPos4.y, worldPos4.z);
            simd::float4 forward4 = simd_mul(world, simd_make_float4(0.0f, 0.0f, -1.0f, 0.0f));
            simd::float4 up4 = simd_mul(world, simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f));
            simd::float3 forward = simd_normalize(simd_make_float3(forward4.x, forward4.y, forward4.z));
            simd::float3 up = simd_normalize(simd_make_float3(up4.x, up4.y, up4.z));
            const GltfCamera& cam = cameras[node.camera];
            outCamera->valid = true;
            outCamera->hasPerspective = cam.isPerspective;
            outCamera->yfov = cam.yfov;
            outCamera->znear = cam.znear;
            outCamera->zfar = cam.zfar;
            outCamera->position = worldPos;
            outCamera->forward = forward;
            outCamera->up = up;
        }

        if (node.mesh >= 0 && node.mesh < static_cast<int>(meshes.size())) {
            const GltfMesh& mesh = meshes[node.mesh];
            for (size_t primIndex = 0; primIndex < mesh.primitives.size(); ++primIndex) {
                std::string meshName = nodeName;
                if (!mesh.name.empty()) {
                    if (!meshName.empty()) {
                        meshName += "/";
                    }
                    meshName += mesh.name;
                }
                if (mesh.primitives.size() > 1) {
                    meshName += ".prim" + std::to_string(primIndex);
                }
                if (!loadPrimitive(mesh.primitives[primIndex], world, meshName)) {
                    return false;
                }
            }
        }
        for (int child : node.children) {
            if (!traverse(child, world, nodeName)) {
                return false;
            }
        }
        return true;
    };

    simd::float4x4 identity = matrix_identity_float4x4;
    for (int nodeIndex : sceneNodes) {
        if (!traverse(nodeIndex, identity, "")) {
            return false;
        }
    }

    return true;
}

}  // namespace PathTracer
