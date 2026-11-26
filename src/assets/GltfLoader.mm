#import "assets/GltfLoader.h"

#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cstdint>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <simd/simd.h>

#include "MetalShaderTypes.h"
#include "assets/TangentGen.h"
#include "renderer/SceneResources.h"
#import "renderer/LoadEventPump.h"

namespace fs = std::filesystem;

namespace PathTracer {

namespace {

using PathTracerShaderTypes::MaterialType;
using PathTracerShaderTypes::MaterialData;

constexpr uint32_t kInvalidTextureIndex = 0xFFFFFFFFu;

using LoadEventPump::PumpEventsEvery;

bool TransformHasNegativeDeterminant(const simd::float4x4& transform) {
    const simd::float3x3 linear = simd_matrix(transform.columns[0].xyz,
                                              transform.columns[1].xyz,
                                              transform.columns[2].xyz);
    return simd_determinant(linear) < 0.0f;
}

void ReverseTriangleWindingForMirroredTransform(std::vector<uint32_t>& indices,
                                                const simd::float4x4& localToWorld) {
    if (!TransformHasNegativeDeterminant(localToWorld)) {
        return;
    }
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        std::swap(indices[i + 1], indices[i + 2]);
    }
}

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
    // The foliage texture-name rule made this material a double-sided MASK at parse time; the build loop keeps that
    // only if the uploaded base-colour texture has alpha holes, else it restores the authored values below.
    bool foliageTextureNameCandidate = false;
    std::string authoredAlphaMode;
    float authoredAlphaCutoff = 0.5f;
    bool authoredDoubleSided = false;
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
    bool forceBaseColorAlphaMask = false;
    bool thinFoliageShadow = false;
    bool alphaCutoutContinuation = false;
    bool foliageBeautyLift = false;
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

struct GltfLight {
    std::string name;
    std::string type;
    simd::float3 color = simd_make_float3(1.0f, 1.0f, 1.0f);
    float intensity = 1.0f;
};

struct GltfNode {
    std::string name;
    int mesh = -1;
    int camera = -1;
    int light = -1;
    std::vector<int> children;
    simd::float3 translation = simd_make_float3(0.0f, 0.0f, 0.0f);
    simd::float4 rotation = simd_make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    simd::float3 scale = simd_make_float3(1.0f, 1.0f, 1.0f);
    bool hasMatrix = false;
    simd::float4x4 matrix = matrix_identity_float4x4;
};

struct GltfAnimationSampler {
    int input = -1;
    int output = -1;
    std::string interpolation = "LINEAR";
    std::vector<float> times;
    std::vector<float> values;
    size_t components = 0;
};

struct GltfAnimationChannel {
    int sampler = -1;
    int targetNode = -1;
    std::string targetPath;
};

struct GltfAnimation {
    std::string name;
    std::vector<GltfAnimationSampler> samplers;
    std::vector<GltfAnimationChannel> channels;
    float startTime = 0.0f;
    float endTime = 0.0f;
    bool hasTimeRange = false;
};

struct GltfAnimatedNodeTransform {
    bool hasTranslation = false;
    bool hasRotation = false;
    bool hasScale = false;
    simd::float3 translation = simd_make_float3(0.0f, 0.0f, 0.0f);
    simd::float4 rotation = simd_make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    simd::float3 scale = simd_make_float3(1.0f, 1.0f, 1.0f);
};

struct PrimitiveTask {
    uint32_t meshIndex = 0u;
    uint32_t primitiveIndex = 0u;
    simd::float4x4 localToWorld = matrix_identity_float4x4;
    simd::float4x4 sourceLocalTransform = matrix_identity_float4x4;
    std::string baseName;
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

std::string HexFnv1a64(const std::string& text) {
    uint64_t hash = 1469598103934665603ull;
    for (const char rawC : text) {
        const unsigned char c = static_cast<unsigned char>(rawC);
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ull;
    }
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(16, '0');
    for (int i = 15; i >= 0; --i) {
        out[static_cast<size_t>(i)] = kHex[hash & 0xFu];
        hash >>= 4u;
    }
    return out;
}

std::string StableAssetId(const std::string& type, const std::string& source) {
    return "gltf:" + type + ":" + HexFnv1a64(source);
}

void AddSupportMatrixEntry(AssetPipelineReport& report,
                           std::string area,
                           std::string status,
                           std::string details) {
    AssetSupportMatrixEntry entry{};
    entry.area = std::move(area);
    entry.status = std::move(status);
    entry.details = std::move(details);
    report.supportMatrix.push_back(std::move(entry));
}

void AddStableResourceId(AssetPipelineReport& report,
                         std::string resourceType,
                         std::string name,
                         int32_t sourceIndex,
                         std::string stableId) {
    AssetStableResourceId entry{};
    entry.resourceType = std::move(resourceType);
    entry.name = std::move(name);
    entry.sourceIndex = sourceIndex;
    entry.stableId = std::move(stableId);
    report.stableResourceIds.push_back(std::move(entry));
}

void AddMaterialOverrideRecord(AssetPipelineReport& report,
                               std::string target,
                               std::string action,
                               std::string reason) {
    AssetMaterialOverrideRecord entry{};
    entry.target = std::move(target);
    entry.action = std::move(action);
    entry.reason = std::move(reason);
    report.materialOverrides.push_back(std::move(entry));
}

const char* TextureBudgetPolicyLabel(TextureBudgetPolicy policy) {
    switch (policy) {
        case TextureBudgetPolicy::Strict: return "strict";
        case TextureBudgetPolicy::Warn: return "warn";
        case TextureBudgetPolicy::Ignore: return "ignore";
    }
    return "strict";
}

const char* PilotModeLabel(PilotMode mode) {
    switch (mode) {
        case PilotMode::FullClamped: return "full_clamped";
        case PilotMode::GeoOnly: return "geo_only";
        case PilotMode::AlbedoOnly: return "albedo_only";
    }
    return "full_clamped";
}

bool MaterialNameSuggestsDoubleSided(const std::string& materialName) {
    return ContainsCaseInsensitive(materialName, ".doublesided") ||
           ContainsCaseInsensitive(materialName, "_doublesided");
}

bool MaterialNameSuggestsMaskedFoliage(const std::string& materialName) {
    if (!(ContainsCaseInsensitive(materialName, "foliage") ||
          ContainsCaseInsensitive(materialName, "leaf") ||
          ContainsCaseInsensitive(materialName, "leaves") ||
          ContainsCaseInsensitive(materialName, "ivy") ||
          ContainsCaseInsensitive(materialName, "flower") ||
          ContainsCaseInsensitive(materialName, "flowers") ||
          ContainsCaseInsensitive(materialName, "bux"))) {
        return false;
    }
    return MaterialNameSuggestsDoubleSided(materialName) ||
           ContainsCaseInsensitive(materialName, "masked");
}

bool MaterialNameSuggestsMaskedSignage(const std::string& materialName) {
    return ContainsCaseInsensitive(materialName, "sign") ||
           ContainsCaseInsensitive(materialName, "logo") ||
           ContainsCaseInsensitive(materialName, "shop") ||
           ContainsCaseInsensitive(materialName, "cafe") ||
           ContainsCaseInsensitive(materialName, "coin") ||
           ContainsCaseInsensitive(materialName, "petit") ||
           ContainsCaseInsensitive(materialName, "decal");
}

bool MaterialNameSuggestsSparseAtlasSignage(const std::string& materialName) {
    return ContainsCaseInsensitive(materialName, "sign") ||
           ContainsCaseInsensitive(materialName, "banner") ||
           ContainsCaseInsensitive(materialName, "menu") ||
           ContainsCaseInsensitive(materialName, "shop") ||
           ContainsCaseInsensitive(materialName, "letters") ||
           ContainsCaseInsensitive(materialName, "text") ||
           ContainsCaseInsensitive(materialName, "coin") ||
           ContainsCaseInsensitive(materialName, "petit");
}

bool MaterialNameLooksLikeBulkF1LiveryAtlas(const std::string& materialName) {
    return ContainsCaseInsensitive(materialName, "_badges") ||
           ContainsCaseInsensitive(materialName, "_ext_");
}

bool MaterialNameNeedsBlackKeyMask(const std::string& materialName) {
    return ContainsCaseInsensitive(materialName, "master_side_letters");
}

bool MaterialNameNeedsUv1Override(const std::string& materialName) {
    return ContainsCaseInsensitive(materialName, "master_wood_painted3.002");
}

bool MaterialNameLooksLikeBistroPlaceholderOverlay(const std::string& materialName) {
    return ContainsCaseInsensitive(materialName, "bistro_sign_letters");
}

bool MaterialNameShouldHideBistroSideLetters(const std::string& materialName) {
    return ContainsCaseInsensitive(materialName, "master_side_letters");
}

struct BistroPracticalEmission {
    simd::float3 emission;
    simd::float3 baseColor;
};

constexpr const char* kBistroPracticalEmissionAction = "bistro_practical_emission";

std::optional<BistroPracticalEmission> BistroPracticalEmissionForMaterial(const std::string& materialName) {
    if (ContainsCaseInsensitive(materialName, "master_light_bulb")) {
        return BistroPracticalEmission{simd_make_float3(12.0f, 7.2f, 3.8f),
                                       simd_make_float3(1.0f, 1.0f, 1.0f)};
    }
    if (ContainsCaseInsensitive(materialName, "paris_stringlights_01_") &&
        ContainsCaseInsensitive(materialName, "color_emissive")) {
        if (ContainsCaseInsensitive(materialName, "orange")) {
            return BistroPracticalEmission{simd_make_float3(310.0f, 46.0f, 3.2f),
                                           simd_make_float3(1.0f, 0.22f, 0.03f)};
        }
        if (ContainsCaseInsensitive(materialName, "blue")) {
            return BistroPracticalEmission{simd_make_float3(0.28f, 2.0f, 32.0f),
                                           simd_make_float3(0.03f, 0.12f, 1.0f)};
        }
        if (ContainsCaseInsensitive(materialName, "pink") ||
            ContainsCaseInsensitive(materialName, "violet")) {
            return BistroPracticalEmission{simd_make_float3(58.0f, 0.75f, 58.0f),
                                           simd_make_float3(0.95f, 0.04f, 1.0f)};
        }
        if (ContainsCaseInsensitive(materialName, "red")) {
            return BistroPracticalEmission{simd_make_float3(340.0f, 1.4f, 0.28f),
                                           simd_make_float3(1.0f, 0.03f, 0.015f)};
        }
        if (ContainsCaseInsensitive(materialName, "green")) {
            return BistroPracticalEmission{simd_make_float3(0.18f, 34.0f, 1.3f),
                                           simd_make_float3(0.02f, 1.0f, 0.08f)};
        }
        if (ContainsCaseInsensitive(materialName, "white")) {
            return BistroPracticalEmission{simd_make_float3(0.0f, 0.0f, 0.0f),
                                           simd_make_float3(0.015f, 0.015f, 0.012f)};
        }
    }
    if (ContainsCaseInsensitive(materialName, "spotlight_glass_emissive") ||
        ContainsCaseInsensitive(materialName, "spotlight_emissive")) {
        return BistroPracticalEmission{simd_make_float3(8.0f, 4.8f, 2.5f),
                                       simd_make_float3(1.0f, 1.0f, 1.0f)};
    }
    if (ContainsCaseInsensitive(materialName, "paris_wall_light_interior") ||
        ContainsCaseInsensitive(materialName, "paris_wall_bulb_light")) {
        return BistroPracticalEmission{simd_make_float3(8.5f, 5.0f, 2.6f),
                                       simd_make_float3(1.0f, 1.0f, 1.0f)};
    }
    if (ContainsCaseInsensitive(materialName, "emissive_streetlight") ||
        ContainsCaseInsensitive(materialName, "streetlight_support_bulb")) {
        return BistroPracticalEmission{simd_make_float3(13.0f, 7.6f, 3.9f),
                                       simd_make_float3(1.0f, 1.0f, 1.0f)};
    }
    return std::nullopt;
}

bool BistroStringLightCableMaterial(const std::string& materialName) {
    if (ContainsCaseInsensitive(materialName, "stringlights") &&
        !ContainsCaseInsensitive(materialName, "paris_stringlights_01_")) {
        return true;
    }
    if (ContainsCaseInsensitive(materialName, "lanternemissive") ||
        ContainsCaseInsensitive(materialName, "lantern_cord_dark_nonemissive")) {
        return true;
    }
    return ContainsCaseInsensitive(materialName, "paris_stringlights_01_white_color") &&
           !ContainsCaseInsensitive(materialName, "emissive");
}

bool BistroColoredStringLightEmitterMaterial(const std::string& materialName) {
    return ContainsCaseInsensitive(materialName, "paris_stringlights_01_") &&
           ContainsCaseInsensitive(materialName, "color_emissive") &&
           !ContainsCaseInsensitive(materialName, "white");
}

bool PathLooksLikeBistroAsset(const std::string& path) {
    return ContainsCaseInsensitive(path, "canonical/bistro") ||
           ContainsCaseInsensitive(path, "bistro_openfront") ||
           ContainsCaseInsensitive(path, "/bistro/");
}

// Only for the warning when a Pavilion-looking file loads without gltfBarcelonaPavilionRules; the rules follow the token.
bool PathLooksLikeBarcelonaPavilionAsset(const std::string& path) {
    return ContainsCaseInsensitive(path, "/barcelona_pavilion/") ||
           ContainsCaseInsensitive(path, "\\barcelona_pavilion\\") ||
           ContainsCaseInsensitive(path, "barcelona_pavilion_");
}

bool BarcelonaPavilionWaterMaterial(const std::string& materialName) {
    return ContainsCaseInsensitive(materialName, "water");
}

bool BarcelonaPavilionArchitecturalGlassMaterial(const std::string& materialName) {
    return ContainsCaseInsensitive(materialName, "glass_architectural");
}

bool PathLooksLikeSponzaAsset(const std::string& path) {
    return ContainsCaseInsensitive(path, "canonical/khronos/gltf/sponza") ||
           ContainsCaseInsensitive(path, "/sponza/");
}

bool PathLooksLikeBulkF1Asset(const std::string& path) {
    return ContainsCaseInsensitive(path, "/f1_bulk_15/") ||
           ContainsCaseInsensitive(path, "\\f1_bulk_15\\");
}

bool PathLooksLikeEmeraldSquareAsset(const std::string& path) {
    return ContainsCaseInsensitive(path, "/emerald_square_day/") ||
           ContainsCaseInsensitive(path, "\\emerald_square_day\\") ||
           ContainsCaseInsensitive(path, "/emerald_square_dusk/") ||
           ContainsCaseInsensitive(path, "\\emerald_square_dusk\\") ||
           ContainsCaseInsensitive(path, "emeraldsquare");
}

bool TextureBindingLooksLikeBistroSpecularWorkflowFallback(const std::string& gltfPath,
                                                           const GltfTextureBinding& binding,
                                                           const std::vector<GltfTexture>& textures,
                                                           const std::vector<GltfImage>& images) {
    if (!PathLooksLikeBistroAsset(gltfPath) || binding.index < 0 ||
        binding.index >= static_cast<int>(textures.size())) {
        return false;
    }
    const GltfTexture& texture = textures[static_cast<size_t>(binding.index)];
    if (texture.source < 0 || texture.source >= static_cast<int>(images.size())) {
        return false;
    }
    const GltfImage& image = images[static_cast<size_t>(texture.source)];
    return ContainsCaseInsensitive(image.uri, "specular") ||
           ContainsCaseInsensitive(image.name, "specular");
}

bool TextureBindingLooksLikeSponzaBlackKeyCutout(const std::string& gltfPath,
                                                 const GltfTextureBinding& binding,
                                                 const std::vector<GltfTexture>& textures,
                                                 const std::vector<GltfImage>& images) {
    if (!PathLooksLikeSponzaAsset(gltfPath) || binding.index < 0 ||
        binding.index >= static_cast<int>(textures.size())) {
        return false;
    }
    const GltfTexture& texture = textures[static_cast<size_t>(binding.index)];
    if (texture.source < 0 || texture.source >= static_cast<int>(images.size())) {
        return false;
    }
    const GltfImage& image = images[static_cast<size_t>(texture.source)];
    const std::string label = image.uri + " " + image.name;
    // Khronos Sponza stores these cutout decals as opaque JPGs with black-key
    // borders. Treat only the known decorative shield/lion base-color atlases
    // as masked so internal dark sculpting detail remains intact.
    return ContainsCaseInsensitive(label, "8750083169368950601") ||
           ContainsCaseInsensitive(label, "6772804448157695701");
}

bool TextureBindingImageSuggestsMaskedFoliage(const GltfTextureBinding& binding,
                                              const std::vector<GltfTexture>& textures,
                                              const std::vector<GltfImage>& images) {
    if (binding.index < 0 || binding.index >= static_cast<int>(textures.size())) {
        return false;
    }
    const GltfTexture& texture = textures[static_cast<size_t>(binding.index)];
    if (texture.source < 0 || texture.source >= static_cast<int>(images.size())) {
        return false;
    }
    const GltfImage& image = images[static_cast<size_t>(texture.source)];
    const std::string label = image.uri + " " + image.name;
    return ContainsCaseInsensitive(label, "foliage") ||
           ContainsCaseInsensitive(label, "leaf") ||
           ContainsCaseInsensitive(label, "leaves") ||
           ContainsCaseInsensitive(label, "hoja") ||
           ContainsCaseInsensitive(label, "ivy") ||
           ContainsCaseInsensitive(label, "petal") ||
           ContainsCaseInsensitive(label, "flower") ||
           ContainsCaseInsensitive(label, "flowers") ||
           ContainsCaseInsensitive(label, "lef");
}

bool TextureBindingImageLooksLikeEmeraldCanopyCards(const GltfTextureBinding& binding,
                                                    const std::vector<GltfTexture>& textures,
                                                    const std::vector<GltfImage>& images) {
    if (binding.index < 0 || binding.index >= static_cast<int>(textures.size())) {
        return false;
    }
    const GltfTexture& texture = textures[static_cast<size_t>(binding.index)];
    if (texture.source < 0 || texture.source >= static_cast<int>(images.size())) {
        return false;
    }
    const GltfImage& image = images[static_cast<size_t>(texture.source)];
    const std::string label = image.uri + " " + image.name;
    return ContainsCaseInsensitive(label, "cards_basecolor");
}

bool LabelLooksLikeEmeraldPlantCard(const std::string& label) {
    return ContainsCaseInsensitive(label, "hedge") ||
           ContainsCaseInsensitive(label, "fern") ||
           ContainsCaseInsensitive(label, "grass_blades") ||
           ContainsCaseInsensitive(label, "grass_blade");
}

bool TextureBindingImageLooksLikeEmeraldPlantCard(const GltfTextureBinding& binding,
                                                  const std::vector<GltfTexture>& textures,
                                                  const std::vector<GltfImage>& images) {
    if (binding.index < 0 || binding.index >= static_cast<int>(textures.size())) {
        return false;
    }
    const GltfTexture& texture = textures[static_cast<size_t>(binding.index)];
    if (texture.source < 0 || texture.source >= static_cast<int>(images.size())) {
        return false;
    }
    const GltfImage& image = images[static_cast<size_t>(texture.source)];
    const std::string label = image.uri + " " + image.name;
    return LabelLooksLikeEmeraldPlantCard(label);
}

bool EmeraldSquareMaterialLooksLikeFoliage(const GltfMaterial& material,
                                           const std::vector<GltfTexture>& textures,
                                           const std::vector<GltfImage>& images) {
    return MaterialNameSuggestsMaskedFoliage(material.name) ||
           ContainsCaseInsensitive(material.name, "foliage_ground_cover") ||
           LabelLooksLikeEmeraldPlantCard(material.name) ||
           (ContainsCaseInsensitive(material.name, "cards.doublesided") &&
            TextureBindingImageLooksLikeEmeraldCanopyCards(material.baseColorTexture,
                                                           textures,
                                                           images)) ||
           TextureBindingImageLooksLikeEmeraldPlantCard(material.baseColorTexture,
                                                        textures,
                                                        images) ||
           TextureBindingImageSuggestsMaskedFoliage(material.baseColorTexture, textures, images);
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

simd::float4 NormalizeQuat(const simd::float4& q) {
    const float len = simd::length(q);
    if (!(len > 0.0f) || !std::isfinite(len)) {
        return simd_make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    return q / len;
}

float WrapAnimationTime(float time, float start, float end) {
    const float duration = end - start;
    if (!(duration > 0.0f) || !std::isfinite(duration)) {
        return time;
    }
    float wrapped = std::fmod(time - start, duration);
    if (wrapped < 0.0f) {
        wrapped += duration;
    }
    return start + wrapped;
}

std::vector<float> SampleAnimationValue(const GltfAnimationSampler& sampler, float time) {
    std::vector<float> result(sampler.components, 0.0f);
    if (sampler.times.empty() || sampler.values.empty() || sampler.components == 0u) {
        return result;
    }

    const bool cubic = sampler.interpolation == "CUBICSPLINE";
    const size_t stride = sampler.components * (cubic ? 3u : 1u);
    auto copyValue = [&](size_t keyIndex) {
        const size_t base = keyIndex * stride + (cubic ? sampler.components : 0u);
        for (size_t c = 0; c < sampler.components; ++c) {
            if (base + c < sampler.values.size()) {
                result[c] = sampler.values[base + c];
            }
        }
    };

    if (time <= sampler.times.front() || sampler.times.size() == 1u) {
        copyValue(0u);
        return result;
    }
    if (time >= sampler.times.back()) {
        copyValue(sampler.times.size() - 1u);
        return result;
    }

    size_t right = 1u;
    while (right < sampler.times.size() && sampler.times[right] < time) {
        ++right;
    }
    const size_t left = right - 1u;
    if (sampler.interpolation == "STEP") {
        copyValue(left);
        return result;
    }

    const float t0 = sampler.times[left];
    const float t1 = sampler.times[right];
    const float denom = t1 - t0;
    const float alpha = (denom > 0.0f) ? std::clamp((time - t0) / denom, 0.0f, 1.0f) : 0.0f;
    const size_t leftBase = left * stride + (cubic ? sampler.components : 0u);
    const size_t rightBase = right * stride + (cubic ? sampler.components : 0u);
    for (size_t c = 0; c < sampler.components; ++c) {
        const float a = (leftBase + c < sampler.values.size()) ? sampler.values[leftBase + c] : 0.0f;
        const float b = (rightBase + c < sampler.values.size()) ? sampler.values[rightBase + c] : a;
        result[c] = a + (b - a) * alpha;
    }
    return result;
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

const char* AssetTextureSemanticLabel(MaterialTextureSemantic semantic) {
    switch (semantic) {
        case MaterialTextureSemantic::BaseColor: return "baseColor";
        case MaterialTextureSemantic::Orm: return "orm";
        case MaterialTextureSemantic::Normal: return "normal";
        case MaterialTextureSemantic::Occlusion: return "occlusion";
        case MaterialTextureSemantic::Emissive: return "emissive";
        case MaterialTextureSemantic::Transmission: return "transmission";
        case MaterialTextureSemantic::Generic:
        default: return "generic";
    }
}

void AddAssetDiagnostic(AssetPipelineReport& report,
                        std::string severity,
                        std::string category,
                        std::string message,
                        std::string assetPath = {},
                        std::string objectType = {},
                        std::string objectName = {},
                        int32_t objectIndex = -1) {
    AssetPipelineDiagnostic diagnostic{};
    diagnostic.severity = std::move(severity);
    diagnostic.category = std::move(category);
    diagnostic.message = std::move(message);
    diagnostic.assetPath = std::move(assetPath);
    diagnostic.objectType = std::move(objectType);
    diagnostic.objectName = std::move(objectName);
    diagnostic.objectIndex = objectIndex;
    if (diagnostic.severity == "error") {
        ++report.stats.errorCount;
    } else if (diagnostic.severity == "warning") {
        ++report.stats.warningCount;
    }
    report.diagnostics.push_back(std::move(diagnostic));
}

bool GltfExtensionSupportedForImport(const std::string& extension, const GltfLoadOptions& loadOptions) {
    return extension == "KHR_texture_transform" ||
           extension == "KHR_materials_transmission" ||
           extension == "KHR_materials_volume" ||
           extension == "KHR_materials_ior" ||
           extension == "KHR_materials_emissive_strength" ||
           (extension == "KHR_lights_punctual" && loadOptions.importPunctualLights);
}

void RecordUnsupportedExtensions(NSDictionary* root,
                                 AssetPipelineReport& report,
                                 const std::string& path,
                                 const GltfLoadOptions& loadOptions) {
    auto inspect = [&](NSString* key, bool required) {
        NSArray* extensions = root[key];
        if (![extensions isKindOfClass:[NSArray class]]) {
            return;
        }
        for (id value in extensions) {
            if (![value isKindOfClass:[NSString class]]) {
                continue;
            }
            std::string extension([(NSString*)value UTF8String]);
            if (GltfExtensionSupportedForImport(extension, loadOptions)) {
                continue;
            }
            ++report.stats.unsupportedExtensionCount;
            AddAssetDiagnostic(report,
                               required ? "error" : "warning",
                               "unsupported_extension",
                               std::string(required ? "Required" : "Used") +
                                   " glTF extension is not mapped by the importer: " + extension,
                               path,
                               "extension",
                               extension);
        }
    };
    inspect(@"extensionsRequired", true);
    inspect(@"extensionsUsed", false);
}

std::string GltfImageSourceLabel(const std::string& gltfPath,
                                 int imageIndex,
                                 const GltfImage& image) {
    if (!image.uri.empty()) {
        if (image.uri.rfind("data:", 0) == 0) {
            return gltfPath + ":image:" + std::to_string(imageIndex);
        }
        return (fs::path(gltfPath).parent_path() / image.uri).string();
    }
    if (image.bufferView >= 0) {
        return gltfPath + ":bufferView:" + std::to_string(image.bufferView);
    }
    return gltfPath + ":image:" + std::to_string(imageIndex);
}

void RecordTextureColorSpace(AssetPipelineReport* report,
                             const std::string& gltfPath,
                             int textureIndex,
                             int imageIndex,
                             const GltfImage* image,
                             bool srgb,
                             MaterialTextureSemantic semantic,
                             std::string status) {
    if (!report) {
        return;
    }
    AssetTextureColorSpace entry{};
    entry.textureIndex = textureIndex;
    entry.imageIndex = imageIndex;
    entry.semantic = AssetTextureSemanticLabel(semantic);
    entry.colorSpace = srgb ? "srgb" : "linear";
    entry.status = std::move(status);
    if (image) {
        entry.source = GltfImageSourceLabel(gltfPath, imageIndex, *image);
    }
    report->textureColorSpaces.push_back(std::move(entry));
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
    const auto& view = views[static_cast<size_t>(accessor.bufferView)];
    if (view.buffer < 0 || view.buffer >= static_cast<int>(buffers.size())) {
        return false;
    }
    const auto& buffer = buffers[static_cast<size_t>(view.buffer)];
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
        PumpEventsEvery(i);
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
        PumpEventsEvery(i);
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
                             SceneResources& resources,
                             AssetPipelineReport* report,
                             const std::string& materialName,
                             bool barcelonaPavilionRules) {
    if (textureIndex < 0 || textureIndex >= static_cast<int>(textures.size())) {
        if (textureIndex >= 0 && report) {
            RecordTextureColorSpace(report, gltfPath, textureIndex, -1, nullptr, srgb, semantic, "invalid_texture_index");
            AddAssetDiagnostic(*report,
                               "warning",
                               "missing_texture",
                               "Material references a texture index outside the glTF texture table",
                               gltfPath,
                               "material",
                               materialName);
        }
        return kInvalidTextureIndex;
    }
    int imageIndex = textures[static_cast<size_t>(textureIndex)].source;
    if (imageIndex < 0 || imageIndex >= static_cast<int>(images.size())) {
        RecordTextureColorSpace(report, gltfPath, textureIndex, imageIndex, nullptr, srgb, semantic, "invalid_image_index");
        if (report) {
            AddAssetDiagnostic(*report,
                               "warning",
                               "missing_texture",
                               "Texture references a missing image source",
                               gltfPath,
                               "texture",
                               std::to_string(textureIndex),
                               textureIndex);
        }
        return kInvalidTextureIndex;
    }
    const GltfTexture& texture = textures[static_cast<size_t>(textureIndex)];
    MaterialTextureSamplerDesc samplerDesc{};
    const MaterialTextureSamplerDesc* samplerDescPtr = nullptr;
    bool hasGltfSampler = false;
    if (texture.sampler >= 0 && texture.sampler < static_cast<int>(samplers.size())) {
        const GltfSampler& gltfSampler = samplers[static_cast<size_t>(texture.sampler)];
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
    if (barcelonaPavilionRules) {
        // The renderer that produced the Pavilion references uses its maximum
        // anisotropy. Keep narrow lily veins at the pool's grazing angles.
        samplerDesc.maxAnisotropy = 16u;
        samplerDescPtr = &samplerDesc;
    }
    if (PathLooksLikeEmeraldSquareAsset(gltfPath)) {
        // Falcor's G-buffer, path tracer and alpha tests fetch every Emerald
        // texture at mip 0; pixel jitter does the filtering.
        samplerDesc.magFilter = 9729;  // LINEAR
        samplerDesc.minFilter = 9729;  // LINEAR, not mipmapped
        samplerDescPtr = &samplerDesc;
    }
    const auto& image = images[static_cast<size_t>(imageIndex)];
    std::string error;
    RecordTextureColorSpace(report, gltfPath, textureIndex, imageIndex, &image, srgb, semantic, "classified");

    if (!image.uri.empty()) {
        std::vector<uint8_t> data;
        std::string mime;
        if (DecodeDataUri(image.uri, data, mime)) {
            std::string label = gltfPath + ":image:" + std::to_string(imageIndex);
            if (!image.name.empty()) {
                label += ":" + image.name;
            }
            const uint32_t resolved = resources.addMaterialTextureFromData(data.data(),
                                                                           data.size(),
                                                                           label,
                                                                           srgb,
                                                                           &error,
                                                                           samplerDescPtr,
                                                                           semantic);
            if (resolved == kInvalidTextureIndex && report) {
                AddAssetDiagnostic(*report,
                                   "warning",
                                   "missing_texture",
                                   error.empty() ? ("Failed to load embedded texture: " + label) : error,
                                   gltfPath,
                                   "material",
                                   materialName);
            }
            return resolved;
        }
        fs::path imagePath = fs::path(gltfPath).parent_path() / image.uri;
        const uint32_t resolved = resources.addMaterialTextureFromFile(imagePath.string(),
                                                                       srgb,
                                                                       &error,
                                                                       samplerDescPtr,
                                                                       semantic);
        if (resolved == kInvalidTextureIndex && report) {
            AddAssetDiagnostic(*report,
                               "warning",
                               "missing_texture",
                               error.empty() ? ("Failed to load texture: " + imagePath.string()) : error,
                               imagePath.string(),
                               "material",
                               materialName);
        }
        return resolved;
    }

    if (image.bufferView >= 0 && image.bufferView < static_cast<int>(views.size())) {
        const auto& view = views[static_cast<size_t>(image.bufferView)];
        if (view.buffer >= 0 && view.buffer < static_cast<int>(buffers.size())) {
            const auto& buffer = buffers[static_cast<size_t>(view.buffer)];
            size_t offset = view.offset;
            size_t length = view.length;
            if (offset + length <= buffer.data.size() && length > 0) {
                std::string label = gltfPath + ":image:" + std::to_string(imageIndex);
                if (!image.name.empty()) {
                    label += ":" + image.name;
                }
                const uint32_t resolved = resources.addMaterialTextureFromData(buffer.data.data() + offset,
                                                                              length,
                                                                              label,
                                                                              srgb,
                                                                              &error,
                                                                              samplerDescPtr,
                                                                              semantic);
                if (resolved == kInvalidTextureIndex && report) {
                    AddAssetDiagnostic(*report,
                                       "warning",
                                       "missing_texture",
                                       error.empty() ? ("Failed to load bufferView texture: " + label) : error,
                                       gltfPath,
                                       "material",
                                       materialName);
                }
                return resolved;
            }
        }
    }

    if (report) {
        AddAssetDiagnostic(*report,
                           "warning",
                           "missing_texture",
                           "Texture image has no loadable URI or bufferView payload",
                           gltfPath,
                           "image",
                           image.name,
                           imageIndex);
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
                               SceneResources& resources,
                               AssetPipelineReport* report) {
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
                                                MaterialTextureSemantic::BaseColor,
                                                resources,
                                                report,
                                                src.name,
                                                loadOptions.barcelonaPavilionRules);
    uint32_t mrIdx = ResolveTextureIndex(gltfPath, textures, samplers, images, buffers, views,
                                         src.metallicRoughnessTexture.index, false,
                                         MaterialTextureSemantic::Orm,
                                         resources,
                                         report,
                                         src.name,
                                         loadOptions.barcelonaPavilionRules);
    uint32_t normalIdx = ResolveTextureIndex(gltfPath, textures, samplers, images, buffers, views,
                                             src.normalTexture.index, false,
                                             MaterialTextureSemantic::Normal,
                                             resources,
                                             report,
                                             src.name,
                                             loadOptions.barcelonaPavilionRules);
    MaterialTextureSemantic occlusionSemantic = MaterialTextureSemantic::Occlusion;
    if (src.occlusionTexture.index >= 0 &&
        src.occlusionTexture.index == src.metallicRoughnessTexture.index) {
        occlusionSemantic = MaterialTextureSemantic::Orm;
    }
    uint32_t occlusionIdx = ResolveTextureIndex(gltfPath, textures, samplers, images, buffers, views,
                                                src.occlusionTexture.index, false,
                                                occlusionSemantic,
                                                resources,
                                                report,
                                                src.name,
                                                loadOptions.barcelonaPavilionRules);
    uint32_t emissiveIdx = ResolveTextureIndex(gltfPath, textures, samplers, images, buffers, views,
                                               src.emissiveTexture.index, emissiveSrgb,
                                               MaterialTextureSemantic::Emissive,
                                               resources,
                                               report,
                                               src.name,
                                               loadOptions.barcelonaPavilionRules);
    uint32_t transmissionIdx = ResolveTextureIndex(gltfPath, textures, samplers, images, buffers, views,
                                                   src.transmissionTexture.index, false,
                                                   MaterialTextureSemantic::Transmission,
                                                   resources,
                                                   report,
                                                   src.name,
                                                   loadOptions.barcelonaPavilionRules);
    const bool disableOrmBySpecularWorkflowFallback =
        TextureBindingLooksLikeBistroSpecularWorkflowFallback(gltfPath,
                                                              src.metallicRoughnessTexture,
                                                              textures,
                                                              images);
    if (disableOrmBySpecularWorkflowFallback) {
        mrIdx = kInvalidTextureIndex;
        if (src.occlusionTexture.index >= 0 &&
            src.occlusionTexture.index == src.metallicRoughnessTexture.index) {
            occlusionIdx = kInvalidTextureIndex;
        }
        if (report) {
            AddAssetDiagnostic(*report,
                               "warning",
                               "material_fallback",
                               "Specular-workflow ORM fallback disabled metallic-roughness texture for this material",
                               gltfPath,
                               "material",
                               src.name);
        }
    }
    if (mrIdx != kInvalidTextureIndex) {
        const auto& loadedTextures = resources.materialTextures();
        if (mrIdx < loadedTextures.size()) {
            id<MTLTexture> mrTexture = loadedTextures[mrIdx];
            const bool interactiveTextureClamp =
                resources.textureLoadPolicy().maxDimension > 0u;
            if (mrTexture && !interactiveTextureClamp) {
                if (mrTexture.mipmapLevelCount <= 1) {
                    NSLog(@"[glTF] Warning: ORM texture %u has mipCount=%lu (expected > 1)",
                          mrIdx,
                          static_cast<unsigned long>(mrTexture.mipmapLevelCount));
                } else {
                    NSLog(@"[glTF] ORM texture %u format=%lu mips=%lu",
                          mrIdx,
                          static_cast<unsigned long>(mrTexture.pixelFormat),
                          static_cast<unsigned long>(mrTexture.mipmapLevelCount));
                }
            }
        }
    }

    material.textureIndices0 = simd_make_uint4(baseColorIdx, mrIdx, normalIdx, occlusionIdx);
    material.textureIndices1 =
        simd_make_uint4(emissiveIdx, transmissionIdx, kInvalidTextureIndex, kInvalidTextureIndex);
    if (PathLooksLikeBistroAsset(gltfPath)) {
        if (BistroStringLightCableMaterial(src.name)) {
            material.baseColorRoughness.x = 0.015f;
            material.baseColorRoughness.y = 0.015f;
            material.baseColorRoughness.z = 0.012f;
            material.emission = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            material.textureIndices0.x = kInvalidTextureIndex;
            material.textureIndices1.x = kInvalidTextureIndex;
        }
        if (std::optional<BistroPracticalEmission> practicalEmission = BistroPracticalEmissionForMaterial(src.name)) {
            material.baseColorRoughness.x = practicalEmission->baseColor.x;
            material.baseColorRoughness.y = practicalEmission->baseColor.y;
            material.baseColorRoughness.z = practicalEmission->baseColor.z;
            // The table replaces the authored emissive factor, strength and texture; gltfEmissiveScale scales it
            // like any other glTF emission, so a scale of 0 leaves no glTF emitter on.
            const float practicalScale = std::max(loadOptions.emissiveScale, 0.0f);
            const simd::float3 scaledEmission = practicalEmission->emission * practicalScale;
            material.emission = simd_make_float4(scaledEmission, 0.0f);
            material.textureIndices0.x = kInvalidTextureIndex;
            material.textureIndices1.x = kInvalidTextureIndex;
            // The coloured string lights also carry kMaterialFlagCameraVisibleEmissionTint: seen directly by the camera
            // they show their base colour x 16 whenever their emission is non-zero. That display rule ignores the
            // scale's size; only a scale of 0 turns it off.
            NSLog(@"[glTF] Bistro practical light %s: calibrated emission %g/%g/%g x gltfEmissiveScale %g = %g/%g/%g%s",
                  src.name.c_str(),
                  practicalEmission->emission.x, practicalEmission->emission.y, practicalEmission->emission.z,
                  practicalScale, scaledEmission.x, scaledEmission.y, scaledEmission.z,
                  (BistroColoredStringLightEmitterMaterial(src.name) && practicalScale > 0.0f)
                      ? "; seen directly by the camera it shows its base colour x 16, which the scale does not change"
                      : "");
            if (report) {
                AddMaterialOverrideRecord(*report, src.name, kBistroPracticalEmissionAction,
                    "Calibrated emission " + std::to_string(practicalEmission->emission.x) + "/" +
                    std::to_string(practicalEmission->emission.y) + "/" +
                    std::to_string(practicalEmission->emission.z) + " x gltfEmissiveScale " +
                    std::to_string(practicalScale) + " replaces the authored emissive, its strength and its texture");
            }
        }
    }
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
    float roughnessFactor = std::clamp(src.roughnessFactor * roughnessScale, 0.0f, 1.0f);
    if (loadOptions.roughnessOverride >= 0.0f) {
        roughnessFactor = std::clamp(loadOptions.roughnessOverride, 0.0f, 1.0f);
    }
    float transmissionFactor =
        std::clamp(src.transmissionFactor * std::max(loadOptions.transmissionScale, 0.0f), 0.0f, 1.0f);
    if (loadOptions.barcelonaPavilionRules) {
        if (BarcelonaPavilionWaterMaterial(src.name)) {
            roughnessFactor = 0.0f;
            transmissionFactor = 0.0f;
            material.baseColorRoughness.xyz =
                simd_clamp(src.baseColorFactor.xyz,
                           simd_make_float3(0.0f, 0.0f, 0.0f),
                           simd_make_float3(1.0f, 1.0f, 1.0f));
            material.typeEta.x = static_cast<float>(MaterialType::Dielectric);
            material.typeEta.y = std::clamp(src.hasIor ? src.ior : 1.333f, 1.0f, 3.0f);
            material.typeEta.w = 0.0f;
        } else if (BarcelonaPavilionArchitecturalGlassMaterial(src.name)) {
            roughnessFactor = 0.0f;
            transmissionFactor = 0.0f;
            material.baseColorRoughness.xyz =
                simd_make_float3(1.0f, 1.0f, 1.0f);
            material.typeEta.x = static_cast<float>(MaterialType::Dielectric);
            material.typeEta.y = std::clamp(src.hasIor ? src.ior : 1.5f, 1.0f, 3.0f);
            material.typeEta.w = 0.0f;
        } else {
            // Like the reference renderer, keep GGX roughness >= 0.01: a zero-roughness
            // PBR surface (the white "Material" ground) would otherwise be classified
            // as delta and skip direct lighting.
            roughnessFactor = std::max(roughnessFactor, 0.01f);
        }
        material.baseColorRoughness.w = roughnessFactor;
    }
    material.pbrParams = simd_make_float4(std::clamp(src.metallicFactor, 0.0f, 1.0f),
                                          roughnessFactor,
                                          occlusionStrength,
                                          std::max(src.normalScale * normalScale, 0.0f));
    const bool disableOrmTexture =
        src.disableOrmTexture || disableOrmBySpecularWorkflowFallback;
    if (disableOrmTexture) {
        material.materialFlags |= PathTracerShaderTypes::kMaterialFlagDisableOrm;
    }
    if (loadOptions.thinWalledTransmissionFallback && transmissionFactor > 0.0f &&
        material.typeEta.x == static_cast<float>(MaterialType::PbrMetallicRoughness) &&
        (!src.hasVolume || !(src.thicknessFactor > 0.0f))) {
        // KHR_materials_transmission without KHR_materials_volume (or with a zero
        // thicknessFactor, which KHR_materials_volume defines as thin-walled) is an
        // infinitely thin sheet: transmitted light passes straight through.
        material.materialFlags |= PathTracerShaderTypes::kMaterialFlagThinWalledTransmission;
    }
    if (loadOptions.referenceDielectricContinuation && transmissionFactor > 0.0f &&
        material.typeEta.x == static_cast<float>(MaterialType::PbrMetallicRoughness)) {
        // Path-Tracing (the reference renderer) cancels eta^2 between its BTDF and
        // PDF, so a refracted path keeps its throughput instead of scaling radiance
        // by 1/eta^2 on entry. Unpaired entries/exits (open or overlapping shells)
        // then match its references.
        material.materialFlags |= PathTracerShaderTypes::kMaterialFlagReferenceDielectricContinuation;
    }
    if ((PathLooksLikeEmeraldSquareAsset(gltfPath) || loadOptions.texturedEmissionNee) &&
        material.textureIndices1.x != kInvalidTextureIndex) {
        // Include authored textured emitters in texture-aware direct-light
        // sampling. Without this opt-in the emissive inventory excludes them;
        // matching surface-hit emission improves sampling without a BSDF change.
        material.materialFlags |= PathTracerShaderTypes::kMaterialFlagTexturedEmissionNee;
    }
    if (PathLooksLikeEmeraldSquareAsset(gltfPath)) {
        // Match Falcor's StandardMaterial, which rendered the references:
        // diffuse is Frostbite and GGX has no multiple-scattering compensation.
        material.materialFlags |= PathTracerShaderTypes::kMaterialFlagFalcorStandardBsdf;
        if (src.name == "Picnic_Table1") {
            // Per-material finding: this table's bevels match the Falcor
            // references only with its normal-map G negated. Not an ORCA-wide
            // rule: Sun Temple, imported the same way, matches with G kept.
            material.materialFlags |= PathTracerShaderTypes::kMaterialFlagSourceNormalGreen;
        }
    }
    if (PathLooksLikeEmeraldSquareAsset(gltfPath) && !src.doubleSided &&
        !ContainsCaseInsensitive(src.name, "polebanner")) {
        // Falcor, which rendered the Emerald references, traces without culling
        // and shades back faces of single-sided materials black. Pole banners are
        // two opposite-facing sheets a hair apart, so either may win the hit.
        material.materialFlags |= PathTracerShaderTypes::kMaterialFlagBlackBackFace;
    }
    if (loadOptions.barcelonaPavilionRules) {
        material.materialFlags |= PathTracerShaderTypes::kMaterialFlagSingleScatterSpecular;
    }
    if (loadOptions.barcelonaPavilionRules &&
        BarcelonaPavilionWaterMaterial(src.name)) {
        material.materialFlags |= PathTracerShaderTypes::kMaterialFlagDielectricTransmissionTint;
    }
    if (loadOptions.barcelonaPavilionRules &&
        (BarcelonaPavilionWaterMaterial(src.name) ||
         BarcelonaPavilionArchitecturalGlassMaterial(src.name))) {
        // Match the reference renderer's path-regularized water and glass (Metal).
        material.materialFlags |= PathTracerShaderTypes::kMaterialFlagRegularizedDielectric;
        material.materialFlags |= PathTracerShaderTypes::kMaterialFlagReferenceDielectricContinuation;
    }
    const bool forceBaseColorMip0 =
        (PathLooksLikeBistroAsset(gltfPath) &&
         MaterialNameSuggestsSparseAtlasSignage(src.name) &&
         !ContainsCaseInsensitive(src.name, "glass")) ||
        (PathLooksLikeBulkF1Asset(gltfPath) &&
         MaterialNameLooksLikeBulkF1LiveryAtlas(src.name)) ||
        (ContainsCaseInsensitive(gltfPath, "ferrari_f14_t_2014") &&
         src.name == "Vehicle_Exterior_mm_ext");
    if (forceBaseColorMip0) {
        material.materialFlags |= PathTracerShaderTypes::kMaterialFlagForceBaseColorMip0;
    }
    if (PathLooksLikeBistroAsset(gltfPath) && MaterialNameNeedsBlackKeyMask(src.name)) {
        material.materialFlags |= PathTracerShaderTypes::kMaterialFlagBlackKeyAlphaFromRgb;
    }
    if (PathLooksLikeBistroAsset(gltfPath) && BistroColoredStringLightEmitterMaterial(src.name)) {
        material.materialFlags |= PathTracerShaderTypes::kMaterialFlagCameraVisibleEmissionTint;
    }
    if (src.thinFoliageShadow) {
        material.materialFlags |= PathTracerShaderTypes::kMaterialFlagThinFoliageShadow;
    }
    if (src.alphaCutoutContinuation) {
        material.materialFlags |= PathTracerShaderTypes::kMaterialFlagAlphaCutoutContinuation;
    }
    if (src.foliageBeautyLift) {
        material.materialFlags |= PathTracerShaderTypes::kMaterialFlagFoliageBeautyLift;
    }
    if (disableOrmBySpecularWorkflowFallback) {
        material.pbrParams.x = 0.0f;
    }
    if (disableOrmTexture && loadOptions.disableOrmRoughnessOverride >= 0.0f) {
        material.pbrParams.y = std::clamp(loadOptions.disableOrmRoughnessOverride, 0.0f, 1.0f);
    }
    material.interiorMediumId = 0u;
    material.exteriorMediumId = 0u;
    material.materialPad2 = 0u;
    material.spectralParams = simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    int alphaMode = 0;
    const bool forceBlackKeyMask =
        PathLooksLikeBistroAsset(gltfPath) && MaterialNameNeedsBlackKeyMask(src.name);
    if (forceBlackKeyMask || src.forceBaseColorAlphaMask || src.alphaMode == "MASK") {
        alphaMode = 1;
    } else if (src.alphaMode == "BLEND") {
        alphaMode = 2;
    }
    float alphaFactor = std::clamp(src.baseColorFactor.w, 0.0f, 1.0f);
    if (loadOptions.alphaOverride >= 0.0f) {
        alphaFactor = std::clamp(loadOptions.alphaOverride, 0.0f, 1.0f);
    }
    if (loadOptions.barcelonaPavilionRules &&
        (BarcelonaPavilionWaterMaterial(src.name) ||
         BarcelonaPavilionArchitecturalGlassMaterial(src.name))) {
        alphaFactor = 1.0f;
        alphaMode = 0;
    }
    float alphaCutoff = (forceBlackKeyMask || src.forceBaseColorAlphaMask)
        ? 0.5f
        : std::clamp(src.alphaCutoff, 0.0f, 1.0f);
    material.pbrExtras = simd_make_float4(alphaFactor,
                                          alphaCutoff,
                                          transmissionFactor,
                                          static_cast<float>(alphaMode));

    if (auto it = loadOptions.materialFactorsByName.find(src.name);
        it != loadOptions.materialFactorsByName.end()) {
        const GltfMaterialFactors& factors = it->second;
        material.baseColorRoughness.xyz *= factors.baseColorMultiplier;
        material.pbrParams.y *= factors.roughnessMultiplier;
        material.baseColorRoughness.w = material.pbrParams.y;
        material.pbrParams.x *= factors.metallicMultiplier;
        if (report) {
            AddMaterialOverrideRecord(*report, src.name, "user_material_calibration",
                "Explicit user scene calibration multipliers on authored material/texture factors: roughness=" +
                std::to_string(factors.roughnessMultiplier) + ", linear RGB=" +
                std::to_string(factors.baseColorMultiplier.x) + "/" +
                std::to_string(factors.baseColorMultiplier.y) + "/" +
                std::to_string(factors.baseColorMultiplier.z) + ", metallic=" +
                std::to_string(factors.metallicMultiplier) + "; not source data");
        }
    }

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

    AssetPipelineReport assetReport{};
    assetReport.manifest.sourcePath = path;
    assetReport.manifest.sourceFormat = ToLowerAscii(gltfPath.extension().string());
    assetReport.manifest.importer = "PathTracer glTF 2.0 loader";
    assetReport.manifest.importerVersion = "runtime";
    NSDictionary* assetObject = root[@"asset"];
    if ([assetObject isKindOfClass:[NSDictionary class]]) {
        assetReport.manifest.generator = GetString(assetObject, @"generator");
    }
    assetReport.manifest.unitPolicy =
        "glTF scene units are treated as meters by convention; no automatic unit conversion is applied";
    assetReport.manifest.maturityLabel = "production_diagnostics";
    assetReport.manifest.sceneUnitMeters = 1.0f;
    const TextureLoadPolicy& texturePolicy = resources.textureLoadPolicy();
    assetReport.texturePolicy.residencyPolicy =
        "material textures are registered once by stable source key and promoted to private Metal textures when a device is available";
    assetReport.texturePolicy.mipPolicy =
        "runtime image textures generate mip chains; KTX2/DDS source mip chains are preserved when present; non-color glTF maps force trilinear mip sampling";
    assetReport.texturePolicy.compressionPolicy =
        "KTX2 and supported DDS block-compressed textures stay compressed; common raster inputs decode to RGBA texture storage";
    assetReport.texturePolicy.budgetPolicy =
        std::string(TextureBudgetPolicyLabel(texturePolicy.policy)) +
        "; pilot_mode=" + PilotModeLabel(texturePolicy.pilotMode);
    assetReport.texturePolicy.determinismPolicy =
        "texture identity is derived from manifest output hashes, file paths, embedded data hashes, or glTF bufferView/image indices";
    assetReport.texturePolicy.maxDimension = texturePolicy.maxDimension;
    assetReport.texturePolicy.maxTextureBytes = texturePolicy.maxTextureBytes;
    AddSupportMatrixEntry(assetReport,
                          "geometry",
                          "supported",
                          "glTF triangle primitives are imported; non-TRIANGLES primitive modes are skipped with diagnostics");
    AddSupportMatrixEntry(assetReport,
                          "transforms",
                          "supported",
                          "node TRS and matrix transforms are preserved; glTF units remain authored meters by convention");
    AddSupportMatrixEntry(assetReport,
                          "cameras",
                          "partial",
                          "embedded-camera framing uses the first scene camera's perspective position, direction and FOV only; roll, aspect and clipping are ignored; orthographic cameras use scene bounds");
    AddSupportMatrixEntry(assetReport,
                          "textures",
                          "supported",
                          "uri, data URI, and GLB bufferView images are loaded with explicit color-space and sampler policy diagnostics");
    AddSupportMatrixEntry(assetReport,
                          "materials",
                          "partial",
                          "metallic-roughness, alpha, normal, occlusion, emission, transmission, volume, IOR, and emissive strength are mapped; full DCC graphs remain out of scope");
    AddSupportMatrixEntry(assetReport,
                          "lights",
                          loadOptions.importPunctualLights ? "partial" : "unsupported",
                          loadOptions.importPunctualLights
                              ? "KHR_lights_punctual directional records are converted to renderer directional lights; point/spot lights are diagnosed"
                              : "KHR_lights_punctual records are diagnosed and not converted to renderer light primitives");
    AddSupportMatrixEntry(assetReport,
                          "animation",
                          loadOptions.enableAnimationSampling ? "partial" : "unsupported",
                          loadOptions.enableAnimationSampling
                              ? "node TRS animation channels are sampled at a requested scene time; skinning and morph target animation remain diagnosed"
                              : "animation, skinning, and morph targets are diagnosed and ignored by the static runtime scene loader");
    AddSupportMatrixEntry(assetReport,
                          "alpha_transmission_emission",
                          "partial",
                          "MASK/BLEND alpha, transmission factors/textures, emissive textures/factors, and emissive strength are mapped to renderer material data");
    AddSupportMatrixEntry(assetReport,
                          "ktx2_mip_compression",
                          "supported",
                          "KTX2/DDS texture loading, source mip preservation, generated mips, and budget skip/clamp decisions are reported");
    AddSupportMatrixEntry(assetReport,
                          "assimp_formats",
                          "external_tool",
                          "Assimp-supported static formats are converted through PathTracerImport before runtime loading");
    AddSupportMatrixEntry(assetReport,
                          "usd_materialx",
                          "out_of_scope",
                          "USD and MaterialX are not runtime import targets in this milestone; use glTF/GLB or Assimp conversion");
    AddAssetDiagnostic(assetReport,
                       "info",
                       "unit_scale_assumption",
                       "Importer assumes 1 glTF unit equals 1 meter and preserves authored transforms",
                       path,
                       "scene");
    AddAssetDiagnostic(assetReport,
                       "info",
                       "usd_materialx_boundary",
                       "USD and MaterialX are out of scope for the current production asset pipeline; no silent bridge is attempted",
                       path,
                       "scene");
    RecordUnsupportedExtensions(root, assetReport, path, loadOptions);
    if ([root[@"animations"] isKindOfClass:[NSArray class]] && [root[@"animations"] count] > 0 &&
        !loadOptions.enableAnimationSampling) {
        AddAssetDiagnostic(assetReport,
                           "warning",
                           "unsupported_feature",
                           "glTF animations are present but the runtime loader imports a deterministic static pose only",
                           path,
                           "animation",
                           "animations",
                           static_cast<int32_t>([root[@"animations"] count]));
    } else if ([root[@"animations"] isKindOfClass:[NSArray class]] && [root[@"animations"] count] > 0) {
        AddAssetDiagnostic(assetReport,
                           "info",
                           "animation_sampling",
                           "glTF node TRS animation is sampled for this scene load",
                           path,
                           "animation",
                           "animations",
                           static_cast<int32_t>([root[@"animations"] count]));
    }
    if ([root[@"skins"] isKindOfClass:[NSArray class]] && [root[@"skins"] count] > 0) {
        AddAssetDiagnostic(assetReport,
                           "warning",
                           "unsupported_feature",
                           "glTF skinning is present but not evaluated by the runtime loader",
                           path,
                           "skin",
                           "skins",
                           static_cast<int32_t>([root[@"skins"] count]));
    }
    NSDictionary* rootExtensions = root[@"extensions"];
    NSDictionary* punctualLights =
        [rootExtensions isKindOfClass:[NSDictionary class]] ? rootExtensions[@"KHR_lights_punctual"] : nil;
    if ([punctualLights isKindOfClass:[NSDictionary class]] &&
        [punctualLights[@"lights"] isKindOfClass:[NSArray class]] &&
        [punctualLights[@"lights"] count] > 0 &&
        !loadOptions.importPunctualLights) {
        AddAssetDiagnostic(assetReport,
                           "warning",
                           "unsupported_feature",
                           "KHR_lights_punctual lights are declared but not converted to renderer light primitives",
                           path,
                           "light",
                           "KHR_lights_punctual",
                           static_cast<int32_t>([punctualLights[@"lights"] count]));
    } else if ([punctualLights isKindOfClass:[NSDictionary class]] &&
               [punctualLights[@"lights"] isKindOfClass:[NSArray class]] &&
               [punctualLights[@"lights"] count] > 0) {
        AddAssetDiagnostic(assetReport,
                           "info",
                           "punctual_lights_imported",
                           "KHR_lights_punctual directional lights are imported for this scene load",
                           path,
                           "light",
                           "KHR_lights_punctual",
                           static_cast<int32_t>([punctualLights[@"lights"] count]));
    }

    std::vector<GltfBuffer> buffers;
    std::vector<GltfBufferView> views;
    std::vector<GltfAccessor> accessors;
    std::vector<GltfImage> images;
    std::vector<GltfSampler> samplers;
    std::vector<GltfTexture> textures;
    std::vector<GltfMaterial> materials;
    std::vector<GltfMesh> meshes;
    std::vector<GltfCamera> cameras;
    std::vector<GltfLight> lights;
    std::vector<GltfNode> nodes;
    std::vector<GltfAnimation> animations;

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
            PumpEventsEvery(static_cast<size_t>(i), 64u);
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
            PumpEventsEvery(static_cast<size_t>(i), 64u);
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
            PumpEventsEvery(static_cast<size_t>(i), 64u);
            NSDictionary* img = imageArray[i];
            images[i].name = GetString(img, @"name");
            images[i].uri = GetString(img, @"uri");
            images[i].mimeType = GetString(img, @"mimeType");
            images[i].bufferView = GetInt(img, @"bufferView", -1);
        }
    }
    assetReport.stats.imageCount = static_cast<uint32_t>(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        AddStableResourceId(assetReport,
                            "texture",
                            images[i].name.empty() ? images[i].uri : images[i].name,
                            static_cast<int32_t>(i),
                            StableAssetId("texture",
                                          GltfImageSourceLabel(path,
                                                               static_cast<int>(i),
                                                               images[i])));
    }

    NSArray* samplerArray = root[@"samplers"];
    if (samplerArray && [samplerArray isKindOfClass:[NSArray class]]) {
        samplers.resize(samplerArray.count);
        for (NSUInteger i = 0; i < samplerArray.count; ++i) {
            PumpEventsEvery(static_cast<size_t>(i), 64u);
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
            PumpEventsEvery(static_cast<size_t>(i), 64u);
            NSDictionary* tex = textureArray[i];
            textures[i].source = GetInt(tex, @"source", -1);
            textures[i].sampler = GetInt(tex, @"sampler", -1);
        }
    }
    assetReport.stats.textureCount = static_cast<uint32_t>(textures.size());
    if (textures.size() > PathTracerShaderTypes::kMaxMaterialTextures) {
        AddAssetDiagnostic(assetReport,
                           "warning",
                           "excessive_texture_count",
                           "glTF declares more textures than the renderer material texture table can bind",
                           path,
                           "scene");
    }

    if ([punctualLights isKindOfClass:[NSDictionary class]]) {
        NSArray* lightArray = punctualLights[@"lights"];
        if ([lightArray isKindOfClass:[NSArray class]]) {
            lights.resize(lightArray.count);
            for (NSUInteger i = 0; i < lightArray.count; ++i) {
                PumpEventsEvery(static_cast<size_t>(i), 16u);
                NSDictionary* light = lightArray[i];
                if (![light isKindOfClass:[NSDictionary class]]) {
                    continue;
                }
                lights[i].name = GetString(light, @"name");
                lights[i].type = GetString(light, @"type");
                std::vector<float> color;
                if (GetFloatArray(light, @"color", color) && color.size() >= 3u) {
                    lights[i].color = simd_make_float3(color[0], color[1], color[2]);
                }
                lights[i].intensity = std::max(GetFloat(light, @"intensity", 1.0f), 0.0f);
                if (loadOptions.importPunctualLights && lights[i].type != "directional") {
                    AddAssetDiagnostic(assetReport,
                                       "warning",
                                       "unsupported_feature",
                                       "KHR_lights_punctual point/spot lights are declared but only directional lights are imported",
                                       path,
                                       "light",
                                       lights[i].name,
                                       static_cast<int32_t>(i));
                }
            }
        }
    }

    NSArray* materialArray = root[@"materials"];
    if (materialArray && [materialArray isKindOfClass:[NSArray class]]) {
        materials.resize(materialArray.count);
        for (NSUInteger i = 0; i < materialArray.count; ++i) {
            PumpEventsEvery(static_cast<size_t>(i), 32u);
            NSDictionary* mat = materialArray[i];
            GltfMaterial& dst = materials[i];
            dst.name = GetString(mat, @"name");
            std::string alphaMode = GetString(mat, @"alphaMode");
            if (!alphaMode.empty()) {
                dst.alphaMode = alphaMode;
            }
            dst.alphaCutoff = GetFloat(mat, @"alphaCutoff", 0.5f);
            dst.doubleSided = GetBool(mat, @"doubleSided", false);
            if (MaterialNameSuggestsDoubleSided(dst.name)) {
                dst.doubleSided = true;
                AddMaterialOverrideRecord(assetReport,
                                          dst.name,
                                          "double_sided_from_name_policy",
                                          "material name matches established double-sided import compatibility pattern");
            }
            if ((dst.alphaMode == "OPAQUE" || dst.alphaMode == "BLEND") &&
                MaterialNameSuggestsMaskedFoliage(dst.name)) {
                dst.alphaMode = "MASK";
                dst.alphaCutoff = std::clamp(dst.alphaCutoff, 0.1f, 0.9f);
                AddMaterialOverrideRecord(assetReport,
                                          dst.name,
                                          "alpha_mask_from_foliage_policy",
                                          "foliage-like material imported as deterministic alpha mask to preserve cutout geometry");
            } else if (dst.alphaMode == "OPAQUE" &&
                       PathLooksLikeBistroAsset(gltfPath) &&
                       MaterialNameSuggestsMaskedSignage(dst.name)) {
                dst.alphaMode = "MASK";
                dst.alphaCutoff = std::clamp(dst.alphaCutoff, 0.1f, 0.9f);
                AddMaterialOverrideRecord(assetReport,
                                          dst.name,
                                          "alpha_mask_from_bistro_signage_policy",
                                          "Bistro signage compatibility policy maps opaque signage atlases to alpha mask");
            }
            if (PathLooksLikeBistroAsset(gltfPath) &&
                MaterialNameNeedsUv1Override(dst.name)) {
                dst.baseColorTexture.texCoord = 1;
                dst.normalTexture.texCoord = 1;
                dst.metallicRoughnessTexture.texCoord = 1;
                dst.occlusionTexture.texCoord = 1;
                AddMaterialOverrideRecord(assetReport,
                                          dst.name,
                                          "uv1_texture_coordinate_policy",
                                          "Bistro material compatibility policy selects TEXCOORD_1 for all material texture bindings");
            }
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
                if (TextureBindingLooksLikeSponzaBlackKeyCutout(path,
                                                                dst.baseColorTexture,
                                                                textures,
                                                                images)) {
                    dst.forceBaseColorAlphaMask = true;
                    dst.alphaMode = "MASK";
                    dst.alphaCutoff = 0.5f;
                    AddMaterialOverrideRecord(assetReport,
                                              dst.name,
                                              "sponza_black_key_alpha_policy",
                                              "known Sponza decorative atlas uses black-key cutout regions");
                }
                if ((dst.alphaMode == "OPAQUE" || dst.alphaMode == "BLEND") &&
                    TextureBindingImageSuggestsMaskedFoliage(dst.baseColorTexture, textures, images)) {
                    // A name says foliage; the uploaded image decides in the material build loop below, from the
                    // smallest alpha the texture decode records (no second decode). Until then: the old rule.
                    dst.foliageTextureNameCandidate = true;
                    dst.authoredAlphaMode = dst.alphaMode;
                    dst.authoredAlphaCutoff = dst.alphaCutoff;
                    dst.authoredDoubleSided = dst.doubleSided;
                    dst.alphaMode = "MASK";
                    dst.alphaCutoff = std::clamp(dst.alphaCutoff, 0.1f, 0.9f);
                    dst.doubleSided = true;
                }
                if (PathLooksLikeEmeraldSquareAsset(gltfPath) &&
                    EmeraldSquareMaterialLooksLikeFoliage(dst, textures, images)) {
                    dst.alphaMode = "MASK";
                    dst.alphaCutoff = 0.5f;
                    dst.doubleSided = true;
                    dst.foliageTextureNameCandidate = false;  // Emerald's own rule decides these.
                    // Alpha holes continue the ray; accepted leaves retain the
                    // same GGX shading and occlusion as other PBR surfaces.
                    dst.thinFoliageShadow = false;
                    dst.alphaCutoutContinuation = true;
                    dst.foliageBeautyLift = false;
                    AddMaterialOverrideRecord(assetReport,
                                              dst.name,
                                              "emerald_square_foliage_alpha_cutoff_policy",
                                              "Emerald Square foliage uses source opacity with the standard mask cutoff and continues through alpha holes");
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
            if (PathLooksLikeEmeraldSquareAsset(gltfPath) && dst.occlusionTexture.index >= 0) {
                // Every source Emerald Specular DDS has an unused black R
                // channel. Binding it as AO would erase diffuse illumination.
                dst.occlusionTexture = GltfTextureBinding{};
                dst.occlusionTexture.index = -1;
                dst.occlusionStrength = 0.0f;
                AddMaterialOverrideRecord(assetReport,
                                          dst.name,
                                          "emerald_square_unused_occlusion_channel",
                                          "Emerald Square packed Specular R is unused black; preserve authored roughness and metalness without applying it as AO");
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
                    if (volume[@"thicknessTexture"]) {
                        AddAssetDiagnostic(assetReport,
                                           "warning",
                                           "unsupported_feature",
                                           "KHR_materials_volume thicknessTexture is ignored; scalar thicknessFactor is used",
                                           path,
                                           "material",
                                           dst.name,
                                           static_cast<int32_t>(i));
                        NSLog(@"[glTF] Warning: material %lu '%s': thicknessTexture is ignored; scalar thicknessFactor is used",
                              static_cast<unsigned long>(i), dst.name.c_str());
                    }
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

            if (PathLooksLikeBistroAsset(gltfPath) &&
                (MaterialNameLooksLikeBistroPlaceholderOverlay(dst.name) ||
                 ContainsCaseInsensitive(dst.name, "master_focus"))) {
                // Bistro includes MASTER_Focus overlay quads on Paris_Building_01 that
                // show up as dark plaques under the lateral sign in path-traced shading.
                // Treat this overlay family as fully transparent.
                dst.alphaMode = "BLEND";
                dst.baseColorFactor.w = 0.0f;
                AddMaterialOverrideRecord(assetReport,
                                          dst.name,
                                          "transparent_placeholder_overlay_policy",
                                          "Bistro placeholder/focus overlay quads are imported as transparent overlays");
            }
        }
    }
    NSArray* meshArray = root[@"meshes"];
    const size_t declaredMaterialCount = materials.size();
    bool needsDefaultMaterial = false;
    if ([meshArray isKindOfClass:[NSArray class]]) {
        for (NSDictionary* mesh in meshArray) {
            NSArray* primitives = mesh[@"primitives"];
            if (![primitives isKindOfClass:[NSArray class]]) {
                continue;
            }
            for (NSDictionary* primitive in primitives) {
                const int materialIndex = GetInt(primitive, @"material", -1);
                if (materialIndex < 0 ||
                    materialIndex >= static_cast<int>(declaredMaterialCount)) {
                    needsDefaultMaterial = true;
                    break;
                }
            }
            if (needsDefaultMaterial) {
                break;
            }
        }
    }

    int defaultSourceMaterialIndex = -1;
    if (materials.empty()) {
        defaultSourceMaterialIndex = 0;
        materials.emplace_back();
        materials.back().name = "glTF default material";
        AddAssetDiagnostic(assetReport,
                           "warning",
                           "material_fallback",
                           "glTF contains no materials; importer created a default renderer material",
                           path,
                           "material");
    } else if (needsDefaultMaterial) {
        defaultSourceMaterialIndex = static_cast<int>(materials.size());
        materials.emplace_back();
        materials.back().name = "glTF default material";
        AddAssetDiagnostic(assetReport,
                           "info",
                           "material_fallback",
                           "Primitive without a valid glTF material uses the glTF default material",
                           path,
                           "material",
                           materials.back().name);
    }

    for (size_t i = 0; i < materials.size(); ++i) {
        PumpEventsEvery(i, 32u);
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
            AddMaterialOverrideRecord(assetReport,
                                      materials[i].name,
                                      "disable_orm_texture",
                                      "material matched import compatibility policy for ORM bypass");
            AddAssetDiagnostic(assetReport,
                               "warning",
                               "material_fallback",
                               "Material ORM texture was disabled by import compatibility policy",
                               path,
                               "material",
                               materials[i].name,
                               static_cast<int32_t>(i));
        }
    }
    assetReport.stats.materialCount = static_cast<uint32_t>(materials.size());

    // Decode the material images ahead of the loop below, in the order it requests them (the foliage rule's base colour,
    // then BuildGltfMaterial's slots). The guard ends the prefetch, waiting for decodes that read `buffers`, on every
    // return path.
    struct MaterialImagePrefetchGuard {
        SceneResources& resources;
        ~MaterialImagePrefetchGuard() { resources.endMaterialImagePrefetch(); }
    } materialImagePrefetchGuard{resources};
    {
        std::vector<MaterialImagePrefetchSource> prefetch;
        auto addTexture = [&](int textureIndex, bool normalMap = false, bool srgbColor = false) {
            if (textureIndex < 0 || textureIndex >= static_cast<int>(textures.size())) {
                return;
            }
            const int imageIndex = textures[static_cast<size_t>(textureIndex)].source;
            if (imageIndex < 0 || imageIndex >= static_cast<int>(images.size())) {
                return;
            }
            const GltfImage& image = images[static_cast<size_t>(imageIndex)];
            MaterialImagePrefetchSource source;
            source.normalMap = normalMap;
            source.srgbColor = srgbColor;
            if (!image.uri.empty()) {
                if (image.uri.rfind("data:", 0) == 0) {
                    return;  // decoded from a temporary buffer at request time
                }
                source.path = (fs::path(path).parent_path() / image.uri).string();
            } else if (image.bufferView >= 0 && image.bufferView < static_cast<int>(views.size())) {
                const GltfBufferView& view = views[static_cast<size_t>(image.bufferView)];
                if (view.buffer < 0 || view.buffer >= static_cast<int>(buffers.size())) {
                    return;
                }
                const GltfBuffer& buffer = buffers[static_cast<size_t>(view.buffer)];
                if (view.length == 0 || view.offset + view.length > buffer.data.size()) {
                    return;
                }
                source.data = buffer.data.data() + view.offset;
                source.size = view.length;
            } else {
                return;
            }
            prefetch.push_back(std::move(source));
        };
        for (const GltfMaterial& material : materials) {
            addTexture(material.baseColorTexture.index, /*normalMap=*/false, /*srgbColor=*/true);
            addTexture(material.metallicRoughnessTexture.index);
            addTexture(material.normalTexture.index, /*normalMap=*/true);
            addTexture(material.occlusionTexture.index);
            addTexture(material.emissiveTexture.index, /*normalMap=*/false, /*srgbColor=*/true);
            addTexture(material.transmissionTexture.index);
        }
        resources.beginMaterialImagePrefetch(prefetch);
    }
    std::vector<uint32_t> materialMap(materials.size(), 0u);
    size_t foliageRuleApplied = 0;
    size_t foliageRuleKeptBlend = 0;
    size_t foliageRuleUnchecked = 0;
    size_t foliageRuleSkipped = 0;
    for (size_t i = 0; i < materials.size(); ++i) {
        PumpEventsEvery(i, 16u);
        GltfMaterial& candidate = materials[i];
        if (candidate.foliageTextureNameCandidate) {
            // The foliage texture-name rule, decided by the base-colour texture's smallest alpha, which its decode
            // recorded (BuildGltfMaterial's upload of the same texture below is then a cache hit; no second decode).
            // With no texel below the cutoff a MASK cuts nothing and would only make the surface a two-sided cutout lit
            // from behind (Bistro's pavement and trunks). An authored BLEND is restored only when fully opaque: path
            // rays discard BLEND texels stochastically (rand > alpha), so a BLEND with alpha below 1 would turn partly
            // transparent where the MASK was opaque. KTX2 / DDS textures record nothing and keep the rule.
            const uint32_t baseColorIdx = ResolveTextureIndex(path, textures, samplers, images, buffers, views,
                                                              candidate.baseColorTexture.index,
                                                              !loadOptions.forceLinearBaseColor,
                                                              MaterialTextureSemantic::BaseColor,
                                                              resources,
                                                              nullptr,
                                                              candidate.name,
                                                              loadOptions.barcelonaPavilionRules);
            const int minTexel = (baseColorIdx == kInvalidTextureIndex) ? -1
                                                                        : resources.materialTextureMinAlpha(baseColorIdx);
            const float minAlpha = minTexel < 0 ? -1.0f
                : (static_cast<float>(minTexel) / 255.0f) * std::clamp(candidate.baseColorFactor.w, 0.0f, 1.0f);
            const bool checked = minAlpha >= 0.0f;
            const bool holes = checked && minAlpha < candidate.alphaCutoff;
            const bool skip = checked && !holes && (candidate.authoredAlphaMode == "OPAQUE" || minAlpha >= 1.0f);
            if (skip) {
                ++foliageRuleSkipped;
                candidate.alphaMode = candidate.authoredAlphaMode;
                candidate.alphaCutoff = candidate.authoredAlphaCutoff;
                candidate.doubleSided = candidate.authoredDoubleSided;
                NSLog(@"[Loader] Foliage texture-name rule skipped for material '%s': its base-colour texture has no "
                      @"texel with alpha below the cutoff %g (smallest %g), so it keeps alphaMode %s and doubleSided %d "
                      @"as authored",
                      candidate.name.c_str(), std::clamp(candidate.authoredAlphaCutoff, 0.1f, 0.9f), minAlpha,
                      candidate.alphaMode.c_str(), candidate.doubleSided ? 1 : 0);
                AddMaterialOverrideRecord(assetReport,
                                          candidate.name,
                                          "texture_name_foliage_rule_skipped_no_alpha_holes",
                                          "base-color texture name suggests foliage, but its texture has no texel "
                                          "below the alpha cutoff; alpha mode and sidedness kept as authored");
            } else {
                ++(holes ? foliageRuleApplied : (checked ? foliageRuleKeptBlend : foliageRuleUnchecked));
                AddMaterialOverrideRecord(assetReport,
                                          candidate.name,
                                          "alpha_mask_from_texture_name_policy",
                                          holes ? "base-color texture name suggests foliage/cutout content and its "
                                                  "texture has alpha holes"
                                                : (checked ? "base-color texture name suggests foliage; no alpha holes, "
                                                             "but authored BLEND with alpha below 1 stays an opaque MASK"
                                                           : "base-color texture name suggests foliage/cutout content "
                                                             "(texture format not checked for alpha holes)"));
            }
        }
        MaterialData material = BuildGltfMaterial(materials[i],
                                                 path,
                                                 textures,
                                                 samplers,
                                                 images,
                                                 buffers,
                                                 views,
                                                 loadOptions,
                                                 resources,
                                                 &assetReport);
        materialMap[i] = resources.addMaterialData(material, materials[i].name);
        AddStableResourceId(assetReport,
                            "material",
                            materials[i].name,
                            static_cast<int32_t>(i),
                            StableAssetId("material",
                                          path + "|material|" + std::to_string(i) + "|" +
                                              materials[i].name));
    }
    resources.endMaterialImagePrefetch();
    if (foliageRuleApplied + foliageRuleKeptBlend + foliageRuleUnchecked + foliageRuleSkipped > 0) {
        NSLog(@"[Loader] Foliage texture-name rule: %zu materials made double-sided MASK (alpha holes found), %zu made "
              @"double-sided MASK as before (authored BLEND with alpha below 1, no holes), %zu made double-sided MASK "
              @"without a check (texture format not decoded here), %zu skipped (no alpha holes)",
              foliageRuleApplied, foliageRuleKeptBlend, foliageRuleUnchecked, foliageRuleSkipped);
    }
    if (loadOptions.barcelonaPavilionRules) {
        size_t waterMaterials = 0u, glassMaterials = 0u;
        for (const GltfMaterial& material : materials) {
            waterMaterials += BarcelonaPavilionWaterMaterial(material.name) ? 1u : 0u;
            glassMaterials += BarcelonaPavilionArchitecturalGlassMaterial(material.name) ? 1u : 0u;
        }
        NSLog(@"[Loader] Barcelona Pavilion rules (mesh gltfBarcelonaPavilionRules=1): sampler anisotropy 16, "
              @"SingleScatterSpecular on all %zu materials, water rules on %zu, architectural-glass rules on %zu",
              materials.size(), waterMaterials, glassMaterials);
    } else if (PathLooksLikeBarcelonaPavilionAsset(path)) {
        NSLog(@"[Loader] warning: '%s' looks like the Barcelona Pavilion asset, but its mesh line does not set "
              @"gltfBarcelonaPavilionRules=1, so its reference-matching rules (anisotropy 16, SingleScatterSpecular, "
              @"water and glass) are off",
              path.c_str());
    }
    const uint32_t defaultMaterialIndex =
        (defaultSourceMaterialIndex >= 0)
            ? materialMap[static_cast<size_t>(defaultSourceMaterialIndex)]
            : 0u;
    if (meshArray && [meshArray isKindOfClass:[NSArray class]]) {
        meshes.resize(meshArray.count);
        for (NSUInteger i = 0; i < meshArray.count; ++i) {
            PumpEventsEvery(static_cast<size_t>(i), 16u);
            NSDictionary* mesh = meshArray[i];
            meshes[i].name = GetString(mesh, @"name");
            NSArray* prims = mesh[@"primitives"];
            if (![prims isKindOfClass:[NSArray class]]) {
                continue;
            }
            for (NSDictionary* prim in prims) {
                PumpEventsEvery(meshes[i].primitives.size(), 8u);
                if ([prim[@"targets"] isKindOfClass:[NSArray class]] && [prim[@"targets"] count] > 0) {
                    AddAssetDiagnostic(assetReport,
                                       "warning",
                                       "unsupported_feature",
                                       "glTF morph targets are present but not evaluated by the runtime loader",
                                       path,
                                       "mesh",
                                       meshes[i].name,
                                       static_cast<int32_t>(i));
                }
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
            PumpEventsEvery(static_cast<size_t>(i), 16u);
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
            AddStableResourceId(assetReport,
                                "camera",
                                GetString(cam, @"name"),
                                static_cast<int32_t>(i),
                                StableAssetId("camera",
                                              path + "|camera|" + std::to_string(i) + "|" +
                                                  GetString(cam, @"name") + "|" +
                                                  std::string([(NSString*)typeStr UTF8String])));
        }
    }

    NSArray* nodeArray = root[@"nodes"];
    if (nodeArray && [nodeArray isKindOfClass:[NSArray class]]) {
        nodes.resize(nodeArray.count);
        for (NSUInteger i = 0; i < nodeArray.count; ++i) {
            PumpEventsEvery(static_cast<size_t>(i), 16u);
            NSDictionary* node = nodeArray[i];
            nodes[i].name = GetString(node, @"name");
            nodes[i].mesh = GetInt(node, @"mesh", -1);
            nodes[i].camera = GetInt(node, @"camera", -1);
            NSDictionary* nodeExtensions = node[@"extensions"];
            if ([nodeExtensions isKindOfClass:[NSDictionary class]] &&
                [nodeExtensions[@"KHR_lights_punctual"] isKindOfClass:[NSDictionary class]]) {
                NSDictionary* lightRef = nodeExtensions[@"KHR_lights_punctual"];
                nodes[i].light = GetInt(lightRef, @"light", -1);
                if (!loadOptions.importPunctualLights) {
                    AddAssetDiagnostic(assetReport,
                                       "warning",
                                       "unsupported_feature",
                                       "Node references KHR_lights_punctual data, which is not converted to renderer light primitives",
                                       path,
                                       "node",
                                       nodes[i].name,
                                       static_cast<int32_t>(i));
                }
            }

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

    NSArray* animationArray = root[@"animations"];
    if ([animationArray isKindOfClass:[NSArray class]]) {
        animations.resize(animationArray.count);
        for (NSUInteger i = 0; i < animationArray.count; ++i) {
            PumpEventsEvery(static_cast<size_t>(i), 4u);
            NSDictionary* anim = animationArray[i];
            if (![anim isKindOfClass:[NSDictionary class]]) {
                continue;
            }
            GltfAnimation& dst = animations[i];
            dst.name = GetString(anim, @"name");
            NSArray* samplerArrayAnim = anim[@"samplers"];
            if ([samplerArrayAnim isKindOfClass:[NSArray class]]) {
                dst.samplers.resize(samplerArrayAnim.count);
                for (NSUInteger s = 0; s < samplerArrayAnim.count; ++s) {
                    NSDictionary* sampler = samplerArrayAnim[s];
                    if (![sampler isKindOfClass:[NSDictionary class]]) {
                        continue;
                    }
                    GltfAnimationSampler& dstSampler = dst.samplers[s];
                    dstSampler.input = GetInt(sampler, @"input", -1);
                    dstSampler.output = GetInt(sampler, @"output", -1);
                    std::string interpolation = GetString(sampler, @"interpolation");
                    if (!interpolation.empty()) {
                        dstSampler.interpolation = interpolation;
                    }
                }
            }
            NSArray* channelArray = anim[@"channels"];
            if ([channelArray isKindOfClass:[NSArray class]]) {
                dst.channels.reserve(channelArray.count);
                for (NSDictionary* channel in channelArray) {
                    if (![channel isKindOfClass:[NSDictionary class]]) {
                        continue;
                    }
                    NSDictionary* target = channel[@"target"];
                    if (![target isKindOfClass:[NSDictionary class]]) {
                        continue;
                    }
                    GltfAnimationChannel dstChannel{};
                    dstChannel.sampler = GetInt(channel, @"sampler", -1);
                    dstChannel.targetNode = GetInt(target, @"node", -1);
                    dstChannel.targetPath = GetString(target, @"path");
                    dst.channels.push_back(std::move(dstChannel));
                }
            }
            for (const GltfAnimationChannel& channel : dst.channels) {
                if (channel.sampler < 0 || channel.sampler >= static_cast<int>(dst.samplers.size())) {
                    continue;
                }
                GltfAnimationSampler& sampler = dst.samplers[static_cast<size_t>(channel.sampler)];
                if (sampler.input < 0 || sampler.input >= static_cast<int>(accessors.size()) ||
                    sampler.output < 0 || sampler.output >= static_cast<int>(accessors.size())) {
                    continue;
                }
                const GltfAccessor& inputAccessor = accessors[static_cast<size_t>(sampler.input)];
                if (sampler.times.empty()) {
                    ReadAccessorFloatN(buffers, views, inputAccessor, 1u, sampler.times);
                    for (float t : sampler.times) {
                        if (!dst.hasTimeRange) {
                            dst.startTime = t;
                            dst.endTime = t;
                            dst.hasTimeRange = true;
                        } else {
                            dst.startTime = std::min(dst.startTime, t);
                            dst.endTime = std::max(dst.endTime, t);
                        }
                    }
                }
                size_t components = 0u;
                if (channel.targetPath == "translation" || channel.targetPath == "scale") {
                    components = 3u;
                } else if (channel.targetPath == "rotation") {
                    components = 4u;
                }
                if (components == 0u) {
                    continue;
                }
                sampler.components = components;
                if (sampler.values.empty()) {
                    const GltfAccessor& outputAccessor = accessors[static_cast<size_t>(sampler.output)];
                    ReadAccessorFloatN(buffers, views, outputAccessor, components, sampler.values);
                }
            }
        }
    }

    std::vector<GltfAnimatedNodeTransform> animatedNodeTransforms(nodes.size());
    bool sampledAnimation = false;
    if (loadOptions.enableAnimationSampling && !animations.empty() && !nodes.empty()) {
        const int selectedAnimationIndex =
            std::clamp(loadOptions.animationIndex, 0, static_cast<int>(animations.size()) - 1);
        const GltfAnimation& animation = animations[static_cast<size_t>(selectedAnimationIndex)];
        float sampleTime = loadOptions.animationTimeSeconds;
        if (loadOptions.loopAnimation && animation.hasTimeRange) {
            sampleTime = WrapAnimationTime(sampleTime, animation.startTime, animation.endTime);
        }
        for (const GltfAnimationChannel& channel : animation.channels) {
            if (channel.targetNode < 0 ||
                channel.targetNode >= static_cast<int>(animatedNodeTransforms.size()) ||
                channel.sampler < 0 ||
                channel.sampler >= static_cast<int>(animation.samplers.size())) {
                continue;
            }
            const GltfAnimationSampler& sampler = animation.samplers[static_cast<size_t>(channel.sampler)];
            std::vector<float> value = SampleAnimationValue(sampler, sampleTime);
            if (channel.targetPath == "translation" && value.size() >= 3u) {
                auto& dst = animatedNodeTransforms[static_cast<size_t>(channel.targetNode)];
                dst.hasTranslation = true;
                dst.translation = simd_make_float3(value[0], value[1], value[2]);
                sampledAnimation = true;
            } else if (channel.targetPath == "scale" && value.size() >= 3u) {
                auto& dst = animatedNodeTransforms[static_cast<size_t>(channel.targetNode)];
                dst.hasScale = true;
                dst.scale = simd_make_float3(value[0], value[1], value[2]);
                sampledAnimation = true;
            } else if (channel.targetPath == "rotation" && value.size() >= 4u) {
                auto& dst = animatedNodeTransforms[static_cast<size_t>(channel.targetNode)];
                dst.hasRotation = true;
                dst.rotation = NormalizeQuat(simd_make_float4(value[0], value[1], value[2], value[3]));
                sampledAnimation = true;
            }
        }
        if (sampledAnimation) {
            AddAssetDiagnostic(assetReport,
                               "info",
                               "animation_sampling",
                               "glTF animation node transforms were evaluated at time " +
                                   std::to_string(sampleTime),
                               path,
                               "animation",
                               animation.name.empty() ? "animation" : animation.name,
                               selectedAnimationIndex);
        }
    }

    NSArray* scenesArray = root[@"scenes"];
    int sceneIndex = GetInt(root, @"scene", 0);
    std::vector<int> sceneNodes;
    if (scenesArray && [scenesArray isKindOfClass:[NSArray class]] &&
        sceneIndex >= 0 && sceneIndex < static_cast<int>(scenesArray.count)) {
        NSDictionary* scene = scenesArray[static_cast<NSUInteger>(sceneIndex)];
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

    simd::float3 importedBoundsMin = simd_make_float3(0.0f, 0.0f, 0.0f);
    simd::float3 importedBoundsMax = simd_make_float3(0.0f, 0.0f, 0.0f);
    bool importedBoundsValid = false;

    // Decoded geometry of one glTF primitive, shared by every node that instances it. barcelona_pavilion_day has
    // 21,089 mesh instances of far fewer primitives, and each instance used to re-read its accessors and re-run
    // MikkTSpace (about a fifth of that scene's load). The data depends only on the primitive's accessors, the
    // global UV flip and the mirror flag (a mirrored transform reverses the winding before the tangents are
    // built), so a cached copy is byte-for-byte what a fresh decode returns. Only primitives that more than one
    // node references are cached, and an entry is dropped after its last reference.
    struct DecodedPrimitive {
        std::vector<float> positions;  // the POSITION accessor as read, for the per-instance bounds and counts
        // Shared with every mesh added from this decode (SceneResources::addMesh keeps them, not copies).
        std::shared_ptr<const std::vector<SceneResources::MeshVertex>> vertices;
        std::shared_ptr<const std::vector<uint32_t>> indices;
        bool hasNormals = false;
        bool hasUvs = false;
        bool hasTangents = false;
        uint64_t degenerateTriangles = 0u;  // the normals check, or the normal generation when there are none
    };
    using PrimitiveAccessorKey = std::tuple<int, int, int, int, int, int>;
    auto primitiveAccessorKey = [](const GltfPrimitive& prim) {
        return PrimitiveAccessorKey(prim.positionAccessor,
                                    prim.normalAccessor,
                                    prim.texcoordAccessor,
                                    prim.texcoord1Accessor,
                                    prim.tangentAccessor,
                                    prim.indexAccessor);
    };
    std::map<PrimitiveAccessorKey, uint32_t> primitiveReferencesLeft;
    for (const GltfNode& node : nodes) {
        if (node.mesh >= 0 && node.mesh < static_cast<int>(meshes.size())) {
            for (const GltfPrimitive& prim : meshes[static_cast<size_t>(node.mesh)].primitives) {
                ++primitiveReferencesLeft[primitiveAccessorKey(prim)];
            }
        }
    }
    std::map<std::pair<PrimitiveAccessorKey, bool>, DecodedPrimitive> decodedPrimitiveCache;
    uint64_t decodedPrimitiveCacheHits = 0u;
    uint64_t decodedPrimitiveDecodes = 0u;

    // Reads and builds the primitive's vertices and indices. Returns false with errorMessage set on a read
    // failure; an empty POSITION accessor gives no vertices.
    auto decodePrimitive = [&](const GltfPrimitive& prim,
                               const simd::float4x4& localToWorld,
                               DecodedPrimitive& out) -> bool {
        ++decodedPrimitiveDecodes;
        const GltfAccessor& posAcc = accessors[static_cast<size_t>(prim.positionAccessor)];
        std::vector<float> positions;
        if (!ReadAccessorFloatN(buffers, views, posAcc, 3, positions)) {
            errorMessage = "Failed reading POSITION accessor";
            return false;
        }
        size_t vertexCount = posAcc.count;
        if (vertexCount == 0) {
            return true;
        }

        std::vector<float> normals;
        bool hasNormals = false;
        if (prim.normalAccessor >= 0 && prim.normalAccessor < static_cast<int>(accessors.size())) {
            const GltfAccessor& nAcc = accessors[static_cast<size_t>(prim.normalAccessor)];
            hasNormals = ReadAccessorFloatN(buffers, views, nAcc, 3, normals);
        }

        std::vector<float> uvs;
        bool hasUvs = false;
        if (prim.texcoordAccessor >= 0 && prim.texcoordAccessor < static_cast<int>(accessors.size())) {
            const GltfAccessor& uvAcc = accessors[static_cast<size_t>(prim.texcoordAccessor)];
            hasUvs = ReadAccessorFloatN(buffers, views, uvAcc, 2, uvs);
        }
        std::vector<float> uvs1;
        bool hasUvs1 = false;
        if (prim.texcoord1Accessor >= 0 && prim.texcoord1Accessor < static_cast<int>(accessors.size())) {
            const GltfAccessor& uv1Acc = accessors[static_cast<size_t>(prim.texcoord1Accessor)];
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
                const GltfMaterial& primMaterial = materials[static_cast<size_t>(prim.material)];
                mrTexIndex = primMaterial.metallicRoughnessTexture.index;
                mrTexCoord = primMaterial.metallicRoughnessTexture.texCoord;
                if (mrTexIndex >= 0 && mrTexIndex < static_cast<int>(textures.size())) {
                    int samplerIndex = textures[static_cast<size_t>(mrTexIndex)].sampler;
                    if (samplerIndex >= 0 && samplerIndex < static_cast<int>(samplers.size())) {
                        wrapS = samplers[static_cast<size_t>(samplerIndex)].wrapS;
                        wrapT = samplers[static_cast<size_t>(samplerIndex)].wrapT;
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
            const GltfAccessor& tAcc = accessors[static_cast<size_t>(prim.tangentAccessor)];
            hasTangents = ReadAccessorFloatN(buffers, views, tAcc, 4, tangents);
        }
        std::vector<uint32_t> indices;
        if (prim.indexAccessor >= 0 && prim.indexAccessor < static_cast<int>(accessors.size())) {
            const GltfAccessor& iAcc = accessors[static_cast<size_t>(prim.indexAccessor)];
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
        ReverseTriangleWindingForMirroredTransform(indices, localToWorld);
        std::vector<SceneResources::MeshVertex> vertices(vertexCount);
        for (size_t i = 0; i < vertexCount; ++i) {
            PumpEventsEvery(i);
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
                                        loadOptions.flipTexcoordV ? (1.0f - uvs[i * 2 + 1]) : uvs[i * 2 + 1]);
            }
            if (hasUvs1) {
                v.uv1 = simd_make_float2(uvs1[i * 2 + 0],
                                         loadOptions.flipTexcoordV ? (1.0f - uvs1[i * 2 + 1]) : uvs1[i * 2 + 1]);
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

        if (hasNormals) {
            uint64_t degenerateTriangles = 0u;
            for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                uint32_t i0 = indices[i + 0];
                uint32_t i1 = indices[i + 1];
                uint32_t i2 = indices[i + 2];
                if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) {
                    continue;
                }
                const simd::float3 p0 = vertices[i0].position;
                const simd::float3 p1 = vertices[i1].position;
                const simd::float3 p2 = vertices[i2].position;
                if (simd::length(simd::cross(p1 - p0, p2 - p0)) <= 0.0f) {
                    ++degenerateTriangles;
                }
            }
            out.degenerateTriangles = degenerateTriangles;
        }
        if (!hasNormals) {
            std::vector<simd::float3> normalAccum(vertexCount, simd_make_float3(0.0f, 0.0f, 0.0f));
            uint64_t degenerateTriangles = 0u;
            for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                PumpEventsEvery(i, 8192u);
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
                } else {
                    ++degenerateTriangles;
                }
            }
            out.degenerateTriangles = degenerateTriangles;
            for (size_t i = 0; i < vertexCount; ++i) {
                PumpEventsEvery(i);
                if (simd::length(normalAccum[i]) > 0.0f) {
                    vertices[i].normal = simd::normalize(normalAccum[i]);
                }
            }
        }

        if (!hasTangents && hasUvs) {
            // MikkTSpace's sign (tangent.w) follows the index winding. Build the tangents on the authored winding, so a
            // mirrored primitive gets the same object-space frame as an unmirrored one; the renderers then multiply w by
            // the determinant's sign, which carries the bitangent through the mirror. Built on the reversed winding, w
            // was flipped twice and the normal map's bitangent pointed the wrong way on mirrored nodes.
            const bool mirrored = TransformHasNegativeDeterminant(localToWorld);
            if (mirrored) {
                ReverseTriangleWindingForMirroredTransform(indices, localToWorld);  // back to the authored winding
            }
            GenerateTangents(vertices, indices);
            if (mirrored) {
                ReverseTriangleWindingForMirroredTransform(indices, localToWorld);
            }
        }
        out.positions = std::move(positions);
        out.vertices = std::make_shared<const std::vector<SceneResources::MeshVertex>>(std::move(vertices));
        out.indices = std::make_shared<const std::vector<uint32_t>>(std::move(indices));
        out.hasNormals = hasNormals;
        out.hasUvs = hasUvs;
        out.hasTangents = hasTangents;
        return true;
    };

    auto loadPrimitive = [&](const GltfPrimitive& prim,
                             const simd::float4x4& localToWorld,
                             const simd::float4x4& sourceLocalTransform,
                             const std::string& baseName) -> bool {
        if (prim.mode != 4) {
            AddAssetDiagnostic(assetReport,
                               "warning",
                               "unsupported_feature",
                               "glTF primitive mode is not TRIANGLES and was skipped",
                               path,
                               "mesh",
                               baseName,
                               prim.mode);
            return true;
        }
        if (prim.positionAccessor < 0 ||
            prim.positionAccessor >= static_cast<int>(accessors.size())) {
            errorMessage = "glTF primitive missing POSITION accessor";
            return false;
        }

        if (PathLooksLikeBistroAsset(gltfPath) &&
            prim.material >= 0 &&
            prim.material < static_cast<int>(materials.size())) {
            const GltfMaterial& sourceMaterial = materials[static_cast<size_t>(prim.material)];
            if (MaterialNameShouldHideBistroSideLetters(sourceMaterial.name)) {
                NSLog(@"[SignLettersDiag] skipSideLettersPrimitive mesh='%s' source='%s'",
                      baseName.c_str(),
                      sourceMaterial.name.c_str());
                return true;
            }
        }

        const PrimitiveAccessorKey accessorKey = primitiveAccessorKey(prim);
        auto referencesLeft = primitiveReferencesLeft.find(accessorKey);
        const uint32_t usesAfterThis =
            (referencesLeft != primitiveReferencesLeft.end() && referencesLeft->second > 0u) ? --referencesLeft->second
                                                                                             : 0u;
        const auto cacheKey = std::make_pair(accessorKey, TransformHasNegativeDeterminant(localToWorld));
        DecodedPrimitive freshDecode;
        const DecodedPrimitive* decoded = nullptr;
        auto cached = decodedPrimitiveCache.find(cacheKey);
        if (cached != decodedPrimitiveCache.end()) {
            decoded = &cached->second;
            ++decodedPrimitiveCacheHits;
        } else {
            if (!decodePrimitive(prim, localToWorld, freshDecode)) {
                return false;
            }
            if (usesAfterThis > 0u && !freshDecode.positions.empty()) {
                decoded = &decodedPrimitiveCache.emplace(cacheKey, std::move(freshDecode)).first->second;
            } else {
                decoded = &freshDecode;
            }
        }
        const std::vector<float>& positions = decoded->positions;
        const size_t vertexCount = positions.size() / 3u;
        if (vertexCount == 0) {
            return true;
        }
        const std::vector<SceneResources::MeshVertex>& vertices = *decoded->vertices;
        const std::vector<uint32_t>& indices = *decoded->indices;
        const bool hasNormals = decoded->hasNormals;
        const bool hasUvs = decoded->hasUvs;
        const bool hasTangents = decoded->hasTangents;
        ++assetReport.stats.primitiveCount;
        assetReport.stats.vertexCount += static_cast<uint64_t>(vertexCount);

        if (outCamera) {
            simd::float3 sceneMin = simd_make_float3(0.0f, 0.0f, 0.0f);
            simd::float3 sceneMax = simd_make_float3(0.0f, 0.0f, 0.0f);
            bool hasBounds = false;
            for (size_t i = 0; i < vertexCount; ++i) {
                PumpEventsEvery(i);
                simd::float3 localPos = simd_make_float3(positions[i * 3 + 0],
                                                        positions[i * 3 + 1],
                                                        positions[i * 3 + 2]);
                simd::float4 worldPos4 = simd_mul(localToWorld, simd_make_float4(localPos, 1.0f));
                simd::float3 worldPos = simd_make_float3(worldPos4.x, worldPos4.y, worldPos4.z);
                if (!importedBoundsValid) {
                    importedBoundsMin = worldPos;
                    importedBoundsMax = worldPos;
                    importedBoundsValid = true;
                } else {
                    importedBoundsMin = simd_min(importedBoundsMin, worldPos);
                    importedBoundsMax = simd_max(importedBoundsMax, worldPos);
                }
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

        if (!hasNormals) {
            ++assetReport.stats.missingNormalPrimitiveCount;
            AddAssetDiagnostic(assetReport,
                               "warning",
                               "missing_normals",
                               "Primitive has no readable normals; importer generated face-weighted fallback normals",
                               path,
                               "mesh",
                               baseName);
        }

        if (!hasTangents && hasUvs) {
            ++assetReport.stats.missingTangentPrimitiveCount;
            AddAssetDiagnostic(assetReport,
                               "info",
                               "missing_tangents",
                               "Primitive has UVs but no readable tangents; importer generated tangent frames",
                               path,
                               "mesh",
                               baseName);
        }

        assetReport.stats.indexCount += static_cast<uint64_t>(indices.size());
        assetReport.stats.triangleCount += static_cast<uint64_t>(indices.size() / 3u);

        if (decoded->hasNormals) {
            const uint64_t degenerateTriangles = decoded->degenerateTriangles;
            if (degenerateTriangles > 0u) {
                assetReport.stats.degenerateTriangleCount += degenerateTriangles;
                AddAssetDiagnostic(assetReport,
                                   "warning",
                                   "degenerate_geometry",
                                   "Primitive contains degenerate triangles: " +
                                       std::to_string(degenerateTriangles),
                                   path,
                                   "mesh",
                                   baseName);
            }
        }

        if (!decoded->hasNormals) {
            const uint64_t degenerateTriangles = decoded->degenerateTriangles;
            if (degenerateTriangles > 0u) {
                assetReport.stats.degenerateTriangleCount += degenerateTriangles;
                AddAssetDiagnostic(assetReport,
                                   "warning",
                                   "degenerate_geometry",
                                   "Primitive contains degenerate triangles while generating normals: " +
                                       std::to_string(degenerateTriangles),
                                   path,
                                   "mesh",
                                   baseName);
            }
        }
        uint32_t materialIndex = defaultMaterialIndex;
        if (prim.material >= 0 && prim.material < static_cast<int>(materialMap.size())) {
            materialIndex = materialMap[static_cast<size_t>(prim.material)];
        }
        const std::string geometryStableId =
            StableAssetId("geometry",
                          path + "|geometry|" + baseName +
                              "|position_accessor=" + std::to_string(prim.positionAccessor) +
                              "|index_accessor=" + std::to_string(prim.indexAccessor) +
                              "|vertex_count=" + std::to_string(vertices.size()) +
                              "|index_count=" + std::to_string(indices.size()) +
                              "|material=" + std::to_string(materialIndex));
        AddStableResourceId(assetReport,
                            "geometry",
                            baseName,
                            static_cast<int32_t>(resources.meshes().size()),
                            geometryStableId);

        resources.addMesh(decoded->vertices,
                          decoded->indices,
                          localToWorld,
                          sourceLocalTransform,
                          materialIndex,
                          baseName,
                          geometryStableId);
        if (usesAfterThis == 0u) {
            decodedPrimitiveCache.erase(std::make_pair(accessorKey, false));
            decodedPrimitiveCache.erase(std::make_pair(accessorKey, true));
        }
        return true;
    };

    std::function<bool(int, const simd::float4x4&, const std::string&)> traverse;
    traverse = [&](int nodeIndex, const simd::float4x4& parent, const std::string& prefix) -> bool {
        if (nodeIndex < 0 || nodeIndex >= static_cast<int>(nodes.size())) {
            return true;
        }
        const GltfNode& node = nodes[static_cast<size_t>(nodeIndex)];
        simd::float4x4 local = node.hasMatrix ? node.matrix : ComposeTrs(node.translation, node.rotation, node.scale);
        if (loadOptions.enableAnimationSampling &&
            nodeIndex < static_cast<int>(animatedNodeTransforms.size()) &&
            !node.hasMatrix) {
            const GltfAnimatedNodeTransform& animated = animatedNodeTransforms[static_cast<size_t>(nodeIndex)];
            const simd::float3 translation =
                animated.hasTranslation ? animated.translation : node.translation;
            const simd::float4 rotation =
                animated.hasRotation ? animated.rotation : node.rotation;
            const simd::float3 scale =
                animated.hasScale ? animated.scale : node.scale;
            local = ComposeTrs(translation, rotation, scale);
        }
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
            const GltfCamera& cam = cameras[static_cast<size_t>(node.camera)];
            if (!cam.isPerspective) {
                NSLog(@"[glTF] Warning: selected non-perspective camera %d is not used for runtime projection; embedded-camera framing uses scene bounds",
                      node.camera);
            }
            outCamera->valid = true;
            outCamera->hasPerspective = cam.isPerspective;
            outCamera->yfov = cam.yfov;
            outCamera->znear = cam.znear;
            outCamera->zfar = cam.zfar;
            outCamera->position = worldPos;
            outCamera->forward = forward;
            outCamera->up = up;
        }

        if (loadOptions.importPunctualLights &&
            node.light >= 0 &&
            node.light < static_cast<int>(lights.size())) {
            const GltfLight& light = lights[static_cast<size_t>(node.light)];
            if (light.type == "directional") {
                simd::float4 dir4 = simd_mul(world, simd_make_float4(0.0f, 0.0f, 1.0f, 0.0f));
                simd::float3 directionToLight = simd_make_float3(dir4.x, dir4.y, dir4.z);
                if (simd::length(directionToLight) > 0.0f) {
                    directionToLight = simd::normalize(directionToLight);
                } else {
                    directionToLight = simd_make_float3(0.0f, 1.0f, 0.0f);
                }
                const simd::float3 radiance =
                    light.color * (light.intensity * std::max(loadOptions.punctualLightIntensityScale, 0.0f));
                resources.addDirectionalLight(directionToLight,
                                              simd_make_float3(std::max(radiance.x, 0.0f),
                                                               std::max(radiance.y, 0.0f),
                                                               std::max(radiance.z, 0.0f)),
                                              1.0f);
            }
        }

        if (node.mesh >= 0 && node.mesh < static_cast<int>(meshes.size())) {
            const GltfMesh& mesh = meshes[static_cast<size_t>(node.mesh)];
            for (size_t primIndex = 0; primIndex < mesh.primitives.size(); ++primIndex) {
                PumpEventsEvery(primIndex, 1u);
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
                if (!loadPrimitive(mesh.primitives[primIndex], world, local, meshName)) {
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

    NSLog(@"[glTF] decoded primitive cache: %llu primitive decodes, %llu instances reused a cached decode",
          static_cast<unsigned long long>(decodedPrimitiveDecodes),
          static_cast<unsigned long long>(decodedPrimitiveCacheHits));
    assetReport.stats.meshCount = static_cast<uint32_t>(meshes.size());
    assetReport.stats.meshInstanceCount = static_cast<uint32_t>(resources.meshes().size());
    assetReport.stats.loadedTextureCount = resources.materialTextureCount();
    if (importedBoundsValid) {
        const simd::float3 extent = importedBoundsMax - importedBoundsMin;
        const float radius = simd::length(extent) * 0.5f;
        if (std::isfinite(radius) && (radius > 10000.0f || (radius > 0.0f && radius < 0.001f))) {
            AddAssetDiagnostic(assetReport,
                               "warning",
                               "unit_scale_assumption",
                               "Imported scene bounds look unusually scaled for meter-based glTF units; radius=" +
                                   std::to_string(radius),
                               path,
                               "scene");
        }
    }
    resources.setAssetPipelineReport(std::move(assetReport));

    return true;
}

}  // namespace PathTracer
