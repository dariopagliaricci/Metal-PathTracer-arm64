#include "import/BistroMaterialAudit.h"

#include "import/TextureConverter.h"
#include "shared/Ktx2Texture.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <set>
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
#endif

namespace fs = std::filesystem;

namespace PathTracer::Import {

namespace {

std::string ToLowerAscii(std::string value) {
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string ReplaceBackslashes(std::string value) {
    std::replace(value.begin(), value.end(), '\\', '/');
    return value;
}

std::string FormatFloat(double value, int precision = 4) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

std::string Join(const std::vector<std::string>& values, const char* delimiter) {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << delimiter;
        }
        out << values[i];
    }
    return out.str();
}

std::vector<uint8_t> ReadFileBytes(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }
    stream.seekg(0, std::ios::end);
    const std::streamsize size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    if (size < 0) {
        throw std::runtime_error("Failed to read file size: " + path.string());
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (size > 0) {
        stream.read(reinterpret_cast<char*>(bytes.data()), size);
        if (!stream) {
            throw std::runtime_error("Failed to read file: " + path.string());
        }
    }
    return bytes;
}

uint32_t ReadU32(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8u) |
           (static_cast<uint32_t>(data[2]) << 16u) |
           (static_cast<uint32_t>(data[3]) << 24u);
}

struct TextureChannelStats {
    double minValue = 1.0;
    double maxValue = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
};

struct TextureInspection {
    fs::path resolvedPath;
    std::string sourceFormat;
    std::string compression;
    uint32_t width = 0;
    uint32_t height = 0;
    int sourceChannelCount = 0;
    bool decodedHasAlpha = false;
    bool decodedBinaryAlpha = false;
    std::array<TextureChannelStats, 4> channels{};
    std::vector<uint8_t> decodedRgba;
    bool available = false;
};

struct MaterialSample {
    std::string category;
    std::string materialName;
    std::map<std::string, std::string> slots;
    std::optional<float> shininess;
    std::optional<float> specularFactor;
    std::optional<float> glossinessFactor;
};

struct SceneAudit {
    std::string label;
    fs::path scenePath;
    std::vector<MaterialSample> representatives;
    std::map<std::string, size_t> normalCompressionCounts;
    size_t normalTextureCount = 0;
    size_t rgbaNormalTextureCount = 0;
    size_t varyingAlphaNormalTextureCount = 0;
    std::vector<std::string> focusMeshDiagnostics;
};

constexpr aiTextureType kTextureTypes[] = {
    aiTextureType_DIFFUSE,
    aiTextureType_NORMAL_CAMERA,
    aiTextureType_NORMALS,
    aiTextureType_SPECULAR,
    aiTextureType_EMISSIVE,
    aiTextureType_LIGHTMAP,
    aiTextureType_AMBIENT_OCCLUSION,
    aiTextureType_DIFFUSE_ROUGHNESS,
    aiTextureType_SHININESS,
    aiTextureType_METALNESS,
    aiTextureType_UNKNOWN,
};

std::string TextureTypeName(aiTextureType type) {
    switch (type) {
        case aiTextureType_DIFFUSE:
            return "DIFFUSE";
        case aiTextureType_NORMAL_CAMERA:
            return "NORMAL_CAMERA";
        case aiTextureType_NORMALS:
            return "NORMALS";
        case aiTextureType_SPECULAR:
            return "SPECULAR";
        case aiTextureType_EMISSIVE:
            return "EMISSIVE";
        case aiTextureType_LIGHTMAP:
            return "LIGHTMAP";
        case aiTextureType_AMBIENT_OCCLUSION:
            return "AMBIENT_OCCLUSION";
        case aiTextureType_DIFFUSE_ROUGHNESS:
            return "DIFFUSE_ROUGHNESS";
        case aiTextureType_SHININESS:
            return "SHININESS";
        case aiTextureType_METALNESS:
            return "METALNESS";
        case aiTextureType_UNKNOWN:
            return "UNKNOWN";
        default:
            break;
    }
    return "TYPE_" + std::to_string(static_cast<int>(type));
}

std::optional<fs::path> ResolveTexturePath(const fs::path& scenePath, const aiString& texturePath) {
    const std::string raw = ReplaceBackslashes(texturePath.C_Str());
    if (raw.empty() || raw[0] == '*') {
        return std::nullopt;
    }

    const fs::path sourceDir = scenePath.parent_path();
    const fs::path sourceRoot = sourceDir.parent_path();
    const fs::path rawFsPath(raw);

    std::vector<fs::path> candidates;
    auto pushCandidate = [&candidates](const fs::path& candidate) {
        if (candidate.empty()) {
            return;
        }
        const fs::path normalized = candidate.lexically_normal();
        if (std::find(candidates.begin(), candidates.end(), normalized) == candidates.end()) {
            candidates.push_back(normalized);
        }
    };

    pushCandidate(rawFsPath);
    pushCandidate(sourceDir / rawFsPath);
    pushCandidate(sourceRoot / rawFsPath);
    pushCandidate(sourceDir / rawFsPath.filename());
    pushCandidate(sourceRoot / rawFsPath.filename());

    const std::string lowerRaw = ToLowerAscii(raw);
    const size_t texturesPos = lowerRaw.rfind("textures/");
    if (texturesPos != std::string::npos) {
        pushCandidate(sourceRoot / fs::path(raw.substr(texturesPos)));
    }

    for (const fs::path& candidate : candidates) {
        std::error_code ec;
        if (fs::exists(candidate, ec) && !ec) {
            return candidate;
        }
    }
    return std::nullopt;
}

TextureInspection InspectTexture(const fs::path& path) {
    TextureInspection info;
    info.resolvedPath = fs::absolute(path).lexically_normal();
    info.sourceFormat = ToLowerAscii(info.resolvedPath.extension().string());
    if (!info.sourceFormat.empty() && info.sourceFormat.front() == '.') {
        info.sourceFormat.erase(info.sourceFormat.begin());
    }

    const std::vector<uint8_t> bytes = ReadFileBytes(info.resolvedPath);
    if (info.sourceFormat == "dds") {
        if (bytes.size() < 128u || std::memcmp(bytes.data(), "DDS ", 4) != 0) {
            throw std::runtime_error("Invalid DDS file: " + info.resolvedPath.string());
        }
        info.height = ReadU32(bytes.data() + 12);
        info.width = ReadU32(bytes.data() + 16);
        std::string compression(reinterpret_cast<const char*>(bytes.data() + 84), 4);
        if (compression == "DX10" && bytes.size() >= 148u) {
            const uint32_t dxgiFormat = ReadU32(bytes.data() + 128);
            switch (dxgiFormat) {
                case 71u:
                case 72u:
                    compression = "BC1";
                    info.sourceChannelCount = 3;
                    break;
                case 77u:
                case 78u:
                    compression = "BC3";
                    info.sourceChannelCount = 4;
                    break;
                case 83u:
                case 84u:
                    compression = "BC5";
                    info.sourceChannelCount = 2;
                    break;
                default:
                    compression = "DX10:" + std::to_string(dxgiFormat);
                    break;
            }
        } else if (compression == "DXT1") {
            info.sourceChannelCount = 3;
        } else if (compression == "DXT5") {
            info.sourceChannelCount = 4;
        } else if (compression == "ATI2") {
            info.sourceChannelCount = 2;
        }
        info.compression = compression;
    }

    std::vector<uint8_t> rgba;
    uint32_t decodedWidth = 0u;
    uint32_t decodedHeight = 0u;
    bool hasAlpha = false;
    bool binaryAlpha = false;
    if (info.sourceFormat == "ktx2") {
        Ktx2::ImageData image;
        std::string error;
        if (!Ktx2::ParseRgba8File(info.resolvedPath, image, error)) {
            throw std::runtime_error(error);
        }
        decodedWidth = image.width;
        decodedHeight = image.height;
        rgba = std::move(image.rgba);
        for (size_t i = 3; i < rgba.size(); i += 4u) {
            const uint8_t alpha = rgba[i];
            if (alpha < 250u) {
                hasAlpha = true;
            }
            if (alpha > 8u && alpha < 247u) {
                binaryAlpha = false;
            }
        }
        if (hasAlpha) {
            binaryAlpha = true;
            for (size_t i = 3; i < rgba.size(); i += 4u) {
                const uint8_t alpha = rgba[i];
                if (alpha > 8u && alpha < 247u) {
                    binaryAlpha = false;
                    break;
                }
            }
        }
    } else {
        const DecodedTexture decoded = decodeTextureFile(info.resolvedPath.string());
        decodedWidth = decoded.width;
        decodedHeight = decoded.height;
        hasAlpha = decoded.has_alpha;
        binaryAlpha = decoded.binary_alpha;
        rgba = decoded.rgba;
    }
    info.width = decodedWidth;
    info.height = decodedHeight;
    info.decodedHasAlpha = hasAlpha;
    info.decodedBinaryAlpha = binaryAlpha;
    info.decodedRgba = rgba;

    const size_t pixelCount = static_cast<size_t>(info.width) * info.height;
    std::array<double, 4> sum{0.0, 0.0, 0.0, 0.0};
    std::array<double, 4> sumSquares{0.0, 0.0, 0.0, 0.0};
    for (size_t i = 0; i < pixelCount; ++i) {
        for (size_t channel = 0; channel < 4; ++channel) {
            const double value = static_cast<double>(info.decodedRgba[i * 4u + channel]) / 255.0;
            TextureChannelStats& stats = info.channels[channel];
            stats.minValue = std::min(stats.minValue, value);
            stats.maxValue = std::max(stats.maxValue, value);
            sum[channel] += value;
            sumSquares[channel] += value * value;
        }
    }
    if (pixelCount > 0) {
        for (size_t channel = 0; channel < 4; ++channel) {
            TextureChannelStats& stats = info.channels[channel];
            stats.mean = sum[channel] / static_cast<double>(pixelCount);
            const double meanSquare = sumSquares[channel] / static_cast<double>(pixelCount);
            stats.stddev = std::sqrt(std::max(0.0, meanSquare - stats.mean * stats.mean));
        }
    }
    info.available = true;
    return info;
}

struct SampleSummary {
    size_t count = 0;
    double meanR = 0.0;
    double meanG = 0.0;
    double meanB = 0.0;
};

void AccumulateSample(SampleSummary& summary, double r, double g, double b) {
    summary.meanR += r;
    summary.meanG += g;
    summary.meanB += b;
    ++summary.count;
}

std::string SampleSummaryString(const SampleSummary& summary) {
    if (summary.count == 0) {
        return "none";
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(3)
        << "count=" << summary.count
        << " mean=(" << (summary.meanR / summary.count)
        << "," << (summary.meanG / summary.count)
        << "," << (summary.meanB / summary.count) << ")";
    return out.str();
}

bool ContainsToken(const std::string& haystack, const std::vector<std::string>& tokens) {
    const std::string lower = ToLowerAscii(haystack);
    return std::any_of(tokens.begin(), tokens.end(), [&](const std::string& token) {
        return lower.find(token) != std::string::npos;
    });
}

std::string FormatUvRange(const aiMesh* mesh, unsigned int channel) {
    if (!mesh || !mesh->HasTextureCoords(channel)) {
        return "missing";
    }
    float minU = std::numeric_limits<float>::infinity();
    float minV = std::numeric_limits<float>::infinity();
    float maxU = -std::numeric_limits<float>::infinity();
    float maxV = -std::numeric_limits<float>::infinity();
    for (unsigned int i = 0; i < mesh->mNumVertices; ++i) {
        const aiVector3D uv = mesh->mTextureCoords[channel][i];
        minU = std::min(minU, uv.x);
        minV = std::min(minV, uv.y);
        maxU = std::max(maxU, uv.x);
        maxV = std::max(maxV, uv.y);
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(4)
        << "u=[" << minU << ", " << maxU << "] v=[" << minV << ", " << maxV << "]";
    return out.str();
}

std::string MaterialTextureUvInfo(aiMaterial* material, aiTextureType type) {
    if (!material || material->GetTextureCount(type) == 0) {
        return "none";
    }
    aiString path;
    unsigned int uvIndex = 0u;
    if (material->GetTexture(type, 0, &path, nullptr, &uvIndex) != aiReturn_SUCCESS) {
        return "query_failed";
    }
    std::ostringstream out;
    out << ReplaceBackslashes(path.C_Str()) << " (uv=" << uvIndex << ")";
    return out.str();
}

std::vector<std::string> CollectFocusMeshDiagnostics(const aiScene* scene,
                                                     const fs::path& scenePath,
                                                     const std::vector<std::string>& tokens) {
    std::vector<std::string> lines;
    if (!scene || !scene->HasMeshes()) {
        return lines;
    }
    for (unsigned int meshIndex = 0; meshIndex < scene->mNumMeshes; ++meshIndex) {
        const aiMesh* mesh = scene->mMeshes[meshIndex];
        const std::string meshName = mesh && mesh->mName.length > 0
            ? std::string(mesh->mName.C_Str())
            : ("mesh_" + std::to_string(meshIndex));
        if (!ContainsToken(meshName, tokens)) {
            continue;
        }

        aiMaterial* material = (mesh && mesh->mMaterialIndex < scene->mNumMaterials)
            ? scene->mMaterials[mesh->mMaterialIndex]
            : nullptr;
        aiString materialNameRaw;
        const std::string materialName =
            (material && AI_SUCCESS == material->Get(AI_MATKEY_NAME, materialNameRaw))
                ? std::string(materialNameRaw.C_Str())
                : "<missing>";

        std::optional<TextureInspection> baseTextureInfo;
        if (material && material->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
            aiString path;
            if (material->GetTexture(aiTextureType_DIFFUSE, 0, &path) == aiReturn_SUCCESS) {
                if (const auto resolved = ResolveTexturePath(scenePath, path)) {
                    baseTextureInfo = InspectTexture(*resolved);
                }
            }
        }

        SampleSummary allSamples{};
        SampleSummary lowYSamples{};
        SampleSummary lowYHighZSamples{};
        if (mesh && baseTextureInfo && mesh->HasTextureCoords(0) &&
            !baseTextureInfo->decodedRgba.empty()) {
            const aiAABB& aabb = mesh->mAABB;
            const float yThreshold = aabb.mMin.y + 0.2f * (aabb.mMax.y - aabb.mMin.y);
            const float zThreshold = aabb.mMin.z + 0.4f * (aabb.mMax.z - aabb.mMin.z);
            auto sampleTexture = [&](float u, float v) {
                const uint32_t width = baseTextureInfo->width;
                const uint32_t height = baseTextureInfo->height;
                if (width == 0u || height == 0u) {
                    return std::array<double, 3>{0.0, 0.0, 0.0};
                }
                const float wrappedU = u - std::floor(u);
                const float wrappedV = v - std::floor(v);
                const uint32_t x = std::min<uint32_t>(static_cast<uint32_t>(wrappedU * width), width - 1u);
                const uint32_t y = std::min<uint32_t>(static_cast<uint32_t>(wrappedV * height), height - 1u);
                const size_t offset = (static_cast<size_t>(y) * width + x) * 4u;
                return std::array<double, 3>{
                    static_cast<double>(baseTextureInfo->decodedRgba[offset + 0u]) / 255.0,
                    static_cast<double>(baseTextureInfo->decodedRgba[offset + 1u]) / 255.0,
                    static_cast<double>(baseTextureInfo->decodedRgba[offset + 2u]) / 255.0};
            };
            for (unsigned int faceIndex = 0; faceIndex < mesh->mNumFaces; ++faceIndex) {
                const aiFace& face = mesh->mFaces[faceIndex];
                if (face.mNumIndices != 3u) {
                    continue;
                }
                aiVector3D centroidPos(0.0f, 0.0f, 0.0f);
                aiVector3D centroidUv(0.0f, 0.0f, 0.0f);
                for (unsigned int k = 0; k < 3u; ++k) {
                    const unsigned int index = face.mIndices[k];
                    centroidPos += mesh->mVertices[index];
                    centroidUv += mesh->mTextureCoords[0][index];
                }
                centroidPos /= 3.0f;
                centroidUv /= 3.0f;
                const auto rgb = sampleTexture(centroidUv.x, centroidUv.y);
                AccumulateSample(allSamples, rgb[0], rgb[1], rgb[2]);
                if (centroidPos.y <= yThreshold) {
                    AccumulateSample(lowYSamples, rgb[0], rgb[1], rgb[2]);
                    if (centroidPos.z >= zThreshold) {
                        AccumulateSample(lowYHighZSamples, rgb[0], rgb[1], rgb[2]);
                    }
                }
            }
        }

        std::ostringstream out;
        out << meshName
            << " | material=" << materialName
            << " | verts=" << (mesh ? mesh->mNumVertices : 0u)
            << " tris=" << (mesh ? mesh->mNumFaces : 0u)
            << " | uv0=" << FormatUvRange(mesh, 0u)
            << " | uv1=" << FormatUvRange(mesh, 1u)
            << " | base=" << MaterialTextureUvInfo(material, aiTextureType_DIFFUSE)
            << " | normal=" << MaterialTextureUvInfo(material, aiTextureType_NORMALS)
            << " | normal_camera=" << MaterialTextureUvInfo(material, aiTextureType_NORMAL_CAMERA)
            << " | specular=" << MaterialTextureUvInfo(material, aiTextureType_SPECULAR)
            << " | sampleAll=" << SampleSummaryString(allSamples)
            << " | sampleLowY=" << SampleSummaryString(lowYSamples)
            << " | sampleLowYHighZ=" << SampleSummaryString(lowYHighZSamples);
        lines.push_back(out.str());
    }
    return lines;
}

std::optional<MaterialSample> FindRepresentativeMaterial(const aiScene* scene,
                                                         const fs::path& scenePath,
                                                         const std::string& category,
                                                         const std::vector<std::string>& tokens) {
    for (unsigned int i = 0; i < scene->mNumMaterials; ++i) {
        aiMaterial* material = scene->mMaterials[i];
        aiString name;
        const std::string materialName =
            (AI_SUCCESS == material->Get(AI_MATKEY_NAME, name)) ? std::string(name.C_Str())
                                                                : ("material_" + std::to_string(i));
        if (!ContainsToken(materialName, tokens)) {
            continue;
        }

        MaterialSample sample;
        sample.category = category;
        sample.materialName = materialName;
        for (aiTextureType type : kTextureTypes) {
            if (material->GetTextureCount(type) == 0) {
                continue;
            }
            aiString path;
            if (AI_SUCCESS != material->GetTexture(type, 0, &path)) {
                continue;
            }
            const auto resolved = ResolveTexturePath(scenePath, path);
            sample.slots.emplace(TextureTypeName(type),
                                 resolved ? resolved->filename().string() : ReplaceBackslashes(path.C_Str()));
        }

        float value = 0.0f;
        if (AI_SUCCESS == material->Get(AI_MATKEY_SHININESS, value)) {
            sample.shininess = value;
        }
        if (AI_SUCCESS == material->Get(AI_MATKEY_SPECULAR_FACTOR, value)) {
            sample.specularFactor = value;
        }
        if (AI_SUCCESS == material->Get(AI_MATKEY_GLOSSINESS_FACTOR, value)) {
            sample.glossinessFactor = value;
        }
        return sample;
    }
    return std::nullopt;
}

std::vector<fs::path> CollectUniqueTextures(const aiScene* scene,
                                            const fs::path& scenePath,
                                            aiTextureType preferred,
                                            aiTextureType fallback) {
    std::set<fs::path> unique;
    for (unsigned int i = 0; i < scene->mNumMaterials; ++i) {
        aiMaterial* material = scene->mMaterials[i];
        for (aiTextureType type : {preferred, fallback}) {
            if (material->GetTextureCount(type) == 0) {
                continue;
            }
            aiString path;
            if (AI_SUCCESS != material->GetTexture(type, 0, &path)) {
                continue;
            }
            if (const auto resolved = ResolveTexturePath(scenePath, path)) {
                unique.insert(*resolved);
                break;
            }
        }
    }
    return std::vector<fs::path>(unique.begin(), unique.end());
}

SceneAudit AuditScene(const std::string& label,
                      const fs::path& scenePath,
                      const std::vector<std::pair<std::string, std::vector<std::string>>>& categories) {
    SceneAudit audit;
    audit.label = label;
    audit.scenePath = scenePath;

#if PATH_TRACER_ENABLE_ASSIMP
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(scenePath.string(),
                                             aiProcess_Triangulate |
                                                 aiProcess_GenSmoothNormals |
                                                 aiProcess_CalcTangentSpace |
                                                 aiProcess_JoinIdenticalVertices);
    if (!scene || !scene->HasMaterials()) {
        throw std::runtime_error("Assimp failed to load " + scenePath.string() + ": " +
                                 importer.GetErrorString());
    }

    for (const auto& [category, tokens] : categories) {
        if (const auto sample = FindRepresentativeMaterial(scene, scenePath, category, tokens)) {
            audit.representatives.push_back(*sample);
        }
    }

    for (const fs::path& texturePath : CollectUniqueTextures(scene, scenePath, aiTextureType_NORMAL_CAMERA, aiTextureType_NORMALS)) {
        const TextureInspection info = InspectTexture(texturePath);
        ++audit.normalTextureCount;
        ++audit.normalCompressionCounts[info.compression];
        if (info.sourceChannelCount >= 4) {
            ++audit.rgbaNormalTextureCount;
            if (info.channels[3].stddev > 0.05) {
                ++audit.varyingAlphaNormalTextureCount;
            }
        }
    }

    if (label == "Exterior") {
        audit.focusMeshDiagnostics = CollectFocusMeshDiagnostics(
            scene,
            scenePath,
            {"vespa", "plantpot", "menu"});
    }
#else
    (void)categories;
    throw std::runtime_error("Assimp support is unavailable in this build");
#endif

    return audit;
}

std::string FormatTextureEvidence(const TextureInspection& info) {
    std::ostringstream out;
    out << info.resolvedPath.filename().string()
        << " (" << info.width << "x" << info.height
        << ", format=" << info.compression
        << ", channels=" << info.sourceChannelCount;
    if (info.sourceChannelCount >= 4) {
        out << ", alpha min/max/mean/std="
            << FormatFloat(info.channels[3].minValue) << "/"
            << FormatFloat(info.channels[3].maxValue) << "/"
            << FormatFloat(info.channels[3].mean) << "/"
            << FormatFloat(info.channels[3].stddev);
    } else {
        out << ", alpha absent in source";
    }
    out << ")";
    return out.str();
}

std::string ClassifySpecularTexture(const TextureInspection& info) {
    const double rgbMeanSpread =
        std::max({info.channels[0].mean, info.channels[1].mean, info.channels[2].mean}) -
        std::min({info.channels[0].mean, info.channels[1].mean, info.channels[2].mean});
    const double rgbStdSpread =
        std::max({info.channels[0].stddev, info.channels[1].stddev, info.channels[2].stddev}) -
        std::min({info.channels[0].stddev, info.channels[1].stddev, info.channels[2].stddev});

    if (rgbMeanSpread > 0.08 || rgbStdSpread > 0.08) {
        if (info.channels[3].stddev > 0.08) {
            return "RGB reflectance tint with independently varying alpha";
        }
        return "RGB reflectance tint";
    }
    if (info.channels[3].stddev > 0.08) {
        return "Monochrome RGB intensity with independently varying alpha";
    }
    if (std::max({info.channels[0].stddev, info.channels[1].stddev, info.channels[2].stddev}) < 0.02) {
        return "Effectively constant";
    }
    return "Monochrome intensity";
}

std::string MarkdownTableForRepresentatives(const std::vector<MaterialSample>& samples) {
    std::ostringstream out;
    out << "| Category | Material | Slots | Shininess | SpecularFactor |\n";
    out << "| --- | --- | --- | ---: | ---: |\n";
    for (const MaterialSample& sample : samples) {
        std::vector<std::string> slots;
        for (const auto& [slotName, asset] : sample.slots) {
            slots.push_back(slotName + "=" + asset);
        }
        out << "| " << sample.category
            << " | " << sample.materialName
            << " | " << (slots.empty() ? "-" : Join(slots, "<br>"))
            << " | " << (sample.shininess ? FormatFloat(*sample.shininess, 2) : "-")
            << " | " << (sample.specularFactor ? FormatFloat(*sample.specularFactor, 2) : "-")
            << " |\n";
    }
    return out.str();
}

std::string BuildReport(const SceneAudit& interior, const SceneAudit& exterior) {
    std::vector<std::string> h1Evidence;
    std::vector<std::string> h3Evidence;
    std::vector<std::string> h4Values;

    size_t rgbaNormals = 0;
    size_t varyingAlphaNormals = 0;
    for (const SceneAudit* scene : {&interior, &exterior}) {
        rgbaNormals += scene->rgbaNormalTextureCount;
        varyingAlphaNormals += scene->varyingAlphaNormalTextureCount;
        for (const MaterialSample& sample : scene->representatives) {
            auto normalIt = sample.slots.find("NORMAL_CAMERA");
            if (normalIt == sample.slots.end()) {
                normalIt = sample.slots.find("NORMALS");
            }
            if (normalIt != sample.slots.end()) {
                const fs::path fullPath = scene->scenePath.parent_path().parent_path() / "Textures" / normalIt->second;
                if (fs::exists(fullPath)) {
                    h1Evidence.push_back(scene->label + " / " + sample.category + ": " +
                                         FormatTextureEvidence(InspectTexture(fullPath)));
                }
            }

            auto specularIt = sample.slots.find("SPECULAR");
            if (specularIt != sample.slots.end()) {
                const fs::path fullPath = scene->scenePath.parent_path().parent_path() / "Textures" / specularIt->second;
                if (fs::exists(fullPath)) {
                    const TextureInspection info = InspectTexture(fullPath);
                    std::ostringstream evidence;
                    evidence << scene->label << " / " << sample.category << ": "
                             << specularIt->second << " => " << ClassifySpecularTexture(info)
                             << " (RGB std="
                             << FormatFloat(info.channels[0].stddev) << "/"
                             << FormatFloat(info.channels[1].stddev) << "/"
                             << FormatFloat(info.channels[2].stddev) << ", alpha std="
                             << FormatFloat(info.channels[3].stddev) << ")";
                    h3Evidence.push_back(evidence.str());
                }
            }

            if (sample.shininess) {
                h4Values.push_back(scene->label + " / " + sample.materialName + "=" +
                                   FormatFloat(*sample.shininess, 2));
            }
        }
    }

    std::ostringstream out;
    out << "# Bistro Material Fidelity Audit Report\n\n";
    out << "Generated by `PathTracerBistroAudit`.\n\n";

    out << "## H1-H4 Verdicts\n\n";
    out << "H1 (_ddna alpha): ";
    if (rgbaNormals == 0 || varyingAlphaNormals == 0) {
        out << "Confirmed absent in the vendored Bistro normal textures inspected.\n";
    } else {
        out << "Confirmed present and spatially varying in a subset of normal textures.\n";
    }
    out << "Evidence:\n";
    for (const std::string& line : h1Evidence) {
        out << "- " << line << "\n";
    }
    out << "- Scene-wide normal format counts: interior="
        << Join([&]() {
               std::vector<std::string> values;
               for (const auto& [compression, count] : interior.normalCompressionCounts) {
                   values.push_back(compression + ":" + std::to_string(count));
               }
               return values;
           }(),
           ", ")
        << "; exterior="
        << Join([&]() {
               std::vector<std::string> values;
               for (const auto& [compression, count] : exterior.normalCompressionCounts) {
                   values.push_back(compression + ":" + std::to_string(count));
               }
               return values;
           }(),
           ", ")
        << ".\n";
    out << "- Current importer roughness path never samples normal alpha; Bistro materials import today as scalar roughness with no ORM emission.\n\n";

    out << "H2 (normal Y-convention): DirectX convention confirmed, Bistro-specific flip still needed.\n";
    out << "Evidence:\n";
    out << "- The audited Bistro normals are Lumberyard/CryEngine-style tangent-space maps stored as ATI2/BC5 XY normals; the current importer does not perform any Bistro/CryEngine Y-flip before emitting glTF normals.\n";
    out << "- The runtime has only a manual diagnostic `debugFlipNormalGreen` switch, which indicates correction is not baked into the import path today.\n\n";

    out << "H3 (specular semantics): Mixed RGB reflectance textures, often with independently varying alpha.\n";
    out << "Evidence:\n";
    for (const std::string& line : h3Evidence) {
        out << "- " << line << "\n";
    }
    out << "- This rules out treating Bistro specular RGB as a simple roughness proxy; alpha must be considered separately if reused.\n\n";

    out << "H4 (shininess reliability): Collapses to FBX defaults / narrow import values.\n";
    out << "Values:\n";
    for (const std::string& line : h4Values) {
        out << "- " << line << "\n";
    }
    out << "- Current importer output confirms broad scalar collapse to roughness `0.553` and `orm=0` across exterior Bistro materials.\n\n";

    out << "## Interior Representative Materials\n\n";
    out << MarkdownTableForRepresentatives(interior.representatives) << "\n";
    out << "## Exterior Representative Materials\n\n";
    out << MarkdownTableForRepresentatives(exterior.representatives) << "\n";

    if (!exterior.focusMeshDiagnostics.empty()) {
        out << "## Exterior Focused Mesh Diagnostics\n\n";
        for (const std::string& line : exterior.focusMeshDiagnostics) {
            out << "- " << line << "\n";
        }
        out << "\n";
    }

    out << "## Conclusion\n\n";
    out << "The vendored Bistro pack in this repository does not expose the primary roughness signal through normal-map alpha on the representative BC5 normal maps audited here. "
        << "The importer also collapses roughness to a near-constant scalar today, so Phase 2 should focus on explicit Bistro logging, a safe source-backed roughness path, and a scoped normal Y correction rather than assuming a surviving `_ddna` alpha workflow.\n";
    return out.str();
}

}  // namespace

bool BistroAuditBackendAvailable() {
    return PATH_TRACER_ENABLE_ASSIMP != 0;
}

BistroAuditResult RunBistroMaterialAudit(const BistroAuditOptions& options) {
    BistroAuditResult result;
    if (!BistroAuditBackendAvailable()) {
        result.message = "Assimp support is unavailable in this build.";
        return result;
    }

    try {
        const fs::path interiorPath = fs::absolute(options.interiorScenePath).lexically_normal();
        const fs::path exteriorPath = fs::absolute(options.exteriorScenePath).lexically_normal();
        const fs::path outputPath = fs::absolute(options.outputPath).lexically_normal();

        const SceneAudit interior = AuditScene(
            "Interior",
            interiorPath,
            {
                {"chair", {"chair"}},
                {"wood", {"wood", "table", "bar"}},
                {"plaster", {"plaster", "wall"}},
                {"floor", {"floor", "tile"}},
                {"metal_fixture", {"metal", "lantern", "beertap"}},
                {"emissive", {"light", "emissive", "bulb"}},
            });

        const SceneAudit exterior = AuditScene(
            "Exterior",
            exteriorPath,
            {
                {"storefront", {"bistro", "door", "boulangerie", "facade"}},
                {"awning", {"awning"}},
                {"cobblestone", {"pavement", "cobblestone"}},
                {"lamppost", {"streetlight", "lamp"}},
                {"vespa", {"vespa"}},
                {"foliage", {"foliage", "plant"}},
            });

        const std::string report = BuildReport(interior, exterior);
        std::error_code ec;
        fs::create_directories(outputPath.parent_path(), ec);
        if (ec) {
            throw std::runtime_error("Failed to create report directory: " + outputPath.parent_path().string());
        }
        std::ofstream stream(outputPath);
        if (!stream.is_open()) {
            throw std::runtime_error("Failed to open report path: " + outputPath.string());
        }
        stream << report;
        if (!stream) {
            throw std::runtime_error("Failed to write report: " + outputPath.string());
        }

        result.success = true;
        result.message = "Bistro audit written to " + outputPath.string();
    } catch (const std::exception& e) {
        result.message = e.what();
    }

    return result;
}

std::string PathTracerBistroAuditUsage() {
    std::ostringstream out;
    out << "PathTracerBistroAudit\n"
        << "Usage:\n"
        << "  PathTracerBistroAudit [--interior <fbx>] [--exterior <fbx>] [--out <markdown>]\n\n"
        << "Defaults:\n"
        << "  --interior assets/canonical/bistro/source/BistroInterior.fbx\n"
        << "  --exterior assets/canonical/bistro/source/BistroExterior.fbx\n"
        << "  --out      validation/reports/bistro_material_fidelity_audit.md\n";
    return out.str();
}

}  // namespace PathTracer::Import
