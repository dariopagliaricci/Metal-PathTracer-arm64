#import "renderer/SceneManager.h"

#import <Foundation/Foundation.h>
#include <TargetConditionals.h>

#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <filesystem>
#include <functional>
#include <type_traits>
#include <unordered_map>
#include <simd/simd.h>

#include "renderer/RenderSettings.h"
#include "renderer/SceneResources.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

namespace {

using PathTracer::SceneManager;
using PathTracer::SceneResources;
using PathTracer::RenderSettings;
namespace fs = std::filesystem;

constexpr const char* kKeywordToken = "_keyword";
constexpr float kPi = 3.14159265358979323846f;
constexpr float kDefaultCarpaintBaseEta[3] = {1.3456f, 0.9652f, 0.6172f};
constexpr float kDefaultCarpaintBaseK[3] = {7.4746f, 6.3995f, 5.3031f};

struct LoadedMeshData {
    std::vector<SceneResources::MeshVertex> vertices;
    std::vector<uint32_t> indices;
};

struct VertexKey {
    int position = -1;
    int normal = -1;
    int texcoord = -1;

    bool operator==(const VertexKey& other) const noexcept {
        return position == other.position &&
               normal == other.normal &&
               texcoord == other.texcoord;
    }
};

struct VertexKeyHasher {
    size_t operator()(const VertexKey& key) const noexcept {
        size_t h1 = std::hash<int>{}(key.position);
        size_t h2 = std::hash<int>{}(key.normal);
        size_t h3 = std::hash<int>{}(key.texcoord);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

static void ApplyFallbackNormals(SceneResources::MeshVertex& v0,
                                 SceneResources::MeshVertex& v1,
                                 SceneResources::MeshVertex& v2) {
    simd::float3 n0 = v0.normal;
    simd::float3 n1 = v1.normal;
    simd::float3 n2 = v2.normal;

    bool hasNormal = (simd::length(n0) > 0.0f) ||
                     (simd::length(n1) > 0.0f) ||
                     (simd::length(n2) > 0.0f);
    if (hasNormal) {
        return;
    }

    simd::float3 edge1 = v1.position - v0.position;
    simd::float3 edge2 = v2.position - v0.position;
    simd::float3 normal = simd::cross(edge1, edge2);
    float normalLength = simd::length(normal);
    if (normalLength <= 0.0f) {
        return;
    }
    normal = simd::normalize(normal);
    v0.normal = normal;
    v1.normal = normal;
    v2.normal = normal;
}

static bool LoadObjMesh(const fs::path& filePath,
                        LoadedMeshData& outMesh,
                        std::string& errorMessage) {
    tinyobj::ObjReaderConfig config;
    config.triangulate = true;
    config.vertex_color = false;
    config.mtl_search_path = filePath.parent_path().string();

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(filePath.string(), config)) {
        if (!reader.Error().empty()) {
            errorMessage = reader.Error();
        } else {
            errorMessage = "Failed to parse OBJ file: " + filePath.string();
        }
        return false;
    }

    const auto& attrib = reader.GetAttrib();
    const auto& shapes = reader.GetShapes();

    if (attrib.vertices.empty()) {
        errorMessage = "OBJ file contains no vertex positions: " + filePath.string();
        return false;
    }

    size_t totalIndices = 0;
    for (const auto& shape : shapes) {
        totalIndices += shape.mesh.indices.size();
    }

    if (totalIndices == 0) {
        errorMessage = "OBJ file contains no triangle data: " + filePath.string();
        return false;
    }

    outMesh.vertices.clear();
    outMesh.indices.clear();
    outMesh.indices.reserve(totalIndices);
    outMesh.vertices.reserve(totalIndices);  // Worst-case: no shared vertices.

    std::unordered_map<VertexKey, uint32_t, VertexKeyHasher> vertexLookup;
    vertexLookup.reserve(totalIndices);

    for (const auto& shape : shapes) {
        const auto& mesh = shape.mesh;
        if (mesh.indices.size() % 3 != 0) {
            errorMessage = "OBJ mesh is not triangulated: " + filePath.string();
            return false;
        }

        for (const auto& index : mesh.indices) {
            if (index.vertex_index < 0) {
                errorMessage = "OBJ references a position index that is out of range";
                return false;
            }

            VertexKey key{
                .position = index.vertex_index,
                .normal = index.normal_index,
                .texcoord = index.texcoord_index,
            };

            auto [it, inserted] = vertexLookup.try_emplace(key, 0);
            if (inserted) {
                const int posIndex = index.vertex_index;
                const size_t posOffset = static_cast<size_t>(posIndex) * 3;
                if (posOffset + 2 >= attrib.vertices.size()) {
                    errorMessage = "OBJ references a position index that is out of range";
                    return false;
                }

                SceneResources::MeshVertex vertex{};
                vertex.position = simd_make_float3(
                    attrib.vertices[posOffset + 0],
                    attrib.vertices[posOffset + 1],
                    attrib.vertices[posOffset + 2]);

                if (index.normal_index >= 0 && !attrib.normals.empty()) {
                    const size_t normOffset = static_cast<size_t>(index.normal_index) * 3;
                    if (normOffset + 2 < attrib.normals.size()) {
                        vertex.normal = simd_make_float3(
                            attrib.normals[normOffset + 0],
                            attrib.normals[normOffset + 1],
                            attrib.normals[normOffset + 2]);
                    }
                }

                if (index.texcoord_index >= 0 && !attrib.texcoords.empty()) {
                    const size_t uvOffset = static_cast<size_t>(index.texcoord_index) * 2;
                    if (uvOffset + 1 < attrib.texcoords.size()) {
                        vertex.uv = simd_make_float2(
                            attrib.texcoords[uvOffset + 0],
                            attrib.texcoords[uvOffset + 1]);
                    }
                }

                uint32_t newIndex = static_cast<uint32_t>(outMesh.vertices.size());
                outMesh.vertices.push_back(vertex);
                it->second = newIndex;
            }

            outMesh.indices.push_back(it->second);
        }
    }

    for (size_t i = 0; i + 2 < outMesh.indices.size(); i += 3) {
        auto& v0 = outMesh.vertices[outMesh.indices[i + 0]];
        auto& v1 = outMesh.vertices[outMesh.indices[i + 1]];
        auto& v2 = outMesh.vertices[outMesh.indices[i + 2]];
        ApplyFallbackNormals(v0, v1, v2);
    }

    return true;
}

static std::shared_ptr<tinyply::PlyData> TryRequestPlyProperties(tinyply::PlyFile& file,
                                                                 const std::string& element,
                                                                 const std::vector<std::string>& properties,
                                                                 uint32_t listSizeHint = 0) {
    try {
        return file.request_properties_from_element(element, properties, listSizeHint);
    } catch (const std::exception&) {
        return nullptr;
    }
}

static bool LoadPlyMesh(const fs::path& filePath,
                        LoadedMeshData& outMesh,
                        std::string& errorMessage) {
    std::ifstream stream(filePath, std::ios::binary);
    if (!stream.is_open()) {
        errorMessage = "Failed to open PLY file: " + filePath.string();
        return false;
    }

    tinyply::PlyFile ply;
    try {
        if (!ply.parse_header(stream)) {
            errorMessage = "PLY header parsing failed: " + filePath.string();
            return false;
        }
    } catch (const std::exception& e) {
        errorMessage = std::string("PLY header parsing error: ") + e.what();
        return false;
    }

    auto positions = TryRequestPlyProperties(ply, "vertex", {"x", "y", "z"});
    auto normals = TryRequestPlyProperties(ply, "vertex", {"nx", "ny", "nz"});

    std::shared_ptr<tinyply::PlyData> texcoords = TryRequestPlyProperties(ply, "vertex", {"u", "v"});
    if (!texcoords) {
        texcoords = TryRequestPlyProperties(ply, "vertex", {"s", "t"});
    }
    if (!texcoords) {
        texcoords = TryRequestPlyProperties(ply, "vertex", {"texture_u", "texture_v"});
    }

    auto faces = TryRequestPlyProperties(ply, "face", {"vertex_indices"}, 3);
    if (!faces) {
        faces = TryRequestPlyProperties(ply, "face", {"vertex_index"}, 3);
    }

    try {
        ply.read(stream);
    } catch (const std::exception& e) {
        errorMessage = std::string("PLY payload read error: ") + e.what();
        return false;
    }

    if (!positions) {
        errorMessage = "PLY file is missing vertex position data";
        return false;
    }
    if (!faces) {
        errorMessage = "PLY file is missing face index data";
        return false;
    }

    const size_t vertexCount = positions->count;
    if (vertexCount == 0) {
        errorMessage = "PLY contains no vertices";
        return false;
    }

    auto assignPositions = [&](auto typeTag) -> bool {
        using ValueType = decltype(typeTag);
        const ValueType* src = reinterpret_cast<const ValueType*>(positions->buffer.get());
        const size_t values = positions->buffer.size_bytes() / sizeof(ValueType);
        if (values < vertexCount * 3) {
            errorMessage = "PLY vertex position buffer is smaller than expected";
            return false;
        }

        outMesh.vertices.resize(vertexCount);
        for (size_t i = 0; i < vertexCount; ++i) {
            const ValueType x = src[i * 3 + 0];
            const ValueType y = src[i * 3 + 1];
            const ValueType z = src[i * 3 + 2];
            SceneResources::MeshVertex vertex{};
            vertex.position = simd_make_float3(static_cast<float>(x),
                                               static_cast<float>(y),
                                               static_cast<float>(z));
            vertex.normal = simd_make_float3(0.0f, 0.0f, 0.0f);
            vertex.uv = simd_make_float2(0.0f, 0.0f);
            outMesh.vertices[i] = vertex;
        }
        return true;
    };

    switch (positions->t) {
        case tinyply::Type::FLOAT32:
            if (!assignPositions(float{})) {
                return false;
            }
            break;
        case tinyply::Type::FLOAT64:
            if (!assignPositions(double{})) {
                return false;
            }
            break;
        default:
            errorMessage = "PLY vertex positions must be stored as float or double";
            return false;
    }

    if (normals) {
        if (normals->count != vertexCount) {
            errorMessage = "PLY vertex normal count does not match position count";
            return false;
        }

        auto assignNormals = [&](auto typeTag) -> bool {
            using ValueType = decltype(typeTag);
            const ValueType* src = reinterpret_cast<const ValueType*>(normals->buffer.get());
            const size_t values = normals->buffer.size_bytes() / sizeof(ValueType);
            if (values < vertexCount * 3) {
                errorMessage = "PLY vertex normal buffer is smaller than expected";
                return false;
            }
            for (size_t i = 0; i < vertexCount; ++i) {
                const ValueType nx = src[i * 3 + 0];
                const ValueType ny = src[i * 3 + 1];
                const ValueType nz = src[i * 3 + 2];
                outMesh.vertices[i].normal = simd_make_float3(static_cast<float>(nx),
                                                              static_cast<float>(ny),
                                                              static_cast<float>(nz));
            }
            return true;
        };

        switch (normals->t) {
            case tinyply::Type::FLOAT32:
                if (!assignNormals(float{})) {
                    return false;
                }
                break;
            case tinyply::Type::FLOAT64:
                if (!assignNormals(double{})) {
                    return false;
                }
                break;
            default:
                errorMessage = "PLY vertex normals must be stored as float or double";
                return false;
        }
    }

    if (texcoords) {
        if (texcoords->count != vertexCount) {
            errorMessage = "PLY texture coordinate count does not match position count";
            return false;
        }

        auto assignTexcoords = [&](auto typeTag) -> bool {
            using ValueType = decltype(typeTag);
            const ValueType* src = reinterpret_cast<const ValueType*>(texcoords->buffer.get());
            const size_t values = texcoords->buffer.size_bytes() / sizeof(ValueType);
            if (values < vertexCount * 2) {
                errorMessage = "PLY texture coordinate buffer is smaller than expected";
                return false;
            }
            for (size_t i = 0; i < vertexCount; ++i) {
                const ValueType u = src[i * 2 + 0];
                const ValueType v = src[i * 2 + 1];
                outMesh.vertices[i].uv = simd_make_float2(static_cast<float>(u),
                                                          static_cast<float>(v));
            }
            return true;
        };

        switch (texcoords->t) {
            case tinyply::Type::FLOAT32:
                if (!assignTexcoords(float{})) {
                    return false;
                }
                break;
            case tinyply::Type::FLOAT64:
                if (!assignTexcoords(double{})) {
                    return false;
                }
                break;
            default:
                errorMessage = "PLY texture coordinates must be stored as float or double";
                return false;
        }
    }

    const size_t faceCount = faces->count;
    if (faceCount == 0) {
        errorMessage = "PLY contains no faces";
        return false;
    }

    outMesh.indices.clear();

    auto appendFaces = [&](auto typeTag) -> bool {
        using IndexType = decltype(typeTag);
        const IndexType* src = reinterpret_cast<const IndexType*>(faces->buffer.get());
        const size_t valueCount = faces->buffer.size_bytes() / sizeof(IndexType);
        if (valueCount == 0) {
            errorMessage = "PLY face index buffer is empty";
            return false;
        }
        if (valueCount % faceCount != 0) {
            errorMessage = "PLY uses variable-length face lists which are not currently supported";
            return false;
        }

        const size_t vertsPerFace = valueCount / faceCount;
        if (vertsPerFace < 3) {
            errorMessage = "PLY face has fewer than three indices";
            return false;
        }

        outMesh.indices.reserve(faceCount * (vertsPerFace - 2) * 3);

        size_t offset = 0;
        for (size_t face = 0; face < faceCount; ++face) {
            IndexType first = src[offset + 0];
            if constexpr (std::is_signed_v<IndexType>) {
                if (first < 0) {
                    errorMessage = "PLY face index is negative";
                    return false;
                }
            }
            if (static_cast<size_t>(first) >= vertexCount) {
                errorMessage = "PLY face index is out of range";
                return false;
            }

            for (size_t k = 1; k + 1 < vertsPerFace; ++k) {
                IndexType b = src[offset + k];
                IndexType c = src[offset + k + 1];

                if constexpr (std::is_signed_v<IndexType>) {
                    if (b < 0 || c < 0) {
                        errorMessage = "PLY face index is negative";
                        return false;
                    }
                }

                if (static_cast<size_t>(b) >= vertexCount || static_cast<size_t>(c) >= vertexCount) {
                    errorMessage = "PLY face index is out of range";
                    return false;
                }

                outMesh.indices.push_back(static_cast<uint32_t>(first));
                outMesh.indices.push_back(static_cast<uint32_t>(b));
                outMesh.indices.push_back(static_cast<uint32_t>(c));
            }

            offset += vertsPerFace;
        }

        return true;
    };

    switch (faces->t) {
        case tinyply::Type::UINT8:
            if (!appendFaces(uint8_t{})) {
                return false;
            }
            break;
        case tinyply::Type::INT8:
            if (!appendFaces(int8_t{})) {
                return false;
            }
            break;
        case tinyply::Type::UINT16:
            if (!appendFaces(uint16_t{})) {
                return false;
            }
            break;
        case tinyply::Type::INT16:
            if (!appendFaces(int16_t{})) {
                return false;
            }
            break;
        case tinyply::Type::UINT32:
            if (!appendFaces(uint32_t{})) {
                return false;
            }
            break;
        case tinyply::Type::INT32:
            if (!appendFaces(int32_t{})) {
                return false;
            }
            break;
        default:
            errorMessage = "PLY face indices must be 8/16/32-bit integers";
            return false;
    }

    for (size_t i = 0; i + 2 < outMesh.indices.size(); i += 3) {
        auto& v0 = outMesh.vertices[outMesh.indices[i + 0]];
        auto& v1 = outMesh.vertices[outMesh.indices[i + 1]];
        auto& v2 = outMesh.vertices[outMesh.indices[i + 2]];
        ApplyFallbackNormals(v0, v1, v2);
    }

    return true;
}

static simd::float4x4 MakeScaleMatrix(const simd::float3& scale) {
    simd::float4x4 matrix = matrix_identity_float4x4;
    matrix.columns[0] = simd_make_float4(scale.x, 0.0f, 0.0f, 0.0f);
    matrix.columns[1] = simd_make_float4(0.0f, scale.y, 0.0f, 0.0f);
    matrix.columns[2] = simd_make_float4(0.0f, 0.0f, scale.z, 0.0f);
    return matrix;
}

static simd::float4x4 MakeTranslationMatrix(const simd::float3& translation) {
    simd::float4x4 matrix = matrix_identity_float4x4;
    matrix.columns[3] = simd_make_float4(translation, 1.0f);
    return matrix;
}

static simd::float4x4 MakeRotationMatrix(const simd::float3& degreesEuler) {
    const float radX = degreesEuler.x * (kPi / 180.0f);
    const float radY = degreesEuler.y * (kPi / 180.0f);
    const float radZ = degreesEuler.z * (kPi / 180.0f);

    const float sx = std::sinf(radX);
    const float cx = std::cosf(radX);
    const float sy = std::sinf(radY);
    const float cy = std::cosf(radY);
    const float sz = std::sinf(radZ);
    const float cz = std::cosf(radZ);

    simd::float4x4 rotX = matrix_identity_float4x4;
    rotX.columns[1] = simd_make_float4(0.0f, cx, sx, 0.0f);
    rotX.columns[2] = simd_make_float4(0.0f, -sx, cx, 0.0f);

    simd::float4x4 rotY = matrix_identity_float4x4;
    rotY.columns[0] = simd_make_float4(cy, 0.0f, -sy, 0.0f);
    rotY.columns[2] = simd_make_float4(sy, 0.0f, cy, 0.0f);

    simd::float4x4 rotZ = matrix_identity_float4x4;
    rotZ.columns[0] = simd_make_float4(cz, sz, 0.0f, 0.0f);
    rotZ.columns[1] = simd_make_float4(-sz, cz, 0.0f, 0.0f);

    return simd_mul(simd_mul(rotZ, rotY), rotX);
}

static simd::float4x4 ComposeTransform(const simd::float3& translation,
                                       const simd::float3& rotationDegrees,
                                       const simd::float3& scale) {
    simd::float4x4 scaleMat = MakeScaleMatrix(scale);
    simd::float4x4 rotMat = MakeRotationMatrix(rotationDegrees);
    simd::float4x4 transMat = MakeTranslationMatrix(translation);
    return simd_mul(transMat, simd_mul(rotMat, scaleMat));
}

std::optional<fs::path> ResolveDefaultScenesDirectory() {
#if TARGET_OS_OSX
    @autoreleasepool {
        NSBundle* bundle = [NSBundle mainBundle];
        if (bundle) {
            NSString* resourcePath = [bundle resourcePath];
            if (resourcePath) {
                fs::path candidate([resourcePath UTF8String]);
                candidate /= "assets";
                std::error_code ec;
                if (fs::exists(candidate, ec) && fs::is_directory(candidate, ec)) {
                    fs::path canonical = fs::canonical(candidate, ec);
                    if (!ec) {
                        return canonical;
                    }
                }
            }
        }
    }
#endif

    std::error_code ec;
    fs::path cwd = fs::current_path(ec);
    if (!ec) {
        fs::path candidate = cwd / "assets";
        if (fs::exists(candidate, ec) && fs::is_directory(candidate, ec)) {
            fs::path canonical = fs::canonical(candidate, ec);
            if (!ec) {
                return canonical;
            }
        }
    }
    return std::nullopt;
}

}  // namespace

namespace PathTracer {

SceneManager::SceneManager() {
    if (auto defaultDir = ResolveDefaultScenesDirectory()) {
        setSceneDirectory(defaultDir->string());
    } else {
        refresh(nullptr);
    }
}

SceneManager::SceneManager(std::string scenesDirectory) {
    setSceneDirectory(scenesDirectory);
}

bool SceneManager::setSceneDirectory(const std::string& directory, std::string* errorMessage) {
    std::error_code ec;
    fs::path absolute;
    if (!directory.empty()) {
        absolute = fs::weakly_canonical(fs::path(directory), ec);
        if (ec) {
            if (errorMessage) {
                *errorMessage = "Failed to canonicalize scenes directory: " + directory;
            }
            return false;
        }
        if (!fs::exists(absolute, ec) || !fs::is_directory(absolute, ec)) {
            if (errorMessage) {
                *errorMessage = "Scene directory does not exist or is not a directory: " +
                                absolute.string();
            }
            return false;
        }
        m_sceneDirectory = absolute.string();
    } else {
        m_sceneDirectory.clear();
    }
    return refresh(errorMessage);
}

bool SceneManager::refresh(std::string* errorMessage) {
    return discoverScenes(errorMessage);
}

const SceneManager::SceneInfo* SceneManager::currentScene() const {
    if (auto it = m_sceneIndexById.find(m_currentSceneId); it != m_sceneIndexById.end()) {
        return &m_scenes[it->second];
    }
    return nullptr;
}

bool SceneManager::loadScene(const std::string& identifier,
                             SceneResources& resources,
                             RenderSettings& inOutSettings,
                             std::string* errorMessage) {
    const SceneInfo* info = findScene(identifier);
    if (!info) {
        if (errorMessage) {
            *errorMessage = "Unknown scene identifier: " + identifier;
        }
        return false;
    }

    if (!loadSceneFromPath(info->filePath, resources, inOutSettings, errorMessage)) {
        return false;
    }

    m_currentSceneId = identifier;
    return true;
}

bool SceneManager::loadSceneFromPath(const std::string& path,
                                     SceneResources& resources,
                                     RenderSettings& inOutSettings,
                                     std::string* errorMessage) {
    std::ifstream stream(path);
    if (!stream.is_open()) {
        if (errorMessage) {
            *errorMessage = "Failed to open scene file: " + path;
        }
        return false;
    }

    resources.clear();
    RenderSettings parsedSettings = inOutSettings;
    const RenderSettings defaults{};
    parsedSettings.cameraVerticalFov = defaults.cameraVerticalFov;
    parsedSettings.cameraDefocusAngle = defaults.cameraDefocusAngle;
    parsedSettings.cameraFocusDistance = defaults.cameraFocusDistance;
    parsedSettings.backgroundMode = defaults.backgroundMode;
    parsedSettings.backgroundColor = defaults.backgroundColor;
    parsedSettings.environmentMapPath = defaults.environmentMapPath;
    parsedSettings.environmentRotation = defaults.environmentRotation;
    parsedSettings.environmentIntensity = defaults.environmentIntensity;
    parsedSettings.fireflyClampEnabled = defaults.fireflyClampEnabled;
    parsedSettings.fireflyClampFactor = defaults.fireflyClampFactor;
    parsedSettings.fireflyClampFloor = defaults.fireflyClampFloor;
    parsedSettings.throughputClamp = defaults.throughputClamp;
    parsedSettings.specularTailClampBase = defaults.specularTailClampBase;
    parsedSettings.specularTailClampRoughnessScale = defaults.specularTailClampRoughnessScale;
    parsedSettings.minSpecularPdf = defaults.minSpecularPdf;
    parsedSettings.renderWidth = defaults.renderWidth;
    parsedSettings.renderHeight = defaults.renderHeight;
    parsedSettings.enableSoftwareRayTracing = defaults.enableSoftwareRayTracing;

    std::string parseError;
    if (!parseScene(stream, resources, parsedSettings, parseError)) {
        resources.clear();
        if (errorMessage) {
            *errorMessage = "Failed parsing scene '" + path + "': " + parseError;
        }
        return false;
    }

    inOutSettings = parsedSettings;
    return true;
}

bool SceneManager::discoverScenes(std::string* errorMessage) {
    m_scenes.clear();
    m_sceneIndexById.clear();

    if (m_sceneDirectory.empty()) {
        return true;
    }

    std::error_code ec;
    fs::path directoryPath(m_sceneDirectory);
    if (!fs::exists(directoryPath, ec) || !fs::is_directory(directoryPath, ec)) {
        if (errorMessage) {
            *errorMessage = "Scene directory is not accessible: " + m_sceneDirectory;
        }
        return false;
    }

    std::vector<SceneInfo> discovered;
    for (const auto& entry : fs::directory_iterator(directoryPath, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file()) {
            continue;
        }
        const fs::path& filePath = entry.path();
        if (filePath.extension() != ".scene") {
            continue;
        }

        SceneInfo info{};
        fs::path absolutePath = fs::absolute(filePath, ec);
        std::string pathString;
        if (!ec) {
            pathString = absolutePath.string();
        } else {
            ec.clear();
            pathString = filePath.string();
        }
        info.filePath = std::move(pathString);
        info.identifier = filePath.stem().string();
        info.displayName = readDisplayName(info.filePath);
        if (info.displayName.empty()) {
            info.displayName = info.identifier;
        }

        discovered.push_back(std::move(info));
    }

    if (ec) {
        if (errorMessage) {
            *errorMessage = "Failed iterating scene directory: " + ec.message();
        }
        return false;
    }

    std::sort(discovered.begin(), discovered.end(), [](const SceneInfo& a, const SceneInfo& b) {
        return a.displayName < b.displayName;
    });

    m_scenes = std::move(discovered);
    for (size_t i = 0; i < m_scenes.size(); ++i) {
        m_sceneIndexById[m_scenes[i].identifier] = i;
    }
    return true;
}

bool SceneManager::parseScene(std::istream& stream,
                              SceneResources& resources,
                              RenderSettings& inOutSettings,
                              std::string& errorMessage) const {
    std::string line;
    size_t lineNumber = 0;
    std::string pending;
    size_t pendingStartLine = 0;
    std::unordered_map<std::string, uint32_t> materialIndicesByName;

    auto flush = [&](const std::string& content, size_t startLine) -> bool {
        if (content.empty()) {
            return true;
        }

        auto tokens = tokenize(content);
        auto keywordIt = tokens.find(kKeywordToken);
        if (keywordIt == tokens.end()) {
            return true;
        }

        const std::string& keyword = keywordIt->second;
        std::string localError;
        bool success = false;

        if (keyword == "camera") {
            success = parseCamera(tokens, inOutSettings, localError);
        } else if (keyword == "renderer") {
            success = parseRenderer(tokens, inOutSettings, localError);
        } else if (keyword == "background") {
            success = parseBackground(tokens, inOutSettings, localError, m_sceneDirectory);
        } else if (keyword == "material") {
            success = parseMaterial(tokens, resources, localError, materialIndicesByName);
        } else if (keyword == "sphere") {
            success = parseSphere(tokens, resources, localError);
        } else if (keyword == "box") {
            success = parseBox(tokens, resources, localError);
        } else if (keyword == "rectangle" || keyword == "rect") {
            success = parseRectangle(tokens, resources, localError);
        } else if (keyword == "mesh") {
            success = parseMesh(tokens, resources, localError, m_sceneDirectory, materialIndicesByName);
        } else {
            return true;
        }

        if (!success) {
            std::ostringstream oss;
            oss << "line " << startLine << ": " << localError;
            errorMessage = oss.str();
        }
        return success;
    };

    while (std::getline(stream, line)) {
        ++lineNumber;
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            if (!pending.empty()) {
                if (!flush(pending, pendingStartLine == 0 ? lineNumber : pendingStartLine)) {
                    return false;
                }
                pending.clear();
                pendingStartLine = 0;
            }
            continue;
        }

        bool continuation = false;
        if (!trimmed.empty() && trimmed.back() == '\\') {
            continuation = true;
            trimmed.pop_back();
            trimmed = trim(trimmed);
        }

        if (!trimmed.empty()) {
            if (pending.empty()) {
                pending = trimmed;
                pendingStartLine = lineNumber;
            } else {
                pending.append(" ");
                pending.append(trimmed);
            }
        }

        if (continuation) {
            continue;
        }

        if (!pending.empty()) {
            if (!flush(pending, pendingStartLine)) {
                return false;
            }
            pending.clear();
            pendingStartLine = 0;
        }
    }

    if (!pending.empty()) {
        if (!flush(pending, pendingStartLine == 0 ? lineNumber : pendingStartLine)) {
            return false;
        }
    }

    return true;
}

std::unordered_map<std::string, std::string> SceneManager::tokenize(const std::string& line) {
    std::unordered_map<std::string, std::string> tokens;

    std::istringstream stream(line);
    std::string word;
    if (!(stream >> word)) {
        return tokens;
    }

    tokens.emplace(kKeywordToken, word);
    while (stream >> word) {
        auto equalsPos = word.find('=');
        if (equalsPos == std::string::npos) {
            continue;
        }

        std::string key = word.substr(0, equalsPos);
        std::string value = word.substr(equalsPos + 1);
        tokens[std::move(key)] = std::move(value);
    }

    return tokens;
}

std::string SceneManager::trim(const std::string& value) {
    size_t start = 0;
    size_t end = value.size();
    while (start < end && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

bool SceneManager::parseFloat(const std::string& value, float& out) {
    std::string trimmed = trim(value);
    if (trimmed.empty()) {
        return false;
    }

    char* end = nullptr;
    errno = 0;
    float parsed = std::strtof(trimmed.c_str(), &end);
    if (errno != 0 || end == trimmed.c_str()) {
        return false;
    }

    while (end && *end != '\0') {
        if (!std::isspace(static_cast<unsigned char>(*end))) {
            return false;
        }
        ++end;
    }

    out = parsed;
    return true;
}

bool SceneManager::parseUInt(const std::string& value, uint32_t& out) {
    std::string trimmed = trim(value);
    if (trimmed.empty()) {
        return false;
    }

    char* end = nullptr;
    errno = 0;
    unsigned long parsed = std::strtoul(trimmed.c_str(), &end, 10);
    if (errno != 0 || end == trimmed.c_str()) {
        return false;
    }

    while (end && *end != '\0') {
        if (!std::isspace(static_cast<unsigned char>(*end))) {
            return false;
        }
        ++end;
    }

    if (parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    out = static_cast<uint32_t>(parsed);
    return true;
}

bool SceneManager::parseFloat3(const std::string& value, simd::float3& out) {
    std::istringstream stream(value);
    std::string component;
    float components[3] = {0.0f, 0.0f, 0.0f};
    int index = 0;

    while (std::getline(stream, component, ',')) {
        if (index >= 3) {
            return false;
        }
        if (!parseFloat(component, components[index])) {
            return false;
        }
        ++index;
    }

    if (index != 3) {
        return false;
    }

    out = simd_make_float3(components[0], components[1], components[2]);
    return true;
}

bool SceneManager::parseFloatRange(const std::string& value,
                                   float& outMin,
                                   float& outMax,
                                   bool& outIsFixed) {
    std::string trimmed = trim(value);
    if (trimmed.empty()) {
        return false;
    }

    auto commaPos = trimmed.find(',');
    if (commaPos == std::string::npos) {
        if (!parseFloat(trimmed, outMin)) {
            return false;
        }
        outMax = outMin;
        outIsFixed = true;
        return true;
    }

    std::string first = trimmed.substr(0, commaPos);
    std::string second = trimmed.substr(commaPos + 1);
    if (!parseFloat(first, outMin)) {
        return false;
    }
    if (!parseFloat(second, outMax)) {
        return false;
    }

    if (outMin > outMax) {
        std::swap(outMin, outMax);
    }

    outIsFixed = std::fabs(outMax - outMin) < 1e-6f;
    return true;
}

bool SceneManager::parseMaterialType(const std::string& value,
                                     PathTracerShaderTypes::MaterialType& out) {
    std::string lower;
    lower.reserve(value.size());
    for (char ch : value) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }

    if (lower == "lambert" || lower == "lambertian") {
        out = PathTracerShaderTypes::MaterialType::Lambertian;
        return true;
    }
    if (lower == "metal" || lower == "metallic") {
        out = PathTracerShaderTypes::MaterialType::Metal;
        return true;
    }
    if (lower == "dielectric" || lower == "glass") {
        out = PathTracerShaderTypes::MaterialType::Dielectric;
        return true;
    }
    if (lower == "diffuse_light" || lower == "light" || lower == "emissive") {
        out = PathTracerShaderTypes::MaterialType::DiffuseLight;
        return true;
    }
    if (lower == "plastic") {
        out = PathTracerShaderTypes::MaterialType::Plastic;
        return true;
    }
    if (lower == "sss" || lower == "subsurface") {
        out = PathTracerShaderTypes::MaterialType::Subsurface;
        return true;
    }
    if (lower == "carpaint" || lower == "car_paint" || lower == "automotive") {
        out = PathTracerShaderTypes::MaterialType::CarPaint;
        return true;
    }

    return false;
}

bool SceneManager::parseCamera(const std::unordered_map<std::string, std::string>& tokens,
                               RenderSettings& inOutSettings,
                               std::string& errorMessage) {
    if (auto it = tokens.find("target"); it != tokens.end()) {
        simd::float3 target{};
        if (!parseFloat3(it->second, target)) {
            errorMessage = "camera target expects three comma-separated floats";
            return false;
        }
        inOutSettings.cameraTarget = target;
    }

    if (auto it = tokens.find("distance"); it != tokens.end()) {
        float distance = 0.0f;
        if (!parseFloat(it->second, distance)) {
            errorMessage = "camera distance expects a float";
            return false;
        }
        inOutSettings.cameraDistance = std::max(distance, 0.0f);
    }

    if (auto it = tokens.find("yaw"); it != tokens.end()) {
        float yaw = 0.0f;
        if (!parseFloat(it->second, yaw)) {
            errorMessage = "camera yaw expects a float (radians)";
            return false;
        }
        inOutSettings.cameraYaw = yaw;
    }

    if (auto it = tokens.find("pitch"); it != tokens.end()) {
        float pitch = 0.0f;
        if (!parseFloat(it->second, pitch)) {
            errorMessage = "camera pitch expects a float (radians)";
            return false;
        }
        inOutSettings.cameraPitch = pitch;
    }

    if (auto it = tokens.find("vfov"); it != tokens.end()) {
        float vfov = 0.0f;
        if (!parseFloat(it->second, vfov)) {
            errorMessage = "camera vfov expects a float (degrees)";
            return false;
        }
        inOutSettings.cameraVerticalFov = vfov;
    }

    if (auto it = tokens.find("defocusAngle"); it != tokens.end()) {
        float defocus = 0.0f;
        if (!parseFloat(it->second, defocus)) {
            errorMessage = "camera defocusAngle expects a float (degrees)";
            return false;
        }
        inOutSettings.cameraDefocusAngle = std::max(defocus, 0.0f);
    }

    if (auto it = tokens.find("focusDist"); it != tokens.end()) {
        float focus = 0.0f;
        if (!parseFloat(it->second, focus)) {
            errorMessage = "camera focusDist expects a float";
            return false;
        }
        inOutSettings.cameraFocusDistance = focus;
    }

    return true;
}

bool SceneManager::parseRenderer(const std::unordered_map<std::string, std::string>& tokens,
                                 RenderSettings& inOutSettings,
                                 std::string& errorMessage) {
    if (auto it = tokens.find("samplesPerFrame"); it != tokens.end()) {
        uint32_t value = 0;
        if (!parseUInt(it->second, value)) {
            errorMessage = "renderer samplesPerFrame expects an integer";
            return false;
        }
        inOutSettings.samplesPerFrame = std::max<uint32_t>(1, value);
    }

    if (auto it = tokens.find("width"); it != tokens.end()) {
        uint32_t value = 0;
        if (!parseUInt(it->second, value)) {
            errorMessage = "renderer width expects an integer";
            return false;
        }
        inOutSettings.renderWidth = std::max<uint32_t>(value, 8u);
    }

    if (auto it = tokens.find("height"); it != tokens.end()) {
        uint32_t value = 0;
        if (!parseUInt(it->second, value)) {
            errorMessage = "renderer height expects an integer";
            return false;
        }
        inOutSettings.renderHeight = std::max<uint32_t>(value, 8u);
    }

    if (auto it = tokens.find("maxDepth"); it != tokens.end()) {
        uint32_t value = 0;
        if (!parseUInt(it->second, value)) {
            errorMessage = "renderer maxDepth expects an integer";
            return false;
        }
        inOutSettings.maxDepth = value;
    }

    if (auto it = tokens.find("tonemap"); it != tokens.end()) {
        uint32_t value = 0;
        if (!parseUInt(it->second, value)) {
            errorMessage = "renderer tonemap expects an integer";
            return false;
        }
        inOutSettings.tonemapMode = std::max<uint32_t>(1u, std::min<uint32_t>(value, 4u));
    }

    if (auto it = tokens.find("exposure"); it != tokens.end()) {
        float exposure = 0.0f;
        if (!parseFloat(it->second, exposure)) {
            errorMessage = "renderer exposure expects a float";
            return false;
        }
        inOutSettings.exposure = exposure;
    }

    if (auto it = tokens.find("envRotation"); it != tokens.end()) {
        float rotationDegrees = 0.0f;
        if (!parseFloat(it->second, rotationDegrees)) {
            errorMessage = "renderer envRotation expects a float (degrees)";
            return false;
        }
        inOutSettings.environmentRotation = rotationDegrees * (kPi / 180.0f);
    }

    if (auto it = tokens.find("envIntensity"); it != tokens.end()) {
        float intensity = 0.0f;
        if (!parseFloat(it->second, intensity)) {
            errorMessage = "renderer envIntensity expects a float";
            return false;
        }
        inOutSettings.environmentIntensity = std::max(intensity, 0.0f);
    }

    if (auto it = tokens.find("reinhardWhite"); it != tokens.end()) {
        float white = 0.0f;
        if (!parseFloat(it->second, white)) {
            errorMessage = "renderer reinhardWhite expects a float";
            return false;
        }
        inOutSettings.reinhardWhitePoint = white;
    }

    if (auto it = tokens.find("seed"); it != tokens.end()) {
        uint32_t seed = 0;
        if (!parseUInt(it->second, seed)) {
            errorMessage = "renderer seed expects an integer";
            return false;
        }
        inOutSettings.fixedRngSeed = seed;
    }

    if (auto it = tokens.find("russianRoulette"); it != tokens.end()) {
        uint32_t flag = 0;
        if (!parseUInt(it->second, flag)) {
            errorMessage = "renderer russianRoulette expects 0 or 1";
            return false;
        }
        inOutSettings.enableRussianRoulette = (flag != 0);
    }

    if (auto it = tokens.find("acesVariant"); it != tokens.end()) {
        uint32_t variant = 0;
        if (!parseUInt(it->second, variant)) {
            errorMessage = "renderer acesVariant expects an integer";
            return false;
        }
        inOutSettings.acesVariant = variant;
    }

    auto parseSoftwareFlag = [&](const std::string& tokenName) -> bool {
        auto itTok = tokens.find(tokenName);
        if (itTok == tokens.end()) {
            return true;
        }
        uint32_t flag = 0;
        if (!parseUInt(itTok->second, flag)) {
            errorMessage = "renderer " + tokenName + " expects 0 or 1";
            return false;
        }
        inOutSettings.enableSoftwareRayTracing = (flag != 0);
        return true;
    };

    if (!parseSoftwareFlag("enableSoftwareRayTracing")) {
        return false;
    }
    if (!parseSoftwareFlag("softwareRayTracing")) {
        return false;
    }
    if (!parseSoftwareFlag("forceSoftwareBvh")) {
        return false;
    }

    if (auto it = tokens.find("sss"); it != tokens.end()) {
        std::string lower;
        lower.reserve(it->second.size());
        for (char ch : it->second) {
            lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
        if (lower == "off" || lower == "disabled" || lower == "0") {
            inOutSettings.sssMode = RenderSettings::SssMode::Off;
        } else if (lower == "separable" || lower == "diffusion" || lower == "approx") {
            inOutSettings.sssMode = RenderSettings::SssMode::Separable;
        } else if (lower == "randomwalk" || lower == "random_walk" || lower == "random-walk") {
            inOutSettings.sssMode = RenderSettings::SssMode::RandomWalk;
        } else {
            errorMessage = "renderer sss expects off, separable, or randomwalk";
            return false;
        }
    }

    if (auto it = tokens.find("sssMaxSteps"); it != tokens.end()) {
        uint32_t value = 0;
        if (!parseUInt(it->second, value)) {
            errorMessage = "renderer sssMaxSteps expects an integer";
            return false;
        }
        inOutSettings.sssMaxSteps = std::max<uint32_t>(1u, value);
    }

    if (auto it = tokens.find("fireflyClampEnabled"); it != tokens.end()) {
        uint32_t flag = 0;
        if (!parseUInt(it->second, flag)) {
            errorMessage = "renderer fireflyClampEnabled expects 0 or 1";
            return false;
        }
        inOutSettings.fireflyClampEnabled = (flag != 0);
    }

    if (auto it = tokens.find("fireflyClampFactor"); it != tokens.end()) {
        float value = 0.0f;
        if (!parseFloat(it->second, value)) {
            errorMessage = "renderer fireflyClampFactor expects a float";
            return false;
        }
        inOutSettings.fireflyClampFactor = std::max(value, 0.0f);
    }

    if (auto it = tokens.find("fireflyClampFloor"); it != tokens.end()) {
        float value = 0.0f;
        if (!parseFloat(it->second, value)) {
            errorMessage = "renderer fireflyClampFloor expects a float";
            return false;
        }
        inOutSettings.fireflyClampFloor = std::max(value, 0.0f);
    }

    if (auto it = tokens.find("throughputClamp"); it != tokens.end()) {
        float value = 0.0f;
        if (!parseFloat(it->second, value)) {
            errorMessage = "renderer throughputClamp expects a float";
            return false;
        }
        inOutSettings.throughputClamp = std::max(value, 0.0f);
    }

    if (auto it = tokens.find("specularTailClampBase"); it != tokens.end()) {
        float value = 0.0f;
        if (!parseFloat(it->second, value)) {
            errorMessage = "renderer specularTailClampBase expects a float";
            return false;
        }
        inOutSettings.specularTailClampBase = std::max(value, 0.0f);
    }

    if (auto it = tokens.find("specularTailClampRoughnessScale"); it != tokens.end()) {
        float value = 0.0f;
        if (!parseFloat(it->second, value)) {
            errorMessage = "renderer specularTailClampRoughnessScale expects a float";
            return false;
        }
        inOutSettings.specularTailClampRoughnessScale = std::max(value, 0.0f);
    }

    if (auto it = tokens.find("minSpecularPdf"); it != tokens.end()) {
        float value = 0.0f;
        if (!parseFloat(it->second, value)) {
            errorMessage = "renderer minSpecularPdf expects a float";
            return false;
        }
        inOutSettings.minSpecularPdf = std::max(value, 1.0e-7f);
    }

    return true;
}

bool SceneManager::parseBackground(const std::unordered_map<std::string, std::string>& tokens,
                                   RenderSettings& inOutSettings,
                                   std::string& errorMessage,
                                   const std::string& sceneDirectory) {
    auto solidIt = tokens.find("solid");
    auto envIt = tokens.find("env");

    if (solidIt != tokens.end() && envIt != tokens.end()) {
        errorMessage = "background cannot specify both solid and env";
        return false;
    }

    if (solidIt != tokens.end()) {
        simd::float3 color{};
        if (!parseFloat3(solidIt->second, color)) {
            errorMessage = "background solid expects three floats";
            return false;
        }
        inOutSettings.backgroundMode = RenderSettings::BackgroundMode::Solid;
        inOutSettings.backgroundColor = color;
        inOutSettings.environmentMapPath.clear();
        return true;
    }

    if (envIt != tokens.end()) {
        std::string value = envIt->second;
        fs::path envPath(value);
        if (envPath.is_relative()) {
            fs::path base(sceneDirectory);
            if (envPath.has_parent_path()) {
                envPath = base / envPath;
            } else {
                envPath = base / "HDR" / envPath;
            }
        }

        std::error_code ec;
        fs::path canonical = fs::weakly_canonical(envPath, ec);
        if (ec || !fs::exists(canonical)) {
            errorMessage = "background env map not found: " + envPath.string();
            return false;
        }

        inOutSettings.backgroundMode = RenderSettings::BackgroundMode::Environment;
        inOutSettings.backgroundColor = simd_make_float3(0.0f, 0.0f, 0.0f);
        inOutSettings.environmentMapPath = canonical.string();
        return true;
    }

    inOutSettings.backgroundMode = RenderSettings::BackgroundMode::Gradient;
    inOutSettings.backgroundColor = simd_make_float3(0.0f, 0.0f, 0.0f);
    inOutSettings.environmentMapPath.clear();
    return true;
}

bool SceneManager::parseMaterial(const std::unordered_map<std::string, std::string>& tokens,
                                 SceneResources& resources,
                                 std::string& errorMessage,
                                 std::unordered_map<std::string, uint32_t>& materialIndicesByName) {
    auto typeIt = tokens.find("type");
    if (typeIt == tokens.end()) {
        errorMessage = "material requires a type token";
        return false;
    }

    PathTracerShaderTypes::MaterialType type{};
    if (!parseMaterialType(typeIt->second, type)) {
        errorMessage = "material type is not recognized";
        return false;
    }

    simd::float3 baseColor = simd_make_float3(1.0f, 1.0f, 1.0f);
    if (auto itBase = tokens.find("base"); itBase != tokens.end()) {
        if (!parseFloat3(itBase->second, baseColor)) {
            errorMessage = "material base expects three floats";
            return false;
        }
    } else if (auto it = tokens.find("albedo"); it != tokens.end()) {
        if (!parseFloat3(it->second, baseColor)) {
            errorMessage = "material albedo expects three floats";
            return false;
        }
    } else if (auto itColor = tokens.find("color"); itColor != tokens.end()) {
        if (!parseFloat3(itColor->second, baseColor)) {
            errorMessage = "material color expects three floats";
            return false;
        }
    }

    float roughness = 0.0f;
    bool roughnessExplicit = false;
    if (auto it = tokens.find("roughness"); it != tokens.end()) {
        if (!parseFloat(it->second, roughness)) {
            errorMessage = "material roughness expects a float";
            return false;
        }
        roughness = std::clamp(roughness, 0.0f, 1.0f);
        roughnessExplicit = true;
    }

    float fuzz = 0.0f;
    if (auto it = tokens.find("fuzz"); it != tokens.end()) {
        if (!parseFloat(it->second, fuzz)) {
            errorMessage = "material fuzz expects a float";
            return false;
        }
        fuzz = std::clamp(fuzz, 0.0f, 1.0f);
    }
    if (!roughnessExplicit) {
        roughness = fuzz;
    }

    float ior = 1.5f;
    bool iorExplicit = false;
    if (auto it = tokens.find("ior"); it != tokens.end()) {
        if (!parseFloat(it->second, ior)) {
            errorMessage = "material ior expects a float";
            return false;
        }
        iorExplicit = true;
    }
    float coatIorValue = 1.5f;

    simd::float3 emission = simd_make_float3(0.0f, 0.0f, 0.0f);
    if (auto it = tokens.find("emit"); it != tokens.end()) {
        if (!parseFloat3(it->second, emission)) {
            errorMessage = "material emit expects three floats";
            return false;
        }
    } else if (auto itE = tokens.find("emission"); itE != tokens.end()) {
        if (!parseFloat3(itE->second, emission)) {
            errorMessage = "material emission expects three floats";
            return false;
        }
    }

    bool emissionUsesEnvironment = false;
    if (auto it = tokens.find("emitEnv"); it != tokens.end()) {
        uint32_t flag = 0;
        if (!parseUInt(it->second, flag)) {
            errorMessage = "material emitEnv expects 0 or 1";
            return false;
        }
        emissionUsesEnvironment = (flag != 0);
    } else if (auto itPortal = tokens.find("envPortal"); itPortal != tokens.end()) {
        uint32_t flag = 0;
        if (!parseUInt(itPortal->second, flag)) {
            errorMessage = "material envPortal expects 0 or 1";
            return false;
        }
        emissionUsesEnvironment = (flag != 0);
    }

    if (type == PathTracerShaderTypes::MaterialType::DiffuseLight) {
       roughness = 0.0f;
       ior = 1.0f;
   }

    std::string materialName;
    if (auto itName = tokens.find("name"); itName != tokens.end()) {
        materialName = itName->second;
    }

    const bool isPlastic = (type == PathTracerShaderTypes::MaterialType::Plastic);
    const bool isSubsurface = (type == PathTracerShaderTypes::MaterialType::Subsurface);
    const bool isCarPaint = (type == PathTracerShaderTypes::MaterialType::CarPaint);

    float coatRoughnessValue = 0.0f;
    if (isPlastic || isSubsurface) {
        coatRoughnessValue = 0.05f;
    } else if (isCarPaint) {
        coatRoughnessValue = 0.04f;
    }
    float coatThicknessValue = 0.0f;
    simd::float3 coatTintValue = simd_make_float3(1.0f, 1.0f, 1.0f);
    simd::float3 coatAbsorptionValue = simd_make_float3(0.0f, 0.0f, 0.0f);
    bool sssCoatEnabledValue = false;

    float carpaintBaseMetallicValue = 0.0f;
    float carpaintBaseRoughnessValue = roughness;
    bool carpaintBaseRoughnessExplicit = false;
    simd::float3 carpaintBaseEtaValue = simd_make_float3(kDefaultCarpaintBaseEta[0],
                                                         kDefaultCarpaintBaseEta[1],
                                                         kDefaultCarpaintBaseEta[2]);
    simd::float3 carpaintBaseKValue = simd_make_float3(kDefaultCarpaintBaseK[0],
                                                       kDefaultCarpaintBaseK[1],
                                                       kDefaultCarpaintBaseK[2]);
    bool carpaintBaseConductorExplicit = false;
    bool carpaintHasBaseConductorValue = false;
    simd::float3 carpaintBaseTintValue = simd_make_float3(1.0f, 1.0f, 1.0f);
    float carpaintFlakeDensityValue = 0.0f;
    float carpaintFlakeRoughnessValue = 0.15f;
    float carpaintFlakeAnisotropyValue = 0.0f;
    float carpaintFlakeScaleValue = 1.0f;
    float carpaintFlakeNormalStrengthValue = 0.35f;
    float carpaintFlakeReflectanceScaleValue = 1.0f;

    if (isCarPaint) {
        if (auto it = tokens.find("baseMetallic"); it != tokens.end()) {
            if (!parseFloat(it->second, carpaintBaseMetallicValue)) {
                errorMessage = "material baseMetallic expects a float";
                return false;
            }
            carpaintBaseMetallicValue = std::clamp(carpaintBaseMetallicValue, 0.0f, 1.0f);
        }
        if (!roughnessExplicit) {
            carpaintBaseRoughnessValue = 0.2f;
        }
        if (auto it = tokens.find("baseRoughness"); it != tokens.end()) {
            if (!parseFloat(it->second, carpaintBaseRoughnessValue)) {
                errorMessage = "material baseRoughness expects a float";
                return false;
            }
            carpaintBaseRoughnessValue = std::clamp(carpaintBaseRoughnessValue, 0.0f, 1.0f);
            carpaintBaseRoughnessExplicit = true;
        }
        if (!carpaintBaseRoughnessExplicit && roughnessExplicit) {
            carpaintBaseRoughnessValue = std::clamp(roughness, 0.0f, 1.0f);
        }
        if (auto it = tokens.find("flakeDensity"); it != tokens.end()) {
            if (!parseFloat(it->second, carpaintFlakeDensityValue)) {
                errorMessage = "material flakeDensity expects a float";
                return false;
            }
            carpaintFlakeDensityValue = std::max(carpaintFlakeDensityValue, 0.0f);
        } else {
            carpaintFlakeDensityValue = 2000000.0f;
        }
        if (auto it = tokens.find("flakeRoughness"); it != tokens.end()) {
            if (!parseFloat(it->second, carpaintFlakeRoughnessValue)) {
                errorMessage = "material flakeRoughness expects a float";
                return false;
            }
            carpaintFlakeRoughnessValue = std::clamp(carpaintFlakeRoughnessValue, 0.0f, 1.0f);
        } else {
            carpaintFlakeRoughnessValue = 0.15f;
        }
        if (auto it = tokens.find("flakeAnisotropy"); it != tokens.end()) {
            if (!parseFloat(it->second, carpaintFlakeAnisotropyValue)) {
                errorMessage = "material flakeAnisotropy expects a float";
                return false;
            }
            carpaintFlakeAnisotropyValue = std::clamp(carpaintFlakeAnisotropyValue, -0.99f, 0.99f);
        } else {
            carpaintFlakeAnisotropyValue = 0.3f;
        }
        if (auto it = tokens.find("flakeScale"); it != tokens.end()) {
            if (!parseFloat(it->second, carpaintFlakeScaleValue)) {
                errorMessage = "material flakeScale expects a float";
                return false;
            }
            carpaintFlakeScaleValue = std::max(carpaintFlakeScaleValue, 1.0e-4f);
        } else {
            carpaintFlakeScaleValue = 0.5f;
        }
        if (auto it = tokens.find("flakeNormalStrength"); it != tokens.end()) {
            if (!parseFloat(it->second, carpaintFlakeNormalStrengthValue)) {
                errorMessage = "material flakeNormalStrength expects a float";
                return false;
            }
            carpaintFlakeNormalStrengthValue = std::clamp(carpaintFlakeNormalStrengthValue, 0.0f, 1.0f);
        }
        if (auto it = tokens.find("flakeReflectanceScale"); it != tokens.end()) {
            if (!parseFloat(it->second, carpaintFlakeReflectanceScaleValue)) {
                errorMessage = "material flakeReflectanceScale expects a float";
                return false;
            }
            carpaintFlakeReflectanceScaleValue = std::clamp(carpaintFlakeReflectanceScaleValue, 0.0f, 1.0f);
        }
        if (auto it = tokens.find("baseTint"); it != tokens.end()) {
            simd::float3 tint{};
            if (!parseFloat3(it->second, tint)) {
                errorMessage = "material baseTint expects three floats";
                return false;
            }
            carpaintBaseTintValue = simd_make_float3(std::clamp(tint.x, 0.0f, 1.0f),
                                                     std::clamp(tint.y, 0.0f, 1.0f),
                                                     std::clamp(tint.z, 0.0f, 1.0f));
        }
        if (auto it = tokens.find("baseEta"); it != tokens.end()) {
            simd::float3 eta{};
            if (!parseFloat3(it->second, eta)) {
                errorMessage = "material baseEta expects three floats";
                return false;
            }
            carpaintBaseEtaValue = simd_make_float3(std::max(eta.x, 0.0f),
                                                    std::max(eta.y, 0.0f),
                                                    std::max(eta.z, 0.0f));
            carpaintBaseConductorExplicit = true;
        }
        if (auto it = tokens.find("baseK"); it != tokens.end()) {
            simd::float3 k{};
            if (!parseFloat3(it->second, k)) {
                errorMessage = "material baseK expects three floats";
                return false;
            }
            carpaintBaseKValue = simd_make_float3(std::max(k.x, 0.0f),
                                                  std::max(k.y, 0.0f),
                                                  std::max(k.z, 0.0f));
            carpaintBaseConductorExplicit = true;
        }
        roughness = carpaintBaseRoughnessValue;
        carpaintHasBaseConductorValue = carpaintBaseConductorExplicit || (carpaintBaseMetallicValue > 1.0e-4f);
    }

    float carpaintFlakeSampleWeightValue = 0.0f;
    if (isCarPaint) {
        carpaintFlakeSampleWeightValue = std::clamp(carpaintFlakeDensityValue * 1.0e-7f, 0.0f, 0.6f);
    }
    if (!isCarPaint) {
        carpaintBaseMetallicValue = 0.0f;
        carpaintBaseRoughnessValue = 0.0f;
        carpaintFlakeDensityValue = 0.0f;
        carpaintFlakeSampleWeightValue = 0.0f;
        carpaintFlakeRoughnessValue = 0.0f;
        carpaintFlakeAnisotropyValue = 0.0f;
        carpaintFlakeNormalStrengthValue = 0.0f;
        carpaintFlakeScaleValue = 1.0f;
        carpaintFlakeReflectanceScaleValue = 1.0f;
        carpaintBaseEtaValue = simd_make_float3(0.0f, 0.0f, 0.0f);
        carpaintBaseKValue = simd_make_float3(0.0f, 0.0f, 0.0f);
        carpaintHasBaseConductorValue = false;
        carpaintBaseTintValue = simd_make_float3(1.0f, 1.0f, 1.0f);
    }

    if (isPlastic || isSubsurface || isCarPaint) {
        if (auto it = tokens.find("coatRoughness"); it != tokens.end()) {
            if (!parseFloat(it->second, coatRoughnessValue)) {
                errorMessage = "material coatRoughness expects a float";
                return false;
            }
            coatRoughnessValue = std::clamp(coatRoughnessValue, 0.0f, 1.0f);
        }
        if (auto it = tokens.find("coatThickness"); it != tokens.end()) {
            if (!parseFloat(it->second, coatThicknessValue)) {
                errorMessage = "material coatThickness expects a float";
                return false;
            }
            coatThicknessValue = std::max(coatThicknessValue, 0.0f);
        }
        if (auto it = tokens.find("coatTint"); it != tokens.end()) {
            simd::float3 tint{};
            if (!parseFloat3(it->second, tint)) {
                errorMessage = "material coatTint expects three floats";
                return false;
            }
            coatTintValue = simd_make_float3(std::clamp(tint.x, 0.0f, 1.0f),
                                             std::clamp(tint.y, 0.0f, 1.0f),
                                             std::clamp(tint.z, 0.0f, 1.0f));
        }
        if (auto it = tokens.find("coatAbsorption"); it != tokens.end()) {
            simd::float3 absorption{};
            if (!parseFloat3(it->second, absorption)) {
                errorMessage = "material coatAbsorption expects three floats";
                return false;
            }
            coatAbsorptionValue = simd_make_float3(std::max(absorption.x, 0.0f),
                                                   std::max(absorption.y, 0.0f),
                                                   std::max(absorption.z, 0.0f));
        }
    }
    if (auto it = tokens.find("coatIOR"); it != tokens.end()) {
        if (!parseFloat(it->second, coatIorValue)) {
            errorMessage = "material coatIOR expects a float";
            return false;
        }
    }

    if (isPlastic && !iorExplicit) {
        ior = coatIorValue;
    }

    if (isCarPaint && !iorExplicit) {
        ior = 1.5f;
    }

    if (isSubsurface) {
        if (auto it = tokens.find("coat"); it != tokens.end()) {
            std::string value = it->second;
            std::string lower;
            lower.reserve(value.size());
            for (char ch : value) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
            }
            if (lower == "on" || lower == "true" || lower == "1") {
                sssCoatEnabledValue = true;
            } else if (lower == "off" || lower == "false" || lower == "0") {
                sssCoatEnabledValue = false;
            } else {
                errorMessage = "material coat expects on/off";
                return false;
            }
        }
    }

    simd::float3 conductorEta = simd_make_float3(0.0f, 0.0f, 0.0f);
    simd::float3 conductorK = simd_make_float3(0.0f, 0.0f, 0.0f);
    bool hasConductorParameters = false;
    if (type == PathTracerShaderTypes::MaterialType::Metal) {
        if (auto itEta = tokens.find("eta"); itEta != tokens.end()) {
            if (!parseFloat3(itEta->second, conductorEta)) {
                errorMessage = "material eta expects three floats";
                return false;
            }
            hasConductorParameters = true;
        }
        if (auto itK = tokens.find("k"); itK != tokens.end()) {
            if (!parseFloat3(itK->second, conductorK)) {
                errorMessage = "material k expects three floats";
                return false;
            }
            hasConductorParameters = true;
        }
    }

    float sssMeanFreePathValue = 0.0f;
    float sssAnisotropyValue = 0.0f;
    uint32_t sssMethodValue = 0u;
    simd::float3 sssSigmaAValue = simd_make_float3(0.0f, 0.0f, 0.0f);
    simd::float3 sssSigmaSValue = simd_make_float3(0.0f, 0.0f, 0.0f);
    bool sssSigmaAProvided = false;
    bool sssSigmaSProvided = false;
    bool sssSigmaOverrideValue = false;

    if (isSubsurface) {
        sssMeanFreePathValue = 1.0f;
        if (auto it = tokens.find("method"); it != tokens.end()) {
            std::string lower;
            lower.reserve(it->second.size());
            for (char ch : it->second) {
                lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
            }
            if (lower == "separable" || lower == "diffusion") {
                sssMethodValue = 0u;
            } else if (lower == "randomwalk" || lower == "random_walk") {
                sssMethodValue = 1u;
            } else {
                errorMessage = "material method for sss must be separable or randomwalk";
                return false;
            }
        }
        if (auto it = tokens.find("mfp"); it != tokens.end()) {
            if (!parseFloat(it->second, sssMeanFreePathValue)) {
                errorMessage = "material mfp expects a float";
                return false;
            }
        }
        if (auto it = tokens.find("g"); it != tokens.end()) {
            if (!parseFloat(it->second, sssAnisotropyValue)) {
                errorMessage = "material g expects a float";
                return false;
            }
            sssAnisotropyValue = std::clamp(sssAnisotropyValue, -0.99f, 0.99f);
        }
        if (auto it = tokens.find("sigma_a"); it != tokens.end()) {
            if (!parseFloat3(it->second, sssSigmaAValue)) {
                errorMessage = "material sigma_a expects three floats";
                return false;
            }
            sssSigmaAValue = simd_make_float3(std::max(sssSigmaAValue.x, 0.0f),
                                              std::max(sssSigmaAValue.y, 0.0f),
                                              std::max(sssSigmaAValue.z, 0.0f));
            sssSigmaAProvided = true;
        }
        if (auto it = tokens.find("sigma_s"); it != tokens.end()) {
            if (!parseFloat3(it->second, sssSigmaSValue)) {
                errorMessage = "material sigma_s expects three floats";
                return false;
            }
            sssSigmaSValue = simd_make_float3(std::max(sssSigmaSValue.x, 0.0f),
                                              std::max(sssSigmaSValue.y, 0.0f),
                                              std::max(sssSigmaSValue.z, 0.0f));
            sssSigmaSProvided = true;
        }
        if (sssSigmaAProvided != sssSigmaSProvided) {
            errorMessage = "material sigma_a and sigma_s must both be provided together";
            return false;
        }
        sssSigmaOverrideValue = sssSigmaAProvided && sssSigmaSProvided;
        sssMeanFreePathValue = std::max(sssMeanFreePathValue, 1.0e-4f);
    }

    simd::float3 dielectricSigmaAValue = simd_make_float3(0.0f, 0.0f, 0.0f);
    if (auto it = tokens.find("sigmaA"); it != tokens.end()) {
        if (!parseFloat3(it->second, dielectricSigmaAValue)) {
            errorMessage = "material sigmaA expects three floats";
            return false;
        }
        dielectricSigmaAValue = simd_make_float3(std::max(dielectricSigmaAValue.x, 0.0f),
                                                 std::max(dielectricSigmaAValue.y, 0.0f),
                                                 std::max(dielectricSigmaAValue.z, 0.0f));
    } else {
        auto absorbIt = tokens.find("absorption");
        auto thicknessIt = tokens.find("thickness");
        if (absorbIt != tokens.end() && thicknessIt != tokens.end()) {
            simd::float3 absorption{};
            float thicknessValue = 0.0f;
            if (!parseFloat3(absorbIt->second, absorption)) {
                errorMessage = "material absorption expects three floats";
                return false;
            }
            if (!parseFloat(thicknessIt->second, thicknessValue)) {
                errorMessage = "material thickness expects a float";
                return false;
            }
            float denom = std::max(thicknessValue, 1.0e-6f);
            dielectricSigmaAValue = simd_make_float3(std::max(absorption.x / denom, 0.0f),
                                                     std::max(absorption.y / denom, 0.0f),
                                                     std::max(absorption.z / denom, 0.0f));
        }
    }

    uint32_t materialIndex = resources.addMaterial(baseColor,
                          roughness,
                          type,
                          ior,
                          emission,
                          emissionUsesEnvironment,
                          conductorEta,
                          conductorK,
                          hasConductorParameters,
                          coatRoughnessValue,
                          coatThicknessValue,
                          coatTintValue,
                          coatAbsorptionValue,
                          coatIorValue,
                          dielectricSigmaAValue,
                          sssSigmaAValue,
                          sssSigmaSValue,
                          sssMeanFreePathValue,
                          sssAnisotropyValue,
                          sssMethodValue,
                          sssCoatEnabledValue,
                          sssSigmaOverrideValue,
                          carpaintBaseMetallicValue,
                          carpaintBaseRoughnessValue,
                          carpaintFlakeSampleWeightValue,
                          carpaintFlakeRoughnessValue,
                          carpaintFlakeAnisotropyValue,
                          carpaintFlakeNormalStrengthValue,
                          carpaintFlakeScaleValue,
                          carpaintFlakeReflectanceScaleValue,
                          carpaintBaseEtaValue,
                          carpaintBaseKValue,
                          carpaintHasBaseConductorValue,
                          carpaintBaseTintValue,
                          materialName);

    if (!materialName.empty()) {
        materialIndicesByName[materialName] = materialIndex;
    }
    return true;
}

bool SceneManager::parseSphere(const std::unordered_map<std::string, std::string>& tokens,
                               SceneResources& resources,
                               std::string& errorMessage) {
    auto centerIt = tokens.find("center");
    auto radiusIt = tokens.find("radius");
    auto materialIt = tokens.find("material");
    if (centerIt == tokens.end() || radiusIt == tokens.end() || materialIt == tokens.end()) {
        errorMessage = "sphere requires center, radius, and material tokens";
        return false;
    }

    simd::float3 center{};
    float radius = 0.0f;
    uint32_t materialIndex = 0;

    if (!parseFloat3(centerIt->second, center)) {
        errorMessage = "sphere center expects three floats";
        return false;
    }
    if (!parseFloat(radiusIt->second, radius)) {
        errorMessage = "sphere radius expects a float";
        return false;
    }
    if (!parseUInt(materialIt->second, materialIndex)) {
        errorMessage = "sphere material expects an integer index";
        return false;
    }
    if (materialIndex >= resources.materialCount()) {
        errorMessage = "sphere references material index that has not been defined yet";
        return false;
    }

    resources.addSphere(center, radius, materialIndex);
    return true;
}

bool SceneManager::parseBox(const std::unordered_map<std::string, std::string>& tokens,
                            SceneResources& resources,
                            std::string& errorMessage) {
    auto minIt = tokens.find("min");
    auto maxIt = tokens.find("max");
    auto materialIt = tokens.find("material");
    if (minIt == tokens.end() || maxIt == tokens.end() || materialIt == tokens.end()) {
        errorMessage = "box requires min, max, and material tokens";
        return false;
    }

    simd::float3 minCorner{};
    simd::float3 maxCorner{};
    if (!parseFloat3(minIt->second, minCorner)) {
        errorMessage = "box min expects three floats";
        return false;
    }
    if (!parseFloat3(maxIt->second, maxCorner)) {
        errorMessage = "box max expects three floats";
        return false;
    }

    uint32_t materialIndex = 0;
    if (!parseUInt(materialIt->second, materialIndex)) {
        errorMessage = "box material expects an integer index";
        return false;
    }
    if (materialIndex >= resources.materialCount()) {
        errorMessage = "box references material index that has not been defined yet";
        return false;
    }

    bool includeBottom = true;
    if (auto includeIt = tokens.find("includeBottom"); includeIt != tokens.end()) {
        uint32_t flag = 1;
        if (!parseUInt(includeIt->second, flag)) {
            errorMessage = "box includeBottom expects 0 or 1";
            return false;
        }
        includeBottom = (flag != 0);
    }

    bool twoSided = false;
    if (auto twoSidedIt = tokens.find("twoSided"); twoSidedIt != tokens.end()) {
        uint32_t flag = 0;
        if (!parseUInt(twoSidedIt->second, flag)) {
            errorMessage = "box twoSided expects 0 or 1";
            return false;
        }
        twoSided = (flag != 0);
    }

    simd::float3 translate = simd_make_float3(0.0f, 0.0f, 0.0f);
    bool hasTranslate = false;
    if (auto translateIt = tokens.find("translate"); translateIt != tokens.end()) {
        if (!parseFloat3(translateIt->second, translate)) {
            errorMessage = "box translate expects three floats";
            return false;
        }
        hasTranslate = true;
    }

    float rotateYDegrees = 0.0f;
    bool hasRotate = false;
    if (auto rotateIt = tokens.find("rotateY"); rotateIt != tokens.end()) {
        if (!parseFloat(rotateIt->second, rotateYDegrees)) {
            errorMessage = "box rotateY expects a float (degrees)";
            return false;
        }
        hasRotate = true;
    }

    if (!hasTranslate && !hasRotate) {
        resources.addBox(minCorner, maxCorner, materialIndex, includeBottom, twoSided);
        return true;
    }

    const float radians = rotateYDegrees * (kPi / 180.0f);
    float cosTheta = std::cosf(radians);
    float sinTheta = std::sinf(radians);

    simd::float4x4 rotation = matrix_identity_float4x4;
    rotation.columns[0] = simd_make_float4(cosTheta, 0.0f, -sinTheta, 0.0f);
    rotation.columns[1] = simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f);
    rotation.columns[2] = simd_make_float4(sinTheta, 0.0f, cosTheta, 0.0f);

    simd::float4x4 translation = matrix_identity_float4x4;
    translation.columns[3] = simd_make_float4(translate, 1.0f);

    simd::float4x4 transform = simd_mul(translation, rotation);
    resources.addBoxTransformed(minCorner, maxCorner, materialIndex, transform, includeBottom, twoSided);
    return true;
}

bool SceneManager::parseRectangle(const std::unordered_map<std::string, std::string>& tokens,
                                  SceneResources& resources,
                                  std::string& errorMessage) {
    auto materialIt = tokens.find("material");
    if (materialIt == tokens.end()) {
        errorMessage = "rectangle requires a material token";
        return false;
    }

    uint32_t materialIndex = 0;
    if (!parseUInt(materialIt->second, materialIndex)) {
        errorMessage = "rectangle material expects an integer index";
        return false;
    }
    if (materialIndex >= resources.materialCount()) {
        errorMessage = "rectangle references material index that has not been defined yet";
        return false;
    }

    const char* axisLabels[3] = {"x", "y", "z"};
    struct AxisRange {
        float min = 0.0f;
        float max = 0.0f;
        bool isFixed = false;
    } axes[3];

    for (uint32_t axis = 0; axis < 3; ++axis) {
        auto it = tokens.find(axisLabels[axis]);
        if (it == tokens.end()) {
            std::ostringstream oss;
            oss << "rectangle requires " << axisLabels[axis] << " token";
            errorMessage = oss.str();
            return false;
        }
        bool isFixed = false;
        if (!parseFloatRange(it->second, axes[axis].min, axes[axis].max, isFixed)) {
            std::ostringstream oss;
            oss << "rectangle " << axisLabels[axis]
                << " expects either a single value or a min,max range";
            errorMessage = oss.str();
            return false;
        }
        axes[axis].isFixed = isFixed;
    }

    uint32_t fixedAxisCount = 0;
    uint32_t normalAxis = 0;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        if (axes[axis].isFixed) {
            normalAxis = axis;
            ++fixedAxisCount;
        }
    }

    if (fixedAxisCount != 1) {
        errorMessage = "rectangle requires exactly one axis to be fixed to a single value";
        return false;
    }

    for (uint32_t axis = 0; axis < 3; ++axis) {
        if (axis == normalAxis) {
            continue;
        }
        if (axes[axis].isFixed) {
            std::ostringstream oss;
            oss << "rectangle " << axisLabels[axis]
                << " must provide a range (min,max) for in-plane axes";
            errorMessage = oss.str();
            return false;
        }
    }

    simd::float3 boundsMin = simd_make_float3(axes[0].min, axes[1].min, axes[2].min);
    simd::float3 boundsMax = simd_make_float3(axes[0].max, axes[1].max, axes[2].max);

    bool normalPositive = true;
    if (auto normalIt = tokens.find("normal"); normalIt != tokens.end()) {
        float normalValue = 1.0f;
        if (!parseFloat(normalIt->second, normalValue)) {
            errorMessage = "rectangle normal expects a float";
            return false;
        }
        normalPositive = normalValue >= 0.0f;
    }

    bool twoSided = false;
    if (auto twoSidedIt = tokens.find("twoSided"); twoSidedIt != tokens.end()) {
        uint32_t two = 0;
        if (!parseUInt(twoSidedIt->second, two)) {
            errorMessage = "rectangle twoSided expects 0 or 1";
            return false;
        }
        twoSided = (two != 0);
    }

    resources.addRectangle(boundsMin, boundsMax, normalAxis, normalPositive, twoSided, materialIndex);
    return true;
}

bool SceneManager::parseMesh(const std::unordered_map<std::string, std::string>& tokens,
                             SceneResources& resources,
                             std::string& errorMessage,
                             const std::string& sceneDirectory,
                             const std::unordered_map<std::string, uint32_t>& materialIndicesByName) {
    auto materialIt = tokens.find("material");
    if (materialIt == tokens.end()) {
        errorMessage = "mesh requires material token";
        return false;
    }

    std::string meshName;
    if (auto nameIt = tokens.find("name"); nameIt != tokens.end()) {
        meshName = nameIt->second;
    }

    std::string typeLower;
    if (auto typeIt = tokens.find("type"); typeIt != tokens.end()) {
        typeLower.reserve(typeIt->second.size());
        for (char ch : typeIt->second) {
            typeLower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }
    const bool isPlane = (typeLower == "plane" || typeLower == "quad");

    const auto pathIt = tokens.find("path");
    const auto fileIt = tokens.find("file");

    fs::path meshPath;
    if (!isPlane) {
        const std::string* pathValue = nullptr;
        if (pathIt != tokens.end()) {
            pathValue = &pathIt->second;
        } else if (fileIt != tokens.end()) {
            pathValue = &fileIt->second;
        }

        if (!pathValue) {
            errorMessage = "mesh requires path or file token";
            return false;
        }

        meshPath = fs::path(*pathValue);
        if (meshPath.is_relative()) {
            if (!sceneDirectory.empty()) {
                meshPath = fs::path(sceneDirectory) / meshPath;
            } else {
                std::error_code cwdEc;
                meshPath = fs::current_path(cwdEc) / meshPath;
            }
        }

        std::error_code canonicalEc;
        fs::path canonical = fs::weakly_canonical(meshPath, canonicalEc);
        if (canonicalEc || !fs::exists(canonical)) {
            std::ostringstream oss;
            oss << "mesh file not found: " << meshPath.string();
            errorMessage = oss.str();
            return false;
        }
        meshPath = canonical;
    }

    uint32_t materialIndex = 0;
    if (!parseUInt(materialIt->second, materialIndex)) {
        auto nameIt = materialIndicesByName.find(materialIt->second);
        if (nameIt == materialIndicesByName.end()) {
            errorMessage = "mesh material expects an index or known material name";
            return false;
        }
        materialIndex = nameIt->second;
    }
    if (materialIndex >= resources.materialCount()) {
        errorMessage = "mesh references material index that has not been defined yet";
        return false;
    }

    simd::float3 translation = simd_make_float3(0.0f, 0.0f, 0.0f);
    simd::float3 rotationDeg = simd_make_float3(0.0f, 0.0f, 0.0f);
    simd::float3 scale = simd_make_float3(1.0f, 1.0f, 1.0f);

    if (auto translateIt = tokens.find("translate"); translateIt != tokens.end()) {
        if (!parseFloat3(translateIt->second, translation)) {
            errorMessage = "mesh translate expects three floats";
            return false;
        }
    } else if (auto positionIt = tokens.find("position"); positionIt != tokens.end()) {
        if (!parseFloat3(positionIt->second, translation)) {
            errorMessage = "mesh position expects three floats";
            return false;
        }
    }

    if (auto rotateIt = tokens.find("rotate"); rotateIt != tokens.end()) {
        if (!parseFloat3(rotateIt->second, rotationDeg)) {
            errorMessage = "mesh rotate expects three floats (degrees)";
            return false;
        }
    }

    if (auto scaleIt = tokens.find("scale"); scaleIt != tokens.end()) {
        if (!parseFloat3(scaleIt->second, scale)) {
            errorMessage = "mesh scale expects three floats";
            return false;
        }
    }

    LoadedMeshData meshData;
    std::string loadError;
    if (isPlane) {
        meshData.vertices.resize(4);
        meshData.indices = {0, 1, 2, 0, 2, 3};

        meshData.vertices[0].position = simd_make_float3(-0.5f, 0.0f, -0.5f);
        meshData.vertices[1].position = simd_make_float3(0.5f, 0.0f, -0.5f);
        meshData.vertices[2].position = simd_make_float3(0.5f, 0.0f, 0.5f);
        meshData.vertices[3].position = simd_make_float3(-0.5f, 0.0f, 0.5f);

        for (auto& vertex : meshData.vertices) {
            vertex.normal = simd_make_float3(0.0f, 1.0f, 0.0f);
        }
        meshData.vertices[0].uv = simd_make_float2(0.0f, 0.0f);
        meshData.vertices[1].uv = simd_make_float2(1.0f, 0.0f);
        meshData.vertices[2].uv = simd_make_float2(1.0f, 1.0f);
        meshData.vertices[3].uv = simd_make_float2(0.0f, 1.0f);
    } else {
        std::string extension = meshPath.extension().string();
        std::string extensionLower;
        extensionLower.reserve(extension.size());
        for (char ch : extension) {
            extensionLower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }

        bool loaded = false;
        if (extensionLower == ".obj") {
            loaded = LoadObjMesh(meshPath, meshData, loadError);
        } else if (extensionLower == ".ply") {
            loaded = LoadPlyMesh(meshPath, meshData, loadError);
        } else {
            errorMessage = "mesh format not supported: " + extension;
            return false;
        }

        if (!loaded) {
            errorMessage = loadError;
            return false;
        }
    }

    if (meshData.indices.size() % 3 != 0) {
        errorMessage = "mesh loader produced a non-triangle index buffer";
        return false;
    }

    simd::float4x4 localToWorld = ComposeTransform(translation, rotationDeg, scale);
    uint32_t vertexCount = static_cast<uint32_t>(meshData.vertices.size());
    uint32_t indexCount = static_cast<uint32_t>(meshData.indices.size());

    if (vertexCount == 0 || indexCount == 0) {
        errorMessage = "mesh contains no renderable geometry";
        return false;
    }

    resources.addMesh(meshData.vertices.data(),
                      vertexCount,
                      meshData.indices.data(),
                      indexCount,
                      localToWorld,
                      materialIndex,
                      std::move(meshName));
    return true;
}

std::string SceneManager::readDisplayName(const std::string& filePath) {
    std::ifstream stream(filePath);
    if (!stream.is_open()) {
        return {};
    }

    std::string line;
    while (std::getline(stream, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty()) {
            continue;
        }
        if (trimmed.front() == '#') {
            trimmed.erase(trimmed.begin());
            return trim(trimmed);
        }
        break;
    }

    return {};
}

const SceneManager::SceneInfo* SceneManager::findScene(const std::string& identifier) const {
    auto it = m_sceneIndexById.find(identifier);
    if (it == m_sceneIndexById.end()) {
        return nullptr;
    }
    return &m_scenes[it->second];
}

}  // namespace PathTracer
