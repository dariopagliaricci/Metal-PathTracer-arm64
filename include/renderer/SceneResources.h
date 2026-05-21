#pragma once

// ⚠️ INTERNAL HEADER - Subject to change without notice
// This header is part of PathTracer's internal implementation.
// Use only the public API from MetalRenderer.h

#include <array>
#include <cstdint>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <simd/simd.h>

#include "renderer/MetalHandles.h"
#include "MetalShaderTypes.h"
#include "IntersectionProvider.h"
#include "renderer/SceneAccel.h"

namespace PathTracer {

struct EnvGpuHandles {
    MTLTextureHandle texture = nullptr;
    MTLBufferHandle conditionalAlias = nullptr;
    MTLBufferHandle marginalAlias = nullptr;
    MTLBufferHandle pdf = nullptr;
    uint32_t aliasCount = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    double thresholdHeadSum = 0.0;
    double thresholdTotalSum = 0.0;
};

struct SceneMemoryReport {
    uint32_t triangleCount = 0;
    uint32_t instanceCount = 0;
    uint32_t meshCount = 0;
    uint32_t textureCount = 0;
    uint64_t geometryMemoryBytes = 0;
    uint64_t textureMemoryBytes = 0;
    uint64_t textureAllocatedMemoryBytes = 0;
    uint64_t blasMemoryBytes = 0;
    uint64_t tlasMemoryBytes = 0;
    uint64_t scratchMemoryBytes = 0;  // Scratch still resident after build.
    uint64_t peakScratchMemoryBytes = 0;
    uint64_t totalEstimatedMemoryBytes = 0;
    uint64_t recommendedMaxWorkingSetSizeBytes = 0;
    double budgetUsagePercent = 0.0;
};

struct MaterialTextureSamplerDesc {
    int32_t magFilter = -1;  // glTF enum (9728/9729), -1 uses default
    int32_t minFilter = -1;  // glTF enum (9728/9729/9984..9987), -1 uses default
    int32_t wrapS = 10497;   // glTF enum (33071/33648/10497)
    int32_t wrapT = 10497;   // glTF enum (33071/33648/10497)
};

enum class MaterialTextureSemantic : uint32_t {
    Generic = 0,
    BaseColor = 1,
    Orm = 2,          // glTF metallic-roughness (G=roughness, B=metallic; AO may share)
    Normal = 3,
    Occlusion = 4,
    Emissive = 5,
    Transmission = 6,
};

enum class TextureBudgetPolicy : uint32_t {
    Strict = 0,
    Warn = 1,
    Ignore = 2,
};

enum class PilotMode : uint32_t {
    FullClamped = 0,
    GeoOnly = 1,
    AlbedoOnly = 2,
};

enum class TextureLoadStatus : uint32_t {
    Loaded = 0,
    SkippedByPolicy = 1,
    Failed = 2,
};

struct TextureLoadPolicy {
    TextureBudgetPolicy policy = TextureBudgetPolicy::Strict;
    PilotMode pilotMode = PilotMode::FullClamped;
    uint32_t maxDimension = 0;
    uint64_t maxTextureBytes = 0;
};

struct TextureInventoryEntry {
    std::string label;
    std::string formatLabel;
    MaterialTextureSemantic semantic = MaterialTextureSemantic::Generic;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t originalWidth = 0;
    uint32_t originalHeight = 0;
    uint32_t mipCount = 0;
    uint64_t estimatedBytes = 0;
    uint64_t allocatedBytes = 0;
    uint64_t originalEstimatedBytes = 0;
    uint64_t originalAllocatedBytes = 0;
    bool clampedByDimension = false;
    std::string decision;
};

struct MaterialTextureCpuData {
    struct MipLevel {
        uint32_t width = 0;
        uint32_t height = 0;
        std::vector<float> rgba;
    };

    uint32_t width = 0;
    uint32_t height = 0;
    bool srgb = false;
    std::vector<float> rgba;
    std::vector<MipLevel> mipLevels;
    MaterialTextureSamplerDesc samplerDesc{};
};

struct MeshInventoryEntry {
    std::string name;
    uint32_t triangleCount = 0;
    uint64_t estimatedBytes = 0;
};

struct AssetPipelineDiagnostic {
    std::string severity;
    std::string category;
    std::string message;
    std::string assetPath;
    std::string objectType;
    std::string objectName;
    int32_t objectIndex = -1;
};

struct AssetTextureColorSpace {
    int32_t textureIndex = -1;
    int32_t imageIndex = -1;
    std::string semantic;
    std::string colorSpace;
    std::string source;
    std::string status;
};

struct AssetPipelineStats {
    uint32_t materialCount = 0;
    uint32_t meshCount = 0;
    uint32_t meshInstanceCount = 0;
    uint32_t primitiveCount = 0;
    uint64_t vertexCount = 0;
    uint64_t indexCount = 0;
    uint64_t triangleCount = 0;
    uint64_t degenerateTriangleCount = 0;
    uint32_t imageCount = 0;
    uint32_t textureCount = 0;
    uint32_t loadedTextureCount = 0;
    uint32_t missingNormalPrimitiveCount = 0;
    uint32_t missingTangentPrimitiveCount = 0;
    uint32_t unsupportedExtensionCount = 0;
    uint32_t warningCount = 0;
    uint32_t errorCount = 0;
};

struct AssetPipelineManifest {
    std::string sourcePath;
    std::string sourceFormat;
    std::string importer;
    std::string importerVersion;
    std::string generator;
    std::string unitPolicy;
    std::string maturityLabel;
    float sceneUnitMeters = 1.0f;
};

struct AssetPipelineReport {
    AssetPipelineManifest manifest;
    AssetPipelineStats stats;
    std::vector<AssetTextureColorSpace> textureColorSpaces;
    std::vector<AssetPipelineDiagnostic> diagnostics;
};

class MetalContext;

/// Manages scene geometry, materials, and acceleration structures
/// Owns spheres, materials, and their GPU buffers
/// Coordinates BVH building and hardware raytracing setup
class SceneResources {
public:
    SceneResources();
    ~SceneResources();
    
    // Non-copyable
    SceneResources(const SceneResources&) = delete;
    SceneResources& operator=(const SceneResources&) = delete;
    
    /// Initialize with Metal context
    void initialize(const MetalContext& context);
    void initialize(MTLDeviceHandle device,
                    MTLCommandQueueHandle commandQueue,
                    bool supportsRaytracing);
    
    /// Add a material to the scene
    /// @return Material index for use with addSphere
    uint32_t addMaterial(const simd::float3& albedo,
                         float fuzz,
                         PathTracerShaderTypes::MaterialType type,
                         float indexOfRefraction,
                         const simd::float3& emission = simd_make_float3(0.0f, 0.0f, 0.0f),
                         bool emissionUsesEnvironment = false,
                         std::string name = {});

    /// Add a material with full parameter control (coat, SSS, car paint, etc.)
    uint32_t addMaterial(const simd::float3& baseColor,
                         float roughness,
                         PathTracerShaderTypes::MaterialType type,
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
                         float carpaintFlakeReflectanceScale = 1.0f,
                         simd::float3 carpaintBaseEta = simd_make_float3(0.0f, 0.0f, 0.0f),
                         simd::float3 carpaintBaseK = simd_make_float3(0.0f, 0.0f, 0.0f),
                         bool carpaintHasBaseConductor = false,
                         simd::float3 carpaintBaseTint = simd_make_float3(1.0f, 1.0f, 1.0f),
                         bool thinDielectric = false,
                         std::string name = {});

    /// Add a fully specified material data block (used for glTF/PBR materials).
    uint32_t addMaterialData(const PathTracerShaderTypes::MaterialData& material,
                             std::string name = {});
    
    /// Add a sphere to the scene
    void addSphere(const simd::float3& center, 
                   float radius, 
                   uint32_t materialIndex);

    /// Add an axis-aligned rectangle with outward-facing normal.
    /// @param boundsMin Minimum corner of the rectangle (before thickness expansion)
    /// @param boundsMax Maximum corner of the rectangle (before thickness expansion)
    /// @param normalAxis Axis index perpendicular to the rectangle (0 = X, 1 = Y, 2 = Z)
    /// @param normalPositive True if the outward normal points in the positive axis direction
    /// @param twoSided True if both sides should be considered front faces
    /// @param materialIndex Material index previously returned by addMaterial
    void addRectangle(const simd::float3& boundsMin,
                      const simd::float3& boundsMax,
                      uint32_t normalAxis,
                      bool normalPositive,
                      bool twoSided,
                      uint32_t materialIndex);
    void addDisk(const simd::float3& center,
                 uint32_t normalAxis,
                 bool normalPositive,
                 float radius,
                 bool twoSided,
                 uint32_t materialIndex);
    void addDirectionalLight(const simd::float3& directionToLight,
                             const simd::float3& radiance,
                             float selectionWeight = 1.0f);
    void addBox(const simd::float3& minCorner,
                const simd::float3& maxCorner,
                uint32_t materialIndex,
                bool includeBottomFace = true,
                bool twoSided = false);
    void addBoxTransformed(const simd::float3& minCorner,
                           const simd::float3& maxCorner,
                           uint32_t materialIndex,
                           const simd::float4x4& transform,
                           bool includeBottomFace = true,
                           bool twoSided = false);

    struct MeshVertex {
        simd::float3 position{0.0f, 0.0f, 0.0f};
        simd::float3 normal{0.0f, 1.0f, 0.0f};
        simd::float2 uv{0.0f, 0.0f};   // TEXCOORD_0
        simd::float2 uv1{0.0f, 0.0f};  // TEXCOORD_1 (falls back to uv when absent)
        simd::float4 tangent{1.0f, 0.0f, 0.0f, 1.0f};
    };

    /// Add a triangle mesh to the scene
    /// @return Mesh index
    uint32_t addMesh(const MeshVertex* vertices,
                     uint32_t vertexCount,
                     const uint32_t* indices,
                     uint32_t indexCount,
                     const simd::float4x4& localToWorld,
                     const simd::float4x4& sourceLocalTransform,
                     uint32_t materialIndex,
                     std::string name = {},
                     std::string geometryCacheKey = {});

    /// Material inspection helpers
    const std::string& materialName(uint32_t index) const;
    const PathTracerShaderTypes::MaterialData* materialsData() const { return m_materials.data(); }
    bool updateMaterial(uint32_t index, const PathTracerShaderTypes::MaterialData& material);
    bool resetMaterial(uint32_t index);

    /// Mesh transform helpers
    bool setMeshTransform(uint32_t meshIndex, const simd::float4x4& localToWorld);
    bool resetMeshTransform(uint32_t meshIndex);
    const simd::float4x4& meshTransform(uint32_t meshIndex) const;
    const std::string& meshName(uint32_t meshIndex) const;

    /// Environment map support
    bool setEnvironmentMap(const std::string& path);
    bool reloadEnvironmentIfNeeded(const std::string& path, EnvGpuHandles* outHandles = nullptr);
    void clearEnvironmentMap();
    const std::string& environmentPath() const { return m_environmentPath; }
    MTLTextureHandle environmentTexture() const { return m_environmentTexture; }
    MTLBufferHandle environmentConditionalAliasBuffer() const { return m_environmentConditionalAliasBuffer; }
    MTLBufferHandle environmentMarginalAliasBuffer() const { return m_environmentMarginalAliasBuffer; }
    MTLBufferHandle environmentPdfBuffer() const { return m_environmentPdfBuffer; }
    uint32_t environmentAliasCount() const { return m_environmentAliasCount; }
    uint32_t environmentMapWidth() const { return m_environmentWidth; }
    uint32_t environmentMapHeight() const { return m_environmentHeight; }
    bool hasEnvironmentDistribution() const {
        return m_environmentAliasCount > 0 &&
               m_environmentConditionalAliasBuffer &&
               m_environmentMarginalAliasBuffer &&
               m_environmentPdfBuffer;
    }

    /// Material texture support
    uint32_t addMaterialTextureFromFile(const std::string& path,
                                        bool srgb,
                                        std::string* errorMessage = nullptr,
                                        const MaterialTextureSamplerDesc* samplerDesc = nullptr,
                                        MaterialTextureSemantic semantic = MaterialTextureSemantic::Generic,
                                        TextureLoadStatus* outStatus = nullptr);
    uint32_t addMaterialTextureFromData(const uint8_t* data,
                                        size_t size,
                                        const std::string& label,
                                        bool srgb,
                                        std::string* errorMessage = nullptr,
                                        const MaterialTextureSamplerDesc* samplerDesc = nullptr,
                                        MaterialTextureSemantic semantic = MaterialTextureSemantic::Generic,
                                        TextureLoadStatus* outStatus = nullptr);
    uint32_t materialTextureCount() const {
        return static_cast<uint32_t>(m_materialTextures.size());
    }
    bool materialTextureCpuData(uint32_t index, MaterialTextureCpuData& out) const;
    const std::vector<MTLTextureHandle>& materialTextures() const {
        return m_materialTextures;
    }
    const std::vector<MTLSamplerStateHandle>& materialSamplers() const {
        return m_materialSamplers;
    }
    MTLBufferHandle materialTextureInfoBuffer() const { return m_materialTextureInfoBuffer; }
    bool canLoadMaterialTextures() const { return m_device != nullptr; }
    
    /// Clear all scene data
    void clear();
    void buildCPUPackedSceneData();

    /// Copy CPU-only scene content from a staging SceneResources and resolve any
    /// deferred texture requests on the current (GPU-capable) instance.
    bool adoptCPUScene(SceneResources& src, std::string* errorMessage = nullptr);
    
    /// Upload geometry to GPU buffers
    void uploadBuffers();
    bool beginIncrementalUpload();
    bool stepIncrementalUpload(bool* outComplete = nullptr);
    bool incrementalUploadActive() const { return m_incrementalUploadInProgress; }
    
    /// Rebuild acceleration structures (BVH or hardware RT)
    void rebuildAccelerationStructures();
    bool beginAccelerationStructureBuild(uint32_t* outBlasBatchesTotal = nullptr);
    bool stepAccelerationStructureBuild(uint32_t* outBlasBatchesDone = nullptr,
                                        uint32_t* outBlasBatchesTotal = nullptr,
                                        bool* outComplete = nullptr);
    bool accelerationStructureBuildInProgress() const { return m_accelBuildInProgress; }
    
    /// Check if scene needs rebuilding
    bool isDirty() const { return m_dirty; }
    
    /// Mark scene as clean (called after rebuild)
    void markClean() { m_dirty = false; }
    
    // Buffer accessors
    MTLBufferHandle sphereBuffer() const { return m_sphereBuffer; }
    MTLBufferHandle materialBuffer() const { return m_materialBuffer; }
    MTLBufferHandle rectangleBuffer() const { return m_rectangleBuffer; }
    MTLBufferHandle meshInfoBuffer() const { return m_meshInfoBuffer; }
    MTLBufferHandle meshVertexBuffer() const { return m_meshVertexBuffer; }
    MTLBufferHandle meshIndexBuffer() const { return m_meshIndexBuffer; }
    MTLBufferHandle emissivePrimitivesBuffer() const { return m_emissivePrimitivesBuffer; }
    
    // Acceleration structure accessor
    const IntersectionProvider& intersectionProvider() const { 
        return m_intersectionProvider; 
    }
    MTLBufferHandle triangleBuffer() const { return m_triangleBuffer; }
    
    // Scene stats
    uint32_t sphereCount() const { return m_sphereCount; }
    uint32_t rectangleCount() const { return m_rectangleCount; }
    uint32_t materialCount() const { return m_materialCount; }
    uint32_t triangleCount() const { return m_triangleCount; }
    uint32_t primitiveCount() const { return m_primitiveCount; }
    uint32_t emissivePrimitiveCount() const { return m_emissivePrimitiveCount; }
    float totalEmittedPower() const { return m_totalEmittedPower; }
    uint32_t skippedTexturedEmissivePrimitiveCount() const { return m_skippedTexturedEmissivePrimitiveCount; }
    uint32_t skippedZeroPowerEmissivePrimitiveCount() const { return m_skippedZeroPowerEmissivePrimitiveCount; }
    const std::vector<PathTracerShaderTypes::LightPrimitive>& emissivePrimitives() const { return m_emissivePrimitives; }
    const PathTracerShaderTypes::SphereData* spheresData() const { return m_spheres.data(); }
    const PathTracerShaderTypes::RectData* rectanglesData() const { return m_rectangles.data(); }

    struct Mesh {
        std::vector<MeshVertex> vertices;
        std::vector<uint32_t> indices;
        std::string geometryCacheKey;
        simd::float4x4 localToWorld = matrix_identity_float4x4;
        simd::float4x4 defaultLocalToWorld = matrix_identity_float4x4;
        simd::float4x4 sourceLocalTransform = matrix_identity_float4x4;
        uint32_t materialIndex = 0;
        std::string name;
        MTLBufferHandle vertexBuffer = nullptr;
        MTLBufferHandle indexBuffer = nullptr;
    };

    const std::vector<Mesh>& meshes() const { return m_meshes; }
    SceneMemoryReport sceneMemoryReport() const;
    SceneAccelBuildStats estimateAccelerationStructureBuildStats();
    uint64_t estimatedGeometryMemoryBytes() const;
    uint64_t estimatedTextureMemoryBytes() const;
    uint64_t allocatedTextureMemoryBytes() const;
    uint64_t recommendedMaxWorkingSetSizeBytes() const;
    const SceneAccelBuildStats& accelerationBuildStats() const { return m_lastBuildStats; }
    void setTextureLoadPolicy(const TextureLoadPolicy& policy);
    const TextureLoadPolicy& textureLoadPolicy() const { return m_textureLoadPolicy; }
    const std::vector<TextureInventoryEntry>& textureInventory() const { return m_textureInventory; }
    std::vector<MeshInventoryEntry> meshInventory() const;
    bool textureBudgetExceeded() const { return m_textureBudgetExceeded; }
    const std::string& textureBudgetMessage() const { return m_textureBudgetMessage; }
    void setAssetPipelineReport(AssetPipelineReport report);
    const AssetPipelineReport& assetPipelineReport() const { return m_assetPipelineReport; }

    void setForceSoftwareBvh(bool force);
    void setSoftwareRayTracingOverride(bool force);
    bool forceSoftwareBvh() const { return m_forceSoftwareOverride; }
    bool supportsRaytracing() const { return m_supportsRaytracing; }
    bool hardwareRaytracingEnabled() const {
        return m_supportsRaytracing && !m_forceSoftwareOverride;
    }
    
private:
    MTLDeviceHandle m_device = nullptr;
    MTLCommandQueueHandle m_commandQueue = nullptr;
    bool m_supportsRaytracing = false;
    
    // Scene data
    std::array<PathTracerShaderTypes::SphereData, 
               PathTracerShaderTypes::kMaxSpheres> m_spheres{};
    std::array<PathTracerShaderTypes::MaterialData,
               PathTracerShaderTypes::kMaxMaterials> m_materials{};
    std::array<PathTracerShaderTypes::MaterialData,
               PathTracerShaderTypes::kMaxMaterials> m_materialDefaults{};
    std::array<std::string,
               PathTracerShaderTypes::kMaxMaterials> m_materialNames{};
    
    uint32_t m_sphereCount = 0;
    uint32_t m_materialCount = 0;
    uint32_t m_triangleCount = 0;
    uint32_t m_primitiveCount = 0;
    std::vector<Mesh> m_meshes;
    
    // GPU buffers
    MTLBufferHandle m_sphereBuffer = nullptr;
    MTLBufferHandle m_materialBuffer = nullptr;
    MTLBufferHandle m_rectangleBuffer = nullptr;
    MTLBufferHandle m_meshInfoBuffer = nullptr;
    MTLBufferHandle m_triangleBuffer = nullptr;
    MTLBufferHandle m_meshVertexBuffer = nullptr;
    MTLBufferHandle m_meshIndexBuffer = nullptr;
    MTLBufferHandle m_emissivePrimitivesBuffer = nullptr;
    
    // Acceleration structures
    IntersectionProvider m_intersectionProvider{};
    std::unique_ptr<SceneAccel> m_sceneAccel;

    bool m_dirty = true;
    MTLTextureHandle m_environmentTexture = nullptr;
    bool m_forceSoftwareOverride = false;
    
    // Helper methods
    void uploadMeshes();
    void uploadRectangles();
    void buildPackedSceneCpuData();
    void rebuildEmissivePrimitiveInventory();
    void uploadEmissivePrimitiveBuffer();
    void uploadPackedSceneBuffers(bool includeTriangleData = true);
    bool replaceDeferredTexturePlaceholder(uint32_t placeholderIndex, uint32_t resolvedIndex);
    void uploadMaterialToGpu(uint32_t index);
    void storeRectangleOriented(const simd::float3& corner,
                                const simd::float3& edgeU,
                                const simd::float3& edgeV,
                                bool twoSided,
                                uint32_t materialIndex,
                                const simd::float3& desiredNormal);
    void storeDiskOriented(const simd::float3& center,
                           const simd::float3& radiusAxisU,
                           const simd::float3& radiusAxisV,
                           bool twoSided,
                           uint32_t materialIndex,
                           const simd::float3& desiredNormal);

    std::array<PathTracerShaderTypes::RectData,
               PathTracerShaderTypes::kMaxRectangles>
        m_rectangles{};
    uint32_t m_rectangleCount = 0;

    // Environment sampling data
    MTLBufferHandle m_environmentConditionalAliasBuffer = nullptr;
    MTLBufferHandle m_environmentMarginalAliasBuffer = nullptr;
    MTLBufferHandle m_environmentPdfBuffer = nullptr;
    uint32_t m_environmentAliasCount = 0;
    uint32_t m_environmentWidth = 0;
    uint32_t m_environmentHeight = 0;
    std::string m_environmentPath;

    void clearEnvironmentDistribution();
    bool buildEnvironmentDistribution(const float* rgba32,
                                      uint32_t width,
                                      uint32_t height,
                                      EnvGpuHandles& outHandles);

    uint32_t registerMaterialTexture(MTLTextureHandle texture,
                                     const std::string& key,
                                     const std::string& label,
                                     const MaterialTextureSamplerDesc* samplerDesc,
                                     MaterialTextureSemantic semantic,
                                     const TextureInventoryEntry* inventoryTemplate = nullptr);
    uint32_t materialSamplerIndexForDesc(const MaterialTextureSamplerDesc* samplerDesc);
    void rebuildMaterialTextureInfoBuffer();
    bool isTextureAllowedByPilotMode(MaterialTextureSemantic semantic, std::string& reason) const;
    bool shouldSkipTextureForBudget(MaterialTextureSemantic semantic,
                                    uint64_t estimatedBytes,
                                    std::string& reason);
    void resetTextureBudgetState();
    void populateSceneAccelBuildInput(SceneAccelBuildInput& buildInput,
                                      std::vector<SceneAccelMeshInput>& meshInputs) const;

    std::vector<MTLTextureHandle> m_materialTextures;
    std::vector<MTLSamplerStateHandle> m_materialSamplers;
    std::vector<uint32_t> m_materialTextureSamplerIndices;
    std::vector<MaterialTextureSamplerDesc> m_materialTextureCpuSamplers;
    MTLBufferHandle m_materialTextureInfoBuffer = nullptr;
    MTLBufferHandle m_textureUploadBuffer = nullptr;
    std::vector<std::string> m_materialTextureLabels;
    std::vector<TextureInventoryEntry> m_textureInventory;
    std::unordered_map<std::string, uint32_t> m_materialTextureIndex;
    std::unordered_map<uint64_t, uint32_t> m_materialSamplerIndices;
    TextureLoadPolicy m_textureLoadPolicy{};
    uint64_t m_textureBytesLoaded = 0;
    bool m_textureBudgetExceeded = false;
    std::string m_textureBudgetMessage;
    SceneAccelBuildStats m_lastBuildStats{};

    struct DeferredMaterialTexture {
        uint32_t placeholderIndex = 0xFFFFFFFFu;
        bool fromFile = true;
        std::string path;
        std::vector<uint8_t> data;
        std::string label;
        bool srgb = false;
        bool hasSamplerDesc = false;
        MaterialTextureSamplerDesc samplerDesc{};
        MaterialTextureSemantic semantic = MaterialTextureSemantic::Generic;
        uint32_t decodedWidth = 0;
        uint32_t decodedHeight = 0;
        std::vector<uint8_t> decodedRgba;
    };

    static constexpr uint32_t kDeferredTextureIndexBase = 0x80000000u;

    bool isDeferredTextureIndex(uint32_t index) const;
    uint32_t registerDeferredMaterialTexture(const std::string& key,
                                             const std::string& label,
                                             bool fromFile,
                                             const std::string& path,
                                             const uint8_t* data,
                                             size_t size,
                                             bool srgb,
                                             const MaterialTextureSamplerDesc* samplerDesc,
                                             MaterialTextureSemantic semantic);
    uint32_t resolveDeferredTextureIndex(const SceneResources& src,
                                         uint32_t index,
                                         std::unordered_map<uint32_t, uint32_t>& cache,
                                         std::string* errorMessage);
    std::vector<DeferredMaterialTexture> m_deferredMaterialTextures;
    std::vector<PathTracerShaderTypes::MeshInfo> m_packedMeshInfos;
    std::vector<PathTracerShaderTypes::TriangleData> m_packedTriangleData;
    std::vector<PathTracerShaderTypes::LightPrimitive> m_emissivePrimitives;
    std::vector<PathTracerShaderTypes::LightPrimitive> m_analyticLightPrimitives;
    std::vector<PathTracerShaderTypes::SceneVertex> m_packedSceneVertices;
    std::vector<simd::uint3> m_packedSceneIndices;
    uint32_t m_emissivePrimitiveCount = 0u;
    float m_totalEmittedPower = 0.0f;
    uint32_t m_skippedTexturedEmissivePrimitiveCount = 0u;
    uint32_t m_skippedZeroPowerEmissivePrimitiveCount = 0u;
    enum class UploadPhase : uint8_t {
        Idle = 0,
        ResolveTextures,
        ScalarsSpheres,
        ScalarsMaterials,
        ScalarsRectangles,
        Meshes,
        PackedMeshInfos,
        PackedTriangles,
        PackedVertices,
        PackedIndices,
        Complete,
    };
    UploadPhase m_incrementalUploadPhase = UploadPhase::Idle;
    bool m_incrementalUploadInProgress = false;
    size_t m_incrementalUploadDeferredTextureIndex = 0;
    size_t m_incrementalUploadMeshIndex = 0;
    NSUInteger m_incrementalUploadPackedOffset = 0u;
    bool m_stagedUploadMode = false;
    bool m_buffersUploadedViaStagedLoad = false;  // true after stepIncrementalUpload completes
    bool m_deferTriangleBufferForHardware = false;
    struct SharedMeshGpuBuffers {
        MTLBufferHandle vertexBuffer = nullptr;
        MTLBufferHandle indexBuffer = nullptr;
    };
    std::unordered_map<std::string, SharedMeshGpuBuffers> m_incrementalSharedMeshBuffers;
    std::unordered_map<uint32_t, uint32_t> m_resolvedDeferredTextureIndices;
    SceneAccelBuildInput m_stagedAccelBuildInput{};
    std::vector<SceneAccelMeshInput> m_stagedAccelMeshInputs;
    bool m_accelBuildInProgress = false;
    AssetPipelineReport m_assetPipelineReport;
};

}  // namespace PathTracer
