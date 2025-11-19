#pragma once

// ⚠️ INTERNAL HEADER - Subject to change without notice
// This header is part of PathTracer's internal implementation.
// Use only the public API from MetalRenderer.h

#include <array>
#include <vector>
#include <memory>
#include <string>
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
        simd::float2 uv{0.0f, 0.0f};
    };

    /// Add a triangle mesh to the scene
    /// @return Mesh index
    uint32_t addMesh(const MeshVertex* vertices,
                     uint32_t vertexCount,
                     const uint32_t* indices,
                     uint32_t indexCount,
                     const simd::float4x4& localToWorld,
                     uint32_t materialIndex,
                     std::string name = {});

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
    
    /// Clear all scene data
    void clear();
    
    /// Upload geometry to GPU buffers
    void uploadBuffers();
    
    /// Rebuild acceleration structures (BVH or hardware RT)
    void rebuildAccelerationStructures();
    
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

    struct Mesh {
        std::vector<MeshVertex> vertices;
        std::vector<uint32_t> indices;
        simd::float4x4 localToWorld = matrix_identity_float4x4;
        simd::float4x4 defaultLocalToWorld = matrix_identity_float4x4;
        uint32_t materialIndex = 0;
        std::string name;
        MTLBufferHandle vertexBuffer = nullptr;
        MTLBufferHandle indexBuffer = nullptr;
    };

    const std::vector<Mesh>& meshes() const { return m_meshes; }

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
    
    // Acceleration structures
    IntersectionProvider m_intersectionProvider{};
    std::unique_ptr<SceneAccel> m_sceneAccel;

    bool m_dirty = true;
    MTLTextureHandle m_environmentTexture = nullptr;
    bool m_forceSoftwareOverride = false;
    
    // Async rebuild tracking
    id<MTLCommandBuffer> m_rebuildCommandBuffer = nullptr;
    bool m_rebuildInProgress = false;
    
    // Helper methods
    void uploadMeshes();
    void uploadRectangles();
    void uploadMaterialToGpu(uint32_t index);
    void storeRectangleOriented(const simd::float3& corner,
                                const simd::float3& edgeU,
                                const simd::float3& edgeV,
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
};

}  // namespace PathTracer
