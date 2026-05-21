#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <simd/simd.h>

#include "renderer/MetalHandles.h"
#include "MetalShaderTypes.h"
#include "IntersectionProvider.h"

namespace PathTracer {

struct SceneAccelMeshInput {
    MTLBufferHandle vertexBuffer = nullptr;
    MTLBufferHandle indexBuffer = nullptr;
    uint32_t vertexStride = 0;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    simd::float4x4 localToWorldTransform = matrix_identity_float4x4;
    uint32_t materialIndex = 0;
};

struct SceneAccelBuildInput {
    MTLDeviceHandle device = nullptr;
    MTLCommandQueueHandle commandQueue = nullptr;
    const PathTracerShaderTypes::SphereData* spheres = nullptr;
    uint32_t sphereCount = 0;
    const SceneAccelMeshInput* meshes = nullptr;
    uint32_t meshCount = 0;
};

struct SceneAccelConfig {
    bool hardwareRaytracingSupported = false;
    MTLCommandQueueHandle commandQueue = nullptr;
};

enum class SceneAccelBuildPolicy : uint32_t {
    SoftwareReference = 0,
    HardwareFastBuild,
    HardwareFastTrace,
    HardwareCompactMemory,
    HardwareDynamicUpdate,
};

struct SceneAccelBuildStats {
    bool usedHardwareRaytracing = false;
    SceneAccelBuildPolicy buildPolicy = SceneAccelBuildPolicy::SoftwareReference;
    bool compactionSupported = false;
    bool deviceSupportsHardwareRaytracing = false;
    bool identityStable = false;
    bool customIntersectionFunctionsSupported = false;
    bool customIntersectionFunctionsEnabled = false;
    uint32_t primitiveCount = 0;
    uint32_t instanceCount = 0;
    uint32_t blasCount = 0;
    double blasBuildMs = 0.0;
    double tlasBuildMs = 0.0;
    double blasCompactionRatio = 0.0;
    double tlasCompactionRatio = 0.0;
    uint64_t estimatedUniqueGeometryBytes = 0;
    uint64_t blasMemoryBytes = 0;
    uint64_t tlasMemoryBytes = 0;
    uint64_t blasAllocatedMemoryBytes = 0;
    uint64_t tlasAllocatedMemoryBytes = 0;
    uint64_t scratchMemoryBytes = 0;  // Peak scratch required during AS build.
    uint64_t blasScratchMemoryBytes = 0;
    uint64_t tlasScratchMemoryBytes = 0;
    uint64_t retainedScratchMemoryBytes = 0;  // Scratch still resident after build.
    std::string buildPolicyLabel;
    std::string buildPolicyReason;
    std::string metalUsageFlags;
    std::string identityStrategy;
    std::string intersectionFunctionStrategy;
    std::string fallbackReason;
};

struct SceneAccelBuildProgress {
    uint32_t blasBatchesDone = 0;
    uint32_t blasBatchesTotal = 0;
    bool complete = false;
};

class SceneAccel {
public:
    virtual ~SceneAccel() = default;

    virtual void rebuild(const SceneAccelBuildInput& input,
                         IntersectionProvider& outProvider) = 0;
    virtual SceneAccelBuildStats estimateBuildStats(const SceneAccelBuildInput& input) = 0;
    virtual bool beginIncrementalRebuild(const SceneAccelBuildInput& input) = 0;
    virtual bool stepIncrementalRebuild(IntersectionProvider& outProvider,
                                        SceneAccelBuildProgress& outProgress) = 0;
    virtual bool incrementalRebuildActive() const = 0;
    virtual void clear() = 0;
    virtual PathTracerShaderTypes::IntersectionMode mode() const = 0;
    virtual uint32_t primitiveCount() const = 0;
    virtual const SceneAccelBuildStats& buildStats() const = 0;
};

std::unique_ptr<SceneAccel> CreateSceneAccel(const SceneAccelConfig& config);

}  // namespace PathTracer
