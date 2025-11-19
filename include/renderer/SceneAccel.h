#pragma once

#include <cstdint>
#include <memory>
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

class SceneAccel {
public:
    virtual ~SceneAccel() = default;

    virtual void rebuild(const SceneAccelBuildInput& input,
                         IntersectionProvider& outProvider) = 0;
    virtual void clear() = 0;
    virtual PathTracerShaderTypes::IntersectionMode mode() const = 0;
    virtual uint32_t primitiveCount() const = 0;
};

std::unique_ptr<SceneAccel> CreateSceneAccel(const SceneAccelConfig& config);

}  // namespace PathTracer
