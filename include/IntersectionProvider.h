#pragma once

#include <vector>

#include "renderer/MetalHandles.h"
#include "MetalShaderTypes.h"

namespace PathTracer {

struct SoftwareBvhResources {
    // Sphere BVH (legacy path)
    MTLBufferHandle nodes = nullptr;
    MTLBufferHandle primitiveIndices = nullptr;
    uint32_t nodeCount = 0;
    uint32_t primitiveCount = 0;

    // Software TLAS over mesh instances
    MTLBufferHandle tlasNodes = nullptr;
    MTLBufferHandle tlasPrimitiveIndices = nullptr; // instance indices
    uint32_t tlasNodeCount = 0;
    uint32_t tlasPrimitiveCount = 0;

    // Packed BLAS over triangles for all instances
    MTLBufferHandle blasNodes = nullptr;
    MTLBufferHandle blasPrimitiveIndices = nullptr; // local triangle indices
    uint32_t blasNodeCount = 0;
    uint32_t blasPrimitiveCount = 0;

    // Per-instance offsets into packed BLAS and triangle ranges
    MTLBufferHandle instanceInfoBuffer = nullptr; // SoftwareInstanceInfo[]
    uint32_t instanceCount = 0;
};

struct HardwareRtResources {
#ifdef __OBJC__
    __unsafe_unretained MTLAccelerationStructureHandle blas = nullptr;  // Optional: first BLAS handle for debugging
    __unsafe_unretained MTLAccelerationStructureHandle tlas = nullptr;
    __unsafe_unretained MTLBufferHandle instanceBuffer = nullptr;
    __unsafe_unretained MTLBufferHandle instanceUserIDBuffer = nullptr;
    __unsafe_unretained MTLBufferHandle scratchBuffer = nullptr;
#else
    MTLAccelerationStructureHandle blas = nullptr;  // Optional: first BLAS handle for debugging
    MTLAccelerationStructureHandle tlas = nullptr;
    MTLBufferHandle instanceBuffer = nullptr;
    MTLBufferHandle instanceUserIDBuffer = nullptr;
    MTLBufferHandle scratchBuffer = nullptr;
#endif
    uint32_t blasCount = 0;
    uint32_t instanceCount = 0;
    // Added: all BLAS handles for scenes with multiple meshes (non-owning).
    std::vector<void*> blasHandles;
};

struct IntersectionProvider {
    PathTracerShaderTypes::IntersectionMode mode =
        PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
    SoftwareBvhResources software{};
    HardwareRtResources hardware{};
};

}  // namespace PathTracer
