#pragma once

// ⚠️ INTERNAL HEADER - Subject to change without notice
// This header is part of PathTracer's internal implementation.
// Use only the public API from MetalRenderer.h

#include <vector>
#include <cstdint>
#include <simd/simd.h>

#include "renderer/MetalHandles.h"
#include "MetalShaderTypes.h"

namespace PathTracer {

/// Input data for BVH construction
struct BvhBuildInput {
    const PathTracerShaderTypes::SphereData* spheres = nullptr;
    uint32_t sphereCount = 0;
    // Optional triangle input; when provided and count > 0, a BVH will be built over triangles
    const PathTracerShaderTypes::TriangleData* triangles = nullptr;
    uint32_t triangleCount = 0;
};

/// Output from BVH construction
struct BvhBuildOutput {
    std::vector<PathTracerShaderTypes::BvhNode> nodes;
    std::vector<uint32_t> primitiveIndices;
    uint32_t nodeCount = 0;
    uint32_t primitiveCount = 0;
};

/// Builds linear BVH acceleration structures for ray tracing
/// Uses surface area heuristic (SAH) for splitting
class BvhBuilder {
public:
    explicit BvhBuilder(uint32_t leafSize = 4);
    ~BvhBuilder() = default;
    
    /// Build BVH from sphere data
    /// @param input Scene geometry to build BVH from
    /// @return BVH nodes and reordered primitive indices
    BvhBuildOutput build(const BvhBuildInput& input);
    
    /// Create Metal buffers from BVH build output
    /// @param device Metal device for buffer creation
    /// @param output BVH data to upload
    /// @param outNodeBuffer Receives BVH node buffer
    /// @param outIndexBuffer Receives primitive index buffer
    /// @return true if buffers created successfully
    static bool createBuffers(MTLDeviceHandle device,
                              MTLCommandQueueHandle commandQueue,
                              const BvhBuildOutput& output,
                              MTLBufferHandle* outNodeBuffer,
                              MTLBufferHandle* outIndexBuffer);
    
private:
    uint32_t m_leafSize;
    
    struct BuildPrimitive {
        simd::float3 boundsMin;
        simd::float3 boundsMax;
        simd::float3 centroid;
        uint32_t primitiveIndex;
    };
    
    uint32_t buildRecursive(std::vector<uint32_t>& indices,
                           const std::vector<BuildPrimitive>& primitives,
                           BvhBuildOutput& output,
                           uint32_t start,
                           uint32_t end);
};

}  // namespace PathTracer
