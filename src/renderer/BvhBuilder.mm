#import "renderer/BvhBuilder.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <cstring>

namespace PathTracer {

namespace {

inline simd::float3 MinVec(const simd::float3& a, const simd::float3& b) {
    return simd::min(a, b);
}

inline simd::float3 MaxVec(const simd::float3& a, const simd::float3& b) {
    return simd::max(a, b);
}

}  // namespace

BvhBuilder::BvhBuilder(uint32_t leafSize) 
    : m_leafSize(leafSize) {
}

BvhBuildOutput BvhBuilder::build(const BvhBuildInput& input) {
    BvhBuildOutput result{};
    
    // Convert input primitives to build primitives (triangles preferred if provided)
    std::vector<BuildPrimitive> primitives;
    if (input.triangles && input.triangleCount > 0) {
        primitives.reserve(input.triangleCount);
        for (uint32_t i = 0; i < input.triangleCount; ++i) {
            const auto& tri = input.triangles[i];
            simd::float3 v0 = tri.v0.xyz;
            simd::float3 v1 = tri.v1.xyz;
            simd::float3 v2 = tri.v2.xyz;
            simd::float3 bmin = MinVec(v0, MinVec(v1, v2));
            simd::float3 bmax = MaxVec(v0, MaxVec(v1, v2));
            simd::float3 centroid = (v0 + v1 + v2) / 3.0f;
            BuildPrimitive prim{};
            prim.boundsMin = bmin;
            prim.boundsMax = bmax;
            prim.centroid = centroid;
            prim.primitiveIndex = i;
            primitives.push_back(prim);
        }
    } else if (input.spheres && input.sphereCount > 0) {
        primitives.reserve(input.sphereCount);
        for (uint32_t i = 0; i < input.sphereCount; ++i) {
            const auto& sphere = input.spheres[i];
            simd::float3 center = sphere.centerRadius.xyz;
            float radius = sphere.centerRadius.w;
            BuildPrimitive prim{};
            prim.boundsMin = center - simd::float3{radius, radius, radius};
            prim.boundsMax = center + simd::float3{radius, radius, radius};
            prim.centroid = center;
            prim.primitiveIndex = i;
            primitives.push_back(prim);
        }
    } else {
        return result;
    }

    // Initialize indices
    std::vector<uint32_t> indices(primitives.size());
    std::iota(indices.begin(), indices.end(), 0u);

    result.primitiveIndices.resize(primitives.size());
    result.nodes.reserve(primitives.size() * 2u);

    // Build BVH recursively
    buildRecursive(indices, primitives, result, 0u, static_cast<uint32_t>(primitives.size()));
    
    result.nodeCount = static_cast<uint32_t>(result.nodes.size());
    result.primitiveCount = static_cast<uint32_t>(result.primitiveIndices.size());
    
    return result;
}

uint32_t BvhBuilder::buildRecursive(std::vector<uint32_t>& indices,
                                    const std::vector<BuildPrimitive>& primitives,
                                    BvhBuildOutput& output,
                                    uint32_t start,
                                    uint32_t end) {
    const uint32_t count = end - start;

    simd::float3 boundsMin = {std::numeric_limits<float>::max(),
                              std::numeric_limits<float>::max(),
                              std::numeric_limits<float>::max()};
    simd::float3 boundsMax = {-std::numeric_limits<float>::max(),
                              -std::numeric_limits<float>::max(),
                              -std::numeric_limits<float>::max()};
    simd::float3 centroidMin = boundsMin;
    simd::float3 centroidMax = -boundsMax;

    for (uint32_t i = start; i < end; ++i) {
        const BuildPrimitive& prim = primitives[indices[i]];
        boundsMin = MinVec(boundsMin, prim.boundsMin);
        boundsMax = MaxVec(boundsMax, prim.boundsMax);
        centroidMin = MinVec(centroidMin, prim.centroid);
        centroidMax = MaxVec(centroidMax, prim.centroid);
    }

    PathTracerShaderTypes::BvhNode node{};
    node.boundsMin = simd_make_float4(boundsMin, 0.0f);
    node.boundsMax = simd_make_float4(boundsMax, 0.0f);

    const uint32_t nodeIndex = static_cast<uint32_t>(output.nodes.size());
    output.nodes.emplace_back();  // placeholder

    if (count <= m_leafSize) {
        // Create leaf node
        node.leftChild = std::numeric_limits<uint32_t>::max();
        node.rightChild = std::numeric_limits<uint32_t>::max();
        node.primitiveOffset = start;
        node.primitiveCount = count;
        for (uint32_t i = 0; i < count; ++i) {
            output.primitiveIndices[start + i] = primitives[indices[start + i]].primitiveIndex;
        }
        output.nodes[nodeIndex] = node;
        return nodeIndex;
    }

    // Choose split axis
    simd::float3 centroidExtent = centroidMax - centroidMin;
    uint32_t axis = 0;
    if (centroidExtent.y > centroidExtent.x && centroidExtent.y > centroidExtent.z) {
        axis = 1;
    } else if (centroidExtent.z > centroidExtent.x) {
        axis = 2;
    }

    if (centroidExtent[axis] < 1e-4f) {
        // Degenerate case - create leaf
        node.leftChild = std::numeric_limits<uint32_t>::max();
        node.rightChild = std::numeric_limits<uint32_t>::max();
        node.primitiveOffset = start;
        node.primitiveCount = count;
        for (uint32_t i = 0; i < count; ++i) {
            output.primitiveIndices[start + i] = primitives[indices[start + i]].primitiveIndex;
        }
        output.nodes[nodeIndex] = node;
        return nodeIndex;
    }

    // Partition primitives
    uint32_t mid = start + count / 2;
    std::nth_element(indices.begin() + start,
                     indices.begin() + mid,
                     indices.begin() + end,
                     [&](uint32_t lhs, uint32_t rhs) {
                         return primitives[lhs].centroid[axis] < primitives[rhs].centroid[axis];
                     });

    // Recursively build children
    uint32_t leftChild = buildRecursive(indices, primitives, output, start, mid);
    uint32_t rightChild = buildRecursive(indices, primitives, output, mid, end);

    node.leftChild = leftChild;
    node.rightChild = rightChild;
    node.primitiveOffset = 0;
    node.primitiveCount = 0;
    output.nodes[nodeIndex] = node;
    return nodeIndex;
}

bool BvhBuilder::createBuffers(MTLDeviceHandle device,
                               MTLCommandQueueHandle commandQueue,
                               const BvhBuildOutput& output,
                               MTLBufferHandle* outNodeBuffer,
                               MTLBufferHandle* outIndexBuffer) {
    if (!device || !outNodeBuffer || !outIndexBuffer) {
        return false;
    }

    *outNodeBuffer = nil;
    *outIndexBuffer = nil;

    if (output.nodes.empty() || output.primitiveIndices.empty()) {
        return false;
    }

    const NSUInteger nodeBytes = output.nodes.size() * sizeof(PathTracerShaderTypes::BvhNode);
    const NSUInteger indexBytes = output.primitiveIndices.size() * sizeof(uint32_t);

    auto createSharedBuffer = [&](const void* source, NSUInteger size, const char* label) -> MTLBufferHandle {
        if (size == 0) {
            return nil;
        }
        MTLBufferHandle buffer =
            [device newBufferWithLength:size options:MTLResourceStorageModeShared];
        if (!buffer) {
            NSLog(@"Failed to allocate shared %@ buffer", [NSString stringWithUTF8String:label]);
            return nil;
        }
        memcpy([buffer contents], source, size);
        return buffer;
    };

    auto tryCreatePrivateBuffers = [&]() -> bool {
        if (!commandQueue) {
            return false;
        }

        MTLBufferHandle nodeUpload = createSharedBuffer(output.nodes.data(), nodeBytes, "BVH node upload");
        if (!nodeUpload) {
            return false;
        }
        MTLBufferHandle indexUpload =
            createSharedBuffer(output.primitiveIndices.data(), indexBytes, "BVH index upload");
        if (!indexUpload) {
            return false;
        }

        MTLBufferHandle nodeDevice =
            [device newBufferWithLength:nodeBytes options:MTLResourceStorageModePrivate];
        if (!nodeDevice) {
            NSLog(@"Failed to allocate private BVH node buffer");
            return false;
        }
        MTLBufferHandle indexDevice =
            [device newBufferWithLength:indexBytes options:MTLResourceStorageModePrivate];
        if (!indexDevice) {
            NSLog(@"Failed to allocate private BVH index buffer");
            return false;
        }

        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        if (!commandBuffer) {
            NSLog(@"Failed to create command buffer for BVH upload");
            return false;
        }

        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        if (!blit) {
            NSLog(@"Failed to create blit encoder for BVH upload");
            [commandBuffer commit];
            // Don't wait here - let the caller handle synchronization if needed
            return false;
        }

        if (nodeBytes > 0) {
            [blit copyFromBuffer:nodeUpload
                     sourceOffset:0
                         toBuffer:nodeDevice
                destinationOffset:0
                             size:nodeBytes];
        }
        if (indexBytes > 0) {
            [blit copyFromBuffer:indexUpload
                     sourceOffset:0
                         toBuffer:indexDevice
                destinationOffset:0
                             size:indexBytes];
        }
        [blit endEncoding];

        [commandBuffer commit];
        // Don't wait here - Metal will automatically sequence commands correctly.
        // This avoids blocking the main thread during BVH uploads.

        *outNodeBuffer = nodeDevice;
        *outIndexBuffer = indexDevice;
        return true;
    };

    if (tryCreatePrivateBuffers()) {
        return true;
    }

    // Fallback to shared buffers when no command queue is available.
    MTLBufferHandle nodeBuffer = createSharedBuffer(output.nodes.data(), nodeBytes, "BVH node");
    if (!nodeBuffer) {
        return false;
    }
    MTLBufferHandle indexBuffer = createSharedBuffer(output.primitiveIndices.data(),
                                                     indexBytes,
                                                     "BVH index");
    if (!indexBuffer) {
        return false;
    }

    *outNodeBuffer = nodeBuffer;
    *outIndexBuffer = indexBuffer;
    return true;
}

}  // namespace PathTracer
