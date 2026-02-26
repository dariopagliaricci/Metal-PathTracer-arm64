#import "renderer/SceneAccel.h"

#import <Metal/Metal.h>

#import "renderer/BvhBuilder.h"

#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <vector>

namespace PathTracer {

class SoftwareBvhAccel final : public SceneAccel {
public:
    SoftwareBvhAccel()
        : m_bvhBuilder(std::make_unique<BvhBuilder>(4)) {
    }

    void rebuild(const SceneAccelBuildInput& input,
                 IntersectionProvider& outProvider) override {
        m_primitiveCount = 0;
        outProvider.hardware = HardwareRtResources{};

        if (!input.device) {
            outProvider.mode = PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
            outProvider.software = SoftwareBvhResources{};
            return;
        }

        // If we have meshes, build per-mesh BLAS and a TLAS over instance AABBs
        if (input.meshes && input.meshCount > 0) {
            struct Aabb { simd::float3 min; simd::float3 max; };

            std::vector<PathTracerShaderTypes::BvhNode> packedBlasNodes;
            std::vector<uint32_t> packedBlasPrimIndices;
            std::vector<PathTracerShaderTypes::SoftwareInstanceInfo> instanceInfos;
            std::vector<Aabb> instanceAabbs;
            instanceInfos.reserve(input.meshCount);
            instanceAabbs.reserve(input.meshCount);

            uint32_t triangleBaseOffset = 0;
            for (uint32_t i = 0; i < input.meshCount; ++i) {
                const SceneAccelMeshInput& mesh = input.meshes[i];
                const uint32_t triCount = mesh.indexCount / 3u;
                if (!mesh.vertexBuffer || !mesh.indexBuffer || triCount == 0) {
                    instanceInfos.push_back({});
                    instanceAabbs.push_back({{0,0,0}, {0,0,0}});
                    continue;
                }

                const uint8_t* vbase = reinterpret_cast<const uint8_t*>([mesh.vertexBuffer contents]);
                const uint32_t* indices = reinterpret_cast<const uint32_t*>([mesh.indexBuffer contents]);
                if (!vbase || !indices) {
                    instanceInfos.push_back({});
                    instanceAabbs.push_back({{0,0,0}, {0,0,0}});
                    triangleBaseOffset += triCount;
                    continue;
                }

                // Build per-mesh triangles in OBJECT SPACE for BLAS
                // tinyBVH expects bvhvec4 format: 3 vertices per triangle, each as (x,y,z,w)
                std::vector<tinybvh::bvhvec4> blasVerts;
                blasVerts.reserve(triCount * 3);
                simd::float3 localMin{std::numeric_limits<float>::max(),
                                      std::numeric_limits<float>::max(),
                                      std::numeric_limits<float>::max()};
                simd::float3 localMax{-std::numeric_limits<float>::max(),
                                      -std::numeric_limits<float>::max(),
                                      -std::numeric_limits<float>::max()};

                for (uint32_t t = 0; t < triCount; ++t) {
                    uint32_t i0 = indices[t * 3u + 0u];
                    uint32_t i1 = indices[t * 3u + 1u];
                    uint32_t i2 = indices[t * 3u + 2u];
                    if (i0 >= mesh.vertexCount || i1 >= mesh.vertexCount || i2 >= mesh.vertexCount) {
                        // Invalid triangle - add degenerate
                        blasVerts.push_back(tinybvh::bvhvec4(0,0,0,0));
                        blasVerts.push_back(tinybvh::bvhvec4(0,0,0,0));
                        blasVerts.push_back(tinybvh::bvhvec4(0,0,0,0));
                        continue;
                    }
                    const float* p0 = reinterpret_cast<const float*>(vbase + static_cast<size_t>(i0) * mesh.vertexStride);
                    const float* p1 = reinterpret_cast<const float*>(vbase + static_cast<size_t>(i1) * mesh.vertexStride);
                    const float* p2 = reinterpret_cast<const float*>(vbase + static_cast<size_t>(i2) * mesh.vertexStride);

                    // Keep in local/object space
                    simd::float3 v0{p0[0], p0[1], p0[2]};
                    simd::float3 v1{p1[0], p1[1], p1[2]};
                    simd::float3 v2{p2[0], p2[1], p2[2]};

                    blasVerts.push_back(tinybvh::bvhvec4(v0.x, v0.y, v0.z, 0.0f));
                    blasVerts.push_back(tinybvh::bvhvec4(v1.x, v1.y, v1.z, 0.0f));
                    blasVerts.push_back(tinybvh::bvhvec4(v2.x, v2.y, v2.z, 0.0f));

                    // Track local-space AABB
                    localMin = simd::min(localMin, simd::min(v0, simd::min(v1, v2)));
                    localMax = simd::max(localMax, simd::max(v0, simd::max(v1, v2)));
                }

                // Build BLAS using tinyBVH
                tinybvh::BVH blas;
                blas.Build(blasVerts.data(), triCount);

                // Convert tinyBVH nodes to our format and pack them
                uint32_t nodeBase = static_cast<uint32_t>(packedBlasNodes.size());
                uint32_t primBase = static_cast<uint32_t>(packedBlasPrimIndices.size());

                // tinyBVH uses node index 0 and 1 is typically reserved, start from node 2
                uint32_t usedNodeCount = blas.usedNodes;
                const uint32_t invalidIndex = std::numeric_limits<uint32_t>::max();
                for (uint32_t n = 0; n < usedNodeCount; ++n) {
                    const tinybvh::BVH::BVHNode& srcNode = blas.bvhNode[n];
                    PathTracerShaderTypes::BvhNode dstNode{};
                    dstNode.boundsMin = simd_make_float4(srcNode.aabbMin.x, srcNode.aabbMin.y, srcNode.aabbMin.z, 0.0f);
                    dstNode.boundsMax = simd_make_float4(srcNode.aabbMax.x, srcNode.aabbMax.y, srcNode.aabbMax.z, 0.0f);
                    if (srcNode.triCount > 0) {
                        // Leaf nodes refer to triangle ranges; no child nodes.
                        dstNode.leftChild = invalidIndex;
                        dstNode.rightChild = invalidIndex;
                        dstNode.primitiveOffset = srcNode.leftFirst;
                        dstNode.primitiveCount = srcNode.triCount;
                    } else {
                        // Interior nodes reference children relative to this mesh's node segment.
                        uint32_t leftChildIndex = nodeBase + srcNode.leftFirst;
                        uint32_t rightChildIndex = nodeBase + srcNode.leftFirst + 1;
                        if (srcNode.leftFirst >= usedNodeCount) {
                            leftChildIndex = invalidIndex;
                        }
                        if ((srcNode.leftFirst + 1u) >= usedNodeCount) {
                            rightChildIndex = invalidIndex;
                        }
                        dstNode.leftChild = leftChildIndex;
                        dstNode.rightChild = rightChildIndex;
                        dstNode.primitiveOffset = 0;
                        dstNode.primitiveCount = 0;
                    }
                    packedBlasNodes.push_back(dstNode);
                }

                // Pack primitive indices from tinyBVH
                for (uint32_t p = 0; p < triCount; ++p) {
                    packedBlasPrimIndices.push_back(blas.primIdx[p]);
                }

                // Transform local AABB to world space for TLAS
                simd::float4 corners[8] = {
                    simd_make_float4(localMin.x, localMin.y, localMin.z, 1.0f),
                    simd_make_float4(localMax.x, localMin.y, localMin.z, 1.0f),
                    simd_make_float4(localMin.x, localMax.y, localMin.z, 1.0f),
                    simd_make_float4(localMax.x, localMax.y, localMin.z, 1.0f),
                    simd_make_float4(localMin.x, localMin.y, localMax.z, 1.0f),
                    simd_make_float4(localMax.x, localMin.y, localMax.z, 1.0f),
                    simd_make_float4(localMin.x, localMax.y, localMax.z, 1.0f),
                    simd_make_float4(localMax.x, localMax.y, localMax.z, 1.0f)
                };
                simd::float3 worldMin{std::numeric_limits<float>::max(),
                                      std::numeric_limits<float>::max(),
                                      std::numeric_limits<float>::max()};
                simd::float3 worldMax{-std::numeric_limits<float>::max(),
                                      -std::numeric_limits<float>::max(),
                                      -std::numeric_limits<float>::max()};
                for (int c = 0; c < 8; ++c) {
                    simd::float4 worldCorner = mesh.localToWorldTransform * corners[c];
                    simd::float3 wc = simd_make_float3(worldCorner);
                    worldMin = simd::min(worldMin, wc);
                    worldMax = simd::max(worldMax, wc);
                }

                PathTracerShaderTypes::SoftwareInstanceInfo info{};
                info.blasRootNodeOffset = nodeBase;
                info.blasNodeCount = usedNodeCount;
                info.blasPrimIndexOffset = primBase;
                info.blasPrimIndexCount = triCount;
                info.triangleBaseOffset = triangleBaseOffset;
                info.triangleCount = triCount;
                info.localToWorld = mesh.localToWorldTransform;
                info.worldToLocal = simd_inverse(mesh.localToWorldTransform);
                instanceInfos.push_back(info);
                instanceAabbs.push_back({worldMin, worldMax});

                triangleBaseOffset += triCount;
            }

            // Build TLAS over instance AABBs
            auto buildTLAS = [&](const std::vector<Aabb>& aabbs,
                                 std::vector<PathTracerShaderTypes::BvhNode>& outNodes,
                                 std::vector<uint32_t>& outPrimIdx) {
                struct Prim { simd::float3 bmin, bmax, centroid; uint32_t idx; };
                std::vector<Prim> prims;
                prims.reserve(aabbs.size());
                for (uint32_t i = 0; i < aabbs.size(); ++i) {
                    const auto& b = aabbs[i];
                    Prim p{b.min, b.max, (b.min + b.max) * 0.5f, i};
                    prims.push_back(p);
                }
                outNodes.clear(); outPrimIdx.resize(prims.size());

                std::function<uint32_t(uint32_t,uint32_t)> buildRec = [&](uint32_t start, uint32_t end) -> uint32_t {
                    PathTracerShaderTypes::BvhNode node{};
                    simd::float3 bmin{ std::numeric_limits<float>::max() };
                    simd::float3 bmax{ -std::numeric_limits<float>::max() };
                    simd::float3 cmin = bmin, cmax = -bmax;
                    for (uint32_t i = start; i < end; ++i) {
                        const Prim& pr = prims[i];
                        bmin = simd::min(bmin, pr.bmin);
                        bmax = simd::max(bmax, pr.bmax);
                        cmin = simd::min(cmin, pr.centroid);
                        cmax = simd::max(cmax, pr.centroid);
                    }
                    node.boundsMin = simd_make_float4(bmin, 0.0f);
                    node.boundsMax = simd_make_float4(bmax, 0.0f);
                    const uint32_t nodeIdx = static_cast<uint32_t>(outNodes.size());
                    outNodes.emplace_back(); // placeholder

                    const uint32_t count = end - start;
                    if (count <= 4) {
                        node.leftChild = std::numeric_limits<uint32_t>::max();
                        node.rightChild = std::numeric_limits<uint32_t>::max();
                        node.primitiveOffset = start;
                        node.primitiveCount = count;
                        for (uint32_t i = 0; i < count; ++i) {
                            outPrimIdx[start + i] = prims[start + i].idx;
                        }
                        outNodes[nodeIdx] = node;
                        return nodeIdx;
                    }

                    simd::float3 ext = cmax - cmin;
                    uint32_t axis = 0;
                    if (ext.y > ext.x && ext.y > ext.z) axis = 1; else if (ext.z > ext.x) axis = 2;
                    uint32_t mid = (start + end) / 2;
                    std::nth_element(prims.begin() + start, prims.begin() + mid, prims.begin() + end,
                                     [&](const Prim& a, const Prim& b){ return a.centroid[axis] < b.centroid[axis]; });
                    uint32_t L = buildRec(start, mid);
                    uint32_t R = buildRec(mid, end);
                    node.leftChild = L; node.rightChild = R;
                    node.primitiveOffset = 0; node.primitiveCount = 0;
                    outNodes[nodeIdx] = node;
                    return nodeIdx;
                };

                if (!prims.empty()) buildRec(0u, static_cast<uint32_t>(prims.size()));
            };

            std::vector<PathTracerShaderTypes::BvhNode> tlasNodes;
            std::vector<uint32_t> tlasPrimIdx;
            buildTLAS(instanceAabbs, tlasNodes, tlasPrimIdx);

            // Upload buffers
            MTLBufferHandle tlasNodeBuf = nil, tlasPrimBuf = nil;
            MTLBufferHandle blasNodeBuf = nil, blasPrimBuf = nil;
            MTLBufferHandle instInfoBuf = nil;

            auto createShared = [&](const void* data, NSUInteger bytes) -> id<MTLBuffer> {
                if (bytes == 0) return nil;
                id<MTLBuffer> b = [input.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                if (b && data) { memcpy([b contents], data, bytes); }
                return b;
            };

            tlasNodeBuf = createShared(tlasNodes.data(), tlasNodes.size() * sizeof(PathTracerShaderTypes::BvhNode));
            tlasPrimBuf = createShared(tlasPrimIdx.data(), tlasPrimIdx.size() * sizeof(uint32_t));
            blasNodeBuf = createShared(packedBlasNodes.data(), packedBlasNodes.size() * sizeof(PathTracerShaderTypes::BvhNode));
            blasPrimBuf = createShared(packedBlasPrimIndices.data(), packedBlasPrimIndices.size() * sizeof(uint32_t));
            instInfoBuf = createShared(instanceInfos.data(), instanceInfos.size() * sizeof(PathTracerShaderTypes::SoftwareInstanceInfo));

            outProvider.mode = PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
            outProvider.software.tlasNodes = tlasNodeBuf;
            outProvider.software.tlasPrimitiveIndices = tlasPrimBuf;
            outProvider.software.tlasNodeCount = static_cast<uint32_t>(tlasNodes.size());
            outProvider.software.tlasPrimitiveCount = static_cast<uint32_t>(tlasPrimIdx.size());
            outProvider.software.blasNodes = blasNodeBuf;
            outProvider.software.blasPrimitiveIndices = blasPrimBuf;
            outProvider.software.blasNodeCount = static_cast<uint32_t>(packedBlasNodes.size());
            outProvider.software.blasPrimitiveCount = static_cast<uint32_t>(packedBlasPrimIndices.size());
            outProvider.software.instanceInfoBuffer = instInfoBuf;
            outProvider.software.instanceCount = static_cast<uint32_t>(instanceInfos.size());

            // Also clear the legacy sphere BVH fields for triangle-only scenes
            outProvider.software.nodes = nil;
            outProvider.software.primitiveIndices = nil;
            outProvider.software.nodeCount = 0;
            outProvider.software.primitiveCount = 0;

            m_primitiveCount = static_cast<uint32_t>(packedBlasPrimIndices.size());
            return;
        }

        // Fallback to sphere BVH when no meshes are present
        if (!input.spheres || input.sphereCount == 0) {
            outProvider.mode = PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
            outProvider.software = SoftwareBvhResources{};
            return;
        }

        BvhBuildInput builderInput{};
        builderInput.spheres = input.spheres;
        builderInput.sphereCount = input.sphereCount;
        BvhBuildOutput builderOutput = m_bvhBuilder->build(builderInput);

        MTLBufferHandle nodeBuffer = nil;
        MTLBufferHandle indexBuffer = nil;
        if (!BvhBuilder::createBuffers(input.device,
                                       input.commandQueue,
                                       builderOutput,
                                       &nodeBuffer,
                                       &indexBuffer)) {
            NSLog(@"Failed to create BVH buffers");
            outProvider.mode = PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
            outProvider.software = SoftwareBvhResources{};
            return;
        }

        outProvider.mode = PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
        outProvider.software.nodes = nodeBuffer;
        outProvider.software.primitiveIndices = indexBuffer;
        outProvider.software.nodeCount = builderOutput.nodeCount;
        outProvider.software.primitiveCount = builderOutput.primitiveCount;

        m_primitiveCount = builderOutput.primitiveCount;
    }

    void clear() override {
        m_primitiveCount = 0;
    }

    PathTracerShaderTypes::IntersectionMode mode() const override {
        return PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
    }

    uint32_t primitiveCount() const override {
        return m_primitiveCount;
    }

private:
    std::unique_ptr<BvhBuilder> m_bvhBuilder;
    uint32_t m_primitiveCount = 0;
};

namespace {

static MTLPackedFloat4x3 ToPackedTransform(const simd::float4x4& matrix) {
    MTLPackedFloat4x3 packed{};
    for (int column = 0; column < 4; ++column) {
        const simd::float4 col = matrix.columns[column];
        packed.columns[column] = MTLPackedFloat3Make(col.x, col.y, col.z);
    }
    return packed;
}

}  // namespace

class MetalRtAccel final : public SceneAccel {
public:
    explicit MetalRtAccel(const SceneAccelConfig& config)
        : m_commandQueue(config.commandQueue),
          m_canUseHardware(config.hardwareRaytracingSupported && config.commandQueue),
          m_softwareFallback(std::make_unique<SoftwareBvhAccel>()) {
    }

    void rebuild(const SceneAccelBuildInput& input,
                 IntersectionProvider& outProvider) override {
        m_primitiveCount = 0;

        if (!m_canUseHardware || !input.device || !input.meshes || input.meshCount == 0) {
            fallbackToSoftware(input, outProvider);
            return;
        }

        if (!input.commandQueue && !m_commandQueue) {
            fallbackToSoftware(input, outProvider);
            return;
        }

        if (!buildHardware(input, outProvider)) {
            fallbackToSoftware(input, outProvider);
            return;
        }

        m_mode = PathTracerShaderTypes::IntersectionMode::HardwareRayTracing;
    }

    void clear() override {
        releaseHardwareResources();
        if (m_softwareFallback) {
            m_softwareFallback->clear();
        }
        m_mode = PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
        m_primitiveCount = 0;
    }

    PathTracerShaderTypes::IntersectionMode mode() const override {
        return m_mode;
    }

    uint32_t primitiveCount() const override {
        return m_primitiveCount;
    }

private:
    bool buildHardware(const SceneAccelBuildInput& input,
                       IntersectionProvider& outProvider) {
        releaseHardwareResources();

        id<MTLCommandQueue> queue = input.commandQueue ? input.commandQueue : m_commandQueue;
        if (!queue) {
            return false;
        }

        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        if (!commandBuffer) {
            return false;
        }

        id<MTLAccelerationStructureCommandEncoder> encoder = [commandBuffer accelerationStructureCommandEncoder];
        if (!encoder) {
            [commandBuffer commit];
            return false;
        }

        const uint32_t meshCount = input.meshCount;
        if (meshCount == 0) {
            [encoder endEncoding];
            [commandBuffer commit];
            return false;
        }

        auto fail = [&](const char* message) -> bool {
            if (message) {
                NSLog(@"MetalRtAccel::buildHardware - %s", message);
            }
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            releaseHardwareResources();
            return false;
        };

        std::vector<MTLPrimitiveAccelerationStructureDescriptor*> primitiveDescs;
        primitiveDescs.reserve(meshCount);
        std::vector<MTLAccelerationStructureSizes> blasSizes;
        blasSizes.reserve(meshCount);

        NSUInteger maxScratch = 0;
        uint32_t totalPrimitiveCount = 0;
        for (uint32_t i = 0; i < meshCount; ++i) {
            const SceneAccelMeshInput& mesh = input.meshes[i];

            if (!mesh.vertexBuffer || !mesh.indexBuffer || mesh.vertexCount == 0 || mesh.indexCount < 3) {
                return fail("Invalid mesh data for BLAS build");
            }

            const uint32_t triangleCount = mesh.indexCount / 3;
            if (triangleCount == 0) {
                return fail("Mesh contains no triangles");
            }

            totalPrimitiveCount += triangleCount;

            auto tri = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
            tri.vertexBuffer = mesh.vertexBuffer;
            tri.vertexStride = mesh.vertexStride;
            tri.vertexBufferOffset = 0;
            tri.vertexFormat = MTLAttributeFormatFloat3;
            tri.indexBuffer = mesh.indexBuffer;
            tri.indexBufferOffset = 0;
            tri.indexType = MTLIndexTypeUInt32;
            tri.triangleCount = triangleCount;
            tri.opaque = YES;

            auto primDesc = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
            primDesc.geometryDescriptors = @[ tri ];
            primDesc.usage = MTLAccelerationStructureUsageNone;
            primitiveDescs.push_back(primDesc);

            MTLAccelerationStructureSizes sizes = [input.device accelerationStructureSizesWithDescriptor:primDesc];
            maxScratch = std::max(maxScratch, sizes.buildScratchBufferSize);
            blasSizes.push_back(sizes);
        }

        if (totalPrimitiveCount == 0) {
            return fail("No triangles found while building BLAS (totalPrimitiveCount == 0)");
        }

        NSLog(@"MetalRtAccel::buildHardware - starting BLAS build for %u triangles across %u mesh instances",
              totalPrimitiveCount,
              meshCount);

        if (primitiveDescs.empty()) {
            return fail("No primitive descriptors created");
        }

        ensureScratchBuffer(input.device, maxScratch);

        NSMutableArray<id<MTLAccelerationStructure>>* blasHandles =
            [NSMutableArray arrayWithCapacity:meshCount];

        for (NSUInteger i = 0; i < primitiveDescs.size(); ++i) {
            MTLAccelerationStructureSizes sizes = blasSizes[i];
            id<MTLAccelerationStructure> blas =
                [input.device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
            if (!blas) {
                return fail("Failed to allocate BLAS acceleration structure");
            }

            [blasHandles addObject:blas];

            [encoder useResource:input.meshes[i].vertexBuffer usage:MTLResourceUsageRead];
            [encoder useResource:input.meshes[i].indexBuffer usage:MTLResourceUsageRead];
                [encoder useResource:blas usage:MTLResourceUsageRead | MTLResourceUsageWrite];

            [encoder buildAccelerationStructure:blas
                                      descriptor:primitiveDescs[i]
                                   scratchBuffer:m_scratchBuffer
                             scratchBufferOffset:0];
        }

        std::vector<MTLAccelerationStructureUserIDInstanceDescriptor> instanceDescriptors(meshCount);
        for (uint32_t i = 0; i < meshCount; ++i) {
            const SceneAccelMeshInput& mesh = input.meshes[i];
            MTLAccelerationStructureUserIDInstanceDescriptor descriptor{};
            descriptor.transformationMatrix = ToPackedTransform(mesh.localToWorldTransform);
            descriptor.options = MTLAccelerationStructureInstanceOptionOpaque |
                                  MTLAccelerationStructureInstanceOptionDisableTriangleCulling |
                                  MTLAccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise;
            descriptor.mask = 0xFF;
            descriptor.intersectionFunctionTableOffset = 0;
            descriptor.accelerationStructureIndex = i;
            descriptor.userID = i;
            instanceDescriptors[i] = descriptor;
        }

        const NSUInteger instanceDataSize =
            static_cast<NSUInteger>(instanceDescriptors.size() * sizeof(MTLAccelerationStructureUserIDInstanceDescriptor));
        m_instanceBuffer = [input.device newBufferWithLength:instanceDataSize
                                                     options:MTLResourceStorageModeShared];
        if (!m_instanceBuffer) {
            return fail("Failed to allocate TLAS instance buffer");
        }

        memcpy([m_instanceBuffer contents], instanceDescriptors.data(), instanceDataSize);

        if (meshCount > 0) {
            const NSUInteger userIdBytes = static_cast<NSUInteger>(meshCount * sizeof(uint32_t));
            m_instanceUserIdBuffer = [input.device newBufferWithLength:userIdBytes
                                                               options:MTLResourceStorageModeShared];
            if (!m_instanceUserIdBuffer) {
                return fail("Failed to allocate instance user ID buffer");
            }
            uint32_t* ids = reinterpret_cast<uint32_t*>([m_instanceUserIdBuffer contents]);
            for (uint32_t i = 0; i < meshCount; ++i) {
                ids[i] = i;
            }
        } else {
            m_instanceUserIdBuffer = nil;
        }

        auto tlasDesc = [MTLInstanceAccelerationStructureDescriptor descriptor];
        tlasDesc.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeUserID;
        tlasDesc.instanceDescriptorBuffer = m_instanceBuffer;
        tlasDesc.instanceDescriptorStride = sizeof(MTLAccelerationStructureUserIDInstanceDescriptor);
        tlasDesc.instanceCount = meshCount;
        tlasDesc.instancedAccelerationStructures = blasHandles;
    tlasDesc.usage = MTLAccelerationStructureUsageNone;
        [encoder useResource:m_instanceBuffer usage:MTLResourceUsageRead];

        MTLAccelerationStructureSizes tlasSizes =
            [input.device accelerationStructureSizesWithDescriptor:tlasDesc];
        maxScratch = std::max(maxScratch, tlasSizes.buildScratchBufferSize);
        ensureScratchBuffer(input.device, maxScratch);

        m_tlas = [input.device newAccelerationStructureWithSize:tlasSizes.accelerationStructureSize];
        if (!m_tlas) {
            return fail("Failed to allocate TLAS");
        }

        [encoder useResource:m_tlas usage:MTLResourceUsageRead | MTLResourceUsageWrite];

      [encoder buildAccelerationStructure:m_tlas
                        descriptor:tlasDesc
                     scratchBuffer:m_scratchBuffer
                 scratchBufferOffset:0];

      m_instanceCount = meshCount;

      NSLog(@"MetalRtAccel::buildHardware - submitted TLAS build (%u instances, %u total primitives)",
          meshCount,
          totalPrimitiveCount);

        [encoder endEncoding];
        [commandBuffer commit];

      [commandBuffer waitUntilCompleted];

      if (commandBuffer.status != MTLCommandBufferStatusCompleted || commandBuffer.error) {
        releaseHardwareResources();
        return false;
      }

        m_blasHandles = [blasHandles mutableCopy];
        m_primitiveCount = totalPrimitiveCount;

        outProvider.mode = PathTracerShaderTypes::IntersectionMode::HardwareRayTracing;
        outProvider.software = SoftwareBvhResources{};
        outProvider.hardware = HardwareRtResources{};
        outProvider.hardware.tlas = m_tlas;
        outProvider.hardware.instanceBuffer = m_instanceBuffer;
        outProvider.hardware.instanceUserIDBuffer = m_instanceUserIdBuffer;
        outProvider.hardware.scratchBuffer = m_scratchBuffer;
        outProvider.hardware.instanceCount = m_instanceCount;
        outProvider.hardware.blasCount = static_cast<uint32_t>(m_blasHandles.count);
        outProvider.hardware.blas = m_blasHandles.count > 0 ? (MTLAccelerationStructureHandle)m_blasHandles[0] : nil;
        // Added: record all BLAS handles for encoder resource usage later.
        outProvider.hardware.blasHandles.clear();
        for (id<MTLAccelerationStructure> blasHandle in m_blasHandles) {
            outProvider.hardware.blasHandles.push_back((void*)blasHandle);
        }

#if PT_DEBUG_TOOLS || PT_MNEE_SWRT_RAYS || PT_MNEE_OCCLUSION_PARITY
        // Build a software BVH alongside HWRT so debug fallbacks can run without a mode switch.
        if (m_softwareFallback) {
            HardwareRtResources savedHardware = outProvider.hardware;
            PathTracerShaderTypes::IntersectionMode savedMode = outProvider.mode;
            m_softwareFallback->rebuild(input, outProvider);
            outProvider.hardware = savedHardware;
            outProvider.mode = savedMode;
        }
#endif

        NSLog(@"MetalRtAccel::buildHardware - TLAS ready (blasCount=%u, instanceCount=%u, primitiveCount=%u)",
              outProvider.hardware.blasCount,
              outProvider.hardware.instanceCount,
              m_primitiveCount);

        return true;
    }

    void fallbackToSoftware(const SceneAccelBuildInput& input,
                            IntersectionProvider& outProvider) {
        releaseHardwareResources();
        outProvider.hardware = HardwareRtResources{};
        if (!m_softwareFallback) {
            m_softwareFallback = std::make_unique<SoftwareBvhAccel>();
        }
        m_softwareFallback->rebuild(input, outProvider);
        m_mode = PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
        m_primitiveCount = m_softwareFallback ? m_softwareFallback->primitiveCount() : 0;
    }

    void ensureScratchBuffer(MTLDeviceHandle device, NSUInteger requiredSize) {
        if (!device || requiredSize == 0) {
            return;
        }

        if (!m_scratchBuffer || m_scratchBuffer.length < requiredSize) {
            m_scratchBuffer = [device newBufferWithLength:requiredSize
                                                  options:MTLResourceStorageModePrivate];
        }
    }

    void releaseHardwareResources() {
        m_blasHandles = nil;
        m_tlas = nil;
        m_instanceBuffer = nil;
        m_instanceUserIdBuffer = nil;
        m_instanceCount = 0;
        m_scratchBuffer = nil;
    }

    MTLCommandQueueHandle m_commandQueue = nullptr;
    bool m_canUseHardware = false;
    std::unique_ptr<SoftwareBvhAccel> m_softwareFallback;

    NSMutableArray<id<MTLAccelerationStructure>>* m_blasHandles = nil;
    MTLAccelerationStructureHandle m_tlas = nil;
    MTLBufferHandle m_instanceBuffer = nil;
    MTLBufferHandle m_instanceUserIdBuffer = nil;
    MTLBufferHandle m_scratchBuffer = nil;

    PathTracerShaderTypes::IntersectionMode m_mode =
        PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
    uint32_t m_primitiveCount = 0;
    uint32_t m_instanceCount = 0;
};

std::unique_ptr<SceneAccel> CreateSceneAccel(const SceneAccelConfig& config) {
    if (config.hardwareRaytracingSupported && config.commandQueue) {
        return std::make_unique<MetalRtAccel>(config);
    }
    return std::make_unique<SoftwareBvhAccel>();
}

}  // namespace PathTracer
