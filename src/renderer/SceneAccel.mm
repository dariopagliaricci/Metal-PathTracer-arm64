#import "renderer/SceneAccel.h"

#import <AppKit/AppKit.h>
#import <CoreFoundation/CoreFoundation.h>
#import <Metal/Metal.h>

#import "renderer/BvhBuilder.h"

#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace PathTracer {

namespace {

constexpr NSUInteger kIncrementalBlasBatchSize = 1u;
constexpr MTLAccelerationStructureUsage kAccelerationUsagePreferFastIntersection =
    static_cast<MTLAccelerationStructureUsage>(1u << 4u);
constexpr MTLAccelerationStructureUsage kAccelerationUsageMinimizeMemory =
    static_cast<MTLAccelerationStructureUsage>(1u << 5u);

// Shared signal struct captured by GPU completion handlers.
// Using shared_ptr ensures the struct stays alive even if MetalRtAccel is
// destroyed while a BLAS/TLAS command buffer is still in-flight on the GPU.
struct PendingBuildSignals {
    std::atomic<bool> batchFinished{false};
    std::atomic<bool> batchSucceeded{false};
    std::atomic<bool> tlasFinished{false};
    std::atomic<bool> tlasSucceeded{false};
};

struct MeshGeometryKey {
    const void* vertexBuffer = nullptr;
    const void* indexBuffer = nullptr;
    uint32_t vertexStride = 0u;
    uint32_t vertexCount = 0u;
    uint32_t indexCount = 0u;

    bool operator==(const MeshGeometryKey& other) const noexcept {
        return vertexBuffer == other.vertexBuffer &&
               indexBuffer == other.indexBuffer &&
               vertexStride == other.vertexStride &&
               vertexCount == other.vertexCount &&
               indexCount == other.indexCount;
    }
};

struct MeshGeometryKeyHasher {
    size_t operator()(const MeshGeometryKey& key) const noexcept {
        const size_t h0 = std::hash<const void*>{}(key.vertexBuffer);
        const size_t h1 = std::hash<const void*>{}(key.indexBuffer);
        const size_t h2 = std::hash<uint32_t>{}(key.vertexStride);
        const size_t h3 = std::hash<uint32_t>{}(key.vertexCount);
        const size_t h4 = std::hash<uint32_t>{}(key.indexCount);
        return h0 ^ (h1 << 1u) ^ (h2 << 2u) ^ (h3 << 3u) ^ (h4 << 4u);
    }
};

struct HardwareBuildPlan {
    std::vector<uint32_t> uniqueMeshIndices;
    std::vector<uint32_t> instanceToBlasIndex;
    uint32_t instancePrimitiveCount = 0u;
    uint64_t uniqueGeometryBytes = 0u;
};

HardwareBuildPlan BuildHardwareBuildPlan(const SceneAccelBuildInput& input) {
    HardwareBuildPlan plan{};
    if (!input.meshes || input.meshCount == 0u) {
        return plan;
    }

    plan.instanceToBlasIndex.resize(input.meshCount, 0u);
    plan.uniqueMeshIndices.reserve(input.meshCount);
    std::unordered_map<MeshGeometryKey, uint32_t, MeshGeometryKeyHasher> uniqueGeometryToBlasIndex;
    uniqueGeometryToBlasIndex.reserve(input.meshCount);

    for (uint32_t instanceIndex = 0; instanceIndex < input.meshCount; ++instanceIndex) {
        const SceneAccelMeshInput& mesh = input.meshes[instanceIndex];
        const uint32_t triangleCount = mesh.indexCount / 3u;
        plan.instancePrimitiveCount += triangleCount;

        const MeshGeometryKey key{
            .vertexBuffer = (__bridge const void*)mesh.vertexBuffer,
            .indexBuffer = (__bridge const void*)mesh.indexBuffer,
            .vertexStride = mesh.vertexStride,
            .vertexCount = mesh.vertexCount,
            .indexCount = mesh.indexCount,
        };

        auto it = uniqueGeometryToBlasIndex.find(key);
        if (it != uniqueGeometryToBlasIndex.end()) {
            plan.instanceToBlasIndex[instanceIndex] = it->second;
            continue;
        }

        const uint32_t blasIndex = static_cast<uint32_t>(plan.uniqueMeshIndices.size());
        uniqueGeometryToBlasIndex.emplace(key, blasIndex);
        plan.uniqueMeshIndices.push_back(instanceIndex);
        plan.instanceToBlasIndex[instanceIndex] = blasIndex;
        plan.uniqueGeometryBytes +=
            static_cast<uint64_t>(mesh.vertexStride) * static_cast<uint64_t>(mesh.vertexCount) +
            static_cast<uint64_t>(sizeof(uint32_t)) * static_cast<uint64_t>(mesh.indexCount);
    }

    return plan;
}

const char* BuildPolicyLabel(SceneAccelBuildPolicy policy) {
    switch (policy) {
        case SceneAccelBuildPolicy::SoftwareReference:
            return "software_reference";
        case SceneAccelBuildPolicy::HardwareFastBuild:
            return "hardware_fast_build";
        case SceneAccelBuildPolicy::HardwareFastTrace:
            return "hardware_fast_trace";
        case SceneAccelBuildPolicy::HardwareCompactMemory:
            return "hardware_compact_memory";
        case SceneAccelBuildPolicy::HardwareDynamicUpdate:
            return "hardware_dynamic_update";
    }
    return "unknown";
}

bool EnvStringEquals(const char* name, const char* expected) {
    const char* value = std::getenv(name);
    return value && std::strcmp(value, expected) == 0;
}

SceneAccelBuildPolicy SelectHardwareBuildPolicy(const SceneAccelBuildInput& input,
                                                const HardwareBuildPlan& plan) {
    constexpr uint32_t kLargeStaticPrimitiveThreshold = 1u << 20u;
    constexpr uint32_t kManyInstanceThreshold = 2048u;
    constexpr uint64_t kCompactGeometryByteThreshold = 512ull * 1024ull * 1024ull;
    if (EnvStringEquals("PT_HWRT_BUILD_POLICY", "fast_build")) {
        return SceneAccelBuildPolicy::HardwareFastBuild;
    }
    if (EnvStringEquals("PT_HWRT_BUILD_POLICY", "fast_trace")) {
        return SceneAccelBuildPolicy::HardwareFastTrace;
    }
    if (EnvStringEquals("PT_HWRT_BUILD_POLICY", "compact_memory")) {
        return SceneAccelBuildPolicy::HardwareCompactMemory;
    }
    if (EnvStringEquals("PT_HWRT_BUILD_POLICY", "dynamic_update")) {
        return SceneAccelBuildPolicy::HardwareDynamicUpdate;
    }
    if (input.meshCount >= kManyInstanceThreshold) {
        return SceneAccelBuildPolicy::HardwareFastBuild;
    }
    if (plan.uniqueGeometryBytes >= kCompactGeometryByteThreshold) {
        return SceneAccelBuildPolicy::HardwareCompactMemory;
    }
    if (plan.instancePrimitiveCount >= kLargeStaticPrimitiveThreshold) {
        return SceneAccelBuildPolicy::HardwareFastTrace;
    }
    return SceneAccelBuildPolicy::HardwareFastTrace;
}

MTLAccelerationStructureUsage MetalUsageForBuildPolicy(SceneAccelBuildPolicy policy) {
    MTLAccelerationStructureUsage usage = MTLAccelerationStructureUsageNone;
    switch (policy) {
        case SceneAccelBuildPolicy::HardwareFastBuild:
            usage |= MTLAccelerationStructureUsagePreferFastBuild;
            break;
        case SceneAccelBuildPolicy::HardwareFastTrace:
            if (@available(macOS 26.0, *)) {
                usage |= kAccelerationUsagePreferFastIntersection;
            }
            break;
        case SceneAccelBuildPolicy::HardwareCompactMemory:
            if (@available(macOS 26.0, *)) {
                usage |= kAccelerationUsageMinimizeMemory;
            }
            break;
        case SceneAccelBuildPolicy::HardwareDynamicUpdate:
            usage |= MTLAccelerationStructureUsageRefit | MTLAccelerationStructureUsagePreferFastBuild;
            break;
        case SceneAccelBuildPolicy::SoftwareReference:
            break;
    }
    return usage;
}

std::string MetalUsageFlagsLabel(MTLAccelerationStructureUsage usage) {
    if (usage == MTLAccelerationStructureUsageNone) {
        return "none";
    }
    std::string label;
    auto append = [&](const char* value) {
        if (!label.empty()) {
            label += "|";
        }
        label += value;
    };
    if ((usage & MTLAccelerationStructureUsageRefit) != 0) {
        append("refit");
    }
    if ((usage & MTLAccelerationStructureUsagePreferFastBuild) != 0) {
        append("prefer_fast_build");
    }
    if ((usage & MTLAccelerationStructureUsageExtendedLimits) != 0) {
        append("extended_limits");
    }
    if (@available(macOS 26.0, *)) {
        if ((usage & kAccelerationUsagePreferFastIntersection) != 0) {
            append("prefer_fast_intersection");
        }
        if ((usage & kAccelerationUsageMinimizeMemory) != 0) {
            append("minimize_memory");
        }
    }
    return label;
}

const char* BuildPolicyReason(SceneAccelBuildPolicy policy,
                              const SceneAccelBuildInput& input,
                              const HardwareBuildPlan& plan) {
    if (std::getenv("PT_HWRT_BUILD_POLICY")) {
        return "explicit PT_HWRT_BUILD_POLICY override";
    }
    switch (policy) {
        case SceneAccelBuildPolicy::HardwareFastBuild:
            return "many mesh instances favor faster rebuilds";
        case SceneAccelBuildPolicy::HardwareCompactMemory:
            return "large unique geometry footprint favors compact AS memory";
        case SceneAccelBuildPolicy::HardwareDynamicUpdate:
            return "dynamic update policy selected for refittable AS builds";
        case SceneAccelBuildPolicy::HardwareFastTrace:
            return plan.instancePrimitiveCount >= (1u << 20u)
                       ? "large static scene favors ray traversal speed"
                       : "default production HWRT policy favors ray traversal speed";
        case SceneAccelBuildPolicy::SoftwareReference:
            break;
    }
    (void)input;
    return "software reference fallback";
}

void PopulateSoftwarePolicyStats(SceneAccelBuildStats& stats) {
    stats.usedHardwareRaytracing = false;
    stats.buildPolicy = SceneAccelBuildPolicy::SoftwareReference;
    stats.buildPolicyLabel = BuildPolicyLabel(stats.buildPolicy);
    stats.deviceSupportsHardwareRaytracing = false;
    stats.identityStable = true;
    stats.identityStrategy =
        "software BVH primitive identity is derived from packed scene triangle and sphere indices";
    stats.intersectionFunctionStrategy =
        "software analytic triangle/sphere/rectangle intersection";
}

void PopulateHardwarePolicyStats(SceneAccelBuildStats& stats,
                                 SceneAccelBuildPolicy policy,
                                 const HardwareBuildPlan& plan,
                                 const SceneAccelBuildInput& input = SceneAccelBuildInput{}) {
    stats.usedHardwareRaytracing = true;
    stats.buildPolicy = policy;
    stats.buildPolicyLabel = BuildPolicyLabel(policy);
    stats.buildPolicyReason = BuildPolicyReason(policy, input, plan);
    stats.metalUsageFlags = MetalUsageFlagsLabel(MetalUsageForBuildPolicy(policy));
    stats.deviceSupportsHardwareRaytracing = true;
    stats.estimatedUniqueGeometryBytes = plan.uniqueGeometryBytes;
    stats.identityStable = plan.instanceToBlasIndex.size() > 0u ||
                           plan.instancePrimitiveCount == 0u;
    stats.identityStrategy =
        "TLAS userID is the scene mesh instance index; primitiveID is the BLAS local triangle index; material identity is resolved through MeshInfo";
    stats.customIntersectionFunctionsSupported = false;
    stats.customIntersectionFunctionsEnabled = false;
    stats.intersectionFunctionStrategy =
        "native Metal triangle acceleration for mesh BLAS; spheres/rectangles use shader-side analytic fallback; custom intersection functions are disabled";
}

void PumpMainThreadEventsBriefly() {
    if (![NSThread isMainThread]) {
        [NSThread sleepForTimeInterval:0.001];
        return;
    }
    NSApplication* app = [NSApplication sharedApplication];
    if (!app) {
        [NSThread sleepForTimeInterval:0.001];
        return;
    }
    NSDate* deadline = [NSDate dateWithTimeIntervalSinceNow:0.001];
    NSArray<NSString*>* modes = @[
        NSDefaultRunLoopMode,
        NSEventTrackingRunLoopMode,
        NSModalPanelRunLoopMode
    ];
    for (NSString* mode in modes) {
        while (true) {
            NSEvent* event = [app nextEventMatchingMask:NSEventMaskAny
                                              untilDate:deadline
                                                 inMode:mode
                                                dequeue:YES];
            if (!event) {
                break;
            }
            [app sendEvent:event];
        }
    }
}

void WaitForCommandBufferResponsive(id<MTLCommandBuffer> commandBuffer) {
    if (!commandBuffer) {
        return;
    }
    while (true) {
        const MTLCommandBufferStatus status = commandBuffer.status;
        if (status == MTLCommandBufferStatusCompleted ||
            status == MTLCommandBufferStatusError) {
            break;
        }
        PumpMainThreadEventsBriefly();
    }
}

}  // namespace

class SoftwareBvhAccel final : public SceneAccel {
public:
    SoftwareBvhAccel()
        : m_bvhBuilder(std::make_unique<BvhBuilder>(4)) {
    }

    void rebuild(const SceneAccelBuildInput& input,
                 IntersectionProvider& outProvider) override {
        m_primitiveCount = 0;
        m_buildStats = SceneAccelBuildStats{};
        outProvider.hardware = HardwareRtResources{};
        PopulateSoftwarePolicyStats(m_buildStats);

        if (!input.device) {
            outProvider.mode = PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
            outProvider.software = SoftwareBvhResources{};
            m_buildStats.fallbackReason = "Metal device unavailable for acceleration structure build";
            return;
        }

        // If we have meshes, build per-mesh BLAS and a TLAS over instance AABBs
        if (input.meshes && input.meshCount > 0) {
            struct Aabb { simd::float3 min; simd::float3 max; };
            const CFAbsoluteTime blasStart = CFAbsoluteTimeGetCurrent();

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
            m_buildStats.blasBuildMs = (CFAbsoluteTimeGetCurrent() - blasStart) * 1000.0;

            // Build TLAS over instance AABBs
            const CFAbsoluteTime tlasStart = CFAbsoluteTimeGetCurrent();
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
            m_buildStats.tlasBuildMs = (CFAbsoluteTimeGetCurrent() - tlasStart) * 1000.0;

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
            m_buildStats.primitiveCount = m_primitiveCount;
            m_buildStats.instanceCount = static_cast<uint32_t>(instanceInfos.size());
            m_buildStats.blasCount = static_cast<uint32_t>(instanceInfos.size());
            m_buildStats.blasMemoryBytes =
                static_cast<uint64_t>(packedBlasNodes.size()) * sizeof(PathTracerShaderTypes::BvhNode) +
                static_cast<uint64_t>(packedBlasPrimIndices.size()) * sizeof(uint32_t);
            m_buildStats.tlasMemoryBytes =
                static_cast<uint64_t>(tlasNodes.size()) * sizeof(PathTracerShaderTypes::BvhNode) +
                static_cast<uint64_t>(tlasPrimIdx.size()) * sizeof(uint32_t) +
                static_cast<uint64_t>(instanceInfos.size()) * sizeof(PathTracerShaderTypes::SoftwareInstanceInfo);
            PopulateSoftwarePolicyStats(m_buildStats);
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
        const CFAbsoluteTime blasStart = CFAbsoluteTimeGetCurrent();
        BvhBuildOutput builderOutput = m_bvhBuilder->build(builderInput);
        m_buildStats.blasBuildMs = (CFAbsoluteTimeGetCurrent() - blasStart) * 1000.0;

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
        m_buildStats.primitiveCount = builderOutput.primitiveCount;
        m_buildStats.instanceCount = input.sphereCount;
        m_buildStats.blasCount = input.sphereCount > 0 ? 1u : 0u;
        m_buildStats.blasMemoryBytes =
            static_cast<uint64_t>(builderOutput.nodeCount) * sizeof(PathTracerShaderTypes::BvhNode) +
            static_cast<uint64_t>(builderOutput.primitiveCount) * sizeof(uint32_t);
        PopulateSoftwarePolicyStats(m_buildStats);
        m_buildStats.fallbackReason = "Software BVH path active";
    }

    SceneAccelBuildStats estimateBuildStats(const SceneAccelBuildInput& input) override {
        SceneAccelBuildStats stats{};
        PopulateSoftwarePolicyStats(stats);

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
                    instanceAabbs.push_back({{0, 0, 0}, {0, 0, 0}});
                    continue;
                }

                const uint8_t* vbase = reinterpret_cast<const uint8_t*>([mesh.vertexBuffer contents]);
                const uint32_t* indices = reinterpret_cast<const uint32_t*>([mesh.indexBuffer contents]);
                if (!vbase || !indices) {
                    instanceInfos.push_back({});
                    instanceAabbs.push_back({{0, 0, 0}, {0, 0, 0}});
                    triangleBaseOffset += triCount;
                    continue;
                }

                std::vector<tinybvh::bvhvec4> blasVerts;
                blasVerts.reserve(triCount * 3u);
                simd::float3 localMin{std::numeric_limits<float>::max(),
                                      std::numeric_limits<float>::max(),
                                      std::numeric_limits<float>::max()};
                simd::float3 localMax{-std::numeric_limits<float>::max(),
                                      -std::numeric_limits<float>::max(),
                                      -std::numeric_limits<float>::max()};

                for (uint32_t t = 0; t < triCount; ++t) {
                    const uint32_t i0 = indices[t * 3u + 0u];
                    const uint32_t i1 = indices[t * 3u + 1u];
                    const uint32_t i2 = indices[t * 3u + 2u];
                    if (i0 >= mesh.vertexCount || i1 >= mesh.vertexCount || i2 >= mesh.vertexCount) {
                        blasVerts.push_back(tinybvh::bvhvec4(0, 0, 0, 0));
                        blasVerts.push_back(tinybvh::bvhvec4(0, 0, 0, 0));
                        blasVerts.push_back(tinybvh::bvhvec4(0, 0, 0, 0));
                        continue;
                    }
                    const float* p0 = reinterpret_cast<const float*>(vbase + static_cast<size_t>(i0) * mesh.vertexStride);
                    const float* p1 = reinterpret_cast<const float*>(vbase + static_cast<size_t>(i1) * mesh.vertexStride);
                    const float* p2 = reinterpret_cast<const float*>(vbase + static_cast<size_t>(i2) * mesh.vertexStride);

                    simd::float3 v0{p0[0], p0[1], p0[2]};
                    simd::float3 v1{p1[0], p1[1], p1[2]};
                    simd::float3 v2{p2[0], p2[1], p2[2]};

                    blasVerts.push_back(tinybvh::bvhvec4(v0.x, v0.y, v0.z, 0.0f));
                    blasVerts.push_back(tinybvh::bvhvec4(v1.x, v1.y, v1.z, 0.0f));
                    blasVerts.push_back(tinybvh::bvhvec4(v2.x, v2.y, v2.z, 0.0f));

                    localMin = simd::min(localMin, simd::min(v0, simd::min(v1, v2)));
                    localMax = simd::max(localMax, simd::max(v0, simd::max(v1, v2)));
                }

                tinybvh::BVH blas;
                blas.Build(blasVerts.data(), triCount);

                const uint32_t usedNodeCount = blas.usedNodes;
                for (uint32_t n = 0; n < usedNodeCount; ++n) {
                    const tinybvh::BVH::BVHNode& srcNode = blas.bvhNode[n];
                    PathTracerShaderTypes::BvhNode dstNode{};
                    dstNode.boundsMin = simd_make_float4(srcNode.aabbMin.x, srcNode.aabbMin.y, srcNode.aabbMin.z, 0.0f);
                    dstNode.boundsMax = simd_make_float4(srcNode.aabbMax.x, srcNode.aabbMax.y, srcNode.aabbMax.z, 0.0f);
                    packedBlasNodes.push_back(dstNode);
                }
                for (uint32_t p = 0; p < triCount; ++p) {
                    packedBlasPrimIndices.push_back(blas.primIdx[p]);
                }

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
                info.triangleBaseOffset = triangleBaseOffset;
                info.triangleCount = triCount;
                info.localToWorld = mesh.localToWorldTransform;
                info.worldToLocal = simd_inverse(mesh.localToWorldTransform);
                instanceInfos.push_back(info);
                instanceAabbs.push_back({worldMin, worldMax});
                triangleBaseOffset += triCount;
            }

            auto buildTLAS = [&](const std::vector<Aabb>& aabbs,
                                 std::vector<PathTracerShaderTypes::BvhNode>& outNodes,
                                 std::vector<uint32_t>& outPrimIdx) {
                struct Prim {
                    simd::float3 bmin;
                    simd::float3 bmax;
                    simd::float3 centroid;
                    uint32_t idx;
                };
                std::vector<Prim> prims;
                prims.reserve(aabbs.size());
                for (uint32_t i = 0; i < aabbs.size(); ++i) {
                    const auto& b = aabbs[i];
                    prims.push_back(Prim{b.min, b.max, (b.min + b.max) * 0.5f, i});
                }
                outNodes.clear();
                outPrimIdx.resize(prims.size());

                std::function<uint32_t(uint32_t, uint32_t)> buildRec =
                    [&](uint32_t start, uint32_t end) -> uint32_t {
                        PathTracerShaderTypes::BvhNode node{};
                        simd::float3 bmin{std::numeric_limits<float>::max()};
                        simd::float3 bmax{-std::numeric_limits<float>::max()};
                        simd::float3 cmin = bmin;
                        simd::float3 cmax = -bmax;
                        for (uint32_t i = start; i < end; ++i) {
                            const Prim& pr = prims[i];
                            bmin = simd::min(bmin, pr.bmin);
                            bmax = simd::max(bmax, pr.bmax);
                            cmin = simd::min(cmin, pr.centroid);
                            cmax = simd::max(cmax, pr.centroid);
                        }
                        const uint32_t nodeIdx = static_cast<uint32_t>(outNodes.size());
                        outNodes.emplace_back(node);
                        const uint32_t count = end - start;
                        if (count <= 4u) {
                            for (uint32_t i = 0; i < count; ++i) {
                                outPrimIdx[start + i] = prims[start + i].idx;
                            }
                            return nodeIdx;
                        }
                        simd::float3 ext = cmax - cmin;
                        uint32_t axis = 0;
                        if (ext.y > ext.x && ext.y > ext.z) {
                            axis = 1;
                        } else if (ext.z > ext.x) {
                            axis = 2;
                        }
                        const uint32_t mid = (start + end) / 2u;
                        std::nth_element(prims.begin() + start,
                                         prims.begin() + mid,
                                         prims.begin() + end,
                                         [&](const Prim& a, const Prim& b) {
                                             return a.centroid[axis] < b.centroid[axis];
                                         });
                        buildRec(start, mid);
                        buildRec(mid, end);
                        return nodeIdx;
                    };

                if (!prims.empty()) {
                    buildRec(0u, static_cast<uint32_t>(prims.size()));
                }
            };

            std::vector<PathTracerShaderTypes::BvhNode> tlasNodes;
            std::vector<uint32_t> tlasPrimIdx;
            buildTLAS(instanceAabbs, tlasNodes, tlasPrimIdx);

            stats.primitiveCount = static_cast<uint32_t>(packedBlasPrimIndices.size());
            stats.instanceCount = static_cast<uint32_t>(instanceInfos.size());
            stats.blasCount = static_cast<uint32_t>(instanceInfos.size());
            stats.blasMemoryBytes =
                static_cast<uint64_t>(packedBlasNodes.size()) * sizeof(PathTracerShaderTypes::BvhNode) +
                static_cast<uint64_t>(packedBlasPrimIndices.size()) * sizeof(uint32_t);
            stats.tlasMemoryBytes =
                static_cast<uint64_t>(tlasNodes.size()) * sizeof(PathTracerShaderTypes::BvhNode) +
                static_cast<uint64_t>(tlasPrimIdx.size()) * sizeof(uint32_t) +
                static_cast<uint64_t>(instanceInfos.size()) * sizeof(PathTracerShaderTypes::SoftwareInstanceInfo);
            stats.fallbackReason = "Software BVH path active";
            return stats;
        }

        if (input.spheres && input.sphereCount > 0) {
            BvhBuilder builder(4);
            BvhBuildInput builderInput{};
            builderInput.spheres = input.spheres;
            builderInput.sphereCount = input.sphereCount;
            BvhBuildOutput builderOutput = builder.build(builderInput);
            stats.primitiveCount = builderOutput.primitiveCount;
            stats.instanceCount = input.sphereCount;
            stats.blasCount = input.sphereCount > 0 ? 1u : 0u;
            stats.blasMemoryBytes =
                static_cast<uint64_t>(builderOutput.nodeCount) * sizeof(PathTracerShaderTypes::BvhNode) +
                static_cast<uint64_t>(builderOutput.primitiveCount) * sizeof(uint32_t);
            stats.fallbackReason = "Software BVH path active";
        }

        return stats;
    }

    bool beginIncrementalRebuild(const SceneAccelBuildInput& input) override {
        m_incrementalInput = input;
        m_incrementalActive = true;
        return true;
    }

    bool stepIncrementalRebuild(IntersectionProvider& outProvider,
                                SceneAccelBuildProgress& outProgress) override {
        if (!m_incrementalActive) {
            outProgress.complete = true;
            return true;
        }
        rebuild(m_incrementalInput, outProvider);
        m_incrementalActive = false;
        outProgress.blasBatchesDone = 0;
        outProgress.blasBatchesTotal = 0;
        outProgress.complete = true;
        return true;
    }

    bool incrementalRebuildActive() const override {
        return m_incrementalActive;
    }

    void clear() override {
        m_primitiveCount = 0;
        m_buildStats = SceneAccelBuildStats{};
        m_incrementalInput = SceneAccelBuildInput{};
        m_incrementalActive = false;
    }

    PathTracerShaderTypes::IntersectionMode mode() const override {
        return PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
    }

    uint32_t primitiveCount() const override {
        return m_primitiveCount;
    }

    const SceneAccelBuildStats& buildStats() const override {
        return m_buildStats;
    }

private:
    std::unique_ptr<BvhBuilder> m_bvhBuilder;
    uint32_t m_primitiveCount = 0;
    SceneAccelBuildStats m_buildStats{};
    SceneAccelBuildInput m_incrementalInput{};
    bool m_incrementalActive = false;
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
        SceneAccelBuildProgress progress{};
        if (!beginIncrementalRebuild(input)) {
            fallbackToSoftware(input, outProvider);
            return;
        }
        while (incrementalRebuildActive()) {
            if (!stepIncrementalRebuild(outProvider, progress)) {
                fallbackToSoftware(input, outProvider);
                return;
            }
        }
    }

    SceneAccelBuildStats estimateBuildStats(const SceneAccelBuildInput& input) override {
        auto estimateSoftware = [&](const char* reason) -> SceneAccelBuildStats {
            if (!m_softwareFallback) {
                m_softwareFallback = std::make_unique<SoftwareBvhAccel>();
            }
            SceneAccelBuildStats stats = m_softwareFallback->estimateBuildStats(input);
            stats.deviceSupportsHardwareRaytracing = m_canUseHardware;
            if (reason && *reason) {
                stats.fallbackReason = reason;
            }
            return stats;
        };

        if (!m_canUseHardware) {
            return estimateSoftware("Hardware ray tracing unsupported or disabled");
        }
        if (!input.device) {
            return estimateSoftware("Metal device unavailable");
        }
        if (!input.meshes || input.meshCount == 0) {
            return estimateSoftware("Scene has no mesh instances for hardware ray tracing");
        }
        if (!input.commandQueue && !m_commandQueue) {
            return estimateSoftware("Metal command queue unavailable for hardware ray tracing");
        }

        SceneAccelBuildStats stats{};
        stats.usedHardwareRaytracing = true;

        const uint32_t meshCount = input.meshCount;
        const HardwareBuildPlan plan = BuildHardwareBuildPlan(input);
        const SceneAccelBuildPolicy buildPolicy = SelectHardwareBuildPolicy(input, plan);
        const MTLAccelerationStructureUsage accelerationUsage =
            MetalUsageForBuildPolicy(buildPolicy);
        const uint32_t uniqueMeshCount = static_cast<uint32_t>(plan.uniqueMeshIndices.size());
        std::vector<MTLPrimitiveAccelerationStructureDescriptor*> primitiveDescs;
        primitiveDescs.reserve(uniqueMeshCount);
        NSMutableArray<id<MTLAccelerationStructure>>* blasHandles =
            [NSMutableArray arrayWithCapacity:uniqueMeshCount];

        uint64_t totalBlasBytes = 0;
        NSUInteger maxScratch = 0;
        NSUInteger maxBlasScratch = 0;
        for (uint32_t uniqueIndex = 0; uniqueIndex < uniqueMeshCount; ++uniqueIndex) {
            const uint32_t meshIndex = plan.uniqueMeshIndices[uniqueIndex];
            const SceneAccelMeshInput& mesh = input.meshes[meshIndex];
            if (!mesh.vertexBuffer || !mesh.indexBuffer || mesh.vertexCount == 0 || mesh.indexCount < 3) {
                return estimateSoftware("Invalid mesh data for hardware ray tracing");
            }
            const uint32_t triangleCount = mesh.indexCount / 3u;
            if (triangleCount == 0) {
                return estimateSoftware("Mesh contains no triangles");
            }

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
            primDesc.usage = accelerationUsage;
            primitiveDescs.push_back(primDesc);

            const MTLAccelerationStructureSizes sizes =
                [input.device accelerationStructureSizesWithDescriptor:primDesc];
            id<MTLAccelerationStructure> blas =
                [input.device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
            if (!blas) {
                return estimateSoftware("Failed to allocate temporary BLAS for size estimate");
            }
            [blasHandles addObject:blas];
            totalBlasBytes += static_cast<uint64_t>(sizes.accelerationStructureSize);
            maxBlasScratch = std::max(maxBlasScratch, sizes.buildScratchBufferSize);
            maxScratch = std::max(maxScratch, sizes.buildScratchBufferSize);
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
            descriptor.accelerationStructureIndex = plan.instanceToBlasIndex[i];
            descriptor.userID = i;
            instanceDescriptors[i] = descriptor;
        }

        const NSUInteger instanceDataSize =
            static_cast<NSUInteger>(instanceDescriptors.size() * sizeof(MTLAccelerationStructureUserIDInstanceDescriptor));
        id<MTLBuffer> instanceBuffer =
            [input.device newBufferWithLength:instanceDataSize options:MTLResourceStorageModeShared];
        if (!instanceBuffer) {
            return estimateSoftware("Failed to allocate temporary TLAS instance buffer for size estimate");
        }
        memcpy([instanceBuffer contents], instanceDescriptors.data(), instanceDataSize);

        auto tlasDesc = [MTLInstanceAccelerationStructureDescriptor descriptor];
        tlasDesc.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeUserID;
        tlasDesc.instanceDescriptorBuffer = instanceBuffer;
        tlasDesc.instanceDescriptorStride = sizeof(MTLAccelerationStructureUserIDInstanceDescriptor);
        tlasDesc.instanceCount = meshCount;
        tlasDesc.instancedAccelerationStructures = blasHandles;
        tlasDesc.usage = accelerationUsage;

        const MTLAccelerationStructureSizes tlasSizes =
            [input.device accelerationStructureSizesWithDescriptor:tlasDesc];
        const NSUInteger tlasScratch = tlasSizes.buildScratchBufferSize;
        maxScratch = std::max(maxScratch, tlasSizes.buildScratchBufferSize);

        stats.primitiveCount = plan.instancePrimitiveCount;
        stats.instanceCount = meshCount;
        stats.blasCount = uniqueMeshCount;
        stats.blasMemoryBytes = totalBlasBytes;
        stats.tlasMemoryBytes = static_cast<uint64_t>(tlasSizes.accelerationStructureSize);
        stats.scratchMemoryBytes = static_cast<uint64_t>(maxScratch);
        stats.blasScratchMemoryBytes = static_cast<uint64_t>(maxBlasScratch);
        stats.tlasScratchMemoryBytes = static_cast<uint64_t>(tlasScratch);
        PopulateHardwarePolicyStats(stats, buildPolicy, plan, input);
        return stats;
    }

    bool beginIncrementalRebuild(const SceneAccelBuildInput& input) override {
        releaseHardwareResources();
        m_incrementalInput = input;
        m_incrementalPrimitiveDescs = nil;
        m_incrementalBlasSizes.clear();
        m_incrementalTotalBlasBytes = 0u;
        m_incrementalUniqueGeometryBytes = 0u;
        m_incrementalMaxScratch = 0u;
        m_incrementalMaxBlasScratch = 0u;
        m_incrementalTlasScratch = 0u;
        m_incrementalTotalPrimitiveCount = 0u;
        m_incrementalBuildPolicy = SceneAccelBuildPolicy::SoftwareReference;
        m_incrementalBlasSourceMeshIndices.clear();
        m_incrementalInstanceToBlasIndex.clear();
        m_incrementalCurrentBatch = 0u;
        m_incrementalBatchCount = 0u;
        m_incrementalStage = IncrementalStage::Idle;
        // Allocate fresh signals so any stale in-flight completion handlers from a
        // previous (possibly cancelled) build write into an orphaned struct that no
        // one reads, preventing use-after-free and stale signal races.
        m_buildSignals = std::make_shared<PendingBuildSignals>();
        m_incrementalTlasStart = 0.0;
        m_incrementalTlasSizeBytes = 0u;
        m_primitiveCount = 0;
        m_buildStats = SceneAccelBuildStats{};
        m_buildStats.usedHardwareRaytracing = m_canUseHardware;
        m_buildStats.deviceSupportsHardwareRaytracing = m_canUseHardware;

        if (!m_canUseHardware || !input.device || !input.meshes || input.meshCount == 0) {
            m_incrementalStage = IncrementalStage::SoftwareFallback;
            return true;
        }

        if (!input.commandQueue && !m_commandQueue) {
            m_buildStats.fallbackReason = "Metal command queue unavailable for hardware ray tracing";
            m_incrementalStage = IncrementalStage::SoftwareFallback;
            return true;
        }

        const uint32_t meshCount = input.meshCount;
        const HardwareBuildPlan plan = BuildHardwareBuildPlan(input);
        m_incrementalBuildPolicy = SelectHardwareBuildPolicy(input, plan);
        const MTLAccelerationStructureUsage accelerationUsage =
            MetalUsageForBuildPolicy(m_incrementalBuildPolicy);
        const uint32_t uniqueMeshCount = static_cast<uint32_t>(plan.uniqueMeshIndices.size());
        m_incrementalPrimitiveDescs = [[NSMutableArray alloc] initWithCapacity:uniqueMeshCount];
        m_incrementalBlasSizes.reserve(uniqueMeshCount);
        m_blasHandles = [[NSMutableArray alloc] initWithCapacity:uniqueMeshCount];
        m_incrementalBlasSourceMeshIndices = plan.uniqueMeshIndices;
        m_incrementalInstanceToBlasIndex = plan.instanceToBlasIndex;
        m_incrementalTotalPrimitiveCount = plan.instancePrimitiveCount;
        m_incrementalUniqueGeometryBytes = plan.uniqueGeometryBytes;

        for (uint32_t uniqueIndex = 0; uniqueIndex < uniqueMeshCount; ++uniqueIndex) {
            const uint32_t meshIndex = m_incrementalBlasSourceMeshIndices[uniqueIndex];
            const SceneAccelMeshInput& mesh = input.meshes[meshIndex];

            if (!mesh.vertexBuffer || !mesh.indexBuffer || mesh.vertexCount == 0 || mesh.indexCount < 3) {
                m_buildStats.fallbackReason = "Invalid mesh data for BLAS build";
                m_incrementalStage = IncrementalStage::SoftwareFallback;
                return true;
            }

            const uint32_t triangleCount = mesh.indexCount / 3u;
            if (triangleCount == 0) {
                m_buildStats.fallbackReason = "Mesh contains no triangles";
                m_incrementalStage = IncrementalStage::SoftwareFallback;
                return true;
            }

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
            primDesc.usage = accelerationUsage;
            [m_incrementalPrimitiveDescs addObject:primDesc];

            MTLAccelerationStructureSizes sizes =
                [input.device accelerationStructureSizesWithDescriptor:primDesc];
            m_incrementalBlasSizes.push_back(sizes);
            m_incrementalMaxBlasScratch =
                std::max(m_incrementalMaxBlasScratch, sizes.buildScratchBufferSize);
            m_incrementalMaxScratch =
                std::max(m_incrementalMaxScratch, sizes.buildScratchBufferSize);
            m_incrementalTotalBlasBytes += static_cast<uint64_t>(sizes.accelerationStructureSize);

            id<MTLAccelerationStructure> blas =
                [input.device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
            if (!blas) {
                m_buildStats.fallbackReason = "Failed to allocate BLAS acceleration structure";
                m_incrementalStage = IncrementalStage::SoftwareFallback;
                return true;
            }
            [m_blasHandles addObject:blas];
        }

        if (m_incrementalTotalPrimitiveCount == 0u) {
            m_buildStats.fallbackReason = "No triangles found while building BLAS";
            m_incrementalStage = IncrementalStage::SoftwareFallback;
            return true;
        }

        ensureScratchBuffer(input.device, m_incrementalMaxScratch);
        constexpr NSUInteger kBlasBatchSize = kIncrementalBlasBatchSize;
        m_incrementalBatchCount =
            (static_cast<NSUInteger>(m_incrementalPrimitiveDescs.count) + kBlasBatchSize - 1u) /
            kBlasBatchSize;
        m_incrementalBlasStart = CFAbsoluteTimeGetCurrent();
        m_incrementalStage = IncrementalStage::BuildBLASBatches;
        return true;
    }

    bool stepIncrementalRebuild(IntersectionProvider& outProvider,
                                SceneAccelBuildProgress& outProgress) override {
        outProgress = SceneAccelBuildProgress{};

        if (m_incrementalStage == IncrementalStage::Idle) {
            outProgress.complete = true;
            return true;
        }

        if (m_incrementalStage == IncrementalStage::SoftwareFallback) {
            fallbackToSoftware(m_incrementalInput, outProvider);
            m_incrementalStage = IncrementalStage::Idle;
            outProgress.complete = true;
            return true;
        }

        id<MTLCommandQueue> queue =
            m_incrementalInput.commandQueue ? m_incrementalInput.commandQueue : m_commandQueue;
        if (!queue) {
            fallbackToSoftware(m_incrementalInput, outProvider);
            m_incrementalStage = IncrementalStage::Idle;
            outProgress.complete = true;
            return true;
        }

        auto fallbackAndComplete = [&]() -> bool {
            // Orphan current signals so any in-flight handler writes to a dead struct.
            m_buildSignals = std::make_shared<PendingBuildSignals>();
            fallbackToSoftware(m_incrementalInput, outProvider);
            m_incrementalStage = IncrementalStage::Idle;
            outProgress.complete = true;
            return true;
        };

        if (m_incrementalStage == IncrementalStage::BuildBLASBatches) {
            constexpr NSUInteger kBlasBatchSize = kIncrementalBlasBatchSize;
            const NSUInteger batchIndex = m_incrementalCurrentBatch;
            const NSUInteger batchStart = batchIndex * kBlasBatchSize;
            const NSUInteger batchEnd =
                std::min<NSUInteger>(m_incrementalPrimitiveDescs.count, batchStart + kBlasBatchSize);

            id<MTLCommandBuffer> batchCommandBuffer = [queue commandBuffer];
            if (!batchCommandBuffer) {
                m_buildStats.fallbackReason = "Failed to allocate Metal command buffer for BLAS batch";
                return fallbackAndComplete();
            }
            batchCommandBuffer.label = [NSString stringWithFormat:@"BLAS Batch %lu/%lu",
                                        static_cast<unsigned long>(batchIndex + 1u),
                                        static_cast<unsigned long>(m_incrementalBatchCount)];
            id<MTLAccelerationStructureCommandEncoder> batchEncoder =
                [batchCommandBuffer accelerationStructureCommandEncoder];
            if (!batchEncoder) {
                m_buildStats.fallbackReason =
                    "Failed to create acceleration structure encoder for BLAS batch";
                return fallbackAndComplete();
            }

            for (NSUInteger i = batchStart; i < batchEnd; ++i) {
                const uint32_t sourceMeshIndex = m_incrementalBlasSourceMeshIndices[i];
                [batchEncoder useResource:m_incrementalInput.meshes[sourceMeshIndex].vertexBuffer
                                    usage:MTLResourceUsageRead];
                [batchEncoder useResource:m_incrementalInput.meshes[sourceMeshIndex].indexBuffer
                                    usage:MTLResourceUsageRead];
                [batchEncoder useResource:(id<MTLAccelerationStructure>)m_blasHandles[i]
                                    usage:MTLResourceUsageRead | MTLResourceUsageWrite];
                [batchEncoder buildAccelerationStructure:(id<MTLAccelerationStructure>)m_blasHandles[i]
                                              descriptor:m_incrementalPrimitiveDescs[i]
                                           scratchBuffer:m_scratchBuffer
                                     scratchBufferOffset:0];
            }

            [batchEncoder endEncoding];
            m_buildSignals->batchFinished.store(false, std::memory_order_release);
            m_buildSignals->batchSucceeded.store(false, std::memory_order_release);
            {
                // Capture the shared_ptr by value so the handler is safe even if
                // MetalRtAccel is destroyed before the command buffer completes.
                auto signals = m_buildSignals;
                [batchCommandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                    const bool ok =
                        (cb.status == MTLCommandBufferStatusCompleted) && (cb.error == nil);
                    signals->batchSucceeded.store(ok, std::memory_order_release);
                    signals->batchFinished.store(true, std::memory_order_release);
                }];
            }
            [batchCommandBuffer commit];
            m_incrementalStage = IncrementalStage::AwaitBLASBatch;
            outProgress.blasBatchesDone = static_cast<uint32_t>(m_incrementalCurrentBatch);
            outProgress.blasBatchesTotal = static_cast<uint32_t>(m_incrementalBatchCount);
            return true;
        }

        if (m_incrementalStage == IncrementalStage::AwaitBLASBatch) {
            if (m_buildSignals->batchFinished.load(std::memory_order_acquire)) {
                if (!m_buildSignals->batchSucceeded.load(std::memory_order_acquire)) {
                    m_buildStats.fallbackReason = "Metal BLAS batch build did not complete successfully";
                    return fallbackAndComplete();
                }
                ++m_incrementalCurrentBatch;
                outProgress.blasBatchesDone = static_cast<uint32_t>(m_incrementalCurrentBatch);
                outProgress.blasBatchesTotal = static_cast<uint32_t>(m_incrementalBatchCount);
                if (m_incrementalCurrentBatch >= m_incrementalBatchCount) {
                    m_buildStats.blasBuildMs =
                        (CFAbsoluteTimeGetCurrent() - m_incrementalBlasStart) * 1000.0;
                    m_incrementalStage = IncrementalStage::BuildTLAS;
                } else {
                    m_incrementalStage = IncrementalStage::BuildBLASBatches;
                }
            }
            outProgress.blasBatchesDone = static_cast<uint32_t>(m_incrementalCurrentBatch);
            outProgress.blasBatchesTotal = static_cast<uint32_t>(m_incrementalBatchCount);
            return true;
        }

        if (m_incrementalStage == IncrementalStage::BuildTLAS) {
            const uint32_t meshCount = m_incrementalInput.meshCount;
            std::vector<MTLAccelerationStructureUserIDInstanceDescriptor> instanceDescriptors(meshCount);
            for (uint32_t i = 0; i < meshCount; ++i) {
                const SceneAccelMeshInput& mesh = m_incrementalInput.meshes[i];
                MTLAccelerationStructureUserIDInstanceDescriptor descriptor{};
                descriptor.transformationMatrix = ToPackedTransform(mesh.localToWorldTransform);
                descriptor.options = MTLAccelerationStructureInstanceOptionOpaque |
                                     MTLAccelerationStructureInstanceOptionDisableTriangleCulling |
                                     MTLAccelerationStructureInstanceOptionTriangleFrontFacingWindingCounterClockwise;
                descriptor.mask = 0xFF;
                descriptor.intersectionFunctionTableOffset = 0;
                descriptor.accelerationStructureIndex =
                    (i < m_incrementalInstanceToBlasIndex.size()) ? m_incrementalInstanceToBlasIndex[i] : i;
                descriptor.userID = i;
                instanceDescriptors[i] = descriptor;
            }

            const NSUInteger instanceDataSize = static_cast<NSUInteger>(
                instanceDescriptors.size() * sizeof(MTLAccelerationStructureUserIDInstanceDescriptor));
            m_instanceBuffer =
                [m_incrementalInput.device newBufferWithLength:instanceDataSize
                                                       options:MTLResourceStorageModeShared];
            if (!m_instanceBuffer) {
                m_buildStats.fallbackReason = "Failed to allocate TLAS instance buffer";
                return fallbackAndComplete();
            }
            memcpy([m_instanceBuffer contents], instanceDescriptors.data(), instanceDataSize);

            if (meshCount > 0) {
                const NSUInteger userIdBytes = static_cast<NSUInteger>(meshCount * sizeof(uint32_t));
                m_instanceUserIdBuffer =
                    [m_incrementalInput.device newBufferWithLength:userIdBytes
                                                           options:MTLResourceStorageModeShared];
                if (!m_instanceUserIdBuffer) {
                    m_buildStats.fallbackReason = "Failed to allocate instance user ID buffer";
                    return fallbackAndComplete();
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
            tlasDesc.instancedAccelerationStructures = m_blasHandles;
            tlasDesc.usage = MetalUsageForBuildPolicy(m_incrementalBuildPolicy);

            m_incrementalTlasStart = CFAbsoluteTimeGetCurrent();
            MTLAccelerationStructureSizes tlasSizes =
                [m_incrementalInput.device accelerationStructureSizesWithDescriptor:tlasDesc];
            m_incrementalTlasSizeBytes = static_cast<uint64_t>(tlasSizes.accelerationStructureSize);
            m_incrementalTlasScratch = tlasSizes.buildScratchBufferSize;
            m_incrementalMaxScratch =
                std::max(m_incrementalMaxScratch, tlasSizes.buildScratchBufferSize);
            ensureScratchBuffer(m_incrementalInput.device, m_incrementalMaxScratch);

            m_tlas =
                [m_incrementalInput.device newAccelerationStructureWithSize:tlasSizes.accelerationStructureSize];
            if (!m_tlas) {
                m_buildStats.fallbackReason = "Failed to allocate TLAS";
                return fallbackAndComplete();
            }

            id<MTLCommandBuffer> tlasCommandBuffer = [queue commandBuffer];
            if (!tlasCommandBuffer) {
                m_buildStats.fallbackReason = "Failed to allocate Metal command buffer for TLAS build";
                return fallbackAndComplete();
            }
            tlasCommandBuffer.label = @"TLAS Build";
            id<MTLAccelerationStructureCommandEncoder> tlasEncoder =
                [tlasCommandBuffer accelerationStructureCommandEncoder];
            if (!tlasEncoder) {
                m_buildStats.fallbackReason =
                    "Failed to create acceleration structure encoder for TLAS build";
                return fallbackAndComplete();
            }
            [tlasEncoder useResource:m_instanceBuffer usage:MTLResourceUsageRead];
            [tlasEncoder useResource:m_tlas usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            [tlasEncoder buildAccelerationStructure:m_tlas
                                         descriptor:tlasDesc
                                      scratchBuffer:m_scratchBuffer
                                scratchBufferOffset:0];
            [tlasEncoder endEncoding];
            m_buildSignals->tlasFinished.store(false, std::memory_order_release);
            m_buildSignals->tlasSucceeded.store(false, std::memory_order_release);
            {
                // Capture the shared_ptr by value for the same use-after-free safety.
                auto signals = m_buildSignals;
                [tlasCommandBuffer addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                    const bool ok =
                        (cb.status == MTLCommandBufferStatusCompleted) && (cb.error == nil);
                    signals->tlasSucceeded.store(ok, std::memory_order_release);
                    signals->tlasFinished.store(true, std::memory_order_release);
                }];
            }
            [tlasCommandBuffer commit];
            m_incrementalStage = IncrementalStage::AwaitTLAS;
            outProgress.blasBatchesDone = static_cast<uint32_t>(m_incrementalBatchCount);
            outProgress.blasBatchesTotal = static_cast<uint32_t>(m_incrementalBatchCount);
            return true;
        }

        if (m_incrementalStage == IncrementalStage::AwaitTLAS) {
            const uint32_t meshCount = m_incrementalInput.meshCount;
            if (!m_buildSignals->tlasFinished.load(std::memory_order_acquire)) {
                outProgress.blasBatchesDone = static_cast<uint32_t>(m_incrementalBatchCount);
                outProgress.blasBatchesTotal = static_cast<uint32_t>(m_incrementalBatchCount);
                return true;
            }

            m_buildStats.tlasBuildMs = (CFAbsoluteTimeGetCurrent() - m_incrementalTlasStart) * 1000.0;
            if (!m_buildSignals->tlasSucceeded.load(std::memory_order_acquire)) {
                m_buildStats.fallbackReason = "Metal TLAS build did not complete successfully";
                return fallbackAndComplete();
            }

            m_instanceCount = meshCount;
            m_primitiveCount = m_incrementalTotalPrimitiveCount;
            m_buildStats.usedHardwareRaytracing = true;
            m_buildStats.primitiveCount = m_incrementalTotalPrimitiveCount;
            m_buildStats.instanceCount = meshCount;
            m_buildStats.blasCount = static_cast<uint32_t>(m_blasHandles.count);
            m_buildStats.blasMemoryBytes = m_incrementalTotalBlasBytes;
            m_buildStats.tlasMemoryBytes = m_incrementalTlasSizeBytes;
            m_buildStats.scratchMemoryBytes = static_cast<uint64_t>(m_incrementalMaxScratch);
            m_buildStats.blasScratchMemoryBytes = static_cast<uint64_t>(m_incrementalMaxBlasScratch);
            m_buildStats.tlasScratchMemoryBytes = static_cast<uint64_t>(m_incrementalTlasScratch);
            m_buildStats.retainedScratchMemoryBytes = 0u;
            HardwareBuildPlan completedPlan{};
            completedPlan.instanceToBlasIndex = m_incrementalInstanceToBlasIndex;
            completedPlan.instancePrimitiveCount = m_incrementalTotalPrimitiveCount;
            completedPlan.uniqueGeometryBytes = m_incrementalUniqueGeometryBytes;
            PopulateHardwarePolicyStats(m_buildStats,
                                        m_incrementalBuildPolicy,
                                        completedPlan,
                                        m_incrementalInput);

            uint64_t totalBlasAllocatedBytes = 0u;
            for (id<MTLAccelerationStructure> blasHandle in m_blasHandles) {
                if (blasHandle && [blasHandle respondsToSelector:@selector(allocatedSize)]) {
                    totalBlasAllocatedBytes += static_cast<uint64_t>(blasHandle.allocatedSize);
                }
            }
            uint64_t tlasAllocatedBytes = 0u;
            if (m_tlas && [m_tlas respondsToSelector:@selector(allocatedSize)]) {
                tlasAllocatedBytes = static_cast<uint64_t>(m_tlas.allocatedSize);
            }
            m_buildStats.blasAllocatedMemoryBytes = totalBlasAllocatedBytes;
            m_buildStats.tlasAllocatedMemoryBytes = tlasAllocatedBytes;

            outProvider.mode = PathTracerShaderTypes::IntersectionMode::HardwareRayTracing;
            outProvider.software = SoftwareBvhResources{};
            outProvider.hardware = HardwareRtResources{};
            outProvider.hardware.tlas = m_tlas;
            outProvider.hardware.instanceBuffer = m_instanceBuffer;
            outProvider.hardware.instanceUserIDBuffer = m_instanceUserIdBuffer;
            outProvider.hardware.scratchBuffer = nil;
            outProvider.hardware.instanceCount = m_instanceCount;
            outProvider.hardware.blasCount = static_cast<uint32_t>(m_blasHandles.count);
            outProvider.hardware.blas =
                m_blasHandles.count > 0 ? (MTLAccelerationStructureHandle)m_blasHandles[0] : nil;
            outProvider.hardware.blasHandles.clear();
            for (id<MTLAccelerationStructure> blasHandle in m_blasHandles) {
                outProvider.hardware.blasHandles.push_back((__bridge void*)blasHandle);
            }

#if PT_DEBUG_TOOLS || PT_MNEE_SWRT_RAYS || PT_MNEE_OCCLUSION_PARITY
            if (m_softwareFallback) {
                HardwareRtResources savedHardware = outProvider.hardware;
                PathTracerShaderTypes::IntersectionMode savedMode = outProvider.mode;
                m_softwareFallback->rebuild(m_incrementalInput, outProvider);
                outProvider.hardware = savedHardware;
                outProvider.mode = savedMode;
            }
#endif

            m_mode = PathTracerShaderTypes::IntersectionMode::HardwareRayTracing;
            m_incrementalStage = IncrementalStage::Idle;
            m_scratchBuffer = nil;
            outProgress.blasBatchesDone = static_cast<uint32_t>(m_incrementalBatchCount);
            outProgress.blasBatchesTotal = static_cast<uint32_t>(m_incrementalBatchCount);
            outProgress.complete = true;
            return true;
        }

        outProgress.complete = true;
        m_incrementalStage = IncrementalStage::Idle;
        return true;
    }

    bool incrementalRebuildActive() const override {
        return m_incrementalStage != IncrementalStage::Idle;
    }

    void clear() override {
        releaseHardwareResources();
        if (m_softwareFallback) {
            m_softwareFallback->clear();
        }
        m_mode = PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
        m_primitiveCount = 0;
        m_buildStats = SceneAccelBuildStats{};
        m_incrementalInput = SceneAccelBuildInput{};
        m_incrementalPrimitiveDescs = nil;
        m_incrementalBlasSizes.clear();
        m_incrementalStage = IncrementalStage::Idle;
        m_incrementalTotalBlasBytes = 0u;
        m_incrementalUniqueGeometryBytes = 0u;
        m_incrementalMaxScratch = 0u;
        m_incrementalMaxBlasScratch = 0u;
        m_incrementalTlasScratch = 0u;
        m_incrementalTotalPrimitiveCount = 0u;
        m_incrementalBlasSourceMeshIndices.clear();
        m_incrementalInstanceToBlasIndex.clear();
        m_incrementalCurrentBatch = 0u;
        m_incrementalBatchCount = 0u;
        m_incrementalBlasStart = 0.0;
        m_incrementalTlasStart = 0.0;
        m_incrementalTlasSizeBytes = 0u;
        m_buildSignals = std::make_shared<PendingBuildSignals>();
    }

    PathTracerShaderTypes::IntersectionMode mode() const override {
        return m_mode;
    }

    uint32_t primitiveCount() const override {
        return m_primitiveCount;
    }

    const SceneAccelBuildStats& buildStats() const override {
        return m_buildStats;
    }

private:
    enum class IncrementalStage : uint8_t {
        Idle = 0,
        SoftwareFallback,
        BuildBLASBatches,
        AwaitBLASBatch,
        BuildTLAS,
        AwaitTLAS,
    };

    bool buildHardware(const SceneAccelBuildInput& input,
                       IntersectionProvider& outProvider) {
        releaseHardwareResources();
        m_buildStats = SceneAccelBuildStats{};
        m_buildStats.usedHardwareRaytracing = true;

        id<MTLCommandQueue> queue = input.commandQueue ? input.commandQueue : m_commandQueue;
        if (!queue) {
            m_buildStats.fallbackReason = "Metal command queue unavailable for hardware ray tracing";
            return false;
        }

        const uint32_t meshCount = input.meshCount;
        if (meshCount == 0) {
            m_buildStats.fallbackReason = "No mesh instances available for hardware ray tracing";
            return false;
        }
        const HardwareBuildPlan plan = BuildHardwareBuildPlan(input);
        const SceneAccelBuildPolicy buildPolicy = SelectHardwareBuildPolicy(input, plan);
        const MTLAccelerationStructureUsage accelerationUsage =
            MetalUsageForBuildPolicy(buildPolicy);
        const uint32_t uniqueMeshCount = static_cast<uint32_t>(plan.uniqueMeshIndices.size());

        auto fail = [&](const char* message) -> bool {
            if (message) {
                NSLog(@"MetalRtAccel::buildHardware - %s", message);
                m_buildStats.fallbackReason = message;
            }
            releaseHardwareResources();
            return false;
        };

        std::vector<MTLPrimitiveAccelerationStructureDescriptor*> primitiveDescs;
        primitiveDescs.reserve(uniqueMeshCount);
        std::vector<MTLAccelerationStructureSizes> blasSizes;
        blasSizes.reserve(uniqueMeshCount);
        std::vector<uint32_t> blasSourceMeshIndices;
        blasSourceMeshIndices.reserve(uniqueMeshCount);
        uint64_t totalBlasBytes = 0;

        NSUInteger maxScratch = 0;
        NSUInteger maxBlasScratch = 0;
        for (uint32_t uniqueIndex = 0; uniqueIndex < uniqueMeshCount; ++uniqueIndex) {
            const uint32_t meshIndex = plan.uniqueMeshIndices[uniqueIndex];
            const SceneAccelMeshInput& mesh = input.meshes[meshIndex];

            if (!mesh.vertexBuffer || !mesh.indexBuffer || mesh.vertexCount == 0 || mesh.indexCount < 3) {
                return fail("Invalid mesh data for BLAS build");
            }

            const uint32_t triangleCount = mesh.indexCount / 3;
            if (triangleCount == 0) {
                return fail("Mesh contains no triangles");
            }

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
            primDesc.usage = accelerationUsage;
            primitiveDescs.push_back(primDesc);

            MTLAccelerationStructureSizes sizes = [input.device accelerationStructureSizesWithDescriptor:primDesc];
            maxBlasScratch = std::max(maxBlasScratch, sizes.buildScratchBufferSize);
            maxScratch = std::max(maxScratch, sizes.buildScratchBufferSize);
            blasSizes.push_back(sizes);
            totalBlasBytes += static_cast<uint64_t>(sizes.accelerationStructureSize);
            blasSourceMeshIndices.push_back(meshIndex);
        }

        if (plan.instancePrimitiveCount == 0) {
            return fail("No triangles found while building BLAS (totalPrimitiveCount == 0)");
        }

        NSLog(@"MetalRtAccel::buildHardware - starting BLAS build for %u triangles across %u mesh instances",
              plan.instancePrimitiveCount,
              meshCount);
        NSLog(@"MetalRtAccel::buildHardware - scratch requirements BLASmax=%.2f MiB",
              static_cast<double>(maxBlasScratch) / (1024.0 * 1024.0));

        if (primitiveDescs.empty()) {
            return fail("No primitive descriptors created");
        }

        ensureScratchBuffer(input.device, maxScratch);

        NSMutableArray<id<MTLAccelerationStructure>>* blasHandles =
            [NSMutableArray arrayWithCapacity:uniqueMeshCount];
        const CFAbsoluteTime blasStart = CFAbsoluteTimeGetCurrent();

        for (NSUInteger i = 0; i < primitiveDescs.size(); ++i) {
            MTLAccelerationStructureSizes sizes = blasSizes[i];
            id<MTLAccelerationStructure> blas =
                [input.device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
            if (!blas) {
                return fail("Failed to allocate BLAS acceleration structure");
            }
            [blasHandles addObject:blas];
        }

        constexpr NSUInteger kBlasBatchSize = 32u;
        const NSUInteger blasBatchCount =
            (static_cast<NSUInteger>(primitiveDescs.size()) + kBlasBatchSize - 1u) / kBlasBatchSize;
        for (NSUInteger batchIndex = 0; batchIndex < blasBatchCount; ++batchIndex) {
            const NSUInteger batchStart = batchIndex * kBlasBatchSize;
            const NSUInteger batchEnd = std::min<NSUInteger>(primitiveDescs.size(), batchStart + kBlasBatchSize);

            id<MTLCommandBuffer> batchCommandBuffer = [queue commandBuffer];
            if (!batchCommandBuffer) {
                return fail("Failed to allocate Metal command buffer for BLAS batch");
            }
            batchCommandBuffer.label = [NSString stringWithFormat:@"BLAS Batch %lu/%lu",
                                        static_cast<unsigned long>(batchIndex + 1u),
                                        static_cast<unsigned long>(blasBatchCount)];
            id<MTLAccelerationStructureCommandEncoder> batchEncoder =
                [batchCommandBuffer accelerationStructureCommandEncoder];
            if (!batchEncoder) {
                [batchCommandBuffer commit];
                WaitForCommandBufferResponsive(batchCommandBuffer);
                return fail("Failed to create acceleration structure encoder for BLAS batch");
            }

            for (NSUInteger i = batchStart; i < batchEnd; ++i) {
                const uint32_t sourceMeshIndex = blasSourceMeshIndices[i];
                [batchEncoder useResource:input.meshes[sourceMeshIndex].vertexBuffer usage:MTLResourceUsageRead];
                [batchEncoder useResource:input.meshes[sourceMeshIndex].indexBuffer usage:MTLResourceUsageRead];
                [batchEncoder useResource:(id<MTLAccelerationStructure>)blasHandles[i]
                                    usage:MTLResourceUsageRead | MTLResourceUsageWrite];
                [batchEncoder buildAccelerationStructure:(id<MTLAccelerationStructure>)blasHandles[i]
                                              descriptor:primitiveDescs[i]
                                           scratchBuffer:m_scratchBuffer
                                     scratchBufferOffset:0];
            }

            [batchEncoder endEncoding];
            [batchCommandBuffer commit];
            WaitForCommandBufferResponsive(batchCommandBuffer);
            if (batchCommandBuffer.status != MTLCommandBufferStatusCompleted || batchCommandBuffer.error) {
                m_buildStats.fallbackReason = "Metal BLAS batch build did not complete successfully";
                releaseHardwareResources();
                return false;
            }
            PumpMainThreadEventsBriefly();
        }
        m_buildStats.blasBuildMs = (CFAbsoluteTimeGetCurrent() - blasStart) * 1000.0;

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
            descriptor.accelerationStructureIndex = plan.instanceToBlasIndex[i];
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
        tlasDesc.usage = accelerationUsage;

        const CFAbsoluteTime tlasStart = CFAbsoluteTimeGetCurrent();
        MTLAccelerationStructureSizes tlasSizes =
            [input.device accelerationStructureSizesWithDescriptor:tlasDesc];
        const NSUInteger tlasScratch = tlasSizes.buildScratchBufferSize;
        maxScratch = std::max(maxScratch, tlasSizes.buildScratchBufferSize);
        ensureScratchBuffer(input.device, maxScratch);
        NSLog(@"MetalRtAccel::buildHardware - scratch requirements TLAS=%.2f MiB peakAllocated=%.2f MiB",
              static_cast<double>(tlasScratch) / (1024.0 * 1024.0),
              static_cast<double>(maxScratch) / (1024.0 * 1024.0));

        m_tlas = [input.device newAccelerationStructureWithSize:tlasSizes.accelerationStructureSize];
        if (!m_tlas) {
            return fail("Failed to allocate TLAS");
        }

        id<MTLCommandBuffer> tlasCommandBuffer = [queue commandBuffer];
        if (!tlasCommandBuffer) {
            return fail("Failed to allocate Metal command buffer for TLAS build");
        }
        tlasCommandBuffer.label = @"TLAS Build";
        id<MTLAccelerationStructureCommandEncoder> tlasEncoder =
            [tlasCommandBuffer accelerationStructureCommandEncoder];
        if (!tlasEncoder) {
            [tlasCommandBuffer commit];
            WaitForCommandBufferResponsive(tlasCommandBuffer);
            return fail("Failed to create acceleration structure encoder for TLAS build");
        }
        [tlasEncoder useResource:m_instanceBuffer usage:MTLResourceUsageRead];
        [tlasEncoder useResource:m_tlas usage:MTLResourceUsageRead | MTLResourceUsageWrite];
        [tlasEncoder buildAccelerationStructure:m_tlas
                                     descriptor:tlasDesc
                                  scratchBuffer:m_scratchBuffer
                            scratchBufferOffset:0];
        [tlasEncoder endEncoding];
        [tlasCommandBuffer commit];
        WaitForCommandBufferResponsive(tlasCommandBuffer);
        m_buildStats.tlasBuildMs = (CFAbsoluteTimeGetCurrent() - tlasStart) * 1000.0;

        m_instanceCount = meshCount;

        NSLog(@"MetalRtAccel::buildHardware - submitted TLAS build (%u instances, %u total primitives)",
              meshCount,
              plan.instancePrimitiveCount);

        if (tlasCommandBuffer.status != MTLCommandBufferStatusCompleted || tlasCommandBuffer.error) {
            m_buildStats.fallbackReason = "Metal TLAS build did not complete successfully";
            releaseHardwareResources();
            return false;
        }

        m_blasHandles = [blasHandles mutableCopy];
        m_primitiveCount = plan.instancePrimitiveCount;
        m_buildStats.primitiveCount = plan.instancePrimitiveCount;
        m_buildStats.instanceCount = meshCount;
        m_buildStats.blasCount = static_cast<uint32_t>(blasHandles.count);
        m_buildStats.blasMemoryBytes = totalBlasBytes;
        m_buildStats.tlasMemoryBytes = static_cast<uint64_t>(tlasSizes.accelerationStructureSize);
        m_buildStats.scratchMemoryBytes = static_cast<uint64_t>(maxScratch);
        m_buildStats.blasScratchMemoryBytes = static_cast<uint64_t>(maxBlasScratch);
        m_buildStats.tlasScratchMemoryBytes = static_cast<uint64_t>(tlasScratch);
        m_buildStats.retainedScratchMemoryBytes = 0u;
        PopulateHardwarePolicyStats(m_buildStats, buildPolicy, plan, input);

        uint64_t totalBlasAllocatedBytes = 0u;
        for (id<MTLAccelerationStructure> blasHandle in blasHandles) {
            if (blasHandle && [blasHandle respondsToSelector:@selector(allocatedSize)]) {
                totalBlasAllocatedBytes += static_cast<uint64_t>(blasHandle.allocatedSize);
            }
        }
        uint64_t tlasAllocatedBytes = 0u;
        if (m_tlas && [m_tlas respondsToSelector:@selector(allocatedSize)]) {
            tlasAllocatedBytes = static_cast<uint64_t>(m_tlas.allocatedSize);
        }
        m_buildStats.blasAllocatedMemoryBytes = totalBlasAllocatedBytes;
        m_buildStats.tlasAllocatedMemoryBytes = tlasAllocatedBytes;
        NSLog(@"[AS] total BLAS allocated: %.2f MiB across %u structures",
              static_cast<double>(totalBlasAllocatedBytes) / (1024.0 * 1024.0),
              static_cast<unsigned>(blasHandles.count));
        NSLog(@"[AS] TLAS allocated: %.2f MiB",
              static_cast<double>(tlasAllocatedBytes) / (1024.0 * 1024.0));

        outProvider.mode = PathTracerShaderTypes::IntersectionMode::HardwareRayTracing;
        outProvider.software = SoftwareBvhResources{};
        outProvider.hardware = HardwareRtResources{};
        outProvider.hardware.tlas = m_tlas;
        outProvider.hardware.instanceBuffer = m_instanceBuffer;
        outProvider.hardware.instanceUserIDBuffer = m_instanceUserIdBuffer;
        outProvider.hardware.scratchBuffer = nil;
        outProvider.hardware.instanceCount = m_instanceCount;
        outProvider.hardware.blasCount = static_cast<uint32_t>(m_blasHandles.count);
        outProvider.hardware.blas = m_blasHandles.count > 0 ? (MTLAccelerationStructureHandle)m_blasHandles[0] : nil;
        // Added: record all BLAS handles for encoder resource usage later.
        outProvider.hardware.blasHandles.clear();
        for (id<MTLAccelerationStructure> blasHandle in m_blasHandles) {
            outProvider.hardware.blasHandles.push_back((__bridge void*)blasHandle);
        }

        if (m_scratchBuffer) {
            NSLog(@"MetalRtAccel::buildHardware - releasing scratch buffer after build (%.2f MiB)",
                  static_cast<double>(m_scratchBuffer.length) / (1024.0 * 1024.0));
        }
        m_scratchBuffer = nil;

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
        if (m_buildStats.fallbackReason.empty()) {
            if (!m_canUseHardware) {
                m_buildStats.fallbackReason = "Hardware ray tracing unsupported or disabled";
            } else if (!input.device) {
                m_buildStats.fallbackReason = "Metal device unavailable";
            } else if (!input.meshes || input.meshCount == 0) {
                m_buildStats.fallbackReason = "Scene has no mesh instances for hardware ray tracing";
            } else {
                m_buildStats.fallbackReason = "Hardware acceleration build failed";
            }
        }
        releaseHardwareResources();
        outProvider.hardware = HardwareRtResources{};
        if (!m_softwareFallback) {
            m_softwareFallback = std::make_unique<SoftwareBvhAccel>();
        }
        m_softwareFallback->rebuild(input, outProvider);
        m_mode = PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
        m_primitiveCount = m_softwareFallback ? m_softwareFallback->primitiveCount() : 0;
        if (m_softwareFallback) {
            const std::string fallbackReason = m_buildStats.fallbackReason;
            SceneAccelBuildStats softwareStats = m_softwareFallback->buildStats();
            softwareStats.deviceSupportsHardwareRaytracing = m_canUseHardware;
            softwareStats.fallbackReason = fallbackReason;
            m_buildStats = std::move(softwareStats);
        }
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
    SceneAccelBuildInput m_incrementalInput{};
    NSMutableArray<MTLPrimitiveAccelerationStructureDescriptor*>* __strong m_incrementalPrimitiveDescs = nil;
    std::vector<MTLAccelerationStructureSizes> m_incrementalBlasSizes;
    std::vector<uint32_t> m_incrementalBlasSourceMeshIndices;
    std::vector<uint32_t> m_incrementalInstanceToBlasIndex;
    IncrementalStage m_incrementalStage = IncrementalStage::Idle;
    uint64_t m_incrementalTotalBlasBytes = 0u;
    uint64_t m_incrementalUniqueGeometryBytes = 0u;
    NSUInteger m_incrementalMaxScratch = 0u;
    NSUInteger m_incrementalMaxBlasScratch = 0u;
    NSUInteger m_incrementalTlasScratch = 0u;
    uint32_t m_incrementalTotalPrimitiveCount = 0u;
    SceneAccelBuildPolicy m_incrementalBuildPolicy = SceneAccelBuildPolicy::SoftwareReference;
    NSUInteger m_incrementalCurrentBatch = 0u;
    NSUInteger m_incrementalBatchCount = 0u;
    CFAbsoluteTime m_incrementalBlasStart = 0.0;
    CFAbsoluteTime m_incrementalTlasStart = 0.0;
    uint64_t m_incrementalTlasSizeBytes = 0u;
    std::shared_ptr<PendingBuildSignals> m_buildSignals = std::make_shared<PendingBuildSignals>();

    NSMutableArray<id<MTLAccelerationStructure>>* __strong m_blasHandles = nil;
    id<MTLAccelerationStructure> __strong m_tlas = nil;
    id<MTLBuffer> __strong m_instanceBuffer = nil;
    id<MTLBuffer> __strong m_instanceUserIdBuffer = nil;
    id<MTLBuffer> __strong m_scratchBuffer = nil;

    PathTracerShaderTypes::IntersectionMode m_mode =
        PathTracerShaderTypes::IntersectionMode::SoftwareBVH;
    uint32_t m_primitiveCount = 0;
    uint32_t m_instanceCount = 0;
    SceneAccelBuildStats m_buildStats{};
};

std::unique_ptr<SceneAccel> CreateSceneAccel(const SceneAccelConfig& config) {
    if (config.hardwareRaytracingSupported && config.commandQueue) {
        return std::make_unique<MetalRtAccel>(config);
    }
    return std::make_unique<SoftwareBvhAccel>();
}

}  // namespace PathTracer
