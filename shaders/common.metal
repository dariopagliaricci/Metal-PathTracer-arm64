#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

#ifndef ENABLE_MNEE_CAUSTICS
#define ENABLE_MNEE_CAUSTICS 1
#endif
#ifndef ENABLE_MNEE
#define ENABLE_MNEE ENABLE_MNEE_CAUSTICS
#endif

constant uint kMaxSpheres = 512;
constant uint kMaxMaterials = 512;
constant uint kMaxRectangles = 128;
constant uint kMaxMaterialTextures = 64;
constant uint kMaxMaterialSamplers = 14;
constant uint kMaterialFlagDisableOrm = 1u << 0;

struct SphereData {
    float4 centerRadius; // xyz center, w radius
    uint4 materialIndex; // x material index
};

struct RectData {
    float4 corner;          // xyz corner
    float4 edgeU;           // xyz edge vector, w inverse length squared
    float4 edgeV;           // xyz edge vector, w inverse length squared
    float4 normalAndPlane;  // xyz normal, w plane constant
    uint4 materialTwoSided; // x material index, y two-sided
};

struct MaterialData {
    float4 baseColorRoughness;  // xyz base color/F0 tint, w roughness
    float4 typeEta;             // x type, y base IOR, z coat IOR / PBR double-sided, w thin dielectric flag / PBR thickness
    float4 emission;            // xyz emission radiance, w env-sampled flag
    float4 conductorEta;        // xyz conductor eta, w flag (>0 when valid)
    float4 conductorK;          // xyz conductor k, w flag (>0 when valid)
    float4 coatParams;          // x coat roughness, y coat thickness, z coat sample weight, w coat Fresnel average
    float4 coatTint;            // xyz coat tint, w unused
    float4 coatAbsorption;      // xyz coat absorption coefficient, w unused
    float4 dielectricSigmaA;    // xyz glass absorption coefficient, w unused
    float4 sssSigmaA;           // xyz sigma_a, w override flag
    float4 sssSigmaS;           // xyz sigma_s, w anisotropy g
    float4 sssParams;           // x mean free path, y method, z coat enabled flag, w unused
    float4 carpaintBaseParams;  // x base metallic, y base roughness, z flake scale, w flake reflectance scale
    float4 carpaintFlakeParams; // x flake sample weight, y flake roughness, z flake anisotropy, w flake normal strength
    float4 carpaintBaseEta;     // xyz carpaint base eta, w flag (>0 when valid)
    float4 carpaintBaseK;       // xyz carpaint base k, w flag (>0 when valid)
    float4 carpaintBaseTint;    // xyz conductor tint multiplier, w unused
    uint4 textureIndices0;      // x = baseColor, y = metallicRoughness, z = normal, w = occlusion
    uint4 textureIndices1;      // x = emissive, yzw unused
    uint materialFlags;         // bitfield (kMaterialFlag*)
    uint materialPad0;
    uint materialPad1;
    uint materialPad2;
    float4 pbrParams;           // x = metallic, y = roughness, z = occlusion strength, w = normal scale
    float4 pbrExtras;           // x = alpha factor, y = alpha cutoff, z = transmission factor, w = alpha mode
    uint4 textureUvSet0;        // x = baseColor, y = metallicRoughness, z = normal, w = occlusion
    uint4 textureUvSet1;        // x = emissive, y = transmission, zw unused
    float4 textureTransform0;   // baseColor: row0 = (m00, m01, m02)
    float4 textureTransform1;   // baseColor: row1 = (m10, m11, m12)
    float4 textureTransform2;   // metallicRoughness: row0
    float4 textureTransform3;   // metallicRoughness: row1
    float4 textureTransform4;   // normal: row0
    float4 textureTransform5;   // normal: row1
    float4 textureTransform6;   // occlusion: row0
    float4 textureTransform7;   // occlusion: row1
    float4 textureTransform8;   // emissive: row0
    float4 textureTransform9;   // emissive: row1
    float4 textureTransform10;  // transmission: row0
    float4 textureTransform11;  // transmission: row1
};

struct EnvironmentAliasEntry {
    float threshold;
    uint alias;
    uint padding0;
    uint padding1;
};

struct PathtraceUniforms {
    uint width;
    uint height;
    uint frameIndex;
    uint sampleCount;

    float3 cameraOrigin;
    float cameraPad0;
    float3 lowerLeftCorner;
    float cameraPad1;
    float3 horizontal;
    float cameraPad2;
    float3 vertical;
    float cameraPad3;
    float3 cameraU;
    float lensRadius;
    float3 cameraV;
    float cameraPad4;

    uint sphereCount;
    uint rectangleCount;
    uint materialCount;
    uint materialTextureCount;
    uint maxDepth;
    uint useRussianRoulette;
    uint intersectionMode;
    uint softwareBvhType;
    uint primitiveCount;
    uint meshCount;
    uint triangleCount;
    uint fixedRngSeed;  // If non-zero, use as deterministic RNG seed
    uint backgroundMode;
    uint workingColorSpace;
    float environmentRotation;
    float environmentIntensity;
    float padding0;
    float3 backgroundColor;
    uint environmentAliasCount;
    uint environmentMapWidth;
    uint environmentMapHeight;
    uint environmentHasDistribution;
    uint fireflyClampEnabled;
    float fireflyClampFactor;
    float fireflyClampFloor;
    float throughputClamp;
    float specularTailClampBase;
    float specularTailClampRoughnessScale;
    float minSpecularPdf;
    float fireflyClampMaxContribution;
    float paddingFirefly;
    uint sssMode;
    uint sssMaxSteps;
    uint paddingSss0;
    uint enableSpecularNee;
    uint enableMnee;
    uint enableMneeSecondary;
    uint debugPathActive;
    uint debugPixelX;
    uint debugPixelY;
    uint debugMaxEntries;
    uint hardwareExcludeMaxAttempts;
    float hardwareExitNormalBias;
    float hardwareExitDirectionalBias;
    uint enableHardwareMissFallback;
    uint enableHardwareFirstHitFromSoftware;
    uint enableHardwareForceSoftware;
    uint paddingHardware1;
    uint parityAssertEnabled;
    uint parityAssertMode;
    uint parityPixelX;
    uint parityPixelY;
    uint parityOncePerFrame;
    uint parityPadding0;
    uint parityPadding1;
    uint parityPadding2;
    uint forcePureHWRTForGlass;
    uint parityPadding3;
    uint parityPadding4;
    uint parityPadding5;
    uint debugViewMode;
    uint debugDisableAO;
    uint debugAoIndirectOnly;
    uint debugDisableNormalMap;
    uint debugDisableOrmTexture;
    uint debugFlipNormalGreen;
    uint debugSpecularOnly;
    float debugNormalStrengthScale;
    float debugNormalLodBias;
    float debugOrmLodBias;
    float debugEnvMipOverride;
    uint debugEnableVisorOverride;
    int debugVisorOverrideMaterialId;
    float debugVisorOverrideRoughness;
    float debugVisorOverrideF0;
    uint debugEnvNearest;
    uint debugPadding0;
};

struct DisplayUniforms {
    uint tonemapMode;
    uint acesVariant;
    float exposure;
    float reinhardWhite;
    uint bloomEnabled;
    float bloomThreshold;
    float bloomIntensity;
    float bloomRadius;
};

struct DisplayVertexOut {
    float4 position [[position]];
    float2 uv;
};

struct Ray {
    float3 origin;
    float3 direction;
};

struct BvhNode {
    float4 boundsMin;
    float4 boundsMax;
    uint leftChild;
    uint rightChild;
    uint primitiveOffset;
    uint primitiveCount;
    uint padding0;
    uint padding1;
};

constant uint kIntersectionModeSoftwareBVH = 0u;
constant uint kIntersectionModeHardwareRT = 1u;

constant uint kSoftwareBvhNone = 0u;
constant uint kSoftwareBvhSpheres = 1u;
constant uint kSoftwareBvhTriangles = 2u;
constant uint kPathtraceDebugMaxEntries = 512u;
constant uint kPathtraceParityMaxEntries = 16u;

struct PathtraceStats {
    atomic_uint primaryRayCount;
    atomic_uint nodesVisited;
    atomic_uint leafPrimTests;
    atomic_uint internalNodeVisits;
    atomic_uint internalBothVisited;
    atomic_uint shadowRayCount;
    atomic_uint shadowRayEarlyExitCount;
    atomic_uint hardwareRayCount;
    atomic_uint hardwareHitCount;
    atomic_uint hardwareMissCount;
    // Extended diagnostics mirroring CPU-side PathtraceStats
    atomic_uint hardwareResultNoneCount;
    atomic_uint hardwareRejectedCount;
    atomic_uint hardwareUnavailableCount;
    atomic_uint hardwareLastResultType;
    atomic_uint hardwareLastInstanceId;
    atomic_uint hardwareLastPrimitiveId;
    atomic_uint hardwareLastDistanceBits;
    atomic_uint specularNeeOcclusionHitCount;
    atomic_uint specNeeEnvAddedCount;
    atomic_uint specNeeRectAddedCount;
    atomic_uint mneeEnvHwOccludedCount;
    atomic_uint mneeEnvSwOccludedCount;
    atomic_uint mneeEnvHwSwMismatchCount;
    atomic_uint mneeRectHwOccludedCount;
    atomic_uint mneeRectSwOccludedCount;
    atomic_uint mneeRectHwSwMismatchCount;
    atomic_uint mneeHitHwSwHitMissCount;
    atomic_uint mneeHitHwSwNormalMismatchCount;
    atomic_uint mneeHitHwSwIdMismatchCount;
    atomic_uint mneeHitHwSwTDiffCount;
    atomic_uint mneeChainHwSwHitMissCount;
    atomic_uint mneeChainHwSwNormalMismatchCount;
    atomic_uint mneeChainHwSwIdMismatchCount;
    atomic_uint mneeChainHwSwTDiffCount;
    atomic_uint mneeEligibleCount;
    atomic_uint mneeEnvAttemptCount;
    atomic_uint mneeEnvAddedCount;
    atomic_uint mneeRectAttemptCount;
    atomic_uint mneeRectAddedCount;
    atomic_uint mneeContributionCount;
    atomic_uint mneeContributionLumaSumLo;
    atomic_uint mneeContributionLumaSumHi;
    atomic_uint hardwareSelfHitRejectedCount;
    atomic_uint hardwareMissDistanceBins[32];
    atomic_uint hardwareMissLastDistanceBits;
    atomic_uint hardwareSelfHitLastDistanceBits;
    atomic_uint hardwareExcludeRetryHistogram[4];
    atomic_uint hardwareMissLastInstanceId;
    atomic_uint hardwareMissLastPrimitiveId;
    atomic_uint hardwareFallbackHitCount;
    atomic_uint hardwareFirstHitFallbackCount;
};

struct PathtraceDebugEntry {
    uint integrator;
    uint pixelX;
    uint pixelY;
    uint sampleIndex;
    uint depth;
    uint mediumDepthBefore;
    uint mediumDepthAfter;
    int mediumEvent;
    uint frontFace;
    uint scatterIsDelta;
    uint materialIndex;
    uint reserved0;
    float4 throughput;
};

struct PathtraceParityEntry {
    uint frameIndex;
    uint pixelX;
    uint pixelY;
    uint depth;
    uint reasonMask;
    uint hwHit;
    uint swHit;
    uint hwFrontFace;
    uint swFrontFace;
    uint hwMaterialIndex;
    uint swMaterialIndex;
    uint hwMeshIndex;
    uint swMeshIndex;
    uint hwPrimitiveIndex;
    uint swPrimitiveIndex;
    float hwT;
    float swT;
    float tMin;
    float tMax;
    float4 rayOrigin;
    float4 rayDirection;
    float4 hwNormal;
    float4 swNormal;
};

struct PathtraceDebugBuffer {
    atomic_uint writeIndex;
    uint maxEntries;
    uint reserved0;
    uint reserved1;
    PathtraceDebugEntry entries[kPathtraceDebugMaxEntries];
    atomic_uint parityWriteIndex;
    uint parityMaxEntries;
    uint parityReserved0;
    uint parityReserved1;
    PathtraceParityEntry parityEntries[kPathtraceParityMaxEntries];
    atomic_uint parityChecksPerformed;
    atomic_uint parityChecksInMedium;
    uint parityCountsReserved0;
    uint parityCountsReserved1;
};

struct MaterialTextureInfo {
    uint width;
    uint height;
    uint mipCount;
    uint flags;
};

struct RayHit {
    float3 normal;
    float t;
    uint primitiveIndex;
    uint materialIndex;
    uint frontFace;
    uint hit;
    uint padding0;
    uint padding1;
};

constant uint kPrimitiveTypeNone = 0u;
constant uint kPrimitiveTypeSphere = 1u;
constant uint kPrimitiveTypeRectangle = 2u;
constant uint kPrimitiveTypeTriangle = 3u;

struct MeshInfo {
    uint materialIndex;
    uint triangleOffset;
    uint triangleCount;
    uint vertexOffset;
    uint vertexCount;
    uint indexOffset;
    uint indexCount;
    float4x4 localToWorld;
    float4x4 worldToLocal;
};

struct SceneVertex {
    float4 position;
    float4 normal;
    float4 tangent;
    float4 uv;
};

struct TriangleData {
    float4 v0;
    float4 v1;
    float4 v2;
    uint4 metadata; // x material index
};

struct SoftwareInstanceInfo {
    uint blasRootNodeOffset;
    uint blasNodeCount;
    uint blasPrimIndexOffset;
    uint blasPrimIndexCount;
    uint triangleBaseOffset;
    uint triangleCount;
    uint padding0;
    uint padding1;
    float4x4 localToWorld;
    float4x4 worldToLocal;
};

inline float3 ray_at(const Ray ray, float t) {
    return ray.origin + t * ray.direction;
}
