#pragma once

#include "renderer/RenderSettings.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace PathTracer {

enum class RenderPassType : uint32_t {
    Host = 0,
    Compute = 1,
    Render = 2,
    External = 3,
};

enum class RenderResourceKind : uint32_t {
    Buffer = 0,
    Texture = 1,
    AccelerationStructure = 2,
    External = 3,
};

enum class RenderResourceGroup : uint32_t {
    FrameConstants = 0,
    SceneBuffers = 1,
    LightBuffers = 2,
    GBufferResources = 3,
    ReSTIRDIResources = 4,
    ReGIRResources = 5,
    ReSTIRGIResources = 6,
    ReSTIRPTResources = 7,
    WavefrontResources = 8,
    DenoiserResources = 9,
    FinalOutputResources = 10,
    DebugResources = 11,
    DenoiserScratchResources = 12,
    ReSTIRDIReservoirCurrent = 13,
    ReSTIRDIReservoirPrevious = 14,
    ReSTIRDIReservoirTemporal = 15,
    ReSTIRDIReservoirSpatial = 16,
    ReSTIRDIVisibility = 17,
    ReSTIRDIDebugAOVs = 18,
    ReSTIRDINeighborPattern = 19,
    PathGuidingResources = 20,
    RadianceCacheResources = 21,
    ReconstructionAOVResources = 22,
    ReconstructionHistoryResources = 23,
    CausticTransportResources = 24,
    VolumeTransportResources = 25,
    MLBackendResources = 26,
};

enum class ResourceLifetime : uint32_t {
    Imported = 0,
    Transient = 1,
    Persistent = 2,
    History = 3,
    DebugReadback = 4,
};

enum class ResourceAccess : uint32_t {
    Read = 0,
    Write = 1,
    ReadWrite = 2,
    Clear = 3,
    Preserve = 4,
};

enum class StorageClass : uint32_t {
    Unknown = 0,
    Private = 1,
    Shared = 2,
    Managed = 3,
    Memoryless = 4,
    External = 5,
};

enum class PixelFormatClass : uint32_t {
    Unknown = 0,
    StructuredBuffer = 1,
    R32Uint = 2,
    RGBA16Float = 3,
    RGBA32Float = 4,
};

enum class HistoryPolicy : uint32_t {
    None = 0,
    Persistent = 1,
    CurrentFrame = 2,
    CurrentPrevious = 3,
    DebugReadback = 4,
};

enum class RenderPassFeature : uint32_t {
    None = 0,
    Optional = 1u << 0u,
    UsesHistory = 1u << 1u,
    ProducesHistory = 1u << 2u,
    DebugOnly = 1u << 3u,
    Wavefront = 1u << 4u,
    Denoise = 1u << 5u,
    Reconstruction = 1u << 6u,
    MLBackend = 1u << 7u,
};

enum class RenderPassExecutionId : uint32_t {
    NoOp = 0,
    UpdateAccelerationStructures = 1,
    ClearAccumulation = 2,
    MegakernelIntegrate = 3,
    WavefrontWorkGraph = 4,
    WavefrontStageMarker = 5,
    Present = 6,
    SvgfDenoise = 7,
    OidnDenoise = 8,
    Display = 9,
    RestirDiInitializeReservoirs = 10,
    RestirDiInitialCandidates = 11,
    RestirDiTemporalReuse = 12,
    RestirDiSpatialReuse = 13,
    RestirDiVisibility = 14,
    RestirDiApplyLighting = 15,
    RestirDiDebugAov = 16,
    MlBackendNoOp = 17,
};

enum class DispatchKind : uint32_t {
    None = 0,
    FullScreen = 1,
    Linear = 2,
    Indirect = 3,
    External = 4,
};

enum class DebugCapturePolicy : uint32_t {
    None = 0,
    LabelOnly = 1,
    Scoped = 2,
};

enum class TimingPolicy : uint32_t {
    None = 0,
    Host = 1,
    Encoder = 2,
    BucketedEncoder = 3,
    External = 4,
};

enum class HistoryResetReason : uint32_t {
    None = 0,
    ResolutionChanged = 1,
    SceneReloaded = 2,
    CameraCut = 3,
    LightTableChanged = 4,
    FeatureToggleChanged = 5,
    AccelerationStructureRebuilt = 6,
    EmissiveTableChanged = 7,
    MaterialTableChanged = 8,
    FrameAgeExpired = 9,
    ManualDebugReset = 10,
};

struct ResourceUse {
    RenderResourceGroup group = RenderResourceGroup::FrameConstants;
    std::string name;
    ResourceAccess access = ResourceAccess::Read;
};

struct FrameResourceDesc {
    std::string name;
    RenderResourceGroup group = RenderResourceGroup::FrameConstants;
    RenderResourceKind kind = RenderResourceKind::Buffer;
    ResourceLifetime lifetime = ResourceLifetime::Imported;
    StorageClass storageClass = StorageClass::Unknown;
    StorageClass targetStorageClass = StorageClass::Unknown;
    uint64_t sizeBytes = 0;
    std::string sizeFormula;
    PixelFormatClass format = PixelFormatClass::Unknown;
    HistoryPolicy historyPolicy = HistoryPolicy::None;
    std::vector<HistoryResetReason> invalidatesOn;
    std::string debugLabel;
    std::string currentOwner;
    std::string targetOwner;
    std::string featureGate;
    std::string debugReadbackPolicy;
    std::string migrationRisk;
    bool hotPath = false;
    bool temporaryShared = false;
};

struct DispatchDesc {
    DispatchKind kind = DispatchKind::None;
    std::string dimensions;
    std::string activeWork;
};

struct RenderPassDesc {
    std::string name;
    RenderPassType type = RenderPassType::Compute;
    uint32_t featureMask = 0;
    std::vector<RenderResourceGroup> inputs;
    std::vector<RenderResourceGroup> outputs;
    std::vector<ResourceUse> reads;
    std::vector<ResourceUse> writes;
    std::vector<std::string> settingsDependencies;
    std::vector<RenderResourceGroup> historyDependencies;
    std::string pipelineKey;
    DispatchDesc dispatch;
    std::string debugLabel;
    std::string timingLabel;
    DebugCapturePolicy debugPolicy = DebugCapturePolicy::LabelOnly;
    TimingPolicy timingPolicy = TimingPolicy::None;
    RenderPassExecutionId executionId = RenderPassExecutionId::NoOp;
};

struct RenderPassGraphDesc {
    uint32_t version = 2;
    std::vector<FrameResourceDesc> resources;
    std::vector<RenderPassDesc> passes;
};

struct FrameResourceLifetimePlan {
    std::string name;
    RenderResourceGroup group = RenderResourceGroup::FrameConstants;
    ResourceLifetime lifetime = ResourceLifetime::Imported;
    int32_t firstPassIndex = -1;
    int32_t lastPassIndex = -1;
    int32_t aliasPoolIndex = -1;
    bool used = false;
    bool aliasable = false;
};

struct FrameResourceTransitionPlan {
    uint32_t passIndex = 0;
    std::string passName;
    std::string resourceName;
    RenderResourceGroup group = RenderResourceGroup::FrameConstants;
    ResourceAccess previousAccess = ResourceAccess::Preserve;
    ResourceAccess access = ResourceAccess::Read;
    bool firstUse = true;
    bool requiresBarrier = false;
    bool writesResource = false;
};

struct RenderPassGraphCompileResult {
    bool valid = true;
    std::vector<uint32_t> passOrder;
    uint32_t executablePassCount = 0;
    uint32_t skippedPassCount = 0;
    uint32_t usedResourceCount = 0;
    uint32_t aliasableTransientResourceCount = 0;
    uint32_t transientAliasPoolCount = 0;
    uint32_t resourceTransitionCount = 0;
    uint32_t requiredBarrierCount = 0;
    uint32_t writeTransitionCount = 0;
    std::vector<FrameResourceLifetimePlan> resourcePlans;
    std::vector<FrameResourceTransitionPlan> resourceTransitions;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
};

struct RenderPassExecutionResult {
    bool compiled = false;
    uint32_t compiledExecutablePassCount = 0;
    uint32_t compiledSkippedPassCount = 0;
    uint32_t compiledUsedResourceCount = 0;
    uint32_t compiledAliasableTransientResourceCount = 0;
    uint32_t compiledTransientAliasPoolCount = 0;
    uint32_t compiledResourceTransitionCount = 0;
    uint32_t compiledRequiredBarrierCount = 0;
    uint32_t compiledWriteTransitionCount = 0;
    std::vector<FrameResourceLifetimePlan> resourcePlans;
    std::vector<FrameResourceTransitionPlan> resourceTransitions;
    std::vector<std::string> compileWarnings;
    std::vector<std::string> compileErrors;
    std::vector<std::string> executedPasses;
    std::vector<std::string> skippedPasses;
    std::vector<std::string> failedPasses;
};

const char* RenderPassTypeLabel(RenderPassType type);
const char* RenderResourceGroupLabel(RenderResourceGroup group);
const char* ResourceLifetimeLabel(ResourceLifetime lifetime);
const char* StorageClassLabel(StorageClass storageClass);
const char* ResourceAccessLabel(ResourceAccess access);
const char* RenderPassExecutionLabel(RenderPassExecutionId executionId);
const char* DebugCapturePolicyLabel(DebugCapturePolicy policy);
const char* TimingPolicyLabel(TimingPolicy policy);
const char* HistoryResetReasonLabel(HistoryResetReason reason);
RenderPassGraphDesc BuildRenderPassGraph(const RenderSettings& settings,
                                         bool needsAccumulationClear,
                                         bool needsSceneRebuild,
                                         bool hasDisplayPass,
                                         bool explicitRestirDiEnabled = true);
RenderPassGraphCompileResult CompileRenderPassGraph(const RenderPassGraphDesc& graph);
RenderPassExecutionResult ExecuteRenderPassGraph(
    const RenderPassGraphDesc& graph,
    const std::function<bool(const RenderPassDesc&)>& executePass);

}  // namespace PathTracer
