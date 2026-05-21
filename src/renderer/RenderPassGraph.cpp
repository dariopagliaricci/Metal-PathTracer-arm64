#include "renderer/RenderPassGraph.h"

#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <string>
#include <utility>

namespace PathTracer {

namespace {

uint32_t FeatureMask(std::initializer_list<RenderPassFeature> features) {
    uint32_t mask = 0u;
    for (RenderPassFeature feature : features) {
        mask |= static_cast<uint32_t>(feature);
    }
    return mask;
}

RenderPassDesc MakePass(std::string name,
                        RenderPassType type,
                        uint32_t featureMask,
                        std::vector<RenderResourceGroup> inputs,
                        std::vector<RenderResourceGroup> outputs,
                        RenderPassExecutionId executionId,
                        std::string pipelineKey,
                        DispatchDesc dispatch,
                        std::vector<std::string> settingsDependencies = {},
                        std::vector<RenderResourceGroup> historyDependencies = {}) {
    RenderPassDesc pass{};
    pass.name = std::move(name);
    pass.type = type;
    pass.featureMask = featureMask;
    pass.inputs = std::move(inputs);
    pass.outputs = std::move(outputs);
    for (RenderResourceGroup input : pass.inputs) {
        pass.reads.push_back(ResourceUse{input, RenderResourceGroupLabel(input), ResourceAccess::Read});
    }
    for (RenderResourceGroup output : pass.outputs) {
        const bool alsoRead = std::find(pass.inputs.begin(), pass.inputs.end(), output) != pass.inputs.end();
        pass.writes.push_back(ResourceUse{
            output,
            RenderResourceGroupLabel(output),
            alsoRead ? ResourceAccess::ReadWrite : ResourceAccess::Write});
    }
    pass.settingsDependencies = std::move(settingsDependencies);
    pass.historyDependencies = std::move(historyDependencies);
    pass.pipelineKey = std::move(pipelineKey);
    pass.dispatch = std::move(dispatch);
    pass.debugLabel = "FrameGraph.Pass." + pass.name;
    pass.timingLabel = pass.name;
    const bool executable =
        executionId != RenderPassExecutionId::NoOp &&
        executionId != RenderPassExecutionId::WavefrontStageMarker;
    pass.debugPolicy = executable ? DebugCapturePolicy::Scoped : DebugCapturePolicy::LabelOnly;
    if (!executable) {
        pass.timingPolicy = TimingPolicy::None;
    } else if (type == RenderPassType::Host) {
        pass.timingPolicy = TimingPolicy::Host;
    } else if (type == RenderPassType::External) {
        pass.timingPolicy = TimingPolicy::External;
    } else if (executionId == RenderPassExecutionId::MegakernelIntegrate ||
               executionId == RenderPassExecutionId::WavefrontWorkGraph ||
               executionId == RenderPassExecutionId::Present) {
        pass.timingPolicy = TimingPolicy::BucketedEncoder;
    } else {
        pass.timingPolicy = TimingPolicy::Encoder;
    }
    pass.executionId = executionId;
    return pass;
}

DispatchDesc Dispatch(DispatchKind kind, std::string dimensions, std::string activeWork = {}) {
    DispatchDesc desc{};
    desc.kind = kind;
    desc.dimensions = std::move(dimensions);
    desc.activeWork = std::move(activeWork);
    return desc;
}

FrameResourceDesc MakeResource(std::string name,
                               RenderResourceGroup group,
                               RenderResourceKind kind,
                               ResourceLifetime lifetime,
                               StorageClass storageClass,
                               std::string sizeFormula,
                               PixelFormatClass format,
                               HistoryPolicy historyPolicy,
                               std::vector<HistoryResetReason> invalidatesOn = {}) {
    FrameResourceDesc resource{};
    resource.name = std::move(name);
    resource.group = group;
    resource.kind = kind;
    resource.lifetime = lifetime;
    resource.storageClass = storageClass;
    resource.sizeFormula = std::move(sizeFormula);
    resource.format = format;
    resource.historyPolicy = historyPolicy;
    resource.invalidatesOn = std::move(invalidatesOn);
    resource.debugLabel = "FrameGraph.Resource." + resource.name;
    return resource;
}

std::vector<HistoryResetReason> FrameHistoryInvalidations() {
    return {
        HistoryResetReason::ResolutionChanged,
        HistoryResetReason::CameraCut,
        HistoryResetReason::SceneReloaded,
        HistoryResetReason::FeatureToggleChanged,
        HistoryResetReason::ManualDebugReset,
    };
}

std::vector<HistoryResetReason> TransportHistoryInvalidations() {
    return {
        HistoryResetReason::ResolutionChanged,
        HistoryResetReason::CameraCut,
        HistoryResetReason::SceneReloaded,
        HistoryResetReason::LightTableChanged,
        HistoryResetReason::EmissiveTableChanged,
        HistoryResetReason::MaterialTableChanged,
        HistoryResetReason::AccelerationStructureRebuilt,
        HistoryResetReason::FeatureToggleChanged,
        HistoryResetReason::ManualDebugReset,
    };
}

void ApplyProductionBridgeMetadata(std::vector<FrameResourceDesc>& resources) {
    for (FrameResourceDesc& resource : resources) {
        resource.targetStorageClass = resource.storageClass;
        resource.currentOwner = "RenderLoop";
        resource.targetOwner = "FrameGraph";
        resource.featureGate = "always";
        resource.debugReadbackPolicy =
            resource.lifetime == ResourceLifetime::DebugReadback ? "debug_readback_only"
                                                                 : "metadata_only";
        resource.migrationRisk = "low";
        resource.hotPath = false;
        resource.temporaryShared = false;

        switch (resource.group) {
            case RenderResourceGroup::FrameConstants:
                resource.currentOwner = "RenderLoop.UniformEncoder";
                resource.targetOwner = "FrameGraph";
                resource.debugReadbackPolicy = "cpu_authored";
                break;
            case RenderResourceGroup::SceneBuffers:
                resource.currentOwner = "SceneResources";
                resource.targetOwner = "SceneResources";
                resource.featureGate = "scene_loaded";
                break;
            case RenderResourceGroup::LightBuffers:
                resource.currentOwner = "SceneResources.LightTables";
                resource.targetOwner = "LightCandidateProvider";
                resource.featureGate = "analytic_lights || emissive_primitives || environment";
                break;
            case RenderResourceGroup::GBufferResources:
                resource.currentOwner = "RenderLoop.Accumulation";
                resource.targetOwner = "Reconstruction";
                resource.featureGate = "accumulation || denoiser || debug_aov";
                resource.migrationRisk = "medium";
                break;
            case RenderResourceGroup::ReSTIRDIResources:
                resource.currentOwner = "FrameGraph+TransportMemory.ReSTIRDI";
                resource.targetOwner = "FrameGraph+TransportMemory.ReSTIRDI";
                resource.storageClass = StorageClass::Private;
                resource.targetStorageClass = StorageClass::Private;
                resource.featureGate = "directLightMode=restir_di*";
                resource.debugReadbackPolicy = "staging";
                resource.migrationRisk = "medium";
                resource.hotPath = true;
                break;
            case RenderResourceGroup::ReGIRResources:
                resource.currentOwner = "FusedShader.ReGIR";
                resource.targetOwner = "LightCandidateProvider.ReGIR";
                resource.featureGate = "directLightMode=restir_di_regir_hybrid";
                resource.migrationRisk = "medium";
                resource.hotPath = true;
                break;
            case RenderResourceGroup::ReSTIRGIResources:
                resource.currentOwner = "PrototypeGI";
                resource.targetOwner = "TransportMemory.ReSTIRGI";
                resource.storageClass = StorageClass::Shared;
                resource.targetStorageClass = StorageClass::Private;
                resource.featureGate = "restirGiMode=diffuse_prototype";
                resource.migrationRisk = "high";
                resource.hotPath = true;
                resource.temporaryShared = true;
                break;
            case RenderResourceGroup::ReSTIRPTResources:
                resource.currentOwner = "PathReservoirPrototype";
                resource.targetOwner = "TransportMemory.ReSTIRPT";
                resource.storageClass = StorageClass::Shared;
                resource.targetStorageClass = StorageClass::Private;
                resource.featureGate = "restirPtMode!=off";
                resource.migrationRisk = "high";
                resource.hotPath = true;
                resource.temporaryShared = true;
                break;
            case RenderResourceGroup::PathGuidingResources:
                resource.currentOwner = "PathGuidingPrototype.WorldGrid";
                resource.targetOwner = "GuidingProvider";
                resource.targetStorageClass = StorageClass::Private;
                resource.featureGate = "pathGuidingMode=prototype";
                resource.debugReadbackPolicy = "direct_cpu_visible_for_metrics";
                resource.migrationRisk = "high";
                resource.hotPath = true;
                resource.temporaryShared = true;
                break;
            case RenderResourceGroup::RadianceCacheResources:
                resource.currentOwner = "RadianceProvider.WorldGrid";
                resource.targetOwner = "RadianceProvider";
                resource.targetStorageClass = StorageClass::Private;
                resource.featureGate = "radianceCacheMode=prototype";
                resource.debugReadbackPolicy = "metadata_only_metrics";
                resource.migrationRisk = "low";
                resource.hotPath = true;
                resource.temporaryShared = false;
                break;
            case RenderResourceGroup::CausticTransportResources:
                resource.currentOwner = "CausticTransport.PathSpacePrototype";
                resource.targetOwner = "PathSpaceTransport";
                resource.targetStorageClass = StorageClass::Private;
                resource.featureGate = "causticTransportMode=path_space";
                resource.debugReadbackPolicy = "metadata_only_counters";
                resource.migrationRisk = "medium";
                resource.hotPath = true;
                resource.temporaryShared = false;
                break;
            case RenderResourceGroup::VolumeTransportResources:
                resource.currentOwner = "VolumeTransport.Homogeneous";
                resource.targetOwner = "VolumeTransport";
                resource.targetStorageClass = StorageClass::Private;
                resource.featureGate = "volumeTransportMode=homogeneous";
                resource.debugReadbackPolicy = "metadata_only_counters";
                resource.migrationRisk = "medium";
                resource.hotPath = true;
                resource.temporaryShared = false;
                break;
            case RenderResourceGroup::MLBackendResources:
                resource.currentOwner = "MLBackend.NoOp";
                resource.targetOwner = "FrameGraph.MLBackend";
                resource.storageClass = StorageClass::Private;
                resource.targetStorageClass = StorageClass::Private;
                resource.featureGate = "mlBackendMode!=off";
                resource.debugReadbackPolicy = "metadata_only_capability_and_fallback";
                resource.migrationRisk = "medium";
                resource.hotPath = false;
                resource.temporaryShared = false;
                break;
            case RenderResourceGroup::WavefrontResources:
                resource.currentOwner = "WavefrontScheduler";
                resource.targetOwner = "WavefrontScheduler";
                resource.targetStorageClass = StorageClass::Private;
                resource.featureGate = "executionMode=wavefront";
                resource.debugReadbackPolicy = "counters_direct_cpu_visible";
                resource.migrationRisk = "high";
                resource.hotPath = true;
                resource.temporaryShared = true;
                break;
            case RenderResourceGroup::DenoiserResources:
            case RenderResourceGroup::DenoiserScratchResources:
                resource.currentOwner = "Denoiser";
                resource.targetOwner = "Reconstruction";
                resource.featureGate = "svgfDenoiseEnabled || oidn_export";
                resource.migrationRisk = "medium";
                break;
            case RenderResourceGroup::ReconstructionAOVResources:
                resource.currentOwner = "RenderLoop.Accumulation";
                resource.targetOwner = "FrameGraph.Reconstruction";
                resource.featureGate = "always";
                resource.debugReadbackPolicy = "metrics_and_optional_aov_export";
                resource.migrationRisk = "medium";
                resource.hotPath = true;
                break;
            case RenderResourceGroup::ReconstructionHistoryResources:
                resource.currentOwner = "Denoiser";
                resource.targetOwner = "FrameGraph.Reconstruction";
                resource.featureGate = "temporalAccumulation || svgfDenoiseEnabled || denoiseEnabled";
                resource.debugReadbackPolicy = "metadata_only";
                resource.migrationRisk = "medium";
                break;
            case RenderResourceGroup::FinalOutputResources:
                resource.currentOwner = "RenderLoop.Accumulation";
                resource.targetOwner = "FrameGraph";
                resource.featureGate = "always";
                break;
            case RenderResourceGroup::DebugResources:
                resource.currentOwner = "Debug";
                resource.targetOwner = "Debug";
                resource.targetStorageClass = StorageClass::Shared;
                resource.featureGate = "debug || metrics";
                resource.debugReadbackPolicy = "direct_cpu_visible_or_staging";
                resource.migrationRisk = "low";
                break;
            case RenderResourceGroup::ReSTIRDIDebugAOVs:
                resource.currentOwner = "Debug";
                resource.targetOwner = "Debug";
                resource.storageClass = StorageClass::Private;
                resource.targetStorageClass = StorageClass::Private;
                resource.featureGate = "restirDebugEnabled && explicitRestirDi";
                resource.debugReadbackPolicy = "staging";
                resource.migrationRisk = "low";
                break;
            case RenderResourceGroup::ReSTIRDIReservoirCurrent:
            case RenderResourceGroup::ReSTIRDIReservoirTemporal:
            case RenderResourceGroup::ReSTIRDIReservoirSpatial:
            case RenderResourceGroup::ReSTIRDIVisibility:
                resource.currentOwner = "FrameGraph";
                resource.targetOwner = "FrameGraph";
                resource.storageClass = StorageClass::Private;
                resource.targetStorageClass = StorageClass::Private;
                resource.featureGate = "directLightMode=restir_di* && explicitRestirDi";
                resource.debugReadbackPolicy = "staging_if_debug";
                resource.migrationRisk = "medium";
                resource.hotPath = true;
                resource.temporaryShared = false;
                break;
            case RenderResourceGroup::ReSTIRDIReservoirPrevious:
            case RenderResourceGroup::ReSTIRDINeighborPattern:
                resource.currentOwner = "TransportMemory.ReSTIRDI";
                resource.targetOwner = "TransportMemory.ReSTIRDI";
                resource.storageClass = StorageClass::Private;
                resource.targetStorageClass = StorageClass::Private;
                resource.featureGate = "directLightMode=restir_di* && explicitRestirDi";
                resource.debugReadbackPolicy =
                    resource.group == RenderResourceGroup::ReSTIRDINeighborPattern
                        ? "metadata_only"
                        : "staging";
                resource.migrationRisk = "medium";
                resource.hotPath = true;
                resource.temporaryShared = false;
                break;
        }
    }
}

bool IsExecutablePass(RenderPassExecutionId executionId) {
    return executionId != RenderPassExecutionId::NoOp &&
           executionId != RenderPassExecutionId::WavefrontStageMarker;
}

bool HasDeclaredResource(const std::vector<FrameResourceDesc>& resources,
                         RenderResourceGroup group) {
    return std::find_if(
               resources.begin(),
               resources.end(),
               [group](const FrameResourceDesc& resource) {
                   return resource.group == group;
               }) != resources.end();
}

bool HasGroup(const std::vector<RenderResourceGroup>& groups, RenderResourceGroup group) {
    return std::find(groups.begin(), groups.end(), group) != groups.end();
}

bool PassUsesResource(const RenderPassDesc& pass, RenderResourceGroup group) {
    if (HasGroup(pass.inputs, group) || HasGroup(pass.outputs, group) ||
        HasGroup(pass.historyDependencies, group)) {
        return true;
    }
    const auto useMatches = [group](const ResourceUse& use) {
        return use.group == group;
    };
    return std::find_if(pass.reads.begin(), pass.reads.end(), useMatches) != pass.reads.end() ||
           std::find_if(pass.writes.begin(), pass.writes.end(), useMatches) != pass.writes.end();
}

bool ResourceCanAlias(const FrameResourceDesc& resource,
                      const FrameResourceLifetimePlan& plan) {
    if (!plan.used || resource.lifetime != ResourceLifetime::Transient) {
        return false;
    }
    if (resource.kind == RenderResourceKind::AccelerationStructure ||
        resource.kind == RenderResourceKind::External) {
        return false;
    }
    return resource.storageClass == StorageClass::Private ||
           resource.storageClass == StorageClass::Memoryless;
}

bool AccessReads(ResourceAccess access) {
    return access == ResourceAccess::Read ||
           access == ResourceAccess::ReadWrite ||
           access == ResourceAccess::Preserve;
}

bool AccessWrites(ResourceAccess access) {
    return access == ResourceAccess::Write ||
           access == ResourceAccess::ReadWrite ||
           access == ResourceAccess::Clear;
}

ResourceAccess MergeAccess(ResourceAccess current, ResourceAccess next) {
    if (current == next) {
        return current;
    }
    if (current == ResourceAccess::Clear || next == ResourceAccess::Clear) {
        return ResourceAccess::Clear;
    }
    const bool reads = AccessReads(current) || AccessReads(next);
    const bool writes = AccessWrites(current) || AccessWrites(next);
    if (reads && writes) {
        return ResourceAccess::ReadWrite;
    }
    if (writes) {
        return ResourceAccess::Write;
    }
    return ResourceAccess::Read;
}

bool TransitionRequiresBarrier(ResourceAccess previous, ResourceAccess next) {
    if (previous == ResourceAccess::Preserve) {
        return false;
    }
    return AccessWrites(previous) || AccessWrites(next);
}

std::vector<ResourceUse> BuildMergedPassResourceUses(const RenderPassDesc& pass) {
    std::vector<ResourceUse> uses;
    const auto appendUse = [&uses](const ResourceUse& use) {
        auto existing = std::find_if(
            uses.begin(),
            uses.end(),
            [&use](const ResourceUse& candidate) {
                return candidate.group == use.group;
            });
        if (existing == uses.end()) {
            uses.push_back(use);
            return;
        }
        existing->access = MergeAccess(existing->access, use.access);
        if (existing->name.empty()) {
            existing->name = use.name;
        }
    };

    for (const ResourceUse& read : pass.reads) {
        appendUse(read);
    }
    for (const ResourceUse& write : pass.writes) {
        appendUse(write);
    }
    return uses;
}

void ValidateResourceUse(const RenderPassDesc& pass,
                         const ResourceUse& use,
                         const std::vector<FrameResourceDesc>& resources,
                         const char* direction,
                         std::vector<std::string>& errors,
                         std::vector<std::string>& warnings) {
    if (!HasDeclaredResource(resources, use.group)) {
        errors.push_back(
            pass.name + " " + direction + " undeclared resource group " +
            RenderResourceGroupLabel(use.group));
    }
    if (use.name.empty()) {
        warnings.push_back(pass.name + " has an unnamed " + direction + " resource use for " +
                           RenderResourceGroupLabel(use.group));
    }
}

std::vector<FrameResourceLifetimePlan> BuildResourceLifetimePlan(
    const RenderPassGraphDesc& graph,
    uint32_t& usedResourceCount,
    uint32_t& aliasableTransientResourceCount,
    uint32_t& transientAliasPoolCount) {
    std::vector<FrameResourceLifetimePlan> plans;
    plans.reserve(graph.resources.size());
    usedResourceCount = 0u;
    aliasableTransientResourceCount = 0u;
    transientAliasPoolCount = 0u;

    for (const FrameResourceDesc& resource : graph.resources) {
        FrameResourceLifetimePlan plan{};
        plan.name = resource.name;
        plan.group = resource.group;
        plan.lifetime = resource.lifetime;
        for (size_t passIndex = 0; passIndex < graph.passes.size(); ++passIndex) {
            if (!PassUsesResource(graph.passes[passIndex], resource.group)) {
                continue;
            }
            const int32_t index = static_cast<int32_t>(passIndex);
            if (!plan.used) {
                plan.firstPassIndex = index;
                plan.used = true;
            }
            plan.lastPassIndex = index;
        }
        plan.aliasable = ResourceCanAlias(resource, plan);
        if (plan.used) {
            usedResourceCount += 1u;
        }
        if (plan.aliasable) {
            aliasableTransientResourceCount += 1u;
        }
        plans.push_back(std::move(plan));
    }

    std::vector<int32_t> aliasPoolLastUse;
    for (FrameResourceLifetimePlan& plan : plans) {
        if (!plan.aliasable) {
            continue;
        }
        for (size_t poolIndex = 0; poolIndex < aliasPoolLastUse.size(); ++poolIndex) {
            if (aliasPoolLastUse[poolIndex] < plan.firstPassIndex) {
                plan.aliasPoolIndex = static_cast<int32_t>(poolIndex);
                aliasPoolLastUse[poolIndex] = plan.lastPassIndex;
                break;
            }
        }
        if (plan.aliasPoolIndex < 0) {
            plan.aliasPoolIndex = static_cast<int32_t>(aliasPoolLastUse.size());
            aliasPoolLastUse.push_back(plan.lastPassIndex);
        }
    }
    transientAliasPoolCount = static_cast<uint32_t>(aliasPoolLastUse.size());
    return plans;
}

std::vector<FrameResourceTransitionPlan> BuildResourceTransitionPlan(
    const RenderPassGraphDesc& graph,
    uint32_t& resourceTransitionCount,
    uint32_t& requiredBarrierCount,
    uint32_t& writeTransitionCount) {
    std::vector<FrameResourceTransitionPlan> transitions;
    std::vector<ResourceAccess> lastAccess(graph.resources.size(), ResourceAccess::Preserve);
    std::vector<bool> hasLastAccess(graph.resources.size(), false);
    resourceTransitionCount = 0u;
    requiredBarrierCount = 0u;
    writeTransitionCount = 0u;

    for (size_t passIndex = 0; passIndex < graph.passes.size(); ++passIndex) {
        const RenderPassDesc& pass = graph.passes[passIndex];
        for (const ResourceUse& use : BuildMergedPassResourceUses(pass)) {
            const auto resourceIt = std::find_if(
                graph.resources.begin(),
                graph.resources.end(),
                [&use](const FrameResourceDesc& resource) {
                    return resource.group == use.group;
                });
            if (resourceIt == graph.resources.end()) {
                continue;
            }
            const size_t resourceIndex =
                static_cast<size_t>(std::distance(graph.resources.begin(), resourceIt));
            const ResourceAccess previous =
                hasLastAccess[resourceIndex] ? lastAccess[resourceIndex] : ResourceAccess::Preserve;
            FrameResourceTransitionPlan transition{};
            transition.passIndex = static_cast<uint32_t>(passIndex);
            transition.passName = pass.name;
            transition.resourceName = resourceIt->name;
            transition.group = use.group;
            transition.previousAccess = previous;
            transition.access = use.access;
            transition.firstUse = !hasLastAccess[resourceIndex];
            transition.requiresBarrier = TransitionRequiresBarrier(previous, use.access);
            transition.writesResource = AccessWrites(use.access);
            if (transition.requiresBarrier) {
                requiredBarrierCount += 1u;
            }
            if (transition.writesResource) {
                writeTransitionCount += 1u;
            }
            transitions.push_back(std::move(transition));
            lastAccess[resourceIndex] = use.access;
            hasLastAccess[resourceIndex] = true;
        }
    }
    resourceTransitionCount = static_cast<uint32_t>(transitions.size());
    return transitions;
}

}  // namespace

const char* RenderPassTypeLabel(RenderPassType type) {
    switch (type) {
        case RenderPassType::Host:
            return "host";
        case RenderPassType::Compute:
            return "compute";
        case RenderPassType::Render:
            return "render";
        case RenderPassType::External:
            return "external";
    }
    return "unknown";
}

const char* RenderResourceGroupLabel(RenderResourceGroup group) {
    switch (group) {
        case RenderResourceGroup::FrameConstants:
            return "FrameConstants";
        case RenderResourceGroup::SceneBuffers:
            return "SceneBuffers";
        case RenderResourceGroup::LightBuffers:
            return "LightBuffers";
        case RenderResourceGroup::GBufferResources:
            return "GBufferResources";
        case RenderResourceGroup::ReSTIRDIResources:
            return "ReSTIRDIResources";
        case RenderResourceGroup::ReGIRResources:
            return "ReGIRResources";
        case RenderResourceGroup::ReSTIRGIResources:
            return "ReSTIRGIResources";
        case RenderResourceGroup::ReSTIRPTResources:
            return "ReSTIRPTResources";
        case RenderResourceGroup::WavefrontResources:
            return "WavefrontResources";
        case RenderResourceGroup::DenoiserResources:
            return "DenoiserResources";
        case RenderResourceGroup::FinalOutputResources:
            return "FinalOutputResources";
        case RenderResourceGroup::DebugResources:
            return "DebugResources";
        case RenderResourceGroup::DenoiserScratchResources:
            return "DenoiserScratchResources";
        case RenderResourceGroup::ReSTIRDIReservoirCurrent:
            return "ReSTIRDIReservoirCurrent";
        case RenderResourceGroup::ReSTIRDIReservoirPrevious:
            return "ReSTIRDIReservoirPrevious";
        case RenderResourceGroup::ReSTIRDIReservoirTemporal:
            return "ReSTIRDIReservoirTemporal";
        case RenderResourceGroup::ReSTIRDIReservoirSpatial:
            return "ReSTIRDIReservoirSpatial";
        case RenderResourceGroup::ReSTIRDIVisibility:
            return "ReSTIRDIVisibility";
        case RenderResourceGroup::ReSTIRDIDebugAOVs:
            return "ReSTIRDIDebugAOVs";
        case RenderResourceGroup::ReSTIRDINeighborPattern:
            return "ReSTIRDINeighborPattern";
        case RenderResourceGroup::PathGuidingResources:
            return "PathGuidingResources";
        case RenderResourceGroup::RadianceCacheResources:
            return "RadianceCacheResources";
        case RenderResourceGroup::ReconstructionAOVResources:
            return "ReconstructionAOVResources";
        case RenderResourceGroup::ReconstructionHistoryResources:
            return "ReconstructionHistoryResources";
        case RenderResourceGroup::CausticTransportResources:
            return "CausticTransportResources";
        case RenderResourceGroup::VolumeTransportResources:
            return "VolumeTransportResources";
        case RenderResourceGroup::MLBackendResources:
            return "MLBackendResources";
    }
    return "Unknown";
}

const char* ResourceLifetimeLabel(ResourceLifetime lifetime) {
    switch (lifetime) {
        case ResourceLifetime::Imported:
            return "Imported";
        case ResourceLifetime::Transient:
            return "Transient";
        case ResourceLifetime::Persistent:
            return "Persistent";
        case ResourceLifetime::History:
            return "History";
        case ResourceLifetime::DebugReadback:
            return "DebugReadback";
    }
    return "Unknown";
}

const char* StorageClassLabel(StorageClass storageClass) {
    switch (storageClass) {
        case StorageClass::Unknown:
            return "Unknown";
        case StorageClass::Private:
            return "Private";
        case StorageClass::Shared:
            return "Shared";
        case StorageClass::Managed:
            return "Managed";
        case StorageClass::Memoryless:
            return "Memoryless";
        case StorageClass::External:
            return "External";
    }
    return "Unknown";
}

const char* ResourceAccessLabel(ResourceAccess access) {
    switch (access) {
        case ResourceAccess::Read:
            return "Read";
        case ResourceAccess::Write:
            return "Write";
        case ResourceAccess::ReadWrite:
            return "ReadWrite";
        case ResourceAccess::Clear:
            return "Clear";
        case ResourceAccess::Preserve:
            return "Preserve";
    }
    return "Unknown";
}

const char* RenderPassExecutionLabel(RenderPassExecutionId executionId) {
    switch (executionId) {
        case RenderPassExecutionId::NoOp:
            return "NoOp";
        case RenderPassExecutionId::UpdateAccelerationStructures:
            return "UpdateAccelerationStructures";
        case RenderPassExecutionId::ClearAccumulation:
            return "ClearAccumulation";
        case RenderPassExecutionId::MegakernelIntegrate:
            return "MegakernelIntegrate";
        case RenderPassExecutionId::WavefrontWorkGraph:
            return "WavefrontWorkGraph";
        case RenderPassExecutionId::WavefrontStageMarker:
            return "WavefrontStageMarker";
        case RenderPassExecutionId::Present:
            return "Present";
        case RenderPassExecutionId::SvgfDenoise:
            return "SvgfDenoise";
        case RenderPassExecutionId::OidnDenoise:
            return "OidnDenoise";
        case RenderPassExecutionId::Display:
            return "Display";
        case RenderPassExecutionId::RestirDiInitializeReservoirs:
            return "RestirDiInitializeReservoirs";
        case RenderPassExecutionId::RestirDiInitialCandidates:
            return "RestirDiInitialCandidates";
        case RenderPassExecutionId::RestirDiTemporalReuse:
            return "RestirDiTemporalReuse";
        case RenderPassExecutionId::RestirDiSpatialReuse:
            return "RestirDiSpatialReuse";
        case RenderPassExecutionId::RestirDiVisibility:
            return "RestirDiVisibility";
        case RenderPassExecutionId::RestirDiApplyLighting:
            return "RestirDiApplyLighting";
        case RenderPassExecutionId::RestirDiDebugAov:
            return "RestirDiDebugAov";
        case RenderPassExecutionId::MlBackendNoOp:
            return "MlBackendNoOp";
    }
    return "Unknown";
}

const char* DebugCapturePolicyLabel(DebugCapturePolicy policy) {
    switch (policy) {
        case DebugCapturePolicy::None:
            return "None";
        case DebugCapturePolicy::LabelOnly:
            return "LabelOnly";
        case DebugCapturePolicy::Scoped:
            return "Scoped";
    }
    return "Unknown";
}

const char* TimingPolicyLabel(TimingPolicy policy) {
    switch (policy) {
        case TimingPolicy::None:
            return "None";
        case TimingPolicy::Host:
            return "Host";
        case TimingPolicy::Encoder:
            return "Encoder";
        case TimingPolicy::BucketedEncoder:
            return "BucketedEncoder";
        case TimingPolicy::External:
            return "External";
    }
    return "Unknown";
}

const char* HistoryResetReasonLabel(HistoryResetReason reason) {
    switch (reason) {
        case HistoryResetReason::None:
            return "None";
        case HistoryResetReason::ResolutionChanged:
            return "ResolutionChanged";
        case HistoryResetReason::SceneReloaded:
            return "SceneReloaded";
        case HistoryResetReason::CameraCut:
            return "CameraCut";
        case HistoryResetReason::LightTableChanged:
            return "LightTableChanged";
        case HistoryResetReason::FeatureToggleChanged:
            return "FeatureToggleChanged";
        case HistoryResetReason::AccelerationStructureRebuilt:
            return "AccelerationStructureRebuilt";
        case HistoryResetReason::EmissiveTableChanged:
            return "EmissiveTableChanged";
        case HistoryResetReason::MaterialTableChanged:
            return "MaterialTableChanged";
        case HistoryResetReason::FrameAgeExpired:
            return "FrameAgeExpired";
        case HistoryResetReason::ManualDebugReset:
            return "ManualDebugReset";
    }
    return "Unknown";
}

RenderPassGraphDesc BuildRenderPassGraph(const RenderSettings& settings,
                                         bool needsAccumulationClear,
                                         bool needsSceneRebuild,
                                         bool hasDisplayPass,
                                         bool explicitRestirDiEnabled) {
    RenderPassGraphDesc graph{};
    const bool restirDiEnabled =
        explicitRestirDiEnabled &&
        (settings.directLightMode == RenderSettings::DirectLightMode::RestirDi ||
         settings.directLightMode == RenderSettings::DirectLightMode::RestirDiRegirHybrid);
    const bool volumeTransportEnabled =
        settings.volumeTransportMode == RenderSettings::VolumeTransportMode::Homogeneous;
    const bool mlBackendEnabled =
        settings.mlBackendMode != RenderSettings::MlBackendMode::Off;
    const bool restirDiWavefrontStageEnabled =
        restirDiEnabled && settings.executionMode == RenderSettings::ExecutionMode::Wavefront;
    const bool restirDiScreenPassesEnabled = restirDiEnabled && !restirDiWavefrontStageEnabled;
    const bool reconstructionConfidenceInputsEnabled =
        restirDiEnabled ||
        settings.pathGuidingMode == RenderSettings::PathGuidingMode::Prototype ||
        settings.radianceCacheMode == RenderSettings::RadianceCacheMode::Prototype ||
        settings.causticTransportMode == RenderSettings::CausticTransportMode::PathSpacePrototype ||
        volumeTransportEnabled ||
        mlBackendEnabled ||
        settings.restirPtMode != RenderSettings::RestirPtMode::Off ||
        settings.restirGiMode != RenderSettings::RestirGiMode::Off;
    graph.resources = {
        MakeResource(
            "FrameConstants",
            RenderResourceGroup::FrameConstants,
            RenderResourceKind::Buffer,
            ResourceLifetime::Imported,
            StorageClass::Shared,
            "sizeof(PathtraceUniforms) per encoded sample plus DisplayUniforms",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::None),
        MakeResource(
            "SceneBuffers",
            RenderResourceGroup::SceneBuffers,
            RenderResourceKind::Buffer,
            ResourceLifetime::Imported,
            StorageClass::Private,
            "Scene-dependent geometry, material, texture metadata, BVH, and acceleration structures",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::None,
            {HistoryResetReason::SceneReloaded, HistoryResetReason::AccelerationStructureRebuilt}),
        MakeResource(
            "LightBuffers",
            RenderResourceGroup::LightBuffers,
            RenderResourceKind::Buffer,
            ResourceLifetime::Imported,
            StorageClass::Private,
            "Analytic lights, emissive primitive table, and environment alias tables",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::None,
            {HistoryResetReason::LightTableChanged, HistoryResetReason::EmissiveTableChanged}),
        MakeResource(
            "GBufferResources",
            RenderResourceGroup::GBufferResources,
            RenderResourceKind::Texture,
            ResourceLifetime::History,
            StorageClass::Private,
            "width * height * (albedo + normal + position + sample-count/AOV history)",
            PixelFormatClass::RGBA32Float,
            HistoryPolicy::CurrentPrevious,
            FrameHistoryInvalidations()),
        MakeResource(
            "ReconstructionAOVResources",
            RenderResourceGroup::ReconstructionAOVResources,
            RenderResourceKind::Texture,
            ResourceLifetime::History,
            StorageClass::Private,
            "stable AOV contract: radiance, albedo, shadingNormal, geometricNormal, roughness, "
            "specular, depth, motion, emission, directRadiance, indirectRadiance, sampleCount, "
            "variance, confidence, disocclusion",
            PixelFormatClass::RGBA32Float,
            HistoryPolicy::CurrentPrevious,
            FrameHistoryInvalidations()),
        MakeResource(
            "ReconstructionHistoryResources",
            RenderResourceGroup::ReconstructionHistoryResources,
            RenderResourceKind::Texture,
            ResourceLifetime::History,
            StorageClass::Private,
            "temporal reconstruction history: moments, guide normals/depth, history length, "
            "validity/disocclusion, backend output, and backend capability/fallback metadata",
            PixelFormatClass::RGBA32Float,
            HistoryPolicy::CurrentPrevious,
            FrameHistoryInvalidations()),
        MakeResource(
            "ReSTIRDIResources",
            RenderResourceGroup::ReSTIRDIResources,
            RenderResourceKind::Buffer,
            ResourceLifetime::History,
            StorageClass::Private,
            "width * height * reservoir/state bytes for current and previous direct-light reuse",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::CurrentPrevious,
            TransportHistoryInvalidations()),
        MakeResource(
            "ReGIRResources",
            RenderResourceGroup::ReGIRResources,
            RenderResourceKind::Buffer,
            ResourceLifetime::Persistent,
            StorageClass::Private,
            "world-grid cells * candidates-per-cell * light-candidate state",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::Persistent,
            {HistoryResetReason::LightTableChanged,
             HistoryResetReason::EmissiveTableChanged,
             HistoryResetReason::SceneReloaded,
             HistoryResetReason::FeatureToggleChanged}),
        MakeResource(
            "ReSTIRGIResources",
            RenderResourceGroup::ReSTIRGIResources,
            RenderResourceKind::Buffer,
            ResourceLifetime::History,
            StorageClass::Private,
            "width * height * kRestirPathReservoirDepthCount * RestirPtReservoirState",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::CurrentPrevious,
            TransportHistoryInvalidations()),
        MakeResource(
            "ReSTIRPTResources",
            RenderResourceGroup::ReSTIRPTResources,
            RenderResourceKind::Buffer,
            ResourceLifetime::History,
            StorageClass::Private,
            "width * height * kRestirPathReservoirDepthCount * RestirPtReservoirState",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::CurrentPrevious,
            TransportHistoryInvalidations()),
        MakeResource(
            "PathGuidingResources",
            RenderResourceGroup::PathGuidingResources,
            RenderResourceKind::Buffer,
            ResourceLifetime::History,
            StorageClass::Shared,
            "hashed world-grid cells * 64 octahedral directional bins * PathGuidingReservoirState",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::Persistent,
            TransportHistoryInvalidations()),
        MakeResource(
            "RadianceCacheResources",
            RenderResourceGroup::RadianceCacheResources,
            RenderResourceKind::Buffer,
            ResourceLifetime::History,
            StorageClass::Private,
            "hashed world-grid cells * kRestirPathReservoirDepthCount * RadianceCacheState with query metadata",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::Persistent,
            TransportHistoryInvalidations()),
        MakeResource(
            "CausticTransportResources",
            RenderResourceGroup::CausticTransportResources,
            RenderResourceKind::Buffer,
            ResourceLifetime::History,
            StorageClass::Private,
            "bounded PathVertex + PathConnection records for eye/light subpaths and MIS metadata",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::Persistent,
            TransportHistoryInvalidations()),
        MakeResource(
            "VolumeTransportResources",
            RenderResourceGroup::VolumeTransportResources,
            RenderResourceKind::Buffer,
            ResourceLifetime::History,
            StorageClass::Private,
            "MediumDescriptor + MediumState path metadata + transmittance/scattering debug counters",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::Persistent,
            TransportHistoryInvalidations()),
        MakeResource(
            "MLBackendResources",
            RenderResourceGroup::MLBackendResources,
            RenderResourceKind::Buffer,
            ResourceLifetime::Transient,
            StorageClass::Private,
            "tensor descriptors for reconstruction inputs/outputs: radiance, albedo, normal, depth, "
            "motion, variance, confidence, optional model package metadata, and scratch tensors",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::None,
            FrameHistoryInvalidations()),
        MakeResource(
            "WavefrontResources",
            RenderResourceGroup::WavefrontResources,
            RenderResourceKind::Buffer,
            ResourceLifetime::Persistent,
            StorageClass::Shared,
            "width * height * (2 ray queues + hit records + path state + shadow queue multiplier + counters)",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::CurrentFrame,
            {HistoryResetReason::ResolutionChanged, HistoryResetReason::FeatureToggleChanged}),
        MakeResource(
            "DenoiserResources",
            RenderResourceGroup::DenoiserResources,
            RenderResourceKind::Texture,
            ResourceLifetime::History,
            StorageClass::Private,
            "width * height * denoiser color/moments/history/AOV textures",
            PixelFormatClass::RGBA32Float,
            HistoryPolicy::CurrentPrevious,
            FrameHistoryInvalidations()),
        MakeResource(
            "FinalOutputResources",
            RenderResourceGroup::FinalOutputResources,
            RenderResourceKind::Texture,
            ResourceLifetime::History,
            StorageClass::Private,
            "width * height * (radianceSum RGBA32F + sampleCount R32U + present RGBA16F)",
            PixelFormatClass::RGBA32Float,
            HistoryPolicy::Persistent,
            {HistoryResetReason::ResolutionChanged, HistoryResetReason::ManualDebugReset}),
        MakeResource(
            "DebugResources",
            RenderResourceGroup::DebugResources,
            RenderResourceKind::Buffer,
            ResourceLifetime::DebugReadback,
            StorageClass::Shared,
            "debug/stats buffer sizes only when debug or metrics are enabled",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::DebugReadback),
    };
    if (settings.svgfDenoiseEnabled) {
        graph.resources.push_back(MakeResource(
            "DenoiserScratchResources",
            RenderResourceGroup::DenoiserScratchResources,
            RenderResourceKind::Texture,
            ResourceLifetime::Transient,
            StorageClass::Private,
            "width * height * SVGF scratch/ping-pong intermediates reused within the frame",
            PixelFormatClass::RGBA32Float,
            HistoryPolicy::None));
    }
    if (restirDiScreenPassesEnabled) {
        graph.resources.push_back(MakeResource(
            "ReSTIRDIReservoirCurrent",
            RenderResourceGroup::ReSTIRDIReservoirCurrent,
            RenderResourceKind::Buffer,
            ResourceLifetime::History,
            StorageClass::Private,
            "width * height * RestirDIReservoirState current-frame candidates",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::CurrentFrame,
            TransportHistoryInvalidations()));
        graph.resources.push_back(MakeResource(
            "ReSTIRDIReservoirPrevious",
            RenderResourceGroup::ReSTIRDIReservoirPrevious,
            RenderResourceKind::Buffer,
            ResourceLifetime::History,
            StorageClass::Private,
            "width * height * RestirDIReservoirState previous-frame temporal history",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::CurrentPrevious,
            TransportHistoryInvalidations()));
        graph.resources.push_back(MakeResource(
            "ReSTIRDIReservoirTemporal",
            RenderResourceGroup::ReSTIRDIReservoirTemporal,
            RenderResourceKind::Buffer,
            ResourceLifetime::Transient,
            StorageClass::Private,
            "width * height * RestirDIReservoirState temporal-reuse intermediate",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::None));
        graph.resources.push_back(MakeResource(
            "ReSTIRDIReservoirSpatial",
            RenderResourceGroup::ReSTIRDIReservoirSpatial,
            RenderResourceKind::Buffer,
            ResourceLifetime::Transient,
            StorageClass::Private,
            "width * height * RestirDIReservoirState spatial-reuse intermediate",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::None));
        graph.resources.push_back(MakeResource(
            "ReSTIRDIVisibility",
            RenderResourceGroup::ReSTIRDIVisibility,
            RenderResourceKind::Buffer,
            ResourceLifetime::Transient,
            StorageClass::Private,
            "width * height * RestirDIVisibilityState selected-sample visibility/radiance",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::None));
        graph.resources.push_back(MakeResource(
            "ReSTIRDINeighborPattern",
            RenderResourceGroup::ReSTIRDINeighborPattern,
            RenderResourceKind::Buffer,
            ResourceLifetime::Persistent,
            StorageClass::Private,
            "spatialReuseNeighborCount * deterministic neighbor offsets",
            PixelFormatClass::StructuredBuffer,
            HistoryPolicy::Persistent,
            {HistoryResetReason::FeatureToggleChanged}));
        if (settings.restirDebugEnabled) {
            graph.resources.push_back(MakeResource(
                "ReSTIRDIDebugAOVs",
                RenderResourceGroup::ReSTIRDIDebugAOVs,
                RenderResourceKind::Buffer,
                ResourceLifetime::DebugReadback,
                StorageClass::Shared,
                "width * height * reservoir debug AOV state",
                PixelFormatClass::StructuredBuffer,
                HistoryPolicy::DebugReadback));
        }
    }
    ApplyProductionBridgeMetadata(graph.resources);
    auto& passes = graph.passes;

    if (needsAccumulationClear) {
        passes.push_back(MakePass(
            "Clear Accumulation",
            RenderPassType::Compute,
            FeatureMask({RenderPassFeature::Optional,
                         RenderPassFeature::ProducesHistory,
                         RenderPassFeature::Reconstruction}),
            {RenderResourceGroup::FinalOutputResources},
            {RenderResourceGroup::FinalOutputResources,
             RenderResourceGroup::ReconstructionAOVResources,
             RenderResourceGroup::ReconstructionHistoryResources,
             RenderResourceGroup::DenoiserResources},
            RenderPassExecutionId::ClearAccumulation,
            "pathtraceClearKernel",
            Dispatch(DispatchKind::FullScreen, "render width * render height")));
    }

    if (needsSceneRebuild) {
        passes.push_back(MakePass(
            "Update Acceleration Structures",
            RenderPassType::Host,
            FeatureMask({RenderPassFeature::Optional}),
            {RenderResourceGroup::SceneBuffers},
            {RenderResourceGroup::SceneBuffers},
            RenderPassExecutionId::UpdateAccelerationStructures,
            "SceneResources.rebuildAccelerationStructures",
            Dispatch(DispatchKind::External, "scene-dependent host work")));
    }

    const std::vector<RenderResourceGroup> commonInputs = {
        RenderResourceGroup::FrameConstants,
        RenderResourceGroup::SceneBuffers,
        RenderResourceGroup::LightBuffers,
        RenderResourceGroup::DebugResources,
    };
    std::vector<RenderResourceGroup> integrateOutputs = {
        RenderResourceGroup::FinalOutputResources,
        RenderResourceGroup::GBufferResources,
        RenderResourceGroup::ReconstructionAOVResources,
        RenderResourceGroup::ReSTIRDIResources,
        RenderResourceGroup::ReGIRResources,
        RenderResourceGroup::ReSTIRGIResources,
        RenderResourceGroup::PathGuidingResources,
        RenderResourceGroup::ReSTIRPTResources,
        RenderResourceGroup::RadianceCacheResources,
        RenderResourceGroup::CausticTransportResources,
        RenderResourceGroup::DebugResources,
    };
    if (volumeTransportEnabled) {
        integrateOutputs.insert(
            std::prev(integrateOutputs.end()),
            RenderResourceGroup::VolumeTransportResources);
    }

    if (restirDiScreenPassesEnabled) {
        passes.push_back(MakePass(
            "ReSTIR DI Reservoir Initialize",
            RenderPassType::Compute,
            FeatureMask({RenderPassFeature::Optional, RenderPassFeature::ProducesHistory}),
            {RenderResourceGroup::FrameConstants},
            {RenderResourceGroup::ReSTIRDIReservoirCurrent,
             RenderResourceGroup::ReSTIRDIReservoirTemporal,
             RenderResourceGroup::ReSTIRDIReservoirSpatial,
             RenderResourceGroup::ReSTIRDIVisibility},
            RenderPassExecutionId::RestirDiInitializeReservoirs,
            "restirDiInitializeReservoirsKernel",
            Dispatch(DispatchKind::FullScreen, "render width * render height"),
            {"directLightMode"}));
        passes.push_back(MakePass(
            "ReSTIR DI Initial Candidates",
            RenderPassType::Compute,
            FeatureMask({RenderPassFeature::Optional, RenderPassFeature::UsesHistory, RenderPassFeature::ProducesHistory}),
            {RenderResourceGroup::FrameConstants,
             RenderResourceGroup::SceneBuffers,
             RenderResourceGroup::LightBuffers,
             RenderResourceGroup::GBufferResources},
            {RenderResourceGroup::ReSTIRDIReservoirCurrent},
            RenderPassExecutionId::RestirDiInitialCandidates,
            "restirDiInitialCandidatesKernel",
            Dispatch(DispatchKind::FullScreen, "render width * render height"),
            {"directLightMode", "risCandidateCount", "spatialReuseNeighborCount"},
            {RenderResourceGroup::ReSTIRDIReservoirCurrent}));
        passes.push_back(MakePass(
            "ReSTIR DI Temporal Reuse",
            RenderPassType::Compute,
            FeatureMask({RenderPassFeature::Optional, RenderPassFeature::UsesHistory, RenderPassFeature::ProducesHistory}),
            {RenderResourceGroup::FrameConstants,
             RenderResourceGroup::GBufferResources,
             RenderResourceGroup::ReSTIRDIReservoirCurrent,
             RenderResourceGroup::ReSTIRDIReservoirPrevious},
            {RenderResourceGroup::ReSTIRDIReservoirTemporal},
            RenderPassExecutionId::RestirDiTemporalReuse,
            "restirDiTemporalReuseKernel",
            Dispatch(DispatchKind::FullScreen, "render width * render height"),
            {"directLightMode"},
            {RenderResourceGroup::ReSTIRDIReservoirPrevious}));
        passes.push_back(MakePass(
            "ReSTIR DI Spatial Reuse",
            RenderPassType::Compute,
            FeatureMask({RenderPassFeature::Optional, RenderPassFeature::UsesHistory, RenderPassFeature::ProducesHistory}),
            {RenderResourceGroup::FrameConstants,
             RenderResourceGroup::GBufferResources,
             RenderResourceGroup::ReSTIRDIReservoirTemporal,
             RenderResourceGroup::ReSTIRDINeighborPattern},
            {RenderResourceGroup::ReSTIRDIReservoirSpatial},
            RenderPassExecutionId::RestirDiSpatialReuse,
            "restirDiSpatialReuseKernel",
            Dispatch(DispatchKind::FullScreen, "render width * render height", "spatialReuseNeighborCount"),
            {"directLightMode", "spatialReuseNeighborCount"}));
        passes.push_back(MakePass(
            "ReSTIR DI Visibility",
            RenderPassType::Compute,
            FeatureMask({RenderPassFeature::Optional, RenderPassFeature::UsesHistory, RenderPassFeature::ProducesHistory}),
            {RenderResourceGroup::FrameConstants,
             RenderResourceGroup::SceneBuffers,
             RenderResourceGroup::ReSTIRDIReservoirSpatial},
            {RenderResourceGroup::ReSTIRDIVisibility,
             RenderResourceGroup::ReSTIRDIReservoirPrevious},
            RenderPassExecutionId::RestirDiVisibility,
            "restirDiVisibilityKernel",
            Dispatch(DispatchKind::FullScreen, "render width * render height"),
            {"directLightMode"}));
        std::vector<RenderResourceGroup> debugInputs = {
            RenderResourceGroup::FrameConstants,
            RenderResourceGroup::ReSTIRDIReservoirCurrent,
            RenderResourceGroup::ReSTIRDIReservoirTemporal,
            RenderResourceGroup::ReSTIRDIReservoirSpatial,
            RenderResourceGroup::ReSTIRDIVisibility,
        };
        if (settings.restirDebugEnabled) {
            passes.push_back(MakePass(
                "ReSTIR DI Reservoir Debug AOV",
                RenderPassType::Compute,
                FeatureMask({RenderPassFeature::Optional, RenderPassFeature::DebugOnly}),
                debugInputs,
                {RenderResourceGroup::ReSTIRDIDebugAOVs, RenderResourceGroup::DebugResources},
                RenderPassExecutionId::RestirDiDebugAov,
                "restirDiDebugAovKernel",
                Dispatch(DispatchKind::FullScreen, "render width * render height"),
                {"restirDebugEnabled", "restirDebugView"},
                {RenderResourceGroup::ReSTIRDIReservoirPrevious}));
        }
        passes.push_back(MakePass(
            "Legacy ReSTIR DI Fused Reference",
            RenderPassType::Compute,
            FeatureMask({RenderPassFeature::Optional, RenderPassFeature::DebugOnly}),
            {RenderResourceGroup::FrameConstants,
             RenderResourceGroup::SceneBuffers,
             RenderResourceGroup::LightBuffers,
             RenderResourceGroup::GBufferResources,
             RenderResourceGroup::ReSTIRDIResources},
            {RenderResourceGroup::ReSTIRDIResources, RenderResourceGroup::DebugResources},
            RenderPassExecutionId::NoOp,
            "legacy fused integrator-local ReSTIR DI path; retained until D shader passes replace it",
            Dispatch(DispatchKind::FullScreen, "render width * render height"),
                {"directLightMode", "risCandidateCount", "spatialReuseNeighborCount"},
                {RenderResourceGroup::ReSTIRDIResources}));
    }

    if (settings.executionMode == RenderSettings::ExecutionMode::Wavefront) {
        const uint32_t wavefrontMask = FeatureMask({RenderPassFeature::Wavefront});
        passes.push_back(MakePass(
            "Wavefront Reset Queues",
            RenderPassType::Compute,
            wavefrontMask,
            {RenderResourceGroup::WavefrontResources},
            {RenderResourceGroup::WavefrontResources},
            RenderPassExecutionId::WavefrontWorkGraph,
            "wavefrontResetKernel",
            Dispatch(DispatchKind::Linear, "1 thread; scheduler expands per sample/depth")));
        passes.push_back(MakePass(
            "Wavefront Generate Primary",
            RenderPassType::Compute,
            wavefrontMask,
            {RenderResourceGroup::FrameConstants},
            {RenderResourceGroup::WavefrontResources},
            RenderPassExecutionId::WavefrontStageMarker,
            "wavefrontGeneratePrimaryKernel",
            Dispatch(DispatchKind::Linear, "width * height")));
        passes.push_back(MakePass(
            "Wavefront Prepare Intersections",
            RenderPassType::Compute,
            wavefrontMask,
            {RenderResourceGroup::FrameConstants, RenderResourceGroup::WavefrontResources},
            {RenderResourceGroup::WavefrontResources},
            RenderPassExecutionId::WavefrontStageMarker,
            "wavefrontPrepareIterationKernel",
            Dispatch(DispatchKind::Linear, "width * height per depth", "active queue metadata")));
        passes.push_back(MakePass(
            "Wavefront Compact Queue",
            RenderPassType::Compute,
            wavefrontMask,
            {RenderResourceGroup::FrameConstants, RenderResourceGroup::WavefrontResources},
            {RenderResourceGroup::WavefrontResources, RenderResourceGroup::DebugResources},
            RenderPassExecutionId::WavefrontStageMarker,
            "wavefrontCompactQueueKernel",
            Dispatch(DispatchKind::Indirect,
                     "1 metadata thread per depth when enabled",
                     "wavefrontSchedulingPolicy,wavefrontCompactionEnabled")));
        passes.push_back(MakePass(
            "Wavefront Intersect",
            RenderPassType::Compute,
            wavefrontMask,
            {RenderResourceGroup::SceneBuffers, RenderResourceGroup::WavefrontResources},
            {RenderResourceGroup::WavefrontResources},
            RenderPassExecutionId::WavefrontStageMarker,
            "wavefrontIntersectKernel|wavefrontIntersectHardwareKernel",
            Dispatch(DispatchKind::Indirect,
                     "active queue count per depth (active_count_indirect)",
                     "active_count_indirect")));
        if (restirDiWavefrontStageEnabled) {
            passes.push_back(MakePass(
                "Wavefront ReSTIR DI Direct Lighting",
                RenderPassType::Compute,
                FeatureMask({RenderPassFeature::Wavefront, RenderPassFeature::Optional}),
                commonInputs,
                {RenderResourceGroup::WavefrontResources, RenderResourceGroup::DebugResources},
                RenderPassExecutionId::WavefrontStageMarker,
                "wavefrontRestirDiDirectLightingKernel",
                Dispatch(DispatchKind::Linear, "width * height per depth", "activeRayCount"),
                {"directLightMode", "risCandidateCount", "worldReuseCellSize"}));
        }
        std::vector<RenderResourceGroup> wavefrontShadeOutputs = {
            RenderResourceGroup::WavefrontResources,
            RenderResourceGroup::ReSTIRDIResources,
            RenderResourceGroup::ReGIRResources,
            RenderResourceGroup::ReSTIRGIResources,
            RenderResourceGroup::PathGuidingResources,
            RenderResourceGroup::ReSTIRPTResources,
            RenderResourceGroup::RadianceCacheResources,
            RenderResourceGroup::CausticTransportResources,
            RenderResourceGroup::DebugResources,
        };
        if (volumeTransportEnabled) {
            wavefrontShadeOutputs.insert(
                std::prev(wavefrontShadeOutputs.end()),
                RenderResourceGroup::VolumeTransportResources);
        }
        passes.push_back(MakePass(
            "Wavefront Shade",
            RenderPassType::Compute,
            wavefrontMask,
            commonInputs,
            wavefrontShadeOutputs,
            RenderPassExecutionId::WavefrontStageMarker,
            "wavefrontShadeKernel",
            Dispatch(DispatchKind::Linear,
                     "width * height per depth (direct_grid_fallback)",
                     "direct_grid_fallback(activeRayCount metadata)")));
        passes.push_back(MakePass(
            "Wavefront Shadow",
            RenderPassType::Compute,
            wavefrontMask,
            {RenderResourceGroup::SceneBuffers, RenderResourceGroup::WavefrontResources},
            {RenderResourceGroup::WavefrontResources},
            RenderPassExecutionId::WavefrontStageMarker,
            "wavefrontShadowKernel|wavefrontShadowHardwareKernel",
            Dispatch(DispatchKind::Linear,
                     "width * height per depth (direct_grid_sparse_storage)",
                     "direct_grid_sparse_storage(shadowRayCount metadata)")));
        passes.push_back(MakePass(
            "Wavefront Swap Queues",
            RenderPassType::Compute,
            wavefrontMask,
            {RenderResourceGroup::WavefrontResources},
            {RenderResourceGroup::WavefrontResources},
            RenderPassExecutionId::WavefrontStageMarker,
            "wavefrontSwapQueuesKernel",
            Dispatch(DispatchKind::Linear, "1 thread per depth")));
        passes.push_back(MakePass(
            "Wavefront Accumulate",
            RenderPassType::Compute,
            wavefrontMask | FeatureMask({RenderPassFeature::Reconstruction}),
            {RenderResourceGroup::FrameConstants, RenderResourceGroup::WavefrontResources},
            {RenderResourceGroup::FinalOutputResources,
             RenderResourceGroup::GBufferResources,
             RenderResourceGroup::ReconstructionAOVResources},
            RenderPassExecutionId::WavefrontStageMarker,
            "wavefrontAccumulateKernel",
            Dispatch(DispatchKind::Linear, "width * height")));
    } else {
        passes.push_back(MakePass(
            "Megakernel Integrate",
            RenderPassType::Compute,
            FeatureMask({RenderPassFeature::Reconstruction}),
            commonInputs,
            integrateOutputs,
            RenderPassExecutionId::MegakernelIntegrate,
            "pathtraceIntegrateKernel|pathtraceIntegrateHardwareKernel",
            Dispatch(DispatchKind::FullScreen, "render width * render height"),
            {"executionMode", "directLightMode", "restirGiMode", "pathGuidingMode", "restirPtMode"}));
    }

    if (settings.restirPtMode == RenderSettings::RestirPtMode::Research ||
        settings.restirPtMode == RenderSettings::RestirPtMode::ExperimentalPathReuse) {
        passes.push_back(MakePass(
            settings.restirPtMode == RenderSettings::RestirPtMode::ExperimentalPathReuse
                ? "ReSTIR PT Experimental Path Reuse Audit"
                : "ReSTIR PT Research Audit",
            RenderPassType::Host,
            FeatureMask({RenderPassFeature::Optional, RenderPassFeature::DebugOnly}),
            {RenderResourceGroup::FrameConstants,
             RenderResourceGroup::ReSTIRPTResources,
             RenderResourceGroup::DebugResources},
            {RenderResourceGroup::ReSTIRPTResources,
             RenderResourceGroup::DebugResources},
            RenderPassExecutionId::NoOp,
            "host debug audit",
            Dispatch(DispatchKind::External, "host metrics inspection"),
            {"restirPtMode", "restirPtMaxReservoirs", "restirPtDebug", "restirPtReuseStrength"},
            {RenderResourceGroup::ReSTIRPTResources}));
    }

    if (settings.restirDebugEnabled) {
        passes.push_back(MakePass(
            "ReSTIR Debug Inspector Audit",
            RenderPassType::Host,
            FeatureMask({RenderPassFeature::Optional, RenderPassFeature::DebugOnly}),
            {RenderResourceGroup::FrameConstants,
             RenderResourceGroup::ReSTIRDIResources,
             RenderResourceGroup::ReGIRResources,
             RenderResourceGroup::ReSTIRGIResources,
             RenderResourceGroup::ReSTIRPTResources,
             RenderResourceGroup::DenoiserResources,
             RenderResourceGroup::DebugResources},
            {RenderResourceGroup::DebugResources},
            RenderPassExecutionId::NoOp,
            "host debug inspector",
            Dispatch(DispatchKind::External, "host metrics inspection"),
            {"restirDebugEnabled", "restirDebugView", "restirDebugCounters"},
            {RenderResourceGroup::DebugResources}));
    }

    if (restirDiScreenPassesEnabled) {
        passes.push_back(MakePass(
            "ReSTIR DI Apply Lighting",
            RenderPassType::Compute,
            FeatureMask({RenderPassFeature::Optional, RenderPassFeature::UsesHistory}),
            {RenderResourceGroup::FrameConstants,
             RenderResourceGroup::ReSTIRDIVisibility,
             RenderResourceGroup::FinalOutputResources},
            {RenderResourceGroup::FinalOutputResources},
            RenderPassExecutionId::RestirDiApplyLighting,
            "restirDiApplyLightingKernel",
            Dispatch(DispatchKind::FullScreen, "render width * render height"),
            {"directLightMode"}));
    }

    passes.push_back(MakePass(
        "Path Tracer Present",
        RenderPassType::Compute,
        0u,
        {RenderResourceGroup::FinalOutputResources},
        {RenderResourceGroup::FinalOutputResources},
        RenderPassExecutionId::Present,
        "pathtracePresentKernel",
        Dispatch(DispatchKind::FullScreen, "render width * render height")));

    if (reconstructionConfidenceInputsEnabled) {
        std::vector<RenderResourceGroup> confidenceInputs = {
            RenderResourceGroup::FrameConstants,
            RenderResourceGroup::ReSTIRDIResources,
            RenderResourceGroup::ReGIRResources,
            RenderResourceGroup::ReSTIRGIResources,
            RenderResourceGroup::ReSTIRPTResources,
            RenderResourceGroup::PathGuidingResources,
            RenderResourceGroup::RadianceCacheResources,
            RenderResourceGroup::ReconstructionAOVResources,
        };
        std::vector<RenderResourceGroup> confidenceHistory = {
            RenderResourceGroup::ReSTIRDIResources,
            RenderResourceGroup::PathGuidingResources,
            RenderResourceGroup::RadianceCacheResources,
            RenderResourceGroup::ReconstructionAOVResources,
        };
        std::vector<std::string> confidenceSettings = {
            "directLightMode",
            "restirGiMode",
            "pathGuidingMode",
            "restirPtMode",
            "radianceCacheMode",
        };
        if (volumeTransportEnabled) {
            confidenceInputs.push_back(RenderResourceGroup::VolumeTransportResources);
            confidenceHistory.push_back(RenderResourceGroup::VolumeTransportResources);
            confidenceSettings.push_back("volumeTransportMode");
        }
        if (mlBackendEnabled) {
            confidenceInputs.push_back(RenderResourceGroup::MLBackendResources);
            confidenceHistory.push_back(RenderResourceGroup::MLBackendResources);
            confidenceSettings.push_back("mlBackendMode");
        }
        passes.push_back(MakePass(
            "Reconstruction Transport Confidence Inputs",
            RenderPassType::Host,
            FeatureMask({RenderPassFeature::Optional,
                         RenderPassFeature::Reconstruction,
                         RenderPassFeature::UsesHistory}),
            confidenceInputs,
            {RenderResourceGroup::ReconstructionAOVResources,
             RenderResourceGroup::ReconstructionHistoryResources},
            RenderPassExecutionId::NoOp,
            "host reconstruction confidence contract",
            Dispatch(DispatchKind::External, "transport confidence metadata inspection"),
            confidenceSettings,
            confidenceHistory));
    }

    if (mlBackendEnabled) {
        passes.push_back(MakePass(
            "ML Reconstruction Backend Audit",
            RenderPassType::Host,
            FeatureMask({RenderPassFeature::Optional,
                         RenderPassFeature::Reconstruction,
                         RenderPassFeature::MLBackend}),
            {RenderResourceGroup::FrameConstants,
             RenderResourceGroup::FinalOutputResources,
             RenderResourceGroup::ReconstructionAOVResources,
             RenderResourceGroup::ReconstructionHistoryResources,
             RenderResourceGroup::MLBackendResources},
            {RenderResourceGroup::MLBackendResources,
             RenderResourceGroup::ReconstructionHistoryResources,
             RenderResourceGroup::DebugResources},
            RenderPassExecutionId::MlBackendNoOp,
            "MLBackend.NoOp.Reconstruction",
            Dispatch(DispatchKind::External, "host capability/fallback audit; no tensor dispatch"),
            {"mlBackendMode", "mlModelPackagePath", "mlDebug"},
            {RenderResourceGroup::ReconstructionAOVResources,
             RenderResourceGroup::ReconstructionHistoryResources,
             RenderResourceGroup::MLBackendResources}));
    }

    if (settings.svgfDenoiseEnabled) {
        passes.push_back(MakePass(
            "SVGF Temporal Moments + A-Trous Filter",
            RenderPassType::Compute,
            FeatureMask({RenderPassFeature::Optional,
                         RenderPassFeature::Denoise,
                         RenderPassFeature::Reconstruction,
                         RenderPassFeature::UsesHistory,
                         RenderPassFeature::ProducesHistory}),
            {RenderResourceGroup::FinalOutputResources,
             RenderResourceGroup::GBufferResources,
             RenderResourceGroup::ReconstructionAOVResources,
             RenderResourceGroup::ReconstructionHistoryResources,
             RenderResourceGroup::DenoiserResources},
            {RenderResourceGroup::DenoiserResources,
             RenderResourceGroup::DenoiserScratchResources,
             RenderResourceGroup::ReconstructionHistoryResources},
            RenderPassExecutionId::SvgfDenoise,
            "svgfTemporalMomentsKernel + svgfAtrousFilterKernel",
            Dispatch(DispatchKind::FullScreen, "render width * render height"),
            {"svgfDenoiseEnabled", "svgfAtrousIterations"},
            {RenderResourceGroup::ReconstructionAOVResources,
             RenderResourceGroup::ReconstructionHistoryResources,
             RenderResourceGroup::DenoiserResources}));
    } else if (settings.denoiseEnabled) {
        passes.push_back(MakePass(
            "OIDN Denoise",
            RenderPassType::External,
            FeatureMask({RenderPassFeature::Optional,
                         RenderPassFeature::Denoise,
                         RenderPassFeature::Reconstruction}),
            {RenderResourceGroup::FinalOutputResources,
             RenderResourceGroup::GBufferResources,
             RenderResourceGroup::ReconstructionAOVResources},
            {RenderResourceGroup::DenoiserResources,
             RenderResourceGroup::ReconstructionHistoryResources},
            RenderPassExecutionId::OidnDenoise,
            "OIDN CPU denoise",
            Dispatch(DispatchKind::External, "CPU postprocess at denoise frequency"),
            {"denoiseEnabled", "denoiseFrequency"},
            {RenderResourceGroup::ReconstructionAOVResources,
             RenderResourceGroup::DenoiserResources}));
    }

    if (hasDisplayPass) {
        passes.push_back(MakePass(
            "Display Quad",
            RenderPassType::Render,
            0u,
            {RenderResourceGroup::FinalOutputResources,
             RenderResourceGroup::DenoiserResources,
             RenderResourceGroup::ReconstructionHistoryResources},
            {RenderResourceGroup::FinalOutputResources},
            RenderPassExecutionId::Display,
            "Display Quad",
            Dispatch(DispatchKind::FullScreen, "drawable extent")));
    }

    return graph;
}

RenderPassGraphCompileResult CompileRenderPassGraph(const RenderPassGraphDesc& graph) {
    RenderPassGraphCompileResult result{};
    result.resourcePlans = BuildResourceLifetimePlan(
        graph,
        result.usedResourceCount,
        result.aliasableTransientResourceCount,
        result.transientAliasPoolCount);
    result.resourceTransitions = BuildResourceTransitionPlan(
        graph,
        result.resourceTransitionCount,
        result.requiredBarrierCount,
        result.writeTransitionCount);
    if (graph.version != 2u) {
        result.errors.push_back("unsupported render pass graph version " + std::to_string(graph.version));
    }
    if (graph.resources.empty()) {
        result.errors.push_back("render pass graph declares no resources");
    }

    for (size_t i = 0; i < graph.resources.size(); ++i) {
        const FrameResourceDesc& resource = graph.resources[i];
        if (resource.name.empty()) {
            result.warnings.push_back(
                "resource " + std::to_string(i) + " has no name");
        }
        if (resource.debugLabel.empty()) {
            result.warnings.push_back(
                resource.name + " has no resource debug label");
        }
        if ((resource.lifetime == ResourceLifetime::History ||
             resource.historyPolicy == HistoryPolicy::CurrentPrevious ||
             resource.historyPolicy == HistoryPolicy::Persistent) &&
            resource.invalidatesOn.empty()) {
            result.warnings.push_back(
                resource.name + " has history lifetime/policy without invalidation reasons");
        }
        if (resource.lifetime == ResourceLifetime::Transient &&
            i < result.resourcePlans.size() &&
            !result.resourcePlans[i].used) {
            result.warnings.push_back(resource.name + " is transient but unused by this graph");
        }
        for (size_t j = i + 1u; j < graph.resources.size(); ++j) {
            if (graph.resources[j].group == resource.group) {
                result.errors.push_back(
                    "duplicate resource group " + std::string(RenderResourceGroupLabel(resource.group)));
            }
        }
    }

    if (graph.passes.empty()) {
        result.errors.push_back("render pass graph declares no passes");
    }

    for (size_t i = 0; i < graph.passes.size(); ++i) {
        const RenderPassDesc& pass = graph.passes[i];
        result.passOrder.push_back(static_cast<uint32_t>(i));
        if (IsExecutablePass(pass.executionId)) {
            result.executablePassCount += 1u;
        } else {
            result.skippedPassCount += 1u;
        }

        if (pass.name.empty()) {
            result.errors.push_back("pass " + std::to_string(i) + " has no name");
        }
        if (pass.pipelineKey.empty()) {
            result.warnings.push_back(pass.name + " has no pipeline key");
        }
        if (pass.debugLabel.empty()) {
            result.warnings.push_back(pass.name + " has no debug label");
        }
        if (pass.timingLabel.empty()) {
            result.warnings.push_back(pass.name + " has no timing label");
        }
        if (pass.dispatch.kind == DispatchKind::None) {
            result.warnings.push_back(pass.name + " has no dispatch descriptor");
        }
        if (pass.dispatch.dimensions.empty()) {
            result.warnings.push_back(pass.name + " has no dispatch dimensions");
        }
        if (pass.debugPolicy == DebugCapturePolicy::Scoped && pass.debugLabel.empty()) {
            result.errors.push_back(pass.name + " requests scoped debug capture without a debug label");
        }
        if (pass.timingPolicy != TimingPolicy::None && pass.timingLabel.empty()) {
            result.errors.push_back(pass.name + " requests timing without a timing label");
        }
        if (IsExecutablePass(pass.executionId) && pass.debugPolicy == DebugCapturePolicy::None) {
            result.warnings.push_back(pass.name + " is executable but has debug capture disabled");
        }
        if (IsExecutablePass(pass.executionId) && pass.timingPolicy == TimingPolicy::None) {
            result.warnings.push_back(pass.name + " is executable but has timing disabled");
        }

        for (RenderResourceGroup input : pass.inputs) {
            if (!HasDeclaredResource(graph.resources, input)) {
                result.errors.push_back(
                    pass.name + " input references undeclared resource group " +
                    RenderResourceGroupLabel(input));
            }
        }
        for (RenderResourceGroup output : pass.outputs) {
            if (!HasDeclaredResource(graph.resources, output)) {
                result.errors.push_back(
                    pass.name + " output references undeclared resource group " +
                    RenderResourceGroupLabel(output));
            }
        }
        for (const ResourceUse& read : pass.reads) {
            ValidateResourceUse(pass, read, graph.resources, "read", result.errors, result.warnings);
            if (!HasGroup(pass.inputs, read.group)) {
                result.warnings.push_back(
                    pass.name + " read use is not mirrored in legacy inputs for " +
                    RenderResourceGroupLabel(read.group));
            }
        }
        for (const ResourceUse& write : pass.writes) {
            ValidateResourceUse(pass, write, graph.resources, "write", result.errors, result.warnings);
            if (!HasGroup(pass.outputs, write.group)) {
                result.warnings.push_back(
                    pass.name + " write use is not mirrored in legacy outputs for " +
                    RenderResourceGroupLabel(write.group));
            }
        }
        for (RenderResourceGroup historyDependency : pass.historyDependencies) {
            if (!HasDeclaredResource(graph.resources, historyDependency)) {
                result.errors.push_back(
                    pass.name + " history dependency references undeclared resource group " +
                    RenderResourceGroupLabel(historyDependency));
            }
        }
    }

    result.valid = result.errors.empty();
    return result;
}

RenderPassExecutionResult ExecuteRenderPassGraph(
    const RenderPassGraphDesc& graph,
    const std::function<bool(const RenderPassDesc&)>& executePass) {
    RenderPassExecutionResult result{};
    RenderPassGraphCompileResult compileResult = CompileRenderPassGraph(graph);
    result.compiled = compileResult.valid;
    result.compiledExecutablePassCount = compileResult.executablePassCount;
    result.compiledSkippedPassCount = compileResult.skippedPassCount;
    result.compiledUsedResourceCount = compileResult.usedResourceCount;
    result.compiledAliasableTransientResourceCount = compileResult.aliasableTransientResourceCount;
    result.compiledTransientAliasPoolCount = compileResult.transientAliasPoolCount;
    result.compiledResourceTransitionCount = compileResult.resourceTransitionCount;
    result.compiledRequiredBarrierCount = compileResult.requiredBarrierCount;
    result.compiledWriteTransitionCount = compileResult.writeTransitionCount;
    result.resourcePlans = std::move(compileResult.resourcePlans);
    result.resourceTransitions = std::move(compileResult.resourceTransitions);
    result.compileWarnings = std::move(compileResult.warnings);
    result.compileErrors = std::move(compileResult.errors);
    if (!result.compiled) {
        return result;
    }
    if (!executePass) {
        for (uint32_t passIndex : compileResult.passOrder) {
            const RenderPassDesc& pass = graph.passes[passIndex];
            if (IsExecutablePass(pass.executionId)) {
                result.failedPasses.push_back(pass.name);
            } else {
                result.skippedPasses.push_back(pass.name);
            }
        }
        return result;
    }

    for (uint32_t passIndex : compileResult.passOrder) {
        const RenderPassDesc& pass = graph.passes[passIndex];
        if (!IsExecutablePass(pass.executionId)) {
            result.skippedPasses.push_back(pass.name);
            continue;
        }
        if (executePass(pass)) {
            result.executedPasses.push_back(pass.name);
        } else {
            result.failedPasses.push_back(pass.name);
        }
    }
    return result;
}

}  // namespace PathTracer
