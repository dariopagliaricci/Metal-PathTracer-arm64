#pragma once

#include "renderer/MetalHandles.h"
#include "renderer/RenderPassGraph.h"

#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace PathTracer {

enum class TransportMemoryKind : uint32_t {
    RestirDIReservoir = 0,
    RestirGIReservoir,
    RestirPTReservoir,
    PathGuidingField,
    RadianceCache,
    VisibilityCache,
    DenoiserHistory,
    VarianceConfidence,
    ReprojectionValidityMask,
    DebugReadback,
    WavefrontQueue,
    CausticPathSpace,
    MLBackendTensor,
    Placeholder,
};

enum class TransportMemoryOwner : uint32_t {
    RestirDI = 0,
    RestirGI,
    RestirPT,
    PathGuiding,
    RadianceCache,
    VisibilityCache,
    Denoiser,
    FrameHistory,
    Debug,
    Wavefront,
    CausticTransport,
    MLBackend,
};

enum class TransportMemoryStorageClass : uint32_t {
    Private = 0,
    Shared,
    Managed,
    Staging,
    External,
};

enum class TransportMemoryHistoryPolicy : uint32_t {
    PersistentHistory = 0,
    CurrentFrame,
    Transient,
    Placeholder,
};

enum class TransportMemoryDebugReadbackPolicy : uint32_t {
    None = 0,
    MetadataOnly,
    StagingCopy,
    DirectCpuVisible,
};

using TransportMemoryInvalidationMask = uint32_t;

struct TransportConfidenceState {
    float confidence = 0.0f;
    uint32_t sampleCount = 0;
    float variance = 0.0f;
    uint32_t age = 0;
    HistoryResetReason lastInvalidationReason = HistoryResetReason::None;
};

struct TransportMemoryRecord {
    std::string name;
    TransportMemoryKind kind = TransportMemoryKind::Placeholder;
    TransportMemoryOwner owner = TransportMemoryOwner::FrameHistory;
    TransportMemoryOwner targetOwner = TransportMemoryOwner::FrameHistory;
    RenderResourceGroup frameGraphGroup = RenderResourceGroup::FrameConstants;
    ResourceLifetime lifetime = ResourceLifetime::Transient;
    MTLBufferHandle ownedBuffer = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depthOrCellCount = 0;
    uint64_t bytes = 0;
    TransportMemoryStorageClass storage = TransportMemoryStorageClass::Private;
    TransportMemoryStorageClass targetStorage = TransportMemoryStorageClass::Private;
    TransportMemoryHistoryPolicy history = TransportMemoryHistoryPolicy::PersistentHistory;
    TransportMemoryDebugReadbackPolicy debug = TransportMemoryDebugReadbackPolicy::None;
    TransportMemoryInvalidationMask invalidatesOn = 0;
    TransportConfidenceState confidence{};
    HistoryResetReason lastInvalidationReason = HistoryResetReason::None;
    uint32_t invalidationCount = 0;
    std::string byteSizeFormula;
    std::string featureGate;
    std::string migrationRisk;
    bool hotPath = false;
    bool temporaryShared = false;
    bool frameGraphOwned = false;
    bool transportMemoryOwned = false;
    uint32_t allocationGeneration = 0;
};

inline TransportMemoryInvalidationMask TransportInvalidationBit(HistoryResetReason reason) {
    const uint32_t index = static_cast<uint32_t>(reason);
    return index < 32u ? (1u << index) : 0u;
}

inline TransportMemoryInvalidationMask TransportInvalidationMaskFor(
    std::initializer_list<HistoryResetReason> reasons) {
    TransportMemoryInvalidationMask mask = 0u;
    for (HistoryResetReason reason : reasons) {
        mask |= TransportInvalidationBit(reason);
    }
    return mask;
}

inline bool TransportMemoryInvalidatesOn(TransportMemoryInvalidationMask mask,
                                         HistoryResetReason reason) {
    return (mask & TransportInvalidationBit(reason)) != 0u;
}

inline const char* TransportMemoryKindLabel(TransportMemoryKind kind) {
    switch (kind) {
        case TransportMemoryKind::RestirDIReservoir:
            return "restir_di_reservoir";
        case TransportMemoryKind::RestirGIReservoir:
            return "restir_gi_reservoir";
        case TransportMemoryKind::RestirPTReservoir:
            return "restir_pt_reservoir";
        case TransportMemoryKind::PathGuidingField:
            return "path_guiding_field";
        case TransportMemoryKind::RadianceCache:
            return "radiance_cache";
        case TransportMemoryKind::VisibilityCache:
            return "visibility_cache";
        case TransportMemoryKind::DenoiserHistory:
            return "denoiser_history";
        case TransportMemoryKind::VarianceConfidence:
            return "variance_confidence";
        case TransportMemoryKind::ReprojectionValidityMask:
            return "reprojection_validity_mask";
        case TransportMemoryKind::DebugReadback:
            return "debug_readback";
        case TransportMemoryKind::WavefrontQueue:
            return "wavefront_queue";
        case TransportMemoryKind::CausticPathSpace:
            return "caustic_path_space";
        case TransportMemoryKind::MLBackendTensor:
            return "ml_backend_tensor";
        case TransportMemoryKind::Placeholder:
            return "placeholder";
    }
    return "unknown";
}

inline const char* TransportMemoryOwnerLabel(TransportMemoryOwner owner) {
    switch (owner) {
        case TransportMemoryOwner::RestirDI:
            return "restir_di";
        case TransportMemoryOwner::RestirGI:
            return "restir_gi";
        case TransportMemoryOwner::RestirPT:
            return "restir_pt";
        case TransportMemoryOwner::PathGuiding:
            return "path_guiding";
        case TransportMemoryOwner::RadianceCache:
            return "radiance_cache";
        case TransportMemoryOwner::VisibilityCache:
            return "visibility_cache";
        case TransportMemoryOwner::Denoiser:
            return "denoiser";
        case TransportMemoryOwner::FrameHistory:
            return "frame_history";
        case TransportMemoryOwner::Debug:
            return "debug";
        case TransportMemoryOwner::Wavefront:
            return "wavefront";
        case TransportMemoryOwner::CausticTransport:
            return "caustic_transport";
        case TransportMemoryOwner::MLBackend:
            return "ml_backend";
    }
    return "unknown";
}

inline const char* TransportMemoryStorageClassLabel(TransportMemoryStorageClass storage) {
    switch (storage) {
        case TransportMemoryStorageClass::Private:
            return "private";
        case TransportMemoryStorageClass::Shared:
            return "shared";
        case TransportMemoryStorageClass::Managed:
            return "managed";
        case TransportMemoryStorageClass::Staging:
            return "staging";
        case TransportMemoryStorageClass::External:
            return "external";
    }
    return "unknown";
}

inline const char* TransportMemoryHistoryPolicyLabel(TransportMemoryHistoryPolicy history) {
    switch (history) {
        case TransportMemoryHistoryPolicy::PersistentHistory:
            return "persistent_history";
        case TransportMemoryHistoryPolicy::CurrentFrame:
            return "current_frame";
        case TransportMemoryHistoryPolicy::Transient:
            return "transient";
        case TransportMemoryHistoryPolicy::Placeholder:
            return "placeholder";
    }
    return "unknown";
}

inline const char* TransportMemoryDebugReadbackPolicyLabel(
    TransportMemoryDebugReadbackPolicy debug) {
    switch (debug) {
        case TransportMemoryDebugReadbackPolicy::None:
            return "none";
        case TransportMemoryDebugReadbackPolicy::MetadataOnly:
            return "metadata_only";
        case TransportMemoryDebugReadbackPolicy::StagingCopy:
            return "staging_copy";
        case TransportMemoryDebugReadbackPolicy::DirectCpuVisible:
            return "direct_cpu_visible";
    }
    return "unknown";
}

class TransportMemoryRegistry {
public:
    void clear() {
        m_records.clear();
        m_invalidatedRecordCount = 0u;
    }

    void registerResource(TransportMemoryRecord record) {
        if (m_lastInvalidationReason != HistoryResetReason::None &&
            TransportMemoryInvalidatesOn(record.invalidatesOn, m_lastInvalidationReason)) {
            record.lastInvalidationReason = m_lastInvalidationReason;
            record.confidence.lastInvalidationReason = m_lastInvalidationReason;
            record.invalidationCount = 1u;
        }
        m_records.push_back(std::move(record));
        rebuildSummary();
    }

    void noteInvalidation(HistoryResetReason reason) {
        if (reason == HistoryResetReason::None) {
            m_lastInvalidationReason = reason;
            m_invalidatedRecordCount = 0u;
            return;
        }
        m_lastInvalidationReason = reason;
        m_invalidatedRecordCount = 0u;
        for (auto& record : m_records) {
            if (TransportMemoryInvalidatesOn(record.invalidatesOn, reason)) {
                record.lastInvalidationReason = reason;
                record.confidence.lastInvalidationReason = reason;
                record.invalidationCount += 1u;
                m_invalidatedRecordCount += 1u;
            }
        }
    }

    uint64_t totalBytes() const { return m_totalBytes; }
    uint64_t privateBytes() const { return m_privateBytes; }
    uint64_t sharedBytes() const { return m_sharedBytes; }
    uint64_t stagingBytes() const { return m_stagingBytes; }
    uint32_t recordCount() const { return static_cast<uint32_t>(m_records.size()); }
    uint32_t invalidatedRecordCount() const { return m_invalidatedRecordCount; }
    HistoryResetReason lastInvalidationReason() const { return m_lastInvalidationReason; }
    const std::vector<TransportMemoryRecord>& records() const { return m_records; }

    const TransportMemoryRecord* findRecord(const std::string& name) const {
        for (const auto& record : m_records) {
            if (record.name == name) {
                return &record;
            }
        }
        return nullptr;
    }

    MTLBufferHandle bufferForName(const std::string& name) const {
        const TransportMemoryRecord* record = findRecord(name);
        return record ? record->ownedBuffer : nullptr;
    }

private:
    void rebuildSummary() {
        m_totalBytes = 0u;
        m_privateBytes = 0u;
        m_sharedBytes = 0u;
        m_stagingBytes = 0u;
        m_invalidatedRecordCount = 0u;
        for (const auto& record : m_records) {
            m_totalBytes += record.bytes;
            switch (record.storage) {
                case TransportMemoryStorageClass::Private:
                    m_privateBytes += record.bytes;
                    break;
                case TransportMemoryStorageClass::Shared:
                    m_sharedBytes += record.bytes;
                    break;
                case TransportMemoryStorageClass::Staging:
                    m_stagingBytes += record.bytes;
                    break;
                case TransportMemoryStorageClass::Managed:
                case TransportMemoryStorageClass::External:
                    break;
            }
            if (record.lastInvalidationReason != HistoryResetReason::None) {
                m_invalidatedRecordCount += 1u;
            }
        }
    }

    std::vector<TransportMemoryRecord> m_records;
    uint64_t m_totalBytes = 0u;
    uint64_t m_privateBytes = 0u;
    uint64_t m_sharedBytes = 0u;
    uint64_t m_stagingBytes = 0u;
    uint32_t m_invalidatedRecordCount = 0u;
    HistoryResetReason m_lastInvalidationReason = HistoryResetReason::None;
};

}  // namespace PathTracer
