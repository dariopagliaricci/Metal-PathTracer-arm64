#pragma once

#include "renderer/TransportMemoryRegistry.h"

#include <initializer_list>
#include <string>
#include <vector>

namespace PathTracer {

class TransportMemoryAllocator {
public:
    void clear();
    void releaseOwner(TransportMemoryOwner owner);
    void releaseOwners(std::initializer_list<TransportMemoryOwner> owners);

    MTLBufferHandle allocateBuffer(MTLDeviceHandle device,
                                   TransportMemoryRecord record,
                                   StorageClass storage,
                                   const char* label);

    MTLBufferHandle bufferForName(const std::string& name) const;
    const TransportMemoryRecord* recordForName(const std::string& name) const;
    bool ownsBufferNamed(const std::string& name) const { return recordForName(name) != nullptr; }
    const std::vector<TransportMemoryRecord>& records() const { return m_records; }
    uint32_t generation() const { return m_generation; }

private:
    std::vector<TransportMemoryRecord> m_records;
    uint32_t m_generation = 0;
};

}  // namespace PathTracer
