#pragma once

#include "renderer/TransportMemoryRegistry.h"

#include <string>
#include <vector>

namespace PathTracer {

class TransportMemoryAllocator {
public:
    void clear();

    MTLBufferHandle allocateBuffer(MTLDeviceHandle device,
                                   TransportMemoryRecord record,
                                   StorageClass storage,
                                   const char* label);

    MTLBufferHandle bufferForName(const std::string& name) const;
    const TransportMemoryRecord* recordForName(const std::string& name) const;
    const std::vector<TransportMemoryRecord>& records() const { return m_records; }
    uint32_t generation() const { return m_generation; }

private:
    std::vector<TransportMemoryRecord> m_records;
    uint32_t m_generation = 0;
};

}  // namespace PathTracer
