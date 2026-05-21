#import "renderer/TransportMemoryAllocator.h"

#import <Metal/Metal.h>

namespace PathTracer {

namespace {

MTLResourceOptions MetalResourceOptionsFor(StorageClass storage) {
    switch (storage) {
        case StorageClass::Private:
            return MTLResourceStorageModePrivate;
        case StorageClass::Managed:
#if TARGET_OS_OSX
            return MTLResourceStorageModeManaged;
#else
            return MTLResourceStorageModeShared;
#endif
        case StorageClass::Shared:
        case StorageClass::Unknown:
        case StorageClass::External:
        case StorageClass::Memoryless:
            return MTLResourceStorageModeShared;
    }
    return MTLResourceStorageModeShared;
}

TransportMemoryStorageClass TransportStorageClassFor(StorageClass storage) {
    switch (storage) {
        case StorageClass::Private:
            return TransportMemoryStorageClass::Private;
        case StorageClass::Managed:
            return TransportMemoryStorageClass::Managed;
        case StorageClass::External:
            return TransportMemoryStorageClass::External;
        case StorageClass::Shared:
        case StorageClass::Unknown:
        case StorageClass::Memoryless:
            return TransportMemoryStorageClass::Shared;
    }
    return TransportMemoryStorageClass::Shared;
}

}  // namespace

void TransportMemoryAllocator::clear() {
    m_records.clear();
    ++m_generation;
}

MTLBufferHandle TransportMemoryAllocator::allocateBuffer(MTLDeviceHandle device,
                                                         TransportMemoryRecord record,
                                                         StorageClass storage,
                                                         const char* label) {
    if (!device || record.bytes == 0u) {
        return nullptr;
    }

    id<MTLBuffer> buffer =
        [device newBufferWithLength:static_cast<NSUInteger>(record.bytes)
                            options:MetalResourceOptionsFor(storage)];
    if (!buffer) {
        return nullptr;
    }
    if (label && label[0] != '\0') {
        buffer.label = [NSString stringWithUTF8String:label];
    }

    record.ownedBuffer = buffer;
    record.storage = TransportStorageClassFor(storage);
    record.targetStorage = record.storage;
    record.transportMemoryOwned = true;
    record.allocationGeneration = m_generation;
    m_records.push_back(std::move(record));
    return buffer;
}

MTLBufferHandle TransportMemoryAllocator::bufferForName(const std::string& name) const {
    const TransportMemoryRecord* record = recordForName(name);
    return record ? record->ownedBuffer : nullptr;
}

const TransportMemoryRecord* TransportMemoryAllocator::recordForName(const std::string& name) const {
    for (const auto& record : m_records) {
        if (record.name == name) {
            return &record;
        }
    }
    return nullptr;
}

}  // namespace PathTracer
