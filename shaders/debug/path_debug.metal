struct HitRecord {
    float3 point;
    float3 normal;
    float3 shadingNormal;
    float paddingNormal;
    float t;
    uint materialIndex;
    uint twoSided;
    uint frontFace;
    uint primitiveType;
    uint primitiveIndex;
    uint meshIndex;
    float2 barycentric;
    uint padding;
};

struct HardwareShadowTraceAudit {
    bool rawResultAvailable = false;
    bool hwTraceUsed = false;
    uint rawResultType = kInvalidIndex;
    uint rawInstanceId = kInvalidIndex;
    uint rawPrimitiveId = kInvalidIndex;
    float rawDistance = 0.0f;
    bool resolvedHit = false;
    uint resolvedMeshIndex = kInvalidIndex;
    uint resolvedMaterialIndex = kInvalidIndex;
    uint resolvedPrimitiveIndex = kInvalidIndex;
    uint resolvedPrimitiveType = kInvalidIndex;
    bool resolvedFrontFace = false;
    float resolvedDistance = 0.0f;
    uint rejectionMask = 0u;
};

struct SoftwareShadowTraceAudit {
    bool consulted = false;
    bool hit = false;
    uint softwareBvhType = kInvalidIndex;
    bool hasTlas = false;
    bool hasBlas = false;
    bool hasInstanceInfo = false;
    uint instanceIndex = kInvalidIndex;
    uint meshIndex = kInvalidIndex;
    uint materialIndex = kInvalidIndex;
    uint primitiveIndex = kInvalidIndex;
    uint primitiveType = kInvalidIndex;
    bool frontFace = false;
    float distance = 0.0f;
    float3 hitPosition = float3(0.0f);
    uint blasRootNodeOffset = 0u;
    uint blasPrimIndexOffset = 0u;
    uint triangleBaseOffset = 0u;
};

inline void compute_exclusion_indices(thread const HitRecord& rec,
                                      thread uint& meshIndexOut,
                                      thread uint& primitiveIndexOut) {
    if (rec.primitiveType == kPrimitiveTypeTriangle) {
        meshIndexOut = rec.meshIndex;
        primitiveIndexOut = rec.primitiveIndex;
    } else {
        meshIndexOut = kInvalidIndex;
        primitiveIndexOut = kInvalidIndex;
    }
}

struct TraversalCounters;

struct PathtraceDebugContext {
    device PathtraceDebugBuffer* buffer;
    uint integrator;
    uint pixelX;
    uint pixelY;
    uint sampleIndex;
    uint maxEntries;
    uint active;
};

#if PT_DEBUG_TOOLS
inline PathtraceDebugContext make_debug_context(constant PathtraceUniforms& uniforms,
                                                device PathtraceDebugBuffer* buffer,
                                                uint2 gid,
                                                uint sampleIndex,
                                                uint integrator) {
    PathtraceDebugContext ctx;
    ctx.buffer = buffer;
    ctx.integrator = integrator;
    ctx.pixelX = gid.x;
    ctx.pixelY = gid.y;
    ctx.sampleIndex = sampleIndex;
    ctx.maxEntries = (buffer != nullptr)
        ? min(uniforms.debugMaxEntries, kPathtraceDebugMaxEntries)
        : 0u;
    bool active = (buffer != nullptr) &&
                  (uniforms.debugPathActive != 0u) &&
                  (ctx.maxEntries > 0u) &&
                  (gid.x == uniforms.debugPixelX) &&
                  (gid.y == uniforms.debugPixelY);
    ctx.active = active ? 1u : 0u;
    return ctx;
}

inline void record_debug_event(thread const PathtraceDebugContext& ctx,
                               uint eventKind,
                               uint depth,
                               uint mediumDepthBefore,
                               uint mediumDepthAfter,
                               int mediumEvent,
                               uint frontFace,
                               uint materialIndex,
                               bool scatterIsDelta,
                               float3 throughput,
                               float4 scalars,
                               float3 aux) {
    if (!ctx.active || ctx.buffer == nullptr) {
        return;
    }
    uint allowed = min(ctx.maxEntries, ctx.buffer->maxEntries);
    if (allowed == 0u) {
        return;
    }
    uint writeIndex = atomic_fetch_add_explicit(&ctx.buffer->writeIndex, 1u, memory_order_relaxed);
    if (writeIndex >= allowed) {
        return;
    }

    PathtraceDebugEntry entry;
    entry.integrator = ctx.integrator;
    entry.pixelX = ctx.pixelX;
    entry.pixelY = ctx.pixelY;
    entry.sampleIndex = ctx.sampleIndex;
    entry.depth = depth;
    entry.mediumDepthBefore = mediumDepthBefore;
    entry.mediumDepthAfter = mediumDepthAfter;
    entry.mediumEvent = mediumEvent;
    entry.frontFace = frontFace;
    entry.scatterIsDelta = scatterIsDelta ? 1u : 0u;
    entry.materialIndex = materialIndex;
    entry.eventKind = eventKind;
    entry.scalars = scalars;
    entry.throughput = float4(throughput, 0.0f);
    entry.aux = float4(aux, 0.0f);
    ctx.buffer->entries[writeIndex] = entry;
}

inline void record_debug_event(thread const PathtraceDebugContext& ctx,
                               uint depth,
                               uint mediumDepthBefore,
                               uint mediumDepthAfter,
                               int mediumEvent,
                               uint frontFace,
                               uint materialIndex,
                               bool scatterIsDelta,
                               float3 throughput) {
    record_debug_event(ctx,
                       kDebugEventScatter,
                       depth,
                       mediumDepthBefore,
                       mediumDepthAfter,
                       mediumEvent,
                       frontFace,
                       materialIndex,
                       scatterIsDelta,
                       throughput,
                       float4(0.0f),
                       float3(0.0f));
}

inline void record_shadow_path_audit(thread const PathtraceDebugContext& ctx,
                                     uint depth,
                                     uint mediumDepth,
                                     thread const HardwareShadowTraceAudit& hwAudit,
                                     thread const SoftwareShadowTraceAudit& swAudit,
                                     uint lightIndex,
                                     int lightAuditKind,
                                     float shadowTMin,
                                     float shadowTMax,
                                     bool lightTwoSided,
                                     bool finalVisible) {
    record_debug_event(ctx,
                       kDebugEventDirectLightAuditShadowRaw,
                       depth,
                       mediumDepth,
                       mediumDepth,
                       /*mediumEvent=*/lightAuditKind,
                       /*frontFace=*/0u,
                       /*materialIndex=*/0u,
                       /*scatterIsDelta=*/false,
                       float3(shadowTMin, shadowTMax, hwAudit.rawDistance),
                       float4(float(lightIndex),
                              hwAudit.rawResultAvailable ? 1.0f : 0.0f,
                              float(hwAudit.rawResultType),
                              hwAudit.hwTraceUsed ? 1.0f : 0.0f),
                       float3(float(hwAudit.rawInstanceId),
                              float(hwAudit.rawPrimitiveId),
                              lightTwoSided ? 1.0f : 0.0f));
    record_debug_event(ctx,
                       kDebugEventDirectLightAuditShadowMapped,
                       depth,
                       mediumDepth,
                       mediumDepth,
                       /*mediumEvent=*/lightAuditKind,
                       /*frontFace=*/0u,
                       /*materialIndex=*/0u,
                       /*scatterIsDelta=*/false,
                       float3(hwAudit.resolvedDistance,
                              kRayOriginEpsilon,
                              kHardwareOcclusionEpsilon),
                       float4(float(lightIndex),
                              hwAudit.resolvedHit ? 1.0f : 0.0f,
                              float(hwAudit.resolvedMeshIndex),
                              float(hwAudit.resolvedPrimitiveIndex)),
                       float3(float(hwAudit.resolvedMaterialIndex),
                              float(hwAudit.resolvedPrimitiveType),
                              hwAudit.resolvedFrontFace ? 1.0f : 0.0f));
    record_debug_event(ctx,
                       kDebugEventDirectLightAuditShadowEmbeddedSw,
                       depth,
                       mediumDepth,
                       mediumDepth,
                       /*mediumEvent=*/lightAuditKind,
                       /*frontFace=*/0u,
                       /*materialIndex=*/0u,
                       /*scatterIsDelta=*/false,
                       float3(swAudit.distance,
                              swAudit.hit ? 1.0f : 0.0f,
                              float(swAudit.softwareBvhType)),
                       float4(float(lightIndex),
                              swAudit.consulted ? 1.0f : 0.0f,
                              float(swAudit.instanceIndex),
                              float(swAudit.meshIndex)),
                       float3(float(swAudit.materialIndex),
                              float(swAudit.primitiveIndex),
                              float(swAudit.primitiveType)));
    record_debug_event(ctx,
                       kDebugEventDirectLightAuditShadowBinding,
                       depth,
                       mediumDepth,
                       mediumDepth,
                       /*mediumEvent=*/lightAuditKind,
                       /*frontFace=*/0u,
                       /*materialIndex=*/0u,
                       /*scatterIsDelta=*/false,
                       float3(float(swAudit.blasRootNodeOffset),
                              float(swAudit.blasPrimIndexOffset),
                              float(swAudit.triangleBaseOffset)),
                       float4(float(lightIndex),
                              swAudit.hasTlas ? 1.0f : 0.0f,
                              swAudit.hasBlas ? 1.0f : 0.0f,
                              swAudit.hasInstanceInfo ? 1.0f : 0.0f),
                       float3(swAudit.hitPosition.x,
                              swAudit.hitPosition.y,
                              swAudit.hitPosition.z));
    record_debug_event(ctx,
                       kDebugEventDirectLightAuditShadowReject,
                       depth,
                       mediumDepth,
                       mediumDepth,
                       /*mediumEvent=*/lightAuditKind,
                       /*frontFace=*/0u,
                       /*materialIndex=*/0u,
                       /*scatterIsDelta=*/false,
                       float3(float(hwAudit.rejectionMask),
                              finalVisible ? 1.0f : 0.0f,
                              swAudit.frontFace ? 1.0f : 0.0f),
                       float4(float(lightIndex),
                              hwAudit.rawResultAvailable ? 1.0f : 0.0f,
                              hwAudit.resolvedHit ? 1.0f : 0.0f,
                              swAudit.consulted ? 1.0f : 0.0f),
                       float3(swAudit.frontFace ? 1.0f : 0.0f,
                              0.0f,
                              0.0f));
}

inline HitRecord make_empty_hit_record() {
    HitRecord rec;
    rec.point = float3(0.0f);
    rec.normal = float3(0.0f);
    rec.shadingNormal = float3(0.0f);
    rec.paddingNormal = 0.0f;
    rec.t = 0.0f;
    rec.materialIndex = 0u;
    rec.twoSided = 0u;
    rec.frontFace = 0u;
    rec.primitiveType = kPrimitiveTypeNone;
    rec.primitiveIndex = kInvalidIndex;
    rec.meshIndex = kInvalidIndex;
    rec.barycentric = float2(0.0f);
    rec.padding = 0u;
    return rec;
}

inline void record_parity_entry(thread const PathtraceDebugContext& ctx,
                                constant PathtraceUniforms& uniforms,
                                uint depth,
                                uint probeKind,
                                thread const Ray& ray,
                                float tMin,
                                float tMax,
                                bool hwHit,
                                thread const HitRecord& hwRec,
                                bool swHit,
                                thread const HitRecord& swRec,
                                uint reasonMask) {
    if (ctx.buffer == nullptr) {
        return;
    }
    uint allowed = min(ctx.buffer->parityMaxEntries, kPathtraceParityMaxEntries);
    if (allowed == 0u) {
        return;
    }
    uint writeIndex =
        atomic_fetch_add_explicit(&ctx.buffer->parityWriteIndex, 1u, memory_order_relaxed);
    if (writeIndex >= allowed) {
        return;
    }

    PathtraceParityEntry entry;
    entry.frameIndex = uniforms.frameIndex;
    entry.pixelX = ctx.pixelX;
    entry.pixelY = ctx.pixelY;
    entry.depth = depth;
    entry.probeKind = probeKind;
    entry.reasonMask = reasonMask;
    entry.hwHit = hwHit ? 1u : 0u;
    entry.swHit = swHit ? 1u : 0u;
    entry.hwFrontFace = hwRec.frontFace;
    entry.swFrontFace = swRec.frontFace;
    entry.hwMaterialIndex = hwRec.materialIndex;
    entry.swMaterialIndex = swRec.materialIndex;
    entry.hwMeshIndex = hwRec.meshIndex;
    entry.swMeshIndex = swRec.meshIndex;
    entry.hwPrimitiveIndex = hwRec.primitiveIndex;
    entry.swPrimitiveIndex = swRec.primitiveIndex;
    entry.hwT = hwRec.t;
    entry.swT = swRec.t;
    entry.tMin = tMin;
    entry.tMax = tMax;
    entry.hwBarycentric = hwRec.barycentric;
    entry.swBarycentric = swRec.barycentric;
    entry.rayOrigin = float4(ray.origin, 0.0f);
    entry.rayDirection = float4(ray.direction, 0.0f);
    entry.hwNormal = float4(hwRec.normal, 0.0f);
    entry.swNormal = float4(swRec.normal, 0.0f);
    entry.hwShadingNormal = float4(hwRec.shadingNormal, 0.0f);
    entry.swShadingNormal = float4(swRec.shadingNormal, 0.0f);
    ctx.buffer->parityEntries[writeIndex] = entry;
}
#else
inline PathtraceDebugContext make_debug_context(constant PathtraceUniforms& uniforms,
                                                device PathtraceDebugBuffer* buffer,
                                                uint2 gid,
                                                uint sampleIndex,
                                                uint integrator) {
    (void)uniforms;
    PathtraceDebugContext ctx;
    ctx.buffer = nullptr;
    ctx.integrator = integrator;
    ctx.pixelX = gid.x;
    ctx.pixelY = gid.y;
    ctx.sampleIndex = sampleIndex;
    ctx.maxEntries = 0u;
    ctx.active = 0u;
    return ctx;
}

inline void record_debug_event(thread const PathtraceDebugContext& ctx,
                               uint eventKind,
                               uint depth,
                               uint mediumDepthBefore,
                               uint mediumDepthAfter,
                               int mediumEvent,
                               uint frontFace,
                               uint materialIndex,
                               bool scatterIsDelta,
                               float3 throughput,
                               float4 scalars,
                               float3 aux) {
    (void)ctx;
    (void)eventKind;
    (void)depth;
    (void)mediumDepthBefore;
    (void)mediumDepthAfter;
    (void)mediumEvent;
    (void)frontFace;
    (void)materialIndex;
    (void)scatterIsDelta;
    (void)throughput;
    (void)scalars;
    (void)aux;
}

inline void record_debug_event(thread const PathtraceDebugContext& ctx,
                               uint depth,
                               uint mediumDepthBefore,
                               uint mediumDepthAfter,
                               int mediumEvent,
                               uint frontFace,
                               uint materialIndex,
                               bool scatterIsDelta,
                               float3 throughput) {
    (void)ctx;
    (void)depth;
    (void)mediumDepthBefore;
    (void)mediumDepthAfter;
    (void)mediumEvent;
    (void)frontFace;
    (void)materialIndex;
    (void)scatterIsDelta;
    (void)throughput;
}

inline void record_shadow_path_audit(thread const PathtraceDebugContext& ctx,
                                     uint depth,
                                     uint mediumDepth,
                                     thread const HardwareShadowTraceAudit& hwAudit,
                                     thread const SoftwareShadowTraceAudit& swAudit,
                                     uint lightIndex,
                                     int lightAuditKind,
                                     float shadowTMin,
                                     float shadowTMax,
                                     bool lightTwoSided,
                                     bool finalVisible) {
    (void)ctx;
    (void)depth;
    (void)mediumDepth;
    (void)hwAudit;
    (void)swAudit;
    (void)lightIndex;
    (void)lightAuditKind;
    (void)shadowTMin;
    (void)shadowTMax;
    (void)lightTwoSided;
    (void)finalVisible;
}

inline HitRecord make_empty_hit_record() {
    HitRecord rec;
    rec.point = float3(0.0f);
    rec.normal = float3(0.0f);
    rec.shadingNormal = float3(0.0f);
    rec.paddingNormal = 0.0f;
    rec.t = 0.0f;
    rec.materialIndex = 0u;
    rec.twoSided = 0u;
    rec.frontFace = 0u;
    rec.primitiveType = kPrimitiveTypeNone;
    rec.primitiveIndex = kInvalidIndex;
    rec.meshIndex = kInvalidIndex;
    rec.barycentric = float2(0.0f);
    rec.padding = 0u;
    return rec;
}

inline void record_parity_entry(thread const PathtraceDebugContext& ctx,
                                constant PathtraceUniforms& uniforms,
                                uint depth,
                                thread const Ray& ray,
                                float tMin,
                                float tMax,
                                bool hwHit,
                                thread const HitRecord& hwRec,
                                bool swHit,
                                thread const HitRecord& swRec,
                                uint reasonMask) {
    (void)ctx;
    (void)uniforms;
    (void)depth;
    (void)ray;
    (void)tMin;
    (void)tMax;
    (void)hwHit;
    (void)hwRec;
    (void)swHit;
    (void)swRec;
    (void)reasonMask;
}
#endif
