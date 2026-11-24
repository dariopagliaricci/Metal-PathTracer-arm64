constant uint kCausticTransportModePathSpacePrototype = 1u;
constant uint kCausticClassDiffuseConnection = 1u;
constant uint kCausticClassDeltaChain = 2u;
constant uint kCausticClassLightVertex = 3u;
constant uint kCausticClassRoughGlossyConnection = 4u;
constant uint kCausticClassSpecularConnection = 5u;
constant uint kCausticClassEmissiveConnection = 6u;
constant uint kCausticMisFlagBalance = 1u << 0u;
constant uint kCausticMisFlagDeltaChain = 1u << 1u;
constant uint kCausticMisFlagConnectable = 1u << 2u;
constant float kCausticPdfEpsilon = 1.0e-6f;

inline bool caustic_transport_enabled(constant PathtraceUniforms& uniforms) {
    return PT_DEFAULT_OFF(uniforms.causticTransportMode) == kCausticTransportModePathSpacePrototype;
}

inline uint caustic_transport_vertex_budget(constant PathtraceUniforms& uniforms) {
    return max(uniforms.causticMaxVertices, 1u);
}

inline uint caustic_transport_classification_budget(constant PathtraceUniforms& uniforms) {
    return max(caustic_transport_vertex_budget(uniforms) / 2u, 1u);
}

inline float caustic_transport_balance_mis(const float pdfFwd, const float pdfRev) {
    const float fwd = max(pdfFwd, 0.0f);
    const float rev = max(pdfRev, 0.0f);
    return fwd / max(fwd + rev, kCausticPdfEpsilon);
}

inline uint caustic_transport_classify(const uint closureFlags,
                                       const bool sampledDelta,
                                       const bool connectable,
                                       const bool emissive) {
    if (emissive) {
        return kCausticClassEmissiveConnection;
    }
    if (sampledDelta || ((closureFlags & kMaterialClosureFlagDelta) != 0u)) {
        return kCausticClassDeltaChain;
    }
    if (!connectable) {
        return kCausticClassSpecularConnection;
    }
    if ((closureFlags & kMaterialClosureFlagDiffuse) != 0u) {
        return kCausticClassDiffuseConnection;
    }
    if ((closureFlags & kMaterialClosureFlagGlossy) != 0u) {
        return kCausticClassRoughGlossyConnection;
    }
    return kCausticClassSpecularConnection;
}

inline PathVertex caustic_transport_make_vertex(const MaterialData material,
                                                const float3 position,
                                                const float3 shadingNormal,
                                                const float3 geometricNormal,
                                                const float3 direction,
                                                const float3 throughput,
                                                const float pdfFwd,
                                                const float pdfRev,
                                                const float etaScale,
                                                const uint depth,
                                                const uint lobeType,
                                                const uint mediumState,
                                                const uint pathClass,
                                                const bool connectable) {
    const MaterialClosureMetadata closure = material_primary_closure_metadata(material);
    PathVertex pathVertex;
    pathVertex.positionPdfFwd = float4(position, max(pdfFwd, 0.0f));
    pathVertex.normalPdfRev = float4(safe_normalize(shadingNormal), max(pdfRev, 0.0f));
    pathVertex.throughputEta = float4(max(throughput, float3(0.0f)), max(etaScale, 0.0f));
    pathVertex.directionDepth = float4(safe_normalize(direction), float(depth));
    pathVertex.geometricNormalMedium = float4(safe_normalize(geometricNormal),
                                              as_type<float>(mediumState));
    pathVertex.metadata = uint4(closure.flags,
                                depth,
                                connectable ? 1u : 0u,
                                closure.type);
    pathVertex.transport = uint4(lobeType, pathClass, mediumState, 0u);
    return pathVertex;
}

inline void caustic_transport_store_eye_vertex(device EyeSubpathVertex* eyeVertices,
                                               const uint index,
                                               const PathVertex pathVertex) {
    if (eyeVertices) {
        eyeVertices[index].pathVertex = pathVertex;
    }
}

inline void caustic_transport_store_light_vertex(device LightSubpathVertex* lightVertices,
                                                 const uint index,
                                                 const PathVertex pathVertex) {
    if (lightVertices) {
        lightVertices[index].pathVertex = pathVertex;
    }
}

inline void caustic_transport_store_connection(device PathConnection* connections,
                                               device ConnectionResult* results,
                                               const uint index,
                                               const PathConnection connection,
                                               const uint pathClass,
                                               const uint flags) {
    if (connections) {
        connections[index].vertexIndices = connection.vertexIndices;
        connections[index].contributionMis = connection.contributionMis;
        connections[index].pdfs = connection.pdfs;
    }
    if (results) {
        results[index].radianceMis = connection.contributionMis;
        results[index].classification = uint4(pathClass, flags, connection.vertexIndices.x, connection.vertexIndices.y);
    }
}

inline void caustic_transport_note_path_vertex(constant PathtraceUniforms& uniforms,
                                               const MaterialData material,
                                               const float3 position,
                                               const float3 shadingNormal,
                                               const float3 geometricNormal,
                                               const float3 direction,
                                               const float3 throughput,
                                               const uint depth,
                                               const uint mediumState,
                                               const BsdfSampleResult sample,
                                               device EyeSubpathVertex* eyeVertices,
                                               device LightSubpathVertex* lightVertices,
                                               device PathConnection* connections,
                                               device ConnectionResult* connectionResults,
                                               device PathtraceStats* stats) {
    if (!caustic_transport_enabled(uniforms) || !stats) {
        return;
    }

    const MaterialClosureMetadata closure = material_closure_metadata_for_sample(material, sample);
    const bool emissive = material_closure_is_emissive(material);
    const bool connectable =
        (closure.flags & kMaterialClosureFlagConnectable) != 0u &&
        !sample.isDelta &&
        sample.pdf > 0.0f &&
        sample.directionalPdf > 0.0f;
    const uint pathClass =
        caustic_transport_classify(closure.flags, sample.isDelta, connectable, emissive);
    const uint vertexBudget = caustic_transport_classification_budget(uniforms);

    const uint eyeIndex =
        atomic_fetch_add_explicit(&stats->causticEyeVertexCount, 1u, memory_order_relaxed);
    if (eyeIndex >= vertexBudget) {
        atomic_fetch_sub_explicit(&stats->causticEyeVertexCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->causticDebugClassificationCount,
                                  1u,
                                  memory_order_relaxed);
        return;
    }

    const PathVertex pathVertex =
        caustic_transport_make_vertex(material,
                                      position,
                                      shadingNormal,
                                      geometricNormal,
                                      direction,
                                      throughput,
                                      sample.pdf,
                                      sample.directionalPdf,
                                      1.0f,
                                      depth,
                                      sample.lobeType,
                                      mediumState,
                                      pathClass,
                                      connectable);
    caustic_transport_store_eye_vertex(eyeVertices, eyeIndex, pathVertex);

    if (emissive) {
        const uint lightIndex =
            atomic_fetch_add_explicit(&stats->causticLightVertexCount, 1u, memory_order_relaxed);
        if (lightIndex < vertexBudget) {
            caustic_transport_store_light_vertex(lightVertices, lightIndex, pathVertex);
        }
        atomic_fetch_add_explicit(&stats->causticDebugClassificationCount, 1u, memory_order_relaxed);
        return;
    }

    if (pathClass == kCausticClassDeltaChain) {
        atomic_fetch_add_explicit(&stats->causticDeltaChainCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->causticDebugClassificationCount, 1u, memory_order_relaxed);
        return;
    }

    if (connectable) {
        atomic_fetch_add_explicit(&stats->causticDebugClassificationCount,
                                  1u,
                                  memory_order_relaxed);
    }
}

inline void caustic_transport_note_light_endpoint(constant PathtraceUniforms& uniforms,
                                                  const float3 endpoint,
                                                  const float3 normal,
                                                  const float3 directionToLight,
                                                  const float3 emittedRadiance,
                                                  const float pdfFwd,
                                                  const float pdfRev,
                                                  const uint depth,
                                                  const uint mediumState,
                                                  device LightSubpathVertex* lightVertices,
                                                  device PathtraceStats* stats) {
    if (!caustic_transport_enabled(uniforms) || !stats) {
        return;
    }
    const uint vertexBudget = caustic_transport_classification_budget(uniforms);
    const uint lightIndex =
        atomic_fetch_add_explicit(&stats->causticLightVertexCount, 1u, memory_order_relaxed);
    if (lightIndex >= vertexBudget) {
        atomic_fetch_sub_explicit(&stats->causticLightVertexCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->causticDebugClassificationCount,
                                  1u,
                                  memory_order_relaxed);
        return;
    }

    PathVertex lightVertex;
    lightVertex.positionPdfFwd = float4(endpoint, max(pdfFwd, 0.0f));
    lightVertex.normalPdfRev = float4(safe_normalize(normal), max(pdfRev, 0.0f));
    lightVertex.throughputEta = float4(max(emittedRadiance, float3(0.0f)), 1.0f);
    lightVertex.directionDepth = float4(safe_normalize(directionToLight), float(depth));
    lightVertex.geometricNormalMedium = float4(safe_normalize(normal), as_type<float>(mediumState));
    lightVertex.metadata = uint4(kMaterialClosureFlagEmissive | kMaterialClosureFlagConnectable,
                                 depth,
                                 1u,
                                 kMaterialClosureTypeEmission);
    lightVertex.transport = uint4(0u, kCausticClassLightVertex, mediumState, 0u);
    caustic_transport_store_light_vertex(lightVertices, lightIndex, lightVertex);
    atomic_fetch_add_explicit(&stats->causticDebugClassificationCount, 1u, memory_order_relaxed);
}

inline bool caustic_transport_connect_light_sample(
    constant PathtraceUniforms& uniforms,
    const MaterialData material,
    const float3 eyePosition,
    const float3 shadingNormal,
    const float3 geometricNormal,
    const float3 directionToLight,
    const float3 lightPosition,
    const float3 lightNormal,
    const bool infiniteLight,
    const float3 throughput,
    const float3 emittedRadiance,
    const float3 bsdfValue,
    const float bsdfPdf,
    const float lightPdf,
    const float nDotL,
    const uint depth,
    const uint mediumState,
    device EyeSubpathVertex* eyeVertices,
    device LightSubpathVertex* lightVertices,
    device PathConnection* connections,
    device ConnectionResult* connectionResults,
    device PathtraceStats* stats,
    thread float3& outContribution,
    thread float& outMisWeight) {
    outContribution = float3(0.0f);
    outMisWeight = 0.0f;
    if (!caustic_transport_enabled(uniforms) || !stats) {
        return false;
    }
    if (!(lightPdf > 0.0f) || !(nDotL > 0.0f) ||
        !all(isfinite(bsdfValue)) || !all(isfinite(emittedRadiance))) {
        atomic_fetch_add_explicit(&stats->causticDebugClassificationCount,
                                  1u,
                                  memory_order_relaxed);
        return false;
    }

    const MaterialClosureMetadata closure = material_primary_closure_metadata(material);
    const bool closureConnectable =
        (closure.flags & kMaterialClosureFlagConnectable) != 0u ||
        (closure.flags & kMaterialClosureFlagDiffuse) != 0u ||
        (closure.flags & kMaterialClosureFlagGlossy) != 0u;
    const bool connectable =
        closureConnectable &&
        (closure.flags & kMaterialClosureFlagDelta) == 0u &&
        (closure.flags & kMaterialClosureFlagVolumetric) == 0u;
    if (!connectable) {
        atomic_fetch_add_explicit(&stats->causticDebugClassificationCount,
                                  1u,
                                  memory_order_relaxed);
        return false;
    }

    const uint vertexBudget = caustic_transport_vertex_budget(uniforms);
    const uint eyeIndex =
        atomic_fetch_add_explicit(&stats->causticEyeVertexCount, 1u, memory_order_relaxed);
    if (eyeIndex >= vertexBudget) {
        atomic_fetch_sub_explicit(&stats->causticEyeVertexCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->causticDebugClassificationCount,
                                  1u,
                                  memory_order_relaxed);
        return false;
    }

    const uint lightIndex =
        atomic_fetch_add_explicit(&stats->causticLightVertexCount, 1u, memory_order_relaxed);
    if (lightIndex >= vertexBudget) {
        atomic_fetch_sub_explicit(&stats->causticLightVertexCount, 1u, memory_order_relaxed);
        atomic_fetch_sub_explicit(&stats->causticEyeVertexCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->causticDebugClassificationCount,
                                  1u,
                                  memory_order_relaxed);
        return false;
    }

    const uint connectionIndex =
        atomic_fetch_add_explicit(&stats->causticConnectionCandidateCount,
                                  1u,
                                  memory_order_relaxed);
    if (connectionIndex >= vertexBudget) {
        atomic_fetch_sub_explicit(&stats->causticConnectionCandidateCount,
                                  1u,
                                  memory_order_relaxed);
        atomic_fetch_sub_explicit(&stats->causticLightVertexCount, 1u, memory_order_relaxed);
        atomic_fetch_sub_explicit(&stats->causticEyeVertexCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->causticDebugClassificationCount,
                                  1u,
                                  memory_order_relaxed);
        return false;
    }

    const float bsdfPdfSafe = max(bsdfPdf, 0.0f);
    const float misWeight = caustic_transport_balance_mis(lightPdf, bsdfPdfSafe);
    const float invLightPdf = 1.0f / max(lightPdf, kCausticPdfEpsilon);
    float3 contribution = emittedRadiance * bsdfValue * nDotL * (misWeight * invLightPdf);
    if (!all(isfinite(contribution))) {
        atomic_fetch_sub_explicit(&stats->causticConnectionCandidateCount,
                                  1u,
                                  memory_order_relaxed);
        atomic_fetch_sub_explicit(&stats->causticLightVertexCount, 1u, memory_order_relaxed);
        atomic_fetch_sub_explicit(&stats->causticEyeVertexCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->causticDebugClassificationCount,
                                  1u,
                                  memory_order_relaxed);
        return false;
    }
    contribution = max(contribution, float3(0.0f));

    const uint pathClass = caustic_transport_classify(closure.flags,
                                                     false,
                                                     true,
                                                     false);
    const PathVertex eyeVertex =
        caustic_transport_make_vertex(material,
                                      eyePosition,
                                      shadingNormal,
                                      geometricNormal,
                                      directionToLight,
                                      throughput,
                                      bsdfPdfSafe,
                                      lightPdf,
                                      1.0f,
                                      depth,
                                      closure.lobeIndex,
                                      mediumState,
                                      pathClass,
                                      true);
    caustic_transport_store_eye_vertex(eyeVertices, eyeIndex, eyeVertex);

    const float3 safeLightDirection = safe_normalize(directionToLight);
    const float3 safeLightNormal =
        any(abs(lightNormal) > float3(0.0f)) ? safe_normalize(lightNormal) : -safeLightDirection;
    PathVertex lightVertex;
    lightVertex.positionPdfFwd = float4(lightPosition, lightPdf);
    lightVertex.normalPdfRev = float4(safeLightNormal, bsdfPdfSafe);
    lightVertex.throughputEta = float4(max(emittedRadiance, float3(0.0f)), 1.0f);
    lightVertex.directionDepth = float4(-safeLightDirection, float(depth));
    lightVertex.geometricNormalMedium = float4(safeLightNormal, as_type<float>(mediumState));
    lightVertex.metadata = uint4(kMaterialClosureFlagEmissive | kMaterialClosureFlagConnectable,
                                 depth,
                                 1u,
                                 kMaterialClosureTypeEmission);
    lightVertex.transport = uint4(0u, kCausticClassLightVertex, mediumState, infiniteLight ? 1u : 0u);
    caustic_transport_store_light_vertex(lightVertices, lightIndex, lightVertex);

    const uint misFlags = kCausticMisFlagBalance | kCausticMisFlagConnectable;
    PathConnection connection;
    connection.vertexIndices = uint4(eyeIndex, lightIndex, pathClass, misFlags);
    connection.contributionMis = float4(contribution, misWeight);
    connection.pdfs = float4(bsdfPdfSafe, lightPdf, bsdfPdfSafe, lightPdf);
    caustic_transport_store_connection(connections,
                                       connectionResults,
                                       connectionIndex,
                                       connection,
                                       pathClass,
                                       misFlags);
    atomic_fetch_add_explicit(&stats->causticDebugClassificationCount,
                              1u,
                              memory_order_relaxed);
    outContribution = contribution;
    outMisWeight = misWeight;
    return true;
}
