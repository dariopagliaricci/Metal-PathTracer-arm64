constant uint kCausticTransportModePathSpacePrototype = 1u;
constant uint kCausticClassDiffuseConnection = 1u;
constant uint kCausticClassDeltaChain = 2u;
constant uint kCausticClassLightVertex = 3u;
constant float kCausticPdfEpsilon = 1.0e-6f;

inline uint caustic_transport_closure_flags(const MaterialData material) {
    return material_primary_closure_metadata(material).flags;
}

inline bool caustic_transport_enabled(constant PathtraceUniforms& uniforms) {
    return uniforms.causticTransportMode == kCausticTransportModePathSpacePrototype;
}

inline float caustic_transport_balance_mis(const float pdfFwd, const float pdfRev) {
    const float fwd = max(pdfFwd, 0.0f);
    const float rev = max(pdfRev, 0.0f);
    return fwd / max(fwd + rev, kCausticPdfEpsilon);
}

inline PathVertex caustic_transport_make_eye_vertex(const MaterialData material,
                                                    const float3 position,
                                                    const float3 normal,
                                                    const float3 throughput,
                                                    const float pdfFwd,
                                                    const float pdfRev,
                                                    const float etaScale,
                                                    const uint depth,
                                                    const bool connectable) {
    PathVertex pathVertex;
    pathVertex.positionPdfFwd = float4(position, max(pdfFwd, 0.0f));
    pathVertex.normalPdfRev = float4(safe_normalize(normal), max(pdfRev, 0.0f));
    pathVertex.throughputEta = float4(throughput, max(etaScale, 0.0f));
    pathVertex.metadata = uint4(caustic_transport_closure_flags(material),
                                depth,
                                connectable ? 1u : 0u,
                                static_cast<uint>(material.typeEta.x));
    return pathVertex;
}

inline void caustic_transport_note_path_vertex(constant PathtraceUniforms& uniforms,
                                               const MaterialData material,
                                               const float3 position,
                                               const float3 normal,
                                               const float3 throughput,
                                               const uint depth,
                                               const BsdfSampleResult sample,
                                               device PathtraceStats* stats) {
    if (!caustic_transport_enabled(uniforms) || !stats) {
        return;
    }
    const bool connectable =
        material_closure_is_connectable(material) &&
        !sample.isDelta &&
        sample.pdf > 0.0f &&
        sample.directionalPdf > 0.0f;
    const uint vertexBudget = max(uniforms.causticMaxVertices, 1u);
    const uint eyeIndex =
        atomic_fetch_add_explicit(&stats->causticEyeVertexCount, 1u, memory_order_relaxed);
    if (eyeIndex >= vertexBudget) {
        atomic_fetch_sub_explicit(&stats->causticEyeVertexCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->causticDebugClassificationCount,
                                  1u,
                                  memory_order_relaxed);
        return;
    }
    const PathVertex pathVertex = caustic_transport_make_eye_vertex(material,
                                                                   position,
                                                                   normal,
                                                                   throughput,
                                                                   sample.pdf,
                                                                   sample.directionalPdf,
                                                                   1.0f,
                                                                   depth,
                                                                   connectable);
    (void)pathVertex;

    if (material_closure_is_emissive(material)) {
        atomic_fetch_add_explicit(&stats->causticLightVertexCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->causticDebugClassificationCount, 1u, memory_order_relaxed);
        return;
    }
    if (sample.isDelta || ((pathVertex.metadata.x & kMaterialClosureFlagDelta) != 0u)) {
        atomic_fetch_add_explicit(&stats->causticDeltaChainCount, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->causticDebugClassificationCount, 1u, memory_order_relaxed);
        return;
    }
    if (connectable) {
        PathConnection connection;
        const float mis = caustic_transport_balance_mis(sample.pdf, sample.directionalPdf);
        const float3 contributionEstimate =
            max(pathVertex.throughputEta.xyz, float3(0.0f)) * mis;
        connection.vertexIndices = uint4(eyeIndex, 0u, kCausticClassDiffuseConnection, 0u);
        connection.contributionMis = float4(contributionEstimate, mis);
        connection.pdfs = float4(sample.pdf, sample.directionalPdf, 0.0f, 0.0f);
        if (!all(isfinite(connection.contributionMis.xyz)) ||
            !isfinite(connection.contributionMis.w)) {
            atomic_fetch_add_explicit(&stats->causticDebugClassificationCount,
                                      1u,
                                      memory_order_relaxed);
            return;
        }
        atomic_fetch_add_explicit(&stats->causticConnectionCandidateCount,
                                  1u,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(&stats->causticDebugClassificationCount,
                                  1u,
                                  memory_order_relaxed);
    }
}
