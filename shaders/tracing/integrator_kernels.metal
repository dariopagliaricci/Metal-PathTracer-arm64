inline void write_pathtrace_denoise_guides(
    texture2d<float, access::read_write> albedoTexture,
    texture2d<float, access::read_write> normalTexture,
    const uint2 pixel,
    const uint previousCount,
    const bool averageGuides,
    float3 albedo,
    const float3 normal) {
    float3 encodedNormal = normal * 0.5f + 0.5f;
    float4 previousAlbedo = float4(0.0f);
    // These guide textures are not cleared with radiance. Never read their
    // previous contents when accumulation has just been reset.
    if (averageGuides && previousCount > 0u) {
        previousAlbedo = albedoTexture.read(pixel);
    }
    // OIDN expects guides filtered like the color: average every jittered
    // sample, or the last sample's aliased edges reappear in the output.
    // SVGF and ML reconstruction read per-frame guides, so they pass false.
    const bool hasGuideHistory = previousCount > 0u && previousAlbedo.w >= 2.0f;
    float guideHistoryTag = 2.0f;
    if (averageGuides && hasGuideHistory) {
        // Alpha is the number of accumulated guide samples plus one. Radiance
        // may have older samples from before this guide history was enabled.
        const float sampleWeight = 1.0f / previousAlbedo.w;
        albedo = mix(previousAlbedo.xyz, albedo, sampleWeight);
        encodedNormal = mix(normalTexture.read(pixel).xyz, encodedNormal, sampleWeight);
        guideHistoryTag = previousAlbedo.w + 1.0f;
    }
    // The unused alpha marks guide history so a jittered sky miss still
    // contributes its zero guide to the same pixel's running average.
    albedoTexture.write(float4(albedo, averageGuides ? guideHistoryTag : 1.0f), pixel);
    normalTexture.write(float4(encodedNormal, 1.0f), pixel);
}

kernel void pathtraceIntegrateKernel(texture2d<float, access::read_write> radianceTexture [[texture(0)]],
                                     texture2d<uint, access::read_write> sampleCountTexture [[texture(1)]],
                                     texture2d<float, access::sample> environmentTexture [[texture(2)]],
                                     texture2d<float, access::read_write> albedoTexture [[texture(3)]],
                                     texture2d<float, access::read_write> normalTexture [[texture(4)]],
                                     texture2d<float, access::write> positionTexture [[texture(5)]],
                                     texture2d<float, access::write> materialFeatureTexture [[texture(6)]],
                                     texture2d<float, access::write> motionVectorTexture [[texture(7)]],
                                     texture2d<float, access::read_write> directRadianceTexture [[texture(8)]],
                                     texture2d<float, access::read_write> indirectRadianceTexture [[texture(9)]],
                                     texture2d<float, access::read_write> radianceMomentTexture [[texture(10)]],
                                     texture2d<float, access::write> reconstructionMetricTexture [[texture(11)]],
                                     constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                     device const BvhNode* nodes [[buffer(1)]],
                                     device const uint* primitiveIndices [[buffer(2)]],
                                     device const SphereData* spheres [[buffer(3)]],
                                     device const MaterialData* materials [[buffer(4)]],
                                     device PathtraceStats* stats [[buffer(5)]],
                                     device const RectData* rectangles [[buffer(6)]],
                                     device const EnvironmentAliasEntry* environmentConditionalAlias [[buffer(7)]],
                                     device const EnvironmentAliasEntry* environmentMarginalAlias [[buffer(8)]],
                                     device const float* environmentPdf [[buffer(9)]],
                                     device const TriangleData* triangleData [[buffer(10)]],
                                     device const BvhNode* tlasNodes [[buffer(11)]],
                                     device const uint* tlasPrimIndices [[buffer(12)]],
                                     device const BvhNode* blasNodes [[buffer(13)]],
                                     device const uint* blasPrimIndices [[buffer(14)]],
                                     device const SoftwareInstanceInfo* instanceInfos [[buffer(15)]],
                                     device const MeshInfo* meshInfos [[buffer(16)]],
                                     device const SceneVertex* sceneVertices [[buffer(17)]],
                                     device const uint3* meshIndices [[buffer(18)]],
                                     device PathtraceDebugBuffer* debugBuffer [[buffer(19)]],
                                     device const MaterialTextureInfo* materialTextureInfos [[buffer(20)]],
                                     device const LightPrimitive* emissivePrimitives [[buffer(21)]],
                                     device PathGuidingReservoirState* pathGuidingStates [[buffer(22)]],
                                     constant MaterialTextureArgumentBuffer& materialResources [[buffer(23)]],
                                     device RestirPtReservoirState* restirPtReservoirs [[buffer(24)]],
                                     uint2 gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }

    float3 accumulated = radianceTexture.read(gid).xyz;
    uint previousCount = sampleCountTexture.read(gid).x;

    uint seed = uniforms.fixedRngSeed +                   // Deterministic base (0 if not set)
                uniforms.frameIndex * 9781u +
                gid.x * 6271u +
                gid.y * 13007u +
                (uniforms.sampleCount + previousCount) * 211u;
    thread uint rngState = seed;

    bool deterministicDebugRay = (PT_DEFAULT_OFF(uniforms.debugPathActive) != 0u) &&
                                 (gid.x == uniforms.debugPixelX) &&
                                 (gid.y == uniforms.debugPixelY);
    float uJitter = deterministicDebugRay ? 0.5f : rand_uniform(rngState);
    float vJitter = deterministicDebugRay ? 0.5f : rand_uniform(rngState);
    float u = (float(gid.x) + uJitter) / float(uniforms.width);
    float v = (float(gid.y) + vJitter) / float(uniforms.height);
    v = 1.0f - v;

    Ray ray;
    float3 pixelPosition =
        uniforms.lowerLeftCorner + u * uniforms.horizontal + v * uniforms.vertical;
    float2 diskSample = deterministicDebugRay ? float2(0.0f) :
                                                (uniforms.lensRadius * random_in_unit_disk(rngState));
    float3 offset = uniforms.cameraU * diskSample.x + uniforms.cameraV * diskSample.y;
    ray.origin = uniforms.cameraOrigin + offset;
    PrimaryRayDiff primaryRayDiff;
    primaryRayDiff.dOdx = float3(0.0f);
    primaryRayDiff.dOdy = float3(0.0f);
    primaryRayDiff.dDdx = uniforms.horizontal / max(float(uniforms.width), 1.0f);
    primaryRayDiff.dDdy = -uniforms.vertical / max(float(uniforms.height), 1.0f);
    ray.direction = normalize_camera_ray_direction(pixelPosition - ray.origin, primaryRayDiff);

    float3 hitAlbedo = float3(0.0f);
    float3 hitNormal = float3(0.0f);
    float4 hitPosition = float4(0.0f);
    float4 hitMaterial = float4(0.0f);
    float3 directSample = float3(0.0f);
    float3 indirectSample = float3(0.0f);
    float reconstructionConfidence = 0.0f;
    float reconstructionVariance = 0.0f;
    float reconstructionBiasRisk = 0.0f;

    PathtraceDebugContext debugCtx = make_debug_context(uniforms,
                                                        debugBuffer,
                                                        gid,
                                                        previousCount,
                                                        0u);
    thread PathtraceDebugContext* debugCtxPtr = nullptr;
#if PT_DEBUG_TOOLS
    debugCtxPtr = (debugBuffer &&
                   (PT_DEFAULT_OFF(uniforms.debugPathActive) != 0u ||
                    PT_DEFAULT_OFF(uniforms.debugDirectLightAudit) != 0u))
        ? &debugCtx
        : nullptr;
#endif
    float3 sample = trace_path_software(uniforms,
                                        spheres,
                                        rectangles,
                                        triangleData,
                                        emissivePrimitives,
                                        materials,
                                        meshInfos,
                                        sceneVertices,
                                        meshIndices,
                                        ray,
                                        primaryRayDiff,
                                        rngState,
                                        tlasNodes,
                                        tlasPrimIndices,
                                        blasNodes,
                                        blasPrimIndices,
                                        instanceInfos,
                                        nodes,
                                        primitiveIndices,
                                        stats,
                                        environmentTexture,
                                        materialResources.textures,
                                        materialResources.samplers,
                                        materialTextureInfos,
                                        environmentConditionalAlias,
                                        environmentMarginalAlias,
                                        environmentPdf,
                                        pathGuidingStates,
                                        restirPtReservoirs,
                                        materialResources.radianceCacheStates,
                                        materialResources.causticEyeVertices,
                                        materialResources.causticLightVertices,
                                        materialResources.causticConnections,
                                        materialResources.causticConnectionResults,
                                        gid,
                                        &hitAlbedo,
                                        &hitNormal,
                                        &hitPosition,
                                        &hitMaterial,
                                        &directSample,
                                        &indirectSample,
                                        &reconstructionConfidence,
                                        &reconstructionVariance,
                                        &reconstructionBiasRisk,
                                        debugCtxPtr);
    if (!all(isfinite(sample))) {
        sample = float3(0.0f);
    } else {
        sample = max(sample, float3(0.0f));
    }

    uint newCount = previousCount + 1u;
    float3 newSum = accumulated + sample;

    radianceTexture.write(float4(newSum, 0.0f), gid);
    sampleCountTexture.write(newCount, gid);
    directRadianceTexture.write(float4(directRadianceTexture.read(gid).xyz + max(directSample, float3(0.0f)), 0.0f), gid);
    indirectRadianceTexture.write(float4(indirectRadianceTexture.read(gid).xyz + max(indirectSample, float3(0.0f)), 0.0f), gid);
    const float sampleLuma = luminance_rgb(sample);
    radianceMomentTexture.write(float4(radianceMomentTexture.read(gid).x + sampleLuma * sampleLuma, 0.0f, 0.0f, 0.0f), gid);
    reconstructionMetricTexture.write(float4(reconstructionVariance,
                                             max(clamp(log2(float(newCount) + 1.0f) / 8.0f, 0.0f, 1.0f),
                                                 clamp(reconstructionConfidence, 0.0f, 1.0f)),
                                             clamp(reconstructionBiasRisk, 0.0f, 1.0f),
                                             float(newCount)), gid);

    write_pathtrace_denoise_guides(albedoTexture, normalTexture, gid, previousCount,
                                   uniforms.oidnDeltaGuidesAllowed != 0u,
                                   hitAlbedo, hitNormal);
    positionTexture.write(hitPosition, gid);
    materialFeatureTexture.write(hitMaterial, gid);
    motionVectorTexture.write(float4(0.0f), gid);
}

#if __METAL_VERSION__ >= 310
kernel void pathtraceIntegrateHardwareKernel(texture2d<float, access::read_write> radianceTexture [[texture(0)]],
                                             texture2d<uint, access::read_write> sampleCountTexture [[texture(1)]],
                                             texture2d<float, access::sample> environmentTexture [[texture(2)]],
                                             texture2d<float, access::read_write> albedoTexture [[texture(3)]],
                                             texture2d<float, access::read_write> normalTexture [[texture(4)]],
                                             texture2d<float, access::write> positionTexture [[texture(5)]],
                                             texture2d<float, access::write> materialFeatureTexture [[texture(6)]],
                                             texture2d<float, access::write> motionVectorTexture [[texture(7)]],
                                             texture2d<float, access::read_write> directRadianceTexture [[texture(8)]],
                                             texture2d<float, access::read_write> indirectRadianceTexture [[texture(9)]],
                                             texture2d<float, access::read_write> radianceMomentTexture [[texture(10)]],
                                             texture2d<float, access::write> reconstructionMetricTexture [[texture(11)]],
                                             constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
                                             acceleration_structure<instancing> accel [[buffer(1)]],
                                             device const MeshInfo* meshInfos [[buffer(2)]],
                                             device const TriangleData* triangleData [[buffer(3)]],
                                             device const uint* instanceUserIds [[buffer(13)]],
                                             device const BvhNode* nodes [[buffer(4)]],
                                             device const uint* primitiveIndices [[buffer(5)]],
                                             device const SphereData* spheres [[buffer(6)]],
                                             device const MaterialData* materials [[buffer(7)]],
                                             device PathtraceStats* stats [[buffer(8)]],
                                             device const RectData* rectangles [[buffer(9)]],
                                             device const EnvironmentAliasEntry* environmentConditionalAlias [[buffer(10)]],
                                             device const EnvironmentAliasEntry* environmentMarginalAlias [[buffer(11)]],
                                             device const float* environmentPdf [[buffer(12)]],
                                             device const SceneVertex* sceneVertices [[buffer(14)]],
                                             device const uint3* meshIndices [[buffer(15)]],
                                             device PathtraceDebugBuffer* debugBuffer [[buffer(16)]],
                                             device const BvhNode* tlasNodes [[buffer(17)]],
                                             device const uint* tlasPrimIndices [[buffer(18)]],
                                             device const BvhNode* blasNodes [[buffer(19)]],
                                             device const uint* blasPrimIndices [[buffer(20)]],
                                             device const SoftwareInstanceInfo* instanceInfos [[buffer(21)]],
                                             device const MaterialTextureInfo* materialTextureInfos [[buffer(22)]],
                                             constant MaterialTextureArgumentBuffer& materialResources [[buffer(23)]],
                                             device const LightPrimitive* emissivePrimitives [[buffer(24)]],
                                             device PathGuidingReservoirState* pathGuidingStates [[buffer(25)]],
                                             device RestirPtReservoirState* restirPtReservoirs [[buffer(26)]],
                                             uint2 gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }
    if (kMegakernelDefaultFeatures && gid.x == 0u && gid.y == 0u && stats &&
        !default_feature_gates_off(uniforms)) {
        // The host picked the default-features pipeline with a gate on: its list and
        // default_feature_gates_off() have drifted. The host logs this count as an error.
        atomic_fetch_add_explicit(&stats->defaultFeatureGateViolationCount, 1u, memory_order_relaxed);
    }

    float3 accumulated = radianceTexture.read(gid).xyz;
    uint previousCount = sampleCountTexture.read(gid).x;

    uint seed = uniforms.fixedRngSeed +
                uniforms.frameIndex * 9781u +
                gid.x * 6271u +
                gid.y * 13007u +
                (uniforms.sampleCount + previousCount) * 211u;
    thread uint rngState = seed;

    bool deterministicDebugRay = (PT_DEFAULT_OFF(uniforms.debugPathActive) != 0u) &&
                                 (gid.x == uniforms.debugPixelX) &&
                                 (gid.y == uniforms.debugPixelY);
    float uJitter = deterministicDebugRay ? 0.5f : rand_uniform(rngState);
    float vJitter = deterministicDebugRay ? 0.5f : rand_uniform(rngState);
    float u = (float(gid.x) + uJitter) / float(uniforms.width);
    float v = (float(gid.y) + vJitter) / float(uniforms.height);
    v = 1.0f - v;

    Ray ray;
    float3 pixelPosition =
        uniforms.lowerLeftCorner + u * uniforms.horizontal + v * uniforms.vertical;
    float2 diskSample = deterministicDebugRay ? float2(0.0f) :
                                                (uniforms.lensRadius * random_in_unit_disk(rngState));
    float3 offset = uniforms.cameraU * diskSample.x + uniforms.cameraV * diskSample.y;
    ray.origin = uniforms.cameraOrigin + offset;
    PrimaryRayDiff primaryRayDiff;
    primaryRayDiff.dOdx = float3(0.0f);
    primaryRayDiff.dOdy = float3(0.0f);
    primaryRayDiff.dDdx = uniforms.horizontal / max(float(uniforms.width), 1.0f);
    primaryRayDiff.dDdy = -uniforms.vertical / max(float(uniforms.height), 1.0f);
    ray.direction = normalize_camera_ray_direction(pixelPosition - ray.origin, primaryRayDiff);

    float3 hitAlbedo = float3(0.0f);
    float3 hitNormal = float3(0.0f);
    float4 hitPosition = float4(0.0f);
    float4 hitMaterial = float4(0.0f);
    float3 directSample = float3(0.0f);
    float3 indirectSample = float3(0.0f);
    float reconstructionConfidence = 0.0f;
    float reconstructionVariance = 0.0f;
    float reconstructionBiasRisk = 0.0f;

    PathtraceDebugContext hwDebugCtx = make_debug_context(uniforms,
                                                          debugBuffer,
                                                          gid,
                                                          previousCount,
                                                          1u);
    thread PathtraceDebugContext* hwDebugPtr = nullptr;
#if PT_DEBUG_TOOLS
    hwDebugPtr =
        (debugBuffer &&
         (PT_DEFAULT_OFF(uniforms.debugPathActive) != 0u ||
          PT_DEFAULT_OFF(uniforms.debugDirectLightAudit) != 0u ||
          PT_DEFAULT_OFF(uniforms.parityAssertEnabled) != 0u))
            ? &hwDebugCtx
            : nullptr;
#endif
    float3 sample = trace_path_hardware(uniforms,
                                        accel,
                                        meshInfos,
                                        triangleData,
                                        emissivePrimitives,
                                        instanceUserIds,
                                        spheres,
                                        rectangles,
                                        materials,
                                        sceneVertices,
                                        meshIndices,
                                        tlasNodes,
                                        tlasPrimIndices,
                                        blasNodes,
                                        blasPrimIndices,
                                        instanceInfos,
                                        ray,
                                        primaryRayDiff,
                                        rngState,
                                        nodes,
                                        primitiveIndices,
                                        stats,
                                        environmentTexture,
                                        materialResources.textures,
                                        materialResources.samplers,
                                        materialTextureInfos,
                                        environmentConditionalAlias,
                                        environmentMarginalAlias,
                                        environmentPdf,
                                        pathGuidingStates,
                                        restirPtReservoirs,
                                        materialResources.radianceCacheStates,
                                        materialResources.causticEyeVertices,
                                        materialResources.causticLightVertices,
                                        materialResources.causticConnections,
                                        materialResources.causticConnectionResults,
                                        gid,
                                        &hitAlbedo,
                                        &hitNormal,
                                        &hitPosition,
                                        &hitMaterial,
                                        &directSample,
                                        &indirectSample,
                                        &reconstructionConfidence,
                                        &reconstructionVariance,
                                        &reconstructionBiasRisk,
                                        hwDebugPtr);
    if (!all(isfinite(sample))) {
        sample = float3(0.0f);
    } else {
        sample = max(sample, float3(0.0f));
    }

    uint newCount = previousCount + 1u;
    float3 newSum = accumulated + sample;

    radianceTexture.write(float4(newSum, 0.0f), gid);
    sampleCountTexture.write(newCount, gid);
    directRadianceTexture.write(float4(directRadianceTexture.read(gid).xyz + max(directSample, float3(0.0f)), 0.0f), gid);
    indirectRadianceTexture.write(float4(indirectRadianceTexture.read(gid).xyz + max(indirectSample, float3(0.0f)), 0.0f), gid);
    const float sampleLuma = luminance_rgb(sample);
    radianceMomentTexture.write(float4(radianceMomentTexture.read(gid).x + sampleLuma * sampleLuma, 0.0f, 0.0f, 0.0f), gid);
    reconstructionMetricTexture.write(float4(reconstructionVariance,
                                             max(clamp(log2(float(newCount) + 1.0f) / 8.0f, 0.0f, 1.0f),
                                                 clamp(reconstructionConfidence, 0.0f, 1.0f)),
                                             clamp(reconstructionBiasRisk, 0.0f, 1.0f),
                                             float(newCount)), gid);

    write_pathtrace_denoise_guides(albedoTexture, normalTexture, gid, previousCount,
                                   uniforms.oidnDeltaGuidesAllowed != 0u,
                                   hitAlbedo, hitNormal);
    positionTexture.write(hitPosition, gid);
    materialFeatureTexture.write(hitMaterial, gid);
    motionVectorTexture.write(float4(0.0f), gid);
}
#endif
