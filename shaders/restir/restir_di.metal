inline bool build_ris_reservoir_for_hit(constant PathtraceUniforms& uniforms,
                                        device const LightPrimitive* emissivePrimitives,
                                        thread const HitRecord& rec,
                                        const MaterialData material,
                                        const float3 shadingNormal,
                                        const float3 wo,
                                        const FireflyClampParams clampParams,
                                        const float diffuseOcclusion,
                                        const bool specularOnly,
                                        uint candidateCount,
                                        thread uint& state,
                                        thread Reservoir& reservoir) {
    reservoir = make_empty_reservoir();
    uint risTargetM = max(candidateCount, 1u);
    for (uint i = 0u; i < risTargetM; ++i) {
        RISSamplePayload candidate;
        if (!sample_direct_light_baseline(uniforms,
                                          emissivePrimitives,
                                          rec,
                                          state,
                                          candidate)) {
            continue;
        }
        float distSq = ris_payload_distance_sq(rec, candidate);
        if (!(distSq > 0.0f) || !isfinite(distSq)) {
            continue;
        }
        float3 omegaL = ris_payload_omega_l(rec, candidate);
        float cosThetaSurface = saturate(dot(shadingNormal, omegaL));
        float cosThetaLight = ris_payload_is_directional(candidate)
            ? 1.0f
            : saturate(dot(candidate.normal, -omegaL));
        BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                rec.point,
                                                shadingNormal,
                                                wo,
                                                omegaL,
                                                clampParams,
                                                uniforms.sssMode,
                                                diffuseOcclusion,
                                                specularOnly);
        if (bsdfEval.isDelta || bsdfEval.isBssrdf) {
            continue;
        }
        float phat = p_hat(candidate.emission,
                           bsdfEval.value,
                           cosThetaSurface,
                           cosThetaLight,
                           distSq);
        if (!(phat > 0.0f) || !(candidate.pdf > 0.0f)) {
            continue;
        }
        float weight = phat / candidate.pdf;
        float u = (reservoir.M == 0u) ? 0.0f : rand_uniform(state);
        update_reservoir(reservoir, candidate, weight, u);
    }
    return reservoir.valid && reservoir.M > 0u && reservoir.wSum > 0.0f;
}

constant uint kRestirDILightProviderEmissive = 1u << 0u;
constant uint kRestirDILightProviderAnalytic = 1u << 1u;
constant uint kRestirDILightProviderEnvironment = 1u << 2u;
constant uint kRestirDILightProviderRegir = 1u << 3u;

struct RestirDILightCandidateProviderSet {
    uint providerMask;
    uint providerCount;
    uint rectLightCount;
};

inline RestirDILightCandidateProviderSet restir_di_make_provider_set(
    constant PathtraceUniforms& uniforms,
    device const RectData* rectangles,
    device const MaterialData* materials) {
    RestirDILightCandidateProviderSet providers;
    providers.providerMask = 0u;
    providers.providerCount = 0u;
    providers.rectLightCount =
        (rectangles && materials && uniforms.rectangleCount > 0u && uniforms.materialCount > 0u)
            ? count_rect_lights(uniforms, rectangles, materials)
            : 0u;
    if (uniforms.emissivePrimitiveCount > 0u) {
        providers.providerMask |= kRestirDILightProviderEmissive;
        providers.providerCount += 1u;
    }
    if (providers.rectLightCount > 0u) {
        providers.providerMask |= kRestirDILightProviderAnalytic;
        providers.providerCount += 1u;
    }
    if (uniforms.environmentAliasCount > 0u &&
        uniforms.environmentMapWidth > 0u &&
        uniforms.environmentMapHeight > 0u) {
        providers.providerMask |= kRestirDILightProviderEnvironment;
    }
    if (uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) {
        providers.providerMask |= kRestirDILightProviderRegir;
    }
    return providers;
}

inline bool restir_di_provider_sample_emissive(constant PathtraceUniforms& uniforms,
                                               device const LightPrimitive* emissivePrimitives,
                                               thread const HitRecord& rec,
                                               thread uint& state,
                                               float providerPdf,
                                               thread RISSamplePayload& outSample) {
    if (!sample_direct_light_baseline(uniforms, emissivePrimitives, rec, state, outSample)) {
        return false;
    }
    outSample.pdf *= providerPdf;
    return outSample.pdf > 0.0f && isfinite(outSample.pdf);
}

inline bool restir_di_provider_sample_analytic_rect(
    constant PathtraceUniforms& uniforms,
    device const RectData* rectangles,
    device const MaterialData* materials,
    texture2d<float, access::sample> environmentTexture,
    thread const HitRecord& rec,
    thread uint& state,
    uint rectLightCount,
    float providerPdf,
    thread RISSamplePayload& outSample) {
    RectLightSample rectSample;
    if (!sample_rect_light(uniforms,
                           rectangles,
                           materials,
                           environmentTexture,
                           rec,
                           state,
                           rectLightCount,
                           rectSample)) {
        return false;
    }
    outSample.primitiveIndex = rectSample.rectIndex;
    outSample.flags = 0u;
    outSample.bary = float2(0.0f);
    outSample.position = rec.point + rectSample.direction * rectSample.distance;
    outSample.normal = (rectSample.rectIndex < uniforms.rectangleCount)
        ? rectangles[rectSample.rectIndex].normalAndPlane.xyz
        : -rectSample.direction;
    outSample.emission = rectSample.emission;
    outSample.pdf = rectSample.pdf * providerPdf;
    return outSample.pdf > 0.0f && isfinite(outSample.pdf);
}

inline bool restir_di_provider_sample_environment(thread RISSamplePayload& outSample) {
    outSample.primitiveIndex = kInvalidIndex;
    outSample.flags = kLightPrimitiveFlagDirectional;
    outSample.bary = float2(0.0f);
    outSample.position = float3(0.0f, 1.0f, 0.0f);
    outSample.normal = float3(0.0f, -1.0f, 0.0f);
    outSample.emission = float3(0.0f);
    outSample.pdf = 0.0f;
    return false;
}

inline bool restir_di_provider_sample_regir(thread RISSamplePayload& outSample) {
    outSample.primitiveIndex = kInvalidIndex;
    outSample.flags = 0u;
    outSample.bary = float2(0.0f);
    outSample.position = float3(0.0f);
    outSample.normal = float3(0.0f, 1.0f, 0.0f);
    outSample.emission = float3(0.0f);
    outSample.pdf = 0.0f;
    return false;
}

inline bool restir_di_sample_light_candidate(
    constant PathtraceUniforms& uniforms,
    device const LightPrimitive* emissivePrimitives,
    device const RectData* rectangles,
    device const MaterialData* materials,
    texture2d<float, access::sample> environmentTexture,
    thread const HitRecord& rec,
    thread uint& state,
    const RestirDILightCandidateProviderSet providers,
    thread RISSamplePayload& outSample) {
    if (providers.providerCount == 0u) {
        return false;
    }
    const float providerPdf = 1.0f / float(providers.providerCount);
    const uint selected = min(static_cast<uint>(rand_uniform(state) * float(providers.providerCount)),
                              providers.providerCount - 1u);
    uint ordinal = 0u;
    if ((providers.providerMask & kRestirDILightProviderEmissive) != 0u) {
        if (selected == ordinal) {
            return restir_di_provider_sample_emissive(uniforms,
                                                      emissivePrimitives,
                                                      rec,
                                                      state,
                                                      providerPdf,
                                                      outSample);
        }
        ordinal += 1u;
    }
    if ((providers.providerMask & kRestirDILightProviderAnalytic) != 0u) {
        if (selected == ordinal) {
            return restir_di_provider_sample_analytic_rect(uniforms,
                                                          rectangles,
                                                          materials,
                                                          environmentTexture,
                                                          rec,
                                                          state,
                                                          providers.rectLightCount,
                                                          providerPdf,
                                                          outSample);
        }
        ordinal += 1u;
    }
    if ((providers.providerMask & kRestirDILightProviderEnvironment) != 0u &&
        selected == ordinal) {
        return restir_di_provider_sample_environment(outSample);
    }
    return restir_di_provider_sample_regir(outSample);
}

inline bool build_restir_di_provider_reservoir_for_hit(
    constant PathtraceUniforms& uniforms,
    device const LightPrimitive* emissivePrimitives,
    device const RectData* rectangles,
    device const MaterialData* materials,
    texture2d<float, access::sample> environmentTexture,
    thread const HitRecord& rec,
    const MaterialData material,
    const float3 shadingNormal,
    const float3 wo,
    const FireflyClampParams clampParams,
    const uint candidateCount,
    thread uint& state,
    thread Reservoir& reservoir) {
    reservoir = make_empty_reservoir();
    const RestirDILightCandidateProviderSet providers =
        restir_di_make_provider_set(uniforms, rectangles, materials);
    if (providers.providerCount == 0u) {
        return false;
    }

    const uint risTargetM = max(candidateCount, 1u);
    for (uint i = 0u; i < risTargetM; ++i) {
        RISSamplePayload candidate;
        if (!restir_di_sample_light_candidate(uniforms,
                                              emissivePrimitives,
                                              rectangles,
                                              materials,
                                              environmentTexture,
                                              rec,
                                              state,
                                              providers,
                                              candidate)) {
            continue;
        }
        float distSq = ris_payload_distance_sq(rec, candidate);
        if (!(distSq > 0.0f) || !isfinite(distSq)) {
            continue;
        }
        float3 omegaL = ris_payload_omega_l(rec, candidate);
        float cosThetaSurface = saturate(dot(shadingNormal, omegaL));
        float cosThetaLight = ris_payload_is_directional(candidate)
            ? 1.0f
            : saturate(dot(candidate.normal, -omegaL));
        BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                rec.point,
                                                shadingNormal,
                                                wo,
                                                omegaL,
                                                clampParams,
                                                uniforms.sssMode,
                                                1.0f,
                                                false);
        if (bsdfEval.isDelta || bsdfEval.isBssrdf) {
            continue;
        }
        float phat = p_hat(candidate.emission,
                           bsdfEval.value,
                           cosThetaSurface,
                           cosThetaLight,
                           distSq);
        if (!(phat > 0.0f) || !(candidate.pdf > 0.0f)) {
            continue;
        }
        float weight = phat / candidate.pdf;
        float u = (reservoir.M == 0u) ? 0.0f : rand_uniform(state);
        update_reservoir(reservoir, candidate, weight, u);
    }
    return reservoir.valid && reservoir.M > 0u && reservoir.wSum > 0.0f;
}

inline float reservoir_winner_phat_for_hit(constant PathtraceUniforms& uniforms,
                                           thread const HitRecord& rec,
                                           const MaterialData material,
                                           const float3 shadingNormal,
                                           const float3 wo,
                                           const FireflyClampParams clampParams,
                                           const float diffuseOcclusion,
                                           const bool specularOnly,
                                           const RISSamplePayload winner) {
    float distSq = ris_payload_distance_sq(rec, winner);
    if (!(distSq > 0.0f) || !isfinite(distSq)) {
        return 0.0f;
    }
    float3 omegaL = ris_payload_omega_l(rec, winner);
    float cosThetaSurface = saturate(dot(shadingNormal, omegaL));
    float cosThetaLight = ris_payload_is_directional(winner)
        ? 1.0f
        : saturate(dot(winner.normal, -omegaL));
    BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                            rec.point,
                                            shadingNormal,
                                            wo,
                                            omegaL,
                                            clampParams,
                                            uniforms.sssMode,
                                            diffuseOcclusion,
                                            specularOnly);
    if (bsdfEval.isDelta || bsdfEval.isBssrdf) {
        return 0.0f;
    }
    return p_hat(winner.emission, bsdfEval.value, cosThetaSurface, cosThetaLight, distSq);
}

inline RestirDIReservoirState make_empty_restir_di_reservoir(uint frameTag) {
    RestirDIReservoirState reservoir;
    reservoir.positionPdf = float4(0.0f);
    reservoir.normalTargetPdf = float4(0.0f, 1.0f, 0.0f, 0.0f);
    reservoir.emissionWeight = float4(0.0f);
    reservoir.sampleMeta = uint4(0u, 0u, 0u, frameTag);
    reservoir.candidateCount = 0u;
    reservoir.temporalAccepted = 0u;
    reservoir.spatialAccepted = 0u;
    reservoir.reserved0 = 0u;
    return reservoir;
}

inline RestirDIVisibilityState make_empty_restir_di_visibility(uint frameTag) {
    RestirDIVisibilityState visibility;
    visibility.contributionVisibility = float4(0.0f);
    visibility.sampleMeta = uint4(0u, 0u, 0u, frameTag);
    return visibility;
}

inline bool restir_di_reservoir_state_valid(const RestirDIReservoirState reservoir) {
    return (reservoir.sampleMeta.z & 1u) != 0u &&
           reservoir.candidateCount > 0u &&
           reservoir.emissionWeight.w > 0.0f &&
           reservoir.positionPdf.w > 0.0f &&
           all(isfinite(reservoir.positionPdf)) &&
           all(isfinite(reservoir.normalTargetPdf)) &&
           all(isfinite(reservoir.emissionWeight));
}

inline bool restir_di_previous_history_valid(const RestirDIReservoirState currentReservoir,
                                             const RestirDIReservoirState previousReservoir,
                                             uint frameIndex) {
    if (!restir_di_reservoir_state_valid(currentReservoir) ||
        !restir_di_reservoir_state_valid(previousReservoir)) {
        return false;
    }
    if (frameIndex == 0u || previousReservoir.sampleMeta.w + 1u != frameIndex) {
        return false;
    }
    if (currentReservoir.sampleMeta.x != previousReservoir.sampleMeta.x ||
        currentReservoir.sampleMeta.y != previousReservoir.sampleMeta.y) {
        return false;
    }
    const float currentTarget = currentReservoir.normalTargetPdf.w;
    const float previousTarget = previousReservoir.normalTargetPdf.w;
    if (!(currentTarget > 0.0f) || !(previousTarget > 0.0f)) {
        return false;
    }
    const float targetRatio =
        min(currentTarget, previousTarget) / max(max(currentTarget, previousTarget), 1.0e-6f);
    return targetRatio >= 0.25f;
}

inline RestirDIReservoirState restir_di_merge_temporal(const RestirDIReservoirState currentReservoir,
                                                       const RestirDIReservoirState previousReservoir,
                                                       uint frameIndex) {
    RestirDIReservoirState outReservoir = currentReservoir;
    if (!restir_di_previous_history_valid(currentReservoir, previousReservoir, frameIndex)) {
        outReservoir.temporalAccepted = 0u;
        outReservoir.sampleMeta.z &= ~2u;
        return outReservoir;
    }

    const float currentWeight = currentReservoir.emissionWeight.w;
    const float previousWeight = previousReservoir.emissionWeight.w;
    const float mergedWeight = currentWeight + previousWeight;
    if (!(mergedWeight > 0.0f) || !isfinite(mergedWeight)) {
        outReservoir.temporalAccepted = 0u;
        outReservoir.sampleMeta.z &= ~2u;
        return outReservoir;
    }

    outReservoir.emissionWeight.w = mergedWeight;
    outReservoir.candidateCount =
        max(1u, currentReservoir.candidateCount + previousReservoir.candidateCount);
    outReservoir.temporalAccepted = 1u;
    outReservoir.sampleMeta.z |= 2u;
    outReservoir.sampleMeta.w = frameIndex;
    return outReservoir;
}

inline RestirDIReservoirState restir_di_merge_spatial_neighbor(
    RestirDIReservoirState reservoir,
    const RestirDIReservoirState neighborReservoir,
    uint frameIndex) {
    if (!restir_di_reservoir_state_valid(neighborReservoir)) {
        return reservoir;
    }
    if (!restir_di_reservoir_state_valid(reservoir)) {
        reservoir = neighborReservoir;
        reservoir.spatialAccepted = 1u;
        reservoir.sampleMeta.z |= 4u;
        reservoir.sampleMeta.w = frameIndex;
        return reservoir;
    }

    const float currentWeight = reservoir.emissionWeight.w;
    const float neighborWeight = neighborReservoir.emissionWeight.w;
    const float mergedWeight = currentWeight + neighborWeight;
    if (!(mergedWeight > 0.0f) || !isfinite(mergedWeight)) {
        return reservoir;
    }

    if (neighborWeight > currentWeight &&
        neighborReservoir.normalTargetPdf.w > 0.0f &&
        neighborReservoir.positionPdf.w > 0.0f) {
        reservoir.positionPdf = neighborReservoir.positionPdf;
        reservoir.normalTargetPdf = neighborReservoir.normalTargetPdf;
        reservoir.emissionWeight.xyz = neighborReservoir.emissionWeight.xyz;
        reservoir.sampleMeta.xy = neighborReservoir.sampleMeta.xy;
        reservoir.reserved0 = neighborReservoir.reserved0;
    }
    reservoir.emissionWeight.w = mergedWeight;
    reservoir.candidateCount =
        max(1u, reservoir.candidateCount + neighborReservoir.candidateCount);
    reservoir.spatialAccepted += 1u;
    reservoir.sampleMeta.z |= 4u;
    reservoir.sampleMeta.w = frameIndex;
    return reservoir;
}

inline float3 restir_di_debug_hash_color(uint value) {
    value = pcg_hash(value);
    return float3(float((value >> 0u) & 0xffu) / 255.0f,
                  float((value >> 8u) & 0xffu) / 255.0f,
                  float((value >> 16u) & 0xffu) / 255.0f);
}

inline Ray restir_di_center_primary_ray_for_pixel(constant PathtraceUniforms& uniforms,
                                                  uint2 pixelCoord) {
    float u = (float(pixelCoord.x) + 0.5f) / float(max(uniforms.width, 1u));
    float v = (float(pixelCoord.y) + 0.5f) / float(max(uniforms.height, 1u));
    v = 1.0f - v;

    Ray ray;
    float3 pixelPosition =
        uniforms.lowerLeftCorner + u * uniforms.horizontal + v * uniforms.vertical;
    ray.origin = uniforms.cameraOrigin;
    ray.direction = pixelPosition - ray.origin;
    return ray;
}

inline RestirDIReservoirState restir_di_state_from_reservoir(
    constant PathtraceUniforms& uniforms,
    thread const HitRecord& rec,
    const MaterialData material,
    const float3 shadingNormal,
    const float3 wo,
    const FireflyClampParams clampParams,
    const Reservoir reservoir) {
    RestirDIReservoirState state = make_empty_restir_di_reservoir(uniforms.frameIndex);
    if (!reservoir.valid || reservoir.M == 0u || !(reservoir.wSum > 0.0f)) {
        return state;
    }

    const float targetPdf = reservoir_winner_phat_for_hit(uniforms,
                                                          rec,
                                                          material,
                                                          shadingNormal,
                                                          wo,
                                                          clampParams,
                                                          1.0f,
                                                          false,
                                                          reservoir.winner);
    state.positionPdf = float4(reservoir.winner.position, reservoir.winner.pdf);
    state.normalTargetPdf = float4(reservoir.winner.normal, targetPdf);
    state.emissionWeight = float4(reservoir.winner.emission, reservoir.wSum);
    state.sampleMeta = uint4(reservoir.winner.primitiveIndex,
                             reservoir.winner.flags,
                             1u,
                             uniforms.frameIndex);
    state.candidateCount = reservoir.M;
    state.temporalAccepted = 0u;
    state.spatialAccepted = 0u;
    state.reserved0 = rec.primitiveIndex;
    return state;
}

kernel void restirDiInitializeReservoirsKernel(
    device RestirDIReservoirState* currentReservoirs [[buffer(0)]],
    device RestirDIReservoirState* temporalReservoirs [[buffer(1)]],
    device RestirDIReservoirState* spatialReservoirs [[buffer(2)]],
    device RestirDIVisibilityState* visibilityStates [[buffer(3)]],
    constant RestirDIInitUniforms& uniforms [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uniforms.pixelCount) {
        return;
    }

    const uint frameTag = uniforms.frameIndex;
    const RestirDIReservoirState emptyReservoir = make_empty_restir_di_reservoir(frameTag);
    currentReservoirs[gid] = emptyReservoir;
    temporalReservoirs[gid] = emptyReservoir;
    spatialReservoirs[gid] = emptyReservoir;
    visibilityStates[gid] = make_empty_restir_di_visibility(frameTag);
}

kernel void restirDiInitialCandidatesKernel(
    constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
    device const BvhNode* nodes [[buffer(1)]],
    device const uint* primitiveIndices [[buffer(2)]],
    device const SphereData* spheres [[buffer(3)]],
    device const MaterialData* materials [[buffer(4)]],
    device PathtraceStats* stats [[buffer(5)]],
    device const RectData* rectangles [[buffer(6)]],
    device const TriangleData* triangleData [[buffer(7)]],
    device const BvhNode* tlasNodes [[buffer(8)]],
    device const uint* tlasPrimIndices [[buffer(9)]],
    device const BvhNode* blasNodes [[buffer(10)]],
    device const uint* blasPrimIndices [[buffer(11)]],
    device const SoftwareInstanceInfo* instanceInfos [[buffer(12)]],
    device const LightPrimitive* emissivePrimitives [[buffer(13)]],
    device RestirDIReservoirState* currentReservoirs [[buffer(14)]],
    texture2d<float, access::sample> environmentTexture [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }

    const uint pixelIndex = gid.y * uniforms.width + gid.x;
    currentReservoirs[pixelIndex] = make_empty_restir_di_reservoir(uniforms.frameIndex);
    if (uniforms.materialCount == 0u) {
        return;
    }

    Ray ray = restir_di_center_primary_ray_for_pixel(uniforms, gid);
    HitRecord rec = make_empty_hit_record();
    bool hit = trace_scene_software(uniforms,
                                    spheres,
                                    rectangles,
                                    triangleData,
                                    tlasNodes,
                                    tlasPrimIndices,
                                    instanceInfos,
                                    blasNodes,
                                    blasPrimIndices,
                                    nodes,
                                    primitiveIndices,
                                    stats,
                                    ray,
                                    kEpsilon,
                                    kInfinity,
                                    /*anyHitOnly=*/false,
                                    /*includeTriangles=*/true,
                                    rec);
    if (!hit || rec.materialIndex >= uniforms.materialCount) {
        return;
    }

    MaterialData material = materials[rec.materialIndex];
    const uint type = static_cast<uint>(material.typeEta.x);
    if (type == 3u || type == 7u) {
        return;
    }

    const float3 shadingNormal = safe_normalize(rec.shadingNormal);
    if (!all(isfinite(shadingNormal)) || dot(shadingNormal, shadingNormal) <= 0.0f) {
        return;
    }

    const float3 wo = safe_normalize(-ray.direction);
    FireflyClampParams clampParams = make_firefly_params(uniforms);
    uint rngState = uniforms.fixedRngSeed ^
                    (uniforms.frameIndex * 9781u) ^
                    (gid.x * 6271u) ^
                    (gid.y * 13007u) ^
                    0xD1B54A35u;
    Reservoir reservoir = make_empty_reservoir();
    const uint candidateCount = max(uniforms.risCandidateCount, 1u);
    build_restir_di_provider_reservoir_for_hit(uniforms,
                                               emissivePrimitives,
                                               rectangles,
                                               materials,
                                               environmentTexture,
                                               rec,
                                               material,
                                               shadingNormal,
                                               wo,
                                               clampParams,
                                               candidateCount,
                                               rngState,
                                               reservoir);
    currentReservoirs[pixelIndex] = restir_di_state_from_reservoir(uniforms,
                                                                   rec,
                                                                   material,
                                                                   shadingNormal,
                                                                   wo,
                                                                   clampParams,
                                                                   reservoir);
}

#if __METAL_VERSION__ >= 310
kernel void restirDiInitialCandidatesHardwareKernel(
    constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
    acceleration_structure<instancing> accel [[buffer(1)]],
    device const MeshInfo* meshInfos [[buffer(2)]],
    device const TriangleData* triangleData [[buffer(3)]],
    device const SceneVertex* sceneVertices [[buffer(4)]],
    device const uint3* meshIndices [[buffer(5)]],
    device const uint* instanceUserIds [[buffer(6)]],
    device const SphereData* spheres [[buffer(7)]],
    device const RectData* rectangles [[buffer(8)]],
    device const MaterialData* materials [[buffer(9)]],
    device PathtraceStats* stats [[buffer(10)]],
    device const BvhNode* nodes [[buffer(11)]],
    device const uint* primitiveIndices [[buffer(12)]],
    device const LightPrimitive* emissivePrimitives [[buffer(13)]],
    device RestirDIReservoirState* currentReservoirs [[buffer(14)]],
    texture2d<float, access::sample> environmentTexture [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }

    const uint pixelIndex = gid.y * uniforms.width + gid.x;
    currentReservoirs[pixelIndex] = make_empty_restir_di_reservoir(uniforms.frameIndex);
    if (uniforms.materialCount == 0u) {
        return;
    }

    Ray ray = restir_di_center_primary_ray_for_pixel(uniforms, gid);
    HitRecord rec = make_empty_hit_record();
    bool hit = trace_scene_hardware(uniforms,
                                    accel,
                                    meshInfos,
                                    triangleData,
                                    sceneVertices,
                                    meshIndices,
                                    instanceUserIds,
                                    spheres,
                                    rectangles,
                                    nodes,
                                    primitiveIndices,
                                    stats,
                                    ray,
                                    kEpsilon,
                                    kInfinity,
                                    /*anyHitOnly=*/false,
                                    kInvalidIndex,
                                    kInvalidIndex,
                                    rec);
    if (!hit || rec.materialIndex >= uniforms.materialCount) {
        return;
    }

    MaterialData material = materials[rec.materialIndex];
    const uint type = static_cast<uint>(material.typeEta.x);
    if (type == 3u || type == 7u) {
        return;
    }

    const float3 shadingNormal = safe_normalize(rec.shadingNormal);
    if (!all(isfinite(shadingNormal)) || dot(shadingNormal, shadingNormal) <= 0.0f) {
        return;
    }

    const float3 wo = safe_normalize(-ray.direction);
    FireflyClampParams clampParams = make_firefly_params(uniforms);
    uint rngState = uniforms.fixedRngSeed ^
                    (uniforms.frameIndex * 9781u) ^
                    (gid.x * 6271u) ^
                    (gid.y * 13007u) ^
                    0xD1B54A35u;
    Reservoir reservoir = make_empty_reservoir();
    const uint candidateCount = max(uniforms.risCandidateCount, 1u);
    build_restir_di_provider_reservoir_for_hit(uniforms,
                                               emissivePrimitives,
                                               rectangles,
                                               materials,
                                               environmentTexture,
                                               rec,
                                               material,
                                               shadingNormal,
                                               wo,
                                               clampParams,
                                               candidateCount,
                                               rngState,
                                               reservoir);
    currentReservoirs[pixelIndex] = restir_di_state_from_reservoir(uniforms,
                                                                   rec,
                                                                   material,
                                                                   shadingNormal,
                                                                   wo,
                                                                   clampParams,
                                                                   reservoir);
}
#endif

kernel void restirDiTemporalReuseKernel(
    device const RestirDIReservoirState* currentReservoirs [[buffer(0)]],
    device const RestirDIReservoirState* previousReservoirs [[buffer(1)]],
    device RestirDIReservoirState* temporalReservoirs [[buffer(2)]],
    constant RestirDIInitUniforms& uniforms [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uniforms.pixelCount) {
        return;
    }

    RestirDIReservoirState currentReservoir = currentReservoirs[gid];
    RestirDIReservoirState previousReservoir = previousReservoirs[gid];
    temporalReservoirs[gid] =
        restir_di_merge_temporal(currentReservoir, previousReservoir, uniforms.frameIndex);
}

kernel void restirDiSpatialReuseKernel(
    device const RestirDIReservoirState* temporalReservoirs [[buffer(0)]],
    device RestirDIReservoirState* spatialReservoirs [[buffer(1)]],
    constant RestirDIInitUniforms& uniforms [[buffer(2)]],
    constant uint& neighborCount [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint width = max(uniforms.reserved0, 1u);
    const uint height = max(uniforms.reserved1, 1u);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    const uint pixelIndex = gid.y * width + gid.x;
    RestirDIReservoirState reservoir = temporalReservoirs[pixelIndex];
    constexpr int2 offsets[4] = {
        int2(-1, 0),
        int2(1, 0),
        int2(0, -1),
        int2(0, 1)
    };
    const uint count = min(max(neighborCount, 1u), 4u);
    for (uint i = 0u; i < count; ++i) {
        int2 neighbor = int2(gid) + offsets[i];
        if (neighbor.x < 0 || neighbor.y < 0 ||
            neighbor.x >= int(width) || neighbor.y >= int(height)) {
            continue;
        }
        const uint neighborIndex = uint(neighbor.y) * width + uint(neighbor.x);
        reservoir = restir_di_merge_spatial_neighbor(reservoir,
                                                     temporalReservoirs[neighborIndex],
                                                     uniforms.frameIndex);
    }
    spatialReservoirs[pixelIndex] = reservoir;
}

kernel void restirDiVisibilityKernel(
    constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
    device const BvhNode* nodes [[buffer(1)]],
    device const uint* primitiveIndices [[buffer(2)]],
    device const SphereData* spheres [[buffer(3)]],
    device const MaterialData* materials [[buffer(4)]],
    device PathtraceStats* stats [[buffer(5)]],
    device const RectData* rectangles [[buffer(6)]],
    device const TriangleData* triangleData [[buffer(7)]],
    device const BvhNode* tlasNodes [[buffer(8)]],
    device const uint* tlasPrimIndices [[buffer(9)]],
    device const BvhNode* blasNodes [[buffer(10)]],
    device const uint* blasPrimIndices [[buffer(11)]],
    device const SoftwareInstanceInfo* instanceInfos [[buffer(12)]],
    device const RestirDIReservoirState* spatialReservoirs [[buffer(13)]],
    device RestirDIVisibilityState* visibilityStates [[buffer(14)]],
    device RestirDIReservoirState* previousReservoirs [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }

    const uint pixelIndex = gid.y * uniforms.width + gid.x;
    RestirDIReservoirState reservoir = spatialReservoirs[pixelIndex];
    visibilityStates[pixelIndex] = make_empty_restir_di_visibility(uniforms.frameIndex);
    previousReservoirs[pixelIndex] = reservoir;
    if (!restir_di_reservoir_state_valid(reservoir) || uniforms.materialCount == 0u) {
        return;
    }

    Ray ray = restir_di_center_primary_ray_for_pixel(uniforms, gid);
    HitRecord rec = make_empty_hit_record();
    bool hit = trace_scene_software(uniforms,
                                    spheres,
                                    rectangles,
                                    triangleData,
                                    tlasNodes,
                                    tlasPrimIndices,
                                    instanceInfos,
                                    blasNodes,
                                    blasPrimIndices,
                                    nodes,
                                    primitiveIndices,
                                    stats,
                                    ray,
                                    kEpsilon,
                                    kInfinity,
                                    /*anyHitOnly=*/false,
                                    /*includeTriangles=*/true,
                                    rec);
    if (!hit || rec.materialIndex >= uniforms.materialCount) {
        return;
    }

    MaterialData material = materials[rec.materialIndex];
    const float3 shadingNormal = safe_normalize(rec.shadingNormal);
    if (!all(isfinite(shadingNormal)) || dot(shadingNormal, shadingNormal) <= 0.0f) {
        return;
    }

    const bool directional = (reservoir.sampleMeta.y & kLightPrimitiveFlagDirectional) != 0u;
    RISSamplePayload winner;
    winner.primitiveIndex = reservoir.sampleMeta.x;
    winner.flags = reservoir.sampleMeta.y;
    winner.bary = float2(0.0f);
    winner.position = directional ? safe_normalize(reservoir.positionPdf.xyz)
                                  : reservoir.positionPdf.xyz;
    winner.normal = reservoir.normalTargetPdf.xyz;
    winner.emission = reservoir.emissionWeight.xyz;
    winner.pdf = reservoir.positionPdf.w;

    const float3 omegaL = ris_payload_omega_l(rec, winner);
    const float cosThetaSurface = saturate(dot(shadingNormal, omegaL));
    const float distSq = ris_payload_distance_sq(rec, winner);
    if (!(cosThetaSurface > 0.0f) || !(distSq > 0.0f)) {
        return;
    }

    FireflyClampParams clampParams = make_firefly_params(uniforms);
    BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                            rec.point,
                                            shadingNormal,
                                            safe_normalize(-ray.direction),
                                            omegaL,
                                            clampParams,
                                            uniforms.sssMode,
                                            1.0f,
                                            false);
    if (bsdfEval.isDelta || bsdfEval.isBssrdf) {
        return;
    }

    Ray shadowRay;
    shadowRay.origin = offset_ray_origin(rec, omegaL);
    shadowRay.direction = omegaL;
    HitRecord shadowRec = make_empty_hit_record();
    const float shadowMax = directional
        ? 1.0e20f
        : max(length(winner.position - rec.point) - kEpsilon, kEpsilon);
    bool occluded = trace_scene_software(uniforms,
                                         spheres,
                                         rectangles,
                                         triangleData,
                                         tlasNodes,
                                         tlasPrimIndices,
                                         instanceInfos,
                                         blasNodes,
                                         blasPrimIndices,
                                         nodes,
                                         primitiveIndices,
                                         stats,
                                         shadowRay,
                                         kEpsilon,
                                         shadowMax,
                                         /*anyHitOnly=*/true,
                                         /*includeTriangles=*/true,
                                         shadowRec);
    if (!directional &&
        occluded &&
        shadowRec.primitiveType == kPrimitiveTypeTriangle &&
        shadowRec.primitiveIndex == winner.primitiveIndex) {
        occluded = false;
    }

    const float targetPdf = max(reservoir.normalTargetPdf.w, 1.0e-6f);
    const float reservoirW =
        (reservoir.emissionWeight.w / float(max(reservoir.candidateCount, 1u))) / targetPdf;
    float3 contribution = float3(0.0f);
    const float visible = occluded ? 0.0f : 1.0f;
    if (visible > 0.0f && reservoirW > 0.0f) {
        contribution = bsdfEval.value * winner.emission * (cosThetaSurface * reservoirW);
        if (!all(isfinite(contribution))) {
            contribution = float3(0.0f);
        }
    }

    RestirDIVisibilityState visibility = make_empty_restir_di_visibility(uniforms.frameIndex);
    visibility.contributionVisibility = float4(contribution, visible);
    visibility.sampleMeta = uint4(reservoir.sampleMeta.x,
                                  reservoir.sampleMeta.y,
                                  reservoir.sampleMeta.z,
                                  uniforms.frameIndex);
    visibilityStates[pixelIndex] = visibility;
}

#if __METAL_VERSION__ >= 310
kernel void restirDiVisibilityHardwareKernel(
    constant PathtraceUniforms* uniformsBuffer [[buffer(0)]],
    acceleration_structure<instancing> accel [[buffer(1)]],
    device const MeshInfo* meshInfos [[buffer(2)]],
    device const TriangleData* triangleData [[buffer(3)]],
    device const SceneVertex* sceneVertices [[buffer(4)]],
    device const uint3* meshIndices [[buffer(5)]],
    device const uint* instanceUserIds [[buffer(6)]],
    device const SphereData* spheres [[buffer(7)]],
    device const RectData* rectangles [[buffer(8)]],
    device const MaterialData* materials [[buffer(9)]],
    device PathtraceStats* stats [[buffer(10)]],
    device const BvhNode* nodes [[buffer(11)]],
    device const uint* primitiveIndices [[buffer(12)]],
    device const RestirDIReservoirState* spatialReservoirs [[buffer(13)]],
    device RestirDIVisibilityState* visibilityStates [[buffer(14)]],
    device RestirDIReservoirState* previousReservoirs [[buffer(15)]],
    uint2 gid [[thread_position_in_grid]]) {
    constant PathtraceUniforms& uniforms = uniformsBuffer[0];
    if (gid.x >= uniforms.width || gid.y >= uniforms.height) {
        return;
    }

    const uint pixelIndex = gid.y * uniforms.width + gid.x;
    RestirDIReservoirState reservoir = spatialReservoirs[pixelIndex];
    visibilityStates[pixelIndex] = make_empty_restir_di_visibility(uniforms.frameIndex);
    previousReservoirs[pixelIndex] = reservoir;
    if (!restir_di_reservoir_state_valid(reservoir) || uniforms.materialCount == 0u) {
        return;
    }

    Ray ray = restir_di_center_primary_ray_for_pixel(uniforms, gid);
    HitRecord rec = make_empty_hit_record();
    bool hit = trace_scene_hardware(uniforms,
                                    accel,
                                    meshInfos,
                                    triangleData,
                                    sceneVertices,
                                    meshIndices,
                                    instanceUserIds,
                                    spheres,
                                    rectangles,
                                    nodes,
                                    primitiveIndices,
                                    stats,
                                    ray,
                                    kEpsilon,
                                    kInfinity,
                                    /*anyHitOnly=*/false,
                                    kInvalidIndex,
                                    kInvalidIndex,
                                    rec);
    if (!hit || rec.materialIndex >= uniforms.materialCount) {
        return;
    }

    MaterialData material = materials[rec.materialIndex];
    const float3 shadingNormal = safe_normalize(rec.shadingNormal);
    if (!all(isfinite(shadingNormal)) || dot(shadingNormal, shadingNormal) <= 0.0f) {
        return;
    }

    const bool directional = (reservoir.sampleMeta.y & kLightPrimitiveFlagDirectional) != 0u;
    RISSamplePayload winner;
    winner.primitiveIndex = reservoir.sampleMeta.x;
    winner.flags = reservoir.sampleMeta.y;
    winner.bary = float2(0.0f);
    winner.position = directional ? safe_normalize(reservoir.positionPdf.xyz)
                                  : reservoir.positionPdf.xyz;
    winner.normal = reservoir.normalTargetPdf.xyz;
    winner.emission = reservoir.emissionWeight.xyz;
    winner.pdf = reservoir.positionPdf.w;

    const float3 omegaL = ris_payload_omega_l(rec, winner);
    const float cosThetaSurface = saturate(dot(shadingNormal, omegaL));
    const float distSq = ris_payload_distance_sq(rec, winner);
    if (!(cosThetaSurface > 0.0f) || !(distSq > 0.0f)) {
        return;
    }

    FireflyClampParams clampParams = make_firefly_params(uniforms);
    BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                            rec.point,
                                            shadingNormal,
                                            safe_normalize(-ray.direction),
                                            omegaL,
                                            clampParams,
                                            uniforms.sssMode,
                                            1.0f,
                                            false);
    if (bsdfEval.isDelta || bsdfEval.isBssrdf) {
        return;
    }

    Ray shadowRay;
    shadowRay.origin = offset_ray_origin(rec, omegaL);
    shadowRay.direction = omegaL;
    HitRecord shadowRec = make_empty_hit_record();
    const float shadowMax = directional
        ? 1.0e20f
        : max(length(winner.position - rec.point) - kEpsilon, kEpsilon);
    bool occluded = trace_scene_hardware(uniforms,
                                         accel,
                                         meshInfos,
                                         triangleData,
                                         sceneVertices,
                                         meshIndices,
                                         instanceUserIds,
                                         spheres,
                                         rectangles,
                                         nodes,
                                         primitiveIndices,
                                         stats,
                                         shadowRay,
                                         kEpsilon,
                                         shadowMax,
                                         /*anyHitOnly=*/true,
                                         kInvalidIndex,
                                         kInvalidIndex,
                                         shadowRec);
    if (!directional &&
        occluded &&
        shadowRec.primitiveType == kPrimitiveTypeTriangle &&
        shadowRec.primitiveIndex == winner.primitiveIndex) {
        occluded = false;
    }

    const float targetPdf = max(reservoir.normalTargetPdf.w, 1.0e-6f);
    const float reservoirW =
        (reservoir.emissionWeight.w / float(max(reservoir.candidateCount, 1u))) / targetPdf;
    float3 contribution = float3(0.0f);
    const float visible = occluded ? 0.0f : 1.0f;
    if (visible > 0.0f && reservoirW > 0.0f) {
        contribution = bsdfEval.value * winner.emission * (cosThetaSurface * reservoirW);
        if (!all(isfinite(contribution))) {
            contribution = float3(0.0f);
        }
    }

    RestirDIVisibilityState visibility = make_empty_restir_di_visibility(uniforms.frameIndex);
    visibility.contributionVisibility = float4(contribution, visible);
    visibility.sampleMeta = uint4(reservoir.sampleMeta.x,
                                  reservoir.sampleMeta.y,
                                  reservoir.sampleMeta.z,
                                  uniforms.frameIndex);
    visibilityStates[pixelIndex] = visibility;
}
#endif

kernel void restirDiApplyLightingKernel(
    texture2d<float, access::read_write> radianceSum [[texture(0)]],
    device const RestirDIVisibilityState* visibilityStates [[buffer(0)]],
    constant RestirDIInitUniforms& uniforms [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint width = max(uniforms.reserved0, 1u);
    const uint height = max(uniforms.reserved1, 1u);
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    const uint pixelIndex = gid.y * width + gid.x;
    const RestirDIVisibilityState visibility = visibilityStates[pixelIndex];
    float3 contribution = max(visibility.contributionVisibility.xyz, float3(0.0f));
    if (!all(isfinite(contribution))) {
        contribution = float3(0.0f);
    }

    const float4 current = radianceSum.read(gid);
    radianceSum.write(float4(current.xyz + contribution, current.w), gid);
}

kernel void restirDiDebugAovKernel(
    device const RestirDIReservoirState* currentReservoirs [[buffer(0)]],
    device const RestirDIReservoirState* temporalReservoirs [[buffer(1)]],
    device const RestirDIReservoirState* spatialReservoirs [[buffer(2)]],
    device const RestirDIVisibilityState* visibilityStates [[buffer(3)]],
    device RestirDIReservoirState* debugReservoirs [[buffer(4)]],
    constant RestirDIInitUniforms& uniforms [[buffer(5)]],
    constant uint& debugView [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uniforms.pixelCount) {
        return;
    }

    const RestirDIReservoirState current = currentReservoirs[gid];
    const RestirDIReservoirState temporal = temporalReservoirs[gid];
    const RestirDIReservoirState spatial = spatialReservoirs[gid];
    const RestirDIVisibilityState visibility = visibilityStates[gid];
    RestirDIReservoirState out = spatial;
    if (!restir_di_reservoir_state_valid(out)) {
        out = restir_di_reservoir_state_valid(temporal) ? temporal : current;
    }

    const bool anyInvalid =
        !all(isfinite(current.positionPdf)) ||
        !all(isfinite(current.normalTargetPdf)) ||
        !all(isfinite(current.emissionWeight)) ||
        !all(isfinite(temporal.positionPdf)) ||
        !all(isfinite(temporal.normalTargetPdf)) ||
        !all(isfinite(temporal.emissionWeight)) ||
        !all(isfinite(spatial.positionPdf)) ||
        !all(isfinite(spatial.normalTargetPdf)) ||
        !all(isfinite(spatial.emissionWeight)) ||
        !all(isfinite(visibility.contributionVisibility));
    const float confidence = restir_di_reservoir_state_valid(spatial)
        ? clamp(spatial.emissionWeight.w / max(spatial.emissionWeight.w + 1.0f, 1.0f), 0.0f, 1.0f)
        : 0.0f;

    out.sampleMeta.w = uniforms.frameIndex;
    if (debugView == kDebugViewRestirCandidateSourceId) {
        out.emissionWeight.xyz = restir_di_debug_hash_color(out.sampleMeta.x ^ (out.sampleMeta.y * 8191u));
    } else if (debugView == kDebugViewRestirReservoirConfidence) {
        out.emissionWeight.xyz = float3(confidence, confidence * 0.85f, confidence * 0.2f);
        out.normalTargetPdf.w = confidence;
    } else if (debugView == kDebugViewRestirNanInfMask) {
        out.emissionWeight.xyz = anyInvalid ? float3(1.0f, 0.0f, 0.0f) : float3(0.0f, 0.2f, 0.0f);
        out.sampleMeta.z = anyInvalid ? (out.sampleMeta.z | 8u) : out.sampleMeta.z;
    } else {
        out.emissionWeight.xyz = max(visibility.contributionVisibility.xyz, float3(0.0f));
        out.emissionWeight.w = visibility.contributionVisibility.w;
    }

    debugReservoirs[gid] = out;
}
