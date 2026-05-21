inline float3 trace_path_software(constant PathtraceUniforms& uniforms,
                                  device const SphereData* spheres,
                                  device const RectData* rectangles,
                                  device const TriangleData* triangleData,
                                  device const LightPrimitive* emissivePrimitives,
                                  device const MaterialData* materials,
                                  device const MeshInfo* meshInfos,
                                  device const SceneVertex* sceneVertices,
                                  device const uint3* meshIndices,
                                  Ray ray,
                                  const PrimaryRayDiff primaryRayDiff,
                                  thread uint& state,
                                  // TLAS/BLAS resources (software)
                                  device const BvhNode* tlasNodes,
                                  device const uint* tlasPrimIndices,
                                  device const BvhNode* blasNodes,
                                  device const uint* blasPrimIndices,
                                  device const SoftwareInstanceInfo* instanceInfos,
                                  device const BvhNode* nodes,
                                  device const uint* primitiveIndices,
                                  device PathtraceStats* stats,
                                  texture2d<float, access::sample> environmentTexture,
                                  array<texture2d<float, access::sample>, kMaxMaterialTextures> materialTextures,
                                  array<sampler, kMaxMaterialSamplers> materialSamplers,
                                  device const MaterialTextureInfo* materialTextureInfos,
                                  device const EnvironmentAliasEntry* environmentConditionalAlias,
                                  device const EnvironmentAliasEntry* environmentMarginalAlias,
                                  device const float* environmentPdf,
                                  device PathGuidingReservoirState* pathGuidingStates,
                                  device RestirPtReservoirState* restirPtReservoirs,
                                  device RadianceCacheState* radianceCacheStates,
                                  uint2 pixelCoord,
                                  // Optional AOV outputs
                                  thread float3* outFirstHitAlbedo = nullptr,
                                  thread float3* outFirstHitNormal = nullptr,
                                  thread float4* outFirstHitPosition = nullptr,
                                  thread float4* outFirstHitMaterial = nullptr,
                                  thread PathtraceDebugContext* debugContext = nullptr) {
    float3 throughput = float3(1.0f, 1.0f, 1.0f);
    float3 radiance = float3(0.0f, 0.0f, 0.0f);
    float lastBsdfPdf = 1.0f;
    bool lastScatterWasDelta = true;
    bool isFirstHit = true;
    uint specularDepth = 0u;
    bool hadTransmission = false;
    float envLod = 0.0f;
    bool envLodActive = false;
    RayCone rayCone = make_primary_ray_cone(uniforms);
    HitRecord prevRec;
    bool prevValid = false;
    uint rectLightCount = (rectangles && uniforms.rectangleCount > 0 && materials)
                              ? count_rect_lights(uniforms, rectangles, materials)
                              : 0u;
    const bool envSampling = environment_sampling_available(uniforms,
                                                            environmentConditionalAlias,
                                                            environmentMarginalAlias,
                                                            environmentPdf);
    FireflyClampParams clampParams = make_firefly_params(uniforms);
#define clamp_firefly_contribution(throughput, contribution, params) \
    clamp_firefly_contribution((throughput), (contribution), (params), stats)
#define clamp_path_throughput(throughput, params) \
    clamp_path_throughput((throughput), (params), stats)
    constexpr uint kMaxMediumStack = 8u;
    float3 mediumSigmaStack[kMaxMediumStack];
    for (uint i = 0; i < kMaxMediumStack; ++i) {
        mediumSigmaStack[i] = float3(0.0f);
    }
    uint mediumDepth = 0u;
    uint volumeScatteringDepth = 0u;
    SpectralPathState spectralState = spectral_make_path_state(uniforms, state, stats);

    for (uint depth = 0; depth < uniforms.maxDepth; ++depth) {
        HitRecord rec;
        uint excludeMeshIndex = kInvalidIndex;
        uint excludePrimitiveIndex = kInvalidIndex;
        if (prevValid) {
            compute_exclusion_indices(prevRec, excludeMeshIndex, excludePrimitiveIndex);
        }
        if (!trace_scene_software_with_exclusion(uniforms,
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
                          excludeMeshIndex,
                          excludePrimitiveIndex,
                          rec)) {
            if (uniforms.debugViewMode != kDebugViewNone) {
                return float3(0.0f);
            }
            float3 background = sky_color(ray.direction);
            if (uniforms.backgroundMode == 1u) {
                background = uniforms.backgroundColor;
            } else if (uniforms.backgroundMode == 2u && environmentTexture.get_width() > 0 && environmentTexture.get_height() > 0) {
                float overrideLod = 0.0f;
                bool useOverride = environment_mip_override(uniforms, environmentTexture, overrideLod);
                if (useOverride) {
                    background = environment_color_lod(environmentTexture,
                                                       ray.direction,
                                                       uniforms.environmentRotation,
                                                       uniforms.environmentIntensity,
                                                       overrideLod,
                                                       uniforms);
                } else if (envLodActive) {
                    background = environment_color_lod(environmentTexture,
                                                       ray.direction,
                                                       uniforms.environmentRotation,
                                                       uniforms.environmentIntensity,
                                                       envLod,
                                                       uniforms);
                } else {
                    background = environment_color(environmentTexture,
                                                   ray.direction,
                                                   uniforms.environmentRotation,
                                                   uniforms.environmentIntensity,
                                                   uniforms);
                }
            }
            if (uniforms.backgroundMode != 2u) {
                background = to_working_space(background, uniforms);
            }
            if (debugContext) {
                record_debug_event(*debugContext,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   /*frontFace=*/0u,
                                   kInvalidIndex,
                                   /*scatterIsDelta=*/false,
                                   throughput);
            }
            float misWeight = 1.0f;
            bool useSpecularMis =
                use_visible_emitter_mis(depth, lastScatterWasDelta, uniforms);
            if (useSpecularMis && envSampling) {
                float lightPdf = environment_pdf(uniforms, environmentPdf, ray.direction);
                float denom = lastBsdfPdf + lightPdf;
                if (denom > 0.0f) {
                    misWeight = clamp(lastBsdfPdf / denom,
                                      kMisWeightClampMin,
                                      kMisWeightClampMax);
                }
            }
            float3 contribution = background * misWeight;
            radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
            if (debugContext) {
                record_debug_event(*debugContext,
                                   kDebugEventBackground,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   /*frontFace=*/0u,
                                   kInvalidIndex,
                                   /*scatterIsDelta=*/false,
                                   throughput,
                                   float4(misWeight, lastBsdfPdf, 0.0f, 0.0f),
                                   contribution);
            }
            break;
        }

        prevRec = rec;
        prevValid = true;

        if (!materials || uniforms.materialCount == 0) {
            break;
        }
        if (volume_transport_enabled(uniforms)) {
            MediumDescriptor medium = volume_make_global_medium(uniforms);
            const float segment = max(rec.t, 0.0f);
            const float3 segmentEmission = volume_emission_integral(medium, segment);
            if (any(segmentEmission > float3(0.0f))) {
                radiance += clamp_firefly_contribution(throughput, segmentEmission, clampParams);
            }
            const uint maxVolumeEvents = max(medium.metadata.y, 1u);
            bool sampledVolumeEvent = false;
            if (volumeScatteringDepth < maxVolumeEvents &&
                any(medium.sigmaSAnisotropy.xyz > float3(0.0f))) {
                float distancePdf = 1.0f;
                const float sampledDistance = volume_sample_distance(medium, state, distancePdf);
                sampledVolumeEvent = sampledDistance > 0.0f && sampledDistance < segment;
                if (sampledVolumeEvent) {
                    const float3 scatterPoint = ray.origin + ray.direction * sampledDistance;
                    const float3 transmittance = volume_transmittance(medium, sampledDistance, stats);
                    const float majorant = max(volume_majorant(medium), 1.0e-6f);
                    throughput *= transmittance * (medium.sigmaSAnisotropy.xyz / majorant);
                    if (rectLightCount > 0u && medium.metadata.z != 0u) {
                        HitRecord volumeRec = rec;
                        volumeRec.point = scatterPoint;
                        RectLightSample lightSample;
                        if (sample_rect_light(uniforms,
                                              rectangles,
                                              materials,
                                              environmentTexture,
                                              volumeRec,
                                              state,
                                              rectLightCount,
                                              lightSample)) {
                            Ray shadowRay;
                            shadowRay.origin = scatterPoint + lightSample.direction * kRayOriginEpsilon;
                            shadowRay.direction = lightSample.direction;
                            HitRecord shadowRec;
                            const float shadowMax = max(lightSample.distance - kEpsilon, kEpsilon);
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
                                                                 nullptr,
                                                                 shadowRec);
                            if (!occluded && lightSample.pdf > 0.0f) {
                                const float phase = volume_phase_eval(medium, -ray.direction, lightSample.direction);
                                const float3 shadowTr = volume_transmittance(medium, shadowMax, stats);
                                const float3 contribution =
                                    lightSample.emission * phase * shadowTr / max(lightSample.pdf, 1.0e-6f);
                                radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                                if (stats) {
                                    atomic_fetch_add_explicit(&stats->volumeNeeRayCount, 1u, memory_order_relaxed);
                                }
                            }
                        }
                    }
                    float phasePdf = 1.0f;
                    const float3 phaseDir = volume_phase_sample(medium, -ray.direction, state, phasePdf);
                    if (stats) {
                        atomic_fetch_add_explicit(&stats->volumeScatterEventCount, 1u, memory_order_relaxed);
                        atomic_fetch_add_explicit(&stats->volumePhaseSampleCount, 1u, memory_order_relaxed);
                    }
                    volumeScatteringDepth += 1u;
                    lastScatterWasDelta = false;
                    lastBsdfPdf = max(phasePdf, 1.0e-6f);
                    ray.origin = scatterPoint + phaseDir * kRayOriginEpsilon;
                    ray.direction = phaseDir;
                    prevValid = false;
                    continue;
                }
            } else if (any(medium.sigmaSAnisotropy.xyz > float3(0.0f)) && stats) {
                atomic_fetch_add_explicit(&stats->volumeFallbackCount, 1u, memory_order_relaxed);
            }
            throughput *= volume_transmittance(medium, segment, stats);
        }
        if (mediumDepth > 0u) {
            float3 sigma = mediumSigmaStack[mediumDepth - 1u];
            if (any(sigma > float3(0.0f))) {
                float segment = max(rec.t, 0.0f);
                float3 attenuation = exp(-sigma * segment);
                throughput *= attenuation;
            }
        }
        uint matIndex = min(rec.materialIndex, uniforms.materialCount - 1);
        MaterialData material = materials[matIndex];
        spectral_apply_material(material, uniforms, spectralState, stats);
        uint type = static_cast<uint>(material.typeEta.x);
        float3 incidentDir = normalize(ray.direction);
        float3 wo = -incidentDir;
        float hitDistanceWorld = ray_segment_world_length(ray, rec.t);
        bool surfaceIsDelta = material_is_delta(material);
        bool specularOnly = (uniforms.debugSpecularOnly != 0u);
        float diffuseOcclusion = 1.0f;
        float3 debugBaseColor = material_base_color(material);
        float2 debugBaseColorUv = float2(0.0f);
        float debugBaseColorLod = 0.0f;
        float debugMetallic = 0.0f;
        float debugRoughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        float debugAO = 1.0f;
        float3 shadingNormal = rec.shadingNormal;
        float3 debugVtxNormalRaw = float3(0.0f);
        float3 debugVtxNormal = float3(0.0f);
        if (!all(isfinite(shadingNormal)) || dot(shadingNormal, shadingNormal) <= 0.0f) {
            shadingNormal = rec.normal;
        }
        if (rec.primitiveType == kPrimitiveTypeTriangle &&
            meshInfos && sceneVertices && meshIndices && uniforms.meshCount > 0u) {
            float3 candidateRaw = interpolate_shading_normal_raw(uniforms,
                                                                 rec.meshIndex,
                                                                 rec.primitiveIndex,
                                                                 rec.barycentric,
                                                                 meshInfos,
                                                                 sceneVertices,
                                                                 meshIndices);
            float3 candidate = interpolate_shading_normal(uniforms,
                                                          rec.meshIndex,
                                                          rec.primitiveIndex,
                                                          rec.barycentric,
                                                          meshInfos,
                                                          sceneVertices,
                                                          meshIndices);
            if (all(isfinite(candidate)) && dot(candidate, candidate) > 0.0f) {
                debugVtxNormalRaw = candidateRaw;
                if (dot(candidate, rec.normal) < 0.0f) {
                    candidate = -candidate;
                }
                shadingNormal = normalize(candidate);
                debugVtxNormal = shadingNormal;
            }
        }
        if (type == 2u) { // Dielectric: force geometric normal for shading.
            float3 geomNormal = rec.normal;
            if (all(isfinite(geomNormal)) && dot(geomNormal, geomNormal) > 0.0f) {
                shadingNormal = geomNormal;
            }
            // Keep ray offsets consistent between SWRT/HWRT for glass.
            rec.shadingNormal = shadingNormal;
        }

        if (type != 7u &&
            rec.primitiveType == kPrimitiveTypeTriangle &&
            meshInfos && sceneVertices && meshIndices && uniforms.meshCount > 0u &&
            material_texture_valid(uniforms, material.textureIndices0.x)) {
            float2 uv0 = interpolate_uv(uniforms,
                                        rec.meshIndex,
                                        rec.primitiveIndex,
                                        rec.barycentric,
                                        0u,
                                        meshInfos,
                                        sceneVertices,
                                        meshIndices);
            float2 uv1 = interpolate_uv(uniforms,
                                        rec.meshIndex,
                                        rec.primitiveIndex,
                                        rec.barycentric,
                                        1u,
                                        meshInfos,
                                        sceneVertices,
                                        meshIndices);
            PbrTextureSamplingContext baseColorCtx = make_pbr_texture_sampling_context(material,
                                                                                       kPbrTextureSlotBaseColor,
                                                                                       uv0,
                                                                                       uv1,
                                                                                       false,
                                                                                       float2(0.0f),
                                                                                       float2(0.0f),
                                                                                       0.0f,
                                                                                       false,
                                                                                       float2(0.0f),
                                                                                       float2(0.0f),
                                                                                       0.0f);
            float4 baseColorSample =
                sample_material_texture_filtered(materialTextures,
                                                materialSamplers,
                                                materialTextureInfos,
                                                uniforms,
                                                material.textureIndices0.x,
                                                baseColorCtx.uv,
                                                float4(1.0f),
                                                0.0f,
                                                false,
                                                float2(0.0f),
                                                float2(0.0f));
            float3 baseFactor = to_working_space(material.baseColorRoughness.xyz, uniforms);
            float3 baseColorSampleRgb = to_working_space(baseColorSample.xyz, uniforms);
            material.baseColorRoughness.xyz = baseFactor * baseColorSampleRgb;
            debugBaseColor = material.baseColorRoughness.xyz;
            debugBaseColorUv = baseColorCtx.uv;
            debugBaseColorLod = 0.0f;
        }

        if (type == 7u && rec.primitiveType == kPrimitiveTypeTriangle &&
            meshInfos && sceneVertices && meshIndices && uniforms.meshCount > 0u) {
            float2 uv0 = interpolate_uv(uniforms,
                                        rec.meshIndex,
                                        rec.primitiveIndex,
                                        rec.barycentric,
                                        0u,
                                        meshInfos,
                                        sceneVertices,
                                        meshIndices);
            float2 uv1 = interpolate_uv(uniforms,
                                        rec.meshIndex,
                                        rec.primitiveIndex,
                                        rec.barycentric,
                                        1u,
                                        meshInfos,
                                        sceneVertices,
                                        meshIndices);
            float4 tangent = interpolate_tangent(uniforms,
                                                 rec.meshIndex,
                                                 rec.primitiveIndex,
                                                 rec.barycentric,
                                                 meshInfos,
                                                 sceneVertices,
                                                 meshIndices);
            float4 tangentRaw4 = interpolate_tangent_raw(uniforms,
                                                         rec.meshIndex,
                                                         rec.primitiveIndex,
                                                         rec.barycentric,
                                                         meshInfos,
                                                         sceneVertices,
                                                         meshIndices);
            float3 debugBaryWeights = barycentric_weights_saturated(rec.barycentric);
            float3 debugTriIndices = float3(0.0f);
            float2 debugNormalUv = float2(0.0f);
            float3 debugTangentRaw = tangentRaw4.xyz;
            float debugTangentW = tangentRaw4.w;
            float3 debugTangent = float3(0.0f);
            float3 debugBitangent = float3(0.0f);
            float3 debugTexelRaw = float3(0.5f, 0.5f, 1.0f);
            float3 debugTexelDecoded = float3(0.0f, 0.0f, 1.0f);
            float3 debugSnTangentSpace = float3(0.0f, 0.0f, 1.0f);
            float3 debugSnWorld = shadingNormal;
            if (material.typeEta.z > 0.5f) {
                rec.twoSided = 1u;
            }
            if (meshInfos && meshIndices && uniforms.meshCount > 0u) {
                uint clampedMesh = min(rec.meshIndex, uniforms.meshCount - 1u);
                MeshInfo info = meshInfos[clampedMesh];
                if (rec.primitiveIndex >= info.triangleOffset) {
                    uint localIndex = rec.primitiveIndex - info.triangleOffset;
                    if (localIndex < info.indexCount) {
                        uint indexEntry = info.indexOffset + localIndex;
                        uint3 triIndices = meshIndices[indexEntry];
                        debugTriIndices = float3(float(triIndices.x),
                                                 float(triIndices.y),
                                                 float(triIndices.z));
                    }
                }
            }
            float coneFootprintWorld = ray_cone_width_at_distance(rayCone, hitDistanceWorld);
            float surfaceFootprintWorld =
                surface_footprint_from_cone(coneFootprintWorld, rec.normal, wo);
            float3 dPdu0 = float3(0.0f);
            float3 dPdv0 = float3(0.0f);
            float uvPerWorld0 = 0.0f;
            bool hasSurfacePartials0 = triangle_surface_partials(uniforms,
                                                                 rec.meshIndex,
                                                                 rec.primitiveIndex,
                                                                 0u,
                                                                 meshInfos,
                                                                 sceneVertices,
                                                                 meshIndices,
                                                                 dPdu0,
                                                                 dPdv0,
                                                                 uvPerWorld0);
            float3 dPdu1 = float3(0.0f);
            float3 dPdv1 = float3(0.0f);
            float uvPerWorld1 = 0.0f;
            bool hasSurfacePartials1 = triangle_surface_partials(uniforms,
                                                                 rec.meshIndex,
                                                                 rec.primitiveIndex,
                                                                 1u,
                                                                 meshInfos,
                                                                 sceneVertices,
                                                                 meshIndices,
                                                                 dPdu1,
                                                                 dPdv1,
                                                                 uvPerWorld1);
            float2 dUVdx0 = float2(0.0f);
            float2 dUVdy0 = float2(0.0f);
            bool hasIgehyGradients0 = false;
            if (depth == 0u && hasSurfacePartials0) {
                float3 dudP = float3(0.0f);
                float3 dvdP = float3(0.0f);
                if (uv_world_gradients_from_partials(dPdu0, dPdv0, dudP, dvdP)) {
                    hasIgehyGradients0 = first_hit_uv_gradients_igehy(ray,
                                                                      primaryRayDiff,
                                                                      rec.t,
                                                                      rec.normal,
                                                                      dudP,
                                                                      dvdP,
                                                                      dUVdx0,
                                                                      dUVdy0);
                }
            }
            float2 dUVdx1 = float2(0.0f);
            float2 dUVdy1 = float2(0.0f);
            bool hasIgehyGradients1 = false;
            if (depth == 0u && hasSurfacePartials1) {
                float3 dudP = float3(0.0f);
                float3 dvdP = float3(0.0f);
                if (uv_world_gradients_from_partials(dPdu1, dPdv1, dudP, dvdP)) {
                    hasIgehyGradients1 = first_hit_uv_gradients_igehy(ray,
                                                                      primaryRayDiff,
                                                                      rec.t,
                                                                      rec.normal,
                                                                      dudP,
                                                                      dvdP,
                                                                      dUVdx1,
                                                                      dUVdy1);
                }
            }

            PbrTextureSamplingContext baseColorCtx = make_pbr_texture_sampling_context(material,
                                                                                        kPbrTextureSlotBaseColor,
                                                                                        uv0,
                                                                                        uv1,
                                                                                        hasIgehyGradients0,
                                                                                        dUVdx0,
                                                                                        dUVdy0,
                                                                                        uvPerWorld0,
                                                                                        hasIgehyGradients1,
                                                                                        dUVdx1,
                                                                                        dUVdy1,
                                                                                        uvPerWorld1);
            PbrTextureSamplingContext ormCtx = make_pbr_texture_sampling_context(material,
                                                                                  kPbrTextureSlotMetallicRoughness,
                                                                                  uv0,
                                                                                  uv1,
                                                                                  hasIgehyGradients0,
                                                                                  dUVdx0,
                                                                                  dUVdy0,
                                                                                  uvPerWorld0,
                                                                                  hasIgehyGradients1,
                                                                                  dUVdx1,
                                                                                  dUVdy1,
                                                                                  uvPerWorld1);
            PbrTextureSamplingContext normalCtx = make_pbr_texture_sampling_context(material,
                                                                                     kPbrTextureSlotNormal,
                                                                                     uv0,
                                                                                     uv1,
                                                                                     hasIgehyGradients0,
                                                                                     dUVdx0,
                                                                                     dUVdy0,
                                                                                     uvPerWorld0,
                                                                                     hasIgehyGradients1,
                                                                                     dUVdx1,
                                                                                     dUVdy1,
                                                                                     uvPerWorld1);
            debugNormalUv = normalCtx.uv;
            PbrTextureSamplingContext occlusionCtx = make_pbr_texture_sampling_context(material,
                                                                                        kPbrTextureSlotOcclusion,
                                                                                        uv0,
                                                                                        uv1,
                                                                                        hasIgehyGradients0,
                                                                                        dUVdx0,
                                                                                        dUVdy0,
                                                                                        uvPerWorld0,
                                                                                        hasIgehyGradients1,
                                                                                        dUVdx1,
                                                                                        dUVdy1,
                                                                                        uvPerWorld1);
            PbrTextureSamplingContext emissiveCtx = make_pbr_texture_sampling_context(material,
                                                                                       kPbrTextureSlotEmissive,
                                                                                       uv0,
                                                                                       uv1,
                                                                                       hasIgehyGradients0,
                                                                                       dUVdx0,
                                                                                       dUVdy0,
                                                                                       uvPerWorld0,
                                                                                       hasIgehyGradients1,
                                                                                       dUVdx1,
                                                                                       dUVdy1,
                                                                                       uvPerWorld1);
            PbrTextureSamplingContext transmissionCtx = make_pbr_texture_sampling_context(material,
                                                                                           kPbrTextureSlotTransmission,
                                                                                           uv0,
                                                                                           uv1,
                                                                                           hasIgehyGradients0,
                                                                                           dUVdx0,
                                                                                           dUVdy0,
                                                                                           uvPerWorld0,
                                                                                           hasIgehyGradients1,
                                                                                           dUVdx1,
                                                                                           dUVdy1,
                                                                                           uvPerWorld1);
            float3 baseFactor = to_working_space(material.baseColorRoughness.xyz, uniforms);
            float baseColorLod =
                material_texture_lod_with_fallback(materialTextures,
                                                   materialTextureInfos,
                                                   uniforms,
                                                   material.textureIndices0.x,
                                                   baseColorCtx.hasIgehyGradients,
                                                   baseColorCtx.dUVdx,
                                                   baseColorCtx.dUVdy,
                                                   baseColorCtx.uvPerWorld,
                                                   surfaceFootprintWorld);
            if ((material.materialFlags & kMaterialFlagForceBaseColorMip0) != 0u) {
                baseColorLod = 0.0f;
            }
            float4 baseColorSample =
                sample_material_texture_filtered(materialTextures,
                                                materialSamplers,
                                                materialTextureInfos,
                                                uniforms,
                                                material.textureIndices0.x,
                                                baseColorCtx.uv,
                                                float4(1.0f),
                                                baseColorLod,
                                                baseColorCtx.hasIgehyGradients,
                                                baseColorCtx.dUVdx,
                                                baseColorCtx.dUVdy);
            if ((material.materialFlags & kMaterialFlagBlackKeyAlphaFromRgb) != 0u) {
                // MASTER_Side_Letters atlas carries dark plaque texels in RGB.
                // Gate alpha by luminance so only bright lettering survives.
                float luma = dot(baseColorSample.xyz, kLuminanceWeights);
                float alphaFromRgb = smoothstep(0.72f, 0.90f, luma);
                baseColorSample.w = alphaFromRgb;
            }
            float3 baseColorSampleRgb = to_working_space(baseColorSample.xyz, uniforms);
            float3 baseColor = baseFactor * baseColorSampleRgb;
            debugBaseColorUv = baseColorCtx.uv;
            debugBaseColorLod = baseColorLod;

            float metallic = clamp(material.pbrParams.x, 0.0f, 1.0f);
            float roughness = clamp(material.pbrParams.y, 0.0f, 1.0f);
            float normalStrengthScale = 1.0f;
#if PT_DEBUG_TOOLS
            normalStrengthScale = max(uniforms.debugNormalStrengthScale, 0.0f);
#endif
            float normalScale = material.pbrParams.w * normalStrengthScale;
            bool disableOrmByMaterial = (material.materialFlags & kMaterialFlagDisableOrm) != 0u;
            bool useOrmTexture = !disableOrmByMaterial &&
                                 material_texture_valid(uniforms, material.textureIndices0.y);
#if PT_DEBUG_TOOLS
            useOrmTexture = useOrmTexture && (uniforms.debugDisableOrmTexture == 0u);
#endif
            if (useOrmTexture) {
                float ormLod =
                    material_texture_lod_with_fallback(materialTextures,
                                                       materialTextureInfos,
                                                       uniforms,
                                                       material.textureIndices0.y,
                                                       ormCtx.hasIgehyGradients,
                                                       ormCtx.dUVdx,
                                                       ormCtx.dUVdy,
                                                       ormCtx.uvPerWorld,
                                                       surfaceFootprintWorld);
#if PT_DEBUG_TOOLS
                ormLod = max(ormLod + uniforms.debugOrmLodBias, 0.0f);
#endif
                float3 mrSample =
                    sample_material_texture_level(materialTextures,
                                                  materialSamplers,
                                                  materialTextureInfos,
                                                  uniforms,
                                                  material.textureIndices0.y,
                                                  ormCtx.uv,
                                                  float4(1.0f),
                                                  ormLod).xyz;
                metallic = clamp(mrSample.z * metallic, 0.0f, 1.0f);
                roughness = clamp(mrSample.y * roughness, 0.0f, 1.0f);
            }
            float visorMask = visor_override_blend(baseColor, metallic, roughness, matIndex, uniforms);
            if (visorMask > 0.0f) {
                    float overrideRoughness =
                        clamp(uniforms.debugVisorOverrideRoughness, 0.0f, 1.0f);
                    float overrideF0 = clamp(uniforms.debugVisorOverrideF0, 0.0f, 0.12f);
                    metallic = mix(metallic, 0.0f, visorMask);
                    roughness = mix(roughness, overrideRoughness, visorMask);
                    material.typeEta.y = mix(material.typeEta.y,
                                             ior_from_f0(overrideF0),
                                             visorMask);
            }
            float normalLod =
                material_texture_lod_with_fallback(materialTextures,
                                                   materialTextureInfos,
                                                   uniforms,
                                                   material.textureIndices0.z,
                                                   normalCtx.hasIgehyGradients,
                                                   normalCtx.dUVdx,
                                                   normalCtx.dUVdy,
                                                   normalCtx.uvPerWorld,
                                                   surfaceFootprintWorld);
#if PT_DEBUG_TOOLS
            normalLod = max(normalLod + uniforms.debugNormalLodBias, 0.0f);
#endif

            float transmission = clamp(material.pbrExtras.z, 0.0f, 1.0f);
            if (material_texture_valid(uniforms, material.textureIndices1.y)) {
                float transmissionLod =
                    material_texture_lod_with_fallback(materialTextures,
                                                       materialTextureInfos,
                                                       uniforms,
                                                       material.textureIndices1.y,
                                                       transmissionCtx.hasIgehyGradients,
                                                       transmissionCtx.dUVdx,
                                                       transmissionCtx.dUVdy,
                                                       transmissionCtx.uvPerWorld,
                                                       surfaceFootprintWorld);
                float transmissionSample =
                    sample_material_texture_filtered(materialTextures,
                                                    materialSamplers,
                                                    materialTextureInfos,
                                                    uniforms,
                                                    material.textureIndices1.y,
                                                    transmissionCtx.uv,
                                                    float4(1.0f),
                                                    transmissionLod,
                                                    transmissionCtx.hasIgehyGradients,
                                                    transmissionCtx.dUVdx,
                                                    transmissionCtx.dUVdy).x;
                transmission = clamp(transmission * transmissionSample, 0.0f, 1.0f);
            }
            transmission *= (1.0f - metallic);

            float alpha = clamp(material.pbrExtras.x, 0.0f, 1.0f);
            alpha = clamp(alpha * baseColorSample.w, 0.0f, 1.0f);
            float alphaCutoff = clamp(material.pbrExtras.y, 0.0f, 1.0f);
            float alphaMode = material.pbrExtras.w;
            if (alphaMode > 0.5f) {
                bool discard = false;
                if (alphaMode < 1.5f) {
                    discard = alpha < alphaCutoff;
                } else {
                    discard = rand_uniform(state) > alpha;
                }
                if (discard) {
                    ray.origin = offset_ray_origin(rec, ray.direction);
                    prevRec = rec;
                    prevValid = true;
                    lastBsdfPdf = 1.0f;
                    lastScatterWasDelta = true;
                    specularDepth += 1u;
                    continue;
                }
            }

            material.pbrExtras.z = transmission;

            float occlusion = 1.0f;
            if (!disableOrmByMaterial && material_texture_valid(uniforms, material.textureIndices0.w)) {
                float occlusionLod =
                    material_texture_lod_with_fallback(materialTextures,
                                                       materialTextureInfos,
                                                       uniforms,
                                                       material.textureIndices0.w,
                                                       occlusionCtx.hasIgehyGradients,
                                                       occlusionCtx.dUVdx,
                                                       occlusionCtx.dUVdy,
                                                       occlusionCtx.uvPerWorld,
                                                       surfaceFootprintWorld);
                float occSample =
                    sample_material_texture_filtered(materialTextures,
                                                    materialSamplers,
                                                    materialTextureInfos,
                                                    uniforms,
                                                    material.textureIndices0.w,
                                                    occlusionCtx.uv,
                                                    float4(1.0f),
                                                    occlusionLod,
                                                    occlusionCtx.hasIgehyGradients,
                                                    occlusionCtx.dUVdx,
                                                    occlusionCtx.dUVdy).x;
                occlusion = mix(1.0f, occSample, clamp(material.pbrParams.z, 0.0f, 1.0f));
            }
            debugAO = occlusion;
            diffuseOcclusion = (uniforms.debugDisableAO != 0u) ? 1.0f : occlusion;
            if (uniforms.debugAoIndirectOnly != 0u && depth == 0u) {
                diffuseOcclusion = 1.0f;
            }
            debugBaseColor = baseColor;
            debugMetallic = metallic;
            debugRoughness = roughness;

            float3 emissive = to_working_space(material.emission.xyz, uniforms);
            if (material_texture_valid(uniforms, material.textureIndices1.x)) {
                float emissiveLod =
                    material_texture_lod_with_fallback(materialTextures,
                                                       materialTextureInfos,
                                                       uniforms,
                                                       material.textureIndices1.x,
                                                       emissiveCtx.hasIgehyGradients,
                                                       emissiveCtx.dUVdx,
                                                       emissiveCtx.dUVdy,
                                                       emissiveCtx.uvPerWorld,
                                                       surfaceFootprintWorld);
                float3 emissiveSample =
                    sample_material_texture_filtered(materialTextures,
                                                    materialSamplers,
                                                    materialTextureInfos,
                                                    uniforms,
                                                    material.textureIndices1.x,
                                                    emissiveCtx.uv,
                                                    float4(1.0f),
                                                    emissiveLod,
                                                    emissiveCtx.hasIgehyGradients,
                                                    emissiveCtx.dUVdx,
                                                    emissiveCtx.dUVdy).xyz;
                emissiveSample = to_working_space(emissiveSample, uniforms);
                emissive *= emissiveSample;
            }

            bool useNormalMap = material_texture_valid(uniforms, material.textureIndices0.z);
#if PT_DEBUG_TOOLS
            if (uniforms.debugDisableNormalMap != 0u) {
                useNormalMap = false;
            }
#endif
            if (normalScale <= 1.0e-4f) {
                useNormalMap = false;
            }
            float normalLength = 1.0f;
            float3 normalSampleTs = float3(0.0f, 0.0f, 1.0f);
            bool flipNormalGreen = false;
#if PT_DEBUG_TOOLS
            flipNormalGreen = uniforms.debugFlipNormalGreen != 0u;
#endif
            if (useNormalMap) {
                debugTexelRaw =
                    sample_material_texture_filtered(materialTextures,
                                                    materialSamplers,
                                                    materialTextureInfos,
                                                    uniforms,
                                                    material.textureIndices0.z,
                                                    normalCtx.uv,
                                                    float4(0.5f, 0.5f, 1.0f, 1.0f),
                                                    normalLod,
                                                    normalCtx.hasIgehyGradients,
                                                    normalCtx.dUVdx,
                                                    normalCtx.dUVdy).xyz;
                normalSampleTs = decode_normal_map(debugTexelRaw,
                                                   normalScale,
                                                   flipNormalGreen,
                                                   normalLength);
                debugTexelDecoded = normalSampleTs;
                debugSnTangentSpace = normalSampleTs;
                float3 t = tangent.xyz;
                float3 b = float3(0.0f);
                bool hasBasis = false;
                bool trustVertexTangent = fabs(tangent.w) > 0.5f;
                if (trustVertexTangent && all(isfinite(t)) && dot(t, t) > 1.0e-6f) {
                    t = normalize(t - shadingNormal * dot(shadingNormal, t));
                    if (all(isfinite(t)) && dot(t, t) > 1.0e-6f) {
                        float tangentSign = (tangent.w < 0.0f) ? -1.0f : 1.0f;
                        b = normalize(cross(shadingNormal, t)) * tangentSign;
                        if (all(isfinite(b)) && dot(b, b) > 1.0e-6f) {
                            hasBasis = true;
                        }
                    }
                }
                if (!hasBasis) {
                    uint normalUvSet = pbr_texture_uv_set(material, kPbrTextureSlotNormal);
                    hasBasis = compute_tangent_basis_from_uv(uniforms,
                                                             rec.meshIndex,
                                                             rec.primitiveIndex,
                                                             normalUvSet,
                                                             meshInfos,
                                                             sceneVertices,
                                                             meshIndices,
                                                             shadingNormal,
                                                             t,
                                                             b);
                }
                if (!hasBasis) {
                    build_onb(shadingNormal, t, b);
                }
                float3 mapped = normalize(t * normalSampleTs.x +
                                          b * normalSampleTs.y +
                                          shadingNormal * normalSampleTs.z);
                if (dot(mapped, rec.normal) < 0.0f) {
                    mapped = -mapped;
                }
                debugTangent = t;
                debugBitangent = b;
                shadingNormal = mapped;
                debugSnWorld = shadingNormal;
            }

            if (useNormalMap) {
                float tok = max((1.0f - normalLength) / max(normalLength, 1.0e-6f), 0.0f);
                if (normalCtx.hasIgehyGradients &&
                    all(isfinite(normalCtx.dUVdx)) &&
                    all(isfinite(normalCtx.dUVdy))) {
                    float gradMag = max(max(fabs(normalCtx.dUVdx.x), fabs(normalCtx.dUVdx.y)),
                                        max(fabs(normalCtx.dUVdy.x), fabs(normalCtx.dUVdy.y)));
                    if (gradMag > 1.0e-6f && gradMag < 4.0f) {
                        float3 nDx = sample_material_texture_level(materialTextures,
                                                                   materialSamplers,
                                                                   materialTextureInfos,
                                                                   uniforms,
                                                                   material.textureIndices0.z,
                                                                   normalCtx.uv + normalCtx.dUVdx,
                                                                   float4(0.5f, 0.5f, 1.0f, 1.0f),
                                                                   normalLod).xyz;
                        float3 nDy = sample_material_texture_level(materialTextures,
                                                                   materialSamplers,
                                                                   materialTextureInfos,
                                                                   uniforms,
                                                                   material.textureIndices0.z,
                                                                   normalCtx.uv + normalCtx.dUVdy,
                                                                   float4(0.5f, 0.5f, 1.0f, 1.0f),
                                                                   normalLod).xyz;
                        float tmpLenDx = 1.0f;
                        float tmpLenDy = 1.0f;
                        nDx = decode_normal_map(nDx, normalScale, flipNormalGreen, tmpLenDx);
                        nDy = decode_normal_map(nDy, normalScale, flipNormalGreen, tmpLenDy);
                        float varianceX = max(1.0f - dot(normalSampleTs, nDx), 0.0f);
                        float varianceY = max(1.0f - dot(normalSampleTs, nDy), 0.0f);
                        float normalVariance = max(varianceX, varianceY);
                        tok += 0.35f * normalVariance;
                    }
                }
                roughness = clamp(sqrt(roughness * roughness + tok), 0.0f, 1.0f);
            }

            material.baseColorRoughness = float4(baseColor, roughness);
            material.pbrParams.x = metallic;
            material.emission = float4(emissive, 0.0f);
            rec.shadingNormal = shadingNormal;
            if (debugContext) {
                record_debug_event(*debugContext,
                                   kDebugEventTbnBasis0,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   debugVtxNormalRaw,
                                   float4(debugNormalUv.x,
                                          debugNormalUv.y,
                                          debugTangentW,
                                          rec.t),
                                   debugVtxNormal);
                record_debug_event(*debugContext,
                                   kDebugEventTbnBasis1,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   debugTangentRaw,
                                   float4(debugTangent, 0.0f),
                                   debugBitangent);
                record_debug_event(*debugContext,
                                   kDebugEventTbnBasis2,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   debugTexelRaw,
                                   float4(debugTexelDecoded, 0.0f),
                                   debugSnTangentSpace);
                record_debug_event(*debugContext,
                                   kDebugEventTbnBasis3,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   float3(0.0f),
                                   float4(0.0f),
                                   debugSnWorld);
                record_debug_event(*debugContext,
                                   kDebugEventTbnBasis4,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   debugBaryWeights,
                                   float4(0.0f),
                                   debugTriIndices);
                record_debug_event(*debugContext,
                                   kDebugEventBsdfMaterial0,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   baseColor,
                                   float4(debugNormalUv.x,
                                          debugNormalUv.y,
                                          metallic,
                                          roughness),
                                   float3(normalScale,
                                          transmission,
                                          0.0f));
                record_debug_event(*debugContext,
                                   kDebugEventBsdfMaterial1,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   emissive,
                                   float4(alpha,
                                          alphaCutoff,
                                          alphaMode,
                                          material.baseColorRoughness.w),
                                   material.baseColorRoughness.xyz);
                record_debug_event(*debugContext,
                                   kDebugEventBsdfMaterial2,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   material.pbrExtras.xyz,
                                   material.pbrParams,
                                   float3(material.pbrExtras.w, 0.0f, 0.0f));
                record_debug_event(*debugContext,
                                   kDebugEventBsdfGradients,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   float3(normalCtx.dUVdx.x,
                                          normalCtx.dUVdx.y,
                                          normalCtx.dUVdy.x),
                                   float4(normalCtx.hasIgehyGradients ? 1.0f : 0.0f,
                                          normalCtx.dUVdy.y,
                                          normalCtx.uvPerWorld,
                                          normalLod),
                                   float3(0.0f));
            }
        }

        if (uniforms.debugViewMode != kDebugViewNone) {
            const float3 debugColor =
                restir_debug_view_color(uniforms,
                                        rec,
                                        material,
                                        pixelCoord,
                                        depth,
                                        pathGuidingStates,
                                        restirPtReservoirs,
                                        radianceCacheStates,
                                        debugBaseColor,
                                        debugMetallic,
                                        debugRoughness,
                                        debugAO);
            radiance = debugColor;
            break;
        }

        // Capture first hit AOVs (albedo and normal) for denoising
        if (isFirstHit) {
            isFirstHit = false;
            if (outFirstHitAlbedo != nullptr) {
                *outFirstHitAlbedo = material_closure_aov_albedo(material);
            }
            if (outFirstHitNormal != nullptr) {
                // Store the world-space normal of first hit
                *outFirstHitNormal = shadingNormal;
            }
            if (outFirstHitPosition != nullptr) {
                *outFirstHitPosition = float4(rec.point, 1.0f);
            }
            if (outFirstHitMaterial != nullptr) {
                *outFirstHitMaterial = material_closure_aov_features(material);
            }
        }

        Reservoir risAuditReservoir = make_empty_reservoir();
        float3 risAuditBsdfValue = float3(0.0f);
        float3 risAuditContribution = float3(0.0f);
        float risAuditDistance = 0.0f;
        float risAuditNDotL = 0.0f;
        float risAuditBsdfPdf = 0.0f;
        bool risAuditVisible = false;
        bool risAuditActive = false;
        bool spatialReuseAttempted = false;
        uint spatialNeighborTarget = 0u;
        uint spatialNeighborsConsidered = 0u;
        uint spatialNeighborsAccepted = 0u;
        uint spatialRejectedDepth = 0u;
        uint spatialRejectedNormal = 0u;
        uint spatialRejectedInvalid = 0u;
        int2 spatialWinnerOffset = int2(0, 0);
        float spatialLastMisWeight = 0.0f;
        float spatialLastMergeWeight = 0.0f;
        bool temporalReuseAttempted = false;
        bool temporalPreviousAvailable = false;
        bool temporalReuseAccepted = false;
        uint temporalRejectedDepth = 0u;
        uint temporalRejectedNormal = 0u;
        uint temporalRejectedInvalid = 0u;
        uint temporalPreviousPrimitiveIndex = kInvalidIndex;
        float temporalLastMisWeight = 0.0f;
        float temporalLastMergeWeight = 0.0f;
        bool worldReuseAttempted = false;
        bool worldReuseAccepted = false;
        int3 worldCell = int3(0);
        uint worldCellHash = 0u;
        uint worldCandidatesConsidered = 0u;
        uint worldCandidatesAccepted = 0u;
        uint worldRejectedDepth = 0u;
        uint worldRejectedNormal = 0u;
        uint worldRejectedInvalid = 0u;
        uint worldRejectedCell = 0u;
        uint worldCandidatePrimitiveIndex = kInvalidIndex;
        float worldLastMisWeight = 0.0f;
        float worldLastMergeWeight = 0.0f;
        bool cacheReuseAttempted = false;
        bool cacheStateAvailable = false;
        bool cacheReuseAccepted = false;
        bool cacheFallbackUsed = false;
        int3 cacheCell = int3(0);
        uint cacheCellHash = 0u;
        uint cacheCandidatesConsidered = 0u;
        uint cacheEntriesAvailable = 0u;
        uint cacheCandidatesAccepted = 0u;
        uint cacheRejectedDepth = 0u;
        uint cacheRejectedNormal = 0u;
        uint cacheRejectedInvalid = 0u;
        uint cacheRejectedCell = 0u;
        uint cacheCandidatePrimitiveIndex = kInvalidIndex;
        uint cacheSourceFrameIndex = 0u;
        float cacheLastMisWeight = 0.0f;
        float cacheLastMergeWeight = 0.0f;

        if (!specularOnly &&
            type == 7u &&
            any(material.emission.xyz != float3(0.0f)) &&
            (rec.frontFace != 0u || rec.twoSided != 0u)) {
            float3 visibleEmission = camera_visible_emission(material, depth);
            radiance += clamp_firefly_contribution(throughput, visibleEmission, clampParams);
            if (debugContext) {
                record_debug_event(*debugContext,
                                   kDebugEventVisibleEmitter,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   throughput,
                                   float4(rec.t, 0.0f, 0.0f, 0.0f),
                                   visibleEmission);
            }
        }

        if (type == 3u) {  // DiffuseLight
            if (specularOnly) {
                break;
            }
            float3 emission = material.emission.xyz;
            if (material.emission.w > 0.0f &&
                environmentTexture.get_width() > 0 &&
                environmentTexture.get_height() > 0 &&
                rec.frontFace != 0u) {
                float3 sampleDir = -shadingNormal;
                float3 envColor = environment_color(environmentTexture,
                                                    sampleDir,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity,
                                                    uniforms);
                emission *= envColor;
            }
            if (any(emission != float3(0.0f)) &&
                (rec.frontFace != 0u || rec.twoSided != 0u)) {
                float misWeight = 1.0f;
                bool useSpecularMis =
                    use_visible_emitter_mis(depth, lastScatterWasDelta, uniforms);
                if (useSpecularMis && rectLightCount > 0u) {
                    float lightPdf = rect_light_pdf_for_hit(uniforms,
                                                            rectangles,
                                                            materials,
                                                            rectLightCount,
                                                            rec,
                                                            ray.origin);
                    float denom = lastBsdfPdf + lightPdf;
                    if (denom > 0.0f) {
                        misWeight = clamp(lastBsdfPdf / denom,
                                          kMisWeightClampMin,
                                          kMisWeightClampMax);
                    }
                }
                float3 contribution = emission * misWeight;
                radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                if (debugContext) {
                    record_debug_event(*debugContext,
                                       kDebugEventVisibleEmitter,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       throughput,
                                       float4(misWeight, lastBsdfPdf, rec.t, 0.0f),
                                       contribution);
                }
            }
            break;
        }

        if (!surfaceIsDelta &&
            uniforms.directLightMode != kDirectLightModeLegacyRect &&
            uniforms.emissivePrimitiveCount > 0u) {
            if (uniforms.directLightMode == kDirectLightModeBaselineEmissive) {
                RISSamplePayload lightSample;
                if (sample_direct_light_baseline(uniforms,
                                                 emissivePrimitives,
                                                 rec,
                                                 state,
                                                 lightSample)) {
                    const bool lightSampleDirectional = ris_payload_is_directional(lightSample);
                    float distSq = ris_payload_distance_sq(rec, lightSample);
                    float3 omegaL = ris_payload_omega_l(rec, lightSample);
                    float cosThetaSurface = saturate(dot(shadingNormal, omegaL));
                    float cosThetaLight = lightSampleDirectional ? 1.0f : saturate(dot(lightSample.normal, -omegaL));
                    if (lightSample.pdf > 0.0f &&
                        distSq > 0.0f &&
                        cosThetaSurface > 0.0f &&
                        cosThetaLight > 0.0f) {
                        Ray shadowRay;
                        shadowRay.origin = offset_ray_origin(rec, omegaL);
                        shadowRay.direction = omegaL;
                        HitRecord shadowRec = make_empty_hit_record();
                        float shadowMax = lightSampleDirectional ? 1.0e20f : max(length(lightSample.position - rec.point) - kEpsilon, kEpsilon);
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
                        if (!occluded) {
                            BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                                    rec.point,
                                                                    shadingNormal,
                                                                    wo,
                                                                    omegaL,
                                                                    clampParams,
                                                                    uniforms.sssMode,
                                                                    diffuseOcclusion,
                                                                    specularOnly);
                            if (!bsdfEval.isDelta && !bsdfEval.isBssrdf) {
                                float3 contribution = bsdfEval.value * lightSample.emission;
                                contribution *= cosThetaSurface / max(lightSample.pdf, 1.0e-6f);
                                if (all(isfinite(contribution))) {
                                    radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                                }
                            }
                        }
                    }
                }
            } else if (uniforms.directLightMode == kDirectLightModeRis ||
                       uniforms.directLightMode == kDirectLightModeRisSpatialReuse ||
                       uniforms.directLightMode == kDirectLightModeRisTemporalReuse ||
                       uniforms.directLightMode == kDirectLightModeRisWorldReuse ||
                       uniforms.directLightMode == kDirectLightModeRisRegirCache ||
                       uniforms.directLightMode == kDirectLightModeRestirDi ||
                       uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) {
                Reservoir reservoir = make_empty_reservoir();
                uint risTargetM = max(uniforms.risCandidateCount, 1u);
                if ((uniforms.directLightMode == kDirectLightModeRisRegirCache ||
                     uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
                    uniforms.frameIndex > 0u) {
                    risTargetM *= 2u;
                }
                build_ris_reservoir_for_hit(uniforms,
                                            emissivePrimitives,
                                            rec,
                                            material,
                                            shadingNormal,
                                            wo,
                                            clampParams,
                                            diffuseOcclusion,
                                            specularOnly,
                                            risTargetM,
                                            state,
                                            reservoir);

                if (uniforms.directLightMode == kDirectLightModeRisSpatialReuse &&
                    reservoir.valid &&
                    depth == 0u &&
                    rec.t > 0.0f &&
                    uniforms.width > 0u &&
                    uniforms.height > 0u) {
                    spatialReuseAttempted = true;
                    spatialNeighborTarget = min(max(uniforms.spatialReuseNeighborCount, 1u), 4u);
                    constexpr int2 offsets[4] = {
                        int2(-1, 0),
                        int2(1, 0),
                        int2(0, -1),
                        int2(0, 1)
                    };
                    for (uint n = 0u; n < spatialNeighborTarget; ++n) {
                        int2 offset = offsets[n];
                        int2 neighborCoordI = int2(pixelCoord) + offset;
                        if (neighborCoordI.x < 0 ||
                            neighborCoordI.y < 0 ||
                            neighborCoordI.x >= int(uniforms.width) ||
                            neighborCoordI.y >= int(uniforms.height)) {
                            spatialRejectedInvalid += 1u;
                            continue;
                        }

                        uint2 neighborCoord = uint2(neighborCoordI);
                        Ray neighborRay = make_center_primary_ray_for_pixel(uniforms, neighborCoord);
                        HitRecord neighborRec = make_empty_hit_record();
                        bool neighborHit = trace_scene_software(uniforms,
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
                                                                 neighborRay,
                                                                 kEpsilon,
                                                                 kInfinity,
                                                                 /*anyHitOnly=*/false,
                                                                 /*includeTriangles=*/true,
                                                                 neighborRec);
                        if (!neighborHit || neighborRec.materialIndex >= uniforms.materialCount) {
                            spatialRejectedInvalid += 1u;
                            continue;
                        }

                        float depthRel = fabs(rec.t - neighborRec.t) / max(rec.t, kEpsilon);
                        if (!(depthRel <= 0.1f) || !isfinite(depthRel)) {
                            spatialRejectedDepth += 1u;
                            continue;
                        }
                        float normalDot = dot(safe_normalize(shadingNormal),
                                              safe_normalize(neighborRec.shadingNormal));
                        if (!(normalDot >= 0.9f) || !isfinite(normalDot)) {
                            spatialRejectedNormal += 1u;
                            continue;
                        }

                        spatialNeighborsConsidered += 1u;
                        MaterialData neighborMaterial = materials[neighborRec.materialIndex];
                        float3 neighborWo = safe_normalize(-neighborRay.direction);
                        uint neighborState =
                            pcg_hash(state ^ (uint(neighborCoord.x) * 1664525u) ^
                                     (uint(neighborCoord.y) * 1013904223u) ^
                                     (n + 1u) * 747796405u);
                        Reservoir neighborReservoir = make_empty_reservoir();
                        if (!build_ris_reservoir_for_hit(uniforms,
                                                         emissivePrimitives,
                                                         neighborRec,
                                                         neighborMaterial,
                                                         neighborRec.shadingNormal,
                                                         neighborWo,
                                                         clampParams,
                                                         diffuseOcclusion,
                                                         specularOnly,
                                                         risTargetM,
                                                         neighborState,
                                                         neighborReservoir)) {
                            spatialRejectedInvalid += 1u;
                            continue;
                        }

                        float phatCurrent = reservoir_winner_phat_for_hit(uniforms,
                                                                          rec,
                                                                          material,
                                                                          shadingNormal,
                                                                          wo,
                                                                          clampParams,
                                                                          diffuseOcclusion,
                                                                          specularOnly,
                                                                          neighborReservoir.winner);
                        float phatNeighbor = reservoir_winner_phat_for_hit(uniforms,
                                                                           neighborRec,
                                                                           neighborMaterial,
                                                                           neighborRec.shadingNormal,
                                                                           neighborWo,
                                                                           clampParams,
                                                                           diffuseOcclusion,
                                                                           specularOnly,
                                                                           neighborReservoir.winner);
                        float denom = phatCurrent * float(max(reservoir.M, 1u)) +
                                      phatNeighbor * float(max(neighborReservoir.M, 1u));
                        if (!(denom > 0.0f) || !isfinite(denom) || !(phatCurrent > 0.0f)) {
                            spatialRejectedInvalid += 1u;
                            continue;
                        }
                        float mis = phatCurrent / denom;
                        float mergeWeight = mis * neighborReservoir.wSum;
                        if (!(mergeWeight > 0.0f) || !isfinite(mergeWeight)) {
                            spatialRejectedInvalid += 1u;
                            continue;
                        }
                        uint previousWinner = reservoir.winner.primitiveIndex;
                        bool changed = update_reservoir(reservoir,
                                                        neighborReservoir.winner,
                                                        mergeWeight,
                                                        rand_uniform(state));
                        spatialNeighborsAccepted += 1u;
                        spatialLastMisWeight = mis;
                        spatialLastMergeWeight = mergeWeight;
                        if (changed || reservoir.winner.primitiveIndex != previousWinner) {
                            spatialWinnerOffset = offset;
                        }
                    }
                }

                if ((uniforms.directLightMode == kDirectLightModeRisTemporalReuse ||
                     uniforms.directLightMode == kDirectLightModeRestirDi ||
                     uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
                    depth == 0u) {
                    temporalReuseAttempted = true;
                    if (reservoir.valid &&
                        rec.t > 0.0f &&
                        uniforms.frameIndex > 0u) {
                        temporalPreviousAvailable = true;
                        uint temporalState = pcg_hash(uniforms.fixedRngSeed ^
                                                      (uint(pixelCoord.x) * 1664525u) ^
                                                      (uint(pixelCoord.y) * 1013904223u) ^
                                                      ((uniforms.frameIndex - 1u) * 9781u) ^
                                                      0x9E3779B9u);
                        Reservoir previousReservoir = make_empty_reservoir();
                        if (!build_ris_reservoir_for_hit(uniforms,
                                                         emissivePrimitives,
                                                         rec,
                                                         material,
                                                         shadingNormal,
                                                         wo,
                                                         clampParams,
                                                         diffuseOcclusion,
                                                         specularOnly,
                                                         risTargetM,
                                                         temporalState,
                                                         previousReservoir)) {
                            temporalRejectedInvalid += 1u;
                        } else {
                            temporalPreviousPrimitiveIndex = previousReservoir.winner.primitiveIndex;
                            float phatCurrent = reservoir_winner_phat_for_hit(uniforms,
                                                                              rec,
                                                                              material,
                                                                              shadingNormal,
                                                                              wo,
                                                                              clampParams,
                                                                              diffuseOcclusion,
                                                                              specularOnly,
                                                                              previousReservoir.winner);
                            if (!(phatCurrent > 0.0f) || !isfinite(phatCurrent)) {
                                temporalRejectedInvalid += 1u;
                            } else {
                                float mergeWeight = previousReservoir.wSum;
                                if (!(mergeWeight > 0.0f) || !isfinite(mergeWeight)) {
                                    temporalRejectedInvalid += 1u;
                                } else {
                                    merge_reservoir_winner(reservoir,
                                                           previousReservoir.winner,
                                                           mergeWeight,
                                                           previousReservoir.M,
                                                           rand_uniform(state));
                                    temporalReuseAccepted = true;
                                    temporalLastMisWeight = 1.0f;
                                    temporalLastMergeWeight = mergeWeight;
                                }
                            }
                        }
                    } else {
                        temporalRejectedInvalid += 1u;
                    }
                }

                if (uniforms.directLightMode == kDirectLightModeRisWorldReuse &&
                    depth == 0u) {
                    worldReuseAttempted = true;
                    if (reservoir.valid &&
                        rec.t > 0.0f &&
                        uniforms.width > 0u &&
                        uniforms.height > 0u) {
                        worldCell = world_reuse_cell(rec.point, uniforms.worldReuseCellSize);
                        worldCellHash = world_reuse_cell_hash(worldCell);
                        constexpr int2 offsets[4] = {
                            int2(-2, 0),
                            int2(2, 0),
                            int2(0, -2),
                            int2(0, 2)
                        };
                        for (uint n = 0u; n < 4u; ++n) {
                            int2 offset = offsets[n];
                            int2 candidateCoordI = int2(pixelCoord) + offset;
                            if (candidateCoordI.x < 0 ||
                                candidateCoordI.y < 0 ||
                                candidateCoordI.x >= int(uniforms.width) ||
                                candidateCoordI.y >= int(uniforms.height)) {
                                worldRejectedInvalid += 1u;
                                continue;
                            }

                            uint2 candidateCoord = uint2(candidateCoordI);
                            Ray candidateRay = make_center_primary_ray_for_pixel(uniforms, candidateCoord);
                            HitRecord candidateRec = make_empty_hit_record();
                            bool candidateHit = trace_scene_software(uniforms,
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
                                                                      candidateRay,
                                                                      kEpsilon,
                                                                      kInfinity,
                                                                      /*anyHitOnly=*/false,
                                                                      /*includeTriangles=*/true,
                                                                      candidateRec);
                            if (!candidateHit || candidateRec.materialIndex >= uniforms.materialCount) {
                                worldRejectedInvalid += 1u;
                                continue;
                            }

                            int3 candidateCell =
                                world_reuse_cell(candidateRec.point, uniforms.worldReuseCellSize);
                            if (!world_reuse_cell_compatible(worldCell, candidateCell)) {
                                worldRejectedCell += 1u;
                                continue;
                            }

                            float depthRel = fabs(rec.t - candidateRec.t) / max(rec.t, kEpsilon);
                            if (!(depthRel <= 0.2f) || !isfinite(depthRel)) {
                                worldRejectedDepth += 1u;
                                continue;
                            }
                            float normalDot = dot(safe_normalize(shadingNormal),
                                                  safe_normalize(candidateRec.shadingNormal));
                            if (!(normalDot >= 0.85f) || !isfinite(normalDot)) {
                                worldRejectedNormal += 1u;
                                continue;
                            }

                            worldCandidatesConsidered += 1u;
                            MaterialData candidateMaterial = materials[candidateRec.materialIndex];
                            float3 candidateWo = safe_normalize(-candidateRay.direction);
                            uint candidateState =
                                pcg_hash(state ^
                                         (uint(candidateCell.x) * 1664525u) ^
                                         (uint(candidateCell.y) * 1013904223u) ^
                                         (uint(candidateCell.z) * 747796405u) ^
                                         ((n + 1u) * 2891336453u));
                            Reservoir candidateReservoir = make_empty_reservoir();
                            if (!build_ris_reservoir_for_hit(uniforms,
                                                             emissivePrimitives,
                                                             candidateRec,
                                                             candidateMaterial,
                                                             candidateRec.shadingNormal,
                                                             candidateWo,
                                                             clampParams,
                                                             diffuseOcclusion,
                                                             specularOnly,
                                                             risTargetM,
                                                             candidateState,
                                                             candidateReservoir)) {
                                worldRejectedInvalid += 1u;
                                continue;
                            }

                            float phatCurrent = reservoir_winner_phat_for_hit(uniforms,
                                                                              rec,
                                                                              material,
                                                                              shadingNormal,
                                                                              wo,
                                                                              clampParams,
                                                                              diffuseOcclusion,
                                                                              specularOnly,
                                                                              candidateReservoir.winner);
                            float phatCandidate = reservoir_winner_phat_for_hit(uniforms,
                                                                                candidateRec,
                                                                                candidateMaterial,
                                                                                candidateRec.shadingNormal,
                                                                                candidateWo,
                                                                                clampParams,
                                                                                diffuseOcclusion,
                                                                                specularOnly,
                                                                                candidateReservoir.winner);
                            float denom = phatCurrent * float(max(reservoir.M, 1u)) +
                                          phatCandidate * float(max(candidateReservoir.M, 1u));
                            if (!(denom > 0.0f) || !isfinite(denom) || !(phatCurrent > 0.0f)) {
                                worldRejectedInvalid += 1u;
                                continue;
                            }
                            float mis = phatCurrent / denom;
                            float mergeWeight = mis * candidateReservoir.wSum;
                            if (!(mergeWeight > 0.0f) || !isfinite(mergeWeight)) {
                                worldRejectedInvalid += 1u;
                                continue;
                            }

                            update_reservoir(reservoir,
                                             candidateReservoir.winner,
                                             mergeWeight,
                                             rand_uniform(state));
                            worldCandidatesAccepted += 1u;
                            worldReuseAccepted = true;
                            worldCandidatePrimitiveIndex = candidateReservoir.winner.primitiveIndex;
                            worldLastMisWeight = mis;
                            worldLastMergeWeight = mergeWeight;
                        }
                    } else {
                        worldRejectedInvalid += 1u;
                    }
                }

                if ((uniforms.directLightMode == kDirectLightModeRisRegirCache ||
                     uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
                    depth == 0u) {
                    cacheReuseAttempted = true;
                    cacheFallbackUsed = true;
                    cacheCell = world_reuse_cell(rec.point, uniforms.worldReuseCellSize);
                    cacheCellHash = world_reuse_cell_hash(cacheCell);
                    if (uniforms.frameIndex > 1u) {
                        cacheSourceFrameIndex = uniforms.frameIndex - 2u;
                    } else if (uniforms.frameIndex > 0u) {
                        cacheSourceFrameIndex = uniforms.frameIndex - 1u;
                    }
                    if (reservoir.valid &&
                        rec.t > 0.0f &&
                        uniforms.frameIndex > 0u) {
                        cacheStateAvailable = true;
                        cacheFallbackUsed = false;
                        cacheCandidatesConsidered = 0u;
                        cacheEntriesAvailable = 0u;
                        Reservoir retainedReservoir0 = make_empty_reservoir();
                        Reservoir retainedReservoir1 = make_empty_reservoir();
                        uint retainedSourceFrame0 = 0u;
                        uint retainedSourceFrame1 = 0u;
                        uint retainedPrimitive0 = 0xFFFFFFFFu;
                        uint retainedPrimitive1 = 0xFFFFFFFFu;
                        float retainedMis0 = 0.0f;
                        float retainedMis1 = 0.0f;
                        float retainedMerge0 = 0.0f;
                        float retainedMerge1 = 0.0f;
                        float retainedScore0 = -1.0f;
                        float retainedScore1 = -1.0f;
                        uint retainedCount = 0u;
                        for (uint cacheProbe = 0u; cacheProbe < 4u; ++cacheProbe) {
                            if (cacheProbe > cacheSourceFrameIndex) {
                                continue;
                            }
                            uint candidateSourceFrameIndex = cacheSourceFrameIndex - cacheProbe;
                            cacheCandidatesConsidered += 1u;
                            uint cacheState =
                                pcg_hash(uniforms.fixedRngSeed ^
                                         (uint(cacheCell.x) * 1664525u) ^
                                         (uint(cacheCell.y) * 1013904223u) ^
                                         (uint(cacheCell.z) * 747796405u) ^
                                         (candidateSourceFrameIndex * 9781u) ^
                                         (cacheProbe * 0x9E3779B9u) ^
                                         0xD1B54A35u);
                            Reservoir cachedReservoir = make_empty_reservoir();
                            if (!build_ris_reservoir_for_hit(uniforms,
                                                             emissivePrimitives,
                                                             rec,
                                                             material,
                                                             shadingNormal,
                                                             wo,
                                                             clampParams,
                                                             diffuseOcclusion,
                                                             specularOnly,
                                                             risTargetM,
                                                             cacheState,
                                                             cachedReservoir)) {
                                cacheRejectedInvalid += 1u;
                                continue;
                            }
                            if (!cachedReservoir.valid) {
                                cacheRejectedInvalid += 1u;
                                continue;
                            }
                            int3 cachedCell = world_reuse_cell(rec.point, uniforms.worldReuseCellSize);
                            if (!world_reuse_cell_compatible(cacheCell, cachedCell)) {
                                cacheRejectedCell += 1u;
                                continue;
                            }
                            float depthRel = 0.0f;
                            if (!(depthRel <= 0.2f) || !isfinite(depthRel)) {
                                cacheRejectedDepth += 1u;
                                continue;
                            }
                            float normalDot = dot(safe_normalize(shadingNormal),
                                                  safe_normalize(shadingNormal));
                            if (!(normalDot >= 0.85f) || !isfinite(normalDot)) {
                                cacheRejectedNormal += 1u;
                                continue;
                            }
                            float phatCurrent = reservoir_winner_phat_for_hit(uniforms,
                                                                              rec,
                                                                              material,
                                                                              shadingNormal,
                                                                              wo,
                                                                              clampParams,
                                                                              diffuseOcclusion,
                                                                              specularOnly,
                                                                              cachedReservoir.winner);
                            float phatCached = reservoir_winner_phat_for_hit(uniforms,
                                                                             rec,
                                                                             material,
                                                                             shadingNormal,
                                                                             wo,
                                                                             clampParams,
                                                                             diffuseOcclusion,
                                                                             specularOnly,
                                                                             cachedReservoir.winner);
                            float denom = phatCurrent * float(max(reservoir.M, 1u)) +
                                          phatCached * float(max(cachedReservoir.M, 1u));
                            if (!(denom > 0.0f) || !isfinite(denom) || !(phatCurrent > 0.0f)) {
                                cacheRejectedInvalid += 1u;
                            } else {
                                float mis = phatCurrent / denom;
                                float mergeWeight = mis * cachedReservoir.wSum;
                                if (!(mergeWeight > 0.0f) || !isfinite(mergeWeight)) {
                                    cacheRejectedInvalid += 1u;
                                } else {
                                    cacheEntriesAvailable += 1u;
                                    if (mergeWeight > retainedScore0) {
                                        if (retainedCount > 0u) {
                                            retainedReservoir1 = retainedReservoir0;
                                            retainedSourceFrame1 = retainedSourceFrame0;
                                            retainedPrimitive1 = retainedPrimitive0;
                                            retainedMis1 = retainedMis0;
                                            retainedMerge1 = retainedMerge0;
                                            retainedScore1 = retainedScore0;
                                        }
                                        retainedReservoir0 = cachedReservoir;
                                        retainedSourceFrame0 = candidateSourceFrameIndex;
                                        retainedPrimitive0 = cachedReservoir.winner.primitiveIndex;
                                        retainedMis0 = mis;
                                        retainedMerge0 = mergeWeight;
                                        retainedScore0 = mergeWeight;
                                        retainedCount = min(retainedCount + 1u, 2u);
                                    } else if (mergeWeight > retainedScore1) {
                                        retainedReservoir1 = cachedReservoir;
                                        retainedSourceFrame1 = candidateSourceFrameIndex;
                                        retainedPrimitive1 = cachedReservoir.winner.primitiveIndex;
                                        retainedMis1 = mis;
                                        retainedMerge1 = mergeWeight;
                                        retainedScore1 = mergeWeight;
                                        retainedCount = min(retainedCount + 1u, 2u);
                                    }
                                }
                            }
                        }
                        if (retainedCount > 0u && retainedReservoir0.valid) {
                            update_reservoir(reservoir,
                                             retainedReservoir0.winner,
                                             retainedMerge0,
                                             rand_uniform(state));
                            cacheCandidatesAccepted += 1u;
                            cacheReuseAccepted = true;
                            cacheSourceFrameIndex = retainedSourceFrame0;
                            cacheCandidatePrimitiveIndex = retainedPrimitive0;
                            cacheLastMisWeight = retainedMis0;
                            cacheLastMergeWeight = retainedMerge0;
                        }
                        if (retainedCount > 1u && retainedReservoir1.valid) {
                            update_reservoir(reservoir,
                                             retainedReservoir1.winner,
                                             retainedMerge1,
                                             rand_uniform(state));
                            cacheCandidatesAccepted += 1u;
                            cacheReuseAccepted = true;
                            cacheSourceFrameIndex = retainedSourceFrame1;
                            cacheCandidatePrimitiveIndex = retainedPrimitive1;
                            cacheLastMisWeight = retainedMis1;
                            cacheLastMergeWeight = retainedMerge1;
                        }
                    } else {
                        cacheRejectedInvalid += 1u;
                    }
                }

                if (reservoir.valid && reservoir.M > 0u && reservoir.wSum > 0.0f) {
                    RISSamplePayload winner = reservoir.winner;
                    float distSq = ris_payload_distance_sq(rec, winner);
                    if (distSq > 0.0f && isfinite(distSq)) {
                        const bool winnerDirectional = ris_payload_is_directional(winner);
                        float3 omegaL = ris_payload_omega_l(rec, winner);
                        float cosThetaSurface = saturate(dot(shadingNormal, omegaL));
                        float cosThetaLight = winnerDirectional ? 1.0f : saturate(dot(winner.normal, -omegaL));
                        BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                                rec.point,
                                                                shadingNormal,
                                                                wo,
                                                                omegaL,
                                                                clampParams,
                                                                uniforms.sssMode,
                                                                diffuseOcclusion,
                                                                specularOnly);
                        float phatW = p_hat(winner.emission,
                                            bsdfEval.value,
                                            cosThetaSurface,
                                            cosThetaLight,
                                            distSq);
                        reservoir.W = (phatW > 0.0f)
                                    ? ((reservoir.wSum / float(reservoir.M)) / phatW)
                                    : 0.0f;

                        Ray shadowRay;
                        shadowRay.origin = offset_ray_origin(rec, omegaL);
                        shadowRay.direction = omegaL;
                        HitRecord shadowRec = make_empty_hit_record();
                        float shadowMax = winnerDirectional ? 1.0e20f : max(length(winner.position - rec.point) - kEpsilon, kEpsilon);
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
                        if (!winnerDirectional &&
                            occluded &&
                            shadowRec.primitiveType == kPrimitiveTypeTriangle &&
                            shadowRec.primitiveIndex == winner.primitiveIndex) {
                            occluded = false;
                        }

                        float3 contribution = float3(0.0f);
                        if (!occluded &&
                            !bsdfEval.isDelta &&
                            !bsdfEval.isBssrdf &&
                            reservoir.W > 0.0f &&
                            cosThetaSurface > 0.0f) {
                            contribution = bsdfEval.value * winner.emission;
                            contribution *= cosThetaSurface * reservoir.W;
                            if (all(isfinite(contribution))) {
                                radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                            } else {
                                contribution = float3(0.0f);
                            }
                        }

                        risAuditReservoir = reservoir;
                        risAuditBsdfValue = bsdfEval.value;
                        risAuditContribution = contribution;
                        risAuditDistance = sqrt(distSq);
                        risAuditNDotL = cosThetaSurface;
                        risAuditBsdfPdf = bsdfEval.pdf;
                        risAuditVisible = !occluded;
                        risAuditActive = true;
                    }
                }
            }
        } else if (!surfaceIsDelta && rectLightCount > 0u) {
            RectLightSample lightSample;
            if (sample_rect_light(uniforms,
                                  rectangles,
                                  materials,
                                  environmentTexture,
                                  rec,
                                  state,
                                  rectLightCount,
                                  lightSample)) {
                float nDotL = max(dot(shadingNormal, lightSample.direction), 0.0f);
                if (lightSample.pdf > 0.0f && nDotL > 0.0f) {
                    if (debugContext) {
                        record_debug_event(*debugContext,
                                           kDebugEventRectSample,
                                           depth,
                                           mediumDepth,
                                           mediumDepth,
                                           /*mediumEvent=*/0,
                                           rec.frontFace,
                                           rec.materialIndex,
                                           /*scatterIsDelta=*/false,
                                           throughput,
                                           float4(lightSample.pdf,
                                                  lightSample.distance,
                                                  nDotL,
                                                  float(lightSample.rectIndex)),
                                           lightSample.emission);
                    }
                    Ray shadowRay;
                    shadowRay.origin = offset_ray_origin(rec, lightSample.direction);
                    shadowRay.direction = lightSample.direction;
                    HitRecord shadowRec;
                    float shadowMax = max(lightSample.distance - kEpsilon, kEpsilon);
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
                    if (debugContext) {
                        record_debug_event(*debugContext,
                                           kDebugEventRectShadow,
                                           depth,
                                           mediumDepth,
                                           mediumDepth,
                                           /*mediumEvent=*/0,
                                           rec.frontFace,
                                           rec.materialIndex,
                                           /*scatterIsDelta=*/false,
                                           throughput,
                                           float4(occluded ? 1.0f : 0.0f,
                                                  shadowMax,
                                                  shadowRec.t,
                                                  float(lightSample.rectIndex)),
                                           float3(float(shadowRec.materialIndex),
                                                  float(shadowRec.meshIndex),
                                                  float(shadowRec.primitiveIndex)));
                    }
                    if (!occluded) {
                        BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                                rec.point,
                                                                shadingNormal,
                                                                wo,
                                                                lightSample.direction,
                                                                clampParams,
                                                                uniforms.sssMode,
                                                                diffuseOcclusion,
                                                                specularOnly);
                        float3 bsdfValue = bsdfEval.value;
                        float maxComponent = max(max(bsdfValue.x, bsdfValue.y), bsdfValue.z);
                        if (debugContext) {
                            float flags = (bsdfEval.isDelta ? 1.0f : 0.0f) +
                                          (bsdfEval.isBssrdf ? 2.0f : 0.0f);
                            record_debug_event(*debugContext,
                                               kDebugEventRectEval,
                                               depth,
                                               mediumDepth,
                                               mediumDepth,
                                               /*mediumEvent=*/0,
                                               rec.frontFace,
                                               rec.materialIndex,
                                               /*scatterIsDelta=*/false,
                                               throughput,
                                               float4(lightSample.pdf,
                                                      bsdfEval.pdf,
                                                      maxComponent,
                                                      flags),
                                               bsdfValue);
                        }
                        if (!bsdfEval.isDelta && !bsdfEval.isBssrdf) {
                            if (maxComponent > 0.0f && lightSample.pdf > 0.0f) {
                                float bsdfPdf = bsdfEval.pdf;
                                float weight = 1.0f;
                                if (bsdfPdf > 0.0f) {
                                    float denom = lightSample.pdf + bsdfPdf;
                                    if (denom > 0.0f) {
                                        weight = clamp(lightSample.pdf / denom,
                                                       kMisWeightClampMin,
                                                       kMisWeightClampMax);
                                    }
                                }
                                float3 contribution = lightSample.emission * bsdfValue * nDotL;
                                contribution *= weight / lightSample.pdf;
                                if (all(isfinite(contribution))) {
                                    radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                                    if (debugContext) {
                                        record_debug_event(*debugContext,
                                                           kDebugEventRectNee,
                                                           depth,
                                                           mediumDepth,
                                                           mediumDepth,
                                                           /*mediumEvent=*/0,
                                                           rec.frontFace,
                                                           rec.materialIndex,
                                                           /*scatterIsDelta=*/false,
                                                           throughput,
                                                           float4(lightSample.pdf,
                                                                  bsdfPdf,
                                                                  weight,
                                                                  nDotL),
                                                           contribution);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (debugContext &&
            depth == 0u &&
            uniforms.debugDirectLightAudit != 0u &&
            (rectLightCount > 0u || uniforms.emissivePrimitiveCount > 0u)) {
            float3 auditBaseColor = debugBaseColor;
            float2 auditBaseColorUv = debugBaseColorUv;
            float auditBaseColorLod = debugBaseColorLod;
            record_debug_event(*debugContext,
                               kDebugEventDirectLightAuditMeta,
                               depth,
                               mediumDepth,
                               mediumDepth,
                               /*mediumEvent=*/0,
                               rec.frontFace,
                               rec.materialIndex,
                               /*scatterIsDelta=*/false,
                               rec.point,
                               float4(float(type),
                                      float(rec.primitiveIndex),
                                      0.0f,
                                      0.0f),
                               shadingNormal);
            record_debug_event(*debugContext,
                               kDebugEventDirectLightAuditMaterial,
                               depth,
                               mediumDepth,
                               mediumDepth,
                               /*mediumEvent=*/0,
                               rec.frontFace,
                               rec.materialIndex,
                               /*scatterIsDelta=*/false,
                               auditBaseColor,
                               float4(material_closure_aov_roughness(material),
                                      auditBaseColorUv.x,
                                      auditBaseColorUv.y,
                                      auditBaseColorLod),
                               float3(float(material.textureIndices0.x), 0.0f, 0.0f));
            if (uniforms.directLightMode == kDirectLightModeRis ||
                uniforms.directLightMode == kDirectLightModeRisSpatialReuse ||
                uniforms.directLightMode == kDirectLightModeRisTemporalReuse ||
                uniforms.directLightMode == kDirectLightModeRisWorldReuse ||
                uniforms.directLightMode == kDirectLightModeRisRegirCache ||
                uniforms.directLightMode == kDirectLightModeRestirDi ||
                uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) {
                record_debug_event(*debugContext,
                                   kDebugEventRisAuditState,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   float3(risAuditReservoir.wSum,
                                          risAuditReservoir.W,
                                          risAuditReservoir.winner.pdf),
                                   float4(float(risAuditReservoir.winner.primitiveIndex),
                                          float(risAuditReservoir.M),
                                          risAuditReservoir.valid ? 1.0f : 0.0f,
                                          risAuditVisible ? 1.0f : 0.0f),
                                   float3(risAuditDistance,
                                          risAuditNDotL,
                                          risAuditBsdfPdf));
                if (risAuditReservoir.valid) {
                    record_debug_event(*debugContext,
                                       kDebugEventRisAuditWinnerA,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       risAuditReservoir.winner.position,
                                       float4(risAuditReservoir.winner.bary.x,
                                              risAuditReservoir.winner.bary.y,
                                              0.0f,
                                              0.0f),
                                       risAuditReservoir.winner.normal);
                    record_debug_event(*debugContext,
                                       kDebugEventRisAuditWinnerB,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       risAuditReservoir.winner.emission,
                                       float4(0.0f),
                                       risAuditBsdfValue);
                    record_debug_event(*debugContext,
                                       kDebugEventRisAuditWinnerC,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       risAuditContribution,
                                       float4(0.0f),
                                       float3(0.0f));
                }
                if (uniforms.directLightMode == kDirectLightModeRisSpatialReuse) {
                    record_debug_event(*debugContext,
                                       kDebugEventSpatialReuseAudit,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       float3(float(spatialNeighborTarget),
                                              float(spatialNeighborsConsidered),
                                              float(spatialNeighborsAccepted)),
                                       float4(float(spatialRejectedDepth),
                                              float(spatialRejectedNormal),
                                              float(spatialRejectedInvalid),
                                              spatialReuseAttempted ? 1.0f : 0.0f),
                                       float3(float(spatialWinnerOffset.x),
                                              float(spatialWinnerOffset.y),
                                              0.0f));
                    record_debug_event(*debugContext,
                                       kDebugEventSpatialReuseWeights,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       float3(spatialLastMisWeight,
                                              spatialLastMergeWeight,
                                              0.0f),
                                       float4(0.0f),
                                       float3(0.0f));
                }
                if (uniforms.directLightMode == kDirectLightModeRisTemporalReuse ||
                    uniforms.directLightMode == kDirectLightModeRestirDi ||
                    uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) {
                    record_debug_event(*debugContext,
                                       kDebugEventTemporalReuseAudit,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       float3(temporalPreviousAvailable ? 1.0f : 0.0f,
                                              temporalReuseAccepted ? 1.0f : 0.0f,
                                              float(temporalPreviousPrimitiveIndex)),
                                       float4(float(temporalRejectedDepth),
                                              float(temporalRejectedNormal),
                                              float(temporalRejectedInvalid),
                                              temporalReuseAttempted ? 1.0f : 0.0f),
                                       float3(0.0f));
                    record_debug_event(*debugContext,
                                       kDebugEventTemporalReuseWeights,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       float3(temporalLastMisWeight,
                                              temporalLastMergeWeight,
                                              0.0f),
                                       float4(0.0f),
                                       float3(0.0f));
                }
                if (uniforms.directLightMode == kDirectLightModeRisWorldReuse) {
                    record_debug_event(*debugContext,
                                       kDebugEventWorldReuseAudit,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       float3(float(worldCandidatesConsidered),
                                              float(worldCandidatesAccepted),
                                              float(worldCellHash)),
                                       float4(float(worldRejectedDepth),
                                              float(worldRejectedNormal),
                                              float(worldRejectedInvalid),
                                              worldReuseAttempted ? 1.0f : 0.0f),
                                       float3(float(worldCell.x),
                                              float(worldCell.y),
                                              float(worldCell.z)));
                    record_debug_event(*debugContext,
                                       kDebugEventWorldReuseWeights,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       float3(worldLastMisWeight,
                                              worldLastMergeWeight,
                                              float(worldRejectedCell)),
                                       float4(float(worldCandidatePrimitiveIndex),
                                              worldReuseAccepted ? 1.0f : 0.0f,
                                              1.0f,
                                              4.0f),
                                       float3(0.0f));
                }
                if (uniforms.directLightMode == kDirectLightModeRisRegirCache ||
                    uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) {
                    record_debug_event(*debugContext,
                                       kDebugEventCacheReuseAudit,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       float3(float(cacheCandidatesConsidered),
                                              float(cacheCandidatesAccepted),
                                              float(cacheCellHash)),
                                       float4(float(cacheRejectedDepth),
                                              float(cacheRejectedNormal),
                                              float(cacheRejectedInvalid),
                                              cacheReuseAttempted ? 1.0f : 0.0f),
                                       float3(float(cacheCell.x),
                                              float(cacheCell.y),
                                              float(cacheCell.z)));
                    record_debug_event(*debugContext,
                                       kDebugEventCacheReuseWeights,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       float3(cacheLastMisWeight,
                                              cacheLastMergeWeight,
                                              float(cacheRejectedCell)),
                                       float4(float(cacheCandidatePrimitiveIndex),
                                              cacheReuseAccepted ? 1.0f : 0.0f,
                                              cacheStateAvailable ? 1.0f : 0.0f,
                                              4.0f),
                                       float3(cacheFallbackUsed ? 1.0f : 0.0f,
                                              float(cacheSourceFrameIndex),
                                              float(cacheEntriesAvailable)));
                }
            }
            if (rectLightCount > 0u) {
                for (uint rectIndex = 0u; rectIndex < uniforms.rectangleCount; ++rectIndex) {
                    RectLightSample auditLight;
                    if (!deterministic_rect_light_sample(uniforms,
                                                         rectangles,
                                                         materials,
                                                         environmentTexture,
                                                         rec,
                                                         rectLightCount,
                                                         rectIndex,
                                                         auditLight)) {
                        continue;
                    }
                    float nDotL = max(dot(shadingNormal, auditLight.direction), 0.0f);
                    BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                            rec.point,
                                                            shadingNormal,
                                                            wo,
                                                            auditLight.direction,
                                                            clampParams,
                                                            uniforms.sssMode,
                                                            diffuseOcclusion,
                                                            specularOnly);
                    float bsdfPdf = bsdfEval.pdf;
                    float misWeight = 1.0f;
                    if (bsdfPdf > 0.0f) {
                        float denom = auditLight.pdf + bsdfPdf;
                        if (denom > 0.0f) {
                            misWeight = clamp(auditLight.pdf / denom,
                                              kMisWeightClampMin,
                                              kMisWeightClampMax);
                        }
                    }
                    HitRecord shadowRec;
                    HardwareShadowTraceAudit hwShadowAudit;
                    SoftwareShadowTraceAudit swShadowAudit;
                    Ray auditShadowRay;
                    auditShadowRay.origin = offset_ray_origin(rec, auditLight.direction);
                    auditShadowRay.direction = auditLight.direction;
                    float shadowMax = max(auditLight.distance - kEpsilon, kEpsilon);
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
                                                         /*stats=*/nullptr,
                                                         auditShadowRay,
                                                         kEpsilon,
                                                         shadowMax,
                                                         /*anyHitOnly=*/true,
                                                         /*includeTriangles=*/true,
                                                         &swShadowAudit,
                                                         shadowRec);
                    swShadowAudit.consulted = true;
                    bool lightTwoSided = (rectangles[rectIndex].materialTwoSided.y != 0u);
                    if (debugContext) {
                        record_shadow_path_audit(*debugContext,
                                                 depth,
                                                 mediumDepth,
                                                 hwShadowAudit,
                                                 swShadowAudit,
                                                 rectIndex,
                                                 kDirectLightAuditKindRect,
                                                 kEpsilon,
                                                 shadowMax,
                                                 lightTwoSided,
                                                 !occluded);
                    }
                    float3 contribution = float3(0.0f);
                    if (!occluded && !bsdfEval.isDelta && !bsdfEval.isBssrdf &&
                        nDotL > 0.0f && auditLight.pdf > 0.0f) {
                        contribution = auditLight.emission * bsdfEval.value * nDotL;
                        contribution *= misWeight / auditLight.pdf;
                    }
                    record_debug_event(*debugContext,
                                       kDebugEventDirectLightAuditEval,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/kDirectLightAuditKindRect,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       auditLight.direction,
                                       float4(float(rectIndex),
                                              auditLight.distance,
                                              nDotL,
                                              0.0f),
                                       bsdfEval.value);
                    record_debug_event(*debugContext,
                                       kDebugEventDirectLightAuditContrib,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/kDirectLightAuditKindRect,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       contribution,
                                       float4(float(rectIndex),
                                              occluded ? 0.0f : 1.0f,
                                              misWeight,
                                              auditLight.pdf),
                                       float3(bsdfPdf, 0.0f, 0.0f));
                }
            }
            if (uniforms.emissivePrimitiveCount > 0u) {
                for (uint primitiveAuditIndex = 0u; primitiveAuditIndex < uniforms.emissivePrimitiveCount; ++primitiveAuditIndex) {
                    EmissivePrimitiveAuditSample auditLight;
                    if (!deterministic_emissive_primitive_sample(uniforms,
                                                                 emissivePrimitives,
                                                                 rec,
                                                                 primitiveAuditIndex,
                                                                 auditLight)) {
                        continue;
                    }
                    float nDotL = max(dot(shadingNormal, auditLight.direction), 0.0f);
                    BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                            rec.point,
                                                            shadingNormal,
                                                            wo,
                                                            auditLight.direction,
                                                            clampParams,
                                                            uniforms.sssMode,
                                                            diffuseOcclusion,
                                                            specularOnly);
                    float bsdfPdf = bsdfEval.pdf;
                    float misWeight = 1.0f;
                    HitRecord shadowRec;
                    HardwareShadowTraceAudit hwShadowAudit;
                    SoftwareShadowTraceAudit swShadowAudit;
                    Ray auditShadowRay;
                    auditShadowRay.origin = offset_ray_origin(rec, auditLight.direction);
                    auditShadowRay.direction = auditLight.direction;
                    float shadowMax = max(auditLight.distance - kEpsilon, kEpsilon);
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
                                                         /*stats=*/nullptr,
                                                         auditShadowRay,
                                                         kEpsilon,
                                                         shadowMax,
                                                         /*anyHitOnly=*/true,
                                                         /*includeTriangles=*/true,
                                                         &swShadowAudit,
                                                         shadowRec);
                    swShadowAudit.consulted = true;
                    if (occluded &&
                        shadowRec.primitiveType == kPrimitiveTypeTriangle &&
                        shadowRec.primitiveIndex == auditLight.primitiveIndex) {
                        occluded = false;
                    }
                    if (debugContext) {
                        record_shadow_path_audit(*debugContext,
                                                 depth,
                                                 mediumDepth,
                                                 hwShadowAudit,
                                                 swShadowAudit,
                                                 auditLight.primitiveIndex,
                                                 kDirectLightAuditKindEmissivePrimitive,
                                                 kEpsilon,
                                                 shadowMax,
                                                 /*lightTwoSided=*/false,
                                                 !occluded);
                    }
                    float3 contribution = float3(0.0f);
                    if (!occluded && !bsdfEval.isDelta && !bsdfEval.isBssrdf &&
                        nDotL > 0.0f && auditLight.pdf > 0.0f) {
                        contribution = auditLight.emission * bsdfEval.value * nDotL;
                        contribution *= misWeight / auditLight.pdf;
                    }
                    record_debug_event(*debugContext,
                                       kDebugEventDirectLightAuditEval,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/kDirectLightAuditKindEmissivePrimitive,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       auditLight.direction,
                                       float4(float(auditLight.primitiveIndex),
                                              auditLight.distance,
                                              nDotL,
                                              0.0f),
                                       bsdfEval.value);
                    record_debug_event(*debugContext,
                                       kDebugEventDirectLightAuditContrib,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/kDirectLightAuditKindEmissivePrimitive,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       contribution,
                                       float4(float(auditLight.primitiveIndex),
                                              occluded ? 0.0f : 1.0f,
                                              misWeight,
                                              auditLight.pdf),
                                       float3(bsdfPdf, 0.0f, 0.0f));
                }
            }
        }

        if (!surfaceIsDelta && envSampling) {
            EnvironmentSample envSample;
            if (sample_environment(uniforms,
                                   environmentTexture,
                                   environmentConditionalAlias,
                                   environmentMarginalAlias,
                                   environmentPdf,
                                   state,
                                   envSample)) {
                float overrideLod = 0.0f;
                bool useOverride = environment_mip_override(uniforms, environmentTexture, overrideLod);
                if (environmentTexture.get_num_mip_levels() > 1u) {
                    float envRoughness = environment_lighting_roughness(material);
                    if (envRoughness < 0.95f) {
                        float envLod = environment_lod_from_roughness(envRoughness,
                                                                      environmentTexture);
                        envSample.radiance = environment_color_lod(environmentTexture,
                                                                   envSample.direction,
                                                                   uniforms.environmentRotation,
                                                                   uniforms.environmentIntensity,
                                                                   envLod,
                                                                   uniforms);
                    }
                }
                if (useOverride) {
                    envSample.radiance = environment_color_lod(environmentTexture,
                                                               envSample.direction,
                                                               uniforms.environmentRotation,
                                                               uniforms.environmentIntensity,
                                                               overrideLod,
                                                               uniforms);
                }
                float nDotL = max(dot(shadingNormal, envSample.direction), 0.0f);
                if (envSample.pdf > 0.0f && nDotL > 0.0f) {
                    BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                            rec.point,
                                                            shadingNormal,
                                                            wo,
                                                            envSample.direction,
                                                            clampParams,
                                                            uniforms.sssMode,
                                                            diffuseOcclusion,
                                                            specularOnly);
                    float3 bsdfValue = bsdfEval.value;
                    float maxComponent = max(max(bsdfValue.x, bsdfValue.y), bsdfValue.z);
                    bool bsdfConnectable = !bsdfEval.isDelta &&
                                           !bsdfEval.isBssrdf &&
                                           (maxComponent > 0.0f);
                    if (bsdfConnectable) {
                        float bsdfPdf = bsdfEval.pdf;
                        float weight = 1.0f;
                        if (bsdfPdf > 0.0f) {
                            float denom = envSample.pdf + bsdfPdf;
                            if (denom > 0.0f) {
                                weight = clamp(envSample.pdf / denom,
                                               kMisWeightClampMin,
                                               kMisWeightClampMax);
                            }
                        }

                        Ray shadowRay;
                        shadowRay.origin = offset_ray_origin(rec, envSample.direction);
                        shadowRay.direction = envSample.direction;
                        HitRecord shadowRec = make_empty_hit_record();
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
                                                             kInfinity,
                                                             /*anyHitOnly=*/true,
                                                             /*includeTriangles=*/true,
                                                             shadowRec);
                        float3 envDirection = envSample.direction;
                        float3 chainWeight = float3(1.0f);
                        bool usedDeltaChain = false;
                        bool connected = !occluded;

#if ENABLE_MNEE_CAUSTICS
                        bool useMneeEnvChain = (uniforms.enableMnee != 0u) &&
                                               (uniforms.enableMneeSecondary != 0u);
                        if (!connected && useMneeEnvChain) {
                            if (stats) {
                                atomic_fetch_add_explicit(&stats->mneeEnvAttemptCount, 1u, memory_order_relaxed);
                            }
                            connected = trace_environment_delta_chain_software(uniforms,
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
                                                                               materials,
                                                                               stats,
                                                                               state,
                                                                               clampParams,
                                                                               specularOnly,
                                                                               shadowRay,
                                                                               envDirection,
                                                                               chainWeight,
                                                                               usedDeltaChain);
                        }
#endif
                        if (connected) {
                            float3 envRadiance = envSample.radiance;
                            if (usedDeltaChain) {
                                envRadiance = environment_color(environmentTexture,
                                                                envDirection,
                                                                uniforms.environmentRotation,
                                                                uniforms.environmentIntensity,
                                                                uniforms);
                            }
                            float3 contribution = envRadiance * bsdfValue * nDotL;
                            contribution *= weight / envSample.pdf;
                            contribution *= chainWeight;
                            if (all(isfinite(contribution))) {
                                radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                                if (debugContext) {
                                    record_debug_event(*debugContext,
                                                       kDebugEventEnvNee,
                                                       depth,
                                                       mediumDepth,
                                                       mediumDepth,
                                                       /*mediumEvent=*/0,
                                                       rec.frontFace,
                                                       rec.materialIndex,
                                                       /*scatterIsDelta=*/false,
                                                       throughput,
                                                       float4(envSample.pdf,
                                                              bsdfPdf,
                                                              weight,
                                                              nDotL),
                                                       contribution);
                                }
                                if (usedDeltaChain && stats) {
                                    atomic_fetch_add_explicit(&stats->mneeEnvAddedCount, 1u, memory_order_relaxed);
                                    stats_add_mnee_luma(stats, contribution);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (debugContext) {
            record_debug_event(*debugContext,
                               kDebugEventShadingNormal,
                               depth,
                               mediumDepth,
                               mediumDepth,
                               /*mediumEvent=*/0,
                               rec.frontFace,
                               rec.materialIndex,
                               /*scatterIsDelta=*/false,
                               throughput,
                               float4(rec.t,
                                      float(type),
                                      0.0f,
                                      0.0f),
                               shadingNormal);
        }

        BsdfSampleResult bsdfSample;
        uint rngStateBeforeBsdf = state;
        bool usedRandomWalk = false;
        bool enableRandomWalk = material_is_subsurface(material) &&
                                sss_use_random_walk(uniforms.sssMode, material) &&
                                rec.frontFace != 0u;
        if (enableRandomWalk) {
            bsdfSample = sample_sss_random_walk_software(uniforms,
                                                         material,
                                                         rec,
                                                         wo,
                                                         incidentDir,
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
                                                         state,
                                                         clampParams);
            usedRandomWalk = (bsdfSample.pdf > 0.0f);
        }
        if (!usedRandomWalk) {
            bsdfSample = material_closure_sample(material,
                                                 rec.point,
                                                 shadingNormal,
                                                 wo,
                                                 incidentDir,
                                                 rec.frontFace != 0u,
                                                 state,
                                                 clampParams,
                                                 uniforms.sssMode,
                                                 diffuseOcclusion,
                                                 specularOnly);
        }
        apply_restir_gi_diffuse_prototype(uniforms,
                                          material,
                                          rec.point,
                                          shadingNormal,
                                          wo,
                                          incidentDir,
                                          rec.frontFace != 0u,
                                          clampParams,
                                          diffuseOcclusion,
                                          specularOnly,
                                          depth,
                                          pixelCoord,
                                          restirPtReservoirs,
                                          stats,
                                          state,
                                          bsdfSample);
        apply_path_guiding_prototype(uniforms,
                                     material,
                                     rec.point,
                                     shadingNormal,
                                     wo,
                                     clampParams,
                                     diffuseOcclusion,
                                     specularOnly,
                                     depth,
                                     pixelCoord,
                                     pathGuidingStates,
                                     stats,
                                     state,
                                     bsdfSample);
        apply_restir_pt_experimental_path_reuse(uniforms,
                                                material,
                                                rec.point,
                                                shadingNormal,
                                                wo,
                                                throughput,
                                                clampParams,
                                                diffuseOcclusion,
                                                specularOnly,
                                                depth,
                                                pixelCoord,
                                                restirPtReservoirs,
                                                stats,
                                                bsdfSample);
        capture_restir_pt_research_scaffold(uniforms,
                                            material,
                                            rec.point,
                                            shadingNormal,
                                            throughput,
                                            specularOnly,
                                            depth,
                                            pixelCoord,
                                            restirPtReservoirs,
                                            stats,
                                            bsdfSample);
        uint rngStateAfterBsdf = state;
        if (debugContext) {
            record_debug_event(*debugContext,
                               kDebugEventBsdfRng,
                               depth,
                               mediumDepth,
                               mediumDepth,
                               /*mediumEvent=*/0,
                               rec.frontFace,
                               rec.materialIndex,
                               bsdfSample.isDelta,
                               shadingNormal,
                               float4(as_type<float>(rngStateBeforeBsdf),
                                      as_type<float>(rngStateAfterBsdf),
                                      float(type),
                                      usedRandomWalk ? 1.0f : 0.0f),
                               wo);
        }
        if (debugContext) {
            record_debug_event(*debugContext,
                               kDebugEventBsdfState,
                               depth,
                               mediumDepth,
                               mediumDepth,
                               /*mediumEvent=*/0,
                               rec.frontFace,
                               rec.materialIndex,
                               bsdfSample.isDelta,
                               incidentDir,
                               float4(bsdfSample.pdf,
                                      bsdfSample.directionalPdf,
                                      float(bsdfSample.lobeType),
                                      bsdfSample.isDelta ? 1.0f : 0.0f),
                               bsdfSample.direction);
        }
        if (bsdfSample.pdf <= 0.0f) {
            break;
        }
        caustic_transport_note_path_vertex(uniforms,
                                           material,
                                           rec.point,
                                           shadingNormal,
                                           throughput,
                                           depth,
                                           bsdfSample,
                                           stats);
        if (radiance_cache_query_and_maybe_terminate(uniforms,
                                                     material,
                                                     rec.point,
                                                     shadingNormal,
                                                     specularOnly,
                                                     depth,
                                                     bsdfSample,
                                                     radianceCacheStates,
                                                     stats,
                                                     radiance,
                                                     throughput)) {
            break;
        }
        radiance_cache_train(radianceCacheStates,
                             uniforms,
                             material,
                             rec.point,
                             shadingNormal,
                             specularOnly,
                             depth,
                             bsdfSample,
                             stats);

        uint mediumDepthBefore = mediumDepth;
        if (bsdfSample.mediumEvent == 1) {
            float3 sigma = dielectric_sigma_a(material);
            sigma = max(sigma, float3(0.0f));
            if (mediumDepth < kMaxMediumStack) {
                mediumSigmaStack[mediumDepth] = sigma;
                mediumDepth += 1u;
            } else {
                mediumSigmaStack[kMaxMediumStack - 1u] = sigma;
            }
        } else if (bsdfSample.mediumEvent == -1) {
            if (mediumDepth > 0u) {
                mediumDepth -= 1u;
            }
        }
        volume_note_boundary_event(bsdfSample.mediumEvent, stats);
        uint mediumDepthAfter = mediumDepth;

        if (debugContext) {
            record_debug_event(*debugContext,
                               kDebugEventScatter,
                               depth,
                               mediumDepthBefore,
                               mediumDepthAfter,
                               bsdfSample.mediumEvent,
                               rec.frontFace,
                               rec.materialIndex,
                               bsdfSample.isDelta,
                               throughput,
                               float4(bsdfSample.pdf,
                                      bsdfSample.directionalPdf,
                                      rec.t,
                                      float(bsdfSample.lobeType)),
                               bsdfSample.weight);
        }

        bool causticCandidate = (!surfaceIsDelta) && (specularDepth > 0u);
        uint nextSpecularDepth = bsdfSample.isDelta ? (specularDepth + 1u) : 0u;
        bool didTransmission = false;
        if (bsdfSample.isDelta && type == 2u) {
            float3 dir = bsdfSample.direction;
            if (all(isfinite(dir)) && dot(dir, dir) > 0.0f) {
                float side = (rec.frontFace != 0u) ? 1.0f : -1.0f;
                didTransmission = (dot(shadingNormal, dir) * side) < 0.0f;
            }
        }
        if (didTransmission) {
            hadTransmission = true;
        }
        specularDepth = nextSpecularDepth;
        (void)causticCandidate;
        (void)hadTransmission;

        float3 nextOrigin;
        if (bsdfSample.hasExitPoint) {
            float3 exitNormal = bsdfSample.exitNormal;
            bool normalValid = all(isfinite(exitNormal)) && dot(exitNormal, exitNormal) > 0.0f;
            if (!normalValid) {
                exitNormal = rec.normal;
                normalValid = all(isfinite(exitNormal)) && dot(exitNormal, exitNormal) > 0.0f;
            }
            if (!normalValid) {
                exitNormal = float3(0.0f, 1.0f, 0.0f);
            }
            exitNormal = normalize(exitNormal);
            nextOrigin = offset_surface_point(bsdfSample.exitPoint, exitNormal, bsdfSample.direction);
            // HWRT still reports misses when a refracted ray's origin sits inside the mesh.
            // Push further along the exit normal plus a bit down the outgoing direction
            // so the TLAS starts well outside the surface.
            float normalBias = max(kHardwareOcclusionEpsilon * 4.0f, kRayOriginEpsilon * 32.0f);
            nextOrigin += exitNormal * normalBias;
            float3 dir = bsdfSample.direction;
            if (!all(isfinite(dir)) || dot(dir, dir) <= 0.0f) {
                dir = exitNormal;
            } else {
                dir = normalize(dir);
            }
            float directionalBias = max(kHardwareOcclusionEpsilon * 8.0f, kRayOriginEpsilon * 32.0f);
            nextOrigin += dir * directionalBias;
        } else {
            nextOrigin = offset_ray_origin(rec, bsdfSample.direction);
        }

        bool useMnee = (ENABLE_MNEE_CAUSTICS != 0) && (uniforms.enableMnee != 0u);
        bool specNeeEnabled = (uniforms.enableSpecularNee != 0u);
        float dirLenSq = dot(bsdfSample.direction, bsdfSample.direction);
        bool specDirectionValid = (dirLenSq > 0.0f) && all(isfinite(bsdfSample.direction));
        bool mneeEligible = false;
#if ENABLE_MNEE_CAUSTICS
        mneeEligible = useMnee &&
                       bsdfSample.isDelta &&
                       ((bsdfSample.mediumEvent <= 0) || didTransmission) &&
                       (type == 2u) &&
                       (nextSpecularDepth == 1u) &&
                       specDirectionValid;
#endif
        if (mneeEligible) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEligibleCount, 1u, memory_order_relaxed);
            }
#if PT_MNEE_OCCLUSION_PARITY
            HitRecord mneeSwRec;
            bool swHit = trace_scene_software_with_exclusion(uniforms,
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
                                                             excludeMeshIndex,
                                                             excludePrimitiveIndex,
                                                             mneeSwRec);
            if (stats) {
                if (!swHit) {
                    atomic_fetch_add_explicit(&stats->mneeHitHwSwHitMissCount,
                                              1u,
                                              memory_order_relaxed);
                } else {
                    float epsT = max(1.0e-3f, 1.0e-4f * fabs(rec.t));
                    float tDiff = fabs(rec.t - mneeSwRec.t);
                    if (tDiff > epsT) {
                        atomic_fetch_add_explicit(&stats->mneeHitHwSwTDiffCount,
                                                  1u,
                                                  memory_order_relaxed);
                    }
                    if (rec.frontFace != mneeSwRec.frontFace ||
                        rec.materialIndex != mneeSwRec.materialIndex ||
                        rec.meshIndex != mneeSwRec.meshIndex ||
                        rec.primitiveIndex != mneeSwRec.primitiveIndex) {
                        atomic_fetch_add_explicit(&stats->mneeHitHwSwIdMismatchCount,
                                                  1u,
                                                  memory_order_relaxed);
                    }
                    float3 hwN = rec.normal;
                    float3 swN = mneeSwRec.normal;
                    if (all(isfinite(hwN)) && all(isfinite(swN)) &&
                        dot(hwN, hwN) > 0.0f && dot(swN, swN) > 0.0f) {
                        float nDot = dot(normalize(hwN), normalize(swN));
                        if (nDot < 0.99f) {
                            atomic_fetch_add_explicit(&stats->mneeHitHwSwNormalMismatchCount,
                                                      1u,
                                                      memory_order_relaxed);
                        }
                    }
                }
            }
#endif
        }
        bool specNeeEligible = specNeeEnabled &&
                               bsdfSample.isDelta &&
                               (bsdfSample.mediumEvent <= 0) &&
                               specDirectionValid &&
                               !mneeEligible;

        if (specNeeEligible && envSampling &&
            environmentTexture.get_width() > 0 &&
            environmentTexture.get_height() > 0) {
            Ray neeRay;
            neeRay.origin = nextOrigin;
            neeRay.direction = normalize(bsdfSample.direction);
            HitRecord shadowRec = make_empty_hit_record();
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
                                                 neeRay,
                                                 kEpsilon,
                                                 kInfinity,
                                                 /*anyHitOnly=*/true,
                                                 /*includeTriangles=*/true,
                                                 shadowRec);
            if (!occluded) {
                float envPdf = environment_pdf(uniforms, environmentPdf, neeRay.direction);
                envPdf = max(envPdf, kSpecularNeePdfFloor);
                float invEnvPdf = min(1.0f / envPdf, kSpecularNeeInvPdfClamp);
                float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                float denom = envPdf + bsdfPdf;
                float misWeight = (denom > 0.0f) ? (envPdf / denom) : 0.0f;
                misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                float3 envColor = environment_color(environmentTexture,
                                                    neeRay.direction,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity,
                                                    uniforms);
                float3 neeContribution = bsdfSample.weight * envColor * (misWeight * invEnvPdf);
                if (all(isfinite(neeContribution))) {
                    radiance += clamp_firefly_contribution(throughput, neeContribution, clampParams);
                    if (debugContext) {
                        record_debug_event(*debugContext,
                                           kDebugEventSpecEnvNee,
                                           depth,
                                           mediumDepth,
                                           mediumDepth,
                                           /*mediumEvent=*/0,
                                           rec.frontFace,
                                           rec.materialIndex,
                                           bsdfSample.isDelta,
                                           throughput,
                                           float4(envPdf,
                                                  bsdfPdf,
                                                  misWeight,
                                                  invEnvPdf),
                                           neeContribution);
                    }
                    if (stats) {
                        atomic_fetch_add_explicit(&stats->specNeeEnvAddedCount, 1u, memory_order_relaxed);
                    }
                }
            } else if (stats) {
                atomic_fetch_add_explicit(&stats->specularNeeOcclusionHitCount,
                                          1u,
                                          memory_order_relaxed);
            }
        }

        if (specNeeEligible && rectLightCount > 0u) {
            Ray neeRay;
            neeRay.origin = nextOrigin;
            neeRay.direction = normalize(bsdfSample.direction);
            HitRecord lightRec;
            bool hitLight = trace_scene_software(uniforms,
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
                                                 neeRay,
                                                 kEpsilon,
                                                 kInfinity,
                                                 /*anyHitOnly=*/false,
                                                 /*includeTriangles=*/true,
                                                 lightRec);
            if (hitLight) {
                MneeRectHit mneeHit;
                if (mnee_rect_light_hit(uniforms,
                                        rectangles,
                                        materials,
                                        environmentTexture,
                                        rectLightCount,
                                        lightRec,
                                        nextOrigin,
                                        mneeHit)) {
                    float lightPdf = max(mneeHit.pdf, kSpecularNeePdfFloor);
                    float invLightPdf = min(1.0f / lightPdf, kSpecularNeeInvPdfClamp);
                    float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                    float denom = lightPdf + bsdfPdf;
                    float misWeight = (denom > 0.0f) ? (lightPdf / denom) : 0.0f;
                    misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                    float3 contribution = bsdfSample.weight * mneeHit.emission *
                                          (misWeight * invLightPdf);
                    if (all(isfinite(contribution))) {
                        radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                        if (debugContext) {
                            record_debug_event(*debugContext,
                                               kDebugEventSpecRectNee,
                                               depth,
                                               mediumDepth,
                                               mediumDepth,
                                               /*mediumEvent=*/0,
                                               rec.frontFace,
                                               rec.materialIndex,
                                               bsdfSample.isDelta,
                                               throughput,
                                               float4(lightPdf,
                                                      bsdfPdf,
                                                      misWeight,
                                                      invLightPdf),
                                               contribution);
                        }
                        if (stats) {
                            atomic_fetch_add_explicit(&stats->specNeeRectAddedCount, 1u, memory_order_relaxed);
                        }
                    }
                }
            }
        }

#if ENABLE_MNEE_CAUSTICS
        if (mneeEligible && envSampling &&
            environmentTexture.get_width() > 0 &&
            environmentTexture.get_height() > 0) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEnvAttemptCount, 1u, memory_order_relaxed);
            }
            Ray mneeRay;
            mneeRay.origin = nextOrigin;
            mneeRay.direction = normalize(bsdfSample.direction);
            HitRecord shadowRec;
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
                                                 mneeRay,
                                                 kEpsilon,
                                                 kInfinity,
                                                 /*anyHitOnly=*/true,
                                                 /*includeTriangles=*/true,
                                                 shadowRec);
            if (!occluded) {
                float envPdf = environment_pdf(uniforms, environmentPdf, mneeRay.direction);
                envPdf = max(envPdf, kSpecularNeePdfFloor);
                float invEnvPdf = min(1.0f / envPdf, kSpecularNeeInvPdfClamp);
                float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                float denom = envPdf + bsdfPdf;
                float misWeight = (denom > 0.0f) ? (envPdf / denom) : 0.0f;
                misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                float3 envColor = environment_color(environmentTexture,
                                                    mneeRay.direction,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity,
                                                    uniforms);
                float3 neeContribution = bsdfSample.weight * envColor * (misWeight * invEnvPdf);
                if (all(isfinite(neeContribution))) {
                    radiance += clamp_firefly_contribution(throughput, neeContribution, clampParams);
                    if (stats) {
                        atomic_fetch_add_explicit(&stats->mneeEnvAddedCount, 1u, memory_order_relaxed);
                        stats_add_mnee_luma(stats, neeContribution);
                    }
                }
            }
        }

        if (mneeEligible && rectLightCount > 0u) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeRectAttemptCount, 1u, memory_order_relaxed);
            }
            Ray mneeRay;
            mneeRay.origin = nextOrigin;
            mneeRay.direction = normalize(bsdfSample.direction);
            HitRecord lightRec;
            bool hitLight = trace_scene_software(uniforms,
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
                                                 mneeRay,
                                                 kEpsilon,
                                                 kInfinity,
                                                 /*anyHitOnly=*/false,
                                                 /*includeTriangles=*/true,
                                                 lightRec);
            if (hitLight) {
                MneeRectHit mneeHit;
                if (mnee_rect_light_hit(uniforms,
                                        rectangles,
                                        materials,
                                        environmentTexture,
                                        rectLightCount,
                                        lightRec,
                                        nextOrigin,
                                        mneeHit)) {
                    float lightPdf = max(mneeHit.pdf, kSpecularNeePdfFloor);
                    float invLightPdf = min(1.0f / lightPdf, kSpecularNeeInvPdfClamp);
                    float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                    float denom = lightPdf + bsdfPdf;
                    float misWeight = (denom > 0.0f) ? (lightPdf / denom) : 0.0f;
                    misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                    float3 contribution = bsdfSample.weight * mneeHit.emission *
                                          (misWeight * invLightPdf);
                    if (all(isfinite(contribution))) {
                        radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                        if (stats) {
                            atomic_fetch_add_explicit(&stats->mneeRectAddedCount, 1u, memory_order_relaxed);
                            stats_add_mnee_luma(stats, contribution);
                        }
                    }
                }
            }
        }

        if (mneeEligible && uniforms.enableMneeSecondary != 0u) {
            Ray chainRay;
            chainRay.origin = nextOrigin;
            chainRay.direction = normalize(bsdfSample.direction);
            HitRecord chainRec;
            bool chainHit = trace_scene_software(uniforms,
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
                                                 chainRay,
                                                 kEpsilon,
                                                 kInfinity,
                                                 /*anyHitOnly=*/false,
                                                 /*includeTriangles=*/true,
                                                 chainRec);
            if (chainHit && materials && uniforms.materialCount > 0u) {
                bool chainHitIsLight = false;
                if (rectLightCount > 0u) {
                    MneeRectHit chainLightHit;
                    if (mnee_rect_light_hit(uniforms,
                                            rectangles,
                                            materials,
                                            environmentTexture,
                                            rectLightCount,
                                            chainRec,
                                            chainRay.origin,
                                            chainLightHit)) {
                        chainHitIsLight = true;
                    }
                }
                if (!chainHitIsLight) {
                    uint chainMatIndex = min(chainRec.materialIndex, uniforms.materialCount - 1u);
                    MaterialData chainMaterial = materials[chainMatIndex];
                    if (material_is_delta(chainMaterial)) {
                        float3 chainNormal = chainRec.normal;
                        if (!all(isfinite(chainNormal)) || dot(chainNormal, chainNormal) <= 0.0f) {
                            chainNormal = float3(0.0f, 1.0f, 0.0f);
                        }
                        chainNormal = normalize(chainNormal);
                        float3 chainIncident = normalize(chainRay.direction);
                        float3 chainWo = -chainIncident;
                        uint chainState = state;
                        BsdfSampleResult chainSample = material_closure_sample(chainMaterial,
                                                                               chainRec.point,
                                                                               chainNormal,
                                                                               chainWo,
                                                                               chainIncident,
                                                                               chainRec.frontFace != 0u,
                                                                               chainState,
                                                                               clampParams,
                                                                               uniforms.sssMode,
                                                                               1.0f,
                                                                   specularOnly);
                        if (chainSample.pdf > 0.0f &&
                            chainSample.isDelta &&
                            (chainSample.mediumEvent <= 0)) {
                            float3 chainDir = safe_normalize(chainSample.direction);
                            if (all(isfinite(chainDir)) && dot(chainDir, chainDir) > 0.0f) {
                                float3 chainOrigin = offset_ray_origin(chainRec, chainDir);
                                float3 combinedWeight = bsdfSample.weight * chainSample.weight;
                                float bsdfPdf = max(bsdfSample.directionalPdf * chainSample.directionalPdf,
                                                    kSpecularNeePdfFloor);
                                if (envSampling &&
                                    environmentTexture.get_width() > 0 &&
                                    environmentTexture.get_height() > 0) {
                                    Ray envRay;
                                    envRay.origin = chainOrigin;
                                    envRay.direction = normalize(chainDir);
                                    HitRecord envRec;
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
                                                                         envRay,
                                                                         kEpsilon,
                                                                         kInfinity,
                                                                         /*anyHitOnly=*/true,
                                                                         /*includeTriangles=*/true,
                                                                         envRec);
                                    if (!occluded) {
                                        float envPdf = environment_pdf(uniforms, environmentPdf, envRay.direction);
                                        envPdf = max(envPdf, kSpecularNeePdfFloor);
                                        float invEnvPdf = min(1.0f / envPdf, kSpecularNeeInvPdfClamp);
                                        float denom = envPdf + bsdfPdf;
                                        float misWeight = (denom > 0.0f) ? (envPdf / denom) : 0.0f;
                                        misWeight = clamp(misWeight,
                                                          kMisWeightClampMin,
                                                          kMisWeightClampMax);
                                        float3 envColor = environment_color(environmentTexture,
                                                                            envRay.direction,
                                                                            uniforms.environmentRotation,
                                                                            uniforms.environmentIntensity,
                                                                            uniforms);
                                        float3 contribution = combinedWeight * envColor *
                                                              (misWeight * invEnvPdf);
                                        if (all(isfinite(contribution))) {
                                            radiance += clamp_firefly_contribution(throughput,
                                                                                   contribution,
                                                                                   clampParams);
                                        }
                                    }
                                }
                                if (rectLightCount > 0u) {
                                    Ray lightRay;
                                    lightRay.origin = chainOrigin;
                                    lightRay.direction = normalize(chainDir);
                                    HitRecord lightRec;
                                    bool hitLight = trace_scene_software(uniforms,
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
                                                                         lightRay,
                                                                         kEpsilon,
                                                                         kInfinity,
                                                                         /*anyHitOnly=*/false,
                                                                         /*includeTriangles=*/true,
                                                                         lightRec);
                                    if (hitLight) {
                                        MneeRectHit mneeHit;
                                        if (mnee_rect_light_hit(uniforms,
                                                                rectangles,
                                                                materials,
                                                                environmentTexture,
                                                                rectLightCount,
                                                                lightRec,
                                                                chainOrigin,
                                                                mneeHit)) {
                                            float lightPdf = max(mneeHit.pdf, kSpecularNeePdfFloor);
                                            float invLightPdf = min(1.0f / lightPdf, kSpecularNeeInvPdfClamp);
                                            float denom = lightPdf + bsdfPdf;
                                            float misWeight = (denom > 0.0f) ? (lightPdf / denom) : 0.0f;
                                            misWeight = clamp(misWeight,
                                                              kMisWeightClampMin,
                                                              kMisWeightClampMax);
                                            float3 contribution = combinedWeight * mneeHit.emission *
                                                                  (misWeight * invLightPdf);
                                            if (all(isfinite(contribution))) {
                                                radiance += clamp_firefly_contribution(throughput,
                                                                                       contribution,
                                                                                       clampParams);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

#endif

        float3 throughputBeforeScatter = throughput;
        throughput *= bsdfSample.weight;
        throughput = clamp_path_throughput(throughput, clampParams);
        if (debugContext) {
            record_debug_event(*debugContext,
                               kDebugEventThroughput,
                               depth,
                               mediumDepthBefore,
                               mediumDepthAfter,
                               bsdfSample.mediumEvent,
                               rec.frontFace,
                               rec.materialIndex,
                               bsdfSample.isDelta,
                               throughputBeforeScatter,
                               float4(bsdfSample.pdf,
                                      bsdfSample.directionalPdf,
                                      rec.t,
                                      float(bsdfSample.lobeType)),
                               throughput);
        }

        if (!all(isfinite(throughput))) {
            break;
        }

        float maxThroughput = max(max(throughput.x, throughput.y), throughput.z);
        if (maxThroughput <= 0.0f) {
            break;
        }

        if (debugContext) {
            record_debug_event(*debugContext,
                               kDebugEventRay,
                               depth,
                               mediumDepthBefore,
                               mediumDepthAfter,
                               bsdfSample.mediumEvent,
                               rec.frontFace,
                               rec.materialIndex,
                               bsdfSample.isDelta,
                               throughput,
                               float4(bsdfSample.pdf,
                                      bsdfSample.directionalPdf,
                                      rec.t,
                                      float(bsdfSample.lobeType)),
                               bsdfSample.direction);
        }

        bool nextEnvLodActive = false;
        float nextEnvLod = 0.0f;
        if (bsdfSample.lobeType == 1u && !bsdfSample.isDelta) {
            float maxMip = environment_max_mip(environmentTexture);
            if (maxMip > 0.0f) {
                nextEnvLod = environment_lod_from_roughness(bsdfSample.lobeRoughness,
                                                            environmentTexture);
                nextEnvLodActive = true;
            }
        }
        envLodActive = nextEnvLodActive;
        envLod = nextEnvLod;

        rayCone.width = ray_cone_width_at_distance(rayCone, hitDistanceWorld);
        rayCone.spread = min(rayCone.spread +
                             bsdf_cone_spread_increment(bsdfSample.lobeType,
                                                        bsdfSample.lobeRoughness,
                                                        bsdfSample.isDelta),
                             1.5f);

        lastBsdfPdf = (bsdfSample.directionalPdf > 0.0f) ? bsdfSample.directionalPdf : bsdfSample.pdf;
        lastScatterWasDelta = bsdfSample.isDelta;
        ray.origin = nextOrigin;
        ray.direction = bsdfSample.direction;

        if (uniforms.useRussianRoulette != 0 && depth >= 5) {
            float continueProbability = clamp(maxThroughput, 0.05f, 0.95f);
            if (rand_uniform(state) > continueProbability) {
                break;
            }
            throughput /= continueProbability;
        }
    }

    #undef clamp_path_throughput
    #undef clamp_firefly_contribution
    return radiance;
}

#if __METAL_VERSION__ >= 310
inline float3 trace_path_hardware(constant PathtraceUniforms& uniforms,
                                  acceleration_structure<instancing> accel,
                                  device const MeshInfo* meshInfos,
                                  device const TriangleData* triangleData,
                                  device const LightPrimitive* emissivePrimitives,
                                  device const uint* instanceUserIds,
                                  device const SphereData* spheres,
                                  device const RectData* rectangles,
                                  device const MaterialData* materials,
                                  device const SceneVertex* sceneVertices,
                                  device const uint3* meshIndices,
                                  device const BvhNode* tlasNodes,
                                  device const uint* tlasPrimIndices,
                                  device const BvhNode* blasNodes,
                                  device const uint* blasPrimIndices,
                                  device const SoftwareInstanceInfo* instanceInfos,
                                  Ray ray,
                                  const PrimaryRayDiff primaryRayDiff,
                                  thread uint& state,
                                  device const BvhNode* nodes,
                                  device const uint* primitiveIndices,
                                  device PathtraceStats* stats,
                                  texture2d<float, access::sample> environmentTexture,
                                  array<texture2d<float, access::sample>, kMaxMaterialTextures> materialTextures,
                                  array<sampler, kMaxMaterialSamplers> materialSamplers,
                                  device const MaterialTextureInfo* materialTextureInfos,
                                  device const EnvironmentAliasEntry* environmentConditionalAlias,
                                  device const EnvironmentAliasEntry* environmentMarginalAlias,
                                  device const float* environmentPdf,
                                  device PathGuidingReservoirState* pathGuidingStates,
                                  device RestirPtReservoirState* restirPtReservoirs,
                                  device RadianceCacheState* radianceCacheStates,
                                  uint2 pixelCoord,
                                  // Optional AOV outputs
                                  thread float3* outFirstHitAlbedo = nullptr,
                                  thread float3* outFirstHitNormal = nullptr,
                                  thread float4* outFirstHitPosition = nullptr,
                                  thread float4* outFirstHitMaterial = nullptr,
                                  thread PathtraceDebugContext* debugContext = nullptr) {
    float3 throughput = float3(1.0f, 1.0f, 1.0f);
    float3 radiance = float3(0.0f, 0.0f, 0.0f);
    float lastBsdfPdf = 1.0f;
    bool lastScatterWasDelta = true;
    bool isFirstHit = true;
    float envLod = 0.0f;
    bool envLodActive = false;
    RayCone rayCone = make_primary_ray_cone(uniforms);
    uint rectLightCount = (rectangles && uniforms.rectangleCount > 0 && materials)
                              ? count_rect_lights(uniforms, rectangles, materials)
                              : 0u;
    const bool envSampling = environment_sampling_available(uniforms,
                                                            environmentConditionalAlias,
                                                            environmentMarginalAlias,
                                                            environmentPdf);
    FireflyClampParams clampParams = make_firefly_params(uniforms);
#define clamp_firefly_contribution(throughput, contribution, params) \
    clamp_firefly_contribution((throughput), (contribution), (params), stats)
#define clamp_path_throughput(throughput, params) \
    clamp_path_throughput((throughput), (params), stats)
    constexpr uint kMaxMediumStack = 8u;
    float3 mediumSigmaStack[kMaxMediumStack];
    for (uint i = 0; i < kMaxMediumStack; ++i) {
        mediumSigmaStack[i] = float3(0.0f);
    }
    uint mediumDepth = 0u;
    uint volumeScatteringDepth = 0u;
    SpectralPathState spectralState = spectral_make_path_state(uniforms, state, stats);
    HitRecord prevRec;
    bool prevValid = false;
    uint specularDepth = 0u;
    bool hadTransmission = false;
    bool parityInMediumDone = false;
    const bool softwareTrianglesAvailable =
        (tlasNodes && tlasPrimIndices && blasNodes && blasPrimIndices && instanceInfos && triangleData);
    const bool forcePureHWRTForGlass = (uniforms.forcePureHWRTForGlass != 0u);
    const bool enableMissFallback =
        !forcePureHWRTForGlass && (uniforms.enableHardwareMissFallback != 0u);
    const bool enableFirstHitFallback =
        !forcePureHWRTForGlass && (uniforms.enableHardwareFirstHitFromSoftware != 0u);
    const bool forceSoftware =
        !forcePureHWRTForGlass &&
        (uniforms.enableHardwareForceSoftware != 0u) &&
        softwareTrianglesAvailable;

    for (uint depth = 0; depth < uniforms.maxDepth; ++depth) {
        HitRecord rec;
        uint excludeMeshIndex = kInvalidIndex;
        uint excludePrimitiveIndex = kInvalidIndex;
        if (prevValid) {
            compute_exclusion_indices(prevRec, excludeMeshIndex, excludePrimitiveIndex);
        }
        const bool preferSoftwareForMedium =
            !forcePureHWRTForGlass && (mediumDepth > 0u) && softwareTrianglesAvailable;

        bool doParity = false;
#if PT_DEBUG_TOOLS
        if (debugContext &&
            uniforms.parityAssertEnabled != 0u &&
            uniforms.parityAssertMode != 0u &&
            debugContext->buffer != nullptr &&
            debugContext->pixelX == uniforms.parityPixelX &&
            debugContext->pixelY == uniforms.parityPixelY) {
            if (uniforms.parityAssertMode == kParityModeProbePixel) {
                doParity = (depth == uniforms.parityPadding0);
            } else if (uniforms.parityAssertMode == kParityModeFirstInMedium) {
                if (!parityInMediumDone && mediumDepth > 0u) {
                    doParity = true;
                    parityInMediumDone = true;
                }
            }
        }
#endif

#if PT_DEBUG_TOOLS
        if (doParity) {
            uint parityAllowed = min(debugContext->buffer->parityMaxEntries,
                                     kPathtraceParityMaxEntries);
            if (parityAllowed > 0u) {
                uint parityRecorded =
                    atomic_load_explicit(&debugContext->buffer->parityWriteIndex,
                                         memory_order_relaxed);
                if (parityRecorded < parityAllowed) {
                    atomic_fetch_add_explicit(&debugContext->buffer->parityChecksPerformed,
                                              1u,
                                              memory_order_relaxed);
                    if (mediumDepth > 0u) {
                        atomic_fetch_add_explicit(&debugContext->buffer->parityChecksInMedium,
                                                  1u,
                                                  memory_order_relaxed);
                    }
                    HitRecord hwRec = make_empty_hit_record();
                    HitRecord swRec = make_empty_hit_record();
                    bool hwHit = trace_scene_hardware(uniforms,
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
                                                      /*stats=*/nullptr,
                                                      ray,
                                                      kEpsilon,
                                                      kInfinity,
                                                      /*anyHitOnly=*/false,
                                                      excludeMeshIndex,
                                                      excludePrimitiveIndex,
                                                      hwRec);
                    bool swHit = trace_scene_software_with_exclusion(uniforms,
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
                                                                     /*stats=*/nullptr,
                                                                     ray,
                                                                     kEpsilon,
                                                                     kInfinity,
                                                                     excludeMeshIndex,
                                                                     excludePrimitiveIndex,
                                                                     swRec);
                    uint reasonMask = 0u;
                    if (hwHit != swHit) {
                        reasonMask |= kParityReasonHitMiss;
                    }
                    if (hwHit && swHit) {
                        float epsT = max(1.0e-3f, 1.0e-4f * fabs(hwRec.t));
                        float tDiff = fabs(hwRec.t - swRec.t);
                        if (tDiff > epsT) {
                            reasonMask |= kParityReasonT;
                        }
                        float3 hwN = hwRec.normal;
                        float3 swN = swRec.normal;
                        if (all(isfinite(hwN)) && all(isfinite(swN)) &&
                            dot(hwN, hwN) > 0.0f && dot(swN, swN) > 0.0f) {
                            float nDot = dot(normalize(hwN), normalize(swN));
                            if (nDot < 0.99f) {
                                reasonMask |= kParityReasonNormal;
                            }
                        }
                        float3 hwShadingN = hwRec.shadingNormal;
                        float3 swShadingN = swRec.shadingNormal;
                        if (all(isfinite(hwShadingN)) && all(isfinite(swShadingN)) &&
                            dot(hwShadingN, hwShadingN) > 0.0f &&
                            dot(swShadingN, swShadingN) > 0.0f) {
                            float shadingNDot = dot(normalize(hwShadingN), normalize(swShadingN));
                            if (shadingNDot < 0.99f) {
                                reasonMask |= kParityReasonShadingNormal;
                            }
                        }
                        if (hwRec.frontFace != swRec.frontFace) {
                            reasonMask |= kParityReasonFrontFace;
                        }
                        if (hwRec.materialIndex != swRec.materialIndex ||
                            hwRec.meshIndex != swRec.meshIndex ||
                            hwRec.primitiveIndex != swRec.primitiveIndex) {
                            reasonMask |= kParityReasonId;
                        }
                    }
                    record_parity_entry(*debugContext,
                                        uniforms,
                                        depth,
                                        kParityProbeMain,
                                        ray,
                                        kEpsilon,
                                        kInfinity,
                                        hwHit,
                                        hwRec,
                                        swHit,
                                        swRec,
                                        reasonMask);
                }
            }
        }
#endif

        bool hit = false;
#if PT_DEBUG_TOOLS
        if (forceSoftware) {
            hit = trace_scene_software(uniforms,
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
            if (hit && stats) {
                atomic_fetch_add_explicit(&stats->hardwareFallbackHitCount,
                                          1u,
                                          memory_order_relaxed);
            }
        } else if (depth == 0u && enableFirstHitFallback && softwareTrianglesAvailable) {
            hit = trace_scene_software(uniforms,
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
            if (hit && stats) {
                atomic_fetch_add_explicit(&stats->hardwareFirstHitFallbackCount,
                                          1u,
                                          memory_order_relaxed);
            }
        }

        if (!hit && preferSoftwareForMedium && softwareTrianglesAvailable) {
            hit = trace_scene_software(uniforms,
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
            if (hit && stats) {
                if (depth == 0u) {
                    atomic_fetch_add_explicit(&stats->hardwareFirstHitFallbackCount,
                                              1u,
                                              memory_order_relaxed);
                } else {
                    atomic_fetch_add_explicit(&stats->hardwareFallbackHitCount,
                                              1u,
                                              memory_order_relaxed);
                }
            }
        }
#endif

        if (!hit && !forceSoftware) {
            hit = trace_scene_hardware(uniforms,
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
                                       excludeMeshIndex,
                                       excludePrimitiveIndex,
                                       rec);
        }

        if (!hit && !forceSoftware && enableMissFallback && softwareTrianglesAvailable) {
#if PT_DEBUG_TOOLS
            hit = trace_scene_software(uniforms,
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
            if (hit && stats) {
                atomic_fetch_add_explicit(&stats->hardwareFallbackHitCount,
                                          1u,
                                          memory_order_relaxed);
            }
#endif
        }

        if (!hit) {
            if (uniforms.debugViewMode != kDebugViewNone) {
                return float3(0.0f);
            }
            float3 background = sky_color(ray.direction);
            if (uniforms.backgroundMode == 1u) {
                background = uniforms.backgroundColor;
            } else if (uniforms.backgroundMode == 2u && environmentTexture.get_width() > 0 && environmentTexture.get_height() > 0) {
                float overrideLod = 0.0f;
                bool useOverride = environment_mip_override(uniforms, environmentTexture, overrideLod);
                if (useOverride) {
                    background = environment_color_lod(environmentTexture,
                                                       ray.direction,
                                                       uniforms.environmentRotation,
                                                       uniforms.environmentIntensity,
                                                       overrideLod,
                                                       uniforms);
                } else if (envLodActive) {
                    background = environment_color_lod(environmentTexture,
                                                       ray.direction,
                                                       uniforms.environmentRotation,
                                                       uniforms.environmentIntensity,
                                                       envLod,
                                                       uniforms);
                } else {
                    background = environment_color(environmentTexture,
                                                   ray.direction,
                                                   uniforms.environmentRotation,
                                                   uniforms.environmentIntensity,
                                                   uniforms);
                }
            }
            if (uniforms.backgroundMode != 2u) {
                background = to_working_space(background, uniforms);
            }
            if (debugContext) {
                record_debug_event(*debugContext,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   /*frontFace=*/0u,
                                   kInvalidIndex,
                                   /*scatterIsDelta=*/false,
                                   throughput);
            }
            float misWeight = 1.0f;
            bool useSpecularMis =
                use_visible_emitter_mis(depth, lastScatterWasDelta, uniforms);
            if (useSpecularMis && envSampling) {
                float lightPdf = environment_pdf(uniforms, environmentPdf, ray.direction);
                float denom = lastBsdfPdf + lightPdf;
                if (denom > 0.0f) {
                    misWeight = clamp(lastBsdfPdf / denom,
                                      kMisWeightClampMin,
                                      kMisWeightClampMax);
                }
            }
            float3 contribution = background * misWeight;
            radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
            if (debugContext) {
                record_debug_event(*debugContext,
                                   kDebugEventBackground,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   /*frontFace=*/0u,
                                   kInvalidIndex,
                                   /*scatterIsDelta=*/false,
                                   throughput,
                                   float4(misWeight, lastBsdfPdf, 0.0f, 0.0f),
                                   contribution);
            }
            break;
        }
        prevRec = rec;
        prevValid = true;

        if (!materials || uniforms.materialCount == 0) {
            break;
        }
        if (volume_transport_enabled(uniforms)) {
            MediumDescriptor medium = volume_make_global_medium(uniforms);
            const float segment = max(rec.t, 0.0f);
            const float3 segmentEmission = volume_emission_integral(medium, segment);
            if (any(segmentEmission > float3(0.0f))) {
                radiance += clamp_firefly_contribution(throughput, segmentEmission, clampParams);
            }
            const uint maxVolumeEvents = max(medium.metadata.y, 1u);
            bool sampledVolumeEvent = false;
            if (volumeScatteringDepth < maxVolumeEvents &&
                any(medium.sigmaSAnisotropy.xyz > float3(0.0f))) {
                float distancePdf = 1.0f;
                const float sampledDistance = volume_sample_distance(medium, state, distancePdf);
                sampledVolumeEvent = sampledDistance > 0.0f && sampledDistance < segment;
                if (sampledVolumeEvent) {
                    const float3 scatterPoint = ray.origin + ray.direction * sampledDistance;
                    const float3 transmittance = volume_transmittance(medium, sampledDistance, stats);
                    const float majorant = max(volume_majorant(medium), 1.0e-6f);
                    throughput *= transmittance * (medium.sigmaSAnisotropy.xyz / majorant);
                    if (rectLightCount > 0u && medium.metadata.z != 0u) {
                        HitRecord volumeRec = rec;
                        volumeRec.point = scatterPoint;
                        RectLightSample lightSample;
                        if (sample_rect_light(uniforms,
                                              rectangles,
                                              materials,
                                              environmentTexture,
                                              volumeRec,
                                              state,
                                              rectLightCount,
                                              lightSample)) {
                            Ray shadowRay;
                            shadowRay.origin = scatterPoint + lightSample.direction * kRayOriginEpsilon;
                            shadowRay.direction = lightSample.direction;
                            HitRecord shadowRec;
                            const float shadowMax = max(lightSample.distance - kEpsilon, kEpsilon);
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
                                                                 nullptr,
                                                                 shadowRec);
                            if (!occluded && lightSample.pdf > 0.0f) {
                                const float phase = volume_phase_eval(medium, -ray.direction, lightSample.direction);
                                const float3 shadowTr = volume_transmittance(medium, shadowMax, stats);
                                const float3 contribution =
                                    lightSample.emission * phase * shadowTr / max(lightSample.pdf, 1.0e-6f);
                                radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                                if (stats) {
                                    atomic_fetch_add_explicit(&stats->volumeNeeRayCount, 1u, memory_order_relaxed);
                                }
                            }
                        }
                    }
                    float phasePdf = 1.0f;
                    const float3 phaseDir = volume_phase_sample(medium, -ray.direction, state, phasePdf);
                    if (stats) {
                        atomic_fetch_add_explicit(&stats->volumeScatterEventCount, 1u, memory_order_relaxed);
                        atomic_fetch_add_explicit(&stats->volumePhaseSampleCount, 1u, memory_order_relaxed);
                    }
                    volumeScatteringDepth += 1u;
                    lastScatterWasDelta = false;
                    lastBsdfPdf = max(phasePdf, 1.0e-6f);
                    ray.origin = scatterPoint + phaseDir * kRayOriginEpsilon;
                    ray.direction = phaseDir;
                    prevValid = false;
                    continue;
                }
            } else if (any(medium.sigmaSAnisotropy.xyz > float3(0.0f)) && stats) {
                atomic_fetch_add_explicit(&stats->volumeFallbackCount, 1u, memory_order_relaxed);
            }
            throughput *= volume_transmittance(medium, segment, stats);
        }
        if (mediumDepth > 0u) {
            float3 sigma = mediumSigmaStack[mediumDepth - 1u];
            if (any(sigma > float3(0.0f))) {
                float segment = max(rec.t, 0.0f);
                float3 attenuation = exp(-sigma * segment);
                throughput *= attenuation;
            }
        }
        uint matIndex = min(rec.materialIndex, uniforms.materialCount - 1);
        MaterialData material = materials[matIndex];
        spectral_apply_material(material, uniforms, spectralState, stats);
        uint type = static_cast<uint>(material.typeEta.x);
        float3 incidentDir = normalize(ray.direction);
        float3 wo = -incidentDir;
        float hitDistanceWorld = ray_segment_world_length(ray, rec.t);
        bool surfaceIsDelta = material_is_delta(material);
        bool specularOnly = (uniforms.debugSpecularOnly != 0u);
        float diffuseOcclusion = 1.0f;
        float3 debugBaseColor = material_base_color(material);
        float2 debugBaseColorUv = float2(0.0f);
        float debugBaseColorLod = 0.0f;
        float debugMetallic = 0.0f;
        float debugRoughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        float debugAO = 1.0f;
        float3 shadingNormal = rec.shadingNormal;
        float3 debugVtxNormalRaw = float3(0.0f);
        float3 debugVtxNormal = float3(0.0f);
        if (!all(isfinite(shadingNormal)) || dot(shadingNormal, shadingNormal) <= 0.0f) {
            shadingNormal = rec.normal;
        }
        if (rec.primitiveType == kPrimitiveTypeTriangle &&
            meshInfos && sceneVertices && meshIndices && uniforms.meshCount > 0u) {
            float3 candidateRaw = interpolate_shading_normal_raw(uniforms,
                                                                 rec.meshIndex,
                                                                 rec.primitiveIndex,
                                                                 rec.barycentric,
                                                                 meshInfos,
                                                                 sceneVertices,
                                                                 meshIndices);
            float3 candidate = interpolate_shading_normal(uniforms,
                                                          rec.meshIndex,
                                                          rec.primitiveIndex,
                                                          rec.barycentric,
                                                          meshInfos,
                                                          sceneVertices,
                                                          meshIndices);
            if (all(isfinite(candidate)) && dot(candidate, candidate) > 0.0f) {
                debugVtxNormalRaw = candidateRaw;
                if (dot(candidate, rec.normal) < 0.0f) {
                    candidate = -candidate;
                }
                shadingNormal = normalize(candidate);
                debugVtxNormal = shadingNormal;
            }
        }
        if (type == 2u) { // Dielectric: force geometric normal for shading.
            float3 geomNormal = rec.normal;
            if (all(isfinite(geomNormal)) && dot(geomNormal, geomNormal) > 0.0f) {
                shadingNormal = geomNormal;
            }
            // Keep ray offsets consistent between SWRT/HWRT for glass.
            rec.shadingNormal = shadingNormal;
        }

        if (type != 7u &&
            rec.primitiveType == kPrimitiveTypeTriangle &&
            meshInfos && sceneVertices && meshIndices && uniforms.meshCount > 0u &&
            material_texture_valid(uniforms, material.textureIndices0.x)) {
            float2 uv0 = interpolate_uv(uniforms,
                                        rec.meshIndex,
                                        rec.primitiveIndex,
                                        rec.barycentric,
                                        0u,
                                        meshInfos,
                                        sceneVertices,
                                        meshIndices);
            float2 uv1 = interpolate_uv(uniforms,
                                        rec.meshIndex,
                                        rec.primitiveIndex,
                                        rec.barycentric,
                                        1u,
                                        meshInfos,
                                        sceneVertices,
                                        meshIndices);
            PbrTextureSamplingContext baseColorCtx = make_pbr_texture_sampling_context(material,
                                                                                       kPbrTextureSlotBaseColor,
                                                                                       uv0,
                                                                                       uv1,
                                                                                       false,
                                                                                       float2(0.0f),
                                                                                       float2(0.0f),
                                                                                       0.0f,
                                                                                       false,
                                                                                       float2(0.0f),
                                                                                       float2(0.0f),
                                                                                       0.0f);
            float4 baseColorSample =
                sample_material_texture_filtered(materialTextures,
                                                materialSamplers,
                                                materialTextureInfos,
                                                uniforms,
                                                material.textureIndices0.x,
                                                baseColorCtx.uv,
                                                float4(1.0f),
                                                0.0f,
                                                false,
                                                float2(0.0f),
                                                float2(0.0f));
            float3 baseFactor = to_working_space(material.baseColorRoughness.xyz, uniforms);
            float3 baseColorSampleRgb = to_working_space(baseColorSample.xyz, uniforms);
            material.baseColorRoughness.xyz = baseFactor * baseColorSampleRgb;
            debugBaseColor = material.baseColorRoughness.xyz;
            debugBaseColorUv = baseColorCtx.uv;
            debugBaseColorLod = 0.0f;
        }

        if (type == 7u && rec.primitiveType == kPrimitiveTypeTriangle &&
            meshInfos && sceneVertices && meshIndices && uniforms.meshCount > 0u) {
            float2 uv0 = interpolate_uv(uniforms,
                                        rec.meshIndex,
                                        rec.primitiveIndex,
                                        rec.barycentric,
                                        0u,
                                        meshInfos,
                                        sceneVertices,
                                        meshIndices);
            float2 uv1 = interpolate_uv(uniforms,
                                        rec.meshIndex,
                                        rec.primitiveIndex,
                                        rec.barycentric,
                                        1u,
                                        meshInfos,
                                        sceneVertices,
                                        meshIndices);
            float4 tangent = interpolate_tangent(uniforms,
                                                 rec.meshIndex,
                                                 rec.primitiveIndex,
                                                 rec.barycentric,
                                                 meshInfos,
                                                 sceneVertices,
                                                 meshIndices);
            float4 tangentRaw4 = interpolate_tangent_raw(uniforms,
                                                         rec.meshIndex,
                                                         rec.primitiveIndex,
                                                         rec.barycentric,
                                                         meshInfos,
                                                         sceneVertices,
                                                         meshIndices);
            float3 debugBaryWeights = barycentric_weights_saturated(rec.barycentric);
            float3 debugTriIndices = float3(0.0f);
            float2 debugNormalUv = float2(0.0f);
            float3 debugTangentRaw = tangentRaw4.xyz;
            float debugTangentW = tangentRaw4.w;
            float3 debugTangent = float3(0.0f);
            float3 debugBitangent = float3(0.0f);
            float3 debugTexelRaw = float3(0.5f, 0.5f, 1.0f);
            float3 debugTexelDecoded = float3(0.0f, 0.0f, 1.0f);
            float3 debugSnTangentSpace = float3(0.0f, 0.0f, 1.0f);
            float3 debugSnWorld = shadingNormal;
            if (material.typeEta.z > 0.5f) {
                rec.twoSided = 1u;
            }
            if (meshInfos && meshIndices && uniforms.meshCount > 0u) {
                uint clampedMesh = min(rec.meshIndex, uniforms.meshCount - 1u);
                MeshInfo info = meshInfos[clampedMesh];
                if (rec.primitiveIndex >= info.triangleOffset) {
                    uint localIndex = rec.primitiveIndex - info.triangleOffset;
                    if (localIndex < info.indexCount) {
                        uint indexEntry = info.indexOffset + localIndex;
                        uint3 triIndices = meshIndices[indexEntry];
                        debugTriIndices = float3(float(triIndices.x),
                                                 float(triIndices.y),
                                                 float(triIndices.z));
                    }
                }
            }
            float coneFootprintWorld = ray_cone_width_at_distance(rayCone, hitDistanceWorld);
            float surfaceFootprintWorld =
                surface_footprint_from_cone(coneFootprintWorld, rec.normal, wo);
            float3 dPdu0 = float3(0.0f);
            float3 dPdv0 = float3(0.0f);
            float uvPerWorld0 = 0.0f;
            bool hasSurfacePartials0 = triangle_surface_partials(uniforms,
                                                                 rec.meshIndex,
                                                                 rec.primitiveIndex,
                                                                 0u,
                                                                 meshInfos,
                                                                 sceneVertices,
                                                                 meshIndices,
                                                                 dPdu0,
                                                                 dPdv0,
                                                                 uvPerWorld0);
            float3 dPdu1 = float3(0.0f);
            float3 dPdv1 = float3(0.0f);
            float uvPerWorld1 = 0.0f;
            bool hasSurfacePartials1 = triangle_surface_partials(uniforms,
                                                                 rec.meshIndex,
                                                                 rec.primitiveIndex,
                                                                 1u,
                                                                 meshInfos,
                                                                 sceneVertices,
                                                                 meshIndices,
                                                                 dPdu1,
                                                                 dPdv1,
                                                                 uvPerWorld1);
            float2 dUVdx0 = float2(0.0f);
            float2 dUVdy0 = float2(0.0f);
            bool hasIgehyGradients0 = false;
            if (depth == 0u && hasSurfacePartials0) {
                float3 dudP = float3(0.0f);
                float3 dvdP = float3(0.0f);
                if (uv_world_gradients_from_partials(dPdu0, dPdv0, dudP, dvdP)) {
                    hasIgehyGradients0 = first_hit_uv_gradients_igehy(ray,
                                                                      primaryRayDiff,
                                                                      rec.t,
                                                                      rec.normal,
                                                                      dudP,
                                                                      dvdP,
                                                                      dUVdx0,
                                                                      dUVdy0);
                }
            }
            float2 dUVdx1 = float2(0.0f);
            float2 dUVdy1 = float2(0.0f);
            bool hasIgehyGradients1 = false;
            if (depth == 0u && hasSurfacePartials1) {
                float3 dudP = float3(0.0f);
                float3 dvdP = float3(0.0f);
                if (uv_world_gradients_from_partials(dPdu1, dPdv1, dudP, dvdP)) {
                    hasIgehyGradients1 = first_hit_uv_gradients_igehy(ray,
                                                                      primaryRayDiff,
                                                                      rec.t,
                                                                      rec.normal,
                                                                      dudP,
                                                                      dvdP,
                                                                      dUVdx1,
                                                                      dUVdy1);
                }
            }

            PbrTextureSamplingContext baseColorCtx = make_pbr_texture_sampling_context(material,
                                                                                        kPbrTextureSlotBaseColor,
                                                                                        uv0,
                                                                                        uv1,
                                                                                        hasIgehyGradients0,
                                                                                        dUVdx0,
                                                                                        dUVdy0,
                                                                                        uvPerWorld0,
                                                                                        hasIgehyGradients1,
                                                                                        dUVdx1,
                                                                                        dUVdy1,
                                                                                        uvPerWorld1);
            PbrTextureSamplingContext ormCtx = make_pbr_texture_sampling_context(material,
                                                                                  kPbrTextureSlotMetallicRoughness,
                                                                                  uv0,
                                                                                  uv1,
                                                                                  hasIgehyGradients0,
                                                                                  dUVdx0,
                                                                                  dUVdy0,
                                                                                  uvPerWorld0,
                                                                                  hasIgehyGradients1,
                                                                                  dUVdx1,
                                                                                  dUVdy1,
                                                                                  uvPerWorld1);
            PbrTextureSamplingContext normalCtx = make_pbr_texture_sampling_context(material,
                                                                                     kPbrTextureSlotNormal,
                                                                                     uv0,
                                                                                     uv1,
                                                                                     hasIgehyGradients0,
                                                                                     dUVdx0,
                                                                                     dUVdy0,
                                                                                     uvPerWorld0,
                                                                                     hasIgehyGradients1,
                                                                                     dUVdx1,
                                                                                     dUVdy1,
                                                                                     uvPerWorld1);
            debugNormalUv = normalCtx.uv;
            PbrTextureSamplingContext occlusionCtx = make_pbr_texture_sampling_context(material,
                                                                                        kPbrTextureSlotOcclusion,
                                                                                        uv0,
                                                                                        uv1,
                                                                                        hasIgehyGradients0,
                                                                                        dUVdx0,
                                                                                        dUVdy0,
                                                                                        uvPerWorld0,
                                                                                        hasIgehyGradients1,
                                                                                        dUVdx1,
                                                                                        dUVdy1,
                                                                                        uvPerWorld1);
            PbrTextureSamplingContext emissiveCtx = make_pbr_texture_sampling_context(material,
                                                                                       kPbrTextureSlotEmissive,
                                                                                       uv0,
                                                                                       uv1,
                                                                                       hasIgehyGradients0,
                                                                                       dUVdx0,
                                                                                       dUVdy0,
                                                                                       uvPerWorld0,
                                                                                       hasIgehyGradients1,
                                                                                       dUVdx1,
                                                                                       dUVdy1,
                                                                                       uvPerWorld1);
            PbrTextureSamplingContext transmissionCtx = make_pbr_texture_sampling_context(material,
                                                                                           kPbrTextureSlotTransmission,
                                                                                           uv0,
                                                                                           uv1,
                                                                                           hasIgehyGradients0,
                                                                                           dUVdx0,
                                                                                           dUVdy0,
                                                                                           uvPerWorld0,
                                                                                           hasIgehyGradients1,
                                                                                           dUVdx1,
                                                                                           dUVdy1,
                                                                                           uvPerWorld1);
            float3 baseFactor = to_working_space(material.baseColorRoughness.xyz, uniforms);
            float baseColorLod =
                material_texture_lod_with_fallback(materialTextures,
                                                   materialTextureInfos,
                                                   uniforms,
                                                   material.textureIndices0.x,
                                                   baseColorCtx.hasIgehyGradients,
                                                   baseColorCtx.dUVdx,
                                                   baseColorCtx.dUVdy,
                                                   baseColorCtx.uvPerWorld,
                                                   surfaceFootprintWorld);
            if ((material.materialFlags & kMaterialFlagForceBaseColorMip0) != 0u) {
                baseColorLod = 0.0f;
            }
            float4 baseColorSample =
                sample_material_texture_filtered(materialTextures,
                                                materialSamplers,
                                                materialTextureInfos,
                                                uniforms,
                                                material.textureIndices0.x,
                                                baseColorCtx.uv,
                                                float4(1.0f),
                                                baseColorLod,
                                                baseColorCtx.hasIgehyGradients,
                                                baseColorCtx.dUVdx,
                                                baseColorCtx.dUVdy);
            if ((material.materialFlags & kMaterialFlagBlackKeyAlphaFromRgb) != 0u) {
                // MASTER_Side_Letters atlas carries dark plaque texels in RGB.
                // Gate alpha by luminance so only bright lettering survives.
                float luma = dot(baseColorSample.xyz, kLuminanceWeights);
                float alphaFromRgb = smoothstep(0.72f, 0.90f, luma);
                baseColorSample.w = alphaFromRgb;
            }
            float3 baseColorSampleRgb = to_working_space(baseColorSample.xyz, uniforms);
            float3 baseColor = baseFactor * baseColorSampleRgb;
            debugBaseColorUv = baseColorCtx.uv;
            debugBaseColorLod = baseColorLod;

            float metallic = clamp(material.pbrParams.x, 0.0f, 1.0f);
            float roughness = clamp(material.pbrParams.y, 0.0f, 1.0f);
            float normalStrengthScale = 1.0f;
#if PT_DEBUG_TOOLS
            normalStrengthScale = max(uniforms.debugNormalStrengthScale, 0.0f);
#endif
            float normalScale = material.pbrParams.w * normalStrengthScale;
            bool disableOrmByMaterial = (material.materialFlags & kMaterialFlagDisableOrm) != 0u;
            bool useOrmTexture = !disableOrmByMaterial &&
                                 material_texture_valid(uniforms, material.textureIndices0.y);
#if PT_DEBUG_TOOLS
            useOrmTexture = useOrmTexture && (uniforms.debugDisableOrmTexture == 0u);
#endif
            if (useOrmTexture) {
                float ormLod =
                    material_texture_lod_with_fallback(materialTextures,
                                                       materialTextureInfos,
                                                       uniforms,
                                                       material.textureIndices0.y,
                                                       ormCtx.hasIgehyGradients,
                                                       ormCtx.dUVdx,
                                                       ormCtx.dUVdy,
                                                       ormCtx.uvPerWorld,
                                                       surfaceFootprintWorld);
#if PT_DEBUG_TOOLS
                ormLod = max(ormLod + uniforms.debugOrmLodBias, 0.0f);
#endif
                float3 mrSample =
                    sample_material_texture_level(materialTextures,
                                                  materialSamplers,
                                                  materialTextureInfos,
                                                  uniforms,
                                                  material.textureIndices0.y,
                                                  ormCtx.uv,
                                                  float4(1.0f),
                                                  ormLod).xyz;
                metallic = clamp(mrSample.z * metallic, 0.0f, 1.0f);
                roughness = clamp(mrSample.y * roughness, 0.0f, 1.0f);
            }
            float visorMask = visor_override_blend(baseColor, metallic, roughness, matIndex, uniforms);
            if (visorMask > 0.0f) {
                    float overrideRoughness =
                        clamp(uniforms.debugVisorOverrideRoughness, 0.0f, 1.0f);
                    float overrideF0 = clamp(uniforms.debugVisorOverrideF0, 0.0f, 0.12f);
                    metallic = mix(metallic, 0.0f, visorMask);
                    roughness = mix(roughness, overrideRoughness, visorMask);
                    material.typeEta.y = mix(material.typeEta.y,
                                             ior_from_f0(overrideF0),
                                             visorMask);
            }
            float normalLod =
                material_texture_lod_with_fallback(materialTextures,
                                                   materialTextureInfos,
                                                   uniforms,
                                                   material.textureIndices0.z,
                                                   normalCtx.hasIgehyGradients,
                                                   normalCtx.dUVdx,
                                                   normalCtx.dUVdy,
                                                   normalCtx.uvPerWorld,
                                                   surfaceFootprintWorld);
#if PT_DEBUG_TOOLS
            normalLod = max(normalLod + uniforms.debugNormalLodBias, 0.0f);
#endif

            float transmission = clamp(material.pbrExtras.z, 0.0f, 1.0f);
            if (material_texture_valid(uniforms, material.textureIndices1.y)) {
                float transmissionLod =
                    material_texture_lod_with_fallback(materialTextures,
                                                       materialTextureInfos,
                                                       uniforms,
                                                       material.textureIndices1.y,
                                                       transmissionCtx.hasIgehyGradients,
                                                       transmissionCtx.dUVdx,
                                                       transmissionCtx.dUVdy,
                                                       transmissionCtx.uvPerWorld,
                                                       surfaceFootprintWorld);
                float transmissionSample =
                    sample_material_texture_filtered(materialTextures,
                                                    materialSamplers,
                                                    materialTextureInfos,
                                                    uniforms,
                                                    material.textureIndices1.y,
                                                    transmissionCtx.uv,
                                                    float4(1.0f),
                                                    transmissionLod,
                                                    transmissionCtx.hasIgehyGradients,
                                                    transmissionCtx.dUVdx,
                                                    transmissionCtx.dUVdy).x;
                transmission = clamp(transmission * transmissionSample, 0.0f, 1.0f);
            }
            transmission *= (1.0f - metallic);

            float alpha = clamp(material.pbrExtras.x, 0.0f, 1.0f);
            alpha = clamp(alpha * baseColorSample.w, 0.0f, 1.0f);
            float alphaCutoff = clamp(material.pbrExtras.y, 0.0f, 1.0f);
            float alphaMode = material.pbrExtras.w;
            if (alphaMode > 0.5f) {
                bool discard = false;
                if (alphaMode < 1.5f) {
                    discard = alpha < alphaCutoff;
                } else {
                    discard = rand_uniform(state) > alpha;
                }
                if (discard) {
                    ray.origin = offset_ray_origin(rec, ray.direction);
                    prevRec = rec;
                    prevValid = true;
                    lastBsdfPdf = 1.0f;
                    lastScatterWasDelta = true;
                    specularDepth += 1u;
                    continue;
                }
            }

            material.pbrExtras.z = transmission;

            float occlusion = 1.0f;
            if (!disableOrmByMaterial && material_texture_valid(uniforms, material.textureIndices0.w)) {
                float occlusionLod =
                    material_texture_lod_with_fallback(materialTextures,
                                                       materialTextureInfos,
                                                       uniforms,
                                                       material.textureIndices0.w,
                                                       occlusionCtx.hasIgehyGradients,
                                                       occlusionCtx.dUVdx,
                                                       occlusionCtx.dUVdy,
                                                       occlusionCtx.uvPerWorld,
                                                       surfaceFootprintWorld);
                float occSample =
                    sample_material_texture_filtered(materialTextures,
                                                    materialSamplers,
                                                    materialTextureInfos,
                                                    uniforms,
                                                    material.textureIndices0.w,
                                                    occlusionCtx.uv,
                                                    float4(1.0f),
                                                    occlusionLod,
                                                    occlusionCtx.hasIgehyGradients,
                                                    occlusionCtx.dUVdx,
                                                    occlusionCtx.dUVdy).x;
                occlusion = mix(1.0f, occSample, clamp(material.pbrParams.z, 0.0f, 1.0f));
            }
            debugAO = occlusion;
            diffuseOcclusion = (uniforms.debugDisableAO != 0u) ? 1.0f : occlusion;
            if (uniforms.debugAoIndirectOnly != 0u && depth == 0u) {
                diffuseOcclusion = 1.0f;
            }
            debugBaseColor = baseColor;
            debugMetallic = metallic;
            debugRoughness = roughness;

            float3 emissive = to_working_space(material.emission.xyz, uniforms);
            if (material_texture_valid(uniforms, material.textureIndices1.x)) {
                float emissiveLod =
                    material_texture_lod_with_fallback(materialTextures,
                                                       materialTextureInfos,
                                                       uniforms,
                                                       material.textureIndices1.x,
                                                       emissiveCtx.hasIgehyGradients,
                                                       emissiveCtx.dUVdx,
                                                       emissiveCtx.dUVdy,
                                                       emissiveCtx.uvPerWorld,
                                                       surfaceFootprintWorld);
                float3 emissiveSample =
                    sample_material_texture_filtered(materialTextures,
                                                    materialSamplers,
                                                    materialTextureInfos,
                                                    uniforms,
                                                    material.textureIndices1.x,
                                                    emissiveCtx.uv,
                                                    float4(1.0f),
                                                    emissiveLod,
                                                    emissiveCtx.hasIgehyGradients,
                                                    emissiveCtx.dUVdx,
                                                    emissiveCtx.dUVdy).xyz;
                emissiveSample = to_working_space(emissiveSample, uniforms);
                emissive *= emissiveSample;
            }

            bool useNormalMap = material_texture_valid(uniforms, material.textureIndices0.z);
#if PT_DEBUG_TOOLS
            if (uniforms.debugDisableNormalMap != 0u) {
                useNormalMap = false;
            }
#endif
            if (normalScale <= 1.0e-4f) {
                useNormalMap = false;
            }
            float normalLength = 1.0f;
            float3 normalSampleTs = float3(0.0f, 0.0f, 1.0f);
            bool flipNormalGreen = false;
#if PT_DEBUG_TOOLS
            flipNormalGreen = uniforms.debugFlipNormalGreen != 0u;
#endif
            if (useNormalMap) {
                debugTexelRaw =
                    sample_material_texture_filtered(materialTextures,
                                                    materialSamplers,
                                                    materialTextureInfos,
                                                    uniforms,
                                                    material.textureIndices0.z,
                                                    normalCtx.uv,
                                                    float4(0.5f, 0.5f, 1.0f, 1.0f),
                                                    normalLod,
                                                    normalCtx.hasIgehyGradients,
                                                    normalCtx.dUVdx,
                                                    normalCtx.dUVdy).xyz;
                normalSampleTs = decode_normal_map(debugTexelRaw,
                                                   normalScale,
                                                   flipNormalGreen,
                                                   normalLength);
                debugTexelDecoded = normalSampleTs;
                debugSnTangentSpace = normalSampleTs;
                float3 t = tangent.xyz;
                float3 b = float3(0.0f);
                bool hasBasis = false;
                bool trustVertexTangent = fabs(tangent.w) > 0.5f;
                if (trustVertexTangent && all(isfinite(t)) && dot(t, t) > 1.0e-6f) {
                    t = normalize(t - shadingNormal * dot(shadingNormal, t));
                    if (all(isfinite(t)) && dot(t, t) > 1.0e-6f) {
                        float tangentSign = (tangent.w < 0.0f) ? -1.0f : 1.0f;
                        b = normalize(cross(shadingNormal, t)) * tangentSign;
                        if (all(isfinite(b)) && dot(b, b) > 1.0e-6f) {
                            hasBasis = true;
                        }
                    }
                }
                if (!hasBasis) {
                    uint normalUvSet = pbr_texture_uv_set(material, kPbrTextureSlotNormal);
                    hasBasis = compute_tangent_basis_from_uv(uniforms,
                                                             rec.meshIndex,
                                                             rec.primitiveIndex,
                                                             normalUvSet,
                                                             meshInfos,
                                                             sceneVertices,
                                                             meshIndices,
                                                             shadingNormal,
                                                             t,
                                                             b);
                }
                if (!hasBasis) {
                    build_onb(shadingNormal, t, b);
                }
                float3 mapped = normalize(t * normalSampleTs.x +
                                          b * normalSampleTs.y +
                                          shadingNormal * normalSampleTs.z);
                if (dot(mapped, rec.normal) < 0.0f) {
                    mapped = -mapped;
                }
                debugTangent = t;
                debugBitangent = b;
                shadingNormal = mapped;
                debugSnWorld = shadingNormal;
            }

            if (useNormalMap) {
                float tok = max((1.0f - normalLength) / max(normalLength, 1.0e-6f), 0.0f);
                if (normalCtx.hasIgehyGradients &&
                    all(isfinite(normalCtx.dUVdx)) &&
                    all(isfinite(normalCtx.dUVdy))) {
                    float gradMag = max(max(fabs(normalCtx.dUVdx.x), fabs(normalCtx.dUVdx.y)),
                                        max(fabs(normalCtx.dUVdy.x), fabs(normalCtx.dUVdy.y)));
                    if (gradMag > 1.0e-6f && gradMag < 4.0f) {
                        float3 nDx = sample_material_texture_level(materialTextures,
                                                                   materialSamplers,
                                                                   materialTextureInfos,
                                                                   uniforms,
                                                                   material.textureIndices0.z,
                                                                   normalCtx.uv + normalCtx.dUVdx,
                                                                   float4(0.5f, 0.5f, 1.0f, 1.0f),
                                                                   normalLod).xyz;
                        float3 nDy = sample_material_texture_level(materialTextures,
                                                                   materialSamplers,
                                                                   materialTextureInfos,
                                                                   uniforms,
                                                                   material.textureIndices0.z,
                                                                   normalCtx.uv + normalCtx.dUVdy,
                                                                   float4(0.5f, 0.5f, 1.0f, 1.0f),
                                                                   normalLod).xyz;
                        float tmpLenDx = 1.0f;
                        float tmpLenDy = 1.0f;
                        nDx = decode_normal_map(nDx, normalScale, flipNormalGreen, tmpLenDx);
                        nDy = decode_normal_map(nDy, normalScale, flipNormalGreen, tmpLenDy);
                        float varianceX = max(1.0f - dot(normalSampleTs, nDx), 0.0f);
                        float varianceY = max(1.0f - dot(normalSampleTs, nDy), 0.0f);
                        float normalVariance = max(varianceX, varianceY);
                        tok += 0.35f * normalVariance;
                    }
                }
                roughness = clamp(sqrt(roughness * roughness + tok), 0.0f, 1.0f);
            }

            material.baseColorRoughness = float4(baseColor, roughness);
            material.pbrParams.x = metallic;
            material.emission = float4(emissive, 0.0f);
            rec.shadingNormal = shadingNormal;
            if (debugContext) {
                record_debug_event(*debugContext,
                                   kDebugEventTbnBasis0,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   debugVtxNormalRaw,
                                   float4(debugNormalUv.x,
                                          debugNormalUv.y,
                                          debugTangentW,
                                          rec.t),
                                   debugVtxNormal);
                record_debug_event(*debugContext,
                                   kDebugEventTbnBasis1,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   debugTangentRaw,
                                   float4(debugTangent, 0.0f),
                                   debugBitangent);
                record_debug_event(*debugContext,
                                   kDebugEventTbnBasis2,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   debugTexelRaw,
                                   float4(debugTexelDecoded, 0.0f),
                                   debugSnTangentSpace);
                record_debug_event(*debugContext,
                                   kDebugEventTbnBasis3,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   float3(0.0f),
                                   float4(0.0f),
                                   debugSnWorld);
                record_debug_event(*debugContext,
                                   kDebugEventTbnBasis4,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   debugBaryWeights,
                                   float4(0.0f),
                                   debugTriIndices);
                record_debug_event(*debugContext,
                                   kDebugEventBsdfMaterial0,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   baseColor,
                                   float4(debugNormalUv.x,
                                          debugNormalUv.y,
                                          metallic,
                                          roughness),
                                   float3(normalScale,
                                          transmission,
                                          0.0f));
                record_debug_event(*debugContext,
                                   kDebugEventBsdfMaterial1,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   emissive,
                                   float4(alpha,
                                          alphaCutoff,
                                          alphaMode,
                                          material.baseColorRoughness.w),
                                   material.baseColorRoughness.xyz);
                record_debug_event(*debugContext,
                                   kDebugEventBsdfMaterial2,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   material.pbrExtras.xyz,
                                   material.pbrParams,
                                   float3(material.pbrExtras.w, 0.0f, 0.0f));
                record_debug_event(*debugContext,
                                   kDebugEventBsdfGradients,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   float3(normalCtx.dUVdx.x,
                                          normalCtx.dUVdx.y,
                                          normalCtx.dUVdy.x),
                                   float4(normalCtx.hasIgehyGradients ? 1.0f : 0.0f,
                                          normalCtx.dUVdy.y,
                                          normalCtx.uvPerWorld,
                                          normalLod),
                                   float3(0.0f));
            }
        }

        if (uniforms.debugViewMode != kDebugViewNone) {
            const float3 debugColor =
                restir_debug_view_color(uniforms,
                                        rec,
                                        material,
                                        pixelCoord,
                                        depth,
                                        pathGuidingStates,
                                        restirPtReservoirs,
                                        radianceCacheStates,
                                        debugBaseColor,
                                        debugMetallic,
                                        debugRoughness,
                                        debugAO);
            radiance = debugColor;
            break;
        }

        // Capture first hit AOVs (albedo and normal) for denoising
        if (isFirstHit) {
            isFirstHit = false;
            if (outFirstHitAlbedo != nullptr) {
                *outFirstHitAlbedo = material_closure_aov_albedo(material);
            }
            if (outFirstHitNormal != nullptr) {
                // Store the world-space normal of first hit
                *outFirstHitNormal = shadingNormal;
            }
            if (outFirstHitPosition != nullptr) {
                *outFirstHitPosition = float4(rec.point, 1.0f);
            }
            if (outFirstHitMaterial != nullptr) {
                *outFirstHitMaterial = material_closure_aov_features(material);
            }
        }

        Reservoir risAuditReservoir = make_empty_reservoir();
        float3 risAuditBsdfValue = float3(0.0f);
        float3 risAuditContribution = float3(0.0f);
        float risAuditDistance = 0.0f;
        float risAuditNDotL = 0.0f;
        float risAuditBsdfPdf = 0.0f;
        bool risAuditVisible = false;
        bool risAuditActive = false;
        bool spatialReuseAttempted = false;
        uint spatialNeighborTarget = 0u;
        uint spatialNeighborsConsidered = 0u;
        uint spatialNeighborsAccepted = 0u;
        uint spatialRejectedDepth = 0u;
        uint spatialRejectedNormal = 0u;
        uint spatialRejectedInvalid = 0u;
        int2 spatialWinnerOffset = int2(0, 0);
        float spatialLastMisWeight = 0.0f;
        float spatialLastMergeWeight = 0.0f;
        bool temporalReuseAttempted = false;
        bool temporalPreviousAvailable = false;
        bool temporalReuseAccepted = false;
        uint temporalRejectedDepth = 0u;
        uint temporalRejectedNormal = 0u;
        uint temporalRejectedInvalid = 0u;
        uint temporalPreviousPrimitiveIndex = kInvalidIndex;
        float temporalLastMisWeight = 0.0f;
        float temporalLastMergeWeight = 0.0f;
        bool worldReuseAttempted = false;
        bool worldReuseAccepted = false;
        int3 worldCell = int3(0);
        uint worldCellHash = 0u;
        uint worldCandidatesConsidered = 0u;
        uint worldCandidatesAccepted = 0u;
        uint worldRejectedDepth = 0u;
        uint worldRejectedNormal = 0u;
        uint worldRejectedInvalid = 0u;
        uint worldRejectedCell = 0u;
        uint worldCandidatePrimitiveIndex = kInvalidIndex;
        float worldLastMisWeight = 0.0f;
        float worldLastMergeWeight = 0.0f;
        bool cacheReuseAttempted = false;
        bool cacheStateAvailable = false;
        bool cacheReuseAccepted = false;
        bool cacheFallbackUsed = false;
        int3 cacheCell = int3(0);
        uint cacheCellHash = 0u;
        uint cacheCandidatesConsidered = 0u;
        uint cacheEntriesAvailable = 0u;
        uint cacheCandidatesAccepted = 0u;
        uint cacheRejectedDepth = 0u;
        uint cacheRejectedNormal = 0u;
        uint cacheRejectedInvalid = 0u;
        uint cacheRejectedCell = 0u;
        uint cacheCandidatePrimitiveIndex = kInvalidIndex;
        uint cacheSourceFrameIndex = 0u;
        float cacheLastMisWeight = 0.0f;
        float cacheLastMergeWeight = 0.0f;

        if (!specularOnly &&
            type == 7u &&
            any(material.emission.xyz != float3(0.0f)) &&
            (rec.frontFace != 0u || rec.twoSided != 0u)) {
            float3 visibleEmission = camera_visible_emission(material, depth);
            radiance += clamp_firefly_contribution(throughput, visibleEmission, clampParams);
            if (debugContext) {
                record_debug_event(*debugContext,
                                   kDebugEventVisibleEmitter,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   throughput,
                                   float4(rec.t, 0.0f, 0.0f, 0.0f),
                                   visibleEmission);
            }
        }

        if (type == 3u) {  // DiffuseLight
            if (specularOnly) {
                break;
            }
            float3 emission = material.emission.xyz;
            if (material.emission.w > 0.0f &&
                environmentTexture.get_width() > 0 &&
                environmentTexture.get_height() > 0 &&
                rec.frontFace != 0u) {
                float3 sampleDir = -shadingNormal;
                float3 envColor = environment_color(environmentTexture,
                                                    sampleDir,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity,
                                                    uniforms);
                emission *= envColor;
            }
            if (any(emission != float3(0.0f)) &&
                (rec.frontFace != 0u || rec.twoSided != 0u)) {
                float misWeight = 1.0f;
                bool useSpecularMis =
                    use_visible_emitter_mis(depth, lastScatterWasDelta, uniforms);
                if (useSpecularMis && rectLightCount > 0u) {
                    float lightPdf = rect_light_pdf_for_hit(uniforms,
                                                            rectangles,
                                                            materials,
                                                            rectLightCount,
                                                            rec,
                                                            ray.origin);
                    float denom = lastBsdfPdf + lightPdf;
                    if (denom > 0.0f) {
                        misWeight = clamp(lastBsdfPdf / denom,
                                          kMisWeightClampMin,
                                          kMisWeightClampMax);
                    }
                }
                float3 contribution = emission * misWeight;
                radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                if (debugContext) {
                    record_debug_event(*debugContext,
                                       kDebugEventVisibleEmitter,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       throughput,
                                       float4(misWeight, lastBsdfPdf, rec.t, 0.0f),
                                       contribution);
                }
            }
            break;
        }

        if (!surfaceIsDelta &&
            uniforms.directLightMode != kDirectLightModeLegacyRect &&
            uniforms.emissivePrimitiveCount > 0u) {
            if (uniforms.directLightMode == kDirectLightModeBaselineEmissive) {
                RISSamplePayload lightSample;
                if (sample_direct_light_baseline(uniforms,
                                                 emissivePrimitives,
                                                 rec,
                                                 state,
                                                 lightSample)) {
                    const bool lightSampleDirectional = ris_payload_is_directional(lightSample);
                    float distSq = ris_payload_distance_sq(rec, lightSample);
                    float distance = lightSampleDirectional ? 1.0e20f : length(lightSample.position - rec.point);
                    float3 omegaL = ris_payload_omega_l(rec, lightSample);
                    float cosThetaSurface = saturate(dot(shadingNormal, omegaL));
                    float cosThetaLight = lightSampleDirectional ? 1.0f : saturate(dot(lightSample.normal, -omegaL));
                    if (lightSample.pdf > 0.0f &&
                        distSq > 0.0f &&
                        cosThetaSurface > 0.0f &&
                        cosThetaLight > 0.0f) {
                        Ray shadowRay;
                        shadowRay.origin = offset_ray_origin(rec, omegaL);
                        shadowRay.direction = omegaL;
                        HitRecord shadowRec = make_empty_hit_record();
                        float shadowMax = max(distance - kEpsilon, kEpsilon);
                        uint shadowExcludeMesh;
                        uint shadowExcludePrim;
                        compute_exclusion_indices(rec, shadowExcludeMesh, shadowExcludePrim);
                        bool occluded = false;
#if PT_DEBUG_TOOLS
                        if (forceSoftware) {
                            occluded = trace_scene_software(uniforms,
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
                        } else {
#endif
                            occluded = trace_scene_hardware(uniforms,
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
                                                            shadowExcludeMesh,
                                                            shadowExcludePrim,
                                                            shadowRec);
                            if (!lightSampleDirectional &&
                                occluded &&
                                shadowRec.primitiveType == kPrimitiveTypeTriangle &&
                                shadowRec.primitiveIndex == lightSample.primitiveIndex) {
                                occluded = false;
                            }
                            if (!occluded) {
                                HitRecord swShadowRec = make_empty_hit_record();
                                bool swOccluded = trace_scene_software(uniforms,
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
                                                                       /*stats=*/nullptr,
                                                                       shadowRay,
                                                                       kEpsilon,
                                                                       shadowMax,
                                                                       /*anyHitOnly=*/true,
                                                                       /*includeTriangles=*/true,
                                                                       swShadowRec);
                                if (swOccluded &&
                                    (lightSampleDirectional ||
                                     !(swShadowRec.primitiveType == kPrimitiveTypeTriangle &&
                                       swShadowRec.primitiveIndex == lightSample.primitiveIndex))) {
                                    occluded = true;
                                    shadowRec = swShadowRec;
                                }
                            }
#if PT_DEBUG_TOOLS
                        }
#endif
                        if (!occluded) {
                            BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                                    rec.point,
                                                                    shadingNormal,
                                                                    wo,
                                                                    omegaL,
                                                                    clampParams,
                                                                    uniforms.sssMode,
                                                                    diffuseOcclusion,
                                                                    specularOnly);
                            if (!bsdfEval.isDelta && !bsdfEval.isBssrdf) {
                                float3 contribution = bsdfEval.value * lightSample.emission;
                                contribution *= cosThetaSurface / max(lightSample.pdf, 1.0e-6f);
                                if (all(isfinite(contribution))) {
                                    radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                                }
                            }
                        }
                    }
                }
            } else if (uniforms.directLightMode == kDirectLightModeRis ||
                       uniforms.directLightMode == kDirectLightModeRisSpatialReuse ||
                       uniforms.directLightMode == kDirectLightModeRisTemporalReuse ||
                       uniforms.directLightMode == kDirectLightModeRisWorldReuse ||
                       uniforms.directLightMode == kDirectLightModeRisRegirCache ||
                       uniforms.directLightMode == kDirectLightModeRestirDi ||
                       uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) {
                Reservoir reservoir = make_empty_reservoir();
                uint risTargetM = max(uniforms.risCandidateCount, 1u);
                if ((uniforms.directLightMode == kDirectLightModeRisRegirCache ||
                     uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
                    uniforms.frameIndex > 0u) {
                    risTargetM *= 2u;
                }
                build_ris_reservoir_for_hit(uniforms,
                                            emissivePrimitives,
                                            rec,
                                            material,
                                            shadingNormal,
                                            wo,
                                            clampParams,
                                            diffuseOcclusion,
                                            specularOnly,
                                            risTargetM,
                                            state,
                                            reservoir);

                if (uniforms.directLightMode == kDirectLightModeRisSpatialReuse &&
                    reservoir.valid &&
                    depth == 0u &&
                    rec.t > 0.0f &&
                    uniforms.width > 0u &&
                    uniforms.height > 0u) {
                    spatialReuseAttempted = true;
                    spatialNeighborTarget = min(max(uniforms.spatialReuseNeighborCount, 1u), 4u);
                    constexpr int2 offsets[4] = {
                        int2(-1, 0),
                        int2(1, 0),
                        int2(0, -1),
                        int2(0, 1)
                    };
                    for (uint n = 0u; n < spatialNeighborTarget; ++n) {
                        int2 offset = offsets[n];
                        int2 neighborCoordI = int2(pixelCoord) + offset;
                        if (neighborCoordI.x < 0 ||
                            neighborCoordI.y < 0 ||
                            neighborCoordI.x >= int(uniforms.width) ||
                            neighborCoordI.y >= int(uniforms.height)) {
                            spatialRejectedInvalid += 1u;
                            continue;
                        }

                        uint2 neighborCoord = uint2(neighborCoordI);
                        Ray neighborRay = make_center_primary_ray_for_pixel(uniforms, neighborCoord);
                        HitRecord neighborRec = make_empty_hit_record();
                        bool neighborHit = trace_scene_software(uniforms,
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
                                                                 neighborRay,
                                                                 kEpsilon,
                                                                 kInfinity,
                                                                 /*anyHitOnly=*/false,
                                                                 /*includeTriangles=*/true,
                                                                 neighborRec);
                        if (!neighborHit || neighborRec.materialIndex >= uniforms.materialCount) {
                            spatialRejectedInvalid += 1u;
                            continue;
                        }

                        float depthRel = fabs(rec.t - neighborRec.t) / max(rec.t, kEpsilon);
                        if (!(depthRel <= 0.1f) || !isfinite(depthRel)) {
                            spatialRejectedDepth += 1u;
                            continue;
                        }
                        float normalDot = dot(safe_normalize(shadingNormal),
                                              safe_normalize(neighborRec.shadingNormal));
                        if (!(normalDot >= 0.9f) || !isfinite(normalDot)) {
                            spatialRejectedNormal += 1u;
                            continue;
                        }

                        spatialNeighborsConsidered += 1u;
                        MaterialData neighborMaterial = materials[neighborRec.materialIndex];
                        float3 neighborWo = safe_normalize(-neighborRay.direction);
                        uint neighborState =
                            pcg_hash(state ^ (uint(neighborCoord.x) * 1664525u) ^
                                     (uint(neighborCoord.y) * 1013904223u) ^
                                     (n + 1u) * 747796405u);
                        Reservoir neighborReservoir = make_empty_reservoir();
                        if (!build_ris_reservoir_for_hit(uniforms,
                                                         emissivePrimitives,
                                                         neighborRec,
                                                         neighborMaterial,
                                                         neighborRec.shadingNormal,
                                                         neighborWo,
                                                         clampParams,
                                                         diffuseOcclusion,
                                                         specularOnly,
                                                         risTargetM,
                                                         neighborState,
                                                         neighborReservoir)) {
                            spatialRejectedInvalid += 1u;
                            continue;
                        }

                        float phatCurrent = reservoir_winner_phat_for_hit(uniforms,
                                                                          rec,
                                                                          material,
                                                                          shadingNormal,
                                                                          wo,
                                                                          clampParams,
                                                                          diffuseOcclusion,
                                                                          specularOnly,
                                                                          neighborReservoir.winner);
                        float phatNeighbor = reservoir_winner_phat_for_hit(uniforms,
                                                                           neighborRec,
                                                                           neighborMaterial,
                                                                           neighborRec.shadingNormal,
                                                                           neighborWo,
                                                                           clampParams,
                                                                           diffuseOcclusion,
                                                                           specularOnly,
                                                                           neighborReservoir.winner);
                        float denom = phatCurrent * float(max(reservoir.M, 1u)) +
                                      phatNeighbor * float(max(neighborReservoir.M, 1u));
                        if (!(denom > 0.0f) || !isfinite(denom) || !(phatCurrent > 0.0f)) {
                            spatialRejectedInvalid += 1u;
                            continue;
                        }
                        float mis = phatCurrent / denom;
                        float mergeWeight = mis * neighborReservoir.wSum;
                        if (!(mergeWeight > 0.0f) || !isfinite(mergeWeight)) {
                            spatialRejectedInvalid += 1u;
                            continue;
                        }
                        uint previousWinner = reservoir.winner.primitiveIndex;
                        bool changed = update_reservoir(reservoir,
                                                        neighborReservoir.winner,
                                                        mergeWeight,
                                                        rand_uniform(state));
                        spatialNeighborsAccepted += 1u;
                        spatialLastMisWeight = mis;
                        spatialLastMergeWeight = mergeWeight;
                        if (changed || reservoir.winner.primitiveIndex != previousWinner) {
                            spatialWinnerOffset = offset;
                        }
                    }
                }

                if ((uniforms.directLightMode == kDirectLightModeRisTemporalReuse ||
                     uniforms.directLightMode == kDirectLightModeRestirDi ||
                     uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
                    depth == 0u) {
                    temporalReuseAttempted = true;
                    if (reservoir.valid &&
                        rec.t > 0.0f &&
                        uniforms.frameIndex > 0u) {
                        temporalPreviousAvailable = true;
                        uint temporalState = pcg_hash(uniforms.fixedRngSeed ^
                                                      (uint(pixelCoord.x) * 1664525u) ^
                                                      (uint(pixelCoord.y) * 1013904223u) ^
                                                      ((uniforms.frameIndex - 1u) * 9781u) ^
                                                      0x9E3779B9u);
                        Reservoir previousReservoir = make_empty_reservoir();
                        if (!build_ris_reservoir_for_hit(uniforms,
                                                         emissivePrimitives,
                                                         rec,
                                                         material,
                                                         shadingNormal,
                                                         wo,
                                                         clampParams,
                                                         diffuseOcclusion,
                                                         specularOnly,
                                                         risTargetM,
                                                         temporalState,
                                                         previousReservoir)) {
                            temporalRejectedInvalid += 1u;
                        } else {
                            temporalPreviousPrimitiveIndex = previousReservoir.winner.primitiveIndex;
                            float phatCurrent = reservoir_winner_phat_for_hit(uniforms,
                                                                              rec,
                                                                              material,
                                                                              shadingNormal,
                                                                              wo,
                                                                              clampParams,
                                                                              diffuseOcclusion,
                                                                              specularOnly,
                                                                              previousReservoir.winner);
                            if (!(phatCurrent > 0.0f) || !isfinite(phatCurrent)) {
                                temporalRejectedInvalid += 1u;
                            } else {
                                float mergeWeight = previousReservoir.wSum;
                                if (!(mergeWeight > 0.0f) || !isfinite(mergeWeight)) {
                                    temporalRejectedInvalid += 1u;
                                } else {
                                    merge_reservoir_winner(reservoir,
                                                           previousReservoir.winner,
                                                           mergeWeight,
                                                           previousReservoir.M,
                                                           rand_uniform(state));
                                    temporalReuseAccepted = true;
                                    temporalLastMisWeight = 1.0f;
                                    temporalLastMergeWeight = mergeWeight;
                                }
                            }
                        }
                    } else {
                        temporalRejectedInvalid += 1u;
                    }
                }

                if (uniforms.directLightMode == kDirectLightModeRisWorldReuse &&
                    depth == 0u) {
                    worldReuseAttempted = true;
                    if (reservoir.valid &&
                        rec.t > 0.0f &&
                        uniforms.width > 0u &&
                        uniforms.height > 0u) {
                        worldCell = world_reuse_cell(rec.point, uniforms.worldReuseCellSize);
                        worldCellHash = world_reuse_cell_hash(worldCell);
                        constexpr int2 offsets[4] = {
                            int2(-2, 0),
                            int2(2, 0),
                            int2(0, -2),
                            int2(0, 2)
                        };
                        for (uint n = 0u; n < 4u; ++n) {
                            int2 offset = offsets[n];
                            int2 candidateCoordI = int2(pixelCoord) + offset;
                            if (candidateCoordI.x < 0 ||
                                candidateCoordI.y < 0 ||
                                candidateCoordI.x >= int(uniforms.width) ||
                                candidateCoordI.y >= int(uniforms.height)) {
                                worldRejectedInvalid += 1u;
                                continue;
                            }

                            uint2 candidateCoord = uint2(candidateCoordI);
                            Ray candidateRay = make_center_primary_ray_for_pixel(uniforms, candidateCoord);
                            HitRecord candidateRec = make_empty_hit_record();
                            bool candidateHit = trace_scene_software(uniforms,
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
                                                                      candidateRay,
                                                                      kEpsilon,
                                                                      kInfinity,
                                                                      /*anyHitOnly=*/false,
                                                                      /*includeTriangles=*/true,
                                                                      candidateRec);
                            if (!candidateHit || candidateRec.materialIndex >= uniforms.materialCount) {
                                worldRejectedInvalid += 1u;
                                continue;
                            }

                            int3 candidateCell =
                                world_reuse_cell(candidateRec.point, uniforms.worldReuseCellSize);
                            if (!world_reuse_cell_compatible(worldCell, candidateCell)) {
                                worldRejectedCell += 1u;
                                continue;
                            }

                            float depthRel = fabs(rec.t - candidateRec.t) / max(rec.t, kEpsilon);
                            if (!(depthRel <= 0.2f) || !isfinite(depthRel)) {
                                worldRejectedDepth += 1u;
                                continue;
                            }
                            float normalDot = dot(safe_normalize(shadingNormal),
                                                  safe_normalize(candidateRec.shadingNormal));
                            if (!(normalDot >= 0.85f) || !isfinite(normalDot)) {
                                worldRejectedNormal += 1u;
                                continue;
                            }

                            worldCandidatesConsidered += 1u;
                            MaterialData candidateMaterial = materials[candidateRec.materialIndex];
                            float3 candidateWo = safe_normalize(-candidateRay.direction);
                            uint candidateState =
                                pcg_hash(state ^
                                         (uint(candidateCell.x) * 1664525u) ^
                                         (uint(candidateCell.y) * 1013904223u) ^
                                         (uint(candidateCell.z) * 747796405u) ^
                                         ((n + 1u) * 2891336453u));
                            Reservoir candidateReservoir = make_empty_reservoir();
                            if (!build_ris_reservoir_for_hit(uniforms,
                                                             emissivePrimitives,
                                                             candidateRec,
                                                             candidateMaterial,
                                                             candidateRec.shadingNormal,
                                                             candidateWo,
                                                             clampParams,
                                                             diffuseOcclusion,
                                                             specularOnly,
                                                             risTargetM,
                                                             candidateState,
                                                             candidateReservoir)) {
                                worldRejectedInvalid += 1u;
                                continue;
                            }

                            float phatCurrent = reservoir_winner_phat_for_hit(uniforms,
                                                                              rec,
                                                                              material,
                                                                              shadingNormal,
                                                                              wo,
                                                                              clampParams,
                                                                              diffuseOcclusion,
                                                                              specularOnly,
                                                                              candidateReservoir.winner);
                            float phatCandidate = reservoir_winner_phat_for_hit(uniforms,
                                                                                candidateRec,
                                                                                candidateMaterial,
                                                                                candidateRec.shadingNormal,
                                                                                candidateWo,
                                                                                clampParams,
                                                                                diffuseOcclusion,
                                                                                specularOnly,
                                                                                candidateReservoir.winner);
                            float denom = phatCurrent * float(max(reservoir.M, 1u)) +
                                          phatCandidate * float(max(candidateReservoir.M, 1u));
                            if (!(denom > 0.0f) || !isfinite(denom) || !(phatCurrent > 0.0f)) {
                                worldRejectedInvalid += 1u;
                                continue;
                            }
                            float mis = phatCurrent / denom;
                            float mergeWeight = mis * candidateReservoir.wSum;
                            if (!(mergeWeight > 0.0f) || !isfinite(mergeWeight)) {
                                worldRejectedInvalid += 1u;
                                continue;
                            }

                            update_reservoir(reservoir,
                                             candidateReservoir.winner,
                                             mergeWeight,
                                             rand_uniform(state));
                            worldCandidatesAccepted += 1u;
                            worldReuseAccepted = true;
                            worldCandidatePrimitiveIndex = candidateReservoir.winner.primitiveIndex;
                            worldLastMisWeight = mis;
                            worldLastMergeWeight = mergeWeight;
                        }
                    } else {
                        worldRejectedInvalid += 1u;
                    }
                }

                if ((uniforms.directLightMode == kDirectLightModeRisRegirCache ||
                     uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
                    depth == 0u) {
                    cacheReuseAttempted = true;
                    cacheFallbackUsed = true;
                    cacheCell = world_reuse_cell(rec.point, uniforms.worldReuseCellSize);
                    cacheCellHash = world_reuse_cell_hash(cacheCell);
                    if (uniforms.frameIndex > 1u) {
                        cacheSourceFrameIndex = uniforms.frameIndex - 2u;
                    } else if (uniforms.frameIndex > 0u) {
                        cacheSourceFrameIndex = uniforms.frameIndex - 1u;
                    }
                    if (reservoir.valid &&
                        rec.t > 0.0f &&
                        uniforms.frameIndex > 0u) {
                        cacheStateAvailable = true;
                        cacheFallbackUsed = false;
                        cacheCandidatesConsidered = 0u;
                        cacheEntriesAvailable = 0u;
                        Reservoir retainedReservoir0 = make_empty_reservoir();
                        Reservoir retainedReservoir1 = make_empty_reservoir();
                        uint retainedSourceFrame0 = 0u;
                        uint retainedSourceFrame1 = 0u;
                        uint retainedPrimitive0 = 0xFFFFFFFFu;
                        uint retainedPrimitive1 = 0xFFFFFFFFu;
                        float retainedMis0 = 0.0f;
                        float retainedMis1 = 0.0f;
                        float retainedMerge0 = 0.0f;
                        float retainedMerge1 = 0.0f;
                        float retainedScore0 = -1.0f;
                        float retainedScore1 = -1.0f;
                        uint retainedCount = 0u;
                        for (uint cacheProbe = 0u; cacheProbe < 4u; ++cacheProbe) {
                            if (cacheProbe > cacheSourceFrameIndex) {
                                continue;
                            }
                            uint candidateSourceFrameIndex = cacheSourceFrameIndex - cacheProbe;
                            cacheCandidatesConsidered += 1u;
                            uint cacheState =
                                pcg_hash(uniforms.fixedRngSeed ^
                                         (uint(cacheCell.x) * 1664525u) ^
                                         (uint(cacheCell.y) * 1013904223u) ^
                                         (uint(cacheCell.z) * 747796405u) ^
                                         (candidateSourceFrameIndex * 9781u) ^
                                         (cacheProbe * 0x9E3779B9u) ^
                                         0xD1B54A35u);
                            Reservoir cachedReservoir = make_empty_reservoir();
                            if (!build_ris_reservoir_for_hit(uniforms,
                                                             emissivePrimitives,
                                                             rec,
                                                             material,
                                                             shadingNormal,
                                                             wo,
                                                             clampParams,
                                                             diffuseOcclusion,
                                                             specularOnly,
                                                             risTargetM,
                                                             cacheState,
                                                             cachedReservoir)) {
                                cacheRejectedInvalid += 1u;
                                continue;
                            }
                            if (!cachedReservoir.valid) {
                                cacheRejectedInvalid += 1u;
                                continue;
                            }
                            int3 cachedCell = world_reuse_cell(rec.point, uniforms.worldReuseCellSize);
                            if (!world_reuse_cell_compatible(cacheCell, cachedCell)) {
                                cacheRejectedCell += 1u;
                                continue;
                            }
                            float depthRel = 0.0f;
                            if (!(depthRel <= 0.2f) || !isfinite(depthRel)) {
                                cacheRejectedDepth += 1u;
                                continue;
                            }
                            float normalDot = dot(safe_normalize(shadingNormal),
                                                  safe_normalize(shadingNormal));
                            if (!(normalDot >= 0.85f) || !isfinite(normalDot)) {
                                cacheRejectedNormal += 1u;
                                continue;
                            }
                            float phatCurrent = reservoir_winner_phat_for_hit(uniforms,
                                                                              rec,
                                                                              material,
                                                                              shadingNormal,
                                                                              wo,
                                                                              clampParams,
                                                                              diffuseOcclusion,
                                                                              specularOnly,
                                                                              cachedReservoir.winner);
                            float phatCached = reservoir_winner_phat_for_hit(uniforms,
                                                                             rec,
                                                                             material,
                                                                             shadingNormal,
                                                                             wo,
                                                                             clampParams,
                                                                             diffuseOcclusion,
                                                                             specularOnly,
                                                                             cachedReservoir.winner);
                            float denom = phatCurrent * float(max(reservoir.M, 1u)) +
                                          phatCached * float(max(cachedReservoir.M, 1u));
                            if (!(denom > 0.0f) || !isfinite(denom) || !(phatCurrent > 0.0f)) {
                                cacheRejectedInvalid += 1u;
                            } else {
                                float mis = phatCurrent / denom;
                                float mergeWeight = mis * cachedReservoir.wSum;
                                if (!(mergeWeight > 0.0f) || !isfinite(mergeWeight)) {
                                    cacheRejectedInvalid += 1u;
                                } else {
                                    cacheEntriesAvailable += 1u;
                                    if (mergeWeight > retainedScore0) {
                                        if (retainedCount > 0u) {
                                            retainedReservoir1 = retainedReservoir0;
                                            retainedSourceFrame1 = retainedSourceFrame0;
                                            retainedPrimitive1 = retainedPrimitive0;
                                            retainedMis1 = retainedMis0;
                                            retainedMerge1 = retainedMerge0;
                                            retainedScore1 = retainedScore0;
                                        }
                                        retainedReservoir0 = cachedReservoir;
                                        retainedSourceFrame0 = candidateSourceFrameIndex;
                                        retainedPrimitive0 = cachedReservoir.winner.primitiveIndex;
                                        retainedMis0 = mis;
                                        retainedMerge0 = mergeWeight;
                                        retainedScore0 = mergeWeight;
                                        retainedCount = min(retainedCount + 1u, 2u);
                                    } else if (mergeWeight > retainedScore1) {
                                        retainedReservoir1 = cachedReservoir;
                                        retainedSourceFrame1 = candidateSourceFrameIndex;
                                        retainedPrimitive1 = cachedReservoir.winner.primitiveIndex;
                                        retainedMis1 = mis;
                                        retainedMerge1 = mergeWeight;
                                        retainedScore1 = mergeWeight;
                                        retainedCount = min(retainedCount + 1u, 2u);
                                    }
                                }
                            }
                        }
                        if (retainedCount > 0u && retainedReservoir0.valid) {
                            update_reservoir(reservoir,
                                             retainedReservoir0.winner,
                                             retainedMerge0,
                                             rand_uniform(state));
                            cacheCandidatesAccepted += 1u;
                            cacheReuseAccepted = true;
                            cacheSourceFrameIndex = retainedSourceFrame0;
                            cacheCandidatePrimitiveIndex = retainedPrimitive0;
                            cacheLastMisWeight = retainedMis0;
                            cacheLastMergeWeight = retainedMerge0;
                        }
                        if (retainedCount > 1u && retainedReservoir1.valid) {
                            update_reservoir(reservoir,
                                             retainedReservoir1.winner,
                                             retainedMerge1,
                                             rand_uniform(state));
                            cacheCandidatesAccepted += 1u;
                            cacheReuseAccepted = true;
                            cacheSourceFrameIndex = retainedSourceFrame1;
                            cacheCandidatePrimitiveIndex = retainedPrimitive1;
                            cacheLastMisWeight = retainedMis1;
                            cacheLastMergeWeight = retainedMerge1;
                        }
                    } else {
                        cacheRejectedInvalid += 1u;
                    }
                }

                if (reservoir.valid && reservoir.M > 0u && reservoir.wSum > 0.0f) {
                    RISSamplePayload winner = reservoir.winner;
                    float distSq = ris_payload_distance_sq(rec, winner);
                    if (distSq > 0.0f && isfinite(distSq)) {
                        const bool winnerDirectional = ris_payload_is_directional(winner);
                        float distance = winnerDirectional ? 1.0e20f : length(winner.position - rec.point);
                        float3 omegaL = ris_payload_omega_l(rec, winner);
                        float cosThetaSurface = saturate(dot(shadingNormal, omegaL));
                        float cosThetaLight = winnerDirectional ? 1.0f : saturate(dot(winner.normal, -omegaL));
                        BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                                rec.point,
                                                                shadingNormal,
                                                                wo,
                                                                omegaL,
                                                                clampParams,
                                                                uniforms.sssMode,
                                                                diffuseOcclusion,
                                                                specularOnly);
                        float phatW = p_hat(winner.emission,
                                            bsdfEval.value,
                                            cosThetaSurface,
                                            cosThetaLight,
                                            distSq);
                        reservoir.W = (phatW > 0.0f)
                                    ? ((reservoir.wSum / float(reservoir.M)) / phatW)
                                    : 0.0f;

                        Ray shadowRay;
                        shadowRay.origin = offset_ray_origin(rec, omegaL);
                        shadowRay.direction = omegaL;
                        HitRecord shadowRec = make_empty_hit_record();
                        float shadowMax = max(distance - kEpsilon, kEpsilon);
                        uint shadowExcludeMesh;
                        uint shadowExcludePrim;
                        compute_exclusion_indices(rec, shadowExcludeMesh, shadowExcludePrim);
                        bool occluded = false;
#if PT_DEBUG_TOOLS
                        if (forceSoftware) {
                            occluded = trace_scene_software(uniforms,
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
                        } else {
#endif
                            occluded = trace_scene_hardware(uniforms,
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
                                                            shadowExcludeMesh,
                                                            shadowExcludePrim,
                                                            shadowRec);
                            if (!winnerDirectional &&
                                occluded &&
                                shadowRec.primitiveType == kPrimitiveTypeTriangle &&
                                shadowRec.primitiveIndex == winner.primitiveIndex) {
                                occluded = false;
                            }
                            if (!occluded) {
                                HitRecord swShadowRec = make_empty_hit_record();
                                bool swOccluded = trace_scene_software(uniforms,
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
                                                                       /*stats=*/nullptr,
                                                                       shadowRay,
                                                                       kEpsilon,
                                                                       shadowMax,
                                                                       /*anyHitOnly=*/true,
                                                                       /*includeTriangles=*/true,
                                                                       swShadowRec);
                                if (swOccluded &&
                                    (winnerDirectional ||
                                     !(swShadowRec.primitiveType == kPrimitiveTypeTriangle &&
                                       swShadowRec.primitiveIndex == winner.primitiveIndex))) {
                                    occluded = true;
                                    shadowRec = swShadowRec;
                                }
                            }
#if PT_DEBUG_TOOLS
                        }
#endif

                        float3 contribution = float3(0.0f);
                        if (!occluded &&
                            !bsdfEval.isDelta &&
                            !bsdfEval.isBssrdf &&
                            reservoir.W > 0.0f &&
                            cosThetaSurface > 0.0f) {
                            contribution = bsdfEval.value * winner.emission;
                            contribution *= cosThetaSurface * reservoir.W;
                            if (all(isfinite(contribution))) {
                                radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                            } else {
                                contribution = float3(0.0f);
                            }
                        }

                        risAuditReservoir = reservoir;
                        risAuditBsdfValue = bsdfEval.value;
                        risAuditContribution = contribution;
                        risAuditDistance = distance;
                        risAuditNDotL = cosThetaSurface;
                        risAuditBsdfPdf = bsdfEval.pdf;
                        risAuditVisible = !occluded;
                        risAuditActive = true;
                    }
                }
            }
        } else if (!surfaceIsDelta && rectLightCount > 0u) {
            RectLightSample lightSample;
            if (sample_rect_light(uniforms,
                                  rectangles,
                                  materials,
                                  environmentTexture,
                                  rec,
                                  state,
                                  rectLightCount,
                                  lightSample)) {
                float nDotL = max(dot(shadingNormal, lightSample.direction), 0.0f);
                if (lightSample.pdf > 0.0f && nDotL > 0.0f) {
                    if (debugContext) {
                        record_debug_event(*debugContext,
                                           kDebugEventRectSample,
                                           depth,
                                           mediumDepth,
                                           mediumDepth,
                                           /*mediumEvent=*/0,
                                           rec.frontFace,
                                           rec.materialIndex,
                                           /*scatterIsDelta=*/false,
                                           throughput,
                                           float4(lightSample.pdf,
                                                  lightSample.distance,
                                                  nDotL,
                                                  float(lightSample.rectIndex)),
                                           lightSample.emission);
                    }
                    Ray shadowRay;
                    shadowRay.origin = offset_ray_origin(rec, lightSample.direction);
                    shadowRay.direction = lightSample.direction;
                    HitRecord shadowRec = make_empty_hit_record();
                    float shadowMax = max(lightSample.distance - kEpsilon, kEpsilon);
                    uint shadowExcludeMesh;
                    uint shadowExcludePrim;
                    compute_exclusion_indices(rec, shadowExcludeMesh, shadowExcludePrim);
                    bool occluded = false;
#if PT_DEBUG_TOOLS
                    if (forceSoftware) {
                        occluded = trace_scene_software(uniforms,
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
                    } else {
#endif
                        occluded = trace_scene_hardware(uniforms,
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
                                                        shadowExcludeMesh,
                                                        shadowExcludePrim,
                                                        shadowRec);
                        if (occluded &&
                            shadowRec.primitiveType == kPrimitiveTypeRectangle &&
                            shadowRec.primitiveIndex == lightSample.rectIndex) {
                            occluded = false;
                        }
                        if (!occluded) {
                            HitRecord swShadowRec = make_empty_hit_record();
                            bool swOccluded = trace_scene_software(uniforms,
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
                                                                   /*stats=*/nullptr,
                                                                   shadowRay,
                                                                   kEpsilon,
                                                                   shadowMax,
                                                                   /*anyHitOnly=*/true,
                                                                   /*includeTriangles=*/true,
                                                                   swShadowRec);
                            if (swOccluded &&
                                !(swShadowRec.primitiveType == kPrimitiveTypeRectangle &&
                                  swShadowRec.primitiveIndex == lightSample.rectIndex)) {
                                occluded = true;
                                shadowRec = swShadowRec;
                            }
                        }
                    if (debugContext) {
                        record_debug_event(*debugContext,
                                           kDebugEventRectShadow,
                                           depth,
                                           mediumDepth,
                                           mediumDepth,
                                           /*mediumEvent=*/0,
                                           rec.frontFace,
                                           rec.materialIndex,
                                           /*scatterIsDelta=*/false,
                                           throughput,
                                           float4(occluded ? 1.0f : 0.0f,
                                                  shadowMax,
                                                  shadowRec.t,
                                                  float(lightSample.rectIndex)),
                                           float3(float(shadowRec.materialIndex),
                                                  float(shadowRec.meshIndex),
                                                  float(shadowRec.primitiveIndex)));
                    }
#if PT_DEBUG_TOOLS
                    }
                    if (doParity) {
                        HitRecord swShadowRec = make_empty_hit_record();
                        bool swOccluded = trace_scene_software(uniforms,
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
                                                               /*stats=*/nullptr,
                                                               shadowRay,
                                                               kEpsilon,
                                                               shadowMax,
                                                               /*anyHitOnly=*/true,
                                                               /*includeTriangles=*/true,
                                                               swShadowRec);
                        uint reasonMask = 0u;
                        if (occluded != swOccluded) {
                            reasonMask |= kParityReasonHitMiss;
                            // If HWRT hit something at t < tMin it was found only
                            // via the tMin=0 retry — SWRT correctly skips it.
                            if (occluded && shadowRec.t < kEpsilon) {
                                reasonMask |= kParityReasonBelowTMin;
                            }
                        }
                        if (occluded && swOccluded) {
                            float epsT = max(1.0e-3f, 1.0e-4f * fabs(shadowRec.t));
                            if (fabs(shadowRec.t - swShadowRec.t) > epsT) {
                                reasonMask |= kParityReasonT;
                            }
                            if (shadowRec.frontFace != swShadowRec.frontFace) {
                                reasonMask |= kParityReasonFrontFace;
                            }
                            if (shadowRec.materialIndex != swShadowRec.materialIndex ||
                                shadowRec.meshIndex != swShadowRec.meshIndex ||
                                shadowRec.primitiveIndex != swShadowRec.primitiveIndex) {
                                reasonMask |= kParityReasonId;
                            }
                            float3 hwN = shadowRec.normal;
                            float3 swN = swShadowRec.normal;
                            if (all(isfinite(hwN)) && all(isfinite(swN)) &&
                                dot(hwN, hwN) > 0.0f && dot(swN, swN) > 0.0f &&
                                dot(normalize(hwN), normalize(swN)) < 0.99f) {
                                reasonMask |= kParityReasonNormal;
                            }
                        }
                        record_parity_entry(*debugContext,
                                            uniforms,
                                            depth,
                                            kParityProbeRectShadow,
                                            shadowRay,
                                            kEpsilon,
                                            shadowMax,
                                            occluded,
                                            shadowRec,
                                            swOccluded,
                                            swShadowRec,
                                            reasonMask);
                    }
#endif
                    if (!occluded) {
                        BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                                rec.point,
                                                                shadingNormal,
                                                                wo,
                                                                lightSample.direction,
                                                                clampParams,
                                                                uniforms.sssMode,
                                                                diffuseOcclusion,
                                                                specularOnly);
                        float3 bsdfValue = bsdfEval.value;
                        float maxComponent = max(max(bsdfValue.x, bsdfValue.y), bsdfValue.z);
                        if (debugContext) {
                            float flags = (bsdfEval.isDelta ? 1.0f : 0.0f) +
                                          (bsdfEval.isBssrdf ? 2.0f : 0.0f);
                            record_debug_event(*debugContext,
                                               kDebugEventRectEval,
                                               depth,
                                               mediumDepth,
                                               mediumDepth,
                                               /*mediumEvent=*/0,
                                               rec.frontFace,
                                               rec.materialIndex,
                                               /*scatterIsDelta=*/false,
                                               throughput,
                                               float4(lightSample.pdf,
                                                      bsdfEval.pdf,
                                                      maxComponent,
                                                      flags),
                                               bsdfValue);
                        }
                        if (!bsdfEval.isDelta && !bsdfEval.isBssrdf) {
                            if (maxComponent > 0.0f && lightSample.pdf > 0.0f) {
                                float bsdfPdf = bsdfEval.pdf;
                                float weight = 1.0f;
                                if (bsdfPdf > 0.0f) {
                                    float denom = lightSample.pdf + bsdfPdf;
                                    if (denom > 0.0f) {
                                        weight = clamp(lightSample.pdf / denom,
                                                       kMisWeightClampMin,
                                                       kMisWeightClampMax);
                                    }
                                }
                                float3 contribution = lightSample.emission * bsdfValue * nDotL;
                                contribution *= weight / lightSample.pdf;
                                if (all(isfinite(contribution))) {
                                    radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                                    if (debugContext) {
                                        record_debug_event(*debugContext,
                                                           kDebugEventRectNee,
                                                           depth,
                                                           mediumDepth,
                                                           mediumDepth,
                                                           /*mediumEvent=*/0,
                                                           rec.frontFace,
                                                           rec.materialIndex,
                                                           /*scatterIsDelta=*/false,
                                                           throughput,
                                                           float4(lightSample.pdf,
                                                                  bsdfPdf,
                                                                  weight,
                                                                  nDotL),
                                                           contribution);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (debugContext &&
            depth == 0u &&
            uniforms.debugDirectLightAudit != 0u &&
            (rectLightCount > 0u || uniforms.emissivePrimitiveCount > 0u)) {
            float3 auditBaseColor = debugBaseColor;
            float2 auditBaseColorUv = debugBaseColorUv;
            float auditBaseColorLod = debugBaseColorLod;
            record_debug_event(*debugContext,
                               kDebugEventDirectLightAuditMeta,
                               depth,
                               mediumDepth,
                               mediumDepth,
                               /*mediumEvent=*/0,
                               rec.frontFace,
                               rec.materialIndex,
                               /*scatterIsDelta=*/false,
                               rec.point,
                               float4(float(type),
                                      float(rec.primitiveIndex),
                                      0.0f,
                                      0.0f),
                               shadingNormal);
            record_debug_event(*debugContext,
                               kDebugEventDirectLightAuditMaterial,
                               depth,
                               mediumDepth,
                               mediumDepth,
                               /*mediumEvent=*/0,
                               rec.frontFace,
                               rec.materialIndex,
                               /*scatterIsDelta=*/false,
                               auditBaseColor,
                               float4(material_closure_aov_roughness(material),
                                      auditBaseColorUv.x,
                                      auditBaseColorUv.y,
                                      auditBaseColorLod),
                               float3(float(material.textureIndices0.x), 0.0f, 0.0f));
            if (uniforms.directLightMode == kDirectLightModeRis ||
                uniforms.directLightMode == kDirectLightModeRisSpatialReuse ||
                uniforms.directLightMode == kDirectLightModeRisTemporalReuse ||
                uniforms.directLightMode == kDirectLightModeRisWorldReuse ||
                uniforms.directLightMode == kDirectLightModeRisRegirCache ||
                uniforms.directLightMode == kDirectLightModeRestirDi ||
                uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) {
                record_debug_event(*debugContext,
                                   kDebugEventRisAuditState,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   float3(risAuditReservoir.wSum,
                                          risAuditReservoir.W,
                                          risAuditReservoir.winner.pdf),
                                   float4(float(risAuditReservoir.winner.primitiveIndex),
                                          float(risAuditReservoir.M),
                                          risAuditReservoir.valid ? 1.0f : 0.0f,
                                          risAuditVisible ? 1.0f : 0.0f),
                                   float3(risAuditDistance,
                                          risAuditNDotL,
                                          risAuditBsdfPdf));
                if (risAuditReservoir.valid) {
                    record_debug_event(*debugContext,
                                       kDebugEventRisAuditWinnerA,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       risAuditReservoir.winner.position,
                                       float4(risAuditReservoir.winner.bary.x,
                                              risAuditReservoir.winner.bary.y,
                                              0.0f,
                                              0.0f),
                                       risAuditReservoir.winner.normal);
                    record_debug_event(*debugContext,
                                       kDebugEventRisAuditWinnerB,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       risAuditReservoir.winner.emission,
                                       float4(0.0f),
                                       risAuditBsdfValue);
                    record_debug_event(*debugContext,
                                       kDebugEventRisAuditWinnerC,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       risAuditContribution,
                                       float4(0.0f),
                                       float3(0.0f));
                }
                if (uniforms.directLightMode == kDirectLightModeRisSpatialReuse) {
                    record_debug_event(*debugContext,
                                       kDebugEventSpatialReuseAudit,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       float3(float(spatialNeighborTarget),
                                              float(spatialNeighborsConsidered),
                                              float(spatialNeighborsAccepted)),
                                       float4(float(spatialRejectedDepth),
                                              float(spatialRejectedNormal),
                                              float(spatialRejectedInvalid),
                                              spatialReuseAttempted ? 1.0f : 0.0f),
                                       float3(float(spatialWinnerOffset.x),
                                              float(spatialWinnerOffset.y),
                                              0.0f));
                    record_debug_event(*debugContext,
                                       kDebugEventSpatialReuseWeights,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       float3(spatialLastMisWeight,
                                              spatialLastMergeWeight,
                                              0.0f),
                                       float4(0.0f),
                                       float3(0.0f));
                }
                if (uniforms.directLightMode == kDirectLightModeRisTemporalReuse ||
                    uniforms.directLightMode == kDirectLightModeRestirDi ||
                    uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) {
                    record_debug_event(*debugContext,
                                       kDebugEventTemporalReuseAudit,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       float3(temporalPreviousAvailable ? 1.0f : 0.0f,
                                              temporalReuseAccepted ? 1.0f : 0.0f,
                                              float(temporalPreviousPrimitiveIndex)),
                                       float4(float(temporalRejectedDepth),
                                              float(temporalRejectedNormal),
                                              float(temporalRejectedInvalid),
                                              temporalReuseAttempted ? 1.0f : 0.0f),
                                       float3(0.0f));
                    record_debug_event(*debugContext,
                                       kDebugEventTemporalReuseWeights,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       float3(temporalLastMisWeight,
                                              temporalLastMergeWeight,
                                              0.0f),
                                       float4(0.0f),
                                       float3(0.0f));
                }
                if (uniforms.directLightMode == kDirectLightModeRisWorldReuse) {
                    record_debug_event(*debugContext,
                                       kDebugEventWorldReuseAudit,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       float3(float(worldCandidatesConsidered),
                                              float(worldCandidatesAccepted),
                                              float(worldCellHash)),
                                       float4(float(worldRejectedDepth),
                                              float(worldRejectedNormal),
                                              float(worldRejectedInvalid),
                                              worldReuseAttempted ? 1.0f : 0.0f),
                                       float3(float(worldCell.x),
                                              float(worldCell.y),
                                              float(worldCell.z)));
                    record_debug_event(*debugContext,
                                       kDebugEventWorldReuseWeights,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       float3(worldLastMisWeight,
                                              worldLastMergeWeight,
                                              float(worldRejectedCell)),
                                       float4(float(worldCandidatePrimitiveIndex),
                                              worldReuseAccepted ? 1.0f : 0.0f,
                                              1.0f,
                                              4.0f),
                                       float3(0.0f));
                }
                if (uniforms.directLightMode == kDirectLightModeRisRegirCache ||
                    uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) {
                    record_debug_event(*debugContext,
                                       kDebugEventCacheReuseAudit,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       float3(float(cacheCandidatesConsidered),
                                              float(cacheCandidatesAccepted),
                                              float(cacheCellHash)),
                                       float4(float(cacheRejectedDepth),
                                              float(cacheRejectedNormal),
                                              float(cacheRejectedInvalid),
                                              cacheReuseAttempted ? 1.0f : 0.0f),
                                       float3(float(cacheCell.x),
                                              float(cacheCell.y),
                                              float(cacheCell.z)));
                    record_debug_event(*debugContext,
                                       kDebugEventCacheReuseWeights,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/0,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       float3(cacheLastMisWeight,
                                              cacheLastMergeWeight,
                                              float(cacheRejectedCell)),
                                       float4(float(cacheCandidatePrimitiveIndex),
                                              cacheReuseAccepted ? 1.0f : 0.0f,
                                              cacheStateAvailable ? 1.0f : 0.0f,
                                              4.0f),
                                       float3(cacheFallbackUsed ? 1.0f : 0.0f,
                                              float(cacheSourceFrameIndex),
                                              float(cacheEntriesAvailable)));
                }
            }
            if (rectLightCount > 0u) {
                for (uint rectIndex = 0u; rectIndex < uniforms.rectangleCount; ++rectIndex) {
                    RectLightSample auditLight;
                    if (!deterministic_rect_light_sample(uniforms,
                                                         rectangles,
                                                         materials,
                                                         environmentTexture,
                                                         rec,
                                                         rectLightCount,
                                                         rectIndex,
                                                         auditLight)) {
                        continue;
                    }
                    float nDotL = max(dot(shadingNormal, auditLight.direction), 0.0f);
                    BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                            rec.point,
                                                            shadingNormal,
                                                            wo,
                                                            auditLight.direction,
                                                            clampParams,
                                                            uniforms.sssMode,
                                                            diffuseOcclusion,
                                                            specularOnly);
                    float bsdfPdf = bsdfEval.pdf;
                    float misWeight = 1.0f;
                    if (bsdfPdf > 0.0f) {
                        float denom = auditLight.pdf + bsdfPdf;
                        if (denom > 0.0f) {
                            misWeight = clamp(auditLight.pdf / denom,
                                              kMisWeightClampMin,
                                              kMisWeightClampMax);
                        }
                    }
                    HitRecord auditShadowRec = make_empty_hit_record();
                    HardwareShadowTraceAudit hwShadowAudit;
                    SoftwareShadowTraceAudit swShadowAudit;
                    Ray auditShadowRay;
                    auditShadowRay.origin = offset_ray_origin(rec, auditLight.direction);
                    auditShadowRay.direction = auditLight.direction;
                    float shadowMax = max(auditLight.distance - kEpsilon, kEpsilon);
                    bool lightTwoSided = (rectangles[rectIndex].materialTwoSided.y != 0u);
                    bool occluded = false;
#if PT_DEBUG_TOOLS
                    if (forceSoftware) {
                        occluded = trace_scene_software(uniforms,
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
                                                         /*stats=*/nullptr,
                                                         auditShadowRay,
                                                         kEpsilon,
                                                         shadowMax,
                                                         /*anyHitOnly=*/true,
                                                         /*includeTriangles=*/true,
                                                         &swShadowAudit,
                                                         auditShadowRec);
                    } else {
                        uint shadowExcludeMesh;
                        uint shadowExcludePrim;
                        compute_exclusion_indices(rec, shadowExcludeMesh, shadowExcludePrim);
                        occluded = trace_scene_hardware(uniforms,
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
                                                        /*stats=*/nullptr,
                                                        auditShadowRay,
                                                        kEpsilon,
                                                        shadowMax,
                                                        /*anyHitOnly=*/true,
                                                        shadowExcludeMesh,
                                                        shadowExcludePrim,
                                                        &hwShadowAudit,
                                                        auditShadowRec);
                        if (occluded &&
                            auditShadowRec.primitiveType == kPrimitiveTypeRectangle &&
                            auditShadowRec.primitiveIndex == auditLight.rectIndex) {
                            occluded = false;
                            hwShadowAudit.rejectionMask |= kShadowAuditRejectSelfLightRect;
                        }
                        if (!occluded) {
                            HitRecord swShadowRec = make_empty_hit_record();
                            bool swOccluded = trace_scene_software(uniforms,
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
                                                                   /*stats=*/nullptr,
                                                                   auditShadowRay,
                                                                   kEpsilon,
                                                                   shadowMax,
                                                                   /*anyHitOnly=*/true,
                                                                   /*includeTriangles=*/true,
                                                                   &swShadowAudit,
                                                                   swShadowRec);
                            if (swOccluded &&
                                !(swShadowRec.primitiveType == kPrimitiveTypeRectangle &&
                                  swShadowRec.primitiveIndex == auditLight.rectIndex)) {
                                occluded = true;
                                auditShadowRec = swShadowRec;
                                hwShadowAudit.rejectionMask |= kShadowAuditRejectEmbeddedSwOverride;
                            }
                        }
                    }
#else
                    occluded = trace_scene_software(uniforms,
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
                                                     /*stats=*/nullptr,
                                                     auditShadowRay,
                                                     kEpsilon,
                                                     shadowMax,
                                                     /*anyHitOnly=*/true,
                                                     /*includeTriangles=*/true,
                                                     &swShadowAudit,
                                                     auditShadowRec);
#endif
                    if (debugContext) {
                        record_shadow_path_audit(*debugContext,
                                                 depth,
                                                 mediumDepth,
                                                 hwShadowAudit,
                                                 swShadowAudit,
                                                 rectIndex,
                                                 kDirectLightAuditKindRect,
                                                 kEpsilon,
                                                 shadowMax,
                                                 lightTwoSided,
                                                 !occluded);
                    }
                    float3 contribution = float3(0.0f);
                    if (!occluded && !bsdfEval.isDelta && !bsdfEval.isBssrdf &&
                        nDotL > 0.0f && auditLight.pdf > 0.0f) {
                        contribution = auditLight.emission * bsdfEval.value * nDotL;
                        contribution *= misWeight / auditLight.pdf;
                    }
                    record_debug_event(*debugContext,
                                       kDebugEventDirectLightAuditEval,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/kDirectLightAuditKindRect,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       auditLight.direction,
                                       float4(float(rectIndex),
                                              auditLight.distance,
                                              nDotL,
                                              0.0f),
                                       bsdfEval.value);
                    record_debug_event(*debugContext,
                                       kDebugEventDirectLightAuditContrib,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/kDirectLightAuditKindRect,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       contribution,
                                       float4(float(rectIndex),
                                              occluded ? 0.0f : 1.0f,
                                              misWeight,
                                              auditLight.pdf),
                                       float3(bsdfPdf, 0.0f, 0.0f));
                }
            }
            if (uniforms.emissivePrimitiveCount > 0u) {
                for (uint primitiveAuditIndex = 0u; primitiveAuditIndex < uniforms.emissivePrimitiveCount; ++primitiveAuditIndex) {
                    EmissivePrimitiveAuditSample auditLight;
                    if (!deterministic_emissive_primitive_sample(uniforms,
                                                                 emissivePrimitives,
                                                                 rec,
                                                                 primitiveAuditIndex,
                                                                 auditLight)) {
                        continue;
                    }
                    float nDotL = max(dot(shadingNormal, auditLight.direction), 0.0f);
                    BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                            rec.point,
                                                            shadingNormal,
                                                            wo,
                                                            auditLight.direction,
                                                            clampParams,
                                                            uniforms.sssMode,
                                                            diffuseOcclusion,
                                                            specularOnly);
                    float bsdfPdf = bsdfEval.pdf;
                    float misWeight = 1.0f;
                    HitRecord auditShadowRec = make_empty_hit_record();
                    HardwareShadowTraceAudit hwShadowAudit;
                    SoftwareShadowTraceAudit swShadowAudit;
                    Ray auditShadowRay;
                    auditShadowRay.origin = offset_ray_origin(rec, auditLight.direction);
                    auditShadowRay.direction = auditLight.direction;
                    float shadowMax = max(auditLight.distance - kEpsilon, kEpsilon);
                    bool occluded = false;
#if PT_DEBUG_TOOLS
                    if (forceSoftware) {
                        occluded = trace_scene_software(uniforms,
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
                                                         /*stats=*/nullptr,
                                                         auditShadowRay,
                                                         kEpsilon,
                                                         shadowMax,
                                                         /*anyHitOnly=*/true,
                                                         /*includeTriangles=*/true,
                                                         &swShadowAudit,
                                                         auditShadowRec);
                    } else {
                        uint shadowExcludeMesh;
                        uint shadowExcludePrim;
                        compute_exclusion_indices(rec, shadowExcludeMesh, shadowExcludePrim);
                        occluded = trace_scene_hardware(uniforms,
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
                                                        /*stats=*/nullptr,
                                                        auditShadowRay,
                                                        kEpsilon,
                                                        shadowMax,
                                                        /*anyHitOnly=*/true,
                                                        shadowExcludeMesh,
                                                        shadowExcludePrim,
                                                        &hwShadowAudit,
                                                        auditShadowRec);
                        if (occluded &&
                            auditShadowRec.primitiveType == kPrimitiveTypeTriangle &&
                            auditShadowRec.primitiveIndex == auditLight.primitiveIndex) {
                            occluded = false;
                        }
                        if (!occluded) {
                            HitRecord swShadowRec = make_empty_hit_record();
                            bool swOccluded = trace_scene_software(uniforms,
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
                                                                   /*stats=*/nullptr,
                                                                   auditShadowRay,
                                                                   kEpsilon,
                                                                   shadowMax,
                                                                   /*anyHitOnly=*/true,
                                                                   /*includeTriangles=*/true,
                                                                   &swShadowAudit,
                                                                   swShadowRec);
                            if (swOccluded &&
                                !(swShadowRec.primitiveType == kPrimitiveTypeTriangle &&
                                  swShadowRec.primitiveIndex == auditLight.primitiveIndex)) {
                                occluded = true;
                                auditShadowRec = swShadowRec;
                                hwShadowAudit.rejectionMask |= kShadowAuditRejectEmbeddedSwOverride;
                            }
                        }
                    }
#else
                    occluded = trace_scene_software(uniforms,
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
                                                     /*stats=*/nullptr,
                                                     auditShadowRay,
                                                     kEpsilon,
                                                     shadowMax,
                                                     /*anyHitOnly=*/true,
                                                     /*includeTriangles=*/true,
                                                     &swShadowAudit,
                                                     auditShadowRec);
                    if (occluded &&
                        auditShadowRec.primitiveType == kPrimitiveTypeTriangle &&
                        auditShadowRec.primitiveIndex == auditLight.primitiveIndex) {
                        occluded = false;
                    }
#endif
                    if (debugContext) {
                        record_shadow_path_audit(*debugContext,
                                                 depth,
                                                 mediumDepth,
                                                 hwShadowAudit,
                                                 swShadowAudit,
                                                 auditLight.primitiveIndex,
                                                 kDirectLightAuditKindEmissivePrimitive,
                                                 kEpsilon,
                                                 shadowMax,
                                                 /*lightTwoSided=*/false,
                                                 !occluded);
                    }
                    float3 contribution = float3(0.0f);
                    if (!occluded && !bsdfEval.isDelta && !bsdfEval.isBssrdf &&
                        nDotL > 0.0f && auditLight.pdf > 0.0f) {
                        contribution = auditLight.emission * bsdfEval.value * nDotL;
                        contribution *= misWeight / auditLight.pdf;
                    }
                    record_debug_event(*debugContext,
                                       kDebugEventDirectLightAuditEval,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/kDirectLightAuditKindEmissivePrimitive,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       auditLight.direction,
                                       float4(float(auditLight.primitiveIndex),
                                              auditLight.distance,
                                              nDotL,
                                              0.0f),
                                       bsdfEval.value);
                    record_debug_event(*debugContext,
                                       kDebugEventDirectLightAuditContrib,
                                       depth,
                                       mediumDepth,
                                       mediumDepth,
                                       /*mediumEvent=*/kDirectLightAuditKindEmissivePrimitive,
                                       rec.frontFace,
                                       rec.materialIndex,
                                       /*scatterIsDelta=*/false,
                                       contribution,
                                       float4(float(auditLight.primitiveIndex),
                                              occluded ? 0.0f : 1.0f,
                                              misWeight,
                                              auditLight.pdf),
                                       float3(bsdfPdf, 0.0f, 0.0f));
                }
            }
        }
        if (!surfaceIsDelta && envSampling) {
            EnvironmentSample envSample;
            if (sample_environment(uniforms,
                                   environmentTexture,
                                   environmentConditionalAlias,
                                   environmentMarginalAlias,
                                   environmentPdf,
                                   state,
                                   envSample)) {
                float overrideLod = 0.0f;
                bool useOverride = environment_mip_override(uniforms, environmentTexture, overrideLod);
                if (environmentTexture.get_num_mip_levels() > 1u) {
                    float envRoughness = environment_lighting_roughness(material);
                    if (envRoughness < 0.95f) {
                        float envLod = environment_lod_from_roughness(envRoughness,
                                                                      environmentTexture);
                        envSample.radiance = environment_color_lod(environmentTexture,
                                                                   envSample.direction,
                                                                   uniforms.environmentRotation,
                                                                   uniforms.environmentIntensity,
                                                                   envLod,
                                                                   uniforms);
                    }
                }
                if (useOverride) {
                    envSample.radiance = environment_color_lod(environmentTexture,
                                                               envSample.direction,
                                                               uniforms.environmentRotation,
                                                               uniforms.environmentIntensity,
                                                               overrideLod,
                                                               uniforms);
                }
                float nDotL = max(dot(shadingNormal, envSample.direction), 0.0f);
                if (envSample.pdf > 0.0f && nDotL > 0.0f) {
                    BsdfEvalResult bsdfEval = evaluate_bsdf(material,
                                                            rec.point,
                                                            shadingNormal,
                                                            wo,
                                                            envSample.direction,
                                                            clampParams,
                                                            uniforms.sssMode,
                                                            diffuseOcclusion,
                                                            specularOnly);
                    float3 bsdfValue = bsdfEval.value;
                    float maxComponent = max(max(bsdfValue.x, bsdfValue.y), bsdfValue.z);
                    bool bsdfConnectable = !bsdfEval.isDelta &&
                                           !bsdfEval.isBssrdf &&
                                           (maxComponent > 0.0f);
                    if (bsdfConnectable) {
                        float bsdfPdf = bsdfEval.pdf;
                        float weight = 1.0f;
                        if (bsdfPdf > 0.0f) {
                            float denom = envSample.pdf + bsdfPdf;
                            if (denom > 0.0f) {
                                weight = clamp(envSample.pdf / denom,
                                               kMisWeightClampMin,
                                               kMisWeightClampMax);
                            }
                        }
                        Ray shadowRay;
                        shadowRay.origin = offset_ray_origin(rec, envSample.direction);
                        shadowRay.direction = envSample.direction;
                        HitRecord shadowRec;
                        uint shadowExcludeMesh;
                        uint shadowExcludePrim;
                        compute_exclusion_indices(rec, shadowExcludeMesh, shadowExcludePrim);
                        bool occluded = false;
#if PT_DEBUG_TOOLS
                        if (forceSoftware) {
                            occluded = trace_scene_software(uniforms,
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
                                                            kInfinity,
                                                            /*anyHitOnly=*/true,
                                                            /*includeTriangles=*/true,
                                                            shadowRec);
                        } else {
#endif
                            occluded = trace_scene_hardware(uniforms,
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
                                                            kHardwareOcclusionEpsilon,
                                                            kInfinity,
                                                            /*anyHitOnly=*/true,
                                                            shadowExcludeMesh,
                                                            shadowExcludePrim,
                                                            shadowRec);
#if PT_DEBUG_TOOLS
                        }
                        if (doParity) {
                            HitRecord swShadowRec = make_empty_hit_record();
                            bool swOccluded = trace_scene_software(uniforms,
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
                                                                   /*stats=*/nullptr,
                                                                   shadowRay,
                                                                   kEpsilon,
                                                                   kInfinity,
                                                                   /*anyHitOnly=*/true,
                                                                   /*includeTriangles=*/true,
                                                                   swShadowRec);
                            uint reasonMask = 0u;
                            if (occluded != swOccluded) {
                                reasonMask |= kParityReasonHitMiss;
                            }
                            if (occluded && swOccluded) {
                                float epsT = max(1.0e-3f, 1.0e-4f * fabs(shadowRec.t));
                                if (fabs(shadowRec.t - swShadowRec.t) > epsT) {
                                    reasonMask |= kParityReasonT;
                                }
                                if (shadowRec.frontFace != swShadowRec.frontFace) {
                                    reasonMask |= kParityReasonFrontFace;
                                }
                                if (shadowRec.materialIndex != swShadowRec.materialIndex ||
                                    shadowRec.meshIndex != swShadowRec.meshIndex ||
                                    shadowRec.primitiveIndex != swShadowRec.primitiveIndex) {
                                    reasonMask |= kParityReasonId;
                                }
                                float3 hwN = shadowRec.normal;
                                float3 swN = swShadowRec.normal;
                                if (all(isfinite(hwN)) && all(isfinite(swN)) &&
                                    dot(hwN, hwN) > 0.0f && dot(swN, swN) > 0.0f &&
                                    dot(normalize(hwN), normalize(swN)) < 0.99f) {
                                    reasonMask |= kParityReasonNormal;
                                }
                            }
                            record_parity_entry(*debugContext,
                                                uniforms,
                                                depth,
                                                kParityProbeEnvShadow,
                                                shadowRay,
                                                kEpsilon,
                                                kInfinity,
                                                occluded,
                                                shadowRec,
                                                swOccluded,
                                                swShadowRec,
                                                reasonMask);
                        }
#endif
                        float3 envDirection = envSample.direction;
                        float3 chainWeight = float3(1.0f);
                        bool usedDeltaChain = false;
                        bool connected = !occluded;

#if ENABLE_MNEE_CAUSTICS
                        bool useMneeEnvChain = (uniforms.enableMnee != 0u) &&
                                               (uniforms.enableMneeSecondary != 0u) &&
                                               softwareTrianglesAvailable;
                        if (!connected && useMneeEnvChain) {
                            if (stats) {
                                atomic_fetch_add_explicit(&stats->mneeEnvAttemptCount, 1u, memory_order_relaxed);
                            }
                            connected = trace_environment_delta_chain_software(uniforms,
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
                                                                               materials,
                                                                               stats,
                                                                               state,
                                                                               clampParams,
                                                                               specularOnly,
                                                                               shadowRay,
                                                                               envDirection,
                                                                               chainWeight,
                                                                               usedDeltaChain);
                        }
#endif
                        if (connected) {
                            float3 envRadiance = envSample.radiance;
                            if (usedDeltaChain) {
                                envRadiance = environment_color(environmentTexture,
                                                                envDirection,
                                                                uniforms.environmentRotation,
                                                                uniforms.environmentIntensity,
                                                                uniforms);
                            }
                            float3 contribution = envRadiance * bsdfValue * nDotL;
                            contribution *= weight / envSample.pdf;
                            contribution *= chainWeight;
                            if (all(isfinite(contribution))) {
                                radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                                if (debugContext) {
                                    record_debug_event(*debugContext,
                                                       kDebugEventEnvNee,
                                                       depth,
                                                       mediumDepth,
                                                       mediumDepth,
                                                       /*mediumEvent=*/0,
                                                       rec.frontFace,
                                                       rec.materialIndex,
                                                       /*scatterIsDelta=*/false,
                                                       throughput,
                                                       float4(envSample.pdf,
                                                              bsdfPdf,
                                                              weight,
                                                              nDotL),
                                                       contribution);
                                }
                                if (usedDeltaChain && stats) {
                                    atomic_fetch_add_explicit(&stats->mneeEnvAddedCount, 1u, memory_order_relaxed);
                                    stats_add_mnee_luma(stats, contribution);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (debugContext) {
            record_debug_event(*debugContext,
                               kDebugEventShadingNormal,
                               depth,
                               mediumDepth,
                               mediumDepth,
                               /*mediumEvent=*/0,
                               rec.frontFace,
                               rec.materialIndex,
                               /*scatterIsDelta=*/false,
                               throughput,
                               float4(rec.t,
                                      float(type),
                                      0.0f,
                                      0.0f),
                               shadingNormal);
        }

        BsdfSampleResult bsdfSample;
        uint rngStateBeforeBsdf = state;
        bool usedRandomWalk = false;
        bool enableRandomWalk = material_is_subsurface(material) &&
                                sss_use_random_walk(uniforms.sssMode, material) &&
                                rec.frontFace != 0u;
        if (enableRandomWalk) {
#if PT_DEBUG_TOOLS
            if (forceSoftware) {
                bsdfSample = sample_sss_random_walk_software(uniforms,
                                                             material,
                                                             rec,
                                                             wo,
                                                             incidentDir,
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
                                                             state,
                                                             clampParams);
            } else {
#endif
                bsdfSample = sample_sss_random_walk_hardware(uniforms,
                                                             material,
                                                             rec,
                                                             wo,
                                                             incidentDir,
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
                                                             state,
                                                             clampParams);
#if PT_DEBUG_TOOLS
            }
#endif
            usedRandomWalk = (bsdfSample.pdf > 0.0f);
        }
        if (!usedRandomWalk) {
            bsdfSample = material_closure_sample(material,
                                                 rec.point,
                                                 shadingNormal,
                                                 wo,
                                                 incidentDir,
                                                 rec.frontFace != 0u,
                                                 state,
                                                 clampParams,
                                                 uniforms.sssMode,
                                                 diffuseOcclusion,
                                     specularOnly);
        }
        apply_restir_gi_diffuse_prototype(uniforms,
                                          material,
                                          rec.point,
                                          shadingNormal,
                                          wo,
                                          incidentDir,
                                          rec.frontFace != 0u,
                                          clampParams,
                                          diffuseOcclusion,
                                          specularOnly,
                                          depth,
                                          pixelCoord,
                                          restirPtReservoirs,
                                          stats,
                                          state,
                                          bsdfSample);
        apply_path_guiding_prototype(uniforms,
                                     material,
                                     rec.point,
                                     shadingNormal,
                                     wo,
                                     clampParams,
                                     diffuseOcclusion,
                                     specularOnly,
                                     depth,
                                     pixelCoord,
                                     pathGuidingStates,
                                     stats,
                                     state,
                                     bsdfSample);
        apply_restir_pt_experimental_path_reuse(uniforms,
                                                material,
                                                rec.point,
                                                shadingNormal,
                                                wo,
                                                throughput,
                                                clampParams,
                                                diffuseOcclusion,
                                                specularOnly,
                                                depth,
                                                pixelCoord,
                                                restirPtReservoirs,
                                                stats,
                                                bsdfSample);
        capture_restir_pt_research_scaffold(uniforms,
                                            material,
                                            rec.point,
                                            shadingNormal,
                                            throughput,
                                            specularOnly,
                                            depth,
                                            pixelCoord,
                                            restirPtReservoirs,
                                            stats,
                                            bsdfSample);
        uint rngStateAfterBsdf = state;
        if (debugContext) {
            record_debug_event(*debugContext,
                               kDebugEventBsdfRng,
                               depth,
                               mediumDepth,
                               mediumDepth,
                               /*mediumEvent=*/0,
                               rec.frontFace,
                               rec.materialIndex,
                               bsdfSample.isDelta,
                               shadingNormal,
                               float4(as_type<float>(rngStateBeforeBsdf),
                                      as_type<float>(rngStateAfterBsdf),
                                      float(type),
                                      usedRandomWalk ? 1.0f : 0.0f),
                               wo);
        }
        if (debugContext) {
            record_debug_event(*debugContext,
                               kDebugEventBsdfState,
                               depth,
                               mediumDepth,
                               mediumDepth,
                               /*mediumEvent=*/0,
                               rec.frontFace,
                               rec.materialIndex,
                               bsdfSample.isDelta,
                               incidentDir,
                               float4(bsdfSample.pdf,
                                      bsdfSample.directionalPdf,
                                      float(bsdfSample.lobeType),
                                      bsdfSample.isDelta ? 1.0f : 0.0f),
                               bsdfSample.direction);
        }
        if (bsdfSample.pdf <= 0.0f) {
            break;
        }
        caustic_transport_note_path_vertex(uniforms,
                                           material,
                                           rec.point,
                                           shadingNormal,
                                           throughput,
                                           depth,
                                           bsdfSample,
                                           stats);
        if (radiance_cache_query_and_maybe_terminate(uniforms,
                                                     material,
                                                     rec.point,
                                                     shadingNormal,
                                                     specularOnly,
                                                     depth,
                                                     bsdfSample,
                                                     radianceCacheStates,
                                                     stats,
                                                     radiance,
                                                     throughput)) {
            break;
        }
        radiance_cache_train(radianceCacheStates,
                             uniforms,
                             material,
                             rec.point,
                             shadingNormal,
                             specularOnly,
                             depth,
                             bsdfSample,
                             stats);

        uint mediumDepthBefore = mediumDepth;
        if (bsdfSample.mediumEvent == 1) {
            float3 sigma = dielectric_sigma_a(material);
            sigma = max(sigma, float3(0.0f));
            if (mediumDepth < kMaxMediumStack) {
                mediumSigmaStack[mediumDepth] = sigma;
                mediumDepth += 1u;
            } else {
                mediumSigmaStack[kMaxMediumStack - 1u] = sigma;
            }
        } else if (bsdfSample.mediumEvent == -1) {
            if (mediumDepth > 0u) {
                mediumDepth -= 1u;
            }
        }
        volume_note_boundary_event(bsdfSample.mediumEvent, stats);
        uint mediumDepthAfter = mediumDepth;

        if (debugContext) {
            record_debug_event(*debugContext,
                               kDebugEventScatter,
                               depth,
                               mediumDepthBefore,
                               mediumDepthAfter,
                               bsdfSample.mediumEvent,
                               rec.frontFace,
                               rec.materialIndex,
                               bsdfSample.isDelta,
                               throughput,
                               float4(bsdfSample.pdf,
                                      bsdfSample.directionalPdf,
                                      rec.t,
                                      float(bsdfSample.lobeType)),
                               bsdfSample.weight);
        }

        bool causticCandidate = (!surfaceIsDelta) && (specularDepth > 0u);
        uint nextSpecularDepth = bsdfSample.isDelta ? (specularDepth + 1u) : 0u;
        bool didTransmission = false;
        if (bsdfSample.isDelta && type == 2u) {
            float3 dir = bsdfSample.direction;
            if (all(isfinite(dir)) && dot(dir, dir) > 0.0f) {
                float side = (rec.frontFace != 0u) ? 1.0f : -1.0f;
                didTransmission = (dot(shadingNormal, dir) * side) < 0.0f;
            }
        }
        if (didTransmission) {
            hadTransmission = true;
        }
        specularDepth = nextSpecularDepth;
        (void)causticCandidate;

        float3 nextOrigin;
        if (bsdfSample.hasExitPoint) {
            float3 exitNormal = bsdfSample.exitNormal;
            bool normalValid = all(isfinite(exitNormal)) && dot(exitNormal, exitNormal) > 0.0f;
            if (!normalValid) {
                exitNormal = rec.normal;
                normalValid = all(isfinite(exitNormal)) && dot(exitNormal, exitNormal) > 0.0f;
            }
            if (!normalValid) {
                exitNormal = float3(0.0f, 1.0f, 0.0f);
            }
            exitNormal = normalize(exitNormal);
            nextOrigin = offset_surface_point(bsdfSample.exitPoint, exitNormal, bsdfSample.direction);
            // Match software/HWRT parity: bias exit points to avoid self-occlusion in HWRT.
            float normalBias = max(kHardwareOcclusionEpsilon * 4.0f, kRayOriginEpsilon * 32.0f);
            float directionalBias = max(kHardwareOcclusionEpsilon * 8.0f, kRayOriginEpsilon * 32.0f);
            if (uniforms.hardwareExitNormalBias > 0.0f) {
                normalBias = max(normalBias, uniforms.hardwareExitNormalBias);
            }
            if (uniforms.hardwareExitDirectionalBias > 0.0f) {
                directionalBias = max(directionalBias, uniforms.hardwareExitDirectionalBias);
            }
            nextOrigin += exitNormal * normalBias;
            float3 dir = bsdfSample.direction;
            if (!all(isfinite(dir)) || dot(dir, dir) <= 0.0f) {
                dir = exitNormal;
            } else {
                dir = normalize(dir);
            }
            nextOrigin += dir * directionalBias;
        } else {
            nextOrigin = offset_ray_origin(rec, bsdfSample.direction);
        }

        bool useMnee = (ENABLE_MNEE_CAUSTICS != 0) && (uniforms.enableMnee != 0u);
        bool specNeeEnabled = (uniforms.enableSpecularNee != 0u);
        float dirLenSq = dot(bsdfSample.direction, bsdfSample.direction);
        bool specDirectionValid = (dirLenSq > 0.0f) && all(isfinite(bsdfSample.direction));
        bool mneeEligible = false;
#if ENABLE_MNEE_CAUSTICS
        mneeEligible = useMnee &&
                       bsdfSample.isDelta &&
                       ((bsdfSample.mediumEvent <= 0) || didTransmission) &&
                       (type == 2u) &&
                       (nextSpecularDepth == 1u) &&
                       specDirectionValid;
#endif
        if (mneeEligible) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEligibleCount, 1u, memory_order_relaxed);
            }
        }
        bool specNeeEligible = specNeeEnabled &&
                               bsdfSample.isDelta &&
                               (bsdfSample.mediumEvent <= 0) &&
                               specDirectionValid &&
                               !mneeEligible;

        if (specNeeEligible && envSampling &&
            environmentTexture.get_width() > 0 &&
            environmentTexture.get_height() > 0) {
            Ray neeRay;
            neeRay.origin = nextOrigin;
            neeRay.direction = normalize(bsdfSample.direction);
            HitRecord shadowRec;
            uint neeExcludeMesh = kInvalidIndex;
            uint neeExcludePrim = kInvalidIndex;
#if !PT_MNEE_OCCLUSION_PARITY
            if (didTransmission) {
                compute_exclusion_indices(rec, neeExcludeMesh, neeExcludePrim);
                neeExcludePrim = kInvalidIndex;
            }
#endif
            bool occluded = false;
#if PT_DEBUG_TOOLS
            if (forceSoftware) {
                occluded = trace_scene_software(uniforms,
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
                                                neeRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/true,
                                                /*includeTriangles=*/true,
                                                shadowRec);
            } else {
#endif
                occluded = trace_scene_hardware(uniforms,
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
                                                neeRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/true,
                                                neeExcludeMesh,
                                                neeExcludePrim,
                                                shadowRec);
#if PT_DEBUG_TOOLS
            }
            if (doParity) {
                HitRecord swShadowRec = make_empty_hit_record();
                bool swOccluded = trace_scene_software(uniforms,
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
                                                       /*stats=*/nullptr,
                                                       neeRay,
                                                       kEpsilon,
                                                       kInfinity,
                                                       /*anyHitOnly=*/true,
                                                       /*includeTriangles=*/true,
                                                       swShadowRec);
                uint reasonMask = 0u;
                if (occluded != swOccluded) {
                    reasonMask |= kParityReasonHitMiss;
                }
                if (occluded && swOccluded) {
                    float epsT = max(1.0e-3f, 1.0e-4f * fabs(shadowRec.t));
                    if (fabs(shadowRec.t - swShadowRec.t) > epsT) {
                        reasonMask |= kParityReasonT;
                    }
                    if (shadowRec.frontFace != swShadowRec.frontFace) {
                        reasonMask |= kParityReasonFrontFace;
                    }
                    if (shadowRec.materialIndex != swShadowRec.materialIndex ||
                        shadowRec.meshIndex != swShadowRec.meshIndex ||
                        shadowRec.primitiveIndex != swShadowRec.primitiveIndex) {
                        reasonMask |= kParityReasonId;
                    }
                    float3 hwN = shadowRec.normal;
                    float3 swN = swShadowRec.normal;
                    if (all(isfinite(hwN)) && all(isfinite(swN)) &&
                        dot(hwN, hwN) > 0.0f && dot(swN, swN) > 0.0f &&
                        dot(normalize(hwN), normalize(swN)) < 0.99f) {
                        reasonMask |= kParityReasonNormal;
                    }
                }
                record_parity_entry(*debugContext,
                                    uniforms,
                                    depth,
                                    kParityProbeSpecEnv,
                                    neeRay,
                                    kEpsilon,
                                    kInfinity,
                                    occluded,
                                    shadowRec,
                                    swOccluded,
                                    swShadowRec,
                                    reasonMask);
            }
#endif
            if (!occluded) {
                float envPdf = environment_pdf(uniforms, environmentPdf, neeRay.direction);
                envPdf = max(envPdf, kSpecularNeePdfFloor);
                float invEnvPdf = min(1.0f / envPdf, kSpecularNeeInvPdfClamp);
                float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                float denom = envPdf + bsdfPdf;
                float misWeight = (denom > 0.0f) ? (envPdf / denom) : 0.0f;
                misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                float3 envColor = environment_color(environmentTexture,
                                                    neeRay.direction,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity,
                                                    uniforms);
                float3 neeContribution = bsdfSample.weight * envColor * (misWeight * invEnvPdf);
                if (all(isfinite(neeContribution))) {
                    radiance += clamp_firefly_contribution(throughput, neeContribution, clampParams);
                    if (debugContext) {
                        record_debug_event(*debugContext,
                                           kDebugEventSpecEnvNee,
                                           depth,
                                           mediumDepth,
                                           mediumDepth,
                                           /*mediumEvent=*/0,
                                           rec.frontFace,
                                           rec.materialIndex,
                                           bsdfSample.isDelta,
                                           throughput,
                                           float4(envPdf,
                                                  bsdfPdf,
                                                  misWeight,
                                                  invEnvPdf),
                                           neeContribution);
                    }
                    if (stats) {
                        atomic_fetch_add_explicit(&stats->specNeeEnvAddedCount, 1u, memory_order_relaxed);
                    }
                }
            }
        }

        if (specNeeEligible && rectLightCount > 0u) {
            Ray neeRay;
            neeRay.origin = nextOrigin;
            neeRay.direction = normalize(bsdfSample.direction);
            HitRecord lightRec = make_empty_hit_record();
            uint neeExcludeMesh = kInvalidIndex;
            uint neeExcludePrim = kInvalidIndex;
            if (didTransmission) {
                compute_exclusion_indices(rec, neeExcludeMesh, neeExcludePrim);
                neeExcludePrim = kInvalidIndex;
            }
            bool hitLight = false;
#if PT_DEBUG_TOOLS
            if (forceSoftware) {
                hitLight = trace_scene_software(uniforms,
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
                                                neeRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/false,
                                                /*includeTriangles=*/true,
                                                lightRec);
            } else {
#endif
                hitLight = trace_scene_hardware(uniforms,
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
                                                neeRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/false,
                                                neeExcludeMesh,
                                                neeExcludePrim,
                                                lightRec);
#if PT_DEBUG_TOOLS
            }
            if (doParity) {
                HitRecord swLightRec = make_empty_hit_record();
                bool swHitLight = trace_scene_software(uniforms,
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
                                                       /*stats=*/nullptr,
                                                       neeRay,
                                                       kEpsilon,
                                                       kInfinity,
                                                       /*anyHitOnly=*/false,
                                                       /*includeTriangles=*/true,
                                                       swLightRec);
                uint reasonMask = 0u;
                if (hitLight != swHitLight) {
                    reasonMask |= kParityReasonHitMiss;
                }
                if (hitLight && swHitLight) {
                    float epsT = max(1.0e-3f, 1.0e-4f * fabs(lightRec.t));
                    if (fabs(lightRec.t - swLightRec.t) > epsT) {
                        reasonMask |= kParityReasonT;
                    }
                    if (lightRec.frontFace != swLightRec.frontFace) {
                        reasonMask |= kParityReasonFrontFace;
                    }
                    if (lightRec.materialIndex != swLightRec.materialIndex ||
                        lightRec.meshIndex != swLightRec.meshIndex ||
                        lightRec.primitiveIndex != swLightRec.primitiveIndex) {
                        reasonMask |= kParityReasonId;
                    }
                    float3 hwN = lightRec.normal;
                    float3 swN = swLightRec.normal;
                    if (all(isfinite(hwN)) && all(isfinite(swN)) &&
                        dot(hwN, hwN) > 0.0f && dot(swN, swN) > 0.0f &&
                        dot(normalize(hwN), normalize(swN)) < 0.99f) {
                        reasonMask |= kParityReasonNormal;
                    }
                }
                record_parity_entry(*debugContext,
                                    uniforms,
                                    depth,
                                    kParityProbeSpecRect,
                                    neeRay,
                                    kEpsilon,
                                    kInfinity,
                                    hitLight,
                                    lightRec,
                                    swHitLight,
                                    swLightRec,
                                    reasonMask);
            }
#endif
            if (hitLight) {
                MneeRectHit mneeHit;
                if (mnee_rect_light_hit(uniforms,
                                        rectangles,
                                        materials,
                                        environmentTexture,
                                        rectLightCount,
                                        lightRec,
                                        nextOrigin,
                                        mneeHit)) {
                    float lightPdf = max(mneeHit.pdf, kSpecularNeePdfFloor);
                    float invLightPdf = min(1.0f / lightPdf, kSpecularNeeInvPdfClamp);
                    float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                    float denom = lightPdf + bsdfPdf;
                    float misWeight = (denom > 0.0f) ? (lightPdf / denom) : 0.0f;
                    misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                    float3 contribution = bsdfSample.weight * mneeHit.emission *
                                          (misWeight * invLightPdf);
                    if (all(isfinite(contribution))) {
                        radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                        if (debugContext) {
                            record_debug_event(*debugContext,
                                               kDebugEventSpecRectNee,
                                               depth,
                                               mediumDepth,
                                               mediumDepth,
                                               /*mediumEvent=*/0,
                                               rec.frontFace,
                                               rec.materialIndex,
                                               bsdfSample.isDelta,
                                               throughput,
                                               float4(lightPdf,
                                                      bsdfPdf,
                                                      misWeight,
                                                      invLightPdf),
                                               contribution);
                        }
                        if (stats) {
                            atomic_fetch_add_explicit(&stats->specNeeRectAddedCount, 1u, memory_order_relaxed);
                        }
                    }
                }
            }
        }

#if ENABLE_MNEE_CAUSTICS
        if (mneeEligible && envSampling &&
            environmentTexture.get_width() > 0 &&
            environmentTexture.get_height() > 0) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeEnvAttemptCount, 1u, memory_order_relaxed);
            }
            Ray mneeRay;
            mneeRay.origin = nextOrigin;
            mneeRay.direction = normalize(bsdfSample.direction);
            HitRecord shadowRec;
            uint neeExcludeMesh = kInvalidIndex;
            uint neeExcludePrim = kInvalidIndex;
#if !PT_MNEE_OCCLUSION_PARITY
            if (didTransmission) {
                compute_exclusion_indices(rec, neeExcludeMesh, neeExcludePrim);
                neeExcludePrim = kInvalidIndex;
            }
#endif
            bool occluded = false;
#if PT_MNEE_SWRT_RAYS
            occluded = trace_scene_software(uniforms,
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
                                            mneeRay,
                                            kEpsilon,
                                            kInfinity,
                                            /*anyHitOnly=*/true,
                                            /*includeTriangles=*/true,
                                            shadowRec);
#else
#if PT_DEBUG_TOOLS
            if (forceSoftware) {
                occluded = trace_scene_software(uniforms,
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
                                                mneeRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/true,
                                                /*includeTriangles=*/true,
                                                shadowRec);
            } else {
#endif
#if PT_MNEE_OCCLUSION_PARITY
                bool occludedHw = trace_scene_hardware(uniforms,
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
                                                       mneeRay,
                                                       kEpsilon,
                                                       kInfinity,
                                                       /*anyHitOnly=*/true,
                                                       neeExcludeMesh,
                                                       neeExcludePrim,
                                                       shadowRec);
                HitRecord shadowRecSw;
                bool occludedSw = trace_scene_software(uniforms,
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
                                                       mneeRay,
                                                       kEpsilon,
                                                       kInfinity,
                                                       /*anyHitOnly=*/true,
                                                       /*includeTriangles=*/true,
                                                       shadowRecSw);
                if (stats) {
                    if (occludedHw) {
                        atomic_fetch_add_explicit(&stats->mneeEnvHwOccludedCount, 1u, memory_order_relaxed);
                    }
                    if (occludedSw) {
                        atomic_fetch_add_explicit(&stats->mneeEnvSwOccludedCount, 1u, memory_order_relaxed);
                    }
                    if (occludedHw != occludedSw) {
                        atomic_fetch_add_explicit(&stats->mneeEnvHwSwMismatchCount, 1u, memory_order_relaxed);
                    }
                }
                occluded = occludedHw;
#else
                occluded = trace_scene_hardware(uniforms,
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
                                                mneeRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/true,
                                                neeExcludeMesh,
                                                neeExcludePrim,
                                                shadowRec);
#endif
#if PT_DEBUG_TOOLS
            }
#endif
#endif
            if (!occluded) {
                float envPdf = environment_pdf(uniforms, environmentPdf, mneeRay.direction);
                envPdf = max(envPdf, kSpecularNeePdfFloor);
                float invEnvPdf = min(1.0f / envPdf, kSpecularNeeInvPdfClamp);
                float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                float denom = envPdf + bsdfPdf;
                float misWeight = (denom > 0.0f) ? (envPdf / denom) : 0.0f;
                misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                float3 envColor = environment_color(environmentTexture,
                                                    mneeRay.direction,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity,
                                                    uniforms);
                float3 neeContribution = bsdfSample.weight * envColor * (misWeight * invEnvPdf);
                if (all(isfinite(neeContribution))) {
                    radiance += clamp_firefly_contribution(throughput, neeContribution, clampParams);
                    if (stats) {
                        atomic_fetch_add_explicit(&stats->mneeEnvAddedCount, 1u, memory_order_relaxed);
                        stats_add_mnee_luma(stats, neeContribution);
                    }
                }
            }
        }

        if (mneeEligible && rectLightCount > 0u) {
            if (stats) {
                atomic_fetch_add_explicit(&stats->mneeRectAttemptCount, 1u, memory_order_relaxed);
            }
            Ray mneeRay;
            mneeRay.origin = nextOrigin;
            mneeRay.direction = normalize(bsdfSample.direction);
            HitRecord lightRec;
            bool hitLight = false;
            MneeRectHit mneeHit;
            bool mneeLight = false;
#if PT_MNEE_OCCLUSION_PARITY
            HitRecord lightRecSw;
            MneeRectHit mneeHitSw;
#endif
            uint neeExcludeMesh = kInvalidIndex;
            uint neeExcludePrim = kInvalidIndex;
            if (didTransmission) {
                compute_exclusion_indices(rec, neeExcludeMesh, neeExcludePrim);
                neeExcludePrim = kInvalidIndex;
            }
#if PT_MNEE_SWRT_RAYS
            hitLight = trace_scene_software(uniforms,
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
                                            mneeRay,
                                            kEpsilon,
                                            kInfinity,
                                            /*anyHitOnly=*/false,
                                            /*includeTriangles=*/true,
                                            lightRec);
#else
#if PT_DEBUG_TOOLS
            if (forceSoftware) {
                hitLight = trace_scene_software(uniforms,
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
                                                mneeRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/false,
                                                /*includeTriangles=*/true,
                                                lightRec);
            } else {
#endif
#if PT_MNEE_OCCLUSION_PARITY
                bool hitLightHw = trace_scene_hardware(uniforms,
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
                                                       mneeRay,
                                                       kEpsilon,
                                                       kInfinity,
                                                       /*anyHitOnly=*/false,
                                                       neeExcludeMesh,
                                                       neeExcludePrim,
                                                       lightRec);
                bool hitLightSw = trace_scene_software(uniforms,
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
                                                       mneeRay,
                                                       kEpsilon,
                                                       kInfinity,
                                                       /*anyHitOnly=*/false,
                                                       /*includeTriangles=*/true,
                                                       lightRecSw);
                bool hwMneeLight = false;
                bool swMneeLight = false;
                if (hitLightHw) {
                    hwMneeLight = mnee_rect_light_hit(uniforms,
                                                      rectangles,
                                                      materials,
                                                      environmentTexture,
                                                      rectLightCount,
                                                      lightRec,
                                                      nextOrigin,
                                                      mneeHit);
                }
                if (hitLightSw) {
                    swMneeLight = mnee_rect_light_hit(uniforms,
                                                      rectangles,
                                                      materials,
                                                      environmentTexture,
                                                      rectLightCount,
                                                      lightRecSw,
                                                      nextOrigin,
                                                      mneeHitSw);
                }
                if (stats) {
                    if (hwMneeLight) {
                        atomic_fetch_add_explicit(&stats->mneeRectHwOccludedCount, 1u, memory_order_relaxed);
                    }
                    if (swMneeLight) {
                        atomic_fetch_add_explicit(&stats->mneeRectSwOccludedCount, 1u, memory_order_relaxed);
                    }
                    if (hwMneeLight != swMneeLight) {
                        atomic_fetch_add_explicit(&stats->mneeRectHwSwMismatchCount, 1u, memory_order_relaxed);
                    }
                }
                hitLight = hitLightHw;
                mneeLight = hwMneeLight;
#else
                hitLight = trace_scene_hardware(uniforms,
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
                                                mneeRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/false,
                                                neeExcludeMesh,
                                                neeExcludePrim,
                                                lightRec);
#endif
#if PT_DEBUG_TOOLS
            }
#endif
#endif
            if (!mneeLight && hitLight) {
                mneeLight = mnee_rect_light_hit(uniforms,
                                                rectangles,
                                                materials,
                                                environmentTexture,
                                                rectLightCount,
                                                lightRec,
                                                nextOrigin,
                                                mneeHit);
            }
            if (mneeLight) {
                float lightPdf = max(mneeHit.pdf, kSpecularNeePdfFloor);
                float invLightPdf = min(1.0f / lightPdf, kSpecularNeeInvPdfClamp);
                float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                float denom = lightPdf + bsdfPdf;
                float misWeight = (denom > 0.0f) ? (lightPdf / denom) : 0.0f;
                misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                float3 contribution = bsdfSample.weight * mneeHit.emission *
                                      (misWeight * invLightPdf);
                if (all(isfinite(contribution))) {
                    radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                    if (stats) {
                        atomic_fetch_add_explicit(&stats->mneeRectAddedCount, 1u, memory_order_relaxed);
                        stats_add_mnee_luma(stats, contribution);
                    }
                }
            }
        }

        if (mneeEligible && uniforms.enableMneeSecondary != 0u) {
            Ray chainRay;
            chainRay.origin = nextOrigin;
            chainRay.direction = normalize(bsdfSample.direction);
            HitRecord chainRec;
            uint chainExcludeMesh = kInvalidIndex;
            uint chainExcludePrim = kInvalidIndex;
            bool chainHit = false;
#if PT_MNEE_SWRT_RAYS
            chainHit = trace_scene_software(uniforms,
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
                                            chainRay,
                                            kEpsilon,
                                            kInfinity,
                                            /*anyHitOnly=*/false,
                                            /*includeTriangles=*/true,
                                            chainRec);
#else
#if PT_DEBUG_TOOLS
            if (forceSoftware) {
                chainHit = trace_scene_software(uniforms,
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
                                                chainRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/false,
                                                /*includeTriangles=*/true,
                                                chainRec);
            } else {
#endif
                chainHit = trace_scene_hardware(uniforms,
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
                                                chainRay,
                                                kEpsilon,
                                                kInfinity,
                                                /*anyHitOnly=*/false,
                                                chainExcludeMesh,
                                                chainExcludePrim,
                                                chainRec);
#if PT_DEBUG_TOOLS
            }
#endif
#endif
            if (chainHit && materials && uniforms.materialCount > 0u) {
                bool chainHitIsLight = false;
                if (rectLightCount > 0u) {
                    MneeRectHit chainLightHit;
                    if (mnee_rect_light_hit(uniforms,
                                            rectangles,
                                            materials,
                                            environmentTexture,
                                            rectLightCount,
                                            chainRec,
                                            chainRay.origin,
                                            chainLightHit)) {
                        chainHitIsLight = true;
                    }
                }
                if (!chainHitIsLight) {
                    uint chainMatIndex = min(chainRec.materialIndex, uniforms.materialCount - 1u);
                    MaterialData chainMaterial = materials[chainMatIndex];
                    if (material_is_delta(chainMaterial)) {
                        float3 chainNormal = chainRec.normal;
                        if (!all(isfinite(chainNormal)) || dot(chainNormal, chainNormal) <= 0.0f) {
                            chainNormal = float3(0.0f, 1.0f, 0.0f);
                        }
                        chainNormal = normalize(chainNormal);
                        float3 chainIncident = normalize(chainRay.direction);
                        float3 chainWo = -chainIncident;
                        uint chainState = state;
                        BsdfSampleResult chainSample = material_closure_sample(chainMaterial,
                                                                               chainRec.point,
                                                                               chainNormal,
                                                                               chainWo,
                                                                               chainIncident,
                                                                               chainRec.frontFace != 0u,
                                                                               chainState,
                                                                               clampParams,
                                                                               uniforms.sssMode,
                                                                   1.0f,
                                                                   specularOnly);
                        if (chainSample.pdf > 0.0f &&
                            chainSample.isDelta &&
                            (chainSample.mediumEvent <= 0)) {
                            float3 chainDir = safe_normalize(chainSample.direction);
                            if (all(isfinite(chainDir)) && dot(chainDir, chainDir) > 0.0f) {
                                float3 chainOrigin = offset_ray_origin(chainRec, chainDir);
                                float3 combinedWeight = bsdfSample.weight * chainSample.weight;
                                float bsdfPdf = max(bsdfSample.directionalPdf * chainSample.directionalPdf,
                                                    kSpecularNeePdfFloor);
                                if (envSampling &&
                                    environmentTexture.get_width() > 0 &&
                                    environmentTexture.get_height() > 0) {
                                    if (stats) {
                                        atomic_fetch_add_explicit(&stats->mneeEnvAttemptCount, 1u, memory_order_relaxed);
                                    }
                                    Ray envRay;
                                    envRay.origin = chainOrigin;
                                    envRay.direction = normalize(chainDir);
                                    HitRecord envRec;
                                    uint chainOccMesh = kInvalidIndex;
                                    uint chainOccPrim = kInvalidIndex;
                                    bool occluded = false;
#if PT_MNEE_SWRT_RAYS
                                    occluded = trace_scene_software(uniforms,
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
                                                                    envRay,
                                                                    kEpsilon,
                                                                    kInfinity,
                                                                    /*anyHitOnly=*/true,
                                                                    /*includeTriangles=*/true,
                                                                    envRec);
#else
#if PT_DEBUG_TOOLS
                                    if (forceSoftware) {
                                        occluded = trace_scene_software(uniforms,
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
                                                                        envRay,
                                                                        kEpsilon,
                                                                        kInfinity,
                                                                        /*anyHitOnly=*/true,
                                                                        /*includeTriangles=*/true,
                                                                        envRec);
                                    } else {
#endif
#if PT_MNEE_OCCLUSION_PARITY
                                        bool occludedHw = trace_scene_hardware(uniforms,
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
                                                                               envRay,
                                                                               kEpsilon,
                                                                               kInfinity,
                                                                               /*anyHitOnly=*/true,
                                                                               chainOccMesh,
                                                                               chainOccPrim,
                                                                               envRec);
                                        HitRecord envRecSw;
                                        bool occludedSw = trace_scene_software(uniforms,
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
                                                                               envRay,
                                                                               kEpsilon,
                                                                               kInfinity,
                                                                               /*anyHitOnly=*/true,
                                                                               /*includeTriangles=*/true,
                                                                               envRecSw);
                                        if (stats) {
                                            if (occludedHw) {
                                                atomic_fetch_add_explicit(&stats->mneeEnvHwOccludedCount, 1u, memory_order_relaxed);
                                            }
                                            if (occludedSw) {
                                                atomic_fetch_add_explicit(&stats->mneeEnvSwOccludedCount, 1u, memory_order_relaxed);
                                            }
                                            if (occludedHw != occludedSw) {
                                                atomic_fetch_add_explicit(&stats->mneeEnvHwSwMismatchCount, 1u, memory_order_relaxed);
                                            }
                                        }
                                        occluded = occludedHw;
#else
                                        occluded = trace_scene_hardware(uniforms,
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
                                                                        envRay,
                                                                        kEpsilon,
                                                                        kInfinity,
                                                                        /*anyHitOnly=*/true,
                                                                        chainOccMesh,
                                                                        chainOccPrim,
                                                                        envRec);
#endif
#if PT_DEBUG_TOOLS
                                    }
#endif
#endif
                                    if (!occluded) {
                                        float envPdf = environment_pdf(uniforms, environmentPdf, envRay.direction);
                                        envPdf = max(envPdf, kSpecularNeePdfFloor);
                                        float invEnvPdf = min(1.0f / envPdf, kSpecularNeeInvPdfClamp);
                                        float denom = envPdf + bsdfPdf;
                                        float misWeight = (denom > 0.0f) ? (envPdf / denom) : 0.0f;
                                        misWeight = clamp(misWeight,
                                                          kMisWeightClampMin,
                                                          kMisWeightClampMax);
                                        float3 envColor = environment_color(environmentTexture,
                                                                            envRay.direction,
                                                                            uniforms.environmentRotation,
                                                                            uniforms.environmentIntensity,
                                                                            uniforms);
                                        float3 contribution = combinedWeight * envColor *
                                                              (misWeight * invEnvPdf);
                                        if (all(isfinite(contribution))) {
                                            radiance += clamp_firefly_contribution(throughput,
                                                                                   contribution,
                                                                                   clampParams);
                                            if (stats) {
                                                atomic_fetch_add_explicit(&stats->mneeEnvAddedCount, 1u, memory_order_relaxed);
                                                stats_add_mnee_luma(stats, contribution);
                                            }
                                        }
                                    }
                                }
                                if (rectLightCount > 0u) {
                                    if (stats) {
                                        atomic_fetch_add_explicit(&stats->mneeRectAttemptCount, 1u, memory_order_relaxed);
                                    }
                                    Ray lightRay;
                                    lightRay.origin = chainOrigin;
                                    lightRay.direction = normalize(chainDir);
                                    HitRecord lightRec;
                                    bool hitLight = false;
                                    MneeRectHit mneeHit;
                                    bool mneeLight = false;
#if PT_MNEE_OCCLUSION_PARITY
                                    HitRecord lightRecSw;
                                    MneeRectHit mneeHitSw;
#endif
                                    uint chainLightMesh = kInvalidIndex;
                                    uint chainLightPrim = kInvalidIndex;
#if PT_MNEE_SWRT_RAYS
                                    hitLight = trace_scene_software(uniforms,
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
                                                                    lightRay,
                                                                    kEpsilon,
                                                                    kInfinity,
                                                                    /*anyHitOnly=*/false,
                                                                    /*includeTriangles=*/true,
                                                                    lightRec);
#else
#if PT_DEBUG_TOOLS
                                    if (forceSoftware) {
                                        hitLight = trace_scene_software(uniforms,
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
                                                                        lightRay,
                                                                        kEpsilon,
                                                                        kInfinity,
                                                                        /*anyHitOnly=*/false,
                                                                        /*includeTriangles=*/true,
                                                                        lightRec);
                                    } else {
#endif
#if PT_MNEE_OCCLUSION_PARITY
                                        bool hitLightHw = trace_scene_hardware(uniforms,
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
                                                                               lightRay,
                                                                               kEpsilon,
                                                                               kInfinity,
                                                                               /*anyHitOnly=*/false,
                                                                               chainLightMesh,
                                                                               chainLightPrim,
                                                                               lightRec);
                                        bool hitLightSw = trace_scene_software(uniforms,
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
                                                                               lightRay,
                                                                               kEpsilon,
                                                                               kInfinity,
                                                                               /*anyHitOnly=*/false,
                                                                               /*includeTriangles=*/true,
                                                                               lightRecSw);
                                        bool hwMneeLight = false;
                                        bool swMneeLight = false;
                                        if (hitLightHw) {
                                            hwMneeLight = mnee_rect_light_hit(uniforms,
                                                                              rectangles,
                                                                              materials,
                                                                              environmentTexture,
                                                                              rectLightCount,
                                                                              lightRec,
                                                                              chainOrigin,
                                                                              mneeHit);
                                        }
                                        if (hitLightSw) {
                                            swMneeLight = mnee_rect_light_hit(uniforms,
                                                                              rectangles,
                                                                              materials,
                                                                              environmentTexture,
                                                                              rectLightCount,
                                                                              lightRecSw,
                                                                              chainOrigin,
                                                                              mneeHitSw);
                                        }
                                        if (stats) {
                                            if (hwMneeLight) {
                                                atomic_fetch_add_explicit(&stats->mneeRectHwOccludedCount, 1u, memory_order_relaxed);
                                            }
                                            if (swMneeLight) {
                                                atomic_fetch_add_explicit(&stats->mneeRectSwOccludedCount, 1u, memory_order_relaxed);
                                            }
                                            if (hwMneeLight != swMneeLight) {
                                                atomic_fetch_add_explicit(&stats->mneeRectHwSwMismatchCount, 1u, memory_order_relaxed);
                                            }
                                        }
                                        hitLight = hitLightHw;
                                        mneeLight = hwMneeLight;
#else
                                        hitLight = trace_scene_hardware(uniforms,
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
                                                                        lightRay,
                                                                        kEpsilon,
                                                                        kInfinity,
                                                                        /*anyHitOnly=*/false,
                                                                        chainLightMesh,
                                                                        chainLightPrim,
                                                                        lightRec);
#endif
#if PT_DEBUG_TOOLS
                                    }
#endif
#endif
                                    if (!mneeLight && hitLight) {
                                        mneeLight = mnee_rect_light_hit(uniforms,
                                                                        rectangles,
                                                                        materials,
                                                                        environmentTexture,
                                                                        rectLightCount,
                                                                        lightRec,
                                                                        chainOrigin,
                                                                        mneeHit);
                                    }
                                    if (mneeLight) {
                                        float lightPdf = max(mneeHit.pdf, kSpecularNeePdfFloor);
                                        float invLightPdf = min(1.0f / lightPdf, kSpecularNeeInvPdfClamp);
                                        float denom = lightPdf + bsdfPdf;
                                        float misWeight = (denom > 0.0f) ? (lightPdf / denom) : 0.0f;
                                        misWeight = clamp(misWeight,
                                                          kMisWeightClampMin,
                                                          kMisWeightClampMax);
                                        float3 contribution = combinedWeight * mneeHit.emission *
                                                              (misWeight * invLightPdf);
                                        if (all(isfinite(contribution))) {
                                            radiance += clamp_firefly_contribution(throughput,
                                                                                   contribution,
                                                                                   clampParams);
                                            if (stats) {
                                                atomic_fetch_add_explicit(&stats->mneeRectAddedCount, 1u, memory_order_relaxed);
                                                stats_add_mnee_luma(stats, contribution);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
#endif

        float3 throughputBeforeScatter = throughput;
        throughput *= bsdfSample.weight;
        throughput = clamp_path_throughput(throughput, clampParams);
        if (debugContext) {
            record_debug_event(*debugContext,
                               kDebugEventThroughput,
                               depth,
                               mediumDepthBefore,
                               mediumDepthAfter,
                               bsdfSample.mediumEvent,
                               rec.frontFace,
                               rec.materialIndex,
                               bsdfSample.isDelta,
                               throughputBeforeScatter,
                               float4(bsdfSample.pdf,
                                      bsdfSample.directionalPdf,
                                      rec.t,
                                      float(bsdfSample.lobeType)),
                               throughput);
        }

        if (!all(isfinite(throughput))) {
            break;
        }

        float maxThroughput = max(max(throughput.x, throughput.y), throughput.z);
        if (maxThroughput <= 0.0f) {
            break;
        }

        if (debugContext) {
            record_debug_event(*debugContext,
                               kDebugEventRay,
                               depth,
                               mediumDepthBefore,
                               mediumDepthAfter,
                               bsdfSample.mediumEvent,
                               rec.frontFace,
                               rec.materialIndex,
                               bsdfSample.isDelta,
                               throughput,
                               float4(bsdfSample.pdf,
                                      bsdfSample.directionalPdf,
                                      rec.t,
                                      float(bsdfSample.lobeType)),
                               bsdfSample.direction);
        }

        bool nextEnvLodActive = false;
        float nextEnvLod = 0.0f;
        if (bsdfSample.lobeType == 1u && !bsdfSample.isDelta) {
            float maxMip = environment_max_mip(environmentTexture);
            if (maxMip > 0.0f) {
                nextEnvLod = environment_lod_from_roughness(bsdfSample.lobeRoughness,
                                                            environmentTexture);
                nextEnvLodActive = true;
            }
        }
        envLodActive = nextEnvLodActive;
        envLod = nextEnvLod;

        rayCone.width = ray_cone_width_at_distance(rayCone, hitDistanceWorld);
        rayCone.spread = min(rayCone.spread +
                             bsdf_cone_spread_increment(bsdfSample.lobeType,
                                                        bsdfSample.lobeRoughness,
                                                        bsdfSample.isDelta),
                             1.5f);

        lastBsdfPdf = (bsdfSample.directionalPdf > 0.0f) ? bsdfSample.directionalPdf : bsdfSample.pdf;
        lastScatterWasDelta = bsdfSample.isDelta;
        ray.origin = nextOrigin;
        ray.direction = bsdfSample.direction;

        if (uniforms.useRussianRoulette != 0 && depth >= 5) {
            float continueProbability = clamp(maxThroughput, 0.05f, 0.95f);
            if (rand_uniform(state) > continueProbability) {
                break;
            }
            throughput /= continueProbability;
        }
    }

    #undef clamp_path_throughput
    #undef clamp_firefly_contribution
    return radiance;
}
#endif
