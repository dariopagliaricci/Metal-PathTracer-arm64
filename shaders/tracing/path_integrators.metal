constant uint kInvalidTextureIndex = 0xFFFFFFFFu;

// The Pavilion reference carries the maximum authored/textured roughness to
// every later surface, including PBR walls and plants, not only dielectrics.
// Do this after alpha acceptance and texture resolution so a rejected card
// cannot change the scattering model. Other asset profiles remain unchanged.
inline void regularize_pavilion_pbr_material(thread MaterialData& material,
                                             thread float& pathMaxRoughness,
                                             const bool frontFace) {
    if (static_cast<uint>(material.typeEta.x) == 7u &&
        (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u) {
        pathMaxRoughness = max(pathMaxRoughness,
                               clamp(material.baseColorRoughness.w, 0.0f, 1.0f));
        material.baseColorRoughness.w = max(pathMaxRoughness, 0.01f);
        material.pbrParams.y = material.baseColorRoughness.w;
        // Shading normals face wo on either side of a leaf/petal. The geometric
        // side, not the shading cosine, determines its reference relative IOR.
        const float ior = max(material.typeEta.y, 1.0f);
        material.typeEta.y = frontFace ? ior : 1.0f / ior;
    }
}

// Preserve the camera footprint only through the Pavilion's flat, smooth
// dielectric prefix. The reference propagates specular ray differentials before
// textureGrad; scalar cones overblur the wide, short green-marble texture.
inline bool propagate_pavilion_planar_delta_differentials(
    thread PrimaryRayDiff& diff,
    const Ray incomingRay,
    thread const HitRecord& rec,
    const MaterialData material,
    const BsdfSampleResult sample) {
    if ((material.materialFlags & kMaterialFlagReferenceDielectricContinuation) == 0u ||
        static_cast<uint>(material.typeEta.x) != 2u ||
        material.typeEta.w > 0.5f || !sample.isDelta || sample.hasExitPoint ||
        rec.primitiveType != kPrimitiveTypeTriangle) {
        return false;
    }
    const float3 N = safe_normalize(rec.normal);
    const float3 D = incomingRay.direction;
    const float lengthD = length(D);
    const float nDotD = dot(N, D);
    if (!isfinite(lengthD) || !(lengthD > 1.0e-6f) ||
        !isfinite(nDotD) || fabs(nDotD) < 1.0e-6f ||
        dot(N, safe_normalize(rec.shadingNormal)) < 0.9999f) {
        return false;
    }
    const float3 d = D / lengthD;
    const bool transmitted = dot(N, sample.direction) < 0.0f;
    const float ior = max(material.typeEta.y, 1.0f);
    const float eta = rec.frontFace != 0u ? 1.0f / ior : ior;
    const float c = -dot(N, d);
    const float ct = fabs(dot(N, sample.direction));
    if (transmitted && ct < 1.0e-4f) {
        return false;
    }
    float3 nextDO[2];
    float3 nextDD[2];
    for (uint axis = 0u; axis < 2u; ++axis) {
        const float3 dO = axis == 0u ? diff.dOdx : diff.dOdy;
        const float3 dD = axis == 0u ? diff.dDdx : diff.dDdy;
        const float3 X = dO + rec.t * dD;
        const float dt = -dot(N, X) / nDotD;
        const float3 dP = X + D * dt;
        const float3 dd = (dD - d * dot(d, dD)) / lengthD;
        const float dc = -dot(N, dd);
        const float3 outgoingDD = transmitted
            ? eta * dd + (eta - eta * eta * c / ct) * dc * N
            : dd - 2.0f * dot(N, dd) * N;
        // Differentiate the existing offset_ray_origin bias for a flat plane.
        const float side = transmitted ? -1.0f : 1.0f;
        const float offsetDt = rec.t > 1.0f ? 1.0e-4f * dt : 0.0f;
        nextDO[axis] = dP + side * N * offsetDt +
                       outgoingDD * (0.5f * kRayOriginEpsilon);
        nextDD[axis] = outgoingDD;
        if (!all(isfinite(nextDO[axis])) || !all(isfinite(nextDD[axis]))) {
            return false;
        }
    }
    diff.dOdx = nextDO[0];
    diff.dOdy = nextDO[1];
    diff.dDdx = nextDD[0];
    diff.dDdy = nextDD[1];
    return true;
}

// Alpha holes in the Pavilion are visibility events, like the reference
// renderer's any-hit rejection. Preserve the pixel differential when the ray
// origin moves to the rejected card so later cards keep the primary footprint.
inline void advance_pavilion_alpha_visibility_ray(thread Ray& ray,
                                                   thread PrimaryRayDiff& rayDiff,
                                                   thread RayCone& rayCone,
                                                   thread const HitRecord& rec) {
    const float denom = dot(rec.normal, ray.direction);
    if (isfinite(denom) && fabs(denom) > 1.0e-6f) {
        const float3 dx = rayDiff.dOdx + rec.t * rayDiff.dDdx;
        const float3 dy = rayDiff.dOdy + rec.t * rayDiff.dDdy;
        rayDiff.dOdx = dx - ray.direction * (dot(rec.normal, dx) / denom);
        rayDiff.dOdy = dy - ray.direction * (dot(rec.normal, dy) / denom);
    }
    rayCone.width = ray_cone_width_at_distance(rayCone,
                                               ray_segment_world_length(ray, rec.t));
    ray.origin = offset_ray_origin(rec, ray.direction);
}

inline float3 reconstruction_split_record(const float3 contribution,
                                          const uint depth,
                                          thread float3* outDirectRadiance,
                                          thread float3* outIndirectRadiance) {
    if (depth == 0u) {
        if (outDirectRadiance) {
            *outDirectRadiance += contribution;
        }
    } else if (outIndirectRadiance) {
        *outIndirectRadiance += contribution;
    }
    return contribution;
}

// Whether this vertex's fused mesh-emitter NEE can produce direction wi, so a
// BSDF scatter along wi reaches emission that NEE also estimates: plain NEE's
// surface cosine, or the reservoir target's (direct_light_target_cos).
inline bool mesh_emitter_nee_covers_direction(constant PathtraceUniforms& uniforms,
                                              const MaterialData material,
                                              thread const HitRecord& rec,
                                              const float3 normal,
                                              const float3 wi) {
    return (uniforms.directLightMode == kDirectLightModeBaselineEmissive
                ? two_sided_alpha_cutout_direct_cos(uniforms, material, rec, normal, wi)
                : direct_light_target_cos(uniforms, material, rec, normal, wi)) > 0.0f;
}

inline bool material_is_two_sided_alpha_cutout(const MaterialData material,
                                               thread const HitRecord& rec) {
    return rec.twoSided != 0u &&
           material.typeEta.z > 0.5f &&
           material.pbrExtras.w > 0.5f && material.pbrExtras.w < 1.5f &&
           material.textureIndices0.x != kInvalidTextureIndex;
}

inline float two_sided_alpha_cutout_direct_cos(const MaterialData material,
                                               thread const HitRecord& rec,
                                               const float3 normal,
                                               const float3 wi) {
    float nDotL = dot(source_light_cos_normal(material, normal), wi);
    if ((material_is_two_sided_alpha_cutout(material, rec) &&
         (material.materialFlags & kMaterialFlagAlphaCutoutContinuation) == 0u) ||
        (material_is_regularized_dielectric(material) || source_rough_transmission(material))) {
        return saturate(fabs(nDotL));
    }
    return saturate(nDotL) * falcor_face_visibility(material, rec, wi);
}

// Returns true when it supplied the value, a term no BSDF strategy produces (see the copy in
// lighting/direct_light_target.metal), so NEE's MIS weight must not count the pdf it sets here.
inline bool apply_two_sided_alpha_cutout_diffuse_fallback(const MaterialData material,
                                                          thread const HitRecord& rec,
                                                          const float3 normal,
                                                          const float3 wi,
                                                          const float diffuseOcclusion,
                                                          thread BsdfEvalResult& bsdfEval) {
    if ((material.materialFlags & (kMaterialFlagSingleScatterSpecular | kMaterialFlagAlphaCutoutContinuation)) != 0u ||
        !material_is_two_sided_alpha_cutout(material, rec) ||
        dot(normal, wi) >= 0.0f ||
        any(bsdfEval.value > float3(0.0f))) {
        return false;
    }

    // The back-side copy of the PBR closure's diffuse lobe, so it carries that lobe's weight:
    // (1 - metallic) and (1 - transmission) (pbr eval in closures_eval.metal). A fully transmissive cutout gets none.
    const float metallic = clamp(material.pbrParams.x, 0.0f, 1.0f);
    float3 diffuseColor = clamp01(material.baseColorRoughness.xyz) * (1.0f - metallic);
    diffuseColor *= 1.0f - clamp(material.pbrExtras.z, 0.0f, 1.0f) * (1.0f - metallic);
    diffuseColor *= clamp(diffuseOcclusion, 0.0f, 1.0f);
    bsdfEval.value = diffuseColor / kPi;
    bsdfEval.pdf = max(fabs(dot(normal, wi)) / kPi, 1.0e-6f);
    bsdfEval.directionalPdf = bsdfEval.pdf;
    bsdfEval.areaPdf = 0.0f;
    bsdfEval.isDelta = false;
    bsdfEval.isBssrdf = false;
    return true;
}

inline bool pavilion_punctual_shadow_enabled(const MaterialData receiver,
                                              const bool punctual) {
    return punctual &&
        (receiver.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u;
}

inline bool pavilion_mask_shadow_hit(constant PathtraceUniforms& uniforms,
                                     thread const HitRecord& rec,
                                     device const MaterialData* materials) {
    if (!materials || rec.primitiveType != kPrimitiveTypeTriangle ||
        rec.materialIndex >= uniforms.materialCount) {
        return false;
    }
    const MaterialData blocker = materials[rec.materialIndex];
    return (blocker.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u &&
           blocker.pbrExtras.w > 0.5f && blocker.pbrExtras.w < 1.5f;
}

inline bool alpha_cutout_continuation_shadow_hit(constant PathtraceUniforms& uniforms,
                                                 thread const HitRecord& rec,
                                                 device const MaterialData* materials) {
    if (!materials || rec.primitiveType != kPrimitiveTypeTriangle ||
        rec.materialIndex >= uniforms.materialCount) return false;
    const MaterialData blocker = materials[rec.materialIndex];
    return (blocker.materialFlags & (kMaterialFlagAlphaCutoutContinuation | kMaterialFlagSingleScatterSpecular)) != 0u &&
           blocker.pbrExtras.w > 0.5f && blocker.pbrExtras.w < 1.5f;
}

// A shadow ray passes a BLEND texel with probability 1 - alpha, the probability a path ray passes it, so its expected
// visibility is the product of (1 - alpha) over the BLEND texels on the segment. NEE then sees the light behind BLEND
// that BSDF sampling sees, and a BLEND pass is a MIS null event like a MASK hole. This is PBRT-v4's stochastic alpha
// test. The number is a hash of the hit (primitive, mesh, barycentrics, distance) and the frame: it draws nothing from
// the path's RNG stream, so the shadow tracers need no RNG state and other images keep their samples.
inline float shadow_blend_alpha_sample(constant PathtraceUniforms& uniforms, thread const HitRecord& rec) {
    uint h = pcg_hash((uniforms.frameIndex * 0x9E3779B9u) ^ rec.primitiveIndex);
    h = pcg_hash(h ^ rec.meshIndex);
    h = pcg_hash(h ^ as_type<uint>(rec.barycentric.x));
    h = pcg_hash(h ^ as_type<uint>(rec.barycentric.y));
    h = pcg_hash(h ^ as_type<uint>(rec.t));
    return float(h >> 8u) * (1.0f / 16777216.0f);
}

template<bool allowSourceOpacity>
inline bool shadow_hit_alpha_cutout_discarded(
    constant PathtraceUniforms& uniforms,
    thread const HitRecord& rec,
    device const MaterialData* materials,
    constant array<texture2d<float, access::sample>, kMaxMaterialTextures>& materialTextures,
    constant array<sampler, kMaxMaterialSamplers>& materialSamplers,
    device const MaterialTextureInfo* materialTextureInfos,
    device const MeshInfo* meshInfos,
    device const SceneVertex* sceneVertices,
    device const uint3* meshIndices,
    const bool referencePunctualShadow = false) {
    if (!materials ||
        rec.materialIndex >= uniforms.materialCount) {
        return false;
    }

    MaterialData material = materials[rec.materialIndex];
    if (allowSourceOpacity && source_has_opacity(material)) {
        return source_opacity(uniforms,material,rec,materialTextures,materialSamplers,
                              materialTextureInfos,meshInfos,sceneVertices,meshIndices) < 1.0f;
    }
    if (rec.primitiveType != kPrimitiveTypeTriangle) return false;
    float alphaMode = material.pbrExtras.w;
    if (alphaMode <= 0.5f) {
        return false;
    }

    float alpha = clamp(material.pbrExtras.x, 0.0f, 1.0f);
    if (material_texture_valid(uniforms, material.textureIndices0.x) &&
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
        PbrTextureSamplingContext baseColorCtx =
            make_pbr_texture_sampling_context(material,
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
        if ((material.materialFlags & kMaterialFlagBlackKeyAlphaFromRgb) != 0u) {
            float luma = dot(baseColorSample.xyz, kLuminanceWeights);
            baseColorSample.w = smoothstep(0.72f, 0.90f, luma);
        }
        baseColorSample = resolve_alpha_cutout_foliage_base_color_sample(material,
                                                                         baseColorSample);
        alpha = clamp(alpha * baseColorSample.w, 0.0f, 1.0f);
    }

    if (alphaMode < 1.5f) {
        // The reference has no geometry/env NEE. Its alpha<1 shadow rejection
        // is safe only for discrete point/sun rays; their BSDF-hit competitor
        // does not exist. Other domains keep the authored primary MASK cutoff.
        const bool sourceMask = referencePunctualShadow &&
            (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u;
        float alphaCutoff = sourceMask ? 1.0f : clamp(material.pbrExtras.y, 0.0f, 1.0f);
        if (alpha < alphaCutoff) {
            return true;
        }
        return false;
    }

    // BLEND: a fully transparent texel always passes, a partly transparent one with probability 1 - alpha.
    return alpha <= 0.0f || (alpha < 1.0f && shadow_blend_alpha_sample(uniforms, rec) >= alpha);
}

template<bool allowSourceOpacity>
inline bool trace_scene_software_alpha_aware_shadow(
    constant PathtraceUniforms& uniforms,
    device const SphereData* spheres,
    device const RectData* rectangles,
    device const TriangleData* triangleData,
    device const BvhNode* tlasNodes,
    device const uint* tlasPrimIndices,
    device const SoftwareInstanceInfo* instanceInfos,
    device const BvhNode* blasNodes,
    device const uint* blasPrimIndices,
    device const BvhNode* nodes,
    device const uint* primitiveIndices,
    device PathtraceStats* stats,
    Ray ray,
    const float tMin,
    const float tMax,
    device const MaterialData* materials,
    constant array<texture2d<float, access::sample>, kMaxMaterialTextures>& materialTextures,
    constant array<sampler, kMaxMaterialSamplers>& materialSamplers,
    device const MaterialTextureInfo* materialTextureInfos,
    device const MeshInfo* meshInfos,
    device const SceneVertex* sceneVertices,
    device const uint3* meshIndices,
    thread HitRecord& rec,
    const bool referencePunctualShadow = false,
    const uint sourceDepth = 0u,
    const float3 sourceSigma = float3(0.0f), const float sourceOffset = 0.0f) {
    Ray currentRay = ray;
    float remainingDistance = tMax;
    float opacityTransmission = 1.0f;
    const bool sourceVisibility = allowSourceOpacity && source_transport_enabled(uniforms);
    float currentMin = allowSourceOpacity && (source_transport_enabled(uniforms) || source_opacity_present(uniforms)) ? source_ray_epsilon(uniforms) : tMin;
    float previousT = -sourceOffset;
    float3 sigma = sourceSigma, sourceRgb = float3(1.0f);
    uint sourceShadowBounce = sourceDepth + 1u;
    uint maxAlphaShadowSkips = 16u;
    for (uint skip = 0u; skip < maxAlphaShadowSkips; ++skip) {
        rec = make_empty_hit_record();
        // The nearest hit must be tested first so a transparent triangle cannot
        // skip over an opaque blocker that lies in front of it.
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
                                             currentRay,
                                             currentMin,
                                             remainingDistance,
                                             /*anyHitOnly=*/false,
                                             /*includeTriangles=*/true,
                                             rec);
        if (kSourceFeatures) rec.sourceShadowTransmission = opacityTransmission;
        if (sourceVisibility) {
            const float endT = occluded ? rec.t : remainingDistance;
            const float segment = max(endT-previousT,0.0f);
            // Avoid 0*infinity for vacuum environment connections.
            sourceRgb *= exp(-sigma*min(segment,1.0e20f));
            previousT = endT;
            rec.sourceShadowRgb = sourceRgb;
        }
        if (!occluded) return false;
        const bool sourceThin = sourceVisibility && materials && rec.materialIndex < uniforms.materialCount && source_is_thin(materials[rec.materialIndex]);
        if (sourceThin) {
            float3 thinNormal = rec.shadingNormal;
            if (rec.primitiveType == kPrimitiveTypeTriangle && meshInfos && sceneVertices && meshIndices) {
                float3 authored = interpolate_shading_normal(uniforms,rec.meshIndex,rec.primitiveIndex,rec.barycentric,meshInfos,sceneVertices,meshIndices);
                if (all(isfinite(authored)) && dot(authored,authored)>0.0f) thinNormal = authored;
            }
            sourceRgb *= source_thin_forward(materials[rec.materialIndex],thinNormal,-currentRay.direction);
            rec.sourceShadowRgb = sourceRgb;
            if (++sourceShadowBounce >= uniforms.maxDepth || !any(sourceRgb > float3(0.0f))) return true;
            maxAlphaShadowSkips = 64u;
        }
        if (materials && rec.materialIndex < uniforms.materialCount &&
            allowSourceOpacity && source_has_opacity(materials[rec.materialIndex])) {
            if (++sourceShadowBounce >= uniforms.maxDepth) {
                rec.sourceShadowTransmission = 0.0f;
                return true;
            }
            opacityTransmission *= 1.0f-source_opacity(uniforms,materials[rec.materialIndex],rec,
                materialTextures,materialSamplers,materialTextureInfos,meshInfos,sceneVertices,meshIndices);
            rec.sourceShadowTransmission = opacityTransmission;
            maxAlphaShadowSkips = 64u;
            if (opacityTransmission <= 0.0f) return true;
        }
        if (!sourceThin && !shadow_hit_alpha_cutout_discarded<allowSourceOpacity>(uniforms,
                                               rec,
                                               materials,
                                               materialTextures,
                                               materialSamplers,
                                               materialTextureInfos,
                                               meshInfos,
                                               sceneVertices,
                                               meshIndices,
                                               referencePunctualShadow)) {
            return true;
        }
        if (allowSourceOpacity && (sourceVisibility || (materials && rec.materialIndex < uniforms.materialCount && source_has_opacity(materials[rec.materialIndex])))) {
            if (sourceVisibility) {
                uint mediumDepth = any(sigma > float3(0.0f)) ? 1u : 0u;
                interface_select_medium(materials[rec.materialIndex],rec,currentRay.direction,sigma,mediumDepth);
            }
            currentMin = source_forward_min_t(currentRay,rec.t,source_ray_epsilon(uniforms));
            if (currentMin >= remainingDistance) return false;
            continue;
        }
        if (materials && rec.materialIndex < uniforms.materialCount &&
            (materials[rec.materialIndex].materialFlags &
             (kMaterialFlagAlphaCutoutContinuation | kMaterialFlagSingleScatterSpecular)) != 0u) {
            maxAlphaShadowSkips = 64u;
        }

        float3 direction = currentRay.direction;
        float directionLengthSq = dot(direction, direction);
        if (!all(isfinite(direction)) || directionLengthSq <= 1.0e-12f) {
            return false;
        }
        direction *= rsqrt(directionLengthSq);
        float hitDistance = ray_segment_world_length(currentRay, rec.t);
        if (!isfinite(hitDistance) || hitDistance < 0.0f) {
            hitDistance = 0.0f;
        }
        float3 nextOrigin = rec.point;
        if (!all(isfinite(nextOrigin))) {
            nextOrigin = ray_at(currentRay, rec.t);
        }
        currentRay.origin = nextOrigin + direction * kEpsilon;
        if (isfinite(remainingDistance)) {
            remainingDistance = max(remainingDistance - hitDistance - kEpsilon, 0.0f);
            if (remainingDistance <= kEpsilon) {
                return false;
            }
        }
    }
    return true;
}

#if __METAL_VERSION__ >= 310
template<bool allowSourceOpacity>
inline bool trace_scene_hardware_alpha_aware_shadow(
    constant PathtraceUniforms& uniforms,
    acceleration_structure<instancing> accel,
    device const MeshInfo* meshInfos,
    device const TriangleData* triangleData,
    device const SceneVertex* sceneVertices,
    device const uint3* meshIndices,
    device const uint* instanceUserIds,
    device const SphereData* spheres,
    device const RectData* rectangles,
    device const BvhNode* nodes,
    device const uint* primitiveIndices,
    device PathtraceStats* stats,
    Ray ray,
    const float tMin,
    const float tMax,
    const uint excludeMesh,
    const uint excludePrim,
    device const MaterialData* materials,
    constant array<texture2d<float, access::sample>, kMaxMaterialTextures>& materialTextures,
    constant array<sampler, kMaxMaterialSamplers>& materialSamplers,
    device const MaterialTextureInfo* materialTextureInfos,
    thread HitRecord& rec,
    const bool referencePunctualShadow = false,
    const uint sourceDepth = 0u,
    const float3 sourceSigma = float3(0.0f), const float sourceOffset = 0.0f) {
    Ray currentRay = ray;
    float currentTMax = tMax;
    uint currentExcludeMesh = excludeMesh;
    uint currentExcludePrim = excludePrim;

    float opacityTransmission = 1.0f;
    const bool sourceVisibility = allowSourceOpacity && source_transport_enabled(uniforms);
    float currentMin = allowSourceOpacity && (source_transport_enabled(uniforms) || source_opacity_present(uniforms)) ? source_ray_epsilon(uniforms) : tMin;
    float previousT = -sourceOffset;
    float3 sigma = sourceSigma, sourceRgb = float3(1.0f);
    uint sourceShadowBounce = sourceDepth + 1u;
    uint maxAlphaShadowSkips = 16u;
    for (uint skip = 0u; skip < maxAlphaShadowSkips; ++skip) {
        rec = make_empty_hit_record();
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
                                             currentRay,
                                             currentMin,
                                             currentTMax,
                                             /*anyHitOnly=*/false,
                                             currentExcludeMesh,
                                             currentExcludePrim,
                                             rec,
                                             /*shadowQuery=*/true);
        if (kSourceFeatures) rec.sourceShadowTransmission = opacityTransmission;
        if (sourceVisibility) {
            const float endT = occluded ? rec.t : currentTMax;
            const float segment = max(endT-previousT,0.0f);
            // Avoid 0*infinity for vacuum environment connections.
            sourceRgb *= exp(-sigma*min(segment,1.0e20f));
            previousT = endT;
            rec.sourceShadowRgb = sourceRgb;
        }
        if (!occluded) return false;
        const bool sourceThin = sourceVisibility && materials && rec.materialIndex < uniforms.materialCount && source_is_thin(materials[rec.materialIndex]);
        if (sourceThin) {
            float3 thinNormal = rec.shadingNormal;
            if (rec.primitiveType == kPrimitiveTypeTriangle && meshInfos && sceneVertices && meshIndices) {
                float3 authored = interpolate_shading_normal(uniforms,rec.meshIndex,rec.primitiveIndex,rec.barycentric,meshInfos,sceneVertices,meshIndices);
                if (all(isfinite(authored)) && dot(authored,authored)>0.0f) thinNormal = authored;
            }
            sourceRgb *= source_thin_forward(materials[rec.materialIndex],thinNormal,-currentRay.direction);
            rec.sourceShadowRgb = sourceRgb;
            if (++sourceShadowBounce >= uniforms.maxDepth || !any(sourceRgb > float3(0.0f))) return true;
            maxAlphaShadowSkips = 64u;
        }
        if (materials && rec.materialIndex < uniforms.materialCount &&
            allowSourceOpacity && source_has_opacity(materials[rec.materialIndex])) {
            if (++sourceShadowBounce >= uniforms.maxDepth) {
                rec.sourceShadowTransmission = 0.0f;
                return true;
            }
            opacityTransmission *= 1.0f-source_opacity(uniforms,materials[rec.materialIndex],rec,
                materialTextures,materialSamplers,materialTextureInfos,meshInfos,sceneVertices,meshIndices);
            rec.sourceShadowTransmission = opacityTransmission;
            maxAlphaShadowSkips = 64u;
            if (opacityTransmission <= 0.0f) return true;
        }
        if (!sourceThin && !shadow_hit_alpha_cutout_discarded<allowSourceOpacity>(uniforms,
                                               rec,
                                               materials,
                                               materialTextures,
                                               materialSamplers,
                                               materialTextureInfos,
                                               meshInfos,
                                               sceneVertices,
                                               meshIndices,
                                               referencePunctualShadow)) {
            return true;
        }
        if (materials && rec.materialIndex < uniforms.materialCount &&
            (materials[rec.materialIndex].materialFlags &
             (kMaterialFlagAlphaCutoutContinuation | kMaterialFlagSingleScatterSpecular)) != 0u) {
            maxAlphaShadowSkips = 64u;
        }
        if (allowSourceOpacity && (sourceVisibility || (materials && rec.materialIndex < uniforms.materialCount && source_has_opacity(materials[rec.materialIndex])))) {
            if (sourceVisibility) {
                uint mediumDepth = any(sigma > float3(0.0f)) ? 1u : 0u;
                interface_select_medium(materials[rec.materialIndex],rec,currentRay.direction,sigma,mediumDepth);
            }
            currentMin = source_forward_min_t(currentRay,rec.t,source_ray_epsilon(uniforms));
            if (currentMin >= currentTMax) return false;
            continue;
        }
        float3 dir = currentRay.direction;
        float dirLenSq = dot(dir, dir);
        if (!all(isfinite(dir)) || dirLenSq <= 1.0e-12f) {
            return false;
        }
        dir *= rsqrt(dirLenSq);

        float hitDistance = ray_segment_world_length(currentRay, rec.t);
        if (!isfinite(hitDistance) || hitDistance < 0.0f) {
            hitDistance = 0.0f;
        }
        float3 nextOrigin = rec.point;
        if (!all(isfinite(nextOrigin))) {
            nextOrigin = ray_at(currentRay, rec.t);
        }
        currentRay.origin = nextOrigin + dir * kHardwareOcclusionEpsilon;
        currentExcludeMesh = kInvalidIndex;
        currentExcludePrim = kInvalidIndex;
        if (isfinite(currentTMax)) {
            currentTMax = max(currentTMax - hitDistance - kHardwareOcclusionEpsilon,
                              0.0f);
            if (currentTMax <= kHardwareOcclusionEpsilon) {
                return false;
            }
        }
    }

    return true;
}

#endif

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
                                  PrimaryRayDiff primaryRayDiff,
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
                                  constant array<texture2d<float, access::sample>, kMaxMaterialTextures>& materialTextures,
                                  constant array<sampler, kMaxMaterialSamplers>& materialSamplers,
                                  device const MaterialTextureInfo* materialTextureInfos,
                                  device const EnvironmentAliasEntry* environmentConditionalAlias,
                                  device const EnvironmentAliasEntry* environmentMarginalAlias,
                                  device const float* environmentPdf,
                                  device PathGuidingReservoirState* pathGuidingStates,
                                  device RestirPtReservoirState* restirPtReservoirs,
                                  device RadianceCacheState* radianceCacheStates,
                                  device EyeSubpathVertex* causticEyeVertices,
                                  device LightSubpathVertex* causticLightVertices,
                                  device PathConnection* causticConnections,
                                  device ConnectionResult* causticConnectionResults,
                                  uint2 pixelCoord,
                                  // Optional AOV outputs
                                  thread float3* outFirstHitAlbedo = nullptr,
                                  thread float3* outFirstHitNormal = nullptr,
                                  thread float4* outFirstHitPosition = nullptr,
                                  thread float4* outFirstHitMaterial = nullptr,
                                  thread float3* outDirectRadiance = nullptr,
                                  thread float3* outIndirectRadiance = nullptr,
                                  thread float* outReconstructionConfidence = nullptr,
                                  thread float* outReconstructionVariance = nullptr,
                                  thread float* outReconstructionBiasRisk = nullptr,
                                  thread PathtraceDebugContext* debugContext = nullptr) {
    float3 throughput = float3(1.0f, 1.0f, 1.0f);
    float3 radiance = float3(0.0f, 0.0f, 0.0f);
    RadianceCachePathTraining cacheTraining;
    radiance_cache_training_begin(cacheTraining,
                                  uniforms,
                                  radianceCacheStates,
                                  pixelCoord.y * uniforms.width + pixelCoord.x);
    float lastBsdfPdf = 1.0f;
    // The last scatter was in the medium (lastBsdfPdf is its phase pdf), and whether volume NEE sampled the rect lights
    // there: a phase-sampled emitter hit or escape is MIS-weighted only against a strategy that ran at that vertex.
    bool lastScatterWasVolume = false;
    bool lastVolumeRectNee = false;
    bool lastScatterWasDelta = true;
    bool lastDeltaUnpaired = false;
    bool previousMeshNeeEligible = false;
    float3 previousMeshNeeOrigin = float3(0.0f);
    // Where the last real scatter left from, kept across alpha holes (MIS null events): the rect-light pdf of the next
    // emitter hit is measured from there, as that vertex's NEE was.
    float3 holeMisOrigin = float3(0.0f);
    bool holeMisOriginValid = false;
    bool isFirstHit = true;
    bool isGuidePending = true;
    bool pavilionGuideAverage = false;
    bool primaryVisibilityRay = true;
    uint specularDepth = 0u;
    float pathMaxRoughness = 0.0f;
    bool pavilionSmoothDifferentials = true;
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
    reconstruction_split_record( \
        clamp_firefly_contribution((throughput), (contribution), (params), stats), \
        depth, \
        outDirectRadiance, \
        outIndirectRadiance)
#define clamp_path_throughput(throughput, params) \
    clamp_path_throughput((throughput), (params), stats)
    constexpr uint kMaxMediumStack = 8u;
    float3 mediumSigmaStack[kMaxMediumStack];
    for (uint i = 0; i < kMaxMediumStack; ++i) {
        mediumSigmaStack[i] = float3(0.0f);
    }
    float interfaceMinT = kEpsilon;
    float mediumSegmentOffset = 0.0f;
    uint mediumDepth = 0u;
    uint volumeScatteringDepth = 0u;
    SpectralPathState spectralState = spectral_make_path_state(uniforms, state, stats);
    if (spectralState.enabled) {
        // Channel j of the throughput carries wavelength j; inactive channels start at zero.
        throughput = spectral_channel_mask(spectralState);
    }

    uint foliageAlphaVisibilitySkips = 0u;
    uint depthAdvance = 1u;
    for (uint depth = 0; depth < uniforms.maxDepth; depth += depthAdvance) {
        depthAdvance = 1u;
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
                          interfaceMinT,
                          kInfinity,
                          excludeMeshIndex,
                          excludePrimitiveIndex,
                          rec)) {
            if (PT_DEFAULT_OFF(uniforms.debugViewMode) != kDebugViewNone) {
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
            background = spectral_radiance_to_channels(background, spectralState);
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
                use_visible_emitter_mis(depth, lastScatterWasDelta, lastDeltaUnpaired, uniforms);
            // Scatter points sample no environment light, so an escape after one keeps weight 1.
            if (useSpecularMis && envSampling && !lastScatterWasVolume) {
                float lightPdf = environment_pdf(uniforms, environmentPdf, ray.direction);
                float denom = lastBsdfPdf + lightPdf;
                if (denom > 0.0f) {
                    misWeight = clamp(lastBsdfPdf / denom,
                                      kMisWeightClampMin,
                                      kMisWeightClampMax);
                }
            }
            float3 contribution = background * misWeight;
            if (depth == 0u && uniforms.backgroundMode == 2u &&
                environmentTexture.get_width() > 0u && environmentTexture.get_height() > 0u &&
                uniforms.environmentBackgroundScaleOffset != 0.0f) {
                contribution *= max(1.0f + uniforms.environmentBackgroundScaleOffset, 0.0f);
            }
            // Keep the physical HDR radiance for light endpoints and PDFs.
            caustic_transport_note_light_endpoint(uniforms,
                                                  ray.origin + safe_normalize(ray.direction),
                                                  -ray.direction,
                                                  ray.direction,
                                                  background,
                                                  envSampling ? environment_pdf(uniforms, environmentPdf, ray.direction) : 0.0f,
                                                  lastBsdfPdf,
                                                  depth,
                                                  mediumDepth,
                                                  causticLightVertices,
                                                  stats);
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
        const float segmentOffset = mediumSegmentOffset;
        mediumSegmentOffset = 0.0f;
        interfaceMinT = kEpsilon;
        if (volume_transport_enabled(uniforms)) {
            MediumDescriptor medium = volume_make_global_medium(uniforms);
            const float segment = max(rec.t, 0.0f);
            const float3 segmentEmission = volume_emission_integral(medium, segment);
            const float3 segmentTransmittance = volume_transmittance(medium, segment, stats);
            if (PT_DEFAULT_OFF(uniforms.volumeDebug) != 0u && outFirstHitMaterial) {
                *outFirstHitMaterial =
                    volume_debug_aov_features(segmentTransmittance,
                                              volumeScatteringDepth,
                                              volume_phase_id(uniforms),
                                              segmentEmission);
            }
            if (any(segmentEmission > float3(0.0f))) {
                radiance += clamp_firefly_contribution(throughput, segmentEmission, clampParams);
            }
            const uint maxVolumeEvents = max(medium.metadata.y, 1u);
            bool sampledVolumeEvent = false;
            bool sampledFreeFlight = false;
            float3 freeFlightWeight = float3(1.0f);
            if (volumeScatteringDepth < maxVolumeEvents &&
                any(medium.sigmaSAnisotropy.xyz > float3(0.0f))) {
                const VolumeFreeFlight flight = volume_sample_free_flight(medium, segment, state);
                const float sampledDistance = flight.distance;
                sampledFreeFlight = true;
                freeFlightWeight = flight.weight;
                sampledVolumeEvent = flight.scattered;
                if (sampledVolumeEvent) {
                    const float3 scatterPoint = ray.origin + ray.direction * sampledDistance;
                    const float3 transmittance = volume_transmittance(medium, sampledDistance, stats);
                    if (PT_DEFAULT_OFF(uniforms.volumeDebug) != 0u && outFirstHitMaterial) {
                        *outFirstHitMaterial =
                            volume_debug_aov_features(transmittance,
                                                      volumeScatteringDepth + 1u,
                                                      volume_phase_id(uniforms),
                                                      segmentEmission);
                    }
                    // Free-flight weight sigma_s T(t) / mean(sigma_t T(t)) (volume_sample_free_flight).
                    throughput *= freeFlightWeight;
                    if (rectLightCount > 0u && medium.metadata.z != 0u) {
                        HitRecord volumeRec = rec;
                        volumeRec.point = scatterPoint;
                        for (uint neeDraw = 0u; neeDraw < rect_light_nee_draw_count(uniforms); ++neeDraw) {
                            RectLightSample lightSample;
                            if (sample_rect_light_nee_draw(uniforms,
                                                  rectangles,
                                                  materials,
                                                  environmentTexture,
                                                  volumeRec,
                                                  state,
                                                  rectLightCount,
                                                  lightSample, kAnalyticLightFilterAll, neeDraw)) {
                                Ray shadowRay;
                                shadowRay.origin = scatterPoint + lightSample.direction * kRayOriginEpsilon;
                                shadowRay.direction = lightSample.direction;
                                shadowRay.mediumState =
                                    volume_make_global_medium_state(uniforms,
                                                                    mediumDepth,
                                                                    volumeScatteringDepth + 1u);
                                HitRecord shadowRec = make_empty_hit_record();
                                const float shadowMax = max(lightSample.distance - kEpsilon, kEpsilon);
                                // Alpha-tested cutouts are skipped, as this megakernel's surface NEE skips them.
                                bool occluded = trace_scene_software_alpha_aware_shadow<false>(
                                    uniforms, spheres, rectangles, triangleData,
                                    tlasNodes, tlasPrimIndices, instanceInfos,
                                    blasNodes, blasPrimIndices, nodes, primitiveIndices,
                                    stats, shadowRay, kEpsilon, shadowMax, materials,
                                    materialTextures, materialSamplers, materialTextureInfos,
                                    meshInfos, sceneVertices, meshIndices, shadowRec);
                                if (!occluded && lightSample.pdf > 0.0f) {
                                    const float phase = volume_phase_eval(medium, -ray.direction, lightSample.direction);
                                    const float3 shadowTr = volume_transmittance(medium, shadowMax, stats);
                                    // MIS against the phase-sampled hit of the same light (volume_nee_mis_weight).
                                    const float misWeight = lightSample.isReferencePoint
                                        ? 1.0f : volume_nee_mis_weight(lightSample.pdf, phase);
                                    const float3 contribution = lightSample.emission * phase * shadowTr * misWeight /
                                                                max(lightSample.pdf, 1.0e-6f);
                                    radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                                    if (stats) {
                                        atomic_fetch_add_explicit(&stats->volumeNeeRayCount, 1u, memory_order_relaxed);
                                    }
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
                    volume_apply_medium_state(ray, uniforms, mediumDepth, volumeScatteringDepth);
                    lastScatterWasDelta = false;
                    lastDeltaUnpaired = false;
                    previousMeshNeeEligible = false;
                    pavilionSmoothDifferentials = false;
                    lastBsdfPdf = max(phasePdf, 1.0e-6f);
                    lastScatterWasVolume = true;
                    lastVolumeRectNee = rectLightCount > 0u && medium.metadata.z != 0u;
                    ray.origin = scatterPoint + phaseDir * kRayOriginEpsilon;
                    holeMisOriginValid = false;
                    ray.direction = phaseDir;
                    prevValid = false;
                    primaryVisibilityRay = false;
                    continue;
                }
            } else if (any(medium.sigmaSAnisotropy.xyz > float3(0.0f)) && stats) {
                atomic_fetch_add_explicit(&stats->volumeFallbackCount, 1u, memory_order_relaxed);
            }
            // Passed the segment: the free-flight weight T(s) / mean(T(s)) when a distance was sampled, else T(s).
            throughput *= sampledFreeFlight ? freeFlightWeight : segmentTransmittance;
        }
        if (mediumDepth > 0u) {
            float3 sigma = mediumSigmaStack[mediumDepth - 1u];
            if (any(sigma > float3(0.0f))) {
                float segment = max(rec.t + segmentOffset, 0.0f);
                float3 attenuation = exp(-sigma * segment);
                throughput *= attenuation;
            }
        }
        uint matIndex = min(rec.materialIndex, uniforms.materialCount - 1);
        MaterialData material = materials[matIndex];
        regularize_dielectric_material(material, rec.frontFace != 0u, pathMaxRoughness);
        uint type = static_cast<uint>(material.typeEta.x);
        MaterialClosureMetadata primaryClosure = material_primary_closure_metadata(material);
        uint closureType = primaryClosure.type;
        float3 incidentDir = normalize(ray.direction);
        float3 wo = -incidentDir;
        float hitDistanceWorld = ray_segment_world_length(ray, rec.t);
        bool surfaceIsDelta = material_closure_is_delta(material);
        bool explicitRestirDiLightingStage =
            uniforms.explicitRestirDiLightingStage != 0u &&
            (uniforms.directLightMode == kDirectLightModeRestirDi ||
             uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid);
        bool specularOnly = (uniforms.debugSpecularOnly != 0u);
        float diffuseOcclusion = 1.0f;
        float3 debugBaseColor = material_base_color(material);
        float2 debugBaseColorUv = float2(0.0f);
        float debugBaseColorLod = 0.0f;
        float debugMetallic = 0.0f;
        float debugRoughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        float debugAO = 1.0f;
        float3 shadingNormal = rec.shadingNormal;
        bool authoredNormalWasFaced = false;
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
                if (dot(candidate, rec.normal) < 0.0f &&
                    !source_material_flag(material, kMaterialFlagAuthoredDielectricNormal)) {
                    authoredNormalWasFaced = true;
                    candidate = -candidate;
                }
                shadingNormal = normalize(candidate);
                debugVtxNormal = shadingNormal;
            }
        }
        if (type == 2u && !source_material_flag(material, kMaterialFlagAuthoredDielectricNormal)) {
            float3 geomNormal = rec.normal;
            if (all(isfinite(geomNormal)) && dot(geomNormal, geomNormal) > 0.0f) {
                shadingNormal = geomNormal;
            }
            // Keep ray offsets consistent between SWRT/HWRT for glass.
            rec.shadingNormal = shadingNormal;
        }


        if (rec.primitiveType == kPrimitiveTypeSphere &&
            source_material_flag(material, kMaterialFlagAuthoredDielectricNormal)) {
            // An analytic sphere's authored normal is its outward normal (triangles keep their raw interpolated
            // normal above); set_face_normal faced it, so source_dielectric_frame never saw an exit (EP8-f3).
            shadingNormal = rec.frontFace != 0u ? rec.normal : -rec.normal;
        }
        if (kSourceFeatures && material_texture_valid(uniforms,material.textureIndices1.w)) {
            shadingNormal = source_bump_normal(uniforms,material,rec,shadingNormal,
                                              materialTextures,meshInfos,sceneVertices,meshIndices);
            rec.shadingNormal = shadingNormal;
        }
        source_mark_closure_side(material, shadingNormal, wo);

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
            baseColorSample = resolve_alpha_cutout_foliage_base_color_sample(material,
                                                                             baseColorSample);
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
                                                                 uvPerWorld0,
                                                                 (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u);
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
            if ((depth == 0u || (pavilionSmoothDifferentials &&
                 (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u)) &&
                hasSurfacePartials0) {
                float3 dudP = float3(0.0f);
                float3 dvdP = float3(0.0f);
                if (uv_world_gradients_from_partials(dPdu0, dPdv0, dudP, dvdP)) {
                    hasIgehyGradients0 = first_hit_uv_gradients_igehy(ray,
                                                                      primaryRayDiff,
                                                                      rec.t,
                                                                      ((material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u
                                                                      ? shadingNormal : rec.normal),
                                                                      dudP,
                                                                      dvdP,
                                                                      dUVdx0,
                                                                      dUVdy0);
                }
            }
            float2 dUVdx1 = float2(0.0f);
            float2 dUVdy1 = float2(0.0f);
            bool hasIgehyGradients1 = false;
            if ((depth == 0u || (pavilionSmoothDifferentials &&
                 (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u)) &&
                hasSurfacePartials1) {
                float3 dudP = float3(0.0f);
                float3 dvdP = float3(0.0f);
                if (uv_world_gradients_from_partials(dPdu1, dPdv1, dudP, dvdP)) {
                    hasIgehyGradients1 = first_hit_uv_gradients_igehy(ray,
                                                                      primaryRayDiff,
                                                                      rec.t,
                                                                      ((material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u
                                                                      ? shadingNormal : rec.normal),
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
            baseColorSample = resolve_alpha_cutout_foliage_base_color_sample(material,
                                                                             baseColorSample);
            float3 baseColorSampleRgb = to_working_space(baseColorSample.xyz, uniforms);
            float3 baseColor = baseFactor * baseColorSampleRgb;
            debugBaseColorUv = baseColorCtx.uv;
            debugBaseColorLod = baseColorLod;

            float metallic = clamp(material.pbrParams.x, 0.0f, 1.0f);
            // Preserve scene multipliers above one through source-G multiplication.
            float roughness = max(material.pbrParams.y, 0.0f);
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
            roughness = clamp(roughness, 0.0f, 1.0f);
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
            alpha = clamp(alpha * path_cutout_alpha(materialTextures, materialSamplers, materialTextureInfos,
                                                    uniforms, material, baseColorCtx, baseColorLod,
                                                    baseColorSample.w),
                          0.0f, 1.0f);
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
                    const bool nonCountingPavilionAlphaSkip =
                        (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u &&
                        alphaMode < 1.5f && foliageAlphaVisibilitySkips < 64u;
                    const bool nonCountingFoliageAlphaSkip =
                        (material.materialFlags & kMaterialFlagAlphaCutoutContinuation) != 0u &&
                        foliageAlphaVisibilitySkips < 64u;
                    if (!holeMisOriginValid) {
                        holeMisOrigin = ray.origin;
                        holeMisOriginValid = true;
                    }
                    if (nonCountingPavilionAlphaSkip || nonCountingFoliageAlphaSkip) {
                        advance_pavilion_alpha_visibility_ray(ray, primaryRayDiff, rayCone, rec);
                    } else {
                        ray.origin = offset_ray_origin(rec, ray.direction);
                    }
                    prevRec = rec;
                    prevValid = true;
                    if (nonCountingPavilionAlphaSkip || nonCountingFoliageAlphaSkip) {
                        foliageAlphaVisibilitySkips += 1u;
                        depthAdvance = 0u;
                    } else {
                        // A counted MASK hole or BLEND pass is a null event for MIS: the last real vertex's BSDF pdf,
                        // delta flags and origin stay, so the next emitter hit or miss takes the BSDF side of the
                        // balance heuristic whose light side that vertex's NEE took (its alpha-aware shadow rays, the
                        // specular-NEE ones included, pass MASK holes, and BLEND texels with the path's probability
                        // 1 - alpha). A hole before any scatter is unpaired: no NEE samples what lies behind it.
                        if (depth == 0u) {
                            lastScatterWasDelta = true;
                            lastDeltaUnpaired = true;
                        }
                        specularDepth += 1u;
                    }
                    continue;
                }
            }

            if ((material.materialFlags & kMaterialFlagBlackBackFace) != 0u && rec.frontFace == 0u) {
                break;  // Single-sided back face: the reference shades it black.
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
            float normalDispersion = 0.0f;
            float3 normalSampleTs = float3(0.0f, 0.0f, 1.0f);
            bool flipNormalGreen = false;
#if PT_DEBUG_TOOLS
            flipNormalGreen = uniforms.debugFlipNormalGreen != 0u;
#endif
            if (useNormalMap) {
                float4 normalTexel =
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
                                                    normalCtx.dUVdy);
                debugTexelRaw = normalTexel.xyz;
                normalDispersion = normalTexel.w;
                normalSampleTs = decode_normal_map(debugTexelRaw,
                                                   normalScale,
                                                   flipNormalGreen);
                debugTexelDecoded = normalSampleTs;
                debugSnTangentSpace = normalSampleTs;
                float3 t = tangent.xyz;
                float3 b = float3(0.0f);
                bool hasBasis = false;
                bool hasVertexTangentBasis = false;
                bool trustVertexTangent = fabs(tangent.w) > 0.5f;
                if (trustVertexTangent && all(isfinite(t)) && dot(t, t) > 1.0e-6f) {
                    t = normalize(t - shadingNormal * dot(shadingNormal, t));
                    if (all(isfinite(t)) && dot(t, t) > 1.0e-6f) {
                        float tangentSign = (tangent.w < 0.0f) ? -1.0f : 1.0f;
                        b = normalize(cross(shadingNormal, t)) * tangentSign;
                        if (all(isfinite(b)) && dot(b, b) > 1.0e-6f) {
                            hasBasis = true;
                            hasVertexTangentBasis = true;
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
                float3 mappedNormalTs = normalSampleTs;
                if (hasVertexTangentBasis &&
                    (material.materialFlags & (kMaterialFlagAlphaCutoutContinuation |
                                               kMaterialFlagSourceNormalGreen)) != 0u &&
                    triangle_has_valid_vertex_tangents(uniforms, rec.meshIndex, rec.primitiveIndex,
                                                       meshInfos, sceneVertices, meshIndices)) {
                    // Per-material finding: these Emerald materials match the Falcor
                    // references only with G negated (other ORCA normal maps match with
                    // G kept). Then apply Falcor's complete-normal flip when N faced.
                    // Keep the original decoded sample for normal-variance roughness.
                    mappedNormalTs.y = -mappedNormalTs.y;
                    if (authoredNormalWasFaced) {
                        mappedNormalTs.x = -mappedNormalTs.x;
                    }
                }
                float3 mapped = normalize(t * mappedNormalTs.x +
                                          b * mappedNormalTs.y +
                                          shadingNormal * mappedNormalTs.z);
                if (dot(mapped, rec.normal) < 0.0f) {
                    mapped = -mapped;
                }
                if ((material.materialFlags & kMaterialFlagSourceNormalGreen) != 0u) {
                    // Falcor 4.4 adjustShadingNormal(): avoid back-facing view vectors
                    // at normal-map bevels. This flag is Picnic_Table1-only. Same rule
                    // as the wavefront (material_access.metal).
                    float nDotV = dot(mapped, wo);
                    if (nDotV <= 0.1f) {
                        mapped = normalize(mix(rec.normal, mapped, saturate(nDotV * 10.0f)));
                    }
                }
                debugTangent = t;
                debugBitangent = b;
                shadingNormal = mapped;
                debugSnWorld = shadingNormal;
            }

            if (useNormalMap) {
                float directionalVariance = normalDispersion * normalDispersion;
                float meanLength = sqrt(1.0f - directionalVariance);
                float tok = directionalVariance / (meanLength * (1.0f + meanLength));
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
                        nDx = decode_normal_map(nDx, normalScale, flipNormalGreen);
                        nDy = decode_normal_map(nDy, normalScale, flipNormalGreen);
                        float varianceX = 0.5f * dot(normalSampleTs - nDx, normalSampleTs - nDx);
                        float varianceY = 0.5f * dot(normalSampleTs - nDy, normalSampleTs - nDy);
                        float normalVariance = max(varianceX, varianceY);
                        tok += 0.35f * normalVariance;
                    }
                }
                // Falcor's StandardMaterial shades the authored roughness unfiltered.
                if (tok > 0.0f && uniforms.disableNormalVarianceRoughness == 0u &&
                    (material.materialFlags & kMaterialFlagFalcorStandardBsdf) == 0u) {
                    roughness = clamp(sqrt(roughness * roughness + tok), 0.0f, 1.0f);
                }
            }

            if ((material.materialFlags & kMaterialFlagFalcorStandardBsdf) != 0u) {
                roughness = max(roughness, 0.08f);  // Falcor floors GGX alpha at 0.0064; never delta.
            }
            material.baseColorRoughness = float4(baseColor, roughness);
            material.pbrParams.x = metallic;
            material.emission = float4(emissive, 0.0f);
            material = apply_foliage_beauty_lift(material);
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

        regularize_pavilion_pbr_material(material, pathMaxRoughness, rec.frontFace != 0u);
        // Denoiser guides stay RGB: capture them before the material becomes per-wavelength.
        const float3 spectralGuideAlbedo =
            spectralState.enabled ? material_closure_aov_albedo(material) : float3(0.0f);
        const float4 spectralGuideFeatures =
            spectralState.enabled ? material_closure_aov_features(material) : float4(0.0f);
        spectral_apply_material(material, uniforms, spectralState, stats);
        spectral_collapse_at_dispersive_vertex(spectralState, material, uniforms, throughput, stats);
        if ((material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u) {
            surfaceIsDelta = material_closure_is_delta(material);
        }
        primaryClosure = material_primary_closure_metadata(material);
        closureType = primaryClosure.type;
        debugBaseColor = material_base_color(material);

        const bool deferredRestirActivityDebug =
            PT_DEFAULT_OFF(uniforms.debugViewMode) == kDebugViewRestirPathGuidingMask ||
            PT_DEFAULT_OFF(uniforms.debugViewMode) == kDebugViewRestirPtReuseMask;
        if (PT_DEFAULT_OFF(uniforms.debugViewMode) != kDebugViewNone &&
            PT_DEFAULT_OFF(uniforms.debugViewMode) != kDebugViewCausticTransport &&
            !deferredRestirActivityDebug) {
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

        const bool primaryVisibilityHit = primaryVisibilityRay &&
                                          volumeScatteringDepth == 0u;

        // Keep geometric features at the primary surface. OIDN can use the
        // underlying non-delta surface through Pavilion water and glass.
        if (isFirstHit) {
            isFirstHit = false;
            pavilionGuideAverage = uniforms.oidnDeltaGuidesAllowed != 0u &&
                (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u;
            if (outFirstHitPosition != nullptr) {
                *outFirstHitPosition = float4(rec.point, 1.0f);
            }
            if (outFirstHitMaterial != nullptr) {
                *outFirstHitMaterial = spectralState.enabled ? spectralGuideFeatures
                                                             : material_closure_aov_features(material);
            }
        }
        if (isGuidePending && (!pavilionGuideAverage || !surfaceIsDelta)) {
            isGuidePending = false;
            if (outFirstHitAlbedo != nullptr) {
                *outFirstHitAlbedo = spectralState.enabled ? spectralGuideAlbedo
                                                           : material_closure_aov_albedo(material);
            }
            if (outFirstHitNormal != nullptr) {
                *outFirstHitNormal = shadingNormal;
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
        uint risAuditRngBefore = 0u;
        uint risAuditRngAfter = 0u;
        uint risAuditTemporalRngBefore = 0u;
        uint risAuditTemporalRngAfter = 0u;
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
            primaryVisibilityHit &&
            (material.materialFlags & kMaterialFlagFoliageBeautyLift) != 0u &&
            (rec.frontFace != 0u || rec.twoSided != 0u)) {
            float3 foliageVisible = clamp(material.baseColorRoughness.xyz * 1.15f,
                                          float3(0.0f),
                                          float3(1.0f));
            radiance += clamp_firefly_contribution(throughput, foliageVisible, clampParams);
            break;
        }

        if (!specularOnly &&
            material_closure_is_emissive(material) &&
            !material_closure_primary_is_emissive(material) &&
            (rec.frontFace != 0u || rec.twoSided != 0u)) {
            float3 visibleEmission = camera_visible_emission(material, depth);
            if (depth > 0u && !lastScatterWasDelta && previousMeshNeeEligible) {
                visibleEmission *= mesh_emitter_bsdf_hit_weight(uniforms, emissivePrimitives, rec,
                                                                previousMeshNeeOrigin, lastBsdfPdf);
            }
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

        if (material_closure_primary_is_emissive(material)) {
            if (specularOnly) {
                break;
            }
            float3 emission = material_closure_aov_emission(material);
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
                emission *= spectral_radiance_to_channels(envColor, spectralState);
            }
            if (any(emission != float3(0.0f)) &&
                (rec.frontFace != 0u || rec.twoSided != 0u)) {
                float misWeight = 1.0f;
                bool useSpecularMis =
                    use_visible_emitter_mis(depth, lastScatterWasDelta, lastDeltaUnpaired, uniforms);
                if (useSpecularMis && rectLightCount > 0u && (!lastScatterWasVolume || lastVolumeRectNee)) {
                    float lightPdf = rect_light_pdf_for_hit(uniforms,
                                                            rectangles,
                                                            materials,
                                                            rectLightCount,
                                                            rec,
                                                            holeMisOriginValid ? holeMisOrigin : ray.origin);
                    float denom = lastBsdfPdf + lightPdf;
                    if (denom > 0.0f) {
                        misWeight = clamp(lastBsdfPdf / denom,
                                          kMisWeightClampMin,
                                          kMisWeightClampMax);
                    }
                }
                // Mesh emitters pair with the mesh-emitter NEE, not the rect lights.
                if (rec.primitiveType == kPrimitiveTypeTriangle &&
                    depth > 0u && !lastScatterWasDelta && previousMeshNeeEligible) {
                    misWeight = mesh_emitter_bsdf_hit_weight(uniforms, emissivePrimitives, rec,
                                                             previousMeshNeeOrigin, lastBsdfPdf);
                }
                float3 contribution = emission * misWeight;
                caustic_transport_note_light_endpoint(uniforms,
                                                      rec.point,
                                                      shadingNormal,
                                                      ray.direction,
                                                      emission,
                                                      rectLightCount > 0u
                                                          ? rect_light_pdf_for_hit(uniforms,
                                                                                   rectangles,
                                                                                   materials,
                                                                                   rectLightCount,
                                                                                   rec,
                                                                                   ray.origin)
                                                          : 0.0f,
                                                      lastBsdfPdf,
                                                      depth,
                                                      mediumDepth,
                                                      causticLightVertices,
                                                      stats);
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

        // Subsurface (see subsurface_vertex_begin): a random-walk hit moves the shading vertex to the point where
        // the walk leaves the mesh; any other subsurface hit is shaded as its dense limit.
        bool subsurfaceExitVertex = false;
        if (material_is_subsurface(material)) {
            if (subsurface_vertex_begin(uniforms, material, rec, shadingNormal, wo, specularOnly, state, throughput) ==
                kSubsurfaceVertexWalk) {
                HitRecord exitRec;
                float3 walkWeight = float3(0.0f);
                uint walkEvent = sss_random_walk_software(uniforms, material, rec, spheres, rectangles, triangleData,
                                                          tlasNodes, tlasPrimIndices, instanceInfos, blasNodes,
                                                          blasPrimIndices, nodes, primitiveIndices,
                                                          meshInfos, sceneVertices, meshIndices, stats, state,
                                                          exitRec, walkWeight);
                if (walkEvent != kSssWalkExit) {
                    break;
                }
                throughput *= walkWeight;
                subsurface_vertex_exit(material, rec, exitRec);
                shadingNormal = rec.shadingNormal;
                wo = shadingNormal;
                incidentDir = -wo;
                diffuseOcclusion = 1.0f;
                subsurfaceExitVertex = true;
            }
            if (!material_is_subsurface(material)) {
                // The explicit ReSTIR DI passes skip every subsurface surface (its evaluation is marked BSSRDF), so a
                // vertex shaded as a walk's exit point or as the dense limit takes its direct light here instead.
                explicitRestirDiLightingStage = false;
            }
            primaryClosure = material_primary_closure_metadata(material);
            closureType = primaryClosure.type;
            surfaceIsDelta = material_closure_is_delta(material);
        }

        const bool restirDiProviderDirectLighting =
            (uniforms.directLightMode == kDirectLightModeRestirDi ||
             uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
            restir_di_make_provider_set(uniforms, rectangles, materials).providerCount > 0u;
        if (!explicitRestirDiLightingStage &&
            !surfaceIsDelta &&
            uniforms.directLightMode != kDirectLightModeLegacyRect &&
            (uniforms.emissivePrimitiveCount > 0u || restirDiProviderDirectLighting)) {
            if (uniforms.directLightMode == kDirectLightModeBaselineEmissive) {
                const uint lightSampleCount = baseline_direct_light_sample_count(uniforms, emissivePrimitives);
                // Keep one copy of the visibility/texture path in the megakernel.
#pragma clang loop unroll(disable)
                for (uint lightSampleIndex = 0u; lightSampleIndex < lightSampleCount; ++lightSampleIndex) {
                    RISSamplePayload lightSample;
                    if (sample_direct_light_baseline(uniforms,
                                                     emissivePrimitives,
                                                     rec,
                                                     state,
                                                     lightSample) &&
                        resolve_textured_direct_light_emission(uniforms, lightSample, materials,
                            materialTextures, materialSamplers, materialTextureInfos,
                            meshInfos, sceneVertices, meshIndices)) {
                        lightSample.emission = spectral_radiance_to_channels(lightSample.emission, spectralState);
                        const bool lightSampleDirectional = ris_payload_is_directional(lightSample);
                        float distSq = ris_payload_distance_sq(rec, lightSample);
                        float3 omegaL = ris_payload_omega_l(rec, lightSample);
                        float cosThetaSurface =
                            two_sided_alpha_cutout_direct_cos(uniforms, material, rec, shadingNormal, omegaL);
                        float cosThetaLight = lightSampleDirectional ? 1.0f : saturate(emissive_light_cosine(lightSample.flags, lightSample.normal, -omegaL));
                        if (lightSample.pdf > 0.0f &&
                            distSq > 0.0f &&
                            cosThetaSurface > 0.0f &&
                            cosThetaLight > 0.0f) {
                            Ray shadowRay;
                            shadowRay.origin = offset_spawn_origin(rec, omegaL, material);
                            shadowRay.direction = omegaL;
                            HitRecord shadowRec = make_empty_hit_record();
                            float shadowMax = lightSampleDirectional ? 1.0e20f : max(length(lightSample.position - rec.point) - kEpsilon, kEpsilon);
                            if (!lightSampleDirectional && (lightSample.flags & (kLightPrimitiveFlagTwoSided | kLightPrimitiveFlagTexturedEmission)) != 0u) {
                                const float3 visibilityDelta = lightSample.position - shadowRay.origin;
                                const float visibilityDistance = length(visibilityDelta);
                                shadowRay.direction = visibilityDelta / visibilityDistance;
                                shadowMax = max(visibilityDistance - kEpsilon, kEpsilon);
                            }
                            bool occluded = trace_scene_software_alpha_aware_shadow<false>(
                                uniforms, spheres, rectangles, triangleData,
                                tlasNodes, tlasPrimIndices, instanceInfos,
                                blasNodes, blasPrimIndices, nodes, primitiveIndices,
                                stats, shadowRay, spawn_ray_min(material, kEpsilon), shadowMax, materials,
                                materialTextures, materialSamplers, materialTextureInfos,
                                meshInfos, sceneVertices, meshIndices, shadowRec,
                                pavilion_punctual_shadow_enabled(material, lightSampleDirectional), depth);
                            // As the hardware megakernel: the origin is offset toward the light by up to t * 1e-4 while
                            // shadowMax is measured from rec.point, so a far or slanted segment can reach the sampled
                            // light itself. That hit is the light, not a blocker.
                            if (!lightSampleDirectional &&
                                occluded &&
                                shadowRec.primitiveType == kPrimitiveTypeTriangle &&
                                shadowRec.primitiveIndex == lightSample.primitiveIndex) {
                                occluded = false;
                            }
                            if (!occluded) {
                                BsdfEvalResult bsdfEval = material_closure_eval(material,
                                                                        rec.point,
                                                                        shadingNormal,
                                                                        wo,
                                                                        omegaL,
                                                                        clampParams,
                                                                        PT_DEFAULT_OFF(uniforms.sssMode),
                                                                        diffuseOcclusion,
                                                                        specularOnly,
                                                                        lightSampleDirectional);
                                const bool cutoutBackLightUnsampled = apply_two_sided_alpha_cutout_diffuse_fallback(uniforms,
                                                                              material,
                                                                              rec,
                                                                              shadingNormal,
                                                                              omegaL,
                                                                              diffuseOcclusion,
                                                                              bsdfEval);
                                if (!bsdfEval.isDelta && !bsdfEval.isBssrdf) {
                                    float3 value = lightSampleDirectional ? bsdfEval.value
                                        : reference_continuation_nee_value(material, shadingNormal, wo, omegaL, bsdfEval.value);
                                    // Every mesh emitter is also reachable by the BSDF scatter.
                                    const float weight = lightSampleDirectional ? 1.0f
                                        : mesh_emitter_nee_mis_weight(uniforms, depth, lightSample.pdf, nee_mis_bsdf_pdf(bsdfEval, cutoutBackLightUnsampled));
                                    float3 contribution = value * lightSample.emission;
                                    contribution *= weight * cosThetaSurface / max(lightSample.pdf, 1.0e-6f);
                                    // Medium transmittance over the shadow segment (see volume_surface_nee_transmittance).
                                    contribution *= volume_surface_nee_transmittance(
                                        uniforms, sqrt(distSq), lightSampleDirectional, stats);
                                    if (all(isfinite(contribution))) {
                                        // Average before recording the reconstruction split, while
                                        // keeping the existing clamp on each independent draw.
                                        const float3 averaged = (clamp_firefly_contribution)(
                                            throughput, contribution, clampParams, stats) / float(lightSampleCount);
                                        radiance += reconstruction_split_record(averaged, depth,
                                            outDirectRadiance, outIndirectRadiance);
                                    }
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
                risAuditRngBefore = state;
                risAuditRngAfter = state;
                risAuditTemporalRngBefore = 0u;
                risAuditTemporalRngAfter = 0u;
#if PT_DEBUG_TOOLS
                record_ris_reservoir_candidate_audit(debugContext,
                                                     uniforms,
                                                     emissivePrimitives,
                                                     rec,
                                                     material,
                                                     shadingNormal,
                                                     wo,
                                                     clampParams,
                                                     diffuseOcclusion,
                                                     specularOnly,
                                                     risTargetM,
                                                     risAuditRngBefore,
                                                     depth,
                                                     mediumDepth,
                                                     rec.materialIndex);
#endif
                if (uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) {
                    build_restir_di_provider_reservoir_for_hit(uniforms,
                                                               emissivePrimitives,
                                                               rectangles,
                                                               materials,
                                                               environmentTexture,
                                                               environmentConditionalAlias,
                                                               environmentMarginalAlias,
                                                               environmentPdf,
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
                    if (stats) {
                        atomic_fetch_add_explicit(&stats->restirDiRegirCandidateCount,
                                                  1u,
                                                  memory_order_relaxed);
                        if (reservoir.valid) {
                            atomic_fetch_add_explicit(&stats->restirDiRegirAcceptedCount,
                                                      1u,
                                                      memory_order_relaxed);
                        } else {
                            atomic_fetch_add_explicit(&stats->restirDiRegirFallbackCount,
                                                      1u,
                                                      memory_order_relaxed);
                        }
                    }
                } else {
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
                }
                risAuditRngAfter = state;

                // Runs when the pixel's own reservoir is empty too: the neighbours' MIS weights assume the
                // canonical technique always takes part, so skipping them there would lose their share.
                // A default-features gate: the fast hardware pipeline folds this block out.
                if (PT_DEFAULT_OFF(uniforms.directLightMode == kDirectLightModeRisSpatialReuse ? 1u : 0u) != 0u &&
                    (depth == 0u && !subsurfaceExitVertex) &&
                    rec.t > 0.0f &&
                    uniforms.width > 0u &&
                    uniforms.height > 0u) {
                    spatialReuseAttempted = true;
                    spatialNeighborTarget = min(max(uniforms.spatialReuseNeighborCount, 1u), 4u);
                    // Generalized RIS with pairwise MIS (reservoir.metal): neighbour winners stream into
                    // `merged`, then the canonical winner joins with its share of every pair. The old merge
                    // weighted each neighbour by p_c / (p_c M + p_n M), kept the canonical unweighted and added
                    // 1 to M per neighbour; on one light that converges to (M + 2) / (M + 4) of the exact value.
                    // Targets are compared per unit light area (target times cos(theta_light) / d^2), which carries
                    // the solid-angle Jacobian of moving a triangle-light sample between pixels.
                    const float pairCount = float(spatialNeighborTarget);
                    const bool canonicalValid = reservoir.valid;
                    const RISSamplePayload canonicalWinner = reservoir.winner;
                    const float canonicalAreaTarget = canonicalValid
                        ? reservoir_winner_phat_for_hit(uniforms,
                                                        rec,
                                                        material,
                                                        shadingNormal,
                                                        wo,
                                                        clampParams,
                                                        diffuseOcclusion,
                                                        specularOnly,
                                                        canonicalWinner) *
                              ris_payload_solid_angle_per_area(rec, canonicalWinner)
                        : 0.0f;
                    // A pair whose neighbour is out of frame, misses, or fails the depth or normal test belongs
                    // to the canonical.
                    float canonicalMis = 0.0f;
                    Reservoir merged = make_empty_reservoir();
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
                            canonicalMis += 1.0f / pairCount;
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
                            canonicalMis += 1.0f / pairCount;
                            continue;
                        }

                        float depthRel = fabs(rec.t - neighborRec.t) / max(rec.t, kEpsilon);
                        if (!(depthRel <= 0.1f) || !isfinite(depthRel)) {
                            spatialRejectedDepth += 1u;
                            canonicalMis += 1.0f / pairCount;
                            continue;
                        }
                        float normalDot = dot(safe_normalize(shadingNormal),
                                              safe_normalize(neighborRec.shadingNormal));
                        if (!(normalDot >= 0.9f) || !isfinite(normalDot)) {
                            spatialRejectedNormal += 1u;
                            canonicalMis += 1.0f / pairCount;
                            continue;
                        }

                        spatialNeighborsConsidered += 1u;
                        MaterialData neighborMaterial = materials[neighborRec.materialIndex];
                        float3 neighborWo = safe_normalize(-neighborRay.direction);
                        if (canonicalValid) {
                            canonicalMis += pairwise_mis_canonical_share(
                                canonicalAreaTarget,
                                reservoir_winner_phat_for_hit(uniforms,
                                                              neighborRec,
                                                              neighborMaterial,
                                                              neighborRec.shadingNormal,
                                                              neighborWo,
                                                              clampParams,
                                                              diffuseOcclusion,
                                                              specularOnly,
                                                              canonicalWinner) *
                                    ris_payload_solid_angle_per_area(neighborRec, canonicalWinner),
                                pairCount);
                        }
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
                        float mergeWeight = pairwise_mis_neighbor_weight(
                            phatCurrent * ris_payload_solid_angle_per_area(rec, neighborReservoir.winner),
                            phatNeighbor * ris_payload_solid_angle_per_area(neighborRec, neighborReservoir.winner),
                            neighborReservoir.wSum,
                            pairCount);
                        if (!(mergeWeight > 0.0f)) {
                            spatialRejectedInvalid += 1u;
                            continue;
                        }
                        bool changed = update_reservoir(merged,
                                                        neighborReservoir.winner,
                                                        mergeWeight,
                                                        rand_uniform(state));
                        spatialNeighborsAccepted += 1u;
                        spatialLastMisWeight = mergeWeight / neighborReservoir.wSum;
                        spatialLastMergeWeight = mergeWeight;
                        if (changed) {
                            spatialWinnerOffset = offset;
                        }
                    }
                    if (canonicalValid &&
                        update_reservoir(merged,
                                         canonicalWinner,
                                         canonicalMis * reservoir.wSum,
                                         rand_uniform(state))) {
                        spatialWinnerOffset = int2(0, 0);
                    }
                    if (merged.valid) {
                        merged.M = risTargetM;  // every reservoir merged here holds risTargetM candidates
                    }
                    reservoir = merged;
                }

                if ((uniforms.directLightMode == kDirectLightModeRisTemporalReuse ||
                     uniforms.directLightMode == kDirectLightModeRestirDi ||
                     uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
                    (depth == 0u && !subsurfaceExitVertex)) {
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
                        risAuditTemporalRngBefore = temporalState;
                        Reservoir previousReservoir = make_empty_reservoir();
                        const bool previousReservoirValid =
                            uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid
                                ? build_restir_di_provider_reservoir_for_hit(uniforms,
                                                                             emissivePrimitives,
                                                                             rectangles,
                                                                             materials,
                                                                             environmentTexture,
                                                                             environmentConditionalAlias,
                                                                             environmentMarginalAlias,
                                                                             environmentPdf,
                                                                             rec,
                                                                             material,
                                                                             shadingNormal,
                                                                             wo,
                                                                             clampParams,
                                                                             diffuseOcclusion,
                                                                             specularOnly,
                                                                             risTargetM,
                                                                             temporalState,
                                                                             previousReservoir)
                                : build_ris_reservoir_for_hit(uniforms,
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
                                                              previousReservoir);
                        if (!previousReservoirValid) {
                            risAuditTemporalRngAfter = temporalState;
                            temporalRejectedInvalid += 1u;
                        } else {
                            risAuditTemporalRngAfter = temporalState;
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

                if (PT_DEFAULT_OFF(uniforms.directLightMode == kDirectLightModeRisWorldReuse ? 1u : 0u) != 0u &&
                    (depth == 0u && !subsurfaceExitVertex)) {
                    worldReuseAttempted = true;
                    if (rec.t > 0.0f &&
                        uniforms.width > 0u &&
                        uniforms.height > 0u) {
                        worldCell = world_reuse_cell(rec.point, uniforms.worldReuseCellSize);
                        worldCellHash = world_reuse_cell_hash(worldCell);
                        // Generalized RIS with pairwise MIS, as ris_spatial (reservoir.metal): each candidate pixel's
                        // fresh reservoir streams into `merged` with its pairwise weight, then the canonical winner
                        // joins with its share of every pair. The old merge weighted each candidate by p_c / (p_c M +
                        // p_n M), kept the canonical unweighted and added 1 to M per candidate: 0.824 of the exact
                        // light on the EP12 sun plane. Targets are compared per unit light area, which carries the
                        // solid-angle Jacobian of moving a triangle-light sample between pixels.
                        constexpr uint kWorldReusePairs = 4u;
                        const float pairCount = float(kWorldReusePairs);
                        const bool canonicalValid = reservoir.valid;
                        const RISSamplePayload canonicalWinner = reservoir.winner;
                        const float canonicalAreaTarget = canonicalValid
                            ? reservoir_winner_phat_for_hit(uniforms,
                                                            rec,
                                                            material,
                                                            shadingNormal,
                                                            wo,
                                                            clampParams,
                                                            diffuseOcclusion,
                                                            specularOnly,
                                                            canonicalWinner) *
                                  ris_payload_solid_angle_per_area(rec, canonicalWinner)
                            : 0.0f;
                        // A pair whose candidate is out of frame, misses, or fails the cell, depth or normal test
                        // belongs to the canonical.
                        float canonicalMis = 0.0f;
                        Reservoir merged = make_empty_reservoir();
                        constexpr int2 offsets[4] = {
                            int2(-2, 0),
                            int2(2, 0),
                            int2(0, -2),
                            int2(0, 2)
                        };
                        for (uint n = 0u; n < kWorldReusePairs; ++n) {
                            int2 candidateCoordI = int2(pixelCoord) + offsets[n];
                            if (candidateCoordI.x < 0 ||
                                candidateCoordI.y < 0 ||
                                candidateCoordI.x >= int(uniforms.width) ||
                                candidateCoordI.y >= int(uniforms.height)) {
                                worldRejectedInvalid += 1u;
                                canonicalMis += 1.0f / pairCount;
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
                                canonicalMis += 1.0f / pairCount;
                                continue;
                            }

                            int3 candidateCell =
                                world_reuse_cell(candidateRec.point, uniforms.worldReuseCellSize);
                            if (!world_reuse_cell_compatible(worldCell, candidateCell)) {
                                worldRejectedCell += 1u;
                                canonicalMis += 1.0f / pairCount;
                                continue;
                            }

                            float depthRel = fabs(rec.t - candidateRec.t) / max(rec.t, kEpsilon);
                            if (!(depthRel <= 0.2f) || !isfinite(depthRel)) {
                                worldRejectedDepth += 1u;
                                canonicalMis += 1.0f / pairCount;
                                continue;
                            }
                            float normalDot = dot(safe_normalize(shadingNormal),
                                                  safe_normalize(candidateRec.shadingNormal));
                            if (!(normalDot >= 0.85f) || !isfinite(normalDot)) {
                                worldRejectedNormal += 1u;
                                canonicalMis += 1.0f / pairCount;
                                continue;
                            }

                            worldCandidatesConsidered += 1u;
                            MaterialData candidateMaterial = materials[candidateRec.materialIndex];
                            float3 candidateWo = safe_normalize(-candidateRay.direction);
                            if (canonicalValid) {
                                canonicalMis += pairwise_mis_canonical_share(
                                    canonicalAreaTarget,
                                    reservoir_winner_phat_for_hit(uniforms,
                                                                  candidateRec,
                                                                  candidateMaterial,
                                                                  candidateRec.shadingNormal,
                                                                  candidateWo,
                                                                  clampParams,
                                                                  diffuseOcclusion,
                                                                  specularOnly,
                                                                  canonicalWinner) *
                                        ris_payload_solid_angle_per_area(candidateRec, canonicalWinner),
                                    pairCount);
                            }
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
                                                             candidateReservoir) ||
                                !candidateReservoir.valid) {
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
                            float mergeWeight = pairwise_mis_neighbor_weight(
                                phatCurrent * ris_payload_solid_angle_per_area(rec, candidateReservoir.winner),
                                phatCandidate *
                                    ris_payload_solid_angle_per_area(candidateRec, candidateReservoir.winner),
                                candidateReservoir.wSum,
                                pairCount);
                            if (!(mergeWeight > 0.0f)) {
                                worldRejectedInvalid += 1u;
                                continue;
                            }
                            update_reservoir(merged,
                                             candidateReservoir.winner,
                                             mergeWeight,
                                             rand_uniform(state));
                            worldCandidatesAccepted += 1u;
                            worldReuseAccepted = true;
                            worldCandidatePrimitiveIndex = candidateReservoir.winner.primitiveIndex;
                            worldLastMisWeight = mergeWeight / candidateReservoir.wSum;
                            worldLastMergeWeight = mergeWeight;
                        }
                        if (canonicalValid) {
                            update_reservoir(merged,
                                             canonicalWinner,
                                             canonicalMis * reservoir.wSum,
                                             rand_uniform(state));
                        }
                        if (merged.valid) {
                            merged.M = risTargetM;  // every reservoir merged here holds risTargetM candidates
                        }
                        reservoir = merged;
                    } else {
                        worldRejectedInvalid += 1u;
                    }
                }

                if ((uniforms.directLightMode == kDirectLightModeRisRegirCache ||
                     uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
                    (depth == 0u && !subsurfaceExitVertex) &&
                    uniforms.frameIndex > 0u) {
                    cacheReuseAttempted = true;
                    cacheFallbackUsed = true;
                    cacheCell = world_reuse_cell(rec.point, uniforms.worldReuseCellSize);
                    cacheCellHash = world_reuse_cell_hash(cacheCell);
                    if (uniforms.frameIndex > 1u) {
                        cacheSourceFrameIndex = uniforms.frameIndex - 2u;
                    } else if (uniforms.frameIndex > 0u) {
                        cacheSourceFrameIndex = uniforms.frameIndex - 1u;
                    }
                    if (rec.t > 0.0f &&
                        uniforms.frameIndex > 0u) {
                        cacheStateAvailable = true;
                        cacheFallbackUsed = false;
                        cacheCandidatesConsidered = 0u;
                        cacheEntriesAvailable = 0u;
                        // The cached reservoirs are fresh RIS reservoirs built at this pixel's own hit point, with the
                        // same builder and target function as the pixel's reservoir and with random numbers seeded per
                        // cell and frame. Reservoirs with one target at one point merge by plain streaming: each adds
                        // its wSum and M grows by its candidate count (generalized RIS with MIS weights in proportion
                        // to M). Every probe is merged, valid or not, since each drew risTargetM candidates. The old
                        // merge weighted each cached reservoir by p(y_c) / (p(y) M + p(y_c) M_c), kept the two with the
                        // largest weights (a choice that depends on the samples) and added 1 to M per merge: 0.946 of
                        // the exact light on the EP12 sun plane under ris_regir and 0.962 under the hybrid.
                        // The pixel's reservoir enters with its own contribution weight wSum / M and a fixed share of
                        // nominalM candidates: risTargetM, or twice that for the hybrid, whose temporal step merges a
                        // second reservoir of risTargetM candidates when both are valid. Its actual M depends on the
                        // samples there, so it cannot be its share: MIS weights that depend on the samples would bias
                        // the merge.
                        const uint nominalM =
                            (uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid ? 2u : 1u) * risTargetM;
                        if (reservoir.valid && reservoir.M != nominalM) {
                            reservoir.wSum *= float(nominalM) / float(reservoir.M);
                        }
                        uint mergedM = nominalM;
                        const uint firstSourceFrameIndex = cacheSourceFrameIndex;
                        for (uint cacheProbe = 0u; cacheProbe < 4u; ++cacheProbe) {
                            if (cacheProbe > firstSourceFrameIndex) {
                                continue;
                            }
                            uint candidateSourceFrameIndex = firstSourceFrameIndex - cacheProbe;
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
                            const bool cachedReservoirValid =
                                uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid
                                ? build_restir_di_provider_reservoir_for_hit(uniforms,
                                                                             emissivePrimitives,
                                                                             rectangles,
                                                                             materials,
                                                                             environmentTexture,
                                                                             environmentConditionalAlias,
                                                                             environmentMarginalAlias,
                                                                             environmentPdf,
                                                                             rec,
                                                                             material,
                                                                             shadingNormal,
                                                                             wo,
                                                                             clampParams,
                                                                             diffuseOcclusion,
                                                                             specularOnly,
                                                                             risTargetM,
                                                                             cacheState,
                                                                             cachedReservoir)
                                : build_ris_reservoir_for_hit(uniforms,
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
                                                              cachedReservoir);
                            mergedM += risTargetM;
                            if (!cachedReservoirValid || !cachedReservoir.valid) {
                                cacheRejectedInvalid += 1u;
                                continue;
                            }
                            cacheEntriesAvailable += 1u;
                            if (update_reservoir(reservoir,
                                                 cachedReservoir.winner,
                                                 cachedReservoir.wSum,
                                                 rand_uniform(state))) {
                                cacheSourceFrameIndex = candidateSourceFrameIndex;
                                cacheCandidatePrimitiveIndex = cachedReservoir.winner.primitiveIndex;
                            }
                            cacheCandidatesAccepted += 1u;
                            cacheReuseAccepted = true;
                            cacheLastMergeWeight = cachedReservoir.wSum;
                        }
                        if (reservoir.valid) {
                            reservoir.M = mergedM;
                            // MIS weight of one cached reservoir.
                            cacheLastMisWeight = float(risTargetM) / float(mergedM);
                        }
                    } else {
                        cacheRejectedInvalid += 1u;
                    }
                    if (stats) {
                        if (cacheCandidatesConsidered > 0u) {
                            atomic_fetch_add_explicit(&stats->restirDiRegirCandidateCount,
                                                      cacheCandidatesConsidered,
                                                      memory_order_relaxed);
                        }
                        if (cacheCandidatesAccepted > 0u) {
                            atomic_fetch_add_explicit(&stats->restirDiRegirAcceptedCount,
                                                      cacheCandidatesAccepted,
                                                      memory_order_relaxed);
                        } else if (cacheReuseAttempted) {
                            atomic_fetch_add_explicit(&stats->restirDiRegirFallbackCount,
                                                      1u,
                                                      memory_order_relaxed);
                        }
                    }
                }

                if (reservoir.valid && reservoir.M > 0u && reservoir.wSum > 0.0f) {
                    RISSamplePayload winner = reservoir.winner;
                    float distSq = ris_payload_distance_sq(rec, winner);
                    if (distSq > 0.0f && isfinite(distSq)) {
                        const bool winnerDirectional = ris_payload_is_directional(winner);
                        float3 omegaL = ris_payload_omega_l(rec, winner);
                        float cosThetaSurface = direct_light_target_cos(uniforms, material, rec, shadingNormal, omegaL);
                        float cosThetaLight = winnerDirectional ? 1.0f : saturate(emissive_light_cosine(winner.flags, winner.normal, -omegaL));
                        bool cutoutBackLightUnsampled = false;
                        BsdfEvalResult bsdfEval = direct_light_target_bsdf(uniforms,
                                                                           material,
                                                                           rec,
                                                                           shadingNormal,
                                                                           wo,
                                                                           omegaL,
                                                                           clampParams,
                                                                           diffuseOcclusion,
                                                                           specularOnly,
                                                                           winnerDirectional,
                                                                           cutoutBackLightUnsampled);
                        float phatW = p_hat(winner.emission,
                                            bsdfEval.value,
                                            cosThetaSurface,
                                            cosThetaLight,
                                            distSq);
                        reservoir.W = (phatW > 0.0f)
                                    ? ((reservoir.wSum / float(reservoir.M)) / phatW)
                                    : 0.0f;

                        Ray shadowRay;
                        shadowRay.origin = offset_spawn_origin(rec, omegaL, material);
                        shadowRay.direction = omegaL;
                        HitRecord shadowRec = make_empty_hit_record();
                        float shadowMax = winnerDirectional ? 1.0e20f : max(length(winner.position - rec.point) - kEpsilon, kEpsilon);
                        // The same alpha-aware shadow test as plain NEE and the hardware megakernel's reservoir
                        // shading: cutouts let light through their holes, and Pavilion punctual shadows apply.
                        bool occluded = trace_scene_software_alpha_aware_shadow<false>(
                            uniforms, spheres, rectangles, triangleData,
                            tlasNodes, tlasPrimIndices, instanceInfos,
                            blasNodes, blasPrimIndices, nodes, primitiveIndices,
                            stats, shadowRay, spawn_ray_min(material, kEpsilon), shadowMax, materials,
                            materialTextures, materialSamplers, materialTextureInfos,
                            meshInfos, sceneVertices, meshIndices, shadowRec,
                            pavilion_punctual_shadow_enabled(material, winnerDirectional), depth);
                        if (!winnerDirectional &&
                            occluded &&
                            shadowRec.primitiveType == kPrimitiveTypeTriangle &&
                            shadowRec.primitiveIndex == winner.primitiveIndex) {
                            occluded = false;
                        }
                        if (!winnerDirectional &&
                            (winner.flags & kLightPrimitiveFlagRectangle) != 0u &&
                            occluded &&
                            shadowRec.primitiveType == kPrimitiveTypeRectangle &&
                            shadowRec.primitiveIndex == winner.primitiveIndex) {
                            occluded = false;
                        }

                        float3 contribution = float3(0.0f);
                        // Candidates are weighted by the light's emission factor; shade with
                        // the textured emission at the chosen point, which the BSDF hit sees.
                        RISSamplePayload shadedWinner = winner;
                        const bool winnerEmissionResolved =
                            resolve_textured_direct_light_emission(uniforms, shadedWinner, materials,
                                materialTextures, materialSamplers, materialTextureInfos,
                                meshInfos, sceneVertices, meshIndices);
                        // Plain RIS shares mesh emitters with the BSDF hit
                        // (mesh_emitter_bsdf_hit_weight); reuse modes keep weight 1.
                        const float risMisWeight = winnerDirectional ? 1.0f
                            : mesh_emitter_nee_mis_weight(uniforms, depth,
                                  mesh_emitter_nee_candidate_count(uniforms) * winner.pdf, nee_mis_bsdf_pdf(bsdfEval, cutoutBackLightUnsampled));
                        if (!occluded &&
                            winnerEmissionResolved &&
                            !bsdfEval.isDelta &&
                            !bsdfEval.isBssrdf &&
                            reservoir.W > 0.0f &&
                            cosThetaSurface > 0.0f) {
                            contribution = bsdfEval.value * shadedWinner.emission;
                            contribution *= cosThetaSurface * reservoir.W * risMisWeight;
                            contribution *= volume_surface_nee_transmittance(
                                uniforms, sqrt(distSq), winnerDirectional, stats);
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
        }
        // Analytic rectangles are a separate light population from emissive mesh
        // primitives. Sample both when RIS handles only the latter.
        const uint independentRectFilter = kAnalyticLightFilterAll;
        const uint independentRectCount = rectLightCount;
        if (!surfaceIsDelta && independentRectCount > 0u) {
            for (uint neeDraw = 0u; neeDraw < rect_light_nee_draw_count(uniforms); ++neeDraw) {
                RectLightSample lightSample;
                if (sample_rect_light_nee_draw(uniforms,
                                      rectangles,
                                      materials,
                                      environmentTexture,
                                      rec,
                                      state,
                                      independentRectCount,
                                      lightSample,
                                      independentRectFilter, neeDraw)) {
                    lightSample.emission = spectral_radiance_to_channels(lightSample.emission, spectralState);
                    float nDotL =
                        two_sided_alpha_cutout_direct_cos(uniforms, material, rec, shadingNormal, lightSample.direction);
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
                        shadowRay.origin = offset_spawn_origin(rec, lightSample.direction, material);
                        shadowRay.direction = lightSample.direction;
                        HitRecord shadowRec;
                        float shadowMax = max(lightSample.distance - kEpsilon, kEpsilon);
                        bool occluded;
                        if (pavilion_punctual_shadow_enabled(material, lightSample.isReferencePoint)) {
                            occluded = trace_scene_software_alpha_aware_shadow<false>(
                                uniforms, spheres, rectangles, triangleData,
                                tlasNodes, tlasPrimIndices, instanceInfos,
                                blasNodes, blasPrimIndices, nodes, primitiveIndices,
                                stats, shadowRay, spawn_ray_min(material, kEpsilon), shadowMax, materials,
                                materialTextures, materialSamplers, materialTextureInfos,
                                meshInfos, sceneVertices, meshIndices, shadowRec, true, depth);
                        } else {
                            occluded = trace_scene_software_alpha_aware_shadow<false>(
                                uniforms, spheres, rectangles, triangleData,
                                tlasNodes, tlasPrimIndices, instanceInfos,
                                blasNodes, blasPrimIndices, nodes, primitiveIndices,
                                stats, shadowRay, spawn_ray_min(material, kEpsilon), shadowMax, materials,
                                materialTextures, materialSamplers, materialTextureInfos,
                                meshInfos, sceneVertices, meshIndices, shadowRec, false, depth);
                        }
                        if (occluded &&
                            shadowRec.primitiveType == kPrimitiveTypeRectangle &&
                            shadowRec.primitiveIndex == lightSample.rectIndex) {
                            occluded = false;
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
                        if (!occluded) {
                            BsdfEvalResult bsdfEval = material_closure_eval(material,
                                                                    rec.point,
                                                                    shadingNormal,
                                                                    wo,
                                                                    lightSample.direction,
                                                                    clampParams,
                                                                    PT_DEFAULT_OFF(uniforms.sssMode),
                                                                    diffuseOcclusion,
                                                                    specularOnly,
                                                                    lightSample.isReferencePoint);
                            const bool cutoutBackLightUnsampled = apply_two_sided_alpha_cutout_diffuse_fallback(uniforms,
                                                                          material,
                                                                          rec,
                                                                          shadingNormal,
                                                                          lightSample.direction,
                                                                          diffuseOcclusion,
                                                                          bsdfEval);
                            float3 bsdfValue = lightSample.isReferencePoint ? bsdfEval.value
                                : reference_continuation_nee_value(material, shadingNormal, wo,
                                                                   lightSample.direction, bsdfEval.value);
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
                                    float bsdfPdf = nee_mis_bsdf_pdf(bsdfEval, cutoutBackLightUnsampled);
                                    float weight = 1.0f;
                                    const bool referenceSurface = (material.materialFlags &
                                        (kMaterialFlagSingleScatterSpecular | kMaterialFlagReferenceDielectricContinuation)) != 0u;
                                    if (bsdfPdf > 0.0f && !lightSample.isReferencePoint &&
                                        (!referenceSurface || depth + 1u < uniforms.maxDepth)) {
                                        float denom = lightSample.pdf + bsdfPdf;
                                        if (denom > 0.0f) {
                                            weight = clamp(lightSample.pdf / denom,
                                                           kMisWeightClampMin,
                                                           kMisWeightClampMax);
                                        }
                                    }
                                    float3 contribution = float3(0.0f);
                                    float causticMisWeight = weight;
                                    const bool causticConnected = !lightSample.isReferencePoint &&
                                        caustic_transport_connect_light_sample(uniforms,
                                                                               material,
                                                                               rec.point,
                                                                               shadingNormal,
                                                                               rec.normal,
                                                                               lightSample.direction,
                                                                               lightSample.position,
                                                                               lightSample.normal,
                                                                               false,
                                                                               throughput,
                                                                               lightSample.emission,
                                                                               bsdfValue,
                                                                               bsdfPdf,
                                                                               lightSample.pdf,
                                                                               nDotL,
                                                                               depth,
                                                                               mediumDepth,
                                                                               causticEyeVertices,
                                                                               causticLightVertices,
                                                                               causticConnections,
                                                                               causticConnectionResults,
                                                                               stats,
                                                                               contribution,
                                                                               causticMisWeight);
                                    if (!causticConnected) {
                                        contribution = lightSample.emission * bsdfValue * nDotL;
                                        contribution *= weight / lightSample.pdf;
                                    } else {
                                        weight = causticMisWeight;
                                    }
                                    contribution *= volume_surface_nee_transmittance(
                                        uniforms, lightSample.distance, false, stats);
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
}

        if (debugContext &&
            PT_DEFAULT_OFF(uniforms.debugDirectLightAudit) != 0u &&
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
                               float4(float(closureType),
                                      float(rec.primitiveIndex),
                                      material.typeEta.x,
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
            record_debug_event(*debugContext,
                               kDebugEventDirectLightAuditMaterialExt,
                               depth,
                               mediumDepth,
                               mediumDepth,
                               /*mediumEvent=*/0,
                               rec.frontFace,
                               rec.materialIndex,
                               /*scatterIsDelta=*/false,
                               material.pbrParams.xyz,
                               float4(material.pbrParams.w,
                                      material.typeEta.y,
                                      material.pbrExtras.z,
                                      diffuseOcclusion),
                               float3(material.pbrExtras.x,
                                      material.pbrExtras.y,
                                      material.pbrExtras.w));
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
                record_debug_event(*debugContext,
                                   kDebugEventRisAuditRng,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   float3(as_type<float>(risAuditRngBefore),
                                          as_type<float>(risAuditRngAfter),
                                          0.0f),
                                   float4(as_type<float>(risAuditTemporalRngBefore),
                                          as_type<float>(risAuditTemporalRngAfter),
                                          0.0f,
                                          0.0f),
                                   float3(0.0f));
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
                    float nDotL =
                        two_sided_alpha_cutout_direct_cos(uniforms, material, rec, shadingNormal, auditLight.direction);
                    BsdfEvalResult bsdfEval = material_closure_eval(material,
                                                            rec.point,
                                                            shadingNormal,
                                                            wo,
                                                            auditLight.direction,
                                                            clampParams,
                                                            PT_DEFAULT_OFF(uniforms.sssMode),
                                                            diffuseOcclusion,
                                                            specularOnly,
                                                            auditLight.isReferencePoint);
                    const bool cutoutBackLightUnsampled = apply_two_sided_alpha_cutout_diffuse_fallback(uniforms,
                                                                  material,
                                                                  rec,
                                                                  shadingNormal,
                                                                  auditLight.direction,
                                                                  diffuseOcclusion,
                                                                  bsdfEval);
                    float bsdfPdf = nee_mis_bsdf_pdf(bsdfEval, cutoutBackLightUnsampled);
                    float misWeight = 1.0f;
                    if (bsdfPdf > 0.0f && !auditLight.isReferencePoint) {
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
                    auditShadowRay.origin = offset_spawn_origin(rec, auditLight.direction, material);
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
                                                         spawn_ray_min(material, kEpsilon),
                                                         shadowMax,
                                                         /*anyHitOnly=*/true,
                                                         /*includeTriangles=*/true,
                                                         &swShadowAudit,
                                                         shadowRec);
                    swShadowAudit.consulted = true;
                    bool lightTwoSided = (rectangles[rectIndex].materialTwoSided.y != 0u);
                    // The audit reports what plain NEE sees. NEE's shadow rays pass alpha-cutout holes and BLEND texels (the
                    // alpha-aware tracers); the raw trace above, kept for its traversal record, stops at the first one.
                    {
                        HitRecord alphaShadowRec = make_empty_hit_record();
                        occluded = trace_scene_software_alpha_aware_shadow<false>(
                            uniforms, spheres, rectangles, triangleData,
                            tlasNodes, tlasPrimIndices, instanceInfos,
                            blasNodes, blasPrimIndices, nodes, primitiveIndices,
                            /*stats=*/nullptr, auditShadowRay, spawn_ray_min(material, kEpsilon), shadowMax, materials,
                            materialTextures, materialSamplers, materialTextureInfos,
                            meshInfos, sceneVertices, meshIndices, alphaShadowRec, pavilion_punctual_shadow_enabled(material, auditLight.isReferencePoint));
                    }
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
                    float nDotL =
                        two_sided_alpha_cutout_direct_cos(uniforms, material, rec, shadingNormal, auditLight.direction);
                    BsdfEvalResult bsdfEval = material_closure_eval(material,
                                                            rec.point,
                                                            shadingNormal,
                                                            wo,
                                                            auditLight.direction,
                                                            clampParams,
                                                            PT_DEFAULT_OFF(uniforms.sssMode),
                                                            diffuseOcclusion,
                                                            specularOnly,
                                                            (emissivePrimitives[primitiveAuditIndex].flags & kLightPrimitiveFlagDirectional) != 0u);
                    const bool cutoutBackLightUnsampled = apply_two_sided_alpha_cutout_diffuse_fallback(uniforms,
                                                                  material,
                                                                  rec,
                                                                  shadingNormal,
                                                                  auditLight.direction,
                                                                  diffuseOcclusion,
                                                                  bsdfEval);
                    float bsdfPdf = nee_mis_bsdf_pdf(bsdfEval, cutoutBackLightUnsampled);
                    float misWeight = 1.0f;
                    HitRecord shadowRec;
                    HardwareShadowTraceAudit hwShadowAudit;
                    SoftwareShadowTraceAudit swShadowAudit;
                    Ray auditShadowRay;
                    auditShadowRay.origin = offset_spawn_origin(rec, auditLight.direction, material);
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
                                                         spawn_ray_min(material, kEpsilon),
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
                    // The audit reports what plain NEE sees. NEE's shadow rays pass alpha-cutout holes and BLEND texels (the
                    // alpha-aware tracers); the raw trace above, kept for its traversal record, stops at the first one.
                    {
                        HitRecord alphaShadowRec = make_empty_hit_record();
                        occluded = trace_scene_software_alpha_aware_shadow<false>(
                            uniforms, spheres, rectangles, triangleData,
                            tlasNodes, tlasPrimIndices, instanceInfos,
                            blasNodes, blasPrimIndices, nodes, primitiveIndices,
                            /*stats=*/nullptr, auditShadowRay, spawn_ray_min(material, kEpsilon), shadowMax, materials,
                            materialTextures, materialSamplers, materialTextureInfos,
                            meshInfos, sceneVertices, meshIndices, alphaShadowRec, false);
                        if (occluded && alphaShadowRec.primitiveType == kPrimitiveTypeTriangle &&
                            alphaShadowRec.primitiveIndex == auditLight.primitiveIndex) {
                            occluded = false;
                        }
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
                if (useOverride) {
                    envSample.radiance = environment_color_lod(environmentTexture,
                                                               envSample.direction,
                                                               uniforms.environmentRotation,
                                                               uniforms.environmentIntensity,
                                                               overrideLod,
                                                               uniforms);
                }
                float nDotL =
                    two_sided_alpha_cutout_direct_cos(uniforms, material, rec, shadingNormal, envSample.direction);
                if (envSample.pdf > 0.0f && nDotL > 0.0f) {
                    BsdfEvalResult bsdfEval = material_closure_eval(material,
                                                            rec.point,
                                                            shadingNormal,
                                                            wo,
                                                            envSample.direction,
                                                            clampParams,
                                                            PT_DEFAULT_OFF(uniforms.sssMode),
                                                            diffuseOcclusion,
                                                            specularOnly);
                    const bool cutoutBackLightUnsampled = apply_two_sided_alpha_cutout_diffuse_fallback(uniforms,
                                                                  material,
                                                                  rec,
                                                                  shadingNormal,
                                                                  envSample.direction,
                                                                  diffuseOcclusion,
                                                                  bsdfEval);
                    float3 bsdfValue = reference_continuation_nee_value(material, shadingNormal,
                                                                                   wo, envSample.direction, bsdfEval.value);
                    float maxComponent = max(max(bsdfValue.x, bsdfValue.y), bsdfValue.z);
                    bool bsdfConnectable = !bsdfEval.isDelta &&
                                           !bsdfEval.isBssrdf &&
                                           (maxComponent > 0.0f);
                    if (bsdfConnectable) {
                        float bsdfPdf = nee_mis_bsdf_pdf(bsdfEval, cutoutBackLightUnsampled);
                        float weight = 1.0f;
                        const bool referenceSurface = (material.materialFlags &
                            (kMaterialFlagSingleScatterSpecular | kMaterialFlagReferenceDielectricContinuation)) != 0u;
                        if (bsdfPdf > 0.0f && (!referenceSurface || depth + 1u < uniforms.maxDepth)) {
                            float denom = envSample.pdf + bsdfPdf;
                            if (denom > 0.0f) {
                                weight = clamp(envSample.pdf / denom,
                                               kMisWeightClampMin,
                                               kMisWeightClampMax);
                            }
                        }

                        Ray shadowRay;
                        shadowRay.origin = offset_spawn_origin(rec, envSample.direction, material);
                        shadowRay.direction = envSample.direction;
                        HitRecord shadowRec = make_empty_hit_record();
                        // Alpha-aware, as the hardware megakernel's, the wavefront's and Embree's environment NEE: a
                        // path ray passes MASK holes as MIS null events, so this vertex's NEE must see through them.
                        bool occluded = trace_scene_software_alpha_aware_shadow<false>(
                            uniforms, spheres, rectangles, triangleData,
                            tlasNodes, tlasPrimIndices, instanceInfos,
                            blasNodes, blasPrimIndices, nodes, primitiveIndices,
                            stats, shadowRay, spawn_ray_min(material, kEpsilon), kInfinity, materials,
                            materialTextures, materialSamplers, materialTextureInfos,
                            meshInfos, sceneVertices, meshIndices, shadowRec, false, depth);
                        float3 envDirection = envSample.direction;
                        float3 chainWeight = float3(1.0f);
                        bool usedDeltaChain = false;
                        bool connected = !occluded;

#if ENABLE_MNEE_CAUSTICS
                        bool useMneeEnvChain = (PT_DEFAULT_OFF(uniforms.enableMnee) != 0u) &&
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
                            envRadiance = spectral_radiance_to_channels(envRadiance, spectralState);
                            float3 contribution = float3(0.0f);
                            float causticMisWeight = weight;
                            const bool causticConnected =
                                !usedDeltaChain &&
                                caustic_transport_connect_light_sample(uniforms,
                                                                       material,
                                                                       rec.point,
                                                                       shadingNormal,
                                                                       rec.normal,
                                                                       envSample.direction,
                                                                       rec.point + safe_normalize(envSample.direction),
                                                                       -safe_normalize(envSample.direction),
                                                                       true,
                                                                       throughput,
                                                                       envRadiance,
                                                                       bsdfValue,
                                                                       bsdfPdf,
                                                                       envSample.pdf,
                                                                       nDotL,
                                                                       depth,
                                                                       mediumDepth,
                                                                       causticEyeVertices,
                                                                       causticLightVertices,
                                                                       causticConnections,
                                                                       causticConnectionResults,
                                                                       stats,
                                                                       contribution,
                                                                       causticMisWeight);
                            if (!causticConnected) {
                                contribution = envRadiance * bsdfValue * nDotL;
                                contribution *= weight / envSample.pdf;
                            } else {
                                weight = causticMisWeight;
                            }
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
                                      float(closureType),
                                      0.0f,
                                      0.0f),
                               shadingNormal);
        }

        BsdfSampleResult bsdfSample;
        uint rngStateBeforeBsdf = state;
        bsdfSample = material_closure_sample(material,
                                             rec.point,
                                             shadingNormal,
                                             wo,
                                             incidentDir,
                                             rec.frontFace != 0u,
                                             state,
                                             clampParams,
                                             PT_DEFAULT_OFF(uniforms.sssMode),
                                             diffuseOcclusion,
                                             specularOnly);
        if (subsurfaceExitVertex) {
            // Keeps this vertex out of ReSTIR GI, ReSTIR PT, path guiding and the radiance cache, whose
            // per-pixel state belongs to the primary hit and not to the walk's exit point.
            bsdfSample.isBssrdf = true;
        }
        apply_restir_gi_production_diffuse(uniforms,
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
        apply_restir_pt_production_path_reuse(uniforms,
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
        capture_restir_pt_production_reservoir(uniforms,
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
        if (deferredRestirActivityDebug && depth >= 1u) {
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
                                      float(closureType),
                                      subsurfaceExitVertex ? 1.0f : 0.0f),
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
        if (bsdfSample.pdf <= 0.0f ||
            falcor_face_visibility(material, rec, bsdfSample.direction) == 0.0f) {
            break;
        }
        if (!bsdfSample.isDelta) {
            pathMaxRoughness = max(pathMaxRoughness, path_regularization_roughness(material));
        }
        caustic_transport_note_path_vertex(uniforms,
                                           material,
                                           rec.point,
                                           shadingNormal,
                                           rec.normal,
                                           bsdfSample.direction,
                                           throughput,
                                           depth,
                                           mediumDepth,
                                           bsdfSample,
                                           causticEyeVertices,
                                           causticLightVertices,
                                           causticConnections,
                                           causticConnectionResults,
                                           stats);
        float cacheConfidence = 0.0f;
        float cacheVariance = 0.0f;
        float cacheBiasRisk = 0.0f;
        // A training path never loads from the cache: its suffix is the unbiased sample the cache learns from.
        if (!cacheTraining.active &&
            radiance_cache_query_and_maybe_terminate(uniforms,
                                                     material,
                                                     rec.point,
                                                     shadingNormal,
                                                     specularOnly,
                                                     depth,
                                                     bsdfSample,
                                                     radianceCacheStates,
                                                     stats,
                                                     radiance,
                                                     throughput,
                                                     cacheConfidence,
                                                     cacheVariance,
                                                     cacheBiasRisk)) {
            if (outReconstructionConfidence) {
                *outReconstructionConfidence = max(*outReconstructionConfidence, cacheConfidence);
            }
            if (outReconstructionVariance) {
                *outReconstructionVariance = max(*outReconstructionVariance, cacheVariance);
            }
            if (outReconstructionBiasRisk) {
                *outReconstructionBiasRisk = max(*outReconstructionBiasRisk, cacheBiasRisk);
            }
            break;
        }
        radiance_cache_training_note_vertex(cacheTraining,
                                            uniforms,
                                            material,
                                            rec.point,
                                            shadingNormal,
                                            specularOnly,
                                            depth,
                                            bsdfSample,
                                            radiance,
                                            throughput);

        uint mediumDepthBefore = mediumDepth;
        float3 selectedSigma = float3(0.0f);
        if (interface_select_medium(material, rec, bsdfSample.direction, selectedSigma, mediumDepth)) {
            if (source_material_flag(material, kMaterialFlagExplicitInterior)) mediumSigmaStack[0] = selectedSigma;
        } else if (bsdfSample.mediumEvent == 1) {
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
        volume_apply_medium_state(ray, uniforms, mediumDepthAfter, volumeScatteringDepth);

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

        uint nextSpecularDepth = bsdfSample.isDelta ? (specularDepth + 1u) : 0u;
        bool didTransmission = false;
        if (bsdfSample.isDelta && material_closure_is_transmissive(material)) {
            float3 dir = bsdfSample.direction;
            if (all(isfinite(dir)) && dot(dir, dir) > 0.0f) {
                float side = (rec.frontFace != 0u) ? 1.0f : -1.0f;
                didTransmission = (dot(shadingNormal, dir) * side) < 0.0f;
            }
        }
        specularDepth = nextSpecularDepth;

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
            nextOrigin = interface_precision(material)
                ? offset_interface_origin(rec, bsdfSample.direction)
                : offset_ray_origin(rec, bsdfSample.direction);
        }
        interfaceMinT = spawn_ray_min(material, kEpsilon);
        mediumSegmentOffset = interface_precision(material)
            ? dot(nextOrigin - rec.point, bsdfSample.direction) : 0.0f;

        bool useMnee = (ENABLE_MNEE_CAUSTICS != 0) && (PT_DEFAULT_OFF(uniforms.enableMnee) != 0u);
        bool specNeeEnabled = (uniforms.enableSpecularNee != 0u);
        float dirLenSq = dot(bsdfSample.direction, bsdfSample.direction);
        bool specDirectionValid = (dirLenSq > 0.0f) && all(isfinite(bsdfSample.direction));
        bool mneeEligible = false;
#if ENABLE_MNEE_CAUSTICS
        mneeEligible = useMnee &&
                       bsdfSample.isDelta &&
                       ((bsdfSample.mediumEvent <= 0) || didTransmission) &&
                       material_closure_is_transmissive(material) &&
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
                                                             spawn_ray_min(material, kEpsilon),
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
            bool occluded = trace_scene_software_alpha_aware_shadow<false>(uniforms,
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
                                                                           spawn_ray_min(material, kEpsilon),
                                                                           kInfinity,
                                                                           materials,
                                                                           materialTextures,
                                                                           materialSamplers,
                                                                           materialTextureInfos,
                                                                           meshInfos,
                                                                           sceneVertices,
                                                                           meshIndices,
                                                                           shadowRec);
            if (!occluded) {
                float envPdf = environment_pdf(uniforms, environmentPdf, neeRay.direction);
                envPdf = max(envPdf, kSpecularNeePdfFloor);
                float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                float denom = envPdf + bsdfPdf;
                float misWeight = (denom > 0.0f) ? (envPdf / denom) : 0.0f;
                misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                float3 envColor = environment_color(environmentTexture,
                                                    neeRay.direction,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity,
                                                    uniforms);
                // The delta sample fixed this direction; it was not drawn from the
                // environment, so there is no 1/envPdf. This connection takes the
                // light strategy's MIS share and the continuation's environment miss
                // takes bsdfPdf / (bsdfPdf + envPdf), so the two sum to one.
                float3 neeContribution = bsdfSample.weight * envColor * misWeight;
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
                                                  /*invPdf=*/1.0f),
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
            bool hitLight = trace_scene_software_alpha_aware_shadow<false>(uniforms,
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
                                                                           spawn_ray_min(material, kEpsilon),
                                                                           kInfinity,
                                                                           materials,
                                                                           materialTextures,
                                                                           materialSamplers,
                                                                           materialTextureInfos,
                                                                           meshInfos,
                                                                           sceneVertices,
                                                                           meshIndices,
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
                    float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                    float denom = lightPdf + bsdfPdf;
                    float misWeight = (denom > 0.0f) ? (lightPdf / denom) : 0.0f;
                    misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                    // Delta direction, not a light sample: MIS share only, no 1/lightPdf
                    // (the continuation's emitter hit takes the complement).
                    float3 contribution = bsdfSample.weight * mneeHit.emission * misWeight;
                    // Delta-direction connection to a rect light: medium transmittance over the connection segment
                    // (see volume_surface_nee_transmittance).
                    contribution *= volume_surface_nee_transmittance(uniforms, distance(lightRec.point, nextOrigin), false, stats);
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
                                                      /*invPdf=*/1.0f),
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
            bool occluded = trace_scene_software_alpha_aware_shadow<false>(uniforms,
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
                                                                           spawn_ray_min(material, kEpsilon),
                                                                           kInfinity,
                                                                           materials,
                                                                           materialTextures,
                                                                           materialSamplers,
                                                                           materialTextureInfos,
                                                                           meshInfos,
                                                                           sceneVertices,
                                                                           meshIndices,
                                                                           shadowRec);
            if (!occluded) {
                float envPdf = environment_pdf(uniforms, environmentPdf, mneeRay.direction);
                envPdf = max(envPdf, kSpecularNeePdfFloor);
                float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                float denom = envPdf + bsdfPdf;
                float misWeight = (denom > 0.0f) ? (envPdf / denom) : 0.0f;
                misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                float3 envColor = environment_color(environmentTexture,
                                                    mneeRay.direction,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity,
                                                    uniforms);
                // Delta direction, not an environment sample: MIS share only, no
                // 1/envPdf (the continuation's environment miss takes the complement).
                float3 neeContribution = bsdfSample.weight * envColor * misWeight;
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
            bool hitLight = trace_scene_software_alpha_aware_shadow<false>(uniforms,
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
                                                                           spawn_ray_min(material, kEpsilon),
                                                                           kInfinity,
                                                                           materials,
                                                                           materialTextures,
                                                                           materialSamplers,
                                                                           materialTextureInfos,
                                                                           meshInfos,
                                                                           sceneVertices,
                                                                           meshIndices,
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
                    float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                    float denom = lightPdf + bsdfPdf;
                    float misWeight = (denom > 0.0f) ? (lightPdf / denom) : 0.0f;
                    misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                    // Delta direction, not a light sample: MIS share only, no 1/lightPdf
                    // (the continuation's emitter hit takes the complement).
                    float3 contribution = bsdfSample.weight * mneeHit.emission * misWeight;
                    contribution *= volume_surface_nee_transmittance(uniforms, distance(lightRec.point, nextOrigin), false, stats);
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

        rayCone.width = ray_cone_width_at_distance(rayCone, hitDistanceWorld);
        rayCone.spread = min(rayCone.spread +
                             bsdf_cone_spread_increment(bsdfSample.lobeType,
                                                        bsdfSample.lobeRoughness,
                                                        bsdfSample.isDelta),
                             1.5f);

        lastBsdfPdf = (bsdfSample.directionalPdf > 0.0f) ? bsdfSample.directionalPdf : bsdfSample.pdf;
        lastScatterWasVolume = false;
        // This vertex's fused NEE sampled the mesh emitters (baseline draw or a
        // reservoir) and could have produced the scattered direction.
        previousMeshNeeEligible = !explicitRestirDiLightingStage &&
            mesh_emitter_nee_mode(uniforms) &&
            uniforms.emissivePrimitiveCount > 0u && !surfaceIsDelta &&
            (!bsdfSample.isBssrdf || subsurfaceExitVertex) &&
            mesh_emitter_nee_covers_direction(uniforms, material, rec, shadingNormal, bsdfSample.direction);
        previousMeshNeeOrigin = rec.point;
        lastScatterWasDelta = bsdfSample.isDelta;
        // Without a specular connection along this delta direction, the next
        // emitter hit or environment miss is the only estimate for the path.
        lastDeltaUnpaired = bsdfSample.isDelta && !(specNeeEligible || mneeEligible);
        if (pavilionSmoothDifferentials) {
            pavilionSmoothDifferentials = propagate_pavilion_planar_delta_differentials(
                primaryRayDiff, ray, rec, material, bsdfSample);
        }
        ray.origin = nextOrigin;
        holeMisOriginValid = false;
        ray.direction = bsdfSample.direction;
        primaryVisibilityRay = false;

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
    if (spectralState.enabled) {
        // The path carried per-wavelength radiance; apply the sensor and white balance once.
        radiance = spectral_channels_to_rgb(radiance, spectralState, uniforms);
        if (outDirectRadiance) {
            *outDirectRadiance = spectral_channels_to_rgb(*outDirectRadiance, spectralState, uniforms);
        }
        if (outIndirectRadiance) {
            *outIndirectRadiance = spectral_channels_to_rgb(*outIndirectRadiance, spectralState, uniforms);
        }
    }
    radiance_cache_training_finish(cacheTraining, radianceCacheStates, uniforms, radiance, stats);
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
                                  PrimaryRayDiff primaryRayDiff,
                                  thread uint& state,
                                  device const BvhNode* nodes,
                                  device const uint* primitiveIndices,
                                  device PathtraceStats* stats,
                                  texture2d<float, access::sample> environmentTexture,
                                  constant array<texture2d<float, access::sample>, kMaxMaterialTextures>& materialTextures,
                                  constant array<sampler, kMaxMaterialSamplers>& materialSamplers,
                                  device const MaterialTextureInfo* materialTextureInfos,
                                  device const EnvironmentAliasEntry* environmentConditionalAlias,
                                  device const EnvironmentAliasEntry* environmentMarginalAlias,
                                  device const float* environmentPdf,
                                  device PathGuidingReservoirState* pathGuidingStates,
                                  device RestirPtReservoirState* restirPtReservoirs,
                                  device RadianceCacheState* radianceCacheStates,
                                  device EyeSubpathVertex* causticEyeVertices,
                                  device LightSubpathVertex* causticLightVertices,
                                  device PathConnection* causticConnections,
                                  device ConnectionResult* causticConnectionResults,
                                  uint2 pixelCoord,
                                  // Optional AOV outputs
                                  thread float3* outFirstHitAlbedo = nullptr,
                                  thread float3* outFirstHitNormal = nullptr,
                                  thread float4* outFirstHitPosition = nullptr,
                                  thread float4* outFirstHitMaterial = nullptr,
                                  thread float3* outDirectRadiance = nullptr,
                                  thread float3* outIndirectRadiance = nullptr,
                                  thread float* outReconstructionConfidence = nullptr,
                                  thread float* outReconstructionVariance = nullptr,
                                  thread float* outReconstructionBiasRisk = nullptr,
                                  thread PathtraceDebugContext* debugContext = nullptr) {
    float3 throughput = float3(1.0f, 1.0f, 1.0f);
    float3 radiance = float3(0.0f, 0.0f, 0.0f);
    RadianceCachePathTraining cacheTraining;
    radiance_cache_training_begin(cacheTraining,
                                  uniforms,
                                  radianceCacheStates,
                                  pixelCoord.y * uniforms.width + pixelCoord.x);
    float lastBsdfPdf = 1.0f;
    // The last scatter was in the medium (lastBsdfPdf is its phase pdf), and whether volume NEE sampled the rect lights
    // there: a phase-sampled emitter hit or escape is MIS-weighted only against a strategy that ran at that vertex.
    bool lastScatterWasVolume = false;
    bool lastVolumeRectNee = false;
    bool lastScatterWasDelta = true;
    bool lastDeltaUnpaired = false;
    bool previousMeshNeeEligible = false;
    float3 previousMeshNeeOrigin = float3(0.0f);
    // Where the last real scatter left from, kept across alpha holes (MIS null events): the rect-light pdf of the next
    // emitter hit is measured from there, as that vertex's NEE was.
    float3 holeMisOrigin = float3(0.0f);
    bool holeMisOriginValid = false;
    bool isFirstHit = true;
    bool isGuidePending = true;
    bool pavilionGuideAverage = false;
    bool primaryVisibilityRay = true;
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
    reconstruction_split_record( \
        clamp_firefly_contribution((throughput), (contribution), (params), stats), \
        depth, \
        outDirectRadiance, \
        outIndirectRadiance)
#define clamp_path_throughput(throughput, params) \
    clamp_path_throughput((throughput), (params), stats)
    constexpr uint kMaxMediumStack = 8u;
    float3 mediumSigmaStack[kMaxMediumStack];
    for (uint i = 0; i < kMaxMediumStack; ++i) {
        mediumSigmaStack[i] = float3(0.0f);
    }
    float interfaceMinT = kEpsilon;
    float mediumSegmentOffset = 0.0f;
    uint mediumDepth = 0u;
    uint volumeScatteringDepth = 0u;
    SpectralPathState spectralState = spectral_make_path_state(uniforms, state, stats);
    if (spectralState.enabled) {
        // Channel j of the throughput carries wavelength j; inactive channels start at zero.
        throughput = spectral_channel_mask(spectralState);
    }
    HitRecord prevRec;
    bool prevValid = false;
    uint specularDepth = 0u;
    float pathMaxRoughness = 0.0f;
    bool pavilionSmoothDifferentials = true;
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

    uint foliageAlphaVisibilitySkips = 0u;
    uint depthAdvance = 1u;
    for (uint depth = 0; depth < uniforms.maxDepth; depth += depthAdvance) {
        depthAdvance = 1u;
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
            PT_DEFAULT_OFF(uniforms.parityAssertEnabled) != 0u &&
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
                                                      interfaceMinT,
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
                                                                     interfaceMinT,
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
                                        interfaceMinT,
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
                                       interfaceMinT,
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
                                       interfaceMinT,
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
                                       interfaceMinT,
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
                                       interfaceMinT,
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
                                       interfaceMinT,
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
            if (PT_DEFAULT_OFF(uniforms.debugViewMode) != kDebugViewNone) {
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
            background = spectral_radiance_to_channels(background, spectralState);
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
                use_visible_emitter_mis(depth, lastScatterWasDelta, lastDeltaUnpaired, uniforms);
            // Scatter points sample no environment light, so an escape after one keeps weight 1.
            if (useSpecularMis && envSampling && !lastScatterWasVolume) {
                float lightPdf = environment_pdf(uniforms, environmentPdf, ray.direction);
                float denom = lastBsdfPdf + lightPdf;
                if (denom > 0.0f) {
                    misWeight = clamp(lastBsdfPdf / denom,
                                      kMisWeightClampMin,
                                      kMisWeightClampMax);
                }
            }
            float3 contribution = background * misWeight;
            if (depth == 0u && uniforms.backgroundMode == 2u &&
                environmentTexture.get_width() > 0u && environmentTexture.get_height() > 0u &&
                uniforms.environmentBackgroundScaleOffset != 0.0f) {
                contribution *= max(1.0f + uniforms.environmentBackgroundScaleOffset, 0.0f);
            }
            // Keep the physical HDR radiance for light endpoints and PDFs.
            caustic_transport_note_light_endpoint(uniforms,
                                                  ray.origin + safe_normalize(ray.direction),
                                                  -ray.direction,
                                                  ray.direction,
                                                  background,
                                                  envSampling ? environment_pdf(uniforms, environmentPdf, ray.direction) : 0.0f,
                                                  lastBsdfPdf,
                                                  depth,
                                                  mediumDepth,
                                                  causticLightVertices,
                                                  stats);
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
        const float segmentOffset = mediumSegmentOffset;
        mediumSegmentOffset = 0.0f;
        interfaceMinT = kEpsilon;
        if (volume_transport_enabled(uniforms)) {
            MediumDescriptor medium = volume_make_global_medium(uniforms);
            const float segment = max(rec.t, 0.0f);
            const float3 segmentEmission = volume_emission_integral(medium, segment);
            const float3 segmentTransmittance = volume_transmittance(medium, segment, stats);
            if (PT_DEFAULT_OFF(uniforms.volumeDebug) != 0u && outFirstHitMaterial) {
                *outFirstHitMaterial =
                    volume_debug_aov_features(segmentTransmittance,
                                              volumeScatteringDepth,
                                              volume_phase_id(uniforms),
                                              segmentEmission);
            }
            if (any(segmentEmission > float3(0.0f))) {
                radiance += clamp_firefly_contribution(throughput, segmentEmission, clampParams);
            }
            const uint maxVolumeEvents = max(medium.metadata.y, 1u);
            bool sampledVolumeEvent = false;
            bool sampledFreeFlight = false;
            float3 freeFlightWeight = float3(1.0f);
            if (volumeScatteringDepth < maxVolumeEvents &&
                any(medium.sigmaSAnisotropy.xyz > float3(0.0f))) {
                const VolumeFreeFlight flight = volume_sample_free_flight(medium, segment, state);
                const float sampledDistance = flight.distance;
                sampledFreeFlight = true;
                freeFlightWeight = flight.weight;
                sampledVolumeEvent = flight.scattered;
                if (sampledVolumeEvent) {
                    const float3 scatterPoint = ray.origin + ray.direction * sampledDistance;
                    const float3 transmittance = volume_transmittance(medium, sampledDistance, stats);
                    if (PT_DEFAULT_OFF(uniforms.volumeDebug) != 0u && outFirstHitMaterial) {
                        *outFirstHitMaterial =
                            volume_debug_aov_features(transmittance,
                                                      volumeScatteringDepth + 1u,
                                                      volume_phase_id(uniforms),
                                                      segmentEmission);
                    }
                    // Free-flight weight sigma_s T(t) / mean(sigma_t T(t)) (volume_sample_free_flight).
                    throughput *= freeFlightWeight;
                    if (rectLightCount > 0u && medium.metadata.z != 0u) {
                        HitRecord volumeRec = rec;
                        volumeRec.point = scatterPoint;
                        for (uint neeDraw = 0u; neeDraw < rect_light_nee_draw_count(uniforms); ++neeDraw) {
                            RectLightSample lightSample;
                            if (sample_rect_light_nee_draw(uniforms,
                                                  rectangles,
                                                  materials,
                                                  environmentTexture,
                                                  volumeRec,
                                                  state,
                                                  rectLightCount,
                                                  lightSample, kAnalyticLightFilterAll, neeDraw)) {
                                Ray shadowRay;
                                shadowRay.origin = scatterPoint + lightSample.direction * kRayOriginEpsilon;
                                shadowRay.direction = lightSample.direction;
                                shadowRay.mediumState =
                                    volume_make_global_medium_state(uniforms,
                                                                    mediumDepth,
                                                                    volumeScatteringDepth + 1u);
                                HitRecord shadowRec = make_empty_hit_record();
                                const float shadowMax = max(lightSample.distance - kEpsilon, kEpsilon);
                                // Hardware tracing, as this integrator's surface NEE uses: the software BVH normally holds no
                                // triangles when hardware ray tracing runs, so meshes did not occlude these rays. A scatter point
                                // is not on a surface, so nothing is excluded; alpha-tested cutouts are skipped as for surface NEE.
                                bool occluded = false;
#if PT_DEBUG_TOOLS
                                if (forceSoftware) {
                                    occluded = trace_scene_software_alpha_aware_shadow<false>(
                                        uniforms, spheres, rectangles, triangleData,
                                        tlasNodes, tlasPrimIndices, instanceInfos,
                                        blasNodes, blasPrimIndices, nodes, primitiveIndices,
                                        stats, shadowRay, kEpsilon, shadowMax, materials,
                                        materialTextures, materialSamplers, materialTextureInfos,
                                        meshInfos, sceneVertices, meshIndices, shadowRec);
                                } else {
#endif
                                    occluded = trace_scene_hardware_alpha_aware_shadow<false>(uniforms,
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
                                                                                              kInvalidIndex,
                                                                                              kInvalidIndex,
                                                                                              materials,
                                                                                              materialTextures,
                                                                                              materialSamplers,
                                                                                              materialTextureInfos,
                                                                                              shadowRec);
#if PT_DEBUG_TOOLS
                                }
#endif
                                if (!occluded && lightSample.pdf > 0.0f) {
                                    const float phase = volume_phase_eval(medium, -ray.direction, lightSample.direction);
                                    const float3 shadowTr = volume_transmittance(medium, shadowMax, stats);
                                    // MIS against the phase-sampled hit of the same light (volume_nee_mis_weight).
                                    const float misWeight = lightSample.isReferencePoint
                                        ? 1.0f : volume_nee_mis_weight(lightSample.pdf, phase);
                                    const float3 contribution = lightSample.emission * phase * shadowTr * misWeight /
                                                                max(lightSample.pdf, 1.0e-6f);
                                    radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                                    if (stats) {
                                        atomic_fetch_add_explicit(&stats->volumeNeeRayCount, 1u, memory_order_relaxed);
                                    }
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
                    volume_apply_medium_state(ray, uniforms, mediumDepth, volumeScatteringDepth);
                    lastScatterWasDelta = false;
                    lastDeltaUnpaired = false;
                    previousMeshNeeEligible = false;
                    pavilionSmoothDifferentials = false;
                    lastBsdfPdf = max(phasePdf, 1.0e-6f);
                    lastScatterWasVolume = true;
                    lastVolumeRectNee = rectLightCount > 0u && medium.metadata.z != 0u;
                    ray.origin = scatterPoint + phaseDir * kRayOriginEpsilon;
                    holeMisOriginValid = false;
                    ray.direction = phaseDir;
                    prevValid = false;
                    primaryVisibilityRay = false;
                    continue;
                }
            } else if (any(medium.sigmaSAnisotropy.xyz > float3(0.0f)) && stats) {
                atomic_fetch_add_explicit(&stats->volumeFallbackCount, 1u, memory_order_relaxed);
            }
            // Passed the segment: the free-flight weight T(s) / mean(T(s)) when a distance was sampled, else T(s).
            throughput *= sampledFreeFlight ? freeFlightWeight : segmentTransmittance;
        }
        if (mediumDepth > 0u) {
            float3 sigma = mediumSigmaStack[mediumDepth - 1u];
            if (any(sigma > float3(0.0f))) {
                float segment = max(rec.t + segmentOffset, 0.0f);
                float3 attenuation = exp(-sigma * segment);
                throughput *= attenuation;
            }
        }
        uint matIndex = min(rec.materialIndex, uniforms.materialCount - 1);
        MaterialData material = materials[matIndex];
        regularize_dielectric_material(material, rec.frontFace != 0u, pathMaxRoughness);
        uint type = static_cast<uint>(material.typeEta.x);
        MaterialClosureMetadata primaryClosure = material_primary_closure_metadata(material);
        uint closureType = primaryClosure.type;
        float3 incidentDir = normalize(ray.direction);
        float3 wo = -incidentDir;
        float hitDistanceWorld = ray_segment_world_length(ray, rec.t);
        bool surfaceIsDelta = material_closure_is_delta(material);
        bool explicitRestirDiLightingStage =
            uniforms.explicitRestirDiLightingStage != 0u &&
            (uniforms.directLightMode == kDirectLightModeRestirDi ||
             uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid);
        bool specularOnly = (uniforms.debugSpecularOnly != 0u);
        float diffuseOcclusion = 1.0f;
        float3 debugBaseColor = material_base_color(material);
        float2 debugBaseColorUv = float2(0.0f);
        float debugBaseColorLod = 0.0f;
        float debugMetallic = 0.0f;
        float debugRoughness = clamp(material.baseColorRoughness.w, 0.0f, 1.0f);
        float debugAO = 1.0f;
        float3 shadingNormal = rec.shadingNormal;
        bool authoredNormalWasFaced = false;
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
                if (dot(candidate, rec.normal) < 0.0f &&
                    !source_material_flag(material, kMaterialFlagAuthoredDielectricNormal)) {
                    authoredNormalWasFaced = true;
                    candidate = -candidate;
                }
                shadingNormal = normalize(candidate);
                debugVtxNormal = shadingNormal;
            }
        }
        if (type == 2u && !source_material_flag(material, kMaterialFlagAuthoredDielectricNormal)) {
            float3 geomNormal = rec.normal;
            if (all(isfinite(geomNormal)) && dot(geomNormal, geomNormal) > 0.0f) {
                shadingNormal = geomNormal;
            }
            // Keep ray offsets consistent between SWRT/HWRT for glass.
            rec.shadingNormal = shadingNormal;
        }


        if (rec.primitiveType == kPrimitiveTypeSphere &&
            source_material_flag(material, kMaterialFlagAuthoredDielectricNormal)) {
            // An analytic sphere's authored normal is its outward normal (triangles keep their raw interpolated
            // normal above); set_face_normal faced it, so source_dielectric_frame never saw an exit (EP8-f3).
            shadingNormal = rec.frontFace != 0u ? rec.normal : -rec.normal;
        }
        if (kSourceFeatures && material_texture_valid(uniforms,material.textureIndices1.w)) {
            shadingNormal = source_bump_normal(uniforms,material,rec,shadingNormal,
                                              materialTextures,meshInfos,sceneVertices,meshIndices);
            rec.shadingNormal = shadingNormal;
        }
        source_mark_closure_side(material, shadingNormal, wo);

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
            baseColorSample = resolve_alpha_cutout_foliage_base_color_sample(material,
                                                                             baseColorSample);
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
                                                                 uvPerWorld0,
                                                                 (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u);
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
            if ((depth == 0u || (pavilionSmoothDifferentials &&
                 (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u)) &&
                hasSurfacePartials0) {
                float3 dudP = float3(0.0f);
                float3 dvdP = float3(0.0f);
                if (uv_world_gradients_from_partials(dPdu0, dPdv0, dudP, dvdP)) {
                    hasIgehyGradients0 = first_hit_uv_gradients_igehy(ray,
                                                                      primaryRayDiff,
                                                                      rec.t,
                                                                      ((material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u
                                                                      ? shadingNormal : rec.normal),
                                                                      dudP,
                                                                      dvdP,
                                                                      dUVdx0,
                                                                      dUVdy0);
                }
            }
            float2 dUVdx1 = float2(0.0f);
            float2 dUVdy1 = float2(0.0f);
            bool hasIgehyGradients1 = false;
            if ((depth == 0u || (pavilionSmoothDifferentials &&
                 (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u)) &&
                hasSurfacePartials1) {
                float3 dudP = float3(0.0f);
                float3 dvdP = float3(0.0f);
                if (uv_world_gradients_from_partials(dPdu1, dPdv1, dudP, dvdP)) {
                    hasIgehyGradients1 = first_hit_uv_gradients_igehy(ray,
                                                                      primaryRayDiff,
                                                                      rec.t,
                                                                      ((material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u
                                                                      ? shadingNormal : rec.normal),
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
            baseColorSample = resolve_alpha_cutout_foliage_base_color_sample(material,
                                                                             baseColorSample);
            float3 baseColorSampleRgb = to_working_space(baseColorSample.xyz, uniforms);
            float3 baseColor = baseFactor * baseColorSampleRgb;
            debugBaseColorUv = baseColorCtx.uv;
            debugBaseColorLod = baseColorLod;

            float metallic = clamp(material.pbrParams.x, 0.0f, 1.0f);
            // Preserve scene multipliers above one through source-G multiplication.
            float roughness = max(material.pbrParams.y, 0.0f);
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
            roughness = clamp(roughness, 0.0f, 1.0f);
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
            alpha = clamp(alpha * path_cutout_alpha(materialTextures, materialSamplers, materialTextureInfos,
                                                    uniforms, material, baseColorCtx, baseColorLod,
                                                    baseColorSample.w),
                          0.0f, 1.0f);
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
                    const bool nonCountingPavilionAlphaSkip =
                        (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u &&
                        alphaMode < 1.5f && foliageAlphaVisibilitySkips < 64u;
                    const bool nonCountingFoliageAlphaSkip =
                        (material.materialFlags & kMaterialFlagAlphaCutoutContinuation) != 0u &&
                        foliageAlphaVisibilitySkips < 64u;
                    if (!holeMisOriginValid) {
                        holeMisOrigin = ray.origin;
                        holeMisOriginValid = true;
                    }
                    if (nonCountingPavilionAlphaSkip || nonCountingFoliageAlphaSkip) {
                        advance_pavilion_alpha_visibility_ray(ray, primaryRayDiff, rayCone, rec);
                    } else {
                        ray.origin = offset_ray_origin(rec, ray.direction);
                    }
                    prevRec = rec;
                    prevValid = true;
                    if (nonCountingPavilionAlphaSkip || nonCountingFoliageAlphaSkip) {
                        foliageAlphaVisibilitySkips += 1u;
                        depthAdvance = 0u;
                    } else {
                        // A counted MASK hole or BLEND pass is a null event for MIS: the last real vertex's BSDF pdf,
                        // delta flags and origin stay, so the next emitter hit or miss takes the BSDF side of the
                        // balance heuristic whose light side that vertex's NEE took (its alpha-aware shadow rays, the
                        // specular-NEE ones included, pass MASK holes, and BLEND texels with the path's probability
                        // 1 - alpha). A hole before any scatter is unpaired: no NEE samples what lies behind it.
                        if (depth == 0u) {
                            lastScatterWasDelta = true;
                            lastDeltaUnpaired = true;
                        }
                        specularDepth += 1u;
                    }
                    continue;
                }
            }

            if ((material.materialFlags & kMaterialFlagBlackBackFace) != 0u && rec.frontFace == 0u) {
                break;  // Single-sided back face: the reference shades it black.
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
            float normalDispersion = 0.0f;
            float3 normalSampleTs = float3(0.0f, 0.0f, 1.0f);
            bool flipNormalGreen = false;
#if PT_DEBUG_TOOLS
            flipNormalGreen = uniforms.debugFlipNormalGreen != 0u;
#endif
            if (useNormalMap) {
                float4 normalTexel =
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
                                                    normalCtx.dUVdy);
                debugTexelRaw = normalTexel.xyz;
                normalDispersion = normalTexel.w;
                normalSampleTs = decode_normal_map(debugTexelRaw,
                                                   normalScale,
                                                   flipNormalGreen);
                debugTexelDecoded = normalSampleTs;
                debugSnTangentSpace = normalSampleTs;
                float3 t = tangent.xyz;
                float3 b = float3(0.0f);
                bool hasBasis = false;
                bool hasVertexTangentBasis = false;
                bool trustVertexTangent = fabs(tangent.w) > 0.5f;
                if (trustVertexTangent && all(isfinite(t)) && dot(t, t) > 1.0e-6f) {
                    t = normalize(t - shadingNormal * dot(shadingNormal, t));
                    if (all(isfinite(t)) && dot(t, t) > 1.0e-6f) {
                        float tangentSign = (tangent.w < 0.0f) ? -1.0f : 1.0f;
                        b = normalize(cross(shadingNormal, t)) * tangentSign;
                        if (all(isfinite(b)) && dot(b, b) > 1.0e-6f) {
                            hasBasis = true;
                            hasVertexTangentBasis = true;
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
                float3 mappedNormalTs = normalSampleTs;
                if (hasVertexTangentBasis &&
                    (material.materialFlags & (kMaterialFlagAlphaCutoutContinuation |
                                               kMaterialFlagSourceNormalGreen)) != 0u &&
                    triangle_has_valid_vertex_tangents(uniforms, rec.meshIndex, rec.primitiveIndex,
                                                       meshInfos, sceneVertices, meshIndices)) {
                    // Per-material finding: these Emerald materials match the Falcor
                    // references only with G negated (other ORCA normal maps match with
                    // G kept). Then apply Falcor's complete-normal flip when N faced.
                    // Keep the original decoded sample for normal-variance roughness.
                    mappedNormalTs.y = -mappedNormalTs.y;
                    if (authoredNormalWasFaced) {
                        mappedNormalTs.x = -mappedNormalTs.x;
                    }
                }
                float3 mapped = normalize(t * mappedNormalTs.x +
                                          b * mappedNormalTs.y +
                                          shadingNormal * mappedNormalTs.z);
                if (dot(mapped, rec.normal) < 0.0f) {
                    mapped = -mapped;
                }
                if ((material.materialFlags & kMaterialFlagSourceNormalGreen) != 0u) {
                    // Falcor 4.4 adjustShadingNormal(): avoid back-facing view vectors
                    // at normal-map bevels. This flag is Picnic_Table1-only. Same rule
                    // as the wavefront (material_access.metal).
                    float nDotV = dot(mapped, wo);
                    if (nDotV <= 0.1f) {
                        mapped = normalize(mix(rec.normal, mapped, saturate(nDotV * 10.0f)));
                    }
                }
                debugTangent = t;
                debugBitangent = b;
                shadingNormal = mapped;
                debugSnWorld = shadingNormal;
            }

            if (useNormalMap) {
                float directionalVariance = normalDispersion * normalDispersion;
                float meanLength = sqrt(1.0f - directionalVariance);
                float tok = directionalVariance / (meanLength * (1.0f + meanLength));
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
                        nDx = decode_normal_map(nDx, normalScale, flipNormalGreen);
                        nDy = decode_normal_map(nDy, normalScale, flipNormalGreen);
                        float varianceX = 0.5f * dot(normalSampleTs - nDx, normalSampleTs - nDx);
                        float varianceY = 0.5f * dot(normalSampleTs - nDy, normalSampleTs - nDy);
                        float normalVariance = max(varianceX, varianceY);
                        tok += 0.35f * normalVariance;
                    }
                }
                // Falcor's StandardMaterial shades the authored roughness unfiltered.
                if (tok > 0.0f && uniforms.disableNormalVarianceRoughness == 0u &&
                    (material.materialFlags & kMaterialFlagFalcorStandardBsdf) == 0u) {
                    roughness = clamp(sqrt(roughness * roughness + tok), 0.0f, 1.0f);
                }
            }

            if ((material.materialFlags & kMaterialFlagFalcorStandardBsdf) != 0u) {
                roughness = max(roughness, 0.08f);  // Falcor floors GGX alpha at 0.0064; never delta.
            }
            material.baseColorRoughness = float4(baseColor, roughness);
            material.pbrParams.x = metallic;
            material.emission = float4(emissive, 0.0f);
            material = apply_foliage_beauty_lift(material);
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

        regularize_pavilion_pbr_material(material, pathMaxRoughness, rec.frontFace != 0u);
        // Denoiser guides stay RGB: capture them before the material becomes per-wavelength.
        const float3 spectralGuideAlbedo =
            spectralState.enabled ? material_closure_aov_albedo(material) : float3(0.0f);
        const float4 spectralGuideFeatures =
            spectralState.enabled ? material_closure_aov_features(material) : float4(0.0f);
        spectral_apply_material(material, uniforms, spectralState, stats);
        spectral_collapse_at_dispersive_vertex(spectralState, material, uniforms, throughput, stats);
        if ((material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u) {
            surfaceIsDelta = material_closure_is_delta(material);
        }
        primaryClosure = material_primary_closure_metadata(material);
        closureType = primaryClosure.type;
        debugBaseColor = material_base_color(material);

        const bool deferredRestirActivityDebug =
            PT_DEFAULT_OFF(uniforms.debugViewMode) == kDebugViewRestirPathGuidingMask ||
            PT_DEFAULT_OFF(uniforms.debugViewMode) == kDebugViewRestirPtReuseMask;
        if (PT_DEFAULT_OFF(uniforms.debugViewMode) != kDebugViewNone &&
            !deferredRestirActivityDebug) {
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

        const bool primaryVisibilityHit = primaryVisibilityRay &&
                                          volumeScatteringDepth == 0u;

        // Keep geometric features at the primary surface. OIDN can use the
        // underlying non-delta surface through Pavilion water and glass.
        if (isFirstHit) {
            isFirstHit = false;
            pavilionGuideAverage = uniforms.oidnDeltaGuidesAllowed != 0u &&
                (material.materialFlags & kMaterialFlagSingleScatterSpecular) != 0u;
            if (outFirstHitPosition != nullptr) {
                *outFirstHitPosition = float4(rec.point, 1.0f);
            }
            if (outFirstHitMaterial != nullptr) {
                *outFirstHitMaterial = spectralState.enabled ? spectralGuideFeatures
                                                             : material_closure_aov_features(material);
            }
        }
        if (isGuidePending && (!pavilionGuideAverage || !surfaceIsDelta)) {
            isGuidePending = false;
            if (outFirstHitAlbedo != nullptr) {
                *outFirstHitAlbedo = spectralState.enabled ? spectralGuideAlbedo
                                                           : material_closure_aov_albedo(material);
            }
            if (outFirstHitNormal != nullptr) {
                *outFirstHitNormal = shadingNormal;
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
        uint risAuditRngBefore = 0u;
        uint risAuditRngAfter = 0u;
        uint risAuditTemporalRngBefore = 0u;
        uint risAuditTemporalRngAfter = 0u;
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
            primaryVisibilityHit &&
            (material.materialFlags & kMaterialFlagFoliageBeautyLift) != 0u &&
            (rec.frontFace != 0u || rec.twoSided != 0u)) {
            float3 foliageVisible = clamp(material.baseColorRoughness.xyz * 1.15f,
                                          float3(0.0f),
                                          float3(1.0f));
            radiance += clamp_firefly_contribution(throughput, foliageVisible, clampParams);
            break;
        }

        if (!specularOnly &&
            material_closure_is_emissive(material) &&
            !material_closure_primary_is_emissive(material) &&
            (rec.frontFace != 0u || rec.twoSided != 0u)) {
            float3 visibleEmission = camera_visible_emission(material, depth);
            if (depth > 0u && !lastScatterWasDelta && previousMeshNeeEligible) {
                visibleEmission *= mesh_emitter_bsdf_hit_weight(uniforms, emissivePrimitives, rec,
                                                                previousMeshNeeOrigin, lastBsdfPdf);
            }
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

        if (material_closure_primary_is_emissive(material)) {
            if (specularOnly) {
                break;
            }
            float3 emission = material_closure_aov_emission(material);
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
                emission *= spectral_radiance_to_channels(envColor, spectralState);
            }
            if (any(emission != float3(0.0f)) &&
                (rec.frontFace != 0u || rec.twoSided != 0u)) {
                float misWeight = 1.0f;
                bool useSpecularMis =
                    use_visible_emitter_mis(depth, lastScatterWasDelta, lastDeltaUnpaired, uniforms);
                if (useSpecularMis && rectLightCount > 0u && (!lastScatterWasVolume || lastVolumeRectNee)) {
                    float lightPdf = rect_light_pdf_for_hit(uniforms,
                                                            rectangles,
                                                            materials,
                                                            rectLightCount,
                                                            rec,
                                                            holeMisOriginValid ? holeMisOrigin : ray.origin);
                    float denom = lastBsdfPdf + lightPdf;
                    if (denom > 0.0f) {
                        misWeight = clamp(lastBsdfPdf / denom,
                                          kMisWeightClampMin,
                                          kMisWeightClampMax);
                    }
                }
                // Mesh emitters pair with the mesh-emitter NEE, not the rect lights.
                if (rec.primitiveType == kPrimitiveTypeTriangle &&
                    depth > 0u && !lastScatterWasDelta && previousMeshNeeEligible) {
                    misWeight = mesh_emitter_bsdf_hit_weight(uniforms, emissivePrimitives, rec,
                                                             previousMeshNeeOrigin, lastBsdfPdf);
                }
                float3 contribution = emission * misWeight;
                caustic_transport_note_light_endpoint(uniforms,
                                                      rec.point,
                                                      shadingNormal,
                                                      ray.direction,
                                                      emission,
                                                      rectLightCount > 0u
                                                          ? rect_light_pdf_for_hit(uniforms,
                                                                                   rectangles,
                                                                                   materials,
                                                                                   rectLightCount,
                                                                                   rec,
                                                                                   ray.origin)
                                                          : 0.0f,
                                                      lastBsdfPdf,
                                                      depth,
                                                      mediumDepth,
                                                      causticLightVertices,
                                                      stats);
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

        // Subsurface (see subsurface_vertex_begin): a random-walk hit moves the shading vertex to the point where
        // the walk leaves the mesh; any other subsurface hit is shaded as its dense limit.
        bool subsurfaceExitVertex = false;
        if (material_is_subsurface(material)) {
            if (subsurface_vertex_begin(uniforms, material, rec, shadingNormal, wo, specularOnly, state, throughput) ==
                kSubsurfaceVertexWalk) {
                HitRecord exitRec;
                float3 walkWeight = float3(0.0f);
                uint walkEvent;
#if PT_DEBUG_TOOLS
                if (forceSoftware) {
                    walkEvent = sss_random_walk_software(uniforms, material, rec, spheres, rectangles, triangleData,
                                                         tlasNodes, tlasPrimIndices, instanceInfos, blasNodes,
                                                         blasPrimIndices, nodes, primitiveIndices,
                                                         meshInfos, sceneVertices, meshIndices, stats, state,
                                                         exitRec, walkWeight);
                } else
#endif
                {
                    walkEvent = sss_random_walk_hardware(uniforms, material, rec, accel, meshInfos, triangleData,
                                                         sceneVertices, meshIndices, instanceUserIds, spheres,
                                                         rectangles, nodes, primitiveIndices, stats, state,
                                                         exitRec, walkWeight);
                }
                if (walkEvent != kSssWalkExit) {
                    break;
                }
                throughput *= walkWeight;
                subsurface_vertex_exit(material, rec, exitRec);
                shadingNormal = rec.shadingNormal;
                wo = shadingNormal;
                incidentDir = -wo;
                diffuseOcclusion = 1.0f;
                subsurfaceExitVertex = true;
            }
            if (!material_is_subsurface(material)) {
                // The explicit ReSTIR DI passes skip every subsurface surface (its evaluation is marked BSSRDF), so a
                // vertex shaded as a walk's exit point or as the dense limit takes its direct light here instead.
                explicitRestirDiLightingStage = false;
            }
            primaryClosure = material_primary_closure_metadata(material);
            closureType = primaryClosure.type;
            surfaceIsDelta = material_closure_is_delta(material);
        }

        const bool restirDiProviderDirectLighting =
            (uniforms.directLightMode == kDirectLightModeRestirDi ||
             uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
            restir_di_make_provider_set(uniforms, rectangles, materials).providerCount > 0u;
        if (!explicitRestirDiLightingStage &&
            !surfaceIsDelta &&
            uniforms.directLightMode != kDirectLightModeLegacyRect &&
            (uniforms.emissivePrimitiveCount > 0u || restirDiProviderDirectLighting)) {
            if (uniforms.directLightMode == kDirectLightModeBaselineEmissive) {
                const uint lightSampleCount = baseline_direct_light_sample_count(uniforms, emissivePrimitives);
                // Keep one copy of the visibility/texture path in the megakernel.
#pragma clang loop unroll(disable)
                for (uint lightSampleIndex = 0u; lightSampleIndex < lightSampleCount; ++lightSampleIndex) {
                    RISSamplePayload lightSample;
                    if (sample_direct_light_baseline(uniforms,
                                                     emissivePrimitives,
                                                     rec,
                                                     state,
                                                     lightSample) &&
                        resolve_textured_direct_light_emission(uniforms, lightSample, materials,
                            materialTextures, materialSamplers, materialTextureInfos,
                            meshInfos, sceneVertices, meshIndices)) {
                        lightSample.emission = spectral_radiance_to_channels(lightSample.emission, spectralState);
                        const bool lightSampleDirectional = ris_payload_is_directional(lightSample);
                        float distSq = ris_payload_distance_sq(rec, lightSample);
                        float distance = lightSampleDirectional ? 1.0e20f : length(lightSample.position - rec.point);
                        float3 omegaL = ris_payload_omega_l(rec, lightSample);
                        float cosThetaSurface =
                            two_sided_alpha_cutout_direct_cos(uniforms, material, rec, shadingNormal, omegaL);
                        float cosThetaLight = lightSampleDirectional ? 1.0f : saturate(emissive_light_cosine(lightSample.flags, lightSample.normal, -omegaL));
                        if (lightSample.pdf > 0.0f &&
                            distSq > 0.0f &&
                            cosThetaSurface > 0.0f &&
                            cosThetaLight > 0.0f) {
                            Ray shadowRay;
                            shadowRay.origin = offset_spawn_origin(rec, omegaL, material);
                            shadowRay.direction = omegaL;
                            HitRecord shadowRec = make_empty_hit_record();
                            float shadowMax = max(distance - kEpsilon, kEpsilon);
                            if (!lightSampleDirectional && (lightSample.flags & (kLightPrimitiveFlagTwoSided | kLightPrimitiveFlagTexturedEmission)) != 0u) {
                                const float3 visibilityDelta = lightSample.position - shadowRay.origin;
                                const float visibilityDistance = length(visibilityDelta);
                                shadowRay.direction = visibilityDelta / visibilityDistance;
                                shadowMax = max(visibilityDistance - kEpsilon, kEpsilon);
                            }
                            uint shadowExcludeMesh;
                            uint shadowExcludePrim;
                            compute_exclusion_indices(rec, shadowExcludeMesh, shadowExcludePrim);
                            bool occluded = false;
#if PT_DEBUG_TOOLS
                            if (forceSoftware) {
                                occluded = trace_scene_software_alpha_aware_shadow<false>(
                                    uniforms, spheres, rectangles, triangleData,
                                    tlasNodes, tlasPrimIndices, instanceInfos,
                                    blasNodes, blasPrimIndices, nodes, primitiveIndices,
                                    stats, shadowRay, spawn_ray_min(material, kEpsilon), shadowMax, materials,
                                    materialTextures, materialSamplers, materialTextureInfos,
                                    meshInfos, sceneVertices, meshIndices, shadowRec,
                                    pavilion_punctual_shadow_enabled(material, lightSampleDirectional), depth);
                            } else {
#endif
                                occluded = trace_scene_hardware_alpha_aware_shadow<false>(uniforms,
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
                                                                                          spawn_ray_min(material, kEpsilon),
                                                                                          shadowMax,
                                                                                          shadowExcludeMesh,
                                                                                          shadowExcludePrim,
                                                                                          materials,
                                                                                          materialTextures,
                                                                                          materialSamplers,
                                                                                          materialTextureInfos,
                                                                                          shadowRec,
                                                                                          pavilion_punctual_shadow_enabled(material, lightSampleDirectional), depth);
                                if (!lightSampleDirectional &&
                                    occluded &&
                                    shadowRec.primitiveType == kPrimitiveTypeTriangle &&
                                    shadowRec.primitiveIndex == lightSample.primitiveIndex) {
                                    occluded = false;
                                }
#if PT_DEBUG_TOOLS
                            }
#endif
                            if (!occluded) {
                                BsdfEvalResult bsdfEval = material_closure_eval(material,
                                                                        rec.point,
                                                                        shadingNormal,
                                                                        wo,
                                                                        omegaL,
                                                                        clampParams,
                                                                        PT_DEFAULT_OFF(uniforms.sssMode),
                                                                        diffuseOcclusion,
                                                                        specularOnly,
                                                                        lightSampleDirectional);
                                const bool cutoutBackLightUnsampled = apply_two_sided_alpha_cutout_diffuse_fallback(uniforms,
                                                                              material,
                                                                              rec,
                                                                              shadingNormal,
                                                                              omegaL,
                                                                              diffuseOcclusion,
                                                                              bsdfEval);
                                if (!bsdfEval.isDelta && !bsdfEval.isBssrdf) {
                                    float3 value = lightSampleDirectional ? bsdfEval.value
                                        : reference_continuation_nee_value(material, shadingNormal, wo, omegaL, bsdfEval.value);
                                    // Every mesh emitter is also reachable by the BSDF scatter.
                                    const float weight = lightSampleDirectional ? 1.0f
                                        : mesh_emitter_nee_mis_weight(uniforms, depth, lightSample.pdf, nee_mis_bsdf_pdf(bsdfEval, cutoutBackLightUnsampled));
                                    float3 contribution = value * lightSample.emission;
                                    contribution *= weight * cosThetaSurface / max(lightSample.pdf, 1.0e-6f);
                                    // Medium transmittance over the shadow segment (see volume_surface_nee_transmittance).
                                    contribution *= volume_surface_nee_transmittance(
                                        uniforms, sqrt(distSq), lightSampleDirectional, stats);
                                    if (all(isfinite(contribution))) {
                                        // Average before recording the reconstruction split, while
                                        // keeping the existing clamp on each independent draw.
                                        const float3 averaged = (clamp_firefly_contribution)(
                                            throughput, contribution, clampParams, stats) / float(lightSampleCount);
                                        radiance += reconstruction_split_record(averaged, depth,
                                            outDirectRadiance, outIndirectRadiance);
                                    }
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
                risAuditRngBefore = state;
                risAuditRngAfter = state;
                risAuditTemporalRngBefore = 0u;
                risAuditTemporalRngAfter = 0u;
#if PT_DEBUG_TOOLS
                record_ris_reservoir_candidate_audit(debugContext,
                                                     uniforms,
                                                     emissivePrimitives,
                                                     rec,
                                                     material,
                                                     shadingNormal,
                                                     wo,
                                                     clampParams,
                                                     diffuseOcclusion,
                                                     specularOnly,
                                                     risTargetM,
                                                     risAuditRngBefore,
                                                     depth,
                                                     mediumDepth,
                                                     rec.materialIndex);
#endif
                if (uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) {
                    build_restir_di_provider_reservoir_for_hit(uniforms,
                                                               emissivePrimitives,
                                                               rectangles,
                                                               materials,
                                                               environmentTexture,
                                                               environmentConditionalAlias,
                                                               environmentMarginalAlias,
                                                               environmentPdf,
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
                    if (stats) {
                        atomic_fetch_add_explicit(&stats->restirDiRegirCandidateCount,
                                                  1u,
                                                  memory_order_relaxed);
                        if (reservoir.valid) {
                            atomic_fetch_add_explicit(&stats->restirDiRegirAcceptedCount,
                                                      1u,
                                                      memory_order_relaxed);
                        } else {
                            atomic_fetch_add_explicit(&stats->restirDiRegirFallbackCount,
                                                      1u,
                                                      memory_order_relaxed);
                        }
                    }
                } else {
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
                }
                risAuditRngAfter = state;

                // Runs when the pixel's own reservoir is empty too: the neighbours' MIS weights assume the
                // canonical technique always takes part, so skipping them there would lose their share.
                // A default-features gate: the fast hardware pipeline folds this block out.
                if (PT_DEFAULT_OFF(uniforms.directLightMode == kDirectLightModeRisSpatialReuse ? 1u : 0u) != 0u &&
                    (depth == 0u && !subsurfaceExitVertex) &&
                    rec.t > 0.0f &&
                    uniforms.width > 0u &&
                    uniforms.height > 0u) {
                    spatialReuseAttempted = true;
                    spatialNeighborTarget = min(max(uniforms.spatialReuseNeighborCount, 1u), 4u);
                    // Generalized RIS with pairwise MIS (reservoir.metal): neighbour winners stream into
                    // `merged`, then the canonical winner joins with its share of every pair. The old merge
                    // weighted each neighbour by p_c / (p_c M + p_n M), kept the canonical unweighted and added
                    // 1 to M per neighbour; on one light that converges to (M + 2) / (M + 4) of the exact value.
                    // Targets are compared per unit light area (target times cos(theta_light) / d^2), which carries
                    // the solid-angle Jacobian of moving a triangle-light sample between pixels.
                    const float pairCount = float(spatialNeighborTarget);
                    const bool canonicalValid = reservoir.valid;
                    const RISSamplePayload canonicalWinner = reservoir.winner;
                    const float canonicalAreaTarget = canonicalValid
                        ? reservoir_winner_phat_for_hit(uniforms,
                                                        rec,
                                                        material,
                                                        shadingNormal,
                                                        wo,
                                                        clampParams,
                                                        diffuseOcclusion,
                                                        specularOnly,
                                                        canonicalWinner) *
                              ris_payload_solid_angle_per_area(rec, canonicalWinner)
                        : 0.0f;
                    // A pair whose neighbour is out of frame, misses, or fails the depth or normal test belongs
                    // to the canonical.
                    float canonicalMis = 0.0f;
                    Reservoir merged = make_empty_reservoir();
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
                            canonicalMis += 1.0f / pairCount;
                            continue;
                        }

                        uint2 neighborCoord = uint2(neighborCoordI);
                        Ray neighborRay = make_center_primary_ray_for_pixel(uniforms, neighborCoord);
                        HitRecord neighborRec = make_empty_hit_record();
                        // The hardware megakernel traces its neighbours like its own camera rays: in hardware mode the
                        // software BVH has no triangles (the triangle buffer is not uploaded), so a software trace here
                        // missed every neighbour and ris_spatial ran as plain RIS.
                        bool neighborHit = trace_scene_hardware(uniforms,
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
                                                                neighborRay,
                                                                kEpsilon,
                                                                kInfinity,
                                                                /*anyHitOnly=*/false,
                                                                /*excludeMeshIndex=*/kInvalidIndex,
                                                                /*excludePrimitiveIndex=*/kInvalidIndex,
                                                                neighborRec);
                        if (!neighborHit || neighborRec.materialIndex >= uniforms.materialCount) {
                            spatialRejectedInvalid += 1u;
                            canonicalMis += 1.0f / pairCount;
                            continue;
                        }

                        float depthRel = fabs(rec.t - neighborRec.t) / max(rec.t, kEpsilon);
                        if (!(depthRel <= 0.1f) || !isfinite(depthRel)) {
                            spatialRejectedDepth += 1u;
                            canonicalMis += 1.0f / pairCount;
                            continue;
                        }
                        float normalDot = dot(safe_normalize(shadingNormal),
                                              safe_normalize(neighborRec.shadingNormal));
                        if (!(normalDot >= 0.9f) || !isfinite(normalDot)) {
                            spatialRejectedNormal += 1u;
                            canonicalMis += 1.0f / pairCount;
                            continue;
                        }

                        spatialNeighborsConsidered += 1u;
                        MaterialData neighborMaterial = materials[neighborRec.materialIndex];
                        float3 neighborWo = safe_normalize(-neighborRay.direction);
                        if (canonicalValid) {
                            canonicalMis += pairwise_mis_canonical_share(
                                canonicalAreaTarget,
                                reservoir_winner_phat_for_hit(uniforms,
                                                              neighborRec,
                                                              neighborMaterial,
                                                              neighborRec.shadingNormal,
                                                              neighborWo,
                                                              clampParams,
                                                              diffuseOcclusion,
                                                              specularOnly,
                                                              canonicalWinner) *
                                    ris_payload_solid_angle_per_area(neighborRec, canonicalWinner),
                                pairCount);
                        }
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
                        float mergeWeight = pairwise_mis_neighbor_weight(
                            phatCurrent * ris_payload_solid_angle_per_area(rec, neighborReservoir.winner),
                            phatNeighbor * ris_payload_solid_angle_per_area(neighborRec, neighborReservoir.winner),
                            neighborReservoir.wSum,
                            pairCount);
                        if (!(mergeWeight > 0.0f)) {
                            spatialRejectedInvalid += 1u;
                            continue;
                        }
                        bool changed = update_reservoir(merged,
                                                        neighborReservoir.winner,
                                                        mergeWeight,
                                                        rand_uniform(state));
                        spatialNeighborsAccepted += 1u;
                        spatialLastMisWeight = mergeWeight / neighborReservoir.wSum;
                        spatialLastMergeWeight = mergeWeight;
                        if (changed) {
                            spatialWinnerOffset = offset;
                        }
                    }
                    if (canonicalValid &&
                        update_reservoir(merged,
                                         canonicalWinner,
                                         canonicalMis * reservoir.wSum,
                                         rand_uniform(state))) {
                        spatialWinnerOffset = int2(0, 0);
                    }
                    if (merged.valid) {
                        merged.M = risTargetM;  // every reservoir merged here holds risTargetM candidates
                    }
                    reservoir = merged;
                }

                if ((uniforms.directLightMode == kDirectLightModeRisTemporalReuse ||
                     uniforms.directLightMode == kDirectLightModeRestirDi ||
                     uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
                    (depth == 0u && !subsurfaceExitVertex)) {
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
                        risAuditTemporalRngBefore = temporalState;
                        Reservoir previousReservoir = make_empty_reservoir();
                        const bool previousReservoirValid =
                            uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid
                                ? build_restir_di_provider_reservoir_for_hit(uniforms,
                                                                             emissivePrimitives,
                                                                             rectangles,
                                                                             materials,
                                                                             environmentTexture,
                                                                             environmentConditionalAlias,
                                                                             environmentMarginalAlias,
                                                                             environmentPdf,
                                                                             rec,
                                                                             material,
                                                                             shadingNormal,
                                                                             wo,
                                                                             clampParams,
                                                                             diffuseOcclusion,
                                                                             specularOnly,
                                                                             risTargetM,
                                                                             temporalState,
                                                                             previousReservoir)
                                : build_ris_reservoir_for_hit(uniforms,
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
                                                              previousReservoir);
                        if (!previousReservoirValid) {
                            risAuditTemporalRngAfter = temporalState;
                            temporalRejectedInvalid += 1u;
                        } else {
                            risAuditTemporalRngAfter = temporalState;
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

                if (PT_DEFAULT_OFF(uniforms.directLightMode == kDirectLightModeRisWorldReuse ? 1u : 0u) != 0u &&
                    (depth == 0u && !subsurfaceExitVertex)) {
                    worldReuseAttempted = true;
                    if (rec.t > 0.0f &&
                        uniforms.width > 0u &&
                        uniforms.height > 0u) {
                        worldCell = world_reuse_cell(rec.point, uniforms.worldReuseCellSize);
                        worldCellHash = world_reuse_cell_hash(worldCell);
                        // Generalized RIS with pairwise MIS, as ris_spatial (reservoir.metal): each candidate pixel's
                        // fresh reservoir streams into `merged` with its pairwise weight, then the canonical winner
                        // joins with its share of every pair. The old merge weighted each candidate by p_c / (p_c M +
                        // p_n M), kept the canonical unweighted and added 1 to M per candidate: 0.824 of the exact
                        // light on the EP12 sun plane. Targets are compared per unit light area, which carries the
                        // solid-angle Jacobian of moving a triangle-light sample between pixels.
                        constexpr uint kWorldReusePairs = 4u;
                        const float pairCount = float(kWorldReusePairs);
                        const bool canonicalValid = reservoir.valid;
                        const RISSamplePayload canonicalWinner = reservoir.winner;
                        const float canonicalAreaTarget = canonicalValid
                            ? reservoir_winner_phat_for_hit(uniforms,
                                                            rec,
                                                            material,
                                                            shadingNormal,
                                                            wo,
                                                            clampParams,
                                                            diffuseOcclusion,
                                                            specularOnly,
                                                            canonicalWinner) *
                                  ris_payload_solid_angle_per_area(rec, canonicalWinner)
                            : 0.0f;
                        // A pair whose candidate is out of frame, misses, or fails the cell, depth or normal test
                        // belongs to the canonical.
                        float canonicalMis = 0.0f;
                        Reservoir merged = make_empty_reservoir();
                        constexpr int2 offsets[4] = {
                            int2(-2, 0),
                            int2(2, 0),
                            int2(0, -2),
                            int2(0, 2)
                        };
                        for (uint n = 0u; n < kWorldReusePairs; ++n) {
                            int2 candidateCoordI = int2(pixelCoord) + offsets[n];
                            if (candidateCoordI.x < 0 ||
                                candidateCoordI.y < 0 ||
                                candidateCoordI.x >= int(uniforms.width) ||
                                candidateCoordI.y >= int(uniforms.height)) {
                                worldRejectedInvalid += 1u;
                                canonicalMis += 1.0f / pairCount;
                                continue;
                            }

                            uint2 candidateCoord = uint2(candidateCoordI);
                            Ray candidateRay = make_center_primary_ray_for_pixel(uniforms, candidateCoord);
                            HitRecord candidateRec = make_empty_hit_record();
                            // The hardware megakernel traces its candidates like its own camera rays: in hardware mode
                            // the software BVH has no triangles (the triangle buffer is not uploaded), so a software
                            // trace here missed every candidate and ris_world ran as plain RIS.
                            bool candidateHit = trace_scene_hardware(uniforms,
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
                                                                     candidateRay,
                                                                     kEpsilon,
                                                                     kInfinity,
                                                                     /*anyHitOnly=*/false,
                                                                     /*excludeMeshIndex=*/kInvalidIndex,
                                                                     /*excludePrimitiveIndex=*/kInvalidIndex,
                                                                     candidateRec);
                            if (!candidateHit || candidateRec.materialIndex >= uniforms.materialCount) {
                                worldRejectedInvalid += 1u;
                                canonicalMis += 1.0f / pairCount;
                                continue;
                            }

                            int3 candidateCell =
                                world_reuse_cell(candidateRec.point, uniforms.worldReuseCellSize);
                            if (!world_reuse_cell_compatible(worldCell, candidateCell)) {
                                worldRejectedCell += 1u;
                                canonicalMis += 1.0f / pairCount;
                                continue;
                            }

                            float depthRel = fabs(rec.t - candidateRec.t) / max(rec.t, kEpsilon);
                            if (!(depthRel <= 0.2f) || !isfinite(depthRel)) {
                                worldRejectedDepth += 1u;
                                canonicalMis += 1.0f / pairCount;
                                continue;
                            }
                            float normalDot = dot(safe_normalize(shadingNormal),
                                                  safe_normalize(candidateRec.shadingNormal));
                            if (!(normalDot >= 0.85f) || !isfinite(normalDot)) {
                                worldRejectedNormal += 1u;
                                canonicalMis += 1.0f / pairCount;
                                continue;
                            }

                            worldCandidatesConsidered += 1u;
                            MaterialData candidateMaterial = materials[candidateRec.materialIndex];
                            float3 candidateWo = safe_normalize(-candidateRay.direction);
                            if (canonicalValid) {
                                canonicalMis += pairwise_mis_canonical_share(
                                    canonicalAreaTarget,
                                    reservoir_winner_phat_for_hit(uniforms,
                                                                  candidateRec,
                                                                  candidateMaterial,
                                                                  candidateRec.shadingNormal,
                                                                  candidateWo,
                                                                  clampParams,
                                                                  diffuseOcclusion,
                                                                  specularOnly,
                                                                  canonicalWinner) *
                                        ris_payload_solid_angle_per_area(candidateRec, canonicalWinner),
                                    pairCount);
                            }
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
                                                             candidateReservoir) ||
                                !candidateReservoir.valid) {
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
                            float mergeWeight = pairwise_mis_neighbor_weight(
                                phatCurrent * ris_payload_solid_angle_per_area(rec, candidateReservoir.winner),
                                phatCandidate *
                                    ris_payload_solid_angle_per_area(candidateRec, candidateReservoir.winner),
                                candidateReservoir.wSum,
                                pairCount);
                            if (!(mergeWeight > 0.0f)) {
                                worldRejectedInvalid += 1u;
                                continue;
                            }
                            update_reservoir(merged,
                                             candidateReservoir.winner,
                                             mergeWeight,
                                             rand_uniform(state));
                            worldCandidatesAccepted += 1u;
                            worldReuseAccepted = true;
                            worldCandidatePrimitiveIndex = candidateReservoir.winner.primitiveIndex;
                            worldLastMisWeight = mergeWeight / candidateReservoir.wSum;
                            worldLastMergeWeight = mergeWeight;
                        }
                        if (canonicalValid) {
                            update_reservoir(merged,
                                             canonicalWinner,
                                             canonicalMis * reservoir.wSum,
                                             rand_uniform(state));
                        }
                        if (merged.valid) {
                            merged.M = risTargetM;  // every reservoir merged here holds risTargetM candidates
                        }
                        reservoir = merged;
                    } else {
                        worldRejectedInvalid += 1u;
                    }
                }

                if ((uniforms.directLightMode == kDirectLightModeRisRegirCache ||
                     uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid) &&
                    (depth == 0u && !subsurfaceExitVertex) &&
                    uniforms.frameIndex > 0u) {
                    cacheReuseAttempted = true;
                    cacheFallbackUsed = true;
                    cacheCell = world_reuse_cell(rec.point, uniforms.worldReuseCellSize);
                    cacheCellHash = world_reuse_cell_hash(cacheCell);
                    if (uniforms.frameIndex > 1u) {
                        cacheSourceFrameIndex = uniforms.frameIndex - 2u;
                    } else if (uniforms.frameIndex > 0u) {
                        cacheSourceFrameIndex = uniforms.frameIndex - 1u;
                    }
                    if (rec.t > 0.0f &&
                        uniforms.frameIndex > 0u) {
                        cacheStateAvailable = true;
                        cacheFallbackUsed = false;
                        cacheCandidatesConsidered = 0u;
                        cacheEntriesAvailable = 0u;
                        // The cached reservoirs are fresh RIS reservoirs built at this pixel's own hit point, with the
                        // same builder and target function as the pixel's reservoir and with random numbers seeded per
                        // cell and frame. Reservoirs with one target at one point merge by plain streaming: each adds
                        // its wSum and M grows by its candidate count (generalized RIS with MIS weights in proportion
                        // to M). Every probe is merged, valid or not, since each drew risTargetM candidates. The old
                        // merge weighted each cached reservoir by p(y_c) / (p(y) M + p(y_c) M_c), kept the two with the
                        // largest weights (a choice that depends on the samples) and added 1 to M per merge: 0.946 of
                        // the exact light on the EP12 sun plane under ris_regir and 0.962 under the hybrid.
                        // The pixel's reservoir enters with its own contribution weight wSum / M and a fixed share of
                        // nominalM candidates: risTargetM, or twice that for the hybrid, whose temporal step merges a
                        // second reservoir of risTargetM candidates when both are valid. Its actual M depends on the
                        // samples there, so it cannot be its share: MIS weights that depend on the samples would bias
                        // the merge.
                        const uint nominalM =
                            (uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid ? 2u : 1u) * risTargetM;
                        if (reservoir.valid && reservoir.M != nominalM) {
                            reservoir.wSum *= float(nominalM) / float(reservoir.M);
                        }
                        uint mergedM = nominalM;
                        const uint firstSourceFrameIndex = cacheSourceFrameIndex;
                        for (uint cacheProbe = 0u; cacheProbe < 4u; ++cacheProbe) {
                            if (cacheProbe > firstSourceFrameIndex) {
                                continue;
                            }
                            uint candidateSourceFrameIndex = firstSourceFrameIndex - cacheProbe;
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
                            const bool cachedReservoirValid =
                                uniforms.directLightMode == kDirectLightModeRestirDiRegirHybrid
                                ? build_restir_di_provider_reservoir_for_hit(uniforms,
                                                                             emissivePrimitives,
                                                                             rectangles,
                                                                             materials,
                                                                             environmentTexture,
                                                                             environmentConditionalAlias,
                                                                             environmentMarginalAlias,
                                                                             environmentPdf,
                                                                             rec,
                                                                             material,
                                                                             shadingNormal,
                                                                             wo,
                                                                             clampParams,
                                                                             diffuseOcclusion,
                                                                             specularOnly,
                                                                             risTargetM,
                                                                             cacheState,
                                                                             cachedReservoir)
                                : build_ris_reservoir_for_hit(uniforms,
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
                                                              cachedReservoir);
                            mergedM += risTargetM;
                            if (!cachedReservoirValid || !cachedReservoir.valid) {
                                cacheRejectedInvalid += 1u;
                                continue;
                            }
                            cacheEntriesAvailable += 1u;
                            if (update_reservoir(reservoir,
                                                 cachedReservoir.winner,
                                                 cachedReservoir.wSum,
                                                 rand_uniform(state))) {
                                cacheSourceFrameIndex = candidateSourceFrameIndex;
                                cacheCandidatePrimitiveIndex = cachedReservoir.winner.primitiveIndex;
                            }
                            cacheCandidatesAccepted += 1u;
                            cacheReuseAccepted = true;
                            cacheLastMergeWeight = cachedReservoir.wSum;
                        }
                        if (reservoir.valid) {
                            reservoir.M = mergedM;
                            // MIS weight of one cached reservoir.
                            cacheLastMisWeight = float(risTargetM) / float(mergedM);
                        }
                    } else {
                        cacheRejectedInvalid += 1u;
                    }
                    if (stats) {
                        if (cacheCandidatesConsidered > 0u) {
                            atomic_fetch_add_explicit(&stats->restirDiRegirCandidateCount,
                                                      cacheCandidatesConsidered,
                                                      memory_order_relaxed);
                        }
                        if (cacheCandidatesAccepted > 0u) {
                            atomic_fetch_add_explicit(&stats->restirDiRegirAcceptedCount,
                                                      cacheCandidatesAccepted,
                                                      memory_order_relaxed);
                        } else if (cacheReuseAttempted) {
                            atomic_fetch_add_explicit(&stats->restirDiRegirFallbackCount,
                                                      1u,
                                                      memory_order_relaxed);
                        }
                    }
                }

                if (reservoir.valid && reservoir.M > 0u && reservoir.wSum > 0.0f) {
                    RISSamplePayload winner = reservoir.winner;
                    float distSq = ris_payload_distance_sq(rec, winner);
                    if (distSq > 0.0f && isfinite(distSq)) {
                        const bool winnerDirectional = ris_payload_is_directional(winner);
                        float distance = winnerDirectional ? 1.0e20f : length(winner.position - rec.point);
                        float3 omegaL = ris_payload_omega_l(rec, winner);
                        float cosThetaSurface = direct_light_target_cos(uniforms, material, rec, shadingNormal, omegaL);
                        float cosThetaLight = winnerDirectional ? 1.0f : saturate(emissive_light_cosine(winner.flags, winner.normal, -omegaL));
                        bool cutoutBackLightUnsampled = false;
                        BsdfEvalResult bsdfEval = direct_light_target_bsdf(uniforms,
                                                                           material,
                                                                           rec,
                                                                           shadingNormal,
                                                                           wo,
                                                                           omegaL,
                                                                           clampParams,
                                                                           diffuseOcclusion,
                                                                           specularOnly,
                                                                           winnerDirectional,
                                                                           cutoutBackLightUnsampled);
                        float phatW = p_hat(winner.emission,
                                            bsdfEval.value,
                                            cosThetaSurface,
                                            cosThetaLight,
                                            distSq);
                        reservoir.W = (phatW > 0.0f)
                                    ? ((reservoir.wSum / float(reservoir.M)) / phatW)
                                    : 0.0f;

                        Ray shadowRay;
                        shadowRay.origin = offset_spawn_origin(rec, omegaL, material);
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
                                                            spawn_ray_min(material, kEpsilon),
                                                            shadowMax,
                                                            /*anyHitOnly=*/true,
                                                            /*includeTriangles=*/true,
                                                            shadowRec);
                        } else {
#endif
                            occluded = trace_scene_hardware_alpha_aware_shadow<false>(uniforms,
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
                                                                                      spawn_ray_min(material, kEpsilon),
                                                                                      shadowMax,
                                                                                      shadowExcludeMesh,
                                                                                      shadowExcludePrim,
                                                                                      materials,
                                                                                      materialTextures,
                                                                                      materialSamplers,
                                                                                      materialTextureInfos,
                                                                                      shadowRec,
                                                                                      pavilion_punctual_shadow_enabled(material, winnerDirectional), depth);
                            if (!winnerDirectional &&
                                occluded &&
                                shadowRec.primitiveType == kPrimitiveTypeTriangle &&
                                shadowRec.primitiveIndex == winner.primitiveIndex) {
                                occluded = false;
                            }
                            if (!winnerDirectional &&
                                (winner.flags & kLightPrimitiveFlagRectangle) != 0u &&
                                occluded &&
                                shadowRec.primitiveType == kPrimitiveTypeRectangle &&
                                shadowRec.primitiveIndex == winner.primitiveIndex) {
                                occluded = false;
                            }
#if PT_DEBUG_TOOLS
                        }
#endif

                        float3 contribution = float3(0.0f);
                        // Candidates are weighted by the light's emission factor; shade with
                        // the textured emission at the chosen point, which the BSDF hit sees.
                        RISSamplePayload shadedWinner = winner;
                        const bool winnerEmissionResolved =
                            resolve_textured_direct_light_emission(uniforms, shadedWinner, materials,
                                materialTextures, materialSamplers, materialTextureInfos,
                                meshInfos, sceneVertices, meshIndices);
                        // Plain RIS shares mesh emitters with the BSDF hit
                        // (mesh_emitter_bsdf_hit_weight); reuse modes keep weight 1.
                        const float risMisWeight = winnerDirectional ? 1.0f
                            : mesh_emitter_nee_mis_weight(uniforms, depth,
                                  mesh_emitter_nee_candidate_count(uniforms) * winner.pdf, nee_mis_bsdf_pdf(bsdfEval, cutoutBackLightUnsampled));
                        if (!occluded &&
                            winnerEmissionResolved &&
                            !bsdfEval.isDelta &&
                            !bsdfEval.isBssrdf &&
                            reservoir.W > 0.0f &&
                            cosThetaSurface > 0.0f) {
                            contribution = bsdfEval.value * shadedWinner.emission;
                            contribution *= cosThetaSurface * reservoir.W * risMisWeight;
                            contribution *= volume_surface_nee_transmittance(
                                uniforms, sqrt(distSq), winnerDirectional, stats);
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
        }
        // The emissive-primitive estimator does not sample analytic rectangles.
        const uint independentRectFilter = kAnalyticLightFilterAll;
        const uint independentRectCount = rectLightCount;
        if (!surfaceIsDelta && independentRectCount > 0u) {
            for (uint neeDraw = 0u; neeDraw < rect_light_nee_draw_count(uniforms); ++neeDraw) {
                RectLightSample lightSample;
                if (sample_rect_light_nee_draw(uniforms,
                                      rectangles,
                                      materials,
                                      environmentTexture,
                                      rec,
                                      state,
                                      independentRectCount,
                                      lightSample,
                                      independentRectFilter, neeDraw)) {
                    lightSample.emission = spectral_radiance_to_channels(lightSample.emission, spectralState);
                    float nDotL =
                        two_sided_alpha_cutout_direct_cos(uniforms, material, rec, shadingNormal, lightSample.direction);
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
                        shadowRay.origin = offset_spawn_origin(rec, lightSample.direction, material);
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
                                                            spawn_ray_min(material, kEpsilon),
                                                            shadowMax,
                                                            /*anyHitOnly=*/true,
                                                            /*includeTriangles=*/true,
                                                            shadowRec);
                            if (occluded &&
                                shadowRec.primitiveType == kPrimitiveTypeRectangle &&
                                shadowRec.primitiveIndex == lightSample.rectIndex) {
                                occluded = false;
                            }
                        } else {
    #endif
                            occluded = trace_scene_hardware_alpha_aware_shadow<false>(uniforms,
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
                                                                                      spawn_ray_min(material, kEpsilon),
                                                                                      shadowMax,
                                                                                      shadowExcludeMesh,
                                                                                      shadowExcludePrim,
                                                                                      materials,
                                                                                      materialTextures,
                                                                                      materialSamplers,
                                                                                      materialTextureInfos,
                                                                                      shadowRec,
                                                                                      pavilion_punctual_shadow_enabled(material, lightSample.isReferencePoint), depth);
                            if (occluded &&
                                shadowRec.primitiveType == kPrimitiveTypeRectangle &&
                                shadowRec.primitiveIndex == lightSample.rectIndex) {
                                occluded = false;
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
                                                                   spawn_ray_min(material, kEpsilon),
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
                            BsdfEvalResult bsdfEval = material_closure_eval(material,
                                                                    rec.point,
                                                                    shadingNormal,
                                                                    wo,
                                                                    lightSample.direction,
                                                                    clampParams,
                                                                    PT_DEFAULT_OFF(uniforms.sssMode),
                                                                    diffuseOcclusion,
                                                                    specularOnly,
                                                                    lightSample.isReferencePoint);
                            const bool cutoutBackLightUnsampled = apply_two_sided_alpha_cutout_diffuse_fallback(uniforms,
                                                                          material,
                                                                          rec,
                                                                          shadingNormal,
                                                                          lightSample.direction,
                                                                          diffuseOcclusion,
                                                                          bsdfEval);
                            float3 bsdfValue = lightSample.isReferencePoint ? bsdfEval.value
                                : reference_continuation_nee_value(material, shadingNormal, wo,
                                                                   lightSample.direction, bsdfEval.value);
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
                                    float bsdfPdf = nee_mis_bsdf_pdf(bsdfEval, cutoutBackLightUnsampled);
                                    float weight = 1.0f;
                                    const bool referenceSurface = (material.materialFlags &
                                        (kMaterialFlagSingleScatterSpecular | kMaterialFlagReferenceDielectricContinuation)) != 0u;
                                    if (bsdfPdf > 0.0f && !lightSample.isReferencePoint &&
                                        (!referenceSurface || depth + 1u < uniforms.maxDepth)) {
                                        float denom = lightSample.pdf + bsdfPdf;
                                        if (denom > 0.0f) {
                                            weight = clamp(lightSample.pdf / denom,
                                                           kMisWeightClampMin,
                                                           kMisWeightClampMax);
                                        }
                                    }
                                    float3 contribution = float3(0.0f);
                                    float causticMisWeight = weight;
                                    const bool causticConnected = !lightSample.isReferencePoint &&
                                        caustic_transport_connect_light_sample(uniforms,
                                                                               material,
                                                                               rec.point,
                                                                               shadingNormal,
                                                                               rec.normal,
                                                                               lightSample.direction,
                                                                               lightSample.position,
                                                                               lightSample.normal,
                                                                               false,
                                                                               throughput,
                                                                               lightSample.emission,
                                                                               bsdfValue,
                                                                               bsdfPdf,
                                                                               lightSample.pdf,
                                                                               nDotL,
                                                                               depth,
                                                                               mediumDepth,
                                                                               causticEyeVertices,
                                                                               causticLightVertices,
                                                                               causticConnections,
                                                                               causticConnectionResults,
                                                                               stats,
                                                                               contribution,
                                                                               causticMisWeight);
                                    if (!causticConnected) {
                                        contribution = lightSample.emission * bsdfValue * nDotL;
                                        contribution *= weight / lightSample.pdf;
                                    } else {
                                        weight = causticMisWeight;
                                    }
                                    contribution *= volume_surface_nee_transmittance(
                                        uniforms, lightSample.distance, false, stats);
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
}
        if (debugContext &&
            PT_DEFAULT_OFF(uniforms.debugDirectLightAudit) != 0u &&
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
                               float4(float(closureType),
                                      float(rec.primitiveIndex),
                                      material.typeEta.x,
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
            record_debug_event(*debugContext,
                               kDebugEventDirectLightAuditMaterialExt,
                               depth,
                               mediumDepth,
                               mediumDepth,
                               /*mediumEvent=*/0,
                               rec.frontFace,
                               rec.materialIndex,
                               /*scatterIsDelta=*/false,
                               material.pbrParams.xyz,
                               float4(material.pbrParams.w,
                                      material.typeEta.y,
                                      material.pbrExtras.z,
                                      diffuseOcclusion),
                               float3(material.pbrExtras.x,
                                      material.pbrExtras.y,
                                      material.pbrExtras.w));
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
                record_debug_event(*debugContext,
                                   kDebugEventRisAuditRng,
                                   depth,
                                   mediumDepth,
                                   mediumDepth,
                                   /*mediumEvent=*/0,
                                   rec.frontFace,
                                   rec.materialIndex,
                                   /*scatterIsDelta=*/false,
                                   float3(as_type<float>(risAuditRngBefore),
                                          as_type<float>(risAuditRngAfter),
                                          0.0f),
                                   float4(as_type<float>(risAuditTemporalRngBefore),
                                          as_type<float>(risAuditTemporalRngAfter),
                                          0.0f,
                                          0.0f),
                                   float3(0.0f));
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
                    BsdfEvalResult bsdfEval = material_closure_eval(material,
                                                            rec.point,
                                                            shadingNormal,
                                                            wo,
                                                            auditLight.direction,
                                                            clampParams,
                                                            PT_DEFAULT_OFF(uniforms.sssMode),
                                                            diffuseOcclusion,
                                                            specularOnly,
                                                            auditLight.isReferencePoint);
                    float bsdfPdf = bsdfEval.pdf;
                    float misWeight = 1.0f;
                    if (bsdfPdf > 0.0f && !auditLight.isReferencePoint) {
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
                    auditShadowRay.origin = offset_spawn_origin(rec, auditLight.direction, material);
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
                                                         spawn_ray_min(material, kEpsilon),
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
                                                        spawn_ray_min(material, kEpsilon),
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
                                                                   spawn_ray_min(material, kEpsilon),
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
                                                     spawn_ray_min(material, kEpsilon),
                                                     shadowMax,
                                                     /*anyHitOnly=*/true,
                                                     /*includeTriangles=*/true,
                                                     &swShadowAudit,
                                                     auditShadowRec);
#endif
                    // The audit reports what plain NEE sees. NEE's shadow rays pass alpha-cutout holes and BLEND texels (the
                    // alpha-aware tracers); the raw trace above, kept for its traversal record, stops at the first one.
                    {
                        HitRecord alphaShadowRec = make_empty_hit_record();
#if PT_DEBUG_TOOLS
                        if (forceSoftware) {
                            occluded = trace_scene_software_alpha_aware_shadow<false>(
                                uniforms, spheres, rectangles, triangleData,
                                tlasNodes, tlasPrimIndices, instanceInfos,
                                blasNodes, blasPrimIndices, nodes, primitiveIndices,
                                /*stats=*/nullptr, auditShadowRay, spawn_ray_min(material, kEpsilon), shadowMax, materials,
                                materialTextures, materialSamplers, materialTextureInfos,
                                meshInfos, sceneVertices, meshIndices, alphaShadowRec, pavilion_punctual_shadow_enabled(material, auditLight.isReferencePoint));
                        } else
#endif
                        {
                            uint alphaExcludeMesh;
                            uint alphaExcludePrim;
                            compute_exclusion_indices(rec, alphaExcludeMesh, alphaExcludePrim);
                            occluded = trace_scene_hardware_alpha_aware_shadow<false>(
                                uniforms, accel, meshInfos, triangleData, sceneVertices, meshIndices, instanceUserIds,
                                spheres, rectangles, nodes, primitiveIndices,
                                /*stats=*/nullptr, auditShadowRay, spawn_ray_min(material, kEpsilon), shadowMax,
                                alphaExcludeMesh, alphaExcludePrim, materials, materialTextures, materialSamplers,
                                materialTextureInfos, alphaShadowRec, pavilion_punctual_shadow_enabled(material, auditLight.isReferencePoint));
                        }
                        if (occluded && alphaShadowRec.primitiveType == kPrimitiveTypeRectangle &&
                            alphaShadowRec.primitiveIndex == auditLight.rectIndex) {
                            occluded = false;
                        }
                    }
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
                    BsdfEvalResult bsdfEval = material_closure_eval(material,
                                                            rec.point,
                                                            shadingNormal,
                                                            wo,
                                                            auditLight.direction,
                                                            clampParams,
                                                            PT_DEFAULT_OFF(uniforms.sssMode),
                                                            diffuseOcclusion,
                                                            specularOnly,
                                                            (emissivePrimitives[primitiveAuditIndex].flags & kLightPrimitiveFlagDirectional) != 0u);
                    float bsdfPdf = bsdfEval.pdf;
                    float misWeight = 1.0f;
                    HitRecord auditShadowRec = make_empty_hit_record();
                    HardwareShadowTraceAudit hwShadowAudit;
                    SoftwareShadowTraceAudit swShadowAudit;
                    Ray auditShadowRay;
                    auditShadowRay.origin = offset_spawn_origin(rec, auditLight.direction, material);
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
                                                         spawn_ray_min(material, kEpsilon),
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
                                                        spawn_ray_min(material, kEpsilon),
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
                                                                   spawn_ray_min(material, kEpsilon),
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
                                                     spawn_ray_min(material, kEpsilon),
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
                    // The audit reports what plain NEE sees. NEE's shadow rays pass alpha-cutout holes and BLEND texels (the
                    // alpha-aware tracers); the raw trace above, kept for its traversal record, stops at the first one.
                    {
                        HitRecord alphaShadowRec = make_empty_hit_record();
#if PT_DEBUG_TOOLS
                        if (forceSoftware) {
                            occluded = trace_scene_software_alpha_aware_shadow<false>(
                                uniforms, spheres, rectangles, triangleData,
                                tlasNodes, tlasPrimIndices, instanceInfos,
                                blasNodes, blasPrimIndices, nodes, primitiveIndices,
                                /*stats=*/nullptr, auditShadowRay, spawn_ray_min(material, kEpsilon), shadowMax, materials,
                                materialTextures, materialSamplers, materialTextureInfos,
                                meshInfos, sceneVertices, meshIndices, alphaShadowRec, false);
                        } else
#endif
                        {
                            uint alphaExcludeMesh;
                            uint alphaExcludePrim;
                            compute_exclusion_indices(rec, alphaExcludeMesh, alphaExcludePrim);
                            occluded = trace_scene_hardware_alpha_aware_shadow<false>(
                                uniforms, accel, meshInfos, triangleData, sceneVertices, meshIndices, instanceUserIds,
                                spheres, rectangles, nodes, primitiveIndices,
                                /*stats=*/nullptr, auditShadowRay, spawn_ray_min(material, kEpsilon), shadowMax,
                                alphaExcludeMesh, alphaExcludePrim, materials, materialTextures, materialSamplers,
                                materialTextureInfos, alphaShadowRec, false);
                        }
                        if (occluded && alphaShadowRec.primitiveType == kPrimitiveTypeTriangle &&
                            alphaShadowRec.primitiveIndex == auditLight.primitiveIndex) {
                            occluded = false;
                        }
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
                if (useOverride) {
                    envSample.radiance = environment_color_lod(environmentTexture,
                                                               envSample.direction,
                                                               uniforms.environmentRotation,
                                                               uniforms.environmentIntensity,
                                                               overrideLod,
                                                               uniforms);
                }
                float nDotL =
                    two_sided_alpha_cutout_direct_cos(uniforms, material, rec, shadingNormal, envSample.direction);
                if (envSample.pdf > 0.0f && nDotL > 0.0f) {
                    BsdfEvalResult bsdfEval = material_closure_eval(material,
                                                            rec.point,
                                                            shadingNormal,
                                                            wo,
                                                            envSample.direction,
                                                            clampParams,
                                                            PT_DEFAULT_OFF(uniforms.sssMode),
                                                            diffuseOcclusion,
                                                            specularOnly);
                    const bool cutoutBackLightUnsampled = apply_two_sided_alpha_cutout_diffuse_fallback(uniforms,
                                                                  material,
                                                                  rec,
                                                                  shadingNormal,
                                                                  envSample.direction,
                                                                  diffuseOcclusion,
                                                                  bsdfEval);
                    float3 bsdfValue = reference_continuation_nee_value(material, shadingNormal,
                                                                                   wo, envSample.direction, bsdfEval.value);
                    float maxComponent = max(max(bsdfValue.x, bsdfValue.y), bsdfValue.z);
                    bool bsdfConnectable = !bsdfEval.isDelta &&
                                           !bsdfEval.isBssrdf &&
                                           (maxComponent > 0.0f);
                    if (bsdfConnectable) {
                        float bsdfPdf = nee_mis_bsdf_pdf(bsdfEval, cutoutBackLightUnsampled);
                        float weight = 1.0f;
                        const bool referenceSurface = (material.materialFlags &
                            (kMaterialFlagSingleScatterSpecular | kMaterialFlagReferenceDielectricContinuation)) != 0u;
                        if (bsdfPdf > 0.0f && (!referenceSurface || depth + 1u < uniforms.maxDepth)) {
                            float denom = envSample.pdf + bsdfPdf;
                            if (denom > 0.0f) {
                                weight = clamp(envSample.pdf / denom,
                                               kMisWeightClampMin,
                                               kMisWeightClampMax);
                            }
                        }
                        Ray shadowRay;
                        shadowRay.origin = offset_spawn_origin(rec, envSample.direction, material);
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
                                                            spawn_ray_min(material, kEpsilon),
                                                            kInfinity,
                                                            /*anyHitOnly=*/true,
                                                            /*includeTriangles=*/true,
                                                            shadowRec);
                        } else {
#endif
                            occluded = trace_scene_hardware_alpha_aware_shadow<false>(uniforms,
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
                                                                               spawn_ray_min(material, kEpsilon),
                                                                               kInfinity,
                                                                               shadowExcludeMesh,
                                                                               shadowExcludePrim,
                                                                               materials,
                                                                               materialTextures,
                                                                               materialSamplers,
                                                                               materialTextureInfos,
                                                                               shadowRec, false, depth);
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
                                                                   spawn_ray_min(material, kEpsilon),
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
                        bool useMneeEnvChain = (PT_DEFAULT_OFF(uniforms.enableMnee) != 0u) &&
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
                            envRadiance = spectral_radiance_to_channels(envRadiance, spectralState);
                            float3 contribution = float3(0.0f);
                            float causticMisWeight = weight;
                            const bool causticConnected =
                                !usedDeltaChain &&
                                caustic_transport_connect_light_sample(uniforms,
                                                                       material,
                                                                       rec.point,
                                                                       shadingNormal,
                                                                       rec.normal,
                                                                       envSample.direction,
                                                                       rec.point + safe_normalize(envSample.direction),
                                                                       -safe_normalize(envSample.direction),
                                                                       true,
                                                                       throughput,
                                                                       envRadiance,
                                                                       bsdfValue,
                                                                       bsdfPdf,
                                                                       envSample.pdf,
                                                                       nDotL,
                                                                       depth,
                                                                       mediumDepth,
                                                                       causticEyeVertices,
                                                                       causticLightVertices,
                                                                       causticConnections,
                                                                       causticConnectionResults,
                                                                       stats,
                                                                       contribution,
                                                                       causticMisWeight);
                            if (!causticConnected) {
                                contribution = envRadiance * bsdfValue * nDotL;
                                contribution *= weight / envSample.pdf;
                            } else {
                                weight = causticMisWeight;
                            }
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
                                      float(closureType),
                                      0.0f,
                                      0.0f),
                               shadingNormal);
        }

        BsdfSampleResult bsdfSample;
        uint rngStateBeforeBsdf = state;
        bsdfSample = material_closure_sample(material,
                                             rec.point,
                                             shadingNormal,
                                             wo,
                                             incidentDir,
                                             rec.frontFace != 0u,
                                             state,
                                             clampParams,
                                             PT_DEFAULT_OFF(uniforms.sssMode),
                                             diffuseOcclusion,
                                 specularOnly);
        if (subsurfaceExitVertex) {
            // Keeps this vertex out of ReSTIR GI, ReSTIR PT, path guiding and the radiance cache, whose
            // per-pixel state belongs to the primary hit and not to the walk's exit point.
            bsdfSample.isBssrdf = true;
        }
        apply_restir_gi_production_diffuse(uniforms,
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
        apply_restir_pt_production_path_reuse(uniforms,
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
        capture_restir_pt_production_reservoir(uniforms,
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
        if (deferredRestirActivityDebug && depth >= 1u) {
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
                                      float(closureType),
                                      subsurfaceExitVertex ? 1.0f : 0.0f),
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
        if (bsdfSample.pdf <= 0.0f ||
            falcor_face_visibility(material, rec, bsdfSample.direction) == 0.0f) {
            break;
        }
        if (!bsdfSample.isDelta) {
            pathMaxRoughness = max(pathMaxRoughness, path_regularization_roughness(material));
        }
        caustic_transport_note_path_vertex(uniforms,
                                           material,
                                           rec.point,
                                           shadingNormal,
                                           rec.normal,
                                           bsdfSample.direction,
                                           throughput,
                                           depth,
                                           mediumDepth,
                                           bsdfSample,
                                           causticEyeVertices,
                                           causticLightVertices,
                                           causticConnections,
                                           causticConnectionResults,
                                           stats);
        float cacheConfidence = 0.0f;
        float cacheVariance = 0.0f;
        float cacheBiasRisk = 0.0f;
        // A training path never loads from the cache: its suffix is the unbiased sample the cache learns from.
        if (!cacheTraining.active &&
            radiance_cache_query_and_maybe_terminate(uniforms,
                                                     material,
                                                     rec.point,
                                                     shadingNormal,
                                                     specularOnly,
                                                     depth,
                                                     bsdfSample,
                                                     radianceCacheStates,
                                                     stats,
                                                     radiance,
                                                     throughput,
                                                     cacheConfidence,
                                                     cacheVariance,
                                                     cacheBiasRisk)) {
            if (outReconstructionConfidence) {
                *outReconstructionConfidence = max(*outReconstructionConfidence, cacheConfidence);
            }
            if (outReconstructionVariance) {
                *outReconstructionVariance = max(*outReconstructionVariance, cacheVariance);
            }
            if (outReconstructionBiasRisk) {
                *outReconstructionBiasRisk = max(*outReconstructionBiasRisk, cacheBiasRisk);
            }
            break;
        }
        radiance_cache_training_note_vertex(cacheTraining,
                                            uniforms,
                                            material,
                                            rec.point,
                                            shadingNormal,
                                            specularOnly,
                                            depth,
                                            bsdfSample,
                                            radiance,
                                            throughput);

        uint mediumDepthBefore = mediumDepth;
        float3 selectedSigma = float3(0.0f);
        if (interface_select_medium(material, rec, bsdfSample.direction, selectedSigma, mediumDepth)) {
            if (source_material_flag(material, kMaterialFlagExplicitInterior)) mediumSigmaStack[0] = selectedSigma;
        } else if (bsdfSample.mediumEvent == 1) {
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
        volume_apply_medium_state(ray, uniforms, mediumDepthAfter, volumeScatteringDepth);

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

        uint nextSpecularDepth = bsdfSample.isDelta ? (specularDepth + 1u) : 0u;
        bool didTransmission = false;
        if (bsdfSample.isDelta && material_closure_is_transmissive(material)) {
            float3 dir = bsdfSample.direction;
            if (all(isfinite(dir)) && dot(dir, dir) > 0.0f) {
                float side = (rec.frontFace != 0u) ? 1.0f : -1.0f;
                didTransmission = (dot(shadingNormal, dir) * side) < 0.0f;
            }
        }
        specularDepth = nextSpecularDepth;

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
            nextOrigin = interface_precision(material)
                ? offset_interface_origin(rec, bsdfSample.direction)
                : offset_ray_origin(rec, bsdfSample.direction);
        }
        interfaceMinT = spawn_ray_min(material, kEpsilon);
        mediumSegmentOffset = interface_precision(material)
            ? dot(nextOrigin - rec.point, bsdfSample.direction) : 0.0f;

        bool useMnee = (ENABLE_MNEE_CAUSTICS != 0) && (PT_DEFAULT_OFF(uniforms.enableMnee) != 0u);
        bool specNeeEnabled = (uniforms.enableSpecularNee != 0u);
        float dirLenSq = dot(bsdfSample.direction, bsdfSample.direction);
        bool specDirectionValid = (dirLenSq > 0.0f) && all(isfinite(bsdfSample.direction));
        bool mneeEligible = false;
#if ENABLE_MNEE_CAUSTICS
        mneeEligible = useMnee &&
                       bsdfSample.isDelta &&
                       ((bsdfSample.mediumEvent <= 0) || didTransmission) &&
                       material_closure_is_transmissive(material) &&
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
                                                spawn_ray_min(material, kEpsilon),
                                                kInfinity,
                                                /*anyHitOnly=*/true,
                                                /*includeTriangles=*/true,
                                                shadowRec);
            } else {
#endif
                occluded = trace_scene_hardware_alpha_aware_shadow<false>(uniforms,
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
                                                                   spawn_ray_min(material, kEpsilon),
                                                                   kInfinity,
                                                                   neeExcludeMesh,
                                                                   neeExcludePrim,
                                                                   materials,
                                                                   materialTextures,
                                                                   materialSamplers,
                                                                   materialTextureInfos,
                                                                   shadowRec, false, depth);
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
                                                       spawn_ray_min(material, kEpsilon),
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
                float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                float denom = envPdf + bsdfPdf;
                float misWeight = (denom > 0.0f) ? (envPdf / denom) : 0.0f;
                misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                float3 envColor = environment_color(environmentTexture,
                                                    neeRay.direction,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity,
                                                    uniforms);
                // The delta sample fixed this direction; it was not drawn from the
                // environment, so there is no 1/envPdf. This connection takes the
                // light strategy's MIS share and the continuation's environment miss
                // takes bsdfPdf / (bsdfPdf + envPdf), so the two sum to one.
                float3 neeContribution = bsdfSample.weight * envColor * misWeight;
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
                                                  /*invPdf=*/1.0f),
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
                                                spawn_ray_min(material, kEpsilon),
                                                kInfinity,
                                                /*anyHitOnly=*/false,
                                                /*includeTriangles=*/true,
                                                lightRec);
            } else {
#endif
                hitLight = trace_scene_hardware_alpha_aware_shadow<false>(uniforms,
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
                                                                          spawn_ray_min(material, kEpsilon),
                                                                          kInfinity,
                                                                          neeExcludeMesh,
                                                                          neeExcludePrim,
                                                                          materials,
                                                                          materialTextures,
                                                                          materialSamplers,
                                                                          materialTextureInfos,
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
                                                       spawn_ray_min(material, kEpsilon),
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
                    float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                    float denom = lightPdf + bsdfPdf;
                    float misWeight = (denom > 0.0f) ? (lightPdf / denom) : 0.0f;
                    misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                    // Delta direction, not a light sample: MIS share only, no 1/lightPdf
                    // (the continuation's emitter hit takes the complement).
                    float3 contribution = bsdfSample.weight * mneeHit.emission * misWeight;
                    contribution *= volume_surface_nee_transmittance(uniforms, distance(lightRec.point, nextOrigin), false, stats);
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
                                                      /*invPdf=*/1.0f),
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
                                            spawn_ray_min(material, kEpsilon),
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
                                                spawn_ray_min(material, kEpsilon),
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
                                                       spawn_ray_min(material, kEpsilon),
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
                                                       spawn_ray_min(material, kEpsilon),
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
                occluded = trace_scene_hardware_alpha_aware_shadow<false>(uniforms,
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
                                                                   spawn_ray_min(material, kEpsilon),
                                                                   kInfinity,
                                                                   neeExcludeMesh,
                                                                   neeExcludePrim,
                                                                   materials,
                                                                   materialTextures,
                                                                   materialSamplers,
                                                                   materialTextureInfos,
                                                                   shadowRec, false, depth);
#endif
#if PT_DEBUG_TOOLS
            }
#endif
#endif
            if (!occluded) {
                float envPdf = environment_pdf(uniforms, environmentPdf, mneeRay.direction);
                envPdf = max(envPdf, kSpecularNeePdfFloor);
                float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                float denom = envPdf + bsdfPdf;
                float misWeight = (denom > 0.0f) ? (envPdf / denom) : 0.0f;
                misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                float3 envColor = environment_color(environmentTexture,
                                                    mneeRay.direction,
                                                    uniforms.environmentRotation,
                                                    uniforms.environmentIntensity,
                                                    uniforms);
                // Delta direction, not an environment sample: MIS share only, no
                // 1/envPdf (the continuation's environment miss takes the complement).
                float3 neeContribution = bsdfSample.weight * envColor * misWeight;
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
                                            spawn_ray_min(material, kEpsilon),
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
                                                spawn_ray_min(material, kEpsilon),
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
                                                       spawn_ray_min(material, kEpsilon),
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
                                                       spawn_ray_min(material, kEpsilon),
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
                hitLight = trace_scene_hardware_alpha_aware_shadow<false>(uniforms,
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
                                                                          spawn_ray_min(material, kEpsilon),
                                                                          kInfinity,
                                                                          neeExcludeMesh,
                                                                          neeExcludePrim,
                                                                          materials,
                                                                          materialTextures,
                                                                          materialSamplers,
                                                                          materialTextureInfos,
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
                float bsdfPdf = max(bsdfSample.directionalPdf, kSpecularNeePdfFloor);
                float denom = lightPdf + bsdfPdf;
                float misWeight = (denom > 0.0f) ? (lightPdf / denom) : 0.0f;
                misWeight = clamp(misWeight, kMisWeightClampMin, kMisWeightClampMax);
                // Delta direction, not a light sample: MIS share only, no 1/lightPdf
                // (the continuation's emitter hit takes the complement).
                float3 contribution = bsdfSample.weight * mneeHit.emission * misWeight;
                contribution *= volume_surface_nee_transmittance(uniforms, distance(lightRec.point, nextOrigin), false, stats);
                if (all(isfinite(contribution))) {
                    radiance += clamp_firefly_contribution(throughput, contribution, clampParams);
                    if (stats) {
                        atomic_fetch_add_explicit(&stats->mneeRectAddedCount, 1u, memory_order_relaxed);
                        stats_add_mnee_luma(stats, contribution);
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

        rayCone.width = ray_cone_width_at_distance(rayCone, hitDistanceWorld);
        rayCone.spread = min(rayCone.spread +
                             bsdf_cone_spread_increment(bsdfSample.lobeType,
                                                        bsdfSample.lobeRoughness,
                                                        bsdfSample.isDelta),
                             1.5f);

        lastBsdfPdf = (bsdfSample.directionalPdf > 0.0f) ? bsdfSample.directionalPdf : bsdfSample.pdf;
        lastScatterWasVolume = false;
        // This vertex's fused NEE sampled the mesh emitters (baseline draw or a
        // reservoir) and could have produced the scattered direction.
        previousMeshNeeEligible = !explicitRestirDiLightingStage &&
            mesh_emitter_nee_mode(uniforms) &&
            uniforms.emissivePrimitiveCount > 0u && !surfaceIsDelta &&
            (!bsdfSample.isBssrdf || subsurfaceExitVertex) &&
            mesh_emitter_nee_covers_direction(uniforms, material, rec, shadingNormal, bsdfSample.direction);
        previousMeshNeeOrigin = rec.point;
        lastScatterWasDelta = bsdfSample.isDelta;
        // Without a specular connection along this delta direction, the next
        // emitter hit or environment miss is the only estimate for the path.
        lastDeltaUnpaired = bsdfSample.isDelta && !(specNeeEligible || mneeEligible);
        if (pavilionSmoothDifferentials) {
            pavilionSmoothDifferentials = propagate_pavilion_planar_delta_differentials(
                primaryRayDiff, ray, rec, material, bsdfSample);
        }
        ray.origin = nextOrigin;
        holeMisOriginValid = false;
        ray.direction = bsdfSample.direction;
        primaryVisibilityRay = false;

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
    if (spectralState.enabled) {
        // The path carried per-wavelength radiance; apply the sensor and white balance once.
        radiance = spectral_channels_to_rgb(radiance, spectralState, uniforms);
        if (outDirectRadiance) {
            *outDirectRadiance = spectral_channels_to_rgb(*outDirectRadiance, spectralState, uniforms);
        }
        if (outIndirectRadiance) {
            *outIndirectRadiance = spectral_channels_to_rgb(*outIndirectRadiance, spectralState, uniforms);
        }
    }
    radiance_cache_training_finish(cacheTraining, radianceCacheStates, uniforms, radiance, stats);
    return radiance;
}
#endif
