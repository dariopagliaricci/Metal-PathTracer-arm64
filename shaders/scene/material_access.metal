                                                device const SphereData* spheres,
                                                device const RectData* rectangles,
                                                device const TriangleData* triangles,
                                                device const BvhNode* tlasNodes,
                                                device const uint* tlasPrimIndices,
                                                device const SoftwareInstanceInfo* instanceInfos,
                                                device const BvhNode* blasNodes,
                                                device const uint* blasPrimIndices,
                                                device const BvhNode* nodes,
                                                device const uint* primitiveIndices,
                                                device PathtraceStats* stats,
                                                thread const Ray& ray,
                                                float tMin,
                                                float tMax,
                                                uint excludeMeshIndex,
                                                uint excludePrimitiveIndex,
                                                thread HitRecord& rec) {
    if (excludeMeshIndex == kInvalidIndex || excludePrimitiveIndex == kInvalidIndex) {
        return trace_scene_software(uniforms,
                                    spheres,
                                    rectangles,
                                    triangles,
                                    tlasNodes,
                                    tlasPrimIndices,
                                    instanceInfos,
                                    blasNodes,
                                    blasPrimIndices,
                                    nodes,
                                    primitiveIndices,
                                    stats,
                                    ray,
                                    tMin,
                                    tMax,
                                    /*anyHitOnly=*/false,
                                    /*includeTriangles=*/true,
                                    rec);
    }

    const uint kMaxAttempts = 4u;
    uint maxAttempts = min(max(uniforms.hardwareExcludeMaxAttempts, 1u), kMaxAttempts);
    Ray currentRay = ray;
    float currentTMin = tMin;
    for (uint attempt = 0u; attempt < maxAttempts; ++attempt) {
        HitRecord tempRec = make_empty_hit_record();
        device PathtraceStats* attemptStats = (attempt == 0u) ? stats : nullptr;
        if (!trace_scene_software(uniforms,
                                  spheres,
                                  rectangles,
                                  triangles,
                                  tlasNodes,
                                  tlasPrimIndices,
                                  instanceInfos,
                                  blasNodes,
                                  blasPrimIndices,
                                  nodes,
                                  primitiveIndices,
                                  attemptStats,
                                  currentRay,
                                  currentTMin,
                                  tMax,
                                  /*anyHitOnly=*/false,
                                  /*includeTriangles=*/true,
                                  tempRec)) {
            return false;
        }

        bool excluded = (tempRec.primitiveType == kPrimitiveTypeTriangle &&
                         tempRec.meshIndex == excludeMeshIndex &&
                         tempRec.primitiveIndex == excludePrimitiveIndex);
        if (!excluded) {
            rec = tempRec;
            return true;
        }

        float advance = max(tempRec.t + kHardwareOcclusionEpsilon, kHardwareOcclusionEpsilon);
        currentRay.origin = ray_at(currentRay, advance);
        currentRay.origin += currentRay.direction * kHardwareOcclusionEpsilon;
        currentTMin = kHardwareOcclusionEpsilon;
    }

    return false;
}

inline float clamp01(const float value) {
    return clamp(value, 0.0f, 1.0f);
}

inline float3 clamp01(const float3 value) {
    return clamp(value, float3(0.0f), float3(1.0f));
}

inline float3 camera_visible_emission(const MaterialData material, const uint depth) {
    if (depth == 0u &&
        (material.materialFlags & kMaterialFlagCameraVisibleEmissionTint) != 0u &&
        any(material.emission.xyz != float3(0.0f))) {
        float3 tint = clamp(material.baseColorRoughness.xyz, float3(0.0f), float3(1.0f));
        return tint * 16.0f;
    }
    return material.emission.xyz;
}

inline bool material_texture_valid(constant PathtraceUniforms& uniforms, uint index) {
    const uint compiledTextureCount = min(uniforms.materialTextureCount, kMaxMaterialTextures);
    return index != kInvalidIndex && index < compiledTextureCount;
}

inline MaterialTextureInfo material_texture_info(
    array<texture2d<float, access::sample>, kMaxMaterialTextures> textures,
    device const MaterialTextureInfo* textureInfos,
    constant PathtraceUniforms& uniforms,
    uint index) {
    MaterialTextureInfo info{};
    if (!material_texture_valid(uniforms, index)) {
        return info;
    }
    if (textureInfos) {
        info = textureInfos[index];
    }
    if (info.width == 0u) {
        info.width = textures[index].get_width();
    }
    if (info.height == 0u) {
        info.height = textures[index].get_height();
    }
    if (info.mipCount == 0u) {
        info.mipCount = textures[index].get_num_mip_levels();
    }
    return info;
}

constant uint kPbrTextureSlotBaseColor = 0u;
constant uint kPbrTextureSlotMetallicRoughness = 1u;
constant uint kPbrTextureSlotNormal = 2u;
constant uint kPbrTextureSlotOcclusion = 3u;
constant uint kPbrTextureSlotEmissive = 4u;
constant uint kPbrTextureSlotTransmission = 5u;

inline uint pbr_texture_uv_set(const MaterialData material, const uint slot) {
    switch (slot) {
        case kPbrTextureSlotBaseColor:
            return min(material.textureUvSet0.x, 1u);
        case kPbrTextureSlotMetallicRoughness:
            return min(material.textureUvSet0.y, 1u);
        case kPbrTextureSlotNormal:
            return min(material.textureUvSet0.z, 1u);
        case kPbrTextureSlotOcclusion:
            return min(material.textureUvSet0.w, 1u);
        case kPbrTextureSlotEmissive:
            return min(material.textureUvSet1.x, 1u);
        case kPbrTextureSlotTransmission:
            return min(material.textureUvSet1.y, 1u);
        default:
            return 0u;
    }
}

inline void pbr_texture_transform_rows(const MaterialData material,
                                       const uint slot,
                                       thread float3& row0,
                                       thread float3& row1) {
    switch (slot) {
        case kPbrTextureSlotBaseColor:
            row0 = material.textureTransform0.xyz;
            row1 = material.textureTransform1.xyz;
            break;
        case kPbrTextureSlotMetallicRoughness:
            row0 = material.textureTransform2.xyz;
            row1 = material.textureTransform3.xyz;
            break;
        case kPbrTextureSlotNormal:
            row0 = material.textureTransform4.xyz;
            row1 = material.textureTransform5.xyz;
            break;
        case kPbrTextureSlotOcclusion:
            row0 = material.textureTransform6.xyz;
            row1 = material.textureTransform7.xyz;
            break;
        case kPbrTextureSlotEmissive:
            row0 = material.textureTransform8.xyz;
            row1 = material.textureTransform9.xyz;
            break;
        case kPbrTextureSlotTransmission:
            row0 = material.textureTransform10.xyz;
            row1 = material.textureTransform11.xyz;
            break;
        default:
            row0 = float3(1.0f, 0.0f, 0.0f);
            row1 = float3(0.0f, 1.0f, 0.0f);
            break;
    }

    float linearSum = fabs(row0.x) + fabs(row0.y) + fabs(row1.x) + fabs(row1.y);
    if (!all(isfinite(row0)) || !all(isfinite(row1)) || !(linearSum > 1.0e-8f)) {
        row0 = float3(1.0f, 0.0f, 0.0f);
        row1 = float3(0.0f, 1.0f, 0.0f);
    }
}

inline float2 pbr_apply_uv_transform(const float2 uv,
                                     const float3 row0,
                                     const float3 row1) {
    return float2(dot(row0.xy, uv) + row0.z,
                  dot(row1.xy, uv) + row1.z);
}

inline float2 pbr_transform_uv_gradient(const float2 grad,
                                        const float3 row0,
                                        const float3 row1) {
    return float2(dot(row0.xy, grad),
                  dot(row1.xy, grad));
}

inline float pbr_transform_uv_per_world(const float uvPerWorld,
                                        const float3 row0,
                                        const float3 row1) {
    float sx = length(float2(row0.x, row1.x));
    float sy = length(float2(row0.y, row1.y));
    float scaleBound = max(max(sx, sy), 1.0e-6f);
    return uvPerWorld * scaleBound;
}

struct PbrTextureSamplingContext {
    float2 uv = float2(0.0f);
    bool hasIgehyGradients = false;
    float2 dUVdx = float2(0.0f);
    float2 dUVdy = float2(0.0f);
    float uvPerWorld = 0.0f;
};

inline PbrTextureSamplingContext make_pbr_texture_sampling_context(const MaterialData material,
                                                                   const uint slot,
                                                                   const float2 uv0,
                                                                   const float2 uv1,
                                                                   const bool hasGradients0,
                                                                   const float2 dUVdx0,
                                                                   const float2 dUVdy0,
                                                                   const float uvPerWorld0,
                                                                   const bool hasGradients1,
                                                                   const float2 dUVdx1,
                                                                   const float2 dUVdy1,
                                                                   const float uvPerWorld1) {
    PbrTextureSamplingContext ctx;
    uint uvSet = pbr_texture_uv_set(material, slot);
    float2 baseUv = (uvSet == 0u) ? uv0 : uv1;
    bool baseHasGradients = (uvSet == 0u) ? hasGradients0 : hasGradients1;
    float2 baseDx = (uvSet == 0u) ? dUVdx0 : dUVdx1;
    float2 baseDy = (uvSet == 0u) ? dUVdy0 : dUVdy1;
    float baseUvPerWorld = (uvSet == 0u) ? uvPerWorld0 : uvPerWorld1;

    float3 row0 = float3(1.0f, 0.0f, 0.0f);
    float3 row1 = float3(0.0f, 1.0f, 0.0f);
    pbr_texture_transform_rows(material, slot, row0, row1);
    ctx.uv = pbr_apply_uv_transform(baseUv, row0, row1);
    ctx.hasIgehyGradients = baseHasGradients;
    if (ctx.hasIgehyGradients) {
        ctx.dUVdx = pbr_transform_uv_gradient(baseDx, row0, row1);
        ctx.dUVdy = pbr_transform_uv_gradient(baseDy, row0, row1);
        if (!all(isfinite(ctx.dUVdx)) || !all(isfinite(ctx.dUVdy))) {
            ctx.hasIgehyGradients = false;
            ctx.dUVdx = float2(0.0f);
            ctx.dUVdy = float2(0.0f);
        }
    }
    ctx.uvPerWorld = pbr_transform_uv_per_world(baseUvPerWorld, row0, row1);
    return ctx;
}

inline float4 sample_material_texture(
    array<texture2d<float, access::sample>, kMaxMaterialTextures> textures,
    array<sampler, kMaxMaterialSamplers> materialSamplers,
    device const MaterialTextureInfo* textureInfos,
    constant PathtraceUniforms& uniforms,
    uint index,
    float2 uv,
    float4 fallback) {
    if (!material_texture_valid(uniforms, index)) {
        return fallback;
    }
    MaterialTextureInfo info = material_texture_info(textures, textureInfos, uniforms, index);
    uint samplerIndex = min(info.flags, kMaxMaterialSamplers - 1u);
    return textures[index].sample(materialSamplers[samplerIndex], uv);
}

inline float4 sample_material_texture_level(
    array<texture2d<float, access::sample>, kMaxMaterialTextures> textures,
    array<sampler, kMaxMaterialSamplers> materialSamplers,
    device const MaterialTextureInfo* textureInfos,
    constant PathtraceUniforms& uniforms,
    uint index,
    float2 uv,
    float4 fallback,
    float lod) {
    if (!material_texture_valid(uniforms, index)) {
        return fallback;
    }
    MaterialTextureInfo info = material_texture_info(textures, textureInfos, uniforms, index);
    uint samplerIndex = min(info.flags, kMaxMaterialSamplers - 1u);
    if (info.mipCount <= 1u) {
        return textures[index].sample(materialSamplers[samplerIndex], uv);
    }
    float maxMip = float(info.mipCount - 1u);
    float clampedLod = clamp(lod, 0.0f, maxMip);
    return textures[index].sample(materialSamplers[samplerIndex], uv, level(clampedLod));
}

inline float4 sample_material_texture_filtered(
    array<texture2d<float, access::sample>, kMaxMaterialTextures> textures,
    array<sampler, kMaxMaterialSamplers> materialSamplers,
    device const MaterialTextureInfo* textureInfos,
    constant PathtraceUniforms& uniforms,
    uint index,
    float2 uv,
    float4 fallback,
    float lod,
    bool hasIgehyGradients,
    float2 dUVdx,
    float2 dUVdy) {
    if (!material_texture_valid(uniforms, index)) {
        return fallback;
    }
    MaterialTextureInfo info = material_texture_info(textures, textureInfos, uniforms, index);
    uint samplerIndex = min(info.flags, kMaxMaterialSamplers - 1u);

    if (hasIgehyGradients &&
        all(isfinite(dUVdx)) &&
        all(isfinite(dUVdy))) {
        float gradMag = max(max(fabs(dUVdx.x), fabs(dUVdx.y)),
                            max(fabs(dUVdy.x), fabs(dUVdy.y)));
        if (gradMag > 0.0f && isfinite(gradMag)) {
            return textures[index].sample(materialSamplers[samplerIndex],
                                          uv,
                                          gradient2d(dUVdx, dUVdy));
        }
    }

    if (info.mipCount <= 1u) {
        return textures[index].sample(materialSamplers[samplerIndex], uv);
    }
    float maxMip = float(info.mipCount - 1u);
    float clampedLod = clamp(lod, 0.0f, maxMip);
    return textures[index].sample(materialSamplers[samplerIndex], uv, level(clampedLod));
}

inline float material_texture_lod_from_cone(
    array<texture2d<float, access::sample>, kMaxMaterialTextures> textures,
    device const MaterialTextureInfo* textureInfos,
    constant PathtraceUniforms& uniforms,
    uint index,
    float uvPerWorld,
    float footprintWorld) {
    if (!material_texture_valid(uniforms, index)) {
        return 0.0f;
    }
    MaterialTextureInfo info = material_texture_info(textures, textureInfos, uniforms, index);
    return ray_cone_lod_from_footprint(info, uvPerWorld, footprintWorld);
}

inline bool material_texture_lod_from_gradients(
    array<texture2d<float, access::sample>, kMaxMaterialTextures> textures,
    device const MaterialTextureInfo* textureInfos,
    constant PathtraceUniforms& uniforms,
    uint index,
    float2 dUVdx,
    float2 dUVdy,
    thread float& outLod) {
    outLod = 0.0f;
    if (!material_texture_valid(uniforms, index)) {
        return false;
    }
    if (!all(isfinite(dUVdx)) || !all(isfinite(dUVdy))) {
        return false;
    }
    MaterialTextureInfo info = material_texture_info(textures, textureInfos, uniforms, index);
    if (info.width == 0u || info.height == 0u || info.mipCount <= 1u) {
        return false;
    }
    float W = float(info.width);
    float H = float(info.height);
    float rho = max(max(fabs(dUVdx.x) * W, fabs(dUVdx.y) * H),
                    max(fabs(dUVdy.x) * W, fabs(dUVdy.y) * H));
    if (!isfinite(rho) || !(rho > 0.0f)) {
        return false;
    }
    float lod = log2(max(rho, 1.0e-8f));
    if (!isfinite(lod)) {
        return false;
    }
    float maxMip = float(info.mipCount - 1u);
    outLod = clamp(lod, 0.0f, maxMip);
    return isfinite(outLod);
}

inline float material_texture_lod_with_fallback(
    array<texture2d<float, access::sample>, kMaxMaterialTextures> textures,
    device const MaterialTextureInfo* textureInfos,
    constant PathtraceUniforms& uniforms,
    uint index,
    bool hasIgehyGradients,
    float2 dUVdx,
    float2 dUVdy,
    float uvPerWorld,
    float surfaceFootprintWorld) {
    float lod = 0.0f;
    bool gradientOk = false;
    if (hasIgehyGradients) {
        gradientOk = material_texture_lod_from_gradients(textures,
                                                         textureInfos,
                                                         uniforms,
                                                         index,
                                                         dUVdx,
                                                         dUVdy,
                                                         lod);
    }
    if (!gradientOk) {
        lod = material_texture_lod_from_cone(textures,
                                             textureInfos,
                                             uniforms,
                                             index,
                                             uvPerWorld,
                                             surfaceFootprintWorld);
    }
    if (!isfinite(lod)) {
        lod = material_texture_lod_from_cone(textures,
                                             textureInfos,
                                             uniforms,
                                             index,
                                             uvPerWorld,
                                             surfaceFootprintWorld);
    }
    return max(lod, 0.0f);
}

struct WavefrontSurfaceState {
    MaterialData material;
    HitRecord rec;
    float diffuseOcclusion;
    float3 firstHitAlbedo;
    float3 firstHitNormal;
    float3 debugBaseColor;
    float debugMetallic;
    float debugRoughness;
    float debugAO;
    float2 debugNormalUv;
    float debugTangentW;
    float3 debugVtxNormalRaw;
    float3 debugVtxNormal;
    float3 debugTangentRaw;
    float3 debugTangent;
    float3 debugBitangent;
    float3 debugTexelRaw;
    float3 debugTexelDecoded;
    float3 debugSnWorld;
    bool debugHasNormalMap;
    bool alphaDiscarded;
};

inline float3 material_base_color(const MaterialData material);
inline float ior_from_f0(float f0);

inline WavefrontSurfaceState resolve_wavefront_surface_state(
    constant PathtraceUniforms& uniforms,
    thread const HitRecord& inputRec,
    const MaterialData baseMaterial,
    const uint matIndex,
    const uint depth,
    const Ray ray,
    const PrimaryRayDiff primaryRayDiff,
    const float3 incidentDir,
    const float hitDistanceWorld,
    const RayCone rayCone,
    thread uint& state,
    const bool skipStochasticAlphaDiscard,
    array<texture2d<float, access::sample>, kMaxMaterialTextures> materialTextures,
    array<sampler, kMaxMaterialSamplers> materialSamplers,
    device const MaterialTextureInfo* materialTextureInfos,
    device const MeshInfo* meshInfos,
    device const SceneVertex* sceneVertices,
    device const uint3* meshIndices) {
    WavefrontSurfaceState result{};
    HitRecord rec = inputRec;
    MaterialData material = baseMaterial;
    result.diffuseOcclusion = 1.0f;
    result.firstHitAlbedo = material_base_color(baseMaterial);
    result.firstHitNormal = inputRec.shadingNormal;
    result.debugBaseColor = material_base_color(baseMaterial);
    result.debugMetallic = 0.0f;
    result.debugRoughness = clamp(baseMaterial.baseColorRoughness.w, 0.0f, 1.0f);
    result.debugAO = 1.0f;
    result.debugNormalUv = float2(0.0f);
    result.debugTangentW = 0.0f;
    result.debugVtxNormalRaw = float3(0.0f);
    result.debugVtxNormal = float3(0.0f);
    result.debugTangentRaw = float3(0.0f);
    result.debugTangent = float3(0.0f);
    result.debugBitangent = float3(0.0f);
    result.debugTexelRaw = float3(0.5f, 0.5f, 1.0f);
    result.debugTexelDecoded = float3(0.0f, 0.0f, 1.0f);
    result.debugSnWorld = inputRec.shadingNormal;
    result.debugHasNormalMap = false;
    result.alphaDiscarded = false;
    uint type = static_cast<uint>(material.typeEta.x);

    float3 shadingNormal = rec.shadingNormal;
    if (!all(isfinite(shadingNormal)) || dot(shadingNormal, shadingNormal) <= 0.0f) {
        shadingNormal = rec.normal;
    }
    if (rec.primitiveType == kPrimitiveTypeTriangle &&
        meshInfos && sceneVertices && meshIndices && uniforms.meshCount > 0u) {
        float3 candidate = interpolate_shading_normal(uniforms,
                                                      rec.meshIndex,
                                                      rec.primitiveIndex,
                                                      rec.barycentric,
                                                      meshInfos,
                                                      sceneVertices,
                                                      meshIndices);
        if (all(isfinite(candidate)) && dot(candidate, candidate) > 0.0f) {
            result.debugVtxNormalRaw = candidate;
            if (dot(candidate, rec.normal) < 0.0f) {
                candidate = -candidate;
            }
            shadingNormal = normalize(candidate);
            result.debugVtxNormal = shadingNormal;
        }
    }
    if (type == 2u) {
        float3 geomNormal = rec.normal;
        if (all(isfinite(geomNormal)) && dot(geomNormal, geomNormal) > 0.0f) {
            shadingNormal = geomNormal;
        }
    }
    rec.shadingNormal = shadingNormal;

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
        result.debugNormalUv = uv0;
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
        float coneFootprintWorld = ray_cone_width_at_distance(rayCone, hitDistanceWorld);
        float surfaceFootprintWorld =
            surface_footprint_from_cone(coneFootprintWorld, rec.normal, -incidentDir);
        PbrTextureSamplingContext baseColorCtx = make_pbr_texture_sampling_context(material,
                                                                                   kPbrTextureSlotBaseColor,
                                                                                   uv0,
                                                                                   uv1,
                                                                                   hasIgehyGradients0,
                                                                                   dUVdx0,
                                                                                   dUVdy0,
                                                                                   hasSurfacePartials0 ? uvPerWorld0 : 0.0f,
                                                                                   hasIgehyGradients1,
                                                                                   dUVdx1,
                                                                                   dUVdy1,
                                                                                   hasSurfacePartials1 ? uvPerWorld1 : 0.0f);
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
        float3 baseFactor = to_working_space(material.baseColorRoughness.xyz, uniforms);
        float3 baseColorSampleRgb = to_working_space(baseColorSample.xyz, uniforms);
        material.baseColorRoughness.xyz = baseFactor * baseColorSampleRgb;
        result.debugBaseColor = material.baseColorRoughness.xyz;
    }

    if (type == 7u &&
        rec.primitiveType == kPrimitiveTypeTriangle &&
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
        result.debugNormalUv = uv0;
        float4 tangent = interpolate_tangent(uniforms,
                                             rec.meshIndex,
                                             rec.primitiveIndex,
                                             rec.barycentric,
                                             meshInfos,
                                             sceneVertices,
                                             meshIndices);
        result.debugTangentRaw = tangent.xyz;
        result.debugTangentW = tangent.w;

        if (material.typeEta.z > 0.5f) {
            rec.twoSided = 1u;
        }

        float coneFootprintWorld = ray_cone_width_at_distance(rayCone, hitDistanceWorld);
        float surfaceFootprintWorld =
            surface_footprint_from_cone(coneFootprintWorld, rec.normal, -incidentDir);
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
                                                                                   hasSurfacePartials0 ? uvPerWorld0 : 0.0f,
                                                                                   hasIgehyGradients1,
                                                                                   dUVdx1,
                                                                                   dUVdy1,
                                                                                   hasSurfacePartials1 ? uvPerWorld1 : 0.0f);
        PbrTextureSamplingContext ormCtx = make_pbr_texture_sampling_context(material,
                                                                             kPbrTextureSlotMetallicRoughness,
                                                                             uv0,
                                                                             uv1,
                                                                             hasIgehyGradients0,
                                                                             dUVdx0,
                                                                             dUVdy0,
                                                                             hasSurfacePartials0 ? uvPerWorld0 : 0.0f,
                                                                             hasIgehyGradients1,
                                                                             dUVdx1,
                                                                             dUVdy1,
                                                                             hasSurfacePartials1 ? uvPerWorld1 : 0.0f);
        PbrTextureSamplingContext normalCtx = make_pbr_texture_sampling_context(material,
                                                                                kPbrTextureSlotNormal,
                                                                                uv0,
                                                                                uv1,
                                                                                hasIgehyGradients0,
                                                                                dUVdx0,
                                                                                dUVdy0,
                                                                                hasSurfacePartials0 ? uvPerWorld0 : 0.0f,
                                                                                hasIgehyGradients1,
                                                                                dUVdx1,
                                                                                dUVdy1,
                                                                                hasSurfacePartials1 ? uvPerWorld1 : 0.0f);
        PbrTextureSamplingContext occlusionCtx = make_pbr_texture_sampling_context(material,
                                                                                   kPbrTextureSlotOcclusion,
                                                                                   uv0,
                                                                                   uv1,
                                                                                   hasIgehyGradients0,
                                                                                   dUVdx0,
                                                                                   dUVdy0,
                                                                                   hasSurfacePartials0 ? uvPerWorld0 : 0.0f,
                                                                                   hasIgehyGradients1,
                                                                                   dUVdx1,
                                                                                   dUVdy1,
                                                                                   hasSurfacePartials1 ? uvPerWorld1 : 0.0f);
        PbrTextureSamplingContext emissiveCtx = make_pbr_texture_sampling_context(material,
                                                                                  kPbrTextureSlotEmissive,
                                                                                  uv0,
                                                                                  uv1,
                                                                                  hasIgehyGradients0,
                                                                                  dUVdx0,
                                                                                  dUVdy0,
                                                                                  hasSurfacePartials0 ? uvPerWorld0 : 0.0f,
                                                                                  hasIgehyGradients1,
                                                                                  dUVdx1,
                                                                                  dUVdy1,
                                                                                  hasSurfacePartials1 ? uvPerWorld1 : 0.0f);
        PbrTextureSamplingContext transmissionCtx = make_pbr_texture_sampling_context(material,
                                                                                      kPbrTextureSlotTransmission,
                                                                                      uv0,
                                                                                      uv1,
                                                                                      hasIgehyGradients0,
                                                                                      dUVdx0,
                                                                                      dUVdy0,
                                                                                      hasSurfacePartials0 ? uvPerWorld0 : 0.0f,
                                                                                      hasIgehyGradients1,
                                                                                      dUVdx1,
                                                                                      dUVdy1,
                                                                                      hasSurfacePartials1 ? uvPerWorld1 : 0.0f);

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
            float luma = dot(baseColorSample.xyz, kLuminanceWeights);
            baseColorSample.w = smoothstep(0.72f, 0.90f, luma);
        }
        float3 baseColorSampleRgb = to_working_space(baseColorSample.xyz, uniforms);
        float3 baseColor = baseFactor * baseColorSampleRgb;

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
            } else if (skipStochasticAlphaDiscard) {
                discard = false;
            } else {
                discard = rand_uniform(state) > alpha;
            }
            if (discard) {
                result.alphaDiscarded = true;
                return result;
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
        result.diffuseOcclusion = (uniforms.debugDisableAO != 0u) ? 1.0f : occlusion;
        if (uniforms.debugAoIndirectOnly != 0u && depth == 0u) {
            result.diffuseOcclusion = 1.0f;
        }
        result.debugAO = occlusion;
        result.debugBaseColor = baseColor;
        result.debugMetallic = metallic;

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
            emissive *= to_working_space(emissiveSample, uniforms);
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
        bool flipNormalGreen = false;
#if PT_DEBUG_TOOLS
        flipNormalGreen = uniforms.debugFlipNormalGreen != 0u;
#endif
        if (useNormalMap) {
            result.debugHasNormalMap = true;
            float3 normalTexel =
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
            result.debugTexelRaw = normalTexel;
            float3 normalSampleTs = decode_normal_map(normalTexel,
                                                      normalScale,
                                                      flipNormalGreen,
                                                      normalLength);
            result.debugTexelDecoded = normalSampleTs;
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
            result.debugTangent = t;
            result.debugBitangent = b;
            shadingNormal = mapped;
            result.debugSnWorld = shadingNormal;

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
        material.pbrParams.w = normalScale;
        material.pbrParams.x = metallic;
        material.emission = float4(emissive, 0.0f);
        rec.shadingNormal = shadingNormal;
        result.debugRoughness = roughness;
    }

    result.material = material;
    result.rec = rec;
    result.firstHitAlbedo = material_base_color(material);
    result.firstHitNormal = rec.shadingNormal;
    return result;
}
