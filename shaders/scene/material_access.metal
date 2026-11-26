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
    // How far along the ray the origin has moved past rejected hits. The returned t and the reach stay in the
    // original ray's frame, as Embree's IntersectSceneExcludingShaded keeps them: callers read rec.t as the
    // distance from the ray they traced (texture footprint, medium segments, shadow distances).
    float tBase = 0.0f;
    for (uint attempt = 0u; attempt < maxAttempts; ++attempt) {
        if (attempt > 0u && !(tMax - tBase > currentTMin)) {
            return false;  // the re-spawn passed the segment's end
        }
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
                                  tMax - tBase,
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
            rec.t += tBase;
            return true;
        }

        if (is_interface_min_t(uniforms, tMin)) {
            currentTMin = nextafter(max(tempRec.t, currentTMin), INFINITY);
            continue;
        }
        // Resume kHardwareOcclusionEpsilon past the excluded hit with tMin 0, as trace_scene_hardware does after a
        // rejected hit. The old resume (t + 5 mm, another 5 mm along the ray, then tMin 5 mm) skipped 15 mm past the
        // hit, so a grazing bounce ray that re-crosses its own triangle skipped thin geometry the hardware path still
        // reaches (Zero Day's light towers: SW/HW 0.957 on their tiles).
        float3 dirStep = float3(0.0f, 0.0f, 1.0f);
        const float dirLenSq = dot(currentRay.direction, currentRay.direction);
        if (all(isfinite(currentRay.direction)) && dirLenSq > 1.0e-12f) {
            dirStep = currentRay.direction * rsqrt(dirLenSq);
        }
        currentRay.origin = ray_at(currentRay, tempRec.t) + dirStep * kHardwareOcclusionEpsilon;
        currentTMin = 0.0f;
        tBase += tempRec.t + kHardwareOcclusionEpsilon * rsqrt(max(dirLenSq, 1.0e-12f));
    }

    return false;
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
    constant array<texture2d<float, access::sample>, kMaxMaterialTextures>& textures,
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

inline float4 sample_material_texture_level(
    constant array<texture2d<float, access::sample>, kMaxMaterialTextures>& textures,
    constant array<sampler, kMaxMaterialSamplers>& materialSamplers,
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
    constant array<texture2d<float, access::sample>, kMaxMaterialTextures>& textures,
    constant array<sampler, kMaxMaterialSamplers>& materialSamplers,
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
                                          gradient2d(dUVdx * exp2(uniforms.textureLodBias),
                                                     dUVdy * exp2(uniforms.textureLodBias)));
        }
    }

    if (info.mipCount <= 1u) {
        return textures[index].sample(materialSamplers[samplerIndex], uv);
    }
    float maxMip = float(info.mipCount - 1u);
    float clampedLod = clamp(lod, 0.0f, maxMip);
    return textures[index].sample(materialSamplers[samplerIndex], uv, level(clampedLod));
}

// The alpha a path ray's cutoff or BLEND coverage test reads: the hit point's own alpha at mip 0, as the
// shadow-ray test reads. A hit is a point, and the samples average the coverage. Filtered alpha at the
// path's cone LOD decides whole texture regions at once, so coarse levels turn MASK cutouts fully opaque
// or fully empty (barcelona_pavilion_day leaves: 42% coverage at mip 0, 100% at the 1x1 level). BLEND
// reads mip 0 too: its pass is a MIS null event whose light side NEE takes through the shadow test's mip-0
// alpha, so the two strategies must see the same alpha at a point. With a filtered path alpha above 0
// where mip 0 is 0, the pass and NEE both counted the light behind a BLEND hole. With mip 0 already read
// (no LOD, no gradients) there is no second fetch.
inline float path_cutout_alpha(constant array<texture2d<float, access::sample>, kMaxMaterialTextures>& textures,
                               constant array<sampler, kMaxMaterialSamplers>& materialSamplers,
                               device const MaterialTextureInfo* textureInfos,
                               constant PathtraceUniforms& uniforms,
                               const MaterialData material,
                               const PbrTextureSamplingContext baseColorCtx,
                               float baseColorLod,
                               float filteredAlpha) {
    const bool alphaTested = material.pbrExtras.w > 0.5f;
    if (!alphaTested || !material_texture_valid(uniforms, material.textureIndices0.x) ||
        (!(baseColorLod > 0.0f) && !baseColorCtx.hasIgehyGradients)) {
        return filteredAlpha;
    }
    const float4 texel = sample_material_texture_level(textures,
                                                       materialSamplers,
                                                       textureInfos,
                                                       uniforms,
                                                       material.textureIndices0.x,
                                                       baseColorCtx.uv,
                                                       float4(1.0f),
                                                       0.0f);
    if ((material.materialFlags & kMaterialFlagBlackKeyAlphaFromRgb) != 0u) {
        return smoothstep(0.72f, 0.90f, dot(texel.xyz, kLuminanceWeights));
    }
    return texel.w;
}

inline float material_texture_lod_from_cone(
    constant array<texture2d<float, access::sample>, kMaxMaterialTextures>& textures,
    device const MaterialTextureInfo* textureInfos,
    constant PathtraceUniforms& uniforms,
    uint index,
    float uvPerWorld,
    float footprintWorld) {
    if (!material_texture_valid(uniforms, index)) {
        return 0.0f;
    }
    MaterialTextureInfo info = material_texture_info(textures, textureInfos, uniforms, index);
    return clamp(ray_cone_lod_from_footprint(info, uvPerWorld, footprintWorld) + uniforms.textureLodBias,
                 0.0f, float(max(info.mipCount, 1u) - 1u));
}

inline bool material_texture_lod_from_gradients(
    constant array<texture2d<float, access::sample>, kMaxMaterialTextures>& textures,
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
    outLod = clamp(lod + uniforms.textureLodBias, 0.0f, maxMip);
    return isfinite(outLod);
}

inline float material_texture_lod_with_fallback(
    constant array<texture2d<float, access::sample>, kMaxMaterialTextures>& textures,
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
    HitRecord rec;
    float4 resolvedBaseColorRoughness;
    float4 resolvedTypeEta;
    float4 resolvedEmission;
    float4 resolvedPbrParams;
    float4 resolvedPbrExtras;
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
    float3 sourceForwardWeight;
};

inline float3 material_base_color(const MaterialData material);
inline float ior_from_f0(float f0);

inline bool triangle_has_valid_vertex_tangents(
    constant PathtraceUniforms& uniforms,
    uint meshIndex,
    uint primitiveIndex,
    device const MeshInfo* meshInfos,
    device const SceneVertex* vertices,
    device const uint3* meshIndices) {
    if (!meshInfos || !vertices || !meshIndices || uniforms.meshCount == 0u) {
        return false;
    }
    MeshInfo info = meshInfos[min(meshIndex, uniforms.meshCount - 1u)];
    if (primitiveIndex < info.triangleOffset ||
        primitiveIndex - info.triangleOffset >= info.indexCount) {
        return false;
    }
    uint3 indices = meshIndices[info.indexOffset + primitiveIndex - info.triangleOffset];
    float4 t0 = vertices[indices.x].tangent;
    float4 t1 = vertices[indices.y].tangent;
    float4 t2 = vertices[indices.z].tangent;
    // Interpolation historically replaces zero vectors/signs with defaults.
    // Check the stored vertex data so Emerald's source repair skips that fallback.
    return all(isfinite(t0)) && all(isfinite(t1)) && all(isfinite(t2)) &&
        fabs(t0.w) > 0.5f && fabs(t1.w) > 0.5f && fabs(t2.w) > 0.5f &&
        dot(t0.xyz, t0.xyz) > 1.0e-6f && dot(t1.xyz, t1.xyz) > 1.0e-6f &&
        dot(t2.xyz, t2.xyz) > 1.0e-6f;
}

inline bool source_is_thin(const MaterialData m);
inline float3 source_thin_forward(const MaterialData m, float3 n, float3 wo);

inline bool scene_has_source_opacity(constant PathtraceUniforms& uniforms,
                                    device const MaterialData* materials) {
    return materials && source_opacity_present(uniforms);
}

inline float3 source_shadow_weight(float3 value, thread const HitRecord& rec) {
    if (!kSourceFeatures) return value;
    value = rec.sourceShadowTransmission < 1.0f ? value * rec.sourceShadowTransmission : value;
    return any(rec.sourceShadowRgb < float3(1.0f)) ? value * rec.sourceShadowRgb : value;
}

// Tungsten scalar maps have independent dimensions and linear (raw byte) values.
inline float2 source_texture_uv(const MaterialData material, const float2 uv) {
    return float2(dot(material.textureTransform0.xy, uv) + material.textureTransform0.z,
                  dot(material.textureTransform1.xy, uv) + material.textureTransform1.z);
}

inline float source_opacity(constant PathtraceUniforms& uniforms,
    const MaterialData material, thread const HitRecord& rec,
    constant array<texture2d<float, access::sample>, kMaxMaterialTextures>& textures,
    constant array<sampler, kMaxMaterialSamplers>& samplers,
    device const MaterialTextureInfo* infos, device const MeshInfo* meshes,
    device const SceneVertex* vertices, device const uint3* indices) {
    if (!source_has_opacity(material)) return 1.0f;
    float alpha = material.textureControl.x;
    if (material_texture_valid(uniforms, material.textureIndices1.z) &&
        rec.primitiveType == kPrimitiveTypeTriangle && meshes && vertices && indices) {
        float2 uv = interpolate_uv(uniforms, rec.meshIndex, rec.primitiveIndex,
            rec.barycentric, 0u, meshes, vertices, indices);
        alpha *= sample_material_texture_level(textures, samplers, infos, uniforms,
            material.textureIndices1.z, source_texture_uv(material, uv), float4(1.0f), 0.0f).x;
    }
    return saturate(alpha);
}

inline float source_height_texel(texture2d<float, access::sample> texture, int2 p) {
    int2 size = int2(texture.get_width(), texture.get_height());
    return texture.read(uint2((p % size + size) % size), 0).x;
}

inline float2 source_height_derivatives(texture2d<float, access::sample> texture, float2 uv) {
    // BitmapTexture::derivatives: interpolate two-texel differences, with no
    // division by two or texel size. y is positive in bitmap row coordinates.
    float2 xy = uv * float2(texture.get_width(), texture.get_height()) - 0.5f;
    int2 p = int2(xy); // Tungsten truncates toward zero, including at the repeat seam.
    float2 f = xy - float2(p);
    float2 gradients[4];
    for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x) {
        int2 q = p + int2(x,y);
        gradients[y*2+x] = float2(
            source_height_texel(texture,q+int2(1,0))-source_height_texel(texture,q-int2(1,0)),
            source_height_texel(texture,q+int2(0,1))-source_height_texel(texture,q-int2(0,1)));
    }
    return mix(mix(gradients[0],gradients[1],f.x), mix(gradients[2],gradients[3],f.x),f.y);
}

inline float3 source_bump_normal(constant PathtraceUniforms& uniforms,
    const MaterialData material, thread const HitRecord& rec, float3 ns,
    constant array<texture2d<float, access::sample>, kMaxMaterialTextures>& textures,
    device const MeshInfo* meshes, device const SceneVertex* vertices, device const uint3* indices) {
    if (!kSourceFeatures || !material_texture_valid(uniforms, material.textureIndices1.w) ||
        material.textureControl.z == 0.0f || rec.primitiveType != kPrimitiveTypeTriangle ||
        !meshes || !vertices || !indices) return ns;
    // Tungsten perturbs the authored normal before facing the completed frame.
    const float3 facedNs = ns;
    const bool backFace = rec.frontFace == 0u;
    if (backFace) ns = -ns;
    MeshInfo mesh = meshes[min(rec.meshIndex, uniforms.meshCount - 1u)];
    uint local = rec.primitiveIndex - mesh.triangleOffset;
    if (local >= mesh.indexCount) return facedNs;
    uint3 tri = indices[mesh.indexOffset + local];
    SceneVertex v0 = vertices[tri.x], v1 = vertices[tri.y], v2 = vertices[tri.z];
    float3 q1 = (mesh.localToWorld * float4(v1.position.xyz-v0.position.xyz,0.0f)).xyz;
    float3 q2 = (mesh.localToWorld * float4(v2.position.xyz-v0.position.xyz,0.0f)).xyz;
    float2 uv0 = vertex_uv_set(v0,0u), uv1 = vertex_uv_set(v1,0u), uv2 = vertex_uv_set(v2,0u);
    float2 e1 = uv1-uv0, e2 = uv2-uv0;
    if (fabs(e1.x*e2.y-e2.x*e1.y) < 1.0e-6f) return facedNs;
    // TriangleMesh::tangentSpace normalizes the numerators, without dividing det.
    float3 t = normalize(q1*e2.y-e1.y*q2), b = normalize(q2*e1.x-e2.x*q1);
    float2 uv = interpolate_uv(uniforms,rec.meshIndex,rec.primitiveIndex,rec.barycentric,
                              0u,meshes,vertices,indices);
    float2 dh = source_height_derivatives(textures[material.textureIndices1.w],
                                          source_texture_uv(material,uv)) * material.textureControl.z;
    t += ns*(dh.x-dot(ns,t));
    b += ns*(dh.y-dot(ns,b));
    float3 n = cross(t,b);
    if (!all(isfinite(n)) || dot(n,n)<1.0e-16f) return facedNs;
    n = normalize(n);
    n = dot(n,ns)<0.0f ? -n : n;
    return backFace ? -n : n;
}

inline WavefrontSurfaceState resolve_wavefront_surface_state(
    constant PathtraceUniforms& uniforms,
    thread const HitRecord& inputRec,
    const MaterialData baseMaterial,
    const uint matIndex,
    const uint depth,
    const bool pavilionSmoothDifferentials,
    const Ray ray,
    const PrimaryRayDiff primaryRayDiff,
    const float3 incidentDir,
    const float hitDistanceWorld,
    const RayCone rayCone,
    thread uint& state,
    const bool skipStochasticAlphaDiscard,
    constant array<texture2d<float, access::sample>, kMaxMaterialTextures>& materialTextures,
    constant array<sampler, kMaxMaterialSamplers>& materialSamplers,
    device const MaterialTextureInfo* materialTextureInfos,
    device const MeshInfo* meshInfos,
    device const SceneVertex* sceneVertices,
    device const uint3* meshIndices) {
    WavefrontSurfaceState result{};
    HitRecord rec = inputRec;
    MaterialData material = baseMaterial;
    result.resolvedBaseColorRoughness = baseMaterial.baseColorRoughness;
    result.resolvedTypeEta = baseMaterial.typeEta;
    result.resolvedEmission = baseMaterial.emission;
    result.resolvedPbrParams = baseMaterial.pbrParams;
    result.resolvedPbrExtras = baseMaterial.pbrExtras;
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
    result.sourceForwardWeight = float3(1.0f);
    uint type = static_cast<uint>(material.typeEta.x);
    if (!skipStochasticAlphaDiscard && source_has_opacity(material) &&
        rand_uniform(state) >= source_opacity(uniforms,material,rec,materialTextures,
            materialSamplers,materialTextureInfos,meshInfos,sceneVertices,meshIndices)) {
        result.alphaDiscarded = true;
        return result;
    }

    float3 shadingNormal = rec.shadingNormal;
    bool authoredNormalWasFaced = false;
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
            if (dot(candidate, rec.normal) < 0.0f &&
                !source_material_flag(material, kMaterialFlagAuthoredDielectricNormal)) {
                authoredNormalWasFaced = true;
                candidate = -candidate;
            }
            shadingNormal = normalize(candidate);
            result.debugVtxNormal = shadingNormal;
        }
    }
    if (type == 2u && !source_material_flag(material, kMaterialFlagAuthoredDielectricNormal)) {
        float3 geomNormal = rec.normal;
        if (all(isfinite(geomNormal)) && dot(geomNormal, geomNormal) > 0.0f) {
            shadingNormal = geomNormal;
        }
    }
    if (rec.primitiveType == kPrimitiveTypeSphere &&
        source_material_flag(material, kMaterialFlagAuthoredDielectricNormal)) {
        // An analytic sphere's authored normal is its outward normal, as the megakernel uses (EP8-f3).
        shadingNormal = rec.frontFace != 0u ? rec.normal : -rec.normal;
    }
    shadingNormal = source_bump_normal(uniforms,material,rec,shadingNormal,
                                      materialTextures,meshInfos,sceneVertices,meshIndices);
    rec.shadingNormal = shadingNormal;
    if (!skipStochasticAlphaDiscard && source_transport_enabled(uniforms) && source_is_thin(material)) {
        float3 forward = source_thin_forward(material,shadingNormal,-incidentDir);
        float p = clamp(dot(forward,float3(1.0f/3.0f)),0.0f,1.0f);
        if (rand_uniform(state) < p) {
            result.alphaDiscarded = true;
            result.sourceForwardWeight = forward/max(p,1.0e-30f);
            result.rec = rec;
            return result;
        }
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
        // Pavilion textureGrad uses the interpolated normal's tangent plane,
        // including smooth lily geometry, as in the reference renderer.
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
        // Non-PBR base color is filtered only where ray differentials exist (the primary hit),
        // and sampled at mip 0 elsewhere, as in the megakernel and the Embree backend. The
        // ray-cone LOD reaches the coarsest mips after one diffuse bounce, and those levels do
        // not keep the texture mean for non-power-of-two sizes (bitterli_staircase: +4.7%).
        float baseColorLod = baseColorCtx.hasIgehyGradients
            ? material_texture_lod_with_fallback(materialTextures,
                                                 materialTextureInfos,
                                                 uniforms,
                                                 material.textureIndices0.x,
                                                 baseColorCtx.hasIgehyGradients,
                                                 baseColorCtx.dUVdx,
                                                 baseColorCtx.dUVdy,
                                                 baseColorCtx.uvPerWorld,
                                                 surfaceFootprintWorld)
            : 0.0f;
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
        baseColorSample = resolve_alpha_cutout_foliage_base_color_sample(material,
                                                                         baseColorSample);
        float3 baseFactor = material.baseColorRoughness.xyz;
        float3 baseColorSampleRgb = baseColorSample.xyz;
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

        float3 baseFactor = material.baseColorRoughness.xyz;
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
        baseColorSample = resolve_alpha_cutout_foliage_base_color_sample(material,
                                                                         baseColorSample);
        float3 baseColorSampleRgb = baseColorSample.xyz;
        float3 baseColor = baseFactor * baseColorSampleRgb;

        float metallic = clamp(material.pbrParams.x, 0.0f, 1.0f);
        // Scene multipliers may exceed one; preserve them until source G is applied.
        float roughness = max(material.pbrParams.y, 0.0f);
        float normalStrengthScale = 1.0f;
#if PT_DEBUG_TOOLS
        normalStrengthScale = max(PT_DEFAULT_VALUE(uniforms.debugNormalStrengthScale, 1.0f), 0.0f);
#endif
        float normalScale = material.pbrParams.w * normalStrengthScale;
        bool disableOrmByMaterial = (material.materialFlags & kMaterialFlagDisableOrm) != 0u;
        bool useOrmTexture = !disableOrmByMaterial &&
                             material_texture_valid(uniforms, material.textureIndices0.y);
#if PT_DEBUG_TOOLS
        useOrmTexture = useOrmTexture && (PT_DEFAULT_OFF(uniforms.debugDisableOrmTexture) == 0u);
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
            ormLod = max(ormLod + PT_DEFAULT_VALUE(uniforms.debugOrmLodBias, 0.0f), 0.0f);
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
        normalLod = max(normalLod + PT_DEFAULT_VALUE(uniforms.debugNormalLodBias, 0.0f), 0.0f);
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
            } else if (skipStochasticAlphaDiscard) {
                discard = false;
            } else {
                discard = rand_uniform(state) > alpha;
            }
            if (discard) {
                result.alphaDiscarded = true;
                result.resolvedBaseColorRoughness = material.baseColorRoughness;
                result.resolvedTypeEta = material.typeEta;
                result.resolvedEmission = material.emission;
                result.resolvedPbrParams = material.pbrParams;
                result.resolvedPbrExtras = material.pbrExtras;
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

        float3 emissive = material.emission.xyz;
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
            emissive *= emissiveSample;
        }

        bool useNormalMap = material_texture_valid(uniforms, material.textureIndices0.z);
#if PT_DEBUG_TOOLS
        if (PT_DEFAULT_OFF(uniforms.debugDisableNormalMap) != 0u) {
            useNormalMap = false;
        }
#endif
        if (normalScale <= 1.0e-4f) {
            useNormalMap = false;
        }
        bool flipNormalGreen = false;
#if PT_DEBUG_TOOLS
        flipNormalGreen = PT_DEFAULT_OFF(uniforms.debugFlipNormalGreen) != 0u;
#endif
        if (useNormalMap) {
            result.debugHasNormalMap = true;
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
            result.debugTexelRaw = normalTexel.xyz;
            float3 normalSampleTs = decode_normal_map(normalTexel.xyz,
                                                      normalScale,
                                                      flipNormalGreen);
            result.debugTexelDecoded = normalSampleTs;
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
                // No vertex tangent and no UV frame (degenerate UVs): no normal map, as on Embree and in Falcor.
                build_onb(shadingNormal, t, b);
                normalSampleTs = float3(0.0f, 0.0f, 1.0f);
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
                // at normal-map bevels. This flag is Picnic_Table1-only. V must be a
                // unit vector, as in Falcor; incidentDir is the traced direction, unit
                // length on every ray, and is normalized here again for safety.
                float nDotV = dot(mapped, -safe_normalize(incidentDir));
                if (nDotV <= 0.1f) {
                    mapped = normalize(mix(rec.normal, mapped, saturate(nDotV * 10.0f)));
                }
            }
            result.debugTangent = t;
            result.debugBitangent = b;
            shadingNormal = mapped;
            result.debugSnWorld = shadingNormal;

            float directionalVariance = normalTexel.w * normalTexel.w;
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
        material.pbrParams.w = normalScale;
        material.pbrParams.x = metallic;
        material.emission = float4(emissive, 0.0f);
        material = apply_foliage_beauty_lift(material);
        rec.shadingNormal = shadingNormal;
        result.debugRoughness = roughness;
    }

    result.rec = rec;
    result.resolvedBaseColorRoughness = material.baseColorRoughness;
    result.resolvedTypeEta = material.typeEta;
    result.resolvedEmission = material.emission;
    result.resolvedPbrParams = material.pbrParams;
    result.resolvedPbrExtras = material.pbrExtras;
    result.firstHitAlbedo = material_base_color(material);
    result.firstHitNormal = rec.shadingNormal;
    return result;
}

// Backlog #47: bends a mapped coat normal N toward the geometric normal Ng (facing the view direction I) so that the
// mirror reflection of I about it stays above the geometric surface, by at least min(0.9 * dot(Ng, I), 0.01). This is the
// rule of Cycles' ensure_valid_reflection (Blender, Apache-2.0), rewritten here. N is returned unchanged when its
// reflection is already above that threshold. Without the bend a strongly tilted coat normal reflects into the mesh: dark
// bands (-12% on the stripes probe) where Arnold's stay faint.
inline float3 subsurface_coat_valid_reflection_normal(const float3 Ng, const float3 I, const float3 N) {
    const float cosGeometric = dot(Ng, I);
    if (!(cosGeometric > 0.0f)) {
        return N;
    }
    const float3 R = 2.0f * dot(N, I) * N - I;
    const float threshold = min(0.9f * cosGeometric, 0.01f);
    if (dot(Ng, R) >= threshold) {
        return N;
    }
    // Frame with Ng as z and N in the x-z plane; solve for the normal whose reflection of I has height `threshold`.
    const float3 X = normalize(N - dot(N, Ng) * Ng);
    const float Ix = dot(I, X);
    const float Iz = cosGeometric;
    const float Ix2 = Ix * Ix;
    const float a = Ix2 + Iz * Iz;
    const float b = sqrt(max(Ix2 * (a - threshold * threshold), 0.0f));
    const float c = Iz * threshold + a;
    const float fac = 0.5f / a;
    const float n1z2 = fac * (b + c);
    const float n2z2 = fac * (c - b);
    bool valid1 = n1z2 > 1.0e-5f && n1z2 <= 1.0f + 1.0e-5f;
    bool valid2 = n2z2 > 1.0e-5f && n2z2 <= 1.0f + 1.0e-5f;
    float2 bent;
    if (valid1 && valid2) {
        const float2 n1 = float2(sqrt(max(1.0f - n1z2, 0.0f)), sqrt(max(n1z2, 0.0f)));
        const float2 n2 = float2(sqrt(max(1.0f - n2z2, 0.0f)), sqrt(max(n2z2, 0.0f)));
        const float r1 = 2.0f * (n1.x * Ix + n1.y * Iz) * n1.y - Iz;
        const float r2 = 2.0f * (n2.x * Ix + n2.y * Iz) * n2.y - Iz;
        valid1 = r1 >= 1.0e-5f;
        valid2 = r2 >= 1.0e-5f;
        if (valid1 && valid2) {
            bent = (r1 < r2) ? n1 : n2;  // both reflect above: the shallower one is closer to N
        } else {
            bent = (r1 > r2) ? n1 : n2;
        }
    } else if (valid1 || valid2) {
        const float z2 = valid1 ? n1z2 : n2z2;
        bent = float2(sqrt(max(1.0f - z2, 0.0f)), sqrt(max(z2, 0.0f)));
    } else {
        return Ng;
    }
    const float3 result = bent.x * X + bent.y * Ng;
    return all(isfinite(result)) ? normalize(result) : Ng;
}

// Backlog #47 (Dario 2026-11-24, as in Arnold): a type=sss material's normal map (normalTexture, SceneManager.mm) shapes
// only its coat lobe and the coat/walk lobe choice at the walk's entry. The walk's direction, its coat weights and its
// exit keep the interpolated normal, so without a coat a map changes nothing. This
// returns the mapped normal at `rec` without changing the record. The frame is the vertex's own: `vertexNormal` (its
// interpolated shading normal, as the integrator resolved it; the software tracer's rec.shadingNormal is the face normal
// at a fresh hit) and its geometric normal, UV set 0 through the normal slot's transform, and the vertex tangent or else the triangle's UV
// frame, as at an ordinary normal-mapped vertex. The walk has no ray footprint, so the map is read at mip 0. The result is
// flipped toward the geometric normal, as at an ordinary vertex. No map, a scale at or below 1e-4, or no tangent frame
// (degenerate UVs) returns false.
inline bool subsurface_coat_mapped_normal(
    constant PathtraceUniforms& uniforms,
    constant array<texture2d<float, access::sample>, kMaxMaterialTextures>& materialTextures,
    constant array<sampler, kMaxMaterialSamplers>& materialSamplers,
    device const MaterialTextureInfo* materialTextureInfos,
    device const MeshInfo* meshInfos,
    device const SceneVertex* sceneVertices,
    device const uint3* meshIndices,
    const MaterialData material,
    thread const HitRecord& rec,
    const float3 vertexNormal,
    thread float3& mappedNormal) {
    if (!material_texture_valid(uniforms, material.textureIndices0.z) ||
        rec.primitiveType != kPrimitiveTypeTriangle || !meshInfos || !sceneVertices || !meshIndices ||
        uniforms.meshCount == 0u) {
        return false;
    }
#if PT_DEBUG_TOOLS
    if (PT_DEFAULT_OFF(uniforms.debugDisableNormalMap) != 0u) {
        return false;
    }
#endif
    const float normalScale = material.pbrParams.w;
    float3 shadingNormal = vertexNormal;
    if (normalScale <= 1.0e-4f || !all(isfinite(shadingNormal)) || dot(shadingNormal, shadingNormal) <= 0.0f) {
        return false;
    }
    shadingNormal = normalize(shadingNormal);
    const float2 uv0 = interpolate_uv(uniforms, rec.meshIndex, rec.primitiveIndex, rec.barycentric, 0u,
                                      meshInfos, sceneVertices, meshIndices);
    float3 row0 = float3(1.0f, 0.0f, 0.0f);
    float3 row1 = float3(0.0f, 1.0f, 0.0f);
    pbr_texture_transform_rows(material, kPbrTextureSlotNormal, row0, row1);
    const float2 uv = pbr_apply_uv_transform(uv0, row0, row1);
    const float4 texel = sample_material_texture_filtered(materialTextures,
                                                          materialSamplers,
                                                          materialTextureInfos,
                                                          uniforms,
                                                          material.textureIndices0.z,
                                                          uv,
                                                          float4(0.5f, 0.5f, 1.0f, 1.0f),
                                                          /*lod=*/0.0f,
                                                          /*hasIgehyGradients=*/false,
                                                          float2(0.0f),
                                                          float2(0.0f));
    const float3 n = decode_normal_map(texel.xyz, normalScale, /*flipGreen=*/false);
    const float4 tangent = interpolate_tangent(uniforms, rec.meshIndex, rec.primitiveIndex, rec.barycentric,
                                               meshInfos, sceneVertices, meshIndices);
    float3 t = tangent.xyz;
    float3 b = float3(0.0f);
    bool hasBasis = false;
    if (fabs(tangent.w) > 0.5f && all(isfinite(t)) && dot(t, t) > 1.0e-6f) {
        t = normalize(t - shadingNormal * dot(shadingNormal, t));
        if (all(isfinite(t)) && dot(t, t) > 1.0e-6f) {
            b = normalize(cross(shadingNormal, t)) * (tangent.w < 0.0f ? -1.0f : 1.0f);
            hasBasis = all(isfinite(b)) && dot(b, b) > 1.0e-6f;
        }
    }
    if (!hasBasis) {
        hasBasis = compute_tangent_basis_from_uv(uniforms, rec.meshIndex, rec.primitiveIndex, 0u,
                                                 meshInfos, sceneVertices, meshIndices, shadingNormal, t, b);
    }
    if (!hasBasis) {
        return false;
    }
    float3 mapped = normalize(t * n.x + b * n.y + shadingNormal * n.z);
    if (!all(isfinite(mapped))) {
        return false;
    }
    if (dot(mapped, rec.normal) < 0.0f) {
        mapped = -mapped;
    }
    mappedNormal = mapped;
    return true;
}
