inline bool mnee_rect_light_hit(constant PathtraceUniforms& uniforms,
                                device const RectData* rectangles,
                                device const MaterialData* materials,
                                texture2d<float, access::sample> environmentTexture,
                                uint lightCount,
                                thread const HitRecord& lightRec,
                                const float3 origin,
                                thread MneeRectHit& outHit);

// Invisible punctual intensity with only discrete light-selection probability.
// The default preserves the reference's disk jitter and 1 m attenuation cap.
// materialTwoSided.w enables uncapped inverse-square falloff; radius 0 gives
// a true point light. Neither mode has a BSDF-hit counterpart.
inline bool sample_reference_point_light(const RectData light,
                                         const MaterialData material,
                                         const float3 receiver,
                                         const float2 diskOffset,
                                         const uint lightCount,
                                         const uint lightIndex,
                                         thread RectLightSample& outSample) {
    const float3 centerDirection = light.corner.xyz - receiver;
    if (!(dot(centerDirection, centerDirection) > 1.0e-12f)) {
        return false;
    }
    float3 tangent, bitangent;
    build_onb(normalize(centerDirection), tangent, bitangent);
    const float3 samplePoint = light.corner.xyz + max(light.corner.w, 0.0f) *
        (diskOffset.x * tangent + diskOffset.y * bitangent);
    const float3 toLight = samplePoint - receiver;
    const float distanceSq = dot(toLight, toLight);
    if (!(distanceSq > 0.0f)) {
        return false;
    }
    outSample.distance = sqrt(distanceSq);
    outSample.direction = toLight / outSample.distance;
    outSample.position = samplePoint;
    outSample.normal = -outSample.direction;
    outSample.pdf = 1.0f / float(max(lightCount, 1u));
    const float legacyCap = light.edgeU.w > 0.0f ? 1.0f / light.edgeU.w : 1.0f;
    const float attenuation = light.materialTwoSided.w != 0u
        ? 1.0f / distanceSq : min(1.0f / distanceSq, legacyCap);
    outSample.emission = material_closure_aov_emission(material) * attenuation;
    outSample.rectIndex = lightIndex;
    outSample.isReferencePoint = true;
    return any(outSample.emission > float3(0.0f));
}

inline bool sample_rect_light(constant PathtraceUniforms& uniforms,
                              device const RectData* rectangles,
                              device const MaterialData* materials,
                              texture2d<float, access::sample> environmentTexture,
                              thread const HitRecord& rec,
                              thread uint& state,
                              uint lightCount,
                              thread RectLightSample& outSample,
                              uint filter = kAnalyticLightFilterAll,
                              bool excludeExactPoints = false) {
    outSample.pdf = 0.0f;
    outSample.distance = 0.0f;
    outSample.direction = float3(0.0f);
    outSample.position = float3(0.0f);
    outSample.normal = float3(0.0f);
    outSample.emission = float3(0.0f);
    outSample.rectIndex = kInvalidIndex;
    outSample.isReferencePoint = false;

    if (lightCount == 0u || !rectangles || !materials || uniforms.materialCount == 0) {
        return false;
    }

    uint selected = min(static_cast<uint>(rand_uniform(state) * float(lightCount)), lightCount - 1u);
    uint current = 0u;
    uint chosenRectIndex = kInvalidIndex;
    MaterialData chosenMaterial{};

    for (uint i = 0; i < uniforms.rectangleCount; ++i) {
        if (!analytic_light_matches_filter(rectangles[i], filter) ||
            (excludeExactPoints && reference_point_is_exact_nee_light(rectangles[i]))) {
            continue;
        }
        uint matIndex = min(rectangles[i].materialTwoSided.x, uniforms.materialCount - 1);
        MaterialData material = materials[matIndex];
        if (!material_closure_is_emissive(material)) {
            continue;
        }
        if (current == selected) {
            chosenRectIndex = i;
            chosenMaterial = material;
            break;
        }
        current += 1u;
    }

    if (chosenRectIndex == kInvalidIndex) {
        return false;
    }

    const device RectData& rect = rectangles[chosenRectIndex];
    if (area_primitive_is_reference_point(rect)) {
        return sample_reference_point_light(rect, chosenMaterial, rec.point,
                                            random_in_unit_disk(state), lightCount,
                                            chosenRectIndex, outSample);
    }
    float3 edgeU = rect.edgeU.xyz;
    float3 edgeV = rect.edgeV.xyz;
    float3 samplePoint = rect.corner.xyz;
    if (area_primitive_is_disk(rect)) {
        float2 diskUv = random_in_unit_disk(state);
        samplePoint += diskUv.x * edgeU + diskUv.y * edgeV;
    } else {
        float u = rand_uniform(state);
        float v = rand_uniform(state);
        samplePoint += u * edgeU + v * edgeV;
    }
    float3 toLight = samplePoint - rec.point;
    float distSq = dot(toLight, toLight);
    if (distSq <= 0.0f) {
        return false;
    }

    float distance = sqrt(distSq);
    float3 direction = toLight / distance;

    float area = area_primitive_area(rect);
    if (area <= 0.0f) {
        return false;
    }

    float3 normal = rect.normalAndPlane.xyz;
    float cosLight = dot(-direction, normal);
    if (rect.materialTwoSided.y != 0u) {
        cosLight = fabs(cosLight);
    } else {
        if (cosLight <= 0.0f) {
            return false;
        }
    }
    if (cosLight <= 0.0f) {
        return false;
    }

    float pdfArea = 1.0f / area;
    float pdfDir = pdfArea * distSq / max(cosLight, 1e-6f);
    float selectionPdf = 1.0f / float(lightCount);
    float pdf = pdfDir * selectionPdf;
    if (pdf <= 0.0f || !isfinite(pdf)) {
        return false;
    }

    float3 emission = material_closure_aov_emission(chosenMaterial);
    if (chosenMaterial.emission.w > 0.0f &&
        environmentTexture.get_width() > 0 &&
        environmentTexture.get_height() > 0) {
        float3 sampleDir = -normal;
        float3 envColor = environment_color(environmentTexture,
                                            sampleDir,
                                            uniforms.environmentRotation,
                                            uniforms.environmentIntensity,
                                            uniforms);
        emission *= envColor;
    }

    if (all(emission == float3(0.0f))) {
        return false;
    }

    outSample.direction = direction;
    outSample.distance = distance;
    outSample.position = samplePoint;
    outSample.normal = normal;
    outSample.pdf = pdf;
    outSample.emission = emission;
    outSample.rectIndex = chosenRectIndex;
    return true;
}

inline float rect_light_pdf_for_hit(constant PathtraceUniforms& uniforms,
                                    device const RectData* rectangles,
                                    device const MaterialData* materials,
                                    uint lightCount,
                                    thread const HitRecord& lightRec,
                                    const float3 origin) {
    if (reference_point_exact_nee_active(uniforms)) {
        lightCount = count_rect_lights(uniforms, rectangles, materials, kAnalyticLightFilterAll, true);
    }
    if (lightCount == 0u || !rectangles || !materials || uniforms.materialCount == 0) {
        return 0.0f;
    }
    if (lightRec.primitiveType != kPrimitiveTypeRectangle) {
        return 0.0f;
    }
    uint rectIndex = lightRec.primitiveIndex;
    if (rectIndex >= uniforms.rectangleCount) {
        return 0.0f;
    }

    const device RectData& rect = rectangles[rectIndex];
    if (area_primitive_is_reference_point(rect)) {
        return 0.0f;
    }
    uint matIndex = min(rect.materialTwoSided.x, uniforms.materialCount - 1);
    MaterialData material = materials[matIndex];
    if (!material_closure_is_emissive(material)) {
        return 0.0f;
    }

    float3 edgeU = rect.edgeU.xyz;
    float3 edgeV = rect.edgeV.xyz;
    float area = area_primitive_area(rect);
    if (area <= 0.0f) {
        return 0.0f;
    }

    float3 toLight = lightRec.point - origin;
    float distSq = dot(toLight, toLight);
    if (distSq <= 0.0f) {
        return 0.0f;
    }

    float distance = sqrt(distSq);
    float3 direction = toLight / distance;
    float3 normal = rect.normalAndPlane.xyz;
    float cosLight = dot(-direction, normal);
    if (rect.materialTwoSided.y != 0u) {
        cosLight = fabs(cosLight);
    } else {
        if (cosLight <= 0.0f) {
            return 0.0f;
        }
    }
    if (cosLight <= 0.0f) {
        return 0.0f;
    }

    float pdfArea = 1.0f / area;
    float pdfDir = pdfArea * distSq / max(cosLight, 1e-6f);
    float selectionPdf = 1.0f / float(lightCount);
    return pdfDir * selectionPdf;
}

inline bool deterministic_rect_light_sample(constant PathtraceUniforms& uniforms,
                                            device const RectData* rectangles,
                                            device const MaterialData* materials,
                                            texture2d<float, access::sample> environmentTexture,
                                            thread const HitRecord& rec,
                                            uint lightCount,
                                            uint rectIndex,
                                            thread RectLightSample& outSample) {
    outSample.pdf = 0.0f;
    outSample.distance = 0.0f;
    outSample.direction = float3(0.0f);
    outSample.emission = float3(0.0f);
    outSample.rectIndex = kInvalidIndex;
    outSample.isReferencePoint = false;

    if (lightCount == 0u || !rectangles || !materials || uniforms.materialCount == 0u) {
        return false;
    }
    if (rectIndex >= uniforms.rectangleCount) {
        return false;
    }

    const device RectData& rect = rectangles[rectIndex];
    uint matIndex = min(rect.materialTwoSided.x, uniforms.materialCount - 1u);
    MaterialData material = materials[matIndex];
    if (!material_closure_is_emissive(material)) {
        return false;
    }

    if (area_primitive_is_reference_point(rect)) {
        return sample_reference_point_light(rect, material, rec.point, float2(0.0f),
                                            lightCount, rectIndex, outSample);
    }

    float3 samplePoint = rect.corner.xyz;
    if (!area_primitive_is_disk(rect)) {
        samplePoint += 0.5f * rect.edgeU.xyz + 0.5f * rect.edgeV.xyz;
    }

    float3 toLight = samplePoint - rec.point;
    float distSq = dot(toLight, toLight);
    if (distSq <= 0.0f) {
        return false;
    }
    float distance = sqrt(distSq);
    float3 direction = toLight / distance;
    float area = area_primitive_area(rect);
    if (area <= 0.0f) {
        return false;
    }
    float3 normal = rect.normalAndPlane.xyz;
    float cosLight = dot(-direction, normal);
    if (rect.materialTwoSided.y != 0u) {
        cosLight = fabs(cosLight);
    } else if (cosLight <= 0.0f) {
        return false;
    }
    if (cosLight <= 0.0f) {
        return false;
    }

    float pdfArea = 1.0f / area;
    float pdfDir = pdfArea * distSq / max(cosLight, 1e-6f);
    float selectionPdf = 1.0f / float(lightCount);
    float pdf = pdfDir * selectionPdf;
    if (pdf <= 0.0f || !isfinite(pdf)) {
        return false;
    }

    float3 emission = material_closure_aov_emission(material);
    if (material.emission.w > 0.0f &&
        environmentTexture.get_width() > 0 &&
        environmentTexture.get_height() > 0) {
        float3 sampleDir = -normal;
        float3 envColor = environment_color(environmentTexture,
                                            sampleDir,
                                            uniforms.environmentRotation,
                                            uniforms.environmentIntensity,
                                            uniforms);
        emission *= envColor;
    }
    if (all(emission == float3(0.0f))) {
        return false;
    }

    outSample.direction = direction;
    outSample.distance = distance;
    outSample.pdf = pdf;
    outSample.emission = emission;
    outSample.rectIndex = rectIndex;
    return true;
}

inline float emissive_primitive_selection_pdf(constant PathtraceUniforms& uniforms,
                                              const device LightPrimitive& light) {
    if (uniforms.emissiveTotalPower > 0.0f &&
        isfinite(uniforms.emissiveTotalPower) &&
        light.power > 0.0f &&
        isfinite(light.power)) {
        return light.power / uniforms.emissiveTotalPower;
    }
    return uniforms.emissivePrimitiveCount > 0u
        ? (1.0f / float(uniforms.emissivePrimitiveCount))
        : 0.0f;
}

// The light stores integrated irradiance, rather than cone radiance. Sampling
// uniformly in solid angle makes radiance / cone-PDF cancel; keep E * cos and
// the discrete light-selection PDF. The invisible sun has no BSDF-hit MIS term.
inline float3 sample_directional_cone(const float3 axis,
                                      const float cosDelta,
                                      thread uint& state) {
    if (!(cosDelta > 0.0f)) {
        return axis; // Preserve delta direction and RNG sequence exactly.
    }
    const float delta = rand_uniform(state) * cosDelta;
    const float cosine = 1.0f - delta;
    const float sine = sqrt(max(delta * (2.0f - delta), 0.0f));
    const float phi = 2.0f * kPi * rand_uniform(state);
    float3 tangent, bitangent;
    build_onb(axis, tangent, bitangent);
    return cosine * axis + sine * (cos(phi) * tangent + sin(phi) * bitangent);
}

inline bool deterministic_emissive_primitive_sample(constant PathtraceUniforms& uniforms,
                                                    device const LightPrimitive* emissivePrimitives,
                                                    thread const HitRecord& rec,
                                                    uint primitiveIndex,
                                                    thread EmissivePrimitiveAuditSample& outSample) {
    outSample.direction = float3(0.0f);
    outSample.distance = 0.0f;
    outSample.pdf = 0.0f;
    outSample.emission = float3(0.0f);
    outSample.primitiveIndex = kInvalidIndex;

    if (!emissivePrimitives || uniforms.emissivePrimitiveCount == 0u) {
        return false;
    }
    if (primitiveIndex >= uniforms.emissivePrimitiveCount) {
        return false;
    }

    const device LightPrimitive& light = emissivePrimitives[primitiveIndex];
    const bool isDirectional = (light.flags & kLightPrimitiveFlagDirectional) != 0u;
    if (isDirectional) {
        float3 omegaL = safe_normalize(light.centroid);
        float selectionPdf = emissive_primitive_selection_pdf(uniforms, light);
        if (!all(isfinite(omegaL)) ||
            !(selectionPdf > 0.0f) ||
            !isfinite(selectionPdf) ||
            !all(isfinite(light.emission))) {
            return false;
        }
        outSample.direction = omegaL;
        outSample.distance = 1.0e20f;
        outSample.pdf = selectionPdf;
        outSample.emission = light.emission;
        outSample.primitiveIndex = kInvalidIndex;
        return true;
    }
    if (light.triangleIndex >= uniforms.triangleCount) {
        return false;
    }
    if (!(light.area > 0.0f) || !all(isfinite(light.emission)) || !all(isfinite(light.normal))) {
        return false;
    }
    float3 samplePoint = light.centroid;
    float3 toLight = samplePoint - rec.point;
    float distSq = dot(toLight, toLight);
    if (!(distSq > 0.0f) || !isfinite(distSq)) {
        return false;
    }

    float distance = sqrt(distSq);
    if (!(distance > 0.0f) || !isfinite(distance)) {
        return false;
    }
    float3 direction = toLight / distance;
    float3 normal = light.normal;
    float normalLenSq = dot(normal, normal);
    if (!(normalLenSq > 1.0e-12f)) {
        return false;
    }
    normal *= rsqrt(normalLenSq);
    float cosLight = emissive_light_cosine(light.flags, normal, -direction);
    if (!(cosLight > 0.0f) || !isfinite(cosLight)) {
        return false;
    }

    float pdfArea = 1.0f / light.area;
    float pdfDir = pdfArea * distSq / max(cosLight, 1.0e-6f);
    float selectionPdf = 1.0f / float(uniforms.emissivePrimitiveCount);
    if (uniforms.emissiveTotalPower > 0.0f &&
        isfinite(uniforms.emissiveTotalPower) &&
        light.power > 0.0f &&
        isfinite(light.power)) {
        selectionPdf = light.power / uniforms.emissiveTotalPower;
    }
    float pdf = pdfDir * selectionPdf;
    if (!(pdf > 0.0f) || !isfinite(pdf)) {
        return false;
    }

    outSample.direction = direction;
    outSample.distance = distance;
    outSample.pdf = pdf;
    outSample.emission = light.emission;
    outSample.primitiveIndex = light.triangleIndex;
    return true;
}

inline uint sample_emissive_primitive_index_power(constant PathtraceUniforms& uniforms,
                                                  device const LightPrimitive* emissivePrimitives,
                                                  float u) {
    uint count = uniforms.emissivePrimitiveCount;
    if (count == 0u) {
        return 0u;
    }
    if (!(uniforms.emissiveTotalPower > 0.0f) || !isfinite(uniforms.emissiveTotalPower)) {
        return min(static_cast<uint>(u * float(count)), count - 1u);
    }

    float target = saturate(u);
    uint lo = 0u;
    uint hi = count - 1u;
    while (lo < hi) {
        uint mid = lo + ((hi - lo) >> 1u);
        float cdf = emissivePrimitives[mid].cumulativePower;
        if (target <= cdf) {
            hi = mid;
        } else {
            lo = mid + 1u;
        }
    }
    return min(lo, count - 1u);
}

inline float tagged_emissive_selection_pdf(constant PathtraceUniforms& uniforms,
                                          device const LightPrimitive* lights,
                                          const uint index) {
    const device LightPrimitive& light = lights[index];
    if ((light.flags & (kLightPrimitiveFlagTwoSided | kLightPrimitiveFlagTexturedEmission)) == 0u) {
        return emissive_primitive_selection_pdf(uniforms, light);
    }
    if (!(uniforms.emissiveTotalPower > 0.0f) || !isfinite(uniforms.emissiveTotalPower)) {
        return 1.0f / float(max(uniforms.emissivePrimitiveCount, 1u));
    }
    const float previous = index > 0u ? lights[index - 1u].cumulativePower : 0.0f;
    return max(light.cumulativePower - previous, 0.0f);
}

// Modes whose fused NEE draws mesh emitters from the emissive inventory
// (sample_direct_light_baseline, or RIS/ReSTIR reservoirs built from it).
inline bool mesh_emitter_nee_mode(constant PathtraceUniforms& uniforms) {
    return uniforms.directLightMode != kDirectLightModeLegacyRect;
}

// Modes whose reservoir can carry a sample drawn at another shading point,
// frame or cache cell (or, for restir_di, extra candidate sets), so the
// density of the shaded mesh sample is intractable.
inline bool mesh_emitter_nee_reuses_reservoirs(constant PathtraceUniforms& uniforms) {
    return uniforms.directLightMode != kDirectLightModeLegacyRect &&
           uniforms.directLightMode != kDirectLightModeBaselineEmissive &&
           uniforms.directLightMode != kDirectLightModeRis;
}

// Draws of sample_direct_light_baseline behind one NEE estimate: one for
// baseline NEE, M candidates for plain RIS.
inline float mesh_emitter_nee_candidate_count(constant PathtraceUniforms& uniforms) {
    return uniforms.directLightMode == kDirectLightModeRis
        ? float(max(uniforms.risCandidateCount, 1u)) : 1.0f;
}

// Inventory entry of a mesh triangle, or kInvalidIndex. Mesh emitters are
// built in ascending global triangle order; analytic lights follow with
// triangleIndex=UINT_MAX. A binary search avoids scanning the hundreds of
// thousands of emitter triangles of Emerald Square or the Bistro.
inline uint mesh_emitter_inventory_index(constant PathtraceUniforms& uniforms,
                                         device const LightPrimitive* lights,
                                         const uint triangleIndex) {
    uint lo = 0u;
    uint hi = uniforms.emissivePrimitiveCount;
    while (lo < hi) {
        const uint mid = lo + (hi - lo) / 2u;
        if (lights[mid].triangleIndex < triangleIndex) lo = mid + 1u;
        else hi = mid;
    }
    return (lo < uniforms.emissivePrimitiveCount && lights[lo].triangleIndex == triangleIndex)
        ? lo : kInvalidIndex;
}

// Solid-angle density with which `candidates` draws of
// sample_direct_light_baseline from `origin` reach the hit point of a mesh
// emitter. 0 for triangles outside the inventory or seen from a side that
// does not emit, which NEE never samples.
inline float mesh_emitter_light_pdf_for_hit(constant PathtraceUniforms& uniforms,
                                            device const LightPrimitive* lights,
                                            thread const HitRecord& hit,
                                            const float3 origin,
                                            const float candidates) {
    if (!lights || hit.primitiveType != kPrimitiveTypeTriangle) {
        return 0.0f;
    }
    const float3 delta = hit.point - origin;
    const float distanceSq = dot(delta, delta);
    if (!(distanceSq > 0.0f)) {
        return 0.0f;
    }
    const uint index = mesh_emitter_inventory_index(uniforms, lights, hit.primitiveIndex);
    if (index == kInvalidIndex) {
        return 0.0f;
    }
    const LightPrimitive light = lights[index];
    if (!(light.area > 0.0f)) {
        return 0.0f;
    }
    const float cosine = emissive_light_cosine(light.flags, light.normal,
                                               -delta * rsqrt(distanceSq));
    return cosine > 0.0f
        ? candidates * tagged_emissive_selection_pdf(uniforms, lights, index) * distanceSq / (light.area * cosine)
        : 0.0f;
}

// Weight of mesh emission reached by a non-delta BSDF scatter from `origin`
// whose vertex ran fused mesh NEE covering that direction. NEE and this hit
// must sum to one for every emitter point:
// - baseline: balance heuristic against the single light draw;
// - plain RIS: balance heuristic against M draws of the same proposal; the
//   RIS shading side applies the complement, so the pair stays a partition
//   of unity whatever the resampled density is;
// - reservoir reuse (ris_spatial/temporal/world/regir, restir_di, hybrid): 0,
//   the reservoir NEE carries the whole contribution.
// Emission the NEE cannot sample (outside the inventory, non-emitting side)
// keeps weight 1.
inline float mesh_emitter_bsdf_hit_weight(constant PathtraceUniforms& uniforms,
                                          device const LightPrimitive* lights,
                                          thread const HitRecord& hit,
                                          const float3 origin,
                                          const float bsdfPdf) {
    const float lightPdf = mesh_emitter_light_pdf_for_hit(uniforms, lights, hit, origin,
                                                          mesh_emitter_nee_candidate_count(uniforms));
    if (!(lightPdf > 0.0f)) {
        return 1.0f;
    }
    if (mesh_emitter_nee_reuses_reservoirs(uniforms)) {
        return 0.0f;
    }
    return bsdfPdf > 0.0f ? bsdfPdf / (bsdfPdf + lightPdf) : 1.0f;
}

// NEE side of the same partition for a non-directional mesh sample with
// combined draw density `lightPdf` (light pdf times draws): the BSDF scatter
// only continues to an emitter hit while depth + 1 < maxDepth.
inline float mesh_emitter_nee_mis_weight(constant PathtraceUniforms& uniforms,
                                         const uint depth,
                                         const float lightPdf,
                                         const float bsdfPdf) {
    if (mesh_emitter_nee_reuses_reservoirs(uniforms) ||
        !(depth + 1u < uniforms.maxDepth) || !(bsdfPdf > 0.0f)) {
        return 1.0f;
    }
    return lightPdf / (lightPdf + bsdfPdf);
}

// Emerald's textured mesh emitters lead the inventory; ordinary scenes keep
// one draw and their original RNG sequence. Averaging independent draws keeps
// the existing light/BSDF MIS partition and expected radiance unchanged.
inline uint baseline_direct_light_sample_count(constant PathtraceUniforms& uniforms,
                                                device const LightPrimitive* lights) {
    return lights && uniforms.emissivePrimitiveCount > 0u &&
        (lights[0].flags & kLightPrimitiveFlagTexturedEmission) != 0u ? 4u : 1u;
}

inline bool sample_direct_light_baseline(constant PathtraceUniforms& uniforms,
                                         device const LightPrimitive* emissivePrimitives,
                                         thread const HitRecord& rec,
                                         thread uint& state,
                                         thread RISSamplePayload& outSample) {
    outSample.primitiveIndex = kInvalidIndex;
    outSample.flags = 0u;
    outSample.bary = float2(0.0f);
    outSample.position = float3(0.0f);
    outSample.normal = float3(0.0f);
    outSample.emission = float3(0.0f);
    outSample.pdf = 0.0f;

    if (!emissivePrimitives || uniforms.emissivePrimitiveCount == 0u) {
        return false;
    }

    uint selected = sample_emissive_primitive_index_power(uniforms,
                                                          emissivePrimitives,
                                                          rand_uniform(state));
    const device LightPrimitive& light = emissivePrimitives[selected];
    const bool isDirectional = (light.flags & kLightPrimitiveFlagDirectional) != 0u;
    if (!(light.area > 0.0f) ||
        !all(isfinite(light.normal)) ||
        !all(isfinite(light.emission)) ||
        !all(isfinite(light.edge1)) ||
        !all(isfinite(light.edge2))) {
        return false;
    }

    if (isDirectional) {
        float3 omegaL = safe_normalize(light.centroid);
        if (!all(isfinite(omegaL)) || dot(omegaL, omegaL) <= 0.0f) {
            return false;
        }
        omegaL = sample_directional_cone(omegaL, light.directionalCosDelta, state);
        float selectionPdf = emissive_primitive_selection_pdf(uniforms, light);
        if (!(selectionPdf > 0.0f) || !isfinite(selectionPdf)) {
            return false;
        }
        outSample.primitiveIndex = kInvalidIndex;
        outSample.flags = light.flags;
        outSample.bary = float2(0.0f);
        outSample.position = omegaL;
        outSample.normal = -omegaL;
        outSample.emission = light.emission;
        outSample.pdf = selectionPdf;
        return true;
    }

    if (light.triangleIndex >= uniforms.triangleCount) {
        return false;
    }

    float3 normal = light.normal;
    float normalLenSq = dot(normal, normal);
    if (!(normalLenSq > 1.0e-12f)) {
        return false;
    }
    normal *= rsqrt(normalLenSq);

    float3 v0 = light.centroid - (light.edge1 + light.edge2) * (1.0f / 3.0f);
    float sqrtU = sqrt(rand_uniform(state));
    float b1 = sqrtU * (1.0f - rand_uniform(state));
    float b2 = sqrtU - b1;
    float3 samplePoint = v0 + b1 * light.edge1 + b2 * light.edge2;

    float3 toLight = samplePoint - rec.point;
    float distSq = dot(toLight, toLight);
    if (!(distSq > 0.0f) || !isfinite(distSq)) {
        return false;
    }
    float distance = sqrt(distSq);
    if (!(distance > 0.0f) || !isfinite(distance)) {
        return false;
    }
    float3 direction = toLight / distance;
    float cosLight = emissive_light_cosine(light.flags, normal, -direction);
    if (!(cosLight > 0.0f) || !isfinite(cosLight)) {
        return false;
    }

    float selectionPdf = tagged_emissive_selection_pdf(uniforms, emissivePrimitives, selected);
    float pdfArea = 1.0f / light.area;
    float pdfDir = pdfArea * distSq / max(cosLight, 1.0e-6f);
    float pdf = pdfDir * selectionPdf;
    if (!(pdf > 0.0f) || !isfinite(pdf)) {
        return false;
    }

    outSample.primitiveIndex = light.triangleIndex;
    // The low byte holds light flags; textured lights carry their mesh index
    // in the remaining bits without changing reservoir or light-buffer layout.
    outSample.flags = light.flags |
        (((light.flags & kLightPrimitiveFlagTexturedEmission) != 0u) ? (light.reserved1 << 8u) : 0u);
    outSample.bary = float2(b1, b2);
    outSample.position = samplePoint;
    outSample.normal = normal;
    outSample.emission = light.emission;
    outSample.pdf = pdf;
    return true;
}

// Selection power is a proposal only. Evaluate the authored emissive texture
// at the sampled point so black atlas regions never become area lights.
inline bool resolve_textured_direct_light_emission(
    constant PathtraceUniforms& uniforms,
    thread RISSamplePayload& sample,
    device const MaterialData* materials,
    constant array<texture2d<float, access::sample>, kMaxMaterialTextures>& materialTextures,
    constant array<sampler, kMaxMaterialSamplers>& materialSamplers,
    device const MaterialTextureInfo* materialTextureInfos,
    device const MeshInfo* meshInfos,
    device const SceneVertex* sceneVertices,
    device const uint3* meshIndices) {
    if ((sample.flags & kLightPrimitiveFlagTexturedEmission) == 0u) return true;
    const uint meshIndex = sample.flags >> 8u;
    if (!materials || !meshInfos || meshIndex >= uniforms.meshCount) return false;
    const uint materialIndex = meshInfos[meshIndex].materialIndex;
    if (materialIndex >= uniforms.materialCount) return false;
    const MaterialData material = materials[materialIndex];
    if (!material_texture_valid(uniforms, material.textureIndices1.x)) return false;
    const float2 uv0 = interpolate_uv(uniforms, meshIndex, sample.primitiveIndex,
                                      sample.bary, 0u, meshInfos, sceneVertices, meshIndices);
    const float2 uv1 = interpolate_uv(uniforms, meshIndex, sample.primitiveIndex,
                                      sample.bary, 1u, meshInfos, sceneVertices, meshIndices);
    // NEE samples a point, so mip zero has no ray differentials. Keep this
    // lookup small inside the hardware megakernel rather than constructing a
    // full surface texture context with unused gradient/LOD branches.
    float2 emissiveUv = material.textureUvSet1.x == 1u ? uv1 : uv0;
    emissiveUv = float2(dot(material.textureTransform8.xy, emissiveUv) + material.textureTransform8.z,
                        dot(material.textureTransform9.xy, emissiveUv) + material.textureTransform9.z);
    const uint emissiveIndex = material.textureIndices1.x;
    const uint emissiveSampler = min(materialTextureInfos[emissiveIndex].flags, kMaxMaterialSamplers - 1u);
    const float3 color = materialTextures[emissiveIndex].sample(materialSamplers[emissiveSampler],
                                                               emissiveUv, level(0.0f)).xyz;
    float alpha = clamp(material.pbrExtras.x, 0.0f, 1.0f);
    if (material.pbrExtras.w > 0.5f && material_texture_valid(uniforms, material.textureIndices0.x)) {
        float2 baseUv = material.textureUvSet0.x == 1u ? uv1 : uv0;
        baseUv = float2(dot(material.textureTransform0.xy, baseUv) + material.textureTransform0.z,
                        dot(material.textureTransform1.xy, baseUv) + material.textureTransform1.z);
        const uint baseIndex = material.textureIndices0.x;
        const uint baseSampler = min(materialTextureInfos[baseIndex].flags, kMaxMaterialSamplers - 1u);
        alpha *= materialTextures[baseIndex].sample(materialSamplers[baseSampler], baseUv, level(0.0f)).w;
    }
    if (material.pbrExtras.w > 0.5f && material.pbrExtras.w < 1.5f &&
        alpha < material.pbrExtras.y) return false;
    const float coverage = material.pbrExtras.w >= 1.5f ? alpha : 1.0f;
    sample.emission = sample.emission * color * coverage;
    return all(isfinite(sample.emission)) && any(sample.emission > float3(0.0f));
}

// The default calls the original sampler unchanged. Exact draws consume no RNG.
inline bool sample_rect_light_nee_draw(constant PathtraceUniforms& uniforms,
                                      device const RectData* rectangles,
                                      device const MaterialData* materials,
                                      texture2d<float, access::sample> environmentTexture,
                                      thread const HitRecord& rec,
                                      thread uint& state,
                                      uint legacyLightCount,
                                      thread RectLightSample& outSample,
                                      uint filter,
                                      uint draw) {
    if (!reference_point_exact_nee_active(uniforms)) {
        return sample_rect_light(uniforms, rectangles, materials, environmentTexture,
                                 rec, state, legacyLightCount, outSample, filter);
    }
    if (draw == 0u) {
        const uint stochasticCount = count_rect_lights(uniforms, rectangles, materials, filter, true);
        return sample_rect_light(uniforms, rectangles, materials, environmentTexture,
                                 rec, state, stochasticCount, outSample, filter, true);
    }
    const uint index = draw - 1u;
    if (index >= uniforms.rectangleCount || !rectangles ||
        !analytic_light_matches_filter(rectangles[index], filter) ||
        !reference_point_is_exact_nee_light(rectangles[index])) return false;
    return deterministic_rect_light_sample(uniforms, rectangles, materials, environmentTexture,
                                           rec, 1u, index, outSample);
}
