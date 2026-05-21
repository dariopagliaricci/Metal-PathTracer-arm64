inline bool mnee_rect_light_hit(constant PathtraceUniforms& uniforms,
                                device const RectData* rectangles,
                                device const MaterialData* materials,
                                texture2d<float, access::sample> environmentTexture,
                                uint lightCount,
                                thread const HitRecord& lightRec,
                                const float3 origin,
                                thread MneeRectHit& outHit);

inline bool sample_rect_light(constant PathtraceUniforms& uniforms,
                              device const RectData* rectangles,
                              device const MaterialData* materials,
                              texture2d<float, access::sample> environmentTexture,
                              thread const HitRecord& rec,
                              thread uint& state,
                              uint lightCount,
                              thread RectLightSample& outSample) {
    outSample.pdf = 0.0f;
    outSample.distance = 0.0f;
    outSample.direction = float3(0.0f);
    outSample.emission = float3(0.0f);
    outSample.rectIndex = kInvalidIndex;

    if (lightCount == 0u || !rectangles || !materials || uniforms.materialCount == 0) {
        return false;
    }

    uint selected = min(static_cast<uint>(rand_uniform(state) * float(lightCount)), lightCount - 1u);
    uint current = 0u;
    uint chosenRectIndex = kInvalidIndex;
    MaterialData chosenMaterial{};

    for (uint i = 0; i < uniforms.rectangleCount; ++i) {
        uint matIndex = min(rectangles[i].materialTwoSided.x, uniforms.materialCount - 1);
        MaterialData material = materials[matIndex];
        if (static_cast<uint>(material.typeEta.x) != 3u ||
            all(material.emission.xyz == float3(0.0f))) {
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

    float3 emission = chosenMaterial.emission.xyz;
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
    uint matIndex = min(rect.materialTwoSided.x, uniforms.materialCount - 1);
    MaterialData material = materials[matIndex];
    if (static_cast<uint>(material.typeEta.x) != 3u ||
        all(material.emission.xyz == float3(0.0f))) {
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

    if (lightCount == 0u || !rectangles || !materials || uniforms.materialCount == 0u) {
        return false;
    }
    if (rectIndex >= uniforms.rectangleCount) {
        return false;
    }

    const device RectData& rect = rectangles[rectIndex];
    uint matIndex = min(rect.materialTwoSided.x, uniforms.materialCount - 1u);
    MaterialData material = materials[matIndex];
    if (static_cast<uint>(material.typeEta.x) != 3u || all(material.emission.xyz == float3(0.0f))) {
        return false;
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

    float3 emission = material.emission.xyz;
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
    float cosLight = dot(-direction, normal);
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
    float cosLight = dot(-direction, normal);
    if (!(cosLight > 0.0f) || !isfinite(cosLight)) {
        return false;
    }

    float selectionPdf = emissive_primitive_selection_pdf(uniforms, light);
    float pdfArea = 1.0f / light.area;
    float pdfDir = pdfArea * distSq / max(cosLight, 1.0e-6f);
    float pdf = pdfDir * selectionPdf;
    if (!(pdf > 0.0f) || !isfinite(pdf)) {
        return false;
    }

    outSample.primitiveIndex = light.triangleIndex;
    outSample.flags = light.flags;
    outSample.bary = float2(b1, b2);
    outSample.position = samplePoint;
    outSample.normal = normal;
    outSample.emission = light.emission;
    outSample.pdf = pdf;
    return true;
}
