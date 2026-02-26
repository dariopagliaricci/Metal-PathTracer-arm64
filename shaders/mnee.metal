inline bool mnee_rect_light_hit(constant PathtraceUniforms& uniforms,
                                device const RectData* rectangles,
                                device const MaterialData* materials,
                                texture2d<float, access::sample> environmentTexture,
                                uint lightCount,
                                thread const HitRecord& lightRec,
                                const float3 origin,
                                thread MneeRectHit& outHit) {
    outHit.emission = float3(0.0f);
    outHit.pdf = 0.0f;

    if (lightCount == 0u || !rectangles || !materials || uniforms.materialCount == 0) {
        return false;
    }
    if (lightRec.primitiveType != kPrimitiveTypeRectangle ||
        lightRec.primitiveIndex >= uniforms.rectangleCount) {
        return false;
    }

    uint matIndex = min(rectangles[lightRec.primitiveIndex].materialTwoSided.x,
                        uniforms.materialCount - 1u);
    MaterialData material = materials[matIndex];
    if (static_cast<uint>(material.typeEta.x) != 3u ||
        all(material.emission.xyz == float3(0.0f))) {
        return false;
    }

    if (lightRec.frontFace == 0u && lightRec.twoSided == 0u) {
        return false;
    }

    float3 emission = material.emission.xyz;
    if (material.emission.w > 0.0f &&
        environmentTexture.get_width() > 0 &&
        environmentTexture.get_height() > 0 &&
        lightRec.frontFace != 0u) {
        float3 sampleDir = -lightRec.shadingNormal;
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

    float pdf = rect_light_pdf_for_hit(uniforms,
                                       rectangles,
                                       materials,
                                       lightCount,
                                       lightRec,
                                       origin);
    if (pdf <= 0.0f || !isfinite(pdf)) {
        return false;
    }

    outHit.emission = emission;
    outHit.pdf = pdf;
    return true;
}
