inline uint count_rect_lights(constant PathtraceUniforms& uniforms,
                              device const RectData* rectangles,
                              device const MaterialData* materials) {
    if (!rectangles || !materials || uniforms.rectangleCount == 0 || uniforms.materialCount == 0) {
        return 0u;
    }
    uint lightCount = 0u;
    for (uint i = 0; i < uniforms.rectangleCount; ++i) {
        uint matIndex = min(rectangles[i].materialTwoSided.x, uniforms.materialCount - 1);
        MaterialData material = materials[matIndex];
        if (static_cast<uint>(material.typeEta.x) == 3u &&
            any(material.emission.xyz != float3(0.0f))) {
            lightCount += 1u;
        }
    }
    return lightCount;
}

struct RectLightSample {
    float3 direction;
    float distance;
    float pdf;
    float3 emission;
    uint rectIndex;
};

struct MneeRectHit {
    float3 emission;
    float pdf;
};

struct EmissivePrimitiveAuditSample {
    float3 direction;
    float distance;
    float pdf;
    float3 emission;
    uint primitiveIndex;
};
