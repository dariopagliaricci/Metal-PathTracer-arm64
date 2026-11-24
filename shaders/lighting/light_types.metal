inline bool reference_point_is_exact_nee_light(const RectData light) {
    return area_primitive_is_reference_point(light) && light.corner.w == 0.0f;
}

inline bool reference_point_exact_nee_active(constant PathtraceUniforms& uniforms) {
    return uniforms.referencePointExactNee != 0u && uniforms.referencePointNeeShadowExtra != 0u;
}

inline uint rect_light_nee_draw_count(constant PathtraceUniforms& uniforms) {
    return reference_point_exact_nee_active(uniforms) ? uniforms.rectangleCount + 1u : 1u;
}

constant uint kAnalyticLightFilterAll = 0u;
constant uint kAnalyticLightFilterReferencePoint = 1u;
constant uint kAnalyticLightFilterArea = 2u;

inline float emissive_light_cosine(const uint flags, const float3 normal, const float3 towardReceiver) {
    const float cosine = dot(normal, towardReceiver);
    return (flags & kLightPrimitiveFlagTwoSided) != 0u ? fabs(cosine) : cosine;
}

inline bool analytic_light_matches_filter(const RectData light, const uint filter) {
    const bool point = area_primitive_is_reference_point(light);
    return filter == kAnalyticLightFilterAll ||
           (filter == kAnalyticLightFilterReferencePoint ? point : !point);
}

inline uint count_rect_lights(constant PathtraceUniforms& uniforms,
                              device const RectData* rectangles,
                              device const MaterialData* materials,
                              uint filter = kAnalyticLightFilterAll,
                              bool excludeExactPoints = false) {
    if (!rectangles || !materials || uniforms.rectangleCount == 0 || uniforms.materialCount == 0) {
        return 0u;
    }
    uint lightCount = 0u;
    for (uint i = 0; i < uniforms.rectangleCount; ++i) {
        if (!analytic_light_matches_filter(rectangles[i], filter) ||
            (excludeExactPoints && reference_point_is_exact_nee_light(rectangles[i]))) {
            continue;
        }
        uint matIndex = min(rectangles[i].materialTwoSided.x, uniforms.materialCount - 1);
        MaterialData material = materials[matIndex];
        if (material_closure_is_emissive(material)) {
            lightCount += 1u;
        }
    }
    return lightCount;
}

struct RectLightSample {
    float3 direction;
    float distance;
    float3 position;
    float3 normal;
    float pdf;
    float3 emission;
    uint rectIndex;
    bool isReferencePoint;
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
