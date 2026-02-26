inline float3 RRTAndODTFit(float3 v) {
    float3 a = v * (v + 0.0245786f) - 0.000090537f;
    float3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
    return a / b;
}

inline float3 ACESFitted(float3 color) {
    const float3x3 inputMat = float3x3(float3(0.59719f, 0.07600f, 0.02840f),
                                       float3(0.35458f, 0.90834f, 0.13383f),
                                       float3(0.04823f, 0.01566f, 0.83777f));
    const float3x3 outputMat = float3x3(float3(1.60475f, -0.10208f, -0.00327f),
                                        float3(-0.53108f, 1.10813f, -0.07276f),
                                        float3(-0.07367f, -0.00605f, 1.07602f));

    color = inputMat * color;
    color = RRTAndODTFit(color);
    color = outputMat * color;
    return clamp(color, float3(0.0f), float3(1.0f));
}

inline float3 ACESSimple(float3 color) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), float3(0.0f), float3(1.0f));
}

inline float luminance(float3 c) {
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}

inline float3 tonemapReinhard(float3 c, float whitePoint) {
    float L = luminance(c);
    float denom = 1.0f + L / max(whitePoint, 1e-4f);
    return clamp(c / denom, float3(0.0f), float3(1.0f));
}

inline float3 tonemapHable(float3 color) {
    const float A = 0.15f;
    const float B = 0.50f;
    const float C = 0.10f;
    const float D = 0.20f;
    const float E = 0.02f;
    const float F = 0.30f;
    const float W = 11.2f;

    float3 numerator = ((color * (A * color + B)) + C * color + D);
    float3 denominator = ((color * (A * color + B)) + E * color + F);
    float3 mapped = numerator / denominator - float3(D / F);
    float white = ((W * (A * W + B)) + C * W + D) / ((W * (A * W + B)) + E * W + F) - D / F;
    return clamp(mapped / white, float3(0.0f), float3(1.0f));
}

inline float3 extractBloom(float3 hdrColor, float threshold) {
    float luma = luminance(hdrColor);
    if (luma <= threshold) {
        return float3(0.0f);
    }
    float soft = luma - threshold;
    return hdrColor * (soft / max(luma, 1.0e-4f));
}

fragment float4 displayFragment(DisplayVertexOut in [[stage_in]],
                               texture2d<float> accumulationTexture [[texture(0)]],
                               constant DisplayUniforms& displayUniforms [[buffer(0)]]) {
    constexpr sampler pointSampler(filter::nearest,
                                   address::clamp_to_edge);
    constexpr sampler linearSampler(filter::linear,
                                    address::clamp_to_edge);
    float2 flippedUv = float2(in.uv.x, 1.0 - in.uv.y);
    float4 accum = accumulationTexture.sample(pointSampler, flippedUv);

    if (accum.a <= 0.0f) {
        return float4(0.0);
    }

    float exposureScale = pow(2.0f, displayUniforms.exposure);
    float3 color = max(accum.rgb, float3(0.0f)) * exposureScale;

    if (displayUniforms.bloomEnabled != 0u && displayUniforms.bloomIntensity > 0.0f) {
        float radius = max(displayUniforms.bloomRadius, 0.0f);
        if (radius > 0.0f) {
            float2 texel = float2(1.0f / max(float(accumulationTexture.get_width()), 1.0f),
                                  1.0f / max(float(accumulationTexture.get_height()), 1.0f));
            float threshold = max(displayUniforms.bloomThreshold, 0.0f);
            constexpr float2 offsets[8] = {
                float2(-1.0f, 0.0f), float2(1.0f, 0.0f), float2(0.0f, -1.0f), float2(0.0f, 1.0f),
                float2(-1.0f, -1.0f), float2(1.0f, -1.0f), float2(-1.0f, 1.0f), float2(1.0f, 1.0f)
            };
            constexpr float weights[9] = {
                0.24f, 0.12f, 0.12f, 0.12f, 0.12f, 0.07f, 0.07f, 0.07f, 0.07f
            };

            float3 bloom = weights[0] * extractBloom(color, threshold);
            for (uint i = 0u; i < 8u; ++i) {
                float2 tapUv = flippedUv + offsets[i] * texel * radius;
                float3 tapHdr = max(accumulationTexture.sample(linearSampler, tapUv).rgb, float3(0.0f));
                tapHdr *= exposureScale;
                bloom += weights[i + 1u] * extractBloom(tapHdr, threshold);
            }
            color += bloom * displayUniforms.bloomIntensity;
        }
    }

    switch (displayUniforms.tonemapMode) {
        case 2:
            color = (displayUniforms.acesVariant == 0) ? ACESFitted(color) : ACESSimple(color);
            break;
        case 3:
            color = tonemapReinhard(color, displayUniforms.reinhardWhite);
            break;
        case 4:
            color = tonemapHable(color);
            break;
        default:
            color = clamp(color, float3(0.0f), float3(1.0f));
            break;
    }
    const float gamma = 1.0f / 2.2f;
    color = pow(color, float3(gamma));
    return float4(color, 1.0);
}

vertex DisplayVertexOut displayVertex(uint vertexID [[vertex_id]]) {
    constexpr float2 positions[6] = {
        {-1.0, -1.0},
        { 1.0, -1.0},
        {-1.0,  1.0},
        { 1.0, -1.0},
        { 1.0,  1.0},
        {-1.0,  1.0},
    };

    constexpr float2 uvs[6] = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0},
        {0.0, 1.0},
    };

    DisplayVertexOut out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    out.uv = uvs[vertexID];
    return out;
}
