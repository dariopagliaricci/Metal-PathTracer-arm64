kernel void pathtracePresentKernel(texture2d<float, access::read> radianceTexture [[texture(0)]],
                                   texture2d<uint, access::read> sampleCountTexture [[texture(1)]],
                                   texture2d<float, access::write> outputTexture [[texture(2)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }

    float3 sum = radianceTexture.read(gid).xyz;
    uint count = sampleCountTexture.read(gid).x;
    float sampleCount = static_cast<float>(count);
    float3 average = (count > 0u) ? (sum / sampleCount) : float3(0.0f, 0.0f, 0.0f);

    outputTexture.write(float4(average, sampleCount), gid);
}

kernel void svgfTemporalMomentsKernel(constant SvgfDenoiseUniforms& uniforms [[buffer(0)]],
                                      texture2d<float, access::read> sourceTexture [[texture(0)]],
                                      texture2d<float, access::read> previousMomentsTexture [[texture(1)]],
                                      texture2d<float, access::write> momentsTexture [[texture(2)]],
                                      texture2d<float, access::read> positionTexture [[texture(3)]],
                                      texture2d<float, access::read> normalTexture [[texture(4)]],
                                      texture2d<float, access::read> previousGuideTexture [[texture(5)]],
                                      texture2d<float, access::write> guideTexture [[texture(6)]],
                                      texture2d<float, access::read> previousColorTexture [[texture(7)]],
                                      texture2d<float, access::write> colorTexture [[texture(8)]],
                                      texture2d<float, access::read> materialFeatureTexture [[texture(9)]],
                                      texture2d<float, access::write> motionVectorTexture [[texture(10)]],
                                      uint gid [[thread_position_in_grid]]) {
    uint width = uniforms.width;
    width = min(width, sourceTexture.get_width());
    width = min(width, previousMomentsTexture.get_width());
    width = min(width, momentsTexture.get_width());
    width = min(width, positionTexture.get_width());
    width = min(width, normalTexture.get_width());
    width = min(width, previousGuideTexture.get_width());
    width = min(width, guideTexture.get_width());
    width = min(width, previousColorTexture.get_width());
    width = min(width, colorTexture.get_width());
    width = min(width, materialFeatureTexture.get_width());
    width = min(width, motionVectorTexture.get_width());
    uint height = uniforms.height;
    height = min(height, sourceTexture.get_height());
    height = min(height, previousMomentsTexture.get_height());
    height = min(height, momentsTexture.get_height());
    height = min(height, positionTexture.get_height());
    height = min(height, normalTexture.get_height());
    height = min(height, previousGuideTexture.get_height());
    height = min(height, guideTexture.get_height());
    height = min(height, previousColorTexture.get_height());
    height = min(height, colorTexture.get_height());
    height = min(height, materialFeatureTexture.get_height());
    height = min(height, motionVectorTexture.get_height());
    const uint pixelCount = width * height;
    if (gid >= pixelCount || width == 0u || height == 0u) {
        return;
    }

    const uint2 pixel = uint2(gid % width, gid / width);
    const float4 centerSource = sourceTexture.read(pixel);
    const float3 centerColor = all(isfinite(centerSource.xyz))
        ? max(centerSource.xyz, float3(0.0f))
        : float3(0.0f);
    const float centerLuma = luminance_rgb(centerColor);
    const float sampleCount = isfinite(centerSource.w) ? max(centerSource.w, 1.0f) : 1.0f;
    const float4 currentPosition = positionTexture.read(pixel);
    const float4 materialFeatures = materialFeatureTexture.read(pixel);
    const float materialRoughness = isfinite(materialFeatures.x) ? clamp(materialFeatures.x, 0.0f, 1.0f) : 1.0f;
    const float materialTransmission = isfinite(materialFeatures.y) ? clamp(materialFeatures.y, 0.0f, 1.0f) : 0.0f;
    const float materialSpecular = isfinite(materialFeatures.z) ? max(materialFeatures.z, 0.0f) : 0.0f;
    const float3 currentNormalRaw = normalTexture.read(pixel).xyz * 2.0f - 1.0f;
    const float3 currentNormal = all(isfinite(currentNormalRaw))
        ? safe_normalize(currentNormalRaw)
        : float3(0.0f);
    const bool currentGuideValid =
        currentPosition.w > 0.0f &&
        all(isfinite(currentPosition.xyz)) &&
        dot(currentNormal, currentNormal) > 0.0f;

    float localLumaSum = 0.0f;
    float localLumaSqSum = 0.0f;
    float localWeightSum = 0.0f;
    float localLumaMin = centerLuma;
    float localLumaMax = centerLuma;
    float3 localColorMin = centerColor;
    float3 localColorMax = centerColor;
    float3 localColorSum = float3(0.0f);
    for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
            const int sx = clamp(int(pixel.x) + ox, 0, int(width) - 1);
            const int sy = clamp(int(pixel.y) + oy, 0, int(height) - 1);
            const float3 sampleColor = sourceTexture.read(uint2(uint(sx), uint(sy))).xyz;
            if (!all(isfinite(sampleColor))) {
                continue;
            }
            const float luma = luminance_rgb(max(sampleColor, float3(0.0f)));
            const float3 positiveSampleColor = max(sampleColor, float3(0.0f));
            localLumaSum += luma;
            localLumaSqSum += luma * luma;
            localColorSum += positiveSampleColor;
            localWeightSum += 1.0f;
            localLumaMin = min(localLumaMin, luma);
            localLumaMax = max(localLumaMax, luma);
            localColorMin = min(localColorMin, positiveSampleColor);
            localColorMax = max(localColorMax, positiveSampleColor);
        }
    }
    const float localMean = (localWeightSum > 0.0f) ? (localLumaSum / localWeightSum) : centerLuma;
    const float localSecond = (localWeightSum > 0.0f) ? (localLumaSqSum / localWeightSum) : centerLuma * centerLuma;
    const float localVariance = max(localSecond - localMean * localMean, 0.0f);
    const float3 localColorMean = (localWeightSum > 0.0f) ? (localColorSum / localWeightSum) : centerColor;

    uint2 historyPixel = pixel;
    bool historyReprojected = false;
    if (uniforms.reprojectionEnabled != 0u && currentGuideValid) {
        const float3 previousOrigin = uniforms.previousCameraOrigin.xyz;
        const float3 previousLowerLeft = uniforms.previousLowerLeftCorner.xyz;
        const float3 previousHorizontal = uniforms.previousHorizontal.xyz;
        const float3 previousVertical = uniforms.previousVertical.xyz;
        const float3 previousPlaneNormal = safe_normalize(cross(previousHorizontal, previousVertical));
        const float3 toPoint = currentPosition.xyz - previousOrigin;
        const float planeDenom = dot(toPoint, previousPlaneNormal);
        const float planeNumer = dot(previousLowerLeft - previousOrigin, previousPlaneNormal);
        if (isfinite(planeDenom) && isfinite(planeNumer) && fabs(planeDenom) > 1.0e-6f) {
            const float planeT = planeNumer / planeDenom;
            const float3 projected = previousOrigin + toPoint * planeT;
            const float3 rel = projected - previousLowerLeft;
            const float hLen2 = max(dot(previousHorizontal, previousHorizontal), 1.0e-12f);
            const float vLen2 = max(dot(previousVertical, previousVertical), 1.0e-12f);
            const float prevU = dot(rel, previousHorizontal) / hLen2;
            const float prevV = dot(rel, previousVertical) / vLen2;
            if (isfinite(prevU) && isfinite(prevV) &&
                prevU >= 0.0f && prevU <= 1.0f &&
                prevV >= 0.0f && prevV <= 1.0f) {
                const uint hx = min(uint(prevU * float(width)), width - 1u);
                const uint hy = min(uint((1.0f - prevV) * float(height)), height - 1u);
                historyPixel = uint2(hx, hy);
                historyReprojected = true;
            }
        }
    }

    const float motionPixels = length(float2(historyPixel) - float2(pixel));
    const float2 motionVector = float2(historyPixel) - float2(pixel);
    const float4 previous = previousMomentsTexture.read(historyPixel);
    const float4 previousColorSource = previousColorTexture.read(historyPixel);
    const float4 previousGuide = previousGuideTexture.read(historyPixel);
    bool historyAccepted = historyReprojected || uniforms.reprojectionEnabled == 0u;
    if (historyAccepted && uniforms.reprojectionEnabled != 0u && currentGuideValid) {
        const float3 previousNormal = safe_normalize(previousGuide.xyz);
        const float currentPreviousDistance = length(currentPosition.xyz - uniforms.previousCameraOrigin.xyz);
        const float previousDistance = previousGuide.w;
        const float depthRel = fabs(currentPreviousDistance - previousDistance) /
                               max(max(currentPreviousDistance, previousDistance), 1.0e-4f);
        const float normalAgreement = dot(currentNormal, previousNormal);
        const float normalThreshold = mix(0.92f, 0.78f, materialRoughness);
        const float depthThreshold = mix(0.04f, 0.10f, materialRoughness) *
                                     (materialTransmission > 0.05f ? 0.6f : 1.0f);
        const float specularHistoryPenalty = clamp(materialSpecular * (1.0f - materialRoughness), 0.0f, 0.5f);
        historyAccepted =
            previousGuide.w > 0.0f &&
            all(isfinite(previousGuide)) &&
            depthRel <= depthThreshold &&
            normalAgreement >= normalThreshold &&
            specularHistoryPenalty < 0.45f;
    }
    const float previousHistory =
        (uniforms.historyValid != 0u && all(isfinite(previous)) && previous.z > 0.0f)
            ? previous.z
            : 0.0f;
    const float motionHistoryScale =
        (uniforms.reprojectionEnabled != 0u)
            ? clamp(1.0f / (1.0f + motionPixels * 0.08f), 0.15f, 1.0f)
            : 1.0f;
    const float acceptedHistory = historyAccepted ? (previousHistory * motionHistoryScale) : 0.0f;
    const float alpha = (acceptedHistory > 0.0f)
        ? clamp(1.0f / min(sampleCount, acceptedHistory + 1.0f), 0.025f, 0.25f)
        : 1.0f;
    const float localStdDev = sqrt(max(localVariance, 1.0e-6f));
    const float clampSlack = max(2.0f * localStdDev, 0.02f * max(localMean, 1.0f));
    const float historyClampMin = max(localLumaMin - clampSlack, 0.0f);
    const float historyClampMax = max(localLumaMax + clampSlack, historyClampMin);
    const float previousMeanRaw = (acceptedHistory > 0.0f) ? max(previous.x, 0.0f) : centerLuma;
    const float previousMean = clamp(previousMeanRaw, historyClampMin, historyClampMax);
    const float previousVarianceRaw =
        (acceptedHistory > 0.0f) ? max(previous.y - previousMeanRaw * previousMeanRaw, 0.0f) : 0.0f;
    const float previousVariance = min(previousVarianceRaw, max(localVariance * 4.0f, 1.0e-4f));
    const float previousSecond = previousMean * previousMean + previousVariance;
    const float mean = mix(previousMean, centerLuma, alpha);
    const float secondMoment = mix(previousSecond, centerLuma * centerLuma, alpha);
    const float temporalVariance = max(secondMoment - mean * mean, 0.0f);
    const float variance = max(temporalVariance, localVariance / sampleCount);
    const float historyLength = min(acceptedHistory + 1.0f, 64.0f);

    momentsTexture.write(float4(mean, secondMoment, historyLength, variance), pixel);
    const float3 colorSlack = float3(max(localStdDev * 2.0f, max(abs(localMean) * 0.02f, 0.01f)));
    const float3 colorClampMin = max(localColorMin - colorSlack, float3(0.0f));
    const float3 colorClampMax = max(localColorMax + colorSlack, colorClampMin);
    const float3 historyColor = (acceptedHistory > 0.0f && all(isfinite(previousColorSource.xyz)))
        ? clamp(max(previousColorSource.xyz, float3(0.0f)),
                colorClampMin,
                colorClampMax)
        : centerColor;
    float3 temporalColor = mix(historyColor, centerColor, alpha);
    const float lowSppAggression = clamp(1.0f - sampleCount / 128.0f, 0.0f, 1.0f);
    const float bootstrapBlend = (acceptedHistory > 0.0f)
        ? (0.20f * lowSppAggression)
        : (0.65f * lowSppAggression);
    temporalColor = mix(temporalColor, localColorMean, bootstrapBlend);
    colorTexture.write(float4(max(temporalColor, float3(0.0f)), centerSource.w), pixel);
    motionVectorTexture.write(float4(motionVector, historyAccepted ? 1.0f : 0.0f, motionPixels), pixel);
    const float currentPreviousDistance = currentGuideValid
        ? length(currentPosition.xyz - uniforms.currentCameraOrigin.xyz)
        : 0.0f;
    guideTexture.write(float4(currentNormal, currentPreviousDistance), pixel);
}

kernel void svgfAtrousFilterKernel(constant SvgfDenoiseUniforms& uniforms [[buffer(0)]],
                                   texture2d<float, access::read> sourceTexture [[texture(0)]],
                                   texture2d<float, access::read> albedoTexture [[texture(1)]],
                                   texture2d<float, access::read> normalTexture [[texture(2)]],
                                   texture2d<float, access::write> outputTexture [[texture(3)]],
                                   texture2d<float, access::read> momentsTexture [[texture(4)]],
                                   uint gid [[thread_position_in_grid]]) {
    const uint width = min(min(min(min(uniforms.width, sourceTexture.get_width()),
                                  albedoTexture.get_width()),
                              normalTexture.get_width()),
                          min(outputTexture.get_width(), momentsTexture.get_width()));
    const uint height = min(min(min(min(uniforms.height, sourceTexture.get_height()),
                                   albedoTexture.get_height()),
                               normalTexture.get_height()),
                           min(outputTexture.get_height(), momentsTexture.get_height()));
    const uint pixelCount = width * height;
    if (gid >= pixelCount || width == 0u || height == 0u) {
        return;
    }

    const uint2 pixel = uint2(gid % width, gid / width);
    const float4 centerSource = sourceTexture.read(pixel);
    const float3 centerColor = all(isfinite(centerSource.xyz))
        ? max(centerSource.xyz, float3(0.0f))
        : float3(0.0f);
    const float3 centerAlbedoRaw = albedoTexture.read(pixel).xyz;
    const float3 centerAlbedo = all(isfinite(centerAlbedoRaw))
        ? clamp(centerAlbedoRaw, float3(0.0f), float3(64.0f))
        : float3(0.0f);
    const float3 centerNormalRaw = normalTexture.read(pixel).xyz * 2.0f - 1.0f;
    const float3 centerNormal = all(isfinite(centerNormalRaw))
        ? safe_normalize(centerNormalRaw)
        : float3(0.0f);
    const float centerLuma = luminance_rgb(centerColor);
    const float centerAlpha = isfinite(centerSource.w) ? max(centerSource.w, 0.0f) : 0.0f;
    const float centerSampleCount = max(centerAlpha, 1.0f);
    const float centerVariance = max(momentsTexture.read(pixel).w, 0.0f);
    const float lowSppFactor = clamp(centerSampleCount / 128.0f, 0.0f, 1.0f);
    const float guideSigmaScale = mix(0.35f, 1.0f, lowSppFactor);
    const float lumaSigmaScale = mix(0.25f, 1.0f, lowSppFactor);
    const float normalSigma = max(uniforms.normalSigma * guideSigmaScale, 1.0e-4f);
    const float albedoSigma = max(uniforms.albedoSigma * guideSigmaScale, 1.0e-4f);
    const float lumaSigma = max(uniforms.luminanceSigma /
                                    sqrt(max(centerVariance * centerSampleCount, 1.0e-6f)),
                                1.0e-4f) * lumaSigmaScale;
    const int step = int(max(uniforms.step, 1u));
    constexpr float kernelWeights[5] = {1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f};

    float3 sum = float3(0.0f);
    float weightSum = 0.0f;
    for (int oy = -2; oy <= 2; ++oy) {
        for (int ox = -2; ox <= 2; ++ox) {
            const int sx = clamp(int(pixel.x) + ox * step, 0, int(width) - 1);
            const int sy = clamp(int(pixel.y) + oy * step, 0, int(height) - 1);
            const uint2 samplePixel = uint2(uint(sx), uint(sy));
            const float4 sampleSource = sourceTexture.read(samplePixel);
            if (!all(isfinite(sampleSource.xyz))) {
                continue;
            }
            const float3 sampleColor = max(sampleSource.xyz, float3(0.0f));
            const float3 sampleAlbedoRaw = albedoTexture.read(samplePixel).xyz;
            const float3 sampleAlbedo = all(isfinite(sampleAlbedoRaw))
                ? clamp(sampleAlbedoRaw, float3(0.0f), float3(64.0f))
                : centerAlbedo;
            const float3 sampleNormalRaw = normalTexture.read(samplePixel).xyz * 2.0f - 1.0f;
            const float3 sampleNormal = all(isfinite(sampleNormalRaw))
                ? safe_normalize(sampleNormalRaw)
                : centerNormal;

            const float kernelWeight = kernelWeights[ox + 2] * kernelWeights[oy + 2];
            const float normalDelta = max(1.0f - dot(centerNormal, sampleNormal), 0.0f);
            const float albedoDelta = length(centerAlbedo - sampleAlbedo);
            const float lumaDelta = fabs(centerLuma - luminance_rgb(sampleColor));
            const float edgeWeight =
                exp(-(normalDelta * normalSigma +
                      albedoDelta * albedoSigma +
                      lumaDelta * lumaSigma));
            const float w = kernelWeight * edgeWeight;
            sum += sampleColor * w;
            weightSum += w;
        }
    }

    float3 filtered = (weightSum > 1.0e-6f) ? (sum / weightSum) : centerColor;
    if (!all(isfinite(filtered))) {
        filtered = centerColor;
    }
    outputTexture.write(float4(max(filtered, float3(0.0f)), centerAlpha), pixel);
}

kernel void pathtraceClearKernel(texture2d<float, access::write> radianceTexture [[texture(0)]],
                                 texture2d<uint, access::write> sampleCountTexture [[texture(1)]],
                                 texture2d<float, access::write> outputTexture [[texture(2)]],
                                 texture2d<float, access::write> svgfMomentsTextureA [[texture(3)]],
                                 texture2d<float, access::write> svgfMomentsTextureB [[texture(4)]],
                                 texture2d<float, access::write> positionTexture [[texture(5)]],
                                 texture2d<float, access::write> svgfGuideTextureA [[texture(6)]],
                                 texture2d<float, access::write> svgfGuideTextureB [[texture(7)]],
                                 texture2d<float, access::write> svgfColorTextureA [[texture(8)]],
                                 texture2d<float, access::write> svgfColorTextureB [[texture(9)]],
                                 texture2d<float, access::write> materialFeatureTexture [[texture(10)]],
                                 texture2d<float, access::write> motionVectorTexture [[texture(11)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }

    radianceTexture.write(float4(0.0f), gid);
    sampleCountTexture.write(0u, gid);
    outputTexture.write(float4(0.0f), gid);
    if (gid.x < positionTexture.get_width() && gid.y < positionTexture.get_height()) {
        positionTexture.write(float4(0.0f), gid);
    }
    if (gid.x < materialFeatureTexture.get_width() && gid.y < materialFeatureTexture.get_height()) {
        materialFeatureTexture.write(float4(0.0f), gid);
    }
    if (gid.x < motionVectorTexture.get_width() && gid.y < motionVectorTexture.get_height()) {
        motionVectorTexture.write(float4(0.0f), gid);
    }
    (void)svgfMomentsTextureA;
    (void)svgfMomentsTextureB;
    (void)svgfGuideTextureA;
    (void)svgfGuideTextureB;
    (void)svgfColorTextureA;
    (void)svgfColorTextureB;
}
