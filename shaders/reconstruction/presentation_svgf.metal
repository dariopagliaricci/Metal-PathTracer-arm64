kernel void pathtracePresentKernel(texture2d<float, access::read> radianceTexture [[texture(0)]],
                                   texture2d<uint, access::read> sampleCountTexture [[texture(1)]],
                                   texture2d<float, access::write> outputTexture [[texture(2)]],
                                   texture2d<float, access::read> directRadianceTexture [[texture(3)]],
                                   texture2d<float, access::read> indirectRadianceTexture [[texture(4)]],
                                   texture2d<float, access::read> radianceMomentTexture [[texture(5)]],
                                   texture2d<float, access::write> directOutputTexture [[texture(6)]],
                                   texture2d<float, access::write> indirectOutputTexture [[texture(7)]],
                                   texture2d<float, access::write> reconstructionMetricTexture [[texture(8)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }

    float3 sum = radianceTexture.read(gid).xyz;
    uint count = sampleCountTexture.read(gid).x;
    float sampleCount = static_cast<float>(count);
    float3 average = (count > 0u) ? (sum / sampleCount) : float3(0.0f, 0.0f, 0.0f);

    outputTexture.write(float4(average, sampleCount), gid);
    if (gid.x < directOutputTexture.get_width() && gid.y < directOutputTexture.get_height() &&
        gid.x < indirectOutputTexture.get_width() && gid.y < indirectOutputTexture.get_height() &&
        gid.x < reconstructionMetricTexture.get_width() && gid.y < reconstructionMetricTexture.get_height()) {
        const float3 directAverage =
            (count > 0u && gid.x < directRadianceTexture.get_width() && gid.y < directRadianceTexture.get_height())
                ? directRadianceTexture.read(gid).xyz / sampleCount
                : float3(0.0f);
        const float3 indirectAverage =
            (count > 0u && gid.x < indirectRadianceTexture.get_width() && gid.y < indirectRadianceTexture.get_height())
                ? indirectRadianceTexture.read(gid).xyz / sampleCount
                : float3(0.0f);
        float variance = 0.0f;
        if (count > 1u &&
            gid.x < radianceMomentTexture.get_width() &&
            gid.y < radianceMomentTexture.get_height()) {
            const float meanLuma = luminance_rgb(max(average, float3(0.0f)));
            const float meanLuma2 = max(radianceMomentTexture.read(gid).x / sampleCount, 0.0f);
            variance = max(meanLuma2 - meanLuma * meanLuma, 0.0f);
        }
        const float sampleConfidence = clamp(log2(sampleCount + 1.0f) / 8.0f, 0.0f, 1.0f);
        directOutputTexture.write(float4(max(directAverage, float3(0.0f)), sampleCount), gid);
        indirectOutputTexture.write(float4(max(indirectAverage, float3(0.0f)), sampleCount), gid);
        reconstructionMetricTexture.write(float4(variance, sampleConfidence, 0.0f, sampleCount), gid);
    }
}

struct MlReconstructionTinyModel {
    uint magic;
    uint version;
    uint featureCount;
    uint outputCount;
    float confidenceBias;
    float varianceScale;
    uint flags;
    uint reserved;
    float weights[36];
};

inline float ml_tiny_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-clamp(x, -16.0f, 16.0f)));
}

kernel void mlReconstructionTinyKernel(constant MlReconstructionTinyModel& model [[buffer(0)]],
                                       texture2d<float, access::read> sourceTexture [[texture(0)]],
                                       texture2d<float, access::read> albedoTexture [[texture(1)]],
                                       texture2d<float, access::read> normalTexture [[texture(2)]],
                                       texture2d<float, access::read> positionTexture [[texture(3)]],
                                       texture2d<float, access::read> motionTexture [[texture(4)]],
                                       texture2d<float, access::read_write> reconstructionMetricTexture [[texture(5)]],
                                       texture2d<float, access::write> outputTexture [[texture(6)]],
                                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height() ||
        gid.x >= sourceTexture.get_width() || gid.y >= sourceTexture.get_height()) {
        return;
    }

    const float4 source = sourceTexture.read(gid);
    float4 metric = float4(0.0f, 0.0f, 0.0f, source.w);
    if (gid.x < reconstructionMetricTexture.get_width() &&
        gid.y < reconstructionMetricTexture.get_height()) {
        metric = reconstructionMetricTexture.read(gid);
    }

    if (model.magic != 0x4d4c5231u || model.version != 1u ||
        model.featureCount != 8u || model.outputCount != 4u) {
        outputTexture.write(float4(max(source.xyz, float3(0.0f)), source.w), gid);
        if (gid.x < reconstructionMetricTexture.get_width() &&
            gid.y < reconstructionMetricTexture.get_height()) {
            reconstructionMetricTexture.write(float4(metric.x, 0.0f, 0.0f, metric.w), gid);
        }
        return;
    }

    const float3 albedo =
        (gid.x < albedoTexture.get_width() && gid.y < albedoTexture.get_height())
            ? clamp(albedoTexture.read(gid).xyz, float3(0.0f), float3(1.0f))
            : float3(0.0f);
    const float3 normal =
        (gid.x < normalTexture.get_width() && gid.y < normalTexture.get_height())
            ? normalize(normalTexture.read(gid).xyz * 2.0f - 1.0f)
            : float3(0.0f, 0.0f, 1.0f);
    const float depth =
        (gid.x < positionTexture.get_width() && gid.y < positionTexture.get_height())
            ? max(positionTexture.read(gid).w, 0.0f)
            : 0.0f;
    const float2 motion =
        (gid.x < motionTexture.get_width() && gid.y < motionTexture.get_height())
            ? motionTexture.read(gid).xy
            : float2(0.0f);

    const float variance = max(metric.x, 0.0f);
    const float inputConfidence = clamp(metric.y, 0.0f, 1.0f);
    const float luma = luminance_rgb(max(source.xyz, float3(0.0f)));
    float features[8] = {
        source.x,
        source.y,
        source.z,
        luminance_rgb(albedo),
        normal.z * 0.5f + 0.5f,
        clamp(log2(depth + 1.0f) / 16.0f, 0.0f, 1.0f),
        clamp(log2(variance + 1.0f) / 8.0f, 0.0f, 1.0f),
        clamp(inputConfidence - length(motion) * 0.25f + luma * 0.02f, 0.0f, 1.0f),
    };

    float raw[4] = {model.weights[8], model.weights[17], model.weights[26], model.weights[35]};
    for (uint o = 0u; o < 4u; ++o) {
        const uint base = o * 9u;
        raw[o] = model.weights[base + 8u];
        for (uint f = 0u; f < 8u; ++f) {
            raw[o] += features[f] * model.weights[base + f];
        }
    }

    const float3 inferred = max(float3(raw[0], raw[1], raw[2]), float3(0.0f));
    const float confidence =
        clamp(ml_tiny_sigmoid(raw[3] + model.confidenceBias) *
                  inputConfidence *
                  exp(-max(model.varianceScale, 0.0f) * variance),
              0.0f,
              1.0f);
    const float validity =
        all(inferred == inferred) && confidence > 0.0f && source.w > 0.0f ? 1.0f : 0.0f;
    const float blend = clamp(confidence * validity, 0.0f, 1.0f);
    const float3 reconstructed = mix(max(source.xyz, float3(0.0f)), inferred, blend);

    outputTexture.write(float4(reconstructed, source.w), gid);
    if (gid.x < reconstructionMetricTexture.get_width() &&
        gid.y < reconstructionMetricTexture.get_height()) {
        reconstructionMetricTexture.write(float4(variance, confidence, validity, metric.w), gid);
    }
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
                                      texture2d<float, access::read> previousMaterialHistoryTexture [[texture(11)]],
                                      texture2d<float, access::write> materialHistoryTexture [[texture(12)]],
                                      texture2d<float, access::write> reconstructionMetricTexture [[texture(13)]],
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
    width = min(width, previousMaterialHistoryTexture.get_width());
    width = min(width, materialHistoryTexture.get_width());
    width = min(width, reconstructionMetricTexture.get_width());
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
    height = min(height, previousMaterialHistoryTexture.get_height());
    height = min(height, materialHistoryTexture.get_height());
    height = min(height, reconstructionMetricTexture.get_height());
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
    const float materialType = isfinite(materialFeatures.w) ? materialFeatures.w : -1.0f;
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
    const float4 previousMaterialFeatures = previousMaterialHistoryTexture.read(historyPixel);
    bool historyAccepted = historyReprojected || uniforms.reprojectionEnabled == 0u;
    if (historyAccepted && uniforms.reprojectionEnabled != 0u && currentGuideValid) {
        const float3 previousNormal = safe_normalize(previousGuide.xyz);
        const float currentPreviousDistance = length(currentPosition.xyz - uniforms.previousCameraOrigin.xyz);
        const float previousDistance = previousGuide.w;
        const float depthRel = fabs(currentPreviousDistance - previousDistance) /
                               max(max(currentPreviousDistance, previousDistance), 1.0e-4f);
        const float normalAgreement = dot(currentNormal, previousNormal);
        const float previousRoughness =
            isfinite(previousMaterialFeatures.x) ? clamp(previousMaterialFeatures.x, 0.0f, 1.0f) : materialRoughness;
        const float previousTransmission =
            isfinite(previousMaterialFeatures.y) ? clamp(previousMaterialFeatures.y, 0.0f, 1.0f) : materialTransmission;
        const float previousSpecular =
            isfinite(previousMaterialFeatures.z) ? max(previousMaterialFeatures.z, 0.0f) : materialSpecular;
        const float previousMaterialType =
            isfinite(previousMaterialFeatures.w) ? previousMaterialFeatures.w : materialType;
        const float normalThreshold = mix(0.92f, 0.78f, materialRoughness);
        const float depthThreshold = mix(0.04f, 0.10f, materialRoughness) *
                                     (materialTransmission > 0.05f ? 0.6f : 1.0f);
        const float specularHistoryPenalty = clamp(materialSpecular * (1.0f - materialRoughness), 0.0f, 0.5f);
        const float roughnessDelta = fabs(materialRoughness - previousRoughness);
        const float transmissionDelta = fabs(materialTransmission - previousTransmission);
        const float specularDelta = fabs(materialSpecular - previousSpecular);
        const bool materialCompatible =
            fabs(materialType - previousMaterialType) <= 0.5f &&
            roughnessDelta <= 0.18f &&
            transmissionDelta <= 0.12f &&
            specularDelta <= 0.25f;
        historyAccepted =
            previousGuide.w > 0.0f &&
            all(isfinite(previousGuide)) &&
            all(isfinite(previousMaterialFeatures)) &&
            depthRel <= depthThreshold &&
            normalAgreement >= normalThreshold &&
            materialCompatible &&
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
    const float disocclusion = historyAccepted ? 0.0f : 1.0f;
    const float sampleConfidence = clamp(sampleCount / 64.0f, 0.0f, 1.0f);
    const float historyConfidence = clamp(historyLength / 32.0f, 0.0f, 1.0f);
    motionVectorTexture.write(float4(motionVector, disocclusion, motionPixels), pixel);
    reconstructionMetricTexture.write(float4(variance,
                                             max(sampleConfidence, historyConfidence),
                                             disocclusion,
                                             sampleCount),
                                      pixel);
    materialHistoryTexture.write(materialFeatures, pixel);
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
                                   texture2d<float, access::read> positionTexture [[texture(5)]],
                                   texture2d<float, access::read> materialFeatureTexture [[texture(6)]],
                                   uint gid [[thread_position_in_grid]]) {
    uint width = uniforms.width;
    width = min(width, sourceTexture.get_width());
    width = min(width, albedoTexture.get_width());
    width = min(width, normalTexture.get_width());
    width = min(width, outputTexture.get_width());
    width = min(width, momentsTexture.get_width());
    width = min(width, positionTexture.get_width());
    width = min(width, materialFeatureTexture.get_width());
    uint height = uniforms.height;
    height = min(height, sourceTexture.get_height());
    height = min(height, albedoTexture.get_height());
    height = min(height, normalTexture.get_height());
    height = min(height, outputTexture.get_height());
    height = min(height, momentsTexture.get_height());
    height = min(height, positionTexture.get_height());
    height = min(height, materialFeatureTexture.get_height());
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
    const float4 centerPosition = positionTexture.read(pixel);
    const float4 centerMaterialFeatures = materialFeatureTexture.read(pixel);
    const float centerRoughness = isfinite(centerMaterialFeatures.x)
        ? clamp(centerMaterialFeatures.x, 0.0f, 1.0f)
        : 1.0f;
    const float centerSpecular = isfinite(centerMaterialFeatures.z)
        ? max(centerMaterialFeatures.z, 0.0f)
        : 0.0f;
    const float centerMaterialType = isfinite(centerMaterialFeatures.w)
        ? centerMaterialFeatures.w
        : -1.0f;
    const bool centerGuideValid =
        centerPosition.w > 0.0f &&
        all(isfinite(centerPosition.xyz)) &&
        dot(centerNormal, centerNormal) > 0.0f;
    const float centerViewDepth = centerGuideValid
        ? length(centerPosition.xyz - uniforms.currentCameraOrigin.xyz)
        : 0.0f;
    const float centerLuma = luminance_rgb(centerColor);
    const float centerAlpha = isfinite(centerSource.w) ? max(centerSource.w, 0.0f) : 0.0f;
    const float centerSampleCount = max(centerAlpha, 1.0f);
    const float4 centerMoments = momentsTexture.read(pixel);
    const float centerVariance = max(centerMoments.w, 0.0f);
    const float centerHistoryLength = max(centerMoments.z, 0.0f);
    const float lowSppFactor = clamp(centerSampleCount / 128.0f, 0.0f, 1.0f);
    const float guideSigmaScale = mix(0.35f, 1.0f, lowSppFactor);
    const float lumaSigmaScale = mix(0.25f, 1.0f, lowSppFactor);
    const int step = int(max(uniforms.step, 1u));
    const float wideStepScale = sqrt(float(step));
    const float glossyScale = mix(0.45f, 1.0f, centerRoughness) *
                              (centerSpecular > 0.08f ? 0.65f : 1.0f);
    const float guideScale = guideSigmaScale * wideStepScale / max(glossyScale, 0.2f);
    const float normalSigma = max(uniforms.normalSigma * guideScale, 1.0e-4f);
    const float albedoSigma = max(uniforms.albedoSigma * guideScale, 1.0e-4f);
    const float depthSigma = max(96.0f * guideScale, 1.0e-4f);
    const float materialSigma = max(8.0f * guideScale, 1.0e-4f);
    const float lumaSigma = max(uniforms.luminanceSigma /
                                    sqrt(max(centerVariance * centerSampleCount, 1.0e-6f)),
                                1.0e-4f) * lumaSigmaScale;
    constexpr float kernelWeights[5] = {1.0f / 16.0f, 4.0f / 16.0f, 6.0f / 16.0f, 4.0f / 16.0f, 1.0f / 16.0f};

    float3 sum = float3(0.0f);
    float weightSum = 0.0f;
    float kernelWeightSum = 0.0f;
    float weightedAlbedoDeltaSum = 0.0f;
    float weightedLumaDeltaSum = 0.0f;
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
            const float4 samplePosition = positionTexture.read(samplePixel);
            const float4 sampleMaterialFeatures = materialFeatureTexture.read(samplePixel);
            const bool sampleGuideValid =
                samplePosition.w > 0.0f &&
                all(isfinite(samplePosition.xyz)) &&
                dot(sampleNormal, sampleNormal) > 0.0f;
            const float sampleViewDepth = sampleGuideValid
                ? length(samplePosition.xyz - uniforms.currentCameraOrigin.xyz)
                : centerViewDepth;
            const float sampleMaterialType = isfinite(sampleMaterialFeatures.w)
                ? sampleMaterialFeatures.w
                : centerMaterialType;

            const float kernelWeight = kernelWeights[ox + 2] * kernelWeights[oy + 2];
            const float normalDelta = max(1.0f - dot(centerNormal, sampleNormal), 0.0f);
            const float albedoDelta = length(centerAlbedo - sampleAlbedo);
            const float lumaDelta = fabs(centerLuma - luminance_rgb(sampleColor));
            const float depthDelta =
                (centerGuideValid && sampleGuideValid)
                    ? fabs(centerViewDepth - sampleViewDepth) /
                          max(max(centerViewDepth, sampleViewDepth), 1.0e-4f)
                    : (centerGuideValid == sampleGuideValid ? 0.0f : 1.0f);
            const float materialDelta =
                (fabs(centerMaterialType - sampleMaterialType) > 0.5f) ? 1.0f : 0.0f;
            const float edgeWeight =
                exp(-(normalDelta * normalSigma +
                      albedoDelta * albedoSigma +
                      depthDelta * depthSigma +
                      materialDelta * materialSigma +
                      lumaDelta * lumaSigma));
            const float w = kernelWeight * edgeWeight;
            sum += sampleColor * w;
            weightSum += w;
            kernelWeightSum += kernelWeight;
            weightedAlbedoDeltaSum += kernelWeight * albedoDelta;
            weightedLumaDeltaSum += kernelWeight * lumaDelta;
        }
    }

    float3 filtered = (weightSum > 1.0e-6f) ? (sum / weightSum) : centerColor;
    if (!all(isfinite(filtered))) {
        filtered = centerColor;
    }
    const float neighborhoodConfidence = clamp(weightSum / max(kernelWeightSum, 1.0e-6f), 0.0f, 1.0f);
    const float sampleConfidence = smoothstep(16.0f, 256.0f, centerSampleCount);
    const float historyConfidence = smoothstep(4.0f, 48.0f, centerHistoryLength);
    const float reconstructionConfidence = max(sampleConfidence, historyConfidence);
    const float relativeNoise =
        sqrt(max(centerVariance, 0.0f)) / max(abs(centerLuma) + 0.03f, 0.03f);
    const float noiseDemand = smoothstep(0.015f, 0.18f, relativeNoise);
    const float lowSampleDemand = 1.0f - sampleConfidence;
    const float guideDemand = smoothstep(0.04f, 0.35f, neighborhoodConfidence);
    const float averageAlbedoDelta = weightedAlbedoDeltaSum / max(kernelWeightSum, 1.0e-6f);
    const float averageLumaDelta = weightedLumaDeltaSum / max(kernelWeightSum, 1.0e-6f);
    const float textureDetailProtection =
        smoothstep(0.025f, 0.18f, averageAlbedoDelta) *
        smoothstep(0.01f, 0.14f, averageLumaDelta);
    const float glossyWidePassProtection =
        (step >= 4)
            ? clamp((1.0f - centerRoughness) * centerSpecular * 4.0f, 0.0f, 1.0f)
            : 0.0f;
    float filterStrength = max(noiseDemand, lowSampleDemand * 0.75f) * guideDemand;
    filterStrength *= mix(1.0f, 0.45f, textureDetailProtection * reconstructionConfidence);
    filterStrength *= mix(1.0f, 0.55f, glossyWidePassProtection);
    filterStrength = clamp(filterStrength, 0.0f, 1.0f);
    filtered = mix(centerColor, filtered, filterStrength);
    outputTexture.write(float4(max(filtered, float3(0.0f)), centerAlpha), pixel);
}

kernel void svgfCompositeSplitKernel(constant SvgfDenoiseUniforms& uniforms [[buffer(0)]],
                                     texture2d<float, access::read> directRawTexture [[texture(0)]],
                                     texture2d<float, access::read> indirectRawTexture [[texture(1)]],
                                     texture2d<float, access::read> directFilteredTexture [[texture(2)]],
                                     texture2d<float, access::read> indirectFilteredTexture [[texture(3)]],
                                     texture2d<float, access::write> outputTexture [[texture(4)]],
                                     uint gid [[thread_position_in_grid]]) {
    uint width = uniforms.width;
    width = min(width, directRawTexture.get_width());
    width = min(width, indirectRawTexture.get_width());
    width = min(width, directFilteredTexture.get_width());
    width = min(width, indirectFilteredTexture.get_width());
    width = min(width, outputTexture.get_width());
    uint height = uniforms.height;
    height = min(height, directRawTexture.get_height());
    height = min(height, indirectRawTexture.get_height());
    height = min(height, directFilteredTexture.get_height());
    height = min(height, indirectFilteredTexture.get_height());
    height = min(height, outputTexture.get_height());
    const uint pixelCount = width * height;
    if (gid >= pixelCount || width == 0u || height == 0u) {
        return;
    }
    const uint2 pixel = uint2(gid % width, gid / width);
    const float4 directRaw = directRawTexture.read(pixel);
    const float4 indirectRaw = indirectRawTexture.read(pixel);
    const float3 directColor = uniforms.denoiseDirect != 0u
        ? directFilteredTexture.read(pixel).xyz
        : directRaw.xyz;
    const float3 indirectColor = uniforms.denoiseIndirect != 0u
        ? indirectFilteredTexture.read(pixel).xyz
        : indirectRaw.xyz;
    const float sampleCount = max(max(directRaw.w, indirectRaw.w), 0.0f);
    outputTexture.write(float4(max(directColor + indirectColor, float3(0.0f)), sampleCount), pixel);
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
                                 texture2d<float, access::write> svgfMaterialHistoryTextureA [[texture(12)]],
                                 texture2d<float, access::write> svgfMaterialHistoryTextureB [[texture(13)]],
                                 texture2d<float, access::write> directRadianceTexture [[texture(14)]],
                                 texture2d<float, access::write> indirectRadianceTexture [[texture(15)]],
                                 texture2d<float, access::write> radianceMomentTexture [[texture(16)]],
                                 texture2d<float, access::write> directPresentTexture [[texture(17)]],
                                 texture2d<float, access::write> indirectPresentTexture [[texture(18)]],
                                 texture2d<float, access::write> reconstructionMetricTexture [[texture(19)]],
                                 texture2d<float, access::write> svgfIndirectMomentsTextureA [[texture(20)]],
                                 texture2d<float, access::write> svgfIndirectMomentsTextureB [[texture(21)]],
                                 texture2d<float, access::write> svgfIndirectColorTextureA [[texture(22)]],
                                 texture2d<float, access::write> svgfIndirectColorTextureB [[texture(23)]],
                                 texture2d<float, access::write> svgfDirectComponentTexture [[texture(24)]],
                                 texture2d<float, access::write> svgfIndirectComponentTexture [[texture(25)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= outputTexture.get_width() || gid.y >= outputTexture.get_height()) {
        return;
    }

    radianceTexture.write(float4(0.0f), gid);
    sampleCountTexture.write(0u, gid);
    outputTexture.write(float4(0.0f), gid);
    if (gid.x < directRadianceTexture.get_width() && gid.y < directRadianceTexture.get_height()) {
        directRadianceTexture.write(float4(0.0f), gid);
    }
    if (gid.x < indirectRadianceTexture.get_width() && gid.y < indirectRadianceTexture.get_height()) {
        indirectRadianceTexture.write(float4(0.0f), gid);
    }
    if (gid.x < radianceMomentTexture.get_width() && gid.y < radianceMomentTexture.get_height()) {
        radianceMomentTexture.write(float4(0.0f), gid);
    }
    if (gid.x < directPresentTexture.get_width() && gid.y < directPresentTexture.get_height()) {
        directPresentTexture.write(float4(0.0f), gid);
    }
    if (gid.x < indirectPresentTexture.get_width() && gid.y < indirectPresentTexture.get_height()) {
        indirectPresentTexture.write(float4(0.0f), gid);
    }
    if (gid.x < reconstructionMetricTexture.get_width() && gid.y < reconstructionMetricTexture.get_height()) {
        reconstructionMetricTexture.write(float4(0.0f), gid);
    }
    if (gid.x < svgfIndirectMomentsTextureA.get_width() && gid.y < svgfIndirectMomentsTextureA.get_height()) {
        svgfIndirectMomentsTextureA.write(float4(0.0f), gid);
    }
    if (gid.x < svgfIndirectMomentsTextureB.get_width() && gid.y < svgfIndirectMomentsTextureB.get_height()) {
        svgfIndirectMomentsTextureB.write(float4(0.0f), gid);
    }
    if (gid.x < svgfIndirectColorTextureA.get_width() && gid.y < svgfIndirectColorTextureA.get_height()) {
        svgfIndirectColorTextureA.write(float4(0.0f), gid);
    }
    if (gid.x < svgfIndirectColorTextureB.get_width() && gid.y < svgfIndirectColorTextureB.get_height()) {
        svgfIndirectColorTextureB.write(float4(0.0f), gid);
    }
    if (gid.x < svgfDirectComponentTexture.get_width() && gid.y < svgfDirectComponentTexture.get_height()) {
        svgfDirectComponentTexture.write(float4(0.0f), gid);
    }
    if (gid.x < svgfIndirectComponentTexture.get_width() && gid.y < svgfIndirectComponentTexture.get_height()) {
        svgfIndirectComponentTexture.write(float4(0.0f), gid);
    }
    if (gid.x < positionTexture.get_width() && gid.y < positionTexture.get_height()) {
        positionTexture.write(float4(0.0f), gid);
    }
    if (gid.x < materialFeatureTexture.get_width() && gid.y < materialFeatureTexture.get_height()) {
        materialFeatureTexture.write(float4(0.0f), gid);
    }
    if (gid.x < motionVectorTexture.get_width() && gid.y < motionVectorTexture.get_height()) {
        motionVectorTexture.write(float4(0.0f), gid);
    }
    if (gid.x < svgfMaterialHistoryTextureA.get_width() && gid.y < svgfMaterialHistoryTextureA.get_height()) {
        svgfMaterialHistoryTextureA.write(float4(0.0f), gid);
    }
    if (gid.x < svgfMaterialHistoryTextureB.get_width() && gid.y < svgfMaterialHistoryTextureB.get_height()) {
        svgfMaterialHistoryTextureB.write(float4(0.0f), gid);
    }
    (void)svgfMomentsTextureA;
    (void)svgfMomentsTextureB;
    (void)svgfGuideTextureA;
    (void)svgfGuideTextureB;
    (void)svgfColorTextureA;
    (void)svgfColorTextureB;
}
