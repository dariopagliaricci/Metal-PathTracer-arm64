inline uint pcg_hash(uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

inline float rand_uniform(thread uint& state) {
    state = pcg_hash(state);
    return float(state) / 4294967296.0f;
}

inline float3 random_in_unit_sphere(thread uint& state) {
    while (true) {
        float3 p = float3(rand_uniform(state), rand_uniform(state), rand_uniform(state)) * 2.0f - 1.0f;
        if (dot(p, p) < 1.0f) {
            return p;
        }
    }
}

inline float3 random_unit_vector(thread uint& state) {
    return normalize(random_in_unit_sphere(state));
}

inline float2 random_in_unit_disk(thread uint& state) {
    while (true) {
        float2 p = float2(rand_uniform(state), rand_uniform(state)) * 2.0f - 1.0f;
        if (dot(p, p) < 1.0f) {
            return p;
        }
    }
}

inline bool near_zero(const float3 v) {
    const float s = 1e-6f;
    return (fabs(v.x) < s) && (fabs(v.y) < s) && (fabs(v.z) < s);
}

inline float3 linear_srgb_to_acescg(const float3 color) {
    const float3x3 m = float3x3(float3(0.613097f, 0.339523f, 0.047380f),
                                float3(0.070194f, 0.916354f, 0.013452f),
                                float3(0.020615f, 0.109569f, 0.869816f));
    return m * color;
}

inline float3 to_working_space(const float3 color,
                               constant PathtraceUniforms& uniforms) {
    if (uniforms.workingColorSpace == kWorkingColorSpaceACEScg) {
        return linear_srgb_to_acescg(color);
    }
    return color;
}

inline float3 decode_normal_map(float3 sample,
                                float normalScale,
                                bool flipGreen,
                                thread float& outLength) {
    float3 n = sample * 2.0f - 1.0f;
    if (flipGreen) {
        n.y = -n.y;
    }
    n.xy *= normalScale;
    outLength = length(n);
    float xyLen2 = dot(n.xy, n.xy);
    n.z = sqrt(max(1.0f - xyLen2, 0.0f));
    float len2 = dot(n, n);
    if (len2 > 1.0e-12f) {
        n *= rsqrt(len2);
    } else {
        n = float3(0.0f, 0.0f, 1.0f);
    }
    return n;
}

struct RayCone {
    float width;
    float spread;
};

struct PrimaryRayDiff {
    float3 dOdx;
    float3 dOdy;
    float3 dDdx;
    float3 dDdy;
};

inline RayCone make_primary_ray_cone(constant PathtraceUniforms& uniforms) {
    float pixelWorldX = length(uniforms.horizontal) / max(float(uniforms.width), 1.0f);
    float pixelWorldY = length(uniforms.vertical) / max(float(uniforms.height), 1.0f);
    float pixelFootprint = max(max(pixelWorldX, pixelWorldY), 1.0e-6f);
    float3 viewportCenter =
        uniforms.lowerLeftCorner + 0.5f * uniforms.horizontal + 0.5f * uniforms.vertical;
    float focusDistance = length(viewportCenter - uniforms.cameraOrigin);
    RayCone cone;
    cone.width = max(2.0f * uniforms.lensRadius, 0.0f);
    cone.spread = pixelFootprint / max(focusDistance, 1.0e-6f);
    return cone;
}

inline float ray_segment_world_length(const Ray ray, float t) {
    return max(t, 0.0f) * max(length(ray.direction), 1.0e-6f);
}

inline float ray_cone_width_at_distance(const RayCone cone, float distanceWorld) {
    return max(cone.width + cone.spread * max(distanceWorld, 0.0f), 1.0e-7f);
}

inline float ray_cone_lod_from_footprint(const MaterialTextureInfo textureInfo,
                                         float uvPerWorld,
                                         float footprintWorld) {
    if (textureInfo.width == 0u || textureInfo.height == 0u) {
        return 0.0f;
    }
    if (textureInfo.mipCount <= 1u || uvPerWorld <= 0.0f || footprintWorld <= 0.0f) {
        return 0.0f;
    }
    float maxRes = max(float(textureInfo.width), float(textureInfo.height));
    float texelFootprint = footprintWorld * uvPerWorld * maxRes;
    float lod = log2(max(texelFootprint, 1.0e-7f));
    float maxMip = float(textureInfo.mipCount - 1u);
    return clamp(lod, 0.0f, maxMip);
}

inline float surface_footprint_from_cone(float coneFootprintWorld,
                                         const float3 surfaceNormal,
                                         const float3 viewDir) {
    float3 n = normalize(surfaceNormal);
    float3 v = normalize(viewDir);
    float cosTheta = fabs(dot(n, v));
    return coneFootprintWorld / max(cosTheta, 1.0e-3f);
}

inline bool uv_world_gradients_from_partials(const float3 dPdu,
                                             const float3 dPdv,
                                             thread float3& dudP,
                                             thread float3& dvdP) {
    float a00 = dot(dPdu, dPdu);
    float a01 = dot(dPdu, dPdv);
    float a11 = dot(dPdv, dPdv);
    float det = a00 * a11 - a01 * a01;
    if (fabs(det) <= 1.0e-12f) {
        return false;
    }
    dudP = (a11 * dPdu - a01 * dPdv) / det;
    dvdP = (a00 * dPdv - a01 * dPdu) / det;
    return all(isfinite(dudP)) && all(isfinite(dvdP));
}

inline bool first_hit_uv_gradients_igehy(const Ray ray,
                                         const PrimaryRayDiff rayDiff,
                                         float tHit,
                                         const float3 geometricNormal,
                                         const float3 dudP,
                                         const float3 dvdP,
                                         thread float2& dUVdx,
                                         thread float2& dUVdy) {
    dUVdx = float2(0.0f);
    dUVdy = float2(0.0f);
    float3 N = normalize(geometricNormal);
    if (!all(isfinite(N)) || dot(N, N) <= 1.0e-12f) {
        return false;
    }
    float3 D = ray.direction;
    float denom = dot(N, D);
    if (!isfinite(denom) || fabs(denom) < 1.0e-6f) {
        return false;
    }
    float3 dtdxTerm = rayDiff.dOdx + tHit * rayDiff.dDdx;
    float3 dtdyTerm = rayDiff.dOdy + tHit * rayDiff.dDdy;
    if (!all(isfinite(dtdxTerm)) || !all(isfinite(dtdyTerm))) {
        return false;
    }
    float dtdx = -dot(N, dtdxTerm) / denom;
    float dtdy = -dot(N, dtdyTerm) / denom;
    if (!isfinite(dtdx) || !isfinite(dtdy)) {
        return false;
    }
    float3 dPdx = dtdxTerm + dtdx * D;
    float3 dPdy = dtdyTerm + dtdy * D;
    if (!all(isfinite(dPdx)) || !all(isfinite(dPdy))) {
        return false;
    }
    dUVdx = float2(dot(dudP, dPdx), dot(dvdP, dPdx));
    dUVdy = float2(dot(dudP, dPdy), dot(dvdP, dPdy));
    return all(isfinite(dUVdx)) && all(isfinite(dUVdy));
}
