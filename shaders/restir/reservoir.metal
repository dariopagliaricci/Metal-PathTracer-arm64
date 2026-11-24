struct RISSamplePayload {
    uint primitiveIndex;
    uint flags;
    float2 bary;
    float3 position;
    float3 normal;
    float3 emission;
    float pdf;
};

struct Reservoir {
    RISSamplePayload winner;
    float wSum;
    uint M;
    float W;
    bool valid;
};

inline Reservoir make_empty_reservoir() {
    Reservoir r;
    r.winner.primitiveIndex = kInvalidIndex;
    r.winner.flags = 0u;
    r.winner.bary = float2(0.0f);
    r.winner.position = float3(0.0f);
    r.winner.normal = float3(0.0f);
    r.winner.emission = float3(0.0f);
    r.winner.pdf = 0.0f;
    r.wSum = 0.0f;
    r.M = 0u;
    r.W = 0.0f;
    r.valid = false;
    return r;
}

inline bool update_reservoir(thread Reservoir& r,
                             const RISSamplePayload candidate,
                             float weight,
                             float u) {
    if (!(weight > 0.0f) || !isfinite(weight)) {
        return false;
    }

    float nextWSum = r.wSum + weight;
    if (!(nextWSum > 0.0f) || !isfinite(nextWSum)) {
        return false;
    }

    r.wSum = nextWSum;
    if (r.M == 0u) {
        r.M = 1u;
        r.winner = candidate;
        r.valid = true;
        return true;
    }

    r.M += 1u;
    float acceptProb = weight / r.wSum;
    if (!(acceptProb >= 0.0f) || !isfinite(acceptProb)) {
        return false;
    }
    if (u < acceptProb) {
        r.winner = candidate;
        r.valid = true;
        return true;
    }
    return false;
}

inline bool merge_reservoir_winner(thread Reservoir& r,
                                   const RISSamplePayload candidate,
                                   float weight,
                                   uint representedM,
                                   float u) {
    if (!(weight > 0.0f) || !isfinite(weight)) {
        return false;
    }
    representedM = max(representedM, 1u);

    float nextWSum = r.wSum + weight;
    if (!(nextWSum > 0.0f) || !isfinite(nextWSum)) {
        return false;
    }

    r.wSum = nextWSum;
    if (r.M == 0u) {
        r.M = representedM;
        r.winner = candidate;
        r.valid = true;
        return true;
    }

    r.M += representedM;
    float acceptProb = weight / r.wSum;
    if (!(acceptProb >= 0.0f) || !isfinite(acceptProb)) {
        return false;
    }
    if (u < acceptProb) {
        r.winner = candidate;
        r.valid = true;
        return true;
    }
    return false;
}

// Pairwise MIS for a canonical reservoir merged with k neighbour reservoirs that all hold the same candidate count M.
// The arguments a_c and a_n are the two pixels' targets per unit light area, p_hat * cos(theta_light) / d^2
// (ris_payload_solid_angle_per_area): RIS candidate pdfs are solid-angle densities at the point that built the
// reservoir, so a sample moves between pixels through the light point, with Jacobian a ratio of those terms.
// Neighbour n is paired with 1/k of the canonical domain. For a light sample x, neighbour n's MIS weight is
// a_n(x) / (k a_n(x) + a_c(x)) and the canonical's share of that pair is a_c(x) / (k a_n(x) + a_c(x)) / k. They sum to
// 1/k per pair and so to one over the k pairs, which keeps the merge unbiased for any k. Weights are expressed on the
// canonical 1/M scale, so the merged reservoir keeps M and W = wSum / M / p_c(y) still holds.

// Resampling weight of neighbour n's winner y_n: m_n(y_n) p_c(y_n) W_n J times M, with W_n = wSum_n / (M p_n(y_n)) and
// J the solid-angle Jacobian from the neighbour to the canonical; that reduces to a_c wSum_n / (a_c + k a_n).
inline float pairwise_mis_neighbor_weight(float areaTargetCanonical,
                                          float areaTargetNeighbor,
                                          float neighborWSum,
                                          float k) {
    float denom = areaTargetCanonical + k * areaTargetNeighbor;
    if (!(areaTargetCanonical > 0.0f) || !(denom > 0.0f) || !isfinite(denom)) {
        return 0.0f;
    }
    float weight = areaTargetCanonical * neighborWSum / denom;
    return isfinite(weight) ? weight : 0.0f;
}

// The canonical winner y_c's share of pair n. A pair whose neighbour was rejected, or where neither target is
// positive, belongs to the canonical entirely (1/k).
inline float pairwise_mis_canonical_share(float areaTargetCanonical,
                                          float areaTargetNeighbor,
                                          float k) {
    float denom = areaTargetCanonical + k * areaTargetNeighbor;
    if (!(denom > 0.0f) || !isfinite(denom)) {
        return 1.0f / k;
    }
    return areaTargetCanonical / denom / k;
}

inline float p_hat(float3 Le,
                   float3 bsdf,
                   float cosThetaSurface,
                   float cosThetaLight,
                   float distSq) {
    if (!(distSq > 0.0f) || !isfinite(distSq)) {
        return 0.0f;
    }
    float lumLe = dot(Le, kLuminanceWeights);
    float lumBsdf = dot(bsdf, kLuminanceWeights);
    float v = lumLe * lumBsdf * cosThetaSurface * cosThetaLight / distSq;
    return (isfinite(v) && v > 0.0f) ? v : 0.0f;
}

inline bool ris_payload_is_directional(const RISSamplePayload payload) {
    return (payload.flags & kLightPrimitiveFlagDirectional) != 0u;
}

inline float3 ris_payload_omega_l(thread const HitRecord& rec,
                                  const RISSamplePayload payload) {
    return ris_payload_is_directional(payload)
        ? safe_normalize(payload.position)
        : safe_normalize(payload.position - rec.point);
}

inline float ris_payload_distance_sq(thread const HitRecord& rec,
                                     const RISSamplePayload payload) {
    if (ris_payload_is_directional(payload)) {
        return 1.0f;
    }
    return dot(payload.position - rec.point, payload.position - rec.point);
}

inline int3 world_reuse_cell(float3 position, float cellSize) {
    float safeCellSize = max(cellSize, 1.0e-4f);
    return int3(floor(position / safeCellSize));
}

inline uint world_reuse_cell_hash(int3 cell) {
    uint x = uint(cell.x) * 2654435761u;
    uint y = uint(cell.y) * 805459861u;
    uint z = uint(cell.z) * 3674653429u;
    return (x ^ y ^ z) & 0x00ffffffu;
}

inline bool world_reuse_cell_compatible(int3 currentCell, int3 candidateCell) {
    int3 delta = abs(candidateCell - currentCell);
    return delta.x <= 1 && delta.y <= 1 && delta.z <= 1;
}
