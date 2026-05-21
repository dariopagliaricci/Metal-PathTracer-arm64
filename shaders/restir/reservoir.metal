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
