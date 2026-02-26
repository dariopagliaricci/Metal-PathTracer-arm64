#include "assets/TangentGen.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

extern "C" {
#include "mikktspace.h"
}

namespace PathTracer {

namespace {

inline simd::float3 NormalizeSafe(const simd::float3& v) {
    float len = simd::length(v);
    if (len <= 1.0e-8f) {
        return simd_make_float3(1.0f, 0.0f, 0.0f);
    }
    return v / len;
}

void GenerateTangentsFallback(std::vector<SceneResources::MeshVertex>& vertices,
                              const std::vector<uint32_t>& indices) {
    struct Accum {
        simd::float3 tangent = simd_make_float3(0.0f, 0.0f, 0.0f);
        simd::float3 bitangent = simd_make_float3(0.0f, 0.0f, 0.0f);
        float weight = 0.0f;
    };
    std::vector<Accum> posAccum(vertices.size());
    std::vector<Accum> negAccum(vertices.size());

    for (size_t i = 0; i < indices.size(); i += 3) {
        uint32_t i0 = indices[i + 0];
        uint32_t i1 = indices[i + 1];
        uint32_t i2 = indices[i + 2];
        if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size()) {
            continue;
        }

        const auto& v0 = vertices[i0];
        const auto& v1 = vertices[i1];
        const auto& v2 = vertices[i2];

        simd::float3 p0 = v0.position;
        simd::float3 p1 = v1.position;
        simd::float3 p2 = v2.position;

        simd::float2 uv0 = v0.uv;
        simd::float2 uv1 = v1.uv;
        simd::float2 uv2 = v2.uv;

        simd::float3 edge1 = p1 - p0;
        simd::float3 edge2 = p2 - p0;
        simd::float2 dUV1 = uv1 - uv0;
        simd::float2 dUV2 = uv2 - uv0;

        float denom = (dUV1.x * dUV2.y - dUV1.y * dUV2.x);
        if (std::fabs(denom) < 1.0e-8f) {
            continue;
        }
        float r = 1.0f / denom;

        simd::float3 tangent = (edge1 * dUV2.y - edge2 * dUV1.y) * r;
        simd::float3 bitangent = (edge2 * dUV1.x - edge1 * dUV2.x) * r;

        simd::float3 e0 = NormalizeSafe(p1 - p0);
        simd::float3 e1 = NormalizeSafe(p2 - p0);
        simd::float3 e2 = NormalizeSafe(p0 - p1);
        simd::float3 e3 = NormalizeSafe(p2 - p1);
        simd::float3 e4 = NormalizeSafe(p0 - p2);
        simd::float3 e5 = NormalizeSafe(p1 - p2);

        float angle0 = std::acos(std::clamp(simd::dot(e0, e1), -1.0f, 1.0f));
        float angle1 = std::acos(std::clamp(simd::dot(e2, e3), -1.0f, 1.0f));
        float angle2 = std::acos(std::clamp(simd::dot(e4, e5), -1.0f, 1.0f));

        simd::float3 n0 = NormalizeSafe(v0.normal);
        simd::float3 n1 = NormalizeSafe(v1.normal);
        simd::float3 n2 = NormalizeSafe(v2.normal);

        float sign0 = (simd::dot(simd::cross(n0, tangent), bitangent) < 0.0f) ? -1.0f : 1.0f;
        float sign1 = (simd::dot(simd::cross(n1, tangent), bitangent) < 0.0f) ? -1.0f : 1.0f;
        float sign2 = (simd::dot(simd::cross(n2, tangent), bitangent) < 0.0f) ? -1.0f : 1.0f;

        Accum& a0 = (sign0 < 0.0f) ? negAccum[i0] : posAccum[i0];
        Accum& a1 = (sign1 < 0.0f) ? negAccum[i1] : posAccum[i1];
        Accum& a2 = (sign2 < 0.0f) ? negAccum[i2] : posAccum[i2];

        a0.tangent += tangent * angle0;
        a0.bitangent += bitangent * angle0;
        a0.weight += angle0;

        a1.tangent += tangent * angle1;
        a1.bitangent += bitangent * angle1;
        a1.weight += angle1;

        a2.tangent += tangent * angle2;
        a2.bitangent += bitangent * angle2;
        a2.weight += angle2;
    }

    for (size_t i = 0; i < vertices.size(); ++i) {
        simd::float3 n = NormalizeSafe(vertices[i].normal);
        const Accum& pos = posAccum[i];
        const Accum& neg = negAccum[i];
        const Accum& best = (pos.weight >= neg.weight) ? pos : neg;
        simd::float3 t = best.tangent;
        simd::float3 b = best.bitangent;

        t = NormalizeSafe(t - n * simd::dot(n, t));
        float w = (simd::dot(simd::cross(n, t), b) < 0.0f) ? -1.0f : 1.0f;

        vertices[i].tangent = simd_make_float4(t, w);
    }
}

struct MikkMeshContext {
    std::vector<SceneResources::MeshVertex>* vertices = nullptr;
    const std::vector<uint32_t>* indices = nullptr;
};

int MikkGetNumFaces(const SMikkTSpaceContext* context) {
    auto* data = static_cast<MikkMeshContext*>(context->m_pUserData);
    return static_cast<int>(data->indices->size() / 3);
}

int MikkGetNumVerticesOfFace(const SMikkTSpaceContext* /*context*/, const int /*iFace*/) {
    return 3;
}

void MikkGetPosition(const SMikkTSpaceContext* context,
                     float posOut[],
                     const int iFace,
                     const int iVert) {
    auto* data = static_cast<MikkMeshContext*>(context->m_pUserData);
    uint32_t index = (*data->indices)[static_cast<size_t>(iFace) * 3 + iVert];
    const auto& v = (*data->vertices)[index];
    posOut[0] = v.position.x;
    posOut[1] = v.position.y;
    posOut[2] = v.position.z;
}

void MikkGetNormal(const SMikkTSpaceContext* context,
                   float normOut[],
                   const int iFace,
                   const int iVert) {
    auto* data = static_cast<MikkMeshContext*>(context->m_pUserData);
    uint32_t index = (*data->indices)[static_cast<size_t>(iFace) * 3 + iVert];
    simd::float3 n = NormalizeSafe((*data->vertices)[index].normal);
    normOut[0] = n.x;
    normOut[1] = n.y;
    normOut[2] = n.z;
}

void MikkGetTexCoord(const SMikkTSpaceContext* context,
                     float texOut[],
                     const int iFace,
                     const int iVert) {
    auto* data = static_cast<MikkMeshContext*>(context->m_pUserData);
    uint32_t index = (*data->indices)[static_cast<size_t>(iFace) * 3 + iVert];
    const auto& v = (*data->vertices)[index];
    texOut[0] = v.uv.x;
    texOut[1] = v.uv.y;
}

void MikkSetTSpaceBasic(const SMikkTSpaceContext* context,
                        const float tangent[],
                        const float sign,
                        const int iFace,
                        const int iVert) {
    auto* data = static_cast<MikkMeshContext*>(context->m_pUserData);
    uint32_t index = (*data->indices)[static_cast<size_t>(iFace) * 3 + iVert];
    auto& v = (*data->vertices)[index];
    v.tangent = simd_make_float4(tangent[0], tangent[1], tangent[2], sign);
}

}  // namespace

void GenerateTangents(std::vector<SceneResources::MeshVertex>& vertices,
                      std::vector<uint32_t>& indices) {
    if (vertices.empty() || indices.empty() || indices.size() % 3 != 0) {
        return;
    }
    bool needsDeindex = indices.size() != vertices.size();
    if (!needsDeindex) {
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] != i) {
                needsDeindex = true;
                break;
            }
        }
    }

    std::vector<SceneResources::MeshVertex> expandedVertices;
    std::vector<uint32_t> expandedIndices;
    if (needsDeindex) {
        expandedVertices.reserve(indices.size());
        expandedIndices.resize(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            uint32_t index = indices[i];
            if (index >= vertices.size()) {
                return;
            }
            expandedVertices.push_back(vertices[index]);
            expandedIndices[i] = static_cast<uint32_t>(i);
        }
        vertices.swap(expandedVertices);
        indices.swap(expandedIndices);
    }

    MikkMeshContext userData{};
    userData.vertices = &vertices;
    userData.indices = &indices;

    SMikkTSpaceInterface iface{};
    iface.m_getNumFaces = MikkGetNumFaces;
    iface.m_getNumVerticesOfFace = MikkGetNumVerticesOfFace;
    iface.m_getPosition = MikkGetPosition;
    iface.m_getNormal = MikkGetNormal;
    iface.m_getTexCoord = MikkGetTexCoord;
    iface.m_setTSpaceBasic = MikkSetTSpaceBasic;

    SMikkTSpaceContext ctx{};
    ctx.m_pInterface = &iface;
    ctx.m_pUserData = &userData;

    if (!genTangSpaceDefault(&ctx)) {
        GenerateTangentsFallback(vertices, indices);
    }
}

}  // namespace PathTracer
