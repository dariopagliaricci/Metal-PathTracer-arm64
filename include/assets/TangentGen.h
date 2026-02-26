#pragma once

#include <vector>

#include "renderer/SceneResources.h"

namespace PathTracer {

/// Generate MikkTSpace tangents for triangle meshes.
/// Requires valid positions, normals, and UVs and may deindex the mesh.
void GenerateTangents(std::vector<SceneResources::MeshVertex>& vertices,
                      std::vector<uint32_t>& indices);

}  // namespace PathTracer
