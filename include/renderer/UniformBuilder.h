#pragma once

// ⚠️ INTERNAL HEADER - Subject to change without notice
// This header is part of PathTracer's internal implementation.
// Use only the public API from MetalRenderer.h

#include <CoreGraphics/CoreGraphics.h>

#include "MetalShaderTypes.h"
#include "RenderSettings.h"

namespace PathTracer {

// Forward declarations
class Accumulation;
class SceneResources;

/// Builds uniform buffer data for path tracing and display
/// Centralizes camera setup and uniform population
class UniformBuilder {
public:
    /// Build path tracing uniforms
    /// @param settings Render settings
    /// @param accumulation Accumulation state for frame/sample indices
    /// @param scene Scene resources for geometry counts
    /// @param renderSize Internal render resolution
    /// @return Populated PathtraceUniforms struct
    static PathTracerShaderTypes::PathtraceUniforms buildPathtraceUniforms(
        const RenderSettings& settings,
        const Accumulation& accumulation,
        const SceneResources& scene,
        CGSize renderSize);
    
    /// Build display uniforms for tonemapping
    /// @param settings Render settings
    /// @return Populated DisplayUniforms struct
    static PathTracerShaderTypes::DisplayUniforms buildDisplayUniforms(
        const RenderSettings& settings);
};

}  // namespace PathTracer