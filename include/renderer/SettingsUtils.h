#pragma once
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#if __has_include("renderer/RenderSettings.h")
  #include "renderer/RenderSettings.h"
#else
  // IntelliSense fallback so the parser doesn't error if includes aren't resolved yet.
  namespace PathTracer { struct RenderSettings; }
#endif

namespace PathTracer {

class SceneResources;
// strict=true rejects unsupported headless/backend requests; interactive callers
// switch explicitly and report the diagnostic before encoding any frame.
bool ResolveSourceCapabilities(const SceneResources& scene, RenderSettings& settings,
                               bool embree, bool strict, std::string& diagnostic);
// First combination that hero-wavelength spectral transport does not support on the Metal backend, as
// "<setting>=<value>"; empty when spectral mode is off or everything requested is supported. Call it after
// ResolveSourceCapabilities, which settles sourceFeatures.
std::string SpectralUnsupportedCombination(const RenderSettings& settings);
// directLightMode=legacy samples rect lights and the environment only, so analytic `sun` lines are not lit there.
// Returns a warning when the scene has such lights in that mode; empty otherwise.
std::string IgnoredDirectionalLightsDiagnostic(const SceneResources& scene, const RenderSettings& settings);

// One line per subsurface material that will not scatter under the surface with these settings (sss off, or a
// method the renderer's sss mode does not run). Such a material is shaded as plain diffuse in its base colour, under
// its coat if it has one; headless and the GUI print these lines so that this is never silent.
std::vector<std::string> SubsurfaceNotScatteringDiagnostics(const SceneResources& scene,
                                                            const RenderSettings& settings);

// Random walks of one material that ended with no light: at the step limit (sssMaxSteps is cutting walks) or outside
// a mesh that is not closed. Returns one line naming the material, with counts and shares of its walks; empty when
// none of its walks ended that way.
std::string SubsurfaceWalkLossDiagnostic(const std::string& material, uint64_t walks, uint64_t openMesh,
                                         uint64_t stepLimit, uint32_t sssMaxSteps);

struct RadiometricChangeResult {
    bool changed = false;
    const char* reason = nullptr;
};

RadiometricChangeResult DetectRadiometricChange(const PathTracer::RenderSettings& prev,
                                                const PathTracer::RenderSettings& next);

// Compare two settings and tell us whether we must reset accumulation.
// Radiometry-affecting changes = camera, environment, tonemap, materials (future).
bool MarkRadiometricChange(const PathTracer::RenderSettings& prev,
                           const PathTracer::RenderSettings& next);

// Optional helper for floating-point comparison.
inline bool nearlyEqual(float a, float b, float eps = 1e-5f) {
    const float aa = std::fabs(a), bb = std::fabs(b);
    return std::fabs(a - b) <= eps * std::max(1.0f, std::max(aa, bb));
}

} // namespace PathTracer
