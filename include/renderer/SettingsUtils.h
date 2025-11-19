#pragma once
#include <algorithm>
#include <cmath>

#if __has_include("renderer/RenderSettings.h")
  #include "renderer/RenderSettings.h"
#else
  // IntelliSense fallback so the parser doesn't error if includes aren't resolved yet.
  namespace PathTracer { struct RenderSettings; }
#endif

namespace PathTracer {

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
