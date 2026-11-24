// The source-features switch and the material-only source predicates (EP8). Metal includes this from
// core/constants.metal; the predicates that read PathtraceUniforms stay there.
#ifndef PT_SHARED_SOURCE_FEATURES_H
#define PT_SHARED_SOURCE_FEATURES_H

#include "portability.h"

#ifdef __METAL_VERSION__
#ifndef PT_SOURCE_FEATURES
#define PT_SOURCE_FEATURES 0
#endif
constant bool kSourceFeatures = PT_SOURCE_FEATURES != 0;
#endif
// C++: cpp_prelude.h defines kSourceFeatures as true; Embree has no separate legacy build of these paths.

PT_SHARED_BEGIN

inline float source_model(PT_IN(MaterialData) m) { return kSourceFeatures ? m.sourceParams.x : 0.0f; }
inline bool source_material_flag(PT_IN(MaterialData) m, const uint flag) {
    return kSourceFeatures && (m.materialFlags & flag) != 0u;
}
inline bool source_has_opacity(PT_IN(MaterialData) m) { return kSourceFeatures && m.textureControl.y != 0.0f; }

PT_SHARED_END

#endif  // PT_SHARED_SOURCE_FEATURES_H
