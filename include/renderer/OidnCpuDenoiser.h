#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace PathTracer {

enum class OidnFilterType : uint32_t {
    RT = 0,
    RTLightmap = 1,
};

struct OidnDenoiseRequest {
    const float* color = nullptr;
    const float* albedo = nullptr;
    const float* normal = nullptr;
    uint32_t width = 0;
    uint32_t height = 0;
    bool hdr = true;
    bool cleanAux = true;
    OidnFilterType filterType = OidnFilterType::RT;
};

bool DenoiseWithOidn(const OidnDenoiseRequest& request,
                     std::vector<float>& outDenoised,
                     std::string& error);

}  // namespace PathTracer