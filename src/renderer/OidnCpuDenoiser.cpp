#include "renderer/OidnCpuDenoiser.h"

#ifndef PATH_TRACER_ENABLE_OIDN
#define PATH_TRACER_ENABLE_OIDN 0
#endif

#if PATH_TRACER_ENABLE_OIDN
#include <OpenImageDenoise/oidn.h>
#endif

#include <cstddef>

namespace PathTracer {

#if PATH_TRACER_ENABLE_OIDN
namespace {

bool CheckOidnError(OIDNDevice device, const char* stage, std::string& error) {
    const char* message = nullptr;
    const OIDNError code = oidnGetDeviceError(device, &message);
    if (code == OIDN_ERROR_NONE) {
        return true;
    }

    error = stage;
    if (message && *message) {
        error += ": ";
        error += message;
    }
    return false;
}

const char* ToOidnFilterName(OidnFilterType filterType) {
    return filterType == OidnFilterType::RTLightmap ? "RTLightmap" : "RT";
}

bool ValidateGuideBuffer(const float* guide,
                         size_t expectedValues,
                         const char* name,
                         std::string& error) {
    if (!guide) {
        return true;
    }
    if (expectedValues == 0) {
        error = std::string("Invalid ") + name + " guide dimensions";
        return false;
    }
    return true;
}

}  // namespace

bool DenoiseWithOidn(const OidnDenoiseRequest& request,
                     std::vector<float>& outDenoised,
                     std::string& error) {
    if (!request.color) {
        error = "OIDN color input is null";
        return false;
    }
    if (request.width == 0 || request.height == 0) {
        error = "OIDN input dimensions must be non-zero";
        return false;
    }

    const size_t pixelCount = static_cast<size_t>(request.width) * static_cast<size_t>(request.height);
    const size_t valueCount = pixelCount * 3u;
    if (!ValidateGuideBuffer(request.albedo, valueCount, "albedo", error) ||
        !ValidateGuideBuffer(request.normal, valueCount, "normal", error)) {
        return false;
    }

    outDenoised.resize(valueCount);

    OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_CPU);
    if (!device) {
        error = "Failed to create OIDN CPU device";
        return false;
    }

    oidnCommitDevice(device);
    if (!CheckOidnError(device, "OIDN device commit failed", error)) {
        oidnReleaseDevice(device);
        return false;
    }

    OIDNFilter filter = oidnNewFilter(device, ToOidnFilterName(request.filterType));
    if (!filter) {
        error = "Failed to create OIDN filter";
        oidnReleaseDevice(device);
        return false;
    }

    const size_t pixelStride = sizeof(float) * 3u;
    const size_t rowStride = pixelStride * static_cast<size_t>(request.width);

    oidnSetSharedFilterImage(filter,
                             "color",
                             const_cast<float*>(request.color),
                             OIDN_FORMAT_FLOAT3,
                             request.width,
                             request.height,
                             0,
                             pixelStride,
                             rowStride);

    const bool useAlbedo = request.albedo != nullptr;
    if (useAlbedo) {
        oidnSetSharedFilterImage(filter,
                                 "albedo",
                                 const_cast<float*>(request.albedo),
                                 OIDN_FORMAT_FLOAT3,
                                 request.width,
                                 request.height,
                                 0,
                                 pixelStride,
                                 rowStride);
    }

    const bool useNormal = request.normal != nullptr;
    if (useNormal) {
        oidnSetSharedFilterImage(filter,
                                 "normal",
                                 const_cast<float*>(request.normal),
                                 OIDN_FORMAT_FLOAT3,
                                 request.width,
                                 request.height,
                                 0,
                                 pixelStride,
                                 rowStride);
    }

    oidnSetSharedFilterImage(filter,
                             "output",
                             outDenoised.data(),
                             OIDN_FORMAT_FLOAT3,
                             request.width,
                             request.height,
                             0,
                             pixelStride,
                             rowStride);

    oidnSetFilterBool(filter, "hdr", request.hdr);
    oidnSetFilterBool(filter, "srgb", false);
    oidnSetFilterBool(filter, "cleanAux", request.cleanAux && (useAlbedo || useNormal));
    oidnCommitFilter(filter);
    if (!CheckOidnError(device, "OIDN filter commit failed", error)) {
        oidnReleaseFilter(filter);
        oidnReleaseDevice(device);
        return false;
    }

    oidnExecuteFilter(filter);
    const bool ok = CheckOidnError(device, "OIDN filter execution failed", error);

    oidnReleaseFilter(filter);
    oidnReleaseDevice(device);
    return ok;
}
#else
bool DenoiseWithOidn(const OidnDenoiseRequest&,
                     std::vector<float>& outDenoised,
                     std::string& error) {
    outDenoised.clear();
    error = "OIDN support is disabled in this build";
    return false;
}
#endif

}  // namespace PathTracer
