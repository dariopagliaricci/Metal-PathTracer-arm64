#include <embree4/rtcore.h>

#include <cstdint>
#include <iostream>

int main() {
    RTCDevice device = rtcNewDevice(nullptr);
    if (!device) {
        std::cerr << "Failed to create Embree device" << std::endl;
        return 1;
    }

    RTCScene scene = rtcNewScene(device);
    RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    float* vertices = reinterpret_cast<float*>(
        rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0,
                                RTC_FORMAT_FLOAT3, sizeof(float) * 3, 3));
    if (!vertices) {
        std::cerr << "Failed to allocate vertex buffer" << std::endl;
        rtcReleaseGeometry(geometry);
        rtcReleaseScene(scene);
        rtcReleaseDevice(device);
        return 1;
    }

    vertices[0] = -1.0f; vertices[1] = 0.0f; vertices[2] = 0.0f;
    vertices[3] = 1.0f; vertices[4] = 0.0f; vertices[5] = 0.0f;
    vertices[6] = 0.0f; vertices[7] = 1.0f; vertices[8] = 0.0f;

    uint32_t* indices = reinterpret_cast<uint32_t*>(
        rtcSetNewGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0,
                                RTC_FORMAT_UINT3, sizeof(uint32_t) * 3, 1));
    if (!indices) {
        std::cerr << "Failed to allocate index buffer" << std::endl;
        rtcReleaseGeometry(geometry);
        rtcReleaseScene(scene);
        rtcReleaseDevice(device);
        return 1;
    }

    indices[0] = 0;
    indices[1] = 1;
    indices[2] = 2;

    rtcCommitGeometry(geometry);
    rtcAttachGeometry(scene, geometry);
    rtcReleaseGeometry(geometry);

    rtcCommitScene(scene);

    RTCIntersectArguments args;
    rtcInitIntersectArguments(&args);

    RTCRayHit rayHit{};
    rayHit.ray.org_x = 0.0f;
    rayHit.ray.org_y = 0.2f;
    rayHit.ray.org_z = -1.0f;
    rayHit.ray.dir_x = 0.0f;
    rayHit.ray.dir_y = 0.0f;
    rayHit.ray.dir_z = 1.0f;
    rayHit.ray.tnear = 0.0f;
    rayHit.ray.tfar = 1000.0f;
    rayHit.ray.mask = 0xFFFFFFFFu;
    rayHit.ray.flags = 0;
    rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

    rtcIntersect1(scene, &rayHit, &args);

    int result = (rayHit.hit.geomID != RTC_INVALID_GEOMETRY_ID) ? 0 : 2;

    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
    return result;
}
