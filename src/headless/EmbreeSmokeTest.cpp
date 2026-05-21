#include <embree4/rtcore.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options {
    uint32_t width = 64;
    uint32_t height = 36;
    std::string outputPath;
};

void PrintUsage(const char* exe) {
    std::cout << "Usage: " << exe << " [--width=<int>] [--height=<int>] [--output=<path>]\n";
}

bool ParseUint(const std::string& value, uint32_t& out) {
    if (value.empty()) {
        return false;
    }
    try {
        unsigned long parsed = std::stoul(value);
        if (parsed == 0) {
            return false;
        }
        out = static_cast<uint32_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool ParseOptions(int argc, const char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto eq = arg.find('=');
        std::string key = arg;
        std::string value;
        if (eq != std::string::npos) {
            key = arg.substr(0, eq);
            value = arg.substr(eq + 1);
        }

        if (key == "--help" || key == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        }
        if (key == "--width") {
            if ((eq == std::string::npos && (++i >= argc || !ParseUint(argv[i], options.width))) ||
                (eq != std::string::npos && !ParseUint(value, options.width))) {
                std::cerr << "Invalid value for --width" << std::endl;
                return false;
            }
            continue;
        }
        if (key == "--height") {
            if ((eq == std::string::npos && (++i >= argc || !ParseUint(argv[i], options.height))) ||
                (eq != std::string::npos && !ParseUint(value, options.height))) {
                std::cerr << "Invalid value for --height" << std::endl;
                return false;
            }
            continue;
        }
        if (key == "--output") {
            if (eq == std::string::npos) {
                if (++i >= argc) {
                    std::cerr << "--output requires a value" << std::endl;
                    return false;
                }
                options.outputPath = argv[i];
            } else {
                options.outputPath = value;
            }
            continue;
        }

        std::cerr << "Unknown option: " << arg << std::endl;
        return false;
    }

    return true;
}

bool WritePpm(const std::string& path,
              uint32_t width,
              uint32_t height,
              const std::vector<uint8_t>& rgb) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out << "P6\n" << width << " " << height << "\n255\n";
    out.write(reinterpret_cast<const char*>(rgb.data()),
              static_cast<std::streamsize>(rgb.size()));
    return out.good();
}

float Clamp01(float value) {
    return std::max(0.0f, std::min(1.0f, value));
}

}  // namespace

int main(int argc, const char** argv) {
    Options options;
    if (!ParseOptions(argc, argv, options)) {
        PrintUsage(argv[0]);
        return 1;
    }

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

    vertices[0] = -1.25f; vertices[1] = -0.90f; vertices[2] = 0.0f;
    vertices[3] = 1.10f; vertices[4] = -0.95f; vertices[5] = 0.0f;
    vertices[6] = 0.00f; vertices[7] = 1.10f; vertices[8] = 0.0f;

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

    std::vector<uint8_t> pixels(static_cast<size_t>(options.width) * options.height * 3u, 0u);
    uint32_t hitCount = 0;
    const float aspect = static_cast<float>(options.width) / static_cast<float>(options.height);

    for (uint32_t y = 0; y < options.height; ++y) {
        for (uint32_t x = 0; x < options.width; ++x) {
            const float u = ((static_cast<float>(x) + 0.5f) / static_cast<float>(options.width)) * 2.0f - 1.0f;
            const float v = 1.0f - ((static_cast<float>(y) + 0.5f) / static_cast<float>(options.height)) * 2.0f;

            RTCRayHit rayHit{};
            rayHit.ray.org_x = 0.0f;
            rayHit.ray.org_y = 0.0f;
            rayHit.ray.org_z = -2.0f;
            rayHit.ray.dir_x = u * aspect;
            rayHit.ray.dir_y = v;
            rayHit.ray.dir_z = 1.8f;
            rayHit.ray.tnear = 0.001f;
            rayHit.ray.tfar = 1000.0f;
            rayHit.ray.mask = 0xFFFFFFFFu;
            rayHit.ray.flags = 0;
            rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

            RTCIntersectArguments args;
            rtcInitIntersectArguments(&args);
            rtcIntersect1(scene, &rayHit, &args);

            const size_t pixelIndex = static_cast<size_t>(y) * options.width * 3u + x * 3u;
            if (rayHit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                ++hitCount;
                const float baryU = rayHit.hit.u;
                const float baryV = rayHit.hit.v;
                const float baryW = std::max(0.0f, 1.0f - baryU - baryV);
                pixels[pixelIndex + 0u] = static_cast<uint8_t>(Clamp01(0.15f + baryW * 0.85f) * 255.0f);
                pixels[pixelIndex + 1u] = static_cast<uint8_t>(Clamp01(0.10f + baryU * 0.90f) * 255.0f);
                pixels[pixelIndex + 2u] = static_cast<uint8_t>(Clamp01(0.10f + baryV * 0.90f) * 255.0f);
            } else {
                const float sky = Clamp01(0.20f + 0.60f * (0.5f * (v + 1.0f)));
                pixels[pixelIndex + 0u] = static_cast<uint8_t>(sky * 64.0f);
                pixels[pixelIndex + 1u] = static_cast<uint8_t>(sky * 80.0f);
                pixels[pixelIndex + 2u] = static_cast<uint8_t>(sky * 112.0f);
            }
        }
    }

    if (hitCount == 0) {
        std::cerr << "Embree smoke render produced zero hits" << std::endl;
        rtcReleaseScene(scene);
        rtcReleaseDevice(device);
        return 2;
    }

    if (!options.outputPath.empty() &&
        !WritePpm(options.outputPath, options.width, options.height, pixels)) {
        std::cerr << "Failed to write smoke image: " << options.outputPath << std::endl;
        rtcReleaseScene(scene);
        rtcReleaseDevice(device);
        return 3;
    }

    std::cout << "Embree smoke render passed (" << hitCount << " hits";
    if (!options.outputPath.empty()) {
        std::cout << ", wrote " << options.outputPath;
    }
    std::cout << ")" << std::endl;

    rtcReleaseScene(scene);
    rtcReleaseDevice(device);
    return 0;
}
