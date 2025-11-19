#import <Foundation/Foundation.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <optional>
#include <string>
#include <vector>

#include "MetalRenderer.h"
#include "renderer/RenderSettings.h"
#include "renderer/ImageWriter.h"

namespace fs = std::filesystem;
constexpr double kPi = 3.14159265358979323846;

struct CliOptions {
    std::string scene;
    bool sceneProvided = false;
    bool sceneIsPath = false;

    std::string outputPath;
    bool outputProvided = false;

    uint32_t width = 0;
    uint32_t height = 0;
    bool widthSet = false;
    bool heightSet = false;

    uint32_t sppTotal = 1024;
    uint32_t maxDepth = 0;
    bool maxDepthSet = false;

    uint32_t seed = 0;
    bool seedSet = false;

    float envRotationDegrees = 0.0f;
    bool envRotationSet = false;

    float envIntensity = 0.0f;
    bool envIntensitySet = false;

    uint32_t tonemapMode = 0;
    bool tonemapSet = false;

    float exposure = 0.0f;
    bool exposureSet = false;

    bool enableSoftwareRayTracing = false;
    bool enableSoftwareRayTracingSet = false;

    std::string formatString = "exr";
    PathTracer::ImageFileFormat format = PathTracer::ImageFileFormat::EXR;

    bool verbose = false;
};

void PrintUsage(const char* exe) {
    std::cout << "Usage: " << exe << " [options]\n\n"
              << "Required:\n"
              << "  --scene=<id-or-path>          Scene identifier or path to .scene file\n\n"
              << "Rendering overrides:\n"
              << "  --width=<int>                 Override render width (>=8)\n"
              << "  --height=<int>                Override render height (>=8)\n"
              << "  --sppTotal=<int>              Total samples to accumulate (default 1024)\n"
              << "  --maxDepth=<int>              Override max path depth\n"
              << "  --seed=<int>                  Fixed RNG seed (0 = random)\n"
              << "  --enableSoftwareRayTracing[=0|1]  Force software tracing kernels\n\n"
              << "Environment overrides:\n"
              << "  --envRotation=<deg>           Environment rotation in degrees\n"
              << "  --envIntensity=<float>        Environment intensity multiplier\n\n"
              << "Tonemapping overrides (for LDR outputs):\n"
              << "  --tonemap=<1|2|3|4>           1=Linear, 2=ACES, 3=Reinhard, 4=Hable\n"
              << "  --exposure=<float>            Exposure in stops\n\n"
              << "Output controls:\n"
              << "  --output=<path>               Output filename\n"
              << "  --format=<exr|png|pfm|ppm>    Output format (default exr)\n"
              << "  --verbose                     Print per-frame progress\n"
              << "  --help                        Show this message\n\n"
              << "Examples:\n"
              << "  " << exe << " --scene=dragons_rtow --width=3840 --height=2160 --sppTotal=4096 \\\n"
              << "     --output=renders/dragons_4k.exr\n"
              << "  " << exe << " --scene=assets/dragon.scene --enableSoftwareRayTracing=1 \\\n"
              << "     --envRotation=30 --envIntensity=1.5 --sppTotal=1024 --format=png\n";
}

bool ParseBoolFlag(const std::string& value, bool defaultIfEmpty, bool& outValue) {
    if (value.empty()) {
        outValue = defaultIfEmpty;
        return true;
    }
    std::string lower;
    lower.reserve(value.size());
    for (char c : value) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower == "1" || lower == "true" || lower == "yes") {
        outValue = true;
        return true;
    }
    if (lower == "0" || lower == "false" || lower == "no") {
        outValue = false;
        return true;
    }
    return false;
}

bool ParseOptions(int argc, const char** argv, CliOptions& options, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string value;
        auto eqPos = arg.find('=');
        bool hasInlineValue = eqPos != std::string::npos;
        if (hasInlineValue) {
            value = arg.substr(eqPos + 1);
            arg = arg.substr(0, eqPos);
        }

        auto requireValue = [&](const char* name) -> bool {
            if (hasInlineValue) {
                return true;
            }
            if (i + 1 >= argc) {
                error = std::string(name) + " requires a value";
                return false;
            }
            value = argv[++i];
            return true;
        };

        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--verbose" || arg == "-v") {
            bool v = true;
            if (hasInlineValue && !ParseBoolFlag(value, true, v)) {
                error = "Invalid value for --verbose";
                return false;
            }
            options.verbose = v;
        } else if (arg == "--scene") {
            if (!requireValue("--scene")) {
                return false;
            }
            options.scene = value;
            options.sceneProvided = true;
        } else if (arg == "--output") {
            if (!requireValue("--output")) {
                return false;
            }
            options.outputPath = value;
            options.outputProvided = true;
        } else if (arg == "--width") {
            if (!requireValue("--width")) {
                return false;
            }
            try {
                int parsed = std::stoi(value);
                if (parsed < 8) {
                    error = "--width must be >= 8";
                    return false;
                }
                options.width = static_cast<uint32_t>(parsed);
                options.widthSet = true;
            } catch (...) {
                error = "Invalid integer for --width";
                return false;
            }
        } else if (arg == "--height") {
            if (!requireValue("--height")) {
                return false;
            }
            try {
                int parsed = std::stoi(value);
                if (parsed < 8) {
                    error = "--height must be >= 8";
                    return false;
                }
                options.height = static_cast<uint32_t>(parsed);
                options.heightSet = true;
            } catch (...) {
                error = "Invalid integer for --height";
                return false;
            }
        } else if (arg == "--sppTotal") {
            if (!requireValue("--sppTotal")) {
                return false;
            }
            try {
                int parsed = std::stoi(value);
                if (parsed < 1) {
                    error = "--sppTotal must be >= 1";
                    return false;
                }
                options.sppTotal = static_cast<uint32_t>(parsed);
            } catch (...) {
                error = "Invalid integer for --sppTotal";
                return false;
            }
        } else if (arg == "--maxDepth") {
            if (!requireValue("--maxDepth")) {
                return false;
            }
            try {
                int parsed = std::stoi(value);
                if (parsed < 1) {
                    error = "--maxDepth must be >= 1";
                    return false;
                }
                options.maxDepth = static_cast<uint32_t>(parsed);
                options.maxDepthSet = true;
            } catch (...) {
                error = "Invalid integer for --maxDepth";
                return false;
            }
        } else if (arg == "--seed") {
            if (!requireValue("--seed")) {
                return false;
            }
            try {
                options.seed = static_cast<uint32_t>(std::stoul(value));
                options.seedSet = true;
            } catch (...) {
                error = "Invalid integer for --seed";
                return false;
            }
        } else if (arg == "--envRotation") {
            if (!requireValue("--envRotation")) {
                return false;
            }
            try {
                options.envRotationDegrees = std::stof(value);
                options.envRotationSet = true;
            } catch (...) {
                error = "Invalid float for --envRotation";
                return false;
            }
        } else if (arg == "--envIntensity") {
            if (!requireValue("--envIntensity")) {
                return false;
            }
            try {
                float parsed = std::stof(value);
                options.envIntensity = std::max(parsed, 0.0f);
                options.envIntensitySet = true;
            } catch (...) {
                error = "Invalid float for --envIntensity";
                return false;
            }
        } else if (arg == "--tonemap") {
            if (!requireValue("--tonemap")) {
                return false;
            }
            try {
                int parsed = std::stoi(value);
                if (parsed < 1 || parsed > 4) {
                    error = "--tonemap must be in [1,4]";
                    return false;
                }
                options.tonemapMode = static_cast<uint32_t>(parsed);
                options.tonemapSet = true;
            } catch (...) {
                error = "Invalid integer for --tonemap";
                return false;
            }
        } else if (arg == "--exposure") {
            if (!requireValue("--exposure")) {
                return false;
            }
            try {
                options.exposure = std::stof(value);
                options.exposureSet = true;
            } catch (...) {
                error = "Invalid float for --exposure";
                return false;
            }
        } else if (arg == "--enableSoftwareRayTracing") {
            std::string boolValue = value;
            if (!hasInlineValue) {
                boolValue.clear();
            }
            bool flag = true;
            if (!ParseBoolFlag(boolValue, true, flag)) {
                error = "Invalid value for --enableSoftwareRayTracing";
                return false;
            }
            options.enableSoftwareRayTracing = flag;
            options.enableSoftwareRayTracingSet = true;
        } else if (arg == "--format") {
            if (!requireValue("--format")) {
                return false;
            }
            options.formatString = value;
        } else {
            error = "Unknown option: " + arg;
            return false;
        }
    }

    if (!options.sceneProvided) {
        error = "--scene is required";
        return false;
    }

    std::string formatError;
    if (!PathTracer::ParseImageFileFormat(options.formatString, options.format)) {
        error = "Unknown format: " + options.formatString;
        return false;
    }

    fs::path scenePath(options.scene);
    options.sceneIsPath = scenePath.extension() == ".scene" || scenePath.has_parent_path() || scenePath.is_absolute();
    if (!options.sceneIsPath) {
        std::error_code sceneEc;
        if (fs::exists(scenePath, sceneEc)) {
            options.sceneIsPath = true;
        }
    }

    return true;
}

std::string SanitizeSceneName(const std::string& input) {
    if (input.empty()) {
        return "scene";
    }
    fs::path p(input);
    std::string name = p.stem().string();
    if (name.empty()) {
        name = input;
    }
    for (char& c : name) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')) {
            c = '_';
        }
    }
    return name;
}

int main(int argc, const char** argv) {
    @autoreleasepool {
        CliOptions options;
        std::string error;
        if (!ParseOptions(argc, argv, options, error)) {
            if (!error.empty()) {
                std::cerr << "Error: " << error << "\n\n";
            }
            PrintUsage(argv[0]);
            return 1;
        }

        MetalRendererOptions rendererOptions;
        rendererOptions.headless = true;
        rendererOptions.width = options.widthSet ? static_cast<int>(options.width) : 1280;
        rendererOptions.height = options.heightSet ? static_cast<int>(options.height) : 720;
        rendererOptions.windowTitle = "PathTracerCLI";
        rendererOptions.fixedRngSeed = options.seedSet ? options.seed : 0;
        rendererOptions.enableSoftwareRayTracing = options.enableSoftwareRayTracingSet ? options.enableSoftwareRayTracing : false;

        MetalRenderer renderer;
        if (!renderer.init(nullptr, rendererOptions)) {
            std::cerr << "Failed to initialize Metal renderer" << std::endl;
            return 1;
        }

        bool sceneLoaded = false;
        if (options.sceneIsPath) {
            sceneLoaded = renderer.loadSceneFromPath(options.scene.c_str());
        } else {
            sceneLoaded = renderer.setScene(options.scene.c_str());
        }

        if (!sceneLoaded) {
            std::cerr << "Failed to load scene: " << options.scene << std::endl;
            auto ids = renderer.sceneIdentifiers();
            if (!ids.empty()) {
                std::cerr << "Available scenes:" << std::endl;
                for (const auto& id : ids) {
                    std::cerr << "  " << id << std::endl;
                }
            }
            return 1;
        }

        PathTracer::RenderSettings settings = renderer.settings();

        if (options.widthSet) {
            settings.renderWidth = options.width;
        } else if (settings.renderWidth == 0 && rendererOptions.width > 0) {
            settings.renderWidth = static_cast<uint32_t>(rendererOptions.width);
        }

        if (options.heightSet) {
            settings.renderHeight = options.height;
        } else if (settings.renderHeight == 0 && rendererOptions.height > 0) {
            settings.renderHeight = static_cast<uint32_t>(rendererOptions.height);
        }

        if (options.maxDepthSet) {
            settings.maxDepth = options.maxDepth;
        }
        if (options.seedSet) {
            settings.fixedRngSeed = options.seed;
        }
        if (options.tonemapSet) {
            settings.tonemapMode = options.tonemapMode;
        }
        if (options.exposureSet) {
            settings.exposure = options.exposure;
        }
        if (options.envRotationSet) {
            settings.environmentRotation = options.envRotationDegrees * static_cast<float>(kPi / 180.0);
        }
        if (options.envIntensitySet) {
            settings.environmentIntensity = std::max(options.envIntensity, 0.0f);
        }
        if (options.enableSoftwareRayTracingSet) {
            settings.enableSoftwareRayTracing = options.enableSoftwareRayTracing;
        }

        renderer.applySettings(settings, true);

        const uint32_t targetSamples = options.sppTotal;
        uint32_t accumulatedSamples = renderer.sampleCount();
        const uint32_t maxBatch = options.enableSoftwareRayTracingSet && options.enableSoftwareRayTracing ? 1u : 16u;

        CFAbsoluteTime renderStart = CFAbsoluteTimeGetCurrent();
        double accumulatedFrameTime = 0.0;
        CFAbsoluteTime lastProgress = renderStart;

        while (accumulatedSamples < targetSamples) {
            uint32_t remaining = targetSamples - accumulatedSamples;
            uint32_t request = (maxBatch == 0) ? remaining : std::min(maxBatch, remaining);
            renderer.setSamplesPerFrame(request);

            uint32_t before = renderer.sampleCount();
            CFAbsoluteTime frameStart = CFAbsoluteTimeGetCurrent();
            renderer.drawFrame();
            CFAbsoluteTime frameEnd = CFAbsoluteTimeGetCurrent();

            uint32_t after = renderer.sampleCount();
            uint32_t produced = (after > before) ? (after - before) : std::min(request, remaining);
            if (produced == 0) {
                produced = 1;
            }
            accumulatedSamples += produced;
            accumulatedFrameTime += (frameEnd - frameStart);

            if (options.verbose) {
                CFAbsoluteTime now = frameEnd;
                if ((now - lastProgress) >= 0.5 || accumulatedSamples >= targetSamples) {
                    double pct = (static_cast<double>(accumulatedSamples) / targetSamples) * 100.0;
                    std::cout << "Progress: " << accumulatedSamples << "/" << targetSamples
                              << " spp (" << std::fixed << std::setprecision(1) << pct << "%)\r";
                    std::cout.flush();
                    lastProgress = now;
                }
            }
        }

        if (options.verbose) {
            std::cout << std::endl;
        }

        CFAbsoluteTime renderEnd = CFAbsoluteTimeGetCurrent();

        std::vector<float> linearRGB;
        uint32_t width = 0;
        uint32_t height = 0;
        if (!renderer.captureAverageImage(linearRGB, width, height, nullptr)) {
            std::cerr << "Failed to capture rendered image" << std::endl;
            return 1;
        }

        std::string outputPath = options.outputPath;
        if (outputPath.empty()) {
            fs::path defaultDir("renders");
            std::string sceneName = SanitizeSceneName(options.scene);
            std::ostringstream filename;
            filename << sceneName << "_" << width << "x" << height << "." << PathTracer::FormatExtension(options.format);
            outputPath = (defaultDir / filename.str()).string();
        }

        fs::path outputFsPath(outputPath);
        if (!outputFsPath.parent_path().empty()) {
            std::error_code dirEc;
            fs::create_directories(outputFsPath.parent_path(), dirEc);
        }

        PathTracer::TonemapSettings tonemapSettings{};
        PathTracer::RenderSettings finalSettings = renderer.settings();
        tonemapSettings.tonemapMode = finalSettings.tonemapMode;
        tonemapSettings.acesVariant = finalSettings.acesVariant;
        tonemapSettings.exposure = finalSettings.exposure;
        tonemapSettings.reinhardWhitePoint = finalSettings.reinhardWhitePoint;

        std::string writeError;
        if (!PathTracer::WriteImage(outputFsPath.string(), options.format,
                                    linearRGB.data(), width, height,
                                    tonemapSettings, &writeError)) {
            std::cerr << "Failed to write output image: " << writeError << std::endl;
            return 1;
        }

        double totalSeconds = renderEnd - renderStart;
        double avgMsPerSample = (accumulatedSamples > 0)
                                    ? (accumulatedFrameTime * 1000.0 / accumulatedSamples)
                                    : 0.0;

        std::cout << "Rendered " << accumulatedSamples << " spp at " << width << "x" << height
                  << " in " << std::fixed << std::setprecision(2) << totalSeconds << " s"
                  << " (~" << std::setprecision(3) << avgMsPerSample << " ms/sample)." << std::endl;
        std::cout << "Output written to: " << outputFsPath << std::endl;

        return 0;
    }
}
