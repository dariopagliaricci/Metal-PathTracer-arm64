#include "import/ImportPipeline.h"

#include <cstdio>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

namespace {

using PathTracer::ImportOptions;

std::optional<ImportOptions::OutputFormat> ParseFormat(std::string_view value) {
    if (value == "glb") {
        return ImportOptions::OutputFormat::Glb;
    }
    if (value == "gltf") {
        return ImportOptions::OutputFormat::Gltf;
    }
    return std::nullopt;
}

std::optional<ImportOptions::TextureMode> ParseTextureMode(std::string_view value) {
    if (value == "copy") {
        return ImportOptions::TextureMode::Copy;
    }
    if (value == "convert") {
        return ImportOptions::TextureMode::Convert;
    }
    if (value == "embed") {
        return ImportOptions::TextureMode::Embed;
    }
    if (value == "link") {
        return ImportOptions::TextureMode::Link;
    }
    return std::nullopt;
}

std::optional<ImportOptions::ImportMode> ParseImportMode(std::string_view value) {
    if (value == "generic") {
        return ImportOptions::ImportMode::Generic;
    }
    if (value == "canonical") {
        return ImportOptions::ImportMode::Canonical;
    }
    return std::nullopt;
}

bool RequireValue(int index, int argc, const char* flag, std::string& error) {
    if (index + 1 >= argc) {
        error = std::string("Missing value for ") + flag;
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        PathTracer::ImportOptions options;
        std::string error;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto nextValue = [&]() -> std::string {
                ++i;
                return std::string(argv[i]);
            };

            if (arg == "--help" || arg == "-h") {
                std::cout << PathTracer::PathTracerImportUsage();
                return 0;
            } else if (arg == "--in" || arg == "--input") {
                const char* flag = (arg == "--input") ? "--input" : "--in";
                if (!RequireValue(i, argc, flag, error)) {
                    break;
                }
                options.inputPath = nextValue();
            } else if (arg == "--out" || arg == "--output") {
                const char* flag = (arg == "--output") ? "--output" : "--out";
                if (!RequireValue(i, argc, flag, error)) {
                    break;
                }
                options.outputDirectory = nextValue();
            } else if (arg == "--format") {
                if (!RequireValue(i, argc, "--format", error)) {
                    break;
                }
                const auto parsed = ParseFormat(nextValue());
                if (!parsed) {
                    error = "Invalid value for --format (expected glb or gltf)";
                    break;
                }
                options.outputFormat = *parsed;
            } else if (arg == "--textures") {
                if (!RequireValue(i, argc, "--textures", error)) {
                    break;
                }
                const auto parsed = ParseTextureMode(nextValue());
                if (!parsed) {
                    error = "Invalid value for --textures (expected copy, convert, embed, or link)";
                    break;
                }
                options.textureMode = *parsed;
            } else if (arg == "--import-mode") {
                if (!RequireValue(i, argc, "--import-mode", error)) {
                    break;
                }
                const auto parsed = ParseImportMode(nextValue());
                if (!parsed) {
                    error = "Invalid value for --import-mode (expected generic or canonical)";
                    break;
                }
                options.importMode = *parsed;
            } else {
                error = "Unknown argument: " + arg;
                break;
            }
        }

        if (!error.empty()) {
            std::cerr << error << "\n\n" << PathTracer::PathTracerImportUsage();
            return 1;
        }

        if (!PathTracer::ImportBackendAvailable() && argc > 1) {
            std::fprintf(stderr,
                         "[PathTracerImport] Assimp not available in this build. "
                         "Rebuild with -DWITH_ASSIMP=ON.\n");
            return 1;
        }

        if (options.inputPath.empty() || options.outputDirectory.empty()) {
            std::cerr << "Both --in/--input and --out/--output are required.\n\n"
                      << PathTracer::PathTracerImportUsage();
            return 1;
        }

        const PathTracer::ImportResult result = PathTracer::RunImportPipeline(options);
        if (!result.success) {
            std::cerr << result.message << "\n";
            return 1;
        }

        std::cout << result.message << "\n";
        if (!result.scenePath.empty()) {
            std::cout << "Scene written to " << result.scenePath << "\n";
        }
        return 0;
    }
}
