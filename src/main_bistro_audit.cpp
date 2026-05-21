#include "import/BistroMaterialAudit.h"

#include <iostream>
#include <string>

int main(int argc, const char* argv[]) {
    PathTracer::Import::BistroAuditOptions options;
    options.interiorScenePath = "assets/canonical/bistro/source/BistroInterior.fbx";
    options.exteriorScenePath = "assets/canonical/bistro/source/BistroExterior.fbx";
    options.outputPath = "validation/reports/bistro_material_fidelity_audit.md";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto needValue = [&](const char* flag) -> bool {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n\n"
                          << PathTracer::Import::PathTracerBistroAuditUsage();
                return false;
            }
            return true;
        };

        if (arg == "--help" || arg == "-h") {
            std::cout << PathTracer::Import::PathTracerBistroAuditUsage();
            return 0;
        }
        if (arg == "--interior") {
            if (!needValue("--interior")) {
                return 1;
            }
            options.interiorScenePath = argv[++i];
            continue;
        }
        if (arg == "--exterior") {
            if (!needValue("--exterior")) {
                return 1;
            }
            options.exteriorScenePath = argv[++i];
            continue;
        }
        if (arg == "--out" || arg == "--output") {
            if (!needValue("--out")) {
                return 1;
            }
            options.outputPath = argv[++i];
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n\n"
                  << PathTracer::Import::PathTracerBistroAuditUsage();
        return 1;
    }

    if (!PathTracer::Import::BistroAuditBackendAvailable()) {
        std::cerr << "[PathTracerBistroAudit] Assimp not available in this build.\n";
        return 1;
    }

    const PathTracer::Import::BistroAuditResult result =
        PathTracer::Import::RunBistroMaterialAudit(options);
    if (!result.success) {
        std::cerr << result.message << "\n";
        return 1;
    }
    std::cout << result.message << "\n";
    return 0;
}
