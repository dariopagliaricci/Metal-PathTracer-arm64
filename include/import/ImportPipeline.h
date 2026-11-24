#pragma once

#include <string>

namespace PathTracer {

struct ImportOptions {
    enum class ImportMode {
        Generic,
        Canonical,
    };

    enum class OutputFormat {
        Glb,
        Gltf,
    };

    enum class TextureMode {
        Copy,
        Convert,
        Embed,
        Link,
    };

    std::string inputPath;
    std::string outputDirectory;
    ImportMode importMode = ImportMode::Generic;
    OutputFormat outputFormat = OutputFormat::Glb;
    TextureMode textureMode = TextureMode::Copy;
    // OBJ only: serve Assimp the file with every distinct `mtllib` ahead of all geometry
    // (works around Assimp's repeated/late `mtllib` material mis-assignment).
    bool objHoistMaterialLibraries = true;
    // OBJ only: downgrade a per-material face-count mismatch from an import failure to a warning.
    bool allowObjMaterialMismatch = false;
};

struct ImportResult {
    bool success = false;
    std::string message;
    std::string scenePath;
};

bool ImportBackendAvailable();
ImportResult RunImportPipeline(const ImportOptions& options);
std::string PathTracerImportUsage();

}  // namespace PathTracer
