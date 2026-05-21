#pragma once

#include <string>

namespace PathTracer::Import {

struct BistroAuditOptions {
    std::string interiorScenePath;
    std::string exteriorScenePath;
    std::string outputPath;
};

struct BistroAuditResult {
    bool success = false;
    std::string message;
};

bool BistroAuditBackendAvailable();
BistroAuditResult RunBistroMaterialAudit(const BistroAuditOptions& options);
std::string PathTracerBistroAuditUsage();

}  // namespace PathTracer::Import
