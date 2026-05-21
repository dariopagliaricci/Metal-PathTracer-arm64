#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace PathTracer::Import {

struct ConvertResult {
    std::string output_path;
    std::string output_sha256;
    std::string source_format;
    std::string output_format;
    std::string transcode_tool;
    std::string transcode_version;
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t output_bytes = 0;
    bool has_alpha = false;
    bool binary_alpha = false;
    bool fell_back_to_copy = false;
};

struct DecodedTexture {
    std::string source_format;
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> rgba;
    bool has_alpha = false;
    bool binary_alpha = false;
};

ConvertResult convertTexture(const std::string& source_path, const std::string& output_dir);
DecodedTexture decodeTextureFile(const std::string& source_path);

}  // namespace PathTracer::Import
