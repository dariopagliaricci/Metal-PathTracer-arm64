#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace PathTracer {

/// A copy of the progressive accumulation: the frame and sample counters and the bytes of every texture the
/// integrators accumulate into. Restoring it continues an accumulation exactly where it stopped (the per-sample RNG
/// seeds read the counters and the per-pixel sample counts).
struct AccumulationTextureState {
    std::string name;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t bytesPerPixel = 0;
    std::vector<uint8_t> bytes;
};

struct AccumulationState {
    uint32_t frameIndex = 0;
    uint32_t sampleCount = 0;
    std::vector<AccumulationTextureState> textures;
};

/// Binary file I/O for an AccumulationState (headless sample checkpoints). WriteAccumulationStateFile writes to a
/// temporary file and renames it, so an interrupted write never leaves a truncated state behind.
bool WriteAccumulationStateFile(const std::string& path, const AccumulationState& state, std::string& error);
bool ReadAccumulationStateFile(const std::string& path, AccumulationState& state, std::string& error);

}  // namespace PathTracer
