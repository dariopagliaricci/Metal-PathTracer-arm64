# Golden Image Testing - Implementation Guide

## Overview

This document provides the complete implementation details for the golden image testing system. The system is **fully production-ready** with all headless rendering support, deterministic output, and comprehensive testing tools.

## Components Status

✅ **Complete - Milestone 0:**
- [x] Test directory structure (`tests/golden/{before,after}`, `tests/tools/`)
- [x] SHA-256 comparison tool (`tests/tools/sha256.py`)
- [x] PSNR comparison tool (`tests/tools/psnr.py`)
- [x] Test orchestrator script (`tests/tools/golden_test.sh`)
- [x] CMake integration with CTest
- [x] Comprehensive documentation (`GOLDEN_IMAGE_TESTING.md`)
- [x] Headless entry point (`src/main_headless.mm`)
- [x] Extended renderer types (`MetalRendererTypes.h`)
- [x] Extended renderer interface (`MetalRenderer.h`)
- [x] MetalRenderer headless mode support with proper initialization
- [x] PPM export functionality with gamma correction
- [x] Fixed RNG seed integration in renderer
- [x] Shader RNG seed parameter with deterministic per-pixel seeding
- [x] Accumulation texture management for headless rendering
- [x] CTest integration with automated golden image validation

## Implementation Details (Completed)

### 1. Headless Initialization

**Status:** ✅ Complete

The `MetalRenderer::Impl::initialize()` method now supports headless mode:
- Skips MTKView creation when `options.headless` is true
- Disables ImGui setup for headless rendering
- Maintains full Metal device and command queue setup
- Allows window parameter to be null in headless mode

### 2. Pipeline Configuration for Headless

**Status:** ✅ Complete

The `loadPipeline()` method now:
- Uses fixed `MTLPixelFormatBGRA8Unorm` in headless mode
- Falls back to view's color format in GUI mode
- Maintains compatibility with both rendering paths

### 3. Frame Rendering for Headless

**Status:** ✅ Complete

The `drawFrame()` method now:
- Uses fixed dimensions from `options.width` and `options.height` in headless mode
- Handles dynamic view size in GUI mode
- Skips display pipeline and ImGui in headless mode
- Waits for command buffer completion in headless (synchronous) mode
- Properly manages accumulation textures for both modes

### 4. RNG Seed Support for Deterministic Rendering

**Status:** ✅ Complete

Implementation includes:
- `PathtraceUniforms` struct extended with `fixedRngSeed` field
- `populateUniforms()` passes seed from renderer options to shader
- Shader kernel uses seed in per-pixel RNG computation:
  ```metal
  uint seed = uniforms.fixedRngSeed + frameIndex * 9781u + gid.x * 6271u +
              gid.y * 13007u + sampleCount * 211u;
  ```
- Ensures identical random sequences for same seed across runs

### 5. PPM Export Functionality

**Status:** ✅ Complete

The `exportToPPM()` method:
- Reads accumulated radiance and sample count textures from GPU
- Performs proper HDR-to-LDR conversion with gamma correction (2.2)
- Clamps values to [0, 1] range before conversion
- Writes PPM P3 ASCII format for portability
- Returns success/failure status

**Method signatures:**
```cpp
// In MetalRenderer::Impl
bool exportToPPM(const char* filepath);

// Public interface in MetalRenderer
bool MetalRenderer::exportToPPM(const char* filepath);
```

### 6. Accumulation Texture Management

**Status:** ✅ Complete

The `Accumulation` class:
- Supports explicit texture creation with CGSize parameter
- Method `ensureTextures(const CGSize& size)` creates textures based on provided dimensions
- Works seamlessly in both headless and GUI modes
- No longer relies on view for dimension queries

## Testing & Validation

### 1. Build and Test Headless Renderer

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRICT=ON -DM0_TESTS=ON
cmake --build build --target PathTracerHeadless -j

# Test headless rendering
./build/PathTracerHeadless \
    --width 256 --height 144 --spp 64 \
    --seed 42 --out test.ppm --verbose

# Verify output
file test.ppm  # Should show: "PPM image text"
open test.ppm  # View in Preview.app (macOS)
```

### 2. Verify Determinism

The system produces byte-for-byte identical output when rendered with the same seed:

```bash
# Render twice with same seed
./build/PathTracerHeadless --seed 42 --out run1.ppm
./build/PathTracerHeadless --seed 42 --out run2.ppm

# Should be byte-for-byte identical
python3 tests/tools/sha256.py run1.ppm
python3 tests/tools/sha256.py run2.ppm

# Hashes should match exactly!
```

### 3. Golden Image Testing Workflow

**Initial Setup:**
```bash
# Generate baseline render (before any refactoring)
./build/PathTracerHeadless --width 256 --height 144 --spp 256 \
    --seed 42 --out tests/golden/before/scene01.ppm --verbose

git add tests/golden/before/scene01.ppm
git commit -m "Add golden image baseline for M0"
```

**After Code Changes:**
```bash
# Rebuild and generate new render
cmake --build build
./build/PathTracerHeadless --width 256 --height 144 --spp 256 \
    --seed 42 --out tests/golden/after/scene01.ppm --verbose

# Compare renders
python3 tests/tools/psnr.py \
    tests/golden/before/scene01.ppm \
    tests/golden/after/scene01.ppm --threshold 60.0 --verbose
```

### 4. Automated Testing with CTest

```bash
# Run all tests including golden image validation
cmake --build build && ctest --test-dir build --verbose

# Run only golden image tests
ctest --test-dir build -R "golden" --verbose
```

### Validation Results

✅ **All Tests Pass:**
- Headless rendering produces valid PPM output
- Determinism verified: multiple runs with same seed produce identical output
- PSNR validation confirms visual consistency (≥ 60 dB threshold)
- CMake integration working correctly with CTest
- M0 test suite fully automated and reproducible

## Debugging Tips

### Problem: Headless renderer crashes on init

**Check:**
- Metal device creation (works without window)
- Shader compilation (doesn't depend on view)
- Uniform buffer allocation (should work)

**Debug:**
```cpp
NSLog(@"Metal device: %@", m_device);
NSLog(@"Command queue: %@", m_commandQueue);
NSLog(@"Integrate pipeline: %@", m_integratePipeline);
```

### Problem: exportToPPM produces black/white image

**Check:**
- `m_accumulation.radianceSum()` is non-nil
- Sample count > 0
- HDR-to-LDR conversion (gamma, clamping)

**Debug:**
```cpp
NSLog(@"Radiance texture: %@ (%lux%lu)", radianceSum, radianceSum.width, radianceSum.height);
NSLog(@"Sample count at (0,0): %u", sampleCounts[0]);
NSLog(@"Radiance at (0,0): (%f, %f, %f)", radianceData[0], radianceData[1], radianceData[2]);
```

### Problem: Non-deterministic output even with fixed seed

**Check:**
- Shader actually uses `uniforms.fixedRngSeed`
- No uninitialized variables in C++ or Metal code
- Floating-point operations are associative (avoid reordering)
- No system randomness leaking in (e.g., `rand()`, `std::random_device`)

**Test:**
```bash
# Run 3 times, compare all hashes
for i in 1 2 3; do
    ./build/PathTracerHeadless --seed 42 --out run$i.ppm
    python3 tests/tools/sha256.py run$i.ppm
done
# All hashes should match!
```

## Milestone 0 Completion Summary

The golden image testing system is **fully complete and production-ready**. All components have been implemented and tested:

### Completed in This Session

1. ✅ **MetalRenderer headless initialization** - Full support for window-free operation
2. ✅ **Pipeline configuration** - Proper pixel format handling for headless mode
3. ✅ **Frame rendering** - Both GUI and headless rendering paths fully supported
4. ✅ **RNG seed implementation** - Deterministic rendering with fixed seeds
5. ✅ **PPM export** - Complete pixel readback and format conversion
6. ✅ **Accumulation management** - Flexible texture creation for both modes
7. ✅ **SettingsUtils integration** - Settings change detection and accumulation reset logic
8. ✅ **CMake build system** - PathTracerHeadless target and test automation
9. ✅ **Test suite** - M0 validation tests with automated execution
10. ✅ **Documentation** - Comprehensive guides for usage and implementation

### Production Capabilities

With this implementation, you can now:
- ✅ Generate **byte-for-byte reproducible renders** using fixed RNG seeds
- ✅ Validate refactors **don't change output** using hash comparison
- ✅ Perform **visual regression testing** using PSNR metrics
- ✅ **Automate golden image testing** in CI/CD pipelines
- ✅ **Track rendering correctness** across compiler and optimization changes
- ✅ **Build headless renders** with deterministic frame-by-frame output

### Integration Status

The system is integrated into:
- **CMake:** PathTracerHeadless target builds cleanly
- **CTest:** Automated test execution with clear pass/fail status
- **Git:** Golden image baselines committed and version-controlled
- **Python Tools:** SHA-256 and PSNR validation scripts fully functional

The testing infrastructure is ready for immediate use in development and CI/CD pipelines!
