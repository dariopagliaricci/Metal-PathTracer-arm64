# Tests

This folder holds the public checks that run from a fresh clone.

- `public/headless_smoke_test.sh` renders `scenes/smoke.scene` at 64x64 with
  4 samples per pixel and fails when no image is written. Pass the headless
  binary as its first argument:

  ```bash
  tests/public/headless_smoke_test.sh build/PathTracerHeadless
  ```

- `scenes/` holds the smoke scene and the ReSTIR fixtures (`restir/r1a`,
  `r1b`, `r2`, `r3`), which go with the thresholds in `validation/baselines`.
- `tools/` holds image comparison helpers (`psnr.py`, `pfm_metrics.py`,
  `sha256.py`) and two C++ test sources that the default CMake targets do not
  build.

`ctest --test-dir build` runs the unit checks that CMake registers in this tree:
`radiance_cache_training_target` and `shared_closures`.

The golden image test script of earlier releases is not part of v4.0.0.
`-DM0_TESTS=ON` registers the golden tests only when
`tests/tools/golden_test.sh` exists, and CMake prints a warning when it does not.
