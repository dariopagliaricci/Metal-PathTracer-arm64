# Contributing to Metal PathTracer

Thank you for your interest in contributing. This guide describes the project
scope, local build and validation workflow, and the renderer invariants that
contributions must preserve.

---

## Table of Contents

- [Scope and Platform Requirements](#scope-and-platform-requirements)
- [Repository Orientation](#repository-orientation)
- [Filing Issues](#filing-issues)
- [Proposing Changes](#proposing-changes)
- [Building Locally](#building-locally)
- [Validation](#validation)
- [Code Style](#code-style)
- [Pull Request Workflow](#pull-request-workflow)
- [Design Invariants](#design-invariants)

---

## Scope and Platform Requirements

Metal PathTracer targets macOS with Apple Silicon as the primary runtime. The
Metal GPU renderer is the main backend, with software BVH fallback/debug paths
and an optional Embree CPU backend for reference rendering and parity checks.

**Required for normal development:**

- macOS 12 or newer on Apple Silicon.
- Xcode / Apple Clang with Metal compiler support.
- CMake 3.24 or newer.
- A C++20-capable compiler.

**Optional dependencies:**

- Assimp, for the `PathTracerImport` and `PathTracerBistroAudit` tools.
- Embree 4.4 or newer, for the CPU headless reference backend.
- Vendored Intel Open Image Denoise 2.4.1 plus TBB under `external/oidn`, for
  CPU denoising. Source-only builds can disable this with
  `-DPATH_TRACER_ENABLE_OIDN=OFF`.

Do not add CUDA, Vulkan, OpenGL, or broad cross-platform abstraction layers. The
single-platform Metal focus is part of the renderer's research and validation
scope.

---

## Repository Orientation

- `src/renderer`, `include/renderer` - Metal renderer, scene resources,
  acceleration setup, pass graph, accumulation, denoise, UI, and transport
  memory management.
- `src/headless`, `include/headless` - deterministic CLI rendering, Metal
  headless execution, and optional Embree integration.
- `src/import`, `include/import` - Assimp import pipeline, Bistro audit, texture
  conversion, KTX2 writing/loading support, and import manifests.
- `src/assets`, `include/assets` - glTF loading and tangent generation.
- `include/shared` - shared data formats used by host code, tools, and shaders.
- `shaders` - Metal shader modules for core math, tracing, BSDFs, lighting,
  ReSTIR, wavefront scheduling, reconstruction, guiding, cache, volumes,
  spectral transport, caustics, scene access, and debug views.
- `assets` - public scene files and canonical assets. See `assets/CREDITS.md`
  for source and license notes.
- `tests` - public smoke tests, golden-image utilities, PFM metrics, and small
  test scenes.
- `docs` - public operational notes, including GUI hardware expectations.

---

## Filing Issues

Use the [GitHub issue tracker](../../issues) to report bugs, request features,
or ask questions.

**Bug reports should include:**

- macOS version, Apple Silicon chip model, and unified memory size.
- Xcode and CMake versions:
  ```bash
  xcodebuild -version
  cmake --version
  ```
- The smallest scene and command line that reproduces the problem.
- The selected backend and execution mode, for example `--backend=metal`,
  `--backend=embree`, `--executionMode=megakernel`, or
  `--executionMode=wavefront`.
- Whether the issue reproduces with `--force-hwrt`, `--force-swrt`, or
  `--enableSoftwareRayTracing=1`.
- Relevant rendering controls such as `--renderProfile`, `--directLightMode`,
  ReSTIR/ReGIR/RIS flags, denoise flags, or advanced transport modes.
- Any output metadata, debug bundle, PBR metrics JSON, direct-light audit JSON,
  material debug JSON, Metal logs, or GPU capture details that help isolate the
  issue.

**Feature requests should include:**

- The rendering problem or production workflow gap the feature addresses.
- Whether it affects Metal, Embree, importer tooling, shaders, metadata, tests,
  or assets.
- References to relevant papers, standards, or prior implementations.
- Expected validation strategy, especially if rendered output changes.

---

## Proposing Changes

Open an issue before starting significant work. This is especially important for
changes that affect:

- Shared material, light, vertex, scene, or uniform encodings.
- glTF loading, Assimp import, texture conversion, KTX2 handling, or asset
  manifests.
- Metal acceleration setup, HWRT/SWRT fallback behavior, or Embree parity.
- Megakernel, wavefront, pass graph, queue compaction, or scheduling policy.
- Direct lighting, MIS, RIS, ReGIR, ReSTIR DI/GI/PT, path guiding, or radiance
  cache behavior.
- Advanced transport prototypes: caustics, volumes, spectral transport, or
  experimental denoise paths.
- Output formats, deterministic settings, render queue JSON, metadata sidecars,
  debug bundles, checkpoints, or tiled render manifests.
- CMake options, CI behavior, public release packaging, or large public assets.

Small, well-scoped fixes such as typos, documentation updates, narrow build
fixes, or obvious single-line bug fixes may be opened as PRs directly.

---

## Building Locally

Basic Metal build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target PathTracer PathTracerHeadless
```

CI-equivalent source-only headless build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPATH_TRACER_ENABLE_OIDN=OFF
cmake --build build --target PathTracerHeadless
./build/PathTracerHeadless --help
```

Embree-enabled build:

```bash
brew install embree
cmake -S . -B build-embree -DCMAKE_BUILD_TYPE=Release -DPATH_TRACER_ENABLE_EMBREE=ON
cmake --build build-embree --target PathTracerHeadless EmbreeSmokeTest
```

Importer build with Assimp:

```bash
brew install assimp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target PathTracerImport PathTracerBistroAudit
```

Useful CMake options:

- `PATH_TRACER_ENABLE_EMBREE=ON` builds and links the Embree CPU backend.
- `PATH_TRACER_ENABLE_ASSIMP=ON` enables Assimp-backed import when Assimp is
  found. Without Assimp, importer targets build in stub mode.
- `PATH_TRACER_ENABLE_OIDN=OFF` disables optional OIDN CPU denoising.
- `PATH_TRACER_BUILD_EMBREE_SMOKE_ONLY=ON` builds only the Embree smoke target.
- `PT_DEBUG_TOOLS=ON` enables HWRT/SWRT debug tooling in the GUI target.
- `PT_MNEE_SWRT_RAYS=ON` and `PT_MNEE_OCCLUSION_PARITY=ON` enable MNEE parity
  debugging.
- `STRICT=ON` enables stricter warnings.
- `M0_TESTS=ON` enables optional golden-image CTest entries.

---

## Validation

At minimum, build the changed targets and run a deterministic smoke render:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

build/PathTracerHeadless \
  --scene=tests/scenes/smoke.scene \
  --width=64 \
  --height=64 \
  --sppTotal=4 \
  --maxDepth=4 \
  --seed=1337 \
  --enableSoftwareRayTracing=1 \
  --format=ppm \
  --output=/tmp/pathtracer_smoke.ppm
```

If the public smoke-test helper is present, prefer it for quick regression
checks:

```bash
tests/public/headless_smoke_test.sh
```

Run CTest when tests are configured:

```bash
ctest --test-dir build --verbose
```

Optional Embree validation:

```bash
ctest --test-dir build-embree -R embree --verbose
```

For rendering changes, compare deterministic outputs with fixed scene, seed,
resolution, format, sample count, backend, and execution mode. Use PFM/EXR for
linear HDR comparisons and PNG/PPM only for visual LDR checks.

```bash
python3 tests/tools/pfm_metrics.py before.pfm after.pfm --clamp 1.0 --verbose
```

Optional golden-image workflow:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DM0_TESTS=ON
cmake --build build --target PathTracerHeadless
tests/tools/golden_test.sh generate after
tests/tools/golden_test.sh compare --verbose
```

If a change intentionally alters rendered output, include before/after images or
metrics in the PR and explain why the new result is correct.

---

## Code Style

### C++ and Objective-C++ Host Code

- Use C++20.
- Match the style of nearby files.
- Use 4-space indentation and no tabs.
- Use `snake_case` for variables and functions; use `PascalCase` for types and
  structs unless the surrounding file already establishes a different convention.
- Keep includes minimal and ordered: standard library, system frameworks,
  third-party, then project headers.
- Do not put `using namespace std;` in headers.
- Prefer RAII, `const`, and `constexpr` where they clarify ownership or intent.
- Avoid raw `new`/`delete`.

### Metal Shading Language

- Match the existing shader module's naming and layout.
- Prefer explicit address-space qualifiers such as `device`, `constant`, and
  `threadgroup`.
- Keep host/shader struct layouts synchronized when editing shared data.
- Avoid data-dependent inner-loop bounds in hot kernels unless the pattern is
  already established and measured.
- Keep kernel entry points and helpers in `snake_case`.

### General

- Do not introduce compiler warnings.
- Do not check in generated build artifacts, `.air`, `.metallib`, local render
  outputs, or temporary debug files.
- Do not add large binary assets without prior discussion. Update
  `assets/CREDITS.md` for new public assets.
- Keep docs, CLI help, and examples aligned when adding or renaming flags.

---

## Pull Request Workflow

1. Fork the repository and create a topic branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make a focused change. Keep unrelated refactors out of the PR.

3. Build and validate the affected targets. Include the exact commands you ran
   in the PR description.

4. If rendered output changes, include before/after comparisons, the scene,
   seed, resolution, sample count, backend, execution mode, and comparison
   metric.

5. Open a pull request against `main`. The PR description should summarize what
   changed, why it changed, which paths are affected, and any remaining risk.

6. CI currently builds a source-only public headless configuration on `macos-14`
   with OIDN disabled and runs `PathTracerHeadless --help`. PRs should pass CI
   before review proceeds.

---

## Design Invariants

These invariants are core to the renderer's correctness and research utility.

### 1. Deterministic Headless Rendering

Headless renders must remain reproducible when scene, seed, resolution, sample
count, backend, execution mode, and relevant feature flags are fixed. Do not
introduce system randomness, uninitialized shader state, or nondeterministic
host-side ordering into deterministic paths.

### 2. HWRT/SWRT Parity

The Metal hardware ray tracing path and software BVH path must produce equivalent
output within documented numerical tolerance unless an intentional algorithmic
difference is explicitly discussed. Backend fallback must be logged and must not
silently hide missing HWRT support in validation workflows that use
`--force-hwrt` or `--force-swrt`.

### 3. Embree as Reference Backend

Embree is an optional CPU reference backend for visual-output parity, asset
validation, and backend comparison. Shared scene, material, lighting, and output
format changes should consider whether Embree behavior or documentation also
needs to change.

### 4. Shared Host/Shader Encodings

Shared material, light, vertex, scene, and uniform structs cross host code,
tools, and Metal shaders. Changes must preserve binary layout expectations or
explicitly document and version the break. Update both CPU and shader consumers
together.

### 5. MIS and Sampling Correctness

Direct lighting uses MIS across BSDF and light sampling. New light types,
sampling strategies, RIS/ReGIR/ReSTIR paths, or BSDF changes must remain
unbiased or clearly marked as experimental with validation evidence.

### 6. Research Paths Stay Gated

High-risk systems such as ReSTIR GI/PT, path guiding, radiance cache, SVGF,
path-space caustics, volumes, and spectral transport are opt-in research paths.
Do not make experimental behavior the default without an issue, validation plan,
and documentation update. Record selected modes in metadata/debug output where
the current workflow supports it.

### 7. Large-Scene Memory Policy

Large-scene checks against `MTLDevice.recommendedMaxWorkingSetSize` are part of
the production workflow. Changes to texture handling, acceleration builds,
wavefront queues, ReSTIR buffers, denoise buffers, or metadata must preserve
clear memory-budget diagnostics.

### 8. Import and Asset Determinism

`PathTracerImport` should produce deterministic glTF/GLB output and an
`import_manifest.json` for supported static scenes. Texture conversion and KTX2
handling should be reproducible. New public assets must include source and
license notes.

### 9. Apple Silicon and Metal Scope

Keep the main renderer focused on Metal and Apple Silicon. Optional reference
tools are welcome when they support validation, but they should not turn the
project into a cross-platform renderer.

---

## Questions

If you are unsure whether a change is in scope or compatible with these
invariants, open a [GitHub issue](../../issues) with the label `question` before
investing significant implementation time.
