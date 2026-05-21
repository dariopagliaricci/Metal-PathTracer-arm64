# Metal PathTracer

A physically based renderer for macOS and Apple Silicon, written in C++20,
Objective-C++, and Metal.

Metal PathTracer is a research-grade production renderer with multiple transport
backends, large-scene asset onboarding, deterministic headless rendering,
runtime diagnostics, and explicitly gated experimental systems.

The current public release line is `v3.0.0`. This release includes the renderer
scalability foundation, Bistro-class large-scene support, ReSTIR/RIS/ReGIR
research paths, wavefront execution work, Embree reference rendering,
importer/texture conversion tooling, and production metadata/debug-bundle
support.

## Current Status

The default renderer remains conservative: core path tracing paths are enabled by
default, while high-risk research modes are opt-in and controlled through CLI
flags and UI/debug settings.

Production-ready or primary paths:

- Metal GPU renderer with hardware ray tracing where available and a software
  BVH fallback/debug path.
- Progressive physically based path tracing with deterministic headless output.
- Megakernel execution path plus an actively developed wavefront execution path.
- glTF 2.0 / GLB runtime loading with PBR metallic-roughness materials, texture
  transforms, tangent generation, normal maps, emissive materials, transmission
  fallback, and KTX2 texture loading for imported assets.
- Large-scene reporting and memory-budget policy checks before acceleration
  structure builds.
- Headless render profiles, metadata sidecars, settings JSON, debug bundles,
  checkpoint manifests, tiled render manifests, PBR metrics, and deterministic
  render queue item JSON.
- Intel Open Image Denoise CPU postprocess when the vendored native libraries are
  present, plus Metal SVGF-style denoise as a gated research path.
- `PathTracerImport`, an importer-only Assimp pipeline that converts static FBX,
  OBJ, and other supported source formats into deterministic glTF/GLB output.

Gated research paths that are present but opt-in:

- RIS-family direct lighting, ReGIR world reuse, ReSTIR DI, and DI+ReGIR hybrid.
- Bounded ReSTIR GI prototype and ReSTIR PT research/experimental path reuse.
- Path guiding and radiance cache prototypes.
- Path-space caustic transport, homogeneous volume transport, and
  hero-wavelength spectral transport.
- ReSTIR debug inspector views and counters.
- Wavefront active-work scheduling through selectable preview, final, offline,
  and research scheduling policies.

Embree is maintained as a selectable CPU reference renderer for visual-output
parity checks, asset validation, and backend comparison. Metal remains the
primary GPU renderer.

## Repository Map

- `src/renderer`, `include/renderer` - Metal renderer, scene resources,
  pass graph, transport memory registry, accumulation, denoise, UI, and
  acceleration setup.
- `src/headless`, `include/headless` - Metal and Embree headless backends.
- `src/import`, `include/import` - Assimp import pipeline, Bistro audit, texture
  conversion, KTX2 writing/loading support.
- `shaders` - Metal shader tree split by core math, tracing, BSDF, lighting,
  ReSTIR, wavefront, reconstruction, guiding, cache, volume, spectral, caustics,
  scene access, and debug views.
- `assets` - scene files and canonical assets, including Bistro,
  San Miguel, Living Room, Bitterli, automotive, Khronos, Blender, and synthetic
  fixtures.
- `tests` - public smoke tests and optional renderer regression fixtures when
  included in the release package.
- `Images/Images-Legacy`, `Images/Metal-Wavefront-Images`, `Images/Embree-Renders` - public gallery
  render sets used below.

## Renderings Gallery

The galleries below show selected release renders from the legacy material set,
current Metal wavefront output, and Embree reference output.

### Legacy Gallery

Images from `Images/Images-Legacy`.

<div align="center">

![Hygieia hardware ray tracing legacy render](Images/Images-Legacy/hygieia_HWRT.jpg)

![Hygieia metal material legacy render](Images/Images-Legacy/hygieia_metal.jpeg)

![glTF BoomBox legacy render](Images/Images-Legacy/boombox.jpg)

![Damaged Helmet legacy render 1](Images/Images-Legacy/damaged-helmet-01.jpg)

![Damaged Helmet legacy render 2](Images/Images-Legacy/damaged-helmet-02.jpg)

![Marble and wax validation legacy render](Images/Images-Legacy/marble-wax-validation.jpg)

![Plastic material validation legacy render](Images/Images-Legacy/plastic_validation.jpg)

</div>

### Metal Wavefront Gallery

Current Metal wavefront renders from `Images/Metal-Wavefront-Images`.

<div align="center">

![Bistro full sun ReSTIR wavefront render](Images/Metal-Wavefront-Images/bistro_full_sun_restir_alt_denoised_1024spp_denoised.jpg)

![Bistro night open-front ReSTIR wavefront render](Images/Metal-Wavefront-Images/bistro_night_openfront_restir_denoised_1024spp_denoised.jpg)

![San Miguel balcony sun ReSTIR wavefront render](Images/Metal-Wavefront-Images/san_miguel_balcony_sun_restir_denoised_1024spp_denoised.jpg)

![San Miguel sunlit alternate ReSTIR wavefront render](Images/Metal-Wavefront-Images/san_miguel_2_restir_sunlit_alternate_denoised_1024spp_denoised.jpg)

![Sponza sunlight HDRI ReSTIR wavefront render](Images/Metal-Wavefront-Images/sponza_sunlight_hdri_restir_denoised_1024spp_denoised.jpg)

![Bitterli staircase wavefront render](Images/Metal-Wavefront-Images/bitterli_staircase_denoised_1024spp_denoised.jpg)

![Living room ReGIR stress wavefront render](Images/Metal-Wavefront-Images/living_room_regir_stress_denoised_1024spp_denoised.jpg)

![Living room practical lights ReGIR stress wavefront render](Images/Metal-Wavefront-Images/living_room_regir_practicals_stress_denoised_1024spp_denoised.jpg)

![LuxCore balls wavefront render](Images/Metal-Wavefront-Images/luxcoreballs_denoised_1024spp_denoised.jpg)

![McLaren MCL35M wavefront render](Images/Metal-Wavefront-Images/f1_bulk_2021_f1_mclaren_mcl35m_denoised_1024spp_denoised.jpg)

![Mercedes W12 wavefront render](Images/Metal-Wavefront-Images/f1_bulk_2021_f1_mercedes_benz_w12_denoised_1024spp_denoised.jpg)

![Alpine A521 wavefront render](Images/Metal-Wavefront-Images/f1_bulk_2021_f1_alpine_a521_denoised_1024spp_denoised.jpg)

![Mercedes W11 wavefront render](Images/Metal-Wavefront-Images/f1_bulk_2020_f1_mercedes_benz_w11_denoised_1024spp_denoised.jpg)

![McLaren MCL35 wavefront render](Images/Metal-Wavefront-Images/f1_bulk_2020_f1_mclaren_mcl35_denoised_1024spp_denoised.jpg)

![Racing Point RP20 wavefront render](Images/Metal-Wavefront-Images/f1_bulk_2020_f1_racing_point_rp20_denoised_1024spp_denoised.jpg)

![Alfa Romeo C39 wavefront render](Images/Metal-Wavefront-Images/f1_bulk_2020_f1_alfa_romeo_c39_denoised_1024spp_denoised.jpg)

![Haas VF20 wavefront render](Images/Metal-Wavefront-Images/f1_bulk_2020_f1_haas_vf20_denoised_1024spp_denoised.jpg)

![Williams FW43 wavefront render](Images/Metal-Wavefront-Images/f1_bulk_2020_f1_whilliams_fw43_denoised_1024spp_denoised.jpg)

![Mercedes W10 wavefront render](Images/Metal-Wavefront-Images/f1_bulk_2019_f1_mercedes_benz_w10_denoised_1024spp_denoised.jpg)

![McLaren MCL34 wavefront render](Images/Metal-Wavefront-Images/f1_bulk_2019_mclaren_mcl34_denoised_1024spp_denoised.jpg)

![Red Bull RB15 wavefront render](Images/Metal-Wavefront-Images/f1_bulk_2019_f1_red_bull_rb15_denoised_1024spp_denoised.jpg)

![Alfa Romeo C38 wavefront render](Images/Metal-Wavefront-Images/f1_bulk_2019_f1_alfa_romeo_c38_denoised_1024spp_denoised.jpg)

![Haas VF-19 wavefront render](Images/Metal-Wavefront-Images/f1_bulk_2019_haas_vf_19_denoised_1024spp_denoised.jpg)

</div>

### Embree Gallery

CPU reference renders from `Images/Embree-Renders`.

<div align="center">

![Bistro sun Embree render](Images/Embree-Renders/bistro_full_sun_restir_alt_16spp_denoised_denoised.jpg)

![Sponza sunlight Embree render](Images/Embree-Renders/sponza_sunlight_hdri_restir_denoised_16spp_denoised.jpg)

![Bitterli staircase Embree render](Images/Embree-Renders/bitterli_staircase_denoised_16spp_denoised.jpg)

![San Miguel balcony Embree render](Images/Embree-Renders/san_miguel_balcony_sun_restir_denoised_16spp_denoised.jpg)

![San Miguel alternate Embree render](Images/Embree-Renders/san_miguel_2_restir_sunlit_alternate_denoised_16spp_denoised.jpg)

![Living room Embree render](Images/Embree-Renders/living_room_denoised_16spp_denoised.jpg)

![glTF Damaged Helmet Embree render](Images/Embree-Renders/gltf_damaged_helmet_denoised_16spp_denoised.jpg)

![Ajax Embree render](Images/Embree-Renders/ajax_denoised_16spp_denoised.jpg)

![Hygieia Embree render](Images/Embree-Renders/hygieia_denoised_16spp_denoised.jpg)

![Hygieia other Embree render](Images/Embree-Renders/hygieia_other_denoised_16spp_denoised.jpg)

![Jason alloys Embree render](Images/Embree-Renders/jason_alloys_denoised_16spp_denoised.jpg)

![Lucy glass Embree render](Images/Embree-Renders/lucy_glass_denoised_16spp_denoised.jpg)

![Lucy plastic Embree render](Images/Embree-Renders/lucy_plastic_denoised_16spp_denoised.jpg)

![Dragon car paint Embree render](Images/Embree-Renders/dragon_carpaint_denoised_16spp_denoised.jpg)

![Bugatti Chiron Embree render](Images/Embree-Renders/bugatti_chiron_top_edition_matchref_16spp_denoised_denoised.jpg)

![Ferrari F14 T Embree render](Images/Embree-Renders/ferrari_f14_t_2014_matchref_denoised_16spp_denoised.jpg)

![McLaren MCL35M Embree render](Images/Embree-Renders/f1_bulk_2021_f1_mclaren_mcl35m_denoised_16spp_denoised.jpg)

![Mercedes W12 Embree render](Images/Embree-Renders/f1_bulk_2021_f1_mercedes_benz_w12_denoised_16spp_denoised.jpg)

![Alpine A521 Embree render](Images/Embree-Renders/f1_bulk_2021_f1_alpine_a521_denoised_16spp_denoised.jpg)

![Mercedes W11 Embree render](Images/Embree-Renders/f1_bulk_2020_f1_mercedes_benz_w11_denoised_16spp_denoised.jpg)

![McLaren MCL35 Embree render](Images/Embree-Renders/f1_bulk_2020_f1_mclaren_mcl35_denoised_16spp_denoised.jpg)

![Racing Point RP20 Embree render](Images/Embree-Renders/f1_bulk_2020_f1_racing_point_rp20_denoised_16spp_denoised.jpg)

![Alfa Romeo C39 Embree render](Images/Embree-Renders/f1_bulk_2020_f1_alfa_romeo_c39_denoised_16spp_denoised.jpg)

![Haas VF20 Embree render](Images/Embree-Renders/f1_bulk_2020_f1_haas_vf20_denoised_16spp_denoised.jpg)

![Williams FW43 Embree render](Images/Embree-Renders/f1_bulk_2020_f1_whilliams_fw43_denoised_16spp_denoised.jpg)

![Mercedes W10 Embree render](Images/Embree-Renders/f1_bulk_2019_f1_mercedes_benz_w10_denoised_16spp_denoised.jpg)

![McLaren MCL34 Embree render](Images/Embree-Renders/f1_bulk_2019_mclaren_mcl34_denoised_16spp_denoised.jpg)

![Red Bull RB15 Embree render](Images/Embree-Renders/f1_bulk_2019_f1_red_bull_rb15_denoised_16spp_denoised.jpg)

![Alfa Romeo C38 Embree render](Images/Embree-Renders/f1_bulk_2019_f1_alfa_romeo_c38_denoised_16spp_denoised.jpg)

![Haas VF-19 Embree render](Images/Embree-Renders/f1_bulk_2019_haas_vf_19_denoised_16spp_denoised.jpg)

</div>

## Features

### Transport and Rendering

- Progressive path tracing with temporal accumulation.
- Metal hardware ray tracing path using TLAS/BLAS where supported.
- Metal software BVH path for fallback, parity checks, and debug workflows.
- Megakernel and wavefront execution modes.
- Configurable path depth, Russian roulette, render scale, exposure, tonemapping,
  bloom, color space, and deterministic seeds.
- Environment map importance sampling and MIS.
- Specular NEE and MNEE caustics.
- Optional firefly clamp modes for production output control.
- Path-space caustic transport prototype.
- Homogeneous volume transport prototype with isotropic or Henyey-Greenstein
  phase.
- Hero-wavelength spectral transport prototype with dielectric dispersion.

### Materials and Assets

- Lambertian, metal, dielectric/glass, diffuse light, plastic, subsurface, car
  paint, and glTF PBR metallic-roughness materials.
- Analytic spheres, rectangles, boxes, OBJ/PLY meshes, and glTF/GLB meshes.
- MikkTSpace tangent generation for normal mapped assets.
- Runtime KTX2 texture loading for converted importer output.
- Canonical scenes for Bistro, Sponza, San Miguel, Living Room, Bitterli,
  LuxCore balls, automotive assets, glTF sample assets, and synthetic fixtures.
- Texture clamp and texture-budget controls for large-scene bring-up.

### Research Lighting Stack

- Direct light modes: `legacy`, `baseline_emissive`, `ris`, `ris_spatial`,
  `ris_temporal`, `ris_world`, `ris_regir`, `restir_di`, and
  `restir_di_regir_hybrid`.
- ReSTIR DI explicit pass graph path for headless and opt-in GUI profiling.
- ReSTIR GI diffuse prototype.
- ReSTIR PT research and experimental path reuse modes.
- ReGIR/world-space candidate reuse.
- Path guiding and radiance cache prototypes.
- ReSTIR debug AOVs for candidate source, reservoir confidence, ReGIR cell ID,
  path-guiding mask, ReSTIR PT reuse mask, SVGF variance, NaN/Inf mask, queue
  failures, and radiance cache diagnostics.

### Tooling

- GUI app with ImGui renderer controls, material editing, object transform
  editing through ImGuizmo, performance panels, EXR export, and presentation
  mode.
- Headless renderer for reproducible offline rendering and validation.
- Optional Embree CPU backend for reference rendering and visual-output parity
  checks.
- Importer executable for static scene conversion into glTF/GLB.
- Bistro material audit executable.
- PBR metrics, debug AOVs, material-channel inspection, and deterministic
  render metadata for repeatable renderer analysis.

## Requirements

- macOS 12 or newer.
- Apple Silicon is the primary target.
- Xcode / Apple Clang with Metal compiler support.
- CMake 3.24 or newer.
- Optional: Assimp for `PathTracerImport`.
- Optional: Embree 4.4 or newer for the CPU headless backend.
- Vendored Intel Open Image Denoise 2.4.1 and TBB native libraries in
  `external/oidn` for CPU denoising. The release is validated against the
  bundled OIDN/TBB runtime. Newer OIDN releases should be tested by replacing
  `external/oidn`, reconfiguring CMake, and rerunning validation; until then,
  the bundled version is the supported runtime. Source-only rebuilds can disable
  it with `-DPATH_TRACER_ENABLE_OIDN=OFF`.

Apple exposes Metal ray tracing capability differently across devices and OS
versions. The renderer logs backend selection explicitly, and `--force-hwrt` /
`--force-swrt` can be used when a validation run must fail instead of silently
falling back.

Large scenes are checked against `MTLDevice.recommendedMaxWorkingSetSize` before
acceleration-structure build. This is a device-specific working-set budget, not
the same as total unified memory.

For practical interactive GUI expectations across Apple Silicon memory tiers,
see [GUI Hardware Expectations](docs/GUI_HARDWARE_EXPECTATIONS.md). In short,
16 GB unified memory is not enough to guarantee that every asset-pack scene can
load in the GUI; 32 GB to 36 GB is the practical large-scene baseline, and
64 GB or more is recommended for comfortable large-scene GUI work.

## Building

Basic Metal build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target PathTracer PathTracerHeadless
```

Dedicated wavefront/Metal development build:

```bash
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal --clean-first --target PathTracer PathTracerHeadless
```

Embree-enabled build:

```bash
brew install embree
cmake -S . -B build-embree -DCMAKE_BUILD_TYPE=Release -DPATH_TRACER_ENABLE_EMBREE=ON
cmake --build build-embree --clean-first --target PathTracerHeadless EmbreeSmokeTest
```

Importer build with Assimp:

```bash
brew install assimp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target PathTracerImport PathTracerBistroAudit
```

Optional OIDN CPU denoising runtime:

```bash
OIDN_VERSION=2.4.1
curl -L \
  -o /tmp/oidn-${OIDN_VERSION}.arm64.macos.tar.gz \
  https://github.com/RenderKit/oidn/releases/download/v${OIDN_VERSION}/oidn-${OIDN_VERSION}.arm64.macos.tar.gz

tar -xzf /tmp/oidn-${OIDN_VERSION}.arm64.macos.tar.gz -C /tmp

mkdir -p external/oidn/include external/oidn/lib
cp -a /tmp/oidn-${OIDN_VERSION}.arm64.macos/include/. external/oidn/include/
cp -a /tmp/oidn-${OIDN_VERSION}.arm64.macos/lib/. external/oidn/lib/

xattr -dr com.apple.quarantine external/oidn/lib/*.dylib 2>/dev/null || true
```

Reconfigure after installing OIDN:

```bash
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Main CMake options:

| Option | Default | Description |
| --- | --- | --- |
| `PATH_TRACER_ENABLE_EMBREE` | `OFF` | Build and link the Embree CPU backend. |
| `PATH_TRACER_ENABLE_ASSIMP` | `ON` | Enable the Assimp-backed importer when Assimp is found. |
| `PATH_TRACER_ENABLE_OIDN` | `ON` | Enable OIDN if vendored native libraries are present. |
| `PATH_TRACER_BUILD_EMBREE_SMOKE_ONLY` | `OFF` | Build only the Embree smoke target. |
| `PT_DEBUG_TOOLS` | `OFF` | Enable HWRT/SWRT path debug and parity tooling in the GUI target. |
| `PT_MNEE_SWRT_RAYS` | `OFF` | Force MNEE rays through SWRT for HWRT debug. |
| `PT_MNEE_OCCLUSION_PARITY` | `OFF` | Compare MNEE visibility through HWRT and SWRT. |
| `STRICT` | `OFF` | Enable strict warnings. |
| `M0_TESTS` | `OFF` | Enable optional golden-image CTest entries. |

Generated targets:

- `PathTracer.app` - interactive GUI renderer.
- `PathTracerHeadless` - deterministic CLI renderer.
- `PathTracerImport` - importer-only static scene converter.
- `PathTracerBistroAudit` - Bistro material/texture audit tool.
- `EmbreeSmokeTest` - Embree API smoke target when Embree is enabled.

## Running the GUI

```bash
open build/PathTracer.app
```

Debug launch:

```bash
./build/PathTracer.app/Contents/MacOS/PathTracer
```

Large-scene / ReSTIR DI interactive profiling launch:

```bash
PT_HWRT_BUILD_POLICY=compact_memory \
PT_ENABLE_INTERACTIVE_EXPLICIT_RESTIR_DI=1 \
./build-metal/PathTracer.app/Contents/MacOS/PathTracer --scene=bistro_full
```

### Large-Scene Apple Silicon Guidance

Large scenes such as Bistro, San Miguel, and Sponza can put heavy pressure on
Apple Silicon unified memory, especially in the interactive GUI. Geometry,
textures, Metal acceleration structures, render targets, denoising state, and
optional ReSTIR buffers all share the same physical memory pool as macOS and
other applications.

On older or lower-memory Apple Silicon systems, expect the GUI to become less
responsive when these scenes are resident. `PT_HWRT_BUILD_POLICY=compact_memory`
is recommended for large static scenes because it reduces acceleration-structure
memory pressure, but it does not reduce texture memory, scene buffers, render
targets, or ReSTIR state. For constrained systems, prefer headless rendering for
final output, close other memory-heavy applications, start with smaller scenes,
and avoid enabling the explicit interactive ReSTIR DI path unless profiling that
mode specifically. See [GUI Hardware Expectations](docs/GUI_HARDWARE_EXPECTATIONS.md)
for the practical memory-tier guidance.

Presentation mode:

```bash
./build/PathTracer.app/Contents/MacOS/PathTracer --presentation=1
```

Supported interactive GUI command-line flags:

| Flag | Values | Purpose |
| --- | --- | --- |
| `--scene=<id-or-path>` / `--scene <id-or-path>` | Scene ID or `.scene` path | Starts the GUI on a specific scene instead of the default startup scene. |
| `--presentation=1` | `1`, `true`, `on`, `0`, `false`, `off` | Starts with presentation mode enabled or disabled. Presentation mode can also be changed from the UI. |

Useful interactive runtime environment flags:

| Variable | Values | Purpose |
| --- | --- | --- |
| `PT_HWRT_BUILD_POLICY` | `fast_build`, `fast_trace`, `compact_memory`, `dynamic_update` | Overrides the Metal hardware acceleration-structure build policy. Use `compact_memory` for large static scenes when memory footprint matters. |
| `PT_ENABLE_INTERACTIVE_EXPLICIT_RESTIR_DI` | `1` / `0` | Enables the explicit ReSTIR DI pass graph in the interactive GUI when `restir_di` or `restir_di_regir_hybrid` direct lighting is selected. Without it, the GUI keeps the lower-overhead fused path for responsiveness. |
| `PT_SCENE_LOAD_DIAGNOSTICS` | `1` / `0` | Prints scene-loading, asset-resolution, and staged-load diagnostics. |
| `PT_SCENE_LOAD_DIAGNOSTICS_VERBOSE` | `1` / `0` | Adds verbose scene-load diagnostics; requires `PT_SCENE_LOAD_DIAGNOSTICS=1`. |
| `PT_HWRT_KEEP_TRIANGLE_BUFFER` | `1` / `0` | Keeps CPU triangle data available after hardware acceleration setup for debugging and parity inspection. |
| `PT_CAPTURE_DELAY_SECONDS` | Floating-point seconds | Delays automated capture/export flows that need the first frames to settle. |

Low-level dispatch isolation/debug variables:

| Variable | Values | Purpose |
| --- | --- | --- |
| `PT_SKIP_BLAS_USE_RESOURCES` | `1` / `0` | Skips BLAS `useResource` binding for isolation when debugging Metal resource residency issues. |
| `PT_SKIP_TEXTURE_USE_RESOURCES` | `1` / `0` | Skips texture `useResource` binding for isolation when debugging texture residency issues. |
| `PT_FORCE_TG_WIDTH` / `PT_FORCE_TG_HEIGHT` | Positive integers | Overrides selected compute threadgroup dimensions for kernel-dispatch experiments. |
| `PT_RESTIR_DI_VALIDATE_FUSED_REFERENCE` | `1` / `0` | Headless validation switch for the fused ReSTIR DI reference path; keep disabled for normal interactive GUI use. |

Environment flags can be prefixed on the launch command, as above, or exported
once in the shell before starting `PathTracer.app`.

## Headless Rendering

Minimal render:

```bash
build/PathTracerHeadless \
  --scene=sponza \
  --width=1280 \
  --height=720 \
  --spp=1024 \
  --seed=42 \
  --out=renders/sponza.exr \
  --verbose
```

Metal wavefront render:

```bash
build/PathTracerHeadless \
  --scene=bistro_full \
  --renderProfile=final \
  --executionMode=wavefront \
  --directLightMode=restir_di_regir_hybrid \
  --denoise=1 \
  --out=renders/bistro_full_wavefront.exr \
  --verbose
```

Embree reference render:

```bash
build-embree/PathTracerHeadless \
  --backend=embree \
  --scene=assets/plastic.scene \
  --width=512 \
  --height=512 \
  --sppTotal=32 \
  --threads=8 \
  --denoise=1 \
  --format=png \
  --output=renders/plastic_embree.png
```

Render profiles:

| Profile | Defaults |
| --- | --- |
| `preview` | 64 spp, wavefront preview scheduling. |
| `lookdev` | 256 spp, wavefront final scheduling. |
| `final` | 1024 spp, wavefront offline scheduling. |
| `reference` | 4096 spp, megakernel, fixed seed if none is supplied. |
| `debug` | 16 spp, megakernel, ReSTIR debug counters, PBR metrics. |
| `custom` | No profile defaults. |

Important headless flag groups:

- Scene/output: `--scene`, `--output` / `--out`, `--format`, `--width`,
  `--height`, `--sppTotal` / `--spp`, `--seed`, `--maxDepth`.
- Backend: `--backend=metal|embree`, `--enableEmbree`, `--threads`,
  `--force-hwrt`, `--force-swrt`, `--enableSoftwareRayTracing`.
- Execution: `--executionMode=megakernel|wavefront`,
  `--wavefront-policy=preview|final|offline|research`,
  `--wavefront-compaction=0|1`.
- Lighting/reuse: `--directLightMode`, `--ris-m`, `--spatial-reuse-k`,
  `--world-reuse-cell-size`, `--restirGiMode`, `--restirPtMode`,
  `--pathGuidingMode`, `--radianceCacheMode`.
- Advanced transport: `--causticTransportMode`, `--volumeTransportMode`,
  `--spectralMode`, plus their related tuning flags.
- Large scenes: `--stats`, `--budget-policy=strict|warn|ignore`,
  `--pilot-mode=geo_only|albedo_only|full_clamped`, `--texture-max-dim`,
  `--max-texture-bytes`.
- Output quality: `--tonemap`, `--exposure`, `--firefly-clamp-mode`,
  `--firefly-clamp-value`, `--denoise`, OIDN guide flags, `--svgfDenoise`.
- Production metadata: `--pbrMetrics`, `--pbrMetricsJson`,
  `--outputMetadata`, `--outputMetadataJson`, `--settingsJson`,
  `--renderQueueItemJson`, `--runRenderQueueItemJson`, `--debugBundleDir`,
  `--checkpointManifestJson`, `--resumeCheckpointJson`, `--tileManifestJson`,
  `--tileSize`.
- Debug/audit: `--materialMrDebugJson`, `--directLightAuditJson`,
  `--restirDebug`, `--restirDebugView`, `--restirDebugCounters`,
  `--restirDebugMetricsJson`, `--cameraSearchReport`, `--debugPath`,
  `--debugPixel`, `--parityAssert`, material texture debug flags, and AOV
  visualization flags.

Run `build/PathTracerHeadless --help` for the complete current CLI surface.

### Gated Research Mode Examples

The advanced transport systems are explicit opt-in paths. They are intended for
renderer research, parity work, and controlled production experiments where the
selected mode is recorded in metadata/debug output.

ReSTIR DI plus ReGIR hybrid on the Metal wavefront path:

```bash
build/PathTracerHeadless \
  --scene=bistro_full \
  --renderProfile=final \
  --executionMode=wavefront \
  --wavefront-policy=offline \
  --directLightMode=restir_di_regir_hybrid \
  --outputMetadataJson=renders/bistro_restir_metadata.json \
  --out=renders/bistro_restir_regir.exr
```

ReSTIR GI diffuse reuse:

```bash
build/PathTracerHeadless \
  --scene=living_room \
  --renderProfile=lookdev \
  --restirGiMode=restir_gi_diffuse \
  --directLightMode=restir_di \
  --out=renders/living_room_restir_gi.exr
```

Path guiding and radiance cache prototypes:

```bash
build/PathTracerHeadless \
  --scene=sponza \
  --renderProfile=lookdev \
  --pathGuidingMode=path_guiding \
  --pathGuidingStrength=0.35 \
  --radianceCacheMode=radiance_cache \
  --radianceCacheMinConfidence=0.35 \
  --out=renders/sponza_guiding_cache.exr
```

Path-space caustics, homogeneous volumes, and hero-wavelength spectral transport:

```bash
build/PathTracerHeadless \
  --scene=suzanne_caustics \
  --renderProfile=lookdev \
  --causticTransportMode=path_space \
  --volumeTransportMode=homogeneous \
  --volumeSigmaA=0.02,0.02,0.02 \
  --volumeSigmaS=0.04,0.04,0.04 \
  --spectralMode=hero_wavelength \
  --spectralDispersionStrength=0.012 \
  --out=renders/suzanne_research_transport.exr
```

Notes:

- HDR outputs (`exr`, `pfm`) are written in linear space.
- Embree uses the same `--format` output surface as Metal. EXR/PFM are linear
  HDR outputs; PNG/PPM are tonemapped visual outputs.
- `--denoise=1` writes a second file with `_denoised` before the extension.
- Metal backend logs and large-scene reports are diagnostics, not progress
  indicators. Use `--verbose` for progress with percentage and ETA.

## Import Pipeline

Runtime rendering is glTF-first. FBX and broad DCC import are handled offline by
`PathTracerImport`.

```bash
build/PathTracerImport \
  --input assets/canonical/bistro/source/BistroExterior.fbx \
  --output assets/canonical/bistro/imported_exterior \
  --format glb \
  --textures convert \
  --import-mode canonical
```

Importer flags:

- `--format glb|gltf` - output format, default `glb`.
- `--textures copy|convert|embed|link` - texture policy, default `copy`.
- `--import-mode generic|canonical` - import mode, default `generic`.

`--textures=convert` supports Bistro's validated DDS set (`DXT1`/BC1,
`DXT5`/BC3, `ATI2`/BC5) plus common raster image formats. The importer emits a
deterministic output directory containing the scene, textures, and
`import_manifest.json` with schema/version metadata.

If Assimp is not found, the executable still builds in stub mode and exits with
an explicit diagnostic.

## Scene Format

Scene files live under `assets` and can also reference external asset files.
The renderer accepts scene identifiers such as `sponza`, `bistro_full`,
`bitterli_staircase`, `living_room`, `san_miguel_balcony_sun_restir`, or a path
to a `.scene`, `.gltf`, or `.glb` file.

Example:

```text
camera target=0,0,0 distance=10 yaw=0 pitch=0 vfov=40 defocusAngle=0 focusDist=10
renderer samplesPerFrame=1 maxDepth=12 width=1280 height=720 envRotation=30 envIntensity=1

background env=assets/HDR/studio.hdr

material type=metal albedo=0.9,0.9,0.9 fuzz=0.05
material type=diffuse_light albedo=5,5,5
mesh path=models/dragon.obj material=0 translate=0,0,0 scale=1,1,1 rotate=0,180,0
sphere center=0,-1001,0 radius=1000 material=1
```

Supported primitive declarations include `sphere`, `rectangle`, `box`, and
`mesh`. Meshes may be OBJ, PLY, glTF, or GLB. glTF materials are imported
automatically.

Renderer overrides can be specified in the scene file and/or on the CLI. CLI
overrides are applied after parsing.

## Validation

Build the public release targets:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Run the public smoke render:

```bash
build/PathTracerHeadless \
  --scene=sponza \
  --width=320 \
  --height=180 \
  --spp=1 \
  --seed=42 \
  --out=renders/public_smoke.png \
  --format=png \
  --verbose
```

If a release package includes CTest entries, run:

```bash
ctest --test-dir build --verbose
```

On runners without Metal support, Metal-dependent CTest entries are reported as
skipped.

Optional Embree smoke validation when Embree is installed:

```bash
cmake -S . -B build-embree -DCMAKE_BUILD_TYPE=Release -DPATH_TRACER_ENABLE_EMBREE=ON
cmake --build build-embree --target PathTracerHeadless EmbreeSmokeTest
ctest --test-dir build-embree -R embree --verbose
```

If a public smoke-test helper is included in the release package, it can be run
directly:

```bash
tests/public/headless_smoke_test.sh
```

## Release

[![Latest Version](https://img.shields.io/github/v/tag/dariopagliaricci/Metal-PathTracer-arm64?sort=semver)](https://github.com/dariopagliaricci/Metal-PathTracer-arm64/tags)

Release notes, source archives, and asset-pack links are published through
[GitHub Releases](https://github.com/dariopagliaricci/Metal-PathTracer-arm64/releases).

## Assets Pack

The public release contains many scene wrappers and canonical assets. Large
scenes, high-resolution meshes, HDRIs, and externally licensed datasets may be
distributed separately to keep the Git repository manageable.

- Download `Metal-PathTracer-Assets-v3-0-0.zip` from the
  [public asset-pack link](https://drive.google.com/file/d/1Xsb5R_wzJUWYKn3FKulJlHEt4Whyxs1A/view?usp=share_link).
- Unzip this file.
- Copy or replace the `assets` folder into the root of `Metal-PathTracer-arm64` before
  rendering asset-pack-dependent scenes.
- If Git reports mesh assets such as `assets/ajax.obj` as modified after
  replacement, normalize downloaded Wavefront text files back to the repository
  LF line-ending policy:

  ```bash
  find assets -type f \( -name '*.obj' -o -name '*.mtl' \) -exec perl -pi -e 's/\r\n/\n/g' {} +
  git status --short
  ```

Important scene IDs include:

- `bistro_full`, `bistro_full_sun_restir_alt`, `bistro_night_openfront_restir`
- `sponza`, `sponza_sunlight_hdri_restir`
- `bitterli_staircase`
- `living_room`, `living_room_regir_stress`,
  `living_room_regir_practicals_stress`
- `san_miguel_2_0`, `san_miguel_balcony_sun_restir`,
  `san_miguel_2_restir_sunlit_alternate`
- `luxcoreballs`
- `gltf-boombox`, `gltf-lantern`, `gltf-flight-helmet`,
  `gltf-damaged-helmet`, `gltf-emissive-strength`
- automotive/F1 scenes for Ferrari, Red Bull, McLaren, Mercedes, Alfa Romeo,
  Alpine, Williams, Haas, Toro Rosso, Racing Point, and Bugatti scenes.

## Operational Notes

- macOS on Apple Silicon is the intended supported platform for this renderer.
- OIDN/TBB native libraries are expected under `external/oidn/lib` for release
  builds with CPU denoising. The supported runtime is the bundled OIDN 2.4.1
  package; newer OIDN drops are upgrade candidates until validated. If a
  source-only checkout omits those native libraries, CMake disables OIDN and
  core rendering remains usable.
- Embree is a selectable CPU reference renderer for visual-output parity,
  asset validation, and backend comparison.
- Advanced transport systems are explicit opt-in research/production
  experiments; use the commands in the gated research mode examples above.
- Wavefront execution is selectable through `--executionMode=wavefront`, with
  scheduling controlled by `--wavefront-policy`.
- GUI EXR export intentionally writes timestamped captures to
  `renders/render-YYYYMMDD-HHMMSS.exr`. Headless runs use `--output` / `--out`
  when an exact path is required.
- The importer handles static scenes only; animation, skinning, morph targets,
  and full DCC material graphs are outside the current importer scope.

## License

This project is licensed under the MIT License. See `LICENSE`.

## References

### Physically Based Rendering

- [*Physically Based Rendering: From Theory to Implementation*](https://www.pbr-book.org/)
  - Matt Pharr, Wenzel Jakob, and Greg Humphreys.
- [*Robust Monte Carlo Methods for Light Transport Simulation*](https://graphics.stanford.edu/papers/veach_thesis/)
  - Eric Veach's thesis; MIS and bidirectional light transport foundation.
- [*Optimally Combining Sampling Techniques for Monte Carlo Rendering*](https://graphics.stanford.edu/courses/cs348b-03/papers/veach-chapter9.pdf)
  - Veach and Guibas, SIGGRAPH 1995.

### Transport, Caustics, and Reuse

- [*Microfacet Models for Refraction through Rough Surfaces*](https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf)
  - Walter et al., EGSR 2007.
- [*Manifold Next Event Estimation*](https://jo.dreggn.org/home/2015_mnee.pdf)
  - Hanika et al., EGSR 2015.
- [*Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing with Dynamic Direct Lighting*](https://research.nvidia.com/publication/2020-07_spatiotemporal-reservoir-resampling-real-time-ray-tracing-dynamic-direct)
  - Bitterli et al., ReSTIR DI.
- [*ReSTIR GI: Path Resampling for Real-Time Path Tracing*](https://research.nvidia.com/publication/2021-07_restir-gi-path-resampling-real-time-path-tracing)
  - Ouyang et al., ReSTIR GI.

### Reconstruction and Color

- [Intel Open Image Denoise](https://www.openimagedenoise.org/)
  - CPU denoising backend.
- [ACES Filmic Tone Mapping Curve](https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/)
  - Krzysztof Narkowicz ACES approximation.
- [Filmic Tonemapping Operators](https://filmicworlds.com/blog/filmic-tonemapping-operators/)
  - John Hable's filmic tonemapping notes.
- [Photographic Tone Reproduction for Digital Images](https://www.cs.utah.edu/docs/techreports/2002/pdf/UUCS-02-001.pdf)
  - Reinhard et al.

### Standards and Platform APIs

- [glTF 2.0 Specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
  - Khronos Group scene and material format.
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
  - Apple.
- [Metal Ray Tracing](https://developer.apple.com/metal/)
  - Apple Metal platform documentation.
- [MikkTSpace](http://www.mikktspace.com/)
  - Tangent-space standard used for normal mapped assets.

### Dependencies and Libraries

- [Intel Embree](https://www.embree.org/)
  - Optional CPU ray tracing backend.
- [Dear ImGui](https://github.com/ocornut/imgui)
  - Immediate-mode UI.
- [ImGuizmo](https://github.com/CedricGuillemet/ImGuizmo)
  - 3D transform gizmos.
- [TinyBVH](https://github.com/jbikker/tinybvh)
  - BVH construction and traversal reference.
- [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader)
  - OBJ loading.
- [tinyply](https://github.com/ddiakopoulos/tinyply)
  - PLY loading.
- [stb](https://github.com/nothings/stb)
  - Image loading utilities.
