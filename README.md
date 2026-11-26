# Metal PathTracer

A physically based renderer for macOS on Apple Silicon, written in C++20,
Objective-C++ and Metal.

Metal PathTracer is a research renderer. Its primary backend is a Metal GPU path
tracer with hardware ray tracing. An Embree CPU backend is kept as a reference
for parity checks. The renderer has deterministic headless output and a
static-scene import pipeline. Its research lighting systems are off by default;
each one is enabled through its own CLI flag.

The current release is `v4.1.0`. Development happens in a private line, and each
release is imported into this repository as a single snapshot commit. What
changed since `v3.0.0` is in [CHANGELOG.md](CHANGELOG.md),
[release_notes/v4.0.0.md](release_notes/v4.0.0.md) and
[release_notes/v4.1.0.md](release_notes/v4.1.0.md).

## Current Status

The defaults are conservative. The core path tracer runs by default, and every
research system is opt-in through CLI flags or the GUI's debug settings. Each
system below carries one of four labels: Absent, Prototype,
Implemented/runnable, or Production-complete.

| System | Label | Notes |
| --- | --- | --- |
| Metal megakernel path tracer, hardware ray tracing with a software BVH fallback | Production-complete | The default renderer. Headless output is deterministic. A fallback from hardware to software ray tracing is logged, and `--force-hwrt` / `--force-swrt` make a run fail instead. |
| Metal wavefront execution (`--executionMode=wavefront`) | Implemented/runnable | Matches the megakernel on the scenes measured. Runs `ris_spatial` as plain RIS and logs it. |
| Embree CPU reference backend | Implemented/runnable | For parity checks against Metal. Runs Metal's random-walk subsurface and the Bitterli scenes' source closures. Since v4.1.0 it also runs Metal's homogeneous media, hero-wavelength spectral mode and plain RIS; the RIS reuse modes render plain RIS and log a warning. Refuses by name the features it does not implement. |
| glTF 2.0 PBR, KHR transmission (thin-walled, without volume), KTX2 textures | Implemented/runnable | |
| `PathTracerImport` (Assimp, static scenes) | Implemented/runnable | One texture V-origin rule for every source format. |
| OIDN denoise | Production-complete | When the vendored native libraries are present; CMake disables it otherwise. |
| SVGF denoise | Prototype | Gated. |
| Direct light: `legacy`, `baseline_emissive` | Production-complete | |
| Direct light: `ris` | Implemented/runnable | |
| `ris_spatial` | Implemented/runnable | Unbiased on the megakernel (2.33004 against the exact 2.33005 on an analytic sun-plane test). Plain RIS on the wavefront, logged. |
| `ris_world` | Implemented/runnable | Unbiased on both executors, with and without hardware ray tracing (2.33004 on the sun-plane test). |
| ReGIR (`ris_regir`) and the DI+ReGIR hybrid (`restir_di_regir_hybrid`) | Prototype | Both give 2.33004 on the sun-plane test. |
| ReSTIR DI, fused path (`restir_di`) | Implemented/runnable | Exact on the sun-plane test. |
| ReSTIR DI explicit passes on the megakernel | Prototype | Refused: a headless run exits with code 2 and says why. The wavefront's explicit stage is exact on the sun-plane test. |
| ReSTIR GI, ReSTIR PT | Prototype | |
| Path guiding | Prototype | |
| Radiance cache | Prototype | Measured bias at most 0.03% of the frame's mean luminance on three scenes. No error reduction at equal render time. |
| MNEE | Implemented/runnable | |
| Path-space caustics | Prototype | `--causticTransportMode=path_space` records environment and rect-light NEE connections and returns plain NEE's contribution. It traces no light subpaths and adds no caustic paths; the run logs a `[Caustics]` line. |
| Homogeneous volumes | Prototype | Metal, and Embree since v4.1.0. Under a medium, `restir_di` and the hybrid render as `baseline_emissive`, and the run logs it. Henyey-Greenstein anisotropy follows pbrt's sign; a forward/backward glow ratio 6-8% above Arnold's is unexplained. |
| Hero-wavelength spectral transport | Implemented/runnable | Refuses specular NEE, MNEE and volumes; spectral runs need `--enableSpecularNee=0`. See [Hero-Wavelength Spectral Mode](#hero-wavelength-spectral-mode). |
| Random-walk subsurface scattering (Metal) | Implemented/runnable | On a unit sphere under one emitter: within 0.06% of Arnold's randomwalk at mean free path 0.1 and 0.31% at 0.25. Objects thinner than about two mean free paths differ more (up to 5%), where Arnold departs from a volumetric walk by design. |
| Tungsten source closures (Bitterli scenes) | Implemented/runnable | Opt-in per scene. |
| ML reconstruction backend (`--mlBackendMode`) | Prototype | Off by default. No quality claim. |

## Render Modes

Headless rendering separates user-facing render modes from backend, render
profile, and Metal execution controls.

- `metal_final_frame` selects the controlled Metal final-frame/headless contract.
  It preserves expert controls such as `--renderProfile`, `--executionMode`,
  `--wavefront-policy`, and feature flags. It does not claim original exact
  shade active-count indirect closure; production wavefront shade remains
  explicitly reported as direct-grid/platform-limited where applicable.
- `cpu_reference_headless` selects the Embree CPU reference/offline contract.
  It is not GUI viewport IPR and does not prove CPU tiled final rendering.

Use `--renderMode=<metal_final_frame|cpu_reference_headless>` to select the
render intention directly. If omitted, the mode is inferred from `--backend`.
Contradictory combinations, such as `--renderMode=cpu_reference_headless
--backend=metal`, are rejected. Generated settings, queue item, checkpoint,
metadata, PBR metrics, and debug reports record render mode, mode maturity,
mode context, backend, profile, and execution mode where applicable.

## Repository Map

- `src/renderer`, `include/renderer`: Metal renderer, scene resources, pass
  graph, transport memory registry, accumulation, denoise, UI, and acceleration
  setup.
- `src/headless`, `include/headless`: Metal and Embree headless backends.
- `src/import`, `include/import`: Assimp import pipeline, Bistro audit, texture
  conversion, KTX2 writing and loading.
- `shaders`: Metal shader tree split by core math, tracing, BSDF, lighting,
  ReSTIR, wavefront, reconstruction, guiding, cache, volume, spectral, caustics,
  scene access, and debug views. `shaders/shared` holds closure code that the
  Metal shaders and the C++ Embree backend both compile.
- `assets`: the bundled sample assets (two busts, two HDRIs, five scenes) and
  `CREDITS.md`. The full scene set is the separate [asset pack](#assets-pack).
- `tests`: the public smoke test and test scenes.
- `validation/baselines`: threshold files for the ReSTIR and RIS checks.
- `Images/Gallery`: the README gallery (JPEG, 1920x1080).
- `paper`: the JOSS paper sources.

## Renderings Gallery

Every image below is a v4.0.0 release render at 1920x1080, made with the
headless renderer and denoised with OIDN (albedo and normal guides). Metal
images use 1024 samples per pixel; the two Embree images use 256. Each caption
names the scene ID, so the image can be reproduced from the asset pack (see
[Assets Pack](#assets-pack)). Asset credits are under [Asset Credits](#asset-credits).

### Large Scenes

| | |
| --- | --- |
| ![Amazon Lumberyard Bistro in sunlight](Images/Gallery/bistro_full_sun_restir_alt_mk_1920x1080_1024spp.jpg)<br>Bistro, day: `bistro_full_sun_restir_alt`, megakernel, `restir_di`, exposure +1 stop. | ![Amazon Lumberyard Bistro at night](Images/Gallery/bistro_night_openfront_restir_mk_1920x1080_1024spp.jpg)<br>Bistro, night: `bistro_night_openfront_restir`, megakernel, `restir_di`. |
| ![San Miguel balcony in sunlight](Images/Gallery/san_miguel_balcony_sun_restir_mk_1920x1080_1024spp.jpg)<br>San Miguel, balcony: `san_miguel_balcony_sun_restir`, megakernel. | ![San Miguel courtyard in sunlight](Images/Gallery/san_miguel_2_restir_sunlit_alternate_mk_1920x1080_1024spp.jpg)<br>San Miguel, courtyard: `san_miguel_2_restir_sunlit_alternate`, megakernel. |
| ![Sponza in sunlight](Images/Gallery/sponza_sunlight_hdri_restir_mk_1920x1080_1024spp.jpg)<br>Sponza: `sponza_sunlight_hdri_restir`, megakernel, `restir_di`. | ![Unreal Engine Sun Temple](Images/Gallery/sun_temple_mk_1920x1080_1024spp.jpg)<br>Sun Temple: `sun_temple`, megakernel. Asset CC BY-NC-SA 4.0 (Epic Games, via NVIDIA ORCA). |
| ![NVIDIA Emerald Square by day](Images/Gallery/emerald_square_day_wf_1920x1080_1024spp.jpg)<br>Emerald Square, day: `emerald_square_day`, wavefront. Asset CC BY-NC-SA 3.0 (NVIDIA ORCA). | ![NVIDIA Emerald Square at dusk](Images/Gallery/emerald_square_dusk_wf_1920x1080_1024spp.jpg)<br>Emerald Square, dusk: `emerald_square_dusk`, wavefront. Asset CC BY-NC-SA 3.0 (NVIDIA ORCA). |
| ![Barcelona Pavilion by day](Images/Gallery/barcelona_pavilion_day_mk_1920x1080_1024spp.jpg)<br>Barcelona Pavilion, day: `barcelona_pavilion_day`, megakernel. | ![Barcelona Pavilion at night](Images/Gallery/barcelona_pavilion_night_mk_1920x1080_1024spp.jpg)<br>Barcelona Pavilion, night: `barcelona_pavilion_night`, megakernel. |
| ![Zero-Day, Measure Seven](Images/Gallery/zero_day_measure_seven_mk_1920x1080_1024spp.jpg)<br>Zero-Day, Measure Seven: `zero_day_measure_seven`, megakernel. | |

### Bitterli Scenes

| | |
| --- | --- |
| ![Wooden staircase](Images/Gallery/bitterli_staircase_mk_1920x1080_1024spp.jpg)<br>Staircase: `bitterli_staircase`, megakernel. | ![Glass of water](Images/Gallery/bitterli_glass_of_water_wf_1920x1080_1024spp.jpg)<br>Glass of Water: `bitterli_glass_of_water`, wavefront. |
| ![Spaceship](Images/Gallery/bitterli_spaceship_wf_1920x1080_1024spp.jpg)<br>Spaceship: `bitterli_spaceship`, wavefront. | ![Veach, Ajar](Images/Gallery/bitterli_veach_ajar_wf_1920x1080_1024spp.jpg)<br>Veach, Ajar: `bitterli_veach_ajar`, wavefront. |

### Materials, Subsurface and Spectral Transport

| | |
| --- | --- |
| ![Damaged Helmet](Images/Gallery/gltf-damaged-helmet_mk_1920x1080_1024spp.jpg)<br>Damaged Helmet: `gltf-damaged-helmet`, megakernel. | ![Bugatti Chiron Top Edition](Images/Gallery/bugatti_chiron_top_edition_matchref_mk_1920x1080_1024spp.jpg)<br>Bugatti Chiron: `bugatti_chiron_top_edition_matchref`, megakernel. |
| ![Marble and wax busts with random-walk subsurface scattering](Images/Gallery/sss_marble_wax_mk_1920x1080_1024spp.jpg)<br>Random-walk subsurface, marble and wax: `sss_marble_wax`, megakernel. | ![Transparent Machines film camera](Images/Gallery/transparent_machines_film_mk_1920x1080_1024spp.jpg)<br>Transparent Machines, film camera: `transparent_machines_film`, megakernel. |
| ![Suzanne caustics in hero-wavelength spectral mode](Images/Gallery/suzanne_caustics_hero_spectral_mk_1920x1080_1024spp.jpg)<br>Hero-wavelength spectral transport: `suzanne_caustics_hero`, megakernel, three wavelengths per path, specular NEE and MNEE off (spectral mode refuses them). | |

### Formula 1 Cars

| | |
| --- | --- |
| ![2021 McLaren MCL35M](Images/Gallery/f1_bulk_2021_f1_mclaren_mcl35m_mk_1920x1080_1024spp.jpg)<br>2021 McLaren MCL35M: `f1_bulk_2021_f1_mclaren_mcl35m`. | ![2021 Mercedes-Benz W12](Images/Gallery/f1_bulk_2021_f1_mercedes_benz_w12_mk_1920x1080_1024spp.jpg)<br>2021 Mercedes-Benz W12: `f1_bulk_2021_f1_mercedes_benz_w12`. |
| ![2021 Alpine A521](Images/Gallery/f1_bulk_2021_f1_alpine_a521_mk_1920x1080_1024spp.jpg)<br>2021 Alpine A521: `f1_bulk_2021_f1_alpine_a521`. | ![2020 Mercedes-Benz W11](Images/Gallery/f1_bulk_2020_f1_mercedes_benz_w11_mk_1920x1080_1024spp.jpg)<br>2020 Mercedes-Benz W11: `f1_bulk_2020_f1_mercedes_benz_w11`. |
| ![2020 McLaren MCL35](Images/Gallery/f1_bulk_2020_f1_mclaren_mcl35_mk_1920x1080_1024spp.jpg)<br>2020 McLaren MCL35: `f1_bulk_2020_f1_mclaren_mcl35`. | ![2020 Racing Point RP20](Images/Gallery/f1_bulk_2020_f1_racing_point_rp20_mk_1920x1080_1024spp.jpg)<br>2020 Racing Point RP20: `f1_bulk_2020_f1_racing_point_rp20`. |
| ![2020 Alfa Romeo C39](Images/Gallery/f1_bulk_2020_f1_alfa_romeo_c39_mk_1920x1080_1024spp.jpg)<br>2020 Alfa Romeo C39: `f1_bulk_2020_f1_alfa_romeo_c39`. | ![2020 Haas VF-20](Images/Gallery/f1_bulk_2020_f1_haas_vf20_mk_1920x1080_1024spp.jpg)<br>2020 Haas VF-20: `f1_bulk_2020_f1_haas_vf20`. |
| ![2020 Williams FW43](Images/Gallery/f1_bulk_2020_f1_whilliams_fw43_mk_1920x1080_1024spp.jpg)<br>2020 Williams FW43: `f1_bulk_2020_f1_whilliams_fw43`. | ![2019 Mercedes-Benz W10](Images/Gallery/f1_bulk_2019_f1_mercedes_benz_w10_mk_1920x1080_1024spp.jpg)<br>2019 Mercedes-Benz W10: `f1_bulk_2019_f1_mercedes_benz_w10`. |
| ![2019 McLaren MCL34](Images/Gallery/f1_bulk_2019_mclaren_mcl34_mk_1920x1080_1024spp.jpg)<br>2019 McLaren MCL34: `f1_bulk_2019_mclaren_mcl34`. | ![2019 Red Bull RB15](Images/Gallery/f1_bulk_2019_f1_red_bull_rb15_mk_1920x1080_1024spp.jpg)<br>2019 Red Bull RB15: `f1_bulk_2019_f1_red_bull_rb15`. |
| ![2019 Alfa Romeo C38](Images/Gallery/f1_bulk_2019_f1_alfa_romeo_c38_mk_1920x1080_1024spp.jpg)<br>2019 Alfa Romeo C38: `f1_bulk_2019_f1_alfa_romeo_c38`. | ![2019 Haas VF-19](Images/Gallery/f1_bulk_2019_haas_vf_19_mk_1920x1080_1024spp.jpg)<br>2019 Haas VF-19: `f1_bulk_2019_haas_vf_19`. |

All fourteen cars render on the megakernel. The car models are by OUTPISTON on Sketchfab, CC BY-NC-SA 4.0; these images are shared under the same licence.

### Embree CPU Reference

The Embree backend renders the same scene files on the CPU and is used to check
Metal's output. These two images are Embree renders at 256 samples per pixel.

| | |
| --- | --- |
| ![Sponza on Embree](Images/Gallery/sponza_sunlight_hdri_restir_embree_1920x1080_256spp.jpg)<br>Sponza: `sponza_sunlight_hdri_restir`, Embree, `restir_di`. | ![Staircase on Embree](Images/Gallery/bitterli_staircase_embree_1920x1080_256spp.jpg)<br>Staircase: `bitterli_staircase`, Embree. |

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
- Optional firefly clamp modes.
- Path-space caustic connection transport (Prototype: it adds no caustic paths
  yet).
- Gated homogeneous/global volume transport (Prototype) with isotropic or
  Henyey-Greenstein phase, rect-light volume NEE, debug metadata, and explicit
  fallbacks for non-medium-aware providers. Heterogeneous volumes remain future
  work.
- Gated hero-wavelength spectral transport: three wavelengths per path, a
  collapse to the hero wavelength at dispersive vertices, a host-computed white
  balance, explicit sampled-triplet material policy, RGB approximation fallback
  metadata, and dielectric dispersion. See
  [Hero-Wavelength Spectral Mode](#hero-wavelength-spectral-mode).

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
- ReSTIR GI diffuse reuse (Prototype).
- ReSTIR PT reservoir capture and path reuse (Prototype).
- ReGIR/world-space candidate reuse.
- Gated path guiding (Prototype).
- A prototype non-neural radiance cache that learns the radiance paths gather
  after a cached vertex and returns it in place of the rest of the path. It is
  biased by design; see the radiance cache notes under the headless examples.
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
  `external/oidn` for CPU denoising. Release builds are tested with the
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

Large-scene interactive launch:

```bash
PT_HWRT_BUILD_POLICY=compact_memory \
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
| `PT_ENABLE_INTERACTIVE_EXPLICIT_RESTIR_DI` | `1` / `0` | Requests the explicit ReSTIR DI pass graph in the interactive GUI. Plain `restir_di` and `restir_di_regir_hybrid` use the explicit path when this is set; otherwise the GUI keeps the fused/default path. This is orthogonal to `PT_HWRT_BUILD_POLICY`: one selects the ReSTIR DI pass graph, the other selects acceleration-structure build policy. Use this for profiling, debugging, or validating the explicit pass graph rather than as the default large-scene launch. |
| `PT_SCENE_LOAD_DIAGNOSTICS` | `1` / `0` | Prints scene-loading, asset-resolution, and staged-load diagnostics. |
| `PT_SCENE_LOAD_DIAGNOSTICS_VERBOSE` | `1` / `0` | Adds verbose scene-load diagnostics; requires `PT_SCENE_LOAD_DIAGNOSTICS=1`. |
| `PT_INTERACTIVE_STATS` | `1` / `0` | Enables interactive GUI terminal telemetry and writes a per-scene JSON session summary on shutdown. |
| `PT_INTERACTIVE_STATS_INTERVAL` | Floating-point seconds | Live terminal telemetry interval. Defaults to `1`. |
| `PT_INTERACTIVE_STATS_SUMMARY_INTERVAL` | Floating-point seconds | Rolling summary interval. Defaults to `60`; set `0` to disable summary lines. |
| `PT_INTERACTIVE_STATS_JSON` | File path | Optional explicit JSON output path. If unset, the GUI writes `benchmarks/interactive_stats_session_YYYYMMDD_HHMMSS.json`. |
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

### Interactive Benchmark Telemetry

For GUI benchmark runs, prefer terminal telemetry over screenshot-only overlay
readings. The interactive logger separates app/UI pacing from path-trace GPU
work and writes a per-scene JSON report when the GUI closes.

Recommended large-scene benchmark launch:

```bash
PT_INTERACTIVE_STATS=1 \
PT_INTERACTIVE_STATS_INTERVAL=1 \
PT_INTERACTIVE_STATS_SUMMARY_INTERVAL=60 \
PT_HWRT_BUILD_POLICY=compact_memory \
./build-metal/PathTracer.app/Contents/MacOS/PathTracer --scene=bistro_full
```

Explicit ReSTIR DI GUI profiling launch:

```bash
PT_HWRT_BUILD_POLICY=compact_memory \
PT_ENABLE_INTERACTIVE_EXPLICIT_RESTIR_DI=1 \
./build-metal/PathTracer.app/Contents/MacOS/PathTracer --scene=assets/bistro_full_sun_restir_alt.scene
```

The GUI accepts `--scene`, but direct-light mode is selected in the UI. Use the
explicit launch above only when profiling or validating the explicit `restir_di`
or `restir_di_regir_hybrid` graph; omit `PT_ENABLE_INTERACTIVE_EXPLICIT_RESTIR_DI`
for the normal fused/default interactive path. When `restir_di_regir_hybrid` is
selected in explicit mode, the ReSTIR overlay and PBR metrics report
`restir_di_regir_candidate_count`, `restir_di_regir_accepted_count`, and
`restir_di_regir_fallback_count` so selected hybrid mode can be distinguished
from actual hybrid ReGIR candidate activity.

To choose the JSON report path explicitly:

```bash
PT_INTERACTIVE_STATS=1 \
PT_INTERACTIVE_STATS_JSON=benchmarks/bistro_session.json \
PT_HWRT_BUILD_POLICY=compact_memory \
./build-metal/PathTracer.app/Contents/MacOS/PathTracer --scene=bistro_full
```

If `PT_INTERACTIVE_STATS_JSON` is unset, the default output is timestamped:

```text
benchmarks/interactive_stats_session_YYYYMMDD_HHMMSS.json
```

Live terminal lines use these key fields:

```text
[InteractiveStats] scene="bistro_full" resolution=1280x720 scale=0.50 backend=Metal-HWRT execution=Wavefront wavefrontPolicy=Research wavefrontCompaction=on restir="di=on,explicitDI=on,gi=off,pt=off" appFrameMs=1940.367 appFps=0.5 spp=9 samplesPerFrame=1 appSppPerMin=4151.1 appSppPerSec=69.2 pathSppPerSec=0.5 triangles=3846586 pathGpuMs=1915.611 cpuEncodeMs=0.039 drawableWaitMs=3.295 raysThisFrame=123456789 raysPerSec=64446954 denoiser=off
```

`appFrameMs` / `appFps` describe the interactive application update cadence.
`pathGpuMs`, `pathSppPerSec`, and `raysPerSec` describe the completed
path-tracing GPU work for the sampled frame. Missing counters are printed as
`unavailable` rather than estimated. The shutdown JSON report uses schema
`metal-pathtracer.interactiveStats.v2`; each scene contains one or more
`segments`, where a segment is a stable renderer configuration
(resolution, render scale, backend, execution mode, Wavefront policy, ReSTIR
state, denoiser state, and samples per frame). This keeps scene-load stalls,
mode switches, and denoiser toggles from being averaged into a single
misleading number.

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

Metal SVGF preview denoise:

```bash
build/PathTracerHeadless \
  --backend=metal \
  --scene=assets/ajax.scene \
  --width=1280 \
  --height=720 \
  --sppTotal=64 \
  --seed=42 \
  --svgfDenoise=1 \
  --svgf-atrous-iterations=2 \
  --output=renders/ajax_svgf_preview.exr \
  --format=exr \
  --outputMetadata=1 \
  --debugBundleDir=renders/ajax_svgf_preview_bundle
```

Smoother SVGF pass after the camera has settled:

```bash
build/PathTracerHeadless \
  --backend=metal \
  --scene=assets/ajax.scene \
  --width=1280 \
  --height=720 \
  --sppTotal=128 \
  --seed=42 \
  --svgfDenoise=1 \
  --svgf-atrous-iterations=3 \
  --output=renders/ajax_svgf_smooth.exr \
  --format=exr \
  --outputMetadata=1 \
  --debugBundleDir=renders/ajax_svgf_smooth_bundle
```

SVGF direct/indirect component controls:

```bash
# Denoise direct lighting only; leave indirect raw.
build/PathTracerHeadless \
  --backend=metal \
  --scene=assets/ajax.scene \
  --width=1280 \
  --height=720 \
  --sppTotal=64 \
  --seed=42 \
  --svgfDenoise=1 \
  --svgf-denoise-direct=1 \
  --svgf-denoise-indirect=0 \
  --svgf-atrous-iterations=2 \
  --output=renders/ajax_svgf_direct_only.exr \
  --format=exr

# Denoise indirect lighting only; leave direct raw.
build/PathTracerHeadless \
  --backend=metal \
  --scene=assets/ajax.scene \
  --width=1280 \
  --height=720 \
  --sppTotal=64 \
  --seed=42 \
  --svgfDenoise=1 \
  --svgf-denoise-direct=0 \
  --svgf-denoise-indirect=1 \
  --svgf-atrous-iterations=2 \
  --output=renders/ajax_svgf_indirect_only.exr \
  --format=exr
```

SVGF is a Metal reconstruction path. `--svgf-atrous-iterations=2` is the
recommended interactive preview setting; `3` is smoother but softer. Use OIDN
with `--denoise=1` for CPU postprocess denoising.

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
  `--spectralMode`, `--spectralWavelengthsPerPath`, plus their related tuning
  flags.
- Large scenes: `--stats`, `--budget-policy=strict|warn|ignore`,
  `--pilot-mode=geo_only|albedo_only|full_clamped`, `--texture-max-dim`,
  `--max-texture-bytes`.
- Output quality: `--tonemap`, `--exposure`, `--firefly-clamp-mode`,
  `--firefly-clamp-value`, `--denoise`, OIDN guide flags, `--svgfDenoise`,
  `--svgf-denoise-direct`, `--svgf-denoise-indirect`,
  `--svgf-atrous-iterations`, `--svgf-normal-sigma`,
  `--svgf-albedo-sigma`, `--svgf-luminance-sigma`.
- Production metadata: `--pbrMetrics`, `--pbrMetricsJson`,
  `--outputMetadata`, `--outputMetadataJson`, `--settingsJson`,
  `--renderQueueItemJson`, `--runRenderQueueItemJson`, `--debugBundleDir`,
  `--checkpointManifestJson`, `--resumeCheckpointJson`, `--tileManifestJson`,
  `--tileSize`.
  `--runRenderQueueItemJson` accepts a single queue item, a root JSON array, or
  a `{ "queue_items": [...] }` batch and writes a per-item batch summary.
  Checkpoint manifests are completed-frame rerender manifests; tile manifests
  are deterministic full-frame tile plans. Debug bundles record the completed
  render artifact set and the explicit failure-context policy for early setup
  aborts.
- Debug/audit: `--materialMrDebugJson`, `--directLightAuditJson`,
  `--restirDebug`, `--restirDebugView`, `--restirDebugCounters`,
  `--restirDebugMetricsJson`, `--cameraSearchReport`, `--debugPath`,
  `--debugPixel`, `--parityAssert`, material texture debug flags, and AOV
  visualization flags.

Run `build/PathTracerHeadless --help` for the complete current CLI surface.

### Gated Research Mode Examples

The advanced transport systems are explicit opt-in paths. Each run records the
selected modes in its metadata and debug output. The maturity of each system
is in the table under [Current Status](#current-status).

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

ReSTIR GI diffuse reuse (Prototype; the mode value is `restir_gi_production`):

```bash
build/PathTracerHeadless \
  --scene=living_room \
  --renderProfile=lookdev \
  --restirGiMode=restir_gi_production \
  --directLightMode=restir_di \
  --out=renders/living_room_restir_gi.exr
```

Gated path guiding and radiance cache providers:

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

Radiance cache notes: the cache is a prototype and is biased by design. For
each world-space cell it stores the mean radiance that training paths (one
pixel in eight per frame) gather after a vertex in that cell, divided by the
path throughput at that vertex. A path that reaches a confident cell adds the
cached value and stops. Measured at 320x180 with 4096 samples per pixel against
cache-off references (seed 42, default cache settings), the full-frame mean
luminance differs from the reference by +0.01% on `living_room`, 0.00% on
`sponza` and +0.03% on `emerald_square_day`, and no named test region differs
by more than 0.15%. The low bias comes with little reuse: a cache hit ends about
0.01% of query attempts on `living_room` and about 15% on `sponza` and
`emerald_square_day`, and at equal render time the cache does not reduce error
on `living_room` or `sponza`. The cache is refused, with a warning and a
metadata reason, when ReSTIR GI or a ReSTIR PT path-reuse mode is enabled, and
the headless renderer refuses to run spectral hero-wavelength mode with the
cache on.

Why the cache gives no equal-time gain. These were measured in November 2026
at 320x180 on the hardware megakernel. Costs per sample come from alternating
exclusive runs. Errors are RMSE against cache-off references of 11k to 385k
samples per pixel.

- Cost. Turning the cache on makes the megakernel use its general pipeline,
  where every research and debug gate is live.
  - Per sample, cache on costs 2.76x cache off on `living_room`, 2.23x on
    `sponza` and 1.38x on `emerald_square_day` (at 64 spp).
  - With no cache queries or training (`--radianceCacheMinDepth=64`), it still
    costs 2.47x and 2.18x.
  - A diagnostic shader build that folds every other gate out of that pipeline
    brings cache on down to 1.18x and 1.15x (1.10x and 1.09x with no queries).
  - The cache's diagnostic counters cost about 1%.
- Gate. The gate refuses a cell when one sample's relative variance exceeds
  0.85. On `living_room` it passes 0.07% of the loads that reach it:
  - 98% of those loads find a cell holding 1024 samples;
  - 80% of them find a relative variance above 16;
  - so RMSE at equal samples is that of cache off: 0.04398 against 0.04401 at
    256 spp.
- Other gates, all on `living_room`:
  - Opening the gate, with cells also keyed by the dominant axis of the normal,
    cuts RMSE at equal samples by 12%. But it moves the full-frame mean by
    -0.28% and the left sofa by -1.4%.
  - With the gate open, a training clamp of 256 instead of 16, a cell size of
    0.25 instead of 1, or both, still leave the frame 0.25% to 0.34% off.
  - Gates on the variance of the cell mean move the frame by -0.17% to +1.37%.
    They refuse when var / (n mean^2) exceeds 0.01 or 0.04, with or without
    the normal key. They pass the cells whose mean came out high or low.
  - Keying cells by the normal under the current gate moves the frame by
    -0.05% and reduces RMSE at equal samples by 0.4%.

None of these gates reduces error at equal render time while staying within
0.03% of the frame. Folding the general pipeline would make the cache cheap,
but with the current gate it would still reduce no error.

Homogeneous volumes, with the path-space caustics prototype (it adds no caustic
paths):

```bash
build/PathTracerHeadless \
  --scene=suzanne_caustics \
  --renderProfile=lookdev \
  --causticTransportMode=path_space \
  --volumeTransportMode=homogeneous \
  --volumeSigmaA=0.02,0.02,0.02 \
  --volumeSigmaS=0.04,0.04,0.04 \
  --out=renders/suzanne_research_transport.exr
```

ReSTIR DI reservoirs carry no medium state, so with
`--volumeTransportMode=homogeneous` the Metal backend does not run `restir_di`
or `restir_di_regir_hybrid`: it renders `baseline_emissive` instead. The log
says so, and the render metadata field `volume_direct_light_fallback` names the
mode that ran. On Metal, ReSTIR GI, ReSTIR PT, path guiding and the radiance
cache are also off with volumes, as the metrics JSON fallback report records.
Volume NEE at scatter points samples rect lights only. Surface NEE to a light
with a position (mesh emitters, rect lights, point lights) is attenuated by the
medium's transmittance over the shadow segment in every direct-light mode, on
the megakernels and the wavefront; suns and the environment are not attenuated,
as escaped paths are not. The `volume_transmittance_query_count` metric counts
those queries too. Since v4.1.0 Embree runs the same media, with Metal's mode
demotions.

Hero-wavelength spectral transport (the path-space caustics prototype adds no
caustic paths). Spectral mode
refuses specular NEE, MNEE and volumes (see
[Hero-Wavelength Spectral Mode](#hero-wavelength-spectral-mode)), so the scene's
`enableSpecularNee=1 enableMnee=1` are turned off here:

```bash
build/PathTracerHeadless \
  --scene=suzanne_caustics \
  --renderProfile=lookdev \
  --causticTransportMode=path_space \
  --enableSpecularNee=0 \
  --enableMnee=0 \
  --spectralMode=hero_wavelength \
  --spectralWavelengthsPerPath=3 \
  --spectralDispersionStrength=0.012 \
  --out=renders/suzanne_spectral.exr
```

Notes:

- HDR outputs (`exr`, `pfm`) are written in linear space.
- Embree uses the same `--format` output surface as Metal. EXR/PFM are linear
  HDR outputs; PNG/PPM are tonemapped visual outputs.
- `--denoise=1` writes `<stem>_raw.<ext>` and `<stem>_denoised.<ext>` in place of the requested unsuffixed filename.
- Viewport bloom (`bloomEnabled`/`bloom`, `bloomThreshold`, `bloomIntensity`,
  `bloomRadius`) affects only the interactive display. The separate scene
  setting `referenceBloom` applies reference bloom before tonemapping to saved
  PNG/PPM images; neither bloom path affects saved linear EXR/PFM output.
- Metal backend logs and large-scene reports are diagnostics, not progress
  indicators. Use `--verbose` for progress with percentage and ETA.

### Hero-Wavelength Spectral Mode

`--spectralMode=hero_wavelength` replaces RGB transport with transport over
wavelengths sampled uniformly between `--spectralMinNm` and `--spectralMaxNm`
(default 380-720 nm). It is a gated mode of the Metal backend; the Embree
backend ignores it with a warning and renders RGB.

- **Wavelengths per path.** Each path samples a hero wavelength and carries
  C wavelengths in its three throughput channels,
  `lambda_j = min + (max - min) * frac(u + j / C)`, so the channels are spread
  evenly over the range and each one is uniformly distributed.
  `--spectralWavelengthsPerPath=3` (default) carries three; `=1` carries the
  hero wavelength only and is kept as a baseline for comparisons. The queue
  JSON key is `spectral_wavelengths_per_path`.
- **Conversion.** RGB reflectance, emission and light radiance become one value
  per carried wavelength where they enter the path: materials at each hit,
  lights at emission, background/environment and NEE. RGB textures use the
  `rgb_reflectance_d65` approximation and are counted as spectral fallbacks.
  Scene `spectral*` triplets (450/550/650 nm) are evaluated at each carried
  wavelength.
- **Sensor and white balance.** When a path ends, the radiance of each
  wavelength is weighted by a fixed RGB sensor (three Gaussians centred at
  610, 545 and 455 nm, widths 48, 42 and 36 nm) and averaged over the C
  wavelengths, once per path. The host multiplies the result by a white balance
  `W_c = 1 / E[s_c]` over the sampled range, so a spectrally flat input of 1
  renders as RGB (1, 1, 1). For 380-720 nm, W = (1.9228, 2.7271, 2.5664); the
  output sidecar records it as `spectral_white_balance`.
- **Collapse at dispersive vertices.** A vertex is dispersive when its material
  is transmissive and its dispersion strength (material `spectralParams` or
  `--spectralDispersionStrength`, default 0.012) is nonzero, so by default every
  glass surface is dispersive. The sampled direction there is valid for the hero
  wavelength only: the path keeps the hero channel, scaled by C, and drops the
  others. Light gathered before that vertex keeps all wavelengths; after it the
  path is a single-wavelength path.
- **Clamps and guides.** The luminance-mode firefly and throughput clamps
  measure brightness as the mean over the wavelength channels; max-component
  mode is unchanged. Denoiser guides (albedo, normal, material features) stay
  RGB.
- **Metadata.** The output sidecar records `spectral_wavelengths_per_path`,
  `spectral_sampling_strategy`, `spectral_white_balance` and
  `spectral_last_frame_stats` (wavelength samples, channel samples, collapses
  and dispersion events of the last frame; the stats buffer is cleared every
  frame). `--pbrMetrics=1` adds the same counters to the metrics JSON.

Support matrix (headless Metal backend):

| Supported | Refused |
| --- | --- |
| `--directLightMode=legacy`, `baseline_emissive` | RIS, ReGIR, ReSTIR DI and the DI+ReGIR hybrid |
| Megakernel and wavefront execution | `--restirGiMode`, `--restirPtMode` other than `off` |
| Path-space caustics (`--causticTransportMode=path_space`) | Path guiding, radiance cache |
| OIDN denoising with RGB guides | Homogeneous volumes |
| `--spectralWavelengthsPerPath=1` or `3` | Specular NEE (on by default: pass `--enableSpecularNee=0`), MNEE |
| | SVGF, ML reconstruction |
| | Tungsten source closures and `sourceTransport` |
| | Debug views (`--debugShow*`, `--restirDebugView` other than `beauty`) |

A refused combination prints `spectral mode does not support <combination>`,
writes a sidecar with `"status": "refused"` and
`spectral_refused_combination`, and exits with status 2 without rendering.

Limits:

- The sensor is three Gaussians, not the CIE color-matching functions, and RGB
  inputs are converted linearly onto it, so saturated RGB colors come out less
  saturated: RGB (1, 0, 0) comes out as about (0.766, 0.309, 0.021).
- There is no spectral MIS: the carried wavelengths share the hero wavelength's
  sampled directions with equal weights, and after a collapse only the hero
  wavelength contributes.
- An emitter given as explicit spectral triplets is converted slightly
  differently when a path hits it than when NEE samples it, and
  environment-modulated emission converts the emission and the environment
  separately on a hit but their product in NEE.

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
`import_manifest.json` with schema/version metadata, support-matrix entries,
texture policy, stable geometry/material/texture IDs, output hashes, and
diagnostics for omitted static-import features.

Runtime glTF/GLB and `.scene` loads expose the same asset contract in
`--outputMetadataJson` and `--pbrMetricsJson` under `asset_pipeline`. The report
includes support boundaries, stable resource IDs, material override records,
texture color-space decisions, texture residency/mip/compression/budget policy,
and diagnostics for unsupported lights, animation, skinning, morph targets,
USD, and MaterialX. The current pipeline covers static scenes:
glTF/GLB geometry, transforms, cameras, textures, PBR-compatible materials,
alpha/transmission/emission mappings, KTX2/DDS/raster texture handling, and
Assimp conversion are supported; animation, skinning, morph evaluation,
KHR punctual-light conversion, USD, and MaterialX are reported rather than
silently imported.

If Assimp is not found, the executable still builds in stub mode and exits with
an explicit diagnostic.

## Scene Format

Scene files live under `assets` and can also reference external asset files.
The renderer accepts scene identifiers such as `sponza`, `bistro_full`,
`bitterli_staircase`, `living_room`, `san_miguel_balcony_sun_restir`, or a path
to a `.scene`, `.gltf`, or `.glb` file. Direct `.gltf` and `.glb` paths are
routed through the same glTF mesh loader and asset-pipeline report path used by
scene-file `mesh path=...` directives.

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

[![Latest Release](.github/badges/latest-release.svg)](https://github.com/dariopagliaricci/Metal-PathTracer-arm64/releases/latest)

Release notes, source archives, and asset-pack links are published through
[GitHub Releases](https://github.com/dariopagliaricci/Metal-PathTracer-arm64/releases).

## Assets Pack

This repository bundles a small asset set, so the renderer can be tried from a
fresh clone: the Ajax and Hygieia busts, two Poly Haven HDRIs, and five scenes
that use them (`ajax`, `ajax-other`, `ajax-glass`, `hygieia`, `hygieia-other`).
Run from the repository root:

```bash
build/PathTracerHeadless --scene=ajax --width=640 --height=360 --spp=64 \
  --seed=42 --format=png --out=renders/ajax.png
```

The other scenes in this README, including every gallery scene, need the v4
asset pack, which is distributed separately because of its size.

- Download the v4 asset pack from [this link](https://drive.google.com/file/d/1N7OLGjs4FPK57eoMsdOd8tS2iHmQTO7X/view).
- Unzip it.
- Copy or replace the `assets` folder in the root of your clone before you
  render a scene from the pack. The headless renderer resolves scenes from the
  first folder, from the current directory upward, that holds both `.git` and
  `assets/`.
- If Git then reports mesh assets such as `assets/ajax.obj` as modified,
  normalize the Wavefront text files back to the repository's LF line endings:

  ```bash
  find assets -type f \( -name '*.obj' -o -name '*.mtl' \) -exec perl -pi -e 's/\r\n/\n/g' {} +
  git status --short
  ```

Scene IDs used in the gallery:

- `bistro_full_sun_restir_alt`, `bistro_night_openfront_restir`
- `san_miguel_balcony_sun_restir`, `san_miguel_2_restir_sunlit_alternate`
- `sponza_sunlight_hdri_restir`, `sun_temple`
- `emerald_square_day`, `emerald_square_dusk`
- `barcelona_pavilion_day`, `barcelona_pavilion_night`, `zero_day_measure_seven`
- `bitterli_staircase`, `bitterli_glass_of_water`, `bitterli_spaceship`,
  `bitterli_veach_ajar`
- `gltf-damaged-helmet`, `bugatti_chiron_top_edition_matchref`,
  `sss_marble_wax`, `transparent_machines_film`, `suzanne_caustics_hero`
- the fourteen `f1_bulk_*` scenes listed in the gallery

## Operational Notes

- macOS on Apple Silicon is the intended supported platform for this renderer.
- OIDN/TBB native libraries are expected under `external/oidn/lib` for release
  builds with CPU denoising. The supported runtime is the bundled OIDN 2.4.1
  package; newer OIDN drops are upgrade candidates until validated. If a
  source-only checkout omits those native libraries, CMake disables OIDN and
  core rendering remains usable.
- Embree is a selectable CPU reference renderer for visual-output parity,
  asset validation, and backend comparison.
- Advanced transport systems are explicit opt-in paths; use the commands in
  the gated mode examples above. Their maturity is in the Current Status table.
- Wavefront execution is selectable through `--executionMode=wavefront`, with
  scheduling controlled by `--wavefront-policy`.
- GUI EXR export intentionally writes timestamped captures to
  `renders/render-YYYYMMDD-HHMMSS.exr`. Headless runs use `--output` / `--out`
  when an exact path is required.
- The importer handles static scenes only; animation, skinning, morph targets,
  and full DCC material graphs are outside the current importer scope.

## Asset Credits

The renderer's code is MIT licensed. The scenes shown in this README are third-party
assets, each under its own terms; they are converted, and many are re-materialed,
for this renderer. `assets/CREDITS.md` has the full list.

Metal PathTracer is a non-commercial project. Some of the assets below are
licensed for non-commercial use only (CC BY-NC-SA 4.0, CC BY-NC 4.0 and
CC BY-NC-SA 3.0). They are used, and their images are shared, only for
non-commercial purposes, as those licences require. If you reuse these assets
or images, their licences apply to you.

Bundled in this repository:

- Ajax bust: courtesy of Jotero (http://forum.jotero.com), as credited by
  Miles Macklin's Tinsel (https://github.com/mmacklin/tinsel); this copy comes
  from knightcrawler25's GLSL-PathTracer
  (https://github.com/knightcrawler25/GLSL-PathTracer). Licence: none stated
  by the upstream sources (Jotero, via mmacklin/tinsel and
  knightcrawler25/GLSL-PathTracer).
- Hygieia bust: "Hygieia" by noe-3d.at on Sketchfab,
  https://sketchfab.com/3d-models/hygieia-fd53f578e9bc4fbaa52551c2f5175cde,
  CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/).
- HDRIs `san_giuseppe_bridge_4k.hdr` and `venice_sunrise_4k.hdr`: Poly Haven
  (https://polyhaven.com/), CC0.

Gallery assets (from the asset pack):

- Amazon Lumberyard Bistro, Open Research Content Archive (ORCA), Amazon
  Lumberyard, 2017, http://developer.nvidia.com/orca/amazon-lumberyard-bistro,
  CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); converted to glTF
  for this renderer.
- San Miguel by Guillermo M. Leal Llaguno (Evolucien Visual,
  http://www.evvisual.com/), 2017 version improved by Morgan McGuire, Guedis
  Cardenas, Michael Mara and Nicholas Hull, http://casual-effects.com/data; free
  for research and educational use; textures re-authored for this renderer.
- Sponza, Khronos glTF Sample Assets
  (https://github.com/KhronosGroup/glTF-Sample-Assets): original by Marko
  Dabrovic, model by Frank Meinl (Crytek), PBR textures by Alexandre Pestana;
  under the Cryengine Limited License Agreement.
- Unreal Engine Sun Temple by Epic Games (exported by Kai-Hwa Yao and Nicholas
  Hull), Open Research Content Archive,
  http://developer.nvidia.com/orca/epic-games-sun-temple, CC BY-NC-SA 4.0
  (http://creativecommons.org/licenses/by-nc-sa/4.0/); converted for this
  renderer. The Sun Temple image is shared under the same licence.
- NVIDIA Emerald Square by Nicholas Hull, Kate Anderson and Nir Benty (NVIDIA),
  Open Research Content Archive,
  http://developer.nvidia.com/orca/nvidia-emerald-square, CC BY-NC-SA 3.0
  (http://creativecommons.org/licenses/by-nc-sa/3.0/); foliage by SpeedTree;
  converted for this renderer. The Emerald Square images are shared under the
  same licence.
- Zero-Day by Mike Winkelmann (Beeple) and Kate Anderson, Open Research Content
  Archive, https://developer.nvidia.com/orca/beeple-zero-day, CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/); converted for this renderer.
- Barcelona Pavilion by Claudio Andres (base model), Hamza Cheggour / eMirage
  (https://www.emirage.org/), glTF conversion by Piotr Przybysz, CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/); HDRIs from Poly Haven (CC0);
  re-materialed for this renderer.
- Transparent Machines by Mike Winkelmann (Beeple),
  https://www.beeple-crap.com/resources, glTF conversion by Piotr Przybysz,
  CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); re-materialed for
  this renderer.
- Bitterli scenes via Benedikt Bitterli's Rendering Resources
  (https://benedikt-bitterli.me/resources/), converted for this renderer:
  - "Modern Hall" (staircase) by NewSee2l035,
    http://www.blendswap.com/blends/view/51997, CC BY 3.0
    (https://creativecommons.org/licenses/by/3.0/).
  - "Glass of Water" by aXel (Blend Swap 21697), CC0.
  - "4060.b Spaceship" by thecali (Blend Swap 75686), CC0.
  - "Veach, Ajar" by Benedikt Bitterli, CC0.
- Damaged Helmet by ctxwing (rebuild, CC BY 4.0,
  https://creativecommons.org/licenses/by/4.0/) and theblueturtle_ (earlier
  version, CC BY-NC 4.0), via Khronos glTF Sample Assets
  (https://github.com/KhronosGroup/glTF-Sample-Assets); re-materialed for this
  renderer.
- Bugatti Chiron Top Edition by iamsunroy,
  https://sketchfab.com/3d-models/bugatti-chiron-top-edition-3d1f955d5e7a4df4b44787ceb3e2e975,
  Sketchfab Standard licence; imported and re-materialed. The model itself is
  not distributed.
- Marble and wax busts (`sss_marble_wax`): the Ajax bust, courtesy of Jotero
  (see the bundled assets above).
- Suzanne, Blender's test model (CC0 copy via Khronos glTF Sample Assets, UX3D).
- Formula 1 cars: all fourteen models are by OUTPISTON
  (https://sketchfab.com/outpiston) on Sketchfab, licensed CC BY-NC-SA 4.0
  (https://creativecommons.org/licenses/by-nc-sa/4.0/), converted and
  re-materialed for this renderer. Their images are shared under the same
  licence. Each model's page is listed in `assets/CREDITS.md`. The W10, RB15
  and C39 pages state that the models were extracted from Real Racing 3. Team
  names, liveries and car designs are trademarks of their owners. The images
  are renderer tests, not endorsed by any team.

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
