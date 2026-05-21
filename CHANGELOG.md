# Changelog

## v3.0.0 - Public Renderer Scalability Release

### Renderer Scalability and Large-Scene Infrastructure

- Added headless scene statistics reporting for triangle, mesh, instance,
  texture, geometry-memory, texture-memory, BLAS, TLAS, and estimated
  working-set totals.
- Added `--budget-policy=<strict|warn|ignore>` to enforce or audit scene memory
  budgets against `MTLDevice.recommendedMaxWorkingSetSize`.
- Added BLAS/TLAS build instrumentation and explicit backend logging so Metal
  HWRT fallback to SWRT is never silent.
- Added explicit runtime/backend controls for headless renders:
  `--stats`, `--force-hwrt`, and `--force-swrt`.
- Added large-scene pilot controls for Bistro-class scene bring-up:
  `--pilot-mode=<geo_only|albedo_only|full_clamped>`,
  `--texture-max-dim`, and `--max-texture-bytes`.

### Metal Wavefront Execution and Pass Graph

- Added selectable execution modes through `--executionMode=<megakernel|wavefront>`.
- Added wavefront scheduling policies:
  `--wavefront-policy=<preview|final|offline|research>`.
- Added optional queue compaction through `--wavefront-compaction=<0|1>`.
- Added Metal pass graph and wavefront metrics infrastructure for headless
  validation and profiling.
- Added public wavefront render gallery images under `Images/Metal-Wavefront-Images`.

### ReSTIR / RIS / ReGIR Research Lighting Stack

- Added direct-light modes: `legacy`, `baseline_emissive`, `ris`,
  `ris_spatial`, `ris_temporal`, `ris_world`, `ris_regir`, `restir_di`,
  and `restir_di_regir_hybrid`.
- Added ReSTIR DI explicit pass graph support for headless rendering and
  opt-in GUI profiling.
- Added ReGIR/world-space candidate reuse and a DI+ReGIR hybrid path.
- Added bounded ReSTIR GI diffuse prototype.
- Added ReSTIR PT research scaffold and experimental diffuse-safe path reuse.
- Added path guiding and radiance cache prototypes.
- Kept high-risk research paths disabled by default and gated behind explicit
  CLI/UI controls.

### Advanced Transport Prototypes

- Added path-space caustic transport prototype.
- Added homogeneous volume transport prototype with isotropic and
  Henyey-Greenstein phase options.
- Added hero-wavelength spectral transport prototype with dielectric dispersion
  controls.
- Added Metal SVGF-style denoise as an experimental sidecar path.
- Added optional firefly clamp modes for offline beauty output control.

### Import Pipeline, glTF Cache, and Texture Conversion

- Added `PathTracerImport`, a standalone importer for Assimp-supported static
  scenes.
- Added deterministic glTF/GLB import output with `import_manifest.json`.
- Added import manifest schema emission with `schema_version: "1.0"`.
- Added optional CMake Assimp discovery with a stub fallback when Assimp is not
  installed.
- Added importer texture policies for copied external textures plus experimental
  embedded and linked texture modes.
- Added deterministic `.ktx2` texture conversion for Bistro DDS textures and
  common raster formats.
- Added runtime KTX2 texture loading for converted importer output.

### Production Metadata and Offline Workflow

- Added render profiles: `custom`, `preview`, `lookdev`, `final`, `reference`,
  and `debug`.
- Added deterministic settings JSON and render queue item JSON output.
- Added render queue replay through `--runRenderQueueItemJson`.
- Added production metadata sidecars and debug bundle directory output.
- Added checkpoint manifest, resume support, and tiled render manifest output.
- Added PBR metrics JSON, material-channel inspection, Material_MR debug JSON,
  direct-light audit JSON, and camera search reports.

### Public Scenes, Assets, and Gallery Updates

- Added public Ajax scenes: `ajax.scene`, `ajax-other.scene`, and
  `ajax-glass.scene`.
- Added public Hygieia scenes: `hygieia.scene` and `hygieia-other.scene`.
- Added Ajax and Hygieia meshes plus the two HDRIs used by those scenes.
- Consolidated public render galleries under `Images/`.
- Added `assets/CREDITS.md` with upstream asset acknowledgements and
  licensing/source notes.
- Updated README image paths for the consolidated `Images/` layout.
- Updated the public asset-pack link and installation instructions for
  `Metal-PathTracer-Assets-v3-0-0.zip`.

### Optional Reference Backends and Public CI

- Kept Embree as an optional CPU reference backend when installed and configured
  with `-DPATH_TRACER_ENABLE_EMBREE=ON`.
- Kept `--backend=<metal|embree>` and `--threads=<int>` for compatible builds.
- Kept Intel Open Image Denoise optional; clean builds without vendored OIDN/TBB
  libraries compile with OIDN disabled and report denoise unavailable.
- Updated public CI so source-only Metal/headless builds do not require local
  Embree or OIDN binaries.
- Removed private `docs/` and `scripts/` content from the public repository while
  preserving public tests and validation baselines.

## v2.0.4 - Environment Robustness (Sponza)

- Added canonical `sponza` scene discovery via `assets/sponza.scene`.
- Added fixed Sponza camera and headless baseline settings for deterministic 1280x720 EXR renders.
- Added `--spp` and `--out` CLI aliases for milestone reproduction commands.
- Added a dedicated `scripts/v2_0_4_sponza_validate.py` acceptance runner for determinism and HWRT/SWRT parity.
- Documented the canonical Sponza baseline and HWRT/SWRT parity workflow in `README.md`.
