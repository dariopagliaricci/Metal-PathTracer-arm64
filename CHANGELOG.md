# Changelog

## v3.2.1 - Texture Closure Milestone

- Added importer `schema_version: "1.0"` emission in `import_manifest.json`.
- Added `--textures=convert` with deterministic `.ktx2` output for Bistro’s DDS set plus common raster image formats.
- Added explicit non-zero stub diagnostics when `PathTracerImport` is built without Assimp.
- Added pending Bistro canonical manifest metadata in `assets/canonical/bistro.yaml`.
- Added runtime `.ktx2` texture loading in the renderer for converted import output.

## v3.2.0 - Assimp Import Pipeline + glTF Cache

- Added `PathTracerImport`, a standalone importer that converts Assimp-supported static scenes into deterministic glTF 2.0 output with `import_manifest.json`.
- Added optional CMake Assimp discovery with a stub fallback so the project still configures and builds cleanly when Assimp is not installed.
- Added importer texture policies for copied external textures by default plus experimental embedded and linked texture modes.
- Documented the Bistro FBX import workflow and importer CLI in `README.md`.

## v3.1.0 - Large Scene Pilot Framework (Bistro Readiness)

- Added stable `bistro_exterior` and `bistro_interior` pilot scene IDs backed by the existing ORCA Bistro asset pack under `assets/canonical/bistro`.
- Added a pre-build Large Scene Report for headless runs with geometry totals, top meshes, top textures, device budget reporting, and pilot-policy outcome logging.
- Added `--pilot-mode=<geo_only|albedo_only|full_clamped>` for controlled Bistro bring-up without touching the runtime integrator.
- Added `--texture-max-dim` and `--max-texture-bytes` with explicit clamp/skip logging integrated into material texture loading.
- Added pre-build strict fail-fast handling when pilot texture policy or pre-build working-set estimates already exceed the selected budget policy.

## v3.0.0 - Infrastructure & Scalability Foundation

- Added headless scene statistics reporting for triangle / mesh / instance / texture counts plus geometry, texture, BLAS, TLAS, and total estimated working-set memory.
- Added `--budget-policy=<strict|warn|ignore>` to enforce scene memory budgets against `MTLDevice.recommendedMaxWorkingSetSize`.
- Added BLAS / TLAS build instrumentation and explicit backend logging so Metal HWRT fallback to SWRT is never silent.
- Added `--stats` plus `--force-hwrt` / `--force-swrt` for explicit runtime/backend control during headless renders.
- Documented the v3 canonical benchmark commands for `bitterli_staircase` and `sponza` in `README.md`.

## v2.0.4 - Environment Robustness (Sponza)

- Added canonical `sponza` scene discovery via `assets/sponza.scene`.
- Added fixed Sponza camera and headless baseline settings for deterministic 1280x720 EXR renders.
- Added `--spp` and `--out` CLI aliases for milestone reproduction commands.
- Added a dedicated `scripts/v2_0_4_sponza_validate.py` acceptance runner for determinism and HWRT/SWRT parity.
- Documented the canonical Sponza baseline and HWRT/SWRT parity workflow in `README.md`.
