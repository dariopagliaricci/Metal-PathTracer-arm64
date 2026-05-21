# R13 Remote-Clean Reproducibility and PR-Readiness Validation

Final status: PASS

Stage: R13_REMOTE_CLEAN_REPRODUCIBILITY_AND_PR_READINESS_GATE

Branch: `restir-r13-clean`
Source remote branch: `origin/feature/restir-di-proto`
Target pushed branch: `origin/feature/restir-di-proto`
Clean worktree path: `/Users/dariopagliaricci/Metal-PathTracer-restir-r13-clean`
Parent R12 commit: `f9f89ce42a6647256936325f610bbdd7438c0fe7`
Final R13 commit hash: recorded by the pushed R13 commit for this report; exact hash is reported by `git log -1 origin/feature/restir-di-proto` after push.

This stage did not add a rendering algorithm. It validated that the pushed roadmap branch can be checked out from the remote, configured, built, validated, and documented without relying on dirty local files, ignored build products, stale shaders, or local-only assets.

## Files Changed

- `CMakeLists.txt`
- `src/MetalRenderer.mm`
- `src/renderer/DenoiserContext.mm`
- `src/renderer/OidnCpuDenoiser.cpp`
- `scripts/validate_restir_roadmap_full_stack.py`
- `scripts/validate_restir_remote_clean_reproducibility.py`
- `docs/restir/ROADMAP_STATUS.md`
- `docs/restir/VALIDATION_COMMANDS.md`
- `docs/restir/validation/R13_REMOTE_CLEAN_REPRODUCIBILITY_PR_READINESS.md`
- `assets/living_room.scene`
- `assets/living_room/living_room.obj`
- `assets/living_room/living_room.mtl`
- `assets/canonical/living_room/import_manifest.json`
- `assets/canonical/living_room/living_room.glb`
- `assets/canonical/living_room/textures/tex_0_apple.jpg`
- `assets/canonical/living_room/textures/tex_1_book-spines.jpg`
- `assets/canonical/living_room/textures/tex_2_carpet-text3b.jpg`
- `assets/canonical/living_room/textures/tex_3_shade-paper.jpg`
- `assets/canonical/living_room/textures/tex_4_cushion-green-circles.jpg`
- `assets/canonical/living_room/textures/tex_5_cushion-stripe-purple.jpg`
- `assets/canonical/living_room/textures/tex_6_cushion-purple-yellow-stripe.jpg`
- `assets/canonical/living_room/textures/tex_7_wood4.jpg`
- `assets/canonical/living_room/textures/tex_8_shade-stripes.jpg`
- `assets/canonical/living_room/textures/tex_9_pic5wide.jpg`
- `assets/canonical/living_room/textures/tex_10_magazine.jpg`
- `assets/canonical/living_room/textures/tex_11_picture11-vert.jpg`
- `assets/canonical/living_room/textures/tex_12_photo1.jpg`
- `assets/canonical/living_room/textures/tex_13_photo3.jpg`
- `assets/canonical/living_room/textures/tex_14_photo2.jpg`
- `assets/canonical/living_room/textures/tex_15_photo4.jpg`

## Reproducibility Fixes

- OIDN is now optional at configure time. The clean remote checkout contains OIDN headers but not the ignored local native dylibs. `PATH_TRACER_ENABLE_OIDN` remains enabled when vendored OIDN/TBB libraries exist, and clean builds degrade to OIDN disabled when they do not. Normal rendering and all ReSTIR/SVGF roadmap validators remain independent of OIDN; `--denoise` reports unavailable in a clean build without the dylibs.
- `MetalRenderer` no longer attempts OIDN startup when the binary was built without OIDN support.
- CPU denoiser entry points now return clean unavailable results when OIDN is not compiled in.
- The full-stack validator now honors the `--binary` path passed by the caller instead of hardcoding `./build-restir/PathTracerHeadless`.
- Canonical Living Room assets required by the existing ReGIR validators are now present on the remote branch. The assets were copied from the local canonical source and were not modified or reframed.

## Exact Commands Run

From the original worktree:

```sh
cd /Users/dariopagliaricci/Metal-PathTracer-feature-restir-di-proto
git fetch origin
git status --short
git log -1 --oneline
git worktree add ../Metal-PathTracer-restir-r13-clean -b restir-r13-clean origin/feature/restir-di-proto
```

From the clean worktree:

```sh
cd /Users/dariopagliaricci/Metal-PathTracer-restir-r13-clean
git status --short
git log -1 --oneline
git branch --show-current
git log --oneline --decorate -20
ls docs/restir
ls docs/restir/validation
grep -R "R12_FINAL_FULL_STACK_LONG_RUN_VALIDATION" -n docs/restir scripts
grep -R "R11_ROADMAP" -n docs/restir scripts
grep -R "R10_" -n docs/restir scripts
grep -R "R9_" -n docs/restir scripts
grep -R "R8_" -n docs/restir scripts
grep -R "restir_pt" -n include src shaders scripts docs
grep -R "path_guiding" -n include src shaders scripts docs
grep -R "svgfDenoise" -n include src shaders scripts docs
grep -R "RenderPassGraph" -n include src shaders scripts docs
grep -R "restir_gi_prototype" -n include src shaders scripts docs
cmake -S . -B build-r13-clean
cmake --build build-r13-clean --target PathTracerHeadless
python3 scripts/validate_living_room_regir_stress.py
python3 scripts/validate_living_room_regir_practicals_stress.py
python3 scripts/validate_restir_remote_clean_reproducibility.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r13_remote_clean_reproducibility --verbose
python3 scripts/validate_restir_di_regir_hybrid.py --binary ./build-r13-clean/PathTracerHeadless
python3 scripts/validate_restir_gi_prototype.py --binary ./build-r13-clean/PathTracerHeadless
python3 scripts/validate_svgf_denoise_prototype.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r13_svgf_regression --verbose
python3 scripts/validate_metal_pass_graph.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r13_pass_graph_regression --verbose
python3 scripts/validate_path_guiding_prototype.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r13_path_guiding_regression --verbose
python3 scripts/validate_restir_pt_research_scaffold.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r13_restir_pt_scaffold_regression --verbose
python3 scripts/validate_restir_pt_experimental_path_reuse.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r13_restir_pt_reuse_regression --verbose
python3 scripts/validate_restir_pt_interactions.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r13_restir_pt_interactions_regression --verbose
python3 scripts/validate_restir_r6_structural.py --binary ./build-r13-clean/PathTracerHeadless
python3 scripts/validate_restir_roadmap_full_stack.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r13_full_stack_regression --verbose
python3 scripts/validate_restir_final_full_stack_long_run.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r13_final_full_stack_long_run --verbose
```

## Build Result

Build directory: `build-r13-clean`

Result: PASS

The build completed from a remote-clean worktree using:

```sh
cmake -S . -B build-r13-clean
cmake --build build-r13-clean --target PathTracerHeadless
```

Shader-copy dependency remained effective through the existing `PathTracerHeadless` dependency chain. OIDN emitted a configure-time warning because the ignored local dylibs were absent from the clean checkout, then the renderer built with OIDN disabled as intended.

## Validators Run

All validators passed from the clean worktree:

| Validator | Result | Evidence |
| --- | --- | --- |
| `validate_living_room_regir_stress.py` | PASS | `stress_emissive_triangles=131072`, `total_area=11.700911561`, `total_luminance_power=40.518020247` |
| `validate_living_room_regir_practicals_stress.py` | PASS | `panel_emissive_triangles=131072`, `practical_emissive_triangles=24025`, `total_emissive_triangles=155097`, `practical_flux_percent=3.500000` |
| `validate_restir_remote_clean_reproducibility.py` | PASS | Summary JSON: `/private/tmp/restir_r13_remote_clean_reproducibility/r13_remote_clean_reproducibility_summary.json` |
| `validate_restir_di_regir_hybrid.py` | PASS | seed 42 ratio `0.997144`, seed 123 ratio `0.963065` |
| `validate_restir_gi_prototype.py` | PASS | seed 42 off/gi means `0.031680079` / `0.031662512`; seed 123 `0.030651083` / `0.031017195` |
| `validate_svgf_denoise_prototype.py` | PASS | variance reduction `0.193198`, megakernel/wavefront RMSE `0.0` |
| `validate_metal_pass_graph.py` | PASS | megakernel passes `3`, wavefront passes `10` |
| `validate_path_guiding_prototype.py` | PASS | active guide used count `1456`, parity cases `4` |
| `validate_restir_pt_research_scaffold.py` | PASS | active candidates `1472`, active updates `1472`, parity cases `4` |
| `validate_restir_pt_experimental_path_reuse.py` | PASS | active applied count `1440`, parity cases `4` |
| `validate_restir_pt_interactions.py` | PASS | reuse applied `1440`, combinations `4`, parity cases `2` |
| `validate_restir_r6_structural.py` | PASS | invalid fallback rejects and cache accepts validated for seeds `42`, `123`, `321`, `777`, `999` |
| `validate_restir_roadmap_full_stack.py` | PASS | modes `9`, parity cases `4`, validators `12` |
| `validate_restir_final_full_stack_long_run.py` | PASS | controls `5`, active modes `6`, parity cases `4`, reuse applied `2555` |

## Artifact Directories

- `/private/tmp/restir_r13_remote_clean_reproducibility`
- `/private/tmp/restir_r13_svgf_regression`
- `/private/tmp/restir_r13_pass_graph_regression`
- `/private/tmp/restir_r13_path_guiding_regression`
- `/private/tmp/restir_r13_restir_pt_scaffold_regression`
- `/private/tmp/restir_r13_restir_pt_reuse_regression`
- `/private/tmp/restir_r13_restir_pt_interactions_regression`
- `/private/tmp/restir_r13_full_stack_regression`
- `/private/tmp/restir_r13_final_full_stack_long_run`

## Render and Metric Outputs

- R13 reproducibility summary: `/private/tmp/restir_r13_remote_clean_reproducibility/r13_remote_clean_reproducibility_summary.json`
- R13 default smoke image: `/private/tmp/restir_r13_remote_clean_reproducibility/r13_default_smoke.pfm`
- R13 default smoke metrics: `/private/tmp/restir_r13_remote_clean_reproducibility/r13_default_smoke_metrics.json`
- R6 full-stack clean-worktree summary: `/private/tmp/restir_r13_full_stack_regression/restir_r6_full_stack_summary.json`
- R12 final clean-worktree summary: `/private/tmp/restir_r13_final_full_stack_long_run/r12_final_full_stack_summary.json`

## Scenes and Settings

Scenes tested:

- `assets/living_room_regir_stress.scene`
- `assets/living_room_regir_practicals_stress.scene`
- `tests/scenes/restir/r3/r3_diffuse_indirect.scene`
- the R5/R6/R7/R8/R9/R10/R12 validator fixtures referenced by the stage validators

Core clean-worktree R12 settings:

- Backend: Metal
- Software ray tracing: enabled
- Resolution: `128x72`
- SPP: `4`
- Seed: `42`
- Execution modes: megakernel and wavefront
- Path guiding strength: `0.5`
- ReSTIR PT reuse strength: `0.25`
- ReSTIR PT max reservoirs: `4096`
- Denoise: disabled for core parity; SVGF enabled only in explicit sidecar checks

Feature modes tested:

- `baseline_emissive`
- `ris`
- `ris_regir`
- `restir_di`
- `restir_di_regir_hybrid`
- `restir_gi_prototype`
- `path_guiding_prototype`
- `restir_pt_research`
- `restir_pt_experimental_path_reuse`
- `svgfDenoise`
- pass graph metrics export

## Megakernel/Wavefront Parity Metrics

R13 clean-worktree R12 longer-run parity:

| Fixture | PSNR | SSIM | Log RMSE | RGB RMSE | NaN/Inf |
| --- | ---: | ---: | ---: | ---: | --- |
| `living_room_clean_ris_regir_reuse` | `32.104748951` | `0.996934569` | `0.024817758` | `0.036904589` | zero |
| `living_room_practicals_ris_regir_reuse` | `31.052707441` | `0.996270971` | `0.028013323` | `0.040782348` | zero |
| `r3_diffuse_full_stack` | `43.210455255` | `0.998055169` | `0.006909987` | `0.010905338` | zero |
| `r3_diffuse_reuse` | `inf` | `1.000000000` | `0.000000000` | `0.000000000` | zero |

R13 clean-worktree R6 full-stack parity:

| Fixture | PSNR | SSIM | Log RMSE | RGB RMSE | NaN/Inf |
| --- | ---: | ---: | ---: | ---: | --- |
| `living_room_clean_regir` | `32.760615028` | `0.997474310` | `0.023012789` | `0.026211163` | zero |
| `living_room_practicals_regir` | `30.564209639` | `0.995835369` | `0.029633948` | `0.041698471` | zero |
| `r3_diffuse_indirect_gi` | `inf` | `1.000000000` | `0.000000000` | `0.000000000` | zero |
| `r5_pass_graph_svgf` | `inf` | `1.000000000` | `0.000000000` | `0.000000000` | zero |

## NaN/Inf and Energy Checks

R13 default smoke:

- Mean luminance: `0.031309634871571325`
- Max component: `3.5999999046325684`
- NaN/Inf pixel count: `0`
- NaN/Inf component count: `0`
- NaN/Inf throughput count: `0`
- NaN/Inf radiance count: `0`

R12 clean-worktree longer run:

- NaN/Inf counts: zero in required validations
- No black-out detected
- No severe energy explosion detected

## Active-Mode Counters

R12 clean-worktree active-mode proof:

- Path guiding used count: `2560`
- ReSTIR PT research candidates: `2558`
- ReSTIR PT research reservoir updates: `2558`
- ReSTIR PT research rejected invalid: `0`
- ReSTIR PT research rejected unsupported: `3334`
- ReSTIR PT experimental reuse candidates: `2555`
- ReSTIR PT experimental reuse applied: `2555`
- ReSTIR PT experimental reuse fallback: `3334`
- ReSTIR PT experimental reuse rejected invalid: `0`
- Full-stack sidecar reuse candidates: `2572`
- Full-stack sidecar reuse applied: `2572`
- Full-stack sidecar guiding candidates: `2572`
- Full-stack sidecar guiding used: `2572`

## Default-State Audit

The R13 reproducibility validator ran a default Metal smoke render and verified:

- `restir_gi_mode=off`
- `path_guiding_mode=off`
- `restir_pt_mode=off`
- `svgf_denoise_enabled=false`
- path-guiding counters remain zero by default
- ReSTIR PT counters remain zero by default
- debug pixel and direct-light audit outputs are created only when explicit paths/flags are supplied

Result: PASS

## Gating Audit

Experimental features remain gated:

- ReSTIR GI prototype requires `--restirGiMode=restir_gi_prototype`
- SVGF requires `--svgfDenoise=1`
- Path guiding requires `--pathGuidingMode=path_guiding_prototype`
- ReSTIR PT research requires `--restirPtMode=restir_pt_research`
- ReSTIR PT experimental reuse requires `--restirPtMode=restir_pt_experimental_path_reuse`
- ReSTIR PT debug requires `--restirPtDebug=1`
- debug pixel capture requires `--debugPixel=x,y`
- direct-light audit JSON requires `--directLightAuditJson=<path>`
- pass graph metrics require explicit metrics output in validators

Result: PASS

## Documentation Consistency Audit

Checked:

- `docs/restir/ROADMAP_STATUS.md`
- `docs/restir/VALIDATION_COMMANDS.md`
- `docs/restir/FEATURE_GATES.md`
- `docs/restir/validation/R12_FINAL_FULL_STACK_LONG_RUN_VALIDATION.md`
- all R1-R12 validation reports
- all validators referenced by the release-candidate command ladder

R4 remains marked `EXPERIMENTAL_PASS`, not production PASS. R8/R9/R10/R11/R12 reports and validators are present. Feature gate names match the headless help text checked by the R13 validator. The R13 report and command matrix document clean-worktree validation.

Result: PASS

## PR-Readiness Audit

- Branch: `feature/restir-di-proto`
- Remote branch: `origin/feature/restir-di-proto`
- Clean validation branch: `restir-r13-clean`
- Base branch: not changed by R13; infer during PR creation from repository default branch.
- Latest pre-R13 roadmap commit: `f9f89ce42a6647256936325f610bbdd7438c0fe7`
- Roadmap validation reports present before R13: `12`
- R13 validation report added: `1`
- Validators run in R13: `14`
- Build directory used: `build-r13-clean`
- Experimental features remain disabled by default.
- Required canonical Living Room assets are now present on the remote branch.
- Known merge risk: large canonical asset files are now tracked so the branch is self-contained for validators.
- Human review recommendation: code review and optional visual review of representative validation outputs before merge.

Result: PASS

## Known Limitations

- OIDN native dylibs remain local/ignored unless a developer vendors them into `external/oidn/lib`. Clean remote builds compile with OIDN disabled and report `--denoise` unavailable. This does not affect ReSTIR/SVGF roadmap validators.
- R8/R9 remain bounded research/prototype ReSTIR PT modes, not production ReSTIR PT.
- R4 remains an experimental SVGF-style sidecar.
- R13 did not perform subjective visual quality approval; it is a reproducibility and PR-readiness gate.

## Unrelated Files

The original worktree contained unrelated dirty/untracked files. R13 validation and commits were performed from `/Users/dariopagliaricci/Metal-PathTracer-restir-r13-clean`, and the original unrelated files were not modified, staged, reset, deleted, or stashed.

## Explicit Non-Changes

- No new rendering algorithm was implemented.
- ReSTIR PT behavior was not expanded beyond existing R8/R9 roadmap modes.
- Path guiding behavior was not expanded.
- SVGF behavior was not expanded.
- Canonical scene camera framing was not modified.
- Neural caching was not implemented.
