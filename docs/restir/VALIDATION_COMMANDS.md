# ReSTIR Validation Commands

Use the established build directory:

```sh
cmake --build build-restir --target PathTracerHeadless
```

The build must show the shader-copy dependency before PathTracerHeadless completion.

## Regression Ladder

```sh
python3 scripts/validate_living_room_regir_stress.py
python3 scripts/validate_living_room_regir_practicals_stress.py
python3 scripts/validate_restir_di_regir_hybrid.py --binary ./build-restir/PathTracerHeadless
python3 scripts/validate_restir_gi_prototype.py --binary ./build-restir/PathTracerHeadless
python3 scripts/validate_svgf_denoise_prototype.py --binary ./build-restir/PathTracerHeadless
python3 scripts/validate_metal_pass_graph.py --binary ./build-restir/PathTracerHeadless
python3 scripts/validate_path_guiding_prototype.py --binary ./build-restir/PathTracerHeadless --artifact-dir /private/tmp/restir_stage_path_guiding_regression --verbose
python3 scripts/validate_restir_pt_research_scaffold.py --binary ./build-restir/PathTracerHeadless --artifact-dir /private/tmp/restir_stage_restir_pt_scaffold_regression
python3 scripts/validate_restir_pt_experimental_path_reuse.py --binary ./build-restir/PathTracerHeadless --artifact-dir /private/tmp/restir_stage_experimental_path_reuse_regression
python3 scripts/validate_restir_pt_interactions.py --binary ./build-restir/PathTracerHeadless --artifact-dir /private/tmp/restir_stage_pt_interactions_regression
python3 scripts/validate_restir_gui_debug_inspector.py --binary ./build-restir/PathTracerHeadless --artifact-dir /private/tmp/restir_r16_gui_debug_inspector --verbose
python3 scripts/validate_restir_r6_structural.py --binary ./build-restir/PathTracerHeadless
python3 scripts/validate_restir_roadmap_full_stack.py --binary ./build-restir/PathTracerHeadless --artifact-dir /private/tmp/restir_stage_full_stack_regression --verbose
python3 scripts/validate_restir_final_full_stack_long_run.py --binary ./build-restir/PathTracerHeadless --artifact-dir /private/tmp/restir_r12_final_full_stack_long_run --verbose
python3 scripts/validate_restir_remote_clean_reproducibility.py --binary ./build-restir/PathTracerHeadless --artifact-dir /private/tmp/restir_remote_clean_reproducibility --verbose
```

## Stage Validators

| Stage | Command |
| --- | --- |
| R2 | `python3 scripts/validate_restir_di_regir_hybrid.py --binary ./build-restir/PathTracerHeadless` |
| R3 | `python3 scripts/validate_restir_gi_prototype.py --binary ./build-restir/PathTracerHeadless` |
| R4 | `python3 scripts/validate_svgf_denoise_prototype.py --binary ./build-restir/PathTracerHeadless` |
| R5 | `python3 scripts/validate_metal_pass_graph.py --binary ./build-restir/PathTracerHeadless` |
| R6 | `python3 scripts/validate_restir_roadmap_full_stack.py --binary ./build-restir/PathTracerHeadless --artifact-dir /private/tmp/restir_r6_full_stack_validation --verbose` |
| R7 | `python3 scripts/validate_path_guiding_prototype.py --binary ./build-restir/PathTracerHeadless --artifact-dir /private/tmp/restir_r7_path_guiding_validation --verbose` |
| R8 | `python3 scripts/validate_restir_pt_research_scaffold.py --binary ./build-restir/PathTracerHeadless --artifact-dir /private/tmp/restir_r8_restir_pt_scaffold_validation --verbose` |
| R9 | `python3 scripts/validate_restir_pt_experimental_path_reuse.py --binary ./build-restir/PathTracerHeadless --artifact-dir /private/tmp/restir_r9_experimental_path_reuse_validation --verbose` |
| R10 | `python3 scripts/validate_restir_pt_interactions.py --binary ./build-restir/PathTracerHeadless --artifact-dir /private/tmp/restir_r10_pt_interactions_validation --verbose` |
| R12 | `python3 scripts/validate_restir_final_full_stack_long_run.py --binary ./build-restir/PathTracerHeadless --artifact-dir /private/tmp/restir_r12_final_full_stack_long_run --verbose` |
| R13 | `python3 scripts/validate_restir_remote_clean_reproducibility.py --binary ./build-restir/PathTracerHeadless --artifact-dir /private/tmp/restir_remote_clean_reproducibility --verbose` |
| R14 | Documentation package plus `validate_restir_remote_clean_reproducibility.py`, `validate_restir_roadmap_full_stack.py`, and `validate_restir_final_full_stack_long_run.py` from a clean worktree. |
| R15 | Visual review package plus `validate_restir_remote_clean_reproducibility.py` and `validate_restir_roadmap_full_stack.py` from a clean worktree. |
| R16 | `python3 scripts/validate_restir_gui_debug_inspector.py --binary ./build-restir/PathTracerHeadless --artifact-dir /private/tmp/restir_r16_gui_debug_inspector --verbose` |

## Checkpoint and Structural Validators

The full-stack validator orchestrates the current checkpoint ladder:

- `scripts/validate_restir_r3_checkpoint1.py`
- `scripts/validate_restir_r4_checkpoint1.py`
- `scripts/validate_restir_r5_checkpoint1.py`
- `scripts/validate_restir_r5_structural.py`
- `scripts/validate_restir_r6_checkpoint1.py`
- `scripts/validate_restir_r6_structural.py`

Run `find scripts tests docs -iname '*restir*' -o -iname '*regir*' -o -iname '*ris*' -o -iname '*gi*' -o -iname '*svgf*' -o -iname '*pass*graph*' -o -iname '*guid*' -o -iname '*pt*' -o -iname '*r6*'` before adding new validators to avoid duplicating an existing checkpoint.

## Core Render Settings Used by Prototype Validators

- Backend: Metal
- Software ray tracing: `--enableSoftwareRayTracing=1`
- Resolution: 96x54 for iterative prototype validators; 128x72 for the R12 longer-run gate
- SPP: 2 or 3 for iterative prototype validators; 4 for the R12 longer-run gate
- Seed: 42 unless a validator enumerates multiple seeds
- Denoise: disabled for core parity, enabled only for explicit SVGF sidecar checks
- Output format: PFM for metric validators

## Pass Criteria

- Build passes.
- Required stage validator passes.
- Previous-stage regression validators pass.
- Megakernel and wavefront render paths both run.
- Parity metrics remain inside the stage threshold.
- NaN/Inf pixel and component counts are zero.
- Experimental/debug features remain gated and disabled by default.

## Remote-Clean Reproducibility

R13 uses a clean worktree created from `origin/feature/restir-di-proto` and may use a fresh build directory such as `build-r13-clean`:

```sh
git worktree add ../Metal-PathTracer-restir-r13-clean -b restir-r13-clean origin/feature/restir-di-proto
cd ../Metal-PathTracer-restir-r13-clean
cmake -S . -B build-r13-clean
cmake --build build-r13-clean --target PathTracerHeadless
python3 scripts/validate_restir_remote_clean_reproducibility.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r13_remote_clean_reproducibility --verbose
python3 scripts/validate_restir_final_full_stack_long_run.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r13_final_full_stack_long_run --verbose
```

Clean builds do not require ignored local OIDN dylibs. When vendored OIDN/TBB binaries are absent, the renderer builds with OIDN disabled and `--denoise` reports unavailable; ReSTIR/SVGF roadmap validators do not depend on OIDN.

## PR Package Gate

R14 uses the clean worktree and compares `feature/restir-di-proto` against `origin/main`:

```sh
git fetch origin
git log --oneline --decorate --graph origin/main..HEAD
git diff --stat origin/main..HEAD
git diff --name-status origin/main..HEAD
git diff --shortstat origin/main..HEAD
find assets -type f -size +25M -print -exec ls -lh {} \;
cmake --build build-r13-clean --target PathTracerHeadless
python3 scripts/validate_restir_remote_clean_reproducibility.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r14_remote_clean_reproducibility --verbose
python3 scripts/validate_restir_roadmap_full_stack.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r14_full_stack_regression --verbose
python3 scripts/validate_restir_final_full_stack_long_run.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r14_final_full_stack_long_run --verbose
```
