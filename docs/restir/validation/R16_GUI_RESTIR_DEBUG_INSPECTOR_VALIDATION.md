# R16 GUI ReSTIR Debug Inspector Validation

Date: 2026-05-01

Status: PASS for the implemented GUI/headless debug inspector plumbing.

Branch: `feature/restir-di-proto`
Remote branch: `origin/feature/restir-di-proto`
Clean worktree: `/Users/dariopagliaricci/Metal-PathTracer-restir-r13-clean`
Parent R15 commit: `74ce5696c5a98d83aea4bba9e826e49b4eba566f`
Final R16 commit hash: recorded in the R16 closure response after commit.

## Scope

R16 adds a gated `ReSTIR Debug / Audit` ImGui panel, headless debug CLI flags, compact debug metrics JSON, and derived debug views for audit/validation. It does not add new ReSTIR algorithms and keeps default beauty rendering unchanged when debug is disabled.

## Files Changed For R16

- `include/renderer/RenderSettings.h`
- `include/renderer/PerformanceStats.h`
- `include/MetalShaderTypes.h`
- `shaders/common.metal`
- `src/renderer/UniformBuilder.mm`
- `src/renderer/UIOverlay.mm`
- `src/renderer/RenderPassGraph.cpp`
- `src/main_headless.mm`
- `scripts/validate_restir_gui_debug_inspector.py`
- `docs/restir/RESTIR_DEBUG_INSPECTOR.md`
- `docs/restir/FEATURE_GATES.md`
- `docs/restir/ROADMAP_STATUS.md`
- `docs/restir/VALIDATION_COMMANDS.md`
- `docs/restir/validation/R16_GUI_RESTIR_DEBUG_INSPECTOR_VALIDATION.md`

## Commands Run

```sh
python3 -m py_compile scripts/validate_restir_gui_debug_inspector.py
cmake --build build-r13-clean --target PathTracerHeadless
python3 scripts/validate_restir_gui_debug_inspector.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r16_gui_debug_inspector --verbose
python3 scripts/validate_restir_roadmap_full_stack.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r16_full_stack_regression --verbose
python3 scripts/validate_restir_final_full_stack_long_run.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r16_final_full_stack_long_run --verbose
```

Validation was rerun from the clean R15 worktree before commit.

## Build Result

Build directory: `build-r13-clean`

Result:

```text
[100%] Built target PathTracerHeadless
```

## Validator Results

R16 debug inspector:

```text
R16 GUI ReSTIR debug inspector validation PASSED artifactDir=/private/tmp/restir_r16_gui_debug_inspector
```

Full-stack regression:

```text
R6 ReSTIR roadmap full-stack validation PASSED artifactDir=/private/tmp/restir_r16_full_stack_regression modes=9 parityCases=4 validators=12
```

Final long-run regression:

```text
R12 final full-stack longer-run validation PASSED artifactDir=/private/tmp/restir_r16_final_full_stack_long_run controls=5 activeModes=6 parityCases=4 reuseApplied=2555
```

## GUI Panel Summary

The ImGui panel is named `ReSTIR Debug / Audit`. It exposes current mode state, debug view selection, debug pixel controls when `PT_DEBUG_TOOLS` is available, counter summaries, and compact sanity alarms.

## Counters Exposed

- `directLightCandidateCount`
- `regirCellHitCount`
- `regirCellMissCount`
- `giCandidateCount`
- `giAcceptCount`
- `giRejectCount`
- `pathGuidingUsedCount`
- `pathGuidingInvalidCount`
- `pathGuidingMaterialRejectCount`
- `restirPtCandidateCount`
- `restirPtReservoirUpdateCount`
- `restirPtReuseAppliedCount`
- `restirPtRejectedInvalidCount`
- `restirPtDebugRecordCount`
- `svgfActivePixelCount`
- `nanInfPixelCount`
- `rendererNanInfCount`

## Debug Views Exposed

- `beauty`
- `candidate_source_id`
- `reservoir_confidence`
- `regir_cell_id`
- `path_guiding_used_mask`
- `restir_pt_reuse_mask`
- `svgf_variance`
- `nan_inf_mask`

These are derived audit views for headless validation. They do not allocate new reservoir storage or change sampling algorithms.

## Sanity Alarms

- NaN/Inf
- default experimental gates
- RIS-family activity
- path-guiding activity
- ReSTIR PT activity
- ReSTIR PT reuse
- history reset availability
- ReGIR state availability

## Validation Evidence

R16 summary JSON:

- `/private/tmp/restir_r16_gui_debug_inspector/restir_gui_debug_inspector_summary.json`

Key checks:

- Debug-disabled beauty unchanged: `max_abs=0.0`, `rmse=0.0`
- Debug-disabled megakernel/wavefront parity smoke: `max_abs=2.9802322387695312e-08`, `rmse=1.3028156200840783e-09`
- Candidate-source debug run: `directLightCandidateCount=18432`
- Path-guiding debug run: `pathGuidingUsedCount=574`, `pathGuidingInvalidCount=0`
- ReSTIR PT reuse debug run: `restirPtCandidateCount=562`, `restirPtReuseAppliedCount=562`
- NaN/Inf debug run: `nanInfPixelCount=0`, output mask finite

Full-stack summary:

- `/private/tmp/restir_r16_full_stack_regression/restir_r6_full_stack_summary.json`

Final long-run summary:

- `/private/tmp/restir_r16_final_full_stack_long_run/r12_final_full_stack_summary.json`

## Default-State Audit

The full-stack default state remains:

- `direct_light_mode=legacy`
- `restir_gi_mode=off`
- `svgf_denoise_enabled=false`
- pass names: `Megakernel Integrate`, `Path Tracer Present`

R16 debug inspector and debug AOV output are off by default.

## Debug-Gating Audit

`--restirDebugMetricsJson` explicitly enables the debug inspector and counter audit. Non-beauty debug views require explicit `--restirDebugView=<name>`. The default and `--restirDebug=0 --restirDebugView=beauty` paths produced identical beauty output.

## Unavailable Debug Items

- selected-light ID as a true per-pixel image
- temporal rejection image
- spatial reuse-count image
- exact live ReGIR hit/miss hardware counters
- GI accept/reject hardware counters
- persistent GUI history-reset token in the live panel

## Known Limitations

The AOVs are derived audit views, not full internal reservoir dumps. They are suitable for validation plumbing and activity inspection, not for judging beauty quality. Human visual validation of the ImGui layout is still required in the interactive app.

## Unrelated Files Left Untouched

The original worktree contained unrelated modifications and untracked assets/docs. R16 was isolated and committed from the clean R15 worktree so those unrelated files were left untouched.
