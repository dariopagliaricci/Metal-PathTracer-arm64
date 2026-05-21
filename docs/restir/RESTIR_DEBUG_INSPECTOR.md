# ReSTIR Debug Inspector

R16 adds a gated `ReSTIR Debug / Audit` ImGui panel and matching headless validation path for inspecting ReSTIR-related state that is difficult to judge from beauty output alone.

The inspector is disabled by default. With default settings, it does not enable ReSTIR GI, path guiding, ReSTIR PT, SVGF, debug pixel capture, direct-light audit JSON, or debug AOV output.

## Panel Layout

The ImGui panel exposes:

- Current state: execution mode, direct-light mode, RIS candidate count, ReSTIR GI mode, path-guiding mode, ReSTIR PT mode, SVGF state, SPP, seed, audit status, and selected debug view.
- Debug pixel: shared `PT_DEBUG_TOOLS` path-debug controls and the current captured record count.
- Counters: direct-light candidate estimate, path-guiding counters, ReSTIR PT research/reuse counters, NaN/Inf counters, and explicit unavailable fields.
- Sanity alarms: NaN/Inf, default experimental gates, RIS-family activity, path-guiding activity, ReSTIR PT activity, ReSTIR PT reuse, history reset availability, and ReGIR state.

## Headless Flags

```sh
--restirDebug=0|1
--restirDebugView=<beauty|candidate_source_id|reservoir_confidence|regir_cell_id|path_guiding_used_mask|restir_pt_reuse_mask|svgf_variance|nan_inf_mask>
--restirDebugCounters=0|1
--restirDebugMetricsJson=<path>
```

`--restirDebugMetricsJson` enables the debug inspector and counter audit state for validation. `--restirDebugView=beauty` leaves output on the normal beauty path.

## Counter Meanings

- `directLightCandidateCount`: estimated frame-scope direct-light candidates from active RIS/ReSTIR direct-light settings.
- `regirCellHitCount` / `regirCellMissCount`: derived availability markers for world/ReGIR-family modes; exact live cache hit/miss hardware counters are unavailable.
- `giCandidateCount`: derived GI activity marker for the bounded GI prototype; accept/reject hardware counters are unavailable.
- `pathGuidingUsedCount`, `pathGuidingInvalidCount`, `pathGuidingMaterialRejectCount`: existing renderer counters exposed through the PBR metrics path.
- `restirPtCandidateCount`, `restirPtReservoirUpdateCount`, `restirPtReuseAppliedCount`, `restirPtRejectedInvalidCount`, `restirPtDebugRecordCount`: existing R8/R9 ReSTIR PT scaffold counters.
- `svgfActivePixelCount`: derived from resolution when SVGF is enabled.
- `nanInfPixelCount` and `rendererNanInfCount`: image-side and renderer-side invalid-value checks.

## Debug Views

The current headless debug views are derived audit views, not new renderer algorithms:

- `beauty`: normal output.
- `candidate_source_id`: hashed mode/candidate-source visualization.
- `reservoir_confidence`: RIS/ReSTIR direct-light candidate-count confidence proxy.
- `regir_cell_id`: hashed world/ReGIR cell visualization when a world/ReGIR mode is active.
- `path_guiding_used_mask`: binary activity mask from live path-guiding counters.
- `restir_pt_reuse_mask`: binary activity mask from live ReSTIR PT reuse counters.
- `svgf_variance`: local luminance-delta variance proxy when SVGF is active.
- `nan_inf_mask`: red invalid-value mask, black when finite.

## Good Signs

- Debug disabled and `beauty` view match the normal beauty render.
- NaN/Inf alarms stay green on standard fixtures.
- Path-guiding and ReSTIR PT counters become nonzero only when their gated modes are enabled.
- Default experimental gates remain green in a default render.

## Red Flags

- Debug-disabled beauty differs from default beauty.
- ReSTIR PT experimental counters are active while `restirPtMode=off`.
- Path-guiding counters are active while `pathGuidingMode=off`.
- `nan_inf_mask` reports nonzero invalid pixels on standard fixtures.
- History reset behavior does not update after changing rendering modes.

## Known Limitations

The inspector does not add new per-pixel reservoir storage. Selected-light ID, temporal rejection reason, spatial reuse-count, exact ReGIR hit/miss, and GI accept/reject images are unavailable without broader instrumentation. The debug views should be used for plumbing and activity inspection, not beauty-quality judgment.

## Validation

```sh
python3 scripts/validate_restir_gui_debug_inspector.py \
  --binary ./build-restir/PathTracerHeadless \
  --artifact-dir /private/tmp/restir_r16_gui_debug_inspector \
  --verbose
```
