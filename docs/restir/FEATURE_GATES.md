# ReSTIR Feature Gates

This document records the release-candidate feature gates for the current ReSTIR roadmap stack.

## Default State

Default headless rendering must not enable these features unless explicitly requested:

- ReSTIR GI prototype
- SVGF denoise prototype
- path guiding prototype
- ReSTIR PT research scaffold
- ReSTIR PT experimental path reuse
- ReSTIR debug inspector/views
- debug pixel capture
- direct-light audit JSON
- optional pass graph metrics export

## Headless Flags

| Feature | Default | Enable Flag | Notes |
| --- | --- | --- | --- |
| Direct-light baseline | Project default | `--directLightMode=baseline_emissive` | Control mode for baseline emissive direct lighting. |
| RIS | Off unless selected | `--directLightMode=ris` | RIS-family direct-light candidate mode. |
| ReGIR/RIS world reuse | Off unless selected | `--directLightMode=ris_regir` | R1/RIS-family world-space reservoir mode. |
| ReSTIR-DI | Off unless selected | `--directLightMode=restir_di` | Direct-light reservoir reuse mode. |
| DI+ReGIR hybrid | Off unless selected | `--directLightMode=restir_di_regir_hybrid` | R2 candidate composition mode. |
| ReSTIR GI prototype | Off | `--restirGiMode=restir_gi_prototype` | Diffuse-first bounded indirect reuse prototype. |
| SVGF denoise prototype | Off | `--svgfDenoise=1` | Experimental sidecar denoise. |
| Path guiding prototype | Off | `--pathGuidingMode=path_guiding_prototype` | Diffuse-first indirect sampling guide. |
| Path guiding strength | 0.35 | `--pathGuidingStrength=<0..1>` | Applies only when path guiding is enabled. |
| Path guiding cell size | 1.0 | `--pathGuidingCellSize=<float>` | Applies only when path guiding is enabled. |
| ReSTIR PT research | Off | `--restirPtMode=restir_pt_research` | R8 capture/metrics scaffold, no image-affecting reuse. |
| ReSTIR PT experimental reuse | Off | `--restirPtMode=restir_pt_experimental_path_reuse` | R9 bounded diffuse-safe image-affecting experiment. |
| ReSTIR PT max reservoirs | 4096 | `--restirPtMaxReservoirs=<int>` | Applies to R8/R9 PT modes. |
| ReSTIR PT debug | Off | `--restirPtDebug=1` | Research debug counter path. |
| ReSTIR PT reuse strength | 0.25 | `--restirPtReuseStrength=<0..1>` | Applies only to R9 experimental path reuse. |
| ReSTIR debug inspector | Off | `--restirDebug=1` | R16 GUI/headless audit plumbing. |
| ReSTIR debug view | Beauty | `--restirDebugView=<name>` | Derived audit output views; non-beauty views explicitly replace headless output. |
| ReSTIR debug metrics JSON | Off | `--restirDebugMetricsJson=<path>` | Compact R16 counter/sanity JSON. |
| Debug pixel | Off | `--debugPixel=x,y` | Debug/audit only. |
| Direct-light audit JSON | Off | `--directLightAuditJson=<path>` | Explicit audit output only. |

## ImGui Gates

- ReSTIR PT mode selection exposes `off`, `restir_pt_research`, and `restir_pt_experimental_path_reuse`.
- `PT Max Reservoirs` is enabled for both ReSTIR PT modes.
- `PT Research Debug` is enabled only for `restir_pt_research`.
- `PT Reuse Strength` is enabled only for `restir_pt_experimental_path_reuse`.
- Path-guiding strength and cell size are enabled only for `path_guiding_prototype`.
- SVGF controls are independent from path guiding and ReSTIR PT modes.

- ReSTIR debug inspector controls are gated by `Enable Inspector`; derived AOV output is selected explicitly.

## History Reset Settings

The accumulation/history reset audit covers:

- direct-light mode changes
- RIS candidate count changes
- world reuse cell size changes
- ReSTIR GI mode changes
- SVGF enable/disable and parameter changes
- path guiding enable/disable, strength, and cell-size changes
- ReSTIR PT mode changes
- ReSTIR PT max reservoir changes
- ReSTIR PT debug changes
- ReSTIR PT reuse-strength changes
- execution mode changes
- camera/scene changes

## Experimental Boundaries

- R8 is a research scaffold and may produce identical images to baseline while emitting nonzero path-reservoir counters.
- R9 is a bounded experimental path-reuse mode and may affect images only when explicitly selected.
- R9 excludes specular, glass, transmission, delta, unsupported, and invalid-PDF path states.
- No current stage implements production ReSTIR PT, complex shift mapping, reconnection, or neural caching.
