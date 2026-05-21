# ReSTIR Roadmap PR Summary

Branch: `feature/restir-di-proto`
Base branch used for comparison: `origin/main`
Latest packaged stage: `R14_BRANCH_INTEGRATION_AND_PR_PACKAGE_GATE`

## Roadmap Stages

| Stage | Status | Scope |
| --- | --- | --- |
| R1_REGIR_WAVEFRONT_PARITY | PASS | ReGIR/RIS-family wavefront parity. |
| R2_RESTIR_DI_REGIR_HYBRID_CANDIDATE_FLOW | PASS | ReGIR/world-space candidate source for ReSTIR-DI. |
| R3_BOUNDED_RESTIR_GI_PROTOTYPE | PASS | Disabled-by-default diffuse-first indirect reuse prototype. |
| R4_SVGF_STYLE_DENOISING_PROTOTYPE | EXPERIMENTAL_PASS | Experimental SVGF-style sidecar denoise. |
| R5_METAL_PASS_GRAPH_WAVEFRONT | PASS | Metal pass graph visibility and wavefront pass metrics. |
| R6_RESTIR_ROADMAP_RELEASE_HYGIENE_AND_FULL_REGRESSION | PASS | Full-stack release hygiene and regression gate. |
| R7_BOUNDED_PATH_GUIDING_LAYER | PASS | Disabled-by-default diffuse-first path-guiding prototype. |
| R8_BOUNDED_RESTIR_PT_RESEARCH_SCAFFOLD | PASS | Disabled-by-default ReSTIR PT research scaffold. |
| R9_MINIMAL_RESTIR_PT_EXPERIMENTAL_PATH_REUSE | PASS | Disabled-by-default bounded diffuse-safe path reuse experiment. |
| R10_RESTIR_PT_PARITY_AND_INTERACTION_HARDENING | PASS | Interaction hardening across PT reuse, GI, guiding, SVGF, and pass graph. |
| R11_ROADMAP_DOCUMENTATION_AND_RELEASE_CANDIDATE_AUDIT | PASS | Roadmap docs, gate docs, validation command matrix. |
| R12_FINAL_FULL_STACK_LONG_RUN_VALIDATION | PASS | Longer-run final full-stack validation. |
| R13_REMOTE_CLEAN_REPRODUCIBILITY_AND_PR_READINESS_GATE | PASS | Remote-clean checkout/build/validation reproducibility. |
| R14_BRANCH_INTEGRATION_AND_PR_PACKAGE_GATE | PASS | Branch diff audit, large-asset audit, PR package, merge risk register. |

## Major Feature Groups

- ReSTIR-DI, RIS/ReGIR, and DI+ReGIR hybrid direct-light paths.
- Bounded ReSTIR GI prototype, path-guiding prototype, and ReSTIR PT research/reuse prototypes, all disabled by default.
- SVGF sidecar denoise, marked experimental and disabled by default.
- Metal pass graph metrics and wavefront parity infrastructure.
- Headless validators and documentation for clean reproducibility.

## Validation Summary

Final R14 validation from `/Users/dariopagliaricci/Metal-PathTracer-restir-r13-clean`:

- Build: `cmake --build build-r13-clean --target PathTracerHeadless` PASS.
- Remote-clean validator: PASS, artifact dir `/private/tmp/restir_r14_remote_clean_reproducibility`.
- Full-stack validator: PASS, artifact dir `/private/tmp/restir_r14_full_stack_regression`.
- Final long-run validator: PASS, artifact dir `/private/tmp/restir_r14_final_full_stack_long_run`.

Representative R14 long-run parity:

| Fixture | PSNR | SSIM | Log RMSE | RGB RMSE |
| --- | ---: | ---: | ---: | ---: |
| `living_room_clean_ris_regir_reuse` | `32.104748951` | `0.996934569` | `0.024817758` | `0.036904589` |
| `living_room_practicals_ris_regir_reuse` | `31.052707441` | `0.996270971` | `0.028013323` | `0.040782348` |
| `r3_diffuse_full_stack` | `43.210455255` | `0.998055169` | `0.006909987` | `0.010905338` |
| `r3_diffuse_reuse` | `inf` | `1.000000000` | `0.000000000` | `0.000000000` |

NaN/Inf counts were zero in required R14 validators.

## Default State and Gates

Default rendering keeps these disabled unless explicitly selected:

- `restir_gi_prototype`
- `path_guiding_prototype`
- `restir_pt_research`
- `restir_pt_experimental_path_reuse`
- `svgfDenoise`
- `debugPixel`
- `directLightAuditJson`
- optional pass graph metrics export

## Large Asset Warning

`assets/living_room/living_room.obj` is tracked in Git and is approximately `54M` on disk (`52.76 MB` GitHub push warning). It is above GitHub's recommended comfort range but below observed hard rejection thresholds. It is required by the remote-clean ReGIR/Living Room validation path.

Recommended human decision: keep as-is for this PR, or plan a future history-managed Git LFS/asset-fixture migration. Do not force-push or rewrite history as part of this PR package.

## Known Limitations

- R4 SVGF remains `EXPERIMENTAL_PASS`.
- R8/R9 are research/prototype ReSTIR PT modes, not production ReSTIR PT.
- Path guiding and ReSTIR GI are bounded diffuse-first prototypes.
- OIDN native dylibs are optional in clean builds; `--denoise` reports unavailable when vendored dylibs are absent.
- Subjective visual quality review is still recommended before merge.

## Human Review Checklist

- Review the large asset decision for `assets/living_room/living_room.obj`.
- Review feature gates and default-state behavior.
- Review Metal/wavefront parity validation evidence.
- Review experimental status wording for SVGF and ReSTIR PT prototypes.
- Optionally inspect representative rendered artifacts before merge.
