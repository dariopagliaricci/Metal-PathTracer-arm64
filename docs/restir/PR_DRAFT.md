# Add validated ReSTIR/ReGIR roadmap stack for Metal PathTracer

## Summary

This PR adds the completed ReSTIR roadmap branch for the Metal PathTracer. It includes ReGIR/RIS-family parity, ReSTIR-DI + ReGIR hybrid candidate flow, bounded ReSTIR GI, SVGF sidecar denoise, Metal pass graph/wavefront validation, path guiding, ReSTIR PT research scaffolding, experimental diffuse-safe path reuse, and release/PR-readiness validation.

This does not claim production ReSTIR PT. R8/R9/R10 are research/experimental scaffold and reuse stages. R4 SVGF remains `EXPERIMENTAL_PASS`. Experimental systems are disabled by default.

## Roadmap Stages

| Stage | Status | Scope |
| --- | --- | --- |
| R1_REGIR_WAVEFRONT_PARITY | PASS | ReGIR/RIS wavefront parity. |
| R2_RESTIR_DI_REGIR_HYBRID_CANDIDATE_FLOW | PASS | ReSTIR-DI + ReGIR candidate flow. |
| R3_BOUNDED_RESTIR_GI_PROTOTYPE | PASS | Diffuse-first ReSTIR GI prototype. |
| R4_SVGF_STYLE_DENOISING_PROTOTYPE | EXPERIMENTAL_PASS | SVGF-style sidecar denoise. |
| R5_METAL_PASS_GRAPH_WAVEFRONT | PASS | Metal pass graph and wavefront metrics. |
| R6_RESTIR_ROADMAP_RELEASE_HYGIENE_AND_FULL_REGRESSION | PASS | Full-stack regression gate. |
| R7_BOUNDED_PATH_GUIDING_LAYER | PASS | Diffuse-first path-guiding prototype. |
| R8_BOUNDED_RESTIR_PT_RESEARCH_SCAFFOLD | PASS | ReSTIR PT research scaffold. |
| R9_MINIMAL_RESTIR_PT_EXPERIMENTAL_PATH_REUSE | PASS | Diffuse-safe experimental path reuse. |
| R10_RESTIR_PT_PARITY_AND_INTERACTION_HARDENING | PASS | Interaction hardening. |
| R11_ROADMAP_DOCUMENTATION_AND_RELEASE_CANDIDATE_AUDIT | PASS | Roadmap documentation. |
| R12_FINAL_FULL_STACK_LONG_RUN_VALIDATION | PASS | Longer-run full-stack validation. |
| R13_REMOTE_CLEAN_REPRODUCIBILITY_AND_PR_READINESS_GATE | PASS | Remote-clean reproducibility. |
| R14_BRANCH_INTEGRATION_AND_PR_PACKAGE_GATE | PASS | PR package and merge-risk audit. |
| R15_VISUAL_REVIEW_PACKAGE_AND_PR_DRAFT_GATE | PASS | Visual review package and PR draft. |

## Key Features

- Direct-light controls: `baseline_emissive`, `ris`, `ris_regir`, `restir_di`, `restir_di_regir_hybrid`.
- Indirect/prototype controls: `restir_gi_prototype`, `path_guiding_prototype`, `restir_pt_research`, `restir_pt_experimental_path_reuse`.
- Sidecar denoise: `svgfDenoise`.
- Metal pass graph and wavefront parity instrumentation.
- Headless validation scripts and validation reports from R1 through R15.

## Validation Matrix

Final R15 validation used clean worktree `/Users/dariopagliaricci/Metal-PathTracer-restir-r13-clean` and binary `./build-r13-clean/PathTracerHeadless`.

| Validation | Result | Artifact Directory |
| --- | --- | --- |
| Build `cmake --build build-r13-clean --target PathTracerHeadless` | PASS | `build-r13-clean` |
| Remote-clean reproducibility | PASS | `/private/tmp/restir_r15_remote_clean_reproducibility` |
| Roadmap full-stack regression | PASS | `/private/tmp/restir_r15_full_stack_regression` |
| Visual review package | PASS | `/private/tmp/restir_r15_visual_review` |

Representative R15 full-stack parity:

| Fixture | PSNR | SSIM | Log RMSE | RGB RMSE |
| --- | ---: | ---: | ---: | ---: |
| `living_room_clean_regir` | `32.760615028` | `0.997474310` | `0.023012789` | `0.026211163` |
| `living_room_practicals_regir` | `30.564209639` | `0.995835369` | `0.029633948` | `0.041698471` |
| `r3_diffuse_indirect_gi` | `inf` | `1.000000000` | `0.000000000` | `0.000000000` |
| `r5_pass_graph_svgf` | `inf` | `1.000000000` | `0.000000000` | `0.000000000` |

NaN/Inf counts were zero in required R15 validation and review renders.

## Visual Artifacts

Visual review artifact directory:

`/private/tmp/restir_r15_visual_review`

Key artifacts:

- `matrix_baseline_emissive.pfm`
- `matrix_ris_regir.pfm`
- `matrix_restir_di_regir_hybrid.pfm`
- `matrix_restir_gi_prototype.pfm`
- `matrix_svgfDenoise.pfm`
- `parity_living_room_clean_regir_megakernel.pfm`
- `parity_living_room_practicals_regir_megakernel.pfm`
- `r3_path_guiding.pfm`
- `r3_restir_pt_research.pfm`
- `r3_restir_pt_experimental_reuse.pfm`

No contact sheet is included. The repository has `scripts/compare_exr.py` but no contact-sheet generator; review uses documented PFM paths and metrics.

## Large Asset Warning

`assets/living_room/living_room.obj` is tracked in Git and is about `54 MB` on disk. It triggered a GitHub warning at `52.76 MB`. It is required by remote-clean Living Room/ReGIR validation.

No history rewrite, force-push, deletion, or Git LFS migration was performed. Recommended human decision: keep for this PR, or plan a future Git LFS / smaller-fixture migration.

## Optional OIDN Note

OIDN native dylibs are optional in clean builds. If vendored OIDN/TBB dylibs are absent, the renderer builds with OIDN disabled and `--denoise` reports unavailable. ReSTIR/SVGF roadmap validators do not depend on OIDN.

## Default States

Disabled by default unless explicitly selected:

- ReSTIR GI prototype
- SVGF denoise prototype
- path guiding prototype
- ReSTIR PT research scaffold
- ReSTIR PT experimental path reuse
- debug pixel capture
- direct-light audit JSON
- optional pass graph metrics export

## Known Limitations

- R4 SVGF is experimental.
- R8/R9/R10 are research/experimental ReSTIR PT work, not production ReSTIR PT.
- ReSTIR GI and path guiding are diffuse-first bounded prototypes.
- Metal headless validation is the primary validation target for this branch.
- Subjective visual quality still needs human review before merge if visual approval is a release criterion.

## Reviewer Checklist

- Confirm the large asset policy for `assets/living_room/living_room.obj`.
- Inspect visual artifacts in `/private/tmp/restir_r15_visual_review`.
- Confirm default-state and feature-gate expectations.
- Confirm R4/R8/R9 experimental wording is acceptable.
- Confirm no production ReSTIR PT claim is made.
- Decide whether to open/merge the PR or request a future asset/LFS follow-up.
