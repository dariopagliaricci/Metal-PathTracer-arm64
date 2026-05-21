# ReSTIR Roadmap Status

Branch: `feature/restir-di-proto`
Remote: `origin/feature/restir-di-proto`
Last audited stage: R16_GUI_RESTIR_DEBUG_INSPECTOR_AND_AOV_PACKAGE

## Closed Stages

| Stage | Status | Commit | Scope |
| --- | --- | --- | --- |
| R1_REGIR_WAVEFRONT_PARITY | PASS | `ddea58afe7f3889426cde102b54db1374c4c5933` | ReGIR/RIS-family wavefront parity and structural validation. |
| R2_RESTIR_DI_REGIR_HYBRID_CANDIDATE_FLOW | PASS | `e4c4242f5560c7641895db884bef57ac93c4ea3c` | ReGIR/world-space reservoir candidate source for ReSTIR-DI. |
| R3_BOUNDED_RESTIR_GI_PROTOTYPE | PASS | `2389998f9e26657bcdd5ba4fcb0401d01814840a` | Gated diffuse-first indirect reuse prototype. |
| R4_SVGF_STYLE_DENOISING_PROTOTYPE | EXPERIMENTAL_PASS | `24f2bb0c5dd6824e9b7b37447b4ebc1f55828eac` | Experimental Metal SVGF-style sidecar denoise. |
| R5_METAL_PASS_GRAPH_WAVEFRONT | PASS | `660b368b07cac16db421e57a7284c1704f4b5515` | Metal pass graph visibility and wavefront pass metrics. |
| R6_RESTIR_ROADMAP_RELEASE_HYGIENE_AND_FULL_REGRESSION | PASS | `24431773049201ceff3345d97be4d98f9eaf014e` | Full-stack release hygiene and regression gate. |
| R7_BOUNDED_PATH_GUIDING_LAYER | PASS | `b440a7b2cea080d1ff790169903525d36a98b8e3` | Disabled-by-default diffuse-first path-guiding prototype. |
| R8_BOUNDED_RESTIR_PT_RESEARCH_SCAFFOLD | PASS | `a9c4165f804cd8ca7758d564e1879d4998bbde67` | Disabled-by-default ReSTIR PT research scaffold and path-reservoir metrics. |
| R9_MINIMAL_RESTIR_PT_EXPERIMENTAL_PATH_REUSE | PASS | `97d304cca659448f93cafd1046e50ae8a8035a1d` | Disabled-by-default bounded diffuse-safe experimental path reuse. |
| R10_RESTIR_PT_PARITY_AND_INTERACTION_HARDENING | PASS | `0f9d787cf4d081d69eb45462b959256fe55b9204` | Interaction hardening across PT research/reuse, GI, guiding, SVGF, and pass graph. |
| R11_ROADMAP_DOCUMENTATION_AND_RELEASE_CANDIDATE_AUDIT | PASS | `b6d690cd65bd2d56b9225e43757d75df5ab1c692` | Documentation, feature gates, command matrix, and release-candidate audit. |
| R12_FINAL_FULL_STACK_LONG_RUN_VALIDATION | PASS | recorded by `docs/restir/validation/R12_FINAL_FULL_STACK_LONG_RUN_VALIDATION.md` | Longer-run final full-stack validation. |
| R13_REMOTE_CLEAN_REPRODUCIBILITY_AND_PR_READINESS_GATE | PASS | recorded by `docs/restir/validation/R13_REMOTE_CLEAN_REPRODUCIBILITY_PR_READINESS.md` | Remote-clean checkout, build, full validation ladder, and PR-readiness gate. |
| R14_BRANCH_INTEGRATION_AND_PR_PACKAGE_GATE | PASS | recorded by `docs/restir/validation/R14_BRANCH_INTEGRATION_PR_PACKAGE.md` | Base-branch diff audit, large-asset audit, PR summary, merge risk register, and final pre-PR validation. |
| R15_VISUAL_REVIEW_PACKAGE_AND_PR_DRAFT_GATE | PASS | `74ce5696c5a98d83aea4bba9e826e49b4eba566f` | Visual review artifacts, PR draft, and human review checklist. |
| R16_GUI_RESTIR_DEBUG_INSPECTOR_AND_AOV_PACKAGE | PASS | recorded by `docs/restir/validation/R16_GUI_RESTIR_DEBUG_INSPECTOR_VALIDATION.md` | Gated ImGui/headless ReSTIR debug inspector, counter audit, and derived debug views. |

## Current Release-Candidate State

- Default rendering keeps ReSTIR GI, SVGF, path guiding, ReSTIR PT research, ReSTIR PT experimental path reuse, ReSTIR debug inspector/views, debug pixel, direct-light audit JSON, and optional pass graph metrics disabled.
- Prototype features are headless-controllable and validated under Metal.
- Megakernel and wavefront parity is validated for core and prototype fixtures by stage validators.
- R8 and R9 are explicitly research/prototype modes, not production ReSTIR PT.
- Neural caching was not implemented.
- Canonical scene assets and canonical camera framing were not mutated by the roadmap closures.
- The remote-clean R13 gate validates that the pushed branch contains the canonical Living Room assets required by the ReGIR validators and can build without relying on ignored local OIDN dylibs.

## Remaining Ladder

The current autonomous roadmap ladder through R12 is closed. Future stages should start from a new explicit roadmap scope.

## Validation Reports

- `docs/restir/validation/R1_REGIR_WAVEFRONT_PARITY_VALIDATION.md`
- `docs/restir/validation/R2_RESTIR_DI_REGIR_HYBRID_VALIDATION.md`
- `docs/restir/validation/R3_RESTIR_GI_PROTOTYPE_VALIDATION.md`
- `docs/restir/validation/R4_SVGF_DENOISING_PROTOTYPE_VALIDATION.md`
- `docs/restir/validation/R5_METAL_PASS_GRAPH_VALIDATION.md`
- `docs/restir/validation/R6_RESTIR_ROADMAP_RELEASE_HYGIENE_VALIDATION.md`
- `docs/restir/validation/R7_PATH_GUIDING_VALIDATION.md`
- `docs/restir/validation/R8_RESTIR_PT_RESEARCH_SCAFFOLD_VALIDATION.md`
- `docs/restir/validation/R9_RESTIR_PT_EXPERIMENTAL_PATH_REUSE_VALIDATION.md`
- `docs/restir/validation/R10_RESTIR_PT_PARITY_HARDENING_VALIDATION.md`
- `docs/restir/validation/R11_ROADMAP_DOCUMENTATION_RELEASE_AUDIT.md`
- `docs/restir/validation/R12_FINAL_FULL_STACK_LONG_RUN_VALIDATION.md`
- `docs/restir/validation/R13_REMOTE_CLEAN_REPRODUCIBILITY_PR_READINESS.md`
- `docs/restir/validation/R14_BRANCH_INTEGRATION_PR_PACKAGE.md`
- `docs/restir/validation/R15_VISUAL_REVIEW_PACKAGE_PR_DRAFT.md`

- `docs/restir/validation/R16_GUI_RESTIR_DEBUG_INSPECTOR_VALIDATION.md`
