# ReSTIR Roadmap Merge Risk Register

Branch: `feature/restir-di-proto`
Base branch used for R14 audit: `origin/main`

| Risk | Impact | Mitigation | Merge Recommendation |
| --- | --- | --- | --- |
| Large tracked asset: `assets/living_room/living_room.obj` is about `54M` on disk and triggered a GitHub warning at `52.76 MB`. | Slower clone/fetch and repository growth. | R13 proved it is required for remote-clean Living Room/ReGIR validation. No history rewrite or force-push was performed. | Accept for this branch if validation self-containment is preferred; plan future Git LFS or smaller-fixture migration in a separate history-managed effort. |
| R4 SVGF is `EXPERIMENTAL_PASS`, not production denoise. | Reviewers may overinterpret denoise readiness. | Documentation and feature gates label it as an explicit sidecar prototype, disabled by default. | Merge only as experimental gated functionality. |
| R8/R9 ReSTIR PT modes are research/prototype paths. | Risk of being mistaken for production ReSTIR PT. | Reports and docs state they are bounded research/reuse modes and disabled by default. | Merge with prototype wording intact; do not advertise production ReSTIR PT. |
| Path guiding prototype is diffuse-first and bounded. | Limited applicability to specular/glass/complex paths. | Validators prove active mode and stability; default remains off. | Merge as gated prototype only. |
| ReSTIR GI prototype is diffuse-first and conservative. | Limited GI correctness/coverage. | Specular/glass/transmission are excluded or fall back; default remains off. | Merge as gated prototype only. |
| Metal-only validation assumptions. | Non-Metal platforms may need separate validation. | Roadmap validators target Metal headless with software ray tracing where specified. | Merge for Metal path; require separate non-Metal validation before claiming broader support. |
| OIDN native dylibs are optional and absent from remote-clean checkout. | `--denoise` can be unavailable in clean builds. | CMake now disables OIDN cleanly when dylibs are absent; ReSTIR/SVGF validators do not depend on OIDN. | Merge with OIDN optionality documented. |
| Performance unknowns outside validation resolutions. | Higher resolutions/SPP may expose cost regressions. | R12/R14 longer-run validators use core modes and Living Room fixtures; pass graph metrics are available. | Merge after reviewer accepts prototype performance scope; do not claim production performance. |
| Subjective visual quality not human-approved. | Quantitative metrics may miss visual artifacts. | Reports recommend optional visual review before merge. | Run human visual spot-checks before final merge if visual quality is a release criterion. |
| Canonical scene/asset status. | Accidental asset changes could affect reproducibility. | R13/R14 tracked the required canonical assets and did not mutate camera framing. | Merge with asset decision acknowledged. |
| Original worktree contained unrelated dirty/untracked files. | Risk of accidental unrelated commits. | R13/R14 were performed from a clean worktree and staged explicit docs only for R14. | Merge package is isolated from original local dirt. |
