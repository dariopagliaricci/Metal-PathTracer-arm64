# R14 Branch Integration and PR Package Validation

Final status: PASS

Stage: `R14_BRANCH_INTEGRATION_AND_PR_PACKAGE_GATE`

Clean worktree path: `/Users/dariopagliaricci/Metal-PathTracer-restir-r13-clean`
Branch: `restir-r13-clean`
Remote branch: `origin/feature/restir-di-proto`
Base branch used: `origin/main`
Parent R13 commit: `e7985530b5da768a2558e094e4ad01a53ff96ed3`
Final R14 commit hash: recorded by the pushed R14 commit for this report; exact hash is reported by `git log -1 origin/feature/restir-di-proto` after push.

R14 is an integration/package gate only. It did not implement new rendering algorithms, expand experimental systems, rewrite history, force-push, or modify canonical scene camera framing.

## Files Changed

- `docs/restir/PR_SUMMARY.md`
- `docs/restir/MERGE_RISK_REGISTER.md`
- `docs/restir/validation/R14_BRANCH_INTEGRATION_PR_PACKAGE.md`
- `docs/restir/ROADMAP_STATUS.md`
- `docs/restir/VALIDATION_COMMANDS.md`

## Exact Commands Run

```sh
cd /Users/dariopagliaricci/Metal-PathTracer-restir-r13-clean
git fetch origin
git status --short
git log -1 --oneline
git branch --show-current
git remote show origin
git branch -a
git symbolic-ref refs/remotes/origin/HEAD || true
git log --oneline --decorate --graph origin/main..HEAD
git diff --stat origin/main..HEAD
git diff --name-status origin/main..HEAD
git diff --shortstat origin/main..HEAD
find assets -type f -size +25M -print -exec ls -lh {} \;
git ls-files assets/living_room/living_room.obj || true
git check-attr -a -- assets/living_room/living_room.obj || true
test -f .gitattributes && cat .gitattributes || true
git lfs ls-files || true
git lfs track || true
cmake --build build-r13-clean --target PathTracerHeadless
python3 scripts/validate_restir_remote_clean_reproducibility.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r14_remote_clean_reproducibility --verbose
python3 scripts/validate_restir_roadmap_full_stack.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r14_full_stack_regression --verbose
python3 scripts/validate_restir_final_full_stack_long_run.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r14_final_full_stack_long_run --verbose
```

## Build Result

Build directory: `build-r13-clean`

Result: PASS

`cmake --build build-r13-clean --target PathTracerHeadless` completed and rebuilt shader-copy outputs. CMake reported the expected OIDN warning because vendored OIDN/TBB dylibs are absent in the clean worktree; the build completed with OIDN disabled.

## Validators Run

| Command | Result | Artifact Directory |
| --- | --- | --- |
| `python3 scripts/validate_restir_remote_clean_reproducibility.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r14_remote_clean_reproducibility --verbose` | PASS | `/private/tmp/restir_r14_remote_clean_reproducibility` |
| `python3 scripts/validate_restir_roadmap_full_stack.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r14_full_stack_regression --verbose` | PASS | `/private/tmp/restir_r14_full_stack_regression` |
| `python3 scripts/validate_restir_final_full_stack_long_run.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r14_final_full_stack_long_run --verbose` | PASS | `/private/tmp/restir_r14_final_full_stack_long_run` |

## Branch Diff Summary

Compared branch: `HEAD`
Base branch: `origin/main`

- Commit range: `origin/main..HEAD`
- Commits in range: R1 through R13 closure commits plus R14 package commit after push.
- Pre-R14 diff shortstat against `origin/main`: `181 files changed, 1940965 insertions(+), 643 deletions(-)`.

Changed files by category:

| Category | Count |
| --- | ---: |
| shaders | `2` |
| renderer source | `21` |
| headless CLI | `3` |
| settings/uniforms | `1` |
| validation scripts | `38` |
| docs | `67` |
| test scenes | `10` |
| assets | `37` |
| build system | `1` |
| other | `1` |

## Roadmap Stage Table

| Stage | Status |
| --- | --- |
| R1_REGIR_WAVEFRONT_PARITY | PASS |
| R2_RESTIR_DI_REGIR_HYBRID_CANDIDATE_FLOW | PASS |
| R3_BOUNDED_RESTIR_GI_PROTOTYPE | PASS |
| R4_SVGF_STYLE_DENOISING_PROTOTYPE | EXPERIMENTAL_PASS |
| R5_METAL_PASS_GRAPH_WAVEFRONT | PASS |
| R6_RESTIR_ROADMAP_RELEASE_HYGIENE_AND_FULL_REGRESSION | PASS |
| R7_BOUNDED_PATH_GUIDING_LAYER | PASS |
| R8_BOUNDED_RESTIR_PT_RESEARCH_SCAFFOLD | PASS |
| R9_MINIMAL_RESTIR_PT_EXPERIMENTAL_PATH_REUSE | PASS |
| R10_RESTIR_PT_PARITY_AND_INTERACTION_HARDENING | PASS |
| R11_ROADMAP_DOCUMENTATION_AND_RELEASE_CANDIDATE_AUDIT | PASS |
| R12_FINAL_FULL_STACK_LONG_RUN_VALIDATION | PASS |
| R13_REMOTE_CLEAN_REPRODUCIBILITY_AND_PR_READINESS_GATE | PASS |
| R14_BRANCH_INTEGRATION_AND_PR_PACKAGE_GATE | PASS |

## Final Validation Summary

Remote-clean reproducibility:

- Status: PASS
- Default smoke output: `/private/tmp/restir_r14_remote_clean_reproducibility/r13_default_smoke.pfm`
- Default smoke metrics: `/private/tmp/restir_r14_remote_clean_reproducibility/r13_default_smoke_metrics.json`
- Mean luminance: `0.031309634871571325`
- Max component: `3.5999999046325684`
- NaN/Inf pixels/components/throughput/radiance: `0`

Full-stack regression:

- Status: PASS
- Modes: `9`
- Parity cases: `4`
- Prior-stage validators orchestrated: `12`

Final long-run:

- Status: PASS
- Backend: Metal
- Software ray tracing: enabled
- Resolution: `128x72`
- SPP: `4`
- Seed: `42`
- Controls: `5`
- Active modes: `6`
- Parity cases: `4`
- Reuse applied: `2555`

## Parity Metrics

R14 final long-run:

| Fixture | PSNR | SSIM | Log RMSE | RGB RMSE |
| --- | ---: | ---: | ---: | ---: |
| `living_room_clean_ris_regir_reuse` | `32.104748951` | `0.996934569` | `0.024817758` | `0.036904589` |
| `living_room_practicals_ris_regir_reuse` | `31.052707441` | `0.996270971` | `0.028013323` | `0.040782348` |
| `r3_diffuse_full_stack` | `43.210455255` | `0.998055169` | `0.006909987` | `0.010905338` |
| `r3_diffuse_reuse` | `inf` | `1.000000000` | `0.000000000` | `0.000000000` |

R14 full-stack regression:

| Fixture | PSNR | SSIM | Log RMSE | RGB RMSE |
| --- | ---: | ---: | ---: | ---: |
| `living_room_clean_regir` | `32.760615028` | `0.997474310` | `0.023012789` | `0.026211163` |
| `living_room_practicals_regir` | `30.564209639` | `0.995835369` | `0.029633948` | `0.041698471` |
| `r3_diffuse_indirect_gi` | `inf` | `1.000000000` | `0.000000000` | `0.000000000` |
| `r5_pass_graph_svgf` | `inf` | `1.000000000` | `0.000000000` | `0.000000000` |

## Active-Mode Counters

From R14 final long-run:

- Path guiding used count: `2560`
- ReSTIR PT research candidates: `2558`
- ReSTIR PT research reservoir updates: `2558`
- ReSTIR PT research rejected invalid: `0`
- ReSTIR PT research rejected unsupported: `3334`
- ReSTIR PT experimental reuse candidates: `2555`
- ReSTIR PT experimental reuse applied: `2555`
- ReSTIR PT experimental reuse fallback: `3334`
- Full-stack sidecar reuse applied: `2572`
- Full-stack sidecar guiding used: `2572`

## Default-State Audit

R14 remote-clean default smoke verified:

- `restir_gi_mode=off`
- `path_guiding_mode=off`
- `restir_pt_mode=off`
- path-guiding counters zero by default
- ReSTIR PT counters zero by default
- NaN/Inf counts zero

Result: PASS

## Gating Audit

Gated features remain explicit:

- ReSTIR GI: `--restirGiMode=restir_gi_prototype`
- Path guiding: `--pathGuidingMode=path_guiding_prototype`
- ReSTIR PT research/reuse: `--restirPtMode=restir_pt_research` or `--restirPtMode=restir_pt_experimental_path_reuse`
- SVGF: `--svgfDenoise=1`
- Debug pixel: `--debugPixel=x,y`
- Direct-light audit: `--directLightAuditJson=<path>`

Result: PASS

## Documentation Audit

Created:

- `docs/restir/PR_SUMMARY.md`
- `docs/restir/MERGE_RISK_REGISTER.md`
- `docs/restir/validation/R14_BRANCH_INTEGRATION_PR_PACKAGE.md`

Updated:

- `docs/restir/ROADMAP_STATUS.md`
- `docs/restir/VALIDATION_COMMANDS.md`

Result: PASS

## Large Asset Audit

Files over 25 MB:

- `assets/canonical/living_room/living_room.glb`: `33M`
- `assets/HDR/sunset.hdr`: `26M`
- `assets/living_room/living_room.obj`: `54M` on disk; GitHub push warning reported `52.76 MB`

Evidence:

- `git ls-files assets/living_room/living_room.obj` returns `assets/living_room/living_room.obj`.
- `git check-attr -a -- assets/living_room/living_room.obj` reports `text: auto` and `eol: lf`.
- `.gitattributes` marks image/HDR/exr/pfm/ppm/png/jpg/jpeg/zip/dylib/tbb as binary but does not route OBJ through LFS.
- `git lfs` is not installed in this environment.
- R13/R14 remote-clean validation depends on the canonical Living Room asset path.

Decision for R14: leave the asset in Git, document the merge risk, and do not rewrite history or force-push. Recommended human decision: keep as-is for this PR, or plan a future history-managed Git LFS or smaller fixture migration.

## PR-Readiness Audit

- Base branch identified: `origin/main`
- Branch diff audited.
- PR summary created.
- Merge risk register created.
- Large asset risk documented.
- Final validation passed from a clean worktree.
- No PR was opened by R14.

Result: PASS

## Known Limitations

- R4 SVGF remains `EXPERIMENTAL_PASS`.
- R8/R9 are research/prototype ReSTIR PT modes, not production ReSTIR PT.
- Path guiding and ReSTIR GI are bounded diffuse-first prototypes.
- OIDN native dylibs are optional and absent from the clean checkout; `--denoise` reports unavailable when OIDN is not present.
- Human visual validation is recommended before merge if visual quality approval is required.
- Large asset handling requires a human merge-policy decision if repository size is a concern.

## Unrelated Files

The original working tree at `/Users/dariopagliaricci/Metal-PathTracer-feature-restir-di-proto` still contains unrelated dirty/untracked files. R14 was performed from the clean worktree and did not modify, stage, reset, delete, or stash those files.
