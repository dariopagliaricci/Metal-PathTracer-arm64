# R15 Visual Review Package and PR Draft Validation

Final status: PASS

Stage: `R15_VISUAL_REVIEW_PACKAGE_AND_PR_DRAFT_GATE`

Clean worktree path: `/Users/dariopagliaricci/Metal-PathTracer-restir-r13-clean`
Branch: `restir-r13-clean`
Remote branch: `origin/feature/restir-di-proto`
Base branch: `origin/main`
Parent R14 commit: `2cec17325d0c1e15c978be690acecba984d4398d`
Final R15 commit hash: recorded by the pushed R15 commit for this report; exact hash is reported by `git log -1 origin/feature/restir-di-proto` after push.

R15 is a human-review packaging gate. It did not implement renderer features, modify shaders, modify renderer code, modify canonical assets, rewrite history, force-push, or migrate assets to Git LFS.

## Files Changed

- `docs/restir/PR_DRAFT.md`
- `docs/restir/VISUAL_REVIEW_CHECKLIST.md`
- `docs/restir/validation/R15_VISUAL_REVIEW_PACKAGE_PR_DRAFT.md`

## Exact Commands Run

```sh
cd /Users/dariopagliaricci/Metal-PathTracer-restir-r13-clean
git fetch origin
git status --short
git log -1 --oneline
git branch --show-current
git log -1 --oneline origin/feature/restir-di-proto
cmake --build build-r13-clean --target PathTracerHeadless
find scripts tests docs -iname "*contact*" -o -iname "*montage*" -o -iname "*compare*" -o -iname "*image*"
python3 scripts/validate_restir_remote_clean_reproducibility.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r15_remote_clean_reproducibility --verbose
python3 scripts/validate_restir_roadmap_full_stack.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r15_full_stack_regression --verbose
mkdir -p /private/tmp/restir_r15_visual_review
cp /private/tmp/restir_r15_full_stack_regression/matrix_baseline_emissive.pfm /private/tmp/restir_r15_visual_review/
cp /private/tmp/restir_r15_full_stack_regression/matrix_ris_regir.pfm /private/tmp/restir_r15_visual_review/
cp /private/tmp/restir_r15_full_stack_regression/matrix_restir_di_regir_hybrid.pfm /private/tmp/restir_r15_visual_review/
cp /private/tmp/restir_r15_full_stack_regression/matrix_restir_gi_prototype.pfm /private/tmp/restir_r15_visual_review/
cp /private/tmp/restir_r15_full_stack_regression/matrix_svgfDenoise.pfm /private/tmp/restir_r15_visual_review/
cp /private/tmp/restir_r15_full_stack_regression/parity_living_room_clean_regir_megakernel.pfm /private/tmp/restir_r15_visual_review/
cp /private/tmp/restir_r15_full_stack_regression/parity_living_room_practicals_regir_megakernel.pfm /private/tmp/restir_r15_visual_review/
cp /private/tmp/restir_r15_full_stack_regression/restir_r6_full_stack_summary.json /private/tmp/restir_r15_visual_review/
./build-r13-clean/PathTracerHeadless --scene=tests/scenes/restir/r3/r3_diffuse_indirect.scene --backend=metal --enableSoftwareRayTracing=1 --width=128 --height=72 --sppTotal=2 --seed=42 --directLightMode=legacy --restirGiMode=off --pathGuidingMode=path_guiding_prototype --pathGuidingStrength=0.5 --pathGuidingCellSize=1.0 --restirPtMode=off --svgfDenoise=0 --pbrMetricsJson=/private/tmp/restir_r15_visual_review/r3_path_guiding_metrics.json --output=/private/tmp/restir_r15_visual_review/r3_path_guiding.pfm --format=pfm
./build-r13-clean/PathTracerHeadless --scene=tests/scenes/restir/r3/r3_diffuse_indirect.scene --backend=metal --enableSoftwareRayTracing=1 --width=128 --height=72 --sppTotal=2 --seed=42 --directLightMode=legacy --restirGiMode=off --pathGuidingMode=off --restirPtMode=restir_pt_research --restirPtMaxReservoirs=4096 --restirPtDebug=1 --svgfDenoise=0 --pbrMetricsJson=/private/tmp/restir_r15_visual_review/r3_restir_pt_research_metrics.json --output=/private/tmp/restir_r15_visual_review/r3_restir_pt_research.pfm --format=pfm
./build-r13-clean/PathTracerHeadless --scene=tests/scenes/restir/r3/r3_diffuse_indirect.scene --backend=metal --enableSoftwareRayTracing=1 --width=128 --height=72 --sppTotal=2 --seed=42 --directLightMode=legacy --restirGiMode=off --pathGuidingMode=off --restirPtMode=restir_pt_experimental_path_reuse --restirPtMaxReservoirs=4096 --restirPtReuseStrength=0.25 --svgfDenoise=0 --pbrMetricsJson=/private/tmp/restir_r15_visual_review/r3_restir_pt_experimental_reuse_metrics.json --output=/private/tmp/restir_r15_visual_review/r3_restir_pt_experimental_reuse.pfm --format=pfm
```

## Build Result

Build directory: `build-r13-clean`

Result: PASS

`cmake --build build-r13-clean --target PathTracerHeadless` completed and copied shaders through the existing dependency chain.

## Validators Run

| Validator | Result | Artifact Directory |
| --- | --- | --- |
| `validate_restir_remote_clean_reproducibility.py` | PASS | `/private/tmp/restir_r15_remote_clean_reproducibility` |
| `validate_restir_roadmap_full_stack.py` | PASS | `/private/tmp/restir_r15_full_stack_regression` |

## Visual Artifact Package

Artifact directory: `/private/tmp/restir_r15_visual_review`

Contact sheet: not created. The repository has `scripts/compare_exr.py` but no contact-sheet tooling; PFM artifact paths are documented for review.

Visual artifact paths:

- `/private/tmp/restir_r15_visual_review/matrix_baseline_emissive.pfm`
- `/private/tmp/restir_r15_visual_review/matrix_ris_regir.pfm`
- `/private/tmp/restir_r15_visual_review/matrix_restir_di_regir_hybrid.pfm`
- `/private/tmp/restir_r15_visual_review/matrix_restir_gi_prototype.pfm`
- `/private/tmp/restir_r15_visual_review/matrix_svgfDenoise.pfm`
- `/private/tmp/restir_r15_visual_review/parity_living_room_clean_regir_megakernel.pfm`
- `/private/tmp/restir_r15_visual_review/parity_living_room_practicals_regir_megakernel.pfm`
- `/private/tmp/restir_r15_visual_review/r3_path_guiding.pfm`
- `/private/tmp/restir_r15_visual_review/r3_restir_pt_research.pfm`
- `/private/tmp/restir_r15_visual_review/r3_restir_pt_experimental_reuse.pfm`

Visual artifact metrics:

| Artifact | Mean Luma | Max Luma | NaN/Inf Pixels | NaN/Inf Components | Active Counters |
| --- | ---: | ---: | ---: | ---: | --- |
| `r3_path_guiding.pfm` | `0.034716343` | `2.880000114` | `0` | `0` | path guiding used `2458` |
| `r3_restir_pt_research.pfm` | `0.034976208` | `2.880000114` | `0` | `0` | PT candidates/updates `2433/2433`, debug records `2433` |
| `r3_restir_pt_experimental_reuse.pfm` | `0.036264541` | `2.880000114` | `0` | `0` | PT candidates/updates/reuse applied `2442/2442/2442` |

## Validation Summary

Remote-clean reproducibility:

- Status: PASS
- Default smoke output: `/private/tmp/restir_r15_remote_clean_reproducibility/r13_default_smoke.pfm`
- Mean luminance: `0.031309634871571325`
- NaN/Inf pixels/components/throughput/radiance: `0`

Full-stack regression:

- Status: PASS
- Modes: `9`
- Parity cases: `4`
- Previous-stage validators orchestrated: `12`

## Parity Summary

R15 full-stack parity:

| Fixture | PSNR | SSIM | Log RMSE | RGB RMSE |
| --- | ---: | ---: | ---: | ---: |
| `living_room_clean_regir` | `32.760615028` | `0.997474310` | `0.023012789` | `0.026211163` |
| `living_room_practicals_regir` | `30.564209639` | `0.995835369` | `0.029633948` | `0.041698471` |
| `r3_diffuse_indirect_gi` | `inf` | `1.000000000` | `0.000000000` | `0.000000000` |
| `r5_pass_graph_svgf` | `inf` | `1.000000000` | `0.000000000` | `0.000000000` |

## Default-State Audit

R15 remote-clean default smoke verified:

- `restir_gi_mode=off`
- `path_guiding_mode=off`
- `restir_pt_mode=off`
- `svgf_denoise_enabled=false`
- path-guiding counters are zero by default
- ReSTIR PT counters are zero by default

Result: PASS

## Gating Audit

Explicit gates remain documented:

- `--restirGiMode=restir_gi_prototype`
- `--pathGuidingMode=path_guiding_prototype`
- `--restirPtMode=restir_pt_research`
- `--restirPtMode=restir_pt_experimental_path_reuse`
- `--svgfDenoise=1`
- `--debugPixel=x,y`
- `--directLightAuditJson=<path>`

Result: PASS

## Large Asset Reminder

`assets/living_room/living_room.obj` is tracked in Git and is about `54 MB` on disk. It triggered a GitHub warning at `52.76 MB`. It is required by remote-clean Living Room/ReGIR validation.

No history rewrite, force-push, deletion, or Git LFS migration was performed. Recommended human decision: keep for this PR, or plan a future Git LFS / smaller-fixture migration.

## PR Draft and Checklist

- PR draft: `docs/restir/PR_DRAFT.md`
- Visual review checklist: `docs/restir/VISUAL_REVIEW_CHECKLIST.md`

## Known Limitations

- Human visual validation is required for subjective image quality approval.
- R4 SVGF remains `EXPERIMENTAL_PASS`.
- R8/R9/R10 are research/experimental ReSTIR PT stages, not production ReSTIR PT.
- ReSTIR GI and path guiding are diffuse-first bounded prototypes.
- Clean builds without vendored OIDN/TBB dylibs build with OIDN disabled; `--denoise` reports unavailable.

## Unrelated Files

The original working tree at `/Users/dariopagliaricci/Metal-PathTracer-feature-restir-di-proto` still contains unrelated dirty/untracked files. R15 was performed from the clean worktree and did not modify, stage, reset, delete, or stash those files.
