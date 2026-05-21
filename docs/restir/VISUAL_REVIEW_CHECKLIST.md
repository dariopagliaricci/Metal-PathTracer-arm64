# ReSTIR Visual Review Checklist

Artifact directory: `/private/tmp/restir_r15_visual_review`

Contact sheet: not created. The repository has comparison tooling but no contact-sheet generator; use the listed PFM artifacts and metrics.

## Scenes to Inspect

- Living Room clean ReGIR fixture.
- Living Room practicals ReGIR fixture.
- R3 diffuse indirect fixture.

## Modes to Inspect

- `baseline_emissive`
- `ris_regir`
- `restir_di_regir_hybrid`
- `restir_gi_prototype`
- `path_guiding_prototype`
- `restir_pt_research`
- `restir_pt_experimental_path_reuse`
- `svgfDenoise`

## Artifact Paths

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

## What to Compare

- Default output should not unexpectedly enable GI, path guiding, ReSTIR PT, SVGF, debug pixel, or audit output.
- Megakernel and wavefront parity artifacts should be visually close for the same fixture and settings.
- ReGIR/RIS and hybrid modes should remain stable and non-black.
- R3 GI/path-guiding/ReSTIR PT prototype samples should remain bounded and finite.
- SVGF sidecar should reduce noise without excessive blur or energy loss.

## Expected Non-Goals

- R8/R9/R10 are not production ReSTIR PT.
- R4 SVGF is not production denoise.
- Path guiding and ReSTIR GI are diffuse-first bounded prototypes.
- Visual improvement is not guaranteed for every mode at low SPP.

## Acceptable Artifacts

- Low-SPP noise in raw path-traced outputs.
- Small stochastic differences between modes.
- Experimental/prototype mode differences when explicitly enabled.
- Slight denoise smoothing in SVGF output.

## Red Flags

- Black frames or unexpectedly near-zero energy.
- Fireflies, severe energy explosions, or obvious NaN/Inf artifacts.
- Mode changes altering default output when features are disabled.
- Obvious megakernel/wavefront visual divergence.
- SVGF excessive blur or obvious structure loss.
- ReSTIR PT experimental output presented as production-quality ReSTIR PT.
- Path guiding or ReSTIR GI affecting output when disabled.

## Reproduce Visual Artifacts

```sh
cd /Users/dariopagliaricci/Metal-PathTracer-restir-r13-clean
cmake --build build-r13-clean --target PathTracerHeadless
python3 scripts/validate_restir_roadmap_full_stack.py --binary ./build-r13-clean/PathTracerHeadless --artifact-dir /private/tmp/restir_r15_full_stack_regression --verbose
```

Targeted R3 prototype review renders:

```sh
./build-r13-clean/PathTracerHeadless --scene=tests/scenes/restir/r3/r3_diffuse_indirect.scene --backend=metal --enableSoftwareRayTracing=1 --width=128 --height=72 --sppTotal=2 --seed=42 --directLightMode=legacy --restirGiMode=off --pathGuidingMode=path_guiding_prototype --pathGuidingStrength=0.5 --pathGuidingCellSize=1.0 --restirPtMode=off --svgfDenoise=0 --pbrMetricsJson=/private/tmp/restir_r15_visual_review/r3_path_guiding_metrics.json --output=/private/tmp/restir_r15_visual_review/r3_path_guiding.pfm --format=pfm
./build-r13-clean/PathTracerHeadless --scene=tests/scenes/restir/r3/r3_diffuse_indirect.scene --backend=metal --enableSoftwareRayTracing=1 --width=128 --height=72 --sppTotal=2 --seed=42 --directLightMode=legacy --restirGiMode=off --pathGuidingMode=off --restirPtMode=restir_pt_research --restirPtMaxReservoirs=4096 --restirPtDebug=1 --svgfDenoise=0 --pbrMetricsJson=/private/tmp/restir_r15_visual_review/r3_restir_pt_research_metrics.json --output=/private/tmp/restir_r15_visual_review/r3_restir_pt_research.pfm --format=pfm
./build-r13-clean/PathTracerHeadless --scene=tests/scenes/restir/r3/r3_diffuse_indirect.scene --backend=metal --enableSoftwareRayTracing=1 --width=128 --height=72 --sppTotal=2 --seed=42 --directLightMode=legacy --restirGiMode=off --pathGuidingMode=off --restirPtMode=restir_pt_experimental_path_reuse --restirPtMaxReservoirs=4096 --restirPtReuseStrength=0.25 --svgfDenoise=0 --pbrMetricsJson=/private/tmp/restir_r15_visual_review/r3_restir_pt_experimental_reuse_metrics.json --output=/private/tmp/restir_r15_visual_review/r3_restir_pt_experimental_reuse.pfm --format=pfm
```
