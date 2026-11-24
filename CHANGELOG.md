# Changelog

## v4.0.0 - Accuracy, Embree Parity and Research Systems Release

Scene names below refer to the five bundled scenes (`ajax`, `ajax-other`,
`ajax-glass`, `hygieia`, `hygieia-other`) or to scenes in the v4 asset pack.
Each number is quoted with the scene or test it was measured on.

### Transport Correctness

- Fixed delta specular connections and the MIS of unpaired continuations. Glass,
  mirrors and smooth metals change under specular NEE, which is on by default.
- Fixed the PBR delta lobe (roughness 0), the specular energy compensation and
  the glass transmission transport. glTF transmission without a volume now
  renders as thin-walled glass, tinted by the base colour at every crossing.
- Fixed the Plastic closure. Its sampling now matches its evaluation. Under a
  constant environment, BSDF-only and light-only renders were 0.972 and 1.104
  of the MIS render; they now agree within 0.01% on ten plastic profiles.
- Made CarPaint a consistent layered BSDF and fixed the conductor Fresnel term.
- Skipped the diffuse lobe for pure metals (metallic 1 with black diffuse).
  The 1024 spp mean ratio is 1.00005 and 1.00003.
- Changed Metal camera rays to unit length. `t` is now a world distance on
  camera rays, as on bounce rays and on Embree. Before, the 1e-3 minimum
  distance on camera rays was parametric, and a camera ray continuing past a
  cutout hole skipped about 2 cm on sponza. Sponza ivy tile, first-hit base
  colour, Embree/Metal: 0.7764 to 0.9870.
- Fixed a double count in MIS. A path ray that passed an alpha hole reset the
  MIS state as if it had made a delta bounce. When every light path crossed a
  hole, the image was 1.42 times too bright on Embree and both Metal executors.
  A pass through an alpha hole is now a null event for MIS.
- Made shadow rays pass a BLEND texel with probability 1 - alpha. Path rays read
  BLEND alpha at mip 0, as shadow rays do.
- Made specular NEE and MNEE rays on Metal alpha-aware, as other shadow rays
  already were.
- Ignored near same-mesh hits in closest-hit shadow traces too. On the Ferrari
  scene the megakernel and the wavefront were 2 pixels apart at 1 spp. They are
  now bit-identical at seeds 42, 7 and 1234.
- Used the bounce-ray near distance (1e-3) for Metal environment shadow rays.
  BoomBox antenna root, environment sampling on against off: +1.954% before,
  -0.221% now.
- Fixed environment sampling: longitude and alias-table jitter on GPU and CPU,
  an importance distribution accumulated in double, and HDR suns decoded as
  RGBA32F so that texels above the half-float range are no longer clipped.
- Removed the roughness blur of the environment on Metal and Embree. NEE and
  diffuse-lobe misses read different mips, so the two techniques did not sum to
  one integral. On the Embree Damaged Helmet inlet walls, sampling on read
  0.0089 and sampling off read 0.0147 before the fix.
- Exported the mixture weight for rough PBR reflection on Metal. About 5% of the
  rays on dark walls carried roughly 100 times the weight of the rest, which
  caused speckle on Damaged Helmet.
- Faced software-BVH hits on mirrored instances by their world-space winding,
  as hardware ray tracing does. Software tracing saw a mirrored instance from
  behind and dropped its one-sided emitters. `zero_day_measure_seven` under
  `--force-swrt=1`, software/hardware: 0.8424 to 0.9976 (megakernel, full
  depth).
- Treated a software megakernel NEE shadow ray that hits the sampled light's
  own triangle as unoccluded, as the hardware megakernel does. Far floor under
  a quad lamp, software/hardware: 0.9854 to 1.0000.

### Materials and Textures

- Changed the display and LDR export to the piecewise sRGB encode by default,
  with reference tone mapping. `srgbOutput=0` restores gamma 2.2.
- Fixed blurred reflections from normal-map length variance. Normal textures now
  build mips from normalized directions and store the spread of the directions
  in alpha. On Damaged Helmet, with a constant map, visor roughness went from
  0.040 to 0.132 before the fix.
- Read lambert and plastic base-colour textures at mip 0 after the primary hit
  on the wavefront. The wavefront rendered `bitterli_staircase` 4.7% brighter
  than the megakernel; the executors now agree within 0.02%.
- Tested MASK cutouts on path rays against the hit's own mip-0 alpha, as shadow
  rays do. MASK foliage lit from behind now follows one rule in both
  integrators. Bistro sun, 1024 spp: wavefront/megakernel 0.922 at depth 1
  and 0.949 at depth 10 before; 1.0000 at both now.
- Kept the texture mean in the mip chains of non-power-of-two material textures,
  on Metal and Embree. Power-of-two chains are unchanged. San Miguel is 1-4%
  darker, barcelona_pavilion_day about 1% darker, and living_room +0.3% in red.
  The environment map keeps Metal's blit chain.
- Added per-mesh `gltfTexturedEmissionNee=1` for textured emitters. It replaces
  a path match on a Sun Temple folder name.
- Added `gltfReferenceDielectricContinuation=1` for assets whose reference
  renderer keeps throughput through refraction. It is off by default.
- Added `gltfMaterialFactors` for per-material glTF calibration factors in
  scene files.
- Removed the inert `TransmissionKeepsSpecular` material flag.

### Direct Lighting (NEE, RIS, ReSTIR, ReGIR)

- Made `ris_spatial` unbiased on the megakernel. It now merges neighbours with
  pairwise MIS and uses the solid-angle Jacobian for triangle-light samples.
  Sun-plane test: 2.33004 against the exact 2.33005 (0.822 before). The
  wavefront still runs `ris_spatial` as plain RIS, logs it and sets
  `wavefront_direct_light_fallback` in the render metadata.
- Made `ris_world` unbiased on both executors, with and without hardware ray
  tracing. Sun-plane test on a mesh copy: 2.33004 (1.92047 before, where its
  candidates hit).
- Merged ReGIR probes by plain streaming in `ris_regir` and in the ReSTIR
  DI/ReGIR hybrid. Sun-plane test (2.33005): 2.20459 and 2.24151 before,
  2.33004 now. The hybrid now builds its provider reservoirs with the vertex's
  own occlusion and specular-only flag.
- Gave every RIS and ReSTIR DI target and winner shading plain NEE's surface
  terms. Two-sided MASK cutouts and thin transmissive sheets lit from behind
  are now lit in the reservoir modes. San Miguel balcony, `ris` and
  `restir_di` against plain NEE: 0.9417 to 1.0000.
- Alpha-tested the reservoir shadow rays of the software megakernel and the
  shadow kernels of the software wavefront. On a holed leaf sheet, plain NEE now
  gives 0.48105 on both executors under hardware and software tracing.
- Kept rect lights and the environment out of the hybrid ReSTIR DI reservoir.
- Refused the explicit ReSTIR DI passes (`--restirDebug=1`) on the megakernel.
  Those passes read no material textures, normal maps or occlusion, and gave
  1e28-1e30 on the sun-plane test. Headless runs exit with code 2 before the
  acceleration structures build. The window logs it once and renders the fused
  `restir_di`. The wavefront's explicit stage gives 2.33 on the sun-plane test.
- Gave a wavefront frame that falls back to the megakernel its direct light
  under explicit ReSTIR DI, and reported each fallback in the log and in the
  metadata fields `wavefront_megakernel_fallback` and
  `wavefront_direct_light_fallback`.

### Subsurface, Volumes, Spectral, Caustics and Radiance Cache

- Rebuilt Metal's random-walk subsurface (`method=randomwalk`) on the megakernel
  (hardware and software tracing) and the wavefront. Before, no walk left the
  mesh, and the busts in `sss_marble_wax` were 2.7 to 3.2 times darker than
  Lambert busts of the same colour. On a unit sphere under one emitter the
  result is within 0.06% of Arnold's randomwalk at mean free path 0.1, 0.31% at
  0.25 and 0.08% per channel in a marble colour. Walk endings are counted per
  material in the log and in the metadata field `sss_walk`. `method=separable`
  is not implemented.
- Added hero-wavelength spectral transport with three wavelengths per path
  (`--spectralWavelengthsPerPath=3`, the default; `1` is a corrected
  single-wavelength baseline). The sensor and white balance apply once at path
  exit. Paths collapse to the hero wavelength at dispersive vertices. Headless
  runs refuse unsupported combinations with exit code 2, including the default
  specular NEE, so spectral runs need `--enableSpecularNee=0`. Spectral off is
  unchanged. The README section "Hero-Wavelength Spectral Mode" has the support
  matrix.
- Changed the radiance cache to store radiance. It now learns the radiance that
  paths gather after a cached vertex, trained by one pixel in eight per frame.
  Its measured bias is at most 0.03% of the full-frame mean luminance on
  `living_room`, `sponza` and `emerald_square_day`. It gives no error reduction
  at equal render time. Per sample, cache on costs 2.76 times cache off on
  `living_room`, 2.23 times on `sponza` and 1.38 times on
  `emerald_square_day`. It is refused with ReSTIR GI, ReSTIR PT path reuse and
  spectral mode. Cache-off output is byte-identical to before.
- Attenuated surface NEE to a light with a position by the transmittance of a
  homogeneous medium. Quad lamp, sigma_a 0.5: 0.1947 before, 0.0922 now
  (reference 0.0921). Suns and the environment are not attenuated.
- Fixed homogeneous-volume transport on every executor.
  - Free flight is drawn as in pbrt-v3, so the transmittance counts once.
    Unscattered light through optical depth 1: 0.3678 before, 1.0016 now, of
    exp(-1).
  - Rect lights seen from the medium take the MIS weight against the phase
    sample, and the environment is no longer weighted against a pdf that
    scatter points never sample. Phase-only plus NEE-only: 1.2977 before,
    0.9997 now, of the reference.
  - Meshes now occlude volume NEE on the hardware megakernel. Light under a
    black mesh plate: 0.01569 before, 0.00397 now. The software megakernel
    skips discarded texels. The wavefront's volume NEE carries the path
    throughput; it was 2.6 times too bright.
  - Camera segments are attenuated over world length. Sun-plane test in an
    absorbing medium (sigma_a 0.5): 1.41324 before, 1.81348 now, the exact
    image mean.
- Changed the Henyey-Greenstein phase function to pbrt's convention.
  `--volumeAnisotropy` greater than 0 now scatters forward. The sign was
  mirrored before. The glow ratio g +0.7 over g -0.7 is 19.6 to 25.0 against
  Arnold's 17.1 to 23.5. The 6-8% residual is unexplained.
- With `--volumeTransportMode=homogeneous`, Metal renders `restir_di` and the
  hybrid as `baseline_emissive`, because the reservoirs carry no medium state.
  It logs this and writes `volume_direct_light_fallback`. Both modes gave 0 on
  the sun-plane test before.
- Embree ignores volumes.

### Embree CPU Reference Backend

- Brought Embree toward Metal on many material and light rules. The list below
  is the user-visible part; each rule follows the Metal rule it names.
  - Samplers honour the mip mode; Emerald canopy coverage 0.8750 to 0.8982
    against Metal's 0.8973.
  - Analytic suns are lit in the same direct-light modes as Metal.
  - ReSTIR DI lights the primary vertex per sample and uses plain NEE's surface
    terms. San Miguel balcony, sun only, `restir_di` against plain NEE:
    0.9418 to 1.0000.
  - Disks are intersected analytically. Metal's mip rules for BLEND alpha, DDS
    mips and deferred ORM mips apply.
  - Two-sided PBR triangles, back-lit alpha cutouts and normal-mapped back
    faces follow Metal. Ray offsets follow the resolved shading normal.
  - Shadow rays follow Metal's alpha-aware rules and stop 1e-3 short of
    positional lights. Sponza ivy tile, direct light, Embree/Metal:
    0.926 to 0.976.
  - Path rays start 1e-3 out in world units and skip the previous hit's
    triangle. Sponza ivy tile, full depth, Embree/Metal: 0.961 to 0.979.
  - Solid dielectrics absorb light with Metal's medium stack.
  - The subsurface random walk follows Metal's.
  - The provider reservoir of ReSTIR DI counts every drawn candidate in M.
    Emerald `restir_di` over plain NEE: 1.3003 to 1.0016. Bistro alt,
    Embree/Metal: 1.0951 to 0.9961.
- Ran Tungsten source closures on Embree through headers that Metal and C++
  both compile. With `sourceTransport=0` on local Bitterli copies, Embree/Metal:
  classroom 1.0018, dining_room 1.0006, material_testball 0.9995. Embree now
  also runs source transport, nearest texture filtering, source opacity, bump
  maps and thin sheets. `material_testball`, `veach_ajar`, `bedroom`,
  `living_room`, `bathroom` and `kitchen` pass 120/120 tiles, with a global
  Embree/Metal ratio of 0.9995 to 1.0006. Embree still refuses by name opacity
  and `interfacePrecision` without source transport, alpha cutouts on
  explicit-interior materials under transport, and a source texture that mixes
  nearest and linear filters.
- Followed Metal in alpha cutouts and BLEND on Metal and Embree.
  - A pass through a MASK hole, or a BLEND texel with alpha 0, is a null event
    for MIS. Black MASK checker: 1.422 to 1.0013 on all four executors.
  - Shadow rays pass a BLEND texel with probability 1 - alpha. Sun Temple
    per-pixel variance at 128 spp fell 6.2 times (11.07 to 1.78).
  - On a back-lit two-sided cutout, the fallback's pdf stays out of NEE's MIS
    weight, and the fallback carries the diffuse lobe's weight
    1 - transmission x (1 - metallic). Balcony leaf crop, Embree/Metal:
    0.858 to 1.005.
  - A foliage texture name makes a cutout only where the base-colour image has
    alpha holes. Bistro frame: -0.6%.
- Decoded DDS blocks (BC1, BC3, BC5) on Embree with the rules of Metal's GPU.
  Sun Temple, Metal/Embree: 1.0067 to 1.0001.
- Skipped Embree's multiple-scattering specular compensation on Falcor-standard
  materials (Emerald Square's), as Metal does. `emerald_square_day`
  `baseline_emissive`, Embree/Metal: 1.0042 to 1.0000 at maxDepth 1, and
  1.0047 to 0.9993 at full depth.
- Offset Embree's NEE and bounce rays along the resolved shading normal, and
  transformed vertex tangents by `localToWorld`, as Metal does.
- Reduced Embree load memory with bit-identical images. Peak footprint:
  Emerald 64.0 to 20.4 GB, Bistro sun 44.5 to 18.4 GB, Barcelona Pavilion
  27.4 to 17.2 GB, Sun Temple 22.6 to 7.2 GB. Sun Temple renders 3-6% slower on
  Embree.
- Reported every material feature, debug switch and setting that Embree ignores,
  once per render, instead of dropping it silently.
- Faced an analytic sphere's shading normal toward the ray for every material.
  Camera inside a lambert sphere, Embree/Metal: 1.735 to 0.9993.

### Wavefront and Megakernel Performance

- Added a second hardware megakernel pipeline that folds 19 research and debug
  gates to 0. The host picks it when every gate is off. Timed on two scenes only
  (megakernel, 1280x720, 64 spp): sponza 3.3% slower (2.54 s to 2.63 s) after
  the direct-light mode was kept live in it; Sun Temple 1.6 times faster. Other
  scenes were not timed. The two pipelines render bit-identically under
  `MTLMathModeSafe`.
- Removed the software shadow re-trace from the hardware megakernel.
- Built megakernel integrate pipelines on first use. Cold start of
  `bitterli_veach_ajar` on the wavefront: 283 s before, 69 s now. Warm starts
  are unchanged.
- The metadata field `megakernel_integrate_pipeline` and the
  `[Megakernel] integrate pipeline` log line report which pipeline ran.

### Scene Loading and Import

- Sped up scene loading with byte-identical images, measured on one Apple
  Silicon machine.
  - The loaders' event pumps no longer wait. Pavilion 92.9 s to 23.2 s, Bistro
    26.6 s to 16.1 s, Emerald 30.0 s to 17.1 s.
  - Each glTF primitive is decoded once for all its instances. Pavilion
    23.0 s to 6.7 s.
  - Material images, KTX2 files and normal-map mip chains are prepared ahead on
    worker threads. Bistro 20.5 s to 11.9 s. The prefetch lists only distinct
    images (Bistro 591 to 277).
  - Instances of one mesh share one software BLAS (Pavilion build 3.4 s to
    0.47 s, peak 2.4-3.0 GB lower). Different meshes build in parallel
    (Emerald 1580 ms to 690 ms, Bistro 666 ms to 271 ms).
- Fixed `PathTracerImport` texture V. It writes a top-left V origin for every
  path, with one rule for all formats. It hoists OBJ `mtllib` directives and
  checks OBJ materials, which fixes the repeated-`mtllib` material bug.
- Imported FBX-connected textures before the file-name convention. A textured
  emitter bound through its connection replaces the FBX emissive colour. On the
  Zero Day measure scene the linear mean went from 0.00995 to 0.00823, because
  white emitters had been bound to the wrong map.
- Scaled Bistro's table of 13 practical lights by `gltfEmissiveScale`, and
  logged each light. A scale of 0 now leaves no glTF emitter on.
- Added opt-in Tungsten source closures (conductors, plastics, dielectrics,
  Oren-Nayar diffuse), source bitmap sampling, bump maps, source transport and
  OIDN guides that follow mirrors and glass. They are set per scene and are
  built into a second shader library the first time a scene uses them.
- Added a Tungsten filmic display tone map (tonemap mode 6), accepted by the
  `tonemap` scene token, `--tonemap` and the GUI operator list.
- Added scene profiles and transport fixes for Barcelona Pavilion, Falcor-matched
  material shading for Emerald Square, and import fixes for Sun Temple and the
  vintage cars. These scenes are in the v4 asset pack.

### Denoising

- Averaged OIDN albedo and normal guides over every jittered sample on both
  Metal executors. Before, most scenes wrote the last sample's aliased guide,
  and OIDN reproduced stair-stepped edges. SVGF and ML reconstruction keep
  per-frame guides.
- Made the wavefront's OIDN guides follow water and glass to the first
  non-delta surface, as the megakernel does. The pool in Barcelona Pavilion day
  was milky before.
- Added `--writeDenoiseGuides` to `PathTracerHeadless`.

### GUI and Output

- Sized the GUI render from the window: drawable size times the viewport
  quality scale, for every scene. Before, a scene that set `renderer width=...
  height=...` rendered at that size times the scale, whatever the window. The
  Bitterli scenes rendered at 640x360 at the default quality 0.50. Presentation
  Mode's locked resolution and window resizes were not checked.
- Showed scenes with an authored render size at the window's aspect.
- Reset glTF load options and `directLightMode` on every scene load. Before,
  Damaged Helmet loaded with its texture V flipped after San Miguel.
- Restored the viewport's execution mode after a scene that forced the
  wavefront, and reset `russianRoulette` on every scene load.
- Fixed EXR files saved from the GUI. Channels are now written in sorted order,
  so standard readers show the right colours. Alpha is 1 and the sample count
  stays in the `SAMPLES` channel.
- Wrote non-finite values in the material and direct-light debug JSON as null,
  counted and logged. Before, `--materialMrDebugJson` could write `nan`.
- Logged the resolved shader directory and wrote it to the render metadata as
  `shader_directory`. The lookup order is unchanged.

### Build, Platform and Tests

- Fixed a macOS 27 internal compiler error in the AGX GPU compiler. The wavefront
  shadow kernels failed to compile on an M5 Max. The environment delta chain
  functions are now `noinline`.
- Made the `STRICT=ON` build pass. It had 344 unique errors, 107 in project code
  and the rest in vendored code.
- Replaced deprecated Managed storage with Shared storage.
- Built without bundled scene assets when `assets/` is absent. Disabled the
  Bistro smoke test when its scene is missing.
- Registered the `M0_TESTS` golden image tests only when their script exists.
- Added the CTest entries `shared_closures` and `radiance_cache_training_target`.
- Removed dead code, corrected stale README and CLI help text, and added
  warnings where a setting or material feature is ignored. The `[Beauty]` log
  line now reads resolved settings.

## v3.0.0 - Public Renderer Scalability Release

### Renderer Scalability and Large-Scene Infrastructure

- Added headless scene statistics reporting for triangle, mesh, instance,
  texture, geometry-memory, texture-memory, BLAS, TLAS, and estimated
  working-set totals.
- Added `--budget-policy=<strict|warn|ignore>` to enforce or audit scene memory
  budgets against `MTLDevice.recommendedMaxWorkingSetSize`.
- Added BLAS/TLAS build instrumentation and explicit backend logging so Metal
  HWRT fallback to SWRT is never silent.
- Added explicit runtime/backend controls for headless renders:
  `--stats`, `--force-hwrt`, and `--force-swrt`.
- Added large-scene pilot controls for Bistro-class scene bring-up:
  `--pilot-mode=<geo_only|albedo_only|full_clamped>`,
  `--texture-max-dim`, and `--max-texture-bytes`.

### Metal Wavefront Execution and Pass Graph

- Added selectable execution modes through `--executionMode=<megakernel|wavefront>`.
- Added wavefront scheduling policies:
  `--wavefront-policy=<preview|final|offline|research>`.
- Added optional queue compaction through `--wavefront-compaction=<0|1>`.
- Added Metal pass graph and wavefront metrics infrastructure for headless
  validation and profiling.
- Added public wavefront render gallery images under `Images/Metal-Wavefront-Images`.

### ReSTIR / RIS / ReGIR Research Lighting Stack

- Added direct-light modes: `legacy`, `baseline_emissive`, `ris`,
  `ris_spatial`, `ris_temporal`, `ris_world`, `ris_regir`, `restir_di`,
  and `restir_di_regir_hybrid`.
- Added ReSTIR DI explicit pass graph support for headless rendering and
  opt-in GUI profiling.
- Added ReGIR/world-space candidate reuse and a DI+ReGIR hybrid path.
- Added bounded ReSTIR GI diffuse prototype.
- Added ReSTIR PT research scaffold and experimental diffuse-safe path reuse.
- Added path guiding and radiance cache prototypes.
- Kept high-risk research paths disabled by default and gated behind explicit
  CLI/UI controls.

### Advanced Transport Prototypes

- Added path-space caustic transport prototype.
- Added homogeneous volume transport prototype with isotropic and
  Henyey-Greenstein phase options.
- Added hero-wavelength spectral transport prototype with dielectric dispersion
  controls.
- Added Metal SVGF-style denoise as an experimental sidecar path.
- Added optional firefly clamp modes for offline beauty output control.

### Import Pipeline, glTF Cache, and Texture Conversion

- Added `PathTracerImport`, a standalone importer for Assimp-supported static
  scenes.
- Added deterministic glTF/GLB import output with `import_manifest.json`.
- Added import manifest schema emission with `schema_version: "1.0"`.
- Added optional CMake Assimp discovery with a stub fallback when Assimp is not
  installed.
- Added importer texture policies for copied external textures plus experimental
  embedded and linked texture modes.
- Added deterministic `.ktx2` texture conversion for Bistro DDS textures and
  common raster formats.
- Added runtime KTX2 texture loading for converted importer output.

### Production Metadata and Offline Workflow

- Added render profiles: `custom`, `preview`, `lookdev`, `final`, `reference`,
  and `debug`.
- Added deterministic settings JSON and render queue item JSON output.
- Added render queue replay through `--runRenderQueueItemJson`.
- Added production metadata sidecars and debug bundle directory output.
- Added checkpoint manifest, resume support, and tiled render manifest output.
- Added PBR metrics JSON, material-channel inspection, Material_MR debug JSON,
  direct-light audit JSON, and camera search reports.

### Public Scenes, Assets, and Gallery Updates

- Added public Ajax scenes: `ajax.scene`, `ajax-other.scene`, and
  `ajax-glass.scene`.
- Added public Hygieia scenes: `hygieia.scene` and `hygieia-other.scene`.
- Added Ajax and Hygieia meshes plus the two HDRIs used by those scenes.
- Consolidated public render galleries under `Images/`.
- Added `assets/CREDITS.md` with upstream asset acknowledgements and
  licensing/source notes.
- Updated README image paths for the consolidated `Images/` layout.
- Updated the public asset-pack link and installation instructions for
  `Metal-PathTracer-Assets-v3-0-0.zip`.

### Optional Reference Backends and Public CI

- Kept Embree as an optional CPU reference backend when installed and configured
  with `-DPATH_TRACER_ENABLE_EMBREE=ON`.
- Kept `--backend=<metal|embree>` and `--threads=<int>` for compatible builds.
- Kept Intel Open Image Denoise optional; clean builds without vendored OIDN/TBB
  libraries compile with OIDN disabled and report denoise unavailable.
- Updated public CI so source-only Metal/headless builds do not require local
  Embree or OIDN binaries.
- Removed private `docs/` and `scripts/` content from the public repository while
  preserving public tests and validation baselines.

## v2.0.4 - Environment Robustness (Sponza)

- Added canonical `sponza` scene discovery via `assets/sponza.scene`.
- Added fixed Sponza camera and headless baseline settings for deterministic 1280x720 EXR renders.
- Added `--spp` and `--out` CLI aliases for milestone reproduction commands.
- Added a dedicated `scripts/v2_0_4_sponza_validate.py` acceptance runner for determinism and HWRT/SWRT parity.
- Documented the canonical Sponza baseline and HWRT/SWRT parity workflow in `README.md`.
