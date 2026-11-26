# Changelog

## v4.1.0 - Embree Parity, Software Tracing and Importer Fixes

Covers PRs #144 to #191 of the private line, merged after the v4.0.0 freeze.
Scene names refer to the five bundled scenes (`ajax`, `ajax-other`,
`ajax-glass`, `hygieia`, `hygieia-other`) or to scenes in the v4 asset pack.
Each number is quoted with the scene or test it was measured on.
Where a bullet says "Metal does not change", the 18-row Metal gate (the v4.0.0
freeze rows) was 18/18 IDENTICAL.

### Transport Correctness

- Traced the closest hit before the wavefront software shadow kernels count a
  hit on the sampled light as unoccluded (#147). An any-hit trace could return
  the light although a blocker lay in front of it. Glossy-floor probe,
  wavefront SW/HW: 1.0009 to 1.0000 at maxDepth 1, 1.0007 to 1.0000 at
  maxDepth 8. Hardware kernels were not changed in this PR.
- Did the same for the wavefront hardware shadow kernels (#150). An analytic
  sphere or rectangle in front of the light was missed. Wavefront HW/megakernel
  HW at scale 10: sphere blocker 1.0699 to 1.0000, rectangle blocker 1.2735 to
  1.0000. Triangle blockers were not affected.
- Tested analytic spheres against the closest hit in the software tracer, also
  after a triangle hit (#156). Before, a sphere in front of a triangle was
  invisible to camera, bounce and shadow rays under software ray tracing, and a
  rectangle behind a sphere could replace it. Software/hardware: sphere at
  scale 1 1.0312 to 1.0000, sphere at scale 10 1.1010 to 1.0000. Timing on
  sponza and sun_temple (megakernel SW, 1280x720, 64 spp, 4 alternating
  pairs): the fastest runs agree within 0.2%.
- Resumed software ray tracing 5 mm past an excluded spawn-triangle hit with
  tMin 0, as the hardware trace and Embree do (#175). Before, it resumed about
  15 mm past the hit and jumped over the back face of Zero Day's light-tower
  shells. The software wavefront and the delta-chain visibility test move the
  same way. SW/HW, 480x270, 2 x 256 spp: `zero_day_measure_one` megakernel
  0.9917 to 0.9983, wavefront 0.9895 to 0.9953; `zero_day_measure_seven`
  stays 0.9999; `sponza` 1.0000 to 0.9999. Embree/SW on `zero_day_measure_one`:
  1.0019 to 0.9953. Hardware rows of the Metal gate are IDENTICAL; `swrt_sponza`
  differs on 0.2% of pixels. Grazing rays that re-cross their own triangle
  still skip 5 mm of geometry on every backend.
- Ran `ris_spatial`'s neighbour reuse on the wavefront instead of plain RIS
  (#179). It uses the megakernel's rules: 1 to 4 neighbours at 1 pixel offsets,
  depth within 10%, shading normals within 0.9, no world-cell test.
  `wavefront_direct_light_fallback` is now "none" for `ris_spatial`. Sun-plane
  mesh probe (exact 2.33005): wavefront 2.33004 on hardware and software, and
  megakernel 2.33004. `living_room` direct light against plain NEE:
  wavefront 0.9997, megakernel 0.9997. RMSE at 16 spp: 0.06239 to 0.05537.
  Cost at 1280x720, 64 spp: wavefront `ris` 0.77 s, `ris_spatial` 1.60 s. No
  shipped preset uses `ris_spatial`. Every gate row is byte-identical.
- Kept a re-spawned trace's hit distance and reach in the original ray's frame
  (#180). Both tracers reject a hit on the spawn triangle, a self-hit or a
  near-mesh hit within 5 mm, then trace again from 5 mm past it. After that
  re-spawn, the returned `t` was short by the distance moved (about 5 mm), and
  the ray's reach (tMax) was not shortened. Callers read `t` as the distance
  from the traced ray: ray-cone texture footprint, medium and shadow-segment
  lengths, the alpha shadow loop and the BLEND shadow hash. The tracers now add
  the moved distance back, and a trace whose re-spawn passes the segment's end
  stops with no hit. Rays without a re-spawn do not change. Megakernel, 480x270,
  2 x 256 spp, luminance new/old: `sponza` (also in a homogeneous medium),
  `living_room`, `bistro_full_sun_restir_alt` and
  `san_miguel_balcony_sun_restir` 1.00000 on hardware and software;
  `zero_day_measure_one` hardware 0.99989, software 0.99978 (worst 32-pixel tile
  -0.34%). At 960x540, 2 x 1024 spp, `zero_day_measure_one` hardware frame
  1.00000; the most-changed 160x90 region 0.9995 +-0.0003. Embree/Metal on
  `zero_day_measure_one`, 2 x 512 spp: hardware 0.9973 to 0.9974, software
  0.9990 to 0.9992. Hardware images change by noise-level amounts. In the
  Metal gate at gate spp, `zero_day_measure_one` moves through a few fireflies
  (megakernel 1.00182, wavefront 0.99928); at 2 x 1024 spp it is 1.00000.

### Materials and Textures

- Fixed the normal-map bitangent on mirrored glTF primitives that have no
  TANGENT accessor (#151). The loader now generates MikkTSpace tangents on the
  authored winding. On the `tangent_mirror` probe, the mirrored quad's centre
  luminance went from 0.30906 to 0.00076 on all five backends. No shipped
  preset has such a primitive with a normal map (44 glTF/GLB files checked).
- Gave a normal map on an all-zero vertex tangent one rule on Metal and Embree
  (#157). The frame now comes from the triangle's UVs, with no second mirror
  flip. With degenerate UVs neither backend applies the normal map. On the
  `tangent_mirror` probe the all-zero-tangent quads read 0.00077 on every
  backend; Metal had read 0.30883 on the identity quad and Embree 0.19358.
  `zero_day_measure_seven`: Embree is byte-identical, and Metal hardware
  differs from main within its run-to-run noise (frame mean ratio 1.00000).
- Selected the Barcelona Pavilion's loader rules with the mesh token
  `gltfBarcelonaPavilionRules=1` instead of the file path (#152). A copy of the
  asset under another name used to lose them without a message. The three
  Pavilion presets carry the token and render byte-identically. A file whose
  path looks like the Pavilion and has no token logs a warning.
- Built power-of-two 8-bit colour mips from `MipChain.h` instead of Metal's
  blit (#172). The blit rounds each level to the nearest sRGB code, and the
  rise adds up: at the 1x1 level the chain was 1.0106 times the mip-0 mean on
  Bistro's base-colour textures, and up to 1.04 on dark ones. Embree's CPU copy
  decodes the same levels. `bistro_full_sun_restir_alt`, Embree/Metal, full
  depth: 0.9953 to 0.9998 (tiles 114/0/6 to 118/0/2); maxDepth 2: 0.9965 to
  0.9999. Metal frame mean, new/old: `bistro_full_sun_restir_alt` 0.9955,
  `bistro_night_openfront_restir` 0.9989, `sponza` 0.9992,
  `f1_ferrari_sf1000_restir_photoshoot` 0.9998, `living_room` 1.0000, 11 other
  presets 1.0000. Presets with power-of-two 8-bit colour textures change on
  Metal. `emerald_square_day`, `sun_temple` and `barcelona_pavilion_day` keep
  their mean. A table quantizer replaces the per-texel `pow` (0 mismatches on
  1,065,353,217 floats), and the prefetch builds the chains ahead. Load,
  before to after: Bistro 4.20 to 4.23 s, Emerald 4.63 to 4.69 s, San Miguel
  1.42 to 1.29 s.
- Decoded the CPU copy of BC sRGB textures from sRGB (#174). It was decoded as
  linear, so Metal weighted textured emitters by sRGB-encoded brightness for
  next-event estimation and Embree by linear brightness. Zero Day Seven total
  emitted power: Metal 69.45 to 27.30 (Embree 27.39). Zero Day Seven,
  maxDepth 1, Embree/Metal HW: floor 0.9821 to 0.9997, frame 0.9962 to 0.9993.
  Full depth: 0.9984 to 0.9985. Of 20 presets at 320x180, 4 spp,
  `zero_day_measure_seven`, `zero_day_measure_one`,
  `zero_day_measure_seven_colored_lights` and `emerald_square_day` change; the
  other 16 are byte-identical. At 960x540 the frame means stay within noise
  (Zero Day Seven 1.0000, One 1.0007). Metal gate: IDENTICAL 16, DIFFERS 2
  (the two `zero_day` rows).
- Layered a coated random-walk subsurface material as Arnold does on Metal
  (#186) and on Embree (#191). Metal was 0.80 of Arnold (kick 7.4.3.2) on the
  matched sphere, for two reasons. The coat lobe used Schlick's Fresnel, which
  is 0.72-0.84 of the exact dielectric value at the highlight's half-angle
  (IOR 1.5). The walk under the coat was also weighted at both its entry and
  its exit. The coat lobe of a `type=sss` material now uses the exact
  dielectric Fresnel. The walk is chosen and weighted once, at its entry, by
  what the coat does not reflect toward the view, and its exit is white diffuse
  under the coat's tint with no Fresnel. `type=plastic` keeps Schlick and its
  layering, and uncoated walks, coat-off materials and dense-limit hits do not
  change. Matched sphere, base 0.8/0.6/0.5, coat IOR 1.5, roughness 0.02,
  Metal megakernel HW 2048 spp, against Arnold: mean free path 0.1, 0.07083
  (0.802) to 0.08826 (0.9994 of Arnold's 0.08831); mean free path 0.25, 0.09340
  (0.815) to 0.11446 (0.9988 of 0.11460); coat lobe alone 0.784 to 1.0000. Coat
  off is byte-identical. Shipped looks change on purpose, wavefront HW, 960x540,
  256 spp, frame mean: `sss_marble_wax` 0.30115 to 0.32285 (+7.2%; the rosso
  verona and jade busts, the separable wax bust does not change) and
  `stanford-dragon` 0.66757 to 0.69255 (+3.7%). The owner accepted the look.
  Embree/Metal, 2 x 8192 spp, 480x270: `stanford-dragon` 1.0001,
  `sss_marble_wax` 1.0000 (megakernel) and 0.9999 (wavefront), 120/120 tiles on
  both. The 18-row Metal gate is IDENTICAL. The choice between coat and walk
  still uses Schlick and the plastic layering in its probabilities. That changes
  variance only.
- Let a normal map shape the coat of a coated random-walk subsurface material
  on Metal (#188) and on Embree (#189), as in Arnold. A `type=sss` material
  takes `normalTexture=` and `normalScale=`. The map shapes only the coat: the
  mapped coat normal sets the choice between coat and walk and shades the coat
  lobe. The walk's entry weight, its direction and its exit keep the
  interpolated normal. Coat light that the mapped normal reflects below the
  interpolated normal is not lost: it enters the walk, as in Arnold, whose
  subsurface output gains what its specular output loses. With the coat off or
  `method=separable`, the map does nothing, and the parser logs it. Ratio of
  the frame mean to the same scene with no map, coated UV sphere, Metal
  megakernel HW 2048 spp, at the #188 merge (Arnold in brackets): stripes at
  mean free path 0.1, 0.984 (0.998); 0.25, 0.992 (1.0024); flat 1.0010
  (1.0005); 20-degree tilt 1.007 (1.019); sine ridges 0.998 (1.0075). With #186
  layering, stripes read 0.9815 and 0.9895 and flat 1.0014. Embree/Metal on the
  coat probe, 2 x 4096 spp: no map 0.9999, flat 0.9997, stripes 1.0003. With the
  coat off the map has no effect: the coat-off probes are byte-identical. Embree is byte-identical on 8 shipped presets
  (480x270, 128 spp). On Metal, the 18 gate rows are byte-identical; on the
  wavefront, `sss_marble_wax`, `stanford-dragon` and the no-map coat probe differ by
  float order only (maximum 1e-6, byte-identical under `MTLMathModeSafe`).
- Gave OBJ and PLY scene meshes a zero tangent, so a normal map on them uses
  the triangle's UV frame (#187). The loaders left a default tangent of
  (1, 0, 0), which both backends read as a valid vertex tangent. The frame
  was built on world +x, not on dP/du, as Arnold's `normal_map` does. glTF
  meshes keep their own tangents. No shipped preset reaches a tangent reader on
  an OBJ or PLY mesh: the 18 gate rows, 3 Bitterli wavefront rows and 15 SSS
  preset and probe rows are byte-identical. Embree was not rendered.

### Embree CPU Reference Backend

- Gave Embree's constant-UV triangles on SingleScatterSpecular materials the
  unit tangent frame for uv set 0, as Metal does (#145). Only Barcelona
  Pavilion materials have the flag. `barcelona_pavilion_day`, Embree/Metal:
  tiles pass / fail / unresolved 113 / 1 / 6 to 114 / 0 / 6, global 0.9984 to
  0.9998, tile (352,96) 0.8848 to 0.9894. Embree is byte-identical on `sponza`,
  `living_room`, `sun_temple` and `zero_day_measure_seven`.
- Ported source opacity and `interfacePrecision` spawn rules to Embree without
  source transport (#148), and removed its refusal and two stale warnings.
  Embree/Metal on local `sourceTransport=0` copies: `bedroom` 0.8524 to 1.0002
  (17/120 to 120/120 tiles), `kitchen` 1.0230 to 0.9986 (117/120 to 120/120),
  the five opacity probes 0.9792 to 0.9841 before and 0.9999 to 1.0000 after.
- Sampled textures that mix a nearest and a linear filter on Embree as Metal
  does (#149), and removed its refusal for them. Per-pixel error on the `nl`
  probe, near band: 0.1682 to 0.0001.
- Read Embree's anisotropic texture lookups at the hardware's level when the
  footprint exceeds `maxAnisotropy` (#158). The rule was measured on Metal:
  max(log2(minor), log2(major / maxAnisotropy)) within 0.02. Zero Day
  `measure_seven`, maxDepth 1, box at (370,190): 0.9073 to 1.0052. Other
  presets move by 0.01% or less on Embree; Zero Day's floor stays open (the
  full-depth tiles at (224,192) and (384,192) did not move).
- Gave Embree Metal's homogeneous medium without scattering (#159): segment
  attenuation, the medium's emission, NEE attenuation to positional lights, and
  the `baseline_emissive` demotion of `restir_di` and the hybrid. Sun plane
  with sigma_a 0.5: 2.33004 to 1.81348 (exact 1.81348). Quad lamp,
  `baseline_emissive`, sigma_a 0.5: 0.09218 against reference 0.09212.
- Gave Embree Metal's medium scattering (#160): pbrt-v3 free flight,
  Henyey-Greenstein phase function, volume NEE and the MIS rules after a medium
  scatter. `rect_big`, single scattering: 0.05902 against reference 0.059010.
  `sponza` in haze, Embree/Metal 1.0009 +-0.0009. Before, Embree gave 0 on
  `rect_big` and `env_single`. This replaces "Embree ignores volumes".
- Gave Embree Metal's hero-wavelength spectral transport, with the same
  refusals (#162). Red Lambert patch under a white environment, Metal / Embree:
  0.229 / 0.093 / 0.006 and 0.229 / 0.092 / 0.006 (Embree RGB on main gave
  0.300 / 0.000 / 0.000). `living_room` PASS 0.9999, 50/50 tiles;
  `caustics-glass-rectlight` PASS 0.9999, 22/22 tiles. Path-space caustics stay
  ignored on Embree.
- Ran Metal's plain RIS on Embree in `ris`, `ris_spatial`, `ris_temporal`,
  `ris_world` and `ris_regir` (#163). Before, Embree ran them as `legacy`. The
  reuse of the spatial, temporal, world and ReGIR modes is not ported; those
  modes render plain RIS and Embree says so. `living_room` `ris` Embree/Metal:
  0.9942 to 1.0007. Quad lamp: 0.9971 to 1.0000.
- Re-selected an Embree source shadow ray's medium at explicit-interior
  pass-throughs (#164). Probe with an absorbing glass slab, under-slab
  luminance Embree/Metal: 1.2216 to 1.0196; frame mean 1.0084 to 1.0003. No
  shipped scene reaches this case.
- Computed Embree mesh hits' world normals and tangents at the hit instead of
  keeping per-instance copies (#166). Images are byte-identical. Peak load
  memory, `barcelona_pavilion_day`: 17.16 GB to 13.70 GB; `emerald_square_day`
  20.32 GB to 19.93 GB; `bistro_full` 18.34 GB to 17.98 GB. No slowdown above 3%
  was resolvable on a noisy machine.
- Kept one mip level in Embree's copy of source opacity and bump textures, as
  Metal does (#168). Neither backend reads those levels. Images are
  byte-identical. The count of footprint-filtered Embree textures fell, for
  example `kitchen` 10 to 8 of 13.
- Matched Embree's anisotropic texture filter to Metal's sampler (#173). The
  minor axis is now the footprint's area over its major axis, and a texture with
  a side that is not a power of two is capped at 4x anisotropy. Zero Day Seven,
  maxDepth 1, Embree/Metal HW, floor tile (384,192): 1.1192 to 1.0022; floor
  tiles off by more than 3%: 11 to 7. The floor's mean at maxDepth 1 goes from
  1.0014 to 0.9821, because the inputs now equal Metal's and a gap in the glossy
  floor shading shows. #174 closes that gap (0.9821 to 0.9997). Full-depth tiles do not get worse
  on six presets. The Metal gate is 18/18 IDENTICAL.
- Kept Embree texture levels that were computed in float as bytes when every
  value is an 8-bit step (#177). A per-level table maps each byte back to the
  exact float, so images are byte-identical on 11 scenes. Peak load memory:
  `emerald_square_day` 19.93 GB to 18.25 GB, `bistro_full_sun_restir_alt`
  16.13 GB to 14.40 GB. The check costs about 1.3 s of CPU on Emerald and 0.6 s
  on Bistro. ORM levels that are not 8-bit steps stay float.
- Copied Embree's material textures that are decoded from files on six threads
  (#184). Before, each KTX2 or DDS file was decoded, mip-chained and converted
  one after another on one thread; on `emerald_square_day` that took 18.5 s of
  the load. The copies keep the order in which materials first name them, so
  the `[LoadMemory]` and `[MipChain]` lines do not change. The `[LoadMemory]`
  stage lines now log each stage's wall and CPU time. Load, 64x36, 1 spp, 3
  alternating pairs, before to after: `emerald_square_day` 13.1-13.9 s to
  4.6-5.1 s, `bistro_full_sun_restir_alt` 14.8-15.3 s to 6.8-7.2 s,
  `sun_temple` 4.7-4.9 s to 1.6-1.7 s, `barcelona_pavilion_day` 3.0-3.5 s to
  3.0-3.1 s (9 textures). Peak footprint rises by 0.6-1.1 GB (Emerald 17.53 GB
  to 18.34-18.50 GB, Bistro 14.38 GB to 15.25-15.48 GB, Sun Temple 6.48 GB to
  7.10-7.16 GB). Images are byte-identical on 11 scenes, and the Metal gate is
  18/18 IDENTICAL. The six copy threads do not follow `--threads`. A texture
  whose copy fails is now reported once, at the first material that names it.

### Importer

- Skipped FBX Models that the source file hides (#161). Assimp does not read
  `Visibility`, so hidden alternate states rendered over the visible geometry.
  The importer now reads the binary FBX file. `zero_day_measure_seven`, full
  depth: Embree/Metal HW 1.0050 to 0.9997, tiles 114/5/1 to 118/1/1; Metal HW
  frame x1.0747. Frame-mean shifts at 960x540: `zero_day_measure_one` 1.0232,
  `zero_day_measure_seven` 1.0752. ASCII FBX files are not read, and this is
  logged. The Zero Day packs are local-only and were not re-installed.
- Applied Cinema 4D polygon selections when importing (#169). A base mesh loses
  the triangles a `*_Polygon_Selection*` sibling covers. `zero_day_measure_seven`,
  full depth: SW/HW 0.9932 to 0.9999, Embree/HW 0.9997 to 1.0009, HW frame
  x0.9877. 59,980 triangles dropped on Seven and 17,640 on One. Selections that
  coincide with different materials are logged as unresolved and both kept (59
  on Seven, 6 on One). No pack was installed.
- Documented that the importer's DDS decode follows the GPU's block rules since
  #131 (#171). Comment only, no image change. Re-importing Zero Day
  `measure_one` writes KTX2 files that differ from the Nov 19 import in 223 of
  356 files: up to 2-3 8-bit steps in colour, up to 15 in the reconstructed Z of
  17 BC5 normal maps.

### Headless

- Added `--checkpointEverySamples=N` (#144). Every N samples the accumulation
  is saved to `<output>.accum` with a partial manifest, and
  `--resumeCheckpointJson` on that manifest renders only the remaining samples.
  Sponza resumes are byte-identical to uninterrupted renders on the megakernel
  (hardware and software) and the wavefront (hardware). Refused by name (exit
  code 2): direct-light modes other than `legacy`, `baseline_emissive` and
  `ris`; SVGF, ReSTIR GI, ReSTIR PT, path guiding, the radiance cache, caustic
  transport and the ML backend; `--validate-ris`, Embree, and runs without
  `--output`.
- Wrote a completed checkpoint manifest's output from its saved accumulation
  without rendering (#167). The finished `.accum` is now kept, and the completed
  manifest says `"resume_policy": "write_output_from_state"`. Sponza, 64 spp:
  resumed output byte-identical on megakernel HW and SW and wavefront HW,
  and with OIDN guides. A completed manifest without a state still renders the
  frame again.

### Build, Platform and Tests

- Linked the scene assets into the build directories and the app bundle
  instead of copying them (#170). The old build copied about 53 GB per
  destination, five times on the development machine (about 266 GB). After
  `touch shaders/common.metal`, the build takes 0.28 s. A real directory from
  an older build is left in place with a warning. A checkout without `assets/`
  still builds.
- Removed the ACEScg working colour space (#165). No setting selected it, and
  its matrix was transposed. The EXR files and the render metadata still say
  "Linear sRGB". Images are unchanged: the Metal gate is 18/18 IDENTICAL, and
  five Embree scenes are byte-identical.
- Shared each decoded glTF primitive's vertices and indices among its
  instances instead of copying them per instance (#178). Images are
  byte-identical on Metal (megakernel and wavefront, hardware and software) and
  Embree. Peak footprint, before to after: `barcelona_pavilion_day` Metal 16.46
  GB to 9.17 GB, Embree 13.70 GB to 6.44 GB; `emerald_square_day` Metal 15.23
  GB to 14.56 GB, Embree 18.25 GB to 17.53 GB; `zero_day_measure_seven` does not
  change. Pavilion on Embree loaded in 3.4 s before and 2.9 s after.
- Reserved Metal's packed triangle data, vertices and indices for what the
  packing loop appends (#181). The triangle data no longer grows by doubling,
  and vertices and indices are reserved once per geometry key. Only capacity
  changes: images are byte-identical on `barcelona_pavilion_day`,
  `zero_day_measure_seven` and `sponza`. `barcelona_pavilion_day`, Metal peak
  footprint: 9.18 GB to 5.43 GB. `zero_day_measure_seven` rose from 6.40 GB to
  6.57 GB while its live heap fell from 4.593 GB to 4.569 GB; this was not
  investigated further.
- Named the gates that keep the megakernel on its general pipeline, and logged
  that `causticTransportMode=path_space` adds no caustic paths (#176). This
  changes logs and help text only. The megakernel line now reads
  `hardware, general (gates on: causticTransportMode)`. The three Barcelona
  Pavilion presets set the mode and ran the general pipeline for no image
  change: `barcelona_pavilion_day` megakernel 54 ms/spp as shipped against
  11-20 ms/spp without the token, frame 1.0000 on all three presets. The Metal
  gate is 18/18 IDENTICAL.
- Folded the headless-only debug reads and the headless-only software trace in
  media out of the hardware megakernel's default-features pipeline (#182,
  #183), and gave the wavefront shade kernel its own default-features pipeline
  (#185). These apply to headless builds (`PT_DEBUG_TOOLS`); the GUI build
  already compiles them out.
  - #182 folds the three software-trace fallbacks and the normal-map and ORM
    debug knobs. A headless run that sets one takes the general pipeline, and
    the knob still works. Register spill of the integrate pipeline: 3472 B to
    2960 B.
  - #183 compiles out the rule that traced path rays inside a medium with the
    software BVH. Spill 2960 B to 1824 B (the GUI build: 1856 B). The rule
    fired in no scene tried: a marker build showed no marked pixel on 7 glass
    scenes, and those 7 are byte-identical to main. The general pipeline keeps
    the rule; on the default-features pipeline `--forcePureHWRTForGlass=0` no
    longer brings it back.
  - Time of the megakernel, 1280x720, 32 spp, against main before #182, with
    both: `sponza` 0.926, `living_room` 0.926, `sun_temple` 0.962,
    `f1_ferrari_sf1000_restir_photoshoot` 0.973. #182 alone: 0.945, 0.956,
    0.960 and 1.005.
  - #185 builds two more pipelines on first use: the wavefront shade kernel
    with the default-features constant set. A frame never waits on another
    pipeline's build; it keeps the general shade pipeline meanwhile.
    `[Wavefront] shade pipeline` logs which one ran. Headless wavefront time:
    `emerald_square_day` 0.961, `sponza` 0.912 of main.
  - Images: the megakernel is byte-identical on every gate row (rows with the
    Bitterli megakernel refuse on both sides). The wavefront is not
    byte-identical: with the gates folded, the compiler orders some arithmetic
    differently, so pixels change in their last bits and about 0.2% take
    another path. Frame means at 2 x 256 spp, 480x270, new/old: `sun_temple`,
    `emerald_square_day`, `bitterli_staircase`, `bitterli_spaceship` and
    `barcelona_pavilion_night` 1.00000. Under `MTLMathModeSafe` the results
    are byte-identical on `sponza` software.

### Documentation

- Pointed the README's release badge at "latest" (#154), then at a badge file
  served from the repository (#155). The SVG lives only in the public repo, so
  the badge shows only there.

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
