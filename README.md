# Path Tracer Metal (v2.0)

A physically based, progressive path tracer for **macOS + Apple Silicon**, written in C++ and Metal.
It started as a *"Ray Tracing in One Weekend"* clone and evolved into a production-grade renderer with:

- Hardware **ray tracing** (Metal RT) and a **software BVH** reference path tracer
- **glTF 2.0** scene loading with full PBR metallic-roughness materials and texture support
- **MNEE** caustics (Manifold Next Event Estimation)
- HDR environment lighting with importance sampling and MIS
- Extended material models (plastic, subsurface, car paint, PBR metallic-roughness)
- Intel® Open Image Denoise integration
- Headless CLI rendering with Metal and optional Embree CPU backends
- **Presentation Mode** for clean, UI-free demo display

> This README describes the **v2.0 public release** of the Metal path tracer core.
> Heavy assets (statues, high-res HDRIs) are provided as a separate download.

---

## Features

### Rendering & performance

- **Progressive path tracing** with temporal accumulation
- **Dual backends**:
  - **Hardware Ray Tracing (HWRT)** using Metal's ray tracing API (TLAS/BLAS)
  - **Software Ray Tracing (SWRT)** using a CPU-built BVH (tinybvh-style) and compute kernels
- **Full HWRT/SWRT parity** — both backends produce identical results for all material types including glass and subsurface scattering
- **Configurable path depth** and Russian Roulette termination
- **Internal render scale** (0.5×–2.0×) for trading quality vs performance
- Accurate **environment map sampling** (importance sampling + MIS)
- **Specular Next Event Estimation (NEE)** for direct specular connections
- **MNEE** (Manifold Next Event Estimation) caustics — toggleable at runtime and via CLI

### Materials & lighting

- Physically based material model implemented in `shaders/pathtrace.metal`:
  - Lambertian diffuse
  - Conductor metal with `eta`/`k` and roughness
  - Dielectric glass with IOR, absorption, and coat support
  - Emissive / diffuse light materials
  - **Plastic** with clear coat, tint, and absorption
  - **Subsurface scattering** (random-walk style) for marble/wax/jade-like materials
  - **Car paint** with configurable base layer and flake controls
  - **PBR Metallic-Roughness** — full glTF 2.0 material model (baseColor, metallic, roughness, occlusion, emissive, transmission)
- Analytic primitives: spheres, rectangles, boxes
- Triangle meshes via OBJ/PLY (powered by tinyobjloader + tinyply)
- **glTF 2.0 / GLB** mesh and material loading (custom loader, no extra runtime dependency)
  - Multi-UV-set textures (TEXCOORD_0 / TEXCOORD_1) with KHR_texture_transform support
  - Mip-mapped material textures with per-sampler filtering (glTF sampler descriptors)
  - Thin-dielectric transmission fallback for scenes lacking explicit volume data
  - Camera auto-import from embedded glTF camera nodes
  - MikkTSpace-compliant tangent generation for correct normal-mapped shading
- HDR environment maps with rotation + intensity overrides

### Post-processing & output

- **Intel® Open Image Denoise (OIDN)** 2.3.3 integration
  - AOV path supports a sample-count channel for better denoising
- **Working color space**: Linear sRGB or ACEScg (ACES AP1) — selectable per-render
- Multiple tonemappers:
  - Linear, ACES (Fitted or Simple variant), Reinhard (configurable white point), Hable
- **Bloom** post-processing — luminance threshold, intensity, and radius controls
- GUI EXR export:
  - **Save EXR…** button in the ImGui "Output / Export" panel
  - Writes a **linear HDR EXR** at internal render resolution
  - Optional multilayer EXR with sample count AOV
- Headless output formats:
  - EXR (linear)
  - PNG (LDR, tonemapped)
  - PFM (HDR float32)
  - PPM (debug/simple)

### Developer tooling

- **Headless renderer** (`PathTracerHeadless`) for batch/offline rendering
  - **Metal GPU backend** (default)
  - **Embree CPU backend** (optional, enabled via `-DPATH_TRACER_ENABLE_EMBREE=ON`)
- ImGui-based UI overlay:
  - Real-time stats (GPU time, BVH stats, samples/min, backend mode, etc.)
  - Camera + renderer controls
  - Render scale, tonemapping, denoiser toggle, bloom, EXR export
  - **Scene panel**: live material editing with per-material reset
  - **Object panel**: mesh transform editing via ImGuizmo 3D gizmos (translate / rotate / scale, local/world space)
- **Presentation Mode** — borderless fullscreen, minimal overlay, resolution lock (720p / 1080p), target screen selection


## Renderings Gallery

<div align="center">

![Ajax Alloys Render](screenshots/ajax00.jpg)

![Ajax Other Alloys Render ](screenshots/ajax01.jpg)

![Ajax Car Paint Render](screenshots/car-paint-validation.jpg)

![Hygieia Silver - Glass - Gold Render](screenshots/hygieia_HWRT.jpg)

![Hygieia Alloys Render](screenshots/hygieia_metal.jpg)

![Jason Alloys Render](screenshots/jason-alloys.jpg)

![Lucy Plastic Render](screenshots/lucy-plastic.jpg)

![Lucy Glass Render](screenshots/lucy-glass-HWRT.jpg)

![Ajax Marble - Wax Render](screenshots/marble-wax-validation.jpg)

![Ajax Plastic Render](screenshots/plastic.jpg)

![Sci-Fi Damaged Helmet Render](screenshots/damaged-helmet-03.jpg)

![BoomBox Render](screenshots/boombox.jpg)

![Lantern Render](screenshots/lantern.jpg)

![Flight Helmet Render](screenshots/flight-helmet.jpg)

![Stanford Dragon Render](screenshots/stanford_dragon.jpg)

</div>

---

### Ray Tracing Acceleration

The path tracer supports two acceleration structures:

**Hardware Ray Tracing** (Apple Silicon):
- Uses Metal's native ray tracing API
- TLAS (top-level) over mesh instances
- BLAS (bottom-level) for triangle meshes
- Automatic BVH construction on GPU
- Lower traversal overhead than software

**Software Ray Tracing** (Fallback):
- Custom BVH with Surface Area Heuristic (SAH) construction
- Linear BVH layout for cache-friendly traversal
- Stack-based traversal with 128-entry stack
- Early exit optimization for shadow rays
- Per-ray statistics: nodes visited, primitive tests, coherency metrics

Real-time statistics displayed in ImGui:
- Average nodes visited per ray
- Average primitive tests per ray
- Shadow ray early exit percentage
- Both-children-visited percentage (traversal coherency)

## Requirements

- **macOS:** 12.0 or later
- **Xcode:** 13.0 or later (for Metal compiler)
- **CMake:** 3.24 or later
- **Apple Silicon:**
  - M3 and later: Hardware ray tracing acceleration enabled
  - M1/M2: Software ray tracing (fallback)
  - **Important** Apple exposes `MTLDevice.supportsRaytracing` even on M1/M2, so the renderer still launches the HWRT pipeline and Metal silently emulates it on the GPU cores. The ImGui stats therefore show "Hardware Ray Tracing" as *active* even though traversal runs in software. Toggle **Software Ray Tracing** in the Settings panel or pass `--enableSoftwareRayTracing=1` if you want to stay on the pure BVH path for debugging or parity.
  - **Large-scene memory guard:** the renderer refuses HWRT builds if the estimated BLAS/TLAS size plus scratch exceeds `recommendedMaxWorkingSetSize`. This budget is device-specific and **not** equal to total unified memory, so very large mesh scenes can be blocked even on machines with plenty of RAM. Practical implication: scenes with tens of millions of triangles (e.g. `lucy-plastic.scene`) or multiple instances of heavy meshes can exceed the budget. In our testing on an M1 Pro 16 GB, two `lucy-scaled` instances render, while three can exceed the budget and be refused. Higher-end GPUs (e.g. M1/M2 Max or M3+) typically have a larger budget, but limits still apply.

### Toolchain

- CMake ≥ 3.24
- Xcode / Apple Clang with C++20 and ObjC++

### Dependencies (vendored in `external/`)

- [Dear ImGui](https://github.com/ocornut/imgui)
- [ImGuizmo](https://github.com/CedricGuillemet/ImGuizmo) — 3D gizmo widgets for object transform editing
- [Intel® Open Image Denoise](https://www.openimagedenoise.org) 2.3.3
- [MikkTSpace](http://www.mikktspace.com) — industry-standard tangent space for normal map baking
- [tinybvh](https://github.com/jbikker/tinybvh)
- [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader)
- [tinyply](https://github.com/ddiakopoulos/tinyply)

---

## Building

### CMake (Command-line)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This will produce:
- `build/PathTracer.app` – GUI application
- `build/PathTracerHeadless` – headless CLI renderer
- Shaders and assets copied next to the binaries (via CMake post-build steps)

### CMake Options

| Option | Default | Description |
|---|---|---|
| `PATH_TRACER_ENABLE_EMBREE` | `OFF` | Build the Embree CPU headless backend |
| `PT_DEBUG_TOOLS` | `OFF` | Enable HWRT/SWRT debug tooling (path debug, parity assertions) |
| `PT_MNEE_SWRT_RAYS` | `OFF` | Force MNEE rays to use SWRT in the HWRT path (debug) |
| `PT_MNEE_OCCLUSION_PARITY` | `OFF` | Compare HWRT vs SWRT MNEE visibility (debug) |
| `STRICT` | `OFF` | Enable strict compiler warnings (`-Wall -Wextra -Werror`) |
| `M0_TESTS` | `OFF` | Enable milestone golden-image tests |

### Running the GUI

Run the app bundle to watch the progressive path tracer converge:

```bash
open build/PathTracer.app
```

Or (for debugging purposes):

```bash
./build/PathTracer.app/Contents/MacOS/PathTracer
```

Pass `--presentation=1` to start directly in Presentation Mode (borderless fullscreen, no UI panels):

```bash
./build/PathTracer.app/Contents/MacOS/PathTracer --presentation=1
```

### Controls and UI Panels

**Camera:**
- Typical FPS/orbit controls (exact bindings depend on your local setup)

**Settings Panel:**
- Backend: Hardware Ray Tracing vs Software BVH
- Samples per frame
- Max path depth, Russian Roulette toggle
- Specular NEE toggle
- MNEE toggle (Manifold Next Event Estimation caustics)
- Denoiser on/off
- Working color space (Linear sRGB / ACEScg)
- Tonemapper (Linear / ACES Fitted / ACES Simple / Reinhard / Hable)
- Exposure (in stops)
- Bloom (enable, threshold, intensity, radius)

**Scene Panel:**
- Live material editor — modify base color, roughness, metallic, emission, IOR, coat, SSS, and all other per-material parameters
- Per-material reset to scene defaults
- Object list — select any mesh and edit its world transform with ImGuizmo 3D gizmos

**Output / Export Panel:**
- Save EXR… button
- Saves a linear EXR snapshot of the current accumulation
- Outputs to `./renders/render-YYYYMMDD-HHMMSS.exr` (see limitations)

**Performance Panel:**
- FPS and GPU/CPU timings
- Sample count and samples/min
- BVH statistics (nodes, prims, average nodes/leaf tests per ray)
- Intersection mode label: "Hardware Ray Tracing" vs "Software Ray Tracing"

**Presentation Mode:**
- Toggle via the View menu or `--presentation=1` CLI flag
- Hides all UI panels and shows only a minimal status overlay
- Supports borderless fullscreen or maximized window
- Resolution lock at 720p or 1080p
- Automatically resets accumulation on toggle

## Headless CLI Rendering

For offline renders without a GUI, use the `PathTracerHeadless` CLI:

```bash
build/PathTracerHeadless --scene=hygieia-other --enableSoftwareRayTracing=1 \
    --width=1920 --height=1080 --sppTotal=4096 \
    --output=renders/hygieia.exr
```

**Key flags:**

- `--scene=<id-or-path>` — Required. Scene identifier (from assets) or path to a `.scene` / `.gltf` / `.glb` file.
- `--output=<path>` — Output filename. Defaults to `renders/<scene>_<WxH>.<format>`.
- `--width/--height` — Override render resolution (clamped to ≥ 8).
- `--sppTotal` — Total samples to accumulate (default 1024).
- `--maxDepth` — Override maximum path depth.
- `--seed` — Fixed RNG seed (0 = random).
- `--envRotation`, `--envIntensity` — Environment overrides (degrees / multiplier).
- `--enableSoftwareRayTracing[=0|1]` — Force software ray tracing instead of hardware acceleration.
- `--enableMnee[=0|1]` — Enable MNEE caustics (default: disabled).
- `--denoiser[=0|1]` — Enable/disable Intel OIDN denoising (default: enabled).
- `--tonemap`, `--exposure` — Override tonemapping/exposure for LDR outputs.
- `--format=<exr|png|pfm|ppm>` — Output format (default EXR, linear HDR).
- `--backend=<metal|embree>` — Select headless backend: `metal` (default, GPU) or `embree` (CPU, requires `-DPATH_TRACER_ENABLE_EMBREE=ON`).
- `--enableEmbree[=0|1]` — Alias for `--backend=embree`.
- `--threads=<int>` — Worker thread count for the Embree backend.
- `--verbose` — Print per-frame progress.

All overrides apply **after** scene parsing. HDR formats (`exr`, `pfm`) are saved linear.


## Scene Format

Scenes are described with plain-text `.scene` files under `assets/`. Scenes support spheres, rectangles, boxes, triangle meshes (OBJ/PLY), and **glTF 2.0 / GLB** assets:

```
camera target=0,0,0 distance=10 yaw=0 pitch=0 vfov=40 defocusAngle=0.6 focusDist=10
renderer samplesPerFrame=8 maxDepth=20 envRotation=30 envIntensity=1.5 enableSoftwareRayTracing=0 width=1920 height=1080

background env=assets/HDR/studio.hdr

material type=metal albedo=0.9,0.9,0.9 fuzz=0.05
material type=diffuse_light albedo=5,5,5
mesh path=models/dragon.obj material=0 translate=0,0,0 scale=1,1,1 rotate=0,180,0
sphere center=0,-1001,0 radius=1000 material=1
```

**glTF / GLB loading** — pass a `.gltf` or `.glb` file as the mesh path; materials are imported automatically:

```
mesh path=models/DamagedHelmet.glb translate=0,0,0 scale=1,1,1 rotate=0,0,0
```

**Geometry Types**:
- `sphere center=X,Y,Z radius=R material=INDEX`
- `rectangle axis=MIN,MAX perpendicular=VALUE offset=MIN,MAX material=INDEX`
- `box min=X,Y,Z max=X,Y,Z material=INDEX [translate] [scale] [rotate]`
- `mesh path=FILE material=INDEX [translate] [scale] [rotate]` — OBJ, PLY, glTF, or GLB

**Material Types**:
- `type=lambert` — Matte diffuse
- `type=metal` — Reflective with Fresnel (optional `fuzz`)
- `type=dielectric` — Glass/clear (optional `refractiveIndex`)
- `type=diffuse_light` — Emissive for area lights
- `type=plastic` — Clear-coat diffuse with tint and absorption
- `type=subsurface` — Random-walk BSSRDF (marble/wax/jade)
- `type=carpaint` — Metallic base + procedural flake layer
- glTF meshes import `PbrMetallicRoughness` materials automatically

**Backgrounds**:
- `background env=path/to/file.hdr` — HDR environment map
- `background solid=R,G,B` — Solid color
- `background gradient=R,G,B` — Sky gradient
- `envRotation` (degrees) and `envIntensity` (multiplier) can be supplied via the `renderer` block to orient and scale the environment map.

**Renderer Overrides**:
- `samplesPerFrame`, `maxDepth`, `tonemap`, `exposure`, `seed`
- `envRotation` (degrees) and `envIntensity` (multiplier)
- `enableSoftwareRayTracing[=0|1]`
- `enableMnee[=0|1]`
- `width` / `height` (clamped to ≥ 8)
- `gltfViewerCompatibilityMode` / `gltfCompat` — opt-in glTF viewer parity behavior
- `gltfThinWalledFallback` / `gltfThinFallback` — treat transmission-only materials as thin dielectrics
- `gltfEmissiveScale` — extra emissive multiplier for glTF scenes
- `gltfCompatLinearBaseColor` / `gltfCompatLinearEmissive` — treat textures as linear (debug/compat)

OBJ files should provide vertex positions and may include normals/UVs for better shading. glTF files are loaded with MikkTSpace-compliant tangent generation.

## Release

[![Latest Release](https://img.shields.io/github/v/release/dariopagliaricci/Metal-PathTracer-arm64)](https://github.com/dariopagliaricci/Metal-PathTracer-arm64/releases/latest)

## Assets Pack

The scenes (Hygieia statue, Ajax bust, higher-res HDRIs, detailed dragon meshes) live in a separate asset pack to keep the repo size reasonable:

- Download `Metal-PathTracer-Assets.zip` from the [link provided in the GitHub release](https://drive.google.com/file/d/1fbB77stxZzF48T0oHLZS5FIInNqcGZT2/view?usp=share_link)
- Copy/replace the `assets` folder in the root of the project


## Known limitations

**EXR save UX:**
- The ImGui "Save EXR…" button currently:
  - Saves immediately to `./renders/render-YYYYMMDD-HHMMSS.exr`
  - Does not show a native file dialog

**Platform support:**
- Binaries and OIDN libraries in `external/oidn/lib` are built for Apple Silicon (arm64)
- Intel macs and non-macOS platforms are not supported in this configuration

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

### Core Ray Tracing Foundation
- [Ray Tracing in One Weekend Series](https://raytracing.github.io/) - Peter Shirley
  - Introduces fundamental path tracing concepts and scene format

### Physically-Based Rendering
- [Physically Based Rendering: From Theory to Implementation](https://www.pbr-book.org/) - Matt Pharr, Wenzel Jakob, and Greg Humphreys
  - GGX microfacet BRDF implementation
  - Multiple Importance Sampling (balance heuristic)
  - Environment map importance sampling using alias method
  - Conductor and dielectric Fresnel equations

### Key Papers and Techniques
- **Walter et al. 2007** - "Microfacet Models for Refraction through Rough Surfaces" (EGSR 2007)
  - GGX distribution and Smith masking functions
- **Veach & Guibas 1995** - "Optimally Combining Sampling Techniques for Monte Carlo Rendering" (SIGGRAPH 1995)
  - Multiple Importance Sampling theory and balance heuristic
- **Hanika et al. 2015** - "Manifold Next Event Estimation" (EGSR 2015)
  - MNEE caustics algorithm

### Tonemapping Operators
- **Stephen Hill & Krzysztof Narkowicz** - ACES fitted approximation
- **John Hable** - Uncharted 2 tonemap operator (GDC 2010 presentation "Filmic Tonemapping for Real-time Rendering")
- **Erik Reinhard** - Reinhard tone mapping operator

### Standards & Formats
- [glTF 2.0 Specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) - Khronos Group
- [MikkTSpace](http://www.mikktspace.com/) - Morten Mikkelsen — tangent space standard

### Dependencies and Libraries
- [Intel Open Image Denoise (OIDN)](https://www.openimagedenoise.org/) - ML-based denoising (Apache 2.0)
- [Dear ImGui](https://github.com/ocornut/imgui) - Immediate mode GUI (MIT)
- [ImGuizmo](https://github.com/CedricGuillemet/ImGuizmo) - 3D gizmo widgets (MIT)
- [TinyBVH](https://github.com/jacco/tinybvh) - BVH construction (MIT)
- [TinyObjLoader](https://github.com/tinyobjloader/tinyobjloader) - OBJ parsing (MIT)
- [MikkTSpace](http://www.mikktspace.com/) - Tangent space generation (zlib/libpng)

### Platform
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) - Apple
