---
title: 'Path Tracer Metal: A Dual-Backend Physically Based Path Tracer for Apple Silicon'
tags:
  - path tracing
  - physically based rendering
  - Apple Silicon
  - Metal
  - ray tracing
  - C++
authors:
  - name: Dario Pagliaricci
    orcid: 0009-0007-4663-1325
    affiliation: 1
affiliations:
  - name: Universidad Nacional de Río Cuarto (UNRC), Río Cuarto, Argentina
    index: 1
date: 1 March 2026
bibliography: paper.bib
---

# Summary

Metal Path Tracer is an open-source, production-grade physically based progressive
path tracer targeting macOS on Apple Silicon (`arm64`). Written in C++20 and Metal
Shading Language, it provides two ray tracing backends that share a common material
encoding and shading pipeline: a hardware-accelerated backend exploiting Apple's
Metal ray tracing API with GPU-native TLAS/BLAS acceleration structures (HWRT),
and a reference software backend using a CPU-built surface-area-heuristic BVH
dispatched through Metal compute kernels (SWRT). Both backends are validated to
produce equivalent output within an RMSE tolerance on linear HDR renders, enabling
reproducible cross-backend comparison on the same hardware.

The renderer supports eight physically based material models, glTF 2.0/GLB scene
loading with full PBR metallic-roughness materials and MikkTSpace tangent
generation [@mikkelsen2008], Manifold Next Event Estimation (MNEE) for specular
caustics [@hanika2015], power-heuristic Multiple Importance Sampling (MIS)
[@veach1995], alias-table HDR environment importance sampling [@vose1991], Intel
Open Image Denoise (OIDN) 2.3.3 integration, and output in EXR, PNG, PFM, and PPM
formats. An interactive `PathTracer.app` (ImGui-based) and a headless CLI
`PathTracerHeadless` are provided, with an optional Intel Embree CPU backend for
cross-validation. The project is built with CMake and carries an MIT license.

# Statement of Need

Apple Silicon (M1 through M4 and beyond) introduced dedicated hardware ray tracing
units alongside a tiled GPU architecture. Apple's Metal ray tracing API — available
since macOS 13 on Apple Silicon — exposes TLAS/BLAS hardware acceleration analogous
to DirectX Raytracing (DXR) and Vulkan Ray Tracing. Despite this hardware
availability, no open-source, feature-complete path tracer targets Metal RT as a
primary rendering backend.

Existing research renderers do not fill this gap. `pbrt-v4` [@pharr2023] renders on
GPU via CUDA/OptiX and requires NVIDIA hardware. Mitsuba 3 [@jakob2022] provides CPU
and CUDA backends but no Metal support. Tungsten [@bitterli2015] is CPU-only. Blender
Cycles introduced a Metal compute backend in 2022 but is tightly coupled to Blender's
data model and is not designed for research experimentation or standalone use. Apple's
own Metal sample code demonstrates the API in isolation but provides no complete
renderer, material system, or scene format.

This absence creates barriers for three communities: (1) graphics researchers working
exclusively on Apple hardware who cannot access CUDA-dependent renderers; (2) educators
teaching physically based rendering who need platform-native, buildable reference
implementations; (3) developers evaluating Metal RT for production pipelines who lack
a validated reference combining hardware–software output parity for independent
correctness and performance assessment.

Metal Path Tracer addresses all three needs by providing a Metal-native HWRT renderer
validated against a documented SWRT reference, a comprehensive material and
light-transport system, glTF 2.0 compatibility, OIDN denoising, and an MIT license
permitting unrestricted academic and commercial use.

# State of the Field

Production and research path tracers can be grouped by platform target. CPU-only
frameworks (Embree [@wald2014], PBRT scalar mode, Mitsuba 3 scalar mode) offer
flexibility and serve as correctness references but do not exercise GPU-native ray
tracing. GPU renderers based on CUDA/OptiX (`pbrt-v4` GPU mode, Falcor [@kallweit2022])
require NVIDIA hardware and are unavailable on Apple platforms. Vulkan-based pipelines
target Windows and Linux.

Among Metal-native work, Apple's own Metal sample code provides isolated API
demonstrations, and Blender Cycles provides a Metal compute backend, but neither
constitutes a standalone, research-grade renderer with a documented feature set and
public API. The RTOW book series [@shirley2020] provides pedagogical CPU path tracers
that have been ported to Metal in tutorial form, but these stop well short of
production features.

To the author's knowledge, no existing open-source path tracer combines all of:
(1) HWRT via Metal's native acceleration structure API alongside a validated SWRT
reference in the same codebase; (2) MNEE caustics in Metal Shading Language;
(3) a complete glTF 2.0 PBR material pipeline on macOS; and (4) OIDN 2.x integration
using its Metal device backend. Metal Path Tracer provides this combination as a
documented, installable software package.

# Software Design

## Dual-Backend Architecture

The renderer provides two ray tracing backends sharing a common material encoding and
shading pipeline. The **HWRT backend** constructs per-mesh bottom-level acceleration
structures (BLAS) using Metal's `MTLAccelerationStructure` API and assembles them into
a top-level structure (TLAS), issuing rays via Metal's `intersect()` intrinsic. A
memory guard on `MTLDevice.recommendedMaxWorkingSetSize` prevents over-allocation and
degrades gracefully to SWRT on constrained devices.

The **SWRT backend** builds a surface-area-heuristic BVH [@macdonald1990] on the CPU,
uploads nodes and indexed geometry as Metal buffers, and traverses the hierarchy in a
compute kernel. Both kernels read from the same material, texture, and environment
buffers. A debug build mode (`PT_DEBUG_TOOLS`) enables per-pixel output assertions
between backends. Backend selection is configurable at launch and switchable at runtime.
Full implementation is in `src/renderer/` and `shaders/pathtrace.metal`.

## Material System

Eight physically based material types are encoded in a shared fixed-size descriptor:
Lambertian diffuse; GGX conductor (complex-index Fresnel with per-channel $\eta/k$);
dielectric glass (exact Fresnel, Beer–Lambert absorption, optional coat); plastic with
clear coat; random-walk subsurface scattering; car paint with procedural flake normals;
diffuse emitter; and PBR metallic-roughness [@khronos2022] including baseColor, metallic,
roughness, occlusion, emissive, and transmission. All specular BRDFs use GGX microfacets
[@walter2007] with visible-normal distribution function (VNDF) importance sampling
[@heitz2018]. The PBR pipeline supports multi-UV-set textures and
`KHR_texture_transform` UV transforms; normal maps are applied using MikkTSpace-compliant
tangent frames [@mikkelsen2008].

## Sampling and Light Transport

Direct lighting uses power-heuristic MIS [@veach1995] combining BSDF and area-light
sampling. HDR environment maps are importance-sampled via marginal-conditional alias
tables [@vose1991] for O(1) sample generation. Specular NEE handles direct glossy
interactions. For caustics, MNEE [@hanika2015] constructs specular transport paths
through refractive interfaces via manifold walking, implemented in Metal Shading Language
and exposed as both a runtime toggle and a CLI flag. Standard firefly suppression
(luminance and specular-tail clamping) and Russian roulette termination are also provided.
Full light-transport source is in `shaders/pathtrace.metal` and `shaders/mnee.metal`.

## glTF 2.0 Pipeline

A custom glTF 2.0/GLB loader imports mesh geometry, PBR materials, texture coordinates,
`KHR_texture_transform` UV transforms, per-sampler filtering, camera nodes, and scene
hierarchy. MikkTSpace tangent generation is applied at load time to ensure correct
normal-mapped shading. The loader supports both embedded (GLB) and external textures
without additional runtime dependencies.

## Post-Processing and Output

Progressive accumulation stores per-pixel radiance sums and sample counts in separate
`MTLTexture` instances, enabling refinement across frames without resampling overhead.
The display pipeline applies OIDN 2.3.3 denoising [@intel_oidn] with optional albedo
and normal AOVs, selectable tone mapping (ACES Fitted [@hill2014], ACES Simple,
Reinhard, Hable/Uncharted 2, Linear), bloom post-processing, and selectable working
color space (linear sRGB or ACEScg). Outputs are written as linear-HDR EXR, LDR PNG,
HDR PFM, or PPM.

# Research Impact Statement

Metal Path Tracer provides a validated, open-source reference for GPU path tracing on
Apple Silicon. Publishing verified HWRT and SWRT backends in a single codebase enables
controlled experiments on Metal RT performance and correctness that are otherwise
impractical without access to NVIDIA hardware. The glTF 2.0 importer with full PBR
support allows standardised, interoperable scenes, reducing the barrier to reproducing
results across different renderers.

An open-source implementation of MNEE in Metal Shading Language is included, providing
a reproducible reference for caustic rendering research on Apple hardware. The MIT
license and CMake build system permit integration as a subcomponent in larger research
toolchains. The headless CLI with deterministic RNG seeding supports automated
benchmarking and CI-driven regression testing. The built-in asset pack (OBJ/PLY meshes,
HDR environment maps, and Khronos sample glTF scenes) allows reviewers and researchers
to reproduce renders immediately after build, without external downloads.

# AI Usage Disclosure

Portions of this paper were drafted with assistance from Claude (Anthropic), a large
language model. All technical descriptions, algorithmic claims, and design decisions
were independently verified against the source code by the author. No AI-generated
content was accepted without manual verification. The Metal Path Tracer software was
developed without AI assistance.

# Acknowledgements

The author thanks the developers of the open-source libraries incorporated in this
project: Dear ImGui (Omar Cornut), ImGuizmo (Cédric Guillemet), tinybvh (Jacco Bikker),
tinyobjloader (Syoyo Fujita), tinyply (Dimitri Diakopoulos), MikkTSpace (Morten S.
Mikkelsen), and Intel Open Image Denoise. Special thanks to the Khronos Group for the
glTF 2.0 specification and sample assets.

# References
