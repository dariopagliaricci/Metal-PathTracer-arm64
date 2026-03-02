# Contributing to Metal Path Tracer

Thank you for your interest in contributing. This document describes how to report
issues, propose changes, and submit code.

---

## Table of Contents

- [Scope and Platform Requirements](#scope-and-platform-requirements)
- [Filing Issues](#filing-issues)
- [Proposing Changes](#proposing-changes)
- [Code Style](#code-style)
- [Pull Request Workflow](#pull-request-workflow)
- [Design Invariants](#design-invariants)

---

## Scope and Platform Requirements

Metal Path Tracer targets **macOS on Apple Silicon (arm64)** exclusively. The Metal
ray tracing API is only available on Apple Silicon with macOS 13 or later. Contributions
that introduce platform-specific code for other operating systems or architectures are
outside scope and will not be merged.

**Prerequisites for building from source:**

- macOS 13 (Ventura) or later on Apple Silicon (M1 or newer)
- Xcode 15 or later (Metal Shading Language compiler)
- CMake 3.24 or later
- A C++20-capable compiler (Apple Clang 15+)

Optional dependencies (OIDN, Embree) are documented in the README.

---

## Filing Issues

Use the [GitHub issue tracker](../../issues) to report bugs, request features, or ask
questions.

**Bug reports should include:**

- macOS version and Apple Silicon chip model (e.g., M2, M3 Pro)
- Xcode and CMake versions (`xcodebuild -version`, `cmake --version`)
- Steps to reproduce with the smallest possible scene or command line
- Whether the issue reproduces with `--enableSoftwareRayTracing=0` (HWRT path) and
  `--enableSoftwareRayTracing=1` (SWRT path), or only one mode
- If relevant, whether it reproduces on the Embree backend (`--backend=embree`)
- Output of `PathTracerHeadless --help` if the issue is a CLI flag problem
- Any error messages or Metal GPU capture logs

**Feature requests should include:**

- A clear description of the rendering problem or workflow gap the feature addresses
- Whether the feature is relevant to both backends or backend-specific
- References to any relevant papers or prior implementations, if applicable

---

## Proposing Changes

Before writing code for a significant change, open an issue first to discuss the
approach. This is especially important for:

- New material types or BSDF models
- Changes to the shared material encoding struct
- New sampling strategies that affect the MIS balance
- Any modification to the HWRT/SWRT rendering pipeline
- Changes to the glTF loading pipeline

For small, well-scoped fixes (typos, build system corrections, documentation updates,
single-line bug fixes) you may open a PR directly without a prior issue.

---

## Code Style

### C++ (host code, `src/`)

- C++20; no platform-specific extensions beyond Metal Objective-C++ where required
- 4-space indentation; no tabs
- `snake_case` for variables and functions; `PascalCase` for types and structs
- Keep includes minimal and ordered: standard library, system frameworks, third-party,
  project headers
- No `using namespace std;` in header files
- Prefer `const` and `constexpr` where the value is known at compile time
- Avoid raw `new`/`delete`; use RAII wrappers or Metal-managed objects

### Metal Shading Language (`shaders/`)

- Match indentation and naming conventions of the existing shader files
- Prefer explicit type qualifiers (`device`, `constant`, `threadgroup`)
- Avoid dynamic loops with data-dependent bounds in inner shading kernels unless
  already established in the codebase
- Kernel entry points are `snake_case`; helper functions follow the same convention

### General

- Do not introduce warnings; the project compiles cleanly with `-Wall -Wextra`
- Do not check in generated files (`.air`, `.metallib`, build artefacts)
- Do not check in large binary assets; reference existing assets in `assets/` or
  Khronos sample scenes

---

## Pull Request Workflow

1. Fork the repository and create a feature branch from `main`:
   ```
   git checkout -b feature/your-feature-name
   ```

2. Make your changes. Ensure the project builds cleanly:
   ```
   cmake -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel
   ```

3. Run the smoke test to verify basic renderer correctness:
   ```
   tests/public/headless_smoke_test.sh
   ```
   If you have Embree available:
   ```
   tests/public/headless_smoke_test.sh ./build-embree/PathTracerHeadless --backend=embree
   ```

4. If your change touches rendering output, verify that HWRT and SWRT produce
   equivalent results (see [Design Invariants](#design-invariants) below).

5. Push your branch and open a pull request against `main`. The PR description should:
   - Summarise what changed and why
   - Note whether the change is backend-specific or shared
   - Include before/after render comparisons for visual changes (64×64 PNG or PPM
     at 64 spp is sufficient for a quick comparison)

6. CI will build both the default configuration and the Embree configuration on
   `macos-14`. Both must pass before a PR can be merged.

7. PRs are reviewed by the maintainer. Expect one round of feedback. Please respond to
   review comments within a reasonable time; stale PRs (no activity for 60 days) may
   be closed.

---

## Design Invariants

These invariants are **core to the project's correctness and research utility** and
must be preserved by all contributions.

### 1. HWRT/SWRT Output Parity

The hardware ray tracing path and the software BVH path must produce equivalent rendered output
within a small RMSE tolerance on linear HDR images. This cross-backend parity is the
primary correctness guarantee of the project and is cited in the associated paper.

Any change that causes one backend to diverge from the other in a way that is not
justified by an intentional algorithmic difference must be flagged in the PR and
discussed before merging.

### 2. Shared Material Encoding

All eight material types are encoded in a single fixed-size `MaterialDescriptor` struct
shared between CPU host code and Metal shaders. Changes to this struct affect both
backends simultaneously. Modifications must:

- Maintain binary compatibility with existing scene serialisation, or explicitly
  document a breaking change and bump the relevant version marker
- Preserve the same material evaluation in both the HWRT and SWRT shading kernels

### 3. GGX VNDF Sampling

All specular BRDFs use GGX microfacet models with visible-normal distribution function
(VNDF) importance sampling. Do not replace VNDF sampling with hemisphere sampling or
other approximations in the main rendering path; these would break energy conservation
and degrade convergence.

### 4. MIS Balance

Direct lighting uses power-heuristic MIS combining BSDF and area-light sampling. New
light types or sampling strategies must correctly participate in this MIS framework;
unbalanced additions will cause bias.

### 5. macOS/Apple Silicon Only

This renderer is intentionally scoped to Apple Silicon and Metal. Do not add CUDA,
Vulkan, OpenGL, or cross-platform abstraction layers. The single-platform focus is
what makes the Metal RT validation meaningful.

---

## Questions

If you are unsure whether a proposed change is in scope or compatible with the design
invariants, open a [GitHub issue](../../issues) with the label **question** before
investing time in implementation.
