# Changelog

All notable changes to this project are documented in this file.

## [v2.0.0] - 2026-03-01

### Added
- glTF 2.0 / GLB loader with PBR metallic-roughness material support.
- Multi-UV texture support (`TEXCOORD_0` / `TEXCOORD_1`) and `KHR_texture_transform` handling.
- Camera import from glTF camera nodes.
- MikkTSpace tangent generation for normal-mapped glTF assets.
- MNEE (Manifold Next Event Estimation) caustics path.
- Bloom post-processing controls.
- ACEScg working color-space option.
- Headless output formats: PNG, PFM, and PPM (in addition to EXR).
- Optional Embree CPU backend (`PATH_TRACER_ENABLE_EMBREE=ON`) with `EmbreeSmokeTest` target.
- Presentation Mode for UI-free demos.
- GUI "Save EXR..." button in the Output/Export panel.
- Initial GitHub Actions CI workflow for configure/build validation.

### Changed
- Embree is now vendored under `external/embree` for opt-in local builds.
- `M0_TESTS` now fails fast with an explicit message when internal golden-test scripts are unavailable.
- Asset handling in CMake now supports public checkouts without a local asset pack by creating empty asset directories instead of failing.
- Documentation clarified that public repository scope excludes internal golden-image scripts and large non-versioned asset packs.

### Notes
- Internal golden-image scripts (`tests/tools/golden_test.sh`) remain private and are intentionally not distributed in this public repository.
- Large scene assets/HDRIs are intentionally not versioned in Git and should be provided via the external asset pack.

## [v1.0.0] - 2025-11-19

### Added
- Initial public release targeting Apple Silicon (arm64).
- Metal HWRT + software BVH path tracing backends.
- Progressive accumulation renderer.
- HDR environment lighting with MIS and rotation/intensity controls.
- Core material models (Lambertian, metal, dielectric, plastic, car paint, subsurface, emissive).
- OBJ/PLY mesh support.
- OIDN integration.
- ImGui + ImGuizmo GUI application and headless renderer.
