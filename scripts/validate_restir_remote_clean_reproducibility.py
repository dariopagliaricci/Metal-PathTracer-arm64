#!/usr/bin/env python3
"""Validate R13 remote-clean ReSTIR roadmap reproducibility."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from validate_path_guiding_prototype import luma_stats, read_pfm_rgb, resolve_binary
from validate_restir_r3_checkpoint1 import validate_pfm_finite


REQUIRED_REPORTS = [
    "R1_REGIR_WAVEFRONT_PARITY_VALIDATION.md",
    "R2_RESTIR_DI_REGIR_HYBRID_VALIDATION.md",
    "R3_RESTIR_GI_PROTOTYPE_VALIDATION.md",
    "R4_SVGF_DENOISING_PROTOTYPE_VALIDATION.md",
    "R5_METAL_PASS_GRAPH_VALIDATION.md",
    "R6_RESTIR_ROADMAP_RELEASE_HYGIENE_VALIDATION.md",
    "R7_PATH_GUIDING_VALIDATION.md",
    "R8_RESTIR_PT_RESEARCH_SCAFFOLD_VALIDATION.md",
    "R9_RESTIR_PT_EXPERIMENTAL_PATH_REUSE_VALIDATION.md",
    "R10_RESTIR_PT_PARITY_HARDENING_VALIDATION.md",
    "R11_ROADMAP_DOCUMENTATION_RELEASE_AUDIT.md",
    "R12_FINAL_FULL_STACK_LONG_RUN_VALIDATION.md",
]

REQUIRED_VALIDATORS = [
    "scripts/validate_living_room_regir_stress.py",
    "scripts/validate_living_room_regir_practicals_stress.py",
    "scripts/validate_restir_di_regir_hybrid.py",
    "scripts/validate_restir_gi_prototype.py",
    "scripts/validate_svgf_denoise_prototype.py",
    "scripts/validate_metal_pass_graph.py",
    "scripts/validate_path_guiding_prototype.py",
    "scripts/validate_restir_pt_research_scaffold.py",
    "scripts/validate_restir_pt_experimental_path_reuse.py",
    "scripts/validate_restir_pt_interactions.py",
    "scripts/validate_restir_r6_structural.py",
    "scripts/validate_restir_roadmap_full_stack.py",
    "scripts/validate_restir_final_full_stack_long_run.py",
]

REQUIRED_SCENES = [
    "assets/living_room.scene",
    "assets/living_room_regir_stress.scene",
    "assets/living_room_regir_practicals_stress.scene",
    "tests/scenes/restir/r2/r2_emissive_closure.scene",
    "tests/scenes/restir/r3/r3_diffuse_indirect.scene",
]

REQUIRED_ASSETS = [
    "assets/living_room/living_room.obj",
    "assets/living_room/living_room.mtl",
    "assets/canonical/living_room/import_manifest.json",
    "assets/canonical/living_room/living_room.glb",
]

REQUIRED_DOCS = [
    "docs/restir/ROADMAP_STATUS.md",
    "docs/restir/VALIDATION_COMMANDS.md",
    "docs/restir/FEATURE_GATES.md",
]

REQUIRED_FLAGS = [
    "--directLightMode",
    "baseline_emissive",
    "ris_regir",
    "restir_di",
    "restir_di_regir_hybrid",
    "--restirGiMode",
    "restir_gi_prototype",
    "--pathGuidingMode",
    "path_guiding_prototype",
    "--restirPtMode",
    "restir_pt_research",
    "restir_pt_experimental_path_reuse",
    "--svgfDenoise",
    "--debugPixel",
    "--directLightAuditJson",
    "--pbrMetricsJson",
]

DOC_REQUIRED_TOKENS = [
    "R4_SVGF_STYLE_DENOISING_PROTOTYPE",
    "EXPERIMENTAL_PASS",
    "R8_BOUNDED_RESTIR_PT_RESEARCH_SCAFFOLD",
    "R9_MINIMAL_RESTIR_PT_EXPERIMENTAL_PATH_REUSE",
    "R10_RESTIR_PT_PARITY_AND_INTERACTION_HARDENING",
    "R11_ROADMAP_DOCUMENTATION_AND_RELEASE_CANDIDATE_AUDIT",
    "R12_FINAL_FULL_STACK_LONG_RUN_VALIDATION",
    "restir_pt_research",
    "restir_pt_experimental_path_reuse",
    "path_guiding_prototype",
    "svgfDenoise",
]


def run_command(cmd: list[str], cwd: Path, verbose: bool) -> str:
    if verbose:
        print(shlex.join(cmd))
    completed = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    combined = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {shlex.join(cmd)}\n{combined}")
    if verbose and combined.strip():
        print(combined.strip())
    return combined


def require_paths(root: Path, paths: list[str], label: str) -> list[str]:
    missing = [path for path in paths if not (root / path).exists()]
    if missing:
        raise RuntimeError(f"Missing required {label}: {missing}")
    return paths


def inspect_docs(root: Path) -> dict:
    require_paths(root, REQUIRED_DOCS, "docs")
    validation_dir = root / "docs/restir/validation"
    require_paths(validation_dir, REQUIRED_REPORTS, "validation reports")

    combined_docs = "\n".join((root / path).read_text(encoding="utf-8") for path in REQUIRED_DOCS)
    r12_report = (validation_dir / "R12_FINAL_FULL_STACK_LONG_RUN_VALIDATION.md").read_text(encoding="utf-8")
    missing_tokens = [token for token in DOC_REQUIRED_TOKENS if token not in combined_docs and token not in r12_report]
    if missing_tokens:
        raise RuntimeError(f"Roadmap docs missing required tokens: {missing_tokens}")

    referenced_scripts = sorted(set(re.findall(r"scripts/[A-Za-z0-9_./-]+\.py", combined_docs + "\n" + r12_report)))
    missing_scripts = [path for path in referenced_scripts if not (root / path).exists()]
    if missing_scripts:
        raise RuntimeError(f"Docs reference missing scripts: {missing_scripts}")

    referenced_reports = sorted(set(re.findall(r"docs/restir/validation/[A-Z0-9_./-]+\.md", combined_docs + "\n" + r12_report)))
    missing_reports = [path for path in referenced_reports if not (root / path).exists()]
    if missing_reports:
        raise RuntimeError(f"Docs reference missing reports: {missing_reports}")

    return {
        "required_docs": REQUIRED_DOCS,
        "required_reports": [str(validation_dir / report) for report in REQUIRED_REPORTS],
        "referenced_scripts_checked": referenced_scripts,
        "referenced_reports_checked": referenced_reports,
    }


def inspect_cli(binary: Path, root: Path) -> dict:
    text = run_command([str(binary), "--help"], root, verbose=False)
    missing = [token for token in REQUIRED_FLAGS if token not in text]
    if missing:
        raise RuntimeError(f"PathTracerHeadless --help missing required flags/tokens: {missing}")
    return {"required_flags": REQUIRED_FLAGS}


def run_default_smoke(root: Path, binary: Path, artifact_dir: Path, verbose: bool) -> dict:
    output = artifact_dir / "r13_default_smoke.pfm"
    metrics = artifact_dir / "r13_default_smoke_metrics.json"
    cmd = [
        str(binary),
        "--scene=tests/scenes/restir/r3/r3_diffuse_indirect.scene",
        "--backend=metal",
        "--enableSoftwareRayTracing=1",
        "--width=64",
        "--height=36",
        "--sppTotal=1",
        "--seed=1313",
        f"--pbrMetricsJson={metrics}",
        f"--output={output}",
        "--format=pfm",
    ]
    run_command(cmd, root, verbose)
    validate_pfm_finite(output)
    read_pfm_rgb(output)
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    image = payload.get("image", {})
    renderer = payload.get("renderer", {})
    for key in ("nan_inf_pixel_count", "nan_inf_component_count"):
        if image.get(key) != 0:
            raise RuntimeError(f"default smoke image {key} is nonzero: {image.get(key)}")
    for key in ("throughput_nan_inf_count", "radiance_nan_inf_count"):
        if renderer.get(key) != 0:
            raise RuntimeError(f"default smoke renderer {key} is nonzero: {renderer.get(key)}")

    settings = payload.get("settings", {})
    expected_off = {
        "restir_gi_mode": "off",
        "path_guiding_mode": "off",
        "restir_pt_mode": "off",
    }
    for key, expected in expected_off.items():
        if settings.get(key) != expected:
            raise RuntimeError(f"default {key} expected {expected}, got {settings.get(key)}")
    if settings.get("svgf_denoise_enabled"):
        raise RuntimeError("default smoke unexpectedly enabled SVGF")
    if renderer.get("path_guided_sample_count", 0) != 0:
        raise RuntimeError("default smoke unexpectedly used path guiding")
    if renderer.get("restir_pt_candidate_count", 0) != 0:
        raise RuntimeError("default smoke unexpectedly captured ReSTIR PT candidates")

    stats = luma_stats(output)
    if stats["mean_luma"] <= 1.0e-8:
        raise RuntimeError("default smoke rendered black or near-black unexpectedly")
    if stats["max_component"] > 1.0e4:
        raise RuntimeError(f"default smoke energy explosion: {stats['max_component']}")

    return {
        "command": shlex.join(cmd),
        "output": str(output),
        "metrics": str(metrics),
        "settings": settings,
        "stats": stats,
        "nan_inf": {
            "pixel_count": image.get("nan_inf_pixel_count"),
            "component_count": image.get("nan_inf_component_count"),
            "throughput_count": renderer.get("throughput_nan_inf_count"),
            "radiance_count": renderer.get("radiance_nan_inf_count"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate R13 remote-clean roadmap reproducibility")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--artifact-dir", default="/private/tmp/restir_r13_remote_clean_reproducibility")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    binary = resolve_binary(root, args.binary)
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    require_paths(root, REQUIRED_VALIDATORS, "validators")
    require_paths(root, REQUIRED_SCENES, "scene fixtures")
    require_paths(root, REQUIRED_ASSETS, "canonical assets")
    docs = inspect_docs(root)
    cli = inspect_cli(binary, root)
    smoke = run_default_smoke(root, binary, artifact_dir, args.verbose)

    summary = {
        "stage": "R13_REMOTE_CLEAN_REPRODUCIBILITY_AND_PR_READINESS_GATE",
        "status": "PASS",
        "binary": str(binary),
        "artifact_dir": str(artifact_dir),
        "validators": REQUIRED_VALIDATORS,
        "scenes": REQUIRED_SCENES,
        "canonical_assets": REQUIRED_ASSETS,
        "docs": docs,
        "cli": cli,
        "default_smoke": smoke,
    }
    summary_path = artifact_dir / "r13_remote_clean_reproducibility_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "R13 remote-clean reproducibility validation PASSED "
        f"artifactDir={artifact_dir} validators={len(REQUIRED_VALIDATORS)} reports={len(REQUIRED_REPORTS)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
