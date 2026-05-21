#!/usr/bin/env python3
"""Validate the R6 ReSTIR roadmap full-stack release hygiene gate."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import struct
import subprocess
import sys
from pathlib import Path

from validate_restir_r3_checkpoint1 import resolve_binary, validate_pfm_finite


R2_SCENE = "tests/scenes/restir/r2/r2_emissive_closure.scene"
R3_SCENE = "tests/scenes/restir/r3/r3_diffuse_indirect.scene"
LIVING_ROOM_CLEAN = "assets/living_room_regir_stress.scene"
LIVING_ROOM_PRACTICALS = "assets/living_room_regir_practicals_stress.scene"

DEFAULT_WIDTH = 96
DEFAULT_HEIGHT = 54
DEFAULT_SPP = 2
DEFAULT_SEED = 42

EXISTING_VALIDATORS = [
    ["python3", "scripts/validate_living_room_regir_stress.py"],
    ["python3", "scripts/validate_living_room_regir_practicals_stress.py"],
    ["python3", "scripts/validate_restir_di_regir_hybrid.py", "--binary", "{binary}"],
    ["python3", "scripts/validate_restir_gi_prototype.py", "--binary", "{binary}"],
    ["python3", "scripts/validate_svgf_denoise_prototype.py", "--binary", "{binary}"],
    ["python3", "scripts/validate_metal_pass_graph.py", "--binary", "{binary}"],
    ["python3", "scripts/validate_restir_r3_checkpoint1.py", "--binary", "{binary}"],
    ["python3", "scripts/validate_restir_r4_checkpoint1.py", "--binary", "{binary}"],
    ["python3", "scripts/validate_restir_r5_checkpoint1.py", "--binary", "{binary}"],
    ["python3", "scripts/validate_restir_r5_structural.py", "--binary", "{binary}"],
    ["python3", "scripts/validate_restir_r6_checkpoint1.py", "--binary", "{binary}"],
    ["python3", "scripts/validate_restir_r6_structural.py", "--binary", "{binary}"],
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


def materialize_validator_command(cmd: list[str], binary: Path) -> list[str]:
    return [str(binary) if token == "{binary}" else token for token in cmd]


def read_pfm_rgb(path: Path) -> tuple[int, int, list[float]]:
    with path.open("rb") as file:
        magic = file.readline().strip()
        if magic != b"PF":
            raise RuntimeError(f"{path}: expected RGB PFM")
        dims = file.readline().split()
        if len(dims) != 2:
            raise RuntimeError(f"{path}: malformed PFM dimensions")
        width = int(dims[0])
        height = int(dims[1])
        scale = float(file.readline().strip())
        endian = "<" if scale < 0.0 else ">"
        count = width * height * 3
        data = file.read()
    if len(data) != count * 4:
        raise RuntimeError(f"{path}: expected {count * 4} bytes, got {len(data)}")
    values = list(struct.unpack(endian + "f" * count, data))
    if abs(scale) != 1.0:
        values = [value * abs(scale) for value in values]
    return width, height, values


def luma_stats(path: Path) -> dict[str, float]:
    _, _, values = read_pfm_rgb(path)
    lumas: list[float] = []
    max_component = 0.0
    for index in range(0, len(values), 3):
        r, g, b = values[index], values[index + 1], values[index + 2]
        if not (math.isfinite(r) and math.isfinite(g) and math.isfinite(b)):
            raise RuntimeError(f"{path}: found non-finite RGB component")
        max_component = max(max_component, abs(r), abs(g), abs(b))
        lumas.append(0.2126 * r + 0.7152 * g + 0.0722 * b)
    mean = sum(lumas) / max(len(lumas), 1)
    variance = sum((value - mean) * (value - mean) for value in lumas) / max(len(lumas), 1)
    return {"mean_luma": mean, "variance_luma": variance, "max_component": max_component}


def log_luma(path: Path) -> list[float]:
    _, _, values = read_pfm_rgb(path)
    result: list[float] = []
    for index in range(0, len(values), 3):
        r, g, b = values[index], values[index + 1], values[index + 2]
        luma = max(0.2126 * r + 0.7152 * g + 0.0722 * b, 0.0)
        result.append(math.log1p(luma))
    return result


def scalar_rmse(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise RuntimeError("image dimensions differ")
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)) / max(len(a), 1))


def scalar_ssim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise RuntimeError("image dimensions differ")
    mean_a = sum(a) / max(len(a), 1)
    mean_b = sum(b) / max(len(b), 1)
    var_a = var_b = cov_ab = 0.0
    for x, y in zip(a, b):
        dx = x - mean_a
        dy = y - mean_b
        var_a += dx * dx
        var_b += dy * dy
        cov_ab += dx * dy
    count = max(len(a), 1)
    var_a /= count
    var_b /= count
    cov_ab /= count
    c1 = 0.01 * 0.01
    c2 = 0.03 * 0.03
    numerator = (2.0 * mean_a * mean_b + c1) * (2.0 * cov_ab + c2)
    denominator = (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2)
    return numerator / denominator if denominator > 0.0 else float("nan")


def rgb_rmse(a: Path, b: Path) -> float:
    width_a, height_a, values_a = read_pfm_rgb(a)
    width_b, height_b, values_b = read_pfm_rgb(b)
    if (width_a, height_a) != (width_b, height_b):
        raise RuntimeError("image dimensions differ")
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(values_a, values_b)) / max(len(values_a), 1))


def compare_images(reference: Path, candidate: Path) -> dict[str, float]:
    ref_log = log_luma(reference)
    cand_log = log_luma(candidate)
    log_rmse = scalar_rmse(ref_log, cand_log)
    return {
        "rgb_rmse": rgb_rmse(reference, candidate),
        "log_luma_rmse": log_rmse,
        "log_luma_psnr": float("inf") if log_rmse == 0.0 else 20.0 * math.log10(1.0 / log_rmse),
        "ssim": scalar_ssim(ref_log, cand_log),
    }


def make_execution_scene(source: Path, artifact_dir: Path, execution_mode: str, label: str) -> Path:
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("renderer "):
            tokens = [token for token in line.split() if not token.startswith("executionMode=")]
            tokens.append(f"executionMode={execution_mode}")
            lines[index] = " ".join(tokens)
            break
    else:
        lines.append(f"renderer executionMode={execution_mode}")
    path = artifact_dir / f"{label}_{execution_mode}.scene"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def render(
    root: Path,
    binary: Path,
    scene: Path,
    artifact_dir: Path,
    label: str,
    width: int,
    height: int,
    spp: int,
    seed: int,
    direct_mode: str | None = None,
    restir_gi_mode: str | None = None,
    svgf: bool | None = None,
    audit: bool = False,
    verbose: bool = False,
) -> dict:
    output = artifact_dir / f"{label}.pfm"
    metrics = artifact_dir / f"{label}_metrics.json"
    cmd = [
        str(binary),
        f"--scene={scene}",
        "--backend=metal",
        "--enableSoftwareRayTracing=1",
        f"--width={width}",
        f"--height={height}",
        f"--sppTotal={spp}",
        f"--seed={seed}",
        f"--pbrMetricsJson={metrics}",
        f"--output={output}",
        "--format=pfm",
    ]
    if direct_mode:
        cmd.append(f"--directLightMode={direct_mode}")
    if restir_gi_mode:
        cmd.append(f"--restirGiMode={restir_gi_mode}")
    if svgf is not None:
        cmd.append(f"--svgfDenoise={1 if svgf else 0}")
    audit_path = None
    if audit:
        audit_path = artifact_dir / f"{label}_audit.json"
        cmd.extend(["--debugPixel=64,36", f"--directLightAuditJson={audit_path}"])
    run_command(cmd, root, verbose)
    validate_pfm_finite(output)
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    validate_metrics_payload(payload, label)
    stats = luma_stats(output)
    if stats["mean_luma"] <= 1.0e-8:
        raise RuntimeError(f"{label}: rendered black or near-black unexpectedly")
    if stats["max_component"] > 1.0e4:
        raise RuntimeError(f"{label}: severe energy explosion, max_component={stats['max_component']}")
    if audit and (audit_path is None or not audit_path.exists() or audit_path.stat().st_size == 0):
        raise RuntimeError(f"{label}: explicit direct-light audit did not write output")
    return {
        "output": str(output),
        "metrics": str(metrics),
        "audit": str(audit_path) if audit_path else None,
        "payload": payload,
        "stats": stats,
    }


def validate_metrics_payload(payload: dict, label: str) -> None:
    image = payload.get("image", {})
    renderer = payload.get("renderer", {})
    if image.get("nan_inf_pixel_count") != 0:
        raise RuntimeError(f"{label}: image NaN/Inf pixel count is nonzero")
    if image.get("nan_inf_component_count") != 0:
        raise RuntimeError(f"{label}: image NaN/Inf component count is nonzero")
    if renderer.get("throughput_nan_inf_count") != 0:
        raise RuntimeError(f"{label}: throughput NaN/Inf count is nonzero")
    if renderer.get("radiance_nan_inf_count") != 0:
        raise RuntimeError(f"{label}: radiance NaN/Inf count is nonzero")


def inspect_cli(binary: Path, root: Path) -> str:
    completed = subprocess.run([str(binary), "--help"], cwd=root, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError("PathTracerHeadless --help failed")
    text = completed.stdout + completed.stderr
    for token in (
        "--directLightMode",
        "restir_di_regir_hybrid",
        "--restirGiMode",
        "restir_gi_prototype",
        "--svgfDenoise",
        "--debugPixel",
        "--directLightAuditJson",
    ):
        if token not in text:
            raise RuntimeError(f"CLI help missing required token {token}")
    return text


def validate_history_reset_source(root: Path) -> list[str]:
    source = (root / "src/renderer/SettingsUtils.mm").read_text(encoding="utf-8")
    required = [
        "RENDER_MODE",
        "DIRECT_LIGHT_MODE",
        "RIS_CANDIDATE_COUNT",
        "WORLD_REUSE_CELL_SIZE",
        "RESTIR_GI_MODE",
        "SVGF_DENOISE_ENABLED",
        "SVGF_ATROUS_ITERATIONS",
        "SVGF_NORMAL_SIGMA",
        "SVGF_ALBEDO_SIGMA",
        "SVGF_LUMINANCE_SIGMA",
    ]
    missing = [token for token in required if token not in source]
    if missing:
        raise RuntimeError(f"SettingsUtils history reset audit missing {missing}")
    return required


def pass_names(payload: dict) -> list[str]:
    return [entry.get("name", "") for entry in payload.get("pass_graph", {}).get("passes", [])]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate R6 ReSTIR roadmap full-stack release hygiene")
    parser.add_argument("--binary", default=None)
    parser.add_argument("--artifact-dir", default="/private/tmp/restir_r6_full_stack_validation")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--spp", type=int, default=DEFAULT_SPP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    binary = resolve_binary(root, args.binary)
    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    inspect_cli(binary, root)
    history_tokens = validate_history_reset_source(root)
    validator_results: list[dict[str, str]] = []
    for cmd in EXISTING_VALIDATORS:
        materialized = materialize_validator_command(cmd, binary)
        output = run_command(materialized, root, args.verbose)
        validator_results.append({"command": shlex.join(materialized), "tail": output.strip().splitlines()[-1] if output.strip() else "PASS"})

    r2_scene = (root / R2_SCENE).resolve()
    r3_scene = (root / R3_SCENE).resolve()
    lr_clean = (root / LIVING_ROOM_CLEAN).resolve()
    lr_practicals = (root / LIVING_ROOM_PRACTICALS).resolve()
    for scene in (r2_scene, r3_scene, lr_clean, lr_practicals):
        if not scene.exists():
            raise RuntimeError(f"Required scene missing: {scene}")

    matrix: dict[str, dict] = {}
    matrix["default"] = render(
        root, binary, r2_scene, artifact_dir, "matrix_default",
        args.width, args.height, args.spp, args.seed, verbose=args.verbose)
    default_payload = matrix["default"]["payload"]
    default_settings = default_payload.get("settings", {})
    default_post = default_payload.get("offline_post", {})
    default_passes = pass_names(default_payload)
    if default_settings.get("restir_gi_mode") != "off":
        raise RuntimeError(f"default restir_gi_mode should be off, got {default_settings.get('restir_gi_mode')}")
    if default_post.get("svgf_denoise_enabled") is not False:
        raise RuntimeError("default svgf_denoise_enabled should be false")
    if "SVGF A-Trous Filter" in default_passes:
        raise RuntimeError("default pass graph unexpectedly includes SVGF pass")

    for mode in ("baseline_emissive", "ris", "ris_regir", "restir_di", "restir_di_regir_hybrid"):
        matrix[mode] = render(
            root, binary, r2_scene, artifact_dir, f"matrix_{mode}",
            args.width, args.height, args.spp, args.seed,
            direct_mode=mode, restir_gi_mode="off", svgf=False, verbose=args.verbose)

    matrix["restir_gi_prototype"] = render(
        root, binary, r3_scene, artifact_dir, "matrix_restir_gi_prototype",
        args.width, args.height, args.spp, args.seed,
        direct_mode="legacy", restir_gi_mode="restir_gi_prototype", svgf=False, verbose=args.verbose)
    matrix["svgfDenoise"] = render(
        root, binary, r3_scene, artifact_dir, "matrix_svgfDenoise",
        args.width, args.height, args.spp, args.seed,
        direct_mode="legacy", restir_gi_mode="off", svgf=True, verbose=args.verbose)
    if "SVGF A-Trous Filter" not in pass_names(matrix["svgfDenoise"]["payload"]):
        raise RuntimeError("explicit SVGF render did not include SVGF pass graph entry")

    matrix["direct_light_audit"] = render(
        root, binary, r2_scene, artifact_dir, "matrix_direct_light_audit",
        args.width, args.height, 1, args.seed,
        direct_mode="ris", restir_gi_mode="off", svgf=False, audit=True, verbose=args.verbose)

    parity_cases = [
        ("living_room_clean_regir", lr_clean, "ris_regir", "off", False),
        ("living_room_practicals_regir", lr_practicals, "ris_regir", "off", False),
        ("r3_diffuse_indirect_gi", r3_scene, "legacy", "restir_gi_prototype", False),
        ("r5_pass_graph_svgf", r3_scene, "legacy", "off", True),
    ]
    parity: dict[str, dict] = {}
    for label, source_scene, direct_mode, gi_mode, svgf in parity_cases:
        mk_scene = make_execution_scene(source_scene, artifact_dir, "megakernel", label)
        wf_scene = make_execution_scene(source_scene, artifact_dir, "wavefront", label)
        mk = render(
            root, binary, mk_scene, artifact_dir, f"parity_{label}_megakernel",
            args.width, args.height, args.spp, args.seed,
            direct_mode=direct_mode, restir_gi_mode=gi_mode, svgf=svgf, verbose=args.verbose)
        wf = render(
            root, binary, wf_scene, artifact_dir, f"parity_{label}_wavefront",
            args.width, args.height, args.spp, args.seed,
            direct_mode=direct_mode, restir_gi_mode=gi_mode, svgf=svgf, verbose=args.verbose)
        metrics = compare_images(Path(mk["output"]), Path(wf["output"]))
        if metrics["log_luma_rmse"] > 0.05 or metrics["ssim"] < 0.98:
            raise RuntimeError(f"{label}: megakernel/wavefront parity failed {metrics}")
        parity[label] = {
            "megakernel_output": mk["output"],
            "megakernel_metrics": mk["metrics"],
            "wavefront_output": wf["output"],
            "wavefront_metrics": wf["metrics"],
            "metrics": metrics,
            "megakernel_pass_count": len(pass_names(mk["payload"])),
            "wavefront_pass_count": len(pass_names(wf["payload"])),
            "megakernel_dispatch_count": mk["payload"].get("performance", {}).get("total_compute_dispatch_count", 0),
            "wavefront_dispatch_count": wf["payload"].get("performance", {}).get("total_compute_dispatch_count", 0),
            "megakernel_mean_luma": mk["stats"]["mean_luma"],
            "wavefront_mean_luma": wf["stats"]["mean_luma"],
        }

    summary = {
        "status": "PASS",
        "artifact_dir": str(artifact_dir),
        "binary": str(binary),
        "validators": validator_results,
        "history_reset_tokens": history_tokens,
        "default_state": {
            "direct_light_mode": default_settings.get("direct_light_mode"),
            "restir_gi_mode": default_settings.get("restir_gi_mode"),
            "svgf_denoise_enabled": default_post.get("svgf_denoise_enabled"),
            "pass_names": default_passes,
        },
        "mode_matrix": {
            key: {
                "output": value["output"],
                "metrics": value["metrics"],
                "mean_luma": value["stats"]["mean_luma"],
                "max_component": value["stats"]["max_component"],
                "pass_count": len(pass_names(value["payload"])),
            }
            for key, value in matrix.items()
        },
        "parity": parity,
    }
    summary_path = artifact_dir / "restir_r6_full_stack_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")

    print(
        "R6 ReSTIR roadmap full-stack validation PASSED "
        f"artifactDir={artifact_dir} modes={len(matrix)} parityCases={len(parity)} "
        f"validators={len(validator_results)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
