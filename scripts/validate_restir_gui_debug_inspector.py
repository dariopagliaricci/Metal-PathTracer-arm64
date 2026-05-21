#!/usr/bin/env python3
"""Validate R16 ReSTIR GUI/debug inspector plumbing headlessly."""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENE = "tests/scenes/restir/r3/r3_diffuse_indirect.scene"
WIDTH = 64
HEIGHT = 36
SEED = 42


def run(cmd: list[str], cwd: Path, verbose: bool) -> subprocess.CompletedProcess[str]:
    if verbose:
        print("$ " + " ".join(cmd))
    completed = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(cmd)
            + "\nstdout:\n"
            + completed.stdout
            + "\nstderr:\n"
            + completed.stderr
        )
    return completed


def resolve_binary(path: str) -> Path:
    binary = Path(path)
    if not binary.is_absolute():
        binary = ROOT / binary
    if not binary.exists():
        raise RuntimeError(f"binary not found: {binary}")
    return binary


def inspect_help(binary: Path) -> list[str]:
    completed = run([str(binary), "--help"], ROOT, False)
    text = completed.stdout + completed.stderr
    required = [
        "--restirDebug",
        "--restirDebugView",
        "--restirDebugCounters",
        "--restirDebugMetricsJson",
        "candidate_source_id",
        "restir_pt_reuse_mask",
        "nan_inf_mask",
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(f"CLI help missing R16 tokens: {missing}")
    return required


def inspect_source_tokens() -> dict[str, list[str]]:
    checks = {
        "settings": (
            ROOT / "include/renderer/RenderSettings.h",
            ["RestirDebugView", "restirDebugEnabled", "restirDebugCounters"],
        ),
        "ui": (
            ROOT / "src/renderer/UIOverlay.mm",
            ["ReSTIR Debug / Audit", "DrawRestirDebugPanel", "Sanity Alarms"],
        ),
        "headless": (
            ROOT / "src/main_headless.mm",
            ["--restirDebugView", "WriteRestirDebugMetricsJson", "ApplyRestirDebugView"],
        ),
        "pass_graph": (
            ROOT / "src/renderer/RenderPassGraph.cpp",
            ["ReSTIR Debug Inspector Audit", "restirDebugView"],
        ),
    }
    found: dict[str, list[str]] = {}
    for name, (path, tokens) in checks.items():
        text = path.read_text(encoding="utf-8")
        missing = [token for token in tokens if token not in text]
        if missing:
            raise RuntimeError(f"{path} missing R16 tokens: {missing}")
        found[name] = list(tokens)
    return found


def read_pfm(path: Path) -> tuple[int, int, list[float]]:
    with path.open("rb") as f:
        magic = f.readline().strip()
        if magic != b"PF":
            raise RuntimeError(f"{path}: expected PF PFM, got {magic!r}")
        dims = f.readline().strip().split()
        if len(dims) != 2:
            raise RuntimeError(f"{path}: invalid dimensions")
        width = int(dims[0])
        height = int(dims[1])
        scale = float(f.readline().strip())
        little = scale < 0.0
        data = f.read()
    count = width * height * 3
    endian = "<" if little else ">"
    values = list(struct.unpack(endian + "f" * count, data[: count * 4]))
    return width, height, values


def image_stats(path: Path) -> dict[str, float | int]:
    width, height, values = read_pfm(path)
    finite = [v for v in values if math.isfinite(v)]
    nan_inf = len(values) - len(finite)
    if not finite:
        raise RuntimeError(f"{path}: image has no finite values")
    return {
        "width": width,
        "height": height,
        "nan_inf_components": nan_inf,
        "mean": sum(finite) / len(finite),
        "max": max(finite),
    }


def compare_pfm(a: Path, b: Path) -> dict[str, float]:
    aw, ah, av = read_pfm(a)
    bw, bh, bv = read_pfm(b)
    if (aw, ah, len(av)) != (bw, bh, len(bv)):
        raise RuntimeError(f"image dimensions differ: {a} vs {b}")
    max_abs = 0.0
    mse = 0.0
    for x, y in zip(av, bv):
        if not math.isfinite(x) or not math.isfinite(y):
            raise RuntimeError("comparison saw non-finite component")
        d = abs(x - y)
        max_abs = max(max_abs, d)
        mse += d * d
    mse /= max(len(av), 1)
    return {"max_abs": max_abs, "rmse": math.sqrt(mse)}


def make_execution_scene(artifact_dir: Path, mode: str) -> Path:
    source = (ROOT / SCENE).read_text(encoding="utf-8")
    lines: list[str] = []
    replaced = False
    for line in source.splitlines():
        if line.startswith("renderer "):
            line = line.replace("renderer ", f"renderer executionMode={mode} ", 1)
            replaced = True
        lines.append(line)
    if not replaced:
        lines.append(f"renderer executionMode={mode}")
    path = artifact_dir / f"r16_{mode}.scene"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def render(
    binary: Path,
    artifact_dir: Path,
    name: str,
    extra: list[str],
    verbose: bool,
    scene: str | Path = SCENE,
) -> dict:
    output = artifact_dir / f"{name}.pfm"
    metrics = artifact_dir / f"{name}.restir_debug.json"
    cmd = [
        str(binary),
        f"--scene={scene}",
        "--backend=metal",
        "--enableSoftwareRayTracing=1",
        f"--width={WIDTH}",
        f"--height={HEIGHT}",
        "--sppTotal=2",
        f"--seed={SEED}",
        "--format=pfm",
        f"--output={output}",
        *extra,
    ]
    if "--restirDebugMetricsJson" in " ".join(extra):
        pass
    run(cmd, ROOT, verbose)
    stats = image_stats(output)
    if stats["nan_inf_components"] != 0:
        raise RuntimeError(f"{name}: output contains non-finite components")
    payload = None
    if metrics.exists():
        payload = json.loads(metrics.read_text(encoding="utf-8"))
    return {"name": name, "output": str(output), "metrics": str(metrics), "stats": stats, "payload": payload}


def render_debug(binary: Path, artifact_dir: Path, name: str, view: str, extra: list[str], verbose: bool) -> dict:
    metrics = artifact_dir / f"{name}.restir_debug.json"
    record = render(
        binary,
        artifact_dir,
        name,
        [
            "--restirDebug=1",
            "--restirDebugCounters=1",
            f"--restirDebugView={view}",
            f"--restirDebugMetricsJson={metrics}",
            *extra,
        ],
        verbose,
    )
    if record["payload"] is None:
        raise RuntimeError(f"{name}: missing ReSTIR debug metrics JSON")
    return record


def counter(payload: dict, key: str) -> int:
    return int(payload.get("restirDebug", {}).get("counters", {}).get(key, 0))


def validate_payload(record: dict, expected_view: str) -> None:
    payload = record["payload"]
    debug = payload.get("restirDebug", {})
    if not debug.get("enabled", False):
        raise RuntimeError(f"{record['name']}: debug payload not enabled")
    if debug.get("view") != expected_view:
        raise RuntimeError(f"{record['name']}: expected view {expected_view}, got {debug.get('view')}")
    if counter(payload, "nanInfPixelCount") != 0:
        raise RuntimeError(f"{record['name']}: nanInfPixelCount is nonzero")
    if "ReSTIR Debug Inspector Audit" not in [
        p.get("name") for p in payload.get("restirDebug", {}).get("passes", [])
    ]:
        # Older compact payloads do not duplicate pass names under restirDebug.
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    binary = resolve_binary(args.binary)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    help_tokens = inspect_help(binary)
    source_tokens = inspect_source_tokens()

    beauty = render(
        binary,
        artifact_dir,
        "beauty_default",
        ["--directLightMode=restir_di_regir_hybrid", "--ris-m=4"],
        args.verbose,
    )
    disabled = render(
        binary,
        artifact_dir,
        "beauty_debug_disabled",
        [
            "--directLightMode=restir_di_regir_hybrid",
            "--ris-m=4",
            "--restirDebug=0",
            "--restirDebugCounters=0",
            "--restirDebugView=beauty",
        ],
        args.verbose,
    )
    unchanged = compare_pfm(Path(beauty["output"]), Path(disabled["output"]))
    if unchanged["max_abs"] > 1.0e-7:
        raise RuntimeError(f"debug-disabled beauty changed: {unchanged}")

    candidate = render_debug(
        binary,
        artifact_dir,
        "debug_candidate_source",
        "candidate_source_id",
        ["--directLightMode=restir_di_regir_hybrid", "--ris-m=4"],
        args.verbose,
    )
    validate_payload(candidate, "candidate_source_id")
    if counter(candidate["payload"], "directLightCandidateCount") <= 0:
        raise RuntimeError("candidate-source debug run did not report direct-light candidates")

    path_guiding = render_debug(
        binary,
        artifact_dir,
        "debug_path_guiding_mask",
        "path_guiding_used_mask",
        ["--directLightMode=legacy", "--pathGuidingMode=path_guiding_prototype", "--pathGuidingStrength=0.5"],
        args.verbose,
    )
    validate_payload(path_guiding, "path_guiding_used_mask")
    if counter(path_guiding["payload"], "pathGuidingUsedCount") <= 0:
        raise RuntimeError("path-guiding debug run did not report used samples")

    restir_pt = render_debug(
        binary,
        artifact_dir,
        "debug_restir_pt_reuse_mask",
        "restir_pt_reuse_mask",
        ["--directLightMode=legacy", "--restirPtMode=restir_pt_experimental_path_reuse", "--restirPtReuseStrength=0.25"],
        args.verbose,
    )
    validate_payload(restir_pt, "restir_pt_reuse_mask")
    if counter(restir_pt["payload"], "restirPtCandidateCount") <= 0:
        raise RuntimeError("ReSTIR PT debug run did not report candidates")

    nan_mask = render_debug(
        binary,
        artifact_dir,
        "debug_nan_inf_mask",
        "nan_inf_mask",
        ["--directLightMode=restir_di_regir_hybrid", "--ris-m=4"],
        args.verbose,
    )
    validate_payload(nan_mask, "nan_inf_mask")

    mk_scene = make_execution_scene(artifact_dir, "megakernel")
    wf_scene = make_execution_scene(artifact_dir, "wavefront")
    mk = render(
        binary,
        artifact_dir,
        "parity_megakernel_debug_disabled",
        ["--directLightMode=restir_di_regir_hybrid", "--ris-m=4", "--restirDebug=0"],
        args.verbose,
        mk_scene,
    )
    wf = render(
        binary,
        artifact_dir,
        "parity_wavefront_debug_disabled",
        ["--directLightMode=restir_di_regir_hybrid", "--ris-m=4", "--restirDebug=0"],
        args.verbose,
        wf_scene,
    )
    parity = compare_pfm(Path(mk["output"]), Path(wf["output"]))
    if parity["rmse"] > 0.25:
        raise RuntimeError(f"megakernel/wavefront debug-disabled parity too loose: {parity}")

    summary = {
        "status": "PASS",
        "binary": str(binary),
        "artifact_dir": str(artifact_dir),
        "help_tokens": help_tokens,
        "source_tokens": source_tokens,
        "beauty_unchanged": unchanged,
        "parity": parity,
        "records": [beauty, disabled, candidate, path_guiding, restir_pt, nan_mask, mk, wf],
    }
    (artifact_dir / "restir_gui_debug_inspector_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"R16 GUI ReSTIR debug inspector validation PASSED artifactDir={artifact_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"R16 GUI ReSTIR debug inspector validation FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
