#!/usr/bin/env python3
"""Compute MAE/RMSE/Max error between two PFM images."""

import argparse
import json
import math
import struct
import sys


def read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode("ascii", errors="strict").strip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"{path}: unsupported PFM header '{header}'")
        color = header == "PF"
        dims = f.readline().decode("ascii", errors="strict").strip().split()
        if len(dims) != 2:
            raise ValueError(f"{path}: invalid dimensions line")
        width, height = map(int, dims)
        scale_line = f.readline().decode("ascii", errors="strict").strip()
        try:
            scale = float(scale_line)
        except ValueError as exc:
            raise ValueError(f"{path}: invalid scale value") from exc
        endian = "<" if scale < 0 else ">"
        scale = abs(scale)
        channels = 3 if color else 1
        count = width * height * channels
        data = f.read()
        expected_bytes = count * 4
        if len(data) != expected_bytes:
            raise ValueError(f"{path}: expected {expected_bytes} bytes, got {len(data)}")
        fmt = f"{endian}{count}f"
        values = struct.unpack(fmt, data)
        if scale != 1.0:
            values = [v * scale for v in values]
        return width, height, channels, values


def compute_metrics(values_a, values_b):
    if len(values_a) != len(values_b):
        raise ValueError("Input sizes do not match")
    abs_sum = 0.0
    sq_sum = 0.0
    max_err = 0.0
    invalid = 0
    for a, b in zip(values_a, values_b):
        if not (math.isfinite(a) and math.isfinite(b)):
            invalid += 1
            continue
        diff = abs(a - b)
        abs_sum += diff
        sq_sum += diff * diff
        if diff > max_err:
            max_err = diff
    count = len(values_a) - invalid
    if count <= 0:
        raise ValueError("No valid samples to compare")
    mae = abs_sum / count
    rmse = math.sqrt(sq_sum / count)
    return mae, rmse, max_err, invalid, count


def compute_mean_luminance(values, channels):
    if channels == 1:
        total = 0.0
        count = 0
        for v in values:
            if not math.isfinite(v):
                continue
            total += v
            count += 1
        if count == 0:
            raise ValueError("No valid samples to compute mean luminance")
        return total / count
    if channels == 3:
        total = 0.0
        count = 0
        for i in range(0, len(values), 3):
            r = values[i]
            g = values[i + 1]
            b = values[i + 2]
            if not (math.isfinite(r) and math.isfinite(g) and math.isfinite(b)):
                continue
            total += 0.2126 * r + 0.7152 * g + 0.0722 * b
            count += 1
        if count == 0:
            raise ValueError("No valid samples to compute mean luminance")
        return total / count
    raise ValueError(f"Unsupported channel count: {channels}")


def apply_exposure(values, exposure_stops):
    if exposure_stops == 0.0:
        return values
    scale = math.pow(2.0, exposure_stops)
    return [v * scale for v in values]


def tonemap_reinhard(values):
    out = []
    for v in values:
        if v <= 0.0:
            out.append(0.0)
        else:
            out.append(v / (1.0 + v))
    return out


def clamp_and_normalize(values, clamp_value):
    if clamp_value <= 0.0:
        raise ValueError("Clamp value must be > 0")
    inv = 1.0 / clamp_value
    out = []
    for v in values:
        if v <= 0.0:
            out.append(0.0)
        else:
            out.append(min(v, clamp_value) * inv)
    return out


def compute_psnr(values_a, values_b, max_value=1.0):
    if len(values_a) != len(values_b):
        raise ValueError("Input sizes do not match")
    sq_sum = 0.0
    count = 0
    for a, b in zip(values_a, values_b):
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        diff = a - b
        sq_sum += diff * diff
        count += 1
    if count <= 0:
        raise ValueError("No valid samples to compare for PSNR")
    mse = sq_sum / count
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10((max_value * max_value) / mse)


def main():
    parser = argparse.ArgumentParser(description="PFM error metrics (MAE/RMSE/Max)")
    parser.add_argument("before", help="Baseline PFM path")
    parser.add_argument("after", help="Comparison PFM path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--exposure", type=float, default=0.0,
                        help="Exposure in stops applied before clamp/tonemap")
    parser.add_argument("--tonemap", choices=("none", "reinhard"), default="none",
                        help="Tonemap operator applied before PSNR (default: none)")
    parser.add_argument("--clamp", type=float, default=1.0,
                        help="Clamp value before normalization (default: 1.0)")
    parser.add_argument("--psnr-threshold", type=float, default=None,
                        help="Fail if PSNR is below this threshold (dB)")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON summary only")
    parser.add_argument("--mean-luma-threshold", type=float, default=None,
                        help="Fail if relative mean luminance delta exceeds this fraction")
    args = parser.parse_args()

    w0, h0, c0, v0 = read_pfm(args.before)
    w1, h1, c1, v1 = read_pfm(args.after)
    if (w0, h0, c0) != (w1, h1, c1):
        raise ValueError("PFM dimensions/channels mismatch")

    mae, rmse, max_err, invalid, count = compute_metrics(v0, v1)
    v0_exp = apply_exposure(v0, args.exposure)
    v1_exp = apply_exposure(v1, args.exposure)
    mean0 = compute_mean_luminance(v0_exp, c0)
    mean1 = compute_mean_luminance(v1_exp, c1)
    mean_diff = abs(mean0 - mean1)
    mean_rel = mean_diff / max(abs(mean0), 1.0e-8)

    if not args.json:
        if args.verbose:
            print(f"Resolution: {w0}x{h0} channels={c0}")
            print(f"Valid samples: {count} (invalid: {invalid})")
        print(f"MAE (raw):  {mae:.8f}")
        print(f"RMSE (raw): {rmse:.8f}")
        print(f"Max (raw):  {max_err:.8f}")
        print(f"Mean luma: {mean0:.8f} vs {mean1:.8f} (delta {mean_diff:.8f}, rel {mean_rel:.6f})")

    v0_proc = v0_exp
    v1_proc = v1_exp
    if args.tonemap == "reinhard":
        v0_proc = tonemap_reinhard(v0_proc)
        v1_proc = tonemap_reinhard(v1_proc)
    else:
        v0_proc = clamp_and_normalize(v0_proc, args.clamp)
        v1_proc = clamp_and_normalize(v1_proc, args.clamp)

    p_mae, p_rmse, p_max, p_invalid, p_count = compute_metrics(v0_proc, v1_proc)
    if not args.json:
        if args.verbose:
            print(f"Processed samples: {p_count} (invalid: {p_invalid})")
        print(f"MAE (proc):  {p_mae:.8f}")
        print(f"RMSE (proc): {p_rmse:.8f}")
        print(f"Max (proc):  {p_max:.8f}")
    psnr = compute_psnr(v0_proc, v1_proc, max_value=1.0)
    if not args.json:
        print(f"PSNR (proc): {psnr:.2f} dB")
    else:
        payload = {
            "resolution": {"width": w0, "height": h0, "channels": c0},
            "raw": {
                "mae": mae,
                "rmse": rmse,
                "max": max_err,
                "invalid": invalid,
                "count": count,
            },
            "mean_luma": {
                "a": mean0,
                "b": mean1,
                "abs_diff": mean_diff,
                "rel_diff": mean_rel,
            },
            "processed": {
                "mae": p_mae,
                "rmse": p_rmse,
                "max": p_max,
                "invalid": p_invalid,
                "count": p_count,
            },
            "psnr": psnr,
            "settings": {
                "exposure": args.exposure,
                "tonemap": args.tonemap,
                "clamp": args.clamp,
            },
        }
        print(json.dumps(payload))
    if args.psnr_threshold is not None and psnr < args.psnr_threshold:
        raise SystemExit(f"PSNR {psnr:.2f} dB below threshold {args.psnr_threshold:.2f} dB")
    if args.mean_luma_threshold is not None and mean_rel > args.mean_luma_threshold:
        raise SystemExit(
            f"Mean luminance delta {mean_rel:.6f} exceeds threshold {args.mean_luma_threshold:.6f}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
