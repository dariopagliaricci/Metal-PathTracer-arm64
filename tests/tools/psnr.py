#!/usr/bin/env python3
"""
PSNR (Peak Signal-to-Noise Ratio) comparison tool for golden image testing.

Compares two images and computes the PSNR metric to determine visual similarity.
A PSNR >= 60 dB indicates the images are visually identical.

Usage:
    python3 psnr.py <image1_path> <image2_path> [--threshold THRESHOLD]

Exit codes:
    0: Images are visually identical (PSNR >= threshold)
    1: Images differ visually (PSNR < threshold)
    2: Error (different sizes, missing files, etc.)
"""

import sys
import math
import argparse
import os

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: Required packages not found.", file=sys.stderr)
    print("Install with: pip3 install Pillow numpy", file=sys.stderr)
    sys.exit(2)


def load_image(path: str) -> np.ndarray:
    """Load an image and convert to normalized float32 RGB array."""
    try:
        with Image.open(path) as img:
            # Convert to RGB if not already (handles RGBA, grayscale, etc.)
            rgb_img = img.convert('RGB')
            # Normalize to [0, 1] float32
            return np.asarray(rgb_img, dtype=np.float32) / 255.0
    except FileNotFoundError:
        print(f"Error: Image not found: {path}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error loading image {path}: {e}", file=sys.stderr)
        sys.exit(2)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute PSNR between two normalized images.

    Returns:
        PSNR in decibels. 100.0 dB if images are identical (MSE = 0).
    """
    # Compute Mean Squared Error
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0.0:
        # Images are identical
        return 100.0

    # PSNR formula: 10 * log10(MAX^2 / MSE)
    # Since our images are normalized to [0, 1], MAX = 1.0
    psnr = 10.0 * math.log10(1.0 / mse)
    return psnr


def compute_statistics(img1: np.ndarray, img2: np.ndarray):
    """Compute additional statistics for debugging."""
    diff = np.abs(img1 - img2)
    return {
        'max_difference': np.max(diff),
        'mean_difference': np.mean(diff),
        'std_difference': np.std(diff),
        'pixels_different': np.sum(diff > 0),
        'total_pixels': diff.size
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare two images using PSNR metric'
    )
    parser.add_argument('image1', help='Path to first image')
    parser.add_argument('image2', help='Path to second image')
    parser.add_argument(
        '--threshold',
        type=float,
        default=60.0,
        help='PSNR threshold in dB (default: 60.0)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Show detailed statistics'
    )

    args = parser.parse_args()

    # Verify files exist
    if not os.path.exists(args.image1):
        print(f"Error: File does not exist: {args.image1}", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.image2):
        print(f"Error: File does not exist: {args.image2}", file=sys.stderr)
        sys.exit(2)

    # Load images
    img1 = load_image(args.image1)
    img2 = load_image(args.image2)

    # Check dimensions match
    if img1.shape != img2.shape:
        print(f"Error: Image dimensions differ:", file=sys.stderr)
        print(f"  {args.image1}: {img1.shape}", file=sys.stderr)
        print(f"  {args.image2}: {img2.shape}", file=sys.stderr)
        sys.exit(2)

    # Compute PSNR
    psnr = compute_psnr(img1, img2)

    # Print results
    print(f"PSNR: {psnr:.2f} dB")

    if args.verbose:
        stats = compute_statistics(img1, img2)
        print(f"\nDetailed Statistics:")
        print(f"  Image dimensions: {img1.shape}")
        print(f"  Max pixel difference: {stats['max_difference']:.6f}")
        print(f"  Mean pixel difference: {stats['mean_difference']:.6f}")
        print(f"  Std deviation: {stats['std_difference']:.6f}")
        print(f"  Pixels different: {stats['pixels_different']} / {stats['total_pixels']}")
        print(f"  Percentage different: {100.0 * stats['pixels_different'] / stats['total_pixels']:.4f}%")

    # Determine pass/fail
    if psnr >= args.threshold:
        if args.verbose:
            print(f"\nResult: PASS (PSNR >= {args.threshold:.1f} dB)")
        sys.exit(0)
    else:
        if args.verbose:
            print(f"\nResult: FAIL (PSNR < {args.threshold:.1f} dB)")
        else:
            print(f"Images differ significantly (threshold: {args.threshold:.1f} dB)")
        sys.exit(1)


if __name__ == "__main__":
    main()
