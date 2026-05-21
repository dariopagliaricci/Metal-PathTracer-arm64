#!/usr/bin/env python3
"""
SHA-256 hash comparison tool for golden image testing.

Usage:
    python3 sha256.py <file_path>

Returns the SHA-256 hash of the specified file.
Exit codes:
    0: Success
    1: File not found or read error
"""

import hashlib
import sys
import os


def compute_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            # Read in 64kb chunks for memory efficiency
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 sha256.py <file_path>", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: File does not exist: {file_path}", file=sys.stderr)
        sys.exit(1)

    hash_value = compute_sha256(file_path)
    print(hash_value)


if __name__ == "__main__":
    main()
