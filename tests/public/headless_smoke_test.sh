#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -ge 1 ]]; then
  BINARY_PATH="$1"
  shift
else
  BINARY_PATH="${ROOT_DIR}/build/PathTracerHeadless"
fi
declare -a EXTRA_ARGS=("$@")
SCENE_PATH="${ROOT_DIR}/tests/scenes/smoke.scene"

if [[ ! -x "${BINARY_PATH}" ]]; then
  echo "error: PathTracerHeadless binary not found or not executable: ${BINARY_PATH}" >&2
  exit 1
fi

if [[ ! -f "${SCENE_PATH}" ]]; then
  echo "error: smoke scene not found: ${SCENE_PATH}" >&2
  exit 1
fi

OUT_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${OUT_DIR}"
}
trap cleanup EXIT

OUT_FILE="${OUT_DIR}/smoke.ppm"

if (( ${#EXTRA_ARGS[@]} > 0 )); then
  "${BINARY_PATH}" \
    --scene="${SCENE_PATH}" \
    --width=64 \
    --height=64 \
    --sppTotal=4 \
    --maxDepth=4 \
    --seed=1337 \
    --enableSoftwareRayTracing=1 \
    --format=ppm \
    --output="${OUT_FILE}" \
    "${EXTRA_ARGS[@]}"
else
  "${BINARY_PATH}" \
    --scene="${SCENE_PATH}" \
    --width=64 \
    --height=64 \
    --sppTotal=4 \
    --maxDepth=4 \
    --seed=1337 \
    --enableSoftwareRayTracing=1 \
    --format=ppm \
    --output="${OUT_FILE}"
fi

if [[ ! -s "${OUT_FILE}" ]]; then
  echo "error: smoke render failed, output file missing or empty: ${OUT_FILE}" >&2
  exit 1
fi

bytes="$(wc -c < "${OUT_FILE}")"
echo "smoke test passed: ${OUT_FILE} (${bytes} bytes)"
