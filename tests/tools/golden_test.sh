#!/usr/bin/env bash

# Golden Image Test Orchestrator
#
# This script automates the golden image testing workflow:
# 1. Build the headless path tracer
# 2. Generate renders with fixed parameters
# 3. Compare using SHA-256 and PSNR metrics
#
# Usage:
#   ./golden_test.sh generate before|after
#   ./golden_test.sh compare [--verbose]
#   ./golden_test.sh clean

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GOLDEN_DIR="${PROJECT_ROOT}/tests/golden"
TOOLS_DIR="${PROJECT_ROOT}/tests/tools"

# Default render settings (fast for CI)
WIDTH=256
HEIGHT=144
SPP=512
SEED=42
SCENE_NAME="ajax"
SCENE_ID="assets/ajax.scene"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect build directory
if [ -d "${PROJECT_ROOT}/build" ]; then
    BUILD_DIR="${PROJECT_ROOT}/build"
elif [ -d "${PROJECT_ROOT}/cmake-build-debug" ]; then
    BUILD_DIR="${PROJECT_ROOT}/cmake-build-debug"
elif [ -d "${PROJECT_ROOT}/cmake-build-release" ]; then
    BUILD_DIR="${PROJECT_ROOT}/cmake-build-release"
else
    BUILD_DIR="${PROJECT_ROOT}/build"
fi

HEADLESS_BIN="${BUILD_DIR}/PathTracerHeadless"

# Print usage
usage() {
    cat <<EOF
Usage: $0 COMMAND [OPTIONS]

Commands:
    generate before|after   Generate a golden image render
    compare                 Compare before/after renders
    clean                   Remove generated images
    help                    Show this help message

Generate Options:
    --width WIDTH           Image width (default: ${WIDTH})
    --height HEIGHT         Image height (default: ${HEIGHT})
    --spp SAMPLES           Samples per pixel (default: ${SPP})
    --seed SEED             RNG seed (default: ${SEED})
    --scene PATH            Scene path (default: ${SCENE_ID})
    --scene-name NAME       Override output file name

Compare Options:
    --verbose, -v           Show detailed comparison output
    --threshold DB          PSNR threshold in dB (default: 60.0)
    --scene NAME|PATH       Scene name (or path, for convenience)

Examples:
    # Generate baseline before making changes
    $0 generate before

    # After refactoring, generate new render
    $0 generate after

    # Compare the two renders
    $0 compare --verbose

    # High-quality validation render
    $0 generate before --width 512 --height 288 --spp 1024
EOF
    exit 0
}

# Print error and exit
error() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

# Print info
info() {
    echo -e "${BLUE}$1${NC}"
}

# Print success
success() {
    echo -e "${GREEN}$1${NC}"
}

# Print warning
warning() {
    echo -e "${YELLOW}$1${NC}"
}

scene_name_from_id() {
    local scene_id="$1"
    local base="${scene_id##*/}"
    base="${base%.scene}"
    echo "${base}"
}

# Build the headless renderer
build_renderer() {
    info "Building headless path tracer..."

    if [ ! -d "${BUILD_DIR}" ]; then
        info "Creating build directory..."
        cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}"
    fi

    cmake --build "${BUILD_DIR}" --target PathTracerHeadless || \
        error "Failed to build PathTracerHeadless"

    if [ ! -f "${HEADLESS_BIN}" ]; then
        error "PathTracerHeadless binary not found at ${HEADLESS_BIN}"
    fi

    success "Build complete!"
}

# Generate a golden image
generate() {
    local TARGET="$1"
    shift

    if [ "${TARGET}" != "before" ] && [ "${TARGET}" != "after" ]; then
        error "Target must be 'before' or 'after'"
    fi

    # Parse additional options
    while [ $# -gt 0 ]; do
        case "$1" in
            --width) WIDTH="$2"; shift 2;;
            --height) HEIGHT="$2"; shift 2;;
            --spp) SPP="$2"; shift 2;;
            --seed) SEED="$2"; shift 2;;
            --scene)
                SCENE_ID="$2"
                SCENE_NAME="$(scene_name_from_id "${SCENE_ID}")"
                shift 2
                ;;
            --scene-name) SCENE_NAME="$2"; shift 2;;
            *) error "Unknown option: $1";;
        esac
    done

    local OUTPUT_DIR="${GOLDEN_DIR}/${TARGET}"
    local OUTPUT_FILE="${OUTPUT_DIR}/${SCENE_NAME}.ppm"

    info "Generating golden image: ${TARGET}/${SCENE_NAME}.ppm"
    info "Settings: ${WIDTH}x${HEIGHT} @ ${SPP} spp, seed=${SEED}"

    # Ensure output directory exists
    mkdir -p "${OUTPUT_DIR}"

    # Build if necessary
    if [ ! -f "${HEADLESS_BIN}" ]; then
        build_renderer
    fi

    # Generate the render
    info "Rendering (this may take a minute)..."
    HEADLESS_ARGS=(
        "${HEADLESS_BIN}"
        --width "${WIDTH}"
        --height "${HEIGHT}"
        --sppTotal "${SPP}"
        --seed "${SEED}"
        --output "${OUTPUT_FILE}"
        --format ppm
        --verbose
    )

    if [ -n "${SCENE_ID}" ]; then
        HEADLESS_ARGS+=(--scene "${SCENE_ID}")
    fi

    "${HEADLESS_ARGS[@]}" || error "Render failed"

    # Verify output exists
    if [ ! -f "${OUTPUT_FILE}" ]; then
        error "Output file was not created: ${OUTPUT_FILE}"
    fi

    local FILE_SIZE=$(wc -c < "${OUTPUT_FILE}" | tr -d ' ')
    success "Generated: ${OUTPUT_FILE} (${FILE_SIZE} bytes)"

    # Show hash
    local HASH=$(python3 "${TOOLS_DIR}/sha256.py" "${OUTPUT_FILE}")
    info "SHA-256: ${HASH}"
}

# Compare before and after renders
compare() {
    local VERBOSE=0
    local THRESHOLD=60.0

    # Parse options
    while [ $# -gt 0 ]; do
        case "$1" in
            --verbose|-v) VERBOSE=1; shift;;
            --threshold) THRESHOLD="$2"; shift 2;;
            --scene)
                SCENE_NAME="$2"
                if [[ "${SCENE_NAME}" == *"/"* || "${SCENE_NAME}" == *.scene ]]; then
                    SCENE_NAME="$(scene_name_from_id "${SCENE_NAME}")"
                fi
                shift 2
                ;;
            *) error "Unknown option: $1";;
        esac
    done

    local BEFORE_FILE="${GOLDEN_DIR}/before/${SCENE_NAME}.ppm"
    local AFTER_FILE="${GOLDEN_DIR}/after/${SCENE_NAME}.ppm"

    info "Comparing golden images: ${SCENE_NAME}.ppm"

    # Check files exist
    if [ ! -f "${BEFORE_FILE}" ]; then
        error "Baseline not found: ${BEFORE_FILE}\nRun: $0 generate before"
    fi

    if [ ! -f "${AFTER_FILE}" ]; then
        error "New render not found: ${AFTER_FILE}\nRun: $0 generate after"
    fi

    # Step 1: SHA-256 comparison (fastest)
    info "Computing SHA-256 hashes..."
    local BEFORE_HASH=$(python3 "${TOOLS_DIR}/sha256.py" "${BEFORE_FILE}")
    local AFTER_HASH=$(python3 "${TOOLS_DIR}/sha256.py" "${AFTER_FILE}")

    echo "  Before: ${BEFORE_HASH}"
    echo "  After:  ${AFTER_HASH}"

    if [ "${BEFORE_HASH}" = "${AFTER_HASH}" ]; then
        echo ""
        success "✓ PASS: Images are byte-for-byte identical (SHA-256 match)"
        return 0
    else
        warning "⚠ SHA-256 hashes differ, checking visual similarity..."
    fi

    # Step 2: PSNR comparison (visual similarity)
    info "Computing PSNR..."

    local PSNR_ARGS=("${BEFORE_FILE}" "${AFTER_FILE}" "--threshold" "${THRESHOLD}")
    if [ ${VERBOSE} -eq 1 ]; then
        PSNR_ARGS+=("--verbose")
    fi

    if python3 "${TOOLS_DIR}/psnr.py" "${PSNR_ARGS[@]}"; then
        echo ""
        success "✓ PASS: Images are visually identical (PSNR ≥ ${THRESHOLD} dB)"
        return 0
    else
        local EXIT_CODE=$?
        echo ""
        if [ ${EXIT_CODE} -eq 2 ]; then
            error "✗ ERROR: PSNR comparison failed (check image dimensions)"
        else
            error "✗ FAIL: Images differ significantly (PSNR < ${THRESHOLD} dB)"
        fi
    fi
}

# Clean generated images
clean() {
    info "Cleaning generated images..."

    if [ -d "${GOLDEN_DIR}/after" ]; then
        rm -rf "${GOLDEN_DIR}/after"
        success "Removed: ${GOLDEN_DIR}/after"
    fi

    info "Baseline images preserved in: ${GOLDEN_DIR}/before"
    info "To remove baseline: rm -rf ${GOLDEN_DIR}/before"
}

# Main
main() {
    if [ $# -eq 0 ]; then
        usage
    fi

    local COMMAND="$1"
    shift

    case "${COMMAND}" in
        generate)
            if [ $# -eq 0 ]; then
                error "generate requires 'before' or 'after' argument"
            fi
            generate "$@"
            ;;
        compare)
            compare "$@"
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            error "Unknown command: ${COMMAND}\nUse '$0 help' for usage"
            ;;
    esac
}

main "$@"
