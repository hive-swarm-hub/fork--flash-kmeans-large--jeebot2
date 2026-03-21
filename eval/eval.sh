#!/usr/bin/env bash
# Evaluate flash-kmeans: preflight checks, run benchmark, output summary.
# Always outputs a parseable summary block, even on failure.
set -uo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# --- Helper: output summary and exit ---
summary() {
    local throughput="${1:-0.0000}"
    local valid_wl="${2:-0}"
    local total_wl="${3:-3}"
    local line_count="${4:-0}"
    local valid="${5:-false}"
    echo "---"
    printf "throughput_mpps:   %s\n" "$throughput"
    printf "valid_workloads:   %s\n" "$valid_wl"
    printf "total_workloads:   %s\n" "$total_wl"
    printf "line_count:        %s\n" "$line_count"
    printf "valid:             %s\n" "$valid"
}

# --- Modifiable files ---
MODIFIABLE_FILES=(
    "flash_kmeans/__init__.py"
    "flash_kmeans/assign_euclid_triton.py"
    "flash_kmeans/centroid_update_triton.py"
    "flash_kmeans/kmeans_triton_impl.py"
    "flash_kmeans/kmeans_large.py"
    "flash_kmeans/interface.py"
)
MAX_LINES=2500

# --- Pre-flight checks ---

# 1. Check flash_kmeans/ directory exists
if [ ! -d "flash_kmeans" ]; then
    echo "ERROR: flash_kmeans/ directory not found." >&2
    summary "0.0000" "0" "3" "0" "false"
    exit 0
fi

# 2. Count lines across modifiable files
LINE_COUNT=0
for f in "${MODIFIABLE_FILES[@]}"; do
    if [ -f "$f" ]; then
        LINE_COUNT=$((LINE_COUNT + $(wc -l < "$f")))
    fi
done

if [ "$LINE_COUNT" -gt "$MAX_LINES" ]; then
    echo "ERROR: Total line count $LINE_COUNT exceeds limit $MAX_LINES." >&2
    summary "0.0000" "0" "3" "$LINE_COUNT" "false"
    exit 0
fi

# 3. Verify torch_fallback.py integrity (anti-tamper)
EXPECTED_HASH="8a10450f701f1329fe14bb57de7cac57c28fd5bfb502815ad6085ffc92b8d20a"
ACTUAL_HASH=$(shasum -a 256 flash_kmeans/torch_fallback.py 2>/dev/null | awk '{print $1}')
if [ -z "$ACTUAL_HASH" ]; then
    # Try sha256sum (Linux)
    ACTUAL_HASH=$(sha256sum flash_kmeans/torch_fallback.py 2>/dev/null | awk '{print $1}')
fi

if [ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]; then
    echo "ERROR: torch_fallback.py has been modified (hash mismatch)." >&2
    echo "  Expected: $EXPECTED_HASH" >&2
    echo "  Actual:   $ACTUAL_HASH" >&2
    summary "0.0000" "0" "3" "$LINE_COUNT" "false"
    exit 0
fi

# 4. Check CUDA available
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null; then
    echo "ERROR: CUDA not available." >&2
    summary "0.0000" "0" "3" "$LINE_COUNT" "false"
    exit 0
fi

# --- Run benchmark ---

TMPLOG=$(mktemp)
trap 'rm -f "$TMPLOG"' EXIT

echo "Running: python3 eval/benchmark.py (timeout 600s)" >&2

BENCH_EXIT=0
timeout 600 python3 eval/benchmark.py > "$TMPLOG" || BENCH_EXIT=$?

if [ "$BENCH_EXIT" -eq 124 ]; then
    echo "ERROR: Benchmark timed out after 600 seconds." >&2
    summary "0.0000" "0" "3" "$LINE_COUNT" "false"
    exit 0
fi

if [ "$BENCH_EXIT" -ne 0 ]; then
    echo "ERROR: Benchmark exited with code $BENCH_EXIT." >&2
    summary "0.0000" "0" "3" "$LINE_COUNT" "false"
    exit 0
fi

# --- Parse benchmark output ---

THROUGHPUT=$(grep -oP '^throughput_mpps=\K[0-9.]+' "$TMPLOG" | tail -1)
VALID_WL=$(grep -oP '^valid_workloads=\K[0-9]+' "$TMPLOG" | tail -1)
TOTAL_WL=$(grep -oP '^total_workloads=\K[0-9]+' "$TMPLOG" | tail -1)
VALID=$(grep -oP '^valid=\K(true|false)' "$TMPLOG" | tail -1)

if [ -z "$THROUGHPUT" ]; then
    echo "ERROR: Could not parse throughput from benchmark output." >&2
    summary "0.0000" "0" "3" "$LINE_COUNT" "false"
    exit 0
fi

summary "${THROUGHPUT}" "${VALID_WL:-0}" "${TOTAL_WL:-3}" "$LINE_COUNT" "${VALID:-false}"
