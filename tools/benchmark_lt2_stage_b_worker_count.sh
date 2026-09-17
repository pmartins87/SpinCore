#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOURCE="${SPINCORE_LT2B_SOURCE_CHECKPOINT:-/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt}"
EXPECTED_SOURCE_SHA256="e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c"

if [ ! -x "$PYTHON_RUN" ]; then
    echo "ERROR: lean Python environment not found: $PYTHON_RUN" >&2
    exit 3
fi
if [ ! -f "$SOURCE" ]; then
    echo "ERROR: LT2 Stage A checkpoint not found: $SOURCE" >&2
    exit 4
fi
if [ ! -f "$ROOT/build/libspincore_solver_c.so" ]; then
    echo "ERROR: solver missing: $ROOT/build/libspincore_solver_c.so" >&2
    exit 5
fi
ACTUAL_SHA256="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [ "$ACTUAL_SHA256" != "$EXPECTED_SOURCE_SHA256" ]; then
    echo "ERROR: LT2 Stage A checkpoint SHA256 mismatch." >&2
    exit 6
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked source/index changes present; benchmark requires HEAD." >&2
    exit 7
fi

printf '=== LT2 Stage B throughput tuning gate ===\n'
printf 'logical_cpus=%s\n' "$(nproc)"
printf 'cpu_model=%s\n' "$(awk -F: '/model name/ {gsub(/^ +/,"",$2); print $2; exit}' /proc/cpuinfo)"
printf 'mem_total_gib='; awk '/^MemTotal:/ {printf "%.2f\n", $2/1024/1024}' /proc/meminfo
if command -v powershell.exe >/dev/null 2>&1; then
    printf 'windows_power_scheme=' 
    powershell.exe -NoProfile -Command 'powercfg /getactivescheme' 2>/dev/null | tr -d '\r' | tail -n 1 || true
fi
printf '\nThis benchmark does not save or continue the training checkpoint.\n'
printf 'It tests workers=31,16,20,24,28 at the proven 8 Torch threads.\n\n'

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/runs/lt2_stage_b_worker_benchmark/$STAMP"
mkdir -p "$(dirname "$OUT")"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export SPINCORE_TORCH_THREADS=8

set +e
"$PYTHON_RUN" tools/benchmark_lt2_stage_b_worker_count.py \
  --checkpoint "$SOURCE" \
  --solver build/libspincore_solver_c.so \
  --workers 31,16,20,24,28 \
  --threads 8 \
  --iterations 6 \
  --out "$OUT"
STATUS=$?
set -e

REPORT="$OUT/report.json"
WIN_DEST="/mnt/c/Users/Rz9/Downloads/SpinCore_LT2_STAGE_B_worker_benchmark.json"
if [ -f "$REPORT" ] && [ -d "/mnt/c/Users/Rz9/Downloads" ]; then
    cp "$REPORT" "$WIN_DEST"
    printf '\nCopied report to Windows Downloads:\n%s\n' "$WIN_DEST"
    if command -v explorer.exe >/dev/null 2>&1; then
        explorer.exe /select,"$(wslpath -w "$WIN_DEST")" >/dev/null 2>&1 || true
    fi
fi

printf '\nSTOP HERE. Do not start LT2 Stage B yet.\n'
printf 'Send the benchmark report to ChatGPT.\n'
exit "$STATUS"
