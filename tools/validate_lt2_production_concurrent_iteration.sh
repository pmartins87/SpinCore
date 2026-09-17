#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOURCE="${SPINCORE_LT2A_CHECKPOINT:-/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt}"
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
    echo "expected=$EXPECTED_SOURCE_SHA256" >&2
    echo "actual=$ACTUAL_SHA256" >&2
    exit 6
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked source/index changes present; gate requires HEAD." >&2
    exit 7
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/runs/lt2_production_concurrent_iteration_parity/$STAMP"
mkdir -p "$(dirname "$OUT")"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export SPINCORE_TORCH_THREADS=8

"$PYTHON_RUN" tools/validate_lt2_production_concurrent_iteration.py \
  --checkpoint "$SOURCE" \
  --solver build/libspincore_solver_c.so \
  --workers 31 \
  --threads 8 \
  --out "$OUT"

REPORT="$OUT/report.json"
WIN_DEST="/mnt/c/Users/Rz9/Downloads/SpinCore_LT2_production_concurrent_iteration_parity.json"
if [ -d "/mnt/c/Users/Rz9/Downloads" ]; then
    cp "$REPORT" "$WIN_DEST"
    printf '\nCopied report to Windows Downloads:\n%s\n' "$WIN_DEST"
    if command -v explorer.exe >/dev/null 2>&1; then
        explorer.exe /select,"$(wslpath -w "$WIN_DEST")" >/dev/null 2>&1 || true
    fi
fi

printf '\nSTOP HERE. Do not start LT2 Stage B yet.\n'
printf 'Send this file to ChatGPT:\n%s\n' "$REPORT"
