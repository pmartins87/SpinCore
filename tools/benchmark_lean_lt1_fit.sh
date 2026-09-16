#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOURCE="${1:-$ROOT/runs/long_training_lt1/20260915_181249/checkpoint.pt}"
OUT="$ROOT/runs/lt1_fit_benchmark/$(date +%Y%m%d_%H%M%S)_$$"
WORKERS="$(tr -d '[:space:]' < "$ROOT/runs/worker_benchmark/selected_workers.txt")"
export PYTHONPATH="$ROOT/python"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
[ -f "$SOURCE" ] || { echo "Checkpoint not found: $SOURCE"; exit 2; }
[ -x "$PYTHON_RUN" ] || { echo "Python environment not found"; exit 2; }
# Isolated process; an outer timeout bounds a malfunction without touching LT1.
mkdir -p "$(dirname "$OUT")"
timeout --signal=TERM 20m "$PYTHON_RUN" -u tools/benchmark_lean_lt1_fit.py \
  --checkpoint "$SOURCE" --workers "$WORKERS" --out "$OUT" \
  2>&1 | tee "${OUT}.log"
