#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
THREADS="${SPINCORE_TORCH_THREADS:-2}"
BUILD_JOBS="${SPINCORE_BUILD_JOBS:-$(nproc)}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lean_ryzen_pilot/$STAMP"
mkdir -p "$RUN_DIR"

printf '=== SpinCore lean Ryzen pilot ===\n'
printf 'repo=%s\n' "$ROOT"
printf 'commit=%s\n' "$(git rev-parse HEAD)"
printf 'python=%s\n' "$($PYTHON_BIN --version 2>&1)"
printf 'torch_threads=%s build_jobs=%s\n' "$THREADS" "$BUILD_JOBS"

"$PYTHON_BIN" - <<'PY'
import numpy, torch
print(f"torch={torch.__version__}")
print(f"numpy={numpy.__version__}")
PY

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$BUILD_JOBS" --target spincore_solver_c

export PYTHONPATH="$ROOT/python"
export SPINCORE_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

printf '\n=== 5 iterations / 1000 roots ===\n'
/usr/bin/time -v "$PYTHON_BIN" tools/run_lean_functional_training.py \
  --solver build/libspincore_solver_c.so \
  --seed 20260915 \
  --iterations 5 \
  --roots-per-iteration 200 \
  --exact-opponent-levels 0 \
  --advantage-steps 50 \
  --policy-steps 50 \
  --batch-size 256 \
  --checkpoint "$RUN_DIR/checkpoint.pt" \
  --report "$RUN_DIR/report.json" \
  2>&1 | tee "$RUN_DIR/training.log"

printf '\n=== 100-hand offline self-play ===\n'
"$PYTHON_BIN" tools/play_lean_functional_selfplay.py \
  --solver build/libspincore_solver_c.so \
  --checkpoint "$RUN_DIR/checkpoint.pt" \
  --hands 100 \
  --seed 20260916 \
  | tee "$RUN_DIR/selfplay.json"

printf '\nPILOT_PASS\n'
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'report=%s\n' "$RUN_DIR/report.json"
printf 'selfplay=%s\n' "$RUN_DIR/selfplay.json"
