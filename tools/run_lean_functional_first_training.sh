#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
THREADS="${SPINCORE_TORCH_THREADS:-2}"
BUILD_JOBS="${SPINCORE_BUILD_JOBS:-$(nproc)}"
SELECTED_FILE="$ROOT/runs/worker_benchmark/selected_workers.txt"
if [ -n "${SPINCORE_WORKERS:-}" ]; then
    WORKERS="$SPINCORE_WORKERS"
elif [ -f "$SELECTED_FILE" ]; then
    WORKERS="$(tr -d '[:space:]' < "$SELECTED_FILE")"
else
    WORKERS="$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))"
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lean_first_training/$STAMP"
mkdir -p "$RUN_DIR"

if [ ! -x "$PYTHON_RUN" ]; then
    echo "ERROR: lean Python environment not found at $PYTHON_RUN" >&2
    echo "Run tools/run_lean_functional_ryzen_pilot.sh once, then rerun this command." >&2
    exit 3
fi
if ! "$PYTHON_RUN" - <<'PY' >/dev/null 2>&1
import numpy, torch
PY
then
    echo "ERROR: lean Python environment is incomplete." >&2
    echo "Rerun tools/run_lean_functional_ryzen_pilot.sh once, then rerun this command." >&2
    exit 4
fi

printf '=== SpinCore first substantive training — Ryzen optimized ===\n'
printf 'repo=%s\n' "$ROOT"
printf 'commit=%s\n' "$(git rev-parse HEAD)"
printf 'python=%s\n' "$($PYTHON_RUN --version 2>&1)"
printf 'parent_torch_threads=%s root_workers=%s build_jobs=%s logical_cpus=%s\n' "$THREADS" "$WORKERS" "$BUILD_JOBS" "$(nproc)"
printf 'profile=200 iterations x 600 roots = 120000 roots\n'
printf 'legacy-scale note: 600 roots/iteration => 1527 advantage traversals and 255 sampled-policy episodes/iteration, approximately the mature DeepSpin defaults (1536 / 256).\n'
printf 'parallelism=independent advantage roots only; one Torch thread per worker; parent merges samples into the same authoritative reservoirs.\n'
printf 'run_dir=%s\n' "$RUN_DIR"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$BUILD_JOBS" --target spincore_solver_c

export PYTHONPATH="$ROOT/python"
export SPINCORE_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

printf '\n=== Training ===\n'
/usr/bin/time -v "$PYTHON_RUN" tools/run_lean_functional_training.py \
  --solver build/libspincore_solver_c.so \
  --seed 20260915 \
  --iterations 200 \
  --roots-per-iteration 600 \
  --exact-opponent-levels 0 \
  --reservoir-capacity 100000 \
  --advantage-steps 50 \
  --policy-steps 400 \
  --batch-size 256 \
  --workers "$WORKERS" \
  --checkpoint-every 5 \
  --checkpoint "$RUN_DIR/checkpoint.pt" \
  --report "$RUN_DIR/report.json" \
  2>&1 | tee "$RUN_DIR/training.log"

printf '\n=== 5000-hand offline self-play ===\n'
"$PYTHON_RUN" tools/play_lean_functional_selfplay.py \
  --solver build/libspincore_solver_c.so \
  --checkpoint "$RUN_DIR/checkpoint.pt" \
  --hands 5000 \
  --seed 20260918 \
  | tee "$RUN_DIR/selfplay.json"

printf '\nFIRST_TRAINING_PASS\n'
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'checkpoint=%s\n' "$RUN_DIR/checkpoint.pt"
printf 'report=%s\n' "$RUN_DIR/report.json"
printf 'selfplay=%s\n' "$RUN_DIR/selfplay.json"
printf '\nThis checkpoint can later be extended in-place with --resume --additional-iterations N.\n'
