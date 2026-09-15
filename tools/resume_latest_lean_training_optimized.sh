#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
THREADS="${SPINCORE_TORCH_THREADS:-2}"
SELECTED_FILE="$ROOT/runs/worker_benchmark/selected_workers.txt"

if [ -n "${SPINCORE_WORKERS:-}" ]; then
    WORKERS="$SPINCORE_WORKERS"
elif [ -f "$SELECTED_FILE" ]; then
    WORKERS="$(tr -d '[:space:]' < "$SELECTED_FILE")"
else
    WORKERS="$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))"
fi

CHECKPOINT="$(find "$ROOT/runs/lean_first_training" -type f -name checkpoint.pt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: no lean_first_training checkpoint found; use run_lean_functional_first_training.sh." >&2
    exit 4
fi
RUN_DIR="$(dirname "$CHECKPOINT")"

export PYTHONPATH="$ROOT/python"
export SPINCORE_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

printf '=== Resume latest SpinCore training — Ryzen optimized ===\n'
printf 'checkpoint=%s\n' "$CHECKPOINT"
printf 'workers=%s parent_torch_threads=%s logical_cpus=%s\n' "$WORKERS" "$THREADS" "$(nproc)"

"$PYTHON_RUN" tools/run_lean_functional_training.py \
  --solver build/libspincore_solver_c.so \
  --resume \
  --workers "$WORKERS" \
  --checkpoint-every 5 \
  --checkpoint "$CHECKPOINT" \
  --report "$RUN_DIR/report.json" \
  2>&1 | tee -a "$RUN_DIR/training.log"

printf '\n=== 5000-hand offline self-play ===\n'
"$PYTHON_RUN" tools/play_lean_functional_selfplay.py \
  --solver build/libspincore_solver_c.so \
  --checkpoint "$CHECKPOINT" \
  --hands 5000 \
  --seed 20260918 \
  | tee "$RUN_DIR/selfplay.json"

printf '\nOPTIMIZED_RESUME_PASS\n'
printf 'run_dir=%s\n' "$RUN_DIR"
