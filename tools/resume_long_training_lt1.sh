#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
WORKER_FILE="$ROOT/runs/worker_benchmark/selected_workers.txt"
THREAD_FILE="$ROOT/runs/worker_benchmark/selected_torch_threads.txt"

if [ ! -x "$PYTHON_RUN" ]; then
    echo "ERROR: lean Python environment not found." >&2
    exit 3
fi
if [ ! -f "$WORKER_FILE" ] || [ ! -f "$THREAD_FILE" ]; then
    echo "ERROR: Ryzen worker/thread profile missing." >&2
    exit 4
fi

WORKERS="$(tr -d '[:space:]' < "$WORKER_FILE")"
THREADS="$(tr -d '[:space:]' < "$THREAD_FILE")"
CHECKPOINT="$(find "$ROOT/runs/long_training_lt1" -type f -name checkpoint.pt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: no LT1 checkpoint found." >&2
    exit 5
fi
RUN_DIR="$(dirname "$CHECKPOINT")"
ADDITIONAL="${SPINCORE_LONG_ADDITIONAL_ITERATIONS:-2000}"

export PYTHONPATH="$ROOT/python"
export SPINCORE_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

printf '=== Resume SpinCore long-training campaign ===\n'
printf 'checkpoint=%s\n' "$CHECKPOINT"
printf 'additional_iterations=%s\n' "$ADDITIONAL"
printf 'workers=%s parent_torch_threads=%s logical_cpus=%s\n' "$WORKERS" "$THREADS" "$(nproc)"
printf 'note=this preserves the large-reservoir campaign; it does not restart training\n\n'

/usr/bin/time -v "$PYTHON_RUN" tools/run_lean_functional_training.py \
  --solver build/libspincore_solver_c.so \
  --resume \
  --additional-iterations "$ADDITIONAL" \
  --workers "$WORKERS" \
  --checkpoint-every 100 \
  --checkpoint "$CHECKPOINT" \
  --report "$RUN_DIR/report.json" \
  2>&1 | tee -a "$RUN_DIR/training.log"

printf '\nLONG_TRAINING_RESUME_PASS\n'
printf 'run_dir=%s\n' "$RUN_DIR"
