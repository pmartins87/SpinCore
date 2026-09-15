#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SELECTED_WORKERS_FILE="$ROOT/runs/worker_benchmark/selected_workers.txt"

if [ -n "${SPINCORE_EVAL_WORKERS:-}" ]; then
    WORKERS="$SPINCORE_EVAL_WORKERS"
elif [ -f "$SELECTED_WORKERS_FILE" ]; then
    WORKERS="$(tr -d '[:space:]' < "$SELECTED_WORKERS_FILE")"
else
    WORKERS="$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))"
fi

SCENARIOS="${SPINCORE_EVAL_SCENARIOS:-1000}"
CHECKPOINT="$(find "$ROOT/runs/lean_first_training" -type f -name checkpoint.pt -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: no lean_first_training checkpoint found." >&2
    exit 4
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/strategy_quality/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/report.json"
LOG="$RUN_DIR/eval.log"

export PYTHONPATH="$ROOT/python"
# Each evaluation worker owns one policy copy and one solver instance.  Keep
# numerical libraries single-threaded inside workers to avoid 31x oversubscription.
export SPINCORE_TORCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf '=== SpinCore strategy-quality evaluation ===\n'
printf 'checkpoint=%s\n' "$CHECKPOINT"
printf 'workers=%s scenarios=%s logical_cpus=%s\n' "$WORKERS" "$SCENARIOS" "$(nproc)"
printf 'opponents=UNIFORM_LEGAL,PASSIVE_CALLER,JAMMER\n'
printf 'control=uniform-legal hero on paired scenario/deal\n\n'

"$PYTHON_RUN" tools/evaluate_lean_strategy_quality.py \
  --solver build/libspincore_solver_c.so \
  --checkpoint "$CHECKPOINT" \
  --scenarios "$SCENARIOS" \
  --workers "$WORKERS" \
  --seed 20260915 \
  --report "$REPORT" \
  2>&1 | tee "$LOG"

printf '\nSTRATEGY_QUALITY_RUN_DONE\n'
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'report=%s\n' "$REPORT"
