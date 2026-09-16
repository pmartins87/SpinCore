#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SCENARIOS="${SPINCORE_EVAL_SCENARIOS:-1000}"
WORKERS="${SPINCORE_EVAL_WORKERS:-31}"
SEED=20260915

LT0="/home/rz9/spincore_lean_functional/runs/lean_first_training/20260915_131133/checkpoint.pt"
LT1="/home/rz9/spincore_lean_functional/runs/long_training_lt1/20260915_181249/checkpoint.pt"
LT2A="/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt"

for f in "$LT0" "$LT1" "$LT2A"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing checkpoint: $f" >&2
        exit 4
    fi
done
if [ ! -x "$PYTHON_RUN" ]; then
    echo "ERROR: lean Python environment not found: $PYTHON_RUN" >&2
    exit 5
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/learning_curve_eval/$STAMP"
mkdir -p "$RUN_DIR"

export PYTHONPATH="$ROOT/python"
export SPINCORE_TORCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf '=== SpinCore learning-curve weak-baseline evaluation ===\n'
printf 'scenarios=%s workers=%s seed=%s\n' "$SCENARIOS" "$WORKERS" "$SEED"
printf 'checkpoints=120k,1.2M,1.8M roots\n\n'

# Export policy-only artifacts one source at a time. This avoids loading the
# multi-GB reservoirs independently in every evaluation worker.
for label in lt0 lt1 lt2a; do
    case "$label" in
        lt0) SRC="$LT0" ;;
        lt1) SRC="$LT1" ;;
        lt2a) SRC="$LT2A" ;;
    esac
    "$PYTHON_RUN" tools/export_lean_inference_checkpoint.py \
      --source "$SRC" --out "$RUN_DIR/${label}_policy.pt"
done

run_eval() {
    local label="$1"
    "$PYTHON_RUN" tools/evaluate_lean_strategy_quality.py \
      --solver build/libspincore_solver_c.so \
      --checkpoint "$RUN_DIR/${label}_policy.pt" \
      --scenarios "$SCENARIOS" \
      --workers "$WORKERS" \
      --seed "$SEED" \
      --report "$RUN_DIR/${label}_report.json" \
      2>&1 | tee "$RUN_DIR/${label}_eval.log"
}

run_eval lt0
run_eval lt1
run_eval lt2a

"$PYTHON_RUN" tools/compare_lean_strategy_reports.py \
  --lt0 "$RUN_DIR/lt0_report.json" \
  --lt1 "$RUN_DIR/lt1_report.json" \
  --lt2a "$RUN_DIR/lt2a_report.json" \
  --out "$RUN_DIR/comparison.json" \
  2>&1 | tee "$RUN_DIR/comparison.log"

printf '\nLEARNING_CURVE_EVAL_PASS\n'
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'comparison=%s\n' "$RUN_DIR/comparison.json"
printf 'lt0_report=%s\n' "$RUN_DIR/lt0_report.json"
printf 'lt1_report=%s\n' "$RUN_DIR/lt1_report.json"
printf 'lt2a_report=%s\n' "$RUN_DIR/lt2a_report.json"
