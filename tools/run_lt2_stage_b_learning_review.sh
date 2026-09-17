#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SCENARIOS="${SPINCORE_EVAL_SCENARIOS:-1000}"
WORKERS="${SPINCORE_EVAL_WORKERS:-31}"
SEED=20260915

LT2A="/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt"
LT2B="/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt"
LT2A_SHA="e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c"
LT2B_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"

if [ ! -x "$PYTHON_RUN" ]; then
    echo "ERROR: lean Python environment not found: $PYTHON_RUN" >&2
    exit 3
fi
if [ ! -f "$ROOT/build/libspincore_solver_c.so" ]; then
    echo "ERROR: solver missing: $ROOT/build/libspincore_solver_c.so" >&2
    exit 4
fi
for f in "$LT2A" "$LT2B"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing checkpoint: $f" >&2
        exit 5
    fi
done
if [ "$(sha256sum "$LT2A" | awk '{print $1}')" != "$LT2A_SHA" ]; then
    echo "ERROR: LT2A checkpoint hash mismatch" >&2
    exit 6
fi
if [ "$(sha256sum "$LT2B" | awk '{print $1}')" != "$LT2B_SHA" ]; then
    echo "ERROR: LT2B checkpoint hash mismatch" >&2
    exit 7
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked source/index changes present; learning review requires HEAD." >&2
    exit 8
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lt2_stage_b_learning_review/$STAMP"
mkdir -p "$RUN_DIR"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export SPINCORE_TORCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf '=== SpinCore LT2 Stage B learning review ===\n'
printf 'comparison=LT2A_1.8M -> LT2B_4.5M\n'
printf 'scenarios=%s workers=%s seed=%s\n' "$SCENARIOS" "$WORKERS" "$SEED"
printf 'mode=read-only policy evaluation; NO TRAINING\n\n'

printf 'Exporting policy-only inference checkpoints...\n'
"$PYTHON_RUN" tools/export_lean_inference_checkpoint.py --source "$LT2A" --out "$RUN_DIR/lt2a_policy.pt"
"$PYTHON_RUN" tools/export_lean_inference_checkpoint.py --source "$LT2B" --out "$RUN_DIR/lt2b_policy.pt"

printf '\nEvaluating LT2 Stage A...\n'
"$PYTHON_RUN" tools/evaluate_lean_strategy_quality_with_rows.py \
  --solver build/libspincore_solver_c.so \
  --checkpoint "$RUN_DIR/lt2a_policy.pt" \
  --scenarios "$SCENARIOS" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --report "$RUN_DIR/lt2a_report.json" \
  --rows-report "$RUN_DIR/lt2a_rows.json" \
  2>&1 | tee "$RUN_DIR/lt2a_eval.log"

printf '\nEvaluating LT2 Stage B...\n'
"$PYTHON_RUN" tools/evaluate_lean_strategy_quality_with_rows.py \
  --solver build/libspincore_solver_c.so \
  --checkpoint "$RUN_DIR/lt2b_policy.pt" \
  --scenarios "$SCENARIOS" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --report "$RUN_DIR/lt2b_report.json" \
  --rows-report "$RUN_DIR/lt2b_rows.json" \
  2>&1 | tee "$RUN_DIR/lt2b_eval.log"

printf '\nComputing paired checkpoint delta...\n'
"$PYTHON_RUN" tools/compare_lean_checkpoint_delta.py \
  --before "$RUN_DIR/lt2a_rows.json" \
  --after "$RUN_DIR/lt2b_rows.json" \
  --before-label LT2A_1P8M \
  --after-label LT2B_4P5M \
  --out "$RUN_DIR/checkpoint_delta.json" \
  2>&1 | tee "$RUN_DIR/checkpoint_delta.log"

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
    cp "$RUN_DIR/lt2a_report.json" "$DEST/SpinCore_LT2A_learning_report.json"
    cp "$RUN_DIR/lt2b_report.json" "$DEST/SpinCore_LT2B_learning_report.json"
    cp "$RUN_DIR/checkpoint_delta.json" "$DEST/SpinCore_LT2A_to_LT2B_checkpoint_delta.json"
fi

printf '\nLT2_STAGE_B_LEARNING_REVIEW_EVAL_PASS\n'
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'lt2b_report=%s\n' "$RUN_DIR/lt2b_report.json"
printf 'delta_report=%s\n' "$RUN_DIR/checkpoint_delta.json"
printf 'STOP HERE. Do not continue training until the Stage A -> Stage B learning delta is reviewed.\n'
