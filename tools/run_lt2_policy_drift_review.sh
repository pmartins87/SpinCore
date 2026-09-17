#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SCENARIOS="${SPINCORE_POLICY_DRIFT_SCENARIOS:-3000}"
WORKERS="${SPINCORE_POLICY_DRIFT_WORKERS:-31}"
SEED=20260917
LT2A="/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt"
LT2B="/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt"
LT2A_SHA="e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c"
LT2B_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"

if [ ! -x "$PYTHON_RUN" ]; then
  echo "ERROR: lean Python environment not found: $PYTHON_RUN" >&2
  exit 3
fi
if [ ! -f "$ROOT/build/libspincore_solver_c.so" ]; then
  echo "ERROR: solver missing" >&2
  exit 4
fi
for pair in "$LT2A:$LT2A_SHA" "$LT2B:$LT2B_SHA"; do
  path="${pair%%:*}"
  expected="${pair##*:}"
  if [ ! -f "$path" ]; then
    echo "ERROR: missing checkpoint: $path" >&2
    exit 5
  fi
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [ "$actual" != "$expected" ]; then
    echo "ERROR: checkpoint hash mismatch: $path" >&2
    echo "expected=$expected actual=$actual" >&2
    exit 6
  fi
done

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lt2_policy_drift_review/$STAMP"
mkdir -p "$RUN_DIR"
A_POLICY="$RUN_DIR/lt2a_policy.pt"
B_POLICY="$RUN_DIR/lt2b_policy.pt"
REPORT="$RUN_DIR/policy_drift.json"

export PYTHONPATH="$ROOT/python"
export SPINCORE_TORCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== SpinCore LT2 policy drift review ==="
echo "comparison=LT2A_1.8M -> LT2B_4.5M"
echo "scenarios=$SCENARIOS workers=$WORKERS seed=$SEED"
echo "mode=read-only; checkpoint-independent uniform-legal probe; NO TRAINING"
echo

"$PYTHON_RUN" tools/export_lean_inference_checkpoint.py --source "$LT2A" --out "$A_POLICY"
"$PYTHON_RUN" tools/export_lean_inference_checkpoint.py --source "$LT2B" --out "$B_POLICY"

"$PYTHON_RUN" tools/compare_lt2_policy_drift.py \
  --solver build/libspincore_solver_c.so \
  --before "$A_POLICY" \
  --after "$B_POLICY" \
  --scenarios "$SCENARIOS" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --report "$REPORT"

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2A_to_LT2B_policy_drift.json"
fi

echo
echo "LT2_POLICY_DRIFT_REVIEW_PASS"
echo "report=$REPORT"
echo "STOP HERE. Do not continue training until policy drift is interpreted together with the paired weak-baseline delta."
