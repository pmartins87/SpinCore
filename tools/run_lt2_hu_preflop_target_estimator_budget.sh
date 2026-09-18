#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt"
SOURCE_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"
REF_DEALS="${SPINCORE_EST_REF_DEALS:-64}"
CAND_DEALS="${SPINCORE_EST_CAND_DEALS:-64}"
ROOT_STATES="${SPINCORE_EST_ROOT_STATES:-16}"
CONT1_STATES="${SPINCORE_EST_CONT1_STATES:-32}"
CONT2_STATES="${SPINCORE_EST_CONT2_STATES:-16}"
THREADS="${SPINCORE_EST_THREADS:-8}"

if [ ! -x "$PYTHON_RUN" ]; then
  echo "ERROR: lean Python environment not found: $PYTHON_RUN" >&2
  exit 3
fi
for f in "$SOURCE" "$ROOT/build/libspincore_solver_c.so"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing required input: $f" >&2
    exit 4
  fi
done

ACTUAL_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [ "$ACTUAL_SHA" != "$SOURCE_SHA" ]; then
  echo "ERROR: Stage B checkpoint SHA mismatch" >&2
  exit 5
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lt2_hu_preflop_target_estimator_budget/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/hu_preflop_target_estimator_budget.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export SPINCORE_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf '=== SpinCore LT2 HU-preflop target-estimator budget sweep ===\n'
printf 'mode=READ ONLY; NO TRAINING ROOTS; NO OPTIMIZER STEPS\n'
printf 'anchors root=%s cont1=%s cont2plus=%s\n' "$ROOT_STATES" "$CONT1_STATES" "$CONT2_STATES"
printf 'reference exact1 hidden-deals=%s; candidate hidden-deals=%s at exact0/exact1\n' "$REF_DEALS" "$CAND_DEALS"
printf 'budgets=1,2,4,8,16,32,64 threads=%s\n\n' "$THREADS"

"$PYTHON_RUN" tools/audit_lt2_hu_preflop_target_estimator_budget.py \
  --solver build/libspincore_solver_c.so \
  --checkpoint "$SOURCE" \
  --reference-deals "$REF_DEALS" \
  --candidate-deals "$CAND_DEALS" \
  --root-states "$ROOT_STATES" \
  --cont1-states "$CONT1_STATES" \
  --cont2-states "$CONT2_STATES" \
  --threads "$THREADS" \
  --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json, math, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
assert d.get("schema") == "SPINCORE_LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_V1"
assert d.get("completed_iteration") == 7500
m=d["method"]
assert m["new_training_roots"] == 0 and m["optimizer_steps"] == 0
assert m["candidate_exact_levels"] == [0,1]
assert m["budgets"] == [1,2,4,8,16,32,64]
s=d["summaries"]["ALL_ANCHORS_EQUAL_WEIGHT"]
assert s["n"] == 64
for level in ("exact_level_0","exact_level_1"):
    for k in ("1","2","4","8","16","32","64"):
        x=s["candidate_estimators"][level][k]
        for field in ("target_mse_to_reference","policy_tv_to_reference","candidate_policy_regret_to_reference_best_action_chips","nodes"):
            assert math.isfinite(float(x[field]["mean"]))
print("LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_POSTVALIDATION_PASS")
PY

AFTER_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [ "$AFTER_SHA" != "$SOURCE_SHA" ]; then
  echo "ERROR: source checkpoint changed during target-estimator audit" >&2
  exit 6
fi

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_hu_preflop_target_estimator_budget.json"
fi

printf '\nLT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_PASS\n'
printf 'source_unchanged=true\n'
printf 'report=%s\n' "$REPORT"
printf 'Evidence copied to Windows Downloads when available.\n'
printf 'STOP HERE. Do not resume root training until the compute-normalized estimator frontier is reviewed.\n'
