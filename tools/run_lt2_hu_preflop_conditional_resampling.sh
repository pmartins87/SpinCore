#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt"
SOURCE_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"
HANDS="${SPINCORE_COND_HANDS:-16}"
BOARDS="${SPINCORE_COND_BOARDS:-4}"
REPEATS="${SPINCORE_COND_REPEATS:-4}"
ROOT_STATES="${SPINCORE_COND_ROOT_STATES:-16}"
CONT1_STATES="${SPINCORE_COND_CONT1_STATES:-32}"
CONT2_STATES="${SPINCORE_COND_CONT2_STATES:-16}"
THREADS="${SPINCORE_COND_THREADS:-8}"

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
RUN_DIR="$ROOT/runs/lt2_hu_preflop_conditional_resampling/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/hu_preflop_conditional_resampling.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export SPINCORE_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf '=== SpinCore LT2 HU-preflop conditional resampling ===\n'
printf 'mode=READ ONLY; NO TRAINING ROOTS; NO OPTIMIZER STEPS\n'
printf 'anchors root=%s cont1=%s cont2plus=%s\n' "$ROOT_STATES" "$CONT1_STATES" "$CONT2_STATES"
printf 'nested resampling hands=%s boards/hand=%s repeats/deal=%s exact_level=1\n' "$HANDS" "$BOARDS" "$REPEATS"
printf 'threads=%s\n\n' "$THREADS"

"$PYTHON_RUN" tools/audit_lt2_hu_preflop_conditional_resampling.py \
  --solver build/libspincore_solver_c.so \
  --checkpoint "$SOURCE" \
  --hands-per-state "$HANDS" \
  --boards-per-hand "$BOARDS" \
  --repeats-per-deal "$REPEATS" \
  --root-states "$ROOT_STATES" \
  --cont1-states "$CONT1_STATES" \
  --cont2-states "$CONT2_STATES" \
  --threads "$THREADS" \
  --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json, math, sys
from pathlib import Path

p=Path(sys.argv[1]); d=json.loads(p.read_text())
assert d.get("schema") == "SPINCORE_LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_V1"
assert d.get("completed_iteration") == 7500
method=d["method"]
assert method["new_training_roots"] == 0
assert method["exact_opponent_levels"] == 1
rows=d.get("rows") or []
expected=sum(int(x) for x in d["method"]["anchor_region_quotas"].values())
assert len(rows) == expected and expected > 0
for row in rows:
    m=row["mse_components"]
    lhs=float(m["sample_target_mse"])
    rhs=(float(m["within_deal_opponent_action"])
         +float(m["future_board_within_hand"])
         +float(m["opponent_hand_posterior"])
         +float(m["model_to_conditional_mean"]))
    assert math.isfinite(lhs) and math.isfinite(rhs)
    assert abs(lhs-rhs) < 2e-6, (row["anchor_index"], lhs, rhs)
    post=row["posterior"]
    assert post["ordered_hand_support"] == 2450
    assert post["positive_weight_hands"] > 0
    assert post["effective_sample_size"] > 0.0
print("LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_POSTVALIDATION_PASS")
PY

AFTER_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [ "$AFTER_SHA" != "$SOURCE_SHA" ]; then
  echo "ERROR: source checkpoint changed during conditional resampling" >&2
  exit 6
fi

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_hu_preflop_conditional_resampling.json"
fi

printf '\nLT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_PASS\n'
printf 'source_unchanged=true\n'
printf 'report=%s\n' "$REPORT"
printf 'Evidence copied to Windows Downloads when available.\n'
printf 'STOP HERE. Do not resume root training until this conditional decomposition is reviewed.\n'
