#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt"
SOURCE_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"
THREADS="${SPINCORE_BOARD_ONLY_THREADS:-8}"

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
RUN_DIR="$ROOT/runs/lt2_hu_preflop_board_only_averaging/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/hu_preflop_board_only_averaging.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf '=== SpinCore LT2 HU-preflop board-only averaging ===\n'
printf 'mode=READ ONLY; NO TRAINING ROOTS; NO OPTIMIZER STEPS\n'
printf 'reference=16 posterior hands x 4 boards exact1\n'
printf 'candidate=16 posterior hands x 8 boards exact0; K=1,2,4,8\n'
printf 'threads=%s\n\n' "$THREADS"

"$PYTHON_RUN" tools/audit_lt2_hu_preflop_board_only_averaging.py \
  --solver build/libspincore_solver_c.so \
  --checkpoint "$SOURCE" \
  --reference-hands 16 \
  --reference-boards-per-hand 4 \
  --candidate-hands 16 \
  --candidate-boards-per-hand 8 \
  --root-states 16 \
  --cont1-states 32 \
  --cont2-states 16 \
  --threads "$THREADS" \
  --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json, math, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
assert d.get("schema") == "SPINCORE_LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_V1"
assert d.get("completed_iteration") == 7500
m=d["method"]
assert m["new_training_roots"] == 0 and m["optimizer_steps"] == 0
assert m["candidate"]["board_average_budgets"] == [1,2,4,8]
s=d["summaries"]["ALL_ANCHORS_EQUAL_WEIGHT"]
assert s["n"] == 64
for k in ("1","2","4","8"):
    x=s["board_only_exact0"][k]
    for field in ("target_mse_to_reference","policy_tv_to_reference","candidate_policy_regret_to_reference_best_action_chips","nodes"):
        assert math.isfinite(float(x[field]["mean"]))
print("LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_POSTVALIDATION_PASS")
PY

AFTER_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [ "$AFTER_SHA" != "$SOURCE_SHA" ]; then
  echo "ERROR: source checkpoint changed during board-only audit" >&2
  exit 6
fi

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_hu_preflop_board_only_averaging.json"
fi

printf '\nLT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_PASS\n'
printf 'source_unchanged=true\n'
printf 'report=%s\n' "$REPORT"
printf 'Evidence copied to Windows Downloads when available.\n'
printf 'STOP HERE. Do not resume root training until board-only feasibility is reviewed.\n'
