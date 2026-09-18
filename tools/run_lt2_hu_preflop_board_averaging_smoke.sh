#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt"
SOURCE_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"
ROOTS="${SPINCORE_BOARDAVG_SMOKE_ROOTS:-8}"

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
RUN_DIR="$ROOT/runs/lt2_hu_preflop_board_averaging_smoke/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/board_averaging_smoke.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf '=== SpinCore LT2 HU-preflop board-averaging mechanics smoke ===\n'
printf 'mode=READ ONLY; NO TRAINING MEMORY WRITES; NO OPTIMIZER STEPS\n'
printf 'same HU roots compared at K1 vs K4; roots=%s\n\n' "$ROOTS"

"$PYTHON_RUN" tools/smoke_lt2_hu_preflop_board_averaging.py \
  --solver build/libspincore_solver_c.so \
  --checkpoint "$SOURCE" \
  --roots "$ROOTS" \
  --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json, math, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
assert d.get("schema") == "SPINCORE_LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_V1"
assert d.get("completed_iteration") == 7500
assert d.get("read_only") is True
assert d.get("optimizer_steps") == 0
assert d.get("training_memory_writes") == 0
assert int(d.get("roots_compared", 0)) > 0
assert d.get("k1") == 1 and d.get("k4") == 4
assert int(d.get("preflop_samples", 0)) > 0
assert int(d.get("preflop_changed_targets", 0)) > 0
assert int(d.get("postflop_nonidentical_targets", -1)) == 0
assert int(d.get("nodes_k4", 0)) > int(d.get("nodes_k1", 0))
assert math.isfinite(float(d.get("node_multiplier_k4_over_k1")))
c=d["contract"]
assert all(bool(c[k]) for k in (
    "same_root_jobs",
    "same_sample_count",
    "same_sample_order_and_identity",
    "canonical_postflop_targets_unchanged",
    "preflop_targets_board_averaged",
    "canonical_rng_progression_preserved_by_implementation",
))
print("LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_POSTVALIDATION_PASS")
PY

AFTER_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [ "$AFTER_SHA" != "$SOURCE_SHA" ]; then
  echo "ERROR: source checkpoint changed during board-averaging smoke" >&2
  exit 6
fi

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_hu_preflop_board_averaging_smoke.json"
fi

printf '\nLT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_PASS\n'
printf 'source_unchanged=true\n'
printf 'report=%s\n' "$REPORT"
printf 'Evidence copied to Windows Downloads when available.\n'
printf 'STOP HERE. Do not start the training pilot until this mechanics smoke is reviewed.\n'
