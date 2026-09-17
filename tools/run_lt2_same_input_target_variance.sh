#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt"
SOURCE_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"
BATCH="${SPINCORE_SAME_INPUT_BATCH:-2048}"
THREADS="${SPINCORE_SAME_INPUT_THREADS:-8}"

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
RUN_DIR="$ROOT/runs/lt2_same_input_target_variance/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/same_input_target_variance.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export SPINCORE_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf '=== SpinCore LT2 same-input target-variance audit ===\n'
printf 'mode=READ ONLY; NO ROOTS; NO OPTIMIZER STEPS\n'
printf 'grouping=exact SPNNIV1 observation + exact legal mask; thresholds=2,4,8\n'
printf 'batch=%s threads=%s\n\n' "$BATCH" "$THREADS"

"$PYTHON_RUN" tools/audit_lt2_same_input_target_variance.py \
  --solver build/libspincore_solver_c.so \
  --checkpoint "$SOURCE" \
  --batch-size "$BATCH" \
  --threads "$THREADS" \
  --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
assert d.get('schema') == 'SPINCORE_LT2_SAME_INPUT_TARGET_VARIANCE_V1'
assert d.get('completed_iteration') == 7500
assert set(d.get('domains') or {}) == {'THREE_HANDED','TRUE_HEADS_UP'}
for domain, block in d['domains'].items():
    assert block['memory_items'] > 0
    assert set(block['streets']) == {'PREFLOP','FLOP','TURN','RIVER'}
    for street in block['streets'].values():
        assert street['street_item_count'] >= 0
        for threshold, result in street.get('thresholds', {}).items():
            assert int(threshold) in (2,4,8)
            w=result['training_weighted_proxy']
            assert abs((w['within_same_input_target_mse'] + w['model_mse_to_same_input_mean']) - w['sample_target_mse']) < 1e-8
print('LT2_SAME_INPUT_TARGET_VARIANCE_POSTVALIDATION_PASS')
PY

AFTER_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [ "$AFTER_SHA" != "$SOURCE_SHA" ]; then
  echo "ERROR: source checkpoint changed during same-input audit" >&2
  exit 6
fi

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_same_input_target_variance.json"
fi

printf '\nLT2_SAME_INPUT_TARGET_VARIANCE_PASS\n'
printf 'source_unchanged=true\n'
printf 'report=%s\n' "$REPORT"
printf 'Evidence copied to Windows Downloads when available.\n'
printf 'STOP HERE. Do not resume root training until this conditional-variance result is reviewed.\n'
