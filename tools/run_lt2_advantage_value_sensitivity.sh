#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt"
SOURCE_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"
SAMPLES="${SPINCORE_VALUE_SENSITIVITY_SAMPLES:-100000}"
BATCH="${SPINCORE_VALUE_SENSITIVITY_BATCH:-2048}"
THREADS="${SPINCORE_VALUE_SENSITIVITY_THREADS:-8}"

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
RUN_DIR="$ROOT/runs/lt2_advantage_value_sensitivity/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/value_sensitivity.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export SPINCORE_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf '=== SpinCore LT2 Advantage value-sensitivity audit ===\n'
printf 'mode=READ ONLY; NO ROOTS; NO OPTIMIZER STEPS\n'
printf 'samples_per_domain=%s batch=%s threads=%s\n' "$SAMPLES" "$BATCH" "$THREADS"
printf 'metric=target-value regret in chip-equivalent units plus branch/street/span diagnostics\n\n'

"$PYTHON_RUN" tools/audit_lt2_advantage_value_sensitivity.py \
  --solver build/libspincore_solver_c.so \
  --checkpoint "$SOURCE" \
  --samples-per-domain "$SAMPLES" \
  --batch-size "$BATCH" \
  --threads "$THREADS" \
  --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
assert d.get('schema') == 'SPINCORE_LT2_ADVANTAGE_VALUE_SENSITIVITY_V1'
assert d.get('completed_iteration') == 7500
assert set(d.get('domains') or {}) == {'THREE_HANDED','TRUE_HEADS_UP'}
for block in d['domains'].values():
    assert block['sample_count'] > 0
    groups=block['weighted_groups']
    assert 'OVERALL' in groups
    assert 'TARGET_ALL_NONPOS' in groups
    assert 'TARGET_HAS_POSITIVE' in groups
    w=groups['OVERALL']['weighted']
    for key in ('tv','branch_mismatch','model_policy_regret_to_best_chips','positive_model_value_loss_vs_target_policy_chips'):
        assert key in w
print('LT2_ADVANTAGE_VALUE_SENSITIVITY_POSTVALIDATION_PASS')
PY

AFTER_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [ "$AFTER_SHA" != "$SOURCE_SHA" ]; then
  echo "ERROR: source checkpoint changed during value-sensitivity audit" >&2
  exit 6
fi

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_advantage_value_sensitivity.json"
fi

printf '\nLT2_ADVANTAGE_VALUE_SENSITIVITY_PASS\n'
printf 'source_unchanged=true\n'
printf 'report=%s\n' "$REPORT"
printf 'Evidence copied to Windows Downloads when available.\n'
printf 'STOP HERE. Do not resume root training until this value-sensitivity result is reviewed.\n'
