#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt"
SOURCE_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"
STATES="${SPINCORE_TARGET_VARIANCE_STATES_PER_STREET:-64}"
REPEATS="${SPINCORE_TARGET_VARIANCE_REPEATS:-8}"
THREADS="${SPINCORE_TARGET_VARIANCE_THREADS:-8}"
MAX_EPISODES="${SPINCORE_TARGET_VARIANCE_MAX_EPISODES:-20000}"

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
RUN_DIR="$ROOT/runs/lt2_repeated_target_variance/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/repeated_target_variance.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export SPINCORE_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf '=== SpinCore LT2 repeated-state target variance audit ===\n'
printf 'mode=DIAGNOSTIC ONLY; NO OPTIMIZER STEPS; NO TRAINING MEMORY WRITES\n'
printf 'states_per_domain_per_street=%s repeats=%s exact_levels=0,1 threads=%s\n' "$STATES" "$REPEATS" "$THREADS"
printf 'question=how much held target error is repeated-state external-sampling noise, and what does exact opponent level 1 buy?\n\n'

"$PYTHON_RUN" tools/audit_lt2_repeated_target_variance.py \
  --solver build/libspincore_solver_c.so \
  --checkpoint "$SOURCE" \
  --states-per-street "$STATES" \
  --repeats "$REPEATS" \
  --threads "$THREADS" \
  --max-episodes "$MAX_EPISODES" \
  --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
assert d.get('schema') == 'SPINCORE_LT2_REPEATED_TARGET_VARIANCE_V1'
assert d.get('completed_iteration') == 7500
assert set(d.get('summaries') or {}) == {'THREE_HANDED','TRUE_HEADS_UP'}
for domain in ('THREE_HANDED','TRUE_HEADS_UP'):
    for street in ('PREFLOP','FLOP','TURN','RIVER'):
        b=d['summaries'][domain][street]
        assert b['exact_level_0']['model_mse_to_sampled_targets']['n'] > 0
        assert b['exact_level_1']['model_mse_to_sampled_targets']['n'] > 0
        assert 'level1_vs_level0' in b
print('LT2_REPEATED_TARGET_VARIANCE_POSTVALIDATION_PASS')
PY

AFTER_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [ "$AFTER_SHA" != "$SOURCE_SHA" ]; then
  echo "ERROR: source checkpoint changed during repeated-target audit" >&2
  exit 6
fi

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_repeated_target_variance.json"
fi

printf '\nLT2_REPEATED_TARGET_VARIANCE_AUDIT_PASS\n'
printf 'source_unchanged=true\n'
printf 'report=%s\n' "$REPORT"
printf 'Evidence copied to Windows Downloads when available.\n'
printf 'STOP HERE. Do not resume root training until repeated-target variance is reviewed.\n'
