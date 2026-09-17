#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt"
SOURCE_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"
HELDOUT="${SPINCORE_ADV_SWEEP_HELDOUT:-25000}"
THREADS="${SPINCORE_ADV_SWEEP_THREADS:-8}"
REPLICATES="${SPINCORE_ADV_SWEEP_REPLICATES:-3}"
BATCH=1024
EVAL_BATCH="${SPINCORE_ADV_SWEEP_EVAL_BATCH:-2048}"

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
RUN_DIR="$ROOT/runs/lt2_stage_b_advantage_budget_sweep_v2/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/advantage_budget_sweep_v2.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export SPINCORE_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf '=== SpinCore LT2 Stage-B Advantage budget sweep V2 ===\n'
printf 'mode=NO ROOTS; source checkpoint READ ONLY\n'
printf 'metric semantics=production Lean regret matching + masked-softmax fallback\n'
printf 'heldout=%s batch=%s eval_batch=%s threads=%s replicates=%s\n' "$HELDOUT" "$BATCH" "$EVAL_BATCH" "$THREADS" "$REPLICATES"
printf 'budgets=0,25,50,100,200,400,800,1600 (100=canonical)\n\n'

"$PYTHON_RUN" tools/sweep_lt2_stage_b_advantage_budget_v2.py \
  --solver build/libspincore_solver_c.so \
  --checkpoint "$SOURCE" \
  --heldout "$HELDOUT" \
  --batch-size "$BATCH" \
  --eval-batch-size "$EVAL_BATCH" \
  --threads "$THREADS" \
  --replicates "$REPLICATES" \
  --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
assert d.get('schema') == 'SPINCORE_LT2_STAGE_B_ADVANTAGE_BUDGET_SWEEP_V2'
assert d.get('completed_iteration') == 7500
assert d['method']['policy_semantics'].startswith('production Lean')
assert set(d.get('domains') or {}) == {'THREE_HANDED','TRUE_HEADS_UP'}
for domain in d['domains'].values():
    assert domain['heldout_excluded_from_optimization'] is True
    assert len(domain['replicates']) == d['method']['replicates']
    assert [x['steps'] for x in domain['aggregate']] == [0,25,50,100,200,400,800,1600]
print('LT2_STAGE_B_ADVANTAGE_BUDGET_SWEEP_V2_POSTVALIDATION_PASS')
PY

AFTER_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [ "$AFTER_SHA" != "$SOURCE_SHA" ]; then
  echo "ERROR: source checkpoint changed during V2 sweep" >&2
  exit 6
fi

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_stage_b_advantage_budget_sweep_v2.json"
fi

printf '\nLT2_STAGE_B_ADVANTAGE_BUDGET_SWEEP_V2_PASS\n'
printf 'source_unchanged=true\n'
printf 'report=%s\n' "$REPORT"
printf 'Evidence copied to Windows Downloads when available.\n'
printf 'STOP HERE. Do not resume root training until this corrected multi-seed curve is reviewed.\n'
