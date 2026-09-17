#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
A_SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt"
B_SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt"
A_SHA="e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c"
B_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"
SAMPLES="${SPINCORE_FIT_AUDIT_SAMPLES:-25000}"
BATCH="${SPINCORE_FIT_AUDIT_BATCH:-2048}"
THREADS="${SPINCORE_FIT_AUDIT_THREADS:-8}"

if [ ! -x "$PYTHON_RUN" ]; then
  echo "ERROR: lean Python environment not found: $PYTHON_RUN" >&2
  exit 3
fi
for f in "$A_SOURCE" "$B_SOURCE" "$ROOT/build/libspincore_solver_c.so"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing required input: $f" >&2
    exit 4
  fi
done

ACTUAL_A_SHA="$(sha256sum "$A_SOURCE" | awk '{print $1}')"
ACTUAL_B_SHA="$(sha256sum "$B_SOURCE" | awk '{print $1}')"
[ "$ACTUAL_A_SHA" = "$A_SHA" ] || { echo "ERROR: Stage A SHA mismatch" >&2; exit 5; }
[ "$ACTUAL_B_SHA" = "$B_SHA" ] || { echo "ERROR: Stage B SHA mismatch" >&2; exit 6; }

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lt2_training_dynamics_fit_audit/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/fit_audit.json"

export PYTHONPATH="$ROOT/python"
export SPINCORE_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf '=== SpinCore LT2 training-dynamics fit audit V2 ===\n'
printf 'mode=READ ONLY; NO TRAINING; NO CHECKPOINT MUTATION\n'
printf 'advantage policy metric semantics=production lean RM + softmax all-nonpositive fallback\n'
printf 'samples_per_memory=%s batch=%s torch_threads=%s\n' "$SAMPLES" "$BATCH" "$THREADS"

"$PYTHON_RUN" tools/audit_lt2_checkpoint_fit.py \
  --solver build/libspincore_solver_c.so \
  --stage-a "$A_SOURCE" \
  --stage-b "$B_SOURCE" \
  --samples-per-memory "$SAMPLES" \
  --batch-size "$BATCH" \
  --threads "$THREADS" \
  --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
assert d.get('schema') == 'SPINCORE_LT2_TRAINING_FIT_AUDIT_V2'
assert 'softmax' in d['method']['advantage_policy_semantics']
for stage in ('stage_a','stage_b'):
    assert set(d[stage]['domains']) == {'THREE_HANDED','TRUE_HEADS_UP'}
    for domain in d[stage]['domains'].values():
        assert domain['advantage_fit']['sample_count'] > 0
        assert domain['policy_fit']['sample_count'] > 0
print('LT2_TRAINING_FIT_AUDIT_POSTVALIDATION_PASS')
PY

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_training_fit_audit_v2.json"
fi

printf '\nLT2_TRAINING_DYNAMICS_FIT_AUDIT_PASS\n'
printf 'report=%s\n' "$REPORT"
printf 'STOP HERE. Do not resume long training until the fit audit is reviewed.\n'
