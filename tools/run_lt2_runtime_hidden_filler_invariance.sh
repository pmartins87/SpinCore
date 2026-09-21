#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"

[ -x "$PY" ] || { echo "ERROR: missing $PY" >&2; exit 3; }
[ -e "$SOLVER" ] || { echo "ERROR: missing $SOLVER" >&2; exit 4; }

export PYTHONPATH="$ROOT/python:$ROOT/tools"
"$PY" -m py_compile tools/audit_lt2_runtime_hidden_filler_invariance.py
echo "PYTHON_PREFLIGHT_PASS"

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="$ROOT/runs/lt2_runtime_hidden_filler_invariance/$STAMP"
mkdir -p "$DIR"
REPORT="$DIR/runtime_hidden_filler_invariance.json"

export SPINCORE_TORCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== SpinCore LT2 OpenHoldem reconstruction prerequisite ==="
echo "prove hidden-opponent/unseen-board filler invariance"
echo "old forensic seeds only; NO EV; NO TRAINING; HOLDOUT NOT REUSED"

"$PY" tools/audit_lt2_runtime_hidden_filler_invariance.py   --solver "$SOLVER"   --scenarios-per-seed 500   --fillers-per-state 6   --max-target-states 6000   --workers 31   --report "$REPORT"

"$PY" - "$REPORT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT2_RUNTIME_HIDDEN_FILLER_INVARIANCE_V1"
assert d["verdict"]=="PASS"
m=d["method"]
assert m["training_roots"]==0
assert m["optimizer_steps"]==0
assert m["strategic_ev_evaluations"]==0
assert m["holdout_reused"] is False
assert d["failure_count"]==0
assert d["states_by_domain"]["THREE_HANDED"]>0
assert d["states_by_domain"]["TRUE_HEADS_UP"]>0
assert d["states_by_street"]["0"]>0
assert sum(d["states_by_street"][str(x)] for x in (1,2,3))>0
print("LT2_RUNTIME_HIDDEN_FILLER_INVARIANCE_POSTVALIDATION_PASS")
PY

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_runtime_hidden_filler_invariance.json"
fi

echo "LT2_RUNTIME_HIDDEN_FILLER_INVARIANCE_PASS"
echo "STOP HERE. Send SpinCore_LT2_runtime_hidden_filler_invariance.json to ChatGPT."
