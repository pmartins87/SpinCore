#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
"$PY" -m py_compile   python/spincore/solver.py   python/spincore/runtime_transition_reconciler.py   tools/audit_lt2_public_snapshot_reconciler.py
echo "PYTHON_PREFLIGHT_PASS"

cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build" --target spincore_solver_c -j 31
echo "SOLVER_BUILD_PASS"

SOLVER="$ROOT/build/libspincore_solver_c.so"
[ -e "$SOLVER" ] || { echo "ERROR: missing rebuilt solver $SOLVER" >&2; exit 3; }

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="$ROOT/runs/lt2_public_snapshot_reconciler/$STAMP"
mkdir -p "$DIR"
REPORT="$DIR/public_snapshot_reconciler.json"

echo "=== SpinCore LT2 public snapshot -> exact transcript reconciler ==="
echo "old forensic seeds only; NO MODEL/EV/TRAINING; HOLDOUT NOT REUSED"

"$PY" tools/audit_lt2_public_snapshot_reconciler.py   --solver "$SOLVER"   --scenarios-per-seed 900   --max-transitions 12000   --fault-trials 1500   --report "$REPORT"

"$PY" - "$REPORT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT2_PUBLIC_SNAPSHOT_RECONCILER_V1"
assert d["verdict"]=="PASS"
m=d["method"]
assert m["training_roots"]==0
assert m["optimizer_steps"]==0
assert m["strategic_ev_evaluations"]==0
assert m["holdout_reused"] is False
assert d["failure_count"]==0
assert d["alias_failure_count"]==0
assert d["states_by_domain"]["THREE_HANDED"]>0
assert d["states_by_domain"]["TRUE_HEADS_UP"]>0
assert d["states_by_street"]["0"]>0
assert sum(d["states_by_street"][str(i)] for i in (1,2,3))>0
assert d["actions"]["CALL"]>0
assert d["actions"]["ALL_IN"]>0
assert d["actions"]["BET_TO"]+d["actions"]["RAISE_TO"]>0
assert d["noop_snapshot_rejections"]==d["fault_attempts"]
assert d["corrupt_snapshot_rejections"]==d["fault_attempts"]
assert d["skipped_action_snapshot_rejections"]>0
print("LT2_PUBLIC_SNAPSHOT_RECONCILER_POSTVALIDATION_PASS")
PY

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_public_snapshot_reconciler.json"
fi

echo "LT2_PUBLIC_SNAPSHOT_RECONCILER_PASS"
echo "STOP HERE. Send SpinCore_LT2_public_snapshot_reconciler.json to ChatGPT."
