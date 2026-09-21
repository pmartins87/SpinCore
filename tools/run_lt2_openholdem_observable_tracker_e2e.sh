#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
"$PY" -m py_compile   python/spincore/openholdem_symbol_adapter.py   python/spincore/runtime_observable_tracker.py   tools/audit_lt2_openholdem_observable_tracker_e2e.py
echo "PYTHON_PREFLIGHT_PASS"

[ -e "$SOLVER" ] || { echo "ERROR: missing solver $SOLVER" >&2; exit 3; }

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="$ROOT/runs/lt2_openholdem_observable_tracker_e2e/$STAMP"
mkdir -p "$DIR"
REPORT="$DIR/openholdem_observable_tracker_e2e.json"

echo "=== SpinCore LT2 OpenHoldem observable tracker E2E gate ==="
echo "raw OH frame -> adapter -> transcript -> rebuild -> canonical Hero parity"
echo "old forensic seeds only; NO MODEL/EV/TRAINING; HOLDOUT NOT REUSED"

"$PY" tools/audit_lt2_openholdem_observable_tracker_e2e.py   --solver "$SOLVER"   --scenarios-per-seed 700   --max-transitions 10000   --fault-hands 500   --report "$REPORT"

"$PY" - "$REPORT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT2_OPENHOLDEM_OBSERVABLE_TRACKER_E2E_V1"
assert d["verdict"]=="PASS"
m=d["method"]
assert m["training_roots"]==0
assert m["optimizer_steps"]==0
assert m["strategic_ev_evaluations"]==0
assert m["deployment_model_inference"]==0
assert m["holdout_reused"] is False
assert d["failure_count"]==0
assert d["transitions_checked"]>0
assert d["hero_canonical_state_checks"]>0
assert d["street_reveals_checked"]>0
assert d["exact_action_mismatches"]==0
assert d["canonical_state_mismatches"]==0
assert d["transcript_mismatches"]==0
assert d["hero_states_by_domain"]["THREE_HANDED"]>0
assert d["hero_states_by_domain"]["TRUE_HEADS_UP"]>0
assert d["hero_states_by_street"]["0"]>0
assert sum(d["hero_states_by_street"][str(i)] for i in (1,2,3))>0
f=d["faults"]
assert f["corrupt_attempts"]==f["corrupt_rejections"]
assert f["skipped_attempts"]>0
assert f["skipped_attempts"]==f["skipped_rejections"]
print("LT2_OPENHOLDEM_OBSERVABLE_TRACKER_E2E_POSTVALIDATION_PASS")
PY

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_openholdem_observable_tracker_e2e.json"
fi

echo "LT2_OPENHOLDEM_OBSERVABLE_TRACKER_E2E_PASS"
echo "STOP HERE. Send SpinCore_LT2_openholdem_observable_tracker_e2e.json to ChatGPT."
