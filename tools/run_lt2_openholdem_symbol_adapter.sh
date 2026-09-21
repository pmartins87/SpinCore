#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
"$PY" -m py_compile   python/spincore/openholdem_symbol_adapter.py   tools/audit_lt2_openholdem_symbol_adapter.py
echo "PYTHON_PREFLIGHT_PASS"

[ -e "$SOLVER" ] || {
  cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$ROOT/build" --target spincore_solver_c -j 31
}
[ -e "$SOLVER" ] || { echo "ERROR: missing solver $SOLVER" >&2; exit 3; }

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="$ROOT/runs/lt2_openholdem_symbol_adapter/$STAMP"
mkdir -p "$DIR"
REPORT="$DIR/openholdem_symbol_adapter.json"

echo "=== SpinCore LT2 OpenHoldem symbol/scrape adapter gate ==="
echo "old forensic seeds only; NO MODEL/EV/TRAINING; HOLDOUT NOT REUSED"

"$PY" tools/audit_lt2_openholdem_symbol_adapter.py   --solver "$SOLVER"   --scenarios-per-seed 650   --max-frames 9000   --fault-trials 1200   --report "$REPORT"

"$PY" - "$REPORT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT2_OPENHOLDEM_SYMBOL_ADAPTER_V1"
assert d["verdict"]=="PASS"
m=d["method"]
assert m["training_roots"]==0
assert m["optimizer_steps"]==0
assert m["strategic_ev_evaluations"]==0
assert m["deployment_model_inference"]==0
assert m["holdout_reused"] is False
assert d["failure_count"]==0
assert d["frames_checked"]>0
assert d["states_by_domain"]["THREE_HANDED"]>0
assert d["states_by_domain"]["TRUE_HEADS_UP"]>0
assert all(d["states_by_street"][str(i)]>0 for i in range(4))
assert d["fault_attempts"]==d["fault_rejections"]
assert d["distinct_physical_chair_layouts"]>20
print("LT2_OPENHOLDEM_SYMBOL_ADAPTER_POSTVALIDATION_PASS")
PY

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_openholdem_symbol_adapter.json"
fi

echo "LT2_OPENHOLDEM_SYMBOL_ADAPTER_PASS"
echo "STOP HERE. Send SpinCore_LT2_openholdem_symbol_adapter.json to ChatGPT."
