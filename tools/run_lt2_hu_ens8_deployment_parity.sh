#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"
P8100="/home/rz9/spincore_lean_functional/runs/lt2_hu_ens8_online_pilot/20260920_154905/checkpoint.pt"
E8100="/home/rz9/spincore_lean_functional/runs/lt2_hu_ens8_online_pilot/20260920_154905/hu_ensemble_state.pt"
P8100_SHA="a51dbbed71090e45f2c4f5db6297f72eab848990b38650702b436e2aa60ca4bf"
E8100_SHA="c44b817f75304db352eedc33febbd180e6d038e0580c91be0b9befdbdb08f181"

for f in "$PY" "$SOLVER" "$P8100" "$E8100"; do
  [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 3; }
done
[ "$(sha256sum "$P8100"|awk '{print $1}')" = "$P8100_SHA" ] || exit 4
[ "$(sha256sum "$E8100"|awk '{print $1}')" = "$E8100_SHA" ] || exit 5

export PYTHONPATH="$ROOT/python:$ROOT/tools"
"$PY" -m py_compile   python/spincore/lean_hybrid_deployment_agent.py   tools/export_lt2_hu_ens8_hybrid_deployment.py   tools/audit_lt2_hu_ens8_deployment_parity.py
echo "PYTHON_PREFLIGHT_PASS"

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="$ROOT/runs/lt2_hu_ens8_deployment_parity/$STAMP"
mkdir -p "$DIR"
BUNDLE="$DIR/SpinCore_LT2_hybrid_deployment_8100.pt"
REPORT="$DIR/deployment_parity.json"

"$PY" tools/export_lt2_hu_ens8_hybrid_deployment.py   --checkpoint "$P8100"   --ensemble "$E8100"   --out "$BUNDLE"

"$PY" tools/audit_lt2_hu_ens8_deployment_parity.py   --solver "$SOLVER"   --checkpoint "$P8100"   --ensemble "$E8100"   --bundle "$BUNDLE"   --scenarios-per-seed 1000   --tolerance 1e-6   --report "$REPORT"

"$PY" - "$REPORT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT2_HYBRID_DEPLOYMENT_PARITY_V1"
assert d["verdict"]=="PASS"
assert d["method"]["training_roots"]==0
assert d["method"]["optimizer_steps"]==0
assert d["method"]["strategic_ev_evaluations"]==0
assert d["method"]["holdout_reused"] is False
assert d["legal_context_mismatches"]==0
assert d["argmax_mismatches"]==0
assert d["exact_resolution_mismatches"]==0
assert d["max_abs_probability_diff"]<=d["method"]["tolerance"]
print("LT2_HYBRID_DEPLOYMENT_PARITY_POSTVALIDATION_PASS")
PY

[ "$(sha256sum "$P8100"|awk '{print $1}')" = "$P8100_SHA" ] || exit 6
[ "$(sha256sum "$E8100"|awk '{print $1}')" = "$E8100_SHA" ] || exit 7

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$BUNDLE" "$DEST/SpinCore_LT2_hybrid_deployment_8100.pt"
  cp "$REPORT" "$DEST/SpinCore_LT2_hybrid_deployment_parity.json"
fi

echo "LT2_HYBRID_DEPLOYMENT_PARITY_PASS"
echo "STOP HERE. Send SpinCore_LT2_hybrid_deployment_parity.json to ChatGPT."
