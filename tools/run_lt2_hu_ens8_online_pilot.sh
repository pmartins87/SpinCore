#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"
SOURCE="/home/rz9/spincore_lean_functional/runs/lt2_hu_b400_refresh_to_8000/20260920_101117/checkpoint.pt"
SOURCE_SHA="773b5d523c7fc5fcbfc3d10cb1f5be6429e50f4283259df9134963db8d274886"

for f in "$PY" "$SOLVER" "$SOURCE"; do
  [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 3; }
done
[ "$(sha256sum "$SOURCE"|awk '{print $1}')" = "$SOURCE_SHA" ] || {
  echo "ERROR: iteration-8000 SHA mismatch" >&2; exit 4;
}

"$PY" -m py_compile   python/spincore/lean_action_policy.py   python/spincore/lean_parallel.py   tools/run_lt2_hu_ens8_online_pilot.py
echo "PYTHON_PREFLIGHT_PASS"

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="$ROOT/runs/lt2_hu_ens8_online_pilot/$STAMP"
mkdir -p "$DIR"
REPORT="$DIR/hu_ens8_online_pilot.json"
CHECKPOINT="$DIR/checkpoint.pt"
ENSEMBLE="$DIR/hu_ensemble_state.pt"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export SPINCORE_TORCH_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "=== SpinCore LT2 HU ENS8 online-feedback pilot ==="
echo "source=8000 exact SHA; target=8100; +100 iterations"
echo "3H=fresh100 unchanged; HU=8 x fresh400 raw-output ensemble"
echo "ENS8 composition=predeclared group A, never outcome-selected"
echo "SOURCE READ ONLY; HOLDOUT SEALED"

"$PY" tools/run_lt2_hu_ens8_online_pilot.py   --solver "$SOLVER"   --source-checkpoint "$SOURCE"   --output-checkpoint "$CHECKPOINT"   --ensemble-state "$ENSEMBLE"   --report "$REPORT"   --additional-iterations 100   --checkpoint-every 25   --workers 31   --threads 8

"$PY" - "$REPORT" "$CHECKPOINT" "$ENSEMBLE" <<'PY'
import json,sys,torch
from pathlib import Path
report=Path(sys.argv[1]); checkpoint=Path(sys.argv[2]); ensemble=Path(sys.argv[3])
d=json.loads(report.read_text())
assert d["schema"]=="SPINCORE_LT2_HU_ENS8_ONLINE_PILOT_V1"
assert d["status"]=="PASS"
assert d["source_iteration"]==8000 and d["completed_iteration"]==8100
assert d["intervention"]["ensemble_size"]==8
assert d["intervention"]["member_steps"]==400
assert d["method"]["source_checkpoint_read_only"] is True
assert d["method"]["holdout_touched"] is False
assert checkpoint.is_file() and ensemble.is_file()
e=torch.load(ensemble,map_location="cpu",weights_only=False)
assert e["schema"]=="SPINCORE_LT2_HU_ENS8_CURRENT_STATE_V1"
assert e["completed_iteration"]==8100 and len(e["members"])==8
print("LT2_HU_ENS8_ONLINE_PILOT_POSTVALIDATION_PASS")
PY

[ "$(sha256sum "$SOURCE"|awk '{print $1}')" = "$SOURCE_SHA" ] || {
  echo "ERROR: source checkpoint changed" >&2; exit 5;
}

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_hu_ens8_online_pilot.json"
fi

echo "LT2_HU_ENS8_ONLINE_PILOT_PASS"
echo "STOP HERE. Send SpinCore_LT2_hu_ens8_online_pilot.json to ChatGPT."
