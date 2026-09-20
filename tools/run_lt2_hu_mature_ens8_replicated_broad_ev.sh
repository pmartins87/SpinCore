#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"

P7500="/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt"
P7600="/home/rz9/spincore_lean_functional/runs/lt2_hu_b400_online_pilot/20260919_214344/checkpoint.pt"
P8000="/home/rz9/spincore_lean_functional/runs/lt2_hu_b400_refresh_to_8000/20260920_101117/checkpoint.pt"

S7500="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"
S7600="c34f19802d3ad5ad3a5d131b083ffcee0e0867667ae0d57324cf35d5fa246b80"
S8000="773b5d523c7fc5fcbfc3d10cb1f5be6429e50f4283259df9134963db8d274886"

for f in "$PY" "$SOLVER" "$P7500" "$P7600" "$P8000"; do
  [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 3; }
done
[ "$(sha256sum "$P7500"|awk '{print $1}')" = "$S7500" ] || exit 4
[ "$(sha256sum "$P7600"|awk '{print $1}')" = "$S7600" ] || exit 5
[ "$(sha256sum "$P8000"|awk '{print $1}')" = "$S8000" ] || exit 6

"$PY" -m py_compile tools/audit_lt2_hu_mature_ens8_replicated_broad_ev.py
echo "PYTHON_PREFLIGHT_PASS"

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="$ROOT/runs/lt2_hu_mature_ens8_replicated_broad_ev/$STAMP"
mkdir -p "$DIR"
REPORT="$DIR/hu_mature_ens8_replicated_broad_ev.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export SPINCORE_TORCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== SpinCore LT2 mature HU replicated size-8 broad EV ==="
echo "ENS8_A=replicas0..7; ENS8_B=replicas8..15"
echo "groups evaluated sequentially to bound memory"
echo "READ ONLY sources; NO CFR ROOTS; HOLDOUT UNTOUCHED"

"$PY" tools/audit_lt2_hu_mature_ens8_replicated_broad_ev.py   --solver "$SOLVER"   --stage-7500 "$P7500"   --stage-7600 "$P7600"   --stage-8000 "$P8000"   --budget 400   --scenarios-per-seed 5000   --workers 31   --threads-fit 8   --report "$REPORT"

"$PY" - "$REPORT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT2_HU_MATURE_ENS8_REPLICATED_BROAD_EV_V1"
m=d["method"]
assert m["read_only_source_checkpoints"] is True
assert m["new_training_roots"]==m["source_training_memory_writes"]==0
assert m["future_holdout_seeds_touched"] is False
assert len(d["candidate_meta"]["ENS8_A"])==8
assert len(d["candidate_meta"]["ENS8_B"])==8
assert set(d["ens8_b_minus_a"])=={"UNIFORM_LEGAL","PASSIVE_CALLER","JAMMER"}
for g in ("ENS8_A","ENS8_B"):
    assert set(d["summary"][g])=={"UNIFORM_LEGAL","PASSIVE_CALLER","JAMMER"}
print("LT2_HU_MATURE_ENS8_REPLICATED_BROAD_EV_POSTVALIDATION_PASS")
PY

[ "$(sha256sum "$P7500"|awk '{print $1}')" = "$S7500" ] || exit 7
[ "$(sha256sum "$P7600"|awk '{print $1}')" = "$S7600" ] || exit 8
[ "$(sha256sum "$P8000"|awk '{print $1}')" = "$S8000" ] || exit 9

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_hu_mature_ens8_replicated_broad_ev.json"
fi

echo "LT2_HU_MATURE_ENS8_REPLICATED_BROAD_EV_PASS"
echo "STOP HERE. Send SpinCore_LT2_hu_mature_ens8_replicated_broad_ev.json to ChatGPT."
