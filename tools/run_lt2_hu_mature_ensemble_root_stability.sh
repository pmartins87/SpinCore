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

"$PY" -m py_compile tools/audit_lt2_hu_mature_ensemble_root_stability.py
echo "PYTHON_PREFLIGHT_PASS"

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="$ROOT/runs/lt2_hu_mature_ensemble_root_stability/$STAMP"
mkdir -p "$DIR"
REPORT="$DIR/hu_mature_ensemble_root_stability.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export SPINCORE_TORCH_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "=== SpinCore LT2 mature HU Advantage ensemble stability ==="
echo "8 x fresh400 fits; disjoint ensemble sizes 1/2/4"
echo "READ ONLY source; NO CFR ROOTS; HOLDOUT UNTOUCHED"

"$PY" tools/audit_lt2_hu_mature_ensemble_root_stability.py   --solver "$SOLVER"   --checkpoint "$SOURCE"   --budget 400   --replicas 8   --scenarios-per-seed 5000   --threads 8   --report "$REPORT"

"$PY" - "$REPORT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT2_HU_MATURE_ENSEMBLE_ROOT_STABILITY_V1"
assert d["checkpoint"]["iteration"]==8000
m=d["method"]
assert m["read_only_source_checkpoint"] is True
assert m["new_training_roots"]==m["source_training_memory_writes"]==0
assert m["future_holdout_seeds_touched"] is False
assert m["replicas"]==8 and m["budget_per_replica"]==400
assert m["root_count"]==sum(m["hu_scenarios_by_seed"].values())
assert len(d["results"]["size_1"]["pairs"])==4
assert len(d["results"]["size_2"]["pairs"])==2
assert len(d["results"]["size_4"]["pairs"])==1
print("LT2_HU_MATURE_ENSEMBLE_ROOT_STABILITY_POSTVALIDATION_PASS")
PY

[ "$(sha256sum "$SOURCE"|awk '{print $1}')" = "$SOURCE_SHA" ] || {
  echo "ERROR: source changed" >&2; exit 5;
}

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_hu_mature_ensemble_root_stability.json"
fi

echo "LT2_HU_MATURE_ENSEMBLE_ROOT_STABILITY_PASS"
echo "STOP HERE. Send SpinCore_LT2_hu_mature_ensemble_root_stability.json to ChatGPT."
