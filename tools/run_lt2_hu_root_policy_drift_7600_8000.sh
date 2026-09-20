#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"
A_SOURCE="/home/rz9/spincore_lean_functional/runs/lt2_hu_b400_online_pilot/20260919_214344/checkpoint.pt"
B_SOURCE="/home/rz9/spincore_lean_functional/runs/lt2_hu_b400_refresh_to_8000/20260920_101117/checkpoint.pt"
A_SHA="c34f19802d3ad5ad3a5d131b083ffcee0e0867667ae0d57324cf35d5fa246b80"
B_SHA="773b5d523c7fc5fcbfc3d10cb1f5be6429e50f4283259df9134963db8d274886"

for f in "$PYTHON_RUN" "$SOLVER" "$A_SOURCE" "$B_SOURCE"; do
  [ -e "$f" ] || { echo "ERROR: missing required input: $f" >&2; exit 3; }
done
[ "$(sha256sum "$A_SOURCE" | awk '{print $1}')" = "$A_SHA" ] || { echo "ERROR: 7600 SHA mismatch" >&2; exit 4; }
[ "$(sha256sum "$B_SOURCE" | awk '{print $1}')" = "$B_SHA" ] || { echo "ERROR: 8000 SHA mismatch" >&2; exit 5; }

"$PYTHON_RUN" -m py_compile tools/audit_lt2_hu_root_policy_drift.py
echo "PYTHON_PREFLIGHT_PASS"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lt2_hu_root_policy_drift_7600_8000/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/hu_root_policy_drift.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export SPINCORE_TORCH_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "=== SpinCore LT2 HU deterministic root-policy drift 7600 -> 8000 ==="
echo "READ ONLY; holdout untouched; no sampled-action selection"

"$PYTHON_RUN" tools/audit_lt2_hu_root_policy_drift.py   --solver "$SOLVER"   --stage-a "$A_SOURCE"   --stage-b "$B_SOURCE"   --scenarios-per-seed 5000   --threads 8   --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT2_HU_ROOT_POLICY_DRIFT_V1"
assert d["stage_a"]["completed_iteration"]==7600
assert d["stage_b"]["completed_iteration"]==8000
m=d["method"]
assert m["read_only"] and not m["future_holdout_seeds_touched"]
assert m["new_training_roots"]==m["optimizer_steps"]==m["training_memory_writes"]==0
assert d["summary"]["n"]==sum(m["hu_scenarios_by_seed"].values())
print("LT2_HU_ROOT_POLICY_DRIFT_POSTVALIDATION_PASS")
PY

[ "$(sha256sum "$A_SOURCE" | awk '{print $1}')" = "$A_SHA" ] || { echo "ERROR: 7600 source changed" >&2; exit 6; }
[ "$(sha256sum "$B_SOURCE" | awk '{print $1}')" = "$B_SHA" ] || { echo "ERROR: 8000 source changed" >&2; exit 7; }

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_hu_root_policy_drift_7600_8000.json"
fi

echo "LT2_HU_ROOT_POLICY_DRIFT_PASS"
echo "STOP HERE. Send SpinCore_LT2_hu_root_policy_drift_7600_8000.json to ChatGPT."
