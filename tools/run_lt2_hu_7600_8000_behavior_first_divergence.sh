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
WORKERS="${SPINCORE_7600_8000_FIRST_DIV_WORKERS:-16}"
SCENARIOS="${SPINCORE_7600_8000_FIRST_DIV_SCENARIOS_PER_SEED:-5000}"

for f in "$PYTHON_RUN" "$SOLVER" "$A_SOURCE" "$B_SOURCE"; do
  [ -e "$f" ] || { echo "ERROR: missing required input: $f" >&2; exit 3; }
done
[ "$(sha256sum "$A_SOURCE" | awk '{print $1}')" = "$A_SHA" ] || { echo "ERROR: iteration-7600 SHA mismatch" >&2; exit 4; }
[ "$(sha256sum "$B_SOURCE" | awk '{print $1}')" = "$B_SHA" ] || { echo "ERROR: iteration-8000 SHA mismatch" >&2; exit 5; }

"$PYTHON_RUN" -m py_compile tools/audit_lt2_hu_behavior_first_divergence.py
echo "PYTHON_PREFLIGHT_PASS"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lt2_hu_7600_8000_behavior_first_divergence/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/hu_7600_8000_behavior_first_divergence.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SPINCORE_TORCH_THREADS=1

printf '=== SpinCore LT2 HU 7600 -> 8000 current-behavior first-divergence ===\n'
printf 'mode=READ ONLY; NO TRAINING ROOTS; NO OPTIMIZER STEPS\n'
printf 'primary=localize newly resolved PASSIVE_CALLER current-behavior regression\n'
printf 'forensic seeds=20260920..20260925; holdout untouched\n'
printf 'scenarios/seed=%s workers=%s\n\n' "$SCENARIOS" "$WORKERS"

"$PYTHON_RUN" tools/audit_lt2_hu_behavior_first_divergence.py   --solver "$SOLVER"   --stage-a "$A_SOURCE"   --stage-b "$B_SOURCE"   --scenarios-per-seed "$SCENARIOS"   --workers "$WORKERS"   --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json, math, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"] == "SPINCORE_LT2_HU_BEHAVIOR_FIRST_DIVERGENCE_V1"
assert d["stage_a"]["completed_iteration"] == 7600
assert d["stage_b"]["completed_iteration"] == 8000
m=d["method"]
assert m["read_only"] is True
assert m["new_training_roots"] == 0
assert m["optimizer_steps"] == 0
assert m["training_memory_writes"] == 0
assert m["future_holdout_seeds_touched"] is False
assert set(d["summary"]) == {"UNIFORM_LEGAL","PASSIVE_CALLER","JAMMER"}
for b,x in d["summary"].items():
    total=x["total_b_minus_a"]
    assert math.isfinite(float(total["mean"]))
    s=sum(float(x["groups"][g]["contribution_b_minus_a"]["mean"]) for g in m["groups"])
    assert abs(s-float(total["mean"])) < 1e-8, (b,s,total["mean"])
print("LT2_HU_7600_8000_BEHAVIOR_FIRST_DIVERGENCE_POSTVALIDATION_PASS")
PY

[ "$(sha256sum "$A_SOURCE" | awk '{print $1}')" = "$A_SHA" ] || { echo "ERROR: iteration-7600 source changed" >&2; exit 6; }
[ "$(sha256sum "$B_SOURCE" | awk '{print $1}')" = "$B_SHA" ] || { echo "ERROR: iteration-8000 source changed" >&2; exit 7; }

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_hu_7600_8000_behavior_first_divergence.json"
fi

printf '\nLT2_HU_7600_8000_BEHAVIOR_FIRST_DIVERGENCE_PASS\n'
printf 'sources_unchanged=true\n'
printf 'holdout_untouched=true\n'
printf 'report=%s\n' "$REPORT"
printf 'STOP HERE. Do not train beyond iteration 8000. Send SpinCore_LT2_hu_7600_8000_behavior_first_divergence.json to ChatGPT.\n'
