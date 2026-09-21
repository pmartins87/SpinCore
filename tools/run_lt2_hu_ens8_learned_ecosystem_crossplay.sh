#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"

P7600="/home/rz9/spincore_lean_functional/runs/lt2_hu_b400_online_pilot/20260919_214344/checkpoint.pt"
P8000="/home/rz9/spincore_lean_functional/runs/lt2_hu_b400_refresh_to_8000/20260920_101117/checkpoint.pt"
P8100="/home/rz9/spincore_lean_functional/runs/lt2_hu_ens8_online_pilot/20260920_154905/checkpoint.pt"
E8100="/home/rz9/spincore_lean_functional/runs/lt2_hu_ens8_online_pilot/20260920_154905/hu_ensemble_state.pt"

S7600="c34f19802d3ad5ad3a5d131b083ffcee0e0867667ae0d57324cf35d5fa246b80"
S8000="773b5d523c7fc5fcbfc3d10cb1f5be6429e50f4283259df9134963db8d274886"

for f in "$PY" "$SOLVER" "$P7600" "$P8000" "$P8100" "$E8100"; do
  [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 3; }
done
[ "$(sha256sum "$P7600"|awk '{print $1}')" = "$S7600" ] || exit 4
[ "$(sha256sum "$P8000"|awk '{print $1}')" = "$S8000" ] || exit 5

export PYTHONPATH="$ROOT/python:$ROOT/tools"
"$PY" -m py_compile tools/audit_lt2_hu_ens8_learned_ecosystem_crossplay.py
echo "PYTHON_PREFLIGHT_PASS"

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="$ROOT/runs/lt2_hu_ens8_learned_ecosystem_crossplay/$STAMP"
mkdir -p "$DIR"
REPORT="$DIR/hu_ens8_learned_ecosystem_crossplay.json"

P8100_SHA="$(sha256sum "$P8100"|awk '{print $1}')"
E8100_SHA="$(sha256sum "$E8100"|awk '{print $1}')"

export SPINCORE_TORCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== SpinCore LT2 HU ENS8 independent learned-ecosystem crossplay ==="
echo "design seeds=20260926..20260930; 6000 sampler scenarios/seed"
echo "historical ecosystem excludes candidate 8100"
echo "READ ONLY sources; NO CFR ROOTS; HOLDOUT UNTOUCHED"

"$PY" tools/audit_lt2_hu_ens8_learned_ecosystem_crossplay.py   --solver "$SOLVER"   --stage-7600 "$P7600"   --stage-8000 "$P8000"   --stage-8100 "$P8100"   --ensemble-8100 "$E8100"   --scenarios-per-seed 6000   --workers 31   --threads-fit 8   --report "$REPORT"

"$PY" - "$REPORT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT2_HU_ENS8_LEARNED_ECOSYSTEM_CROSSPLAY_V1"
m=d["method"]
assert m["read_only_source_artifacts"] is True
assert m["new_training_roots"]==m["source_training_memory_writes"]==0
assert m["future_holdout_seeds_touched"] is False
assert m["design_seeds"]==[20260926,20260927,20260928,20260929,20260930]
assert set(d["direct"])=={
    "ENS8_8100_vs_ENS8_8000",
    "AVG_8100_vs_AVG_8000",
    "ENS8_8100_vs_AVG_8100",
}
assert d["ecosystem"]["clusters"]==sum(m["hu_scenarios_by_seed"].values())
print("LT2_HU_ENS8_LEARNED_ECOSYSTEM_CROSSPLAY_POSTVALIDATION_PASS")
PY

[ "$(sha256sum "$P7600"|awk '{print $1}')" = "$S7600" ] || exit 6
[ "$(sha256sum "$P8000"|awk '{print $1}')" = "$S8000" ] || exit 7
[ "$(sha256sum "$P8100"|awk '{print $1}')" = "$P8100_SHA" ] || exit 8
[ "$(sha256sum "$E8100"|awk '{print $1}')" = "$E8100_SHA" ] || exit 9

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_hu_ens8_learned_ecosystem_crossplay.json"
fi

echo "LT2_HU_ENS8_LEARNED_ECOSYSTEM_CROSSPLAY_PASS"
echo "STOP HERE. Send SpinCore_LT2_hu_ens8_learned_ecosystem_crossplay.json to ChatGPT."
