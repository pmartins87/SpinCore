#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"
A="/home/rz9/spincore_lean_functional/runs/lt2_hu_b400_online_pilot/20260919_214344/checkpoint.pt"
B="/home/rz9/spincore_lean_functional/runs/lt2_hu_b400_refresh_to_8000/20260920_101117/checkpoint.pt"
ASHA="c34f19802d3ad5ad3a5d131b083ffcee0e0867667ae0d57324cf35d5fa246b80"
BSHA="773b5d523c7fc5fcbfc3d10cb1f5be6429e50f4283259df9134963db8d274886"
for f in "$PY" "$SOLVER" "$A" "$B"; do [ -e "$f" ] || { echo "missing $f" >&2; exit 3; }; done
[ "$(sha256sum "$A"|awk '{print $1}')" = "$ASHA" ] || exit 4
[ "$(sha256sum "$B"|awk '{print $1}')" = "$BSHA" ] || exit 5
"$PY" -m py_compile tools/audit_lt2_hu_paired_fresh400_root_refit.py
STAMP="$(date +%Y%m%d_%H%M%S)"; DIR="$ROOT/runs/lt2_hu_paired_fresh400_root_refit/$STAMP"; mkdir -p "$DIR"
REPORT="$DIR/paired_fresh400_root_refit.json"
export PYTHONPATH="$ROOT/python:$ROOT/tools"; export SPINCORE_TORCH_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "=== Paired fresh 400-step root refit: reservoir 7600 vs 8000 ==="
"$PY" tools/audit_lt2_hu_paired_fresh400_root_refit.py --solver "$SOLVER" --stage-a "$A" --stage-b "$B" --replicates 3 --budget 400 --scenarios-per-seed 5000 --threads 8 --report "$REPORT"
"$PY" - "$REPORT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT2_HU_PAIRED_FRESH400_ROOT_REFIT_V1"
assert d["stage_a"]["iteration"]==7600 and d["stage_b"]["iteration"]==8000
m=d["method"]; assert m["read_only_source_checkpoints"] and not m["future_holdout_seeds_touched"]
assert m["new_training_roots"]==m["source_training_memory_writes"]==0
assert len(d["trials"])==3 and m["budget"]==400
print("LT2_HU_PAIRED_FRESH400_ROOT_REFIT_POSTVALIDATION_PASS")
PY
[ "$(sha256sum "$A"|awk '{print $1}')" = "$ASHA" ] || exit 6
[ "$(sha256sum "$B"|awk '{print $1}')" = "$BSHA" ] || exit 7
DEST="/mnt/c/Users/Rz9/Downloads"; [ ! -d "$DEST" ] || cp "$REPORT" "$DEST/SpinCore_LT2_hu_paired_fresh400_root_refit.json"
echo "LT2_HU_PAIRED_FRESH400_ROOT_REFIT_PASS"
echo "STOP HERE. Send SpinCore_LT2_hu_paired_fresh400_root_refit.json to ChatGPT."
