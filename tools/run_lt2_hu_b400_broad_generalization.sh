#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"
A_SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt"
B_SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt"
A_SHA="e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c"
B_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"

for f in "$PYTHON_RUN" "$SOLVER" "$A_SOURCE" "$B_SOURCE"; do
  [ -e "$f" ] || { echo "ERROR: missing required input: $f" >&2; exit 3; }
done
[ "$(sha256sum "$A_SOURCE" | awk '{print $1}')" = "$A_SHA" ] || {
  echo "ERROR: Stage A SHA mismatch" >&2; exit 4;
}
[ "$(sha256sum "$B_SOURCE" | awk '{print $1}')" = "$B_SHA" ] || {
  echo "ERROR: Stage B SHA mismatch" >&2; exit 5;
}

CONTROLLED_BASE="$ROOT/runs/lt2_jammer_fai_controlled_refit"
CONTROLLED="$(
  find "$CONTROLLED_BASE" -mindepth 2 -maxdepth 2 -type f     -name 'jammer_fai_controlled_refit.json'     -printf '%T@ %p\n' 2>/dev/null   | sort -nr | head -1 | cut -d' ' -f2-
)"
[ -n "$CONTROLLED" ] && [ -f "$CONTROLLED" ] || {
  echo "ERROR: controlled-refit report not found" >&2
  exit 6
}

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lt2_hu_b400_broad_generalization/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/hu_b400_broad_generalization.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SPINCORE_TORCH_THREADS=1

printf "=== SpinCore LT2 HU B400 broad generalization gate ===\n"
printf "source checkpoints READ ONLY; no roots; holdout untouched\n"
printf "candidates=all three deterministic 400-step Stage-B refits\n"
printf "population=forensic seeds 20260920..20260925, 5000 scenarios/seed\n"
printf "baselines=UNIFORM_LEGAL,PASSIVE_CALLER,JAMMER\n\n"

"$PYTHON_RUN" tools/audit_lt2_hu_b400_broad_generalization.py   --solver "$SOLVER"   --stage-a "$A_SOURCE"   --stage-b "$B_SOURCE"   --controlled-report "$CONTROLLED"   --scenarios-per-seed 5000   --workers 31   --threads-fit 8   --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json, math, sys
from pathlib import Path

d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"] == "SPINCORE_LT2_HU_B400_BROAD_GENERALIZATION_V1"
m=d["method"]
assert m["read_only_source_checkpoints"] is True
assert m["new_training_roots"] == 0
assert m["source_training_memory_writes"] == 0
assert m["candidate_optimizer_steps"] == 1200
assert m["future_holdout_seeds_touched"] is False
assert len(d["candidate_meta"]) == 3
assert d["rows_count"] > 0

# Reproduce the already-established Stage-B current-behavior Jammer regression.
x=d["summary"]["JAMMER"]["prod_b_minus_a"]
assert math.isfinite(float(x["mean"]))
assert abs(float(x["mean"]) - (-8.4304)) < 0.02

for baseline in ("UNIFORM_LEGAL","PASSIVE_CALLER","JAMMER"):
    for rep in range(3):
        y=d["summary"][baseline]["candidates"][f"B400_R{rep}"]["minus_prod_b"]
        assert math.isfinite(float(y["mean"]))

print("LT2_HU_B400_BROAD_GENERALIZATION_POSTVALIDATION_PASS")
PY

[ "$(sha256sum "$A_SOURCE" | awk '{print $1}')" = "$A_SHA" ] || {
  echo "ERROR: Stage A source changed" >&2; exit 7;
}
[ "$(sha256sum "$B_SOURCE" | awk '{print $1}')" = "$B_SHA" ] || {
  echo "ERROR: Stage B source changed" >&2; exit 8;
}

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_hu_b400_broad_generalization.json"
fi

printf "\nLT2_HU_B400_BROAD_GENERALIZATION_PASS\n"
printf "sources_unchanged=true\n"
printf "holdout_seeds_untouched=true\n"
printf "report=%s\n" "$REPORT"
printf "STOP HERE. Do not resume long training. Send SpinCore_LT2_hu_b400_broad_generalization.json to ChatGPT.\n"
