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
ANCHORS="${SPINCORE_FAI_CALIBRATION_ANCHORS_PER_SEED:-8}"
THREADS="${SPINCORE_FAI_CALIBRATION_THREADS:-8}"

for f in "$PYTHON_RUN" "$SOLVER" "$A_SOURCE" "$B_SOURCE"; do
  [ -e "$f" ] || { echo "ERROR: missing required input: $f" >&2; exit 3; }
done
[ "$(sha256sum "$A_SOURCE" | awk '{print $1}')" = "$A_SHA" ] || { echo "ERROR: Stage A SHA mismatch" >&2; exit 4; }
[ "$(sha256sum "$B_SOURCE" | awk '{print $1}')" = "$B_SHA" ] || { echo "ERROR: Stage B SHA mismatch" >&2; exit 5; }

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lt2_jammer_fai_broad_calibration/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/jammer_fai_broad_calibration.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SPINCORE_TORCH_THREADS="$THREADS"

printf "=== SpinCore LT2 Jammer FAI broad action-gap/RM calibration ===\n"
printf "mode=READ ONLY; NO TRAINING ROOTS; NO OPTIMIZER STEPS\n"
printf "forensic seeds=20260920..20260925; holdout 20261001..20261006 untouched\n"
printf "selection=all common FAI states before any earlier A/B behavior divergence; sample independent of FAI outcome\n"
printf "anchors/seed=%s total=%s\n" "$ANCHORS" "$((ANCHORS * 6))"
printf "reference=32 uniform opponent hands x 8 future boards; common action-gap gauge\n"
printf "metrics=raw-target MSE, action-gap MSE, support/sign errors, fallback, policy regret\n\n"

"$PYTHON_RUN" tools/audit_lt2_jammer_fai_broad_calibration.py \
  --solver "$SOLVER" \
  --stage-a "$A_SOURCE" \
  --stage-b "$B_SOURCE" \
  --scenarios-per-seed 5000 \
  --anchors-per-seed "$ANCHORS" \
  --reference-hands 32 \
  --reference-boards-per-hand 8 \
  --threads "$THREADS" \
  --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json, math, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"] == "SPINCORE_LT2_JAMMER_FAI_BROAD_CALIBRATION_V1"
assert d["stage_a"]["completed_iteration"] == 3000
assert d["stage_b"]["completed_iteration"] == 7500
m=d["method"]
assert m["read_only"] is True
assert m["new_training_roots"] == 0
assert m["optimizer_steps"] == 0
assert m["training_memory_writes"] == 0
assert m["future_holdout_seeds_touched"] is False
assert d["summary"]["n"] == int(m["anchors_per_seed"]) * 6
assert float(d["max_stage_a_b_canonical_gap_delta"]) <= 1e-7
for sk in ("stage_a","stage_b"):
    for f in ("policy_regret_chips","canonical_action_gap_mse","raw_target_mse"):
        assert math.isfinite(float(d["summary"][sk][f]["seed_cluster_ci"]["mean"]))
print("LT2_JAMMER_FAI_BROAD_CALIBRATION_POSTVALIDATION_PASS")
PY

[ "$(sha256sum "$A_SOURCE" | awk '{print $1}')" = "$A_SHA" ] || { echo "ERROR: Stage A source changed" >&2; exit 6; }
[ "$(sha256sum "$B_SOURCE" | awk '{print $1}')" = "$B_SHA" ] || { echo "ERROR: Stage B source changed" >&2; exit 7; }

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_jammer_fai_broad_calibration.json"
fi

printf "\nLT2_JAMMER_FAI_BROAD_CALIBRATION_PASS\n"
printf "sources_unchanged=true\n"
printf "holdout_seeds_untouched=true\n"
printf "report=%s\n" "$REPORT"
printf "STOP HERE. Do not train until broad FAI calibration is reviewed.\n"
