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
[ "$(sha256sum "$A_SOURCE" | awk '{print $1}')" = "$A_SHA" ] || { echo "ERROR: Stage A SHA mismatch" >&2; exit 4; }
[ "$(sha256sum "$B_SOURCE" | awk '{print $1}')" = "$B_SHA" ] || { echo "ERROR: Stage B SHA mismatch" >&2; exit 5; }

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lt2_jammer_fai_fold_shift_infoset/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/jammer_fai_fold_shift_infoset.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SPINCORE_TORCH_THREADS=8

printf "=== SpinCore LT2 Jammer FAI fold-shift low-noise infoset audit ===\n"
printf "mode=READ ONLY; NO TRAINING ROOTS; NO OPTIMIZER STEPS\n"
printf "forensic seeds=20260920..20260925; holdout 20261001..20261006 untouched\n"
printf "selection=fold-mass shift sign only; NO sampled FAI action/outcome/Q selection\n"
printf "groups=B_MORE_FOLD and B_LESS_FOLD; 8 anchors/group/seed; total=96\n"
printf "reference=64 uniform compatible hands x 8 future boards = 512 deals/anchor\n\n"

"$PYTHON_RUN" tools/audit_lt2_jammer_fai_fold_shift_infoset.py \
  --solver "$SOLVER" \
  --stage-a "$A_SOURCE" \
  --stage-b "$B_SOURCE" \
  --scenarios-per-seed 5000 \
  --anchors-per-shift-per-seed 8 \
  --reference-hands 64 \
  --reference-boards-per-hand 8 \
  --threads 8 \
  --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json, math, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"] == "SPINCORE_LT2_JAMMER_FAI_FOLD_SHIFT_INFOSET_V1"
assert d["stage_a"]["completed_iteration"] == 3000
assert d["stage_b"]["completed_iteration"] == 7500
m=d["method"]
assert m["read_only"] is True
assert m["new_training_roots"] == 0
assert m["optimizer_steps"] == 0
assert m["training_memory_writes"] == 0
assert m["future_holdout_seeds_touched"] is False
assert d["summary"]["n"] == 96
for g in ("B_MORE_FOLD","B_LESS_FOLD"):
    s=d["summary"]["groups"][g]
    assert s["n"] == 48
    x=s["policy_value_b_minus_a_chips"]["seed_cluster_ci"]
    assert math.isfinite(float(x["mean"]))
assert float(d["max_stage_a_b_canonical_gap_delta"]) <= 1e-7
print("LT2_JAMMER_FAI_FOLD_SHIFT_INFOSET_POSTVALIDATION_PASS")
PY

[ "$(sha256sum "$A_SOURCE" | awk '{print $1}')" = "$A_SHA" ] || { echo "ERROR: Stage A source changed" >&2; exit 6; }
[ "$(sha256sum "$B_SOURCE" | awk '{print $1}')" = "$B_SHA" ] || { echo "ERROR: Stage B source changed" >&2; exit 7; }

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_jammer_fai_fold_shift_infoset.json"
fi

printf "\nLT2_JAMMER_FAI_FOLD_SHIFT_INFOSET_PASS\n"
printf "sources_unchanged=true\n"
printf "holdout_seeds_untouched=true\n"
printf "report=%s\n" "$REPORT"
printf "STOP HERE. Do not train until the fold-shift infoset audit is reviewed.\n"
