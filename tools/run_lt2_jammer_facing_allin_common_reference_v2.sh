#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"
STAGE_A_SHA="e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c"
STAGE_B_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"
THREADS="${SPINCORE_COMMON_REF_THREADS:-8}"
ANCHORS_PER_SEED="${SPINCORE_COMMON_REF_ANCHORS_PER_SEED:-4}"

if [ ! -x "$PYTHON_RUN" ]; then
  echo "ERROR: lean Python environment not found: $PYTHON_RUN" >&2
  exit 3
fi
if [ ! -f "$SOLVER" ]; then
  echo "ERROR: solver not found: $SOLVER" >&2
  exit 4
fi

find_checkpoint_by_sha() {
  local wanted="$1"
  local candidate actual
  while IFS= read -r -d "" candidate; do
    actual="$(sha256sum "$candidate" | awk '{print $1}')"
    if [ "$actual" = "$wanted" ]; then
      printf "%s" "$candidate"
      return 0
    fi
  done < <(find "$ROOT/runs" -type f -name checkpoint.pt -print0)
  return 1
}

STAGE_A="$(find_checkpoint_by_sha "$STAGE_A_SHA")" || {
  echo "ERROR: Stage A checkpoint not found" >&2
  exit 5
}
STAGE_B="$(find_checkpoint_by_sha "$STAGE_B_SHA")" || {
  echo "ERROR: Stage B checkpoint not found" >&2
  exit 6
}

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lt2_jammer_facing_allin_common_reference_v2/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/jammer_facing_allin_common_reference_v2.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf "=== SpinCore LT2 Jammer facing-all-in COMMON REFERENCE V2 ===\n"
printf "mode=READ ONLY; NO TRAINING ROOTS; NO OPTIMIZER STEPS\n"
printf "V1 stage-specific self-play posterior is NOT used as A-vs-B benchmark reference\n"
printf "common reference=uniform opponent hands under fixed hand-independent JAMMER + uniform future boards\n"
printf "forensic seeds=20260920..20260925 only; holdout 20261001..20261006 untouched\n"
printf "anchors/seed=%s total=%s threads=%s\n" "$ANCHORS_PER_SEED" "$((ANCHORS_PER_SEED * 6))" "$THREADS"
printf "reference=32 hands x 8 boards; candidate=16 hands x K1/K4\n\n"

"$PYTHON_RUN" tools/audit_lt2_jammer_facing_allin_common_reference_v2.py \
  --solver "$SOLVER" \
  --stage-a "$STAGE_A" \
  --stage-b "$STAGE_B" \
  --scenarios-per-seed 5000 \
  --anchors-per-seed "$ANCHORS_PER_SEED" \
  --reference-hands 32 \
  --reference-boards-per-hand 8 \
  --candidate-hands 16 \
  --candidate-boards-per-hand 4 \
  --threads "$THREADS" \
  --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json, math, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"] == "SPINCORE_LT2_JAMMER_FACING_ALLIN_COMMON_REFERENCE_V2"
assert d["stage_a"]["completed_iteration"] == 3000
assert d["stage_b"]["completed_iteration"] == 7500
m=d["method"]
assert m["read_only"] is True
assert m["training_memory_writes"] == 0
assert m["optimizer_steps"] == 0
assert m["new_training_roots"] == 0
assert m["future_holdout_seeds_touched"] is False
assert m["forensic_seeds"] == [20260920,20260921,20260922,20260923,20260924,20260925]
s=d["summary"]
assert s["n"] == int(m["anchors_per_seed"]) * 6
assert float(s["max_stage_a_b_fixed_deal_target_abs_delta"]) <= 1e-7
for k in ("average_policy_tv","average_policy_regret_chips","advantage_policy_tv","advantage_policy_regret_chips"):
    assert math.isfinite(float(s["stage_b_minus_a"][k]["mean"]))
for k in ("target_mse_to_reference","policy_tv_to_reference","regret_chips"):
    assert math.isfinite(float(s["k4_minus_k1"][k]["mean"]))
print("LT2_JAMMER_FACING_ALLIN_COMMON_REFERENCE_V2_POSTVALIDATION_PASS")
PY

AFTER_A="$(sha256sum "$STAGE_A" | awk '{print $1}')"
AFTER_B="$(sha256sum "$STAGE_B" | awk '{print $1}')"
if [ "$AFTER_A" != "$STAGE_A_SHA" ] || [ "$AFTER_B" != "$STAGE_B_SHA" ]; then
  echo "ERROR: source checkpoint changed during V2 overlay" >&2
  exit 7
fi

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_jammer_facing_allin_common_reference_v2.json"
fi

printf "\nLT2_JAMMER_FACING_ALLIN_COMMON_REFERENCE_V2_PASS\n"
printf "sources_unchanged=true\n"
printf "holdout_seeds_untouched=true\n"
printf "report=%s\n" "$REPORT"
printf "STOP HERE. Do not train K4 until V2 is reviewed.\n"
