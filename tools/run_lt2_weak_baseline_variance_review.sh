#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
A_SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt"
B_SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt"
A_SHA="e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c"
B_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"
WORKERS="${SPINCORE_WEAK_BASELINE_WORKERS:-31}"
SCENARIOS_PER_SEED="${SPINCORE_WEAK_BASELINE_SCENARIOS_PER_SEED:-5000}"
SEEDS=(20260920 20260921 20260922 20260923 20260924 20260925)

if [ ! -x "$PYTHON_RUN" ]; then
    echo "ERROR: lean Python environment not found: $PYTHON_RUN" >&2
    exit 3
fi
for f in "$A_SOURCE" "$B_SOURCE" "$ROOT/build/libspincore_solver_c.so"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing required input: $f" >&2
        exit 4
    fi
done

ACTUAL_A_SHA="$(sha256sum "$A_SOURCE" | awk '{print $1}')"
ACTUAL_B_SHA="$(sha256sum "$B_SOURCE" | awk '{print $1}')"
if [ "$ACTUAL_A_SHA" != "$A_SHA" ]; then
    echo "ERROR: Stage A checkpoint SHA mismatch." >&2
    exit 5
fi
if [ "$ACTUAL_B_SHA" != "$B_SHA" ]; then
    echo "ERROR: Stage B checkpoint SHA mismatch." >&2
    exit 6
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lt2_weak_baseline_variance_review/$STAMP"
mkdir -p "$RUN_DIR"
A_POLICY="$RUN_DIR/lt2a_policy.pt"
B_POLICY="$RUN_DIR/lt2b_policy.pt"
FINAL_REPORT="$RUN_DIR/weak_baseline_multiseed.json"

export PYTHONPATH="$ROOT/python"
export SPINCORE_TORCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf '=== SpinCore LT2 weak-baseline variance review ===\n'
printf 'comparison=LT2A_1.8M and LT2B_4.5M vs weak baselines\n'
printf 'seeds=%s scenarios_per_seed=%s total_per_checkpoint=%s workers=%s\n' \
  "${SEEDS[*]}" "$SCENARIOS_PER_SEED" "$((SCENARIOS_PER_SEED * ${#SEEDS[@]}))" "$WORKERS"
printf 'design=6 independent seed blocks; paired A/B within each block; NO TRAINING\n'
printf 'precision_rationale=30k total targets <= about 5 chips/hand simultaneous half-width in the worst pilot domain if variance remains similar\n\n'

"$PYTHON_RUN" tools/export_lean_inference_checkpoint.py --source "$A_SOURCE" --out "$A_POLICY"
"$PYTHON_RUN" tools/export_lean_inference_checkpoint.py --source "$B_SOURCE" --out "$B_POLICY"

for SEED in "${SEEDS[@]}"; do
    echo
    echo "--- seed=$SEED Stage A ---"
    "$PYTHON_RUN" tools/evaluate_lean_strategy_quality_with_rows.py \
      --solver build/libspincore_solver_c.so \
      --checkpoint "$A_POLICY" \
      --scenarios "$SCENARIOS_PER_SEED" \
      --workers "$WORKERS" \
      --seed "$SEED" \
      --report "$RUN_DIR/seed_${SEED}_a_report.json" \
      --rows-report "$RUN_DIR/seed_${SEED}_a_rows.json"

    echo
    echo "--- seed=$SEED Stage B ---"
    "$PYTHON_RUN" tools/evaluate_lean_strategy_quality_with_rows.py \
      --solver build/libspincore_solver_c.so \
      --checkpoint "$B_POLICY" \
      --scenarios "$SCENARIOS_PER_SEED" \
      --workers "$WORKERS" \
      --seed "$SEED" \
      --report "$RUN_DIR/seed_${SEED}_b_report.json" \
      --rows-report "$RUN_DIR/seed_${SEED}_b_rows.json"
done

"$PYTHON_RUN" tools/analyze_lt2_weak_baseline_multiseed.py \
  --run-dir "$RUN_DIR" \
  --report "$FINAL_REPORT"

"$PYTHON_RUN" - "$FINAL_REPORT" "$((SCENARIOS_PER_SEED * ${#SEEDS[@]}))" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
expected = int(sys.argv[2])
d = json.loads(p.read_text(encoding="utf-8"))
assert d.get("schema") == "SPINCORE_LT2_WEAK_BASELINE_MULTISEED_V1"
assert int(d.get("total_scenarios_per_checkpoint", -1)) == expected
assert len(d.get("seeds") or []) == 6
assert set((d.get("primary") or {})) == {"UNIFORM_LEGAL", "PASSIVE_CALLER", "JAMMER"}
print("WEAK_BASELINE_MULTISEED_POSTVALIDATION_PASS")
PY

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
    cp "$FINAL_REPORT" "$DEST/SpinCore_LT2_weak_baseline_multiseed_30k.json"
fi

printf '\nLT2_WEAK_BASELINE_VARIANCE_REVIEW_PASS\n'
printf 'report=%s\n' "$FINAL_REPORT"
printf 'Evidence copied to Windows Downloads when available.\n'
printf 'STOP HERE. Do not resume training until this weak-baseline variance result is reviewed.\n'
