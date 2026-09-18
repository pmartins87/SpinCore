#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"
STAGE_A_SHA="e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c"
STAGE_B_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"
SCENARIOS="${SPINCORE_FORENSIC_SCENARIOS_PER_SEED:-5000}"
WORKERS="${SPINCORE_FORENSIC_WORKERS:-31}"

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

printf "Locating preserved Stage A/B checkpoints by SHA...\n"
STAGE_A="$(find_checkpoint_by_sha "$STAGE_A_SHA")" || {
  echo "ERROR: preserved Stage A checkpoint not found under $ROOT/runs" >&2
  exit 5
}
STAGE_B="/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt"
if [ ! -f "$STAGE_B" ] || [ "$(sha256sum "$STAGE_B" | awk '{print $1}')" != "$STAGE_B_SHA" ]; then
  STAGE_B="$(find_checkpoint_by_sha "$STAGE_B_SHA")" || {
    echo "ERROR: preserved Stage B checkpoint not found under $ROOT/runs" >&2
    exit 6
  }
fi

printf "Stage A: %s\n" "$STAGE_A"
printf "Stage B: %s\n" "$STAGE_B"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lt2_stage_a_b_first_divergence/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/stage_a_b_first_divergence.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SPINCORE_TORCH_THREADS=1

printf "\n=== SpinCore LT2 Stage-A -> Stage-B first-divergence forensic ===\n"
printf "mode=READ ONLY; NO TRAINING ROOTS; NO OPTIMIZER STEPS\n"
printf "domain=TRUE_HEADS_UP; baselines=UNIFORM_LEGAL,PASSIVE_CALLER,JAMMER\n"
printf "forensic seeds=20260920..20260925 (already seen diagnostic family)\n"
printf "scenarios/seed=%s workers=%s\n\n" "$SCENARIOS" "$WORKERS"

"$PYTHON_RUN" tools/audit_lt2_stage_a_b_first_divergence.py \
  --solver "$SOLVER" \
  --stage-a "$STAGE_A" \
  --stage-b "$STAGE_B" \
  --seeds 20260920 20260921 20260922 20260923 20260924 20260925 \
  --scenarios-per-seed "$SCENARIOS" \
  --workers "$WORKERS" \
  --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json, math, sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads(p.read_text())
assert d.get("schema") == "SPINCORE_LT2_STAGE_A_B_FIRST_DIVERGENCE_FORENSIC_V1"
assert d["stage_a"]["completed_iteration"] == 3000
assert d["stage_b"]["completed_iteration"] == 7500
m=d["method"]
assert m["read_only"] is True
assert m["new_training_roots"] == 0
assert m["optimizer_steps"] == 0
assert m["domain"] == "TRUE_HEADS_UP"
assert m["seeds"] == [20260920,20260921,20260922,20260923,20260924,20260925]
assert m["opponent_families"] == ["UNIFORM_LEGAL","PASSIVE_CALLER","JAMMER"]
assert d["scenario_counts"]["hu_tasks"] > 0
assert d["scenario_counts"]["seat_runs"] > 0
for baseline in ("UNIFORM_LEGAL","PASSIVE_CALLER","JAMMER"):
    b=d["summary"][baseline]
    assert float(b["contribution_closure_abs_error"]) < 1e-9
    stat=b["overall"]["b_minus_a_chip_ev"]
    for k in ("mean","ci95_low","ci95_high"):
        assert math.isfinite(float(stat[k]))
print("LT2_STAGE_A_B_FIRST_DIVERGENCE_FORENSIC_POSTVALIDATION_PASS")
PY

AFTER_A="$(sha256sum "$STAGE_A" | awk '{print $1}')"
AFTER_B="$(sha256sum "$STAGE_B" | awk '{print $1}')"
if [ "$AFTER_A" != "$STAGE_A_SHA" ] || [ "$AFTER_B" != "$STAGE_B_SHA" ]; then
  echo "ERROR: preserved checkpoint changed during forensic audit" >&2
  exit 7
fi

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_stage_a_b_first_divergence.json"
fi

printf "\nLT2_STAGE_A_B_FIRST_DIVERGENCE_FORENSIC_PASS\n"
printf "sources_unchanged=true\n"
printf "report=%s\n" "$REPORT"
printf "Evidence copied to Windows Downloads when available.\n"
printf "STOP HERE. Do not train K4. The causal attribution result must be reviewed first.\n"
