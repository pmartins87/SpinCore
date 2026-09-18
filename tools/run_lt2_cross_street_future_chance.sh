#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"
STAGE_A_SHA="e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c"
STAGE_B_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"
THREADS="${SPINCORE_CROSS_STREET_THREADS:-8}"
STATES="${SPINCORE_CROSS_STREET_STATES_PER_SEED_STATUS:-2}"

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

STAGE_A="$(find_checkpoint_by_sha "$STAGE_A_SHA")" || { echo "ERROR: Stage A checkpoint not found" >&2; exit 5; }
STAGE_B="$(find_checkpoint_by_sha "$STAGE_B_SHA")" || { echo "ERROR: Stage B checkpoint not found" >&2; exit 6; }

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lt2_cross_street_future_chance/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/cross_street_future_chance.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf "=== SpinCore LT2 cross-street future-chance audit ===\n"
printf "mode=READ ONLY; NO TRAINING ROOTS; NO OPTIMIZER STEPS\n"
printf "forensic seeds=20260920..20260925 only; holdout 20261001..20261006 untouched\n"
printf "groups=Jammer preflop fold-vs-continue; PassiveCaller flop; UniformLegal turn\n"
printf "failure + matched-context controls; states/seed/status=%s; total expected=%s\n" "$STATES" "$((3 * 2 * 6 * STATES))"
printf "reference=8 future boards x 4 repeats exact1; candidate=8 future boards exact0, K1 vs K4\n"
printf "hidden opponent hand fixed to actual dealt hand\n\n"

"$PYTHON_RUN" tools/audit_lt2_cross_street_future_chance.py \
  --solver "$SOLVER" \
  --stage-a "$STAGE_A" \
  --stage-b "$STAGE_B" \
  --scenarios-per-seed 5000 \
  --states-per-seed-status "$STATES" \
  --reference-boards 8 \
  --reference-repeats 4 \
  --candidate-boards 8 \
  --threads "$THREADS" \
  --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json, math, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"] == "SPINCORE_LT2_CROSS_STREET_FUTURE_CHANCE_AUDIT_V1"
assert d["stage_a"]["completed_iteration"] == 3000
assert d["stage_b"]["completed_iteration"] == 7500
m=d["method"]
assert m["read_only"] is True
assert m["training_memory_writes"] == 0
assert m["optimizer_steps"] == 0
assert m["new_training_roots"] == 0
assert m["future_holdout_seeds_touched"] is False
assert m["forensic_seeds"] == [20260920,20260921,20260922,20260923,20260924,20260925]
for group in m["groups"]:
    for status in ("FAILURE","CONTROL"):
        s=d["summaries"][group][status]
        assert s["n"] == int(m["states_per_seed_status"]) * 6
        assert float(s["variance"]["sample_target_mse"]["mean"]) >= 0.0
        assert math.isfinite(float(s["k4_minus_k1"]["target_mse_to_reference"]["mean"]))
print("LT2_CROSS_STREET_FUTURE_CHANCE_AUDIT_POSTVALIDATION_PASS")
PY

AFTER_A="$(sha256sum "$STAGE_A" | awk '{print $1}')"
AFTER_B="$(sha256sum "$STAGE_B" | awk '{print $1}')"
if [ "$AFTER_A" != "$STAGE_A_SHA" ] || [ "$AFTER_B" != "$STAGE_B_SHA" ]; then
  echo "ERROR: source checkpoint changed during cross-street audit" >&2
  exit 7
fi

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_cross_street_future_chance.json"
fi

printf "\nLT2_CROSS_STREET_FUTURE_CHANCE_AUDIT_PASS\n"
printf "sources_unchanged=true\n"
printf "holdout_seeds_untouched=true\n"
printf "report=%s\n" "$REPORT"
printf "STOP HERE. Do not start K4 or any new training until this audit is reviewed.\n"
