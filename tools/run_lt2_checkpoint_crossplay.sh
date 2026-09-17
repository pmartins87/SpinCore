#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
A_SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt"
B_SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt"
A_SHA="e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c"
B_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"
SCENARIOS="${SPINCORE_CROSSPLAY_SCENARIOS:-3000}"
WORKERS="${SPINCORE_CROSSPLAY_WORKERS:-31}"
SEED="${SPINCORE_CROSSPLAY_SEED:-20260918}"

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
RUN_DIR="$ROOT/runs/lt2_checkpoint_crossplay/$STAMP"
mkdir -p "$RUN_DIR"
A_POLICY="$RUN_DIR/lt2a_policy.pt"
B_POLICY="$RUN_DIR/lt2b_policy.pt"
REPORT="$RUN_DIR/crossplay.json"
ROWS="$RUN_DIR/crossplay_rows.json"

export PYTHONPATH="$ROOT/python"
export SPINCORE_TORCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf '=== SpinCore LT2 checkpoint cross-play ===\n'
printf 'comparison=LT2A_1.8M -> LT2B_4.5M\n'
printf 'scenarios=%s workers=%s seed=%s\n' "$SCENARIOS" "$WORKERS" "$SEED"
printf 'mode=read-only contemporary-policy cross-play; NO TRAINING\n\n'

"$PYTHON_RUN" tools/export_lean_inference_checkpoint.py \
  --source "$A_SOURCE" \
  --out "$A_POLICY"

"$PYTHON_RUN" tools/export_lean_inference_checkpoint.py \
  --source "$B_SOURCE" \
  --out "$B_POLICY"

"$PYTHON_RUN" tools/evaluate_lt2_checkpoint_crossplay.py \
  --solver build/libspincore_solver_c.so \
  --before "$A_POLICY" \
  --after "$B_POLICY" \
  --scenarios "$SCENARIOS" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --report "$REPORT" \
  --rows-report "$ROWS"

"$PYTHON_RUN" - "$REPORT" "$SCENARIOS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
scenarios = int(sys.argv[2])
data = json.loads(path.read_text(encoding="utf-8"))
assert data.get("schema") == "SPINCORE_LT2_CHECKPOINT_CROSSPLAY_V1"
assert int(data.get("scenarios", -1)) == scenarios
assert set((data.get("mixture_hero") or {})) == {"ALL", "THREE_HANDED", "TRUE_HEADS_UP"}
for domain in ("ALL", "THREE_HANDED", "TRUE_HEADS_UP"):
    block = data["mixture_hero"][domain]["paired_delta_after_minus_before"]
    assert int(block.get("scenario_clusters", 0)) > 0
assert int((data.get("hu_direct_b_vs_a") or {}).get("scenario_clusters", 0)) > 0
assert int((data.get("three_handed_invasion") or {}).get("difference_b_vs_aa_minus_a_vs_bb", {}).get("scenario_clusters", 0)) > 0
print("CROSSPLAY_POSTVALIDATION_PASS")
PY

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
    cp "$REPORT" "$DEST/SpinCore_LT2A_to_LT2B_crossplay.json"
    cp "$ROWS" "$DEST/SpinCore_LT2A_to_LT2B_crossplay_rows.json"
fi

printf '\nLT2_CHECKPOINT_CROSSPLAY_PASS\n'
printf 'report=%s\n' "$REPORT"
printf 'rows=%s\n' "$ROWS"
printf 'Evidence copied to Windows Downloads when available.\n'
printf 'STOP HERE. Do not resume training until cross-play is interpreted together with policy drift and weak-baseline evidence.\n'
