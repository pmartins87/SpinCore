#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"
SCENARIOS="${SPINCORE_LT3_DEV_SCENARIOS:-3000}"
QUALITY_SCENARIOS="${SPINCORE_LT3_DEV_QUALITY_SCENARIOS:-2000}"
WORKERS="${SPINCORE_LT3_DEV_WORKERS:-31}"
SEED="${SPINCORE_LT3_DEV_SEED:-20260922}"

P8100="/home/rz9/spincore_lean_functional/runs/lt2_hu_ens8_online_pilot/20260920_154905/checkpoint.pt"
E8100="/home/rz9/spincore_lean_functional/runs/lt2_hu_ens8_online_pilot/20260920_154905/hu_ensemble_state.pt"

RUN9105="/home/rz9/spincore_lean_functional/runs/lt3_parallel_8200_9105/20260921_131118"
P8600_RAW="$RUN9105/milestones/checkpoint_8600.pt"
E8600="$RUN9105/milestones/hu_ensemble_state_8600.pt"
P9105="$RUN9105/checkpoint.pt"
E9105="$RUN9105/hu_ensemble_state.pt"

SHA8600="11cee67aaa626224f59a89ee2b18a1a521e2679f0152144728729186ee580cd6"
SHAE8600="4717fcb5ea3c132a19687cf0848c95694c920e7ac4e42300d740c1bdf3c73816"
SHA9105="21945e27c43c7e6c1cdb77018cd66dc90b3fab72c29a9034bb4a9f97cc0e6c68"
SHAE9105="b9c3ffffc7139eeb77c4b4182136e10ada5cad2f023aa6e560e164e2e0ac9256"

for f in "$PY" "$SOLVER" "$P8100" "$E8100" "$P8600_RAW" "$E8600" "$P9105" "$E9105"; do
  [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 3; }
done

check_hash() {
  local path="$1" expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [ "$actual" = "$expected" ] || {
    echo "ERROR: hash mismatch $path" >&2
    echo "expected=$expected" >&2
    echo "actual=$actual" >&2
    exit 4
  }
}
check_hash "$P8600_RAW" "$SHA8600"
check_hash "$E8600" "$SHAE8600"
check_hash "$P9105" "$SHA9105"
check_hash "$E9105" "$SHAE9105"

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export SPINCORE_TORCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$PY" - "$P8100" "$P8600_RAW" "$P9105" <<'PY'
import sys,torch
for path,iteration,want_finalized in (
    (sys.argv[1],8100,True),
    (sys.argv[2],8600,False),
    (sys.argv[3],9105,True),
):
    d=torch.load(path,map_location="cpu",weights_only=False)
    got=int(d.get("completed_iteration",-1))
    fin=bool(d.get("finalized"))
    if got!=iteration or fin!=want_finalized:
        raise SystemExit(f"checkpoint contract mismatch: {path} iteration={got} finalized={fin}")
print("LT3_POST9105_CHECKPOINT_PREFLIGHT_PASS")
PY

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="$ROOT/runs/lt3_post9105_dev_battery/$STAMP"
mkdir -p "$DIR"
P8600="$DIR/checkpoint_8600_finalized.pt"
P8100_INF="$DIR/policy_8100_inference.pt"
P8600_INF="$DIR/policy_8600_inference.pt"
P9105_INF="$DIR/policy_9105_inference.pt"

echo "=== SpinCore LT3 post-9105 development battery ==="
echo "mode=READ_ONLY DEVELOPMENT EVALUATION; NO TRAINING; NO SEALED HOLDOUT"
echo "scenarios=$SCENARIOS quality_scenarios=$QUALITY_SCENARIOS workers=$WORKERS seed=$SEED"
echo "run_dir=$DIR"

"$PY" tools/finalize_lt3_raw_milestone.py   --solver "$SOLVER"   --source-checkpoint "$P8600_RAW"   --expected-source-sha256 "$SHA8600"   --expected-iteration 8600   --output-checkpoint "$P8600"   --report "$DIR/finalize_8600.json"   --threads 8

# Full training checkpoints contain multi-GB reservoirs/optimizer state. Never
# fan those objects out into evaluation workers. Export compact AveragePolicy-only
# artifacts once, then let all multiprocessing evaluators load only those.
echo "--- Export compact inference checkpoints ---"
"$PY" tools/export_lean_inference_checkpoint.py --source "$P8100" --out "$P8100_INF"
"$PY" tools/export_lean_inference_checkpoint.py --source "$P8600" --out "$P8600_INF"
"$PY" tools/export_lean_inference_checkpoint.py --source "$P9105" --out "$P9105_INF"

"$PY" - "$P8100_INF" "$P8600_INF" "$P9105_INF" <<'PY'
import sys, torch
for path, iteration in zip(sys.argv[1:], (8100, 8600, 9105)):
    d=torch.load(path,map_location="cpu",weights_only=False)
    if d.get("schema")!="SPINCORE_LEAN_FUNCTIONAL_TRAINING_V1":
        raise SystemExit(f"compact schema mismatch: {path}")
    if not bool(d.get("finalized")):
        raise SystemExit(f"compact artifact not finalized: {path}")
    if int(d.get("completed_iteration",-1))!=iteration:
        raise SystemExit(f"compact iteration mismatch: {path}")
    domains=d.get("domains") or {}
    if set(domains)!={"THREE_HANDED","TRUE_HEADS_UP"}:
        raise SystemExit(f"compact domain mismatch: {path}")
    for domain,payload in domains.items():
        if set(payload)!={"policy"}:
            raise SystemExit(f"compact artifact contains non-policy state for {domain}: {path}")
print("LT3_POST9105_COMPACT_INFERENCE_PREFLIGHT_PASS")
PY

# AveragePolicy pairwise cross-play. Same sampler seed and scenario count for all three pairs.
for spec in   "8100 8600 $P8100_INF $P8600_INF"   "8600 9105 $P8600_INF $P9105_INF"   "8100 9105 $P8100_INF $P9105_INF"
do
  set -- $spec
  A="$1"; B="$2"; PA="$3"; PB="$4"
  echo "--- AveragePolicy crossplay $A -> $B ---"
  "$PY" tools/evaluate_lt2_checkpoint_crossplay.py     --solver "$SOLVER"     --before "$PA"     --after "$PB"     --scenarios "$SCENARIOS"     --workers "$WORKERS"     --seed "$SEED"     --report "$DIR/crossplay_${A}_${B}.json"     --rows-report "$DIR/crossplay_${A}_${B}_rows.json"
done

# HU current ENS8 behavior pairwise, also on shared development seed.
for spec in   "8100 8600 $E8100 $E8600"   "8600 9105 $E8600 $E9105"   "8100 9105 $E8100 $E9105"
do
  set -- $spec
  A="$1"; B="$2"; EA="$3"; EB="$4"
  echo "--- HU current ENS8 $A -> $B ---"
  "$PY" tools/evaluate_lt3_hu_ens8_pairwise.py     --solver "$SOLVER"     --before-ensemble "$EA"     --after-ensemble "$EB"     --scenarios "$SCENARIOS"     --workers "$WORKERS"     --seed "$SEED"     --report "$DIR/ens8_${A}_${B}.json"
done

# AveragePolicy movement diagnostics. Use the same compact policy-only artifacts.
for spec in   "8100 8600 $P8100_INF $P8600_INF"   "8600 9105 $P8600_INF $P9105_INF"   "8100 9105 $P8100_INF $P9105_INF"
do
  set -- $spec
  A="$1"; B="$2"; PA="$3"; PB="$4"
  "$PY" tools/compare_lt2_policy_drift.py     --solver "$SOLVER"     --before "$PA"     --after "$PB"     --scenarios "$SCENARIOS"     --workers "$WORKERS"     --seed "$SEED"     --report "$DIR/drift_${A}_${B}.json"
done

# Transparent weak-baseline context for each finalized AveragePolicy.
for spec in "8100 $P8100_INF" "8600 $P8600_INF" "9105 $P9105_INF"; do
  set -- $spec
  I="$1"; P="$2"
  "$PY" tools/evaluate_lean_strategy_quality.py     --solver "$SOLVER"     --checkpoint "$P"     --scenarios "$QUALITY_SCENARIOS"     --workers "$WORKERS"     --seed "$SEED"     --report "$DIR/quality_${I}.json"
done

"$PY" tools/summarize_lt3_post9105_dev_battery.py   --dir "$DIR"   --out "$DIR/lt3_post9105_dev_battery.json"

# Recheck immutable sources after all evaluations.
check_hash "$P8600_RAW" "$SHA8600"
check_hash "$E8600" "$SHAE8600"
check_hash "$P9105" "$SHA9105"
check_hash "$E9105" "$SHAE9105"

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$DIR/lt3_post9105_dev_battery.json" "$DEST/SpinCore_LT3_post9105_dev_battery.json"
  cp "$DIR/finalize_8600.json" "$DEST/SpinCore_LT3_finalize_8600.json"
fi

echo "LT3_POST9105_DEV_BATTERY_COMPLETE"
echo "report=$DIR/lt3_post9105_dev_battery.json"
echo "STOP HERE. Send SpinCore_LT3_post9105_dev_battery.json to ChatGPT."
