#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2_stage_b/20260917_004911/checkpoint.pt"
SOURCE_SHA="3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0"
ADDITIONAL_ITERATIONS=100
TARGET_ITERATION=7600
EXPECTED_TOTAL_ROOTS=4560000
WORKERS=31
THREADS=8
CHECKPOINT_EVERY=25

for f in "$PYTHON_RUN" "$SOURCE" "$ROOT/build/libspincore_solver_c.so"; do
  [ -e "$f" ] || { echo "ERROR: missing required input: $f" >&2; exit 3; }
done
[ "$(sha256sum "$SOURCE" | awk '{print $1}')" = "$SOURCE_SHA" ] || {
  echo "ERROR: Stage B source SHA mismatch" >&2; exit 4;
}

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source/index changes present; pilot requires code == HEAD." >&2
  git status --short --untracked-files=no >&2
  exit 5
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lt2_hu_b400_online_pilot/$STAMP"
mkdir -p "$RUN_DIR"
CHECKPOINT="$RUN_DIR/checkpoint.pt"
REPORT="$RUN_DIR/training_report.json"
LOG="$RUN_DIR/training.log"
MEMLOG="$RUN_DIR/memory.log"
PROVENANCE="$RUN_DIR/provenance.txt"
CHAIN="$RUN_DIR/hu_policy_chain_stage_b_vs_pilot.json"
SUMMARY="$RUN_DIR/pilot_summary.json"

cp --reflink=auto "$SOURCE" "$CHECKPOINT"
[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" = "$SOURCE_SHA" ] || {
  echo "ERROR: isolated source copy hash mismatch" >&2; exit 6;
}

cat > "$PROVENANCE" <<EOF
schema=SPINCORE_LT2_HU_B400_ONLINE_PILOT_V1
created_at=$(date --iso-8601=seconds)
git_commit=$(git rev-parse HEAD)
source_checkpoint=$SOURCE
source_sha256=$SOURCE_SHA
pilot_checkpoint=$CHECKPOINT
source_iteration=7500
target_iteration=$TARGET_ITERATION
additional_iterations=$ADDITIONAL_ITERATIONS
additional_roots=60000
three_handed_advantage_steps=100
true_heads_up_advantage_steps=400
workers=$WORKERS
torch_threads=$THREADS
batch_mode=vectorized
iteration_mode=concurrent_fit
checkpoint_every=$CHECKPOINT_EVERY
holdout_touched=false
EOF

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export SPINCORE_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

snapshot_memory() {
  local tag="$1"
  local now
  now="$(date --iso-8601=seconds)"
  awk -v ts="$now" -v tag="$tag" '
    /^MemTotal:/ {mt=$2}
    /^MemAvailable:/ {ma=$2}
    /^SwapTotal:/ {st=$2}
    /^SwapFree:/ {sf=$2}
    END {
      printf "%s tag=%s mem_total_kib=%s mem_available_kib=%s swap_total_kib=%s swap_free_kib=%s swap_used_kib=%s\n", ts, tag, mt, ma, st, sf, st-sf
    }
  ' /proc/meminfo >> "$MEMLOG"
}

monitor_memory() {
  local watched_pid="$1"
  while kill -0 "$watched_pid" 2>/dev/null; do
    snapshot_memory periodic
    sleep 60
  done
}

printf '=== SpinCore LT2 HU B400 bounded online pilot ===\n'
printf 'source=Stage B iteration 7500 / 4.5M roots (READ ONLY)\n'
printf 'pilot=iterations 7501..7600 / +60k roots\n'
printf '3H advantage_steps=100; HU advantage_steps=400; K4=off\n'
printf 'postgate=full forensic HU policy-chain Stage B vs pilot\n\n'

snapshot_memory before
set +e
(
  /usr/bin/time -v "$PYTHON_RUN" tools/run_lean_functional_training.py     --solver build/libspincore_solver_c.so     --resume     --additional-iterations "$ADDITIONAL_ITERATIONS"     --hu-advantage-steps 400     --workers "$WORKERS"     --checkpoint-every "$CHECKPOINT_EVERY"     --checkpoint "$CHECKPOINT"     --report "$REPORT"     --batch-mode vectorized     --iteration-mode concurrent_fit     2>&1 | tee -a "$LOG"
  exit "${PIPESTATUS[0]}"
) &
TRAIN_PID=$!
monitor_memory "$TRAIN_PID" &
MONITOR_PID=$!
wait "$TRAIN_PID"
STATUS=$?
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
set -e
snapshot_memory after

[ "$(sha256sum "$SOURCE" | awk '{print $1}')" = "$SOURCE_SHA" ] || {
  echo "ERROR: preserved Stage B source changed" >&2; exit 7;
}
if [ "$STATUS" -ne 0 ]; then
  echo "LT2_HU_B400_ONLINE_PILOT_TRAINING_FAIL status=$STATUS" >&2
  echo "run_dir=$RUN_DIR" >&2
  exit "$STATUS"
fi

"$PYTHON_RUN" - "$REPORT" "$TARGET_ITERATION" "$EXPECTED_TOTAL_ROOTS" <<'PY'
import json, sys
from pathlib import Path

d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
target=int(sys.argv[2])
expected_roots=int(sys.argv[3])
assert int(d["config"]["iterations"]) == target
assert int(d["config"]["advantage_steps"]) == 100
assert int(d["config"]["hu_advantage_steps"]) == 400
assert int(d["config"]["hu_preflop_board_average_k"]) == 1
assert d["batch_mode"] == "vectorized"
assert d["iteration_mode"] == "concurrent_fit"
assert int(d["execution_workers"]) == 31
final=d["final"]["domains"]
assert sum(int(x["roots"]) for x in final.values()) == expected_roots
recent=d["history"][-100:]
assert len(recent) == 100
for row in recent:
    assert int(row["domains"]["THREE_HANDED"]["advantage_steps"]) == 100
    assert int(row["domains"]["TRUE_HEADS_UP"]["advantage_steps"]) == 400
print("LT2_HU_B400_ONLINE_PILOT_TRAINING_POSTVALIDATION_PASS")
PY

printf '\nTraining passed. Running full forensic HU policy-chain comparison...\n'
export SPINCORE_TORCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$PYTHON_RUN" tools/audit_lt2_hu_policy_chain.py   --solver build/libspincore_solver_c.so   --stage-a "$SOURCE"   --stage-b "$CHECKPOINT"   --scenarios-per-seed 5000   --workers "$WORKERS"   --report "$CHAIN"

"$PYTHON_RUN" - "$REPORT" "$CHAIN" "$CHECKPOINT" "$SUMMARY" <<'PY'
import hashlib, json, statistics, sys
from pathlib import Path

report_path, chain_path, ckpt_path, summary_path = map(Path, sys.argv[1:])
r=json.loads(report_path.read_text(encoding="utf-8"))
c=json.loads(chain_path.read_text(encoding="utf-8"))

recent=r["history"][-100:]
timing={}
for domain in ("THREE_HANDED","TRUE_HEADS_UP"):
    timing[domain]={
        "mean_tree_seconds": statistics.fmean(
            float(x["domains"][domain]["tree_seconds"]) for x in recent
        ),
        "mean_advantage_fit_seconds": statistics.fmean(
            float(x["domains"][domain]["advantage_fit_seconds"]) for x in recent
        ),
        "mean_advantage_samples_added": statistics.fmean(
            float(x["domains"][domain]["advantage_samples"]) for x in recent
        ),
        "mean_strategy_samples_added": statistics.fmean(
            float(x["domains"][domain]["strategy_samples"]) for x in recent
        ),
    }

h=hashlib.sha256()
with ckpt_path.open("rb") as f:
    for chunk in iter(lambda:f.read(8*1024*1024), b""):
        h.update(chunk)

out={
    "schema":"SPINCORE_LT2_HU_B400_ONLINE_PILOT_SUMMARY_V1",
    "source_iteration":7500,
    "pilot_iteration":7600,
    "additional_iterations":100,
    "additional_roots":60000,
    "config":{
        "three_handed_advantage_steps":100,
        "true_heads_up_advantage_steps":400,
        "hu_preflop_board_average_k":1,
    },
    "pilot_checkpoint":str(ckpt_path.resolve()),
    "pilot_checkpoint_sha256":h.hexdigest(),
    "timing_recent_100":timing,
    "final":r["final"],
    "hu_policy_chain_summary":c["summary"],
    "holdout_touched":False,
}
summary_path.write_text(json.dumps(out, indent=2, sort_keys=True)+"\n", encoding="utf-8")
print("LT2_HU_B400_ONLINE_PILOT_SUMMARY_PASS")
PY

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$SUMMARY" "$DEST/SpinCore_LT2_HU_B400_online_pilot_summary.json"
  cp "$CHAIN" "$DEST/SpinCore_LT2_HU_B400_online_pilot_chain.json"
  cp "$PROVENANCE" "$DEST/SpinCore_LT2_HU_B400_online_pilot_provenance.txt"
  cp "$MEMLOG" "$DEST/SpinCore_LT2_HU_B400_online_pilot_memory.log"
fi

printf '\nLT2_HU_B400_ONLINE_PILOT_PASS\n'
printf 'source_unchanged=true\n'
printf 'holdout_untouched=true\n'
printf 'checkpoint=%s\n' "$CHECKPOINT"
printf 'summary=%s\n' "$SUMMARY"
printf 'chain=%s\n' "$CHAIN"
printf 'STOP HERE. Do not extend the pilot yet. Send SpinCore_LT2_HU_B400_online_pilot_summary.json to ChatGPT.\n'
