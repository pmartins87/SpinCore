#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"

SOURCE_DIR="/home/rz9/spincore_lean_functional/runs/lt2_hu_ens8_online_pilot/20260920_154905"
SOURCE_CHECKPOINT="$SOURCE_DIR/checkpoint.pt"
SOURCE_ENSEMBLE="$SOURCE_DIR/hu_ensemble_state.pt"
SOURCE_CHECKPOINT_SHA="a51dbbed71090e45f2c4f5db6297f72eab848990b38650702b436e2aa60ca4bf"
SOURCE_ENSEMBLE_SHA="c44b817f75304db352eedc33febbd180e6d038e0580c91be0b9befdbdb08f181"

ADDITIONAL_ITERATIONS=500
SOURCE_ITERATION=8100
TARGET_ITERATION=8600
ROOTS_PER_ITERATION=600
EXPECTED_NEW_ROOTS=300000
EXPECTED_TOTAL_ROOTS=5160000
WORKERS=31
THREADS=8
CHECKPOINT_EVERY=50

for f in "$PY" "$SOLVER" "$SOURCE_CHECKPOINT" "$SOURCE_ENSEMBLE"; do
  [ -e "$f" ] || { echo "ERROR: missing required input: $f" >&2; exit 3; }
done

[ "$(sha256sum "$SOURCE_CHECKPOINT"|awk '{print $1}')" = "$SOURCE_CHECKPOINT_SHA" ] || {
  echo "ERROR: frozen LT2@8100 checkpoint SHA mismatch" >&2; exit 4;
}
[ "$(sha256sum "$SOURCE_ENSEMBLE"|awk '{print $1}')" = "$SOURCE_ENSEMBLE_SHA" ] || {
  echo "ERROR: frozen LT2@8100 ENS8 sidecar SHA mismatch" >&2; exit 5;
}

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source/index changes present; LT3 H1 requires code == HEAD." >&2
  git status --short --untracked-files=no >&2
  exit 6
fi

MEMTOTAL_KIB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
if [ -z "$MEMTOTAL_KIB" ] || [ "$MEMTOTAL_KIB" -lt $((28*1024*1024)) ]; then
  echo "ERROR: WSL exposes less than 28 GiB RAM; refusing LT3 H1." >&2
  exit 7
fi
DISK_KIB="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
if [ -z "$DISK_KIB" ] || [ "$DISK_KIB" -lt $((15*1024*1024)) ]; then
  echo "ERROR: less than 15 GiB free on training filesystem." >&2
  exit 8
fi

export PYTHONPATH="$ROOT/python:$ROOT/tools"
"$PY" -m py_compile   python/spincore/lean_action_policy.py   python/spincore/lean_parallel.py   tools/run_lt2_hu_ens8_online_pilot.py   tools/run_lt3_heavy_ens8_h1.py
echo "LT3_H1_PYTHON_PREFLIGHT_PASS"

"$PY" python_tests/test_lean_action_policy_ensemble.py
echo "LT3_H1_ENSEMBLE_POLICY_UNIT_TEST_PASS"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lt3_heavy_ens8_h1/$STAMP"
mkdir -p "$RUN_DIR"

CHECKPOINT="$RUN_DIR/checkpoint.pt"
ENSEMBLE="$RUN_DIR/hu_ensemble_state.pt"
REPORT="$RUN_DIR/lt3_heavy_h1.json"
LOG="$RUN_DIR/training.log"
MEMLOG="$RUN_DIR/memory.log"
PROVENANCE="$RUN_DIR/provenance.txt"

cat > "$PROVENANCE" <<EOF
schema=SPINCORE_LT3_HEAVY_ENS8_H1_PROVENANCE_V1
created_at=$(date --iso-8601=seconds)
git_commit=$(git rev-parse HEAD)
research_lane=LT3_HEAVY_H1
production_status=RESEARCH_ONLY_NOT_PROMOTED
source_checkpoint=$SOURCE_CHECKPOINT
source_checkpoint_sha256=$SOURCE_CHECKPOINT_SHA
source_ensemble=$SOURCE_ENSEMBLE
source_ensemble_sha256=$SOURCE_ENSEMBLE_SHA
source_iteration=$SOURCE_ITERATION
target_iteration=$TARGET_ITERATION
additional_iterations=$ADDITIONAL_ITERATIONS
roots_per_iteration=$ROOTS_PER_ITERATION
new_training_roots=$EXPECTED_NEW_ROOTS
expected_total_roots=$EXPECTED_TOTAL_ROOTS
three_handed_advantage_steps=100
true_heads_up_ensemble_size=8
true_heads_up_member_steps=400
true_heads_up_optimizer_steps_per_iteration=3200
workers=$WORKERS
torch_threads=$THREADS
checkpoint_every=$CHECKPOINT_EVERY
hu_preflop_board_average_k=1
lt2_production_mutated=false
lt2_final_holdout_touched=false
lt3_holdout_touched=false
EOF

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
  local pid="$1"
  while kill -0 "$pid" 2>/dev/null; do
    snapshot_memory periodic
    sleep 60
  done
}

copy_evidence() {
  local dest="/mnt/c/Users/Rz9/Downloads"
  [ -d "$dest" ] || return 0
  [ -f "$REPORT" ] && cp "$REPORT" "$dest/SpinCore_LT3_heavy_ens8_H1.json" || true
  [ -f "$PROVENANCE" ] && cp "$PROVENANCE" "$dest/SpinCore_LT3_heavy_ens8_H1_provenance.txt" || true
  [ -f "$MEMLOG" ] && cp "$MEMLOG" "$dest/SpinCore_LT3_heavy_ens8_H1_memory.log" || true
}

echo "=== SpinCore LT3 Heavy ENS8 H1 ==="
echo "RESEARCH ONLY — LT2@8100 remains frozen production baseline"
echo "source=LT2 ENS8@8100 exact checkpoint+sidecar hashes"
echo "target=8600 / +500 iterations / +300,000 roots"
echo "3H=fresh100; HU=8 x fresh400; K4=off; workers=31; threads=8"
echo "LT2 final holdout untouched; LT3 sealed holdout untouched"
echo "run_dir=$RUN_DIR"
echo

snapshot_memory before
set +e
(
  /usr/bin/time -v "$PY" tools/run_lt3_heavy_ens8_h1.py     --solver "$SOLVER"     --source-checkpoint "$SOURCE_CHECKPOINT"     --source-ensemble-state "$SOURCE_ENSEMBLE"     --output-checkpoint "$CHECKPOINT"     --output-ensemble-state "$ENSEMBLE"     --report "$REPORT"     --additional-iterations "$ADDITIONAL_ITERATIONS"     --checkpoint-every "$CHECKPOINT_EVERY"     --workers "$WORKERS"     --threads "$THREADS"     2>&1 | tee -a "$LOG"
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

[ "$(sha256sum "$SOURCE_CHECKPOINT"|awk '{print $1}')" = "$SOURCE_CHECKPOINT_SHA" ] || {
  copy_evidence
  echo "ERROR: frozen LT2 source checkpoint changed" >&2; exit 9;
}
[ "$(sha256sum "$SOURCE_ENSEMBLE"|awk '{print $1}')" = "$SOURCE_ENSEMBLE_SHA" ] || {
  copy_evidence
  echo "ERROR: frozen LT2 source ensemble changed" >&2; exit 10;
}

if [ "$STATUS" -ne 0 ]; then
  copy_evidence
  echo "LT3_HEAVY_ENS8_H1_TRAINING_FAIL status=$STATUS" >&2
  echo "run_dir=$RUN_DIR" >&2
  echo "Do not restart automatically; send training.log / report to ChatGPT." >&2
  exit "$STATUS"
fi

"$PY" - "$REPORT" "$CHECKPOINT" "$ENSEMBLE" "$TARGET_ITERATION" "$EXPECTED_NEW_ROOTS" "$EXPECTED_TOTAL_ROOTS" <<'PY'
import hashlib,json,sys,torch
from pathlib import Path

report,checkpoint,ensemble=map(Path,sys.argv[1:4])
target=int(sys.argv[4]); new_roots=int(sys.argv[5]); expected_total=int(sys.argv[6])
d=json.loads(report.read_text(encoding="utf-8"))
assert d["schema"]=="SPINCORE_LT3_HEAVY_ENS8_H1_V1"
assert d["status"]=="PASS"
assert d["research_lane"]=="LT3_HEAVY_H1"
assert d["production_status"]=="RESEARCH_ONLY_NOT_PROMOTED"
assert d["source_iteration"]==8100
assert d["completed_iteration"]==target
assert d["method"]["new_training_roots"]==new_roots
assert d["method"]["lt2_production_artifacts_mutated"] is False
assert d["method"]["lt2_final_holdout_touched"] is False
assert d["method"]["lt3_holdout_touched"] is False
assert checkpoint.is_file() and ensemble.is_file()
final=d["final"]["domains"]
assert sum(int(x["roots"]) for x in final.values())==expected_total

e=torch.load(ensemble,map_location="cpu",weights_only=False)
assert e["schema"]=="SPINCORE_LT3_HU_ENS8_CURRENT_STATE_V1"
assert e["completed_iteration"]==target
assert e["ensemble_size"]==8 and e["member_steps"]==400
assert len(e["members"])==8

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

print("LT3_H1_POSTVALIDATION_PASS")
print("checkpoint_sha256="+sha(checkpoint))
print("ensemble_sha256="+sha(ensemble))
PY

copy_evidence

echo
echo "LT3_HEAVY_ENS8_H1_TRAINING_PASS"
echo "research_only=true"
echo "lt2_production_unchanged=true"
echo "lt2_final_holdout_untouched=true"
echo "lt3_holdout_untouched=true"
echo "run_dir=$RUN_DIR"
echo "report=$REPORT"
echo "checkpoint=$CHECKPOINT"
echo "ensemble_state=$ENSEMBLE"
echo "STOP HERE. Do not extend beyond 8600 before LT3 H1 development-set adjudication."
echo "Send SpinCore_LT3_heavy_ens8_H1.json to ChatGPT."
