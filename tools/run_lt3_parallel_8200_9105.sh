#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(git -C "$ROOT" rev-parse --show-toplevel)"
cd "$ROOT"

PY="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"

SOURCE_DIR="$ROOT/runs/lt3_heavy_ens8_h1/20260921_041723"
SOURCE_CHECKPOINT="$SOURCE_DIR/checkpoint.pt"
SOURCE_ENSEMBLE="$SOURCE_DIR/hu_ensemble_state.pt"
SOURCE_CHECKPOINT_SHA="36d8008a300a78b0f250d0f7c30d0f2d4bfbcb5a93a5070632f5c9421e14daea"
SOURCE_ENSEMBLE_SHA="9777188239034200deedf5423e8cda121e503730eee42ab420eaaa2c8c5dd6f4"

GATE_REPORT="$ROOT/runs/lt3_parallel_8200_gate/20260921_123225/end_to_end_gate.json"

SOURCE_ITERATION=8200
TARGET_ITERATION=9105
ADDITIONAL_ITERATIONS=905
MILESTONE_ITERATION=8600
ROOTS_PER_ITERATION=600
EXPECTED_NEW_ROOTS=543000
EXPECTED_TOTAL_ROOTS=5463000
WORKERS=31
THREADS=8
PARALLEL_CONCURRENCY=4
PARALLEL_MEMBER_THREADS=8
CHECKPOINT_EVERY=50
PROJECTED_HOURS=20.997936387040777

for f in "$PY" "$SOLVER" "$SOURCE_CHECKPOINT" "$SOURCE_ENSEMBLE" "$GATE_REPORT"; do
  [ -e "$f" ] || { echo "ERROR: missing required input: $f" >&2; exit 3; }
done

[ "$(sha256sum "$SOURCE_CHECKPOINT"|awk '{print $1}')" = "$SOURCE_CHECKPOINT_SHA" ] || {
  echo "ERROR: frozen LT3@8200 checkpoint SHA mismatch" >&2; exit 4;
}
[ "$(sha256sum "$SOURCE_ENSEMBLE"|awk '{print $1}')" = "$SOURCE_ENSEMBLE_SHA" ] || {
  echo "ERROR: frozen LT3@8200 ensemble SHA mismatch" >&2; exit 5;
}

"$PY" - "$GATE_REPORT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT3_PARALLEL_8200_GATE_V1"
assert d["verdict"]=="PASS"
assert d["semantic_parity_exact"] is True
assert d["source_unchanged"] is True
assert d["parallel_fit_layout"]=="4x8"
assert d["holdout_touched"] is False
print("LT3_21H_GATE_CONTRACT_PASS")
PY

if pgrep -af "run_lt3_heavy_ens8_h1.py|run_lt3_parallel_ens8_continuation.py" >/dev/null 2>&1; then
  echo "ERROR: an LT3 long trainer is already running." >&2
  pgrep -af "run_lt3_heavy_ens8_h1.py|run_lt3_parallel_ens8_continuation.py" >&2 || true
  exit 7
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source/index changes present; long run requires code == HEAD." >&2
  git status --short --untracked-files=no >&2
  exit 8
fi

MEMTOTAL_KIB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
MEMAVAIL_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if [ -z "$MEMTOTAL_KIB" ] || [ "$MEMTOTAL_KIB" -lt $((28*1024*1024)) ]; then
  echo "ERROR: WSL exposes less than 28 GiB RAM." >&2; exit 9
fi
if [ -z "$MEMAVAIL_KIB" ] || [ "$MEMAVAIL_KIB" -lt $((16*1024*1024)) ]; then
  echo "ERROR: less than 16 GiB WSL MemAvailable." >&2
  free -h >&2 || true
  exit 10
fi

DISK_KIB="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
if [ -z "$DISK_KIB" ] || [ "$DISK_KIB" -lt $((15*1024*1024)) ]; then
  echo "ERROR: less than 15 GiB free on training filesystem." >&2; exit 11
fi

HOST_C_KIB="$(df -Pk /mnt/c 2>/dev/null | awk 'NR==2 {print $4}' || true)"
if [ -n "$HOST_C_KIB" ] && [ "$HOST_C_KIB" -lt $((20*1024*1024)) ]; then
  echo "ERROR: less than 20 GiB free on Windows C: host filesystem." >&2
  exit 12
fi

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export SPINCORE_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

"$PY" -m py_compile   python/spincore_nn/reservoir.py   tools/lt3_hu_ens8_parallel_fit.py   tools/lt3_parallel_continuation_core.py   tools/run_lt3_parallel_ens8_continuation.py

"$PY" python_tests/test_reservoir_write_observer.py
"$PY" python_tests/test_lean_action_policy_ensemble.py
echo "LT3_PARALLEL_21H_PREFLIGHT_PASS"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lt3_parallel_8200_9105/$STAMP"
mkdir -p "$RUN_DIR/work" "$RUN_DIR/milestones"

CHECKPOINT="$RUN_DIR/checkpoint.pt"
ENSEMBLE="$RUN_DIR/hu_ensemble_state.pt"
REPORT="$RUN_DIR/lt3_parallel_8200_9105.json"
LOG="$RUN_DIR/training.log"
MEMLOG="$RUN_DIR/memory.log"

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
  [ -f "$REPORT" ] && cp "$REPORT" "$dest/SpinCore_LT3_parallel_8200_9105.json" || true
  [ -f "$MEMLOG" ] && cp "$MEMLOG" "$dest/SpinCore_LT3_parallel_8200_9105_memory.log" || true
}

echo "=== SpinCore LT3 exact-parity parallel continuation (~21 h) ==="
echo "source=durable LT3@8200 checkpoint+sidecar"
echo "target=9105 / +905 iterations / +543,000 roots"
echo "projected_duration_from_gate≈20.998h"
echo "3H=fresh100; HU=ENS8 8xfresh400 executed 4x8 exact-parity"
echo "milestone 8600 will be preserved automatically"
echo "checkpoint every 50 iterations plus final 9105"
echo "run_dir=$RUN_DIR"
free -h
echo

snapshot_memory before
set +e
(
  /usr/bin/time -v "$PY" tools/run_lt3_parallel_ens8_continuation.py     --solver "$SOLVER"     --source-checkpoint "$SOURCE_CHECKPOINT"     --source-ensemble "$SOURCE_ENSEMBLE"     --output-checkpoint "$CHECKPOINT"     --output-ensemble "$ENSEMBLE"     --report "$REPORT"     --work-dir "$RUN_DIR/work"     --milestone-dir "$RUN_DIR/milestones"     --workers "$WORKERS"     --threads "$THREADS"     --parallel-concurrency "$PARALLEL_CONCURRENCY"     --parallel-member-threads "$PARALLEL_MEMBER_THREADS"     2>&1 | tee -a "$LOG"
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
  echo "ERROR: source 8200 checkpoint changed" >&2; exit 13;
}
[ "$(sha256sum "$SOURCE_ENSEMBLE"|awk '{print $1}')" = "$SOURCE_ENSEMBLE_SHA" ] || {
  copy_evidence
  echo "ERROR: source 8200 ensemble changed" >&2; exit 14;
}

if [ "$STATUS" -ne 0 ]; then
  copy_evidence
  echo "LT3_PARALLEL_8200_9105_TRAINING_FAIL status=$STATUS" >&2
  echo "run_dir=$RUN_DIR" >&2
  echo "Do not restart automatically. Send training.log/report to ChatGPT." >&2
  exit "$STATUS"
fi

"$PY" - "$REPORT" "$CHECKPOINT" "$ENSEMBLE" <<'PY'
import json,sys,torch
from pathlib import Path
report,checkpoint,ensemble=map(Path,sys.argv[1:4])
d=json.loads(report.read_text())
assert d["schema"]=="SPINCORE_LT3_PARALLEL_CONTINUATION_8200_9105_V1"
assert d["status"]=="PASS"
assert d["source_iteration"]==8200
assert d["target_iteration"]==9105
assert d["completed_iteration"]==9105
assert d["source_unchanged"] is True
assert d["intervention"]["parallel_fit_layout"]=="4x8"
assert d["method"]["new_training_roots"]==543000
assert d["holdout_touched"] is False
assert checkpoint.is_file() and ensemble.is_file()
m=d["milestone_files"]
assert int(m["iteration"])==8600
assert Path(m["checkpoint"]).is_file() and Path(m["ensemble"]).is_file()
e=torch.load(ensemble,map_location="cpu",weights_only=False)
assert e["schema"]=="SPINCORE_LT3_HU_ENS8_CURRENT_STATE_V1"
assert int(e["completed_iteration"])==9105
assert int(e["ensemble_size"])==8 and int(e["member_steps"])==400
final=d["final"]["domains"]
assert sum(int(v["roots"]) for v in final.values())==5463000
print("LT3_PARALLEL_8200_9105_POSTVALIDATION_PASS")
print("checkpoint_sha256="+d["output_checkpoint_sha256"])
print("ensemble_sha256="+d["output_ensemble_sha256"])
print("wall_hours="+str(d["wall_seconds"]/3600.0))
PY

copy_evidence
rm -rf "$RUN_DIR/work/packed_hu_adv" || true

echo
echo "LT3_PARALLEL_8200_9105_TRAINING_PASS"
echo "source_8200_unchanged=true"
echo "milestone_8600_preserved=true"
echo "target_9105_complete=true"
echo "run_dir=$RUN_DIR"
echo "report=$REPORT"
echo "STOP HERE. Send SpinCore_LT3_parallel_8200_9105.json to ChatGPT."
