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
SOURCE_PROGRESS="$SOURCE_DIR/lt3_heavy_h1.json"

LT2_8100_REPORT="$ROOT/runs/lt2_hu_ens8_online_pilot/20260920_154905/hu_ens8_online_pilot.json"

for f in "$PY" "$SOLVER" "$SOURCE_CHECKPOINT" "$SOURCE_ENSEMBLE"; do
  [ -e "$f" ] || { echo "ERROR: missing required input: $f" >&2; exit 3; }
done

if pgrep -af "run_lt3_heavy_ens8_h1.py|run_lt3_parallel_ens8_continuation.py" >/dev/null 2>&1; then
  echo "ERROR: a long LT3 trainer is still running." >&2
  pgrep -af "run_lt3_heavy_ens8_h1.py|run_lt3_parallel_ens8_continuation.py" >&2 || true
  exit 4
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source/index changes present; gate requires code == HEAD." >&2
  git status --short --untracked-files=no >&2
  exit 5
fi

export PYTHONPATH="$ROOT/python:$ROOT/tools"
export SPINCORE_TORCH_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

"$PY" -m py_compile   python/spincore_nn/reservoir.py   tools/lt3_hu_ens8_parallel_fit.py   tools/lt3_parallel_continuation_core.py   tools/validate_lt3_parallel_8200_gate.py

"$PY" -m unittest python_tests.test_reservoir_write_observer
"$PY" python_tests/test_lean_action_policy_ensemble.py
echo "LT3_8200_PARALLEL_GATE_PREFLIGHT_PASS"

# Verify the local interrupted run really is the matched durable 8200 pair.
"$PY" - "$SOURCE_CHECKPOINT" "$SOURCE_ENSEMBLE" <<'PY'
import sys,torch
from pathlib import Path
cp=Path(sys.argv[1]); ep=Path(sys.argv[2])
c=torch.load(cp,map_location="cpu",weights_only=False)
e=torch.load(ep,map_location="cpu",weights_only=False)
assert int(c["completed_iteration"])==8200
assert int(e["completed_iteration"])==8200
assert e["schema"]=="SPINCORE_LT3_HU_ENS8_CURRENT_STATE_V1"
assert int(e["ensemble_size"])==8 and int(e["member_steps"])==400
print("LT3_8200_SOURCE_PAIR_PASS")
PY

MEM_AVAIL_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if [ "$MEM_AVAIL_KIB" -lt $((16*1024*1024)) ]; then
  echo "ERROR: less than 16 GiB WSL MemAvailable before gate." >&2
  free -h >&2 || true
  exit 6
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="$ROOT/runs/lt3_parallel_8200_gate/$STAMP"
mkdir -p "$DIR/sequential_work" "$DIR/parallel_work"

SEQ="$DIR/sequential_arm.json"
PAR="$DIR/parallel_arm.json"
REPORT="$DIR/end_to_end_gate.json"
LOG="$DIR/gate.log"

SOURCE_CP_SHA_BEFORE="$(sha256sum "$SOURCE_CHECKPOINT"|awk '{print $1}')"
SOURCE_ENS_SHA_BEFORE="$(sha256sum "$SOURCE_ENSEMBLE"|awk '{print $1}')"

echo "=== LT3 @8200 exact parallel integration gate ===" | tee "$LOG"
echo "source_checkpoint_sha256=$SOURCE_CP_SHA_BEFORE" | tee -a "$LOG"
echo "source_ensemble_sha256=$SOURCE_ENS_SHA_BEFORE" | tee -a "$LOG"
echo "arm1=one disposable sequential iteration 8201" | tee -a "$LOG"
echo "arm2=three disposable parallel 4x8 iterations 8201..8203" | tee -a "$LOG"
echo "source artifacts remain read-only; no holdout is touched" | tee -a "$LOG"
free -h | tee -a "$LOG"

set +e
"$PY" tools/validate_lt3_parallel_8200_gate.py sequential   --solver "$SOLVER"   --source-checkpoint "$SOURCE_CHECKPOINT"   --source-ensemble "$SOURCE_ENSEMBLE"   --output "$SEQ"   --work-dir "$DIR/sequential_work"   --workers 31   --threads 8 2>&1 | tee -a "$LOG"
S1=${PIPESTATUS[0]}
set -e
if [ "$S1" -ne 0 ]; then
  echo "LT3_8200_PARALLEL_GATE_FAIL sequential_status=$S1" >&2
  echo "STOP HERE. Send $LOG to ChatGPT." >&2
  exit "$S1"
fi

echo "AFTER_SEQUENTIAL_ARM" | tee -a "$LOG"
free -h | tee -a "$LOG"

set +e
"$PY" tools/validate_lt3_parallel_8200_gate.py parallel   --solver "$SOLVER"   --source-checkpoint "$SOURCE_CHECKPOINT"   --source-ensemble "$SOURCE_ENSEMBLE"   --output "$PAR"   --work-dir "$DIR/parallel_work"   --workers 31   --threads 8   --parallel-concurrency 4   --parallel-member-threads 8 2>&1 | tee -a "$LOG"
S2=${PIPESTATUS[0]}
set -e
if [ "$S2" -ne 0 ]; then
  echo "LT3_8200_PARALLEL_GATE_FAIL parallel_status=$S2" >&2
  echo "STOP HERE. Send $LOG to ChatGPT." >&2
  exit "$S2"
fi

COMPARE_ARGS=(
  compare
  --sequential-report "$SEQ"
  --parallel-report "$PAR"
  --output "$REPORT"
)
[ -f "$SOURCE_PROGRESS" ] && COMPARE_ARGS+=(--source-progress-report "$SOURCE_PROGRESS")
[ -f "$LT2_8100_REPORT" ] && COMPARE_ARGS+=(--lt2-8100-report "$LT2_8100_REPORT")

"$PY" tools/validate_lt3_parallel_8200_gate.py "${COMPARE_ARGS[@]}" 2>&1 | tee -a "$LOG"

SOURCE_CP_SHA_AFTER="$(sha256sum "$SOURCE_CHECKPOINT"|awk '{print $1}')"
SOURCE_ENS_SHA_AFTER="$(sha256sum "$SOURCE_ENSEMBLE"|awk '{print $1}')"
[ "$SOURCE_CP_SHA_AFTER" = "$SOURCE_CP_SHA_BEFORE" ] || {
  echo "ERROR: 8200 source checkpoint changed" >&2; exit 7;
}
[ "$SOURCE_ENS_SHA_AFTER" = "$SOURCE_ENS_SHA_BEFORE" ] || {
  echo "ERROR: 8200 source ensemble changed" >&2; exit 8;
}

"$PY" - "$REPORT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT3_PARALLEL_8200_GATE_V1"
assert d["verdict"]=="PASS"
assert d["semantic_parity_exact"] is True
assert d["source_unchanged"] is True
assert float(d["whole_iteration_speedup_same_iteration"])>=1.15
assert d["holdout_touched"] is False
print("LT3_8200_PARALLEL_END_TO_END_GATE_PASS")
print("whole_iteration_speedup="+str(d["whole_iteration_speedup_same_iteration"]))
print("parallel_median_seconds="+str(d["parallel_median_wall_seconds"]))
print("recommended_24h_target="+str(d["recommended_24h"]["target_iteration"]))
print("recommended_24h_additional="+str(d["recommended_24h"]["additional_iterations"]))
print("projected_hours="+str(d["recommended_24h"]["projected_total_hours"]))
PY

# The mmap mirror is disposable gate infrastructure. Remove it after evidence is
# sealed so it does not waste host/WSL disk.
rm -rf "$DIR/parallel_work/packed_hu_adv" || true

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT3_8200_parallel_end_to_end_gate.json"
  cp "$LOG" "$DEST/SpinCore_LT3_8200_parallel_end_to_end_gate.log"
fi

echo
echo "LT3_8200_PARALLEL_END_TO_END_GATE_PASS"
echo "source_8200_unchanged=true"
echo "long_training_started=false"
echo "report=$REPORT"
echo "STOP HERE. Send SpinCore_LT3_8200_parallel_end_to_end_gate.json to ChatGPT."
