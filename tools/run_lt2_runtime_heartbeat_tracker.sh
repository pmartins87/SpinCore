#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
export PYTHONPATH="$ROOT/python:$ROOT/tools"

"$PY" -m py_compile   python/spincore/runtime_transition_reconciler.py   python/spincore/runtime_heartbeat_tracker.py   tools/audit_lt2_runtime_heartbeat_tracker.py
echo "PYTHON_PREFLIGHT_PASS"

cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build" --target spincore_solver_c -j 31
echo "SOLVER_BUILD_PASS"

SOLVER="$ROOT/build/libspincore_solver_c.so"
[ -e "$SOLVER" ] || { echo "ERROR: missing rebuilt solver $SOLVER" >&2; exit 3; }

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="$ROOT/runs/lt2_runtime_heartbeat_tracker/$STAMP"
mkdir -p "$DIR"
REPORT="$DIR/runtime_heartbeat_tracker.json"

echo "=== SpinCore LT2 OpenHoldem heartbeat/lifecycle tracker gate ==="
echo "old forensic seeds only; NO MODEL/EV/TRAINING; HOLDOUT NOT REUSED"

"$PY" tools/audit_lt2_runtime_heartbeat_tracker.py   --solver "$SOLVER"   --scenarios-per-seed 650   --max-transitions 10000   --fault-hands 500   --report "$REPORT"

"$PY" - "$REPORT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT2_RUNTIME_HEARTBEAT_TRACKER_V1"
assert d["verdict"]=="PASS"
m=d["method"]
assert m["training_roots"]==0
assert m["optimizer_steps"]==0
assert m["strategic_ev_evaluations"]==0
assert m["deployment_model_inference"]==0
assert m["holdout_reused"] is False
assert d["failure_count"]==0
assert d["action_events"]==d["transitions_checked"]
assert d["action_mismatches"]==0
assert d["transcript_mismatches"]==0
assert d["duplicate_heartbeats_checked"]>0
assert d["decision_computations"]>0
assert d["repeated_myturn_cache_hits"]==d["decision_computations"]
assert d["cache_invalidation_checks"]==d["cache_invalidation_pass"]
f=d["faults"]
assert f["corrupt_attempts"]==f["corrupt_rejections"]
assert f["wrong_hand_attempts"]==f["wrong_hand_rejections"]
assert f["skipped_attempts"]>0
assert f["skipped_attempts"]==f["skipped_rejections"]
assert f["failure_latch_checks"]==f["failure_latch_pass"]
assert f["reset_recovery_attempts"]==f["reset_recovery_pass"]
print("LT2_RUNTIME_HEARTBEAT_TRACKER_POSTVALIDATION_PASS")
PY

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_runtime_heartbeat_tracker.json"
fi

echo "LT2_RUNTIME_HEARTBEAT_TRACKER_PASS"
echo "STOP HERE. Send SpinCore_LT2_runtime_heartbeat_tracker.json to ChatGPT."
