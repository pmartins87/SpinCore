#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build" --target spincore_lt2_native_openholdem_tracker_audit -j 31
echo "NATIVE_BUILD_PASS"

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="$ROOT/runs/lt2_native_openholdem_tracker/$STAMP"
mkdir -p "$DIR"
REPORT="$DIR/native_openholdem_tracker.json"

"$ROOT/build/spincore_lt2_native_openholdem_tracker_audit" "$REPORT"

python3 - "$REPORT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT2_NATIVE_OPENHOLDEM_TRACKER_V1"
assert d["verdict"]=="PASS"
m=d["method"]
assert m["training_roots"]==0
assert m["optimizer_steps"]==0
assert m["strategic_ev_evaluations"]==0
assert m["deployment_model_inference"]==0
assert m["holdout_reused"] is False
assert d["failure_count"]==0
assert d["transitions_checked"]==12000
assert d["hero_canonical_state_checks"]>0
assert d["silent_check_deferrals"]>0
assert d["delayed_actions_reconciled_at_myturn"]>0 or d["multi_action_sync_events"]>0
assert d["street_reveals_checked"]>0
f=d["faults"]
assert f["corrupt_attempts"]==500==f["corrupt_rejections"]
assert f["skipped_attempts"]==500==f["skipped_rejections"]
print("LT2_NATIVE_OPENHOLDEM_TRACKER_POSTVALIDATION_PASS")
PY

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_native_openholdem_tracker.json"
fi

echo "LT2_NATIVE_OPENHOLDEM_TRACKER_PASS"
echo "STOP HERE. Send SpinCore_LT2_native_openholdem_tracker.json to ChatGPT."
