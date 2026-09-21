#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXPECTED_SHA="2b79ab7ff746a9c1c3dd73dbc0a1d6884a471813cf34b9cb126790c4c4cbb123"
DEFAULT_BUNDLE="/mnt/c/Users/Rz9/Downloads/SpinCore_LT2_cpp_deployment_8100.bin"
BUNDLE="${SPINCORE_LT2_BUNDLE:-$DEFAULT_BUNDLE}"

if [ ! -f "$BUNDLE" ]; then
  ALT="$(find "$ROOT/runs" -type f -name 'SpinCore_LT2_cpp_deployment_8100.bin' -print -quit 2>/dev/null || true)"
  if [ -n "$ALT" ]; then
    BUNDLE="$ALT"
  fi
fi

if [ ! -f "$BUNDLE" ]; then
  echo "ERROR: frozen native deployment bundle not found." >&2
  echo "Expected default: $DEFAULT_BUNDLE" >&2
  echo "Or set SPINCORE_LT2_BUNDLE=/path/to/SpinCore_LT2_cpp_deployment_8100.bin" >&2
  exit 3
fi

ACTUAL_SHA="$(sha256sum "$BUNDLE" | awk '{print $1}')"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  echo "ERROR: native deployment bundle SHA256 mismatch" >&2
  echo "expected=$EXPECTED_SHA" >&2
  echo "actual=$ACTUAL_SHA" >&2
  exit 4
fi
echo "FROZEN_BUNDLE_SHA256_PASS"

cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build" --target spincore_lt2_native_openholdem_shadow_engine_audit -j 31
echo "NATIVE_SHADOW_BUILD_PASS"

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="$ROOT/runs/lt2_native_openholdem_shadow_engine/$STAMP"
mkdir -p "$DIR"
REPORT="$DIR/native_openholdem_shadow_engine.json"

"$ROOT/build/spincore_lt2_native_openholdem_shadow_engine_audit" "$BUNDLE" "$REPORT"

python3 - "$REPORT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT2_NATIVE_OPENHOLDEM_SHADOW_ENGINE_V1"
assert d["verdict"]=="PASS"
assert d["failure_count"]==0
assert d["hero_decisions_checked"]==1600
assert d["native_inference_calls"]==1600
assert d["repeated_myturn_cache_hits"]==1600
assert d["exact_selected_action_matches"]==1600
assert d["cache_invalidations_checked"]>0
assert d["decisions_by_domain"]["THREE_HANDED"]>0
assert d["decisions_by_domain"]["TRUE_HEADS_UP"]>0
assert all(d["decisions_by_street"][str(i)]>0 for i in range(4))
assert d["bundle_identity"]["native_file_sha256_expected"]=="2b79ab7ff746a9c1c3dd73dbc0a1d6884a471813cf34b9cb126790c4c4cbb123"
m=d["method"]
assert m["training_roots"]==0
assert m["optimizer_steps"]==0
assert m["strategic_ev_evaluations"]==0
assert m["holdout_reused"] is False
assert m["mode"]=="SHADOW_NO_TABLE_ACTION"
print("LT2_NATIVE_OPENHOLDEM_SHADOW_ENGINE_POSTVALIDATION_PASS")
PY

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_native_openholdem_shadow_engine.json"
fi

echo "LT2_NATIVE_OPENHOLDEM_SHADOW_ENGINE_PASS"
echo "STOP HERE. Send SpinCore_LT2_native_openholdem_shadow_engine.json to ChatGPT."
