#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"
BUNDLE="/mnt/c/Users/Rz9/Downloads/SpinCore_LT2_hybrid_deployment_8100.pt"
BUNDLE_SHA="87e46b40cb43bb89cb46bf3b760bbac5c8282491fd3d6da73d1bbe28329b278c"
CHECKPOINT_SHA="a51dbbed71090e45f2c4f5db6297f72eab848990b38650702b436e2aa60ca4bf"
ENSEMBLE_SHA="c44b817f75304db352eedc33febbd180e6d038e0580c91be0b9befdbdb08f181"

for f in "$PY" "$BUNDLE"; do
  [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 3; }
done
[ "$(sha256sum "$BUNDLE"|awk '{print $1}')" = "$BUNDLE_SHA" ] || {
  echo "ERROR: validated hybrid deployment bundle SHA mismatch" >&2
  exit 4
}

export PYTHONPATH="$ROOT/python:$ROOT/tools"
"$PY" -m py_compile   tools/export_lt2_cpp_deployment.py   tools/generate_lt2_cpp_parity_fixtures.py
echo "PYTHON_PREFLIGHT_PASS"

cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build"   --target spincore_solver_c spincore_lt2_cpp_inference_parity   -j 31
echo "CPP_BUILD_PASS"

[ -e "$SOLVER" ] || { echo "ERROR: solver library missing after build" >&2; exit 5; }
CPP="$ROOT/build/spincore_lt2_cpp_inference_parity"
[ -x "$CPP" ] || { echo "ERROR: native parity executable missing" >&2; exit 6; }

STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="$ROOT/runs/lt2_native_cpp_inference_parity/$STAMP"
mkdir -p "$DIR"
NATIVE="$DIR/SpinCore_LT2_cpp_deployment_8100.bin"
FIXTURES="$DIR/lt2_cpp_parity_fixtures.bin"
REPORT="$DIR/lt2_cpp_inference_parity.json"

"$PY" tools/export_lt2_cpp_deployment.py   --source "$BUNDLE"   --out "$NATIVE"

"$PY" tools/generate_lt2_cpp_parity_fixtures.py   --solver "$SOLVER"   --bundle "$BUNDLE"   --out "$FIXTURES"   --scenarios-per-seed 700   --max-decisions 20000

"$CPP" "$NATIVE" "$FIXTURES" "$REPORT" 0.0002

"$PY" - "$REPORT" "$BUNDLE_SHA" "$CHECKPOINT_SHA" "$ENSEMBLE_SHA" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT2_NATIVE_CPP_INFERENCE_PARITY_V1"
assert d["verdict"]=="PASS"
assert d["source_identity"]["deployment_bundle_sha256"]==sys.argv[2]
assert d["source_identity"]["source_checkpoint_sha256"]==sys.argv[3]
assert d["source_identity"]["source_ensemble_sha256"]==sys.argv[4]
assert d["three_handed_records"]>0
assert d["heads_up_records"]>0
assert d["streets"]["preflop"]>0
assert d["streets"]["flop"]+d["streets"]["turn"]+d["streets"]["river"]>0
assert d["argmax_mismatches"]==0
assert d["illegal_mass_failures"]==0
assert d["nonfinite_failures"]==0
assert d["max_abs_probability_diff"]<=d["tolerance"]
assert d["max_probability_mass_error"]<=d["tolerance"]
print("LT2_NATIVE_CPP_INFERENCE_PARITY_POSTVALIDATION_PASS")
PY

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$NATIVE" "$DEST/SpinCore_LT2_cpp_deployment_8100.bin"
  cp "$REPORT" "$DEST/SpinCore_LT2_cpp_inference_parity.json"
fi

echo "LT2_NATIVE_CPP_INFERENCE_PARITY_PASS"
echo "STOP HERE. Send SpinCore_LT2_cpp_inference_parity.json to ChatGPT."
echo "Keep SpinCore_LT2_cpp_deployment_8100.bin in Downloads."
