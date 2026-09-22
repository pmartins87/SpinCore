#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${1:-}"

if [[ -z "$SOURCE" ]]; then
  echo "usage: bash tools/run_deepcrusher_dc0_static_prepare.sh /path/to/DeepCrusher_R8_v22_CANDIDATE_OPENHOLDEM_ASCII_20260914.txt"
  exit 2
fi
if [[ ! -f "$SOURCE" ]]; then
  echo "DEEPC_RUSHER_SOURCE_NOT_FOUND: $SOURCE"
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/runs/deepcrusher_dc0_static_prepare/$STAMP"
mkdir -p "$OUT"

export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

echo "=== SpinCore DeepCrusher DC0 static preparation ==="
echo "root=$ROOT"
echo "source=$SOURCE"
echo "out=$OUT"

python "$ROOT/python_tests/test_openppl_expr.py"
python "$ROOT/python_tests/test_openppl_program.py"

python "$ROOT/tools/preflight_deepcrusher_benchmark.py" \
  --deepcrusher-source "$SOURCE" \
  --json "$OUT/preflight.json"

python "$ROOT/tools/audit_deepcrusher_r8_openppl_compile.py" \
  --deepcrusher-source "$SOURCE" \
  --report "$OUT/openppl_compile.json"

# Readiness is expected to remain BLOCKED until the remaining semantic/runtime
# capabilities are proved.  Preserve the report without treating BLOCKED as a
# shell failure.
set +e
python "$ROOT/tools/audit_deepcrusher_r8_dc0_readiness.py" \
  --deepcrusher-source "$SOURCE" \
  --report "$OUT/dc0_readiness.json"
READINESS_RC=$?
set -e

if [[ "$READINESS_RC" -ne 0 && "$READINESS_RC" -ne 2 ]]; then
  echo "DEEPC_RUSHER_DC0_READINESS_AUDIT_ERROR rc=$READINESS_RC"
  exit "$READINESS_RC"
fi

echo "DEEPC_RUSHER_DC0_STATIC_PREPARE_PASS"
echo "readiness_expected_blocked_rc=$READINESS_RC"
echo "reports=$OUT"
