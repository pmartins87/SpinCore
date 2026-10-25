#!/usr/bin/env bash
set -euo pipefail

ROOT="${SPINCORE_ROOT:-$HOME/spincore_lean_functional}"
cd "$ROOT"

RUN_DIR="$ROOT/runs/teacher_sanity_17315"
mkdir -p "$RUN_DIR"

SOLVER="$ROOT/build/libspincore_solver_c.so"
TEACHER="$ROOT/runs/3h_semantic_long_15315_17315/semantic_long_ensemble_final_17315.pt"
REPORT="$RUN_DIR/teacher_sanity_17315.json"
LOG="$RUN_DIR/teacher_sanity_17315.log"

if [[ ! -f "$SOLVER" ]]; then
  echo "missing solver: $SOLVER" >&2
  exit 2
fi
if [[ ! -f "$TEACHER" ]]; then
  echo "missing teacher 17315: $TEACHER" >&2
  exit 2
fi

echo "SPINCORE_3H_TEACHER_SANITY_17315_START"
echo "root=$ROOT"
echo "teacher=$TEACHER"
echo "scenarios=10000 seed=20261025 workers=16"

PY="$ROOT/.venv_lean/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "missing python: $PY" >&2
  exit 2
fi

set +e
nice -n 5 taskset -c 0-15 "$PY" tools/audit_3h_teacher_sanity_17315.py   --solver "$SOLVER"   --teacher "$TEACHER"   --scenarios 10000   --workers 16   --seed 20261025   --max-decisions 300   --report "$REPORT"   2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

if [[ "$RC" -ne 0 ]]; then
  echo "SPINCORE_3H_TEACHER_SANITY_17315_FAIL rc=$RC"
  exit "$RC"
fi

grep -q "SPINCORE_3H_TEACHER_SANITY_17315_COMPLETE" "$LOG"
test -s "$REPORT"

cp -f "$REPORT" "$HOME/Desktop/SpinCore_teacher_sanity_17315.json" 2>/dev/null || true
cp -f "$LOG" "$HOME/Desktop/SpinCore_teacher_sanity_17315.log" 2>/dev/null || true

echo "SPINCORE_3H_TEACHER_SANITY_17315_RUNNER_PASS"
echo "report=$REPORT"
echo "log=$LOG"
