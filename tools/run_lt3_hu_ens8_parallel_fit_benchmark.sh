#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(git -C "$ROOT" rev-parse --show-toplevel)"
cd "$ROOT"

PY="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOLVER="$ROOT/build/libspincore_solver_c.so"
SOURCE="/home/rz9/spincore_lean_functional/runs/lt2_hu_ens8_online_pilot/20260920_154905/checkpoint.pt"
SOURCE_SHA="a51dbbed71090e45f2c4f5db6297f72eab848990b38650702b436e2aa60ca4bf"
STAMP="$(date +%Y%m%d_%H%M%S)"
DIR="$ROOT/runs/lt3_hu_ens8_parallel_fit_benchmark/$STAMP"
REPORT="$DIR/parallel_fit_benchmark.json"

[ -x "$PY" ] || { echo "ERROR: Python venv missing" >&2; exit 3; }
[ -f "$SOLVER" ] || { echo "ERROR: solver missing" >&2; exit 3; }
[ -f "$SOURCE" ] || { echo "ERROR: source checkpoint missing" >&2; exit 3; }
[ "$(sha256sum "$SOURCE"|awk '{print $1}')" = "$SOURCE_SHA" ] || {
  echo "ERROR: frozen LT2@8100 checkpoint SHA mismatch" >&2
  exit 4
}

if pgrep -af "tools/run_lt3_heavy_ens8_h1.py" >/dev/null 2>&1; then
  echo "ERROR: LT3 H1 training is still running. Stop it before benchmarking." >&2
  pgrep -af "tools/run_lt3_heavy_ens8_h1.py" >&2 || true
  exit 5
fi

mkdir -p "$DIR"
export PYTHONPATH="$ROOT/python:$ROOT/tools"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

"$PY" -m py_compile tools/lt3_hu_ens8_parallel_fit.py tools/benchmark_lt3_hu_ens8_parallel_fit.py
echo "LT3_PARALLEL_FIT_PREFLIGHT_PASS"

"$PY" tools/benchmark_lt3_hu_ens8_parallel_fit.py   --solver "$SOLVER"   --source-checkpoint "$SOURCE"   --report "$REPORT"   --concurrency 4   --threads-per-member 8   --parent-threads 8

"$PY" - "$REPORT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT3_HU_ENS8_PARALLEL_FIT_BENCHMARK_V1"
assert d["verdict"]=="PASS"
assert d["all_member_states_exact"] is True
assert d["all_member_loss_last_exact"] is True
assert float(d["speedup"])>=1.25
assert d["source_checkpoint_read_only"] is True
assert d["training_roots"]==0
assert d["holdout_touched"] is False
print("LT3_HU_ENS8_PARALLEL_FIT_BENCHMARK_PASS")
PY

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT3_HU_ENS8_parallel_fit_benchmark.json"
fi

echo "STOP HERE. Send SpinCore_LT3_HU_ENS8_parallel_fit_benchmark.json to ChatGPT."
