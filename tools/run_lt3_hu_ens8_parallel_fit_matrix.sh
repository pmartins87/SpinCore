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
DIR="$ROOT/runs/lt3_hu_ens8_parallel_fit_matrix/$STAMP"
REPORT="$DIR/parallel_fit_matrix.json"

for f in "$PY" "$SOLVER" "$SOURCE"; do
  [ -e "$f" ] || { echo "ERROR: missing required input: $f" >&2; exit 3; }
done
[ "$(sha256sum "$SOURCE"|awk '{print $1}')" = "$SOURCE_SHA" ] || {
  echo "ERROR: frozen LT2@8100 checkpoint SHA mismatch" >&2; exit 4;
}

if pgrep -af "tools/run_lt3_heavy_ens8_h1.py" >/dev/null 2>&1; then
  echo "ERROR: LT3 H1 training is still running. Stop it before benchmarking." >&2
  pgrep -af "tools/run_lt3_heavy_ens8_h1.py" >&2 || true
  exit 5
fi

mkdir -p "$DIR"
export PYTHONPATH="$ROOT/python:$ROOT/tools"

"$PY" -m py_compile   tools/lt3_hu_ens8_parallel_fit.py   tools/benchmark_lt3_hu_ens8_parallel_fit_matrix.py
MEM_AVAIL_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if [ "$MEM_AVAIL_KIB" -lt $((12*1024*1024)) ]; then
  echo "ERROR: less than 12 GiB WSL MemAvailable before V2 matrix." >&2
  free -h >&2 || true
  exit 12
fi
echo "LT3_PARALLEL_FIT_MATRIX_V2_PREFLIGHT_PASS"
free -h || true

"$PY" tools/benchmark_lt3_hu_ens8_parallel_fit_matrix.py   --solver "$SOLVER"   --source-checkpoint "$SOURCE"   --report "$REPORT"   --parent-threads 8

"$PY" - "$REPORT" <<'PY'
import json,sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
assert d["schema"]=="SPINCORE_LT3_HU_ENS8_PARALLEL_FIT_MATRIX_V2"
assert d["verdict"]=="PASS"
best=d["best_exact_candidate"]
assert best is not None
assert best["all_member_states_exact"] is True
assert best["all_member_loss_last_exact"] is True
assert float(best["speedup_steady_state"])>=1.25
assert d["source_checkpoint_read_only"] is True
assert d["training_roots"]==0
assert d["holdout_touched"] is False
print("LT3_HU_ENS8_PARALLEL_FIT_MATRIX_PASS")
print("best_exact_candidate="+best["name"])
print("best_speedup="+str(best["speedup_steady_state"]))
PY

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT3_HU_ENS8_parallel_fit_matrix_v2.json"
fi

echo "STOP HERE. Send SpinCore_LT3_HU_ENS8_parallel_fit_matrix_v2.json to ChatGPT."
