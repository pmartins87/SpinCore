#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_PYTHON="${PYTHON_BIN:-python3}"
THREADS="${SPINCORE_TORCH_THREADS:-2}"
BUILD_JOBS="${SPINCORE_BUILD_JOBS:-$(nproc)}"
VENV_DIR="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lean_ryzen_pilot/$STAMP"
mkdir -p "$RUN_DIR"

printf '=== SpinCore lean Ryzen pilot ===\n'
printf 'repo=%s\n' "$ROOT"
printf 'commit=%s\n' "$(git rev-parse HEAD)"
printf 'base_python=%s\n' "$($BASE_PYTHON --version 2>&1)"
printf 'torch_threads=%s build_jobs=%s\n' "$THREADS" "$BUILD_JOBS"

# The pilot is meant to be a one-command local measurement.  A fresh WSL
# install often has Python but not numpy/torch, so bootstrap an isolated venv
# instead of asking the user to mutate the system Python.  Reuse it on later
# runs.  If python3-venv itself is absent, fail with the single apt command
# needed to enable the bootstrap.
if "$BASE_PYTHON" - <<'PY' >/dev/null 2>&1
import numpy, torch
PY
then
    PYTHON_RUN="$BASE_PYTHON"
    printf 'python_env=system (dependencies already present)\n'
else
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        printf 'python_env=creating %s\n' "$VENV_DIR"
        if ! "$BASE_PYTHON" -m venv "$VENV_DIR"; then
            printf '\nERROR: Python venv support is missing.\n' >&2
            printf 'Run: sudo apt update && sudo apt install -y python3-venv\n' >&2
            printf 'Then rerun this same pilot command.\n' >&2
            exit 3
        fi
    else
        printf 'python_env=reusing %s\n' "$VENV_DIR"
    fi

    PYTHON_RUN="$VENV_DIR/bin/python"
    if ! "$PYTHON_RUN" - <<'PY' >/dev/null 2>&1
import numpy, torch
PY
    then
        printf 'Installing lean pilot Python dependencies once...\n'
        "$PYTHON_RUN" -m pip install --upgrade pip
        "$PYTHON_RUN" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu
        "$PYTHON_RUN" -m pip install numpy==2.3.5
    fi
fi

printf 'python=%s\n' "$($PYTHON_RUN --version 2>&1)"
"$PYTHON_RUN" - <<'PY'
import numpy, torch
print(f"torch={torch.__version__}")
print(f"numpy={numpy.__version__}")
PY

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$BUILD_JOBS" --target spincore_solver_c

export PYTHONPATH="$ROOT/python"
export SPINCORE_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

printf '\n=== 5 iterations / 1000 roots ===\n'
/usr/bin/time -v "$PYTHON_RUN" tools/run_lean_functional_training.py \
  --solver build/libspincore_solver_c.so \
  --seed 20260915 \
  --iterations 5 \
  --roots-per-iteration 200 \
  --exact-opponent-levels 0 \
  --advantage-steps 50 \
  --policy-steps 50 \
  --batch-size 256 \
  --checkpoint "$RUN_DIR/checkpoint.pt" \
  --report "$RUN_DIR/report.json" \
  2>&1 | tee "$RUN_DIR/training.log"

printf '\n=== 100-hand offline self-play ===\n'
"$PYTHON_RUN" tools/play_lean_functional_selfplay.py \
  --solver build/libspincore_solver_c.so \
  --checkpoint "$RUN_DIR/checkpoint.pt" \
  --hands 100 \
  --seed 20260916 \
  | tee "$RUN_DIR/selfplay.json"

printf '\nPILOT_PASS\n'
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'report=%s\n' "$RUN_DIR/report.json"
printf 'selfplay=%s\n' "$RUN_DIR/selfplay.json"
