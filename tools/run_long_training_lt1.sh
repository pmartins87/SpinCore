#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
WORKER_FILE="$ROOT/runs/worker_benchmark/selected_workers.txt"
THREAD_FILE="$ROOT/runs/worker_benchmark/selected_torch_threads.txt"
BUILD_JOBS="${SPINCORE_BUILD_JOBS:-$(nproc)}"

if [ ! -x "$PYTHON_RUN" ]; then
    echo "ERROR: lean Python environment not found at $PYTHON_RUN" >&2
    exit 3
fi
if [ ! -f "$WORKER_FILE" ] || [ ! -f "$THREAD_FILE" ]; then
    echo "ERROR: Ryzen profile not found. Run tools/benchmark_lean_ryzen_workers.sh first." >&2
    exit 4
fi

WORKERS="$(tr -d '[:space:]' < "$WORKER_FILE")"
THREADS="$(tr -d '[:space:]' < "$THREAD_FILE")"
MEM_KIB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
MEM_GIB="$($PYTHON_RUN - "$MEM_KIB" <<'PY'
import sys
print(f"{int(sys.argv[1]) / 1024 / 1024:.2f}")
PY
)"
FREE_KIB="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
FREE_GIB="$($PYTHON_RUN - "$FREE_KIB" <<'PY'
import sys
print(f"{int(sys.argv[1]) / 1024 / 1024:.2f}")
PY
)"

# The physical Ryzen has 64 GiB RAM, but WSL can be configured to expose less.
# Long training must be sized from what Linux can actually use, not from the
# DIMMs installed in Windows.
if [ "$MEM_KIB" -lt $((28 * 1024 * 1024)) ]; then
    echo "ERROR: WSL exposes only ${MEM_GIB} GiB RAM." >&2
    echo "LT1 requires at least 28 GiB visible to WSL before starting the production-shaped campaign." >&2
    echo "Do not lower the reservoir merely to make the run start; increase the WSL memory allocation instead." >&2
    exit 5
fi
if [ "$FREE_KIB" -lt $((30 * 1024 * 1024)) ]; then
    echo "ERROR: only ${FREE_GIB} GiB free on the training filesystem." >&2
    echo "LT1 requires at least 30 GiB free for large resumable checkpoints and logs." >&2
    exit 6
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/long_training_lt1/$STAMP"
mkdir -p "$RUN_DIR"

printf '=== SpinCore LT1 production-shaped long-training campaign ===\n'
printf 'purpose=infrastructure/scale milestone that can be preserved and extended into LT2\n'
printf 'physical_ram_gib=64\n'
printf 'wsl_visible_ram_gib=%s\n' "$MEM_GIB"
printf 'filesystem_free_gib=%s\n' "$FREE_GIB"
printf 'logical_cpus=%s\n' "$(nproc)"
printf 'root_workers=%s parent_torch_threads=%s build_jobs=%s\n' "$WORKERS" "$THREADS" "$BUILD_JOBS"
printf 'iterations=2000 roots_per_iteration=600 total_roots=1200000\n'
printf 'reservoir_capacity_per_memory_per_domain=2000000\n'
printf 'max_reservoir_slots=8000000 (2 domains x advantage/policy x 2M)\n'
printf 'advantage_fit=100 steps x batch 1024 per domain per iteration\n'
printf 'average_policy_final_fit=4000 steps x batch 1024 per domain\n'
printf 'checkpoint_every=100 iterations\n'
printf 'run_dir=%s\n\n' "$RUN_DIR"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$BUILD_JOBS" --target spincore_solver_c

export PYTHONPATH="$ROOT/python"
export SPINCORE_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

/usr/bin/time -v "$PYTHON_RUN" tools/run_lean_functional_training.py \
  --solver build/libspincore_solver_c.so \
  --seed 20260922 \
  --iterations 2000 \
  --roots-per-iteration 600 \
  --exact-opponent-levels 0 \
  --reservoir-capacity 2000000 \
  --advantage-steps 100 \
  --policy-steps 4000 \
  --batch-size 1024 \
  --workers "$WORKERS" \
  --checkpoint-every 100 \
  --checkpoint "$RUN_DIR/checkpoint.pt" \
  --report "$RUN_DIR/report.json" \
  2>&1 | tee "$RUN_DIR/training.log"

printf '\nLT1_SCALE_VALIDATION_PASS\n'
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'checkpoint=%s\n' "$RUN_DIR/checkpoint.pt"
printf 'report=%s\n' "$RUN_DIR/report.json"
printf 'next=inspect throughput/RSS/checkpoint cost, then extend this same checkpoint into LT2 rather than restarting if healthy\n'
