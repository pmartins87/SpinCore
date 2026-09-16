#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOURCE="${SPINCORE_LT2_SOURCE_CHECKPOINT:-/home/rz9/spincore_lean_functional/runs/long_training_lt1/20260915_181249/checkpoint.pt}"
EXPECTED_SOURCE_SHA256="beef9bee9439de8d9153450d190e62b0388a678b0778c4185108361f626b4337"
ADDITIONAL_ITERATIONS=1000
WORKERS=31
THREADS=8
BATCH_MODE=vectorized
CHECKPOINT_EVERY=100
MIN_MEMTOTAL_KIB=$((28 * 1024 * 1024))
MIN_DISK_KIB=$((15 * 1024 * 1024))

if [ ! -x "$PYTHON_RUN" ]; then
    echo "ERROR: lean Python environment not found: $PYTHON_RUN" >&2
    exit 3
fi
if [ ! -f "$SOURCE" ]; then
    echo "ERROR: LT1 source checkpoint not found: $SOURCE" >&2
    exit 4
fi
if [ ! -f "$ROOT/build/libspincore_solver_c.so" ]; then
    echo "ERROR: solver missing: $ROOT/build/libspincore_solver_c.so" >&2
    exit 5
fi

MEMTOTAL_KIB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
if [ -z "$MEMTOTAL_KIB" ] || [ "$MEMTOTAL_KIB" -lt "$MIN_MEMTOTAL_KIB" ]; then
    echo "ERROR: WSL exposes less than 28 GiB RAM; refusing LT2 Stage A." >&2
    exit 6
fi

DISK_KIB="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
if [ -z "$DISK_KIB" ] || [ "$DISK_KIB" -lt "$MIN_DISK_KIB" ]; then
    echo "ERROR: less than 15 GiB free on the training filesystem." >&2
    exit 7
fi

SOURCE_SHA256="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [ "$SOURCE_SHA256" != "$EXPECTED_SOURCE_SHA256" ]; then
    echo "ERROR: LT1 source checkpoint SHA256 mismatch." >&2
    echo "expected=$EXPECTED_SOURCE_SHA256" >&2
    echo "actual=$SOURCE_SHA256" >&2
    exit 8
fi

# Generated run artifacts are intentionally untracked. Fail only on changes to
# tracked source/index content, and do this before creating the new run directory.
GIT_COMMIT="$(git rev-parse HEAD)"
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked source/index changes present; LT2 Stage A requires tracked code to match HEAD." >&2
    git status --short --untracked-files=no >&2
    exit 10
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/long_training_lt2/$STAMP"
mkdir -p "$RUN_DIR"
CHECKPOINT="$RUN_DIR/checkpoint.pt"
REPORT="$RUN_DIR/report.json"
LOG="$RUN_DIR/training.log"
MEMLOG="$RUN_DIR/memory.log"
PROVENANCE="$RUN_DIR/provenance.txt"

printf 'Copying preserved LT1 checkpoint into isolated LT2 run directory...\n'
cp --reflink=auto "$SOURCE" "$CHECKPOINT"
COPY_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [ "$COPY_SHA256" != "$SOURCE_SHA256" ]; then
    echo "ERROR: copied checkpoint hash mismatch." >&2
    exit 9
fi

cat > "$PROVENANCE" <<EOF
schema=SPINCORE_LT2_STAGE_A_V1
created_at=$(date --iso-8601=seconds)
git_commit=$GIT_COMMIT
source_checkpoint=$SOURCE
source_sha256=$SOURCE_SHA256
lt2_checkpoint=$CHECKPOINT
additional_iterations=$ADDITIONAL_ITERATIONS
workers=$WORKERS
parent_torch_threads=$THREADS
batch_mode=$BATCH_MODE
checkpoint_every=$CHECKPOINT_EVERY
EOF

export PYTHONPATH="$ROOT/python"
export SPINCORE_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

snapshot_memory() {
    local tag="$1"
    local now
    now="$(date --iso-8601=seconds)"
    awk -v ts="$now" -v tag="$tag" '
        /^MemTotal:/ {mt=$2}
        /^MemAvailable:/ {ma=$2}
        /^SwapTotal:/ {st=$2}
        /^SwapFree:/ {sf=$2}
        END {
            printf "%s tag=%s mem_total_kib=%s mem_available_kib=%s swap_total_kib=%s swap_free_kib=%s swap_used_kib=%s\n", ts, tag, mt, ma, st, sf, st-sf
        }
    ' /proc/meminfo >> "$MEMLOG"
}

monitor_memory() {
    local watched_pid="$1"
    while kill -0 "$watched_pid" 2>/dev/null; do
        snapshot_memory periodic
        sleep 60
    done
}

printf '=== SpinCore LT2 Stage A ===\n'
printf 'source=%s\n' "$SOURCE"
printf 'source_sha256=%s\n' "$SOURCE_SHA256"
printf 'git_commit=%s\n' "$GIT_COMMIT"
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'continuation=iteration 2001 -> 3000 (%d additional iterations / 600000 roots)\n' "$ADDITIONAL_ITERATIONS"
printf 'workers=%d parent_torch_threads=%d batch_mode=%s checkpoint_every=%d\n' "$WORKERS" "$THREADS" "$BATCH_MODE" "$CHECKPOINT_EVERY"
printf 'purpose=bounded memory/reservoir/checkpoint gate before larger LT2 blocks\n\n'

snapshot_memory before

set +e
(
    /usr/bin/time -v "$PYTHON_RUN" tools/run_lean_functional_training.py \
      --solver build/libspincore_solver_c.so \
      --resume \
      --additional-iterations "$ADDITIONAL_ITERATIONS" \
      --workers "$WORKERS" \
      --checkpoint-every "$CHECKPOINT_EVERY" \
      --checkpoint "$CHECKPOINT" \
      --report "$REPORT" \
      --batch-mode "$BATCH_MODE" \
      2>&1 | tee -a "$LOG"
    exit "${PIPESTATUS[0]}"
) &
TRAIN_PID=$!
monitor_memory "$TRAIN_PID" &
MONITOR_PID=$!
wait "$TRAIN_PID"
STATUS=$?
kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
set -e

snapshot_memory after

SOURCE_SHA256_AFTER="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [ "$SOURCE_SHA256_AFTER" != "$EXPECTED_SOURCE_SHA256" ]; then
    echo "ERROR: preserved LT1 source changed during LT2 Stage A." >&2
    exit 11
fi

if [ "$STATUS" -ne 0 ]; then
    echo "LT2_STAGE_A_FAIL status=$STATUS" >&2
    echo "run_dir=$RUN_DIR" >&2
    echo "Do not restart automatically; inspect training.log and memory.log." >&2
    exit "$STATUS"
fi

if [ ! -f "$REPORT" ]; then
    echo "ERROR: trainer exited 0 but report.json is missing." >&2
    exit 12
fi

printf '\nLT2_STAGE_A_PASS\n'
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'report=%s\n' "$REPORT"
printf 'memory_log=%s\n' "$MEMLOG"
printf 'source_unchanged=true\n'
printf 'STOP HERE. Do not launch a larger LT2 block until Stage A evidence is reviewed.\n'
