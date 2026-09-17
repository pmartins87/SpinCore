#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
SOURCE="${SPINCORE_LT2B_SOURCE_CHECKPOINT:-/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt}"
EXPECTED_SOURCE_SHA256="e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c"
ADDITIONAL_ITERATIONS=4500
TARGET_ITERATION=7500
EXPECTED_TOTAL_ROOTS=4500000
WORKERS=31
THREADS=8
BATCH_MODE=vectorized
ITERATION_MODE=concurrent_fit
CHECKPOINT_EVERY=250
MIN_MEMTOTAL_KIB=$((28 * 1024 * 1024))
MIN_DISK_KIB=$((15 * 1024 * 1024))

if [ ! -x "$PYTHON_RUN" ]; then
    echo "ERROR: lean Python environment not found: $PYTHON_RUN" >&2
    exit 3
fi
if [ ! -f "$SOURCE" ]; then
    echo "ERROR: LT2 Stage A source checkpoint not found: $SOURCE" >&2
    exit 4
fi
if [ ! -f "$ROOT/build/libspincore_solver_c.so" ]; then
    echo "ERROR: solver missing: $ROOT/build/libspincore_solver_c.so" >&2
    exit 5
fi

MEMTOTAL_KIB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
if [ -z "$MEMTOTAL_KIB" ] || [ "$MEMTOTAL_KIB" -lt "$MIN_MEMTOTAL_KIB" ]; then
    echo "ERROR: WSL exposes less than 28 GiB RAM; refusing LT2 Stage B." >&2
    exit 6
fi

DISK_KIB="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
if [ -z "$DISK_KIB" ] || [ "$DISK_KIB" -lt "$MIN_DISK_KIB" ]; then
    echo "ERROR: less than 15 GiB free on the training filesystem." >&2
    exit 7
fi

SOURCE_SHA256="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [ "$SOURCE_SHA256" != "$EXPECTED_SOURCE_SHA256" ]; then
    echo "ERROR: LT2 Stage A source checkpoint SHA256 mismatch." >&2
    echo "expected=$EXPECTED_SOURCE_SHA256" >&2
    echo "actual=$SOURCE_SHA256" >&2
    exit 8
fi

GIT_COMMIT="$(git rev-parse HEAD)"
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked source/index changes present; Stage B requires tracked code to match HEAD." >&2
    git status --short --untracked-files=no >&2
    exit 10
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/long_training_lt2_stage_b/$STAMP"
mkdir -p "$RUN_DIR"
CHECKPOINT="$RUN_DIR/checkpoint.pt"
REPORT="$RUN_DIR/report.json"
LOG="$RUN_DIR/training.log"
MEMLOG="$RUN_DIR/memory.log"
PROVENANCE="$RUN_DIR/provenance.txt"

printf 'Copying preserved LT2 Stage A checkpoint into isolated Stage B run directory...\n'
cp --reflink=auto "$SOURCE" "$CHECKPOINT"
COPY_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
if [ "$COPY_SHA256" != "$SOURCE_SHA256" ]; then
    echo "ERROR: copied checkpoint hash mismatch." >&2
    exit 9
fi

cat > "$PROVENANCE" <<EOF
schema=SPINCORE_LT2_STAGE_B_V1
created_at=$(date --iso-8601=seconds)
git_commit=$GIT_COMMIT
source_checkpoint=$SOURCE
source_sha256=$SOURCE_SHA256
stage_b_checkpoint=$CHECKPOINT
source_iteration=3000
target_iteration=$TARGET_ITERATION
additional_iterations=$ADDITIONAL_ITERATIONS
expected_total_roots=$EXPECTED_TOTAL_ROOTS
workers=$WORKERS
parent_torch_threads=$THREADS
batch_mode=$BATCH_MODE
iteration_mode=$ITERATION_MODE
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

copy_evidence_to_windows() {
    local dest="/mnt/c/Users/Rz9/Downloads"
    if [ ! -d "$dest" ]; then
        return 0
    fi
    [ -f "$REPORT" ] && cp "$REPORT" "$dest/SpinCore_LT2_STAGE_B_report.json" || true
    [ -f "$MEMLOG" ] && cp "$MEMLOG" "$dest/SpinCore_LT2_STAGE_B_memory.log" || true
    [ -f "$PROVENANCE" ] && cp "$PROVENANCE" "$dest/SpinCore_LT2_STAGE_B_provenance.txt" || true
    [ -f "$LOG" ] && cp "$LOG" "$dest/SpinCore_LT2_STAGE_B_training.log" || true
}

printf '=== SpinCore LT2 Stage B ===\n'
printf 'source=%s\n' "$SOURCE"
printf 'source_sha256=%s\n' "$SOURCE_SHA256"
printf 'git_commit=%s\n' "$GIT_COMMIT"
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'continuation=iteration 3001 -> %d (%d additional iterations / 2.7M roots)\n' "$TARGET_ITERATION" "$ADDITIONAL_ITERATIONS"
printf 'workers=%d parent_torch_threads=%d batch_mode=%s iteration_mode=%s checkpoint_every=%d\n' \
  "$WORKERS" "$THREADS" "$BATCH_MODE" "$ITERATION_MODE" "$CHECKPOINT_EVERY"
printf 'purpose=cross HU AveragePolicy saturation and review all-four-reservoir steady state\n\n'

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
      --iteration-mode "$ITERATION_MODE" \
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
    copy_evidence_to_windows
    echo "ERROR: preserved LT2 Stage A source changed during Stage B." >&2
    exit 11
fi

if [ "$STATUS" -ne 0 ]; then
    copy_evidence_to_windows
    echo "LT2_STAGE_B_FAIL status=$STATUS" >&2
    echo "run_dir=$RUN_DIR" >&2
    echo "Do not restart automatically; inspect training.log and memory.log." >&2
    exit "$STATUS"
fi

if [ ! -f "$REPORT" ]; then
    copy_evidence_to_windows
    echo "ERROR: trainer exited 0 but report.json is missing." >&2
    exit 12
fi

# Preserve evidence before post-validation too, so a validator defect can never
# hide a successfully completed expensive run from Windows Downloads.
copy_evidence_to_windows

"$PYTHON_RUN" - "$REPORT" "$TARGET_ITERATION" "$EXPECTED_TOTAL_ROOTS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
target = int(sys.argv[2])
expected_roots = int(sys.argv[3])
data = json.loads(path.read_text(encoding="utf-8"))
if int(data["config"]["iterations"]) != target:
    raise SystemExit("Stage B report target-iteration mismatch")
if int(data.get("execution_workers", -1)) != 31:
    raise SystemExit("Stage B worker-count mismatch")
if data.get("batch_mode") != "vectorized":
    raise SystemExit("Stage B batch-mode mismatch")
if data.get("iteration_mode") != "concurrent_fit":
    raise SystemExit("Stage B iteration-mode mismatch")
final_domains = (data.get("final") or {}).get("domains") or {}
if set(final_domains) != {"THREE_HANDED", "TRUE_HEADS_UP"}:
    raise SystemExit("Stage B final-domain set mismatch")
roots = sum(int(domain["roots"]) for domain in final_domains.values())
if roots != expected_roots:
    raise SystemExit(f"Stage B total-root mismatch: {roots} != {expected_roots}")
metrics = data.get("checkpoint_metrics") or []
if not metrics or not metrics[-1].get("finalized") or int(metrics[-1].get("iteration", -1)) != target:
    raise SystemExit("Stage B finalized checkpoint metric missing")
print(f"STAGE_B_REPORT_VALID target_iteration={target} total_roots={roots}")
PY

FINAL_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
copy_evidence_to_windows

printf '\nLT2_STAGE_B_PASS\n'
printf 'run_dir=%s\n' "$RUN_DIR"
printf 'report=%s\n' "$REPORT"
printf 'memory_log=%s\n' "$MEMLOG"
printf 'checkpoint=%s\n' "$CHECKPOINT"
printf 'checkpoint_sha256=%s\n' "$FINAL_SHA256"
printf 'source_unchanged=true\n'
printf 'STOP HERE. Do not extend beyond iteration 7500 until Stage B resource and learning evidence is reviewed.\n'
printf 'Windows evidence copies: SpinCore_LT2_STAGE_B_report.json, SpinCore_LT2_STAGE_B_memory.log, SpinCore_LT2_STAGE_B_provenance.txt, SpinCore_LT2_STAGE_B_training.log\n'
