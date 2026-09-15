#!/usr/bin/env bash
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
BUILD_JOBS="${SPINCORE_BUILD_JOBS:-$(nproc)}"
OUT="$ROOT/runs/worker_benchmark"
mkdir -p "$OUT"

if [ ! -x "$PYTHON_RUN" ]; then
    echo "ERROR: run tools/run_lean_functional_ryzen_pilot.sh once first." >&2
    exit 3
fi

set -e
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$BUILD_JOBS" --target spincore_solver_c >/dev/null
set +e

export PYTHONPATH="$ROOT/python"

LOGICAL="$(nproc)"
CANDIDATES=(1 8 16 24)
if [ "$LOGICAL" -ge 32 ]; then
    CANDIDATES+=(31)
elif [ "$LOGICAL" -gt 1 ]; then
    CANDIDATES+=("$((LOGICAL-1))")
fi
FILTERED=()
for W in "${CANDIDATES[@]}"; do
    if [ "$W" -gt "$LOGICAL" ]; then continue; fi
    DUP=0
    for X in "${FILTERED[@]:-}"; do [ "$X" = "$W" ] && DUP=1; done
    [ "$DUP" -eq 0 ] && FILTERED+=("$W")
done

printf '=== SpinCore Ryzen optimization benchmark ===\n'
printf 'logical_cpus=%s\n' "$LOGICAL"
printf 'phase1_worker_candidates=%s\n' "${FILTERED[*]}"
printf 'phase1_metric=iteration-2 advantage-tree seconds\n'
printf 'phase2=parent Torch thread tuning after worker selection\n'
printf 'note=failed/oversubscribed candidates are skipped, not fatal\n\n'

RESULTS="$OUT/results.tsv"
printf 'workers\ttree_seconds\twall_seconds\tstatus\n' > "$RESULTS"
SUCCESS=0

# Phase 1: isolate the expensive independent root collector. Two parent threads
# are deliberately fixed here so only worker count changes.
for W in "${FILTERED[@]}"; do
    RUN="$OUT/workers_w${W}"
    rm -rf "$RUN"
    mkdir -p "$RUN"
    printf -- '--- phase1 workers=%s ---\n' "$W"
    SPINCORE_TORCH_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    "$PYTHON_RUN" tools/run_lean_functional_training.py \
      --solver build/libspincore_solver_c.so \
      --seed 20260920 \
      --iterations 2 \
      --roots-per-iteration 300 \
      --exact-opponent-levels 0 \
      --reservoir-capacity 30000 \
      --advantage-steps 10 \
      --policy-steps 1 \
      --batch-size 128 \
      --workers "$W" \
      --checkpoint-every 2 \
      --checkpoint "$RUN/checkpoint.pt" \
      --report "$RUN/report.json" \
      > "$RUN/run.log" 2>&1
    RC=$?

    if [ "$RC" -ne 0 ] || [ ! -s "$RUN/report.json" ]; then
        printf 'workers=%s FAILED rc=%s (see %s)\n' "$W" "$RC" "$RUN/run.log"
        printf '%s\t\t\tFAILED\n' "$W" >> "$RESULTS"
        continue
    fi

    read -r TREE WALL < <("$PYTHON_RUN" - "$RUN/report.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1], 'r', encoding='utf-8'))
last=r['history'][-1]['domains']
tree=sum(float(last[d]['tree_seconds']) for d in last)
print(f"{tree:.6f} {float(r['wall_seconds']):.6f}")
PY
    )
    printf 'workers=%s tree_seconds=%s wall_seconds=%s\n' "$W" "$TREE" "$WALL"
    printf '%s\t%s\t%s\tPASS\n' "$W" "$TREE" "$WALL" >> "$RESULTS"
    SUCCESS=$((SUCCESS+1))
done

if [ "$SUCCESS" -eq 0 ]; then
    printf '\nRYZEN_OPTIMIZATION_BENCHMARK_FAILED: no worker candidate completed.\n' >&2
    exit 5
fi

SELECTED="$($PYTHON_RUN - "$RESULTS" <<'PY'
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1], encoding='utf-8'), delimiter='\t') if r['status']=='PASS']
best=min(rows, key=lambda r: float(r['tree_seconds']))
print(best['workers'])
PY
)"
printf '%s\n' "$SELECTED" > "$OUT/selected_workers.txt"

# Phase 2: after root parallelism is chosen, tune the parent neural-fit thread
# count on the actual Ryzen. We score one warmed fitted iteration, not process
# startup/final checkpoint time. This avoids "100% CPU" theatre and selects the
# thread count that actually minimizes useful iteration time.
THREAD_CANDIDATES=(1 2 4 8 16)
THREAD_RESULTS="$OUT/thread_results.tsv"
printf 'threads\tsteady_iteration_seconds\twall_seconds\tstatus\n' > "$THREAD_RESULTS"
THREAD_SUCCESS=0
printf '\nselected_workers_phase1=%s\n' "$SELECTED"
printf 'phase2_thread_candidates=%s\n' "${THREAD_CANDIDATES[*]}"

for T in "${THREAD_CANDIDATES[@]}"; do
    if [ "$T" -gt "$LOGICAL" ]; then continue; fi
    RUN="$OUT/threads_t${T}_w${SELECTED}"
    rm -rf "$RUN"
    mkdir -p "$RUN"
    printf -- '--- phase2 parent_threads=%s workers=%s ---\n' "$T" "$SELECTED"
    SPINCORE_TORCH_THREADS="$T" OMP_NUM_THREADS="$T" MKL_NUM_THREADS="$T" \
    "$PYTHON_RUN" tools/run_lean_functional_training.py \
      --solver build/libspincore_solver_c.so \
      --seed 20260921 \
      --iterations 2 \
      --roots-per-iteration 300 \
      --exact-opponent-levels 0 \
      --reservoir-capacity 30000 \
      --advantage-steps 50 \
      --policy-steps 1 \
      --batch-size 256 \
      --workers "$SELECTED" \
      --checkpoint-every 2 \
      --checkpoint "$RUN/checkpoint.pt" \
      --report "$RUN/report.json" \
      > "$RUN/run.log" 2>&1
    RC=$?

    if [ "$RC" -ne 0 ] || [ ! -s "$RUN/report.json" ]; then
        printf 'threads=%s FAILED rc=%s (see %s)\n' "$T" "$RC" "$RUN/run.log"
        printf '%s\t\t\tFAILED\n' "$T" >> "$THREAD_RESULTS"
        continue
    fi

    read -r STEADY WALL < <("$PYTHON_RUN" - "$RUN/report.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1], 'r', encoding='utf-8'))
last=r['history'][-1]['domains']
steady=sum(
    float(last[d]['tree_seconds'])
    + float(last[d]['advantage_fit_seconds'])
    + float(last[d]['sampled_policy']['seconds'])
    for d in last
)
print(f"{steady:.6f} {float(r['wall_seconds']):.6f}")
PY
    )
    printf 'threads=%s steady_iteration_seconds=%s wall_seconds=%s\n' "$T" "$STEADY" "$WALL"
    printf '%s\t%s\t%s\tPASS\n' "$T" "$STEADY" "$WALL" >> "$THREAD_RESULTS"
    THREAD_SUCCESS=$((THREAD_SUCCESS+1))
done

if [ "$THREAD_SUCCESS" -eq 0 ]; then
    printf '\nWARNING: no parent-thread candidate passed; falling back to 2 threads.\n' >&2
    SELECTED_THREADS=2
else
    SELECTED_THREADS="$($PYTHON_RUN - "$THREAD_RESULTS" <<'PY'
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1], encoding='utf-8'), delimiter='\t') if r['status']=='PASS']
best=min(rows, key=lambda r: float(r['steady_iteration_seconds']))
print(best['threads'])
PY
)"
fi
printf '%s\n' "$SELECTED_THREADS" > "$OUT/selected_torch_threads.txt"

printf '\nRYZEN_WORKER_BENCHMARK_PASS\n'
printf 'selected_workers=%s\n' "$SELECTED"
printf 'selected_parent_torch_threads=%s\n' "$SELECTED_THREADS"
printf 'successful_worker_candidates=%s/%s\n' "$SUCCESS" "${#FILTERED[@]}"
printf 'successful_thread_candidates=%s/%s\n' "$THREAD_SUCCESS" "${#THREAD_CANDIDATES[@]}"
printf 'worker_results=%s\n' "$RESULTS"
printf 'thread_results=%s\n' "$THREAD_RESULTS"
printf 'selected_worker_file=%s\n' "$OUT/selected_workers.txt"
printf 'selected_thread_file=%s\n' "$OUT/selected_torch_threads.txt"
