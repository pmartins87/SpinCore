#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
BUILD_JOBS="${SPINCORE_BUILD_JOBS:-$(nproc)}"
THREADS="${SPINCORE_TORCH_THREADS:-2}"
OUT="$ROOT/runs/worker_benchmark"
mkdir -p "$OUT"

if [ ! -x "$PYTHON_RUN" ]; then
    echo "ERROR: run tools/run_lean_functional_ryzen_pilot.sh once first." >&2
    exit 3
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$BUILD_JOBS" --target spincore_solver_c >/dev/null

export PYTHONPATH="$ROOT/python"
export SPINCORE_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

LOGICAL="$(nproc)"
CANDIDATES=(1 8 16 24)
if [ "$LOGICAL" -ge 31 ]; then CANDIDATES+=(31); else CANDIDATES+=("$((LOGICAL>1 ? LOGICAL-1 : 1))"); fi

printf '=== SpinCore Ryzen worker benchmark ===\n'
printf 'logical_cpus=%s parent_torch_threads=%s\n' "$LOGICAL" "$THREADS"
printf 'candidates=%s\n' "${CANDIDATES[*]}"
printf 'metric=iteration-2 advantage-tree seconds (same 300-root profile)\n\n'

RESULTS="$OUT/results.tsv"
printf 'workers\ttree_seconds\twall_seconds\n' > "$RESULTS"

for W in "${CANDIDATES[@]}"; do
    RUN="$OUT/w${W}"
    rm -rf "$RUN"
    mkdir -p "$RUN"
    printf -- '--- workers=%s ---\n' "$W"
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

    read -r TREE WALL < <("$PYTHON_RUN" - "$RUN/report.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1], 'r', encoding='utf-8'))
last=r['history'][-1]['domains']
tree=sum(float(last[d]['tree_seconds']) for d in last)
print(f"{tree:.6f} {float(r['wall_seconds']):.6f}")
PY
    )
    printf 'workers=%s tree_seconds=%s wall_seconds=%s\n' "$W" "$TREE" "$WALL"
    printf '%s\t%s\t%s\n' "$W" "$TREE" "$WALL" >> "$RESULTS"
done

SELECTED="$($PYTHON_RUN - "$RESULTS" <<'PY'
import csv, sys
rows=list(csv.DictReader(open(sys.argv[1], encoding='utf-8'), delimiter='\t'))
best=min(rows, key=lambda r: float(r['tree_seconds']))
print(best['workers'])
PY
)"
printf '%s\n' "$SELECTED" > "$OUT/selected_workers.txt"

printf '\nRYZEN_WORKER_BENCHMARK_PASS\n'
printf 'selected_workers=%s\n' "$SELECTED"
printf 'results=%s\n' "$RESULTS"
printf 'selected_file=%s\n' "$OUT/selected_workers.txt"
