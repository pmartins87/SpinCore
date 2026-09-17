#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${SPINCORE_LT2B_COMPLETED_RUN:-$ROOT/runs/long_training_lt2_stage_b/20260917_004911}"
REPORT="$RUN_DIR/report.json"
MEMLOG="$RUN_DIR/memory.log"
PROVENANCE="$RUN_DIR/provenance.txt"
LOG="$RUN_DIR/training.log"
CHECKPOINT="$RUN_DIR/checkpoint.pt"
SOURCE="/home/rz9/spincore_lean_functional/runs/long_training_lt2/20260916_015559/checkpoint.pt"
EXPECTED_SOURCE_SHA256="e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c"
TARGET_ITERATION=7500
EXPECTED_TOTAL_ROOTS=4500000

for f in "$REPORT" "$MEMLOG" "$PROVENANCE" "$LOG" "$CHECKPOINT" "$SOURCE"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file missing: $f" >&2
        exit 3
    fi
done

SOURCE_SHA256="$(sha256sum "$SOURCE" | awk '{print $1}')"
if [ "$SOURCE_SHA256" != "$EXPECTED_SOURCE_SHA256" ]; then
    echo "ERROR: preserved LT2 Stage A source hash changed." >&2
    echo "expected=$EXPECTED_SOURCE_SHA256" >&2
    echo "actual=$SOURCE_SHA256" >&2
    exit 4
fi

CHECKPOINT_SHA256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
if [ ! -x "$PYTHON_RUN" ]; then
    echo "ERROR: lean Python environment not found: $PYTHON_RUN" >&2
    exit 5
fi

"$PYTHON_RUN" - "$REPORT" "$MEMLOG" "$TARGET_ITERATION" "$EXPECTED_TOTAL_ROOTS" <<'PY'
import json
import re
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
memlog_path = Path(sys.argv[2])
target = int(sys.argv[3])
expected_roots = int(sys.argv[4])

data = json.loads(report_path.read_text(encoding="utf-8"))
assert data.get("schema") == "SPINCORE_LEAN_FUNCTIONAL_REPORT_V1", data.get("schema")
assert int(data["config"]["iterations"]) == target
assert int(data.get("execution_workers", -1)) == 31
assert int(data.get("torch_threads", -1)) == 8
assert data.get("batch_mode") == "vectorized"
assert data.get("iteration_mode") == "concurrent_fit"

final_domains = (data.get("final") or {}).get("domains") or {}
assert set(final_domains) == {"THREE_HANDED", "TRUE_HEADS_UP"}, set(final_domains)
roots = sum(int(v["roots"]) for v in final_domains.values())
assert roots == expected_roots, (roots, expected_roots)
assert int(final_domains["THREE_HANDED"]["roots"]) == 2452500
assert int(final_domains["TRUE_HEADS_UP"]["roots"]) == 2047500
assert int(final_domains["TRUE_HEADS_UP"]["strategy_seen"]) >= 2_000_000
assert int(final_domains["THREE_HANDED"]["strategy_seen"]) >= 2_000_000

metrics = data.get("checkpoint_metrics") or []
assert metrics, "missing checkpoint metrics"
last = metrics[-1]
assert bool(last.get("finalized"))
assert int(last.get("iteration", -1)) == target
assert int(last.get("size_bytes", -1)) > 2_000_000_000

min_mem = None
max_swap = 0
rows = 0
for line in memlog_path.read_text(encoding="utf-8", errors="replace").splitlines():
    pairs = dict(re.findall(r"([a-z_]+)=([^ ]+)", line))
    if "mem_available_kib" in pairs:
        value = int(pairs["mem_available_kib"])
        min_mem = value if min_mem is None else min(min_mem, value)
        rows += 1
    if "swap_used_kib" in pairs:
        max_swap = max(max_swap, int(pairs["swap_used_kib"]))
assert rows > 0, "memory log has no parseable rows"

print("LT2_STAGE_B_POSTVALIDATION_PASS")
print(f"target_iteration={target}")
print(f"total_roots={roots}")
print(f"wall_seconds={float(data.get('wall_seconds', -1)):.3f}")
print(f"checkpoint_size_gib={float(last.get('size_gib', -1)):.6f}")
print(f"checkpoint_save_seconds={float(last.get('seconds', -1)):.3f}")
print(f"three_handed_strategy_seen={int(final_domains['THREE_HANDED']['strategy_seen'])}")
print(f"heads_up_strategy_seen={int(final_domains['TRUE_HEADS_UP']['strategy_seen'])}")
print(f"min_mem_available_gib={min_mem / (1024**2):.3f}")
print(f"max_swap_used_gib={max_swap / (1024**2):.3f}")
PY

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
    cp "$REPORT" "$DEST/SpinCore_LT2_STAGE_B_report.json"
    cp "$MEMLOG" "$DEST/SpinCore_LT2_STAGE_B_memory.log"
    cp "$PROVENANCE" "$DEST/SpinCore_LT2_STAGE_B_provenance.txt"
    cp "$LOG" "$DEST/SpinCore_LT2_STAGE_B_training.log"
fi

printf 'checkpoint=%s\n' "$CHECKPOINT"
printf 'checkpoint_sha256=%s\n' "$CHECKPOINT_SHA256"
printf 'source_unchanged=true\n'
printf 'Evidence copied to Windows Downloads when available.\n'
printf 'STOP HERE. Do not continue training beyond iteration 7500 until Stage B learning/resource review.\n'
