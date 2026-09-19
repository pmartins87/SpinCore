#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_RUN="${SPINCORE_LEAN_VENV:-$ROOT/.venv_lean}/bin/python"
[ -x "$PYTHON_RUN" ] || { echo "ERROR: missing python: $PYTHON_RUN" >&2; exit 3; }

BASE="$ROOT/runs/lt2_jammer_fai_structural_infoset_confirmation"
SOURCE="$(
  find "$BASE" -mindepth 2 -maxdepth 2 -type f     -name 'jammer_fai_structural_infoset_confirmation.json'     -printf '%T@ %p\n' 2>/dev/null   | sort -nr | head -1 | cut -d' ' -f2-
)"

[ -n "$SOURCE" ] && [ -f "$SOURCE" ] || {
  echo "ERROR: structural infoset confirmation report not found under $BASE" >&2
  exit 4
}

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/lt2_jammer_fai_raw_margin_decomposition/$STAMP"
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/jammer_fai_raw_margin_decomposition.json"

export PYTHONPATH="$ROOT/python:$ROOT/tools"

printf "=== SpinCore LT2 Jammer FAI raw-margin decomposition ===\n"
printf "mode=READ ONLY; no solver; no roots; no optimizer; no memory writes\n"
printf "source=%s\n\n" "$SOURCE"

"$PYTHON_RUN" tools/analyze_lt2_jammer_fai_raw_margin_decomposition.py   --input "$SOURCE"   --report "$REPORT"

"$PYTHON_RUN" - "$REPORT" <<'PY'
import json, math, sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
assert d["schema"] == "SPINCORE_LT2_JAMMER_FAI_RAW_MARGIN_DECOMPOSITION_V1"
assert d["method"]["read_only"] is True
assert d["method"]["new_training_roots"] == 0
assert d["method"]["optimizer_steps"] == 0
assert d["method"]["training_memory_writes"] == 0
assert d["method"]["future_holdout_seeds_touched"] is False
assert d["summary"]["n"] == 384

primary = d["summary"]["groups"]["B_MORE_FOLD"]
assert primary["n"] == 192
full = primary["full_b_minus_a_chips"]["seed_cluster_ci"]
assert math.isfinite(float(full["mean"]))
assert abs(float(full["mean"]) - (-24.5492074957383)) < 1e-4

print("LT2_JAMMER_FAI_RAW_MARGIN_DECOMPOSITION_POSTVALIDATION_PASS")
PY

DEST="/mnt/c/Users/Rz9/Downloads"
if [ -d "$DEST" ]; then
  cp "$REPORT" "$DEST/SpinCore_LT2_jammer_fai_raw_margin_decomposition.json"
fi

printf "\nLT2_JAMMER_FAI_RAW_MARGIN_DECOMPOSITION_PASS\n"
printf "report=%s\n" "$REPORT"
printf "STOP HERE. Send SpinCore_LT2_jammer_fai_raw_margin_decomposition.json to ChatGPT.\n"
