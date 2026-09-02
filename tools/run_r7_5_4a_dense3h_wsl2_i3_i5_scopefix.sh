#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${SPINCORE_R754_WORK_ROOT:-$HOME/spincore_r754_dense3h_recovery}"
REPO="$WORK_ROOT/recovery-impl"
SOURCE_COMMIT="02e1261c4ec6c8101186ff81711b3a63c7360d13"
SOURCE_PATH="tools/run_r7_5_4a_dense3h_wsl2_i3_i5.sh"
PATCHED="$WORK_ROOT/run_r7_5_4a_dense3h_wsl2_i3_i5_scopefixed.sh"

[[ -d "$REPO/.git" ]] || { echo "ERROR: missing $REPO" >&2; exit 1; }

python3 - "$REPO" "$SOURCE_COMMIT" "$SOURCE_PATH" "$PATCHED" <<'PY'
from pathlib import Path
import subprocess
import sys

repo, commit, source_path, out_path = sys.argv[1:]
text = subprocess.check_output(
    ["git", "-C", repo, "show", f"{commit}:{source_path}"],
    text=True,
)

replacements = [
    (
        'quarantine(){ d="$1"; label="$2"; cp=0; rp=0;',
        'quarantine(){ local d="$1" label="$2" cp=0 rp=0 q;',
    ),
    (
        '  seed="$1"; iter="$2"; mode="$3"; stage="$4"; expected_in="$5"; expected_out="${6:-}"; d="$STATE/$seed/$([[ "$mode" == fit ]] && echo i$iter || echo i${iter}c${stage})"',
        '  local seed="$1" iter="$2" mode="$3" stage="$4" expected_in="$5" expected_out="${6:-}" d actual\n  d="$STATE/$seed/$([[ "$mode" == fit ]] && echo i$iter || echo i${iter}c${stage})"',
    ),
    (
        'progress(){ iter="$1"; stage="$2"; mode="$3"; ITER="$iter" STAGE="$stage" MODE="$mode" STATE_ROOT="$STATE" "$PY" - <<\'PYP\' > "$EXPORT_BASE/PROGRESS.json"',
        'progress(){ local iter="$1" stage="$2" mode="$3"; ITER="$iter" STAGE="$stage" MODE="$mode" STATE_ROOT="$STATE" "$PY" - <<\'PYP\' > "$EXPORT_BASE/PROGRESS.json"',
    ),
    (
        'worker(){ seed="$1"; iter="$2"; mode="$3"; input="$4"; out="$5"; stage="$6"; label="i${iter}$([[ "$mode" == fit ]] && echo fit || echo c$stage)";',
        'worker(){ local seed="$1" iter="$2" mode="$3" input="$4" out="$5" stage="$6" label; label="i${iter}$([[ "$mode" == fit ]] && echo fit || echo c$stage)";',
    ),
]

for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"scopefix refused: expected exactly one occurrence, got {count}: {old[:80]!r}")
    text = text.replace(old, new, 1)

# Guard against the exact bug that caused i3c1 to become i2c32.
if 'validate(){\n  local seed=' not in text:
    raise SystemExit('scopefix refused: validate parameters are not local')
if 'progress(){ local iter=' not in text:
    raise SystemExit('scopefix refused: progress parameters are not local')

Path(out_path).write_text(text, encoding="utf-8")
print(f"scopefixed driver written: {out_path}")
PY

chmod +x "$PATCHED"
bash -n "$PATCHED"
echo "scopefixed driver syntax PASS"
exec bash "$PATCHED"
