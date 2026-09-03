#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${SPINCORE_R754_WORK_ROOT:-$HOME/spincore_r754_dense3h_recovery}"
REPO="$WORK_ROOT/recovery-impl"
FROZEN="$WORK_ROOT/frozen-source"
PY="$WORK_ROOT/.venv/bin/python"
SOLVER="$FROZEN/build/libspincore_solver_c.so"
SOURCE_SHA="457996944f76e9f1fa0475691df978f450259641"
LEAN_COMMIT="5afc1501806290a0083edc22572f4e959e890f38"
BUNDLE="${SPINCORE_R754_ELIGIBLE_BUNDLE:-}"
EXPECTED_BUNDLE_SHA="63204dd372639b7284819b797986b32ebddc11708e5ecfc4cc59fedb519ad824"
CACHE="${SPINCORE_R754_LEAN_CACHE:-$HOME/spincore_r754_lean_screen_cache}"
OUTPUT_DIR="${SPINCORE_R754_LEAN_OUTPUT_DIR:-/mnt/c/SpinCoreAI/SpinCore/SpinCore_R754_LEAN_SCREEN}"
EVAL="$WORK_ROOT/r7_5_4_lean_crossplay_screen.py"

say(){ printf '[SpinCore lean screen] %s\n' "$*"; }
die(){ printf '[SpinCore lean screen] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -n "$BUNDLE" ]] || die "set SPINCORE_R754_ELIGIBLE_BUNDLE to the downloaded 30-cell bundle"
[[ -f "$BUNDLE" ]] || die "bundle not found: $BUNDLE"
[[ -d "$REPO/.git" ]] || die "missing recovery repo: $REPO"
[[ -d "$FROZEN/.git" || -f "$FROZEN/.git" ]] || die "missing frozen source: $FROZEN"
[[ -x "$PY" ]] || die "missing frozen Python: $PY"
[[ -s "$SOLVER" ]] || die "missing solver: $SOLVER"
[[ "$(git -C "$FROZEN" rev-parse HEAD)" == "$SOURCE_SHA" ]] || die "frozen source SHA mismatch"

actual_sha="$(sha256sum "$BUNDLE" | awk '{print $1}')"
[[ "$actual_sha" == "$EXPECTED_BUNDLE_SHA" ]] || die "bundle SHA-256 mismatch: $actual_sha"
say "30-cell bundle hash PASS"

git -C "$REPO" cat-file -e "$LEAN_COMMIT^{commit}" 2>/dev/null || \
  die "lean evaluator commit missing locally; fetch lean-quality-roadmap-20260902 first"
git -C "$REPO" show "$LEAN_COMMIT:tools/r7_5_4_lean_crossplay_screen.py" > "$EVAL"
"$PY" -m py_compile "$EVAL"
say "evaluator syntax PASS"

export PYTHONPATH="$FROZEN/python:$FROZEN/tools"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
mkdir -p "$CACHE" "$OUTPUT_DIR"

say "Running tiny mechanical smoke test (1 HU scenario-cycle hand per scenario, one seed)..."
"$PY" "$EVAL" \
  --repo-root "$FROZEN" \
  --solver "$SOLVER" \
  --bundle "$BUNDLE" \
  --cache "$CACHE" \
  --output "$OUTPUT_DIR/smoke_hu_seed173.json" \
  --domains TRUE_HEADS_UP \
  --training-seeds 1737995611 \
  --hands-per-scenario 1 \
  --torch-threads 2 \
  --progress-every 0

say "Smoke PASS. Starting decision-focused first tranche across all 30 eligible finals..."
"$PY" "$EVAL" \
  --repo-root "$FROZEN" \
  --solver "$SOLVER" \
  --bundle "$BUNDLE" \
  --cache "$CACHE" \
  --output "$OUTPUT_DIR/lean_crossplay_pf0_baseline_tranche1.json" \
  --baseline PF0_CONTROL_33_75_AI \
  --domains TRUE_HEADS_UP THREE_HANDED \
  --training-seeds 1737995611 645939859 1311335590 \
  --hands-per-scenario 20 \
  --torch-threads 2 \
  --progress-every 25

say "COMPLETE: $OUTPUT_DIR/lean_crossplay_pf0_baseline_tranche1.json"
