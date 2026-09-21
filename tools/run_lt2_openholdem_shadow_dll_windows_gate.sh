#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "\${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE="/mnt/c/Users/Rz9/Downloads/SpinCore_LT2_cpp_deployment_8100.bin"
EXPECTED_SHA="2b79ab7ff746a9c1c3dd73dbc0a1d6884a471813cf34b9cb126790c4c4cbb123"
STAGE="/mnt/c/SpinCore_LT2_ShadowBuild"

[ -f "$BUNDLE" ] || {
  echo "ERROR: missing frozen native bundle: $BUNDLE" >&2
  exit 3
}
ACTUAL_SHA="$(sha256sum "$BUNDLE" | awk '{print $1}')"
[ "$ACTUAL_SHA" = "$EXPECTED_SHA" ] || {
  echo "ERROR: bundle SHA256 mismatch" >&2
  echo "expected=$EXPECTED_SHA" >&2
  echo "actual=$ACTUAL_SHA" >&2
  exit 4
}
command -v powershell.exe >/dev/null 2>&1 || {
  echo "ERROR: powershell.exe not available from WSL" >&2
  exit 5
}

rm -rf "$STAGE"
mkdir -p "$STAGE" "$STAGE/artifacts"

tar \
  --exclude='./.git' \
  --exclude='./build' \
  --exclude='./build_*' \
  --exclude='./runs' \
  --exclude='./.venv_lean' \
  --exclude='./__pycache__' \
  -C "$ROOT" -cf - . | tar -C "$STAGE" -xf -

cp "$BUNDLE" "$STAGE/artifacts/SpinCore_LT2_cpp_deployment_8100.bin"

PS1_WIN="$(wslpath -w "$STAGE/tools/build_and_test_lt2_openholdem_shadow_dll.ps1")"
STAGE_WIN="$(wslpath -w "$STAGE")"

echo "=== SpinCore LT2 Windows OpenHoldem shadow DLL gate ==="
echo "staged_source=$STAGE_WIN"
echo "mode=SHADOW_NO_TABLE_ACTION"
echo "No OpenHoldem table action will be executed."

powershell.exe -NoProfile -ExecutionPolicy Bypass \
  -File "$PS1_WIN" \
  -SourceDir "$STAGE_WIN"
