#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(git -C "$ROOT" rev-parse --show-toplevel)"
PS1="$ROOT/tools/inspect_lt2_openholdem_host.ps1"
PS1_WIN="$(wslpath -w "$PS1")"

if [ -n "${OPENHOLDEM_EXE_WIN:-}" ]; then
  powershell.exe -NoProfile -ExecutionPolicy Bypass \
    -File "$PS1_WIN" \
    -OpenHoldemExe "$OPENHOLDEM_EXE_WIN"
else
  powershell.exe -NoProfile -ExecutionPolicy Bypass \
    -File "$PS1_WIN"
fi
