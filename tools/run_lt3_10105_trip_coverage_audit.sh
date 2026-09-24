#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"

PY="${ROOT}/.venv_lean/bin/python"
CHECKPOINT="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104/checkpoint.pt"
EXPECTED_SHA="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"

[[ -x "${PY}" ]] || { echo "ERROR: missing Python venv: ${PY}" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing checkpoint: ${CHECKPOINT}" >&2; exit 2; }
ACTUAL_SHA="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
[[ "${ACTUAL_SHA}" == "${EXPECTED_SHA}" ]] || {
  echo "ERROR: checkpoint SHA mismatch: ${ACTUAL_SHA}" >&2
  exit 3
}

export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
"${PY}" -m py_compile tools/audit_lt3_10105_trip_coverage.py

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${ROOT}/runs/lt3_10105_trip_coverage/${STAMP}"
mkdir -p "${RUN}"
REPORT="${RUN}/trip_coverage_10105.json"

"${PY}" tools/audit_lt3_10105_trip_coverage.py \
  --checkpoint "${CHECKPOINT}" \
  --report "${REPORT}"

EXPLORER_REPORT="$(wslpath -w "${REPORT}" 2>/dev/null || printf '%s' "${REPORT}")"
echo "explorer_report=${EXPLORER_REPORT}"

if command -v powershell.exe >/dev/null 2>&1; then
  DESKTOP_WIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n 1)"
  if [[ -n "${DESKTOP_WIN}" ]]; then
    DESKTOP_WSL="$(wslpath -u "${DESKTOP_WIN}" 2>/dev/null || true)"
    if [[ -n "${DESKTOP_WSL}" && -d "${DESKTOP_WSL}" ]]; then
      cp -f "${REPORT}" "${DESKTOP_WSL}/SpinCore_LT3_10105_trip_coverage.json"
      echo "desktop_report=${DESKTOP_WIN}\\SpinCore_LT3_10105_trip_coverage.json"
    fi
  fi
fi

echo "LT3_10105_TRIP_COVERAGE_RUN_COMPLETE"
