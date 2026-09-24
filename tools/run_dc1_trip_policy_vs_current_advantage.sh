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
[[ "${ACTUAL_SHA}" == "${EXPECTED_SHA}" ]] || { echo "ERROR: checkpoint SHA mismatch: ${ACTUAL_SHA}" >&2; exit 3; }

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c

export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
"${PY}" -m py_compile tools/audit_dc1_trip_policy_vs_current_advantage.py

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${ROOT}/runs/dc1_trip_policy_vs_current_advantage/${STAMP}"
mkdir -p "${RUN}"
REPORT="${RUN}/dc1_trip_policy_vs_current_advantage.json"

"${PY}" tools/audit_dc1_trip_policy_vs_current_advantage.py \
  --solver build/libspincore_solver_c.so \
  --checkpoint "${CHECKPOINT}" \
  --report "${REPORT}"

echo "explorer_report=$(wslpath -w "${REPORT}" 2>/dev/null || printf '%s' "${REPORT}")"
if command -v powershell.exe >/dev/null 2>&1; then
  DESKTOP_WIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n 1)"
  if [[ -n "${DESKTOP_WIN}" ]]; then
    DESKTOP_WSL="$(wslpath -u "${DESKTOP_WIN}" 2>/dev/null || true)"
    if [[ -n "${DESKTOP_WSL}" && -d "${DESKTOP_WSL}" ]]; then
      cp -f "${REPORT}" "${DESKTOP_WSL}/SpinCore_DC1_trip_policy_vs_current_advantage.json"
      echo "desktop_report=${DESKTOP_WIN}\\SpinCore_DC1_trip_policy_vs_current_advantage.json"
    fi
  fi
fi
echo "DC1_TRIPS_POLICY_VS_CURRENT_ADVANTAGE_RUN_COMPLETE"
