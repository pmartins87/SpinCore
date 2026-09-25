#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
CHECKPOINT="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104/checkpoint.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"

[[ -x "${PY}" ]] || { echo "ERROR: missing Python venv" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing checkpoint" >&2; exit 2; }
[[ "$(sha256sum "${CHECKPOINT}"|awk '{print $1}')" == "${EXPECTED_CP}" ]] || { echo "ERROR: checkpoint SHA mismatch" >&2; exit 3; }

LATEST_RUN="$(find "${ROOT}/runs/3h_advantage_budget_stability" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
[[ -n "${LATEST_RUN}" ]] || { echo "ERROR: no 3H budget-stability run found" >&2; exit 4; }
PROBE="${LATEST_RUN}/3h_advantage_budget_probe.pt"
[[ -f "${PROBE}" ]] || { echo "ERROR: latest run has no probe artifact: ${PROBE}" >&2; exit 4; }

export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
"${PY}" -m py_compile tools/audit_3h_advantage_budget_clean_holdout_10105.py

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${ROOT}/runs/3h_advantage_clean_holdout/${STAMP}"
mkdir -p "${RUN}"
REPORT="${RUN}/3h_advantage_clean_holdout.json"

"${PY}" tools/audit_3h_advantage_budget_clean_holdout_10105.py \
  --checkpoint "${CHECKPOINT}" \
  --probe "${PROBE}" \
  --report "${REPORT}" \
  --holdout-size 50000

echo "source_probe=$(wslpath -w "${PROBE}" 2>/dev/null || printf '%s' "${PROBE}")"
echo "explorer_report=$(wslpath -w "${REPORT}" 2>/dev/null || printf '%s' "${REPORT}")"

if command -v powershell.exe >/dev/null 2>&1; then
  DESKTOP_WIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n 1)"
  if [[ -n "${DESKTOP_WIN}" ]]; then
    DESKTOP_WSL="$(wslpath -u "${DESKTOP_WIN}" 2>/dev/null || true)"
    if [[ -n "${DESKTOP_WSL}" && -d "${DESKTOP_WSL}" ]]; then
      cp -f "${REPORT}" "${DESKTOP_WSL}/SpinCore_3H_advantage_clean_holdout.json"
      echo "desktop_report=${DESKTOP_WIN}\\SpinCore_3H_advantage_clean_holdout.json"
    fi
  fi
fi

echo "3H_ADVANTAGE_CLEAN_HOLDOUT_RUN_COMPLETE"
