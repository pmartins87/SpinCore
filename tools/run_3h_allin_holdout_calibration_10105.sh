#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
SOURCE="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE}/checkpoint.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"

[[ -x "${PY}" ]] || { echo "ERROR: missing Python venv" >&2; exit 2; }
[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: missing checkpoint" >&2; exit 2; }
[[ "$(sha256sum "${CHECKPOINT}"|awk '{print $1}')" == "${EXPECTED_CP}" ]] || { echo "ERROR: checkpoint SHA mismatch" >&2; exit 3; }

PROBE_RUN="$(find "${ROOT}/runs/3h_advantage_controlled_budget_curve" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
GAP_RUN="$(find "${ROOT}/runs/3h_high_card_model_target_gap" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
[[ -n "${PROBE_RUN}" && -n "${GAP_RUN}" ]] || { echo "ERROR: prerequisite diagnostic run missing" >&2; exit 4; }
PROBE="${PROBE_RUN}/3h_advantage_controlled_split_probe.pt"
GAP="${GAP_RUN}/3h_high_card_model_target_gap.json"
[[ -f "${PROBE}" && -f "${GAP}" ]] || { echo "ERROR: prerequisite diagnostic artifact missing" >&2; exit 4; }

export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
"${PY}" -m py_compile tools/audit_3h_allin_holdout_calibration_10105.py

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${ROOT}/runs/3h_allin_holdout_calibration/${STAMP}"
mkdir -p "${RUN}"
REPORT="${RUN}/3h_allin_holdout_calibration.json"

"${PY}" tools/audit_3h_allin_holdout_calibration_10105.py \
  --checkpoint "${CHECKPOINT}" \
  --probe "${PROBE}" \
  --gap-report "${GAP}" \
  --report "${REPORT}" \
  --threads 8 \
  --batch-size 4096

echo "explorer_report=$(wslpath -w "${REPORT}" 2>/dev/null || printf '%s' "${REPORT}")"
if command -v powershell.exe >/dev/null 2>&1; then
  DESKTOP_WIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n 1)"
  if [[ -n "${DESKTOP_WIN}" ]]; then
    DESKTOP_WSL="$(wslpath -u "${DESKTOP_WIN}" 2>/dev/null || true)"
    if [[ -n "${DESKTOP_WSL}" && -d "${DESKTOP_WSL}" ]]; then
      cp -f "${REPORT}" "${DESKTOP_WSL}/SpinCore_3H_allin_holdout_calibration.json"
      echo "desktop_report=${DESKTOP_WIN}\\SpinCore_3H_allin_holdout_calibration.json"
    fi
  fi
fi

echo "3H_ALLIN_HOLDOUT_CALIBRATION_RUN_COMPLETE"
