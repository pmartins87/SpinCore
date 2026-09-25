#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
SOURCE_RUN="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE_RUN}/checkpoint.pt"
ENSEMBLE="${SOURCE_RUN}/hu_ensemble_state.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_ENS="8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"

[[ -x "${PY}" ]] || { echo "ERROR: missing Python venv: ${PY}" >&2; exit 2; }
[[ -f "${CHECKPOINT}" && -f "${ENSEMBLE}" ]] || { echo "ERROR: missing source artifacts" >&2; exit 2; }
[[ "$(sha256sum "${CHECKPOINT}"|awk '{print $1}')" == "${EXPECTED_CP}" ]] || { echo "ERROR: checkpoint SHA mismatch" >&2; exit 3; }
[[ "$(sha256sum "${ENSEMBLE}"|awk '{print $1}')" == "${EXPECTED_ENS}" ]] || { echo "ERROR: ensemble SHA mismatch" >&2; exit 3; }

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c

export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
"${PY}" -m py_compile tools/export_3h_current_advantage_probe.py tools/evaluate_dc1_3h_average_vs_current_advantage.py

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${ROOT}/runs/dc1_3h_average_vs_current_advantage/${STAMP}"
mkdir -p "${RUN}"
BUNDLE="${RUN}/spincore_hybrid_10105.pt"
PROBE="${RUN}/three_handed_current_advantage.pt"
REPORT="${RUN}/dc1_3h_average_vs_current_advantage.json"

"${PY}" tools/export_spincore_hybrid_inference.py --checkpoint "${CHECKPOINT}" --ensemble "${ENSEMBLE}" --out "${BUNDLE}"
"${PY}" tools/export_3h_current_advantage_probe.py --checkpoint "${CHECKPOINT}" --out "${PROBE}"
"${PY}" tools/evaluate_dc1_3h_average_vs_current_advantage.py \
  --solver build/libspincore_solver_c.so \
  --spin-bundle "${BUNDLE}" \
  --current-3h-probe "${PROBE}" \
  --scenarios 200 \
  --workers 8 \
  --seed 20260923 \
  --report "${REPORT}"

echo "explorer_report=$(wslpath -w "${REPORT}" 2>/dev/null || printf '%s' "${REPORT}")"
if command -v powershell.exe >/dev/null 2>&1; then
  DESKTOP_WIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n 1)"
  if [[ -n "${DESKTOP_WIN}" ]]; then
    DESKTOP_WSL="$(wslpath -u "${DESKTOP_WIN}" 2>/dev/null || true)"
    if [[ -n "${DESKTOP_WSL}" && -d "${DESKTOP_WSL}" ]]; then
      cp -f "${REPORT}" "${DESKTOP_WSL}/SpinCore_DC1_3H_average_vs_current_advantage.json"
      echo "desktop_report=${DESKTOP_WIN}\\SpinCore_DC1_3H_average_vs_current_advantage.json"
    fi
  fi
fi

echo "DC1_3H_AVERAGE_VS_CURRENT_ADVANTAGE_RUN_COMPLETE"
