#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
SOURCE="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE}/checkpoint.pt"
HU_ENSEMBLE="${SOURCE}/hu_ensemble_state.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_ENS="8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"

[[ -x "${PY}" ]] || { echo "ERROR: missing Python venv" >&2; exit 2; }
[[ -f "${CHECKPOINT}" && -f "${HU_ENSEMBLE}" ]] || { echo "ERROR: missing 10105 source artifacts" >&2; exit 2; }
[[ "$(sha256sum "${CHECKPOINT}"|awk '{print $1}')" == "${EXPECTED_CP}" ]] || { echo "ERROR: checkpoint SHA mismatch" >&2; exit 3; }
[[ "$(sha256sum "${HU_ENSEMBLE}"|awk '{print $1}')" == "${EXPECTED_ENS}" ]] || { echo "ERROR: HU ensemble SHA mismatch" >&2; exit 3; }

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"

"${PY}" -m py_compile \
  tools/build_3h_advantage_budget_probe_10105.py \
  tools/evaluate_3h_advantage_budget_stability_10105.py

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${ROOT}/runs/3h_advantage_budget_stability/${STAMP}"
mkdir -p "${RUN}"
BUNDLE="${RUN}/spincore_hybrid_10105.pt"
PROBE="${RUN}/3h_advantage_budget_probe.pt"
REPORT="${RUN}/3h_advantage_budget_stability.json"

"${PY}" tools/export_spincore_hybrid_inference.py \
  --checkpoint "${CHECKPOINT}" \
  --ensemble "${HU_ENSEMBLE}" \
  --out "${BUNDLE}"

"${PY}" tools/build_3h_advantage_budget_probe_10105.py \
  --solver build/libspincore_solver_c.so \
  --checkpoint "${CHECKPOINT}" \
  --out "${PROBE}" \
  --threads 8

"${PY}" tools/evaluate_3h_advantage_budget_stability_10105.py \
  --solver build/libspincore_solver_c.so \
  --spin-bundle "${BUNDLE}" \
  --probe "${PROBE}" \
  --scenarios 200 \
  --workers 4 \
  --seed 20260923 \
  --report "${REPORT}"

echo "explorer_report=$(wslpath -w "${REPORT}" 2>/dev/null || printf '%s' "${REPORT}")"
if command -v powershell.exe >/dev/null 2>&1; then
  DESKTOP_WIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n 1)"
  if [[ -n "${DESKTOP_WIN}" ]]; then
    DESKTOP_WSL="$(wslpath -u "${DESKTOP_WIN}" 2>/dev/null || true)"
    if [[ -n "${DESKTOP_WSL}" && -d "${DESKTOP_WSL}" ]]; then
      cp -f "${REPORT}" "${DESKTOP_WSL}/SpinCore_3H_advantage_budget_stability.json"
      echo "desktop_report=${DESKTOP_WIN}\\SpinCore_3H_advantage_budget_stability.json"
    fi
  fi
fi

echo "3H_ADVANTAGE_BUDGET_STABILITY_RUN_COMPLETE"
