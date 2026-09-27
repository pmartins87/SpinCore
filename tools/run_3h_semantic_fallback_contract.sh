#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
SEM_RUN="${ROOT}/runs/3h_semantic_research_10105_10115"

INIT_RUN="$(find "${ROOT}/runs/3h_v1_semantic_shadow" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
INIT="${INIT_RUN}/3h_v1_semantic_shadow_models.pt"
E107="${SEM_RUN}/semantic_ensemble_10107.pt"
E110="${SEM_RUN}/semantic_ensemble_10110.pt"
E115="${SEM_RUN}/semantic_ensemble_final_10115.pt"

for f in "${PY}" "${INIT}" "${E107}" "${E110}" "${E115}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing artifact: ${f}" >&2; exit 2; }
done

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

"${PY}" -m py_compile tools/audit_3h_semantic_fallback_contract.py

RUN="${ROOT}/runs/3h_semantic_fallback_contract"
mkdir -p "${RUN}"
REPORT="${RUN}/3h_semantic_fallback_contract.json"

"${PY}" tools/audit_3h_semantic_fallback_contract.py \
  --solver build/libspincore_solver_c.so \
  --initial-ensemble "${INIT}" \
  --ensemble-10107 "${E107}" \
  --ensemble-10110 "${E110}" \
  --ensemble-10115 "${E115}" \
  --report "${REPORT}" \
  --threads 8

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${REPORT}" "${DWSL}/SpinCore_3H_semantic_fallback_contract.json"
    echo "desktop_report=${DWIN}\\SpinCore_3H_semantic_fallback_contract.json"
  fi
fi

echo "SEMANTIC_FALLBACK_CONTRACT_AUDIT_RUN_COMPLETE"
