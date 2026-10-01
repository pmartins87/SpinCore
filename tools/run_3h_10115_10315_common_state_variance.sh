#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: VALID-025,VALID-030,VALID-031,VALID-032,SAFE-001,PERF-010,PERF-013,RNG-001,RNG-002,RNG-003,ART-001,SRC-003

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
SOLVER="${ROOT}/build/libspincore_solver_c.so"
SRC="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CP="${SRC}/checkpoint.pt"
HU="${SRC}/hu_ensemble_state.pt"
BASE="${ROOT}/runs/variance_diag_10115_10315/spincore_hybrid_10105.pt"
OLD_SPEC="${ROOT}/runs/3h_semantic_stratified_specialist_moe_10115/semantic_stratified_specialist_moe_10115.pt"
NEW_SPEC="${ROOT}/runs/3h_semantic_postlong_10315/semantic_stratified_specialist_10315.pt"
OLD_ADV="${ROOT}/runs/3h_semantic_research_10105_10115/semantic_ensemble_final_10115.pt"
NEW_ADV="${ROOT}/runs/3h_semantic_long_10115_10315/semantic_long_ensemble_final_10315.pt"
RUN="${ROOT}/runs/variance_diag_10115_10315"
REPORT="${RUN}/common_state_variance_10115_10315.json"

mkdir -p "${RUN}"
for f in "${PY}" "${SOLVER}" "${CP}" "${HU}" "${OLD_SPEC}" "${NEW_SPEC}" "${OLD_ADV}" "${NEW_ADV}"; do
  [[ -e "${f}" ]] || { echo "ERROR missing prerequisite: ${f}" >&2; exit 2; }
done

export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${PY}" -m py_compile tools/audit_3h_10115_10315_common_state_variance.py

if [[ ! -f "${BASE}" ]]; then
  "${PY}" tools/export_spincore_hybrid_inference.py     --checkpoint "${CP}" --ensemble "${HU}" --out "${BASE}"
fi

"${PY}" tools/audit_3h_10115_10315_common_state_variance.py   --solver "${SOLVER}"   --base-bundle "${BASE}"   --old-specialist "${OLD_SPEC}"   --new-specialist "${NEW_SPEC}"   --old-advantage "${OLD_ADV}"   --new-advantage "${NEW_ADV}"   --episodes 20000   --threads 8   --report "${REPORT}"

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${REPORT}" "${DWSL}/SpinCore_10115_10315_common_state_variance.json"
    echo "desktop_report=${DWIN}\\SpinCore_10115_10315_common_state_variance.json"
  fi
fi

echo "VARIANCE_DIAG_RUN_COMPLETE"
echo "STOP HERE. Send SpinCore_10115_10315_common_state_variance.json to ChatGPT."
