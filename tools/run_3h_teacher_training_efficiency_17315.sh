#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: DC-002,SAFE-001,SAFE-011,VALID-025,VALID-031,VALID-040,RNG-001,RNG-002,RNG-003

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_ROOT="${SPINCORE_ACTIVE_ROOT:-${HOME}/spincore_lean_functional}"
cd "${CODE_ROOT}"

PY="${ARTIFACT_ROOT}/.venv_lean/bin/python"
export PYTHONPATH="${CODE_ROOT}/python:${CODE_ROOT}/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SPINCORE_TORCH_THREADS=1

SOLVER="${ARTIFACT_ROOT}/build/libspincore_solver_c.so"
T10315="${ARTIFACT_ROOT}/runs/3h_semantic_long_10115_10315/semantic_long_ensemble_final_10315.pt"
T14815="${ARTIFACT_ROOT}/runs/3h_semantic_long_10315_15315/semantic_long_ensemble_14815.pt"
T15315="${ARTIFACT_ROOT}/runs/3h_semantic_long_10315_15315/semantic_long_ensemble_final_15315.pt"
T17315="${ARTIFACT_ROOT}/runs/3h_semantic_long_15315_17315/semantic_long_ensemble_final_17315.pt"
SANITY="${ARTIFACT_ROOT}/runs/teacher_sanity_17315/teacher_sanity_17315.json"

RAW_OUT="${CODE_ROOT}/runs/teacher_raw_advantage_17315"
TREND_OUT="${CODE_ROOT}/runs/teacher_flagged_trend_17315"
RAW_REPORT="${RAW_OUT}/teacher_raw_advantage_17315.json"
TREND_REPORT="${TREND_OUT}/teacher_flagged_checkpoint_trend_17315.json"
mkdir -p "${RAW_OUT}" "${TREND_OUT}"

for path in "${PY}" "${SOLVER}" "${T10315}" "${T14815}" "${T15315}" "${T17315}" "${SANITY}"; do
  [[ -e "${path}" ]] || { echo "ERROR: missing prerequisite: ${path}" >&2; exit 2; }
done

"${PY}" -m py_compile   tools/diagnose_3h_teacher_raw_advantages_17315.py   tools/diagnose_3h_teacher_flagged_checkpoint_trend.py

echo "SPINCORE_3H_TRAINING_EFFICIENCY_17315_START"

"${PY}" tools/diagnose_3h_teacher_raw_advantages_17315.py   --solver "${SOLVER}"   --teacher "${T17315}"   --sanity-report "${SANITY}"   --workers 16   --seed 20261025   --max-decisions 300   --report "${RAW_REPORT}"

echo "SPINCORE_3H_RAW_ADVANTAGE_STAGE_PASS"

"${PY}" tools/diagnose_3h_teacher_flagged_checkpoint_trend.py   --solver "${SOLVER}"   --teacher-10315 "${T10315}"   --teacher-14815 "${T14815}"   --teacher-15315 "${T15315}"   --teacher-17315 "${T17315}"   --sanity-report "${SANITY}"   --workers 8   --seed 20261025   --max-decisions 300   --report "${TREND_REPORT}"

echo "SPINCORE_3H_FLAGGED_TREND_STAGE_PASS"

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${RAW_REPORT}" "${DWSL}/SpinCore_teacher_raw_advantage_17315.json"
    cp -f "${TREND_REPORT}" "${DWSL}/SpinCore_teacher_flagged_checkpoint_trend_17315.json"
    echo "desktop_raw=${DWIN}\\SpinCore_teacher_raw_advantage_17315.json"
    echo "desktop_trend=${DWIN}\\SpinCore_teacher_flagged_checkpoint_trend_17315.json"
  fi
fi

echo "SPINCORE_3H_TRAINING_EFFICIENCY_17315_PASS"
echo "STOP HERE. Send both JSON reports to ChatGPT."
