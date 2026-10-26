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
TEACHER="${ARTIFACT_ROOT}/runs/3h_semantic_long_15315_17315/semantic_long_ensemble_final_17315.pt"
SANITY="${ARTIFACT_ROOT}/runs/teacher_sanity_17315/teacher_sanity_17315.json"
OUTDIR="${CODE_ROOT}/runs/teacher_raw_advantage_17315"
REPORT="${OUTDIR}/teacher_raw_advantage_17315.json"
LOG="${OUTDIR}/teacher_raw_advantage_17315.log"
mkdir -p "${OUTDIR}"

for path in "${PY}" "${SOLVER}" "${TEACHER}" "${SANITY}"; do
  [[ -e "${path}" ]] || { echo "ERROR: missing prerequisite: ${path}" >&2; exit 2; }
done

"${PY}" -m py_compile tools/diagnose_3h_teacher_raw_advantages_17315.py

echo "SPINCORE_3H_TEACHER_RAW_ADVANTAGE_17315_START"
echo "teacher=${TEACHER}"
echo "sanity_report=${SANITY}"
echo "workers=16 seed=20261025"

CMD=(
  "${PY}" tools/diagnose_3h_teacher_raw_advantages_17315.py
  --solver "${SOLVER}"
  --teacher "${TEACHER}"
  --sanity-report "${SANITY}"
  --workers 16
  --seed 20261025
  --max-decisions 300
  --report "${REPORT}"
)

set +e
if command -v taskset >/dev/null 2>&1; then
  nice -n 5 taskset -c 0-15 "${CMD[@]}" 2>&1 | tee "${LOG}"
else
  nice -n 5 "${CMD[@]}" 2>&1 | tee "${LOG}"
fi
RC=${PIPESTATUS[0]}
set -e

if [[ "${RC}" -ne 0 ]]; then
  echo "SPINCORE_3H_TEACHER_RAW_ADVANTAGE_17315_FAIL rc=${RC}"
  exit "${RC}"
fi

grep -q "SPINCORE_3H_TEACHER_RAW_ADVANTAGE_17315_COMPLETE" "${LOG}"
test -s "${REPORT}"

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${REPORT}" "${DWSL}/SpinCore_teacher_raw_advantage_17315.json"
    cp -f "${LOG}" "${DWSL}/SpinCore_teacher_raw_advantage_17315.log"
    echo "desktop_report=${DWIN}\\SpinCore_teacher_raw_advantage_17315.json"
  fi
fi

echo "SPINCORE_3H_TEACHER_RAW_ADVANTAGE_17315_RUNNER_PASS"
echo "STOP HERE. Send SpinCore_teacher_raw_advantage_17315.json to ChatGPT."
