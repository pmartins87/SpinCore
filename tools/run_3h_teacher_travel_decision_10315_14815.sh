#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: DC-002,SAFE-001,TRAIN-022,VALID-001,VALID-003,VALID-025,VALID-031,VALID-034,VALID-039,RNG-001,RNG-002,RNG-003

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
MANIFEST="${CODE_ROOT}/contracts/run_manifests/teacher_travel_decision_10315_14815.json"
RUNNER="tools/run_3h_teacher_travel_decision_10315_14815.sh"
OLD="${ARTIFACT_ROOT}/runs/3h_semantic_long_10115_10315/semantic_long_ensemble_final_10315.pt"
NEW="${ARTIFACT_ROOT}/runs/3h_semantic_long_10315_15315/semantic_long_ensemble_14815.pt"
OUTDIR="${CODE_ROOT}/runs/teacher_travel_decision_10315_14815"
REPORT="${OUTDIR}/teacher_travel_decision_10315_14815.json"
mkdir -p "${OUTDIR}"

for f in "${PY}" "${SOLVER}" "${OLD}" "${NEW}" "${MANIFEST}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing prerequisite: ${f}" >&2; exit 2; }
done

"${PY}" tools/check_stage_manifest.py --manifest "${MANIFEST}" --runner "${RUNNER}"

"${PY}" -m py_compile tools/evaluate_3h_teacher_travel_decision_10315_14815.py

# Read-only diagnostic. Keep it low priority and confined to eight logical CPUs so
# the active 4x8 macrotrainer keeps scheduling priority.
CMD=(
  "${PY}" tools/evaluate_3h_teacher_travel_decision_10315_14815.py
  --solver "${SOLVER}"
  --teacher-10315 "${OLD}"
  --teacher-14815 "${NEW}"
  --scenarios 5000
  --workers 8
  --seed 20261017
  --max-decisions 300
  --report "${REPORT}"
)

if command -v taskset >/dev/null 2>&1; then
  nice -n 19 taskset -c 24-31 "${CMD[@]}"
else
  nice -n 19 "${CMD[@]}"
fi

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${REPORT}" "${DWSL}/SpinCore_teacher_travel_decision_10315_14815.json"
    echo "desktop_report=${DWIN}\\SpinCore_teacher_travel_decision_10315_14815.json"
  fi
fi

echo "TEACHER_TRAVEL_DECISION_RUN_COMPLETE"
echo "STOP HERE. Send SpinCore_teacher_travel_decision_10315_14815.json to ChatGPT."
