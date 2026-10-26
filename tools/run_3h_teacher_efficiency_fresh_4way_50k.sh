#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: DC-002,SAFE-001,SAFE-011,VALID-025,VALID-031,VALID-034,RNG-001,RNG-002,RNG-003

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
MANIFEST="${CODE_ROOT}/contracts/run_manifests/teacher_efficiency_fresh_4way_50k.json"
OUTDIR="${CODE_ROOT}/runs/teacher_efficiency_fresh_4way_50k"
REPORT="${OUTDIR}/teacher_efficiency_fresh_4way_50k.json"
LOG="${OUTDIR}/teacher_efficiency_fresh_4way_50k.log"
mkdir -p "${OUTDIR}"

for path in "${PY}" "${SOLVER}" "${T10315}" "${T14815}" "${T15315}" "${T17315}" "${MANIFEST}"; do
  [[ -e "${path}" ]] || { echo "ERROR: missing prerequisite: ${path}" >&2; exit 2; }
done

"${PY}" tools/check_stage_manifest.py   --manifest "${MANIFEST}"   --runner tools/run_3h_teacher_efficiency_fresh_4way_50k.sh
"${PY}" -m py_compile tools/evaluate_3h_teacher_efficiency_fresh_4way.py

echo "SPINCORE_3H_TEACHER_EFFICIENCY_FRESH_4WAY_50K_START"
echo "scenarios=50000 seed=20261027 workers=16"
echo "This benchmark is fresh, paired, development-only, and excludes DC0 promotion claims."

CMD=(
  "${PY}" tools/evaluate_3h_teacher_efficiency_fresh_4way.py
  --solver "${SOLVER}"
  --teacher-10315 "${T10315}"
  --teacher-14815 "${T14815}"
  --teacher-15315 "${T15315}"
  --teacher-17315 "${T17315}"
  --scenarios 50000
  --workers 16
  --seed 20261027
  --max-decisions 300
  --report "${REPORT}"
)

set +e
nice -n 5 "${CMD[@]}" 2>&1 | tee "${LOG}"
RC=${PIPESTATUS[0]}
set -e

if [[ "${RC}" -ne 0 ]]; then
  echo "SPINCORE_3H_TEACHER_EFFICIENCY_FRESH_4WAY_50K_FAIL rc=${RC}"
  exit "${RC}"
fi

grep -q "SPINCORE_3H_TEACHER_EFFICIENCY_FRESH_4WAY_COMPLETE" "${LOG}"
test -s "${REPORT}"

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${REPORT}" "${DWSL}/SpinCore_teacher_efficiency_fresh_4way_50k.json"
    cp -f "${LOG}" "${DWSL}/SpinCore_teacher_efficiency_fresh_4way_50k.log"
    echo "desktop_report=${DWIN}\\SpinCore_teacher_efficiency_fresh_4way_50k.json"
  fi
fi

echo "SPINCORE_3H_TEACHER_EFFICIENCY_FRESH_4WAY_50K_RUNNER_PASS"
echo "STOP HERE. Send SpinCore_teacher_efficiency_fresh_4way_50k.json to ChatGPT."
