#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: DC-002,SAFE-001,TRAIN-022,VALID-001,VALID-003,VALID-025,VALID-031,VALID-034,RNG-001,RNG-002,RNG-003

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"

PY="${ROOT}/.venv_lean/bin/python"
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SPINCORE_TORCH_THREADS=1

SOLVER="${ROOT}/build/libspincore_solver_c.so"
MANIFEST="${ROOT}/contracts/run_manifests/shadow_teacher_learning_10315_12815.json"
RUNNER="tools/run_3h_teacher_shadow_learning_10315_12815.sh"
OLD="${ROOT}/runs/3h_semantic_long_10115_10315/semantic_long_ensemble_final_10315.pt"
NEW="${ROOT}/runs/3h_semantic_long_10315_15315/semantic_long_ensemble_12815.pt"
OUTDIR="${ROOT}/runs/shadow_teacher_learning_10315_12815"
REPORT="${OUTDIR}/shadow_teacher_learning_10315_12815.json"
mkdir -p "${OUTDIR}"

for f in "${PY}" "${SOLVER}" "${OLD}" "${NEW}" "${MANIFEST}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing prerequisite: ${f}" >&2; exit 2; }
done

"${PY}" tools/check_stage_manifest.py --manifest "${MANIFEST}" --runner "${RUNNER}"

"${PY}" -m py_compile tools/evaluate_3h_teacher_shadow_learning_10315_12815.py

# Read-only diagnostic. Keep it low priority and confined to two logical CPUs so
# the active 4x8 macrotrainer keeps scheduling priority.
CMD=(
  "${PY}" tools/evaluate_3h_teacher_shadow_learning_10315_12815.py
  --solver "${SOLVER}"
  --teacher-10315 "${OLD}"
  --teacher-12815 "${NEW}"
  --scenarios 1000
  --workers 2
  --seed 20261011
  --max-decisions 300
  --report "${REPORT}"
)

if command -v taskset >/dev/null 2>&1; then
  nice -n 19 taskset -c 30,31 "${CMD[@]}"
else
  nice -n 19 "${CMD[@]}"
fi

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${REPORT}" "${DWSL}/SpinCore_shadow_teacher_learning_10315_12815.json"
    echo "desktop_report=${DWIN}\\SpinCore_shadow_teacher_learning_10315_12815.json"
  fi
fi

echo "SHADOW_TEACHER_LEARNING_RUN_COMPLETE"
echo "STOP HERE. Send SpinCore_shadow_teacher_learning_10315_12815.json to ChatGPT."
