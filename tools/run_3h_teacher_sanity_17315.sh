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
MANIFEST="${CODE_ROOT}/contracts/run_manifests/teacher_sanity_17315.json"
RUNNER="tools/run_3h_teacher_sanity_17315.sh"
OUTDIR="${CODE_ROOT}/runs/teacher_sanity_17315"
REPORT="${OUTDIR}/teacher_sanity_17315.json"
LOG="${OUTDIR}/teacher_sanity_17315.log"
mkdir -p "${OUTDIR}"

for path in "${PY}" "${SOLVER}" "${TEACHER}" "${MANIFEST}"; do
  [[ -e "${path}" ]] || { echo "ERROR: missing prerequisite: ${path}" >&2; exit 2; }
done

"${PY}" tools/check_stage_manifest.py --manifest "${MANIFEST}" --runner "${RUNNER}"
"${PY}" -m py_compile tools/audit_3h_teacher_sanity_17315.py

"${PY}" - <<'PY'
import tools.audit_3h_teacher_sanity_17315 as audit

class _SmokeState:
    def neural_bytes_v2(self):
        payload = bytearray(830)
        payload[:8] = b"SPNNIV2" + bytes((0,))
        payload[112] = 0
        return bytes(payload)

assert audit.street_from_state(_SmokeState()) == 0
print("SPNNIV2_SANITY_SMOKE_PASS")
PY

echo "SPINCORE_3H_TEACHER_SANITY_17315_START"
echo "code_root=${CODE_ROOT}"
echo "artifact_root=${ARTIFACT_ROOT}"
echo "teacher=${TEACHER}"
echo "scenarios=10000 seed=20261025 workers=16"

CMD=(
  "${PY}" tools/audit_3h_teacher_sanity_17315.py
  --solver "${SOLVER}"
  --teacher "${TEACHER}"
  --scenarios 10000
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
  echo "SPINCORE_3H_TEACHER_SANITY_17315_FAIL rc=${RC}"
  exit "${RC}"
fi

grep -q "SPINCORE_3H_TEACHER_SANITY_17315_COMPLETE" "${LOG}"
test -s "${REPORT}"

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${REPORT}" "${DWSL}/SpinCore_teacher_sanity_17315.json"
    cp -f "${LOG}" "${DWSL}/SpinCore_teacher_sanity_17315.log"
    echo "desktop_report=${DWIN}\\SpinCore_teacher_sanity_17315.json"
    echo "desktop_log=${DWIN}\\SpinCore_teacher_sanity_17315.log"
  fi
fi

echo "SPINCORE_3H_TEACHER_SANITY_17315_RUNNER_PASS"
echo "STOP HERE. Send SpinCore_teacher_sanity_17315.json to ChatGPT."
