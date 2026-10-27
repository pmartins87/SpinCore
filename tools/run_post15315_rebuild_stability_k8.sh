#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: TRAIN-022,TRAIN-023,MODEL-025,RNG-001,RNG-003,RNG-013,VALID-025,VALID-031,VALID-032,VALID-034,VALID-035,VALID-036,VALID-037,VALID-038,SAFE-001,SAFE-010,SAFE-011,CKPT-001,ART-001

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"

PY="${ROOT}/.venv_lean/bin/python"
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"

SOLVER="${ROOT}/build/libspincore_solver_c.so"
TEACHER="${ROOT}/runs/3h_semantic_long_10315_15315/semantic_long_ensemble_final_15315.pt"
CONSTRUCTION="${ROOT}/runs/post15315_multirebuild_k8"
CONSTRUCTION_REPORT="${CONSTRUCTION}/post15315_multirebuild_k8_construction.json"
PROTOCOL="${ROOT}/contracts/post15315_multirebuild_preregistration.json"
ADJUDICATION="${ROOT}/contracts/post15315_stability_coverage_adjudication_v1.json"
MANIFEST="${ROOT}/contracts/run_manifests/post15315_rebuild_stability_k8.json"
WORK="${ROOT}/runs/post15315_rebuild_stability_k8"
REPORT="${WORK}/post15315_k8_stability_report.json"
GATE="${WORK}/post15315_k8_stability_gate.json"
META="${WORK}/post15315_k8_stability_bank_meta.json"
LOG="${WORK}/post15315_k8_stability.log"

for f in "${PY}" "${SOLVER}" "${TEACHER}" "${CONSTRUCTION_REPORT}" "${PROTOCOL}" "${ADJUDICATION}" "${MANIFEST}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing prerequisite: ${f}" >&2; exit 2; }
done

ACTIVE_PATTERN='[p]ython(3)? .*tools/(run_post15315_multirebuild_production|run_post15315_rebuild_stability_k8|build_post15315_rebuild_member|run_3h_semantic_long_[^ ]*|evaluate_post15315_rebuild_stability)\.py|[b]ash .*tools/(run_post15315_multirebuild|run_post15315_rebuild_stability|run_3h_semantic_long_[^ ]*)\.sh'
ACTIVE_PROCS="$(ps -eo pid=,args= | grep -E "${ACTIVE_PATTERN}" || true)"
if [[ -n "${ACTIVE_PROCS}" ]]; then
  echo "ERROR: conflicting SpinCore heavy process is already running." >&2
  printf '%s\n' "${ACTIVE_PROCS}" >&2
  exit 3
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source/index changes present; stability run requires clean HEAD." >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

"${PY}" tools/check_stage_manifest.py \
  --manifest contracts/run_manifests/post15315_rebuild_stability_k8.json \
  --runner tools/run_post15315_rebuild_stability_k8.sh

bash tools/run_post15315_multirebuild_preregistration_check.sh

"${PY}" -m py_compile \
  tools/run_post15315_rebuild_stability_k8.py \
  tools/evaluate_post15315_rebuild_stability.py

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c

mkdir -p "${WORK}"

echo "=== SpinCore post-15315 K=8 rebuild stability ==="
echo "seed=153159001 forced_three_handed_episodes=100000 coverage_only_adjudication=true"
echo "member_set=1..8 common_state_bank=true"
echo "No training, member selection, K=12 construction, or strength benchmark is executed by this runner."

set +e
/usr/bin/time -v "${PY}" tools/run_post15315_rebuild_stability_k8.py \
  --solver "${SOLVER}" \
  --teacher "${TEACHER}" \
  --construction-dir "${CONSTRUCTION}" \
  --protocol "${PROTOCOL}" \
  --coverage-adjudication "${ADJUDICATION}" \
  --out-dir "${WORK}" \
  --threads 8 \
  --batch-size 4096 \
  2>&1 | tee "${LOG}"
STATUS=${PIPESTATUS[0]}
set -e

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    [[ -f "${REPORT}" ]] && cp -f "${REPORT}" "${DWSL}/SpinCore_post15315_k8_stability_report.json" || true
    [[ -f "${GATE}" ]] && cp -f "${GATE}" "${DWSL}/SpinCore_post15315_k8_stability_gate.json" || true
    [[ -f "${META}" ]] && cp -f "${META}" "${DWSL}/SpinCore_post15315_k8_stability_bank_meta.json" || true
    [[ -f "${LOG}" ]] && cp -f "${LOG}" "${DWSL}/SpinCore_post15315_k8_stability.log" || true
    echo "desktop_report=${DWIN}\\SpinCore_post15315_k8_stability_report.json"
    echo "desktop_gate=${DWIN}\\SpinCore_post15315_k8_stability_gate.json"
  fi
fi

if [[ "${STATUS}" -ne 0 ]]; then
  echo "POST15315_K8_STABILITY_RUNNER_STOP status=${STATUS}" >&2
  echo "STOP HERE. Send the stability log/report to ChatGPT; do not launch K=12 or strength." >&2
  exit "${STATUS}"
fi

grep -q "POST15315_K8_STABILITY_RUNNER_PASS" "${LOG}"
test -s "${REPORT}"
test -s "${GATE}"
test -s "${META}"

echo "POST15315_K8_STABILITY_SHELL_PASS"
echo "STOP HERE. Send SpinCore_post15315_k8_stability_gate.json and SpinCore_post15315_k8_stability_report.json to ChatGPT."
echo "Do not launch K=12 unless the gate explicitly says it is authorized; do not launch any strength benchmark."
