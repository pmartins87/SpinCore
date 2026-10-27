#!/usr/bin/env bash
# PROJECT_CONTRACT_IDS: PERF-001,PERF-002,TRAIN-022,TRAIN-023,MODEL-025,RNG-001,RNG-003,RNG-013,VALID-025,VALID-031,VALID-032,VALID-034,VALID-035,VALID-036,VALID-037,VALID-038,SAFE-001,SAFE-010,SAFE-011,CKPT-001,ART-001

set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"

CHECKPOINT="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104/checkpoint.pt"
TEACHER="${ROOT}/runs/3h_semantic_long_10315_15315/semantic_long_ensemble_final_15315.pt"
SOLVER="${ROOT}/build/libspincore_solver_c.so"
PROTOCOL="${ROOT}/contracts/post15315_multirebuild_preregistration.json"
ADJUDICATION="${ROOT}/contracts/post15315_stability_coverage_adjudication_v1.json"
K8_GATE="${ROOT}/runs/post15315_rebuild_stability_k8/post15315_k8_stability_gate.json"
PERF_REPORT="${ROOT}/runs/post15315_multirebuild_performance_gate/post15315_multirebuild_performance_gate.json"
MANIFEST="${ROOT}/contracts/run_manifests/post15315_multirebuild_k12_extension.json"
WORK="${ROOT}/runs/post15315_multirebuild_k12_extension"
REPORT="${WORK}/post15315_multirebuild_k12_extension_construction.json"
LOG="${WORK}/post15315_multirebuild_k12_extension.log"

for f in "${PY}" "${CHECKPOINT}" "${TEACHER}" "${SOLVER}" "${PROTOCOL}" "${ADJUDICATION}" "${K8_GATE}" "${PERF_REPORT}" "${MANIFEST}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing prerequisite: ${f}" >&2; exit 2; }
done

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked changes present; clean HEAD required." >&2
  exit 4
fi

"${PY}" tools/check_stage_manifest.py --manifest contracts/run_manifests/post15315_multirebuild_k12_extension.json --runner tools/run_post15315_multirebuild_k12_extension.sh
bash tools/run_post15315_multirebuild_preregistration_check.sh
"${PY}" -m py_compile tools/run_post15315_multirebuild_k12_extension.py tools/run_post15315_multirebuild_production.py tools/build_post15315_rebuild_member_cached.py tools/post15315_member_cache.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c

PERF_SHA="4f724458cc5304caf95842e32b3a010178042494b15045d08d9b9bb3eaacbc48"
mkdir -p "${WORK}"

echo "=== SpinCore post-15315 K=12 conditional extension construction ==="
echo "members=9,10,11,12"
echo "profile=4 collection workers x 2 threads; fits sequential x 8 threads"
echo "Authorization source=mechanically clean K=8 stability FAIL"
echo "No K=12 stability or strength benchmark is executed by this runner."

set +e
/usr/bin/time -v "${PY}" tools/run_post15315_multirebuild_k12_extension.py   --checkpoint "${CHECKPOINT}"   --teacher "${TEACHER}"   --solver "${SOLVER}"   --protocol "${PROTOCOL}"   --coverage-adjudication "${ADJUDICATION}"   --k8-stability-gate "${K8_GATE}"   --performance-report "${PERF_REPORT}"   --performance-report-sha256 "${PERF_SHA}"   --out-dir "${WORK}"   2>&1 | tee "${LOG}"
STATUS=${PIPESTATUS[0]}
set -e

if [[ "${STATUS}" -ne 0 ]]; then
  echo "POST15315_K12_EXTENSION_RUNNER_STOP status=${STATUS}" >&2
  exit "${STATUS}"
fi

grep -q "POST15315_MULTIREBUILD_K12_EXTENSION_CONSTRUCTION_PASS" "${LOG}"
test -s "${REPORT}"
cp -f "${REPORT}" /mnt/c/Users/Rz9/Desktop/SpinCore_post15315_k12_extension_construction.json
cp -f "${LOG}" /mnt/c/Users/Rz9/Desktop/SpinCore_post15315_k12_extension.log

echo "POST15315_K12_EXTENSION_SHELL_PASS"
echo "STOP HERE. Send SpinCore_post15315_k12_extension_construction.json to ChatGPT."
