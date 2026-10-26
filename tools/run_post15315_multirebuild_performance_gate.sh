#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: PERF-001,PERF-002,TRAIN-022,TRAIN-023,MODEL-025,RNG-001,RNG-003,RNG-013,VALID-025,VALID-031,VALID-032,VALID-034,VALID-035,VALID-036,VALID-037,VALID-038,SAFE-001,SAFE-010,SAFE-011,CKPT-001,ART-001

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"

PY="${ROOT}/.venv_lean/bin/python"
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"

SOURCE="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE}/checkpoint.pt"
LONG="${ROOT}/runs/3h_semantic_long_10315_15315"
LONG_REPORT="${LONG}/3h_semantic_long_10315_15315.json"
TEACHER="${LONG}/semantic_long_ensemble_final_15315.pt"
SOLVER="${ROOT}/build/libspincore_solver_c.so"
PROTOCOL="${ROOT}/contracts/post15315_multirebuild_preregistration.json"
MANIFEST="${ROOT}/contracts/run_manifests/post15315_multirebuild_performance_gate.json"
WORK="${ROOT}/runs/post15315_multirebuild_performance_gate"
REPORT="${WORK}/post15315_multirebuild_performance_gate.json"
LOG="${WORK}/post15315_multirebuild_performance_gate.log"

EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_TEACHER="ac49b77c8ec3204e07f6d44d3718c68880e60675c5521f538438a3fa3ceae746"

for f in "${PY}" "${CHECKPOINT}" "${LONG_REPORT}" "${TEACHER}" "${PROTOCOL}" "${MANIFEST}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing prerequisite: ${f}" >&2; exit 2; }
done

[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${EXPECTED_CP}" ]] || {
  echo "ERROR: frozen 10105 checkpoint SHA mismatch" >&2; exit 3;
}
[[ "$(sha256sum "${TEACHER}" | awk '{print $1}')" == "${EXPECTED_TEACHER}" ]] || {
  echo "ERROR: frozen 15315 teacher SHA mismatch" >&2; exit 4;
}

ACTIVE_PATTERN='[p]ython(3)? .*tools/(benchmark_post15315_multirebuild_performance|build_post15315_rebuild_member|run_3h_semantic_long_[^ ]*|evaluate_post15315_rebuild_stability)\.py|[b]ash .*tools/(run_post15315_multirebuild|run_3h_semantic_long_[^ ]*)\.sh'
ACTIVE_PROCS="$(ps -eo pid=,args= | grep -E "${ACTIVE_PATTERN}" || true)"
if [[ -n "${ACTIVE_PROCS}" ]]; then
  echo "ERROR: conflicting SpinCore heavy process is already running." >&2
  printf '%s\n' "${ACTIVE_PROCS}" >&2
  exit 5
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source/index changes present; gate requires clean HEAD." >&2
  git status --short --untracked-files=no >&2
  exit 6
fi

"${PY}" tools/check_stage_manifest.py   --manifest "${MANIFEST}"   --runner tools/run_post15315_multirebuild_performance_gate.sh

bash tools/run_post15315_multirebuild_preregistration_check.sh

"${PY}" -m py_compile   tools/post15315_collection.py   tools/build_post15315_rebuild_member.py   tools/benchmark_post15315_multirebuild_performance.py

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c

mkdir -p "${WORK}"

echo "=== SpinCore post-15315 K=8 Ryzen performance/parity gate ==="
echo "teacher_sha256=${EXPECTED_TEACHER}"
echo "No multi-GB checkpoint is fanned out into collection workers."

set +e
/usr/bin/time -v "${PY}" tools/benchmark_post15315_multirebuild_performance.py   --checkpoint "${CHECKPOINT}"   --teacher "${TEACHER}"   --solver "${SOLVER}"   --long-report "${LONG_REPORT}"   --protocol "${PROTOCOL}"   --report "${REPORT}"   2>&1 | tee "${LOG}"
STATUS=${PIPESTATUS[0]}
set -e

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    [[ -f "${REPORT}" ]] && cp -f "${REPORT}" "${DWSL}/SpinCore_post15315_multirebuild_performance_gate.json" || true
    [[ -f "${LOG}" ]] && cp -f "${LOG}" "${DWSL}/SpinCore_post15315_multirebuild_performance_gate.log" || true
    echo "desktop_report=${DWIN}\\SpinCore_post15315_multirebuild_performance_gate.json"
  fi
fi

if [[ "${STATUS}" -ne 0 ]]; then
  echo "POST15315_MULTIREBUILD_PERFORMANCE_GATE_STOP status=${STATUS}" >&2
  echo "STOP HERE. Send the JSON/log to ChatGPT; do not launch K=8 rebuilds." >&2
  exit "${STATUS}"
fi

echo "POST15315_MULTIREBUILD_PERFORMANCE_GATE_RUNNER_PASS"
echo "STOP HERE. Send SpinCore_post15315_multirebuild_performance_gate.json to ChatGPT."
echo "Do not launch K=8 until the production runner is bound to this exact PASS evidence."
