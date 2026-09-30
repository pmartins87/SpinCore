#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: GOV-023,PERF-001,PERF-002,PERF-010,PERF-013,PERF-014,PERF-015,PERF-017,TRAIN-021,MODEL-021,MODEL-022,MODEL-023,RNG-001,RNG-002,RNG-003,VALID-002,VALID-020,VALID-021,VALID-022,CKPT-001,ART-001,SRC-003

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"

PY="${ROOT}/.venv_lean/bin/python"
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"

SOURCE="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE}/checkpoint.pt"
LONG="${ROOT}/runs/3h_semantic_long_10115_10315"
LONG_REPORT="${LONG}/3h_semantic_long_10115_10315.json"
ADV="${LONG}/semantic_long_ensemble_final_10315.pt"
SOLVER="${ROOT}/build/libspincore_solver_c.so"
WORK="${LONG}/postlong_performance_gate"
REPORT="${WORK}/postlong_10315_performance_gate.json"

EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_ADV="acf86fa05d7e6ac611fa60a346758a3d838c7918937d14488810af9c95abf5ca"

for f in "${PY}" "${CHECKPOINT}" "${LONG_REPORT}" "${ADV}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing prerequisite: ${f}" >&2; exit 2; }
done

[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${EXPECTED_CP}" ]] || {
  echo "ERROR: frozen 10105 checkpoint SHA mismatch" >&2; exit 3;
}
[[ "$(sha256sum "${ADV}" | awk '{print $1}')" == "${EXPECTED_ADV}" ]] || {
  echo "ERROR: frozen 10315 teacher SHA mismatch" >&2; exit 4;
}

if pgrep -af "run_3h_semantic_long_10115_10315|run_3h_semantic_postlong_10315" >/dev/null 2>&1; then
  echo "ERROR: semantic long/postlong process is already running." >&2
  pgrep -af "run_3h_semantic_long_10115_10315|run_3h_semantic_postlong_10315" >&2 || true
  exit 5
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source/index changes present; gate requires clean HEAD." >&2
  git status --short --untracked-files=no >&2
  exit 6
fi

"${PY}" tools/check_project_contract.py

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c

"${PY}" -m py_compile   tools/postlong_10315_collection.py   tools/benchmark_3h_semantic_postlong_10315_performance.py   tools/audit_3h_semantic_stratified_specialist_multiseed_10315_parallel.py   tools/build_3h_semantic_postlong_policy_10315.py   tools/build_3h_semantic_stratified_specialist_10315.py

mkdir -p "${WORK}"

echo "=== SpinCore postlong 10315 Ryzen performance/parity gate ==="
set +e
/usr/bin/time -v "${PY}" tools/benchmark_3h_semantic_postlong_10315_performance.py   --checkpoint "${CHECKPOINT}"   --semantic-advantage "${ADV}"   --solver "${SOLVER}"   --long-report "${LONG_REPORT}"   --report "${REPORT}"   2>&1 | tee "${WORK}/postlong_10315_performance_gate.log"
STATUS=${PIPESTATUS[0]}
set -e

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    [[ -f "${REPORT}" ]] && cp -f "${REPORT}" "${DWSL}/SpinCore_3H_postlong_10315_performance_gate.json" || true
    [[ -f "${WORK}/postlong_10315_performance_gate.log" ]] && cp -f "${WORK}/postlong_10315_performance_gate.log" "${DWSL}/SpinCore_3H_postlong_10315_performance_gate.log" || true
  fi
fi

if [[ "${STATUS}" -ne 0 ]]; then
  echo "POSTLONG_10315_PERFORMANCE_GATE_STOP status=${STATUS}" >&2
  echo "STOP HERE. Send the JSON/log to ChatGPT; do not launch postlong." >&2
  exit "${STATUS}"
fi

echo "POSTLONG_10315_PERFORMANCE_GATE_PASS"
echo "STOP HERE. Do not launch postlong until its manifest is promoted to READY from this exact evidence."
echo "report=${REPORT}"
