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
TEACHER="${LONG}/semantic_long_ensemble_final_15315.pt"
SOLVER="${ROOT}/build/libspincore_solver_c.so"
PROTOCOL="${ROOT}/contracts/post15315_multirebuild_preregistration.json"
PERF_REPORT="${ROOT}/runs/post15315_multirebuild_performance_gate/post15315_multirebuild_performance_gate.json"
MANIFEST="${ROOT}/contracts/run_manifests/post15315_multirebuild_production.json"
WORK="${ROOT}/runs/post15315_multirebuild_k8"
REPORT="${WORK}/post15315_multirebuild_k8_construction.json"
LOG="${WORK}/post15315_multirebuild_k8.log"

for f in "${PY}" "${CHECKPOINT}" "${TEACHER}" "${PROTOCOL}" "${PERF_REPORT}" "${MANIFEST}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing prerequisite: ${f}" >&2; exit 2; }
done

ACTIVE_PATTERN='[p]ython(3)? .*tools/(benchmark_post15315_multirebuild_performance|run_post15315_multirebuild_production|build_post15315_rebuild_member|run_3h_semantic_long_[^ ]*|evaluate_post15315_rebuild_stability)\.py|[b]ash .*tools/(run_post15315_multirebuild|run_3h_semantic_long_[^ ]*)\.sh'
ACTIVE_PROCS="$(ps -eo pid=,args= | grep -E "${ACTIVE_PATTERN}" || true)"
if [[ -n "${ACTIVE_PROCS}" ]]; then
  echo "ERROR: conflicting SpinCore heavy process is already running." >&2
  printf '%s\n' "${ACTIVE_PROCS}" >&2
  exit 3
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source/index changes present; production run requires clean HEAD." >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

"${PY}" tools/check_stage_manifest.py \
  --manifest contracts/run_manifests/post15315_multirebuild_production.json \
  --runner tools/run_post15315_multirebuild_production.sh

bash tools/run_post15315_multirebuild_preregistration_check.sh

"${PY}" -m py_compile \
  tools/post15315_collection.py \
  tools/post15315_member_cache.py \
  tools/build_post15315_rebuild_member.py \
  tools/run_post15315_multirebuild_production.py

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c

PERF_SHA="$("${PY}" - "${MANIFEST}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
print(p["performance_gate"]["evidence_sha256"])
PY
)"

mkdir -p "${WORK}"

echo "=== SpinCore post-15315 K=8 production construction ==="
echo "profile=4 collection workers x 2 threads; fits sequential x 8 threads"
echo "performance_report_sha256=${PERF_SHA}"
echo "No strength or stability evaluation is executed by this runner."

set +e
/usr/bin/time -v "${PY}" tools/run_post15315_multirebuild_production.py \
  --checkpoint "${CHECKPOINT}" \
  --teacher "${TEACHER}" \
  --solver "${SOLVER}" \
  --protocol "${PROTOCOL}" \
  --performance-report "${PERF_REPORT}" \
  --performance-report-sha256 "${PERF_SHA}" \
  --out-dir "${WORK}" \
  2>&1 | tee "${LOG}"
STATUS=${PIPESTATUS[0]}
set -e

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    [[ -f "${REPORT}" ]] && cp -f "${REPORT}" "${DWSL}/SpinCore_post15315_multirebuild_k8_construction.json" || true
    [[ -f "${LOG}" ]] && cp -f "${LOG}" "${DWSL}/SpinCore_post15315_multirebuild_k8.log" || true
    echo "desktop_report=${DWIN}\\SpinCore_post15315_multirebuild_k8_construction.json"
  fi
fi

if [[ "${STATUS}" -ne 0 ]]; then
  echo "POST15315_MULTIREBUILD_K8_RUNNER_STOP status=${STATUS}" >&2
  echo "STOP HERE. Send the construction JSON/log to ChatGPT; do not open stability or strength gates." >&2
  exit "${STATUS}"
fi

echo "POST15315_MULTIREBUILD_K8_RUNNER_PASS"
echo "STOP HERE. Send SpinCore_post15315_multirebuild_k8_construction.json to ChatGPT."
echo "Do not launch K=12 or any strength benchmark; K=8 stability is the next gate."
