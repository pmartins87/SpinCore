#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: GOV-023,PERF-001,PERF-002,PERF-010,PERF-011,PERF-012,PERF-013,PERF-014,PERF-015,PERF-016,PERF-017,PERF-019,PERF-020,PERF-022,PERF-024,TRAIN-020,MODEL-020,RNG-001,RNG-002,RNG-003,CKPT-004,SAFE-002,ART-015

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"

PY="${ROOT}/.venv_lean/bin/python"
SOLVER="${ROOT}/build/libspincore_solver_c.so"
MANIFEST="${ROOT}/contracts/run_manifests/semantic_long_10115_10315_parallel_resume.json"
RUNNER="tools/run_3h_semantic_long_10115_10315_parallel_resume.sh"

SOURCE="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE}/checkpoint.pt"
HU="${SOURCE}/hu_ensemble_state.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_HU="8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"

CANON="${ROOT}/runs/3h_semantic_research_canonical_10105_10115"
SOURCE_RESUME="${CANON}/resume_state.pt"
SOURCE_FINAL_ENSEMBLE="${CANON}/semantic_ensemble_final_10115.pt"
SPIN_BUNDLE="${CANON}/spincore_hybrid_10105.pt"

ADV_RUN="$(find "${ROOT}/runs/3h_v1_semantic_shadow" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
PROBE_RUN="$(find "${ROOT}/runs/3h_advantage_controlled_budget_curve" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
ADV_MODELS="${ADV_RUN}/3h_v1_semantic_shadow_models.pt"
PROBE="${PROBE_RUN}/3h_advantage_controlled_split_probe.pt"

RUN="${ROOT}/runs/3h_semantic_long_10115_10315"
RESUME="${RUN}/resume_state.pt"
REPORT="${RUN}/3h_semantic_long_10115_10315.json"
LOG="${RUN}/training_parallel_resume.log"
MEMLOG="${RUN}/memory_parallel_resume.log"
PACK="${RUN}/parallel_semantic_pack"

for f in "${PY}" "${MANIFEST}" "${CHECKPOINT}" "${HU}"   "${SOURCE_RESUME}" "${SOURCE_FINAL_ENSEMBLE}" "${SPIN_BUNDLE}"   "${ADV_MODELS}" "${PROBE}" "${RESUME}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing prerequisite: ${f}" >&2; exit 2; }
done

# This is the primary fail-closed authorization.  While the performance gate is
# pending the manifest is BLOCKED and this runner exits here.
"${PY}" tools/check_stage_manifest.py --manifest "${MANIFEST}" --runner "${RUNNER}"

readarray -t PROFILE < <("${PY}" - "${MANIFEST}" <<'PY'
import json,sys
m=json.load(open(sys.argv[1],"r",encoding="utf-8"))
p=m.get("selected_profile") or {}
if m.get("status")!="READY":
    raise SystemExit("stage manifest is not READY")
for key in ("concurrency","threads_per_member"):
    if key not in p:
        raise SystemExit("selected profile incomplete")
print(int(p["concurrency"]))
print(int(p["threads_per_member"]))
for path,sha in sorted((m.get("support_blobs") or {}).items()):
    print("BLOB\t"+path+"\t"+sha)
PY
)

CONCURRENCY="${PROFILE[0]}"
THREADS_PER_MEMBER="${PROFILE[1]}"
[[ "${CONCURRENCY}" -gt 0 && "${THREADS_PER_MEMBER}" -gt 0 ]] || {
  echo "ERROR: invalid selected parallel profile" >&2; exit 3;
}
[[ $((CONCURRENCY*THREADS_PER_MEMBER)) -le 32 ]] || {
  echo "ERROR: selected profile oversubscribes 32 logical threads" >&2; exit 4;
}

for row in "${PROFILE[@]:2}"; do
  IFS=$'\t' read -r tag path expected <<< "${row}"
  [[ "${tag}" == "BLOB" ]] || { echo "ERROR: malformed support blob row" >&2; exit 5; }
  [[ -f "${path}" ]] || { echo "ERROR: support file missing: ${path}" >&2; exit 6; }
  actual="$(git hash-object "${path}")"
  [[ "${actual}" == "${expected}" ]] || {
    echo "ERROR: support blob drift: ${path} actual=${actual} expected=${expected}" >&2
    exit 7
  }
done

[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${EXPECTED_CP}" ]] || {
  echo "ERROR: frozen 10105 checkpoint SHA mismatch" >&2; exit 8;
}
[[ "$(sha256sum "${HU}" | awk '{print $1}')" == "${EXPECTED_HU}" ]] || {
  echo "ERROR: frozen HU sidecar SHA mismatch" >&2; exit 9;
}

if pgrep -af "run_3h_semantic_long_10115_10315.py|run_3h_semantic_long_10115_10315_parallel.py" >/dev/null 2>&1; then
  echo "ERROR: a semantic long trainer is already running." >&2
  pgrep -af "run_3h_semantic_long_10115_10315.py|run_3h_semantic_long_10115_10315_parallel.py" >&2 || true
  exit 10
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source/index changes present; optimized resume requires clean HEAD." >&2
  git status --short --untracked-files=no >&2
  exit 11
fi

"${PY}" - "${RESUME}" "${EXPECTED_CP}" <<'PY'
import sys,torch
p=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
assert p["schema"]=="SPINCORE_3H_SEMANTIC_LONG_RESUME_10115_10315_V1"
assert p["source_checkpoint_sha256"]==sys.argv[2]
assert p["target_iteration"]==10315
completed=int(p["completed_iteration"])
assert 10115 < completed < 10315
assert len(p["semantic_members"])==8
assert len(p["member_seed_contract"])==8
print("SEMANTIC_PARALLEL_RESUME_STATE_PASS")
print("completed_iteration="+str(completed))
print("remaining_iterations="+str(10315-completed))
PY

MEMTOTAL_KIB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
MEMAVAIL_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if [[ -z "${MEMTOTAL_KIB}" || "${MEMTOTAL_KIB}" -lt $((28*1024*1024)) ]]; then
  echo "ERROR: WSL exposes less than 28 GiB RAM." >&2
  exit 12
fi
if [[ -z "${MEMAVAIL_KIB}" || "${MEMAVAIL_KIB}" -lt $((12*1024*1024)) ]]; then
  echo "ERROR: less than 12 GiB WSL MemAvailable before optimized resume." >&2
  free -h >&2 || true
  exit 13
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c

export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export SPINCORE_SEMANTIC_FIT_CONCURRENCY="${CONCURRENCY}"
export SPINCORE_SEMANTIC_THREADS_PER_MEMBER="${THREADS_PER_MEMBER}"
export SPINCORE_SEMANTIC_PACK_DIR="${PACK}"

"${PY}" -m py_compile   tools/lt3_3h_semantic_parallel_fit.py   tools/run_3h_semantic_long_10115_10315_parallel.py   tools/run_3h_semantic_long_10115_10315.py

"${PY}" -m pytest -q python_tests/test_lt3_3h_semantic_parallel_fit.py

snapshot_memory() {
  local tag="$1"
  local now
  now="$(date --iso-8601=seconds)"
  awk -v ts="${now}" -v tag="${tag}" '
    /^MemTotal:/ {mt=$2}
    /^MemAvailable:/ {ma=$2}
    /^SwapTotal:/ {st=$2}
    /^SwapFree:/ {sf=$2}
    END {
      printf "%s tag=%s mem_total_kib=%s mem_available_kib=%s swap_total_kib=%s swap_free_kib=%s swap_used_kib=%s\n", ts, tag, mt, ma, st, sf, st-sf
    }
  ' /proc/meminfo >> "${MEMLOG}"
}

monitor_memory() {
  local pid="$1"
  while kill -0 "${pid}" 2>/dev/null; do
    snapshot_memory periodic
    sleep 30
  done
}

copy_evidence() {
  if ! command -v powershell.exe >/dev/null 2>&1; then return 0; fi
  local dwin dwsl
  dwin="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  dwsl="$(wslpath -u "${dwin}" 2>/dev/null || true)"
  [[ -n "${dwsl}" && -d "${dwsl}" ]] || return 0
  [[ -f "${REPORT}" ]] && cp -f "${REPORT}" "${dwsl}/SpinCore_3H_semantic_long_10115_10315.json" || true
  [[ -f "${MEMLOG}" ]] && cp -f "${MEMLOG}" "${dwsl}/SpinCore_3H_semantic_parallel_resume_memory.log" || true
  [[ -f "${LOG}" ]] && cp -f "${LOG}" "${dwsl}/SpinCore_3H_semantic_parallel_resume.log" || true
}

echo "=== SpinCore 3H semantic long optimized resume ==="
echo "parallel_profile=${CONCURRENCY}x${THREADS_PER_MEMBER}"
echo "resume=${RESUME}"
echo "pack=${PACK}"
free -h
echo

rm -rf "${PACK}"
snapshot_memory before

set +e
(
  /usr/bin/time -v "${PY}" tools/run_3h_semantic_long_10115_10315_parallel.py     --checkpoint "${CHECKPOINT}"     --semantic-advantage-models "${ADV_MODELS}"     --probe "${PROBE}"     --solver "${SOLVER}"     --spin-bundle "${SPIN_BUNDLE}"     --source-resume "${SOURCE_RESUME}"     --source-final-ensemble "${SOURCE_FINAL_ENSEMBLE}"     --run-dir "${RUN}"     --threads 8     2>&1 | tee -a "${LOG}"
  exit "${PIPESTATUS[0]}"
) &
TRAIN_PID=$!

monitor_memory "${TRAIN_PID}" &
MONITOR_PID=$!

wait "${TRAIN_PID}"
STATUS=$?
kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
set -e
snapshot_memory after
copy_evidence

if [[ "${STATUS}" -ne 0 ]]; then
  echo "SEMANTIC_LONG_PARALLEL_RESUME_STOP status=${STATUS}" >&2
  echo "Do not restart blindly. Send report/log to ChatGPT." >&2
  exit "${STATUS}"
fi

"${PY}" - "${REPORT}" "${EXPECTED_CP}" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],"r",encoding="utf-8"))
assert d["status"]=="PASS"
assert d["source_checkpoint_sha256"]==sys.argv[2]
assert d["completed_iteration"]==10315
assert d["new_roots_recorded"]==12800
assert d["final_safety_guard_pass"] is True
assert d["source_10105_mutated"] is False
assert d["hu_training_performed"] is False
print("SEMANTIC_LONG_PARALLEL_RESUME_POSTVALIDATION_PASS")
print("final_ensemble_sha256="+d["final_ensemble_sha256"])
print("resume_state_sha256="+d["resume_state_sha256"])
PY

echo
echo "SEMANTIC_LONG_10115_10315_TRAINING_PASS"
echo "STOP HERE. Send the final JSON and parallel resume log to ChatGPT."
