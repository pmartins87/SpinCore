#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: GOV-023,PERF-001,PERF-002,PERF-010,PERF-011,PERF-012,PERF-013,PERF-014,PERF-015,PERF-016,PERF-017,PERF-019,PERF-020,PERF-022,PERF-024,TRAIN-020,MODEL-020,RNG-001,RNG-002,RNG-003,CKPT-004,ART-015

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"

PY="${ROOT}/.venv_lean/bin/python"
SOURCE="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104/checkpoint.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
RUN="${ROOT}/runs/3h_semantic_long_10115_10315"
RESUME="${RUN}/resume_state.pt"
WORK="${RUN}/parallel_fit_gate"
REPORT="${WORK}/semantic_parallel_fit_gate.json"

PROBE_RUN="$(find "${ROOT}/runs/3h_advantage_controlled_budget_curve" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
PROBE="${PROBE_RUN}/3h_advantage_controlled_split_probe.pt"

for f in "${PY}" "${SOURCE}" "${PROBE}" "${RESUME}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing prerequisite: ${f}" >&2; exit 2; }
done

[[ "$(sha256sum "${SOURCE}" | awk '{print $1}')" == "${EXPECTED_CP}" ]] || {
  echo "ERROR: frozen 10105 checkpoint SHA mismatch" >&2
  exit 3
}

if pgrep -af "run_3h_semantic_long_10115_10315.py|run_3h_semantic_long_10115_10315_parallel.py" >/dev/null 2>&1; then
  echo "ERROR: semantic long trainer is still running; benchmark only after a durable stop." >&2
  pgrep -af "run_3h_semantic_long_10115_10315.py|run_3h_semantic_long_10115_10315_parallel.py" >&2 || true
  exit 4
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source/index changes present; benchmark requires clean HEAD." >&2
  git status --short --untracked-files=no >&2
  exit 5
fi

export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

"${PY}" -m py_compile   tools/lt3_3h_semantic_parallel_fit.py   tools/benchmark_3h_semantic_parallel_fit_10115_10315.py   tools/run_3h_semantic_long_10115_10315_parallel.py

"${PY}" -m pytest -q python_tests/test_lt3_3h_semantic_parallel_fit.py

mkdir -p "${WORK}"
rm -rf "${WORK}/packed"

echo "=== SpinCore 3H semantic parallel fit gate ==="
"${PY}" - "${RESUME}" <<'PY'
import sys,torch
p=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
print("resume_completed_iteration="+str(p.get("completed_iteration")))
print("resume_target_iteration="+str(p.get("target_iteration")))
PY

set +e
/usr/bin/time -v "${PY}" tools/benchmark_3h_semantic_parallel_fit_10115_10315.py   --checkpoint "${SOURCE}"   --probe "${PROBE}"   --resume "${RESUME}"   --work-dir "${WORK}"   --report "${REPORT}"   2>&1 | tee "${WORK}/benchmark.log"
STATUS=${PIPESTATUS[0]}
set -e

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    [[ -f "${REPORT}" ]] && cp -f "${REPORT}" "${DWSL}/SpinCore_3H_semantic_parallel_fit_gate.json" || true
    [[ -f "${WORK}/benchmark.log" ]] && cp -f "${WORK}/benchmark.log" "${DWSL}/SpinCore_3H_semantic_parallel_fit_gate.log" || true
  fi
fi

if [[ "${STATUS}" -ne 0 ]]; then
  echo "SEMANTIC_PARALLEL_FIT_GATE_STOP status=${STATUS}" >&2
  echo "Send the JSON/report and benchmark.log to ChatGPT; do not resume with the parallel runner." >&2
  exit "${STATUS}"
fi

echo "SEMANTIC_PARALLEL_FIT_GATE_PASS"
echo "STOP HERE. Do not launch the parallel trainer until the stage manifest is promoted to READY from this exact gate evidence."
echo "report=${REPORT}"
