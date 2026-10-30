#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: MODEL-020,MODEL-025,VALID-025,VALID-034,VALID-035,SAFE-001,SAFE-010,RNG-001,RNG-003

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"

PY="${ROOT}/.venv_lean/bin/python"
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"

MANIFEST="contracts/run_manifests/teacher_architecture_benchmark_17315.json"
RUNNER="tools/run_teacher_architecture_benchmark_17315.sh"
PREREG="contracts/teacher_architecture_benchmark_17315_v1.json"

RUN17315="${ROOT}/runs/3h_semantic_long_15315_17315"
RESUME="${RUN17315}/resume_state.pt"
ENSEMBLE="${RUN17315}/semantic_long_ensemble_final_17315.pt"
TRAIN_REPORT="${RUN17315}/3h_semantic_long_15315_17315.json"
RAW_REPORT="${ROOT}/runs/teacher_raw_advantage_17315/teacher_raw_advantage_17315.json"

PROBE_RUN="$(find "${ROOT}/runs/3h_advantage_controlled_budget_curve" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
PROBE="${PROBE_RUN}/3h_advantage_controlled_split_probe.pt"

OUTDIR="${ROOT}/runs/teacher_architecture_benchmark_17315"
REPORT="${OUTDIR}/teacher_architecture_benchmark_17315.json"
mkdir -p "${OUTDIR}"

for f in "${PY}" "${MANIFEST}" "${PREREG}" "${RESUME}" "${ENSEMBLE}" "${TRAIN_REPORT}" "${RAW_REPORT}" "${PROBE}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing prerequisite: ${f}" >&2; exit 2; }
done

"${PY}" tools/check_stage_manifest.py --manifest "${MANIFEST}" --runner "${RUNNER}"

if pgrep -af "run_hu_semantic_ens8_.*\.py|run_3h_semantic_long_.*\.py" >/dev/null 2>&1; then
  echo "ERROR: a heavy SpinCore trainer is running; architecture benchmark will not contend for RAM/CPU." >&2
  pgrep -af "run_hu_semantic_ens8_.*\.py|run_3h_semantic_long_.*\.py" >&2 || true
  exit 3
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source/index changes present." >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

MEMAVAIL_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[[ -n "${MEMAVAIL_KIB}" && "${MEMAVAIL_KIB}" -ge $((18*1024*1024)) ]] || {
  echo "ERROR: less than 18 GiB WSL MemAvailable" >&2
  free -h >&2 || true
  exit 5
}

"${PY}" -m py_compile tools/benchmark_teacher_architecture_17315.py

echo "=== Teacher @17315 aggregation/fallback architecture benchmark ==="
echo "protected_holdout=50000"
echo "candidates=A_CURRENT_RAW_MEAN_RM,B_MEMBER_POLICY_MEAN,C_MEDIAN_RAW_RM,D_TRIMMED_RAW_MEAN_RM"

/usr/bin/time -v "${PY}" tools/benchmark_teacher_architecture_17315.py \
  --resume "${RESUME}" \
  --ensemble "${ENSEMBLE}" \
  --training-report "${TRAIN_REPORT}" \
  --probe "${PROBE}" \
  --raw-report "${RAW_REPORT}" \
  --prereg "${PREREG}" \
  --out "${REPORT}" \
  --batch-size 512

"${PY}" - "${REPORT}" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],"r",encoding="utf-8"))
assert d["schema"]=="SPINCORE_TEACHER_ARCHITECTURE_BENCHMARK_17315_V1"
assert d["status"]=="PASS"
assert int(d["teacher_iteration"])==17315
assert int(d["protected_holdout_count"])==50000
assert d["production_change_authorized"] is False
print("TEACHER_ARCHITECTURE_BENCHMARK_17315_POSTVALIDATION_PASS")
print("classification="+d["decision"]["classification"])
print("passing_candidates="+json.dumps(d["decision"]["passing_candidates"]))
PY

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${REPORT}" "${DWSL}/SpinCore_teacher_architecture_benchmark_17315.json"
    echo "desktop_report=${DWIN}\\SpinCore_teacher_architecture_benchmark_17315.json"
  fi
fi

echo
echo "SPINCORE_TEACHER_ARCHITECTURE_BENCHMARK_17315_SHELL_PASS"
echo "STOP HERE. Send SpinCore_teacher_architecture_benchmark_17315.json to ChatGPT."
