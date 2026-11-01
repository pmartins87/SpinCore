#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: MODEL-020,MODEL-025,VALID-025,VALID-031,VALID-034,VALID-035,SAFE-001,SAFE-010,RNG-001,RNG-002,RNG-003

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SPINCORE_TORCH_THREADS=1

MANIFEST="contracts/run_manifests/reference_fallback_direct_play_17315.json"
PREREG="contracts/reference_fallback_direct_play_17315_v1.json"
RUNNER="tools/run_reference_fallback_direct_play_17315.sh"
RUN17315="${ROOT}/runs/3h_semantic_long_15315_17315"
TEACHER="${RUN17315}/semantic_long_ensemble_final_17315.pt"
TRAIN_REPORT="${RUN17315}/3h_semantic_long_15315_17315.json"
SOLVER="${ROOT}/build/libspincore_solver_c.so"
OUTDIR="${ROOT}/runs/reference_fallback_direct_play_17315"
REPORT="${OUTDIR}/reference_fallback_direct_play_17315.json"

for f in "${PY}" "${MANIFEST}" "${PREREG}" "${TEACHER}" "${TRAIN_REPORT}" "${SOLVER}"; do
  [[ -e "${f}" ]] || { echo "G3_PREREQUISITE_MISSING ${f}" >&2; exit 2; }
done

"${PY}" tools/check_stage_manifest.py --manifest "${MANIFEST}" --runner "${RUNNER}"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "G3_SOURCE_DIRTY: refusing to run" >&2
  git status --short --untracked-files=no >&2
  exit 3
fi
if pgrep -af "run_hu_semantic_ens8_.*\\.py|run_3h_semantic_long_.*\\.py" >/dev/null 2>&1; then
  echo "G3_HEAVY_TRAINER_RUNNING" >&2
  exit 4
fi
MEMAVAIL_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if [[ -z "${MEMAVAIL_KIB}" || "${MEMAVAIL_KIB}" -lt $((12*1024*1024)) ]]; then
  echo "G3_INSUFFICIENT_WSL_MEMORY: need 12 GiB available" >&2
  free -h >&2
  exit 5
fi
"${PY}" -m py_compile tools/validate_reference_fallback_direct_play_17315.py
"${PY}" tools/validate_reference_fallback_direct_play_17315.py --self-test

echo "=== G3 reference fallback smoke (four paired scenarios) ==="
"${PY}" tools/validate_reference_fallback_direct_play_17315.py \
  --solver "${SOLVER}" --teacher "${TEACHER}" \
  --training-report "${TRAIN_REPORT}" --prereg "${PREREG}" \
  --smoke-only

echo "=== G3 fresh paired direct play (10000 scenarios; same weights) ==="
mkdir -p "${OUTDIR}"
/usr/bin/time -v "${PY}" tools/validate_reference_fallback_direct_play_17315.py \
  --solver "${SOLVER}" --teacher "${TEACHER}" \
  --training-report "${TRAIN_REPORT}" --prereg "${PREREG}" \
  --seed 2026110103 --scenarios 10000 --workers 16 --max-decisions 300 \
  --report "${REPORT}"

"${PY}" - "${REPORT}" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding="utf-8"))
assert r["schema"]=="SPINCORE_REFERENCE_FALLBACK_DIRECT_PLAY_17315_V1"
assert r["status"]=="PASS"
assert r["scenarios"]==10000
assert r["decision"]["status"] in ("PASS","FAIL")
assert r["production_change_authorized"] is False
assert r["long_training_unblocked"] is False
print("G3_REFERENCE_FALLBACK_POSTVALIDATION_PASS")
print("decision="+r["decision"]["status"])
PY

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${REPORT}" "${DWSL}/SpinCore_reference_fallback_direct_play_17315.json"
    echo "desktop_report=${DWIN}\\SpinCore_reference_fallback_direct_play_17315.json"
  fi
fi

echo "SPINCORE_REFERENCE_FALLBACK_DIRECT_PLAY_17315_SHELL_PASS"
echo "STOP. Send SpinCore_reference_fallback_direct_play_17315.json to ChatGPT."
