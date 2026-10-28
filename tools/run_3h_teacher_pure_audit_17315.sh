#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: DC-002,SAFE-001,SAFE-011,VALID-025,VALID-031,VALID-040,RNG-001,RNG-002,RNG-003

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"

PY="${ROOT}/.venv_lean/bin/python"
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"

MANIFEST="contracts/run_manifests/teacher_pure_audit_17315.json"
RUNNER="tools/run_3h_teacher_pure_audit_17315.sh"

SANITY="runs/teacher_sanity_17315/teacher_sanity_17315.json"
RAW="runs/teacher_raw_advantage_17315/teacher_raw_advantage_17315.json"
TREND="runs/teacher_flagged_trend_17315/teacher_flagged_checkpoint_trend_17315.json"
OUTDIR="runs/teacher_pure_audit_17315"
SUMMARY="${OUTDIR}/teacher_pure_audit_17315_summary.json"

mkdir -p "${OUTDIR}"

for f in "${PY}" "${MANIFEST}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing prerequisite: ${f}" >&2; exit 2; }
done

"${PY}" tools/check_stage_manifest.py --manifest "${MANIFEST}" --runner "${RUNNER}"

if pgrep -af "run_hu_semantic_ens8_.*\.py|run_3h_semantic_long_.*\.py" >/dev/null 2>&1; then
  echo "ERROR: a heavy SpinCore trainer is still running; pure-teacher audit will not contend for CPU." >&2
  pgrep -af "run_hu_semantic_ens8_.*\.py|run_3h_semantic_long_.*\.py" >&2 || true
  exit 3
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source/index changes present." >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

MEMAVAIL_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[[ -n "${MEMAVAIL_KIB}" && "${MEMAVAIL_KIB}" -ge $((8*1024*1024)) ]] || {
  echo "ERROR: less than 8 GiB WSL MemAvailable" >&2
  free -h >&2 || true
  exit 5
}

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c

"${PY}" -m py_compile \
  tools/audit_3h_teacher_sanity_17315.py \
  tools/diagnose_3h_teacher_raw_advantages_17315.py \
  tools/diagnose_3h_teacher_flagged_checkpoint_trend.py \
  tools/summarize_3h_teacher_pure_audit_17315.py

echo "=== Pure teacher 17315 audit: stage 1/3 sanity + full policy surfaces ==="
bash tools/run_3h_teacher_sanity_17315.sh

echo
echo "=== Pure teacher 17315 audit: stage 2/3 raw-Advantage mechanism + checkpoint trend ==="
bash tools/run_3h_teacher_training_efficiency_17315.sh

for f in "${SANITY}" "${RAW}" "${TREND}"; do
  [[ -s "${f}" ]] || { echo "ERROR: expected audit output missing: ${f}" >&2; exit 6; }
done

echo
echo "=== Pure teacher 17315 audit: stage 3/3 consolidated summary ==="
"${PY}" tools/summarize_3h_teacher_pure_audit_17315.py \
  --sanity "${SANITY}" \
  --raw "${RAW}" \
  --trend "${TREND}" \
  --out "${SUMMARY}"

"${PY}" - "${SUMMARY}" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],"r",encoding="utf-8"))
assert d["schema"]=="SPINCORE_3H_TEACHER_PURE_AUDIT_17315_SUMMARY_V1"
assert d["status"]=="COMPLETE"
assert int(d["teacher_iteration"])==17315
assert int(d["scenarios"])==10000
print("TEACHER_PURE_AUDIT_17315_POSTVALIDATION_PASS")
print("sanity_status="+str(d["sanity_status"]))
print("sampled_flag_counts="+json.dumps(d["sampled_flag_counts"],sort_keys=True))
print("opportunity_policy="+json.dumps(d["opportunity_policy"],sort_keys=True))
print("raw_mechanism_by_reason="+json.dumps(d["raw_mechanism_by_reason"],sort_keys=True))
PY

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${SUMMARY}" "${DWSL}/SpinCore_teacher_pure_audit_17315_summary.json"
    echo "desktop_summary=${DWIN}\\SpinCore_teacher_pure_audit_17315_summary.json"
  fi
fi

echo
echo "SPINCORE_3H_TEACHER_PURE_AUDIT_17315_PASS"
echo "STOP HERE. Send SpinCore_teacher_pure_audit_17315_summary.json to ChatGPT."
