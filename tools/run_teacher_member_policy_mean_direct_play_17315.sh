#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: MODEL-020,MODEL-025,VALID-025,VALID-031,VALID-034,VALID-035,SAFE-001,SAFE-010,RNG-001,RNG-002,RNG-003

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"

PY="${ROOT}/.venv_lean/bin/python"
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"

MANIFEST="contracts/run_manifests/teacher_member_policy_mean_direct_play_17315.json"
RUNNER="tools/run_teacher_member_policy_mean_direct_play_17315.sh"
PREREG="contracts/teacher_member_policy_mean_direct_play_17315_v1.json"

RUN17315="${ROOT}/runs/3h_semantic_long_15315_17315"
TEACHER="${RUN17315}/semantic_long_ensemble_final_17315.pt"
TRAIN_REPORT="${RUN17315}/3h_semantic_long_15315_17315.json"
SOLVER="${ROOT}/build/libspincore_solver_c.so"

OUTDIR="${ROOT}/runs/teacher_member_policy_mean_direct_play_17315"
REPORT="${OUTDIR}/teacher_member_policy_mean_direct_play_17315.json"
mkdir -p "${OUTDIR}"

for f in "${PY}" "${MANIFEST}" "${PREREG}" "${TEACHER}" "${TRAIN_REPORT}" "${SOLVER}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing prerequisite: ${f}" >&2; exit 2; }
done

"${PY}" tools/check_stage_manifest.py --manifest "${MANIFEST}" --runner "${RUNNER}"

if pgrep -af "run_hu_semantic_ens8_.*\.py|run_3h_semantic_long_.*\.py" >/dev/null 2>&1; then
  echo "ERROR: a heavy SpinCore trainer is running; direct-play gate will not contend." >&2
  pgrep -af "run_hu_semantic_ens8_.*\.py|run_3h_semantic_long_.*\.py" >&2 || true
  exit 3
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source/index changes present." >&2
  git status --short --untracked-files=no >&2
  exit 4
fi

MEMAVAIL_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[[ -n "${MEMAVAIL_KIB}" && "${MEMAVAIL_KIB}" -ge $((12*1024*1024)) ]] || {
  echo "ERROR: less than 12 GiB WSL MemAvailable" >&2
  free -h >&2 || true
  exit 5
}

"${PY}" -m py_compile tools/validate_teacher_member_policy_mean_17315.py

echo "=== Teacher @17315 A vs B paired direct-play gate ==="
echo "scenarios=10000"
echo "seed=2026103017"
echo "A=mean(raw)->lean regret matching"
echo "B=member lean policies -> mean(policy)"

/usr/bin/time -v "${PY}" tools/validate_teacher_member_policy_mean_17315.py \
  --solver "${SOLVER}" \
  --teacher "${TEACHER}" \
  --training-report "${TRAIN_REPORT}" \
  --prereg "${PREREG}" \
  --scenarios 10000 \
  --workers 16 \
  --seed 2026103017 \
  --max-decisions 300 \
  --report "${REPORT}"

"${PY}" - "${REPORT}" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],"r",encoding="utf-8"))
assert d["schema"]=="SPINCORE_TEACHER_MEMBER_POLICY_MEAN_DIRECT_PLAY_17315_V1"
assert d["status"]=="PASS"
assert int(d["teacher_iteration"])==17315
assert int(d["scenarios"])==10000
assert d["production_change_authorized"] is False
assert d["long_training_unblocked"] is False
print("TEACHER_MEMBER_POLICY_MEAN_DIRECT_PLAY_17315_POSTVALIDATION_PASS")
print("decision="+d["decision"]["status"])
print("paired_B_minus_A="+json.dumps(d["paired_B_minus_A_spincore_chips_per_policy_seat_hand"],sort_keys=True))
print("checks="+json.dumps(d["decision"]["checks"],sort_keys=True))
PY

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${REPORT}" "${DWSL}/SpinCore_teacher_member_policy_mean_direct_play_17315.json"
    echo "desktop_report=${DWIN}\\SpinCore_teacher_member_policy_mean_direct_play_17315.json"
  fi
fi

echo
echo "SPINCORE_TEACHER_MEMBER_POLICY_MEAN_DIRECT_PLAY_17315_SHELL_PASS"
echo "STOP HERE. Send SpinCore_teacher_member_policy_mean_direct_play_17315.json to ChatGPT."
