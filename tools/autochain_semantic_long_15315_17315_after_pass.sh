#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: TRAIN-024,VALID-039,SAFE-002,GOV-024,CKPT-004

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTIVE_ROOT="${SPINCORE_ACTIVE_ROOT:-${HOME}/spincore_lean_functional}"
POLL_SECONDS="${SPINCORE_CHAIN_POLL_SECONDS:-120}"
CURRENT_RUN="${ACTIVE_ROOT}/runs/3h_semantic_long_10315_15315"
CURRENT_REPORT="${CURRENT_RUN}/3h_semantic_long_10315_15315.json"
CURRENT_RESUME="${CURRENT_RUN}/resume_state.pt"
CURRENT_ENSEMBLE="${CURRENT_RUN}/semantic_long_ensemble_final_15315.pt"
TRAVEL_MANIFEST="${CODE_ROOT}/contracts/run_manifests/teacher_travel_decision_10315_14815.json"
CONT_MANIFEST="${CODE_ROOT}/contracts/run_manifests/semantic_long_15315_17315.json"
LOG="${ACTIVE_ROOT}/runs/travel_week_autochain_15315_17315.log"

mkdir -p "$(dirname "${LOG}")"

exec > >(tee -a "${LOG}") 2>&1

echo "TRAVEL_AUTOCHAIN_ARMED at=$(date --iso-8601=seconds)"
echo "active_root=${ACTIVE_ROOT}"
echo "poll_seconds=${POLL_SECONDS}"

python3 - "${TRAVEL_MANIFEST}" "${CONT_MANIFEST}" <<'PY'
import json,sys
travel=json.load(open(sys.argv[1],"r",encoding="utf-8"))
cont=json.load(open(sys.argv[2],"r",encoding="utf-8"))
assert travel["status"]=="COMPLETE_PASS"
assert travel["validation_result"]["travel_week_decision"]=="CONTINUE_AFTER_15315"
assert float(travel["validation_result"]["paired_14815_minus_10315_spincore_chip_ev"]["mean"])>0.0
assert cont["status"]=="READY"
assert cont["scientific_plan"]["start_iteration"]==15315
assert cont["scientific_plan"]["target_iteration"]==17315
print("TRAVEL_AUTOCHAIN_DECISION_PASS")
PY

while pgrep -af 'run_3h_semantic_long_10315_15315' >/dev/null 2>&1; do
  echo "TRAVEL_AUTOCHAIN_WAIT current_trainer_active at=$(date --iso-8601=seconds)"
  sleep "${POLL_SECONDS}"
done

# Give the parent wrapper time to flush/postvalidate after the Python trainer exits.
sleep 30

for f in "${CURRENT_REPORT}" "${CURRENT_RESUME}" "${CURRENT_ENSEMBLE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "TRAVEL_AUTOCHAIN_STOP missing_source=${f}" >&2
    exit 20
  fi
done

python3 - "${CURRENT_REPORT}" "${CURRENT_RESUME}" "${CURRENT_ENSEMBLE}" <<'PY'
import hashlib,json,sys
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):
            h.update(b)
    return h.hexdigest()
r=json.load(open(sys.argv[1],"r",encoding="utf-8"))
assert r["status"]=="PASS", r.get("status")
assert int(r["completed_iteration"])==15315
assert r["final_safety_guard_pass"] is True
assert r["source_10105_mutated"] is False
assert r["hu_training_performed"] is False
assert r["resume_state_sha256"]==sha(sys.argv[2])
assert r["final_ensemble_sha256"]==sha(sys.argv[3])
print("TRAVEL_AUTOCHAIN_15315_SOURCE_PASS")
print("source_resume_sha256="+r["resume_state_sha256"])
print("source_final_ensemble_sha256="+r["final_ensemble_sha256"])
PY

cd "${ACTIVE_ROOT}"
branch="$(git branch --show-current)"
if [[ "${branch}" != "main" ]]; then
  echo "TRAVEL_AUTOCHAIN_STOP active_repo_not_main branch=${branch}" >&2
  exit 21
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "TRAVEL_AUTOCHAIN_STOP tracked_changes_present" >&2
  git status --short --untracked-files=no >&2
  exit 22
fi

git fetch origin main
git pull --ff-only origin main

python3 tools/check_stage_manifest.py \
  --manifest contracts/run_manifests/semantic_long_15315_17315.json \
  --runner tools/run_3h_semantic_long_15315_17315.sh

echo "TRAVEL_AUTOCHAIN_LAUNCH_15315_17315 at=$(date --iso-8601=seconds)"
exec bash tools/run_3h_semantic_long_15315_17315.sh
