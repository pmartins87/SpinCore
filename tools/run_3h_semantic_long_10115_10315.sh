#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"

PY="${ROOT}/.venv_lean/bin/python"
SOLVER="${ROOT}/build/libspincore_solver_c.so"

SOURCE="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE}/checkpoint.pt"
HU="${SOURCE}/hu_ensemble_state.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_HU="8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"

CANON="${ROOT}/runs/3h_semantic_research_canonical_10105_10115"
CANON_REPORT="${CANON}/3h_semantic_research_10105_10115.json"
SOURCE_RESUME="${CANON}/resume_state.pt"
SOURCE_FINAL_ENSEMBLE="${CANON}/semantic_ensemble_final_10115.pt"
SPIN_BUNDLE="${CANON}/spincore_hybrid_10105.pt"

ADV_RUN="$(find "${ROOT}/runs/3h_v1_semantic_shadow" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
PROBE_RUN="$(find "${ROOT}/runs/3h_advantage_controlled_budget_curve" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
ADV_MODELS="${ADV_RUN}/3h_v1_semantic_shadow_models.pt"
PROBE="${PROBE_RUN}/3h_advantage_controlled_split_probe.pt"

FIVEK_RUN="$(find "${ROOT}/runs/deepcrusher_dc1_stratified_specialist_5k" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
FIVEK_COMPARE="${FIVEK_RUN}/stratified_specialist_5k_comparison.json"

for f in   "${PY}" "${CHECKPOINT}" "${HU}"   "${CANON_REPORT}" "${SOURCE_RESUME}" "${SOURCE_FINAL_ENSEMBLE}" "${SPIN_BUNDLE}"   "${ADV_MODELS}" "${PROBE}" "${FIVEK_COMPARE}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing prerequisite: ${f}" >&2; exit 2; }
done

[[ "$(sha256sum "${CHECKPOINT}"|awk '{print $1}')" == "${EXPECTED_CP}" ]] || {
  echo "ERROR: frozen 10105 checkpoint SHA mismatch" >&2; exit 3;
}
[[ "$(sha256sum "${HU}"|awk '{print $1}')" == "${EXPECTED_HU}" ]] || {
  echo "ERROR: frozen HU sidecar SHA mismatch" >&2; exit 4;
}

export PYTHONPATH="${ROOT}/python:${ROOT}/tools"

"${PY}" -   "${CANON_REPORT}" "${SOURCE_RESUME}" "${SOURCE_FINAL_ENSEMBLE}"   "${FIVEK_COMPARE}" "${EXPECTED_CP}" <<'PY'
import json,sys,torch
from pathlib import Path

canon=Path(sys.argv[1])
resume=Path(sys.argv[2])
ens=Path(sys.argv[3])
fivek=Path(sys.argv[4])
expected=sys.argv[5]

c=json.loads(canon.read_text())
assert c["status"]=="PASS"
assert c["semantic_research_continuation_pass"] is True
assert c["canonical_replay"] is True
assert c["behavior_contract"]=="CANONICAL_LEAN_REGRET_MATCHING_SOFTMAX_FALLBACK"
assert c["completed_iteration"]==10115
assert c["source_checkpoint_sha256"]==expected
assert c["source_checkpoint_mutated"] is False
assert c["hu_training_performed"] is False

r=torch.load(resume,map_location="cpu",weights_only=False)
e=torch.load(ens,map_location="cpu",weights_only=False)
assert r["schema"]=="SPINCORE_3H_SEMANTIC_RESEARCH_RESUME_V1"
assert r["completed_iteration"]==10115
assert r["source_checkpoint_sha256"]==expected
assert e["schema"]=="SPINCORE_3H_SEMANTIC_RESEARCH_ENSEMBLE_V1"
assert e["completed_iteration"]==10115
assert e["source_checkpoint_sha256"]==expected
assert len(r["semantic_members"])==len(e["members"])==8
for a,b in zip(r["semantic_members"],e["members"]):
    assert set(a)==set(b)
    for k in a:
        assert torch.equal(a[k].cpu(),b[k].cpu())

f=json.loads(fivek.read_text())
assert f["schema"]=="SPINCORE_DC1_STRATIFIED_SPECIALIST_5K_COMPARISON_V1"
assert f["seed"]==20260929
assert f["scenarios"]==5000
assert f["fresh_seed_relative_to_prior_1k"] is True
assert f["dc1_5k_long_train_readiness_pass"] is True
assert all(f["precommitted_long_train_readiness_criteria"].values())

print("SEMANTIC_LONG_10115_10315_GATE_CONTRACT_PASS")
PY

if pgrep -af "run_3h_semantic_long_10115_10315.py|run_3h_semantic_research_10105_10115.py|run_lt3_parallel_9105_10105.py" >/dev/null 2>&1; then
  echo "ERROR: a long SpinCore trainer is already running." >&2
  pgrep -af "run_3h_semantic_long_10115_10315.py|run_3h_semantic_research_10105_10115.py|run_lt3_parallel_9105_10105.py" >&2 || true
  exit 7
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source/index changes present; long run requires code == HEAD." >&2
  git status --short --untracked-files=no >&2
  exit 8
fi

MEMTOTAL_KIB="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
MEMAVAIL_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if [[ -z "${MEMTOTAL_KIB}" || "${MEMTOTAL_KIB}" -lt $((28*1024*1024)) ]]; then
  echo "ERROR: WSL exposes less than 28 GiB RAM." >&2
  exit 9
fi
if [[ -z "${MEMAVAIL_KIB}" || "${MEMAVAIL_KIB}" -lt $((16*1024*1024)) ]]; then
  echo "ERROR: less than 16 GiB WSL MemAvailable." >&2
  free -h >&2 || true
  exit 10
fi

DISK_KIB="$(df -Pk "${ROOT}" | awk 'NR==2 {print $4}')"
if [[ -z "${DISK_KIB}" || "${DISK_KIB}" -lt $((15*1024*1024)) ]]; then
  echo "ERROR: less than 15 GiB free on training filesystem." >&2
  exit 11
fi

HOST_C_KIB="$(df -Pk /mnt/c 2>/dev/null | awk 'NR==2 {print $4}' || true)"
if [[ -n "${HOST_C_KIB}" && "${HOST_C_KIB}" -lt $((20*1024*1024)) ]]; then
  echo "ERROR: less than 20 GiB free on Windows C: host filesystem." >&2
  exit 12
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c

"${PY}" -m py_compile   tools/run_3h_semantic_long_10115_10315.py   tools/run_3h_semantic_research_10105_10115.py   tools/run_3h_semantic_online_pilot_10105.py   tools/audit_3h_fresh_semantic_strategy_distill_10105.py

"${PY}" python_tests/test_reservoir_write_observer.py
"${PY}" python_tests/test_lean_action_policy_ensemble.py

echo "SEMANTIC_LONG_10115_10315_PREFLIGHT_PASS"

RUN="${ROOT}/runs/3h_semantic_long_10115_10315"
mkdir -p "${RUN}"
REPORT="${RUN}/3h_semantic_long_10115_10315.json"
LOG="${RUN}/training.log"
MEMLOG="${RUN}/memory.log"

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
    sleep 60
  done
}

copy_evidence() {
  if ! command -v powershell.exe >/dev/null 2>&1; then
    return 0
  fi
  local dwin dwsl
  dwin="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  dwsl="$(wslpath -u "${dwin}" 2>/dev/null || true)"
  [[ -n "${dwsl}" && -d "${dwsl}" ]] || return 0
  [[ -f "${REPORT}" ]] && cp -f "${REPORT}" "${dwsl}/SpinCore_3H_semantic_long_10115_10315.json" || true
  [[ -f "${MEMLOG}" ]] && cp -f "${MEMLOG}" "${dwsl}/SpinCore_3H_semantic_long_10115_10315_memory.log" || true
  echo "desktop_report=${dwin}\\SpinCore_3H_semantic_long_10115_10315.json"
}

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "=== SpinCore 3H semantic long continuation ==="
echo "source_semantic_iteration=10115"
echo "target_semantic_iteration=10315"
echo "additional_iterations=200"
echo "roots_per_iteration=64"
echo "expected_new_roots=12800"
echo "ensemble=8 members x 1600 fresh-fit steps/iteration"
echo "behavior=canonical lean regret matching + softmax fallback"
echo "HU=frozen at 10105; no HU training in this lane"
echo "milestones=10165,10215,10265,10315"
echo "projected_duration_from_10105->10115 measured rate≈27 h"
echo "final AveragePolicy/specialist rebuild happens only AFTER this run"
echo "run_dir=${RUN}"
free -h
echo

snapshot_memory before

set +e
(
  /usr/bin/time -v "${PY}" tools/run_3h_semantic_long_10115_10315.py     --checkpoint "${CHECKPOINT}"     --semantic-advantage-models "${ADV_MODELS}"     --probe "${PROBE}"     --solver "${SOLVER}"     --spin-bundle "${SPIN_BUNDLE}"     --source-resume "${SOURCE_RESUME}"     --source-final-ensemble "${SOURCE_FINAL_ENSEMBLE}"     --run-dir "${RUN}"     --threads 8     2>&1 | tee -a "${LOG}"
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

[[ "$(sha256sum "${CHECKPOINT}"|awk '{print $1}')" == "${EXPECTED_CP}" ]] || {
  copy_evidence
  echo "ERROR: frozen 10105 checkpoint mutated" >&2
  exit 13
}
[[ "$(sha256sum "${HU}"|awk '{print $1}')" == "${EXPECTED_HU}" ]] || {
  copy_evidence
  echo "ERROR: frozen HU sidecar mutated" >&2
  exit 14
}

copy_evidence

if [[ "${STATUS}" -ne 0 ]]; then
  echo "SEMANTIC_LONG_10115_10315_TRAINING_STOP status=${STATUS}" >&2
  echo "run_dir=${RUN}" >&2
  echo "Do not restart blindly. Send training.log/report to ChatGPT." >&2
  exit "${STATUS}"
fi

"${PY}" - "${REPORT}" "${EXPECTED_CP}" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
expected=sys.argv[2]
d=json.loads(p.read_text())
assert d["schema"]=="SPINCORE_3H_SEMANTIC_LONG_CONTINUATION_10115_10315_V1"
assert d["status"]=="PASS"
assert d["source_checkpoint_sha256"]==expected
assert d["source_semantic_iteration"]==10115
assert d["target_iteration"]==10315
assert d["completed_iteration"]==10315
assert d["additional_iterations"]==200
assert d["expected_new_roots"]==12800
assert d["new_roots_recorded"]==12800
assert d["behavior_contract"]=="CANONICAL_LEAN_REGRET_MATCHING_SOFTMAX_FALLBACK"
assert d["source_10105_mutated"] is False
assert d["hu_training_performed"] is False
assert d["final_safety_guard_pass"] is True
assert all(d["final_safety_guard"].values())
print("SEMANTIC_LONG_10115_10315_POSTVALIDATION_PASS")
print("final_ensemble_sha256="+d["final_ensemble_sha256"])
print("resume_state_sha256="+d["resume_state_sha256"])
print("wall_hours="+str(d["wall_seconds_this_invocation"]/3600.0))
PY

copy_evidence

echo
echo "SEMANTIC_LONG_10115_10315_TRAINING_PASS"
echo "STOP HERE. Send SpinCore_3H_semantic_long_10115_10315.json to ChatGPT."
