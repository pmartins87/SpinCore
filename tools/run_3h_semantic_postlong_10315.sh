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

LONG="${ROOT}/runs/3h_semantic_long_10115_10315"
LONG_REPORT="${LONG}/3h_semantic_long_10115_10315.json"
ADV="${LONG}/semantic_long_ensemble_final_10315.pt"

for f in "${PY}" "${CHECKPOINT}" "${HU}" "${LONG_REPORT}" "${ADV}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing prerequisite: ${f}" >&2; exit 2; }
done

[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${EXPECTED_CP}" ]] || {
  echo "ERROR: frozen 10105 checkpoint SHA mismatch" >&2; exit 3;
}
[[ "$(sha256sum "${HU}" | awk '{print $1}')" == "${EXPECTED_HU}" ]] || {
  echo "ERROR: frozen HU sidecar SHA mismatch" >&2; exit 4;
}

export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

"${PY}" - "${LONG_REPORT}" "${ADV}" "${EXPECTED_CP}" <<'PY'
import hashlib,json,sys,torch
from pathlib import Path

report=Path(sys.argv[1]); adv=Path(sys.argv[2]); expected=sys.argv[3]
r=json.loads(report.read_text())
assert r["schema"]=="SPINCORE_3H_SEMANTIC_LONG_CONTINUATION_10115_10315_V1"
assert r["status"]=="PASS"
assert r["source_checkpoint_sha256"]==expected
assert r["source_semantic_iteration"]==10115
assert r["target_iteration"]==10315
assert r["completed_iteration"]==10315
assert r["new_roots_recorded"]==12800
assert r["behavior_contract"]=="CANONICAL_LEAN_REGRET_MATCHING_SOFTMAX_FALLBACK"
assert r["source_10105_mutated"] is False
assert r["hu_training_performed"] is False
assert r["final_safety_guard_pass"] is True
assert all(r["final_safety_guard"].values())

p=torch.load(adv,map_location="cpu",weights_only=False)
assert p["schema"]=="SPINCORE_3H_SEMANTIC_RESEARCH_ENSEMBLE_V1"
assert p["source_checkpoint_sha256"]==expected
assert int(p["completed_iteration"])==10315
assert len(p["members"])==8
h=hashlib.sha256(adv.read_bytes()).hexdigest()
assert h==r["final_ensemble_sha256"]
print("POSTLONG_10315_SOURCE_GATE_PASS")
print("teacher_sha256="+h)
PY

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c

"${PY}" -m py_compile \
  tools/build_3h_semantic_postlong_policy_10315.py \
  tools/build_3h_semantic_stratified_specialist_10315.py \
  tools/audit_3h_semantic_stratified_specialist_multiseed_10315.py

RUN="${ROOT}/runs/3h_semantic_postlong_10315"
mkdir -p "${RUN}"

POLICY_REPORT="${RUN}/postlong_policy_rebuild_10315.json"
TAIL="${RUN}/semantic_tail_policy_10315.pt"
FULLPOOL="${RUN}/semantic_fullpool_tail_10315.pt"

SPECIALIST_REPORT="${RUN}/postlong_stratified_specialist_build_10315.json"
SPECIALIST="${RUN}/semantic_stratified_specialist_10315.pt"

MULTISEED_REPORT="${RUN}/postlong_stratified_specialist_multiseed_10315.json"

echo "POSTLONG_10315_POLICY_REBUILD_START"
"${PY}" tools/build_3h_semantic_postlong_policy_10315.py \
  --checkpoint "${CHECKPOINT}" \
  --semantic-advantage "${ADV}" \
  --solver "${SOLVER}" \
  --report "${POLICY_REPORT}" \
  --out-tail "${TAIL}" \
  --out-fullpool "${FULLPOOL}" \
  --threads 8

POLICY_PASS="@("${PY}" - "${POLICY_REPORT}" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print("1" if r.get("postlong_policy_rebuild_pass") else "0")
PY
)"
if [[ "${POLICY_PASS}" != "1" ]]; then
  echo "POSTLONG_10315_POLICY_REBUILD_FAILED_STOP"
  exit 5
fi
echo "POSTLONG_10315_POLICY_REBUILD_PASS"

echo "POSTLONG_10315_SPECIALIST_BUILD_START"
"${PY}" tools/build_3h_semantic_stratified_specialist_10315.py \
  --checkpoint "${CHECKPOINT}" \
  --semantic-advantage "${ADV}" \
  --fullpool-tail "${FULLPOOL}" \
  --solver "${SOLVER}" \
  --report "${SPECIALIST_REPORT}" \
  --out-model "${SPECIALIST}" \
  --threads 8
echo "POSTLONG_10315_SPECIALIST_BUILD_PASS"

echo "POSTLONG_10315_MULTISEED_START"
"${PY}" tools/audit_3h_semantic_stratified_specialist_multiseed_10315.py \
  --checkpoint "${CHECKPOINT}" \
  --semantic-advantage "${ADV}" \
  --stratified-specialist "${SPECIALIST}" \
  --solver "${SOLVER}" \
  --report "${MULTISEED_REPORT}" \
  --threads 8

MULTISEED_PASS="@("${PY}" - "${MULTISEED_REPORT}" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print("1" if r.get("semantic_stratified_specialist_multiseed_pass") else "0")
PY
)"
if [[ "${MULTISEED_PASS}" != "1" ]]; then
  echo "POSTLONG_10315_MULTISEED_FAILED_STOP"
  exit 6
fi

"${PY}" - "${RUN}" "${LONG_REPORT}" "${POLICY_REPORT}" "${SPECIALIST_REPORT}" "${MULTISEED_REPORT}" <<'PY'
from pathlib import Path
import sys,zipfile
run=Path(sys.argv[1])
sources=[Path(x) for x in sys.argv[2:]]
out=run/"SpinCore_3H_postlong_10315_validation_bundle.zip"
with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for p in sources:
        if not p.is_file():
            raise RuntimeError(f"missing bundle source {p}")
        z.write(p,arcname=p.name)
print(f"bundle={out}")
PY

ZIP="${RUN}/SpinCore_3H_postlong_10315_validation_bundle.zip"
if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${ZIP}" "${DWSL}/SpinCore_3H_postlong_10315_validation_bundle.zip"
    cp -f "${POLICY_REPORT}" "${DWSL}/SpinCore_3H_postlong_policy_rebuild_10315.json"
    cp -f "${MULTISEED_REPORT}" "${DWSL}/SpinCore_3H_postlong_multiseed_10315.json"
    echo "desktop_bundle=${DWIN}\\SpinCore_3H_postlong_10315_validation_bundle.zip"
  fi
fi

echo
echo "3H_SEMANTIC_POSTLONG_10315_VALIDATION_PASS"
echo "STOP HERE. Send SpinCore_3H_postlong_10315_validation_bundle.zip to ChatGPT."
