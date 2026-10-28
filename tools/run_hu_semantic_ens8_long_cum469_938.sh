#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: TRAIN-022,TRAIN-023,MODEL-025,RNG-001,RNG-003,VALID-025,VALID-034,VALID-035,SAFE-001,SAFE-010,CKPT-001,ART-001,PERF-001,PERF-002

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"

MANIFEST="contracts/run_manifests/hu_semantic_long_cum469_938.json"
RUNNER="tools/run_hu_semantic_ens8_long_cum469_938.sh"
SOURCE_DIR="runs/hu_semantic_ens8_continuation_10105_plus469"
SOURCE_RESUME="${SOURCE_DIR}/resume_state.pt"
SOURCE_ENSEMBLE="${SOURCE_DIR}/hu_semantic_ensemble_final_plus469.pt"
SOURCE_REPORT="${SOURCE_DIR}/hu_semantic_ens8_continuation_10105_plus469.json"
CHECKPOINT="runs/lt3_parallel_9105_10105/20260922_132104/checkpoint.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
SOLVER="build/libspincore_solver_c.so"
OUT="runs/hu_semantic_ens8_long_cum469_938"

for f in "${PY}" "${MANIFEST}" "${CHECKPOINT}" "${SOURCE_RESUME}" "${SOURCE_ENSEMBLE}" "${SOURCE_REPORT}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing prerequisite: ${f}" >&2; exit 2; }
done

"${PY}" tools/check_stage_manifest.py --manifest "${MANIFEST}" --runner "${RUNNER}"

[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${EXPECTED_CP}" ]] || {
  echo "ERROR: frozen 10105 checkpoint SHA mismatch" >&2; exit 3;
}

if pgrep -af "run_3h_semantic_long_.*\.py|run_hu_semantic_ens8_.*\.py" >/dev/null 2>&1; then
  echo "ERROR: another SpinCore semantic trainer is already running." >&2
  pgrep -af "run_3h_semantic_long_.*\.py|run_hu_semantic_ens8_.*\.py" >&2 || true
  exit 4
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source/index changes present." >&2
  git status --short --untracked-files=no >&2
  exit 5
fi

"${PY}" - "${SOURCE_RESUME}" "${SOURCE_ENSEMBLE}" "${SOURCE_REPORT}" <<'PY'
import hashlib,json,sys,torch
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):
            h.update(b)
    return h.hexdigest()
resume=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
ens=torch.load(sys.argv[2],map_location="cpu",weights_only=False)
report=json.load(open(sys.argv[3],"r",encoding="utf-8"))
assert resume["schema"]=="SPINCORE_HU_SEMANTIC_ENS8_CONTINUATION_RESUME_V1"
assert int(resume["completed_updates"])==469
assert int(resume["target_updates"])==469
assert ens["schema"]=="SPINCORE_HU_SEMANTIC_ENS8_V1"
assert int(ens["completed_updates"])==469
assert report["status"]=="PASS"
assert int(report["completed_updates"])==469
assert int(report["new_hu_roots"])==128037
assert report["resume_state_sha256"]==sha(sys.argv[1])
assert report["final_semantic_ensemble_sha256"]==sha(sys.argv[2])
assert report["three_handed_training_performed"] is False
assert report["average_policy_training_performed"] is False
print("HU_SEMANTIC_CUM469_SOURCE_PASS")
print("source_resume_sha256="+report["resume_state_sha256"])
print("source_ensemble_sha256="+report["final_semantic_ensemble_sha256"])
PY

MEMAVAIL_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[[ -n "${MEMAVAIL_KIB}" && "${MEMAVAIL_KIB}" -ge $((12*1024*1024)) ]] || {
  echo "ERROR: less than 12 GiB WSL MemAvailable" >&2
  free -h >&2 || true
  exit 6
}

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c

"${PY}" -m py_compile \
  tools/run_hu_semantic_ens8_long_block.py \
  tools/run_hu_semantic_ens8_continuation_10105_plus469.py \
  python/spincore/semantic_parallel.py \
  python/spincore_nn/semantic_v1.py \
  tools/lt3_3h_semantic_parallel_fit.py

mkdir -p "${OUT}"

/usr/bin/time -v "${PY}" tools/run_hu_semantic_ens8_long_block.py \
  --checkpoint "${CHECKPOINT}" \
  --source-resume "${SOURCE_RESUME}" \
  --source-ensemble "${SOURCE_ENSEMBLE}" \
  --source-report "${SOURCE_REPORT}" \
  --start-cumulative-updates 469 \
  --target-cumulative-updates 938 \
  --solver "${SOLVER}" \
  --out-dir "${OUT}" \
  --workers 31 \
  --fit-concurrency 4 \
  --threads-per-member 8

echo "HU_SEMANTIC_LONG_CUM469_938_SHELL_PASS"
echo "STOP HERE. Send final PASS lines/report to ChatGPT."
