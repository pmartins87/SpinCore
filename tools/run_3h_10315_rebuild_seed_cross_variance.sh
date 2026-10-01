#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: VALID-025,VALID-030,VALID-031,VALID-032,SAFE-001,PERF-010,PERF-013,RNG-001,RNG-002,RNG-003,ART-001,SRC-003

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SRC="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CP="${SRC}/checkpoint.pt"
HU="${SRC}/hu_ensemble_state.pt"
SOLVER="${ROOT}/build/libspincore_solver_c.so"

OLD_SPEC="${ROOT}/runs/3h_semantic_stratified_specialist_moe_10115/semantic_stratified_specialist_moe_10115.pt"
NEW_ADV="${ROOT}/runs/3h_semantic_long_10115_10315/semantic_long_ensemble_final_10315.pt"
ORIG_FULL="${ROOT}/runs/3h_semantic_postlong_10315/semantic_fullpool_tail_10315.pt"
ORIG_SPEC="${ROOT}/runs/3h_semantic_postlong_10315/semantic_stratified_specialist_10315.pt"

RUN="${ROOT}/runs/3h_semantic_postlong_10315_seedcross_10115"
ALT_TAIL="${RUN}/semantic_tail_10315_seedcross_10115.pt"
ALT_FULL="${RUN}/semantic_fullpool_10315_seedcross_10115.pt"
ALT_SPEC="${RUN}/semantic_stratified_specialist_10315_seedcross_10115.pt"
POLICY_REPORT="${RUN}/policy_rebuild_seedcross.json"
SPEC_REPORT="${RUN}/specialist_build_seedcross.json"
BASE="${RUN}/spincore_hybrid_10105.pt"
COMPARE="${RUN}/seed_cross_variance.json"

mkdir -p "${RUN}"
for f in "${PY}" "${CP}" "${HU}" "${SOLVER}" "${OLD_SPEC}" "${NEW_ADV}" "${ORIG_FULL}" "${ORIG_SPEC}"; do
  [[ -e "${f}" ]] || { echo "ERROR missing prerequisite: ${f}" >&2; exit 2; }
done

"${PY}" -m py_compile   tools/build_3h_semantic_postlong_policy_10315.py   tools/build_3h_semantic_stratified_specialist_10315.py   tools/audit_3h_10315_rebuild_seed_cross_variance.py

if [[ ! -f "${BASE}" ]]; then
  "${PY}" tools/export_spincore_hybrid_inference.py     --checkpoint "${CP}" --ensemble "${HU}" --out "${BASE}"
fi

echo "SEED_CROSS_POLICY_REBUILD_START"
"${PY}" - "${CP}" "${NEW_ADV}" "${SOLVER}" "${POLICY_REPORT}" "${ALT_TAIL}" "${ALT_FULL}" <<'PY'
import sys
import build_3h_semantic_postlong_policy_10315 as m
m.BASE_TRAIN_SEED = 20260927 ^ 0x10115A
m.AUGMENT_SEED = 20260928 ^ 0x51A7F1
sys.argv=[
    "seedcross_policy",
    "--checkpoint",sys.argv[1],
    "--semantic-advantage",sys.argv[2],
    "--solver",sys.argv[3],
    "--report",sys.argv[4],
    "--out-tail",sys.argv[5],
    "--out-fullpool",sys.argv[6],
    "--threads","8",
    "--collection-threads","1",
]
raise SystemExit(m.main())
PY

echo "SEED_CROSS_SPECIALIST_REBUILD_START"
"${PY}" - "${CP}" "${NEW_ADV}" "${ALT_FULL}" "${SOLVER}" "${SPEC_REPORT}" "${ALT_SPEC}" <<'PY'
import sys
import build_3h_semantic_stratified_specialist_10315 as m
m.TRAIN_SEED = 20260928 ^ 0x57A711
sys.argv=[
    "seedcross_specialist",
    "--checkpoint",sys.argv[1],
    "--semantic-advantage",sys.argv[2],
    "--fullpool-tail",sys.argv[3],
    "--solver",sys.argv[4],
    "--report",sys.argv[5],
    "--out-model",sys.argv[6],
    "--threads","8",
    "--collection-threads","1",
]
raise SystemExit(m.main())
PY

echo "SEED_CROSS_COMMON_STATE_COMPARE_START"
"${PY}" tools/audit_3h_10315_rebuild_seed_cross_variance.py   --solver "${SOLVER}"   --base-bundle "${BASE}"   --accepted-10115 "${OLD_SPEC}"   --original-10315 "${ORIG_SPEC}"   --seedcross-10315 "${ALT_SPEC}"   --episodes 20000   --threads 8   --report "${COMPARE}"

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${COMPARE}" "${DWSL}/SpinCore_10315_rebuild_seed_cross_variance.json"
    cp -f "${POLICY_REPORT}" "${DWSL}/SpinCore_10315_seedcross_policy_rebuild.json"
    cp -f "${SPEC_REPORT}" "${DWSL}/SpinCore_10315_seedcross_specialist_build.json"
    echo "desktop_report=${DWIN}\\SpinCore_10315_rebuild_seed_cross_variance.json"
  fi
fi

echo "SEED_CROSS_VARIANCE_DIAGNOSTIC_COMPLETE"
echo "STOP HERE. Send SpinCore_10315_rebuild_seed_cross_variance.json to ChatGPT."
