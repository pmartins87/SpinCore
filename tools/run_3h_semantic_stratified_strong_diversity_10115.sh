#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
SOURCE="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE}/checkpoint.pt"
CANON="${ROOT}/runs/3h_semantic_research_canonical_10105_10115"
SEM_ADV="${CANON}/semantic_ensemble_final_10115.pt"
SEM_POLICY="${CANON}/semantic_tail_policy_10115.pt"
DIV1="${ROOT}/runs/3h_semantic_strong_diversity_repair_10115/semantic_strong_diversity_tail_candidate.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"

[[ -x "${PY}" ]] || { echo "ERROR: missing Python venv" >&2; exit 2; }
for f in "${CHECKPOINT}" "${SEM_ADV}" "${SEM_POLICY}" "${DIV1}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing artifact: ${f}" >&2; exit 2; }
done
[[ "$(sha256sum "${CHECKPOINT}"|awk '{print $1}')" == "${EXPECTED_CP}" ]] || { echo "ERROR: checkpoint SHA mismatch" >&2; exit 3; }

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

"${PY}" -m py_compile tools/audit_3h_semantic_stratified_strong_diversity_10115.py

RUN="${ROOT}/runs/3h_semantic_stratified_strong_diversity_10115"
mkdir -p "${RUN}"
REPORT="${RUN}/3h_semantic_stratified_strong_diversity_10115.json"
MODEL="${RUN}/semantic_stratified_strong_diversity_tail_candidate.pt"

"${PY}" tools/audit_3h_semantic_stratified_strong_diversity_10115.py \
  --checkpoint "${CHECKPOINT}" \
  --semantic-advantage "${SEM_ADV}" \
  --canonical-tail "${SEM_POLICY}" \
  --diversity-v1-tail "${DIV1}" \
  --solver build/libspincore_solver_c.so \
  --report "${REPORT}" \
  --out-model "${MODEL}" \
  --threads 8

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${REPORT}" "${DWSL}/SpinCore_3H_semantic_stratified_strong_diversity_10115.json"
    echo "desktop_report=${DWIN}\\SpinCore_3H_semantic_stratified_strong_diversity_10115.json"
  fi
fi

echo "SEMANTIC_STRATIFIED_STRONG_DIVERSITY_REPAIR_RUN_COMPLETE"
