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
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"

[[ -x "${PY}" ]] || { echo "ERROR: missing Python venv" >&2; exit 2; }
for f in "${CHECKPOINT}" "${SEM_ADV}" "${SEM_POLICY}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing canonical artifact: ${f}" >&2; exit 2; }
done
[[ "$(sha256sum "${CHECKPOINT}"|awk '{print $1}')" == "${EXPECTED_CP}" ]] || { echo "ERROR: checkpoint SHA mismatch" >&2; exit 3; }

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

"${PY}" -m py_compile tools/audit_3h_semantic_tail_strong_hand_coverage_10115.py

RUN="${ROOT}/runs/3h_semantic_tail_strong_hand_coverage_canonical_10115"
mkdir -p "${RUN}"
REPORT="${RUN}/3h_semantic_tail_strong_hand_coverage_canonical_10115.json"

"${PY}" tools/audit_3h_semantic_tail_strong_hand_coverage_10115.py \
  --checkpoint "${CHECKPOINT}" \
  --semantic-advantage "${SEM_ADV}" \
  --semantic-policy "${SEM_POLICY}" \
  --solver build/libspincore_solver_c.so \
  --report "${REPORT}" \
  --threads 8 \
  --expected-train-samples 22726 \
  --expected-holdout-samples 5701

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${REPORT}" "${DWSL}/SpinCore_3H_semantic_tail_strong_hand_coverage_canonical_10115.json"
    echo "desktop_report=${DWIN}\\SpinCore_3H_semantic_tail_strong_hand_coverage_canonical_10115.json"
  fi
fi

echo "SEMANTIC_TAIL_STRONG_HAND_COVERAGE_CANONICAL_AUDIT_RUN_COMPLETE"
