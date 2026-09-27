#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
SOURCE="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE}/checkpoint.pt"
HU="${SOURCE}/hu_ensemble_state.pt"
SEM_RUN="${ROOT}/runs/3h_semantic_research_10105_10115"
SEM_POLICY="${SEM_RUN}/semantic_tail_policy_10115.pt"
SEM_ADV="${SEM_RUN}/semantic_ensemble_final_10115.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_HU="8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"

[[ -x "${PY}" ]] || { echo "ERROR: missing Python venv" >&2; exit 2; }
for f in "${CHECKPOINT}" "${HU}" "${SEM_POLICY}" "${SEM_ADV}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing artifact: ${f}" >&2; exit 2; }
done
[[ "$(sha256sum "${CHECKPOINT}"|awk '{print $1}')" == "${EXPECTED_CP}" ]] || { echo "ERROR: checkpoint SHA mismatch" >&2; exit 3; }
[[ "$(sha256sum "${HU}"|awk '{print $1}')" == "${EXPECTED_HU}" ]] || { echo "ERROR: HU SHA mismatch" >&2; exit 3; }

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"${PY}" -m py_compile tools/audit_dc1_semantic_strong_hand_surface.py
"${PY}" tools/test_deepcrusher_nouts_native_collision.py

RUN="${ROOT}/runs/dc1_semantic_strong_hand_attribution"
mkdir -p "${RUN}"
BUNDLE="${RUN}/spincore_hybrid_10105.pt"
REPORT="${RUN}/dc1_semantic_strong_hand_attribution.json"

if [[ ! -f "${BUNDLE}" ]]; then
  "${PY}" tools/export_spincore_hybrid_inference.py \
    --checkpoint "${CHECKPOINT}" --ensemble "${HU}" --out "${BUNDLE}"
fi

"${PY}" tools/audit_dc1_semantic_strong_hand_surface.py \
  --solver build/libspincore_solver_c.so \
  --base-spin-bundle "${BUNDLE}" \
  --semantic-policy "${SEM_POLICY}" \
  --semantic-advantage "${SEM_ADV}" \
  --scenarios 1000 --workers 8 --seed 20260923 --max-decisions 200 \
  --report "${REPORT}"

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${REPORT}" "${DWSL}/SpinCore_DC1_semantic_strong_hand_attribution.json"
    echo "desktop_report=${DWIN}\\SpinCore_DC1_semantic_strong_hand_attribution.json"
  fi
fi

echo "DC1_SEMANTIC_STRONG_HAND_ATTRIBUTION_RUN_COMPLETE"
