#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
SOURCE="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE}/checkpoint.pt"
HU="${SOURCE}/hu_ensemble_state.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_HU="8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"

[[ -x "${PY}" && -f "${CHECKPOINT}" && -f "${HU}" ]] || { echo "ERROR: source artifacts missing" >&2; exit 2; }
[[ "$(sha256sum "${CHECKPOINT}"|awk '{print $1}')" == "${EXPECTED_CP}" ]] || { echo "ERROR: checkpoint SHA mismatch" >&2; exit 3; }
[[ "$(sha256sum "${HU}"|awk '{print $1}')" == "${EXPECTED_HU}" ]] || { echo "ERROR: HU SHA mismatch" >&2; exit 3; }

ADV_RUN="$(find "${ROOT}/runs/3h_v1_semantic_shadow" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
PROBE_RUN="$(find "${ROOT}/runs/3h_advantage_controlled_budget_curve" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
[[ -n "${ADV_RUN}" && -n "${PROBE_RUN}" ]] || { echo "ERROR: prerequisite diagnostic runs missing" >&2; exit 4; }
ADV_MODELS="${ADV_RUN}/3h_v1_semantic_shadow_models.pt"
PROBE="${PROBE_RUN}/3h_advantage_controlled_split_probe.pt"
[[ -f "${ADV_MODELS}" && -f "${PROBE}" ]] || { echo "ERROR: prerequisite artifacts missing" >&2; exit 4; }

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

"${PY}" -m py_compile \
  tools/run_3h_semantic_online_pilot_10105.py \
  tools/audit_3h_fresh_semantic_strategy_distill_10105.py \
  tools/audit_3h_fresh_semantic_strategy_independent500_10105.py

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${ROOT}/runs/3h_semantic_online_pilot/${STAMP}"
mkdir -p "${RUN}"
BUNDLE="${RUN}/spincore_hybrid_10105.pt"
ENSEMBLE="${RUN}/3h_semantic_online_ensemble.pt"
POLICY="${RUN}/3h_semantic_online_tail_policy.pt"
REPORT="${RUN}/3h_semantic_online_pilot.json"

"${PY}" tools/export_spincore_hybrid_inference.py \
  --checkpoint "${CHECKPOINT}" --ensemble "${HU}" --out "${BUNDLE}"

"${PY}" tools/run_3h_semantic_online_pilot_10105.py \
  --checkpoint "${CHECKPOINT}" \
  --semantic-advantage-models "${ADV_MODELS}" \
  --probe "${PROBE}" \
  --solver build/libspincore_solver_c.so \
  --spin-bundle "${BUNDLE}" \
  --report "${REPORT}" \
  --out-ensemble "${ENSEMBLE}" \
  --out-policy "${POLICY}" \
  --threads 8

echo "explorer_report=$(wslpath -w "${REPORT}" 2>/dev/null || printf '%s' "${REPORT}")"
if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${REPORT}" "${DWSL}/SpinCore_3H_semantic_online_pilot.json"
    echo "desktop_report=${DWIN}\\SpinCore_3H_semantic_online_pilot.json"
  fi
fi

echo "3H_SEMANTIC_ONLINE_PILOT_RUN_COMPLETE"
