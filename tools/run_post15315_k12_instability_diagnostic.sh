#!/usr/bin/env bash
# PROJECT_CONTRACT_IDS: VALID-025,VALID-031,VALID-032,VALID-034,VALID-035,VALID-036,VALID-037,VALID-038,SAFE-001,SAFE-010,SAFE-011,ART-001

set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"

"${PY}" tools/check_stage_manifest.py \
  --manifest contracts/run_manifests/post15315_k12_instability_diagnostic.json \
  --runner tools/run_post15315_k12_instability_diagnostic.sh

"${PY}" -m py_compile \
  tools/diagnose_post15315_k12_instability.py \
  tools/evaluate_post15315_rebuild_stability.py

"${PY}" tools/diagnose_post15315_k12_instability.py \
  --k12-dir runs/post15315_rebuild_stability_k12 \
  --k8-dir runs/post15315_rebuild_stability_k8 \
  --out runs/post15315_rebuild_stability_k12/post15315_k12_instability_diagnostic.json

echo "POST15315_K12_INSTABILITY_DIAGNOSTIC_SHELL_PASS"
echo "STOP HERE. Send post15315_k12_instability_diagnostic.json or terminal output to ChatGPT."
