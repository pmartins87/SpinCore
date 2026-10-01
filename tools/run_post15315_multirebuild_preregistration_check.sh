#!/usr/bin/env bash
set -euo pipefail

# PROJECT_CONTRACT_IDS: TRAIN-022,TRAIN-023,MODEL-025,RNG-001,RNG-003,RNG-013,VALID-025,VALID-031,VALID-032,VALID-034,VALID-035,VALID-036,VALID-037,VALID-038,SAFE-001,SAFE-010,SAFE-011,PERF-001,PERF-002,DC-001,BENCH-002,BENCH-013

ROOT="${HOME}/spincore_lean_functional"
if [[ ! -d "${ROOT}" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT}"

PY="${ROOT}/.venv_lean/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY=python3
fi

"${PY}" tools/check_post15315_multirebuild_preregistration.py
"${PY}" -m py_compile tools/check_post15315_multirebuild_preregistration.py

echo "POST15315_MULTI_REBUILD_PROTOCOL_FROZEN_PASS"
echo "NO_TRAINING_OR_TEACHER_ARTIFACT_WAS_TOUCHED"
