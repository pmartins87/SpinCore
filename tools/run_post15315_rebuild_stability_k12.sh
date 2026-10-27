#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"

"${PY}" tools/check_stage_manifest.py --manifest contracts/run_manifests/post15315_rebuild_stability_k12.json --runner tools/run_post15315_rebuild_stability_k12.sh
bash tools/run_post15315_multirebuild_preregistration_check.sh
"${PY}" -m py_compile tools/run_post15315_rebuild_stability_k12.py tools/run_post15315_rebuild_stability_k8.py tools/evaluate_post15315_rebuild_stability.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c

mkdir -p runs/post15315_rebuild_stability_k12

"${PY}" tools/run_post15315_rebuild_stability_k12.py   --solver build/libspincore_solver_c.so   --teacher runs/3h_semantic_long_10315_15315/semantic_long_ensemble_final_15315.pt   --k8-construction-dir runs/post15315_multirebuild_k8   --extension-dir runs/post15315_multirebuild_k12_extension   --protocol contracts/post15315_multirebuild_preregistration.json   --coverage-adjudication contracts/post15315_stability_coverage_adjudication_v1.json   --k8-stability-gate runs/post15315_rebuild_stability_k8/post15315_k8_stability_gate.json   --out-dir runs/post15315_rebuild_stability_k12   --threads 8   --batch-size 4096

echo "POST15315_K12_STABILITY_SHELL_PASS"
