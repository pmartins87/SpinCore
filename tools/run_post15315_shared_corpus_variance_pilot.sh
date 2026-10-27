#!/usr/bin/env bash
# PROJECT_CONTRACT_IDS: TRAIN-022,TRAIN-023,MODEL-025,RNG-001,RNG-003,RNG-013,VALID-025,VALID-031,VALID-032,VALID-034,VALID-035,VALID-036,VALID-037,VALID-038,SAFE-001,SAFE-010,SAFE-011,CKPT-001,ART-001

set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"

"${PY}" tools/check_stage_manifest.py \
  --manifest contracts/run_manifests/post15315_shared_corpus_variance_pilot.json \
  --runner tools/run_post15315_shared_corpus_variance_pilot.sh

"${PY}" -m py_compile \
  tools/run_post15315_shared_corpus_variance_pilot.py \
  tools/build_post15315_rebuild_member.py \
  tools/evaluate_post15315_rebuild_stability.py

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c

mkdir -p runs/post15315_shared_corpus_variance_pilot

/usr/bin/time -v "${PY}" tools/run_post15315_shared_corpus_variance_pilot.py \
  --checkpoint runs/lt3_parallel_9105_10105/20260922_132104/checkpoint.pt \
  --teacher runs/3h_semantic_long_10315_15315/semantic_long_ensemble_final_15315.pt \
  --solver build/libspincore_solver_c.so \
  --protocol contracts/post15315_multirebuild_preregistration.json \
  --pilot-prereg contracts/post15315_shared_corpus_variance_pilot_v2.json \
  --k8-dir runs/post15315_multirebuild_k8 \
  --extension-dir runs/post15315_multirebuild_k12_extension \
  --k12-dir runs/post15315_rebuild_stability_k12 \
  --out-dir runs/post15315_shared_corpus_variance_pilot \
  --batch-size 4096

echo "POST15315_SHARED_CORPUS_VARIANCE_PILOT_SHELL_PASS"
echo "STOP HERE. Send the final attribution_result, ratios, summaries, and surface_counts to ChatGPT."
