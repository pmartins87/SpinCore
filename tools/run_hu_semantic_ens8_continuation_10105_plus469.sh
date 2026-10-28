#!/usr/bin/env bash
# PROJECT_CONTRACT_IDS: TRAIN-022,TRAIN-023,MODEL-025,RNG-001,RNG-003,VALID-025,VALID-034,VALID-035,SAFE-001,SAFE-010,CKPT-001,ART-001,PERF-001,PERF-002

set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"

MANIFEST="contracts/run_manifests/hu_semantic_ens8_continuation_10105_plus469.json"
RUNNER="tools/run_hu_semantic_ens8_continuation_10105_plus469.sh"
SOURCE_DIR="runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE_DIR}/checkpoint.pt"
SOURCE_ENSEMBLE="${SOURCE_DIR}/hu_ensemble_state.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_ENS="8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"
SOLVER="build/libspincore_solver_c.so"
OUT="runs/hu_semantic_ens8_continuation_10105_plus469"

for f in "${PY}" "${MANIFEST}" "${CHECKPOINT}" "${SOURCE_ENSEMBLE}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing prerequisite: ${f}" >&2; exit 2; }
done

"${PY}" tools/check_stage_manifest.py --manifest "${MANIFEST}" --runner "${RUNNER}"

[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${EXPECTED_CP}" ]] || {
  echo "ERROR: frozen 10105 checkpoint SHA mismatch" >&2; exit 3;
}
[[ "$(sha256sum "${SOURCE_ENSEMBLE}" | awk '{print $1}')" == "${EXPECTED_ENS}" ]] || {
  echo "ERROR: frozen 10105 HU ensemble SHA mismatch" >&2; exit 4;
}

if pgrep -af "run_3h_semantic_long_.*\.py|run_hu_semantic_ens8_.*\.py|run_hu_ens8_continuation_.*\.py|run_lt3_parallel_.*\.py" >/dev/null 2>&1; then
  echo "ERROR: another SpinCore long trainer is already running." >&2
  pgrep -af "run_3h_semantic_long_.*\.py|run_hu_semantic_ens8_.*\.py|run_hu_ens8_continuation_.*\.py|run_lt3_parallel_.*\.py" >&2 || true
  exit 5
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: tracked source/index changes present." >&2
  git status --short --untracked-files=no >&2
  exit 6
fi

MEMAVAIL_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[[ -n "${MEMAVAIL_KIB}" && "${MEMAVAIL_KIB}" -ge $((12*1024*1024)) ]] || {
  echo "ERROR: less than 12 GiB WSL MemAvailable" >&2
  free -h >&2 || true
  exit 7
}

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c

"${PY}" -m py_compile \
  python/spincore_nn/semantic_v1.py \
  python/spincore/semantic_parallel.py \
  tools/run_hu_semantic_ens8_continuation_10105_plus469.py \
  tools/lt3_3h_semantic_parallel_fit.py

mkdir -p "${OUT}"

/usr/bin/time -v "${PY}" tools/run_hu_semantic_ens8_continuation_10105_plus469.py \
  --checkpoint "${CHECKPOINT}" \
  --source-ensemble "${SOURCE_ENSEMBLE}" \
  --solver "${SOLVER}" \
  --out-dir "${OUT}" \
  --workers 31 \
  --fit-concurrency 4 \
  --threads-per-member 8

echo "HU_SEMANTIC_ENS8_CONTINUATION_10105_PLUS469_SHELL_PASS"
echo "STOP HERE. Send migration parity, initial fit, final PASS lines and report summary to ChatGPT."
