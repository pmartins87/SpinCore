#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"

PY="${ROOT}/.venv_lean/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "ERROR: missing Python venv at ${PY}" >&2
  exit 2
fi

SOURCE_RUN="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE_RUN}/checkpoint.pt"
ENSEMBLE="${SOURCE_RUN}/hu_ensemble_state.pt"
EXPECTED_CHECKPOINT_SHA="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_ENSEMBLE_SHA="8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"

for f in "${CHECKPOINT}" "${ENSEMBLE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: missing artifact: ${f}" >&2
    exit 2
  fi
done

actual_cp="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
actual_ens="$(sha256sum "${ENSEMBLE}" | awk '{print $1}')"
[[ "${actual_cp}" == "${EXPECTED_CHECKPOINT_SHA}" ]] || {
  echo "ERROR: 10105 checkpoint SHA mismatch: ${actual_cp}" >&2
  exit 2
}
[[ "${actual_ens}" == "${EXPECTED_ENSEMBLE_SHA}" ]] || {
  echo "ERROR: 10105 ENS8 SHA mismatch: ${actual_ens}" >&2
  exit 2
}
echo "DC1_10105_SOURCE_HASHES_PASS"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${ROOT}/runs/deepcrusher_dc1_10105_smoke/${STAMP}"
mkdir -p "${RUN}"
exec > >(tee "${RUN}/terminal.log") 2>&1

echo "=== DeepCrusher DC1 10105 development smoke ==="
echo "run_dir=${RUN}"
echo "git_head=$(git rev-parse HEAD)"
echo "checkpoint_sha256=${actual_cp}"
echo "ensemble_sha256=${actual_ens}"
echo "scope=DEVELOPMENT_ONLY_NOT_CANONICAL_STRENGTH"
echo "scenarios=200 workers=8 seed=20260923"

BUNDLE="${RUN}/spincore_hybrid_10105.pt"
REPORT="${RUN}/dc1_report.json"
TRACES="${RUN}/dc1_traces.jsonl"
SANITY="${RUN}/decision_sanity.json"

"${PY}" tools/export_spincore_hybrid_inference.py   --checkpoint "${CHECKPOINT}"   --ensemble "${ENSEMBLE}"   --out "${BUNDLE}"

"${PY}" tools/evaluate_deepcrusher_dc1.py   --solver build/libspincore_solver_c.so   --spin-bundle "${BUNDLE}"   --scenarios 200   --workers 8   --seed 20260923   --report "${REPORT}"   --traces "${TRACES}"   --max-decisions 200

"${PY}" tools/audit_benchmark_decisions.py   --traces "${TRACES}"   --report "${SANITY}"   --max-examples-per-code 30

"${PY}" - "${RUN}" <<'PY'
from pathlib import Path
import sys, zipfile
run=Path(sys.argv[1])
out=run/"SpinCore_DC1_10105_smoke_bundle.zip"
names=("dc1_report.json","decision_sanity.json","dc1_traces.jsonl","terminal.log")
with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for name in names:
        p=run/name
        if p.is_file():
            z.write(p,arcname=name)
print(f"bundle={out}")
PY

ZIP="${RUN}/SpinCore_DC1_10105_smoke_bundle.zip"
EXPLORER_RUN_DIR="$(wslpath -w "${RUN}" 2>/dev/null || printf '%s' "${RUN}")"
EXPLORER_BUNDLE="$(wslpath -w "${ZIP}" 2>/dev/null || printf '%s' "${ZIP}")"
echo "explorer_run_dir=${EXPLORER_RUN_DIR}"
echo "explorer_bundle=${EXPLORER_BUNDLE}"

# Convenience copy for the user: resolve the real Windows Desktop (including
# OneDrive-redirection when configured) instead of assuming C:\\Users\\<name>.
if command -v powershell.exe >/dev/null 2>&1; then
  DESKTOP_WIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n 1)"
  if [[ -n "${DESKTOP_WIN}" ]]; then
    DESKTOP_WSL="$(wslpath -u "${DESKTOP_WIN}" 2>/dev/null || true)"
    if [[ -n "${DESKTOP_WSL}" && -d "${DESKTOP_WSL}" ]]; then
      cp -f "${ZIP}" "${DESKTOP_WSL}/SpinCore_DC1_10105_smoke_bundle.zip"
      echo "desktop_bundle=${DESKTOP_WIN}\\SpinCore_DC1_10105_smoke_bundle.zip"
    fi
  fi
fi

echo "DEEPC_RUSHER_DC1_10105_SMOKE_COMPLETE"
echo "STOP HERE. Send SpinCore_DC1_10105_smoke_bundle.zip to ChatGPT."
