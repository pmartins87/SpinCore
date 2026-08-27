#!/usr/bin/env bash
set -euo pipefail

# SpinCore R7.5.4A -- Linux/WSL2 certified local recovery, first durable root.
# Scientific recovery identity remains frozen at a7eb746b...; this wrapper only
# relocates the already-frozen Linux computation from GitHub-hosted Linux to
# local WSL2 Linux after the hosted runner timed out even at one root.

SOURCE_SHA="457996944f76e9f1fa0475691df978f450259641"
RECOVERY_SHA="a7eb746b0ac32ef730568150e1e2c2757bb212d2"
SOURCE_TRAINING_RUN_ID="31804178848"
PYTHON_VERSION="3.11.15"
TORCH_VERSION="2.13.0+cpu"
NUMPY_VERSION="2.3.5"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="${SPINCORE_R754_BOOTSTRAP_DIR:-$SCRIPT_DIR}"
WORK_ROOT="${SPINCORE_R754_WORK_ROOT:-$HOME/spincore_r754_dense3h_recovery}"
EXPORT_ROOT="${SPINCORE_R754_EXPORT_ROOT:-$BUNDLE_DIR/results_i2c1}"
REPO="$WORK_ROOT/recovery-impl"
FROZEN="$WORK_ROOT/frozen-source"
STATE="$WORK_ROOT/state"
LOGS="$WORK_ROOT/logs"

say() { printf '[SpinCore R7.5.4A WSL2] %s\n' "$*"; }
die() { printf '[SpinCore R7.5.4A WSL2] ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command '$1'. In Ubuntu run: sudo apt update && sudo apt install -y git curl unzip ca-certificates"; }

[[ "$(uname -s)" == "Linux" ]] || die "Linux is required by the frozen recovery contract."
if grep -qi microsoft /proc/version 2>/dev/null; then
  say "WSL Linux detected: $(uname -a)"
else
  say "Native Linux detected: $(uname -a)"
fi

need git
need curl
need unzip
need sha256sum
mkdir -p "$WORK_ROOT" "$STATE" "$LOGS" "$EXPORT_ROOT"

declare -A ZIP_SHA=(
  [1737995611]="44231b6e8ac00da19bcca267511f8355490205baba8e758985419214111ac75a"
  [645939859]="1b6ba187a83fe4f04b106dd3f7f9b8d6d1624776bc379d332114cd328e3ef7bc"
  [1311335590]="ecbf14071085b7f1674d814cb9ff066d6d3db7f4c913dee8359822dab403db07"
)
declare -A CHECKPOINT_SHA=(
  [1737995611]="ea598ec624ee2e4e72fc8c3780c53863d6f116d5d9baa9495bcbbfe7cfadea2c"
  [645939859]="ba02b8a6b27da27b891c51a2e90bb437810ac2c44db6ca498375ca83be8cde09"
  [1311335590]="064713c596b6e860f25240c6b649aba00126346363aa5c6790c179ddb5e2e5ac"
)
declare -A ARTIFACT_ID=(
  [1737995611]="9224957740"
  [645939859]="9225388016"
  [1311335590]="9224960149"
)
SOLVER_ZIP="$BUNDLE_DIR/r7_5_4a_recovery_solver.zip"
[[ -f "$SOLVER_ZIP" ]] || die "missing $SOLVER_ZIP"
echo "0bf6242e6842a58219ac905c88f598f7a35f521a6ab544cdcfb5b0ca434e06cd  $SOLVER_ZIP" | sha256sum -c -

for seed in 1737995611 645939859 1311335590; do
  z="$BUNDLE_DIR/i1_seed_${seed}.zip"
  [[ -f "$z" ]] || die "missing $z"
  echo "${ZIP_SHA[$seed]}  $z" | sha256sum -c -
done

if [[ ! -d "$REPO/.git" ]]; then
  say "Cloning recovery repository into WSL ext4..."
  git clone https://github.com/pmartins87/SpinCore.git "$REPO"
fi
git -C "$REPO" fetch --all --tags --prune
git -C "$REPO" checkout --detach "$RECOVERY_SHA"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$RECOVERY_SHA" ]] || die "recovery checkout mismatch"

if [[ -d "$FROZEN" ]]; then
  current="$(git -C "$FROZEN" rev-parse HEAD 2>/dev/null || true)"
  [[ "$current" == "$SOURCE_SHA" ]] || die "existing frozen-source has wrong HEAD: $current"
else
  say "Creating immutable original-source worktree..."
  git -C "$REPO" worktree add --detach "$FROZEN" "$SOURCE_SHA"
fi
[[ "$(git -C "$FROZEN" rev-parse HEAD)" == "$SOURCE_SHA" ]] || die "source checkout mismatch"

say "Validating frozen original source identities..."
(
  cd "$FROZEN"
  sha256sum -c <<'HASHES'
c1bce8c256177686d2f7d035eb26fc989831141ca243cac198d30fc5632221f2  python/spincore/r7_5_action_stage.py
6dbf1213ef1520cae1ddab0f8bb2cb3cc5ac11f8e045cb6c8b60e32874749d79  python/spincore/r7_5_action_training.py
bb3d269f4b3df4f5e1996e49096d5a5f62c6e2e1a97bbf254b6e51c727f05786  python/spincore/r7_5_action_checkpoint.py
2891d5a8c1f6870899e9dd124850b49ad5b4e7b85a82f5a4c4da5c2a6c15c961  python/spincore/r7_5_action_cfr.py
3ac6411db450152c4fdc4983548f89783b3935208a29056e6622fd862f2f8808  python/spincore/r7_5_action_uncertainty.py
61f4f08b31d7fd4c0572576bc4f6d4c95f3aaa7ad8659055cf4e0ff423fdeba0  python/spincore/r7_5_action_contract.py
73cfa984224f8c677e5d68ce733328c249e69ec9a11a504a6301b541027e5cdb  python/spincore/r7_5_action_stage_contract.py
HASHES
)

install -m 0644 "$REPO/python/spincore/r7_5_action_stage_recovery.py" "$FROZEN/python/spincore/r7_5_action_stage_recovery.py"

mkdir -p "$FROZEN/build"
unzip -oq "$SOLVER_ZIP" -d "$FROZEN/build"
[[ -s "$FROZEN/build/libspincore_solver_c.so" ]] || die "solver .so missing after extraction"

if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv in the WSL user account..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv installation failed"

VENV="$WORK_ROOT/.venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  say "Installing exact Python $PYTHON_VERSION and creating venv..."
  uv python install "$PYTHON_VERSION"
  uv venv --python "$PYTHON_VERSION" "$VENV"
fi
PY="$VENV/bin/python"

if ! "$PY" - <<PY >/dev/null 2>&1
import platform
raise SystemExit(0 if platform.python_version()=="$PYTHON_VERSION" else 1)
PY
then
  die "venv Python is not $PYTHON_VERSION; remove $VENV and rerun"
fi

say "Installing/verifying frozen Python runtime..."
uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cpu "torch==$TORCH_VERSION"
uv pip install --python "$PY" "numpy==$NUMPY_VERSION" pytest

export PYTHONPATH="$FROZEN/python:$FROZEN/tools"
export SPINCORE_RECOVERY_SOURCE_ROOT="$FROZEN"
export SPINCORE_TORCH_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

"$PY" - <<'PY'
import platform, torch, numpy
assert platform.system() == 'Linux'
assert platform.python_version() == '3.11.15'
assert torch.__version__ == '2.13.0+cpu'
assert numpy.__version__ == '2.3.5'
torch.set_num_threads(2)
assert torch.get_num_threads() == 2
print('frozen local Linux runtime PASS', platform.platform(), platform.python_version(), torch.__version__, numpy.__version__)
PY

say "Re-running the frozen mid-iteration equivalence gate locally..."
"$PY" -m py_compile \
  "$FROZEN/python/spincore/r7_5_action_stage_recovery.py" \
  "$REPO/tools/r7_5_4a_160_dense3h_recovery_worker.py" \
  "$REPO/python_tests/test_r7_5_action_stage_recovery.py"
"$PY" -m pytest -q \
  "$FROZEN/python_tests/test_r7_5_action_stage_contract.py" \
  "$FROZEN/python_tests/test_r7_5_action_stage.py" \
  "$REPO/python_tests/test_r7_5_action_stage_recovery.py"

for seed in 1737995611 645939859 1311335590; do
  in_dir="$STATE/$seed/original_i1"
  mkdir -p "$in_dir"
  if [[ ! -f "$in_dir/checkpoint.pt" ]]; then
    unzip -oq "$BUNDLE_DIR/i1_seed_${seed}.zip" -d "$in_dir"
  fi
  echo "${CHECKPOINT_SHA[$seed]}  $in_dir/checkpoint.pt" | sha256sum -c -
done

say "Starting the first durable iteration-2 root for all three seeds in parallel."
say "This is real recovery work, not a throwaway benchmark; each successful output is resumable i2c1."

run_seed() {
  local seed="$1"
  local in_dir="$STATE/$seed/original_i1"
  local out_dir="$STATE/$seed/i2c1"
  local log="$LOGS/i2c1_${seed}.log"
  mkdir -p "$out_dir"

  if [[ -s "$out_dir/checkpoint.pt" && -s "$out_dir/report.json" ]]; then
    say "seed=$seed i2c1 already exists; validating and reusing"
    return 0
  fi

  set +e
  "$PY" "$REPO/tools/run_with_heartbeat.py" \
    --label "r7.5.4a-wsl2-i2c1-${seed}" --interval-seconds 300 -- \
    "$PY" "$REPO/tools/r7_5_4a_160_dense3h_recovery_worker.py" \
      --mode collect --repo-root "$FROZEN" \
      --solver "$FROZEN/build/libspincore_solver_c.so" \
      --training-seed "$seed" --target-iteration 2 --root-budget 1 \
      --resume "$in_dir/checkpoint.pt" \
      --checkpoint-out "$out_dir/checkpoint.pt" --report-out "$out_dir/report.json" \
      --source-execution-sha "$SOURCE_SHA" --recovery-execution-sha "$RECOVERY_SHA" \
      --source-training-run-id "$SOURCE_TRAINING_RUN_ID" \
      --source-iteration1-artifact-id "${ARTIFACT_ID[$seed]}" \
      --source-iteration1-artifact-digest "${ZIP_SHA[$seed]}" \
      --source-checkpoint-sha256 "${CHECKPOINT_SHA[$seed]}" \
      2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}

pids=()
seeds=(1737995611 645939859 1311335590)
for seed in "${seeds[@]}"; do
  run_seed "$seed" &
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    say "seed=${seeds[$i]} FAILED; see $LOGS/i2c1_${seeds[$i]}.log"
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || die "one or more seed workers failed; successful seed outputs were preserved"

say "Validating i2c1 reports..."
for seed in "${seeds[@]}"; do
  out_dir="$STATE/$seed/i2c1"
  SEED="$seed" REPORT="$out_dir/report.json" "$PY" - <<'PY'
import json, os
p=json.load(open(os.environ['REPORT'], encoding='utf-8'))
assert p['schema']=='SPINCORE_R7_5_4A_DENSE3H_RECOVERY_WORKER_V1'
assert p['mode']=='collect'
assert p['candidate_id']=='PF_DENSE_REFERENCE'
assert p['domain']=='THREE_HANDED'
assert p['training_seed']==int(os.environ['SEED'])
assert p['target_iteration']==2
assert p['root_budget']==1
assert p['roots_collected']==1
assert p['source_execution_sha']=='457996944f76e9f1fa0475691df978f450259641'
assert p['recovery_execution_sha']=='a7eb746b0ac32ef730568150e1e2c2757bb212d2'
assert p['finalized'] is False
assert p['production_training_authorized'] is False
assert p['ready_for_tables'] is False
assert p['runtime']['python']=='3.11.15'
assert p['runtime']['torch']=='2.13.0+cpu'
print(os.environ['SEED'], 'i2c1 PASS', 'wall_seconds=', round(float(p['wall_seconds']),2), 'sha=', p['output_checkpoint_sha256'])
PY
  mkdir -p "$EXPORT_ROOT/$seed"
  cp -f "$out_dir/checkpoint.pt" "$EXPORT_ROOT/$seed/checkpoint.pt"
  cp -f "$out_dir/report.json" "$EXPORT_ROOT/$seed/report.json"
  cp -f "$LOGS/i2c1_${seed}.log" "$EXPORT_ROOT/$seed/i2c1.log"
done

EXPORT_ROOT_ENV="$EXPORT_ROOT" "$PY" - <<'PY' > "$EXPORT_ROOT/SUMMARY.json"
import json, os
from pathlib import Path
root=Path(os.environ['EXPORT_ROOT_ENV'])
rows=[]
for seed in (1737995611,645939859,1311335590):
    p=json.loads((root/str(seed)/'report.json').read_text())
    rows.append({'seed':seed,'wall_seconds':p['wall_seconds'],'checkpoint_sha256':p['output_checkpoint_sha256']})
print(json.dumps({
  'schema':'SPINCORE_R7_5_4A_WSL2_I2C1_SUMMARY_V1',
  'source_execution_sha':'457996944f76e9f1fa0475691df978f450259641',
  'recovery_execution_sha':'a7eb746b0ac32ef730568150e1e2c2757bb212d2',
  'status':'I2C1_ALL_THREE_SEEDS_PASS',
  'rows':rows,
}, indent=2, sort_keys=True))
PY

say "I2C1 COMPLETE for all three seeds."
say "Exported result: $EXPORT_ROOT"
say "Send back SUMMARY.json (and any report.json if requested). Do not delete $WORK_ROOT."
