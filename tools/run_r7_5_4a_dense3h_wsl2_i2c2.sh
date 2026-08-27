#!/usr/bin/env bash
set -euo pipefail

# SpinCore R7.5.4A -- certified local Linux/WSL2 recovery continuation, i2c2.
#
# PRECONDITION:
#   The exact i2c1 state produced by the certified i2c1 launcher must already
#   exist under $WORK_ROOT/state/<seed>/i2c1 and match the frozen hashes below.
#
# SCIENTIFIC CONTRACT:
#   This wrapper performs only the next one-root mechanical continuation of
#   frozen iteration 2. It does not change source, seeds, root order, deck seed
#   formula, policy/optimizer/reservoir semantics, budgets, or strategic gates.

SOURCE_SHA="457996944f76e9f1fa0475691df978f450259641"
RECOVERY_SHA="a7eb746b0ac32ef730568150e1e2c2757bb212d2"
SOURCE_TRAINING_RUN_ID="31804178848"
PYTHON_VERSION="3.11.15"
TORCH_VERSION="2.13.0+cpu"
NUMPY_VERSION="2.3.5"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel >/dev/null 2>&1; then
  LOCAL_REPO_ROOT="$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)"
else
  LOCAL_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
BOOTSTRAP_DIR="${SPINCORE_R754_BOOTSTRAP_DIR:-$LOCAL_REPO_ROOT/SpinCore_R7_5_4A_WSL2_BOOTSTRAP}"
WORK_ROOT="${SPINCORE_R754_WORK_ROOT:-$HOME/spincore_r754_dense3h_recovery}"
EXPORT_ROOT="${SPINCORE_R754_EXPORT_ROOT:-$BOOTSTRAP_DIR/results_i2c2}"
REPO="$WORK_ROOT/recovery-impl"
FROZEN="$WORK_ROOT/frozen-source"
STATE="$WORK_ROOT/state"
LOGS="$WORK_ROOT/logs"
VENV="$WORK_ROOT/.venv"
PY="$VENV/bin/python"

say() { printf '[SpinCore R7.5.4A WSL2 i2c2] %s\n' "$*"; }
die() { printf '[SpinCore R7.5.4A WSL2 i2c2] ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command '$1'"; }

[[ "$(uname -s)" == "Linux" ]] || die "Linux is required by the frozen recovery contract"
need git
need sha256sum
[[ -d "$REPO/.git" ]] || die "missing recovery repository: $REPO"
[[ -d "$FROZEN/.git" || -f "$FROZEN/.git" ]] || die "missing frozen source worktree: $FROZEN"
[[ -x "$PY" ]] || die "missing frozen venv Python: $PY"
[[ -s "$FROZEN/build/libspincore_solver_c.so" ]] || die "missing frozen solver .so"
mkdir -p "$STATE" "$LOGS" "$EXPORT_ROOT"

[[ "$(git -C "$REPO" rev-parse HEAD)" == "$RECOVERY_SHA" ]] || die "recovery HEAD mismatch"
[[ "$(git -C "$FROZEN" rev-parse HEAD)" == "$SOURCE_SHA" ]] || die "source HEAD mismatch"

# Original iteration-1 provenance remains frozen across all recovery chunks.
declare -A SOURCE_ARTIFACT_ID=(
  [1737995611]="9224957740"
  [645939859]="9225388016"
  [1311335590]="9224960149"
)
declare -A SOURCE_ARTIFACT_DIGEST=(
  [1737995611]="44231b6e8ac00da19bcca267511f8355490205baba8e758985419214111ac75a"
  [645939859]="1b6ba187a83fe4f04b106dd3f7f9b8d6d1624776bc379d332114cd328e3ef7bc"
  [1311335590]="ecbf14071085b7f1674d814cb9ff066d6d3db7f4c913dee8359822dab403db07"
)
declare -A SOURCE_I1_CHECKPOINT_SHA=(
  [1737995611]="ea598ec624ee2e4e72fc8c3780c53863d6f116d5d9baa9495bcbbfe7cfadea2c"
  [645939859]="ba02b8a6b27da27b891c51a2e90bb437810ac2c44db6ca498375ca83be8cde09"
  [1311335590]="064713c596b6e860f25240c6b649aba00126346363aa5c6790c179ddb5e2e5ac"
)

# Certified outputs of the completed local i2c1 run, transcript + SUMMARY gate.
declare -A I2C1_CHECKPOINT_SHA=(
  [1737995611]="0a7e88af09b3cfd2352cdf76e3a882a416ceb8a2a6edc946b576078bbca4e172"
  [645939859]="1e37a635e2b763c95cc42158cf6f7ed33924eba42ad7d0c27bf2ad024a528987"
  [1311335590]="0b356aa5eef8dc61de55509afdb9ed8fbf7c6728ea34a08b80f29ff60c873e9b"
)

export PYTHONPATH="$FROZEN/python:$FROZEN/tools"
export SPINCORE_RECOVERY_SOURCE_ROOT="$FROZEN"
export SPINCORE_TORCH_THREADS=2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

say "Validating exact frozen runtime..."
"$PY" - <<'PY'
import platform, torch, numpy
assert platform.system() == 'Linux'
assert platform.python_version() == '3.11.15'
assert torch.__version__ == '2.13.0+cpu'
assert numpy.__version__ == '2.3.5'
torch.set_num_threads(2)
assert torch.get_num_threads() == 2
print('frozen runtime PASS', platform.platform(), platform.python_version(), torch.__version__, numpy.__version__)
PY

say "Re-running frozen mid-iteration equivalence gate..."
"$PY" -m py_compile \
  "$FROZEN/python/spincore/r7_5_action_stage_recovery.py" \
  "$REPO/tools/r7_5_4a_160_dense3h_recovery_worker.py" \
  "$REPO/python_tests/test_r7_5_action_stage_recovery.py"
"$PY" -m pytest -q \
  "$FROZEN/python_tests/test_r7_5_action_stage_contract.py" \
  "$FROZEN/python_tests/test_r7_5_action_stage.py" \
  "$REPO/python_tests/test_r7_5_action_stage_recovery.py"

seeds=(1737995611 645939859 1311335590)

say "Validating durable i2c1 inputs byte-for-byte..."
for seed in "${seeds[@]}"; do
  in_dir="$STATE/$seed/i2c1"
  [[ -s "$in_dir/checkpoint.pt" ]] || die "missing i2c1 checkpoint for seed=$seed"
  [[ -s "$in_dir/report.json" ]] || die "missing i2c1 report for seed=$seed"
  echo "${I2C1_CHECKPOINT_SHA[$seed]}  $in_dir/checkpoint.pt" | sha256sum -c -
  SEED="$seed" REPORT="$in_dir/report.json" EXPECTED_SHA="${I2C1_CHECKPOINT_SHA[$seed]}" "$PY" - <<'PY'
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
assert p['output_checkpoint_sha256']==os.environ['EXPECTED_SHA']
assert p['finalized'] is False
assert p['production_training_authorized'] is False
assert p['ready_for_tables'] is False
prov=p['recovery_provenance']
assert prov['deck_seed_formula_changed'] is False
assert prov['optimizer_semantics_changed'] is False
assert prov['policy_semantics_changed'] is False
assert prov['reservoir_semantics_changed'] is False
assert prov['root_order_changed'] is False
print(os.environ['SEED'], 'certified i2c1 input PASS', os.environ['EXPECTED_SHA'])
PY
done

run_seed() {
  local seed="$1"
  local in_dir="$STATE/$seed/i2c1"
  local out_dir="$STATE/$seed/i2c2"
  local log="$LOGS/i2c2_${seed}.log"
  mkdir -p "$out_dir"

  if [[ -s "$out_dir/checkpoint.pt" && -s "$out_dir/report.json" ]]; then
    say "seed=$seed i2c2 already exists; preserving for validation/reuse"
    return 0
  fi

  set +e
  "$PY" "$REPO/tools/run_with_heartbeat.py" \
    --label "r7.5.4a-wsl2-i2c2-${seed}" --interval-seconds 300 -- \
    "$PY" "$REPO/tools/r7_5_4a_160_dense3h_recovery_worker.py" \
      --mode collect --repo-root "$FROZEN" \
      --solver "$FROZEN/build/libspincore_solver_c.so" \
      --training-seed "$seed" --target-iteration 2 --root-budget 1 \
      --resume "$in_dir/checkpoint.pt" \
      --checkpoint-out "$out_dir/checkpoint.pt" --report-out "$out_dir/report.json" \
      --source-execution-sha "$SOURCE_SHA" --recovery-execution-sha "$RECOVERY_SHA" \
      --source-training-run-id "$SOURCE_TRAINING_RUN_ID" \
      --source-iteration1-artifact-id "${SOURCE_ARTIFACT_ID[$seed]}" \
      --source-iteration1-artifact-digest "${SOURCE_ARTIFACT_DIGEST[$seed]}" \
      --source-checkpoint-sha256 "${SOURCE_I1_CHECKPOINT_SHA[$seed]}" \
      2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}

say "Starting exact next durable root: i2c2 (root 2/32) for all three seeds in parallel."
pids=()
for seed in "${seeds[@]}"; do
  run_seed "$seed" &
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    say "seed=${seeds[$i]} FAILED; see $LOGS/i2c2_${seeds[$i]}.log"
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || die "one or more i2c2 workers failed; successful outputs remain durable"

say "Validating i2c2 reports and checkpoints..."
for seed in "${seeds[@]}"; do
  in_dir="$STATE/$seed/i2c1"
  out_dir="$STATE/$seed/i2c2"
  [[ -s "$out_dir/checkpoint.pt" ]] || die "missing i2c2 checkpoint for seed=$seed"
  [[ -s "$out_dir/report.json" ]] || die "missing i2c2 report for seed=$seed"
  actual_sha="$(sha256sum "$out_dir/checkpoint.pt" | awk '{print $1}')"
  SEED="$seed" REPORT="$out_dir/report.json" INPUT_SHA="${I2C1_CHECKPOINT_SHA[$seed]}" ACTUAL_SHA="$actual_sha" "$PY" - <<'PY'
import json, os
p=json.load(open(os.environ['REPORT'], encoding='utf-8'))
assert p['schema']=='SPINCORE_R7_5_4A_DENSE3H_RECOVERY_WORKER_V1'
assert p['mode']=='collect'
assert p['candidate_id']=='PF_DENSE_REFERENCE'
assert p['domain']=='THREE_HANDED'
assert p['training_seed']==int(os.environ['SEED'])
assert p['target_iteration']==2
assert p['root_budget']==1
assert p['roots_collected']==2
assert p['input_checkpoint_sha256']==os.environ['INPUT_SHA']
assert p['output_checkpoint_sha256']==os.environ['ACTUAL_SHA']
assert p['source_execution_sha']=='457996944f76e9f1fa0475691df978f450259641'
assert p['recovery_execution_sha']=='a7eb746b0ac32ef730568150e1e2c2757bb212d2'
assert p['finalized'] is False
assert p['production_training_authorized'] is False
assert p['ready_for_tables'] is False
prov=p['recovery_provenance']
assert prov['deck_seed_formula_changed'] is False
assert prov['optimizer_semantics_changed'] is False
assert prov['policy_semantics_changed'] is False
assert prov['reservoir_semantics_changed'] is False
assert prov['root_order_changed'] is False
print(os.environ['SEED'], 'i2c2 PASS', 'wall_seconds=', round(float(p['wall_seconds']),2), 'sha=', p['output_checkpoint_sha256'])
PY
  mkdir -p "$EXPORT_ROOT/$seed"
  cp -f "$out_dir/checkpoint.pt" "$EXPORT_ROOT/$seed/checkpoint.pt"
  cp -f "$out_dir/report.json" "$EXPORT_ROOT/$seed/report.json"
  [[ -s "$LOGS/i2c2_${seed}.log" ]] && cp -f "$LOGS/i2c2_${seed}.log" "$EXPORT_ROOT/$seed/i2c2.log" || true
done

EXPORT_ROOT_ENV="$EXPORT_ROOT" "$PY" - <<'PY' > "$EXPORT_ROOT/SUMMARY.json"
import json, os
from pathlib import Path
root=Path(os.environ['EXPORT_ROOT_ENV'])
rows=[]
for seed in (1737995611,645939859,1311335590):
    p=json.loads((root/str(seed)/'report.json').read_text())
    rows.append({
        'seed': seed,
        'wall_seconds': p['wall_seconds'],
        'input_checkpoint_sha256': p['input_checkpoint_sha256'],
        'checkpoint_sha256': p['output_checkpoint_sha256'],
        'roots_collected': p['roots_collected'],
    })
print(json.dumps({
  'schema':'SPINCORE_R7_5_4A_WSL2_I2C2_SUMMARY_V1',
  'source_execution_sha':'457996944f76e9f1fa0475691df978f450259641',
  'recovery_execution_sha':'a7eb746b0ac32ef730568150e1e2c2757bb212d2',
  'status':'I2C2_ALL_THREE_SEEDS_PASS',
  'rows':rows,
}, indent=2, sort_keys=True))
PY

say "I2C2 COMPLETE for all three seeds."
say "Exported result: $EXPORT_ROOT"
say "Send back SUMMARY.json. Do not delete or recreate $WORK_ROOT."
