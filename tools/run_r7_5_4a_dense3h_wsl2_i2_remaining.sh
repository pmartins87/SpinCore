#!/usr/bin/env bash
set -euo pipefail

# SpinCore R7.5.4A -- certified local Linux/WSL2 iteration-2 recovery driver.
#
# Purpose:
#   Continue the exact frozen one-root recovery chain from the already sealed
#   i2c1 state through i2c32, then perform the frozen iteration-2 fit (i2).
#
# Scientific contract:
#   - one root per seed per collection stage;
#   - three frozen seeds may run in parallel within a stage;
#   - hard barrier between stages;
#   - exact source/recovery identities and original-i1 provenance;
#   - no seed/root-order/deck-seed/policy/optimizer/reservoir/budget changes;
#   - every completed stage is validated and durable before the next begins;
#   - safe rerun reuses only byte-validated completed stages.
#
# This is mechanical recovery tooling. It does not authorize production
# training, tables, strategic PASS, representation changes, or R7.5.5.

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
EXPORT_BASE="${SPINCORE_R754_EXPORT_BASE:-$BOOTSTRAP_DIR/results_i2_recovery}"
FINAL_EXPORT="${SPINCORE_R754_FINAL_EXPORT:-$BOOTSTRAP_DIR/results_i2}"
REPO="$WORK_ROOT/recovery-impl"
FROZEN="$WORK_ROOT/frozen-source"
STATE="$WORK_ROOT/state"
LOGS="$WORK_ROOT/logs"
VENV="$WORK_ROOT/.venv"
PY="$VENV/bin/python"
LOCKDIR="$WORK_ROOT/.r754_i2_driver_lock"

say() { printf '[SpinCore R7.5.4A WSL2 i2-driver] %s\n' "$*"; }
die() { printf '[SpinCore R7.5.4A WSL2 i2-driver] ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "missing command '$1'"; }

[[ "$(uname -s)" == "Linux" ]] || die "Linux is required by the frozen recovery contract"
need git
need sha256sum
need awk
need tee
need cmp
need seq
need date

if ! mkdir "$LOCKDIR" 2>/dev/null; then
  die "another iteration-2 driver may be active (lock exists: $LOCKDIR)"
fi
printf '%s\n' "$$" > "$LOCKDIR/pid"
cleanup_lock() { rm -rf "$LOCKDIR"; }
abort_driver() {
  say "termination requested; stopping active workers while preserving durable completed stages"
  local jobs_now
  jobs_now="$(jobs -pr || true)"
  if [[ -n "$jobs_now" ]]; then
    kill $jobs_now 2>/dev/null || true
    wait $jobs_now 2>/dev/null || true
  fi
  exit 130
}
trap cleanup_lock EXIT
trap abort_driver INT TERM

[[ -d "$REPO/.git" ]] || die "missing recovery repository: $REPO"
[[ -d "$FROZEN/.git" || -f "$FROZEN/.git" ]] || die "missing frozen source worktree: $FROZEN"
[[ -x "$PY" ]] || die "missing frozen venv Python: $PY"
[[ -s "$FROZEN/build/libspincore_solver_c.so" ]] || die "missing frozen solver .so"
mkdir -p "$STATE" "$LOGS" "$EXPORT_BASE" "$FINAL_EXPORT"

[[ "$(git -C "$REPO" rev-parse HEAD)" == "$RECOVERY_SHA" ]] || die "recovery HEAD mismatch"
[[ "$(git -C "$FROZEN" rev-parse HEAD)" == "$SOURCE_SHA" ]] || die "source HEAD mismatch"
[[ -s "$FROZEN/python/spincore/r7_5_action_stage_recovery.py" ]] || die "recovery module missing from frozen source tree"
cmp -s \
  "$REPO/python/spincore/r7_5_action_stage_recovery.py" \
  "$FROZEN/python/spincore/r7_5_action_stage_recovery.py" \
  || die "installed recovery module differs from immutable recovery implementation"

say "Validating immutable original action-training files..."
(
  cd "$FROZEN"
  sha256sum -c <<'EOF'
c1bce8c256177686d2f7d035eb26fc989831141ca243cac198d30fc5632221f2  python/spincore/r7_5_action_stage.py
6dbf1213ef1520cae1ddab0f8bb2cb3cc5ac11f8e045cb6c8b60e32874749d79  python/spincore/r7_5_action_training.py
bb3d269f4b3df4f5e1996e49096d5a5f62c6e2e1a97bbf254b6e51c727f05786  python/spincore/r7_5_action_checkpoint.py
2891d5a8c1f6870899e9dd124850b49ad5b4e7b85a82f5a4c4da5c2a6c15c961  python/spincore/r7_5_action_cfr.py
3ac6411db450152c4fdc4983548f89783b3935208a29056e6622fd862f2f8808  python/spincore/r7_5_action_uncertainty.py
61f4f08b31d7fd4c0572576bc4f6d4c95f3aaa7ad8659055cf4e0ff423fdeba0  python/spincore/r7_5_action_contract.py
73cfa984224f8c677e5d68ce733328c249e69ec9a11a504a6301b541027e5cdb  python/spincore/r7_5_action_stage_contract.py
EOF
)

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
declare -A I2C1_CHECKPOINT_SHA=(
  [1737995611]="0a7e88af09b3cfd2352cdf76e3a882a416ceb8a2a6edc946b576078bbca4e172"
  [645939859]="1e37a635e2b763c95cc42158cf6f7ed33924eba42ad7d0c27bf2ad024a528987"
  [1311335590]="0b356aa5eef8dc61de55509afdb9ed8fbf7c6728ea34a08b80f29ff60c873e9b"
)
seeds=(1737995611 645939859 1311335590)

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

checkpoint_sha() {
  sha256sum "$1" | awk '{print $1}'
}

quarantine_partial_output() {
  local dir="$1"
  local label="$2"
  local has_checkpoint=0
  local has_report=0
  [[ -e "$dir/checkpoint.pt" ]] && has_checkpoint=1
  [[ -e "$dir/report.json" ]] && has_report=1
  if [[ "$has_checkpoint" -ne "$has_report" ]]; then
    local qroot="$WORK_ROOT/quarantine"
    local qdir="$qroot/${label}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
    mkdir -p "$qroot"
    mv "$dir" "$qdir"
    mkdir -p "$dir"
    say "preserved incomplete prior output at $qdir; stage will be recomputed from last validated checkpoint"
  fi
}

validate_collect_stage() {
  local seed="$1"
  local stage="$2"
  local expected_input_sha="$3"
  local expected_output_sha="${4:-}"
  local dir="$STATE/$seed/i2c${stage}"
  [[ -s "$dir/checkpoint.pt" ]] || die "missing i2c${stage} checkpoint for seed=$seed"
  [[ -s "$dir/report.json" ]] || die "missing i2c${stage} report for seed=$seed"
  local actual_sha
  actual_sha="$(checkpoint_sha "$dir/checkpoint.pt")"
  if [[ -n "$expected_output_sha" && "$actual_sha" != "$expected_output_sha" ]]; then
    die "i2c${stage} checkpoint SHA mismatch for seed=$seed"
  fi

  SEED="$seed" STAGE="$stage" REPORT="$dir/report.json" \
  INPUT_SHA="$expected_input_sha" ACTUAL_SHA="$actual_sha" \
  SOURCE_ARTIFACT_ID_EXPECTED="${SOURCE_ARTIFACT_ID[$seed]}" \
  SOURCE_ARTIFACT_DIGEST_EXPECTED="${SOURCE_ARTIFACT_DIGEST[$seed]}" \
  SOURCE_I1_SHA_EXPECTED="${SOURCE_I1_CHECKPOINT_SHA[$seed]}" \
  "$PY" - <<'PY'
import json, os
p=json.load(open(os.environ['REPORT'], encoding='utf-8'))
seed=int(os.environ['SEED'])
stage=int(os.environ['STAGE'])
assert p['schema']=='SPINCORE_R7_5_4A_DENSE3H_RECOVERY_WORKER_V1'
assert p['mode']=='collect'
assert p['candidate_id']=='PF_DENSE_REFERENCE'
assert p['domain']=='THREE_HANDED'
assert p['training_seed']==seed
assert p['target_iteration']==2
assert p['root_budget']==1
assert p['roots_collected']==stage
assert p['input_checkpoint_sha256']==os.environ['INPUT_SHA']
assert p['output_checkpoint_sha256']==os.environ['ACTUAL_SHA']
assert p['source_execution_sha']=='457996944f76e9f1fa0475691df978f450259641'
assert p['recovery_execution_sha']=='a7eb746b0ac32ef730568150e1e2c2757bb212d2'
assert p['finalized'] is False
assert p['final_report'] is None
assert p['production_training_authorized'] is False
assert p['ready_for_tables'] is False
prov=p['recovery_provenance']
assert prov['source_training_run_id']==31804178848
assert prov['source_iteration1_artifact_id']==int(os.environ['SOURCE_ARTIFACT_ID_EXPECTED'])
assert prov['source_iteration1_artifact_digest']==os.environ['SOURCE_ARTIFACT_DIGEST_EXPECTED']
assert prov['source_iteration1_checkpoint_sha256']==os.environ['SOURCE_I1_SHA_EXPECTED']
assert prov['candidate_id']=='PF_DENSE_REFERENCE'
assert prov['domain']=='THREE_HANDED'
assert prov['training_seed']==seed
assert prov['intervention']=='MECHANICAL_MID_ITERATION_CHECKPOINT_ONLY'
for key in (
    'root_order_changed',
    'deck_seed_formula_changed',
    'reservoir_semantics_changed',
    'optimizer_semantics_changed',
    'policy_semantics_changed',
):
    assert prov[key] is False
print(seed, f'i2c{stage} validated', p['output_checkpoint_sha256'])
PY
}

validate_fit_stage() {
  local seed="$1"
  local expected_input_sha="$2"
  local dir="$STATE/$seed/i2"
  [[ -s "$dir/checkpoint.pt" ]] || die "missing i2 fit checkpoint for seed=$seed"
  [[ -s "$dir/report.json" ]] || die "missing i2 fit report for seed=$seed"
  local actual_sha
  actual_sha="$(checkpoint_sha "$dir/checkpoint.pt")"

  SEED="$seed" REPORT="$dir/report.json" \
  INPUT_SHA="$expected_input_sha" ACTUAL_SHA="$actual_sha" \
  SOURCE_ARTIFACT_ID_EXPECTED="${SOURCE_ARTIFACT_ID[$seed]}" \
  SOURCE_ARTIFACT_DIGEST_EXPECTED="${SOURCE_ARTIFACT_DIGEST[$seed]}" \
  SOURCE_I1_SHA_EXPECTED="${SOURCE_I1_CHECKPOINT_SHA[$seed]}" \
  "$PY" - <<'PY'
import json, os
p=json.load(open(os.environ['REPORT'], encoding='utf-8'))
seed=int(os.environ['SEED'])
assert p['schema']=='SPINCORE_R7_5_4A_DENSE3H_RECOVERY_WORKER_V1'
assert p['mode']=='fit'
assert p['candidate_id']=='PF_DENSE_REFERENCE'
assert p['domain']=='THREE_HANDED'
assert p['training_seed']==seed
assert p['target_iteration']==2
assert p['root_budget']==1
assert p['roots_collected']==32
assert p['input_checkpoint_sha256']==os.environ['INPUT_SHA']
assert p['output_checkpoint_sha256']==os.environ['ACTUAL_SHA']
assert p['source_execution_sha']=='457996944f76e9f1fa0475691df978f450259641'
assert p['recovery_execution_sha']=='a7eb746b0ac32ef730568150e1e2c2757bb212d2'
assert p['finalized'] is False
assert p['final_report'] is None
assert p['production_training_authorized'] is False
assert p['ready_for_tables'] is False
prov=p['recovery_provenance']
assert prov['source_training_run_id']==31804178848
assert prov['source_iteration1_artifact_id']==int(os.environ['SOURCE_ARTIFACT_ID_EXPECTED'])
assert prov['source_iteration1_artifact_digest']==os.environ['SOURCE_ARTIFACT_DIGEST_EXPECTED']
assert prov['source_iteration1_checkpoint_sha256']==os.environ['SOURCE_I1_SHA_EXPECTED']
assert prov['candidate_id']=='PF_DENSE_REFERENCE'
assert prov['domain']=='THREE_HANDED'
assert prov['training_seed']==seed
assert prov['intervention']=='MECHANICAL_MID_ITERATION_CHECKPOINT_ONLY'
for key in (
    'root_order_changed',
    'deck_seed_formula_changed',
    'reservoir_semantics_changed',
    'optimizer_semantics_changed',
    'policy_semantics_changed',
):
    assert prov[key] is False
print(seed, 'i2 fit validated', p['output_checkpoint_sha256'])
PY
}

write_collect_progress() {
  local stage="$1"
  local out="$EXPORT_BASE/i2c${stage}_SUMMARY.json"
  STAGE="$stage" STATE_ROOT="$STATE" "$PY" - <<'PY' > "$out"
import json, os
from pathlib import Path
stage=int(os.environ['STAGE'])
state=Path(os.environ['STATE_ROOT'])
rows=[]
for seed in (1737995611,645939859,1311335590):
    p=json.loads((state/str(seed)/f'i2c{stage}'/'report.json').read_text())
    rows.append({
        'seed': seed,
        'wall_seconds': p['wall_seconds'],
        'input_checkpoint_sha256': p['input_checkpoint_sha256'],
        'checkpoint_sha256': p['output_checkpoint_sha256'],
        'roots_collected': p['roots_collected'],
    })
print(json.dumps({
    'schema':'SPINCORE_R7_5_4A_WSL2_I2_COLLECTION_PROGRESS_V1',
    'source_execution_sha':'457996944f76e9f1fa0475691df978f450259641',
    'recovery_execution_sha':'a7eb746b0ac32ef730568150e1e2c2757bb212d2',
    'status':f'I2C{stage}_ALL_THREE_SEEDS_PASS',
    'target_iteration':2,
    'completed_roots_per_seed':stage,
    'rows':rows,
}, indent=2, sort_keys=True))
PY
  cp -f "$out" "$EXPORT_BASE/PROGRESS.json"
  say "Durable barrier PASS: i2c${stage} (${stage}/32 roots per seed). Progress: $EXPORT_BASE/PROGRESS.json"
}

run_collect_seed() {
  local seed="$1"
  local stage="$2"
  local prev_stage=$((stage - 1))
  local in_dir="$STATE/$seed/i2c${prev_stage}"
  local out_dir="$STATE/$seed/i2c${stage}"
  local log="$LOGS/i2c${stage}_${seed}.log"
  mkdir -p "$out_dir"

  "$PY" "$REPO/tools/run_with_heartbeat.py" \
    --label "r7.5.4a-wsl2-i2c${stage}-${seed}" --interval-seconds 300 -- \
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
  return "${PIPESTATUS[0]}"
}

say "Validating sealed i2c1 starting point..."
for seed in "${seeds[@]}"; do
  validate_collect_stage \
    "$seed" 1 \
    "${SOURCE_I1_CHECKPOINT_SHA[$seed]}" \
    "${I2C1_CHECKPOINT_SHA[$seed]}"
done
write_collect_progress 1

for stage in $(seq 2 32); do
  prev_stage=$((stage - 1))
  say "Preparing i2c${stage}: root ${stage}/32 for all three seeds."

  launch_seeds=()
  pids=()

  for seed in "${seeds[@]}"; do
    prev_dir="$STATE/$seed/i2c${prev_stage}"
    prev_sha="$(checkpoint_sha "$prev_dir/checkpoint.pt")"
    validate_collect_stage "$seed" "$prev_stage" \
      "$([[ "$prev_stage" -eq 1 ]] && printf '%s' "${SOURCE_I1_CHECKPOINT_SHA[$seed]}" || checkpoint_sha "$STATE/$seed/i2c$((prev_stage - 1))/checkpoint.pt")" \
      "$([[ "$prev_stage" -eq 1 ]] && printf '%s' "${I2C1_CHECKPOINT_SHA[$seed]}" || true)"

    out_dir="$STATE/$seed/i2c${stage}"
    mkdir -p "$out_dir"
    quarantine_partial_output "$out_dir" "seed${seed}_i2c${stage}"
    if [[ -s "$out_dir/checkpoint.pt" && -s "$out_dir/report.json" ]]; then
      say "seed=$seed i2c${stage} already present; validating for deterministic reuse."
      validate_collect_stage "$seed" "$stage" "$prev_sha"
      continue
    fi
    launch_seeds+=("$seed")
  done

  for seed in "${launch_seeds[@]}"; do
    run_collect_seed "$seed" "$stage" &
    pids+=("$!")
  done

  failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      say "seed=${launch_seeds[$i]} FAILED at i2c${stage}; successful peers remain durable"
      failed=1
    fi
  done
  [[ "$failed" -eq 0 ]] || die "one or more i2c${stage} workers failed"

  for seed in "${seeds[@]}"; do
    prev_sha="$(checkpoint_sha "$STATE/$seed/i2c${prev_stage}/checkpoint.pt")"
    validate_collect_stage "$seed" "$stage" "$prev_sha"
  done
  write_collect_progress "$stage"
done

say "All 32 collection roots are durable for every seed. Starting exact frozen iteration-2 fit."
launch_seeds=()
pids=()

run_fit_seed() {
  local seed="$1"
  local in_dir="$STATE/$seed/i2c32"
  local out_dir="$STATE/$seed/i2"
  local log="$LOGS/i2fit_${seed}.log"
  mkdir -p "$out_dir"

  "$PY" "$REPO/tools/run_with_heartbeat.py" \
    --label "r7.5.4a-wsl2-i2-fit-${seed}" --interval-seconds 300 -- \
    "$PY" "$REPO/tools/r7_5_4a_160_dense3h_recovery_worker.py" \
      --mode fit --repo-root "$FROZEN" \
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
  return "${PIPESTATUS[0]}"
}

for seed in "${seeds[@]}"; do
  input_sha="$(checkpoint_sha "$STATE/$seed/i2c32/checkpoint.pt")"
  out_dir="$STATE/$seed/i2"
  mkdir -p "$out_dir"
  quarantine_partial_output "$out_dir" "seed${seed}_i2fit"
  if [[ -s "$out_dir/checkpoint.pt" && -s "$out_dir/report.json" ]]; then
    say "seed=$seed i2 fit already present; validating for deterministic reuse."
    validate_fit_stage "$seed" "$input_sha"
    continue
  fi
  launch_seeds+=("$seed")
done

for seed in "${launch_seeds[@]}"; do
  run_fit_seed "$seed" &
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    say "seed=${launch_seeds[$i]} FAILED during i2 fit; successful peers remain durable"
    failed=1
  fi
done
[[ "$failed" -eq 0 ]] || die "one or more iteration-2 fit workers failed"

for seed in "${seeds[@]}"; do
  input_sha="$(checkpoint_sha "$STATE/$seed/i2c32/checkpoint.pt")"
  validate_fit_stage "$seed" "$input_sha"
  mkdir -p "$FINAL_EXPORT/$seed"
  cp -f "$STATE/$seed/i2/checkpoint.pt" "$FINAL_EXPORT/$seed/checkpoint.pt"
  cp -f "$STATE/$seed/i2/report.json" "$FINAL_EXPORT/$seed/report.json"
  [[ -s "$LOGS/i2fit_${seed}.log" ]] && cp -f "$LOGS/i2fit_${seed}.log" "$FINAL_EXPORT/$seed/i2fit.log" || true
done

FINAL_EXPORT_ENV="$FINAL_EXPORT" "$PY" - <<'PY' > "$FINAL_EXPORT/SUMMARY.json"
import json, os
from pathlib import Path
root=Path(os.environ['FINAL_EXPORT_ENV'])
rows=[]
for seed in (1737995611,645939859,1311335590):
    p=json.loads((root/str(seed)/'report.json').read_text())
    rows.append({
        'seed': seed,
        'wall_seconds': p['wall_seconds'],
        'input_checkpoint_sha256': p['input_checkpoint_sha256'],
        'checkpoint_sha256': p['output_checkpoint_sha256'],
        'roots_collected': p['roots_collected'],
        'finalized': p['finalized'],
    })
print(json.dumps({
    'schema':'SPINCORE_R7_5_4A_WSL2_I2_SUMMARY_V1',
    'source_execution_sha':'457996944f76e9f1fa0475691df978f450259641',
    'recovery_execution_sha':'a7eb746b0ac32ef730568150e1e2c2757bb212d2',
    'status':'I2_ALL_THREE_SEEDS_PASS',
    'target_iteration':2,
    'operation':'fit',
    'roots_collected_per_seed':32,
    'production_training_authorized':False,
    'ready_for_tables':False,
    'rows':rows,
}, indent=2, sort_keys=True))
PY

cp -f "$FINAL_EXPORT/SUMMARY.json" "$EXPORT_BASE/PROGRESS.json"
say "ITERATION 2 RECOVERY COMPLETE for all three seeds."
say "Final machine-readable summary: $FINAL_EXPORT/SUMMARY.json"
say "Do not delete or recreate $WORK_ROOT."
say "Stop here. Iteration 3+ is not launched by this driver."
