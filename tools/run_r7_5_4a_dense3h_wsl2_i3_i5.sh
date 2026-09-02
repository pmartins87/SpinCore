#!/usr/bin/env bash
set -euo pipefail

SOURCE_SHA="457996944f76e9f1fa0475691df978f450259641"
RECOVERY_SHA="a7eb746b0ac32ef730568150e1e2c2757bb212d2"
SOURCE_TRAINING_RUN_ID="31804178848"
WORK_ROOT="${SPINCORE_R754_WORK_ROOT:-$HOME/spincore_r754_dense3h_recovery}"
BOOTSTRAP_DIR="${SPINCORE_R754_BOOTSTRAP_DIR:-}"
[[ -n "$BOOTSTRAP_DIR" ]] || { echo "ERROR: set SPINCORE_R754_BOOTSTRAP_DIR" >&2; exit 1; }
REPO="$WORK_ROOT/recovery-impl"; FROZEN="$WORK_ROOT/frozen-source"; STATE="$WORK_ROOT/state"; LOGS="$WORK_ROOT/logs"; VENV="$WORK_ROOT/.venv"; PY="$VENV/bin/python"
EXPORT_BASE="${SPINCORE_R754_EXPORT_BASE:-$BOOTSTRAP_DIR/results_i3_i5_recovery}"
FINAL_EXPORT="${SPINCORE_R754_FINAL_EXPORT:-$BOOTSTRAP_DIR/results_final_dense3h}"
LOCKDIR="$WORK_ROOT/.r754_i3_i5_driver_lock"
say(){ printf '[SpinCore R7.5.4A WSL2 i3-i5] %s\n' "$*"; }
die(){ printf '[SpinCore R7.5.4A WSL2 i3-i5] ERROR: %s\n' "$*" >&2; exit 1; }
need(){ command -v "$1" >/dev/null 2>&1 || die "missing command '$1'"; }
[[ "$(uname -s)" == Linux ]] || die "Linux required"
for c in git sha256sum awk tee cmp seq date; do need "$c"; done
mkdir "$LOCKDIR" 2>/dev/null || die "another i3-i5 driver may be active: $LOCKDIR"
printf '%s\n' "$$" > "$LOCKDIR/pid"
cleanup(){ rm -rf "$LOCKDIR"; }
abort(){ say "termination requested; preserving durable stages"; jobs_now="$(jobs -pr || true)"; [[ -z "$jobs_now" ]] || { kill $jobs_now 2>/dev/null || true; wait $jobs_now 2>/dev/null || true; }; exit 130; }
trap cleanup EXIT; trap abort INT TERM
[[ -d "$REPO/.git" ]] || die "missing recovery repo"
[[ -d "$FROZEN/.git" || -f "$FROZEN/.git" ]] || die "missing frozen worktree"
[[ -x "$PY" ]] || die "missing frozen venv"
[[ -s "$FROZEN/build/libspincore_solver_c.so" ]] || die "missing solver"
mkdir -p "$STATE" "$LOGS" "$EXPORT_BASE" "$FINAL_EXPORT"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$RECOVERY_SHA" ]] || die "recovery HEAD mismatch"
[[ "$(git -C "$FROZEN" rev-parse HEAD)" == "$SOURCE_SHA" ]] || die "source HEAD mismatch"
cmp -s "$REPO/python/spincore/r7_5_action_stage_recovery.py" "$FROZEN/python/spincore/r7_5_action_stage_recovery.py" || die "recovery module mismatch"

declare -A AID=([1737995611]=9224957740 [645939859]=9225388016 [1311335590]=9224960149)
declare -A ADIG=([1737995611]=44231b6e8ac00da19bcca267511f8355490205baba8e758985419214111ac75a [645939859]=1b6ba187a83fe4f04b106dd3f7f9b8d6d1624776bc379d332114cd328e3ef7bc [1311335590]=ecbf14071085b7f1674d814cb9ff066d6d3db7f4c913dee8359822dab403db07)
declare -A I1SHA=([1737995611]=ea598ec624ee2e4e72fc8c3780c53863d6f116d5d9baa9495bcbbfe7cfadea2c [645939859]=ba02b8a6b27da27b891c51a2e90bb437810ac2c44db6ca498375ca83be8cde09 [1311335590]=064713c596b6e860f25240c6b649aba00126346363aa5c6790c179ddb5e2e5ac)
declare -A I2SHA=([1737995611]=a51e15b355de8bdb41cc69c03e0d37facfc10124e9eeadd0c935606344c806f2 [645939859]=7139ff4d50df87695d177cde7371ae9994ad732367182b3a240c8683799f5df3 [1311335590]=21e620ed270945e3a88d3a5f664be4440adf499c90b78a8c9f057b9459ed74cb)
seeds=(1737995611 645939859 1311335590)
export PYTHONPATH="$FROZEN/python:$FROZEN/tools" SPINCORE_RECOVERY_SOURCE_ROOT="$FROZEN" SPINCORE_TORCH_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
say "Validating frozen runtime and recovery tests..."
"$PY" - <<'PYR'
import platform,torch,numpy
assert platform.system()=='Linux' and platform.python_version()=='3.11.15'
assert torch.__version__=='2.13.0+cpu' and numpy.__version__=='2.3.5'
torch.set_num_threads(2); assert torch.get_num_threads()==2
print('frozen runtime PASS',platform.platform(),platform.python_version(),torch.__version__,numpy.__version__)
PYR
"$PY" -m pytest -q "$FROZEN/python_tests/test_r7_5_action_stage_contract.py" "$FROZEN/python_tests/test_r7_5_action_stage.py" "$REPO/python_tests/test_r7_5_action_stage_recovery.py"
sha(){ sha256sum "$1"|awk '{print $1}'; }
quarantine(){ d="$1"; label="$2"; cp=0; rp=0; [[ -e "$d/checkpoint.pt" ]]&&cp=1; [[ -e "$d/report.json" ]]&&rp=1; if [[ $cp -ne $rp ]]; then q="$WORK_ROOT/quarantine/${label}_$(date -u +%Y%m%dT%H%M%SZ)_$$"; mkdir -p "$(dirname "$q")"; mv "$d" "$q"; mkdir -p "$d"; say "quarantined incomplete output: $q"; fi; }

validate(){
  seed="$1"; iter="$2"; mode="$3"; stage="$4"; expected_in="$5"; expected_out="${6:-}"; d="$STATE/$seed/$([[ "$mode" == fit ]] && echo i$iter || echo i${iter}c${stage})"
  [[ -s "$d/checkpoint.pt" && -s "$d/report.json" ]] || die "missing complete $mode output seed=$seed iter=$iter stage=$stage"
  actual="$(sha "$d/checkpoint.pt")"; [[ -z "$expected_out" || "$actual" == "$expected_out" ]] || die "checkpoint SHA mismatch seed=$seed iter=$iter"
  SEED="$seed" ITER="$iter" MODE="$mode" STAGE="$stage" REPORT="$d/report.json" INPUT="$expected_in" ACTUAL="$actual" AID_E="${AID[$seed]}" ADIG_E="${ADIG[$seed]}" I1_E="${I1SHA[$seed]}" "$PY" - <<'PYV'
import json,os
p=json.load(open(os.environ['REPORT'])); seed=int(os.environ['SEED']); it=int(os.environ['ITER']); stage=int(os.environ['STAGE']); mode=os.environ['MODE']
assert p['schema']=='SPINCORE_R7_5_4A_DENSE3H_RECOVERY_WORKER_V1' and p['mode']==mode
assert p['candidate_id']=='PF_DENSE_REFERENCE' and p['domain']=='THREE_HANDED' and p['training_seed']==seed
assert p['target_iteration']==it and p['root_budget']==1 and p['roots_collected']==(32 if mode=='fit' else stage)
assert p['input_checkpoint_sha256']==os.environ['INPUT'] and p['output_checkpoint_sha256']==os.environ['ACTUAL']
assert p['source_execution_sha']=='457996944f76e9f1fa0475691df978f450259641' and p['recovery_execution_sha']=='a7eb746b0ac32ef730568150e1e2c2757bb212d2'
prov=p['recovery_provenance']; assert prov['source_training_run_id']==31804178848 and prov['source_iteration1_artifact_id']==int(os.environ['AID_E']) and prov['source_iteration1_artifact_digest']==os.environ['ADIG_E'] and prov['source_iteration1_checkpoint_sha256']==os.environ['I1_E']
assert prov['candidate_id']=='PF_DENSE_REFERENCE' and prov['domain']=='THREE_HANDED' and prov['training_seed']==seed and prov['intervention']=='MECHANICAL_MID_ITERATION_CHECKPOINT_ONLY'
for k in ('root_order_changed','deck_seed_formula_changed','reservoir_semantics_changed','optimizer_semantics_changed','policy_semantics_changed'): assert prov[k] is False
assert p['production_training_authorized'] is False and p['ready_for_tables'] is False
if mode=='collect' or it<5: assert p['finalized'] is False and p['final_report'] is None
else:
 assert p['finalized'] is True; f=p['final_report']; assert f['schema']=='SPINCORE_R7_5_ACTION_DOMAIN_FINAL_REPORT_V1'; assert f['roots']==160; assert f['average_policy_optimizer_steps']==16384; assert f['side_advantage_optimizer_steps']==3*4096*5; assert f['strategic_selection_permitted_at_160'] is False; assert f['production_training_authorized'] is False and f['ready_for_tables'] is False
print(seed,f'i{it}{" fit" if mode=="fit" else "c"+str(stage)} validated',p['output_checkpoint_sha256'])
PYV
}

progress(){ iter="$1"; stage="$2"; mode="$3"; ITER="$iter" STAGE="$stage" MODE="$mode" STATE_ROOT="$STATE" "$PY" - <<'PYP' > "$EXPORT_BASE/PROGRESS.json"
import json,os
from pathlib import Path
it=int(os.environ['ITER']); st=int(os.environ['STAGE']); mode=os.environ['MODE']; root=Path(os.environ['STATE_ROOT']); rows=[]
for s in (1737995611,645939859,1311335590):
 p=json.loads((root/str(s)/(f'i{it}' if mode=='fit' else f'i{it}c{st}')/'report.json').read_text()); rows.append({'seed':s,'checkpoint_sha256':p['output_checkpoint_sha256'],'wall_seconds':p['wall_seconds'],'roots_collected':p['roots_collected'],'finalized':p['finalized']})
print(json.dumps({'schema':'SPINCORE_R7_5_4A_WSL2_I3_I5_PROGRESS_V1','source_execution_sha':'457996944f76e9f1fa0475691df978f450259641','recovery_execution_sha':'a7eb746b0ac32ef730568150e1e2c2757bb212d2','target_iteration':it,'mode':mode,'completed_roots_per_seed':st,'status':f'I{it}_ALL_THREE_SEEDS_PASS' if mode=='fit' else f'I{it}C{st}_ALL_THREE_SEEDS_PASS','rows':rows},indent=2,sort_keys=True))
PYP
cp -f "$EXPORT_BASE/PROGRESS.json" "$EXPORT_BASE/i${iter}_${mode}_${stage}_SUMMARY.json"; say "Durable barrier PASS: iteration=$iter mode=$mode roots=$stage"; }

worker(){ seed="$1"; iter="$2"; mode="$3"; input="$4"; out="$5"; stage="$6"; label="i${iter}$([[ "$mode" == fit ]] && echo fit || echo c$stage)"; "$PY" "$REPO/tools/run_with_heartbeat.py" --label "r7.5.4a-wsl2-${label}-${seed}" --interval-seconds 300 -- "$PY" "$REPO/tools/r7_5_4a_160_dense3h_recovery_worker.py" --mode "$mode" --repo-root "$FROZEN" --solver "$FROZEN/build/libspincore_solver_c.so" --training-seed "$seed" --target-iteration "$iter" --root-budget 1 --resume "$input" --checkpoint-out "$out/checkpoint.pt" --report-out "$out/report.json" --source-execution-sha "$SOURCE_SHA" --recovery-execution-sha "$RECOVERY_SHA" --source-training-run-id "$SOURCE_TRAINING_RUN_ID" --source-iteration1-artifact-id "${AID[$seed]}" --source-iteration1-artifact-digest "${ADIG[$seed]}" --source-checkpoint-sha256 "${I1SHA[$seed]}" 2>&1 | tee "$LOGS/${label}_${seed}.log"; return "${PIPESTATUS[0]}"; }

say "Validating sealed fitted iteration 2..."
for seed in "${seeds[@]}"; do d="$STATE/$seed/i2"; [[ -s "$d/report.json" ]]||die "missing i2 report seed=$seed"; inp="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["input_checkpoint_sha256"])' "$d/report.json")"; validate "$seed" 2 fit 32 "$inp" "${I2SHA[$seed]}"; done
progress 2 32 fit

for iter in 3 4 5; do
 for stage in $(seq 1 32); do
  say "Preparing i${iter}c${stage} (${stage}/32)"
  launch=(); pids=()
  for seed in "${seeds[@]}"; do
   if [[ $stage -eq 1 ]]; then input="$STATE/$seed/i$((iter-1))/checkpoint.pt"; insha="$(sha "$input")"; prevrep="$STATE/$seed/i$((iter-1))/report.json"; previnp="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["input_checkpoint_sha256"])' "$prevrep")"; validate "$seed" "$((iter-1))" fit 32 "$previnp" "$insha"; else input="$STATE/$seed/i${iter}c$((stage-1))/checkpoint.pt"; insha="$(sha "$input")"; if [[ $stage -eq 2 ]]; then previn="$(sha "$STATE/$seed/i$((iter-1))/checkpoint.pt")"; else previn="$(sha "$STATE/$seed/i${iter}c$((stage-2))/checkpoint.pt")"; fi; validate "$seed" "$iter" collect "$((stage-1))" "$previn"; fi
   out="$STATE/$seed/i${iter}c${stage}"; mkdir -p "$out"; quarantine "$out" "seed${seed}_i${iter}c${stage}"
   if [[ -s "$out/checkpoint.pt" && -s "$out/report.json" ]]; then say "seed=$seed i${iter}c${stage} already present; validating/reusing"; validate "$seed" "$iter" collect "$stage" "$insha"; else launch+=("$seed"); fi
  done
  for seed in "${launch[@]}"; do if [[ $stage -eq 1 ]]; then input="$STATE/$seed/i$((iter-1))/checkpoint.pt"; else input="$STATE/$seed/i${iter}c$((stage-1))/checkpoint.pt"; fi; out="$STATE/$seed/i${iter}c${stage}"; worker "$seed" "$iter" collect "$input" "$out" "$stage" & pids+=("$!"); done
  failed=0; for i in "${!pids[@]}"; do wait "${pids[$i]}" || { say "seed=${launch[$i]} FAILED at i${iter}c${stage}"; failed=1; }; done; [[ $failed -eq 0 ]]||die "collection failure i${iter}c${stage}"
  for seed in "${seeds[@]}"; do if [[ $stage -eq 1 ]]; then expected="$(sha "$STATE/$seed/i$((iter-1))/checkpoint.pt")"; else expected="$(sha "$STATE/$seed/i${iter}c$((stage-1))/checkpoint.pt")"; fi; validate "$seed" "$iter" collect "$stage" "$expected"; done
  progress "$iter" "$stage" collect
 done
 say "All 32 roots durable for iteration $iter; fitting"
 launch=(); pids=()
 for seed in "${seeds[@]}"; do input="$STATE/$seed/i${iter}c32/checkpoint.pt"; insha="$(sha "$input")"; out="$STATE/$seed/i${iter}"; mkdir -p "$out"; quarantine "$out" "seed${seed}_i${iter}fit"; if [[ -s "$out/checkpoint.pt" && -s "$out/report.json" ]]; then say "seed=$seed i${iter} fit already present; validating/reusing"; validate "$seed" "$iter" fit 32 "$insha"; else launch+=("$seed"); fi; done
 for seed in "${launch[@]}"; do input="$STATE/$seed/i${iter}c32/checkpoint.pt"; out="$STATE/$seed/i${iter}"; worker "$seed" "$iter" fit "$input" "$out" 32 & pids+=("$!"); done
 failed=0; for i in "${!pids[@]}"; do wait "${pids[$i]}" || { say "seed=${launch[$i]} FAILED at i${iter} fit"; failed=1; }; done; [[ $failed -eq 0 ]]||die "fit failure iteration $iter"
 for seed in "${seeds[@]}"; do validate "$seed" "$iter" fit 32 "$(sha "$STATE/$seed/i${iter}c32/checkpoint.pt")"; done
 progress "$iter" 32 fit
done

say "Iterations 3-5 COMPLETE; exporting final recovered cells"
for seed in "${seeds[@]}"; do mkdir -p "$FINAL_EXPORT/$seed"; cp -f "$STATE/$seed/i5/checkpoint.pt" "$FINAL_EXPORT/$seed/checkpoint.pt"; cp -f "$STATE/$seed/i5/report.json" "$FINAL_EXPORT/$seed/report.json"; done
STATE_ROOT="$STATE" "$PY" - <<'PYF' > "$FINAL_EXPORT/SUMMARY.json"
import json,hashlib,os
from pathlib import Path
root=Path(os.environ['STATE_ROOT']); rows=[]
for s in (1737995611,645939859,1311335590):
 d=root/str(s)/'i5'; p=json.loads((d/'report.json').read_text()); h=hashlib.sha256((d/'checkpoint.pt').read_bytes()).hexdigest(); assert h==p['output_checkpoint_sha256']; rows.append({'seed':s,'checkpoint_sha256':h,'wall_seconds':p['wall_seconds'],'finalized':p['finalized'],'advantage_gate_pass':p['final_report']['advantage_gate_pass'],'policy_gate_pass':p['final_report']['policy_gate_pass']})
print(json.dumps({'schema':'SPINCORE_R7_5_4A_WSL2_DENSE3H_FINAL_RECOVERY_V1','source_execution_sha':'457996944f76e9f1fa0475691df978f450259641','recovery_execution_sha':'a7eb746b0ac32ef730568150e1e2c2757bb212d2','status':'DENSE3H_THREE_MISSING_CELLS_FINALIZED','candidate_id':'PF_DENSE_REFERENCE','domain':'THREE_HANDED','target_iteration':5,'rows':rows,'production_training_authorized':False,'ready_for_tables':False},indent=2,sort_keys=True))
PYF
say "FINAL COMPLETE: $FINAL_EXPORT/SUMMARY.json"
