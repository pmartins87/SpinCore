$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path '.').Path
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$Dirty = (& git -C $Repo status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) {
    throw "Tracked worktree is dirty. Untracked files are intentionally ignored. Stop and send this output before corrected Phase2B15:`n$Dirty"
}

$Precommit = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B15_POSTERIOR_WEIGHTED_CONTINUATION_CHANCE_PRECOMMIT_20260824.md'
$Correction = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B15_WINDOWS_HELDOUT_REPLAY_RUNTIME_CORRECTION_20260824.md'
$B14Evidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B14_RESULT_EVIDENCE_20260824.json'
$B13Evidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B13_RESULT_EVIDENCE_20260824.json'
foreach ($Path in @($Precommit, $Correction, $B14Evidence, $B13Evidence)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing corrected Phase2B15 frozen contract/evidence: $Path" }
}

$Venv = Join-Path $Repo '.venv-r7_5_3c_x16'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python -PathType Leaf)) { throw "Missing existing frozen Phase2 Python environment: $Python" }
& $Python -c "import struct,sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9),sys.version; assert struct.calcsize('P')==8; assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print('python',sys.version.split()[0],'bits',struct.calcsize('P')*8,'torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Frozen corrected Phase2B15 Python/Torch/Numpy runtime verification failed.' }

$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

$BaseTool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance.py'
$Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance_runtimefix.py'
$BaseTest = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance.py'
$FixTest = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b15_windows_replay_runtimefix.py'
$B10Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition.py'
$B10Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition.py'
$B11Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator.py'
$B13Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training.py'
$B14Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b14_b13_residual_localization.py'
$B7Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b7_residual_localization.py'
$B6Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py'

Write-Host '[V1+ Phase2B15 runtimefix] compiling scripts...'
& $Python -m py_compile $BaseTool $Tool $BaseTest $FixTest $B10Tool $B10Test $B11Tool $B13Tool $B14Tool $B7Tool $B6Tool (Join-Path $Repo 'python/spincore/solver.py')
if ($LASTEXITCODE -ne 0) { throw 'Corrected Phase2B15 py_compile failed.' }
Write-Host '[V1+ Phase2B15 runtimefix] running frozen scientific synthetic tests...'
& $Python $BaseTest
if ($LASTEXITCODE -ne 0) { throw 'Phase2B15 frozen synthetic tests failed.' }
Write-Host '[V1+ Phase2B15 runtimefix] running replay-correction synthetic tests...'
& $Python $FixTest
if ($LASTEXITCODE -ne 0) { throw 'Phase2B15 replay-runtimefix synthetic tests failed.' }

$B13Root = Join-Path $Repo 'ryzen_v1plus_phase2b13'
$B13Result = Join-Path $B13Root 'R7_5_ARCH_RESET_V1PLUS_PHASE2B13_ROOT_IID64_TARGET_TRAINING.json'
$B14Result = Join-Path $Repo 'ryzen_v1plus_phase2b14/R7_5_ARCH_RESET_V1PLUS_PHASE2B14_B13_RESIDUAL_LOCALIZATION.json'
$Heldout = Join-Path $Repo 'heldout_v3_bundle'
foreach ($Path in @($B13Result, $B14Result)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing exact corrected Phase2B15 prerequisite result: $Path" }
}
if (-not (Test-Path $Heldout -PathType Container)) { throw "Missing frozen heldout bundle: $Heldout" }

Write-Host '[V1+ Phase2B15 runtimefix] validating exact Phase2B13/Phase2B14 identities...'
& $Python -c "import hashlib,json,sys; p13,p14,e13,e14=sys.argv[1:5]; expected=('6de7996282236d34adf5e8e53416fd8a443a1fbf5abc89fc807492d0cb3dbf80','7cd1886596d345abdcdef479775498eddf7e014205de86e44afb5bb0ea291f86'); rows=[];`nfor p,e,x in zip((p13,p14),(e13,e14),expected):`n raw=open(p,'rb').read(); h=hashlib.sha256(raw).hexdigest(); ev=json.load(open(e,encoding='utf-8')); assert h==x==ev['uploaded_result_sha256']; rows.append(h);`nj13=json.load(open(p13,encoding='utf-8')); j14=json.load(open(p14,encoding='utf-8')); assert j13['status']=='ROOT_IID64_TRAINING_EFFECT_NOT_SUPPORTED'; assert j14['status']=='PREFLOP_CONTINUATION_RESIDUAL_DOMINANT_AFTER_ROOT_IID64'; assert j14['decision']['root_effect_consistent'] is True; print('Phase2B13/B14 exact prerequisite evidence PASS',rows)" $B13Result $B14Result $B13Evidence $B14Evidence
if ($LASTEXITCODE -ne 0) { throw 'Corrected Phase2B15 prerequisite evidence preflight failed.' }

Write-Host '[V1+ Phase2B15 runtimefix] validating authoritative H2/3H source/model contract...'
& $Python -c "import sys; from spincore.r7_5_representation_v3 import H2_FINAL; from spincore.r7_5_representation_v3_stage_contract import validate_phase2_v3_contract; [validate_phase2_v3_contract(sys.argv[1],representation=H2_FINAL,domain='THREE_HANDED',training_seed=s) for s in (1342191342,1801739323)]; print('H2/3H frozen source contract PASS')" $Repo
if ($LASTEXITCODE -ne 0) { throw 'Corrected Phase2B15 source/model contract preflight failed.' }

Write-Host '[V1+ Phase2B15 runtimefix] validating exact final Phase2B13 candidate behavior checkpoints...'
& $Python -c "import hashlib,sys,torch; from pathlib import Path; root=Path(sys.argv[1]); rows=[];`nfor s in (1342191342,1801739323):`n p=root/'IID64_MEAN_CANDIDATE'/f'seed_{s}'/'resume_checkpoint.pt'; assert p.is_file(),p; q=torch.load(p,map_location='cpu',weights_only=False); e=q['extra']; st=e['stage_state']; pr=q['progress']; assert q['schema']=='SPINCORE_R7_5_3C_REPRESENTATION_V3_CHECKPOINT_V1'; assert q['execution_sha']=='2cd7d1ece46a20d2b8937fe5135a415f6bbe54c2'; assert q['seed']==s and q['domain']=='THREE_HANDED'; assert pr['phase']=='phase2b13_resume' and pr['iteration']==3 and pr['global_root']==384; assert e['schema']=='SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B13_RESUME_V1' and e['arm']=='IID64_MEAN_CANDIDATE' and e['k']==64 and e['stage_index']==6; assert st['completed_iteration']==3 and st['global_root']==384; assert len(e['behavior_model_states'])==4; rows.append((s,hashlib.sha256(p.read_bytes()).hexdigest()));`nprint('Phase2B13 final candidate behavior checkpoints PASS',rows)" $B13Root
if ($LASTEXITCODE -ne 0) { throw 'Corrected Phase2B15 behavior-checkpoint preflight failed.' }

$Build = Join-Path $Repo 'build_phase2b15_runtimefix'
if (Test-Path $Build) { Remove-Item -Recurse -Force $Build }
& cmake -S $Repo -B $Build -G 'Visual Studio 17 2022' -A x64
if ($LASTEXITCODE -ne 0) { throw 'Corrected Phase2B15 CMake x64 configure failed.' }
& cmake --build $Build --config Release --target spincore_solver_c --parallel
if ($LASTEXITCODE -ne 0) { throw 'Corrected Phase2B15 solver x64 build failed.' }
$SolverCandidates = @(@((Join-Path $Build 'Release/spincore_solver_c.dll'),(Join-Path $Build 'spincore_solver_c.dll')) | Where-Object { Test-Path $_ })
if ($SolverCandidates.Count -ne 1) { throw "Expected exactly one x64 solver DLL; found: $($SolverCandidates -join ', ')" }
$Solver = (Resolve-Path $SolverCandidates[0]).Path
& $Python -c "import struct,sys; p=sys.argv[1]; b=open(p,'rb').read(); assert b[:2]==b'MZ'; pe=struct.unpack_from('<I',b,0x3c)[0]; assert b[pe:pe+4]==b'PE\0\0'; m=struct.unpack_from('<H',b,pe+4)[0]; print(f'solver PE machine 0x{m:04X}'); assert m==0x8664" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Corrected Phase2B15 solver PE architecture preflight failed.' }
& $Python -c "import ctypes as C,sys; L=C.CDLL(sys.argv[1]); L.spincore_solver_c_abi_version.argtypes=[]; L.spincore_solver_c_abi_version.restype=C.c_int32; v=int(L.spincore_solver_c_abi_version()); print('solver ABI',v); assert v==2; assert getattr(L,'spincore_solver_state_neural_input_v3',None) is not None; assert getattr(L,'spincore_solver_state_create_v2_deal',None) is not None; assert getattr(L,'spincore_solver_state_deal_snapshot_v1',None) is not None; print('SPNNIV3 + explicit-deal diagnostic solver API PASS')" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Corrected Phase2B15 solver ABI/explicit-deal preflight failed.' }

Write-Host '[V1+ Phase2B15 runtimefix] rerunning explicit-deal solver round-trip tests...'
& $Python $B10Test --repo-root $Repo --solver $Solver
if ($LASTEXITCODE -ne 0) { throw 'Corrected Phase2B15 explicit-deal solver round-trip tests failed.' }

Write-Host '[V1+ Phase2B15 runtimefix] proving all 64 heldout continuation anchors via explicit canonical replay...'
& $Python $Tool --preflight-only --repo-root $Repo --solver $Solver --heldout-root $Heldout --phase2b14-result $B14Result
if ($LASTEXITCODE -ne 0) { throw 'Corrected Phase2B15 64-anchor canonical replay preflight failed.' }

$Output = Join-Path $Repo 'ryzen_v1plus_phase2b15'
$Result = Join-Path $Output 'R7_5_ARCH_RESET_V1PLUS_PHASE2B15_POSTERIOR_WEIGHTED_CONTINUATION_CHANCE.json'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Workers = 30

Write-Host "[V1+ Phase2B15 runtimefix] screen HEAD: $Head"
Write-Host "[V1+ Phase2B15 runtimefix] workers: up to $Workers independent one-thread continuation-target processes"
Write-Host '[V1+ Phase2B15 runtimefix] correction only: historical Linux deck_seed replay is replaced by byte-verified suit-isomorphic explicit actor-card replay.'
Write-Host '[V1+ Phase2B15 runtimefix] scientific design remains frozen: 64 anchors x 2 behavior seeds x 2 blocks x K64 = 16384 target traversals.'
Write-Host '[V1+ Phase2B15 runtimefix] old pre-correction partials are rejected by runtimefix schema identity.'
Write-Host '[V1+ Phase2B15 runtimefix] no network fit, optimizer step, reservoir mutation, full-x4 confirmation, or production training.'

& $Python (Join-Path $Repo 'tools/spincore_ryzen_frozen_runner.py') `
    --expected-commit $Head `
    --run-name 'r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance_runtimefix' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B15_POSTERIOR_WEIGHTED_CONTINUATION_CHANCE_PRECOMMIT_20260824.md' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B15_WINDOWS_HELDOUT_REPLAY_RUNTIME_CORRECTION_20260824.md' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B14_RESULT_EVIDENCE_20260824.json' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B13_RESULT_EVIDENCE_20260824.json' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance_runtimefix.py' `
    --contract 'tools/test_r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance.py' `
    --contract 'tools/test_r7_5_arch_reset_v1plus_phase2b15_windows_replay_runtimefix.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b14_b13_residual_localization.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition.py' `
    --contract 'python/spincore/r7_5_representation_v3_referee_states.py' `
    --contract 'python/spincore/r7_5_representation_v3_referee_artifacts.py' `
    --contract 'python/spincore/solver.py' `
    --contract 'include/spincore/neural_encoder_v3.hpp' `
    --contract 'src/neural_encoder_v3.cpp' `
    --contract 'include/spincore/solver_c_api.h' `
    --contract 'src/solver_c_api.cpp' `
    --artifact $Output `
    -- $Python $Tool `
        --repo-root $Repo `
        --solver $Solver `
        --heldout-root $Heldout `
        --phase2b13-root $B13Root `
        --phase2b13-result $B13Result `
        --phase2b14-result $B14Result `
        --output-root $Output `
        --execution-sha $Head `
        --workers $Workers

if ($LASTEXITCODE -ne 0) { throw "Corrected Phase2B15 failed with exit code $LASTEXITCODE. Preserve ryzen_v1plus_phase2b15 and all prior artifacts; rerun only after review." }
if (-not (Test-Path $Result -PathType Leaf)) { throw 'Corrected Phase2B15 returned success without result JSON.' }
$Hash = (Get-FileHash -Algorithm SHA256 $Result).Hash.ToLowerInvariant()
Write-Host ''
Write-Host '[V1+ Phase2B15 runtimefix] COMPLETE'
Write-Host "[V1+ Phase2B15 runtimefix] result: $Result"
Write-Host "[V1+ Phase2B15 runtimefix] SHA256: $Hash"
Write-Host '[V1+ Phase2B15 runtimefix] This remains the frozen read-only posterior chance screen, not production training. Send the result JSON back.'
