$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path '.').Path
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$Dirty = (& git -C $Repo status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) {
    throw "Tracked worktree is dirty. Untracked files are intentionally ignored. Stop and send this output before Phase2B9:`n$Dirty"
}

$Precommit = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B9_ROBUST_ADVANTAGE_REGRESSION_PRECOMMIT_20260824.md'
$B6Evidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B6_RESULT_EVIDENCE_20260823.json'
$B8Evidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B8_RESULT_EVIDENCE_20260824.json'
foreach ($Path in @($Precommit, $B6Evidence, $B8Evidence)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing frozen Phase2B9 contract/evidence: $Path" }
}

$Venv = Join-Path $Repo '.venv-r7_5_3c_x16'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python -PathType Leaf)) { throw "Missing existing frozen Phase2 Python environment: $Python" }
& $Python -c "import struct,sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9),sys.version; assert struct.calcsize('P')==8; assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print('python',sys.version.split()[0],'bits',struct.calcsize('P')*8,'torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Frozen Phase2B9 Python/Torch/Numpy runtime verification failed.' }

$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'

$Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b9_robust_advantage_regression.py'
$Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b9_robust_advantage_regression.py'
$B6Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py'
$B7Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b7_residual_localization.py'
Write-Host '[V1+ Phase2B9] compiling robust-regression scripts...'
& $Python -m py_compile $Tool $Test $B6Tool $B7Tool
if ($LASTEXITCODE -ne 0) { throw 'Phase2B9 py_compile failed.' }
Write-Host '[V1+ Phase2B9] running deterministic synthetic tests...'
& $Python $Test
if ($LASTEXITCODE -ne 0) { throw 'Phase2B9 synthetic tests failed.' }

$B6Root = Join-Path $Repo 'ryzen_v1plus_phase2b6'
$B6Result = Join-Path $B6Root 'R7_5_ARCH_RESET_V1PLUS_PHASE2B6_PREFLOP_DAMPING_TRAINING_PILOT.json'
$B8Root = Join-Path $Repo 'ryzen_v1plus_phase2b8'
$B8Result = Join-Path $B8Root 'R7_5_ARCH_RESET_V1PLUS_PHASE2B8_LAGGED_PREFLOP_ANCHOR.json'
$Heldout = Join-Path $Repo 'heldout_v3_bundle'
foreach ($Path in @($B6Result, $B8Result)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing exact completed prerequisite result: $Path" }
}
if (-not (Test-Path $Heldout -PathType Container)) { throw "Missing frozen heldout bundle: $Heldout" }

Write-Host '[V1+ Phase2B9] validating exact Phase2B6/Phase2B8 prerequisite identities...'
& $Python -c "import hashlib,json,sys; p6,p8,e6,e8=sys.argv[1:5]; r6=open(p6,'rb').read(); r8=open(p8,'rb').read(); j6=json.loads(r6); j8=json.loads(r8); q6=json.load(open(e6,encoding='utf-8')); q8=json.load(open(e8,encoding='utf-8')); h6=hashlib.sha256(r6).hexdigest(); h8=hashlib.sha256(r8).hexdigest(); assert h6==q6['uploaded_result_sha256']=='33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a'; assert h8==q8['uploaded_result_sha256']=='1fd9144a488cea6de0a7500320d552abf994908b5200146d4baa4bd6f81c4d98'; assert j6['status']=='PREFLOP_DAMPING_CAUSAL_EFFECT_SUPPORTED_BUT_STILL_UNSTABLE'; assert j8['status']=='LAGGED_ANCHOR_EFFECT_NOT_SUPPORTED'; assert j8['decision']['equivalence_before_divergence_pass'] is True; print('Phase2B6/B8 exact prerequisite evidence PASS',h6,h8)" $B6Result $B8Result $B6Evidence $B8Evidence
if ($LASTEXITCODE -ne 0) { throw 'Phase2B9 prerequisite evidence preflight failed.' }

Write-Host '[V1+ Phase2B9] validating authoritative H2/3H model contract...'
& $Python -c "import sys; from spincore.r7_5_representation_v3 import H2_FINAL; from spincore.r7_5_representation_v3_stage_contract import validate_phase2_v3_contract; [validate_phase2_v3_contract(sys.argv[1],representation=H2_FINAL,domain='THREE_HANDED',training_seed=s) for s in (1342191342,1801739323)]; print('H2/3H frozen source contract PASS')" $Repo
if ($LASTEXITCODE -ne 0) { throw 'Phase2B9 source/model contract preflight failed.' }

Write-Host '[V1+ Phase2B9] validating exact completed Phase2B6 final Advantage memories/checkpoints...'
& $Python -c "import sys; from pathlib import Path; import r7_5_arch_reset_v1plus_phase2b9_robust_advantage_regression as p; root=Path(sys.argv[1]); b6=Path(sys.argv[2]); rows=[]; 
for s in (1342191342,1801739323):
  cp,bundle,states=p._load_phase2b6_checkpoint(root,b6,s); assert len(bundle.adv_mem.items)==100000 and len(states)==4; rows.append((s,p._sha256(cp),bundle.adv_mem.seen))
print('Phase2B6 final Advantage checkpoint identity PASS',rows)" $Repo $B6Root
if ($LASTEXITCODE -ne 0) { throw 'Phase2B9 Phase2B6 Advantage-memory/checkpoint preflight failed.' }

$Output = Join-Path $Repo 'ryzen_v1plus_phase2b9'
$Result = Join-Path $Output 'R7_5_ARCH_RESET_V1PLUS_PHASE2B9_ROBUST_ADVANTAGE_REGRESSION.json'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$SeedWorkers = 2
Write-Host "[V1+ Phase2B9] screen HEAD: $Head"
Write-Host "[V1+ Phase2B9] workers: $SeedWorkers independent seed processes x 2 Torch/OMP/MKL threads"
Write-Host '[V1+ Phase2B9] FIT-ONLY screen: exact frozen Phase2B6 Advantage memories; no solver traversal and no reservoir mutation.'
Write-Host '[V1+ Phase2B9] paired losses: canonical MSE vs Huber/Smooth-L1 beta=0.02; identical init and batch sequence per member.'
Write-Host '[V1+ Phase2B9] 2 seeds x 4 paired members x 4096 steps; completed member fits are resumable inside ryzen_v1plus_phase2b9.'
Write-Host '[V1+ Phase2B9] no AveragePolicy fit and no production training.'

& $Python (Join-Path $Repo 'tools/spincore_ryzen_frozen_runner.py') `
    --expected-commit $Head `
    --run-name 'r7_5_arch_reset_v1plus_phase2b9_robust_advantage_regression' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B9_ROBUST_ADVANTAGE_REGRESSION_PRECOMMIT_20260824.md' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B6_RESULT_EVIDENCE_20260823.json' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B8_RESULT_EVIDENCE_20260824.json' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b9_robust_advantage_regression.py' `
    --contract 'tools/test_r7_5_arch_reset_v1plus_phase2b9_robust_advantage_regression.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b7_residual_localization.py' `
    --contract 'python/spincore/r7_5_action_cfr.py' `
    --contract 'python/spincore/r7_5_action_uncertainty.py' `
    --contract 'python/spincore/r7_5_representation_v3.py' `
    --contract 'python/spincore/r7_5_representation_v3_checkpoint.py' `
    --contract 'python/spincore/r7_5_representation_v3_fit.py' `
    --contract 'python/spincore/r7_5_representation_v3_stage_contract.py' `
    --contract 'python/spincore_nn/models_v3_final.py' `
    --contract 'python/spincore_nn/training.py' `
    --artifact $Output `
    -- $Python $Tool `
        --repo-root $Repo `
        --phase2b6-root $B6Root `
        --phase2b6-result $B6Result `
        --phase2b8-result $B8Result `
        --heldout-root $Heldout `
        --output-root $Output `
        --execution-sha $Head `
        --seed-workers $SeedWorkers

if ($LASTEXITCODE -ne 0) { throw "Phase2B9 failed with exit code $LASTEXITCODE. Preserve ryzen_v1plus_phase2b9 and all prior artifacts; do not delete or restart from scratch." }
if (-not (Test-Path $Result -PathType Leaf)) { throw 'Phase2B9 returned success without result JSON.' }
$Hash = (Get-FileHash -Algorithm SHA256 $Result).Hash.ToLowerInvariant()
Write-Host ''
Write-Host '[V1+ Phase2B9] COMPLETE'
Write-Host "[V1+ Phase2B9] result: $Result"
Write-Host "[V1+ Phase2B9] SHA256: $Hash"
Write-Host '[V1+ Phase2B9] No solver traversal or production training was performed. Send the result JSON back for the next causal decision.'
