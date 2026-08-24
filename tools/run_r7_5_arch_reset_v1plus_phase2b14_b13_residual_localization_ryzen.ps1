$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path '.').Path
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$Dirty = (& git -C $Repo status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) {
    throw "Tracked worktree is dirty. Untracked files are intentionally ignored. Stop and send this output before Phase2B14:`n$Dirty"
}

$Precommit = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B14_B13_RESIDUAL_LOCALIZATION_PRECOMMIT_20260824.md'
$B13Evidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B13_RESULT_EVIDENCE_20260824.json'
foreach ($Path in @($Precommit, $B13Evidence)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing frozen Phase2B14 contract/evidence: $Path" }
}

$Venv = Join-Path $Repo '.venv-r7_5_3c_x16'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python -PathType Leaf)) { throw "Missing frozen Phase2 Python environment: $Python" }
& $Python -c "import struct,sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9),sys.version; assert struct.calcsize('P')==8; assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print('python',sys.version.split()[0],'bits',struct.calcsize('P')*8,'torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Frozen Phase2B14 Python/Torch/Numpy runtime verification failed.' }

$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'

$Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b14_b13_residual_localization.py'
$Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b14_b13_residual_localization.py'
$B13Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training.py'
$B7Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b7_residual_localization.py'
$B6Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py'

Write-Host '[V1+ Phase2B14] compiling diagnostic scripts...'
& $Python -m py_compile $Tool $Test $B13Tool $B7Tool $B6Tool
if ($LASTEXITCODE -ne 0) { throw 'Phase2B14 py_compile failed.' }
Write-Host '[V1+ Phase2B14] running deterministic synthetic tests...'
& $Python $Test
if ($LASTEXITCODE -ne 0) { throw 'Phase2B14 synthetic tests failed.' }

$B13Root = Join-Path $Repo 'ryzen_v1plus_phase2b13'
$B13Result = Join-Path $B13Root 'R7_5_ARCH_RESET_V1PLUS_PHASE2B13_ROOT_IID64_TARGET_TRAINING.json'
$Heldout = Join-Path $Repo 'heldout_v3_bundle'
if (-not (Test-Path $B13Result -PathType Leaf)) { throw "Missing exact completed Phase2B13 result: $B13Result" }
if (-not (Test-Path $Heldout -PathType Container)) { throw "Missing frozen heldout bundle: $Heldout" }

Write-Host '[V1+ Phase2B14] validating exact Phase2B13 result identity and frozen FAIL route...'
& $Python -c "import hashlib,json,sys; p,e=sys.argv[1:3]; raw=open(p,'rb').read(); h=hashlib.sha256(raw).hexdigest(); j=json.loads(raw); ev=json.load(open(e,encoding='utf-8')); assert h==ev['uploaded_result_sha256']=='6de7996282236d34adf5e8e53416fd8a443a1fbf5abc89fc807492d0cb3dbf80'; assert j['status']=='ROOT_IID64_TRAINING_EFFECT_NOT_SUPPORTED'; assert j['decision']['causal_effect_supported'] is False; assert j['decision']['common_materiality_pass'] is False; assert j['decision']['full_x4_confirmation_authorized'] is False; print('Phase2B13 exact result/route PASS',h)" $B13Result $B13Evidence
if ($LASTEXITCODE -ne 0) { throw 'Phase2B14 Phase2B13 evidence preflight failed.' }

Write-Host '[V1+ Phase2B14] validating authoritative H2/3H source/model contract...'
& $Python -c "import sys; from spincore.r7_5_representation_v3 import H2_FINAL; from spincore.r7_5_representation_v3_stage_contract import validate_phase2_v3_contract; [validate_phase2_v3_contract(sys.argv[1],representation=H2_FINAL,domain='THREE_HANDED',training_seed=s) for s in (1342191342,1801739323)]; print('H2/3H frozen source contract PASS')" $Repo
if ($LASTEXITCODE -ne 0) { throw 'Phase2B14 source/model contract preflight failed.' }

$Output = Join-Path $Repo 'ryzen_v1plus_phase2b14'
$Result = Join-Path $Output 'R7_5_ARCH_RESET_V1PLUS_PHASE2B14_B13_RESIDUAL_LOCALIZATION.json'
New-Item -ItemType Directory -Force -Path $Output | Out-Null

Write-Host "[V1+ Phase2B14] diagnostic HEAD: $Head"
Write-Host '[V1+ Phase2B14] READ-ONLY: saved Phase2B13 policies only; no solver traversal, no training, no target resampling, no reservoir mutation.'
Write-Host '[V1+ Phase2B14] first gate: reproduce all Phase2B13 heldout control/candidate mean+p95 metrics within 1e-12.'
Write-Host '[V1+ Phase2B14] purpose: determine whether residual instability remains root, preflop-continuation, postflop, scenario-concentrated, or broadly mixed.'

& $Python (Join-Path $Repo 'tools/spincore_ryzen_frozen_runner.py') `
    --expected-commit $Head `
    --run-name 'r7_5_arch_reset_v1plus_phase2b14_b13_residual_localization' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B14_B13_RESIDUAL_LOCALIZATION_PRECOMMIT_20260824.md' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B13_RESULT_EVIDENCE_20260824.json' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b14_b13_residual_localization.py' `
    --contract 'tools/test_r7_5_arch_reset_v1plus_phase2b14_b13_residual_localization.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b7_residual_localization.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py' `
    --contract 'python/spincore/r7_5_representation_v3_referee_artifacts.py' `
    --contract 'python/spincore/r7_5_representation_v3_stage_contract.py' `
    --artifact $Output `
    -- $Python $Tool `
        --repo-root $Repo `
        --heldout-root $Heldout `
        --phase2b13-root $B13Root `
        --phase2b13-result $B13Result `
        --out $Result

if ($LASTEXITCODE -ne 0) { throw "Phase2B14 failed with exit code $LASTEXITCODE. Preserve all Phase2B13 artifacts; do not retrain or delete anything." }
if (-not (Test-Path $Result -PathType Leaf)) { throw 'Phase2B14 returned success without result JSON.' }
$Hash = (Get-FileHash -Algorithm SHA256 $Result).Hash.ToLowerInvariant()
Write-Host ''
Write-Host '[V1+ Phase2B14] COMPLETE'
Write-Host "[V1+ Phase2B14] result: $Result"
Write-Host "[V1+ Phase2B14] SHA256: $Hash"
Write-Host '[V1+ Phase2B14] No training was performed. Send the result JSON back for the next causal design decision.'
