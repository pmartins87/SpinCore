$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path '.').Path
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$Dirty = (& git -C $Repo status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) {
    throw "Tracked worktree is dirty. Commit/revert tracked changes before Phase2A evaluation recovery.`n$Dirty"
}

$RecoveryFreeze = Join-Path $Repo 'validation/R7_5_3D_V1PLUS_PHASE2A_EVALUATION_LEGAL_MASK_RECOVERY_20260822.md'
$Phase2AFreeze = Join-Path $Repo 'validation/R7_5_3D_V1PLUS_PHASE2A_STRATEGY_MEMORY_CAPACITY_ABLATION_PRECOMMIT_20260821.md'
$ParallelFreeze = Join-Path $Repo 'validation/R7_5_3D_V1PLUS_PHASE2A_PARALLEL_FIT_EXECUTION_FREEZE_20260821.md'
foreach ($Path in @($RecoveryFreeze, $Phase2AFreeze, $ParallelFreeze)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing frozen Phase2A recovery contract: $Path" }
}

$Venv = Join-Path $Repo '.venv-r7_5_3c_x16'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python -PathType Leaf)) {
    throw "Missing existing frozen Python environment: $Python"
}
& $Python -c "import struct,sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9),sys.version; assert struct.calcsize('P')==8; assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print('python',sys.version.split()[0],'bits',struct.calcsize('P')*8,'torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Frozen Phase2A recovery runtime verification failed.' }

$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

Write-Host '[V1+ Phase2A recovery] verifying canonical legal-set conversion...'
& $Python -c "from spincore.r7_5_action_cfr import legal_mask; rows=((0,2,9),(1,4,7,8),(3,5)); masks=[legal_mask(x) for x in rows]; assert all(len(x)==10 for x in masks); assert [sum(x) for x in masks]==[3,4,2]; print('variable legal sets -> ten-slot masks PASS')"
if ($LASTEXITCODE -ne 0) { throw 'Canonical legal-mask recovery preflight failed.' }

$Output = Join-Path $Repo 'ryzen_v1plus_phase2a'
if (-not (Test-Path $Output -PathType Container)) {
    throw "Missing completed Phase2A source output: $Output"
}
$Result = Join-Path $Output 'R7_5_3D_V1PLUS_PHASE2A_RESULT.json'

Write-Host "[V1+ Phase2A recovery] recovery HEAD: $Head"
Write-Host '[V1+ Phase2A recovery] source execution SHA: 4bfa55d69029cd69536fa6dbfcadd162719cb887'
Write-Host '[V1+ Phase2A recovery] EVALUATION ONLY: no traversal, no reservoir replay, no policy refit.'

& $Python (Join-Path $Repo 'tools/spincore_ryzen_frozen_runner.py') `
    --expected-commit $Head `
    --run-name 'r7_5_3d_v1plus_phase2a_eval_recovery' `
    --contract 'validation/R7_5_3D_V1PLUS_PHASE2A_EVALUATION_LEGAL_MASK_RECOVERY_20260822.md' `
    --contract 'validation/R7_5_3D_V1PLUS_PHASE2A_STRATEGY_MEMORY_CAPACITY_ABLATION_PRECOMMIT_20260821.md' `
    --contract 'validation/R7_5_3D_V1PLUS_PHASE2A_PARALLEL_FIT_EXECUTION_FREEZE_20260821.md' `
    --contract 'tools/r7_5_3d_v1plus_phase2a_eval_recovery.py' `
    --contract 'tools/r7_5_3d_v1plus_phase2a_strategy_capacity.py' `
    --contract 'python/spincore/r7_5_action_cfr.py' `
    --contract 'python/spincore/r7_5_representation_v3_final_policy.py' `
    --artifact $Result `
    -- $Python (Join-Path $Repo 'tools/r7_5_3d_v1plus_phase2a_eval_recovery.py') `
        --repo-root $Repo `
        --heldout-root (Join-Path $Repo 'heldout_v3_bundle') `
        --output-root $Output `
        --out $Result

if ($LASTEXITCODE -ne 0) {
    throw "Phase2A evaluation-only recovery failed with exit code $LASTEXITCODE. Preserve all source artifacts; do not rerun training."
}
if (-not (Test-Path $Result -PathType Leaf)) { throw 'Recovery returned success without final Phase2A result JSON.' }
$Hash = (Get-FileHash -Algorithm SHA256 $Result).Hash.ToLowerInvariant()
Write-Host ''
Write-Host '[V1+ Phase2A recovery] COMPLETE'
Write-Host "[V1+ Phase2A recovery] result: $Result"
Write-Host "[V1+ Phase2A recovery] SHA256: $Hash"
Write-Host '[V1+ Phase2A recovery] No training was repeated.'
