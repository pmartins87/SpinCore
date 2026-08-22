$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path '.').Path
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$Dirty = (& git -C $Repo status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) {
    throw "Tracked worktree is dirty. Commit/revert tracked changes before Phase2B0.`n$Dirty"
}

$Precommit = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B0_POLICY_ALGEBRA_SCREEN_PRECOMMIT_20260822.md'
$ForensicEvidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_ADVANTAGE_FORENSIC_EVIDENCE_20260822.json'
$Phase2AEvidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2A_RESULT_EVIDENCE_20260822.json'
foreach ($Path in @($Precommit, $ForensicEvidence, $Phase2AEvidence)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing frozen contract/evidence: $Path" }
}

$Venv = Join-Path $Repo '.venv-r7_5_3c_x16'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python -PathType Leaf)) {
    throw "Missing existing frozen Python environment: $Python"
}
& $Python -c "import struct,sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9),sys.version; assert struct.calcsize('P')==8; assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print('python',sys.version.split()[0],'bits',struct.calcsize('P')*8,'torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Frozen Phase2B0 runtime verification failed.' }

$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

$Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b0_policy_algebra_screen.py'
$Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b0_policy_algebra_screen.py'
Write-Host '[V1+ Phase2B0] compiling diagnostic scripts...'
& $Python -m py_compile $Tool $Test
if ($LASTEXITCODE -ne 0) { throw 'Phase2B0 py_compile failed.' }
Write-Host '[V1+ Phase2B0] running deterministic synthetic tests...'
& $Python $Test
if ($LASTEXITCODE -ne 0) { throw 'Phase2B0 synthetic tests failed.' }

$Input = Join-Path $Repo 'ryzen_v1plus_phase2a'
$Heldout = Join-Path $Repo 'heldout_v3_bundle'
if (-not (Test-Path $Input -PathType Container)) { throw "Missing completed Phase2A output: $Input" }
if (-not (Test-Path $Heldout -PathType Container)) { throw "Missing frozen heldout bundle: $Heldout" }
$SourceExecutionSha = '4bfa55d69029cd69536fa6dbfcadd162719cb887'
$Seeds = @(1342191342, 1801739323)
foreach ($Seed in $Seeds) {
    $Checkpoint = Join-Path $Input "seed_$Seed\resume_checkpoint.pt"
    if (-not (Test-Path $Checkpoint -PathType Leaf)) { throw "Missing Phase2A checkpoint: $Checkpoint" }
}

Write-Host '[V1+ Phase2B0] validating source ensemble identities...'
& $Python -c "import sys,torch; from pathlib import Path; root=Path(sys.argv[1]); sha=sys.argv[2]; seeds=(1342191342,1801739323); rows=[];`nfor s in seeds:`n q=torch.load(root/f'seed_{s}'/'resume_checkpoint.pt',map_location='cpu',weights_only=False); e=q.get('extra',{}); assert q.get('execution_sha')==sha and int(q.get('seed',-1))==s; assert q.get('representation')=='H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL' and q.get('domain')=='THREE_HANDED'; assert q.get('progress',{}).get('phase')=='phase2a_resume' and int(q.get('progress',{}).get('global_root',-1))==768; assert int(e.get('stage_index',-1))==12 and len(e.get('behavior_model_states') or [])==4; rows.append((s,len(e['behavior_model_states'])));`nprint('Phase2B0 source ensemble identity PASS',rows)" $Input $SourceExecutionSha
if ($LASTEXITCODE -ne 0) { throw 'Phase2B0 source ensemble identity preflight failed.' }

$Output = Join-Path $Repo 'ryzen_v1plus_phase2b0'
$Result = Join-Path $Output 'R7_5_ARCH_RESET_V1PLUS_PHASE2B0_POLICY_ALGEBRA_SCREEN.json'
New-Item -ItemType Directory -Force -Path $Output | Out-Null

Write-Host "[V1+ Phase2B0] diagnostic HEAD: $Head"
Write-Host "[V1+ Phase2B0] source execution SHA: $SourceExecutionSha"
Write-Host "[V1+ Phase2B0] output: $Result"
Write-Host '[V1+ Phase2B0] READ ONLY: no solver traversal, no reservoir replay, no optimizer step, no model fit.'

& $Python (Join-Path $Repo 'tools/spincore_ryzen_frozen_runner.py') `
    --expected-commit $Head `
    --run-name 'r7_5_arch_reset_v1plus_phase2b0_policy_algebra_screen' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B0_POLICY_ALGEBRA_SCREEN_PRECOMMIT_20260822.md' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_ADVANTAGE_FORENSIC_EVIDENCE_20260822.json' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2A_RESULT_EVIDENCE_20260822.json' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b0_policy_algebra_screen.py' `
    --contract 'tools/test_r7_5_arch_reset_v1plus_phase2b0_policy_algebra_screen.py' `
    --contract 'python/spincore/r7_5_action_uncertainty.py' `
    --contract 'python/spincore/r7_5_action_cfr.py' `
    --contract 'python/spincore_nn/models_v3_final.py' `
    --artifact $Output `
    -- $Python $Tool `
        --input-root $Input `
        --heldout-root $Heldout `
        --source-execution-sha $SourceExecutionSha `
        --out $Result

if ($LASTEXITCODE -ne 0) {
    throw "Phase2B0 failed with exit code $LASTEXITCODE. Preserve Phase2A artifacts; do not rerun training."
}
if (-not (Test-Path $Result -PathType Leaf)) { throw 'Phase2B0 returned success without result JSON.' }
$Hash = (Get-FileHash -Algorithm SHA256 $Result).Hash.ToLowerInvariant()
Write-Host ''
Write-Host '[V1+ Phase2B0] COMPLETE'
Write-Host "[V1+ Phase2B0] result: $Result"
Write-Host "[V1+ Phase2B0] SHA256: $Hash"
Write-Host '[V1+ Phase2B0] No training was performed.'
