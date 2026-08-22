$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path '.').Path
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$Dirty = (& git -C $Repo status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) {
    throw "Tracked worktree is dirty. Commit/revert tracked changes before Advantage forensic.`n$Dirty"
}

$Precommit = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_ADVANTAGE_FORENSIC_PRECOMMIT_20260822.md'
$Phase2AEvidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2A_RESULT_EVIDENCE_20260822.json'
foreach ($Path in @($Precommit, $Phase2AEvidence)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing frozen architecture-reset contract/evidence: $Path" }
}

$Venv = Join-Path $Repo '.venv-r7_5_3c_x16'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python -PathType Leaf)) {
    throw "Missing existing frozen Python environment: $Python"
}
& $Python -c "import struct,sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9),sys.version; assert struct.calcsize('P')==8; assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print('python',sys.version.split()[0],'bits',struct.calcsize('P')*8,'torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Frozen Advantage-forensic runtime verification failed.' }

$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

Write-Host '[V1+ Advantage forensic] running deterministic synthetic tests...'
& $Python (Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_advantage_forensic.py')
if ($LASTEXITCODE -ne 0) { throw 'Advantage forensic synthetic tests failed.' }

$Input = Join-Path $Repo 'ryzen_v1plus_phase2a'
$Heldout = Join-Path $Repo 'heldout_v3_bundle'
if (-not (Test-Path $Input -PathType Container)) { throw "Missing completed Phase2A output: $Input" }
if (-not (Test-Path $Heldout -PathType Container)) { throw "Missing frozen heldout bundle: $Heldout" }

$SourceExecutionSha = '4bfa55d69029cd69536fa6dbfcadd162719cb887'
$Seeds = @(1342191342, 1801739323)
foreach ($Seed in $Seeds) {
    $SeedRoot = Join-Path $Input "seed_$Seed"
    $Checkpoint = Join-Path $SeedRoot 'resume_checkpoint.pt'
    $SeedResult = Join-Path $SeedRoot 'seed_result.json'
    if (-not (Test-Path $Checkpoint -PathType Leaf)) { throw "Missing Phase2A resume checkpoint: $Checkpoint" }
    if (-not (Test-Path $SeedResult -PathType Leaf)) { throw "Missing Phase2A seed result: $SeedResult" }
}

Write-Host '[V1+ Advantage forensic] validating frozen source checkpoint identities...'
& $Python -c "import json,sys,torch; from pathlib import Path; root=Path(sys.argv[1]); sha=sys.argv[2]; seeds=(1342191342,1801739323); rows=[];`nfor s in seeds:`n p=root/f'seed_{s}'/'resume_checkpoint.pt'; r=root/f'seed_{s}'/'seed_result.json'; q=torch.load(p,map_location='cpu',weights_only=False); j=json.loads(r.read_text()); assert q.get('execution_sha')==sha and int(q.get('seed',-1))==s; assert q.get('representation')=='H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL' and q.get('domain')=='THREE_HANDED'; assert q.get('progress',{}).get('phase')=='phase2a_resume' and int(q.get('progress',{}).get('global_root',-1))==768; assert int(q.get('extra',{}).get('stage_index',-1))==12; assert int(q.get('adv_mem',{}).get('capacity',-1))==100000; assert j.get('status')=='SEED_COMPLETE' and j.get('execution_sha')==sha and j.get('all_advantage_gates_pass') is True; rows.append((s,int(q['adv_mem']['seen']),len(q['adv_mem']['items'])));`nprint('Phase2A Advantage source identity PASS',rows)" $Input $SourceExecutionSha
if ($LASTEXITCODE -ne 0) { throw 'Phase2A Advantage source identity preflight failed.' }

$Output = Join-Path $Repo 'ryzen_v1plus_advantage_forensic'
$Result = Join-Path $Output 'R7_5_ARCH_RESET_V1PLUS_ADVANTAGE_FORENSIC.json'
New-Item -ItemType Directory -Force -Path $Output | Out-Null

Write-Host "[V1+ Advantage forensic] diagnostic HEAD: $Head"
Write-Host "[V1+ Advantage forensic] source execution SHA: $SourceExecutionSha"
Write-Host "[V1+ Advantage forensic] Phase2A input: $Input"
Write-Host "[V1+ Advantage forensic] heldout: $Heldout"
Write-Host "[V1+ Advantage forensic] output: $Result"
Write-Host '[V1+ Advantage forensic] READ ONLY: no solver traversal, no reservoir replay/mutation, no optimizer step, no model fit.'

& $Python (Join-Path $Repo 'tools/spincore_ryzen_frozen_runner.py') `
    --expected-commit $Head `
    --run-name 'r7_5_arch_reset_v1plus_advantage_forensic' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_ADVANTAGE_FORENSIC_PRECOMMIT_20260822.md' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2A_RESULT_EVIDENCE_20260822.json' `
    --contract 'tools/r7_5_arch_reset_v1plus_advantage_forensic.py' `
    --contract 'tools/test_r7_5_arch_reset_v1plus_advantage_forensic.py' `
    --contract 'python/spincore/r7_5_action_cfr.py' `
    --contract 'python/spincore/r7_5_representation_v3_checkpoint.py' `
    --contract 'python/spincore_nn/codec_v3.py' `
    --contract 'python/spincore_nn/models_v3_final.py' `
    --artifact $Output `
    -- $Python (Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_advantage_forensic.py') `
        --input-root $Input `
        --heldout-root $Heldout `
        --source-execution-sha $SourceExecutionSha `
        --out $Result

if ($LASTEXITCODE -ne 0) {
    throw "Advantage forensic failed with exit code $LASTEXITCODE. Preserve Phase2A artifacts; do not rerun training."
}
if (-not (Test-Path $Result -PathType Leaf)) { throw 'Advantage forensic returned success without result JSON.' }
$Hash = (Get-FileHash -Algorithm SHA256 $Result).Hash.ToLowerInvariant()
Write-Host ''
Write-Host '[V1+ Advantage forensic] COMPLETE'
Write-Host "[V1+ Advantage forensic] result: $Result"
Write-Host "[V1+ Advantage forensic] SHA256: $Hash"
Write-Host '[V1+ Advantage forensic] No training was performed and no causal remedy was selected.'
