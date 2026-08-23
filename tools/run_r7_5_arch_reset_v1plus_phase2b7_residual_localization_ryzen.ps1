$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path '.').Path
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$Dirty = (& git -C $Repo status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) {
    throw "Tracked worktree is dirty. Untracked files are intentionally ignored. Stop and send this output before Phase2B7:`n$Dirty"
}

$Precommit = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B7_RESIDUAL_LOCALIZATION_PRECOMMIT_20260823.md'
$B6Evidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B6_RESULT_EVIDENCE_20260823.json'
$Phase2AEvidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2A_RESULT_EVIDENCE_20260822.json'
foreach ($Path in @($Precommit, $B6Evidence, $Phase2AEvidence)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing frozen Phase2B7 contract/evidence: $Path" }
}

$Venv = Join-Path $Repo '.venv-r7_5_3c_x16'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python -PathType Leaf)) { throw "Missing frozen Phase2 Python environment: $Python" }
& $Python -c "import struct,sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9),sys.version; assert struct.calcsize('P')==8; assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print('python',sys.version.split()[0],'bits',struct.calcsize('P')*8,'torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Frozen Phase2B7 Python/Torch/Numpy runtime verification failed.' }

$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'

$Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b7_residual_localization.py'
$Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b7_residual_localization.py'
$B6Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py'
Write-Host '[V1+ Phase2B7] compiling diagnostic scripts...'
& $Python -m py_compile $Tool $Test $B6Tool
if ($LASTEXITCODE -ne 0) { throw 'Phase2B7 py_compile failed.' }
Write-Host '[V1+ Phase2B7] running deterministic synthetic tests...'
& $Python $Test
if ($LASTEXITCODE -ne 0) { throw 'Phase2B7 synthetic tests failed.' }

$Phase2ARoot = Join-Path $Repo 'ryzen_v1plus_phase2a'
$Phase2AResult = Join-Path $Phase2ARoot 'R7_5_3D_V1PLUS_PHASE2A_RESULT.json'
$B6Root = Join-Path $Repo 'ryzen_v1plus_phase2b6'
$B6Result = Join-Path $B6Root 'R7_5_ARCH_RESET_V1PLUS_PHASE2B6_PREFLOP_DAMPING_TRAINING_PILOT.json'
$Heldout = Join-Path $Repo 'heldout_v3_bundle'
foreach ($Path in @($Phase2AResult, $B6Result)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing exact completed prerequisite result: $Path" }
}
if (-not (Test-Path $Heldout -PathType Container)) { throw "Missing frozen heldout bundle: $Heldout" }

Write-Host '[V1+ Phase2B7] validating exact Phase2A/Phase2B6 result identities...'
& $Python -c "import hashlib,json,sys; p2a,p6,e2a,e6=sys.argv[1:5]; r2a=open(p2a,'rb').read(); r6=open(p6,'rb').read(); j2a=json.loads(r2a); j6=json.loads(r6); q2a=json.load(open(e2a,encoding='utf-8')); q6=json.load(open(e6,encoding='utf-8')); h2a=hashlib.sha256(r2a).hexdigest(); h6=hashlib.sha256(r6).hexdigest(); assert h2a==q2a['uploaded_result_sha256']=='65f691e6b9cf7fbbddf88852c5ac6e0dcd2211af45f53cc4bb3e8271dbaa6149'; assert h6==q6['uploaded_result_sha256']=='33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a'; assert j2a['status']=='CAPACITY_EFFECT_NOT_SUPPORTED'; assert j6['status']=='PREFLOP_DAMPING_CAUSAL_EFFECT_SUPPORTED_BUT_STILL_UNSTABLE'; assert j6['decision']['next_route']=='LOCALIZE_RESIDUAL_WITHOUT_ESCALATING_DAMPING_FLOOR'; print('Phase2A/B6 exact prerequisite evidence PASS',h2a,h6)" $Phase2AResult $B6Result $Phase2AEvidence $B6Evidence
if ($LASTEXITCODE -ne 0) { throw 'Phase2B7 prerequisite evidence preflight failed.' }

Write-Host '[V1+ Phase2B7] validating authoritative H2/3H source/model contract...'
& $Python -c "import sys; from spincore.r7_5_representation_v3 import H2_FINAL; from spincore.r7_5_representation_v3_stage_contract import validate_phase2_v3_contract; [validate_phase2_v3_contract(sys.argv[1],representation=H2_FINAL,domain='THREE_HANDED',training_seed=s) for s in (1342191342,1801739323)]; print('H2/3H frozen source contract PASS')" $Repo
if ($LASTEXITCODE -ne 0) { throw 'Phase2B7 source/model contract preflight failed.' }

Write-Host '[V1+ Phase2B7] validating exact local Phase2B6 policy artifacts...'
& $Python -c "import hashlib,json,sys; from pathlib import Path; root=Path(sys.argv[1]); rows=[]; 
for s in (1342191342,1801739323):
  sr=json.load(open(root/f'seed_{s}'/'seed_result.json',encoding='utf-8')); assert sr['status']=='SEED_COMPLETE' and sr['execution_sha']=='4fa96434321c32efc734a55ae75982018ff2d091';
  for m in ('COMMON_LEARNER','NATIVE_LEARNER'):
    meta=json.load(open(root/f'seed_{s}'/'policies'/f'{m}.json',encoding='utf-8')); art=root/f'seed_{s}'/'policies'/f'{m}.pt'; h=hashlib.sha256(art.read_bytes()).hexdigest(); assert meta['status']=='POLICY_FIT_COMPLETE' and meta['artifact_sha256']==h and meta['floor_training']==0.25 and meta['floor_inference']==0.0; rows.append((s,m,h))
print('Phase2B6 exact local policies PASS',len(rows))" $B6Root
if ($LASTEXITCODE -ne 0) { throw 'Phase2B7 Phase2B6 policy-artifact preflight failed.' }

$Output = Join-Path $Repo 'ryzen_v1plus_phase2b7'
$Result = Join-Path $Output 'R7_5_ARCH_RESET_V1PLUS_PHASE2B7_RESIDUAL_LOCALIZATION.json'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
Write-Host "[V1+ Phase2B7] diagnostic HEAD: $Head"
Write-Host '[V1+ Phase2B7] READ-ONLY: 2 learner modes x 2 heldouts x 1024 states; no solver traversal, no training, no reservoir mutation.'
Write-Host '[V1+ Phase2B7] purpose: localize the residual after the successful 25% causal pilot without escalating the floor.'

& $Python (Join-Path $Repo 'tools/spincore_ryzen_frozen_runner.py') `
    --expected-commit $Head `
    --run-name 'r7_5_arch_reset_v1plus_phase2b7_residual_localization' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B7_RESIDUAL_LOCALIZATION_PRECOMMIT_20260823.md' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B6_RESULT_EVIDENCE_20260823.json' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2A_RESULT_EVIDENCE_20260822.json' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b7_residual_localization.py' `
    --contract 'tools/test_r7_5_arch_reset_v1plus_phase2b7_residual_localization.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py' `
    --contract 'python/spincore/r7_5_representation_v3_referee_artifacts.py' `
    --contract 'python/spincore/r7_5_representation_v3_stage_contract.py' `
    --artifact $Output `
    -- $Python $Tool `
        --repo-root $Repo `
        --heldout-root $Heldout `
        --phase2a-root $Phase2ARoot `
        --phase2a-result $Phase2AResult `
        --phase2b6-root $B6Root `
        --phase2b6-result $B6Result `
        --out $Result

if ($LASTEXITCODE -ne 0) { throw "Phase2B7 failed with exit code $LASTEXITCODE. Preserve all Phase2A/B6 artifacts; do not retrain anything." }
if (-not (Test-Path $Result -PathType Leaf)) { throw 'Phase2B7 returned success without result JSON.' }
$Hash = (Get-FileHash -Algorithm SHA256 $Result).Hash.ToLowerInvariant()
Write-Host ''
Write-Host '[V1+ Phase2B7] COMPLETE'
Write-Host "[V1+ Phase2B7] result: $Result"
Write-Host "[V1+ Phase2B7] SHA256: $Hash"
Write-Host '[V1+ Phase2B7] No training was performed. Send the result JSON back for the next causal design decision.'
