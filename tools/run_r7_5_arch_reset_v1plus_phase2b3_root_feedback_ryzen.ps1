$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path '.').Path
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$Dirty = (& git -C $Repo status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) {
    throw "Tracked worktree is dirty. Commit/revert tracked changes before Phase2B3.`n$Dirty"
}

$Precommit = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B3_ROOT_BASELINE_CONTINUATION_PRECOMMIT_20260822.md'
$Phase2B2Evidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B2_RESULT_EVIDENCE_20260822.json'
$Phase2B1Evidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B1_RESULT_EVIDENCE_20260822.json'
$Phase2AEvidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2A_RESULT_EVIDENCE_20260822.json'
foreach ($Path in @($Precommit, $Phase2B2Evidence, $Phase2B1Evidence, $Phase2AEvidence)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing frozen Phase2B3 contract/evidence: $Path" }
}

$Venv = Join-Path $Repo '.venv-r7_5_3c_x16'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python -PathType Leaf)) {
    throw "Missing existing frozen Phase2 Python environment: $Python"
}
& $Python -c "import struct,sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9),sys.version; assert struct.calcsize('P')==8; assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print('python',sys.version.split()[0],'bits',struct.calcsize('P')*8,'torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Frozen Phase2B3 Python/Torch/Numpy runtime verification failed.' }

$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

$Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b3_root_feedback_decomposition.py'
$Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b3_root_feedback_decomposition.py'
Write-Host '[V1+ Phase2B3] compiling diagnostic scripts...'
& $Python -m py_compile $Tool $Test
if ($LASTEXITCODE -ne 0) { throw 'Phase2B3 py_compile failed.' }
Write-Host '[V1+ Phase2B3] running deterministic synthetic tests...'
& $Python $Test
if ($LASTEXITCODE -ne 0) { throw 'Phase2B3 synthetic tests failed.' }

$Phase2B1 = Join-Path $Repo 'ryzen_v1plus_phase2b1/R7_5_ARCH_RESET_V1PLUS_PHASE2B1_TARGET_VARIANCE.json'
$Phase2B2 = Join-Path $Repo 'ryzen_v1plus_phase2b2/R7_5_ARCH_RESET_V1PLUS_PHASE2B2_COMMON_CHANCE_FEEDBACK.json'
if (-not (Test-Path $Phase2B1 -PathType Leaf)) { throw "Missing local Phase2B1 result: $Phase2B1" }
if (-not (Test-Path $Phase2B2 -PathType Leaf)) { throw "Missing local Phase2B2 result: $Phase2B2" }

Write-Host '[V1+ Phase2B3] validating exact Phase2B1/Phase2B2 local evidence identities...'
& $Python -c "import hashlib,json,sys; b1=open(sys.argv[1],'rb').read(); b2=open(sys.argv[2],'rb').read(); e1=json.load(open(sys.argv[3],encoding='utf-8')); e2=json.load(open(sys.argv[4],encoding='utf-8')); h1=hashlib.sha256(b1).hexdigest(); h2=hashlib.sha256(b2).hexdigest(); d1=json.loads(b1); d2=json.loads(b2); assert h1==e1['uploaded_result_sha256'],(h1,e1['uploaded_result_sha256']); assert h2==e2['uploaded_result_sha256'],(h2,e2['uploaded_result_sha256']); assert d1['decision']['source_classification']=='CHANCE_DOMINANT'; assert d2['status']=='MIXED_CHANCE_SUPPORT_AND_FEEDBACK'; assert d2['decision']['next_route']=='LOCALIZE_REMAINING_FEEDBACK_BEFORE_TRAINING'; print('Phase2B1/B2 exact local result evidence PASS',h1,h2)" $Phase2B1 $Phase2B2 $Phase2B1Evidence $Phase2B2Evidence
if ($LASTEXITCODE -ne 0) { throw 'Phase2B3 source-result evidence preflight failed.' }

Write-Host '[V1+ Phase2B3] validating authoritative H2/3H source/model contract...'
& $Python -c "import sys; from spincore.r7_5_representation_v3 import H2_FINAL; from spincore.r7_5_representation_v3_stage_contract import validate_phase2_v3_contract; [validate_phase2_v3_contract(sys.argv[1],representation=H2_FINAL,domain='THREE_HANDED',training_seed=s) for s in (1342191342,1801739323)]; print('H2/3H frozen source contract PASS')" $Repo
if ($LASTEXITCODE -ne 0) { throw 'Phase2B3 source/model contract preflight failed.' }

$Build = Join-Path $Repo 'build_phase2b3'
if (Test-Path $Build) { Remove-Item -Recurse -Force $Build }
& cmake -S $Repo -B $Build -G 'Visual Studio 17 2022' -A x64
if ($LASTEXITCODE -ne 0) { throw 'Phase2B3 CMake x64 configure failed.' }
& cmake --build $Build --config Release --target spincore_solver_c --parallel
if ($LASTEXITCODE -ne 0) { throw 'Phase2B3 solver x64 build failed.' }
$SolverCandidates = @(
    @(
        (Join-Path $Build 'Release/spincore_solver_c.dll'),
        (Join-Path $Build 'spincore_solver_c.dll')
    ) | Where-Object { Test-Path $_ }
)
if ($SolverCandidates.Count -ne 1) { throw "Expected exactly one x64 solver DLL; found: $($SolverCandidates -join ', ')" }
$Solver = (Resolve-Path $SolverCandidates[0]).Path
& $Python -c "import struct,sys; p=sys.argv[1]; b=open(p,'rb').read(); assert b[:2]==b'MZ'; pe=struct.unpack_from('<I',b,0x3c)[0]; assert b[pe:pe+4]==b'PE\0\0'; m=struct.unpack_from('<H',b,pe+4)[0]; print(f'solver PE machine 0x{m:04X}'); assert m==0x8664" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2B3 solver PE architecture preflight failed.' }
& $Python -c "import ctypes as C,sys; L=C.CDLL(sys.argv[1]); L.spincore_solver_c_abi_version.argtypes=[]; L.spincore_solver_c_abi_version.restype=C.c_int32; v=int(L.spincore_solver_c_abi_version()); print('solver ABI',v); assert v==2; assert getattr(L,'spincore_solver_state_neural_input_v3',None) is not None; print('SPNNIV3 solver API PASS')" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2B3 solver ABI/SPNNIV3 preflight failed.' }

$Input = Join-Path $Repo 'ryzen_v1plus_phase2a'
if (-not (Test-Path $Input -PathType Container)) { throw "Missing completed Phase2A output: $Input" }
$SourceExecutionSha = '4bfa55d69029cd69536fa6dbfcadd162719cb887'
Write-Host '[V1+ Phase2B3] validating both frozen Phase2A source behavior ensembles...'
& $Python -c "import sys,torch; from pathlib import Path; root=Path(sys.argv[1]); sha=sys.argv[2]; seeds=(1342191342,1801739323); rows=[];`nfor s in seeds:`n q=torch.load(root/f'seed_{s}'/'resume_checkpoint.pt',map_location='cpu',weights_only=False); e=q.get('extra',{}); assert q.get('execution_sha')==sha and int(q.get('seed',-1))==s; assert q.get('representation')=='H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL' and q.get('domain')=='THREE_HANDED'; assert q.get('progress',{}).get('phase')=='phase2a_resume' and int(q.get('progress',{}).get('global_root',-1))==768; assert int(e.get('stage_index',-1))==12 and len(e.get('behavior_model_states') or [])==4; rows.append((s,len(e['behavior_model_states'])));`nprint('Phase2B3 source ensemble identity PASS',rows)" $Input $SourceExecutionSha
if ($LASTEXITCODE -ne 0) { throw 'Phase2B3 Phase2A ensemble identity preflight failed.' }

$Output = Join-Path $Repo 'ryzen_v1plus_phase2b3'
$Result = Join-Path $Output 'R7_5_ARCH_RESET_V1PLUS_PHASE2B3_ROOT_FEEDBACK.json'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Workers = 12

Write-Host "[V1+ Phase2B3] diagnostic HEAD: $Head"
Write-Host "[V1+ Phase2B3] solver: $Solver"
Write-Host "[V1+ Phase2B3] workers: $Workers independent processes x 1 Torch/OMP/MKL thread"
Write-Host '[V1+ Phase2B3] exact work: 15 scenarios x 16 stored deals x 2 source behaviors = 480 root-component reconstructions.'
Write-Host '[V1+ Phase2B3] READ-ONLY LEARNED STATE; fresh solver traversal only, no reservoir insertion, optimizer step, model fit, or checkpoint mutation.'

& $Python (Join-Path $Repo 'tools/spincore_ryzen_frozen_runner.py') `
    --expected-commit $Head `
    --run-name 'r7_5_arch_reset_v1plus_phase2b3_root_feedback' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B3_ROOT_BASELINE_CONTINUATION_PRECOMMIT_20260822.md' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B2_RESULT_EVIDENCE_20260822.json' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B1_RESULT_EVIDENCE_20260822.json' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2A_RESULT_EVIDENCE_20260822.json' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b3_root_feedback_decomposition.py' `
    --contract 'tools/test_r7_5_arch_reset_v1plus_phase2b3_root_feedback_decomposition.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b2_common_chance_feedback.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b1_target_variance.py' `
    --contract 'python/spincore/r7_5_action_cfr.py' `
    --contract 'python/spincore/r7_5_representation_v3.py' `
    --contract 'python/spincore/r7_5_representation_v3_uncertainty.py' `
    --artifact $Output `
    -- $Python $Tool `
        --repo-root $Repo `
        --solver $Solver `
        --input-root $Input `
        --phase2b1-result $Phase2B1 `
        --phase2b2-result $Phase2B2 `
        --source-execution-sha $SourceExecutionSha `
        --workers $Workers `
        --out $Result

if ($LASTEXITCODE -ne 0) {
    throw "Phase2B3 failed with exit code $LASTEXITCODE. Preserve Phase2A/B1/B2 artifacts; do not start a training run."
}
if (-not (Test-Path $Result -PathType Leaf)) { throw 'Phase2B3 returned success without result JSON.' }
$Hash = (Get-FileHash -Algorithm SHA256 $Result).Hash.ToLowerInvariant()
Write-Host ''
Write-Host '[V1+ Phase2B3] COMPLETE'
Write-Host "[V1+ Phase2B3] result: $Result"
Write-Host "[V1+ Phase2B3] SHA256: $Hash"
Write-Host '[V1+ Phase2B3] No training was performed. Send the result JSON back for the next causal decision.'
