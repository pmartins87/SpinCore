$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path '.').Path
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$Dirty = (& git -C $Repo status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) {
    throw "Tracked worktree is dirty. Untracked files are intentionally ignored. Stop and send this output before Phase2B8:`n$Dirty"
}

$Precommit = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B8_LAGGED_PREFLOP_ANCHOR_PRECOMMIT_20260823.md'
$B6Evidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B6_RESULT_EVIDENCE_20260823.json'
$B7Evidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B7_RESULT_EVIDENCE_20260823.json'
foreach ($Path in @($Precommit, $B6Evidence, $B7Evidence)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing frozen Phase2B8 contract/evidence: $Path" }
}

$Venv = Join-Path $Repo '.venv-r7_5_3c_x16'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python -PathType Leaf)) { throw "Missing existing frozen Phase2 Python environment: $Python" }
& $Python -c "import struct,sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9),sys.version; assert struct.calcsize('P')==8; assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print('python',sys.version.split()[0],'bits',struct.calcsize('P')*8,'torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Frozen Phase2B8 Python/Torch/Numpy runtime verification failed.' }

$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'

$Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b8_lagged_preflop_anchor.py'
$Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b8_lagged_preflop_anchor.py'
$B6Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py'
$B7Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b7_residual_localization.py'
$RuntimeFix = Join-Path $Repo 'tools/r7_5_3c_chance_coverage_x4_domain_worker_runtimefix.py'
$Phase2ABase = Join-Path $Repo 'tools/r7_5_3d_v1plus_phase2a_strategy_capacity.py'
Write-Host '[V1+ Phase2B8] compiling lagged-anchor scripts...'
& $Python -m py_compile $Tool $Test $B6Tool $B7Tool $RuntimeFix $Phase2ABase
if ($LASTEXITCODE -ne 0) { throw 'Phase2B8 py_compile failed.' }
Write-Host '[V1+ Phase2B8] running deterministic synthetic tests...'
& $Python $Test
if ($LASTEXITCODE -ne 0) { throw 'Phase2B8 synthetic tests failed.' }

$B6Root = Join-Path $Repo 'ryzen_v1plus_phase2b6'
$B6Result = Join-Path $B6Root 'R7_5_ARCH_RESET_V1PLUS_PHASE2B6_PREFLOP_DAMPING_TRAINING_PILOT.json'
$B7Root = Join-Path $Repo 'ryzen_v1plus_phase2b7'
$B7Result = Join-Path $B7Root 'R7_5_ARCH_RESET_V1PLUS_PHASE2B7_RESIDUAL_LOCALIZATION.json'
$Heldout = Join-Path $Repo 'heldout_v3_bundle'
foreach ($Path in @($B6Result, $B7Result)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing exact completed prerequisite result: $Path" }
}
if (-not (Test-Path $Heldout -PathType Container)) { throw "Missing frozen heldout bundle: $Heldout" }

Write-Host '[V1+ Phase2B8] validating exact Phase2B6/Phase2B7 prerequisite identities...'
& $Python -c "import hashlib,json,sys; p6,p7,e6,e7=sys.argv[1:5]; r6=open(p6,'rb').read(); r7=open(p7,'rb').read(); j6=json.loads(r6); j7=json.loads(r7); q6=json.load(open(e6,encoding='utf-8')); q7=json.load(open(e7,encoding='utf-8')); h6=hashlib.sha256(r6).hexdigest(); h7=hashlib.sha256(r7).hexdigest(); assert h6==q6['uploaded_result_sha256']=='33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a'; assert h7==q7['uploaded_result_sha256']=='ff55a5a047d62952e505b8e4d59d79d4016f30b6696a339318bc696dd6f77fe6'; assert j6['status']=='PREFLOP_DAMPING_CAUSAL_EFFECT_SUPPORTED_BUT_STILL_UNSTABLE'; assert j7['status']=='PREFLOP_CONTINUATION_DOMINANT'; assert j7['decision']['next_route']=='PRECOMMIT_EARLY_PREFLOP_LAGGED_TARGET_OR_ANCHOR_SCREEN'; print('Phase2B6/B7 exact prerequisite evidence PASS',h6,h7)" $B6Result $B7Result $B6Evidence $B7Evidence
if ($LASTEXITCODE -ne 0) { throw 'Phase2B8 prerequisite evidence preflight failed.' }

Write-Host '[V1+ Phase2B8] validating authoritative H2/3H source/model contract...'
& $Python -c "import sys; from spincore.r7_5_representation_v3 import H2_FINAL; from spincore.r7_5_representation_v3_stage_contract import validate_phase2_v3_contract; [validate_phase2_v3_contract(sys.argv[1],representation=H2_FINAL,domain='THREE_HANDED',training_seed=s) for s in (1342191342,1801739323)]; print('H2/3H frozen source contract PASS')" $Repo
if ($LASTEXITCODE -ne 0) { throw 'Phase2B8 source/model contract preflight failed.' }

Write-Host '[V1+ Phase2B8] validating exact completed Phase2B6 control policy artifacts...'
& $Python -c "import hashlib,json,sys; from pathlib import Path; root=Path(sys.argv[1]); rows=[]; 
for s in (1342191342,1801739323):
  sr=json.load(open(root/f'seed_{s}'/'seed_result.json',encoding='utf-8')); assert sr['status']=='SEED_COMPLETE' and sr['execution_sha']=='4fa96434321c32efc734a55ae75982018ff2d091';
  for m in ('COMMON_LEARNER','NATIVE_LEARNER'):
    meta=json.load(open(root/f'seed_{s}'/'policies'/f'{m}.json',encoding='utf-8')); art=root/f'seed_{s}'/'policies'/f'{m}.pt'; h=hashlib.sha256(art.read_bytes()).hexdigest(); assert meta['status']=='POLICY_FIT_COMPLETE' and meta['artifact_sha256']==h and meta['floor_training']==0.25 and meta['floor_inference']==0.0; rows.append((s,m,h))
print('Phase2B6 exact control policies PASS',len(rows))" $B6Root
if ($LASTEXITCODE -ne 0) { throw 'Phase2B8 Phase2B6 control-policy identity preflight failed.' }

$Build = Join-Path $Repo 'build_phase2b8'
if (Test-Path $Build) { Remove-Item -Recurse -Force $Build }
& cmake -S $Repo -B $Build -G 'Visual Studio 17 2022' -A x64
if ($LASTEXITCODE -ne 0) { throw 'Phase2B8 CMake x64 configure failed.' }
& cmake --build $Build --config Release --target spincore_solver_c --parallel
if ($LASTEXITCODE -ne 0) { throw 'Phase2B8 solver x64 build failed.' }
$SolverCandidates = @(@((Join-Path $Build 'Release/spincore_solver_c.dll'),(Join-Path $Build 'spincore_solver_c.dll')) | Where-Object { Test-Path $_ })
if ($SolverCandidates.Count -ne 1) { throw "Expected exactly one x64 solver DLL; found: $($SolverCandidates -join ', ')" }
$Solver = (Resolve-Path $SolverCandidates[0]).Path
& $Python -c "import struct,sys; p=sys.argv[1]; b=open(p,'rb').read(); assert b[:2]==b'MZ'; pe=struct.unpack_from('<I',b,0x3c)[0]; assert b[pe:pe+4]==b'PE\0\0'; m=struct.unpack_from('<H',b,pe+4)[0]; print(f'solver PE machine 0x{m:04X}'); assert m==0x8664" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2B8 solver PE architecture preflight failed.' }
& $Python -c "import ctypes as C,sys; L=C.CDLL(sys.argv[1]); L.spincore_solver_c_abi_version.argtypes=[]; L.spincore_solver_c_abi_version.restype=C.c_int32; v=int(L.spincore_solver_c_abi_version()); print('solver ABI',v); assert v==2; assert getattr(L,'spincore_solver_state_neural_input_v3',None) is not None; print('SPNNIV3 solver API PASS')" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2B8 solver ABI/SPNNIV3 preflight failed.' }

$Output = Join-Path $Repo 'ryzen_v1plus_phase2b8'
$Result = Join-Path $Output 'R7_5_ARCH_RESET_V1PLUS_PHASE2B8_LAGGED_PREFLOP_ANCHOR.json'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$SeedWorkers = 2
Write-Host "[V1+ Phase2B8] screen HEAD: $Head"
Write-Host "[V1+ Phase2B8] training workers: $SeedWorkers independent seed processes x 2 Torch/OMP/MKL threads"
Write-Host '[V1+ Phase2B8] exact candidate training: H2/3H only, 2 seeds x 3 iterations x 4 chunks x 64 roots = 1536 roots total.'
Write-Host '[V1+ Phase2B8] intervention: on preflop continuations only, 75% current native behavior + 25% previous learned behavior.'
Write-Host '[V1+ Phase2B8] root and postflop remain native; heldout inference anchor=0.00.'
Write-Host '[V1+ Phase2B8] iterations 1-2 must reproduce Phase2B6 before the intended iteration-3 divergence.'
Write-Host '[V1+ Phase2B8] exact Phase2B6 artifacts are read-only controls and will not be retrained or modified.'

& $Python (Join-Path $Repo 'tools/spincore_ryzen_frozen_runner.py') `
    --expected-commit $Head `
    --run-name 'r7_5_arch_reset_v1plus_phase2b8_lagged_preflop_anchor' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B8_LAGGED_PREFLOP_ANCHOR_PRECOMMIT_20260823.md' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B6_RESULT_EVIDENCE_20260823.json' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B7_RESULT_EVIDENCE_20260823.json' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b8_lagged_preflop_anchor.py' `
    --contract 'tools/test_r7_5_arch_reset_v1plus_phase2b8_lagged_preflop_anchor.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b7_residual_localization.py' `
    --contract 'tools/r7_5_3c_chance_coverage_x4_domain_worker_runtimefix.py' `
    --contract 'tools/r7_5_3d_v1plus_phase2a_strategy_capacity.py' `
    --contract 'python/spincore/r7_5_action_cfr.py' `
    --contract 'python/spincore/r7_5_representation_v3.py' `
    --contract 'python/spincore/r7_5_representation_v3_stage.py' `
    --contract 'python/spincore/r7_5_representation_v3_stage_contract.py' `
    --contract 'python/spincore/r7_5_representation_v3_uncertainty.py' `
    --artifact $Output `
    -- $Python $Tool `
        --repo-root $Repo `
        --solver $Solver `
        --heldout-root $Heldout `
        --phase2b6-root $B6Root `
        --phase2b6-result $B6Result `
        --phase2b7-result $B7Result `
        --output-root $Output `
        --execution-sha $Head `
        --seed-workers $SeedWorkers

if ($LASTEXITCODE -ne 0) { throw "Phase2B8 failed with exit code $LASTEXITCODE. Preserve ryzen_v1plus_phase2b8 and all Phase2B6/B7 artifacts; do not delete or restart from scratch." }
if (-not (Test-Path $Result -PathType Leaf)) { throw 'Phase2B8 returned success without result JSON.' }
$Hash = (Get-FileHash -Algorithm SHA256 $Result).Hash.ToLowerInvariant()
Write-Host ''
Write-Host '[V1+ Phase2B8] COMPLETE'
Write-Host "[V1+ Phase2B8] result: $Result"
Write-Host "[V1+ Phase2B8] SHA256: $Hash"
Write-Host '[V1+ Phase2B8] This is a small causal training screen, not production training. Send the result JSON back for the next stability/strength decision.'
