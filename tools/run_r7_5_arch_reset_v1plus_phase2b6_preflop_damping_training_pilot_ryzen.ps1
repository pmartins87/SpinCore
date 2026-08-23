$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path '.').Path
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$Dirty = (& git -C $Repo status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) {
    throw "Tracked worktree is dirty. Untracked files are intentionally ignored. Stop and send this output before Phase2B6:`n$Dirty"
}

$Precommit = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B6_PREFLOP_DAMPING_TRAINING_PILOT_PRECOMMIT_20260822.md'
$Phase2B5Evidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B5_RESULT_EVIDENCE_20260822.json'
$Phase2AEvidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2A_RESULT_EVIDENCE_20260822.json'
foreach ($Path in @($Precommit, $Phase2B5Evidence, $Phase2AEvidence)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing frozen Phase2B6 contract/evidence: $Path" }
}

$Venv = Join-Path $Repo '.venv-r7_5_3c_x16'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python -PathType Leaf)) { throw "Missing existing frozen Phase2 Python environment: $Python" }
& $Python -c "import struct,sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9),sys.version; assert struct.calcsize('P')==8; assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print('python',sys.version.split()[0],'bits',struct.calcsize('P')*8,'torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Frozen Phase2B6 Python/Torch/Numpy runtime verification failed.' }

$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'

$Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py'
$Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py'
$Phase2ABase = Join-Path $Repo 'tools/r7_5_3d_v1plus_phase2a_strategy_capacity.py'
$X4RuntimeFix = Join-Path $Repo 'tools/r7_5_3c_chance_coverage_x4_domain_worker_runtimefix.py'
Write-Host '[V1+ Phase2B6] compiling pilot scripts...'
& $Python -m py_compile $Tool $Test $Phase2ABase $X4RuntimeFix
if ($LASTEXITCODE -ne 0) { throw 'Phase2B6 py_compile failed.' }
Write-Host '[V1+ Phase2B6] running deterministic synthetic tests...'
& $Python $Test
if ($LASTEXITCODE -ne 0) { throw 'Phase2B6 synthetic tests failed.' }

$Phase2ARoot = Join-Path $Repo 'ryzen_v1plus_phase2a'
$Phase2AResult = Join-Path $Phase2ARoot 'R7_5_3D_V1PLUS_PHASE2A_RESULT.json'
$Phase2B5 = Join-Path $Repo 'ryzen_v1plus_phase2b5/R7_5_ARCH_RESET_V1PLUS_PHASE2B5_PREFLOP_FEEDBACK_STABILIZATION.json'
$Heldout = Join-Path $Repo 'heldout_v3_bundle'
foreach ($Path in @($Phase2AResult, $Phase2B5)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing exact completed prerequisite result: $Path" }
}
if (-not (Test-Path $Heldout -PathType Container)) { throw "Missing frozen heldout bundle: $Heldout" }

Write-Host '[V1+ Phase2B6] validating exact Phase2A/Phase2B5 prerequisite result identities...'
& $Python -c "import hashlib,json,sys; p2a,p5,e2a,e5=sys.argv[1:5]; r2a=open(p2a,'rb').read(); r5=open(p5,'rb').read(); j2a=json.loads(r2a); j5=json.loads(r5); q2a=json.load(open(e2a,encoding='utf-8')); q5=json.load(open(e5,encoding='utf-8')); h2a=hashlib.sha256(r2a).hexdigest(); h5=hashlib.sha256(r5).hexdigest(); assert h2a==q2a['uploaded_result_sha256']=='65f691e6b9cf7fbbddf88852c5ac6e0dcd2211af45f53cc4bb3e8271dbaa6149'; assert h5==q5['uploaded_result_sha256']=='0fb028c02dbbea0c4fa7a323a3edeed5c4e12789145235be2e851452e16ab5b8'; assert j2a['status']=='CAPACITY_EFFECT_NOT_SUPPORTED'; assert j5['status']=='MILD_PREFLOP_DAMPING_CANDIDATE'; assert j5['decision']['selected_mild_candidate']=='UNIFORM_FLOOR_025'; assert j5['decision']['small_training_pilot_precommit_allowed'] is True; print('Phase2A/B5 exact prerequisite evidence PASS',h2a,h5)" $Phase2AResult $Phase2B5 $Phase2AEvidence $Phase2B5Evidence
if ($LASTEXITCODE -ne 0) { throw 'Phase2B6 prerequisite evidence preflight failed.' }

Write-Host '[V1+ Phase2B6] validating authoritative H2/3H source/model contract...'
& $Python -c "import sys; from spincore.r7_5_representation_v3 import H2_FINAL; from spincore.r7_5_representation_v3_stage_contract import validate_phase2_v3_contract; [validate_phase2_v3_contract(sys.argv[1],representation=H2_FINAL,domain='THREE_HANDED',training_seed=s) for s in (1342191342,1801739323)]; print('H2/3H frozen source contract PASS')" $Repo
if ($LASTEXITCODE -ne 0) { throw 'Phase2B6 source/model contract preflight failed.' }

Write-Host '[V1+ Phase2B6] validating exact completed Phase2A S100K baseline policy artifacts...'
& $Python -c "import sys; from pathlib import Path; import r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot as p; x=p._validate_phase2a_baseline(Path(sys.argv[1]),Path(sys.argv[2])); print('Phase2A exact S100K baseline policies PASS',len(x['policy_artifacts']))" $Phase2ARoot $Phase2AResult
if ($LASTEXITCODE -ne 0) { throw 'Phase2B6 Phase2A baseline-policy identity preflight failed.' }

$Build = Join-Path $Repo 'build_phase2b6'
if (Test-Path $Build) { Remove-Item -Recurse -Force $Build }
& cmake -S $Repo -B $Build -G 'Visual Studio 17 2022' -A x64
if ($LASTEXITCODE -ne 0) { throw 'Phase2B6 CMake x64 configure failed.' }
& cmake --build $Build --config Release --target spincore_solver_c --parallel
if ($LASTEXITCODE -ne 0) { throw 'Phase2B6 solver x64 build failed.' }
$SolverCandidates = @(@((Join-Path $Build 'Release/spincore_solver_c.dll'),(Join-Path $Build 'spincore_solver_c.dll')) | Where-Object { Test-Path $_ })
if ($SolverCandidates.Count -ne 1) { throw "Expected exactly one x64 solver DLL; found: $($SolverCandidates -join ', ')" }
$Solver = (Resolve-Path $SolverCandidates[0]).Path
& $Python -c "import struct,sys; p=sys.argv[1]; b=open(p,'rb').read(); assert b[:2]==b'MZ'; pe=struct.unpack_from('<I',b,0x3c)[0]; assert b[pe:pe+4]==b'PE\0\0'; m=struct.unpack_from('<H',b,pe+4)[0]; print(f'solver PE machine 0x{m:04X}'); assert m==0x8664" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2B6 solver PE architecture preflight failed.' }
& $Python -c "import ctypes as C,sys; L=C.CDLL(sys.argv[1]); L.spincore_solver_c_abi_version.argtypes=[]; L.spincore_solver_c_abi_version.restype=C.c_int32; v=int(L.spincore_solver_c_abi_version()); print('solver ABI',v); assert v==2; assert getattr(L,'spincore_solver_state_neural_input_v3',None) is not None; print('SPNNIV3 solver API PASS')" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2B6 solver ABI/SPNNIV3 preflight failed.' }

$Output = Join-Path $Repo 'ryzen_v1plus_phase2b6'
$Result = Join-Path $Output 'R7_5_ARCH_RESET_V1PLUS_PHASE2B6_PREFLOP_DAMPING_TRAINING_PILOT.json'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$SeedWorkers = 2
Write-Host "[V1+ Phase2B6] pilot HEAD: $Head"
Write-Host "[V1+ Phase2B6] training workers: $SeedWorkers independent seed processes x 2 Torch/OMP/MKL threads"
Write-Host '[V1+ Phase2B6] exact training: H2/3H only, 2 seeds x 3 iterations x 4 chunks x 64 roots = 1536 roots total.'
Write-Host '[V1+ Phase2B6] intervention: 25% uniform floor ONLY after the first voluntary preflop action; root and postflop remain native.'
Write-Host '[V1+ Phase2B6] heldout inference: floor=0.00. The learned AveragePolicy is evaluated without masking/flattening.'
Write-Host '[V1+ Phase2B6] existing Phase2A artifacts are read-only controls and will not be retrained or modified.'

& $Python (Join-Path $Repo 'tools/spincore_ryzen_frozen_runner.py') `
    --expected-commit $Head `
    --run-name 'r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B6_PREFLOP_DAMPING_TRAINING_PILOT_PRECOMMIT_20260822.md' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B5_RESULT_EVIDENCE_20260822.json' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2A_RESULT_EVIDENCE_20260822.json' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py' `
    --contract 'tools/test_r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py' `
    --contract 'tools/r7_5_3d_v1plus_phase2a_strategy_capacity.py' `
    --contract 'tools/r7_5_3c_chance_coverage_x4_domain_worker_runtimefix.py' `
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
        --phase2a-root $Phase2ARoot `
        --phase2a-result $Phase2AResult `
        --phase2b5-result $Phase2B5 `
        --output-root $Output `
        --execution-sha $Head `
        --seed-workers $SeedWorkers

if ($LASTEXITCODE -ne 0) { throw "Phase2B6 failed with exit code $LASTEXITCODE. Preserve ryzen_v1plus_phase2b6 and all prior artifacts; do not delete or restart from scratch." }
if (-not (Test-Path $Result -PathType Leaf)) { throw 'Phase2B6 returned success without result JSON.' }
$Hash = (Get-FileHash -Algorithm SHA256 $Result).Hash.ToLowerInvariant()
Write-Host ''
Write-Host '[V1+ Phase2B6] COMPLETE'
Write-Host "[V1+ Phase2B6] result: $Result"
Write-Host "[V1+ Phase2B6] SHA256: $Hash"
Write-Host '[V1+ Phase2B6] This was a small causal training pilot, not production training. Send the result JSON back for the next stability/strength decision.'
