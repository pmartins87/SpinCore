$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path '.').Path
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$Dirty = (& git -C $Repo status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) {
    throw "Tracked worktree is dirty. Untracked files are intentionally ignored. Stop and send this output before Phase2B11:`n$Dirty"
}

$Precommit = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B11_FACTORIZED_CHANCE_ESTIMATOR_PRECOMMIT_20260824.md'
$B1Evidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B1_RESULT_EVIDENCE_20260822.json'
$B6Evidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B6_RESULT_EVIDENCE_20260823.json'
$B10Evidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B10_RESULT_EVIDENCE_20260824.json'
foreach ($Path in @($Precommit, $B1Evidence, $B6Evidence, $B10Evidence)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing frozen Phase2B11 contract/evidence: $Path" }
}

$Venv = Join-Path $Repo '.venv-r7_5_3c_x16'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python -PathType Leaf)) { throw "Missing existing frozen Phase2 Python environment: $Python" }
& $Python -c "import struct,sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9),sys.version; assert struct.calcsize('P')==8; assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print('python',sys.version.split()[0],'bits',struct.calcsize('P')*8,'torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Frozen Phase2B11 Python/Torch/Numpy runtime verification failed.' }

$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

$Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator.py'
$Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator.py'
$B10Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition.py'
$B10Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition.py'
$B1Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b1_target_variance.py'
$B6Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py'
Write-Host '[V1+ Phase2B11] compiling factorized-estimator scripts...'
& $Python -m py_compile $Tool $Test $B10Tool $B10Test $B1Tool $B6Tool (Join-Path $Repo 'python/spincore/solver.py')
if ($LASTEXITCODE -ne 0) { throw 'Phase2B11 py_compile failed.' }
Write-Host '[V1+ Phase2B11] running deterministic pure-Python tests...'
& $Python $Test
if ($LASTEXITCODE -ne 0) { throw 'Phase2B11 pure synthetic tests failed.' }

$B1Result = Join-Path $Repo 'ryzen_v1plus_phase2b1/R7_5_ARCH_RESET_V1PLUS_PHASE2B1_TARGET_VARIANCE.json'
$B6Root = Join-Path $Repo 'ryzen_v1plus_phase2b6'
$B6Result = Join-Path $B6Root 'R7_5_ARCH_RESET_V1PLUS_PHASE2B6_PREFLOP_DAMPING_TRAINING_PILOT.json'
$B10Result = Join-Path $Repo 'ryzen_v1plus_phase2b10/R7_5_ARCH_RESET_V1PLUS_PHASE2B10_PRIVATE_PUBLIC_CHANCE_DECOMPOSITION.json'
foreach ($Path in @($B1Result, $B6Result, $B10Result)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing exact completed Phase2B11 prerequisite result: $Path" }
}
Write-Host '[V1+ Phase2B11] validating exact Phase2B1/Phase2B6/Phase2B10 local identities...'
& $Python -c "import hashlib,json,sys; p1,p6,p10,e1,e6,e10=sys.argv[1:7]; expected=('f95751afeb17fcd5844bfcb2971577b92a400750444e5dabe2f4ddb5718ba6ef','33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a','0295574c6133eb05866ecbdccf7e31efa4e6e8936dbd8bb7e375e166b27fe4dc'); rows=[];`nfor p,e,x in zip((p1,p6,p10),(e1,e6,e10),expected):`n raw=open(p,'rb').read(); h=hashlib.sha256(raw).hexdigest(); ev=json.load(open(e,encoding='utf-8')); assert h==x==ev['uploaded_result_sha256']; rows.append(h);`nj1=json.load(open(p1,encoding='utf-8')); j6=json.load(open(p6,encoding='utf-8')); j10=json.load(open(p10,encoding='utf-8')); assert j1['decision']['source_classification']=='CHANCE_DOMINANT'; assert j6['status']=='PREFLOP_DAMPING_CAUSAL_EFFECT_SUPPORTED_BUT_STILL_UNSTABLE'; assert j10['status']=='MIXED_PRIVATE_PUBLIC_CHANCE'; assert j10['decision']['next_route']=='PRECOMMIT_FACTORIZED_PRIVATE_PUBLIC_CHANCE_VARIANCE_REDUCTION_DIAGNOSTIC'; print('Phase2B1/B6/B10 exact prerequisite evidence PASS',rows)" $B1Result $B6Result $B10Result $B1Evidence $B6Evidence $B10Evidence
if ($LASTEXITCODE -ne 0) { throw 'Phase2B11 prerequisite evidence preflight failed.' }

Write-Host '[V1+ Phase2B11] validating authoritative H2/3H model contract...'
& $Python -c "import sys; from spincore.r7_5_representation_v3 import H2_FINAL; from spincore.r7_5_representation_v3_stage_contract import validate_phase2_v3_contract; [validate_phase2_v3_contract(sys.argv[1],representation=H2_FINAL,domain='THREE_HANDED',training_seed=s) for s in (1342191342,1801739323)]; print('H2/3H frozen source contract PASS')" $Repo
if ($LASTEXITCODE -ne 0) { throw 'Phase2B11 source/model contract preflight failed.' }

$Build = Join-Path $Repo 'build_phase2b11'
if (Test-Path $Build) { Remove-Item -Recurse -Force $Build }
& cmake -S $Repo -B $Build -G 'Visual Studio 17 2022' -A x64
if ($LASTEXITCODE -ne 0) { throw 'Phase2B11 CMake x64 configure failed.' }
& cmake --build $Build --config Release --target spincore_solver_c --parallel
if ($LASTEXITCODE -ne 0) { throw 'Phase2B11 solver x64 build failed.' }
$SolverCandidates = @(@((Join-Path $Build 'Release/spincore_solver_c.dll'),(Join-Path $Build 'spincore_solver_c.dll')) | Where-Object { Test-Path $_ })
if ($SolverCandidates.Count -ne 1) { throw "Expected exactly one x64 solver DLL; found: $($SolverCandidates -join ', ')" }
$Solver = (Resolve-Path $SolverCandidates[0]).Path
& $Python -c "import struct,sys; p=sys.argv[1]; b=open(p,'rb').read(); assert b[:2]==b'MZ'; pe=struct.unpack_from('<I',b,0x3c)[0]; assert b[pe:pe+4]==b'PE\0\0'; m=struct.unpack_from('<H',b,pe+4)[0]; print(f'solver PE machine 0x{m:04X}'); assert m==0x8664" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2B11 solver PE architecture preflight failed.' }
& $Python -c "import ctypes as C,sys; L=C.CDLL(sys.argv[1]); L.spincore_solver_c_abi_version.argtypes=[]; L.spincore_solver_c_abi_version.restype=C.c_int32; v=int(L.spincore_solver_c_abi_version()); print('solver ABI',v); assert v==2; assert getattr(L,'spincore_solver_state_neural_input_v3',None) is not None; assert getattr(L,'spincore_solver_state_create_v2_deal',None) is not None; assert getattr(L,'spincore_solver_state_deal_snapshot_v1',None) is not None; print('SPNNIV3 + explicit-deal diagnostic solver API PASS')" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2B11 solver ABI/explicit-deal preflight failed.' }

Write-Host '[V1+ Phase2B11] rerunning explicit-deal solver round-trip tests...'
& $Python $B10Test --repo-root $Repo --solver $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2B11 explicit-deal solver round-trip tests failed.' }

Write-Host '[V1+ Phase2B11] validating exact completed Phase2B6 behavior checkpoints...'
& $Python -c "import sys; from pathlib import Path; import r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition as p; root=Path(sys.argv[1]); rows=[];`nfor s in (1342191342,1801739323):`n cp=root/f'seed_{s}'/'resume_checkpoint.pt'; states=p._load_b6_behavior_states(cp,s); rows.append((s,p._sha256(cp),len(states)));`nprint('Phase2B6 final behavior checkpoint identity PASS',rows)" $B6Root
if ($LASTEXITCODE -ne 0) { throw 'Phase2B11 Phase2B6 behavior checkpoint preflight failed.' }

$Output = Join-Path $Repo 'ryzen_v1plus_phase2b11'
$Result = Join-Path $Output 'R7_5_ARCH_RESET_V1PLUS_PHASE2B11_FACTORIZED_CHANCE_ESTIMATOR.json'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Workers = 30
Write-Host "[V1+ Phase2B11] diagnostic HEAD: $Head"
Write-Host "[V1+ Phase2B11] workers: $Workers independent processes x 1 Torch/OMP/MKL/OpenBLAS thread"
Write-Host '[V1+ Phase2B11] exact work: 2 behaviors x 15 scenarios x 4 anchors x 4 blocks x (IID4 4 + FACTOR2X2 4 + IID16 16 + FACTOR4X4 16) = 19200 root target traversals.'
Write-Host '[V1+ Phase2B11] equal-compute comparisons: FACTOR2X2 vs IID4 and primary FACTOR4X4 vs IID16.'
Write-Host '[V1+ Phase2B11] READ-ONLY learned state; no fit, optimizer, reservoir insertion, AveragePolicy fit, or checkpoint mutation.'

& $Python (Join-Path $Repo 'tools/spincore_ryzen_frozen_runner.py') `
    --expected-commit $Head `
    --run-name 'r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B11_FACTORIZED_CHANCE_ESTIMATOR_PRECOMMIT_20260824.md' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B10_RESULT_EVIDENCE_20260824.json' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B6_RESULT_EVIDENCE_20260823.json' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B1_RESULT_EVIDENCE_20260822.json' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator.py' `
    --contract 'tools/test_r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b1_target_variance.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py' `
    --contract 'python/spincore/solver.py' `
    --contract 'include/spincore/solver_c_api.h' `
    --contract 'src/solver_c_api.cpp' `
    --contract 'include/spincore/hand_engine.hpp' `
    --contract 'src/hand_engine.cpp' `
    --contract 'include/spincore/spin_traversal_state.hpp' `
    --contract 'src/spin_traversal_state.cpp' `
    --artifact $Output `
    -- $Python $Tool `
        --repo-root $Repo `
        --solver $Solver `
        --phase2b1-result $B1Result `
        --phase2b6-root $B6Root `
        --phase2b6-result $B6Result `
        --phase2b10-result $B10Result `
        --workers $Workers `
        --out $Result

if ($LASTEXITCODE -ne 0) { throw "Phase2B11 failed with exit code $LASTEXITCODE. Preserve ryzen_v1plus_phase2b11 and all prior artifacts; do not start training." }
if (-not (Test-Path $Result -PathType Leaf)) { throw 'Phase2B11 returned success without result JSON.' }
$Hash = (Get-FileHash -Algorithm SHA256 $Result).Hash.ToLowerInvariant()
Write-Host ''
Write-Host '[V1+ Phase2B11] COMPLETE'
Write-Host "[V1+ Phase2B11] result: $Result"
Write-Host "[V1+ Phase2B11] SHA256: $Hash"
Write-Host '[V1+ Phase2B11] No model training was performed. Send the result JSON back for the next chance-estimator decision.'
