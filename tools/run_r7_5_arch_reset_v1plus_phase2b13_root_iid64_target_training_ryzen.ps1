$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path '.').Path
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$Dirty = (& git -C $Repo status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) {
    throw "Tracked worktree is dirty. Untracked files are intentionally ignored. Stop and send this output before Phase2B13:`n$Dirty"
}

$Precommit = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B13_ROOT_IID64_TARGET_TRAINING_PRECOMMIT_20260824.md'
$B6Evidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B6_RESULT_EVIDENCE_20260823.json'
$B12Evidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B12_RESULT_EVIDENCE_20260824.json'
foreach ($Path in @($Precommit, $B6Evidence, $B12Evidence)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing frozen Phase2B13 contract/evidence: $Path" }
}

$Venv = Join-Path $Repo '.venv-r7_5_3c_x16'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python -PathType Leaf)) { throw "Missing existing frozen Phase2 Python environment: $Python" }
& $Python -c "import struct,sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9),sys.version; assert struct.calcsize('P')==8; assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print('python',sys.version.split()[0],'bits',struct.calcsize('P')*8,'torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Frozen Phase2B13 Python/Torch/Numpy runtime verification failed.' }

$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '2'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

$Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training.py'
$Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training.py'
$B10Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition.py'
$B10Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition.py'
$B11Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator.py'
$B6Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py'
$B1Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b1_target_variance.py'
$Phase2ABase = Join-Path $Repo 'tools/r7_5_3d_v1plus_phase2a_strategy_capacity.py'
$X4RuntimeFix = Join-Path $Repo 'tools/r7_5_3c_chance_coverage_x4_domain_worker_runtimefix.py'

Write-Host '[V1+ Phase2B13] compiling pilot scripts...'
& $Python -m py_compile $Tool $Test $B10Tool $B10Test $B11Tool $B6Tool $B1Tool $Phase2ABase $X4RuntimeFix (Join-Path $Repo 'python/spincore/solver.py')
if ($LASTEXITCODE -ne 0) { throw 'Phase2B13 py_compile failed.' }
Write-Host '[V1+ Phase2B13] running deterministic synthetic tests...'
& $Python $Test
if ($LASTEXITCODE -ne 0) { throw 'Phase2B13 synthetic tests failed.' }

$B6Result = Join-Path $Repo 'ryzen_v1plus_phase2b6/R7_5_ARCH_RESET_V1PLUS_PHASE2B6_PREFLOP_DAMPING_TRAINING_PILOT.json'
$B12Result = Join-Path $Repo 'ryzen_v1plus_phase2b12/R7_5_ARCH_RESET_V1PLUS_PHASE2B12_IID_CHANCE_EXPECTATION_CONVERGENCE.json'
$Heldout = Join-Path $Repo 'heldout_v3_bundle'
foreach ($Path in @($B6Result, $B12Result)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing exact completed Phase2B13 prerequisite result: $Path" }
}
if (-not (Test-Path $Heldout -PathType Container)) { throw "Missing frozen heldout bundle: $Heldout" }

Write-Host '[V1+ Phase2B13] validating exact Phase2B6/Phase2B12 prerequisite identities...'
& $Python -c "import hashlib,json,sys; p6,p12,e6,e12=sys.argv[1:5]; expected=('33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a','dbccadae5805381d0188bef41fb62a72b25b42e03e5564ca88f05d9666e6e182'); rows=[];`nfor p,e,x in zip((p6,p12),(e6,e12),expected):`n raw=open(p,'rb').read(); h=hashlib.sha256(raw).hexdigest(); ev=json.load(open(e,encoding='utf-8')); assert h==x==ev['uploaded_result_sha256']; rows.append(h);`nj6=json.load(open(p6,encoding='utf-8')); j12=json.load(open(p12,encoding='utf-8')); assert j6['status']=='PREFLOP_DAMPING_CAUSAL_EFFECT_SUPPORTED_BUT_STILL_UNSTABLE'; assert j12['status']=='IID_CHANCE_EXPECTATION_CONVERGES_MATERIALLY'; assert j12['decision']['screen_pass'] is True; assert j12['decision']['small_causal_training_pilot_precommit_allowed'] is True; print('Phase2B6/B12 exact prerequisite evidence PASS',rows)" $B6Result $B12Result $B6Evidence $B12Evidence
if ($LASTEXITCODE -ne 0) { throw 'Phase2B13 prerequisite evidence preflight failed.' }

Write-Host '[V1+ Phase2B13] validating authoritative H2/3H source/model contract...'
& $Python -c "import sys; from spincore.r7_5_representation_v3 import H2_FINAL; from spincore.r7_5_representation_v3_stage_contract import validate_phase2_v3_contract; [validate_phase2_v3_contract(sys.argv[1],representation=H2_FINAL,domain='THREE_HANDED',training_seed=s) for s in (1342191342,1801739323)]; print('H2/3H frozen source contract PASS')" $Repo
if ($LASTEXITCODE -ne 0) { throw 'Phase2B13 source/model contract preflight failed.' }

$Build = Join-Path $Repo 'build_phase2b13'
if (Test-Path $Build) { Remove-Item -Recurse -Force $Build }
& cmake -S $Repo -B $Build -G 'Visual Studio 17 2022' -A x64
if ($LASTEXITCODE -ne 0) { throw 'Phase2B13 CMake x64 configure failed.' }
& cmake --build $Build --config Release --target spincore_solver_c --parallel
if ($LASTEXITCODE -ne 0) { throw 'Phase2B13 solver x64 build failed.' }
$SolverCandidates = @(@((Join-Path $Build 'Release/spincore_solver_c.dll'),(Join-Path $Build 'spincore_solver_c.dll')) | Where-Object { Test-Path $_ })
if ($SolverCandidates.Count -ne 1) { throw "Expected exactly one x64 solver DLL; found: $($SolverCandidates -join ', ')" }
$Solver = (Resolve-Path $SolverCandidates[0]).Path
& $Python -c "import struct,sys; p=sys.argv[1]; b=open(p,'rb').read(); assert b[:2]==b'MZ'; pe=struct.unpack_from('<I',b,0x3c)[0]; assert b[pe:pe+4]==b'PE\0\0'; m=struct.unpack_from('<H',b,pe+4)[0]; print(f'solver PE machine 0x{m:04X}'); assert m==0x8664" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2B13 solver PE architecture preflight failed.' }
& $Python -c "import ctypes as C,sys; L=C.CDLL(sys.argv[1]); L.spincore_solver_c_abi_version.argtypes=[]; L.spincore_solver_c_abi_version.restype=C.c_int32; v=int(L.spincore_solver_c_abi_version()); print('solver ABI',v); assert v==2; assert getattr(L,'spincore_solver_state_neural_input_v3',None) is not None; assert getattr(L,'spincore_solver_state_create_v2_deal',None) is not None; assert getattr(L,'spincore_solver_state_deal_snapshot_v1',None) is not None; print('SPNNIV3 + explicit-deal diagnostic solver API PASS')" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2B13 solver ABI/explicit-deal preflight failed.' }

Write-Host '[V1+ Phase2B13] rerunning explicit-deal solver round-trip tests...'
& $Python $B10Test --repo-root $Repo --solver $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2B13 explicit-deal solver round-trip tests failed.' }

$Output = Join-Path $Repo 'ryzen_v1plus_phase2b13'
$Result = Join-Path $Output 'R7_5_ARCH_RESET_V1PLUS_PHASE2B13_ROOT_IID64_TARGET_TRAINING.json'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$ArmWorkers = 2
$ChanceWorkers = 14

Write-Host "[V1+ Phase2B13] pilot HEAD: $Head"
Write-Host "[V1+ Phase2B13] arm workers: $ArmWorkers concurrent arm/seed trajectories; each uses up to $ChanceWorkers one-thread chance workers during K64 target estimation"
Write-Host '[V1+ Phase2B13] logical training: 2 arms x 2 seeds x 3 iterations x 2 chunks x 64 roots = 1536 logical roots.'
Write-Host '[V1+ Phase2B13] equal-compute root targets: both arms compute 64 conditional-IID root targets per logical root; control inserts sample0, candidate inserts raw-target mean64.'
Write-Host '[V1+ Phase2B13] exact intervention boundary: only the initial preflop root Advantage sample is replaced in-place; downstream Advantage and all Strategy collection remain ordinary.'
Write-Host '[V1+ Phase2B13] Phase2B6 25% preflop-continuation behavior floor remains identical in both arms; heldout inference floor=0.00.'
Write-Host '[V1+ Phase2B13] small causal screen only; no architecture selection and no production training authorization.'

& $Python (Join-Path $Repo 'tools/spincore_ryzen_frozen_runner.py') `
    --expected-commit $Head `
    --run-name 'r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B13_ROOT_IID64_TARGET_TRAINING_PRECOMMIT_20260824.md' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B12_RESULT_EVIDENCE_20260824.json' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B6_RESULT_EVIDENCE_20260823.json' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training.py' `
    --contract 'tools/test_r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b1_target_variance.py' `
    --contract 'tools/r7_5_3d_v1plus_phase2a_strategy_capacity.py' `
    --contract 'tools/r7_5_3c_chance_coverage_x4_domain_worker_runtimefix.py' `
    --contract 'python/spincore/solver.py' `
    --contract 'python/spincore/r7_5_action_cfr.py' `
    --contract 'python/spincore/r7_5_representation_v3.py' `
    --contract 'python/spincore/r7_5_representation_v3_stage.py' `
    --contract 'python/spincore/r7_5_representation_v3_stage_contract.py' `
    --contract 'python/spincore/r7_5_representation_v3_uncertainty.py' `
    --contract 'include/spincore/solver_c_api.h' `
    --contract 'src/solver_c_api.cpp' `
    --contract 'include/spincore/hand_engine.hpp' `
    --contract 'src/hand_engine.cpp' `
    --artifact $Output `
    -- $Python $Tool `
        --repo-root $Repo `
        --solver $Solver `
        --heldout-root $Heldout `
        --phase2b6-result $B6Result `
        --phase2b12-result $B12Result `
        --output-root $Output `
        --execution-sha $Head `
        --arm-workers $ArmWorkers `
        --chance-workers $ChanceWorkers

if ($LASTEXITCODE -ne 0) { throw "Phase2B13 failed with exit code $LASTEXITCODE. Preserve ryzen_v1plus_phase2b13 and all prior artifacts; rerun the same launcher only after review." }
if (-not (Test-Path $Result -PathType Leaf)) { throw 'Phase2B13 returned success without result JSON.' }
$Hash = (Get-FileHash -Algorithm SHA256 $Result).Hash.ToLowerInvariant()
Write-Host ''
Write-Host '[V1+ Phase2B13] COMPLETE'
Write-Host "[V1+ Phase2B13] result: $Result"
Write-Host "[V1+ Phase2B13] SHA256: $Hash"
Write-Host '[V1+ Phase2B13] This was a small equal-compute causal target-training screen, not production training. Send the result JSON back for the next stability/strength decision.'
