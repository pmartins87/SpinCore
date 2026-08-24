$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path '.').Path
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$Dirty = (& git -C $Repo status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) {
    throw "Tracked worktree is dirty. Untracked files are intentionally ignored. Stop and send this output before Phase2C1:`n$Dirty"
}

$Venv = Join-Path $Repo '.venv-r7_5_3c_x16'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python -PathType Leaf)) { throw "Missing frozen Phase2 Python environment: $Python" }
& $Python -c "import struct,sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9),sys.version; assert struct.calcsize('P')==8; assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print('python',sys.version.split()[0],'bits',struct.calcsize('P')*8,'torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Phase2C1 frozen runtime verification failed.' }

$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

$Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2c1_exact_range_reach_solver_prototype.py'
$Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2c1_exact_range_reach_solver_prototype.py'
$C0Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2c0_structural_reach_factorization.py'
$B10Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition.py'
$B15Fix = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance_runtimefix.py'

Write-Host '[V1+ Phase2C1] compiling exact range/reach prototype scripts...'
& $Python -m py_compile $Tool $Test $C0Tool $B15Fix
if ($LASTEXITCODE -ne 0) { throw 'Phase2C1 py_compile failed.' }

Write-Host '[V1+ Phase2C1] running deterministic synthetic tests...'
& $Python $Test
if ($LASTEXITCODE -ne 0) { throw 'Phase2C1 synthetic tests failed.' }

$B13Root = Join-Path $Repo 'ryzen_v1plus_phase2b13'
$B13Result = Join-Path $B13Root 'R7_5_ARCH_RESET_V1PLUS_PHASE2B13_ROOT_IID64_TARGET_TRAINING.json'
$B14Root = Join-Path $Repo 'ryzen_v1plus_phase2b14'
$B14Result = Join-Path $B14Root 'R7_5_ARCH_RESET_V1PLUS_PHASE2B14_B13_RESIDUAL_LOCALIZATION.json'
$C0Root = Join-Path $Repo 'ryzen_v1plus_phase2c0'
$C0Result = Join-Path $C0Root 'R7_5_ARCH_RESET_V1PLUS_PHASE2C0_STRUCTURAL_REACH_FACTORIZATION.json'
$Heldout = Join-Path $Repo 'heldout_v3_bundle'
foreach ($Path in @($B13Result,$B14Result,$C0Result)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing exact Phase2C1 prerequisite: $Path" }
}
if (-not (Test-Path $Heldout -PathType Container)) { throw "Missing frozen heldout bundle: $Heldout" }

Write-Host '[V1+ Phase2C1] validating exact Phase2C0 PASS result identity...'
& $Python -c "import hashlib,json,sys; p=sys.argv[1]; raw=open(p,'rb').read(); h=hashlib.sha256(raw).hexdigest(); j=json.loads(raw); assert h=='55e83be4fd8776e0fcdb63e7d4400ed05aff8c48213898ad8f1abe3713a35876',h; assert j['schema']=='SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2C0_STRUCTURAL_REACH_FACTORIZATION_V1'; assert j['status']=='STRUCTURAL_REACH_FACTORIZATION_FEASIBLE'; assert j['decision']['screen_pass'] is True; assert j['decision']['next_route']=='PRECOMMIT_PHASE2C1_EXACT_RANGE_REACH_SOLVER_PROTOTYPE'; print('Phase2C0 exact PASS result',h)" $C0Result
if ($LASTEXITCODE -ne 0) { throw 'Phase2C1 Phase2C0 result preflight failed.' }

$Build = Join-Path $Repo 'build_phase2c1'
if (Test-Path $Build) { Remove-Item -Recurse -Force $Build }
& cmake -S $Repo -B $Build -G 'Visual Studio 17 2022' -A x64
if ($LASTEXITCODE -ne 0) { throw 'Phase2C1 CMake x64 configure failed.' }
& cmake --build $Build --config Release --target spincore_solver_c --parallel
if ($LASTEXITCODE -ne 0) { throw 'Phase2C1 solver x64 build failed.' }
$SolverCandidates = @(@((Join-Path $Build 'Release/spincore_solver_c.dll'),(Join-Path $Build 'spincore_solver_c.dll')) | Where-Object { Test-Path $_ })
if ($SolverCandidates.Count -ne 1) { throw "Expected exactly one x64 solver DLL; found: $($SolverCandidates -join ', ')" }
$Solver = (Resolve-Path $SolverCandidates[0]).Path
& $Python -c "import struct,sys; p=sys.argv[1]; b=open(p,'rb').read(); assert b[:2]==b'MZ'; pe=struct.unpack_from('<I',b,0x3c)[0]; assert b[pe:pe+4]==b'PE\0\0'; m=struct.unpack_from('<H',b,pe+4)[0]; print(f'solver PE machine 0x{m:04X}'); assert m==0x8664" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2C1 solver PE architecture preflight failed.' }
& $Python -c "import ctypes as C,sys; L=C.CDLL(sys.argv[1]); L.spincore_solver_c_abi_version.argtypes=[]; L.spincore_solver_c_abi_version.restype=C.c_int32; v=int(L.spincore_solver_c_abi_version()); print('solver ABI',v); assert v==2; assert getattr(L,'spincore_solver_state_neural_input_v3',None) is not None; assert getattr(L,'spincore_solver_state_create_v2_deal',None) is not None; assert getattr(L,'spincore_solver_state_deal_snapshot_v1',None) is not None; print('SPNNIV3 + explicit-deal API PASS')" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2C1 solver ABI preflight failed.' }

Write-Host '[V1+ Phase2C1] rerunning explicit-deal round-trip tests...'
& $Python $B10Test --repo-root $Repo --solver $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2C1 explicit-deal round-trip tests failed.' }

Write-Host '[V1+ Phase2C1] validating all 64 Windows heldout reconstructions before structural prototype...'
& $Python $B15Fix --preflight-only --repo-root $Repo --solver $Solver --heldout-root $Heldout --phase2b14-result $B14Result
if ($LASTEXITCODE -ne 0) { throw 'Phase2C1 Windows heldout reconstruction preflight failed.' }

$Output = Join-Path $Repo 'ryzen_v1plus_phase2c1'
$Result = Join-Path $Output 'R7_5_ARCH_RESET_V1PLUS_PHASE2C1_EXACT_RANGE_REACH_SOLVER_PROTOTYPE.json'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Workers = 16

Write-Host "[V1+ Phase2C1] prototype HEAD: $Head"
Write-Host '[V1+ Phase2C1] STRUCTURAL range/reach state propagation; no target estimator and no training.'
Write-Host '[V1+ Phase2C1] 8 exact Phase2C0 anchors x 2 behavior seeds; two 2450-entry opponent reach vectors per task.'
Write-Host '[V1+ Phase2C1] PASS permits only one separately precommitted bounded structural causal pilot; FAIL selects certified stable V1 fallback.'
Write-Host "[V1+ Phase2C1] workers: up to $Workers one-thread processes"

& $Python (Join-Path $Repo 'tools/spincore_ryzen_frozen_runner.py') `
    --expected-commit $Head `
    --run-name 'r7_5_arch_reset_v1plus_phase2c1_exact_range_reach_solver_prototype' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2C1_EXACT_RANGE_REACH_SOLVER_PROTOTYPE_PRECOMMIT_20260824.md' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2C0_RESULT_EVIDENCE_20260824.json' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2c1_exact_range_reach_solver_prototype.py' `
    --contract 'tools/test_r7_5_arch_reset_v1plus_phase2c1_exact_range_reach_solver_prototype.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2c0_structural_reach_factorization.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance_runtimefix.py' `
    --contract 'python/spincore/r7_5_representation_v3.py' `
    --contract 'python/spincore/r7_5_action_cfr.py' `
    --contract 'python/spincore/solver.py' `
    --artifact $Output `
    -- $Python $Tool `
        --repo-root $Repo `
        --solver $Solver `
        --heldout-root $Heldout `
        --phase2b13-root $B13Root `
        --phase2b13-result $B13Result `
        --phase2b14-result $B14Result `
        --phase2c0-result $C0Result `
        --output-root $Output `
        --execution-sha $Head `
        --workers $Workers

if ($LASTEXITCODE -ne 0) { throw "Phase2C1 failed with exit code $LASTEXITCODE. Preserve ryzen_v1plus_phase2c1 and send exact output; do not modify thresholds or source." }
if (-not (Test-Path $Result -PathType Leaf)) { throw 'Phase2C1 returned success without result JSON.' }
$Hash = (Get-FileHash -Algorithm SHA256 $Result).Hash.ToLowerInvariant()
Write-Host ''
Write-Host '[V1+ Phase2C1] COMPLETE'
Write-Host "[V1+ Phase2C1] result: $Result"
Write-Host "[V1+ Phase2C1] SHA256: $Hash"
Write-Host '[V1+ Phase2C1] Send the JSON back. PASS permits one bounded structural causal pilot; FAIL selects certified stable V1 fallback.'
