$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path '.').Path
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$Dirty = (& git -C $Repo status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) {
    throw "Tracked worktree is dirty. Untracked files are intentionally ignored. Stop and send this output before Phase2C2:`n$Dirty"
}

$Venv = Join-Path $Repo '.venv-r7_5_3c_x16'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python -PathType Leaf)) { throw "Missing frozen Phase2 Python environment: $Python" }
& $Python -c "import struct,sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9),sys.version; assert struct.calcsize('P')==8; assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print('python',sys.version.split()[0],'bits',struct.calcsize('P')*8,'torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Phase2C2 frozen runtime verification failed.' }

$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

$Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2c2_range_reach_target_kernel_causal_pilot.py'
$Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2c2_range_reach_target_kernel_causal_pilot.py'
$LiveTest = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2c2_live_replacement.py'
$B10Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition.py'
$B13Root = Join-Path $Repo 'ryzen_v1plus_phase2b13'
$B13Result = Join-Path $B13Root 'R7_5_ARCH_RESET_V1PLUS_PHASE2B13_ROOT_IID64_TARGET_TRAINING.json'
$B14Root = Join-Path $Repo 'ryzen_v1plus_phase2b14'
$B14Result = Join-Path $B14Root 'R7_5_ARCH_RESET_V1PLUS_PHASE2B14_B13_RESIDUAL_LOCALIZATION.json'
$C1Root = Join-Path $Repo 'ryzen_v1plus_phase2c1'
$C1Result = Join-Path $C1Root 'R7_5_ARCH_RESET_V1PLUS_PHASE2C1_EXACT_RANGE_REACH_SOLVER_PROTOTYPE.json'
$Heldout = Join-Path $Repo 'heldout_v3_bundle'

Write-Host '[V1+ Phase2C2] compiling structural causal-pilot scripts...'
& $Python -m py_compile $Tool $Test $LiveTest
if ($LASTEXITCODE -ne 0) { throw 'Phase2C2 py_compile failed.' }

Write-Host '[V1+ Phase2C2] running deterministic synthetic tests...'
& $Python $Test
if ($LASTEXITCODE -ne 0) { throw 'Phase2C2 synthetic tests failed.' }

foreach ($Path in @($B13Result,$B14Result,$C1Result)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing exact Phase2C2 prerequisite: $Path" }
}
if (-not (Test-Path $Heldout -PathType Container)) { throw "Missing frozen heldout bundle: $Heldout" }

Write-Host '[V1+ Phase2C2] validating exact Phase2C1/B13/B14 identities and route...'
& $Python -c "import hashlib,json,sys; c1,b13,b14=sys.argv[1:4]; raw=open(c1,'rb').read(); h=hashlib.sha256(raw).hexdigest(); j=json.loads(raw); assert h=='62ad2352c807a3b046bc84df2cbdf66cc8e0217e3422d01f2bcd9ddeafe7875b',h; assert j['status']=='EXACT_RANGE_REACH_TRANSITION_PROTOTYPE_FEASIBLE'; assert j['decision']['screen_pass'] is True; assert j['decision']['next_route']=='PRECOMMIT_SINGLE_BOUNDED_RANGE_REACH_TARGET_KERNEL_CAUSAL_PILOT'; hb13=hashlib.sha256(open(b13,'rb').read()).hexdigest(); hb14=hashlib.sha256(open(b14,'rb').read()).hexdigest(); assert hb13=='6de7996282236d34adf5e8e53416fd8a443a1fbf5abc89fc807492d0cb3dbf80',hb13; assert hb14=='7cd1886596d345abdcdef479775498eddf7e014205de86e44afb5bb0ea291f86',hb14; print('Phase2C1/B13/B14 exact prerequisite evidence PASS',h,hb13,hb14)" $C1Result $B13Result $B14Result
if ($LASTEXITCODE -ne 0) { throw 'Phase2C2 prerequisite evidence preflight failed.' }

Write-Host '[V1+ Phase2C2] validating exact final Phase2B13 bootstrap behavior checkpoints...'
& $Python -c "import hashlib,sys; from pathlib import Path; root=Path(sys.argv[1]); rows=[(1342191342,'f853dedf68a7f94b203d1f1de8b650897b2ddd69338fde56918d685f987b7cd3'),(1801739323,'ef25629fa8e99a024bd3c7f27d6a6140734e5efe3e5d245b390d3028a0e2cca3')]; out=[]; [(lambda p,s,e: ( (_ for _ in ()).throw(AssertionError((s,hashlib.sha256(p.read_bytes()).hexdigest(),e))) if hashlib.sha256(p.read_bytes()).hexdigest()!=e else out.append((s,e)) ))(root/'IID64_MEAN_CANDIDATE'/f'seed_{s}'/'resume_checkpoint.pt',s,e) for s,e in rows]; print('Phase2B13 final candidate behavior checkpoints PASS',out)" $B13Root
if ($LASTEXITCODE -ne 0) { throw 'Phase2C2 B13 checkpoint preflight failed.' }

$Build = Join-Path $Repo 'build_phase2c2'
if (Test-Path $Build) { Remove-Item -Recurse -Force $Build }
& cmake -S $Repo -B $Build -G 'Visual Studio 17 2022' -A x64
if ($LASTEXITCODE -ne 0) { throw 'Phase2C2 CMake x64 configure failed.' }
& cmake --build $Build --config Release --target spincore_solver_c --parallel
if ($LASTEXITCODE -ne 0) { throw 'Phase2C2 solver x64 build failed.' }
$SolverCandidates = @(@((Join-Path $Build 'Release/spincore_solver_c.dll'),(Join-Path $Build 'spincore_solver_c.dll')) | Where-Object { Test-Path $_ })
if ($SolverCandidates.Count -ne 1) { throw "Expected exactly one x64 solver DLL; found: $($SolverCandidates -join ', ')" }
$Solver = (Resolve-Path $SolverCandidates[0]).Path
& $Python -c "import struct,sys; p=sys.argv[1]; b=open(p,'rb').read(); assert b[:2]==b'MZ'; pe=struct.unpack_from('<I',b,0x3c)[0]; assert b[pe:pe+4]==b'PE\0\0'; m=struct.unpack_from('<H',b,pe+4)[0]; print(f'solver PE machine 0x{m:04X}'); assert m==0x8664" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2C2 solver PE architecture preflight failed.' }
& $Python -c "import ctypes as C,sys; L=C.CDLL(sys.argv[1]); L.spincore_solver_c_abi_version.argtypes=[]; L.spincore_solver_c_abi_version.restype=C.c_int32; v=int(L.spincore_solver_c_abi_version()); print('solver ABI',v); assert v==2; assert getattr(L,'spincore_solver_state_neural_input_v3',None) is not None; assert getattr(L,'spincore_solver_state_create_v2_deal',None) is not None; assert getattr(L,'spincore_solver_state_deal_snapshot_v1',None) is not None; print('SPNNIV3 + explicit-deal API PASS')" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2C2 solver ABI preflight failed.' }

Write-Host '[V1+ Phase2C2] rerunning explicit-deal round-trip tests...'
& $Python $B10Test --repo-root $Repo --solver $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2C2 explicit-deal round-trip tests failed.' }

Write-Host '[V1+ Phase2C2] proving a positive-support depth-2 preflop continuation exists across all frozen scenarios/seeds...'
& $Python $Tool --path-preflight-only --repo-root $Repo --solver $Solver --phase2b13-root $B13Root
if ($LASTEXITCODE -ne 0) { throw 'Phase2C2 depth-2 continuation path preflight failed.' }

Write-Host '[V1+ Phase2C2] running one real K64 root + K64 structural continuation kernel preflight...'
& $Python $Tool --kernel-preflight-only --repo-root $Repo --solver $Solver --phase2b13-root $B13Root
if ($LASTEXITCODE -ne 0) { throw 'Phase2C2 structural kernel preflight failed.' }

Write-Host '[V1+ Phase2C2] validating exact root+continuation replacement in live CFR collection for both source seeds...'
& $Python $LiveTest --repo-root $Repo --solver $Solver --phase2b13-root $B13Root
if ($LASTEXITCODE -ne 0) { throw 'Phase2C2 live replacement preflight failed.' }

$Output = Join-Path $Repo 'ryzen_v1plus_phase2c2'
$Result = Join-Path $Output 'R7_5_ARCH_RESET_V1PLUS_PHASE2C2_RANGE_REACH_TARGET_KERNEL_CAUSAL_PILOT.json'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$ArmWorkers = 2
$ChanceWorkers = 14

Write-Host "[V1+ Phase2C2] screen HEAD: $Head"
Write-Host '[V1+ Phase2C2] SINGLE bounded structural causal pilot before x4-or-V1 fallback decision.'
Write-Host '[V1+ Phase2C2] both arms pay identical K64 root + K64 continuation auxiliary target compute.'
Write-Host '[V1+ Phase2C2] 2 arms x 2 seeds x 2 iterations x 64 logical roots = 512 logical roots; total auxiliary target traversals = 65536.'
Write-Host "[V1+ Phase2C2] parallelism: $ArmWorkers arm/seed trajectories x up to $ChanceWorkers one-thread chance workers"
Write-Host '[V1+ Phase2C2] no production training, no architecture winner, no ready-for-tables claim.'

& $Python (Join-Path $Repo 'tools/spincore_ryzen_frozen_runner.py') `
    --expected-commit $Head `
    --run-name 'r7_5_arch_reset_v1plus_phase2c2_range_reach_target_kernel_causal_pilot' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2C2_RANGE_REACH_TARGET_KERNEL_CAUSAL_PILOT_PRECOMMIT_20260824.md' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2C1_RESULT_EVIDENCE_20260824.json' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2c2_range_reach_target_kernel_causal_pilot.py' `
    --contract 'tools/test_r7_5_arch_reset_v1plus_phase2c2_range_reach_target_kernel_causal_pilot.py' `
    --contract 'tools/test_r7_5_arch_reset_v1plus_phase2c2_live_replacement.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2c1_exact_range_reach_solver_prototype.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance.py' `
    --artifact $Output `
    -- $Python $Tool `
        --repo-root $Repo `
        --solver $Solver `
        --heldout-root $Heldout `
        --phase2b13-root $B13Root `
        --phase2b13-result $B13Result `
        --phase2b14-result $B14Result `
        --phase2c1-result $C1Result `
        --output-root $Output `
        --execution-sha $Head `
        --arm-workers $ArmWorkers `
        --chance-workers $ChanceWorkers

if ($LASTEXITCODE -ne 0) { throw "Phase2C2 failed with exit code $LASTEXITCODE. Preserve ryzen_v1plus_phase2c2 and send exact output; do not modify thresholds, source, or partials." }
if (-not (Test-Path $Result -PathType Leaf)) { throw 'Phase2C2 returned success without result JSON.' }
$Hash = (Get-FileHash -Algorithm SHA256 $Result).Hash.ToLowerInvariant()
Write-Host ''
Write-Host '[V1+ Phase2C2] COMPLETE'
Write-Host "[V1+ Phase2C2] result: $Result"
Write-Host "[V1+ Phase2C2] SHA256: $Hash"
Write-Host '[V1+ Phase2C2] Send the JSON back. PASS permits one full x4 confirmation; causal FAIL selects certified stable V1 fallback.'
