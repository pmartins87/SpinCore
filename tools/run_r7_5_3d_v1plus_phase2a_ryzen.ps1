$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path '.').Path
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$Dirty = (& git -C $Repo status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) {
    throw "Tracked worktree is dirty. Commit/revert tracked changes before Phase2A.`n$Dirty"
}

$Precommit = Join-Path $Repo 'validation/R7_5_3D_V1PLUS_PHASE2A_STRATEGY_MEMORY_CAPACITY_ABLATION_PRECOMMIT_20260821.md'
$ImplementationFreeze = Join-Path $Repo 'validation/R7_5_3D_V1PLUS_PHASE2A_IMPLEMENTATION_FREEZE_20260821.md'
$RuntimeGuard = Join-Path $Repo 'validation/R7_5_3D_V1PLUS_PHASE2A_RUNTIME_GUARD_20260821.md'
$ParallelFitFreeze = Join-Path $Repo 'validation/R7_5_3D_V1PLUS_PHASE2A_PARALLEL_FIT_EXECUTION_FREEZE_20260821.md'
foreach ($Path in @($Precommit, $ImplementationFreeze, $RuntimeGuard, $ParallelFitFreeze)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing Phase2A frozen contract: $Path" }
}

$Venv = Join-Path $Repo '.venv-r7_5_3c_x16'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python -PathType Leaf)) {
    & py -3.11 -c "import struct,sys; assert sys.version_info[:3] == (3,11,9), sys.version; assert struct.calcsize('P') == 8"
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11.9 x64 is required.' }
    & py -3.11 -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to recreate frozen Phase2 Python environment.' }
    & $Python -m pip install --upgrade pip
    & $Python -m pip install --index-url 'https://download.pytorch.org/whl/cpu' 'torch==2.13.0+cpu'
    & $Python -m pip install 'numpy==2.3.5'
}
& $Python -c "import struct,sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9),sys.version; assert struct.calcsize('P')==8; assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print('python',sys.version.split()[0],'bits',struct.calcsize('P')*8,'torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Frozen Phase2A Python/Torch/Numpy runtime verification failed.' }

# Clean explicit AMD64 solver build. Training outputs live elsewhere and are not touched.
$Build = Join-Path $Repo 'build_phase2a'
if (Test-Path $Build) { Remove-Item -Recurse -Force $Build }
& cmake -S $Repo -B $Build -G 'Visual Studio 17 2022' -A x64
if ($LASTEXITCODE -ne 0) { throw 'Phase2A CMake x64 configure failed.' }
& cmake --build $Build --config Release --target spincore_solver_c --parallel
if ($LASTEXITCODE -ne 0) { throw 'Phase2A solver x64 build failed.' }
$SolverCandidates = @(
    @(
        (Join-Path $Build 'Release/spincore_solver_c.dll'),
        (Join-Path $Build 'spincore_solver_c.dll')
    ) | Where-Object { Test-Path $_ }
)
if ($SolverCandidates.Count -ne 1) {
    throw "Expected exactly one x64 solver DLL; found: $($SolverCandidates -join ', ')"
}
$Solver = (Resolve-Path $SolverCandidates[0]).Path
& $Python -c "import struct,sys; p=sys.argv[1]; b=open(p,'rb').read(); assert b[:2]==b'MZ'; pe=struct.unpack_from('<I',b,0x3c)[0]; assert b[pe:pe+4]==b'PE\0\0'; m=struct.unpack_from('<H',b,pe+4)[0]; print(f'solver PE machine 0x{m:04X}'); assert m==0x8664" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2A solver PE architecture preflight failed.' }
& $Python -c "import ctypes as C,sys; L=C.CDLL(sys.argv[1]); L.spincore_solver_c_abi_version.argtypes=[]; L.spincore_solver_c_abi_version.restype=C.c_int32; v=int(L.spincore_solver_c_abi_version()); print('solver ABI',v); assert v==2" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2A solver DLL/ABI preflight failed.' }

$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
& $Python -c "import resource; assert int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)>=0; print('resource compatibility PASS')"
if ($LASTEXITCODE -ne 0) { throw 'Windows resource compatibility preflight failed.' }

Write-Host '[V1+ Phase2A] validating deterministic implementation tests...'
& $Python (Join-Path $Repo 'tools/test_r7_5_3d_v1plus_phase2a.py')
if ($LASTEXITCODE -ne 0) { throw 'Phase2A deterministic contract tests failed.' }

Write-Host '[V1+ Phase2A] validating authoritative Phase-2 source/model contract...'
& $Python -c "import sys; from spincore.r7_5_representation_v3 import H2_FINAL; from spincore.r7_5_representation_v3_stage_contract import validate_phase2_v3_contract; [validate_phase2_v3_contract(sys.argv[1],representation=H2_FINAL,domain='THREE_HANDED',training_seed=s) for s in (1342191342,1801739323)]; print('H2/3H frozen source contract PASS')" $Repo
if ($LASTEXITCODE -ne 0) { throw 'Phase2A authoritative Phase-2 contract preflight failed.' }

$Heldout = Join-Path $Repo 'heldout_v3_bundle'
if (-not (Test-Path $Heldout -PathType Container)) { throw "Missing frozen heldout bundle: $Heldout" }
& $Python -c "import sys; from pathlib import Path; import r7_5_3d_v1plus_phase2a_strategy_capacity as p; root=Path(sys.argv[1]); rows=[(s,p._find_heldout(root,s)) for s in (2029384436,1150634112)]; print('3H heldout identity PASS',[(s,str(path)) for s,path in rows])" $Heldout
if ($LASTEXITCODE -ne 0) { throw 'Phase2A frozen 3H heldout preflight failed.' }

$Output = Join-Path $Repo 'ryzen_v1plus_phase2a'
Write-Host "[V1+ Phase2A] frozen execution HEAD: $Head"
Write-Host "[V1+ Phase2A] solver: $Solver"
Write-Host "[V1+ Phase2A] output/resume root: $Output"
Write-Host '[V1+ Phase2A] collection: two independent H2/3H x4 seed trajectories in parallel; canonical root order remains sequential within each seed.'
Write-Host '[V1+ Phase2A] policy fitting: up to 3 independent capacity-arm processes per seed x 2 torch threads = up to 12 fit threads across both seeds.'

& $Python (Join-Path $Repo 'tools/spincore_ryzen_frozen_runner.py') `
    --expected-commit $Head `
    --run-name 'r7_5_3d_v1plus_phase2a' `
    --contract 'validation/R7_5_3D_V1PLUS_PHASE2A_STRATEGY_MEMORY_CAPACITY_ABLATION_PRECOMMIT_20260821.md' `
    --contract 'validation/R7_5_3D_V1PLUS_PHASE2A_IMPLEMENTATION_FREEZE_20260821.md' `
    --contract 'validation/R7_5_3D_V1PLUS_PHASE2A_RUNTIME_GUARD_20260821.md' `
    --contract 'validation/R7_5_3D_V1PLUS_PHASE2A_PARALLEL_FIT_EXECUTION_FREEZE_20260821.md' `
    --contract 'tools/r7_5_3d_v1plus_phase2a_strategy_capacity.py' `
    --contract 'tools/r7_5_3d_v1plus_phase2a_strategy_capacity_runtimefix.py' `
    --contract 'tools/r7_5_3d_v1plus_phase2a_policy_fit_worker.py' `
    --contract 'tools/test_r7_5_3d_v1plus_phase2a.py' `
    --contract 'tools/r7_5_3c_chance_coverage_x4_domain_worker_runtimefix.py' `
    --contract 'python/spincore/r7_5_representation_v3_stage_contract.py' `
    --contract 'python/spincore_nn/reservoir.py' `
    --artifact $Output `
    -- $Python (Join-Path $Repo 'tools/r7_5_3d_v1plus_phase2a_strategy_capacity_runtimefix.py') `
        --repo-root $Repo `
        --solver $Solver `
        --heldout-root $Heldout `
        --output-root $Output `
        --execution-sha $Head `
        --seed-workers 2

if ($LASTEXITCODE -ne 0) {
    throw "Phase2A stopped with exit code $LASTEXITCODE. Do not delete ryzen_v1plus_phase2a; the run is chunk-resumable and policy-fit-resumable from the same frozen commit."
}

$Result = Join-Path $Output 'R7_5_3D_V1PLUS_PHASE2A_RESULT.json'
if (-not (Test-Path $Result -PathType Leaf)) { throw 'Phase2A runner returned success without the final result JSON.' }
$Hash = (Get-FileHash -Algorithm SHA256 $Result).Hash.ToLowerInvariant()
Write-Host ''
Write-Host '[V1+ Phase2A] COMPLETE'
Write-Host "[V1+ Phase2A] result: $Result"
Write-Host "[V1+ Phase2A] SHA256: $Hash"
Write-Host '[V1+ Phase2A] No representation winner or production authorization has been created.'