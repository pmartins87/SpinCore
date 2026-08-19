$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path '.').Path
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$Dirty = (& git -C $Repo status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) {
    throw "Tracked worktree is dirty. Commit/revert tracked changes before the frozen x16 run.`n$Dirty"
}

$Freeze = Join-Path $Repo 'validation/R7_5_3C_FINAL_CONTINGENCY_X16_FREEZE_20260818.json'
$RuntimeCorrection = Join-Path $Repo 'validation/R7_5_3C_FINAL_X16_WINDOWS_RUNTIME_CORRECTION_20260818.json'
$ResourceCorrection = Join-Path $Repo 'validation/R7_5_3C_FINAL_X16_WINDOWS_RESOURCE_IMPORT_CORRECTION_20260818.json'
$ResourceHandleFix = Join-Path $Repo 'validation/R7_5_3C_FINAL_X16_WINDOWS_RESOURCE_HANDLE_FIX_20260818.json'
$LfCorrection = Join-Path $Repo 'validation/R7_5_3C_FINAL_X16_WINDOWS_LF_CHECKOUT_CORRECTION_20260818.json'
$DllCorrection = Join-Path $Repo 'validation/R7_5_3C_FINAL_X16_WINDOWS_DLL_LOAD_CORRECTION_20260818.json'
if (-not (Test-Path $Freeze)) { throw "Missing frozen x16 contract: $Freeze" }
if (-not (Test-Path $RuntimeCorrection)) { throw "Missing frozen Windows runtime correction: $RuntimeCorrection" }
if (-not (Test-Path $ResourceCorrection)) { throw "Missing frozen Windows resource-import correction: $ResourceCorrection" }
if (-not (Test-Path $ResourceHandleFix)) { throw "Missing frozen Windows resource HANDLE fix: $ResourceHandleFix" }
if (-not (Test-Path $LfCorrection)) { throw "Missing frozen Windows LF checkout correction: $LfCorrection" }
if (-not (Test-Path $DllCorrection)) { throw "Missing frozen Windows solver DLL correction: $DllCorrection" }

# Python 3.11.9 is the final Python 3.11 release with an official Windows binary installer.
# The x16 algorithm/seeds/budgets/gates remain unchanged; the runtime correction is execution-only.
& py -3.11 -c "import struct,sys; assert sys.version_info[:3] == (3,11,9), sys.version; assert struct.calcsize('P') == 8, '64-bit Python required'"
if ($LASTEXITCODE -ne 0) {
    throw 'Python 3.11.9 x64 is required for the frozen Windows x16 run.'
}

$Venv = Join-Path $Repo '.venv-r7_5_3c_x16'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python)) {
    & py -3.11 -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create x16 virtual environment.' }
}
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed.' }
& $Python -m pip install --index-url 'https://download.pytorch.org/whl/cpu' 'torch==2.13.0+cpu'
if ($LASTEXITCODE -ne 0) { throw 'Frozen torch install failed.' }
& $Python -m pip install 'numpy==2.3.5'
if ($LASTEXITCODE -ne 0) { throw 'Frozen numpy install failed.' }
& $Python -c "import struct,sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9); assert struct.calcsize('P')==8; assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print(sys.version); print('python_bits',struct.calcsize('P')*8,'torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Frozen Python/Torch/Numpy runtime verification failed.' }

# Rebuild the solver from a clean directory with an explicit AMD64 target.  The
# build directory is execution-only; checkpoints/results live elsewhere and are untouched.
$Build = Join-Path $Repo 'build_x16'
if (Test-Path $Build) {
    Remove-Item -Recurse -Force $Build
}
& cmake -S $Repo -B $Build -G 'Visual Studio 17 2022' -A x64
if ($LASTEXITCODE -ne 0) { throw 'CMake x64 configure failed.' }
& cmake --build $Build --config Release --target spincore_solver_c --parallel
if ($LASTEXITCODE -ne 0) { throw 'spincore_solver_c x64 build failed.' }

$SolverCandidates = @(
    @(
        (Join-Path $Build 'Release/spincore_solver_c.dll'),
        (Join-Path $Build 'spincore_solver_c.dll')
    ) | Where-Object { Test-Path $_ }
)
if ($SolverCandidates.Count -ne 1) {
    throw "Expected exactly one Windows solver DLL after build; found: $($SolverCandidates -join ', ')"
}
$Solver = (Resolve-Path $SolverCandidates[0]).Path

# Verify the produced PE is AMD64, then load it with the exact Python runtime and
# check ABI v2 before any x16 worker is allowed to start.
& $Python -c "import struct,sys; p=sys.argv[1]; b=open(p,'rb').read(); assert b[:2]==b'MZ','missing MZ'; pe=struct.unpack_from('<I',b,0x3c)[0]; assert b[pe:pe+4]==b'PE\0\0','missing PE'; m=struct.unpack_from('<H',b,pe+4)[0]; print(f'solver PE machine 0x{m:04X}'); assert m==0x8664, f'expected AMD64 0x8664, got 0x{m:04X}'" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Solver PE architecture preflight failed.' }
& $Python -c "import ctypes as C,sys; L=C.CDLL(sys.argv[1]); L.spincore_solver_c_abi_version.argtypes=[]; L.spincore_solver_c_abi_version.restype=C.c_int32; v=int(L.spincore_solver_c_abi_version()); print('solver DLL load PASS ABI',v); assert v==2, v" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Solver DLL load/ABI preflight failed before x16 workers.' }

# The shared Phase-2 stage imports Python's Unix-only resource module solely for
# peak-RSS telemetry. Prepend a Windows compatibility shim. A value of zero is
# explicitly allowed and means telemetry unavailable; it never gates training.
$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
& $Python -c "import resource; r=resource.getrusage(resource.RUSAGE_SELF); assert int(r.ru_maxrss)>=0; print('resource shim peak_rss_kib', int(r.ru_maxrss))"
if ($LASTEXITCODE -ne 0) { throw 'Windows resource telemetry compatibility import failed.' }

# Validate the complete frozen model/source contract once before starting any
# worker. .gitattributes pins the six byte-hashed model sources to LF on Windows.
& $Python -c "import sys; from spincore.r7_5_representation_v3 import H2_FINAL; from spincore.r7_5_representation_v3_stage_contract import validate_phase2_v3_contract; validate_phase2_v3_contract(sys.argv[1], representation=H2_FINAL, domain='TRUE_HEADS_UP', training_seed=1342191342); print('phase2 frozen source contract PASS')" $Repo
if ($LASTEXITCODE -ne 0) { throw 'Frozen Phase-2 source/model contract preflight failed before x16 workers.' }

$Output = 'ryzen_x16_final'
Write-Host "[SpinCore x16] frozen HEAD: $Head"
Write-Host "[SpinCore x16] solver: $Solver"
Write-Host "[SpinCore x16] output/resume root: $Output"
Write-Host '[SpinCore x16] starting up to four independent cells in parallel; each cell remains sequential and exact.'

& $Python tools/spincore_ryzen_frozen_runner.py `
    --expected-commit $Head `
    --run-name 'r7_5_3c_final_x16' `
    --contract 'validation/R7_5_3C_FINAL_CONTINGENCY_X16_FREEZE_20260818.json' `
    --contract 'validation/R7_5_3C_FINAL_X16_WINDOWS_RUNTIME_CORRECTION_20260818.json' `
    --contract 'validation/R7_5_3C_FINAL_X16_WINDOWS_RESOURCE_IMPORT_CORRECTION_20260818.json' `
    --contract 'validation/R7_5_3C_FINAL_X16_WINDOWS_RESOURCE_HANDLE_FIX_20260818.json' `
    --contract 'validation/R7_5_3C_FINAL_X16_WINDOWS_LF_CHECKOUT_CORRECTION_20260818.json' `
    --contract 'validation/R7_5_3C_FINAL_X16_WINDOWS_DLL_LOAD_CORRECTION_20260818.json' `
    --contract 'validation/R7_5_3C_CHANCE_COVERAGE_X4_STABILITY_EVIDENCE_20260818.json' `
    --contract 'validation/R7_5_FINITE_CLOSURE_AND_COMPUTE_POLICY_20260816.md' `
    --contract '.gitattributes' `
    --contract 'tools/windows_compat/resource.py' `
    --contract 'tools/r7_5_3c_final_x16_domain_worker.py' `
    --contract 'tools/r7_5_3c_final_x16_ryzen_orchestrator.py' `
    --contract 'python/spincore/r7_5_representation_v3_stage_contract.py' `
    --artifact $Output `
    -- $Python tools/r7_5_3c_final_x16_ryzen_orchestrator.py `
        --repo-root $Repo `
        --solver $Solver `
        --output-root $Output `
        --execution-sha $Head `
        --cell-workers 4

if ($LASTEXITCODE -ne 0) {
    throw "Frozen x16 execution stopped with exit code $LASTEXITCODE. The output is resumable; rerun this same script from the same frozen commit after correcting only the mechanical runtime/resource cause."
}

Write-Host '[SpinCore x16] TRAINING COMPLETE. Preserve the entire ryzen_x16_final directory and the newest ryzen_runs/*r7_5_3c_final_x16*/manifest.json for independent stability certification.'
