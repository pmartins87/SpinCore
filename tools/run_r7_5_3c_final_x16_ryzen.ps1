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
if (-not (Test-Path $Freeze)) { throw "Missing frozen x16 contract: $Freeze" }
if (-not (Test-Path $RuntimeCorrection)) { throw "Missing frozen Windows runtime correction: $RuntimeCorrection" }
if (-not (Test-Path $ResourceCorrection)) { throw "Missing frozen Windows resource-import correction: $ResourceCorrection" }
if (-not (Test-Path $ResourceHandleFix)) { throw "Missing frozen Windows resource HANDLE fix: $ResourceHandleFix" }

# Python 3.11.9 is the final Python 3.11 release with an official Windows binary installer.
# The x16 algorithm/seeds/budgets/gates remain unchanged; the runtime correction is execution-only.
& py -3.11 -c "import sys; assert sys.version_info[:3] == (3,11,9), sys.version"
if ($LASTEXITCODE -ne 0) {
    throw 'Python 3.11.9 is required for the frozen Windows x16 run (py -3.11 must resolve exactly to 3.11.9).'
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
& $Python -c "import sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9); assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print(sys.version); print('torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Frozen Python/Torch/Numpy runtime verification failed.' }

$Build = Join-Path $Repo 'build_x16'
& cmake -S $Repo -B $Build -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }
& cmake --build $Build --config Release --target spincore_solver_c --parallel
if ($LASTEXITCODE -ne 0) { throw 'spincore_solver_c build failed.' }

# Force array semantics even when exactly one DLL exists; StrictMode otherwise
# exposes the single pipeline result as a scalar without a Count property.
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

# The shared Phase-2 stage imports Python's Unix-only resource module solely for
# peak-RSS telemetry. Prepend a Windows-only compatibility shim for this x16 run.
$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
& $Python -c "import resource; r=resource.getrusage(resource.RUSAGE_SELF); assert int(r.ru_maxrss)>0; print('resource shim peak_rss_kib', int(r.ru_maxrss))"
if ($LASTEXITCODE -ne 0) { throw 'Windows resource telemetry compatibility preflight failed.' }

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
    --contract 'validation/R7_5_3C_CHANCE_COVERAGE_X4_STABILITY_EVIDENCE_20260818.json' `
    --contract 'validation/R7_5_FINITE_CLOSURE_AND_COMPUTE_POLICY_20260816.md' `
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
