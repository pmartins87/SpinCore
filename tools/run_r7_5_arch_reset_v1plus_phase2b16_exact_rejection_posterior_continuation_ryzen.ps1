$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo = (Resolve-Path '.').Path
$Head = (& git -C $Repo rev-parse HEAD).Trim()
$Dirty = (& git -C $Repo status --porcelain=v1 --untracked-files=no) -join "`n"
if ($Dirty) {
    throw "Tracked worktree is dirty. Untracked files are intentionally ignored. Stop and send this output before Phase2B16:`n$Dirty"
}

$Precommit = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B16_EXACT_REJECTION_POSTERIOR_CONTINUATION_PRECOMMIT_20260824.md'
$B15Evidence = Join-Path $Repo 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B15_RESULT_EVIDENCE_20260824.json'
foreach ($Path in @($Precommit, $B15Evidence)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing frozen Phase2B16 contract/evidence: $Path" }
}

$Venv = Join-Path $Repo '.venv-r7_5_3c_x16'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python -PathType Leaf)) { throw "Missing existing frozen Phase2 Python environment: $Python" }
& $Python -c "import struct,sys,torch,numpy as np; assert sys.version_info[:3]==(3,11,9),sys.version; assert struct.calcsize('P')==8; assert torch.__version__=='2.13.0+cpu',torch.__version__; assert np.__version__=='2.3.5',np.__version__; print('python',sys.version.split()[0],'bits',struct.calcsize('P')*8,'torch',torch.__version__,'numpy',np.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Frozen Phase2B16 Python/Torch/Numpy runtime verification failed.' }

$env:PYTHONPATH = "$(Join-Path $Repo 'tools/windows_compat');$(Join-Path $Repo 'python');$(Join-Path $Repo 'tools')"
$env:SPINCORE_TORCH_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'

$Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b16_exact_rejection_posterior_continuation.py'
$Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b16_exact_rejection_posterior_continuation.py'
$B15Tool = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance.py'
$B15Fix = Join-Path $Repo 'tools/r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance_runtimefix.py'
$B10Test = Join-Path $Repo 'tools/test_r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition.py'

Write-Host '[V1+ Phase2B16] compiling exact-rejection posterior scripts...'
& $Python -m py_compile $Tool $Test $B15Tool $B15Fix
if ($LASTEXITCODE -ne 0) { throw 'Phase2B16 py_compile failed.' }
Write-Host '[V1+ Phase2B16] running deterministic synthetic tests...'
& $Python $Test
if ($LASTEXITCODE -ne 0) { throw 'Phase2B16 synthetic tests failed.' }

$B13Root = Join-Path $Repo 'ryzen_v1plus_phase2b13'
$B13Result = Join-Path $B13Root 'R7_5_ARCH_RESET_V1PLUS_PHASE2B13_ROOT_IID64_TARGET_TRAINING.json'
$B14Result = Join-Path $Repo 'ryzen_v1plus_phase2b14/R7_5_ARCH_RESET_V1PLUS_PHASE2B14_B13_RESIDUAL_LOCALIZATION.json'
$B15Root = Join-Path $Repo 'ryzen_v1plus_phase2b15'
$B15Result = Join-Path $B15Root 'R7_5_ARCH_RESET_V1PLUS_PHASE2B15_POSTERIOR_WEIGHTED_CONTINUATION_CHANCE.json'
$Heldout = Join-Path $Repo 'heldout_v3_bundle'
foreach ($Path in @($B13Result, $B14Result, $B15Result)) {
    if (-not (Test-Path $Path -PathType Leaf)) { throw "Missing exact completed Phase2B16 prerequisite result: $Path" }
}
if (-not (Test-Path $Heldout -PathType Container)) { throw "Missing frozen heldout bundle: $Heldout" }

Write-Host '[V1+ Phase2B16] validating exact Phase2B15 result identity and failed-screen status...'
& $Python -c "import hashlib,json,sys; p,e=sys.argv[1:3]; raw=open(p,'rb').read(); h=hashlib.sha256(raw).hexdigest(); ev=json.load(open(e,encoding='utf-8')); j=json.loads(raw); expected='0e4f0a5bf2d48fb7f48b2763f8a65e3093d879aa50729f5d8a80d28fa9578f6a'; assert h==expected==ev['uploaded_result_sha256']; assert j['execution_sha']=='ea53812e39efaa41889ea6ff676eeda9f0c0e2b0'; assert j['status']=='POSTERIOR_WEIGHTING_MATERIAL_BUT_STABILITY_NOT_SUPPORTED'; assert j['decision']['importance_weight_health_pass'] is True; assert j['decision']['screen_pass'] is False; assert j['decision']['small_training_pilot_precommit_allowed'] is False; print('Phase2B15 exact failed-screen evidence PASS',h)" $B15Result $B15Evidence
if ($LASTEXITCODE -ne 0) { throw 'Phase2B16 Phase2B15 prerequisite evidence preflight failed.' }

Write-Host '[V1+ Phase2B16] validating authoritative H2/3H source/model contract...'
& $Python -c "import sys; from spincore.r7_5_representation_v3 import H2_FINAL; from spincore.r7_5_representation_v3_stage_contract import validate_phase2_v3_contract; [validate_phase2_v3_contract(sys.argv[1],representation=H2_FINAL,domain='THREE_HANDED',training_seed=s) for s in (1342191342,1801739323)]; print('H2/3H frozen source contract PASS')" $Repo
if ($LASTEXITCODE -ne 0) { throw 'Phase2B16 source/model contract preflight failed.' }

$Build = Join-Path $Repo 'build_phase2b16'
if (Test-Path $Build) { Remove-Item -Recurse -Force $Build }
& cmake -S $Repo -B $Build -G 'Visual Studio 17 2022' -A x64
if ($LASTEXITCODE -ne 0) { throw 'Phase2B16 CMake x64 configure failed.' }
& cmake --build $Build --config Release --target spincore_solver_c --parallel
if ($LASTEXITCODE -ne 0) { throw 'Phase2B16 solver x64 build failed.' }
$SolverCandidates = @(@((Join-Path $Build 'Release/spincore_solver_c.dll'),(Join-Path $Build 'spincore_solver_c.dll')) | Where-Object { Test-Path $_ })
if ($SolverCandidates.Count -ne 1) { throw "Expected exactly one x64 solver DLL; found: $($SolverCandidates -join ', ')" }
$Solver = (Resolve-Path $SolverCandidates[0]).Path
& $Python -c "import ctypes as C,sys; L=C.CDLL(sys.argv[1]); L.spincore_solver_c_abi_version.argtypes=[]; L.spincore_solver_c_abi_version.restype=C.c_int32; v=int(L.spincore_solver_c_abi_version()); print('solver ABI',v); assert v==2; assert getattr(L,'spincore_solver_state_neural_input_v3',None) is not None; assert getattr(L,'spincore_solver_state_create_v2_deal',None) is not None; assert getattr(L,'spincore_solver_state_deal_snapshot_v1',None) is not None; print('SPNNIV3 + explicit-deal diagnostic solver API PASS')" $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2B16 solver ABI/explicit-deal preflight failed.' }

Write-Host '[V1+ Phase2B16] rerunning explicit-deal solver round-trip tests...'
& $Python $B10Test --repo-root $Repo --solver $Solver
if ($LASTEXITCODE -ne 0) { throw 'Phase2B16 explicit-deal solver round-trip tests failed.' }

Write-Host '[V1+ Phase2B16] validating all 64 Windows canonical heldout continuation anchors...'
& $Python $B15Fix --preflight-only --repo-root $Repo --solver $Solver --heldout-root $Heldout --phase2b14-result $B14Result
if ($LASTEXITCODE -ne 0) { throw 'Phase2B16 canonical heldout reconstruction preflight failed.' }

$Output = Join-Path $Repo 'ryzen_v1plus_phase2b16'
$Result = Join-Path $Output 'R7_5_ARCH_RESET_V1PLUS_PHASE2B16_EXACT_REJECTION_POSTERIOR_CONTINUATION.json'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Workers = 30

Write-Host "[V1+ Phase2B16] screen HEAD: $Head"
Write-Host "[V1+ Phase2B16] workers: up to $Workers independent one-thread exact-posterior tasks"
Write-Host '[V1+ Phase2B16] FINAL ESTIMATOR-LEVEL SCREEN: exact rejection posterior, K64 accepted targets per block.'
Write-Host '[V1+ Phase2B16] proposal prior: uniform opponent private cards given current actor holes; accept with exact frozen action-path likelihood.'
Write-Host '[V1+ Phase2B16] future board is sampled only after private-card acceptance; no likelihood floor/clipping/tempering, MCMC, or SIR.'
Write-Host '[V1+ Phase2B16] successful Phase2B15 runtimefix partials must reproduce their aggregate before interpretation.'
Write-Host '[V1+ Phase2B16] no training, optimizer step, reservoir mutation, x4 confirmation, architecture selection, or production authorization.'

& $Python (Join-Path $Repo 'tools/spincore_ryzen_frozen_runner.py') `
    --expected-commit $Head `
    --run-name 'r7_5_arch_reset_v1plus_phase2b16_exact_rejection_posterior_continuation' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B16_EXACT_REJECTION_POSTERIOR_CONTINUATION_PRECOMMIT_20260824.md' `
    --contract 'validation/R7_5_ARCH_RESET_V1PLUS_PHASE2B15_RESULT_EVIDENCE_20260824.json' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b16_exact_rejection_posterior_continuation.py' `
    --contract 'tools/test_r7_5_arch_reset_v1plus_phase2b16_exact_rejection_posterior_continuation.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance_runtimefix.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator.py' `
    --contract 'tools/r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition.py' `
    --contract 'python/spincore/r7_5_representation_v3_referee_states.py' `
    --contract 'python/spincore/r7_5_representation_v3_uncertainty.py' `
    --contract 'python/spincore/r7_5_action_cfr.py' `
    --contract 'python/spincore/solver.py' `
    --contract 'include/spincore/solver_c_api.h' `
    --contract 'src/solver_c_api.cpp' `
    --artifact $Output `
    -- $Python $Tool `
        --repo-root $Repo `
        --solver $Solver `
        --heldout-root $Heldout `
        --phase2b13-root $B13Root `
        --phase2b13-result $B13Result `
        --phase2b14-result $B14Result `
        --phase2b15-root $B15Root `
        --phase2b15-result $B15Result `
        --output-root $Output `
        --execution-sha $Head `
        --workers $Workers

if ($LASTEXITCODE -ne 0) { throw "Phase2B16 failed with exit code $LASTEXITCODE. Preserve ryzen_v1plus_phase2b16 and all prior artifacts; do not delete partials." }
if (-not (Test-Path $Result -PathType Leaf)) { throw 'Phase2B16 returned success without result JSON.' }
$Hash = (Get-FileHash -Algorithm SHA256 $Result).Hash.ToLowerInvariant()
Write-Host ''
Write-Host '[V1+ Phase2B16] COMPLETE'
Write-Host "[V1+ Phase2B16] result: $Result"
Write-Host "[V1+ Phase2B16] SHA256: $Hash"
Write-Host '[V1+ Phase2B16] Send the result JSON back. PASS permits only a separately precommitted small causal pilot; FAIL closes estimator-level posterior repair.'
