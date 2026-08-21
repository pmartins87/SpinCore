param(
    [string]$RepoRoot = "",
    [string]$InputRoot = "",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
} else {
    $RepoRoot = (Resolve-Path $RepoRoot).Path
}
if (-not $InputRoot) {
    $InputRoot = Join-Path $RepoRoot "ryzen_x16_final"
}
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $RepoRoot "ryzen_v1plus_postmortem"
}

$TrainingExecutionSha = "f44e05513721b59f63ed5c61f37de2c115c67315"
$Python = Join-Path $RepoRoot ".venv-r7_5_3c_x16\Scripts\python.exe"
$Out = Join-Path $OutputRoot "R7_5_3D_V1PLUS_PHASE1B_PROJECTION_DECOMPOSITION.json"

Write-Host "[V1+ Phase1B] read-only reservoir projection decomposition"
Write-Host "[V1+ Phase1B] repo:      $RepoRoot"
Write-Host "[V1+ Phase1B] x16 input: $InputRoot"
Write-Host "[V1+ Phase1B] output:    $Out"
Write-Host "[V1+ Phase1B] x16 SHA:   $TrainingExecutionSha"
Write-Host "[V1+ Phase1B] NO TRAINING / NO TRAVERSAL / NO MODEL OR RESERVOIR MUTATION"

if (-not (Test-Path $Python -PathType Leaf)) {
    throw "Frozen x16 Python environment not found: $Python"
}
if (-not (Test-Path $InputRoot -PathType Container)) {
    throw "x16 checkpoint root not found: $InputRoot"
}

$env:PYTHONPATH = ((Join-Path $RepoRoot "python"), (Join-Path $RepoRoot "tools"), (Join-Path $RepoRoot "tools\windows_compat")) -join ";"
$env:SPINCORE_TORCH_THREADS = "2"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"

Write-Host "[V1+ Phase1B] checking runtime..."
& $Python -c "import sys, torch, numpy as np; print('python',sys.version.split()[0]); print('torch',torch.__version__); print('numpy',np.__version__); assert sys.version_info[:2]==(3,11); assert torch.__version__=='2.13.0+cpu'; assert np.__version__=='2.3.5'"
if ($LASTEXITCODE -ne 0) { throw "Frozen runtime preflight failed" }

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

Write-Host "[V1+ Phase1B] starting decomposition; long quiet periods while decoding reservoirs are normal..."
& $Python (Join-Path $RepoRoot "tools\r7_5_3d_v1plus_projection_decomposition.py") `
    --repo-root $RepoRoot `
    --input-root $InputRoot `
    --training-execution-sha $TrainingExecutionSha `
    --out $Out
if ($LASTEXITCODE -ne 0) { throw "V1+ Phase1B projection decomposition failed" }

$Hash = (Get-FileHash -Algorithm SHA256 $Out).Hash.ToLowerInvariant()
Write-Host ""
Write-Host "[V1+ Phase1B] COMPLETE"
Write-Host "[V1+ Phase1B] SHA256: $Hash"
Write-Host "[V1+ Phase1B] preserve: $Out"
Write-Host "[V1+ Phase1B] No architecture winner has been selected."
