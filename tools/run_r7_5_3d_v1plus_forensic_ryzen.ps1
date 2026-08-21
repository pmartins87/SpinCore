param(
    [string]$RepoRoot = "",
    [string]$InputRoot = "",
    [string]$HeldoutRoot = "",
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
if (-not $HeldoutRoot) {
    $HeldoutRoot = Join-Path $RepoRoot "heldout_v3_bundle"
}
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $RepoRoot "ryzen_v1plus_postmortem"
}

$TrainingExecutionSha = "03052a0896d138aa6d2b3ba2ff1473a817e5a113"
$Python = Join-Path $RepoRoot ".venv-r7_5_3c_x16\Scripts\python.exe"
$RawOut = Join-Path $OutputRoot "R7_5_3D_V1PLUS_FORENSIC_RAW.json"
$EnrichedOut = Join-Path $OutputRoot "R7_5_3D_V1PLUS_FORENSIC_ENRICHED.json"
$ManifestOut = Join-Path $OutputRoot "R7_5_3D_V1PLUS_FORENSIC_LOCAL_MANIFEST.json"

Write-Host "[V1+] read-only forensic launcher"
Write-Host "[V1+] repo:       $RepoRoot"
Write-Host "[V1+] x16 input:  $InputRoot"
Write-Host "[V1+] heldout:    $HeldoutRoot"
Write-Host "[V1+] output:     $OutputRoot"
Write-Host "[V1+] x16 SHA:    $TrainingExecutionSha"
Write-Host "[V1+] NO TRAINING / NO MODEL MUTATION / NO PRODUCTION AUTHORIZATION"

if (-not (Test-Path $Python -PathType Leaf)) {
    throw "Frozen x16 Python environment not found: $Python"
}
if (-not (Test-Path $InputRoot -PathType Container)) {
    throw "x16 checkpoint root not found: $InputRoot"
}
if (-not (Test-Path $HeldoutRoot -PathType Container)) {
    throw "heldout bundle root not found: $HeldoutRoot"
}

$RequiredHeldout = @(
    "TRUE_HEADS_UP\2029384436\states.json.gz",
    "TRUE_HEADS_UP\1150634112\states.json.gz",
    "THREE_HANDED\2029384436\states.json.gz",
    "THREE_HANDED\1150634112\states.json.gz"
)
foreach ($Rel in $RequiredHeldout) {
    $Path = Join-Path $HeldoutRoot $Rel
    if (-not (Test-Path $Path -PathType Leaf)) {
        throw "Missing frozen heldout corpus: $Path"
    }
}

$env:PYTHONPATH = ((Join-Path $RepoRoot "python"), (Join-Path $RepoRoot "tools"), (Join-Path $RepoRoot "tools\windows_compat")) -join ";"
$env:SPINCORE_TORCH_THREADS = "2"
$env:OMP_NUM_THREADS = "2"
$env:MKL_NUM_THREADS = "2"

Write-Host "[V1+] checking runtime..."
& $Python -c "import sys, torch, numpy as np; print('python',sys.version.split()[0]); print('torch',torch.__version__); print('numpy',np.__version__); assert sys.version_info[:2]==(3,11); assert torch.__version__=='2.13.0+cpu'; assert np.__version__=='2.3.5'"
if ($LASTEXITCODE -ne 0) { throw "Frozen runtime preflight failed" }

Write-Host "[V1+] running deterministic synthetic tests..."
& $Python (Join-Path $RepoRoot "tools\test_r7_5_3d_v1plus_forensic_enrich.py")
if ($LASTEXITCODE -ne 0) { throw "V1+ forensic synthetic tests failed" }

Write-Host "[V1+] validating exact frozen input inventory..."
$InventoryScript = @'
import json, sys
from pathlib import Path
import torch

root=Path(sys.argv[1])
sha=sys.argv[2]
rows=[]
for cp in sorted(root.rglob('checkpoint.pt')):
    rp=cp.parent/'report.json'
    if not rp.exists():
        continue
    report=json.loads(rp.read_text(encoding='utf-8'))
    if not bool(report.get('finalized')):
        continue
    payload=torch.load(cp,map_location='cpu',weights_only=False)
    rows.append({
        'checkpoint':str(cp),
        'representation':str(payload.get('representation')),
        'domain':str(payload.get('domain')),
        'seed':int(payload.get('seed',-1)),
        'execution_sha':str(payload.get('execution_sha')),
        'phase':str((payload.get('progress') or {}).get('phase')),
    })
if len(rows)!=8:
    raise SystemExit(f'expected exactly 8 finalized x16 checkpoints, found {len(rows)}')
if any(row['execution_sha']!=sha for row in rows):
    raise SystemExit('x16 execution SHA mismatch')
if any(row['phase']!='post_policy_fit' for row in rows):
    raise SystemExit('x16 checkpoint is not post_policy_fit')
keys={(r['representation'],r['domain'],r['seed']) for r in rows}
if len(keys)!=8:
    raise SystemExit('duplicate/missing x16 representation-domain-seed identities')
print(json.dumps(rows,indent=2,sort_keys=True))
'@
& $Python -c $InventoryScript $InputRoot $TrainingExecutionSha
if ($LASTEXITCODE -ne 0) { throw "Frozen x16 inventory preflight failed" }

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

Write-Host "[V1+] running RAW post-mortem (read-only)..."
& $Python (Join-Path $RepoRoot "tools\r7_5_3d_v1plus_forensic_postmortem.py") `
    --repo-root $RepoRoot `
    --input-root $InputRoot `
    --heldout-root $HeldoutRoot `
    --training-execution-sha $TrainingExecutionSha `
    --out $RawOut
if ($LASTEXITCODE -ne 0) { throw "Raw V1+ forensic post-mortem failed" }

Write-Host "[V1+] running deterministic enrichment..."
& $Python (Join-Path $RepoRoot "tools\r7_5_3d_v1plus_forensic_enrich.py") `
    --raw $RawOut `
    --heldout-root $HeldoutRoot `
    --out $EnrichedOut
if ($LASTEXITCODE -ne 0) { throw "Enriched V1+ forensic post-mortem failed" }

$RawHash = (Get-FileHash -Algorithm SHA256 $RawOut).Hash.ToLowerInvariant()
$EnrichedHash = (Get-FileHash -Algorithm SHA256 $EnrichedOut).Hash.ToLowerInvariant()
$HeldoutManifest = Join-Path $HeldoutRoot "MANIFEST.json"
$HeldoutManifestHash = if (Test-Path $HeldoutManifest -PathType Leaf) { (Get-FileHash -Algorithm SHA256 $HeldoutManifest).Hash.ToLowerInvariant() } else { $null }
$Head = (& git -C $RepoRoot rev-parse HEAD).Trim()

$Manifest = [ordered]@{
    schema = "SPINCORE_R7_5_3D_V1PLUS_FORENSIC_LOCAL_MANIFEST_V1"
    status = "COMPLETE_NO_ARCHITECTURE_SELECTED"
    diagnostic_execution_sha = $Head
    x16_training_execution_sha = $TrainingExecutionSha
    input_root = $InputRoot
    heldout_root = $HeldoutRoot
    raw = [ordered]@{ path = $RawOut; sha256 = $RawHash }
    enriched = [ordered]@{ path = $EnrichedOut; sha256 = $EnrichedHash }
    heldout_manifest_sha256 = $HeldoutManifestHash
    production_training_authorized = $false
    ready_for_tables = $false
}
$Manifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $ManifestOut
$ManifestHash = (Get-FileHash -Algorithm SHA256 $ManifestOut).Hash.ToLowerInvariant()

Write-Host ""
Write-Host "[V1+] FORENSIC READOUT COMPLETE"
Write-Host "[V1+] raw SHA256:      $RawHash"
Write-Host "[V1+] enriched SHA256: $EnrichedHash"
Write-Host "[V1+] manifest SHA256: $ManifestHash"
Write-Host "[V1+] preserve directory: $OutputRoot"
Write-Host "[V1+] No architecture winner has been selected."
