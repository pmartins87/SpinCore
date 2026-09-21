param(
    [Parameter(Mandatory=$true)]
    [string]$SourceDir
)

$ErrorActionPreference = "Stop"

$SourceDir = (Resolve-Path $SourceDir).Path
$BuildDir = Join-Path $SourceDir "build_win_shadow_x64"
$BundleSource = Join-Path $SourceDir "artifacts\SpinCore_LT2_cpp_deployment_8100.bin"
$ExpectedSha = "2b79ab7ff746a9c1c3dd73dbc0a1d6884a471813cf34b9cb126790c4c4cbb123"

if (-not (Test-Path $BundleSource)) {
    throw "Frozen native bundle missing: $BundleSource"
}
$ActualSha = (Get-FileHash -Algorithm SHA256 $BundleSource).Hash.ToLowerInvariant()
if ($ActualSha -ne $ExpectedSha) {
    throw "Frozen native bundle SHA256 mismatch. expected=$ExpectedSha actual=$ActualSha"
}
Write-Host "WINDOWS_FROZEN_BUNDLE_SHA256_PASS"

if (Test-Path $BuildDir) {
    Remove-Item -Recurse -Force $BuildDir
}

& cmake -S $SourceDir -B $BuildDir -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed: $LASTEXITCODE" }

& cmake --build $BuildDir --config Release --target spincore_tests spincore_lt2_openholdem_shadow_dll spincore_lt2_openholdem_shadow_dll_mock_host -- /m
if ($LASTEXITCODE -ne 0) { throw "MSVC build failed: $LASTEXITCODE" }
Write-Host "WINDOWS_SHADOW_DLL_BUILD_PASS"

& ctest --test-dir $BuildDir -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { throw "Windows CTest failed: $LASTEXITCODE" }
Write-Host "WINDOWS_CORE_TESTS_PASS"

$ReleaseDir = Join-Path $BuildDir "Release"
$Dll = Join-Path $ReleaseDir "SpinCore_LT2_Shadow.dll"
$Mock = Join-Path $ReleaseDir "spincore_lt2_openholdem_shadow_dll_mock_host.exe"
if (-not (Test-Path $Dll)) { throw "Shadow DLL not produced: $Dll" }
if (-not (Test-Path $Mock)) { throw "Mock host not produced: $Mock" }

$BundleDest = Join-Path $ReleaseDir "SpinCore_LT2_cpp_deployment_8100.bin"
Copy-Item -Force $BundleSource $BundleDest
$CopiedSha = (Get-FileHash -Algorithm SHA256 $BundleDest).Hash.ToLowerInvariant()
if ($CopiedSha -ne $ExpectedSha) {
    throw "Bundle copy SHA256 mismatch"
}

$Report = Join-Path $BuildDir "SpinCore_LT2_openholdem_shadow_dll_mock_gate.json"
if (Test-Path $Report) { Remove-Item -Force $Report }

& $Mock $Dll $Report
if ($LASTEXITCODE -ne 0) { throw "Shadow DLL mock-host gate failed: $LASTEXITCODE" }
if (-not (Test-Path $Report)) { throw "Mock-host report not produced" }

$d = Get-Content -Raw $Report | ConvertFrom-Json
if ($d.schema -ne "SPINCORE_LT2_OPENHOLDEM_SHADOW_DLL_MOCK_HOST_V1") { throw "Wrong report schema" }
if ($d.verdict -ne "PASS") { throw "Mock-host verdict is not PASS" }
if (-not $d.shadow_only) { throw "Shadow-only barrier missing" }
if (-not $d.host_api_resolved) { throw "Host API resolution failed" }
if (-not $d.hand_anchor_started) { throw "Hand anchor did not start" }
if (-not $d.myturn_shadow_decision_ready) { throw "MyTurn decision unavailable" }
if (-not $d.repeated_myturn_cache_stable) { throw "MyTurn cache instability" }
if (-not $d.action_queries_hard_zero) { throw "Action-query safety barrier failed" }
if ($d.table_actions_executed -ne 0) { throw "Shadow gate executed a table action" }
if ([math]::Abs([double]$d.probability_sum - 1.0) -gt 0.00001) { throw "Probability mass drift" }
if ([int]$d.legal_action_count -le 0) { throw "No legal action in mock decision" }

$Downloads = Join-Path $env:USERPROFILE "Downloads"
$OutDll = Join-Path $Downloads "SpinCore_LT2_Shadow.dll"
$OutReport = Join-Path $Downloads "SpinCore_LT2_openholdem_shadow_dll_mock_gate.json"
$OutLog = Join-Path $Downloads "SpinCore_LT2_shadow_mock.log"

Copy-Item -Force $Dll $OutDll
Copy-Item -Force $Report $OutReport
$GeneratedLog = Join-Path $ReleaseDir "SpinCore_LT2_shadow.log"
if (Test-Path $GeneratedLog) {
    Copy-Item -Force $GeneratedLog $OutLog
}

$DllHash = (Get-FileHash -Algorithm SHA256 $OutDll).Hash.ToLowerInvariant()
Write-Host "shadow_dll=$OutDll"
Write-Host "shadow_dll_sha256=$DllHash"
Write-Host "report=$OutReport"
Write-Host "LT2_OPENHOLDEM_SHADOW_DLL_WINDOWS_GATE_PASS"
Write-Host "STOP HERE. Send SpinCore_LT2_openholdem_shadow_dll_mock_gate.json and the terminal output to ChatGPT."
