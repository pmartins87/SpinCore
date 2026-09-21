param(
    [string]$OpenHoldemExe = ""
)

$ErrorActionPreference = "Stop"

function Get-PeMachine([string]$Path) {
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ($bytes.Length -lt 0x40) { throw "File too small for PE: $Path" }
    if ($bytes[0] -ne 0x4D -or $bytes[1] -ne 0x5A) { throw "Missing MZ header: $Path" }
    $pe = [BitConverter]::ToInt32($bytes, 0x3C)
    if ($pe -lt 0 -or ($pe + 6) -gt $bytes.Length) { throw "Invalid PE offset: $Path" }
    if ($bytes[$pe] -ne 0x50 -or $bytes[$pe+1] -ne 0x45 -or
        $bytes[$pe+2] -ne 0 -or $bytes[$pe+3] -ne 0) {
        throw "Missing PE signature: $Path"
    }
    $machine = [BitConverter]::ToUInt16($bytes, $pe + 4)
    switch ($machine) {
        0x014c { return @{ code = "0x014c"; arch = "x86"; bits = 32 } }
        0x8664 { return @{ code = "0x8664"; arch = "x64"; bits = 64 } }
        0xAA64 { return @{ code = "0xAA64"; arch = "ARM64"; bits = 64 } }
        default { return @{ code = ("0x{0:X4}" -f $machine); arch = "UNKNOWN"; bits = 0 } }
    }
}

function Find-OpenHoldemCandidates {
    $found = @()

    # 1) Prefer the exact executable of a running OpenHoldem process.
    try {
        $found += Get-Process -ErrorAction SilentlyContinue |
            Where-Object { $_.ProcessName -like "*OpenHoldem*" } |
            ForEach-Object {
                try { $_.MainModule.FileName } catch { $null }
            } |
            Where-Object { $_ -and (Test-Path $_) }
    } catch {}

    # 2) Resolve shortcuts from Desktop and both Start Menu trees.
    $shortcutRoots = @(
        (Join-Path $env:USERPROFILE "Desktop"),
        (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"),
        (Join-Path $env:ProgramData "Microsoft\Windows\Start Menu\Programs")
    ) | Where-Object { Test-Path $_ } | Select-Object -Unique

    try {
        $shell = New-Object -ComObject WScript.Shell
        foreach ($root in $shortcutRoots) {
            Get-ChildItem -Path $root -Filter "*.lnk" -File -Recurse -ErrorAction SilentlyContinue |
                ForEach-Object {
                    try {
                        $target = $shell.CreateShortcut($_.FullName).TargetPath
                        if ($target -and
                            [IO.Path]::GetFileName($target) -like "OpenHoldem*.exe" -and
                            (Test-Path $target)) {
                            $found += $target
                        }
                    } catch {}
                }
        }
    } catch {}

    # 3) Search bounded/common installation roots only. Do not scan all of C:.
    $roots = @(
        (Join-Path $env:USERPROFILE "Desktop"),
        (Join-Path $env:USERPROFILE "Downloads"),
        (Join-Path $env:USERPROFILE "Documents"),
        $env:ProgramFiles,
        ${env:ProgramFiles(x86)},
        "C:\OpenHoldem",
        "C:\OpenHoldemBot",
        "C:\OH",
        "C:\Poker"
    ) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique

    foreach ($root in $roots) {
        try {
            $found += Get-ChildItem -Path $root -Filter "OpenHoldem*.exe" -File -Recurse -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty FullName
        } catch {}
    }

    return @($found | Sort-Object -Unique)
}

if ([string]::IsNullOrWhiteSpace($OpenHoldemExe)) {
    $candidates = @(Find-OpenHoldemCandidates)
    if ($candidates.Count -eq 0) {
        Write-Host "OPENHOLDEM_HOST_NOT_FOUND"
        Write-Host "Start the OpenHoldem instance you actually use, then rerun this inspection."
        Write-Host "The inspector now prefers the executable path of a running OpenHoldem process."
        Write-Host "You may also pass the exact path with -OpenHoldemExe 'C:\path\OpenHoldem.exe'."
        exit 2
    }
    if ($candidates.Count -gt 1) {
        Write-Host "OPENHOLDEM_HOST_MULTIPLE_CANDIDATES"
        $i = 0
        foreach ($candidate in $candidates) {
            Write-Host ("[{0}] {1}" -f $i, $candidate)
            $i++
        }
        Write-Host "Re-run with -OpenHoldemExe pointing to the instance you actually use."
        exit 2
    }
    $OpenHoldemExe = $candidates[0]
}

$OpenHoldemExe = (Resolve-Path $OpenHoldemExe).Path
if ([IO.Path]::GetFileName($OpenHoldemExe) -ne "OpenHoldem.exe") {
    throw "Expected OpenHoldem.exe, got: $OpenHoldemExe"
}

$dir = Split-Path -Parent $OpenHoldemExe
$pe = Get-PeMachine $OpenHoldemExe
$version = (Get-Item $OpenHoldemExe).VersionInfo.FileVersion
$exeSha = (Get-FileHash -Algorithm SHA256 $OpenHoldemExe).Hash.ToLowerInvariant()

$userDllPath = Join-Path $dir "user.dll"
$userDllExists = Test-Path $userDllPath
$userDllArch = $null
$userDllSha = $null
if ($userDllExists) {
    $udp = Get-PeMachine $userDllPath
    $userDllArch = $udp.arch
    $userDllSha = (Get-FileHash -Algorithm SHA256 $userDllPath).Hash.ToLowerInvariant()
}

$testedShadow = Join-Path $env:USERPROFILE "Downloads\SpinCore_LT2_Shadow.dll"
$testedShadowExists = Test-Path $testedShadow
$testedShadowArch = $null
$testedShadowSha = $null
if ($testedShadowExists) {
    $tdp = Get-PeMachine $testedShadow
    $testedShadowArch = $tdp.arch
    $testedShadowSha = (Get-FileHash -Algorithm SHA256 $testedShadow).Hash.ToLowerInvariant()
}

$expectedTestedSha = "7566be1b3c73207d437171c2b4e94f6a94477786a2a48599994a647808030062"
$testedHashMatch = $testedShadowExists -and $testedShadowSha -eq $expectedTestedSha
$architectureCompatible = $testedShadowExists -and $pe.arch -eq $testedShadowArch

$report = [ordered]@{
    schema = "SPINCORE_LT2_OPENHOLDEM_HOST_INSPECTION_V1"
    openholdem_exe = $OpenHoldemExe
    openholdem_dir = $dir
    openholdem_arch = $pe.arch
    openholdem_machine = $pe.code
    openholdem_file_version = $version
    openholdem_sha256 = $exeSha
    existing_user_dll = [ordered]@{
        exists = $userDllExists
        path = $userDllPath
        arch = $userDllArch
        sha256 = $userDllSha
    }
    tested_shadow_dll = [ordered]@{
        exists = $testedShadowExists
        path = $testedShadow
        arch = $testedShadowArch
        sha256 = $testedShadowSha
        expected_sha256 = $expectedTestedSha
        hash_match = $testedHashMatch
    }
    architecture_compatible = $architectureCompatible
    install_authorized = $false
}

$out = Join-Path $env:USERPROFILE "Downloads\SpinCore_LT2_openholdem_host_inspection.json"
$report | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 $out

Write-Host "=== SpinCore LT2 OpenHoldem host inspection ==="
Write-Host "openholdem=$OpenHoldemExe"
Write-Host "openholdem_arch=$($pe.arch) machine=$($pe.code) version=$version"
Write-Host "existing_user_dll=$userDllExists arch=$userDllArch"
Write-Host "tested_shadow_exists=$testedShadowExists arch=$testedShadowArch"
Write-Host "tested_shadow_sha256=$testedShadowSha"
Write-Host "tested_shadow_hash_match=$testedHashMatch"
Write-Host "architecture_compatible=$architectureCompatible"
Write-Host "report=$out"

if (-not $testedHashMatch) {
    Write-Host "LT2_OPENHOLDEM_HOST_INSPECTION_BLOCKED_TESTED_DLL_IDENTITY"
    exit 3
}
if (-not $architectureCompatible) {
    Write-Host "LT2_OPENHOLDEM_HOST_INSPECTION_ARCH_MISMATCH"
    exit 4
}

Write-Host "LT2_OPENHOLDEM_HOST_INSPECTION_PASS"
Write-Host "STOP HERE. Do not copy user.dll yet. Send the report and terminal output to ChatGPT."
