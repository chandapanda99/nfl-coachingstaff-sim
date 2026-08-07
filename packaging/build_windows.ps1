[CmdletBinding()]
param(
    [switch]$SkipInstall,
    [switch]$SkipChecks
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$frontendRoot = Join-Path $projectRoot "frontend"
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$progressActivity = "Building NFL Virtual Coaching Staff"
$buildStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$progressForegroundProperty = $Host.PrivateData.PSObject.Properties["ProgressForegroundColor"]
$progressBackgroundProperty = $Host.PrivateData.PSObject.Properties["ProgressBackgroundColor"]
$originalProgressForeground = if ($progressForegroundProperty) { $Host.PrivateData.ProgressForegroundColor } else { $null }
$originalProgressBackground = if ($progressBackgroundProperty) { $Host.PrivateData.ProgressBackgroundColor } else { $null }
$psStyleVariable = Get-Variable -Name PSStyle -ErrorAction SilentlyContinue
$originalProgressStyle = if ($psStyleVariable) { $PSStyle.Progress.Style } else { $null }
$originalProgressView = if ($psStyleVariable) { $PSStyle.Progress.View } else { $null }

# Use cyan build information over a dark-blue fill. Classic view keeps the actual
# bar visible while native build tools write their own output below it.
if ($progressForegroundProperty) {
    $Host.PrivateData.ProgressForegroundColor = "Cyan"
}
if ($progressBackgroundProperty) {
    $Host.PrivateData.ProgressBackgroundColor = "DarkBlue"
}
if ($psStyleVariable) {
    $PSStyle.Progress.Style = "$([char]27)[38;2;102;204;255m"
    $PSStyle.Progress.View = "Classic"
}

function Set-BuildProgress {
    param(
        [Parameter(Mandatory)] [string]$Status,
        [Parameter(Mandatory)] [int]$PercentComplete,
        [string]$CurrentOperation = ""
    )

    $elapsed = $buildStopwatch.Elapsed.ToString("hh\:mm\:ss")
    $progressParameters = @{
        Id              = 1
        Activity        = $progressActivity
        Status          = "$Status | Elapsed $elapsed"
        PercentComplete = $PercentComplete
    }
    if ($CurrentOperation) {
        $progressParameters.CurrentOperation = $CurrentOperation
    }
    Write-Progress @progressParameters
}

function Invoke-ProjectCommand {
    param(
        [Parameter(Mandatory)] [string]$Command,
        [Parameter(Mandatory)] [string[]]$Arguments,
        [Parameter(Mandatory)] [string]$WorkingDirectory
    )

    Push-Location $WorkingDirectory
    try {
        & $Command @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $Command $($Arguments -join ' ')"
        }
    }
    finally {
        Pop-Location
    }
}

$buildSteps = @(
    @{
        Name = "Checking build prerequisites"
        Action = {
            foreach ($commandName in @("uv", "node", "npm", "cargo")) {
                if (-not (Get-Command $commandName -ErrorAction SilentlyContinue)) {
                    throw "Required packaging command '$commandName' was not found on PATH."
                }
            }

            if (-not (Test-Path $vswhere)) {
                throw "Visual Studio Build Tools were not found. Install Desktop development with C++ and the Windows SDK."
            }
        }
    }
    @{
        Name = "Loading the Visual Studio C++ toolchain"
        Action = {
            $visualStudioRoot = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
            if (-not $visualStudioRoot) {
                throw "Visual Studio C++ build tools were not found. Install Desktop development with C++ and the Windows SDK."
            }

            $developerCommand = Join-Path $visualStudioRoot "Common7\Tools\VsDevCmd.bat"
            $environmentLines = & cmd.exe /s /c "`"$developerCommand`" -no_logo -arch=x64 -host_arch=x64 && set"
            foreach ($line in $environmentLines) {
                if ($line -match "^([^=]+)=(.*)$") {
                    [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
                }
            }
        }
    }
)

if (-not $SkipInstall) {
    $buildSteps += @{
        Name = "Restoring Python dependencies"
        Action = { Invoke-ProjectCommand "uv" @("sync", "--frozen", "--extra", "package", "--extra", "test") $projectRoot }
    }
    $buildSteps += @{
        Name = "Restoring frontend dependencies"
        Action = { Invoke-ProjectCommand "npm" @("ci") $frontendRoot }
    }
}

if (-not $SkipChecks) {
    $buildSteps += @{
        Name = "Running the critical Python tests"
        Action = { Invoke-ProjectCommand "uv" @("run", "--frozen", "--extra", "test", "pytest") $projectRoot }
    }
    $buildSteps += @{
        Name = "Checking the Svelte frontend"
        Action = { Invoke-ProjectCommand "npm" @("run", "check") $frontendRoot }
    }
}

$iconPath = Join-Path $frontendRoot "src-tauri\icons\icon.ico"
if (-not (Test-Path $iconPath)) {
    $buildSteps += @{
        Name = "Generating desktop application icons"
        Action = { Invoke-ProjectCommand "npm" @("run", "tauri", "icon", "../packaging/app-icon.svg") $frontendRoot }
    }
}

$buildSteps += @{
    Name = "Packaging the Python coaching sidecar"
    Action = { Invoke-ProjectCommand "uv" @("run", "--frozen", "--extra", "package", "python", "packaging/build_sidecar.py") $projectRoot }
}
$buildSteps += @{
    Name = "Building the Tauri MSI and NSIS installers"
    Action = { Invoke-ProjectCommand "npm" @("run", "tauri", "build") $frontendRoot }
}

try {
    $stepCount = $buildSteps.Count
    for ($index = 0; $index -lt $stepCount; $index++) {
        $stepNumber = $index + 1
        $step = $buildSteps[$index]
        $startingPercent = [Math]::Floor(($index / $stepCount) * 100)
        $status = "Step $stepNumber of ${stepCount}: $($step.Name)"
        Set-BuildProgress -Status $status -PercentComplete $startingPercent -CurrentOperation $step.Name
        Write-Host "`n==> [$stepNumber/$stepCount] $($step.Name)" -ForegroundColor Cyan
        & $step.Action
        $completedPercent = [Math]::Floor(($stepNumber / $stepCount) * 100)
        Set-BuildProgress -Status $status -PercentComplete $completedPercent -CurrentOperation "Completed"
    }

    Set-BuildProgress -Status "Build complete" -PercentComplete 100 -CurrentOperation "Installers are ready"
}
finally {
    $buildStopwatch.Stop()
    Write-Progress -Id 1 -Activity $progressActivity -Completed
    if ($progressForegroundProperty) {
        $Host.PrivateData.ProgressForegroundColor = $originalProgressForeground
    }
    if ($progressBackgroundProperty) {
        $Host.PrivateData.ProgressBackgroundColor = $originalProgressBackground
    }
    if ($psStyleVariable) {
        $PSStyle.Progress.Style = $originalProgressStyle
        $PSStyle.Progress.View = $originalProgressView
    }
}

$bundleRoot = Join-Path $frontendRoot "src-tauri\target\release\bundle"
Write-Host "`nWindows packages are ready under: $bundleRoot" -ForegroundColor Green
Write-Host "Total build time: $($buildStopwatch.Elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Green
