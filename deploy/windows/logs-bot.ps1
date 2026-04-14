$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$logPath = Join-Path $repoRoot "logs\windows-live.log"

if (-not (Test-Path $logPath)) {
    Write-Output "No windows-live.log found yet."
    exit 0
}

Get-Content $logPath -Tail 80
