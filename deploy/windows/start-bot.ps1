$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$logPath = Join-Path $repoRoot "logs\windows-live.log"

if (-not (Test-Path $python)) {
    throw "Virtualenv Python not found at $python"
}

$existing = Get-CimInstance Win32_Process |
    Where-Object { $_.Name -match "python" -and $_.CommandLine -match "trade_bot.main live" }
if ($existing) {
    Write-Output "Bot is already running."
    $existing | Select-Object ProcessId, CommandLine
    exit 0
}

New-Item -ItemType Directory -Force (Join-Path $repoRoot "logs") | Out-Null

Start-Process -WindowStyle Minimized -FilePath "cmd.exe" -ArgumentList "/c `"$python -m trade_bot.main live >> `"$logPath`" 2>&1`""
Start-Sleep -Seconds 2

$started = Get-CimInstance Win32_Process |
    Where-Object { $_.Name -match "python" -and $_.CommandLine -match "trade_bot.main live" }

if (-not $started) {
    throw "Bot failed to stay running."
}

$started | Select-Object ProcessId, CommandLine
