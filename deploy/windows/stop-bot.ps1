$ErrorActionPreference = "Stop"

$processes = Get-CimInstance Win32_Process |
    Where-Object { $_.Name -match "python" -and $_.CommandLine -match "trade_bot.main live" }

if (-not $processes) {
    Write-Output "Bot is not running."
    exit 0
}

$processes | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Write-Output "Stopped bot processes."
