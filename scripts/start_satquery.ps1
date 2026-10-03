param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$preflightOutput = & "$PSScriptRoot\preflight_satquery.ps1" 2>&1
$preflightExitCode = $LASTEXITCODE
$preflightText = $preflightOutput | Out-String
$preflightOutput | ForEach-Object { Write-Output $_ }
if ($preflightExitCode -ne 0 -or $preflightText -notmatch '"status"\s*:\s*"(READY|READY_WITH_WARNINGS)"') {
    Write-Host 'STARTUP ABORTED: SatQuery preflight did not pass.' -ForegroundColor Red
    exit 1
}
Write-Host "SatQuery starting at http://127.0.0.1:$Port" -ForegroundColor Green
python -m src.api --host 127.0.0.1 --port $Port
