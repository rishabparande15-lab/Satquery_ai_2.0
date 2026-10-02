param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
& "$PSScriptRoot\preflight_satquery.ps1"
Write-Host "SatQuery starting at http://127.0.0.1:$Port" -ForegroundColor Green
python -m src.api --host 127.0.0.1 --port $Port
