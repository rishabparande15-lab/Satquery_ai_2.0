param()
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$result = python -c "import json; from src.runtime_preflight import run_preflight; print(json.dumps(run_preflight(), indent=2))"
$result
if ($LASTEXITCODE -ne 0 -or $result -match '"status": "BLOCKED"') { Write-Host 'PREFLIGHT BLOCKED — correct the listed configuration or asset errors.' -ForegroundColor Red; exit 1 }
Write-Host 'PREFLIGHT READY' -ForegroundColor Green
