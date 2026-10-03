function Invoke-StartupFixture {
    param(
        [string]$PreflightOutput,
        [int]$PreflightExitCode
    )

    $fixtureRoot = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
    $scripts = Join-Path $fixtureRoot 'scripts'
    $bin = Join-Path $fixtureRoot 'bin'
    New-Item -ItemType Directory -Path $scripts, $bin -Force | Out-Null
    Copy-Item (Join-Path $PSScriptRoot '..\scripts\start_satquery.ps1') (Join-Path $scripts 'start_satquery.ps1')

    $preflight = @"
Write-Output '$PreflightOutput'
`$global:LASTEXITCODE = $PreflightExitCode
"@
    Set-Content -Path (Join-Path $scripts 'preflight_satquery.ps1') -Value $preflight -Encoding ASCII

    $pythonStub = @'
@echo off
if "%1"=="-m" (
  echo API launched > "%SATQUERY_API_MARKER%"
  exit /b 0
)
exit /b 64
'@
    Set-Content -Path (Join-Path $bin 'python.cmd') -Value $pythonStub -Encoding ASCII

    $marker = Join-Path $fixtureRoot 'api-launched.txt'
    $startScript = Join-Path $scripts 'start_satquery.ps1'
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = (Get-Command powershell.exe).Source
    $startInfo.Arguments = '-NoProfile -ExecutionPolicy Bypass -File "{0}" -Port 8000' -f $startScript
    $startInfo.WorkingDirectory = $fixtureRoot
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.EnvironmentVariables['PATH'] = "$bin;$env:PATH"
    $startInfo.EnvironmentVariables['SATQUERY_API_MARKER'] = $marker

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    [void]$process.Start()
    $stdout = $process.StandardOutput.ReadToEnd()
    $stderr = $process.StandardError.ReadToEnd()
    $process.WaitForExit()

    return @{
        ExitCode = $process.ExitCode
        Output = "$stdout`n$stderr"
        ApiLaunched = Test-Path $marker
    }
}

Describe 'start_satquery preflight gate' {
    It 'launches the API only after a passing preflight' {
        $result = Invoke-StartupFixture -PreflightOutput '{"status":"READY"}' -PreflightExitCode 0

        $result.ExitCode | Should Be 0
        $result.ApiLaunched | Should Be $true
    }

    It 'aborts API startup when preflight reports BLOCKED' {
        $result = Invoke-StartupFixture -PreflightOutput '{"status":"BLOCKED"}' -PreflightExitCode 0

        $result.ExitCode | Should Not Be 0
        $result.ApiLaunched | Should Be $false
        $result.Output | Should Match 'STARTUP ABORTED'
    }

    It 'aborts API startup when preflight exits with an error' {
        $result = Invoke-StartupFixture -PreflightOutput '{"status":"READY"}' -PreflightExitCode 23

        $result.ExitCode | Should Not Be 0
        $result.ApiLaunched | Should Be $false
        $result.Output | Should Match 'STARTUP ABORTED'
    }
}
