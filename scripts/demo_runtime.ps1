param([ValidateSet('check', 'start', 'stop')][string]$Action = 'start')
$ErrorActionPreference = 'Stop'
$bundleRoot = Split-Path -Parent $PSScriptRoot
$recordPath = Join-Path $bundleRoot '.runtime\machine.json'
$record = $null
if (Test-Path -LiteralPath $recordPath) {
    $record = Get-Content -LiteralPath $recordPath -Raw -Encoding UTF8 | ConvertFrom-Json
}
$candidates = @($env:FIRE_DEMO_PYTHON, (Join-Path $bundleRoot '.venv\Scripts\python.exe'))
if ($record) { $candidates += $record.python_executable }
$runtimePython = $candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } | Select-Object -First 1
if (-not $runtimePython) {
    Write-Host 'Python environment unavailable. Keep the original project .venv or run scripts\setup.ps1.' -ForegroundColor Red
    exit 1
}
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = $bundleRoot
if ($record -and $record.ffmpeg_executable -and (Test-Path -LiteralPath $record.ffmpeg_executable)) {
    $env:FIRE_ALARM_FFMPEG = $record.ffmpeg_executable
}
Push-Location $bundleRoot
try {
    $arguments = @((Join-Path $PSScriptRoot 'demo_runtime.py'), $Action)
    if ($Action -eq 'check') { $arguments += '--deep' }
    & $runtimePython @arguments
    exit $LASTEXITCODE
}
finally { Pop-Location }
