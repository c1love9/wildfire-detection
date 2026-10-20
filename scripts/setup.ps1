$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $projectPython)) {
        & py -3.11 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Creating the Python 3.11 environment failed.' }
    }
    $requirementsPath = Join-Path $projectRoot 'requirements-lock.txt'
    if (-not (Test-Path -LiteralPath $requirementsPath)) {
        $requirementsPath = Join-Path $projectRoot 'requirements.txt'
    }
    & $projectPython -m pip install -r $requirementsPath
    if ($LASTEXITCODE -ne 0) { throw 'Installing project dependencies failed.' }
    & $projectPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Installed dependencies have conflicts.' }
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -and -not $env:FIRE_ALARM_FFMPEG) {
        Write-Warning 'Alarm video clips also require FFmpeg with libx264 on PATH. See docs\recording.md. Detection and snapshots can run without it.'
    }
    Write-Host 'Environment ready. Run scripts\start.ps1 to open the original demo.'
}
finally {
    Pop-Location
}
