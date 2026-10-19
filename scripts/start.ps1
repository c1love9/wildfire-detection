param([int]$Port = 8501)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Project environment missing. Run scripts\setup.ps1 first.'
}
$entryPoint = (Get-ChildItem -LiteralPath $projectRoot -Filter '1_*_Home.py' -File).FullName
if (-not $entryPoint) { throw 'The upstream Streamlit entry point was not found.' }
Push-Location $projectRoot
try {
    & $projectPython -m streamlit run $entryPoint --server.address 127.0.0.1 --server.port $Port --server.headless true
    if ($LASTEXITCODE -ne 0) { throw 'Streamlit exited with an error.' }
}
finally {
    Pop-Location
}
