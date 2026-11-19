param(
    [string]$Start = '2025-01-01',
    [string]$End = '2025-01-31',
    [ValidateRange(1, 50000)][int]$MaxRows = 10000,
    [ValidateRange(5, 1000)][int]$Rounds = 100
)

# Authentication stays in Python's hidden local prompt. No transcript is started.
$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $env:LOCALAPPDATA 'JraRacingAI\venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    throw 'local_python_missing'
}
$env:PYTHONPATH = Join-Path $taskProject 'src'
$env:PGHOST = '127.0.0.1'
$env:PGPORT = '5432'
$env:PGDATABASE = 'pckeiba'
$env:PGUSER = 'jra_audit_reader'
& $taskPython -B -m jra_racing_ai.ml_cli first-run --execute-local --start $Start --end $End --max-rows $MaxRows --rounds $Rounds
if ($LASTEXITCODE -ne 0) {
    throw 'v01_stopped_see_fixed_reason_code'
}
