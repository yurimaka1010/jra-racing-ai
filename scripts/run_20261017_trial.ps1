param([switch]$Preflight, [switch]$Probe)
$ErrorActionPreference = 'Stop'
$taskRoot = Join-Path $env:LOCALAPPDATA 'JraRacingAI\trial-20261017'
$taskPython = Join-Path $env:LOCALAPPDATA 'JraRacingAI\venv\Scripts\python.exe'
$taskCredential = Join-Path $taskRoot 'credentials\db-password.dpapi'
$taskSecure = $null
$taskBstr = [IntPtr]::Zero
$taskReason = 'launcher_failed'

function Test-TaskLocalPath([string]$Path) {
    $taskFull = [IO.Path]::GetFullPath($Path)
    if ($taskFull -notlike 'C:\Users\acbd5\AppData\Local\JraRacingAI\*' -or $taskFull -match '\.\.' -or $taskFull -match 'OneDrive|Dropbox|Google Drive') { throw 'unsafe_path' }
    $taskCursor = $taskFull
    while ($taskCursor) {
        if (Test-Path -LiteralPath $taskCursor) {
            if ((Get-Item -LiteralPath $taskCursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'unsafe_path' }
        }
        $taskParent = Split-Path -Parent $taskCursor
        if ($taskParent -eq $taskCursor) { break }
        $taskCursor = $taskParent
    }
}

try {
    Test-TaskLocalPath $taskRoot
    Test-TaskLocalPath $taskCredential
    if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) { $taskReason = 'python_missing'; throw 'stop' }
    $taskArgs = @('-B', (Join-Path $PSScriptRoot 'run_20261017_trial.py'))
    if ($Preflight) {
        $taskArgs += '--preflight'
    } else {
        $taskReason = 'credentials_unavailable'
        if (-not (Test-Path -LiteralPath $taskCredential -PathType Leaf)) { throw 'stop' }
        $taskSecure = (Get-Content -LiteralPath $taskCredential -Raw -Encoding ASCII).Trim() | ConvertTo-SecureString
        $taskBstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecure)
        $env:PGPASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskBstr)
        if ([string]::IsNullOrEmpty($env:PGPASSWORD)) { throw 'stop' }
        if ($Probe) { $taskArgs += '--probe' }
    }
    $env:PGHOST = '127.0.0.1'
    $env:PGPORT = '5432'
    $env:PGDATABASE = 'pckeiba'
    $env:PGUSER = 'jra_audit_reader'
    $env:PYTHONUTF8 = '1'
    $taskReason = 'python_launch_failed'
    & $taskPython @taskArgs
    $taskExit = $LASTEXITCODE
} catch {
    $taskExit = 2
    $taskFailure = [ordered]@{status='stopped';reason=$taskReason;stage='powershell_launcher';finished_at_utc=[DateTime]::UtcNow.ToString('o')}
    try {
        Test-TaskLocalPath $taskRoot
        [IO.Directory]::CreateDirectory($taskRoot) | Out-Null
        $taskFailurePath = Join-Path $taskRoot ('launcher-failure-' + [guid]::NewGuid().ToString('N') + '.json')
        [IO.File]::WriteAllText($taskFailurePath, ($taskFailure | ConvertTo-Json -Compress), [Text.UTF8Encoding]::new($false))
    } catch { }
    $taskFailure | ConvertTo-Json -Compress
} finally {
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    if ($taskBstr -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskBstr) }
    if ($null -ne $taskSecure) { $taskSecure.Dispose() }
}
exit $taskExit
