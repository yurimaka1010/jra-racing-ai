param([switch]$Preflight)
$ErrorActionPreference = 'Stop'
$taskBase = 'C:\Users\acbd5\AppData\Local\JraRacingAI'
$taskPython = Join-Path $taskBase 'venv\Scripts\python.exe'
$taskCredential = Join-Path $taskBase 'trial-20261017\credentials\db-password.dpapi'
$taskSecure = $null
$taskBstr = [IntPtr]::Zero
$taskExit = 2
try {
    $taskArgs = @('-B', (Join-Path $PSScriptRoot 'predict_20261018_step2.py'))
    if ($Preflight) {
        $taskArgs += '--preflight'
    } else {
        # Reuse the existing current-user DPAPI credential without displaying it.
        $taskFull = [IO.Path]::GetFullPath($taskCredential)
        if ($taskFull -notlike 'C:\Users\acbd5\AppData\Local\JraRacingAI\*' -or $taskFull -match 'OneDrive|Dropbox|Google Drive') { throw 'unsafe_path' }
        $taskCursor = $taskFull
        while ($taskCursor) {
            if ((Test-Path -LiteralPath $taskCursor) -and ((Get-Item -LiteralPath $taskCursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'unsafe_path' }
            $taskParent = Split-Path -Parent $taskCursor
            if ($taskParent -eq $taskCursor) { break }
            $taskCursor = $taskParent
        }
        $taskSecure = (Get-Content -LiteralPath $taskCredential -Raw -Encoding ASCII).Trim() | ConvertTo-SecureString
        $taskBstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecure)
        $env:PGPASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskBstr)
        if ([string]::IsNullOrEmpty($env:PGPASSWORD)) { throw 'credentials_unavailable' }
    }
    $env:PGHOST = '127.0.0.1'
    $env:PGPORT = '5432'
    $env:PGDATABASE = 'pckeiba'
    $env:PGUSER = 'jra_audit_reader'
    $env:PYTHONUTF8 = '1'
    & $taskPython @taskArgs
    $taskExit = $LASTEXITCODE
} catch {
    [ordered]@{status='stopped';reason='launcher_or_credentials_unavailable';stage='powershell_launcher'} | ConvertTo-Json -Compress
} finally {
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    if ($taskBstr -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskBstr) }
    if ($null -ne $taskSecure) { $taskSecure.Dispose() }
}
exit $taskExit
