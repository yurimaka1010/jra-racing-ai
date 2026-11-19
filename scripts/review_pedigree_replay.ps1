param()
$ErrorActionPreference='Stop'
$taskBase=Join-Path $env:LOCALAPPDATA 'JraRacingAI'
$taskPointer=[IntPtr]::Zero
$taskSecure=$null
$taskExit=1
try {
    $taskCredential=Join-Path $taskBase 'trial-20261017\credentials\db-password.dpapi'
    foreach($taskPath in @($taskCredential,(Join-Path $taskBase 'pedigree-replay-review'))) {
        $taskCursor=$taskPath
        while($taskCursor) {
            if((Test-Path -LiteralPath $taskCursor) -and ((Get-Item -LiteralPath $taskCursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'unsafe_path'}
            $taskNext=Split-Path -Parent $taskCursor
            if($taskNext -eq $taskCursor){break}
            $taskCursor=$taskNext
        }
    }
    Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1') -ErrorAction Stop
    $taskSecure=(Get-Content -Raw -LiteralPath $taskCredential -Encoding ASCII).Trim()|ConvertTo-SecureString
    $taskPointer=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecure)
    $env:PGPASSWORD=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskPointer)
    $env:PGHOST='127.0.0.1';$env:PGPORT='5432';$env:PGDATABASE='pckeiba';$env:PGUSER='jra_audit_reader'
    $env:PYTHONUTF8='1';$env:PYTHONDONTWRITEBYTECODE='1'
    $env:OMP_NUM_THREADS='4';$env:OPENBLAS_NUM_THREADS='4';$env:MKL_NUM_THREADS='4'
    $taskOutput=& (Join-Path $taskBase 'venv\Scripts\python.exe') -B (Join-Path $PSScriptRoot 'review_pedigree_replay.py') collect 2>$null
    $taskExit=$LASTEXITCODE
    if($taskExit -eq 0) {
        $taskReview=$taskOutput|ConvertFrom-Json
        & (Join-Path $taskBase 'venv\Scripts\python.exe') -B (Join-Path $PSScriptRoot 'review_pedigree_replay.py') render --artifact-id $taskReview.artifact_id 2>$null
        $taskExit=$LASTEXITCODE
    } else {Write-Output $taskOutput}
} catch {
    [ordered]@{status='stopped';reason='local_replay_review_unavailable';stage='launcher'}|ConvertTo-Json -Compress
} finally {
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    if($taskPointer -ne [IntPtr]::Zero){[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskPointer)}
    if($taskSecure){$taskSecure.Dispose()}
}
exit $taskExit
