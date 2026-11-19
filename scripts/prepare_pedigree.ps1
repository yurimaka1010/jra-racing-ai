param([ValidateSet('inventory','prepare','verify','report')][string]$Mode='prepare',[string]$RunId='')
$ErrorActionPreference='Stop'
$taskBase=Join-Path $env:LOCALAPPDATA 'JraRacingAI'
$taskPointer=[IntPtr]::Zero
$taskSecure=$null
$taskExit=1
try {
    if($Mode -in @('inventory','prepare')) {
        Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1') -ErrorAction Stop
        $taskCredential=Join-Path $taskBase 'trial-20261017\credentials\db-password.dpapi'
        $taskCursor=$taskCredential
        while($taskCursor) {
            if((Test-Path -LiteralPath $taskCursor) -and ((Get-Item -LiteralPath $taskCursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'unsafe_path'}
            $taskParent=Split-Path -Parent $taskCursor
            if($taskParent -eq $taskCursor){break}
            $taskCursor=$taskParent
        }
        $taskSecure=(Get-Content -Raw -LiteralPath $taskCredential -Encoding ASCII).Trim()|ConvertTo-SecureString
        $taskPointer=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecure)
        $env:PGPASSWORD=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskPointer)
    }
    $env:PGHOST='127.0.0.1';$env:PGPORT='5432';$env:PGDATABASE='pckeiba';$env:PGUSER='jra_audit_reader';$env:PYTHONUTF8='1'
    $taskArgs=@('-B',(Join-Path $PSScriptRoot 'prepare_pedigree.py'),$Mode)
    if($RunId){$taskArgs+=@('--run-id',$RunId)}
    & (Join-Path $taskBase 'venv\Scripts\python.exe') @taskArgs
    $taskExit=$LASTEXITCODE
} catch {
    [ordered]@{status='stopped';reason='local_launcher_or_credentials_unavailable'}|ConvertTo-Json -Compress
} finally {
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    if($taskPointer -ne [IntPtr]::Zero){[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskPointer)}
    if($taskSecure){$taskSecure.Dispose()}
}
exit $taskExit
