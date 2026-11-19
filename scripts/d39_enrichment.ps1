param([ValidateSet('inventory','prepare','train','record','preview','observe')][string]$Mode='record',
      [string]$RunId='',[switch]$Once)
$ErrorActionPreference='Stop'
$taskBase='C:\Users\acbd5\AppData\Local\JraRacingAI'
$taskPointer=[IntPtr]::Zero
$taskSecure=$null
$taskExit=1
$taskStage='clock_check'
try {
    # Background PowerShell may inherit a different PSModulePath from the host.
    # Load the existing Windows encryption cmdlets explicitly; no installation.
    $taskStage='builtin_security_module'
    Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1') -ErrorAction Stop
    $taskStage='clock_check'
    if((Get-TimeZone).Id -ne 'Tokyo Standard Time'){throw 'timezone_mismatch'}
    $taskCredential=Join-Path $taskBase 'trial-20261017\credentials\db-password.dpapi'
    $taskStage='credential_path_check'
    $taskCursor=$taskCredential
    while($taskCursor){
        if((Test-Path -LiteralPath $taskCursor) -and ((Get-Item -LiteralPath $taskCursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'unsafe_path'}
        $taskParent=Split-Path -Parent $taskCursor
        if($taskParent -eq $taskCursor){break}
        $taskCursor=$taskParent
    }
    $taskStage='credential_read'
    $taskSecure=(Get-Content -Raw -LiteralPath $taskCredential -Encoding ASCII).Trim()|ConvertTo-SecureString
    $taskStage='credential_decrypt'
    $taskPointer=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecure)
    $env:PGPASSWORD=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskPointer)
    $env:PGHOST='127.0.0.1';$env:PGPORT='5432';$env:PGDATABASE='pckeiba';$env:PGUSER='jra_audit_reader';$env:PYTHONUTF8='1'
    $taskArgs=@('-B',(Join-Path $PSScriptRoot 'd39_enrichment.py'),$Mode,'--run-id',$RunId)
    if($Once){$taskArgs+='--once'}
    $taskStage='python_start'
    & (Join-Path $taskBase 'venv\Scripts\python.exe') @taskArgs
    $taskExit=$LASTEXITCODE
} catch {
    [ordered]@{status='stopped';reason='local_launcher_or_credentials_unavailable';stage=$taskStage;error_type=$_.Exception.GetType().Name}|ConvertTo-Json -Compress
} finally {
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    if($taskPointer -ne [IntPtr]::Zero){[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskPointer)}
    if($taskSecure){$taskSecure.Dispose()}
}
exit $taskExit

