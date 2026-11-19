param([ValidateSet('init','preflight','prepare-survey','probe','survey','record','report')][string]$Mode='preflight',[string]$RunId='',
      [string]$StartDate='2026-10-24',[switch]$Once)
$ErrorActionPreference='Stop'
$taskBase='C:\Users\acbd5\AppData\Local\JraRacingAI'
$taskPointer=[IntPtr]::Zero
$taskSecure=$null
$taskExit=2
try {
    if((Get-TimeZone).Id -ne 'Tokyo Standard Time'){throw 'timezone_mismatch'}
    $taskArguments=@('-B',(Join-Path $PSScriptRoot 'stage7_local.py'),$Mode)
    if($RunId){$taskArguments+=@('--run-id',$RunId)}
    if($Mode -eq 'init'){$taskArguments+=@('--start-date',$StartDate)}
    if($Once){$taskArguments+='--once'}
    if($Mode -in @('probe','survey','record')){
        $taskCredential=Join-Path $taskBase 'trial-20261017\credentials\db-password.dpapi'
        $taskCursor=[IO.Path]::GetFullPath($taskCredential)
        if($taskCursor -notlike 'C:\Users\acbd5\AppData\Local\JraRacingAI\*'){throw 'unsafe_path'}
        while($taskCursor){
            if((Test-Path -LiteralPath $taskCursor) -and ((Get-Item -LiteralPath $taskCursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'unsafe_path'}
            $taskParent=Split-Path -Parent $taskCursor
            if($taskParent -eq $taskCursor){break}
            $taskCursor=$taskParent
        }
        $taskSecure=(Get-Content -LiteralPath $taskCredential -Raw -Encoding ASCII).Trim() | ConvertTo-SecureString
        $taskPointer=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecure)
        $env:PGPASSWORD=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskPointer)
        $env:PGHOST='127.0.0.1';$env:PGPORT='5432';$env:PGDATABASE='pckeiba';$env:PGUSER='jra_audit_reader'
    }
    $env:PYTHONUTF8='1'
    & (Join-Path $taskBase 'venv\Scripts\python.exe') @taskArguments
    $taskExit=$LASTEXITCODE
} catch {
    [ordered]@{status='stopped';reason='stage7_launcher_or_credentials_unavailable';stage=$Mode} | ConvertTo-Json -Compress
} finally {
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    if($taskPointer -ne [IntPtr]::Zero){[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskPointer)}
    if($null -ne $taskSecure){$taskSecure.Dispose()}
}
exit $taskExit
