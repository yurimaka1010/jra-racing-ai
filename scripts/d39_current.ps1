param([ValidateSet('init','collect','build','train','report','preflight','record','evaluate','prepare-control','train-control',
      'verify-four','preflight-four','record-four','evaluate-four','prepare-addendum','diagnose-control','dry-run-four','verify-addendum')][string]$Mode='preflight',
      [string]$RunId='',[string]$ControlId='',[switch]$Once,[string]$RaceDate='')
$ErrorActionPreference='Stop'
$taskBase=Join-Path $env:LOCALAPPDATA 'JraRacingAI'
$taskPointer=[IntPtr]::Zero
$taskSecure=$null
$taskExit=2
try {
    if((Get-TimeZone).Id -ne 'Tokyo Standard Time'){throw 'timezone_mismatch'}
    $taskArguments=@('-B',(Join-Path $PSScriptRoot 'd39_current.py'),$Mode)
    if($RunId){$taskArguments+=@('--run-id',$RunId)}
    if($ControlId){$taskArguments+=@('--control-id',$ControlId)}
    if($Once){$taskArguments+='--once'}
    if($RaceDate){$taskArguments+=@('--date',$RaceDate)}
    if($Mode -in @('init','collect','record','evaluate','record-four','evaluate-four','dry-run-four')){
        $taskCredential=Join-Path $taskBase 'trial-20261017\credentials\db-password.dpapi'
        $taskCursor=[IO.Path]::GetFullPath($taskCredential)
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
    [ordered]@{status='stopped';reason='local_launcher_or_credentials_unavailable';stage=$Mode} | ConvertTo-Json -Compress
} finally {
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    if($taskPointer -ne [IntPtr]::Zero){[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskPointer)}
    if($null -ne $taskSecure){$taskSecure.Dispose()}
}
exit $taskExit
