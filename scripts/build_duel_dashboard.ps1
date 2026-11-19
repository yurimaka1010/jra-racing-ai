param([switch]$Open,[string]$RunId='',[string]$ReuseRun='')
$ErrorActionPreference='Stop'
$taskBase='C:\Users\acbd5\AppData\Local\JraRacingAI'
$taskSecure=$null
$taskBstr=[IntPtr]::Zero
$taskExit=2
try {
    if(-not $ReuseRun){
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
    $taskBstr=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecure)
    $env:PGPASSWORD=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskBstr)
    $env:PGHOST='127.0.0.1';$env:PGPORT='5432';$env:PGDATABASE='pckeiba';$env:PGUSER='jra_audit_reader';$env:PYTHONUTF8='1'
    }
    $taskArgs=@('-B',(Join-Path $PSScriptRoot 'build_duel_dashboard.py'))
    if($RunId){$taskArgs+=@('--run-id',$RunId)}
    if($ReuseRun){$taskArgs+=@('--reuse-run',$ReuseRun)}
    if($Open){$taskArgs+='--open'}
    & (Join-Path $taskBase 'venv\Scripts\python.exe') @taskArgs
    $taskExit=$LASTEXITCODE
} catch {
    [ordered]@{status='stopped';reason='launcher_or_credentials_unavailable'} | ConvertTo-Json -Compress
} finally {
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    if($taskBstr -ne [IntPtr]::Zero){[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskBstr)}
    if($null -ne $taskSecure){$taskSecure.Dispose()}
}
exit $taskExit
