param([Parameter(Mandatory=$true)][ValidatePattern('^[a-f0-9]{32}$')][string]$RunId,
      [ValidateSet('record','preflight','ops')][string]$Mode='record',[switch]$Once)
$ErrorActionPreference='Stop'
$taskBase=Join-Path $env:LOCALAPPDATA 'JraRacingAI'
$taskPointer=[IntPtr]::Zero
$taskSecure=$null
$taskExit=1
$taskSafe=$false
$taskStage='path_check'
$taskRun=Join-Path $taskBase ('pedigree-forward\evaluations\'+$RunId)
try {
    if((Get-TimeZone).Id -ne 'Tokyo Standard Time'){throw 'timezone_mismatch'}
    $taskCredential=Join-Path $taskBase 'trial-20261017\credentials\db-password.dpapi'
    foreach($taskPath in @($taskRun,$taskCredential)) {
        $taskCursor=$taskPath
        while($taskCursor) {
            if((Test-Path -LiteralPath $taskCursor) -and ((Get-Item -LiteralPath $taskCursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'unsafe_path'}
            $taskNext=Split-Path -Parent $taskCursor
            if($taskNext -eq $taskCursor){break}
            $taskCursor=$taskNext
        }
    }
    $taskSafe=$true
    if($Mode -ne 'ops') {
        $taskStage='credentials'
        Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1') -ErrorAction Stop
        $taskSecure=(Get-Content -Raw -LiteralPath $taskCredential -Encoding ASCII).Trim()|ConvertTo-SecureString
        $taskPointer=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecure)
        $env:PGPASSWORD=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskPointer)
    }
    $env:PGHOST='127.0.0.1';$env:PGPORT='5432';$env:PGDATABASE='pckeiba';$env:PGUSER='jra_audit_reader'
    $env:PYTHONUTF8='1';$env:PYTHONDONTWRITEBYTECODE='1'
    $env:OMP_NUM_THREADS='4';$env:OPENBLAS_NUM_THREADS='4';$env:MKL_NUM_THREADS='4';$env:NUMEXPR_NUM_THREADS='4'
    $taskStage='python_start'
    $taskArgs=@('-B',(Join-Path $PSScriptRoot 'pedigree_forward.py'),$Mode,'--run-id',$RunId)
    if($Once){$taskArgs+='--once'}
    & (Join-Path $taskBase 'venv\Scripts\python.exe') @taskArgs 2>$null
    $taskExit=$LASTEXITCODE
    $taskLog=Join-Path $taskRun 'launcher-logs';[IO.Directory]::CreateDirectory($taskLog)|Out-Null
    $taskData=[ordered]@{finished_at=(Get-Date).ToString('o');mode=$Mode;exit_code=$taskExit;new_fits=0}
    [IO.File]::WriteAllText((Join-Path $taskLog ([guid]::NewGuid().ToString('N')+'.json')),($taskData|ConvertTo-Json -Compress),[Text.UTF8Encoding]::new($false))
} catch {
    if($taskSafe){try {
        $taskLog=Join-Path $taskRun 'launcher-logs';[IO.Directory]::CreateDirectory($taskLog)|Out-Null
        $taskData=[ordered]@{at=(Get-Date).ToString('o');status='stopped';stage=$taskStage;reason='launcher_unavailable';new_fits=0}
        [IO.File]::WriteAllText((Join-Path $taskLog ([guid]::NewGuid().ToString('N')+'.json')),($taskData|ConvertTo-Json -Compress),[Text.UTF8Encoding]::new($false))
    } catch {}}
    [ordered]@{status='stopped';reason='pedigree_forward_launcher_unavailable';stage=$taskStage;new_fits=0}|ConvertTo-Json -Compress
} finally {
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    if($taskPointer -ne [IntPtr]::Zero){[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskPointer)}
    if($taskSecure){$taskSecure.Dispose()}
}
exit $taskExit
