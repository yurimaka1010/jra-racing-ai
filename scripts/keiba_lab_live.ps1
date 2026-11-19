param([switch]$Serve,[switch]$NoBrowser)
$ErrorActionPreference='Stop'
$taskBase='C:\Users\acbd5\AppData\Local\JraRacingAI'
$taskRoot=Join-Path $taskBase 'keiba-lab-live'
$taskUrl='http://127.0.0.1:8767'
$taskPointer=[IntPtr]::Zero
$taskSecure=$null
try {
    Import-Module (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Security\Microsoft.PowerShell.Security.psd1') -ErrorAction Stop
    if((Get-TimeZone).Id -ne 'Tokyo Standard Time'){throw 'timezone_mismatch'}
    $taskCursor=$taskRoot
    while($taskCursor){
        if((Test-Path -LiteralPath $taskCursor) -and ((Get-Item -LiteralPath $taskCursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'unsafe_path'}
        $taskNext=Split-Path -Parent $taskCursor
        if($taskNext -eq $taskCursor){break}
        $taskCursor=$taskNext
    }
    [IO.Directory]::CreateDirectory($taskRoot) | Out-Null
    if(-not $Serve){
        $taskHealthy=$false
        try {$taskHealth=Invoke-RestMethod -Uri ($taskUrl+'/api/health') -TimeoutSec 2; $taskHealthy=$taskHealth.version -eq 'keiba-lab-live.20261024.1'} catch {}
        if(-not $taskHealthy){
            $taskShell=Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
            $taskScript=$PSCommandPath.Replace("'","''")
            $taskCommand="& '$taskScript' -Serve"
            $taskEncoded=[Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($taskCommand))
            Start-Process -FilePath $taskShell -ArgumentList @('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-EncodedCommand',$taskEncoded) -WindowStyle Hidden | Out-Null
            for($taskAttempt=0;$taskAttempt -lt 20;$taskAttempt++){
                Start-Sleep -Milliseconds 500
                try {$taskHealth=Invoke-RestMethod -Uri ($taskUrl+'/api/health') -TimeoutSec 2; if($taskHealth.version -eq 'keiba-lab-live.20261024.1'){$taskHealthy=$true;break}} catch {}
            }
        }
        if(-not $taskHealthy){throw 'viewer_unavailable'}
        if(-not $NoBrowser){Start-Process $taskUrl | Out-Null}
        [ordered]@{status='ready';url=$taskUrl;H_results_included=$false;task_changes=0} | ConvertTo-Json -Compress
        exit 0
    }
    $taskCredential=Join-Path $taskBase 'trial-20261017\credentials\db-password.dpapi'
    $taskCursor=$taskCredential
    while($taskCursor){
        if((Test-Path -LiteralPath $taskCursor) -and ((Get-Item -LiteralPath $taskCursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'unsafe_path'}
        $taskNext=Split-Path -Parent $taskCursor
        if($taskNext -eq $taskCursor){break}
        $taskCursor=$taskNext
    }
    if(Test-Path -LiteralPath $taskCredential){
        $taskSecure=(Get-Content -LiteralPath $taskCredential -Raw -Encoding ASCII).Trim() | ConvertTo-SecureString
        $taskPointer=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecure)
        $env:PGPASSWORD=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskPointer)
    }
    $env:PGHOST='127.0.0.1';$env:PGPORT='5432';$env:PGDATABASE='pckeiba';$env:PGUSER='jra_audit_reader';$env:PYTHONUTF8='1'
    $taskLog=Join-Path $taskRoot ('service-'+[guid]::NewGuid().ToString('N')+'.log')
    & (Join-Path $taskBase 'venv\Scripts\python.exe') -B (Join-Path $PSScriptRoot 'keiba_lab_live.py') *> $taskLog
    # Keep the Scheduler's failure/restart status tied to the actual server.
    exit $LASTEXITCODE
} catch {
    [ordered]@{status='stopped';reason='local_viewer_start_failed'} | ConvertTo-Json -Compress
    exit 1
} finally {
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    if($taskPointer -ne [IntPtr]::Zero){[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskPointer)}
    if($null -ne $taskSecure){$taskSecure.Dispose()}
}
