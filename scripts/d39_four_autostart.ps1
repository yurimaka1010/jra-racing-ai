param([switch]$CheckOnly)
$ErrorActionPreference='Stop'
$taskBase='C:\Users\acbd5\AppData\Local\JraRacingAI'
$taskRunId='86ed126ffc8f4e2e85c8740eacf941f0'
$taskParent=Join-Path $taskBase ('d39-generation\evaluations\'+$taskRunId)
$taskShell=Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$taskWorker=Join-Path $PSScriptRoot 'd39_current.ps1'
$taskEvents=$null
$taskJob=$null
$taskResult=2

function Write-TaskJson([string]$Path,$Value) {
    $taskBytes=[Text.UTF8Encoding]::new($false).GetBytes(($Value | ConvertTo-Json -Depth 20))
    $taskStream=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try {$taskStream.Write($taskBytes,0,$taskBytes.Length);$taskStream.Flush($true)} finally {$taskStream.Dispose()}
}

function Invoke-TaskWorker([string]$Mode,[string]$RaceDate='') {
    $taskArgs=@('-NoProfile','-NonInteractive','-WindowStyle','Hidden','-ExecutionPolicy','Bypass',
                '-File',$taskWorker,'-Mode',$Mode,'-RunId',$taskRunId)
    if($RaceDate){$taskArgs+=@('-RaceDate',$RaceDate)}
    & $taskShell @taskArgs 2>$null | ForEach-Object {
        try {
            $taskValue=$_ | ConvertFrom-Json -ErrorAction Stop
            $taskEvents.WriteLine(($taskValue | ConvertTo-Json -Depth 24 -Compress))
        } catch {
            $taskEvents.WriteLine('{"status":"note","reason":"non_json_output_suppressed"}')
        }
        $taskEvents.Flush()
    }
    if($LASTEXITCODE -ne 0){throw 'worker_failed'}
}

function Get-TaskDayEntries([string]$RaceDate) {
    $taskPointer=[IntPtr]::Zero
    $taskSecure=$null
    try {
        $taskCredential=Join-Path $taskBase 'trial-20261017\credentials\db-password.dpapi'
        $taskCredentialPart=$taskCredential
        while($taskCredentialPart){
            if((Test-Path -LiteralPath $taskCredentialPart) -and ((Get-Item -LiteralPath $taskCredentialPart -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'unsafe_path'}
            $taskCredentialNext=Split-Path -Parent $taskCredentialPart
            if($taskCredentialNext -eq $taskCredentialPart){break}
            $taskCredentialPart=$taskCredentialNext
        }
        $taskSecure=(Get-Content -LiteralPath $taskCredential -Raw -Encoding ASCII).Trim()|ConvertTo-SecureString
        $taskPointer=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecure)
        $env:PGPASSWORD=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskPointer)
        $env:PGHOST='127.0.0.1';$env:PGPORT='5432';$env:PGDATABASE='pckeiba';$env:PGUSER='jra_audit_reader';$env:PYTHONUTF8='1'
        $taskOutput=& (Join-Path $taskBase 'venv\Scripts\python.exe') -B (Join-Path $PSScriptRoot 'd39_four_day_probe.py') --date $RaceDate 2>$null
        if($LASTEXITCODE -ne 0){throw 'day_probe_failed'}
        $taskProbe=$taskOutput|ConvertFrom-Json
        if($taskProbe.status -notin @('ready','no_entries') -or $taskProbe.date -ne $RaceDate){throw 'day_probe_failed'}
        $taskEvents.WriteLine(($taskProbe|ConvertTo-Json -Compress))
        return $taskProbe.entry_races
    } finally {
        Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
        if($taskPointer -ne [IntPtr]::Zero){[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskPointer)}
        if($null -ne $taskSecure){$taskSecure.Dispose()}
    }
}

try {
    if((Get-TimeZone).Id -ne 'Tokyo Standard Time'){throw 'timezone_mismatch'}
    # Use only the existing nonsynchronized private root, including all ancestors.
    $taskCursor=$taskParent
    while($taskCursor){
        if((Test-Path -LiteralPath $taskCursor) -and ((Get-Item -LiteralPath $taskCursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'unsafe_path'}
        $taskNext=Split-Path -Parent $taskCursor
        if($taskNext -eq $taskCursor){break}
        $taskCursor=$taskNext
    }
    if(-not (Test-Path -LiteralPath (Join-Path $taskParent 'four_addendum_plan.json'))){throw 'plan_missing'}
    $taskNow=Get-Date
    $taskDay=$taskNow.ToString('yyyy-MM-dd')
    $taskPlan=Get-Content -LiteralPath (Join-Path $taskParent 'four_addendum_plan.json') -Raw -Encoding UTF8|ConvertFrom-Json
    if($taskPlan.start_date -notmatch '^\d{4}-\d{2}-\d{2}$' -or $taskPlan.end_date -notmatch '^\d{4}-\d{2}-\d{2}$'){throw 'plan_dates_invalid'}
    $taskInWindow=($taskDay -ge $taskPlan.start_date -and $taskDay -le $taskPlan.end_date) -and ($taskNow.TimeOfDay -ge [TimeSpan]::Parse('07:05:00')) -and ($taskNow.Hour -lt 17)
    $taskRecord=$taskInWindow -and -not $CheckOnly
    $taskJob=Join-Path $taskParent ('auto-four\runs\'+[guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($taskJob) | Out-Null
    $taskEventStream=[IO.File]::Open((Join-Path $taskJob 'events.jsonl'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    $taskEvents=[IO.StreamWriter]::new($taskEventStream,[Text.UTF8Encoding]::new($false))
    $taskEvents.AutoFlush=$true
    Write-TaskJson (Join-Path $taskJob 'start.json') ([ordered]@{status='started';started_at=$taskNow.ToString('o');
        recording_enabled=$taskRecord;check_only=(-not $taskRecord);run_id=$taskRunId;results_read=$false;new_fits=0})
    if($taskRecord){
        if((Get-TaskDayEntries -RaceDate $taskDay) -gt 0){
            Invoke-TaskWorker -Mode 'record-four'
            $taskReason='daily_recording_completed'
        } else {
            $taskRecord=$false
            $taskReason='no_JRA_entries_today;no_worker_started'
        }
    } elseif($CheckOnly){
        Invoke-TaskWorker -Mode 'preflight-four'
        $taskReason='noninteractive_saved_model_check_completed'
    } else {$taskReason='outside_registered_plan_dates_or_daily_window;no_worker_started'}
    $taskResult=0
    $taskFinish=[ordered]@{status='ok';reason=$taskReason;finished_at=(Get-Date).ToString('o');recording_enabled=$taskRecord;
        check_only=(-not $taskRecord);results_read=$false;new_fits=0;log_folder=$taskJob}
    Write-TaskJson (Join-Path $taskJob 'finish.json') $taskFinish
    $taskFinish | ConvertTo-Json -Compress
} catch {
    $taskFailure=[ordered]@{status='stopped';reason='four_model_automatic_start_failed';finished_at=(Get-Date).ToString('o');
        results_read=$false;new_fits=0;log_folder=$taskJob}
    if($taskJob){try {Write-TaskJson (Join-Path $taskJob 'failure.json') $taskFailure} catch {}}
    $taskFailure | ConvertTo-Json -Compress
} finally {
    if($null -ne $taskEvents){$taskEvents.Dispose()}
}
exit $taskResult
