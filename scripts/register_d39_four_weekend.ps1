$ErrorActionPreference='Stop'
$taskName='D39_FourModels_20261024_25'
$taskBase='C:\Users\acbd5\AppData\Local\JraRacingAI'
$taskRunId='86ed126ffc8f4e2e85c8740eacf941f0'
$taskParent=Join-Path $taskBase ('d39-generation\evaluations\'+$taskRunId)
$taskShell=Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$taskRunner=Join-Path $PSScriptRoot 'd39_four_autostart.ps1'
$taskProject=Split-Path -Parent $PSScriptRoot
$taskOutput=$null

function Get-ProtectedTasks {
    $taskMap=[ordered]@{}
    # Existing research hashes deliberately cover this established namespace.
    # The new D39-only task is separate; no existing snapshot is rewritten.
    Get-ScheduledTask | Where-Object {$_.TaskName -like '*JraRacingAI*'} | Sort-Object TaskPath,TaskName | ForEach-Object {
        $taskXml=Export-ScheduledTask -TaskName $_.TaskName -TaskPath $_.TaskPath
        $taskSha=[Security.Cryptography.SHA256]::Create()
        try {$taskMap[($_.TaskPath+$_.TaskName)]=([BitConverter]::ToString($taskSha.ComputeHash([Text.Encoding]::UTF8.GetBytes($taskXml)))).Replace('-','').ToLowerInvariant()}
        finally {$taskSha.Dispose()}
    }
    return ($taskMap | ConvertTo-Json -Compress)
}

function Write-TaskJson([string]$Path,$Value) {
    $taskBytes=[Text.UTF8Encoding]::new($false).GetBytes(($Value | ConvertTo-Json -Depth 20))
    $taskStream=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try {$taskStream.Write($taskBytes,0,$taskBytes.Length);$taskStream.Flush($true)} finally {$taskStream.Dispose()}
}

try {
    if((Get-TimeZone).Id -ne 'Tokyo Standard Time'){throw 'timezone_mismatch'}
    if((Get-Date).Date -gt [datetime]'2026-10-25'){throw 'trial_dates_finished'}
    $taskBefore=Get-ProtectedTasks
    $taskCheckOutput=& $taskShell -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File $taskRunner -CheckOnly
    if($LASTEXITCODE -ne 0){throw 'noninteractive_check_failed'}
    $taskCheck=$taskCheckOutput | ConvertFrom-Json
    if($taskCheck.status -ne 'ok' -or -not $taskCheck.check_only -or $taskCheck.recording_enabled){throw 'unexpected_check_status'}
    $taskArguments='-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "'+$taskRunner+'"'
    $taskAt=@([datetime]::ParseExact('2026-10-24 07:05','yyyy-MM-dd HH:mm',[cultureinfo]::InvariantCulture),
              [datetime]::ParseExact('2026-10-25 07:05','yyyy-MM-dd HH:mm',[cultureinfo]::InvariantCulture))
    $taskIdentity=[Security.Principal.WindowsIdentity]::GetCurrent().Name
    $taskExisting=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if($null -ne $taskExisting){
        $taskStarts=@($taskExisting.Triggers | ForEach-Object {$_.StartBoundary.Substring(0,19)} | Sort-Object)
        if($taskExisting.Actions.Count -ne 1 -or $taskExisting.Actions[0].Execute -ne $taskShell -or
           $taskExisting.Actions[0].Arguments -ne $taskArguments -or $taskExisting.Actions[0].WorkingDirectory -ne $taskProject -or
           ($taskStarts -join ',') -ne '2026-10-24T07:05:00,2026-10-25T07:05:00' -or
           [string]$taskExisting.Principal.LogonType -ne 'Interactive'){throw 'existing_task_conflict'}
    } else {
        $taskTriggers=@($taskAt | ForEach-Object {
            $taskTrigger=New-ScheduledTaskTrigger -Once -At $_
            $taskTrigger.EndBoundary=$_.Date.AddHours(18).ToString('yyyy-MM-ddTHH:mm:ss')
            $taskTrigger
        })
        $taskAction=New-ScheduledTaskAction -Execute $taskShell -Argument $taskArguments -WorkingDirectory $taskProject
        $taskPrincipal=New-ScheduledTaskPrincipal -UserId $taskIdentity -LogonType Interactive -RunLevel Limited
        $taskSettings=New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 11) -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)
        Register-ScheduledTask -TaskName $taskName -Action $taskAction -Trigger $taskTriggers -Principal $taskPrincipal -Settings $taskSettings -Description 'Only Oct24/25 2026 07:05 JST: old D39, new D39, fixed-parameter D39 control and frozen H. Independent local T10 records, no fits/results/betting. Existing tasks unchanged.' | Out-Null
    }
    if((Get-ProtectedTasks) -ne $taskBefore){throw 'protected_task_changed'}
    $taskSmoke=$null
    if((Get-Date).Date -eq [datetime]'2026-10-23'){
        # Run through the real interactive Scheduler context today. The date
        # guard forces check-only; tomorrow's predictions cannot start tonight.
        $taskStarted=Get-Date
        Start-ScheduledTask -TaskName $taskName
        $taskCompleted=$false
        for($taskWait=0;$taskWait -lt 40;$taskWait++){
            Start-Sleep -Seconds 1
            $taskState=Get-ScheduledTask -TaskName $taskName
            $taskInfo=Get-ScheduledTaskInfo -TaskName $taskName
            if($taskInfo.LastRunTime -ge $taskStarted.AddSeconds(-2) -and [string]$taskState.State -ne 'Running'){
                if($taskInfo.LastTaskResult -ne 0){throw 'scheduler_smoke_test_failed'}
                $taskCompleted=$true;break
            }
        }
        if(-not $taskCompleted){throw 'scheduler_smoke_test_not_complete'}
        $taskSmoke=[ordered]@{status='ok';real_scheduler_start=$true;last_result=$taskInfo.LastTaskResult;actual_live_predictions_started=$false}
    }
    if((Get-ProtectedTasks) -ne $taskBefore){throw 'protected_task_changed'}
    $taskRegistered=Get-ScheduledTask -TaskName $taskName
    $taskInfo=Get-ScheduledTaskInfo -TaskName $taskName
    $taskOutput=Join-Path $taskParent ('auto-four\registrations\'+[guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($taskOutput) | Out-Null
    Write-TaskJson (Join-Path $taskOutput 'protected_before.json') ($taskBefore | ConvertFrom-Json)
    $taskXml=Export-ScheduledTask -TaskName $taskName
    $taskXmlPath=Join-Path $taskOutput 'task.xml'
    $taskXmlStream=[IO.File]::Open($taskXmlPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try {$taskBytes=[Text.UTF8Encoding]::new($false).GetBytes($taskXml);$taskXmlStream.Write($taskBytes,0,$taskBytes.Length);$taskXmlStream.Flush($true)} finally {$taskXmlStream.Dispose()}
    $taskSummary=[ordered]@{status='ok';task_name=$taskName;state=[string]$taskRegistered.State;
        next_run_jst=$taskInfo.NextRunTime.ToString('o');start_times_jst=@('2026-10-24T07:05:00+09:00','2026-10-25T07:05:00+09:00');
        requires_logged_on_user=$true;screen_lock_allowed=$true;noninteractive_credentials_checked=$true;
        startup_check=$taskCheck;scheduler_smoke_test=$taskSmoke;existing_tasks_unchanged=$true;
        new_task_only=$true;automatic_result_evaluation=$false;new_fits=0;
        restart_count=3;restart_interval_minutes=5;multiple_instances='IgnoreNew';registration_folder=$taskOutput}
    Write-TaskJson (Join-Path $taskOutput 'registration.json') $taskSummary
    $taskSummary | ConvertTo-Json -Depth 12 -Compress
} catch {
    [ordered]@{status='stopped';reason='four_model_task_setup_or_check_failed';task_name=$taskName;registration_folder=$taskOutput} | ConvertTo-Json -Compress
    exit 2
}
