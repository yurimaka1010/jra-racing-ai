param([Parameter(Mandatory=$true)][ValidatePattern('^[a-f0-9]{32}$')][string]$RunId,
      [switch]$ReplaceResearchTask)
$ErrorActionPreference='Stop'
$taskBase=Join-Path $env:LOCALAPPDATA 'JraRacingAI'
$taskRoot=Join-Path $taskBase ('pedigree-forward\evaluations\'+$RunId)
$taskName='JraRacingAI_Pedigree3_Research_T10'
$taskShell=Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'

function Get-OldTasks {
    $taskItems=@(Get-ScheduledTask | Where-Object {$_.TaskName -like '*JraRacingAI*' -and $_.TaskName -ne $taskName} | Sort-Object TaskPath,TaskName | ForEach-Object {
        $taskXml=Export-ScheduledTask -TaskName $_.TaskName -TaskPath $_.TaskPath
        $taskHash=[Security.Cryptography.SHA256]::Create()
        try {$taskDigest=([BitConverter]::ToString($taskHash.ComputeHash([Text.Encoding]::UTF8.GetBytes($taskXml)))).Replace('-','').ToLowerInvariant()} finally {$taskHash.Dispose()}
        [ordered]@{name=$_.TaskName;path=$_.TaskPath;sha256=$taskDigest}
    })
    return ConvertTo-Json -InputObject $taskItems -Compress
}

function Write-NewJson([string]$Path,$Value) {
    $taskStream=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try {$taskBytes=[Text.UTF8Encoding]::new($false).GetBytes(($Value|ConvertTo-Json -Depth 16));$taskStream.Write($taskBytes,0,$taskBytes.Length);$taskStream.Flush($true)} finally {$taskStream.Dispose()}
}

try {
    if((Get-TimeZone).Id -ne 'Tokyo Standard Time'){throw 'timezone_mismatch'}
    $taskFreeze=Get-Content -LiteralPath (Join-Path $taskRoot 'model_freeze.json') -Raw -Encoding UTF8|ConvertFrom-Json
    $taskPlan=Get-Content -LiteralPath (Join-Path $taskRoot 'preregistration.json') -Raw -Encoding UTF8|ConvertFrom-Json
    if($taskPlan.primary_metric -ne 'within_race_auc' -or $taskPlan.thresholds.guardrail_method -ne 'point_estimate'){throw 'protocol_invalid'}
    $taskRuntime=$taskFreeze.runtime_project
    $taskRunner=Join-Path $taskRuntime 'scripts\pedigree_forward.ps1'
    if(-not (Test-Path -LiteralPath $taskRunner)){throw 'runtime_missing'}
    $taskProbe=& $taskShell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $taskRunner -Mode preflight -RunId $RunId
    if($LASTEXITCODE -ne 0){throw 'preflight_failed'}
    $taskProbeValue=$taskProbe|ConvertFrom-Json
    if($taskProbeValue.status -ne 'ready' -or $taskProbeValue.model_count -ne 3){throw 'preflight_failed'}
    $taskBefore=Get-OldTasks
    $taskFolder=Join-Path $taskRoot ('task-registrations\'+[guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($taskFolder)|Out-Null
    $taskArguments='-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "'+$taskRunner+'" -Mode record -RunId '+$RunId
    $taskExisting=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    $taskReplaced=$false
    if($null -ne $taskExisting){
        if($taskExisting.Actions.Count -ne 1 -or $taskExisting.Actions[0].Execute -ne $taskShell -or
           $taskExisting.Actions[0].Arguments -ne $taskArguments -or $taskExisting.Actions[0].WorkingDirectory -ne $taskRuntime){
            $taskScope=Get-Content -LiteralPath (Join-Path $taskRoot 'execution_scope.json') -Raw -Encoding UTF8|ConvertFrom-Json
            $taskPreviousId=[string]$taskScope.supersedes_research_run_id
            if(-not $ReplaceResearchTask -or -not $taskScope.research_task_reconfiguration_authorized -or
               $taskPreviousId -notmatch '^[a-f0-9]{32}$' -or [string]$taskExisting.State -eq 'Running'){throw 'existing_task_conflict'}
            $taskPreviousRoot=Join-Path $taskBase ('pedigree-forward\evaluations\'+$taskPreviousId)
            $taskPreviousFreeze=Get-Content -LiteralPath (Join-Path $taskPreviousRoot 'model_freeze.json') -Raw -Encoding UTF8|ConvertFrom-Json
            $taskPreviousRunner=Join-Path $taskPreviousFreeze.runtime_project 'scripts\pedigree_forward.ps1'
            $taskPreviousArgs='-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "'+$taskPreviousRunner+'" -Mode record -RunId '+$taskPreviousId
            if($taskExisting.Actions.Count -ne 1 -or $taskExisting.Actions[0].Execute -ne $taskShell -or
               $taskExisting.Actions[0].Arguments -ne $taskPreviousArgs -or
               $taskExisting.Actions[0].WorkingDirectory -ne $taskPreviousFreeze.runtime_project){throw 'existing_task_conflict'}
            $taskPreviousXml=Export-ScheduledTask -TaskName $taskName
            $taskBackup=[IO.File]::Open((Join-Path $taskFolder 'previous_task.xml'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
            try {$taskBytes=[Text.UTF8Encoding]::new($false).GetBytes($taskPreviousXml);$taskBackup.Write($taskBytes,0,$taskBytes.Length);$taskBackup.Flush($true)} finally {$taskBackup.Dispose()}
            $taskAction=New-ScheduledTaskAction -Execute $taskShell -Argument $taskArguments -WorkingDirectory $taskRuntime
            $taskExisting.Actions=@($taskAction)
            $taskExisting.Description='Independent frozen pedigree3 T10 research, zero fits/results/betting, run '+$RunId
            Set-ScheduledTask -InputObject $taskExisting|Out-Null
            $taskReplaced=$true
        }
    } else {
        $taskTrigger=New-ScheduledTaskTrigger -Daily -At ([datetime]::ParseExact('2026-10-31 07:00','yyyy-MM-dd HH:mm',[cultureinfo]::InvariantCulture))
        $taskTrigger.EndBoundary=([datetime]::ParseExact(($taskPlan.window_end_date+' 18:00'),'yyyy-MM-dd HH:mm',[cultureinfo]::InvariantCulture)).ToString('yyyy-MM-ddTHH:mm:ss')
        $taskAction=New-ScheduledTaskAction -Execute $taskShell -Argument $taskArguments -WorkingDirectory $taskRuntime
        $taskPrincipal=New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
        $taskSettings=New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Hours 12) -MultipleInstances IgnoreNew -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5)
        Register-ScheduledTask -TaskName $taskName -Action $taskAction -Trigger $taskTrigger -Principal $taskPrincipal -Settings $taskSettings -Description ('Independent frozen pedigree3 T10 research, zero fits/results/betting, run '+$RunId)|Out-Null
    }
    if((Get-OldTasks) -ne $taskBefore){throw 'protected_task_changed'}
    $taskSmoke=$null
    if((Get-Date).Date -lt [datetime]'2026-10-31') {
        $taskStart=Get-Date
        Start-ScheduledTask -TaskName $taskName
        $taskPassed=$false
        for($taskI=0;$taskI -lt 40;$taskI++) {
            Start-Sleep -Seconds 1
            $taskInfo=Get-ScheduledTaskInfo -TaskName $taskName
            $taskState=Get-ScheduledTask -TaskName $taskName
            if($taskInfo.LastRunTime -ge $taskStart.AddSeconds(-2) -and [string]$taskState.State -ne 'Running') {
                if($taskInfo.LastTaskResult -ne 0){throw 'scheduler_smoke_failed'}
                $taskPassed=$true;break
            }
        }
        if(-not $taskPassed){throw 'scheduler_smoke_timeout'}
        $taskSmoke=[ordered]@{status='ok';last_result=$taskInfo.LastTaskResult;actual_recording_started=$false}
    }
    if((Get-OldTasks) -ne $taskBefore){throw 'protected_task_changed'}
    $taskState=Get-ScheduledTask -TaskName $taskName;$taskInfo=Get-ScheduledTaskInfo -TaskName $taskName
    Write-NewJson (Join-Path $taskFolder 'existing_tasks_before.json') ($taskBefore|ConvertFrom-Json)
    $taskXml=Export-ScheduledTask -TaskName $taskName
    $taskStream=[IO.File]::Open((Join-Path $taskFolder 'task.xml'),[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try {$taskBytes=[Text.UTF8Encoding]::new($false).GetBytes($taskXml);$taskStream.Write($taskBytes,0,$taskBytes.Length);$taskStream.Flush($true)} finally {$taskStream.Dispose()}
    $taskSummary=[ordered]@{status='registered';task_name=$taskName;state=[string]$taskState.State;
        next_run_jst=$taskInfo.NextRunTime.ToString('o');first_date='2026-10-31';daily_start='07:00 JST';daily_end='18:00 JST';
        formal_start_week=$taskPlan.window_start_week;formal_end=$taskPlan.window_end_date;
        requires_logged_on_user=$true;screen_lock_allowed=$true;existing_tasks_unchanged=$true;
        scheduler_smoke_test=$taskSmoke;runtime_project=$taskRuntime;new_fits=0;automatic_results_evaluation=$false;
        only_research_task_action_replaced=$taskReplaced}
    Write-NewJson (Join-Path $taskFolder 'registration.json') $taskSummary
    $taskSummary|ConvertTo-Json -Depth 10 -Compress
} catch {
    [ordered]@{status='stopped';reason='pedigree_research_task_registration_failed';task_name=$taskName;new_fits=0}|ConvertTo-Json -Compress
    exit 2
}
