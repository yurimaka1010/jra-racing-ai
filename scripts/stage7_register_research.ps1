param([Parameter(Mandatory=$true)][ValidatePattern('^[a-f0-9]{32}$')][string]$RunId)
$ErrorActionPreference='Stop'
$taskName='JraRacingAI_Stage7_H_Research'
$taskBase='C:\Users\acbd5\AppData\Local\JraRacingAI'
$taskFolder=Join-Path $taskBase ('stage7-development\evaluations\'+$RunId)
$taskPowerShell=Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$taskRunner=Join-Path $PSScriptRoot 'stage7_local.ps1'
try {
    if((Get-TimeZone).Id -ne 'Tokyo Standard Time'){throw 'timezone_mismatch'}
    $taskCheck=& $taskPowerShell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $taskRunner -Mode preflight -RunId $RunId
    if($LASTEXITCODE -ne 0){throw 'preflight_failed'}
    $taskProbe=& $taskPowerShell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $taskRunner -Mode probe -RunId $RunId
    if($LASTEXITCODE -ne 0){throw 'probe_failed'}
    $taskPlan=Get-Content -LiteralPath (Join-Path $taskFolder 'forward_plan.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if($taskPlan.H_model_id -ne '4efe34cdef5b4da9a33c254c4586a2bf' -or $taskPlan.H_temperature -ne 1 -or $taskPlan.stage7b_approved){throw 'plan_invalid'}
    $taskAt=[datetime]::ParseExact(($taskPlan.start_date+' 08:00'),'yyyy-MM-dd HH:mm',[cultureinfo]::InvariantCulture)
    $taskEnd=[datetime]::ParseExact(($taskPlan.end_date+' 18:00'),'yyyy-MM-dd HH:mm',[cultureinfo]::InvariantCulture)
    if((Get-Date) -gt $taskEnd){throw 'plan_expired'}
    $taskArguments='-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "'+$taskRunner+'" -Mode record -RunId '+$RunId
    $taskExisting=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if($null -ne $taskExisting){
        if($taskExisting.Actions.Count -ne 1 -or $taskExisting.Actions[0].Execute -ne $taskPowerShell -or $taskExisting.Actions[0].Arguments -ne $taskArguments){throw 'existing_research_task_conflict'}
    } else {
        $taskAction=New-ScheduledTaskAction -Execute $taskPowerShell -Argument $taskArguments -WorkingDirectory (Split-Path -Parent $PSScriptRoot)
        $taskTrigger=New-ScheduledTaskTrigger -Daily -At $taskAt
        $taskTrigger.EndBoundary=$taskEnd.ToString('yyyy-MM-ddTHH:mm:ss')
        $taskPrincipal=New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
        $taskSettings=New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 10) -MultipleInstances IgnoreNew
        Register-ScheduledTask -TaskName $taskName -Action $taskAction -Trigger $taskTrigger -Principal $taskPrincipal -Settings $taskSettings -Description ('Frozen H T10 research records; no results, fit or betting. Stage7a run '+$RunId) | Out-Null
    }
    $taskRegistered=Get-ScheduledTask -TaskName $taskName
    $taskInfo=Get-ScheduledTaskInfo -TaskName $taskName
    $taskResult=[ordered]@{status='ok';task_name=$taskName;state=[string]$taskRegistered.State;
        next_run_jst=$taskInfo.NextRunTime.ToString('o');start_date=$taskPlan.start_date;end_date=$taskPlan.end_date;
        daily_window='08:00-18:00 JST';requires_logged_on_user=$true;existing_morning_task_modified=$false;
        run_id=$RunId;actual_future_predictions_saved=0;stage7b_fits_allowed=0}
    $taskJson=$taskResult | ConvertTo-Json -Depth 8
    $taskOutput=Join-Path $taskFolder ('task_registration_'+[guid]::NewGuid().ToString('N')+'.json')
    $taskStream=[IO.File]::Open($taskOutput,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try {$taskBytes=[Text.UTF8Encoding]::new($false).GetBytes($taskJson);$taskStream.Write($taskBytes,0,$taskBytes.Length);$taskStream.Flush($true)} finally {$taskStream.Dispose()}
    $taskResult | ConvertTo-Json -Compress
} catch {
    [ordered]@{status='stopped';reason='stage7_research_task_registration_failed';existing_morning_task_modified=$false} | ConvertTo-Json -Compress
    exit 2
}
