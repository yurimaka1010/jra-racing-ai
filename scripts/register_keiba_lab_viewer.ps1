param([switch]$CheckOnly)
$ErrorActionPreference='Stop'
$taskName='KEIBA_LAB_Local_Viewer'
$taskBase='C:\Users\acbd5\AppData\Local\JraRacingAI'
$taskRoot=Join-Path $taskBase 'keiba-lab-live'
$taskProject=Split-Path -Parent $PSScriptRoot
$taskRunner=Join-Path $PSScriptRoot 'keiba_lab_live.ps1'
$taskShell=Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$taskUrl='http://127.0.0.1:8767'
$taskIdentity=[Security.Principal.WindowsIdentity]::GetCurrent().Name
$taskDescription='Local-only KEIBA LAB viewer at user logon; no model fits, prediction recording, DB writes or research-task changes.'

function Test-ViewerReady {
    try {
        $taskHealth=Invoke-RestMethod -Uri ($taskUrl+'/api/health') -TimeoutSec 2
        return ($taskHealth.status -eq 'ok' -and $taskHealth.version -eq 'keiba-lab-live.20261024.1')
    } catch {return $false}
}

try {
    if(($taskIdentity -split '\\')[-1] -ne 'acbd5'){throw 'wrong_windows_user'}
    foreach($taskPath in @($taskRunner,$taskShell,(Join-Path $PSScriptRoot 'keiba_lab_live.py'),
            (Join-Path $taskBase 'venv\Scripts\python.exe'),
            (Join-Path $taskBase 'trial-20261017\credentials\db-password.dpapi'))){
        if(-not (Test-Path -LiteralPath $taskPath -PathType Leaf)){throw 'dependency_missing'}
    }
    if($CheckOnly){
        [ordered]@{status='ok';check_only=$true;task_name=$taskName;user=$taskIdentity;
            dependencies_present=$true;viewer_ready=(Test-ViewerReady);changes=0}|ConvertTo-Json -Compress
        exit 0
    }
    $taskExisting=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    $taskArguments='-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "'+$taskRunner+'" -Serve'
    if($null -ne $taskExisting -and ($taskExisting.Description -ne $taskDescription -or
            $taskExisting.Actions.Count -ne 1 -or $taskExisting.Actions[0].Execute -ne $taskShell -or
            $taskExisting.Actions[0].Arguments -ne $taskArguments)){
        throw 'existing_task_conflict'
    }
    $taskRunFolder=Join-Path $taskRoot ('startup-registrations\'+[guid]::NewGuid().ToString('N'))
    [IO.Directory]::CreateDirectory($taskRunFolder)|Out-Null
    if($null -ne $taskExisting){
        Export-ScheduledTask -TaskName $taskName | Set-Content -LiteralPath (Join-Path $taskRunFolder 'previous-task.xml') -Encoding Unicode
    }
    $taskTrigger=New-ScheduledTaskTrigger -AtLogOn -User $taskIdentity
    $taskTrigger.Delay='PT5S'
    $taskAction=New-ScheduledTaskAction -Execute $taskShell -Argument $taskArguments -WorkingDirectory $taskProject
    $taskPrincipal=New-ScheduledTaskPrincipal -UserId $taskIdentity -LogonType Interactive -RunLevel Limited
    $taskSettings=New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $taskName -Action $taskAction -Trigger $taskTrigger `
        -Principal $taskPrincipal -Settings $taskSettings -Description $taskDescription -Force|Out-Null

    $taskDesktop=[Environment]::GetFolderPath('Desktop')
    $taskShortcutPath=Join-Path $taskDesktop 'KEIBA LAB.lnk'
    $taskLinkShell=New-Object -ComObject WScript.Shell
    if(Test-Path -LiteralPath $taskShortcutPath){
        $taskExistingLink=$taskLinkShell.CreateShortcut($taskShortcutPath)
        if($taskExistingLink.TargetPath -ne $taskShell -or $taskExistingLink.Arguments -notlike ('*'+$taskRunner+'*')){
            $taskShortcutPath=Join-Path $taskDesktop 'KEIBA LAB Local.lnk'
            if(Test-Path -LiteralPath $taskShortcutPath){throw 'existing_shortcut_conflict'}
        }
    }
    $taskShortcut=$taskLinkShell.CreateShortcut($taskShortcutPath)
    $taskShortcut.TargetPath=$taskShell
    $taskShortcut.Arguments='-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "'+$taskRunner+'"'
    $taskShortcut.WorkingDirectory=$taskProject
    $taskShortcut.WindowStyle=7
    $taskShortcut.Description='Start or open local KEIBA LAB. Reuses a healthy viewer; no prediction tasks are started.'
    $taskShortcut.Save()

    Start-ScheduledTask -TaskName $taskName
    $taskWait=[Diagnostics.Stopwatch]::StartNew()
    while($taskWait.Elapsed.TotalSeconds -lt 30 -and -not (Test-ViewerReady)){Start-Sleep -Milliseconds 500}
    if(-not (Test-ViewerReady)){throw 'viewer_start_failed'}
    $taskRegistered=Get-ScheduledTask -TaskName $taskName
    Export-ScheduledTask -TaskName $taskName | Set-Content -LiteralPath (Join-Path $taskRunFolder 'task.xml') -Encoding Unicode
    $taskSummary=[ordered]@{status='ready';url=$taskUrl;task_name=$taskName;state=[string]$taskRegistered.State;
        user=$taskIdentity;start_at_logon=$true;delay_seconds=5;requires_user_logon=$true;
        restart_on_failure_count=3;restart_interval_seconds=60;time_limit='unlimited';
        shortcut=$taskShortcutPath;registration_folder=$taskRunFolder;finished_at=[DateTimeOffset]::Now.ToString('o')}
    $taskSummary|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $taskRunFolder 'result.json') -Encoding UTF8
    $taskSummary|ConvertTo-Json -Depth 5 -Compress
    exit 0
} catch {
    # Fixed local setup codes only; never include a connection string or password.
    $taskReason=$_.Exception.Message
    if($taskReason -notin @('wrong_windows_user','dependency_missing','existing_task_conflict',
            'existing_shortcut_conflict','viewer_start_failed')){$taskReason='viewer_registration_failed'}
    [ordered]@{status='stopped';reason=$taskReason;task_name=$taskName}|ConvertTo-Json -Compress
    exit 1
}
