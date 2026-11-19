param()
$ErrorActionPreference = 'Stop'
$taskName = 'JraRacingAI_V03_20261017_0700'
$taskRoot = Join-Path $env:LOCALAPPDATA 'JraRacingAI\trial-20261017'
$taskCredentialDirectory = Join-Path $taskRoot 'credentials'
$taskCredentialPath = Join-Path $taskCredentialDirectory 'db-password.dpapi'
$taskPowerShell = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$taskRunner = Join-Path $PSScriptRoot 'run_20261017_trial.ps1'
$taskProject = Split-Path -Parent $PSScriptRoot
$taskReason = 'setup_failed'

function Invoke-TaskCheck([string[]]$ExtraArguments) {
    $taskLines = & $taskPowerShell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $taskRunner @ExtraArguments
    $taskCode = $LASTEXITCODE
    $taskResult = ($taskLines -join "`n") | ConvertFrom-Json
    if ($taskCode -ne 0 -or $taskResult.status -ne 'ok') {
        if ($taskResult.reason -match '^[a-z_]{1,64}$') { $script:taskReason = $taskResult.reason }
        throw 'check_failed'
    }
    return $taskResult
}

try {
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) { $taskReason = 'windows_required'; throw 'stop' }
    if ((Get-TimeZone).Id -ne 'Tokyo Standard Time') { $taskReason = 'timezone_mismatch'; throw 'stop' }
    # Offline model/import/path test before asking for a secret or registering.
    $taskReason = 'offline_preflight_failed'
    $taskOffline = Invoke-TaskCheck -ExtraArguments @('-Preflight')
    if (-not (Test-Path -LiteralPath $taskCredentialPath -PathType Leaf)) {
        $taskReason = 'credential_setup_failed'
        [IO.Directory]::CreateDirectory($taskCredentialDirectory) | Out-Null
        $taskIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $taskAcl = New-Object Security.AccessControl.DirectorySecurity
        $taskAcl.SetAccessRuleProtection($true, $false)
        foreach ($taskSid in @($taskIdentity.User, [Security.Principal.SecurityIdentifier]::new('S-1-5-18'))) {
            $taskRule = [Security.AccessControl.FileSystemAccessRule]::new($taskSid, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
            $taskAcl.AddAccessRule($taskRule)
        }
        Set-Acl -LiteralPath $taskCredentialDirectory -AclObject $taskAcl
        $taskSecret = Read-Host 'Local read-only DB password (hidden; encrypted for this Windows user)' -AsSecureString
        try {
            if ($taskSecret.Length -eq 0) { throw 'empty_secret' }
            $taskEncrypted = ConvertFrom-SecureString -SecureString $taskSecret
            $taskStream = [IO.File]::Open($taskCredentialPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
            try {
                $taskBytes = [Text.Encoding]::ASCII.GetBytes($taskEncrypted)
                $taskStream.Write($taskBytes, 0, $taskBytes.Length)
            } finally { $taskStream.Dispose() }
        } finally { $taskSecret.Dispose() }
    }
    # Read-only authentication/schema/counts only. Missing cards may be pending
    # at setup; the scheduled prediction will require actual complete cards.
    $taskReason = 'database_probe_failed'
    $taskProbe = Invoke-TaskCheck -ExtraArguments @('-Probe')
    $taskAt = [datetime]::ParseExact('2026-10-17 07:00', 'yyyy-MM-dd HH:mm', [cultureinfo]::InvariantCulture)
    if ((Get-Date).Date -ne $taskAt.Date) { $taskReason = 'outside_trial_date'; throw 'stop' }
    if ((Get-Date) -ge $taskAt) {
        # Never register a past-only task or pretend a late run was at 07:00.
        $taskReason = 'late_run_failed'
        $taskLate = Invoke-TaskCheck -ExtraArguments @()
        [pscustomobject]@{status='ok';kind='trial_late_once';scheduled_task_registered=$false;prediction=$taskLate} | ConvertTo-Json -Depth 8 -Compress
        exit 0
    }
    $taskReason = 'task_registration_failed'
    $taskArguments = '-NoProfile -NonInteractive -ExecutionPolicy Bypass -File "' + $taskRunner + '"'
    $taskExisting = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    if ($null -ne $taskExisting) {
        if ($taskExisting.Actions.Count -ne 1 -or $taskExisting.Actions[0].Execute -ne $taskPowerShell -or $taskExisting.Actions[0].Arguments -ne $taskArguments) {
            $taskReason = 'existing_task_conflict'; throw 'stop'
        }
    } else {
        $taskAction = New-ScheduledTaskAction -Execute $taskPowerShell -Argument $taskArguments -WorkingDirectory $taskProject
        $taskTrigger = New-ScheduledTaskTrigger -Once -At $taskAt
        $taskPrincipal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
        $taskSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 15) -MultipleInstances IgnoreNew
        Register-ScheduledTask -TaskName $taskName -Action $taskAction -Trigger $taskTrigger -Principal $taskPrincipal -Settings $taskSettings -Description 'One-day V0.3 T=2 morning predictions only; no betting' | Out-Null
    }
    $taskRegistered = Get-ScheduledTask -TaskName $taskName
    $taskInfo = Get-ScheduledTaskInfo -TaskName $taskName
    $taskSummary = [ordered]@{status='ok';kind='trial_task_ready';task_name=$taskName;scheduled_at_jst='2026-10-17T07:00:00+09:00';
        task_state=[string]$taskRegistered.State;next_run_time=$taskInfo.NextRunTime.ToString('o');requires_logged_on_user=$true;
        probe=$taskProbe;output_root=$taskRoot;ev_enabled=$false}
    $taskSetupPath = Join-Path $taskRoot ('setup-' + [guid]::NewGuid().ToString('N') + '.json')
    [IO.File]::WriteAllText($taskSetupPath, ($taskSummary | ConvertTo-Json -Depth 8), [Text.UTF8Encoding]::new($false))
    [pscustomobject]$taskSummary | ConvertTo-Json -Depth 8 -Compress
} catch {
    [pscustomobject]@{status='stopped';reason=$taskReason;stage='trial_setup'} | ConvertTo-Json -Compress
    exit 2
}
