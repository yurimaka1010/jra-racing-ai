param(
    [ValidatePattern('^\d{4}-\d{2}-\d{2}$')]
    [string]$RaceDate = '2026-10-04',
    [ValidatePattern('^[a-f0-9]{32}$')]
    [string]$ModelId = 'f1cce38718a4473ca1967b2252cb7997',
    [ValidatePattern('^[a-f0-9]{32}$')]
    [string]$MappingId = '258d31477be9478d83e6dc9a624bc00b'
)

# User-run only: reuse existing CLIs, hidden authentication, and local storage guards.
# This step makes probabilities/policy only. It does not invent observation times,
# accept final odds as decision prices, enable HR money conversion, or place bets.
$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent $PSScriptRoot
$taskRoot = 'C:\Users\acbd5\AppData\Local\JraRacingAI'
$taskPython = Join-Path $taskRoot 'venv\Scripts\python.exe'
$taskV01 = Join-Path $taskRoot 'ml-v01'
$taskV02 = Join-Path $taskRoot 'ml-v02'
$taskModel = Join-Path $taskV01 "models\$ModelId"
$taskMapping = Join-Path $taskV01 "schema\$MappingId\mapping.reviewed.json"
$script:taskLastReason = 'v02_preparation_stopped'
$script:taskStage = 'preflight'
$script:taskDiagnostics = $null

function Invoke-TaskCli {
    param([string]$Module, [string[]]$TaskArguments)
    $taskLines = & $taskPython -B -m $Module @TaskArguments
    $taskExitCode = $LASTEXITCODE
    try {
        $taskResult = ($taskLines -join "`n") | ConvertFrom-Json -ErrorAction Stop
    } catch {
        throw 'v02_cli_output_invalid'
    }
    if ($taskExitCode -ne 0 -or $taskResult.status -ne 'ok') {
        if ($taskResult.reason -match '^[a-z_]{1,64}$') {
            $script:taskLastReason = $taskResult.reason
        }
        # The Python CLI emits only fixed operation IDs and standard SQLSTATE.
        $script:taskDiagnostics = $taskResult.diagnostics
        throw 'v02_stopped_see_fixed_reason_code'
    }
    if ($taskResult.artifact_id -notmatch '^[a-f0-9]{32}$') {
        throw 'v02_artifact_invalid'
    }
    return $taskResult
}

try {
    if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
        throw 'local_python_missing'
    }
    $taskModelInfo = Get-Content -LiteralPath (Join-Path $taskModel 'model.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskDay = [datetime]::ParseExact($RaceDate, 'yyyy-MM-dd', [cultureinfo]::InvariantCulture)
    $taskFitEnd = [datetime]::ParseExact($taskModelInfo.split.validation.end, 'yyyy-MM-dd', [cultureinfo]::InvariantCulture)
    if ($taskDay -le $taskFitEnd) {
        throw 'invalid_period'
    }

    # Locate the original history by its recorded checksum, never by newest file.
    $taskHistoryCandidates = @(
        Get-ChildItem -LiteralPath (Join-Path $taskV01 'datasets') -Directory |
        Where-Object { $_.Name -match '^[a-f0-9]{32}$' -and -not ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) } |
        ForEach-Object {
            $taskManifestPath = Join-Path $_.FullName 'dataset.json'
            $taskCsvPath = Join-Path $_.FullName 'runners.csv'
            if ((Test-Path -LiteralPath $taskManifestPath -PathType Leaf) -and
                (Test-Path -LiteralPath $taskCsvPath -PathType Leaf)) {
                $taskManifest = Get-Content -LiteralPath $taskManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
                if ($taskManifest.complete -eq $true -and
                    $taskManifest.csv_sha256 -eq $taskModelInfo.dataset_sha256) {
                    $taskCsvPath
                }
            }
        }
    )
    if ($taskHistoryCandidates.Count -ne 1) {
        throw 'training_history_not_unique'
    }
    $taskHistory = $taskHistoryCandidates[0]

    $env:PYTHONPATH = Join-Path $taskProject 'src'
    $env:PGHOST = '127.0.0.1'
    $env:PGPORT = '5432'
    $env:PGDATABASE = 'pckeiba'
    $env:PGUSER = 'jra_audit_reader'

    # Freeze 10-minute / EV 5% / 100-yen policy before fetching evaluation rows.
    $script:taskStage = 'make_policy'
    $taskPolicy = Invoke-TaskCli 'jra_racing_ai.win_sim_cli' @(
        'make-policy', '--model', $taskModel, '--test-start', $RaceDate, '--test-end', $RaceDate
    )
    if ($taskPolicy.kind -ne 'policies') { throw 'v02_artifact_invalid' }
    $script:taskStage = 'export'
    $taskDataset = Invoke-TaskCli 'jra_racing_ai.ml_cli' @(
        'export-entries', '--execute-local', '--mapping', $taskMapping,
        '--start', $RaceDate, '--end', $RaceDate,
        '--max-rows', '2000', '--max-seconds', '60', '--statement-timeout-ms', '15000'
    )
    if ($taskDataset.purpose -ne 'prediction_only' -or $taskDataset.current_results_excluded -ne $true) {
        throw 'v02_artifact_invalid'
    }
    $taskEntries = Join-Path $taskV01 "datasets\$($taskDataset.artifact_id)\runners.csv"
    $script:taskStage = 'predict'
    $taskPredictions = Invoke-TaskCli 'jra_racing_ai.win_sim_cli' @(
        'predict-win', '--model', $taskModel, '--entries', $taskEntries, '--history', $taskHistory
    )
    if ($taskPredictions.kind -ne 'predictions') { throw 'v02_artifact_invalid' }
    [pscustomobject]@{
        status = 'ok'
        kind = 'v02_local_predictions_prepared'
        date = $RaceDate
        model_id = $ModelId
        dataset_id = $taskDataset.artifact_id
        predictions_id = $taskPredictions.artifact_id
        policy_id = $taskPolicy.artifact_id
        race_count = $taskPredictions.race_count
        runner_count = $taskPredictions.runner_count
        source_status_counts = $taskDataset.source_status_counts
        current_results_excluded = $true
        history_scope = 'original_training_dataset_recent_history_not_refreshed'
        next = 'simulate_after_time_and_payout_inputs_review'
    } | ConvertTo-Json -Compress
} catch {
    # Never echo PowerShell exception details, paths from data, or native output.
    if ($_.Exception.Message -in @('local_python_missing', 'invalid_period',
        'training_history_not_unique', 'v02_cli_output_invalid', 'v02_artifact_invalid')) {
        $script:taskLastReason = $_.Exception.Message
    }
    $taskFailure = [ordered]@{status='stopped';reason=$script:taskLastReason;stage=$script:taskStage}
    if ($null -ne $script:taskDiagnostics) { $taskFailure['diagnostics'] = $script:taskDiagnostics }
    [pscustomobject]$taskFailure | ConvertTo-Json -Compress
    exit 2
}
