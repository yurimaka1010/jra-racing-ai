param([Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{32}$')][string]$RunId,
      [ValidateSet('Prepare','Train')][string]$Mode='Prepare')
$ErrorActionPreference='Stop'
$taskBase='C:\Users\acbd5\AppData\Local\JraRacingAI'
$taskPython=Join-Path $taskBase 'venv\Scripts\python.exe'
$taskFolder=Join-Path $taskBase ('multiwager-research\evaluations\'+$RunId)
$taskPointer=[IntPtr]::Zero;$taskSecure=$null
try {
    $env:PYTHONUTF8='1'
    if($Mode -eq 'Prepare'){
        $taskCredential=Join-Path $taskBase 'trial-20261017\credentials\db-password.dpapi'
        $taskCursor=$taskCredential
        while($taskCursor){
            if((Test-Path -LiteralPath $taskCursor) -and ((Get-Item -LiteralPath $taskCursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)){throw 'unsafe_path'}
            $taskNext=Split-Path -Parent $taskCursor;if($taskNext -eq $taskCursor){break};$taskCursor=$taskNext
        }
        $taskSecure=(Get-Content -LiteralPath $taskCredential -Raw -Encoding ASCII).Trim()|ConvertTo-SecureString
        $taskPointer=[Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecure)
        $env:PGPASSWORD=[Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskPointer)
        $env:PGHOST='127.0.0.1';$env:PGPORT='5432';$env:PGDATABASE='pckeiba';$env:PGUSER='jra_audit_reader'
        & $taskPython -B (Join-Path $PSScriptRoot 'multiwager_research.py') prepare-inputs --run-id $RunId
    } else {
        Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
        & $taskPython -B (Join-Path $PSScriptRoot 'multiwager_research.py') train --run-id $RunId --training-manifest (Join-Path $taskFolder 'training_inputs.json')
    }
    exit $LASTEXITCODE
} catch {
    [ordered]@{status='stopped';reason='local_research_start_failed'}|ConvertTo-Json -Compress
    exit 1
} finally {
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
    if($taskPointer -ne [IntPtr]::Zero){[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskPointer)}
    if($null -ne $taskSecure){$taskSecure.Dispose()}
}
