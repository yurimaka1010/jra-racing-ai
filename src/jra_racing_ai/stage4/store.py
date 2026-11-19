from datetime import datetime
from pathlib import Path
import hashlib,json,re,subprocess
from ..ml import safety,model as legacy
from ..win_sim import core
from . import contracts as c


def read(directory,name):return safety.read_json(directory/name,root=c.ROOT)
def digest(path):return safety.digest(path,root=c.ROOT)
def comparable(value):
    if isinstance(value,dict):return {k:comparable(v) for k,v in value.items() if k!='created_at_jst'}
    if isinstance(value,(list,tuple)):return [comparable(v) for v in value]
    return value
def write(directory,name,value):
    target=safety.local_path(directory/name,root=c.ROOT)
    if target.exists():
        if comparable(read(directory,name))!=comparable(value):raise safety.MLFailure('artifact_invalid')
        return
    safety.write_json(target,value,root=c.ROOT)


def run_id(directory):
    for path in (directory,*directory.parents):
        if path.parent.name=='evaluations' and re.fullmatch('[a-f0-9]{32}',path.name):return path.name
    return directory.name


def complete(directory,unit,files,summary):
    write(directory,unit+'.complete.json',{'format_version':1,'run_id':run_id(directory),'status':'complete','unit':unit,
        'files':{str(p):digest(p) for p in files},'summary':summary})


def completed(directory,unit,_seen=None):
    path=directory/(unit+'.complete.json')
    if not path.exists():return None
    seen=set() if _seen is None else set(_seen)
    if str(path) in seen:raise safety.MLFailure('artifact_invalid')
    seen.add(str(path));value=read(directory,path.name)
    if value.get('status')!='complete' or value.get('run_id')!=run_id(directory):raise safety.MLFailure('artifact_invalid')
    for text,expected in value['files'].items():
        target=Path(text)
        if digest(target)!=expected:raise safety.MLFailure('artifact_invalid')
        if target.name.endswith('.complete.json'):completed(target.parent,target.name[:-len('.complete.json')],seen)
    return value


_checked={}
def check_reference(directory,path):
    key=(str(directory),str(path))
    if key in _checked:return
    refs=read(directory,'references.json')
    expected=refs.get(str(path))
    if expected is None:raise safety.MLFailure('artifact_invalid')
    root=c.S2 if path==c.S2 or c.S2 in path.parents else c.S3
    if safety.digest(path,root=root)!=expected:raise safety.MLFailure('artifact_invalid')
    _checked[key]=expected


def ref_json(directory,path):
    check_reference(directory,path)
    root=c.S2 if c.S2 in path.parents else c.S3
    return safety.read_json(path,root=root)


def provenance(directory,config,fold,split=None,parents=()):
    return {'version':c.VERSION,'run_id':run_id(directory),'config_id':config,'fold_id':str(fold),
        'split':split,'parents':list(parents),'reference_manifest_sha256':digest(directory/'references.json'),
        'source_feature_version':'causal_groups_20261019_v2','selection_metric':'race_winner_log_loss',
        'fit_labels_available_through':split.get('calibration_end',split.get('validation_end')) if split else None,
        'history_lag_days':7,'library_versions':dict(legacy.versions(),scipy=__import__('scipy').__version__),
        'created_at_jst':datetime.now(core.JST).isoformat(),'development_only':True,'processing_end':c.END.isoformat()}


def task_configuration():
    script=r"""$ErrorActionPreference='Stop'
$items=@(Get-ScheduledTask | Where-Object {$_.TaskName -like '*JraRacingAI*'} | Sort-Object TaskPath,TaskName | ForEach-Object {
    $xml=Export-ScheduledTask -TaskName $_.TaskName -TaskPath $_.TaskPath
    $bytes=[Text.Encoding]::UTF8.GetBytes($xml)
    $sha=[Security.Cryptography.SHA256]::Create()
    $hash=([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-','').ToLowerInvariant()
    $sha.Dispose()
    [ordered]@{name=$_.TaskName;path=$_.TaskPath;configuration_sha256=$hash}
})
ConvertTo-Json -InputObject $items -Compress
"""
    try:
        result=subprocess.run([r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe','-NoProfile','-NonInteractive','-Command',script],
            capture_output=True,text=True,timeout=30,check=False)
        if result.returncode:return {'status':'unavailable','reason':'task_configuration_read_unavailable','mutations':0}
        return {'status':'observed','tasks':json.loads(result.stdout),'mutations':0}
    except Exception:return {'status':'unavailable','reason':'task_configuration_read_unavailable','mutations':0}


def old_code():
    paths=[p for p in (c.PROJECT/'src').rglob('*.py') if 'stage4' not in p.parts]
    paths.extend(p for p in (c.PROJECT/'scripts').glob('*') if p.is_file() and not p.name.startswith('stage4'))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def init(directory):
    refs={}
    paths=[c.S2/name for name in ('plan.json','candidate_freeze.json','feature_manifest.json','market_manifest.json','source_manifest.json')]
    for year in range(2020,2026):
        paths.append(c.S2/'features'/f'{year}.csv');paths.extend(sorted((c.S2/'market').glob(f'{year}-*.local.json')))
    for year in c.YEARS:
        for config in ('annual_market_control','annual_ability'):
            paths.append(c.S2/'trials'/f'{year}_{config}.json');paths.append(c.S2/'trials'/f'{year}_{config}.predictions.local.json')
            meta=safety.read_json(paths[-2],root=c.S2)
            paths.extend(c.S2/'models'/meta['model_id']/name for name in ('model.json','win.lgb.txt'))
        paths.extend([c.S3/'integration'/f'{year}_predictions.local.json',c.S3/'integration'/f'{year}_results.json',
            c.S3/'E_trials'/f'{year}.json',c.S3/'E_trials'/f'{year}.predictions.local.json'])
    for year in range(2021,2026):paths.extend(c.S3/'oof'/f'{year}{suffix}' for suffix in ('.csv','.complete.json'))
    paths.extend(c.S3/name for name in ('plan.json','candidate_freeze.json','final_evaluation_plan.json','oof_origins.json','experiment_ledger.json'))
    paths.append(c.S2/'final_evaluation_plan.json')
    for path in sorted(set(paths)):
        root=c.S2 if c.S2 in path.parents else c.S3
        refs[str(path)]=safety.digest(path,root=root)
    write(directory,'references.json',refs);write(directory,'old_code_hashes.json',old_code());write(directory,'tasks_before.json',task_configuration())
    plan=c.plan();plan.update(run_id=directory.name,created_at_jst=datetime.now(core.JST).isoformat())
    write(directory,'plan.json',plan)
    return {'reference_files':len(refs),'plan_fixed_before_fit':True,'tasks_not_changed':True,'processing_end':c.END.isoformat()}


def verify_old(directory):
    refs=read(directory,'references.json')
    for path,expected in refs.items():
        root=c.S2 if c.S2 in Path(path).parents else c.S3
        if safety.digest(Path(path),root=root)!=expected:raise safety.MLFailure('artifact_invalid')
    before=read(directory,'old_code_hashes.json')
    if before!=old_code():raise safety.MLFailure('artifact_invalid')
    tasks=task_configuration();start=read(directory,'tasks_before.json')
    match=tasks==start if tasks['status']=='observed' and start['status']=='observed' else None
    if match is False:raise safety.MLFailure('artifact_invalid')
    return {'reference_files':len(refs),'changed_reference_files':0,'old_code_files':len(before),'changed_old_code_files':0,
        'task_configuration_match':match,'task_observation':tasks,'scheduler_mutations':0}
