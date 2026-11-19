"""Exclusive plans and models; only the bounded fit inventory is atomically updated."""
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import hashlib,json,os
from ..ml import safety
from ..win_sim.core import JST
from ..stage4 import store as old
from ..stage5 import store as s5
from . import contracts as c


def now():return datetime.now(JST).isoformat()
def read(directory,name):return safety.read_json(directory/name,root=c.ROOT)
def sha(path):return safety.digest(path,root=c.BASE)
def write(directory,name,value):
    path=directory/name
    if path.exists():
        if old.comparable(read(directory,name))!=old.comparable(value):raise safety.MLFailure('artifact_invalid')
        return
    safety.write_json(path,value,root=c.ROOT)


def inventory(directory,value=None):
    path=safety.local_path(directory/'experiment_inventory.json',root=c.ROOT)
    if value is not None:
        validate_inventory(value)
        if value['plan_sha256']!=sha(directory/'plan.json'):raise safety.MLFailure('artifact_invalid')
        temp=safety.local_path(directory/('inventory-'+uuid4().hex+'.tmp'),root=c.ROOT)
        try:
            with temp.open('x',encoding='utf-8',newline='\n') as f:
                json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
            temp.replace(path)
        finally:
            if temp.exists():temp.unlink()
    result=read(directory,path.name);validate_inventory(result)
    if result['plan_sha256']!=sha(directory/'plan.json'):raise safety.MLFailure('artifact_invalid')
    return result


def validate_inventory(value):
    records=value.get('records',[]);seen=set()
    if len(records)>6:raise safety.MLFailure('input_limit')
    for row in records:
        key=(row.get('config'),row.get('year'))
        if key in seen or key[0] not in c.CONFIGS or key[1] not in c.YEARS or row.get('status') not in ('started','complete','failed'):
            raise safety.MLFailure('artifact_invalid')
        seen.add(key)
    if value.get('fits_started')!=len(records) or value.get('fits_max')!=6:raise safety.MLFailure('artifact_invalid')


def old_code():
    paths=[p for p in (c.PROJECT/'src').rglob('*.py') if 'stage6' not in p.parts]
    for folder in ('scripts','tests','docs'):
        paths.extend(p for p in (c.PROJECT/folder).glob('*') if p.is_file() and 'stage6' not in p.name)
    paths.extend(c.PROJECT/n for n in ('README.md','requirements.txt','.gitignore') if (c.PROJECT/n).exists())
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def origin():
    manifest=safety.read_json(c.S2/'feature_manifest.json',root=c.S2)
    definition=safety.read_json(c.S2/'feature_definitions.v2.json',root=c.S2)
    source_plan=safety.read_json(c.S2/'plan.json',root=c.S2)
    if (manifest.get('formula_version')!=c.SOURCE_FORMULA or definition.get('formula_version')!=c.SOURCE_FORMULA
            or c.SOURCE_COLUMN not in manifest.get('features',[]) or manifest.get('sealed_2026_rows')!=0
            or definition.get('availability')!='all_history_available_only_at_past_race_date_plus_7_days'
            or definition.get('C',{}).get('corner4')!='(past_actual_starters-past_corner4_rank)/(past_actual_starters-1)'
            or not source_plan.get('feature_contracts') or any(v.get('history_lag_days')!=7 for v in source_plan['feature_contracts'].values())
            or source_plan.get('anchor_parameters')!=c.PARAMETERS):raise safety.MLFailure('invalid_input')
    files=[]
    for year in range(2020,2026):
        entries=[v for v in manifest['years'] if v['year']==year]
        if len(entries)!=1:raise safety.MLFailure('artifact_invalid')
        path=c.S2/'features'/f'{year}.csv'
        if entries[0]['file']!=path.name or sha(path)!=entries[0]['sha256']:raise safety.MLFailure('artifact_invalid')
        with path.open(encoding='utf-8') as f:header=f.readline().rstrip('\r\n').split(',')
        if any(header.count(n)!=1 for n in ('entry_key','date',c.SOURCE_COLUMN)):raise safety.MLFailure('invalid_input')
        files.append({'year':year,'file':str(path),'sha256':entries[0]['sha256']})
    return {'formula_version':c.SOURCE_FORMULA,'history_lag_days':7,'source_column':c.SOURCE_COLUMN,'files':files,
        'availability':definition['availability'],'corner_formula':definition['C']['corner4']}


def init(directory):
    if not s5.completed(c.S5,'report'):raise safety.MLFailure('artifact_invalid')
    refs=dict(s5.read(c.S5,'inputs.json')['references'])
    for name in ('plan.json','inputs.json','baseline.json','baseline_bootstrap.local.npz','candidate.json',
                 'development_report.json','development_report.md','completion.json','report.complete.json'):
        path=c.S5/name;refs[str(path)]=sha(path)
    for path in (c.S5/'models').glob('*/*'):
        if path.name in ('model.json','win.lgb.txt'):refs[str(path)]=sha(path)
    for name,expected in refs.items():
        if sha(Path(name))!=expected:raise safety.MLFailure('artifact_invalid')
    provenance=origin()
    write(directory,'inputs.json',{'references':refs,'old_code':old_code(),'tasks':old.task_configuration(),'C_origin':provenance})
    plan=dict(c.plan(),run_id=directory.name,created_at_jst=now(),inputs_sha256=sha(directory/'inputs.json'))
    write(directory,'plan.json',plan)
    inventory(directory,{'plan_sha256':sha(directory/'plan.json'),'fits_started':0,'fits_max':6,'records':[]})
    return {'run_id':directory.name,'plan_sha256':sha(directory/'plan.json'),'C_origin_verified':True,'fits_started':0,'fits_max':6}


def verify_plan(directory):
    plan=read(directory,'plan.json')
    if plan.get('version')!=c.VERSION or any(old.comparable(plan.get(k))!=old.comparable(v) for k,v in c.plan().items()):
        raise safety.MLFailure('artifact_invalid')
    if plan['inputs_sha256']!=sha(directory/'inputs.json'):raise safety.MLFailure('artifact_invalid')
    inventory(directory)
    return sha(directory/'plan.json')


def preserve(directory):
    verify_plan(directory);saved=read(directory,'inputs.json')
    for name,expected in saved['references'].items():
        if sha(Path(name))!=expected:raise safety.MLFailure('artifact_invalid')
    for name,expected in saved['old_code'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise safety.MLFailure('artifact_invalid')
    after=old.task_configuration();before=saved['tasks']
    match=after==before if after['status']==before['status']=='observed' else None
    if match is False:raise safety.MLFailure('artifact_invalid')
    return {'unchanged_reference_files':len(saved['references']),'changed_reference_files':0,'unchanged_old_project_files':len(saved['old_code']),
        'changed_old_project_files':0,'task_configuration_match':match,'task_mutations':0,'2026_data_processed':False,
        'old_final_plans_unchanged':True,'production_model_selection_changed':False}
