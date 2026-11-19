from datetime import datetime
from pathlib import Path
from uuid import uuid4
import json
import re
from ..ml import safety,model as legacy
from ..research import io as oldio,experiment as oldexp,contracts as oldc
from ..win_sim import core
from . import contracts as c


def _comparable(value):
    if isinstance(value,dict):return {k:_comparable(v) for k,v in value.items() if k!='created_at_jst'}
    if isinstance(value,(list,tuple)):return [_comparable(v) for v in value]
    return value


def write(directory,name,value):
    path=safety.local_path(directory/name,root=c.ROOT)
    if path.exists():
        if _comparable(read(directory,name))!=_comparable(value):raise safety.MLFailure('artifact_invalid')
        return  # Exact verified resume; preserve the first artifact and timestamp.
    safety.write_json(path,value,root=c.ROOT)
def read(directory,name):return safety.read_json(directory/name,root=c.ROOT)
def digest(path):return safety.digest(path,root=c.ROOT)
def source(name):return oldio.read(c.SOURCE,name)
def frame(start,end):return oldexp.load_cache(c.SOURCE,start,end)
def trial(fold,parent):return oldexp.read_trial(c.SOURCE,fold,parent if fold=='main' else 'annual_'+{'ABCD':'ability','market_history':'market_control','market_ABCD_history':'anchor','control':'control'}[parent])


def origin(fold,parent):
    record=trial(fold,parent)
    return {'model_id':record['model_id'],'path':str(c.SOURCE/'models'/record['model_id']),
        'manifest_sha256':record['model_manifest_sha256'],'split':record['split'],
        'temperature':record['temperature'],'feature_manifest_sha256':record['feature_manifest_sha256']}


def provenance(directory,config,fold,*,parents=(),split=None):
    return {'run_id':directory.name,'config_id':config,'fold_id':str(fold),'version':c.VERSION,
        'source_models':list(parents),'source_manifests':{'features':safety.digest(c.SOURCE/'feature_manifest.json',root=oldc.ROOT),
            'market':safety.digest(c.SOURCE/'market_manifest.json',root=oldc.ROOT),
            'freeze':safety.digest(c.SOURCE/'candidate_freeze.json',root=oldc.ROOT)},
        'feature_version':oldc.FORMULA_VERSION,'split':split,
        'fit_label_available_through':split['calibration_end'] if split else None,
        'feature_history_available_through':'each_target_date_minus_7_days',
        'selection_metric':'race_winner_log_loss','library_versions':dict(legacy.versions(),scipy=__import__('scipy').__version__),
        'created_at_jst':datetime.now(core.JST).isoformat(),'development_only':True}


def write_predictions(directory,name,parts):
    write(directory,name,[[{'race_key':p.race,'probabilities':p.probabilities} for p in part] for part in parts])


def predictions(directory,name):
    return [[core.PredictionRace(tuple(p['race_key']),p['probabilities']) for p in part] for part in read(directory,name)]


def run_identity(directory):
    for path in (directory,*directory.parents):
        if path.parent.name=='evaluations' and re.fullmatch('[a-f0-9]{32}',path.name):return path.name
    return directory.name


def complete(directory,unit,files,summary):
    write(directory,unit+'.complete.json',{'manifest_version':2,'status':'complete','run_id':run_identity(directory),'unit':unit,
        'files':{str(p.relative_to(directory)) if p.is_relative_to(directory) else str(p):digest(p) for p in files},'summary':summary})


def completed(directory,unit,_seen=None):
    path=directory/(unit+'.complete.json')
    if not path.exists():return None
    seen=set() if _seen is None else set(_seen)
    if str(path) in seen:raise safety.MLFailure('artifact_invalid')
    seen.add(str(path))
    value=read(directory,path.name)
    if value.get('status')!='complete':raise safety.MLFailure('artifact_invalid')
    for rel,expected in value['files'].items():
        path=Path(rel)
        target=path if path.is_absolute() else directory/path
        if digest(target)!=expected:raise safety.MLFailure('artifact_invalid')
        if target.name.endswith('.complete.json'):
            completed(target.parent,target.name[:-len('.complete.json')],_seen=seen)
    # Read initial Stage 3 checkpoint manifests without rewriting their preserved bytes.
    actual=run_identity(directory)
    if value.get('manifest_version',1)==1 and value.get('run_id')==directory.name and actual!=directory.name:
        value=dict(value,run_id=actual,legacy_unit_scope=directory.name,checkpoint_identity_format='legacy_scope_resolved_by_run_path')
    return value


def verify_old(directory):
    before=read(directory,'preserved.json')
    for path,expected in before.items():
        if safety.digest(Path(path),root=oldc.ROOT.parent)!=expected:raise safety.MLFailure('artifact_invalid')
    return {'checked_files':len(before),'changed_files':0}


def init(directory):
    # Hash-only handling of old 2026 artifacts does not open their labels for evaluation.
    protected=dict(source('preserved.json'))
    for path in c.SOURCE.rglob('*'):
        if path.is_file():protected[str(path)]=safety.digest(path,root=oldc.ROOT)
    write(directory,'preserved.json',protected)
    plan=c.plan();plan['run_id']=directory.name;plan['created_at_jst']=datetime.now(core.JST).isoformat()
    write(directory,'plan.json',plan)
    for name in ('plan.json','candidate_freeze.json','feature_manifest.json','market_manifest.json','experiment_ledger.json','development_report.json'):
        source(name)
    return {'run_id':directory.name,'protected_files':len(protected),'sealed_2026_evaluated':False}
