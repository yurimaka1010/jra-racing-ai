from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4
import hashlib,json,os,re
from ..ml import safety
from ..stage4 import store as prior
from ..win_sim import core
from . import contracts as c

def read(path):return safety.read_json(path,root=c.BASE)
def sha(path):return safety.digest(path,root=c.BASE)
def write(path,value):safety.write_json(path,value,root=c.ROOT)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
def directory(run):
    if not re.fullmatch('[a-f0-9]{32}',run):raise safety.MLFailure('invalid_input')
    return safety.local_path(c.ROOT/'evaluations'/run,root=c.ROOT)
def mkdir(path):
    path=safety.local_path(path,root=c.ROOT);path.mkdir(parents=True,exist_ok=True);return path

def event(folder,kind,reason=None,**values):
    p=mkdir(folder/'events')/(c.clock().strftime('%Y%m%dT%H%M%S%f')+'_'+uuid4().hex[:8]+'.json')
    write(p,{'kind':kind,'reason':reason,'recorded_at':c.clock().isoformat(),**values});return p

@contextmanager
def lock(folder):
    path=safety.local_path(folder/'recorder.lock',root=c.ROOT)
    try:fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    except FileExistsError:raise safety.MLFailure('artifact_exists') from None
    try:
        with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
        yield
    finally:path.unlink(missing_ok=True)

def fixed_model(plan):
    from ..research import model
    verify_refs(plan)
    bundle=model.load(c.MODEL,root=c.S2);meta=bundle['metadata'];spec=meta['feature_spec']
    if (tuple(spec['features'])!=c.H_FIELDS or spec['include_ability'] or spec['mode']!='market_offset'
            or meta['calibration']['temperature']!=1. or meta['split']['calibration_end']>='2026-01-01'):
        raise safety.MLFailure('invalid_model')
    return bundle

def refs():
    paths=[c.MODEL/'model.json',c.MODEL/'win.lgb.txt',c.S6/'plan.json',c.S6/'candidate_freeze.json',
           c.S6/'development_report.json',c.S2/'trials/2025_annual_market_control.json',c.S5/'place_mapping.json']
    return {str(p):sha(p) for p in paths}

def verify_refs(plan):
    for path,expected in plan['references'].items():
        if sha(Path(path))!=expected:raise safety.MLFailure('artifact_invalid')

def old_code():
    paths=[p for p in (c.PROJECT/'src').rglob('*.py') if 'stage7' not in p.parts]
    paths.extend(p for p in (c.PROJECT/'scripts').iterdir() if p.is_file() and not p.name.startswith('stage7'))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def verify_old(folder):
    before=read(folder/'old_code_hashes.json')
    if old_code()!=before:raise safety.MLFailure('artifact_invalid')
    plan=read(folder/'forward_plan.json');verify_refs(plan)
    original=read(folder/'tasks_before.json');current=prior.task_configuration()
    matched=None
    if original['status']==current['status']=='observed':
        old={x['name']:x for x in original['tasks']};new={x['name']:x for x in current['tasks']}
        matched=all(new.get(n)==x for n,x in old.items())
        if not matched:raise safety.MLFailure('artifact_invalid')
        if set(new)-set(old)-{c.TASK}:raise safety.MLFailure('artifact_invalid')
    return {'old_source_files':len(before),'changed_old_source_files':0,'changed_frozen_references':0,
            'preexisting_tasks_unchanged':matched,'permitted_new_task':c.TASK,'fits_executed':0}
