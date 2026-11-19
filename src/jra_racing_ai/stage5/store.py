"""Small exclusive Stage 5 records; all legacy reads keep their original roots."""
from pathlib import Path
from datetime import datetime
import hashlib,json
from ..ml import safety
from ..win_sim import core
from ..stage4 import store as previous
from . import contracts as c


def read(directory,name):return safety.read_json(directory/name,root=c.ROOT)
def sha(path,root=None):return safety.digest(path,root=c.ROOT if root is None else root)
def write(directory,name,value):
    if (directory/name).exists():
        if previous.comparable(read(directory,name))!=previous.comparable(value):raise safety.MLFailure('artifact_invalid')
        return
    safety.write_json(directory/name,value,root=c.ROOT)


def complete(directory,unit,files,summary):
    write(directory,unit+'.complete.json',{'run_id':directory.name,'files':{str(p):sha(p) for p in files},'summary':summary,'status':'complete'})


def completed(directory,unit):
    if not (directory/(unit+'.complete.json')).exists():return None
    doc=read(directory,unit+'.complete.json')
    if doc['run_id']!=directory.name or doc['status']!='complete':raise safety.MLFailure('artifact_invalid')
    for name,digest in doc['files'].items():
        if sha(Path(name))!=digest:raise safety.MLFailure('artifact_invalid')
    return doc['summary']


def old_code():
    paths=[p for p in (c.PROJECT/'src').rglob('*.py') if 'stage5' not in p.parts]
    paths.extend(p for p in (c.PROJECT/'scripts').glob('*') if p.is_file() and not p.name.startswith('stage5'))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def snapshot(directory):
    if not previous.completed(c.S4,'freeze'):raise safety.MLFailure('artifact_invalid')
    # Reuse the validated immutable reference manifest, without calling old run/fit/freeze.
    refs=previous.read(c.S4,'references.json');paths=[]
    for folder,names in ((c.S4,('freeze.complete.json','accuracy.json','accuracy_candidate_freeze.json','candidate_freeze.json','references.json','oof_origins.json')),
                         (c.S2,('candidate_freeze.json','final_evaluation_plan.json')),
                         (c.S3,('candidate_freeze.json','final_evaluation_plan.json'))):paths.extend(folder/n for n in names)
    for year in c.YEARS:
        paths.extend(c.S4/'direct'/f'{year}_{name}.json' for name in ('selection','predictions.local'))
        paths.extend([c.S4/'H60'/f'{year}_predictions.local.json',c.S4/'oof'/f'{year}.csv',c.S4/'oof'/f'{year}.complete.json'])
    # Exact files declared in the Stage 4 completion tree are read-only inputs.
    seen=set()
    def walk(path):
        if path in seen:return
        seen.add(path);value=safety.read_json(path,root=c.S4)
        for name in value.get('files',{}):
            p=Path(name);paths.append(p)
            if p.name.endswith('.complete.json'):walk(p)
    walk(c.S4/'freeze.complete.json')
    for path in sorted(set(paths)):
        root=c.S4 if c.S4 in path.parents else c.S2 if c.S2 in path.parents else c.S3
        refs[str(path)]=sha(path,root)
    for name,expected in refs.items():
        if sha(Path(name),c.ROOT.parent)!=expected:raise safety.MLFailure('artifact_invalid')
    write(directory,'inputs.json',{'references':refs,'old_code':old_code(),'tasks':previous.task_configuration(),
        'source_runs':[str(c.S2),str(c.S3),str(c.S4)],'old_stage4_loader_has_no_write_side_effects':True})
    write(directory,'plan.json',dict(c.plan(),created_at_jst=datetime.now(core.JST).isoformat()))
    return {'references':len(refs),'plan_fixed_before_new_fit':True,'LightGBM_fits':0}


def preserve(directory):
    saved=read(directory,'inputs.json')
    for name,expected in saved['references'].items():
        if sha(Path(name),c.ROOT.parent)!=expected:raise safety.MLFailure('artifact_invalid')
    code=saved['old_code'];changed=[p for p,h in code.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    if changed:raise safety.MLFailure('artifact_invalid')
    tasks=previous.task_configuration();before=saved['tasks']
    match=tasks==before if tasks['status']==before['status']=='observed' else None
    if match is False:raise safety.MLFailure('artifact_invalid')
    return {'unchanged_reference_files':len(saved['references']),'unchanged_old_code_files':len(code),
        'changed_old_files':0,'task_configuration_match':match,'task_mutations':0,'2026_data_processed':False,'old_final_plans_unchanged':True}
