from pathlib import Path
from datetime import date
import hashlib
import json
import re

from ..pedigree_research import contracts as prior
from ..ml import safety

BASE=prior.BASE
ROOT=BASE/'pedigree-forward'
PROJECT=Path(__file__).resolve().parents[3]
PREP=prior.ROOT/'evaluations/7be65e783430433ebc7440838360ef30'
TRAIN=prior.ROOT/'evaluations/5426967702da406c85874027fa2afb31'
REFERENCE=prior.REFERENCE
RAW_PARENT=prior.enriched.RAW_PARENT
VERSION='pedigree3_prospective_T10_20261026_v1'
TASK='JraRacingAI_Pedigree3_Research_T10'
FIRST_DAY=date(2026,10,31)
NAMES=('control39','pedigree45','reference47')
MODEL_IDS={'control39':'faee8c8f0d5c428ab71a5c0a0b9d2687','pedigree45':'a380e94c0e09492eada7a26733e27f1a',
           'reference47':'b6888ec9550f437cbad817565a6c39cf'}
TEMPERATURE={'control39':1.25,'pedigree45':1.,'reference47':1.25}
read=prior.read
sha=prior.sha
clock=prior.clock


def write(path,value):safety.write_json(path,value,root=ROOT)


def folder(identifier):
    if not re.fullmatch('[a-f0-9]{32}',str(identifier)):raise safety.MLFailure('invalid_identity')
    return safety.local_path(ROOT/'evaluations'/identifier,root=ROOT)


def mkdir(path):
    path=safety.local_path(path,root=ROOT);path.mkdir(parents=True,exist_ok=True);return path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def prepare(supersedes_run_id=None):
    if supersedes_run_id is not None:
        previous=folder(supersedes_run_id)
        if not (previous/'preregistration.commit.json').is_file():raise safety.MLFailure('artifact_invalid')
    directory=safety.new_artifact('evaluations',root=ROOT)
    write(directory/'execution_scope.json',{'version':VERSION,'created_at':clock().isoformat(),'real_fits':0,
        'model_ids':MODEL_IDS,'temperatures':TEMPERATURE,'record_from':str(FIRST_DAY),
        'primary_metric':'within_race_auc','primary_contrast':'pedigree45_minus_control39',
        'metric_change_authorized_by':'human_user','metric_change_date':'2026-10-26',
        'purchase_rule':'R1_top1_win_100_yen_P3_only','R2_allowed':False,'shuffle_control':False,
        'threshold_values_approved':False,'operational_task_creation_approved':True,
        'H_Stage7_results_access':False,'deployment_to_formal_prediction_or_KEIBA_LAB':False,
        'supersedes_research_run_id':supersedes_run_id,
        'research_task_reconfiguration_authorized':supersedes_run_id is not None,
        'supersession_reason':'cancelled9_deleted0_delayed_history_fix' if supersedes_run_id else None})
    from ..pedigree_research.preparation import preservation
    write(directory/'protected_before.json',preservation())
    return {'status':'prepared','run_id':directory.name,'real_fits':0,'thresholds_pending':True}


def verify_protected(directory):
    from ..pedigree_research.preparation import preservation
    old=read(directory/'protected_before.json');now=preservation()
    for key in ('assets','protected_code'):
        if old['existing'][key]!=now['existing'][key]:raise safety.MLFailure('artifact_invalid')
    for path,value in old['references'].items():
        if sha(Path(path))!=value:raise safety.MLFailure('artifact_invalid')
    a,b=old['existing']['tasks'],now['existing']['tasks']
    if a.get('status')!='observed' or b.get('status')!='observed':raise safety.MLFailure('artifact_invalid')
    previous={v['name']:v for v in a['tasks']};current={v['name']:v for v in b['tasks']}
    scope=read(directory/'execution_scope.json')
    replace=scope.get('research_task_reconfiguration_authorized') is True and scope.get('supersedes_research_run_id')
    if any(current.get(k)!=v for k,v in previous.items() if not (replace and k==TASK)) or set(current)-set(previous)-{TASK}:
        raise safety.MLFailure('artifact_invalid')
    return {'existing_assets_unchanged':len(old['existing']['assets']),
        'existing_protected_code_unchanged':len(old['existing']['protected_code']),
        'additional_immutable_files_unchanged':len(old['references']),
        'preexisting_tasks_unchanged':True,'permitted_new_task':TASK,'new_fits':0}
