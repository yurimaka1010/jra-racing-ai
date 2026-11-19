from datetime import date, datetime, timedelta
from pathlib import Path
import hashlib
import re

from .. import tuner_reduction as old
from ..ml import safety
from ..research import contracts as features
from ..stage7 import contracts as h
from ..win_sim import core

BASE = old.BASE
ROOT = BASE / 'd39-generation'
PROJECT = old.PROJECT
S2 = h.S2
OLD_MODEL = old.OLD_RUN / 'models/19e0b6227c4b44349243207d248fef6e'
FOUR = old.folder('f64af46afda14783806a8dc9860714d1')
VERSION = 'current_D39_2026_training_authorized_20261023_v1'
FEATURE_SPEC = features.spec(('A','B','C','D'))


def clock():
    return datetime.now(core.JST)


def write(path, value):
    safety.write_json(path, value, root=ROOT)


def read(path):
    return safety.read_json(path, root=BASE)


def sha(path):
    return safety.digest(path, root=BASE)


def folder(identifier):
    if not re.fullmatch('[a-f0-9]{32}', str(identifier)):
        raise safety.MLFailure('invalid_identity')
    return safety.local_path(ROOT / 'evaluations' / identifier, root=ROOT)


def split(latest):
    if not date(2026,1,1) <= latest <= clock().date():
        raise safety.MLFailure('invalid_split')
    calibration_start = latest-timedelta(days=20)
    final_end = calibration_start-timedelta(days=8)
    early_start = final_end-timedelta(days=27)
    train_end = early_start-timedelta(days=8)
    return {'train': {'start':'2022-01-01','end':str(train_end)},
            'early_stopping': {'start':str(early_start),'end':str(final_end)},
            'final_train': {'start':'2022-01-01','end':str(final_end)},
            'calibration': {'start':str(calibration_start),'end':str(latest)}}


def guard_day(day, plan):
    if (plan.get('version') != VERSION or plan.get('2026_training_authorized') is not True
            or not date(2015,7,2) <= day <= date.fromisoformat(plan['latest_final_date'])):
        raise safety.MLFailure('invalid_split')


def approved(directory):
    plan = read(directory/'plan.json'); authorization = read(directory/'authorization.json')
    if (plan.get('version') != VERSION or plan.get('feature_spec') != FEATURE_SPEC
            or plan.get('2026_training_authorized') is not True
            or authorization.get('source') != 'human_user' or authorization.get('max_fits') != 9
            or authorization.get('plan_sha256') != sha(directory/'plan.json')):
        raise safety.MLFailure('explicit_execution_required')
    return plan


def preserve(directory):
    from ..stage4.store import task_configuration
    private = dict(read(FOUR/'preservation_before.json')['private_assets'])
    for path in FOUR.rglob('*'):
        if path.is_file():
            private[str(path)] = sha(path)
    for path in (h.MODEL/'model.json',h.MODEL/'win.lgb.txt', S2/'source_manifest.json',S2/'feature_manifest.json'):
        private[str(path)] = sha(path)
    # This new study can evolve without relaxing preservation of existing code.
    def prior(path):
        return ('d39_generation' not in path.parts and not path.name.startswith(('d39_current','test_d39_generation'))
                and path.name != 'd39_generation.md')
    source = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for sub in ('src','scripts','tests','docs')
              for path in (PROJECT/sub).rglob('*') if path.is_file() and '__pycache__' not in path.parts and prior(path)}
    write(directory/'preservation_before.json', {'private_assets':private,'project_files':source,'tasks':task_configuration()})


def verify(directory):
    from ..stage4.store import task_configuration
    before = read(directory/'preservation_before.json')
    for path,value in before['private_assets'].items():
        if sha(Path(path)) != value:
            raise safety.MLFailure('artifact_invalid')
    for path,value in before['project_files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != value:
            raise safety.MLFailure('artifact_invalid')
    if task_configuration() != before['tasks']:
        raise safety.MLFailure('artifact_invalid')
    return {'old_private_files_unchanged':len(before['private_assets']), 'old_source_files_unchanged':len(before['project_files']),
            'tasks_unchanged':True,'H_and_Stage7_changed':False}
