"""Only the four configurations explicitly approved on 2026-10-22."""
from datetime import date
from pathlib import Path
import re

from ..ml import safety

BASE = Path(r'C:\Users\acbd5\AppData\Local\JraRacingAI')
ROOT = BASE / 'objective-pilot'
S2 = BASE / 'stage2-development/evaluations/0311a896a6684d4c8105591160cbf751'
S6 = BASE / 'stage6-development/evaluations/719a0d3915194bc59b6e55eba2659576'
PROPOSAL = BASE / 'model-diagnostics/evaluations/7ebb5992bf2a4cde9bb12fa2419d0f9a/learning_proposal.json'
PROPOSAL_HASH = '08ce75fafd08cfff52e236c23f3ba03d8c165567cadef91db1959edfe0230dcb'
NAMES = ('paired_binary_control', 'multiclass_4', 'finish_percentile_regression', 'lambdarank_4')
PARTS = ('train', 'early_stopping', 'calibration', 'evaluation')
VERSION = 'approved_objective_pilot_20261022_v1'


def plan():
    if safety.digest(PROPOSAL, root=BASE) != PROPOSAL_HASH:
        raise safety.MLFailure('artifact_invalid')
    p = safety.read_json(PROPOSAL, root=BASE)
    if tuple(x['name'] for x in p['model_configurations']) != NAMES:
        raise safety.MLFailure('invalid_input')
    return p


def folder(identifier):
    if not isinstance(identifier, str) or not re.fullmatch('[0-9a-f]{32}', identifier):
        raise safety.MLFailure('invalid_identity')
    return safety.local_path(ROOT / 'evaluations' / identifier, root=ROOT)


def write(directory, name, value):
    safety.write_json(directory / name, value, root=ROOT)


def read(directory, name):
    return safety.read_json(directory / name, root=ROOT)


def day(value):
    try:
        d = date.fromisoformat(str(value))
    except (ValueError, TypeError):
        raise safety.MLFailure('invalid_date') from None
    if not date(2022, 1, 1) <= d <= date(2025, 9, 30):
        raise safety.MLFailure('invalid_split')
    return d


def part_for(value, p):
    d = day(value).isoformat()
    matches = [name for name, span in p['spans'].items() if span['start'] <= d <= span['end']]
    if len(matches) > 1:
        raise safety.MLFailure('invalid_split')
    return matches[0] if matches else None


def reserve(directory, name):
    """Exclusive reservation precedes train(); a crash/failed fit still consumes it."""
    if name not in NAMES:
        raise safety.MLFailure('invalid_input')
    existing = list(directory.glob('fit_*.started.json'))
    if len(existing) >= 4:
        raise safety.MLFailure('input_limit')
    write(directory, f'fit_{name}.started.json', {'name': name, 'consumes_fit_budget': True,
          'automatic_retry': False, 'proposal_sha256': PROPOSAL_HASH})
