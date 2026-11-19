from pathlib import Path
import re
from ..d39_next import contracts as prior
from ..ml import safety

BASE = prior.BASE
ROOT = BASE / 'd39-enrichment'
PARENT = prior.folder('b803140f84ac4ee39d9decc408e85f2a')
RAW_PARENT = prior.PARENT
POST = BASE / 'post-step2/evaluations/2d8fd55592234b4a891974d03ee98ffc'
VERSION = 'D39_verified_time_going_20261025_v1'
clock = prior.clock
read = prior.read
sha = prior.sha


def write(path, value):
    safety.write_json(path, value, root=ROOT)


def folder(identifier):
    if not re.fullmatch('[a-f0-9]{32}', str(identifier)):
        raise safety.MLFailure('invalid_identity')
    return safety.local_path(ROOT / 'evaluations' / identifier, root=ROOT)


def spec():
    value = prior.spec()
    value.update(format_version=8, formula_version=VERSION,
        input_format='clean_RA_SE_plus_reviewed_time_and_verified_going',
        previous_time_profile='conditioned_previous_time_v1;latest_valid_same_distance_exact_track;7day_lag',
        going_profile='WE_initial_and_unbroken_changes_before_cutoff;live_PC_observed_RA_only',
        final_historical_RA_going_used=False)
    return value


def plan():
    previous = read(PARENT / 'training_report.json')
    p = read(PARENT / 'plan.json')
    return {'version': VERSION, 'feature_spec': spec(), 'source_parent': str(PARENT),
        'spans': p['spans'], 'source_end': p['latest_source_date'],
        'parameters': previous['parameters'], 'rounds': previous['selected_rounds'],
        'calibration': p['calibration'], 'budget': {'actual_train_calls_max': 1,
            'CPU_threads': 4, 'private_bytes_max': 12 * 1024**3, 'whole_seconds': 900,
            'automatic_retry': False, 'parameter_search_fits': 0},
        'approval_required': True, '2026_training_authorized_this_candidate_only': True,
        'comparison': 'frozen_next47;identical_clean_cohort;calibration_diagnostic_only;future_formal_records',
        'deployment_authorized_when_ready': True, 'H_or_Stage7_outcomes_used': False,
        'going_missing_never_drops_races': True, 'historical_PC_receipt_unknown': True,
        'historical_WE_basis': 'official_publication_replay_not_genuine_forward_record'}


def approved(directory):
    p = read(directory / 'plan.json')
    try:
        a = read(directory / 'approval.json')
    except safety.MLFailure:
        raise safety.MLFailure('explicit_execution_required') from None
    if (p['version'] != VERSION or p['feature_spec'] != spec()
            or a.get('source') != 'human_user' or a.get('allow_real_fits') is not True
            or a.get('actual_train_calls_max') != 1 or a.get('plan_sha256') != sha(directory / 'plan.json')):
        raise safety.MLFailure('explicit_execution_required')
    return p
