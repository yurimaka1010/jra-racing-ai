"""One finite candidate; preparation never grants permission to fit."""
from datetime import date, datetime
from pathlib import Path
import hashlib
import re

from ..ml import safety
from ..research import contracts as prior
from ..d39_generation import contracts as old
from ..win_sim import core

BASE = old.BASE
ROOT = BASE / 'd39-next'
PARENT = old.folder('86ed126ffc8f4e2e85c8740eacf941f0')
VERSION = 'clean_race_rest_D39_20261025_v1'
EXTRA = ('e_days_since_actual_start', 'e_previous_start_gap_days',
         'e_recent3_start_gap_mean_days', 'e_rest_stage',
         'e_rest_first_past_starts', 'e_rest_first_smoothed_finish_percentile',
         'e_rest_stage_smoothed_finish_percentile', 'e_last_start_irregular')


def clock():
    return datetime.now(core.JST)


def read(path):
    return safety.read_json(path, root=BASE)


def write(path, value):
    safety.write_json(path, value, root=ROOT)


def sha(path):
    return safety.digest(path, root=BASE)


def folder(identifier):
    if not re.fullmatch('[a-f0-9]{32}', str(identifier)):
        raise safety.MLFailure('invalid_identity')
    return safety.local_path(ROOT / 'evaluations' / identifier, root=ROOT)


def spec():
    result = prior.spec(('A', 'B', 'C', 'D'))
    result.update(format_version=7, formula_version=VERSION,
                  input_format='selected_RA_SE_v2_clean_history',
                  features=[*result['features'], *EXTRA],
                  categorical_features=[*result['categorical_features'], 'e_rest_stage'],
                  irregular_policy='entire_race_excluded_from_labels_and_rank_history',
                  rest_threshold_days=84, rest_smoothing_strength=3.,
                  rest_prior_finish_percentile=.5,
                  actual_start_dates='official_2101_0_4_5_6_7;date_and_irregular_flag_only')
    return result


def plan():
    from ..tuner_pilot.experiment import versions
    inherited = read(PARENT / 'plan.json')
    # This specific new study inherits the already authorized 2026 training scope;
    # no old sealed evaluation or Stage 7 entry point is changed.
    return {'version': VERSION, 'feature_spec': spec(), 'source_parent': str(PARENT),
        'latest_source_date': inherited['latest_final_date'], 'spans': inherited['spans'],
        'parameters': dict(prior.plan()['ability_parameters'], objective='binary', metric='binary_logloss'),
        'search': {'lambda_l1': [1e-8, 10.], 'lambda_l2': [1e-8, 10.],
                   'num_leaves': [7, 63], 'feature_fraction': [.6, 1.],
                   'bagging_fraction': [.6, 1.], 'bagging_freq': [1, 7],
                   'min_child_samples': [10, 100], 'num_iterations': [100, 600, 100]},
        'optimizer': 'Optuna_TPESampler_seed42;all_eight_suggested_each_trial',
        'min_child_samples_alias': 'min_data_in_leaf removed before trial',
        'num_iterations_semantics': 'trial_round_cap;early_stopping20_selects_actual_rounds',
        'selection': 'minimum_early_stopping_binary_logloss;no_2025_or_forward_result_feedback',
        'calibration': {'grid': [.75, 1., 1.25, 2.], 'metric': 'race_log_loss',
                        'source': 'calibration_only;not_an_unused_evaluation'},
        'budget': {'tuner_fits_max': 8, 'final_refits_max': 1, 'actual_train_calls_max': 9,
                   'CPU_threads': 4, 'private_bytes_max': 12 * 1024 ** 3,
                   'search_seconds': 900, 'whole_seconds': 1800, 'automatic_retry': False},
        '2026_training_authorized_new_study_only': True,
        'evaluation': {'kind': 'future_saved_before_result_only', 'H_interim_results': False,
                       'training_or_calibration_races_are_unused_test': False,
                       'primary_metric': 'race_log_loss', 'difference': 'next_minus_comparator'},
        'deployment': {'authorized_when_complete': True, 'replace_existing_models': False,
            'reuse_four_recorder_inputs': True, 'cutoff_minutes': 10, 'save_grace_seconds': 60,
            'no_late_or_posthoc_fill': True, 'paper_win_yen': 100, 'EV_threshold': .05},
        'versions': versions(), 'new_fits_without_separate_approval': 0}


def approved(directory):
    p = read(directory / 'plan.json')
    try:
        a = read(directory / 'approval.json')
    except safety.MLFailure:
        raise safety.MLFailure('explicit_execution_required') from None
    if (p['version'] != VERSION or p['feature_spec'] != spec()
            or a.get('source') != 'human_user' or a.get('allow_real_fits') is not True
            or a.get('actual_train_calls_max') != 9
            or a.get('plan_sha256') != sha(directory / 'plan.json')):
        raise safety.MLFailure('explicit_execution_required')
    integrity = read(directory / 'clean_race_verification.json')
    if integrity.get('irregular_race_contamination') != 0 or not integrity.get('all_parts_reloaded'):
        raise safety.MLFailure('artifact_invalid')
    return p


def preservation():
    from ..stage4.store import task_configuration
    protected = dict(read(PARENT / 'four_addendum_plan.json')['references'])
    for name in ('plan.json', 'forward_plan.json', 'forward_four_plan.json', 'four_addendum_plan.json'):
        path = PARENT / name
        protected[str(path)] = sha(path)
    project = Path(__file__).resolve().parents[3]
    code = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for root in (project / 'src/jra_racing_ai/stage7', project / 'src/jra_racing_ai/d39_generation')
            for p in root.glob('*.py')}
    return {'assets': protected, 'protected_code': code, 'tasks': task_configuration()}


def verify_preservation(before):
    after = preservation()
    if before != after:
        raise safety.MLFailure('artifact_invalid')
    return {'old_assets_unchanged': len(before['assets']),
            'protected_code_unchanged': len(before['protected_code']), 'tasks_unchanged': True}
