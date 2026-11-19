"""Versioned, local-only contracts for at most two new research families."""
from datetime import date
from pathlib import Path
import re

from ..ml import safety

BASE = Path(r'C:\Users\acbd5\AppData\Local\JraRacingAI')
ROOT = BASE / 'multiwager-research'
S2 = BASE / 'stage2-development/evaluations/0311a896a6684d4c8105591160cbf751'
S5 = BASE / 'stage5-development/evaluations/598c62b4dbe54142a2abe4a98b435b12'
PILOT = BASE / 'objective-pilot/evaluations/108a75ebe25d48a9a3421516f4b0c033'
VERSION = 'multiwager_research_20261022_v1'
WAGERS = ('place', 'wide', 'umaren')
FAMILIES = ('direct_official_place', 'shared_order_probability')
TRAINING_LOCK = BASE / 'research-training.lock'
SCHEMA = 'keiba-lab.multiwager.research.v1'
COLS = ('schema_version', 'race_id', 'date', 'venue', 'race_no', 'wager', 'horse_no_1', 'horse_no_2',
        'probability', 'pred_rank', 'model_id', 'source_model', 'evaluation_kind', 'source_stage',
        'source_sha256', 'generated_at', 'predicted_at', 'place_paid_positions', 'rank_reference_hit',
        'official_hit', 'official_payout_yen_per_100', 'missing_reason')


def day(value):
    try:
        d = date.fromisoformat(str(value))
    except (ValueError, TypeError):
        raise safety.MLFailure('invalid_date') from None
    if not date(2023, 1, 1) <= d <= date(2025, 9, 30):
        raise safety.MLFailure('invalid_split')
    return d


def identity(race, value):
    d = day(value)
    if (not isinstance(race, str) or not re.fullmatch('[0-9]{16}', race)
            or race[:8] != d.strftime('%Y%m%d') or not 1 <= int(race[8:10]) <= 10
            or not 1 <= int(race[10:12]) <= 99 or not 1 <= int(race[12:14]) <= 99
            or not 1 <= int(race[14:16]) <= 12):
        raise safety.MLFailure('invalid_identity')


def folder(identifier):
    if not isinstance(identifier, str) or not re.fullmatch('[0-9a-f]{32}', identifier):
        raise safety.MLFailure('invalid_identity')
    return safety.local_path(ROOT / 'evaluations' / identifier, root=ROOT)


def write(directory, name, value):
    safety.write_json(directory / name, value, root=ROOT)


def read(directory, name):
    return safety.read_json(directory / name, root=ROOT)


def training_plan():
    return {'version': VERSION, 'status': 'not_approved_not_started',
        'families': {'direct_official_place': {'objective': 'binary', 'targets': 'official_verified_place_winning_set',
            'features': 'ABCD39_plus_same_asof_O1_place_paid_positions', 'probabilities': 'independent_marginals_not_race_normalized',
            'calibration': 'logit_temperature_grid_0.75_1_1.25_2_calibration_period_only'},
            'shared_order_probability': {'objective': 'lambdarank', 'features': 'ABCD39',
            'label': 'finish1=3;2or3=2;4or5=1;other=0', 'probabilities': 'Plackett_Luce_of_exp_raw_scores_over_T',
            'calibration': 'T_grid_0.25_0.5_1_2_by_ordered_top3_NLL_calibration_period_only',
            'assumption': 'no_dead_heats_IIA;wide_and_umaren_share_one_joint_distribution'}},
        'new_family_count_max': 2, 'real_data_fits_max': 2, 'per_family_fit_max': 1,
        'failed_fit_consumes_budget': True, 'automatic_retry': False,
        'train': ['2023-01-01', '2024-08-31'], 'early_stopping': ['2024-09-08', '2024-10-31'],
        'calibration': ['2024-11-08', '2024-12-24'], 'evaluation': ['2025-01-01', '2025-09-30'],
        'evaluation_population': 'saved_2025_same_rosters;label_coverage_and_common_wager_sets_reported',
        'existing_2023_2024_annual_predictions': 'no_fit_reference_only_not_retrained',
        'max_rounds': 300, 'early_stopping_rounds': 20, 'CPU_threads': 4, 'GPU': False,
        'preparation_seconds_max': 300, 'fit_seconds_max_each': 600, 'whole_job_seconds_max': 1800,
        'memory_limit_GiB': 12, 'high_load_execution': 'sequential_exclusive_research_training_lock',
        'train_row_cap': 100000, 'early_and_calibration_row_cap_each': 10000, 'evaluation_row_cap': 30000,
        'data_gates': ['complete_exact_rosters', 'same_O1_pay_key_2_or_3_no_headcount_inference',
            'official_place_HR_normal_winners_and_flags_for_direct_target', 'no_top3_ties_without_official_combinations',
            '7_day_history_lag', 'no_current_results_or_final_odds_as_features'],
        'ROI': 'pending_official_payout_units_flags_versions_and_decision_time_quotes;no_odds_range_midpoints',
        'sealed_2026_read': False, 'Stage7_results_read': False, 'automatic_adoption': False,
        'requires': 'present_plan_and_explicit_fit_authorization_before_execute'}
