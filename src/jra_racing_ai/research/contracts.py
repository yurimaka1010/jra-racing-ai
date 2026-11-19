"""Versioned, bounded Stage 2 input/feature contracts. Legacy globals are untouched."""
from datetime import date
from pathlib import Path

from ..ml.features import FEATURES as LEGACY_FEATURES, CATEGORY_FEATURES as LEGACY_CATEGORIES
from ..ml.safety import MLFailure

ROOT = Path(r'C:\Users\acbd5\AppData\Local\JraRacingAI\stage2-development')
START = date(2015, 7, 2)
END = date(2025, 11, 30)
SEALED_START, SEALED_END = date(2026, 1, 1), date(2026, 9, 30)
FORMULA_VERSION = 'causal_groups_20261019_v2'
INPUT_VERSION = 'selected_RA_SE_v2'
GROUPS = {
    'A': ('a_previous_finish_percentile', 'a_recent3_finish_percentile'),
    'B': ('b_grade', 'b_condition', 'b_race_type', 'b_weight_type', 'b_previous_grade',
          'b_previous_condition', 'b_condition_transition', 'b_grade_transition'),
    'C': ('c_previous_late3f_seconds', 'c_previous_late3f_rank_percentile',
          'c_previous_late3f_minus_race_mean', 'c_recent3_late3f_rank_percentile',
          'c_previous_corner4_percentile', 'c_previous_adjacent_margin_lengths',
          'c_previous_margin_code', 'c_previous_additional_margin_present'),
    'D': ('d_jockey_starts_365d', 'd_jockey_smoothed_win_365d',
          'd_trainer_starts_365d', 'd_trainer_smoothed_win_365d'),
}
CATEGORIES = LEGACY_CATEGORIES + GROUPS['B'] + ('c_previous_margin_code',)
ALL_FEATURES = LEGACY_FEATURES + tuple(f for group in GROUPS.values() for f in group)
MARKET_FEATURES = ('market_probability', 'market_rank', 'market_log_odds', 'market_inverse_sum')
HISTORY_FEATURES = ('market_log_probability_change', 'market_rank_improvement',
                    'market_elapsed_minutes', 'market_history_missing')

# Logical alias -> (RA/SE, physical column, maximum raw bytes, availability role).
RAW_FIELDS = {
    'horse_number': ('se', 'umaban', 2, 'key'), 'horse_id': ('se', 'ketto_toroku_bango', 10, 'identity'),
    'finish_position': ('se', 'kakutei_chakujun', 2, 'label_and_delayed_history'),
    'surface': ('ra', 'track_code', 2, 'declared'), 'distance': ('ra', 'kyori', 4, 'declared'),
    'declared_runners': ('ra', 'toroku_tosu', 2, 'declared_registered_count_unchanged'),
    'actual_runners': ('ra', 'shusso_tosu', 2, 'delayed_history_only'),
    'frame': ('se', 'wakuban', 1, 'declared'), 'age': ('se', 'barei', 2, 'declared'),
    'sex': ('se', 'seibetsu_code', 1, 'declared'), 'weight_raw': ('se', 'futan_juryo', 3, 'declared_0.1kg'),
    'start_time': ('ra', 'hasso_jikoku', 4, 'reference_schedule'),
    'previous_start': ('ra', 'hasso_jikoku_henkomae', 4, 'reference_schedule_change'),
    'grade': ('ra', 'grade_code', 1, 'declared'),
    'previous_grade_revision': ('ra', 'grade_code_henkomae', 1, 'declared_change_unknown_time'),
    'condition_minimum_age': ('ra', 'kyoso_joken_code', 3, 'declared'),
    'condition_2': ('ra', 'kyoso_joken_code_2sai', 3, 'declared'),
    'condition_3': ('ra', 'kyoso_joken_code_3sai', 3, 'declared'),
    'condition_4': ('ra', 'kyoso_joken_code_4sai', 3, 'declared'),
    'condition_5': ('ra', 'kyoso_joken_code_5sai_ijo', 3, 'declared'),
    'race_type': ('ra', 'kyoso_shubetsu_code', 2, 'declared'),
    'weight_type': ('ra', 'juryo_shubetsu_code', 1, 'declared'),
    'jockey': ('se', 'kishu_code', 5, 'declared'),
    'previous_jockey_revision': ('se', 'kishu_code_henkomae', 5, 'declared_change_unknown_time'),
    'trainer': ('se', 'chokyoshi_code', 5, 'declared'),
    'anomaly': ('se', 'ijo_kubun_code', 1, 'population_and_delayed_history'),
    'late3f_raw': ('se', 'kohan_3f', 3, 'delayed_history_only_tenths_second'),
    'corner1': ('se', 'corner_1', 2, 'delayed_history_only'),
    'corner2': ('se', 'corner_2', 2, 'delayed_history_only'),
    'corner3': ('se', 'corner_3', 2, 'delayed_history_only'),
    'corner4': ('se', 'corner_4', 2, 'delayed_history_only'),
    'margin1': ('se', 'chakusa_code_1', 3, 'delayed_adjacent_margin_only'),
    'margin2': ('se', 'chakusa_code_2', 3, 'delayed_disqualification_supplement'),
    'margin3': ('se', 'chakusa_code_3', 3, 'delayed_disqualification_supplement'),
}


def guard_development(start, end):
    if not isinstance(start, date) or not isinstance(end, date) or start > end or end >= SEALED_START:
        raise MLFailure('invalid_split')
    if start < START:
        raise MLFailure('invalid_split')


def spec(groups=(), *, anchor=False, ability=True, history=False):
    if (not isinstance(groups, (tuple, list)) or len(set(groups)) != len(groups)
            or any(g not in GROUPS for g in groups) or (history and not anchor)):
        raise MLFailure('invalid_input')
    groups = tuple(g for g in GROUPS if g in groups)
    names = tuple(LEGACY_FEATURES) if ability else ()
    if ability:
        names += tuple(f for g in groups for f in GROUPS[g])
    if anchor:
        names += MARKET_FEATURES + (HISTORY_FEATURES if history else ())
    if not names:
        raise MLFailure('invalid_input')
    return {'format_version': 2, 'input_format': INPUT_VERSION, 'formula_version': FORMULA_VERSION,
        'groups': list(groups), 'mode': 'market_offset' if anchor else 'ability',
        'include_ability': ability, 'include_odds_history': history,
        'features': list(names), 'categorical_features': [f for f in names if f in CATEGORIES],
        'history_lag_days': 7, 'recent_weights_newest_first': [1., .5, .25],
        'person_window_days': 365, 'person_smoothing_strength': 20.,
        'current_actual_runners_used': False,
        'offset': {'scale': 'logit_normalized_inverse_odds', 'boost_from_average': False,
            'prediction': 'sigmoid(booster_raw_score+offset),then_race_normalize',
            'temperature': 1., 'best242_prediction_as_feature': False} if anchor else None}


def validate_spec(value):
    expected = spec(value.get('groups', []), anchor=value.get('mode') == 'market_offset',
                    ability=value.get('include_ability', True), history=value.get('include_odds_history', False))
    if value != expected:
        raise MLFailure('invalid_model')
    return value


def plan():
    return {'format_version': 2, 'primary_metric': 'race_winner_log_loss',
        'difference_sign': 'new_minus_comparator', 'source_start': START.isoformat(), 'source_end': END.isoformat(),
        'sealed_2026': {'start': SEALED_START.isoformat(), 'end': SEALED_END.isoformat(),
                       'read_labels': False, 'train': False, 'evaluate': False, 'tune': False},
        'main_split': {'train_start': '2016-07-02', 'train_end': '2024-11-30',
            'validation_start': '2024-12-08', 'validation_end': '2024-12-28',
            'calibration_start': '2025-01-11', 'calibration_end': '2025-02-02',
            'evaluation': [('2025-03-01', '2025-05-31'), ('2025-06-01', '2025-08-31'),
                           ('2025-09-01', '2025-11-30')]},
        'all_2025_periods_are_exploration': True, 'annual_development_years': [2023, 2024, 2025],
        'annual_split': 'prior_Aug31_train;prior_Sep08_Oct31_earlystop;prior_Nov08_Dec24_calibrate;year_Jan01_Sep30_evaluate',
        'ability_configs': ['control', 'A', 'B', 'AB', 'C', 'D', 'ABCD'],
        'anchor_configs': ['market_only', 'market_history', 'market_AB', 'market_AB_history', 'market_ABCD_history'],
        'execution_order': ['control','A','B','AB','market_only','market_history','market_AB','market_AB_history',
                            'C','D','ABCD','market_ABCD_history','conditional_leave_one_group_out'],
        'conditional_ablations_max': 4, 'all_group_promising': 'pooled_race_LL_lower_and_at_least_two_of_three_periods_lower_than_control',
        'annual_candidates_max_per_family': 1, 'search_rounds_max': 300, 'early_stopping_rounds': 20,
        'ability_parameters': {'objective':'binary','metric':'binary_logloss','learning_rate':.05,
            'num_leaves':15,'max_depth':5,'min_data_in_leaf':20,'feature_pre_filter':False,
            'num_threads':4,'seed':42,'deterministic':True,'force_col_wise':True,'verbosity':-1},
        'anchor_parameters': {'objective':'binary','metric':'binary_logloss','learning_rate':.03,
            'num_leaves':7,'max_depth':3,'min_data_in_leaf':100,'lambda_l2':10.,'boost_from_average':False,
            'feature_pre_filter':False,'num_threads':4,'seed':42,'deterministic':True,'force_col_wise':True,'verbosity':-1},
        'ability_calibration': {'method':'power_then_race_normalize','temperature_candidates':[.75,1.,1.25,2.]},
        'anchor_calibration': {'method':'identity_after_offset_and_race_normalization','temperature':1.},
        'market_recalibration_candidates': [.75,1.,1.25,2.], 'anchor_source_start': '2020-01-01',
        'quote_contract': {'minutes':10,'early_minutes':30,'precision':'minute_end_not_after_cutoff',
            'max_age_seconds':300,'schedule':'saved_RA_no_recorded_change_asof_unverified',
            'basis':'JRA_publication_posthoc_not_PC_observation'},
        'paper_policy': {'threshold':'.05','stake_yen':100,'max_bets_per_race':1,'settle':'official_normal_HR_only'},
        'bootstrap': {'repetitions':1000,'seed':20261019,'development_confidence':.95,
                      'primary_unit':'Monday_starting_race_week','reference_unit':'race'},
        'candidate_rule': 'select_min_pooled_development_race_LL;at_most_one_per_family;annual_mean_and_period_directions;otherwise_keep_baseline',
        'formal_final_plan': {'not_executed':True,'primary_comparisons':2,'interval_confidence_each':.975,
            'multiple_testing':'Bonferroni','recognition':'new_minus_comparator_CI_upper_strictly_below_zero',
            'ability_comparator':'same_split_legacy_feature_retrained_control',
            'anchor_comparator':'same_split_selected_market_information_only_control'}}
