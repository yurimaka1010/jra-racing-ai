from datetime import date
from ..stage5 import contracts as previous
from ..ml.safety import MLFailure

BASE=previous.ROOT.parent
ROOT=BASE/'stage6-development'
PROJECT=previous.PROJECT
S2,S4,S5=previous.S2,previous.S4,previous.ROOT/'evaluations/598c62b4dbe54142a2abe4a98b435b12'
VERSION='stage6_previous_corner_field_interaction_v1'
SOURCE_FORMULA='causal_groups_20261019_v2'
SOURCE_COLUMN='c_previous_corner4_percentile'
H_FEATURES=previous.H_FEATURES
STYLE=('style_proxy','style_missing')
PACE=('others_front_fraction','others_style_mean','others_style_known_fraction','style_pressure_interaction')
CONFIGS={'H_style':H_FEATURES+STYLE,'H_pace':H_FEATURES+STYLE+PACE}
YEARS=(2023,2024,2025)
START,END=date(2020,1,1),date(2025,9,30)
SEED,REPETITIONS=20261019,5000
PARAMETERS={'objective':'binary','metric':'binary_logloss','learning_rate':.03,'num_leaves':7,'max_depth':3,
    'min_data_in_leaf':100,'lambda_l2':10.,'boost_from_average':False,'feature_pre_filter':False,'num_threads':4,
    'seed':42,'deterministic':True,'force_col_wise':True,'verbosity':-1}


def guard(start,end):
    if not isinstance(start,date) or not isinstance(end,date) or start>end or start<START or end>END:
        raise MLFailure('invalid_split')


def split(year):
    if year not in YEARS:raise MLFailure('invalid_split')
    return previous.split(year)


def plan():
    return {'version':VERSION,'execution_approved':True,'hypothesis':'previous_corner_proxy_by_current_other_runner_composition',
        'development_reuse_not_independent_final_test':True,'processing_start':str(START),'processing_end':str(END),
        'sources':{'Stage2':str(S2),'Stage4':str(S4),'Stage5':str(S5)},'source_formula_version':SOURCE_FORMULA,
        'source_column':SOURCE_COLUMN,'history_lag_days':7,'style_formula':'(past_actual_starters-past_corner4)/(past_actual_starters-1)',
        'features':{k:list(v) for k,v in CONFIGS.items()},'front_threshold':2/3,
        'formulas':{'style_proxy':'saved_c_previous_corner4_percentile;finite_[0,1]_or_NaN',
            'style_missing':'1_if_style_missing_else_0','others_front_fraction':'known_others_s>=2/3_count/(current_roster_n-1);NaN_if_none_known',
            'others_style_mean':'mean_known_others_s;NaN_if_none_known','others_style_known_fraction':'known_others_count/(current_roster_n-1)',
            'style_pressure_interaction':'(own_s-.5)*others_front_fraction;NaN_if_either_missing'},
        'roster':'same_saved_T10_market_frame;self_excluded;optional_missingness_never_drops_rows',
        'timing':'publication_minute_end_before_T10;max_age300;unchanged_T30_history;PC_receipt_and_schedule_asof_unverified',
        'parameters':PARAMETERS,'rounds_max':300,'early_stopping_rounds':20,'fits_max':6,'boosting_rounds_max':1800,
        'failed_fits_count_toward_budget':True,'no_refit_completed_or_failed_unit':True,
        'offset':'logit(q)_in_train_and_early;raw_residual_plus_external_logit(q);sigmoid_then_race_normalize;T1',
        'calibration':'identity_T1_no_search','splits':{str(y):split(y) for y in YEARS},
        'fixed_population':{'races':6373,'runners':87601,'positive_labels':6373,'negative_labels':81228},
        'primary_metric':'equal_race_mean_within_race_AUC','difference_sign':'new_minus_comparator',
        'secondary_metrics':['pooled_ROC_AUC','race_log_loss','horse_log_loss','Brier','fractional_tie_top1'],
        'formal_comparisons':{'A':'H_pace_minus_H','B':'H_pace_minus_H_style'},
        'bootstrap':{'unit':'Monday_starting_week_all_venues','repetitions':REPETITIONS,'seed':SEED,
            'shared_draws':True,'formal_confidence':.975,'formal_quantiles':[.0125,.9875],'multiplicity':'Bonferroni_two_comparisons',
            'descriptive_confidence':.95,'reuse_Stage5_global_U_not_mean_week_AUC':True},
        'retain_all_conditions':{'A_within_delta_at_least':.002,'A_975_low_above':0.,'B_within_delta_at_least':.001,
            'B_975_low_above':0.,'positive_annual_vs_H_at_least':2,'no_annual_vs_H_delta_leq':-.002,
            'pooled_ROC_and_top1_at_least_H':True,'pooled_race_LL_and_Brier_at_most_H':True,'consistency_and_reload':True,'quality_tolerance':1e-12},
        'training_gate':'style_proxy_and_style_pressure_interaction_each_not_all_missing_or_constant_for_each_year_train',
        'H_style_is_control_never_promoted':True,'no_ROI_or_purchase_experiment':True,
        '2026_data_read':False,'no_DB_or_network':True,'no_task_or_deployment_change':True,'no_automatic_next_hypothesis':True}
