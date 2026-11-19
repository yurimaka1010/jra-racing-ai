from pathlib import Path
from ..stage4 import contracts as s4
from ..ml.safety import MLFailure

S4=s4.ROOT/'evaluations/2f26b6573d2249d38ad96b4456f89c0f'
S2,S3=s4.S2,s4.S3
ROOT=s4.BASE/'stage5-development'
PROJECT=s4.PROJECT
VERSION='stage5_win_discrimination_place_v1'
YEARS=s4.YEARS
H_FEATURES=s4.H_FEATURES
EXTRA=('place_relative_support','place_log_range','place_missing')
FEATURES=H_FEATURES+EXTRA
SEED,REPETITIONS=20261019,5000
guard=s4.guard
def split(year):
    if year not in YEARS:raise MLFailure('invalid_split')
    value=s4.h_split(year);value['evaluation']=[(f'{year}-01-01',f'{year}-09-30')];return value


def plan():
    return {'version':VERSION,'approved_by_user':True,'development_reuse_not_independent_final_test':True,
        'processing_start':'2020-01-01','processing_end':'2025-09-30','evaluation_years':list(YEARS),'evaluation_months':'Jan_Sep',
        'fixed_races':6373,'fixed_runners':87601,'positive_labels':6373,'negative_labels':81228,
        'primary_metric':'pooled_horse_binary_ROC_AUC','score':'stored_final_continuous_probability_no_reprocessing',
        'additional_metrics':['equal_race_mean_within_race_AUC','race_winner_log_loss','horse_log_loss','Brier','fractional_tie_top1_accuracy'],
        'AUC_progress_target':{'threshold':.80,'rule':'strictly_greater_before_rounding','not_top1_accuracy':True},
        'reference':'H_fixed_not_selected_by_AUC','existing_highest_AUC':'report_name_and_score_reference_only_not_formal_superiority',
        'bootstrap':{'unit':'Monday_starting_week_all_venues','seed':SEED,'repetitions':REPETITIONS,'confidence':.95,
            'exact_AUC':'recompute_global_U_statistic_with_week_multiplicities;not_mean_week_AUC',
            'implementation':'precompute_cross_week_positive_negative_pair_counts_with_half_score_ties;each_draw_w_transpose_U_w/(weighted_P*weighted_N)',
            'same_week_draws_for_all_models_and_all_metrics':True,'no_Stage4_three_comparison_interval_transfer':True},
        'new_hypothesis':'H_plus_same_T10_O1_place_relative_price_support','new_configs':['H_place'],'new_LightGBM_fits_max':3,
        'place_features':list(FEATURES),'formulas':{'place_relative_support':'log(win_odds)-(log(place_lower)+log(place_upper))/2',
            'place_log_range':'log(place_upper)-log(place_lower)','place_missing':'whole_race_1_if_any_runner_invalid;both_numeric_NaN_for_all'},
        'place_terms_are_prices_not_probabilities':True,'place_2_or_3_payees':'verify_same_O1_place_pay_key;separate_coverage_and_missing_reasons;no_new_feature',
        'advance_to_fit':'verified_semantics_and_timing;each_fold_each_train_early_cal_eval_complete_race_coverage>=.80;training_numeric_features_not_all_constant_or_missing',
        'fit':'same_Stage2_anchor_parameters;binary;logit_market_init_train_and_validation;raw_residual_plus_logit_at_predict;sigmoid_then_race_normalize;T1;300rounds;early20',
        'selection':'H_place_minus_H_only;no_ROI_or_purchase_count',
        'retain_all_conditions':{'pooled_AUC_difference_at_least':.002,'AUC_difference_95_CI_low_strictly_above':0.,
            'positive_AUC_years_at_least':2,'no_annual_AUC_difference_leq':-.002,
            'pooled_within_race_AUC_and_top1_at_least_H':True,'pooled_race_LL_and_Brier_at_most_H':True,
            'same_population_timing_labels_and_reload_valid':True,'floating_tolerance':1e-12},
        'rejection_wording':'distinguish_measured_AUC_improvement_from_failure_of_overall_retention_conditions',
        'stop_without_substitute_if_gate_fails':True,'no_2026_read_train_select_evaluate_or_final_entry':True,
        'old_outputs_and_morning_tasks_unchanged':True,'no_ROI_analysis_or_betting_rule_changes':True,
        'no_new_packages_or_external_requests':True}
