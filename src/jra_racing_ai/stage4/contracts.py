from datetime import date
from pathlib import Path
from ..ml.safety import MLFailure
from ..research import contracts as previous

BASE=previous.ROOT.parent
S2=previous.ROOT/'evaluations/0311a896a6684d4c8105591160cbf751'
S3=BASE/'stage3-development/evaluations/d70f15d205a14cf48e4f2bff3827f8cb'
ROOT=BASE/'stage4-development'
PROJECT=Path(__file__).resolve().parents[3]
START,END=date(2020,1,1),date(2025,9,30)
VERSION='stage4_direct_H_OOF_20261020_v1'
H60_VERSION='market8_plus_T60delta_elapsed_missing_v1'
H_FEATURES=previous.MARKET_FEATURES+previous.HISTORY_FEATURES
EXTRA=('history_60_to_30_log_probability_change','history_60_to_30_elapsed_minutes','history_60_to_30_missing')
REPETITIONS,SEED=5000,20261019
CORRECTED=1-.05/3
STRENGTHS=(.01,.1)
YEARS=(2023,2024,2025)


def guard(start,end):
    if not isinstance(start,date) or not isinstance(end,date) or start<START or end>END or start>end:
        raise MLFailure('invalid_split')


def h_split(year):
    if year not in (2021,2022,2023,2024,2025):raise MLFailure('invalid_split')
    y=year-1
    return dict(train_start='2020-01-01',train_end=f'{y}-08-31',validation_start=f'{y}-09-08',validation_end=f'{y}-10-31',
        calibration_start=f'{y}-11-08',calibration_end=f'{y}-12-24',evaluation=[(f'{year}-01-01',f'{year}-12-31' if year<2025 else '2025-09-30')])


def direct_split(year):
    if year not in YEARS:raise MLFailure('invalid_split')
    value=h_split(year);value['train_start']='2021-01-01';value['evaluation']=[(f'{year}-01-01',f'{year}-09-30')]
    return value


def plan():
    return {'version':VERSION,'development_only':True,'execution_approved':True,
        'primary_metric':'race_winner_log_loss','difference_sign':'new_minus_comparator',
        'processing_start':START.isoformat(),'processing_end':END.isoformat(),
        'no_2026_read_fit_selection_evaluation':True,'main_2025_Mar_Nov_not_reexecuted_or_pooled':True,
        'evaluation_years':list(YEARS),'months_each_year':'January_through_September','expected_common_races':6373,
        'expected_year_races':{'2023':2103,'2024':2101,'2025':2169},'strict_same_race_and_roster':True,
        'H_OOF':{'new_years':[2021,2022],'reuse_years':[2023,2024,2025],'max_new_fits':2,
            'features':list(H_FEATURES),'parameters':'unchanged_Stage2_anchor_parameters','max_rounds':300,'early_stop':20,'temperature':1.,
            'split':'train_2020_to_prior_Aug31;early_Sep08_Oct31;cal_Nov08_Dec24;next_year_predict'},
        'direct':{'configurations':['Hcal','Hstack'],'strengths':list(STRENGTHS),'max_optimizations':12,
            'split':'train_OOF_2021_to_prior_Aug31;select_strength_prior_Sep08_Oct31;no_refit',
            'scales':'sqrt(mean_over_training_races(mean_over_runners((log_p-race_mean_log_p)^2)))',
            'base_score':'log(pH)','terms':{'Hcal':['centered_log_pH/sH'],'Hstack':['centered_log_pH/sH','centered_log_pA/sA']},
            'objective':'mean_race_winner_NLL+strength/2*sum(coefficients^2)',
            'solver':'scipy_L-BFGS-B;analytic_jac;maxiter500;ftol1e-12;gtol1e-8','ties':'stronger_regularization',
            'ABCD_base':'fixed_Stage2_specification_original_Stage3_OOF;not_E_not_expanded_temperature'},
        'conditional':{'divergence':'Jensen_Shannon_natural_log','quantiles':[.5,.9],'quantile_method':'linear',
            'threshold_source':'same_year_direct_training_OOF_without_labels',
            'intervals':['D<q50','q50<=D<q90','D>=q90'],'horse_ratio':'pA/pH',
            'ratio_intervals':['ratio<.8','.8<=ratio<1.25','ratio>=1.25'],'no_subset_renormalization':True,
            'new_rule':'high_D_only_ABCD_existing_EV_rule;otherwise_no_bet','no_crossed_subgroup_search':True},
        'T60':{'table':'public.apd_sokuho_o1','offset_minutes':60,'decision_minutes':10,'max_age_seconds':300,
            'publication_minute_end_before_cutoff':True,'extra_features':list(EXTRA),'new_fits_max':3,
            'max_query_rows':256,'batch_races':32,'statement_timeout_ms':60000,'no_backfill_or_new_times':True,
            'missing':'numeric_NaN_flag1_keep_primary_population','skip':'unavailable_all_missing_or_constant_numeric_history'},
        'LightGBM_new_fits_max':5,'bootstrap':{'unit':'Monday_starting_race_week_all_venues','repetitions':REPETITIONS,
            'seed':SEED,'confidence':[.95,CORRECTED],'same_indices_per_common_population':True,
            'three_LL_comparisons':['Hstack-minus-market_history','Hstack-minus-Hcal','H60-minus-market_history'],
            'three_profit_strategies':['Hstack','H60','high_disagreement_ABCD'],
            'multiplicity':'separate_development_LL_and_profit_families;not_global_or_independent_proof'},
        'accuracy_retention':'negative_pooled_difference_and_at_least_2_of_3_years_improved;Hstack_requires_both_controls;CI_crossing_zero_provisional',
        'financial_retention':'positive_pooled_profit_and_ROI_above_100_and_at_least_2_positive_years;no_purchase_year_not_profitable',
        'paper':{'stake_yen':100,'EV_threshold':'.05','max_per_race':1,'minimum_probability':None,
            'decision':'T10_publication','settlement':'verified_official_HR_not_decision_odds','zero_purchase_ROI':None,
            'months':27,'mean_monthly_profit_includes_zero_purchase_months':True,'no_threshold_or_stake_search':True},
        'frozen_accuracy_before_profit':True,'no_production_or_scheduler_adoption':True,
        'no_final_evaluation_entry_created_or_enabled':True,'no_old_final_plan_changes':True}
