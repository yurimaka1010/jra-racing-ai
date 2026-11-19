from datetime import date
from pathlib import Path
from ..research import contracts as previous
from ..ml.safety import MLFailure

ROOT=previous.ROOT.parent/'stage3-development'
SOURCE=previous.ROOT/'evaluations/0311a896a6684d4c8105591160cbf751'
VERSION='stage3_oof_softmax_20261019_v1'
E_VERSION='signed_SE_time_sa_lag7_recent3_v1'
LAMBDA=(0.,.1,.25,.5,1.)
TEMPERATURE=(.5,.625,.75,.875,1.,1.25,2.)
WEIGHTS=(0.,.05,.1,.25,.5,1.)
STRENGTH=(.01,.1)
TERMS={'M':('q',),'MH':('q','h'),'MA':('q','a'),'MHA':('q','a','h')}
E_FEATURES=('e_previous_time_difference_seconds','e_recent3_time_difference_seconds')
SEED=20261019
REPETITIONS=5000


def guard(start,end):
    previous.guard_development(start,end)


def split(year):
    if year not in range(2020,2026):raise MLFailure('invalid_split')
    old=year-1
    return {'train_start':'2016-07-02','train_end':f'{old}-08-31',
        'validation_start':f'{old}-09-08','validation_end':f'{old}-10-31',
        'calibration_start':f'{old}-11-08','calibration_end':f'{old}-12-24',
        'evaluation':[(f'{year}-01-01',f'{year}-12-31' if year<2025 else f'{year}-09-30')]}


def integration_split(year):
    if year not in (2023,2024,2025):raise MLFailure('invalid_split')
    old=year-1
    return {'train_start':'2020-01-01','train_end':f'{old}-08-31',
        'validation_start':f'{old}-09-08','validation_end':f'{old}-10-31',
        'calibration_start':f'{old}-11-08','calibration_end':f'{old}-12-24',
        'evaluation':[(f'{year}-01-01',f'{year}-09-30')]}


def plan():
    return {'version':VERSION,'primary_metric':'race_winner_log_loss','difference_sign':'new_minus_comparator',
        'development_only':True,'main_2025_not_pooled_with_annual_2025':True,
        'source':str(SOURCE),'source_formula_version':previous.FORMULA_VERSION,
        'sealed_2026':{'start':'2026-01-01','end':'2026-09-30','labels_read':False,'train':False,'evaluate':False},
        'lambda':{'candidates':LAMBDA,'parents':['market_history','market_ABCD_history'],
            'folds':['main','2023','2024','2025'],'max_evaluations':40,
            'selection':'existing_calibration_race_LL;ties_smaller_lambda',
            'formula':'race_normalize(sigmoid(logit(q)+lambda*tree_raw_score))'},
        'temperature':{'candidates':TEMPERATURE,'selection':'existing_calibration_race_LL;ties_closest_old_temperature_then_smaller',
            'input':'uncalibrated_binary_booster_probability_not_calibrated_probability','no_followup_search':True},
        'OOF':{'years':list(range(2020,2026)),'new_training_years':[2020,2021,2022],
            'reused_annual_years':[2023,2024,2025],'new_fits_max':3,
            'spec':'Stage2_ABCD;original_300_rounds_early20_and_4_temperature_grid',
            'split':'prior_Aug31_train;Sep08_Oct31_early;Nov08_Dec24_cal;following_year_predict',
            'modified_temperature_and_E_excluded':True},
        'integration':{'terms':TERMS,'strengths':STRENGTH,'max_optimizations':24,
            'objective':'mean_race_winner_NLL + strength/2 * sum((theta-(1,0,0))**2)',
            'solver':'scipy_L-BFGS-B_analytic_jac_maxiter500_ftol1e-12_gtol1e-8',
            'fit':'OOF_2020_through_prior_Aug31','strength_selection':'prior_Sep08_Oct31_unpenalized_race_LL;ties_stronger',
            'no_refit_after_selection':True,'missing_early_h':0.,'no_intercept_or_auxiliary_features':True},
        'arithmetic_blend':{'weights':WEIGHTS,'max_series':1,
            'selection':'prior_Nov08_Dec24_OOF_race_LL;ties_smaller_w'},
        'E':{'version':E_VERSION,'fields':E_FEATURES,'new_fits_max':4,'groups':'ABCD_plus_E_only',
            'formula':'past_signed_time_sa_seconds;recent3_weights_1_.5_.25;lag7',
            'skip_if_unverified':True,'not_used_for_OOF_or_integration':True},
        'new_LightGBM_fits_max':7,'bootstrap':{'unit':'Monday_starting_race_week_all_venues','repetitions':REPETITIONS,'seed':SEED,'confidence':.95},
        'selection_rule':'pooled_annual_race_LL_negative_and_at_least_two_of_three_years_improve;no_profit_selection',
        'market_control_candidates':['raw_market','market_history','lambda_market_history','M','MH'],
        'ability_candidates':['expanded_temperature','ABCD_E'],'AI_candidates':['MA','MHA','arithmetic','lambda_market_ABCD_history'],
        'paper':{'stake_yen':100,'threshold':'.05','max_per_race':1,'additional_min_probability':'.05',
            'additional_only':['Stage2_ABCD','frozen_market_control','retained_AI_candidate'],'selection_uses_profit':False},
        'final_evaluation':{'executed':False,'enabled':False,'primary_comparisons':2,'confidence_each':.975,'multiplicity':'Bonferroni',
            'formal_success':'difference_upper_strictly_below_zero','final_2026_training_not_done':True}}
