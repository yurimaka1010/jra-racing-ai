"""Frozen models and the user-approved prospective protocol. No result access."""
from datetime import date,datetime,timedelta
from pathlib import Path
from . import contracts as c
from ..ml import safety


def monday(day):return day-timedelta(days=day.weekday())


def formal_window(completed_at,first_cutoff=None):
    """Use timing only. Never pick a window using scores or available outcomes."""
    first=c.FIRST_DAY;week=monday(first)
    if completed_at.date()<first:
        start=week
    elif first_cutoff is not None and completed_at<first_cutoff:
        start=monday(first_cutoff.date())
    else:
        # Without evidence for the first cutoff in the completion week, the next
        # full week is the earliest formally eligible week.
        start=max(week,monday(completed_at.date())+timedelta(weeks=1))
    return start,start+timedelta(weeks=21,days=-1)


def code_files():
    root=c.PROJECT/'src/jra_racing_ai'
    files=list((root/'pedigree_forward').glob('*.py'))
    names=('pedigree_research/contracts.py','pedigree_research/features.py','pedigree_research/data.py',
        'pedigree_research/learning.py','pedigree_research/sources.py','d39_next/features.py','d39_next/forward.py',
        'd39_enrichment/features.py','d39_enrichment/sources.py','d39_enrichment/weather.py','d39_enrichment/data.py',
        'd39_enrichment/learning.py','d39_generation/four_sources.py','d39_generation/data.py','d39_generation/forward.py',
        'research/features.py','research/model.py','ml/features.py','ml/encodings.py','ml/market.py',
        'stage7/inputs.py','stage7/forward.py','win_sim/core.py','win_sim/formats.py','win_sim/publication.py',
        'db_connection.py','reader_gate.py')
    files += [root/n for n in names]
    files += [c.PROJECT/'scripts'/n for n in ('pedigree_forward.py','pedigree_forward.ps1','register_pedigree_forward.ps1')]
    return sorted(set(files))


def models():
    from ..pedigree_research.learning import load as load_new
    from ..d39_enrichment.learning import load as load47
    bundles={}
    for name in c.NAMES:
        path=(Path(c.read(c.REFERENCE/'candidate_freeze.json')['model_path']) if name=='reference47' else
              Path(c.read(c.TRAIN/(name+'.candidate_freeze.json'))['model_path']))
        bundle=(load47 if name=='reference47' else load_new)(path)
        if (bundle[1]['model_id']!=c.MODEL_IDS[name] or bundle[1]['temperature']!=c.TEMPERATURE[name]
                or bundle[1]['best_iteration']!=219):raise safety.MLFailure('invalid_model')
        bundles[name]=bundle
    return bundles


def freeze(directory):
    bundles=models();items={}
    for name,(_,meta) in bundles.items():
        path=Path(c.read(c.REFERENCE/'candidate_freeze.json')['model_path']) if name=='reference47' else Path(c.read(c.TRAIN/(name+'.candidate_freeze.json'))['model_path'])
        types={f:'category' if f in meta['categories'] else 'float64' for f in meta['features']}
        items[name]={'model_id':meta['model_id'],'model_path':str(path),'temperature':meta['temperature'],
            'rounds':219,'model_json_sha256':c.sha(path/'model.json'),'booster_sha256':c.sha(path/'booster.lgb.txt'),
            'features':meta['features'],'features_order_sha256':c.digest(meta['features']),
            'input_types':types,'input_types_sha256':c.digest(types),'categories':meta['categories'],
            'categories_sha256':c.digest(meta['categories']),'formula_spec_sha256':c.digest(meta['feature_spec'])}
    import hashlib
    # A private source snapshot insulates this fixed research from parallel work.
    runtime=c.mkdir(directory/'runtime'/'project');relative={}
    original=list((c.PROJECT/'src/jra_racing_ai').rglob('*.py'))
    original += [c.PROJECT/'scripts'/n for n in ('pedigree_forward.py','pedigree_forward.ps1','register_pedigree_forward.ps1')]
    codes={}
    for path in original:
        if '__pycache__' in path.parts:continue
        rel=path.relative_to(c.PROJECT);target=runtime/rel;c.mkdir(target.parent)
        content=path.read_bytes()
        with target.open('xb') as stream:stream.write(content)
        sha=hashlib.sha256(content).hexdigest();codes[str(target)]=sha;relative[str(rel).replace('\\','/')]=sha
    tree=c.digest(relative)
    # This checkout has no Git metadata. Do not invent a Git commit.
    completed=c.clock()
    value={'version':c.VERSION,'models':items,'frozen_at':completed.isoformat(),'code_files':codes,'runtime_project':str(runtime),
        'code_commit_id':'source-sha256:'+tree,'git_commit_id':None,'git_status':'not_a_Git_checkout;source_hash_used',
        'pedigree_spec_sha256':c.sha(c.PREP/'feature_spec.json'),
        'base_lineage_sha256':c.sha(c.PREP/'lineage.local.json'),'new_fits':0,
        'master_historical_corrections_asof':'unverified_static_parent_IDs',
        'no_model_parameter_or_temperature_changes':True}
    from .history import HISTORY_POPULATION_VERSION
    value['history_population_version']=HISTORY_POPULATION_VERSION
    scope=c.read(directory/'execution_scope.json')
    value['supersedes_research_run_id']=scope.get('supersedes_research_run_id')
    c.write(directory/'model_freeze.json',value)
    c.write(directory/'model_freeze.commit.json',{'sha256':c.sha(directory/'model_freeze.json'),'recorded_at':c.clock().isoformat()})
    start,end=formal_window(completed)
    rules={'version':c.VERSION,'approved_by':'human_user','decision_date':'2026-10-26',
        'primary_metric_changed_from':'pooled_ROC_AUC','primary_metric':'within_race_auc',
        'primary_contrast':'pedigree45_minus_control39','primary_comparisons':1,
        'thresholds':{'minimum_within_race_AUC_gain':.004,'primary95CI_lower_strictly_above':0,
            'guardrail_method':'point_estimate','race_LL_increase_max':.005,'Brier_increase_max':.00010},
        'threshold_authorization':'explicit_user_reply_accept_proposed_values_point_estimate',
        'window_start_week':str(start),'window_end_date':str(end),'calendar_weeks':21,
        'window_rule':'first_full_eligible_Monday_week_after_freeze_and_preregistration_before_first_target_T10;otherwise_provisional;no_result_based_changes',
        'window_decided_without_results':True,'provisional_before_formal_start_excluded':True,
        'P1':{'all3_complete_forecasts':True,'odds_required':False,'whole_race_reasons_first':True,
            'abnormal_race_policy':'existing_d39_next.race_reasons_whole_race_exclusion',
            'post_T10_cancellations':'only_after_race_filter;drop_cancelled_then_full_remaining_roster_renormalize'},
        'P2':{'subset_of':'P1','complete_T10_odds_required':True,'freshness_seconds_max':300,
            'cancelled_handling':'same_as_P1;market_also_renormalized'},
        'P3':{'all3_complete_forecasts':True,'complete_T10_odds_required':True,'freshness_seconds_max':300,
            'abnormal_races_included':True,'not_a_subset_of_P1_or_P2':True},
        'purchase':{'rule':'R1_only_top_probability_win_single_100yen_per_model',
            'tie_break':'highest_probability_then_smallest_horse_number','EV_filter':False,'R2':False,
            'no_odds_no_bet':True,'payout':'official_final_win_payout;decision_odds_never_used_as_payout',
            'scratch_exclusion':'refund','did_not_finish_or_disqualified':'loss','actual_purchase':False,
            'not_for_model_selection':True},
        'excluded_known_races':['known20','calibration133'],'newcomer_auxiliary':'declared_condition701;not_promotion_basis',
        'bootstrap':{'reuse':'stage5.metrics.WeekBootstrap','repetitions':5000,'seed':20261026,
            'unit':'Monday_starting_calendar_week_all_same_day_races_together','paired':True,'confidence':.95},
        'weekly_t_check':'unweighted_mean_of_weekly_paired_differences_and_t20_interval;diagnostic_different_weighting_from_race_weighted_primary',
        'pooled_AUC_reference_temperatures':['each_saved_temperature',1.,1.25],
        'P2_market':'inverse_T10_odds_renormalized_with_same_remaining_roster',
        'interim_scores':False,'operational_counts_only':True,'final_evaluation_openings_max':1,
        'subgroup_probability_renormalization':False,'automatic_promotion':False,
        'H_Stage7_results':False,'shuffle_control':False,'new_fits':0,
        'freeze_sha256':c.sha(directory/'model_freeze.json'),'measurement_sha256':c.sha(directory/'measurement.json')}
    rules['history_population_version']=HISTORY_POPULATION_VERSION
    rules['supersedes_research_run_id']=scope.get('supersedes_research_run_id')
    c.write(directory/'preregistration.json',rules)
    c.write(directory/'preregistration.commit.json',{'sha256':c.sha(directory/'preregistration.json'),
        'recorded_at':c.clock().isoformat(),'human_thresholds_finalized':True})
    return {'status':'frozen_and_preregistered','run_id':directory.name,'formal_start_week':str(start),
        'formal_end_date':str(end),'thresholds':rules['thresholds'],'model_freeze_sha256':c.sha(directory/'model_freeze.json'),
        'preregistration_sha256':c.sha(directory/'preregistration.json'),'new_fits':0}


def load(directory,*,check_code=True,allow_missing_models=False):
    freeze=c.read(directory/'model_freeze.json');p=c.read(directory/'preregistration.json')
    if (c.sha(directory/'model_freeze.json')!=c.read(directory/'model_freeze.commit.json')['sha256']
            or c.sha(directory/'preregistration.json')!=c.read(directory/'preregistration.commit.json')['sha256']
            or freeze['version']!=c.VERSION or p['version']!=c.VERSION
            or p['primary_metric']!='within_race_auc' or p['purchase']['EV_filter'] is not False
            or p['thresholds']!={'minimum_within_race_AUC_gain':.004,'primary95CI_lower_strictly_above':0,
                'guardrail_method':'point_estimate','race_LL_increase_max':.005,'Brier_increase_max':.0001}):
        raise safety.MLFailure('artifact_invalid')
    if check_code:
        import hashlib
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha for path,sha in freeze['code_files'].items()):
            raise safety.MLFailure('artifact_invalid')
    for name,item in freeze['models'].items():
        path=Path(item['model_path'])
        if item['model_id']!=c.MODEL_IDS[name] or item['temperature']!=c.TEMPERATURE[name]:
            raise safety.MLFailure('invalid_model')
        try:
            if (c.sha(path/'model.json')!=item['model_json_sha256']
                    or c.sha(path/'booster.lgb.txt')!=item['booster_sha256']):raise safety.MLFailure('invalid_model')
        except Exception:
            if not allow_missing_models:raise safety.MLFailure('invalid_model') from None
    return freeze,p


def require_closed(directory,now=None):
    _,p=load(directory);now=now or c.clock()
    if now.date()<=date.fromisoformat(p['window_end_date']):raise safety.MLFailure('forward_cutoff')
    return p
