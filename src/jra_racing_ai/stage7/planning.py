"""Forward duration planning uses existing labels; isolated from T3 input work."""
from datetime import date,timedelta
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from ..ml import safety
from ..stage4 import store as old
from ..stage5 import metrics
from . import contracts as c,store as s

def duration(frame,repetitions=10000):
    required={'entry_key','date','winner_label','H','raw_market'}
    if not required<=set(frame) or frame.entry_key.duplicated().any():raise safety.MLFailure('invalid_identity')
    for d in frame.date:
        day=date.fromisoformat(str(d))
        if day.year not in c.YEARS or day.month>9:raise safety.MLFailure('invalid_split')
    scenarios={};rng=np.random.default_rng(c.SEED)
    for tag,part in [('pooled',frame)]+[(str(y),frame.loc[frame.date.str[:4]==str(y)]) for y in c.YEARS]:
        race_ids,rs=np.unique(part.entry_key.str[:16],return_inverse=True)
        weeks=np.asarray([(date.fromisoformat(d)-timedelta(days=date.fromisoformat(d).weekday())).isoformat() for d in part.date])
        week_ids,wc=np.unique(weeks,return_inverse=True);y=part.winner_label.to_numpy(int)
        a=metrics.statistics(y,part.H.to_numpy(float),rs,wc,len(week_ids))
        b=metrics.statistics(y,part.raw_market.to_numpy(float),rs,wc,len(week_ids))
        n=a[:,0];delta=a[:,6]-b[:,6];mean=delta.sum()/n.sum();centered=delta-mean*n
        estimates=[]
        for count in range(12,27):
            idx=rng.integers(len(n),size=(repetitions,count));null=centered[idx].sum(axis=1)/n[idx].sum(axis=1)
            lower,upper=np.quantile(null,[.025,.975]);stable=len(n)>=12 and np.std(null)>1e-12
            power=float(np.mean((null+.002>upper)|(null+.002<lower))) if stable else None
            lo,hi=0.,.1
            for _ in range(35):
                mid=(lo+hi)/2;v=np.mean((null+mid>upper)|(null+mid<lower))
                if v>=.8:hi=mid
                else:lo=mid
            estimates.append({'weeks':count,'power_at_MPID_0002':power,'MDE_80_percent_approx':hi if stable else None,'stable_estimate':bool(stable)})
        scenarios[tag]={'source_weeks':len(n),'source_races':int(n.sum()),'source_mean_difference':float(mean),
            'race_weighted_centering':True,'estimates':estimates}
    chosen=next((w for w in range(12,27) if all(x['estimates'][w-12]['stable_estimate'] and x['estimates'][w-12]['power_at_MPID_0002']>=.8 for x in scenarios.values())),None)
    return {'MPID':.002,'alpha_two_sided':.05,'target_power':.8,'repetitions':repetitions,'seed':c.SEED,
        'scenarios':scenarios,'chosen_weeks':chosen or 26,'adequate_estimate_within_26':chosen is not None,
        'conservative_across_pooled_and_each_year':True,'annual_model_variance_transfer_to_fixed_H_is_approximation':True,
        'future_power_guaranteed':False,'missing_weeks_count_no_extension':True}

def initialize(folder,start_date):
    created=c.clock();start=date.fromisoformat(start_date)
    if start<created.date() or start<date(2026,10,22):raise safety.MLFailure('invalid_split')
    # No database call exists in initialization. Freeze both universes first.
    s.write(folder/'old_code_hashes.json',s.old_code());s.write(folder/'tasks_before.json',old.task_configuration())
    references=s.refs()
    if references[str(c.S6/'plan.json')]!='d1af2a36db36ae713613974383d554cf00a31ad601c6deb6d0df36b1f75891e7':
        raise safety.MLFailure('artifact_invalid')
    path=c.S6/'predictions.local.csv'
    frame=pd.read_csv(path,usecols=['entry_key','date','winner_label','H','raw_market'],dtype={'entry_key':str,'date':str})
    power=duration(frame);del frame
    s.write(folder/'forward_duration_design.json',dict(power,source_prediction_file=str(path),source_sha256=s.sha(path)))
    forward={'version':c.VERSION,'run_id':folder.name,'created_at':created.isoformat(),'start_date':start_date,
        'end_date':c.week_end(start_date,power['chosen_weeks']),'planned_weeks':power['chosen_weeks'],
        'H_model_id':c.H_ID,'H_temperature':1.,'features':list(c.H_FIELDS),'references':references,
        'input_minutes_before_start':10,'early_minutes_before_start':30,'age_seconds_max':300,
        'prediction_save_grace_seconds':60,'poll_seconds':15,'recording_start_hour':8,'recording_end_hour':18,
        'recording_scope':'JRA_01_10_all_registered_future_races;no_roster_rescaling_or_posthoc_fill',
        'observed_timestamp':'recorder_first_seen_at_not_original_PC_receipt','require_publication_minute_end_and_first_seen_before_cutoff':True,
        'schedule_change_policy':'skip_if_changed_before_commit;append_state_change_after_commit',
        'source_tables':['public.jvd_ra','public.jvd_se','public.apd_sokuho_o1'],'results_enabled':False,
        'new_fits_allowed':0,'stage7b_approved':False,'sealed_2026_Jan_Sep_read':False,
        'forward_start_authorized':True,'dedicated_research_execution_setting_authorized':True,
        'evaluation':{'primary':'equal_race_mean_within_race_auc_H_minus_same_time_market','MPID':.002,'week_CI':.95,
            'repetitions':5000,'seed':c.SEED,'success':'difference>=MPID_and_CI_lower>0',
            'secondary':['pooled_auc','fractional_tie_top1','race_log_loss','horse_log_loss','Brier'],
            'outcomes_only_after_fixed_end_date':True,'interim_performance_forbidden':True}}
    s.write(folder/'forward_plan.json',forward)
    s.write(folder/'plan_commit.json',{'forward_plan_sha256':s.sha(folder/'forward_plan.json'),
        'implementation_hashes':implementation_hashes(),'committed_at':c.clock().isoformat(),'DB_not_connected':True,'fits':0})
    return {'run_id':folder.name,'start_date':start_date,'end_date':forward['end_date'],'planned_weeks':forward['planned_weeks'],
        'fits':0,'power_is_approximation':True,'T3_preparation_independent':True}

def implementation_hashes():
    paths=list(Path(__file__).parent.glob('*.py'))+list((c.PROJECT/'scripts').glob('stage7*'))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}

def prepare_survey(folder):
    from . import sources
    created=c.clock()
    if (folder/'T3_commit.json').exists():return {'status':'already_prepared','fits':0}
    targets,target_refs=sources.prepare();s.write(folder/'T3_targets.local.json',{'records':targets})
    survey={'version':c.VERSION,'created_at':created.isoformat(),'targets_file':'T3_targets.local.json',
        'targets_sha256':s.sha(folder/'T3_targets.local.json'),'key_roster_list_sha256':s.digest([(x['race_key'],x['roster'],x['roles']) for x in targets]),
        'source_references':target_refs,'original_B':{'races':6373,'runners':87601,'annual_races':c.EXPECTED},
        'period':['2020-01-01','2025-09-30'],'current_minutes':3,'current_age_seconds_max':120,
        'publication_precision':'minute_end_before_cutoff','early_minutes':30,'early_age_seconds_max':300,
        'fields':list(c.H_FIELDS),'query_races_max':32,'query_rows_max':64,'statement_timeout_ms':60000,
        'DB_elapsed_budget_seconds':3600,'full_batch_retry_allowed':False,'new_fits_allowed':0,
        'labels_and_HR_and_final_odds_forbidden':True,'coverage_85_not_a_gate':True,
        'allowed_O1_columns':['record_id','data_kubun','happyo_tsukihi_jifun','hatsubai_flag_tansho','odds_tansho'],
        'no_current_create_update_time_as_original_receipt':True,
        'preparation_incident':'initial_structure_probe_read_one_packet_as_text_in_full;no_outcome_value_parsed_displayed_or_used;formal_survey_uses_projection_only'}
    s.write(folder/'T3_plan.json',survey)
    s.write(folder/'T3_commit.json',{'T3_plan_sha256':s.sha(folder/'T3_plan.json'),'implementation_hashes':implementation_hashes(),
        'committed_at':c.clock().isoformat(),'DB_not_connected':True,'fits':0})
    return {'run_id':folder.name,'target_races':len(targets),'B_races':6373,'fits':0}

def verify(folder,name):
    commit=s.read(folder/('T3_commit.json' if name=='T3' else 'plan_commit.json'));path=folder/(name+'_plan.json')
    if s.sha(path)!=commit[name+'_plan_sha256']:raise safety.MLFailure('artifact_invalid')
    if implementation_hashes()!=commit['implementation_hashes']:raise safety.MLFailure('artifact_invalid')
    return s.read(path)
