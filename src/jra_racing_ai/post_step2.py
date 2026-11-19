"""Bounded local follow-up experiments. Frozen selections and separate final holdout."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date, datetime, timedelta
import csv
import gc
import json
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from .core import RACE_KEY, SafeAuditError
from .db_connection import read_only_session
from .inspect_schema import COLUMNS_SQL
from .ml import data, features, learning_curve as curve, market, model, safety, v03
from .ml_v03_cli import save_policy
from .win_sim import core, formats, local, publication, settle_saved
from .win_sim import observed
from .win_sim_cli import connection_config

ROOT=Path(r'C:\Users\acbd5\AppData\Local\JraRacingAI\post-step2')
PRIOR=Path(r'C:\Users\acbd5\AppData\Local\JraRacingAI\ml-step2\evaluations\af2409a20b644c01a7381cb83c56231d')
POOL=safety.ML_ROOT/'datasets'/'6c1ee55061424b46a1bf38d7a391c7b1'/'runners.csv'
MAPPING=safety.ML_ROOT/'schema'/'0543279477dd44f3a84fc36bcc61841b'/'mapping.reviewed.json'
FROZEN=Path(r'C:\Users\acbd5\AppData\Local\JraRacingAI\trial-20261018\evaluations\c32ac7a4a2ae404fa7e4ec4342856ade')
MODELS={'old_v03':('1872a2db908b417a9d982b8836377fa6',2.),
        'max100':('1e85f286637e4942b4154ed741d38f9b',.75),
        'best242':('a8178c2b7523499f9463c1743b41c715',.75)}

def now():
    return datetime.now(core.JST)

def write(directory,name,value):
    safety.write_json(directory/name,value,root=ROOT)

def read(directory,name):
    return safety.read_json(directory/name,root=ROOT)

def progress(stage,**values):
    print(json.dumps(dict(status='running',stage=stage,**values),ensure_ascii=True),flush=True)

def session():
    if not os.environ.get('PGPASSWORD'):
        raise safety.MLFailure('needs_authentication')
    return read_only_session(connection_config(60000),os.environ['PGPASSWORD'])

def inventory(directory):
    with session() as db:
        catalog,_=data.inspect_sources(db,include_years=False)
        structures={table:db.aggregate('post.columns',COLUMNS_SQL,('public',table),max_rows=1000)
                    for table in ('jvd_we','jvd_o1','apd_sokuho_o1')}
        months=db.aggregate('post.months',"SELECT substr(kaisai_tsukihi::text,1,2) AS month, count(*) AS races FROM ONLY public.jvd_ra WHERE kaisai_nen::text='2025' AND data_kubun::text='7' AND keibajo_code::text IN ('01','02','03','04','05','06','07','08','09','10') GROUP BY 1 ORDER BY 1",max_rows=12)
        odds={}
        for table in ('jvd_o1','apd_sokuho_o1'):
            if structures[table]:
                odds[table]=db.aggregate('post.odds_inventory','SELECT kaisai_nen::text AS year,data_kubun::text AS category,count(*) AS rows,min(kaisai_tsukihi::text) AS first_month_day,max(kaisai_tsukihi::text) AS last_month_day FROM ONLY public.'+table+" WHERE kaisai_nen::text IN ('2025','2026') GROUP BY 1,2 ORDER BY 1,2",max_rows=50)
        weather=[]
        we_columns={c['name'] for c in structures['jvd_we']}
        if 'kaisai_nen' in we_columns:
            weather=db.aggregate('post.we_inventory',"SELECT kaisai_nen::text AS year,count(*) AS rows FROM ONLY public.jvd_we WHERE kaisai_nen::text IN ('2024','2025','2026') GROUP BY 1 ORDER BY 1",max_rows=3)
        time_format=db.aggregate('post.time_format',"SELECT count(*) AS rows,count(*) FILTER (WHERE soha_time IS NULL) AS nulls,count(*) FILTER (WHERE soha_time::text ~ '^[0-9]{4}$') AS four_digits,count(*) FILTER (WHERE soha_time::text='0000') AS zero_initial,count(*) FILTER (WHERE soha_time::text ~ '^[0-9]{4}$' AND substring(soha_time::text,2,2)::integer>=60) AS invalid_seconds FROM ONLY public.jvd_se WHERE kaisai_nen::text BETWEEN '2016' AND '2025' AND data_kubun::text='7'",max_rows=1)
    result=dict(months=months,odds=odds,weather=weather,time_format=time_format,structures=structures,catalog=catalog)
    write(directory,'inventory.json',result)
    if not all(any(m['month']==f'{month:02d}' and m['races']>0 for m in months) for month in range(3,12)):
        raise safety.MLFailure('insufficient_history')
    frozen={str(p):safety.digest(p,root=FROZEN.parents[1]) for p in (FROZEN/'predictions.csv',FROZEN/'run.json')}
    plan={'created_at_jst':now().isoformat(),'frozen_before_new_prediction_metrics':True,
        'train_start':'2016-07-02','train_end':'2024-11-30','history_start':'2015-07-02',
        'validation_start':'2024-12-08','validation_end':'2024-12-28',
        'calibration_start':'2025-01-11','calibration_end':'2025-02-02',
        'development_start':'2025-03-01','development_end':'2025-05-31',
        'confirmation_start':'2025-06-01','confirmation_end':'2025-08-31',
        'final_start':'2025-09-01','final_end':'2025-11-30',
        'primary_metric':'horse_binary_log_loss','history_lag_days':7,'max_rounds':300,
        'temperature_candidates':[.75,1.,1.25,2.],'bootstrap_repetitions':1000,
        'frozen_models':MODELS,'feature_selection':'development_LL_only_select_lower_LL_than_best242_else_keep_best242',
        'exclusions':'same_complete_single_winner_races_and_priority_features_as_STEP2; never_drop_for_missing_optional_time',
        'going_gate':'require_reconstructable_pre_race_snapshot_or_WE_sequence_with_known_initial_state; final_RA_going_is_not_evidence',
        'time_profile':'latest_eligible_same_distance_same_turf_or_dirt_JV_mss_tenth_4',
        'odds_max_age_seconds':120,'odds_decision_minutes_before_start':10,'threshold':.05,'stake_yen':100,
        'frozen_20261018_artifacts':frozen,'training_cohort_source_model_id':MODELS['best242'][0]}
    write(directory,'plan.json',plan)
    return {'monthly_coverage':months,'odds_inventory':odds,'weather_inventory':weather,'time_format':time_format,
            'going_columns':[c['name'] for c in catalog['tables']['ra'] if 'baba' in c['name']],
            'weather_column_names':sorted(we_columns),'periods_frozen':True}

def export_dataset(db,mapping,directory,start,end,*,max_rows,horse_ids=None):
    directory.mkdir()
    catalog,_=data.inspect_sources(db,include_years=False)
    meta=data.export_dataset(db,mapping,directory,start=date.fromisoformat(start),end=date.fromisoformat(end),
        max_rows=max_rows,max_seconds=300,root=ROOT,catalog=catalog,horse_ids=horse_ids)
    return meta

def fetch_market(db,races):
    rows=[]
    budget=local.Budget(300)
    for table in ('jvd_o1','apd_sokuho_o1'):
        columns=publication._columns(db,table,publication.O1_FIELDS,budget)
        if columns is None:
            continue
        for offset in range(0,len(races),40):
            batch=local._rows(db,budget,table,publication.O1_FIELDS,columns,races[offset:offset+40],50000,join_targets=True)
            for row in batch:
                row['source_table']=table
            rows.extend(batch)
    return rows

def collect(directory):
    plan=read(directory,'plan.json')
    mapping=safety.read_json(MAPPING)
    time_mapping=json.loads(json.dumps(mapping))
    time_mapping.update(format_version=2,formats={'finish_time_seconds':'jv_mss_tenths_4'})
    time_mapping['columns']['finish_time_seconds']='soha_time'
    time_mapping=data.validate_mapping(time_mapping)
    write(directory,'mapping.time.reviewed.json',time_mapping)
    write(directory,'feature_availability.json',{'going':{'status':'withheld',
        'reason':'no_2024_2025_WE_or_pre_race_RA_history; final_RA_state_not_used'},
        'time':{'status':'enabled','official_reference':'JV-Data4901.pdf SE item 44, PDF page 12',
        'local_evidence':'soha_time: 481627/481627 four ASCII digits; invalid second fields=0; 0000=missing',
        'decoder':'jv_mss_tenths_4','profile':'conditioned_previous_time_v1',
        'meaning':'latest prior valid finish time with same distance and exact track code, available after 7 days',
        'independent_official_race_display_crosscheck':'not_performed'}})
    with session() as db:
        progress('collect_training_time_fields')
        train=export_dataset(db,time_mapping,directory/'training_pool','2015-07-02','2025-05-31',max_rows=500000)
        original,original_meta=data.read_dataset(POOL)
        refreshed,_=data.read_dataset(directory/'training_pool'/'runners.csv',root=ROOT)
        # The added time column must not silently change the fixed training cohort.
        old_keys={features.entry_identity(r)[0] for r in original}
        if old_keys!={features.entry_identity(r)[0] for r in refreshed}:
            raise safety.MLFailure('artifact_invalid')
        del original,refreshed,old_keys
        gc.collect()
        progress('collect_independent_periods')
        evaluation=export_dataset(db,time_mapping,directory/'evaluation','2025-06-01','2025-11-30',max_rows=50000)
        targets,_=data.read_dataset(directory/'evaluation'/'runners.csv',root=ROOT)
        history=export_dataset(db,time_mapping,directory/'evaluation_history','2015-07-02','2025-11-30',
            max_rows=450000,horse_ids={r['horse_id'] for r in targets if r.get('horse_id')})
        races=tuple(sorted({features.entry_identity(r)[0][:-1] for r in targets}))
        progress('collect_market_reference',race_count=len(races))
        quotes=fetch_market(db,races)
        write(directory,'market_inputs.local.json',quotes)
        frozen_rows=list(csv.DictReader((FROZEN/'predictions.csv').open(encoding='utf-8-sig',newline='')))
        trial_history=export_dataset(db,time_mapping,directory/'trial_history','2015-07-02','2026-10-11',
            max_rows=100000,horse_ids={r['horse_id'] for r in frozen_rows})
    result={'training_pool_rows':train['rows'],'training_cohort_keys_unchanged':True,
        'evaluation_rows':evaluation['rows'],'evaluation_races':evaluation['races'],
        'evaluation_history_rows':history['rows'],'trial_history_rows':trial_history['rows'],
        'going_status':'withheld_no_historical_pre_race_states',
        'odds_rows':len(quotes),'odds_categories':dict(Counter(r['status'] for r in quotes))}
    write(directory,'collection.json',result)
    return result

def strip_time(rows):
    return [dict(r,finish_time_seconds='',going='') for r in rows]

def predict(bundle,rows,history,*,profile=None):
    metadata=bundle['metadata']
    profile=profile if profile is not None else metadata.get('feature_profile')
    if profile is None:
        rows,history=strip_time(rows),strip_time(history)
    start=date.fromisoformat(metadata.get('history_start','2024-01-01'))
    history=[r for r in history if features.entry_identity(r)[1]>=start]
    matrix=features.build_features(rows,history=history,history_lag_days=7,feature_profile=profile)
    values=bundle['boosters']['win'].predict(model.frame(matrix,metadata['categories']),num_threads=4)
    return v03.prediction_races(rows,values),v03.coverage(matrix)

def eligible_cohort(rows,start,end):
    winners=v03.single_winners(rows)
    matrix=features.build_features(strip_time(rows),history_lag_days=7)
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        groups[features.entry_identity(r)[0][:-1]].append(i)
    selected=[]; reasons=Counter()
    for race,indices in sorted(groups.items()):
        day=curve.features.race_identity(dict(zip(RACE_KEY,race)))[1]
        if not date.fromisoformat(start)<=day<=date.fromisoformat(end):
            continue
        if race not in winners:
            reasons['missing_rank_or_multiple_winners']+=1
        elif any(matrix[i][name] is None for i in indices for name in v03.PRIORITY_FEATURES):
            reasons['missing_priority_feature']+=1
        else:
            selected.extend(indices)
    return [rows[i] for i in selected],winners,dict(reasons)

def policy_for(temperature,start='2025-03-01',end='2025-11-30'):
    return core.WinPolicy(temperature=temperature,evaluation_start=start,evaluation_end=end,
        model_fit_end='2024-12-28',calibration_end='2025-02-02',calibration_source='independent_validation')

def paired(a,b,winners,pa,pb):
    result=curve.paired_bootstrap(curve.race_scores(a,winners,pa),curve.race_scores(b,winners,pb))
    import numpy as np
    aa={p.race:p for p in a}; bb={p.race:p for p in b}; keys=sorted(set(aa)&set(bb)&set(winners))
    differences=np.asarray([-math_log_winner(aa[k],winners[k],pa)+math_log_winner(bb[k],winners[k],pb) for k in keys])
    sample=np.random.default_rng(20261018).integers(0,len(keys),(1000,len(keys)))
    low,high=np.quantile(differences[sample].mean(axis=1),[.025,.975])
    result['race_log_loss']={'difference_new_minus_baseline':float(differences.mean()),'ci95_low':float(low),'ci95_high':float(high)}
    return result

def math_log_winner(prediction,winner,policy):
    import math
    return math.log(max(1e-15,float(core.probabilities(prediction.probabilities,policy)[0][winner])))

def market_results(predictions,winners,quotes,policy):
    groups=defaultdict(list)
    for row in quotes:
        groups[tuple(row['race_key'])].append(row)
    selections=defaultdict(list); reasons=Counter()
    for p in predictions:
        quote,reason=market.select_final(p.race,groups[p.race],tuple(p.probabilities))
        if reason:
            reasons[reason]+=1
        else:
            selections[quote.basis].append(quote)
    return {'by_timing':{basis:market.measure(predictions,winners,subset,policy) for basis,subset in selections.items()},
            'excluded':dict(reasons),'pre_race_status':'unavailable_2025_only_final_category_5'}

def evaluate(directory):
    plan=read(directory,'plan.json')
    rows,_=data.read_dataset(directory/'evaluation'/'runners.csv',root=ROOT)
    history,_=data.read_dataset(directory/'evaluation_history'/'runners.csv',root=ROOT)
    quotes=read(directory,'market_inputs.local.json')
    result={}
    for period in ('confirmation',):
        cohort,winners,excluded=eligible_cohort(rows,plan[period+'_start'],plan[period+'_end'])
        predictions={}; metrics={}
        for name,(model_id,temperature) in MODELS.items():
            progress('evaluate_frozen_models',period=period,model=name)
            bundle=model.load_models(safety.ML_ROOT/'models'/model_id,required_targets=('win',))
            predictions[name],_=predict(bundle,cohort,history)
            metrics[name]=v03.probability_evaluation(predictions[name],winners,policy_for(temperature))
        result[period]={'metrics':metrics,'excluded':excluded,
            'best242_minus_max100':paired(predictions['best242'],predictions['max100'],winners,policy_for(.75),policy_for(.75)),
            'best242_minus_old':paired(predictions['best242'],predictions['old_v03'],winners,policy_for(.75),policy_for(2.)),
            'market':market_results(predictions['best242'],winners,quotes,policy_for(.75))}
        # The final comparison here is frozen-model reporting, never feature selection input.
    write(directory,'independent.json',result)
    return {'periods':{name:{'races':value['metrics']['best242']['races'],
        'metrics':{m:v['model'] for m,v in value['metrics'].items()},
        'best242_minus_max100':value['best242_minus_max100']} for name,value in result.items()}}

def features_experiment(directory):
    plan=read(directory,'plan.json')
    rows,metadata=data.read_dataset(directory/'training_pool'/'runners.csv',root=ROOT)
    progress('prepare_feature_experiment',rows=len(rows))
    original_plan=safety.read_json(PRIOR/'run.json',root=PRIOR.parents[1])['plan']
    matrix=features.build_features(rows,history_lag_days=7,feature_profile='conditioned_previous_time_v1')
    parts,groups,winners,excluded=curve.prepare_cohorts(rows,matrix,original_plan)
    selection=safety.read_json(safety.ML_ROOT/'models'/MODELS['best242'][0]/'training_selection.local.json')
    training_keys=[tuple(r) for r in selection['race_keys']]
    if set(training_keys)!=set(parts['train']) or len(training_keys)!=26173:
        raise safety.MLFailure('artifact_invalid')
    indices=sorted(i for name,races in parts.items() for race in (training_keys if name=='train' else races) for i in groups[race])
    fit_rows=[rows[i] for i in indices]; prepared=[matrix[i] for i in indices]
    coverage={name:v03.coverage([matrix[i] for race in races for i in groups[race]]) for name,races in parts.items()}
    progress('fit_conditioned_time',train_races=len(training_keys),round_limit=300)
    bundle=model.fit_models(fit_rows,rounds=300,targets=('win',),
        train_end=date.fromisoformat(plan['train_end']),valid_end=date.fromisoformat(plan['validation_end']),
        test_end=date.fromisoformat(plan['development_end']),calibration_start=date.fromisoformat(plan['calibration_start']),
        calibration_end=date.fromisoformat(plan['calibration_end']),test_start=date.fromisoformat(plan['development_start']),
        _prepared_features=prepared)
    bundle['metadata'].update(pipeline_version='0.3_post_step2',feature_profile='conditioned_previous_time_v1',
        history_start=plan['history_start'],mapping_sha256=safety.digest(directory/'mapping.time.reviewed.json',root=ROOT),
        mapping_path=str(directory/'mapping.time.reviewed.json'),dataset_sha256=metadata['csv_sha256'],
        feature_coverage=coverage,source_plan_sha256=safety.digest(directory/'plan.json',root=ROOT),
        time_encoding='jv_mss_tenths_4',going_availability='disabled_no_historical_pre_race_snapshot')
    new_directory=safety.new_artifact('models')
    safety.write_json(new_directory/'training_selection.local.json',selection)
    bundle['metadata']['training_selection_sha256']=safety.digest(new_directory/'training_selection.local.json')
    model.save_models(bundle,new_directory)
    loaded=model.load_models(new_directory,required_targets=('win',))
    def partition(name):
        ii=bundle['split'][name]
        return v03.prediction_races([fit_rows[i] for i in ii],[bundle['predictions']['win'][i] for i in ii])
    calibration=core.fit_temperature(partition('calibration'),winners,
        model_fit_end=plan['validation_end'],test_start=plan['development_start'])
    new_policy=policy_for(calibration['temperature'])
    policy_dir,_=save_policy(new_policy,safety.digest(new_directory/'model.json'),calibration)
    development=partition('test')
    dev_rows=[fit_rows[i] for i in bundle['split']['test']]
    baseline=model.load_models(safety.ML_ROOT/'models'/MODELS['best242'][0],required_targets=('win',))
    baseline_predictions,_=predict(baseline,dev_rows,rows)
    base_metrics=v03.probability_evaluation(baseline_predictions,winners,policy_for(.75))
    time_metrics=v03.probability_evaluation(development,winners,new_policy)
    adopt=time_metrics['model']['log_loss']<base_metrics['model']['log_loss']
    selected={'name':'conditioned_time' if adopt else 'best242',
        'model_id':new_directory.name if adopt else MODELS['best242'][0],
        'temperature':new_policy.temperature if adopt else .75,
        'policy_id':policy_dir.name if adopt else '921ebbb9f33a49bbb513fe104e1e375b',
        'reason':'lower_development_horse_log_loss' if adopt else 'no_development_improvement',
        'selection_uses_final_period':False,'selection_uses_20261018':False}
    # Persist selection before touching the final-period model predictions.
    write(directory,'selection.json',selected)
    raw_loaded=loaded['boosters']['win'].predict(model.frame([prepared[i] for i in bundle['split']['test']],loaded['metadata']['categories']),num_threads=4)
    reproduction=max(abs(float(a)-float(bundle['predictions']['win'][i])) for a,i in zip(raw_loaded,bundle['split']['test']))
    if reproduction>1e-12:
        raise safety.MLFailure('artifact_invalid')
    result={'conditions':{'baseline':{'status':'measured','metrics':base_metrics},
        'going':{'status':'withheld','reason':'historical_pre_race_going_unavailable'},
        'time':{'status':'measured','model_id':new_directory.name,'temperature':new_policy.temperature,'metrics':time_metrics},
        'both':{'status':'withheld','reason':'historical_pre_race_going_unavailable'}},
        'selection':selected,'time_minus_baseline':paired(development,baseline_predictions,winners,new_policy,policy_for(.75)),
        'uncalibrated_T1':{'baseline':v03.probability_evaluation(baseline_predictions,winners,policy_for(1.)),
                           'time':v03.probability_evaluation(development,winners,policy_for(1.))},
        'fixed_T075':{'time':v03.probability_evaluation(development,winners,policy_for(.75))},
        'calibration':calibration,'coverage':coverage,'training_rounds':bundle['metadata']['training_rounds'],
        'model_reload_max_difference':reproduction,'new_model_id':new_directory.name,'new_policy_id':policy_dir.name}
    write(directory,'feature_results.json',result)
    del matrix,prepared,fit_rows,rows
    gc.collect()
    final_rows,_=data.read_dataset(directory/'evaluation'/'runners.csv',root=ROOT)
    history,_=data.read_dataset(directory/'evaluation_history'/'runners.csv',root=ROOT)
    cohort,final_winners,exclusions=eligible_cohort(final_rows,plan['final_start'],plan['final_end'])
    all_predictions={}; all_metrics={}; all_policies={}
    for name,(identifier,temp) in dict(MODELS,selected=(selected['model_id'],selected['temperature'])).items():
        progress('locked_final_evaluation',model=name)
        predictor=model.load_models(safety.ML_ROOT/'models'/identifier,required_targets=('win',))
        ps,_=predict(predictor,cohort,history)
        all_predictions[name]=ps; all_policies[name]=policy_for(temp)
        all_metrics[name]=v03.probability_evaluation(ps,final_winners,all_policies[name])
    final={'metrics':all_metrics,'exclusions':exclusions,
        'selected_minus_best242':paired(all_predictions['selected'],all_predictions['best242'],final_winners,all_policies['selected'],all_policies['best242']),
        'best242_minus_max100':paired(all_predictions['best242'],all_predictions['max100'],final_winners,policy_for(.75),policy_for(.75)),
        'market':market_results(all_predictions['selected'],final_winners,read(directory,'market_inputs.local.json'),all_policies['selected']),
        'selection_was_frozen_before_final_prediction':True}
    write(directory,'final_evaluation.json',final)
    return {'development':{name:item['metrics']['model'] for name,item in result['conditions'].items() if item['status']=='measured'},
        'selected':selected,'coverage':coverage,'calibration':calibration,'training_rounds':result['training_rounds'],
        'final_metrics':{name:value['model'] for name,value in all_metrics.items()},
        'final_selected_minus_best242':final['selected_minus_best242']}

def frozen_predictions():
    document=safety.read_json(FROZEN/'run.json',root=FROZEN.parents[1])
    if safety.digest(FROZEN/'predictions.csv',root=FROZEN.parents[1])!=document['predictions_sha256']:
        raise safety.MLFailure('artifact_invalid')
    rows=list(csv.DictReader((FROZEN/'predictions.csv').open(encoding='utf-8-sig',newline='')))
    raw=defaultdict(dict); starts={}; saved_probabilities={}
    for row in rows:
        identity,_=features.entry_identity(row)
        if row['model_id']!=MODELS['best242'][0] or float(row['temperature'])!=.75 or identity in saved_probabilities:
            raise safety.MLFailure('artifact_invalid')
        raw[identity[:-1]][identity[-1]]=float(row['raw_win_score'])
        saved_probabilities[identity]=float(row['win_probability'])
        starts[identity[:-1]]=core.instant(row['scheduled_start_jst'])
    predictions=[core.PredictionRace(race,values) for race,values in sorted(raw.items())]
    for p in predictions:
        probabilities,_=core.probabilities(p.probabilities,policy_for(.75))
        if any(abs(float(probabilities[h])-saved_probabilities[(*p.race,h)])>1e-12 for h in probabilities):
            raise safety.MLFailure('artifact_invalid')
    return document,predictions,starts

def confirmed_hr_source(races):
    prior=local.ROOT/'simulations'/'b465a4023292420ba075385876d8d71c'
    manifest=local.read(prior/'manifest.json')
    if (manifest['provenance'].get('hr_source_confirmed') is not True
            or manifest['provenance'].get('odds_raw_format_and_scale_human_confirmed') is not True
            or local.sha(prior/'hr_payout_source.json')!=manifest['provenance']['hr_source_sha256']):
        raise safety.MLFailure('artifact_invalid')
    source=formats.validate_payout_source(local.read(prior/'hr_payout_source.json'))
    if source['numerator']!=1 or source['denominator']!=1:
        raise safety.MLFailure('mapping_not_reviewed')
    source=dict(source,years=sorted({int(r[0]) for r in races}),venues=sorted({r[2] for r in races}),race_keys=[list(r) for r in races])
    return source

def make_observed_snapshots(directory,predictions,starts):
    rosters={p.race:tuple(p.probabilities) for p in predictions}
    result=defaultdict(list)
    paths=sorted((directory/'odds_snapshots').glob('*.json')) if (directory/'odds_snapshots').exists() else []
    for path in paths:
        content=read(path.parent,path.name)
        observed_at=core.instant(content['read_completed_at_jst'])
        schedules={tuple(r['race_key']):r for r in content['ra_rows']}
        grouped=defaultdict(list)
        for row in content['odds_rows']:
            grouped[tuple(row['race_key'])].append(row)
        for race,versions in grouped.items():
            if race not in rosters:
                continue
            record=schedules.get(race,{})
            # Final/deleted/unknown records are unusable availability barriers,
            # never a reason to fall back to an older normal snapshot. The
            # barrier clock is explicitly local, not a fabricated JRA clock.
            if any(r.get('status') not in ('1',) for r in versions):
                result[race].append(core.Snapshot(race,observed_at,observed_at,{},
                    'local_unusable_record_barrier',True,'unsupported_status','second'))
                continue
            stamps=defaultdict(list)
            for row in versions:
                try:stamp=formats.parse_time(row.get('published'),'mmddhhmm',race)
                except core.SimulationError:
                    result[race].append(core.Snapshot(race,observed_at,observed_at,{},
                        'local_unusable_record_barrier',True,'odds_unverified','second'))
                    continue
                stamps[stamp].append(row)
            for published,copies in stamps.items():
                reason=None; prices={}
                row=min(copies,key=lambda r:r['source_table']!='apd_sokuho_o1')
                try:
                    if (record.get('status')!='2' or record.get('record_type')!='RA'
                            or formats.parse_time(record.get('start'),'hhmm_on_race_date',race)!=starts[race]):
                        reason='schedule_conflict'
                    sources=[r['source_table'] for r in copies]
                    if len(set(sources))!=len(sources) or len({market._signature(r) for r in copies})!=1:
                        reason='odds_conflict'
                    if reason is None:prices,_=market.prices(row,rosters[race])
                except core.SimulationError as exc:reason=exc.code
                result[race].append(core.Snapshot(race,published,observed_at,prices,row['source_table'],True,reason,'minute'))
    return result

def save_due_decisions(directory):
    document,predictions,starts=frozen_predictions()
    policy=replace(policy_for(.75,'2026-10-18','2026-10-18'),timing_basis='observed')
    snapshots=make_observed_snapshots(directory,predictions,starts)
    store=safety.local_path(directory/'decisions',root=ROOT); store.mkdir(exist_ok=True)
    created=already=0; reasons=Counter()
    for p in predictions:
        path=store/(MODELS['best242'][0]+'_'+'_'.join(p.race)+'.json')
        if path.exists():
            already+=1
            continue
        schedule=core.Schedule(p.race,starts[p.race],core.instant(document['data_read_completed_at_jst']),tuple(p.probabilities),True)
        row,reason=observed.decision(p,schedule,snapshots[p.race],policy,recorded_at=now())
        if row is None:
            reasons[reason]+=1
            continue
        row.update(model_id=MODELS['best242'][0],temperature=.75,prediction_created_at=document['first_prediction_at_jst'])
        try:
            write(store,path.name,row)
            path.chmod(0o444)
            created+=1; reasons[row['reason']]+=1
        except safety.MLFailure as exc:
            if exc.code!='artifact_exists':raise
            already+=1
    return {'new_decisions':created,'existing_decisions_preserved':already,'reasons':dict(reasons)}

def capture_once(directory):
    if now().date()!=date(2026,10,18):
        raise safety.MLFailure('invalid_date')
    _,predictions,starts=frozen_predictions()
    races=tuple(p.race for p in predictions)
    confirmed_hr_source(races) # Reuse the previously human-confirmed raw O1 mapping.
    begin=now()
    with session() as db:
        odds=fetch_market(db,races)
        budget=local.Budget(30)
        fields={k:publication.RA_FIELDS[k] for k in ('record_type','status','start')}
        columns=publication._columns(db,'jvd_ra',fields,budget)
        if columns is None:raise safety.MLFailure('invalid_schema')
        ra=local._rows(db,budget,'jvd_ra',fields,columns,races,100,join_targets=True)
    end=now()
    store=safety.local_path(directory/'odds_snapshots',root=ROOT); store.mkdir(exist_ok=True)
    write(store,end.strftime('%Y%m%dT%H%M%S%f')+'.json',{'read_started_at_jst':begin.isoformat(),
        'read_completed_at_jst':end.isoformat(),'acquisition_meaning':'this_program_read_completion_not_PC_KEIBA_original_ingestion',
        'odds_rows':odds,'ra_rows':ra})
    decisions=save_due_decisions(directory)
    return {'local_read_at':end.isoformat(),'snapshot_rows':len(odds),
        'categories':dict(Counter(r['status'] for r in odds)),**decisions}

def capture(directory):
    result=capture_once(directory)
    # A bounded optional loop is deliberately separate from Windows tasks.
    finish=time.monotonic()+CAPTURE_DURATION
    count=1
    while time.monotonic()<finish:
        time.sleep(min(30,max(.1,finish-time.monotonic())))
        if time.monotonic()>=finish:break
        result=capture_once(directory); count+=1
        progress('captured',**result)
    return dict(result,capture_count=count)

def result_cohort(se_rows,outcomes,predictions,identities):
    """Never turn a scratched, missing or pending runner into a loss label."""
    grouped=defaultdict(list)
    for row in se_rows:grouped[tuple(row['race_key'])].append(row)
    winners={}; ranks={}; reasons=Counter(); valid={}
    for p in predictions:
        race=p.race; found={}; reason=None
        for row in grouped[race]:
            h=row.get('horse_number')
            if (h not in p.probabilities or h in found or row.get('horse_id')!=identities.get((*race,h))):
                reason='result_identity_mismatch'; break
            found[h]=row
        if reason is None and set(found)!=set(p.probabilities):reason='result_roster_missing'
        outcome=outcomes.get(race)
        if reason is None and core.outcome_reason(outcome,tuple(p.probabilities)) is not None:
            reason='hr_not_normal_confirmed'
        rr={}
        if reason is None:
            for h,row in found.items():
                if row.get('status') not in ('5','6','7') or row.get('anomaly') not in ('0','4'):
                    reason='cancelled_unknown_or_unconfirmed_result';break
                rank=features.number(row.get('rank'),low=0,high=28)
                if rank is None or not rank.is_integer() or (rank==0 and row.get('anomaly')!='4'):
                    reason='rank_unconfirmed';break
                rr[h]=int(rank)
        if reason is None:
            winning={h for h,rank in rr.items() if rank==1}
            if winning!=set(outcome.payouts_yen_per_100) or len(winning)!=1:
                reason='multiple_winners_or_hr_se_mismatch'
        if reason:
            reasons[reason]+=1
        else:
            winners[race]=next(iter(winning)); ranks[race]=rr;valid[race]=outcome
    return winners,ranks,valid,dict(reasons)

def replay(directory):
    plan=read(directory,'plan.json'); selected=read(directory,'selection.json')
    frozen_run,baseline_predictions,starts=frozen_predictions()
    original_entries,_=data.read_dataset(FROZEN/'entries'/'runners.csv',root=FROZEN.parents[1],allow_prediction_snapshot=True)
    original_history,_=data.read_dataset(FROZEN/'history'/'runners.csv',root=FROZEN.parents[1])
    enriched_history,_=data.read_dataset(directory/'trial_history'/'runners.csv',root=ROOT)
    baseline_bundle=model.load_models(safety.ML_ROOT/'models'/MODELS['best242'][0],required_targets=('win',))
    reproduced,_=predict(baseline_bundle,original_entries,original_history)
    old_by_race={p.race:p for p in baseline_predictions}
    maximum=max(abs(float(p.probabilities[h])-float(old_by_race[p.race].probabilities[h])) for p in reproduced for h in p.probabilities)
    if maximum>1e-12:raise safety.MLFailure('artifact_invalid')
    bundle=model.load_models(safety.ML_ROOT/'models'/selected['model_id'],required_targets=('win',))
    new_predictions,coverage=predict(bundle,original_entries,enriched_history)
    generated=now()
    races=tuple(sorted(old_by_race)); identities={features.entry_identity(r)[0]:r['horse_id'] for r in original_entries}
    with session() as db:
        budget=local.Budget(60)
        fields={'status':('data_kubun',1),'horse_number':('umaban',2),
            'horse_id':('ketto_toroku_bango',10),'rank':('kakutei_chakujun',2),'anomaly':('ijo_kubun_code',1)}
        columns=publication._columns(db,'jvd_se',fields,budget)
        if columns is None:raise safety.MLFailure('invalid_schema')
        se=local._rows(db,budget,'jvd_se',fields,columns,races,2000,join_targets=True)
        source=confirmed_hr_source(races)
        hr,source,hr_status=settle_saved.fetch_hr(db,races,reviewed_source=source)
        quotes=fetch_market(db,races)
    collected=now()
    outcomes={o.race:o for o in settle_saved.decode_existing_hr(hr,source)}
    winners,ranks,valid,exclusions=result_cohort(se,outcomes,baseline_predictions,identities)
    sub=safety.new_artifact('evaluations',root=directory)
    write(sub,'results_input.local.json',{'se_rows':se,'hr_rows':hr,'odds_rows':quotes,'read_completed_at_jst':collected.isoformat()})
    models={'saved_best242':(baseline_predictions,.75),'selected_replay':(new_predictions,selected['temperature'])}
    measurements={}; private_rows=[]; csv_rows=[]
    quoted=defaultdict(list)
    for q in quotes:quoted[tuple(q['race_key'])].append(q)
    observed_inputs=make_observed_snapshots(directory,baseline_predictions,starts)
    comparison=[]; top_changes=0
    for label,(predictions,temp) in models.items():
        policy=policy_for(temp,'2026-10-18','2026-10-18')
        probability=v03.probability_evaluation(predictions,winners,policy)
        top_rows=[]; final_rows=[]; observed_rows=[]; price_reasons=Counter()
        for p in predictions:
            probabilities,_=core.probabilities(p.probabilities,policy)
            top=min(probabilities,key=lambda h:(-probabilities[h],h))
            quote,reason=market.select_final(p.race,quoted[p.race],tuple(p.probabilities))
            outcome=valid.get(p.race)
            favorite=min(quote.odds,key=lambda h:(quote.odds[h],h)) if quote else None
            pending=outcome or core.Outcome(p.race,'pending',reason='result_unconfirmed')
            top_rows.append({'month':'2026-10','scheduled_start':starts[p.race].isoformat(),
                'ai':core.settle(top,pending,tuple(probabilities)),'favorite':core.settle(favorite,pending,tuple(probabilities))})
            if quote:
                expected={h:probabilities[h]*core.decimal(quote.odds[h])-1 for h in probabilities}
                candidates=sorted((h for h in expected if expected[h]>=core.decimal('.05')),key=lambda h:(-expected[h],h))
                chosen=candidates[0] if candidates else None
                final_rows.append({'month':'2026-10','scheduled_start':starts[p.race].isoformat(),
                    'ai':core.settle(chosen,pending,tuple(probabilities)),'favorite':core.settle(favorite,pending,tuple(probabilities))})
            else:price_reasons[reason]+=1
            schedule=core.Schedule(p.race,starts[p.race],core.instant(frozen_run['data_read_completed_at_jst']),tuple(p.probabilities),True)
            row,_=observed.decision(p,schedule,observed_inputs[p.race],replace(policy,timing_basis='observed'),recorded_at=collected)
            if row:
                row['ai']=core.settle(row['ai_horse'],pending,tuple(probabilities))
                row['favorite']=core.settle(row['favorite_horse'],pending,tuple(probabilities))
                row['common_settlement_eligible']=row['simulation_eligible'] and p.race in valid
                row['settlement_reason']=core.outcome_reason(pending,tuple(probabilities))
                row.update(model_id=MODELS['best242'][0] if label=='saved_best242' else selected['model_id'],temperature=temp,
                    prediction_created_at=frozen_run['first_prediction_at_jst'] if label=='saved_best242' else generated.isoformat(),
                    retrospective_model_replay=label=='selected_replay')
                observed_rows.append(row)
            else:
                # Retain all original races in the coverage denominator, but
                # do not make an early EV choice for a future decision time.
                waiting=core.run_simulation([p],[schedule],[],[],policy=replace(policy,timing_basis='observed'))['rows'][0]
                waiting.update(reason='decision_not_due',decision_class='not_due',actual_bet_placed=False)
                observed_rows.append(waiting)
            ordering=sorted(probabilities,key=lambda h:(-probabilities[h],h))
            detail={'model_label':label,'race_key':list(p.race),'top_horse':top,'top_probability':float(probabilities[top]),
                'official_rank':ranks.get(p.race,{}).get(top),'top1_hit':top==winners[p.race] if p.race in winners else None,
                'top3_hit':1<=ranks[p.race][top]<=3 if p.race in ranks else None,
                'scheduled_start':starts[p.race].isoformat(),'status':'completed' if p.race in winners else 'pending_or_excluded',
                'pre_race_decision':row,'final_odds_reference':final_rows[-1] if quote else None}
            private_rows.append(detail)
            for rank,h in enumerate(ordering,1):
                csv_rows.append(dict(zip(RACE_KEY,p.race),model_label=label,model_id=MODELS['best242'][0] if label=='saved_best242' else selected['model_id'],
                    horse_number=h,horse_id=identities[(*p.race,h)],win_probability=float(probabilities[h]),prediction_rank=rank,
                    temperature=temp,generated_at_jst=frozen_run['first_prediction_at_jst'] if label=='saved_best242' else generated.isoformat(),
                    mode='original_frozen_pre_race' if label=='saved_best242' else 'posthoc_historical_input_replay'))
        monthly=core.summarize(observed_rows) if observed_rows else {'target_race_count':0,'by_month':{}}
        month_risk={month:{strategy:observed.risk_metrics([r for r in observed_rows if r['month']==month],strategy)
            for strategy in ('ai','favorite')} for month in monthly['by_month']}
        top3=[d['top3_hit'] for d in private_rows if d['model_label']==label and d['top3_hit'] is not None]
        measurements[label]={'probability':probability,'top3_reference':{'races':len(top3),'hit_count':sum(top3),
            'hit_rate_percent':100*sum(top3)/len(top3) if top3 else None,'definition':'rank<=3_not_verified_place_ticket_label'},
            'observed_before_deadline':{'summary':monthly,'risk_by_month':month_risk,
                'mode':'historical_input_replay_for_new_model' if label=='selected_replay' else 'frozen_predictions_and_timestamped_odds',
                'target_races':36,'not_yet_due_races':sum(r['reason']=='decision_not_due' for r in observed_rows)},
            'top1_flat_100_reference':{s:core.financial(top_rows,s) for s in ('ai','favorite')},
            'final_odds_EV_reference':{'mode':'postrace_reference_not_pre_race_decisions',
                'races_with_final_odds':len(final_rows),'exclusions':dict(price_reasons),
                'financial':{s:core.financial(final_rows,s) for s in ('ai','favorite')},
                'risk':{s:observed.risk_metrics(final_rows,s) for s in ('ai','favorite')}}}
    by_label={label:{tuple(r['race_key']):r for r in private_rows if r['model_label']==label} for label in models}
    for race in races:
        a=by_label['saved_best242'][race]; b=by_label['selected_replay'][race]
        same=a['top_horse']==b['top_horse'];top_changes+=not same
        comparison.append({'race_key':list(race),'venue':VENUES.get(race[2],race[2]),'race_number':int(race[-1]),
            'scheduled_start':starts[race].isoformat(),'baseline_top_horse':a['top_horse'],'baseline_top_probability':a['top_probability'],
            'selected_top_horse':b['top_horse'],'selected_top_probability':b['top_probability'],'same_top1':same,
            'baseline_rank':a['official_rank'],'selected_rank':b['official_rank'],'baseline_hit':a['top1_hit'],'selected_hit':b['top1_hit'],
            'baseline_top3':a['top3_hit'],'selected_top3':b['top3_hit'],
            'pre_race_odds_status':a['pre_race_decision']['reason'] if a['pre_race_decision'] else 'not_yet_due',
            'baseline_ev_horse':a['pre_race_decision']['ai_horse'] if a['pre_race_decision'] else None,
            'selected_ev_horse':b['pre_race_decision']['ai_horse'] if b['pre_race_decision'] else None,
            'baseline_observed_settlement':a['pre_race_decision']['ai'] if a['pre_race_decision'] else None,
            'selected_observed_settlement':b['pre_race_decision']['ai'] if b['pre_race_decision'] else None,
            'result_status':a['status']})
    fields=list(csv_rows[0])
    with safety.local_path(sub/'predictions.local.csv',root=ROOT).open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader();writer.writerows(csv_rows)
    write(sub,'comparison.local.json',comparison)
    write(sub,'ledgers.local.json',private_rows)
    verification={path:safety.digest(Path(path),root=FROZEN.parents[1])==digest for path,digest in plan['frozen_20261018_artifacts'].items()}
    if not all(verification.values()):raise safety.MLFailure('artifact_invalid')
    result={'as_of_jst':collected.isoformat(),'target_races':36,'target_runners':450,'completed_evaluable_races':len(winners),
        'pending_or_excluded_races':36-len(winners),'result_exclusions':exclusions,'hr_source':hr_status,
        'model_comparison':measurements,'top1_changed_races':top_changes,'old_prediction_reproduction_max_difference':maximum,
        'original_artifacts_unchanged':all(verification.values()),'selected_model':selected,'feature_coverage':coverage,
        'market_reference':market_results([p for p in baseline_predictions if p.race in winners],winners,quotes,policy_for(.75)) if winners else {},
        'replay_limitations':['new model is posthoc, not a new original pre-race forecast',
            'entry fields and baseline history reuse the actual 09:26 snapshot',
            'time values for races through 2026-10-11 were read later; correction availability at 09:26 is unverified',
            'going stays disabled; no result/odds from 2026-10-18 enters features or model selection'],
        'monthly_historical_revenue_2025':{'status':'not_evaluated','reason':'no_pre_race_odds_history',
            'net_profit_yen':None,'return_rate_percent':None,'max_drawdown_yen':None},
        'report_directory':str(sub)}
    write(sub,'summary.json',result)
    lines=['# 2026-10-18 モデル比較（事後入力再現）','',
        '元の09:26予測は変更していません。改善候補の出力は事後シミュレーションです。',
        f"結果取得時点: {collected.isoformat()}。評価可能{len(winners)}R、未確定・除外{36-len(winners)}R。",'',
        '| 場 | R | 発走 | best242馬番/勝率 | 新候補馬番/勝率 | 一致 | 旧/新着順 | 旧/新1着 | 旧/新3着以内 | 状態 |',
        '|---|---:|---|---|---|---|---|---|---|---|']
    for r in sorted(comparison,key=lambda r:(r['scheduled_start'],r['venue'])):
        lines.append(f"| {r['venue']} | {r['race_number']} | {r['scheduled_start'][11:16]} | {r['baseline_top_horse']} / {r['baseline_top_probability']:.2%} | {r['selected_top_horse']} / {r['selected_top_probability']:.2%} | {r['same_top1']} | {r['baseline_rank']}/{r['selected_rank']} | {r['baseline_hit']}/{r['selected_hit']} | {r['baseline_top3']}/{r['selected_top3']} | {r['result_status']} |")
    lines+=['','| 場 | R | 発走前オッズ状態 | 旧/新EV選択馬 | 旧投資/払戻 | 新投資/払戻 |',
        '|---|---:|---|---|---|---|']
    for r in sorted(comparison,key=lambda r:(r['scheduled_start'],r['venue'])):
        a=r['baseline_observed_settlement'];b=r['selected_observed_settlement']
        ap=f"{a['stake_yen']}/{a['payout_yen']}" if a else '判断前'
        bp=f"{b['stake_yen']}/{b['payout_yen']}" if b else '判断前'
        lines.append(f"| {r['venue']} | {r['race_number']} | {r['pre_race_odds_status']} | {r['baseline_ev_horse']}/{r['selected_ev_horse']} | {ap} | {bp} |")
    lines+=['','予測1位とEV最大馬はledgers.local.jsonで別項目。実取得時刻のない確定オッズは参考欄だけです。',
        '未確定レースの的中・収益は未算出。後からreplayを再実行すると、新規フォルダへ更新版を保存します。']
    safety.local_path(sub/'comparison.local.md',root=ROOT).write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    return {'report_directory':str(sub),'completed_evaluable_races':len(winners),'pending_or_excluded_races':36-len(winners),
        'top1_changed_races':top_changes,'probability_metrics':{k:v['probability'] for k,v in measurements.items()},
        'original_artifacts_unchanged':True,'reproduction_max_difference':maximum}

VENUES={'01':'札幌','02':'函館','03':'福島','04':'新潟','05':'東京','06':'中山','07':'中京','08':'京都','09':'阪神','10':'小倉'}

def report(directory):
    independent=read(directory,'independent.json')['confirmation']
    experiment=read(directory,'feature_results.json')
    final=read(directory,'final_evaluation.json')
    selected=read(directory,'selection.json')
    inventory_data=read(directory,'inventory.json')
    candidates=[]
    for path in (directory/'evaluations').glob('*/summary.json'):
        value=read(path.parent,path.name)
        if value.get('target_races')==36:candidates.append((path,value))
    if not candidates:raise safety.MLFailure('insufficient_history')
    replay_path,replayed=max(candidates,key=lambda item:item[0].stat().st_mtime_ns)
    output=safety.new_artifact('evaluations',root=directory)
    aggregate={'kind':'post_step2_summary','created_at_jst':now().isoformat(),'confirmation':independent,
        'feature_experiment':experiment,'final_evaluation':final,'selection':selected,
        'trial_replay':{k:v for k,v in replayed.items() if k!='feature_coverage'},
        'deployment_changed':False,'old_0700_task_changed':False,
        'original_predictions_preserved':replayed['original_artifacts_unchanged'],
        'input_verification':read(directory,'verification.json'),
        'replay_summary_path':str(replay_path),'report_directory':str(output)}
    write(output,'summary.json',aggregate)
    lines=['# STEP 2後の3課題：実測結果','',
        '旧モデル・朝タスク・10/18 09:26の予測は保持。新モデルは研究候補であり、運用切替は行っていません。','',
        '学習は2016-07-02～2024-11-30の同一26,173R・363,048行。早期終了2024-12-08～12-28、校正2025-01-11～02-02。',
        '特徴量選択は既知2025-03～05のみ。確認2025-06～08、選択後の最終評価2025-09～11は事前固定。',
        '温度は既存のレース単位Log Lossで選択。主比較は馬単位二値Log Loss。順位同率は確率評価で按分、仮想馬券は最小馬番。','',
        '| 期間 | モデル | R/頭数 | 馬LL | Brier | レースLL | 1位的中率% | 勝馬平均順位 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for label,container in (('6～8月・独立確認',independent),('9～11月・最終',final)):
        for name,measured in container['metrics'].items():
            m=measured['model'];lines.append(f"| {label} | {name} | {measured['races']}/{m['count']} | {m['log_loss']:.6f} | {m['brier_score']:.6f} | {m['race_log_loss']:.6f} | {m['top1_accuracy_percent']:.3f} | {m['winner_mean_rank']:.3f} |")
        for basis,measured in container['market']['by_timing'].items():
            m=measured['metrics']['market'];lines.append(f"| {label} | 市場({basis},事後参考) | {measured['race_count']}/{m['count']} | {m['log_loss']:.6f} | {m['brier_score']:.6f} | {m['race_log_loss']:.6f} | {m['top1_accuracy_percent']:.3f} | {m['winner_mean_rank']:.3f} |")
    lines+=['','## 1. 通常上限とbest242','']
    for label,comparison in (('独立確認',independent['best242_minus_max100']),('最終',final['best242_minus_max100']),('最終・タイム候補－best242',final['selected_minus_best242'])):
        c=comparison['log_loss'];lines.append(f"- {label}: Δ馬LL={c['difference_new_minus_baseline']:.8f}、95%区間[{c['ci95_low']:.8f},{c['ci95_high']:.8f}]。")
    lines+=['','レース単位ペアードbootstrap 1000回。馬数で重み付けした従来指標。開催日・同馬の依存は追加補正なし、多重比較補正なし。',
        '6～8月のbest242優位は不明確。最終のタイム追加差も0をまたぎ、明確な精度向上とは認定しません。','',
        '旧V0.3は学習期間も異なり、旧→新の差を学習量だけの効果とはみなしません。100上限対best242は同じ学習量・T=0.75でラウンドの効果を比較。',
        '## 2. 特徴量と候補選択','',
        '馬場状態は保留。WEは2026年8行のみで2024/2025年履歴がなく、最終RAの芝・ダート馬場を発走前状態として代用していません。',
        '4条件のうち「現行」「過去タイム」は測定、「馬場のみ」「両方」はデータ不足で未実験。',
        '比較前後の479,187入力行と朝の履歴3,702行は、追加タイム列以外の変更0行を確認。',
        'JV SE項番44の4文字MSSdを秒へ明示変換。0000、秒60以上、不正文字・桁は欠損。倍率推測や×0.1による分秒誤変換はありません。',
        '過去タイムは同距離・同一トラックコードの利用可能な直近有効値。過去結果は7日遅れ。現在レースのタイムは特徴量にしません。',
        f"新モデル: {experiment['new_model_id']} / best={experiment['training_rounds']['win']['best_iteration']} / T={selected['temperature']}。",
        f"学習時タイム欠損: 363,048/363,048 → {experiment['coverage']['train']['missing_count']['previous_time_seconds']:,}/363,048 ({experiment['coverage']['train']['missing_rate']['previous_time_seconds']:.2%})。",
        f"開発LL: {experiment['conditions']['baseline']['metrics']['model']['log_loss']:.8f} → {experiment['conditions']['time']['metrics']['model']['log_loss']:.8f}。",
        '開発主指標の点推定のみで試験候補を選択しselection.jsonを保存後、最終評価を実施。最終結果や10/18結果で選択し直していません。','',
        '## 3. オッズ・月別仮想収益','',
        '2025年O1は確定区分5のみ。発走前オッズがないため2025年のEV利益・ROI・DDは未算出。確定価格による利益で代用しません。',
        'apd_sokuho_o1は2026-10-03～04の発表履歴を保持。10/18はjvd_o1の最新値があり、PC側取得時計は両表にありません。',
        '今回のローカル読込完了時刻付きスナップショットを新規保存。これはPC-KEIBA本来の取込時刻を復元したものではありません。',
        '発走10分前、EV5%、最大1頭100円、最大EV。公表・取得の両時計が期限以前、許容経過120秒。期限後・未確認・不完全・古い値は見送り。',
        '予測1位とEV選択馬は別項目。判断ファイルはモデルID＋完全レースキーで排他的作成し、再実行しても二重計上・再選択しません。',
        '月別件数・ROI・未精算・DD・最大払戻への依存は各再生summary.json内。未精算があれば全体損益を確定せず、精算済み部分を分離します。',
        'DDは予定発走時刻順・同時刻を合算した仮想精算順による参考値。実購入の資金推移ではありません。','',
        '## 4. 10月18日・全36Rの事後比較','',
        f"結果取得時点 {replayed['as_of_jst']}。36R/450頭を再予測、1位変更{replayed['top1_changed_races']}R、結果照合可能{replayed['completed_evaluable_races']}R、未確定・除外{replayed['pending_or_excluded_races']}R。",
        '元モデルの保存済み生予測の再現差は0.0。元CSVや温度を更新していません。',
        '新候補は事後の入力再現であり、前向き検証成功ではありません。タイム列は後刻取得なので9:26時点の訂正履歴までは証明できません。',
        f"個別比較（PC内限定）: {replay_path.parent/'comparison.local.md'}",'',
        '| 予測 | 評価R | 馬LL | Brier | 1位的中率% | 3着以内率% |',
        '|---|---:|---:|---:|---:|---:|']
    for name,value in replayed['model_comparison'].items():
        p=value['probability']; m=p.get('model',{});top3=value['top3_reference']
        if m:lines.append(f"| {name} | {p['races']} | {m['log_loss']:.6f} | {m['brier_score']:.6f} | {m['top1_accuracy_percent']:.3f} | {top3['hit_rate_percent']:.3f} |")
    lines+=['','| 2026-10・発走前観測 | 全対象R | 取得R | 取得率% | 購入 | 投資円 | 未精算 | 純利益円 | ROI% |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for name,value in replayed['model_comparison'].items():
        forward=value['observed_before_deadline']['summary']; money=forward.get('ai',{})
        lines.append(f"| {name} | {forward.get('target_race_count')} | {forward.get('odds_acquired_race_count')} | {forward.get('odds_acquisition_rate_percent')} | {money.get('bet_count')} | {money.get('investment_yen')} | {money.get('unsettled_count')} | {money.get('net_profit_yen')} | {money.get('return_rate_percent')} |")
    lines+=['','Noneは未算出・未確定。未来の判断時刻に達していない対象はdecision_not_dueであり、失敗やEV見送りと別集計。',
        '確定オッズEVはsummary.jsonのfinal_odds_EV_referenceへ別保存。発走前EVの実績へ合算していません。','',
        '## 最終期間の確率帯別校正','',
        '| 予測 | 確率帯 | 頭数 | 平均予測% | 実勝率% |', '|---|---|---:|---|---|']
    calibration_sets={name:final['metrics'][name]['low_probability_calibration'] for name in ('best242','selected')}
    for basis,item in final['market']['by_timing'].items():calibration_sets['market_'+basis]=item['calibration']['market']
    for name,bins in calibration_sets.items():
        for item in bins:
            mean=item['mean_probability']; actual=item['observed_rate']
            mean='未算出' if mean is None else f'{100*mean:.3f}'
            actual='未算出' if actual is None else f'{100*actual:.3f}'
            lines.append(f"| {name} | {market.BIN_NAMES[item['bin']]} | {item['count']} | {mean} | {actual} |")
    lines+=['',
        '## 次の優先課題（3件）','',
        '1. 発走前の観測と固定判断を継続し、複数月の同条件収益を蓄積する。',
        '2. 今回は優位が不確か。現行運用を維持し、タイム候補を新しい前向き期間で比較する。',
        '3. 学習データの新しさを別実験にし、2024年11月終端から2026年への時間差を検証する。','',
        '回帰テスト結果: プロジェクト reports/post_step2_tests.txt。原データ・個別表はOneDrive外のみ。']
    safety.local_path(output/'report.md',root=ROOT).write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
    return {'report_directory':str(output),'replay_directory':str(replay_path.parent),
        'original_predictions_preserved':True,'deployment_changed':False}

CAPTURE_DURATION=0

def main():
    global CAPTURE_DURATION
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('inventory','collect','evaluate','features','replay','capture','report'),required=True)
    parser.add_argument('--work-id')
    parser.add_argument('--duration-seconds',type=int,default=0)
    args=parser.parse_args()
    if not 0<=args.duration_seconds<=600:
        parser.error('duration must be between 0 and 600 seconds')
    CAPTURE_DURATION=args.duration_seconds
    directory=None
    result={'status':'stopped','stage':args.stage,'started_at_jst':now().isoformat()}
    try:
        if args.stage=='inventory' and not args.work_id:
            directory=safety.new_artifact('evaluations',root=ROOT)
        else:
            if not args.work_id or not re.fullmatch('[a-f0-9]{32}',args.work_id):
                raise safety.MLFailure('invalid_input')
            directory=safety.local_path(ROOT/'evaluations'/args.work_id,root=ROOT)
            read(directory,'plan.json')
        value=(features_experiment if args.stage=='features' else globals()[args.stage])(directory)
        result.update(status='ok',**value)
    except SafeAuditError as exc:
        result.update(reason=exc.code)
    except Exception:
        result.update(reason='internal_error')
    result.update(finished_at_jst=now().isoformat(),work_id=directory.name if directory else None)
    if directory:
        write(directory,args.stage+'-'+now().strftime('%H%M%S%f')+'.json',result)
    print(json.dumps(result,ensure_ascii=True,allow_nan=False),flush=True)
    return 0 if result['status']=='ok' else 2
