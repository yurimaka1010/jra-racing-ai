"""Answers are opened only for previously committed future records, after race day."""
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import date
from pathlib import Path
from uuid import uuid4
import re
import pandas as pd
import numpy as np

from ..ml import safety
from ..win_sim import core,local,settle_saved
from ..duel.replay import PAYOUT
from ..tuner_four_models import distance_group,surface_group
from . import contracts as c,learning
from .forward import NAMES,mkdir,model_names,record_subdir,FOUR_NAMES
from .report import csv_file


def committed(folder,plan):
    if not (folder/'receipt.json').exists() or (folder/'veto.json').exists():return None
    receipt=c.read(folder/'receipt.json')
    if receipt['status']!='saved':return None
    if (c.sha(folder/'prediction.json')!=receipt['prediction_sha256']
            or c.sha(folder/'predictions.csv')!=receipt['CSV_sha256']):raise safety.MLFailure('artifact_invalid')
    record=c.read(folder/'prediction.json');cutoff=core.instant(record['input_cutoff']);deadline=core.instant(receipt['deadline'])
    if record.get('synthetic_verification') is True:return None
    if c.sha(Path(record['entry_snapshot']['path']))!=record['entry_snapshot']['sha256']:
        raise safety.MLFailure('artifact_invalid')
    if not (cutoff<=core.instant(record['generated_at'])<=core.instant(receipt['written_at'])<=deadline):
        raise safety.MLFailure('forward_cutoff')
    if record.get('result_accessed') is not False or record['model_ids']!=plan['model_ids']:raise safety.MLFailure('artifact_invalid')
    for path,sha in record['model_references'].items():
        if c.sha(Path(path))!=sha:raise safety.MLFailure('artifact_invalid')
    return record


def reviewed_profile(races):
    manifest=c.read(PAYOUT/'manifest.json');profile=c.read(PAYOUT/'hr_payout_source.json')
    if (manifest['provenance'].get('hr_source_confirmed') is not True
            or c.sha(PAYOUT/'hr_payout_source.json')!=manifest['provenance']['hr_source_sha256']
            or profile['numerator']!=1 or profile['denominator']!=1):raise safety.MLFailure('mapping_not_reviewed')
    # Reuse the same owner-reviewed normal JPY/100 adapter, as existing duel
    # evaluation does. Only normal single-current HR2 is accepted for new keys.
    result=deepcopy(profile)
    result.update(years=sorted({int(r[0]) for r in races}),venues=sorted({r[2] for r in races}),
        race_keys=[list(r) for r in races],allowed_final_categories=['2'],final_category='2')
    return result


def result_labels(record,rows):
    roster=record['registered']['roster'];race=tuple(record['race_key'])
    expected={r['horse_number']:r['horse_id'] for r in c.read(Path(record['entry_snapshot']['path']))['payload']}
    if len(rows)!=len(roster) or {r['horse'] for r in rows}!=set(roster):return None,'result_roster_missing_or_duplicate'
    if any(r.get('status')!='7' or r.get('anomaly') not in ('0','4') or r.get('horse_id')!=expected[r['horse']] for r in rows):
        return None,'result_status_identity_or_cancellation_unconfirmed'
    if any(not isinstance(r.get('rank'),str) or not re.fullmatch('[0-9]{2}',r['rank'])
           or not (1<=int(r['rank'])<=len(roster) or (r['rank']=='00' and r['anomaly']=='4')) for r in rows):
        return None,'result_rank_unconfirmed'
    winner=[r['horse'] for r in rows if r.get('rank')=='01']
    if len(winner)!=1:return None,'multiple_or_unconfirmed_winners'
    return {h:int(h==winner[0]) for h in roster},None


def evaluate(directory,session,day_text,*,four=False):
    c.approved(directory);plan=c.read(directory/('forward_four_plan.json' if four else 'forward_plan.json'))
    names=model_names(plan)
    if four and names!=FOUR_NAMES:raise safety.MLFailure('invalid_model')
    try:day=date.fromisoformat(day_text)
    except (TypeError,ValueError):raise safety.MLFailure('invalid_date') from None
    if not plan['start_date']<=str(day)<=plan['end_date'] or c.clock().date()<=day:
        raise safety.MLFailure('forward_cutoff')
    # Verify every prediction/receipt before opening result tables.
    records=[];coverage=Counter()
    for folder in sorted((directory/record_subdir(plan)/str(day)/'races').glob('*')):
        row=committed(folder,plan)
        if row:records.append((folder,row))
        else:coverage['forecast_missing_or_vetoed']+=1
    if not records:return {'status':'ok','reason':'no_eligible_pre_result_records','races':0,'metrics':None,'ROI':None}
    races=[tuple(record['race_key']) for _,record in records]
    if len(races)>50:raise safety.MLFailure('input_limit')
    answer=defaultdict(list);outcomes={};source_status=[]
    with session() as db:
        budget=local.Budget(120);columns=local.table_info(db,'jvd_se',budget)
        fields={'horse':('umaban',2),'horse_id':('ketto_toroku_bango',10),'rank':('kakutei_chakujun',2),
                'status':('data_kubun',1),'anomaly':('ijo_kubun_code',1)}
        if columns is None or any(columns.get(name,{}).get('type_is_builtin') is not True
                or columns[name]['type_name'] not in local.TEXT_TYPES for name,_ in fields.values()):raise safety.MLFailure('invalid_schema')
        rows=local._rows(db,budget,'jvd_se',fields,columns,races,1500,join_targets=True)
        for row in rows:answer[tuple(row['race_key'])].append(row)
        try:
            hr,source,status=settle_saved.fetch_hr(db,races,max_rows=100,max_seconds=30,reviewed_source=reviewed_profile(races))
            source_status.append(status)
            if source:outcomes={o.race:o for o in settle_saved.decode_existing_hr(hr,source)}
        except (safety.MLFailure,core.SimulationError) as error:
            source_status.append({'status':'pending','reason':error.code})
    eval_dir=mkdir(directory/('future-evaluations-four' if four else 'future-evaluations')/uuid4().hex)
    c.write(eval_dir/'results.local.json',{'day':str(day),'SE_rows':rows,'HR_rows':hr if 'hr' in locals() else [],
        'source_status':source_status,'loaded_after_prediction_verification':True,'loaded_at':c.clock().isoformat()})
    frames=[];money={name:[] for name in names};all_money={name:[] for name in names}
    comparable=[]
    for folder,record in records:
        race=tuple(record['race_key']);labels,reason=result_labels(record,answer[race])
        if reason:coverage[reason]+=1;continue
        probabilities=record['probabilities'];roster=record['registered']['roster']
        if any(set(probabilities[name])!=set(roster) for name in names):raise safety.MLFailure('invalid_identity')
        saved=pd.read_csv(folder/'predictions.csv',dtype={'entry_key':str,'date':str,'model':str,'surface':str},float_precision='round_trip')
        unique=saved.loc[saved.model=='new_D39',['entry_key','date','distance','surface']].copy()
        unique['winner_label']=[labels[k[-2:]] for k in unique.entry_key]
        for name in names:unique[name]=[probabilities[name][k[-2:]] for k in unique.entry_key]
        frames.append(unique)
        outcome=outcomes.get(race,core.Outcome(race,'pending',reason='result_unavailable'))
        winners={h for h,y in labels.items() if y}
        if outcome.status=='normal_final' and set(outcome.payouts_yen_per_100)!=winners:
            outcome=core.Outcome(race,'pending',reason='hr_se_winner_mismatch')
        eligible=core.outcome_reason(outcome,tuple(roster)) is None
        if eligible:comparable.append(''.join(race))
        for name in names:
            ledger=record['ledgers'][name];bet=core.settle(ledger['ai_horse'],outcome,tuple(roster))
            item={'model':name,'month':str(day)[:7],'ai':bet,'decision':ledger,'race_key':list(race)}
            all_money[name].append(item)
            if eligible:money[name].append(item)
    measurements={};groups=[]
    if frames:
        combined=pd.concat(frames,ignore_index=True).reset_index(drop=True)
        for name in names:measurements[name]=learning.point(combined,combined[name].to_numpy(float))
        for axis,column,classifier in (('distance','distance',distance_group),('surface','surface',surface_group)):
            tags=combined[column].map(classifier)
            for tag in sorted(set(tags)):
                f=combined.loc[tags==tag]
                for name in names:
                    groups.append({'axis':axis,'group':tag,'model':name,**{k:v for k,v in learning.point(f,f[name]).items() if k!='calibration_bands'}})
        csv_file(eval_dir/'paired_predictions_and_results.local.csv',combined.to_dict('records'))
    result={'status':'ok','day':str(day),'role':('independent_four_model' if four else 'independent_three_model')+'_forward_diagnostic_not_Stage7_interim',
        'forecast_races':len(records),'common_result_races':sum(f.entry_key.str[:16].nunique() for f in frames),
        'exclusions':dict(coverage),'metrics':measurements,'distance_and_surface':groups,
        'financial_all_valid_result_races':{n:core.financial(all_money[n],'ai') for n in names},
        'financial_common_settlement_races':len(comparable),
        'financial_common':{n:core.financial(money[n],'ai') for n in names},
        'fixed_policy':plan['policy'],'HR_source_status':source_status,'no_model_or_policy_changes':True,
        'price_notice':'T10 displayed price does not guarantee final payout. Virtual bets only; actual start and purchase availability unverified.'}
    csv_file(eval_dir/'distance_surface_comparison.csv',groups)
    c.write(eval_dir/'summary.json',result)
    return {'status':'ok','day':str(day),'report_folder':str(eval_dir),'common_result_races':result['common_result_races'],
        'metrics':{name:{k:v for k,v in values.items() if k!='calibration_bands'} for name,values in measurements.items()},
        'financial_common':result['financial_common'],'exclusions':dict(coverage)}
