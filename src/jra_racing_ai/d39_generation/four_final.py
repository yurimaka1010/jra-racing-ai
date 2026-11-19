"""Single end-of-window comparison. No interim results or performance are opened."""
from collections import Counter,defaultdict
from datetime import date,timedelta
from pathlib import Path
import time
import numpy as np
import pandas as pd

from ..ml import safety
from ..win_sim import core,local,settle_saved
from ..stage5 import metrics
from . import contracts as c,addendum,forward,four_forward,learning
from .report import csv_file


def require_closed(plan,now):
    if now.date()<=date.fromisoformat(plan['end_date']):raise safety.MLFailure('forward_cutoff')


def contrasts(points,draws,confidence):
    output={}
    for first,second,meaning in addendum.PAIRS:
        values={}
        for metric in metrics.NAMES:
            change=np.asarray(draws[first][metric])-np.asarray(draws[second][metric])
            level=confidence if metric=='race_log_loss' else .95
            alpha=(1-level)/2
            values[metric]={'difference':float(points[first][metric]-points[second][metric]),
                'confidence':level,'interval':[float(x) for x in np.quantile(change,[alpha,1-alpha])],
                'negative_is_improvement':metric in ('race_log_loss','horse_log_loss','brier_score')}
        output[first+'_minus_'+second]={'interpretation':meaning,'metrics':values}
    return output


def eligible(folder,plan):
    # Pure verification of saved predictions, with no DB access.
    if not (folder/'four_receipt.json').exists() or (folder/'four_veto.json').exists():return None
    receipt=c.read(folder/'four_receipt.json')
    if receipt['status']!='saved':return None
    if (c.sha(folder/'four_prediction.json')!=receipt['prediction_sha256']
            or c.sha(folder/'four_predictions.csv')!=receipt['CSV_sha256']):raise safety.MLFailure('artifact_invalid')
    record=c.read(folder/'four_prediction.json')
    if record.get('synthetic_verification') or record.get('result_accessed') is not False:return None
    if record['model_ids']!=plan['model_ids'] or core.instant(receipt['written_at'])>core.instant(receipt['deadline']):
        raise safety.MLFailure('artifact_invalid')
    for path,sha in record['references'].items():
        if c.sha(Path(path))!=sha:raise safety.MLFailure('artifact_invalid')
    return record


def run(parent,session):
    plan=addendum.approved(parent)
    require_closed(plan,c.clock())  # BEFORE opening any result source, record set, or DB.
    claim=parent/'four_final_claim.json'
    if claim.exists():raise safety.MLFailure('artifact_exists')
    c.write(claim,{'started_at':c.clock().isoformat(),'plan_sha256':c.sha(parent/addendum.PLAN),
                  'comparison_attempts_max':1,'automatic_retry':False})
    from . import outcomes
    baseplan=four_forward.base_plan(parent,plan);records=[];excluded=Counter();missing=Counter()
    for folder in sorted((parent/addendum.OUTPUT).glob('*/races/*')):
        if not plan['start_date']<=folder.parents[1].name<=plan['end_date']:continue
        base=outcomes.committed(folder,baseplan)
        if base is None:excluded['forecast_not_eligible']+=1;continue
        record=eligible(folder,plan)
        if record is None:
            record={'race_key':base['race_key'],'model_status':{n:{'status':'saved'} for n in forward.NAMES},
                    'probabilities':dict(base['probabilities'],control_D39=None),'ledgers':dict(base['ledgers'],control_D39=None)}
        for name in forward.FOUR_NAMES:
            if record['model_status'].get(name,{}).get('status')!='saved':missing[name]+=1
        if any(record['model_status'].get(n,{}).get('status')!='saved' for n in forward.FOUR_NAMES):
            excluded['not_all_four_models_saved']+=1
        if any(record['probabilities'].get(n) is not None and set(record['probabilities'][n])!=set(base['registered']['roster']) for n in forward.FOUR_NAMES):
            raise safety.MLFailure('invalid_identity')
        records.append((folder,record,base))
    root=forward.mkdir(parent/'final-four')
    if not records:
        result={'status':'ok','reason':'no_common_four_model_records','races':0,'differences':None,'missing':dict(missing),
                'exclusions':dict(excluded),'not_evidence_of_no_effect':True,'new_fits':0}
        c.write(root/'summary.json',result);return result
    byday=defaultdict(list)
    for row in records:byday[row[0].parents[1].name].append(row)
    frames=[];financial={n:[] for n in forward.FOUR_NAMES};source_status=[];began=time.monotonic()
    for day,items in sorted(byday.items()):
        if time.monotonic()-began>600:raise safety.MLFailure('query_timeout')
        races=[tuple(r['race_key']) for _,r,_ in items]
        if len(races)>50:raise safety.MLFailure('input_limit')
        answers=defaultdict(list);hr=[];decoded={}
        with session() as db:
            budget=local.Budget(120);columns=local.table_info(db,'jvd_se',budget)
            fields={'horse':('umaban',2),'horse_id':('ketto_toroku_bango',10),'rank':('kakutei_chakujun',2),
                    'status':('data_kubun',1),'anomaly':('ijo_kubun_code',1)}
            if columns is None or any(columns.get(p,{}).get('type_is_builtin') is not True or columns[p]['type_name'] not in local.TEXT_TYPES for p,_ in fields.values()):
                raise safety.MLFailure('invalid_schema')
            rows=local._rows(db,budget,'jvd_se',fields,columns,races,1500,join_targets=True)
            for row in rows:answers[tuple(row['race_key'])].append(row)
            try:
                hr,profile,status=settle_saved.fetch_hr(db,races,max_rows=100,max_seconds=30,reviewed_source=outcomes.reviewed_profile(races))
                source_status.append(status)
                if profile:decoded={o.race:o for o in settle_saved.decode_existing_hr(hr,profile)}
            except (safety.MLFailure,core.SimulationError) as error:source_status.append({'status':'pending','reason':error.code})
        c.write(root/('answers_'+day+'.local.json'),{'SE':rows,'HR':hr,'loaded_at':c.clock().isoformat()})
        for folder,record,base in items:
            race=tuple(record['race_key']);labels,reason=outcomes.result_labels(base,answers[race])
            if reason:excluded[reason]+=1;continue
            roster=base['registered']['roster']
            f=pd.DataFrame({'entry_key':[''.join(race)+h for h in roster],'date':[day]*len(roster),
                            'winner_label':[labels[h] for h in roster]})
            for name in forward.FOUR_NAMES:
                values=record['probabilities'].get(name)
                f[name]=[values[h] for h in roster] if values else np.nan
            frames.append(f)
            result=decoded.get(race,core.Outcome(race,'pending',reason='result_unavailable'))
            if result.status=='normal_final' and set(result.payouts_yen_per_100)!={h for h in roster if labels[h]}:
                result=core.Outcome(race,'pending',reason='hr_se_winner_mismatch')
            for name in forward.FOUR_NAMES:
                if record['ledgers'].get(name) is not None:
                    financial[name].append({'ai':core.settle(record['ledgers'][name]['ai_horse'],result,tuple(roster)),
                                            'month':day[:7],'race':''.join(race)})
    if not frames:
        result={'status':'ok','reason':'no_confirmed_common_results','races':0,'differences':None,'exclusions':dict(excluded),'new_fits':0}
        c.write(root/'summary.json',result);return result
    all_combined=pd.concat(frames,ignore_index=True)
    core_three={name:learning.point(all_combined,all_combined[name].to_numpy(float)) for name in forward.NAMES}
    combined=all_combined.loc[all_combined.control_D39.notna()].copy()
    if combined.empty:
        result={'status':'ok','reason':'no_common_control_predictions','primary_differences':None,
                'core_three_descriptive':core_three,'missing_model_races':dict(missing),'exclusions':dict(excluded),
                'core_three_races':int(all_combined.entry_key.str[:16].nunique()),'new_fits':0,'not_evidence_of_no_effect':True}
        c.write(root/'summary.json',result);return result
    y=combined.winner_label.to_numpy(int);_,race_codes=np.unique(combined.entry_key.str[:16],return_inverse=True)
    weeks=np.array([(d-timedelta(days=d.weekday())).isoformat() for d in map(date.fromisoformat,combined.date)])
    k=len(set(weeks));configuration=plan['evaluation']
    sample=np.random.default_rng(configuration['seed']).integers(k,size=(configuration['bootstrap_repetitions'],k))
    bootstrap=metrics.WeekBootstrap(weeks,sample_indices=sample);points={};draws={};calibrations={}
    for name in forward.FOUR_NAMES:
        points[name],draws[name]=bootstrap.measure(y,combined[name].to_numpy(float),race_codes)
        calibrations[name]=learning.point(combined,combined[name].to_numpy(float))['calibration_bands']
    differences=contrasts(points,draws,configuration['primary_pair_confidence'])
    common_races=set(combined.entry_key.str[:16]);common_financial={n:[v for v in financial[n] if v['race'] in common_races] for n in forward.FOUR_NAMES}
    summary={'status':'ok','races':int(combined.entry_key.str[:16].nunique()),'runners':len(combined),'points':points,
        'differences':differences,'calibration':calibrations,'exclusions':dict(excluded),'missing_model_races':dict(missing),
        'core_three_descriptive_different_cohort':core_three,'core_three_races':int(all_combined.entry_key.str[:16].nunique()),
        'financial_common_four_races':{n:core.financial(common_financial[n],'ai') for n in forward.FOUR_NAMES},
        'financial_available_predictions_different_cohorts':{n:core.financial(financial[n],'ai') for n in forward.FOUR_NAMES},
        'monthly_financial_common_four_races':{month:{n:core.financial([v for v in common_financial[n] if v['month']==month],'ai') for n in forward.FOUR_NAMES}
                            for month in sorted({d[:7] for d in combined.date})},
        'bootstrap':dict(bootstrap.metadata(),seed=configuration['seed']),'source_status':source_status,
        'window':{'start':plan['start_date'],'end':plan['end_date'],'week_slots':21},
        'evaluated_once_after_window':True,'Stage7_interim_results_opened':False,'automatic_adoption':False,'new_fits':0}
    csv_file(root/'paired_predictions_and_results.local.csv',combined.to_dict('records'))
    c.write(root/'summary.json',summary)
    lines=['# 21週終了後・4モデル一括比較','',f"共通 {summary['races']}R・{summary['runners']}頭。差は先のモデル−後のモデル、レースLLは負が改善。",'',
           '| 比較 | レースLL差 | 同時区間（Bonferroni） |','|---|---:|---|']
    for pair,value in differences.items():
        row=value['metrics']['race_log_loss'];lines.append(f"| {pair} | {row['difference']:.6f} | {row['interval']} |")
    lines+=['','3比較は同一レース集合・同じ週再標本化で評価。欠測は欠測として報告し、結果を見てモデルを選び直していません。',
            '新−対照はパラメータ・木数・温度を含む設定一式、対照−旧は追加データとデータ由来のカテゴリ語彙、 新−旧は総合差です。',
            '途中成績の開封・再学習・自動採用は行っていません。']
    with (root/'report.md').open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(lines)+'\n')
    return {'status':'ok','races':summary['races'],'differences':differences,'report_folder':str(root),'new_fits':0}
