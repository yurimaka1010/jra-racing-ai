"""Independent improvements reuse the existing deadline/receipt/settlement pipeline."""
from collections import Counter
from datetime import timedelta
from pathlib import Path
from uuid import uuid4
import csv
import json
import os
import time
from ..ml import safety
from ..win_sim import core
from ..d39_next import forward as previous
from ..d39_generation import data as history_data
from . import contracts as c,sources,learning
from .features import Builder

NAME='enriched_D39'
MODELS=(*previous.MODELS,NAME)


def atomic(path,value):
    temp=path.with_name(uuid4().hex+'.tmp');c.write(temp,value);os.replace(temp,path)


def history(directory,day):
    index,resolver,_=sources.load(directory);builder=Builder(index,resolver)
    cutoff=day-timedelta(days=7)
    if str(cutoff)>c.read(c.RAW_PARENT/'plan.json')['latest_final_date']:
        raise safety.MLFailure('insufficient_history')
    for rows in history_data.iter_days(c.RAW_PARENT):
        if previous.features.legacy.legacy.entry_identity(rows[0])[1]<=cutoff:builder.transform_day(rows)
    builder._available(day)
    return builder


def ready(directory):
    c.approved(directory);freeze=c.read(directory/'candidate_freeze.json');path=Path(freeze['model_path'])
    if c.sha(path/'model.json')!=freeze['model_sha256']:raise safety.MLFailure('artifact_invalid')
    bundle=learning.load(path);day=c.clock().date()
    if str(day)!='2026-10-25' or str(day)<=bundle[1]['latest_training_or_calibration_date']:
        raise safety.MLFailure('invalid_split')
    return freeze,bundle,day


def preview(directory,*,once=False):
    _,bundle,day=ready(directory)
    return previous.dry_run(directory,bundle,history(directory,day),sources.session,store=c,predictor=learning.predict)


def saved(directory,day):
    values=previous.saved(directory,day,model_name=NAME)
    old={r['race_id']:r for r in previous.saved(c.PARENT,day)}
    for r in values:
        base=old.get(r['race_id'])
        if base and base['source_prediction_sha256']==r['source_prediction_sha256']:
            r['models']['next_D39']=base['models']['next_D39']
            r['d39_tickets']['next_D39']=base['d39_tickets']['next_D39']
    return values


def evaluate(directory):
    import pandas as pd
    from ..keiba_lab_live import fetch_results
    day=str(c.clock().date());forecasts=[r for r in saved(directory,day) if core.instant(r['scheduled_start'])<=c.clock()]
    # Project H out before requesting a single label or payout.
    requests=[{**{k:v for k,v in r.items() if k not in ('models','d39_tickets')},
        'models':{n:r['models'][n] for n in MODELS if n in r['models']},
        'd39_tickets':{n:r['d39_tickets'][n] for n in MODELS if n in r['d39_tickets']}} for r in forecasts]
    results,status=fetch_results(requests,sources.session,names=MODELS)
    rows=list(results.values());common=[r for r in rows if r['result_status']=='confirmed'
        and all(r['models'].get(n,{}).get('status')=='saved' for n in MODELS)]
    metrics={};money={}
    for name in MODELS:
        if common:
            frame=pd.DataFrame([dict(entry_key=r['race_id']+v['horse_no'],date=day,
                winner_label=int(v['finish_position']==1),probability=v['probability']) for r in common for v in r['models'][name]['rows']])
            m=learning.point(frame,frame.probability.to_numpy(float));finishes=[r['models'][name]['top_finish'] for r in common]
            m.update(top2_percent=100*sum(v==2 for v in finishes)/len(finishes),
                top3_percent=100*sum(v==3 for v in finishes)/len(finishes),
                winner_prediction_mean_rank=sum(next(v['rank'] for v in r['models'][name]['rows'] if v['finish_position']==1) for r in common)/len(common))
            metrics[name]=m
        tickets=[{'ai':r['models'][name]['paper_ticket']} for r in rows
            if all(r['models'].get(n,{}).get('status')=='saved' for n in MODELS) and r['models'][name].get('paper_ticket')]
        money[name]=core.financial(tickets,'ai')
    result={'version':c.VERSION,'date':day,'checked_at':c.clock().isoformat(),'races':rows,
        'metrics':metrics,'financial':money,'common_result_races':len(common),
        'pending_result_races':len(rows)-len(common),'H_results_included':False,'Stage7_results_read':False,
        'source_status':status,'kind':'prospective_enriched_D39_exploratory'}
    target=directory/'results';target.mkdir(exist_ok=True);c.write(target/(uuid4().hex+'.json'),result)
    atomic(directory/'latest_results.json',result)
    atomic(directory/'results_status.json',{'status':'ok','checked_at':c.clock().isoformat()})


def public_packet(viewer,day):
    from ..keiba_lab_live import day_guard,public_forecast
    day_guard(day)
    if not (c.ROOT/'current.json').exists():return None
    active=c.read(c.ROOT/'current.json');directory=c.folder(active['run_id'])
    values=saved(directory,day);cards_body=viewer.forecasts(day);cards={r['race_id']:r for r in cards_body['races']}
    cache=getattr(viewer,'name_cache',{}).get(day,{})
    for r in values:r['horse_names']=previous.display_names(r,cards.get(r['race_id']),cache.get(r['race_id'],{}))
    result=c.read(directory/'latest_results.json') if (directory/'latest_results.json').exists() else None
    for r in result.get('races',[]) if result and result['date']==day else ():
        r['horse_names']=previous.display_names(next((v for v in values if v['race_id']==r['race_id']),{}),cards.get(r['race_id']),cache.get(r['race_id'],{}))
    ref={'real_forward_record':False,'races':[],'runners':0,'generated_at':None};candidates=[]
    for path in (directory/'dry').glob('*/summary.json'):
        summary=c.read(path)
        if summary.get('date')==day and summary.get('races',0) and summary.get('CSV_sha256'):
            candidates.append((summary['generated_at'],path,summary))
    if candidates:
        _,path,summary=max(candidates,key=lambda v:v[0]);csv_path=path.parent/'predictions.dry.csv'
        if c.sha(csv_path)!=summary['CSV_sha256'] or core.instant(summary['generated_at'])>c.clock():raise safety.MLFailure('artifact_invalid')
        grouped={}
        with csv_path.open(encoding='utf-8-sig',newline='') as stream:
            for i,row in enumerate(csv.DictReader(stream)):
                if i>=1008 or row['model_id']!=active['model_id'] or row['real_forward_record']!='False':raise safety.MLFailure('artifact_invalid')
                grouped.setdefault(row['entry_key'][:16],[]).append(row)
        for key,part in sorted(grouped.items()):
            card=cards.get(key)
            if not card:continue
            p={r['entry_key'][-2:]:float(r['probability']) for r in part}
            ranks={h:i for i,h in enumerate(sorted(p,key=lambda h:(-p[h],h)),1)}
            if (set(p)!=set(card['roster']) or len(p)!=len(part) or abs(sum(p.values())-1)>1e-12
                    or any(not 0<=v<=1 for v in p.values()) or any(int(r['rank'])!=ranks[r['entry_key'][-2:]] for r in part)):
                raise safety.MLFailure('artifact_invalid')
            ref['races'].append({**{k:card[k] for k in ('race_id','venue','race_no','scheduled_start','roster')},
                'horse_names':previous.display_names({},card,{}),'models':{NAME:{'model_id':active['model_id'],
                    'predicted_at':part[0]['generated_at'],'rows':[{'horse_no':h,'probability':p[h],'rank':ranks[h]} for h in sorted(p)]}},
                'reason':'startup_reference_not_formal_T10'})
        ref.update(runners=sum(len(r['roster']) for r in ref['races']),generated_at=summary['generated_at'],CSV_sha256=summary['CSV_sha256'])
    return {'version':c.VERSION,'candidate_name':NAME,'status':'active','date':day,'activation':active,
        'races':[public_forecast(r) for r in values],'results':result,'reference':ref,
        'existing_coverage':cards_body['coverage'],'H_results_included':False,'checked_at':c.clock().isoformat(),
        'result_poll':c.read(directory/'results_status.json') if (directory/'results_status.json').exists() else {'status':'pending'},
        'feature_note':'前走タイムを接続。馬場状態は学習用の発走前履歴が不足し、今回は欠損を維持。'}


def run(directory,*,once=False):
    freeze,bundle,day=ready(directory);lock=directory/'recorder.lock'
    with lock.open('x',encoding='utf-8') as stream:
        stream.write(str(os.getpid()));stream.flush()
        try:
            builder=history(directory,day)
            dry=previous.dry_run(directory,bundle,builder,sources.session,store=c,predictor=learning.predict)
            if not dry['races']:raise safety.MLFailure('insufficient_history')
            activation={'run_id':directory.name,'model_id':freeze['model_id'],'activated_at':c.clock().isoformat(),
                'date':str(day),'model_sha256':freeze['model_sha256'],'temperature':bundle[1]['temperature'],'feature_count':47}
            if (directory/'activation.json').exists():
                old=c.read(directory/'activation.json')
                if old['model_id']!=activation['model_id'] or old['date']!=activation['date']:raise safety.MLFailure('artifact_invalid')
                activation=old
            else:c.write(directory/'activation.json',activation)
            atomic(c.ROOT/'current.json',activation);last_result=0;last_counts=None
            while c.clock().date()==day and c.clock().hour<18:
                counts=Counter();root=c.RAW_PARENT/'forward-four-sealed'/str(day)/'races'
                for folder in sorted(root.iterdir()) if root.exists() else ():
                    output=directory/'forward'/str(day)/'races'/folder.name
                    try:
                        value=previous.decide(folder,output,activation,bundle,builder,store=c,
                            predictor=learning.predict,model_name=NAME)
                        counts[value['status']]+=1
                    except Exception as exc:
                        counts['failed']+=1
                        if not (output/'prediction_error.json').exists():
                            c.write(output/'prediction_error.json',{'reason':exc.code if isinstance(exc,safety.MLFailure) else 'internal_error','at':c.clock().isoformat()})
                if time.monotonic()-last_result>30:
                    try:evaluate(directory)
                    except Exception:atomic(directory/'results_status.json',{'status':'pending','reason':'result_source_unavailable','attempted_at':c.clock().isoformat()})
                    last_result=time.monotonic()
                if counts!=last_counts:
                    print(json.dumps({'status':'running','counts':dict(counts),'at':c.clock().isoformat()}),flush=True);last_counts=counts
                if once:return {'status':'ok','counts':dict(counts)}
                time.sleep(2)
            return {'status':'ok','reason':'daily_window_finished'}
        finally:
            stream.close();lock.unlink()
