"""Isolated four-model sidecars around the unmodified three-model recorder."""
from collections import Counter
from copy import deepcopy
from datetime import date,datetime,timedelta
from pathlib import Path
from uuid import uuid4
import json
import os
import time
import numpy as np

from ..core import SafeAuditError
from ..ml import safety
from ..research import model as matrices
from ..tuner_pilot import experiment as tuner
from ..stage7 import inputs
from ..win_sim import core,publication,local
from . import contracts as c,forward as legacy,learning,addendum,four_sources
from .report import csv_file


def load_bundles(plan):
    bundles={};errors={}
    for name,path in plan['model_paths'].items():
        try:
            path=Path(path)
            for file in (path/'model.json',path/('win.lgb.txt' if name=='H' else 'booster.lgb.txt')):
                if c.sha(file)!=plan['references'][str(file)]:raise safety.MLFailure('artifact_invalid')
            bundles[name]=(matrices.load(path,root=c.S2) if name=='H' else tuner.load_model(path) if name=='old_D39' else learning.load(path))
        except Exception:
            if name!='control_D39':raise safety.MLFailure('invalid_model') from None
            bundles[name]=None;errors[name]='control_load_failed'
    return bundles,errors


def preflight(parent):
    plan=addendum.approved(parent);bundles,errors=load_bundles(plan)
    observed=c.clock();root=legacy.mkdir(parent/'availability-four')
    value={'observed_at':observed.isoformat(),'control_available':bundles['control_D39'] is not None,'errors':errors,
           'availability_deadline':plan['availability_deadline'],'before_deadline':observed<=core.instant(plan['availability_deadline']),
           'model_ids':plan['model_ids'],'new_fits':0}
    c.write(root/(uuid4().hex+'.json'),value)
    return {'status':'ready','effective_models':list(legacy.FOUR_NAMES if value['control_available'] else legacy.NAMES),
            'control_available':value['control_available'],'automatic_fallback':not value['control_available'],
            'start_date':plan['start_date'],'end_date':plan['end_date'],'deadline':plan['availability_deadline'],
            'model_ids':plan['model_ids'],'record_root':str(parent/addendum.OUTPUT),'new_fits':0}


def startup_mode(parent,plan,control_ok,*,clock=None):
    now=(clock or c.clock)();deadline=core.instant(plan['availability_deadline'])
    observations=[c.read(p) for p in (parent/'availability-four').glob('*.json')]
    on_time=[v for v in observations if core.instant(v['observed_at'])<=deadline]
    latest=max(on_time,key=lambda v:v['observed_at']) if on_time else None
    authorized=bool(latest and latest['control_available'])
    mode='four' if authorized and control_ok else 'three_fallback'
    return {'mode':mode,'control_enabled':mode=='four','decided_at':now.isoformat(),'deadline':deadline.isoformat(),
            'reason':None if mode=='four' else 'control_not_available_by_deadline_or_at_startup',
            'on_time_assessment_present':latest is not None,'future_result_or_performance_used':False}


def base_plan(parent,plan):
    p=c.read(parent/'forward_plan.json')
    return dict(p,start_date=plan['start_date'],end_date=plan['end_date'])


def raw_predict(name,bundle,frame,record):
    if name=='H':
        hframe=inputs.features(tuple(record['race_key']),record['quote_T10'],record['quote_T30'])
        raw=matrices.uncalibrated(bundle,hframe);keys=hframe.entry_key.str[-2:]
    else:
        booster,meta=bundle;raw=booster.predict(matrices.matrix(frame,meta['feature_spec'],meta['categories']),num_threads=4);keys=frame.entry_key.str[-2:]
    if not np.isfinite(raw).all() or not ((np.asarray(raw)>=0)&(np.asarray(raw)<=1)).all():raise safety.MLFailure('invalid_model')
    return dict(zip(keys,map(float,raw)))


def sidecar(folder,parent,plan,bundles,history,*,clock=None,control_reason=None):
    clock=clock or c.clock
    if (folder/'four_receipt.json').exists():return c.read(folder/'four_receipt.json')
    receipt=c.read(folder/'receipt.json')
    if receipt['status']!='saved':
        value={'status':'missing','reason':receipt.get('reason'),'base_three_saved':False};c.write(folder/'four_receipt.json',value);return value
    if c.sha(folder/'prediction.json')!=receipt['prediction_sha256']:raise safety.MLFailure('artifact_invalid')
    record=c.read(folder/'prediction.json');entry=c.read(Path(record['entry_snapshot']['path']))
    if c.sha(Path(record['entry_snapshot']['path']))!=record['entry_snapshot']['sha256']:raise safety.MLFailure('artifact_invalid')
    frame=legacy.live_features(history,entry['payload']);roster=record['registered']['roster'];raws={};statuses={};probabilities=deepcopy(record['probabilities'])
    ledgers=deepcopy(record['ledgers']);deadline=core.instant(receipt['deadline'])
    for name in legacy.NAMES:
        try:
            raw=raw_predict(name,bundles[name],frame,record)
            recovered=matrices.normalize(frame,np.array([raw[h] for h in frame.entry_key.str[-2:]]),plan['temperatures'][name])
            if max(abs(float(p)-probabilities[name][h]) for p,h in zip(recovered,frame.entry_key.str[-2:]))>1e-12:
                raise safety.MLFailure('invalid_model')
            raws[name]=raw;statuses[name]={'status':'saved','raw_status':'saved'}
        except Exception:
            # The base prediction is already durable; a diagnostic cannot veto it.
            raws[name]=None;statuses[name]={'status':'saved','raw_status':'missing','raw_reason':'raw_probability_unavailable'}
    try:
        if control_reason or bundles.get('control_D39') is None:raise safety.MLFailure('invalid_model')
        raw=raw_predict('control_D39',bundles['control_D39'],frame,record)
        p=matrices.normalize(frame,np.array([raw[h] for h in frame.entry_key.str[-2:]]),plan['temperatures']['control_D39'])
        probabilities['control_D39']=dict(zip(frame.entry_key.str[-2:],map(float,p)));raws['control_D39']=raw
        ledgers['control_D39']=legacy.paper_decision(tuple(record['race_key']),probabilities['control_D39'],record['quote_T10'],record['registered'],base_plan(parent,plan))
        statuses['control_D39']={'status':'saved','raw_status':'saved'}
    except Exception:
        probabilities['control_D39']=None;raws['control_D39']=None;ledgers['control_D39']=None
        statuses['control_D39']={'status':'missing','reason':control_reason or 'control_prediction_failed'}
    generated=clock();rows=[]
    for name in legacy.FOUR_NAMES:
        values=probabilities[name];rank={h:i for i,h in enumerate(sorted(values,key=lambda h:(-values[h],h)),1)} if values else {}
        raw=raws[name];ledger=ledgers[name]
        for horse in roster:
            rows.append({'entry_key':''.join(record['race_key'])+horse,'model':name,'model_id':plan['model_ids'][name],
                'probability':values.get(horse) if values else None,'rank':rank.get(horse),
                'raw_probability':raw.get(horse) if raw else None,'raw_probability_sum':sum(raw.values()) if raw else None,
                'temperature':plan['temperatures'][name],'model_status':statuses[name]['status'],
                'missing_reason':statuses[name].get('reason'),'raw_status':statuses[name].get('raw_status','missing'),
                'input_cutoff':record['input_cutoff'],'generated_at':generated.isoformat(),
                'base_three_generated_at':record['generated_at'],'odds_published_at':record['quote_T10']['published_at'],
                'EV':values[horse]*record['quote_T10']['odds'][horse]-1 if values else None,
                'virtual_stake_yen':(100 if ledger['ai_horse']==horse else 0) if ledger else None})
    value={'race_key':record['race_key'],'model_ids':plan['model_ids'],'model_status':statuses,'probabilities':probabilities,
        'raw_probabilities':raws,'ledgers':ledgers,'input_cutoff':record['input_cutoff'],'generated_at':generated.isoformat(),
        'base_three_prediction_sha256':receipt['prediction_sha256'],'base_three_CSV_sha256':receipt['CSV_sha256'],
        'references':{p:h for p,h in plan['references'].items() if 'control_D39' not in statuses or statuses['control_D39']['status']=='saved'
                      or not p.startswith(plan['model_paths']['control_D39'])},
        'result_accessed':False,'synthetic_verification':plan.get('synthetic_verification',False)}
    if generated>deadline:
        missing={'status':'missing','reason':'four_sidecar_deadline_missed','base_three_saved':True};c.write(folder/'four_receipt.json',missing);return missing
    csv_file(folder/'four_predictions.csv',rows);c.write(folder/'four_prediction.json',value)
    written=clock();result={'status':'saved' if written<=deadline else 'missing','reason':None if written<=deadline else 'four_sidecar_deadline_missed',
        'written_at':written.isoformat(),'deadline':receipt['deadline'],'base_three_saved':True,
        'control_status':statuses['control_D39']['status'],'prediction_sha256':c.sha(folder/'four_prediction.json'),
        'CSV_sha256':c.sha(folder/'four_predictions.csv')}
    c.write(folder/'four_receipt.json',result)
    if clock()>deadline:c.write(folder/'four_veto.json',{'reason':'receipt_late'})
    return result


def tick(parent,session,plan,bundles,history,control_reason=None):
    day=c.clock().date();root=legacy.mkdir(parent/addendum.OUTPUT/str(day));registered=[]
    with session() as db:
        cards=legacy.existing.live_cards(db,day);entry_groups,seen=four_sources.entries(db,day)
        active=[]
        for race,card in sorted(cards.items()):
            folder=legacy.mkdir(root/'races'/''.join(race))
            if not (folder/'registered.json').exists():
                if card['scheduled_start'] and core.instant(card['scheduled_start'])-timedelta(minutes=10)<=seen:
                    card=dict(card,reason='registered_after_input_cutoff')
                c.write(folder/'registered.json',dict(card,registered_at=seen.isoformat()))
            reg=c.read(folder/'registered.json');registered.append((folder,reg))
            if not legacy.existing.same_saved_clock_and_roster(reg,card) and not (folder/'veto.json').exists():
                c.write(folder/'veto.json',{'reason':'schedule_or_roster_changed','observed_at':seen.isoformat()})
            if (folder/'four_receipt.json').exists():continue
            if reg['scheduled_start']:
                start=core.instant(reg['scheduled_start'])
                if start-timedelta(minutes=35)<=seen<=start-timedelta(minutes=9):
                    legacy.capture(folder,'entries',entry_groups.get(race,[]),seen);active.append((folder,reg))
        if active and publication._columns(db,'jvd_o1',publication.O1_FIELDS,local.Budget(60)) is None:raise safety.MLFailure('invalid_schema')
        for i in range(0,len(active),32):
            part=active[i:i+32];races=[tuple(r['race_key']) for _,r in part]
            sql,args=four_sources.current_odds_query(races,{tuple(r['race_key']):r['scheduled_start'] for _,r in part})
            quotes=four_sources.current_odds_grouped(db.aggregate('four.observed_O1',sql,args,max_rows=128));observed=c.clock()
            for folder,reg in part:
                for minutes in (30,10):
                    for q in quotes.get((tuple(reg['race_key']),minutes),[]):legacy.capture(folder,'odds',q,observed)
    counts=Counter()
    for folder,reg in registered:
        try:
            base=legacy.decide(folder,reg,cards.get(tuple(reg['race_key'])),base_plan(parent,plan),bundles,history)
            if base['status']=='pending':counts['pending']+=1;continue
            result=sidecar(folder,parent,plan,bundles,history,control_reason=control_reason)
            counts[result['status']]+=1
            if result.get('control_status')=='missing':counts['control_missing']+=1
        except Exception:
            logs=legacy.mkdir(folder/'logs');c.write(logs/(uuid4().hex+'.json'),{'reason':'prediction_or_sidecar_failed','at':c.clock().isoformat()})
            counts['failed']+=1
    return {'registered_races':len(cards),'counts':dict(counts),'at':c.clock().isoformat(),'results_read':False}


def run(parent,session,*,once=False):
    ready=preflight(parent);plan=addendum.approved(parent);today=c.clock().date()
    if str(today)<plan['start_date']:
        return dict(ready,reason='before_first_race_day;no_live_predictions_generated')
    if not plan['start_date']<=str(today)<=plan['end_date']:raise safety.MLFailure('invalid_split')
    if c.clock()<core.instant(plan['availability_deadline']):
        return dict(ready,reason='waiting_for_fixed_startup_deadline;restart_after_0700')
    lock=parent/'four-addendum-recorder.lock'
    try:stream=lock.open('x',encoding='utf-8')
    except FileExistsError:raise safety.MLFailure('input_limit') from None
    token=uuid4().hex;json.dump({'token':token,'pid':os.getpid()},stream);stream.flush()
    try:
        bundles,errors=load_bundles(plan);mode=startup_mode(parent,plan,bundles['control_D39'] is not None)
        if not mode['control_enabled']:bundles['control_D39']=None
        c.write(legacy.mkdir(parent/'startup-four')/(uuid4().hex+'.json'),mode)
        history,history_info=four_sources.history(parent,session,today)
        from .four_dry import run as dry_run
        dry=dry_run(parent,session,str(today),bundles=bundles,history=history)
        c.write(legacy.mkdir(parent/addendum.OUTPUT/str(today)/'logs')/(uuid4().hex+'.json'),
                {'event':'startup','at':c.clock().isoformat(),'mode':mode,'history':history_info,'dry_run':dry})
        began=time.monotonic();last=None
        while c.clock().date()==today and c.clock().hour<18 and time.monotonic()-began<10*3600:
            value=tick(parent,session,plan,bundles,history,errors.get('control_D39') or (mode['reason'] if not mode['control_enabled'] else None))
            if value['counts']!=last:print(json.dumps(dict(status='running',**value)),flush=True);last=value['counts']
            if once:return dict(status='ok',**value)
            if legacy.memory_bytes_safe()>12*1024**3:raise safety.MLFailure('input_limit')
            time.sleep(15)
        return {'status':'ok','reason':'daily_window_finished','counts':last or {},'mode':mode['mode']}
    except Exception as error:
        # Startup/history failures need the same fixed-code log as live failures.
        # Never serialize an exception message, SQL, credentials or input rows.
        reason=error.code if isinstance(error,SafeAuditError) else 'internal_error'
        c.write(legacy.mkdir(parent/addendum.OUTPUT/str(today)/'logs')/(uuid4().hex+'.json'),
                {'status':'stopped','reason':reason,'at':c.clock().isoformat(),'results_read':False})
        if isinstance(error,SafeAuditError):raise
        raise safety.MLFailure('internal_error') from None
    finally:
        stream.close()
        if json.loads(lock.read_text(encoding='utf-8')).get('token')==token:lock.unlink()
