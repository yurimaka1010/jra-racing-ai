"""Independent T10 three-model recorder: declared entries and O1 only, never labels."""
from collections import Counter,deque
from copy import copy
from datetime import date,timedelta
import hashlib
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from ..core import RACE_KEY,SafeAuditError
from ..ml import safety
from ..research import features as rf,model as matrices
from ..stage7 import forward as existing,inputs
from ..tuner_pilot import experiment as tuner
from ..win_sim import core,publication,local
from . import contracts as c,data,learning

NAMES=('old_D39','new_D39','H')
FOUR_NAMES=('old_D39','new_D39','control_D39','H')


def model_names(plan):
    names=tuple(plan.get('models',NAMES))
    if names not in (NAMES,FOUR_NAMES) or set(plan.get('model_ids',{}))!=set(names) or set(plan.get('temperature',{}))!=set(names):
        raise safety.MLFailure('invalid_model')
    return names


def record_subdir(plan):
    return 'forward-four' if model_names(plan)==FOUR_NAMES else 'forward'


def mkdir(path):
    path=safety.local_path(path,root=c.ROOT);path.mkdir(parents=True,exist_ok=True);return path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def preflight(directory,*,four=False):
    c.approved(directory);freeze=c.read(directory/'candidate_freeze.json');new_path=Path(freeze['model_path'])
    if c.sha(new_path/'model.json')!=freeze['model_sha256'] or c.sha(new_path/'booster.lgb.txt')!=freeze['tree_sha256']:
        raise safety.MLFailure('artifact_invalid')
    old=tuner.load_model(c.OLD_MODEL);new=learning.load(new_path);h=matrices.load(c.h.MODEL,root=c.S2)
    if (old[1]['features']!=new[1]['features'] or tuple(h['metadata']['features'])!=c.h.H_FIELDS
            or h['metadata']['calibration']['temperature']!=1.):raise safety.MLFailure('invalid_model')
    if not (directory/'forward_plan.json').exists():
        today=c.clock().date();start=today+timedelta(days=(5-today.weekday())%7)
        if start<=date.fromisoformat(new[1]['latest_training_or_calibration_date']):raise safety.MLFailure('invalid_split')
        refs={str(p):c.sha(p) for root in (c.OLD_MODEL,new_path,c.h.MODEL)
              for p in root.iterdir() if p.name in ('model.json','booster.lgb.txt','win.lgb.txt')}
        refs[str(directory/'data_manifest.json')]=c.sha(directory/'data_manifest.json')
        p={'version':c.VERSION,'created_at':c.clock().isoformat(),'start_date':str(start),'end_date':str(start+timedelta(days=1)),
            'scope':'first_weekend_only;refresh_history_before_later_weekends','model_ids':{'old_D39':old[1]['model_id'],
            'new_D39':new[1]['model_id'],'H':c.h.H_ID},'new_model_path':str(new_path),'references':refs,
            'input_cutoff_minutes':10,'quote_T30_minutes':30,'minute_precision':'publication_minute_end_not_after_cutoff',
            'maximum_age_seconds':300,'save_deadline_seconds_after_cutoff':60,'own_first_seen_at_required':True,
            'history_lag_days':7,'temperature':{'old_D39':old[1]['temperature'],'new_D39':new[1]['temperature'],'H':1.},
            'models_already_calibrated_before_betting':True,'policy':{'stake_yen':100,'threshold':'.05','max_points_per_race':1},
            'new_task_registrations':0,'Stage7_formal_record_and_results_access':False,'actual_start_clock_unverified':True}
        c.write(directory/'forward_plan.json',p)
    p=c.read(directory/'forward_plan.json')
    if four:
        frozen=c.read(directory/'control_candidate_freeze.json');control_path=Path(frozen['model_path'])
        _,control_meta=learning.load(control_path)
        if (c.sha(control_path/'model.json')!=frozen['model_sha256'] or c.sha(control_path/'booster.lgb.txt')!=frozen['tree_sha256']
                or control_meta['parameters']!=old[1]['parameters'] or control_meta['best_iteration']!=old[1]['best_iteration']
                or control_meta['temperature']!=old[1]['temperature'] or control_meta['dataset_id']!=new[1]['dataset_id']
                or control_meta['categories']!=new[1]['categories']):raise safety.MLFailure('invalid_model')
        if not (directory/'forward_four_plan.json').exists():
            refs=dict(p['references'])
            refs.update({str(path):c.sha(path) for path in (control_path/'model.json',control_path/'booster.lgb.txt',
                         directory/'control_candidate_freeze.json',directory/'forward_plan.json')})
            four_ids={name:(control_meta['model_id'] if name=='control_D39' else p['model_ids'][name]) for name in FOUR_NAMES}
            temperatures={name:(control_meta['temperature'] if name=='control_D39' else p['temperature'][name]) for name in FOUR_NAMES}
            p=dict(p,models=list(FOUR_NAMES),model_ids=four_ids,temperature=temperatures,references=refs,
                control_model_path=str(control_path),comparison_profile='four_models_fixed_old_parameters_control_v1',
                profile_created_at=c.clock().isoformat(),parent_three_plan_sha256=c.sha(directory/'forward_plan.json'))
            c.write(directory/'forward_four_plan.json',p)
        else:p=c.read(directory/'forward_four_plan.json')
        if model_names(p)!=FOUR_NAMES or p['model_ids']['control_D39']!=control_meta['model_id']:
            raise safety.MLFailure('invalid_model')
    for path,sha in p['references'].items():
        if c.sha(Path(path))!=sha:raise safety.MLFailure('artifact_invalid')
    launcher=directory/('start_four_model_comparison.cmd' if four else 'start_three_model_comparison.cmd')
    if not launcher.exists():
        import base64
        script=str(c.PROJECT/'scripts/d39_current.ps1').replace("'","''")
        mode='record-four' if four else 'record'
        command=f"& '{script}' -Mode {mode} -RunId '{directory.name}'"
        encoded=base64.b64encode(command.encode('utf-16-le')).decode('ascii')
        with launcher.open('x',encoding='ascii',newline='') as stream:
            stream.write('@echo off\r\nC:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -NoProfile -ExecutionPolicy Bypass -EncodedCommand '+encoded+'\r\npause\r\n')
    return {'status':'ready','model_ids':p['model_ids'],'start_date':p['start_date'],'end_date':p['end_date'],
        'same_T10_cutoff':True,'task_changes':0,'record_folder':str(directory/record_subdir(p)),'launcher':str(launcher)}


def warm_history(directory,day):
    builder=rf.Builder()
    for rows in data.iter_days(directory):
        if rf.legacy.entry_identity(rows[0])[1]>=day:raise safety.MLFailure('invalid_split')
        data.replay(builder,rows)
    builder._available(day)
    return builder


def live_features(builder,rows):
    import pandas as pd
    if not rows or any(not row.get('horse_id') for row in rows) or len({row['horse_id'] for row in rows})!=len(rows):
        raise safety.MLFailure('invalid_identity')
    # _available(today) was applied once. A shallow copy protects the shared
    # pending queue; no same-day result can update any history statistic.
    clone=copy(builder);clone.pending=deque(builder.pending)
    normalized=[]
    for raw in rows:
        allowed={k:raw.get(k) for k in (*RACE_KEY,*c.features.RAW_FIELDS)
                 if k in RACE_KEY or (k in c.features.RAW_FIELDS and
                    (c.features.RAW_FIELDS[k][3] in ('key','identity') or c.features.RAW_FIELDS[k][3].startswith('declared')))}
        weight=rf.num(allowed.get('weight_raw'));allowed['weight_kg']=weight/10 if weight is not None else None
        normalized.append(allowed)
    values,meta=clone.transform_day(normalized)
    if any(x[k] is None for x in values for k in ('venue','surface','distance','age','weight_kg')):
        raise safety.MLFailure('insufficient_history')
    return pd.DataFrame([dict(x,entry_key=''.join(m['key']),date=m['date']) for x,m in zip(values,meta)])


def capture(folder,kind,payload,seen):
    # Quotes retain first observation; unchanged entry values get a fresh,
    # independently observed snapshot rather than becoming artificially stale.
    identity={'payload':payload,'observed_at':seen.isoformat()} if kind=='entries' else payload
    path=mkdir(folder/kind)/(digest(identity)+'.json')
    if not path.exists():c.write(path,{'payload':payload,'first_seen_at':seen.isoformat()})


def on_time_entries(folder,roster,cutoff):
    values=[]
    for path in (folder/'entries').glob('*.json'):
        row=c.read(path);seen=core.instant(row['first_seen_at'])
        if seen<=cutoff:values.append((seen,path,row))
    if not values:return None,'no_on_time_entry_snapshot'
    seen,path,row=max(values,key=lambda v:(v[0],v[1].name))
    if (cutoff-seen).total_seconds()>300:return None,'entry_snapshot_stale'
    raw=row['payload']
    if len(raw)!=len(roster) or {r['horse_number'] for r in raw}!=set(roster):return None,'roster_mismatch'
    return dict(row,path=str(path),sha256=c.sha(path)),None


def paper_decision(race,p,quote,registered,plan):
    policy=core.WinPolicy(probability_method='identity',temperature=1.,evaluation_start=plan['start_date'],evaluation_end=plan['end_date'])
    snapshot=core.Snapshot(race,core.instant(quote['published_at']),core.instant(quote['first_seen_at']),
        {h:core.decimal(v) for h,v in quote['odds'].items()},'apd_sokuho_o1',True,publication_precision='minute')
    schedule=core.Schedule(race,core.instant(registered['scheduled_start']),core.instant(registered['registered_at']),tuple(registered['roster']),True)
    # Existing fixed 100yen / EV5% / one-point logic. T is identity here:
    # model predictions have already received their one model-specific temperature.
    result=core.run_simulation([core.PredictionRace(race,p)],[schedule],[snapshot],policy=policy)
    return result['rows'][0]


def decide(folder,registered,current,plan,bundles,history,*,clock=None):
    clock=clock or c.clock;now=clock()
    names=model_names(plan)
    if (folder/'receipt.json').exists():return c.read(folder/'receipt.json')
    race=tuple(registered['race_key']);reason=registered.get('reason')
    start=core.instant(registered['scheduled_start']) if registered['scheduled_start'] else None
    cutoff=start-timedelta(minutes=10) if start else None;deadline=cutoff+timedelta(seconds=60) if cutoff else None
    if not reason and not existing.same_card(registered,current):reason='schedule_or_roster_changed'
    if not reason and now<cutoff:return {'status':'pending'}
    if not reason and now>deadline:reason='save_deadline_missed'
    if (folder/'veto.json').exists():reason='schedule_or_roster_changed'
    current_q=None;early=None;snapshot=None
    if not reason:
        quotes=[dict(v['payload'],first_seen_at=v['first_seen_at']) for p in (folder/'odds').glob('*.json') for v in [c.read(p)]]
        current_q,reason=inputs.select(race,quotes,registered['roster'],cutoff,300,require_seen=True)
        early,early_reason=inputs.select(race,quotes,registered['roster'],start-timedelta(minutes=30),300,require_seen=True)
        snapshot,entry_reason=on_time_entries(folder,registered['roster'],cutoff)
        reason=reason or entry_reason
    if reason:
        value={'status':'missing','reason':reason,'recorded_at':clock().isoformat()};c.write(folder/'receipt.json',value);return value
    frame=live_features(history,snapshot['payload'])
    p={'old_D39':dict(zip(frame.entry_key.str[-2:],map(float,tuner.predict(*bundles['old_D39'],frame)))),
       'new_D39':dict(zip(frame.entry_key.str[-2:],map(float,learning.predict(bundles['new_D39'],frame))))}
    if 'control_D39' in names:
        p['control_D39']=dict(zip(frame.entry_key.str[-2:],map(float,learning.predict(bundles['control_D39'],frame))))
    h,_=inputs.predict(bundles['H'],race,current_q,early);p['H']=h['H']
    if any(set(v)!=set(registered['roster']) or abs(sum(v.values())-1.)>1e-12 for v in p.values()):raise safety.MLFailure('invalid_model')
    ledgers={name:paper_decision(race,values,current_q,registered,plan) for name,values in p.items()}
    rows=[];generated_at=clock().isoformat()
    for name in names:
        ranks={horse:i for i,horse in enumerate(sorted(p[name],key=lambda horse:(-p[name][horse],horse)),1)}
        for row in snapshot['payload']:
            horse=row['horse_number'];rows.append({'entry_key':''.join(race)+horse,'horse_id':row['horse_id'],
                'date':frame.date.iloc[0],'model':name,'model_id':plan['model_ids'][name],'probability':p[name][horse],
                'rank':ranks[horse],'temperature':plan['temperature'][name],'input_cutoff':cutoff.isoformat(),
                'generated_at':generated_at,'odds_published_at':current_q['published_at'],'odds_first_seen_at':current_q['first_seen_at'],
                'odds':current_q['odds'][horse],'EV':p[name][horse]*current_q['odds'][horse]-1,
                'virtual_stake_yen':100 if ledgers[name]['ai_horse']==horse else 0,
                'reason':ledgers[name]['reason'],'distance':float(frame.distance.iloc[0]),'surface':frame.surface.iloc[0]})
    from .report import csv_file
    record={'race_key':list(race),'model_ids':plan['model_ids'],'registered':registered,'input_cutoff':cutoff.isoformat(),
        'probabilities':p,'ledgers':ledgers,'quote_T10':current_q,'quote_T30':early,'T30_missing_reason':early_reason,
        'entry_snapshot':{'path':snapshot['path'],'sha256':snapshot['sha256'],'first_seen_at':snapshot['first_seen_at']},
        'model_references':plan['references'],'result_accessed':False,'generated_at':generated_at,
        'synthetic_verification':plan.get('synthetic_verification',False)}
    if clock()>deadline:
        value={'status':'missing','reason':'calculation_deadline_missed'};c.write(folder/'receipt.json',value);return value
    csv_file(folder/'predictions.csv',rows);c.write(folder/'prediction.json',record)
    written=clock();value={'status':'saved' if written<=deadline else 'missing','reason':None if written<=deadline else 'write_deadline_missed',
        'written_at':written.isoformat(),'deadline':deadline.isoformat(),'prediction_sha256':c.sha(folder/'prediction.json'),
        'CSV_sha256':c.sha(folder/'predictions.csv'),'input_cutoff':cutoff.isoformat()}
    c.write(folder/'receipt.json',value)
    if clock()>deadline:c.write(folder/'veto.json',{'reason':'receipt_write_late'})
    return value


def tick(directory,session,plan,bundles,history):
    day=c.clock().date();day_folder=mkdir(directory/record_subdir(plan)/str(day));counts=Counter()
    with session() as db:
        cards=existing.live_cards(db,day);seen=c.clock()
        schema=data.columns(db);sql,args,_=data.source_query(schema,day,day,c.approved(directory),live=True)
        raw=db.aggregate('d39.live_declared_only',sql,args,max_rows=3000);seen_entries=c.clock();groups={}
        for row in raw:groups.setdefault(tuple(row[k] for k in RACE_KEY),[]).append(row)
        active=[];registered=[]
        for race,card in sorted(cards.items()):
            folder=mkdir(day_folder/'races'/''.join(race))
            if not (folder/'registered.json').exists():
                if card['scheduled_start'] and core.instant(card['scheduled_start'])-timedelta(minutes=10)<=seen:
                    card=dict(card,reason='registered_after_input_cutoff')
                c.write(folder/'registered.json',dict(card,registered_at=seen.isoformat()))
            reg=c.read(folder/'registered.json');registered.append((folder,reg))
            if not existing.same_saved_clock_and_roster(reg,card) and not (folder/'veto.json').exists():
                c.write(folder/'veto.json',{'reason':'schedule_or_roster_changed','observed_at':seen.isoformat()})
            if (folder/'receipt.json').exists():continue
            if reg['scheduled_start']:
                start=core.instant(reg['scheduled_start'])
                if start-timedelta(minutes=35)<=seen<=start-timedelta(minutes=9):
                    capture(folder,'entries',groups.get(race,[]),seen_entries);active.append((folder,reg))
        if active:
            if publication._columns(db,'apd_sokuho_o1',publication.O1_FIELDS,local.Budget(60)) is None:
                raise safety.MLFailure('invalid_schema')
        for offset in range(0,len(active),32):
            chunk=active[offset:offset+32];races=[tuple(reg['race_key']) for _,reg in chunk]
            sql,args=inputs.query(races,{tuple(reg['race_key']):reg['scheduled_start'] for _,reg in chunk},(30,10),historical=False)
            rows=db.aggregate('d39.forward_O1_only',sql,args,max_rows=128);quotes=inputs.grouped(rows);seen_quotes=c.clock()
            for folder,reg in chunk:
                for minutes in (30,10):
                    for row in quotes.get((tuple(reg['race_key']),minutes),[]):capture(folder,'odds',row,seen_quotes)
    for folder,reg in registered:
        value=decide(folder,reg,cards.get(tuple(reg['race_key'])),plan,bundles,history)
        counts[value['status']]+=1
    return {'registered_races':len(cards),'counts':dict(counts),'labels_read':False,'recorded_at':c.clock().isoformat()}


def run(directory,session,*,once=False,four=False):
    ready=preflight(directory,four=True) if four else preflight(directory)
    plan=c.read(directory/('forward_four_plan.json' if four else 'forward_plan.json'));day=c.clock().date()
    if str(day)<plan['start_date']:
        with session() as db:cards=existing.live_cards(db,date.fromisoformat(plan['start_date']))
        readiness={'checked_at':c.clock().isoformat(),'target_date':plan['start_date'],'race_count':len(cards),
            'ready_card_races':sum(v['reason'] is None for v in cards.values()),
            'runner_count':sum(len(v['roster']) for v in cards.values()),
            'reasons':dict(Counter(v['reason'] for v in cards.values() if v['reason'])),
            'forecasts_generated':0,'result_fields_read':False}
        c.write(mkdir(directory/('readiness-four' if four else 'readiness'))/(uuid4().hex+'.json'),readiness)
        return dict(ready,reason='waiting_for_first_race_day;no_future_forecast_generated',card_readiness=readiness)
    if not plan['start_date']<=str(day)<=plan['end_date']:raise safety.MLFailure('invalid_split')
    path=directory/('forward-four-recorder.lock' if four else 'forward-recorder.lock')
    try:stream=path.open('x',encoding='utf-8')
    except FileExistsError:raise safety.MLFailure('input_limit') from None
    token=uuid4().hex
    try:
        json.dump({'pid':os.getpid(),'token':token},stream);stream.flush()
        bundles={'old_D39':tuner.load_model(c.OLD_MODEL),'new_D39':learning.load(Path(plan['new_model_path'])),
                 'H':matrices.load(c.h.MODEL,root=c.S2)}
        if four:bundles['control_D39']=learning.load(Path(plan['control_model_path']))
        history=warm_history(directory,day);began=time.monotonic();last=None
        while c.clock().date()==day and c.clock().hour<18 and time.monotonic()-began<10*3600:
            try:
                result=tick(directory,session,plan,bundles,history)
                if result['counts']!=last:
                    print(json.dumps(dict(status='running',**result)),flush=True);last=result['counts']
                if once:return dict(status='ok',**result)
            except SafeAuditError as error:
                error_dir=mkdir(directory/record_subdir(plan)/str(day)/'logs')
                c.write(error_dir/(uuid4().hex+'.json'),{'status':'stopped','reason':error.code,'at':c.clock().isoformat()})
                raise
            except Exception:
                error_dir=mkdir(directory/record_subdir(plan)/str(day)/'logs')
                c.write(error_dir/(uuid4().hex+'.json'),{'status':'stopped','reason':'internal_error','at':c.clock().isoformat()})
                raise safety.MLFailure('internal_error') from None
            if memory_bytes_safe()>12*1024**3:raise safety.MLFailure('input_limit')
            time.sleep(15)
        return {'status':'ok','reason':'daily_recording_window_finished','date':str(day),'counts':last or {}}
    finally:
        stream.close()
        if json.loads(path.read_text(encoding='utf-8')).get('token')==token:path.unlink()


def memory_bytes_safe():
    from ..objective_pilot.runner import memory_bytes
    value=memory_bytes(os.getpid())
    if value is None:raise safety.MLFailure('input_limit')
    return value
