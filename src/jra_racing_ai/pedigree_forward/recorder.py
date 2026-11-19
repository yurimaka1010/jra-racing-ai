"""Append-only T10 recorder. Current results, payouts and performance are absent."""
from collections import Counter
from datetime import date,timedelta
from pathlib import Path
from uuid import uuid4
import csv
import hashlib
import json
import os
import time

from ..ml import safety
from ..win_sim import core,publication,local
from ..stage7 import forward as cards_api,inputs
from ..d39_generation import forward as entry_api,four_sources
from ..d39_next.forward import live_frame
from ..pedigree_research import data as blood_data,learning as blood_model
from ..d39_enrichment import learning as enriched_model,data as enriched_data
from ..research import features as legacy
from . import contracts as c,protocol,history
from .timing import quote_valid


def frames(context,entries):
    import pandas as pd
    from ..pedigree_research.features import stable_id
    if any(not stable_id(r.get('horse_id')) for r in entries):raise safety.MLFailure('invalid_identity')
    f47=live_frame(context['ability'],entries).sort_values('entry_key').reset_index(drop=True)
    values=context['blood'].transform_day(entries,prediction=True)
    mapped={''.join(legacy.legacy.entry_identity(r)[0]):v for r,v in zip(entries,values)}
    f39=f47.loc[:,['entry_key','date',*blood_data.spec()['features']]].copy()
    f45=f39.copy()
    for name in blood_data.spec(True)['features'][39:]:f45[name]=[mapped[k][name] for k in f39.entry_key]
    if len(f39)!=len(entries) or set(f39.entry_key)!=set(mapped):raise safety.MLFailure('invalid_identity')
    return {'control39':f39,'pedigree45':f45,'reference47':f47}


def capture(folder,kind,payload,seen):
    # New independently observed entry snapshots; quote first_seen never advances.
    identity={'payload':payload,'observed_at':seen.isoformat()} if kind=='entries' else payload
    path=c.mkdir(folder/kind)/(c.digest(identity)+'.json')
    if not path.exists():c.write(path,{'payload':payload,'first_seen_at':seen.isoformat()})
    return path


def quotes(folder):
    return [dict(v['payload'],first_seen_at=v['first_seen_at'])
        for path in sorted((folder/'odds').glob('*.json')) for v in [c.read(path)]]


def load_models(freeze=None):
    # A failed model is a missing model, not a reason to discard other forecasts.
    bundles={};reasons={}
    for name in c.NAMES:
        try:
            path=Path(c.read(c.REFERENCE/'candidate_freeze.json')['model_path']) if name=='reference47' else Path(c.read(c.TRAIN/(name+'.candidate_freeze.json'))['model_path'])
            if freeze:
                expected=freeze['models'][name]
                if (c.sha(path/'model.json')!=expected['model_json_sha256']
                        or c.sha(path/'booster.lgb.txt')!=expected['booster_sha256']):raise safety.MLFailure('invalid_model')
            bundle=(enriched_model.load if name=='reference47' else blood_model.load)(path)
            if bundle[1]['model_id']!=c.MODEL_IDS[name] or bundle[1]['temperature']!=c.TEMPERATURE[name]:
                raise safety.MLFailure('invalid_model')
            bundles[name]=bundle
        except Exception:bundles[name]=None;reasons[name]='frozen_model_unavailable'
    return bundles,reasons


def calculate(folder,registered,snapshot,quote,bundles,context,freeze,*,calculated_at,quote_reason=None,replay=False,clock=c.clock):
    import numpy as np
    roster=registered['roster'];raw_entries=snapshot['payload'];by_horse={r['horse_number']:r for r in raw_entries}
    if len(raw_entries)!=len(roster) or set(by_horse)!=set(roster):raise safety.MLFailure('invalid_identity')
    if quote is not None and not quote_valid(quote,roster,registered['input_cutoff']):
        quote=None;quote_reason='odds_time_or_roster_unverified'
    computed=frames(context,raw_entries);models={};rows=[]
    for name in c.NAMES:
        bundle=bundles.get(name);frame=computed[name]
        try:
            if bundle is None:raise safety.MLFailure('invalid_model')
            booster,meta=bundle
            encoded=(enriched_data.matrix if name=='reference47' else blood_data.matrix)(frame,meta['feature_spec'],meta['categories'])
            types={f:str(t) for f,t in encoded.dtypes.items()}
            if (list(encoded.columns)!=freeze['models'][name]['features']
                    or types!=freeze['models'][name]['input_types']):raise safety.MLFailure('invalid_model')
            raw=(enriched_model.predict if name=='reference47' else blood_model.predict)(bundle,frame,raw=True)
            p=(enriched_model.matrices.normalize if name=='reference47' else blood_model.matrices.normalize)(frame,raw,c.TEMPERATURE[name])
            if (not np.isfinite(raw).all() or not np.isfinite(p).all() or not ((np.asarray(raw)>=0)&(np.asarray(raw)<=1)).all()
                    or len(p)!=len(roster) or abs(float(p.sum())-1)>1e-12):raise safety.MLFailure('invalid_model')
            probs=dict(zip(frame.entry_key.str[-2:],map(float,p)));q=dict(zip(frame.entry_key.str[-2:],map(float,raw)))
            ranks={h:i for i,h in enumerate(sorted(probs,key=lambda h:(-probs[h],h)),1)}
            path=folder/(name+'.encoded.local.csv')
            with path.open('x',encoding='utf-8',newline='') as stream:
                encoded.to_csv(stream,index=False);stream.flush();os.fsync(stream.fileno())
            model={'status':'saved','model_id':meta['model_id'],'temperature':c.TEMPERATURE[name],
                'raw_probability':q,'probability':probs,'rank':ranks,
                'input_matrix_file':path.name,'input_matrix_sha256':c.sha(path),
                'input_column_sha256':{f:hashlib.sha256(encoded[f].to_csv(index=False).encode()).hexdigest() for f in encoded},
                'entry_key_order_sha256':c.digest(list(frame.entry_key)),'input_types':types,
                'categories_sha256':c.digest(meta['categories']),'missing_horse_numbers':[]}
        except Exception:
            model={'status':'missing','reason':'model_prediction_or_matrix_failed','model_id':c.MODEL_IDS[name],
                   'temperature':c.TEMPERATURE[name],'missing_horse_numbers':list(roster)}
        models[name]=model
    all_models=all(v['status']=='saved' for v in models.values())
    price_ok=quote is not None and set(quote['odds'])==set(roster)
    for name,m in models.items():
        best=min(roster,key=lambda h:(-m['probability'][h],h)) if m['status']=='saved' else None
        for horse in roster:
            raw=by_horse[horse]
            # The age-dependent declared condition is generated by the same builder.
            f=computed[name];condition=f.loc[f.entry_key.str[-2:]==horse,'b_condition'].iloc[0]
            import pandas as pd
            rows.append({'race_id':''.join(registered['race_key']),'horse_number':horse,'horse_registration_id':raw['horse_id'],
                'condition_code':None if pd.isna(condition) else str(condition),'model':name,'model_id':c.MODEL_IDS[name],
                'temperature':c.TEMPERATURE[name],'raw_probability_q':m.get('raw_probability',{}).get(horse),
                'probability_p':m.get('probability',{}).get(horse),'rank':m.get('rank',{}).get(horse),
                'prediction_at':calculated_at.isoformat(),'cutoff_at':registered['input_cutoff'],
                'save_completed_receipt':'receipt.json','model_status':m['status'],'missing_reason':m.get('reason'),
                'T10_win_odds':quote['odds'].get(horse) if price_ok else None,
                'odds_first_seen_at':quote.get('first_seen_at') if quote else None,
                'odds_published_at':quote.get('published_at') if quote else None,
                'original_PC_received_at':None,'odds_observation_basis':'research_recorder_DB_first_seen_not_original_PC_receipt',
                'odds_fresh_300s':price_ok,'odds_missing_reason':quote_reason,
                'R1_virtual_stake_yen':100 if all_models and price_ok and horse==best else 0,
                'R1_skip_reason':None if all_models and price_ok else 'odds_unavailable' if not price_ok else 'models_incomplete',
                'code_commit_id':freeze['code_commit_id'],'replay_only':replay})
    completed=clock().isoformat()
    for row in rows:row.update(prediction_started_at=calculated_at.isoformat(),prediction_at=completed)
    return {'version':c.VERSION,'race_key':registered['race_key'],'date':registered['scheduled_start'][:10],
        'scheduled_start':registered['scheduled_start'],'input_cutoff':registered['input_cutoff'],
        'roster':roster,'horse_registration_ids':{h:by_horse[h]['horse_id'] for h in roster},
        'entry_snapshot':{k:snapshot[k] for k in ('path','sha256','first_seen_at')},
        'models':models,'rows':rows,'quote_T10':quote,'odds_missing_reason':quote_reason,
        'prediction_started_at':calculated_at.isoformat(),'prediction_at':completed,'history':context['info'],'code_commit_id':freeze['code_commit_id'],
        'P1_prediction_side_complete':all_models,'P2_P3_odds_ready':price_ok,'purchase_rule':'R1_only',
        'no_current_results_accessed':True,'H_results_accessed':False,'new_fits':0,'replay_only':replay,
        'actual_race_start_time_verified':False,
        'lineage_corrections_asof':'unverified_static_parent_IDs'}


def write_record(folder,record,*,clock=c.clock,deadline=None,replay=False):
    path=folder/'prediction.json';c.write(path,record)
    with (folder/'predictions.local.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(record['rows'][0]));w.writeheader();w.writerows(record['rows'])
        stream.flush();os.fsync(stream.fileno())
    written=clock();on_time=replay or (deadline is not None and written<=deadline)
    receipt={'status':'saved' if on_time else 'missing','reason':None if on_time else 'save_deadline_missed',
        'prediction_sha256':c.sha(path),'csv_sha256':c.sha(folder/'predictions.local.csv'),
        'save_completed_at':written.isoformat(),'save_deadline':deadline.isoformat() if deadline else None,
        'all_models_complete':record['P1_prediction_side_complete'],'odds_ready':record['P2_P3_odds_ready'],
        'per_row_completion':'all_prediction_rows_share_this_receipt_timestamp;prediction_bytes_never_rewritten',
        'replay_only':replay}
    c.write(folder/'receipt.json',receipt)
    if not replay and clock()>deadline:c.write(folder/'veto.json',{'reason':'receipt_write_late'})
    return receipt


def decide(folder,reg,current,bundles,context,freeze,prereg,*,clock=c.clock):
    if (folder/'receipt.json').exists():return c.read(folder/'receipt.json')
    now=clock();start=core.instant(reg['scheduled_start']) if reg.get('scheduled_start') else None
    cutoff=start-timedelta(minutes=10) if start else None;deadline=cutoff+timedelta(seconds=60) if cutoff else None
    if cutoff and now<cutoff:return {'status':'pending'}
    reason=reg.get('reason')
    if (folder/'attempt.json').exists():reason='incomplete_previous_attempt;no_recalculation'
    if cutoff and now>deadline:reason='save_deadline_missed'
    if not reason and not cards_api.same_card(reg,current):reason='schedule_or_roster_changed'
    snapshot=None;quote=None;quote_reason=None
    if not reason:
        snapshot,reason=entry_api.on_time_entries(folder,reg['roster'],cutoff)
        quote,quote_reason=inputs.select(tuple(reg['race_key']),quotes(folder),reg['roster'],cutoff,300,require_seen=True)
    if reason:
        value={'status':'missing','reason':reason,'recorded_at':now.isoformat(),'race_key':reg.get('race_key'),
            'missing_horse_numbers':reg.get('roster',[]),'model_status':{n:'missing' for n in c.NAMES},'new_fits':0,'results_read':False}
        c.write(folder/'receipt.json',value);return value
    c.write(folder/'attempt.json',{'at':now.isoformat(),'cutoff':cutoff.isoformat(),'no_recalculation':True})
    record=calculate(folder,dict(reg,input_cutoff=cutoff.isoformat()),snapshot,quote,bundles,context,freeze,
                     calculated_at=now,quote_reason=quote_reason,clock=clock)
    record.update(formal_period=prereg['window_start_week']<=str(now.date())<=prereg['window_end_date'],
                  model_freeze_content_sha256=c.digest(freeze),preregistration_content_sha256=c.digest(prereg))
    return write_record(folder,record,clock=clock,deadline=deadline)


def event(directory,reason,**counts):
    root=c.mkdir(directory/'operational-logs')
    c.write(root/(c.clock().strftime('%Y%m%dT%H%M%S%f')+'_'+uuid4().hex[:6]+'.json'),
        {'at':c.clock().isoformat(),'reason':reason,'counts':counts,'results_read':False,'new_fits':0})


def tick(directory,freeze,prereg,bundles,context,db_session):
    day=c.clock().date();root=c.mkdir(directory/'forward'/str(day)/'races');registered=[]
    with db_session() as db:
        cards=cards_api.live_cards(db,day);groups,seen=four_sources.entries(db,day)
        active=[]
        for race,card in sorted(cards.items()):
            folder=c.mkdir(root/''.join(race))
            if not (folder/'registered.json').exists():
                reason=card.get('reason')
                if card.get('scheduled_start') and core.instant(card['scheduled_start'])-timedelta(minutes=10)<=seen:
                    reason=reason or 'registered_after_cutoff'
                c.write(folder/'registered.json',dict(card,reason=reason,registered_at=seen.isoformat()))
            reg=c.read(folder/'registered.json');registered.append((folder,reg))
            if (folder/'receipt.json').exists():continue
            if reg.get('scheduled_start'):
                start=core.instant(reg['scheduled_start'])
                if start-timedelta(minutes=15)<=seen<=start-timedelta(minutes=9):
                    capture(folder,'entries',groups.get(race,[]),seen);active.append((folder,reg))
        try:
            if active and publication._columns(db,'jvd_o1',publication.O1_FIELDS,local.Budget(10)) is None:
                raise safety.MLFailure('invalid_schema')
            for offset in range(0,len(active),32):
                part=active[offset:offset+32];races=[tuple(r['race_key']) for _,r in part]
                sql,args=four_sources.current_odds_query(races,{tuple(r['race_key']):r['scheduled_start'] for _,r in part})
                raw=db.aggregate('pedigree_forward.observed_O1',sql,args,max_rows=128)
                grouped=four_sources.current_odds_grouped(raw);observed=c.clock()
                for folder,reg in part:
                    for q in grouped.get((tuple(reg['race_key']),10),[]):capture(folder,'odds',q,observed)
        except Exception:
            # P1 has no odds requirement. A price read/schema failure must not
            # discard the already-observed declared entries or the forecasts.
            event(directory,'odds_fetch_unavailable_P1_continues')
    counts=Counter()
    for folder,reg in registered:
        try:
            result=decide(folder,reg,cards.get(tuple(reg['race_key'])),bundles,context,freeze,prereg)
            counts[result['status']]+=1
            if result.get('status')=='saved' and not result.get('odds_ready'):counts['odds_missing_or_stale']+=1
            if result.get('status')=='saved' and not result.get('all_models_complete'):counts['models_incomplete']+=1
        except Exception:
            counts['recording_error']+=1
            if not (folder/'receipt.json').exists():
                c.write(folder/'receipt.json',{'status':'missing','reason':'prediction_or_write_failed','results_read':False})
    return {'registered_races':len(cards),'counts':dict(counts),'results_read':False,'new_fits':0}


def preflight(directory,*,db_session=history.session):
    freeze,p=protocol.load(directory);bundles,errors=load_models(freeze)
    if errors:raise safety.MLFailure('invalid_model')
    with db_session() as db:
        four_sources.data.columns(db)
        if publication._columns(db,'jvd_o1',publication.O1_FIELDS,local.Budget(10)) is None:raise safety.MLFailure('invalid_schema')
    value={'status':'ready','model_count':len(bundles),'model_ids':c.MODEL_IDS,'temperatures':c.TEMPERATURE,
        'before_record_start':c.clock().date()<c.FIRST_DAY,'actual_predictions_started':False,'new_fits':0,
        'db_read_only_check':True,'formal_start_week':p['window_start_week'],'end_date':p['window_end_date']}
    c.write(c.mkdir(directory/'readiness')/(uuid4().hex+'.json'),value)
    return value


def run(directory,*,once=False,db_session=history.session):
    freeze,p=protocol.load(directory,allow_missing_models=True);now=c.clock();day=now.date()
    if day<c.FIRST_DAY:return dict(preflight(directory,db_session=db_session),reason='before_first_record_day')
    if str(day)>p['window_end_date']:return {'status':'ok','reason':'recording_window_closed','results_read':False,'new_fits':0}
    if now.hour<7 or now.hour>=18:return {'status':'ok','reason':'outside_daily_window','results_read':False,'new_fits':0}
    lock=c.mkdir(directory/'recorder-locks')/(str(day)+'.lock');token=uuid4().hex
    if lock.exists():
        from ..objective_pilot.runner import process_parents
        owner=c.read(lock).get('pid')
        if owner in process_parents():raise safety.MLFailure('input_limit')
        lock.rename(lock.with_name(lock.name+'.stale_'+uuid4().hex+'.json'))
    try:stream=lock.open('x',encoding='utf-8')
    except FileExistsError:raise safety.MLFailure('input_limit') from None
    json.dump({'token':token,'pid':os.getpid()},stream);stream.flush()
    try:
        bundles,errors=load_models(freeze)
        with db_session() as db:cards=cards_api.live_cards(db,day);groups,_=four_sources.entries(db,day)
        if not cards:
            event(directory,'no_JRA_entries');return {'status':'ok','reason':'no_JRA_entries','new_fits':0}
        horses={r['horse_id'] for rows in groups.values() for r in rows if r.get('horse_id')}
        context=history.build(directory,day,horses=horses,db_session=db_session)
        c.write(c.mkdir(directory/'history'/str(day))/('build_'+uuid4().hex+'.json'),context['info'])
        last=None
        while c.clock().date()==day and c.clock().hour<18:
            try:
                value=tick(directory,freeze,p,bundles,context,db_session)
                if value!=last:event(directory,'recording_status',**value);print(json.dumps(value),flush=True);last=value
            except Exception:
                event(directory,'database_or_recording_unavailable')
            if once:return {'status':'ok',**(last or {'reason':'temporarily_unavailable'}),'new_fits':0}
            from ..objective_pilot.runner import memory_bytes
            memory=memory_bytes(os.getpid())
            if memory is None or memory>12*1024**3:raise safety.MLFailure('input_limit')
            time.sleep(15)
        return {'status':'ok','reason':'daily_recording_complete','new_fits':0,'results_read':False}
    finally:
        stream.close()
        if lock.exists() and c.read(lock).get('token')==token:lock.unlink()
