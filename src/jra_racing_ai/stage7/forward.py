"""Append-only research recorder. No result or payout query, no purchase code."""
from collections import Counter,defaultdict
from datetime import date,datetime,timedelta
import json,time
from ..core import RACE_KEY,SafeAuditError
from ..ml import safety
from ..win_sim import core,formats,publication,local
from ..db_connection import validate_audit_sql
from . import contracts as c,store as s,inputs,planning

SE_FIELDS={'record_type':('record_id',2),'status':('data_kubun',1),'horse':('umaban',2)}

def cards_query(day,table,fields):
    if table not in ('jvd_ra','jvd_se'):raise safety.MLFailure('invalid_input')
    if day<date(2026,10,22):raise safety.MLFailure('invalid_split')
    ident=safety.identifier;cols=[ident(k)+'::text AS '+ident(k) for k in RACE_KEY]
    for alias,(physical,width) in fields.items():
        cols.append(f'CASE WHEN pg_catalog.octet_length({ident(physical)})<={width} THEN {ident(physical)}::text ELSE NULL::text END AS '+ident(alias))
    sql=('SELECT '+','.join(cols)+' FROM ONLY public.'+ident(table)+
        " WHERE kaisai_nen=%s AND kaisai_tsukihi=%s AND keibajo_code BETWEEN '01' AND '10' LIMIT %s")
    validate_audit_sql(sql);return sql,(str(day.year),day.strftime('%m%d'),101 if table=='jvd_ra' else 3001)

def live_cards(db,day):
    groups={};ra_fields=publication.RA_FIELDS
    for table,fields in (('jvd_ra',ra_fields),('jvd_se',SE_FIELDS)):
        columns=publication._columns(db,table,fields,local.Budget(60))
        if columns is None:raise safety.MLFailure('invalid_schema')
        sql,args=cards_query(day,table,fields)
        rows=db.aggregate('stage7.live_cards.'+table,sql,args,max_rows=100 if table=='jvd_ra' else 3000)
        for row in rows:
            race=core.key(tuple(row.pop(k) for k in RACE_KEY));v=groups.setdefault(race,{'RA':[],'SE':[]})
            v['RA' if table=='jvd_ra' else 'SE'].append(row)
    out={}
    for race,values in groups.items():
        ra=values['RA'];se=values['SE'];reason=None;start=None;roster=[]
        if len(ra)!=1:reason='schedule_missing_or_duplicate'
        else:
            row=ra[0]
            if row.get('record_type')!='RA' or row.get('status') not in tuple('123456'):reason='not_pre_race_schedule'
            elif row.get('previous_start') not in ('','    ','0000',None):reason='schedule_changed_before_recording'
            try:
                start=formats.parse_time(row['start'],'hhmm_on_race_date',race)
                if row['start']=='0000':reason='schedule_clock_unavailable'
            except (SafeAuditError,KeyError):reason='schedule_clock_unavailable'
        try:
            roster=[core.horse(x['horse']) for x in se]
            if (not 2<=len(roster)<=28 or len(set(roster))!=len(roster)
                    or any(x.get('record_type')!='SE' or x.get('status') not in tuple('123456') for x in se)):
                reason=reason or 'roster_missing_or_unconfirmed'
        except (SafeAuditError,KeyError):reason=reason or 'roster_missing_or_unconfirmed'
        out[race]={'race_key':list(race),'scheduled_start':start.isoformat() if start else None,
            'roster':sorted(roster),'reason':reason,'RA_status':ra[0].get('status') if len(ra)==1 else None}
    return out

def same_card(registered,current):
    return current is not None and current.get('reason') is None and all(registered.get(k)==current.get(k) for k in ('scheduled_start','roster'))

def same_saved_clock_and_roster(registered,current):
    # Data category naturally becomes final after racing. It is not a clock or
    # roster change and must not retrospectively invalidate an on-time record.
    return current is not None and all(registered.get(k)==current.get(k) for k in ('scheduled_start','roster'))

def capture(folder,row,seen_at):
    # A given DB row's first_seen timestamp is never advanced/backdated on reruns.
    payload={k:row.get(k) for k in ('race_key','record_type','status','published','sale','payload','source_table','oversized','unsafe_version')}
    path=s.mkdir(folder/'observations')/(s.digest(payload)+'.json')
    if path.exists():return
    s.write(path,{**payload,'first_seen_at':seen_at.isoformat(),'original_PC_received_at':None})

def observations(folder):
    return [s.read(p) for p in sorted((folder/'observations').glob('*.json'))]

def decide(folder,registered,current,plan,bundle,*,clock=None):
    clock=clock or c.clock;now=clock();race=tuple(registered['race_key']);c.future(race,plan,now)
    if (folder/'decision.json').exists():return s.read(folder/'decision.json')
    start=core.instant(registered['scheduled_start']) if registered['scheduled_start'] else None
    cutoff=start-timedelta(minutes=10) if start else None;deadline=cutoff+timedelta(seconds=60) if cutoff else None
    reason=registered.get('reason')
    if not reason and not same_card(registered,current):reason='schedule_or_roster_changed'
    if not reason and now<cutoff:return {'status':'pending','reason':'before_input_cutoff'}
    if not reason and now>deadline:reason='save_deadline_missed'
    rows=observations(folder);early=None;current_quote=None
    if reason is None:
        current_quote,reason=inputs.select(race,rows,registered['roster'],cutoff,300,require_seen=True)
        early,early_reason=inputs.select(race,rows,registered['roster'],start-timedelta(minutes=30),300,require_seen=True)
    else:early_reason='not_attempted'
    if reason:
        value={'status':'missing','reason':reason,'recorded_at':now.isoformat(),'input_cutoff':cutoff.isoformat() if cutoff else None}
        s.write(folder/'decision.json',value);return value
    # Selection has no path to a results source. The external offset/T1 lives in
    # the unchanged research.model.predict implementation.
    probabilities,frame=inputs.predict(bundle,race,current_quote,early)
    calculated=clock()
    if calculated>deadline:
        value={'status':'missing','reason':'calculation_deadline_missed','recorded_at':calculated.isoformat()}
        s.write(folder/'decision.json',value);return value
    s.verify_refs(plan)
    fields=frame.astype(object).where(frame.notna(),None).to_dict('records')
    record={'version':c.VERSION,'race_key':list(race),'roster':registered['roster'],'model_id':c.H_ID,
        'model_manifest_sha256':plan['references'][str(c.MODEL/'model.json')],'model_sha256':plan['references'][str(c.MODEL/'win.lgb.txt')],
        'temperature':1.,'scheduled_start':registered['scheduled_start'],'schedule_observed_at':registered['registered_at'],
        'schedule_snapshot_sha256':s.sha(folder/'registered.json'),'input_cutoff':cutoff.isoformat(),'input_fixed_at':now.isoformat(),
        'save_deadline':deadline.isoformat(),'quote_T10':current_quote,'quote_T30':early,'T30_missing_reason':early_reason,
        'inputs':fields,'probabilities':probabilities,'calculation_completed_at':calculated.isoformat(),
        'result_accessed':False,'actual_start_clock_verified':False,'interpretation':'T10_input_actual_save_time_with_known_schedule_not_T10_display_claim'}
    s.write(folder/'prediction.json',record)
    # write_json flushes and fsyncs. The completion receipt is separate; a late
    # receipt vetoes evaluation even if the prediction bytes were already saved.
    written=clock()
    value={'status':'saved' if written<=deadline else 'missing','reason':None if written<=deadline else 'write_deadline_missed',
        'prediction_sha256':s.sha(folder/'prediction.json'),'prediction_write_completed_at':written.isoformat(),
        'recorded_at':written.isoformat(),'input_cutoff':cutoff.isoformat(),'save_deadline':deadline.isoformat()}
    s.write(folder/'decision.json',value)
    if clock()>deadline:
        s.event(folder,'save_receipt_late','save_deadline_missed');value=dict(value,status='missing',reason='save_receipt_late')
    return value

def eligible_record(folder,plan):
    if not (folder/'decision.json').exists():return False
    receipt=s.read(folder/'decision.json')
    if receipt['status']!='saved' or any((folder/'events').glob('*.json')):
        # Only veto events affect eligibility; routine status events are not in
        # a per-race folder. Conservatively reject any race-level alteration.
        return False
    if s.sha(folder/'prediction.json')!=receipt['prediction_sha256']:raise safety.MLFailure('artifact_invalid')
    record=s.read(folder/'prediction.json')
    if not (core.instant(record['input_cutoff'])<=core.instant(record['input_fixed_at'])
            <=core.instant(record['save_deadline']) and core.instant(receipt['prediction_write_completed_at'])<=core.instant(record['save_deadline'])):
        return False
    return record['result_accessed'] is False and record['model_id']==c.H_ID and record['temperature']==1.

def frozen_records(folder,plan):
    records=[]
    for path in sorted((folder/'forward').glob('*/races/*')):
        if eligible_record(path,plan):
            record=s.read(path/'prediction.json');receipt=s.read(path/'decision.json')
            records.append(dict(record,save_completed_at=receipt['prediction_write_completed_at']))
    return records

def tick(day_folder,plan,bundle,session):
    begin=c.clock();day=begin.date()
    if not date.fromisoformat(plan['start_date'])<=day<=date.fromisoformat(plan['end_date']):raise safety.MLFailure('invalid_split')
    with session() as db:
        cards=live_cards(db,day);seen=c.clock()
        for race,card in cards.items():
            c.future(race,plan,seen);folder=s.mkdir(day_folder/'races'/''.join(race))
            if not (folder/'registered.json').exists():
                if card['scheduled_start'] and core.instant(card['scheduled_start'])<=seen:
                    card=dict(card,reason='already_started_at_registration')
                s.write(folder/'registered.json',dict(card,registered_at=seen.isoformat(),plan_sha256=s.sha(day_folder.parents[1]/'forward_plan.json')))
        active=[];registered=[]
        for folder in sorted((day_folder/'races').glob('*')):
            reg=s.read(folder/'registered.json');race=tuple(reg['race_key']);registered.append((folder,reg));current=cards.get(race)
            if (folder/'decision.json').exists() and not same_saved_clock_and_roster(reg,current):
                signature=s.digest(current);marker=folder/('change_'+signature+'.json')
                if not marker.exists():
                    s.write(marker,{'status':'post_save_schedule_or_roster_change','observed_at':seen.isoformat()})
                    s.event(folder,'post_save_state_changed','schedule_or_roster_changed')
            if reg['scheduled_start'] and not reg['reason'] and not (folder/'decision.json').exists():
                start=core.instant(reg['scheduled_start'])
                if start-timedelta(minutes=35)<=seen<=start-timedelta(minutes=9):active.append((folder,reg))
        for offset in range(0,len(active),32):
            batch=active[offset:offset+32];races=[tuple(reg['race_key']) for _,reg in batch]
            starts={tuple(reg['race_key']):reg['scheduled_start'] for _,reg in batch}
            columns=publication._columns(db,'apd_sokuho_o1',publication.O1_FIELDS,local.Budget(60))
            if columns is None:raise safety.MLFailure('invalid_schema')
            sql,args=inputs.query(races,starts,(30,10),historical=False)
            raw=db.aggregate('stage7.forward_O1_only',sql,args,max_rows=128);seen_quotes=c.clock();groups=inputs.grouped(raw)
            for folder,reg in batch:
                race=tuple(reg['race_key'])
                for minutes in (30,10):
                    for row in groups.get((race,minutes),[]):capture(folder,row,seen_quotes)
    counts=Counter()
    for folder,reg in registered:
        value=decide(folder,reg,cards.get(tuple(reg['race_key'])),plan,bundle)
        state=value['status']
        if state=='saved' and not eligible_record(folder,plan):state='saved_but_vetoed'
        counts[state]+=1
    return {'planned_races':len(registered),'status_counts':dict(counts),'observed_at':c.clock().isoformat(),
        'fits':0,'result_queries':0}

def record(folder,session,*,once=False):
    plan=planning.verify(folder,'forward');now=c.clock();day=now.date()
    if not date.fromisoformat(plan['start_date'])<=day<=date.fromisoformat(plan['end_date']):raise safety.MLFailure('invalid_split')
    bundle=s.fixed_model(plan);day_folder=s.mkdir(folder/'forward'/day.isoformat());s.mkdir(day_folder/'races')
    with s.lock(day_folder):
        s.event(day_folder,'recorder_started',plan_sha256=s.sha(folder/'forward_plan.json'))
        while True:
            now=c.clock()
            if now.date()!=day or now.hour>=plan['recording_end_hour']:break
            if now.hour<plan['recording_start_hour']:
                if once:break
                time.sleep(15);continue
            try:
                summary=tick(day_folder,plan,bundle,session)
                s.event(day_folder,'input_status',**summary)
                if summary['planned_races']==0:
                    s.event(day_folder,'no_registered_schedule','advance_cards_unavailable_or_no_meeting')
                    break
            except SafeAuditError as exc:
                s.event(day_folder,'read_or_prediction_failed',exc.code)
                if once:raise
            if once:break
            time.sleep(plan['poll_seconds'])
        s.event(day_folder,'recorder_stopped')
    return {'status':'ok','recording_day':str(day),'fits':0,'results_read':False}
