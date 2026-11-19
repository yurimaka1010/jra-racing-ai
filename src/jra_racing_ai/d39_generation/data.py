"""Read-only monthly refresh; shared causal formulas with isolated 2026 permission."""
from collections import Counter, deque
from datetime import date,timedelta
from itertools import groupby
from pathlib import Path
import csv
import time

from ..core import RACE_KEY
from ..db_connection import validate_audit_sql
from ..inspect_schema import COLUMNS_SQL
from ..ml import safety
from ..ml.data import TABLE_SQL
from ..research import features as rf, io as prior_io
from ..objective_pilot import data as cache
from ..win_sim import local
from . import contracts as c


def latest(session):
    sql="SELECT kaisai_nen::text AS year,kaisai_tsukihi::text AS month_day,count(*) AS races FROM ONLY public.jvd_ra WHERE data_kubun::text='7' AND kaisai_nen::text='2026' AND kaisai_tsukihi::text<=%s AND keibajo_code::text IN ('01','02','03','04','05','06','07','08','09','10') GROUP BY 1,2 ORDER BY 2 DESC LIMIT 32"
    with session() as db:
        rows=db.aggregate('d39.latest',sql,(c.clock().strftime('%m%d'),),max_rows=32)
    if not rows:
        raise safety.MLFailure('insufficient_history')
    newest=date(int(rows[0]['year']),int(rows[0]['month_day'][:2]),int(rows[0]['month_day'][2:]))
    if newest>c.clock().date():
        raise safety.MLFailure('invalid_split')
    return newest,rows


def months(start,end):
    while start<=end:
        following=date(start.year+(start.month==12),1 if start.month==12 else start.month+1,1)
        yield start,min(end,following-timedelta(days=1))
        start=following


def source_query(columns,start,end,plan,*,live=False):
    ident=safety.identifier
    if live:
        if start!=end or start!=c.clock().date() or start<=date.fromisoformat(plan['latest_final_date']):
            raise safety.MLFailure('invalid_split')
        fields={k:v for k,v in c.features.RAW_FIELDS.items() if v[3] in ('key','identity') or v[3].startswith('declared')}
    else:
        c.guard_day(start,plan);c.guard_day(end,plan)
        if start.year!=end.year:
            raise safety.MLFailure('invalid_split')
        fields=c.features.RAW_FIELDS
    selections=['ra.'+ident(k)+'::text AS '+ident(k) for k in RACE_KEY]
    for alias,(table,name,width,role) in fields.items():
        if name not in columns[table]:
            if alias in ('horse_number','horse_id','finish_position','actual_runners','surface','distance','declared_runners','frame','age','sex','weight_raw'):
                raise safety.MLFailure('invalid_schema')
            selections.append('NULL::text AS '+ident(alias))
        else:
            ref=table+'.'+ident(name)
            selections.append(f'CASE WHEN pg_catalog.octet_length({ref})<={width} THEN {ref}::text ELSE NULL::text END AS '+ident(alias))
    equal=' AND '.join('ra.'+ident(k)+'=se.'+ident(k) for k in RACE_KEY)
    status="IN ('1','2','3','4','5','6')" if live else "='7'"
    sql=('SELECT '+','.join(selections)+' FROM ONLY public.jvd_ra ra JOIN ONLY public.jvd_se se ON '+equal+
         ' WHERE ra.data_kubun::text '+status+' AND se.data_kubun::text '+status+
         ' AND ra.kaisai_nen::text=%s AND ra.kaisai_tsukihi::text BETWEEN %s AND %s '+
         "AND ra.keibajo_code::text IN ('01','02','03','04','05','06','07','08','09','10') ORDER BY "+
         ','.join('ra.'+ident(k) for k in RACE_KEY)+',se.'+ident('umaban')+' LIMIT %s')
    validate_audit_sql(sql)
    return sql,(str(start.year),start.strftime('%m%d'),end.strftime('%m%d'),3001 if live else 20001),tuple(fields)


def columns(db):
    result={}
    for alias in ('ra','se'):
        table='jvd_'+alias
        info=db.aggregate('d39.table',TABLE_SQL,('public',table),max_rows=1)
        if (len(info)!=1 or info[0]['kind']!='r' or info[0]['access_method']!='heap'
                or any(info[0][k] for k in ('is_partition','descendants','rls','forced_rls'))):
            raise safety.MLFailure('invalid_schema')
        rows=db.aggregate('d39.columns',COLUMNS_SQL,('public',table),max_rows=1000)
        result[alias]={row['name']:row for row in rows}
        for name in [*RACE_KEY,*[v[1] for v in c.features.RAW_FIELDS.values() if v[0]==alias]]:
            if name in result[alias] and (not result[alias][name]['type_is_builtin'] or result[alias][name]['type_name'] not in local.TEXT_TYPES):
                raise safety.MLFailure('invalid_schema')
    return result


def collect(directory,session):
    plan=c.approved(directory)
    with session() as db: schema=columns(db)
    past=c.read(c.S2/'source_manifest.json')
    end=date.fromisoformat(plan['latest_final_date']);start=date.fromisoformat(past['source_end'])+timedelta(days=1)
    out=safety.local_path(directory/'source',root=c.ROOT);out.mkdir(exist_ok=True)
    entries=[];began=time.monotonic()
    for first,last in months(start,end):
        if time.monotonic()-began>900:raise safety.MLFailure('query_timeout')
        path=out/(first.strftime('%Y%m')+'.csv');side=path.with_suffix('.json')
        if side.exists():
            item=c.read(side)
            if c.sha(path)!=item['sha256']:raise safety.MLFailure('artifact_invalid')
            entries.append(item);continue
        sql,args,fields=source_query(schema,first,last,plan)
        missing=Counter();count=0
        # An interrupted incomplete fetch never becomes a reusable completed chunk.
        partial=out/(first.strftime('%Y%m')+'.'+__import__('uuid').uuid4().hex+'.partial.csv')
        with session() as db,partial.open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=(*RACE_KEY,*fields));writer.writeheader()
            for batch in db.iter_batches('d39.month',sql,args,max_rows=20000,batch_size=1000,max_seconds=180):
                for row in batch:
                    writer.writerow({k:'' if row.get(k) is None else row[k] for k in (*RACE_KEY,*fields)})
                    count+=1;missing.update(k for k in fields if row.get(k) is None or str(row[k]).strip()=='')
        if path.exists():raise safety.MLFailure('artifact_exists')
        partial.rename(path)
        item={'file':path.name,'start':str(first),'end':str(last),'rows':count,'sha256':c.sha(path),'raw_missing':dict(missing)}
        c.write(side,item);entries.append(item)
    c.write(directory/'source_manifest.json',{'old_source':str(c.S2),'old_manifest_sha256':c.sha(c.S2/'source_manifest.json'),
        'old_source_end':past['source_end'],'new_months':entries,'latest_final_date':str(end),
        'new_rows':sum(x['rows'] for x in entries),'2026_training_authorized':True,
        'population':'all_JRA_RA7_SE7;not_target_horse_filtered','DB_read_only':True})
    return {'new_source_rows':sum(x['rows'] for x in entries),'latest_final_date':str(end),'months':len(entries)}


def iter_days(directory,*,only_old=False):
    manifests=[(c.S2/'source',c.read(c.S2/'source_manifest.json')['months'])]
    if not only_old:
        manifests.append((directory/'source',c.read(directory/'source_manifest.json')['new_months']))
    for root,items in manifests:
        for item in items:
            path=root/item['file']
            if c.sha(path)!=item['sha256']:raise safety.MLFailure('artifact_invalid')
            with path.open(encoding='utf-8-sig',newline='') as stream:
                for _,group in groupby(csv.DictReader(stream),key=lambda row:row['kaisai_nen']+row['kaisai_tsukihi']):
                    rows=list(group)
                    for row in rows:
                        value=rf.num(row.get('weight_raw'))
                        row['weight_kg']=str(value/10) if value is not None else ''
                    yield rows


def replay(builder,rows):
    """Warm up the unchanged full-population history without materializing old features."""
    day=rf.legacy.entry_identity(rows[0])[1]
    if builder.last_day is not None and day<=builder.last_day:raise safety.MLFailure('invalid_split')
    builder._available(day)
    races={}
    for row in rows:races.setdefault(rf.legacy.entry_identity(row)[0][:-1],[]).append(row)
    events=[event for _,group in sorted(races.items()) for event in rf.race_summaries(group)]
    builder.pending.append((day,events));builder.last_day=day


def validate_frame(frame):
    import numpy as np
    import re
    if frame.empty or frame.entry_key.duplicated().any():raise safety.MLFailure('duplicate_entry')
    for key,day in zip(frame.entry_key,frame.date):
        if not re.fullmatch('[0-9]{18}',str(key)) or key[:8]!=date.fromisoformat(str(day)).strftime('%Y%m%d'):
            raise safety.MLFailure('invalid_identity')
    for _,part in frame.groupby(frame.entry_key.str[:16],sort=False):
        y=part.winner_label.to_numpy(float)
        if not np.isin(y,[0,1]).all() or y.sum()!=1 or not 2<=len(part)<=28:
            raise safety.MLFailure('invalid_identity')


def build(directory):
    import pandas as pd
    plan=c.approved(directory);builder=rf.Builder();created=[];buffers={};excluded=Counter();began=time.monotonic()
    out=safety.local_path(directory/'features',root=c.ROOT);out.mkdir(exist_ok=True)
    old_end=date.fromisoformat(c.read(directory/'source_manifest.json')['old_source_end'])
    for rows in iter_days(directory):
        if time.monotonic()-began>900:raise safety.MLFailure('query_timeout')
        day=rf.legacy.entry_identity(rows[0])[1];c.guard_day(day,plan)
        if day<=old_end:
            replay(builder,rows);continue
        x,meta=builder.transform_day(rows)
        groups={}
        for row in rows:groups.setdefault(rf.legacy.entry_identity(row)[0][:-1],[]).append(row)
        complete={key: len(values)>=2 and all(rf.num(v.get('actual_runners'))==len(values) for v in values)
                  for key,values in groups.items()}
        for key,group in groups.items():
            if not complete[key]:excluded['new_races_incomplete_starter_set']+=1
        eligible_races={tuple(m['key'][:-1]) for m in meta if m['eligible']}
        excluded['new_races_labels_or_required_declared_fields_unavailable']+=len(set(groups)-eligible_races)
        for values,m in zip(x,meta):
            item=dict(entry_key=''.join(m['key']),date=m['date'],winner_label=m['winner_label'],
                      eligible=int(m['eligible'] and complete[tuple(m['key'][:-1])]),
                      start_time=m['start_time'],previous_start=m['previous_start'],**values)
            buffers.setdefault(day.strftime('%Y%m'),[]).append(item)
    for month,rows in sorted(buffers.items()):
        path=out/(month+'.csv');frame=pd.DataFrame(rows)
        with path.open('x',encoding='utf-8',newline='') as stream:frame.to_csv(stream,index=False)
        created.append({'file':str(path),'start':str(frame.date.min()),'end':str(frame.date.max()),
                        'rows':len(frame),'eligible_rows':int(frame.eligible.sum()),'sha256':c.sha(path)})
    old_features=c.read(c.S2/'feature_manifest.json')
    references=[{'file':str(c.S2/'features'/item['file']),'sha256':item['sha256']} for item in old_features['years'] if item['year']>=2022]
    references+=created
    frames=[]
    for item in references:
        path=Path(item['file'])
        if c.sha(path)!=item['sha256']:raise safety.MLFailure('artifact_invalid')
        frame=cache.read_frame(path)
        frames.append(frame.loc[(frame.date>='2022-01-01')&(frame.eligible==1),['entry_key','date','winner_label',*c.features.ALL_FEATURES]].copy())
    all_rows=pd.concat(frames,ignore_index=True).sort_values('entry_key').reset_index(drop=True)
    if len(all_rows)>400000:raise safety.MLFailure('input_limit')
    validate_frame(all_rows)
    parts={};by_year={}
    for year,part in all_rows.groupby(all_rows.date.str[:4]):
        by_year[str(year)]={'rows':len(part),'races':int(part.entry_key.str[:16].nunique()),
                          'missing':{name:int(part[name].isna().sum()) for name in c.features.ALL_FEATURES}}
    for name,span in plan['spans'].items():
        frame=all_rows.loc[(all_rows.date>=span['start'])&(all_rows.date<=span['end'])].copy()
        validate_frame(frame)
        path=directory/(name+'.local.csv')
        with path.open('x',encoding='utf-8',newline='') as stream:frame.to_csv(stream,index=False)
        parts[name]={'file':path.name,'rows':len(frame),'races':int(frame.entry_key.str[:16].nunique()),
                     'start':str(frame.date.min()),'end':str(frame.date.max()),'sha256':c.sha(path),
                     'missing':{field:int(frame[field].isna().sum()) for field in c.features.ALL_FEATURES}}
    c.write(directory/'data_manifest.json',{'version':c.VERSION,'dataset_id':__import__('uuid').uuid4().hex,
        'parts':parts,'by_year':by_year,'feature_spec':c.FEATURE_SPEC,'new_source_exclusions':dict(excluded),
        'old_feature_manifest_sha256':c.sha(c.S2/'feature_manifest.json'),'new_feature_files':created,
        '2026_authorized_for_this_new_model_only':True,'seven_day_delay_unchanged':True,
        'C_population':'full_past_race','D_population':'all_past_JRA_starters','latest_date':plan['latest_final_date']})
    return {'parts':{key:{k:value[k] for k in ('rows','races','start','end')} for key,value in parts.items()},'latest_date':plan['latest_final_date']}


def load_part(directory,name):
    plan=c.approved(directory)
    item=c.read(directory/'data_manifest.json')['parts'][name]
    path=directory/item['file']
    if c.sha(path)!=item['sha256']:raise safety.MLFailure('artifact_invalid')
    frame=cache.read_frame(path)
    if (len(frame)!=item['rows'] or not (frame.date>=plan['spans'][name]['start']).all()
            or not (frame.date<=plan['spans'][name]['end']).all()):raise safety.MLFailure('invalid_split')
    validate_frame(frame)
    return frame
