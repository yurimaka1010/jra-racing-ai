"""Declared future entries and seven-day-old full-population history; no fitting."""
from collections import defaultdict
from datetime import date,datetime,timedelta
from itertools import groupby
import csv
from uuid import uuid4

from ..core import RACE_KEY
from ..db_connection import validate_audit_sql
from ..ml import safety
from ..research import features as rf
from . import contracts as c,data,forward


def current_odds_query(races, starts):
    """Same observed-T10 contract, from the table updated by live O1 delivery.

    Historical apd_sokuho_o1 may not contain today's updates. Reuse every time,
    width, status and duplicate guard of the frozen H input builder; only the
    four-model live transport changes. Stage 7 keeps its own signed adapter.
    """
    from ..stage7 import inputs
    sql, args = inputs.query(races, starts, (30, 10), historical=False)
    token = 'public.apd_sokuho_o1'
    if sql.count(token) != 2: raise safety.MLFailure('artifact_invalid')
    sql = sql.replace(token, 'public.jvd_o1')
    validate_audit_sql(sql)
    return sql, args


def current_odds_grouped(rows):
    from ..stage7 import inputs
    grouped = inputs.grouped(rows)
    for values in grouped.values():
        for row in values: row['source_table'] = 'jvd_o1'
    return grouped


def entry_query(schema,day):
    fields={k:v for k,v in c.features.RAW_FIELDS.items() if v[3] in ('key','identity') or v[3].startswith('declared')}
    ident=safety.identifier;cols=['ra.'+ident(k)+'::text AS '+ident(k) for k in RACE_KEY]
    for alias,(table,name,width,_) in fields.items():
        ref=table+'.'+ident(name)
        cols.append((f'CASE WHEN pg_catalog.octet_length({ref})<={width} THEN {ref}::text ELSE NULL::text END'
                     if name in schema[table] else 'NULL::text')+' AS '+ident(alias))
    equal=' AND '.join('ra.'+ident(k)+'=se.'+ident(k) for k in RACE_KEY)
    sql=('SELECT '+','.join(cols)+' FROM ONLY public.jvd_ra ra JOIN ONLY public.jvd_se se ON '+equal+
         " WHERE ra.data_kubun::text IN ('1','2','3','4','5','6') AND se.data_kubun::text IN ('1','2','3','4','5','6') "
         "AND ra.kaisai_nen::text=%s AND ra.kaisai_tsukihi::text=%s AND ra.keibajo_code::text IN ('01','02','03','04','05','06','07','08','09','10') "
         'ORDER BY '+','.join('ra.'+ident(k) for k in RACE_KEY)+',se.'+ident('umaban')+' LIMIT 3001')
    validate_audit_sql(sql);return sql,(str(day.year),day.strftime('%m%d'))


def entries(db,day):
    schema=data.columns(db);sql,args=entry_query(schema,day)
    rows=db.aggregate('four.declared_only',sql,args,max_rows=3000);out=defaultdict(list)
    for row in rows:out[tuple(row[k] for k in RACE_KEY)].append(row)
    return out,c.clock()


def history(parent,session,day):
    """Only new inference-history snapshots; original cache and all models stay fixed."""
    cutoff=day-timedelta(days=7);base_end=date.fromisoformat(c.read(parent/'source_manifest.json')['latest_final_date'])
    builder=rf.Builder()
    for rows in data.iter_days(parent):data.replay(builder,rows)
    if cutoff<=base_end:
        builder._available(day);return builder,{'extra_rows':0,'cutoff':str(cutoff),'base_cache_reused':True}
    root=forward.mkdir(parent/'history-four'/str(day));manifest=root/'manifest.json'
    if not manifest.exists():
        items=[];plan=dict(c.approved(parent),latest_final_date=str(cutoff))
        with session() as db:schema=data.columns(db)
        for start,end in data.months(base_end+timedelta(days=1),cutoff):
            sql,args,fields=data.source_query(schema,start,end,plan)
            path=root/(start.strftime('%Y%m%d')+'_'+end.strftime('%Y%m%d')+'_'+uuid4().hex[:8]+'.csv');count=0
            with session() as db,path.open('x',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=(*RACE_KEY,*fields));writer.writeheader()
                for batch in db.iter_batches('four.delayed_history',sql,args,max_rows=20000,batch_size=1000,max_seconds=180):
                    for row in batch:writer.writerow({k:'' if row.get(k) is None else row[k] for k in (*RACE_KEY,*fields)});count+=1
            items.append({'file':path.name,'rows':count,'sha256':c.sha(path),'start':str(start),'end':str(end)})
        c.write(manifest,{'target_date':str(day),'cutoff':str(cutoff),'files':items,'observed_at':c.clock().isoformat(),
                         'purpose':'prediction_history_only;no_fit_or_performance_scoring','population':'all_JRA_RA7_SE7'})
    info=c.read(manifest)
    if info['cutoff']!=str(cutoff):raise safety.MLFailure('artifact_invalid')
    for item in info['files']:
        path=root/item['file']
        if c.sha(path)!=item['sha256']:raise safety.MLFailure('artifact_invalid')
        with path.open(encoding='utf-8',newline='') as stream:
            for _,group in groupby(csv.DictReader(stream),key=lambda r:r['kaisai_nen']+r['kaisai_tsukihi']):
                rows=list(group)
                for row in rows:
                    weight=rf.num(row.get('weight_raw'));row['weight_kg']=weight/10 if weight is not None else None
                if rf.legacy.entry_identity(rows[0])[1]>cutoff:raise safety.MLFailure('invalid_split')
                data.replay(builder,rows)
    builder._available(day)
    return builder,{'extra_rows':sum(i['rows'] for i in info['files']),'cutoff':str(cutoff),'manifest_sha256':c.sha(manifest)}
