"""Unchanged builders on full-population history available at least seven days ago."""
from collections import Counter
from datetime import date,timedelta
from itertools import groupby
import csv
import os
import time
from uuid import uuid4

from ..core import RACE_KEY
from ..db_connection import read_only_session,validate_audit_sql
from ..win_sim_cli import connection_config
from ..d39_generation import data as old_data
from ..d39_enrichment import sources as supplements
from ..d39_enrichment.features import Builder as AbilityBuilder
from ..pedigree_research.features import Builder as PedigreeBuilder,stable_id
from ..pedigree_research import sources as ancestry
from ..ml.encodings import jv_time_seconds
from ..research import features as legacy
from ..ml import safety
from . import contracts as c

HISTORY_POPULATION_VERSION='final7_exclude_cancelled9_deleted0_v2'


def check_population(expected,seen):
    """9=cancelled / 0=deleted have no final-runner history to ingest.

    Other non-final categories are incomplete updates, and still stop the read.
    A cancelled/deleted key must not appear in the final joined runner source.
    """
    final=set();ignored=set();counts=Counter();all_keys=set()
    for row in expected:
        key=tuple(row[k] for k in RACE_KEY)
        if key in all_keys:raise safety.MLFailure('duplicate_entry')
        all_keys.add(key);status=row['status'];counts[status]+=1
        if status=='7':final.add(key)
        elif status in ('9','0'):ignored.add(key)
        else:raise safety.MLFailure('insufficient_history')
    if final!=set(seen) or ignored & set(seen):raise safety.MLFailure('insufficient_history')
    return {'version':HISTORY_POPULATION_VERSION,'final_races':len(final),
            'cancelled_races':counts['9'],'deleted_races':counts['0'],
            'cancelled_or_deleted_in_history':0,'unconfirmed_races':0}


def session():
    return read_only_session(connection_config(10000),os.environ.get('PGPASSWORD'),gate_timeout_ms=3000)


def extend_lineage(lineage,horses,db_session):
    missing=sorted(set(horses)-set(lineage))
    if not missing:return lineage,{'new_horse_IDs':0,'current_master_corrections_asof':'unverified'}
    if len(missing)>20000:raise safety.MLFailure('input_limit')
    tables={}
    for table in ('jvd_um','jvd_sk','jvd_hn'):
        wanted=missing if table!='jvd_hn' else sorted({v for rows in tables.values() for row in rows
            for field in ancestry.FIELDS[3:] if (v:=stable_id(row.get(field)))})
        rows=[]
        for offset in range(0,len(wanted),1000):
            sql,args=ancestry.query(table,wanted[offset:offset+1000])
            with db_session() as db:rows.extend(db.aggregate('pedigree_forward.ancestry',sql,args,max_rows=3000))
        tables[table]=rows
    new,summary=ancestry.link(set(missing),tables['jvd_um'],tables['jvd_sk'],tables['jvd_hn'])
    return dict(lineage,**new),summary


def extra_sources(directory,day,db_session):
    """New snapshots only; no modification of the frozen 2026 training caches."""
    cutoff=day-timedelta(days=7)
    base_end=date.fromisoformat(c.read(c.RAW_PARENT/'source_manifest.json')['latest_final_date'])
    if cutoff<=base_end:return []
    root=c.mkdir(directory/'history'/str(day));manifest=root/'manifest.json'
    if manifest.exists():
        info=c.read(manifest)
        if (info['cutoff']!=str(cutoff) or info.get('population_version')!=HISTORY_POPULATION_VERSION):
            raise safety.MLFailure('artifact_invalid')
        return [(root/v['file'],v) for v in info['parts']]
    p=dict(c.read(c.RAW_PARENT/'plan.json'),latest_final_date=str(cutoff))
    with db_session() as db:schema=old_data.columns(db)
    info=schema['se'].get('soha_time',{})
    if not info.get('type_is_builtin') or info.get('type_name') not in ('bpchar','varchar','text'):
        raise safety.MLFailure('invalid_schema')
    items=[];began=time.monotonic()
    for first,last in old_data.months(base_end+timedelta(days=1),cutoff):
        if time.monotonic()-began>300:raise safety.MLFailure('query_timeout')
        sql,args,fields=old_data.source_query(schema,first,last,p)
        token=' FROM ONLY public.jvd_ra ra'
        if sql.count(token)!=1:raise safety.MLFailure('artifact_invalid')
        sql=sql.replace(token,',CASE WHEN pg_catalog.octet_length(se.soha_time)<=4 THEN se.soha_time::text ELSE NULL::text END AS time_raw'+token)
        validate_audit_sql(sql)
        path=root/(first.strftime('%Y%m')+'_'+uuid4().hex[:8]+'.csv');rows=0;seen=set()
        with db_session() as db,path.open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=(*RACE_KEY,*fields,'time_raw'));writer.writeheader()
            for batch in db.iter_batches('pedigree_forward.delayed_history',sql,args,max_rows=20000,batch_size=1000,max_seconds=60):
                for row in batch:
                    writer.writerow({k:'' if v is None else v for k,v in row.items()});rows+=1
                    seen.add(tuple(row[k] for k in RACE_KEY))
            stream.flush();os.fsync(stream.fileno())
        # Check scheduled population by keys/status only; no extra result values.
        keys=','.join(safety.identifier(k)+'::text AS '+safety.identifier(k) for k in RACE_KEY)
        check=('SELECT '+keys+',data_kubun::text AS status FROM ONLY public.jvd_ra WHERE kaisai_nen::text=%s '
               "AND kaisai_tsukihi::text BETWEEN %s AND %s AND keibajo_code::text IN ('01','02','03','04','05','06','07','08','09','10') LIMIT 2001")
        with db_session() as db:expected=db.aggregate('pedigree_forward.history_population',check,args[:3],max_rows=2000)
        population=check_population(expected,seen)
        items.append({'file':path.name,'sha256':c.sha(path),'rows':rows,'races':len(seen),'start':str(first),'end':str(last),
                      'population':population})
    c.write(manifest,{'cutoff':str(cutoff),'observed_at':c.clock().isoformat(),'parts':items,
        'purpose':'seven_day_delayed_features_only;no_fit_no_performance_scoring','full_JRA_population':True,
        'population_version':HISTORY_POPULATION_VERSION})
    return [(root/v['file'],v) for v in items]


def extra_days(parts):
    for path,item in parts:
        if c.sha(path)!=item['sha256']:raise safety.MLFailure('artifact_invalid')
        with path.open(encoding='utf-8',newline='') as stream:
            for _,group in groupby(csv.DictReader(stream),key=lambda r:r['kaisai_nen']+r['kaisai_tsukihi']):
                rows=list(group)
                for row in rows:
                    weight=legacy.num(row.get('weight_raw'));row['weight_kg']=weight/10 if weight is not None else None
                yield rows


def build(directory,day,*,horses=(),db_session=None,replay=False):
    cutoff=day-timedelta(days=7);began=time.monotonic()
    lineage=c.read(c.PREP/'lineage.local.json')
    parts=[] if replay else extra_sources(directory,day,db_session)
    needed=set(horses)
    for rows in extra_days(parts):needed.update(r['horse_id'] for r in rows if stable_id(r.get('horse_id')))
    if not replay:
        root=c.mkdir(directory/'history'/str(day))
        file=root/'lineage.local.json'
        if file.exists():lineage=c.read(file);extra={'cached_daily_ID_snapshot':True}
        else:
            lineage,extra=extend_lineage(lineage,needed,db_session)
            c.write(file,lineage)
        lineage_sha=c.sha(file)
    else:extra={'new_horse_IDs':0,'replay_static_master_asof':'unverified'};lineage_sha=c.sha(c.PREP/'lineage.local.json')
    times,resolver,_=supplements.load(c.REFERENCE)
    for rows in extra_days(parts):
        for row in rows:
            key=''.join(row[k] for k in (*RACE_KEY,'horse_number'))
            seconds=jv_time_seconds(row['time_raw']) if row.get('anomaly')=='0' else None
            supplements.put(times,key,{'horse_id':row['horse_id'],'finish_position':row['finish_position'],'time_seconds':seconds})
    ability=AbilityBuilder(times,resolver);blood=PedigreeBuilder(lineage);count=Counter()
    for iterator in (old_data.iter_days(c.RAW_PARENT),extra_days(parts)):
        for rows in iterator:
            d=legacy.legacy.entry_identity(rows[0])[1]
            if d>cutoff:continue
            if time.monotonic()-began>900:raise safety.MLFailure('query_timeout')
            ability.transform_day(rows);blood.transform_day(rows)
            count['history_rows']+=len(rows);count['history_days']+=1
    ability._available(day);blood._available(day)
    info={'cutoff':str(cutoff),'lag_days':7,**count,'lineage_sha256':lineage_sha,
          'extra_parts':[dict(v,file=str(p)) for p,v in parts],'lineage_extension':extra,
          'base_source_manifest_sha256':c.sha(c.RAW_PARENT/'source_manifest.json'),
          'time_supplement_manifest_sha256':c.sha(c.REFERENCE/'source_manifest.json'),
          'prepared_at':c.clock().isoformat(),'current_target_results_read':False,'new_fits':0,'seconds':time.monotonic()-began}
    return {'ability':ability,'blood':blood,'lineage':lineage,'info':info}
