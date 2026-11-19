"""Small read-only supplements; archived normalized times are not decoded twice."""
from datetime import date
import csv
import math
import os
import time
import hashlib
import json

from ..core import RACE_KEY
from ..db_connection import read_only_session
from ..inspect_schema import COLUMNS_SQL
from ..ml import safety
from ..ml.encodings import jv_time_seconds
from ..d39_generation.data import months
from ..win_sim_cli import connection_config
from . import contracts as c, weather

WE_FIELDS=(*weather.SCOPE,'data_kubun','happyo_tsukihi_jifun','henko_shikibetsu',
    'babajotai_code_shiba','babajotai_code_dirt',
    'babajotai_code_shiba_henkomae','babajotai_code_dirt_henkomae')


def session():
    return read_only_session(connection_config(3000),os.environ.get('PGPASSWORD'),gate_timeout_ms=1000)


def inventory(directory):
    with session() as db:
        schema={table:db.aggregate('enrichment.columns',COLUMNS_SQL,('public',table),max_rows=1000)
                for table in ('jvd_we','jvd_se','jvd_ra')}
        counts=db.aggregate('enrichment.we.counts',
            "SELECT kaisai_nen::text AS year,count(*) AS rows FROM ONLY public.jvd_we WHERE kaisai_nen::text BETWEEN '2015' AND '2026' GROUP BY 1 ORDER BY 1",max_rows=12)
    c.write(directory/'schema.json',schema)
    c.write(directory/'WE_inventory.json',{'years':counts,'historical_RA_going_used':False})
    return {'status':'ok','run_id':directory.name,'weather_inventory':counts,
            'WE_column_names':[v['name'] for v in schema['jvd_we']]}


def put(index,key,item):
    item=dict(item)
    rank=item.get('finish_position')
    if isinstance(rank,str) and rank.isascii() and rank.isdigit():item['finish_position']=str(int(rank))
    if key in index and index[key]!=item:
        index[key]={'horse_id':None,'finish_position':None,'time_seconds':None}
    else:
        index[key]=item


def archived_times():
    index={};proof=[]
    for name in ('training_pool','evaluation_history','evaluation'):
        directory=c.POST/name;meta=c.read(directory/'dataset.json');path=directory/'runners.csv'
        if (meta.get('complete') is not True or c.sha(path)!=meta['csv_sha256']
                or meta['mapping'].get('formats',{}).get('finish_time_seconds')!='jv_mss_tenths_4'):
            raise safety.MLFailure('artifact_invalid')
        count=0
        with path.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                key=''.join(row[k] for k in (*RACE_KEY,'horse_number'))
                raw=row['finish_time_seconds'];seconds=float(raw) if raw else None
                if seconds is not None and (not math.isfinite(seconds) or seconds<=0):
                    raise safety.MLFailure('artifact_invalid')
                put(index,key,{'horse_id':row['horse_id'],'finish_position':row['finish_position'],
                    'time_seconds':seconds})
                count+=1
        if count!=meta['rows']:raise safety.MLFailure('artifact_invalid')
        proof.append({'source':str(path),'sha256':meta['csv_sha256'],'rows':count,'values_already_seconds':True})
    return index,proof


def collect(directory):
    schema=c.read(directory/'schema.json');names={t:{r['name'] for r in rows} for t,rows in schema.items()}
    required=(*RACE_KEY,'umaban','ketto_toroku_bango','kakutei_chakujun','ijo_kubun_code','soha_time','data_kubun')
    if not set(required)<=names['jvd_se']:raise safety.MLFailure('invalid_schema')
    target=directory/'source';target.mkdir(exist_ok=True);began=time.monotonic();items=[]
    for first,last in months(date(2025,6,1),date(2026,10,18)):
        path=target/(first.strftime('%Y%m')+'.json')
        if path.exists():
            item=c.read(path);items.append({'file':path.name,'sha256':c.sha(path),'rows':len(item['rows'])});continue
        if time.monotonic()-began>300:raise safety.MLFailure('query_timeout')
        columns=[*RACE_KEY,'umaban','ketto_toroku_bango','kakutei_chakujun','ijo_kubun_code','soha_time']
        sql=('SELECT '+','.join(safety.identifier(k)+'::text AS '+safety.identifier(k) for k in columns)+
            " FROM ONLY public.jvd_se WHERE data_kubun::text='7' AND kaisai_nen::text=%s AND kaisai_tsukihi::text BETWEEN %s AND %s AND keibajo_code::text IN ('01','02','03','04','05','06','07','08','09','10') ORDER BY "+
            ','.join(safety.identifier(k) for k in (*RACE_KEY,'umaban'))+' LIMIT 20001')
        with session() as db:
            rows=[r for batch in db.iter_batches('enrichment.times',sql,
                (str(first.year),first.strftime('%m%d'),last.strftime('%m%d')),max_rows=20000,batch_size=1000,max_seconds=8) for r in batch]
        c.write(path,{'start':str(first),'end':str(last),'rows':rows,'decoder':'JV_MSSd_4'})
        items.append({'file':path.name,'sha256':c.sha(path),'rows':len(rows)})
        print(__import__('json').dumps({'status':'running','stage':'time_supplement','months':len(items)}),flush=True)
    we_path=directory/'WE.local.json';weather_rows=[]
    if set(WE_FIELDS)<=names['jvd_we']:
        if we_path.exists():weather_rows=c.read(we_path)['rows']
        else:
            sql='SELECT '+','.join(safety.identifier(k)+'::text AS '+safety.identifier(k) for k in WE_FIELDS)+" FROM ONLY public.jvd_we WHERE kaisai_nen::text BETWEEN '2015' AND '2026' AND keibajo_code::text IN ('01','02','03','04','05','06','07','08','09','10') LIMIT 12001"
            with session() as db:weather_rows=db.aggregate('enrichment.weather',sql,max_rows=12000)
            c.write(we_path,{'rows':weather_rows,'basis':'publication_minute_end','historical_PC_receipt_unknown':True})
    elif not we_path.exists():c.write(we_path,{'rows':[],'reason':'WE_fields_unavailable'})
    value={'months':items,'WE_sha256':c.sha(we_path),'WE_rows':len(weather_rows)}
    if not (directory/'source_manifest.json').exists():c.write(directory/'source_manifest.json',value)
    return value


def load(directory):
    index,proof=archived_times();manifest=c.read(directory/'source_manifest.json')
    for item in manifest['months']:
        path=directory/'source'/item['file']
        if c.sha(path)!=item['sha256']:raise safety.MLFailure('artifact_invalid')
        for row in c.read(path)['rows']:
            seconds=jv_time_seconds(row['soha_time']) if row['ijo_kubun_code']=='0' else None
            key=''.join(row[k] for k in (*RACE_KEY,'umaban'))
            put(index,key,{'horse_id':row['ketto_toroku_bango'],
                'finish_position':row['kakutei_chakujun'],'time_seconds':seconds})
    if c.sha(directory/'WE.local.json')!=manifest['WE_sha256']:raise safety.MLFailure('artifact_invalid')
    return index,weather.Resolver(c.read(directory/'WE.local.json')['rows']),proof


def capture_going(directory):
    """Preserve actual PC-observed pre-cutoff states for later approved training.

    This does not enable an untrained column in the already frozen candidate.
    No label, finish time, price, or payout column is selected.
    """
    schema={r['name']:r for r in c.read(directory/'schema.json')['jvd_ra']}
    fields=(*RACE_KEY,'data_kubun','track_code','hasso_jikoku','babajotai_code_shiba','babajotai_code_dirt')
    if any(k not in schema or schema[k]['type_is_builtin'] is not True for k in fields):
        raise safety.MLFailure('invalid_schema')
    now=c.clock();day=now.date()
    if str(day)!='2026-10-25':raise safety.MLFailure('invalid_date')
    sql='SELECT '+','.join(safety.identifier(k)+'::text AS '+safety.identifier(k) for k in fields)+" FROM ONLY public.jvd_ra WHERE kaisai_nen::text=%s AND kaisai_tsukihi::text=%s AND data_kubun::text IN ('1','2','3','4') AND keibajo_code::text IN ('01','02','03','04','05','06','07','08','09','10') AND hasso_jikoku::text>%s ORDER BY "+','.join(safety.identifier(k) for k in RACE_KEY)+' LIMIT 101'
    with session() as db:rows=db.aggregate('enrichment.live_going',sql,(str(day.year),day.strftime('%m%d'),now.strftime('%H%M')),max_rows=100)
    observed=c.clock();eligible=[]
    for row in rows:
        raw=dict(row,surface=row['track_code'],start_time=row['hasso_jikoku'])
        cutoff=weather.cutoff_for(raw)
        if cutoff is not None and observed<=cutoff:eligible.append(row)
    digest=hashlib.sha256(json.dumps(eligible,sort_keys=True,ensure_ascii=True,separators=(',',':')).encode()).hexdigest()
    root=directory/'going-observed'/str(day);root.mkdir(parents=True,exist_ok=True);path=root/(digest+'.json')
    created=False
    if eligible and not path.exists():
        c.write(path,{'observed_at':observed.isoformat(),'rows':eligible,'source':'live_declared_RA_PC_observed',
            'results_read':False,'all_observed_before_T10':True,'used_by_current_candidate':False,
            'historical_final_RA_backdating':False});created=True
    return {'status':'ok','pre_cutoff_races':len(eligible),'new_snapshot':created,'results_read':False}


def observe(directory,*,once=False):
    c.approved(directory);lock=directory/'going-observer.lock'
    with lock.open('x',encoding='utf-8') as stream:
        stream.write(str(os.getpid()));stream.flush()
        try:
            while str(c.clock().date())=='2026-10-25' and c.clock().hour<18:
                try:value=capture_going(directory)
                except Exception:value={'status':'pending','reason':'live_going_source_unavailable'}
                print(json.dumps(value),flush=True)
                if once:return value
                time.sleep(60)
            return {'status':'ok','reason':'daily_window_finished'}
        finally:
            stream.close();lock.unlink()
